"""솔로레이드 자동 수집(6시간 간격) 한 번의 판정·기록 (모델 0, 요청 0). 2026-10-16 사용자 결정, docs/solo-raid-live.md.

python3 -I tools/solo_raid/live_tick.py plan [--now 'YYYY-MM-DD HH:MM']   이번 실행이 할 일을 JSON 한 줄로
python3 -I tools/solo_raid/live_tick.py skip RUN                          빠진 판독 배치를 다시 돌릴 때 Workflow skip 목록
python3 -I tools/solo_raid/live_tick.py block RUN                         403·429로 멈춰야 하는지 (fetchlog.tsv)
python3 -I tools/solo_raid/live_tick.py record PLAN STATUS [RUN] [NOTE]   DEST/live/ticks.jsonl과 README 「자동 실행」 표에 한 줄

켜진 시즌은 data/solo-raid/live.json의 season이다 (사람이 켜고, 마지막 실행이 null로 비운다).
시각은 모두 KST 'YYYY-MM-DD HH:MM:SS'. 리허설용 환경 변수: LIVE_NOW(--now 대신), LIVE_DATA(data/solo-raid 대신 사본 폴더).

plan의 칸 계산
- 칸 경계 b = c_start + 6시간 × k. 예약 작업은 경계 1분 뒤(05·11·17·23시 01분)에 깨지만, 몇 분 일찍 깨도 같은 칸을 잡도록
  지금 + 5분 이하에서 가장 늦은 경계를 고른다. 경계 전이면 wait_sec만큼 기다린 뒤 받는다.
- T = (b − c_start) / 1시간. 구간은 [지난 성공 끝, b) — 게시 시각 기준이라 경계 뒤에 올라온 글은 다음 칸으로 간다.
  지난 성공 끝은 DEST/state/runs.jsonl(run_window.sh가 DONE 때만 덧붙임)의 가장 늦은 end, 없으면 c_start.
- action
  - run    6 ≤ T ≤ T_end(종료를 덮는 첫 경계, S42는 120): 받기 → 판독 → 예측 T. T_end면 after = set_once(e_end + 1분).
  - final  T > T_end이고 e_end가 지났다: [지난 끝, e_end) 받기 → 판독 → 후처리. after = finish.
  - idle   켜진 시즌 없음 · 첫 칸 전 · 이미 한 칸(predict/T{T}/가 있음) · 종료 뒤 e_end 전 · 이미 끝남.
"""
import datetime as dt
import glob
import json
import math
import os
import re
import sys
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.environ.get("LIVE_DATA") or os.path.join(ROOT, "data", "solo-raid")   # 리허설은 레포 밖 사본을 가리킨다
LIVE = os.path.join(D, "live.json")
F = "%Y-%m-%d %H:%M:%S"
KST = dt.timezone(dt.timedelta(hours=9))
SLOT = dt.timedelta(hours=6)
TOL = dt.timedelta(minutes=5)
ONCE = dt.timedelta(minutes=1)   # 종료 후 한 번짜리 실행은 e_end 1분 뒤 (정각은 예약 작업이 붐빔)
BATCH = 19   # raid-window.js 배치당 시트 수 (args.batch 기본)
IMG_HOST = re.compile(r"^dcimg\d*\.dcinside\.(com|co\.kr)$")
IMG403_MAX = int(os.environ.get("IMG403_MAX") or 20)


def p(s):
    return dt.datetime.strptime(s[:19], F)


def fmt(t):
    return t.strftime(F)


def now_kst(arg=None):
    s = arg or os.environ.get("LIVE_NOW")
    if s:
        for f in (F, "%Y-%m-%d %H:%M"):
            try:
                return dt.datetime.strptime(s, f)
            except ValueError:
                pass
        sys.exit(f"--now를 읽지 못했다: {s}")
    return dt.datetime.now(KST).replace(tzinfo=None, microsecond=0)


def last_end(state, c0):
    ends = []
    f = os.path.join(state, "runs.jsonl")
    if os.path.exists(f):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                ends.append(p(json.loads(line)["end"]))
    return max(ends + [c0])


def plan(now):
    out = {"action": "idle", "now": fmt(now)}
    if not os.path.exists(LIVE):
        return dict(out, reason="data/solo-raid/live.json 없음 (켜진 시즌 없음)")
    live = json.load(open(LIVE, encoding="utf-8"))
    name = live.get("season")
    if not name:
        return dict(out, reason="켜진 시즌 없음 (live.json season null)")
    dest = os.path.join(D, name)
    state = os.path.join(dest, "state")
    s = json.load(open(os.path.join(state, "season.json"), encoding="utf-8"))
    if not s.get("c_start"):
        return dict(out, action="stop", season=name, reason="season.json에 c_start가 없다")
    c0, end, e_end = p(s["c_start"]), p(s["end"]), p(s["e_end"])
    k = math.floor((now + TOL - c0) / SLOT)
    b = c0 + SLOT * k
    T = int(round((b - c0).total_seconds() / 3600))
    k_end = math.ceil((end + dt.timedelta(seconds=1) - c0) / SLOT)
    T_end = int(k_end * SLOT.total_seconds() // 3600)
    le = last_end(state, c0)
    out.update(season=name, dest=dest, trigger=live.get("trigger"),
               c_start=s["c_start"], end=s["end"], e_end=s["e_end"], T=T, T_end=T_end, slot=fmt(b), last_end=fmt(le))
    if k < 1:
        return dict(out, reason=f"첫 칸(c_start + 6시간 = {fmt(c0 + SLOT)}) 전")
    if T <= T_end:
        if glob.glob(os.path.join(dest, "predict", f"T{T}", f"predict_T{T}.json")):
            return dict(out, reason=f"T{T} 예측이 이미 있다")
        wend = min(b, end + dt.timedelta(seconds=1))
        return dict(out, action="run", start=fmt(le), end_w=fmt(wend), collect=le < wend,
                    wait_sec=max(0, int((b - now).total_seconds())),
                    after=f"set_once {(e_end + ONCE).isoformat()}+09:00" if T == T_end else None,
                    reason=f"T{T} 받기 → 예측")
    if os.path.exists(os.path.join(dest, "live", "final_done.json")):
        return dict(out, reason="종료 후 수집까지 끝났다", after="finish")
    if now + TOL < e_end:
        return dict(out, reason=f"종료 뒤 — e_end({s['e_end']})에 마지막 수집", after=f"set_once {(e_end + ONCE).isoformat()}+09:00")
    return dict(out, action="final", start=fmt(le), end_w=fmt(e_end), collect=le < e_end,
                wait_sec=max(0, int((e_end - now).total_seconds())), after="finish",
                reason="종료 후 인증 마지막 수집 → 후처리")


def skip(run):
    """merge_obs.py의 missing.txt를 보고, 다시 돌리지 않아도 되는 결과 파일(run 기준 상대 경로) 목록."""
    miss = open(os.path.join(run, "missing.txt")).read().split("\n") if os.path.exists(os.path.join(run, "missing.txt")) else []
    redo = set()
    for m in miss:
        if m.startswith("sheet "):
            n = int(re.sub(r"\D", "", m.split(" ", 1)[1]) or 0)
            if n:
                redo.add(f"reads/reads_{(n - 1) // BATCH + 1:02d}.jsonl")
        elif m.startswith("text "):
            redo.update(os.path.relpath(f, run) for f in glob.glob(os.path.join(run, "text", "out_*.jsonl")))
        elif m.startswith("ctx "):
            redo.update(os.path.relpath(f, run) for f in glob.glob(os.path.join(run, "ctx", "out_*.jsonl")))
    have = [os.path.relpath(f, run) for k in ("reads", "text", "ctx")
            for f in glob.glob(os.path.join(run, k, "*.jsonl")) if os.path.getsize(f) > 0 and not os.path.basename(f).startswith("in_")]
    return sorted(set(have) - redo)


def block(run):
    """fetchlog.tsv에서 멈춤 조건 (docs/solo-raid-backfill.md 「멈출 때」): 429, 목록·본문 403, 사진 서버 403 20번."""
    f = os.path.join(run, "fetchlog.tsv")
    hits, img403 = [], 0
    if os.path.exists(f):
        for line in open(f, encoding="utf-8"):
            r = line.rstrip("\n").split("\t")
            if len(r) < 5:
                continue
            code, host = r[1], urlparse(r[4]).hostname or ""
            if code == "429" or (code == "403" and not IMG_HOST.match(host)):
                hits.append(f"{code} {host} {r[0]}")
            elif code == "403":
                img403 += 1
    if img403 >= IMG403_MAX:
        hits.append(f"403 사진 서버 {img403}번")
    return {"blocked": bool(hits), "hits": hits[:5]}


def record(plan_f, status, run=None, note=""):
    pl = json.load(open(plan_f, encoding="utf-8"))
    dest = os.path.join(ROOT, pl["dest"])
    row = {"T": pl.get("T"), "action": pl["action"], "slot": pl.get("slot"), "window": [pl.get("start"), pl.get("end_w")],
           "status": status, "done": fmt(dt.datetime.now(KST).replace(tzinfo=None, microsecond=0)),
           "session": os.environ.get("CLAUDE_CODE_SESSION_ID"), "note": note}
    if run and os.path.exists(os.path.join(run, "DONE")):
        d = json.loads(open(os.path.join(run, "DONE")).read().strip().splitlines()[-1])
        row.update(list=d.get("list"), new_posts=d.get("new_posts"), sheets=d.get("sheets"), requests=d.get("requests"),
                   req_fail=d.get("req_fail"), sec=d.get("sec"))
    if run and os.path.exists(os.path.join(run, "merge.json")):
        m = json.load(open(os.path.join(run, "merge.json")))
        row["merge"] = {k: m[k] for k in ("obs_new", "obs_in", "obs_total", "claims_run", "missing") if k in m} or None
    pj = os.path.join(dest, "predict", f"T{pl.get('T')}", f"predict_T{pl.get('T')}.json")
    if pl["action"] == "run" and os.path.exists(pj):
        o = json.load(open(pj, encoding="utf-8"))
        row["grade"] = {k: ((v or {}).get("warning") or {}).get("level") for k, v in (o.get("servers") or {}).items()} or None
    os.makedirs(os.path.join(dest, "live"), exist_ok=True)
    with open(os.path.join(dest, "live", "ticks.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rd = os.path.join(dest, "README.md")
    txt = open(rd, encoding="utf-8").read() if os.path.exists(rd) else f"# {pl['season']}\n"
    head = "## 자동 실행"
    if head not in txt:
        txt = txt.rstrip("\n") + (f"\n\n{head}\n\n6시간마다 예약 작업이 남긴다 (`tools/solo_raid/live_tick.py record`, 원본은 `live/ticks.jsonl`).\n\n"
                                  "| T | 구간 (KST) | 상태 | 끝난 시각 | 새 글·시트·요청(실패) | 등급 |\n|---|---|---|---|---|---|\n")
    w = row["window"]
    win = f"{(w[0] or '')[5:16]} ~ {(w[1] or '')[5:16]}" if w[0] else "—"
    nums = f"{row.get('new_posts', '—')}·{row.get('sheets', '—')}·{row.get('requests', '—')}({row.get('req_fail', '—')})"
    grade = " ".join(f"{k}{v}" for k, v in (row.get("grade") or {}).items() if v) or "—"
    label = f"T{row['T']}" if pl["action"] == "run" else "종료 후"
    line = f"| {label} | {win} | {status}{(' — ' + note) if note else ''} | {row['done'][5:16]} | {nums} | {grade} |\n"
    open(rd, "w", encoding="utf-8").write(txt.rstrip("\n") + "\n" + line)   # 「자동 실행」 표는 README 맨 끝에 둔다
    return row


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("plan", "skip", "block", "record"):
        sys.exit(__doc__)
    if a[0] == "plan":
        now = now_kst(a[a.index("--now") + 1] if "--now" in a else None)
        print(json.dumps(plan(now), ensure_ascii=False))
    elif a[0] == "skip":
        print(json.dumps(skip(a[1]), ensure_ascii=False))
    elif a[0] == "block":
        r = block(a[1])
        print(json.dumps(r, ensure_ascii=False))
        sys.exit(3 if r["blocked"] else 0)
    else:
        print(json.dumps(record(a[1], a[2], a[3] if len(a) > 3 and a[3] else None, a[4] if len(a) > 4 else ""), ensure_ascii=False))


if __name__ == "__main__":
    main()
