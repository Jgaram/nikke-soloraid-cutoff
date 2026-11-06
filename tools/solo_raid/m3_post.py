"""끝난 시즌 S{n} → 다음 시즌 S{n+1} M3 고정 (docs/solo-raid-method.md 「WP7 정함」 6, 2026-11-07).

  python3 -I tools/solo_raid/m3_post.py all --season S43            # 1 → 2 → 3 → 고정 → 웹 파일, 단계마다 커밋·push
  python3 -I tools/solo_raid/m3_post.py stage1   --season S43       # 1단계에 S43 더하기 → analysis/m3/stage1.json (+ 고정용 사본 stage1_through_S43.json)
  python3 -I tools/solo_raid/m3_post.py backtest --season S43       # 고정 S43 모델로 끝난 S43 되돌려 보기 → analysis/m3/backtest/pred_S43.json
  python3 -I tools/solo_raid/m3_post.py shape    --season S44       # (주의: 여기만 맞힐 시즌) 앞 맞힐 시즌 꼴 기록 shape_hist.json에 S43 더하고 → analysis/m3/S44/shape.json
  python3 -I tools/solo_raid/m3_post.py check-truth --season S42    # 재현 확인: 꼴 기록의 t_true·m_true를 연구 꼴 파일과 견준다

모두 다시 돌려도 안전하다 (이미 된 단계는 건너뛴다. backtest는 끊긴 데서 이어 간다).
공통 옵션: --out-dir DIR (analysis 폴더 대신 임시 폴더 — 리허설) · --work DIR (작업 폴더, 기본 /tmp/m3post_{명령}_{시즌} — 같은 폴더면 절단 사본을 다시 쓴다) · --keep.

1. stage1: 1단계 적합은 연구와 같은 설정 (r7/stage1_M3_D1.json 머리: 라플라스·noncentered·기저 r13·ℓ12·x항·시작점 8개 — m3.py stage1 기본값 그대로).
   그 시즌 모델 파일(model_S{n}_m3.json)의 고정 커밋 code로 tools만 든 worktree에서 `obs_table.py build --full` → `m3.py stage1` (m3_live와 같은 길).
   analysis/m3/stage1.json이 없으면 r7/stage1_M3_D1.json을 복사해 시작한다 (시즌 번호가 S{n} 이상인 칸은 뺀다 — S42 리허설).
   고정용 사본 stage1_through_S{n}.json을 함께 둔다: 다음 시즌 모델 파일이 sha1을 적으므로, 그 시즌 뒤 stage1.json이 자라도 모델 파일 확인이 깨지지 않게.
2. backtest: 고정 S{n} 모델(code·1단계·꼴·prior cache·예측 스위치 그대로)로 끝난 자료를 T 격자 전부에서 잘라 칸 예측. 칸마다 수 분~55분.
   진행 중에는 analysis/m3/backtest/wip_pred_S{n}.json에 쓰고 (glob pred_S*.json에 안 걸림), 끝나면 pred_S{n}.json으로 바꾼다.
3. shape: S{n}의 m_true·t_true를 연구 침수 꼴 도구 그대로(wp13_flood_emp.post — 학습 = 시계 맞는 앞 시즌) 셈해 shape_hist.json에 더하고
   (없으면 연구 shape_k4_D1.json의 S21~S{n-1} 칸에서 시작), m3_shape.py --prior-shape shape_hist.json --extra-seasons S{n} 로 S{n+1} 꼴 칸.
   끝난 시즌 정답표(answer_key.json)와 state가 필요하다 — 분석 갱신 뒤.
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import m3_live as ML  # noqa: E402 — make_code·drop_code·ensure_deps·run·check_frozen 그대로 쓴다

KST = dt.timezone(dt.timedelta(hours=9))
A = os.path.join(ROOT, "data", "solo-raid", "analysis")
R7 = os.path.join(A, "r7")
RESEARCH_STAGE1 = os.path.join(R7, "stage1_M3_D1.json")
RESEARCH_SHAPE = os.path.join(R7, "wp13", "shape_k4_D1.json")
TRAILER = ["Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "Claude-Session: https://claude.ai/code/session_01MQimn5mxd8NRMZScLVCK7x"]
NO_DEADLINE = 1e9


def snum(s):
    return int(str(s).lstrip("S"))


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(p, o, indent=1):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=indent) if indent else json.dump(o, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    os.replace(tmp, p)


def sha1(p):
    return ML.sha1(p)


def rel(p):
    return os.path.relpath(os.path.abspath(p), ROOT)


def say(msg):
    print(f"[{dt.datetime.now(KST).strftime('%H:%M:%S')}] {msg}", flush=True)


def paths(a):
    out = os.path.abspath(a.out_dir)
    return out, os.path.join(out, "m3")


def default_work(a, name):
    return os.path.abspath(a.work) if a.work else os.path.join(tempfile.gettempdir(), f"m3post_{name}_{a.season}")


def model_path(a, season):
    out, _ = paths(a)
    if getattr(a, "model", None):
        return os.path.abspath(a.model)
    p = os.path.join(out, f"model_{season}_m3.json")
    return p if os.path.exists(p) else os.path.join(A, f"model_{season}_m3.json")


def season_dir(season):
    d = os.path.join(ROOT, "data", "solo-raid", season)
    if not os.path.exists(os.path.join(d, "state", "season.json")):
        sys.exit(f"멈춤: {rel(d)}/state/season.json이 없다 — {season} 자료가 이 레포에 없다")
    return d


# ---------------------------------------------------------------- 1. 1단계

def cmd_stage1(a):
    s = a.season
    out, m3d = paths(a)
    target = os.path.join(m3d, "stage1.json")
    snap = os.path.join(m3d, f"stage1_through_{s}.json")
    cur = jload(target) if os.path.exists(target) else None
    if cur and s in cur["seasons"]:
        say(f"stage1 {s}: {rel(target)}에 이미 있음 — 건너뜀")
        _snapshot(target, snap, s)
        return target
    model = ML.jload(model_path(a, s))
    sdir = season_dir(s)
    work = default_work(a, "stage1")
    os.makedirs(work, exist_ok=True)
    wt = None
    try:
        # 시작 파일: 기존 stage1.json, 없으면 연구 파일에서 (S{n} 이상은 뺀다)
        if cur is None:
            base = jload(RESEARCH_STAGE1)
            base["seasons"] = {k: v for k, v in base["seasons"].items() if snum(k) < snum(s)}
            jdump(os.path.join(work, "stage1_work.json"), base, indent=None)
        else:
            cur["seasons"] = {k: v for k, v in cur["seasons"].items() if snum(k) < snum(s)}
            jdump(os.path.join(work, "stage1_work.json"), cur, indent=None)
        need = [f"S{n}" for n in range(19, snum(s))]
        have = jload(os.path.join(work, "stage1_work.json"))["seasons"]
        if [x for x in need if x not in have]:
            sys.exit(f"멈춤: 1단계에 {s} 앞 시즌이 모두 있어야 한다 — 없는 시즌 {[x for x in need if x not in have]}")
        ML.drop_code(work)
        wt = ML.make_code(model, work, s, sdir)
        ML.ensure_deps(wt)
        L = os.path.join(wt, "tools", "solo_raid", "lab7")
        py = [sys.executable, "-u", "-I"]
        logf = os.path.join(work, "stage1.log")
        obs = os.path.join(work, "obs7")
        t0 = time.time()
        say(f"stage1 {s}: 관측 표 (full) — 코드 {model['code'][:8]}")
        ML.run(py + [os.path.join(L, "obs_table.py"), "build", "--out", obs, "--full", "--seasons", f"{s}-{s}"], time.time() + NO_DEADLINE, logf)
        say(f"stage1 {s}: m3.py stage1 (라플라스, 몇 분~수십 분) — 로그 {logf}")
        t1 = time.time()
        ML.run(py + [os.path.join(L, "m3.py"), "stage1", "--obs-dir", os.path.join(obs, "full"), "--seasons", s, "--out", os.path.join(work, "stage1_work.json")],
               time.time() + NO_DEADLINE, logf)
        new = jload(os.path.join(work, "stage1_work.json"))
        if s not in new["seasons"]:
            sys.exit("멈춤: m3.py stage1이 시즌 칸을 못 냈다")
        # 머리: 연구 1단계의 머리를 그대로 두고, 이 도구가 더한 시즌만 added에 적는다
        head = jload(RESEARCH_STAGE1)["header"] if cur is None else cur["header"]
        if new["header"].get("method") != head.get("method") or new["header"].get("basis") != head.get("basis") or new["header"].get("cfg") != head.get("cfg"):
            sys.exit("멈춤: 이번 1단계 머리(method·basis·cfg)가 기존 파일과 다르다 — 설정이 연구와 같지 않다")
        merged = {"header": dict(head), "seasons": {**have, s: new["seasons"][s]}}
        merged["header"].setdefault("added", {})[s] = {"at_kst": dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M"), "code": model["code"],
                                                       "secs": round(time.time() - t1, 1), "tool": "tools/solo_raid/m3_post.py stage1"}
        jdump(target, merged, indent=None)
        e = new["seasons"][s]
        say(f"stage1 {s}: 끝 — 적합 {e.get('secs')}초 (전체 {time.time() - t0:.0f}초) map_ok {e['map']['ok']} lnC3 {e['lnC_q10_50_90']['3'][1]:.4f} → {rel(target)}")
        _snapshot(target, snap, s)
        return target
    finally:
        if wt is not None and not a.keep:
            ML.drop_code(work)


def _snapshot(target, snap, s):
    d = jload(target)
    if s not in d["seasons"]:
        return
    if os.path.exists(snap):
        return
    shutil.copyfile(target, snap)
    say(f"고정용 사본 {rel(snap)} (S19~{s})")


# ---------------------------------------------------------------- 2. 되돌려 보기

def cmd_backtest(a):
    s = a.season
    out, m3d = paths(a)
    final = os.path.join(m3d, "backtest", f"pred_{s}.json")
    wip = os.path.join(m3d, "backtest", f"wip_pred_{s}.json")
    mp = model_path(a, s)
    if not os.path.exists(mp):
        sys.exit(f"멈춤: 고정 모델 {rel(mp)}가 없다")
    model = ML.jload(mp)
    if model.get("season") != s:
        sys.exit(f"멈춤: 모델은 {model.get('season')}용")
    sdir = season_dir(s)
    state = os.path.join(sdir, "state")
    season, c0, H = ML.season_info(state)
    sch = model.get("schedule") or {}
    c0h = c0.hour + c0.minute / 60
    if float(sch.get("H")) != H or float(sch.get("c0h")) != c0h:
        sys.exit(f"멈춤: 시즌 일정 (H {H:g}·시작 {c0h:g}시)이 모델 가정 (H {sch.get('H')}·시작 {sch.get('c0h')}시)과 다르다 — 이 시즌은 모델의 꼴 가정 밖")
    grid = [float(t) for t in range(6, 121, 6) if t <= H]
    for t in (H - 12, H - 6, H):
        if t not in grid:
            grid.append(float(t))
    grid = sorted(grid)
    Ts = grid if not a.T else [float(t) for t in a.T.split(",")]
    bad = [t for t in Ts if t not in grid]
    if bad:
        sys.exit(f"멈춤: T {bad}가 이 시즌 T 격자에 없다 ({grid})")
    if os.path.exists(final) and not a.T:
        got = {c["T"] for c in jload(final)["cells"]}
        if set(Ts) <= got:
            say(f"backtest {s}: {rel(final)}에 칸이 이미 다 있다 — 건너뜀")
            return final
    ML.check_frozen(model)
    work = default_work(a, "backtest")
    os.makedirs(work, exist_ok=True)
    ML.drop_code(work)
    wt = ML.make_code(model, work, s, sdir)
    try:
        ML.ensure_deps(wt)
        L = os.path.join(wt, "tools", "solo_raid", "lab7")
        py = [sys.executable, "-u", "-I"]
        logf = os.path.join(work, "backtest.log")
        w = os.path.join(work, "w")
        tl = ",".join(f"{t:g}" for t in Ts)
        t0 = time.time()
        say(f"backtest {s}: 절단 사본·관측 표 T {tl} — 코드 {model['code'][:8]}")
        ML.run(py + [os.path.join(L, "trunc.py"), "--work", w, "--jobs", str(a.jobs), "--seasons", f"{s}-{s}", "--T", tl], time.time() + NO_DEADLINE, logf)
        ML.run(py + [os.path.join(L, "obs_table.py"), "build", "--work", w, "--out", os.path.join(w, "obs7"), "--seasons", f"{s}-{s}", "--jobs", str(a.jobs)],
               time.time() + NO_DEADLINE, logf)
        pc = os.path.join(work, "pc")
        if not os.path.exists(pc):
            shutil.copytree(ML.ab(model["prior"]["cache"]), pc)
        os.makedirs(os.path.dirname(wip), exist_ok=True)
        # 이어 돌기: 최종 파일이 일부만 있으면 wip로 옮겨 이어서
        if os.path.exists(final) and not os.path.exists(wip):
            shutil.copyfile(final, wip)
        say(f"backtest {s}: m3.py predict {len(Ts)}칸 (칸마다 수 분~55분, 끊겨도 같은 명령으로 이어 감) — 로그 {logf}")
        n_ch = max(int(a.chunk or 0), 0) or len(Ts)
        chunks = [Ts[i:i + n_ch] for i in range(0, len(Ts), n_ch)]
        for k, ch in enumerate(chunks, 1):
            ML.run(py + [os.path.join(L, "m3.py"), "predict", "--stage1", ML.ab(model["stage1"]["path"]), "--obs-dir", os.path.join(w, "obs7"),
                         "--seasons", s, "--T", ",".join(f"{t:g}" for t in ch), "--shape", ML.ab(model["shape"]["path"]), "--prior-cache", pc, "--out", wip]
                   + list(model["predict_args"]), time.time() + NO_DEADLINE, logf)
            if len(chunks) > 1:
                say(f"backtest {s}: {k}/{len(chunks)} 묶음 끝 (T {ch[0]:g}~{ch[-1]:g}) — {(time.time() - t0) / 60:.0f}분 지남")
                cb = getattr(a, "on_chunk", None)
                if cb and k < len(chunks):
                    cb(wip, k, len(chunks))
        res = jload(wip)
        cells = [c for c in res["cells"] if c["season"] == s]
        got = {c["T"] for c in cells}
        if not set(Ts) <= got:
            sys.exit(f"멈춤: 칸이 모자란다 {sorted(set(Ts) - got)} — 같은 명령으로 다시 돌린다 ({rel(wip)}에서 이어 감)")
        match = res.get("priors", {}).get(s) == jload(ML.ab(model["prior"]["path"]))
        res["post"] = {"about": "고정 모델로 끝난 시즌을 T 격자에서 잘라 낸 되돌려 보기 칸 (tools/solo_raid/m3_post.py backtest) — 다음 시즌 보정 풀·웹 되돌려 보기",
                       "season": s, "model": {"file": rel(mp), "sha1": sha1(mp), "code": model["code"]},
                       "prior_equals_frozen": match, "made_kst": dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
                       "secs": round(time.time() - t0, 1), "T": sorted(got)}
        if not match:
            say("!! 사전(prior)이 고정 prior.json과 다르다")
        if a.T:   # 일부 T 시험: 최종 파일은 만들지 않고 wip만 둔다 (보정 풀에 일부 칸이 들어가면 안 되므로)
            say(f"backtest {s}: 일부 T 시험 — {rel(wip)}에만 둔다")
            return wip
        jdump(final, res, indent=None)
        os.remove(wip)
        wall = [i.get("wall", {}).get("total") or i["secs"] for i in res["info"] if i["season"] == s]
        say(f"backtest {s}: 끝 — {len(cells)}칸 {len(got)}개 T, 칸 시간 합 {sum(wall) / 60:.0f}분 (최대 {max(wall) / 60:.0f}분), 수렴 {sum(1 for i in res['info'] if i['conv'])}/{len(res['info'])} → {rel(final)}")
        return final
    finally:
        if not a.keep:
            ML.drop_code(work)


# ---------------------------------------------------------------- 3. 꼴

def build_full_obs(work, upto):
    """현재 트리 도구로 S19~S{upto} 끝난 시즌 전체 관측 표 (obs_table.py build --full). 이미 있으면 건너뜀. 폴더 반환 (…/obs7/full)."""
    out = os.path.join(work, "obs7")
    full = os.path.join(out, "full")
    need = [f"S{n}" for n in range(19, upto + 1)]
    if all(os.path.exists(os.path.join(full, f"{x}.jsonl.gz")) for x in need):
        return full
    say(f"관측 표 full S19~S{upto} 만들기 (현재 트리 lab7/obs_table.py)")
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "lab7", "obs_table.py"), "build", "--out", out, "--full", "--seasons", f"S19-S{upto}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"obs_table 실패:\n{r.stderr[-1500:]}")
    return full


def cmd_truth(a):
    """(숨은 명령) 끝난 시즌의 m_true·t_true를 셈해 JSON으로 — 연구 wp13_flood_emp.post와 같은 셈. --extra-seasons/--drop-seasons는 import 전에 읽는다."""
    import m3_extend as XT
    XT.pre_import(add=XT.argv_list(sys.argv, "--extra-seasons"), drop=XT.argv_list(sys.argv, "--drop-seasons"))
    import numpy as np  # noqa: F401
    import wp12_flood_curve as FC
    import wp13_flood_emp as FE
    import wp13_flood_shape as WF
    import wp12_tables as WT
    if XT.argv_list(sys.argv, "--extra-seasons") or XT.argv_list(sys.argv, "--drop-seasons"):
        XT.post_import()
    s = a.season
    tn = snum(s)
    if s not in WF.SIDX:
        sys.exit(f"{s}가 시즌 목록에 없다 (--extra-seasons {s})")
    si = WF.SI[s]
    res = {"season": s, "clock_ok": bool(si.get("clock_ok"))}
    if not si.get("clock_ok"):
        res.update(m_true=None, t_true=None, note="05시 시작 120시간이 아니라 침수 꼴 학습 시즌이 아니다 — t_true 없음 (tau_oos가 건너뜀)")
    else:
        if not WT.truth_table(s):
            sys.exit(f"멈춤: answer_key.json에 {s} 한섭 정답이 없다 — 분석 갱신(정답 표) 뒤에 한다")
        FE.init_worker(a.obs_dir)
        train = [x for x in FC.CLOCK if snum(x) < tn]
        rf = FE.recipe(FE.O_FC, train)
        e = FE.post(tn, rf, rf, train)
        res.update(m_true=e["m_true"], t_true=e["t_true"], train=train)
    jdump(a.json_out, res)
    print(json.dumps({k: res[k] for k in ("m_true", "t_true")}, ensure_ascii=False))


def truth_of(season, obs_dir, work, extra, drop=()):
    j = os.path.join(work, f"truth_{season}.json")
    cmd = [sys.executable, "-I", os.path.abspath(__file__), "_truth", "--season", season, "--obs-dir", obs_dir, "--json-out", j]
    if extra:
        cmd += ["--extra-seasons", ",".join(extra)]
    if drop:
        cmd += ["--drop-seasons", ",".join(drop)]
    say(f"{season} m_true·t_true 셈 (연구 침수 꼴 도구)")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{season} m_true 셈 실패:\n{(r.stdout + r.stderr)[-2500:]}")
    return jload(j)


def hist_init():
    """연구 꼴 파일에서 shape_hist 시작 (S21~S42 칸에서 t_true·m_true만)."""
    r = jload(RESEARCH_SHAPE)
    head = {k: r["header"][k] for k in ("flood_rules", "cut_rules", "data_ver", "format") if k in r["header"]}
    head["about"] = "M3 운영 — 앞 맞힐 시즌들의 꼴 기록 (m3_shape.py --prior-shape 용: tau_oos가 읽는 flood.t_true·m_true만). S21~S42 = 연구 shape_k4_D1.json의 칸 그대로, S43~ = m3_post.py shape가 끝난 시즌 자료로 셈해 더함"
    head["source"] = {"research": {"file": rel(RESEARCH_SHAPE), "sha1": sha1(RESEARCH_SHAPE)[:12]}}
    head["added"] = {}
    tg = {j: {"flood": {"m_true": v["flood"]["m_true"], "t_true": v["flood"]["t_true"], "source": "research"}} for j, v in r["targets"].items()}
    return {"header": head, "targets": tg}


def cmd_shape(a):
    t = a.season                      # 맞힐 시즌
    s = f"S{snum(t) - 1}"             # 끝난 시즌
    out, m3d = paths(a)
    hist_p = os.path.join(m3d, "shape_hist.json")
    shape_p = os.path.join(m3d, t, "shape.json")
    work = default_work(a, "shape")
    os.makedirs(work, exist_ok=True)
    hist = jload(hist_p) if os.path.exists(hist_p) else hist_init()
    extra = [f"S{n}" for n in range(43, snum(t))]       # 연구 시즌 목록(S19~S42) 밖의 끝난 시즌
    full = None
    if s not in hist["targets"]:
        season_dir(s)
        full = build_full_obs(work, snum(s))
        tr = truth_of(s, full, work, extra)
        hist["targets"][s] = {"flood": {"m_true": tr["m_true"], "t_true": tr["t_true"], "source": "emp" if tr["clock_ok"] else "none", "train": tr.get("train")}}
        hist["header"]["added"][s] = {"at_kst": dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M"), "tool": "tools/solo_raid/m3_post.py shape", "note": tr.get("note")}
        jdump(hist_p, hist, indent=None)
        say(f"shape: {s} t_true {tr['t_true']} m_true {tr['m_true']} → {rel(hist_p)}")
    else:
        say(f"shape: {rel(hist_p)}에 {s} 칸이 이미 있음")
    if not os.path.exists(hist_p):
        jdump(hist_p, hist, indent=None)
    if os.path.exists(shape_p):
        say(f"shape: {rel(shape_p)}가 이미 있음 — 건너뜀 (시즌 안에서는 꼴 칸을 안 바꾼다)")
        return shape_p
    full = full or build_full_obs(work, snum(s))
    cmd = [sys.executable, "-I", os.path.join(HERE, "m3_shape.py"), "--target", t, "--obs-dir", full, "--prior-shape", hist_p, "--out", shape_p]
    if extra:
        cmd += ["--extra-seasons", ",".join(extra)]
    say("shape: m3_shape.py " + " ".join(cmd[3:]))
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.exit(f"m3_shape.py 실패:\n{r.stderr[-2500:]}")
    return shape_p


def cmd_check_truth(a):
    """재현 확인: 연구 시즌 s의 m_true·t_true를 이 도구로 셈해 연구 꼴 파일과 견준다. --rebuilt면 시즌 목록에서 s를 뺐다 다시 더해 본다."""
    s = a.season
    work = default_work(a, "check")
    os.makedirs(work, exist_ok=True)
    full = build_full_obs(work, snum(s))
    ref = jload(RESEARCH_SHAPE)["targets"][s]["flood"]
    for label, extra, drop in (("기본", [], []), ("시즌 목록에서 뺐다 다시 더함", [s], [s])) if a.rebuilt else (("기본", [], []),):
        tr = truth_of(s, full, work, extra, drop)
        dm = abs(tr["m_true"]["lnm"] - ref["m_true"]["lnm"])
        dse = abs(tr["m_true"]["se"] - ref["m_true"]["se"])
        dt_ = max(abs(x - y) for x, y in zip(tr["t_true"], ref["t_true"])) if all(x is not None for x in tr["t_true"]) and len(tr["t_true"]) == len(ref["t_true"]) else None
        same = tr["t_true"] == ref["t_true"] and tr["m_true"] == ref["m_true"]
        print(f"[재현 {s} {label}] t_true 차 최대 {dt_} · m_true.lnm 차 {dm:.2e} · se 차 {dse:.2e} · 완전히 같음 {same}")
        print("   내 값:", tr["t_true"], tr["m_true"])
        print("   연구 :", ref["t_true"], ref["m_true"])


# ---------------------------------------------------------------- all

def git(*args, check=True):
    r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)} 실패:\n{r.stdout}{r.stderr}")
    return r


def commit_push(a, files, msg):
    if a.no_commit:
        say(f"(커밋 안 함) {msg}")
        return
    fs = [rel(f) for f in files if os.path.exists(f)]
    for f in files:       # 끝나서 지운 wip 파일 등: 이미 커밋돼 있으면 삭제를 커밋에 넣는다
        if not os.path.exists(f) and git("ls-files", "--error-unmatch", "--", rel(f), check=False).returncode == 0:
            git("rm", "-q", "--cached", "--", rel(f))
            fs.append(rel(f))
    git("add", "--", *[x for x in fs if os.path.exists(os.path.join(ROOT, x))])
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        say(f"커밋할 변경 없음 — {msg}")
        return
    body = msg + "\n\n" + "\n".join(a.trailer or TRAILER) + "\n"
    git("commit", "-q", "-m", body, "--", *fs)
    git("fetch", "-q", "origin", "main")
    m = git("merge", "--no-edit", "origin/main", check=False)
    if m.returncode != 0:
        git("merge", "--abort", check=False)
        sys.exit(f"멈춤: origin/main merge 충돌 — 손으로 푼 뒤 같은 명령을 다시 돌린다\n{m.stdout}{m.stderr}")
    git("push", "-q", "origin", "HEAD:main")
    say(f"커밋·push: {msg} ({git('rev-parse', '--short', 'HEAD').stdout.strip()})")


def cmd_all(a):
    s = a.season
    t = f"S{snum(s) + 1}"
    out, m3d = paths(a)
    if os.path.abspath(a.out_dir) != A:
        say("!! --out-dir가 레포 analysis가 아니다 — 모델 고정·웹 파일 단계는 건너뛴다")
    t0 = time.time()
    # 1
    st = cmd_stage1(a)
    commit_push(a, [st, os.path.join(m3d, f"stage1_through_{s}.json")], f"{s} 끝난 뒤 M3: 1단계에 {s} 더함")
    # 2
    a.on_chunk = lambda p, k, n: commit_push(a, [p], f"{s} 끝난 뒤 M3: {s} 되돌려 보기 진행 ({k}/{n} 묶음, 이어 돌릴 수 있게)")
    bt = cmd_backtest(a)
    commit_push(a, [bt, os.path.join(m3d, "backtest", f"wip_pred_{s}.json")], f"{s} 끝난 뒤 M3: {s} 되돌려 보기 칸 (보정 풀·웹)")
    # 3
    sh = cmd_shape(argparse.Namespace(**{**vars(a), "season": t}))
    commit_push(a, [sh, os.path.join(m3d, "shape_hist.json")], f"{s} 끝난 뒤 M3: 앞 맞힐 시즌 꼴 기록에 {s} 더함 + {t} 꼴 칸")
    if os.path.abspath(a.out_dir) != A:
        return
    # 4 고정
    mp = os.path.join(A, f"model_{t}_m3.json")
    if os.path.exists(mp):
        say(f"{rel(mp)}가 이미 있음 — 고정 건너뜀")
    else:
        snap = os.path.join(m3d, f"stage1_through_{s}.json")
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "m3_freeze.py"), "--season", t, "--stage1", snap], capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.exit(f"m3_freeze.py 실패:\n{r.stdout[-1500:]}{r.stderr[-2500:]}")
    commit_push(a, [mp, os.path.join(A, f"model_{t}.json")] + glob.glob(os.path.join(m3d, t, "*")) + glob.glob(os.path.join(m3d, t, "pc", "*")),
                f"{s} 끝난 뒤 M3: {t} 고정 모델 (model_{t}_m3.json · prior · F 대비)")
    # 5 웹 파일
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "backtest_web_m3.py")], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.exit(f"backtest_web_m3.py 실패:\n{r.stderr[-2000:]}")
    commit_push(a, [os.path.join(A, "backtest_M3.json")], f"{s} 끝난 뒤 M3: 웹 되돌려 보기 파일에 {s} 더함")
    say(f"전체 끝 — {s} → {t} 고정 (전체 {(time.time() - t0) / 3600:.1f}시간). 공개 웹 반영은 docs/publish.md 「언제 복사하나」")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("all", "stage1", "backtest", "shape", "check-truth", "_truth"))
    ap.add_argument("--season", required=True, help="stage1·backtest·all·check-truth: 끝난 시즌 (S43) · shape: 맞힐 시즌 (S44)")
    ap.add_argument("--out-dir", default=A, help="analysis 폴더 (기본 레포. 리허설은 임시 폴더)")
    ap.add_argument("--work", default=None)
    ap.add_argument("--keep", action="store_true", help="작업 폴더의 코드 worktree를 지우지 않는다")
    ap.add_argument("--model", default=None, help="backtest·stage1: 고정 모델 파일 (기본 {out-dir 또는 analysis}/model_{시즌}_m3.json)")
    ap.add_argument("--T", default=None, help="backtest: 쉼표로 T만 (시험용 — 이 경우 pred_S{n}.json은 만들지 않고 wip_pred_S{n}.json에만 둔다)")
    ap.add_argument("--chunk", type=int, default=4, help="backtest: T를 이만큼씩 묶어 돌린다 (all은 묶음마다 wip를 커밋·push해 컨테이너가 바뀌어도 이어 간다). 0 = 한 번에")
    ap.add_argument("--jobs", type=int, default=4, help="backtest: 절단 사본·관측 표 프로세스 수")
    ap.add_argument("--no-commit", action="store_true", help="all: 커밋·push 안 함")
    ap.add_argument("--trailer", action="append", help="all: 커밋 메시지 끝 줄 (여러 번). 기본 이 구현 세션의 두 줄")
    ap.add_argument("--rebuilt", action="store_true", help="check-truth: 시즌 목록에서 뺐다 다시 더한 길도 확인")
    ap.add_argument("--obs-dir", help=argparse.SUPPRESS)
    ap.add_argument("--json-out", help=argparse.SUPPRESS)
    ap.add_argument("--extra-seasons", help=argparse.SUPPRESS)
    ap.add_argument("--drop-seasons", help=argparse.SUPPRESS)
    a = ap.parse_args()
    {"all": cmd_all, "stage1": cmd_stage1, "backtest": cmd_backtest, "shape": cmd_shape, "check-truth": cmd_check_truth, "_truth": cmd_truth}[a.cmd](a)


if __name__ == "__main__":
    main()
