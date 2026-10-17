"""솔로레이드 시즌 등록부 — 시즌마다 준비 상태와 몫(개발/시험)을 한 곳에서 본다 (할 일 3.A1·3.A5, 2026-10-16, 모델 0, 요청 0).

python3 -I tools/solo_raid/registry.py [--write]      # 표준 출력에 표, --write면 data/solo-raid/registry.json도 쓴다

읽는 것 (쓰지 않는다): data/solo-raid/seasons.json(일정·형식, 컷은 「있음/없음」만) · split.json(몫) · PROGRESS.md·S{n}/collect_done.json(수집 상태) ·
S{n}/discard.json · S{n}/state/(season.json·core_summary.json·core.jsonl·truth_points.jsonl·post_all_stamp.json).
**정답 값은 읽어도 내보내지 않는다** — 정답 점은 서버·등급별 개수만 센다 (시험 시즌도 같다). 그래서 split 감시를 켜지 않는다.

학습에 넣는 최소 조건 (MIN, 2026-10-16 실무 세션이 시즌별 개수를 보기 전에 정함 — 대 보니 S39가 「집계표만」(한섭 정답 점 6))
- 공통: 수집이 끝났고(collect_status 「완료」), state가 있고 후처리 도장이 fresh, c_start 확인됨(seasons.json check 없음).

수집 상태 (collect_status, 2026-10-16): S{n}/collect_done.json이 있으면 「완료」, 없으면 PROGRESS.md의 그 시즌 칸, 거기에도 없으면
state가 있을 때 「받는 중」. 끝나지 않은 시즌은 다른 세션이 쓰는 중이라 analyze_all이 후처리를 다시 돌리지 않고 학습·채점에서 뺀다.
collect_done.json은 수집(지난 시즌)·실시간 세션이 마지막 합치기와 후처리 뒤에 쓴다 ({"done": KST 시각, "by": 세션}).
- 「전체」(full): 한섭 정답 점 A+A2 ≥ 8 이고 한섭 라벨 종료 전 기본 집합 사람 ≥ 5 → % 변환·곡선 모양·시즌 간 퍼짐 모두에 쓴다.
- 「모양만」(shape_only): 한섭 정답 점 A+A2 ≥ 8 이지만 라벨 종료 전 사람 < 5 → 곡선 모양에만 쓴다 (% 변환은 라벨 미상 줄에 기대게 되어 뺀다).
  또는 1~2일차 한섭 칸(정답 곡선에 붙은 사람·일차 칸, fit_prior.load_units) < 5 → % 변환 수준을 잴 수 없어 모양에만 (2026-10-17 분석 4.1 H4 —
  자료 개수만 보고 정함. 해당 S19 하나: 1칸, 다른 시즌 6~63칸. 한 시즌 prior가 망가지고 여러 시즌 prior의 τ_P를 0.13 → 0.29로 부풀렸다).
- 「집계표만」(table_only): 정답 점이 모자라지만 집계표(kr_cut_3·kr_cut_10)가 있다 → 10%/3% 비율(모양의 한 점)에만.
- 「안 씀」(none): 나머지. 등록부에 이유를 남긴다.
몫이 시험인 시즌은 학습·검증에 넣지 않는다 (조건 판정은 하지만 쓰지 않는다).
"""
import argparse
import collections
import datetime as dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import split as SP  # noqa: E402
import post_stamp  # noqa: E402
import discard  # noqa: E402

DATA = os.path.join(ROOT, "data", "solo-raid")
OUT = os.path.join(DATA, "registry.json")
FMT = "%Y-%m-%d %H:%M:%S"
MIN_TRUTH_KR = 8
MIN_EARLY_UNITS_KR = 5     # 1~2일차 한섭 칸 (2026-10-17, 분석 4.1 H4)
MIN_LABEL_PERSONS = 5


def jl(p):
    with SP._real_open(p, encoding="utf-8") as f:     # 감시가 켜진 프로세스에서 불려도 개수만 세므로 그대로 연다
        return [json.loads(l) for l in f if l.strip()]


def progress_status():
    p = os.path.join(DATA, "PROGRESS.md")
    out = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            m = re.match(r"\|\s*(S\d+)\s*\|[^|]*\|\s*([^|]*?)\s*\|", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def collect_status(name, prog=None):
    if os.path.exists(os.path.join(DATA, name, "collect_done.json")):
        return "완료"
    prog = progress_status() if prog is None else prog
    if prog.get(name):
        return prog[name]
    return "받는 중" if os.path.isdir(os.path.join(DATA, name, "state")) else "없음"


def git_time(path):
    try:
        r = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%cd", "--date=format-local:%Y-%m-%d %H:%M", "--", path],
                           capture_output=True, text=True, env=dict(os.environ, TZ="Asia/Seoul"), timeout=30)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def fmt_kind(s):
    """시즌 형식 (3.A5): 챌린지 시작 규칙과 길이."""
    c0 = dt.datetime.strptime(s["c_start"], FMT)
    st = dt.datetime.strptime(s["start"], FMT)
    end = dt.datetime.strptime(s["end"], FMT)
    H = round((end - c0).total_seconds() / 3600)
    if (c0.date() - st.date()).days == 2 and c0.hour == 5 and c0.minute == 0:
        kind = "3일차 05:00"
    elif c0 == st:
        kind = "시작부터 (재오픈)"
    else:
        kind = "그 밖 (재개분 등)"
    notes = []
    if s.get("check"):
        notes.append(f"확인 필요: {s['check']}")
    if H != 120:
        notes.append(f"길이 {H}시간 (120 아님)")
    if c0.hour != 5 or c0.minute != 0:
        notes.append(f"시작 {c0.strftime('%H:%M')} (05:00 아님)")
    try:
        if int(str(s.get("name", ""))[1:]) >= 41:
            # 표시만 — 모수로 넣지 않는다 (사용자 관점 2026-10-17, 분석 3.5e). S42·S43이 끝나면 m_s가 S19~S40 범위 안인지 본다
            notes.append("모의전 사전 개방 (S41부터 — 개발자 노트 2026-08-28 「첫날부터 도전 난이도 모의전」, 표시만)")
    except ValueError:
        pass
    return {"kind": kind, "H": H, "c_start": s["c_start"], "c_hour": c0.strftime("%H:%M"), "notes": notes}


def state_info(st):
    info = {"state": os.path.relpath(st, ROOT)}
    for f in ("obs.jsonl", "core.jsonl", "truth_points.jsonl", "truth.json"):
        info["has_" + f.split(".")[0]] = os.path.exists(os.path.join(st, f))
    s, why = post_stamp.check(st) if info["has_obs"] else ("never", "obs 없음")
    info["post_all"] = s
    info["post_all_why"] = why
    sp = os.path.join(st, post_stamp.STAMP)
    if os.path.exists(sp):
        stamp = json.load(open(sp, encoding="utf-8"))
        info["stamp_digest"] = stamp["digest"]
        info["input_digest"] = stamp["input_digest"]
    info["post_all_time"] = git_time(os.path.join(st, "core.jsonl")) if info["has_core"] else None
    fx = discard.items(st) if os.path.exists(os.path.join(st, "season.json")) else {}
    info["discard_posts"] = sum(1 for it in fx.values() if it["scope"] == "post" and it.get("action", "discard") == "discard")
    info["fix_items"] = len(fx)   # 버릴 것·고칠 것 항목 전부 (S{n}/discard.json)
    if info["has_core"]:
        cs = json.load(open(os.path.join(st, "core_summary.json"), encoding="utf-8"))
        info["use"] = cs.get("use")
        info["use_indep"] = cs.get("use_indep")
        info["use_persons"] = cs.get("use_persons")
        info["use_by_server"] = {k: v for k, v in cs.get("use_by_server", {}).items() if k != "None"}
        info["use_server_unknown"] = cs.get("use_by_server", {}).get("None", 0)
        lab = collections.defaultdict(set)
        for r in jl(os.path.join(st, "core.jsonl")):
            if r.get("use") and r.get("server") and r.get("day_lo") != "post" and r.get("person") is not None:
                lab[r["server"]].add(r["person"])
        info["label_persons_pre_end"] = {k: len(v) for k, v in sorted(lab.items())}
    if info["has_truth_points"]:
        c = collections.Counter()
        for r in jl(os.path.join(st, "truth_points.jsonl")):
            c[f"{r.get('server') or '?'}/{r.get('grade')}"] += 1      # 개수만 (값은 내보내지 않는다)
        info["truth_points"] = dict(sorted(c.items()))
        info["early_units_kr"] = early_units_kr(st)
    return info


def early_units_kr(st):
    """1~2일차 한섭 칸 수 — fit_prior가 % 변환을 맞추는 사람·일차 칸(정답 곡선 A+A2, 값 칸 + 구간 칸)과 같은 셈 (개수만)."""
    import fit_prior as FP
    _, _, _, units = FP.load_units(st, ("A", "A2"), "all", None)
    return sum(1 for u in units if u["server"] == "KR" and u["day"] in (1, 2))


def tier(row):
    st = row.get("data") or {}
    reasons = []
    if row["collect"] != "완료":
        return "none", [f"수집이 끝나지 않음 ({row['collect']})"]
    if not st.get("has_core"):
        return "none", ["후처리 결과 없음 (수집 중이거나 안 함)"]
    if st.get("post_all") != "fresh":
        reasons.append(f"후처리가 최신이 아님 ({st.get('post_all')})")
    if row["format"]["notes"] and any(n.startswith("확인 필요") for n in row["format"]["notes"]):
        reasons.append("c_start 확인 필요")
    if reasons:
        return "none", reasons
    tp = st.get("truth_points", {})
    kr = tp.get("KR/A", 0) + tp.get("KR/A2", 0)
    lp = st.get("label_persons_pre_end", {}).get("KR", 0)
    eu = st.get("early_units_kr", 0)
    if kr >= MIN_TRUTH_KR and lp >= MIN_LABEL_PERSONS and eu >= MIN_EARLY_UNITS_KR:
        return "full", [f"한섭 정답 점 {kr} · 라벨 종료 전 사람 {lp} · 1~2일차 한섭 칸 {eu}"]
    if kr >= MIN_TRUTH_KR:
        why = []
        if lp < MIN_LABEL_PERSONS:
            why.append(f"라벨 종료 전 한섭 사람 {lp} < {MIN_LABEL_PERSONS}")
        if eu < MIN_EARLY_UNITS_KR:
            why.append(f"1~2일차 한섭 칸 {eu} < {MIN_EARLY_UNITS_KR} (% 변환 수준을 잴 수 없음)")
        return "shape_only", why
    if row["table"]["kr_cut_3"] and row["table"]["kr_cut_10"]:
        return "table_only", [f"한섭 정답 점 {kr} < {MIN_TRUTH_KR}, 집계표 있음"]
    return "none", [f"한섭 정답 점 {kr} < {MIN_TRUTH_KR}, 집계표 없음"]


def build():
    sp = SP.load_split()
    with SP._real_open(SP.SEASONS, encoding="utf-8") as f:
        seasons = json.load(f)       # 컷 값은 있음/없음만 쓴다
    cut_rows = SP.load_server_cutoffs(final=True)       # 있음/없음만 쓴다
    cut_seasons = collections.Counter(int(r["season"]) for r in cut_rows)
    prog = progress_status()
    names = {s["name"] for s in seasons}
    for d in sorted(os.listdir(DATA)):          # seasons.json에 없는 시즌 (S41 등)은 state/season.json으로
        sj = os.path.join(DATA, d, "state", "season.json")
        if re.fullmatch(r"S\d+", d) and d not in names and os.path.exists(sj):
            seasons.append(dict(json.load(SP._real_open(sj, encoding="utf-8")), registry_from="state/season.json"))
    rows = []
    for s in sorted(seasons, key=lambda s: SP.snum(s["name"])):
        n = SP.snum(s["name"])
        row = {"season": s["name"], "n": n, "boss": s.get("boss"), "role": SP.role(n, sp),
               "collect": collect_status(s["name"], prog),
               "format": fmt_kind(s),
               "table": {"kr_cut_3": s.get("kr_cut_3") is not None, "kr_cut_10": s.get("kr_cut_10") is not None,
                         "server_cutoffs_rows": cut_seasons.get(n, 0),
                         "truth_table_json": os.path.exists(os.path.join(DATA, s["name"], "state", "truth_table.json"))}}
        st = os.path.join(DATA, s["name"], "state")
        row["data"] = state_info(st) if os.path.isdir(st) else None
        row["tier"], row["tier_why"] = tier(row)
        row["train"] = row["role"] == "dev" and row["tier"] != "none"
        rows.append(row)
    return {"about": "솔로레이드 시즌 등록부 (tools/solo_raid/registry.py --write가 만든다, 손으로 고치지 않는다). 정답 값은 넣지 않는다 (개수·있음/없음만)",
            "split_version": sp["version"], "min": {"truth_kr": MIN_TRUTH_KR, "label_persons": MIN_LABEL_PERSONS, "early_units_kr": MIN_EARLY_UNITS_KR},
            "seasons": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    reg = build()
    print(f"{'시즌':5} {'몫':4} {'수집':6} {'형식':16} {'H':>4} {'후처리':12} {'use':>5} {'한섭라벨':>6} {'한섭정답':>6} {'표':3} {'학습':10}")
    for r in reg["seasons"]:
        d = r["data"] or {}
        tp = d.get("truth_points", {})
        kr = tp.get("KR/A", 0) + tp.get("KR/A2", 0) if d else ""
        print(f"{r['season']:5} {r['role']:4} {r['collect'][:6]:6} {r['format']['kind']:16} {r['format']['H']:>4} "
              f"{(d.get('post_all') or '-'):12} {str(d.get('use', '-')):>5} {str(d.get('label_persons_pre_end', {}).get('KR', '-')):>6} "
              f"{str(kr if d else '-'):>6} {'3·10' if r['table']['kr_cut_3'] and r['table']['kr_cut_10'] else '3' if r['table']['kr_cut_3'] else '-':3} "
              f"{r['tier'] + ('' if r['train'] or r['role'] == 'test' else ' (안 씀)'):10}")
        for note in r["format"]["notes"]:
            print(f"      형식: {note}")
    if a.write:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("→", os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
