"""할 일 4.3 마무리 (2026-10-18) — 사전 고정 규칙대로 모은다 (docs/solo-raid-analysis.md 「4.3 마무리 — 사전 고정」, 커밋 9d52d30).

  python3 tools/solo_raid/lab4/e_43b.py a RES_A --out OUT      # A 꼴 정하기: RES_A/s1·s2·s3의 base·m4u·m4l (뽑기 2000) → F
  python3 tools/solo_raid/lab4/e_43b.py b RES_B --out OUT      # B M5: RES_B/{변형}.json (F의 scheme·덧붙임, 뽑기 2000·시드 1) → 무리마다 고르기·③ 등급
  python3 tools/solo_raid/lab4/e_43b.py c RES_C --out OUT      # C 10%: diag43 후보 22개 → P10, RES_C/p10.json·p10n.json (P10과 P10+N10, 뽑기 600·시드 1)
OUT(json)은 단계마다 그 단계 열쇠만 고쳐 쓴다 (b는 a의 F, c의 F 위 N10 보고는 b의 결과를 읽는다). 마지막에 data/solo-raid/analysis/diag43b.json으로 옮긴다.
M1은 4.3 e_43.m1 그대로, 요약은 xval.summarize_v2 (criteria_v2).
"""
import argparse
import collections
import itertools
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import xval as XV  # noqa: E402
import e_43 as E43  # noqa: E402

AN = os.path.join(ROOT, "data", "solo-raid", "analysis")
MARGIN = E43.MARGIN          # 0.5%p
LATE_TOL = E43.LATE_TOL      # 0.3%p
EARLY_TOL = 0.05             # early_guard (새로 더함 — 사전 고정)
A_CANDS = ["M0+M4u", "M0+M4u+M1", "M2a+M4u+M1", "M2a+M4l+M1"]
A_REFS = ["M0", "M2a"]
SCHEME_OF = {"M0": "prev", "M2a": "recent5", "M2b": "forward"}
WINDOWS = ("main", "late", "t12", "p10_main", "p10_late")
NUM_KEYS = ("cov80", "cov95", "mean_med_abs_err", "mean_med_abs_err_b1", "low_ratio", "med_w80", "cov80_grade_AB", "n_grade_AB",
            "season_cov80_min", "n_season_cov80_lt50")


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def crit2():
    return jl(os.path.join(AN, "criteria_v2.json"))


def sv_of(rows, name, with_m1):
    """줄 → (M1 얹은 줄, summarize_v2 한 scheme 값, M1 배율)."""
    fac = None
    if with_m1:
        rows, fac = E43.m1(rows)
    sv = XV.summarize_v2([dict(r, scheme=name) for r in rows], crit2())[name]
    return rows, sv, fac


def win(sv, w):
    return sv.get(w) or {}


def brief(sv):
    out = {}
    for w in WINDOWS:
        x = sv.get(w)
        if x:
            out[w] = {k: x.get(k) for k in NUM_KEYS if x.get(k) is not None}
            if w == "main":
                out[w]["usable"] = x.get("usable")
                out[w]["season_bias"] = x.get("season_bias")
                out[w]["season_cov80"] = x.get("season_cov80")
    if sv.get("by_volume_main"):
        out["by_volume_main"] = sv["by_volume_main"]
    return out


def avg_sv(svs):
    """시드 여럿의 summarize_v2 값 → 창마다 숫자 열의 평균."""
    out = {}
    for w in WINDOWS:
        xs = [s.get(w) for s in svs if s.get(w)]
        if len(xs) != len(svs):
            continue
        out[w] = {k: round(statistics.mean(x[k] for x in xs), 5) for k in NUM_KEYS if all(x.get(k) is not None for x in xs)}
    return out


def pick_rule(names, sv, rank, ref, guards=None):
    """4.3 고르기 규칙 (e_43.select와 같은 꼴, 후보와 ref를 나눔): (1) main 포함 ≥ 0.70 (2) |e| 최소 (3) 0.5%p 안 rank 순
    (4) late_guard (T_late 포함 ≥ 0.70, ref보다 T_late |e| 0.3%p 넘게 나쁘지 않음) + guards(이름 → 사유 목록). 걸리면 다음."""
    kept = [n for n in names if sv[n]["main"]["cov80"] >= 0.70]
    step1 = "포함 0.70 이상"
    if not kept:
        bc = max(sv[n]["main"]["cov80"] for n in names)
        kept = [n for n in names if sv[n]["main"]["cov80"] == bc]
        step1 = "모두 0.70 미달 → 포함률이 가장 높은 것"
    mae = lambda n: sv[n]["main"]["mean_med_abs_err"]
    best = min(mae(n) for n in kept)
    near = sorted([n for n in kept if mae(n) <= best + MARGIN + 1e-12], key=lambda n: (rank[n], mae(n)))
    rest = sorted([n for n in kept if n not in near], key=mae)
    tried, pick = [], None
    for n in near + rest:
        why = []
        lt = sv[n].get("late")
        if lt and lt["cov80"] < 0.70:
            why.append(f"T_late 80% 포함 {lt['cov80']:.3f} < 0.70")
        r = ref.get(n)
        if r and lt and sv.get(r, {}).get("late"):
            d = lt["mean_med_abs_err"] - sv[r]["late"]["mean_med_abs_err"]
            if d > LATE_TOL:
                why.append(f"T_late |e|가 {r}보다 {d:.2%}p 나쁨")
        if guards:
            why += guards(n)
        tried.append({"name": n, "guard": why})
        if not why:
            pick = n
            break
    return {"pick": pick, "step1": step1, "kept": kept, "best_mae": best, "within_0_5pp": near, "tried": tried}


def row_table(names, sv):
    return [{"name": n, "cov80": sv[n]["main"]["cov80"], "mae": sv[n]["main"]["mean_med_abs_err"], "low": sv[n]["main"].get("low_ratio"),
             "season_cov_min": sv[n]["main"].get("season_cov80_min"), "late_cov80": win(sv[n], "late").get("cov80"),
             "late_mae": win(sv[n], "late").get("mean_med_abs_err"), "t12_cov80": win(sv[n], "t12").get("cov80"),
             "t12_mae": win(sv[n], "t12").get("mean_med_abs_err"), "p10_cov80": win(sv[n], "p10_main").get("cov80"),
             "p10_mae": win(sv[n], "p10_main").get("mean_med_abs_err")} for n in names]


def show(tab, title):
    print(f"\n{title}")
    print(f"  {'이름':24s} 포함   |e|     낮게 시즌최저 | late 포함 |e|   | T12 포함 |e|   | 10% 포함 |e|")
    f = lambda v, p=False: "  -  " if v is None else (f"{v:.2%}" if p else f"{v:.2f}")
    for t in tab:
        print(f"  {t['name']:24s} {f(t['cov80'])} {f(t['mae'], 1)} {f(t['low'])} {f(t['season_cov_min'])}    | {f(t['late_cov80'])} {f(t['late_mae'], 1)} | "
              f"{f(t['t12_cov80'])} {f(t['t12_mae'], 1)} | {f(t['p10_cov80'])} {f(t['p10_mae'], 1)}")


def load_out(p):
    return jl(p) if os.path.exists(p) else {"about": "할 일 4.3 마무리 (2026-10-18) — 꼴 정하기·M5·10%. 사전 고정 커밋 9d52d30. 코드 tools/solo_raid/lab4/e_43b.py·m5run.py"}


def save_out(p, d):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("→", p)


# ---------------- A 꼴 정하기 ----------------
def stage_a(res, out):
    seeds = sorted(d for d in os.listdir(res) if d.startswith("s") and d[1:].isdigit())
    per = {}
    meta = {}
    rank = ref = None
    for s in seeds:
        cands, m = E43.load_candidates(os.path.join(res, s))
        C, all_sv, _ = E43.build(cands, crit2(), gate=False)
        per[s] = {n: C[n]["sv"] for n in A_CANDS + A_REFS}
        meta[s] = m
        rank = {n: tuple(C[n]["rank"]) for n in A_CANDS + A_REFS}
        ref = {n: C[n]["ref"] for n in A_CANDS + A_REFS}
    avg = {n: avg_sv([per[s][n] for s in seeds]) for n in A_CANDS + A_REFS}
    sel = pick_rule(A_CANDS, avg, rank, ref)
    show(row_table(A_CANDS + A_REFS, avg), f"A 시드 {len(seeds)}개 평균 (뽑기 2000)")
    print("고르기:", sel["pick"], "· 0.5%p 안", sel["within_0_5pp"], "· 시도", sel["tried"])
    per_pick = {}
    for s in seeds:
        sl = pick_rule(A_CANDS, per[s], rank, ref)
        per_pick[s] = {"pick": sl["pick"], "within_0_5pp": sl["within_0_5pp"], "tried": sl["tried"]}
        show(row_table(A_CANDS + A_REFS, per[s]), f"(보고) 시드 {s}: 고름 {sl['pick']}")
    spread = {n: {w: {k: [min(per[s][n][w][k] for s in seeds), max(per[s][n][w][k] for s in seeds)]
                      for k in ("cov80", "mean_med_abs_err")} for w in ("main", "late")} for n in A_CANDS + A_REFS}
    d = load_out(out)
    d["A"] = {"rule": "사전 고정 A — 시드 1·2·3(뽑기 2000) 평균으로 4.3 고르기", "seeds": seeds, "inputs": meta,
              "avg": {n: avg[n] for n in A_CANDS + A_REFS}, "per_seed": {s: {n: brief(per[s][n]) for n in A_CANDS + A_REFS} for s in seeds},
              "rank": {n: list(v) for n, v in rank.items()}, "late_ref": ref, "select": sel, "per_seed_pick": per_pick, "seed_range": spread,
              "F": sel["pick"]}
    save_out(out, d)


# ---------------- B M5 ----------------
# 무리: (이름, [(변형 열쇠, 설명)], 순서 있음?) — 사전 고정 표. 「cur」는 지금 값 (A의 시드 1 F 실행).
GROUPS = [
    ("①⑤ 서버 미상 줄", [("none", "라벨 줄만"), ("cur", "site (지금)"), ("em", "site-em"), ("mix5", "mix gate 최소 5"), ("mix3", "mix gate 최소 3")], True),
    ("① 강건 처리", [("cur", "가중 평균·꼭대기 제외 끔 (지금)"), ("wmedian", "가중 중앙값"), ("topcut", "꼭대기 0.5% 제외")], True),
    ("② 점 하나가 틀림", [("eps0", "끔"), ("cur", "4명 미만 (지금)"), ("epsall", "모든 사람 수")], True),
    ("④ τ_shape", [("cur", "1 (지금)"), ("ts05", "0.5"), ("ts2", "2")], False),
    ("⑥ 쌍 갱신", [("cur", "끔 (지금)"), ("pupd", "켬")], True),
    ("⑥ 외삽 불확실성", [("cur", "전체 (지금)"), ("sfhalf", "절반")], False),
    ("⑥ 구간 관측", [("noint", "뺌"), ("cur", "넣음 (지금)")], True),
    ("⑦ 섞임 민감도", [("cur", "안 넓힘 (지금)"), ("contam", "80% 위 끝을 f 20% 변화만큼 넓힘")], True),
]


def F_parts(F):
    """「M0+M4u+M1」 → (기준 M0, 덧붙임 M4u, M1 켬)."""
    xs = F.split("+")
    return xs[0], next((x for x in xs[1:] if x != "M1"), None), "M1" in xs


def a_rows(resA, F, seed="s1"):
    base, kind, _ = F_parts(F)
    f = {None: "base", "M4u": "m4u", "M4l": "m4l"}[kind]
    d = jl(os.path.join(resA, seed, f + ".json"))
    return [r for r in d["rows"] if r["scheme"] == SCHEME_OF[base]], d["run"]


def contam_rows(rows, f="0.2"):
    """⑦: 80% 위 끝을 f 몫 중앙 변화만큼 넓힌다 (predict i80_ext와 같은 셈, 위로만). 미상 줄이 없거나 섞임 줄이 없는 칸은 그대로."""
    out = []
    for r in rows:
        r2 = dict(r)
        sh = (r.get("c_shift") or {}).get(f)
        if sh is not None and r.get("err") is not None:
            lo, hi = r["i80"]
            hi2 = hi * (1 + max(sh, 0.0))
            r2.update(i80=[lo, hi2], in80=lo <= r["truth"] <= hi2, w80=hi2 / lo - 1)
        out.append(r2)
    return out


def stage_b(res, out, resA):
    d = load_out(out)
    F = d["A"]["F"]
    base, kind, m1on = F_parts(F)
    sch = SCHEME_OF[base]
    rows_v, meta = {}, {}
    cur_rows, cur_run = a_rows(resA, F)
    rows_v["cur"] = cur_rows
    meta["cur"] = {"file": f"A/s1 ({F})", "tool_head": cur_run["tool_head"], "args": cur_run["args"]}
    for fn in sorted(os.listdir(res)):
        if not fn.endswith(".json"):
            continue
        k = fn[:-5]
        x = jl(os.path.join(res, fn))
        rows_v[k] = [r for r in x["rows"] if r["scheme"] == sch]
        meta[k] = {"tool_head": x["run"]["tool_head"], "tools_dirty": x["run"]["tools_dirty"], "args": x["run"]["args"],
                   "lab_patches": x["run"].get("lab_patches"), "failures": x.get("failures", [])}
    # ⑦은 섞임 민감도를 켠 지금 값 실행(contam_run, m5run — 등급 사유도 남김)에서 사후로. 주 결과가 A 시드 1과 같으면 그것을 지금 값 줄로 쓴다
    if "contam_run" not in rows_v:
        raise SystemExit("contam_run.json (지금 값 + 섞임 민감도, m5run) 이 없다")
    rows_v["contam"] = contam_rows(rows_v["contam_run"])
    same = [(r["season"], r["T"], r["pct"], r["pred"], r["i80"]) for r in rows_v["contam_run"]] == \
           [(r["season"], r["T"], r["pct"], r["pred"], r["i80"]) for r in cur_rows]
    meta["contam_run"]["same_as_cur"] = same
    print("섞임 민감도 켠 실행의 주 결과가 A 시드 1 지금 값과 같은가:", same)
    if not same:
        raise SystemExit("contam_run이 A 시드 1과 다르다 — 확인")
    rows_v["cur"] = rows_v["contam_run"]
    sv, m1rows = {}, {}
    for k, rows in rows_v.items():
        if k == "contam_run":
            continue
        m1rows[k], sv[k], _ = sv_of(rows, k, m1on)
    cur = sv["cur"]

    def guards(n):
        if n == "cur":
            return []
        why = []
        if win(sv[n], "t12") and win(cur, "t12") and sv[n]["t12"]["cov80"] < cur["t12"]["cov80"] - EARLY_TOL:
            why.append(f"early_guard: T12 포함 {sv[n]['t12']['cov80']:.3f} < 지금 {cur['t12']['cov80']:.3f} − 0.05")
        return why
    groups = []
    for gname, opts, ordered in GROUPS:
        keys = [k for k, _ in opts if k in sv]
        miss = [k for k, _ in opts if k not in sv]
        if ordered:
            rank = {k: i for i, k in enumerate(keys)}
        else:   # 순서 없음 — 지금 값이 앞, 나머지는 |e| 순
            rank = {k: (0 if k == "cur" else 1) for k in keys}
        ref = {k: (None if k == "cur" else "cur") for k in keys}
        sel = pick_rule(keys, sv, rank, ref, guards)
        groups.append({"group": gname, "options": {k: v for k, v in opts}, "ordered": ordered, "missing": miss, "select": sel,
                       "table": row_table(keys, sv)})
        show(row_table(keys, sv), f"B {gname}: 고름 {sel['pick']} (0.5%p 안 {sel['within_0_5pp']})")
        for t in sel["tried"]:
            if t["guard"]:
                print("   guard:", t)
    changed = [(g["group"], g["select"]["pick"]) for g in groups if g["select"]["pick"] not in (None, "cur")]
    combo = None
    final = "cur"
    if len(changed) == 1:
        final = changed[0][1]
    elif len(changed) > 1:
        if "combo" in sv:
            best1 = min((k for _, k in changed), key=lambda k: sv[k]["main"]["mean_med_abs_err"])
            ok = sv["combo"]["main"]["cov80"] >= 0.70 and sv["combo"]["main"]["mean_med_abs_err"] <= sv[best1]["main"]["mean_med_abs_err"] + LATE_TOL
            gain = max((k for _, k in changed), key=lambda k: cur["main"]["mean_med_abs_err"] - sv[k]["main"]["mean_med_abs_err"])
            final = "combo" if ok else gain
            combo = {"changed": changed, "best_single": best1, "combo_ok": ok, "largest_gain_single": gain}
            show(row_table(["cur", "combo"] + [k for _, k in changed], sv), f"B 함께 켬: {'함께' if ok else '하나만 ' + gain}")
        else:
            combo = {"changed": changed, "note": "함께 켠 실행(combo.json)이 아직 없다 — 돌린 뒤 다시"}
            final = None
    print("B 바뀐 무리:", changed, "→ 최종", final)
    d["B"] = {"rule": "사전 고정 B — F 위 설정값 하나씩 (뽑기 2000·시드 1), M1 다시, early_guard", "F": F, "inputs": meta,
              "groups": groups, "changed": changed, "combo": combo, "final": final,
              "sv": {k: brief(v) for k, v in sv.items()}}
    if final:
        d["B"]["grades"] = grades(m1rows[final], rows_v[final])
    sd = os.path.join(os.path.dirname(os.path.normpath(res)), "Bseed")
    if final and final != "cur" and os.path.isdir(sd):   # 최종 설정의 시드 2·3 (보고만) — 지금 값은 A의 s2·s3
        rep = {}
        for s in ("s2", "s3"):
            p = os.path.join(sd, f"{final}_{s}.json")
            if not os.path.exists(p):
                continue
            x = jl(p)
            _, s_new, _ = sv_of([r for r in x["rows"] if r["scheme"] == sch], final, m1on)
            c_rows, _ = a_rows(resA, F, s)
            _, s_cur, _ = sv_of(c_rows, "cur", m1on)
            rep[s] = {"cur": brief(s_cur), final: brief(s_new), "tool_head": x["run"]["tool_head"]}
            f = lambda v: f"포함 {v['main']['cov80']:.2f} |e| {v['main']['mean_med_abs_err']:.2%} late {v['late']['cov80']:.2f} {v['late']['mean_med_abs_err']:.2%} T12 {v['t12']['cov80']:.2f}"
            print(f"(보고) {s}: 지금 {f(s_cur)} → {final} {f(s_new)}")
        d["B"]["final_other_seeds"] = rep
    save_out(out, d)


# ---------------- ③ 등급 문턱 ----------------
TUNABLE = {"ttrust": lambda s: s.startswith("[C] T ") and "초반 % 변환 근거가 얇다" in s,
           "width": lambda s: s.startswith("[C] 80% 구간 폭이"),
           "few": lambda s: s.startswith("[C] 관측 적음"),
           "bpers": lambda s: s.startswith("[B] 쓴 사람(유효)"),
           "blabel": lambda s: s.startswith("[B] 서버 라벨 사람")}
GRID = {"t_c": [12, 18, 24], "w_c": [0.15, 0.20, 0.25, 0.30, None], "n_c": [3, 5], "n_b": [10, 20], "lab_b": [True, False]}
CURRENT = {"t_c": 18, "w_c": 0.15, "n_c": 3, "n_b": 10, "lab_b": True}


def fixed_level(reasons):
    lv = "A"
    for s in reasons or []:
        if any(f(s) for f in TUNABLE.values()):
            continue
        lv = max(lv, s[1])
    return lv


def regrade(r, wmap, g):
    lv = r["_fixed"]
    if "D" in lv:
        return lv
    add = []
    if r["T"] < g["t_c"]:
        add.append("C")
    w = wmap.get((r["season"], r["T"]))
    if g["w_c"] is not None and w is not None and w > g["w_c"]:
        add.append("C")
    n = r.get("n_persons") or 0
    if n < g["n_c"]:
        add.append("C")
    elif n < g["n_b"]:
        add.append("B")
    if g["lab_b"] and (r.get("n_persons_label") or 0) < 3 and r.get("mixed_rows"):
        add.append("B")
    return max([lv] + add)


def width_map(rows):
    """(시즌, T) → 1·3% 중 큰 80% 폭/중앙 (predict warning과 같은 셈, 넘겨받은 구간 — M1 뒤)."""
    m = {}
    for r in rows:
        if r["pct"] in (1.0, 3.0) and r.get("pred") and r.get("i80") and None not in r["i80"]:
            w = (r["i80"][1] - r["i80"][0]) / r["pred"]
            k = (r["season"], r["T"])
            m[k] = max(m.get(k, 0.0), w)
    return m


def grade_stats(rows, key):
    out = {}
    for lv in "ABCD":
        L = [r for r in rows if r[key] == lv]
        if L:
            out[lv] = {"n": len(L), "cov80": round(sum(r["in80"] for r in L) / len(L), 3),
                       "mean_abs_err": round(statistics.mean(abs(r["err"]) for r in L), 5)}
    AB = [r for r in rows if r[key] in ("A", "B")]
    out["AB"] = {"n": len(AB), "cov80": round(sum(r["in80"] for r in AB) / len(AB), 3) if AB else None}
    bys = collections.defaultdict(list)
    for r in AB:
        bys[r["season"]].append(r["in80"])
    sc = {s: sum(v) / len(v) for s, v in bys.items()}
    out["AB"]["season_cov80_min"] = round(min(sc.values()), 3) if sc else None
    out["AB"]["n_season_lt50"] = sum(v < 0.5 for v in sc.values())
    return out


def sel13(rows):
    R = [dict(r) for r in rows if r.get("err") is not None and r["pct"] in (1.0, 3.0) and r["server"] == "KR"]
    for r in R:
        if r.get("warn") is None:
            raise SystemExit("줄에 등급 사유(warn)가 없다 — m5run으로 낸 줄이어야 한다")
        r["_fixed"] = fixed_level(r["warn"])
    return R


def grades(rows, raw_rows):
    # 확인: 지금 문턱 + M1 앞 폭으로 다시 매기면 predict 등급과 같아야 한다
    R0 = sel13(raw_rows)
    w0 = width_map(R0)
    mism = sum(regrade(r, w0, CURRENT) != r["grade"] for r in R0)
    print(f"③ 확인: 지금 문턱·M1 앞 폭으로 다시 매긴 등급이 predict와 다른 줄 {mism}/{len(R0)}")
    R = sel13(rows)
    wm = width_map(R)
    combos = []
    for vals in itertools.product(*GRID.values()):
        g = dict(zip(GRID.keys(), vals))
        for r in R:
            r["_g"] = regrade(r, wm, g)
        st = grade_stats(R, "_g")
        A = st.get("A", {"n": 0, "cov80": None})
        ok = st["AB"]["n"] > 0 and st["AB"]["cov80"] >= 0.70 and (A["n"] == 0 or A["cov80"] >= 0.75)
        dist = sum(abs(GRID[k].index(g[k]) - GRID[k].index(CURRENT[k])) for k in GRID)
        combos.append({"g": g, "ok": ok, "AB_n": st["AB"]["n"], "A_n": A["n"], "dist": dist, "stats": st})
    okc = [c for c in combos if c["ok"]]
    pick = min(okc, key=lambda c: (-c["AB_n"], -c["A_n"], c["dist"])) if okc else None
    cur = next(c for c in combos if c["g"] == CURRENT)
    # 사전 규칙을 넘는 칸이 없으면 지금 문턱 그대로 (사전 고정에 없던 경우 — 계산 뒤 정한 처리, 문서에 적는다)
    decision = "사전 규칙대로 고름" if pick else "사전 규칙 (1)을 넘는 칸 없음 → 지금 문턱 그대로 (계산 뒤 정한 처리)"
    use = pick or cur
    by_T = {}
    for r in R:
        r["_g"] = regrade(r, wm, use["g"])
    for T in sorted({r["T"] for r in R}):
        L = [r for r in R if r["T"] == T]
        by_T[f"T{T:g}"] = {lv: {"n": len(LL), "cov80": round(sum(r["in80"] for r in LL) / len(LL), 3),
                                "mean_abs_err": round(statistics.mean(abs(r["err"]) for r in LL), 5)}
                           for lv in "ABCD" for LL in [[r for r in L if r["_g"] == lv]] if LL}
    top_a = sorted(combos, key=lambda c: -(c["stats"].get("A", {}).get("cov80") or 0))[:5]
    print(f"\n③ 등급: 지금 문턱 {CURRENT} → A·B {cur['AB_n']}줄 포함 {cur['stats']['AB']['cov80']} · A {cur['A_n']}줄 {cur['stats'].get('A', {}).get('cov80')} · 통과 {cur['ok']}")
    if pick:
        print(f"   고름 {pick['g']} → A·B {pick['AB_n']}줄 포함 {pick['stats']['AB']['cov80']} · A {pick['A_n']}줄 {pick['stats'].get('A', {}).get('cov80')} · 등급별 {pick['stats']}")
    print("   ", decision, "· A 포함이 가장 높은 칸:", [(c["g"], c["stats"].get("A", {}).get("cov80"), c["A_n"]) for c in top_a[:3]])
    print("   T별 (쓴 문턱):", by_T)
    return {"decision": decision, "use": use["g"], "top_A_cov": [{"g": c["g"], "A": c["stats"].get("A"), "AB": c["stats"]["AB"]} for c in top_a],
            "check_regrade_mismatch": [mism, len(R0)], "rule": "사전 고정 ③ — M1 뒤 한섭 1·3% T12~96: (1) A·B 포함 ≥ 0.70, A ≥ 0.75 (2) A·B 줄 수 최대 (3) A 줄 수 (4) 지금 문턱과 격자 거리",
            "grid": GRID, "current": cur, "pick": pick, "n_ok": len(okc), "n_rows": len(R), "pick_by_T": by_T,
            "predict_grade_on_rows": grade_stats(R, "grade")}


# ---------------- C 10% ----------------
def stage_c(res, out, resA=None):
    d = load_out(out)
    g43 = jl(os.path.join(AN, "diag43.json"))
    cand = g43["candidates"]
    names = list(cand)
    sv = {n: {"main": cand[n]["summary_v2"]["p10_main"], "late": cand[n]["summary_v2"].get("p10_late")} for n in names}
    rank = {n: tuple(cand[n]["rank"]) for n in names}
    ref = {n: cand[n]["late_ref"] for n in names}
    sel = pick_rule(names, sv, rank, ref)
    tab = [{"name": n, "rank": list(rank[n]), "cov80": sv[n]["main"]["cov80"], "mae": sv[n]["main"]["mean_med_abs_err"],
            "low": sv[n]["main"]["low_ratio"], "season_cov_min": sv[n]["main"]["season_cov80_min"],
            "late_cov80": (sv[n]["late"] or {}).get("cov80"), "late_mae": (sv[n]["late"] or {}).get("mean_med_abs_err"),
            "usable": sv[n]["main"]["usable"]} for n in sorted(names, key=lambda n: rank[n])]
    print("\nC 10% (diag43 후보 22개, 시드 1·뽑기 600)")
    for t in tab:
        print(f"  {t['name']:20s} 포함 {t['cov80']:.2f} |e| {t['mae']:.2%} 낮게 {t['low']:.2f} 시즌최저 {t['season_cov_min']:.2f} | late {t['late_cov80']:.2f} {t['late_mae']:.2%}")
    print("C 고르기:", sel["pick"], "· 0.5%p 안", sel["within_0_5pp"], "· 시도", sel["tried"])
    C = {"rule": "사전 고정 C — criteria 지점별 규칙을 diag43 후보 22개의 p10_main에, late_guard는 p10_late", "table": tab, "select": sel, "P10": sel["pick"]}
    P10 = sel["pick"]
    if P10 and res and os.path.exists(os.path.join(res, "p10n.json")):
        base, kind, m1on = F_parts(P10)
        sch = SCHEME_OF[base]
        sv2, meta = {}, {}
        for k in ("p10", "p10n"):
            x = jl(os.path.join(res, k + ".json"))
            rows = [r for r in x["rows"] if r["scheme"] == sch]
            _, s, _ = sv_of(rows, k, m1on)
            sv2[k] = {"main": s["p10_main"], "late": s.get("p10_late"), "m13": s["main"], "m13_late": s.get("late")}
            meta[k] = {"tool_head": x["run"]["tool_head"], "tools_dirty": x["run"]["tools_dirty"], "args": x["run"]["args"],
                       "lab_patches": x["run"].get("lab_patches")}
            nr = [r for r in rows if r["pct"] == 10.0 and r.get("near_obs") is not None]
            sv2[k]["near_obs_by_T"] = {f"T{T:g}": round(statistics.mean(r["near_obs"] for r in nr if r["T"] == T), 1) for T in sorted({r["T"] for r in nr})}
        same = abs(sv2["p10"]["main"]["mean_med_abs_err"] - sv[P10]["main"]["mean_med_abs_err"]) < 1e-9 and sv2["p10"]["main"]["cov80"] == sv[P10]["main"]["cov80"]
        sel2 = pick_rule(["p10", "p10n"], sv2, {"p10": 0, "p10n": 1}, {"p10": None, "p10n": "p10"})
        print(f"  P10 다시 낸 값이 diag43과 같은가: {same}")
        for k in ("p10", "p10n"):
            v = sv2[k]
            print(f"  {k:5s} 10% 포함 {v['main']['cov80']:.2f} |e| {v['main']['mean_med_abs_err']:.2%} late {v['late']['cov80']:.2f} {v['late']['mean_med_abs_err']:.2%}"
                  f" | 1·3% 포함 {v['m13']['cov80']:.2f} |e| {v['m13']['mean_med_abs_err']:.2%} | ×2 창 관측 수(10%) {v['near_obs_by_T']}")
        print("  N10 고르기:", sel2["pick"], sel2["tried"])
        C.update({"P10_rerun_same_as_diag43": same, "N10": {"inputs": meta, "sv": sv2, "select": sel2}, "pick_10": "P10+N10" if sel2["pick"] == "p10n" else P10})
    if res and os.path.exists(os.path.join(res, "fn10.json")) and d.get("B"):
        F = d["A"]["F"]
        base, kind, m1on = F_parts(F)
        x = jl(os.path.join(res, "fn10.json"))
        rows = [r for r in x["rows"] if r["scheme"] == SCHEME_OF[base]]
        _, s, _ = sv_of(rows, "fn10", m1on)
        cur_rows, _ = a_rows(resA, F)
        _, s0, _ = sv_of(cur_rows, "cur", m1on)
        C["F_N10_report"] = {"F": F, "cur": {"p10_main": s0["p10_main"], "p10_late": s0.get("p10_late"), "main": brief(s0)["main"]},
                             "n10": {"p10_main": s["p10_main"], "p10_late": s.get("p10_late"), "main": brief(s)["main"]}}
        print(f"  (보고) F 위 N10: 10% |e| {s0['p10_main']['mean_med_abs_err']:.2%} → {s['p10_main']['mean_med_abs_err']:.2%}, 포함 {s0['p10_main']['cov80']:.2f} → {s['p10_main']['cov80']:.2f}")
    d["C"] = C
    save_out(out, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["a", "b", "c"])
    ap.add_argument("res")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resA", help="b·c: A 결과 폴더 (지금 값 = s1의 F)")
    a = ap.parse_args()
    if a.stage == "a":
        stage_a(a.res, a.out)
    elif a.stage == "b":
        stage_b(a.res, a.out, a.resA)
    else:
        stage_c(a.res, a.out, a.resA)


if __name__ == "__main__":
    main()
