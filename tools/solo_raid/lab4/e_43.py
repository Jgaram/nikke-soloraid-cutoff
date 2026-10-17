"""할 일 4.3 — 1·3% 모델 후보 고르기 (2026-10-18 4.3 세션). 후보·관문·고르기 규칙은 계산 전에 문서에 고정했다
(docs/solo-raid-analysis.md 4.3 「사전 고정」, 커밋 e22da7e).

입력 (사본에서 xval.py로 낸 것, --slim — 실행 명령은 lab4 README):
  base.json  scheme prev·recent5·forward·loso       → M0·M2a·M2b·loso(비교)
  m4u.json   prev·recent5·forward + --level-update '{}'                   → +M4u
  m4l.json   prev·recent5·forward + --level-update '{"unknown": false}'   → +M4l
  m4e.json   forward + --predict-opt early_level=true                     → M2b+M4e
  m2s.json   forward + --level-knn '{}'                                    → M2b+M2s
  pairs.json scheme pairs (M0′ 진단, 맞힐 시즌 23개)
하는 일: M1(앞 맞힐 시즌 |z| 80% 분위로 80% 구간 넓히기)을 후보마다 얹고, criteria_v2 창으로 요약(xval.summarize_v2),
M0′ 진단·M2s 관문(상한 vs M0, 0.5%p), 고르기((1) 포함 ≥ 0.70 (2) |e| 최소 (3) 0.5%p 안 단순한 것, late_guard).
  python3 tools/solo_raid/lab4/e_43.py RES_DIR [--out data/solo-raid/analysis/diag43.json]
"""
import argparse
import collections
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
import xval as XV  # noqa: E402

AN = os.path.join(ROOT, "data", "solo-raid", "analysis")
BASES = {"prev": "M0", "recent5": "M2a", "forward": "M2b"}
BASE_IDX = {"M0": 0, "M2a": 1, "M2b": 2}
KIND_IDX = {None: 0, "M4u": 1, "M4l": 2, "M4e": 3, "M2s": 4}   # 사전 고정: (없음 →) M1 → M4u → M4l → M4e → M2s
FILES = {"base": None, "m4u": "M4u", "m4l": "M4l", "m4e": "M4e", "m2s": "M2s"}
MARGIN = 0.005      # 0.5%p (criteria_v2 selection (3), M2s 관문)
LATE_TOL = 0.003    # late_guard 0.3%p
M1_MIN_PREV = 3


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sn(s):
    return int(str(s).lstrip("S"))


# ---------------- M1 구간 보정 (사전 고정 — 넓히기만, 앞 맞힐 시즌 3개 이상) ----------------
def zval(r):
    lo, hi = r["i80"]
    m, t = r["pred"], r["truth"]
    if not (m and t and lo and hi) or lo <= 0 or hi <= m or m <= lo:
        return None
    d = math.log(t / m)
    h = math.log(hi / m) if d >= 0 else math.log(m / lo)
    return d / h if h > 0 else None


def wq(items, q):
    items = sorted(items)
    tot = sum(w for _, w in items)
    acc = 0.0
    for v, w in items:
        acc += w
        if acc >= q * tot - 1e-12:
            return v
    return items[-1][0]


def m1(rows, groups=((1.0, 3.0), (10.0,))):
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("err") is None:
            continue
        z = zval(r)
        g = next((g for g in groups if r["pct"] in g), None)
        if z is not None and g:
            by[(g, r["T"])].append((r["season"], abs(z)))
    cmap = {}
    for (g, T), L in by.items():
        for t in sorted({s for s, _ in L}, key=sn):
            prev = [(s, z) for s, z in L if sn(s) < sn(t)]
            cnt = collections.Counter(s for s, _ in prev)
            c = 1.0 if len(cnt) < M1_MIN_PREV else max(1.0, wq([(z, 1.0 / cnt[s]) for s, z in prev], 0.8))
            cmap[(g, T, t)] = c
    out = []
    for r in rows:
        r2 = dict(r)
        g = next((g for g in groups if r["pct"] in g), None)
        c = cmap.get((g, r["T"], r["season"]))
        if r.get("err") is not None and c and c != 1.0:
            m = r["pred"]
            lo, hi = r["i80"]
            lo2, hi2 = m * (lo / m) ** c, m * (hi / m) ** c
            r2.update(i80=[lo2, hi2], in80=lo2 <= r["truth"] <= hi2, w80=hi2 / lo2 - 1)
        r2["m1_c"] = c
        out.append(r2)
    factors = collections.defaultdict(dict)
    for (g, T, t), c in cmap.items():
        factors[f"{'/'.join(f'{x:g}' for x in g)}"].setdefault(f"T{T:g}", {})[t] = round(c, 3)
    return out, factors


# ---------------- 후보 모으기 ----------------
def load_candidates(res):
    cands, meta = {}, {}
    for f, kind in FILES.items():
        p = os.path.join(res, f + ".json")
        if not os.path.exists(p):
            print("없음:", p)
            continue
        d = jl(p)
        meta[f] = {"tool_head": d["run"]["tool_head"], "tools_dirty": d["run"]["tools_dirty"], "args": d["run"]["args"],
                   "criteria_v2_sha1": d["run"].get("criteria_v2_sha1"), "failures": d.get("failures", [])}
        by = collections.defaultdict(list)
        for r in d["rows"]:
            by[r["scheme"]].append(r)
        for sch, rows in by.items():
            if sch == "loso":
                cands["loso"] = {"rows": rows, "base": None, "kind": None, "comparison_only": True}
                continue
            b = BASES[sch]
            name = b if kind is None else f"{b}+{kind}"
            cands[name] = {"rows": rows, "base": b, "kind": kind}
    return cands, meta


def rank(c, m1_on):
    n_add = (c["kind"] is not None) + m1_on
    return (n_add, KIND_IDX[c["kind"]], BASE_IDX[c["base"]])


def level_stats(rows):
    """수준 덧붙임이 실제로 들어간 (시즌, T) 비율 — T마다."""
    seen = {}
    for r in rows:
        k = (r["season"], r["T"])
        v = r.get("m_upd") or r.get("lvl_knn") or r.get("lvl_early")
        seen[k] = seen.get(k) or bool(v)
    out = {}
    for T in sorted({k[1] for k in seen}):
        L = [v for k, v in seen.items() if k[1] == T]
        out[f"T{T:g}"] = f"{sum(L)}/{len(L)}"
    return out


# ---------------- M0′ 진단 ----------------
def season_feats():
    pd = jl(os.path.join(AN, "prior_dev.json"))
    lv = pd["transform"].get("season_levels") or {}
    ak = jl(os.path.join(AN, "answer_key.json"))["rows"]
    tr = collections.defaultdict(dict)
    for r in ak:
        if r["server"] == "KR" and r["truth"]:
            tr[r["season"]][float(r["pct"])] = r["truth"]
    reg = {x["season"]: x for x in jl(os.path.join(ROOT, "data", "solo-raid", "registry.json"))["seasons"]}
    F = {}
    for s in reg:
        t = tr.get(s, {})
        F[s] = {"ln_ms": math.log(lv[s]) if s in lv else None,
                "ln_10_3": math.log(t[10.0] / t[3.0]) if 10.0 in t and 3.0 in t else None,
                "ln_1_3": math.log(t[1.0] / t[3.0]) if 1.0 in t and 3.0 in t else None,
                "ln_kr_rows": math.log(reg[s]["data"]["use_by_server"]["KR"]) if reg[s]["data"].get("use_by_server", {}).get("KR") else None}
    return F


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def m0prime(res, crit2, F, targets):
    d = jl(os.path.join(res, "pairs.json"))
    P = d.get("pairs_v2") or XV.pairs_v2(d["rows"], crit2)
    cell = {(p["base"], p["target"]): p for p in P}
    bases = sorted({p["base"] for p in P}, key=sn)
    tgts = sorted({p["target"] for p in P}, key=sn)
    # 기준 시즌마다 (다른 모든 맞힐 시즌)
    per_base = {}
    for b in bases:
        L = [cell[(b, t)] for t in tgts if (b, t) in cell]
        per_base[b] = {"n": len(L), "mean_abs_err": round(statistics.mean(x["mean_abs_err"] for x in L), 5),
                       "mean_err": round(statistics.mean(x["mean_err"] for x in L), 5),
                       "cov80": round(statistics.mean(x["cov80"] for x in L), 3)}
    order = sorted(bases, key=lambda b: per_base[b]["mean_abs_err"])
    maes = [per_base[b]["mean_abs_err"] for b in bases]
    dist = {"n_bases": len(bases), "min": min(maes), "q25": round(XV._q(maes, 0.25), 5), "median": round(statistics.median(maes), 5),
            "q75": round(XV._q(maes, 0.75), 5), "max": max(maes), "best": order[:3], "worst": order[-3:],
            "S41_rank": order.index("S41") + 1 if "S41" in order else None, "S41": per_base.get("S41")}
    # 쌍 오차와 두 시즌 차이 (맞힐 − 기준)
    expl = {}
    for f in ("ln_ms", "ln_10_3", "ln_1_3", "ln_kr_rows"):
        xs, ys, ya = [], [], []
        for p in P:
            a, b = F.get(p["target"], {}).get(f), F.get(p["base"], {}).get(f)
            if a is None or b is None:
                continue
            xs.append(a - b)
            ys.append(p["mean_err"])
            ya.append(p["mean_abs_err"])
        r = pearson(xs, ys)
        # |Δ|와 |e|: 차이가 클수록 크게 틀리나
        ra = pearson([abs(x) for x in xs], ya)
        expl[f] = {"n_pairs": len(xs), "r_signed": None if r is None else round(r, 3), "r2_signed": None if r is None else round(r * r, 3),
                   "r_absdelta_abserr": None if ra is None else round(ra, 3)}
    # 상한 (M2s 관문) — 맞힐 시즌 S21~S41
    rows_ub = []
    for t in targets:
        prevs = [b for b in bases if sn(b) < sn(t) and (b, t) in cell]
        if not prevs or F[t]["ln_ms"] is None:
            continue
        m0b = prevs[-1]
        withm = [b for b in prevs if F[b]["ln_ms"] is not None]
        ubp = min(withm, key=lambda b: (abs(F[b]["ln_ms"] - F[t]["ln_ms"]), -sn(b)))
        alls = [b for b in bases if b != t and (b, t) in cell and F[b]["ln_ms"] is not None]
        uba = min(alls, key=lambda b: (abs(F[b]["ln_ms"] - F[t]["ln_ms"]), -sn(b)))
        orc = min(prevs, key=lambda b: cell[(b, t)]["mean_abs_err"])
        rows_ub.append({"target": t, "m0_base": m0b, "m0": cell[(m0b, t)]["mean_abs_err"],
                        "ub_prev_base": ubp, "ub_prev": cell[(ubp, t)]["mean_abs_err"],
                        "ub_all_base": uba, "ub_all": cell[(uba, t)]["mean_abs_err"],
                        "oracle_prev_base": orc, "oracle_prev": cell[(orc, t)]["mean_abs_err"]})
    mean = lambda k: round(statistics.mean(r[k] for r in rows_ub), 5)
    ub = {"targets": [r["target"] for r in rows_ub], "m0": mean("m0"), "ub_prev": mean("ub_prev"), "ub_all": mean("ub_all"),
          "oracle_prev": mean("oracle_prev"), "rows": rows_ub}
    ub["gate_pass"] = ub["ub_prev"] < ub["m0"] - MARGIN
    ub["gate_rule"] = "상한(앞 full 시즌 중 끝난 뒤 ln m_s가 가장 가까운 한 시즌)이 M0(직전 시즌)보다 0.5%p 넘게 작아야 M2s를 후보로 둔다"
    return {"run": {"tool_head": d["run"]["tool_head"], "tools_dirty": d["run"]["tools_dirty"], "failures": len(d.get("failures", []))},
            "per_base": per_base, "distribution": dist, "explain": expl, "upper_bound": ub}, P


# ---------------- 고르기 ----------------
def select(C, crit2):
    """C: {이름: {"sv": summarize_v2 값, "rank": 튜플, "ref": late 비교 후보 이름}}."""
    names = list(C)
    kept = [n for n in names if C[n]["sv"]["main"]["cov80"] >= 0.70]
    step1 = "포함 0.70 이상"
    if not kept:
        best_cov = max(C[n]["sv"]["main"]["cov80"] for n in names)
        kept = [n for n in names if C[n]["sv"]["main"]["cov80"] == best_cov]
        step1 = "모두 0.70 미달 → 포함률이 가장 높은 것"
    mae = lambda n: C[n]["sv"]["main"]["mean_med_abs_err"]
    best = min(mae(n) for n in kept)
    near = sorted([n for n in kept if mae(n) <= best + MARGIN], key=lambda n: (C[n]["rank"], mae(n)))
    rest = sorted([n for n in kept if n not in near], key=mae)
    seq = near + rest
    tried = []
    pick = None
    for n in seq:
        lt = C[n]["sv"].get("late")
        why = []
        if lt and lt["cov80"] < 0.70:
            why.append(f"T_late 80% 포함 {lt['cov80']} < 0.70")
        ref = C[n]["ref"]
        if ref and lt and C.get(ref, {}).get("sv", {}).get("late"):
            d = lt["mean_med_abs_err"] - C[ref]["sv"]["late"]["mean_med_abs_err"]
            if d > LATE_TOL:
                why.append(f"T_late |e|가 {ref}보다 {d:.2%}p 나쁨")
        tried.append({"name": n, "late_guard": why})
        if not why:
            pick = n
            break
    return {"pick": pick, "step1": step1, "kept": kept, "best_mae": best, "within_0_5pp": near, "tried": tried,
            "table": [{"name": n, "rank": list(C[n]["rank"]), "cov80": C[n]["sv"]["main"]["cov80"], "mae": mae(n),
                       "b1": C[n]["sv"]["main"]["mean_med_abs_err_b1"] if not C[n].get("b1_base") else C[C[n]["b1_base"]]["sv"]["main"]["mean_med_abs_err_b1"],
                       "low": C[n]["sv"]["main"]["low_ratio"],
                       "season_cov_min": C[n]["sv"]["main"]["season_cov80_min"], "n_season_cov_lt50": C[n]["sv"]["main"]["n_season_cov80_lt50"],
                       "late_cov80": (C[n]["sv"].get("late") or {}).get("cov80"), "late_mae": (C[n]["sv"].get("late") or {}).get("mean_med_abs_err"),
                       "t12_cov80": (C[n]["sv"].get("t12") or {}).get("cov80"), "t12_mae": (C[n]["sv"].get("t12") or {}).get("mean_med_abs_err"),
                       "p10_cov80": (C[n]["sv"].get("p10_main") or {}).get("cov80"), "p10_mae": (C[n]["sv"].get("p10_main") or {}).get("mean_med_abs_err")}
                      for n in sorted(names, key=lambda n: C[n]["rank"])]}


def slim_sv(sv):
    """diag에 남길 요약 (줄 목록은 빼고)."""
    out = {}
    for k, v in sv.items():
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def build(cands, crit2, gate):
    """후보 22개 (+ M1) → C(고르기 후보), all_sv(요약 전부), factors(M1 배율)."""
    C = {}
    all_sv = {}
    factors = {}
    for name, c in cands.items():
        if c.get("comparison_only"):
            all_sv[name] = XV.summarize_v2([dict(r, scheme=name) for r in c["rows"]], crit2)[name]
            continue
        for m1_on in (False, True):
            rows = c["rows"]
            nm = name
            if m1_on:
                rows, fac = m1(rows)
                nm = name + "+M1"
                factors[nm] = fac
            sv = XV.summarize_v2([dict(r, scheme=nm) for r in rows], crit2)[nm]
            all_sv[nm] = sv
            if c["kind"] == "M2s" and not gate:
                continue   # 관문 미통과 — 후보에서 뺀다 (요약은 남긴다)
            # late_guard 비교 후보 (사전 고정): 같은 기준의 덧붙임 없는 후보, 기준만이면 바로 앞 기준
            if c["kind"] is None and not m1_on:
                ref = {"M0": None, "M2a": "M0", "M2b": "M2a"}[c["base"]]
            else:
                ref = c["base"]
            C[nm] = {"sv": sv, "rank": rank(c, m1_on), "ref": ref, "level": level_stats(c["rows"]) if c["kind"] else None,
                     "b1_base": c["base"] if c["kind"] else None}   # 표의 b1은 덧붙임 없는 같은 기준 값 (위 주석)
    return C, all_sv, factors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res")
    ap.add_argument("--out", default=os.path.join(AN, "diag43.json"))
    ap.add_argument("--seed2", help="같은 후보를 --seed 2로 낸 결과 폴더 (보고용 안정성 확인 — 판정은 사전 규칙대로 첫 폴더)")
    a = ap.parse_args()
    crit2 = jl(os.path.join(AN, "criteria_v2.json"))
    cands, meta = load_candidates(a.res)
    F = season_feats()
    targets = [f"S{n}" for n in range(21, 42)]
    M0p, P = m0prime(a.res, crit2, F, targets)
    gate = M0p["upper_bound"]["gate_pass"]
    print(f"M0′ 상한: M0 {M0p['upper_bound']['m0']:.2%} · 상한(앞) {M0p['upper_bound']['ub_prev']:.2%} · 상한(모두) {M0p['upper_bound']['ub_all']:.2%} "
          f"· 오라클(앞) {M0p['upper_bound']['oracle_prev']:.2%} → M2s 관문 {'통과' if gate else '미통과'}")
    print(f"M0′ 기준 시즌 분포: {M0p['distribution']}")
    print("M0′ 쌍 오차 설명:", M0p["explain"])
    C, all_sv, factors = build(cands, crit2, gate)
    sel = select(C, crit2)
    print("\n이름                 순서      포함   |e|    b1     낮게  시즌최저 <.5 | late 포함 |e| | T12 포함 |e| | 10% 포함 |e|")
    for t in sel["table"]:
        print(f"{t['name']:20s} {str(t['rank']):9s} {t['cov80']:.2f} {t['mae']:.2%} {t['b1']:.2%} {t['low']:.2f} {t['season_cov_min']:.2f} {t['n_season_cov_lt50']:2d} | "
              f"{t['late_cov80']:.2f} {t['late_mae']:.2%} | {t['t12_cov80']:.2f} {t['t12_mae']:.2%} | {t['p10_cov80']:.2f} {t['p10_mae']:.2%}")
    for n in ("loso",) + tuple(n for n in all_sv if n.startswith("M2b+M2s") and n not in C):
        if n in all_sv:
            v = all_sv[n]["main"]
            print(f"(비교) {n:15s} 포함 {v['cov80']:.2f} |e| {v['mean_med_abs_err']:.2%}")
    print(f"\n고르기: {sel['pick']} · {sel['step1']} · 0.5%p 안 {sel['within_0_5pp']} · 시도 {sel['tried']}")
    pick = sel["pick"]
    succ = None
    if pick:
        mv = C[pick]["sv"]["main"]
        # 기준선 1: predict가 b0·b1 중앙 계산에도 _lvl 수준 배수를 곱한다 (predict.Predictor.run, S42 때문에 고치지 않음) →
        # 수준 덧붙임 후보는 같은 기준의 덧붙임 없는 후보의 b1로 판정한다 (2026-10-18, 계산 뒤 찾은 지표 결함 — 고르기에는 b1을 쓰지 않는다)
        bname = cands[pick.replace("+M1", "")]["base"]
        b1 = C[bname]["sv"]["main"]["mean_med_abs_err_b1"]
        succ = {"coverage": mv["coverage"], "cov80": mv["cov80"], "beat_baseline1": mv["mean_med_abs_err"] < b1,
                "mae": mv["mean_med_abs_err"], "b1": b1, "b1_from": bname, "b1_raw_in_run": mv["mean_med_abs_err_b1"],
                "usable": mv["usable"], "usable_all": all(mv["usable"].values()),
                "grade_AB_cov80": mv["cov80_grade_AB"], "n_grade_AB": mv["n_grade_AB"],
                "note": "등급은 predict 등급 그대로 (M1로 넓힌 구간에 맞춰 다시 매기지 않음 — M5)"}
        print("성공 기준:", succ)
    out = {"about": "할 일 4.3 — 1·3% 모델 후보 고르기 (개발 검증, 23시즌 모두 개발). 사전 고정 커밋 e22da7e. 코드 tools/solo_raid/lab4/e_43.py",
           "inputs": meta, "targets": targets, "criteria_v2_sha1": next(iter(meta.values()))["criteria_v2_sha1"] if meta else None,
           "m0prime": M0p, "m2s_gate": gate,
           "candidates": {n: {"rank": list(v["rank"]), "late_ref": v["ref"], "level_applied": v["level"], "summary_v2": slim_sv(v["sv"])} for n, v in C.items()},
           "comparison": {n: slim_sv(v) for n, v in all_sv.items() if n not in C},
           "m1_factors": factors, "select": sel, "success_pick": succ}
    if a.seed2:   # 규칙 밖 확인 (2026-10-18): 시드만 바꿔 같은 규칙으로 고르면 무엇이 되나 — M2s 관문은 첫 폴더의 M0′ 그대로
        c2, meta2 = load_candidates(a.seed2)
        C2, _, _ = build(c2, crit2, gate)
        sel2 = select(C2, crit2)
        print(f"\n(시드 2) 고르기: {sel2['pick']} · 0.5%p 안 {sel2['within_0_5pp']} · 시도 {sel2['tried']}")
        for t in sel2["table"]:
            print(f"  {t['name']:20s} {t['cov80']:.2f} {t['mae']:.2%} | late {t['late_cov80']:.2f} {t['late_mae']:.2%}")
        out["seed2_check"] = {"what": "보고용 — 시드 2(뽑기·붓스트랩)로 같은 후보·같은 규칙. 판정은 시드 1(사전 고정)", "inputs": meta2, "select": sel2}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("→", a.out)


if __name__ == "__main__":
    main()
