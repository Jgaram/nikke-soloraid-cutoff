"""WP9a 진단 1 — 사전만 대 자료 넣음 (H2 과신). docs/solo-raid-method.md 「WP9a 진단」 1번.

M3 `dt`의 앞 시즌 사전만으로, 이번 시즌 자료 없이 낸 최종 한섭 컷 C(q) 예측분포를 22시즌(S21~S42) 모두 만들고
(예측 파일 `pred_prior_D1.json`, method "prior"), 같은 칸에서 M3 `dt`(pred_M3dt_D1.json)·F(F_D1.json)와 견준다.
정답 표(answer_key)는 채점(score.py 함수)에서만 쓴다 — 사전 만들기·뽑기에는 안 쓴다.

  python3 -I tools/solo_raid/lab7/wp9_diag1.py priors --work <임시 폴더>     # 1) 사전 만들기 (m3.prior — 2단계 MCMC, 몇 시간 아닌 분 단위) + 재현 확인
  python3 -I tools/solo_raid/lab7/wp9_diag1.py report --work <임시 폴더>     # 2) 뽑기 -> 예측 파일 -> 견주기 -> json·md

사전 = m3.prior()가 만드는 것과 같다 (predict가 씀): a ~ N(a.mean, a.var) · ξ_k ~ N(0, lambda_scaled_k) · 곡선 계수 = mean_coef + Σ ξ_k phi_coef_k ·
nuisance(lnsig0·lnrho·beta·lnsigd·lnelld·서버 6) ~ MVN(mean, cov + 1e-6 I) (CellProb와 같음) · 시간 어긋남 ζ ~ N(0, I).
`dt`의 목표: 최종 한섭 q% 컷 = exp(lnC(θ, q·10^{-δ(H)})) (m3._lnC_t) — δ(H)는 비중심 정상 AR(1) (3시간 매듭, σ_δ=exp(lnsigd), φ=exp(-3/exp(lnelld))) 의
끝 시각 H 값(왼쪽·오른쪽 매듭 선형 보간). 그래서 σ_δ·ℓ_δ의 사전 분포와 δ(H)의 분산이 사전만 예측에 들어간다 (dep_fields와 같은 셈, numpy).
입력 읽기만: r7/wp5a/stage1_M3dt_D1.json · r7/pred_M3dt_D1.json · r7/F_D1.json · r7/lb_D1.json.
"""
import argparse
import collections
import json
import math
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
import m3 as M3  # noqa: E402
import hier as HR  # noqa: E402
import score as SC  # noqa: E402
import common as C  # noqa: E402

R7 = os.path.join(ROOT, "data", "solo-raid", "analysis", "r7")
OUTD = os.path.join(R7, "wp9")
STAGE1 = os.path.join(R7, "wp5a", "stage1_M3dt_D1.json")
M3PRED = os.path.join(R7, "pred_M3dt_D1.json")
FPRED = os.path.join(R7, "F_D1.json")
LBF = os.path.join(R7, "lb_D1.json")
TARGETS = [f"S{n}" for n in range(21, 43)]
TUNE = SC.TUNE
REST = [s for s in TARGETS if s not in TUNE]
QS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
NDRAW = 20000
SEEDBASE = 20261102
SETS = {"22시즌": TARGETS, "조정6": list(TUNE), "나머지16": REST}
GROUPS = {"1·3%": (1.0, 3.0), "10%": (10.0,)}


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def js(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------- 1) 사전 ----------

def cmd_priors(a):
    import obs_model as OM
    OM.setup_host(4)                                       # m3.cmd_prior와 같다 (hier 2단계 적합 4사슬)
    s1 = M3.load_stage1(STAGE1)
    assert M3.DEP == "dt", M3.DEP
    pdir = os.path.join(a.work, "priors")
    os.makedirs(pdir, exist_ok=True)
    for s in TARGETS:
        path = os.path.join(pdir, f"{s}.json")
        if os.path.exists(path):
            continue
        p = M3.prior(s1, s, cache=os.path.join(a.work, "pc"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(p, f, default=js)
        print(s, "k", p["k"], "J", p["J"], "K", p["K"], flush=True)


def repro_diff(pr, saved):
    """만든 사전 pr과 pred_M3dt_D1.json priors[s]의 최대 절대차 (JSON 왕복한 pr)."""
    d = {"k_same": pr["k"] == saved["k"], "window_same": pr["window"] == saved["window"], "K_same": pr["K"] == saved["K"], "names_same": pr["nuisance"]["names"] == saved["nuisance"]["names"]}
    A = lambda x, y: float(np.abs(np.asarray(x, float) - np.asarray(y, float)).max()) if np.size(x) else 0.0   # noqa: E731
    d["mean_coef"] = A(pr["mean_coef"], saved["mean_coef"])
    d["phi_coef"] = A(pr["phi_coef"], saved["phi_coef"])
    d["lambda_scaled"] = A(pr["lambda_scaled"], saved["lambda_scaled"])
    d["sigma2_res"] = abs(pr["sigma2_res"] - saved["sigma2_res"])
    d["a_mean"] = abs(pr["a"]["mean"] - saved["a"]["mean"])
    d["a_var"] = abs(pr["a"]["var"] - saved["a"]["var"])
    d["nuis_mean"] = A(pr["nuisance"]["mean"], saved["nuisance"]["mean"])
    d["nuis_cov"] = A(pr["nuisance"]["cov"], saved["nuisance"]["cov"])
    d["max_all"] = max(d[k] for k in ("mean_coef", "phi_coef", "lambda_scaled", "sigma2_res", "a_mean", "a_var", "nuis_mean", "nuis_cov"))
    return d


# ---------- 2) 뽑기 ----------

def ar_dH(lnsigd, lnelld, zeta, H):
    """비중심 정상 AR(1) 매듭 δ (m3._ar_mat/dep_fields와 같은 셈, numpy) -> 끝 시각 H의 값 dH (log10 %). lnsigd·lnelld [N], zeta [N, nk]."""
    nk = int(math.floor(H / M3.DEP_STEP)) + 2
    jH = int(math.floor(H / M3.DEP_STEP))
    wH = H / M3.DEP_STEP - jH
    sig, ell = np.exp(lnsigd), np.exp(lnelld)
    phi = np.exp(-M3.DEP_STEP / ell)
    s = np.sqrt(1 - phi ** 2)
    d = sig * zeta[:, 0]
    dl = [d]
    for j in range(1, nk):
        d = phi * d + sig * s * zeta[:, j]
        dl.append(d)
    delta = np.stack(dl, 1)
    return delta[:, jH] * (1 - wH) + delta[:, jH + 1] * wH, nk


def lnC_draws(a, cr, dH):
    """a[N], cr[N,NR], dH[N] -> ln C(q)[N, 6] (m3.lnC, q_i = q·10^{-dH_i}, m3._lnC_t와 같다)."""
    import jax
    import jax.numpy as jnp
    qs = jnp.asarray(QS, jnp.float32)

    def f(aa, cc, dd):
        return M3.lnC({"a": aa, "cr": cc}, qs * jnp.power(10.0, -dd))
    return np.asarray(jax.vmap(f)(jnp.asarray(a, jnp.float32), jnp.asarray(cr, jnp.float32), jnp.asarray(dH, jnp.float32)), float)


def prior_draws(pr, H, seed, with_dep=True, N=None):
    rng = np.random.default_rng(seed)
    N, K = N or NDRAW, pr["K"]
    NR = M3.NR
    a = pr["a"]["mean"] + math.sqrt(pr["a"]["var"]) * rng.standard_normal(N)
    mc = np.asarray(pr["mean_coef"], float)[:NR]
    phi = np.asarray(pr["phi_coef"], float).reshape(K, -1)[:, :NR] if K else np.zeros((0, NR))
    lam = np.asarray(pr["lambda_scaled"], float)[:K]
    xi = rng.standard_normal((N, K)) * np.sqrt(lam)
    cr = mc + xi @ phi
    nu = pr["nuisance"]
    names = nu["names"]
    cov = np.asarray(nu["cov"], float) + 1e-6 * np.eye(len(names))
    nd = np.asarray(nu["mean"], float) + rng.standard_normal((N, len(names))) @ np.linalg.cholesky(cov).T
    nk = int(math.floor(H / M3.DEP_STEP)) + 2
    zeta = rng.standard_normal((N, nk))
    lsd, lel = nd[:, names.index("lnsigd")], nd[:, names.index("lnelld")]
    dH, _ = ar_dH(lsd, lel, zeta, H)
    if not with_dep:
        dH = np.zeros(N)
    return a, cr, dH, lsd, lel


def make_cells(s, L):
    """L[N,6] ln C -> 칸 목록 (T는 아직 안 붙임)."""
    out = []
    for i, q in enumerate(QS):
        lo80, med, hi80, lo95, hi95 = np.quantile(L[:, i], [0.1, 0.5, 0.9, 0.025, 0.975])
        out.append((q, float(math.exp(med)), [float(math.exp(lo80)), float(math.exp(hi80))], [float(math.exp(lo95)), float(math.exp(hi95))]))
    return out


def noise_scale(priors, seasons, dist, nb=60, nd=2000):
    """각 시즌 사전에서 nd뽑기 배치를 nb번 뽑아 중앙·80% 아래끝·위끝(ln -> %)의 배치 간 표준편차 = 표준오차 눈금. -> {시즌: {%: {med, lo, hi}}}"""
    out = {}
    for s in seasons:
        H = dist[s]["H"]
        acc = []
        for b in range(nb):
            aa, cr, dH, _, _ = prior_draws(priors[s], H, 777000 + 1000 * b + HR.snum(s), True, N=nd)
            L = lnC_draws(aa, cr, dH)
            acc.append(np.quantile(L, [0.5, 0.1, 0.9], axis=0))       # [3, 6]
        acc = np.array(acc)                                              # [nb, 3, 6]
        sd = acc.std(0, ddof=1) * 100.0
        out[s] = {f"{q:g}": {"med": float(sd[0, i]), "lo": float(sd[1, i]), "hi": float(sd[2, i])} for i, q in enumerate(QS)}
        print("noise", s, flush=True)
    return out


def fmt(x, d=2):
    return "-" if x is None else f"{x * 100:.{d}f}"


def cmd_report(a):
    import jax
    jax.config.update("jax_platform_name", "cpu")
    M3.load_stage1(STAGE1)                                  # 기저·dep 설정
    saved = jl(M3PRED)
    pdir = os.path.join(a.work, "priors")
    priors = {s: jl(os.path.join(pdir, f"{s}.json")) for s in TARGETS}
    # ---- 재현 확인 1
    repro = {s: repro_diff(priors[s], saved["priors"][s]) for s in TARGETS}
    worst = max(repro.values(), key=lambda d: d["max_all"])
    repro_ok = all(d["k_same"] and d["window_same"] and d["K_same"] and d["names_same"] for d in repro.values())
    print("재현 최대 절대차", worst["max_all"], "구조 같음", repro_ok, flush=True)
    # ---- 사전만 예측 (22시즌, T 무관)
    m3cells = saved["cells"]
    grid = collections.defaultdict(set)
    for c in m3cells:
        grid[c["season"]].add(c["T"])
    cells, dist = [], {}
    for s in TARGETS:
        H = float(C.info(s)[1]) if hasattr(C, "info") else float(SC.A.info(s)[1])
        pr = priors[s]
        sd = SEEDBASE + 100 * HR.snum(s)
        aa, cr, dH, lsd, lel = prior_draws(pr, H, sd, True)
        L = lnC_draws(aa, cr, dH)
        L0 = lnC_draws(aa, cr, np.zeros_like(dH))          # δ 없이 (비교용)
        base = make_cells(s, L)
        base0 = make_cells(s, L0)
        dist[s] = {"H": H, "sigma_d_med": float(np.exp(np.median(lsd))), "ell_d_med": float(np.exp(np.median(lel))), "dH_sd": float(dH.std()),
                   "dH_q10_90": [float(x) for x in np.quantile(dH, [0.1, 0.9])],
                   "ln_width80_with_dH": {f"{q:g}": float(np.log(b[2][1] / b[2][0])) for q, b in zip(QS, [(x[0], x[1], x[2]) for x in base])},
                   "ln_width80_without_dH": {f"{q:g}": float(np.log(b[2][1] / b[2][0])) for q, b in zip(QS, [(x[0], x[1], x[2]) for x in base0])},
                   "pred": {f"{q:g}": p for q, p, _, _ in base}, "pred_no_dH": {f"{q:g}": p for q, p, _, _ in base0}}
        for T in sorted(grid[s]):
            for q, p, i80, i95 in base:
                cells.append({"season": s, "T": float(T), "pct": float(q), "server": "KR", "pred": p, "i80": i80, "i95": i95, "own": True})
        print(s, "H", H, "dH sd", round(dist[s]["dH_sd"], 4), flush=True)
    os.makedirs(OUTD, exist_ok=True)
    pred = {"about": "사전만 예측 — M3 dt의 앞 시즌 사전(stage1_M3dt_D1.json)만으로, 이번 시즌 자료 없이 낸 최종 한섭 컷 (WP9a 진단 1, wp9_diag1.py). "
                     f"시즌마다 {NDRAW}뽑기·고정 난수, 목표에 δ(H) 들어감. T에 무관해서 같은 값을 T마다 복사 (pred_M3dt_D1.json과 같은 (시즌,T,%) 칸).",
            "method": "prior", "data_ver": saved["data_ver"],
            "made": {"tool_head": M3._git_head(), "cmd": "wp9_diag1.py priors + report", "inputs": {"stage1": SC.sha12(STAGE1), "pred_M3dt": SC.sha12(M3PRED)}},
            "cells": cells}
    pred_path = os.path.join(OUTD, "pred_prior_D1.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False)
    # ---- 재현 확인 2: n == 0 칸
    n0 = [(i["season"], i["T"]) for i in saved["info"] if i["n"] == 0]
    byk = {(c["season"], c["T"], c["pct"]): c for c in saved["cells"]}
    pk = {(c["season"], c["T"], c["pct"]): c for c in cells}
    n0rows = []
    for (s, T) in n0:
        row = {"season": s, "T": T, "conv": [i["conv"] for i in saved["info"] if i["season"] == s and i["T"] == T][0], "by_q": {}}
        for q in QS:
            m, p = byk[(s, T, q)], pk[(s, T, q)]
            row["by_q"][f"{q:g}"] = {"med_diff_pct": (m["pred"] / p["pred"] - 1) * 100, "lo80_diff_pct": (m["i80"][0] / p["i80"][0] - 1) * 100,
                                     "hi80_diff_pct": (m["i80"][1] / p["i80"][1] - 1) * 100, "width80_ratio": math.log(m["i80"][1] / m["i80"][0]) / math.log(p["i80"][1] / p["i80"][0])}
        n0rows.append(row)
    # ---- 재현 확인 2의 잡음 눈금: 같은 사전에서 2,000뽑기(= NUTS 칸의 뽑기 수) 배치를 여러 번 뽑았을 때 중앙·80% 끝의 표준오차 (%)
    se = noise_scale(priors, sorted({r["season"] for r in n0rows}), dist)
    for r in n0rows:
        for q, b in r["by_q"].items():
            e = se[r["season"]][q]
            b["z_med"], b["z_lo"], b["z_hi"] = b["med_diff_pct"] / e["med"], b["lo80_diff_pct"] / e["lo"], b["hi80_diff_pct"] / e["hi"]
    # S27: 합친 파일의 priors[S27]는 뒤 조각(x102·x114)의 것이다 — 앞 조각(T 6~96)·조각 파일과 따로 비교
    parts_s27 = {}
    for fn in ("pred_M3dt_D1_S27.json", "pred_M3dt_D1_S27x102.json", "pred_M3dt_D1_S27x114.json"):
        pp = os.path.join(R7, "wp5b", "parts", fn)
        if os.path.exists(pp):
            parts_s27[fn] = repro_diff(priors["S27"], jl(pp)["priors"]["S27"])["max_all"]
    # ---- 견주기 (같은 칸)
    F = jl(FPRED)
    keys = {(c["season"], c["T"], c["pct"]) for c in saved["cells"]}
    Fcells = [c for c in F["cells"] if c.get("server", "KR") == "KR" and (c["season"], float(c["T"]), float(c["pct"])) in keys]
    Fpath = os.path.join(a.work, "F_D1_same_cells.json")
    with open(Fpath, "w", encoding="utf-8") as f:
        json.dump(dict(F, cells=Fcells, about=F["about"] + " [M3dt 444칸으로 거름]"), f, ensure_ascii=False)
    lbh, lb = SC.load_lb(LBF)
    truth = SC.A.truth()
    meth = {"prior": cells, "M3dt": saved["cells"], "F": Fcells}
    res = {}
    for sn, ss in SETS.items():
        res[sn] = {}
        for gn, qs in GROUPS.items():
            res[sn][gn] = {m: SC.evaluate(cs, truth, lb, qs, SC.TOL, seasons=set(ss)) for m, cs in meth.items()}
    # 시즌별 초반 창 |e| (1·3%, 10%) : prior vs M3
    perseason = {}
    for gn, qs in GROUPS.items():
        perseason[gn] = {}
        for s in TARGETS:
            perseason[gn][s] = {m: SC.evaluate(cs, truth, lb, qs, SC.TOL, seasons={s})["main"] for m, cs in meth.items()}
    # 관측 줄 수(info n) 묶음별 — 창 밖 T도 포함한 모든 칸
    nmap = {(i["season"], i["T"]): i["n"] for i in saved["info"]}
    bins = [("0", 0, 0), ("1~4", 1, 4), ("5~19", 5, 19), ("20~99", 20, 99), ("100+", 100, 10 ** 9)]
    nbin = {}
    for gn, qs in GROUPS.items():
        nbin[gn] = {}
        for bn, lo_, hi_ in bins:
            row = {}
            for m, cs in meth.items():
                es, cv, n_s = [], [], set()
                for c in cs:
                    if c["pct"] not in qs or not (lo_ <= nmap[(c["season"], c["T"])] <= hi_):
                        continue
                    tv = truth.get((c["season"], "KR", c["pct"]))
                    if tv is None:
                        continue
                    es.append(c["pred"] / tv - 1)
                    cv.append(c["i80"][0] <= tv <= c["i80"][1])
                    n_s.add((c["season"], c["T"]))
                row[m] = {"n": len(es), "mae": float(np.mean(np.abs(es))) if es else None, "bias": float(np.mean(es)) if es else None, "cov": float(np.mean(cv)) if cv else None, "st": len(n_s)}
            nbin[gn][bn] = row
    # 시즌별 초반 창에서 사전만이 M3보다 낮은 시즌 수
    wins = {}
    for gn in GROUPS:
        wins[gn] = {}
        for sn, ss in SETS.items():
            v = [(perseason[gn][s]["prior"], perseason[gn][s]["M3dt"], perseason[gn][s]["F"]) for s in ss if perseason[gn][s]["prior"] is not None]
            wins[gn][sn] = {"n": len(v), "prior_lt_M3": sum(1 for p_, m_, f_ in v if p_ < m_), "prior_lt_F": sum(1 for p_, m_, f_ in v if p_ < f_), "M3_lt_F": sum(1 for p_, m_, f_ in v if m_ < f_)}
    out = {"about": "WP9a 진단 1 — 사전만 대 M3 dt 대 F, 같은 444칸(시즌×T)×6%. 사전만: M3 dt의 앞 시즌 사전만, 이번 시즌 자료 없음.",
           "ndraw": NDRAW, "files": {"stage1": SC.sha12(STAGE1), "pred_M3dt": SC.sha12(M3PRED), "F": SC.sha12(FPRED), "pred_prior": SC.sha12(pred_path)},
           "repro1": {"all_structure_same": repro_ok, "max_abs_diff": worst["max_all"], "worst_season": [s for s, d in repro.items() if d is worst][0], "per_season": repro},
           "repro1_S27_vs_parts": parts_s27, "repro2_n0_cells": n0rows, "noise_se_pct": se, "by_n_lines": nbin, "season_wins_main": wins, "prior_dist": dist, "result": {}, "perseason_main": perseason}
    for sn in SETS:
        out["result"][sn] = {}
        for gn in GROUPS:
            out["result"][sn][gn] = {}
            for m in meth:
                r = res[sn][gn][m]
                out["result"][sn][gn][m] = {k: r[k] for k in ("main", "end", "score", "cov_main", "cov_end", "g1", "crps", "crps_main", "crps_end", "w80", "n")}
                out["result"][sn][gn][m]["byT"] = r["byT"]
    with open(os.path.join(OUTD, "diag1_D1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=js)
    with open(os.path.join(OUTD, "diag1_D1.md"), "w", encoding="utf-8") as f:
        f.write(make_md(out, meth, res, repro, worst, n0rows, dist, a.summary))
    print("wrote", OUTD, flush=True)


def tsort(k):
    if k.startswith("H"):
        return (1, 0 if k == "H-12" else (1 if k == "H-6" else 2))
    return (0, float(k))


def make_md(o, meth, res, repro, worst, n0rows, dist, summary_path=None):
    R = o["result"]
    L = []
    g = lambda sn, gn, m, k: R[sn][gn][m][k]   # noqa: E731
    r16, r22 = "나머지16", "22시즌"
    p13, m13 = g(r16, "1·3%", "prior", "main"), g(r16, "1·3%", "M3dt", "main")
    p13a, m13a = g(r22, "1·3%", "prior", "main"), g(r22, "1·3%", "M3dt", "main")
    p10, m10 = g(r16, "10%", "prior", "main"), g(r16, "10%", "M3dt", "main")
    p10a, m10a = g(r22, "10%", "prior", "main"), g(r22, "10%", "M3dt", "main")
    L.append("# WP9a 진단 1 — 사전만 대 자료 넣음 (H2 과신), 자료 판 D1\n")
    L.append(open(summary_path, encoding='utf-8').read().rstrip() if summary_path else SUMMARY_PLACEHOLDER)
    L.append("")
    L.append("## 정의\n")
    L.append("- **사전만 예측 `prior`** = M3 `dt`의 앞 시즌 사전만으로, 이번 시즌 자료 없이 낸 최종 한섭 컷 C(q)의 예측분포. 맞힐 시즌 s마다 `m3.prior(stage1_M3dt_D1, s)` (predict가 쓰는 것과 같은 길 — 창 k 고르기, 주성분, 시즌 사이 층 MVN, a의 국소 수준)로 만든 사전에서 "
             f"{NDRAW:,}뽑기(고정 난수): a ~ N(평균, 분산), ξ ~ N(0, λ), nuisance(σ_δ·ℓ_δ 포함) ~ MVN, 시간 어긋남 ζ ~ N(0, I). **`dt`의 목표 그대로 최종 한섭 q% 컷 = C(q·10^(−δ(H)))** 라서 δ(H)의 사전 분산(σ_δ·ℓ_δ 사전에서 옴)이 들어 있다. 중앙 = 뽑기 중앙, 80% = 10~90%, 95% = 2.5~97.5%.")
    L.append("- 예측 파일 `pred_prior_D1.json`은 `pred_M3dt_D1.json`과 같은 (시즌, T, %) 2,664칸. 사전만은 T에 무관해서 같은 값을 T마다 복사했다.")
    L.append("- **M3 `dt`** = `r7/pred_M3dt_D1.json` (WP5b, 칸 노드 K 32, NUTS). **F** = `r7/F_D1.json`을 같은 칸으로 거른 것.")
    L.append("- |e| = |중앙 ÷ 정답 − 1|. 초반 창 = T 18·24·36·48, 막판 창 = H−12·H−6·H (score.py `evaluate`와 같은 함수·정답). 칸 평균. 80% 포함 = 정답이 80% 구간 안인 칸 비율. 1·3% = 1%와 3% 두 목표의 칸 합, 10%는 따로.")
    L.append("- 22시즌 = S21~S42, 조정 6 = S22·S26·S30·S34·S38·S42, 나머지 16 = 그 밖.\n")

    # repro
    L.append("## 재현 확인\n")
    L.append(f"**① 만든 사전 = 저장된 사전?** `pred_M3dt_D1.json` `priors[s]`와 22시즌 모두 비교. 창 k·창 시즌·K·nuisance 이름이 22시즌 모두 {'같다' if o['repro1']['all_structure_same'] else '다르다(!)'}. "
             f"모수 최대 절대차(mean_coef·phi_coef·lambda_scaled·sigma2_res·a 평균·분산·nuisance 평균·공분산 전체 중 최대) = **{worst['max_all']:.2e}** (시즌 {o['repro1']['worst_season']}).\n")
    L.append("| 시즌 | k | K | mean_coef | phi_coef | lambda_scaled | σ²_res | a 평균 | a 분산 | nuis 평균 | nuis 공분산 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for s in TARGETS:
        d = repro[s]
        L.append(f"| {s} | {o['prior_dist'][s].get('k', '')} |  | {d['mean_coef']:.1e} | {d['phi_coef']:.1e} | {d['lambda_scaled']:.1e} | {d['sigma2_res']:.1e} | {d['a_mean']:.1e} | {d['a_var']:.1e} | {d['nuis_mean']:.1e} | {d['nuis_cov']:.1e} |")
    L.append("")
    L.append("S27 안내: 합친 파일의 priors[S27]은 뒤 조각(T 102~120 `x102`·`x114`, 다른 컨테이너)의 것이라 앞 조각(T 6~96)과 nuisance가 0.008 다르다 — 같은 입력의 2단계 MCMC 잡음. "
             f"내가 만든 S27 사전과 조각 파일별 최대 절대차: " + ", ".join(f"`{k}` {v:.1e}" for k, v in o["repro1_S27_vs_parts"].items()) + ". 나머지 21시즌은 0 (바이트 수준으로 같음).\n")
    L.append("**② 관측 0줄 칸 (info n == 0) — M3 `dt` 칸 대 사전만.** 이 칸의 M3 예측은 사실상 사전만이다. 차 = M3 ÷ 사전만 − 1 (%). 폭비 = M3 80% ln 폭 ÷ 사전만 80% ln 폭.\n")
    L.append("| 시즌 T | 수렴 | % | 중앙 차 | 80% 아래끝 차 | 80% 위끝 차 | 폭비 | z(중앙) | z(아래끝) | z(위끝) |\n|---|---|---|---|---|---|---|---|---|---|")
    allmed, allend, allw = [], [], []
    for r in n0rows:
        for q in ("1", "3", "10"):
            b = r["by_q"][q]
            L.append(f"| {r['season']} T{r['T']:g} | {'o' if r['conv'] else 'x'} | {q} | {b['med_diff_pct']:+.2f} | {b['lo80_diff_pct']:+.2f} | {b['hi80_diff_pct']:+.2f} | {b['width80_ratio']:.2f} | {b['z_med']:+.1f} | {b['z_lo']:+.1f} | {b['z_hi']:+.1f} |")
        for q, b in r["by_q"].items():
            allmed.append(abs(b["med_diff_pct"]))
            allend += [abs(b["lo80_diff_pct"]), abs(b["hi80_diff_pct"])]
            allw.append(b["width80_ratio"])
    zs = [abs(b[k]) for r in n0rows for b in r["by_q"].values() for k in ("z_med", "z_lo", "z_hi")]
    L.append(f"\nz = 차 ÷ (같은 사전에서 2,000뽑기 배치를 60번 뽑은 표준오차). 13칸 중 S22·S24는 T6과 T12가 관측 0줄이라 같은 NUTS 결과를 두 번 센 것이다(독립 11칸). |z| 평균 {np.mean(zs):.2f} · 중앙 {np.median(zs):.2f} · 최대 {np.max(zs):.1f} (잡음뿐이면 평균 약 0.8, 최대 3 안팎). ")
    L.append(f"\n모든 {len(n0rows)}칸 × 6% 요약: 중앙 차 |평균| {np.mean(allmed):.2f}%·최대 {np.max(allmed):.2f}% · 80% 끝 차 |평균| {np.mean(allend):.2f}%·최대 {np.max(allend):.2f}% · 폭비 중앙 {np.median(allw):.2f} (범위 {np.min(allw):.2f}~{np.max(allw):.2f}).\n")

    L.append("## 사전만 분포의 시간 어긋남 항\n")
    L.append("| 시즌 | H | σ_δ 사전 중앙 | ℓ_δ 사전 중앙(h) | δ(H) sd (log10 %) | 3% 80% ln 폭 (δ 넣음) | (δ 뺌) | 1% (넣음) | (뺌) |\n|---|---|---|---|---|---|---|---|---|")
    for s in TARGETS:
        d = dist[s]
        L.append(f"| {s} | {d['H']:g} | {d['sigma_d_med']:.4f} | {d['ell_d_med']:.2f} | {d['dH_sd']:.4f} | {d['ln_width80_with_dH']['3']:.3f} | {d['ln_width80_without_dH']['3']:.3f} | {d['ln_width80_with_dH']['1']:.3f} | {d['ln_width80_without_dH']['1']:.3f} |")
    L.append("")

    # 견주기
    def block(gn):
        L.append(f"## 견주기 — {gn}\n")
        L.append("| 시즌 묶음 | 방법 | 초반 창 \\|e\\| % | 막판 창 \\|e\\| % | 80% 포함 초반 | 80% 포함 막판 | 80% 폭 중앙 초반 · 막판 % | CRPS 초반 · 막판 % | 칸 초반/막판 |\n|---|---|---|---|---|---|---|---|---|")
        for sn in SETS:
            for m in ("prior", "M3dt", "F"):
                r = R[sn][gn][m]
                L.append(f"| {sn} | {m} | {fmt(r['main'])} | {fmt(r['end'])} | {r['cov_main']:.2f} | {r['cov_end']:.2f} | {fmt(r['w80']['main'], 1)} · {fmt(r['w80']['end'], 1)} | {fmt(r['crps_main'])} · {fmt(r['crps_end'])} | {r['n']['main']}/{r['n']['end']} |")
        L.append("")
        for sn in SETS:
            L.append(f"### T별 \\|e\\| 평균 % · (80% 포함) — {gn}, {sn}\n")
            keys = sorted({k for m in ("prior", "M3dt", "F") for k in R[sn][gn][m]["byT"]}, key=tsort)
            L.append("| T | 칸 | 사전만 | M3 dt | F | 사전만 − M3 (%p) | 사전만 편향 | M3 편향 |\n|---|---|---|---|---|---|---|---|")
            for k in keys:
                row = {m: R[sn][gn][m]["byT"].get(k) for m in ("prior", "M3dt", "F")}
                if not all(row.values()):
                    continue
                pm, mm, fm_ = row["prior"], row["M3dt"], row["F"]
                L.append(f"| {k} | {pm['n']} | {pm['mae'] * 100:.2f} ({pm['cov']:.2f}) | {mm['mae'] * 100:.2f} ({mm['cov']:.2f}) | {fm_['mae'] * 100:.2f} ({fm_['cov']:.2f}) | {(pm['mae'] - mm['mae']) * 100:+.2f} | {pm['bias'] * 100:+.1f} | {mm['bias'] * 100:+.1f} |")
            L.append("")
    block("1·3%")
    block("10%")

    L.append("## 관측 줄 수(info n)별 — 창 밖 T 포함, 모든 칸\n")
    L.append("칸 = (시즌, T, %). 줄 수가 적은 칸에서 자료를 넣은 M3가 사전만보다 나쁜지 본다 (과신의 직접 증거).\n")
    L.append("| % | 줄 수 | (시즌,T) | 칸 | 사전만 \\|e\\| (편향) | M3 dt \\|e\\| (편향) | F \\|e\\| | 80% 포함 사전만 · M3 · F |\n|---|---|---|---|---|---|---|---|")
    for gn in GROUPS:
        for bn, row in o["by_n_lines"][gn].items():
            if not row["prior"]["n"]:
                continue
            L.append(f"| {gn} | {bn} | {row['prior']['st']} | {row['prior']['n']} | {fmt(row['prior']['mae'])} ({row['prior']['bias'] * 100:+.1f}) | {fmt(row['M3dt']['mae'])} ({row['M3dt']['bias'] * 100:+.1f}) | {fmt(row['F']['mae'])} | {row['prior']['cov']:.2f} · {row['M3dt']['cov']:.2f} · {row['F']['cov']:.2f} |")
    L.append("")
    L.append("## 초반 창에서 시즌별로 사전만이 낮은 시즌 수\n")
    L.append("| % | 묶음 | 시즌 | 사전만 < M3 | 사전만 < F | M3 < F |\n|---|---|---|---|---|---|")
    for gn in GROUPS:
        for sn in SETS:
            w_ = o["season_wins_main"][gn][sn]
            L.append(f"| {gn} | {sn} | {w_['n']} | {w_['prior_lt_M3']} | {w_['prior_lt_F']} | {w_['M3_lt_F']} |")
    L.append("")
    L.append("## 시즌별 초반 창 \\|e\\| % (사전만 / M3 dt / F)\n")
    L.append("| 시즌 | 1·3% 사전만 | M3 dt | F | 10% 사전만 | M3 dt | F |\n|---|---|---|---|---|---|---|")
    ps = o["perseason_main"]
    for s in TARGETS:
        a_, b_ = ps["1·3%"][s], ps["10%"][s]
        L.append(f"| {s}{'*' if s in TUNE else ''} | {fmt(a_['prior'])} | {fmt(a_['M3dt'])} | {fmt(a_['F'])} | {fmt(b_['prior'])} | {fmt(b_['M3dt'])} | {fmt(b_['F'])} |")
    L.append("\n(* = 조정 시즌)")
    return "\n".join(L) + "\n"


SUMMARY_PLACEHOLDER = "@@SUMMARY@@"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("priors", "report"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--summary", default=None, help="md 맨 위 요약 파일 (손으로 쓴 5줄)")
    a = ap.parse_args()
    {"priors": cmd_priors, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
