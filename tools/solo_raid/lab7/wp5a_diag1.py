"""WP5a 진단 1 — M3 2단계(시즌 사이 층) 사전이 「잠재 참값」을 얼마나 덮나 (docs/solo-raid-method.md 「WP3b M3 결과」: 80% 구간이 너무 좁다).
새 MCMC 적합 없음. 정답 표(answer_key)는 읽지 않는다. 앞 시즌 -> 다음 시즌 순서로만.

  python3 -I tools/solo_raid/lab7/wp5a_diag1.py

입력 (읽기만): r7/stage1_M3_D1.json (잠재 참값 = lnC_q10_50_90 중앙, 사전 = 앞 시즌의 1단계 평균·공분산),
              r7/pred_M3_tune_D1.json (조정 6시즌 사전 — 재현 확인에만).
출력: r7/wp5a/diag1_D1.json, r7/wp5a/diag1_D1.md   (숫자만)

사전 = m3.prior()에서 nuisance MVN 적합(MCMC)만 뺀 것: m3.choose_window_pca -> m3.pca_window -> 최소제곱으로 계수 (mean_coef·phi_coef),
a 사전 = hier.fit_level + hier.level_prior (hier.prior_for의 a 부분). ln C = a - R(w(q)) 는 m3.lnC.
변형 (같은 난수): base / t / ppca / fullcov — 정의는 md 머리말.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
import m3 as M3  # noqa: E402
import hier as HR  # noqa: E402

R7 = os.path.join(ROOT, "data", "solo-raid", "analysis", "r7")
OUTD = os.path.join(R7, "wp5a")
TARGETS = [f"S{n}" for n in range(21, 43)]
TUNE = ["S22", "S26", "S30", "S34", "S38", "S42"]
QS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
NDRAW = 4000
VARS = ["base", "t", "ppca", "fullcov"]
GROUPS = {"1·3%": [1.0, 3.0], "10%": [10.0], "전체": QS, "0.5·2·5%": [0.5, 2.0, 5.0]}
SHAPE_GROUPS = {"1%": [1.0], "10%": [10.0], "전체(3% 뺌)": [0.5, 1.0, 2.0, 5.0, 10.0]}
SETS = {"전체22": TARGETS, "조정6": TUNE, "나머지16": [s for s in TARGETS if s not in TUNE]}


def qkey(q):
    return f"{q:g}"


# ---------- 사전 (m3.prior()에서 MCMC만 뺀 것) ----------

def build_prior(stage1, target):
    before = [s for s in sorted(stage1, key=HR.snum) if HR.snum(s) < HR.snum(target)]
    k, scores = M3.choose_window_pca(stage1, target)
    win = before if k == "all" else before[-int(k):]
    xv = [M3._xv(stage1[s]) for s in win]
    X = np.array([x for x, _ in xv])
    V = np.array([v for _, v in xv])
    p = M3.pca_window(X, V)
    NR, NL, NE_R = M3.NR, M3.NL, M3.NE_R
    Br, Bl = M3.BASIS["Br12"], M3.BASIS["Bl12"]
    mean_coef = np.concatenate([np.linalg.lstsq(Br, p["mu"][:NE_R], rcond=None)[0], np.linalg.lstsq(Bl, p["mu"][NE_R:], rcond=None)[0]])
    phi_coef = np.array([np.concatenate([np.linalg.lstsq(Br, f[:NE_R], rcond=None)[0], np.linalg.lstsq(Bl, f[NE_R:], rcond=None)[0]])
                         for f in p["phi"]]).reshape(p["K"], NR + NL)
    ix = [stage1[s]["names"].index("a") for s in win]
    ah = np.array([stage1[s]["mean"][i] for s, i in zip(win, ix)])
    Va = np.array([np.asarray(stage1[s]["cov"], float)[i, i] for s, i in zip(win, ix)])
    ld = HR.fit_level(ah, Va, 0.3, win)
    am, av = HR.level_prior(ld, ah[-1], Va[-1])
    return {"target": target, "k": k, "window": win, "J": len(win), "K": p["K"], "lambda_scaled": p["lam_scaled"], "sres": p["sres"],
            "mean_coef": mean_coef, "phi_coef": phi_coef, "a_mean": am, "a_var": av, "tau_a_mean": float(np.mean(ld["tau_a"])),
            "X": X, "V": V, "mu": p["mu"]}


def check_repro(pr, saved):
    """저장된 priors[S]와 소수 6자리 비교 -> 최대 절대차 dict."""
    d = {"k": (pr["k"] == saved["k"]), "window": (pr["window"] == saved["window"]), "K": (pr["K"] == saved["K"])}
    d["mean_coef"] = float(np.abs(pr["mean_coef"] - np.asarray(saved["mean_coef"])).max())
    d["phi_coef"] = float(np.abs(pr["phi_coef"] - np.asarray(saved["phi_coef"]).reshape(pr["K"], -1)).max()) if pr["K"] else 0.0
    d["lambda_scaled"] = float(np.abs(np.asarray(pr["lambda_scaled"]) - np.asarray(saved["lambda_scaled"])).max()) if pr["K"] else 0.0
    d["a_mean"] = abs(pr["a_mean"] - saved["a"]["mean"])
    d["a_var"] = abs(pr["a_var"] - saved["a"]["var"])
    d["sigma2_res"] = abs(pr["sres"] - saved["sigma2_res"])
    d["ok6"] = bool(d["k"] and d["window"] and d["K"] and all(d[x] < 5e-7 for x in ("mean_coef", "phi_coef", "lambda_scaled", "a_mean", "a_var", "sigma2_res")))
    return d


# ---------- 뽑기 ----------

def lnC_draws(a, cr):
    """a[N], cr[N,NR] -> ln C(q) [N, len(QS)] (m3.lnC, jax)."""
    import jax
    import jax.numpy as jnp
    f = jax.vmap(lambda aa, cc: M3.lnC({"a": aa, "cr": cc}, jnp.asarray(QS, jnp.float32)))
    return np.asarray(f(jnp.asarray(a, jnp.float32), jnp.asarray(cr, jnp.float32)), float)


def draw_variants(pr, seed):
    rng = np.random.default_rng(seed)
    N = NDRAW
    NR, NE_R, DT = M3.NR, M3.NE_R, M3.DT
    BAND = list(M3.BAND)
    K, J = pr["K"], pr["J"]
    za = rng.standard_normal(N)
    zx = rng.standard_normal((N, 3))
    chi = rng.chisquare(max(J - 1, 1), (N, 3))
    eta = rng.standard_normal((N, len(BAND)))
    zf = rng.standard_normal((N, len(BAND)))
    a = pr["a_mean"] + math.sqrt(pr["a_var"]) * za
    mc = pr["mean_coef"][:NR]
    phi = pr["phi_coef"][:, :NR] if K else np.zeros((0, NR))
    lam = np.asarray(pr["lambda_scaled"], float)[:K]
    out = {}
    # base
    xi = zx[:, :K] * np.sqrt(lam)
    cr_base = mc + xi @ phi
    out["base"] = (a, cr_base)
    # t: nu = J-1, 척도 sqrt(lam_scaled)
    nu = max(J - 1, 1)
    tt = zx[:, :K] / np.sqrt(chi[:, :K] / nu)
    out["t"] = (a, mc + (tt * np.sqrt(lam)) @ phi)
    # ppca: 띠 격자 등방 잔차 sqrt(s2) eta -> 정방 B 역행렬로 계수 (CellProb과 같은 셈)
    s2 = pr["sres"] if pr["sres"] > 1e-8 else 1e-4
    Br = np.asarray(M3.BASIS["Br12"])
    Mfull = np.zeros((NR, DT))
    Mfull[:, :NE_R] = np.linalg.inv(Br)
    M = math.sqrt(s2) * Mfull[:, BAND]                       # [NR, ne]
    out["ppca"] = (a, cr_base + eta @ M.T)
    # fullcov: 띠 곡선 벡터 ~ MVN(평균, (1+1/J)(S - mean V)), 음의 고윳값 0, K로 안 자름. 띠 밖 편차 0.
    X, V = pr["X"], pr["V"]
    Xb = X[:, BAND]
    Vb = V[:, BAND][:, :, BAND]
    S = np.cov(Xb.T, ddof=1).reshape(len(BAND), len(BAND)) - Vb.mean(0)
    w, Q = np.linalg.eigh(0.5 * (S + S.T))
    w = np.clip(w, 0, None)
    dev = (zf * np.sqrt(w * (1 + 1 / J))) @ Q.T              # [N, ne]
    r12 = np.tile(pr["mu"][:NE_R], (N, 1))
    for j, b in enumerate(BAND):
        if b < NE_R:
            r12[:, b] += dev[:, j]
    cr_full = r12 @ np.linalg.inv(Br).T
    out["fullcov"] = (a, cr_full)
    return out, {"fullcov_rank": int((w > 1e-12).sum()), "fullcov_trace": float(w.sum() * (1 + 1 / J))}


# ---------- 척도 ----------

def metrics(draws, truth):
    """draws[N], truth 스칼라 -> dict."""
    q10, q25, q50, q75, q90 = np.quantile(draws, [0.1, 0.25, 0.5, 0.75, 0.9])
    sd = float(draws.std(ddof=1))
    return {"pit": float((draws < truth).mean()), "c80": bool(q10 <= truth <= q90), "c50": bool(q25 <= truth <= q75),
            "w80": float(q90 - q10), "z": float((truth - q50) / sd), "zrob": float((truth - q50) / ((q90 - q10) / 2.5631031310892007)),
            "med": float(q50), "sd": sd}


def summ(cells):
    """cells = metrics dict 목록 -> 요약."""
    if not cells:
        return None
    ok = [c for c in cells if c["sd"] > 1e-5]               # 퇴화(K = 0 이라 모양 사전이 점) 칸은 z에서 뺀다 (coverage·PIT에는 둔다)
    z = np.array([c["z"] for c in ok])
    zr = np.array([c["zrob"] for c in ok])
    pit = np.array([c["pit"] for c in cells])
    return {"n": len(cells), "c80": float(np.mean([c["c80"] for c in cells])), "c50": float(np.mean([c["c50"] for c in cells])),
            "n_deg": len(cells) - len(ok), "absz_med": float(np.median(np.abs(z))), "z_sd": float(z.std(ddof=1)) if len(z) > 1 else None,
            "z_mean": float(z.mean()), "zrob_sd": float(zr.std(ddof=1)) if len(zr) > 1 else None,
            "pit_lt10": float((pit < 0.1).mean()), "pit_gt90": float((pit > 0.9).mean()), "w80_med": float(np.median([c["w80"] for c in cells]))}


def main():
    import jax
    jax.config.update("jax_platform_name", "cpu")
    stage1 = M3.load_stage1(os.path.join(R7, "stage1_M3_D1.json"))
    saved = json.load(open(os.path.join(R7, "pred_M3_tune_D1.json"), encoding="utf-8"))["priors"]
    NR = M3.NR
    # 띠 점검
    wq = [math.log(q / 3) for q in QS]
    band_r_w = [float(M3.BASIS["w12"][b]) for b in M3.BAND if b < M3.NE_R]
    band_info = {"band_r_w_range": [min(band_r_w), max(band_r_w)], "target_w_range": [min(wq), max(wq)],
                 "targets_inside": bool(min(band_r_w) <= min(wq) and max(wq) <= max(band_r_w)),
                 "band_r_pct_range": [3 * math.exp(min(band_r_w)), 3 * math.exp(max(band_r_w))]}

    priors, repro, res, seas = {}, {}, {}, {}
    for s in TARGETS:
        pr = build_prior(stage1, s)
        priors[s] = pr
        if s in saved:
            repro[s] = check_repro(pr, saved[s])
        draws, extra = draw_variants(pr, 20261031 + HR.snum(s))
        truth_ln = {qkey(q): stage1[s]["lnC_q10_50_90"][qkey(q)][1] for q in QS}
        ta = stage1[s]["mean"][stage1[s]["names"].index("a")]
        seas[s] = {"k": pr["k"], "J": pr["J"], "K": pr["K"], "window": pr["window"], "sigma2_res": pr["sres"], "a_prior_mean": pr["a_mean"],
                   "a_prior_sd": math.sqrt(pr["a_var"]), "tau_a_mean": pr["tau_a_mean"], "truth_a_mean": ta, "truth_lnC3_med": truth_ln["3"],
                   "lambda_scaled": list(pr["lambda_scaled"]), **extra}
        res[s] = {}
        for v in VARS:
            a, cr = draws[v]
            L = lnC_draws(a, cr)                                  # [N, 6]
            if v == "base":
                seas[s]["max_abs_lnC3_minus_a"] = float(np.abs(L[:, QS.index(3.0)] - a).max())
            res[s][v] = {"full": {}, "shape": {}}
            for i, q in enumerate(QS):
                res[s][v]["full"][qkey(q)] = metrics(L[:, i], truth_ln[qkey(q)])
                if q != 3.0:
                    res[s][v]["shape"][qkey(q)] = metrics(L[:, i] - a, truth_ln[qkey(q)] - truth_ln["3"])
            if v == "base":
                res[s]["a"] = metrics(a, ta)
        print(s, f"k={pr['k']} J={pr['J']} K={pr['K']}", flush=True)

    # 요약
    summary = {}
    for sn, ss in SETS.items():
        summary[sn] = {}
        for v in VARS:
            summary[sn][v] = {}
            for gn, qs in GROUPS.items():
                summary[sn][v][gn] = summ([res[s][v]["full"][qkey(q)] for s in ss for q in qs])
        summary[sn]["a만"] = {"3%=a 사전": summ([res[s]["a"] for s in ss])}
        summary[sn]["shape"] = {}
        for v in VARS:
            summary[sn]["shape"][v] = {gn: summ([res[s][v]["shape"][qkey(q)] for s in ss for q in qs]) for gn, qs in SHAPE_GROUPS.items()}

    repro_ok = {s: repro[s] for s in repro}
    out = {"about": "WP5a 진단 1 — M3 2단계 사전이 잠재 참값(1단계 lnC 중앙)을 덮는 정도. 새 적합·answer_key 없음. 변형 base/t/ppca/fullcov, 목표별 4,000뽑기.",
           "ndraw": NDRAW, "targets": TARGETS, "tune": TUNE, "qs": QS, "band": band_info, "repro_S30": repro.get("S30"), "repro_tune": repro_ok,
           "max_abs_lnC3_minus_a": max(v["max_abs_lnC3_minus_a"] for v in seas.values()),
           "seasons": seas, "summary": summary, "cells": res}
    os.makedirs(OUTD, exist_ok=True)
    with open(os.path.join(OUTD, "diag1_D1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, (np.floating, np.integer, np.bool_)) else o.tolist())
    with open(os.path.join(OUTD, "diag1_D1.md"), "w", encoding="utf-8") as f:
        f.write(make_md(out))
    print("wrote", OUTD)


def fm(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


def row(name, s):
    if s is None:
        return f"| {name} | - |"
    return (f"| {name} | {s['n']} | {s['c80']:.2f} | {s['c50']:.2f} | {s['absz_med']:.2f} | {s['z_mean']:+.2f} | {fm(s['z_sd'])} | {s['pit_lt10']:.2f} | {s['pit_gt90']:.2f} | {s['w80_med']:.4f} |")


HDR = "| 변형 · 묶음 | 칸수 | 80% | 50% | \\|z\\| 중앙 | z 평균 | z sd | PIT<0.1 | PIT>0.9 | 폭 중앙(ln) |\n|---|---|---|---|---|---|---|---|---|---|"


def make_md(o):
    L = []
    L.append("# WP5a 진단 1 — M3 시즌 사이 사전이 잠재 참값을 덮나 (D1)\n")
    L.append("새 적합 없음. 참값 = `stage1_M3_D1.json`의 ln C(q) 중앙(잠재 참값, 정답 표 안 봄). 사전 = 앞 시즌만으로 낸 2단계 사전, 시즌×목표마다 4,000뽑기. "
             "PIT = 참값의 사전 분위, z = (참값 − 사전 중앙)/사전 sd, 폭 = 사전 80%의 ln 폭. 칸 = 시즌×목표 (1·3% = 2목표, 10% = 1, 전체 = 6).\n")
    L.append("변형 (같은 난수): **base** a~N(평균,분산), ξ_k~N(0,λ_k(1+1/J)) · **t** ξ_k~t(ν=J−1) 척도 √λ_scaled · **ppca** base + 띠 격자 등방 잔차 √σ²_res·η · "
             "**fullcov** 띠 곡선 벡터 ~ MVN(창 평균, (1+1/J)(표본 공분산 − 평균 V), 음의 고윳값 0, K로 안 자름, 띠 밖 편차 0). a는 넷 모두 같다.\n")
    r = o["repro_S30"]
    L.append(f"**재현 (S30)** — 저장된 `pred_M3_tune_D1.json` priors[S30]와: {'일치' if r['ok6'] else '불일치'} "
             f"(최대 절대차 mean_coef {r['mean_coef']:.1e} · phi_coef {r['phi_coef']:.1e} · lambda_scaled {r['lambda_scaled']:.1e} · a mean {r['a_mean']:.1e} · a var {r['a_var']:.1e} · k·창·K {'같음' if r['k'] and r['window'] and r['K'] else '다름'}). "
             f"조정 6시즌 모두: {', '.join(s + (' 일치' if v['ok6'] else ' 불일치') for s, v in o['repro_tune'].items())}.\n")
    b = o["band"]
    L.append(f"**띠 확인** — 띠의 r 격자 범위 {b['band_r_pct_range'][0]:.2f}~{b['band_r_pct_range'][1]:.1f}%, 목표 0.5~10%는 {'띠 안' if b['targets_inside'] else '띠 밖 있음'}. "
             f"3% 컷 = a: 모든 시즌·뽑기에서 max|ln C(3) − a| = {o['max_abs_lnC3_minus_a']:.1e}, 즉 3% 목표는 a 사전 그 자체.\n")
    for sn in SETS:
        L.append(f"## 포함·z — {sn} ({len(SETS[sn])}시즌)\n")
        L.append(HDR)
        for v in VARS:
            for gn in ("1·3%", "10%", "전체"):
                L.append(row(f"{v} · {gn}", o["summary"][sn][v][gn]))
        L.append("")
    L.append("## a 몫과 곡선 모양 몫 (ln C(q) − a)\n")
    L.append("a 몫 = 3% 컷의 사전 (변형 무관, 참값은 1단계 a 평균). 모양 몫 = ln C(q) − a, 참값은 ln C(q) − ln C(3%) (3% 제외 5목표).\n")
    L.append(HDR)
    for sn in SETS:
        L.append(row(f"{sn} · a (3%)", o["summary"][sn]["a만"]["3%=a 사전"]))
        for v in VARS:
            for gn in ("1%", "10%", "전체(3% 뺌)"):
                L.append(row(f"{sn} · 모양 {v} · {gn}", o["summary"][sn]["shape"][v][gn]))
    L.append("")
    L.append("## 시즌별 창·K\n")
    L.append("| 시즌 | k | J | K | σ²_res | a 사전 평균 | a 사전 sd | 참값 a | z(a) | t의 ν |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for s in TARGETS:
        e = o["seasons"][s]
        L.append(f"| {s}{'*' if s in TUNE else ''} | {e['k']} | {e['J']} | {e['K']} | {e['sigma2_res']:.4f} | {e['a_prior_mean']:.4f} | {e['a_prior_sd']:.4f} | {e['truth_a_mean']:.4f} | {o['cells'][s]['a']['z']:+.2f} | {max(e['J'] - 1, 1)} |")
    L.append("\n(* = 조정 시즌. K = 0이면 곡선은 창 평균 고정이고 base·t는 같다.)")
    L.append("\nt는 ν = J−1 ≤ 2인 시즌에서 분산이 무한/매우 커 표본 sd가 불안정하다 (z sd 해석 주의; JSON의 zrob_sd는 10~90 폭으로 잰 로버스트 판).")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
