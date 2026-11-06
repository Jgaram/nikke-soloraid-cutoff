"""WP12b 컷 실제 곡선을 재현하는 식 (H1) · 시즌 중 컷 꼴 (H2). 서술만 — 예측·채점 없음.

  python3 -I tools/solo_raid/lab7/wp12_cut_curve.py --obs-dir W/obs7/full --out data/solo-raid/analysis/r7/wp12/cut_curve_D1.json

계획: docs/solo-raid-method.md 「WP12b」 (후보·고르기·재현 기준은 계산 전에 적은 그대로). 공용 코드는 읽기만 (wp12_cut.py 함수 import).
좌표: y = ln 점수, x = log10 %. 실제 곡선 = 끝 화면 점 PCHIP (wp12_cut E와 같음).

H1 최종 컷을 재현하는 식. 맞추기 범위 0.5~15% (그 시즌 점이 덮은 범위 안), 점 무게 같게.
  후보: P2·P3 (다항식) · H2L·H2Q (3%·10% 꺾임 이어진 조각, 직선·2차) · S1·S2 (Muggeo 2003 꺾는 자리 1·2개 추정, 시작 3%·10%) · H4L·H4Q (1·3·5·10% 꺾임).
  H2Q·H4Q의 「2차판」 = 다항식 x² 항 하나 + 꺾임마다 (x − x_b)₊² (조각마다 2차 항 → 조각 모두 2차, 이어짐은 1차 미분까지).
  세 가지 판: 자유 (시즌마다 모든 모수), 공통 꼴 (꼴 모수 θ는 24시즌 공통, 시즌마다 수준 a_s와 꼴 배수 m_s — y = a_s + m_s·θ·φ(x), 평균 m = 1),
  시즌 빼기 교차검증 (θ를 나머지 시즌으로 맞추고 뺀 시즌은 a_s·m_s만).
  Muggeo 후보의 공통 꼴은 꺾는 자리를 훈련 시즌 자유 적합 추정의 중앙값으로 고정한다 (뺀 시즌의 점은 안 씀).
  고르기: (가) 시즌마다 맞춘 BIC 합 (n·ln(RSS/n) + k·ln n, k = 상수 + 꼴 계수 + 꺾는 자리) · (나) 교차검증 오차 (시즌별 RMS ln 점수의 시즌 평균, 1-SE = 최소 후보의 시즌 표준오차).
  「재현 기준을 채움」 = 공통 꼴 판에서 24시즌 가운데 18시즌(75%) 이상이 최대 |Δ ln 점수| ≤ 0.01 (0.5~15%, 문서에 임계 시즌 수가 없어 이 약속을 계산 전에 정함).
H2 시즌 중 컷 꼴: 05시 시작 120시간 21시즌 A·B 사진, 시즌×일차 창 (사진 8장 이상). 시각 차는 F_P로 창 가운데로 옮긴다:
  lpc = log10 p + F_P(h/H) − F_P(h_c/H) (wp12_cut.add_win의 lpc). (가) ln 점수 = y_fin(lpc + Δ_sd). (나) + η·τ·(y_fin(x) − y_fin(log10 3))·1[x > log10 3], τ = (H − h)/H.
  η는 모든 창 SSE 최소 (창마다 Δ_sd를 격자 0.005에서 다시 맞춤), 띠 = 시즌 붓스트랩, 교차검증 = 시즌 빼기 (η 훈련, 뺀 시즌 창은 Δ_sd만 다시 맞춤).
공개 규칙: 글 번호·원문·닉네임·원글 주소·사람 열쇠·사진 열쇠를 쓰지 않는다 (시각·%·점수·개수만).
"""
import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator, BSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import wp12_tables as WT  # noqa: E402
import wp12_cut as WC  # noqa: E402

SEASONS = WT.SEASONS
LN10 = math.log(10.0)
X_LO, X_HI = math.log10(0.5), math.log10(15.0)
B3, B1, B5, B10 = math.log10(3.0), 0.0, math.log10(5.0), 1.0
DEV_TOL = 0.01
FILL_NEED = 18           # 24시즌 가운데 채워야 「채움」 (계산 전 약속)
NGRID = 240
R5 = WC.R5
rl = WC.rl
med = WC.med
pq = WC.pq
B_BOOT = 1000

CANDS = ["P2", "P3", "H2L", "H2Q", "S1", "S2", "H4L", "H4Q"]
CAND_TEXT = {
    "P2": "2차 다항식", "P3": "3차 다항식",
    "H2L": "3%·10%에서 꺾이는 이어진 직선 조각", "H2Q": "3%·10% 꺾임 + 조각마다 2차 항",
    "S1": "꺾는 자리 1개 추정 (Muggeo, 시작 3%)", "S2": "꺾는 자리 2개 추정 (Muggeo, 시작 3%·10%)",
    "H4L": "1·3·5·10%에서 꺾이는 이어진 직선 조각", "H4Q": "1·3·5·10% 꺾임 + 조각마다 2차 항",
}
NPSI = {"S1": 1, "S2": 2}


def hp(x, b):
    return np.maximum(x - b, 0.0)


def basis(name, x, psi=None):
    """꼴 열 (절편 제외). x = log10 %."""
    x = np.asarray(x, float)
    xc = x - 0.5
    if name == "P2":
        return np.column_stack([xc, xc ** 2])
    if name == "P3":
        return np.column_stack([xc, xc ** 2, xc ** 3])
    if name == "H2L":
        return np.column_stack([xc, hp(x, B3), hp(x, B10)])
    if name == "H2Q":
        return np.column_stack([xc, xc ** 2, hp(x, B3), hp(x, B10), hp(x, B3) ** 2, hp(x, B10) ** 2])
    if name == "H4L":
        return np.column_stack([xc, hp(x, B1), hp(x, B3), hp(x, B5), hp(x, B10)])
    if name == "H4Q":
        bs = (B1, B3, B5, B10)
        return np.column_stack([xc, xc ** 2] + [hp(x, b) for b in bs] + [hp(x, b) ** 2 for b in bs])
    if name in ("X6L", "X6Q"):
        bs = [math.log10(q) for q in (0.7, 1.0, 2.0, 3.0, 5.0, 10.0)]
        cols = [xc] + ([xc ** 2] if name == "X6Q" else []) + [hp(x, b) for b in bs] + ([hp(x, b) ** 2 for b in bs] if name == "X6Q" else [])
        return np.column_stack(cols)
    if name == "S1":
        return np.column_stack([xc, hp(x, psi[0])])
    if name == "S2":
        return np.column_stack([xc, hp(x, psi[0]), hp(x, psi[1])])
    raise KeyError(name)


def ncols(name):
    return basis(name, np.array([0.1, 0.2]), [0.3, 0.6]).shape[1]


def npar_common(name):
    """공통 꼴 모수 수: 꼴 계수 − 1 (평균 배수 1로 고정) + 꺾는 자리."""
    return ncols(name) - 1 + NPSI.get(name, 0)


# ======================= 곡선·자료 =======================

def load_curves():
    out = {}
    for s in SEASONS:
        ep = sorted(WT.end_points(s))
        p = np.array([a for a, _ in ep])
        sc = np.array([b for _, b in ep])
        x, y = np.log10(p), np.log(sc)
        pc = PchipInterpolator(x, y)
        lo, hi = float(x.min()), float(x.max())
        fa, fb = max(X_LO, lo), min(X_HI, hi)
        sel = (x >= fa - 1e-9) & (x <= fb + 1e-9)
        out[s] = {"x": x, "y": y, "pc": pc, "lo": lo, "hi": hi, "fa": fa, "fb": fb, "xs": x[sel], "ys": y[sel], "n": int(sel.sum()),
                  "grid": np.linspace(fa, fb, NGRID)}
        out[s]["ref"] = pc(out[s]["grid"])
    return out


def muggeo(x, y, psi0, tries=60):
    """Muggeo (2003) 반복: y ~ 1 + x + Σ(x − ψ)₊ ; ψ ← ψ + γ/c. 이어진 선형 조각 꺾는 자리 추정. 가장 낮은 RSS 상태 반환."""
    k = len(psi0)
    psi = np.array(psi0, float)
    xm, xM = x.min(), x.max()
    best = None
    conv = False
    for it in range(tries):
        U = np.column_stack([hp(x, p) for p in psi])
        V = np.column_stack([-(x > p).astype(float) for p in psi])
        X = np.column_stack([np.ones_like(x), x, U, V])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        c = beta[2:2 + k]
        g = beta[2 + k:]
        # 현재 ψ에서의 RSS
        X0 = np.column_stack([np.ones_like(x), x, U])
        b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
        rss = float(((y - X0 @ b0) ** 2).sum())
        if best is None or rss < best[0] - 1e-15:
            best = (rss, psi.copy())
        step = np.where(np.abs(c) > 1e-9, g / np.where(np.abs(c) > 1e-9, c, 1.0), 0.0)
        step = np.clip(step, -0.15, 0.15)
        if np.all(np.abs(step) < 1e-6):
            conv = True
            break
        psi = psi + step
        psi = np.clip(psi, xm + 0.03, xM - 0.03)
        if k == 2 and psi[1] - psi[0] < 0.05:
            psi[1] = min(xM - 0.03, psi[0] + 0.05)
        psi = np.sort(psi) if k == 2 else psi
    return best[1], conv


def free_fit(name, cv, psi=None):
    x, y = cv["xs"], cv["ys"]
    conv = None
    if name in NPSI:
        start = [B3] if name == "S1" else [B3, B10]
        psi, conv = muggeo(x, y, [min(max(v, x.min() + 0.03), x.max() - 0.03) for v in start])
    Xd = np.column_stack([np.ones_like(x), basis(name, x, psi)])
    beta = np.linalg.lstsq(Xd, y, rcond=None)[0]
    res = y - Xd @ beta
    rss = float(res @ res)
    g = cv["grid"]
    pred = np.column_stack([np.ones_like(g), basis(name, g, psi)]) @ beta
    return {"beta": beta, "psi": None if psi is None else np.array(psi), "rss": rss, "n": len(x), "pred": pred, "conv": conv}


def bic(name, f):
    k = 1 + ncols(name) + NPSI.get(name, 0)
    n = f["n"]
    if n < k + 2:
        return None
    return n * math.log(max(f["rss"], 1e-12) / n) + k * math.log(n)


# ======================= 공통 꼴 (평균 m = 1) =======================

def common_fit(Phis, ys, iters=300):
    S = len(Phis)
    Xs = np.vstack([P - P.mean(0) for P in Phis])
    ys_ = np.concatenate([y - y.mean() for y in ys])
    theta = np.linalg.lstsq(Xs, ys_, rcond=None)[0]
    m = np.ones(S)
    a = np.zeros(S)
    last = None
    for it in range(iters):
        for i in range(S):
            G = Phis[i] @ theta
            Xd = np.column_stack([np.ones_like(G), G])
            a[i], m[i] = np.linalg.lstsq(Xd, ys[i], rcond=None)[0]
        mb = m.mean()
        if mb <= 0:
            break
        m = m / mb
        theta = theta * mb
        Xm = np.vstack([m[i] * Phis[i] for i in range(S)])
        tg = np.concatenate([ys[i] - a[i] for i in range(S)])
        theta = np.linalg.lstsq(Xm, tg, rcond=None)[0]
        rss = sum(float(((ys[i] - a[i] - m[i] * (Phis[i] @ theta)) ** 2).sum()) for i in range(S))
        if last is not None and abs(last - rss) < 1e-13 * max(1.0, rss):
            break
        last = rss
    # 마지막 a, m 다시
    for i in range(S):
        G = Phis[i] @ theta
        a[i], m[i] = np.linalg.lstsq(np.column_stack([np.ones_like(G), G]), ys[i], rcond=None)[0]
    return theta, a, m


def fit_am(theta, Phi, y):
    G = Phi @ theta
    a, m = np.linalg.lstsq(np.column_stack([np.ones_like(G), G]), y, rcond=None)[0]
    return float(a), float(m)


def psi_common(name, seasons, free_psi):
    if name not in NPSI:
        return None
    arr = np.array([free_psi[(name, s)] for s in seasons])
    return np.median(arr, axis=0)


def common_run(name, train, test, curves, free_psi):
    """train 시즌으로 공통 θ를 맞추고 test 시즌마다 (a, m)만 맞춘 결과: {s: (a, m, pred grid, rms)} (test가 train과 같으면 공통 꼴 판)."""
    ps = psi_common(name, train, free_psi)
    Phis = [basis(name, curves[s]["xs"], ps) for s in train]
    theta, a, m = common_fit(Phis, [curves[s]["ys"] for s in train])
    out = {}
    if list(test) == list(train):
        for i, s in enumerate(train):
            out[s] = (float(a[i]), float(m[i]))
    else:
        for s in test:
            out[s] = fit_am(theta, basis(name, curves[s]["xs"], ps), curves[s]["ys"])
    res = {}
    for s, (aa, mm) in out.items():
        cv = curves[s]
        pred_g = aa + mm * (basis(name, cv["grid"], ps) @ theta)
        pred_p = aa + mm * (basis(name, cv["xs"], ps) @ theta)
        rms = float(np.sqrt(np.mean((cv["ys"] - pred_p) ** 2)))
        res[s] = {"a": aa, "m": mm, "pred": pred_g, "rms": rms, "maxdev": float(np.max(np.abs(pred_g - cv["ref"])))}
    return theta, ps, res


# ======================= H1 =======================

def part_H1(curves, plots, notes):
    H1 = {"tables": {}, "notes": notes, "extra": {}}
    t0 = time.time()
    # --- 자유 적합 + BIC
    free, free_psi = {}, {}
    for name in CANDS:
        for s in SEASONS:
            f = free_fit(name, curves[s])
            free[(name, s)] = f
            if f["psi"] is not None:
                free_psi[(name, s)] = f["psi"]
    print("H1 자유 적합", round(time.time() - t0, 1), "초", flush=True)
    feas_all = [s for s in SEASONS if all(bic(n, free[(n, s)]) is not None for n in CANDS)]
    # --- 공통 꼴 (24시즌) · 시즌 빼기
    commonRun, loso = {}, {}
    for name in CANDS:
        commonRun[name] = common_run(name, SEASONS, SEASONS, curves, free_psi)
        per = {}
        for s in SEASONS:
            train = [x for x in SEASONS if x != s]
            _, _, r = common_run(name, train, [s], curves, free_psi)
            per[s] = r[s]
        loso[name] = per
    print("H1 공통·교차검증", round(time.time() - t0, 1), "초", flush=True)

    rows = []
    for name in CANDS:
        cr = commonRun[name][2]
        fr_dev = {s: float(np.max(np.abs(free[(name, s)]["pred"] - curves[s]["ref"]))) for s in SEASONS}
        co_dev = {s: cr[s]["maxdev"] for s in SEASONS}
        lo_dev = {s: loso[name][s]["maxdev"] for s in SEASONS}
        cvrms = np.array([loso[name][s]["rms"] for s in SEASONS])
        bics = [bic(name, free[(name, s)]) for s in feas_all]
        cnt = lambda d: sum(1 for s in SEASONS if d[s] <= DEV_TOL)  # noqa: E731
        rows.append({
            "cand": name, "text": CAND_TEXT[name], "n_params_free_per_season": 1 + ncols(name) + NPSI.get(name, 0), "n_params_common_shape": npar_common(name),
            "bic_sum": R5(sum(b for b in [bic(name, free[(name, s)]) for s in feas_all])), "bic_seasons": len(feas_all),
            "bic_seasons_infeasible": [s for s in SEASONS if bic(name, free[(name, s)]) is None],
            "cv_rms_mean": R5(cvrms.mean()), "cv_rms_se": R5(cvrms.std(ddof=1) / math.sqrt(len(cvrms))), "cv_rms_median": R5(float(np.median(cvrms))),
            "n_ok_free": cnt(fr_dev), "n_ok_common": cnt(co_dev), "n_ok_loso": cnt(lo_dev), "n_seasons": len(SEASONS),
            "maxdev_free_median": R5(med(list(fr_dev.values()))), "maxdev_common_median": R5(med(list(co_dev.values()))), "maxdev_loso_median": R5(med(list(lo_dev.values()))),
            "maxdev_common_worst": R5(max(co_dev.values())), "maxdev_common_worst_season": max(co_dev, key=co_dev.get),
            "maxdev_free_worst": R5(max(fr_dev.values())),
            "free_noconv_seasons": [s for s in SEASONS if free[(name, s)]["conv"] is False],
        })
    H1["tables"]["candidates"] = rows
    # --- 고르기
    best = min(rows, key=lambda r: r["cv_rms_mean"])
    thr = best["cv_rms_mean"] + best["cv_rms_se"]
    inse = [r for r in rows if r["cv_rms_mean"] <= thr]
    fills = [r for r in rows if r["n_ok_common"] >= FILL_NEED]
    elig = [r for r in fills if r["cv_rms_mean"] <= thr]
    rule_txt = ""
    if elig:
        pick = min(elig, key=lambda r: (r["n_params_common_shape"], r["n_params_free_per_season"], r["cv_rms_mean"]))
        rule_txt = "재현 기준을 채우고 (공통 꼴 판에서 18시즌 이상) 교차검증 1-SE 안인 후보 가운데 공통 꼴 모수가 가장 적은 것."
        filled = True
    else:
        # 못 채움: 1-SE 안 후보 중 공통 꼴 판 최대 |Δ| 중앙값이 가장 작은 것 (표시: 못 채움)
        pick = min(inse, key=lambda r: (r["maxdev_common_median"], r["n_params_common_shape"]))
        rule_txt = "재현 기준을 채우는 후보가 없다 → 1-SE 안인 후보 가운데 공통 꼴 판 최대 |Δ| 중앙값이 가장 작은 것을 「가장 가까운 것」으로 적는다 (재현 기준 못 채움)."
        filled = False
    H1["tables"]["selection"] = [{"min_cv_cand": best["cand"], "min_cv": best["cv_rms_mean"], "one_se": R5(thr), "in_one_se": [r["cand"] for r in inse],
                                  "fills_criterion_common": [r["cand"] for r in fills], "eligible": [r["cand"] for r in elig], "chosen": pick["cand"], "chosen_fills_criterion": filled,
                                  "rule": rule_txt, "bic_min_cand": min(rows, key=lambda r: r["bic_sum"])["cand"], "bic_seasons": len(feas_all)}]
    chosen = pick["cand"]
    for r in sorted(rows, key=lambda r: r["bic_sum"]):
        notes.append(f"H1 {r['cand']} ({r['text']}): BIC 합 {r['bic_sum']:.1f} (시즌 {r['bic_seasons']}, 점 {sum(curves[s]['n'] for s in feas_all)}) · 교차검증 RMS {r['cv_rms_mean']:.4f}±{r['cv_rms_se']:.4f} · "
                     f"최대|Δ ln| ≤ {DEV_TOL} 시즌 수: 자유 {r['n_ok_free']}/24 · 공통 꼴 {r['n_ok_common']}/24 · 교차검증 {r['n_ok_loso']}/24 (공통 꼴 판 최대 중앙 {r['maxdev_common_median']:.4f}, 최악 {r['maxdev_common_worst']:.4f} {r['maxdev_common_worst_season']}).")
    notes.append(f"H1 고르기 규칙: {rule_txt} 교차검증 최소 {best['cand']} ({best['cv_rms_mean']:.4f}), 1-SE 문턱 {thr:.4f} 안 {[r['cand'] for r in inse]}; BIC 합 최소 {min(rows, key=lambda r: r['bic_sum'])['cand']}. 고른 것 = {chosen}"
                 f" ({'재현 기준 채움' if filled else '재현 기준 못 채움'}).")

    # --- 어긋나는 %대 (고른 식 + 모든 후보 공통 꼴 판)
    edges = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
    dev_rows = []
    for name in CANDS:
        cr = commonRun[name][2]
        for bi in range(len(edges) - 1):
            a, b = math.log10(edges[bi]), math.log10(edges[bi + 1])
            per_s = []
            for s in SEASONS:
                g = curves[s]["grid"]
                m = (g >= a - 1e-9) & (g <= b + 1e-9)
                if m.sum() < 3:
                    continue
                d = np.abs(cr[s]["pred"][m] - curves[s]["ref"][m])
                per_s.append((s, float(d.max())))
            if per_s:
                v = np.array([z[1] for z in per_s])
                dev_rows.append({"cand": name, "pct_lo": edges[bi], "pct_hi": edges[bi + 1], "n_seasons": len(v), "max_dev_median": R5(np.median(v)), "max_dev_p90": R5(np.percentile(v, 90)),
                                 "max_dev_worst": R5(v.max()), "n_over_tol": int((v > DEV_TOL).sum())})
    H1["tables"]["dev_by_pct_band_common"] = dev_rows
    cd = [r for r in dev_rows if r["cand"] == chosen]
    notes.append(f"H1 고른 식 {chosen}, % 구간별 최대 |Δ ln 점수| (공통 꼴 판, 구간 안 시즌 최대의 중앙 / 최악 / 0.01 넘는 시즌): " +
                 "; ".join(f"{r['pct_lo']}~{r['pct_hi']}% {r['max_dev_median']:.4f}/{r['max_dev_worst']:.4f}/{r['n_over_tol']} (시즌 {r['n_seasons']})" for r in cd))
    cdf = [{"cand": chosen}]
    # --- 고른 식의 모수
    theta, ps, cres = commonRun[chosen]
    a_ = np.array([cres[s]["a"] for s in SEASONS])
    m_ = np.array([cres[s]["m"] for s in SEASONS])
    par = {"cand": chosen, "theta": rl(theta), "psi_log10": None if ps is None else rl(ps), "psi_pct": None if ps is None else rl(10 ** np.asarray(ps)),
           "a_median": R5(np.median(a_)), "a_p25": R5(np.percentile(a_, 25)), "a_p75": R5(np.percentile(a_, 75)),
           "m_median": R5(np.median(m_)), "m_p25": R5(np.percentile(m_, 25)), "m_p75": R5(np.percentile(m_, 75)), "m_min": R5(m_.min()), "m_max": R5(m_.max()),
           "basis_text": "y = a_s + m_s·Σ θ_j φ_j(x), x = log10 %, xc = x − 0.5, 열 순서 = basis() (xc, [xc²], 꺾임 (x − x_b)₊ …); 평균 m = 1"}
    H1["tables"]["chosen_params"] = [par]
    H1["tables"]["chosen_by_season"] = [{"season": s, "n_points_fit": curves[s]["n"], "a": R5(cres[s]["a"]), "m": R5(cres[s]["m"]), "maxdev_common": R5(cres[s]["maxdev"]),
                                          "maxdev_free": R5(float(np.max(np.abs(free[(chosen, s)]["pred"] - curves[s]["ref"])))), "maxdev_loso": R5(loso[chosen][s]["maxdev"]),
                                          "few_points": curves[s]["n"] < 15} for s in SEASONS]
    # 조각 기울기 (공통 꼴 × m_s=1): 구간마다 −dy/dx
    xs_ = np.linspace(X_LO, X_HI, 300)
    yy = basis(chosen, xs_, ps) @ theta
    sl = -np.gradient(yy, xs_)
    notes.append(f"H1 고른 식 {chosen} 공통 꼴 (m_s=1) 기울기 s=−dy/dx: " + ", ".join(f"{q}% {float(np.interp(math.log10(q), xs_, sl)):.3f}" for q in (0.7, 1, 2, 3, 5, 10, 14)) +
                 f"; 수준 a_s 중앙 {par['a_median']:.3f} [{par['a_p25']:.3f}~{par['a_p75']:.3f}], 꼴 배수 m_s 중앙 {par['m_median']:.3f} [{par['m_p25']:.3f}~{par['m_p75']:.3f}] (범위 {par['m_min']:.3f}~{par['m_max']:.3f}).")
    # --- 꺾는 자리 분포 (Muggeo)
    kink_rows = []
    RT = WT_RT
    for name in ("S1", "S2"):
        arr = np.array([free_psi[(name, s)] for s in SEASONS])
        for j in range(arr.shape[1]):
            pct = 10 ** arr[:, j]
            kink_rows.append({"cand": name, "kink_index": j + 1, "pct_median": R5(np.median(pct)), "pct_p25": R5(np.percentile(pct, 25)), "pct_p75": R5(np.percentile(pct, 75)),
                              "pct_min": R5(pct.min()), "pct_max": R5(pct.max()), "n_seasons": len(pct),
                              "n_within_0p05_decade_of_3pct": int((np.abs(arr[:, j] - B3) <= 0.05).sum()), "n_within_0p05_decade_of_10pct": int((np.abs(arr[:, j] - B10) <= 0.05).sum()),
                              "n_within_0p05_decade_of_5pct": int((np.abs(arr[:, j] - B5) <= 0.05).sum()), "n_within_0p05_decade_of_1pct": int((np.abs(arr[:, j] - B1) <= 0.05).sum()),
                              "n_noconv": int(sum(1 for s in SEASONS if free[(name, s)]["conv"] is False))})
    H1["tables"]["muggeo_kinks"] = kink_rows
    kb = []
    for s in SEASONS:
        row = {"season": s, "n_points_fit": curves[s]["n"], "reward_tiers_pct_in_0p5_15": [t for t in RT.get(s, []) if 0.5 <= t <= 15]}
        for name in ("S1", "S2"):
            row[name + "_pct"] = rl(10 ** free_psi[(name, s)])
            row[name + "_rss"] = R5(free[(name, s)]["rss"])
            row[name + "_conv"] = free[(name, s)]["conv"]
        kb.append(row)
    H1["tables"]["muggeo_kinks_by_season"] = kb
    for r in kink_rows:
        notes.append(f"H1 Muggeo {r['cand']} 꺾는 자리 {r['kink_index']}: 시즌 중앙 {r['pct_median']:.2f}% [사분위 {r['pct_p25']:.2f}~{r['pct_p75']:.2f}, 범위 {r['pct_min']:.2f}~{r['pct_max']:.2f}] (시즌 {r['n_seasons']}); "
                     f"3%±0.05decade {r['n_within_0p05_decade_of_3pct']}시즌 · 5% {r['n_within_0p05_decade_of_5pct']} · 10% {r['n_within_0p05_decade_of_10pct']} · 1% {r['n_within_0p05_decade_of_1pct']}; 수렴 못 한 시즌 {r['n_noconv']}.")
    H1["extra"]["더 본 것_muggeo_tier_distance"] = "Muggeo 추정 자리와 같은 시즌 보상표 화면(reward_table) % 가운데 가장 가까운 것: muggeo_kinks_by_season.reward_tiers_pct_in_0p5_15 열 참고 (화면이 있는 시즌만)."

    # --- extra: 더 유연한 공통 꼴 (고르기에 안 씀) — 어느 정도 유연해야 0.01을 채우나
    flex = []
    for df in (6, 8, 10, 12, 14):
        x0, x1 = X_LO, X_HI
        # 균일 매듭 3차 B-스플라인 df개 (절편은 a_s가 맡도록 열 하나 제외 → df−1열)
        nint = df - 4
        kn = np.r_[[x0] * 4, np.linspace(x0, x1, nint + 2)[1:-1], [x1] * 4]

        def bs(x, kn=kn, df=df):
            x = np.clip(np.asarray(x, float), x0, x1 - 1e-12)
            return BSpline.design_matrix(x, kn, 3).toarray()[:, 1:]
        Phis = [bs(curves[s]["xs"]) for s in SEASONS]
        th, a, m = common_fit(Phis, [curves[s]["ys"] for s in SEASONS])
        dev = {}
        for i, s in enumerate(SEASONS):
            g = curves[s]["grid"]
            dev[s] = float(np.max(np.abs(a[i] + m[i] * (bs(g) @ th) - curves[s]["ref"])))
        lo_dev, cvr = {}, []
        for s in SEASONS:
            tr = [z for z in SEASONS if z != s]
            th2, _, _ = common_fit([bs(curves[z]["xs"]) for z in tr], [curves[z]["ys"] for z in tr])
            aa, mm = fit_am(th2, bs(curves[s]["xs"]), curves[s]["ys"])
            lo_dev[s] = float(np.max(np.abs(aa + mm * (bs(curves[s]["grid"]) @ th2) - curves[s]["ref"])))
            cvr.append(float(np.sqrt(np.mean((curves[s]["ys"] - aa - mm * (bs(curves[s]["xs"]) @ th2)) ** 2))))
        flex.append({"bspline_df": df, "n_params_common_shape": df - 2, "n_ok_common": sum(1 for s in SEASONS if dev[s] <= DEV_TOL), "n_ok_loso": sum(1 for s in SEASONS if lo_dev[s] <= DEV_TOL),
                     "maxdev_common_median": R5(np.median(list(dev.values()))), "maxdev_common_worst": R5(max(dev.values())), "maxdev_loso_median": R5(np.median(list(lo_dev.values()))),
                     "cv_rms_mean": R5(np.mean(cvr))})
    H1["extra"]["더 본 것_flexible_common_bspline"] = {"label": "더 본 것 (고르기에 안 씀): 0.5~15% 3차 B-스플라인 공통 꼴 + 시즌 a_s·m_s — 몇 개의 꼴 모수가 있어야 0.01을 채우나", "rows": flex}
    notes.append("H1 더 본 것 (고르기 밖) 3차 B-스플라인 공통 꼴: " + "; ".join(f"자유도 {r['bspline_df']} (꼴 모수 {r['n_params_common_shape']}) 공통 꼴 채운 시즌 {r['n_ok_common']}/24, 교차검증 {r['n_ok_loso']}/24, 최대 중앙 {r['maxdev_common_median']:.4f}, 최악 {r['maxdev_common_worst']:.4f}" for r in flex))

    # --- extra: 보상 경계 전부 (0.7·1·2·3·5·10%)에서 꺾이는 식 — 고르기에 안 씀
    xr = []
    for name in ("X6L", "X6Q"):
        k = 1 + ncols(name)
        feas = [s for s in SEASONS if curves[s]["n"] >= k + 2]
        fdev = {s: float(np.max(np.abs(free_fit(name, curves[s])["pred"] - curves[s]["ref"]))) for s in feas}
        _, _, cr6 = common_run(name, SEASONS, SEASONS, curves, free_psi)
        lo6 = {}
        for s in SEASONS:
            tr = [z for z in SEASONS if z != s]
            _, _, r6 = common_run(name, tr, [s], curves, free_psi)
            lo6[s] = r6[s]
        xr.append({"cand": name, "n_params_free_per_season": k, "n_params_common_shape": npar_common(name), "free_seasons_feasible": len(feas),
                   "n_ok_free": sum(1 for s in feas if fdev[s] <= DEV_TOL), "maxdev_free_median": R5(med(list(fdev.values()))),
                   "n_ok_common": sum(1 for s in SEASONS if cr6[s]["maxdev"] <= DEV_TOL), "maxdev_common_median": R5(med([cr6[s]["maxdev"] for s in SEASONS])),
                   "maxdev_common_worst": R5(max(cr6[s]["maxdev"] for s in SEASONS)),
                   "n_ok_loso": sum(1 for s in SEASONS if lo6[s]["maxdev"] <= DEV_TOL), "cv_rms_mean": R5(np.mean([lo6[s]["rms"] for s in SEASONS]))})
    H1["extra"]["더 본 것_all_tier_hinges"] = {"label": "더 본 것 (고르기에 안 씀): 0.7·1·2·3·5·10% 모두에서 꺾이는 이어진 조각 (X6L 직선, X6Q 2차)", "rows": xr}
    notes.append("H1 더 본 것 (고르기 밖) 보상 경계 0.7·1·2·3·5·10% 전부 꺾임: " + "; ".join(
        f"{r['cand']} (꼴 모수 {r['n_params_common_shape']}) 자유 {r['n_ok_free']}/{r['free_seasons_feasible']}시즌 (점이 모수+2 이상인 시즌만) · 공통 꼴 {r['n_ok_common']}/24 (최대 중앙 {r['maxdev_common_median']:.4f}, 최악 {r['maxdev_common_worst']:.4f}) · 교차검증 {r['n_ok_loso']}/24 (RMS {r['cv_rms_mean']:.4f})" for r in xr))

    # --- 그림
    # H1b: 후보별 |Δ| 대 % (시즌 중앙 선; 공통 꼴 판)
    gp = np.linspace(X_LO, X_HI, 60)
    ser = []
    for name in CANDS:
        cr = commonRun[name][2]
        ys_ = []
        for xv in gp:
            v = []
            for s in SEASONS:
                g = curves[s]["grid"]
                if xv < g[0] - 1e-9 or xv > g[-1] + 1e-9:
                    continue
                k = int(np.argmin(np.abs(g - xv)))
                v.append(abs(cr[s]["pred"][k] - curves[s]["ref"][k]))
            ys_.append(float(np.median(v)) if len(v) >= 8 else None)
        ok = [i for i, v in enumerate(ys_) if v is not None]
        ser.append({"name": f"{name} {CAND_TEXT[name]} (공통 꼴, 시즌 중앙)", "kind": "line", "x": [R5(10 ** gp[i]) for i in ok], "y": [R5(ys_[i]) for i in ok], "group": "data"})
    ser.append({"name": "재현 기준 0.01", "kind": "line", "x": [0.5, 15.0], "y": [DEV_TOL, DEV_TOL], "group": "F"})
    plots.append({"id": "H1b", "part": "H1", "title": "후보 식과 PCHIP 실제 곡선의 차 (|Δ ln 점수|) 대 %", "x": {"label": "% (log)", "log": True, "reverse": False}, "y": {"label": "|Δ ln 점수| (시즌 중앙)", "log": True},
                  "series": ser, "note": f"공통 꼴 + 시즌 a_s·m_s 판, 시즌 {len(SEASONS)}개, 시즌 점 {sum(curves[s]['n'] for s in SEASONS)}개 (0.5~15%). 그 %를 덮은 시즌이 8개 이상인 곳만."})
    # H1c: 꺾는 자리 분포
    ser = []
    for name, nm in (("S1", "S1 꺾는 자리"), ("S2", "S2 꺾는 자리")):
        arr = np.array([free_psi[(name, s)] for s in SEASONS])
        for j in range(arr.shape[1]):
            ser.append({"name": f"{nm} {j + 1}", "kind": "points", "x": [int(s[1:]) for s in SEASONS], "y": rl(10 ** arr[:, j]), "group": "season"})
    for b in (3.0, 10.0):
        ser.append({"name": f"보상 경계 {b:g}%", "kind": "line", "x": [19, 42], "y": [b, b], "group": "F"})
    plots.append({"id": "H1c", "part": "H1", "title": "Muggeo 추정 꺾는 자리 (시즌마다)", "x": {"label": "시즌 번호", "log": False, "reverse": False}, "y": {"label": "꺾는 자리 (%)", "log": True},
                  "series": ser, "note": "시작값 3%·10%, 0.5~15% 점, 시즌 24개."})
    # H1a: 시즌 몇 개 (적은·많은 점, 옛·새)
    pick_s = ["S27", "S22", "S36", "S33", "S42", "S20"]
    for s in pick_s:
        cv = curves[s]
        cr = commonRun[chosen][2][s]
        f = free[(chosen, s)]
        plots.append({"id": f"H1a_{s}", "part": "H1", "title": f"{s} 최종 컷 곡선: 끝 화면 점·PCHIP·식 {chosen}", "x": {"label": "% (log)", "log": True, "reverse": False}, "y": {"label": "ln 점수", "log": False},
                      "series": [{"name": "끝 화면 점", "kind": "points", "x": rl(10 ** cv["xs"]), "y": rl(cv["ys"]), "group": "data"},
                                 {"name": "PCHIP (실제 곡선)", "kind": "line", "x": rl(10 ** cv["grid"]), "y": rl(cv["ref"]), "group": "season"},
                                 {"name": f"{chosen} 공통 꼴 (a_s, m_s)", "kind": "line", "x": rl(10 ** cv["grid"]), "y": rl(cr["pred"]), "group": "F"},
                                 {"name": f"{chosen} 자유", "kind": "line", "x": rl(10 ** cv["grid"]), "y": rl(f["pred"]), "group": "M3"}],
                      "note": f"{s}: 0.5~15% 점 {cv['n']}개. 공통 꼴 최대 |Δ| {cr['maxdev']:.4f}, 자유 {float(np.max(np.abs(f['pred'] - cv['ref']))):.4f}."})
    H1["extra"]["secs"] = round(time.time() - t0, 1)
    return H1, chosen, commonRun, curves


# ======================= H2 =======================

DGRID = np.arange(-0.3, 2.6 + 1e-9, 0.005)
EGRID = np.round(np.arange(-1.0, 3.0 + 1e-9, 0.01), 4)
E0 = int(np.argmin(np.abs(EGRID)))


class Curve:
    def __init__(self, f, lo, hi):
        self.f, self.lo, self.hi = f, lo, hi

    def y(self, x):
        return self.f(np.clip(x, self.lo, self.hi))


def pchip_curve(cv):
    return Curve(cv["pc"], cv["lo"], cv["hi"])


def formula_curve(name, theta, ps, a, m, cv):
    def f(x):
        x = np.asarray(x, float)
        return (a + m * (basis(name, x.ravel(), ps) @ theta)).reshape(x.shape)
    return Curve(f, cv["fa"], cv["fb"])


def build_windows(rows, curves_by_s, min_n=8):
    """rows: add_win된 일차 창 줄. 창마다 A, B, Cc 배열 (Δ 격자)."""
    W = []
    x3 = B3
    keys = sorted({(r["s"], r["win"]) for r in rows})
    for s, d in keys:
        rw = [r for r in rows if r["s"] == s and r["win"] == d]
        cu = curves_by_s[s]
        rw = [r for r in rw if cu.lo - 1e-9 <= math.log10(r["pf"]) <= cu.hi + 1e-9]
        if len(rw) < min_n:
            continue
        lpc = np.array([r["lpc"] for r in rw])
        ls = np.log([r["score"] for r in rw])
        info = WT.season_info()[s]
        tau = np.array([(info["H"] - r["h"]) / info["H"] for r in rw])
        X = lpc[:, None] + DGRID[None, :]
        Yc = cu.y(X)
        y3 = float(cu.y(np.array([x3]))[0])
        U = (Yc - y3) * (X > x3)
        Rr = ls[:, None] - Yc
        TU = tau[:, None] * U
        W.append({"s": s, "day": d, "n": len(rw), "A": (Rr ** 2).sum(0), "B": (Rr * TU).sum(0), "C": (TU ** 2).sum(0), "hc": rw[0]["hc"], "H": info["H"], "rows": rw})
    return W


def eta_tables(W):
    """창×η 최소 SSE (Δ 격자 최소)와 최소 위치."""
    nW = len(W)
    M = np.zeros((nW, len(EGRID)))
    IDX = np.zeros((nW, len(EGRID)), dtype=int)
    for i, w in enumerate(W):
        sse = w["A"][:, None] - 2 * EGRID[None, :] * w["B"][:, None] + EGRID[None, :] ** 2 * w["C"][:, None]
        IDX[i] = np.argmin(sse, axis=0)
        M[i] = sse[IDX[i], np.arange(len(EGRID))]
    return M, IDX


def h2_analysis(W, tag, curves_by_s, plots_flag=False):
    """η 추정·띠·교차검증."""
    M, IDX = eta_tables(W)
    seas = sorted({w["s"] for w in W})
    si = {s: i for i, s in enumerate(seas)}
    S = np.zeros((len(seas), len(EGRID)))
    nS = np.zeros(len(seas))
    for i, w in enumerate(W):
        S[si[w["s"]]] += M[i]
        nS[si[w["s"]]] += w["n"]
    tot = S.sum(0)
    ie = int(np.argmin(tot))
    eta = float(EGRID[ie])
    # 포물선 보정
    if 0 < ie < len(EGRID) - 1:
        y0, y1, y2 = tot[ie - 1], tot[ie], tot[ie + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            eta = float(EGRID[ie] + 0.01 * 0.5 * (y0 - y2) / den)
    rng = np.random.default_rng(1204)
    bs = []
    for _ in range(B_BOOT):
        c = np.bincount(rng.integers(0, len(seas), len(seas)), minlength=len(seas)).astype(float)
        bs.append(EGRID[int(np.argmin(c @ S))])
    bs = np.array(bs)
    loso_eta, per = [], []
    for s in seas:
        k = si[s]
        e_tr = int(np.argmin(tot - S[k]))
        loso_eta.append(float(EGRID[e_tr]))
        per.append({"season": s, "n_photos": int(nS[k]), "n_windows": sum(1 for w in W if w["s"] == s), "eta_train": float(EGRID[e_tr]),
                    "rms_a": R5(math.sqrt(S[k, E0] / nS[k])), "rms_b": R5(math.sqrt(S[k, e_tr] / nS[k])), "sse_a": R5(S[k, E0]), "sse_b": R5(S[k, e_tr])})
    cv_a = sum(p["sse_a"] for p in per)
    cv_b = sum(p["sse_b"] for p in per)
    ntot = nS.sum()
    # 시즌 쌍 차이 (a − b) 평균과 표준오차 (sse 기준)
    diffs = np.array([p["sse_a"] - p["sse_b"] for p in per])
    rel = np.array([1 - p["sse_b"] / p["sse_a"] for p in per])
    res = {"tag": tag, "n_windows": len(W), "n_seasons": len(seas), "n_photos": int(ntot), "eta": R5(eta), "eta_boot_p10": R5(np.percentile(bs, 10)), "eta_boot_p90": R5(np.percentile(bs, 90)),
           "eta_boot_p2p5": R5(np.percentile(bs, 2.5)), "eta_boot_p97p5": R5(np.percentile(bs, 97.5)), "eta_boot_B": B_BOOT, "frac_boot_eta_le0": R5(float((bs <= 0).mean())),
           "eta_loso_min": R5(min(loso_eta)), "eta_loso_max": R5(max(loso_eta)),
           "rms_insample_a": R5(math.sqrt(tot[E0] / ntot)), "rms_insample_b": R5(math.sqrt(tot[ie] / ntot)),
           "rms_cv_a": R5(math.sqrt(cv_a / ntot)), "rms_cv_b": R5(math.sqrt(cv_b / ntot)), "sse_gain_cv_rel": R5(1 - cv_b / cv_a),
           "n_seasons_b_better": int((diffs > 0).sum()), "season_rel_gain_median": R5(np.median(rel)), "season_rel_gain_p25": R5(np.percentile(rel, 25)), "season_rel_gain_p75": R5(np.percentile(rel, 75)),
           "season_diff_se_z": R5(diffs.mean() / (diffs.std(ddof=1) / math.sqrt(len(diffs)))) if len(diffs) > 2 else None}
    # 창별 Δ
    wins = []
    ia = IDX[:, E0]
    ib = np.array([int(np.argmin(np.abs(EGRID - eta))) for _ in W])
    for i, w in enumerate(W):
        da, db = float(DGRID[ia[i]]), float(DGRID[IDX[i, ib[i]]])
        wins.append({"season": w["s"], "day": w["day"], "n_photos": w["n"], "delta_a": R5(da), "delta_b": R5(db), "rms_a": R5(math.sqrt(M[i, E0] / w["n"])),
                     "rms_b": R5(math.sqrt(M[i, ib[i]] / w["n"])), "F_P_center": R5(WT.F_P(w["hc"] / w["H"])), "delta_a_minus_F_P_center": R5(da - WT.F_P(w["hc"] / w["H"]))})
    return res, wins, per, {"eta": eta, "ib": ib, "ia": ia, "IDX": IDX}


def ypred(cu, x, tau, eta):
    y = cu.y(x)
    y3 = float(cu.y(np.array([B3]))[0])
    return y + eta * tau * (y - y3) * (x > B3)


def ratio_windows(W, wins, eta, curves_by_s):
    """창마다 관측 C_q/C3 (국소 선형) 대 (가) (나) 기대 — Δ_sd로 옮김."""
    out = []
    for w, wi in zip(W, wins):
        cu = curves_by_s[w["s"]]
        rw = w["rows"]
        tau_c = (w["H"] - w["hc"]) / w["H"]
        f3 = WC.local_fit(rw, 3.0)
        for q in (1.0, 10.0):
            fq = WC.local_fit(rw, q)
            if fq is None or f3 is None:
                continue
            lq = math.log10(q)
            res = {"s": w["s"], "day": w["day"], "q": q, "obs": math.exp(fq - f3)}
            for key, dl, e in (("a", wi["delta_a"], 0.0), ("b", wi["delta_b"], eta)):
                xq, x3 = lq + dl, B3 + dl
                if not (cu.lo <= xq <= cu.hi and cu.lo <= x3 <= cu.hi):
                    res = None
                    break
                res["exp_" + key] = math.exp(float(ypred(cu, np.array([xq]), tau_c, e)[0] - ypred(cu, np.array([x3]), tau_c, e)[0]))
            if res is None:
                continue
            out.append(res)
    return out


def part_H2(obs_dir, curves, chosen, commonRun, plots, notes):
    H2 = {"tables": {}, "notes": notes, "extra": {}}
    t0 = time.time()
    info = WT.season_info()
    ab = WC.prep_photos(obs_dir, ("A", "B"))
    main = [dict(r) for r in ab if info[r["s"]]["clock_ok"]]
    rows = WC.add_win(main, "day")
    cby = {s: pchip_curve(curves[s]) for s in SEASONS}
    W = build_windows(rows, cby)
    res, wins, per, st = h2_analysis(W, "PCHIP", cby)
    H2["tables"]["eta_pchip"] = [res]
    H2["tables"]["windows_pchip"] = wins
    H2["tables"]["by_season_loso"] = per
    seas = sorted({w["s"] for w in W})
    notes.append(f"H2 자료: 05시 시작 120시간 {len(SEASONS) and sum(1 for s in SEASONS if info[s]['clock_ok'])}시즌의 A·B 사진 {len(main)}장 중 시즌×일차 창 (사진 8장 이상, 그 사진의 최종 % 가 PCHIP 곡선이 덮은 범위 안) {res['n_windows']}개 · 시즌 {res['n_seasons']} · 사진 {res['n_photos']}장.")
    notes.append(f"H2 η (PCHIP 최종 곡선): {res['eta']:.3f}, 시즌 붓스트랩 10~90% [{res['eta_boot_p10']:.3f}, {res['eta_boot_p90']:.3f}] (B={B_BOOT}, 2.5~97.5% [{res['eta_boot_p2p5']:.3f}, {res['eta_boot_p97p5']:.3f}], 붓스트랩 η ≤ 0 비율 {res['frac_boot_eta_le0']:.3f}); 시즌 빼기 η 범위 [{res['eta_loso_min']:.3f}, {res['eta_loso_max']:.3f}].")
    notes.append(f"H2 교차검증 (시즌 빼기, 뺀 시즌 창은 Δ_sd만 다시 맞춤) RMS ln 점수: (가) {res['rms_cv_a']:.4f} → (나) {res['rms_cv_b']:.4f} (SSE 이득 {100 * res['sse_gain_cv_rel']:.2f}%); (나)가 더 나은 시즌 {res['n_seasons_b_better']}/{res['n_seasons']}, 시즌별 이득 중앙 {100 * res['season_rel_gain_median']:.2f}% [{100 * res['season_rel_gain_p25']:.2f}~{100 * res['season_rel_gain_p75']:.2f}], 시즌 쌍 차이 z {res['season_diff_se_z']}. 표본 안 RMS (가) {res['rms_insample_a']:.4f} (나) {res['rms_insample_b']:.4f}.")
    # Δ_sd와 F_P 중앙 비교 (일차별)
    drow = []
    for d in range(1, 7):
        v = [w for w in wins if w["day"] == d]
        if v:
            drow.append({"day": d, "n_windows": len(v), "delta_minus_F_P_median": R5(med([w["delta_a_minus_F_P_center"] for w in v])), "delta_minus_F_P_p25": R5(pq([w["delta_a_minus_F_P_center"] for w in v], 25)),
                         "delta_minus_F_P_p75": R5(pq([w["delta_a_minus_F_P_center"] for w in v], 75)), "rms_a_median": R5(med([w["rms_a"] for w in v])), "rms_b_median": R5(med([w["rms_b"] for w in v]))})
    H2["tables"]["delta_vs_F_P_by_day"] = drow
    notes.append("H2 Δ_sd − F_P(창 가운데) 일차별 중앙 (침수 곡선 F와의 수평 차, log10): " + ", ".join(f"{r['day']}일차 {r['delta_minus_F_P_median']:+.3f} (창 {r['n_windows']}, 단면 RMS {r['rms_a_median']:.4f}→{r['rms_b_median']:.4f})" for r in drow))
    # C1/C3, C10/C3
    rw = ratio_windows(W, wins, res["eta"], cby)
    rrows = []
    for q in (1.0, 10.0):
        for d in range(1, 7):
            v = [r for r in rw if r["q"] == q and r["day"] == d]
            if not v:
                continue
            oa = [r["obs"] / r["exp_a"] for r in v]
            ob = [r["obs"] / r["exp_b"] for r in v]
            rrows.append({"q_pct": q, "day": d, "n_windows": len(v), "n_seasons": len({r["s"] for r in v}), "few": len(v) < 5,
                          "obs_median": R5(med([r["obs"] for r in v])), "exp_a_median": R5(med([r["exp_a"] for r in v])), "exp_b_median": R5(med([r["exp_b"] for r in v])),
                          "obs_over_exp_a_median": R5(med(oa)), "obs_over_exp_a_p25": R5(pq(oa, 25)), "obs_over_exp_a_p75": R5(pq(oa, 75)),
                          "obs_over_exp_b_median": R5(med(ob)), "obs_over_exp_b_p25": R5(pq(ob, 25)), "obs_over_exp_b_p75": R5(pq(ob, 75))})
    H2["tables"]["C_ratio_obs_over_exp"] = rrows
    for q, nm in ((10.0, "C10/C3"), (1.0, "C1/C3")):
        zz = [r for r in rrows if r["q_pct"] == q]
        al = [r for r in rw if r["q"] == q]
        notes.append(f"H2 {nm} 관측÷기대 (가) 분리 꼴 / (나) η 꼴, 일차별 중앙: " + " · ".join(f"{r['day']}일차 {r['obs_over_exp_a_median']:.3f} / {r['obs_over_exp_b_median']:.3f} (창 {r['n_windows']}, 시즌 {r['n_seasons']}{', 적음' if r['few'] else ''})" for r in zz) +
                     (f" · 전체 중앙 {med([r['obs'] / r['exp_a'] for r in al]):.3f} / {med([r['obs'] / r['exp_b'] for r in al]):.3f} (창 {len(al)})" if al else ""))

    # --- extra: 0.5~15% 안 최종 % 사진만, 고른 식을 최종 곡선으로
    ex = {}
    rows2 = [r for r in rows if 0.5 <= r["pf"] <= 15.0]
    W2 = build_windows(rows2, cby)
    r2, _, _, _ = h2_analysis(W2, "PCHIP, 최종 % 0.5~15% 사진만", cby)
    ex["더 본 것_final_pct_0p5_15_only"] = r2
    cform = {s: formula_curve(chosen, commonRun[chosen][0], commonRun[chosen][1], commonRun[chosen][2][s]["a"], commonRun[chosen][2][s]["m"], curves[s]) for s in SEASONS}
    W3 = build_windows(rows2, cform)
    r3, _, _, _ = h2_analysis(W3, f"고른 식 {chosen} (공통 꼴 a_s·m_s), 최종 % 0.5~15% 사진만", cform)
    ex["더 본 것_chosen_formula_as_final_curve"] = r3
    # 전체 사진 이진 η<0 점검: 3% 아래 쪽 기울기도 곱하는 판은 계획 밖이라 안 함
    H2["extra"].update(ex)
    notes.append(f"H2 더 본 것: 최종 % 0.5~15% 사진만 η {r2['eta']:.3f} [{r2['eta_boot_p10']:.3f}, {r2['eta_boot_p90']:.3f}] (창 {r2['n_windows']}, 사진 {r2['n_photos']}), 교차검증 이득 {100 * r2['sse_gain_cv_rel']:.2f}% ; 고른 식을 최종 곡선으로 쓰면 η {r3['eta']:.3f} [{r3['eta_boot_p10']:.3f}, {r3['eta_boot_p90']:.3f}] (창 {r3['n_windows']}), 교차검증 이득 {100 * r3['sse_gain_cv_rel']:.2f}%.")

    # --- 그림 H2a
    ser = []
    for q, nm in ((10.0, "C10/C3"), (1.0, "C1/C3")):
        zz = [r for r in rrows if r["q_pct"] == q and not r["few"]]
        if zz:
            ser.append({"name": f"{nm} 관측÷기대 (가) 분리 꼴", "kind": "line", "x": [r["day"] for r in zz], "y": [r["obs_over_exp_a_median"] for r in zz], "group": "data"})
            ser.append({"name": f"{nm} (가) 사분위", "kind": "band", "x": [r["day"] for r in zz], "y": [r["obs_over_exp_a_median"] for r in zz], "lo": [r["obs_over_exp_a_p25"] for r in zz], "hi": [r["obs_over_exp_a_p75"] for r in zz], "group": "data"})
            ser.append({"name": f"{nm} 관측÷기대 (나) η 꼴", "kind": "line", "x": [r["day"] for r in zz], "y": [r["obs_over_exp_b_median"] for r in zz], "group": "F"})
            ser.append({"name": f"{nm} (나) 사분위", "kind": "band", "x": [r["day"] for r in zz], "y": [r["obs_over_exp_b_median"] for r in zz], "lo": [r["obs_over_exp_b_p25"] for r in zz], "hi": [r["obs_over_exp_b_p75"] for r in zz], "group": "F"})
    plots.append({"id": "H2a", "part": "H2", "title": "C10/C3·C1/C3 관측 ÷ 기대 — (가) 분리 꼴 대 (나) 3% 위 기울기 η 꼴", "x": {"label": "일차", "log": False, "reverse": False}, "y": {"label": "비 (1 = 식 기대와 같음)", "log": False},
                  "series": ser, "note": f"창 = 시즌×일차 (사진 8장 이상) {res['n_windows']}개, Δ_sd는 창마다 맞춤, η = {res['eta']:.3f}. 창 5개 미만 일차는 뺌."})
    H2["extra"]["secs"] = round(time.time() - t0, 1)
    return H2


# ======================= main =======================

WT_RT = {}


def git_head():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    global WT_RT
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    WT_RT = WT.reward_tiers(a.obs_dir)
    curves = load_curves()
    plots, n1, n2 = [], [], []
    H1, chosen, commonRun, curves = part_H1(curves, plots, n1)
    print("H1 끝", round(time.time() - t0, 1), "초 — 고른 식", chosen, flush=True)
    H2 = part_H2(a.obs_dir, curves, chosen, commonRun, plots, n2)
    rules = {
        "실제 곡선": "끝 화면 점 PCHIP (wp12_cut E와 같음). 식은 그 시즌 점이 덮은 0.5~15% 안 점으로 맞추고 (점 무게 같게), 같은 범위 240점 격자에서 PCHIP과 비교한다.",
        "재현 기준": f"|Δ ln 점수| ≤ {DEV_TOL} (0.5~15%, 시즌마다 최대). 「채움」 = 공통 꼴 판에서 24시즌 가운데 {FILL_NEED}시즌 이상 (계산 전 약속).",
        "고르기": "(가) 시즌별 BIC 합 (n ln(RSS/n) + k ln n; 점이 k+2 미만인 시즌은 모든 후보에서 뺌) · (나) 시즌 빼기 교차검증 (θ 나머지 시즌, 뺀 시즌 a_s·m_s; 시즌 RMS 평균, 1-SE = 최소 후보의 시즌 표준오차). 채우고 1-SE 안인 것 가운데 공통 꼴 모수 최소.",
        "Muggeo": "Muggeo 2003 반복 (시작 3%·10%, 걸음 ±0.15 decade 제한, 최대 60회, 가장 낮은 RSS 상태). 공통 꼴의 꺾는 자리 = 훈련 시즌 자유 추정의 중앙값.",
        "2차판": "H2Q·H4Q = x² 항 + 꺾임마다 (x − x_b)₊² (조각마다 2차 항).",
        "H2 창": "05시 시작 120시간 시즌 A·B 사진 (wp12_cut.prep_photos), 시즌×일차 창 (일차 05시 경계) 사진 8장 이상, 옮김 lpc = log10 p + F_P(h/H) − F_P(h_c/H). Δ_sd 격자 −0.3~2.6 간격 0.005, η 격자 −1~3 간격 0.01 (+포물선 보정).",
        "H2 η 띠": f"시즌 붓스트랩 B={B_BOOT}, 시드 1204, 10~90%.",
        "점 무게": "모든 점 무게 같음 (끝 점 잡음 거의 없음).",
    }
    J = {"header": {"about": "WP12b 컷 실제 곡선(PCHIP)을 재현하는 식 H1·시즌 중 컷 꼴 H2. 서술만 — 예측·채점 없음. 24시즌 정답·끝 점을 모두 봄.", "data_ver": C.DATA_VER,
                    "date": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d"), "code_head": git_head(), "secs": round(time.time() - t0, 1), "rules": rules},
         "H1": H1, "H2": H2, "plots": plots}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=1)
    print("쓴 파일", a.out, os.path.getsize(a.out), "바이트", round(time.time() - t0, 1), "초")


if __name__ == "__main__":
    main()
