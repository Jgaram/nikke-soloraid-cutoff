"""WP13a 컷 꼴 G (앞 시즌만) — 최종 컷 ln C(q) = a_j − m_j · G(w), w = ln(q/3), G(0) = 0. 서술·꼴 만들기만 (예측·채점 없음).

  python3 -I tools/solo_raid/lab7/wp13_cut_shape.py --out data/solo-raid/analysis/r7/wp13/cut_shape_D1.json
  (같은 폴더에 .md 요약을 함께 씀. 걸린 시간: 아래 header.secs — 프로세스 하나, 약 수십 초)

계획: docs/solo-raid-method.md 「WP13 5차 연구」 (컷 꼴 · 앞 시즌만 · 이번 시즌 편차 층의 컷 부분). 계산 전에 정한 모형 그대로:
  G = 이어진 직선 조각, 꺾는 자리 q = 1·3·5·10·15·20·30% (WP12 H4L의 1·3·5·10%에 보상 경계 15·20·30%를 더함), 조각 8개 기울기 g_0..g_7 = dG/dw > 0 (>= 1e-4).
  1% 밑·30% 위 조각은 끝까지 직선. 자료 = 시즌마다 끝 화면 점 wp12_tables.end_points(s) 중 0.1 <= q <= 40%.
  맞추기 = 학습 시즌 점 전체 최소제곱, 시즌마다 무게 같게 (점 무게 1/n_j). 번갈아 (a_j, m_j) 시즌별 최소제곱 <-> 기울기 NNLS. 학습 시즌 m_j 기하평균 1.
  능선 벌점(끝 조각: g0-g1 · g6-g5 · g7-g6 의 제곱차에 RIDGE) 쓴다 — 크기는 RIDGE 상수·header.rules에 적음 (24시즌 판에서 결과가 거의 안 바뀌는 크기).
  편차 층: 학습 시즌 PCHIP(log10 % 대 ln 점수, 그 시즌 점 범위 안) 잔차 e_j(w) = [lnC_j(3%) − lnC_j(w)] − m_j G(w) 의 결측 PCA (EM).
  m_j se = 그 시즌 (a, m) 최소제곱 표준 오차 (잔차 분산).
공용 코드는 읽기만 (wp12_tables.end_points · m3.BASIS["wg"] import). 형식은 scratchpad 꼴 파일 형식 v1 의 cut_shape_D1.json.
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
import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import wp12_tables as WT  # noqa: E402
import m3  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")
SEASONS = WT.SEASONS
LN10 = math.log(10.0)
Q_KNOTS = [1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]
W_KNOTS = [math.log(q / 3.0) for q in Q_KNOTS]
Q_LO, Q_HI = 0.1, 40.0
G_MIN = 1e-4
RIDGE = 1e-7           # 끝 조각 능선 벌점 무게 (가로 오차 맞춤 기준: 벌점 0 판과 기울기 최대 차 0.005 안 — 1차 세로 맞춤은 1e-5)
EDGE_PAIRS = [(0, 1), (6, 5), (7, 6)]   # (끝 조각, 이웃 조각)
H4L_REF = {0.7: 0.20, 1.0: 0.19, 2.0: 0.18, 3.0: 0.50, 5.0: 0.81, 10.0: 1.13, 14.0: 1.46}   # WP12 H4L -d ln점수/d log10 %
WG = np.array(m3.BASIS["wg"], float)
GRID_W = None
EDGES = np.array([-1e9] + W_KNOTS + [1e9])


def basis(w):
    """B[:, i] = ∫_0^w 1[조각 i] dt (부호 있는 겹침 길이). G(w) = B @ g."""
    w = np.asarray(w, float)[:, None]
    lo, hi = EDGES[None, :-1], EDGES[None, 1:]
    return np.clip(w, lo, hi) - np.clip(0.0, lo, hi)


def slope_at(g, w):
    """w에서 G'(w) = 든 조각의 기울기 (꺾는 자리 위면 양쪽 평균)."""
    w = np.asarray(w, float)
    idx = np.searchsorted(np.array(W_KNOTS), w, side="right")
    s = np.asarray(g, float)[idx].copy()
    for k, wk in enumerate(W_KNOTS):
        s = np.where(np.abs(w - wk) < 1e-9, 0.5 * (g[k] + g[k + 1]), s)
    return s


def season_points(q_lo=Q_LO, q_hi=Q_HI):
    out = {}
    for s in SEASONS:
        ep = sorted(WT.end_points(s))
        p = np.array([a for a, _ in ep], float)
        sc = np.array([b for _, b in ep], float)
        sel = (p >= q_lo - 1e-9) & (p <= q_hi + 1e-9)
        out[s] = {"w": np.log(p[sel] / 3.0), "y": np.log(sc[sel]), "n": int(sel.sum()),
                  "pc": PchipInterpolator(np.log10(p), np.log(sc)), "lo": float(np.log10(p.min())), "hi": float(np.log10(p.max()))}
    return out


def ls_am(B_j, y, sw=None):
    """y = a − m·G (가중: sw = 점 무게의 제곱근). -> a, m, se_a, se_m, 가중 RSS (잔차 분산 기준)."""
    if sw is None:
        sw = np.ones(len(y))
    X = np.column_stack([np.ones(len(y)), -B_j])
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    r = (y - X @ beta) * sw
    dof = max(len(y) - 2, 1)
    s2 = float(r @ r) / dof
    try:
        cov = s2 * np.linalg.inv((X * sw[:, None]).T @ (X * sw[:, None]))
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.array([np.nan, np.nan])
    return float(beta[0]), float(beta[1]), float(se[0]), float(se[1]), float(r @ r)


def point_sw(g, m, w, horiz):
    """점 무게의 제곱근. 가로 오차(log10 %) 맞춤이면 1/s_i, s_i = m·G'(w_i)·ln10 (ln 점수 per log10 %); 아니면 1."""
    if not horiz:
        return np.ones(len(w))
    return 1.0 / np.maximum(m * slope_at(g, w) * LN10, 1e-9)


def fit_shape(pts, seasons, ridge=RIDGE, iters=500, tol=1e-6, horiz=True):
    """학습 시즌으로 g (8) 와 시즌별 (a, m, se). 시즌 무게 같게, 기하평균 m = 1.
    horiz=True: 가로 오차 IRLS (무게 1/s_i², s_i 는 직전 g·m_j 로 셈). horiz=False: 1차 세로 맞춤 (v1_vertical)."""
    Bs = {s: basis(pts[s]["w"]) for s in seasons}
    g = np.full(8, 0.5)
    mprev = {s: 1.0 for s in seasons}
    for it in range(iters):
        am = {}
        for s in seasons:
            G = Bs[s] @ g
            sw = point_sw(g, mprev[s], pts[s]["w"], horiz)
            a, m = _am_given_G(G, pts[s]["y"], sw)
            am[s] = (a, m)
        gm = math.exp(np.mean([math.log(max(am[s][1], 1e-6)) for s in seasons]))
        g = g * gm
        am = {s: (am[s][0], am[s][1] / gm) for s in seasons}
        rows, tg = [], []
        for s in seasons:
            a, m = am[s]
            mprev[s] = m
            wt = point_sw(g, m, pts[s]["w"], horiz) / math.sqrt(pts[s]["n"])
            A = -m * Bs[s] * wt[:, None]
            rows.append(A)
            tg.append((pts[s]["y"] - a) * wt - A @ np.full(8, G_MIN))
        for i, k in EDGE_PAIRS:
            r = np.zeros(8)
            r[i], r[k] = 1.0, -1.0
            rows.append(math.sqrt(ridge) * r[None, :])
            tg.append(np.zeros(1))
        x, _ = nnls(np.vstack(rows), np.concatenate(tg), maxiter=2000)
        gn = x + G_MIN
        d = float(np.max(np.abs(gn - g)))
        g = gn
        if d < tol:
            break
    # 마무리: 정규화 한 번 더
    fits = {}
    for _ in range(3):
        for s in seasons:
            sw = point_sw(g, mprev[s], pts[s]["w"], horiz)
            fits[s] = ls_am(Bs[s] @ g, pts[s]["y"], sw)
            mprev[s] = fits[s][1]
        gm = math.exp(np.mean([math.log(fits[s][1]) for s in seasons]))
        g = g * gm
        for s in seasons:
            mprev[s] = fits[s][1] / gm
    fits = {s: ls_am(Bs[s] @ g, pts[s]["y"], point_sw(g, mprev[s], pts[s]["w"], horiz)) for s in seasons}
    return g, fits, it + 1


def _am_given_G(G, y, sw=None):
    if sw is None:
        sw = np.ones(len(y))
    X = np.column_stack([np.ones(len(y)), -G])
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return float(beta[0]), float(beta[1])


def pca_em(Y, M, k, iters=500):
    """Y (시즌 × 칸), M = 값 있음. 열 평균을 빼고 결측을 SVD 복원으로 채워 가며 주성분 (wp12_flood.pca_em과 같은 꼴). -> mean, comps, share, scores"""
    Yf = np.where(M, Y, 0.0)
    mu = np.nanmean(np.where(M, Y, np.nan), axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    for j in range(Y.shape[1]):
        Yf[~M[:, j], j] = mu[j]
    for _ in range(iters):
        mu = Yf.mean(0)
        U, s, Vt = np.linalg.svd(Yf - mu, full_matrices=False)
        rec = mu + (U[:, :k] * s[:k]) @ Vt[:k]
        Yn = np.where(M, Y, rec)
        if np.max(np.abs(Yn - Yf)) < 1e-9:
            Yf = Yn
            break
        Yf = Yn
    mu = Yf.mean(0)
    U, s, Vt = np.linalg.svd(Yf - mu, full_matrices=False)
    share = s ** 2 / max((s ** 2).sum(), 1e-300)
    return mu, Vt, share, U * s


def dev_grid():
    qs = np.exp(np.linspace(math.log(Q_LO), math.log(Q_HI), 40))
    w = np.sort(np.concatenate([np.log(qs / 3.0), [0.0]]))
    return w


def residuals(pts, seasons, fits, g, wgrid):
    """e_j(w) = [lnC_j(3%) − lnC_j(w)] − m_j G(w); 시즌 PCHIP 범위 밖·3%가 범위 밖이면 결측. -> Y, M, 빠진 시즌 목록"""
    Gg = basis(wgrid) @ g
    Y = np.full((len(seasons), len(wgrid)), np.nan)
    skipped = []
    for r, s in enumerate(seasons):
        pc, lo, hi = pts[s]["pc"], pts[s]["lo"], pts[s]["hi"]
        if not (lo - 1e-12 <= math.log10(3.0) <= hi + 1e-12):
            skipped.append(s)
            continue
        x = np.log10(3.0 * np.exp(wgrid))
        ok = (x >= lo - 1e-9) & (x <= hi + 1e-9)
        R = float(pc(math.log10(3.0))) - pc(np.clip(x, lo, hi))
        Y[r, ok] = (R - fits[s][1] * Gg)[ok]
    M = ~np.isnan(Y)
    return Y, M, skipped


PC_SHARE = 0.9   # 편차 층 PC 몫 기준 (--share)
PC_KMAX = 2      # 편차 층 PC 수 상한 (--kmax)


def build_pcs(pts, seasons, fits, g, wgrid, horiz=True, cover=True):
    """cover=True (3차): 격자점마다 점 범위 안인 학습 시즌 수 n(w) 를 세어 n >= max(3, ceil(J/2)) 인 3%를 낀 이어진 구간만 PCA에 쓰고,
    그 밖은 phi 를 구간 끝값 상수로 잇는다 (밖에서 R 기울기 = m·G'). cover=False: 2차 (격자 전체, v2_pca_full_grid)."""
    Y, M, skipped = residuals(pts, seasons, fits, g, wgrid)
    Yfull = None
    rows = [i for i in range(len(seasons)) if M[i].any()]
    J = len(rows)
    out = {"K": 0, "phi": [], "lambda": [], "share": [], "w_range": [float(WG[0]), float(WG[-1])], "J_resid": J,
           "skipped_no_3pct": skipped, "mean_resid_max_abs": None, "mean_resid_grid": None, "share_all": []}
    if J < 2:
        return out
    Yr, Mr = Y[rows], M[rows]
    wfull = wgrid
    cov = np.ones(len(wfull), bool)
    out["n_seasons_grid"] = [int(v) for v in Mr.sum(0)]
    if cover:
        thr = max(3, math.ceil(0.5 * J))
        n = Mr.sum(0)
        i0 = int(np.argmin(np.abs(wfull)))
        out["cover_thr"] = thr
        if n[i0] < thr:
            out["w_range"] = [0.0, 0.0]
            return out
        lo = i0
        while lo > 0 and n[lo - 1] >= thr:
            lo -= 1
        hi = i0
        while hi < len(n) - 1 and n[hi + 1] >= thr:
            hi += 1
        cov = np.zeros(len(wfull), bool)
        cov[lo:hi + 1] = True
        Yr, Mr, wgrid, g_ = Yr[:, cov], Mr[:, cov], wfull[cov], g
    kmax = min(J - 1, 6)
    # 가로 단위 무게 PCA: 칸 무게 W(w) = 1/(G'(w) ln10)^2 (m 기하평균 1), Z = e·sqrt(W) 로 PCA 뒤 phi = v / sqrt(W) 로 세로 단위로 되돌림
    sq = 1.0 / (slope_at(g, wgrid) * LN10) if horiz else np.ones(len(wgrid))
    out["weighted"] = bool(horiz)
    mu, comps, share, sc = pca_em(Yr * sq[None, :], Mr, kmax)
    comps = comps / sq[None, :]
    cell_mean = np.nanmean(np.where(Mr, Yr, np.nan), axis=0)
    out["mean_resid_max_abs"] = float(np.nanmax(np.abs(cell_mean)))
    ncell = Mr.sum(0)
    out["mean_resid_n_seasons"] = [int(v) for v in ncell]
    ok5 = ncell >= 5
    out["mean_resid_max_abs_n5"] = float(np.nanmax(np.abs(cell_mean[ok5]))) if ok5.any() else None
    out["mean_resid_over_0.005_q"] = [round(float(3 * math.exp(x)), 2) for x, c, n in zip(wgrid, cell_mean, ncell) if n >= 5 and abs(c) > 0.005]
    out["mean_resid_grid"] = [None if np.isnan(v) else round(float(v), 5) for v in cell_mean]
    out["share_all"] = [round(float(v), 4) for v in share]
    out["n_cover_cells"] = int(cov.sum())
    K = 0
    cum = 0.0
    for kk in range(len(share)):
        cum += share[kk]
        K = kk + 1
        if cum >= PC_SHARE - 1e-12:
            break
    K = min(K, PC_KMAX, max(J - 2, 0))
    out["K"] = K
    if K == 0:
        return out
    ph, lam = [], []
    for k in range(K):
        v = comps[k].copy()
        # 부호 고정: 가장 큰 |값| 이 양수
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
            sc[:, k] = -sc[:, k]
        phg = np.interp(WG, wgrid, v)       # 범위 밖은 끝값 상수 (세로 단위 phi)
        phg = phg - float(np.interp(0.0, WG, phg))   # wg 선형 보간으로 phi(0)이 정확히 0이 되게 상수만큼 이동 (몇 1e-4)
        ph.append(phg)
        lam.append(float(np.var(sc[:, k], ddof=1)))
    out["phi"] = [p.tolist() for p in ph]
    out["lambda"] = lam
    out["share"] = [float(share[k]) for k in range(K)]
    out["w_range"] = [float(wgrid[0]), float(wgrid[-1])]
    out["q_range"] = [round(float(3 * math.exp(wgrid[0])), 3), round(float(3 * math.exp(wgrid[-1])), 3)]
    out["_phi_grid"] = [np.interp(wfull, wgrid, comps[k]).tolist() for k in range(K)]   # 전체 격자 위 (밖은 끝값 상수)
    out["_cov"] = cov
    return out


def lnmc_prior(fits, seasons):
    lm = np.array([math.log(fits[s][1]) for s in seasons])
    J = len(lm)
    if J < 2:
        return {"mean": float(lm.mean()) if J else 0.0, "var": 0.3 ** 2, "raw_var": None, "noise_var": None, "J": J}
    rv = float(np.var(lm, ddof=1))
    nv = float(np.mean([(fits[s][3] / fits[s][1]) ** 2 for s in seasons]))
    return {"mean": float(lm.mean()), "var": max(rv - nv, 0.01 ** 2), "raw_var": rv, "noise_var": nv, "J": J}


def make_shape(pts, train, ridge=RIDGE, horiz=True, cover=True, base=None):
    if base is None:
        g, fits, nit = fit_shape(pts, train, ridge, horiz=horiz)
    else:
        g, fits, nit = base["g"], base["fits"], base["nit"]     # G·기울기·lnmc 는 그대로, 주성분만 다시
    wgrid = dev_grid()
    pcs = build_pcs(pts, train, fits, g, wgrid, horiz, cover)
    return {"g": g, "fits": fits, "nit": nit, "pcs": pcs, "wgrid": wgrid, "horiz": horiz}


def G_of(g, w):
    return basis(w) @ g


def cut_obj(sh, train):
    g, fits, pcs = sh["g"], sh["fits"], sh["pcs"]
    pcs_out = {k: v for k, v in pcs.items() if not k.startswith("_")}
    return {"train": list(train), "q_knots": Q_KNOTS, "w_knots": W_KNOTS, "slopes": [float(v) for v in g],
            "wg": WG.tolist(), "G": G_of(g, WG).tolist(), "lnmc_prior": lnmc_prior(fits, train),
            "train_fit": {s: {"a": fits[s][0], "lnmc": math.log(fits[s][1]), "se_lnmc": fits[s][3] / fits[s][1], "n_pts": None} for s in train},
            "pcs": pcs_out}


def recon_table(pts, seasons, sh, horiz=True):
    """시즌마다 PCHIP 대비 최대 |Δ ln 점수| (0.5~15% · 15~40%) — 공통 꼴 판(적합 a) · 공통 꼴 판(a = PCHIP 3%) · 편차 층 더한 판."""
    g, fits, pcs = sh["g"], sh["fits"], sh["pcs"]
    K = pcs["K"]
    horiz = sh.get("horiz", True)
    rows = {}
    for s in seasons:
        pc, lo, hi = pts[s]["pc"], pts[s]["lo"], pts[s]["hi"]
        a, m = fits[s][0], fits[s][1]
        r = {}
        have3 = lo - 1e-12 <= math.log10(3.0) <= hi + 1e-12
        # 편차 층: 그 시즌 잔차를 주성분 위로 최소제곱 투영
        xi = np.zeros(K)
        if K and have3:
            wgrid = sh["wgrid"]
            x = np.log10(3.0 * np.exp(wgrid))
            ok = (x >= lo - 1e-9) & (x <= hi + 1e-9) & pcs["_cov"]
            Gg = G_of(g, wgrid)
            e = (float(pc(math.log10(3.0))) - pc(np.clip(x, lo, hi))) - m * Gg
            Phi = np.array(pcs["_phi_grid"])[:, ok].T
            sqw = (1.0 / (slope_at(g, wgrid) * LN10))[ok] if horiz else np.ones(int(ok.sum()))
            xi = np.linalg.lstsq(Phi * sqw[:, None], e[ok] * sqw, rcond=None)[0]
        for name, (qa, qb) in {"mid": (0.5, 15.0), "hi": (15.0, 40.0)}.items():
            xa, xb = max(math.log10(qa), lo), min(math.log10(qb), hi)
            if xb <= xa:
                r[name] = None
                continue
            xx = np.linspace(xa, xb, 400)
            w = np.log(10 ** xx / 3.0)
            ref = pc(xx)
            Gv = G_of(g, w)
            dref = np.maximum(np.abs(pc.derivative()(xx)), 1e-3)     # ln 점수 per log10 %
            e1 = np.abs(a - m * Gv - ref)
            e2 = e3 = None
            if have3:
                a3 = float(pc(math.log10(3.0)))
                e2 = np.abs(a3 - m * Gv - ref)
                if K:
                    ph = np.array([np.interp(w, sh["wgrid"], np.array(pcs["_phi_grid"])[k]) for k in range(K)])
                    e3 = np.abs(a3 - m * Gv - xi @ ph - ref)
                else:
                    e3 = e2
            mx = lambda e, f=1.0: None if e is None else float((e / f).max())  # noqa: E731
            r[name] = {"fit_a": mx(e1), "a_pchip3": mx(e2), "with_dev": mx(e3),
                       "h_fit_a": mx(e1, dref), "h_a_pchip3": mx(e2, dref), "h_with_dev": mx(e3, dref)}
        rows[s] = r
    return rows


def summarize_recon(rows):
    out = {}
    for rng in ("mid", "hi"):
        for key in ("fit_a", "a_pchip3", "with_dev", "h_fit_a", "h_a_pchip3", "h_with_dev"):
            v = [rows[s][rng][key] for s in rows if rows[s].get(rng) and rows[s][rng].get(key) is not None]
            out[f"{rng}_{key}"] = {"n": len(v), "median": float(np.median(v)) if v else None, "max": float(np.max(v)) if v else None}
    return out


def stability(g_s, g_full):
    r = {}
    ww = WG
    qq = 3.0 * np.exp(ww)
    sel = (qq >= 0.5) & (qq <= 40.0)
    Gs, Gf = G_of(g_s, ww), G_of(g_full, ww)
    mh = float((Gs[sel] @ Gf[sel]) / (Gs[sel] @ Gs[sel]))
    r["m_hat"] = mh
    for name, (qa, qb) in {"mid": (0.5, 15.0), "hi": (15.0, 40.0)}.items():
        s2 = (qq >= qa) & (qq <= qb)
        r[f"max_dG_{name}"] = float(np.abs(mh * Gs[s2] - Gf[s2]).max())
    r["max_dslope"] = float(np.abs(g_s - g_full).max())
    r["max_dslope_scaled"] = float(np.abs(mh * g_s - g_full).max())
    return r


def git_head():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def self_check(sh, name):
    g, pcs = sh["g"], sh["pcs"]
    res = {}
    wd = np.linspace(-8, 6, 5000)
    res["G_monotone"] = bool(np.all(np.diff(G_of(g, wd)) > 0))
    res["G0"] = float(G_of(g, np.array([0.0]))[0])
    res["slopes_positive"] = bool(np.all(g > 0))
    fits = sh["fits"]
    res["geomean_m"] = float(math.exp(np.mean([math.log(f[1]) for f in fits.values()])))
    if pcs["K"]:
        i0 = int(np.argmin(np.abs(WG)))
        res["phi_at0"] = [float(np.interp(0.0, WG, np.array(p))) for p in pcs["phi"]]
    # 점검 3: m = 1, |xi_k| <= 2 sqrt(lambda_k) 네 귀퉁이에서 R = G + Σ xi phi 가 wg 위에서 증가하나 (깨지면 보고만)
    K = pcs["K"]
    viol = []
    if K:
        import itertools
        Gw = G_of(g, WG)
        for sg in itertools.product([-1, 1], repeat=K):
            R = Gw + sum(sg[k] * 2 * math.sqrt(pcs["lambda"][k]) * np.array(pcs["phi"][k]) for k in range(K))
            bad = np.where(np.diff(R) <= 0)[0]
            if len(bad):
                viol.append({"corner": list(sg), "q_from": round(float(3 * math.exp(WG[bad[0]])), 2), "q_to": round(float(3 * math.exp(WG[bad[-1] + 1])), 2), "n_steps": int(len(bad))})
    res["R_incr_corners_ok"] = not viol
    res["R_incr_violations"] = viol
    ok = res["G_monotone"] and abs(res["G0"]) < 1e-12 and res["slopes_positive"] and abs(res["geomean_m"] - 1) < 1e-6 and all(abs(v) < 1e-6 for v in res.get("phi_at0", []))
    res["ok"] = bool(ok)
    return res


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ridge", type=float, default=RIDGE)
    ap.add_argument("--share", type=float, default=0.9, help="편차 층 PC 몫 기준 (기본 0.9)")
    ap.add_argument("--kmax", type=int, default=2, help="편차 층 PC 수 상한 (기본 2)")
    a = ap.parse_args()
    global PC_SHARE, PC_KMAX
    PC_SHARE, PC_KMAX = a.share, a.kmax
    ridge = a.ridge
    pts = season_points()
    for s in SEASONS:
        pts[s]["snum"] = int(s[1:])
    # ---- full
    full = make_shape(pts, SEASONS, ridge)
    rec_common = recon_table(pts, SEASONS, full)
    gfull = full["g"]
    F = cut_obj(full, SEASONS)
    for s in SEASONS:
        F["train_fit"][s]["n_pts"] = pts[s]["n"]
    F["slopes_log10"] = [float(v * LN10) for v in gfull]
    ref = {}
    for q, v in H4L_REF.items():
        i = int(np.searchsorted(np.array(Q_KNOTS), q, side="right"))   # q가 꺾는 자리면 q에서 시작하는 조각
        ref[str(q)] = {"h4l": v, "ours": float(gfull[i] * LN10), "piece": i}
    F["h4l_compare"] = ref
    F["recon"] = {"per_season": rec_common, "summary": summarize_recon(rec_common)}
    F["self_check"] = self_check(full, "full")
    F["n_seasons_used_by_piece"] = [int(sum(1 for s in SEASONS if pts[s]["w"].min() < (EDGES[i + 1]) and pts[s]["w"].max() > EDGES[i])) for i in range(8)]
    v2 = make_shape(pts, SEASONS, ridge, cover=False, base=full)
    F["v2_pca_full_grid"] = {"recon_summary": summarize_recon(recon_table(pts, SEASONS, v2)),
                             "pcs": {k: v for k, v in v2["pcs"].items() if k in ("K", "share", "lambda", "w_range", "mean_resid_max_abs_n5")}}
    # v1_vertical: 1차 세로(ln 점수) 오차 맞춤 — 견주기용
    v1 = make_shape(pts, SEASONS, ridge, horiz=False, cover=False)
    v1r = recon_table(pts, SEASONS, v1)
    F["v1_vertical"] = {"slopes": [float(x) for x in v1["g"]], "slopes_log10": [float(x * LN10) for x in v1["g"]],
                        "recon_summary": summarize_recon(v1r),
                        "pcs": {k: v for k, v in v1["pcs"].items() if k in ("K", "share", "lambda", "mean_resid_max_abs_n5")},
                        "lnmc_prior": lnmc_prior(v1["fits"], SEASONS)}
    # 범위 진단: 0.5~15% 만으로 맞추면 (WP12 H1 범위) 공통 꼴 중앙은?
    p5 = season_points(0.5, 15.0)
    g5, f5, _ = fit_shape(p5, SEASONS, ridge)
    r5 = recon_table(p5, SEASONS, {"g": g5, "fits": f5, "pcs": {"K": 0}, "wgrid": dev_grid()})
    v5 = [r5[s]["mid"]["fit_a"] for s in SEASONS if r5[s].get("mid")]
    F["diag_fit_range_0.5_15"] = {"mid_fit_a_median": float(np.median(v5)), "mid_fit_a_max": float(np.max(v5)),
                                  "note": "같은 모형을 점 0.5~15%만으로 맞춤 (WP12 H1 범위) — 재현 점검 0.5~15% 중앙. 15% 위 조각은 자료가 없어 의미 없음"}
    # 무벌점 비교
    nr = make_shape(pts, SEASONS, 0.0)
    F["ridge_effect"] = {"ridge": ridge, "max_dslope_vs_ridge0": float(np.abs(nr["g"] - gfull).max()),
                         "max_dG_vs_ridge0": float(np.abs(G_of(nr["g"], WG) - G_of(gfull, WG))[(3 * np.exp(WG) >= 0.1) & (3 * np.exp(WG) <= 40)].max())}
    # ---- targets
    T = {}
    for s in SEASONS:
        n = int(s[1:])
        if n < 21:
            continue
        train = [x for x in SEASONS if int(x[1:]) < n]
        sh = make_shape(pts, train, ridge)
        c = cut_obj(sh, train)
        for x in train:
            c["train_fit"][x]["n_pts"] = pts[x]["n"]
        Gs = G_of(sh["g"], pts[s]["w"])
        a_t, m_t, _, _, _ = ls_am(basis(pts[s]["w"]) @ sh["g"], pts[s]["y"])
        c["true_fit"] = {"a": a_t, "lnmc": math.log(m_t), "n_pts": pts[s]["n"]}
        c["stability"] = stability(sh["g"], gfull)
        c["self_check"] = self_check(sh, s)
        v2t = make_shape(pts, train, ridge, cover=False, base=sh)
        c["v2_pca_full_grid"] = {"pcs": {k: v for k, v in v2t["pcs"].items() if k in ("K", "share", "lambda", "w_range")}}
        sv = make_shape(pts, train, ridge, horiz=False, cover=False)
        c["v1_vertical"] = {"slopes": [float(x) for x in sv["g"]], "lnmc_prior": lnmc_prior(sv["fits"], train),
                            "pcs": {k: v for k, v in sv["pcs"].items() if k in ("K", "share", "lambda")},
                            "stability": stability(sv["g"], v1["g"])}
        T[s] = c
    for x in list(F["pcs"].keys()):
        pass
    secs = round(time.time() - t0, 1)
    rules = {"좌표": "y = ln 점수, w = ln(q/3); ln C = a − m·G(w); 점 0.1~40%, 시즌 무게 같게, m 기하평균 1",
             "능선 벌점": f"끝 조각 (g0−g1, g6−g5, g7−g6) 제곱차 × {ridge:g} (시즌당 총 자료 무게 1). 24시즌 판에서 벌점 0과 비교: header 아래 full.ridge_effect",
             "맞추기(2차)": "가로 오차(log10 %) IRLS: 점 무게 1/s_i², s_i = m_j·G'(w_i)·ln10 (직전 g·m_j로 셈, 꺾는 자리는 양쪽 평균), 시즌 무게 1/n_j. 1차 세로 맞춤은 full.v1_vertical·targets.*.v1_vertical",
             "편차 층(2차)": "무게 있는 PCA: 칸 무게 W(w) = 1/(G'(w)·ln10)² (m 기하평균 1), Z = e·√W 를 EM-PCA 한 뒤 phi = v/√W 로 세로(ln 점수) 단위로 되돌림, 점수 = Z 투영(= 세로 phi 계수), K 규칙은 무게 있는 고윳값 몫. 재현 점검의 투영도 같은 무게 최소제곱. 재현 점검은 세로 |Δ ln 점수| 와 가로 |Δ log10 %| (= 점별 세로 차 ÷ PCHIP 기울기) 둘 다.",
             "편차 층": f"PCHIP 잔차 e_j(w) = [lnC(3%) − lnC(w)] − m_j G(w), 격자 q 0.1~40% 로그 균일 40점 + w=0, 결측 PCA(EM), 몫 {a.share:.0%}에 처음 닿는 수 K ≤ {a.kmax} (--share {a.share:g} --kmax {a.kmax}), K ≤ max(J−2,0). lambda = 점수 표본 분산 (잡음 안 뺌).",
             "공개": "글 번호·원문·닉네임·원글 주소·사람 열쇠·사진 열쇠 없음 (시각·%·점수·개수만)"}
    notes = ["계획대로. 구현 선택: 능선 벌점 크기·부호 고정·재현 점검의 투영 방법은 rules/md 참고."]
    J = {"header": {"about": "WP13a 컷 꼴 G (앞 시즌만) — 서술만", "data_ver": C.DATA_VER,
                    "made": {"git_head": git_head(), "cmd": "python3 -I tools/solo_raid/lab7/wp13_cut_shape.py --out " + a.out + ("" if (a.share, a.kmax) == (0.9, 2) else f" --share {a.share:g} --kmax {a.kmax}"),
                             "date": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d")},
                    "secs": secs, "rules": rules, "notes": notes},
         "full": F, "targets": T}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=1)
    write_md(J, a.out[:-5] + ".md" if a.out.endswith(".json") else a.out + ".md")
    print("쓴 파일", a.out, os.path.getsize(a.out), "바이트", secs, "초")


def fmt(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def write_md(J, path):
    F, T = J["full"], J["targets"]
    L = ["# WP13a 컷 꼴 G (cut_shape_D1) — 요약", "", f"자료 판 {J['header']['data_ver']} · 코드 {J['header']['made']['git_head']} · {J['header']['made']['date']} · {J['header']['secs']}초", "",
         "ln C(q) = a_j − m_j·G(w), w = ln(q/3). 꺾는 자리 1·3·5·10·15·20·30%, 점 0.1~40%, 시즌 무게 같게, 학습 m 기하평균 1. 2차: 가로 오차(log10 %) IRLS 맞춤 · 무게 있는 PCA.", ""]
    S = F["recon"]["summary"]
    L += ["## 재현 점검 (24시즌 전체 적합, PCHIP 대비 시즌별 최대 |Δ ln 점수|)", "", "| 판 | 범위 | 시즌 수 | 시즌 최대 중앙 | 시즌 최대의 최대 |", "|---|---|---|---|---|"]
    names = {"fit_a": "공통 꼴 (a 적합)", "a_pchip3": "공통 꼴 (a = PCHIP 3%)", "with_dev": "공통 꼴 + 편차 층 (a = PCHIP 3%)"}
    L[-2:] = ["| 판 | 단위 | 범위 | 시즌 수 | 시즌 최대 중앙 | 시즌 최대의 최대 | 1차(세로 맞춤) 중앙 / 최대 |", "|---|---|---|---|---|---|---|"]
    S1 = F["v1_vertical"]["recon_summary"]
    for k in ("fit_a", "a_pchip3", "with_dev"):
        for pre, un in (("", "세로 ln점수"), ("h_", "가로 log10 %")):
            for rng, lab in (("mid", "0.5~15%"), ("hi", "15~40%")):
                v, v1_ = S[f"{rng}_{pre}{k}"], S1[f"{rng}_{pre}{k}"]
                L.append(f"| {names[k]} | {un} | {lab} | {v['n']} | {fmt(v['median'])} | {fmt(v['max'])} | {fmt(v1_['median'])} / {fmt(v1_['max'])} |")
    L += ["", "기준: WP12 H1 공통 꼴 시즌 최대 중앙 0.022 (0.5~15%).", ""]
    pc = F["pcs"]
    L += [f"편차 층 (24시즌): K = {pc['K']}, 몫 {[round(x, 3) for x in pc['share']]}, 모든 몫 {pc['share_all']}, lambda {[round(x, 5) for x in pc['lambda']]}, 평균 잔차 최대 |값| {fmt(pc['mean_resid_max_abs'], 4)} (격자 칸이 5시즌 이상인 곳만 {fmt(pc['mean_resid_max_abs_n5'], 4)}; 0.005 넘는 q % {pc['mean_resid_over_0.005_q']}), 3%가 점 범위 밖이라 뺀 시즌 {pc['skipped_no_3pct']}.", ""]
    L += ["## 기울기 (24시즌 전체)", "", "| 조각 | 구간 | g = dG/dw | 단위 맞춤 (−d ln점수/d log10 %) | 쓰인 시즌 |", "|---|---|---|---|---|"]
    qk = Q_KNOTS
    labs = ["<1%"] + [f"{qk[i]:g}~{qk[i+1]:g}%" for i in range(6)] + [">30%"]
    for i in range(8):
        L.append(f"| {i} | {labs[i]} | {F['slopes'][i]:.3f} | {F['slopes_log10'][i]:.3f} | {F['n_seasons_used_by_piece'][i]} |")
    L += ["", "WP12 H4L 기울기와 같은 단위 (q에서 시작하는 조각의 값):", "", "| q % | H4L | 이번 | 조각 |", "|---|---|---|---|"]
    for q, v in F["h4l_compare"].items():
        L.append(f"| {q} | {v['h4l']:.2f} | {v['ours']:.2f} | {v['piece']} |")
    dg = F["diag_fit_range_0.5_15"]
    L += ["", f"범위 진단: 같은 모형을 점 0.5~15%만으로 맞추면 0.5~15% 공통 꼴 시즌 최대 중앙 {dg['mid_fit_a_median']:.3f} (최대 {dg['mid_fit_a_max']:.3f}) — H1 재현. 범위를 0.1~40%로 넓히면 15% 위 자료가 단일 배수 m_j 꼴을 끌어 중앙이 올라간다.", ""]
    re_ = F["ridge_effect"]
    L += ["", f"능선 벌점 {re_['ridge']:g}: 벌점 0 판과 최대 기울기 차 {re_['max_dslope_vs_ridge0']:.4f}, 0.1~40% 최대 |ΔG| {re_['max_dG_vs_ridge0']:.4f}.", "", f"자체 점검 (full): {F['self_check']}", ""]
    L += ["## 맞힐 시즌별 (앞 시즌만)", "", "| 시즌 | J | lnmc 사전 sd | K | 주성분 몫 | 안정성 최대 ΔG 0.5~15% | 15~40% | 기울기 차 최대 | 점검 |", "|---|---|---|---|---|---|---|---|---|"]
    for s, c in T.items():
        pr = c["lnmc_prior"]
        st = c["stability"]
        L.append(f"| {s} | {pr['J']} | {math.sqrt(pr['var']):.3f} | {c['pcs']['K']} | {[round(x, 2) for x in c['pcs']['share']]} | {st['max_dG_mid']:.3f} | {st['max_dG_hi']:.3f} | {st['max_dslope']:.2f} | R증가 {'ok' if c['self_check']['R_incr_corners_ok'] else 'FAIL'} · 구간 q {c['pcs'].get('q_range')} · {'ok' if c['self_check']['ok'] else 'FAIL'} (1차 sd {math.sqrt(c['v1_vertical']['lnmc_prior']['var']):.3f}, K {c['v1_vertical']['pcs']['K']}) |")
    L += ["", "R = G + Σ ξφ 증가 점검(점검 3) 깨진 곳: " + (str({s: c["self_check"]["R_incr_violations"] for s, c in T.items() if not c["self_check"]["R_incr_corners_ok"]}) or "없음"), ""]
    S2 = F["v2_pca_full_grid"]["recon_summary"]
    L += ["## 2차(격자 전체 PCA, v2_pca_full_grid) 재현 — 편차 층 판 시즌 최대 중앙 / 최대", "", "| 단위 | 범위 | 3차(덮개 구간) | 2차 |", "|---|---|---|---|"]
    for pre, un in (("", "세로"), ("h_", "가로")):
        for rng, lab in (("mid", "0.5~15%"), ("hi", "15~40%")):
            a_, b_ = S[f"{rng}_{pre}with_dev"], S2[f"{rng}_{pre}with_dev"]
            L.append(f"| {un} | {lab} | {fmt(a_['median'])} / {fmt(a_['max'])} | {fmt(b_['median'])} / {fmt(b_['max'])} |")
    L += ["", "안정성: m̂ = 맞힐 시즌 앞 시즌 꼴 G_s를 24시즌 G_full에 0.5~40%에서 최소제곱으로 맞추는 배수, |m̂·G_s − G_full| 최대. true_fit(맞힐 시즌 자기 점으로 a, ln m)은 JSON에 진단용.", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
