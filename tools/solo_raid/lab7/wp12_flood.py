"""WP12 곡선 꼴 조사 — 침수 곡선 A~D (2026-11-04, 2단계) — 서술 계산. 예측·채점·고르기 없음, 24시즌 모두 사용.

  python3 -I tools/solo_raid/lab7/wp12_flood.py --obs-dir W/obs7/full --out data/solo-raid/analysis/r7/wp12/flood_D1.json

계획: docs/solo-raid-method.md 「WP12」 A~D. 공용 표는 wp12_tables.py (읽기만), 공용 코드는 읽기만.
  A 마지막 날 마감 몰림 · B 첫날 · C 일차·시즌마다 차이 · D 점수대·%대별 차이 (+ 숫자단 1~7위 시간 흐름).
방법 (표준 서술 기법만):
  기록 쌍 ΔL = log10(p2/p1) = m_s · Σ_k λ_k · (칸 k와 겹친 시간) — 칸별 비음수 최소제곱 (NNLS, m_s 기하평균 1 번갈아 풀기, wp10_diag.fit_rate·act_clock.fit 꼴).
  1시간 칸은 2차 차분 벌점 (Eilers·Marx 1996) — 벌점 크기는 시즌 하나씩 빼기 교차검증 + 1-SE 규칙 (Hastie 외 ESL 7.10; act_clock과 같은 짝지은 판).
  띠 = 시즌 붓스트랩 (시즌을 복원 추출, 10~90%, B = 200). 시즌 하나·묶음이 작은 줄은 쌍 붓스트랩.
  사진 = A·B 컷 사진의 남은 침수 P_obs = log10(최종 % ÷ 지금 %) (P_tab; 지금 % ≥ 0.3, 최종 % 0.5~15). 주성분 = 결측을 EM으로 채운 SVD (Ramsay·Silverman 꼴).
공개 규칙: 글 번호·원문·닉네임·원글 주소·사람 열쇠(pid·mid)·사진 열쇠를 결과에 쓰지 않는다 (시각·%·점수·개수만).
"""
import argparse
import collections
import datetime as dt
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wp12_tables as T  # noqa: E402
import act_clock as AC  # noqa: E402
import wp10_diag as WD  # noqa: E402
import m3 as M3MOD  # noqa: E402

SEAS = T.SEASONS
SI = T.season_info()
NS = len(SEAS)
SIDX = {s: i for i, s in enumerate(SEAS)}
CLOCK = [s for s in SEAS if SI[s]["clock_ok"]]          # 05시 시작 120시간 21시즌
BOOT = 200
MINN = 15                                               # 쌍·사진이 이보다 적으면 칸에 small 표시
KEXTRA = {}                                             # 더 본 것
NB = 8
BIN_NAMES = [f"{(5 + 3 * k) % 24:02d}-{(8 + 3 * k) % 24:02d}" for k in range(NB)]
BANDS5 = ((0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 40.0))
PF_LO, PF_HI, PCT_LO = 0.5, 15.0, 0.3


def r4(x, d=4):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, d)


def rl(v, d=4):
    return [r4(x, d) for x in v]


def q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if len(v) else float("nan")


def band(draws):
    """붓스트랩 열 (B, K) -> (10%, 90%)."""
    a = np.asarray(draws, float)
    return np.nanpercentile(a, 10, axis=0), np.nanpercentile(a, 90, axis=0)


# ---------------- 쌍·사진 표 (복사본) ----------------

def load_pairs(obs):
    P = [dict(p) for p in T.pairs(obs)]
    n = len(P)
    S = np.array([SIDX[p["season"]] for p in P])
    t1 = np.array([p["t1"] for p in P])
    t2 = np.array([p["t2"] for p in P])
    y = np.array([p["dL"] for p in P])
    var = np.array([p["var"] for p in P])
    p1 = np.array([p["p1"] for p in P])
    E = np.zeros((n, 7, NB))
    for i in range(n):
        s = SEAS[S[i]]
        E[i] = AC.exposure(SI[s]["c0h"], float(SI[s]["H"]), t1[i], t2[i])
    return {"S": S, "t1": t1, "t2": t2, "y": y, "w": 1 / np.sqrt(var), "p1": p1, "E": E, "n": n}


def htrue(s):
    """실제 끝 시각 h (season.json end = KST 04:59:59 → 다음 05시까지; S28은 130.5, 나머지는 H)."""
    i = SI[s]
    return i["H"] + ((5.0 - i["c0h"] - i["H"]) % 24.0)


# ---------------- NNLS (시즌 배수 번갈아) ----------------

def fit_m(X, y, w, si, ns, pen=None, iters=25, tol=1e-5):
    """ΔL = m_s · (X @ lam)/1000. lam >= 0 (단위 1e-3 log10/시간), m 기하평균 1. pen = 이미 √벌점을 곱한 (R, K) 행렬 (lam에 곱해 0을 끌어댐)."""
    m = np.ones(ns)
    lam = np.zeros(X.shape[1])
    Xs = X / 1000.0
    pen_s = None if pen is None else pen
    for _ in range(iters):
        sc = (m[si] * w)[:, None]
        A = Xs * sc
        b = y * w
        if pen_s is not None:
            A = np.vstack([A, pen_s])
            b = np.concatenate([b, np.zeros(len(pen_s))])
        lam, _ = nnls(A, b, maxiter=50 * X.shape[1])
        base = Xs @ lam
        den = np.bincount(si, (base * w) ** 2, ns)
        num = np.bincount(si, y * base * w * w, ns)
        ok = den > 0
        mn = m.copy()
        mn[ok] = np.maximum(num[ok] / den[ok], 1e-3)
        gm = math.exp(np.mean(np.log(mn[ok]))) if ok.any() else 1.0
        mn /= gm
        lam = lam * gm
        done = np.max(np.abs(mn - m)) < tol
        m = mn
        if done:
            break
    return lam, m


def boot_idx(Sarr, seasons, rng):
    """시즌을 복원 추출 -> (행 번호, 새 시즌 번호, 시즌 수)."""
    by = {s: np.where(Sarr == s)[0] for s in seasons}
    dr = rng.integers(0, len(seasons), len(seasons))
    rows = [by[seasons[d]] for d in dr]
    idx = np.concatenate(rows)
    si = np.concatenate([np.full(len(r), j) for j, r in enumerate(rows)])
    return idx, si, len(seasons)


def d2_penalty(K, a, b, kappa):
    """열 a..b-1 (b-a개)에 2차 차분 벌점 행 (√kappa)."""
    n = b - a
    R = np.zeros((n - 2, K))
    for i in range(n - 2):
        R[i, a + i], R[i, a + i + 1], R[i, a + i + 2] = 1.0, -2.0, 1.0
    return math.sqrt(kappa) * R


# ---------------- A: 마지막 날 쌍 NNLS ----------------

NE = 18                                                  # 앞쪽 6시간 칸 (S28 106.5시간까지 18칸)
NL = 24                                                  # 마지막 날 1시간 칸 (남은 시간 x ∈ [j, j+1])


def build_A(D, rows=None):
    """열: 앞쪽 6시간 칸 NE개 (h 축, 칸 [6k, 6k+6] ∩ [0, Ht−24]) + 마지막 날 1시간 칸 24개 (남은 시간 축 x = Ht − h)."""
    idx = np.arange(D["n"]) if rows is None else rows
    X = np.zeros((len(idx), NE + NL))
    for r, i in enumerate(idx):
        s = SEAS[D["S"][i]]
        Ht = htrue(s)
        a, b = D["t1"][i], D["t2"][i]
        lim = Ht - 24.0
        for k in range(NE):
            lo, hi = 6.0 * k, min(6.0 * k + 6.0, lim)
            if hi > lo:
                X[r, k] = max(0.0, min(hi, b) - max(lo, a))
        for j in range(NL):
            lo, hi = Ht - j - 1.0, Ht - j
            X[r, NE + j] = max(0.0, min(hi, b) - max(lo, a))
    return X


def drop_cols(X):
    keep = np.where(X.sum(0) > 0)[0]
    return keep


def fit_A(XA, y, w, si, ns, kappa, keep=None):
    keep = drop_cols(XA) if keep is None else keep
    X = XA[:, keep]
    last = np.where(keep >= NE)[0]
    pen = None
    if kappa > 0 and len(last) >= 3:
        K = len(keep)
        pen = d2_penalty(K, last[0], last[-1] + 1, kappa)
    lam, m = fit_m(X, y, w, si, ns, pen)
    full = np.zeros(XA.shape[1])
    full[keep] = lam
    return full, m


def cv_A(XA, D, rows_all, kappas):
    """시즌 하나씩 빼기 -> 벌점마다 (합, 시즌별 오차)."""
    S = D["S"][rows_all]
    seasons = sorted(set(S.tolist()))
    y, w = D["y"][rows_all], D["w"][rows_all]
    out = {k: [] for k in kappas}
    for sv in seasons:
        tr = S != sv
        te = ~tr
        sit = np.unique(S[tr], return_inverse=True)[1]
        keep = drop_cols(XA[tr])
        for k in kappas:
            lam, _ = fit_A(XA[tr], y[tr], w[tr], sit, sit.max() + 1, k, keep)
            base = XA[te] @ lam / 1000.0
            den = ((base * w[te]) ** 2).sum()
            mo = (y[te] * base * w[te] ** 2).sum() / den if den > 0 else 1.0
            out[k].append(float((((y[te] - mo * base) * w[te]) ** 2).sum()))
    return out


def one_se(cv, kappas):
    tot = {k: sum(cv[k]) for k in kappas}
    kmin = min(tot, key=tot.get)
    f0 = np.array(cv[kmin])
    sed = {k: float(np.std(np.array(cv[k]) - f0, ddof=1) * math.sqrt(len(f0))) for k in kappas}
    ok = [k for k in kappas if tot[k] - tot[kmin] <= sed[k]]
    return max(ok), kmin, tot, sed


def a_stats(lam24):
    """마지막 날 1시간 칸 rate (j=0 마지막 시간) -> 합(log10/일), 몫, 다시 오르는 시각."""
    lam = np.asarray(lam24, float) / 1000.0
    tot = lam.sum()
    if tot <= 0:
        return {"total": 0.0, "s6": np.nan, "s3": np.nan, "s1": np.nan, "xmin": np.nan, "xrise": np.nan}
    jmin = int(np.argmin(lam[::-1]))                       # 시간 순서 (j=23 → 0)에서 첫 최소
    jmin = 23 - jmin
    lmin = lam[jmin]
    eps = 0.05 * lam.mean()
    thr = 2 * max(lmin, eps)
    xr = np.nan
    for j in range(jmin - 1, -1, -1):
        if lam[j] > thr:
            xr = j + 1.0
            break
    return {"total": tot, "s6": lam[:6].sum() / tot, "s3": lam[:3].sum() / tot, "s1": lam[0] / tot, "xmin": jmin + 0.5, "xrise": xr}


def kst_of_x(x):
    return (5.0 - x) % 24.0


# ---------------- 사진 표 ----------------

def build_photos(obs, prof, s1_M3, s1_act):
    """A·B·C 사진 줄 + 줄마다 F·M3 기본·act 곡선 (그 줄의 시즌·시각·최종 %로 셈한 남은 침수)."""
    ph = T.photos(obs)
    n = len(ph)
    S = np.array([SIDX[p["season"]] for p in ph])
    kind = np.array([p["kind"] for p in ph])
    h = np.array([p["h"] for p in ph], float)
    pct = np.array([p["pct"] for p in ph], float)
    pf = np.array([np.nan if p["pf_tab"] is None else p["pf_tab"] for p in ph], float)
    Pt = np.array([np.nan if p["P_tab"] is None else p["P_tab"] for p in ph], float)
    Ht = np.array([htrue(SEAS[i]) for i in S])
    x = Ht - h
    F = np.full(n, np.nan)
    Mb = np.full(n, np.nan)
    Ma = np.full(n, np.nan)
    for si_, s in enumerate(SEAS):
        ix = np.where(S == si_)[0]
        if not len(ix):
            continue
        H = float(SI[s]["H"])
        u = np.clip(h[ix] / H, 0, 1)
        F[ix] = T.F_P(u)
        pfi = np.where(np.isfinite(pf[ix]), pf[ix], 3.0)
        Mb[ix] = m3_curve(s1_M3, s, u, pfi)
        us = AC.ustar(prof, s, h[ix], H)
        Ma[ix] = m3_curve(s1_act, s, us, pfi)
    main = (np.isin(kind, ["A", "B"]) & np.isfinite(Pt) & (pct >= PCT_LO) & (pf >= PF_LO) & (pf <= PF_HI) & (h >= 0) & (x >= -0.01))
    return {"n": n, "S": S, "kind": kind, "h": h, "x": x, "pct": pct, "pf": pf, "P": Pt, "F": F, "Mb": Mb, "Ma": Ma, "main": main, "Ht": Ht}


def load_s1(path):
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    return j["header"]["names"], j["seasons"]


def m3_curve(s1, s, u, pf):
    """wp12_tables.m3_P와 같은 numpy 식 (다른 1단계 파일을 받는다)."""
    nm, ss = s1
    B = M3MOD.BASIS
    mean = ss[s]["mean"]
    nl = B["Bl12"].shape[0]
    cl = np.linalg.solve(B["Bl12"], np.array([mean[nm.index(f"l{i}")] for i in range(nl)]))
    beta = mean[nm.index("beta")]
    f = np.exp(B["Blf"] @ cl) * 2 * B["sg"]
    seg = 0.5 * (f[1:] + f[:-1]) * B["ds"]
    Pg = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return np.interp(np.sqrt(np.maximum(np.asarray(u, float), 0.02)), B["sg"], Pg) * (np.asarray(pf, float) / 3.0) ** (beta / math.log(10))


def pstat(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0, "med": None, "q25": None, "q75": None}
    return {"n": int(len(v)), "med": r4(np.median(v)), "q25": r4(np.percentile(v, 25)), "q75": r4(np.percentile(v, 75))}


def med(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


# ---------------- 본 계산 ----------------

def part_A(D, PH, rng, notes, log):
    t0 = time.time()
    allr = np.arange(D["n"])
    XA = build_A(D)
    kappas = [0.0, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5]
    cv = cv_A(XA, D, allr, kappas)
    kbest, kmin, tot, sed = one_se(cv, kappas)
    log(f"A cv {time.time() - t0:.0f}s kbest {kbest} kmin {kmin}")
    y, w, S = D["y"], D["w"], D["S"]
    sit = np.unique(S, return_inverse=True)[1]
    ns = sit.max() + 1
    lam, m = fit_A(XA, y, w, sit, ns, kbest)
    lam_hi, _ = fit_A(XA, y, w, sit, ns, kbest * 10 if kbest > 0 else 1.0)
    lam_lo, _ = fit_A(XA, y, w, sit, ns, kbest * 0.1)
    lam_0, _ = fit_A(XA, y, w, sit, ns, 0.0)
    # 쌍이 건드린 칸 수·겹친 시간
    touch = (XA > 0).sum(0)
    expo = XA.sum(0)
    # 붓스트랩 (시즌 24, 쌍 24시즌)
    seasons = sorted(set(S.tolist()))
    bl = []
    for _ in range(BOOT):
        idx, si, n = boot_idx(S, seasons, rng)
        l, _ = fit_A(XA[idx], y[idx], w[idx], si, n, kbest)
        bl.append(l)
    bl = np.array(bl)
    lo, hi = band(bl)
    log(f"A boot {time.time() - t0:.0f}s")
    return {"XA": XA, "lam": lam, "m": m, "lam_hi": lam_hi, "lam_lo": lam_lo, "lam_0": lam_0, "bl": bl, "lo": lo, "hi": hi, "touch": touch, "expo": expo,
            "kbest": kbest, "kmin": kmin, "cv_tot": tot, "cv_sed": sed, "kappas": kappas}


def fit_C(E, y, w, si, ns, days=5):
    """일차 × 3시간 칸 NNLS (열 = days × 8, 일차 순서). -> (lam (days, 8) 단위 1e-3 log10/시간, m)."""
    X = E[:, :days, :].reshape(len(y), days * NB)
    lam, m = fit_m(X, y, w, si, ns)
    return lam.reshape(days, NB), m


def part_C_boot(D, rng):
    """21 시즌 (05시 시작 120시간) 붓스트랩: C (일차 × 3시간 칸)와 A21 (마지막 날 1시간 칸, 앞쪽 6시간 칸)을 같은 추출에서."""
    S, y, w, E = D["S"], D["y"], D["w"], D["E"]
    cl = [SIDX[s] for s in CLOCK]
    rows = np.where(np.isin(S, cl))[0]
    return rows


def day_summary(lam):
    """lam (5,8) -> 일차 합 (log10/일), 몫 (5,8), 6시간 몫 (5,4)."""
    T_d = 3.0 * lam.sum(1) / 1000.0
    sh = lam / np.maximum(lam.sum(1, keepdims=True), 1e-12)
    sh6 = np.stack([sh[:, 2 * k] + sh[:, 2 * k + 1] for k in range(4)], 1)
    return T_d, sh, sh6


def lines_of(draws, fn):
    """붓스트랩 추출마다 fn -> 값 배열 (B, ...)."""
    return np.array([fn(d) for d in draws])


def season_blocks(D, s, Ht):
    """시즌 s 쌍의 3 블록 노출 [0, Ht−24) · [Ht−24, Ht−6) · [Ht−6, Ht]."""
    ix = np.where(D["S"] == SIDX[s])[0]
    edges = [(0.0, Ht - 24.0), (Ht - 24.0, Ht - 6.0), (Ht - 6.0, Ht)]
    X = np.array([[max(0.0, min(hi, D["t2"][i]) - max(lo, D["t1"][i])) for lo, hi in edges] for i in ix]).reshape(len(ix), 3)
    return ix, X


def share6(mu, Ht=None):
    den = 18 * mu[1] + 6 * mu[2]
    return float(6 * mu[2] / den) if den > 0 else float("nan")


# ---------------- 숫자단 1~7위 ----------------

def top7_flow(obs, kinds):
    rows = [r for r in T.top7(obs) if kinds is None or r["kind"] in kinds]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["season"], r["rank"])].append((r["h"], r["score"]))
    grid = np.arange(6, 121, 6.0)
    series = {}
    for (s, rk), L in by.items():
        Ht = htrue(s)
        fin = [sc for h, sc in L if h >= Ht - 6.0]
        if not fin:
            continue
        final = max(fin)
        L = sorted(L)
        hs = np.array([a for a, _ in L])
        cm = np.maximum.accumulate(np.array([b for _, b in L]))
        val = np.full(len(grid), np.nan)
        for gi, g in enumerate(grid):
            k = np.searchsorted(hs, g, side="right") - 1
            if k >= 0:
                val[gi] = math.log(cm[k] / final)
        hit = {}
        for thr in (0.99, 0.999):
            ok = np.where(cm >= thr * final)[0]
            hit[thr] = float(hs[ok[0]]) if len(ok) else None
        series[(s, rk)] = {"val": val, "hit": hit, "n": len(L), "final": final, "H": Ht}
    return rows, series, grid


def top7_tables(rows, series, grid):
    tab = []
    for gi, g in enumerate(grid):
        v = np.array([x["val"][gi] for x in series.values()])
        vv = v[np.isfinite(v)]
        tab.append({"h_to": float(g), "n_series": int(len(vv)), "n_seasons": len({k[0] for k, x in series.items() if np.isfinite(x["val"][gi])}),
                    "med_ln": r4(np.median(vv)) if len(vv) else None, "q25": r4(np.percentile(vv, 25)) if len(vv) else None,
                    "q75": r4(np.percentile(vv, 75)) if len(vv) else None})
    hits = {}
    for thr in (0.99, 0.999):
        h = np.array([x["hit"][thr] for x in series.values() if x["hit"][thr] is not None])
        hr = {}
        for rk in range(1, 8):
            hh = np.array([x["hit"][thr] for k, x in series.items() if k[1] == rk and x["hit"][thr] is not None])
            hr[str(rk)] = {"n": int(len(hh)), "med_h": r4(np.median(hh), 1) if len(hh) else None}
        hits[str(thr)] = {"n_series": int(len(h)), "q10": r4(q(h, 10), 1), "q25": r4(q(h, 25), 1), "med": r4(q(h, 50), 1), "q75": r4(q(h, 75), 1), "q90": r4(q(h, 90), 1),
                          "by_rank": hr}
    cnt = collections.Counter((r["kind"], str(r["server_basis"])) for r in rows)
    return tab, hits, {f"{k[0]}|{k[1]}": v for k, v in sorted(cnt.items())}


# ---------------- 주성분 (사진 P_obs 시즌 곡선) ----------------

def pca_em(Y, M, k=3, iters=200):
    """Y (시즌 × 칸) 결측 M(=True 있음) -> 열 평균을 빼고 결측을 SVD로 채워 가며 주성분. -> (mean, comps, share, scores, filled)."""
    Yf = Y.copy()
    mu = np.nanmean(np.where(M, Y, np.nan), axis=0)
    for j in range(Y.shape[1]):
        Yf[~M[:, j], j] = mu[j]
    for _ in range(iters):
        mu = Yf.mean(0)
        U, s, Vt = np.linalg.svd(Yf - mu, full_matrices=False)
        rec = mu + (U[:, :k] * s[:k]) @ Vt[:k]
        Yn = np.where(M, Y, rec)
        if np.max(np.abs(Yn - Yf)) < 1e-7:
            Yf = Yn
            break
        Yf = Yn
    mu = Yf.mean(0)
    U, s, Vt = np.linalg.svd(Yf - mu, full_matrices=False)
    share = s ** 2 / (s ** 2).sum()
    return mu, Vt, share, U * s, Yf


# ---------------- 단일 시즌 하루 모양 · 분해 ----------------

def fit_shape8(X8, y, w):
    lam, _ = nnls(X8 * w[:, None], y * w)
    return lam


def fit_season(Z, y, w, kind, kappa, iters=8):
    """Z (n,5,8) = 노출 × 전체 λ (1e-3→log10). ΔL ≈ m · Σ t_d g_b Z. g·t는 평균 1 쪽으로 끌어당긴다 (벌점 kappa)."""
    g = np.ones(NB)
    t = np.ones(5)
    m = 1.0
    for _ in range(iters):
        base = np.einsum("ndb,d,b->n", Z, t, g)
        den = ((base * w) ** 2).sum()
        if den > 0:
            m = (y * base * w * w).sum() / den
        if "g" in kind:
            Xg = np.einsum("ndb,d->nb", Z, t) * m
            A = np.vstack([Xg * w[:, None], math.sqrt(kappa) * np.eye(NB)])
            b = np.concatenate([y * w, math.sqrt(kappa) * np.ones(NB)])
            g, _ = nnls(A, b)
            c = g.mean()
            if c > 0:
                g, m = g / c, m * c
            else:
                g = np.ones(NB)
        if "t" in kind:
            Xt = np.einsum("ndb,b->nd", Z, g) * m
            A = np.vstack([Xt * w[:, None], math.sqrt(kappa) * np.eye(5)])
            b = np.concatenate([y * w, math.sqrt(kappa) * np.ones(5)])
            t, _ = nnls(A, b)
            c = t.mean()
            if c > 0:
                t, m = t / c, m * c
            else:
                t = np.ones(5)
    return m, t, g


def pred_season(Z, m, t, g):
    return m * np.einsum("ndb,d,b->n", Z, t, g)


# ================= 구역별 실행 =================

def bin_of_j(j):
    return int(((kst_of_x(j + 0.5) - 5.0) % 24.0) // 3)


def c_boot(D, RA, rng):
    S = D["S"]
    cl = [SIDX[s] for s in CLOCK]
    rows = np.where(np.isin(S, cl))[0]
    E, y, w = D["E"][rows], D["y"][rows], D["w"][rows]
    sit = np.unique(S[rows], return_inverse=True)[1]
    ns = int(sit.max() + 1)
    lamC, mC = fit_C(E, y, w, sit, ns)
    XA = RA["XA"][rows]
    kb = RA["kbest"]
    lamA, _ = fit_A(XA, y, w, sit, ns, kb)
    Cb, Ab = [], []
    Sr = S[rows]
    for _ in range(BOOT):
        idx, si, n = boot_idx(Sr, cl, rng)
        l, _ = fit_C(E[idx], y[idx], w[idx], si, n)
        a, _ = fit_A(XA[idx], y[idx], w[idx], si, n, kb)
        Cb.append(l)
        Ab.append(a)
    return {"rows": rows, "lamC": lamC, "mC": mC, "lamA": lamA, "Cb": np.array(Cb), "Ab": np.array(Ab), "seasons": [SEAS[i] for i in sorted(set(Sr.tolist()))]}


def run_A(D, PH, RA, CB, rng, plots, log):
    S, y, w = D["S"], D["y"], D["w"]
    out = {"tables": {}, "notes": [], "extra": {}}
    lam, lo, hi, tc, ex = RA["lam"], RA["lo"], RA["hi"], RA["touch"], RA["expo"]
    # 마지막 날 1시간 칸 (시간 순서 j = 23 → 0)
    a21 = CB["lamA"]
    c34 = 0.5 * (CB["lamC"][2] + CB["lamC"][3])
    c34b = 0.5 * (CB["Cb"][:, 2, :] + CB["Cb"][:, 3, :])
    rows = []
    for j in range(NL - 1, -1, -1):
        k = NE + j
        b = bin_of_j(j)
        rat = a21[k] / c34[b] if c34[b] > 0 else None
        rb = CB["Ab"][:, k] / np.where(c34b[:, b] > 0, c34b[:, b], np.nan)
        rows.append({"x_from": j + 1, "x_to": j, "kst": f"{int(kst_of_x(j + 1)):02d}-{int(kst_of_x(j)):02d}", "rate_x1000": r4(lam[k]), "p10": r4(lo[k]), "p90": r4(hi[k]),
                     "k_x10": r4(RA["lam_hi"][k]), "k_x01": r4(RA["lam_lo"][k]), "k_0": r4(RA["lam_0"][k]),
                     "pairs": int(tc[k]), "overlap_h": r4(ex[k], 1), "small": bool(tc[k] < MINN),
                     "A21_rate_x1000": r4(a21[k]), "day34_same_kst_rate_x1000": r4(c34[b]), "crowd_ratio": r4(rat, 3),
                     "crowd_p10": r4(q(rb, 10), 3), "crowd_p90": r4(q(rb, 90), 3), "ratio_unstable": bool(c34[b] < 2.0)})
    out["tables"]["last_day_1h"] = rows
    early = []
    for k in range(NE):
        if ex[k] > 0:
            early.append({"h_from": 6 * k, "h_to": 6 * k + 6, "rate_x1000": r4(lam[k]), "p10": r4(lo[k]), "p90": r4(hi[k]), "pairs": int(tc[k]), "overlap_h": r4(ex[k], 1),
                          "small": bool(tc[k] < MINN)})
    out["tables"]["early_6h"] = early
    # 몫
    st0 = a_stats(lam[NE:])
    bst = [a_stats(b[NE:]) for b in RA["bl"]]
    sens = {nm: a_stats(v[NE:]) for nm, v in (("k_best", lam), ("k_x10", RA["lam_hi"]), ("k_x01", RA["lam_lo"]), ("k_0", RA["lam_0"]))}

    def bq(key, p):
        return q([b[key] for b in bst], p)
    shares = {"total_log10_last_day": r4(st0["total"], 4), "total_p10": r4(bq("total", 10), 4), "total_p90": r4(bq("total", 90), 4)}
    for key, nm in (("s6", "share_H-6_to_H"), ("s3", "share_H-3_to_H"), ("s1", "share_H-1_to_H")):
        shares[nm] = {"value": r4(st0[key], 4), "p10": r4(bq(key, 10), 4), "p90": r4(bq(key, 90), 4)}
    shares["lowest_bin_x_mid"] = r4(st0["xmin"], 1)
    shares["rise_def"] = "마지막 날 24칸 중 속도 최저 칸 뒤(시간 순서)에서 처음 속도 > 2 × max(최저, 마지막 날 평균 × 0.05)가 되는 칸의 시작 (남은 시간)"
    shares["rise_x_hours_left"] = r4(st0["xrise"], 1)
    shares["rise_kst_hour"] = None if not math.isfinite(st0["xrise"]) else r4(kst_of_x(st0["xrise"]), 1)
    shares["rise_boot_p10_med_p90_hours_left"] = [r4(bq("xrise", 10), 1), r4(bq("xrise", 50), 1), r4(bq("xrise", 90), 1)]
    shares["rise_boot_n_found"] = int(sum(math.isfinite(b["xrise"]) for b in bst))
    shares["sensitivity"] = {nm: {"total": r4(v["total"], 4), "s6": r4(v["s6"], 4), "s3": r4(v["s3"], 4), "s1": r4(v["s1"], 4), "xmin": r4(v["xmin"], 1), "rise_x": r4(v["xrise"], 1)}
                             for nm, v in sens.items()}
    shares["n_pairs_touching_last_day"] = int(sum(1 for i in range(D["n"]) if RA["XA"][i, NE:].sum() > 0))
    shares["n_pairs_touching_last_6h"] = int(sum(1 for i in range(D["n"]) if RA["XA"][i, NE:NE + 6].sum() > 0))
    shares["penalty"] = {"kappa_chosen": RA["kbest"], "kappa_cv_min": RA["kmin"], "grid": RA["kappas"], "cv_ss": {str(k): r4(v, 1) for k, v in RA["cv_tot"].items()},
                         "cv_1se": {str(k): r4(v, 1) for k, v in RA["cv_sed"].items()}, "lam_unit": "1e-3 log10 / 시간"}
    out["tables"]["last_day_summary"] = [shares]
    # 몰림 배수 요약: 마지막 6·3·1 시간 평균 속도 ÷ 3~4일차 같은 KST 시각 (A21, C 같은 추출)
    sumr = []
    for nm, js in (("H-6~H", range(0, 6)), ("H-3~H", range(0, 3)), ("H-1~H", range(0, 1)), ("H-24~H", range(0, 24))):
        a = np.mean([a21[NE + j] for j in js])
        c = np.mean([c34[bin_of_j(j)] for j in js])
        ab = np.mean([CB["Ab"][:, NE + j] for j in js], 0)
        cb = np.mean([c34b[:, bin_of_j(j)] for j in js], 0)
        sumr.append({"window": nm, "A21_rate": r4(a), "day34_same_kst_rate": r4(c), "ratio": r4(a / c, 3) if c > 0 else None,
                     "p10": r4(q(ab / np.where(cb > 0, cb, np.nan), 10), 3), "p90": r4(q(ab / np.where(cb > 0, cb, np.nan), 90), 3)})
    out["tables"]["crowd_ratio_windows"] = sumr
    out["tables"]["crowd_ratio_note"] = [{"text": "분자 = 21시즌(05시 시작 120시간, 쌍 1,101) 마지막 날 1시간 칸 NNLS, 분모 = 같은 쌍·같은 붓스트랩 추출의 일차 × 3시간 칸 NNLS 3~4일차 평균. 본안 24시즌 마지막 날 값은 last_day_1h의 rate."}]

    # 사진
    msk = PH["main"] & (PH["x"] < 24.0) & (PH["x"] >= -0.01)
    j = np.floor(np.clip(PH["x"], 0, 23.999)).astype(int)
    prow = []
    for jj in range(NL - 1, -1, -1):
        sel = msk & (j == jj)
        st = pstat(PH["P"][sel])
        mf, mb, ma = med(PH["F"][sel]), med(PH["Mb"][sel]), med(PH["Ma"][sel])
        mp = st["med"]
        dv = lambda c: r4(mp / c, 3) if (mp is not None and c is not None and c > 0.005 and math.isfinite(c)) else None  # noqa: E731
        prow.append({"x_from": jj + 1, "x_to": jj, "n": st["n"], "n_seasons": int(len(set(PH["S"][sel].tolist()))), "P_med": st["med"], "P_q25": st["q25"], "P_q75": st["q75"],
                     "F_med": r4(mf), "M3_med": r4(mb), "act_med": r4(ma), "P_over_F": dv(mf), "P_over_M3": dv(mb), "P_over_act": dv(ma),
                     "P_minus_F": r4(st["med"] - mf) if st["med"] is not None else None, "P_minus_M3": r4(st["med"] - mb) if st["med"] is not None else None,
                     "P_minus_act": r4(st["med"] - ma) if st["med"] is not None else None, "small": bool(st["n"] < MINN)})
    out["tables"]["photos_last_day_1h"] = prow

    # 시즌마다
    srows = []
    for si_, s in enumerate(SEAS):
        Ht = htrue(s)
        ix = np.where(D["S"] == si_)[0]
        XAs = RA["XA"][ix]
        n6 = int((XAs[:, NE:NE + 6].sum(1) > 0).sum())
        nm = int((XAs[:, NE + 6:].sum(1) > 0).sum())
        row = {"season": s, "era": SI[s]["era"], "special": SI[s]["special"], "dow": SI[s]["dow"], "clock_ok": SI[s]["clock_ok"], "pairs": int(len(ix)),
               "pairs_last6": n6, "pairs_H24_H6": nm}
        if n6 >= 5 and nm >= 5:
            wv, yv = D["w"][ix], D["y"][ix]
            # 전체 모양(λ_j)을 고정하고 시즌마다 세 배수 (앞쪽 · H−24~H−6 · H−6~H)만 맞춘다 (NNLS)
            X3 = np.stack([XAs[:, :NE] @ lam[:NE], XAs[:, NE + 6:] @ lam[NE + 6:], XAs[:, NE:NE + 6] @ lam[NE:NE + 6]], 1) / 1000.0
            S6, S18 = lam[NE:NE + 6].sum(), lam[NE + 6:].sum()

            def shr(mu):
                den = mu[2] * S6 + mu[1] * S18
                return float(mu[2] * S6 / den) if den > 0 else float("nan")
            mu, _ = nnls(X3 * wv[:, None], yv * wv)
            sh = shr(mu)
            bs = []
            for _ in range(BOOT):
                r_ = rng.integers(0, len(ix), len(ix))
                mub, _ = nnls(X3[r_] * wv[r_][:, None], yv[r_] * wv[r_])
                bs.append(shr(mub))
            row |= {"share_H6": r4(sh, 3), "p10": r4(q(bs, 10), 3), "p90": r4(q(bs, 90), 3), "mult_early_mid_late": rl(mu, 3)}
        else:
            row |= {"share_H6": None, "p10": None, "p90": None, "why": "H−6~H 쌍 < 5 또는 H−24~H−6 쌍 < 5"}
        selp = PH["main"] & (PH["S"] == si_)
        a_ = PH["P"][selp & (PH["x"] >= 4) & (PH["x"] <= 8)]
        b_ = PH["P"][selp & (PH["x"] >= 22) & (PH["x"] <= 26)]
        row |= {"photo_n_x4_8": int(len(a_)), "photo_n_x22_26": int(len(b_))}
        if len(a_) >= 5 and len(b_) >= 5 and med(b_) > 0:
            row |= {"photo_P_H6": r4(med(a_)), "photo_P_H24": r4(med(b_)), "photo_ratio_H6_over_H24": r4(med(a_) / med(b_), 3)}
        else:
            row |= {"photo_P_H6": None, "photo_P_H24": None, "photo_ratio_H6_over_H24": None}
        srows.append(row)
    out["tables"]["by_season"] = srows
    okS = [r["share_H6"] for r in srows if r["share_H6"] is not None]
    okR = [r["photo_ratio_H6_over_H24"] for r in srows if r["photo_ratio_H6_over_H24"] is not None]
    spread = {"share_H6_n_seasons": len(okS), "share_H6_mean": r4(np.mean(okS), 3) if okS else None, "share_H6_sd": r4(np.std(okS, ddof=1), 3) if len(okS) > 1 else None,
              "share_H6_q25_med_q75": [r4(q(okS, 25), 3), r4(q(okS, 50), 3), r4(q(okS, 75), 3)] if okS else None,
              "photo_ratio_n_seasons": len(okR), "photo_ratio_q25_med_q75": [r4(q(okR, 25), 3), r4(q(okR, 50), 3), r4(q(okR, 75), 3)] if okR else None,
              "photo_ratio_sd": r4(np.std(okR, ddof=1), 3) if len(okR) > 1 else None}
    for key in ("era", "special"):
        for val in (("old", "new") if key == "era" else (True, False)):
            v = [r["share_H6"] for r in srows if r["share_H6"] is not None and r[key] == val]
            spread[f"share_H6_{key}_{val}"] = {"n": len(v), "mean": r4(np.mean(v), 3) if v else None}
    out["tables"]["season_spread"] = [spread]
    # S28·S38 한 줄
    two = []
    for s in ("S28", "S38"):
        r = [x for x in srows if x["season"] == s][0]
        two.append({"season": s, "H_info": SI[s]["H"], "H_true": htrue(s), "start_kst": SI[s]["c0"], **{k: r[k] for k in ("pairs", "pairs_last6", "pairs_H24_H6", "share_H6", "p10", "p90", "photo_n_x4_8", "photo_n_x22_26", "photo_ratio_H6_over_H24")}})
    out["extra"]["S28_S38_remaining_time_axis"] = {"text": "S28·S38은 끝 시각이 KST 04:59:59로 같아 남은 시간 축에 그대로 넣었다 (앞쪽 6시간 칸은 시즌 시작 h 축). S28 H는 info 130, 실제 끝 130.5시간 — 남은 시간은 H_true로 쟀고 쌍 시각은 130에서 잘림(공용 표).", "rows": two}
    out["extra"]["clock_note"] = "마지막 날 칸은 남은 시간 축 = KST 시각 축 (모든 시즌 끝 KST 05시). 앞쪽 6시간 칸은 h 축이라 KST와 무관(시즌 시작이 다른 S19·S28·S38도 넣었다)."
    out["extra"]["pairs_in_fit"] = D["n"]
    out["extra"]["more_seen"] = ("더 본 것(카드 칸 나누기 밖): 벌점 κ=0·×10·×0.1 민감도, 21시즌만(A21) 마지막 날 1시간 칸(몰림 배수의 분자), 시즌별 H−6~H 몫은 전체 모양(λ_j) 고정 + 시즌 세 배수 "
                                 "(앞쪽·H−24~H−6·H−6~H) NNLS (전체 λ에 그 시즌 쌍이 들어 있는 점은 같음). 시즌 단위 쌍 붓스트랩.")
    # 노트
    r0 = rows
    imax = max(range(NL), key=lambda i: r0[i]["rate_x1000"] or 0)
    out["notes"] = [
        f"쌍 {D['n']} ({len(set(S.tolist()))}시즌): 마지막 날 합 {st0['total']:.3f} log10 (붓스트랩 {bq('total', 10):.3f}~{bq('total', 90):.3f}); 벌점 κ={RA['kbest']:g} (1-SE; 최소 CV는 κ={RA['kmin']:g}).",
        f"H−6~H 몫 {st0['s6']:.2f} ({bq('s6', 10):.2f}~{bq('s6', 90):.2f}), H−3~H {st0['s3']:.2f}, H−1~H {st0['s1']:.3f}.",
        f"마지막 날 속도 최고 칸 남은 {r0[imax]['x_from']}~{r0[imax]['x_to']}시간 (KST {r0[imax]['kst']}) {r0[imax]['rate_x1000']:.1f}e-3 log10/h; 최저 칸 남은 약 {st0['xmin']:.1f}시간; 다시 오르는 시각(정의 shares.rise_def) 남은 {st0['xrise']}시간.",
        f"몰림 배수(H−6~H 평균 속도 ÷ 3~4일차 같은 KST 시각) {sumr[0]['ratio']} ({sumr[0]['p10']}~{sumr[0]['p90']}), H−1~H {sumr[2]['ratio']}.",
        f"시즌별 H−6~H 몫: 쌍 기준 충족 {len(okS)}시즌, 중앙 {q(okS, 50):.2f} (사분위 {q(okS, 25):.2f}~{q(okS, 75):.2f}); 사진 P(H−6)/P(H−24) {len(okR)}시즌 중앙 {q(okR, 50) if okR else float('nan'):.2f}.",
    ]
    # 그림
    xc = [r["x_from"] - 0.5 for r in rows]
    plots.append({"id": "A1", "part": "A", "title": "마지막 날 1시간 칸 침수 속도 (기록 쌍, 벌점 NNLS)", "x": {"label": "마감까지 남은 시간 (시간)", "log": False, "reverse": True},
                  "y": {"label": "침수 속도 (1e-3 log10/시간)", "log": False},
                  "series": [{"name": f"기록 쌍 24시즌 (1시간 칸, 벌점 κ={RA['kbest']:g})", "kind": "line", "x": xc, "y": rl([r["rate_x1000"] for r in rows]), "group": "data"},
                             {"name": "시즌 붓스트랩 10~90%", "kind": "band", "x": xc, "y": rl([r["rate_x1000"] for r in rows]), "lo": rl([r["p10"] for r in rows]), "hi": rl([r["p90"] for r in rows]), "group": "data"},
                             {"name": "3~4일차 같은 KST 시각 (3시간 칸, 21시즌)", "kind": "step", "x": [3.0 * i for i in range(9)], "y": rl([c34[7 - i] for i in range(8)]), "group": "act"}],
                  "note": f"쌍 {D['n']}·24시즌, 칸당 쌍 수는 표 last_day_1h. 남은 시간 축 = KST 05시 끝. 3~4일차 선은 3시간 칸 (21시즌 1,101쌍)."})
    xc2 = [r["x_from"] - 0.5 for r in prow]
    plots.append({"id": "A2", "part": "A", "title": "마지막 날 사진 남은 침수 P_obs 대 F·M3·act", "x": {"label": "마감까지 남은 시간 (시간)", "log": False, "reverse": True},
                  "y": {"label": "남은 침수 log10(최종 % ÷ 지금 %)", "log": False},
                  "series": [{"name": "사진 P_obs 중앙 (A·B, 1시간 칸)", "kind": "points", "x": xc2, "y": [r["P_med"] for r in prow], "group": "data"},
                             {"name": "사분위", "kind": "band", "x": xc2, "y": [r["P_med"] for r in prow], "lo": [r["P_q25"] for r in prow], "hi": [r["P_q75"] for r in prow], "group": "data"},
                             {"name": "F (같은 사진 시각)", "kind": "line", "x": xc2, "y": [r["F_med"] for r in prow], "group": "F"},
                             {"name": "M3 기본", "kind": "line", "x": xc2, "y": [r["M3_med"] for r in prow], "group": "M3"},
                             {"name": "act M3", "kind": "line", "x": xc2, "y": [r["act_med"] for r in prow], "group": "act"}],
                  "note": f"A·B 사진 {int(msk.sum())}장 (지금 % ≥ 0.3, 최종 % 0.5~15). 곡선은 같은 사진 줄의 시즌·시각·최종 %로 셈한 값의 중앙."})
    ss = [r for r in srows if r["share_H6"] is not None]
    plots.append({"id": "A3", "part": "A", "title": "시즌별 H−6~H 침수 몫 (기록 쌍)", "x": {"label": "시즌 번호", "log": False, "reverse": False}, "y": {"label": "마지막 날 침수 중 H−6~H 몫", "log": False},
                  "series": [{"name": "시즌 몫 (쌍 붓스트랩 10~90% = lo·hi)", "kind": "points", "x": [int(r["season"][1:]) for r in ss], "y": [r["share_H6"] for r in ss], "lo": [r["p10"] for r in ss], "hi": [r["p90"] for r in ss], "group": "season"},
                             {"name": "전체", "kind": "line", "x": [19, 42], "y": [r4(st0["s6"], 3)] * 2, "group": "data"}],
                  "note": f"쌍 기준(H−6~H ≥ 5, H−24~H−6 ≥ 5) 충족 {len(ss)}시즌. 형식 특수 S26·S28·S38 표시는 표 by_season."})
    return out, st0


def run_B(D, PH, CB, rng, plots, log):
    out = {"tables": {}, "notes": [], "extra": {}}
    h, P, pf, pct = PH["h"], PH["P"], PH["pf"], PH["pct"]
    ab = np.isin(PH["kind"], ["A", "B"]) & np.isfinite(P) & (pf >= PF_LO) & (pf <= PF_HI) & (h >= 0)
    base0 = ab & (pct >= 0.1)
    main = PH["main"]
    edges = np.arange(0, 49, 3.0)
    nb = len(edges) - 1
    bi = np.clip(np.floor(h / 3.0).astype(int), 0, 99)
    curves = {"F": PH["F"], "M3": PH["Mb"], "act": PH["Ma"]}
    rows = []
    byS = {c: [[np.array([]) for _ in range(nb)] for _ in range(NS)] for c in ("F", "M3", "act")}
    for k in range(nb):
        sel = main & (bi == k)
        st = pstat(P[sel])
        b0 = base0 & (bi == k)
        n0 = int(b0.sum())
        sh = [r4(((pct[b0] >= a) & (pct[b0] < b)).sum() / n0, 3) if n0 else None for a, b in ((0.1, 0.3), (0.3, 1.0), (1.0, 3.0))] + [r4((pct[b0] >= 3).sum() / n0, 3) if n0 else None]
        row = {"h_from": 3 * k, "h_to": 3 * k + 3, "n": st["n"], "n_seasons": int(len(set(PH["S"][sel].tolist()))), "P_med": st["med"], "P_q25": st["q25"], "P_q75": st["q75"],
               "n_pf_range_pct_ge_0.1": n0, "pct_share_0.1_0.3": sh[0], "pct_share_0.3_1": sh[1], "pct_share_1_3": sh[2], "pct_share_ge_3": sh[3], "small": bool(st["n"] < MINN)}
        for c, arr in curves.items():
            d = P[sel] - arr[sel]
            row[f"{c}_med"] = r4(med(arr[sel]))
            row[f"P_minus_{c}"] = r4(med(d))
            for si_ in range(NS):
                byS[c][si_][k] = d[PH["S"][sel] == si_]
        rows.append(row)
    # 붓스트랩 띠 (시즌)
    bb = {c: [] for c in curves}
    for _ in range(BOOT):
        dr = rng.integers(0, NS, NS)
        for c in curves:
            bb[c].append([med(np.concatenate([byS[c][s][k] for s in dr])) for k in range(nb)])
    for c in curves:
        lo_, hi_ = band(np.array(bb[c]))
        for k in range(nb):
            rows[k][f"P_minus_{c}_p10"] = r4(lo_[k])
            rows[k][f"P_minus_{c}_p90"] = r4(hi_[k])
    out["tables"]["photos_3h"] = rows
    # WP11b 가설
    hyp = []
    for k in range(2, nb):
        r = rows[k]
        if r["n"] == 0:
            continue
        hyp.append({"h_from": r["h_from"], "h_to": r["h_to"], "n": r["n"], "P_minus_act": r["P_minus_act"], "P_minus_M3": r["P_minus_M3"], "P_minus_F": r["P_minus_F"],
                    "act_closer": bool(abs(r["P_minus_act"]) < abs(r["P_minus_M3"])), "small": r["small"]})
    out["tables"]["wp11b_hypothesis"] = hyp
    nact = sum(x["act_closer"] for x in hyp)
    out["tables"]["wp11b_hypothesis_summary"] = [{"bins": len(hyp), "act_closer_bins": nact,
                                                  "mean_abs_diff_act": r4(np.mean([abs(x["P_minus_act"]) for x in hyp]), 4), "mean_abs_diff_M3": r4(np.mean([abs(x["P_minus_M3"]) for x in hyp]), 4),
                                                  "mean_abs_diff_F": r4(np.mean([abs(x["P_minus_F"]) for x in hyp]), 4)}]
    # 시작 직후 몰림 기울기
    meds = [r["P_med"] for r in rows]
    sl = []
    for k in range(nb - 1):
        if meds[k] is not None and meds[k + 1] is not None:
            sl.append({"h_mid": 3 * (k + 1), "kst_if_05start": f"{(5 + 3 * (k + 1)) % 24:02d}", "slope_dP_dh": r4((meds[k + 1] - meds[k]) / 3.0, 4), "n_pair_bins": [rows[k]["n"], rows[k + 1]["n"]]})
    out["tables"]["slopes_dP_dh"] = sl
    def avg(lo, hi):
        v = [s["slope_dP_dh"] for s in sl if lo <= s["h_mid"] <= hi]
        return r4(np.mean(v), 4) if v else None
    out["tables"]["slope_summary"] = [{"h0_12_mean_slope": avg(3, 12), "h24_36_mean_slope": avg(27, 36), "h36_48_mean_slope": avg(39, 48), "text": "이웃 칸 중앙 차 ÷ 3. h 0~12(=3,6,9,12 경계)와 같은 KST 시각인 h 24~36(경계 27~36)"}]
    # 쌍: 첫날
    d1 = [i for i in range(D["n"]) if D["t1"][i] < 24.0]
    ex = np.zeros(NB)
    cnt = np.zeros(NB, int)
    seas_first = set()
    for i in d1:
        s = SEAS[D["S"][i]]
        e = AC.exposure(SI[s]["c0h"], float(SI[s]["H"]), D["t1"][i], min(D["t2"][i], 24.0)).sum(0)
        ex += e
        cnt += e > 0
        seas_first.add(s)
    out["tables"]["pairs_first_day"] = [{"n_pairs_t1_lt_24": len(d1), "n_seasons": len(seas_first), "n_pairs_end_lt_24": int((D["t2"] <= 24).sum()),
                                         "kst_bins": BIN_NAMES, "pairs_per_kst_bin": cnt.tolist(), "overlap_h_per_kst_bin": [r4(v, 1) for v in ex],
                                         "note": "KST 3시간 칸 = 05-08 … 02-05 (S19·S28·S38은 시작이 05시가 아니라 h 0~24가 KST 하루와 어긋난다)"}]
    # 첫날 하루 안 몫 대 3~5일차 (C 붓스트랩)
    Cb, lamC = CB["Cb"], CB["lamC"]
    sh1 = lamC[0] / max(lamC[0].sum(), 1e-12)
    sh35 = np.mean([lamC[d] / max(lamC[d].sum(), 1e-12) for d in (2, 3, 4)], 0)
    sb1 = Cb[:, 0, :] / np.maximum(Cb[:, 0, :].sum(1, keepdims=True), 1e-12)
    sb35 = np.mean([Cb[:, d, :] / np.maximum(Cb[:, d, :].sum(1, keepdims=True), 1e-12) for d in (2, 3, 4)], 0)
    l1, h1 = band(sb1)
    l35, h35 = band(sb35)
    rc = CB["rows"]
    tch = (D["E"][rc][:, 0, :] > 0)
    e1 = D["E"][rc][:, 0, :].sum(0)
    out["tables"]["day1_vs_day35"] = [{"kst": BIN_NAMES[b], "day1_share": r4(sh1[b], 3), "day1_p10": r4(l1[b], 3), "day1_p90": r4(h1[b], 3), "day35_share": r4(sh35[b], 3), "day35_p10": r4(l35[b], 3),
                                       "day35_p90": r4(h35[b], 3), "day1_pairs_touching": int(tch[:, b].sum()), "day1_overlap_h": r4(e1[b], 1), "small": bool(tch[:, b].sum() < MINN), "unidentified_no_pair": bool(tch[:, b].sum() == 0)} for b in range(NB)]
    out["tables"]["day1_vs_day35_note"] = [{"text": "21시즌(05시 시작) 일차 × 3시간 칸 NNLS의 1일차 줄 대 3~5일차 줄 평균 몫. 1일차 쌍 수는 칸마다 적다 — 띠 폭을 그대로 읽는다.",
                                            "day1_total_log10_identified_cells_only": r4(3 * lamC[0].sum() / 1000, 4), "day1_cells_without_pairs": int((~tch.any(0)).sum()), "day1_total_p10_p90": [r4(q(3 * Cb[:, 0, :].sum(1) / 1000, 10), 4), r4(q(3 * Cb[:, 0, :].sum(1) / 1000, 90), 4)]}]
    # 더 본 것: C 사진 참고 줄, clock21 판
    crow = []
    for k in range(nb):
        sel = np.isin(PH["kind"], ["C"]) & np.isfinite(P) & (pf >= PF_LO) & (pf <= PF_HI) & (pct >= PCT_LO) & (bi == k) & (h >= 0)
        sc = main & (bi == k) & np.isin(PH["S"], [SIDX[s] for s in CLOCK])
        crow.append({"h_from": 3 * k, "h_to": 3 * k + 3, "C_n": int(sel.sum()), "C_P_med": pstat(P[sel])["med"], "clock21_n": int(sc.sum()), "clock21_P_med": pstat(P[sc])["med"]})
    out["extra"]["C_photos_and_clock21_by_3h"] = {"text": "더 본 것: 게시 시각 상한인 C 사진 참고 줄과 05시 시작 21시즌만의 중앙", "rows": crow}
    out["extra"]["note_h_axis"] = "h 칸은 시작부터 시간 (S19·S28·S38 포함 24시즌). 05시 시작 시즌에서만 KST 칸과 같다."
    m_ = [r for r in rows if r["n"] > 0]
    out["notes"] = [
        f"사진(A·B, 지금 % ≥ 0.3, 최종 % 0.5~15) h 0~48 {int(sum(r['n'] for r in rows))}장, 칸 n 최소 {min(r['n'] for r in rows)} 최대 {max(r['n'] for r in rows)}.",
        f"h 6~48 {len(hyp)}칸 중 act 꼴이 M3 기본보다 사진에 가까운 칸 {nact} (평균 |차| act {out['tables']['wp11b_hypothesis_summary'][0]['mean_abs_diff_act']}, M3 {out['tables']['wp11b_hypothesis_summary'][0]['mean_abs_diff_M3']}, F {out['tables']['wp11b_hypothesis_summary'][0]['mean_abs_diff_F']}).",
        f"쌍 첫날(t1 < 24h) {len(d1)}쌍 {len(seas_first)}시즌 / 전체 {D['n']}쌍; 1일차 합 {3 * lamC[0].sum() / 1000:.3f} log10 (21시즌).",
        f"기울기 dP/dh 평균 h 0~12 {avg(3, 12)} 대 h 24~36 {avg(27, 36)}.",
    ]
    xc = [3 * k + 1.5 for k in range(nb)]
    plots.append({"id": "B1", "part": "B", "title": "첫날~2일차 사진 남은 침수 P_obs 대 F·M3·act", "x": {"label": "챌린지 시작부터 시간 h", "log": False, "reverse": False},
                  "y": {"label": "남은 침수 log10(최종 % ÷ 지금 %)", "log": False},
                  "series": [{"name": "사진 P_obs 중앙 (A·B, 3시간 칸)", "kind": "points", "x": xc, "y": [r["P_med"] for r in rows], "group": "data"},
                             {"name": "사분위", "kind": "band", "x": xc, "y": [r["P_med"] for r in rows], "lo": [r["P_q25"] for r in rows], "hi": [r["P_q75"] for r in rows], "group": "data"},
                             {"name": "F", "kind": "line", "x": xc, "y": [r["F_med"] for r in rows], "group": "F"},
                             {"name": "M3 기본", "kind": "line", "x": xc, "y": [r["M3_med"] for r in rows], "group": "M3"},
                             {"name": "act M3", "kind": "line", "x": xc, "y": [r["act_med"] for r in rows], "group": "act"}],
                  "note": f"사진 {int(sum(r['n'] for r in rows))}장 (칸 n 표 photos_3h). 곡선은 같은 사진 줄 값의 중앙. 05시 시작 시즌에서 h 칸 = KST 칸."})
    return out


def sh_norm(l):
    return l / np.maximum(l.sum(-1, keepdims=True), 1e-12)


def run_C(D, PH, CB, rng, plots, log):
    out = {"tables": {}, "notes": [], "extra": {}}
    lamC, Cb, mC = CB["lamC"], CB["Cb"], CB["mC"]
    E = D["E"][CB["rows"]]
    Td, sh, sh6 = day_summary(lamC)
    Tb = np.array([day_summary(c)[0] for c in Cb])
    shb = np.array([day_summary(c)[1] for c in Cb])
    sh6b = np.array([day_summary(c)[2] for c in Cb])
    lo, hi = band(Cb)
    tl, th = band(Tb)
    sl, sh_ = band(shb)
    cell = []
    for d in range(5):
        for b in range(NB):
            tchd = int((E[:, d, b] > 0).sum())
            cell.append({"day": d + 1, "kst": BIN_NAMES[b], "rate_x1000": r4(lamC[d, b]), "p10": r4(lo[d, b]), "p90": r4(hi[d, b]), "share_in_day": r4(sh[d, b], 3),
                         "share_p10": r4(sl[d, b], 3), "share_p90": r4(sh_[d, b], 3), "pairs": tchd, "overlap_h": r4(E[:, d, b].sum(), 1), "small": bool(tchd < MINN),
                         "unidentified_no_pair": bool(tchd == 0)})
    out["tables"]["day_x_3h"] = cell
    ds = []
    DN = ("05-11", "11-17", "17-23", "23-05")
    s6l, s6h = band(sh6b)
    for d in range(5):
        ds.append({"day": d + 1, "total_log10": r4(Td[d], 4), "p10": r4(tl[d], 4), "p90": r4(th[d], 4), "pairs_touching": int((E[:, d, :].sum(1) > 0).sum()),
                   "share_6h": {DN[k]: r4(sh6[d, k], 3) for k in range(4)}, "share_6h_p10": {DN[k]: r4(s6l[d, k], 3) for k in range(4)}, "share_6h_p90": {DN[k]: r4(s6h[d, k], 3) for k in range(4)}})
    out["tables"]["day_summary"] = ds
    m35 = np.mean([sh6[d] for d in (2, 3, 4)], 0)
    m35b = np.mean(sh6b[:, 2:5, :], 1)
    a_, b_ = band(m35b)
    out["tables"]["wp10a_compare"] = [{"day3_5_mean_share_6h": {DN[k]: r4(m35[k], 3) for k in range(4)}, "p10": {DN[k]: r4(a_[k], 3) for k in range(4)}, "p90": {DN[k]: r4(b_[k], 3) for k in range(4)},
                                       "wp10a_cited": {"05-11": 0.05, "11-17": 0.12, "17-23": 0.46, "23-05": 0.37}, "wp10a_day2_cited": [0.23, 0.25, 0.39, 0.13],
                                       "wp10a_day_totals_2_5_cited": [0.276, 0.133, 0.126, 0.160], "n_pairs": int(len(E)), "n_seasons": len(CB["seasons"]),
                                       "season_mult_m": {s: r4(m, 3) for s, m in zip(CB["seasons"], mC)}}]
    # 시즌별 하루 모양 (2~5일차 묶음)
    S, y, w = D["S"], D["y"], D["w"]
    shp = []
    allsh = []
    for si_, s in enumerate(SEAS):
        ix = np.where(S == si_)[0]
        X8 = D["E"][ix][:, 1:5, :].sum(1)
        keep = (X8.sum(1) > 0)
        ix, X8 = ix[keep], X8[keep]
        npair = len(ix)
        if npair < 3:
            shp.append({"season": s, "pairs_days2_5": npair, "share": None})
            continue
        l = fit_shape8(X8, y[ix], w[ix])
        shv = l / max(l.sum(), 1e-12)
        bs = []
        for _ in range(BOOT):
            r_ = rng.integers(0, npair, npair)
            lb = fit_shape8(X8[r_], y[ix][r_], w[ix][r_])
            bs.append(lb / max(lb.sum(), 1e-12))
        bs = np.array(bs)
        lo_, hi_ = band(bs)
        shp.append({"season": s, "era": SI[s]["era"], "special": SI[s]["special"], "dow": SI[s]["dow"], "clock_ok": SI[s]["clock_ok"], "pairs_days2_5": npair, "small": bool(npair < MINN),
                    "share": rl(shv, 3), "p10": rl(lo_, 3), "p90": rl(hi_, 3), "boot_sd": rl(bs.std(0), 3)})
        allsh.append((s, npair, shv, bs.std(0)))
    out["tables"]["season_shapes_days2_5"] = shp
    spread = []
    for sub, nm in ((allsh, "all"), ([a for a in allsh if a[1] >= MINN], "pairs>=15")):
        if len(sub) < 3:
            continue
        M = np.array([a[2] for a in sub])
        Wv = np.array([a[3] for a in sub])
        row = {"set": nm, "n_seasons": len(sub), "kst": BIN_NAMES, "mean": rl(M.mean(0), 3), "sd": rl(M.std(0, ddof=1), 3), "q25": rl(np.percentile(M, 25, 0), 3), "med": rl(np.median(M, 0), 3),
               "q75": rl(np.percentile(M, 75, 0), 3), "within_season_boot_sd_mean": rl(np.sqrt((Wv ** 2).mean(0)), 3),
               "excess_sd": rl(np.sqrt(np.maximum(M.var(0, ddof=1) - (Wv ** 2).mean(0), 0)), 3)}
        spread.append(row)
    out["tables"]["season_shape_spread"] = spread
    # 시대 둘
    era = {}
    for nm, ss in (("old_S19_S30", [s for s in CLOCK if int(s[1:]) <= 30]), ("new_S31_S42", [s for s in CLOCK if int(s[1:]) >= 31])):
        ids = [SIDX[s] for s in ss]
        rows_ = np.where(np.isin(S, ids))[0]
        Ee, ye, we = D["E"][rows_], y[rows_], w[rows_]
        sit = np.unique(S[rows_], return_inverse=True)[1]
        lam_e, _ = fit_C(Ee, ye, we, sit, len(ids))
        bT, bS = [], []
        for _ in range(BOOT):
            idx, si, n = boot_idx(S[rows_], ids, rng)
            lb, _ = fit_C(Ee[idx], ye[idx], we[idx], si, n)
            bT.append(day_summary(lb)[0])
            bS.append(sh_norm(lb[1:5].sum(0)))
        Te, she, _ = day_summary(lam_e)
        pooled = sh_norm(lam_e[1:5].sum(0))
        tl_, th_ = band(np.array(bT))
        sl_, sh__ = band(np.array(bS))
        era[nm] = {"n_seasons": len(ids), "n_pairs": int(len(rows_)), "day_total_log10": rl(Te), "day_total_p10": rl(tl_), "day_total_p90": rl(th_), "pooled_days2_5_share_3h": rl(pooled, 3),
                   "pooled_p10": rl(sl_, 3), "pooled_p90": rl(sh__, 3), "kst": BIN_NAMES, "_bT": np.array(bT), "_bS": np.array(bS)}
    dT = era["new_S31_S42"]["_bT"] - era["old_S19_S30"]["_bT"]
    dS = era["new_S31_S42"]["_bS"] - era["old_S19_S30"]["_bS"]
    dl, dh = band(dT)
    sl_, sh__ = band(dS)
    for nm in era:
        era[nm].pop("_bT")
        era[nm].pop("_bS")
    era["diff_new_minus_old"] = {"day_total_log10": rl(np.array(era["new_S31_S42"]["day_total_log10"]) - np.array(era["old_S19_S30"]["day_total_log10"])), "p10": rl(dl), "p90": rl(dh),
                                 "pooled_share_diff": rl(np.array(era["new_S31_S42"]["pooled_days2_5_share_3h"]) - np.array(era["old_S19_S30"]["pooled_days2_5_share_3h"]), 3), "pooled_p10": rl(sl_, 3),
                                 "pooled_p90": rl(sh__, 3), "note": "두 시대 붓스트랩은 독립 추출의 차"}
    out["tables"]["era"] = [era]
    # 시작 요일: 하루 합 모델 (하루 안 모양을 전체 2~5일차 몫으로 고정, 일차마다 상수 τ_d)
    gbar = sh_norm(lamC[1:5].sum(0)) * NB

    def day_fit(rows_, ids, use_m=True, pair_boot=False):
        Ee, ye, we = D["E"][rows_], y[rows_], w[rows_]
        Xd = np.einsum("ndb,b->nd", Ee[:, :5, :], gbar)
        sit = np.unique(S[rows_], return_inverse=True)[1]
        lam, m = fit_m(Xd, ye, we, sit, len(ids))
        return lam, Xd, ye, we, sit

    def daycfg(ids, label, season_boot):
        rows_ = np.where(np.isin(S, ids))[0]
        lam, Xd, ye, we, sit = day_fit(rows_, ids)
        bs = []
        for _ in range(BOOT):
            if season_boot:
                idx, si, n = boot_idx(S[rows_], ids, rng)
            else:
                idx = rng.integers(0, len(rows_), len(rows_))
                si, n = np.zeros(len(idx), int), 1
            lb, _ = fit_m(Xd[idx], ye[idx], we[idx], si, n)
            bs.append(lb)
        bs = np.array(bs)
        lo_, hi_ = band(bs)
        return {"group": label, "n_seasons": len(ids), "n_pairs": int(len(rows_)), "boot": "시즌" if season_boot else "쌍",
                "day_total_log10": rl(24 * lam / 1000, 4), "p10": rl(24 * lo_ / 1000, 4), "p90": rl(24 * hi_ / 1000, 4),
                "share_of_days2_5": rl(lam[1:5] / max(lam[1:5].sum(), 1e-12), 3),
                "weekday_of_day": None}, lam, bs

    dow_rows = []
    sat = [SIDX[s] for s in CLOCK if SI[s]["dow"] == "Sat"]
    res = {}
    for lab, ids, sb in (("Sat_start", sat, True), ("S29_Thu_start", [SIDX["S29"]], False), ("S35_Sun_start", [SIDX["S35"]], False)):
        r, lam, bs = daycfg(ids, lab, sb)
        wd = {"Sat_start": ["Sat", "Sun", "Mon", "Tue", "Wed"], "S29_Thu_start": ["Thu", "Fri", "Sat", "Sun", "Mon"], "S35_Sun_start": ["Sun", "Mon", "Tue", "Wed", "Thu"]}[lab]
        r["weekday_of_day"] = wd
        dow_rows.append(r)
        res[lab] = (lam, bs)
    lam_s, bs_s = res["Sat_start"]
    wr = lam_s[1] / np.mean(lam_s[2:5])
    wrb = bs_s[:, 1] / np.mean(bs_s[:, 2:5], 1)
    l29, b29 = res["S29_Thu_start"]
    w29 = np.mean(l29[2:4]) / np.mean([l29[1], l29[4]])
    w29b = np.mean(b29[:, 2:4], 1) / np.mean(b29[:, [1, 4]], 1)
    dow_rows.append({"group": "weekend_day_ratio", "Sat_start_day2_Sun_over_days3_5": r4(wr, 3), "p10": r4(q(wrb, 10), 3), "p90": r4(q(wrb, 90), 3),
                     "S29_days3_4_SatSun_over_days2_5": r4(w29, 3), "S29_p10": r4(q(w29b, 10), 3), "S29_p90": r4(q(w29b, 90), 3),
                     "S35": "일요일이 1일차(2~5일차에 주말 없음)", "text": "하루 안 모양은 21시즌 2~5일차 몫으로 고정하고 일차마다 상수 τ_d만 맞췄다. 주말을 낀 일차 = 토요일 시작은 2일차(일), S29는 3·4일차(토·일)"})
    out["tables"]["start_weekday"] = dow_rows
    # 형식 특수: 시즌 배수 m_s (21시즌) + S28·S38은 21시즌 밖
    out["tables"]["special_formats"] = [{"season": s, "special": SI[s]["special"], "in_clock21": SI[s]["clock_ok"], "m_s_in_C_fit": r4(dict(zip(CB["seasons"], mC)).get(s), 3), "H": SI[s]["H"], "c0": SI[s]["c0"]} for s in ("S26", "S28", "S38")]
    # 분해: 하루 합 대 하루 안 모양 (시즌 안 5겹 교차검증)
    log("C decomposition")
    cl = [SIDX[s] for s in CLOCK]
    rows_all = CB["rows"]
    K = 5
    kappas = [1e1, 1e2, 1e3, 1e4, 1e5]
    kinds = ["m", "g", "t", "gt"]
    cvss = {("null", 0): 0.0, ("m", 0): 0.0}
    for kd in kinds[1:]:
        for kp in kappas:
            cvss[(kd, kp)] = 0.0
    per_season = []
    fullfits = {}
    for sid in cl:
        s = SEAS[sid]
        tr = rows_all[S[rows_all] != sid]
        te = rows_all[S[rows_all] == sid]
        sit = np.unique(S[tr], return_inverse=True)[1]
        lam_s_, _ = fit_C(D["E"][tr], y[tr], w[tr], sit, sit.max() + 1)
        Z = D["E"][te][:, :5, :] * lam_s_[None] / 1000.0
        yt, wt = y[te], w[te]
        fold = rng.permutation(len(te)) % K
        loc = {("null", 0): 0.0, ("m", 0): 0.0}
        for kd in kinds[1:]:
            for kp in kappas:
                loc[(kd, kp)] = 0.0
        for f in range(K):
            a, b = fold != f, fold == f
            if not b.any() or not a.any():
                continue
            loc[("null", 0)] += float((((yt[b] - Z[b].sum((1, 2))) * wt[b]) ** 2).sum())
            m, t, g = fit_season(Z[a], yt[a], wt[a], "", 0)
            loc[("m", 0)] += float((((yt[b] - pred_season(Z[b], m, t, g)) * wt[b]) ** 2).sum())
            for kd in kinds[1:]:
                for kp in kappas:
                    m, t, g = fit_season(Z[a], yt[a], wt[a], kd, kp)
                    loc[(kd, kp)] += float((((yt[b] - pred_season(Z[b], m, t, g)) * wt[b]) ** 2).sum())
        for k_, v in loc.items():
            cvss[k_] += v
        per_season.append((s, len(te), loc))
        fullfits[s] = (Z, yt, wt)
    best = {kd: min(kappas, key=lambda kp: cvss[(kd, kp)]) for kd in kinds[1:]}
    tab = [{"model": "null: 시즌 배수 1 (전체 λ 그대로)", "cv_ss": r4(cvss[("null", 0)], 1)},
           {"model": "① m: 시즌 배수만", "cv_ss": r4(cvss[("m", 0)], 1)}]
    names = {"g": "② m + 하루 안 모양 g_b (8칸)", "t": "③ m + 일차 수준 t_d (5일)", "gt": "④ m + g_b + t_d"}
    for kd in ("g", "t", "gt"):
        tab.append({"model": names[kd], "kappa_best": best[kd], "cv_ss": r4(cvss[(kd, best[kd])], 1), "cv_ss_by_kappa": {str(kp): r4(cvss[(kd, kp)], 1) for kp in kappas},
                    "reduction_vs_m": r4(1 - cvss[(kd, best[kd])] / cvss[("m", 0)], 4), "seasons_better_than_m": int(sum(l[(kd, best[kd])] < l[("m", 0)] for _, _, l in per_season))})
    tab[1]["reduction_vs_null"] = r4(1 - cvss[("m", 0)] / cvss[("null", 0)], 4)
    out["tables"]["decomposition_cv"] = tab
    out["tables"]["decomposition_cv_note"] = [{"text": "셈: 시즌 s의 쌍을 쌍 5겹 교차검증 (고정 시드). 전체 λ(일차 × 3시간 칸)는 s를 뺀 20시즌 NNLS. 시즌 s의 추가 모수(m, g_b, t_d; g·t는 평균 1 쪽 릿지 벌점 kappa)를 4/5 쌍으로 맞춰 1/5 쌍을 예측, 가중 오차 제곱합을 21시즌 합. 벌점은 격자 최소(약간 낙관, 모형 사이 같은 방식).",
                                               "n_seasons": len(cl), "n_pairs": int(len(rows_all))}]
    # 시즌별 맞춘 값과 퍼짐
    fit_rows = []
    lnm, lnt, gg = [], [], []
    wint = {"m": [], "t": [], "g": []}
    for sid, (s, nte, loc) in zip(cl, per_season):
        Z, yt, wt = fullfits[s]
        m0, _, _ = fit_season(Z, yt, wt, "", 0)
        _, t_, _ = fit_season(Z, yt, wt, "t", best["t"])
        m1, t1_, _ = fit_season(Z, yt, wt, "t", best["t"])
        _, _, g_ = fit_season(Z, yt, wt, "g", best["g"])
        fit_rows.append({"season": s, "pairs": nte, "m": r4(m0, 3), "t_d": rl(t1_, 3), "g_b": rl(g_, 3), "small": bool(nte < MINN)})
        lnm.append(math.log(max(m0, 1e-3)))
        lnt.append(np.log(np.maximum(t1_, 1e-3)))
        gg.append(g_)
        bm, bt, bg = [], [], []
        for _ in range(40):
            r_ = rng.integers(0, nte, nte)
            bm.append(math.log(max(fit_season(Z[r_], yt[r_], wt[r_], "", 0)[0], 1e-3)))
            bt.append(np.log(np.maximum(fit_season(Z[r_], yt[r_], wt[r_], "t", best["t"])[1], 1e-3)))
            bg.append(fit_season(Z[r_], yt[r_], wt[r_], "g", best["g"])[2])
        wint["m"].append(np.var(bm, ddof=1))
        wint["t"].append(np.var(bt, 0, ddof=1))
        wint["g"].append(np.var(bg, 0, ddof=1))
    lnt, gg = np.array(lnt), np.array(gg)
    out["tables"]["season_fits"] = fit_rows
    out["tables"]["variance_split"] = [{"quantity": "ln m_s (시즌 배수)", "between_sd": r4(np.std(lnm, ddof=1), 4), "within_boot_sd": r4(math.sqrt(np.mean(wint["m"])), 4),
                                        "excess_sd": r4(math.sqrt(max(np.var(lnm, ddof=1) - np.mean(wint["m"]), 0)), 4)},
                                       {"quantity": "ln t_d (일차 수준, 일차 1~5)", "between_sd": rl(np.std(lnt, 0, ddof=1), 4), "within_boot_sd": rl(np.sqrt(np.mean(wint["t"], 0)), 4),
                                        "excess_sd": rl(np.sqrt(np.maximum(np.var(lnt, 0, ddof=1) - np.mean(wint["t"], 0), 0)), 4)},
                                       {"quantity": "g_b (하루 안 모양, 3시간 칸 8개)", "between_sd": rl(np.std(gg, 0, ddof=1), 4), "within_boot_sd": rl(np.sqrt(np.mean(wint["g"], 0)), 4),
                                        "excess_sd": rl(np.sqrt(np.maximum(np.var(gg, 0, ddof=1) - np.mean(wint["g"], 0), 0)), 4)}]
    # 사진: 시즌별 P_obs(h) 곡선과 주성분
    log("C photo pca")
    hb = np.arange(12, 115, 6.0)
    nbh = len(hb) - 1
    mat = np.full((NS, nbh), np.nan)
    cnt = np.zeros((NS, nbh), int)
    main = PH["main"]
    for i in range(NS):
        for k in range(nbh):
            sel = main & (PH["S"] == i) & (PH["h"] >= hb[k]) & (PH["h"] < hb[k + 1])
            cnt[i, k] = sel.sum()
            if sel.sum() >= 5:
                mp = med(PH["P"][sel])
                if mp > 0.002:
                    mat[i, k] = math.log10(mp)
    M = np.isfinite(mat)
    keep = M.sum(1) >= 12
    Yk = mat[keep]
    Mk = M[keep]
    mu, Vt, share, scores, filled = pca_em(np.nan_to_num(Yk), Mk, 3)
    # 완전한 시즌×칸만으로도
    colok = Mk.all(0)
    rowok = Mk[:, colok].all(1)
    cc = None
    if rowok.sum() >= 5 and colok.sum() >= 3:
        Yc = Yk[rowok][:, colok]
        U, s_, Vt2 = np.linalg.svd(Yc - Yc.mean(0), full_matrices=False)
        cc = {"n_seasons": int(rowok.sum()), "n_bins": int(colok.sum()), "share_pc": rl((s_ ** 2 / (s_ ** 2).sum())[:3], 4)}
    kept = [SEAS[i] for i in range(NS) if keep[i]]
    sdscore = scores[:, 0].std(ddof=1)
    out["tables"]["photo_pca"] = [{"bins_h": [[float(hb[k]), float(hb[k + 1])] for k in range(nbh)], "n_seasons": int(keep.sum()), "seasons": kept, "bin_rule": "6시간 칸 중앙, 칸 n ≥ 5, 시즌당 칸 ≥ 12/17 (나머지 칸은 EM 채움)",
                                   "mean_log10_P": rl(mu, 4), "pc1_shape": rl(Vt[0], 4), "pc1_sd_score": r4(sdscore, 4), "var_share_pc1_3": rl(share[:3], 4), "complete_case": cc,
                                   "pc1_scores": {s: r4(v, 3) for s, v in zip(kept, scores[:, 0])}, "n_cells_present": int(Mk.sum()), "n_cells": int(Mk.size),
                                   "cell_n_median": r4(np.median(cnt[keep][Mk]), 1),
                                   "note": "log10 P_obs의 시즌 사이 평균을 뺀 모양. 분석 4.2b의 제1 성분 0.99는 다른 표현(곡선 자체)이라 비교는 대략."}]
    out["extra"]["pca_note"] = "PC1 부호는 임의. 시즌별 사진 곡선은 A·B, 6시간 칸 n ≥ 5, 시즌당 칸 ≥ 12/17만 쓴다(14시즌)."
    out["extra"]["day1_note"] = "1일차 05–11시 칸은 쌍이 0이라 추정 불가(NNLS 0) — 1일차 하루 합·몫은 쌍이 닿은 칸만의 값이고, 분해(variance_split)의 일차 1 수준은 잡음이 크다."
    out["notes"] = [
        f"일차 × 3시간 칸 NNLS: 21시즌 {len(E)}쌍; 일차 합(log10) " + " / ".join(f"{x:.3f}" for x in Td) + " (1~5일차); WP10a 3~5일차 6시간 몫 05-11 {:.2f} 11-17 {:.2f} 17-23 {:.2f} 23-05 {:.2f} (인용 0.05/0.12/0.46/0.37).".format(*m35),
        f"교차검증 오차 감소(시즌 배수만 대비): 하루 안 모양 {tab[2]['reduction_vs_m']:.3f}, 일차 수준 {tab[3]['reduction_vs_m']:.3f}, 둘 {tab[4]['reduction_vs_m']:.3f}.",
        f"사진 시즌 곡선 PC1 분산 몫 {share[0]:.3f} (PC2 {share[1]:.3f}, PC3 {share[2]:.3f}), {int(keep.sum())}시즌 × {nbh}칸.",
        f"시대 하루 합 차(새−옛) 일차 1~5: " + " / ".join(f"{x:+.3f}" for x in era["diff_new_minus_old"]["day_total_log10"]),
    ]
    # 그림
    xb = [1.5 + 3 * b for b in range(NB)]
    plots.append({"id": "C1", "part": "C", "title": "일차 × 3시간 칸 하루 안 몫 (21시즌, 일차마다 선)", "x": {"label": "KST 3시간 칸 중앙 (05시부터 시간)", "log": False, "reverse": False},
                  "y": {"label": "그 일차 하루 안 몫", "log": False},
                  "series": [{"name": f"{d + 1}일차", "kind": "line", "x": xb, "y": rl(sh[d], 4), "group": "data" if d >= 2 else "season"} for d in range(5)] + [{"name": "3~5일차 평균", "kind": "step", "x": [3.0 * i for i in range(9)], "y": rl(np.mean(sh[2:5], 0), 4), "group": "act"}],
                  "note": f"쌍 {len(E)}·21시즌 (05시 시작 120시간), 칸당 쌍 수는 표 day_x_3h; 1일차는 쌍이 적다."})
    ok = [r for r in shp if r.get("share")]
    sr = []
    for r in ok:
        sr.append({"name": r["season"], "kind": "step", "x": [3.0 * i for i in range(9)], "y": r["share"], "group": "season"})
    allm = np.mean([r["share"] for r in ok], 0)
    plots.append({"id": "C2", "part": "C", "title": "시즌별 하루 모양 (2~5일차 묶음, 3시간 칸 몫)", "x": {"label": "KST 3시간 칸 (05시부터 시간)", "log": False, "reverse": False}, "y": {"label": "하루 안 몫", "log": False},
                  "series": sr + [{"name": "시즌 평균", "kind": "step", "x": [3.0 * i for i in range(9)], "y": rl(allm, 4), "group": "data"}],
                  "note": f"시즌 {len(ok)}개 (쌍 3 이상; 쌍 15 미만 시즌은 띠가 넓다 — 표 season_shapes_days2_5)."})
    hc = [0.5 * (hb[k] + hb[k + 1]) for k in range(nbh)]
    cs = []
    for i in range(NS):
        if keep[i]:
            ys_ = [r4(10 ** v, 4) if np.isfinite(v) else None for v in mat[i]]
            xs = [hc[k] for k in range(nbh) if ys_[k] is not None]
            cs.append({"name": SEAS[i], "kind": "line", "x": xs, "y": [v for v in ys_ if v is not None], "group": "season"})
    mp_ = 10 ** mu
    cs.append({"name": "시즌 평균 (log 평균)", "kind": "line", "x": hc, "y": rl(mp_, 4), "group": "data"})
    cs.append({"name": "평균 + 1.5 sd × PC1", "kind": "line", "x": hc, "y": rl(10 ** (mu + 1.5 * sdscore * Vt[0]), 4), "group": "act"})
    cs.append({"name": "평균 − 1.5 sd × PC1", "kind": "line", "x": hc, "y": rl(10 ** (mu - 1.5 * sdscore * Vt[0]), 4), "group": "F"})
    plots.append({"id": "C3", "part": "C", "title": "시즌별 사진 남은 침수 P_obs(h) 곡선과 제1 주성분", "x": {"label": "챌린지 시작부터 시간 h", "log": False, "reverse": False},
                  "y": {"label": "남은 침수 P_obs (log10, 6시간 칸 중앙)", "log": False}, "series": cs,
                  "note": f"{int(keep.sum())}시즌, 6시간 칸 n ≥ 5. PC1 분산 몫 {share[0]:.3f}."})
    return out


def run_D(D, PH, obs, rng, plots, log):
    out = {"tables": {}, "notes": [], "extra": {}}
    S, y, w = D["S"], D["y"], D["w"]
    cl = [SIDX[s] for s in CLOCK]
    rows_all = np.where(np.isin(S, cl))[0]
    p1 = D["p1"]
    bshape = []
    bday = []
    for bi_, (a, b) in enumerate(BANDS5):
        rows_ = rows_all[(p1[rows_all] >= a) & (p1[rows_all] < b)]
        ids = sorted(set(S[rows_].tolist()))
        if len(rows_) < 10 or len(ids) < 3:
            bshape.append({"band": [a, b], "n_pairs": int(len(rows_)), "n_seasons": len(ids), "share": None})
            bday.append({"band": [a, b], "n_pairs": int(len(rows_)), "n_seasons": len(ids), "day_total_share": None})
            continue
        E5 = D["E"][rows_]
        X8 = np.concatenate([E5[:, 1:5, :].sum(1), E5[:, 0, :].sum(1, keepdims=True), E5[:, 5:, :].sum((1, 2))[:, None]], 1)       # 2~5일차 8칸 + 1일차 + 5일차 뒤(없음)
        X8 = X8[:, :9]
        Xd = E5[:, :5, :].sum(2)
        sit = np.unique(S[rows_], return_inverse=True)[1]
        ns = len(ids)
        l8, _ = fit_m(X8, y[rows_], w[rows_], sit, ns)
        ld, _ = fit_m(Xd, y[rows_], w[rows_], sit, ns)
        b8, bd = [], []
        for _ in range(BOOT):
            idx, si, n = boot_idx(S[rows_], ids, rng)
            lb, _ = fit_m(X8[idx], y[rows_][idx], w[rows_][idx], si, n)
            ldb, _ = fit_m(Xd[idx], y[rows_][idx], w[rows_][idx], si, n)
            b8.append(sh_norm(lb[:8]))
            bd.append(ldb[1:5] / max(ldb[1:5].sum(), 1e-12))
        sh8 = sh_norm(l8[:8])
        lo_, hi_ = band(np.array(b8))
        tot = 24 * ld / 1000
        shd = ld[1:5] / max(ld[1:5].sum(), 1e-12)
        dl, dh = band(np.array(bd))
        n_in = int(((E5[:, 1:5, :].sum((1, 2))) > 0).sum())
        bshape.append({"band": [a, b], "n_pairs": int(len(rows_)), "n_seasons": len(ids), "small": bool(len(rows_) < MINN), "kst": BIN_NAMES, "share": rl(sh8, 3), "p10": rl(lo_, 3), "p90": rl(hi_, 3),
                       "pairs_touching_days2_5": n_in})
        bday.append({"band": [a, b], "n_pairs": int(len(rows_)), "n_seasons": len(ids), "small": bool(len(rows_) < MINN), "day_total_log10_days2_5": rl(tot[1:], 4), "days2_5_share": rl(shd, 3), "share_p10": rl(dl, 3), "share_p90": rl(dh, 3)})
    out["tables"]["within_day_3h_by_band"] = bshape
    out["tables"]["day_shares_by_band"] = bday
    # 수준 배수 (WP10a 5와 같은 셈: wp10_diag.fit_rate, 6시간 칸 공통 λ + %대 배수 β, 1–3% = 1)
    pr = [(SEAS[S[i]], D["t1"][i], D["t2"][i], D["y"][i], 1 / D["w"][i] ** 2, p1[i]) for i in rows_all]

    def bandf(p):
        for k, (a, b) in enumerate(BANDS5):
            if a <= p < b:
                return k
        return None
    pr = [p for p in pr if bandf(p[5]) is not None]
    nb_ = len(BANDS5)
    _, beta, _, _ = WD.fit_rate(pr, bandf, nb_)
    cnt = collections.Counter(bandf(p[5]) for p in pr)
    bb = np.array(WD.boot_seasons(pr, lambda Q: WD.fit_rate(Q, bandf, nb_)[1], BOOT, 5))
    bb = bb / bb[:, [2]]
    beta = beta / beta[2]
    lo_, hi_ = band(bb)
    out["tables"]["level_multiplier"] = [{"band": list(BANDS5[k]), "n_pairs": cnt.get(k, 0), "mult_vs_1_3pct": r4(beta[k], 3), "p10": r4(lo_[k], 3), "p90": r4(hi_[k], 3), "small": bool(cnt.get(k, 0) < MINN)} for k in range(nb_)]
    out["tables"]["level_multiplier_note"] = [{"text": "wp10_diag.fit_rate(6시간 칸 공통 λ × %대 배수 × 시즌 배수), 1–3% = 1로 맞춤. WP10a 5의 %대 둘째=1·0.88~1.06과는 %대 경계와 기준이 다르다. 21시즌 1,101쌍(p1 0.1~40%만)."}]
    # 사진: %대 × 6시간 칸
    P, pf, pct, h = PH["P"], PH["pf"], PH["pct"], PH["h"]
    ab = np.isin(PH["kind"], ["A", "B"]) & np.isfinite(P) & (pf >= PF_LO) & (pf <= PF_HI) & (pct >= 0.1) & (h >= 0)
    hb = np.arange(0, 121, 6.0)
    ph = []
    for k in range(len(hb) - 1):
        sel = ab & (h >= hb[k]) & (h < hb[k + 1])
        mall = med(P[sel])
        row = {"h_from": float(hb[k]), "h_to": float(hb[k + 1]), "n_all": int(sel.sum()), "med_all": r4(mall)}
        for (a, b) in BANDS5:
            sb = sel & (pct >= a) & (pct < b)
            mb = med(P[sb])
            row[f"{a}-{b}"] = {"n": int(sb.sum()), "med": r4(mb), "over_all": r4(mb / mall, 3) if (sb.sum() >= 5 and mall > 0.01) else None}
        ph.append(row)
    out["tables"]["photos_band_x_6h"] = ph
    out["tables"]["photos_band_note"] = [{"text": "A·B 사진, 지금 % ≥ 0.1, 최종 % 0.5~15 (pf_tab 범위 때문에 10~40% 칸은 지금 % 10~15%만 남는다 — P가 작다). 칸 n < 5이면 비를 비운다."}]
    # 숫자단 1~7위
    log("D top7")
    outT = {}
    for nm, kinds in (("A_B", ("A", "B")), ("A_B_C_ref", None)):
        rows, series, grid = top7_flow(obs, kinds)
        tab, hits, cnts = top7_tables(rows, series, grid)
        # 순위별 중앙
        rk = {}
        for r_ in range(1, 8):
            v = []
            for gi in range(len(grid)):
                x = np.array([s_["val"][gi] for k, s_ in series.items() if k[1] == r_ and np.isfinite(s_["val"][gi])])
                v.append({"h_to": float(grid[gi]), "n": int(len(x)), "med_ln": r4(np.median(x)) if len(x) else None})
            rk[str(r_)] = v
        outT[nm] = {"n_rows": len(rows), "n_series_(season,rank)": len(series), "n_seasons": len({k[0] for k in series}), "flow_by_6h": tab, "hit_times": hits, "rows_by_kind_and_server_basis": cnts, "by_rank": rk}
    out["tables"]["top7"] = [outT["A_B"]]
    out["extra"]["top7_with_C_reference"] = {"text": "참고: C(게시 시각 상한) 줄 포함. 본안 top7은 A·B만.", **outT["A_B_C_ref"]}
    out["extra"]["top7_rule"] = ("S_k(h) = h까지 본 그 순위 점수의 최대(순위 점수는 줄지 않음). 최종 S_k = 마지막 6시간(h ≥ H_true−6) 관측의 최대, 없으면 시즌·순위 뺌. "
                                 "99%·99.9%에 닿는 h = 관측에서 처음 그 비율을 넘은 시각(성긴 관측이면 늦게 잡힘). server == KR 명시 라벨만.")
    t = outT["A_B"]
    h99 = t["hit_times"]["0.99"]
    h999 = t["hit_times"]["0.999"]
    out["notes"] = [
        f"쌍 %대(p1) 5개 21시즌: " + " / ".join(f"{b['band'][0]}-{b['band'][1]}% {b['n_pairs']}쌍" for b in bshape),
        "수준 배수(1–3%=1): " + " / ".join(f"{r['band'][0]}-{r['band'][1]}% {r['mult_vs_1_3pct']}" for r in out["tables"]["level_multiplier"]),
        f"숫자단 1~7위(A·B) 시즌·순위 {t['n_series_(season,rank)']}줄 {t['n_seasons']}시즌; 99%에 닿는 h 중앙 {h99['med']} (사분위 {h99['q25']}~{h99['q75']}), 99.9% 중앙 {h999['med']} ({h999['q25']}~{h999['q75']}).",
    ]
    # 그림
    xb = [3.0 * i for i in range(9)]
    pl = [{"name": f"{a}–{b}% (쌍 {r['n_pairs']})", "kind": "step", "x": xb, "y": r["share"], "group": "season" if i % 2 else "data"} for i, ((a, b), r) in enumerate(zip(BANDS5, bshape)) if r.get("share")]
    plots.append({"id": "D1", "part": "D", "title": "%대별 하루 안 침수 몫 (2~5일차 묶음, 3시간 칸)", "x": {"label": "KST 3시간 칸 (05시부터 시간)", "log": False, "reverse": False}, "y": {"label": "하루 안 몫", "log": False},
                  "series": pl, "note": "p1(첫 기록의 %)으로 나눈 쌍, 21시즌 시즌 배수 포함 NNLS. 쌍 수는 이름 옆."})
    ds = []
    for r_ in range(1, 8):
        v = outT["A_B"]["by_rank"][str(r_)]
        xs = [a["h_to"] for a in v if a["med_ln"] is not None]
        ys = [a["med_ln"] for a in v if a["med_ln"] is not None]
        ds.append({"name": f"{r_}위 (시즌 중앙)", "kind": "line", "x": xs, "y": ys, "group": "season" if r_ > 1 else "data"})
    plots.append({"id": "D2", "part": "D", "title": "숫자단 1~7위 ln(S_k(h)/최종 S_k) 대 시각", "x": {"label": "챌린지 시작부터 시간 h", "log": False, "reverse": False},
                  "y": {"label": "ln(그때까지 본 최대 점수 ÷ 최종)", "log": False}, "series": ds,
                  "note": f"한섭 명시 라벨, A·B 관측, 순위·시즌 {t['n_series_(season,rank)']}줄 {t['n_seasons']}시즌. 표 top7."})
    return out


def main():
    global BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args()
    BOOT = args.boot
    t00 = time.time()
    rng = np.random.default_rng(12)

    def log(msg):
        print(f"[{time.time() - t00:6.0f}s] {msg}", flush=True)

    obs = args.obs_dir
    prof = AC.load(os.path.join(T.R7, "wp11/act_profile_D1.json"))
    s1_M3 = load_s1(os.path.join(T.R7, "stage1_M3_D1.json"))
    s1_act = load_s1(os.path.join(T.R7, "wp11/stage1_M3act_D1.json"))
    D = load_pairs(obs)
    PH = build_photos(obs, prof, s1_M3, s1_act)
    log(f"pairs {D['n']} photos {PH['n']} main {int(PH['main'].sum())}")
    plots = []
    RA = part_A(D, PH, rng, [], log)
    CB = c_boot(D, RA, rng)
    log("C boot done")
    A, _ = run_A(D, PH, RA, CB, rng, plots, log)
    log("A done")
    B = run_B(D, PH, CB, rng, plots, log)
    log("B done")
    Cc = run_C(D, PH, CB, rng, plots, log)
    log("C done")
    Dd = run_D(D, PH, obs, rng, plots, log)
    log("D done")
    try:
        import subprocess
        head = subprocess.check_output(["git", "-C", os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        head = None
    J = {"header": {"about": "WP12 침수 곡선 A~D 서술 (마지막 날 마감 몰림 · 첫날 · 일차·시즌 차이 · %대·숫자단) — 예측·채점 없음, 24시즌 정답 곡선을 본 서술",
                    "data_ver": "D1", "date": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d"), "code_head": head, "secs": int(time.time() - t00),
                    "pairs": D["n"], "photos_main": int(PH["main"].sum()), "boot_B": BOOT,
                    "rules": {"pair_fit": "ΔL = m_s·Σ λ_k·겹친 시간 NNLS (λ ≥ 0, m_s 기하평균 1, 번갈아 25회), 무게 1/sd(판독 오차)",
                              "A_bins": "h < H_true−24 6시간 칸 + 마지막 날 24개 1시간 칸 (남은 시간 축), 마지막 날 칸에 2차 차분 벌점 √κ·D2 (λ 단위 1e-3 log10/h)",
                              "penalty": "κ ∈ {0, .01 … 1e5} 시즌 하나씩 빼기 CV(가중 오차 제곱합) + 1-SE 규칙(짝지은 시즌 오차 차 sd×√24, 가장 큰 κ) — 10배·0.1배 민감도 병기",
                              "boot": f"시즌 복원 추출 B={BOOT}, 10~90%; 단일 시즌·소집단은 쌍 추출",
                              "photos": "A·B 사진 P_tab = log10(정답 표 보간 최종 % ÷ 지금 %), 지금 % ≥ 0.3, 최종 % 0.5~15; 곡선은 사진 줄 하나하나의 시즌·시각·최종 %로 셈해 칸 중앙",
                              "curves": "F = F_P(h/H), M3 기본 = stage1_M3_D1 시즌 곡선(wp12_tables.m3_P와 같은 식), act = stage1_M3act_D1 곡선을 u* = act_clock.ustar(act_profile_D1)에서",
                              "small": f"쌍·사진 {MINN} 미만 칸은 small 표시",
                              "season_axis": "마지막 날은 남은 시간 축 = KST 시각 축(모든 시즌 KST 04:59:59 끝), S28 H_true = 130.5. 일차·KST 칸은 05시 시작 120시간 21시즌이 본안 (S19·S28·S38 뺌), 시즌별 하루 모양·첫날 쌍 시각은 24시즌 모두 act_clock.exposure 시각 칸"}},
         "A": A, "B": B, "C": Cc, "D": Dd, "plots": plots}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=1)
    log(f"wrote {args.out} {os.path.getsize(args.out) / 1e6:.2f}MB")


if __name__ == "__main__":
    main()
