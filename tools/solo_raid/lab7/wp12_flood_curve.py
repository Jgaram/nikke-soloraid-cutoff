"""WP12b 침수 실제 곡선 + 그 꼴을 재현하는 식 (2026-11-04, 서술) — 칸 없이 자료가 그리는 곡선과 식. 예측·채점 없음, 24시즌 모두 사용.

  python3 -I tools/solo_raid/lab7/wp12_flood_curve.py --obs-dir W/obs7/full --out data/solo-raid/analysis/r7/wp12/flood_curve_D1.json

계획: docs/solo-raid-method.md 「WP12b」 (후보·고르기·재현 기준은 계산 전에 문서에 고정). 공용 코드·표는 읽기만 (wp12_tables, wp12_flood의 사진 거르기·act 셈을 빌림).
1 경험 곡선 — 시즌마다 침수 속도 λ_s(h) ≥ 0 = 3차 B-스플라인 (매듭 1시간 간격, 계수 ≥ 0), 관측 두 종류를 한 최소제곱에:
    사진 P_obs = log10(최종 % ÷ 지금 %) = Σ β_j ∫_h^H B_j, 기록 쌍 ΔL = Σ β_j ∫_t1^t2 B_j.
    무게 쌍 1/var · 사진 1/σ_ph² (σ_ph = 사진 잔차의 강건 sd (1.4826 MAD), 세 번 되풀이), 사진은 Huber 손실 (IRLS, c = 1.345σ).
    벌점 κ‖D2 β‖² (Eilers·Marx 1996), κ는 5겹 교차검증 최소 (관측을 무작위 5겹, 격자는 header.rules에 적음).
  합친 곡선 = 05시 시작 120시간 21시즌 (S19·S28·S38 뺌)을 한 λ(h)와 시즌 배수 m_s (기하평균 1, 번갈아 풀기)로. 시즌 붓스트랩 B → 10~90% 띠.
2 재현하는 식 — 후보 20개 (문서 그대로), λ_θ(h)에서 P_θ(h) = ∫_h^H λ_θ (0.05시간 격자 중점 합).
    ① A·h^(−α) 또는 + B · ② ① × exp(푸리에 K=1·2·3, KST 시각) · ③ ② × (1 + R e^(−x/τ)) · ④ ③에서 푸리에 합에 (1 + γ(d−3)).
  맞추기 = scipy.optimize.least_squares (여러 시작점·Huber 되풀이), 시즌 배수 m_s는 닫힌 꼴로 함께 (변수 소거). 고르기 = 시즌 하나씩 빼기 교차검증 + 1-SE + 재현 기준.
공개 규칙: 글 번호·원문·닉네임·원글 주소·사람 열쇠·사진 열쇠를 쓰지 않는다 (시각·%·점수·개수만).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import datetime as dt
import json
import math
import multiprocessing as mp
import sys
import time

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares, nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wp12_tables as T  # noqa: E402
import wp12_flood as FL  # noqa: E402
import act_clock as AC  # noqa: E402

SEAS = T.SEASONS
SI = T.season_info()
SIDX = {s: i for i, s in enumerate(SEAS)}
CLOCK = [s for s in SEAS if SI[s]["clock_ok"]]                 # 21시즌
NC = len(CLOCK)
CIDX = {s: i for i, s in enumerate(CLOCK)}
HUB = 1.345
KAPPA_GRID_S = [10.0 ** k for k in range(-2, 7)]                # 시즌별: 1e-2 … 1e6 (10배 간격)
KAPPA_GRID_C = [10.0 ** (k / 2) for k in range(-4, 13)]         # 합친 것: 1e-2 … 1e6 (반 자리 간격)
BOOT = 200
HH = 120.0
GD = 0.05                                                       # 식의 적분 격자
NG = int(round(HH / GD))
G = (np.arange(NG) + 0.5) * GD
EDGE = np.arange(NG + 1) * GD
CLK = (5.0 + G) % 24.0                                          # KST 시각 c
XREM = HH - G                                                   # 남은 시간 x
DAY = np.floor(G / 24.0).astype(int) + 1                        # 일차 d (05시 경계, 1~5)
GRID = np.arange(0, HH + 1e-9, 0.25)                            # 결과 격자
r4, rl = FL.r4, FL.rl


# ======================= 관측 =======================

def build_obs(obs):
    """모든 관측을 한 줄 꼴로: 구간 [ta, tb] (h), y, 무게 바탕 W0, 사진 여부, 시즌 번호(SEAS 번호).
    쌍: [t1, t2], ΔL, 1/var · 사진: [h, H_true], P_obs, 1/σ² (σ는 나중에 채움; 여기서는 1)."""
    D = FL.load_pairs(obs)
    prof = AC.load(os.path.join(T.R7, "wp11/act_profile_D1.json"))
    s1_M3 = FL.load_s1(os.path.join(T.R7, "stage1_M3_D1.json"))
    s1_act = FL.load_s1(os.path.join(T.R7, "wp11/stage1_M3act_D1.json"))
    PH = FL.build_photos(obs, prof, s1_M3, s1_act)
    pm = PH["main"]
    P = T.pairs(obs)
    var = np.array([p["var"] for p in P])
    S = np.concatenate([D["S"], PH["S"][pm]])
    ta = np.concatenate([D["t1"], PH["h"][pm]])
    tb = np.concatenate([D["t2"], PH["Ht"][pm]])
    y = np.concatenate([D["y"], PH["P"][pm]])
    isph = np.concatenate([np.zeros(D["n"], bool), np.ones(int(pm.sum()), bool)])
    wp = np.concatenate([1.0 / var, np.clip(np.floor(PH["h"][pm] / 24.0) + 1, 1, 5)])      # 사진 줄의 wp = 일차 번호 (σ_ph(d) 고르기용)
    o = np.argsort(S, kind="stable")
    O = {"S": S[o], "ta": ta[o], "tb": tb[o], "y": y[o], "isph": isph[o], "wp": wp[o], "n": len(S)}
    return O, PH, prof, s1_M3, s1_act


def sub(O, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in O.items() if k != "n"} | {"n": int(mask.sum())}


# ======================= 1. 스플라인 경험 곡선 =======================

_BAS = {}


def basis(Ht):
    k = round(Ht * 2)
    if k not in _BAS:
        nk = max(3, int(round(Ht)))
        kn = np.linspace(0.0, Ht, nk + 1)
        t = np.concatenate([[0.0] * 3, kn, [Ht] * 3])
        nb = nk + 3
        B = BSpline(t, np.eye(nb), 3, extrapolate=False)
        _BAS[k] = (B, B.antiderivative(), nb, Ht)
    return _BAS[k]


def design(Ht, ta, tb):
    """행마다 ∫_{ta}^{tb} B_j (계수 단위 1e-3 log10/시간)."""
    _, AD, nb, H = basis(Ht)
    f = lambda t: np.nan_to_num(AD(np.clip(t, 0.0, H - 1e-9)))  # noqa: E731
    return (f(tb) - f(ta)) * 1e-3


def d2pen(nb):
    D = np.zeros((nb - 2, nb))
    for i in range(nb - 2):
        D[i, i:i + 3] = (1, -2, 1)
    return D.T @ D


def solve_nn(X, y, W, kap, DtD):
    G_ = X.T @ (W[:, None] * X) + kap * DtD
    c = X.T @ (W * y)
    G_[np.diag_indices_from(G_)] += 1e-9 * np.trace(G_) / len(G_)
    L = np.linalg.cholesky(G_)
    b = solve_triangular(L, c, lower=True)
    beta, _ = nnls(L.T, b, maxiter=50 * len(b))
    return beta


class Sg(float):
    """사진 σ_ph (전체 값) + 일차별 σ_ph(d) (sd, 5개) + 쌍 분산 배수 phi (쌍 무게 = 1/(phi·var))."""
    phi = 1.0
    sd = None

    def __new__(cls, v, phi=1.0, sd=None):
        o = float.__new__(cls, v)
        o.phi = float(phi)
        o.sd = None if sd is None else np.asarray(sd, float)
        return o


def W0of(isph, wp, sig):
    """분산의 역수 무게: 사진 1/σ_ph(d)² (d = 일차; 사진 줄의 wp에 일차 번호를 담음), 쌍 1/(φ var) (wp = 1/var)."""
    sd = sig.sd if sig.sd is not None else np.full(5, float(sig))
    dd = np.clip(np.nan_to_num(wp).astype(int) - 1, 0, 4)
    return np.where(isph, 1.0 / sd[dd] ** 2, wp / sig.phi)


def psi(r, W0):
    """Huber 무게 (쌍·사진 모두, 표준화 잔차 u = |r|√W0, c = 1.345)."""
    u = np.abs(r) * np.sqrt(W0)
    return np.where(u > HUB, HUB / np.maximum(u, 1e-12), 1.0)


def rho_loss(r, W0, isph=None, sig=None):
    """가중 Huber 손실 (표준화 잔차 u = |r|√W0): u ≤ c면 u², 아니면 2c·u − c² (쌍·사진 모두)."""
    u = np.abs(r) * np.sqrt(W0)
    return np.where(u <= HUB, u * u, 2 * HUB * u - HUB * HUB)


def fit_spl(X, y, W0, isph, si, ns, kap, DtD, sig, iters=14, m0=None):
    """ΔL = m_s (Xβ) — 번갈아: Huber 무게 → β (NNLS) → m (닫힌 꼴, 기하평균 1). ns == 1이면 m 고정 1."""
    m = np.ones(ns) if m0 is None else m0.copy()
    ps = np.ones(len(y))
    beta = None
    for it in range(iters):
        W = W0 * ps
        Xm = X * m[si][:, None]
        beta = solve_nn(Xm, y, W, kap, DtD)
        f = X @ beta
        if ns > 1:
            num = np.bincount(si, W * y * f, ns)
            den = np.bincount(si, W * f * f, ns)
            ok = den > 0
            mn = m.copy()
            mn[ok] = np.maximum(num[ok] / den[ok], 1e-3)
            gm = math.exp(np.mean(np.log(mn[ok])))
            mn /= gm
            beta = beta * gm
            dm = np.max(np.abs(mn - m))
            m = mn
        else:
            dm = 0.0
        r = y - m[si] * (X @ beta)
        psn = psi(r, W0)
        dp = np.max(np.abs(psn - ps)) if len(ps) else 0
        ps = psn
        if it > 1 and dm < 1e-5 and dp < 1e-4:
            break
    r = y - m[si] * (X @ beta)
    return beta, m, r


def sig_of(r, isph, wp=None):
    """(σ_ph 전체, φ_pair, σ_ph(d) 일차별): 사진 잔차의 1.4826 MAD (일차 1~5마다, 사진 8장 미만이면 전체 값), 쌍 표준화 잔차 r√wp의 (1.4826 MAD)²."""
    v = r[isph]
    mad = lambda z: float(1.4826 * np.median(np.abs(z - np.median(z))))  # noqa: E731
    sp = 0.03 if len(v) < 5 else mad(v)
    sd = np.full(5, max(sp, 0.005))
    if wp is not None and len(v) >= 5:
        dd = np.clip(wp[isph].astype(int) - 1, 0, 4)
        for k in range(5):
            if (dd == k).sum() >= 8:
                sd[k] = max(mad(v[dd == k]), 0.005)
    phi = 1.0
    if wp is not None and (~isph).sum() >= 5:
        z = (r * np.sqrt(wp))[~isph]
        phi = float(mad(z) ** 2)
    return max(sp, 0.005), max(phi, 0.05), sd


def fit_full(X, y, W0pair, isph, si, ns, kap, DtD, sig0=None, rounds=3):
    """σ_ph와 φ_pair를 되풀이로 정하며 적합. W0pair: 쌍 1/var (사진 칸은 쓰지 않음)."""
    sig = sig0 if sig0 is not None else Sg(0.03, 1.0)
    for _ in range(rounds):
        W0 = W0of(isph, W0pair, sig)
        beta, m, r = fit_spl(X, y, W0, isph, si, ns, kap, DtD, sig)
        a, b, c3 = sig_of(r, isph, W0pair)
        sig = Sg(a, b, c3)
    W0 = W0of(isph, W0pair, sig)
    return beta, m, r, sig, W0


def cv_kappa(X, y, W0pair, isph, si, ns, DtD, sig, grid, rng, nf=5, kap_hint=None):
    n = len(y)
    fold = rng.permutation(n) % nf
    W0 = W0of(isph, W0pair, sig)
    err = []
    for kap in grid:
        e = 0.0
        for f in range(nf):
            tr = fold != f
            te = ~tr
            if ns > 1:
                # 훈련에 관측이 없는 시즌은 없다 (시즌마다 수십 줄 이상) — 없으면 m = 1
                pass
            beta, m, _ = fit_spl(X[tr], y[tr], W0[tr], isph[tr], si[tr], ns, kap, DtD, sig, iters=8)
            r = y[te] - m[si[te]] * (X[te] @ beta)
            e += float(rho_loss(r, W0[te], isph[te], sig).sum())
        err.append(e)
    return np.array(err)


def pick_kappa(err, grid):
    return grid[int(np.argmin(err))]


def lam_pred(beta, Ht, g):
    """λ(g) = Σ β_j B_j(g) (단위 1e-3 log10/시간) → log10/시간."""
    B, _, nb, H = basis(Ht)
    return np.nan_to_num(B(np.clip(g, 0, H - 1e-9))) @ beta * 1e-3


def curves_from_beta(beta, Ht, grid):
    lam = lam_pred(beta, Ht, grid)
    _, AD, nb, H = basis(Ht)
    Pc = (np.nan_to_num(AD(np.clip(np.array([Ht]), 0, Ht - 1e-9))) @ beta * 1e-3)[0] - np.nan_to_num(AD(np.clip(grid, 0, Ht - 1e-9))) @ beta * 1e-3
    return lam, Pc


# 합친 적합용 (21시즌) 준비

def pack_clock(O):
    msk = np.isin(O["S"], [SIDX[s] for s in CLOCK])
    Oc = sub(O, msk)
    Oc["si"] = np.array([CIDX[SEAS[i]] for i in Oc["S"]])
    return Oc


def comb_X(Oc):
    return design(HH, Oc["ta"], Oc["tb"])


def boot_job(args):
    seed, = args
    rng = np.random.default_rng(seed)
    X, y, W0p, isph, si, kap, sig = BG["X"], BG["y"], BG["W0p"], BG["isph"], BG["si"], BG["kap"], BG["sig"]
    by = [np.where(si == j)[0] for j in range(NC)]
    dr = rng.integers(0, NC, NC)
    rows = [by[d] for d in dr]
    idx = np.concatenate(rows)
    sj = np.concatenate([np.full(len(r), j) for j, r in enumerate(rows)])
    W0 = W0of(isph[idx], W0p[idx], sig)
    beta, m, _ = fit_spl(X[idx], y[idx], W0, isph[idx], sj, NC, kap, BG["DtD"], sig, iters=10)
    lam, Pc = curves_from_beta(beta, HH, GRID)
    return lam, Pc


BG = {}


# ======================= 2. 식 =======================

class Cand:
    def __init__(self, base, K, end, gam):
        self.base, self.K, self.end, self.gam = base, K, end, gam
        self.name = f"{'①' if K == 0 else ('②' if not end else ('③' if not gam else '④'))}{base}" + (f"K{K}" if K else "")
        self.npar = 1 + (base == "b") + 2 * K + 2 * end + gam           # 적합 모수 수 (A 뺀 θ)
        lo, hi = [0.0], [3.0]
        if base == "b":
            lo.append(0.0)
            hi.append(10.0)
        lo += [-4.0] * (2 * K)
        hi += [4.0] * (2 * K)
        if end:
            lo += [0.0, 0.3]
            hi += [200.0, 400.0]
        if gam:
            lo.append(-0.49)
            hi.append(0.49)
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.i_four = 1 + (base == "b")
        self.i_end = self.i_four + 2 * K
        self.i_gam = self.i_end + 2 * end


def all_cands():
    out = []
    for b in "ab":
        out.append(Cand(b, 0, False, False))
    for b in "ab":
        for K in (1, 2, 3):
            out.append(Cand(b, K, False, False))
    for b in "ab":
        for K in (1, 2, 3):
            out.append(Cand(b, K, True, False))
    for b in "ab":
        for K in (1, 2, 3):
            out.append(Cand(b, K, True, True))
    return out


_COS = {k: np.cos(2 * math.pi * k * CLK / 24.0) for k in (1, 2, 3)}
_SIN = {k: np.sin(2 * math.pi * k * CLK / 24.0) for k in (1, 2, 3)}


def lam_grid(c, th, dal=0.0, wsc=1.0):
    """격자 중점 λ (A = 1). dal = α에 더하는 시즌 몫, wsc = 푸리에 합에 곱하는 시즌 몫."""
    lam = G ** (-(th[0] + dal))
    if c.base == "b":
        lam = lam + th[1]
    if c.K:
        fs = np.zeros(NG)
        for k in range(1, c.K + 1):
            a, b = th[c.i_four + 2 * (k - 1)], th[c.i_four + 2 * (k - 1) + 1]
            fs += a * _COS[k] + b * _SIN[k]
        if c.gam:
            fs = fs * (1.0 + th[c.i_gam] * (DAY - 3))
        lam = lam * np.exp(wsc * fs)
    if c.end:
        lam = lam * (1.0 + th[c.i_end] * np.exp(-XREM / th[c.i_end + 1]))
    return lam


def cum_edges(lam):
    return np.concatenate([[0.0], np.cumsum(lam * GD)])


def interval_f(lam, ta, tb):
    Ce = cum_edges(lam)
    return np.interp(tb, EDGE, Ce) - np.interp(ta, EDGE, Ce)


def mfit(f, y, W, si, ns):
    num = np.bincount(si, W * y * f, ns)
    den = np.bincount(si, W * f * f, ns)
    m = np.ones(ns)
    ok = den > 0
    m[ok] = np.maximum(num[ok] / den[ok], 1e-6)
    return m


def resid_param(th, c, Oc, W):
    f = interval_f(lam_grid(c, th), Oc["ta"], Oc["tb"])
    m = mfit(f, Oc["y"], W, Oc["si"], NC)
    return np.sqrt(W) * (Oc["y"] - m[Oc["si"]] * f)


def fit_param(c, Oc, starts, sig, rounds=2, W0p_key="wp"):
    """θ 적합 (A 소거, m_s 닫힌 꼴, Huber IRLS 2번). 여러 시작점 중 최소 손실. -> (θ, m, cost, W0)"""
    isph = Oc["isph"]
    W0 = W0of(isph, Oc["wp"], sig)
    ps = np.ones(Oc["n"])
    bestth = None
    for rd in range(rounds):
        W = W0 * ps
        best = None
        for s0 in (starts if rd == 0 else [bestth]):
            try:
                res = least_squares(resid_param, np.clip(s0, c.lo + 1e-6, c.hi - 1e-6), args=(c, Oc, W), bounds=(c.lo, c.hi), max_nfev=60, xtol=1e-8, ftol=1e-9)
            except Exception:
                continue
            if best is None or res.cost < best.cost:
                best = res
        bestth = best.x
        f = interval_f(lam_grid(c, bestth), Oc["ta"], Oc["tb"])
        m = mfit(f, Oc["y"], W, Oc["si"], NC)
        r = Oc["y"] - m[Oc["si"]] * f
        ps = psi(r, W0)
    f = interval_f(lam_grid(c, bestth), Oc["ta"], Oc["tb"])
    W = W0 * ps
    m = mfit(f, Oc["y"], W, Oc["si"], NC)
    r = Oc["y"] - m[Oc["si"]] * f
    cost = float(rho_loss(r, W0, isph, sig).sum())
    return bestth, m, cost, W0


def default_starts(c, rng, warm=None):
    st = []
    if warm is not None:
        st.append(warm)
    for al in (0.2, 0.6, 1.0):
        t = np.zeros(c.npar)
        t[0] = al
        if c.base == "b":
            t[1] = 0.05
        if c.end:
            t[c.i_end], t[c.i_end + 1] = 0.5, 3.0
        if c.K:
            t[c.i_four:c.i_four + 2 * c.K] = rng.normal(0, 0.3, 2 * c.K)
        st.append(t)
    return st


def parent_idx(c_all, c):
    key = (c.base, c.K, c.end, c.gam)
    if c.gam:
        want = (c.base, c.K, True, False)
    elif c.end:
        want = (c.base, c.K, False, False)
    elif c.K > 1:
        want = (c.base, c.K - 1, False, False)
    elif c.K == 1:
        want = (c.base, 0, False, False)
    elif c.base == "b":
        want = ("a", 0, False, False)
    else:
        return None
    return next(k for k, cc in enumerate(c_all) if (cc.base, cc.K, cc.end, cc.gam) == want)


def lift(prev_c, prev_th, c):
    """한 단계 작은 식의 해에서 큰 식의 시작점 (새 모수는 0 또는 기본)."""
    t = np.zeros(c.npar)
    t[0] = prev_th[0]
    if c.base == "b":
        t[1] = prev_th[1] if prev_c.base == "b" else 0.05
    nk = 2 * min(prev_c.K, c.K)
    if nk:
        t[c.i_four:c.i_four + nk] = prev_th[prev_c.i_four:prev_c.i_four + nk]
    if c.end:
        if prev_c.end:
            t[c.i_end:c.i_end + 2] = prev_th[prev_c.i_end:prev_c.i_end + 2]
        else:
            t[c.i_end], t[c.i_end + 1] = 0.5, 3.0
    return t


def season_loss(c, th, Oc, sig, s, W0=None):
    """뺀 시즌 s의 관측: m_s만 다시 맞춘 가중 손실 (Huber IRLS 3번)."""
    ix = np.where(Oc["si"] == s)[0]
    f = interval_f(lam_grid(c, th), Oc["ta"][ix], Oc["tb"][ix])
    y = Oc["y"][ix]
    isph = Oc["isph"][ix]
    w0 = W0of(isph, Oc["wp"][ix], sig)
    ps = np.ones(len(ix))
    zero = np.zeros(len(ix), int)
    for _ in range(4):
        m = mfit(f, y, w0 * ps, zero, 1)[0]
        ps = psi(y - m * f, w0)
    r = y - m * f
    return float(rho_loss(r, w0, isph, sig).sum()), len(ix)


def loso_job(args):
    ci, s, th0 = args
    c = CANDS[ci]
    Oc = LG["Oc"]
    msk = Oc["si"] != s
    Os = sub(Oc, msk)
    Os["si"] = Oc["si"][msk]
    rng = np.random.default_rng(1000 + s)
    starts = [th0]
    th, _, _, _ = fit_param(c, Os, starts, LG["sig"])
    loss, n = season_loss(c, th, Oc, LG["sig"], s)
    return ci, s, loss, n, th


LG = {}
CANDS = all_cands()


def boot_param_job(args):
    seed, = args
    c = CANDS[LG["ci"]]
    Oc = LG["Oc"]
    rng = np.random.default_rng(seed)
    by = [np.where(Oc["si"] == j)[0] for j in range(NC)]
    dr = rng.integers(0, NC, NC)
    rows = [by[d] for d in dr]
    idx = np.concatenate(rows)
    sj = np.concatenate([np.full(len(r), j) for j, r in enumerate(rows)])
    Ob = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in Oc.items() if k != "n"}
    Ob["si"] = sj
    Ob["n"] = len(idx)
    th, m, _, _ = fit_param(c, Ob, [LG["th0"]], LG["sig"])
    # A 환산: m 기하평균
    mm = m[np.bincount(sj, minlength=NC) > 0]
    gm = math.exp(np.mean(np.log(mm)))
    lam = gm * lam_grid(c, th)
    Ce = cum_edges(lam)
    Pg = Ce[-1] - np.interp(GRID, EDGE, Ce)
    return th, gm, lam_at(lam, GRID), Pg


def lam_at(lam, grid):
    return np.interp(grid, G, lam)


# ======================= 시즌 차이 (m_s / +α / +파 크기) =======================

def vary_resid(x, c, nth, Oc, W, mode):
    th = x[:nth]
    ex = x[nth:]
    f = np.empty(Oc["n"])
    for s in range(NC):
        ix = Oc["ix"][s]
        if len(ix) == 0:
            continue
        if mode == "alpha":
            lam = lam_grid(c, th, dal=ex[s])
        elif mode == "wave":
            lam = lam_grid(c, th, wsc=ex[s])
        else:
            lam = lam_grid(c, th)
        f[ix] = interval_f(lam, Oc["ta"][ix], Oc["tb"][ix])
    m = mfit(f, Oc["y"], W, Oc["si"], NC)
    return np.sqrt(W) * (Oc["y"] - m[Oc["si"]] * f)


def fit_vary(c, Oc, th0, sig, mode, rounds=2, ex0=None):
    """공통 θ + 시즌마다 α_s (모드 alpha, 0 기준) 또는 파 배수 w_s (모드 wave, 1 기준). m_s 닫힌 꼴. jac 희소."""
    nth = len(th0)
    Oc = dict(Oc)
    Oc["ix"] = [np.where(Oc["si"] == s)[0] for s in range(NC)]
    isph = Oc["isph"]
    W0 = W0of(isph, Oc["wp"], sig)
    ps = np.ones(Oc["n"])
    base_ex = 0.0 if mode == "alpha" else 1.0
    x = np.concatenate([th0, np.full(NC, base_ex)]) if ex0 is None else np.concatenate([th0, ex0])
    lo = np.concatenate([c.lo, np.full(NC, -1.0 if mode == "alpha" else 0.0)])
    hi = np.concatenate([c.hi, np.full(NC, 1.0 if mode == "alpha" else 4.0)])
    # 희소 구조: θ 열은 모두, 시즌 열은 그 시즌 행
    from scipy.sparse import lil_matrix
    sp = lil_matrix((Oc["n"], nth + NC), dtype=int)
    sp[:, :nth] = 1
    for s in range(NC):
        sp[Oc["ix"][s], nth + s] = 1
    # m_s가 모든 시즌 θ에 닿으므로 한 시즌 열은 자기 시즌 행만 — m_s 소거는 시즌 안에서만 닫히므로 성립
    for rd in range(rounds):
        W = W0 * ps
        res = least_squares(vary_resid, np.clip(x, lo + 1e-6, hi - 1e-6), args=(c, nth, Oc, W, mode), bounds=(lo, hi), jac_sparsity=sp, max_nfev=40, xtol=1e-8, ftol=1e-9)
        x = res.x
        rr = _vary_r(x, c, nth, Oc, W, mode)
        ps = psi(rr, W0)
    rr = _vary_r(x, c, nth, Oc, W0 * ps, mode)
    cost = float(rho_loss(rr, W0, isph, sig).sum())
    return x[:nth], x[nth:], cost


def _vary_r(x, c, nth, Oc, W, mode):
    th, ex = x[:nth], x[nth:]
    f = np.empty(Oc["n"])
    for s in range(NC):
        ix = Oc["ix"][s]
        if len(ix) == 0:
            continue
        lam = lam_grid(c, th, dal=ex[s]) if mode == "alpha" else (lam_grid(c, th, wsc=ex[s]) if mode == "wave" else lam_grid(c, th))
        f[ix] = interval_f(lam, Oc["ta"][ix], Oc["tb"][ix])
    m = mfit(f, Oc["y"], W, Oc["si"], NC)
    return Oc["y"] - m[Oc["si"]] * f


def half_loss(c, th, Oc, sig, s, mode, rng):
    """뺀 시즌 s 안을 무작위 반반 (관측 단위): 한 쪽에서 m_s (+ 시즌 몫 하나)를 맞추고 다른 쪽에서 가중 손실, 서로 바꿔 합."""
    ix = np.where(Oc["si"] == s)[0]
    perm = rng.permutation(len(ix))
    halves = [ix[perm[: len(ix) // 2]], ix[perm[len(ix) // 2:]]]
    tot = 0.0
    for a, b in ((0, 1), (1, 0)):
        tr, te = halves[a], halves[b]

        def fx(ex):
            if mode == "alpha":
                return lam_grid(c, th, dal=ex)
            if mode == "wave":
                return lam_grid(c, th, wsc=ex)
            return lam_grid(c, th)

        def fit_half(ex):
            f = interval_f(fx(ex), Oc["ta"][tr], Oc["tb"][tr])
            y = Oc["y"][tr]
            isph = Oc["isph"][tr]
            w0 = W0of(isph, Oc["wp"][tr], sig)
            m = mfit(f, y, w0, np.zeros(len(tr), int), 1)[0]
            return m, w0, y, isph, f

        if mode == "m":
            m, *_ = fit_half(0.0)
            ex = 0.0
        else:
            lo, hi, x0 = ((-1.0, 1.0, 0.0) if mode == "alpha" else (0.0, 4.0, 1.0))

            def rs(z):
                m_, w0, y, isph, f = fit_half(z[0])
                return np.sqrt(w0) * (y - m_ * f)
            res = least_squares(rs, [x0], bounds=([lo], [hi]), max_nfev=30)
            ex = res.x[0]
            m, *_ = fit_half(ex)
        f = interval_f(fx(ex), Oc["ta"][te], Oc["tb"][te])
        y = Oc["y"][te]
        isph = Oc["isph"][te]
        w0 = W0of(isph, Oc["wp"][te], sig)
        tot += float(rho_loss(y - m * f, w0, isph, sig).sum())
    return tot


def vary_loso_job(args):
    s, mode, th0, seed = args
    c = CANDS[LG["ci"]]
    Oc = LG["Oc"]
    msk = Oc["si"] != s
    Os = {k: (v[msk] if isinstance(v, np.ndarray) else v) for k, v in Oc.items() if k != "n"}
    Os["n"] = int(msk.sum())
    Os["si"] = Oc["si"][msk]
    # 빠진 시즌은 관측 0줄 (ix가 비어 건너뜀)
    if mode == "m":
        th = th0
    else:
        th, _, _ = fit_vary(c, Os, th0, LG["sig"], mode, rounds=1)
    rng = np.random.default_rng(seed)
    return s, mode, half_loss(c, th, Oc, LG["sig"], s, mode, rng)


# ======================= 서술 숫자 =======================

def smooth1(lam):
    """1시간 이동 평균 (격자 0.25)."""
    k = np.ones(4) / 4
    return np.convolve(lam, k, mode="same")


def describe(lam, Pc, lamGRID_clock):
    """합친 경험 곡선 λ(GRID)·P(GRID)에서 꼴 숫자."""
    out = {}
    def integ(a, b):
        m = (GRID >= a - 1e-9) & (GRID <= b + 1e-9)
        return float(np.trapezoid(lam[m], GRID[m]))
    out["day_sum"] = [r4(integ(6 if d == 0 else 24 * d, 24 * d + 24)) for d in range(5)]      # 1일차는 자료가 있는 h 6~24만
    out["day1_sum_h0_24_extrapolated_extra"] = r4(integ(0, 24))
    out["P_at_h6"] = r4(float(np.interp(6.0, GRID, Pc)))
    out["P0_extrapolated_extra"] = r4(Pc[0])
    lamc = lam
    # 같은 KST 시각의 2~4일차 평균 (기준 모양)
    def clockmean(days):
        idx = [(GRID >= 24 * d - 1e-9) & (GRID < 24 * d + 24 - 1e-9) for d in days]
        return np.mean([lam[i] for i in idx], axis=0)
    cm24 = clockmean([1, 2, 3])
    cm34 = clockmean([2, 3])
    sm = smooth1(lam)
    out["wave_days"] = []
    for d in range(5):
        seg = sm[(GRID >= 24 * d) & (GRID < 24 * d + 24)]
        tt = GRID[(GRID >= 24 * d) & (GRID < 24 * d + 24)]
        if d == 0:
            seg, tt = seg[tt >= 6.0], tt[tt >= 6.0]            # 첫날은 자료가 있는 h 6~24만
        seg2, tt2 = seg[2:-2], tt[2:-2]
        imax, imin = int(np.argmax(seg2)), int(np.argmin(seg2))
        out["wave_days"].append({"day": d + 1, "note": ("h 6~24만 봄 (h<6은 자료 없음)" if d == 0 else None), "fast_kst": r4((5 + tt2[imax]) % 24, 2), "fast": r4(seg2[imax], 5), "slow_kst": r4((5 + tt2[imin]) % 24, 2), "slow": r4(seg2[imin], 5),
                                  "ratio_fast_slow": r4(seg2[imax] / max(seg2[imin], 1e-9), 2) if seg2[imin] > 1e-6 else None})
    c = (5 + GRID[:96]) % 24
    sm2 = smooth1(cm24)
    t_ = GRID[:96]
    out["wave_days2_4"] = {"fast_kst": r4((5 + t_[3:-3][int(np.argmax(sm2[3:-3]))]) % 24, 2), "fast": r4(sm2[3:-3].max(), 5),
                           "slow_kst": r4((5 + t_[3:-3][int(np.argmin(sm2[3:-3]))]) % 24, 2), "slow": r4(sm2[3:-3].min(), 5),
                           "ratio_fast_slow": r4(sm2[3:-3].max() / max(sm2[3:-3].min(), 1e-9), 2) if sm2[3:-3].min() > 1e-6 else None}
    # 05–11 (h 0–6 같은 시각) 몫: 3~5일차
    def share(days, a, b):
        tot = sum(integ(24 * d, 24 * d + 24) for d in days)
        part = sum(integ(24 * d + a, 24 * d + b) for d in days)
        return part / tot
    out["share_3_5_6h_blocks"] = [r4(share([2, 3, 4], 6 * k, 6 * k + 6)) for k in range(4)]
    # 마지막 날
    out["last_day_sum"] = r4(integ(96, 120))
    out["share_last6_of_last24"] = r4(integ(114, 120) / integ(96, 120))
    out["last_mean_speed"] = {k: r4(integ(120 - k, 120) / k, 5) for k in (1, 3, 6)}
    # 마지막 k시간과 같은 KST 시각의 3~4일차 평균 속도 (마지막 날 x ∈ [0,k] = KST (5−k)~5)
    rat = {}
    for k in (1, 3, 6):
        a, b = 24 - k, 24
        m = (GRID[:96] >= a - 1e-9) & (GRID[:96] <= b + 1e-9)  # 일차 안 시각 24−k~24 = KST 05−k ~ 05
        rat[k] = r4(integ(120 - k, 120) / k / (float(np.trapezoid(cm34[m], GRID[:96][m])) / k), 2)
    out["last_over_days3_4_same_clock"] = rat
    # 몰림 시작: 마지막 날 안 최저 (1시간 평활)와, 3~4일차 같은 시각 대비 1.25배를 끝까지 넘는 첫 시각
    ld = (GRID >= 96) & (GRID <= 120)
    sml = smooth1(lam)
    j0 = np.where(ld)[0]
    jj = j0[2:-2]
    imin = jj[int(np.argmin(sml[jj]))]
    out["last_day_min"] = {"h": r4(GRID[imin], 2), "kst": r4((5 + GRID[imin]) % 24, 2), "x": r4(120 - GRID[imin], 2), "lam": r4(sml[imin], 5)}
    rr = lam[j0] / np.maximum(cm34[(j0 - 96) % 96], 1e-9)
    hq = GRID[j0]
    ok_end = hq <= 118.0 + 1e-9                      # 마지막 2시간은 가장자리라 판정에서 뺌
    start = None
    for q_ in range(len(hq)):
        if ok_end[q_] and np.all(rr[q_:][ok_end[q_:]] >= 1.25):
            start = hq[q_]
            break
    out["crowd_start_ratio125_until_x2"] = {"h": r4(start, 2) if start is not None else None, "x": r4(120 - start, 2) if start is not None else None, "kst": r4((5 + start) % 24, 2) if start is not None else None}
    out["last_day_ratio_to_days3_4_2h"] = [{"x_from": r4(120 - (96 + 2 * k), 1), "kst": r4((5 + 96 + 2 * k) % 24, 1), "lam": r4(float(np.mean(lam[(GRID >= 96 + 2 * k) & (GRID < 98 + 2 * k)])), 5),
                                            "ratio": r4(float(np.mean(lam[(GRID >= 96 + 2 * k) & (GRID < 98 + 2 * k)]) / np.mean(cm34[2 * k * 4: 2 * k * 4 + 8])), 2)} for k in range(12)]
    # 첫날
    out["first_mean_speed"] = {"0-6_EXTRAPOLATED(자료 없음)": r4(integ(0, 6) / 6, 5), "6-12": r4(integ(6, 12) / 6, 5), "12-24": r4(integ(12, 24) / 12, 5)}
    out["first_day_sum_from_h6"] = r4(integ(6, 24))
    out["first_day_sum_0_6_EXTRAPOLATED"] = r4(integ(0, 6))
    out["first_over_days2_4_same_clock"] = {f"{a}-{b}": r4(integ(a, b) / (float(np.trapezoid(cm24[(GRID[:96] >= a) & (GRID[:96] <= b)], GRID[:96][(GRID[:96] >= a) & (GRID[:96] <= b)]))), 2) for a, b in ((6, 12), (12, 24))}
    return out


def family_limit(c, th0, Pe, seed, extra=None):
    """(더 본 것, 고르기에 안 씀) 식 꼴 자체의 한계: 자료가 아니라 합친 경험 P(h) 곡선(h 6~120)에 최소제곱으로 맞췄을 때 (A는 닫힌 꼴) 얼마나 가까워지나."""
    mk = GRID >= 6.0 - 1e-9
    rng = np.random.default_rng(seed)

    def resid(th):
        Ce = cum_edges(lam_grid(c, th))
        f = (Ce[-1] - np.interp(GRID, EDGE, Ce))[mk]
        A = (f @ Pe[mk]) / (f @ f)
        return A * f - Pe[mk]
    best = None
    for s0 in [th0] + ([extra] if extra is not None else []) + default_starts(c, rng):
        try:
            res = least_squares(resid, np.clip(s0, c.lo + 1e-6, c.hi - 1e-6), bounds=(c.lo, c.hi), max_nfev=200, xtol=1e-10, ftol=1e-12)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    d = resid(best.x)
    hh = GRID[mk]
    return best.x, {"max_abs_dP": r4(float(np.max(np.abs(d)))), "rms_dP": r4(float(np.sqrt(np.mean(d * d)))), "worst_h": r4(float(hh[int(np.argmax(np.abs(d)))]), 2),
            "frac_h_over_0.01": r4(float(np.mean(np.abs(d) > 0.01)), 3), "at_bound": [int(i) for i in np.where((best.x <= c.lo + 1e-4) | (best.x >= c.hi - 1e-4))[0]]}


def maxabs(d, mask):
    v = d[mask]
    return float(np.max(np.abs(v))), float(np.sqrt(np.mean(v * v)))


# ======================= 본 계산 =======================

def main():
    global BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    BOOT = args.boot
    t00 = time.time()
    rng = np.random.default_rng(1204)

    def log(msg):
        print(f"[{time.time() - t00:6.0f}s] {msg}", flush=True)

    O, PH, prof, s1_M3, s1_act = build_obs(args.obs_dir)
    npair = int((~O["isph"]).sum())
    log(f"obs {O['n']} (pairs {npair}, photos {O['n'] - npair}), seasons {len(SEAS)}")
    pool = mp.get_context("fork")

    # ---------- 1a 시즌별 ----------
    seas_res = {}
    per_tab = []
    for s in SEAS:
        i = SIDX[s]
        mk = O["S"] == i
        Os = sub(O, mk)
        Ht = FL.htrue(s)
        X = design(Ht, Os["ta"], Os["tb"])
        nb = X.shape[1]
        DtD = d2pen(nb)
        si = np.zeros(Os["n"], int)
        # κ 정하기: 중간 κ로 σ를 먼저 잡고 격자 교차검증
        _, _, r0, sig, _ = fit_full(X, Os["y"], Os["wp"], Os["isph"], si, 1, 10.0, DtD, Sg(0.03, 1.0), rounds=2)
        err = cv_kappa(X, Os["y"], Os["wp"], Os["isph"], si, 1, DtD, sig, KAPPA_GRID_S, np.random.default_rng(7 + i))
        kap = pick_kappa(err, KAPPA_GRID_S)
        beta, m, r, sig, W0 = fit_full(X, Os["y"], Os["wp"], Os["isph"], si, 1, kap, DtD, sig, rounds=3)
        gs = np.arange(0, Ht + 1e-9, 0.25)
        lam, Pc = curves_from_beta(beta, Ht, gs)
        np_ = int((~Os["isph"]).sum())
        seas_res[s] = {"Ht": Ht, "grid": gs, "lam": lam, "P": Pc, "kappa": kap, "sig": sig, "beta": beta, "n_pair": np_, "n_photo": int(Os["isph"].sum()), "cv_edge": bool(kap in (KAPPA_GRID_S[0], KAPPA_GRID_S[-1]))}
        rp = r[~Os["isph"]]
        per_tab.append({"season": s, "pairs": np_, "photos": int(Os["isph"].sum()), "Ht": r4(Ht, 1), "kappa": kap, "kappa_at_grid_edge": seas_res[s]["cv_edge"], "sigma_photo": r4(sig, 4), "phi_pair": r4(sig.phi, 3), "sigma_photo_by_day": rl(sig.sd, 4),
                        "P0": r4(Pc[0]), "rms_pair_resid_over_sd": r4(float(np.sqrt(np.mean(rp ** 2 * Os["wp"][~Os["isph"]]))), 2) if len(rp) else None,
                        "small": bool(np_ < FL.MINN or Os["isph"].sum() < FL.MINN)})
    log("per-season done")

    # ---------- 1b 합친 곡선 ----------
    Oc = pack_clock(O)
    Xc = comb_X(Oc)
    nbc = Xc.shape[1]
    DtDc = d2pen(nbc)
    _, _, r0, sigc, _ = fit_full(Xc, Oc["y"], Oc["wp"], Oc["isph"], Oc["si"], NC, 10.0, DtDc, Sg(0.03, 1.0), rounds=2)
    errc = cv_kappa(Xc, Oc["y"], Oc["wp"], Oc["isph"], Oc["si"], NC, DtDc, sigc, KAPPA_GRID_C, np.random.default_rng(99))
    kapc = pick_kappa(errc, KAPPA_GRID_C)
    betac, mc, rc, sigc, W0c = fit_full(Xc, Oc["y"], Oc["wp"], Oc["isph"], Oc["si"], NC, kapc, DtDc, sigc, rounds=3)
    lamc, Pcc = curves_from_beta(betac, HH, GRID)
    log(f"combined: κ={kapc:g} σ_ph={sigc:.4f} n={Oc['n']} (pairs {int((~Oc['isph']).sum())}, photos {int(Oc['isph'].sum())})")
    BG.update(X=Xc, y=Oc["y"], W0p=Oc["wp"], isph=Oc["isph"], si=Oc["si"], kap=kapc, sig=sigc, DtD=DtDc)
    with pool.Pool(args.workers) as pl:
        bres = pl.map(boot_job, [(5000 + b,) for b in range(BOOT)], chunksize=5)
    bl = np.array([b[0] for b in bres])
    bp = np.array([b[1] for b in bres])
    lam_lo, lam_hi = FL.band(bl)
    P_lo, P_hi = FL.band(bp)
    log(f"boot {BOOT} done")

    # 합친 곡선의 시즌별 선 (시즌 배수 곱한 시즌 곡선 = 합친 곡선 m_s 배)
    desc = describe(lamc, Pcc, None)
    pct_curve = 3.0 * 10.0 ** (-Pcc)
    # 점검: 칸 요약 값
    fl = json.load(open(os.path.join(T.R7, "wp12", "flood_D1.json"), encoding="utf-8"))
    chk = []
    chk.append({"item": "마지막 날 침수 합 log10", "bin_summary": 0.163, "curve": desc["last_day_sum"]})
    chk.append({"item": "H−6~H 몫 (마지막 날 대비)", "bin_summary": 0.38, "curve": desc["share_last6_of_last24"]})
    for d, v in enumerate((0.929, 0.294, 0.122, 0.129, 0.168)):
        chk.append({"item": f"{d + 1}일차 합 log10", "bin_summary": v, "curve": desc["day_sum"][d]})
    for k, v in enumerate((0.10, 0.16, 0.43, 0.31)):
        chk.append({"item": f"3~5일차 6시간 몫 {['05-11', '11-17', '17-23', '23-05'][k]}", "bin_summary": v, "curve": desc["share_3_5_6h_blocks"][k]})
    # 남은 2~4시간 사진 P / F
    ratio_F = {}
    for xx in (2, 3, 4):
        h_ = HH - xx
        ratio_F[xx] = r4(float(np.interp(h_, GRID, Pcc)) / T.F_P(h_ / HH), 2)
    chk.append({"item": "남은 2/3/4시간 경험 P ÷ F", "bin_summary": "0.3~0.7", "curve": ratio_F})
    for c_ in chk:
        if isinstance(c_["bin_summary"], float) and isinstance(c_["curve"], float):
            c_["diff"] = r4(c_["curve"] - c_["bin_summary"])

    # ---------- G1 표·그림 ----------
    tbl_comb = [{"h": float(GRID[i]), "kst": r4((5 + GRID[i]) % 24, 2), "lam": r4(lamc[i], 5), "lam_lo": r4(lam_lo[i], 5), "lam_hi": r4(lam_hi[i], 5),
                 "P": r4(Pcc[i], 5), "P_lo": r4(P_lo[i], 5), "P_hi": r4(P_hi[i], 5), "pct_at_3final": r4(3 * 10 ** (-Pcc[i]), 4)} for i in range(0, len(GRID), 1)]
    mt = [{"season": s, "m": r4(mc[CIDX[s]], 4), "pairs": int(((Oc["si"] == CIDX[s]) & ~Oc["isph"]).sum()), "photos": int(((Oc["si"] == CIDX[s]) & Oc["isph"]).sum())} for s in CLOCK]
    # 시즌 곡선(합친 모양에 시즌 배수)과 시즌 자체 곡선
    gs5 = np.arange(0, HH + 1e-9, 0.5)
    plots = []
    plots.append({"id": "G1a", "part": "G1", "title": "합친 침수 속도 λ(h) (21시즌, 칸 없이 스플라인) + 시즌 붓스트랩 10~90%",
                  "x": {"label": "시작부터 시간 h (05시 시작)", "log": False, "reverse": False}, "y": {"label": "침수 속도 λ (log10 / 시간)", "log": False},
                  "series": [{"name": "λ 10~90% 띠", "kind": "band", "x": rl(GRID, 2), "y": rl(lamc, 5), "lo": rl(lam_lo, 5), "hi": rl(lam_hi, 5), "group": "data"},
                             {"name": "합친 λ(h)", "kind": "line", "x": rl(GRID, 2), "y": rl(lamc, 5), "group": "data"}],
                  "note": f"쌍 {int((~Oc['isph']).sum())}·사진 {int(Oc['isph'].sum())}·21시즌, κ={kapc:g}, 붓스트랩 B={BOOT}"})
    sl = []
    for s in CLOCK:
        r_ = seas_res[s]
        pp = np.interp(gs5, r_["grid"], r_["P"])
        sl.append({"name": s, "kind": "line", "x": rl(gs5, 2), "y": rl(pp, 4), "group": "season"})
    plots.append({"id": "G1b", "part": "G1", "title": "합친 남은 침수 P(h) = ∫_h^H λ + 띠 + 시즌별 자체 곡선",
                  "x": {"label": "시작부터 시간 h", "log": False, "reverse": False}, "y": {"label": "P (log10 최종 ÷ 지금 %)", "log": False},
                  "series": sl + [{"name": "P 10~90% 띠", "kind": "band", "x": rl(GRID, 2), "y": rl(Pcc, 5), "lo": rl(P_lo, 5), "hi": rl(P_hi, 5), "group": "data"},
                                  {"name": "합친 P(h)", "kind": "line", "x": rl(GRID, 2), "y": rl(Pcc, 5), "group": "data"}],
                  "note": "시즌별 선은 시즌마다 따로 맞춘 곡선 (자체 κ·무게 아님: 시즌 안 5겹 교차검증)"})
    plots.append({"id": "G1c", "part": "G1", "title": "같은 점수의 지금 % (최종 3%인 점수) = 3·10^(−P(h))",
                  "x": {"label": "시작부터 시간 h", "log": False, "reverse": False}, "y": {"label": "지금 % (최종 3%)", "log": True},
                  "series": [{"name": "띠 10~90%", "kind": "band", "x": rl(GRID, 2), "y": rl(3 * 10 ** (-Pcc), 4), "lo": rl(3 * 10 ** (-P_hi), 4), "hi": rl(3 * 10 ** (-P_lo), 4), "group": "data"},
                             {"name": "합친 곡선", "kind": "line", "x": rl(GRID, 2), "y": rl(3 * 10 ** (-Pcc), 4), "group": "data"}],
                  "note": "P가 크면 지금 %가 작다 (최종이 3%인 점수가 시작에는 0.4% 안팎)"})
    ml = (GRID >= 96)
    xr = 120 - GRID[ml]
    plots.append({"id": "G1d", "part": "G1", "title": "마지막 24시간 확대 λ (남은 시간 x)",
                  "x": {"label": "남은 시간 x (시간, 오른쪽이 마감)", "log": False, "reverse": True}, "y": {"label": "침수 속도 λ (log10 / 시간)", "log": False},
                  "series": [{"name": "띠 10~90%", "kind": "band", "x": rl(xr, 2), "y": rl(lamc[ml], 5), "lo": rl(lam_lo[ml], 5), "hi": rl(lam_hi[ml], 5), "group": "data"},
                             {"name": "합친 λ", "kind": "line", "x": rl(xr, 2), "y": rl(lamc[ml], 5), "group": "data"}],
                  "note": "x = 0이 마감 (KST 05시)"})

    # ---------- 2 식 ----------
    sc = 1.0
    OC = Oc
    LG.update(Oc=Oc, sig=sigc)
    c_all = CANDS
    fit_rng = np.random.default_rng(31)
    thetas = [None] * len(c_all)
    full_fit = {}
    chain = {}
    for ci, c in enumerate(c_all):
        # 시작: 한 단계 작은 식의 해
        pi = parent_idx(c_all, c)
        warm = lift(c_all[pi], thetas[pi], c) if pi is not None else None
        st = default_starts(c, fit_rng, warm)
        th, m, cost, W0 = fit_param(c, Oc, st, sigc)
        thetas[ci] = th
        full_fit[ci] = (th, m, cost)
        log(f"fit {c.name} npar {c.npar + 1} cost {cost:.3f} th {np.round(th, 3).tolist()}")
    # 시즌 하나씩 빼기
    jobs = [(ci, s, thetas[ci]) for ci in range(len(c_all)) for s in range(NC)]
    with pool.Pool(args.workers) as pl:
        lres = pl.map(loso_job, jobs, chunksize=4)
    cvL = np.zeros((len(c_all), NC))
    cvN = np.zeros(NC)
    loso_th = {}
    for ci, s, loss, n, th in lres:
        cvL[ci, s] = loss
        cvN[s] = n
        loso_th[(ci, s)] = th
    cvT = cvL.sum(1)
    log("LOSO done")
    ibest = int(np.argmin(cvT))
    se = np.array([math.sqrt(NC) * np.std(cvL[ci] - cvL[ibest], ddof=1) for ci in range(len(c_all))])
    within = [ci for ci in range(len(c_all)) if cvT[ci] - cvT[ibest] <= se[ci] + 1e-12 or ci == ibest]
    # 재현 기준
    mrefl = (GRID >= 6.0 - 1e-9)
    cand_rows = []
    P_form = {}
    lam_form = {}
    for ci, c in enumerate(c_all):
        th, m, cost = full_fit[ci]
        gm = math.exp(np.mean(np.log(m)))
        lam = gm * lam_grid(c, th)
        Ce = cum_edges(lam)
        Pg = Ce[-1] - np.interp(GRID, EDGE, Ce)
        P_form[ci], lam_form[ci] = Pg, lam_at(lam, GRID)
        dP = Pg - Pcc
        mx, rms = maxabs(dP, mrefl)
        mx12 = maxabs(dP, GRID >= 12 - 1e-9)[0]
        mx24 = maxabs(dP, GRID >= 24 - 1e-9)[0]
        inband = (Pg >= P_lo - 1e-12) & (Pg <= P_hi + 1e-12)
        outfrac = float(1 - inband[mrefl].mean())
        imx = np.where(mrefl)[0][int(np.argmax(np.abs(dP[mrefl])))]
        # 어긋난 곳: |ΔP| > 0.01인 시각 구간
        bad = (np.abs(dP) > 0.01) & mrefl
        cand_rows.append({"cand": c.name, "n_par": c.npar + 1, "cv_loss": r4(cvT[ci], 3), "cv_minus_best": r4(cvT[ci] - cvT[ibest], 3), "cv_se": r4(se[ci], 3), "within_1se": ci in within,
                          "max_abs_dP": r4(mx), "max_abs_dP_h12": r4(mx12), "max_abs_dP_h24": r4(mx24), "rms_dP": r4(rms), "worst_h": r4(GRID[imx], 2), "worst_kst": r4((5 + GRID[imx]) % 24, 2), "frac_h_over_0.01": r4(float(bad[mrefl].mean()), 3),
                          "frac_out_of_band": r4(outfrac, 3), "meets": bool(mx <= 0.01 and outfrac == 0.0), "meets_dP_only": bool(mx <= 0.01)})
    meets = [ci for ci in within if cand_rows[ci]["meets"]]
    if meets:
        ichosen = min(meets, key=lambda ci: (c_all[ci].npar, cvT[ci]))
        chosen_how = "재현 기준(|ΔP|≤0.01 + 띠 안)을 채우고 교차검증 1-SE 안인 것 가운데 모수가 가장 적은 식"
    else:
        ichosen = min(range(len(c_all)), key=lambda ci: cand_rows[ci]["max_abs_dP"])
        chosen_how = "재현 기준을 채운 식이 없음 → |ΔP| 최대가 가장 작은 식 (가장 가까운 것)"
    inearest = min(range(len(c_all)), key=lambda ci: cand_rows[ci]["max_abs_dP"])
    # 1-SE 안 식 가운데서 기준에 가장 가까운 것
    inear_se = min(within, key=lambda ci: cand_rows[ci]["max_abs_dP"])
    cc = c_all[ichosen]
    log(f"chosen {cc.name} ({chosen_how}); best-cv {c_all[ibest].name}; nearest {c_all[inearest].name}; nearest within-1se {c_all[inear_se].name}")

    flim, fx = [], {}
    for ci, c in enumerate(c_all):
        pi = parent_idx(c_all, c)
        ex = lift(c_all[pi], fx[pi], c) if pi is not None else None
        fx[ci], rr_ = family_limit(c, thetas[ci], Pcc, 5 + ci, ex)
        flim.append(dict(cand=c.name, **rr_))
    log("family limit done: " + ", ".join(f"{r['cand']} {r['max_abs_dP']}" for r in flim))
    # 자료 받침 (h 구간별 관측 수)
    sup = []
    for a, b in ((0, 6), (6, 12), (12, 24), (24, 48), (48, 96), (96, 114), (114, 120)):
        mp_ = (~Oc["isph"]) & (Oc["ta"] >= a) & (Oc["ta"] < b)
        mq_ = Oc["isph"] & (Oc["ta"] >= a) & (Oc["ta"] < b)
        mw_ = (~Oc["isph"]) & (Oc["tb"] > a) & (Oc["ta"] < b)
        sup.append({"h_from": a, "h_to": b, "photos_with_h_here": int(mq_.sum()), "pairs_starting_here": int(mp_.sum()), "pairs_overlapping_here": int(mw_.sum()),
                    "seasons_with_photos": int(len(set(Oc["si"][mq_].tolist()))), "small": bool(mq_.sum() + mw_.sum() < FL.MINN)})
    # 어긋남 구간 (고른 식): 연속 구간
    def segs(dP):
        bad = (np.abs(dP) > 0.01) & mrefl
        out, a = [], None
        for i, b in enumerate(bad):
            if b and a is None:
                a = i
            if (not b or i == len(bad) - 1) and a is not None:
                e = i if b else i - 1
                out.append({"h_from": r4(GRID[a], 2), "h_to": r4(GRID[e], 2), "kst_from": r4((5 + GRID[a]) % 24, 2), "day": int(GRID[a] // 24) + 1,
                            "max_dP": r4(float(dP[a:e + 1][np.argmax(np.abs(dP[a:e + 1]))]), 4)})
                a = None
        return out
    dP_ch = P_form[ichosen] - Pcc
    misfit = segs(dP_ch)
    # 견줌: F·M3·act (21시즌 기하평균)
    Fcur = np.array([T.F_P(h / HH) for h in GRID])
    gmean = lambda A_: np.exp(np.mean(np.log(np.maximum(A_, 1e-6)), axis=0))      # noqa: E731  시즌 P의 기하평균
    M3cur = gmean(np.array([[T.m3_P(s, h / SI[s]["H"], 3.0) for h in GRID] for s in CLOCK]))
    ACcur = gmean(np.array([FL.m3_curve(s1_act, s, AC.ustar(prof, s, GRID, float(SI[s]["H"])), np.full(len(GRID), 3.0)) for s in CLOCK]))
    comp_rows = []
    for nm, cur in (("F f2pow", Fcur), ("M3 기본 (21시즌 기하평균)", M3cur), ("act (21시즌 기하평균)", ACcur), (f"고른 식 {cc.name}", P_form[ichosen])):
        d = cur - Pcc
        mx, rms = maxabs(d, mrefl)
        inb = (cur >= P_lo) & (cur <= P_hi)
        iw = np.where(mrefl)[0][int(np.argmax(np.abs(d[mrefl])))]
        comp_rows.append({"curve": nm, "max_abs_dP": r4(mx), "max_abs_dP_h12": r4(maxabs(d, GRID >= 12 - 1e-9)[0]), "max_abs_dP_h24": r4(maxabs(d, GRID >= 24 - 1e-9)[0]), "rms_dP": r4(rms), "worst_h": r4(GRID[iw], 2), "worst_dP": r4(d[iw], 4), "frac_out_of_band": r4(float(1 - inb[mrefl].mean()), 3),
                          "frac_h_over_0.01": r4(float((np.abs(d[mrefl]) > 0.01).mean()), 3)})
    log("compare done")

    # 고른 식 모수 + 붓스트랩
    LG.update(ci=ichosen, th0=thetas[ichosen])
    with pool.Pool(args.workers) as pl:
        bpar = pl.map(boot_param_job, [(9000 + b,) for b in range(BOOT)], chunksize=4)
    BT = np.array([b[0] for b in bpar])
    BGM = np.array([b[1] for b in bpar])
    BLF = np.array([b[2] for b in bpar])
    BPF = np.array([b[3] for b in bpar])
    log("boot param done")
    thc = thetas[ichosen]
    gmc = math.exp(np.mean(np.log(full_fit[ichosen][1])))
    names = ["alpha"] + (["B"] if cc.base == "b" else []) + sum(([f"a{k}", f"b{k}"] for k in range(1, cc.K + 1)), []) + (["R", "tau"] if cc.end else []) + (["gamma"] if cc.gam else [])
    par_rows = [{"par": "A", "value": r4(gmc, 5), "lo10": r4(np.percentile(BGM, 10), 5), "hi90": r4(np.percentile(BGM, 90), 5)}]
    for k, nm in enumerate(names):
        par_rows.append({"par": nm, "value": r4(thc[k], 4), "lo10": r4(np.percentile(BT[:, k], 10), 4), "hi90": r4(np.percentile(BT[:, k], 90), 4),
                         "at_bound": bool(thc[k] <= cc.lo[k] + 1e-4 or thc[k] >= cc.hi[k] - 1e-4), "bounds": [float(cc.lo[k]), float(cc.hi[k])]})
    # 파 진폭·위상
    waves = []
    for k in range(1, cc.K + 1):
        a, b = thc[cc.i_four + 2 * (k - 1)], thc[cc.i_four + 2 * (k - 1) + 1]
        waves.append({"k": k, "amp": r4(math.hypot(a, b), 4), "peak_kst": r4((math.atan2(b, a) * 24 / (2 * math.pi * k)) % (24 / k), 2)})
    # 식 글
    form_txt = f"λ(h) = {gmc:.4g} · [h^(−{thc[0]:.3f})" + (f" + {thc[1]:.3g}" if cc.base == "b" else "") + "]"
    if cc.K:
        terms = " + ".join(f"{thc[cc.i_four + 2 * (k - 1)]:.3f}·cos(2π{k}c/24) + {thc[cc.i_four + 2 * (k - 1) + 1]:.3f}·sin(2π{k}c/24)" for k in range(1, cc.K + 1))
        form_txt += f" · exp(" + ("(1 + {:.3f}(d−3))·".format(thc[cc.i_gam]) + "[" if cc.gam else "") + terms + ("]" if cc.gam else "") + ")"
    if cc.end:
        form_txt += f" · (1 + {thc[cc.i_end]:.3f}·exp(−x/{thc[cc.i_end + 1]:.3f}))"
    form_txt += "  (c = KST 시각, x = 남은 시간 H−h, d = 일차, λ 단위 log10/시간)"

    # 식 대 경험 표
    Pf, lf = P_form[ichosen], lam_form[ichosen]
    # 띠(식): 붓스트랩 P
    Pf_lo, Pf_hi = FL.band(BPF)
    lf_lo, lf_hi = FL.band(BLF)
    tbl_form = [{"h": float(GRID[i]), "kst": r4((5 + GRID[i]) % 24, 2), "P_emp": r4(Pcc[i], 5), "P_form": r4(Pf[i], 5), "dP": r4(Pf[i] - Pcc[i], 5), "P_form_lo": r4(Pf_lo[i], 5), "P_form_hi": r4(Pf_hi[i], 5),
                 "lam_emp": r4(lamc[i], 5), "lam_form": r4(lf[i], 5), "dP_F": r4(Fcur[i] - Pcc[i], 5), "dP_M3": r4(M3cur[i] - Pcc[i], 5), "dP_act": r4(ACcur[i] - Pcc[i], 5)} for i in range(len(GRID))]

    # ---------- 점검 표: 사진 중앙 대 곡선들 ----------
    cvp = []
    for a, b in ((6, 9), (12, 15), (21, 24), (33, 36), (45, 48), (69, 72), (93, 96), (105, 108), (114, 117), (117, 119)):
        mq_ = Oc["isph"] & (Oc["ta"] >= a) & (Oc["ta"] < b)
        if mq_.sum() == 0:
            cvp.append({"h_from": a, "h_to": b, "n": 0})
            continue
        hm = float(np.median(Oc["ta"][mq_]))
        at = lambda arr: r4(float(np.interp(hm, GRID, arr)), 4)  # noqa: E731
        cvp.append({"h_from": a, "h_to": b, "n": int(mq_.sum()), "seasons": int(len(set(Oc["si"][mq_].tolist()))), "median_photo_h": r4(hm, 2), "photo_P_median": r4(float(np.median(Oc["y"][mq_])), 4),
                    "emp_P": at(Pcc), "F": at(Fcur), "M3": at(M3cur), "act": at(ACcur), "chosen": at(Pf), "small": bool(mq_.sum() < FL.MINN)})
    # (나) 판: pct ≥ 0.1, 끝 화면 점 곡선 pf_pts ≤ 30 (P_pts)
    phb = [q_ for q_ in T.photos(args.obs_dir) if q_["kind"] in ("A", "B") and q_["season"] in CIDX and q_["pct"] >= 0.1 and q_["pf_pts"] is not None and q_["pf_pts"] <= 30 and q_["h"] >= 0]
    hb = np.array([q_["h"] for q_ in phb])
    Pb = np.array([q_["P_pts"] for q_ in phb])
    pb = np.array([q_["pct"] for q_ in phb])
    BANDS = ((0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 40.0))
    for row in cvp:
        if row["n"] == 0:
            continue
        mb_ = (hb >= row["h_from"]) & (hb < row["h_to"])
        row["B_n"] = int(mb_.sum())
        row["B_photo_P_median"] = r4(float(np.median(Pb[mb_])), 4) if mb_.sum() else None
        row["B_by_pct_band"] = [{"pct": f"{a}-{b}", "n": int((mb_ & (pb >= a) & (pb < b)).sum()), "P_median": r4(float(np.median(Pb[mb_ & (pb >= a) & (pb < b)])), 4) if (mb_ & (pb >= a) & (pb < b)).sum() else None} for a, b in BANDS]
        lo_, hi_ = sorted([x_ for x_ in (row["photo_P_median"], row["B_photo_P_median"]) if x_ is not None])[0], sorted([x_ for x_ in (row["photo_P_median"], row["B_photo_P_median"]) if x_ is not None])[-1]
        row["emp_between_A_and_B"] = bool(lo_ - 1e-9 <= row["emp_P"] <= hi_ + 1e-9)
        row["emp_minus_A"] = r4(row["emp_P"] - row["photo_P_median"], 4)
        row["emp_minus_B"] = r4(row["emp_P"] - row["B_photo_P_median"], 4) if row["B_photo_P_median"] is not None else None
    # ---------- 마감 항 확인 ----------
    endnote = None
    if cc.end:
        R_, tau_ = thc[cc.i_end], thc[cc.i_end + 1]
        ml_ = Oc["isph"] & (Oc["ta"] >= 119.0)
        base_ = lam_grid(cc, np.concatenate([thc[:cc.i_end], [0.0, 1.0], thc[cc.i_end + 2:]]))      # 마감 항 R = 0
        lam_ = lam_grid(cc, thc)
        Pend = float(np.sum((lam_ - base_) * GD * (G >= 119.0))) * gmc                              # 마지막 1시간 마감 항이 더한 침수 (A 환산)
        Ptot = float(np.sum(lam_ * GD * (G >= 119.0))) * gmc
        xs = [6, 4, 3, 2, 1, 0.5]
        comp = []
        for x_ in xs:
            k_ = int(round((HH - x_) / GD)) - 1
            fsum = 0.0
            for kk in range(1, cc.K + 1):
                fsum += thc[cc.i_four + 2 * (kk - 1)] * _COS[kk][k_] + thc[cc.i_four + 2 * (kk - 1) + 1] * _SIN[kk][k_]
            sc_ = (1.0 + (thc[cc.i_gam] if cc.gam else 0.0) * (DAY[k_] - 3))
            comp.append({"x": x_, "kst": r4((5 + HH - x_) % 24, 2), "trend": r4(float(G[k_] ** -thc[0] + (thc[1] if cc.base == "b" else 0.0)), 6), "wave_factor_d5": r4(math.exp(sc_ * fsum), 3), "wave_factor_if_gamma0": r4(math.exp(fsum), 3),
                         "end_factor": r4(1.0 + R_ * math.exp(-x_ / tau_), 3)})
        endnote = {"R": r4(R_, 3), "tau": r4(tau_, 3), "last_hour_photos": {"n": int(ml_.sum()), "median_P_obs": r4(float(np.median(Oc["y"][ml_])), 4) if ml_.sum() else None},
                   "last_hour_P_formula_total": r4(Ptot, 5), "last_hour_P_added_by_end_term": r4(Pend, 5), "share_of_last_hour_P_from_end_term": r4(Pend / Ptot, 3), "components_by_x": comp,
                   "lam_empirical_at_x": {str(x_): r4(float(np.interp(HH - x_, GRID, lamc)), 5) for x_ in xs}, "lam_formula_at_x": {str(x_): r4(float(np.interp(HH - x_, GRID, lf)), 5) for x_ in xs}}
    # ---------- 시즌 차이 ----------
    LG.update(ci=ichosen, Oc=Oc, sig=sigc)
    modes = ["m", "alpha"] + (["wave"] if cc.K else [])
    vjobs = [(s, md, loso_th[(ichosen, s)], 77 + s) for md in modes for s in range(NC)]
    with pool.Pool(args.workers) as pl:
        vres = pl.map(vary_loso_job, vjobs, chunksize=2)
    vcv = {md: np.zeros(NC) for md in modes}
    for s, md, loss in vres:
        vcv[md][s] = loss
    log("vary CV done")
    var_rows = []
    season_dP = {}
    for md in modes:
        if md == "m":
            th_v, ex_v = thc, None
            Ov = dict(Oc)
        else:
            Ov = dict(Oc)
            th_v, ex_v, cost_v = fit_vary(cc, Oc, thc, sigc, md, rounds=2)
        # 시즌별 곡선과의 |ΔP|
        dd = []
        sm_rows = []
        # 시즌 곡선: m_s 다시 (닫힌 꼴, 마지막 W 근사 = Huber 무게 없이 W0)
        for s in range(NC):
            ix = np.where(Oc["si"] == s)[0]
            lam_s = lam_grid(cc, th_v, dal=(ex_v[s] if md == "alpha" else 0.0), wsc=(ex_v[s] if md == "wave" else 1.0))
            f = interval_f(lam_s, Oc["ta"][ix], Oc["tb"][ix])
            isph_ = Oc["isph"][ix]
            w0 = W0of(isph_, Oc["wp"][ix], sigc)
            ps_ = np.ones(len(ix))
            for _ in range(4):
                m_ = mfit(f, Oc["y"][ix], w0 * ps_, np.zeros(len(ix), int), 1)[0]
                ps_ = psi(Oc["y"][ix] - m_ * f, w0)
            Ce = cum_edges(m_ * lam_s)
            gsx = seas_res[CLOCK[s]]["grid"]
            Ps = Ce[-1] - np.interp(gsx, EDGE, Ce)
            Pe = seas_res[CLOCK[s]]["P"]
            mk6 = gsx >= 6.0
            dsx = Ps - Pe
            sm_rows.append({"season": CLOCK[s], "m": r4(m_, 4), "extra": (r4(ex_v[s], 4) if ex_v is not None else None), "max_abs_dP_vs_own": r4(float(np.max(np.abs(dsx[mk6]))), 4),
                            "rms_dP_vs_own": r4(float(np.sqrt(np.mean(dsx[mk6] ** 2))), 4)})
            dd.append(sm_rows[-1]["max_abs_dP_vs_own"])
        label = {"m": "m_s만", "alpha": "m_s + α_s", "wave": "m_s + 물결 크기 w_s"}[md]
        var_rows.append({"variant": label, "extra_par_per_season": 0 if md == "m" else 1, "cv_loss_halfsplit": r4(vcv[md].sum(), 3), "cv_minus_m_only": r4(vcv[md].sum() - vcv["m"].sum(), 3),
                         "cv_se": r4(math.sqrt(NC) * np.std(vcv[md] - vcv["m"], ddof=1), 3) if md != "m" else 0.0,
                         "median_max_dP_vs_own_season_curve": r4(float(np.median(dd)), 4), "max_of_max_dP": r4(float(np.max(dd)), 4), "seasons": sm_rows})
    log("vary fits done")

    # ---------- 그림 2 ----------
    # h<6은 자료가 없고 식의 h^(−α)가 h→0에서 커져 축을 망친다 → 재현 기준 구간 h 6~H만 그린다
    g6 = GRID >= 6.0
    G6 = GRID[g6]
    plots.append({"id": "G2a", "part": "G2", "title": "경험 P 대 고른 식 대 F·M3·act (h 6~120 — 재현 기준 구간)",
                  "x": {"label": "시작부터 시간 h", "log": False, "reverse": False}, "y": {"label": "P (log10)", "log": False},
                  "series": [{"name": "경험 P 10~90% 띠", "kind": "band", "x": rl(G6, 2), "y": rl((Pcc)[g6], 5), "lo": rl((P_lo)[g6], 5), "hi": rl((P_hi)[g6], 5), "group": "data"},
                             {"name": "합친 경험 P", "kind": "line", "x": rl(G6, 2), "y": rl((Pcc)[g6], 5), "group": "data"},
                             {"name": f"고른 식 {cc.name}", "kind": "line", "x": rl(G6, 2), "y": rl((Pf)[g6], 5), "group": "data"},
                             {"name": "F f2pow", "kind": "line", "x": rl(G6, 2), "y": rl((Fcur)[g6], 5), "group": "F"},
                             {"name": "M3 기본 (21시즌 기하평균)", "kind": "line", "x": rl(G6, 2), "y": rl((M3cur)[g6], 5), "group": "M3"},
                             {"name": "act (21시즌 기하평균)", "kind": "line", "x": rl(G6, 2), "y": rl((ACcur)[g6], 5), "group": "act"}],
                  "note": f"고른 식 {cc.name}: 모수 {cc.npar + 1}개 (+ 시즌 배수 {NC}), 쌍 {int((~Oc['isph']).sum())}·사진 {int(Oc['isph'].sum())}"})
    plots.append({"id": "G2b", "part": "G2", "title": "경험 곡선과의 차 ΔP(h) = 곡선 − 경험 (±0.01 선)",
                  "x": {"label": "시작부터 시간 h", "log": False, "reverse": False}, "y": {"label": "ΔP (log10)", "log": False},
                  "series": [{"name": "+0.01", "kind": "line", "x": [6, 120], "y": [0.01, 0.01], "group": "data"}, {"name": "−0.01", "kind": "line", "x": [6, 120], "y": [-0.01, -0.01], "group": "data"},
                             {"name": "경험 10~90% 띠 (경험 기준)", "kind": "band", "x": rl(G6, 2), "y": [0.0] * len(G6), "lo": rl((P_lo - Pcc)[g6], 5), "hi": rl((P_hi - Pcc)[g6], 5), "group": "data"},
                             {"name": f"고른 식 {cc.name}", "kind": "line", "x": rl(G6, 2), "y": rl((Pf - Pcc)[g6], 5), "group": "data"},
                             {"name": "F", "kind": "line", "x": rl(G6, 2), "y": rl((Fcur - Pcc)[g6], 5), "group": "F"},
                             {"name": "M3 기본", "kind": "line", "x": rl(G6, 2), "y": rl((M3cur - Pcc)[g6], 5), "group": "M3"},
                             {"name": "act", "kind": "line", "x": rl(G6, 2), "y": rl((ACcur - Pcc)[g6], 5), "group": "act"}],
                  "note": "재현 기준 |ΔP| ≤ 0.01 (h 6~120)"})
    plots.append({"id": "G2c", "part": "G2", "title": "경험 λ(h) 대 고른 식 λ(h)",
                  "x": {"label": "시작부터 시간 h", "log": False, "reverse": False}, "y": {"label": "λ (log10 / 시간)", "log": False},
                  "series": [{"name": "경험 λ 10~90% 띠", "kind": "band", "x": rl(G6, 2), "y": rl((lamc)[g6], 5), "lo": rl((lam_lo)[g6], 5), "hi": rl((lam_hi)[g6], 5), "group": "data"},
                             {"name": "합친 경험 λ", "kind": "line", "x": rl(G6, 2), "y": rl((lamc)[g6], 5), "group": "data"},
                             {"name": f"고른 식 {cc.name}", "kind": "line", "x": rl(G6, 2), "y": rl((lf)[g6], 5), "group": "data"}],
                  "note": "식은 P(적분)를 맞췄고 λ는 그 도함수"})

    # ---------- notes ----------
    notes1 = [f"합친 곡선: 21시즌 쌍 {int((~Oc['isph']).sum())}·사진 {int(Oc['isph'].sum())}, κ={kapc:g} (5겹 교차검증 최소, 격자 {KAPPA_GRID_C[0]:g}~{KAPPA_GRID_C[-1]:g}), σ_ph={sigc:.4f}, φ_pair={sigc.phi:.3f} (쌍 무게 1/(φ var)), σ_ph(일차) {np.round(sigc.sd, 4).tolist()}, P(h=6)={desc['P_at_h6']:.3f} (h<6은 자료 없음, 바깥 늘림 P(0)={Pcc[0]:.3f}은 extra) → 최종 3% 점수의 h=6 % = {3 * 10 ** (-desc['P_at_h6']):.3f}.",
              f"일차 침수 합 {desc['day_sum']} (1일차는 h6~24만, 0~24 늘림은 {desc['day1_sum_h0_24_extrapolated_extra']}); 마지막 날 합 {desc['last_day_sum']}, H−6~H 몫 {desc['share_last6_of_last24']}; 마지막 1/3/6시간 평균 속도 {desc['last_mean_speed']}.",
              f"마지막 날 최저 속도 {desc['last_day_min']}; 3~4일차 같은 시각의 1.25배를 (마지막 2시간 빼고) 끝까지 넘는 시각 {desc['crowd_start_ratio125_until_x2']}; 마지막 k시간/3~4일차 같은 시각 배수 {desc['last_over_days3_4_same_clock']}.",
              f"하루 물결 (2~4일차 같은 시각 평균): 가장 빠른 KST {desc['wave_days2_4']['fast_kst']}시 {desc['wave_days2_4']['fast']}, 가장 느린 KST {desc['wave_days2_4']['slow_kst']}시 {desc['wave_days2_4']['slow']}, 비 {desc['wave_days2_4']['ratio_fast_slow']}.",
              f"첫날 속도 (log10/h): {desc['first_mean_speed']}; 2~4일차 같은 시각 대비 {desc['first_over_days2_4_same_clock']}; 첫날 합 h6~24 {desc['first_day_sum_from_h6']} (h0~6은 자료 없는 바깥 늘림 {desc['first_day_sum_0_6_EXTRAPOLATED']}).",
              "칸 요약 대조: " + "; ".join(f"{c_['item']} 칸 {c_['bin_summary']} / 곡선 {c_['curve']}" for c_ in chk[:7])]
    lh = desc["last_mean_speed"][1]
    if endnote:
        e_ = endnote
        cs_ = "; ".join(f"x={c_['x']}(KST {c_['kst']}) 물결 {c_['wave_factor_d5']}(γ=0이면 {c_['wave_factor_if_gamma0']}) 마감 {c_['end_factor']}" for c_ in e_["components_by_x"])
        end_line = (f"마감 항 확인: R={e_['R']} τ={e_['tau']}시간. 마지막 1시간 사진 n={e_['last_hour_photos']['n']} 중앙 P_obs {e_['last_hour_photos']['median_P_obs']}; 식의 마지막 1시간 P {e_['last_hour_P_formula_total']} 중 마감 항이 더한 몫 {e_['share_of_last_hour_P_from_end_term']}. "
                    f"마감 몰림(x=6~0.5)을 담는 것: {cs_}. λ 경험 {e_['lam_empirical_at_x']} 식 {e_['lam_formula_at_x']}.")
    else:
        end_line = "고른 식에 마감 항 없음."
    notes2 = [end_line, f"마지막 1시간 λ 평균 {lh} (칸 요약 A1 약 0.0018 = H−1~H 몫 0.011 × 마지막 날 합 0.163) → " + ("칸 요약보다 높음: 마지막 격자 λ=0은 스플라인 끝 계수 0의 경계 효과로 보이며 평균은 그 위쪽 계수가 채움" if lh > 0.0022 else "칸 요약과 비슷") + f"; h=120 격자 λ={r4(lamc[-1], 5)}, h=119 λ={r4(float(np.interp(119.0, GRID, lamc)), 5)}.",
              f"고른 식: {cc.name} ({chosen_how}). 교차검증 최소 {c_all[ibest].name}, |ΔP| 최소 {c_all[inearest].name} (max {cand_rows[inearest]['max_abs_dP']}), 1-SE 안 가운데 |ΔP| 최소 {c_all[inear_se].name} (max {cand_rows[inear_se]['max_abs_dP']}).",
              f"재현 기준을 채운 식 {sum(r['meets'] for r in cand_rows)}개 / |ΔP| ≤ 0.01만 채운 식 {sum(r['meets_dP_only'] for r in cand_rows)}개 (20개 중); 1-SE 안 식 {len(within)}개.",
              f"고른 식 |ΔP| 최대 {cand_rows[ichosen]['max_abs_dP']} (h={cand_rows[ichosen]['worst_h']}, KST {cand_rows[ichosen]['worst_kst']}시), RMS {cand_rows[ichosen]['rms_dP']}, 띠 밖 시간 비율 {cand_rows[ichosen]['frac_out_of_band']}.",
              "견줌 (경험 곡선 대비): " + "; ".join(f"{r['curve']} max(h6~) {r['max_abs_dP']} (h12~ {r['max_abs_dP_h12']}, h24~ {r['max_abs_dP_h24']}) RMS {r['rms_dP']} 띠 밖 {r['frac_out_of_band']}" for r in comp_rows),
              "식 꼴 한계(더 본 것, 경험 P 곡선에 직접 맞춤) |ΔP| 최대: " + ", ".join(f"{r['cand']} {r['max_abs_dP']}" for r in flim),
              "식: " + form_txt,
              "시즌 차이: " + "; ".join(f"{v['variant']} 반반 교차검증 {v['cv_loss_halfsplit']} (m만과 차 {v['cv_minus_m_only']}±{v['cv_se']}), 시즌 곡선과 |ΔP| 중앙 {v['median_max_dP_vs_own_season_curve']}" for v in var_rows)]
    for r in misfit[:12]:
        notes2.append(f"고른 식 어긋남 구간: h {r['h_from']}~{r['h_to']} (KST {r['kst_from']}시 시작, {r['day']}일차) 최대 ΔP {r['max_dP']}")

    head = None
    try:
        import subprocess
        head = subprocess.check_output(["git", "-C", T.C.ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        pass
    J = {"header": {"about": "WP12b 침수 실제 곡선 (칸 없이 스플라인) + 그 꼴을 재현하는 식 — 서술, 예측·채점 없음, 24시즌 정답을 본 서술", "data_ver": "D1",
                    "date": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d"), "code_head": head, "secs": int(time.time() - t00),
                    "pairs": npair, "photos_main": O["n"] - npair, "boot_B": BOOT,
                    "rules": {"spline": "3차 B-스플라인, 매듭 1시간 간격 [0, H_true] (S28 130.5), 계수 ≥ 0 (NNLS, 정규방정식 Cholesky 꼴), 2차 차분 벌점 κ‖D2β‖²",
                              "obs": "사진 = A·B 한섭·미상 use, 지금 % ≥ 0.3, 최종 % 0.5~15 (칸 요약과 같은 거르기, C 뺌) P_obs = Σβ∫_h^H; 기록 쌍 = wp12_tables.pairs ΔL = Σβ∫_t1^t2",
                              "weights": f"분산의 역수: 쌍 1/(φ_pair·var), 사진 1/σ_ph² (σ_ph = 1.4826·MAD 사진 잔차, φ_pair = (1.4826·MAD 쌍 표준화 잔차)², 3번 되풀이), 쌍·사진 모두 Huber c = 1.345 (IRLS). 합친 곡선 σ_ph 전체 {float(sigc):.4f}, 일차별(1~5) {np.round(sigc.sd, 4).tolist()}, φ_pair={sigc.phi:.3f} (분산 함수 추정: 일차마다 사진 잔차 MAD, Carroll·Ruppert 1988; Huber는 σ_ph(d) 기준); 시즌 곡선은 시즌마다 (per_season 표)",
                              "kappa": f"관측을 무작위 5겹 교차검증 최소 (가중·Huber 손실); 시즌별 격자 1e-2…1e6 (10배), 합친 것 반 자리 간격 1e-2…1e6; 한 시즌에서 격자 끝에 닿으면 표시",
                              "combined": "05시 시작 120시간 21시즌 (S19·S28·S38 뺌), 한 λ(h) + 시즌 배수 m_s (기하평균 1, 번갈아 풀기); 붓스트랩 = 시즌 복원 추출 (κ·σ 고정)",
                              "formula_fit": "scipy least_squares(trf, 경계), A·m_s 소거(시즌 배수 닫힌 꼴, 기하평균을 A로), 사진 Huber IRLS 2번, 전체 적합 여러 시작점 + 작은 식 해에서 이어받기",
                              "cv": "시즌 하나씩 빼기: 나머지 20시즌으로 θ, 뺀 시즌은 m_s만 다시 맞춘 가중(Huber) 손실. 1-SE = 최소와 짝지은 시즌별 손실 차 sd × √21",
                              "repro": "|ΔP| ≤ 0.01 (h 6~120, 0.25시간 격자 457점) 그리고 경험 10~90% 띠 안 (밖 시간 비율 0)",
                              "season_variants": "뺀 시즌 안을 반반 나눠 한쪽에서 m_s(+시즌 몫)를 맞추고 다른 쪽에서 손실, 서로 바꿔 합 — 시즌 몫이 자기 시즌 관측을 보지 못하게 한 짝지은 비교",
                              "small": f"쌍·사진 {FL.MINN} 미만이면 small"}},
         "G1": {"tables": {"per_season": per_tab, "data_support": sup, "combined_grid_0.25h": tbl_comb, "season_multiplier": mt, "check_vs_bins": chk,
                           "kappa_cv_combined": [{"kappa": k, "cv_loss": r4(e, 3)} for k, e in zip(KAPPA_GRID_C, errc)]},
                "notes": notes1, "describe": desc,
                "extra": {"per_season_curves_0.5h": {s: {"h": rl(np.arange(0, seas_res[s]["Ht"] + 1e-9, 0.5), 1), "lam": rl(np.interp(np.arange(0, seas_res[s]["Ht"] + 1e-9, 0.5), seas_res[s]["grid"], seas_res[s]["lam"]), 5),
                                                      "P": rl(np.interp(np.arange(0, seas_res[s]["Ht"] + 1e-9, 0.5), seas_res[s]["grid"], seas_res[s]["P"]), 5)} for s in CLOCK},
                          "kst_note": "h = 시작부터 시간, 05시 시작 시즌이라 KST 시각 = (5 + h) mod 24. S19·S28·S38은 시즌별 요약(per_season)에만, 합친 곡선·그림에서 뺌"}},
         "G2": {"tables": {"check_vs_photos": cvp, "candidates": cand_rows, "compare_to_empirical": comp_rows, "chosen_params": par_rows, "chosen_waves": waves, "chosen_vs_empirical_grid_0.25h": tbl_form, "end_term_check": [endnote] if endnote else [],
                           "misfit_segments_chosen": misfit, "season_variants": var_rows},
                "notes": notes2,
                "extra": {"formula_text": form_txt, "chosen": cc.name, "chosen_how": chosen_how, "best_cv": c_all[ibest].name, "nearest": c_all[inearest].name, "nearest_within_1se": c_all[inear_se].name,
                          "더 본 것": "후보 식은 문서의 20개만. 더 본 것: 식 꼴 한계 — 같은 20개 식을 자료가 아니라 합친 경험 P(h) 곡선(h 6~120)에 직접 최소제곱으로 맞춘 가장 가까운 값 (고르기에 안 씀).",
                          "family_limit_vs_empirical_curve": flim}},
         "plots": plots}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=1)
    log(f"wrote {args.out} {os.path.getsize(args.out) / 1e6:.2f}MB")


if __name__ == "__main__":
    main()
