"""WP13a 침수 꼴 — 계층 가법 모형 (HGAM) 24시즌 적합 + 맞힐 시즌(S21~S42)마다 앞 시즌만 적합 (2026-11-04).

  python3 -I tools/solo_raid/lab7/wp13_flood_shape.py --obs-dir W/obs7/full --out data/solo-raid/analysis/r7/wp13/flood_hgam_D1.json [--procs 2] [--targets 21,22]
  (W = 관측 표 폴더의 부모. 2코어 Pool. 걸린 시간은 .json header.secs · .md 첫 줄)

계획: docs/solo-raid-method.md 「WP13 5차 연구 — 꼴 — 무엇을 박나」. 출력 형식: flood_hgam_D1.json (shape_format.md).
공용 코드(wp12_*.py)는 읽기만 한다. 관측·무게·Huber·σ 되풀이는 WP12 G1 그대로 (wp12_flood_curve.build_obs·W0of·psi·rho_loss·sig_of·Sg).
모형 (계산 전 조율 세션이 정함):
  시즌 j: 침수 속도 λ_j(h) = m_j · exp(η_j(h)) (log10/시간), h = 시작부터 시간, c = KST 시각, d = min(일차, 5), x = Ht − h
  η = f_T(ln max(h, 0.5)) + f_G(c) + f_d(c) + f_X(x)
  f_T 3차 B-스플라인(기저 12, 균일 매듭, z=ln h' ∈ [ln 0.5, ln 131]) · f_G 24시간 주기 3차 B-스플라인(기저 12, 평균 0) ·
  f_d 같은 주기 기저 일차마다(평균 0, 벌점 κ_D‖β_d‖² → f_G로 당김) · f_X x∈[0,12] 3차 B-스플라인(매듭 1시간, x=12에서 값·기울기 0, 밖 0).
  ln m_j 시즌마다 자유 (맞춘 뒤 학습 시즌 평균 0으로 옮겨 f_T 절편에 더함).
  적분 0.05시간 칸 중점 누적합 (구간 끝은 누적값 선형 보간). 맞추기 = 가중 Huber(IRLS) + 벌점, Levenberg-Marquardt 직접 구현 (해석적 야코비안; scipy least_squares(lm)은 MINPACK QR이 한 번 1초라 느려 바꿈).
  벌점 κ 넷(T·G·D·X) = 관측 무작위 5겹 교차검증, 좌표마다 격자 10^{-2..5}(반 자리), 2바퀴.
재현 점검(24시즌 전체 → 05시 시작 120시간 달력 P_full을 WP12 합친 곡선과 견줌) → 못 미치면 ① 매듭 늘림(16·16·0.5시간) → ② 1-SE 안 최소 κ.
판은 FULL 한 번 정하고 맞힐 시즌 적합에 똑같이 쓴다 (규칙 같게, κ 값은 시즌마다 교차검증).
공개 규칙: 글 번호·원문·닉네임·원글 주소·사람 열쇠·사진 열쇠를 쓰지 않는다 (시각·%·점수·개수만).

걸린 시간: 4370초 (73분, 2프로세스; FULL 교차검증 605초 + 맞힐 시즌 22개).
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
from scipy.linalg import null_space
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as CM  # noqa: E402
import wp12_tables as T  # noqa: E402
import wp12_flood as FL  # noqa: E402
import wp12_flood_curve as FC  # noqa: E402

SEAS = T.SEASONS
SIDX = {s: i for i, s in enumerate(SEAS)}
SI = T.season_info()
HTRUE = {s: FL.htrue(s) for s in SEAS}
DT = 0.05
KGRID = [10.0 ** (k / 2) for k in range(-4, 11)]          # 1e-2 … 1e5, 반 자리 간격 (15개)
KNAMES = ["T", "G", "D", "X"]
NF = 5
SEED = 20261104
ZLO, ZHI = math.log(0.5), math.log(131.0)
XMAX = 12.0
RIDGE_M = 1e-2
W_CON = 1e3
OUT_GRID = 0.25
WP12_JSON = os.path.join(CM.ROOT, "data/solo-raid/analysis/r7/wp12/flood_curve_D1.json")


def pbasis(c, nb):
    """24시간 주기 3차 B-스플라인 (매듭 24/nb 간격), c → (n, nb)."""
    h = 24.0 / nb
    t = (np.arange(nb + 7) - 3) * h
    B = BSpline.design_matrix(np.mod(c, 24.0), t, 3).toarray()      # (n, nb + 3)
    out = B[:, :nb].copy()
    for j in range(nb, nb + 3):
        out[:, j - nb] += B[:, j]
    return out


def tbasis(h, nT):
    z = np.clip(np.log(np.maximum(h, 0.5)), ZLO, ZHI)
    kn = np.linspace(ZLO, ZHI, nT - 2)
    t = np.concatenate([[ZLO] * 3, kn, [ZHI] * 3])
    return BSpline.design_matrix(z, t, 3).toarray()


class Cfg:
    """매듭 설정: k12 (기저 12·12, X 매듭 1시간) · k16 (16·16, 0.5시간)."""

    def __init__(self, name, nT, nG, dk):
        self.name, self.nT, self.nG, self.dk = name, nT, nG, dk
        self.nk = int(round(XMAX / dk))
        self.nX = self.nk + 3 - 2
        self.g = nG - 1
        self.pe = nT + self.g + 5 * self.g + self.nX
        self.iT, self.iG = 0, nT
        self.iD = nT + self.g
        self.iX = self.iD + 5 * self.g
        cg = (np.arange(240) + 0.5) * 0.1
        a = pbasis(cg, nG).mean(axis=0)
        self.Z = null_space(a[None, :])                      # (nG, g) 정규직교, c 격자 평균 0 부분공간
        D2 = np.zeros((nG, nG))
        for i in range(nG):
            D2[i, i] = -2.0
            D2[i, (i - 1) % nG] += 1.0
            D2[i, (i + 1) % nG] += 1.0
        self.PG = D2 @ self.Z
        self.PT = np.diff(np.eye(nT), 2, axis=0)
        self.PX = np.diff(np.eye(self.nX + 2), 2, axis=0)[:, :self.nX]     # 끝 기저 둘 = 0

    def xbasis(self, x):
        kn = np.linspace(0.0, XMAX, self.nk + 1)
        t = np.concatenate([[0.0] * 3, kn, [XMAX] * 3])
        B = BSpline.design_matrix(np.clip(x, 0.0, XMAX), t, 3).toarray()[:, :self.nX]
        B[x > XMAX] = 0.0
        return B

    def pm(self, kap, J):
        """벌점 행렬 PM (행 × (pe + J)): 벌점 줄 + Σ ln m = 0 제약 + ln m 능선."""
        kT, kG, kD, kX = (math.sqrt(k) for k in kap)
        p = self.pe + J
        rows = []
        M = np.zeros((self.PT.shape[0], p)); M[:, self.iT:self.iT + self.nT] = kT * self.PT; rows.append(M)
        M = np.zeros((self.PG.shape[0], p)); M[:, self.iG:self.iG + self.g] = kG * self.PG; rows.append(M)
        M = np.zeros((5 * self.g, p)); M[:, self.iD:self.iD + 5 * self.g] = kD * np.eye(5 * self.g); rows.append(M)
        M = np.zeros((self.PX.shape[0], p)); M[:, self.iX:self.iX + self.nX] = kX * self.PX; rows.append(M)
        M = np.zeros((1, p)); M[0, self.pe:] = W_CON / J; rows.append(M)
        M = np.zeros((J, p)); M[:, self.pe:] = math.sqrt(RIDGE_M) * np.eye(J); rows.append(M)
        return np.vstack(rows)


CFGS = {"k12": Cfg("k12", 12, 12, 1.0), "k16": Cfg("k16", 16, 16, 0.5)}


def day_c0(c0h, h):
    base = math.floor((c0h - 5.0) / 24.0 + 1e-9)
    return (np.floor((c0h + np.asarray(h, float) - 5.0) / 24.0 + 1e-9) - base + 1).astype(int)


def day_edges_of(c0h, Ht):
    """일차 경계 [0, e1, …, Ht] (KST 05시)."""
    e = [0.0]
    k = math.floor((c0h - 5.0) / 24.0 + 1e-9) + 1
    while True:
        h = 24.0 * k + 5.0 - c0h
        if h >= Ht - 1e-6:
            break
        if h > 1e-6:
            e.append(h)
        k += 1
    e.append(Ht)
    return e


class Cal:
    """달력 하나 (c0h, Ht)의 적분 격자와 특징 행렬 F (N, pe): η = F θ_η."""
    _cache = {}

    @classmethod
    def get(cls, cfg, c0h, Ht):
        key = (cfg.name, round(c0h, 6), round(Ht, 6))
        if key not in cls._cache:
            cls._cache[key] = cls(cfg, c0h, Ht)
        return cls._cache[key]

    def __init__(self, cfg, c0h, Ht):
        self.cfg, self.c0h, self.Ht = cfg, c0h, Ht
        N = int(round(Ht / DT))
        self.N, self.dt = N, Ht / N
        self.F = self.feat((np.arange(N) + 0.5) * self.dt)
        self.edges = day_edges_of(c0h, Ht)

    def feat(self, h):
        cfg = self.cfg
        h = np.asarray(h, float)
        c = np.mod(self.c0h + h, 24.0)
        d = np.minimum(day_c0(self.c0h, h), 5)
        F = np.zeros((len(h), cfg.pe))
        F[:, cfg.iT:cfg.iT + cfg.nT] = tbasis(h, cfg.nT)
        BG = pbasis(c, cfg.nG) @ cfg.Z
        F[:, cfg.iG:cfg.iG + cfg.g] = BG
        for k in range(1, 6):
            m = d == k
            F[m, cfg.iD + (k - 1) * cfg.g:cfg.iD + k * cfg.g] = BG[m]
        F[:, cfg.iX:cfg.iX + cfg.nX] = cfg.xbasis(self.Ht - h)
        return F

    def idx(self, t):
        u = np.clip(np.asarray(t, float) / self.dt, 0.0, self.N)
        i = np.minimum(np.floor(u).astype(int), self.N - 1)
        return i, u - i

    def cum_edges(self, lam):
        return np.concatenate([[0.0], np.cumsum(lam * self.dt)])

    def at(self, Ce, t):
        i, f = self.idx(t)
        return Ce[i] * (1 - f) + Ce[i + 1] * f


# ======================= 관측 =======================

def load_obs(obs_dir):
    O, PH, *_ = FC.build_obs(obs_dir)
    S = O["S"].astype(int)
    isph = O["isph"].copy()
    ta, tb, y = O["ta"].copy(), O["tb"].copy(), O["y"].copy()
    wp = O["wp"].copy()
    for i in np.where(isph)[0]:
        s = SEAS[S[i]]
        wp[i] = min(int(T.day_of(s, ta[i])), 5)               # 일차 번호를 일차 경계(KST 05시)로 다시 셈
        assert abs(tb[i] - HTRUE[s]) < 1e-6                    # 사진 적분 끝 = htrue
    return {"S": S, "ta": ta, "tb": tb, "y": y, "isph": isph, "wp": wp, "n": len(y)}


class Block:
    def __init__(self, cal, rows, j, ta, tb):
        self.cal, self.rows, self.j = cal, rows, j
        self.i0, self.f0 = cal.idx(ta)
        self.i1, self.f1 = cal.idx(tb)


class Model:
    """행 부분집합 + 시즌 목록 → 예측·야코비안."""

    def __init__(self, cfg, OBS, rows, seasons):
        self.cfg, self.rows, self.seasons, self.J = cfg, rows, seasons, len(seasons)
        self.y = OBS["y"][rows]
        self.isph = OBS["isph"][rows]
        self.wp = OBS["wp"][rows]
        self.n = len(rows)
        S = OBS["S"][rows]
        ta, tb = OBS["ta"][rows], OBS["tb"][rows]
        self.blocks = []
        for j, sn in enumerate(seasons):
            m = np.where(S == SIDX[sn])[0]
            if len(m):
                self.blocks.append(Block(Cal.get(cfg, SI[sn]["c0h"], HTRUE[sn]), m, j, ta[m], tb[m]))
        self.p = cfg.pe + self.J
        self._x = None

    def eval(self, x, jac=True):
        if self._x is not None and np.array_equal(x, self._x[0]) and (not jac or self._x[2] is not None):
            return self._x[1], self._x[2]
        pe = self.cfg.pe
        th, lnm = x[:pe], x[pe:]
        yh = np.zeros(self.n)
        Jm = np.zeros((self.n, self.p)) if jac else None
        for b in self.blocks:
            cal = b.cal
            lam = np.exp(cal.F @ th + lnm[b.j])
            Ce = cal.cum_edges(lam)
            yb = Ce[b.i1] * (1 - b.f1) + Ce[b.i1 + 1] * b.f1 - Ce[b.i0] * (1 - b.f0) - Ce[b.i0 + 1] * b.f0
            yh[b.rows] = yb
            if jac:
                Cf = np.vstack([np.zeros(pe), np.cumsum((lam * cal.dt)[:, None] * cal.F, axis=0)])
                Jm[b.rows, :pe] = (Cf[b.i1] * (1 - b.f1)[:, None] + Cf[b.i1 + 1] * b.f1[:, None]
                                   - Cf[b.i0] * (1 - b.f0)[:, None] - Cf[b.i0 + 1] * b.f0[:, None])
                Jm[b.rows, pe + b.j] = yb
        self._x = (x.copy(), yh, Jm)
        return yh, Jm

    def W0(self, sig):
        return FC.W0of(self.isph, self.wp, sig)


def lm_fit(M, PM, theta, sqrtW, nfev, tol=0.005):
    """Levenberg-Marquardt (정규방정식, 해석적 야코비안) — 목적 = ‖√W(ŷ−y)‖² + ‖PM θ‖².
    자체 LM(정규방정식)의 MINPACK QR이 이 크기(수천 줄 × 100열)에서 한 번 1초라 같은 목적을 직접 푼다 (속도 때문, 같은 해)."""
    y, W = M.y, sqrtW ** 2
    PtP = PM.T @ PM

    def cost(x):
        yh, _ = M.eval(x, jac=False)
        r = PM @ x
        return float(W @ (yh - y) ** 2 + r @ r)

    x = theta.copy()
    c = cost(x)
    lam = 1e-3
    for it in range(nfev):
        yh, Jm = M.eval(x, jac=True)
        Jw = sqrtW[:, None] * Jm
        H = Jw.T @ Jw + PtP
        g = Jw.T @ (sqrtW * (yh - y)) + PtP @ x
        dH = np.diag(H).copy()
        ok = False
        for _ in range(12):
            try:
                step = np.linalg.solve(H + lam * np.diag(dH + 1e-9), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = x + step
            cn = cost(xn)
            if cn <= c:
                ok = True
                break
            lam *= 4
        if not ok:
            break
        dec = c - cn
        x, c = xn, cn
        lam = max(lam / 3, 1e-9)
        if dec <= tol or np.max(np.abs(step)) < 1e-5:     # 비용 = 표준화 잔차 제곱합 (눈금 ~ 줄 수) — tol 미만 변화는 멈춤
            break
    return x


def fit_model(M, kap, theta0, sig=None, rounds=4, nfev=40, upd_sig=True, tol_ps=None, tol=0.005):
    """가중 Huber(IRLS) + 벌점. σ·φ는 되풀이로 갱신 (upd_sig). tol_ps가 있으면 Huber 무게가 그만큼 안 변하면 일찍 끝냄. 돌려줌: θ, r(=y−ŷ), sig."""
    PM = M.cfg.pm(kap, M.J)
    theta = theta0.copy()
    sig = FC.Sg(0.03, 1.0) if sig is None else sig
    r = M.y - M.eval(theta, jac=False)[0]
    ps_prev = None
    for rnd in range(rounds):
        W0 = M.W0(sig)
        ps = FC.psi(r, W0)
        if tol_ps is not None and ps_prev is not None and np.max(np.abs(ps - ps_prev)) < tol_ps:
            break
        ps_prev = ps
        theta = lm_fit(M, PM, theta, np.sqrt(W0 * ps), nfev, tol)
        r = M.y - M.eval(theta, jac=False)[0]
        if upd_sig and rnd < rounds - 1:
            a, b, c3 = FC.sig_of(r, M.isph, M.wp)
            sig = FC.Sg(a, b, c3)
    return theta, r, sig


def cov_of(M, kap, theta, r, sig):
    """가우스-뉴턴 공분산 (JᵀWJ + 벌점)⁻¹ — 척도 1."""
    PM = M.cfg.pm(kap, M.J)
    W0 = M.W0(sig)
    sw = np.sqrt(W0 * FC.psi(r, W0))
    _, Jm = M.eval(theta, jac=True)
    Jw = sw[:, None] * Jm
    return np.linalg.pinv(Jw.T @ Jw + PM.T @ PM, hermitian=True)


def shift_mean(cfg, theta):
    """ln m 평균을 0으로 옮겨 f_T 절편(기저 합 1)에 더한다."""
    th = theta.copy()
    mu = float(th[cfg.pe:].mean())
    th[cfg.pe:] -= mu
    th[cfg.iT:cfg.iT + cfg.nT] += mu
    return th


# ======================= 교차검증 =======================

def cv_search(cfg, OBS, rows, seasons, kap0, theta0, sig, log=None, sweeps=2):
    """관측 무작위 5겹, κ 넷 좌표별 격자 탐색 (2바퀴). 겹마다 θ를 이웃 κ에서 이어받아 따라 걷는다."""
    rng = np.random.default_rng(SEED)
    fold = rng.permutation(len(rows)) % NF
    models = [(Model(cfg, OBS, rows[fold != f], seasons), Model(cfg, OBS, rows[fold == f], seasons)) for f in range(NF)]
    cache = {}

    def evalk(ik, start_th, first=False):
        if ik in cache:
            return cache[ik]
        kap = [KGRID[i] for i in ik]
        errs = np.zeros(NF)
        ths = []
        for f in range(NF):
            mtr, mte = models[f]
            # 처음 한 번은 겹 자료만 본 차가운 시작에서 끝까지 (전체 자료로 맞춘 해를 시작으로 쓰면 뺀 관측이 새어 든다), 이웃 κ는 이웃 해에서 이어받음
            th, _, _ = fit_model(mtr, kap, start_th[f], sig, rounds=10 if first else 5, nfev=60 if first else 25, upd_sig=False, tol_ps=0.03, tol=0.01 if first else 0.05)
            rr = mte.y - mte.eval(th, jac=False)[0]
            errs[f] = float(FC.rho_loss(rr, mte.W0(sig)).sum())
            ths.append(th)
        cache[ik] = (float(errs.sum()), errs, ths)
        return cache[ik]

    cur = [min(range(len(KGRID)), key=lambda i: abs(math.log10(KGRID[i]) - math.log10(k))) for k in kap0]
    best = evalk(tuple(cur), [theta0.copy() for _ in range(NF)], first=True)
    scans = {}
    for sw in range(sweeps):
        for c in range(4):
            res = {cur[c]: best}
            base_th = best[2]
            se_est = float(np.std(best[1], ddof=1) * math.sqrt(NF))
            for direction in (1, -1):
                th_prev = base_th
                i = cur[c] + direction
                while 0 <= i < len(KGRID):
                    ik = list(cur); ik[c] = i
                    e = evalk(tuple(ik), th_prev)
                    res[i] = e
                    th_prev = e[2]
                    if e[0] > best[0] + 1.5 * se_est:            # 1-SE 범위 밖으로 1.5 SE 넘게 올라가면 그쪽은 더 안 걷는다
                        break
                    i += direction
            ib = min(res, key=lambda i: res[i][0])
            cur[c] = ib
            best = res[ib]
            scans[(sw, c)] = {i: v[0] for i, v in res.items()}
            if log:
                log(f"    바퀴 {sw + 1} κ_{KNAMES[c]} -> {KGRID[ib]:.3g} cv={best[0]:.2f}")
    errs = best[1]
    se = float(np.std(errs, ddof=1) * math.sqrt(NF))
    ranges = {}
    for c in range(4):
        tot = scans[(sweeps - 1, c)]
        ok = sorted(i for i, v in tot.items() if v <= best[0] + se)
        ranges[KNAMES[c]] = {"lo": KGRID[ok[0]], "hi": KGRID[ok[-1]], "min_in_1se": KGRID[ok[0]],
                             "curve": [[KGRID[i], round(tot[i], 3)] for i in sorted(tot)]}
    return {"kappa": [KGRID[i] for i in cur], "cv": float(best[0]), "se": se, "fold_err": [float(e) for e in errs],
            "ranges": ranges, "nfits": len(cache) * NF}


# ======================= 적합 묶음 =======================

def init_theta(cfg, J):
    """f_T = WP12 합친 곡선 λ의 ln을 ln h에 맞춘 것, 나머지 0."""
    tab = json.load(open(WP12_JSON))["G1"]["tables"]["combined_grid_0.25h"]
    h = np.array([t["h"] for t in tab]); lam = np.array([t["lam"] for t in tab])
    m = h >= 0.25
    bT, *_ = np.linalg.lstsq(tbasis(h[m], cfg.nT), np.log(np.maximum(lam[m], 1e-4)), rcond=None)
    th = np.zeros(cfg.pe + J)
    th[cfg.iT:cfg.iT + cfg.nT] = bT
    return th


def fit_cv(cfg, OBS, seasons, kap_start, theta_start, log=None):
    """seasons 모든 관측으로 σ·φ 정하고 κ를 교차검증한다."""
    rows = np.where(np.isin(OBS["S"], [SIDX[s] for s in seasons]))[0]
    M = Model(cfg, OBS, rows, seasons)
    th1, _, sig = fit_model(M, kap_start, theta_start, None, rounds=5, nfev=60)
    cv = cv_search(cfg, OBS, rows, seasons, kap_start, init_theta(cfg, len(seasons)), sig, log=log)
    return {"cfg": cfg, "M": M, "th1": th1, "sig": sig, "cv": cv, "seasons": seasons, "rows": rows}


def fit_final(st, kap):
    cfg, M, J = st["cfg"], st["M"], st["M"].J
    th, r, sig = fit_model(M, kap, st["th1"], st["sig"], rounds=6, nfev=60, tol_ps=0.005)
    cov = cov_of(M, kap, th, r, sig)
    a = np.zeros((J, cfg.pe + J))
    for j in range(J):
        a[j, cfg.pe:] = -1.0 / J
        a[j, cfg.pe + j] += 1.0
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", a, cov, a), 0.0))
    return {"cfg": cfg, "theta": shift_mean(cfg, th), "kap": list(kap), "sig": sig, "se_lnm": se, "r": r,
            "cv": st["cv"], "seasons": st["seasons"], "nrow": len(st["rows"])}


# ======================= 곡선 =======================

def shape_on(cfg, th_eta, c0h, Ht):
    cal = Cal.get(cfg, c0h, Ht)
    lam = np.exp(cal.F @ th_eta)
    return cal, lam, cal.cum_edges(lam)


def out_grid(Ht):
    g = np.arange(int(math.floor(Ht / OUT_GRID + 1e-9)) + 1) * OUT_GRID
    return np.append(g, Ht) if Ht - g[-1] > 1e-6 else g


def repro(cfg, th_eta, mbar21, tab):
    h = np.array([t["h"] for t in tab]); P = np.array([t["P"] for t in tab])
    lo = np.array([t["P_lo"] for t in tab]); hi = np.array([t["P_hi"] for t in tab])
    cal, lam, Ce = shape_on(cfg, th_eta, 5.0, 120.0)
    Pf = (Ce[-1] - cal.at(Ce, h)) * mbar21
    dP = Pf - P
    out = {}
    for name, m in (("h12_120", (h >= 12) & (h <= 120)), ("h6_12", (h >= 6) & (h < 12)), ("h6_120", (h >= 6) & (h <= 120))):
        a = np.abs(dP[m])
        k = int(np.argmax(a))
        out[name] = {"max_abs_dP": round(float(a.max()), 5), "worst_h": float(h[m][k]), "worst_kst": float((5 + h[m][k]) % 24),
                     "rms_dP": round(float(np.sqrt((dP[m] ** 2).mean())), 5),
                     "frac_out_of_band": round(float(((Pf[m] < lo[m]) | (Pf[m] > hi[m])).mean()), 4),
                     "frac_over_0.01": round(float((a > 0.01).mean()), 4), "n": int(m.sum())}
    return out, Pf, dP, h


def curves_for_plot(cfg, th):
    g = cfg.g
    thT = th[cfg.iT:cfg.iT + cfg.nT]; thG = th[cfg.iG:cfg.iG + g]
    thD = th[cfg.iD:cfg.iD + 5 * g].reshape(5, g); thX = th[cfg.iX:cfg.iX + cfg.nX]
    hh = np.concatenate([[0.5], np.arange(1, 131, 1.0)])
    cc = np.arange(0, 24, 0.5)
    BG = pbasis(cc, cfg.nG) @ cfg.Z
    fG = BG @ thG
    fD = [BG @ thD[d] for d in range(5)]
    xx = np.arange(0, XMAX + 1e-9, 0.25)
    r6 = lambda a: [round(float(v), 5) for v in a]  # noqa: E731
    wave = fG + np.mean(fD[1:4], axis=0)
    kmax, kmin = int(np.argmax(wave)), int(np.argmin(wave))
    out = {"f_T": {"h": r6(hh), "val": r6(tbasis(hh, cfg.nT) @ thT)}, "f_G": {"c": r6(cc), "val": r6(fG)},
           "f_d": {"c": r6(cc), "val": {str(d + 1): r6(fD[d]) for d in range(5)}},
           "f_G_plus_f_d": {str(d + 1): r6(fG + fD[d]) for d in range(5)},
           "f_X": {"x": r6(xx), "val": r6(cfg.xbasis(xx) @ thX)},
           "wave_days_2_4": {"fast_kst": float(cc[kmax]), "slow_kst": float(cc[kmin]), "ratio": round(float(math.exp(wave[kmax] - wave[kmin])), 3),
                             "note": "f_G + 2~4일차 f_d 평균 (로그)의 최고·최저 KST 시각과 비 (WP12: 22시·04시·4.3)"},
           "wave_by_day": {}}
    for d in range(5):
        w = fG + fD[d]
        a, b = int(np.argmax(w)), int(np.argmin(w))
        out["wave_by_day"][str(d + 1)] = {"fast_kst": float(cc[a]), "slow_kst": float(cc[b]), "ratio": round(float(math.exp(w[a] - w[b])), 3)}
    # WP12 describe.wave_days2_4와 같은 셈: 05시 시작 달력에서 2~4일차 같은 KST 시각의 λ(추세 포함) 평균 → 최고·최저 (WP12: 22시 0.01338 · 04시 0.00309 · 비 4.33)
    cal, lam, Ce = shape_on(cfg, th[:cfg.pe], 5.0, 120.0)
    hq = np.arange(0, 24, 0.25)
    mean_l = np.mean([np.exp(cal.feat(24.0 * k + hq) @ th[:cfg.pe]) for k in (1, 2, 3)], axis=0)
    cq = np.mod(5.0 + hq, 24.0)
    a, b = int(np.argmax(mean_l)), int(np.argmin(mean_l))
    out["wave_lambda_days_2_4"] = {"fast_kst": float(cq[a]), "fast": round(float(mean_l[a]), 5), "slow_kst": float(cq[b]), "slow": round(float(mean_l[b]), 5),
                                   "ratio": round(float(mean_l[a] / mean_l[b]), 3), "note": "2~4일차 같은 KST 시각 λ(m=1, 추세 포함) 평균의 최고/최저 — WP12 describe.wave_days2_4와 같은 셈 (22시 0.01338·04시 0.00309·4.33)"}
    return out


# ======================= 일차 수준 · ln m 고정 꼴 =======================

def level_fit(OBS, rows, cal, Ce, sig, edges):
    """꼴(Ce) 고정, 일차 수준 L_d 선형 (Huber IRLS, L ≥ 0). 돌려줌: {일차(0부터): (L, se(ln L))}."""
    ta, tb, y = OBS["ta"][rows], OBS["tb"][rows], OBS["y"][rows]
    D = len(edges) - 1
    A = np.zeros((len(rows), D))
    for d in range(D):
        lo = np.maximum(ta, edges[d]); hi = np.minimum(tb, edges[d + 1])
        ok = hi > lo
        A[ok, d] = cal.at(Ce, hi[ok]) - cal.at(Ce, lo[ok])
    W0 = FC.W0of(OBS["isph"][rows], OBS["wp"][rows], sig)
    ps = np.ones(len(y))
    keep = A.sum(axis=0) > 1e-9
    L = np.zeros(D)
    for _ in range(15):
        sw = np.sqrt(W0 * ps)
        Lk, _ = nnls(sw[:, None] * A[:, keep], sw * y, maxiter=2000)
        L = np.zeros(D); L[keep] = Lk
        psn = FC.psi(y - A @ L, W0)
        done = np.max(np.abs(psn - ps)) < 1e-5
        ps = psn
        if done:
            break
    Ak = A[:, keep]
    cov = np.linalg.pinv(Ak.T @ ((W0 * ps)[:, None] * Ak), hermitian=True)
    out = {}
    for a, d in enumerate(np.where(keep)[0]):
        if L[d] > 1e-6:
            out[int(d)] = (float(L[d]), float(math.sqrt(max(cov[a, a], 0.0)) / L[d]))
    return out


def m_fit(OBS, rows, cal, Ce, sig):
    """꼴 고정, ln m만 (Huber IRLS). 돌려줌: (ln m, se)."""
    ta, tb, y = OBS["ta"][rows], OBS["tb"][rows], OBS["y"][rows]
    Y = cal.at(Ce, tb) - cal.at(Ce, ta)
    W0 = FC.W0of(OBS["isph"][rows], OBS["wp"][rows], sig)
    ps = np.ones(len(y))
    for _ in range(30):
        w = W0 * ps
        m = float((w * y * Y).sum() / (w * Y * Y).sum())
        psn = FC.psi(y - m * Y, W0)
        done = np.max(np.abs(psn - ps)) < 1e-6
        ps = psn
        if done:
            break
    return math.log(m), 1.0 / math.sqrt(float((W0 * ps * (m * Y) ** 2).sum()))


# ======================= 맞힐 시즌 =======================

OBS = None


def init_worker(obs_dir):
    global OBS
    OBS = load_obs(obs_dir)


def target_job(args):
    tn, cfg_name, kap_start, full_eta = args
    t0 = time.time()
    cfg = CFGS[cfg_name]
    full_eta = np.array(full_eta)
    tseason = f"S{tn}"
    train = [s for s in SEAS if int(s[1:]) < tn]
    J = len(train)
    th_start = np.zeros(cfg.pe + J)
    th_start[:cfg.pe] = full_eta
    logs = []
    st = fit_cv(cfg, OBS, train, kap_start, th_start, log=logs.append)
    fit = fit_final(st, st["cv"]["kappa"])
    th, sig = fit["theta"], fit["sig"]
    th_eta, lnm, se = th[:cfg.pe], th[cfg.pe:], fit["se_lnm"]
    c0h, Ht = SI[tseason]["c0h"], HTRUE[tseason]
    cal, _, Ce = shape_on(cfg, th_eta, c0h, Ht)
    hg = out_grid(Ht)
    Pbar = Ce[-1] - cal.at(Ce, hg)
    lam_g = np.exp(cal.feat(hg) @ th_eta)
    edges = cal.edges
    Dn = len(edges) - 1
    Ad = np.zeros((Dn, len(hg)))
    for d in range(Dn):
        Ad[d] = np.where(hg >= edges[d + 1] - 1e-9, 0.0,
                         cal.at(Ce, edges[d + 1]) - cal.at(Ce, np.maximum(hg, edges[d])))
    lnm_train = {s: [round(float(lnm[j]), 5), round(float(se[j]), 5)] for j, s in enumerate(train)}
    raw = float(np.var(lnm, ddof=1)) if J >= 2 else None
    noise = float(np.mean(se ** 2))
    var = 0.3 ** 2 if J < 2 else max(raw - noise, 0.01 ** 2)
    lnm_prior = {"mean": round(float(lnm.mean()), 6), "var": round(var, 6), "raw_var": None if raw is None else round(raw, 6),
                 "noise_var": round(noise, 6), "J": J}
    # ---- tau
    samples = {k: [] for k in range(1, 6)}
    noises = {k: [] for k in range(1, 6)}
    allv, alln, dropped = [], [], 0
    for j, s in enumerate(train):
        cj, _, Cej = shape_on(cfg, th_eta, SI[s]["c0h"], HTRUE[s])
        rws = np.where(OBS["S"] == SIDX[s])[0]
        for d, (L, seln) in level_fit(OBS, rws, cj, Cej, sig, cj.edges).items():
            if seln > 1.0:
                dropped += 1
                continue
            lt = math.log(L) - float(lnm[j])
            k = min(d + 1, 5)
            samples[k].append(lt); noises[k].append(seln ** 2)
            allv.append(lt); alln.append(seln ** 2)
    pooled_raw = float(np.var(allv, ddof=1)) if len(allv) >= 2 else 0.3 ** 2
    pooled_noise = float(np.mean(alln)) if alln else 0.0
    pooled = math.sqrt(max(pooled_raw - pooled_noise, 0.01 ** 2)) if len(allv) >= 2 else 0.3
    per_day, n_l, raw_l, noise_l = [], [], [], []
    for k in range(1, 6):
        v = samples[k]
        n_l.append(len(v))
        rw = float(np.var(v, ddof=1)) if len(v) >= 2 else None
        nz = float(np.mean(noises[k])) if v else None
        raw_l.append(None if rw is None else round(rw, 6)); noise_l.append(None if nz is None else round(nz, 6))
        per_day.append(round(math.sqrt(max(rw - nz, 0.01 ** 2)), 5) if len(v) >= 3 else round(pooled, 5))
    tau = {"per_day": per_day, "pooled": round(pooled, 5), "n": n_l, "raw": raw_l, "noise": noise_l,
           "pooled_raw": round(pooled_raw, 6), "pooled_noise": round(pooled_noise, 6), "dropped_se_gt_1": dropped,
           "note": "raw·noise는 분산 (ln t_{j,d}의 시즌 사이 표본 분산 · 델타법 se² 평균), per_day·pooled는 sd. S28 6일차는 5일차 칸에 함께 들어감."}
    # ---- m_true · t_true
    rws = np.where(OBS["S"] == SIDX[tseason])[0]
    lmt, lmt_se = m_fit(OBS, rws, cal, Ce, sig)
    t_true = [None] * Dn
    for d, (L, _) in level_fit(OBS, rws, cal, Ce, sig, edges).items():
        t_true[d] = round(math.log(L) - lmt, 5)
    # ---- stability
    calf, _, Cef = shape_on(cfg, full_eta, c0h, Ht)
    Pf = Cef[-1] - calf.at(Cef, hg)
    mk = (hg >= 12) & (hg <= Ht)
    mk2 = (hg >= 12) & (hg <= Ht - 1)
    dl = np.log(np.maximum(Pbar[mk2], 1e-12)) - np.log(np.maximum(Pf[mk2], 1e-12))
    stab = {"max_dP_vs_full": round(float(np.abs(Pbar - Pf)[mk].max()), 5),
            "max_dlnP_shape_vs_full": round(float(np.abs(dl - dl.mean()).max()), 5)}
    chk = {"A_day_sum_max_err": float(np.abs(Ad.sum(axis=0) - Pbar).max()), "P_monotone": bool(np.all(np.diff(Pbar) <= 1e-12)),
           "lam_pos": bool(np.all(lam_g > 0)), "mean_lnm_train": float(lnm.mean()), "P_last": float(Pbar[-1]),
           "day_edges_check": [int(day_c0(c0h, 0.5 * (edges[d] + edges[d + 1]))) for d in range(Dn)]}
    cvd = st["cv"]
    pen = {"kappa": {k: float(v) for k, v in zip(KNAMES, fit["kap"])}, "cfg": cfg_name,
           "sigma": {"sigma_ph": float(sig), "phi_pair": float(sig.phi), "sigma_ph_day": [round(float(v), 5) for v in sig.sd]}}
    cv = {"cv": round(cvd["cv"], 3), "se": round(cvd["se"], 3), "fold_err": [round(e, 3) for e in cvd["fold_err"]],
          "ranges_1se": {k: {"lo": v["lo"], "hi": v["hi"]} for k, v in cvd["ranges"].items()}, "nrow": fit["nrow"], "nfits": cvd["nfits"]}
    R = lambda a: [round(float(v), 6) for v in a]  # noqa: E731
    res = {"train": train, "H": float(SI[tseason]["H"]), "Ht": float(Ht), "c0h": float(c0h), "h": R(hg), "P": R(Pbar), "lam": R(lam_g),
           "day_edges": [round(e, 4) for e in edges], "A_day": [R(a) for a in Ad], "lnm_prior": lnm_prior, "lnm_train": lnm_train,
           "tau": tau, "m_true": {"lnm": round(lmt, 5), "se": round(lmt_se, 5)}, "t_true": t_true, "stability": stab,
           "penalties": pen, "cv": cv, "checks": chk}
    return tn, res, time.time() - t0, logs


def full_job(args):
    cfg_name, kap_start = args
    t0 = time.time()
    cfg = CFGS[cfg_name]
    st = fit_cv(cfg, OBS, SEAS, kap_start, init_theta(cfg, len(SEAS)), log=lambda m: print(f"  [{cfg_name}] {m}", flush=True))
    out = {"secs": time.time() - t0}
    for rule in ("cvmin", "1se_min"):
        kap = st["cv"]["kappa"] if rule == "cvmin" else [st["cv"]["ranges"][k]["min_in_1se"] for k in KNAMES]
        out[rule] = fit_final(st, kap)
    return out


# ======================= 메인 =======================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--targets", default=None, help="예: 21,22")
    ap.add_argument("--only-full", action="store_true")
    ap.add_argument("--force", default=None, help="판 지정 (예: k16:cvmin) — FULL 점검 순서를 건너뜀 (디버그용)")
    a = ap.parse_args()
    t_all = time.time()

    def log(msg):
        print(f"[{time.time() - t_all:7.1f}s] {msg}", flush=True)

    global OBS
    OBS = load_obs(a.obs_dir)
    log(f"관측 {OBS['n']}줄 (쌍 {int((~OBS['isph']).sum())}, 사진 {int(OBS['isph'].sum())})")
    tab = json.load(open(WP12_JSON))["G1"]["tables"]["combined_grid_0.25h"]
    jclock = [i for i, s in enumerate(SEAS) if SI[s]["clock_ok"]]

    # ---------- FULL: 판 고르기 ----------
    # 두 매듭 판의 교차검증은 서로 독립이라 2프로세스로 동시에 돌리고, 판 고르는 순서(k12 → k16 → k16/1-SE)는 그 결과를 차례로 본다
    plan = [("k12", "cvmin"), ("k16", "cvmin"), ("k16", "1se_min")]
    if a.force:
        c_, r_ = a.force.split(":")
        plan = [(c_, r_)]
    need = sorted({c for c, _ in plan})
    t0 = time.time()
    with mp.Pool(min(a.procs, len(need)), initializer=init_worker, initargs=(a.obs_dir,)) as pool:
        outs = dict(zip(need, pool.map(full_job, [(c, [10.0] * 4) for c in need])))
    log(f"FULL 교차검증 {need} 끝 ({time.time() - t0:.0f}s)")
    tries, fits, chosen = [], {}, None
    for cfg_name, rule in plan:
        cfg = CFGS[cfg_name]
        fit = outs[cfg_name][rule]
        fits[(cfg_name, rule)] = fit
        th = fit["theta"]
        mbar = math.exp(float(np.mean(th[cfg.pe:][jclock])))
        rep, Pf, dP, hh = repro(cfg, th[:cfg.pe], mbar, tab)
        r = rep["h12_120"]
        meets = r["max_abs_dP"] <= 0.01
        cv = fit["cv"]
        log(f"  {cfg_name}/{rule} κ={['%.3g' % k for k in fit['kap']]} cv={cv['cv']:.1f} (se {cv['se']:.1f}) |ΔP| h12~120 max {r['max_abs_dP']} rms {r['rms_dP']} 띠밖 {r['frac_out_of_band']} (h6~12 max {rep['h6_12']['max_abs_dP']}) -> {'통과' if meets else '못 미침'}")
        tries.append({"cfg": cfg_name, "rule": rule, "kappa": dict(zip(KNAMES, fit["kap"])), "cv": round(cv["cv"], 3),
                      "cv_se": round(cv["se"], 3), "repro": rep, "meets": bool(meets), "cv_secs": round(outs[cfg_name]["secs"], 1)})
        if meets:
            chosen = (cfg_name, rule)
            break
    if chosen is None:
        best = min(tries, key=lambda t: t["repro"]["h12_120"]["max_abs_dP"])
        chosen = (best["cfg"], best["rule"])
        log(f"기준 못 미침 — 가장 가까운 판 {chosen}")
    cfg_name, rule = chosen
    cfg = CFGS[cfg_name]
    fit = fits[chosen]
    th = fit["theta"]
    lnm_full = th[cfg.pe:]
    mbar = math.exp(float(np.mean(lnm_full[jclock])))
    rep, Pf, dP, hh = repro(cfg, th[:cfg.pe], mbar, tab)
    se = fit["se_lnm"]
    cvd = fit["cv"]
    sg = fit["sig"]
    full = {
        "chosen": {"cfg": cfg_name, "rule": rule, "meets_repro": bool(rep["h12_120"]["max_abs_dP"] <= 0.01),
                   "knots": {"f_T": cfg.nT, "f_G": cfg.nG, "f_X_knot_h": cfg.dk, "n_basis_X": cfg.nX}},
        "repro": rep, "tries": tries,
        "penalties": {"kappa": dict(zip(KNAMES, fit["kap"])), "rule": rule,
                      "sigma": {"sigma_ph": float(sg), "phi_pair": float(sg.phi), "sigma_ph_day": [round(float(v), 5) for v in sg.sd]}},
        "cv": {"cv": round(cvd["cv"], 3), "se": round(cvd["se"], 3), "fold_err": [round(e, 3) for e in cvd["fold_err"]],
               "ranges_1se": {k: {"lo": v["lo"], "hi": v["hi"], "curve": v["curve"]} for k, v in cvd["ranges"].items()}, "nrow": fit["nrow"]},
        "m_bar21": round(mbar, 5),
        "lnm": {s: [round(float(lnm_full[j]), 5), round(float(se[j]), 5)] for j, s in enumerate(SEAS)},
        "shape": curves_for_plot(cfg, th),
        "grid_compare_0.25h": {"h": [round(float(v), 2) for v in hh], "P_full": [round(float(v), 5) for v in Pf], "dP": [round(float(v), 5) for v in dP]},
        "wp12_compare_note": "WP12 G2 같은 자 (h 6~120 | h12~): ④bK2 max 0.0407 rms 0.0125 띠밖 0.083 · M3 기본 max 0.2217(h6) | h12~ 0.0298, rms 0.029, 띠밖 0.40 · F f2pow 0.4672 | h12~ 0.1136, rms 0.0706, 띠밖 0.602 · act 0.3773 | h12~ 0.0341, rms 0.0386, 띠밖 0.147",
    }
    pack = {"header": None, "full": full, "targets": {}}
    if a.only_full:
        write_out(a, pack, time.time() - t_all, "only_full")
        return
    tl = [int(x) for x in a.targets.split(",")] if a.targets else list(range(21, 43))
    jobs = [(tn, cfg_name, fit["kap"], th[:cfg.pe].tolist()) for tn in tl]
    log(f"맞힐 시즌 {len(jobs)}개 (procs {a.procs})")
    with mp.Pool(a.procs, initializer=init_worker, initargs=(a.obs_dir,)) as pool:
        for tn, res, secs, logs in pool.imap_unordered(target_job, jobs):
            pack["targets"][f"S{tn}"] = res
            log(f"  S{tn}: κ={['%.3g' % v for v in res['penalties']['kappa'].values()]} lnm sd={math.sqrt(res['lnm_prior']['var']):.3f} "
                f"τ={res['tau']['pooled']:.3f} m_true={res['m_true']['lnm']:.3f} stab={res['stability']['max_dP_vs_full']:.4f} {secs:.0f}s")
    pack["targets"] = {k: pack["targets"][k] for k in sorted(pack["targets"], key=lambda s: int(s[1:]))}
    write_out(a, pack, time.time() - t_all, "all")


def write_out(a, pack, secs, mode):
    pack["header"] = {
        "about": "WP13a 침수 꼴 — 계층 가법 모형(HGAM) 24시즌 적합 + 맞힐 시즌 S21~S42 앞 시즌만 (flood_shape_D1)",
        "data_ver": CM.DATA_VER, "date": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST"),
        "made": {"git_head": T._git_head(), "cmd": "python3 -I tools/solo_raid/lab7/wp13_flood_shape.py --obs-dir W/obs7/full --out data/solo-raid/analysis/r7/wp13/flood_hgam_D1.json --procs 2"},
        "secs": round(secs, 1), "mode": mode,
        "rules": {
            "model": "λ_j(h) = m_j exp(f_T(ln max(h,0.5)) + f_G(c) + f_d(c) + f_X(x)), d=min(일차,5), x=Ht−h, c=KST 시각",
            "f_T": "3차 B-스플라인 z=ln h' ∈ [ln0.5, ln131] 균일 매듭, 벌점 κ_T‖D2β‖²",
            "f_G": "24시간 주기 3차 B-스플라인, 평균 0 (정규직교 영공간 재매개화), 벌점 κ_G‖주기 D2β‖²",
            "f_d": "같은 주기 기저 일차(1~5)마다, 평균 0, 벌점 κ_D‖β_d‖²",
            "f_X": "x∈[0,12] 3차 B-스플라인, x=12에서 값·기울기 0 (끝 기저 둘 제거), x>12 0, 벌점 κ_X‖D2β‖²",
            "ln m": "시즌마다 자유; 맞춘 뒤 학습 시즌 평균 0으로 옮겨 f_T 절편에 더함; 맞추는 동안 Σ ln m=0 제약 줄(무게 1e3/J)과 능선 1e-2",
            "obs": "WP12 G1 build_obs (쌍 + A·B 사진 main), 사진 일차 번호는 T.day_of(s, h) min 5로 다시 셈, 사진 적분 끝 Ht=htrue",
            "weights": "쌍 1/(φ var), 사진 1/σ_ph(d)², Huber c=1.345 (표준화 잔차), σ·φ는 잔차 강건 MAD 되풀이 (FC.sig_of)",
            "integral": "0.05시간 칸 중점 누적합, 구간 끝은 누적값 선형 보간",
            "kappa": f"관측 무작위 {NF}겹 CV (가중 Huber 손실 합, 겹 사이 SE=sd·√5), 좌표 T,G,D,X마다 격자 10^[-2..5] 반 자리 ({len(KGRID)}개) 2바퀴, 이웃 κ의 해를 이어받아 따라 걷기 (CV 안에서 σ·φ 고정, IRLS 2회). 1-SE 범위 = 마지막 바퀴의 좌표별 스캔",
            "full_plan": "k12 (기저 12·12, X 매듭 1h) CV최소 → 기준 못 미치면 k16 (16·16, 0.5h) CV최소 → 그래도 못 미치면 k16 + 좌표별 1-SE 안 최소 κ. 모두 못 미치면 h12~120 max|ΔP| 가장 작은 판",
            "repro_criterion": "h 12~120 max|ΔP| ≤ 0.01 (P_full = ∫exp(η)·m̄21 vs WP12 합친 곡선 combined_grid_0.25h)",
            "lnm_prior": "mean=학습 ln m 평균, raw_var=표본분산(ddof1), noise_var=se² 평균, var=max(raw−noise, 0.01²); 학습 2시즌 미만 0.3²",
            "tau": "학습 시즌마다 꼴 고정 + 일차 수준 L_{j,d} 선형 (Huber IRLS, NNLS); ln t=ln L−ln m_j; 잡음=se²(델타법); τ_d²=max(시즌 사이 분산−평균 잡음, 0.01²) (n≥3, 아니면 pooled); se(ln L)>1인 (j,d)는 제외",
        }}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, separators=(",", ":"))
    md = (a.out[:-5] if a.out.endswith(".json") else a.out) + ".md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(make_md(pack))
    print("wrote", a.out, md, f"{secs:.0f}s", flush=True)


def make_md(pack):
    H, F = pack["header"], pack["full"]
    L = [f"# WP13a 침수 꼴 (HGAM) — 요약\n\n{H['date']} · data {H['data_ver']} · git {H['made']['git_head']} · 걸린 시간 {H['secs']}초 · 모드 {H['mode']}\n"]
    ch = F["chosen"]
    L.append(f"## 24시즌 전체 적합과 재현 점검\n\n고른 판: **{ch['cfg']} / {ch['rule']}** (기저 f_T {ch['knots']['f_T']} · f_G {ch['knots']['f_G']} · f_X 매듭 {ch['knots']['f_X_knot_h']}시간), 기준(h12~120 최대 \\|ΔP\\| ≤ 0.01) 통과: **{ch['meets_repro']}**\n")
    L.append("| 판 | κ (T,G,D,X) | CV | max\\|ΔP\\| h12~120 (h, KST) | RMS | 띠 밖 | h6~12 max / 띠 밖 | 통과 |\n|---|---|---|---|---|---|---|---|")
    for t in F["tries"]:
        r = t["repro"]
        L.append(f"| {t['cfg']}/{t['rule']} | {', '.join(f'{v:.3g}' for v in t['kappa'].values())} | {t['cv']:.1f} | {r['h12_120']['max_abs_dP']} ({r['h12_120']['worst_h']}, {r['h12_120']['worst_kst']}시) | {r['h12_120']['rms_dP']} | {r['h12_120']['frac_out_of_band']} | {r['h6_12']['max_abs_dP']} / {r['h6_12']['frac_out_of_band']} | {t['meets']} |")
    r = F["repro"]
    L.append(f"\n고른 판 h6~120: max {r['h6_120']['max_abs_dP']} rms {r['h6_120']['rms_dP']} 띠 밖 {r['h6_120']['frac_out_of_band']}.")
    L.append("\n같은 자의 WP12 G2 (h 6~120 기준 | h12~): ④bK2 max 0.0407 / RMS 0.0125 / 띠 밖 0.083 · M3 기본 0.2217(h6) \\| 0.0298 / 0.029 / 0.40 · F f2pow 0.4672 \\| 0.1136 / 0.0706 / 0.602 · act 0.3773 \\| 0.0341 / 0.0386 / 0.147.\n")
    w = F["shape"]["wave_days_2_4"]
    wl = F["shape"].get("wave_lambda_days_2_4") or {"fast_kst": None, "fast": None, "slow_kst": None, "slow": None, "ratio": None}
    L.append(f"WP12와 같은 셈 (2~4일차 같은 KST 시각 λ 평균, 추세 포함): 최고 {wl['fast_kst']}시 {wl['fast']} · 최저 {wl['slow_kst']}시 {wl['slow']} · 비 **{wl['ratio']}** (WP12 22시 0.01338 · 04시 0.00309 · 4.33).")
    L.append(f"2~4일차 추세를 뺀 물결 (f_G + f_d 평균): 가장 빠른 KST {w['fast_kst']}시, 가장 느린 {w['slow_kst']}시, 비 {w['ratio']} (WP12 22시·04시·4.3). 일차별: " +
             ", ".join(f"{d}일차 {v['fast_kst']}시/{v['slow_kst']}시 비 {v['ratio']}" for d, v in F["shape"]["wave_by_day"].items()) + "\n")
    L.append(f"m̄21 = {F['m_bar21']} · σ_ph(d) {F['penalties']['sigma']['sigma_ph_day']} · φ_pair {F['penalties']['sigma']['phi_pair']:.3f} · κ {F['penalties']['kappa']}\n")
    if pack["targets"]:
        L.append("## 맞힐 시즌별\n\n| 시즌 | 학습 J | κ (T,G,D,X) | lnm_prior sd | τ pooled | τ per_day | m_true | stab max dP | stab dlnP |\n|---|---|---|---|---|---|---|---|---|")
        for k, t in pack["targets"].items():
            L.append(f"| {k} | {t['lnm_prior']['J']} | {', '.join(f'{v:.3g}' for v in t['penalties']['kappa'].values())} | {math.sqrt(t['lnm_prior']['var']):.3f} | {t['tau']['pooled']:.3f} | {t['tau']['per_day']} | {t['m_true']['lnm']:.3f}±{t['m_true']['se']:.3f} | {t['stability']['max_dP_vs_full']} | {t['stability']['max_dlnP_shape_vs_full']} |")
        bad = [k for k, t in pack["targets"].items() if t["checks"]["A_day_sum_max_err"] > 1e-6 or not t["checks"]["P_monotone"] or not t["checks"]["lam_pos"] or abs(t["checks"]["mean_lnm_train"]) > 1e-6]
        L.append(f"\n자체 점검 (A_day 합=P · P 단조 감소 · λ>0 · 학습 ln m 평균 0): 어긋난 시즌 {bad if bad else '없음'}.")
        for k in ("S28", "S38"):
            if k in pack["targets"]:
                t = pack["targets"][k]
                L.append(f"{k} 일차 경계(h): {t['day_edges']} (시작 KST {t['c0h']}시, 구간 중앙의 일차 번호 {t['checks']['day_edges_check']})")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
