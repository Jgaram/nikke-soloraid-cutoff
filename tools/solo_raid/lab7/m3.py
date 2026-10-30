"""M3 함수형 주성분 곡면 (WP3b, 2026-10-29) — PACE(Yao·Müller·Wang 2005)를 베이즈 관측 층에 얹은 후보.

  python3 -I tools/solo_raid/lab7/m3.py selftest [--work W]
  python3 -I tools/solo_raid/lab7/m3.py stage1 --obs-dir W/obs7/full --seasons S19-S42 --out data/solo-raid/analysis/r7/stage1_M3_D1.json [--chains 4 --warmup 500 --samples 500] [--draws-dir DIR]
  python3 -I tools/solo_raid/lab7/m3.py prior --stage1 STAGE1.json --target S30 [--k 6] [--out P.json] [--prior-cache DIR]
  python3 -I tools/solo_raid/lab7/m3.py predict --stage1 STAGE1.json --obs-dir W/obs7 --seasons S22,S26 --T all|24,48 --out PRED.json [--prior-cache DIR]

명세 「WP3 명세」 0·1·2·3·4(M3)·5·6절. 좌표: y = ln 점수, w = ln(q/3), u = h/H.
  컷 곡선  ln C(w) = a - R(w),  R(w) = ∫_0^w exp(r(v)) dv      r = 3차 B-스플라인 12개, w ∈ [ln(0.1/3), ln(40/3)] (밖은 끝값 상수)
  침수     P(u) = ∫_u^1 exp(l(v)) dv  (log10 단위, P(1) = 0)   l = 3차 B-스플라인 12개, s = sqrt(u) 등간격 매듭
  서버 k   a_k = a + d_k, r_k = r + kappa_k (R_k = e^kappa R), l_k = l + psi_k (P_k = e^psi P). KR은 0.
  적분 격자: w 160점 [ln(0.005/3), ln(100/3)] (R(0) = 0으로 맞춤), sqrt(u) 200점. 역함수 w_fin(y)는 격자 안은 보간, 밖은 끝 기울기 직선.
곡면 함수 약속(obs_model): surface(theta, y[N], u[N,K]) -> mu[3,N,K], P[3,N,K].
  theta: a, cr[NR], cl[NL], dJP,dGL,pJP,pGL,kJP,kGL.   lnC(theta, q) = KR 최종 컷 ln 점수.
1단계 파일: {"header": {...}, "seasons": {시즌: {names, mean, cov, 진단...}}} — hier에는 seasons를 넘긴다 (load_stage1).
공개 규칙: 글 번호·원문·닉네임·원글 주소는 쓰지 않는다.
"""
import argparse
import datetime
import functools
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from scipy.interpolate import BSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import m1 as M1  # noqa: E402
import obs_model as OM  # noqa: E402
import hier as HR  # noqa: E402

LN10 = math.log(10.0)
L10_3 = math.log10(3.0)
OFF = M1.OFF                                            # dJP dGL pJP pGL kJP kGL
XTERM = True                                            # x항 β (M1 ②와 같다): P_k *= exp(beta w_fin_k / ln10)
NUIS = ["lnsig0", "lnrho"] + (["beta"] if XTERM else []) + list(OFF)   # prior_for에 넘기는 모수 (m1 순서)
NR, NL = 13, 12   # 2026-10-29 7·8 → 12·12 → r 13 (균일 12 + 3% 매듭, 관측만의 PSIS-LOO 규칙): 기저 7개로는 범위만 바꿔도 3% 컷이 0.57% 움직였다 (P-스플라인은 넉넉한 기저 + 벌점, Eilers·Marx 1996)
W_LO, W_HI = math.log(0.1 / 3), math.log(40 / 3)        # 평가 격자 w12 범위 (고정)
K_LO, K_HI = W_LO, W_HI                                 # r 매듭 범위 (configure로 바꿈)
K_EXTRA = (0.0,)                                        # r 안쪽에 더 끼우는 매듭 (w 값): w=0 = 3% (NR에 센다)
S_LO = math.sqrt(0.02)
CR_MEAN, CR_SD = math.log(0.3), 1.5
CL_MEAN, CL_SD = math.log(0.2 / 0.6), 2.0
TAU_SCALE = 0.5
CACHE = None                                            # 2단계 창 적합 디스크 캐시 폴더 (CLI --prior-cache)


# ---------- 기저·격자 (numpy) ----------

def _bspl_design(x, lo, hi, n, deg=3, extra=()):
    """clamped 균일 매듭 B-스플라인 n개의 설계 행렬 [len(x), n]. extra = 균일 매듭에 더 끼우는 안쪽 매듭 (실험용; n에 센다)."""
    ni = n - deg - 1 - len(extra)
    inner = np.sort(np.concatenate([np.linspace(lo, hi, ni + 2)[1:-1], np.asarray(extra, float)]))
    t = np.concatenate([[lo] * (deg + 1), inner, [hi] * (deg + 1)])
    x = np.clip(np.asarray(x, float), lo, hi)
    out = np.zeros((len(x), n))
    for j in range(n):
        c = np.zeros(n)
        c[j] = 1.0
        out[:, j] = BSpline(t, c, deg, extrapolate=False)(x)
    return np.nan_to_num(out)


def build_basis():
    wg = np.linspace(math.log(0.005 / 3), math.log(100 / 3), 160)
    sg = np.linspace(S_LO, 1.0, 200)
    w12 = np.unique(np.concatenate([np.linspace(W_LO, W_HI, 12), [x for x in K_EXTRA if W_LO <= x <= W_HI]]))   # 평가 격자: 균일 12 + 끼운 매듭 (정방 B)
    s12 = np.linspace(S_LO, 1.0, 12)
    i0 = int(np.clip(np.searchsorted(wg, 0.0) - 1, 0, len(wg) - 2))
    f0 = float((0.0 - wg[i0]) / (wg[i0 + 1] - wg[i0]))
    return {"wg": wg, "sg": sg, "w12": w12, "s12": s12, "u12": s12 ** 2,
            "Brf": _bspl_design(wg, K_LO, K_HI, NR, extra=K_EXTRA), "Blf": _bspl_design(sg, S_LO, 1.0, NL),
            "Br12": _bspl_design(w12, K_LO, K_HI, NR, extra=K_EXTRA), "Bl12": _bspl_design(s12, S_LO, 1.0, NL),
            "dw": float(wg[1] - wg[0]), "ds": float(sg[1] - sg[0]), "i0": i0, "f0": f0}


BASIS = build_basis()
NE_R, S1_NAMES, BAND, OUT, DT = 0, [], [], [], 0


def configure(nr=None, nl=None, r_lo_pct=None, r_hi_pct=None, r_extra=None, x_term=None):
    """기저 개수·r 매듭 범위(% 단위)를 바꾼다 (BASIS를 제자리에서 갱신 — 기본 인자로 묶인 곳도 따라온다). 실험용; 평가 격자 w12·u12는 그대로."""
    global NR, NL, K_LO, K_HI, K_EXTRA, XTERM
    if x_term is not None:
        XTERM = bool(x_term)
    if r_extra is not None:
        K_EXTRA = tuple(r_extra)
    NR = nr or NR
    NL = nl or NL
    if r_lo_pct is not None:
        K_LO = math.log(r_lo_pct / 3)
    if r_hi_pct is not None:
        K_HI = math.log(r_hi_pct / 3)
    BASIS.update(build_basis())
    _update_layout()


def _update_layout():
    """평가 격자 길이에서 파생되는 것: 1단계 벡터 이름, 24+ 곡선 벡터의 밴드 위치 (r: 0.3%<=q<=13.5%인 점, l: 인덱스 2..11)."""
    global NE_R, S1_NAMES, BAND, OUT, DT, NUIS
    NUIS = ["lnsig0", "lnrho"] + (["beta"] if XTERM else []) + list(OFF)
    w = BASIS["w12"]
    NE_R = len(w)
    S1_NAMES = ["a"] + list(NUIS) + [f"r{i}" for i in range(NE_R)] + [f"l{i}" for i in range(12)]
    br = [i for i, x in enumerate(w) if math.log(0.3 / 3) - 0.05 <= x <= math.log(13.5 / 3) + 0.05]
    BAND = br + [NE_R + i for i in range(2, 12)]
    DT = NE_R + 12
    OUT = [i for i in range(DT) if i not in BAND]


def n_stage1():
    return 1 + NR + NL + 2 + len(NUIS)


def basis_spec():
    return {"x_term": XTERM, "r": {"n": NR, "degree": 3, "range_w": [K_LO, K_HI], "extra_knots_w": list(K_EXTRA), "knots": "clamped, 균일", "outside": "끝값 상수"},
            "l": {"n": NL, "degree": 3, "range_s": [S_LO, 1.0], "knots": "clamped, 균일 (s = sqrt(u))"},
            "w_grid": {"n": 160, "lo": float(BASIS["wg"][0]), "hi": float(BASIS["wg"][-1])},
            "u_grid": {"n": 200, "s_lo": S_LO, "s_hi": 1.0},
            "w12": BASIS["w12"].tolist(), "w_eval_note": "r 평가 격자 = 균일 12 + 끼운 매듭, l은 균일 12", "u12": BASIS["u12"].tolist()}


_update_layout()


# ---------- 곡면 (jax) ----------

def R_grid(cr, B=BASIS):
    """격자 위 R(w) [160] (R(0) = 0). 격자 끝 기울기 exp(r) 둘도 돌려준다."""
    import jax.numpy as jnp
    e = jnp.exp(jnp.asarray(B["Brf"]) @ cr)
    Rc = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (e[1:] + e[:-1]) * B["dw"])])
    i0, f0 = B["i0"], B["f0"]
    R = Rc - ((1 - f0) * Rc[i0] + f0 * Rc[i0 + 1])
    return R, e[0], e[-1]


def P_grid(cl, B=BASIS):
    """sqrt(u) 격자 위 P [200] (끝에서 0)."""
    import jax.numpy as jnp
    sg = jnp.asarray(B["sg"])
    f = jnp.exp(jnp.asarray(B["Blf"]) @ cl) * 2.0 * sg
    seg = 0.5 * (f[1:] + f[:-1]) * B["ds"]
    return jnp.concatenate([jnp.cumsum(seg[::-1])[::-1], jnp.zeros(1)])


def w_of_R(t, R, e0, e1, B=BASIS):
    """R(w) = t 의 w. 격자 안은 보간, 밖은 끝 기울기 직선."""
    import jax.numpy as jnp
    wg = jnp.asarray(B["wg"])
    w_in = jnp.interp(t, R, wg)
    w_lo = wg[0] + (t - R[0]) / e0
    w_hi = wg[-1] + (t - R[-1]) / e1
    return jnp.where(t < R[0], w_lo, jnp.where(t > R[-1], w_hi, w_in))


def surface(theta, y, u, B=BASIS):
    """obs_model 곡면 함수 약속: y[N], u[N,K] -> (mu[3,N,K], P[3,N,K])."""
    import jax.numpy as jnp
    R, e0, e1 = R_grid(theta["cr"], B)
    Pg = P_grid(theta["cl"], B)
    P = jnp.interp(jnp.sqrt(u), jnp.asarray(B["sg"]), Pg)
    z = jnp.zeros(())
    dl = [z, theta["dJP"], theta["dGL"]]
    ps = [z, theta["pJP"], theta["pGL"]]
    kp = [z, theta["kJP"], theta["kGL"]]
    mus, Ps = [], []
    for k in range(3):
        t = (theta["a"] + dl[k] - y) * jnp.exp(-kp[k])
        wf = w_of_R(t, R, e0, e1, B)
        Pk = jnp.exp(ps[k]) * P
        if "beta" in theta:       # x항: 그 서버 성분의 최종 w에 따라 침수 크기가 변한다 (m1.surface와 같다)
            Pk = Pk * jnp.exp(theta["beta"] * wf / LN10)[:, None]
        mus.append((L10_3 + wf / LN10)[:, None] - Pk)
        Ps.append(Pk)
    return jnp.stack(mus), jnp.stack(Ps)


def lnC(theta, q, B=BASIS):
    """KR 최종 컷 ln 점수. q = 목표 % (숫자 또는 배열). 0.005~100% 밖은 직선 외삽."""
    import jax.numpy as jnp
    R, e0, e1 = R_grid(theta["cr"], B)
    w = jnp.log(jnp.asarray(q, jnp.float32) / 3.0)
    wg = jnp.asarray(B["wg"])
    Rq = jnp.where(w < wg[0], R[0] + e0 * (w - wg[0]), jnp.where(w > wg[-1], R[-1] + e1 * (w - wg[-1]), jnp.interp(w, wg, R)))
    return theta["a"] - Rq


def curves12(cr, cl, B=BASIS):
    import jax.numpy as jnp
    return jnp.asarray(B["Br12"]) @ cr, jnp.asarray(B["Bl12"]) @ cl


# ---------- 사전·모수 묶음 ----------

def _nlp(x, m, s):
    import jax.numpy as jnp
    return (-0.5 * ((x - m) / s) ** 2 - jnp.log(s) - 0.5 * math.log(2 * math.pi)).sum()


def _rw2(c, ltau):
    """계수 2차 차분 ~ N(0, tau^2)."""
    import jax.numpy as jnp
    d2 = c[2:] - 2 * c[1:-1] + c[:-2]
    tau = jnp.exp(ltau)
    return (-0.5 * (d2 / tau) ** 2 - ltau - 0.5 * math.log(2 * math.pi)).sum()


def _lntau_prior(ltau):
    """HalfNormal(0.5)의 ln tau 모수화 (+ln tau 야코비안)."""
    import jax.numpy as jnp
    tau = jnp.exp(ltau)
    return math.log(2.0) - 0.5 * math.log(2 * math.pi) - math.log(TAU_SCALE) - 0.5 * (tau / TAU_SCALE) ** 2 + ltau


class Stage1Prob:
    """끝난 시즌 1단계: v = [a, 곡선 계수 둘(7+8), ln tau_r, ln tau_l, lnsig0, lnrho, 6 offset] (26).
    param="centered": 계수 c 그대로. 사전 = 계수마다 넓은 정규 x RW2(tau). MAP에서 tau를 고정해야 한다 (결합 MAP이 tau -> 0으로 퇴화).
    param="noncentered" (기본): c = m + L(tau) zeta, zeta ~ N(0, I), L = U diag((e/tau^2 + 1/s^2)^-1/2) (Q = D2'D2/tau^2 + I/s^2의 고유분해).
      사전 c ~ N(m, Q^-1) — 깔때기(tau 대 끝 계수)를 없애려는 표준 비중심화 (Betancourt-Girolami 2015). 중심화판과 tau 둘레 정규화 상수가 다르다."""
    kind = "stage1"
    n = None

    def __init__(self, param="noncentered"):
        self.param = param
        self.n = n_stage1()
        self.it = 1 + NR + NL                     # ln tau_r 위치 (tau_l은 다음)
        self.off_ix = tuple(self.it + 2 + NUIS.index(n) for n in OFF)   # 서버 성분 모수 6개 위치
        self.tau_ix = (self.it, self.it + 1) if param == "centered" else ()
        self.start = np.concatenate([[M1.WEAK["a"][0]], np.full(NR, CR_MEAN if param == "centered" else 0.0),
                                     np.full(NL, CL_MEAN if param == "centered" else 0.0), [math.log(0.3)] * 2,
                                     [M1.WEAK[n][0] for n in NUIS]])
        self._eig = {}
        for key, nn in (("r", NR), ("l", NL)):
            D2 = np.diff(np.eye(nn), 2, axis=0)
            e, U = np.linalg.eigh(D2.T @ D2)
            self._eig[key] = (np.clip(e, 0, None), U)

    def _curve(self, z, ltau, key):
        import jax.numpy as jnp
        e, U = self._eig[key]
        m, sd = (CR_MEAN, CR_SD) if key == "r" else (CL_MEAN, CL_SD)
        q = jnp.asarray(e) / jnp.exp(ltau) ** 2 + 1.0 / sd ** 2
        return m + jnp.asarray(U) @ (z / jnp.sqrt(q))

    def theta(self, v):
        it = self.it
        cr, cl = v[1:1 + NR], v[1 + NR:1 + NR + NL]
        if self.param == "noncentered":
            cr, cl = self._curve(cr, v[it], "r"), self._curve(cl, v[it + 1], "l")
        d = {"a": v[0], "cr": cr, "cl": cl}
        d.update({n: v[it + 2 + i] for i, n in enumerate(NUIS)})
        return d

    def logprior(self, v):
        import jax.numpy as jnp
        th = self.theta(v)
        lp = _nlp(v[0], M1.WEAK["a"][0], M1.WEAK["a"][1])
        if self.param == "centered":
            lp += _nlp(th["cr"], CR_MEAN, CR_SD) + _rw2(th["cr"], v[self.it])
            lp += _nlp(th["cl"], CL_MEAN, CL_SD) + _rw2(th["cl"], v[self.it + 1])
        else:
            lp += _nlp(v[1:1 + NR + NL], 0.0, 1.0)
        lp += _lntau_prior(v[self.it]) + _lntau_prior(v[self.it + 1])
        mu = jnp.asarray([M1.WEAK[n][0] for n in NUIS])
        sd = jnp.asarray([M1.WEAK[n][1] for n in NUIS])
        return lp + _nlp(v[self.it + 2:], mu, sd)


class CellProb:
    """맞힐 시즌 칸: v = [a, xi_1..xi_K, lnsig0, lnrho, 6 offset]. 곡선 계수 = 평균 + sum xi_k phi_k."""
    kind = "cell"

    def __init__(self, pr, ppca=False):
        self.pr = pr
        self.K = int(pr["K"])
        self.ppca = ppca
        # ppca: 밴드 격자에 전계수 잔차 eps = sqrt(sigma2_res) eta (eta ~ N(0, I)) 를 더한다 (확률 PCA의 등방 잔차; 밴드 밖은 0).
        # 격자 편차 -> 계수는 정방 B의 정확한 해.
        self.ne = len(BAND) if ppca else 0
        if ppca:
            s2 = float(pr.get("sigma2_res") or 0.0)
            if s2 <= 1e-8:
                s2 = float(pr.get("disc_pos_mean") or 1e-4)
            self.s2res = s2
            Br, Bl = np.asarray(BASIS["Br12"]), np.asarray(BASIS["Bl12"])
            assert Br.shape[0] == Br.shape[1] and Bl.shape[0] == Bl.shape[1]
            Mfull = np.zeros((NR + NL, DT))
            Mfull[:NR, :NE_R] = np.linalg.inv(Br)
            Mfull[NR:, NE_R:] = np.linalg.inv(Bl)
            self.M = math.sqrt(s2) * Mfull[:, BAND]          # [NR+NL, ne]
        o = 1 + self.K + self.ne
        self.n = o + len(NUIS)
        self.off_ix = tuple(o + NUIS.index(n) for n in OFF)
        self.mc = np.asarray(pr["mean_coef"], float)
        self.phi = np.asarray(pr["phi_coef"], float).reshape(self.K, NR + NL)
        self.lam = np.asarray(pr["lambda_scaled"], float).reshape(self.K)
        nu = pr["nuisance"]
        assert nu["names"] == NUIS
        self.nmean = np.asarray(nu["mean"], float)
        cov = np.asarray(nu["cov"], float) + 1e-6 * np.eye(len(NUIS))
        self.nch = np.linalg.cholesky(cov)
        self.am, self.av = pr["a"]["mean"], pr["a"]["var"]
        self.start = np.concatenate([[self.am], np.zeros(self.K + self.ne), self.nmean])

    def theta(self, v):
        import jax.numpy as jnp
        K = self.K
        c = jnp.asarray(self.mc)
        if K:
            c = c + v[1:1 + K] @ jnp.asarray(self.phi)
        if self.ne:
            c = c + jnp.asarray(self.M) @ v[1 + K:1 + K + self.ne]
        o = 1 + K + self.ne
        d = {"a": v[0], "cr": c[:NR], "cl": c[NR:]}
        d.update({n: v[o + i] for i, n in enumerate(NUIS)})
        return d

    def logprior(self, v):
        import jax.numpy as jnp
        from jax.scipy.linalg import solve_triangular
        K = self.K
        lp = _nlp(v[0], self.am, math.sqrt(self.av))
        if K:
            lp += _nlp(v[1:1 + K], 0.0, jnp.sqrt(jnp.asarray(self.lam)))
        if self.ne:
            lp += _nlp(v[1 + K:1 + K + self.ne], 0.0, 1.0)
        r = v[1 + K + self.ne:] - jnp.asarray(self.nmean)
        ch = jnp.asarray(self.nch)
        zz = solve_triangular(ch, r, lower=True)
        return lp - 0.5 * (zz ** 2).sum() - jnp.log(jnp.diag(ch)).sum() - 0.5 * len(NUIS) * math.log(2 * math.pi)


# ---------- 모형·MAP ----------

def _eps_phi(e, obs):
    """e = [logit eps, stick-breaking 8개] -> (eps, phi[4,3], 사전 로그 밀도 + 야코비안). m1.model과 같은 꼴."""
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from numpyro.distributions.transforms import StickBreakingTransform
    sbt = StickBreakingTransform()
    eps_v = jax.nn.sigmoid(e[0])
    sb = e[1:].reshape(4, 2)
    phi_v = sbt(sb)
    a0, b0 = [jnp.asarray(x, e.dtype) for x in obs["eps_ab"]]
    lp = (dist.Beta(a0, b0).log_prob(eps_v) + jnp.log(eps_v) + jnp.log1p(-eps_v)
          + dist.Dirichlet(jnp.asarray(obs["dirichlet"], e.dtype)).log_prob(phi_v).sum() + sbt.log_abs_det_jacobian(sb, phi_v).sum())
    return eps_v, phi_v, lp


def model(obs, prob, pre):
    """numpyro 모형 (MAP+라플라스 아핀 재매개화 위 z 표집). 표집 이름 z·eps·phi, deterministic v·th24(1단계)·r12·l12·lnC_q."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import constraints
    nfull = len(pre["loc"])
    z = numpyro.sample("z", dist.ImproperUniform(constraints.real_vector, (), (nfull,)))
    vf = jnp.asarray(pre["loc"], jnp.float32) + jnp.asarray(pre["chol"], jnp.float32) @ z
    v = vf[:prob.n]
    lp = prob.logprior(v)
    eps_phi = {}
    if nfull > prob.n:   # eps(logit)·phi(stick-breaking 8개)도 펴서 푼다 (m1과 같은 방식, 야코비안 포함)
        eps_v, phi_v, lp_ep = _eps_phi(vf[prob.n:], obs)
        lp = lp + lp_ep
        numpyro.deterministic("eps", eps_v)
        numpyro.deterministic("phi", phi_v)
        eps_phi = {"eps": eps_v, "phi": phi_v}
    numpyro.factor("prior", lp)
    numpyro.deterministic("v", v)
    th = prob.theta(v)
    if prob.kind == "stage1":
        numpyro.deterministic("th24", jnp.concatenate([v[:prob.it], v[prob.it + 2:]]))
        r12, l12 = curves12(th["cr"], th["cl"])
        numpyro.deterministic("r12", r12)
        numpyro.deterministic("l12", l12)
    if prob.kind == "cell":   # 수렴 판정용: 핵심(a·ξ·lnsig0·lnrho)과 서버 성분 보조 모수를 따로 둔다
        K_ = prob.K
        numpyro.deterministic("core", jnp.concatenate([v[:1 + K_], v[1 + K_ + prob.ne:-len(OFF)]]))      # a, ξ, lnsig0, lnrho, (β); ppca의 η는 뺀다
        numpyro.deterministic("aux", v[-len(OFF):])
    mu, P = surface(th, obs["y"], obs["u"])
    OM.observe(obs, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]), **eps_phi)
    for q in OM.Q_TARGETS:
        numpyro.deterministic(f"lnC_{q:g}", lnC(th, q))


def map_precond(obs, prob, iters=1500, temps=(0.003, 0.03, 0.3, 1.0), full_ep=False, start=None):
    """MAP + 라플라스 (m1.map_precond 꼴, float64). eps·phi는 사전 평균에 고정. -> dict(loc, chol, info)
    1단계는 평활 ln tau를 고정하고 푼다 — 결합 최빈값은 tau -> 0 (곡선이 직선)으로 달아나 퇴화하기 때문 (2026-10-29 확인).
    tau = 0.3으로 풀고, 구한 곡선의 2차 차분 rms로 tau를 한 번 갱신해 다시 푼다. 라플라스 공분산은 나머지 모수만, tau 둘은 ln 척도 sd 0.7 독립으로 둔다."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    n = prob.n
    tau_ix = list(getattr(prob, "tau_ix", []))
    free = np.array([i for i in range(n) if i not in tau_ix])
    with jax.enable_x64(True):
        o = dict(obs)
        for k in OM.NUMERIC:
            a = np.asarray(obs[k])
            o[k] = jnp.asarray(a, jnp.int64 if a.dtype.kind in "iu" else jnp.float64)
        eps0 = obs["eps_ab"][0] / (obs["eps_ab"][0] + obs["eps_ab"][1])
        phi0 = o["dirichlet"] / o["dirichlet"].sum(1, keepdims=True)
        fixed = [jnp.asarray(np.asarray(prob.start, float))]

        def full(vf):
            return fixed[0].at[free].set(vf)

        nf = len(free)

        def negpost(vf, temp):
            v = full(vf[:nf])
            th = prob.theta(v)
            mu, P = surface(th, o["y"], o["u"])
            if full_ep:
                eps_v, phi_v, lp_ep = _eps_phi(vf[nf:], o)
            else:
                eps_v, phi_v, lp_ep = eps0, phi0, 0.0
            ll = OM.row_loglik(o, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]), eps_v, phi_v)
            return -(temp * ll.sum() + prob.logprior(v) + lp_ep)

        def solve(x, tl, nit):
            vg = jax.jit(jax.value_and_grad(negpost))
            for temp in tl:
                def f(xx, temp=temp):
                    v, g = vg(jnp.asarray(xx), temp)
                    v, g = float(v), np.asarray(g, float)
                    return (v, g) if np.isfinite(v) and np.isfinite(g).all() else (1e12, np.zeros_like(g))
                res = minimize(f, x, jac=True, method="L-BFGS-B", options={"maxiter": iters, "maxfun": 3 * iters})
                x, nit = res.x, nit + res.nit
            return x, nit, res

        st0 = np.asarray(prob.start if start is None else start[:n], float)
        x0 = st0[free]
        if full_ep:
            from numpyro.distributions.transforms import StickBreakingTransform
            if start is not None and len(start) > n:
                x0 = np.concatenate([x0, np.asarray(start[n:], float)])
            else:
                x0 = np.concatenate([x0, [float(np.log(eps0 / (1 - eps0)))], np.asarray(StickBreakingTransform().inv(phi0)).reshape(-1)])
        x, nit, res = solve(x0, temps, 0)
        if tau_ix:                                   # tau 갱신 후 다시 (앞 답에서 출발, 온도 1)
            v = np.array(full(jnp.asarray(x[:nf])), float)
            for ti, (c0, c1) in zip(tau_ix, ((1, 1 + NR), (1 + NR, 1 + NR + NL))):
                d2 = v[c0 + 2:c1] - 2 * v[c0 + 1:c1 - 1] + v[c0:c1 - 2]
                v[ti] = math.log(float(np.clip(np.sqrt(np.mean(d2 ** 2)), 0.05, 3.0)))
            fixed[0] = jnp.asarray(v)
            x, nit, res = solve(x, (1.0,), nit)
        hf = jax.jit(jax.hessian(negpost))
        H = np.asarray(hf(jnp.asarray(x), 1.0), float)
        n_escape = 0
        for _ in range(4):                           # 안장점이면 음의 고유방향으로 밀어 다시 푼다 (S19·S24에서 필요했다)
            w0, Q0 = np.linalg.eigh(0.5 * (H + H.T))
            if w0.min() > 1e-2:
                break
            n_escape += 1
            best = None
            for sg in (1.0, -1.0):
                x2, nit, r2 = solve(x + sg * 0.5 * Q0[:, 0], (1.0,), nit)
                if best is None or r2.fun < best[2].fun:
                    best = (x2, nit, r2)
            if best[2].fun < res.fun - 1e-6:
                x, nit, res = best
                H = np.asarray(hf(jnp.asarray(x), 1.0), float)
            else:
                break
        loc = np.asarray(full(jnp.asarray(x[:nf])), float)
        if full_ep:
            loc = np.concatenate([loc, x[nf:]])
    H = 0.5 * (H + H.T)
    w, Q = np.linalg.eigh(H)
    nn = n + (9 if full_ep else 0)
    fr = np.concatenate([free, np.arange(n, nn)]) if full_ep else free
    sq = np.zeros((nn, nn))
    sq[np.ix_(fr, fr)] = Q / np.sqrt(np.maximum(w, 1e-6))   # 공분산의 제곱근 (콜레스키가 아니어도 아핀 재매개화로 충분)
    for i in tau_ix:
        sq[i, i] = 0.7
    return {"loc": loc, "chol": sq, "info": {"fun": float(res.fun), "nit": nit, "ok": bool(res.success),
                                            "min_eig": float(w.min()), "max_eig": float(w.max()), "n_floored": int((w < 1e-6).sum()), "n_escape": n_escape,
                                            "tau_fixed": [math.exp(loc[i]) for i in tau_ix]}}


def multistart_map(obs, prob, full_ep=False, n=8, seed=0, scatter=0.5, tol=0.5):
    """시작점 n개의 MAP (m1.multistart_map과 같은 꼴): 첫째는 기본 시작, 나머지는 그 해에서 서버 성분 모수(dJP dGL pJP pGL kJP kGL)만
    N(0, scatter^2)로 흩어 시작한다. 최고 로그 사후(= fun 최소)를 고르고, 그 값에 tol 안으로 닿은 시작점이 2개 미만이면 n개를 더 한다 (한 번만).
    -> 가장 좋은 pre (info["multistart"]에 시작점 수·최고 로그 사후·닿은 수·기본 시작점과의 차)"""
    base = map_precond(obs, prob, full_ep=full_ep)
    out = [base]
    rng = np.random.default_rng(seed)

    def add(k):
        for _ in range(k):
            st = np.array(base["loc"], float)
            st[list(prob.off_ix)] += rng.normal(0, scatter, len(prob.off_ix))
            out.append(map_precond(obs, prob, full_ep=full_ep, start=st))

    add(n - 1)
    best = min(p["info"]["fun"] for p in out)
    hit = sum(p["info"]["fun"] <= best + tol for p in out)
    if hit < 2:
        add(n)
        best = min(p["info"]["fun"] for p in out)
        hit = sum(p["info"]["fun"] <= best + tol for p in out)
    pre = min(out, key=lambda p: p["info"]["fun"])
    pre["info"] = dict(pre["info"], multistart={"n_starts": len(out), "best_logpost": -best, "n_hit_best": int(hit),
                                                "gain_vs_single": base["info"]["fun"] - best,
                                                "funs": sorted(round(p["info"]["fun"], 2) for p in out)})
    return pre


def fit(rows, H, season, err, prob, delay_key, chains=4, warmup=500, samples=500, seed=0, stage1=False, K=16, target_accept=0.9, retry=True, full_ep=False, multistart=True):
    """관측 줄 -> MAP(시작점 8개) -> NUTS (retry False면 한 번만). (samples, info, diag, obs). info에 map·secs·n·N."""
    obs = OM.prepare(rows, H, season, err, K=K, delay_key=delay_key, stage1=stage1)
    jobs = OM.to_jax(obs)
    t0 = time.time()
    pre = multistart_map(jobs, prob, full_ep=full_ep) if multistart else map_precond(jobs, prob, full_ep=full_ep)
    t_map = time.time() - t0
    mdl = functools.partial(model, prob=prob, pre=pre)
    lnames = [f"lnC_{q:g}" for q in (1, 3, 10)]
    if prob.kind == "cell":   # 칸: 핵심 모수 + 모든 목표 lnC로 판정 (서버 성분은 따로 보고, 재시도도 핵심이 못 지날 때만)
        names = ["core"] + [f"lnC_{q:g}" for q in OM.Q_TARGETS]
    else:
        names = (["th24", "r12", "l12"] if stage1 else ["v"]) + lnames
    init = M1.map_init(len(pre["loc"])) if multistart else None      # 모든 사슬을 MAP(z = 0)에서 시작 (m1과 같다)
    if retry:
        sm, info, dg = OM.fit_with_retry(mdl, {"obs": jobs}, names=names, seed=seed, chains=chains, warmup=warmup, samples=samples,
                                         target_accept=target_accept, init=init)
    else:
        sm, info = OM.run_nuts(mdl, {"obs": jobs}, seed=seed, chains=chains, warmup=warmup, samples=samples, target_accept=target_accept, init=init)
        dg = OM.diagnose(sm, names, divergences=info["divergences"])
        info["retried"] = False
    if prob.kind == "cell":
        ax = OM.diagnose(sm, ["aux"], divergences=0)
        al = OM.diagnose(sm, ["v"] + (["eps"] if "eps" in sm else []), divergences=0)
        dg["aux"] = {"rhat": ax["rhat_max"], "ess": ax["ess_min"]}
        dg["all"] = {"rhat": max(al["rhat_max"], dg["rhat_max"]), "ess": min(al["ess_min"], dg["ess_min"])}
    info["map_secs"] = round(t_map, 1)
    info["map"] = pre["info"]
    info["pre"] = {"loc": pre["loc"], "chol": pre["chol"]}
    info["n"], info["N"], info["counts"] = obs["n"], obs["N"], obs["counts"]
    return sm, info, dg, obs


def laplace_fit(rows, H, season, err, prob, delay_key, ndraw=4000, seed=0, K=16, multistart=True):
    """1단계 라플라스 근사 (명세 6절 C4): 결합 MAP(float64, eps·phi는 사전 평균에 고정) + 역 Hessian 공분산(고윳값 바닥 1e-6) -> v 4000뽑기.
    NUTS 쪽과 같은 모양의 (sm, info, dg, obs)를 돌려준다 (사슬 1개)."""
    import jax
    obs = OM.prepare(rows, H, season, err, K=K, delay_key=delay_key, stage1=True)
    jobs = OM.to_jax(obs)
    t0 = time.time()
    pre = multistart_map(jobs, prob) if multistart else map_precond(jobs, prob)
    t_map = time.time() - t0
    rng = np.random.default_rng(seed)
    v = pre["loc"][None] + rng.standard_normal((ndraw, prob.n)) @ pre["chol"].T

    def one(vv):
        th = prob.theta(vv)
        r12, l12 = curves12(th["cr"], th["cl"])
        return r12, l12, jnp_stack([lnC(th, q) for q in OM.Q_TARGETS])

    import jax.numpy as jnp
    jnp_stack = jnp.stack
    r12, l12, lc = jax.jit(jax.vmap(one))(jnp.asarray(v, jnp.float32))
    sm = {"v": v[None], "r12": np.asarray(r12)[None], "l12": np.asarray(l12)[None],
          "eps": np.full((1, ndraw), jobs["eps_ab"][0] / (jobs["eps_ab"][0] + jobs["eps_ab"][1]))}
    for i, q in enumerate(OM.Q_TARGETS):
        sm[f"lnC_{q:g}"] = np.asarray(lc[:, i])[None]
    mi = pre["info"]
    secs = round(time.time() - t0, 1)
    info = {"secs": secs, "map_secs": round(t_map, 1), "divergences": 0, "retried": False, "tree_cap_frac": 0.0, "map": mi,
            "n": obs["n"], "N": obs["N"], "counts": obs["counts"], "pre": {"loc": pre["loc"], "chol": pre["chol"]}}
    dg = {"rhat_max": None, "ess_min": None, "conv": bool(mi["min_eig"] > 0 and mi["n_floored"] == 0)}
    return sm, info, dg, obs


# ---------- 1단계 ----------

def s1_X(sm):
    v = sm["v"].reshape(-1, sm["v"].shape[-1])
    it = v.shape[1] - len(NUIS) - 2
    return np.concatenate([v[:, [0]], v[:, it + 2:], sm["r12"].reshape(len(v), -1), sm["l12"].reshape(len(v), -1)], axis=1)

def stage1_entry(sm, info, dg, obs):
    v = sm["v"].reshape(-1, sm["v"].shape[-1])
    it = v.shape[1] - len(NUIS) - 2
    X = s1_X(sm)
    lc = {f"{q:g}": np.quantile(sm[f"lnC_{q:g}"].ravel(), [0.1, 0.5, 0.9]).tolist() for q in OM.Q_TARGETS}
    tau_rh = OM.diagnose({"lt": sm["v"][..., it:it + 2]}, ["lt"]) if sm["v"].shape[0] > 1 else {"rhat_max": None, "ess_min": None}
    return {"names": S1_NAMES, "mean": X.mean(0).tolist(), "cov": np.cov(X.T).tolist(),
            "rhat_max": dg["rhat_max"], "ess_min": dg["ess_min"], "divergences": info["divergences"], "conv": dg["conv"],
            "secs": info["secs"], "secs_total": round(info["secs"] + info.get("secs_first_try", 0.0) + info["map_secs"], 1),
            "map_secs": info["map_secs"], "retried": info["retried"], "tree_cap_frac": info["tree_cap_frac"],
            "map": info["map"], "multistart": info["map"].get("multistart"), "n": obs["n"], "N": obs["N"], "counts": obs["counts"],
            "ln_tau_r_mean": float(v[:, it].mean()), "ln_tau_l_mean": float(v[:, it + 1].mean()),
            "tau_rhat": tau_rh["rhat_max"], "tau_ess": tau_rh["ess_min"],
            "eps_mean": float(sm["eps"].mean()), "lnC_q10_50_90": lc}


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=_js)
    os.replace(tmp, path)


def _js(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_stage1(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    b = d.get("header", {}).get("basis")
    if b:   # 1단계 파일의 기저로 맞춘다 (prior·predict가 같은 곡면을 쓰게)
        configure(b["r"]["n"], b["l"]["n"], 3 * math.exp(b["r"]["range_w"][0]), 3 * math.exp(b["r"]["range_w"][1]),
                  r_extra=b["r"].get("extra_knots_w", []), x_term=b.get("x_term", False))
    return d["seasons"] if "seasons" in d else d


def cmd_stage1(a):
    import obs_table
    OM.setup_host(a.chains)
    err = OM.load_err()
    out = {"header": {}, "seasons": {}}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            out = json.load(f)
    out["header"] = {"about": "M3 1단계 — 끝난 시즌 전체 표, 약한 사전, P-스플라인 곡선 (정답 표 안 씀)", "method": "M3", "data_ver": C.DATA_VER,
                     "K": 16, "date": datetime.date.today().isoformat(), "names": S1_NAMES, "basis": basis_spec(),
                     "param": a.param, "method": a.method,
                     "cfg": ({"draws": 4000, "note": "라플라스: 결합 MAP + 역 Hessian, eps·phi는 사전 평균에 고정(MAP과 같음)"} if a.method == "laplace"
                             else {"chains": a.chains, "warmup": a.warmup, "samples": a.samples, "target_accept": a.target_accept, "retry": not a.no_retry})}
    for s in C.parse_seasons(a.seasons):
        if s in out["seasons"]:
            print(s, "건너뜀 (이미 있음)", flush=True)
            continue
        head, rows = obs_table.read_cell(os.path.join(a.obs_dir, f"{s}.jsonl.gz"))
        dk = s if s in err["1a_post_delay"]["expanding"] else None
        if a.method == "laplace":
            sm, info, dg, obs = laplace_fit(rows, head["H"], s, err, Stage1Prob(a.param), dk)
        else:
            sm, info, dg, obs = fit(rows, head["H"], s, err, Stage1Prob(a.param), dk, a.chains, a.warmup, a.samples, stage1=True,
                                    target_accept=a.target_accept, retry=not a.no_retry)
        out["seasons"][s] = stage1_entry(sm, info, dg, obs)
        out["seasons"][s]["param"] = a.param
        out["seasons"][s]["method"] = a.method
        if a.vec_dir:
            os.makedirs(a.vec_dir, exist_ok=True)
            np.savez_compressed(os.path.join(a.vec_dir, f"{s}_{a.method}.npz"), X=s1_X(sm).astype(np.float32),
                                lnC=np.stack([sm[f"lnC_{q:g}"].ravel() for q in OM.Q_TARGETS]).astype(np.float32))
        _atomic_json(a.out, out)
        if a.draws_dir:
            os.makedirs(a.draws_dir, exist_ok=True)
            np.savez_compressed(os.path.join(a.draws_dir, f"{s}.npz"), z=sm["z"], eps=sm["eps"], phi=sm["phi"],
                                loc=info["pre"]["loc"], chol=info["pre"]["chol"])
        e = out["seasons"][s]
        if a.method == "laplace":
            print(s, f"n={e['n']} secs={e['secs']} map_ok={e['map']['ok']} nit={e['map']['nit']} min_eig={e['map']['min_eig']:.3g} floored={e['map']['n_floored']} lnC3={e['lnC_q10_50_90']['3'][1]:.4f}", flush=True)
        else:
            print(s, f"n={e['n']} secs={e['secs']} rhat={e['rhat_max']:.3f} ess={e['ess_min']:.0f} div={e['divergences']} conv={e['conv']}", flush=True)


# ---------- 2단계 사전 (PCA / PACE) ----------

def _xv(e):
    ix = [e["names"].index(f"{p}{i}") for p, nn in (("r", NE_R), ("l", 12)) for i in range(nn)]
    return np.asarray(e["mean"], float)[ix], np.asarray(e["cov"], float)[np.ix_(ix, ix)]


# 가중 함수형 PCA (Ramsay-Silverman 8장, 0/1 가중): 주성분은 BAND(자료가 실제로 있는 구간)에서만 낸다.
# 끝(r12 0~1·10~11 = 0.1~0.3%·13.5~40%, l12 0~1 = u < 0.089)은 자료가 없어 사후 평균·분산이 사전에 끌려 PC를 지배했다 (2026-10-29).
# BAND·OUT는 _update_layout이 평가 격자에서 정한다 (r: 0.3~13.5%인 점들 — 끼운 3% 매듭 포함, l: u >= 0.089).


def pca_window(X, V):
    """창 안 곡선 벡터 X[J,24], 잡음 공분산 V[J,24,24] -> dict(mu[24], phi[K,24], lam(스케일 전), lam_scaled, K, sres, share(밴드), band_phi).
    밴드 PACE: S_band = 표본 공분산 - 평균 V_band, K = 양의 고윳값 합의 90%에 처음 닿는 수 (<= 3, <= J-2).
    밴드 밖 성분은 창 시즌 곡선(평균 뺀 것)을 밴드 점수 xi_jk에 회귀한 값 sum_j d_j,out xi_jk / sum_j xi_jk^2."""
    J = X.shape[0]
    D = len(BAND)
    mu = X.mean(0)
    Xb = X[:, BAND]
    Vb = V[:, BAND][:, :, BAND]
    S = np.cov(Xb.T, ddof=1).reshape(D, D) if J >= 2 else np.zeros((D, D))
    S = S - Vb.mean(0)
    w, Q = np.linalg.eigh(0.5 * (S + S.T))
    o = np.argsort(w)[::-1]
    w, Q = np.clip(w[o], 0, None), Q[:, o]
    tot = w.sum()
    K = int(np.argmax(np.cumsum(w) / tot >= 0.90)) + 1 if tot > 0 else 0
    K = min(K, 3, max(J - 2, 0))
    m = min(J - 1, D)
    disc = w[K:m] if m > K else np.array([])
    sres = float(disc.mean()) if len(disc) else 1e-4
    e = Q[:, :K].T                                   # [K, band]
    phi = np.zeros((K, DT))
    xi = (Xb - mu[BAND]) @ e.T                       # [J, K]
    for k in range(K):
        phi[k, BAND] = e[k]
        den = float((xi[:, k] ** 2).sum())
        phi[k, OUT] = ((X - mu)[:, OUT] * xi[:, [k]]).sum(0) / den if den > 0 else 0.0
    return {"mu": mu, "phi": phi, "band_phi": e, "lam": w[:K], "lam_scaled": w[:K] * (1 + 1 / J), "K": K, "sres": sres,
            "share": (w / tot).tolist() if tot > 0 else [0.0] * D, "J": J}


def _logn(x, mu, cov):
    ch = np.linalg.cholesky(cov)
    z = np.linalg.solve(ch, x - mu)
    return float(-0.5 * z @ z - np.log(np.diag(ch)).sum() - 0.5 * len(x) * math.log(2 * math.pi))


def choose_window_pca(stage1, target, ks=(4, 6, 10, "all")):
    """hier.choose_window와 같은 틀의 시간 순 교차검증, 밀도는 확률 PCA + V_j. -> (k, {k: 점수})"""
    before = [s for s in sorted(stage1, key=HR.snum) if HR.snum(s) < HR.snum(target)]
    evals = [s for i, s in enumerate(before) if i >= 3]
    if len(evals) < 3:
        return "all", {}
    xv = {s: _xv(stage1[s]) for s in before}
    scores = {}
    for k in ks:
        tot = 0.0
        for s2 in evals:
            prev = [s for s in before if HR.snum(s) < HR.snum(s2)]
            win = prev if k == "all" else prev[-int(k):]
            p = pca_window(np.array([xv[s][0] for s in win]), np.array([xv[s][1] for s in win]))
            e = p["band_phi"]
            cov = e.T @ np.diag(p["lam_scaled"]) @ e + (p["sres"] + 1e-6) * np.eye(len(BAND)) + xv[s2][1][np.ix_(BAND, BAND)]
            tot += _logn(xv[s2][0][BAND], p["mu"][BAND], cov)
        scores[str(k)] = tot
    best = max(scores, key=scores.get)
    return (best if best == "all" else int(best)), scores


def prior(stage1, target, k_fixed=None, cache=None):
    """맞힐 시즌 target의 JSON 사전 (앞 시즌만 쓴다)."""
    cache = cache if cache is not None else CACHE
    before = [s for s in sorted(stage1, key=HR.snum) if HR.snum(s) < HR.snum(target)]
    if not before:
        raise ValueError("앞 시즌이 없다")
    if k_fixed is None:
        k, scores = choose_window_pca(stage1, target)
    else:
        k, scores = k_fixed, {}
    win = before if k == "all" else before[-int(k):]
    xv = [_xv(stage1[s]) for s in win]
    p = pca_window(np.array([x for x, _ in xv]), np.array([v for _, v in xv]))
    Br, Bl = BASIS["Br12"], BASIS["Bl12"]
    mean_coef = np.concatenate([np.linalg.lstsq(Br, p["mu"][:NE_R], rcond=None)[0], np.linalg.lstsq(Bl, p["mu"][NE_R:], rcond=None)[0]])
    phi_coef = np.array([np.concatenate([np.linalg.lstsq(Br, f[:NE_R], rcond=None)[0], np.linalg.lstsq(Bl, f[NE_R:], rcond=None)[0]])
                         for f in p["phi"]]).reshape(p["K"], NR + NL)
    mu_loc, mu_scale, tau_scale = M1.hier_args(NUIS)
    pf = HR.prior_for({s: stage1[s] for s in stage1}, target, NUIS, tau_scale, mu_loc, mu_scale, cache=cache, k_fixed=k)
    return {"target": target, "k": k, "window": win, "J": len(win), "K": p["K"], "lambda": p["lam"], "lambda_scaled": p["lam_scaled"],
            "sigma2_res": p["sres"], "explained": p["share"][:6], "band": BAND,
            "pc_band_loadings": p["band_phi"][:2].round(3), "mean_coef": mean_coef, "phi_coef": phi_coef,
            "scores": scores, "nuisance": {"names": pf["names"], "mean": pf["mean"], "cov": pf["cov"]}, "a": pf["a"],
            "fit_info": pf["fit_info"]}


def cmd_prior(a):
    global CACHE
    CACHE = a.prior_cache
    OM.setup_host(4)
    s1 = load_stage1(a.stage1)
    for t in a.target.split(","):
        p = prior(s1, t, a.k, CACHE)
        if a.out and "," not in a.target:
            _atomic_json(a.out, p)
        print(json.dumps({k: p[k] for k in ("target", "k", "window", "K", "lambda", "explained", "scores", "a", "pc_band_loadings")},
                         default=_js, ensure_ascii=False), flush=True)


# ---------- 칸 예측 ----------

def cell_fit(rows, H, s, err, pr, chains=4, warmup=500, samples=500, seed=0, full_ep=False, retry=True, multistart=True, ppca=False):
    prob = CellProb(pr, ppca)
    sm, info, dg, obs = fit(rows, H, s, err, prob, s, chains, warmup, samples, seed, stage1=False, full_ep=full_ep, retry=retry, multistart=multistart)
    return sm, info, dg, obs


def _git_head():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def cmd_predict(a):
    global CACHE
    import obs_table
    CACHE = a.prior_cache
    OM.setup_host(max(a.chains, 4))   # hier 2단계 적합이 4사슬
    err = OM.load_err()
    s1 = load_stage1(a.stage1)
    with open(a.stage1, "rb") as f:
        sha = hashlib.sha1(f.read()).hexdigest()[:12]
    out = {"about": "M3 함수형 주성분 곡면 — T 절단 칸 예측", "method": "M3", "data_ver": C.DATA_VER,
           "made": {"tool_head": _git_head(), "cmd": " ".join(["m3.py predict"] + sys.argv[2:]), "inputs": {"stage1": sha}},
           "cells": [], "info": [], "priors": {}}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            out = json.load(f)
    done = {(i["season"], i["T"]) for i in out["info"]}
    for s in C.parse_seasons(a.seasons):
        Ts = C.t_grid(s) if a.T == "all" else [float(t) for t in a.T.split(",")]
        for T in Ts:
            if (s, T) in done:
                continue
            path = os.path.join(a.obs_dir, s, f"T{T:g}.jsonl.gz")
            if not os.path.exists(path):
                print(s, T, "표 없음", flush=True)
                continue
            if s not in out["priors"]:
                out["priors"][s] = prior(s1, s, cache=CACHE)
            pr = out["priors"][s]
            head, rows = obs_table.read_cell(path)
            sm, info, dg, obs = cell_fit(rows, head["H"], s, err, pr, a.chains, a.warmup, a.samples, full_ep=a.pre_full, multistart=not a.no_multistart, ppca=a.ppca)
            out["cells"] += OM.pred_cells(s, T, {q: sm[f"lnC_{q:g}"] for q in OM.Q_TARGETS}, dg["conv"])
            out["info"].append({"season": s, "T": T, "secs": info["secs"], "map_secs": info["map_secs"], "conv": dg["conv"], "K": pr["K"],
                                "k": pr["k"], "rhat_max": dg["all"]["rhat"], "ess_min": dg["all"]["ess"], "rhat_core": dg["rhat_max"], "ess_core": dg["ess_min"],
                                "rhat_aux": dg["aux"]["rhat"], "ess_aux": dg["aux"]["ess"], "divergences": info["divergences"],
                                "retried": info["retried"], "tree_cap_frac": info["tree_cap_frac"], "n": obs["n"], "N": obs["N"]})
            _atomic_json(a.out, out)
            i = out["info"][-1]
            print(s, T, f"n={i['n']} secs={i['secs']} rhat={i['rhat_max']:.3f} ess={i['ess_min']:.0f} div={i['divergences']} conv={i['conv']}", flush=True)


# ---------- 4.3 1단계 점검 (명세 7절 ③) ----------

ANSWER_KEY = os.path.join(C.ROOT, "data", "solo-raid", "analysis", "answer_key.json")


def _selected_rows(rows, H, err, dk, K=16):
    """prepare가 meta에 놓는 순서 그대로의 선택 줄 (pct·h_top7 등 meta에 없는 칸을 쓰려고)."""
    out = []
    for r in OM.select_rows(rows, H, True):
        t, w, why = OM.nodes_detail(r, H, err, dk, K)
        if t is not None:
            out.append(r)
    return out


def check43_season(rows, H, season, err, ndraw_pp=100, seed=0):
    import jax
    import jax.numpy as jnp
    dk = season if season in err["1a_post_delay"]["expanding"] else None
    prob = Stage1Prob("noncentered")
    sm, info, dg, obs = laplace_fit(rows, H, season, err, prob, dk)
    jobs = OM.to_jax(obs)
    sel = _selected_rows(rows, H, err, dk)
    meta = obs["meta"]
    assert len(sel) == len(meta)
    pos = np.asarray(obs["pos"])
    # 라플라스 뽑기에 eps·phi(사전 평균 고정)를 붙인다
    nd = sm["v"].shape[1]
    ea, eb = obs["eps_ab"]
    phi0 = np.asarray(obs["dirichlet"]) / np.asarray(obs["dirichlet"]).sum(1, keepdims=True)
    smx = dict(sm)
    smx["phi"] = np.broadcast_to(phi0, (1, nd) + phi0.shape).copy()

    def surf(draw, jb):
        th = prob.theta(draw["v"])
        mu, P = surface(th, jb["y"], jb["u"])
        return mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"])

    rp = OM.row_posteriors(jobs, {k: smx[k] for k in ("v", "eps", "phi")}, surf, max_draws=ndraw_pp)
    # 뽑기마다 첫 노드의 mu·잔차 sd
    idx = np.unique(np.linspace(0, nd - 1, ndraw_pp).astype(int))
    vv = smx["v"][0][idx]

    @jax.jit
    def one(v):
        mu, P, s0, rho = surf({"v": v}, jobs)
        return mu[:, :, 0], jnp.sqrt(s0 ** 2 + (rho * P[:, :, 0]) ** 2)

    MU, SG = [], []
    for v in vv:
        m_, s_ = one(jnp.asarray(v, jnp.float32))
        MU.append(np.asarray(m_))
        SG.append(np.asarray(s_))
    MU, SG = np.array(MU), np.array(SG)                       # [D,3,N]
    ps = rp["p_server"]                                       # [n,3] (meta 순서)
    n = len(meta)
    mu_row = np.einsum("dkn,nk->dn", MU[:, :, pos], ps)       # 줄마다 서버 가중 mu
    sg_row = np.einsum("dkn,nk->dn", SG[:, :, pos], ps)
    zlo, zhi = np.asarray(obs["zlo"])[pos], np.asarray(obs["zhi"])[pos]
    kind = np.array([m["kind"] for m in meta])
    tg = np.array([m["tg"] for m in meta])
    contam = rp["p_contam"] > 0.5
    out = {"n": n, "secs_laplace": info["secs"], "map_min_eig": info["map"]["min_eig"]}
    # A 잔차
    isA = (kind == "A") & (tg == "A")
    mb, sb = mu_row.mean(0), sg_row.mean(0)
    dist = np.maximum(np.maximum(zlo - mb, mb - zhi), 0.0)

    def resid(mask):
        k = int(mask.sum())
        return {"n": k, "inside": float((dist[mask] == 0).mean()) if k else None,
                "out_gt2sigma": float((dist[mask] > 2 * sb[mask]).mean()) if k else None}
    out["A_resid"] = {"all": resid(isA), "without_contam": resid(isA & ~contam)}
    out["contam"] = {"n_flag": int(contam.sum()), "mean_p": float(rp["p_contam"].mean())}
    # 서버 미상 줄의 P(KR)
    unk = np.array([m["server"] is None for m in meta])
    pk = ps[unk, 0]
    out["unknown_server"] = {"n": int(unk.sum()), "p_kr_quantiles": np.quantile(pk, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist() if len(pk) else None,
                             "share_gt0.9": float((pk > 0.9).mean()) if len(pk) else None, "share_lt0.1": float((pk < 0.1).mean()) if len(pk) else None,
                             "share_between": float(((pk >= 0.1) & (pk <= 0.9)).mean()) if len(pk) else None}
    # C 줄 촬영 시각
    isC = (kind == "C")
    hp = np.array([m["hp"] if m["hp"] is not None else np.nan for m in meta])
    dmin = (hp - rp["t_mean"]) * 60
    cm = isC & np.isfinite(dmin)
    t7lo = np.array([r.get("h_lo_top7") if r.get("h_lo_top7") is not None else np.nan for r in sel])
    t7hi = np.array([r.get("h_hi_top7") if r.get("h_hi_top7") is not None else np.nan for r in sel])
    has7 = cm & np.isfinite(t7lo) & np.isfinite(t7hi)
    outside = has7 & ((rp["t_mean"] < t7lo - 1e-9) | (rp["t_mean"] > t7hi + 1e-9))
    out["C_time"] = {"n": int(cm.sum()), "hp_minus_t_min_median": float(np.median(dmin[cm])) if cm.any() else None,
                     "hp_minus_t_min_p90": float(np.quantile(dmin[cm], 0.9)) if cm.any() else None,
                     "n_with_top7": int(has7.sum()), "share_outside_top7": float(outside.sum() / has7.sum()) if has7.any() else None}
    # PPC: 같은 사람 A 줄 연속 짝의 % 악화 몫
    rng = np.random.default_rng(seed)
    pid = np.array([m["pid"] for m in meta])
    ht = np.array([m["h"] if m["h"] is not None else np.nan for m in meta])
    pr_real = np.array([r["pct"] for r in sel], float)
    pairs = []
    for p_ in np.unique(pid[isA]):
        ii = np.where(isA & (pid == p_))[0]
        ii = ii[np.argsort(ht[ii])]
        pairs += list(zip(ii[:-1], ii[1:]))
    pairs = [(i, j) for i, j in pairs if ht[j] > ht[i]]
    if pairs:
        I, J = np.array(pairs).T
        real = float((pr_real[J] > pr_real[I]).mean())
        sims = []
        cum = np.cumsum(ps, 1)
        for d in range(len(vv)):
            kk = (rng.random(n)[:, None] > cum).sum(1).clip(0, 2)
            z = MU[d, kk, pos] + SG[d, kk, pos] * rng.standard_normal(n)
            pct = np.round(10 ** z, 2)
            sims.append(float((pct[J] > pct[I]).mean()))
        out["ppc_pairs"] = {"n_pairs": len(pairs), "real_worsen_share": real, "sim_mean": float(np.mean(sims)),
                            "sim_p05_p95": np.quantile(sims, [0.05, 0.95]).round(4).tolist()}
    else:
        out["ppc_pairs"] = {"n_pairs": 0}
    # 정답 곡선과 견줌 (보고만)
    qs = {f"{q:g}": sm[f"lnC_{q:g}"].ravel() for q in OM.Q_TARGETS}
    out["lnC"] = {q: np.quantile(d, [0.025, 0.1, 0.5, 0.9, 0.975]).tolist() for q, d in qs.items()}
    flg = OM.flags(meta, rp, 0.5)
    return out, flg


def cmd_check43(a):
    import obs_table
    OM.setup_host(1)
    err = OM.load_err()
    with open(ANSWER_KEY, encoding="utf-8") as f:
        ak = json.load(f)
    truth = {}
    for r in ak["rows"]:
        if r["server"] == "KR" and r.get("truth") is not None:
            truth[(r["season"], r["pct"])] = (r["truth"], r.get("shaky"))
    res, flags = {}, {}
    for s in C.parse_seasons(a.seasons):
        head, rows = obs_table.read_cell(os.path.join(a.obs_dir, f"{s}.jsonl.gz"))
        o, flg = check43_season(rows, head["H"], s, err)
        tr = {}
        for q, qv in o["lnC"].items():
            t = truth.get((s, float(q)))
            if t is None:
                continue
            lt = math.log(t[0])
            lo95, lo80, med, hi80, hi95 = qv
            tr[q] = {"truth": t[0], "shaky": t[1], "pred": math.exp(med), "abs_e": abs(math.exp(med) / t[0] - 1),
                     "in80": bool(lo80 <= lt <= hi80), "in95": bool(lo95 <= lt <= hi95)}
        o["answer"] = tr
        res[s] = o
        flags[s] = flg
        print(s, "n", o["n"], "A inside", o["A_resid"]["all"]["inside"], "flag", o["contam"]["n_flag"], flush=True)
    meta_ = {"about": "M3 1단계 점검 (4.3) — 라플라스 사후, eps·phi 사전 평균 고정, 정답 곡선은 견주기만", "method": "M3", "data_ver": C.DATA_VER,
             "date": datetime.date.today().isoformat(), "stage1": "라플라스 재적합 (stage1_M3_D1과 같은 코드)"}
    _atomic_json(a.out, {"header": meta_, "seasons": res})
    if a.flags:
        _atomic_json(a.flags, {"header": {"about": "사후 오염 확률 > 0.5 줄 (key·pid만)", "method": "M3", "data_ver": C.DATA_VER}, "seasons": flags})
    md = check43_md(meta_, res)
    with open(os.path.splitext(a.out)[0] + ".md", "w", encoding="utf-8") as f:
        f.write(md)


def check43_md(meta_, res):
    L = ["# M3 1단계 점검 (4.3, 조정 시즌)", "",
         "라플라스 사후(결합 MAP + 역 Hessian), eps·phi는 사전 평균에 고정. 정답 곡선은 견주기만 하고 적합에는 안 썼다. 생성: `m3.py check43`.", ""]
    ss = list(res)
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    L += ["## A 줄 잔차 (사후 평균 mu 대 % 구간)", "", "| 시즌 | 줄 | A 줄 | 구간 안 | 2σ 밖 | 오염 뺀 구간 안 | 오염 뺀 2σ 밖 |", "|---|---|---|---|---|---|---|"]
    for s in ss:
        o = res[s]; A, B = o["A_resid"]["all"], o["A_resid"]["without_contam"]
        L.append(f"| {s} | {o['n']} | {A['n']} | {f(A['inside'])} | {f(A['out_gt2sigma'])} | {f(B['inside'])} | {f(B['out_gt2sigma'])} |")
    L += ["", "## 오염 줄과 서버 미상 줄의 P(KR)", "", "| 시즌 | 오염 > 0.5 | 미상 줄 | P(KR) 사분위(10·25·50·75·90%) | > 0.9 | < 0.1 | 사이 |", "|---|---|---|---|---|---|---|"]
    for s in ss:
        o = res[s]; u = o["unknown_server"]
        q = "-" if u["p_kr_quantiles"] is None else " / ".join(f"{x:.2f}" for x in u["p_kr_quantiles"])
        L.append(f"| {s} | {o['contam']['n_flag']} | {u['n']} | {q} | {f(u['share_gt0.9'], 2)} | {f(u['share_lt0.1'], 2)} | {f(u['share_between'], 2)} |")
    L += ["", "## C 줄 촬영 시각 (게시 시각 − 사후 평균 촬영 시각, 분)", "", "| 시즌 | C 줄 | 중앙 | 90% | top7 있는 줄 | top7 밖 몫 |", "|---|---|---|---|---|---|"]
    for s in ss:
        c = res[s]["C_time"]
        L.append(f"| {s} | {c['n']} | {f(c['hp_minus_t_min_median'], 1)} | {f(c['hp_minus_t_min_p90'], 1)} | {c['n_with_top7']} | {f(c['share_outside_top7'], 3)} |")
    L += ["", "## PPC (같은 사람 A 줄 연속 짝에서 % 가 나빠지는 몫)", "", "| 시즌 | 짝 | 실제 | 모의 평균 | 모의 5~95% |", "|---|---|---|---|---|"]
    for s in ss:
        p = res[s]["ppc_pairs"]
        if p["n_pairs"]:
            L.append(f"| {s} | {p['n_pairs']} | {f(p['real_worsen_share'])} | {f(p['sim_mean'])} | {p['sim_p05_p95'][0]:.3f}~{p['sim_p05_p95'][1]:.3f} |")
        else:
            L.append(f"| {s} | 0 | - | - | - |")
    L += ["", "## 정답 곡선과 견줌 (한섭, 사후 중앙의 상대 오차 |e|, 정답이 80%·95% 구간 안인지)", "",
          "| 시즌 | " + " | ".join(f"{q}%" for q in ("0.5", "1", "2", "3", "5", "10")) + " |", "|---|" + "---|" * 6]
    for s in ss:
        cells = []
        for q in ("0.5", "1", "2", "3", "5", "10"):
            t = res[s]["answer"].get(q)
            cells.append("-" if t is None else f"{100 * t['abs_e']:.1f}% {'●' if t['in80'] else ('○' if t['in95'] else '×')}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "● 80% 안 · ○ 95% 안(80% 밖) · × 95% 밖.", ""]
    return "\n".join(L)


# ---------- 자체 점검 ----------

def selftest_numeric():
    import jax
    import jax.numpy as jnp
    from scipy.integrate import quad
    out = {}
    B = BASIS
    sums = [float(np.abs(B[k].sum(1) - 1).max()) for k in ("Brf", "Blf", "Br12", "Bl12")]
    out["basis_partition_of_unity_maxerr"] = sums
    cr = jnp.full(NR, math.log(0.3), jnp.float32)
    cl = jnp.full(NL, math.log(0.4), jnp.float32)
    # R 격자 = 0.3 w
    R, e0, e1 = R_grid(cr)
    out["R_const_maxerr"] = float(jnp.abs(R - 0.3 * jnp.asarray(B["wg"])).max())
    out["R0_zero_at_w0"] = float(jnp.interp(0.0, jnp.asarray(B["wg"]), R))
    # 역함수 왕복: y -> w -> y (격자 안·밖)
    th = {"a": jnp.float32(4.6), "cr": cr, "cl": cl, "dJP": jnp.float32(-0.2), "dGL": jnp.float32(-0.2), "pJP": jnp.float32(0.1),
          "pGL": jnp.float32(0.1), "kJP": jnp.float32(0.1), "kGL": jnp.float32(-0.1)}
    y = jnp.asarray(np.linspace(0.0, 9.0, 61), jnp.float32)          # w가 -17 ~ +15 정도: 격자 밖 양쪽 포함
    u = jnp.asarray(np.tile(np.linspace(0.02, 1.0, 5), (61, 1)), jnp.float32)
    mu, P = surface(th, y, u)
    zf_exact = lambda a_k, kp: L10_3 + ((a_k - np.asarray(y)) / (0.3 * math.exp(kp))) / LN10
    err_rt = []
    for k, (ak, kp) in enumerate([(4.6, 0.0), (4.4, 0.1), (4.4, -0.1)]):
        zf = np.asarray(mu[k, :, -1])                                 # u = 1 이라 P = 0
        err_rt.append(float(np.abs(zf - zf_exact(ak, kp)).max()))
    out["wfin_vs_exact_maxerr_z_by_server"] = err_rt
    wchk = (np.asarray(mu[0, :, -1]) - L10_3) * LN10
    out["roundtrip_y_maxerr"] = float(np.abs(4.6 - 0.3 * wchk - np.asarray(y)).max())
    out["w_range_tested"] = [float(wchk.min()), float(wchk.max())]
    # P 닫힌 꼴
    uu = np.linspace(0.02, 1.0, 9)
    _, Pc = surface(th, jnp.asarray([4.0], jnp.float32), jnp.asarray(uu[None], jnp.float32))
    out["P_const_maxerr"] = float(np.abs(np.asarray(Pc[0, 0]) - 0.4 * (1 - uu)).max())
    # 단조성
    mu_y = np.diff(np.asarray(mu[0]), axis=0)
    mu_u = np.diff(np.asarray(mu[0]), axis=1)
    out["mu_monotone_in_y"] = bool((mu_y < 0).all())   # 점수가 높을수록 % 숫자가 작다
    out["mu_monotone_in_u"] = bool((mu_u >= -1e-6).all())
    # 기울기 유한
    rng = np.random.default_rng(0)
    th2 = dict(th, cr=jnp.asarray(rng.normal(math.log(0.3), 0.3, NR), jnp.float32), cl=jnp.asarray(rng.normal(-1, 0.3, NL), jnp.float32))
    g = jax.grad(lambda t: surface(t, y, u)[0].sum())(th2)
    out["grad_finite"] = bool(all(np.isfinite(np.asarray(v)).all() for v in g.values()))
    g2 = jax.grad(lambda t: lnC(t, jnp.asarray([0.5, 1, 3, 10.0])).sum())(th2)
    out["grad_lnC_finite"] = bool(all(np.isfinite(np.asarray(v)).all() for v in g2.values()))
    # 임의 곡선: scipy 적분과 비교
    cr_n = np.asarray(th2["cr"], float)
    f_r = lambda w: math.exp(float(np.clip(1, 0, 1) * (_bspl_design([w], K_LO, K_HI, NR) @ cr_n)[0]))
    errs = []
    Rg = np.asarray(R_grid(th2["cr"])[0])
    for w in (-5.0, -2.0, -0.5, 0.5, 1.5, 3.0):
        ex = quad(f_r, 0, w)[0]
        errs.append(abs(float(np.interp(w, B["wg"], Rg)) - ex))
    out["R_random_vs_quad_maxerr"] = max(errs)
    cl_n = np.asarray(th2["cl"], float)
    Pg = np.asarray(P_grid(th2["cl"]))
    f_l = lambda v: math.exp(float((_bspl_design([math.sqrt(v)], S_LO, 1.0, NL) @ cl_n)[0]))
    errs = [abs(float(np.interp(math.sqrt(uv), B["sg"], Pg)) - quad(f_l, uv, 1.0)[0]) for uv in (0.03, 0.1, 0.3, 0.7)]
    out["P_random_vs_quad_maxerr"] = max(errs)
    out["pass"] = bool(max(sums) < 1e-9 and out["R_const_maxerr"] < 1e-4 and max(err_rt) < 2e-3 and out["roundtrip_y_maxerr"] < 2e-3
                       and out["P_const_maxerr"] < 1e-5 and out["mu_monotone_in_y"] and out["mu_monotone_in_u"]
                       and out["grad_finite"] and out["grad_lnC_finite"] and out["R_random_vs_quad_maxerr"] < 5e-3
                       and out["P_random_vs_quad_maxerr"] < 5e-3)
    return out


def selftest_recovery(work, season="S30", chains=2, warmup=300, samples=300, param="noncentered"):
    import jax.numpy as jnp
    import obs_table
    err = OM.load_err()
    head, rows = obs_table.read_cell(os.path.join(work, "obs7", "full", f"{season}.jsonl.gz"))
    H = head["H"]
    dk = season if season in err["1a_post_delay"]["expanding"] else None
    obs0 = OM.to_jax(OM.prepare(rows, H, season, err, K=16, delay_key=dk, stage1=True))
    prob = Stage1Prob(param)
    t0 = time.time()
    pre = map_precond(obs0, prob)
    t_map = time.time() - t0
    vt = pre["loc"].copy()
    th = prob.theta(jnp.asarray(vt, jnp.float32))
    rng = np.random.default_rng(7)
    srows = OM.select_rows(rows, H, stage1=True)
    new, nsrv = [], np.zeros(3, int)
    for r in srows:
        t, w, why = OM.nodes_detail(r, H, err, dk, 16)
        if t is None:
            continue
        tt = float(rng.choice(t, p=w / w.sum()))
        k = OM.SERVERS.index(r["server"]) if r["server"] in OM.SERVERS else 0
        u = float(np.clip(tt / H, 0.02, 1.0))
        mu, P = surface(th, jnp.asarray([math.log(r["score"])], jnp.float32), jnp.asarray([[u]], jnp.float32))
        sd = math.sqrt(math.exp(vt[prob.it + 2]) ** 2 + (math.exp(vt[prob.it + 3]) * float(P[k, 0, 0])) ** 2)
        z = float(mu[k, 0, 0]) + sd * rng.standard_normal()
        p = round(10 ** z, 2)
        if not (0.01 <= p <= 100.0):
            continue
        x = dict(r)
        x["pct"] = p
        new.append(x)
        nsrv[k] += 1
    sm, info, dg, obs = fit(new, H, season, err, prob, dk, chains, warmup, samples, stage1=True)
    tr = {}
    for q in OM.Q_TARGETS:
        tr[q] = float(lnC(th, q))
    rowsq = {}
    ins = 0
    for q in OM.Q_TARGETS:
        d = sm[f"lnC_{q:g}"].ravel()
        lo, med, hi = np.quantile(d, [0.05, 0.5, 0.95])
        ok = bool(lo <= tr[q] <= hi)
        ins += ok
        rowsq[f"{q:g}"] = {"truth": tr[q], "median": float(med), "lo90": float(lo), "hi90": float(hi), "inside90": ok,
                           "rel_err_C": float(math.exp(med - tr[q]) - 1)}
    r12t, l12t = curves12(th["cr"], th["cl"])
    v = sm["v"].reshape(-1, prob.n)
    nm = ["a", "lnsig0", "lnrho"] + list(OFF)
    ix = [0, prob.it + 2, prob.it + 3] + list(range(prob.it + 4, prob.n))
    par = {}
    for n_, i in zip(nm, ix):
        lo, hi = np.quantile(v[:, i], [0.05, 0.95])
        par[n_] = {"truth": float(vt[i]), "mean": float(v[:, i].mean()), "inside90": bool(lo <= vt[i] <= hi)}
    for nme, t_ in (("r12", r12t), ("l12", l12t)):
        lo, hi = np.quantile(sm[nme].reshape(-1, sm[nme].shape[-1]), [0.05, 0.95], axis=0)
        par[nme + "_inside90"] = int(((lo <= np.asarray(t_)) & (np.asarray(t_) <= hi)).sum())
    return {"rows": len(new), "per_server": nsrv.tolist(), "lnC": rowsq, "lnC_inside90": f"{ins}/{len(OM.Q_TARGETS)}", "params": par,
            "rhat_max": dg["rhat_max"], "ess_min": dg["ess_min"], "divergences": info["divergences"], "secs": info["secs"],
            "map_secs_truth": round(t_map, 1), "map_secs_fit": info["map_secs"], "tree_cap_frac": info["tree_cap_frac"],
            "tau_true": [math.exp(vt[prob.it]), math.exp(vt[prob.it + 1])], "tau_post_mean": [float(np.exp(v[:, prob.it]).mean()), float(np.exp(v[:, prob.it + 1]).mean())]}


def cmd_selftest(a):
    OM.setup_host(a.chains)
    res = {"numeric": selftest_numeric()}
    print(json.dumps(res["numeric"], indent=1), flush=True)
    if not a.skip_recovery:
        res["recovery"] = selftest_recovery(a.work, chains=a.chains, param=a.param)
        print(json.dumps(res["recovery"], indent=1, default=_js), flush=True)
    _atomic_json(os.path.join(a.work, "m3_selftest.json"), res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, chains=True):
        if chains:
            p.add_argument("--chains", type=int, default=4)
            p.add_argument("--warmup", type=int, default=500)
            p.add_argument("--samples", type=int, default=500)

    t = sub.add_parser("selftest")
    common(t)
    t.add_argument("--work", required=True)
    t.add_argument("--skip-recovery", action="store_true")
    t.add_argument("--param", choices=("centered", "noncentered"), default="noncentered")
    s = sub.add_parser("stage1")
    common(s)
    s.add_argument("--obs-dir", required=True)
    s.add_argument("--seasons", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--draws-dir", default=None)
    s.add_argument("--n-basis-r", type=int, default=None, help="r 기저 총수 (기본 12 + 끼운 매듭 수)")
    s.add_argument("--r-extra-knots", default="0", help="r에 끼우는 안쪽 매듭 w 값(쉼표). 기본 0 = 3%%, 빈 문자열 = 없음")
    s.add_argument("--n-basis-l", type=int, default=12)
    s.add_argument("--r-range-pct", default="0.1,40", help="r 매듭 범위 하한,상한 (%)")
    s.add_argument("--no-x-term", action="store_true", help="x항 β를 뺀다 (기본: 넣음)")
    s.add_argument("--method", choices=("laplace", "nuts"), default="laplace")
    s.add_argument("--no-retry", action="store_true")
    s.add_argument("--vec-dir", default=None, help="33벡터 뽑기 npz (X, lnC) 저장 폴더")
    s.add_argument("--target-accept", type=float, default=0.9)
    s.add_argument("--param", choices=("centered", "noncentered"), default="noncentered")
    p = sub.add_parser("prior")
    p.add_argument("--stage1", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--k", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--prior-cache", default=None)
    c4 = sub.add_parser("check43")
    c4.add_argument("--obs-dir", required=True)
    c4.add_argument("--seasons", required=True)
    c4.add_argument("--out", required=True)
    c4.add_argument("--flags", default=None)
    q = sub.add_parser("predict")
    common(q)
    q.add_argument("--stage1", required=True)
    q.add_argument("--obs-dir", required=True)
    q.add_argument("--seasons", required=True)
    q.add_argument("--T", default="all")
    q.add_argument("--out", required=True)
    q.add_argument("--prior-cache", default=None)
    q.add_argument("--ppca", action="store_true", help="확률 PCA 잔차: 밴드 격자에 등방 잔차 sqrt(sigma2_res) η를 더한다 (주성분이 못 담는 시즌 사이 변동)")
    q.add_argument("--no-multistart", action="store_true", help="MAP 시작점 8개(기본)를 끈다")
    q.add_argument("--pre-full", action="store_true", help="eps·phi도 사전 펴기에 넣는다 (m1과 같은 방식)")
    a = ap.parse_args()
    if a.cmd == "stage1":
        lo, hi = (float(x) for x in a.r_range_pct.split(","))
        ex = [float(x) for x in a.r_extra_knots.split(',') if x.strip() != '']
        configure(a.n_basis_r or 12 + len(ex), a.n_basis_l, lo, hi, r_extra=ex, x_term=not a.no_x_term)
    if getattr(a, "k", None) not in (None, "all"):
        a.k = int(a.k)
    {"selftest": cmd_selftest, "stage1": cmd_stage1, "prior": cmd_prior, "predict": cmd_predict, "check43": cmd_check43}[a.cmd](a)


if __name__ == "__main__":
    main()
