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
NUIS = ["lnsig0", "lnrho"] + list(OFF)                  # prior_for에 넘기는 8개
NR, NL = 12, 12   # 2026-10-29 7·8 → 12·12: 기저 7개로는 범위만 바꿔도 3% 컷이 0.57% 움직였다 (P-스플라인은 넉넉한 기저 + 벌점, Eilers·Marx 1996)
W_LO, W_HI = math.log(0.1 / 3), math.log(40 / 3)        # 평가 격자 w12 범위 (고정)
K_LO, K_HI = W_LO, W_HI                                 # r 매듭 범위 (configure로 바꿈)
S_LO = math.sqrt(0.02)
CR_MEAN, CR_SD = math.log(0.3), 1.5
CL_MEAN, CL_SD = math.log(0.2 / 0.6), 2.0
TAU_SCALE = 0.5
CACHE = None                                            # 2단계 창 적합 디스크 캐시 폴더 (CLI --prior-cache)
S1_NAMES = ["a", "lnsig0", "lnrho"] + list(OFF) + [f"r{i}" for i in range(12)] + [f"l{i}" for i in range(12)]


# ---------- 기저·격자 (numpy) ----------

def _bspl_design(x, lo, hi, n, deg=3):
    """clamped 균일 매듭 B-스플라인 n개의 설계 행렬 [len(x), n]."""
    ni = n - deg - 1
    t = np.concatenate([[lo] * (deg + 1), np.linspace(lo, hi, ni + 2)[1:-1], [hi] * (deg + 1)])
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
    w12 = np.linspace(W_LO, W_HI, 12)
    s12 = np.linspace(S_LO, 1.0, 12)
    i0 = int(np.clip(np.searchsorted(wg, 0.0) - 1, 0, len(wg) - 2))
    f0 = float((0.0 - wg[i0]) / (wg[i0 + 1] - wg[i0]))
    return {"wg": wg, "sg": sg, "w12": w12, "s12": s12, "u12": s12 ** 2,
            "Brf": _bspl_design(wg, K_LO, K_HI, NR), "Blf": _bspl_design(sg, S_LO, 1.0, NL),
            "Br12": _bspl_design(w12, K_LO, K_HI, NR), "Bl12": _bspl_design(s12, S_LO, 1.0, NL),
            "dw": float(wg[1] - wg[0]), "ds": float(sg[1] - sg[0]), "i0": i0, "f0": f0}


BASIS = build_basis()


def configure(nr=None, nl=None, r_lo_pct=None, r_hi_pct=None):
    """기저 개수·r 매듭 범위(% 단위)를 바꾼다 (BASIS를 제자리에서 갱신 — 기본 인자로 묶인 곳도 따라온다). 실험용; 평가 격자 w12·u12는 그대로."""
    global NR, NL, K_LO, K_HI
    NR = nr or NR
    NL = nl or NL
    if r_lo_pct is not None:
        K_LO = math.log(r_lo_pct / 3)
    if r_hi_pct is not None:
        K_HI = math.log(r_hi_pct / 3)
    BASIS.update(build_basis())


def n_stage1():
    return 1 + NR + NL + 10


def basis_spec():
    return {"r": {"n": NR, "degree": 3, "range_w": [K_LO, K_HI], "knots": "clamped, 균일", "outside": "끝값 상수"},
            "l": {"n": NL, "degree": 3, "range_s": [S_LO, 1.0], "knots": "clamped, 균일 (s = sqrt(u))"},
            "w_grid": {"n": 160, "lo": float(BASIS["wg"][0]), "hi": float(BASIS["wg"][-1])},
            "u_grid": {"n": 200, "s_lo": S_LO, "s_hi": 1.0},
            "w12": BASIS["w12"].tolist(), "u12": BASIS["u12"].tolist()}


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
        d = {"a": v[0], "cr": cr, "cl": cl, "lnsig0": v[it + 2], "lnrho": v[it + 3]}
        d.update({n: v[it + 4 + i] for i, n in enumerate(OFF)})
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

    def __init__(self, pr):
        self.pr = pr
        self.K = int(pr["K"])
        self.n = 1 + self.K + 8
        self.mc = np.asarray(pr["mean_coef"], float)
        self.phi = np.asarray(pr["phi_coef"], float).reshape(self.K, NR + NL)
        self.lam = np.asarray(pr["lambda_scaled"], float).reshape(self.K)
        nu = pr["nuisance"]
        assert nu["names"] == NUIS
        self.nmean = np.asarray(nu["mean"], float)
        cov = np.asarray(nu["cov"], float) + 1e-6 * np.eye(8)
        self.nch = np.linalg.cholesky(cov)
        self.am, self.av = pr["a"]["mean"], pr["a"]["var"]
        self.start = np.concatenate([[self.am], np.zeros(self.K), self.nmean])

    def theta(self, v):
        import jax.numpy as jnp
        K = self.K
        c = jnp.asarray(self.mc)
        if K:
            c = c + v[1:1 + K] @ jnp.asarray(self.phi)
        o = 1 + K
        d = {"a": v[0], "cr": c[:NR], "cl": c[NR:], "lnsig0": v[o], "lnrho": v[o + 1]}
        d.update({n: v[o + 2 + i] for i, n in enumerate(OFF)})
        return d

    def logprior(self, v):
        import jax.numpy as jnp
        from jax.scipy.linalg import solve_triangular
        K = self.K
        lp = _nlp(v[0], self.am, math.sqrt(self.av))
        if K:
            lp += _nlp(v[1:1 + K], 0.0, jnp.sqrt(jnp.asarray(self.lam)))
        r = v[1 + K:] - jnp.asarray(self.nmean)
        ch = jnp.asarray(self.nch)
        zz = solve_triangular(ch, r, lower=True)
        return lp - 0.5 * (zz ** 2).sum() - jnp.log(jnp.diag(ch)).sum() - 4 * math.log(2 * math.pi)


# ---------- 모형·MAP ----------

def model(obs, prob, pre):
    """numpyro 모형 (MAP+라플라스 아핀 재매개화 위 z 표집). 표집 이름 z·eps·phi, deterministic v·th24(1단계)·r12·l12·lnC_q."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import constraints
    z = numpyro.sample("z", dist.ImproperUniform(constraints.real_vector, (), (prob.n,)))
    v = jnp.asarray(pre["loc"], jnp.float32) + jnp.asarray(pre["chol"], jnp.float32) @ z
    numpyro.factor("prior", prob.logprior(v))
    numpyro.deterministic("v", v)
    th = prob.theta(v)
    if prob.kind == "stage1":
        numpyro.deterministic("th24", jnp.concatenate([v[:prob.it], v[prob.it + 2:]]))
        r12, l12 = curves12(th["cr"], th["cl"])
        numpyro.deterministic("r12", r12)
        numpyro.deterministic("l12", l12)
    mu, P = surface(th, obs["y"], obs["u"])
    OM.observe(obs, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]))
    for q in OM.Q_TARGETS:
        numpyro.deterministic(f"lnC_{q:g}", lnC(th, q))


def map_precond(obs, prob, iters=1500, temps=(0.003, 0.03, 0.3, 1.0)):
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

        def negpost(vf, temp):
            v = full(vf)
            th = prob.theta(v)
            mu, P = surface(th, o["y"], o["u"])
            ll = OM.row_loglik(o, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]), eps0, phi0)
            return -(temp * ll.sum() + prob.logprior(v))

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

        x, nit, res = solve(np.asarray(prob.start, float)[free], temps, 0)
        if tau_ix:                                   # tau 갱신 후 다시 (앞 답에서 출발, 온도 1)
            v = np.array(full(jnp.asarray(x)), float)
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
        loc = np.asarray(full(jnp.asarray(x)), float)
    H = 0.5 * (H + H.T)
    w, Q = np.linalg.eigh(H)
    sq = np.zeros((n, n))
    sq[np.ix_(free, free)] = Q / np.sqrt(np.maximum(w, 1e-6))   # 공분산의 제곱근 (콜레스키가 아니어도 아핀 재매개화로 충분)
    for i in tau_ix:
        sq[i, i] = 0.7
    return {"loc": loc, "chol": sq, "info": {"fun": float(res.fun), "nit": nit, "ok": bool(res.success),
                                            "min_eig": float(w.min()), "max_eig": float(w.max()), "n_floored": int((w < 1e-6).sum()), "n_escape": n_escape,
                                            "tau_fixed": [math.exp(loc[i]) for i in tau_ix]}}


def fit(rows, H, season, err, prob, delay_key, chains=4, warmup=500, samples=500, seed=0, stage1=False, K=16, target_accept=0.9, retry=True):
    """관측 줄 -> MAP -> NUTS (retry False면 한 번만). (samples, info, diag, obs). info에 map·secs·n·N."""
    obs = OM.prepare(rows, H, season, err, K=K, delay_key=delay_key, stage1=stage1)
    jobs = OM.to_jax(obs)
    t0 = time.time()
    pre = map_precond(jobs, prob)
    t_map = time.time() - t0
    mdl = functools.partial(model, prob=prob, pre=pre)
    lnames = [f"lnC_{q:g}" for q in (1, 3, 10)]
    names = (["th24", "r12", "l12"] if stage1 else ["v"]) + lnames
    if retry:
        sm, info, dg = OM.fit_with_retry(mdl, {"obs": jobs}, names=names, seed=seed, chains=chains, warmup=warmup, samples=samples,
                                         target_accept=target_accept)
    else:
        sm, info = OM.run_nuts(mdl, {"obs": jobs}, seed=seed, chains=chains, warmup=warmup, samples=samples, target_accept=target_accept)
        dg = OM.diagnose(sm, names, divergences=info["divergences"])
        info["retried"] = False
    info["map_secs"] = round(t_map, 1)
    info["map"] = pre["info"]
    info["pre"] = {"loc": pre["loc"], "chol": pre["chol"]}
    info["n"], info["N"], info["counts"] = obs["n"], obs["N"], obs["counts"]
    return sm, info, dg, obs


def laplace_fit(rows, H, season, err, prob, delay_key, ndraw=4000, seed=0, K=16):
    """1단계 라플라스 근사 (명세 6절 C4): 결합 MAP(float64, eps·phi는 사전 평균에 고정) + 역 Hessian 공분산(고윳값 바닥 1e-6) -> v 4000뽑기.
    NUTS 쪽과 같은 모양의 (sm, info, dg, obs)를 돌려준다 (사슬 1개)."""
    import jax
    obs = OM.prepare(rows, H, season, err, K=K, delay_key=delay_key, stage1=True)
    jobs = OM.to_jax(obs)
    t0 = time.time()
    pre = map_precond(jobs, prob)
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
    it = v.shape[1] - 10
    return np.concatenate([v[:, [0, it + 2, it + 3]], v[:, it + 4:], sm["r12"].reshape(-1, 12), sm["l12"].reshape(-1, 12)], axis=1)

def stage1_entry(sm, info, dg, obs):
    v = sm["v"].reshape(-1, sm["v"].shape[-1])
    it = v.shape[1] - 10
    X = s1_X(sm)
    lc = {f"{q:g}": np.quantile(sm[f"lnC_{q:g}"].ravel(), [0.1, 0.5, 0.9]).tolist() for q in OM.Q_TARGETS}
    tau_rh = OM.diagnose({"lt": sm["v"][..., it:it + 2]}, ["lt"]) if sm["v"].shape[0] > 1 else {"rhat_max": None, "ess_min": None}
    return {"names": S1_NAMES, "mean": X.mean(0).tolist(), "cov": np.cov(X.T).tolist(),
            "rhat_max": dg["rhat_max"], "ess_min": dg["ess_min"], "divergences": info["divergences"], "conv": dg["conv"],
            "secs": info["secs"], "secs_total": round(info["secs"] + info.get("secs_first_try", 0.0) + info["map_secs"], 1),
            "map_secs": info["map_secs"], "retried": info["retried"], "tree_cap_frac": info["tree_cap_frac"],
            "map": info["map"], "n": obs["n"], "N": obs["N"], "counts": obs["counts"],
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
        configure(b["r"]["n"], b["l"]["n"], 3 * math.exp(b["r"]["range_w"][0]), 3 * math.exp(b["r"]["range_w"][1]))
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
    ix = [e["names"].index(f"{p}{i}") for p in "rl" for i in range(12)]
    return np.asarray(e["mean"], float)[ix], np.asarray(e["cov"], float)[np.ix_(ix, ix)]


# 가중 함수형 PCA (Ramsay-Silverman 8장, 0/1 가중): 주성분은 BAND(자료가 실제로 있는 구간)에서만 낸다.
# 끝(r12 0~1·10~11 = 0.1~0.3%·13.5~40%, l12 0~1 = u < 0.089)은 자료가 없어 사후 평균·분산이 사전에 끌려 PC를 지배했다 (2026-10-29).
BAND_R = list(range(2, 10))
BAND_L = list(range(2, 12))
BAND = BAND_R + [12 + i for i in BAND_L]          # 24벡터 안의 위치 (r12 앞 12, l12 뒤 12)
OUT = [i for i in range(24) if i not in BAND]


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
    phi = np.zeros((K, 24))
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
    mean_coef = np.concatenate([np.linalg.lstsq(Br, p["mu"][:12], rcond=None)[0], np.linalg.lstsq(Bl, p["mu"][12:], rcond=None)[0]])
    phi_coef = np.array([np.concatenate([np.linalg.lstsq(Br, f[:12], rcond=None)[0], np.linalg.lstsq(Bl, f[12:], rcond=None)[0]])
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

def cell_fit(rows, H, s, err, pr, chains=4, warmup=500, samples=500, seed=0):
    prob = CellProb(pr)
    sm, info, dg, obs = fit(rows, H, s, err, prob, s, chains, warmup, samples, seed, stage1=False)
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
            sm, info, dg, obs = cell_fit(rows, head["H"], s, err, pr, a.chains, a.warmup, a.samples)
            out["cells"] += OM.pred_cells(s, T, {q: sm[f"lnC_{q:g}"] for q in OM.Q_TARGETS}, dg["conv"])
            out["info"].append({"season": s, "T": T, "secs": info["secs"], "map_secs": info["map_secs"], "conv": dg["conv"], "K": pr["K"],
                                "k": pr["k"], "rhat_max": dg["rhat_max"], "ess_min": dg["ess_min"], "divergences": info["divergences"],
                                "retried": info["retried"], "tree_cap_frac": info["tree_cap_frac"], "n": obs["n"], "N": obs["N"]})
            _atomic_json(a.out, out)
            i = out["info"][-1]
            print(s, T, f"n={i['n']} secs={i['secs']} rhat={i['rhat_max']:.3f} ess={i['ess_min']:.0f} div={i['divergences']} conv={i['conv']}", flush=True)


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
        lo, hi = np.quantile(sm[nme].reshape(-1, 12), [0.05, 0.95], axis=0)
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
    s.add_argument("--n-basis-r", type=int, default=12)
    s.add_argument("--n-basis-l", type=int, default=12)
    s.add_argument("--r-range-pct", default="0.1,40", help="r 매듭 범위 하한,상한 (%)")
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
    q = sub.add_parser("predict")
    common(q)
    q.add_argument("--stage1", required=True)
    q.add_argument("--obs-dir", required=True)
    q.add_argument("--seasons", required=True)
    q.add_argument("--T", default="all")
    q.add_argument("--out", required=True)
    q.add_argument("--prior-cache", default=None)
    a = ap.parse_args()
    if a.cmd == "stage1":
        lo, hi = (float(x) for x in a.r_range_pct.split(","))
        configure(a.n_basis_r, a.n_basis_l, lo, hi)
    if getattr(a, "k", None) not in (None, "all"):
        a.k = int(a.k)
    {"selftest": cmd_selftest, "stage1": cmd_stage1, "prior": cmd_prior, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
