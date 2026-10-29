"""M2 HSGP 곡면 (WP3b M2 첫 단계, 2026-10-29) — 곡면 함수와 모형, MAP + 라플라스, 시간 재기. m1.py 꼴을 따른다.

  import m2 as M2        # sys.path.insert(0, HERE) 뒤
  python3 -I tools/solo_raid/lab7/m2.py bench --work W --case a|b|c [--season S30] [--chains 4 --warmup 500 --samples 500]
  python3 -I tools/solo_raid/lab7/m2.py test --work W --which basis|synth     # 시험 (m2_bench.py가 아니라 이 파일 하위 명령)

명세 「WP3 명세」 1·4·5·6절. 좌표 y = ln 점수, w = ln(q/3), u = h/H.
  최종 컷: ln C_k(w) = a_k - int_0^w exp(r_k(v)) dv,  r_k(w) = rbar(w) + f1(w) + kappa_k,  a_k = a + delta_k
           w 격자 120점 [ln(0.1/3), ln(40/3)] (w = 0이 격자 점 — 음쪽 69점 + 양쪽 52점으로 따로 등간격), 누적 사다리꼴.
           점수 y -> 최종 w_fin: ln C 격자를 역으로 보간, 격자 밖은 끝 기울기 exp(r_end)로 선형 외삽. z_fin = log10 3 + w_fin / ln 10.
  침수:    -dP/du = exp(l(u)),  l(u) = lbar(u) + ln m + psi_k + f2(u),  P(u) = int_u^1 exp(l) dv.
           u 격자 200점 [0.02, 1], 제곱 간격 u = 0.02 + 0.98 s^2 (s 등간격) — 앞쪽이 촘촘하고 끝 간격 약 0.0098.
           역방향 누적 사다리꼴, 관측 u[N,K]에서 선형 보간 (u = 1이면 정확히 0). f2는 u 격자 점 값의 평균을 빼 0으로 묶는다.
           (psi_k는 P 전체에 곱하는 exp(psi_k)로 정확히 같다.)
  f1·f2:   HSGP (Solin-Sarkka 2020), Matern-5/2, M = 20, c = 1.5. phi_j(x) = sqrt(1/L) sin(pi j (x - x0 + L)/(2L)),
           f = sum_j sqrt(s(sqrt_lam_j)) beta_j phi_j(x), beta ~ N(0,1) (비중심). 직접 쓴 순수 jnp (float64 MAP에서도 돈다).
  지금 %: mu = z_fin - P.  서버 k(KR·JP·GL): dJP,dGL (a), kJP,kGL (r에 더함), pJP,pGL (ln m에 더함) — m1과 같은 이름. KR은 0.

평균 함수 rbar[120]·lbar[200]은 인자다 (mean = {"rbar", "lbar"}; None이면 임시 평균):
  임시 평균 (M1 2단계 평균 곡선이 아직 없어서, 2026-10-29): rbar(w) = ln 0.3 + 1.0 w, lbar = M1 kuma eta=0 (F(u)=u)의 -dP/du 로그
  (= -ln u - ln(-ln 0.4)) 를 격자에서 P(0.4) = 1이 되게 다시 정규화.  default_mean() / mean_from_m1(theta, form)  (M1 theta에서 만드는 함수).

초모수 (f1·f2 각각 길이 ell, 크기 alpha):
  hyper=None (1단계): ell ~ InverseGamma(a, b) — 정의역 폭 W의 [W/10, W]가 1%~99% 분위가 되게 (scipy로 풀어 inv_gamma_params(W)),
                       alpha ~ HalfNormal(0.3). 로그 척도 (lnell1, lnalp1, lnell2, lnalp2)로 재매개화해 pre(라플라스)에 같이 넣는다.
  hyper = {"ell1","alpha1","ell2","alpha2"} (2단계): 고정값. hyper_center()가 사전 중앙값 (IG 중앙값, HalfNormal 중앙값 0.202).
  W: f1은 w 정의역 폭 5.99, f2는 u 정의역 폭 0.98.

  mu, P = surface(theta, y, u, G, mean)       # y[N], u[N,K] -> mu[3,N,K], P[3,N,K]  (obs_model 곡면 함수 약속). theta: a,lnm,dJP..kGL,beta1,beta2 + ell1,alpha1,ell2,alpha2
  model(obs, prior, hyper=None, mean=None, pre=None, keep_ll=False)   # numpyro 모형. prior = {"kind": "weak"} 또는 hier.prior_for 꼴 (names에 lnm·lnsig0·lnrho·d/p/k·a가 있어야 한다)
  pre = map_precond(jobs, prior, hyper, mean)  # float64 MAP + 라플라스. model(..., pre=pre)로 넘긴다. (pre 없으면 사전 평균 중심 대각 척도)
  표집 이름: z (불확실 벡터, 실제 값은 deterministic 아래 이름들), eps, phi. deterministic: a, lnm, lnsig0, lnrho, dJP..kGL,
             (hyper=None이면) lnell1, lnalp1, lnell2, lnalp2, beta1[20], beta2[20], lnC_{q} (q = 0.5 1 2 3 5 10, KR).
  v 벡터 순서: [a, lnm, lnsig0, lnrho, dJP, dGL, pJP, pGL, kJP, kGL, (lnell1, lnalp1, lnell2, lnalp2), beta1(20), beta2(20)].
공개 규칙: 글 번호·원문·닉네임은 쓰지 않는다.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_model as OM  # noqa: E402
import m1 as M1  # noqa: E402

LN10 = math.log(10.0)
L10_3 = math.log10(3.0)
MB = 20            # 기저 수
CB = 1.5           # 경계 배수
NW_NEG, NW_POS = 69, 52     # w 격자: [lo, 0] 69점 + (0, hi] 51점 = 120점, 0이 격자 점
NU = 200
U_LO = 0.02
W_LO, W_HI = math.log(0.1 / 3.0), math.log(40.0 / 3.0)
CORE = ["lnm", "lnsig0", "lnrho"] + list(M1.OFF)       # a 다음 9개
HYP = ["lnell1", "lnalp1", "lnell2", "lnalp2"]
ALPHA_SD = 0.3


# ---------- 격자 · 기저 (numpy 상수) ----------

def hsgp_basis(x, lo, hi, M=MB, c=CB):
    """-> (Phi[len(x), M], sqrt_lam[M], L). 정의역 [lo, hi]의 중심 x0, 반폭 S, L = c S."""
    x = np.asarray(x, float)
    x0, S = 0.5 * (lo + hi), 0.5 * (hi - lo)
    L = c * S
    j = np.arange(1, M + 1)
    phi = np.sqrt(1.0 / L) * np.sin(np.pi * j[None, :] * (x[:, None] - x0 + L) / (2 * L))
    return phi, np.pi * j / (2 * L), L


def matern52_spec(omega, ell, alpha):
    """1차원 Matern-5/2 스펙트럼 밀도 (jnp/np 둘 다)."""
    return alpha ** 2 * (16.0 / 3.0) * 5.0 ** 2.5 / ell ** 5 * (5.0 / ell ** 2 + omega ** 2) ** (-3.0)


def matern52_cov(d, ell, alpha):
    r = np.sqrt(5.0) * np.abs(d) / ell
    return alpha ** 2 * (1 + r + r ** 2 / 3.0) * np.exp(-r)


class Grids:
    """w·u 격자와 HSGP 기저 (numpy float64 상수)."""

    def __init__(self):
        wn = np.linspace(W_LO, 0.0, NW_NEG)
        wp = np.linspace(0.0, W_HI, NW_POS)[1:]
        self.w = np.concatenate([wn, wp])
        self.i0 = NW_NEG - 1
        assert self.w[self.i0] == 0.0 and len(self.w) == 120
        s = np.linspace(0.0, 1.0, NU)
        self.u = U_LO + (1.0 - U_LO) * s ** 2
        self.Phi_w, self.sl_w, _ = hsgp_basis(self.w, W_LO, W_HI)
        self.Phi_u, self.sl_u, _ = hsgp_basis(self.u, U_LO, 1.0)
        self.width_w = W_HI - W_LO
        self.width_u = 1.0 - U_LO


_G = None


def grids():
    global _G
    if _G is None:
        _G = Grids()
    return _G


# ---------- 평균 함수 ----------

def _normalize_lbar(u, lbar):
    """격자 사다리꼴 P(0.4) = 1이 되게 lbar에 상수를 더한다 (numpy)."""
    e = np.exp(lbar)
    seg = 0.5 * (e[1:] + e[:-1]) * np.diff(u)
    P = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    p04 = np.interp(0.4, u, P)
    return lbar - math.log(p04)


def default_mean(G=None):
    """임시 평균: rbar = ln 0.3 + w, lbar = log(-dP/du) (M1 kuma eta=0, F(u)=u; P(0.4)=1)."""
    G = G or grids()
    rbar = math.log(0.3) + 1.0 * G.w
    lbar = -np.log(G.u) - math.log(-math.log(0.4))
    return {"rbar": rbar, "lbar": _normalize_lbar(G.u, lbar)}


def mean_from_m1(theta, form, G=None):
    """M1 theta(dict: r0, r1, lnm 안 씀, eta)에서 평균 곡선: rbar = r0 + r1 w, lbar = ln(-dP/du) (m = 1, 중앙차분), 정규화."""
    import jax.numpy as jnp
    G = G or grids()
    eta = [jnp.float32(e) for e in M1._eta_of(theta, form)]
    h = 1e-4
    uc = np.clip(G.u, U_LO + h, 1.0 - h)
    p = lambda uu: np.asarray(M1._p0(jnp.asarray(uu, jnp.float32), eta, form), float)
    p04 = float(M1._p0(jnp.float32(0.4), eta, form))
    d = -(p(uc + h) - p(uc - h)) / (2 * h) / p04
    lbar = np.log(np.maximum(d, 1e-12))
    return {"rbar": float(theta["r0"]) + float(theta["r1"]) * G.w, "lbar": _normalize_lbar(G.u, lbar)}


# ---------- 초모수 사전 ----------

def inv_gamma_params(W):
    """InverseGamma(a, b): 1% 분위 = W/10, 99% 분위 = W 가 되는 (a, b)."""
    from scipy import optimize, stats
    lo, hi = W / 10.0, W

    def f(p):
        a, b = math.exp(p[0]), math.exp(p[1])
        return [stats.invgamma.ppf(0.01, a, scale=b) - lo, stats.invgamma.ppf(0.99, a, scale=b) - hi]
    s = optimize.fsolve(f, [math.log(3.0), math.log(0.5 * W)])
    return math.exp(s[0]), math.exp(s[1])


def hyper_center(G=None):
    """사전 중앙값 (길이 = IG 중앙값, 크기 = HalfNormal(0.3) 중앙값)."""
    from scipy import stats
    G = G or grids()
    out = {}
    for tag, W in (("1", G.width_w), ("2", G.width_u)):
        a, b = inv_gamma_params(W)
        out["ell" + tag] = float(stats.invgamma.ppf(0.5, a, scale=b))
        out["alpha" + tag] = float(stats.halfnorm.ppf(0.5, scale=ALPHA_SD))
    return out


def _hyper_logp(x4, igp):
    """x = ln(ell), ln(alpha) 척도의 로그 밀도 (야코비안 포함). x4 = [lnell1, lnalp1, lnell2, lnalp2]."""
    import jax.numpy as jnp
    from jax.scipy.special import gammaln
    tot = 0.0
    for i, (a, b) in enumerate(igp):
        lx = x4[2 * i]
        ell = jnp.exp(lx)
        tot = tot + a * math.log(b) - gammaln(a) - (a + 1.0) * lx - b / ell + lx
        la = x4[2 * i + 1]
        al = jnp.exp(la)
        tot = tot + math.log(2.0) - 0.5 * math.log(2 * math.pi) - math.log(ALPHA_SD) - 0.5 * (al / ALPHA_SD) ** 2 + la
    return tot


def _igps(G):
    return [inv_gamma_params(G.width_w), inv_gamma_params(G.width_u)]


# ---------- 곡면 ----------

def _hsgp_f(Phi, sl, ell, alpha, beta):
    import jax.numpy as jnp
    sp = jnp.sqrt(matern52_spec(jnp.asarray(sl, beta.dtype), ell, alpha))
    return jnp.asarray(Phi, beta.dtype) @ (sp * beta)


def curves(theta, G, mean):
    """-> dict(lc[3,120] 서버별 ln C 격자, Q[200] 침수 P 격자 (m, f2 포함; 서버 배수 전), rg[3,120])."""
    import jax.numpy as jnp
    dt = jnp.asarray(theta["a"]).dtype
    w = jnp.asarray(G.w, dt)
    u = jnp.asarray(G.u, dt)
    f1 = _hsgp_f(G.Phi_w, G.sl_w, theta["ell1"], theta["alpha1"], theta["beta1"])
    f2 = _hsgp_f(G.Phi_u, G.sl_u, theta["ell2"], theta["alpha2"], theta["beta2"])
    f2 = f2 - f2.mean()
    z = jnp.zeros((), dt)
    dl = [z, theta["dJP"], theta["dGL"]]
    kp = [z, theta["kJP"], theta["kGL"]]
    lcs, rgs = [], []
    dw = jnp.diff(w)
    for k in range(3):
        r = jnp.clip(jnp.asarray(mean["rbar"], dt) + f1 + kp[k], -15.0, 10.0)
        e = jnp.exp(r)
        cum = jnp.concatenate([jnp.zeros(1, dt), jnp.cumsum(0.5 * (e[1:] + e[:-1]) * dw)])
        cum = cum - cum[G.i0]
        lcs.append(theta["a"] + dl[k] - cum)
        rgs.append(r)
    el = jnp.exp(jnp.clip(jnp.asarray(mean["lbar"], dt) + theta["lnm"] + f2, -15.0, 10.0))
    seg = 0.5 * (el[1:] + el[:-1]) * jnp.diff(u)
    Q = jnp.concatenate([jnp.cumsum(seg[::-1])[::-1], jnp.zeros(1, dt)])
    return {"lc": jnp.stack(lcs), "rg": jnp.stack(rgs), "Q": Q, "f1": f1, "f2": f2}


def _inv_w(y, wg, lc, r):
    """y[N] -> 최종 w. lc는 w에 감소. 격자 밖은 끝 기울기 exp(r_end)로 선형 외삽."""
    import jax.numpy as jnp
    xp, fp = lc[::-1], wg[::-1]
    w = jnp.interp(y, xp, fp)
    w_hi = wg[-1] + (lc[-1] - y) / jnp.exp(r[-1])    # y < xp[0]: 점수가 40% 컷 밑
    w_lo = wg[0] + (lc[0] - y) / jnp.exp(r[0])       # y > xp[-1]: 점수가 0.1% 컷 위
    return jnp.where(y < xp[0], w_hi, jnp.where(y > xp[-1], w_lo, w))


def surface(theta, y, u, G=None, mean=None):
    """-> (mu[3,N,K], P[3,N,K]). theta 약속은 머리 참조."""
    import jax.numpy as jnp
    G = G or grids()
    mean = mean or default_mean(G)
    cv = curves(theta, G, mean)
    dt = cv["Q"].dtype
    wg = jnp.asarray(G.w, dt)
    Pb = jnp.interp(u, jnp.asarray(G.u, dt), cv["Q"])             # [N,K]
    ps = [jnp.zeros((), dt), theta["pJP"], theta["pGL"]]
    mus, Ps = [], []
    for k in range(3):
        wf = _inv_w(y, wg, cv["lc"][k], cv["rg"][k])
        zf = L10_3 + wf / LN10
        P = Pb * jnp.exp(ps[k])
        mus.append(zf[:, None] - P)
        Ps.append(P)
    return jnp.stack(mus), jnp.stack(Ps)


def lnC(theta, q, G=None, mean=None):
    """KR 최종 컷 ln 점수 (q = 목표 %, 격자 범위 0.1~40 안)."""
    import jax.numpy as jnp
    G = G or grids()
    mean = mean or default_mean(G)
    cv = curves(theta, G, mean)
    dt = cv["Q"].dtype
    return jnp.interp(jnp.log(jnp.asarray(q, dt) / 3.0), jnp.asarray(G.w, dt), cv["lc"][0])


# ---------- 모형 ----------

def _v_layout(hyper):
    names = ["a"] + CORE + ([] if hyper is not None else HYP)
    return names, len(names) + 2 * MB


def _unpack(v, hyper, G):
    """v -> theta dict (jnp)."""
    import jax.numpy as jnp
    names, n0 = _v_layout(hyper)
    th = {nm: v[i] for i, nm in enumerate(names)}
    if hyper is None:
        th["ell1"], th["alpha1"], th["ell2"], th["alpha2"] = (jnp.exp(th[h]) for h in HYP)
    else:
        for k in ("ell1", "alpha1", "ell2", "alpha2"):
            th[k] = jnp.asarray(hyper[k], v.dtype)
    nh = len(names)
    th["beta1"] = v[nh:nh + MB]
    th["beta2"] = v[nh + MB:nh + 2 * MB]
    return th


def _logprior(v, prior, hyper, G, igp):
    """사전 로그 밀도 (dtype을 따른다 — float64 MAP에서도 쓴다)."""
    import jax.numpy as jnp
    names, n0 = _v_layout(hyper)
    lp = M1._prior_logp64(v[:1 + len(CORE)], CORE, prior)
    nh = len(names)
    if hyper is None:
        lp = lp + _hyper_logp(v[1 + len(CORE):nh], igp)
    b = v[nh:]
    return lp - 0.5 * (b ** 2).sum() - 0.5 * len(b) * math.log(2 * math.pi)


def _center(prior, hyper, G):
    """v의 사전 중심 (loc)과 척도 (sd) — pre가 없을 때의 기본."""
    names, n0 = _v_layout(hyper)
    if prior.get("kind") == "weak":
        loc = [M1.WEAK["a"][0]] + [M1.WEAK[n][0] for n in CORE]
        sd = [M1.WEAK["a"][1]] + [M1.WEAK[n][1] for n in CORE]
    else:
        ix = [prior["names"].index(n) for n in CORE]
        loc = [prior["a"]["mean"]] + list(np.asarray(prior["mean"])[ix])
        sd = [math.sqrt(prior["a"]["var"])] + list(np.sqrt(np.diag(np.asarray(prior["cov"]))[ix]))
    if hyper is None:
        c = hyper_center(G)
        loc += [math.log(c["ell1"]), math.log(c["alpha1"]), math.log(c["ell2"]), math.log(c["alpha2"])]
        sd += [0.5, 0.7, 0.5, 0.7]
    loc += [0.0] * (2 * MB)
    sd += [1.0] * (2 * MB)
    return np.asarray(loc, float), np.asarray(sd, float)


def model(obs, prior, hyper=None, mean=None, pre=None, keep_ll=False):
    """numpyro 모형. 머리 docstring 참조. pre = map_precond 결과 {"loc","chol"}."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import constraints
    G = grids()
    mean = mean or default_mean(G)
    names, n0 = _v_layout(hyper)
    if pre is None:
        loc, sd = _center(prior, hyper, G)
        pre = {"loc": loc, "chol": np.diag(sd)}
    z = numpyro.sample("z", dist.ImproperUniform(constraints.real_vector, (), (n0,)))
    v = jnp.asarray(pre["loc"], jnp.float32) + jnp.asarray(pre["chol"], jnp.float32) @ z
    igp = _igps(G)
    numpyro.factor("prior", _logprior(v, prior, hyper, G, igp))
    th = _unpack(v, hyper, G)
    for nm in names:
        numpyro.deterministic(nm, th[nm])
    numpyro.deterministic("beta1", th["beta1"])
    numpyro.deterministic("beta2", th["beta2"])
    mu, P = surface(th, obs["y"], obs["u"], G, mean)
    OM.observe(obs, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]), keep_ll)
    for q in OM.Q_TARGETS:
        numpyro.deterministic(f"lnC_{q:g}", lnC(th, q, G, mean))


def map_precond(obs, prior, hyper=None, mean=None, start=None, iters=1500, temps=(0.003, 0.03, 0.3, 1.0), two_step=True):
    """float64 MAP + 라플라스 (m1.map_precond와 같은 방식, eps·phi는 사전 평균에 고정). -> dict(loc, chol, info)
    hyper=None(초모수 표집)일 때 two_step=True면: (1) 초모수를 사전 중앙값에 고정하고 나머지(a·core·beta)의 MAP + 라플라스를 구하고,
    (2) 초모수 4개는 사전 중앙값 / 대각 사전 sd(0.5, 0.7, 0.5, 0.7)로 이어 붙인다. 초모수를 함께 푼 MAP은 beta-크기 능선 때문에 헤세 행렬이 불정부호가 되었다
    (2026-10-29 첫 시간 재기에서 min_eig -150). 남은 상관은 NUTS의 dense 질량 행렬이 예열에서 배운다."""
    if hyper is None and two_step:
        hc = hyper_center()
        n_core = 1 + len(CORE)
        n = n_core + 4 + 2 * MB
        keep = list(range(n_core)) + list(range(n_core + 4, n))          # 초모수 자리를 뺀 v 위치
        sub = map_precond(obs, prior, hc, mean, None if start is None else np.asarray(start, float)[keep], iters, temps, two_step=False)
        hp = [n_core + i for i in range(4)]
        cov = np.zeros((n, n))
        cov[np.ix_(keep, keep)] = sub["cov"]
        sdh = [0.5, 0.7, 0.5, 0.7]
        for i, j in enumerate(hp):
            cov[j, j] = sdh[i] ** 2
        loc = np.zeros(n)
        loc[keep] = sub["loc"]
        loc[hp] = [math.log(hc["ell1"]), math.log(hc["alpha1"]), math.log(hc["ell2"]), math.log(hc["alpha2"])]
        return {"loc": loc, "chol": np.linalg.cholesky(cov + 1e-12 * np.eye(n)), "cov": cov,
                "info": dict(sub["info"], note="초모수는 사전 중앙값에 고정한 MAP + 사전 sd")}
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    G = grids()
    mean = mean or default_mean(G)
    igp = _igps(G)
    with jax.enable_x64(True):
        o = dict(obs)
        for k in OM.NUMERIC:
            a = np.asarray(obs[k])
            o[k] = jnp.asarray(a, jnp.int64 if a.dtype.kind in "iu" else jnp.float64)
        eps0 = obs["eps_ab"][0] / (obs["eps_ab"][0] + obs["eps_ab"][1])
        phi0 = o["dirichlet"] / o["dirichlet"].sum(1, keepdims=True)

        def negpost(v, temp):
            th = _unpack(v, hyper, G)
            mu, P = surface(th, o["y"], o["u"], G, mean)
            ll = OM.row_loglik(o, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]), eps0, phi0)
            return -(temp * ll.sum() + _logprior(v, prior, hyper, G, igp))

        vg = jax.jit(jax.value_and_grad(negpost))
        hess = jax.jit(jax.hessian(negpost))
        x = np.asarray(start if start is not None else _center(prior, hyper, G)[0], float)
        nit = 0
        for temp in temps:
            def f(xx, temp=temp):
                val, g = vg(jnp.asarray(xx), temp)
                val, g = float(val), np.asarray(g, float)
                return (val, g) if np.isfinite(val) and np.isfinite(g).all() else (1e12, np.zeros_like(g))
            res = minimize(f, x, jac=True, method="L-BFGS-B", options={"maxiter": iters, "maxfun": 3 * iters})
            x, nit = res.x, nit + res.nit
        f1 = lambda xx: float(vg(jnp.asarray(xx), 1.0)[0])

        def polish(x):
            """뉴턴 다듬기 (m1과 같다): L-BFGS는 조건수가 큰 사후에서 일찍 선다."""
            n_new = 0
            for _ in range(60):
                fv, g = vg(jnp.asarray(x), 1.0)
                fv, g = float(fv), np.asarray(g, float)
                Hn = np.asarray(hess(jnp.asarray(x), 1.0), float)
                Hn = 0.5 * (Hn + Hn.T)
                wv, Qv = np.linalg.eigh(Hn)
                step = Qv @ ((Qv.T @ g) / np.maximum(np.abs(wv), 1e-3))
                t, ok = 1.0, False
                while t > 1e-4:
                    fn = f1(x - t * step)
                    if np.isfinite(fn) and fn < fv - 1e-9:
                        ok = True
                        break
                    t *= 0.5
                if not ok:
                    break
                x = x - t * step
                n_new += 1
                if fv - fn < 1e-6:
                    break
            return x, n_new

        x, n_newton = polish(x)
        n_saddle = 0
        for _ in range(3):   # 안장점: Hessian 최소 고윳값 <= 0이면 그 방향으로 +-0.5 밀어 다시 푼다 (m1과 같다)
            Hs = np.asarray(hess(jnp.asarray(x), 1.0), float)
            ws, Qs = np.linalg.eigh(0.5 * (Hs + Hs.T))
            if ws.min() > 1e-6:
                break
            n_saddle += 1
            best = (f1(x), x)
            for sgn in (1.0, -1.0):
                r2_ = minimize(lambda xx: f(xx, temp=1.0), x + sgn * 0.5 * Qs[:, 0], jac=True, method="L-BFGS-B",
                               options={"maxiter": iters, "maxfun": 3 * iters})
                nit += r2_.nit
                xn, nn = polish(r2_.x)
                n_newton += nn
                fn_ = f1(xn)
                if np.isfinite(fn_) and fn_ < best[0]:
                    best = (fn_, xn)
            x = best[1]
        res.fun = f1(x)
        H = np.asarray(hess(jnp.asarray(x), 1.0), float)
    H = 0.5 * (H + H.T)
    w, Q = np.linalg.eigh(H)
    cov = (Q / np.maximum(w, 1e-6)) @ Q.T
    chol = np.linalg.cholesky(0.5 * (cov + cov.T) + 1e-12 * np.eye(len(x)))
    return {"loc": x, "chol": chol, "cov": 0.5 * (cov + cov.T),
            "info": {"fun": float(res.fun), "nit": nit, "ok": bool(res.success), "min_eig": float(w.min()), "max_eig": float(w.max()),
                     "n_newton": n_newton, "n_saddle": n_saddle}}


# ---------- 다중 시작 MAP ----------

def multistart_map(obs, prior, hyper=None, mean=None, n=8, seed=0, scatter=0.5):
    """시작점 n개의 MAP (m1.multistart_map과 같은 꼴): 첫째는 기본 시작, 나머지는 그 해에서 서버 보조 모수(dJP..kGL)만 N(0, scatter^2)로 흩음.
    -> info.fun 오름차순 [pre dict] (fun = -log 사후 상대값이므로 작을수록 높은 봉우리)."""
    names, _ = _v_layout(hyper)
    base = map_precond(obs, prior, hyper, mean)
    out = [base]
    rng = np.random.default_rng(seed)
    idx = [i for i, nm in enumerate(names) if nm in M1.OFF]
    for _ in range(n - 1):
        st = np.array(base["loc"], float)
        st[idx] += rng.normal(0, scatter, len(idx))
        out.append(map_precond(obs, prior, hyper, mean, start=st))
    return sorted(out, key=lambda p: p["info"]["fun"])


def map_init(n):
    """NUTS 시작: z는 0(= MAP), 나머지(eps·phi)는 init_to_median (m1.map_init)."""
    return M1.map_init(n)


# ---------- 실행 도구 ----------

def _draws_lnc(vdraws, hyper, mean):
    """v 뽑기 [S, n] -> {q: lnC [S]} (float32 vmap)."""
    import jax
    import jax.numpy as jnp
    G = grids()
    mean = mean or default_mean(G)

    def one(v):
        th = _unpack(v, hyper, G)
        return jnp.stack([lnC(th, q, G, mean) for q in OM.Q_TARGETS])
    out = np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(vdraws, jnp.float32)))
    return {q: out[:, i] for i, q in enumerate(OM.Q_TARGETS)}


def laplace_summary(pre, hyper, mean, n=4000, seed=0):
    """라플라스 사후에서 뽑아 core 모수·lnC_{1,3,10}의 평균·sd를 낸다."""
    rng = np.random.default_rng(seed)
    v = pre["loc"] + rng.standard_normal((n, len(pre["loc"]))) @ pre["chol"].T
    lc = _draws_lnc(v, hyper, mean)
    names, _ = _v_layout(hyper)
    out = {nm: {"mean": float(v[:, i].mean()), "sd": float(v[:, i].std())} for i, nm in enumerate(names)}
    for q in (1.0, 3.0, 10.0):
        out[f"lnC_{q:g}"] = {"mean": float(lc[q].mean()), "median": float(np.median(lc[q])), "sd": float(lc[q].std())}
    return out


def bench_case(work, case, season=None, chains=4, warmup=500, samples=500, K=16, target_accept=0.9, probe=False, mean_kind="temp", multistart=True):
    import jax
    import obs_table as O
    OM.setup_host(chains)
    err = OM.load_err()
    season = season or ("S30" if case in ("a", "b") else "S30")
    head, rows = O.read_cell(os.path.join(work, "obs7", "full", f"{season}.jsonl.gz"))
    H = head["H"]
    obs = OM.prepare(rows, H, season, err, K=K, stage1=True)
    jobs = OM.to_jax(obs)
    hyper = None if case == "a" else hyper_center()
    prior = {"kind": "weak"}
    mean = default_mean()
    mean_note = "임시 평균 (rbar = ln 0.3 + w, lbar = M1 kuma eta=0)"
    if mean_kind == "m1":
        # 진단용: 같은 시즌 자료에 M1 kuma를 MAP으로 맞춘 평균 곡선 (시간 재기 원인 가리기용 — 실제 쓰임이 아니다)
        p1 = M1.map_precond(jobs, {"kind": "weak"}, "kuma")
        nm1 = M1.theta_names("kuma")
        th1 = M1.theta_from_vec(list(p1["loc"][1:]), nm1, p1["loc"][0], "kuma")
        mean = mean_from_m1(th1, "kuma")
        mean_note = "진단용: 같은 시즌에 M1 kuma MAP을 맞춘 평균 곡선 (r0, r1, eta)"
        res_m1 = {"r0": float(th1["r0"]), "r1": float(th1["r1"]), "eta": [float(e) for e in th1["eta"]]}
    res = {"case": case, "season": season, "N": int(obs["N"]), "n_rows": int(obs["n"]), "counts": obs["counts"], "K": K,
           "hyper": "표집" if hyper is None else hyper, "mean": mean_note}
    if mean_kind == "m1":
        res["m1_map_theta"] = res_m1
    t0 = time.time()
    pre = map_precond(jobs, prior, hyper, mean)
    res["map_secs_incl_compile"] = round(time.time() - t0, 1)
    t0 = time.time()
    pre2 = map_precond(jobs, prior, hyper, mean)       # 같은 프로세스에서 두 번째 (jit 캐시가 안 맞으면 컴파일이 또 든다)
    res["map_secs_second_call"] = round(time.time() - t0, 1)
    if multistart:
        t0 = time.time()
        modes = multistart_map(jobs, prior, hyper, mean)
        res["multistart_secs"] = round(time.time() - t0, 1)
        res["modes_fun"] = [round(p["info"]["fun"], 3) for p in modes]
        pre = modes[0]
    res["map_info"] = pre["info"]
    lap = laplace_summary(pre, hyper, mean)
    res["laplace"] = lap
    print("MAP", res["map_secs_incl_compile"], res["map_secs_second_call"], pre["info"], flush=True)
    kw = {"obs": jobs, "prior": prior, "hyper": hyper, "mean": mean, "pre": pre}
    init = map_init(len(pre["loc"]))
    if probe:
        _, pi = OM.run_nuts(model, kw, seed=0, chains=chains, warmup=100, samples=50, target_accept=target_accept, init=init)
        print("PROBE", pi, flush=True)
        res["probe"] = pi
        return res
    # 컴파일 시간 추정: 아주 짧은 실행
    _, ci = OM.run_nuts(model, kw, seed=5, chains=chains, warmup=8, samples=4, target_accept=target_accept, init=init)
    res["compile_est_secs"] = ci["secs"]
    print("COMPILE_EST", ci["secs"], flush=True)
    sm, info = OM.run_nuts(model, kw, seed=0, chains=chains, warmup=warmup, samples=samples, target_accept=target_accept, init=init)
    res["nuts"] = info
    res["nuts_secs_excl_compile_est"] = round(info["secs"] - ci["secs"], 1)
    th_names = ["a"] + CORE + ([] if hyper is not None else HYP)
    lc_names = [f"lnC_{q:g}" for q in (1, 3, 10)]
    dg = OM.diagnose(sm, th_names + lc_names, divergences=info["divergences"])
    dgb = OM.diagnose(sm, ["beta1", "beta2"], divergences=info["divergences"])
    res["diag"] = {"rhat_max_theta_lnC": dg["rhat_max"], "ess_min_theta_lnC": dg["ess_min"],
                   "rhat_max_beta": dgb["rhat_max"], "ess_min_beta": dgb["ess_min"],
                   "conv_theta_lnC": dg["conv"], "per": dg["per"]}
    res["nuts_post"] = {nm: {"mean": float(sm[nm].mean()), "sd": float(sm[nm].std())} for nm in th_names}
    for nm in lc_names:
        v = sm[nm].ravel()
        res["nuts_post"][nm] = {"mean": float(v.mean()), "median": float(np.median(v)), "sd": float(v.std())}
    res["laplace_vs_nuts"] = {nm: {"median_diff_pct": round(100 * (math.exp(lap[nm]["median"] - float(np.median(sm[nm]))) - 1), 3),
                                   "sd_ratio": round(lap[nm]["sd"] / float(sm[nm].std()), 3)} for nm in lc_names}
    res["laplace_vs_nuts"]["core_sd_ratio_range"] = [round(min(lap[n]["sd"] / sm[n].std() for n in th_names), 3),
                                                      round(max(lap[n]["sd"] / sm[n].std() for n in th_names), 3)]
    res["total_secs"] = round(res["map_secs_incl_compile"] + info["secs"], 1)
    return res


# ---------- 시험 ----------

def test_basis():
    G = grids()
    out = {}
    have = True
    rng = np.random.default_rng(0)
    for tag, Phi, sl, lo, hi, W in (("w", G.Phi_w, G.sl_w, W_LO, W_HI, G.width_w), ("u", G.Phi_u, G.sl_u, U_LO, 1.0, G.width_u)):
        x = G.w if tag == "w" else G.u
        x0 = 0.5 * (lo + hi)
        mid = np.abs(x - x0) < 0.3 * (hi - lo) / 2 * 2      # 중앙 30% 폭
        for fr in (0.1, 0.5):
            ell, al = fr * W, 1.0
            S = matern52_spec(sl, ell, al)
            cov_h = (Phi * S[None, :]) @ Phi.T
            cov_e = matern52_cov(x[:, None] - x[None, :], ell, al)
            mm = mid[:, None] & mid[None, :]
            err = np.abs(cov_h - cov_e)[mm]
            # 표본 공분산 (20000 함수)
            b = rng.standard_normal((20000, MB))
            F = (b * np.sqrt(S)[None, :]) @ Phi.T
            ii = np.where(mid)[0]
            i0 = ii[len(ii) // 2]
            lag = [int(i0 + d) for d in (0, 3, 8, 15) if i0 + d < len(x)]
            samp = [float(np.mean(F[:, i0] * F[:, j])) for j in lag]
            exact = [float(cov_e[i0, j]) for j in lag]
            hs = [float(cov_h[i0, j]) for j in lag]
            out[f"{tag}_ell{fr}W"] = {"ell": ell, "max_abs_err_cov_center": float(err.max()),
                                      "var_center_hsgp_vs_exact": [float(np.diag(cov_h)[ii].mean()), 1.0],
                                      "lags_x": [float(x[j] - x[i0]) for j in lag], "sample": samp, "hsgp": hs, "exact": exact}
    try:
        from numpyro.contrib.hsgp.laplacian import eigenfunctions
        from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern
        import jax.numpy as jnp
        for tag, x, lo, hi, W, Phi, sl in (("w", G.w, W_LO, W_HI, G.width_w, G.Phi_w, G.sl_w), ("u", G.u, U_LO, 1.0, G.width_u, G.Phi_u, G.sl_u)):
            x0, S_, = 0.5 * (lo + hi), 0.5 * (hi - lo)
            L = CB * S_
            ph = np.asarray(eigenfunctions(jnp.asarray(x - x0), L, MB))
            ell, al = 0.3 * W, 0.7
            sd = np.asarray(diag_spectral_density_matern(2.5, al ** 2, ell, L, MB, 1))   # numpyro의 alpha는 분산 자리(곱하는 값)라 al**2
            ours = np.sqrt(matern52_spec(sl, ell, al))
            out[f"numpyro_{tag}"] = {"eigenfunction_max_abs_diff": float(np.abs(ph - Phi).max()),
                                     "sqrt_spec_max_rel_diff": float(np.max(np.abs(np.sqrt(sd).ravel() - ours) / ours))}
    except Exception as e:  # noqa
        out["numpyro_hsgp_cmp"] = f"비교 실패: {type(e).__name__}: {e}"
    return out


def truth_theta(a_mid, seed=11):
    """합성 시험의 참값: 작은 f1(길이 0.35 W, 크기 0.08)·f2(길이 0.3 W, 크기 0.12)를 HSGP 계수로 정해진 씨앗에서 뽑는다."""
    G = grids()
    rng = np.random.default_rng(seed)
    th = {"a": a_mid, "lnm": math.log(0.2), "lnsig0": math.log(0.01), "lnrho": math.log(0.13), "dJP": math.log(0.8),
          "dGL": math.log(0.85), "pJP": 0.1, "pGL": -0.1, "kJP": 0.05, "kGL": -0.05,
          "ell1": 0.35 * G.width_w, "alpha1": 0.08, "ell2": 0.3 * G.width_u, "alpha2": 0.12,
          "beta1": rng.standard_normal(MB), "beta2": rng.standard_normal(MB)}
    return th


def synth_rows(rows, H, err, th, seed=1):
    """check_obs.synth_rows를 M2 곡면으로 바꾼 것: 실제 줄 배치(점수·종류·시각 정보·서버 라벨·사이트)에 th로 %를 뽑는다."""
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)
    sel = OM.select_rows(rows, H, stage1=True)
    keep, tshot, srv = [], [], []
    dm = err["1e_server"]["unknown_row_site_dirichlet"]
    phi_true = {s: np.array([dm[s]["mean"][k] for k in OM.SERVERS]) for s in dm}
    phi_true = {s: v / v.sum() for s, v in phi_true.items()}
    for r in sel:
        t, w, why = OM.nodes_detail(r, H, err, None, 64)
        if t is None:
            continue
        tshot.append(float(rng.choice(t, p=w / w.sum())))
        site = r["site"] if r["site"] in OM.SITES else "*"
        if r["server"] in OM.SERVERS:
            lab = OM.SERVERS.index(r["server"])
            el = OM._elab(r, err)
            if rng.random() < el:
                pr = phi_true[site].copy()
                pr[lab] = 0
                srv.append(int(rng.choice(3, p=pr / pr.sum())))
            else:
                srv.append(lab)
        else:
            srv.append(int(rng.choice(3, p=phi_true[site])))
        keep.append(r)
    n = len(keep)
    y = np.log([r["score"] for r in keep])
    u = np.clip(np.array(tshot) / H, 0.02, 1.0)
    thj = {k: (jnp.asarray(v, jnp.float32)) for k, v in th.items()}
    mu, P = surface(thj, jnp.asarray(y, jnp.float32), jnp.asarray(u[:, None], jnp.float32))
    mu, P = np.asarray(mu)[:, :, 0], np.asarray(P)[:, :, 0]
    s = np.array(srv)
    mu_row, P_row = mu[s, np.arange(n)], P[s, np.arange(n)]
    sd = np.sqrt(math.exp(th["lnsig0"]) ** 2 + (math.exp(th["lnrho"]) * P_row) ** 2)
    z = mu_row + sd * rng.standard_normal(n)
    cont = rng.random(n) < 0.015
    z = np.where(cont, z + rng.laplace(0, 0.34, n), z)
    x100 = 10 ** z * 100
    pct = np.where(rng.random(n) < 0.5, np.round(x100), np.floor(x100)) / 100
    out, dropped = [], 0
    for r, p in zip(keep, pct):
        if p > 100.0 or p < 0.01:
            dropped += 1
            continue
        x = dict(r)
        x["pct"] = float(p)
        out.append(x)
    return out, {"n_rows": len(out), "dropped_out_of_range": dropped, "n_contam": int(cont.sum()),
                 "server_counts": [int((s == k).sum()) for k in range(3)]}


def test_synth(work, season="S30", chains=4, warmup=500, samples=500, free_hyper=False):
    import jax.numpy as jnp
    import obs_table as O
    OM.setup_host(chains)
    err = OM.load_err()
    head, rows = O.read_cell(os.path.join(work, "obs7", "full", f"{season}.jsonl.gz"))
    H = head["H"]
    # a: w = 0 (3%)에 해당하는 ln 점수 = 3% 근처 줄의 ln 점수 평균
    near = [math.log(r["score"]) for r in rows if r["pct"] is not None and r["score"] and 2.5 <= r["pct"] <= 3.5 and r["kind"] in OM.ABC]
    a_mid = float(np.mean(near)) if near else math.log(100.0)
    th = truth_theta(a_mid)
    srows, tinfo = synth_rows(rows, H, err, th)
    obs = OM.prepare(srows, H, season, err, K=16, stage1=True)
    jobs = OM.to_jax(obs)
    hyper = None if free_hyper else {k: th[k] for k in ("ell1", "alpha1", "ell2", "alpha2")}
    mean = default_mean()
    t0 = time.time()
    pre = map_precond(jobs, {"kind": "weak"}, hyper, mean)
    t_map = time.time() - t0
    kw = {"obs": jobs, "prior": {"kind": "weak"}, "hyper": hyper, "mean": mean, "pre": pre}
    sm, info = OM.run_nuts(model, kw, seed=0, chains=chains, warmup=warmup, samples=samples)
    G = grids()
    names = ["a"] + CORE + ([] if hyper is not None else HYP)
    truth_v = {n: th[n] for n in ["a"] + CORE}
    if hyper is None:
        truth_v.update({"lnell1": math.log(th["ell1"]), "lnalp1": math.log(th["alpha1"]),
                        "lnell2": math.log(th["ell2"]), "lnalp2": math.log(th["alpha2"])})
    rows_out, cover = {}, 0
    lc_names = [f"lnC_{q:g}" for q in (1, 3, 10)]
    thj = {k: jnp.asarray(v, jnp.float32) for k, v in th.items()}
    for q in (1, 3, 10):
        truth_v[f"lnC_{q:g}"] = float(lnC(thj, float(q), G, mean))
    for nm in names + lc_names:
        v = sm[nm].ravel()
        lo, hi = np.quantile(v, [0.025, 0.975])
        ins = bool(lo <= truth_v[nm] <= hi)
        cover += ins
        rows_out[nm] = {"truth": truth_v[nm], "mean": float(v.mean()), "sd": float(v.std()), "lo95": float(lo), "hi95": float(hi), "inside": ins}
    for nm in lc_names:
        rows_out[nm]["rel_err_C_pct"] = round(100 * (math.exp(rows_out[nm]["mean"] - rows_out[nm]["truth"]) - 1), 3)
    # f1·f2 되찾기: 사후 뽑기에서 곡선을 만들어 참값 곡선이 95% 안에 드는 격자 점 비율 (자료가 있는 범위)
    b1 = sm["beta1"].reshape(-1, MB)
    b2 = sm["beta2"].reshape(-1, MB)
    idx = np.linspace(0, len(b1) - 1, 400).astype(int)
    al = {k: np.asarray(sm[k]).ravel() if hyper is None else None for k in ()}
    def curve(Phi, sl, ell, alpha, b):
        S = np.sqrt(matern52_spec(sl, ell, alpha))
        f = (b * S[None, :]) @ Phi.T
        return f
    if hyper is None:
        e1, a1, e2, a2 = (np.exp(sm[h]).ravel()[idx] for h in HYP)
        F1 = np.stack([curve(G.Phi_w, G.sl_w, e1[i], a1[i], b1[idx[i]][None])[0] for i in range(len(idx))])
        F2 = np.stack([curve(G.Phi_u, G.sl_u, e2[i], a2[i], b2[idx[i]][None])[0] for i in range(len(idx))])
    else:
        F1 = curve(G.Phi_w, G.sl_w, th["ell1"], th["alpha1"], b1[idx])
        F2 = curve(G.Phi_u, G.sl_u, th["ell2"], th["alpha2"], b2[idx])
    F2 = F2 - F2.mean(1, keepdims=True)
    T1 = curve(G.Phi_w, G.sl_w, th["ell1"], th["alpha1"], th["beta1"][None])[0]
    T2 = curve(G.Phi_u, G.sl_u, th["ell2"], th["alpha2"], th["beta2"][None])[0]
    T2 = T2 - T2.mean()
    # 자료 범위: 줄 y에서 나온 w (KR 참값 곡선 기준) 5~95%
    cv = curves(thj, G, mean)
    wf = np.asarray(_inv_w(jnp.asarray(np.log([r["score"] for r in srows]), jnp.float32), jnp.asarray(G.w, jnp.float32), cv["lc"][0], cv["rg"][0]))
    wlo, whi = np.quantile(wf, [0.05, 0.95])
    m1_ = (G.w >= wlo) & (G.w <= whi)
    uu = np.array([r.get("h", 0) for r in srows], float) / H
    ulo, uhi = np.quantile(uu[np.isfinite(uu)], [0.05, 0.95])
    m2_ = (G.u >= max(ulo, U_LO)) & (G.u <= uhi)
    q = lambda F: (np.quantile(F, 0.025, axis=0), np.quantile(F, 0.975, axis=0))
    l1, h1 = q(F1)
    l2, h2 = q(F2)
    fcov = {"f1_cover_frac_data_range": float(((T1 >= l1) & (T1 <= h1))[m1_].mean()), "f1_w_range": [float(wlo), float(whi)],
            "f1_rmse": float(np.sqrt(((F1.mean(0) - T1)[m1_] ** 2).mean())), "f1_truth_rms": float(np.sqrt((T1[m1_] ** 2).mean())),
            "f2_cover_frac_data_range": float(((T2 >= l2) & (T2 <= h2))[m2_].mean()), "f2_u_range": [float(ulo), float(uhi)],
            "f2_rmse": float(np.sqrt(((F2.mean(0) - T2)[m2_] ** 2).mean())), "f2_truth_rms": float(np.sqrt((T2[m2_] ** 2).mean()))}
    dg = OM.diagnose(sm, names + lc_names, divergences=info["divergences"])
    return {"season": season, "free_hyper": free_hyper, "synth": tinfo, "N": obs["N"], "n": obs["n"], "map_secs": round(t_map, 1),
            "nuts": info, "param_inside": f"{cover}/{len(names) + 3}", "params": rows_out, "f_recovery": fcov,
            "rhat_max": dg["rhat_max"], "ess_min": dg["ess_min"], "conv": dg["conv"]}


def compare_multistart(work, seasons, chains=4, warmup=500, samples=500, K=16, n=8):
    """경우 b(초모수 고정, 약한 사전, 임시 평균) 단일 MAP 대 시작점 n개 MAP: 봉우리별 log 사후, 사슬별 dJP·dGL 평균, 진단, 시간.
    NUTS는 둘 다 MAP(z=0)에서 시작한다 (차이는 MAP뿐)."""
    import obs_table as O
    OM.setup_host(chains)
    err = OM.load_err()
    hyper, prior, mean = hyper_center(), {"kind": "weak"}, default_mean()
    out = {"note": "임시 평균 (rbar = ln 0.3 + w, lbar = M1 kuma eta=0), 경우 b: 초모수 고정(사전 중앙값)·약한 사전. 2026-10-29",
           "chains": chains, "warmup": warmup, "samples": samples, "n_starts": n, "seasons": {}}
    for season in seasons:
        head, rows = O.read_cell(os.path.join(work, "obs7", "full", f"{season}.jsonl.gz"))
        obs = OM.prepare(rows, head["H"], season, err, K=K, stage1=True)
        jobs = OM.to_jax(obs)
        names, _ = _v_layout(hyper)
        off = [i for i, nm in enumerate(names) if nm in M1.OFF]
        r = {"N": int(obs["N"]), "n_rows": int(obs["n"])}
        for tag in ("single", "multi"):
            t0 = time.time()
            if tag == "single":
                modes = [map_precond(jobs, prior, hyper, mean)]
            else:
                modes = multistart_map(jobs, prior, hyper, mean, n=n)
            tmap = time.time() - t0
            pre = modes[0]
            init = map_init(len(pre["loc"]))
            kw = {"obs": jobs, "prior": prior, "hyper": hyper, "mean": mean, "pre": pre}
            sm, info = OM.run_nuts(model, kw, seed=0, chains=chains, warmup=warmup, samples=samples, init=init)
            watch = ["a"] + CORE + [f"lnC_{q:g}" for q in (1, 3, 10)]
            dg = OM.diagnose(sm, watch, divergences=info["divergences"])
            r[tag] = {
                "map_secs": round(tmap, 1), "nuts_secs": info["secs"],
                "modes": [{"logpost": round(-p["info"]["fun"], 2), "server": dict(zip([names[i] for i in off], [round(float(p["loc"][i]), 3) for i in off])),
                          "saddle": p["info"].get("n_saddle"), "ok": p["info"]["ok"]} for p in modes],
                "chain_mean": {nm: [round(float(sm[nm][c].mean()), 3) for c in range(chains)] for nm in ("dJP", "dGL")},
                "rhat_max": round(dg["rhat_max"], 3), "ess_min": round(dg["ess_min"], 1), "divergences": info["divergences"],
                "tree_cap_frac": info["tree_cap_frac"], "depth_dist": info["depth_dist"],
                "worst": sorted(((k, round(v["rhat"], 3)) for k, v in dg["per"].items()), key=lambda t: -t[1])[:3],
                "lnC_3_mean": round(float(sm["lnC_3"].mean()), 4)}
            print(season, tag, json.dumps(r[tag], ensure_ascii=False, default=float), flush=True)
        out["seasons"][season] = r
        d = os.path.join(work, "m2")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "multistart_partial.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    return out


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("bench")
    b.add_argument("--work", required=True)
    b.add_argument("--case", required=True, choices=["a", "b", "c"])
    b.add_argument("--season", default=None)
    b.add_argument("--chains", type=int, default=4)
    b.add_argument("--warmup", type=int, default=500)
    b.add_argument("--samples", type=int, default=500)
    b.add_argument("--probe", action="store_true")
    b.add_argument("--tag", default="")
    b.add_argument("--mean", default="temp", choices=["temp", "m1"])
    b.add_argument("--ta", type=float, default=0.9)
    b.add_argument("--multistart", type=int, default=1, help="1이면 시작점 8개 MAP(기본), 0이면 단일 MAP")
    c = sp.add_parser("compare")
    c.add_argument("--work", required=True)
    c.add_argument("--seasons", default="S34")
    c.add_argument("--warmup", type=int, default=500)
    c.add_argument("--samples", type=int, default=500)
    t = sp.add_parser("test")
    t.add_argument("--work", required=True)
    t.add_argument("--which", required=True, choices=["basis", "synth"])
    t.add_argument("--season", default="S30")
    t.add_argument("--free-hyper", action="store_true")
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--samples", type=int, default=500)
    a = ap.parse_args()
    if a.cmd == "bench":
        OM.setup_host(a.chains)
        r = bench_case(a.work, a.case, a.season, a.chains, a.warmup, a.samples, target_accept=a.ta, probe=a.probe, mean_kind=a.mean, multistart=bool(a.multistart))
        name = f"bench_{a.case}{a.tag}.json"
    elif a.cmd == "compare":
        r = compare_multistart(a.work, a.seasons.split(","), 4, a.warmup, a.samples)
        name = "multistart.json"
    else:
        OM.setup_host(4)
        r = test_basis() if a.which == "basis" else test_synth(a.work, a.season, 4, a.warmup, a.samples, a.free_hyper)
        name = f"test_{a.which}{'_free' if a.free_hyper else ''}.json"
    d = os.path.join(a.work, "m2")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(r, ensure_ascii=False, indent=1, default=float)[:5000])


if __name__ == "__main__":
    main()
