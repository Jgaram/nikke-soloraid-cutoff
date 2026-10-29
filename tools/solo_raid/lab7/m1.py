"""M1 매개변수 곡면 (WP3b 1단계, 2026-10-29) — 곡면 함수와 합성 시험용 모형까지. 1단계 맞춤·예측 명령은 다음 단계에서 정한다.

  import m1 as M1        # sys.path.insert(0, HERE) 뒤

명세 「WP3 명세」 1·4·5절, 설계서 3절.
  최종 컷 곡선: ln C(q) = a - e^{r0} * ((q/3)^{r1} - 1) / r1     (|r1| < 1e-4이면 극한 e^{r0} * ln(q/3) 항: a - e^{r0} ln(q/3))
  점수 y의 최종 log10 %:  w_fin = ln(1 + r1 (a - y) e^{-r0}) / r1   (r1 -> 0 극한 (a - y) e^{-r0}), 로그 인자는 1e-6 아래로 자름
                          z_fin = log10 3 + w_fin / ln 10
  침수 (log10 단위): P(u) = m * p0(u; eta) / p0(0.4; eta), p0 = -log10 F(u), u = 1에서 P = 0 정확히.
                     F 둘: "f2pow" w u^p + (1-w) u^q (eta = [logit w, ln p, ln q]) · "kuma" 1 - (1-u^alpha)^beta (eta = [ln alpha, ln beta])
                     x항(선택): P * exp(beta * (z_fin - log10 3))
  지금 %: mu = z_fin - P.  서버 k(KR·JP·GL): a_k = a + delta_k, r0_k = r0 + kappa_k, m_k = m e^{psi_k} (KR은 0).

theta (dict): a, r0, r1, lnm, eta(배열 2 또는 3), lnsig0, lnrho, [beta], dJP, dGL, pJP, pGL, kJP, kGL  (d=delta, p=psi, k=kappa)
  mu, P = surface(theta, y, u, form, x_term=False)   # y[N], u[N,K] -> mu[3,N,K], P[3,N,K]  (obs_model 곡면 함수 약속)
  lnC(theta, q)                                        # KR 최종 컷 ln 점수. q(%)는 숫자 또는 배열
  model(obs, prior, form, x_term=False, keep_ll=False) # numpyro 모형. prior = {"kind": "weak"} 또는 hier.prior_for 결과
  names = theta_names(form, x_term)                    # a를 뺀 모수 이름 순서 (hier.prior_for의 names)
  WEAK[name] = (사전 평균, sd, 2단계 tau 척도)           # 명세 5절
  mu_loc, mu_scale, tau_scale = hier_args(names)       # hier.fit_window·prior_for에 넘길 배열 (mu_scale = 2 sd)
  pre = map_precond(jobs, prior, form, x_term)         # MAP + 라플라스 (float64, 약 10~25초). model(..., pre=pre)로 넘기면 사후가 표준정규에 가깝게 펴진다
                                                       # pre 없이는 날카로운 사후(점 관측 + sigma0 0.01) 때문에 NUTS 나무가 깊이 10 상한에 닿아 10분을 넘긴다 (2026-10-29 확인).
                                                       # pre를 쓴 모형이 내는 표집 이름: z, eps, phi (a·모수 이름들은 deterministic)
모형이 내는 표집 이름 (pre 없을 때): a, 그 밖 모수 이름들(eta0.. 포함), eps, phi, deterministic lnC_{q} (q = 0.5 1 2 3 5 10).
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_model as OM  # noqa: E402

LN10 = math.log(10.0)
L10_3 = math.log10(3.0)
ETA_N = {"f2pow": 3, "kuma": 2}
OFF = ("dJP", "dGL", "pJP", "pGL", "kJP", "kGL")
# 명세 5절: (1단계 약한 사전 평균, sd, 2단계 tau 척도)
WEAK = {"a": (math.log(100.0), 1.5, 0.3), "r0": (math.log(0.3), 1.0, 0.3), "r1": (1.0, 1.0, 0.5),
        "lnm": (math.log(0.2), 1.0, 0.3), "lnsig0": (math.log(0.01), 1.0, 0.5), "lnrho": (math.log(0.13), 1.0, 0.5),
        "beta": (0.0, 0.3, 0.2),
        "dJP": (math.log(0.8), 0.5, 0.2), "dGL": (math.log(0.8), 0.5, 0.2),
        "pJP": (0.0, 0.5, 0.3), "pGL": (0.0, 0.5, 0.3), "kJP": (0.0, 0.3, 0.2), "kGL": (0.0, 0.3, 0.2)}
for _i in range(3):
    WEAK[f"eta{_i}"] = (0.0, 1.5, 0.5)


def theta_names(form, x_term=False):
    """a를 뺀 모수 이름 순서."""
    n = ["r0", "r1", "lnm"] + [f"eta{i}" for i in range(ETA_N[form])] + ["lnsig0", "lnrho"]
    if x_term:
        n.append("beta")
    return n + list(OFF)


def hier_args(names):
    """hier에 넘길 (mu_loc, mu_scale = 약한 사전의 2 sd, tau_scale) 배열."""
    return (np.array([WEAK[n][0] for n in names]), np.array([2 * WEAK[n][1] for n in names]),
            np.array([WEAK[n][2] for n in names]))


def theta_from_vec(vec, names, a, form):
    """이름 순서 벡터 -> theta dict (eta는 배열로 묶음)."""
    d = {n: vec[i] for i, n in enumerate(names)}
    d["a"] = a
    d["eta"] = [d[f"eta{i}"] for i in range(ETA_N[form])]
    return d


def _wfin(a, r0, r1, y):
    """y: [N] 또는 broadcast. 점수 y의 최종 w = ln(q/3)."""
    import jax.numpy as jnp
    small = jnp.abs(r1) < 1e-4
    r1s = jnp.where(small, 1.0, r1)
    arg = 1.0 + r1s * (a - y) * jnp.exp(-r0)
    w_pow = jnp.log(jnp.maximum(arg, 1e-6)) / r1s
    w_lim = (a - y) * jnp.exp(-r0)
    return jnp.where(small, w_lim, w_pow)


def _p0(u, eta, form):
    """-log10 F(u) (u < 1에서 쓴다; u는 0.02~1 안으로 이미 잘려 있다)."""
    import jax.numpy as jnp
    lu = jnp.log(u)
    if form == "f2pow":
        w = jax_sigmoid(eta[0])
        p, q = jnp.exp(eta[1]), jnp.exp(eta[2])
        G = w * (-jnp.expm1(p * lu)) + (1 - w) * (-jnp.expm1(q * lu))
    elif form == "kuma":
        al, be = jnp.exp(eta[0]), jnp.exp(eta[1])
        s = jnp.maximum(-jnp.expm1(al * lu), 1e-30)
        G = jnp.exp(be * jnp.log(s))
    else:
        raise ValueError(form)
    G = jnp.clip(G, 0.0, 1.0 - 1e-7)
    return -jnp.log1p(-G) / LN10


def jax_sigmoid(x):
    import jax
    return jax.nn.sigmoid(x)


def _eta_of(theta, form):
    return [theta["eta"][i] for i in range(ETA_N[form])]


def flood(m, eta, u, form):
    """침수 P(u) [같은 모양], u = 1에서 정확히 0."""
    import jax.numpy as jnp
    us = jnp.where(u >= 1.0, 0.5, u)
    p0 = _p0(us, eta, form)
    p04 = _p0(jnp.float32(0.4), eta, form)
    return jnp.where(u >= 1.0, 0.0, m * p0 / jnp.maximum(p04, 1e-12))


def surface(theta, y, u, form, x_term=False):
    """-> (mu[3,N,K], P[3,N,K]) (서버 KR·JP·GL). theta 약속은 머리 참조."""
    import jax.numpy as jnp
    z = jnp.zeros(())
    dl = [z, theta["dJP"], theta["dGL"]]
    ps = [z, theta["pJP"], theta["pGL"]]
    kp = [z, theta["kJP"], theta["kGL"]]
    eta = _eta_of(theta, form)
    mus, Ps = [], []
    for k in range(3):
        a_k = theta["a"] + dl[k]
        r0_k = theta["r0"] + kp[k]
        m_k = jnp.exp(theta["lnm"] + ps[k])
        wf = _wfin(a_k, r0_k, theta["r1"], y)                  # [N]
        zf = L10_3 + wf / LN10
        P = flood(m_k, eta, u, form)                           # [N,K]
        if x_term:
            P = P * jnp.exp(theta["beta"] * wf / LN10)[:, None]
        mus.append(zf[:, None] - P)
        Ps.append(P)
    return jnp.stack(mus), jnp.stack(Ps)


def lnC(theta, q):
    """KR 최종 컷 ln 점수 (q = 목표 %)."""
    import jax.numpy as jnp
    q = jnp.asarray(q, jnp.float32)
    r0, r1 = theta["r0"], theta["r1"]
    small = jnp.abs(r1) < 1e-4
    r1s = jnp.where(small, 1.0, r1)
    lq = jnp.log(q / 3.0)
    pw = (jnp.exp(r1s * lq) - 1.0) / r1s
    return theta["a"] - jnp.exp(r0) * jnp.where(small, lq, pw)


def _prior_logp(vec, names, prior):
    """사전 로그 밀도 (vec = [a, *names 순서 모수])."""
    import jax.numpy as jnp
    import numpyro.distributions as dist
    if prior.get("kind") == "weak":
        mu = jnp.asarray([WEAK["a"][0]] + [WEAK[n][0] for n in names], jnp.float32)
        sd = jnp.asarray([WEAK["a"][1]] + [WEAK[n][1] for n in names], jnp.float32)
        return dist.Normal(mu, sd).log_prob(vec).sum()
    am, av = prior["a"]["mean"], prior["a"]["var"]
    ix = [prior["names"].index(n) for n in names]
    mean = jnp.asarray(np.asarray(prior["mean"])[ix], jnp.float32)
    cov = jnp.asarray(np.asarray(prior["cov"])[np.ix_(ix, ix)], jnp.float32) + 1e-6 * jnp.eye(len(names))
    return (dist.Normal(am, math.sqrt(av)).log_prob(vec[0])
            + dist.MultivariateNormal(mean, covariance_matrix=cov).log_prob(vec[1:]))


def model(obs, prior, form, x_term=False, keep_ll=False, pre=None):
    """numpyro 모형. prior {"kind": "weak"} 또는 hier.prior_for 결과 (names가 theta_names와 같아야 한다 — 순서는 이름으로 맞춘다).
    pre = {"loc": [n], "chol": [n,n]} (map_precond의 결과)를 주면 v = [a, 모수...] = loc + chol z 로 z를 표집한다 (사후가 거의 표준정규라
    NUTS 예열이 빠르다. 같은 사후, 표집 이름은 z·eps·phi와 deterministic a·모수 이름들). 안 주면 사전에서 곧바로 표집한다."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import constraints
    names = theta_names(form, x_term)
    n = len(names) + 1
    if pre is not None:
        z = numpyro.sample("z", dist.ImproperUniform(constraints.real_vector, (), (n,)))
        vec = jnp.asarray(pre["loc"], jnp.float32) + jnp.asarray(pre["chol"], jnp.float32) @ z
        numpyro.factor("prior", _prior_logp(vec, names, prior))
        a = numpyro.deterministic("a", vec[0])
        vals = [numpyro.deterministic(nm, vec[i + 1]) for i, nm in enumerate(names)]
    elif prior.get("kind") == "weak":
        a = numpyro.sample("a", dist.Normal(WEAK["a"][0], WEAK["a"][1]))
        vals = [numpyro.sample(nm, dist.Normal(WEAK[nm][0], WEAK[nm][1])) for nm in names]
    else:
        am, av = prior["a"]["mean"], prior["a"]["var"]
        a = numpyro.sample("a", dist.Normal(am, math.sqrt(av)))
        ix = [prior["names"].index(nm) for nm in names]
        mean = jnp.asarray(np.asarray(prior["mean"])[ix], jnp.float32)
        cov = jnp.asarray(np.asarray(prior["cov"])[np.ix_(ix, ix)], jnp.float32) + 1e-6 * jnp.eye(len(names))
        vec = numpyro.sample("theta_vec", dist.MultivariateNormal(mean, covariance_matrix=cov))
        vals = [numpyro.deterministic(nm, vec[i]) for i, nm in enumerate(names)]
    th = theta_from_vec(vals, names, a, form)
    mu, P = surface(th, obs["y"], obs["u"], form, x_term)
    OM.observe(obs, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]), keep_ll)
    for q in OM.Q_TARGETS:
        numpyro.deterministic(f"lnC_{q:g}", lnC(th, q))


def map_precond(obs, prior, form, x_term=False, start=None, iters=400, temps=(0.003, 0.03, 0.3, 1.0)):
    """MAP + 라플라스: v = [a, 모수...]의 (사후 최빈값 loc, 공분산의 콜레스키 chol). eps·phi는 사전 평균에 고정해 구한다.
    float32로는 L-BFGS가 일찍 서므로 이 함수 안에서만 float64 (jax enable_x64 문맥). 우도에 온도 t를 곱해 t = 0.003 -> 1 로 차례로 풀어
    (앞 답을 다음 시작점으로) 날카로운 사후에도 닿는다. Hessian의 고윳값은 아래서 1e-6으로 자른다. -> dict(loc, chol, info)"""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    names = theta_names(form, x_term)
    with jax.enable_x64(True):
        o = dict(obs)
        for k in OM.NUMERIC:
            a = np.asarray(obs[k])
            o[k] = jnp.asarray(a, jnp.int64 if a.dtype.kind in "iu" else jnp.float64)
        eps0 = obs["eps_ab"][0] / (obs["eps_ab"][0] + obs["eps_ab"][1])
        phi0 = o["dirichlet"] / o["dirichlet"].sum(1, keepdims=True)

        def negpost(v, temp):
            th = theta_from_vec([v[i + 1] for i in range(len(names))], names, v[0], form)
            mu, P = surface(th, o["y"], o["u"], form, x_term)
            ll = OM.row_loglik(o, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]), eps0, phi0)
            return -(temp * ll.sum() + _prior_logp64(v, names, prior))

        vg = jax.jit(jax.value_and_grad(negpost))
        hess = jax.jit(jax.hessian(negpost))
        if start is None:
            if prior.get("kind") == "weak":
                start = np.array([WEAK["a"][0]] + [WEAK[nm][0] for nm in names])
            else:
                ix = [prior["names"].index(nm) for nm in names]
                start = np.concatenate([[prior["a"]["mean"]], np.asarray(prior["mean"])[ix]])
        x = np.asarray(start, float)
        nit = 0
        for temp in temps:
            def f(xx, temp=temp):
                v, g = vg(jnp.asarray(xx), temp)
                v, g = float(v), np.asarray(g, float)
                return (v, g) if np.isfinite(v) and np.isfinite(g).all() else (1e12, np.zeros_like(g))
            res = minimize(f, x, jac=True, method="L-BFGS-B", options={"maxiter": iters, "maxfun": 3 * iters})
            x, nit = res.x, nit + res.nit
        H = np.asarray(hess(jnp.asarray(x), 1.0), float)
    H = 0.5 * (H + H.T)
    w, Q = np.linalg.eigh(H)
    cov = (Q / np.maximum(w, 1e-6)) @ Q.T
    chol = np.linalg.cholesky(0.5 * (cov + cov.T) + 1e-12 * np.eye(len(x)))
    return {"loc": x, "chol": chol, "info": {"fun": float(res.fun), "nit": nit, "ok": bool(res.success),
                                            "min_eig": float(w.min()), "max_eig": float(w.max())}}


def _prior_logp64(vec, names, prior):
    """_prior_logp의 float64판 (map_precond용; jnp 배열 dtype을 그대로 따른다)."""
    import jax.numpy as jnp
    if prior.get("kind") == "weak":
        mu = jnp.asarray([WEAK["a"][0]] + [WEAK[n][0] for n in names])
        sd = jnp.asarray([WEAK["a"][1]] + [WEAK[n][1] for n in names])
        return (-0.5 * ((vec - mu) / sd) ** 2 - jnp.log(sd) - 0.5 * math.log(2 * math.pi)).sum()
    ix = [prior["names"].index(n) for n in names]
    mean = jnp.asarray(np.asarray(prior["mean"])[ix])
    cov = jnp.asarray(np.asarray(prior["cov"])[np.ix_(ix, ix)]) + 1e-6 * jnp.eye(len(names))
    am, av = prior["a"]["mean"], prior["a"]["var"]
    r = vec[1:] - mean
    ch = jnp.linalg.cholesky(cov)
    zz = jnp.linalg.solve(ch, r)
    return (-0.5 * ((vec[0] - am) ** 2 / av + math.log(av) + math.log(2 * math.pi))
            - 0.5 * (zz ** 2).sum() - jnp.log(jnp.diag(ch)).sum() - 0.5 * len(names) * math.log(2 * math.pi))
