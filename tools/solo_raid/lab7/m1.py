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
import common as C  # noqa: E402
import obs_model as OM  # noqa: E402

LN10 = math.log(10.0)
L10_3 = math.log10(3.0)
ETA_N = {"f2pow": 3, "kuma": 2}
OFF = ("dJP", "dGL", "pJP", "pGL", "kJP", "kGL")
# 서버 성분 처방 (2026-10-29 수렴 진단용; 기본은 명세 그대로). SRV_MODE "delta"면 일·글 성분은 점수 비 δ만 (ψ = κ = 0), SRV_SCALE은 δ·ψ·κ 사전 sd에 곱한다
SRV_MODE = "full"
SRV_SCALE = 1.0


def set_server_variant(mode="full", scale=1.0):
    global SRV_MODE, SRV_SCALE
    SRV_MODE, SRV_SCALE = mode, float(scale)


def _wsd(n):
    """약한 사전 sd (서버 보조 모수에는 SRV_SCALE을 곱한다)."""
    return WEAK[n][1] * (SRV_SCALE if n in OFF else 1.0)
# 명세 5절: (1단계 약한 사전 평균, sd, 2단계 tau 척도)
WEAK = {"a": (math.log(100.0), 1.5, 0.3), "r0": (math.log(0.3), 1.0, 0.3), "r1": (1.0, 1.0, 0.5),
        "lnm": (math.log(0.2), 1.0, 0.3), "lnsig0": (math.log(0.01), 1.0, 0.5), "lnrho": (math.log(0.13), 1.0, 0.5),
        "beta": (0.0, 0.3, 0.2),
        "r2": (0.0, 0.3, 0.2),   # 2026-10-29 WP3b 세션 정함 (명세에 없던 값; 2단계 tau 척도 0.2도)
        "dJP": (math.log(0.8), 0.5, 0.2), "dGL": (math.log(0.8), 0.5, 0.2),
        "pJP": (0.0, 0.5, 0.3), "pGL": (0.0, 0.5, 0.3), "kJP": (0.0, 0.3, 0.2), "kGL": (0.0, 0.3, 0.2)}
for _i in range(3):
    WEAK[f"eta{_i}"] = (0.0, 1.5, 0.5)
# 2026-10-29 WP3b 세션: f2pow의 eta = [logit w, ln p, ln(q/p)], q = p exp(|eta2|) -> p <= q. 이전 (w, p, q)는 (1-w, q, p)와 같은 함수라
# 거울 봉우리가 둘이어서 R-hat 4~6이 났다. eta = 0이면 p = q = 1, F(u) = u 그대로; 사전은 그대로 N(0, 1.5^2).


def theta_names(form, x_term=False, r2=False):
    """a를 뺀 모수 이름 순서 (r2=True면 r1 뒤에 r2 — 2차 컷 기울기 변형)."""
    n = ["r0", "r1"] + (["r2"] if r2 else []) + ["lnm"] + [f"eta{i}" for i in range(ETA_N[form])] + ["lnsig0", "lnrho"]
    if x_term:
        n.append("beta")
    return n + (list(OFF) if SRV_MODE == "full" else ["dJP", "dGL"])


def hier_args(names):
    """hier에 넘길 (mu_loc, mu_scale = 약한 사전의 2 sd, tau_scale) 배열."""
    return (np.array([WEAK[n][0] for n in names]), np.array([2 * WEAK[n][1] for n in names]),
            np.array([WEAK[n][2] for n in names]))


def theta_from_vec(vec, names, a, form):
    """이름 순서 벡터 -> theta dict (eta는 배열로 묶음)."""
    import jax.numpy as jnp
    d = {n: vec[i] for i, n in enumerate(names)}
    for n in OFF:
        d.setdefault(n, jnp.zeros(()))
    if form == "f2pow":
        d["eta2"] = jnp.abs(d["eta2"])
    d["a"] = a
    d["eta"] = [d[f"eta{i}"] for i in range(ETA_N[form])]
    return d


WGRID_N = 60   # 0 양쪽 각 60점 (합 121점 ~ 명세의 120점)


def _r2_grid(a, r0, r1, r2):
    """w 격자 [ln(0.1/3), ln(40/3)]에서 ln C(w) = a - integral_0^w exp(r0 + r1 v + r2 v^2) dv (누적 사다리꼴, w = 0에서 a)."""
    import jax.numpy as jnp
    wn = jnp.linspace(math.log(0.1 / 3.0), 0.0, WGRID_N + 1)
    wp = jnp.linspace(0.0, math.log(40.0 / 3.0), WGRID_N + 1)

    def cum(w):
        rr = jnp.exp(jnp.clip(r0 + r1 * w + r2 * w * w, -30.0, 30.0))
        seg = 0.5 * (rr[1:] + rr[:-1]) * (w[1:] - w[:-1])
        return jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    cp = cum(wp)
    cn = cum(wn[::-1])[::-1]       # w < 0에서 적분 I(w) (음수), wn[-1] = 0 에서 0
    wg = jnp.concatenate([wn[:-1], wp])
    cg = jnp.concatenate([cn[:-1], cp])
    return wg, a - cg


def _wfin(a, r0, r1, y, r2=None):
    """y: [N] 또는 broadcast. 점수 y의 최종 w = ln(q/3). r2가 있으면 격자 보간 (격자 밖은 끝 기울기로 직선 연장)."""
    import jax.numpy as jnp
    if r2 is not None:
        wg, lg = _r2_grid(a, r0, r1, r2)
        xp, fp = lg[::-1], wg[::-1]                 # ln C 오름차순
        w = jnp.interp(y, xp, fp)
        s_lo = -jnp.exp(-(r0 + r1 * wg[-1] + r2 * wg[-1] ** 2))   # dw/dlnC at 큰 w
        s_hi = -jnp.exp(-(r0 + r1 * wg[0] + r2 * wg[0] ** 2))
        w = jnp.where(y < xp[0], fp[0] + s_lo * (y - xp[0]), w)
        return jnp.where(y > xp[-1], fp[-1] + s_hi * (y - xp[-1]), w)
    small = jnp.abs(r1) < 1e-4
    r1s = jnp.where(small, 1.0, r1)
    arg = 1.0 + r1s * (a - y) * jnp.exp(-r0)
    w_pow = jnp.log(jnp.maximum(arg, 1e-6)) / r1s
    w_lim = (a - y) * jnp.exp(-r0)
    return jnp.where(small, w_lim, w_pow)


def _log_f(u, eta, form):
    """log F(u), u in (0, 1) — 자르기 없이 안정하게 (2026-10-29: 이전 G 클립 1-1e-7이 float32 꺾임이라 발산의 원인으로 보였다)."""
    import jax
    import jax.numpy as jnp
    lu = jnp.log(u)
    if form == "f2pow":
        p = jnp.exp(eta[1])
        q = p * jnp.exp(eta[2])     # eta[2] = ln(q/p) >= 0 (theta_from_vec가 |.|를 씌운다) -> p <= q
        return jnp.logaddexp(jax.nn.log_sigmoid(eta[0]) + p * lu, jax.nn.log_sigmoid(-eta[0]) + q * lu)
    if form == "kuma":
        al, be = jnp.exp(eta[0]), jnp.exp(eta[1])
        L = al * lu                                   # <= 0
        tiny = L < -15.0
        Ls = jnp.where(tiny, -1.0, L)
        A = jnp.where(Ls < -0.6931472, jnp.log1p(-jnp.exp(jnp.minimum(Ls, -0.6931472))),
                      jnp.log(-jnp.expm1(jnp.minimum(Ls, -1e-12))))   # log(1 - e^L)
        general = jnp.log(-jnp.expm1(jnp.minimum(be * A, -1e-30)))
        return jnp.where(tiny, jnp.log(be) + L, general)
    raise ValueError(form)


def _p0(u, eta, form):
    """-log10 F(u) (u < 1에서 쓴다; u는 0.02~1 안으로 이미 잘려 있다)."""
    return -_log_f(u, eta, form) / LN10


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
        wf = _wfin(a_k, r0_k, theta["r1"], y, theta.get("r2"))   # [N]
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
    if "r2" in theta:
        wg, lg = _r2_grid(theta["a"], r0, r1, theta["r2"])
        return jnp.interp(jnp.log(q / 3.0), wg, lg)
    small = jnp.abs(r1) < 1e-4
    r1s = jnp.where(small, 1.0, r1)
    lq = jnp.log(q / 3.0)
    pw = (jnp.exp(r1s * lq) - 1.0) / r1s
    return theta["a"] - jnp.exp(r0) * jnp.where(small, lq, pw)


def _fold(nm, x, form):
    """f2pow의 eta2 = ln(q/p)는 |.|로 접어 p <= q를 박는다 (거울 봉우리 (w,p,q) <-> (1-w,q,p) 제거). 사전은 접기 전 값에 N(0, 1.5^2)."""
    import jax.numpy as jnp
    return jnp.abs(x) if (form == "f2pow" and nm == "eta2") else x


def _prior_logp(vec, names, prior):
    """사전 로그 밀도 (vec = [a, *names 순서 모수])."""
    import jax.numpy as jnp
    import numpyro.distributions as dist
    if prior.get("kind") == "weak":
        mu = jnp.asarray([WEAK["a"][0]] + [WEAK[n][0] for n in names], jnp.float32)
        sd = jnp.asarray([WEAK["a"][1]] + [_wsd(n) for n in names], jnp.float32)
        return dist.Normal(mu, sd).log_prob(vec).sum()
    am, av = prior["a"]["mean"], prior["a"]["var"]
    ix = [prior["names"].index(n) for n in names]
    mean = jnp.asarray(np.asarray(prior["mean"])[ix], jnp.float32)
    cov = jnp.asarray(np.asarray(prior["cov"])[np.ix_(ix, ix)], jnp.float32) + 1e-6 * jnp.eye(len(names))
    return (dist.Normal(am, math.sqrt(av)).log_prob(vec[0])
            + dist.MultivariateNormal(mean, covariance_matrix=cov).log_prob(vec[1:]))


def model(obs, prior, form, x_term=False, keep_ll=False, pre=None, r2=False):
    """numpyro 모형. prior {"kind": "weak"} 또는 hier.prior_for 결과 (names가 theta_names와 같아야 한다 — 순서는 이름으로 맞춘다).
    pre = {"loc": [n], "chol": [n,n]} (map_precond의 결과)를 주면 v = [a, 모수...] = loc + chol z 로 z를 표집한다 (사후가 거의 표준정규라
    NUTS 예열이 빠르다. 같은 사후, 표집 이름은 z·eps·phi와 deterministic a·모수 이름들). 안 주면 사전에서 곧바로 표집한다."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import constraints
    names = theta_names(form, x_term, r2)
    n = len(names) + 1
    eps_phi = None
    if pre is not None and len(pre["loc"]) > n:
        # 전체 사전 펴기: v = [a, 모수..., logit eps, phi의 stick-breaking 8개] (map_precond가 eps·phi도 같이 푼 것)
        from numpyro.distributions.transforms import StickBreakingTransform
        z = numpyro.sample("z", dist.ImproperUniform(constraints.real_vector, (), (len(pre["loc"]),)))
        full = jnp.asarray(pre["loc"], jnp.float32) + jnp.asarray(pre["chol"], jnp.float32) @ z
        vec, e_logit, sb = full[:n], full[n], full[n + 1:].reshape(4, 2)
        eps_v = jax.nn.sigmoid(e_logit)
        sbt = StickBreakingTransform()
        phi_v = sbt(sb)
        a0, b0 = [jnp.float32(x) for x in obs["eps_ab"]]
        lp_ep = (dist.Beta(a0, b0).log_prob(eps_v) + jnp.log(eps_v) + jnp.log1p(-eps_v)
                 + dist.Dirichlet(obs["dirichlet"]).log_prob(phi_v).sum() + sbt.log_abs_det_jacobian(sb, phi_v).sum())
        numpyro.factor("prior", _prior_logp(vec, names, prior) + lp_ep)
        numpyro.deterministic("eps", eps_v)
        numpyro.deterministic("phi", phi_v)
        eps_phi = (eps_v, phi_v)
        a = numpyro.deterministic("a", vec[0])
        vals = [numpyro.deterministic(nm, _fold(nm, vec[i + 1], form)) for i, nm in enumerate(names)]
    elif pre is not None:
        z = numpyro.sample("z", dist.ImproperUniform(constraints.real_vector, (), (n,)))
        vec = jnp.asarray(pre["loc"], jnp.float32) + jnp.asarray(pre["chol"], jnp.float32) @ z
        numpyro.factor("prior", _prior_logp(vec, names, prior))
        a = numpyro.deterministic("a", vec[0])
        vals = [numpyro.deterministic(nm, _fold(nm, vec[i + 1], form)) for i, nm in enumerate(names)]
    elif prior.get("kind") == "weak":
        a = numpyro.sample("a", dist.Normal(WEAK["a"][0], WEAK["a"][1]))
        vals = [numpyro.sample(nm, dist.Normal(WEAK[nm][0], _wsd(nm))) for nm in names]
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
    OM.observe(obs, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]), keep_ll,
               eps=None if eps_phi is None else eps_phi[0], phi=None if eps_phi is None else eps_phi[1])
    for q in OM.Q_TARGETS:
        numpyro.deterministic(f"lnC_{q:g}", lnC(th, q))


def map_precond(obs, prior, form, x_term=False, start=None, iters=1500, temps=(0.003, 0.03, 0.3, 1.0), r2=False, full=True):
    """MAP + 라플라스: v = [a, 모수...]의 (사후 최빈값 loc, 공분산의 콜레스키 chol). eps·phi는 사전 평균에 고정해 구한다.
    float32로는 L-BFGS가 일찍 서므로 이 함수 안에서만 float64 (jax enable_x64 문맥). 우도에 온도 t를 곱해 t = 0.003 -> 1 로 차례로 풀어
    (앞 답을 다음 시작점으로) 날카로운 사후에도 닿는다. Hessian의 고윳값은 아래서 1e-6으로 자른다. -> dict(loc, chol, info)
    full=True면 eps(logit)·phi(stick-breaking 8개)도 같이 풀어 v에 붙인다 (model이 길이로 알아본다). 그렇지 않으면 eps·phi를 사전 평균에 고정하는데,
    그러면 오염이 많은 시즌(S38)에서 MAP이 사후와 20 sd 어긋나 사슬이 갈렸다 (2026-10-29)."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    names = theta_names(form, x_term, r2)
    with jax.enable_x64(True):
        o = dict(obs)
        for k in OM.NUMERIC:
            a = np.asarray(obs[k])
            o[k] = jnp.asarray(a, jnp.int64 if a.dtype.kind in "iu" else jnp.float64)
        eps0 = obs["eps_ab"][0] / (obs["eps_ab"][0] + obs["eps_ab"][1])
        phi0 = o["dirichlet"] / o["dirichlet"].sum(1, keepdims=True)

        import numpyro.distributions as dist
        from numpyro.distributions.transforms import StickBreakingTransform
        sbt = StickBreakingTransform()
        n1 = len(names) + 1
        ea, eb = obs["eps_ab"]

        def negpost(vf, temp):
            v = vf[:n1]
            th = theta_from_vec([v[i + 1] for i in range(len(names))], names, v[0], form)
            mu, P = surface(th, o["y"], o["u"], form, x_term)
            if full:
                eps_v = jax.nn.sigmoid(vf[n1])
                sb = vf[n1 + 1:].reshape(4, 2)
                phi_v = sbt(sb)
                lp_ep = (dist.Beta(ea, eb).log_prob(eps_v) + jnp.log(eps_v) + jnp.log1p(-eps_v)
                         + dist.Dirichlet(o["dirichlet"]).log_prob(phi_v).sum() + sbt.log_abs_det_jacobian(sb, phi_v).sum())
            else:
                eps_v, phi_v, lp_ep = eps0, phi0, 0.0
            ll = OM.row_loglik(o, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]), eps_v, phi_v)
            return -(temp * ll.sum() + _prior_logp64(v, names, prior) + lp_ep)

        vg = jax.jit(jax.value_and_grad(negpost))
        hess = jax.jit(jax.hessian(negpost))
        if start is None:
            if prior.get("kind") == "weak":
                start = np.array([WEAK["a"][0]] + [WEAK[nm][0] for nm in names])
            else:
                ix = [prior["names"].index(nm) for nm in names]
                start = np.concatenate([[prior["a"]["mean"]], np.asarray(prior["mean"])[ix]])
        x = np.asarray(start, float)
        if full:
            e0 = float(np.log(eps0 / (1 - eps0)))
            sb0 = np.asarray(sbt.inv(phi0)).reshape(-1)
            x = np.concatenate([x, [e0], sb0])
        nit = 0
        for temp in temps:
            def f(xx, temp=temp):
                v, g = vg(jnp.asarray(xx), temp)
                v, g = float(v), np.asarray(g, float)
                return (v, g) if np.isfinite(v) and np.isfinite(g).all() else (1e12, np.zeros_like(g))
            res = minimize(f, x, jac=True, method="L-BFGS-B", options={"maxiter": iters, "maxfun": 3 * iters})
            x, nit = res.x, nit + res.nit
        f1 = lambda xx: float(vg(jnp.asarray(xx), 1.0)[0])

        def polish(x):
            """뉴턴 다듬기: L-BFGS는 조건수 1e7 넘는 사후에서 일찍 선다."""
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
        for _ in range(3):   # 안장점: Hessian 최소 고윳값 <= 0이면 그 방향으로 +-0.5 밀어 다시 푼다 (2026-10-29 M3 세션 제보)
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
    return {"loc": x, "chol": chol, "info": {"fun": float(res.fun), "nit": nit, "ok": bool(res.success),
                                            "min_eig": float(w.min()), "max_eig": float(w.max()), "newton": n_newton, "saddle_pushes": n_saddle}}


def _prior_logp64(vec, names, prior):
    """_prior_logp의 float64판 (map_precond용; jnp 배열 dtype을 그대로 따른다)."""
    import jax.numpy as jnp
    if prior.get("kind") == "weak":
        mu = jnp.asarray([WEAK["a"][0]] + [WEAK[n][0] for n in names])
        sd = jnp.asarray([WEAK["a"][1]] + [_wsd(n) for n in names])
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


# ======================================================================
# 명령줄 (설계서 2): stage1 · compare   (prior·sbc·predict 등은 다음 단계)
#   python3 -I tools/solo_raid/lab7/m1.py stage1 --obs W1/obs7/full --seasons S22,S26 --form kuma [--x-term] [--r2] [--tag T]
#           --out r7/stage1_M1_{tag}_D1.json --ll-dir W1/m1/ll [--resume]
#   python3 -I tools/solo_raid/lab7/m1.py compare --ll-dir W1/m1/ll --a kuma --b f2pow --seasons S22,... --out r7/wp3b/form_kuma_vs_f2pow.json
# 실행 기록은 ll-dir의 부모 폴더 attempts.log에 덧붙는다.
# ======================================================================
import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

DELAY_RULE_NOTE = "S19·S20 -> S21 · S21~S41 -> 다음 시즌 키 · S42 -> 전체 값 (그 시즌까지의 A 사진)"


def delay_key_stage1(season):
    n = int(season.lstrip("S"))
    if n >= 42:
        return None
    return "S21" if n <= 20 else f"S{n + 1}"


def tag_of(form, x_term, r2):
    return form + ("_b" if x_term else "") + ("_r2" if r2 else "")


def _log_attempt(ll_dir, text):
    d = os.path.dirname(os.path.abspath(ll_dir.rstrip("/")))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "attempts.log"), "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {text}\n")


def _git_head():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def multistart_map(jobs, prior, form, x_term=False, r2=False, n=8, seed=0, scatter=0.5):
    """시작점 n개의 MAP (첫째는 기본 시작, 나머지는 그 해에서 서버 보조 모수만 N(0, scatter^2)로 흩음). -> fun 오름차순 [pre dict]."""
    names = theta_names(form, x_term, r2)
    base = map_precond(jobs, prior, form, x_term, r2=r2)
    out = [base]
    rng = np.random.default_rng(seed)
    n1 = len(names) + 1
    idx = [i + 1 for i, nm in enumerate(names) if nm in OFF]
    for _ in range(n - 1):
        st = np.array(base["loc"], float)
        st[idx] += rng.normal(0, scatter, len(idx))
        out.append(map_precond(jobs, prior, form, x_term, start=st[:n1], r2=r2))
    return sorted(out, key=lambda p: p["info"]["fun"])


def _init_site(site=None, n=0, med=None):
    import jax.numpy as jnp
    if site["type"] == "sample" and not site["is_observed"] and site["name"] == "z":
        return jnp.zeros(n, jnp.float32)
    return med(site)


def map_init(n):
    """init_strategy: z는 0(= MAP), 나머지(eps·phi)는 init_to_median."""
    import functools
    from numpyro.infer import init_to_median
    return functools.partial(_init_site, n=n, med=init_to_median(num_samples=15))


def fit_one(rows, H, season, err, form, x_term, r2, K=16, delay_key=None, stage1=True, prior=None, seed=0,
            keep_ll=True, retry=True, max_tree_depth=10, multistart=True, warmup=1000, samples=1000, retry_warmup=1500):
    """한 칸 맞춤: prepare -> map_precond -> fit_with_retry. -> dict(samples, info, diag, obs, jobs, pre, names)"""
    import obs_table  # noqa: F401
    obs = OM.prepare(rows, H, season, err, K=K, delay_key=delay_key, stage1=stage1)
    jobs = OM.to_jax(obs)
    prior = prior or {"kind": "weak"}
    names = theta_names(form, x_term, r2)
    t0 = time.time()
    if multistart:
        modes = multistart_map(jobs, prior, form, x_term, r2=r2)
        rep = sum(1 for p in modes if p["info"]["fun"] - modes[0]["info"]["fun"] < 0.5)
        if rep < 2:   # 최고 봉우리가 시작점 둘 이상에서 안 나오면 8개 더 (명세 7절 「WP3b M1 세션이 더함」)
            modes = sorted(modes + multistart_map(jobs, prior, form, x_term, r2=r2, seed=1), key=lambda p: p["info"]["fun"])
            rep = sum(1 for p in modes if p["info"]["fun"] - modes[0]["info"]["fun"] < 0.5)
        pre = modes[0]
        pre["n_starts"], pre["n_best_repeat"] = len(modes), rep
        pre["modes"] = [{"fun": p["info"]["fun"], "server": [float(p["loc"][1 + i]) for i, nm in enumerate(names) if nm in OFF]} for p in modes]
    else:
        pre = map_precond(jobs, prior, form, x_term, r2=r2)
    map_secs = time.time() - t0
    watch = ["a"] + [n for n in names if n not in OFF] + [f"lnC_{q:g}" for q in OM.Q_TARGETS]   # 수렴 판정: core + lnC (서버 보조 모수는 따로 적기만)
    # 모든 사슬을 MAP(z = 0)에서 시작한다: 서버 성분(JP·GL) 모수의 국소 최빈값에 사슬이 갈려 박히는 것(R-hat 1.6, ESS 3)을 막는다
    init = map_init(len(pre["loc"]))
    kw = {"obs": jobs, "prior": prior, "form": form, "x_term": x_term, "keep_ll": keep_ll, "pre": pre, "r2": r2}
    if retry:
        sm, info, dg = OM.fit_with_retry(model, kw, names=watch, seed=seed, init=init, max_tree_depth=max_tree_depth,
                                         warmup=warmup, samples=samples, retry_warmup=retry_warmup)
    else:
        sm, info = OM.run_nuts(model, kw, seed=seed, init=init, max_tree_depth=max_tree_depth, warmup=warmup, samples=samples)
        dg = OM.diagnose(sm, watch, divergences=info["divergences"])
        info["retried"] = False
    offs = [n for n in names if n in OFF]
    if offs:
        dg["off"] = OM.diagnose(sm, offs, divergences=0)
    info["map_secs"] = round(map_secs, 1)
    info["n_starts"], info["n_best_repeat"] = pre.get("n_starts"), pre.get("n_best_repeat")
    info["map"] = pre["info"]
    return {"samples": sm, "info": info, "diag": dg, "obs": obs, "jobs": jobs, "pre": pre, "names": names}


def theta_surface_fn(names, form, x_term, r2):
    def fn(draw, jobs):
        import jax.numpy as jnp
        th = theta_from_vec([draw[nm] for nm in names], names, draw["a"], form)
        mu, P = surface(th, jobs["y"], jobs["u"], form, x_term)
        return mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"])
    return fn


def core_diag(dg, names, div):
    """수렴 판정은 core 모수(a·r·lnm·eta·sigma·[beta·r2])와 lnC_q (서버 보조 모수 d·p·k는 뺌 — 따로 적기만). conv_core = R-hat < 1.01 · ESS >= 400 · 발산 0."""
    rh = max(v["rhat"] for v in dg["per"].values())
    es = min(v["ess"] for v in dg["per"].values())
    off = dg.get("off")
    return {"rhat_max": rh, "ess_min": es, "conv_core": bool(rh < 1.01 and es >= 400 and div == 0),
            "offsets_rhat_max": off["rhat_max"] if off else None, "offsets_ess_min": off["ess_min"] if off else None}


def cmd_stage1(a):
    import arviz as az
    import obs_table as O
    OM.setup_host(4)
    err = OM.load_err()
    form, x_term, r2 = a.form, a.x_term, a.r2
    set_server_variant(a.srv, a.srv_scale)
    tag = a.tag or tag_of(form, x_term, r2)
    names = theta_names(form, x_term, r2)
    out = {"about": "M1 1단계: 끝난 시즌 전체 표를 약한 사전으로 맞춘 시즌 모수 사후 (정답 표 안 씀)", "data_ver": C.DATA_VER, "model": "M1",
           "form": form, "x_term": x_term, "r2": r2, "tag": tag, "names": ["a"] + names, "delay_key_rule": DELAY_RULE_NOTE,
           "tool_head": _git_head(), "seasons": {}}
    if a.resume and os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            out = json.load(f)
    os.makedirs(a.ll_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for s in a.seasons.split(","):
        if s in out["seasons"] and a.resume:
            continue
        t0 = time.time()
        head, rows = O.read_cell(os.path.join(a.obs, f"{s}.jsonl.gz"))
        r = fit_one(rows, head["H"], s, err, form, x_term, r2, K=16, delay_key=delay_key_stage1(s), stage1=True,
                    retry=not a.no_retry, max_tree_depth=a.max_tree_depth, multistart=not a.single_map,
                    warmup=a.warmup, samples=a.samples, retry_warmup=a.retry_warmup)
        sm, obs, jobs = r["samples"], r["obs"], r["jobs"]
        v = np.stack([sm[k].reshape(-1) for k in ["a"] + names], 1)
        # PSIS-LOO
        pos = obs["pos"]
        ll = sm["ll"][:, :, pos]
        idata = az.from_dict(posterior={k: sm[k] for k in ["a"] + names}, log_likelihood={"y": ll})
        lo = az.loo(idata, pointwise=True)
        loo_i = np.asarray(lo.loo_i).reshape(-1)
        pk = np.asarray(lo.pareto_k).reshape(-1)
        keys = np.array([m["key"] for m in obs["meta"]])
        np.savez(os.path.join(a.ll_dir, f"{tag}_{s}.npz"), loo_i=loo_i, keys=keys, pareto_k=pk)
        rp = OM.row_posteriors(jobs, {k: x for k, x in sm.items() if k != "ll"}, theta_surface_fn(names, form, x_term, r2))
        fl = OM.flags(obs["meta"], rp)
        lq = {f"{q:g}": np.quantile(sm[f"lnC_{q:g}"].reshape(-1), [0.025, 0.1, 0.5, 0.9, 0.975]).tolist() for q in OM.Q_TARGETS}
        secs = time.time() - t0
        info = r["info"]
        out["seasons"][s] = {
            "n_rows": obs["n"], "N": obs["N"], "counts": obs["counts"], "delay_key": delay_key_stage1(s),
            "mean": v.mean(0).tolist(), "cov": np.cov(v.T).tolist(),
            "lnC_q": lq, "diag": {k: r["diag"][k] for k in ("rhat_max", "ess_min", "conv", "divergences")},
            "core": core_diag(r["diag"], names, r["info"]["divergences"]),
            "worst": {"rhat": sorted(((v["rhat"], k) for k, v in r["diag"]["per"].items()), reverse=True)[:3],
                      "ess": sorted((v["ess"], k) for k, v in r["diag"]["per"].items())[:3]},
            "retried": info["retried"], "no_retry_mode": bool(a.no_retry), "srv": a.srv, "srv_scale": a.srv_scale, "multistart": not a.single_map, "warmup": a.warmup, "samples": a.samples, "retry_warmup": a.retry_warmup,
            "modes": r["pre"].get("modes"), "depth_dist": info.get("depth_dist"), "max_tree_depth": a.max_tree_depth, "info": {k: info[k] for k in info if k != "map"}, "map": info["map"],
            "secs_total": round(secs, 1),
            "loo": {"elpd_loo": float(lo.elpd_loo), "se": float(lo.se), "p_loo": float(lo.p_loo),
                    "pareto_k_gt_0.7": int((pk > 0.7).sum()), "n": int(len(pk))},
            "n_starts": info.get("n_starts"), "n_best_repeat": info.get("n_best_repeat"), "n_flags": len(fl), "flags": fl}
        with open(a.out + ".tmp", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=float)
        os.replace(a.out + ".tmp", a.out)
        d = out["seasons"][s]
        line = (f"stage1 {tag} {s} D1: {secs:.0f}s (map {info['map_secs']}s nuts {info['secs']}s) leapfrog {info['mean_leapfrog']:.0f} "
                f"div {info['divergences']} rhat {d['diag']['rhat_max']:.3f} ess {d['diag']['ess_min']:.0f} conv {d['diag']['conv']} "
                f"core rhat {d['core']['rhat_max']:.3f} ess {d['core']['ess_min']:.0f} conv_core {d['core']['conv_core']} offsets rhat {(d['core']['offsets_rhat_max'] or 0):.2f} starts {info.get('n_starts')}/{info.get('n_best_repeat')} worst {d['worst']['rhat'][0][1]}/{d['worst']['ess'][0][1]} retried {info['retried']} elpd {d['loo']['elpd_loo']:.1f}+-{d['loo']['se']:.1f} k>0.7 {d['loo']['pareto_k_gt_0.7']}")
        print(line, flush=True)
        _log_attempt(a.ll_dir, line)


def cmd_compare(a):
    ss = a.seasons.split(",")
    res = {"about": "꼴 고르기 (명세 7절 ④): PSIS-LOO pointwise 차 d = b - a, 같은 줄(key)끼리. 합 차 > 2 se 일 때만 b(복잡한 쪽).",
           "a": a.a, "b": a.b, "data_ver": C.DATA_VER, "seasons": {}}
    tot, var = 0.0, 0.0
    for s in ss:
        za = np.load(os.path.join(a.ll_dir, f"{a.a}_{s}.npz"))
        zb = np.load(os.path.join(a.ll_dir, f"{a.b}_{s}.npz"))
        da = dict(zip(za["keys"].tolist(), za["loo_i"]))
        db = dict(zip(zb["keys"].tolist(), zb["loo_i"]))
        if len(da) != len(za["keys"]) or len(db) != len(zb["keys"]):
            ka, kb = za["keys"], zb["keys"]
            assert len(ka) == len(kb) and (ka == kb).all(), "key 중복이면 같은 순서여야 한다"
            xa, xb = za["loo_i"], zb["loo_i"]
        else:
            common = [k for k in za["keys"].tolist() if k in db]
            xa = np.array([da[k] for k in common])
            xb = np.array([db[k] for k in common])
        d = xb - xa
        n = len(d)
        sd = float(d.sum())
        se = float(np.sqrt(n * d.var(ddof=1)))
        res["seasons"][s] = {"n": n, "elpd_a": float(xa.sum()), "elpd_b": float(xb.sum()), "diff_b_minus_a": sd, "se": se,
                             "favors": a.b if sd > 0 else a.a, "diff_over_se": sd / se if se > 0 else None}
        tot += sd
        var += se ** 2
    se_t = float(np.sqrt(var))
    res["sum_diff_b_minus_a"] = tot
    res["se"] = se_t
    res["diff_over_se"] = tot / se_t if se_t > 0 else None
    res["rule"] = "합 차 > 2 se 일 때만 b"
    res["choice"] = a.b if tot > 2 * se_t else a.a
    if a.stage1_a and a.stage1_b:   # 수렴 규칙: 복잡한 쪽(b)이 재시도 뒤에도 시즌 2개 넘게 core 수렴 못 하면 단순한 쪽
        conv = {}
        for tag, path in (("a", a.stage1_a), ("b", a.stage1_b)):
            with open(path, encoding="utf-8") as f:
                sj = json.load(f)["seasons"]
            conv[tag] = {s: bool(sj[s]["core"]["conv_core"]) for s in ss}
        res["conv_core"] = conv
        res["n_nonconv_a"] = sum(not v for v in conv["a"].values())
        res["n_nonconv_b"] = sum(not v for v in conv["b"].values())
        res["choice_by_elpd_only"] = res["choice"]
        if res["n_nonconv_b"] > 2:
            res["choice"] = a.a
            res["rule_note"] = "b가 6시즌 중 2시즌 넘게 수렴 못 해 단순한 쪽(a)"
    res["seasons_favoring_b"] = [s for s, v in res["seasons"].items() if v["diff_b_minus_a"] > 0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    line = f"compare {a.a} vs {a.b}: sum {tot:.2f} se {se_t:.2f} ({res['diff_over_se']:.2f} se) choice {res['choice']}; per season " + \
        ", ".join(f"{s} {v['diff_b_minus_a']:+.1f}" for s, v in res["seasons"].items())
    print(line)
    _log_attempt(a.ll_dir, line)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("stage1")
    p.add_argument("--obs", required=True)
    p.add_argument("--seasons", required=True)
    p.add_argument("--form", required=True, choices=list(ETA_N))
    p.add_argument("--x-term", action="store_true")
    p.add_argument("--r2", action="store_true")
    p.add_argument("--tag")
    p.add_argument("--out", required=True)
    p.add_argument("--ll-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--srv", choices=["full", "delta"], default="full", help="delta: 일·글 성분은 점수 비 δ만 (ψ=κ=0)")
    p.add_argument("--srv-scale", type=float, default=1.0, help="δ·ψ·κ 사전 sd에 곱함")
    p.add_argument("--single-map", action="store_true", help="시작점 8개 MAP 대신 단일 MAP (옛 방식, 판정에 안 씀)")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--retry-warmup", type=int, default=1500)
    p.add_argument("--max-tree-depth", type=int, default=10, help="NUTS 나무 깊이 상한 (시간 상한용; 7이면 한 번에 127걸음)")
    p.add_argument("--no-retry", action="store_true", help="수렴 못 해도 0.95 재시도를 하지 않는다 (시간 절약; conv 표시는 그대로)")
    p = sp.add_parser("compare")
    p.add_argument("--ll-dir", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--seasons", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage1-a")
    p.add_argument("--stage1-b")
    a = ap.parse_args()
    {"stage1": cmd_stage1, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
