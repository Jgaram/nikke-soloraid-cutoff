"""시즌 사이 층 (WP3b 1단계, 2026-10-29) — 두 단계 경험적 베이즈의 2단계. 후보(M1·M2·M3)와 무관하다 (모수 이름 목록만 다르다).

  import hier as H            # sys.path.insert(0, HERE) 뒤. 실행 진입점에서 obs_model.setup_host(4)를 먼저 부른다

설계서 2절, 명세 「WP3 명세」 3절. 입력 stage1 = {시즌: {"names": [...], "mean": [D], "cov": [D,D]}} (1단계 사후 평균·공분산).
규모 a는 이름 "a"로 들어 있고 따로 다룬다 — 나머지 이름 `names`(a 뺀 것)만 MVN 층에 들어간다.

쓰는 순서
  prior = H.prior_for(stage1, "S30", names, tau_scale, mu_loc, mu_scale, cache_dir)   # 앞 시즌만으로 JSON 사전
  # prior = {"target","k","window","names","mean","cov","a":{"mean","var","tau_a_mean"},"scores":{k: 점수}}
  draws = H.fit_window(theta_hat, V, mu_loc, mu_scale, tau_scale, seed)               # {"mu":[n,D], "L":[n,D,D]} (L = Sigma의 콜레스키)
  mean, cov = H.predictive(draws);  lp = H.logpred(draws, theta, V)
  ld = H.fit_level(a_hat, Va, seasons=[...]);  m, v = H.level_prior(ld, a_prev, Va_prev)

모형: mu ~ N(mu_loc, mu_scale^2), tau ~ HalfNormal(tau_scale), L_Omega ~ LKJCholesky(D, 2), Sigma = diag(tau) Omega diag(tau),
theta_hat_j ~ MVN(mu, Sigma + V_j). J를 24로 채우고 mask로 끈다 (컴파일 다시 쓰기: 같은 D면 MCMC 물체를 모듈 안에 둔다).
창 고르기(choose_window)는 평가 시즌마다 k시즌 창을 맞추고 (평가 시즌, 창 시즌 목록)으로 적합을 캐시한다 (메모리 + cache_dir의 npz).
"""
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

JPAD = 24
JITTER = 1e-5
STATS = {"fit": 0, "hit": 0, "secs": 0.0}
_MEM = {}
_MCMC = {}


def snum(s):
    return int(re.sub(r"\D", "", s))


def _window_model(theta, V, mask, mu_loc, mu_scale, tau_scale):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    D = mu_loc.shape[0]
    mu = numpyro.sample("mu", dist.Normal(mu_loc, mu_scale).to_event(1))
    tau = numpyro.sample("tau", dist.HalfNormal(tau_scale).to_event(1))
    lom = numpyro.sample("L_omega", dist.LKJCholesky(D, 2.0))
    L = tau[:, None] * lom
    cov = (L @ L.T)[None] + V + JITTER * jnp.eye(D)[None]
    ll = dist.MultivariateNormal(mu, covariance_matrix=cov).log_prob(theta)
    numpyro.factor("ll", (ll * mask).sum())


def fit_window(theta_hat, V, mu_loc, mu_scale, tau_scale, seed=0, chains=4, warmup=500, samples=500):
    """창 안 시즌 j의 (theta_hat[J,D], V[J,D,D])로 mu·Sigma 사후를 NUTS로. -> {"mu":[n,D], "L":[n,D,D], "tau":[n,D], "info": {...}}."""
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS, init_to_median
    theta_hat = np.asarray(theta_hat, float)
    V = np.asarray(V, float)
    J, D = theta_hat.shape
    if J > JPAD:
        raise ValueError(f"J={J} > {JPAD}")
    th = np.zeros((JPAD, D))
    VV = np.tile(np.eye(D), (JPAD, 1, 1))
    mk = np.zeros(JPAD)
    th[:J], VV[:J], mk[:J] = theta_hat, V, 1.0
    key = (D, chains, warmup, samples)
    if key not in _MCMC:
        kern = NUTS(_window_model, target_accept_prob=0.9, dense_mass=True, init_strategy=init_to_median(num_samples=15))
        _MCMC[key] = MCMC(kern, num_warmup=warmup, num_samples=samples, num_chains=chains, chain_method="parallel",
                          progress_bar=False, jit_model_args=True)
    m = _MCMC[key]
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    t0 = time.time()
    m.run(jax.random.PRNGKey(seed), f32(th), f32(VV), f32(mk), f32(mu_loc), f32(mu_scale), f32(tau_scale),
          extra_fields=("diverging",))
    sm = m.get_samples()
    ex = m.get_extra_fields()
    secs = time.time() - t0
    STATS["fit"] += 1
    STATS["secs"] += secs
    tau = np.asarray(sm["tau"])
    L = tau[:, :, None] * np.asarray(sm["L_omega"])
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    g = m.get_samples(group_by_chain=True)
    rh = max(float(np.nanmax(np.asarray(split_gelman_rubin(np.asarray(g[k]))))) for k in ("mu", "tau"))
    es = min(float(np.nanmin(np.asarray(effective_sample_size(np.asarray(g[k]))))) for k in ("mu", "tau"))
    return {"mu": np.asarray(sm["mu"]), "L": L, "tau": tau,
            "info": {"secs": round(secs, 1), "divergences": int(np.asarray(ex["diverging"]).sum()),
                     "rhat_max": rh, "ess_min": es, "J": J, "D": D}}


def predictive(draws, n_per=5, seed=0):
    """새 시즌 theta_new = mu + L z (뽑기마다 n_per개) -> (mean[D], cov[D,D]). 초모수 불확실성이 공분산에 들어간다."""
    rng = np.random.default_rng(seed)
    mu, L = draws["mu"], draws["L"]
    n, D = mu.shape
    z = rng.standard_normal((n, n_per, D))
    th = mu[:, None, :] + np.einsum("nij,npj->npi", L, z)
    th = th.reshape(-1, D)
    return th.mean(0), np.cov(th.T)


def logpred(draws, theta, V):
    """log mean_draws MVN(theta | mu, Sigma + V)."""
    theta = np.asarray(theta, float)
    V = np.asarray(V, float)
    mu, L = draws["mu"], draws["L"]
    D = mu.shape[1]
    S = np.einsum("nij,nkj->nik", L, L) + V[None] + JITTER * np.eye(D)[None]
    ch = np.linalg.cholesky(S)
    r = theta[None] - mu
    zz = np.linalg.solve(ch, r[:, :, None])[:, :, 0]
    lp = -0.5 * (zz ** 2).sum(1) - np.log(np.diagonal(ch, axis1=1, axis2=2)).sum(1) - 0.5 * D * np.log(2 * np.pi)
    m = lp.max()
    return float(m + np.log(np.mean(np.exp(lp - m))))


# ---------- 규모 a ----------

def fit_level(a_hat, Va, tau_scale=0.3, seasons=None):
    """창 안 연속 시즌 짝 d_j = a_j - a_{j-1} ~ N(0, tau_a^2 + Va_j + Va_{j-1}), tau_a ~ HalfNormal(tau_scale).
    1차원이라 격자(2000점)로 정확히 푼다. seasons = 시즌 번호 목록 (None이면 모두 연속). 짝이 2개 미만이면 tau_a = tau_scale 그대로.
    -> {"tau_a": 뽑기[4000], "pairs": 짝 수}."""
    a_hat, Va = np.asarray(a_hat, float), np.asarray(Va, float)
    nums = list(range(len(a_hat))) if seasons is None else [snum(s) if isinstance(s, str) else int(s) for s in seasons]
    d, v = [], []
    for j in range(1, len(a_hat)):
        if nums[j] - nums[j - 1] == 1:
            d.append(a_hat[j] - a_hat[j - 1])
            v.append(Va[j] + Va[j - 1])
    rng = np.random.default_rng(0)
    if len(d) < 2:
        return {"tau_a": np.full(4000, float(tau_scale)), "pairs": len(d)}
    d, v = np.array(d), np.array(v)
    grid = np.linspace(1e-4, 4 * tau_scale * 2, 2000)
    var = grid[:, None] ** 2 + v[None]
    lp = -0.5 * (d[None] ** 2 / var + np.log(var)).sum(1) - 0.5 * (grid / tau_scale) ** 2
    p = np.exp(lp - lp.max())
    p /= p.sum()
    draws = rng.choice(grid, size=4000, p=p) + rng.uniform(-0.5, 0.5, 4000) * (grid[1] - grid[0])
    return {"tau_a": np.abs(draws), "pairs": len(d)}


def level_prior(draws, a_prev, Va_prev):
    """새 시즌 a의 사전 = (평균 a_prev, 분산 E[tau_a^2] + Va_prev)."""
    return float(a_prev), float(np.mean(np.asarray(draws["tau_a"]) ** 2) + Va_prev)


# ---------- 창 고르기 ----------

def _arrays(stage1, seasons, names):
    th, V = [], []
    for s in seasons:
        e = stage1[s]
        ix = [e["names"].index(n) for n in names]
        th.append(np.asarray(e["mean"], float)[ix])
        V.append(np.asarray(e["cov"], float)[np.ix_(ix, ix)])
    return np.array(th), np.array(V)


def _fit_key(th, V, mu_loc, mu_scale, tau_scale, seed):
    h = hashlib.sha1()
    for a in (th, V, mu_loc, mu_scale, tau_scale):
        h.update(np.ascontiguousarray(np.asarray(a, float)).tobytes())
    h.update(str(seed).encode())
    return h.hexdigest()[:16]


def cached_fit(stage1, seasons, names, mu_loc, mu_scale, tau_scale, cache_dir=None, seed=0):
    """창 시즌 목록으로 fit_window — (시즌 목록, 입력 자료) 열쇠로 메모리·디스크(cache_dir/win_{열쇠}.npz)에 캐시."""
    th, V = _arrays(stage1, seasons, names)
    key = _fit_key(th, V, mu_loc, mu_scale, tau_scale, seed)
    if key in _MEM:
        STATS["hit"] += 1
        return _MEM[key]
    path = os.path.join(cache_dir, f"win_{key}.npz") if cache_dir else None
    if path and os.path.exists(path):
        z = np.load(path)
        d = {"mu": z["mu"], "L": z["L"], "tau": z["tau"], "info": json.loads(str(z["info"]))}
        STATS["hit"] += 1
    else:
        d = fit_window(th, V, mu_loc, mu_scale, tau_scale, seed)
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(path, mu=d["mu"], L=d["L"], tau=d["tau"], info=json.dumps(d["info"]))
    _MEM[key] = d
    return d


def choose_window(stage1, target, names, mu_loc, mu_scale, tau_scale, ks=(4, 6, 10, "all"), cache_dir=None, seed=0):
    """시간 순 교차검증으로 창 k 고르기. -> (k, {k: 점수}). 평가 시즌 s' = target 앞이고 자기 앞 시즌이 3개 이상인 것.
    평가 시즌이 3개 미만이면 "all"."""
    ss = sorted(stage1, key=snum)
    before = [s for s in ss if snum(s) < snum(target)]
    evals = [s for i, s in enumerate(before) if i >= 3]
    if len(evals) < 3:
        return "all", {}
    scores = {}
    for k in ks:
        tot = 0.0
        for s2 in evals:
            prev = [s for s in before if snum(s) < snum(s2)]
            win = prev if k == "all" else prev[-int(k):]
            d = cached_fit(stage1, win, names, mu_loc, mu_scale, tau_scale, cache_dir, seed)
            th, V = _arrays(stage1, [s2], names)
            tot += logpred(d, th[0], V[0])
        scores[str(k)] = tot
    best = max(scores, key=scores.get)
    return (best if best == "all" else int(best)), scores


def prior_for(stage1, target, names, tau_scale, mu_loc, mu_scale, cache=None, ks=(4, 6, 10, "all"), seed=0, k_fixed=None):
    """target 시즌 칸의 사전 (JSON으로 쓸 수 있는 dict). 창은 target 바로 앞 k시즌. cache = 디스크 캐시 폴더 또는 None.
    k_fixed를 주면 창 고르기를 건너뛴다 (민감도 보고용)."""
    ss = sorted(stage1, key=snum)
    before = [s for s in ss if snum(s) < snum(target)]
    if not before:
        raise ValueError("앞 시즌이 없다")
    if k_fixed is None:
        k, scores = choose_window(stage1, target, names, mu_loc, mu_scale, tau_scale, ks, cache, seed)
    else:
        k, scores = k_fixed, {}
    win = before if k == "all" else before[-int(k):]
    d = cached_fit(stage1, win, names, mu_loc, mu_scale, tau_scale, cache, seed)
    mean, cov = predictive(d)
    ah = np.array([stage1[s]["mean"][stage1[s]["names"].index("a")] for s in win])
    ix = [stage1[s]["names"].index("a") for s in win]
    Va = np.array([np.asarray(stage1[s]["cov"], float)[i, i] for s, i in zip(win, ix)])
    ld = fit_level(ah, Va, 0.3, win)
    m, v = level_prior(ld, ah[-1], Va[-1])
    return {"target": target, "k": k, "window": win, "names": list(names), "mean": mean.tolist(), "cov": cov.tolist(),
            "a": {"mean": m, "var": v, "tau_a_mean": float(np.mean(ld["tau_a"]))},
            "scores": scores, "fit_info": d["info"]}
