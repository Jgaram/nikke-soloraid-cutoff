"""WP3b 1단계 시험 (2026-10-29) — obs_model.py · hier.py · m1.py 곡면 함수. 정답 표(answer_key)는 읽지 않는다.

  python3 -I tools/solo_raid/lab7/check_obs.py --work W [--tests 1,2,3,4] [--season S30] [--chains 4 --warmup 500 --samples 500]

W = 작업 폴더 (obs7/full/S{n}.jsonl.gz가 있는 곳). 결과는 W/check/ 아래 JSON (레포 안에는 쓰지 않는다).
  1  logdiff_ndtr: float32 대 float64(mpmath 없으면 scipy 같은 식)를 a,b 격자(-40..40)에서
  2  노드: 묶음마다 무게 합 1·지지 안·넓은 지지에서 평균 ≈ 분포 평균, 그리고 실제 줄 전체 + 시즌 prepare counts
  3  합성 되찾기 1번: 실제 줄 배치에 알려진 theta로 %를 뽑아 NUTS (kuma 꼴) -> 참값이 95% 안인가
  4  hier: 알려진 mu·Sigma로 만든 theta_j 20개 -> fit_window, choose_window·prior_for·캐시
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_model as OM  # noqa: E402
import obs_table as O  # noqa: E402


def dump(work, name, obj):
    d = os.path.join(work, "check")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)


# ---------- 1 ----------

def test1(work):
    import jax.numpy as jnp
    from scipy.special import log_ndtr
    try:
        import mpmath as mp
        mp.mp.dps = 80
        ref = lambda a, b: float(mp.log(mp.ncdf(-a) - mp.ncdf(-b))) if a + b > 0 else float(mp.log(mp.ncdf(b) - mp.ncdf(a)))
        refname = "mpmath"
    except ImportError:
        def ref(a, b):
            if a + b > 0:
                a, b = -b, -a
            return float(log_ndtr(b) + np.log(-np.expm1(log_ndtr(a) - log_ndtr(b))))
        refname = "scipy float64 같은 식"
    g = np.arange(-40, 40.01, 2.0)
    pairs = [(a, b) for a in g for b in g if b > a]
    pairs += [(a, a + gap) for a in np.arange(-30, 30.1, 3.0) for gap in (1.0, 0.1, 1e-2, 1e-3)]
    A = np.array([p[0] for p in pairs], np.float32)
    B = np.array([p[1] for p in pairs], np.float32)
    f = np.asarray(OM.logdiff_ndtr(jnp.asarray(A), jnp.asarray(B)))
    # 기준은 float32로 반올림된 입력에서 (같은 입력)
    r = np.array([ref(float(a), float(b)) for a, b in zip(A, B)])
    err = np.abs(f - r)
    rel = err / np.maximum(np.abs(r), 1.0)
    gap = (B - A).astype(float)
    out = {"ref": refname, "n": len(pairs), "finite": bool(np.isfinite(f).all()), "max_rel": float(rel.max()),
           "max_abs": float(err.max())}
    for lo, hi in [(0, 0.005), (0.005, 0.05), (0.05, 0.5), (0.5, 1e9)]:
        m = (gap >= lo) & (gap < hi)
        if m.any():
            out[f"gap[{lo},{hi})"] = {"n": int(m.sum()), "max_rel": float(rel[m].max()), "max_abs": float(err[m].max())}
    # 기울기도 유한한가
    import jax
    gr = jax.vmap(jax.grad(OM.logdiff_ndtr, argnums=(0, 1)))(jnp.asarray(A), jnp.asarray(B))
    out["grad_finite"] = bool(np.isfinite(np.asarray(gr[0])).all() and np.isfinite(np.asarray(gr[1])).all())
    out["pass"] = bool(out["finite"] and out["grad_finite"] and out["max_rel"] < 1e-3)
    dump(work, "test1.json", out)
    return out


# ---------- 2 ----------

def _truncated_mean(pdf, xa, xb, n=400000):
    x = np.linspace(xa, xb, n)
    p = pdf(x)
    return float((x * p).sum() / p.sum())


def test2(work, season, err):
    from scipy import stats
    head, rows = O.read_cell(os.path.join(work, "obs7", "full", f"{season}.jsonl.gz"))
    H = head["H"]
    out = {"groups": {}}
    b1 = err["1b_stated_time"]
    base = {"tg": None, "h": 60.0, "hp": 60.5, "h_lo": 59.5, "h_hi": 60.5, "h_lo_top7": None, "h_hi_top7": None}
    # 넓은 지지: h = 100, 지지 [0, 100+]
    wide = dict(base, h=100.0, hp=100.0)
    for tg in ("A", "post", "timer", "now", "clock", "delay"):
        r = dict(wide, tg=tg)
        res = {}
        for K in (16, 256):
            t, w, why = OM.nodes_detail(r, 120.0, err, None, K)
            Lo, Up = OM.support(r, 120.0) if tg not in ("A", "post") else (t.min(), t.max())
            m = float((t * w).sum())
            res[K] = {"wsum": float(w.sum()), "in_support": bool(((t >= Lo - 1e-9) & (t <= Up + 1e-9))[w > 0].all()),
                      "mean": m, "reason": why}
        # 분포 평균 (지지로 자른)
        Lo, Up = OM.support(r, 120.0) if tg not in ("A", "post") else (None, None)
        if tg == "now":
            p = b1["now"]["fit"]["params"]
            xa, xb = max(0, (100 - Up) * 60), (100 - Lo) * 60
            pdf = lambda x: (1 - p["pi_tail"]) * stats.lognorm.pdf(x, p["sigma1"], scale=np.exp(p["mu1"])) + \
                p["pi_tail"] * stats.lognorm.pdf(x, p["sigma2"], scale=np.exp(p["mu2"]))
            true = 100 - _truncated_mean(pdf, max(xa, 1e-6), xb) / 60
        elif tg == "clock":
            p = b1["clock"]["fit"]["params"]
            xa, xb = (100 - Up) * 60, (100 - Lo) * 60
            pdf = lambda x: (1 - p["pi_wide"]) * stats.norm.pdf(x, p["mu"], p["sigma_core"]) + \
                p["pi_wide"] * stats.norm.pdf(x, p["mu"], p["sigma_wide"])
            true = 100 - _truncated_mean(pdf, xa, xb) / 60
        elif tg == "delay":
            p = err["1a_post_delay"]["recommend"]["params"]
            xa, xb = max(0, 100 - Up), 100 - Lo
            pdf = lambda x: p["w_fast"] * stats.lognorm.pdf(x, p["sigma1"], scale=np.exp(p["mu1"])) + \
                (1 - p["w_fast"]) * stats.lognorm.pdf(x, p["sigma2"], scale=np.exp(p["mu2"]))
            true = 100 - _truncated_mean(pdf, max(xa, 1e-6), xb, 2000000)
        elif tg == "timer":
            pw = b1["timer"]["rare"]["pi"]["p"]
            g = b1["timer"]["rare"]["p_k_ge_2_given_wrong"]["p"]
            ks = np.arange(0, 6)
            wk = np.where(ks == 0, 1 - pw, pw * (1 - g) * g ** (np.maximum(ks, 1) - 1))
            t = 100 - 24.0 * ks
            ok = (t >= Lo) & (t <= Up)
            true = float((t[ok] * wk[ok]).sum() / wk[ok].sum())
        else:
            true = 100.0 if tg == "A" else 120.0
        res["dist_mean"] = true
        res["rel_diff_K16"] = (res[16]["mean"] - true) / max(abs(true), 1e-9) if tg != "delay" else \
            (res[16]["mean"] - true) / 100
        res["rel_diff_K256"] = (res[256]["mean"] - true) / max(abs(true), 1e-9)
        res["abs_diff_h_K16"] = res[16]["mean"] - true
        res["abs_diff_h_K256"] = res[256]["mean"] - true
        out["groups"][tg] = res
    # 좁은 지지 (조건: 지지가 좁아도 노드 안)
    # 실제 줄 전체: 노드가 지지 안, 무게 합 1
    sel = OM.select_rows(rows, H, stage1=True)
    bad = 0
    for r in sel:
        t, w, why = OM.nodes_detail(r, H, err, None, 16)
        if t is None:
            continue
        if r["tg"] in ("A", "post"):
            continue
        Lo, Up = OM.support(r, H)
        if abs(w.sum() - 1) > 1e-9 or not ((t >= Lo - 1e-9) & (t <= Up + 1e-9))[w > 0].all():
            bad += 1
    out["real_rows_checked"] = len(sel)
    out["real_rows_bad"] = bad
    obs = OM.prepare(rows, H, season, err, K=16, stage1=True)
    out["prepare_counts"] = obs["counts"]
    out["prepare_N"] = obs["N"]
    out["prepare_n"] = obs["n"]
    out["prepare_finite"] = bool(all(np.isfinite(obs[k]).all() for k in OM.NUMERIC))
    out["prepare_sum_w_ok"] = bool(np.allclose(np.exp(obs["lw"]).sum(1)[obs["pos"]], 1.0, atol=1e-5))
    out["prepare_u_range"] = [float(obs["u"][obs["pos"]].min()), float(obs["u"][obs["pos"]].max())]
    # 노드 하나 줄 칸 나눔(n1)이 전체 계산과 같은가
    import jax.numpy as jnp
    import m1 as M1
    jb = OM.to_jax(obs)
    th = dict(TRUTH)
    th["eta"] = [jnp.float32(0.0), jnp.float32(0.0)]
    th = {k: (v if k == "eta" else jnp.float32(v)) for k, v in th.items()}
    mu, P = M1.surface(th, jb["y"], jb["u"], "kuma")
    phi0 = jb["dirichlet"] / jb["dirichlet"].sum(1, keepdims=True)
    ll_blk = np.asarray(OM.row_loglik(jb, mu, P, 0.01, 0.13, 0.015, phi0))
    ll_all = np.asarray(OM.row_loglik(jb, mu, P, 0.01, 0.13, 0.015, phi0, parts=True)["ll"])
    out["block_split_max_abs_diff"] = float(np.abs(ll_blk - ll_all).max())
    out["block_split_finite"] = bool(np.isfinite(ll_blk).all())
    out["block_N1_N2"] = [obs["n1"], obs["N"] - obs["n1"]]
    ok = out["block_split_max_abs_diff"] < 1e-3 and out["block_split_finite"]
    for tg, res in out["groups"].items():
        ok &= all(abs(res[K]["wsum"] - 1) < 1e-9 and res[K]["in_support"] for K in (16, 256))
        ok &= abs(res["rel_diff_K256"]) < 0.02
    out["pass"] = bool(ok and bad == 0 and out["prepare_finite"] and out["prepare_sum_w_ok"])
    dump(work, "test2.json", out)
    return out


# ---------- 3 ----------

TRUTH = {"a": float(np.log(125.0)), "r0": float(np.log(0.2)), "r1": 0.5, "lnm": float(np.log(0.2)), "eta0": 0.0, "eta1": 0.0,
         "lnsig0": float(np.log(0.01)), "lnrho": float(np.log(0.13)), "dJP": float(np.log(0.8)), "dGL": float(np.log(0.85)),
         "pJP": 0.1, "pGL": -0.1, "kJP": 0.05, "kGL": -0.05}


def synth_rows(rows, H, err, seed=1):
    """실제 줄 배치(점수·종류·시각 정보·서버 라벨·사이트)에 TRUTH로 %를 뽑아 바꾼 줄 목록과 진짜 값들."""
    import jax.numpy as jnp
    import m1 as M1
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
    tshot = np.array(tshot)
    u = np.clip(tshot / H, 0.02, 1.0)
    th = dict(TRUTH)
    th["eta"] = [th["eta0"], th["eta1"]]
    mu, P = M1.surface({k: (jnp.float32(v) if not isinstance(v, list) else [jnp.float32(x) for x in v]) for k, v in th.items()},
                       jnp.asarray(y, jnp.float32), jnp.asarray(u[:, None], jnp.float32), "kuma")
    mu = np.asarray(mu)[:, :, 0]
    mu_row = mu[np.array(srv), np.arange(n)]
    sd = np.sqrt(np.exp(th["lnsig0"]) ** 2 + (np.exp(th["lnrho"]) * np.asarray(P)[:, :, 0][np.array(srv), np.arange(n)]) ** 2)
    z = mu_row + sd * rng.standard_normal(n)
    cont = rng.random(n) < 0.015
    z = np.where(cont, z + rng.laplace(0, 0.34, n), z)
    x100 = 10 ** z * 100
    pct = np.where(rng.random(n) < 0.5, np.round(x100), np.floor(x100)) / 100   # 반올림/버림 반반 (1f의 구간 [p-0.005, p+0.01))
    out = []
    dropped = 0
    for r, p in zip(keep, pct):
        if p > 100.0 or p < 0.01:
            dropped += 1
            continue
        x = dict(r)
        x["pct"] = float(p)
        out.append(x)
    truth = {"n_rows": len(out), "dropped_out_of_range": dropped, "n_contam": int(cont.sum()),
             "server_counts": [int((np.array(srv) == k).sum()) for k in range(3)]}
    return out, truth


def test3(work, season, err, chains, warmup, samples):
    import jax.numpy as jnp
    import m1 as M1
    OM.setup_host(chains)   # jax 배열을 만들기 전에
    head, rows = O.read_cell(os.path.join(work, "obs7", "full", f"{season}.jsonl.gz"))
    H = head["H"]
    srows, truth = synth_rows(rows, H, err)
    obs = OM.prepare(srows, H, season, err, K=16, stage1=True)
    jobs = OM.to_jax(obs)
    t0 = time.time()
    stage = {}
    pre = M1.map_precond(jobs, {"kind": "weak"}, "kuma")   # MAP + 라플라스로 사후를 미리 표준정규 가깝게 편다
    t_map = time.time() - t0
    stage["map"] = round(t_map, 1)
    sm, info = OM.run_nuts(M1.model, {"obs": jobs, "prior": {"kind": "weak"}, "form": "kuma", "pre": pre}, seed=0,
                           chains=chains, warmup=warmup, samples=samples)
    info["map_secs"] = round(t_map, 1)
    info["map"] = pre["info"]
    stage["nuts_total"] = round(time.time() - t0 - t_map, 1)
    names = ["a"] + M1.theta_names("kuma")
    lnc_names = [f"lnC_{q:g}" for q in (1, 3, 10)]
    dg = OM.diagnose(sm, names + lnc_names, divergences=info["divergences"])
    stage["diagnose_done"] = round(time.time() - t0, 1)
    th_true = dict(TRUTH)
    th_true["eta"] = [0.0, 0.0]
    tr = {k: float(M1.lnC({k2: jnp.float32(v) for k2, v in th_true.items() if k2 != "eta"}, q)) for k, q in
          [("lnC_1", 1.0), ("lnC_3", 3.0), ("lnC_10", 10.0)]}
    rows_out = {}
    cover = 0
    for nme in names:
        v = sm[nme].ravel()
        lo, hi = np.quantile(v, [0.025, 0.975])
        inside = bool(lo <= TRUTH[nme] <= hi)
        cover += inside
        rows_out[nme] = {"truth": TRUTH[nme], "mean": float(v.mean()), "sd": float(v.std()), "lo95": float(lo), "hi95": float(hi),
                         "inside": inside, "rhat": dg["per"][nme]["rhat"], "ess": dg["per"][nme]["ess"]}
    for nme in lnc_names:
        v = sm[nme].ravel()
        lo, hi = np.quantile(v, [0.025, 0.975])
        rows_out[nme] = {"truth": tr[nme], "mean": float(v.mean()), "sd": float(v.std()), "lo95": float(lo), "hi95": float(hi),
                         "inside": bool(lo <= tr[nme] <= hi), "rel_err_C": float(np.exp(v.mean() - tr[nme]) - 1),
                         "rhat": dg["per"][nme]["rhat"], "ess": dg["per"][nme]["ess"]}
    # 오염 줄 되찾기: 합성에서 진짜 오염은 모르는 척 두고 사후 확률만 본다
    out = {"truth_batch": truth, "N": obs["N"], "n": obs["n"], "counts": obs["counts"], "params": rows_out,
           "param_inside": f"{cover}/{len(names)}", "info": info, "diag": {k: dg[k] for k in ("rhat_max", "ess_min", "conv", "divergences")},
           "wall_secs_incl_setup": round(time.time() - t0, 1), "stage_secs": stage,
           "note": "다른 백그라운드 작업(T 절단 사본 만들기, 3프로세스)이 4코어를 나눠 쓰는 동안 잰 시간"}
    dump(work, "test3.json", out)
    return out


# ---------- 4 ----------

def test4(work, chains=4, warmup=500, samples=500):
    import hier as Hr
    OM.setup_host(chains)
    rng = np.random.default_rng(3)
    names = ["r0", "r1", "lnm", "eta0", "eta1", "lnsig0"]
    D = len(names)
    mu_true = np.array([-1.5, 0.8, -1.6, 0.1, -0.1, -4.6])
    tau_true = np.array([0.2, 0.3, 0.15, 0.3, 0.3, 0.25])
    om = np.eye(D)
    om[0, 1] = om[1, 0] = -0.5
    om[2, 3] = om[3, 2] = 0.4
    Sig = np.diag(tau_true) @ om @ np.diag(tau_true)
    # 20개 theta_j + 잡음 V_j
    J = 20
    th = rng.multivariate_normal(mu_true, Sig, J)
    Vs = []
    for j in range(J):
        sd = rng.uniform(0.03, 0.12, D)
        c = np.diag(sd) @ np.array([[1 if i == k else 0.3 for k in range(D)] for i in range(D)]) @ np.diag(sd)
        Vs.append(c)
    Vs = np.array(Vs)
    obs = np.array([rng.multivariate_normal(th[j], Vs[j]) for j in range(J)])
    mu_loc = np.zeros(D)
    mu_scale = np.full(D, 3.0)
    tau_scale = np.full(D, 0.5)
    out = {}
    t0 = time.time()
    d1 = Hr.fit_window(obs, Vs, mu_loc, mu_scale, tau_scale, seed=0, chains=chains, warmup=warmup, samples=samples)
    out["fit_first_secs"] = round(time.time() - t0, 1)
    t0 = time.time()
    d2 = Hr.fit_window(obs[:10], Vs[:10], mu_loc, mu_scale, tau_scale, seed=1, chains=chains, warmup=warmup, samples=samples)
    out["fit_second_secs_J10_reuse_compile"] = round(time.time() - t0, 1)
    mu_s = d1["mu"]
    tau_s = d1["tau"]
    out["mu"] = [{"name": n, "truth": float(mu_true[i]), "mean": float(mu_s[:, i].mean()),
                  "lo95": float(np.quantile(mu_s[:, i], 0.025)), "hi95": float(np.quantile(mu_s[:, i], 0.975)),
                  "inside": bool(np.quantile(mu_s[:, i], 0.025) <= mu_true[i] <= np.quantile(mu_s[:, i], 0.975))}
                 for i, n in enumerate(names)]
    out["tau"] = [{"name": n, "truth": float(tau_true[i]), "mean": float(tau_s[:, i].mean()),
                   "lo95": float(np.quantile(tau_s[:, i], 0.025)), "hi95": float(np.quantile(tau_s[:, i], 0.975)),
                   "inside": bool(np.quantile(tau_s[:, i], 0.025) <= tau_true[i] <= np.quantile(tau_s[:, i], 0.975))}
                  for i, n in enumerate(names)]
    out["fit_info"] = d1["info"]
    m, c = Hr.predictive(d1)
    out["predictive_sd_vs_true_sqrt_var"] = [[float(np.sqrt(c[i, i])) for i in range(D)], [float(np.sqrt(Sig[i, i])) for i in range(D)]]
    out["logpred_new"] = Hr.logpred(d1, rng.multivariate_normal(mu_true, Sig), Vs[0])
    # 시즌 24개 가짜 stage1 -> choose_window · prior_for · 캐시
    seasons = [f"S{n}" for n in range(19, 43)]
    a0 = 4.0
    stage1 = {}
    for i, s in enumerate(seasons):
        a0 += rng.normal(0, 0.05)
        tj = rng.multivariate_normal(mu_true + 0.01 * i * np.array([1, 0, 0, 0, 0, 0]), Sig)
        V = Vs[i % J]
        a_cov = 0.002
        full_names = ["a"] + names
        mean = np.concatenate([[a0 + rng.normal(0, np.sqrt(a_cov))], rng.multivariate_normal(tj, V)])
        cov = np.zeros((D + 1, D + 1))
        cov[0, 0] = a_cov
        cov[1:, 1:] = V
        stage1[s] = {"names": full_names, "mean": mean.tolist(), "cov": cov.tolist()}
    cache = os.path.join(work, "check", "hier_cache")
    st0 = dict(Hr.STATS)
    t0 = time.time()
    p1 = Hr.prior_for(stage1, "S26", names, tau_scale, mu_loc, mu_scale, cache)
    out["prior_for_S26"] = {"secs": round(time.time() - t0, 1), "k": p1["k"], "window": p1["window"], "scores": p1["scores"],
                            "a": p1["a"], "fits_run": Hr.STATS["fit"] - st0["fit"], "hits": Hr.STATS["hit"] - st0["hit"]}
    st1 = dict(Hr.STATS)
    t0 = time.time()
    p2 = Hr.prior_for(stage1, "S27", names, tau_scale, mu_loc, mu_scale, cache)
    out["prior_for_S27"] = {"secs": round(time.time() - t0, 1), "k": p2["k"], "window": p2["window"], "scores": p2["scores"],
                            "fits_run": Hr.STATS["fit"] - st1["fit"], "hits": Hr.STATS["hit"] - st1["hit"]}
    # 디스크 캐시: 메모리를 비우고 S26을 다시
    Hr._MEM.clear()
    st2 = dict(Hr.STATS)
    t0 = time.time()
    p3 = Hr.prior_for(stage1, "S26", names, tau_scale, mu_loc, mu_scale, cache)
    out["prior_for_S26_again_from_disk"] = {"secs": round(time.time() - t0, 1), "fits_run": Hr.STATS["fit"] - st2["fit"],
                                            "hits": Hr.STATS["hit"] - st2["hit"],
                                            "same_mean": bool(np.allclose(p1["mean"], p3["mean"]))}
    json.dumps(p1)   # JSON으로 쓸 수 있다
    out["json_ok"] = True
    out["pass"] = bool(sum(x["inside"] for x in out["mu"]) >= D - 1 and sum(x["inside"] for x in out["tau"]) >= D - 1
                       and out["prior_for_S26_again_from_disk"]["fits_run"] == 0)
    dump(work, "test4.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--tests", default="1,2,3,4")
    ap.add_argument("--season", default="S30")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    a = ap.parse_args()
    OM.setup_host(a.chains)   # 모든 jax 계산보다 먼저
    err = OM.load_err()
    for t in a.tests.split(","):
        t0 = time.time()
        if t == "1":
            r = test1(a.work)
        elif t == "2":
            r = test2(a.work, a.season, err)
        elif t == "3":
            r = test3(a.work, a.season, err, a.chains, a.warmup, a.samples)
        elif t == "4":
            r = test4(a.work, a.chains, a.warmup, a.samples)
        else:
            continue
        print(f"=== 시험 {t} ({time.time() - t0:.1f}초)")
        print(json.dumps(r, ensure_ascii=False, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
