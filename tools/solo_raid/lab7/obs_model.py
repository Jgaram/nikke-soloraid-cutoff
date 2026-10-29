"""관측 층 (WP3b 1단계, 2026-10-29) — 베이지안 계층 모형 M1·M2·M3 공통. 곡면 꼴은 모르고, 곡면 값(mu·P)만 받아 줄 가능도를 낸다.

  python3 -I tools/solo_raid/lab7/obs_model.py        # 자체 점검 없음 — 시험은 check_obs.py

설계서 1절 그대로. docs/solo-raid-method.md 「WP3 명세」 2절. 쓰는 곳: m1.py·(M2·M3 세션)·check_obs.py.

쓰는 순서 (후보 모형 쪽)
  import obs_model as OM                  # sys.path.insert(0, HERE) 뒤. jax를 다른 데서 먼저 import하지 않는다
  OM.setup_host(4)                        # 실행 진입점에서, 모든 jax 계산 전에 (모듈 import 때는 안 부른다)
  head, rows = obs_table.read_cell(path)  # 관측 표
  err = OM.load_err()
  sel = OM.select_rows(rows, head["H"], stage1=True)
  obs = OM.prepare(rows, head["H"], head["season"], err, K=16, delay_key=None, stage1=True)  # numpy dict
  jobs = OM.to_jax(obs)                   # jnp dict (meta·counts 는 빠진다)
  def model(obs):                         # numpyro 모형
      ... theta 표집 ...
      mu, P = surface(theta, obs["y"], obs["u"])      # 곡면 함수 약속 ↓
      OM.observe(obs, mu, P, sigma0, rho)
  samples, info = OM.run_nuts(model, {"obs": jobs}, seed=0)      # samples[name] = [chains, draws, ...]
  diag = OM.diagnose(samples, names=[...])                       # R-hat·ESS·conv

곡면 함수 약속 (후보 쪽이 지킨다)
  입력 obs["y"][N] (ln 점수, 억), obs["u"][N,K] (시각 / 시즌 길이, 0.02~1).
  출력 mu[3,N,K], P[3,N,K] — 서버 성분 KR·JP·GL(0·1·2)마다, 그 줄·노드의 곡면 log10 % (= z_fin − P)와 그 자리 침수 P >= 0 (log10 단위).
  observe(obs, mu, P, sigma0, rho)에는 위 둘과 스칼라 sigma0·rho(잔차 sd = sqrt(sigma0^2 + (rho P)^2))를 넘긴다.
  eps(오염)·phi(사이트별 서버 몫)는 observe가 표집한다 — 후보 쪽은 건드리지 않는다.

줄 늘어놓기: prepare는 노드 하나뿐인 줄(A·post·한 노드 타이머)을 앞 obs["n1"]줄에, 노드 여럿 줄을 그 뒤에 놓고 각 칸을 채움 줄로 채운다 (FIX_N1·FIX_N2 칸).
  실제 줄 i(meta[i])는 행 obs["pos"][i]에 있다. row_loglik은 앞 n1줄은 노드 0번만 계산한다 (같은 값, 약 35% 빠름).
  obs는 모형 안에서 닫아(클로저·kwargs) 쓰고 jit 인자로 넘기지 않는다 (n1·fc_scale 같은 파이썬 값이 추적되어 깨진다).
  row_posteriors/flags가 돌려주는 줄 배열은 meta 순서 (pos로 이미 골랐다).
CPU 시간 감: S30 전체 표 441줄(N=512)에서 4사슬 x (500+500)는 M1 kuma 꼴로 약 4.5분 (check_obs.py 3번).
수치 약속: 모든 jax 계산은 float32. 구간 중도절단 정규는 logdiff_ndtr(안정식), 안 쓰는 노드의 log 무게는 -inf가 아니라 -1e4.
공개 규칙: 글 번호·원문·닉네임·원글 주소는 코드·출력에 쓰지 않는다 (줄 식별은 key·pid·mid만).
"""
import collections
import json
import os
import sys
import time

import numpy as np
from scipy import special as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import err_1b  # noqa: E402  (group_of만 쓴다)

ERR_PATH = os.path.join(C.ROOT, "data", "solo-raid", "analysis", "r7", "obs_error.json")
FIX_N1 = (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024)      # 노드 1개 줄 칸 (A·post·한 노드 타이머) + 채움
FIX_N2 = (64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048)  # 노드 여럿 줄 칸
NEG = -1e4
SITES = ("dc", "naver", "blabla")
SERVERS = ("KR", "JP", "GL")
DELAY_GROUPS = ("timer", "now", "clock")
ABC = ("A", "B", "C")
Q_TARGETS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)


def load_err(path=ERR_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- 실행 환경 ----------

def setup_host(n=4):
    """CPU 장치 수 지정 + float32 고정. 다른 jax 계산(배열 만들기 포함)보다 먼저 부른다 (import jax만은 괜찮다)."""
    import numpyro
    numpyro.set_host_device_count(n)
    import jax
    jax.config.update("jax_enable_x64", False)


# ---------- 1.1 줄 고르기 ----------

def select_rows(rows, H, stage1=False):
    """쓸 줄 목록 (복사본에 tg·post 추가). 기본 = kind A·B·C, lo_reason None, rep. stage1이면 종료 뒤 화면 줄도 더한다."""
    out = []
    seen = set()
    for r in rows:
        if r["kind"] not in ABC:
            continue
        post = False
        if r["lo_reason"] is None and r["rep"]:
            pass
        elif stage1 and r["lo_reason"] == "skip_day_post" and r["use"] and r["pct"] is not None \
                and r["score"] is not None and "best_record" not in (r["flag"] or []) \
                and (r["server"] is None or r["server"] in SERVERS):
            k = (r["pid"], round(r["pct"], 2))
            if k in seen:
                continue
            seen.add(k)
            post = True
        else:
            continue
        if r["pct"] is None or r["score"] is None or r["score"] <= 0:
            continue
        x = dict(r)
        x["post"] = post
        if post:
            x["tg"] = "post"
        elif r["kind"] == "A":
            x["tg"] = "A"
        elif r["kind"] == "B":
            g = err_1b.group_of(r["t_basis"])
            x["tg"] = g if g in DELAY_GROUPS else "delay"
        else:
            x["tg"] = "delay"
        out.append(x)
    return out


# ---------- 1.2 시각 노드 ----------

def _mixcdf(x, comps):
    """comps = [(w, kind, a, b)] — lognorm: ln x ~ N(a,b) (x<=0 에서 0), norm: N(a,b)."""
    x = np.asarray(x, float)
    tot = np.zeros_like(x)
    for w, kind, a, b in comps:
        if kind == "lognorm":
            xs = np.maximum(x, 1e-300)
            c = np.where(x > 0, sps.ndtr((np.log(xs) - a) / b), 0.0)
        else:
            c = sps.ndtr((x - a) / b)
        tot = tot + w * c
    return tot


def _quad(comps, xa, xb, K):
    """[xa, xb]로 자른 분포의 (j-1/2)/K 분위 노드. (노드, 사유 또는 None)."""
    fa, fb = float(_mixcdf(np.array([xa]), comps)[0]), float(_mixcdf(np.array([xb]), comps)[0])
    j = (np.arange(1, K + 1) - 0.5) / K
    if fb - fa < 1e-9:
        return xa + j * (xb - xa), "flat_support"
    tau = fa + j * (fb - fa)
    lo = np.full(K, xa, float)
    hi = np.full(K, xb, float)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        up = _mixcdf(mid, comps) < tau
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi), None


def _pad(t, w, K):
    k = len(t)
    if k >= K:
        return t[:K], w[:K] / w[:K].sum()
    return np.concatenate([t, np.full(K - k, t[0])]), np.concatenate([w, np.zeros(K - k)])


def support(r, H):
    """(Lo, Up) 시간 지지 또는 None 항목. A·post는 쓰지 않는다."""
    tg = r["tg"]
    hp = r.get("hp")
    t7lo, t7hi = r.get("h_lo_top7"), r.get("h_hi_top7")
    ups = [H] + [v for v in (hp, t7hi) if v is not None]
    los = [0.0] + [v for v in (t7lo,) if v is not None]
    if tg == "delay" and r.get("h_lo") is not None:
        los.append(r["h_lo"])
    Up = min(ups) + (2.0 / 60 if tg in DELAY_GROUPS else 0.0)
    return max(los), Up


def nodes_detail(r, H, err, delay_key, K):
    """(t[K], w[K], 사유 또는 None) — 사유: bad_support·timer_fallback·flat_support. 지지가 비면 (None, None, "bad_support")."""
    tg = r["tg"]
    if tg == "A":
        return _pad(np.array([r["h"]], float), np.array([1.0]), K) + (None,)
    if tg == "post":
        return _pad(np.array([float(H)]), np.array([1.0]), K) + (None,)
    if tg == "delay" and r.get("hp") is None:
        return None, None, "bad_support"
    if tg != "delay" and r.get("h") is None:
        return None, None, "bad_support"
    if tg in DELAY_GROUPS and r.get("hp") is None:
        return None, None, "bad_support"
    Lo, Up = support(r, H)
    if Lo > Up:
        return None, None, "bad_support"
    b1 = err["1b_stated_time"]
    h = r["h"]
    if tg == "timer":
        rare = b1["timer"]["rare"]
        pw = rare["pi"]["p"]
        g = rare["p_k_ge_2_given_wrong"]["p"]
        ks = np.arange(0, K)
        t = h - 24.0 * ks
        w = np.where(ks == 0, 1 - pw, pw * (1 - g) * g ** (np.maximum(ks, 1) - 1))
        ok = (t >= Lo - 1e-12) & (t <= Up + 1e-12)
        reason = None
        if not ok.any():
            t, w, reason = np.array([h]), np.array([1.0]), "timer_fallback"
        else:
            t, w = t[ok], w[ok] / w[ok].sum()
        return _pad(t, w, K) + (reason,)
    if tg == "now":
        p = b1["now"]["fit"]["params"]
        comps = [(1 - p["pi_tail"], "lognorm", p["mu1"], p["sigma1"]), (p["pi_tail"], "lognorm", p["mu2"], p["sigma2"])]
        xa, xb = max(0.0, (h - Up) * 60), (h - Lo) * 60
        if xb < xa:
            return None, None, "bad_support"
        x, reason = _quad(comps, xa, xb, K)
        return h - x / 60, np.full(K, 1.0 / K), reason
    if tg == "clock":
        p = b1["clock"]["fit"]["params"]
        comps = [(1 - p["pi_wide"], "norm", p["mu"], p["sigma_core"]), (p["pi_wide"], "norm", p["mu"], p["sigma_wide"])]
        xa, xb = (h - Up) * 60, (h - Lo) * 60
        x, reason = _quad(comps, xa, xb, K)
        return h - x / 60, np.full(K, 1.0 / K), reason
    # delay
    a1 = err["1a_post_delay"]
    p = a1["expanding"].get(delay_key) if delay_key else None
    p = p or a1["recommend"]["params"]
    comps = [(p["w_fast"], "lognorm", p["mu1"], p["sigma1"]), (1 - p["w_fast"], "lognorm", p["mu2"], p["sigma2"])]
    hp = r["hp"]
    xa, xb = max(0.0, hp - Up), hp - Lo
    x, reason = _quad(comps, xa, xb, K)
    return hp - x, np.full(K, 1.0 / K), reason


def nodes_for(r, H, err, delay_key, K):
    """줄 r (tg 붙은 것)의 촬영 시각 노드 (t[K'], w[K']), K' <= K. 지지가 비면 (None, None). 안 쓰는 칸은 무게 0 (K로 채움)."""
    t, w, _ = nodes_detail(r, H, err, delay_key, K)
    return t, w


# ---------- 1.3 묶음 ----------

ELAB_KEY = {"top7": "top7", "text": "text", "text+top7": "text+top7", "top7_1": "top7_1",
            "top7_post": "top7_post(+text)", "text+top7_post": "top7_post(+text)", "fix": "top7"}


def _elab(r, err):
    if r["server"] not in SERVERS:
        return 0.0
    key = ELAB_KEY.get(r["server_basis"], "text")
    return float(err["1e_server"]["label_error_by_basis"][key]["beta_central"]["mean"])


def prepare(rows, H, season, err, K=16, delay_key=None, stage1=False):
    """줄 고르기 + 노드 + 채움 → Obs (numpy dict). delay_key = 1a expanding 시즌 키 (맞힐 시즌, 앞 시즌 값) 또는 None(전체 값)."""
    sel = select_rows(rows, H, stage1)
    cnt = collections.Counter()
    nodes_reason = collections.Counter()
    keep = []
    for r in sel:
        t, w, why = nodes_detail(r, H, err, delay_key, K)
        if why:
            nodes_reason[why] += 1
        if t is None:
            cnt["drop_bad_support"] += 1
            continue
        keep.append((r, t, w))
    n = len(keep)
    nodes_n = [int((w > 0).sum()) for _, _, w in keep]
    single = [i for i in range(n) if nodes_n[i] <= 1]
    multi = [i for i in range(n) if nodes_n[i] > 1]
    N1 = next((m for m in FIX_N1 if m >= len(single)), ((len(single) + 1023) // 1024) * 1024)
    N2 = next((m for m in FIX_N2 if m >= len(multi)), ((len(multi) + 1023) // 1024) * 1024)
    N = N1 + N2
    pos = np.zeros(n, int)                      # keep[i]가 놓인 행 번호
    for j, i in enumerate(single):
        pos[i] = j
    for j, i in enumerate(multi):
        pos[i] = N1 + j
    y = np.zeros(N)
    zlo, zhi = np.full(N, -1.0), np.zeros(N)
    u, tt = np.full((N, K), 0.5), np.zeros((N, K))
    lw = np.full((N, K), NEG)
    lw[:, 0] = 0.0
    site, lab, elab = np.zeros(N, int), np.zeros(N, int), np.zeros(N)
    kind, mask = np.zeros(N, int), np.zeros(N)
    meta = []
    for i0, (r, t, w) in enumerate(keep):
        i = pos[i0]
        y[i] = np.log(r["score"])
        p = r["pct"]
        zlo[i] = np.log10(p - 0.005) if p - 0.005 > 0 else -30.0
        zhi[i] = np.log10(p + 0.01)
        tt[i] = t
        u[i] = np.clip(t / H, 0.02, 1.0)
        lw[i] = np.where(w > 0, np.log(np.maximum(w, 1e-300)), NEG)
        site[i] = SITES.index(r["site"]) if r["site"] in SITES else 3
        lab[i] = SERVERS.index(r["server"]) if r["server"] in SERVERS else -1
        elab[i] = _elab(r, err)
        kind[i] = 3 if r["post"] else "ABC".index(r["kind"])
        mask[i] = 1.0
        if r["tg"] == "A":
            Lo = Up = r["h"]
        elif r["tg"] == "post":
            Lo = Up = float(H)
        else:
            Lo, Up = support(r, H)
        meta.append({"key": r["key"], "pid": r["pid"], "kind": r["kind"], "tg": r["tg"], "server": r["server"],
                     "site": r["site"], "h": r["h"], "hp": r["hp"], "Lo": Lo, "Up": Up})
        cnt["sel_" + r["tg"]] += 1
    fill = mask == 0
    if n:
        y[fill] = y[~fill].mean()
    cnt["selected"] = n
    cnt["N"] = N
    cnt["N_single_block"] = N1
    dsite = err["1e_server"]["unknown_row_site_dirichlet"]
    dirichlet = np.array([[dsite[s]["alpha"][k] for k in SERVERS] for s in ("dc", "naver", "blabla", "*")])
    cd = err["1d_contamination"]
    return {"season": season, "H": float(H), "K": K, "n": n, "N": N, "n1": N1, "pos": pos,
            "y": y, "zlo": zlo, "zhi": zhi, "u": u, "lw": lw, "t": tt, "site": site, "lab": lab, "elab": elab,
            "kind": kind, "mask": mask, "meta": meta,
            "counts": {"rows": dict(cnt), "node_reasons": dict(nodes_reason)},
            "dirichlet": dirichlet,
            "eps_ab": (cd["contamination_weight_photo"]["prior_central"]["a"], cd["contamination_weight_photo"]["prior_central"]["b"]),
            "fc_scale": float(cd["contaminant_density"]["family_fits"]["laplace"]["params(shape..., loc, scale)"][-1])}


NUMERIC = ("y", "zlo", "zhi", "u", "lw", "t", "site", "lab", "elab", "kind", "mask", "dirichlet")


def to_jax(obs):
    """numpy Obs → jnp (float32/int32) dict. 배열만 옮기고 스칼라(eps_ab·fc_scale·K·N)는 파이썬 값으로 둔다."""
    import jax.numpy as jnp
    out = {}
    for k in NUMERIC:
        a = np.asarray(obs[k])
        out[k] = jnp.asarray(a, jnp.int32 if a.dtype.kind in "iu" else jnp.float32)
    for k in ("eps_ab", "fc_scale", "K", "N", "n", "n1", "H"):
        out[k] = obs[k]
    out["pos"] = np.asarray(obs["pos"])
    return out


# ---------- 1.4 가능도 ----------

def logdiff_ndtr(a, b):
    """log(Phi(b) - Phi(a)), a < b — float32 안정. a + b > 0이면 (-b, -a)로 뒤집어 log_ndtr(b) + log(-expm1(log_ndtr(a) - log_ndtr(b)))."""
    import jax.numpy as jnp
    from jax.scipy.special import log_ndtr
    flip = (a + b) > 0
    a2 = jnp.where(flip, -b, a)
    b2 = jnp.where(flip, -a, b)
    lb = log_ndtr(b2)
    d = jnp.minimum(log_ndtr(a2) - lb, -1e-30)   # 0 이하로 클립 (a=b여도 -inf가 아니라 약 -69)
    return lb + jnp.log(-jnp.expm1(d))


def _logpi(obs, phi):
    import jax.numpy as jnp
    phis = phi[obs["site"]]                                    # [N,3]
    lab = obs["lab"]
    labc = jnp.clip(lab, 0, 2)
    oh = jnp.arange(3)[None, :] == labc[:, None]
    ph_lab = jnp.take_along_axis(phis, labc[:, None], 1)
    other = jnp.maximum(phis.sum(1, keepdims=True) - ph_lab, 1e-12)
    el = obs["elab"][:, None]
    pi_lab = jnp.where(oh, 1.0 - el, el * phis / other)
    pi = jnp.where((lab < 0)[:, None], phis, pi_lab)
    return jnp.log(jnp.maximum(pi, 1e-30))


def _AB(zlo, zhi, mu, P, lw, sigma0, rho, fc_scale):
    """노드 합 A = logsumexp_n(lw + logPr), B = logsumexp_n(lw + logfc) -> [3,N]. 입력: zlo·zhi [N], mu·P [3,N,K], lw [N,K]."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    sig = jnp.sqrt(sigma0 ** 2 + (rho * P) ** 2)
    zl, zh = zlo[None, :, None], zhi[None, :, None]
    logpr = logdiff_ndtr((zl - mu) / sig, (zh - mu) / sig)
    logfc = -jnp.abs(0.5 * (zl + zh) - mu) / fc_scale - jnp.log(2 * fc_scale) + jnp.log(zh - zl)
    return logsumexp(lw[None] + logpr, axis=2), logsumexp(lw[None] + logfc, axis=2), logpr, logfc


def row_loglik(obs, mu, P, sigma0, rho, eps, phi, parts=False):
    """줄마다 로그 가능도 ll[N] (채움 줄 0). parts=True면 dict(ll, p_contam, p_server, t_mean).
    obs["n1"]이 있으면 앞 n1줄은 노드 하나(0번)만 있는 줄이라 그 노드만 계산한다 (prepare가 그렇게 늘어놓는다; 시간 절약, 값은 같다)."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    s = obs["fc_scale"]
    n1 = obs.get("n1", 0)
    if parts or n1 == 0:
        A, B, logpr, logfc = _AB(obs["zlo"], obs["zhi"], mu, P, obs["lw"], sigma0, rho, s)
    else:
        A1, B1, _, _ = _AB(obs["zlo"][:n1], obs["zhi"][:n1], mu[:, :n1, :1], P[:, :n1, :1], obs["lw"][:n1, :1], sigma0, rho, s)
        A2, B2, _, _ = _AB(obs["zlo"][n1:], obs["zhi"][n1:], mu[:, n1:], P[:, n1:], obs["lw"][n1:], sigma0, rho, s)
        A, B = jnp.concatenate([A1, A2], axis=1), jnp.concatenate([B1, B2], axis=1)
    le, l1e = jnp.log(eps), jnp.log1p(-eps)
    comp = jnp.logaddexp(l1e + A, le + B)
    lp = _logpi(obs, phi)                                       # [N,3]
    joint = lp.T + comp                                         # [3,N]
    ll = logsumexp(joint, axis=0) * obs["mask"]
    if not parts:
        return ll
    tot = logsumexp(joint, axis=0)
    p_c = jnp.exp(logsumexp(lp.T + le + B, axis=0) - tot)
    p_s = jnp.exp(joint - tot).T
    nodell = jnp.logaddexp(l1e + logpr, le + logfc)             # [3,N,K]
    lnode = obs["lw"] + logsumexp(lp.T[:, :, None] + nodell, axis=0)
    pn = jnp.exp(lnode - logsumexp(lnode, axis=1, keepdims=True))
    t_mean = (pn * obs["t"]).sum(1)
    return {"ll": ll, "p_contam": p_c, "p_server": p_s, "t_mean": t_mean}


def observe(obs, mu, P, sigma0, rho, keep_ll=False):
    """numpyro 모형 안에서: eps·phi 표집 + 관측 가능도 factor."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    eps = numpyro.sample("eps", dist.Beta(*[jnp.float32(v) for v in obs["eps_ab"]]))
    phi = numpyro.sample("phi", dist.Dirichlet(obs["dirichlet"]))
    ll = row_loglik(obs, mu, P, sigma0, rho, eps, phi)
    numpyro.factor("obs_ll", ll.sum())
    if keep_ll:
        numpyro.deterministic("ll", ll)


# ---------- 1.5 실행 도구 ----------

def run_nuts(model, model_kwargs, seed=0, chains=4, warmup=500, samples=500, target_accept=0.9,
             max_tree_depth=10, dense=True, init=None):
    """NUTS (한 프로세스 chain_method='parallel'). (samples dict[name -> ndarray[chains, draws, ...]], info)."""
    import jax
    from numpyro.infer import MCMC, NUTS, init_to_median
    kern = NUTS(model, target_accept_prob=target_accept, max_tree_depth=max_tree_depth, dense_mass=dense,
                init_strategy=init if init is not None else init_to_median(num_samples=15))
    mcmc = MCMC(kern, num_warmup=warmup, num_samples=samples, num_chains=chains, chain_method="parallel",
                progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"), **model_kwargs)
    sm = mcmc.get_samples(group_by_chain=True)
    ex = mcmc.get_extra_fields(group_by_chain=True)
    jax.block_until_ready((sm, ex))   # jax는 비동기라 이걸 기다려야 걸린 시간이 맞다
    secs = time.time() - t0
    steps = np.asarray(ex["num_steps"])
    info = {"secs": round(secs, 1), "divergences": int(np.asarray(ex["diverging"]).sum()),
            "tree_cap_frac": float((steps >= 2 ** max_tree_depth - 1).mean()), "mean_leapfrog": float(steps.mean()),
            "chains": chains, "warmup": warmup, "samples": samples, "target_accept": target_accept}
    return {k: np.asarray(v) for k, v in sm.items()}, info


def diagnose(samples, names=None, rhat_max=1.01, ess_min=400, divergences=0):
    """이름마다 분할 R-hat 최대·벌크 ESS 최소. conv = R-hat < 1.01 and 발산 0 and ESS >= 400 (발산은 호출자가 divergences로 준다)."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    names = names or [k for k in samples if k not in ("ll", "phi")]
    per = {}
    for k in names:
        a = np.asarray(samples[k])
        per[k] = {"rhat": float(np.nanmax(np.asarray(split_gelman_rubin(a)))),
                  "ess": float(np.nanmin(np.asarray(effective_sample_size(a))))}
    rmax = max(v["rhat"] for v in per.values())
    emin = min(v["ess"] for v in per.values())
    return {"per": per, "rhat_max": rmax, "ess_min": emin, "divergences": divergences,
            "conv": bool(rmax < rhat_max and divergences == 0 and emin >= ess_min)}


def fit_with_retry(model, model_kwargs, names=None, seed=0, chains=4, warmup=500, samples=500, target_accept=0.9,
                   max_tree_depth=10, dense=True, init=None):
    """run_nuts → diagnose → 못 지나면 target_accept 0.95·warmup 1000으로 한 번 더. (samples, info, diag) — info["retried"]."""
    sm, info = run_nuts(model, model_kwargs, seed, chains, warmup, samples, target_accept, max_tree_depth, dense, init)
    dg = diagnose(sm, names, divergences=info["divergences"])
    info["retried"] = False
    if dg["conv"]:
        return sm, info, dg
    first = info["secs"]
    sm, info = run_nuts(model, model_kwargs, seed + 1, chains, 1000, samples, 0.95, max_tree_depth, dense, init)
    dg = diagnose(sm, names, divergences=info["divergences"])
    info["retried"] = True
    info["secs_first_try"] = first
    return sm, info, dg


def row_posteriors(obs, samples, surface_fn, max_draws=400):
    """사후 뽑기(최대 max_draws, 고르게)마다 row_loglik(parts=True)의 평균. surface_fn(draw: dict[name -> 한 뽑기 값], jobs) → (mu, P, sigma0, rho).
    samples에는 모형 표집 이름 + eps·phi가 있어야 한다. obs는 to_jax 결과에 prepare의 "pos"를 더한 것 (to_jax가 pos를 np 배열로 둔다).
    실제 줄(n개)만 meta 순서로 돌려준다."""
    import jax
    import jax.numpy as jnp
    flat = {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in samples.items() if k != "ll"}
    tot = len(next(iter(flat.values())))
    idx = np.unique(np.linspace(0, tot - 1, min(max_draws, tot)).astype(int))

    @jax.jit
    def one(draw):
        mu, P, s0, rho = surface_fn(draw, obs)
        return row_loglik(obs, mu, P, s0, rho, draw["eps"], draw["phi"], parts=True)

    acc = None
    for i in idx:
        r = one({k: jnp.asarray(v[i]) for k, v in flat.items()})
        r = {k: np.asarray(v) for k, v in r.items() if k != "ll"}
        acc = r if acc is None else {k: acc[k] + r[k] for k in r}
    pos = np.asarray(obs["pos"])
    return {k: v[pos] / len(idx) for k, v in acc.items()}


def flags(meta, rp, thr=0.5):
    """사후 오염 확률이 thr를 넘는 실제 줄 목록 (r7/flag_{M}_{판}.json 용). meta = prepare의 obs["meta"]."""
    out = []
    for i, m in enumerate(meta):
        if rp["p_contam"][i] > thr:
            out.append({"key": m["key"], "pid": m["pid"], "kind": m["kind"], "tg": m["tg"], "server": m["server"],
                        "p_contam": round(float(rp["p_contam"][i]), 4), "p_kr": round(float(rp["p_server"][i][0]), 4)})
    return out


def pred_cells(season, T, draws_lnC, conv, server="KR"):
    """score.py 예측 파일 꼴의 칸 목록. draws_lnC = {q(%): ln 컷 뽑기 배열}. pred = exp(중앙), i80 = 10·90%, i95 = 2.5·97.5%."""
    cells = []
    for q in sorted(draws_lnC):
        d = np.asarray(draws_lnC[q]).ravel()
        lo80, med, hi80, lo95, hi95 = np.quantile(d, [0.1, 0.5, 0.9, 0.025, 0.975])
        cells.append({"season": season, "T": float(T), "pct": float(q), "server": server, "pred": float(np.exp(med)),
                      "i80": [float(np.exp(lo80)), float(np.exp(hi80))], "i95": [float(np.exp(lo95)), float(np.exp(hi95))],
                      "own": True, "conv": bool(conv)})
    return cells
