"""WP5a 진단 2 — M3 관측 층의 잔차가 줄마다 독립인가: 같은 사람(pid)·같은 시간대 안의 상관 (docs/solo-raid-method.md 「WP3 명세」 2절, 「WP3b M3 결과」).
새 MCMC 없음, 정답 표(answer_key) 안 읽음. 조정 6시즌의 끝난 시즌 전체 관측 표 + 1단계 사후 평균 하나로 곡면 mu_i·sigma_i를 낸다.

  python3 -I tools/solo_raid/lab7/obs_table.py build --out W/obs7 --full          # 먼저 (약 10초)
  python3 -I tools/solo_raid/lab7/wp5a_diag2.py --obs-dir W/obs7/full             # W = 레포 밖 스크래치

입력 (읽기만): r7/stage1_M3_D1.json (라플라스 평균; 곡선 r12·l12는 정방 B의 역으로 계수로 되돌린다 — m3.prior와 같은 셈),
              관측 표 W/obs7/full/{시즌}.jsonl.gz.
출력: r7/wp5a/diag2_D1.json, r7/wp5a/diag2_D1.md   (숫자만 — 글 번호·원문·닉네임·pid 값은 쓰지 않는다)

줄: m3.check43_season과 같은 선택(OM.prepare, stage1=True, 지연 열쇠 규칙) 위에서
  집합 1 = 한섭 라벨 + kind A(시각 정확, tg A) + 사후 오염 확률 <= 0.5   (사후 오염: eps·phi는 사전 평균, 평균 모수 하나로 row_loglik(parts=True))
  집합 2 = 서버 미상 포함 A·B·C 전체 (tg=post 뺌) + 오염 <= 0.5; 시각 = 사후 평균 촬영 시각, 서버 성분 = P(서버) 최대.
잔차 r_i = (관측 % 구간 가운데의 log10 - mu_i) / sigma_i, sigma_i = sqrt(sigma0^2 + (rho P_i)^2).
  구간 중도절단(관측은 [log10(p-0.005), log10(p+0.01)])이라 가운데 값을 쓰는 근사다.
묶음 통계는 시즌 안에서만 센다 (pid·mid는 시즌마다 다른 열쇠). 시즌 평균은 층(stratum)으로 빼서 센다 (a가 시즌별로 흡수하는 몫).
부트스트랩은 시즌 안에서 묶음(사람·시간 칸)을 다시 뽑는 500번, 시드 고정.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
import common as C  # noqa: E402,F401
import m3 as M3  # noqa: E402
import obs_model as OM  # noqa: E402
import obs_table  # noqa: E402

R7 = os.path.join(ROOT, "data", "solo-raid", "analysis", "r7")
OUTD = os.path.join(R7, "wp5a")
TUNE = ["S22", "S26", "S30", "S34", "S38", "S42"]
NBOOT = 500
SEED = 20261031
LAGS = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 48)]
UBINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001)]
BINS_H = (3, 6, 24)


# ---------- 줄 만들기 ----------

def theta_from_stage1(e):
    """1단계 사후 평균 -> 곡면 모수. r12·l12 = 평가 격자의 곡선 값 -> 계수 = 정방 B의 역."""
    nm = e["names"]
    mean = np.asarray(e["mean"], float)
    g = lambda k: float(mean[nm.index(k)])
    r12 = np.array([g(f"r{i}") for i in range(M3.NE_R)])
    l12 = np.array([g(f"l{i}") for i in range(12)])
    th = {"a": g("a"), "cr": np.linalg.solve(M3.BASIS["Br12"], r12), "cl": np.linalg.solve(M3.BASIS["Bl12"], l12)}
    for k in M3.NUIS:
        th[k] = g(k)
    return th


def season_rows(s, e, err, obs_dir):
    """한 시즌의 줄별 (사후 평균 모수 하나로) 곡면·잔차. 돌려주는 것은 meta 순서 배열 dict."""
    import jax
    import jax.numpy as jnp
    head, rows = obs_table.read_cell(os.path.join(obs_dir, f"{s}.jsonl.gz"))
    H = head["H"]
    dk = s if s in err["1a_post_delay"]["expanding"] else None
    obs = OM.prepare(rows, H, s, err, K=16, delay_key=dk, stage1=True)
    jobs = OM.to_jax(obs)
    meta = obs["meta"]
    pos = np.asarray(obs["pos"])
    n = len(meta)
    th = {k: (jnp.asarray(v, jnp.float32) if not np.isscalar(v) else jnp.float32(v)) for k, v in theta_from_stage1(e).items()}
    s0, rho = float(np.exp(th["lnsig0"])), float(np.exp(th["lnrho"]))
    ea, eb = obs["eps_ab"]
    phi0 = np.asarray(obs["dirichlet"]) / np.asarray(obs["dirichlet"]).sum(1, keepdims=True)
    samples = {"eps": np.full((1, 1), ea / (ea + eb)), "phi": phi0[None, None].copy()}

    def surf(draw, jb):
        mu, P = M3.surface(th, jb["y"], jb["u"])
        return mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"])

    rp = OM.row_posteriors(jobs, samples, surf, max_draws=1)
    # 사후 평균 촬영 시각에서 곡면 (A 줄은 그 시각 그대로, C 줄은 평균 시각)
    tm = np.asarray(rp["t_mean"], float)
    u0 = np.asarray(jobs["u"]).copy()
    u0[pos, 0] = np.clip(tm / H, 0.02, 1.0)
    jb2 = dict(jobs)
    jb2["u"] = jnp.asarray(u0, jnp.float32)
    mu, P = M3.surface(th, jb2["y"], jb2["u"])
    MU, PP = np.asarray(mu)[:, pos, 0], np.asarray(P)[:, pos, 0]          # [3,n]
    SG = np.sqrt(s0 ** 2 + (rho * PP) ** 2)
    ps = np.asarray(rp["p_server"])                                        # [n,3]
    zobs = 0.5 * (np.asarray(obs["zlo"])[pos] + np.asarray(obs["zhi"])[pos])
    mu_w = np.einsum("kn,nk->n", MU, ps)
    sg_w = np.einsum("kn,nk->n", SG, ps)
    am = ps.argmax(1)
    mu_m, sg_m = MU[am, np.arange(n)], SG[am, np.arange(n)]
    return {
        "n": n, "H": float(H), "s0": s0, "rho": rho,
        "pid": np.array([m["pid"] for m in meta]), "kind": np.array([m["kind"] for m in meta]),
        "tg": np.array([m["tg"] for m in meta]), "server": np.array([m["server"] if m["server"] else "?" for m in meta]),
        "t": tm, "u": np.clip(tm / H, 0.0, 1.0), "z": zobs,
        "p_contam": np.asarray(rp["p_contam"], float), "p_kr": ps[:, 0],
        "mu_w": mu_w, "sg_w": sg_w, "mu_m": mu_m, "sg_m": sg_m,
    }


# ---------- 일원 분산 성분 ----------

class Groups:
    """묶음별 충분통계 (n, 합, 안쪽 제곱합) + 층(시즌). 가중(다시 뽑기 횟수) m으로 ANOVA 적률 추정."""

    def __init__(self, y, g, strat):
        _, gi = np.unique(g, return_inverse=True)
        G = gi.max() + 1
        self.n = np.bincount(gi, minlength=G).astype(float)
        self.sum = np.bincount(gi, weights=y, minlength=G)
        self.mean = self.sum / self.n
        self.ssw = np.bincount(gi, weights=(y - self.mean[gi]) ** 2, minlength=G)
        st = np.zeros(G, int)
        st[gi] = strat
        _, st = np.unique(st, return_inverse=True)       # 빈 층 없이 0..S-1
        self.strat = st
        self.S = int(st.max()) + 1
        self.G = G

    def anova(self, m=None):
        m = np.ones(self.G) if m is None else np.asarray(m, float)
        mn = m * self.n
        Ns = np.bincount(self.strat, weights=mn, minlength=self.S)
        ok = Ns > 0
        ybs = np.bincount(self.strat, weights=m * self.sum, minlength=self.S) / np.where(ok, Ns, 1)
        ssb = (mn * (self.mean - ybs[self.strat]) ** 2).sum()
        ssw = (m * self.ssw).sum()
        Gs = np.bincount(self.strat, weights=m, minlength=self.S)
        dfb = Gs.sum() - ok.sum()
        dfw = mn.sum() - m.sum()
        sq = np.bincount(self.strat, weights=m * self.n ** 2, minlength=self.S)
        n0 = (Ns[ok] - sq[ok] / Ns[ok]).sum() / dfb if dfb > 0 else np.nan
        msb, msw = ssb / dfb, ssw / dfw
        sb2 = (msb - msw) / n0
        icc = sb2 / (sb2 + msw)
        mbar = (m * self.n ** 2).sum() / mn.sum()
        deff = 1.0 + (mbar - 1.0) * max(icc, 0.0)
        return {"icc": icc, "sb2": sb2, "sw2": msw, "mbar": mbar, "deff": deff, "n_eff": mn.sum() / deff,
                "N": int(round(mn.sum())), "G": int(round(m.sum())), "G_multi": int(round((m * (self.n >= 2)).sum()))}

    def reml(self):
        """REML (층마다 평균 GLS) — (sigma_b2, sigma_w2, icc). 안쪽 제곱합 + 묶음 평균으로 쓴 닫힌 꼴의 -2 로그 가능도를 Nelder-Mead로."""
        from scipy.optimize import minimize
        n, ym, ssw, st = self.n, self.mean, self.ssw, self.strat
        N = n.sum()

        def f(x):
            sw2, sb2 = math.exp(x[0]), math.exp(x[1])
            v = sw2 + n * sb2
            w = n / v
            sw = np.bincount(st, weights=w, minlength=self.S)
            mu = np.bincount(st, weights=w * ym, minlength=self.S) / sw
            q = (w * (ym - mu[st]) ** 2).sum()
            return float(((n - 1) * math.log(sw2) + np.log(v)).sum() + ssw.sum() / sw2 + q + np.log(sw).sum())

        a = self.anova()
        x0 = [math.log(max(a["sw2"], 1e-6)), math.log(max(a["sb2"], 1e-4 * a["sw2"]))]
        best = None
        for d in (0.0, -2.0, 1.0):
            rs = minimize(f, [x0[0], x0[1] + d], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
            if best is None or rs.fun < best.fun:
                best = rs
        sw2, sb2 = math.exp(best.x[0]), math.exp(best.x[1])
        return {"sb2": sb2, "sw2": sw2, "icc": sb2 / (sb2 + sw2)}


def boot_mult(rng, strat_of_unit, nb):
    """시즌 안에서 단위(묶음)를 다시 뽑은 횟수 [nb, U]."""
    U = len(strat_of_unit)
    out = np.zeros((nb, U))
    for s in np.unique(strat_of_unit):
        ix = np.where(strat_of_unit == s)[0]
        d = rng.integers(0, len(ix), size=(nb, len(ix)))
        for b in range(nb):
            out[b, ix] = np.bincount(d[b], minlength=len(ix))
    return out


def q90(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))] if len(v) else None


def cluster_stats(y, g, strat, rng, nb=NBOOT, reml=True):
    G = Groups(y, g, strat)
    a = G.anova()
    out = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in a.items()}
    if reml and G.G >= 3:
        out["reml"] = G.reml()
    M = boot_mult(rng, G.strat, nb)
    bs = [G.anova(M[b]) for b in range(nb)]
    out["icc_ci90"] = q90([b["icc"] for b in bs])
    out["deff_ci90"] = q90([b["deff"] for b in bs])
    out["n_eff_ci90"] = q90([b["n_eff"] for b in bs])
    out["icc_boot_mean"] = float(np.nanmean([b["icc"] for b in bs]))
    return out


# ---------- 짝 통계 ----------

def wpearson(x, y, w):
    """대칭화한 짝 (x,y) 와 (y,x) 모두 센 가중 상관."""
    sw = w.sum()
    if sw <= 0:
        return np.nan
    m = (w * (x + y) / 2).sum() / sw
    xc, yc = x - m, y - m
    c = (w * xc * yc).sum() / sw
    v = (w * (xc ** 2 + yc ** 2) / 2).sum() / sw
    return c / v if v > 0 else np.nan


def pair_corr_boot(I, J, x, y, pg, M, extra=None):
    """짝 (I,J 는 사람 번호) 의 상관 + 사람 다시 뽑기. 짝 가중 = m_I m_J."""
    pt = wpearson(x, y, np.ones(len(x)))
    if M is None:
        return {"n_pairs": int(len(x)), "corr": float(pt)}
    bs = np.array([wpearson(x, y, M[b][I] * M[b][J]) for b in range(M.shape[0])])
    return {"n_pairs": int(len(x)), "corr": float(pt), "ci90": q90(bs)}


def season_pairs(d, mask, rc):
    """시즌 안 모든 (i<j) 짝 (다른 사람 아닌 것도 포함; 호출자가 거른다). -> I, J (전체 줄 번호)."""
    out_i, out_j = [], []
    for s in np.unique(d["season"][mask]):
        ix = np.where(mask & (d["season"] == s))[0]
        a, b = np.triu_indices(len(ix), 1)
        out_i.append(ix[a])
        out_j.append(ix[b])
    return np.concatenate(out_i), np.concatenate(out_j)


# ---------- 한 집합의 모든 잴 것 ----------

def analyze(d, mask, label, seed):
    """d = 줄 배열 dict (season, person[전체 번호], t, u, z, r, rc[시즌 평균 뺀 r]), mask = 쓸 줄."""
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    D = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in d.items()}
    seasons = sorted(set(D["season_name"]))
    res = {"label": label, "n": int(len(idx))}
    # 사람 번호를 0..P-1로 다시 매김
    _, pers = np.unique(D["person"], return_inverse=True)
    D["pers"] = pers
    sidx = np.array([seasons.index(x) for x in D["season_name"]])
    D["sidx"] = sidx
    P = pers.max() + 1
    pers_season = np.zeros(P, int)
    pers_season[pers] = sidx

    def per_scope(fn):
        """시즌별 + 합."""
        out = {}
        for s in seasons:
            m = sidx == seasons.index(s)
            out[s] = fn(m)
        out["합"] = fn(np.ones(len(idx), bool))
        return out

    r, rc = D["r"], D["rc"]

    # 5. r 의 sd, u 구간별
    def rstat(m):
        x = r[m]
        o = {"n": int(m.sum()), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "rms": float(np.sqrt((x ** 2).mean())),
             "robust_sd": float(1.4826 * np.median(np.abs(x - np.median(x)))), "share_abs_gt2": float((np.abs(x) > 2).mean())}
        u = D["u"][m]
        o["by_u"] = []
        for lo, hi in UBINS:
            k = (u >= lo) & (u < hi)
            o["by_u"].append({"u": [lo, min(hi, 1.0)], "n": int(k.sum()), "mean": float(x[k].mean()) if k.sum() else None,
                              "sd": float(x[k].std(ddof=1)) if k.sum() > 1 else None})
        return o

    res["r_stats"] = per_scope(rstat)

    # 1. 사람 묶음
    def psize(m):
        c = np.bincount(pers[m], minlength=P)
        c = c[c > 0]
        return {"persons": int(len(c)), "persons_ge2": int((c >= 2).sum()), "rows_in_ge2": int(c[c >= 2].sum()),
                "size": {"1": int((c == 1).sum()), "2": int((c == 2).sum()), "3-5": int(((c >= 3) & (c <= 5)).sum()), "6+": int((c >= 6).sum())},
                "max": int(c.max())}

    res["person_sizes"] = per_scope(psize)
    pc = {}
    for s in seasons + ["합"]:
        m = np.ones(len(idx), bool) if s == "합" else sidx == seasons.index(s)
        pc[s] = cluster_stats(rc[m], pers[m], sidx[m], rng, reml=True)
    res["person_icc"] = pc

    # 같은 사람 연속 두 줄 잔차의 상관 (시간 순)
    order = np.lexsort((D["t"], pers))
    pi_, pj_ = [], []
    for a, b in zip(order[:-1], order[1:]):
        if pers[a] == pers[b] and D["t"][b] > D["t"][a]:
            pi_.append(a)
            pj_.append(b)
    pi_, pj_ = np.array(pi_, int), np.array(pj_, int)
    Mp = boot_mult(rng, pers_season, NBOOT)
    cp = {}
    for s in seasons + ["합"]:
        m = np.ones(len(pi_), bool) if s == "합" else sidx[pi_] == seasons.index(s)
        if m.sum() < 3:
            cp[s] = {"n_pairs": int(m.sum()), "corr": None}
            continue
        x, y = rc[pi_[m]], rc[pj_[m]]
        # 연속 짝은 순서가 있으므로 대칭화 없이 일반 상관 + 사람 다시 뽑기
        def pc_(w):
            sw = w.sum()
            mx, my = (w * x).sum() / sw, (w * y).sum() / sw
            return (w * (x - mx) * (y - my)).sum() / math.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())
        pt = pc_(np.ones(len(x)))
        pp = pers[pi_[m]]
        bs = np.array([pc_(Mp[b][pp]) for b in range(NBOOT)])
        cp[s] = {"n_pairs": int(m.sum()), "persons": int(len(np.unique(pp))), "corr": float(pt), "ci90": q90(bs)}
    res["same_person_consecutive"] = cp

    # 2. 시간 묶음
    tb = {}
    for hb in BINS_H:
        key = f"{hb}h"
        g = sidx * 100000 + np.floor(D["t"] / hb).astype(int)
        o = {}
        for s in seasons + ["합"]:
            m = np.ones(len(idx), bool) if s == "합" else sidx == seasons.index(s)
            o[s] = cluster_stats(rc[m], g[m], sidx[m], rng, reml=(hb == 6))
        # 한 사람은 한 칸에 한 줄만 (사람 효과를 빼고 시간만)
        pgk = pers.astype(np.int64) * 1000000 + g
        first = np.unique(pgk, return_index=True)[1]
        keep = np.zeros(len(idx), bool)
        keep[first] = True
        o1 = {}
        for s in seasons + ["합"]:
            m = keep & (np.ones(len(idx), bool) if s == "합" else sidx == seasons.index(s))
            G = Groups(rc[m], g[m], sidx[m])
            a = G.anova()
            o1[s] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in a.items() if k in ("icc", "mbar", "deff", "N", "G")}
        tb[key] = {"all_rows": o, "one_row_per_person": o1}
    res["time_icc"] = tb

    # 변이도 (다른 사람 쌍)
    pairI, pairJ = season_pairs(D, np.ones(len(idx), bool), rc)
    dt = np.abs(D["t"][pairI] - D["t"][pairJ])
    diff = pers[pairI] != pers[pairJ]
    sq = 0.5 * (rc[pairI] - rc[pairJ]) ** 2
    sp = sidx[pairI]
    Mpers = Mp
    vg = {}
    for s in seasons + ["합"]:
        o = []
        for lo, hi in LAGS:
            k = diff & (dt >= lo) & (dt < hi) & ((sp == seasons.index(s)) if s != "합" else True)
            if k.sum() == 0:
                o.append({"lag_h": [lo, hi], "n_pairs": 0, "gamma": None})
                continue
            row = {"lag_h": [lo, hi], "n_pairs": int(k.sum()), "gamma": float(sq[k].mean())}
            if s == "합":
                wi, wj = pers[pairI[k]], pers[pairJ[k]]
                bs = np.array([(Mpers[b][wi] * Mpers[b][wj] * sq[k]).sum() / (Mpers[b][wi] * Mpers[b][wj]).sum() for b in range(NBOOT)])
                row["ci90"] = q90(bs)
            o.append(row)
        # 기준: 48시간 넘게 떨어진 다른 사람 쌍 (sill)
        k = diff & (dt >= 48) & ((sp == seasons.index(s)) if s != "합" else True)
        o.append({"lag_h": [48, None], "n_pairs": int(k.sum()), "gamma": float(sq[k].mean()) if k.sum() else None})
        vg[s] = o
    # 같은 사람 쌍 (비교용)
    ks = (~diff) & (dt < 48)
    vg["합_같은사람"] = [{"lag_h": [0, 48], "n_pairs": int(ks.sum()), "gamma": float(sq[ks].mean()) if ks.sum() else None}]
    res["variogram"] = vg

    # 사람 효과가 시간 효과를 넘는가: 같은 사람 쌍(48h 안)의 γ 대 「시간 차 구간 x % 가까움」 칸을 맞춘 다른 사람 쌍의 기대 γ
    dz = np.abs(D["z"][pairI] - D["z"][pairJ])
    within = dt < 48
    lagbin = np.digitize(dt, [1, 3, 6, 12, 24, 48])
    cell = (lagbin * 2 + (dz < 0.1)).astype(int)
    cw, sw_, sqw, dw_ = cell[within], (~diff)[within], sq[within], diff[within]
    pi_w, pj_w = pers[pairI][within], pers[pairJ][within]
    ncell = 14

    def matched(wt):
        ws, wd = wt * sw_, wt * dw_
        s_num = np.bincount(cw, weights=ws * sqw, minlength=ncell)
        s_den = np.bincount(cw, weights=ws, minlength=ncell)
        d_num = np.bincount(cw, weights=wd * sqw, minlength=ncell)
        d_den = np.bincount(cw, weights=wd, minlength=ncell)
        ok_ = (d_den > 0) & (s_den > 0)
        gs = s_num[ok_].sum() / s_den[ok_].sum()
        ge = ((s_den[ok_] / s_den[ok_].sum()) * (d_num[ok_] / d_den[ok_])).sum()
        return gs, ge

    gs, ge = matched(np.ones(len(cw)))
    dd_ = [matched(np.where(sw_, Mpers[b][pi_w], Mpers[b][pi_w] * Mpers[b][pj_w])) for b in range(NBOOT)]
    res["same_person_vs_time_matched"] = {
        "n_same_pairs": int(sw_.sum()), "gamma_same": float(gs), "gamma_expected_diff_person": float(ge), "diff": float(gs - ge),
        "ratio": float(gs / ge), "diff_ci90": q90([a - b for a, b in dd_]), "ratio_ci90": q90([a / b for a, b in dd_]),
        "note": "칸 = 시각 차 6구간 x log10 % 차 <0.1 여부. 기대 γ = 같은 사람 쌍의 칸 분포로 다른 사람 쌍 γ를 가중"}

    # 3. % 축: 같은 6시간 칸 안 다른 사람 쌍, log10 % 차 0.1 안
    g6 = np.floor(D["t"] / 6).astype(int)
    same6 = g6[pairI] == g6[pairJ]
    near = dz < 0.1
    sets = {
        "6h칸 안·% 가까움(<0.1)": diff & same6 & near,
        "6h칸 안·% 멂(>=0.1)": diff & same6 & ~near,
        "6h칸 밖(>=24h)·% 가까움(<0.1)": diff & (dt >= 24) & near,
        "6h칸 안·같은 사람": (~diff) & same6,
    }
    pcor = {}
    for name, k in sets.items():
        k = k & True
        I, J = pairI[k], pairJ[k]
        if len(I) < 5:
            pcor[name] = {"n_pairs": int(len(I)), "corr": None}
            continue
        x, y = rc[I], rc[J]
        cb = pair_corr_boot(pers[I], pers[J], x, y, pers, Mpers if "같은 사람" not in name else None)
        cb["persons"] = int(len(np.unique(np.concatenate([pers[I], pers[J]]))))
        # 시즌별 점추정
        cb["by_season"] = {s: (float(wpearson(x[sp[k] == seasons.index(s)], y[sp[k] == seasons.index(s)], np.ones(int((sp[k] == seasons.index(s)).sum()))))
                               if (sp[k] == seasons.index(s)).sum() >= 5 else None) for s in seasons}
        pcor[name] = cb
    res["pct_axis_corr"] = pcor
    return res


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ---------- md ----------

def f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def ci(c, d=3):
    return "-" if not c else f"{c[0]:.{d}f}~{c[1]:.{d}f}"


def build_md(res, hdr):
    A, B = res["set1"], res["set2"]
    ss = TUNE + ["합"]
    L = ["# WP5a 진단 2 — 관측 층 잔차의 사람·시간 상관 (D1)", "",
         "`tools/solo_raid/lab7/wp5a_diag2.py`가 만든다. 새 적합 없음: 1단계 사후 평균 θ 하나로 곡면 μ_i·σ_i(σ² = σ0² + (ρP)²)를 내고 "
         "r_i = (관측 % 구간 가운데 log10 − μ_i)/σ_i. **구간 중도절단이라 근사다** (가운데 값 사용; 표시 올림·버림·0.01 단위 구간 폭은 무시). "
         "조정 6시즌, 오염 사후 > 0.5 줄 뺌. 묶음 통계는 시즌 안에서만, 시즌 평균은 뺌. 부트스트랩 = 시즌 안 묶음 다시 뽑기 500번, 구간은 90%.", "",
         f"- 집합 1 (주): 한섭 라벨 · kind A(시각 정확) — 줄 {A['n']}. 집합 2: 서버 미상 포함 A·B·C 전체(tg=post 뺌, 시각 = 사후 평균 촬영 시각, 서버 성분 = P 최대) — 줄 {B['n']}.",
         f"- 오염으로 뺀 줄: 집합 1 {hdr['dropped_contam_1']}, 집합 2 {hdr['dropped_contam_2']}. 1단계 모수 평균(σ0, ρ): " +
         ", ".join(f"{s} {hdr['sig'][s][0]:.3f}/{hdr['sig'][s][1]:.3f}" for s in TUNE) + ".", ""]

    L += ["## 요약 (6시즌 합)", "",
          "| 집합 | 줄 | r sd | 사람 ICC (90%) | 6h ICC (90%) | 사람 deff | 6h deff | 유효 줄(사람) | 유효 줄(6h) | 변이도 0~1h / 6~12h / 48h+ |", "|---|---|---|---|---|---|---|---|---|---|"]
    for nm, R in (("1 한섭 A", A), ("2 A·B·C 전체", B)):
        p, t, v = R["person_icc"]["합"], R["time_icc"]["6h"]["all_rows"]["합"], R["variogram"]["합"]
        L.append(f"| {nm} | {R['n']} | {f(R['r_stats']['합']['sd'], 2)} | {f(p['icc'])} ({ci(p['icc_ci90'])}) | {f(t['icc'])} ({ci(t['icc_ci90'])}) | "
                 f"{f(p['deff'], 2)} | {f(t['deff'], 2)} | {f(p['n_eff'], 0)} | {f(t['n_eff'], 0)} | {f(v[0]['gamma'], 2)} / {f(v[3]['gamma'], 2)} / {f(v[6]['gamma'], 2)} |")
    L += ["", "γ가 가까운 시각에서 낮으면 같은 시간대 잔차가 함께 움직인다(시간 상관). 6h 칸 deff는 칸 안이 서로 같은 정도로 묶였다고 본 Kish 꼴이라, "
          "변이도가 몇 시간에 걸쳐 오르는 실제 모양에서는 거친 근사다.", ""]

    def sect(R, tag):
        o = ["", f"## {tag}", ""]
        o += ["### r 의 sd · u 구간별 (모형이 맞으면 sd ≈ 1, 평균 ≈ 0)", "",
              "| 시즌 | 줄 | 평균 | sd | 튼튼한 sd(MAD) | \\|r\\|>2 몫 | u 0~.25 평균/sd | .25~.5 | .5~.75 | .75~1 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in ss:
            x = R["r_stats"][s]
            bu = " | ".join(f"{f(b['mean'], 2)} / {f(b['sd'], 2)} (n={b['n']})" for b in x["by_u"])
            o.append(f"| {s} | {x['n']} | {f(x['mean'], 2)} | {f(x['sd'], 2)} | {f(x['robust_sd'], 2)} | {f(x['share_abs_gt2'], 3)} | {bu} |")
        o += ["", "### 1. 사람 묶음", "",
              "| 시즌 | 사람 | 줄 2+인 사람 | 크기 1 / 2 / 3~5 / 6+ | 최대 | σ_b² | σ_w² | ICC (ANOVA) | 90% 구간 | ICC (REML) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in ss:
            z, c = R["person_sizes"][s], R["person_icc"][s]
            sz = z["size"]
            rm = c.get("reml", {})
            o.append(f"| {s} | {z['persons']} | {z['persons_ge2']} | {sz['1']} / {sz['2']} / {sz['3-5']} / {sz['6+']} | {z['max']} | {f(c['sb2'], 3)} | {f(c['sw2'], 3)} | "
                     f"{f(c['icc'], 3)} | {ci(c['icc_ci90'])} | {f(rm.get('icc'), 3)} |")
        o += ["", "같은 사람 연속 두 줄(시간 순)의 잔차 상관:", "", "| 시즌 | 짝 | 상관 | 90% 구간 |", "|---|---|---|---|"]
        for s in ss:
            c = R["same_person_consecutive"][s]
            o.append(f"| {s} | {c['n_pairs']} | {f(c.get('corr'), 3)} | {ci(c.get('ci90'))} |")
        o += ["", "### 2. 시간 묶음 (촬영 시각 칸)", "",
              "| 시즌 | 3h ICC | 6h ICC | 6h 90% | 6h REML | 24h ICC | 3h 한 사람 한 줄 | 6h 한 사람 한 줄 | 24h 한 사람 한 줄 |", "|---|---|---|---|---|---|---|---|---|"]
        for s in ss:
            t3, t6, t24 = (R["time_icc"][k] for k in ("3h", "6h", "24h"))
            o.append(f"| {s} | {f(t3['all_rows'][s]['icc'])} | {f(t6['all_rows'][s]['icc'])} | {ci(t6['all_rows'][s]['icc_ci90'])} | "
                     f"{f(t6['all_rows'][s].get('reml', {}).get('icc'))} | {f(t24['all_rows'][s]['icc'])} | "
                     f"{f(t3['one_row_per_person'][s]['icc'])} | {f(t6['one_row_per_person'][s]['icc'])} | {f(t24['one_row_per_person'][s]['icc'])} |")
        o += ["", "변이도 γ(h) = ½ 평균 (r_i − r_j)², 다른 사람 쌍, 시각 차 구간 (6시즌 합; 90% = 사람 다시 뽑기). 기준 = 48h 이상 쌍.", "",
              "| 시각 차 | " + " | ".join(f"{a}~{b}h" for a, b in LAGS) + " | 48h+ |", "|---|" + "---|" * (len(LAGS) + 1)]
        v = R["variogram"]["합"]
        o.append("| 쌍 수 | " + " | ".join(str(x["n_pairs"]) for x in v) + " |")
        o.append("| γ | " + " | ".join(f(x["gamma"], 3) for x in v) + " |")
        o.append("| 90% 구간 | " + " | ".join(ci(x.get("ci90")) for x in v[:-1]) + " | - |")
        o.append("")
        o.append(f"같은 사람 쌍(48h 안) γ = {f(R['variogram']['합_같은사람'][0]['gamma'], 3)} (쌍 {R['variogram']['합_같은사람'][0]['n_pairs']}).")
        sm_ = R["same_person_vs_time_matched"]
        o.append(f"시간 차·% 가까움 칸을 맞춘 다른 사람 쌍의 기대 γ = {f(sm_['gamma_expected_diff_person'], 3)}, 같은 사람 γ = {f(sm_['gamma_same'], 3)} → "
                 f"차 {f(sm_['diff'], 3)} (90% {ci(sm_['diff_ci90'])}), 비 {f(sm_['ratio'], 2)} (90% {ci(sm_['ratio_ci90'], 2)}). 사람 효과가 시간 효과 위에 더 있으면 차가 음수.")
        o += ["", "시즌별 γ (쌍 수 적으면 흔들림):", "", "| 시즌 | " + " | ".join(f"{a}~{b}h" for a, b in LAGS) + " | 48h+ |", "|---|" + "---|" * (len(LAGS) + 1)]
        for s in TUNE:
            o.append(f"| {s} | " + " | ".join(f"{f(x['gamma'], 3)} ({x['n_pairs']})" for x in R["variogram"][s]) + " |")
        o += ["", "### 3. % 축 (다른 사람 쌍의 잔차 상관, 6시즌 합)", "", "| 쌍 | 쌍 수 | 상관 | 90% 구간 | 시즌별 |", "|---|---|---|---|---|"]
        for name, c in R["pct_axis_corr"].items():
            bsn = " / ".join(f(c["by_season"].get(s), 2) for s in TUNE) if "by_season" in c else "-"
            o.append(f"| {name} | {c['n_pairs']} | {f(c.get('corr'), 3)} | {ci(c.get('ci90'))} | {bsn} |")
        o += ["", "### 4. 설계 효과 deff = 1 + (m̄ − 1)·ICC (m̄ = 묶음 크기 가중 평균, ICC<0이면 0), 유효 줄 수 = n / deff", "",
              "| 시즌 | 줄 | 사람 m̄ | 사람 deff | 90% | 유효 줄(사람) | 6h m̄ | 6h deff | 90% | 유효 줄(6h) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in ss:
            p, t = R["person_icc"][s], R["time_icc"]["6h"]["all_rows"][s]
            o.append(f"| {s} | {R['r_stats'][s]['n']} | {f(p['mbar'], 2)} | {f(p['deff'], 2)} | {ci(p['deff_ci90'], 2)} | {f(p['n_eff'], 0)} | "
                     f"{f(t['mbar'], 2)} | {f(t['deff'], 2)} | {ci(t['deff_ci90'], 2)} | {f(t['n_eff'], 0)} |")
        return o

    L += sect(A, "집합 1 — 한섭 라벨 · kind A")
    L += sect(B, "집합 2 — 서버 미상 포함 A·B·C 전체")
    L.append("")
    return "\n".join(L)


# ---------- 실행 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out-dir", default=OUTD)
    a = ap.parse_args()
    OM.setup_host(1)
    np.seterr(invalid="ignore", divide="ignore")     # 다시 뽑기에서 짝이 한 쌍도 안 남는 경우의 NaN은 q90이 거른다
    err = OM.load_err()
    s1 = M3.load_stage1(os.path.join(R7, "stage1_M3_D1.json"))     # 기저를 1단계 파일에 맞춘다
    cols = {}
    sig = {}
    for i, s in enumerate(TUNE):
        o = season_rows(s, s1[s], err, a.obs_dir)
        o["season_name"] = np.array([s] * o["n"])
        cols[s] = o
        sig[s] = [o["s0"], o["rho"]]
        print(s, "줄", o["n"], "H", o["H"], flush=True)
    keys = ["pid", "kind", "tg", "server", "t", "u", "z", "p_contam", "p_kr", "mu_w", "sg_w", "mu_m", "sg_m", "season_name"]
    d = {k: np.concatenate([cols[s][k] for s in TUNE]) for k in keys}
    d["season"] = d["season_name"]
    d["person"] = np.array([f"{s}|{p}" for s, p in zip(d["season_name"], d["pid"])])
    # 집합 1: check43과 같은 mu_w·sg_w (서버 가중). 집합 2: 서버 성분 P 최대
    base1 = (d["kind"] == "A") & (d["tg"] == "A") & (d["server"] == "KR")
    base2 = d["tg"] != "post"
    ok = d["p_contam"] <= 0.5
    m1, m2 = base1 & ok, base2 & ok
    # 집합 1 줄은 서버 가중, 집합 2 는 P 최대 성분: 두 벌의 r
    r1 = (d["z"] - d["mu_w"]) / d["sg_w"]
    r2 = (d["z"] - d["mu_m"]) / d["sg_m"]
    names = {}
    out = {"header": {"about": "WP5a 진단 2 — 관측 층 잔차의 사람·시간 상관", "data_ver": "D1", "seasons": TUNE, "nboot": NBOOT, "seed": SEED,
                      "sigma0_rho": sig}}
    hdr = {"sig": sig, "dropped_contam_1": int((base1 & ~ok).sum()), "dropped_contam_2": int((base2 & ~ok).sum())}
    out["header"].update({"dropped_contam_set1": hdr["dropped_contam_1"], "dropped_contam_set2": hdr["dropped_contam_2"],
                          "rows_before_contam_set1": int(base1.sum()), "rows_before_contam_set2": int(base2.sum())})
    for tag, mask, rr, sd_ in (("set1", m1, r1, SEED), ("set2", m2, r2, SEED + 1)):
        dd = dict(d)
        dd["r"] = rr
        rc = rr.copy()
        for s in TUNE:   # 시즌 평균 뺀 r (집합 안 줄 기준)
            k = mask & (d["season_name"] == s)
            rc[d["season_name"] == s] -= rr[k].mean()
        dd["rc"] = rc
        out[tag] = analyze(dd, mask, tag, sd_)
        print(tag, "n", out[tag]["n"], "r sd", out[tag]["r_stats"]["합"]["sd"], flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "diag2_D1.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(out), fh, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out_dir, "diag2_D1.md"), "w", encoding="utf-8") as fh:
        fh.write(build_md(out, hdr))


if __name__ == "__main__":
    main()
