"""WP1 1a — 게시 지연 (사진 C의 한쪽 치우침). 2026-10-29. 정답(answer_key·truth·criteria) 안 봄: 관측끼리만 견준다.

  python3 -I tools/solo_raid/lab7/err_1a.py --obs-dir DIR/full --out OUT.json [--boot 400] [--seed 20261029]

관측 표(obs_table.py, 끝난 시즌 전체 `S{n}.jsonl.gz`)만 읽는다. 사진 A는 촬영 시각 h와 게시 시각 hp를 둘 다 안다 → 지연 d = hp − h (시간).
사진 C는 게시 시각만 안다 (h_hi = 게시와 같은 위쪽 한계). 새 모형은 C의 숨은 촬영 시각 = 게시 − 지연 으로 두고 지연 분포로 적분하려 한다.

구성
  1 표본·분위·꼬리 (사람 pid 단위 부트스트랩)       2 나눠 보기 (사이트·시간·본인·화면·종료 무렵·시즌·shot_src)
  3 모형용 꼴 (로그정규·감마·와이블·로그로지스틱·2성분 혼합, 최대가능도 + AIC)   4 독립 점검 (C 의 top7 구간)
  5 앞선 측정과 견줌   6 C 를 게시 시각에 두는 치우침의 지연 시간

표기: 지연은 시간(h) 단위, 로그는 자연로그(시간 단위). 재표본 단위는 (시즌, pid) — 같은 사람의 사진끼리는 독립이 아니다.
가능도: 시각이 초 단위라 지연 d는 초 정수 → 관측 구간 [d−1초, d+1초]의 확률을 쓴다 (구간 중도절단, 0 근처도 같은 식).
  d ≤ 0 (게시가 촬영보다 앞 — 시계 오류) 가운데 5분 이내는 [0, 1분] 으로 보고, 5분 넘는 것은 다른 사본의 시각을 빌린 오류라 뺀다.
공개 규칙: 글 번호·원문·닉네임·원글 주소·사진 이름은 쓰지 않는다. 개별 줄은 (시즌, key) 로만 가리킨다.
"""
import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np
from scipy import optimize, special, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_table as O  # noqa: E402
import common as C  # noqa: E402

QL = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
TAILS = [1.0, 6.0, 24.0]          # 시간
SEC = 1.0 / 3600.0
NEG_OK = 5.0 / 60.0               # 시간: 이 안의 음수 d 는 시계 오차로 본다
R4 = 4


def r4(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return [r4(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, dict):
        return {str(k): r4(v) for k, v in x.items()}
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), R4)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ---------------------------------------------------------------- 자료

def load(obs_dir, seasons=range(19, 43)):
    out = []
    for s in seasons:
        p = os.path.join(obs_dir, f"S{s}.jsonl.gz")
        if not os.path.exists(p):
            continue
        head, rows = O.read_cell(p)
        for r in rows:
            r["S"] = s
            r["H"] = float(head["H"])
        out += rows
    return out


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def wq(xs, w, q):
    """가중 분위 (xs 오름차순, 역누적분포). w 는 정수 개수여도 실수여도."""
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.nan
    i = np.searchsorted(cw, q * cw[-1], side="left")
    return xs[min(i, len(xs) - 1)]


class Sample:
    """A 사진 표본 (지연 d 오름차순으로 정렬해 둔다)."""

    def __init__(self, rows):
        rows = sorted(rows, key=lambda r: r["hp"] - r["h"])
        self.rows = rows
        self.d = np.array([r["hp"] - r["h"] for r in rows])
        self.S = np.array([r["S"] for r in rows])
        keys = {}
        self.cl = np.array([keys.setdefault((r["S"], r["pid"]), len(keys)) for r in rows])
        self.K = len(keys)
        self.n = len(rows)

    def col(self, f):
        return np.array([f(r) for r in self.rows], dtype=object)


def interval_of(d):
    """지연 d (시간) → 가능도 구간 (lo, hi, 쓴다). 음수는 [0, 1분], 5분 넘는 음수는 뺀다."""
    d = np.asarray(d, float)
    keep = d >= -NEG_OK
    lo = np.where(d > 0, np.maximum(d - SEC, 0.0), 0.0)
    hi = np.where(d > 0, d + SEC, 1.0 / 60.0)
    return lo, hi, keep


# ---------------------------------------------------------------- 요약 (분위·꼬리·부트스트랩)

def quants(d):
    return {f"q{int(q*100):02d}": float(np.quantile(d, q)) for q in QL}


def tails(d):
    n = len(d)
    out = {}
    for t in TAILS:
        k = int((d > t).sum())
        out[f"gt{int(t)}h"] = {"n_gt": k, "n": n, "p": k / n if n else None, "wilson95": wilson(k, n)}
    return out


def boot_weights(K, B, rng):
    """B x K 개 (사람) 재표본 횟수."""
    idx = rng.integers(0, K, size=(B, K))
    return np.stack([np.bincount(i, minlength=K) for i in idx])


def stat_vec(d, w, mask=None):
    """d 오름차순, w 가중. [중앙, 75%, 90%, 95%, 99%, P>1, P>6, P>24, 평균(24h 절단), 총가중]."""
    if mask is not None:
        d, w = d[mask], w[mask]
    tw = w.sum()
    if tw <= 0:
        return np.full(10, np.nan)
    out = [wq(d, w, q) for q in (0.5, 0.75, 0.9, 0.95, 0.99)]
    out += [w[d > t].sum() / tw for t in TAILS]
    out += [(np.minimum(d, 24.0) * w).sum() / tw, tw]
    return np.array(out)


STAT_NAMES = ["med", "q75", "q90", "q95", "q99", "gt1h", "gt6h", "gt24h", "mean_cap24h", "n_w"]


def ci(a, lo=2.5, hi=97.5):
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return [None, None]
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


def summarize(smp, mask, W, label=None):
    """부분 표본 하나: n, 분위, 꼬리, 부트스트랩 구간 (전체 재표본 W 를 그대로 써서 부분표본 사람이 같은 비중으로 뽑힌다)."""
    d = smp.d
    m = mask
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    dd = d[m]
    cls = len(set(smp.cl[m]))
    res = {"n": n, "n_persons": cls, "quantiles": r4(quants(dd)), "tails": r4(tails(dd))}
    B = W.shape[0]
    bs = np.array([stat_vec(d, W[b][smp.cl], m) for b in range(B)])
    res["boot95"] = {nm: r4(ci(bs[:, j])) for j, nm in enumerate(STAT_NAMES[:9])}
    res["_bs"] = bs
    return res


def kw_adj(groups_d, groups_cl, d_all, cl_all):
    """Kruskal–Wallis (소박) + 설계효과 보정. 설계효과 = 1 + (평균 m − 1) ρ, ρ 는 순위의 군집내 상관."""
    gs = [g for g in groups_d if len(g) >= 5]
    if len(gs) < 2:
        return {"H": None}
    H, p = stats.kruskal(*gs)
    rk = stats.rankdata(d_all)
    df = {}
    for r_, c_ in zip(rk, cl_all):
        df.setdefault(c_, []).append(r_)
    m = np.array([len(v) for v in df.values()])
    n = m.sum()
    gm = rk.mean()
    ssb = sum(len(v) * (np.mean(v) - gm) ** 2 for v in df.values())
    ssw = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in df.values())
    k = len(m)
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    n0 = (n - (m ** 2).sum() / n) / (k - 1)
    rho = max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))
    deff = 1 + ((m ** 2).sum() / n - 1) * rho
    Hadj = H / deff
    padj = float(stats.chi2.sf(Hadj, len(gs) - 1))
    return {"H": float(H), "df": len(gs) - 1, "p_naive": float(p), "rho_rank": float(rho), "deff": float(deff),
            "H_adj": float(Hadj), "p_adj": padj}


# ---------------------------------------------------------------- 분포족 (구간 가능도)

def _ln_cdf(x, mu, s):
    with np.errstate(divide="ignore"):
        z = (np.log(np.maximum(x, 1e-300)) - mu) / s
    return np.where(x > 0, special.ndtr(z), 0.0), np.where(x > 0, special.ndtr(-z), 1.0)


def _gam_cdf(x, k, th):
    z = np.maximum(x, 0) / th
    return special.gammainc(k, z), special.gammaincc(k, z)


def _wei_cdf(x, k, lam):
    z = (np.maximum(x, 0) / lam) ** k
    return -np.expm1(-z), np.exp(-z)


def _fisk_cdf(x, c, a):
    z = (np.maximum(x, 1e-300) / a) ** c
    return np.where(x > 0, z / (1 + z), 0.0), np.where(x > 0, 1 / (1 + z), 1.0)


def _mix2_cdf(x, w, mu1, s1, mu2, s2):
    F1, S1 = _ln_cdf(x, mu1, s1)
    F2, S2 = _ln_cdf(x, mu2, s2)
    return w * F1 + (1 - w) * F2, w * S1 + (1 - w) * S2


def _mix3_cdf(x, w1, w2, mu1, s1, mu2, s2, mu3, s3):
    F1, S1 = _ln_cdf(x, mu1, s1)
    F2, S2 = _ln_cdf(x, mu2, s2)
    F3, S3 = _ln_cdf(x, mu3, s3)
    w3 = 1 - w1 - w2
    return w1 * F1 + w2 * F2 + w3 * F3, w1 * S1 + w2 * S2 + w3 * S3


def _sig(z):
    return 1 / (1 + np.exp(-z))


# 각 족: (이름, 매개 개수, 제약 없는 theta → 매개 변환, cdf, 이름들)
def _tr_ln(t):
    return (t[0], math.exp(t[1]))


def _tr_pos2(t):
    return (math.exp(t[0]), math.exp(t[1]))


def _tr_mix2(t):
    return (_sig(t[0]), t[1], math.exp(t[2]), t[3], math.exp(t[4]))


def _tr_mix3(t):
    a, b = math.exp(t[0]), math.exp(t[1])
    z = 1 + a + b
    return (1 / z, a / z, t[2], math.exp(t[3]), t[4], math.exp(t[5]), t[6], math.exp(t[7]))


FAM = {
    "lognormal": (2, _tr_ln, _ln_cdf, ["mu", "sigma"]),
    "gamma": (2, _tr_pos2, _gam_cdf, ["k", "theta"]),
    "weibull": (2, _tr_pos2, _wei_cdf, ["k", "lambda"]),
    "loglogistic": (2, _tr_pos2, _fisk_cdf, ["c", "a"]),
    "lognormal_mix2": (5, _tr_mix2, _mix2_cdf, ["w_fast", "mu1", "sigma1", "mu2", "sigma2"]),
    "lognormal_mix3": (8, _tr_mix3, _mix3_cdf, ["w1", "w2", "mu1", "sigma1", "mu2", "sigma2", "mu3", "sigma3"]),
}


def cdf_sf(fam, par, x):
    return FAM[fam][2](np.asarray(x, float), *par)


def interval_p(fam, par, lo, hi, trunc=None):
    """구간 확률 (위쪽 꼬리는 생존함수 차로). trunc = 상한(시간): 조건부 P(lo<d<hi | d<=trunc)."""
    Fl, Sl = cdf_sf(fam, par, lo)
    Fh, Sh = cdf_sf(fam, par, hi)
    p = np.where(Fl > 0.5, Sl - Sh, Fh - Fl)
    if trunc is not None:
        Ft, _ = cdf_sf(fam, par, trunc)
        p = p / np.maximum(Ft, 1e-300)
    return np.maximum(p, 1e-300)


def nll(theta, fam, lo, hi, w, trunc=None):
    try:
        par = FAM[fam][1](theta)
    except OverflowError:
        return 1e18
    if not all(np.isfinite(par)):
        return 1e18
    if fam.startswith("lognormal_mix") and (min(par[2::2] if fam == "lognormal_mix2" else par[3::2]) < 0.02):
        return 1e18
    p = interval_p(fam, par, lo, hi, trunc)
    v = -(w * np.log(p)).sum()
    return float(v) if np.isfinite(v) else 1e18


def starts(fam, lo, hi, w):
    d = 0.5 * (lo + hi)
    ld = np.log(np.maximum(d, 1e-4))
    mu, sd = np.average(ld, weights=w), np.sqrt(np.average((ld - np.average(ld, weights=w)) ** 2, weights=w))
    m = np.average(d, weights=w)
    v = np.average((d - m) ** 2, weights=w)
    if fam == "lognormal":
        return [[mu, math.log(sd)]]
    if fam == "gamma":
        return [[math.log(m * m / v), math.log(v / m)], [math.log(0.5), math.log(m)]]
    if fam == "weibull":
        return [[math.log(0.6), math.log(m)], [math.log(0.4), math.log(m)]]
    if fam == "loglogistic":
        return [[math.log(1.0), math.log(np.exp(mu))], [math.log(0.7), math.log(np.exp(mu))]]
    if fam == "lognormal_mix2":
        out = []
        for w0, m1, s1, m2, s2 in [(0.85, -3.5, 1.0, 0.0, 1.8), (0.7, -3.0, 1.0, 0.5, 1.5), (0.9, -4.0, 0.8, 1.0, 1.5),
                                   (0.6, -3.0, 1.3, 1.0, 1.2), (0.95, -3.4, 1.1, 2.0, 1.5)]:
            out.append([math.log(w0 / (1 - w0)), m1, math.log(s1), m2, math.log(s2)])
        return out
    if fam == "lognormal_mix3":
        return [[math.log(0.2 / 0.7), math.log(0.1 / 0.7), -4.0, math.log(0.8), -2.8, math.log(0.9), 0.8, math.log(1.5)],
                [math.log(0.3 / 0.6), math.log(0.1 / 0.6), -4.5, math.log(0.7), -2.5, math.log(1.0), 1.0, math.log(1.3)]]
    raise KeyError(fam)


def fit(fam, lo, hi, w=None, trunc=None, theta0=None, polish=True, extra_starts=None):
    """최대가능도 (Nelder–Mead 여러 시작 + L-BFGS 다듬기). theta0 가 있으면 그 한 점에서만 시작."""
    w = np.ones(len(lo)) if w is None else w
    best = None
    ss = [theta0] if theta0 is not None else starts(fam, lo, hi, w)
    if extra_starts:
        ss = ss + extra_starts
    for t0 in ss:
        if theta0 is not None:
            r = optimize.minimize(nll, t0, args=(fam, lo, hi, w, trunc), method="L-BFGS-B")
            if not np.isfinite(r.fun) or r.fun > 1e17:
                r = optimize.minimize(nll, t0, args=(fam, lo, hi, w, trunc), method="Nelder-Mead",
                                      options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 3000})
        else:
            r = optimize.minimize(nll, t0, args=(fam, lo, hi, w, trunc), method="Nelder-Mead",
                                  options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 4000, "maxfev": 6000})
            if polish:
                r2 = optimize.minimize(nll, r.x, args=(fam, lo, hi, w, trunc), method="L-BFGS-B")
                if r2.fun < r.fun:
                    r = r2
        if best is None or r.fun < best.fun:
            best = r
    par = FAM[fam][1](best.x)
    if fam == "lognormal_mix2" and par[1] > par[3]:       # 빠른 성분이 앞에 오게
        w_, m1, s1, m2, s2 = par
        par = (1 - w_, m2, s2, m1, s1)
        th = np.array([math.log((1 - w_) / w_), m2, math.log(s2), m1, math.log(s1)])
        return {"fam": fam, "theta": th, "par": par, "nll": float(best.fun)}
    return {"fam": fam, "theta": best.x, "par": tuple(float(v) for v in par), "nll": float(best.fun)}


def par_dict(fam, par):
    return {k: float(v) for k, v in zip(FAM[fam][3], par)}


def fit_quantiles(fam, par, qs):
    """모형 분위 (역함수는 이분법)."""
    out = []
    for q in qs:
        a, b = 1e-9, 1e4
        for _ in range(200):
            m = math.sqrt(a * b)
            F = float(cdf_sf(fam, par, np.array([m]))[0][0])
            if F < q:
                a = m
            else:
                b = m
        out.append(math.sqrt(a * b))
    return np.array(out)


def fit_tails(fam, par):
    return [float(cdf_sf(fam, par, np.array([t]))[1][0]) for t in TAILS]


def mean_cap(fam, par, cap):
    """E[min(d, cap)] = ∫0^cap S(x)dx (사다리꼴, 로그 격자)."""
    x = np.concatenate([[0.0], np.geomspace(1e-5, cap, 4000)])
    S = cdf_sf(fam, par, x)[1]
    return float(np.trapezoid(S, x))


# ---------------------------------------------------------------- 본체

def build_sample(rows):
    A = [r for r in rows if r["kind"] == "A" and r["use_indep"] and r["hp"] is not None and r["h"] is not None
         and 0.0 <= r["h"] <= r["H"]]
    return Sample(A)


def section_groups(smp, W, B_note):
    """2. 나눠 보기."""
    rows = smp.rows
    H = np.array([r["H"] for r in rows])
    h = np.array([r["h"] for r in rows])
    hp = np.array([r["hp"] for r in rows])
    d = smp.d
    levels = {}
    levels["site"] = {k: np.array([r["site"] == k for r in rows]) for k in sorted({r["site"] for r in rows})}
    levels["day_h24"] = {f"h{24*i}-{24*(i+1)}": (np.minimum((h // 24).astype(int), 4) == i) for i in range(5)}
    levels["half_day"] = {f"h{12*i}-{12*(i+1)}": (np.minimum((h // 12).astype(int), 9) == i) for i in range(10)}
    levels["self"] = {"self": np.array([bool(r["self"]) for r in rows]), "other": np.array([not r["self"] for r in rows])}
    levels["screen"] = {k: np.array([r["screen"] == k for r in rows]) for k in sorted({r["screen"] for r in rows})}
    last12 = h > H - 12
    levels["end"] = {"h<=H-12": ~last12, "h>H-12": last12,
                     "h>H-12_posted<=H": last12 & (hp <= H), "h>H-12_posted>H": last12 & (hp > H),
                     "h<=H-12_posted>H": (~last12) & (hp > H)}
    levels["shot_src"] = {str(k): np.array([r["shot_src"] == k for r in rows]) for k in sorted({str(r["shot_src"]) for r in rows})}
    levels["shot_src"] = {k: m for k, m in levels["shot_src"].items()}
    levels["season"] = {f"S{s}": (smp.S == s) for s in sorted(set(smp.S))}
    # shot_src None 은 문자열 'None' 이라 위 비교가 안 맞는다 → 다시
    levels["shot_src"] = {}
    for k in sorted({str(r["shot_src"]) for r in rows}):
        levels["shot_src"][k] = np.array([str(r["shot_src"]) == k for r in rows])
    # 이름 근거 표시 (late>24h 등 표시만)
    tb = [str(r["t_basis"]) for r in rows]
    levels["via_copy"] = {"own_name": np.array(["via" not in t for t in tb]), "via_other_copy": np.array(["via" in t for t in tb])}
    out = {}
    pairs = {}
    for fac, lv in levels.items():
        out[fac] = {}
        for name, m in lv.items():
            if m.sum() == 0:
                continue
            res = summarize(smp, m, W)
            bs = res.pop("_bs")
            out[fac][name] = res
            # 이 수준 vs 나머지: 로그 중앙값 비, P>1h 차
            if fac in ("end", "season") or m.sum() < 15 or (~m).sum() < 15:
                rest_ok = fac == "season" and m.sum() >= 15
            else:
                rest_ok = True
            if rest_ok:
                bsr = np.array([stat_vec(d, W[b][smp.cl], ~m) for b in range(W.shape[0])])
                pt_in = stat_vec(d, np.ones(len(d)), m)
                pt_out = stat_vec(d, np.ones(len(d)), ~m)
                lr = np.log(bs[:, 0] / bsr[:, 0])
                dg = bs[:, 5] - bsr[:, 5]
                res["vs_rest"] = r4({"log_median_ratio": float(np.log(pt_in[0] / pt_out[0])),
                                     "log_median_ratio_boot95": ci(lr),
                                     "gt1h_diff": float(pt_in[5] - pt_out[5]), "gt1h_diff_boot95": ci(dg),
                                     "gt6h_diff": float(pt_in[6] - pt_out[6]), "gt6h_diff_boot95": ci(bs[:, 6] - bsr[:, 6])})
                res["_bs"] = bs
            else:
                res["_bs"] = bs
        # 검정
        names = [n_ for n_ in lv if out[fac].get(n_, {}).get("n", 0) >= 5]
        # 겹치는 수준(end 의 부분집합)은 KW 에 쓰지 않는다
        if fac == "end":
            names = ["h<=H-12", "h>H-12"]
        if fac == "via_copy":
            names = list(lv)
        gd = [d[lv[n_]] for n_ in names]
        out[fac]["_test"] = kw_adj(gd, None, d[np.any([lv[n_] for n_ in names], axis=0)],
                                   smp.cl[np.any([lv[n_] for n_ in names], axis=0)]) if len(names) > 1 else {}
        out[fac]["_test"]["levels"] = names
    # 시즌 이질성: 시즌별 값의 퍼짐 vs 부트스트랩 퍼짐
    seas = [k for k in out["season"] if k.startswith("S") and out["season"][k]["n"] >= 30]
    het = {}
    for j, nm in [(0, "log_median"), (5, "logit_gt1h"), (6, "logit_gt6h"), (3, "log_q95")]:
        g, se, w_ = [], [], []
        for k in seas:
            bs = out["season"][k]["_bs"]
            pt = stat_vec(d, np.ones(len(d)), levels["season"][k])
            if nm.startswith("log_"):
                v = np.log(bs[:, j]); p0 = math.log(pt[j])
            else:
                v = special.logit(np.clip(bs[:, j], 1e-3, 1 - 1e-3)); p0 = float(special.logit(np.clip(pt[j], 1e-3, 1 - 1e-3)))
            g.append(p0); se.append(float(np.std(v))); w_.append(1 / max(np.var(v), 1e-12))
        g, se, w_ = np.array(g), np.array(se), np.array(w_)
        gbar = float((g * w_).sum() / w_.sum())
        Q = float((w_ * (g - gbar) ** 2).sum())
        dfq = len(g) - 1
        tau2 = max(0.0, (Q - dfq) / (w_.sum() - (w_ ** 2).sum() / w_.sum()))
        het[nm] = {"seasons": len(g), "pooled": gbar, "sd_between_obs": float(np.std(g, ddof=1)), "mean_se": float(se.mean()),
                   "Q": Q, "df": dfq, "p_Q": float(stats.chi2.sf(Q, dfq)), "tau_est": math.sqrt(tau2),
                   "range": [float(g.min()), float(g.max())]}
    return out, levels, r4(het)


def strip_bs(o):
    if isinstance(o, dict):
        return {k: strip_bs(v) for k, v in o.items() if k != "_bs"}
    return o


def adjusted_effects(smp):
    """ln d 를 사이트·본인·화면·일차·종료 무렵·shot_src 로 한꺼번에 회귀 (군집 강건 표준오차, 군집 = 시즌·사람). 서로 얽힌 차이를 가르는 거친 점검."""
    rows = smp.rows
    keep = smp.d > 0
    y = np.log(smp.d[keep])
    R = [r for r, k in zip(rows, keep) if k]
    cl = smp.cl[keep]
    cols = {"intercept": np.ones(len(R)),
            "site_naver": np.array([r["site"] == "naver" for r in R], float),
            "not_self": np.array([not r["self"] for r in R], float),
            "screen_my_rank": np.array([r["screen"] == "my_rank" for r in R], float),
            "screen_result": np.array([r["screen"] == "result" for r in R], float),
            "day2(24-48h)": np.array([24 <= r["h"] < 48 for r in R], float),
            "day3(48-72h)": np.array([48 <= r["h"] < 72 for r in R], float),
            "day4(72-96h)": np.array([72 <= r["h"] < 96 for r in R], float),
            "day5(96h-)": np.array([r["h"] >= 96 for r in R], float),
            "last12h": np.array([r["h"] > r["H"] - 12 for r in R], float),
            "shot_epoch_ms": np.array([r["shot_src"] == "epoch_ms" for r in R], float)}
    names = list(cols)
    X = np.column_stack([cols[k] for k in names])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    G = int(cl.max() + 1)
    for g in range(G):
        m = cl == g
        if m.any():
            u = X[m].T @ e[m]
            meat += np.outer(u, u)
    n, k = X.shape
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    out = {"n": int(n), "n_clusters": G, "response": "ln(d 시간), d>0 만", "ref": "site dc · 본인 · rank_list · 1일차(0-24h) · h≤H−12 · shot_src filename",
           "coef": {}}
    for nm, b, s_ in zip(names, beta, se):
        out["coef"][nm] = {"beta": float(b), "se": float(s_), "ci95": [float(b - 1.96 * s_), float(b + 1.96 * s_)], "ratio_exp_beta": float(math.exp(b)),
                           "p": float(2 * stats.norm.sf(abs(b / s_))) if s_ > 0 else None}
    out["resid_sd"] = float(e.std(ddof=k))
    return out


def section_fits(smp, W, seed, B_fit):
    """3. 모형용 꼴."""
    lo, hi, keep = interval_of(smp.d)
    lo, hi = lo[keep], hi[keep]
    cl = smp.cl[keep]
    n = len(lo)
    res = {"n": int(n), "n_dropped_gt5min_early": int((~keep).sum()),
           "n_left_censored_0_1min": int((smp.d <= 0).sum() - (~keep).sum()), "fits": [], "_obj": {}}
    dd = smp.d[keep]
    emp_q = np.quantile(dd, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995])
    emp_t = [float((dd > t).mean()) for t in TAILS]
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995]
    res["empirical_quantiles"] = r4(dict(zip(map(str, qs), emp_q)))
    t0 = time.time()
    for fam in ["lognormal", "gamma", "weibull", "loglogistic", "lognormal_mix2", "lognormal_mix3"]:
        f = fit(fam, lo, hi)
        k = FAM[fam][0]
        aic = 2 * f["nll"] + 2 * k
        bic = 2 * f["nll"] + k * math.log(n)
        mq = fit_quantiles(fam, f["par"], qs)
        ft = fit_tails(fam, f["par"])
        res["fits"].append({"family": fam, "k": k, "params": par_dict(fam, f["par"]), "logLik": -f["nll"], "AIC": aic, "BIC": bic,
                            "quantiles_fit": dict(zip(map(str, qs), mq)),
                            "quantile_log_ratio_fit_over_emp": dict(zip(map(str, qs), np.log(mq / emp_q))),
                            "max_abs_log_ratio_q05_q99": float(np.max(np.abs(np.log(mq / emp_q)[1:9]))),
                            "tails_fit": dict(zip(["gt1h", "gt6h", "gt24h"], ft)), "tails_emp": dict(zip(["gt1h", "gt6h", "gt24h"], emp_t)),
                            "mean_cap24h": mean_cap(fam, f["par"], 24.0), "mean_cap120h": mean_cap(fam, f["par"], 120.0)})
        res["_obj"][fam] = f
    a0 = min(f["AIC"] for f in res["fits"])
    for f in res["fits"]:
        f["dAIC"] = f["AIC"] - a0
    res["fit_seconds"] = time.time() - t0
    # 매개 부트스트랩 (사람 단위, 추천 꼴 = mix2)
    return res, (lo, hi, cl, dd)


def boot_params(fam, f, lo, hi, cl, K, B, seed):
    rng = np.random.default_rng(seed)
    # 군집 인덱스를 0..K-1 로 다시 매기기
    uc, inv = np.unique(cl, return_inverse=True)
    Kc = len(uc)
    out = []
    t0 = time.time()
    for b in range(B):
        cnt = np.bincount(rng.integers(0, Kc, Kc), minlength=Kc)
        w = cnt[inv].astype(float)
        m = w > 0
        g = fit(fam, lo[m], hi[m], w[m], theta0=f["theta"])
        out.append(list(g["par"]))
    return np.array(out), time.time() - t0


def expanding(smp, fam_obj, seasons, pooled_theta, pooled_theta3=None):
    """늘어나는 창: 시즌 s 보다 앞 시즌들만으로 잰 값 (s=21..42)."""
    res = {}
    lo_all, hi_all, keep = interval_of(smp.d)
    for s in seasons:
        m = (smp.S < s) & keep
        if m.sum() < 100:
            continue
        d = smp.d[m]
        row = {"n": int(m.sum()), "n_seasons": int(len(set(smp.S[m]))),
               "med": float(np.median(d)), "q75": float(np.quantile(d, .75)), "q90": float(np.quantile(d, .9)),
               "q95": float(np.quantile(d, .95)),
               "gt1h": float((d > 1).mean()), "gt6h": float((d > 6).mean()), "gt24h": float((d > 24).mean())}
        ln = fit("lognormal", lo_all[m], hi_all[m])
        row["ln"] = par_dict("lognormal", ln["par"])
        mx = fit("lognormal_mix2", lo_all[m], hi_all[m], theta0=pooled_theta)
        # 일부 창에서 국소해에 갇힐 수 있어 다른 시작도 해 본다
        mx2 = fit("lognormal_mix2", lo_all[m], hi_all[m])
        if mx2["nll"] < mx["nll"]:
            mx = mx2
        row["mix2"] = par_dict("lognormal_mix2", mx["par"])
        q = fit_quantiles("lognormal_mix2", mx["par"], [0.5, 0.9])
        row["mix2_med"], row["mix2_q90"] = float(q[0]), float(q[1])
        row["mix2_gt1h"], row["mix2_gt6h"], row["mix2_gt24h"] = fit_tails("lognormal_mix2", mx["par"])
        row["mix2_mean_cap24h"] = mean_cap("lognormal_mix2", mx["par"], 24.0)
        if pooled_theta3 is not None and s >= 25:
            m3 = fit("lognormal_mix3", lo_all[m], hi_all[m], theta0=pooled_theta3)
            m3b = fit("lognormal_mix3", lo_all[m], hi_all[m])        # 다른 시작에서도 맞춰 더 높은 가능도를 쓴다
            if m3b["nll"] < m3["nll"]:
                m3 = m3b
            row["mix3"] = par_dict("lognormal_mix3", m3["par"])
            q3 = fit_quantiles("lognormal_mix3", m3["par"], [0.5, 0.9])
            row["mix3_med"], row["mix3_q90"] = float(q3[0]), float(q3[1])
            row["mix3_gt1h"], row["mix3_gt6h"], row["mix3_gt24h"] = fit_tails("lognormal_mix3", m3["par"])
            row["mix3_mean_cap24h"] = mean_cap("lognormal_mix3", m3["par"], 24.0)
            row["mix3_logLik"] = -m3["nll"]
        # 그 시즌 자신의 값
        own = smp.S == s
        if own.sum() >= 20:
            do = smp.d[own]
            row["own"] = {"n": int(own.sum()), "med": float(np.median(do)), "gt1h": float((do > 1).mean()),
                          "gt6h": float((do > 6).mean()), "q90": float(np.quantile(do, .9))}
        res[f"S{s}"] = row
    return res


def season_tail_weight(smp, pooled_par, seasons):
    """시즌별: 빠른·느린 성분의 모양을 합친 값에 묶고 꼬리 비중 w 만 시즌마다 맞춘다 (1 매개)."""
    lo, hi, keep = interval_of(smp.d)
    out = {}
    w0, m1, s1, m2, s2 = pooled_par
    for s in seasons:
        m = (smp.S == s) & keep
        if m.sum() < 30:
            continue
        def f(wl):
            w = 1 / (1 + math.exp(-wl))
            return nll([wl, m1, math.log(s1), m2, math.log(s2)], "lognormal_mix2", lo[m], hi[m], np.ones(m.sum()))
        r = optimize.minimize_scalar(f, bounds=(-6, 8), method="bounded")
        wf = 1 / (1 + math.exp(-r.x))
        # 프로파일 95%
        L0 = r.fun
        grid = np.linspace(-6, 8, 281)
        vals = np.array([f(g) for g in grid])
        okg = grid[vals - L0 < 1.92]
        out[f"S{s}"] = {"n": int(m.sum()), "w_fast": wf, "w_tail": 1 - wf,
                        "w_tail_profile95": [1 - 1 / (1 + math.exp(-okg.max())), 1 - 1 / (1 + math.exp(-okg.min()))] if len(okg) else None}
    return out


# ---------------------------------------------------------------- 4. top7 독립 점검

def bracket_fast(rank_k, rank_v, rh, rs):
    """post_time.top7_bracket 과 같은 결과 (lo, hi, 어긋남). rh: ref 촬영 시각 (R,), rs: ref 1~7위 점수 (R,7) 없는 칸 nan.
    rank_k: 이 사진의 순위 번호 0..6, rank_v: 점수."""
    cmp_a = np.sum((rs[:, rank_k] > rank_v[None, :]), axis=1)    # ref 점수 > v → 촬영은 ref 전 (ref 가 lo 이하면 어긋남)
    cmp_b = np.sum((rs[:, rank_k] < rank_v[None, :]), axis=1)    # ref 점수 < v → 촬영은 ref 후 (ref 가 hi 이상이면 어긋남)
    # nan 비교는 False 이므로 nansum 안에서 그대로 쓴다
    u = np.unique(rh)
    nu = len(u)
    # 칸 j = (cand[j], cand[j+1]), cand = [-inf] + u + [inf]
    idx = np.searchsorted(u, rh)                  # 각 ref 의 u 위치
    a_by = np.bincount(idx, weights=cmp_a, minlength=nu)
    b_by = np.bincount(idx, weights=cmp_b, minlength=nu)
    prefA = np.concatenate([[0.0], np.cumsum(a_by)])           # j 번째 칸: u[0..j-1] 에 있는 ref
    sufB = np.concatenate([np.cumsum(b_by[::-1])[::-1], [0.0]])  # j 번째 칸: u[j..] 에 있는 ref
    viol = prefA + sufB
    best = viol.min()
    good = np.where(viol == best)[0]
    cand = np.concatenate([[-1e9], u, [1e9]])
    lo = cand[good].min()
    hi = cand[good + 1].max()
    return float(lo), float(hi), int(best)


def top7_snaps(state, season):
    """시즌 하나의 top7 사진: A 참조 (서버별) 와 A 사진들의 (img, server, h, hp, 순위·점수)."""
    core = C.PR.jl(os.path.join(state, "core.jsonl"))
    obs = C.PR.jl(os.path.join(state, "obs.jsonl"))
    c0, H = C.info(f"S{season}")
    im = collections.defaultdict(list)
    for x in core:
        if x["src"] == "photo":
            im[obs[x["oi"]]["img"]].append(x)
    snaps = []
    for img, v in im.items():
        o0 = obs[v[0]["oi"]]
        if o0["season"] != "in" or any(x["exclude"] for x in v):
            continue
        ranks = {}
        for x in v:
            o = obs[x["oi"]]
            if o["rank"] and o["score"] and 1 <= o["rank"] <= 7 and not o["self"]:
                ranks[o["rank"]] = o["score"]
        if len(ranks) < 2:
            continue
        x0 = v[0]
        hp = o0.get("post_t")
        snaps.append({"img": img, "server": x0["server"], "grade": x0["t_grade"], "h": x0["h"],
                      "hp": C.PR.hours_of(hp, c0) if hp else None, "ranks": ranks, "oi": x0["oi"], "key": x0["key"],
                      "keys": [x["key"] for x in v], "person": x0["person"],
                      "self": bool(x0["self"]), "screen": x0["screen"], "site": x0["site"], "H": H})
    return snaps


def top7_modes(rows, seasons):
    """C 줄과 A 사진(leave-one-out)의 top7 구간을 state 의 점수로 다시 구한다 — 두 방식:
      all  = 같은 서버의 A 사진 전부를 참조 (C 는 그대로, A 는 자기만 뺌) = 운영 방식(h_lo_top7)과 같다.
      excl = 같은 사람(person)의 A 사진을 참조에서 뺌 (A 도 C 도). 같은 사람이 같은 때 올린 사진끼리 몰려 있는 효과(군집)를 없앤 비교.
    반환 {mode: {"C": [...], "A": [...]}}, 그리고 all 방식 C 가 관측 표의 h_lo_top7·h_hi_top7 와 같은 정도."""
    byS = collections.defaultdict(dict)
    for r in rows:
        if r["kind"] == "C" and r["use_indep"] and r["hp"] is not None and r["h_hi"] is not None and r["h_lo"] is not None:
            byS[r["S"]][r["key"]] = r
    out = {m: {"C": [], "A": []} for m in ("all", "excl")}
    agree = collections.Counter()
    for s in seasons:
        st = C.state_of(f"S{s}")
        if not os.path.exists(os.path.join(st, "core.jsonl")):
            continue
        sn = top7_snaps(st, s)
        refs = collections.defaultdict(list)
        for i, x in enumerate(sn):
            if x["server"] and x["grade"] == "A" and x["h"] is not None:
                refs[x["server"]].append(i)
        pack = {}
        for sv, idx in refs.items():
            rh = np.array([sn[i]["h"] for i in idx])
            rs = np.full((len(idx), 7), np.nan)
            for j, i in enumerate(idx):
                for k, v in sn[i]["ranks"].items():
                    rs[j, k - 1] = v
            pack[sv] = (idx, rh, rs, np.array([sn[i]["person"] for i in idx], dtype=object))
        key2sn = {}
        for i, x in enumerate(sn):
            if x["grade"] == "C" and x["server"] in pack:
                for k in x["keys"]:
                    key2sn[k] = i
        cache = {}

        def br(i, mode, exclude_self_idx=None):
            x = sn[i]
            idx, rh, rs, pp = pack[x["server"]]
            m = np.ones(len(idx), bool)
            if mode == "all":
                if exclude_self_idx is not None:
                    m[exclude_self_idx] = False
            else:
                m = pp != x["person"]
            if m.sum() < 1:
                return None
            ks = np.array([k - 1 for k in x["ranks"]]); vs = np.array([x["ranks"][k] for k in x["ranks"]])
            return bracket_fast(ks, vs, rh[m], rs[m])
        for key, r in byS.get(s, {}).items():
            i = key2sn.get(key)
            if i is None:
                continue
            for mode in ("all", "excl"):
                if (i, mode) not in cache:
                    cache[(i, mode)] = br(i, mode)
                b_ = cache[(i, mode)]
                if b_ is None or b_[2]:
                    continue
                lo, hi = b_[0], b_[1]
                l2, h2 = max(lo, r["h_lo"]), min(hi, r["h_hi"])
                if l2 > h2:
                    continue
                el = max(0.0, r["h_hi"] - h2); eu = r["h_hi"] - l2
                out[mode]["C"].append({"S": s, "pid": r["pid"], "key": key, "el": el, "eu": eu, "dmax": r["h_hi"] - r["h_lo"],
                                       "clean": abs(r["h_hi"] - r["hp"]) < 2e-3, "screen": r["screen"], "site": r["site"], "self": bool(r["self"]),
                                       "hp": r["hp"], "h_hi": r["h_hi"], "h_lo": r["h_lo"], "server": r["server"],
                                       "h_lo_top7": r["h_lo_top7"], "h_hi_top7": r["h_hi_top7"]})
                if mode == "all":
                    tab = (r["h_lo_top7"], r["h_hi_top7"])
                    mine = (round(l2, 3), round(h2, 3))
                    agree["both_have"] += tab[0] is not None
                    agree["same"] += (tab[0] is not None and abs(tab[0] - mine[0]) < 2e-3 and abs(tab[1] - mine[1]) < 2e-3)
                    agree["mine_only"] += tab[0] is None
        # A leave-one-out
        for sv, (idx, rh, rs, pp) in pack.items():
            for j, i in enumerate(idx):
                x = sn[i]
                if x["hp"] is None or not (0.0 <= x["h"] <= x["H"]) or len(idx) < 2:
                    continue
                for mode in ("all", "excl"):
                    b_ = br(i, mode, exclude_self_idx=j)
                    if b_ is None or b_[2]:
                        continue
                    lo, hi = b_[0], b_[1]
                    l2, h2 = max(lo, 0.0), min(hi, x["hp"])
                    if l2 > h2:
                        continue
                    out[mode]["A"].append({"S": s, "key": x["key"], "d": x["hp"] - x["h"], "el": max(0.0, x["hp"] - h2), "eu": x["hp"] - l2, "dmax": x["hp"],
                                           "contains": bool(lo - 1e-9 <= x["h"] <= hi + 1e-9), "screen": x["screen"], "site": x["site"], "server": sv,
                                           "hp": x["hp"], "pid": x["person"]})
        out.setdefault("_n_snap", {})[s] = len(sn)
    out["_agree"] = dict(agree)
    return out


def check_bracket_fast(seasons, n=300, seed=1):
    """bracket_fast 가 post_time.top7_bracket 과 같은지 (임의 표본)."""
    import post_time as PT  # noqa: E402
    rng = np.random.default_rng(seed)
    bad = tot = 0
    for s in seasons:
        st = C.state_of(f"S{s}")
        sn = top7_snaps(st, s)
        refsx = collections.defaultdict(list)
        for x in sn:
            if x["server"] and x["grade"] == "A" and x["h"] is not None:
                refsx[x["server"]].append(x)
        for sv, L in refsx.items():
            if len(L) < 5:
                continue
            rh = np.array([x["h"] for x in L])
            rs = np.full((len(L), 7), np.nan)
            for i, x in enumerate(L):
                for k, v in x["ranks"].items():
                    rs[i, k - 1] = v
            for q in rng.choice(len(sn), size=min(8, len(sn)), replace=False):
                x = sn[q]
                if x["ranks"] and len(x["ranks"]) >= 2:
                    ks = np.array([k - 1 for k in x["ranks"]]); vs = np.array([x["ranks"][k] for k in x["ranks"]])
                    a = bracket_fast(ks, vs, rh, rs)
                    b = PT.top7_bracket(x["ranks"], [(r_["h"], r_["ranks"]) for r_ in L])
                    tot += 1
                    if not (abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 and a[2] == b[2]):
                        bad += 1
        if tot >= n:
            break
    return {"n_compared": tot, "n_mismatch": bad}


def qtab(a, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
    return {f"q{int(q*100):02d}": float(np.quantile(a, q)) for q in qs}


def _cl_index(L):
    m = {}
    return np.array([m.setdefault((x["S"], x["pid"]), len(m)) for x in L])


EDGES = [0, 1 / 60, 3 / 60, 0.15, 0.5, 1, 2, 6, 24, 1e9]


def calib_curve(A):
    """A leave-one-out: 진짜 지연 d 별 P(el>0 | d)."""
    d = np.array([x["d"] for x in A]); el = np.array([x["el"] for x in A])
    dk = np.clip(np.digitize(d, EDGES) - 1, 0, len(EDGES) - 2)
    g = np.array([np.mean(el[dk == k] > 1e-6) if (dk == k).sum() else np.nan for k in range(len(EDGES) - 1)])
    n = np.array([(dk == k).sum() for k in range(len(EDGES) - 1)])
    return g, n, dk


def w_equiv(target, gk, par):
    """빠른·느린 모양을 mix2 값에 고정하고, 보정곡선으로 el>0 비율 target 을 만드는 느린 성분 비중."""
    _, m1, s1, m2, s2 = par

    def share(wt):
        F = np.array([_mix2_cdf(np.array([e]), 1 - wt, m1, s1, m2, s2)[0][0] if e < 1e8 else 1.0 for e in EDGES])
        return float(np.dot(np.diff(F), np.nan_to_num(gk)))
    lo_w, hi_w = 0.0, 0.95
    if target <= share(lo_w):
        return 0.0
    if target >= share(hi_w):
        return hi_w
    for _ in range(60):
        m_ = 0.5 * (lo_w + hi_w)
        if share(m_) < target:
            lo_w = m_
        else:
            hi_w = m_
    return 0.5 * (lo_w + hi_w)


def tail_interval_fit(el, eu, tr, par, w=None):
    _, m1, s1, m2, s2 = par
    w = np.ones(len(el)) if w is None else w
    f = lambda wl: nll([wl, m1, math.log(s1), m2, math.log(s2)], "lognormal_mix2", el, eu, w, tr)
    r = optimize.minimize_scalar(f, bounds=(-6, 8), method="bounded")
    grid = np.linspace(-6, 8, 141)
    vals = np.array([f(g_) for g_ in grid])
    okg = grid[vals - r.fun < 1.92]
    wt = lambda z: 1 - 1 / (1 + math.exp(-z))
    return {"w_tail_mle": wt(r.x), "w_tail_profile95": [wt(okg.max()), wt(okg.min())] if len(okg) else None, "n": int(len(el))}


def one_mode(Cm, Am, smp, par, W, seed, Bc=300):
    """한 방식(all|excl)의 C 대 A leave-one-out 비교."""
    res = {}
    el = np.array([x["el"] for x in Cm]); eu = np.array([x["eu"] for x in Cm]); dm = np.array([x["dmax"] for x in Cm])
    clean = np.array([x["clean"] for x in Cm])
    scr = np.array([x["screen"] for x in Cm])
    ea = np.array([x["el"] for x in Am]); eua = np.array([x["eu"] for x in Am]); da = np.array([x["d"] for x in Am]); dma = np.array([x["dmax"] for x in Am])
    res["n_C"] = len(Cm); res["n_C_persons"] = len(set((x["S"], x["pid"]) for x in Cm)); res["n_A_loo"] = len(Am)
    res["A_loo_coverage_true_shot_in_interval"] = float(np.mean([x["contains"] for x in Am]))
    res["C_screen"] = dict(collections.Counter(scr)); res["A_screen"] = dict(collections.Counter(x["screen"] for x in Am))
    res["eu_quantiles_C"] = qtab(eu); res["eu_quantiles_A_loo"] = qtab(eua)
    res["width_median_C"] = float(np.median(eu - el)); res["width_median_A_loo"] = float(np.median(eua - ea))
    res["el_gt0_share"] = {"C": float((el > 1e-6).mean()), "C_wilson95": wilson(int((el > 1e-6).sum()), len(el)),
                           "A_loo": float((ea > 1e-6).mean()), "A_loo_wilson95": wilson(int((ea > 1e-6).sum()), len(ea)),
                           "C_rank_list_only": float((el[scr == "rank_list"] > 1e-6).mean()), "n_C_rank_list": int((scr == "rank_list").sum())}
    xs = [1, 6, 24]
    res["el_ge_x_share"] = {"x_h": xs, "C": [float((el >= x).mean()) for x in xs], "A_loo": [float((ea >= x).mean()) for x in xs],
                            "A_all_true_d_gt_x": [float((smp.d > x).mean()) for x in xs],
                            "A_loo_true_d_gt_x": [float((da > x).mean()) for x in xs]}
    ks_ = stats.ks_2samp(eu, eua)
    res["eu_C_vs_A_loo"] = {"KS": float(ks_.statistic), "KS_p_naive": float(ks_.pvalue), "MWU_p_naive": float(stats.mannwhitneyu(eu, eua).pvalue)}
    # 보정곡선 → C 의 el>0 가 A 의 지연 분포에서 나올 값
    g, nk, dk = calib_curve(Am)
    res["calibration"] = {"bins_h": EDGES[:-1], "n_A_loo": nk.tolist(), "P_el_gt0_given_d": g.tolist()}
    mrl = np.array([r["screen"] == "rank_list" for r in smp.rows])
    d_rl = smp.d[mrl]
    rng = np.random.default_rng(seed)
    clC = _cl_index(Cm)
    clA = _cl_index(Am)
    pred_b, obs_b, wq_b = [], [], []
    for b in range(Bc):
        ia = rng.choice(np.arange(len(Am)), len(Am))     # (군집 부트스트랩이 맞지만 A-LOO 는 사람 단위 중복이 적다 — 사진 단위)
        dkb = dk[ia]; elb = ea[ia]
        gb = np.array([np.mean(elb[dkb == k] > 1e-6) if (dkb == k).sum() else np.nan_to_num(g)[k] for k in range(len(EDGES) - 1)])
        wts = W[b % W.shape[0]][smp.cl][mrl].astype(float)
        kk = np.clip(np.digitize(d_rl, EDGES) - 1, 0, len(EDGES) - 2)
        pred_b.append(float(np.average(gb[kk], weights=wts)))
        cnt = np.bincount(rng.integers(0, clC.max() + 1, clC.max() + 1), minlength=clC.max() + 1)[clC]
        obs_b.append(float(np.sum(cnt * (el > 1e-6)) / cnt.sum()))
    kk = np.clip(np.digitize(d_rl, EDGES) - 1, 0, len(EDGES) - 2)
    pred = float(np.mean(np.nan_to_num(g)[kk]))
    obs = float((el > 1e-6).mean())
    pred_b, obs_b = np.array(pred_b), np.array(obs_b)
    res["el_gt0_check"] = {"C_observed": obs, "C_boot95_person": ci(obs_b), "predicted_if_C_delay_equals_A": pred, "predicted_boot95": ci(pred_b),
                           "ratio_obs_over_pred": obs / pred, "ratio_boot95": ci(obs_b / pred_b),
                           "w_tail_equivalent_C": w_equiv(obs, g, par), "w_tail_equivalent_C_boot95": [w_equiv(ci(obs_b)[0], g, par), w_equiv(ci(obs_b)[1], g, par)],
                           "w_tail_of_A_fit(mix2)": 1 - par[0], "w_tail_equivalent_of_predicted": w_equiv(pred, g, par)}
    # 구간 가능도 (비정보적 중도절단 가정)로 느린 성분 비중을 맞춘다: A leave-one-out 은 진짜 d 로도 맞춰 가정이 쓸 만한지 시험
    fit_ = {"C": tail_interval_fit(el, eu, dm, par), "A_loo_intervals": tail_interval_fit(ea, eua, dma, par)}
    lo_t, hi_t, kp = interval_of(da)
    fit_["A_loo_true_d"] = tail_interval_fit(lo_t[kp], hi_t[kp], np.full(kp.sum(), 1e9), par)
    res["interval_fit_w_tail"] = fit_
    # 빈틈 G = eu − d (A) 를 A 의 지연에 합친 분포 vs C 의 eu
    Gk = eua - da
    r2 = np.random.default_rng(seed + 1)
    sim = d_rl[r2.integers(0, len(d_rl), 100000)] + Gk[r2.integers(0, len(Gk), 100000)]
    ksr = stats.ks_2samp(eu, sim)
    res["eu_vs_d_plus_gap"] = {"gap_quantiles": qtab(Gk), "sim_quantiles": qtab(sim), "C_quantiles": qtab(eu), "KS": float(ksr.statistic),
                               "share_C_eu_gt_sim_q90": float((eu > np.quantile(sim, .9)).mean())}
    return res


def top7_checks2(M, rows, fit_obj, smp, W, seed):
    par = fit_obj["lognormal_mix2"]["par"]
    out = {"definition": "C 줄(use_indep, 게시 있음)을 top7 로 좁힌 구간. 지연 e = h_hi − 촬영 (h_hi = 게시, 앞선 사본이 있으면 그것). 촬영 ∈ top7 구간 ∩ [h_lo, h_hi] → e ∈ [el, eu]: "
                      "el = h_hi − 위끝(0 이상), eu = h_hi − 아래끝(h_lo 로 자름).",
           "agree_with_obs_table": M["_agree"], "n_snapshots_by_season": M.get("_n_snap")}
    Cs = M["all"]["C"]
    el = np.array([x["el"] for x in Cs]); eu = np.array([x["eu"] for x in Cs]); cl = _cl_index(Cs)
    out["structure_all_refs(운영 방식)"] = {"n_rows": len(Cs), "n_clean_h_hi_eq_hp": int(sum(x["clean"] for x in Cs)), "n_persons": int(cl.max() + 1),
                                          "n_C_use_indep_with_hp": int(sum(1 for r in rows if r["kind"] == "C" and r["use_indep"] and r["hp"] is not None)),
                                          "interval_width_h_quantiles": qtab(eu - el), "n_el_gt0(위끝이 게시보다 이른 줄 = 지연 하한이 생기는 줄)": int((el > 1e-6).sum()),
                                          "n_width_le_1h": int((eu - el <= 1).sum()), "n_width_le_2h": int((eu - el <= 2).sum()),
                                          "eu_quantiles(지연 상한)": qtab(eu)}
    fam = "lognormal_mix2"
    # 한쪽 점검: C = A 이면 m = F_A(eu)/F_A(dmax) 가 균등분포보다 오른쪽 (eu ≥ d 이므로 P(m ≤ u) ≤ u)
    dm = np.array([x["dmax"] for x in Cs])
    Fu, _ = cdf_sf(fam, par, eu); Fm, _ = cdf_sf(fam, par, dm)
    m_u = np.clip(Fu / np.maximum(Fm, 1e-300), 0, 1)
    out["pit_upper_bound(약한 점검: C 가 A 보다 훨씬 빠를 때만 어긴다)"] = {"u": [0.05, 0.2, 0.5], "share_m_le_u": [float((m_u <= u).mean()) for u in (0.05, 0.2, 0.5)], "median_m": float(np.median(m_u))}
    out["all_refs"] = one_mode(M["all"]["C"], M["all"]["A"], smp, par, W, seed)
    out["excl_same_person_refs"] = one_mode(M["excl"]["C"], M["excl"]["A"], smp, par, W, seed + 11)
    # 설명
    return out


def sensitivity(rows):
    """표본 정의를 바꿔 본 값 (분위·꼬리만)."""
    def go(f):
        X = [r for r in rows if r["kind"] == "A" and r["hp"] is not None and r["h"] is not None and 0 <= r["h"] <= r["H"] and f(r)]
        d = np.array([r["hp"] - r["h"] for r in X])
        return {"n": len(d), "median_min": float(np.median(d) * 60), "q75_min": float(np.quantile(d, .75) * 60), "q90_min": float(np.quantile(d, .9) * 60),
                "q95_h": float(np.quantile(d, .95)), "gt1h": float((d > 1).mean()), "gt6h": float((d > 6).mean())}
    return {"base use_indep": go(lambda r: r["use_indep"]), "use (중복 안 뺌)": go(lambda r: r["use"]),
            "use_indep, dc 만": go(lambda r: r["use_indep"] and r["site"] == "dc"),
            "use_indep, 본인 사진만": go(lambda r: r["use_indep"] and r["self"]),
            "use_indep, shot_src=filename 만": go(lambda r: r["use_indep"] and r["shot_src"] == "filename"),
            "use_indep, late>24h 표시 뺀 것": go(lambda r: r["use_indep"] and "late>24h" not in str(r["t_basis"])),
            "use_indep, via(다른 사본 시각) 뺀 것": go(lambda r: r["use_indep"] and " via" not in str(r["t_basis"])),
            "any A (use 아님 포함, 챌린지 중)": go(lambda r: True)}


def compare_prev(smp, rows):
    s41 = smp.S == 41
    d = smp.d[s41]
    out = {"S41_this_sample(use_indep, 0<=h<=H)": {"n": int(len(d)), "median_min": float(np.median(d) * 60), "q75_min": float(np.quantile(d, .75) * 60),
                                                     "q90_h": float(np.quantile(d, .9))}}
    A41 = [r for r in rows if r["S"] == 41 and r["kind"] == "A" and r["hp"] is not None and r["h"] is not None]
    for tag, f in [("A_any_kind_in_season(0<=h<=H)", lambda r: 0 <= r["h"] <= r["H"]),
                   ("A_use_in_season", lambda r: r["use"] and 0 <= r["h"] <= r["H"]),
                   ("A_use_all_h", lambda r: r["use"])]:
        dd = np.array([r["hp"] - r["h"] for r in A41 if f(r)])
        out[f"S41_{tag}"] = {"n": int(len(dd)), "median_min": float(np.median(dd) * 60), "q75_min": float(np.quantile(dd, .75) * 60),
                             "q90_h": float(np.quantile(dd, .9)), "n_neg": int((dd < 0).sum())}
    out["prev_cutoff_doc_S41_A292"] = {"n": 292, "median_min": 2.6, "q75_min": 10.0, "q90_h": 1.9, "source": "docs/solo-raid-cutoff.md (A 게시 지연)"}
    out["prev_analysis_2_3"] = {"C_vs_A_const_delay_h": 1.5, "boot90_h": [0.2, 3.4], "n_C_KR": 70,
                                "note": "곡면 잔차로 C 의 상수 지연을 거꾸로 추정 (explore_surface.py, 한섭 C 70줄). 분포가 아닌 점추정"}
    return out


def pick_check_rows(smp, T, A_loo):
    rows = smp.rows
    d = smp.d
    out = []
    def pack(r, tag):
        return {"tag": tag, "season": r["S"], "key": r["key"], "h": r["h"], "hp": r["hp"], "d_h": r["hp"] - r["h"] if r["h"] is not None else None,
                "shot_src": r["shot_src"], "t_basis": r["t_basis"], "screen": r["screen"], "site": r["site"], "self": r["self"]}
    neg = [r for r in rows if r["hp"] - r["h"] < 0]
    far = [r for r in neg if r["hp"] - r["h"] < -NEG_OK]
    out += [pack(r, "d<-5min (다른 사본 시각 — 표본에서 뺀다)") for r in far[:3]]
    out += [pack(r, "d<0 (≤5분, 시계 오차)") for r in neg if r not in far][:2]
    out += [pack(rows[i], "d 가장 큰 쪽") for i in (len(rows) - 1, len(rows) - 2)]
    mid = [i for i in range(len(rows)) if abs(d[i] - 1.0) < 0.01][:1]
    out += [pack(rows[i], "d ≈ 1시간") for i in mid]
    late = [r for r in rows if r["hp"] > r["H"]][:2]
    out += [pack(r, "촬영은 챌린지 중, 게시는 종료 뒤") for r in late]
    for t in sorted(T, key=lambda t: t["eu"] - t["el"])[:2]:
        out.append({"tag": "C top7 구간이 좁은 줄", "season": t["S"], "key": t["key"], "h_lo": t["h_lo"], "h_hi": t["h_hi"], "hp": t["hp"],
                    "h_lo_top7": t["h_lo_top7"], "h_hi_top7": t["h_hi_top7"], "el_h": t["el"], "eu_h": t["eu"], "screen": t["screen"], "self": t["self"]})
    for t in sorted([t for t in T if t["el"] > 6], key=lambda t: -t["el"])[:2]:
        out.append({"tag": "C 지연 하한이 6시간 넘는 줄 (위끝이 게시보다 훨씬 이름)", "season": t["S"], "key": t["key"], "h_lo": t["h_lo"], "h_hi": t["h_hi"], "hp": t["hp"],
                    "h_lo_top7": t["h_lo_top7"], "h_hi_top7": t["h_hi_top7"], "el_h": t["el"], "eu_h": t["eu"], "screen": t["screen"], "self": t["self"]})
    for t in [t for t in T if not t["clean"]][:1]:
        out.append({"tag": "C 위쪽 한계가 게시보다 이른 줄 (앞선 사본/이름)", "season": t["S"], "key": t["key"], "h_hi": t["h_hi"], "hp": t["hp"],
                    "h_lo_top7": t["h_lo_top7"], "h_hi_top7": t["h_hi_top7"]})
    for x in [x for x in A_loo if x["el"] > 6][:1] + [x for x in A_loo if x["d"] < 0.03][:1]:
        out.append({"tag": "A leave-one-out 대조 줄", "season": x["S"], "key": x["key"], "d_h": x["d"], "hp": x["hp"], "el_h": x["el"], "eu_h": x["eu"],
                    "contains_true_shot": x["contains"]})
    return r4(out)


def composition_adjusted(smp, rows, W):
    """C 줄의 (사이트 dc/기타 × 본인/남 × 화면) 구성에 A 를 맞춘 지연 (A 줄에 칸비 가중)."""
    def cell(r):
        return ("dc" if r["site"] == "dc" else "other", bool(r["self"]), r["screen"])
    Cs = [r for r in rows if r["kind"] == "C" and r["use_indep"] and r["hp"] is not None and r["h_hi"] is not None and 0 <= r["h_hi"] <= r["H"]]
    cC = collections.Counter(cell(r) for r in Cs)
    cA = collections.Counter(cell(r) for r in smp.rows)
    nC = sum(cC.values()); nA = sum(cA.values())
    wcell = {k: (cC[k] / nC) / (cA[k] / nA) for k in cA if cC.get(k)}
    unm = sum(cC[k] for k in cC if k not in cA) / nC
    w = np.array([wcell.get(cell(r), 0.0) for r in smp.rows])
    d = smp.d

    def stats_w(ww):
        out = {f"q{int(q*100):02d}": float(wq(d, ww, q)) for q in (0.25, 0.5, 0.75, 0.9, 0.95)}
        tw = ww.sum()
        out.update({f"gt{int(t)}h": float(ww[d > t].sum() / tw) for t in TAILS})
        out["mean_cap24h"] = float((np.minimum(d, 24) * ww).sum() / tw)
        return out
    pt = stats_w(w)
    bs = []
    for b in range(W.shape[0]):
        bw = W[b][smp.cl] * w
        if bw.sum() > 0:
            st = stats_w(bw)
            bs.append([st[k] for k in pt])
    bs = np.array(bs)
    comp = {"C_n": nC, "C_cells": {"|".join(map(str, k)): v / nC for k, v in cC.most_common()},
            "A_cells": {"|".join(map(str, k)): v / nA for k, v in cA.most_common()},
            "C_share_in_cells_without_A": unm,
            "A_reweighted": pt, "A_reweighted_boot95": {k: ci(bs[:, j]) for j, k in enumerate(pt)},
            "A_plain": {"q50": float(np.median(d)), "q90": float(np.quantile(d, .9)), "gt1h": float((d > 1).mean()), "gt6h": float((d > 6).mean()),
                        "mean_cap24h": float(np.minimum(d, 24).mean())}}
    return comp


def after_end_A(rows):
    """촬영이 종료 뒤인 A 사진(종료 뒤 화면): 지연이 얼마나 되나 (종료 뒤 게시 C 의 참고)."""
    X = [r for r in rows if r["kind"] == "A" and r["use_indep"] and r["hp"] is not None and r["h"] is not None and r["h"] > r["H"]]
    if not X:
        return {}
    d = np.array([r["hp"] - r["h"] for r in X])
    out = {"n": len(X), "quantiles": r4(quants(d)), "tails": r4(tails(d)), "mean_cap24h": float(np.minimum(d, 24).mean())}
    out["by_screen"] = {k: {"n": int(sum(1 for r in X if r["screen"] == k)),
                            "q50": float(np.median([r["hp"] - r["h"] for r in X if r["screen"] == k])),
                            "q90": float(np.quantile([r["hp"] - r["h"] for r in X if r["screen"] == k], .9))}
                        for k in sorted({r["screen"] for r in X})}
    return r4(out)


def end_expectation(smp, fam, par):
    """종료 뒤에 게시된 비율: 관측 vs 맞춘 분포가 예측하는 값 (Σ S_fit(H−h)). 촬영 시각이 종료에 가까울수록 꼬리 효과가 커진다."""
    H = np.array([r["H"] for r in smp.rows]); h = np.array([r["h"] for r in smp.rows]); hp = np.array([r["hp"] for r in smp.rows])
    rem = H - h
    obs_after = hp > H
    S = cdf_sf(fam, par, rem)[1]
    out = {}
    for tag, m in [("rem0-12h", rem <= 12), ("rem12-24h", (rem > 12) & (rem <= 24)), ("rem24-48h", (rem > 24) & (rem <= 48)), ("rem>48h", rem > 48)]:
        out[tag] = {"n": int(m.sum()), "obs_posted_after_end": int(obs_after[m].sum()), "exp_posted_after_end": float(S[m].sum()),
                    "obs_share": float(obs_after[m].mean()), "exp_share": float(S[m].mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--boot-fit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261029)
    a = ap.parse_args()
    t_start = time.time()
    rows = load(a.obs_dir)
    smp = build_sample(rows)
    rng = np.random.default_rng(a.seed)
    W = boot_weights(smp.K, a.boot, rng)
    d = smp.d
    result = {"item": "1a"}
    # ---------- 1 표본·분위·꼬리
    n_neg = int((d < 0).sum())
    nonneg_far = int((d < -NEG_OK).sum())
    allA = [r for r in rows if r["kind"] == "A"]
    result["sample"] = {
        "definition": "kind A · use_indep(중복 뺌) · hp 있음 · 0<=h<=H (촬영이 챌린지 중). 재표본 단위 = (시즌, pid) 사람",
        "n": smp.n, "n_persons": smp.K, "seasons": "S19-S42", "n_by_season": {f"S{s}": int((smp.S == s).sum()) for s in sorted(set(smp.S))},
        "n_kind_A_all": len(allA), "n_A_use": sum(1 for r in allA if r["use"]), "n_A_use_indep": sum(1 for r in allA if r["use_indep"]),
        "n_A_use_indep_h_in_0_H": smp.n,
        "n_A_use_indep_h_gt_H(종료 뒤 화면, 표본 밖)": sum(1 for r in allA if r["use_indep"] and r["h"] is not None and r["h"] > r["H"]),
        "n_d_negative": n_neg, "n_d_negative_within_5min(시계 오차)": n_neg - nonneg_far, "n_d_negative_over_5min(다른 사본 시각 — 가능도에서 뺌)": nonneg_far,
        "n_d_zero": int((d == 0).sum()),
        "d_min_positive_seconds": float(d[d > 0].min() * 3600), "resolution": "촬영·게시 모두 초 단위 → d 는 초 정수 (정확히 0 은 없음)",
        "persons_multi_photo": {"mean_photos_per_person": smp.n / smp.K, "max": int(np.bincount(smp.cl).max())},
        "post_collection_window": "게시 수집은 종료 뒤 약 90시간까지(시즌별 가장 늦은 게시 H+66~96h) → 90h 넘는 지연만 잘림. 표본 최대 d = %.1fh" % d.max(),
        "max_d_h": float(d.max()),
        "n_d_gt_24h_flagged_late": int(sum(1 for r in smp.rows if "late>24h" in str(r["t_basis"]))),
    }
    result["sample"]["negative_d_rows"] = r4([{"season": r["S"], "key": r["key"], "d_min": (r["hp"] - r["h"]) * 60, "t_basis": r["t_basis"]}
                                              for r in smp.rows if r["hp"] - r["h"] < 0])
    res_all = summarize(smp, np.ones(smp.n, bool), W)
    res_all.pop("_bs")
    result["quantiles"] = {"n": smp.n, "unit": "hours (unit_min 은 분)", "point": res_all["quantiles"],
                           "boot95_person": {k: res_all["boot95"][k] for k in ("med", "q75", "q90", "q95", "q99")},
                           "unit_min": {k: v * 60 for k, v in res_all["quantiles"].items()},
                           "bootstrap": "(시즌, pid) 사람 단위 재표본 %d회" % a.boot}
    result["tails"] = {"n": smp.n, **res_all["tails"], "boot95_person": {k: res_all["boot95"][k] for k in ("gt1h", "gt6h", "gt24h")},
                       "mean_cap24h": float(np.minimum(d, 24).mean()), "mean_cap24h_boot95": res_all["boot95"]["mean_cap24h"],
                       "mean_uncapped": float(d.mean())}
    # ---------- 2 나눠 보기
    grp, levels, het = section_groups(smp, W, None)
    result["by"] = {}
    names = {"site": "site", "day": "day_h24", "half_day": "half_day", "self": "self", "screen": "screen", "end": "end", "season": "season",
             "shot_src": "shot_src", "via_copy": "via_copy"}
    for k, fac in names.items():
        result["by"][k] = strip_bs(grp[fac])
    result["heterogeneity"] = {"season": het, "kw_tests": {k: grp[f]["_test"] for k, f in names.items()},
                               "note": "KW 는 순위 검정, 설계효과(군집내 순위 상관 ρ)로 보정한 p_adj 도 같이. 시즌 이질성: 시즌별 값의 퍼짐 Q 검정 (se 는 시즌 안 사람 부트스트랩)"}
    result["sensitivity_sample"] = r4(sensitivity(rows))
    result["adjusted_effects_ols_ln_d"] = r4(adjusted_effects(smp))
    result["composition_adjusted_A"] = r4(composition_adjusted(smp, rows, W))
    result["after_end_A"] = after_end_A(rows)
    # ---------- 3 모형용 꼴
    fits, (lo, hi, cl, dd) = section_fits(smp, W, a.seed, a.boot_fit)
    objs = fits.pop("_obj")
    result["fits"] = r4(fits["fits"])
    result["fit_meta"] = r4({k: fits[k] for k in fits if k != "fits"})
    mixf = objs["lognormal_mix2"]
    mix3 = objs["lognormal_mix3"]
    result["end_effect"] = r4({"note": "종료 뒤 게시 비율: 관측 vs 맞춘 분포(mix2·mix3)가 예측하는 Σ S(H−h). 촬영이 종료에 가까울수록 꼬리가 종료 뒤로 넘어가는 효과 (오른쪽 중도절단이 아니라 수집이 종료 뒤 약 90시간까지라 거의 안 잘림)",
                              "mix2": end_expectation(smp, "lognormal_mix2", mixf["par"]), "mix3": end_expectation(smp, "lognormal_mix3", mix3["par"])})
    # 부트스트랩 (사람 단위): mix2, mix3
    out_b = {}
    for fam_, f_, Bf in [("lognormal_mix2", mixf, a.boot_fit), ("lognormal_mix3", mix3, max(40, a.boot_fit // 2))]:
        bp, tsec = boot_params(fam_, f_, lo, hi, cl, None, Bf, a.seed + (1 if fam_.endswith("2") else 7))
        der = []
        for p_ in bp:
            q = fit_quantiles(fam_, tuple(p_), [0.5, 0.9])
            tl = fit_tails(fam_, tuple(p_))
            der.append([q[0], q[1], tl[0], tl[1], tl[2], mean_cap(fam_, tuple(p_), 24.0)])
        der = np.array(der)
        out_b[fam_] = {"B": len(bp), "seconds": tsec, "params_boot95": {k: ci(bp[:, j]) for j, k in enumerate(FAM[fam_][3])},
                       "derived_boot95": {"median_h": ci(der[:, 0]), "q90_h": ci(der[:, 1]), "gt1h": ci(der[:, 2]), "gt6h": ci(der[:, 3]),
                                          "gt24h": ci(der[:, 4]), "mean_cap24h": ci(der[:, 5])}}
    result["fit_boot"] = r4(out_b)
    # ---------- 시즌별 / 늘어나는 창
    seasons_all = sorted(set(smp.S))
    expd = expanding(smp, objs, [s for s in seasons_all if s >= 21], mixf["theta"], mix3["theta"])
    tailw = season_tail_weight(smp, mixf["par"], seasons_all)

    def series(key, sub=None, smin=25):
        v = []
        for s, r in expd.items():
            if int(s[1:]) < smin or (sub is not None and sub not in r) or (sub is None and key not in r):
                continue
            v.append(r[key] if sub is None else r[sub][key])
        return np.array(v)
    ex = {}
    for key in ["med", "q90", "gt1h", "gt6h", "mix2_med", "mix2_q90", "mix2_gt1h", "mix2_gt6h", "mix2_mean_cap24h",
                "mix3_med", "mix3_q90", "mix3_gt1h", "mix3_gt6h", "mix3_mean_cap24h"]:
        v = series(key)
        if len(v) > 2:
            ex[key] = {"S25_S42_min": float(v.min()), "max": float(v.max()), "sd": float(v.std(ddof=1)), "cv": float(v.std(ddof=1) / v.mean())}
    for sub, fam_ in [("mix2", "lognormal_mix2"), ("mix3", "lognormal_mix3")]:
        mixs = {k: series(k, sub) for k in FAM[fam_][3]}
        ex[f"{sub}_params_S25_S42"] = {k: {"min": float(v.min()), "max": float(v.max()), "sd": float(v.std(ddof=1))} for k, v in mixs.items() if len(v) > 2}
    errs = []
    for s, r in expd.items():
        if int(s[1:]) >= 25 and "own" in r:
            errs.append([math.log(r["med"] / r["own"]["med"]), r["gt1h"] - r["own"]["gt1h"], r["gt6h"] - r["own"]["gt6h"]])
    errs = np.array(errs)
    ex["expanding_minus_own_season"] = {"n_seasons": len(errs), "log_median_ratio_mean": float(errs[:, 0].mean()), "log_median_ratio_sd": float(errs[:, 0].std(ddof=1)),
                                        "gt1h_diff_mean": float(errs[:, 1].mean()), "gt1h_diff_sd": float(errs[:, 1].std(ddof=1)),
                                        "gt6h_diff_mean": float(errs[:, 2].mean()), "gt6h_diff_sd": float(errs[:, 2].std(ddof=1)),
                                        "note": "앞 시즌들로 잰 값 − 그 시즌 자신의 값 (S25~S42). 시즌 자신의 값에도 표본 잡음이 있다 (시즌당 100~400장)"}
    result["expanding_window"] = {"per_target_season": r4(expd), "stability": r4(ex)}
    result["season_tail_weight"] = r4(tailw)
    wt = np.array([v["w_tail"] for v in tailw.values()])
    result["season_tail_weight_summary"] = r4({"note": "빠른·느린 모양은 전체 mix2 값에 고정하고 시즌마다 느린 성분 비중만 맞춤", "n_seasons": len(wt),
                                               "pooled_w_tail": 1 - mixf["par"][0], "min": wt.min(), "max": wt.max(), "sd": wt.std(ddof=1), "mean": wt.mean()})
    # ---------- 추천
    best = min(fits["fits"], key=lambda f: f["AIC"])
    result["recommend"] = {
        "family": "lognormal_mix2",
        "why": "빠른 성분 + 긴 꼬리 성분의 2성분 로그정규 혼합. 로그정규 하나는 꼬리를 못 맞추고(ΔAIC 큼) 감마·와이블도 그렇다. 3성분(mix3)은 AIC가 더 낫지만 "
               "창마다 다른 국소해로 뛰어(w1 0.14~0.78) 매개변수가 식별되지 않아, 늘어나는 창에서 다시 맞출 때는 2성분을 쓰고 3성분은 꼬리 6시간 쪽 민감도로 둔다",
        "family_alternative_better_AIC": "lognormal_mix3",
        "params": par_dict("lognormal_mix2", mixf["par"]), "params_boot95_person": out_b["lognormal_mix2"]["params_boot95"], "n": int(len(lo)),
        "expanding": {k: {"n": v["n"], **v["mix2"]} for k, v in expd.items()},
        "params_mix3(참고, 창마다 식별 안 됨)": par_dict("lognormal_mix3", mix3["par"]), "params_mix2": par_dict("lognormal_mix2", mixf["par"]),
        "units": "mu 는 ln(시간), sigma 는 ln 척도. 성분 중앙값 = exp(mu) 시간",
        "components_mix3_median_min": [math.exp(mix3["par"][2]) * 60, math.exp(mix3["par"][4]) * 60, math.exp(mix3["par"][6]) * 60],
        "components_mix3_weights": [mix3["par"][0], mix3["par"][1], 1 - mix3["par"][0] - mix3["par"][1]],
        "boot": out_b,
        "expanding_note": "대상 시즌 s 에는 s 보다 앞 시즌 전부로 맞춘 값 (위 expanding[Ss], 자세한 것은 expanding_window.per_target_season). S21~S24 창은 표본 200~570장이라 mix3 는 안 냄(S25부터)",
        "low_end": "d=0 근처: 초 단위 구간 가능도 [d−1초, d+1초]; d≤0(5분 안)은 [0,1분] 으로 중도절단; 5분 넘는 음수는 뺌(다른 사본 시각)",
        "zero_note": "가능도에서 d 가 아주 작은 줄(<10초 ≈ 0.6%)은 구간 확률로 다루므로 0 에서 로그 발산이 없다. 모형에서 적분할 때 지연 하한은 0 으로 두면 된다",
        "best_AIC_family": best["family"]}
    # ---------- 4 top7 독립 점검
    t0 = time.time()
    M = top7_modes(rows, range(19, 43))
    chk = check_bracket_fast([41, 40, 37, 31, 25], n=200)
    top = top7_checks2(M, rows, objs, smp, W, a.seed + 5)
    top["bracket_fast_check_vs_post_time"] = chk
    top["seconds"] = time.time() - t0
    result["top7_check"] = r4(top)
    T = M["all"]["C"]
    result["compare_prev"] = r4(compare_prev(smp, rows))
    # ---------- 6 한 줄
    ml = {}
    for fam_ in ["lognormal", "lognormal_mix2", "lognormal_mix3"]:
        ml[fam_] = {"median_min": fit_quantiles(fam_, objs[fam_]["par"], [0.5])[0] * 60, "mean_cap24h_h": mean_cap(fam_, objs[fam_]["par"], 24.0)}
    result["bias_size_oneline"] = {
        "text": "C 를 게시 시각에 두면 숨은 촬영 시각이 늦게 잡히는 정도: 지연 중앙 %.1f분, 75%% %.1f분, 90%% %.1f분, 95%% %.1f시간, 평균(24h 절단) %.2f시간 (A 표본)" % (
            np.median(d) * 60, np.quantile(d, .75) * 60, np.quantile(d, .9) * 60, np.quantile(d, .95), np.minimum(d, 24).mean()),
        "from_A_sample": r4({"median_min": np.median(d) * 60, "q75_min": np.quantile(d, .75) * 60, "q90_min": np.quantile(d, .9) * 60, "q95_h": np.quantile(d, .95),
                             "mean_cap24h_h": np.minimum(d, 24).mean(), "gt1h": float((d > 1).mean())}),
        "fit": r4(ml)}
    result["check_rows"] = pick_check_rows(smp, T, M["all"]["A"])
    result["run"] = {"seconds": time.time() - t_start, "boot": a.boot, "boot_fit": a.boot_fit, "seed": a.seed}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(r4(result), f, ensure_ascii=False, indent=1)
    print("wrote", a.out, "%.0fs" % (time.time() - t_start))


if __name__ == "__main__":
    main()
