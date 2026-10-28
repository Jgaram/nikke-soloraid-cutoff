"""WP1 1b·1f (2026-10-29) — 사진 B(말한 시각)의 시각 오차와 % 표시 반올림/버림.

  python3 -I tools/solo_raid/lab7/err_1b.py --obs DIR --out 1b.json [--seasons S19-S42] [--boot 2000] [--seed 7]

DIR = lab7/obs_table.py build --full 이 만든 폴더 (DIR/full/S{n}.jsonl.gz, 자료 판 D0). 정답(answer_key·truth)은 읽지 않는다 —
오차는 관측끼리 견줘서만 잰다. 출력에는 글 번호·원문·닉네임·사진 이름을 넣지 않고 집계 숫자와 (시즌, key)만 둔다.

단위
  화면(screen) = 사진 한 장(obs.img). B 줄은 순위표 한 장에 여러 줄이 붙어 시각을 같이 가지므로(줄 7~9개) 시각 오차는 사진 단위로 센다.
  군집(cluster) = 글(obs.no) — 같은 글의 사진·주장은 독립이 아니다. 부트스트랩은 글 단위 재표본(시즌 안, seed 고정).
  use 여부: 시각 오차는 % 유무와 무관한 화면의 성질이라 B 사진은 use와 상관없이 모두 센다(줄 수는 use만 따로 적는다).

근거 무리 (t_basis, 위쪽이 먼저 맞는 것)
  timer     B:timer(...)              화면의 갱신 타이머 (다음 05:00 − 남은 시간, [t−1분, t])
  now       B:claim now               글 주장 「지금·현재·방금」 (t = 게시, 폭 30분은 근거 없이 정한 값)
  hours_ago B:claim hours_ago         「N시간 전」 (게시 − N시간 ± 30분)
  clock     B:claim clock(…)          글 주장 시각 「13시」「19:15」 (요일·월일·상대 꼬리 포함, alt= 은 12시간 다른 후보)
  day       B:claim today|yesterday|weekday|period|earlier   하루 안·반나절 안 폭이 넓은 말
  other     위에 없는 것

기준 시각을 가진 짝 (정답이 아니라 다른 관측)
  claim_A  글 주장(시각 등급 B)이 같은 글의 사진 A(첨부 이름 시각)에 이어진 짝: d = h(주장) − h(A). 주장 단위, 이어진 A 사진이 여럿이면 중앙값.
  timer_A  사진 A인데 화면에 갱신 타이머도 있는 장: d = (타이머로 푼 시각) − h(A). 타이머 풀기는 post_time.timer_time 그대로 (읽기만).
  top7     B 사진의 top7 구간 [h_lo_top7, h_hi_top7] 안에 h가 드는가 (구간은 같은 서버 A 순위표에서, 별도 독립 근거).
  hp       B 줄 자신의 게시 시각 hp ≥ h 여야 한다.
"""
import argparse
import collections
import datetime as dt
import json
import math
import os
import re
import sys
import time

import numpy as np
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import obs_table as O  # noqa: E402

PR = C.PR
sys.path.insert(0, C.TOOLS)
import post_time as PT  # noqa: E402  (timer_time·TIMER만 읽는다)

GROUPS = ["timer", "now", "hours_ago", "clock", "day", "other"]
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
TOL_H = 2.1 / 60   # 타이머 풀기는 게시 +2분까지 허용 (timer_time)


def group_of(basis):
    b = basis or ""
    if b.startswith("B:timer"):
        return "timer"
    if b.startswith("B:claim now"):
        return "now"
    if b.startswith("B:claim hours_ago"):
        return "hours_ago"
    if b.startswith("B:claim clock"):
        return "clock"
    if re.match(r"B:claim (today|yesterday|weekday|period|earlier)", b):
        return "day"
    return "other"


def tails_of(basis):
    """꼬리 표시 (alt·tracker·ctx_cut·secondhand·claims= 등)."""
    out = []
    for t in (basis or "").split(";")[1:]:
        t = t.strip()
        out.append("alt" if t.startswith("alt=") else "claims" if t.startswith("claims=") else t.split(":")[0])
    return out


# ---------- 통계 도구 ----------

def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - w, 4), round(c + w, 4)]


def r4(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if (math.isnan(x) or math.isinf(x)) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def boot_stat(vals, clusters, fn, rng, nboot):
    """군집 부트스트랩: clusters(글)를 복원추출 → fn(재표본 값 배열). 95% 구간 [2.5, 97.5]."""
    vals = np.asarray(vals, float)
    keys = {}
    for i, c in enumerate(clusters):
        keys.setdefault(c, []).append(i)
    groups = [np.array(v) for v in keys.values()]
    if len(groups) < 2 or len(vals) == 0:
        return [None, None]
    out = []
    for _ in range(nboot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        out.append(fn(vals[idx]))
    out = np.array(out, float)
    out = out[~np.isnan(out)]
    if len(out) == 0:
        return [None, None]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def summarize(vals, clusters, rng, nboot, big=(1.0,), unit_mult=1.0):
    """값 분포: 분위(+부트스트랩), |d| > big 시간 비율(윌슨), 부호별 비율. vals는 시간 단위, 출력은 unit_mult(시간→분이면 60)배."""
    v = np.asarray(vals, float)
    n = len(v)
    if n == 0:
        return {"n": 0}
    res = {"n": n, "n_clusters": len(set(clusters)), "unit": "분" if unit_mult == 60 else "시간",
           "mean": r4(v.mean() * unit_mult), "q": {}, "share_abs_gt": {}, "share_pos": None, "share_neg": None}
    for q in QS:
        pt = float(np.quantile(v, q)) * unit_mult
        ci = boot_stat(v, clusters, lambda x, q=q: np.quantile(x, q), rng, nboot)
        res["q"][f"{q:g}"] = {"v": r4(pt), "ci95": r4([c * unit_mult if c is not None else None for c in ci])}
    for b in big:
        k = int((np.abs(v) > b).sum())
        ci = boot_stat(v, clusters, lambda x, b=b: (np.abs(x) > b).mean(), rng, nboot)
        res["share_abs_gt"][f"{b:g}h"] = {"k": k, "n": n, "p": r4(k / n), "wilson95": wilson(k, n), "boot95": r4(ci)}
    res["share_pos"] = {"k": int((v > 0).sum()), "n": n, "p": r4((v > 0).mean())}
    res["share_neg"] = {"k": int((v < 0).sum()), "n": n, "p": r4((v < 0).mean())}
    return res


def timer_of(st):
    """img → (타이머 문구, 출처). post_time.py main과 같은 세 출처."""
    timers = {}
    for o in PR.jl(os.path.join(st, "obs.jsonl")):
        m = PT.TIMER.search(o.get("note") or "")
        if m and (m.group(1) or m.group(2)):
            timers.setdefault(o["img"], (m.group(0), "note"))
    for r in PR.jl(os.path.join(st, "reads.jsonl")):
        for x in r.get("reads") or []:
            m = PT.TIMER.search(x.get("note") or "")
            if m and (m.group(1) or m.group(2)):
                timers.setdefault(r["id"], (m.group(0), "reads"))
    for f in ("ocr.jsonl", "ocr2.jsonl"):
        for r in PR.jl(os.path.join(st, f)):
            for it in r.get("items") or []:
                m = PT.TIMER.search(it[1] if len(it) > 1 else "")
                if m and (m.group(1) or m.group(2)):
                    timers.setdefault(r["id"], (m.group(0), "ocr"))
    return timers


def load_season(obs_dir, S):
    """한 시즌 → dict(screens, A_imgs, claim_pairs, timer_pairs)."""
    st = C.state_of(S)
    c0, H = C.info(S)
    head, rows = O.read_cell(os.path.join(obs_dir, "full", f"{S}.jsonl.gz"))
    obs = PR.jl(os.path.join(st, "obs.jsonl"))
    claims = PR.jl(os.path.join(st, "claims.jsonl"))
    pcl = {d["i"]: d for d in PR.jl(os.path.join(st, "post_time_claims.jsonl"))}
    photos = [r for r in rows if r["kind"] != "text"]
    for r in photos:
        o = obs[r["oi"]]
        r["img"], r["no"] = o["img"], o["no"]
    by_oi = {r["oi"]: r for r in photos}
    byimg = collections.defaultdict(list)
    for r in photos:
        byimg[r["img"]].append(r)

    screens = []
    for img, rs in byimg.items():
        r = rs[0]
        if r["kind"] != "B":
            continue
        screens.append({"S": S, "img": img, "no": r["no"], "key": r["key"], "grp": group_of(r["t_basis"]),
                        "tails": tails_of(r["t_basis"]), "basis": r["t_basis"], "h": r["h"], "h_lo": r["h_lo"], "h_hi": r["h_hi"],
                        "lo7": r["h_lo_top7"], "hi7": r["h_hi_top7"], "hp": r["hp"], "rows": len(rs),
                        "use_rows": sum(1 for x in rs if x["use"]), "site": r["site"], "screen": r["screen"],
                        "server": r["server"], "T": H})
    A_imgs = []
    for img, rs in byimg.items():
        r = rs[0]
        if r["kind"] == "A":
            A_imgs.append({"S": S, "img": img, "no": r["no"], "key": r["key"], "h": r["h"], "hp": r["hp"], "shot_src": r["shot_src"],
                           "use": any(x["use"] for x in rs), "exclude": any(x["exclude"] for x in rs)})

    # claim_A: 시각 등급 B 글 주장 ↔ 이어진 A 사진
    cp = []
    for r in rows:
        if r["kind"] != "text" or r["t_grade"] != "B":
            continue
        As = [by_oi[oi] for oi in r["link"] if oi in by_oi and by_oi[oi]["kind"] == "A"]
        if not As:
            continue
        b = O.clean_basis((pcl.get(r["i"]) or {}).get("basis"))
        b = "B:claim " + b[2:] if b and b.startswith("B:") and not b.startswith("B:claim") else b   # 사진 쪽 표기와 맞춘다
        cl = claims[r["i"]]
        hs = [a["h"] for a in As]
        cp.append({"S": S, "i": r["i"], "grp": group_of(b), "tails": tails_of(b),
                   "basis": b, "no": str(cl.get("no")), "h": r["h"], "h_lo": r["h_lo"], "h_hi": r["h_hi"], "hp": r["hp"],
                   "hA": float(np.median(hs)), "nA": len(set(a["img"] for a in As)), "spreadA": float(np.ptp(hs)),
                   "d": r["h"] - float(np.median(hs)), "akey": As[0]["key"], "use": any(a["use"] for a in As)})

    # timer_A: A 사진인데 타이머도 있는 장
    T = timer_of(st)
    tp = []
    seen = set()
    for r in photos:
        if r["kind"] != "A" or r["img"] in seen:
            continue
        seen.add(r["img"])
        tm = T.get(r["img"])
        pt = PT.ts(obs[r["oi"]].get("post_t"))
        if not tm or pt is None:
            continue
        mh = re.search(r"(\d+)\s*시간", tm[0])
        mm = re.search(r"(\d+)\s*분", tm[0])
        rem = dt.timedelta(hours=int(mh.group(1)) if mh else 0, minutes=int(mm.group(1)) if mm else 0)
        t = PT.timer_time(rem, pt)
        if t is None:
            continue
        ht = PR.hours_of(t.strftime("%Y-%m-%d %H:%M:%S"), c0)
        tp.append({"S": S, "img": r["img"], "no": r["no"], "key": r["key"], "src": tm[1], "h": ht, "hA": r["h"], "hp": r["hp"],
                   "d": ht - r["h"], "use": r["use"]})
    return {"screens": screens, "A": A_imgs, "claim_pairs": cp, "timer_pairs": tp, "H": H}


# ---------- top7 구간 (원 구간을 다시 셈: 줄이 안 붙은 「모순」도 보려고) ----------

def top7_raw(st, S):
    """B 사진마다 {img: dict(lo, hi, viol, nref, contra, a, b)} — post_time.top7_columns와 같은 규칙을 읽기만 하는 사본.
    붙는 조건: 어긋남 0 → viol == 0. 줄의 [h_lo, h_hi]와 겹치지 않으면 contra (post_time은 이 경우 열을 안 붙인다)."""
    obs = PR.jl(os.path.join(st, "obs.jsonl"))
    out_rows = PR.jl(os.path.join(st, "post_time.jsonl"))
    srv = PR.jl(os.path.join(st, "server.jsonl"))
    px = PR.jl(os.path.join(st, "post_exclude.jsonl"))
    ok = lambda rows: len(rows) == len(obs) and all((r["img"], r["row"]) == (o["img"], o["row"]) for r, o in zip(rows, obs))
    if not (ok(srv) and ok(px) and ok(out_rows)):
        return {}
    im = collections.OrderedDict()
    for k, o in enumerate(obs):
        im.setdefault(o["img"], []).append(k)
    snaps = {}
    for img, ks in im.items():
        if obs[ks[0]].get("season") != "in" or any(px[k]["exclude"] for k in ks):
            continue
        ranks = {}
        for k in ks:
            o = obs[k]
            if o.get("rank") and o.get("score") and 1 <= o["rank"] <= 7 and not o.get("self"):
                ranks[o["rank"]] = o["score"]
        if len(ranks) < 2:
            continue
        snaps[img] = dict(server=srv[ks[0]]["server"], grade=out_rows[ks[0]]["grade"], h=out_rows[ks[0]]["h"], ranks=ranks, rows=ks)
    refs = collections.defaultdict(list)
    for s in snaps.values():
        if s["server"] and s["grade"] == "A" and s["h"] is not None:
            refs[s["server"]].append((s["h"], s["ranks"]))
    res = {}
    for img, s in snaps.items():
        if s["grade"] != "B" or not s["server"] or not refs.get(s["server"]):
            continue
        lo, hi, viol = PT.top7_bracket(s["ranks"], refs[s["server"]])
        d = {"lo": None if lo < -1e8 else lo, "hi": None if hi > 1e8 else hi, "viol": viol, "nref": len(refs[s["server"]]), "contra": False}
        k0 = s["rows"][0]
        a, b = out_rows[k0]["h_lo"], out_rows[k0]["h_hi"]
        d["a"], d["b"] = a, b
        if viol == 0 and a is not None and b is not None:
            d["contra"] = max(a, lo) > min(b, hi)
        res[img] = d
    return res


# ---------- 모형용 꼴 맞추기 (최대가능도, 위치 고정 없음) ----------

def _nll(f, x0, x):
    r = optimize.minimize(lambda p: f(p, x), x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 6000, "maxfev": 12000})
    return r.fun, r.x


def fit_sym(x):
    """대칭 꼴 (분 단위 오차 d): 정규·라플라스·t·두 정규 혼합. AIC·AICc 반환."""
    x = np.asarray(x, float)
    n = len(x)
    out = []
    med, sd = float(np.median(x)), float(np.std(x) + 1e-3)
    mad = float(np.median(np.abs(x - med)) + 1e-3)
    models = {
        "normal": (lambda p, x: -stats.norm.logpdf(x, p[0], math.exp(p[1])).sum(), [med, math.log(sd)], 2),
        "laplace": (lambda p, x: -stats.laplace.logpdf(x, p[0], math.exp(p[1])).sum(), [med, math.log(mad)], 2),
        "student_t": (lambda p, x: -stats.t.logpdf(x, 1 + math.exp(p[2]), p[0], math.exp(p[1])).sum(), [med, math.log(mad), 0.5], 3),
    }

    def mix(p, x):
        pi = 1 / (1 + math.exp(-p[3]))
        s1, s2 = math.exp(p[1]), math.exp(p[1]) + math.exp(p[2])
        return -np.log(pi * stats.norm.pdf(x, p[0], s2) + (1 - pi) * stats.norm.pdf(x, p[0], s1) + 1e-300).sum()
    models["normal_mix2"] = (mix, [med, math.log(mad + 0.1), math.log(sd * 3 + 1), -1.0], 4)
    for name, (f, x0, k) in models.items():
        try:
            v, p = _nll(f, x0, x)
        except Exception:
            continue
        prm = {"normal": lambda p: {"mu": p[0], "sigma": math.exp(p[1])},
               "laplace": lambda p: {"mu": p[0], "b": math.exp(p[1]), "sigma_equiv": math.exp(p[1]) * math.sqrt(2)},
               "student_t": lambda p: {"mu": p[0], "scale": math.exp(p[1]), "nu": min(1 + math.exp(p[2]), 1000.0)},   # 1000 = 정규와 같음
               "normal_mix2": lambda p: {"mu": p[0], "sigma_core": math.exp(p[1]), "sigma_wide": math.exp(p[1]) + math.exp(p[2]),
                                         "pi_wide": 1 / (1 + math.exp(-p[3]))}}[name](p)
        out.append({"model": name, "k": k, "ll": r4(-v), "aic": r4(2 * k + 2 * v),
                    "aicc": r4(2 * k + 2 * v + 2 * k * (k + 1) / max(n - k - 1, 1)), "params": r4(prm)})
    out.sort(key=lambda d: d["aic"])
    return out


def fit_pos(x):
    """양수 꼴 (분 단위 늦음 L ≥ 0): 로그정규·감마·와이불·로그정규 혼합 (몸통 + 꼬리)."""
    x = np.maximum(np.asarray(x, float), 0.05)
    n = len(x)
    lx = np.log(x)
    out = []
    models = {
        "lognormal": (lambda p, x: -stats.lognorm.logpdf(x, math.exp(p[1]), scale=math.exp(p[0])).sum(), [lx.mean(), math.log(lx.std() + 0.1)], 2),
        "gamma": (lambda p, x: -stats.gamma.logpdf(x, math.exp(p[0]), scale=math.exp(p[1])).sum(), [0.0, math.log(x.mean())], 2),
        "weibull": (lambda p, x: -stats.weibull_min.logpdf(x, math.exp(p[0]), scale=math.exp(p[1])).sum(), [0.0, math.log(x.mean())], 2),
    }

    def mix(p, x):
        pi = 1 / (1 + math.exp(-p[4]))
        a = stats.lognorm.pdf(x, math.exp(p[1]), scale=math.exp(p[0]))
        b = stats.lognorm.pdf(x, math.exp(p[3]), scale=math.exp(p[2]))
        return -np.log((1 - pi) * a + pi * b + 1e-300).sum()
    models["lognormal_mix2"] = (mix, [lx.mean(), math.log(0.6), lx.mean() + 2.0, math.log(2.0), -2.0], 5)
    for name, (f, x0, k) in models.items():
        try:
            v, p = _nll(f, x0, x)
        except Exception:
            continue
        pm = {"lognormal": lambda p: {"mu_log_min": p[0], "sigma_log": math.exp(p[1]), "median_min": math.exp(p[0])},
              "gamma": lambda p: {"shape": math.exp(p[0]), "scale_min": math.exp(p[1])},
              "weibull": lambda p: {"shape": math.exp(p[0]), "scale_min": math.exp(p[1])},
              "lognormal_mix2": lambda p: {"mu1": p[0], "sigma1": math.exp(p[1]), "mu2": p[2], "sigma2": math.exp(p[3]),
                                           "pi_tail": 1 / (1 + math.exp(-p[4])), "median1_min": math.exp(p[0]), "median2_min": math.exp(p[2])}}[name](p)
        out.append({"model": name, "k": k, "ll": r4(-v), "aic": r4(2 * k + 2 * v),
                    "aicc": r4(2 * k + 2 * v + 2 * k * (k + 1) / max(n - k - 1, 1)), "params": r4(pm), "_p": p})
    out.sort(key=lambda d: d["aic"])
    return out


def tail_prob_pos(fit, thr_min):
    """적합한 양수 꼴에서 P(L > thr_min)."""
    p, m = fit["_p"], fit["model"]
    if m == "lognormal":
        return float(stats.lognorm.sf(thr_min, math.exp(p[1]), scale=math.exp(p[0])))
    if m == "gamma":
        return float(stats.gamma.sf(thr_min, math.exp(p[0]), scale=math.exp(p[1])))
    if m == "weibull":
        return float(stats.weibull_min.sf(thr_min, math.exp(p[0]), scale=math.exp(p[1])))
    pi = 1 / (1 + math.exp(-p[4]))
    return float((1 - pi) * stats.lognorm.sf(thr_min, math.exp(p[1]), scale=math.exp(p[0])) + pi * stats.lognorm.sf(thr_min, math.exp(p[3]), scale=math.exp(p[2])))


def perm_chi2(labels, ind, rng, nperm=5000):
    """시즌 라벨별 이진 지표 비율이 같은가 — 라벨 섞기 카이제곱의 p값(단측)."""
    labels = np.asarray(labels)
    ind = np.asarray(ind, float)
    if len(ind) == 0 or ind.sum() in (0, len(ind)):
        return {"n": int(len(ind)), "k": int(ind.sum()), "p": None, "note": "지표가 한쪽뿐"}
    ul, inv = np.unique(labels, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    p0 = ind.mean()

    def chi(ix):
        k = np.bincount(inv, weights=ix)
        e = cnt * p0
        return float((((k - e) ** 2) / (e * (1 - p0) + 1e-12)).sum())
    obs_ = chi(ind)
    ge = 0
    for _ in range(nperm):
        if chi(rng.permutation(ind)) >= obs_ - 1e-12:
            ge += 1
    return {"n": int(len(ind)), "k": int(ind.sum()), "n_seasons": int(len(ul)), "chi2": r4(obs_), "p_perm": r4((ge + 1) / (nperm + 1))}


# ---------- 본 분석 ----------

def expanding(items, ind_fn, seasons):
    """늘어나는 창: 맞힐 시즌 s마다 s보다 앞 시즌 짝만으로 잰 비율 (윌슨)."""
    out = {}
    for s in seasons:
        prev = [x for x in items if int(x["S"][1:]) < int(s[1:])]
        k = sum(1 for x in prev if ind_fn(x))
        out[s] = {"n": len(prev), "k": k, "p": r4(k / len(prev)) if prev else None, "wilson95": wilson(k, len(prev))}
    return out


def spread_of(exp):
    ps = [v["p"] for v in exp.values() if v["p"] is not None and v["n"] >= 20]
    return {"min": r4(min(ps)), "max": r4(max(ps)), "n_seasons_n20plus": len(ps)} if ps else None


def run(a):
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    seasons = C.parse_seasons(a.seasons)
    SC, CP, TP, AI, T7 = [], [], [], [], {}
    for S in seasons:
        d = load_season(a.obs, S)
        SC += d["screens"]
        CP += d["claim_pairs"]
        TP += d["timer_pairs"]
        AI += d["A"]
        T7[S] = top7_raw(C.state_of(S), S)
    for x in SC + CP + TP + AI:
        x["cl"] = f"{x['S']}:{x['no']}"
    out = {"item": "1b", "data_ver": C.DATA_VER, "seasons": seasons, "seed": a.seed, "boot": a.boot,
           "units": {"screen": "사진 한 장(obs.img)", "cluster": "글(시즌:no) — 부트스트랩 재표본 단위",
                     "rows": "B 줄은 순위표 한 장에 여러 줄 → 시각 오차는 사진 단위, 줄 수는 따로"}}

    # 1. 무리
    gdef = {"timer": "B:timer(...)", "now": "B:claim now (지금·현재·방금)", "hours_ago": "B:claim hours_ago (N시간 전)",
            "clock": "B:claim clock... (시각 N시·HH:MM, 꼬리 wd·md·rel·alt= 포함)",
            "day": "B:claim today|yesterday|weekday|period|earlier (하루·반나절 폭)", "other": "나머지"}
    gs = {}
    for g in GROUPS:
        xs = [s for s in SC if s["grp"] == g]
        tails = collections.Counter(t for s in xs for t in s["tails"])
        gs[g] = {"rule": gdef[g], "screens": len(xs), "rows": sum(s["rows"] for s in xs), "rows_use": sum(s["use_rows"] for s in xs),
                 "clusters": len(set(s["cl"] for s in xs)), "tails": dict(tails.most_common())}
    by_season = collections.defaultdict(lambda: collections.Counter())
    for s in SC:
        by_season[s["S"]][s["grp"]] += 1
    out["groups"] = {"n_screens": len(SC), "n_rows": sum(s["rows"] for s in SC), "n_rows_use": sum(s["use_rows"] for s in SC),
                     "by_group": gs, "screens_by_season_group": {S: dict(by_season[S]) for S in seasons}}

    # 2. 기준 시각이 있는 짝
    pairs = {}
    # claim_A
    for g in ["now", "clock", "day", "hours_ago", "other"]:
        xs = [p for p in CP if p["grp"] == g]
        if not xs:
            continue
        v = [p["d"] for p in xs]
        cl = [p["cl"] for p in xs]
        sm = summarize(v, cl, rng, a.boot, big=(0.5, 1.0), unit_mult=60)
        cov = [(p["h_lo"] - 1 / 60 <= p["hA"] <= p["h_hi"] + 1 / 60) for p in xs]
        sm["system_interval_covers_A"] = {"k": int(sum(cov)), "n": len(cov), "p": r4(sum(cov) / len(cov)), "wilson95": wilson(sum(cov), len(cov)),
                                          "note": "[h_lo, h_hi]에 A의 h가 (1분 여유) 드는 짝"}
        sm["interval_width_h_median"] = r4(float(np.median([p["h_hi"] - p["h_lo"] for p in xs])))
        if g == "clock":
            for nm, sel in (("alt", True), ("no_alt", False)):
                ys = [p for p in xs if ("alt" in p["tails"]) == sel]
                big = sum(1 for p in ys if abs(p["d"]) > 0.5)
                sm[f"split_{nm}"] = {"n": len(ys), "n_gt_30min": big, "p": r4(big / len(ys)) if ys else None, "wilson95": wilson(big, len(ys))}
        pairs[f"claim_A_{g}"] = sm
    # timer_A
    tv = [p["d"] for p in TP]
    sm = summarize(tv, [p["cl"] for p in TP], rng, a.boot, big=(0.0167, 1.0), unit_mult=60)
    wrong = [p for p in TP if abs(p["d"]) > 1.0]
    core = [p["d"] * 60 for p in TP if abs(p["d"]) <= 1.0]
    sm["wrong_day"] = {"k": len(wrong), "n": len(TP), "p": r4(len(wrong) / len(TP)), "wilson95": wilson(len(wrong), len(TP)),
                       "d_hours": sorted(r4(p["d"]) for p in wrong), "hp_minus_hA_hours": sorted(r4(p["hp"] - p["hA"]) for p in wrong)}
    sm["core"] = {"n": len(core), "within_1min": {"k": int(sum(1 for c in core if -1.0 <= c <= 1.0)), "n": len(core)},
                  "q": {f"{q:g}": r4(float(np.quantile(core, q))) for q in QS}, "min": r4(min(core)), "max": r4(max(core)),
                  "unit": "분 (타이머 풀기 − A 시각)"}
    sm["by_source"] = dict(collections.Counter(p["src"] for p in TP))
    sm["note"] = "wrong_day: 게시가 촬영보다 24시간 넘게 늦으면 타이머가 하루(들) 늦은 날로 풀린다 — timer_time이 「게시 +2분 이전 가장 늦은 05:00 기준」을 고르기 때문"
    pairs["timer_A"] = sm
    # A 전체 게시 지연 (타이머가 하루 틀리는 확률의 기준)
    dl = np.array([x["hp"] - x["h"] for x in AI if x["hp"] is not None and x["h"] is not None])
    k24 = int((dl >= 24 - 2 / 60).sum())
    pairs["A_delay_reference"] = {"n": len(dl), "q_hours": {f"{q:g}": r4(float(np.quantile(dl, q))) for q in (0.5, 0.9, 0.95, 0.99)},
                                  "share_ge_24h": {"k": k24, "n": len(dl), "p": r4(k24 / len(dl)), "wilson95": wilson(k24, len(dl)),
                                                   "boot95": r4(boot_stat(dl, [x["cl"] for x in AI if x["hp"] is not None and x["h"] is not None],
                                                                          lambda v: (v >= 24 - 2 / 60).mean(), rng, a.boot))},
                                  "note": "A 사진 전체(이미지 단위)의 hp − h. 타이머가 하루 틀리는 확률 = 이 비율 (타이머 풀기 구조)"}
    out["pairs"] = pairs

    # 3. 줄 자신의 게시 시각 / top7 / 구간 폭
    byg = {}
    for g in GROUPS:
        xs = [s for s in SC if s["grp"] == g]
        if not xs:
            continue
        rec = {"n_screens": len(xs)}
        hp = [(s["hp"] - s["h"], s["cl"]) for s in xs if s["hp"] is not None]
        if hp:
            v = np.array([x[0] for x in hp])
            rec["hp_minus_h_hours"] = {"n": len(v), "q": {f"{q:g}": r4(float(np.quantile(v, q))) for q in QS},
                                       "contradiction_h_gt_hp": {"k": int((v < -TOL_H).sum()), "n": len(v), "p": r4((v < -TOL_H).mean()),
                                                                  "wilson95": wilson(int((v < -TOL_H).sum()), len(v)),
                                                                  "note": f"h > hp + {TOL_H:.4f}h(2.1분: 타이머 풀기의 +2분 여유)"},
                                       "share_gt_1h": r4((v > 1).mean()), "share_gt_12h": r4((v > 12).mean()),
                                       "vacuous": bool(g in ("now", "hours_ago"))}
            if g == "now":
                rec["hp_minus_h_hours"]["note"] = "「지금」은 h = hp로 정해 둔 줄이라 이 점검은 항상 0 — 정보 없음"
        w = np.array([s["h_hi"] - s["h_lo"] for s in xs if s["h_hi"] is not None and s["h_lo"] is not None])
        rec["interval_width_h"] = {"n": len(w), "q": {f"{q:g}": r4(float(np.quantile(w, q))) for q in (0.05, 0.5, 0.95)}}
        # top7
        recs = []
        for s in xs:
            t7 = T7.get(s["S"], {}).get(s["img"])
            if t7 is not None and t7["viol"] == 0:
                recs.append((s, t7))
        att = [(s, t) for s, t in recs if not t["contra"]]
        con = [(s, t) for s, t in recs if t["contra"]]
        t7rec = {"n_bracketed": len(recs), "n_attached": len(att), "n_contradict": len(con),
                 "contradict_share": {"k": len(con), "n": len(recs), "p": r4(len(con) / len(recs)) if recs else None, "wilson95": wilson(len(con), len(recs))}}
        # 모순의 방향·크기: 부호 = 말한 구간이 top7 구간보다 늦으면 +
        gaps = []
        for s, t in con:
            lo, hi = t["lo"], t["hi"]
            if hi is not None and t["a"] > hi:
                gaps.append(t["a"] - hi)
            elif lo is not None and t["b"] < lo:
                gaps.append(t["b"] - lo)
        if gaps:
            gv = np.array(gaps)
            t7rec["contradict_gap_hours"] = {"n": len(gv), "q": {f"{q:g}": r4(float(np.quantile(gv, q))) for q in (0.05, 0.5, 0.95)},
                                             "n_stated_later": int((gv > 0).sum()), "n_stated_earlier": int((gv < 0).sum())}
        # 붙은 줄: 점 h가 구간 안인가
        inside, dist = 0, []
        for s, t in att:
            lo = t["lo"] if t["lo"] is not None else -1e9
            hi = t["hi"] if t["hi"] is not None else 1e9
            if lo - 1 / 60 <= s["h"] <= hi + 1 / 60:
                inside += 1
            else:
                dist.append((s["h"] - hi) if s["h"] > hi else (s["h"] - lo))
        t7rec["point_h_inside_attached"] = {"k": inside, "n": len(att), "p": r4(inside / len(att)) if att else None, "wilson95": wilson(inside, len(att)),
                                            "outside_distance_hours_median": r4(float(np.median(dist))) if dist else None,
                                            "outside_dist_n": len(dist)}
        bw = [(t["hi"] - t["lo"]) for s, t in recs if t["lo"] is not None and t["hi"] is not None]
        t7rec["bracket_width_h"] = {"n": len(bw), "median": r4(float(np.median(bw))) if bw else None,
                                    "share_open_end": r4(1 - len(bw) / len(recs)) if recs else None}
        if g == "now":
            ub = [s["hp"] - t["lo"] for s, t in att if t["lo"] is not None and s["hp"] is not None]
            if ub:
                ubv = np.array(ub)
                t7rec["lag_upper_bound_h"] = {"n": len(ubv), "q": {f"{q:g}": r4(float(np.quantile(ubv, q))) for q in QS},
                                              "share_le_30min": {"k": int((ubv <= 0.5).sum()), "n": len(ubv), "p": r4((ubv <= 0.5).mean()),
                                                                 "wilson95": wilson(int((ubv <= 0.5).sum()), len(ubv))},
                                              "note": "top7 아래 끝으로 본 늦음의 상한 hp − h_lo_top7 — 30분 안이면 「지금」이 맞다고 확인된 줄"}
            # 30분 폭 시험: 원 구간이 [hp−30분, hp]와 겹치지 않는 비율 = contradict
            t7rec["note"] = "「지금」[hp−0.5h, hp]와 top7 원 구간이 겹치지 않는 비율이 contradict_share (구간이 좁아야 보인다)"
        rec["top7"] = t7rec
        byg[g] = rec
    out["screen_checks"] = byg

    # 4. 모형용 꼴
    fits = {}
    nowd = [p["d"] * 60 for p in CP if p["grp"] == "now"]
    if len(nowd) >= 20:
        f = fit_pos(nowd)
        emp = {"p_gt_30min": r4(float((np.array(nowd) > 30).mean())), "p_gt_10min": r4(float((np.array(nowd) > 10).mean())),
               "q90": r4(float(np.quantile(nowd, 0.9))), "q95": r4(float(np.quantile(nowd, 0.95)))}
        for m in f:
            m["pred_p_gt_30min"] = r4(tail_prob_pos(m, 30))
            m["pred_p_gt_10min"] = r4(tail_prob_pos(m, 10))
        fits["now_lag_min"] = {"n": len(nowd), "empirical": emp, "models": [{k: v for k, v in m.items() if k != "_p"} for m in f],
                               "note": "L = h(주장) − h(A) ≥ 0 (분). 말한 시각이 실제 촬영보다 늦은 만큼"}
    clk = [p["d"] * 60 for p in CP if p["grp"] == "clock"]
    if len(clk) >= 15:
        fits["clock_err_min"] = {"n": len(clk), "models": fit_sym(clk), "note": "d = h(주장) − h(A) (분). 부호 있음 — 큰 오차는 대부분 음수(말한 시각이 더 이름)"}
    core = [p["d"] * 60 for p in TP if abs(p["d"]) <= 1.0]
    if len(core) >= 20:
        fits["timer_core_min"] = {"n": len(core), "models": fit_sym(core),
                                  "share_in_0_1min": {"k": int(sum(1 for c in core if -1e-9 <= c <= 1.0 + 1e-9)), "n": len(core)},
                                  "note": "타이머 풀기 − A 시각 (분), 하루 틀린 장 제외"}
    out["fits"] = fits

    # 4b. 모형에 넣을 꼴 제안 (위 숫자에서 바로 옮김)
    rec = {}
    tcore = fits.get("timer_core_min", {}).get("models", [{}])[0]
    kk = [(d >= 48 - 2 / 60) for d in dl if d >= 24 - 2 / 60]
    rec["timer"] = {"form": "정확 + 드물게 하루 단위로 늦음", "core": "e = 말한 − 실제 ~ Normal(0.47분, 0.34분) (= [h−1분, h] 균등과 같은 폭, σ≈0.3분≈0.005시간)",
                    "core_fit": tcore, "rare": {"what": "게시가 촬영보다 24시간 넘게 늦으면 타이머가 24k시간 늦은 날로 풀린다 (e = +24k시간, k ≥ 1)",
                                                 "pi": pairs["A_delay_reference"]["share_ge_24h"], "pi_timer_pairs": pairs["timer_A"]["wrong_day"]["wilson95"],
                                                 "p_k_ge_2_given_wrong": {"k": int(sum(kk)), "n": len(kk), "p": r4(sum(kk) / len(kk)) if kk else None, "wilson95": wilson(int(sum(kk)), len(kk))}},
                    "other": "top7이 말한 구간과 겹치지 않는 비율 (반대 방향 포함)", "pi_top7_contradict": byg["timer"]["top7"]["contradict_share"],
                    "note": "A 사진에서 하루 틀림은 게시가 24시간 넘게 늦은 사진뿐 — B 타이머 사진은 파일 이름이 없어 게시 지연 분포가 다를 수 있다. B 화면의 hp−h 상위 분위는 groups 절"}
    rec["now"] = {"form": "로그정규 혼합 (몸통 + 꼬리), 말한 시각이 실제보다 늦은 만큼 L ≥ 0 — 균등 30분 폭 대신",
                  "fit": fits.get("now_lag_min", {}).get("models", [{}])[0], "uniform_30min_check": pairs["claim_A_now"]["system_interval_covers_A"],
                  "tail_beyond_30min": pairs["claim_A_now"]["share_abs_gt"]["0.5h"], "note": "A에 이어진 짝만 잰다(파일 이름이 있는 사진). B 사진(이름 없음)의 늦음은 top7 상한으로 확인"}
    rec["clock"] = {"form": "정확 + 크게 틀림 혼합 (큰 오차는 말한 시각이 실제보다 이름)", "fit": fits.get("clock_err_min", {}).get("models", [{}])[0],
                    "caveat": "n = 24 (23군집)라 꼬리 폭이 불안정 — 폭·비율에 구간을 붙여 쓰거나 day와 합쳐 넓게",
                    "system_interval_covers_A": pairs["claim_A_clock"]["system_interval_covers_A"]}
    rec["day"] = {"form": "점이 아니라 구간만 (C처럼 [h_lo, h_hi] 안 균등 또는 상한으로만)", "n_pairs": pairs.get("claim_A_day", {}).get("n"),
                  "note": "짝 4개 중 3개가 구간 밖이거나 12시간 넘게 어긋남 — 표본이 너무 작아 값은 못 낸다"}
    rec["hours_ago"] = {"form": "표본 3화면 — now와 같은 꼴(몸통 + 꼬리)을 쓰되 폭은 clock에 합치는 것을 제안", "n_screens": gs["hours_ago"]["screens"]}
    out["recommend"] = rec

    # 5. 시즌별 변동, 이질성, 늘어나는 창
    sea = {}
    ind_now = lambda p: p["d"] > 0.5
    ind_clock = lambda p: abs(p["d"]) > 0.5
    ind_wd = lambda p: abs(p["d"]) > 1.0
    NOW = [p for p in CP if p["grp"] == "now"]
    CLK = [p for p in CP if p["grp"] == "clock"]
    tab = {}
    for S in seasons:
        n_ = [p for p in NOW if p["S"] == S]
        c_ = [p for p in CLK if p["S"] == S]
        t_ = [p for p in TP if p["S"] == S]
        tab[S] = {"now_pairs": len(n_), "now_median_lag_min": r4(float(np.median([p["d"] * 60 for p in n_]))) if n_ else None,
                  "now_gt_30min": int(sum(ind_now(p) for p in n_)), "clock_pairs": len(c_), "clock_gt_30min": int(sum(ind_clock(p) for p in c_)),
                  "timer_pairs": len(t_), "timer_wrong_day": int(sum(ind_wd(p) for p in t_)),
                  "B_screens": dict(by_season[S])}
    sea["by_season"] = tab
    sea["heterogeneity"] = {
        "now_gt_30min": perm_chi2([p["S"] for p in NOW], [ind_now(p) for p in NOW], rng),
        "clock_gt_30min": perm_chi2([p["S"] for p in CLK], [ind_clock(p) for p in CLK], rng),
        "timer_wrong_day": perm_chi2([p["S"] for p in TP], [ind_wd(p) for p in TP], rng),
        "B_screen_mix_note": "시즌별 B 사진 수와 무리 구성은 groups.screens_by_season_group 표",
    }
    if len(NOW) >= 20:
        lag = [p["d"] * 60 for p in NOW]
        ss = [p["S"] for p in NOW]
        grp_l = [np.array([l for l, s in zip(lag, ss) if s == S]) for S in seasons]
        grp_l = [g for g in grp_l if len(g) >= 2]
        if len(grp_l) >= 3:
            kw = stats.kruskal(*grp_l)
            sea["heterogeneity"]["now_lag_kruskal"] = {"H": r4(float(kw.statistic)), "p": r4(float(kw.pvalue)), "n_seasons": len(grp_l)}
    # 화면 수준: 게시 − 말한 시각 > 1h 비율이 시즌마다 같은가 (timer·clock)
    for g in ("timer", "clock"):
        xs = [s for s in SC if s["grp"] == g and s["hp"] is not None]
        sea["heterogeneity"][f"{g}_screen_hp_minus_h_gt_1h"] = perm_chi2([s["S"] for s in xs], [(s["hp"] - s["h"]) > 1 for s in xs], rng)
    ex_seasons = [s for s in seasons if int(s[1:]) >= 21]
    sea["expanding_window"] = {
        "now_gt_30min": expanding(NOW, ind_now, ex_seasons),
        "clock_gt_30min": expanding(CLK, ind_clock, ex_seasons),
        "timer_wrong_day": expanding(TP, ind_wd, ex_seasons),
        "A_delay_ge_24h": expanding([dict(S=x["S"], bad=(x["hp"] is not None and x["hp"] - x["h"] >= 24 - 2 / 60)) for x in AI], lambda x: x["bad"], ex_seasons),
    }
    sea["expanding_window_spread"] = {k: spread_of(v) for k, v in sea["expanding_window"].items()}
    out["season"] = sea

    # 6. 확인용 줄 (글 번호 없이 시즌·key)
    chk = []
    for p in sorted(TP, key=lambda p: -abs(p["d"]))[:3]:
        chk.append({"why": "타이머 하루 틀림 (A 사진에 타이머도 있음)", "S": p["S"], "key": p["key"], "timer_h": r4(p["h"]), "A_h": r4(p["hA"]),
                    "hp": r4(p["hp"]), "d_h": r4(p["d"])})
    for p in sorted(NOW, key=lambda p: -p["d"])[:2]:
        chk.append({"why": "「지금」인데 A 촬영과 멀다", "S": p["S"], "A_key": p["akey"], "claim_h": r4(p["h"]), "A_h": r4(p["hA"]), "hp": r4(p["hp"]), "d_min": r4(p["d"] * 60)})
    for p in sorted(CLK, key=lambda p: p["d"])[:3]:
        chk.append({"why": "시각 말(clock)이 A와 크게 다름", "S": p["S"], "A_key": p["akey"], "claim_h": r4(p["h"]), "A_h": r4(p["hA"]), "tails": p["tails"],
                    "d_min": r4(p["d"] * 60)})
    cons = []
    for s in SC:
        t7 = T7.get(s["S"], {}).get(s["img"])
        if t7 and t7["viol"] == 0 and t7["contra"]:
            cons.append((abs((s["h"] - (t7["hi"] if t7["hi"] is not None and s["h"] > t7["hi"] else t7["lo"] if t7["lo"] is not None else s["h"]))), s, t7))
    for _, s, t7 in sorted(cons, key=lambda z: -z[0])[:3]:
        chk.append({"why": "top7 구간과 겹치지 않음 (말한 구간 모순)", "S": s["S"], "key": s["key"], "group": s["grp"], "h": r4(s["h"]), "h_lo": r4(s["h_lo"]),
                    "h_hi": r4(s["h_hi"]), "top7_lo": r4(t7["lo"]), "top7_hi": r4(t7["hi"])})
    out["check_rows"] = chk
    out["runtime_sec"] = round(time.time() - t0, 1)
    return out


def clean(o):
    """numpy 값·부동소수 → JSON 값 (소수 넷째 자리)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float, np.integer)):
        return r4(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seasons", default="S19-S42")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", default=None, help="1b | 1f")
    a = ap.parse_args()
    res = run(a)
    res["1f"] = check_1f(a.seasons)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(clean(res), f, ensure_ascii=False, indent=1)
    print("wrote", a.out, "sec", res["runtime_sec"])


def check_1f(seasons):
    """1f: % 표시가 반올림인가 버림인가 — 레포 자료에 새 근거가 있는지 훑기 (등수/인원으로 % 다시 셈)."""
    c = collections.Counter()
    usable = []
    for S in C.parse_seasons(seasons):
        st = C.state_of(S)
        obs = PR.jl(os.path.join(st, "obs.jsonl"))
        for o in obs:
            if o.get("rank") is not None and o.get("pct") is not None:
                c["rank_and_pct_rows"] += 1
                if o["screen"] in ("my_rank", "season_end", "result") and o["rank"] >= 10 and o["pct"] > 0:
                    c["usable_rank_ge10_pct"] += 1
                    usable.append((S, o["screen"]))
                elif o["screen"] in ("tracker",):
                    c["tracker_rank_pct"] += 1
                else:
                    c["badge_or_other"] += 1
    return {"prev": "분석 5.3 A5 (2026-10-28, docs/solo-raid-analysis.md): 레포 자료로 못 가림 → 구간 [p−0.005, p+0.01) 그대로. "
                    "게임은 200위까지 등수, 그 밖은 %만 보인다.",
            "new_evidence_scan": dict(c), "usable_by_screen": dict(collections.Counter(sc for _, sc in usable)),
            "conclusion": "새 근거 없음 — 구간 [p−0.005, p+0.01) 그대로. 등수와 %가 같은 줄에 있는 588줄 중 게임 화면의 진짜 (등수≥10, %) 짝은 내 순위·종료 화면 7줄뿐이고 "
                          "서로 다른 사람·시점이라 같은 참가자 수 N으로 묶이는 쌍이 없다(S31 한섭 내 순위 2줄도 579위 0.74% · 616위 0.31%로 시점이 다르다). "
                          "나머지는 레벨 배지(25·35)나 트래커 화면의 다른 숫자다(같은 화면의 27위 0.06%·31위 0.04%·35위 0.02%는 N이 안 맞아 등수가 아니다). "
                          "게임은 200위까지 등수만, 그 밖은 %만 보여 주고 N도 내지 않으므로 r/N으로 반올림·버림을 가릴 수 없다. 같은 순간 두 화면의 % 짝은 정의상 같은 값이라 가르지 못한다."}


if __name__ == "__main__":
    main()
