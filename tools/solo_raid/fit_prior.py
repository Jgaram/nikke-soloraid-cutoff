"""솔로레이드 컷 예측의 사전 정보(prior)를 끝난 시즌 하나로 만든다 (모델 판독 0, 요청 0).

python3 -I tools/solo_raid/fit_prior.py STATE --out PRIOR.json [--grades A|all] [--fit-grades AB|all] [--boot 200] [--seed 1]

입력 (읽기만 한다. state/에 쓰는 것은 --out 하나뿐)
- STATE/season.json · core.jsonl · obs.jsonl
- STATE/truth_points.jsonl  정답 곡선 점 (A+A2 기본, --grades A면 A만; best_record·mono_viol 뺌 — explore_pct_transform와 같다)
- STATE/truth_table.json    커뮤니티 집계표(B) — 일섭–한섭 모양 차이 D의 보조 닻으로만 쓴다

**모든 값은 이 시즌 하나에서 잰 값이다** (S41이면 「S41 한 시즌에서 잰 값」). 시즌 간 차이는 잴 수 없어 season_diff에 가정으로 둔다.

들어가는 것
(a) % 변환 transform: P = log10(같은 점수의 최종 % / 지금 %)를 시각 h의 함수로.
    - 관측: explore_pct_transform.load_obs(기본 집합·시점 대표·종료 전, C는 상한 h)에 정답 곡선 역보간으로 최종 %를 붙인 줄
      (덮인 구간 밖은 값 없음). 사람·일차 칸마다 중앙값 하나로 줄여 맞춘다 (반복 행을 독립으로 세지 않는다).
    - 꼴 후보: sqrt  P = c(1 − √(h/H))  (탐색 2.2의 곡면 기본형) · pow  P = a(1 − h/H)^b  · f2pow  F = w t^p + (1−w) t^q, P = −log10 F
      (F = 지금 %/최종 %, F(0) = 0·F(H) = 1). 한섭 칸으로 Huber 손실 강건 적합, **사람 하나씩 빼기(LOPO) |잔차| 중앙값이 가장 작은 꼴**
      (5% 안이면 모수 적은 쪽)을 고른다. x 항 P·(1 + bx·log10 x)도 같은 기준(5% 넘게 좋아질 때만)으로 넣고 뺀다.
    - 일섭: 한섭 꼴에 수준 배수 m_JP 하나만 맞춘다 (칸 5개 미만이면 1). 글섭: 표본이 없어 **한섭을 그대로 빌린다 (가정)**.
    - 잔차: σ(h) = √(s0² + (ρ·P(h))²) (일차 칸의 1.4826·MAD에 맞춤), 표준화 잔차 z 목록 (predict가 여기서 뽑는다).
    - 모수 불확실성: 사람 단위 붓스트랩 (서버마다 사람을 다시 뽑아 꼴 모수·m·σ를 다시 맞춘 복제본).
    - 기준선 1용 일차별 중앙 P (day_median), 같은 점수 쌍의 잡음 pair_sigma (시즌 P 수준 갱신용), C 줄 잔차 점검.
(b) 최종 곡선 모양 shape: ln r(u) = ln(s(%)/s(3%)), u = ln %, 격자 0.1~40%.
    - 한섭: 정답 점 (ln %, 점수) 꺾은선. 빈 구간(이웃 % 비 > 1.5이고 차 > 0.5)은 꺾은선 그대로 두고 own_gap, 끝 밖은 끝 선분 연장 extrap.
    - 일섭·글섭: 자기 점의 덮인 선분은 own, 빈 구간은 한섭 모양을 양 끝에 맞춰 늘인 KR_warp, 끝 밖은 한섭 비율로 이은 KR_ratio.
    - D(u) = ln r_일섭 − ln r_한섭 (일섭 점 + 집계표 B 일섭 10·30%, 3%에서 0). 시즌 간 모양 차이의 방향 (가정의 재료).
(c) season_diff 기본값 (가정, predict 옵션으로 바꾼다): τ_P, τ_shape, τ_ratio, τ_scale0 — 값과 근거 문자열.
"""
import argparse
import collections
import datetime as dt
import importlib.util
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import predict as PR  # noqa: E402

_spec = importlib.util.spec_from_file_location("explore_pct_transform", os.path.join(HERE, "explore_pct_transform.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

FORMS = {"sqrt": [1.2], "pow": [math.log(0.8), 0.9], "f2pow": [1.0, math.log(0.8), math.log(10.0)]}
NPAR = {"sqrt": 1, "pow": 2, "f2pow": 3}
GRID_LO, GRID_HI, GRID_DU = math.log(0.1), math.log(40.0), 0.02
LABEL = "{season} 한 시즌에서 잰 값"


# ---------------- 최적화 ----------------

import importlib.util as _ilu  # noqa: E402
_sp = _ilu.spec_from_file_location("split", os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.py"))
split = _ilu.module_from_spec(_sp)
_sp.loader.exec_module(split)

def nelder_mead(f, x0, step=0.3, iters=400, tol=1e-9):
    n = len(x0)
    pts = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += step
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = [sum(p[j] for p in pts[:-1]) / n for j in range(n)]
        xr = [cen[j] + (cen[j] - pts[-1][j]) for j in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [cen[j] + 2 * (cen[j] - pts[-1][j]) for j in range(n)]
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = [cen[j] + 0.5 * (pts[-1][j] - cen[j]) for j in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = [pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(n)]
                    vals[i] = f(pts[i])
    i = min(range(n + 1), key=lambda i: vals[i])
    return pts[i], vals[i]


def huber(r, c):
    a = abs(r)
    return 0.5 * r * r if a <= c else c * (a - 0.5 * c)


def mad(v):
    if not v:
        return None
    m = statistics.median(v)
    return 1.4826 * statistics.median(abs(a - m) for a in v)


# ---------------- 자료 ----------------

def load_units(state, grades_curve, fit_grades, persons=None):
    """(H, 관측 줄[P 붙음], 사람·일차 칸). persons가 있으면 그 사람만."""
    season, H = E.season_hours(state)
    H = round(H)   # 04:59:59 → 120 (predict.season_info와 같게)
    curves = {s: E.Curve(E.final_points(state, s, grades_curve), "lin") for s in PR.SERVERS}
    allobs = E.attach(E.load_obs(state, H, with_c=True), curves)
    allobs = [o for o in allobs if persons is None or o["person"] in persons]
    if fit_grades == "AB":
        allobs = [o for o in allobs if o["grade"] in ("A", "B")]
    obs = [o for o in allobs if o["P"] is not None]
    g = collections.defaultdict(list)
    for o in obs:
        g[(o["server"], o["person"], int(o["h"] // 24))].append(o)
    units = []
    for (s, p, d), L in g.items():
        units.append({"server": s, "person": p, "day": d + 1, "h": statistics.median(o["h"] for o in L),
                      "P": statistics.median(o["P"] for o in L), "x": math.exp(statistics.median(math.log(o["x"]) for o in L)),
                      "grade": "".join(sorted({o["grade"] for o in L})), "n": len(L), "cens": None})
    # 구간 관측 (2026-10-16 추가): 정답 곡선의 빈 구간·끝 밖에 떨어져 최종 %를 한 값으로 못 붙인 관측도
    # 「최종 %가 이웃 정답 점 사이」라는 구간 정보로 맞추기에 넣는다 (같은 사람·일차 칸에 값 있는 칸이 있으면 뺀다)
    have = {(u["server"], u["person"], u["day"] - 1) for u in units}
    for o in allobs:
        if o["P"] is not None or o["server"] not in curves:
            continue
        c = curves[o["server"]]
        if c.n < 2 or (o["server"], o["person"], int(o["h"] // 24)) in have:
            continue
        lo = hi = None
        if o["s"] > c.s[0]:
            hi = c.p[0]
        elif o["s"] < c.s[-1]:
            lo = c.p[-1]
        else:
            for i in range(c.n - 1):
                if c.s[i] >= o["s"] >= c.s[i + 1]:
                    lo, hi = c.p[i], c.p[i + 1]
                    break
        if lo is None and hi is None:
            continue
        cl = None if lo is None else math.log10(max(lo, o["x"]) / o["x"])
        ch = None if hi is None else math.log10(hi / o["x"]) if hi >= o["x"] else 0.0
        units.append({"server": o["server"], "person": o["person"], "day": int(o["h"] // 24) + 1, "h": o["h"],
                      "P": None, "x": o["x"], "grade": o["grade"], "n": 1, "cens": [cl, ch]})
        have.add((o["server"], o["person"], int(o["h"] // 24)))
    return season, H, obs, units


def unit_resid(u, p):
    """값 칸은 P − 모형, 구간 칸은 구간 밖으로 벗어난 만큼 (안이면 0)."""
    if u["cens"] is None:
        return u["P"] - p
    lo, hi = u["cens"]
    if lo is not None and p < lo:
        return lo - p
    if hi is not None and p > hi:
        return hi - p
    return 0.0


# ---------------- % 변환 적합 ----------------

def fit_form(form, U, H, x0=None, bx=False, sig=None):
    """U: 한섭 칸. 반환 (theta, bx). Huber(척도 0.04 또는 σ(h)) 손실."""
    start = list(x0) if x0 else list(FORMS[form]) + ([0.0] if bx else [])
    if bx and len(start) == NPAR[form]:
        start = start + [0.0]

    def loss(th):
        tr = {"form": form, "theta": th[:NPAR[form]], "bx": th[NPAR[form]] if bx else 0.0, "m": {}}
        tot = 0.0
        for u in U:
            try:
                p = PR.p_full(tr, "KR", u["h"], u["x"], H)
            except (OverflowError, ValueError):
                return 1e9
            s = sig(u["h"], p) if sig else 0.04
            tot += huber(unit_resid(u, p) / s, 1.345)
        return tot
    best, val = nelder_mead(loss, start, iters=600)
    best, val = nelder_mead(loss, best, step=0.05, iters=400)
    return best[:NPAR[form]], (best[NPAR[form]] if bx else 0.0)


def fit_m(U, tr, H):
    if len(U) < 5:
        return 1.0, "borrow_KR"
    best, _ = nelder_mead(lambda v: sum(huber(unit_resid(u, math.exp(v[0]) * PR.p_full(tr, "KR", u["h"], u["x"], H)), 0.05)
                                        for u in U), [0.0], step=0.1)
    return math.exp(best[0]), "own_level"


def fit_sigma(res):
    """res: [(h, P모형, 잔차)]. 일차 칸 1.4826·MAD에 σ = √(s0² + (ρP)²)를 맞춘다."""
    bins = collections.defaultdict(list)
    for h, p, r in res:
        bins[int(h // 24)].append((p, r))
    pts = []
    for d, L in bins.items():
        if len(L) >= 3:
            pts.append((statistics.median(p for p, _ in L), mad([r for _, r in L]), len(L)))
    if len(pts) < 2:
        m = mad([r for _, _, r in res]) or 0.05
        return max(m, 0.01), 0.0

    def loss(v):
        s0, rho = math.exp(v[0]), math.exp(v[1])
        return sum(n * (math.log(max(sd, 1e-4)) - 0.5 * math.log(s0 ** 2 + (rho * p) ** 2)) ** 2 for p, sd, n in pts)
    best, _ = nelder_mead(loss, [math.log(0.03), math.log(0.1)], step=0.5)
    return max(math.exp(best[0]), 0.005), math.exp(best[1])


def fit_transform(units, H, form=None, use_bx=None, x0=None):
    """한 번 맞추기: 꼴(form이 None이면 이미 정한 것), θ·bx, m_JP, σ. 반환 dict."""
    KR = [u for u in units if u["server"] == "KR"]
    JP = [u for u in units if u["server"] == "JP"]
    th, bx = fit_form(form, KR, H, x0=x0, bx=use_bx)
    tr = {"form": form, "theta": th, "bx": bx, "m": {"KR": 1.0}}
    m_jp, m_src = fit_m(JP, tr, H)
    tr["m"] = {"KR": 1.0, "JP": m_jp, "GL": 1.0}
    res = []
    for u in KR + JP:
        if u["cens"] is None:
            p = PR.p_full(tr, u["server"], u["h"], u["x"], H)
            res.append((u["h"], p, u["P"] - p))
    s0, rho = fit_sigma(res)
    # 한 번 더: σ(h) 가중으로 θ
    sig = lambda h, p: math.sqrt(s0 ** 2 + (rho * p) ** 2)
    th, bx = fit_form(form, KR, H, x0=list(th) + ([bx] if use_bx else []), bx=use_bx, sig=sig)
    tr = {"form": form, "theta": th, "bx": bx, "m": {"KR": 1.0}}
    m_jp, m_src = fit_m(JP, tr, H)
    tr["m"] = {"KR": 1.0, "JP": m_jp, "GL": 1.0}
    res = []
    for u in KR + JP:
        if u["cens"] is None:
            p = PR.p_full(tr, u["server"], u["h"], u["x"], H)
            res.append((u["h"], p, u["P"] - p))
    s0, rho = fit_sigma(res)
    tr.update({"s0": s0, "rho": rho, "m_src": {"KR": "own", "JP": m_src, "GL": "borrow_KR"}})
    return tr, res


def lopo(form, KR, H, bx=False):
    full_th, full_bx = fit_form(form, KR, H, bx=bx)
    x0 = list(full_th) + ([full_bx] if bx else [])
    errs, cerrs = [], []
    for p in sorted({u["person"] for u in KR}):
        tr_u = [u for u in KR if u["person"] != p]
        th, b = fit_form(form, tr_u, H, x0=x0, bx=bx)
        tr = {"form": form, "theta": th, "bx": b, "m": {}}
        for u in KR:
            if u["person"] == p:
                e = unit_resid(u, PR.p_full(tr, "KR", u["h"], u["x"], H))
                (errs if u["cens"] is None else cerrs).append((u["h"], e))
    ab = [abs(e) for _, e in errs]
    early = [abs(e) for h, e in errs if h < 36]
    return {"n": len(errs), "med_abs": statistics.median(ab), "mean_abs": sum(ab) / len(ab),
            "med_abs_h_lt36": statistics.median(early) if early else None, "n_h_lt36": len(early),
            "cens_n": len(cerrs), "cens_outside": sum(1 for _, e in cerrs if abs(e) > 0.02),
            "cens_mean_abs": (sum(abs(e) for _, e in cerrs) / len(cerrs)) if cerrs else None,
            "theta_full": full_th, "bx_full": full_bx}


def day_medians(units, H):
    out = {}
    for s in ("KR", "JP"):
        v = []
        for d in range(1, int(math.ceil(H / 24)) + 1):
            L = [u["P"] for u in units if u["server"] == s and u["day"] == d and u["cens"] is None]
            v.append(statistics.median(L) if L else None)
        out[s] = v
    kr = out["KR"]
    # 빈 일차는 이웃으로 채운다 (앞 일차가 비면 뒤 값)
    for i in range(len(kr)):
        if kr[i] is None:
            nb = [kr[j] for j in range(len(kr)) if kr[j] is not None]
            kr[i] = nb[0] if i == 0 else (kr[i - 1] if kr[i - 1] is not None else nb[0])
    # 일섭 배수: 같은 일차 중앙값 비의 중앙
    rat = [j / k for j, k in zip(out["JP"], kr) if j is not None and k]
    return {"KR": kr, "JP_raw": out["JP"]}, {"KR": 1.0, "JP": statistics.median(rat) if rat else 1.0, "GL": 1.0}


# ---------------- 모양 ----------------

def pav_dec(xs, ys):
    return E.pav_decreasing(xs, ys)


def own_points(state, srv, grades):
    pts = E.final_points(state, srv, grades)
    if not pts:
        return []
    ys = pav_dec([p for p, _ in pts], [s for _, s in pts])
    return [(p, s) for (p, _), s in zip(pts, ys)]


def seg_ok(a, b):
    return not (b / a > 1.5 and b - a > 0.5)


def lin(u, u0, u1, s0, s1):
    return s0 if u1 == u0 else s0 + (u - u0) / (u1 - u0) * (s1 - s0)


def kr_curve(pts):
    """한섭 s(u)·flag 함수."""
    U = [math.log(p) for p, _ in pts]
    S = [s for _, s in pts]

    def f(u):
        if u < U[0]:
            return lin(u, U[0], U[1], S[0], S[1]), "extrap"
        if u > U[-1]:
            return lin(u, U[-2], U[-1], S[-2], S[-1]), "extrap"
        for i in range(len(U) - 1):
            if U[i] <= u <= U[i + 1]:
                fl = "own" if seg_ok(pts[i][0], pts[i + 1][0]) else "own_gap"
                return lin(u, U[i], U[i + 1], S[i], S[i + 1]), fl
        return S[-1], "extrap"
    return f


def other_curve(pts, fkr):
    U = [math.log(p) for p, _ in pts]
    S = [s for _, s in pts]

    def f(u):
        if not pts:
            return fkr(u)[0], "KR_ratio"
        if u < U[0] or u > U[-1] or len(pts) == 1:
            e = 0 if (u < U[0] or len(pts) == 1) else len(U) - 1
            return S[e] * fkr(u)[0] / fkr(U[e])[0], "KR_ratio"
        for i in range(len(U) - 1):
            if U[i] <= u <= U[i + 1]:
                if seg_ok(pts[i][0], pts[i + 1][0]):
                    return lin(u, U[i], U[i + 1], S[i], S[i + 1]), "own"
                ka, kb, ku = fkr(U[i])[0], fkr(U[i + 1])[0], fkr(u)[0]
                t = (ku - ka) / (kb - ka) if kb != ka else (u - U[i]) / (U[i + 1] - U[i])
                return S[i] + t * (S[i + 1] - S[i]), "KR_warp"
        return S[-1], "KR_ratio"
    return f


def build_shape(state, grades, table):
    grid = []
    u = GRID_LO
    while u <= GRID_HI + 1e-9:
        grid.append(round(u, 6))
        u += GRID_DU
    pts = {s: own_points(state, s, grades) for s in PR.SERVERS}
    fkr = kr_curve(pts["KR"])
    fs = {"KR": fkr, "JP": other_curve(pts["JP"], fkr), "GL": other_curve(pts["GL"], fkr)}
    u3 = math.log(3.0)
    out, s3 = {}, {}
    for s in PR.SERVERS:
        s3[s] = fs[s](u3)[0]
        vals = [fs[s](u) for u in grid]
        lr = [math.log(max(v, 0.005 * s3[s]) / s3[s]) for v, _ in vals]
        # 단조: ln r가 u에 대해 줄어들게 (꺾은선이라 대개 이미 그렇다)
        for i in range(1, len(lr)):
            lr[i] = min(lr[i], lr[i - 1])
        out[s] = {"lnp": grid, "lr": lr, "flag": [fl for _, fl in vals]}
    # D(u): 일섭 − 한섭 (3%에서 0)
    lrK = lambda u: math.log(fs["KR"](u)[0] / s3["KR"])
    lrJ = lambda u: math.log(fs["JP"](u)[0] / s3["JP"])
    anchors = [(u3, 0.0, "norm")]
    for p, _ in pts["JP"]:
        u = math.log(p)
        if abs(u - u3) > 0.05:
            anchors.append((u, lrJ(u) - lrK(u), "JP_point"))
    tb = {(r["server"], r["pct"]): r["score_eok"] for r in table.get("table", {}).get("rows", [])}
    for q in (10, 30):
        if ("JP", q) in tb and ("KR", q) in tb and ("JP", 3) in tb and ("KR", 3) in tb:
            u = math.log(q)
            if not any(abs(a[0] - u) < 0.1 for a in anchors):
                anchors.append((u, math.log(tb[("JP", q)] / tb[("JP", 3)]) - math.log(tb[("KR", q)] / tb[("KR", 3)]),
                                "B_table"))
    anchors.sort()
    # 같은 u 근처 닻은 평균
    A = []
    for a in anchors:
        if A and abs(a[0] - A[-1][0]) < 0.03:
            A[-1] = ((A[-1][0] + a[0]) / 2, (A[-1][1] + a[1]) / 2, A[-1][2] + "+" + a[2])
        else:
            A.append(a)

    def Dfun(u):
        if len(A) == 1:
            return 0.0
        if u <= A[0][0]:
            return lin(u, A[0][0], A[1][0], A[0][1], A[1][1])
        if u >= A[-1][0]:
            return lin(u, A[-2][0], A[-1][0], A[-2][1], A[-1][1])
        for i in range(len(A) - 1):
            if A[i][0] <= u <= A[i + 1][0]:
                return lin(u, A[i][0], A[i + 1][0], A[i][1], A[i + 1][1])
    D = [Dfun(u) for u in grid]
    for s in PR.SERVERS:
        out[s]["D"] = D
    return {"servers": out, "s3_S41": s3, "own_points": {s: pts[s] for s in PR.SERVERS},
            "D_anchors": [[math.exp(u), d, src] for u, d, src in A]}


# ---------------- 본체 ----------------

def fit(state, grades=("A", "A2"), fit_grades="all", boot=200, seed=1, persons=None, form=None, quiet=False):
    """prior dict. persons가 있으면 그 사람의 관측만으로 % 변환을 맞춘다 (backtest 교차 적합). form을 주면 꼴 고르기를 건너뛴다."""
    season, H, obs, units = load_units(state, grades, fit_grades, persons)
    sname = season["name"]
    KR = [u for u in units if u["server"] == "KR"]
    cand = {}
    if form is None:
        for f in FORMS:
            cand[f] = lopo(f, KR, H)
        best = min(cand, key=lambda f: cand[f]["med_abs"])
        # 5% 안이면 모수 적은 쪽
        for f in sorted(FORMS, key=lambda f: NPAR[f]):
            if cand[f]["med_abs"] <= 1.05 * cand[best]["med_abs"]:
                best = f
                break
        lx = lopo(best, KR, H, bx=True)
        cand[best + "+x"] = lx
        use_bx = lx["med_abs"] < 0.95 * cand[best]["med_abs"]
        form = best
    else:
        use_bx = False
        if isinstance(form, tuple):
            form, use_bx = form
    tr, res = fit_transform(units, H, form, use_bx)
    # 꼴 외삽 불확실성: 가장 이른 칸보다 앞에서는 세 꼴의 차이를 (ii)에 넣는다
    alt = {f: fit_form(f, KR, H)[0] for f in FORMS}
    hk = sorted(u["h"] for u in KR if u["cens"] is None)   # 값 칸만 (구간 칸은 받침으로 세지 않는다)
    h_support = {"h_min": hk[0] if hk else H, "h_ramp": hk[min(4, len(hk) - 1)] if hk else H,
                 "note": "h ≤ h_min(가장 이른 값 칸)은 세 꼴 P 차이(최대−최소) 전체를 1σ로, h_min~h_ramp(다섯째 값 칸)는 선형으로 줄여 0. "
                         "2026-10-16 절반 → 전체로 넓힘: S41 h=4.5 구간 관측 하나가 모형보다 0.44~0.72(log10) 높았다 (한 점이라 맞춘 값 아님)"}
    # 표준화 잔차
    z = sorted(r / math.sqrt(tr["s0"] ** 2 + (tr["rho"] * p) ** 2) for h, p, r in res)
    # 붓스트랩 (서버마다 사람을 다시 뽑는다)
    rng = random.Random(seed)
    byp = collections.defaultdict(list)
    for u in units:
        byp[(u["server"], u["person"])].append(u)
    keys = {s: [k for k in byp if k[0] == s] for s in PR.SERVERS}
    reps = []
    x0 = list(tr["theta"]) + ([tr["bx"]] if use_bx else [])
    for b in range(boot):
        U = []
        for s in ("KR", "JP"):
            ks = keys[s]
            for _ in range(len(ks)):
                U += byp[ks[rng.randrange(len(ks))]]
        t2, _ = fit_transform(U, H, form, use_bx, x0=x0)
        reps.append({"theta": t2["theta"], "bx": t2["bx"], "m": t2["m"], "s0": t2["s0"], "rho": t2["rho"]})
    dm, dm_m = day_medians(units, H)
    # 같은 점수 쌍 잡음 (정답 없이 쓰는 P 수준 갱신용) — 이 시즌 자체 쌍으로 잰다
    _, c0, H2 = PR.season_info(state)
    rows, _ = PR.load_obs(state, H, H, c0)
    if persons is not None:
        rows = [o for o in rows if o["person"] in persons]
    pairs = PR.same_score_pairs(rows, H, tr)
    pr_res = [p["d_obs"] - p["d_model"] for p in pairs]
    pair_sigma = max(mad(pr_res) or 0.05, 0.02)
    post = PR.p_level_posterior(pairs, 1.0, pair_sigma) if pairs else None
    # C 점검: C 칸 잔차 중앙 vs A·B
    cres = collections.defaultdict(list)
    for u in units:
        if u["server"] in ("KR", "JP") and u["cens"] is None:
            p = PR.p_full(tr, u["server"], u["h"], u["x"], H)
            cres["C" if u["grade"] == "C" else "AB"].append(u["P"] - p)
    table = {}
    tp = os.path.join(state, "truth_table.json")
    if os.path.exists(tp):
        with open(tp, encoding="utf-8") as f:
            table = json.load(f)
    shape = build_shape(state, grades, table)
    lab = LABEL.format(season=sname)
    # 시즌 간 차이 기본값 (가정)
    lm = [math.log(r["m"]["JP"]) for r in reps]
    tau_p_proxy = abs(math.log(tr["m"]["JP"]))
    tau_p_sd = statistics.pstdev(lm) if len(lm) > 1 else 0.0
    tau_p = round(max(0.05, math.sqrt(tau_p_proxy ** 2 + tau_p_sd ** 2)), 3)
    s3 = shape["s3_S41"]
    r_jp, r_gl = s3["JP"] / s3["KR"], s3["GL"] / s3["KR"]
    tau_ratio = round(max(0.05, abs(math.log(r_jp) - math.log(r_gl))), 3)
    sday = collections.defaultdict(list)
    for h, p, r in res:
        sday[int(h // 24) + 1].append(r)
    prior = {
        "season": sname, "generated_by": "tools/solo_raid/fit_prior.py", "H": H,
        "inputs": {"state": state, "curve_grades": list(grades), "fit_grades": fit_grades, "boot": boot, "seed": seed,
                   "persons_subset": None if persons is None else len(persons)},
        "basis": lab,
        "transform": {
            "basis": lab, "form": form, "theta": tr["theta"], "bx": tr["bx"], "m": tr["m"], "m_src": tr["m_src"],
            "m_note": "일섭은 한섭 꼴에 수준만 따로 맞췄다 (칸 5개 미만이면 1). 글섭은 표본이 없어 한섭을 그대로 빌린다 (가정).",
            "sigma": {"s0": tr["s0"], "rho": tr["rho"], "form": "sigma(h) = sqrt(s0^2 + (rho*P(h))^2), log10 단위"},
            "sigma_by_day": {d: {"n": len(v), "mad": mad(v)} for d, v in sorted(sday.items())},
            "z_resid": [round(v, 4) for v in z],
            "boot": reps,
            "alt_forms": alt, "h_support": h_support,
            "candidates": cand, "selection": "한섭 사람·일차 칸 LOPO |잔차| 중앙 최소 (5% 안이면 모수 적은 꼴), x 항은 5% 넘게 좋아질 때만",
            "n_units": {s: sum(1 for u in units if u["server"] == s and u["cens"] is None) for s in PR.SERVERS},
            "n_units_interval": {s: sum(1 for u in units if u["server"] == s and u["cens"] is not None) for s in PR.SERVERS},
            "n_persons": {s: len({u["person"] for u in units if u["server"] == s}) for s in PR.SERVERS},
            "n_obs": {s: sum(1 for o in obs if o["server"] == s) for s in PR.SERVERS},
            "units_by_day": {s: dict(collections.Counter(u["day"] for u in units if u["server"] == s)) for s in ("KR", "JP")},
            "day_median": dm, "day_median_m": dm_m,
            "pair_sigma": pair_sigma, "n_pairs": len(pairs),
            "pair_level_check": None if not post else {"m_mean": math.exp(post["mean_lnm"]), "sd_lnm": post["sd_lnm"],
                                                         "note": "이 시즌 자체 쌍으로 잰 P 수준 배수 (1이면 단면 P와 같은 수준)"},
            "c_check": {k: {"n": len(v), "median_resid": statistics.median(v) if v else None} for k, v in cres.items()},
        },
        "shape": dict(shape, basis=lab,
                      flags_note="own = 그 서버 정답 점 덮인 선분 · own_gap = 빈 구간 꺾은선 (가정) · extrap = 끝 밖 연장 (가정) · "
                                 "KR_warp = 한섭 모양을 양 끝에 맞춤 (가정) · KR_ratio = 끝 밖 한섭 비율 (가정)"),
        "season_diff": {
            "basis": "시즌 간 차이는 한 시즌으로 잴 수 없다 — 아래는 모두 가정이다. 다른 시즌을 모으면 교체한다.",
            "tau_p": {"value": tau_p, "basis": f"{lab}의 일섭–한섭 P 수준 차이 |ln m_JP| = {tau_p_proxy:.3f}와 그 붓스트랩 sd {tau_p_sd:.3f}의 합 "
                                               "(서버 간 차이를 시즌 간 차이 크기로 빌림, 바닥 0.05)"},
            "tau_shape": {"value": 1.0, "basis": "D(u) = S41 일섭–한섭 모양 차이를 그대로 1σ로 둔다 (근거 없는 가정 — 시즌 간 모양 차이는 "
                                               "S41로 잴 수 없어 보수적인 쪽을 골랐다. 2026-10-16 0.5 → 1: S41 tr 점검에서 1·3% 폭은 거의 그대로이고 "
                                               "10%·30% 폭만 넓어진다. 한섭 모양을 일섭 모양으로 바꾼 시험(10% −8~−10%)이 0.5면 2σ, 1이면 1σ)"},
            "tau_ratio": {"value": tau_ratio, "basis": f"{lab}의 일섭/한섭({r_jp:.3f})과 글섭/한섭({r_gl:.3f}) 3% 비율의 ln 차"},
            "tau_scale0": {"value": 0.3, "basis": "관측이 하나도 없을 때 규모 차이 (ln 1σ). 근거 없음 — S41 하나로는 시즌 간 규모 차이를 잴 수 없다"},
        },
    }
    if not quiet:
        report(prior)
    return prior


def report(pr):
    t = pr["transform"]
    print(f"prior {pr['season']} · H={pr['H']} · 곡선 등급 {pr['inputs']['curve_grades']} · 맞춘 시각 등급 {pr['inputs']['fit_grades']}")
    print(f"칸(사람·일차): {t['n_units']} · 구간 칸 {t['n_units_interval']} · 사람 {t['n_persons']} · 관측 줄 {t['n_obs']} · 한섭 일차별 {t['units_by_day']['KR']}")
    for f, c in t["candidates"].items():
        e = c["med_abs_h_lt36"]
        print(f"  꼴 {f:8s} LOPO |잔차| 중앙 {c['med_abs']:.4f} 평균 {c['mean_abs']:.4f} · h<36 중앙 {e if e is None else round(e, 4)} (n {c['n_h_lt36']}) · "
              f"구간 칸 {c.get('cens_n')} 중 벗어남(>0.02) {c.get('cens_outside')} 평균 벗어난 정도 {c.get('cens_mean_abs') and round(c['cens_mean_abs'], 4)}")
    print(f"선택: {t['form']} θ={[round(v, 4) for v in t['theta']]} bx={t['bx']:.3f} m={ {k: round(v, 3) for k, v in t['m'].items()} } "
          f"σ: s0={t['sigma']['s0']:.4f} ρ={t['sigma']['rho']:.3f}")
    H = pr["H"]
    tr = {"form": t["form"], "theta": t["theta"], "bx": t["bx"], "m": t["m"]}
    hs = (1, 3, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 114, 119)
    print("  P(h) [80% 붓스트랩]: " + " · ".join(
        f"h{h}: {PR.p_full(tr, 'KR', h, 1.0, H):.3f}[{PR.quant([PR.p_full(dict(tr, theta=b['theta'], bx=b['bx']), 'KR', h, 1.0, H) for b in t['boot']], .1):.3f},"
        f"{PR.quant([PR.p_full(dict(tr, theta=b['theta'], bx=b['bx']), 'KR', h, 1.0, H) for b in t['boot']], .9):.3f}]" for h in hs))
    print(f"  꼴 외삽 구간: h < {t['h_support']['h_min']:.1f} (줄여 0: {t['h_support']['h_ramp']:.1f}) · σ_form: " + " · ".join(
        f"h{h}: {PR.sigma_form(t, h, H):.3f}" for h in (1, 3, 6, 9, 12, 15, 18)))
    print(f"  일차 중앙 P(기준선 1): {[None if v is None else round(v, 3) for v in t['day_median']['KR']]} · 일섭 배수 {t['day_median_m']['JP']:.3f}")
    print(f"  σ 일차별 MAD: { {d: (v['n'], None if v['mad'] is None else round(v['mad'], 3)) for d, v in t['sigma_by_day'].items()} }")
    print(f"  같은 점수 쌍 {t['n_pairs']}개 · 쌍 잡음 {t['pair_sigma']:.4f} · 쌍으로 잰 수준 배수 {t['pair_level_check']}")
    print(f"  C 점검: {t['c_check']}")
    sh = pr["shape"]
    print(f"모양: s(3%) S41 { {k: round(v, 2) for k, v in sh['s3_S41'].items()} } · D 닻 {[(round(p, 2), round(d, 3), s) for p, d, s in sh['D_anchors']]}")
    for s in PR.SERVERS:
        S = PR.Shape(sh["servers"][s])
        print(f"  {s}: " + " · ".join(f"{q}%: r={math.exp(S.lr_at(math.log(q))):.3f} k={S.k_at(math.log(q)):.3f} D={S.D_at(math.log(q)):+.3f} {S.flag_at(math.log(q))}"
                                      for q in (0.3, 0.5, 1, 2, 3, 5, 10, 15, 30)))
    sd = pr["season_diff"]
    print("시즌 간 차이 (가정): " + " · ".join(f"{k}={v['value']}" for k, v in sd.items() if isinstance(v, dict)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grades", default="all", choices=["A", "all"])
    ap.add_argument("--fit-grades", default="all", choices=["AB", "all"])
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    split.add_final_arg(ap)
    a = ap.parse_args()
    split.guard_args(a)   # 시험 시즌 정답 감시 (할 일 3.0b)
    grades = ("A",) if a.grades == "A" else ("A", "A2")
    pr = fit(a.state, grades, a.fit_grades, a.boot, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(PR.jsonable(pr), f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
