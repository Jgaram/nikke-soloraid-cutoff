"""솔로레이드 컷 예측의 사전 정보(prior)를 끝난 시즌 여러 개로 만든다 (할 일 3.5a, 2026-10-16, 모델 0, 요청 0).

python3 -I tools/solo_raid/fit_prior_multi.py S38,S40,S41 --out PRIOR.json [--boot 100] [--seed 1] [--tiers full,shape_only]

predict.py가 그대로 읽는 prior.json과 같은 꼴을 낸다 (fit_prior.py 한 시즌 버전의 여러 시즌판). 시즌 몫·학습 등급은
data/solo-raid/registry.json(registry.py --write)을 따른다 — 시험 시즌은 넣지 않는다(split 감시, --final 없으면 정답을 못 연다).

들어가는 것 (시즌마다 같은 무게 — 한 시즌의 관측 수백 개는 한 곡면의 측정점이다)
(a) % 변환 P(h) = log10(최종 %/지금 %)
    - 시즌마다 fit_prior.load_units(그 시즌 정답 곡선으로 P를 붙인 사람·일차 칸)를 모은다. 칸마다 그 시즌 길이 H를 쓴다 (S38 113시간).
    - 꼴(sqrt·pow·f2pow)은 시즌 하나씩 빼기(LOSO) |잔차| 중앙으로 고른다 (5% 안이면 모수 적은 꼴). 시즌이 2개 미만이면 sqrt.
    - 시즌 수준 m_s: 공통 꼴에 시즌마다 배수 하나 (기하평균 1로 맞춤). 꼴은 칸을 m_s로 나눈 뒤 다시 맞춘다 (두 번 돈다).
    - τ_P = 시즌 간 ln m_s 퍼짐: √max(분산 − 평균 표본오차², 0) (표본오차는 시즌 안 사람 붓스트랩). 바닥 0.05.
    - σ(h)·잔차 z·모수 붓스트랩(시즌 안 사람 다시 뽑기)·기준선 1 일차 중앙(시즌 일차 중앙의 중앙)·일섭 배수 m_JP.
(b) 곡선 모양 ln r(u) = ln(s(%)/s(3%))
    - 시즌마다 fit_prior.build_shape. 격자마다 시즌 평균, 무게는 그 시즌 flag: own 1 · own_gap/KR_warp 0.3 · extrap/KR_ratio 0.05.
    - D(u) = 시즌 간 퍼짐: 격자마다 무게 0.3 이상인 시즌의 표준편차(n−1), 부호는 공통으로 덮인 구간 첫 주성분의 부호 (3%에서 0).
      τ_shape = 1 (D가 이미 1σ). 시즌 1개면 fit_prior처럼 일섭–한섭 차이를 쓴다.
    - s(3%) 기준값 s3: 시즌 s3의 기하평균 (관측이 하나도 없을 때만 쓴다).
(c) season_diff: τ_P(위), τ_shape 1, τ_ratio = 학습 시즌 집계표의 ln(일섭 3%/한섭 3%)·ln(글섭/한섭) 퍼짐,
    τ_scale0 = 학습 시즌 한섭 3% 컷(집계표, 없으면 곡선) ln 표준편차. 시즌이 적으면 바닥을 둔다 (값과 근거를 남긴다).
**시즌 수가 적을 때(2026-10-16 개발 시즌 넷) 퍼짐 값은 거칠다.** 구간 보정은 3.6 검증에서 따로 정한다.
"""
import argparse
import collections
import importlib.util
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import predict as PR  # noqa: E402
import fit_prior as FP  # noqa: E402
import split as SP  # noqa: E402

E = FP.E
DATA = os.path.join(ROOT, "data", "solo-raid")
FORMS = FP.FORMS
NPAR = FP.NPAR
W_FLAG = {"own": 1.0, "own_gap": 0.3, "KR_warp": 0.3, "extrap": 0.05, "KR_ratio": 0.05}
TAU_P_FLOOR = 0.05
TAU_RATIO_FLOOR = 0.05
TAU_SCALE0_FLOOR = 0.10


def registry():
    with open(os.path.join(DATA, "registry.json"), encoding="utf-8") as f:
        return {r["season"]: r for r in json.load(f)["seasons"]}


def state_of(name):
    return os.path.join(DATA, name, "state")


# ---------------- % 변환 (여러 시즌) ----------------

def p_unit(tr, u, m=1.0):
    return PR.p_full(tr, u["server"], u["h"], u["x"], u["H"], m)


def wloss(form, th, bx, U, sig=None, lvl=None):
    tr = {"form": form, "theta": th, "bx": bx, "m": {}}
    tot = 0.0
    for u in U:
        try:
            p = PR.p_full(tr, "KR", u["h"], u["x"], u["H"], (lvl or {}).get(u["season"], 1.0))
        except (OverflowError, ValueError):
            return 1e9
        s = sig(u["h"], p) if sig else 0.04
        tot += u["w"] * FP.huber(FP.unit_resid(u, p) / s, 1.345)
    return tot


def wfit_form(form, U, x0=None, sig=None, lvl=None):
    start = list(x0) if x0 else list(FORMS[form])
    f = lambda th: wloss(form, th[:NPAR[form]], 0.0, U, sig, lvl)
    best, _ = FP.nelder_mead(f, start, iters=600)
    best, _ = FP.nelder_mead(f, best, step=0.05, iters=400)
    return best[:NPAR[form]]


def fit_levels(form, th, U, seasons):
    """시즌마다 배수 m_s (한섭 칸). 기하평균 1로 맞춘다."""
    tr = {"form": form, "theta": th, "bx": 0.0, "m": {}}
    lv = {}
    for s in seasons:
        Us = [u for u in U if u["season"] == s]
        if len(Us) < 3:
            lv[s] = 1.0
            continue
        best, _ = FP.nelder_mead(lambda v: sum(FP.huber(FP.unit_resid(u, math.exp(v[0]) * p_unit(tr, u)), 0.05) for u in Us),
                                 [0.0], step=0.1)
        lv[s] = math.exp(best[0])
    g = math.exp(statistics.mean(math.log(v) for v in lv.values())) if lv else 1.0
    return {s: v / g for s, v in lv.items()}, g


def fit_pooled(form, U, seasons, x0=None):
    """공통 꼴 θ와 시즌 배수 (두 번 돈다), σ."""
    lvl = {s: 1.0 for s in seasons}
    th = wfit_form(form, U, x0=x0)
    for _ in range(2):
        lvl, g = fit_levels(form, th, U, seasons)
        th = wfit_form(form, U, x0=th, lvl=lvl)
    tr = {"form": form, "theta": th, "bx": 0.0, "m": {"KR": 1.0}}
    res = [(u["h"], p, u["P"] - p) for u in U if u["cens"] is None
           for p in [p_unit(tr, u, lvl.get(u["season"], 1.0))]]
    s0, rho = FP.fit_sigma(res)
    sig = lambda h, p: math.sqrt(s0 ** 2 + (rho * p) ** 2)
    th = wfit_form(form, U, x0=th, sig=sig, lvl=lvl)
    lvl, _ = fit_levels(form, th, U, seasons)
    tr = {"form": form, "theta": th, "bx": 0.0, "m": {"KR": 1.0}}
    res = [(u["h"], p, u["P"] - p) for u in U if u["cens"] is None
           for p in [p_unit(tr, u, lvl.get(u["season"], 1.0))]]
    s0, rho = FP.fit_sigma(res)
    return th, lvl, s0, rho, res


def loso(form, U, seasons):
    """시즌 하나씩 빼고 맞춘 공통 꼴(수준 1)로 뺀 시즌 칸의 |잔차| — 시즌마다 중앙, 그 평균."""
    per = {}
    for s in seasons:
        tr_u = [u for u in U if u["season"] != s]
        te = [u for u in U if u["season"] == s and u["cens"] is None]
        if not te or not tr_u:
            continue
        th = wfit_form(form, tr_u)
        tr = {"form": form, "theta": th, "bx": 0.0, "m": {}}
        per[s] = statistics.median(abs(u["P"] - p_unit(tr, u)) for u in te)
    return {"per_season": per, "mean": statistics.mean(per.values()) if per else None}


def fit_m_jp(UJ, tr, lvl):
    if len(UJ) < 5:
        return 1.0, "borrow_KR"
    best, _ = FP.nelder_mead(lambda v: sum(u["w"] * FP.huber(FP.unit_resid(u, math.exp(v[0]) * p_unit(tr, u, lvl.get(u["season"], 1.0))), 0.05)
                                           for u in UJ), [0.0], step=0.1)
    return math.exp(best[0]), "own_level (여러 시즌)"


# ---------------- 모양 (여러 시즌) ----------------

def shape_multi(shapes):
    """shapes: {시즌: fit_prior.build_shape 결과} → prior["shape"] 꼴."""
    names = sorted(shapes)
    grid = shapes[names[0]]["servers"]["KR"]["lnp"]
    out, Dinfo = {}, {}
    for srv in PR.SERVERS:
        lr_m, fl_m, D = [], [], []
        L = {s: shapes[s]["servers"][srv] for s in names}
        for gi in range(len(grid)):
            ws = [(W_FLAG.get(L[s]["flag"][gi], 0.05), L[s]["lr"][gi], L[s]["flag"][gi]) for s in names]
            tw = sum(w for w, _, _ in ws)
            lr_m.append(sum(w * v for w, v, _ in ws) / tw)
            best = max(ws, key=lambda t: t[0])
            fl_m.append(best[2] if len(names) == 1 else
                        ("own" if sum(1 for w, _, _ in ws if w >= 1.0) >= 2 else best[2]))
        # 단조
        for i in range(1, len(lr_m)):
            lr_m[i] = min(lr_m[i], lr_m[i - 1])
        # 퍼짐
        sd = []
        for gi in range(len(grid)):
            v = [L[s]["lr"][gi] for s in names if W_FLAG.get(L[s]["flag"][gi], 0.05) >= 0.3]
            sd.append(statistics.stdev(v) if len(v) >= 2 else None)
        # 부호: 공통 덮인 구간(모든 시즌 무게 ≥ 0.3)의 첫 주성분
        common = [gi for gi in range(len(grid)) if all(W_FLAG.get(L[s]["flag"][gi], 0.05) >= 0.3 for s in names)]
        sign = [1.0] * len(grid)
        if len(names) >= 2 and len(common) >= 3:
            dev = [[L[s]["lr"][gi] - statistics.mean(L[t]["lr"][gi] for t in names) for gi in common] for s in names]
            v = [1.0] * len(common)
            for _ in range(100):   # 거듭제곱법 (공분산 = devᵀ dev)
                w = [sum(dev[k][i] * sum(dev[k][j] * v[j] for j in range(len(common))) for k in range(len(names)))
                     for i in range(len(common))]
                n = math.sqrt(sum(x * x for x in w)) or 1.0
                v = [x / n for x in w]
            for gi in range(len(grid)):
                near = min(range(len(common)), key=lambda k: abs(common[k] - gi))
                sign[gi] = 1.0 if v[near] >= 0 else -1.0
            u3 = math.log(3.0)
            for gi, u in enumerate(grid):     # 3% 아래·위 부호를 공통 구간 밖으로 잇는다
                if gi < common[0]:
                    sign[gi] = sign[common[0]] if u < u3 else sign[gi]
                if gi > common[-1]:
                    sign[gi] = sign[common[-1]]
        filled = [x for x in sd if x is not None]
        for gi in range(len(grid)):
            if sd[gi] is None:      # 퍼짐을 잴 시즌이 없으면 가장 가까운 잰 값
                nb = [k for k in range(len(grid)) if sd[k] is not None]
                sd[gi] = sd[min(nb, key=lambda k: abs(k - gi))] if nb else 0.0
            D.append(sign[gi] * sd[gi])
        out[srv] = {"lnp": grid, "lr": lr_m, "flag": fl_m, "D": D}
        Dinfo[srv] = {"n_common": len(common), "sd_at": {f"{q}%": round(sd[min(range(len(grid)), key=lambda k: abs(grid[k] - math.log(q)))], 4)
                                                          for q in (0.5, 1, 2, 5, 10, 30)}, "measured_points": len(filled)}
    s3 = {srv: math.exp(statistics.mean(math.log(shapes[s]["s3_S41"][srv]) for s in names)) for srv in PR.SERVERS}
    return {"servers": out, "s3_S41": s3, "s3_basis": f"시즌 {','.join(names)} s(3%)의 기하평균 (키 이름은 predict 호환)",
            "D_info": Dinfo, "own_points": {s: shapes[s]["own_points"] for s in names}}


# ---------------- 집계표 퍼짐 ----------------

def table_spread(names, final=False):
    sp = SP.load_split()
    rows = {SP.snum(s["name"]): s for s in SP.load_seasons(final=final, sp=sp)}
    cuts = collections.defaultdict(dict)
    for r in SP.load_server_cutoffs(final=final, sp=sp):
        try:
            cuts[SP.snum(r["season"])][(r["server"], float(r["percentile"]))] = float(r["score_100m"])
        except (ValueError, TypeError):
            pass
    kr3, rj, rg, src = [], [], [], {}
    for name in names:
        n = SP.snum(name)
        c = dict(cuts.get(n, {}))
        tt = os.path.join(state_of(name), "truth_table.json")
        if os.path.exists(tt):
            for r in json.load(open(tt, encoding="utf-8")).get("table", {}).get("rows", []):
                c.setdefault((r["server"], float(r["pct"])), float(r["score_eok"]))
        k3 = rows.get(n, {}).get("kr_cut_3")
        k3 = float(k3) if k3 is not None else c.get(("KR", 3.0))
        if k3:
            kr3.append(math.log(k3))
            if ("JP", 3.0) in c:
                rj.append(math.log(c[("JP", 3.0)] / k3))
            if ("GL", 3.0) in c:
                rg.append(math.log(c[("GL", 3.0)] / k3))
        src[name] = {"kr3": bool(k3), "jp3": ("JP", 3.0) in c, "gl3": ("GL", 3.0) in c}
    sd = lambda v: statistics.stdev(v) if len(v) >= 2 else None
    return {"sd_ln_kr3": sd(kr3), "sd_ln_jp_kr": sd(rj), "sd_ln_gl_kr": sd(rg), "n": len(kr3), "n_jp": len(rj), "n_gl": len(rg),
            "mean_ln_jp_kr": statistics.mean(rj) if rj else None, "mean_ln_gl_kr": statistics.mean(rg) if rg else None, "src": src}


# ---------------- 본체 ----------------

def fit_multi(names, boot=100, seed=1, tiers=("full", "shape_only"), grades=("A", "A2"), quiet=False, final=False):
    reg = registry()
    use_tr, use_sh = [], []
    for n in names:
        r = reg.get(n)
        if r is None:
            raise SystemExit(f"{n}: 등록부에 없다 (registry.py --write)")
        if r["role"] == "test" and not final:
            raise SystemExit(f"{n}은 시험 시즌이다 — prior에 넣지 않는다")
        if r["tier"] == "full" and "full" in tiers:
            use_tr.append(n)
            use_sh.append(n)
        elif r["tier"] in ("full", "shape_only") and "shape_only" in tiers:
            use_sh.append(n)
    if not use_sh:
        raise SystemExit(f"쓸 시즌이 없다: {names} (등급 {[reg[n]['tier'] for n in names]})")
    # (a) 변환 칸
    U = []
    meta = {}
    for n in use_tr:
        season, H, obs, units = FP.load_units(state_of(n), grades, "all")
        nk = sum(1 for u in units if u["server"] == "KR")
        for u in units:
            u.update(season=n, H=H, person=f"{n}:{u['person']}")
        meta[n] = {"H": H, "units": {s: sum(1 for u in units if u["server"] == s) for s in PR.SERVERS}}
        cnt = collections.Counter(u["server"] for u in units)
        for u in units:
            u["w"] = 1.0 / cnt[u["server"]]
        U += units
    # 시즌마다 같은 무게: 서버 안에서 시즌 칸 무게 합 1 → 평균 무게가 1이 되게 늘린다
    for srv in PR.SERVERS:
        Us = [u for u in U if u["server"] == srv]
        if Us:
            k = len(Us) / sum(u["w"] for u in Us)
            for u in Us:
                u["w"] *= k
    KR = [u for u in U if u["server"] == "KR"]
    JP = [u for u in U if u["server"] == "JP"]
    seasons_kr = sorted({u["season"] for u in KR})
    cand = {}
    if len(seasons_kr) >= 2:
        for f in FORMS:
            cand[f] = loso(f, KR, seasons_kr)
        best = min(cand, key=lambda f: cand[f]["mean"])
        for f in sorted(FORMS, key=lambda f: NPAR[f]):
            if cand[f]["mean"] <= 1.05 * cand[best]["mean"]:
                best = f
                break
        form = best
        sel = "시즌 하나씩 빼기(LOSO) |잔차| 중앙의 시즌 평균 최소 (5% 안이면 모수 적은 꼴)"
    else:
        form = "sqrt"
        sel = "학습 시즌이 2개 미만이라 sqrt (탐색 2.2 기본형)"
    if not KR:
        raise SystemExit(f"% 변환에 쓸 한섭 칸이 없다 (full 시즌 {use_tr})")
    th, lvl, s0, rho, res = fit_pooled(form, KR, seasons_kr)
    tr = {"form": form, "theta": th, "bx": 0.0, "m": {"KR": 1.0}}
    m_jp, m_src = fit_m_jp(JP, tr, lvl)
    tr["m"] = {"KR": 1.0, "JP": m_jp, "GL": 1.0}
    # 시즌 수준의 표본오차 (시즌 안 사람 붓스트랩)
    rng = random.Random(seed)
    se = {}
    for s in seasons_kr:
        Us = [u for u in KR if u["season"] == s]
        byp = collections.defaultdict(list)
        for u in Us:
            byp[u["person"]].append(u)
        ks = sorted(byp)
        vals = []
        for _ in range(30):
            Ub = [u for _ in ks for u in byp[ks[rng.randrange(len(ks))]]]
            # 한 시즌 배수를 직접 맞춘다 (fit_levels는 기하평균으로 나눠 한 시즌이면 1)
            best, _ = FP.nelder_mead(lambda v: sum(FP.huber(FP.unit_resid(u, math.exp(v[0]) * p_unit(tr, u)), 0.05) for u in Ub),
                                     [0.0], step=0.1)
            vals.append(best[0])
        se[s] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    lnm = [math.log(lvl[s]) for s in seasons_kr]
    var = statistics.variance(lnm) if len(lnm) >= 2 else None
    tau_p_raw = None if var is None else math.sqrt(max(var - statistics.mean(se[s] ** 2 for s in seasons_kr), 0.0))
    tau_p = max(TAU_P_FLOOR, tau_p_raw or 0.0)
    # 잔차·모수 붓스트랩 (시즌 안 사람을 다시 뽑는다)
    z = sorted(r / math.sqrt(s0 ** 2 + (rho * p) ** 2) for h, p, r in res)
    reps = []
    byp = collections.defaultdict(list)
    for u in KR + JP:
        byp[(u["server"], u["season"], u["person"])].append(u)
    groups = collections.defaultdict(list)
    for k in byp:
        groups[(k[0], k[1])].append(k)
    for b in range(boot):
        Ub = []
        for g, ks in groups.items():
            for _ in range(len(ks)):
                Ub += byp[ks[rng.randrange(len(ks))]]
        KRb = [u for u in Ub if u["server"] == "KR"]
        thb = wfit_form(form, KRb, x0=th, lvl=lvl)
        lvb, _ = fit_levels(form, thb, KRb, seasons_kr)
        trb = {"form": form, "theta": thb, "bx": 0.0, "m": {"KR": 1.0}}
        mjb, _ = fit_m_jp([u for u in Ub if u["server"] == "JP"], trb, lvb)
        resb = [(u["h"], p, u["P"] - p) for u in KRb if u["cens"] is None for p in [p_unit(trb, u, lvb.get(u["season"], 1.0))]]
        s0b, rhob = FP.fit_sigma(resb)
        reps.append({"theta": thb, "bx": 0.0, "m": {"KR": 1.0, "JP": mjb, "GL": 1.0}, "s0": s0b, "rho": rhob})
    alt = {f: wfit_form(f, KR) for f in FORMS}
    Hmin = min(meta[s]["H"] for s in seasons_kr)
    hk = sorted(u["h"] / u["H"] * Hmin for u in KR if u["cens"] is None)
    h_support = {"h_min": hk[0] if hk else Hmin, "h_ramp": hk[min(4, len(hk) - 1)] if hk else Hmin,
                 "note": "fit_prior와 같은 규칙. 여러 시즌 칸의 h를 h/H로 맞춰 가장 짧은 시즌 길이로 잰 값"}
    # 기준선 1: 시즌 일차 중앙의 중앙 (시즌 수준으로 나누지 않은 날 P — 「지난 시즌들의 평균 수렴 비율」)
    days = int(math.ceil(max(meta[s]["H"] for s in seasons_kr) / 24))
    dm = []
    for d in range(1, days + 1):
        v = []
        for s in seasons_kr:
            L = [u["P"] for u in KR if u["season"] == s and u["cens"] is None and int(u["h"] // 24) + 1 == d]
            if L:
                v.append(statistics.median(L))
        dm.append(statistics.median(v) if v else None)
    for i in range(len(dm)):
        if dm[i] is None:
            nb = [x for x in dm if x is not None]
            dm[i] = (dm[i - 1] if i and dm[i - 1] is not None else nb[0]) if nb else 0.0
    rat = []
    for d in range(1, days + 1):
        a = [u["P"] for u in JP if u["cens"] is None and int(u["h"] // 24) + 1 == d]
        k = dm[d - 1]
        if a and k:
            rat.append(statistics.median(a) / k)
    # 같은 점수 쌍 잡음 (p_update용, 기본 끔)
    pr_res = []
    for s in seasons_kr:
        _, c0, H2 = PR.season_info(state_of(s))
        rows, _ = PR.load_obs(state_of(s), H2, H2, c0)
        for p in PR.same_score_pairs(rows, H2, tr):
            pr_res.append(p["d_obs"] - p["d_model"])
    pair_sigma = max(FP.mad(pr_res) or 0.05, 0.02)
    # (b) 모양
    shapes = {}
    for n in use_sh:
        tt = os.path.join(state_of(n), "truth_table.json")
        table = json.load(open(tt, encoding="utf-8")) if os.path.exists(tt) else {}
        shapes[n] = FP.build_shape(state_of(n), grades, table)
    shape = shape_multi(shapes)
    tb = table_spread(names, final=final)
    tau_ratio_raw = [v for v in (tb["sd_ln_jp_kr"], tb["sd_ln_gl_kr"]) if v is not None]
    tau_ratio = max(TAU_RATIO_FLOOR, max(tau_ratio_raw) if tau_ratio_raw else 0.0)
    tau_scale0 = max(TAU_SCALE0_FLOOR, tb["sd_ln_kr3"] or 0.0)
    if len(use_sh) == 1:   # 한 시즌이면 fit_prior의 D(일섭–한섭)를 쓴다
        one = FP.build_shape(state_of(use_sh[0]), grades, {})
        for srv in PR.SERVERS:
            shape["servers"][srv]["D"] = one["servers"][srv]["D"]
    lab = f"시즌 {','.join(names)} (변환 {','.join(use_tr)} · 모양 {','.join(use_sh)})"
    sday = collections.defaultdict(list)
    for h, p, r in res:
        sday[int(h // 24) + 1].append(r)
    prior = {
        "season": "+".join(names), "generated_by": "tools/solo_raid/fit_prior_multi.py", "H": Hmin,
        "inputs": {"seasons": names, "transform_seasons": use_tr, "shape_seasons": use_sh, "boot": boot, "seed": seed,
                   "curve_grades": list(grades), "tiers": list(tiers),
                   "post_all_digest": {n: (registry()[n].get("data") or {}).get("stamp_digest") for n in names}},
        "basis": lab,
        "transform": {
            "basis": lab, "form": form, "theta": th, "bx": 0.0, "m": tr["m"], "m_src": {"KR": "own (여러 시즌 공통)", "JP": m_src, "GL": "borrow_KR"},
            "season_levels": {s: round(v, 4) for s, v in lvl.items()}, "season_level_se_ln": {s: round(v, 4) for s, v in se.items()},
            "sigma": {"s0": s0, "rho": rho, "form": "sigma(h) = sqrt(s0^2 + (rho*P(h))^2), log10 단위 (시즌 수준을 뺀 시즌 안 잡음)"},
            "sigma_by_day": {d: {"n": len(v), "mad": FP.mad(v)} for d, v in sorted(sday.items())},
            "z_resid": [round(v, 4) for v in z], "boot": reps, "alt_forms": alt, "h_support": h_support,
            "candidates": cand, "selection": sel,
            "n_units": {s: sum(1 for u in U if u["server"] == s and u["cens"] is None) for s in PR.SERVERS},
            "n_units_by_season": meta,
            "day_median": {"KR": dm}, "day_median_m": {"KR": 1.0, "JP": statistics.median(rat) if rat else 1.0, "GL": 1.0},
            "pair_sigma": pair_sigma, "n_pairs": len(pr_res), "pair_level_check": None,
        },
        "shape": dict(shape, basis=lab, flags_note="여러 시즌 무게 평균의 대표 flag (own: own인 시즌이 2개 이상)"),
        "season_diff": {
            "basis": f"{len(names)}개 시즌({lab})에서 잰 퍼짐. 시즌이 적어 거칠다 — 구간 보정은 3.6 검증에서 정한다",
            "tau_p": {"value": round(tau_p, 4), "raw": None if tau_p_raw is None else round(tau_p_raw, 4),
                      "basis": f"시즌 수준 ln m_s {[round(v, 3) for v in lnm]}의 퍼짐에서 시즌 안 표본오차를 뺀 값 (바닥 {TAU_P_FLOOR})"},
            "tau_shape": {"value": 1.0, "basis": "D(u)가 시즌 간 ln r 표준편차(부호 = 첫 주성분)라 1σ = 1"
                                                + (" — 시즌 1개라 fit_prior의 일섭–한섭 D" if len(use_sh) == 1 else "")},
            "tau_ratio": {"value": round(tau_ratio, 4), "raw": tau_ratio_raw,
                          "basis": f"집계표 ln(일섭 3%/한섭 3%) sd {tb['sd_ln_jp_kr']} (n {tb['n_jp']}) · ln(글섭/한섭) sd {tb['sd_ln_gl_kr']} (n {tb['n_gl']}) 중 큰 값, 바닥 {TAU_RATIO_FLOOR}"},
            "tau_scale0": {"value": round(tau_scale0, 4), "raw": tb["sd_ln_kr3"],
                           "basis": f"한섭 3% 컷 ln sd (시즌 {tb['n']}개), 바닥 {TAU_SCALE0_FLOOR}"},
            "table_spread": tb,
        },
    }
    if not quiet:
        report(prior)
    return prior


def report(pr):
    t = pr["transform"]
    print(f"prior {pr['season']} · 변환 {pr['inputs']['transform_seasons']} · 모양 {pr['inputs']['shape_seasons']}")
    print(f"  칸 {t['n_units']} · 시즌별 {[(s, m['units']['KR'], m['H']) for s, m in t['n_units_by_season'].items()]}")
    for f, c in t["candidates"].items():
        print(f"  꼴 {f:6s} LOSO 시즌 평균 {c['mean']:.4f} · {[(s, round(v, 4)) for s, v in c['per_season'].items()]}")
    print(f"  선택 {t['form']} θ={[round(v, 4) for v in t['theta']]} m={ {k: round(v, 3) for k, v in t['m'].items()} } "
          f"σ s0={t['sigma']['s0']:.4f} ρ={t['sigma']['rho']:.3f} · 시즌 수준 {t['season_levels']} (se {t['season_level_se_ln']})")
    H = pr["H"]
    trc = {"form": t["form"], "theta": t["theta"], "bx": 0.0, "m": t["m"]}
    print("  P(h): " + " · ".join(f"h{h}: {PR.p_full(trc, 'KR', h, 1.0, H):.3f}" for h in (6, 12, 18, 24, 36, 48, 72, 96)))
    print(f"  기준선 1 일차 중앙: {[round(v, 3) for v in t['day_median']['KR']]}")
    sh = pr["shape"]
    for s in PR.SERVERS:
        S = PR.Shape(sh["servers"][s])
        print(f"  모양 {s}: " + " · ".join(f"{q}%: r={math.exp(S.lr_at(math.log(q))):.3f} D={S.D_at(math.log(q)):+.3f}" for q in (0.5, 1, 2, 5, 10, 30)))
    print("  시즌 간: " + " · ".join(f"{k}={v['value']} (raw {v.get('raw')})" for k, v in pr["season_diff"].items() if isinstance(v, dict) and "value" in v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", help="쉼표로 (예: S38,S40,S41)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiers", default="full,shape_only")
    ap.add_argument("--grades", default="all", choices=["A", "all"])
    SP.add_final_arg(ap)
    a = ap.parse_args()
    SP.guard_args(a)
    pr = fit_multi(a.seasons.split(","), a.boot, a.seed, tuple(a.tiers.split(",")),
                   ("A",) if a.grades == "A" else ("A", "A2"), final=a.final)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(PR.jsonable(pr), f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
