"""4절 분석 실험실 (세션 임시). 작업 사본 wk의 도구를 그대로 쓰고, T 절단 사본·prior를 캐시한다.

run(tag, plan, opts...) — plan: [(scheme, target, prior_spec, borrow_seasons)] ; prior_spec:
  ("single", r)                  한 시즌 prior (lab/priors1/{r}.json)
  ("multi", (s1, s2, ...))       여러 시즌 prior (fit_multi, 캐시)
  ("hyb", tsrc, ssrc, dsrc)      transform·shape·season_diff를 다른 prior spec에서 (원인 분해용)
  ("file", path)
결과: lab/res/{tag}.jsonl (xval.score 줄 + 몇 열)
"""
import sys, os, json, math, hashlib, statistics, collections, time
SCR = os.environ.get("LAB4_SCR") or os.path.join(os.environ.get("CLAUDE_SCRATCH", "/tmp"), "lab4")
WK = os.path.join(SCR, "wk")
XW = os.path.join(SCR, "xw")
LAB = os.path.join(SCR, "lab")
sys.path.insert(0, os.path.join(WK, "tools/solo_raid"))
os.chdir(WK)
import predict as PR  # noqa
import xval as XV  # noqa
import fit_prior_multi as FM  # noqa
import backtest as BT  # noqa

SEAS = [f"S{n}" for n in range(19, 42)]
# 자료량 규칙 (4.1 H4, 결과 보기 전에 자료 개수로 정함): 1~2일차 한섭 칸 < 5인 시즌은 % 변환 수준을 잴 수 없어 모양에만 쓴다
EARLY_UNITS_MIN = 5
_reg0 = FM.registry


def early_units(s):
    p = json.load(open(os.path.join(LAB, "priors1", f"{s}.json")))
    u = p["transform"]["units_by_day"]["KR"]
    return u.get("1", 0) + u.get("2", 0)


def _reg_patched():
    reg = _reg0()
    for s, r in reg.items():
        if r.get("tier") == "full" and s in SEAS and early_units(s) < EARLY_UNITS_MIN:
            r = dict(r, tier="shape_only", tier_why=list(r.get("tier_why", [])) + ["4절: 1~2일차 한섭 칸 < 5"])
            reg[s] = r
    return reg


FM.registry = _reg_patched


def full_ok(s):
    return early_units(s) >= EARLY_UNITS_MIN
TS = [6, 12, 18, 24, 30, 36, 48, 60, 72, 84, 96, 108]
CRIT = json.load(open(os.path.join(WK, "data/solo-raid/analysis/criteria.json")))
AKEY = json.load(open(os.path.join(WK, "data/solo-raid/analysis/answer_key.json")))["rows"]


def sn(s):
    return int(str(s).lstrip("S"))


def trunc(s, T):
    return os.path.join(XW, "trunc", s, f"trunc_T{float(T):g}", "state")


_PR = {}


def load_prior(spec):
    k = json.dumps(spec)
    if k in _PR:
        return _PR[k]
    kind = spec[0]
    if kind == "single":
        p = json.load(open(os.path.join(LAB, "priors1", f"{spec[1]}.json")))
    elif kind == "file":
        p = json.load(open(spec[1]))
    elif kind == "multi":
        tr = list(spec[1])
        opts = spec[2] if len(spec) > 2 else {}
        key = hashlib.sha1(json.dumps([tr, opts], sort_keys=True).encode()).hexdigest()[:12]
        fp = os.path.join(LAB, "priorsM", f"{'+'.join(tr) if len(tr) < 6 else tr[0] + '..' + tr[-1] + f'_{len(tr)}'}_{key}.json")
        if os.path.exists(fp):
            p = json.load(open(fp))
        else:
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            p = FM.fit_multi(tr, boot=60, seed=1, quiet=True, **opts)
            p = json.loads(json.dumps(PR.jsonable(p)))
            json.dump(p, open(fp + ".tmp", "w"))
            os.replace(fp + ".tmp", fp)
    elif kind == "hyb":
        t, s, d = load_prior(spec[1]), load_prior(spec[2]), load_prior(spec[3])
        p = dict(t)
        p["shape"] = s["shape"]
        p["season_diff"] = d["season_diff"]
        p["season"] = f"hyb({t['season']}|{s['season']}|{d['season']})"
    else:
        raise ValueError(spec)
    _PR[k] = p
    return p


def base_opts(borrow):
    return {"grades": "all", "c_delay": 0.0, "p_update": False, "robust": "mean", "eps_wrong": PR.EPS_WRONG,
            "notes": [], "contam": (),
            "unknown": {"mode": "site", "servers": ["KR"], "borrow_state": [FM.state_of(s) for s in borrow],
                        "borrow_kind": "label", "mix_file": None, "no_gate": False, "fallback": "site"}}


def one(job):
    scheme, tgt, spec, borrow, T, popts, draws = job
    st = trunc(tgt, T)
    if not os.path.exists(os.path.join(os.path.dirname(st), "DONE")):
        return []
    prior = load_prior(tuple(spec) if isinstance(spec, list) else spec)
    season, c0, H = PR.season_info(st)
    Tc = min(float(T), H)
    opts = base_opts(borrow)
    for k, v in (popts or {}).items():
        if k == "unknown":
            opts["unknown"] = dict(opts["unknown"], **v)
        else:
            opts[k] = v
    if opts.get("early_level") is True:
        opts["early_level"] = {"train": [[s, trunc(s, T)] for s in borrow]}
    rws, cnt = PR.load_obs(st, Tc, H, c0, opts["grades"], opts["c_delay"], None, opts["unknown"])
    mu_info = None
    if opts.get("m_update") is not None:
        import mupd
        cfg = dict(opts.pop("m_update"))
        tau = cfg.pop("tau", None) or prior["season_diff"]["tau_p"]["value"]
        mu_info = mupd.m_posterior(prior, rws, H, tau, **cfg)
        if mu_info:
            opts["_lvl"] = (mu_info["mu"], mu_info["sd"])
    try:
        out = PR.predict(st, prior, Tc, opts, draws, 1, comps_split=False, rows=rws)
    except ValueError as e:   # predict 가장자리 (서버 관측이 모두 걸러짐 등) — 실패로 적는다
        with open(os.path.join(LAB, "fail.log"), "a") as f:
            f.write(json.dumps([scheme, tgt, spec, T, repr(e)]) + "\n")
        return []
    out["unknown"] = cnt.get("unknown")
    for srv, sv in out["servers"].items():
        sv["warning"] = PR.warning(Tc, srv, sv, out["unknown"], prior)
    rs = XV.score(out, AKEY, tgt, float(T), scheme, list(borrow), CRIT)
    for r in rs:
        r["prior"] = prior["season"]
        if mu_info:
            r["m_upd"] = [mu_info["mu"], mu_info["sd"], mu_info.get("persons"), mu_info.get("phi")]
        sv = out["servers"].get(r["server"]) or {}
        r["mixed_rows"] = sv.get("mixed_rows")
        r["central"] = (sv.get("points", {}).get(str(r["pct"])) or {}).get("central")
        lo = (sv.get("label_only") or {}).get("points", {}).get(str(r["pct"]))
        r["label_only"] = lo["median"] if lo else None
    return rs


def _fit(spec):
    t = time.time()
    load_prior(spec)
    return spec, round(time.time() - t)


def prefit(specs, nproc=4):
    specs = [json.loads(x) for x in sorted({json.dumps(sp) for sp in specs})]
    specs = [sp for sp in specs if sp[0] == "multi"]
    if not specs:
        return
    from multiprocessing import Pool
    with Pool(nproc) as p:
        for sp, dt_ in p.imap_unordered(_fit, [tuple(x if not isinstance(x, list) else tuple(x) for x in sp) for sp in specs]):
            print("fit", sp[1][0], "..", sp[1][-1], len(sp[1]), dt_, "s", flush=True)


def run(tag, plan, Ts=None, popts=None, draws=600, nproc=4, force=False):
    out = os.path.join(LAB, "res", f"{tag}.jsonl")
    if os.path.exists(out) and not force:
        return [json.loads(l) for l in open(out)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Ts = Ts or [12, 18, 24, 36, 48, 72, 96]
    jobs = [(sch, t, spec, tuple(b), T, popts, draws) for (sch, t, spec, b) in plan for T in Ts]
    # 여러 시즌 prior는 먼저 만들어 둔다 (프로세스마다 다시 맞추지 않게)
    specs = {json.dumps(j[2]) for j in jobs}
    for s in specs:
        sp = json.loads(s)
        if sp[0] in ("multi",):
            load_prior(sp)
    rows = []
    t0 = time.time()
    if nproc > 1:
        from multiprocessing import Pool
        with Pool(nproc) as p:
            for rs in p.imap_unordered(one, jobs, chunksize=1):
                rows += rs
    else:
        for j in jobs:
            rows += one(j)
    with open(out + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(PR.jsonable(r), ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)
    print(f"[{tag}] {len(jobs)} jobs {len(rows)} rows {time.time() - t0:.0f}s", flush=True)
    return rows


# ---------------- 요약 ----------------

def kr(rows, pcts=(1.0, 3.0), Ts=(18, 24, 36, 48, 72, 96), srv="KR"):
    return [r for r in rows if r["server"] == srv and r["pct"] in pcts and r["T"] in Ts and r.get("err") is not None]


def summ(rows, pcts=(1.0, 3.0), Ts=(18, 24, 36, 48, 72, 96), srv="KR", by_season=False):
    R = kr(rows, pcts, Ts, srv)
    if not R:
        return {}
    cells = collections.defaultdict(list)
    for r in R:
        cells[(r["T"], r["pct"])].append(r)
    per = [statistics.median(abs(r["err"]) for r in L) for L in cells.values()]
    perb1 = [statistics.median(abs(r["err_b1"]) for r in L if r.get("err_b1") is not None) for L in cells.values()
             if any(r.get("err_b1") is not None for r in L)]
    seas = collections.defaultdict(list)
    for r in R:
        seas[r["season"]].append(r)
    scov = {s: sum(r["in80"] for r in L) / len(L) for s, L in seas.items()}
    smae = {s: statistics.mean(abs(r["err"]) for r in L) for s, L in seas.items()}
    sbias = {s: statistics.mean(r["err"] for r in L) for s, L in seas.items()}
    AB = [r for r in R if r["grade"] in ("A", "B")]
    o = {"n": len(R), "seasons": len(seas), "cov80": round(sum(r["in80"] for r in R) / len(R), 3),
         "mae_cell": round(statistics.mean(per), 4), "b1_cell": round(statistics.mean(perb1), 4) if perb1 else None,
         "mean_abs": round(statistics.mean(abs(r["err"]) for r in R), 4),
         "season_mae": round(statistics.mean(smae.values()), 4),
         "low": round(sum(r["err"] < 0 for r in R) / len(R), 3),
         "w80": round(statistics.median(r["w80"] for r in R), 4),
         "scov_min": round(min(scov.values()), 2), "scov_lt50": sum(v < 0.5 for v in scov.values()),
         "covAB": round(sum(r["in80"] for r in AB) / len(AB), 3) if AB else None, "nAB": len(AB),
         "usable": round(sum(r["ok"] for r in R) / len(R), 3)}
    if by_season:
        o["by_season"] = {s: {"cov": round(scov[s], 2), "mae": round(smae[s], 4), "bias": round(sbias[s], 4)} for s in sorted(seas, key=sn)}
    return o


def show(name, rows, **kw):
    windows = {"T18-48": (18, 24, 36, 48), "T72-96": (72, 96), "T_main": (18, 24, 36, 48, 72, 96)}
    parts = []
    for w, Ts in windows.items():
        s = summ(rows, Ts=Ts, **kw)
        if s:
            parts.append(f"{w}: cov {s['cov80']:.2f} |e|cell {s['mae_cell']:.2%} sMAE {s['season_mae']:.2%} low {s['low']:.2f} w80 {s['w80']:.1%} scov_min {s['scov_min']:.2f} AB {s['covAB']}")
    print(f"{name:28s} n_seas {summ(rows, **kw).get('seasons')} | " + " | ".join(parts))


# ---------------- M1 구간 보정 (앞 시즌 → 다음, 결과 보기 전에 정한 규칙) ----------------
# 줄마다 z = ln(정답/중앙) ÷ (정답 쪽 80% 반폭, 로그). 맞힐 시즌 t의 배율 c(T) = t보다 앞 번호 맞힐 시즌들의 같은 T·한섭 1·3% 줄 |z|의
# 80% 분위 (시즌마다 같은 무게). 앞 시즌이 3개 미만이면 c = 1. 넓히기만 한다 (c = max(1, 분위)). 10%는 따로 같은 규칙.
def _z(r):
    lo, hi = r["i80"]
    m, t = r["pred"], r["truth"]
    if not (m and t and lo and hi) or lo <= 0 or hi <= m or m <= lo:
        return None
    d = math.log(t / m)
    h = math.log(hi / m) if d >= 0 else math.log(m / lo)
    return d / h if h > 0 else None


def _wq(items, q):
    items = sorted(items)
    tot = sum(w for _, w in items)
    acc = 0.0
    for v, w in items:
        acc += w
        if acc >= q * tot - 1e-12:
            return v
    return items[-1][0]


def calibrate(rows, min_prev=3, widen_only=True, groups=((1.0, 3.0), (10.0,)), loso=False):
    out = []
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("err") is None or r["server"] != "KR":
            continue
        z = _z(r)
        if z is None:
            continue
        for g in groups:
            if r["pct"] in g:
                by[(g, r["T"])].append((r, z))
    cmap = {}
    for (g, T), L in by.items():
        seas = sorted({r["season"] for r, _ in L}, key=sn)
        for t in seas:
            prev = [(r, z) for r, z in L if (r["season"] != t if loso else sn(r["season"]) < sn(t))]
            ps = collections.Counter(r["season"] for r, _ in prev)
            if len(ps) < min_prev:
                c = 1.0
            else:
                c = _wq([(abs(z), 1.0 / ps[r["season"]]) for r, z in prev], 0.8)
                if widen_only:
                    c = max(1.0, c)
            cmap[(g, T, t)] = c
    for r in rows:
        r2 = dict(r)
        if r.get("err") is not None and r["server"] == "KR":
            g = next((g for g in groups if r["pct"] in g), None)
            c = cmap.get((g, r["T"], r["season"]))
            if c and c != 1.0:
                m = r["pred"]
                lo, hi = r["i80"]
                lo2, hi2 = m * (lo / m) ** c, m * (hi / m) ** c
                r2["i80"] = [lo2, hi2]
                r2["in80"] = lo2 <= r["truth"] <= hi2
                r2["w80"] = hi2 / lo2 - 1
            r2["calib_c"] = c
        out.append(r2)
    return out


def subset(rows, seasons=None, Tmin=None):
    return [r for r in rows if (seasons is None or r["season"] in seasons)]
