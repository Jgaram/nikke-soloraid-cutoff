"""시즌별 침수 크기(% 변환 수준 m_s)를 초반 자료로 읽을 수 있는지 (할 일 3.5e, 2026-10-17 분석 ④, 모델 0, 요청 0, 서술 점검 — 새 모수 없음).

python3 -I tools/solo_raid/explore_early_level.py [--T 24,48] [--boot 200] [--work DIR] [--out F.json]

개발 시즌만 (split 감시). % 변환 꼴 θ는 analysis/prior_dev.json(개발 시즌 전체로 맞춘 공통 꼴)을 쓴다 — θ에는 그 시즌도 들어 있지만 m̂는 정답을 쓰지 않는다.

(1) 초반 단면 m̂ (정답 없이). T 절단 재처리 사본(backtest.trunc_state, xval과 같은 tr)의 core 기본 집합 중 한섭 라벨·시각 등급 A·B·
    % 와 점수 있음·h ≥ 1인 줄을 시점(moment)마다 하나씩 쓴다. 같은 점수의 지금 %는 log10 x = g(ln 점수) − m·f(h) 꼴로 줄어든다
    (y = x·10^P, P = m·f(h)). g는 ln 점수의 2차식, f = prior 꼴. (a, b, c, m)을 최소제곱으로 풀고 사람 붓스트랩으로 표본오차를 낸다.
    「쌍만 쓰던 3.5c와 달리 모든 관측(단면)」 — 같은 점수 자리의 %가 시간에 따라 내려가는 폭이 m을 정한다.
    서버 섞임을 피하려고 라벨 줄만 (미상 줄 포함 판은 참고로 함께).
(2) 시즌 전체 m_s와 비교: prior_dev의 season_levels(정답 곡선으로 맞춘 값). ln m̂와 ln m_s의 상관, 시즌 하나씩 빼기 직선 보정
    (ln m_s ≈ α + β ln m̂, 뺀 시즌으로)으로 예측한 잔차의 퍼짐을 ln m_s 자체의 퍼짐(τ_P 쪽)과 견준다 → 「τ_P 중 설명되는 몫」.
(3) P 시간 모양 점검: 정답이 붙은 한섭 칸(fit_prior.load_units)으로 시즌마다 m_s를 1~2일차(h < 48)·3~5일차(h ≥ 48)로 따로 맞춰
    ln(앞/뒤)의 시즌 간 퍼짐을 시즌 안 표본오차(사람 붓스트랩)와 견준다. 형식(재오픈 S38·재개 S28) 표시는 서술용.
"""
import argparse
import json
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import split as SP  # noqa: E402

DATA = os.path.join(ROOT, "data", "solo-raid")
AN = os.path.join(DATA, "analysis")
FORMAT = {"S38": "재오픈", "S28": "재개", "S26": "재시작", "S41": "모의전 사전 개방"}


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        if abs(M[i][i]) < 1e-12:
            return None
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]
                for c in range(i, n + 1):
                    M[r][c] -= f * M[i][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def ls_m(pts):
    """pts = [(ln s, f(h), log10 x)] → m (log10 x = a + b·z + c·z² − m·f)."""
    if len(pts) < 8:
        return None
    zs = [p[0] for p in pts]
    z0 = statistics.mean(zs)
    X = [[1.0, z - z0, (z - z0) ** 2, -f] for z, f, _ in pts]
    y = [v for _, _, v in pts]
    A = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(4)] for i in range(4)]
    b = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(4)]
    s = solve(A, b)
    return None if s is None else s[3]


def early_m(state, T, th, form, H, labeled_only, boot, seed):
    import predict as PR
    core = jl(os.path.join(state, "core.jsonl"))
    by = {}
    for c in core:
        if not (c.get("use") and c.get("src") == "photo" and c.get("pct") and c.get("score") and c.get("h") is not None):
            continue
        if c["t_grade"] not in ("A", "B") or c["h"] < 1 or c["h"] > T:
            continue
        if labeled_only and c.get("server") != "KR":
            continue
        if not labeled_only and c.get("server") not in ("KR", None):
            continue
        by.setdefault(c["moment"], (c["person"], math.log(c["score"]), PR.p_time(form, th, c["h"], H), math.log10(c["pct"])))
    pts = list(by.values())
    persons = sorted({p[0] for p in pts})
    m = ls_m([p[1:] for p in pts])
    if m is None:
        return {"n": len(pts), "persons": len(persons), "m": None}
    rng = random.Random(seed)
    bp = {}
    for p in pts:
        bp.setdefault(p[0], []).append(p[1:])
    bs = []
    for _ in range(boot):
        sample = []
        for _ in persons:
            sample += bp[persons[rng.randrange(len(persons))]]
        v = ls_m(sample)
        if v is not None and v > 0:
            bs.append(math.log(v))
    se = statistics.pstdev(bs) if len(bs) > 10 else None
    return {"n": len(pts), "persons": len(persons), "m": round(m, 4), "se_ln": None if se is None else round(se, 4)}


def loso_explain(pairs):
    """pairs = {시즌: (ln m̂, ln m_s)} → 상관, 하나씩 빼기 직선 보정 잔차 sd, ln m_s sd."""
    S = sorted(pairs)
    xs = [pairs[s][0] for s in S]
    ys = [pairs[s][1] for s in S]
    if len(S) < 4:
        return None
    r = statistics.correlation(xs, ys)
    res = []
    for i, s in enumerate(S):
        X = [x for j, x in enumerate(xs) if j != i]
        Y = [y for j, y in enumerate(ys) if j != i]
        mx, my = statistics.mean(X), statistics.mean(Y)
        vx = sum((x - mx) ** 2 for x in X)
        b = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / vx if vx > 0 else 0.0
        a = my - b * mx
        res.append(ys[i] - (a + b * xs[i]))
    sd_y = statistics.pstdev(ys)
    loo_null = [ys[i] - statistics.mean(y for j, y in enumerate(ys) if j != i) for i in range(len(ys))]
    rms = lambda v: math.sqrt(sum(e * e for e in v) / len(v))
    return {"n": len(S), "corr": round(r, 3), "sd_ln_ms": round(sd_y, 4), "loo_rms_null": round(rms(loo_null), 4),
            "loo_rms_cal": round(rms(res), 4), "explained": round(1 - rms(res) ** 2 / rms(loo_null) ** 2, 3)}


def shape_check(seasons, form, th, boot, seed):
    import fit_prior as FP
    import fit_prior_multi as FM
    tr = {"form": form, "theta": th, "bx": 0.0, "m": {}}
    out = {}
    rng = random.Random(seed)

    def lvl(Us):
        if len(Us) < 3:
            return None
        best, _ = FP.nelder_mead(lambda v: sum(FP.huber(FP.unit_resid(u, math.exp(v[0]) * FM.p_unit(tr, u)), 0.05) for u in Us), [0.0], step=0.1)
        return best[0]
    for s in seasons:
        season, H, obs, units = FP.load_units(FM.state_of(s), ("A", "A2"), "all")
        U = [dict(u, H=H) for u in units if u["server"] == "KR"]
        e = [u for u in U if u["h"] < 48]
        l = [u for u in U if u["h"] >= 48]
        a, b = lvl(e), lvl(l)
        if a is None or b is None:
            out[s] = {"n_early": len(e), "n_late": len(l), "ln_ratio": None}
            continue
        pe, pl = sorted({u["person"] for u in e}), sorted({u["person"] for u in l})
        ge, gl = {}, {}
        for u in e:
            ge.setdefault(u["person"], []).append(u)
        for u in l:
            gl.setdefault(u["person"], []).append(u)
        bs = []
        for _ in range(boot):
            E_ = [u for _ in pe for u in ge[pe[rng.randrange(len(pe))]]]
            L_ = [u for _ in pl for u in gl[pl[rng.randrange(len(pl))]]]
            x, y = lvl(E_), lvl(L_)
            if x is not None and y is not None:
                bs.append(x - y)
        out[s] = {"n_early": len(e), "n_late": len(l), "ln_early": round(a, 4), "ln_late": round(b, 4), "ln_ratio": round(a - b, 4),
                  "se": round(statistics.pstdev(bs), 4) if len(bs) > 10 else None, "format": FORMAT.get(s)}
    R = [v for v in out.values() if v.get("ln_ratio") is not None and v.get("se")]
    if len(R) >= 3:
        var = statistics.variance(v["ln_ratio"] for v in R)
        mse = statistics.mean(v["se"] ** 2 for v in R)
        chi = sum(((v["ln_ratio"] - statistics.mean(x["ln_ratio"] for x in R)) / v["se"]) ** 2 for v in R)
        summ = {"n": len(R), "sd_ln_ratio": round(math.sqrt(var), 4), "rms_se": round(math.sqrt(mse), 4),
                "excess_sd": round(math.sqrt(max(var - mse, 0.0)), 4), "chi2": round(chi, 1), "df": len(R) - 1}
    else:
        summ = None
    return out, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="24,48")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--work", default=None)
    ap.add_argument("--out")
    SP.add_final_arg(ap)
    a = ap.parse_args()
    if a.final:
        raise SystemExit("개발 시즌 서술 점검이다")
    SP.guard_args(a)
    import predict as PR  # noqa: E402
    import backtest as BT  # noqa: E402
    import xval as XV  # noqa: E402
    import fit_prior_multi as FM  # noqa: E402
    pr = json.load(open(os.path.join(AN, "prior_dev.json"), encoding="utf-8"))
    t = pr["transform"]
    form, th = t["form"], t["theta"]
    lv = t["season_levels"]
    seasons = sorted(lv, key=SP.snum)
    work = a.work or os.path.join(os.environ.get("CLAUDE_SCRATCH", "/tmp"), "solo_raid_early")
    res = {"about": "시즌 침수 크기 m_s를 초반 자료로 읽기 (explore_early_level.py, 3.5e 서술 점검). 개발 시즌만",
           "prior": {"form": form, "theta": th, "season_levels": lv}, "early": {}, "compare": {}}
    for T in [float(x) for x in a.T.split(",")]:
        for lab in (True, False):
            key = f"T{T:g}_{'label' if lab else 'label+unknown'}"
            res["early"][key] = {}
            for s in seasons:
                state = FM.state_of(s)
                tw = os.path.join(work, s)
                os.makedirs(tw, exist_ok=True)
                st = BT.trunc_state(state, T, tw, XV.extra_of(state))
                season, c0, H = PR.season_info(st)
                res["early"][key][s] = early_m(st, min(T, H), th, form, H, lab, a.boot, a.seed)
            pairs = {s: (math.log(v["m"]), math.log(lv[s])) for s, v in res["early"][key].items() if v.get("m") and v["m"] > 0}
            res["compare"][key] = loso_explain(pairs)
            c = res["compare"][key]
            print(f"{key}: " + " ".join(f"{s}:{v['m']}({v['persons']})" for s, v in res["early"][key].items()), flush=True)
            print(f"   ↔ m_s: {c}", flush=True)
    res["shape"], res["shape_summary"] = shape_check(seasons, form, th, a.boot // 4, a.seed)
    print("P 시간 모양 (ln m 앞/뒤):", " ".join(f"{s}:{v.get('ln_ratio')}±{v.get('se')}" for s, v in res["shape"].items()))
    print("   요약:", res["shape_summary"])
    out = a.out or os.path.join(AN, "early_level_dev.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("→", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    sys.exit(main())
