"""분석 5 출발점 (2026-10-20) — 관측의 성질 점검: 시각 있는 사진은 정확한가, 오류는 어떤 꼴인가, 점을 지나게 하면 무엇이 남나.

docs/solo-raid-analysis.md 「5. 관측점을 지나는 예측」, 결과 정리는 docs/solo-raid-cutoff.md 「관측의 성질」.
레포를 읽기만 한다 (요청 0, 모델 0, 약 30초). 서버 라벨·사람 잇기는 시즌 전체 state 기준이라 T 뒤 글을 본 것이다 —
모순·하한 점검에는 문제없고, 목표 근처 사진 수·거친 기준선은 실제 진행 중보다 낙관적이다.

  python3 -I tools/solo_raid/lab5/e_premise.py [--out data/solo-raid/analysis/premise.json]

c1 같은 사람·같은 정확 점수(10자리 이상)의 %가 시간이 지나며 줄었나 (시각 A·B, 확실히 뒤인 쌍)
c2 서버 라벨 줄끼리 순서 모순 — 「확실히 먼저 찍었고 점수가 같거나 높은데 %가 더 나쁜」 쌍. 어김이 많은 줄부터 빼서 모순 없는 묶음을 만들 때 빼는 줄
c3 이웃 보간 잔차 log10(실제 % / 보간 %) — 같은 시각 창(±10분·1시간·6시간)의 다른 사람 A 사진 중 점수 위·아래(5% 안) 가장 가까운 둘을 이음.
   A 줄은 대조군, 게시 시각만 있는 C 줄은 게시 시각에 둔다 (사용자 지적 「시각 없는 사진은 한쪽 방향 오차」)
c4 하한 「이미 q% 밖이던(% > q) 가장 높은 점수 ≤ 최종 q% 컷」 — 시각 등급과 무관. 정답·지금 F(backtest_F 중앙·80% 아래끝)가 어기는 칸 (T18 뒤)
c5 직전 6시간 A·B 줄 중 목표 % 근처(÷1.5~×1.5) 줄 수, 21시즌 중앙 (라벨 한섭 / 라벨 + 미상)
c6 지금 컷(직전 6시간 라벨 A·B 줄 중 q%를 사이에 둔 가장 가까운 두 점을 log %-점수 꺾은선으로 이음)과 끝까지 오른 몫,
   거친 기준선 「지금 컷 × 다른 시즌 상승비 중앙」 대 지금 F (참고 — 정식 검증 아님, 지금 컷을 낼 수 있는 시즌만)
"""
import argparse, collections, json, math, os, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SEASONS = [f"S{n}" for n in range(19, 42)]
SERVERS = ["KR", "JP", "GL"]
# c4에서 정답보다 높은 하한을 만든 줄 — 이 점검에서 찾음 (2026-10-20). 하한을 다시 셀 때만 뺀다
BOUND_BAD = {("S26", "p1047"): "1.02%를 10.2%로 읽은 것으로 보임 (199.23억, 시각 C) — 정답 1·3·10% 모두보다 높은 하한",
             ("S23", "p1243"): "3.01%·75.12억 vs 집계표 3% 75.0억 — 표 반올림으로 보임 (분석 4.6)"}

_cache = {}


def load(s):
    if s in _cache:
        return _cache[s]
    rows = []
    for l in open(f"{ROOT}/data/solo-raid/{s}/state/core.jsonl"):
        r = json.loads(l)
        if not r.get("use") or r.get("pct") is None or r.get("score") is None or r["src"] != "photo":
            continue
        raw = r.get("score_raw")
        exact = raw is not None and len(str(raw).replace(".", "")) >= 10
        r["s"] = float(raw) if exact else float(r["score"])
        r["s_tol"] = 0.0 if exact else 0.05
        rows.append(r)
    _cache[s] = rows
    return rows


def sure_before(a, b):
    return a["h_hi"] <= b["h_lo"]


def score_ge(a, b):
    if a["s_tol"] == 0 and b["s_tol"] == 0:
        return a["s"] >= b["s"]
    return a["s"] - a["s_tol"] >= b["s"] + b["s_tol"]


def c1():
    cnt = collections.Counter(); fell = []
    for s in SEASONS:
        by = collections.defaultdict(list)
        for r in load(s):
            if r["s_tol"] == 0 and r["t_grade"] in "AB":
                by[(r["server"], round(r["s"], 6))].append(r)
        for (srv, sc), g in by.items():
            g.sort(key=lambda r: r["h"])
            for a, b in zip(g, g[1:]):
                if not sure_before(a, b) or b["h"] - a["h"] < 0.05:
                    continue
                k = "rose" if b["pct"] > a["pct"] else "same" if b["pct"] == a["pct"] else "fell"
                cnt[k] += 1
                if k == "fell":
                    fell.append(dict(season=s, server=srv, score=round(sc, 2), h=[a["h"], b["h"]], pct=[a["pct"], b["pct"]], keys=[a["key"], b["key"]]))
    return dict(counts=dict(cnt), fell=fell)


def conflicts(R):
    adj = collections.defaultdict(set); pairs = 0
    for i, a in enumerate(R):
        for j, b in enumerate(R):
            if i != j and sure_before(a, b):
                pairs += 1
                if score_ge(a, b) and a["pct"] > b["pct"]:
                    adj[i].add(j); adj[j].add(i)
    nbad = sum(len(v) for v in adj.values()) // 2
    inv = len(adj)
    A = {k: set(v) for k, v in adj.items()}; removed = []
    while any(A.values()):
        k = max(A, key=lambda x: (len(A[x]), -x))
        for v in A[k]:
            A[v].discard(k)
        A[k] = set(); removed.append(R[k])
    return pairs, nbad, inv, removed


def c2():
    out = {}; removed_rows = []
    for grades in ["A", "AB", "ABC"]:
        agg = {srv: collections.Counter() for srv in SERVERS}
        for s in SEASONS:
            for srv in SERVERS:
                R = [r for r in load(s) if r["server"] == srv and r["t_grade"] in grades]
                pairs, nbad, inv, rem = conflicts(R)
                agg[srv].update(rows=len(R), pairs=pairs, bad_pairs=nbad, rows_in_conflict=inv, removed=len(rem))
                if grades == "ABC":
                    removed_rows += [dict(season=s, server=srv, key=r["key"], grade=r["t_grade"], h=r["h"], score=round(r["s"], 2), pct=r["pct"],
                                          screen=r["screen"], server_basis=r["server_basis"]) for r in rem]
        out[grades] = {srv: dict(agg[srv]) for srv in SERVERS}
    return dict(by_grades=out, removed_ABC=removed_rows)


def interp(target, t, pool, win):
    near = [p for p in pool if abs(p["h"] - t) <= win and p["person"] != target["person"]]
    up = [p for p in near if p["s"] > target["s"]]; dn = [p for p in near if p["s"] < target["s"]]
    if not up or not dn:
        return None
    u = min(up, key=lambda p: p["s"]); d = max(dn, key=lambda p: p["s"])
    if (u["s"] - d["s"]) / target["s"] > 0.05:
        return None
    w = (target["s"] - d["s"]) / (u["s"] - d["s"])
    return math.log10(target["pct"]) - (math.log10(d["pct"]) * (1 - w) + math.log10(u["pct"]) * w)


def summ(x):
    x = sorted(x); n = len(x)
    if not n:
        return None
    q = lambda p: x[min(n - 1, int(p * n))]
    frac = lambda c: round(sum(c(v) for v in x) / n, 3)
    med = q(0.5)
    return dict(n=n, med=round(med, 4), p10=round(q(0.1), 4), p90=round(q(0.9), 4), spread_x=round(10 ** (1.4826 * st.median([abs(v - med) for v in x])), 4),
                lt_m05=frac(lambda v: v < -0.05), lt_m02=frac(lambda v: v < -0.02), lt_m01=frac(lambda v: v < -0.01),
                gt_p01=frac(lambda v: v > 0.01), gt_p02=frac(lambda v: v > 0.02), gt_p05=frac(lambda v: v > 0.05))


def c3():
    out = {}
    for win in [0.17, 1.0, 6.0]:
        rA, rC = [], []
        for s in SEASONS:
            KR = [r for r in load(s) if r["server"] == "KR"]
            pool = [r for r in KR if r["t_grade"] == "A"]
            for r in KR:
                if r["t_grade"] == "A":
                    v = interp(r, r["h"], pool, win)
                    if v is not None:
                        rA.append(v)
                elif r["t_grade"] == "C" and r["h_hi"] - r["h_lo"] > 0.5:
                    v = interp(r, r["h_hi"], pool, win)
                    if v is not None:
                        rC.append(v)
        out[str(win)] = dict(A_control=summ(rA), C_at_post=summ(rC))
    return out


def c4(BT):
    out = {}
    for drop in (False, True):
        cnt = collections.Counter(); ex = []
        for s, info in BT["seasons"].items():
            R = [r for r in load(s) if r["server"] == "KR" and not (drop and (s, r["key"]) in BOUND_BAD)]
            for q in (1, 3, 10):
                c = info["srv"]["KR"][str(q)]
                if c["truth"] is None:
                    continue
                for row in c["rows"]:
                    T = row[0]
                    if T < 18:
                        continue
                    outq = [r for r in R if r["h_hi"] <= T and r["pct"] > q]
                    if not outq:
                        continue
                    m = max(outq, key=lambda r: r["s"]); cnt[f"{q}_cells"] += 1
                    if c["truth"] < m["s"] - 0.05:
                        cnt[f"{q}_truth_below"] += 1
                    if row[1] < m["s"] - 0.05:
                        cnt[f"{q}_F_mid_below"] += 1
                        if drop:
                            ex.append(dict(season=s, pct=q, T=T, F=row[1], bound=round(m["s"], 2), bound_pct=m["pct"], bound_h=round(m["h"], 1),
                                           bound_grade=m["t_grade"], truth=c["truth"]))
                    if row[2] < m["s"] - 0.05:
                        cnt[f"{q}_F_lo80_below"] += 1
        out["drop_bad" if drop else "all_rows"] = dict(counts=dict(cnt), F_mid_below=ex if drop else None)
    out["bad_rows"] = {f"{k[0]} {k[1]}": v for k, v in BOUND_BAD.items()}
    return out


def c5(BT):
    out = {}
    for q in (1, 3, 10):
        for T in (24, 48, 72, 96, 114, 120):
            a, b = [], []
            for s in BT["seasons"]:
                near = [r for r in load(s) if r["t_grade"] in "AB" and T - 6 <= r["h"] <= T and q / 1.5 <= r["pct"] <= q * 1.5]
                a.append(sum(r["server"] == "KR" for r in near)); b.append(sum(r["server"] in ("KR", None) for r in near))
            out[f"{q}_T{T}"] = dict(kr_med=st.median(a), kr_unknown_med=st.median(b))
    return out


def cur_cut(rows, T, q, W=6.0):
    near = [r for r in rows if r["t_grade"] in "AB" and T - W <= r["h"] <= T]
    ins = [r for r in near if r["pct"] <= q]; outs = [r for r in near if r["pct"] > q]
    if not ins or not outs:
        return None
    a = max(ins, key=lambda r: r["pct"]); b = min(outs, key=lambda r: r["pct"])
    if a["pct"] < q / 1.5 or b["pct"] > q * 1.5 or a["pct"] == b["pct"]:
        return None
    w = (math.log(q) - math.log(a["pct"])) / (math.log(b["pct"]) - math.log(a["pct"]))
    return a["s"] + (b["s"] - a["s"]) * w


def c6(BT):
    res = collections.defaultdict(list)
    for s, info in BT["seasons"].items():
        H = info["H"]; rows = [r for r in load(s) if r["server"] == "KR"]
        for q in (1, 3, 10):
            cell = info["srv"]["KR"][str(q)]
            if cell["truth"] is None:
                continue
            fr = {r[0]: r for r in cell["rows"]}
            for T, key in [(24, "T24"), (48, "T48"), (72, "T72"), (96, "T96"), (H - 6, "H-6"), (H, "H")]:
                c = cur_cut(rows, T, q)
                if c is not None:
                    f = fr.get(float(T))
                    res[(q, key)].append(dict(season=s, g=math.log(cell["truth"] / c), c=c, truth=cell["truth"], F=f[1] if f else None))
    out = {}
    for (q, key), L in sorted(res.items(), key=lambda kv: (kv[0][0], ["T24", "T48", "T72", "T96", "H-6", "H"].index(kv[0][1]))):
        d = dict(n=len(L), seasons=[x["season"] for x in L])
        if len(L) >= 3:
            gp = sorted((math.exp(x["g"]) - 1) * 100 for x in L)
            eb, ef = [], []
            for x in L:
                pred = x["c"] * math.exp(st.median([y["g"] for y in L if y["season"] != x["season"]]))
                eb.append(abs(pred / x["truth"] - 1) * 100)
                if x["F"] is not None:
                    ef.append(abs(x["F"] / x["truth"] - 1) * 100)
            d.update(growth_med=round(st.median(gp), 2), growth_min=round(gp[0], 2), growth_max=round(gp[-1], 2),
                     base_abs_e=round(st.mean(eb), 2), F_abs_e=round(st.mean(ef), 2) if ef else None, F_n=len(ef))
        out[f"{q}_{key}"] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    BT = json.load(open(f"{ROOT}/data/solo-raid/analysis/backtest_F.json"))
    out = dict(about="분석 5 출발점 — 관측의 성질 점검 (tools/solo_raid/lab5/e_premise.py, 2026-10-20). 시즌 전체 state 기준. docs/solo-raid-cutoff.md 「관측의 성질」",
               seasons=f"{SEASONS[0]}~{SEASONS[-1]} (c4~c6는 backtest_F 시즌 S21~S41)",
               c1=c1(), c2=c2(), c3=c3(), c4=c4(BT), c5=c5(BT), c6=c6(BT))
    print(json.dumps({k: out[k] for k in ("c1", "c3", "c5", "c6")}, ensure_ascii=False, indent=None)[:6000])
    print(json.dumps(out["c2"]["by_grades"], ensure_ascii=False))
    print(json.dumps(out["c4"]["drop_bad"]["counts"], ensure_ascii=False), json.dumps(out["c4"]["all_rows"]["counts"], ensure_ascii=False))
    print("ABC 빼는 줄:", [(r["season"], r["server"], r["key"]) for r in out["c2"]["removed_ABC"]])
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("→", a.out)


if __name__ == "__main__":
    main()
