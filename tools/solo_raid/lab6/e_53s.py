"""분석 5.3 B — 채점·고르기 (사전 고정 규칙 그대로). docs/solo-raid-analysis.md 「5.3 B 사전 고정」.

  python3 -I tools/solo_raid/lab6/e_53s.py --b B.json [--b B2.json ...] --fmain M.json [--fmain ...] --fref R.json [--fref ...] --out SEL.json

- F: xval 결과 (main = 0.5·1·2·3%, ref = 5·10%). 1·3%(와 0.5·2%)에 M1 — backtest_web.py와 같은 셈 (freeze_model.m1_table·predict.m1_factor).
- 후보 칸이 비면(닻 없음) F 값으로 대신한다 (fallback으로 센다).
- +W: 후보 자신의 앞 맞힐 시즌 한섭 1·3% |z| 80% 분위로 T별 구간 넓히기 (F의 M1과 같은 규칙). g4를 못 지킨 후보에만 얹는다.
"""
import argparse
import collections
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import freeze_model as FZ  # noqa: E402
import predict as PR  # noqa: E402
import anchor as A  # noqa: E402

M1_T = (12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0)
MAIN_Q = (1.0, 3.0)
TOL = {0.5: 0.03, 1.0: 0.02, 2.0: 0.03, 3.0: 0.02, 5.0: 0.03, 10.0: 0.03}
SIMPLE = ["B0", "N1", "N2m3", "N2m5"]
MODS = ["", "+k10", "+u1", "+u1+k10"]
NEAR = 0.003   # score 0.3%p


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_f(mains, refs):
    rm = [r for p in mains for r in jl(p)["rows"] if r.get("scheme") == "recent5" and r.get("server", "KR") == "KR"]
    rr = [r for p in refs for r in jl(p)["rows"] if r.get("scheme") == "recent5" and r.get("server", "KR") == "KR"]
    F = {}
    m1s = {}
    for r in rm:
        s = r["season"]
        if s not in m1s:
            tab = FZ.m1_table([x for x in rm if float(x["T"]) in M1_T], s)
            m1s[s] = {"m1": {"T": {f"{T:g}": c for T, (c, _, _) in tab.items()}}} if tab else None
        q = float(r["pct"])
        if q not in (0.5, 1.0, 2.0, 3.0) or not r.get("pred") or not r.get("i80") or None in r["i80"]:
            continue
        m = r["pred"]
        lo, hi = r["i80"]
        if m1s[s]:
            c = PR.m1_factor(m1s[s], float(r["T"]))
            lo, hi = m * (lo / m) ** c, m * (hi / m) ** c
        F[(s, float(r["T"]), q)] = (m, lo, hi)
    for r in rr:
        q = float(r["pct"])
        if q not in (5.0, 10.0) or not r.get("pred") or not r.get("i80") or None in r["i80"]:
            continue
        F[(r["season"], float(r["T"]), q)] = (r["pred"], r["i80"][0], r["i80"][1])
    return F


def windows(s, T):
    _, H, _ = A.info(s)
    if T in (18.0, 24.0, 36.0, 48.0):
        return "main"
    if T in (H - 12, H - 6, H):
        return "end"
    return None


def build(B, F, cand):
    """후보 칸 {(시즌, T, q): dict(pred, lo, hi, lb, own)} — 비면 F."""
    out = {}
    for c in B:
        if c["cand"] != cand:
            continue
        k = (c["season"], float(c["T"]), float(c["q"]))
        if "pred" in c:
            out[k] = dict(pred=c["pred"], lo=c["i80"][0], hi=c["i80"][1], lb=c.get("lb"), own=True, a=c.get("a"))
        elif k in F:
            m, lo, hi = F[k]
            out[k] = dict(pred=m, lo=lo, hi=hi, lb=c.get("lb"), own=False)
    return out


def build_f(F, B):
    lbs = {(c["season"], float(c["T"]), float(c["q"])): c.get("lb") for c in B if c["cand"] == "N1+u1"}
    return {k: dict(pred=m, lo=lo, hi=hi, lb=lbs.get(k), own=True) for k, (m, lo, hi) in F.items()}


def build_flb(F, B):
    """F+하한: 중앙·80% 아래 끝을 라벨+미상 하한에서 자름 (보고만)."""
    out = {}
    for k, d in build_f(F, B).items():
        lb = d["lb"]
        e = dict(d)
        if lb is not None:
            e["pred"] = max(d["pred"], lb)
            e["lo"] = max(d["lo"], lb)
            e["hi"] = max(d["hi"], e["pred"])
        out[k] = e
    return out


def widen(cells, truth):
    """+W: 앞 맞힐 시즌의 그 후보 |z| 80% 분위로 T별 배율 (넓히기만)."""
    rows = []
    for (s, T, q), d in cells.items():
        tv = truth.get((s, "KR", q))
        if tv is None or q not in MAIN_Q:
            continue
        rows.append(dict(season=s, T=T, pct=q, server="KR", pred=d["pred"], i80=[d["lo"], d["hi"]], truth=tv, err=d["pred"] / tv - 1))
    out = {}
    cache = {}
    for (s, T, q), d in cells.items():
        if s not in cache:
            tab = FZ.m1_table([r for r in rows if float(r["T"]) in M1_T], s)
            cache[s] = {"m1": {"T": {f"{t:g}": c for t, (c, _, _) in tab.items()}}} if tab else None
        e = dict(d)
        if cache[s] and q in (0.5, 1.0, 2.0, 3.0):
            c = PR.m1_factor(cache[s], T)
            m = d["pred"]
            if d["lo"] > 0 and m > 0:
                e["lo"], e["hi"] = m * (d["lo"] / m) ** c, m * (d["hi"] / m) ** c
        out[(s, T, q)] = e
    return out


def evaluate(cells, truth, qs, tol):
    win = collections.defaultdict(list)
    per_season = collections.defaultdict(list)
    endH = []
    g1 = 0
    fb = collections.Counter()
    byT = collections.defaultdict(list)
    for (s, T, q), d in cells.items():
        if q not in qs:
            continue
        tv = truth.get((s, "KR", q))
        if tv is None:
            continue
        e = d["pred"] / tv - 1
        w = windows(s, T)
        _, H, _ = A.info(s)
        Tk = "H" if T == H else ("H-6" if T == H - 6 else ("H-12" if T == H - 12 else f"{T:g}"))
        byT[Tk].append((s, e, d["lo"] <= tv <= d["hi"], d["hi"] / d["lo"] - 1 if d["lo"] > 0 else None, d["own"]))
        if T >= 18 and d["own"] and d.get("lb") is not None and d["pred"] < d["lb"] - 0.05:
            g1 += 1
        if w is None:
            continue
        win[w].append((s, abs(e), d["lo"] <= tv <= d["hi"]))
        per_season[s].append(abs(e))
        if not d["own"]:
            fb[w] += 1
        if T == H and abs(e) > tol[q] + 1e-12:
            endH.append(f"{s} {q:g}% {e * 100:+.1f}%")
    sc = {w: st.mean(x[1] for x in L) for w, L in win.items()}
    # 시즌 중앙 score: (T, q) 칸마다 시즌 중앙 |e|의 평균, 창별
    medsc = {}
    for w in ("main", "end"):
        g = collections.defaultdict(list)
        for (s, T, q), d in cells.items():
            tv = truth.get((s, "KR", q))
            if q in qs and tv is not None and windows(s, T) == w:
                _, H, _ = A.info(s)
                g[(q, "H" if T == H else ("H-6" if T == H - 6 else ("H-12" if T == H - 12 else T)))].append(abs(d["pred"] / tv - 1))
        medsc[w] = st.mean(st.median(v) for v in g.values()) if g else None
    worst = max(((st.mean(v), s) for s, v in per_season.items()), default=(None, None))
    cov = {w: sum(x[2] for x in L) / len(L) for w, L in win.items()}
    no2123 = [x for w in ("main", "end") for x in win[w] if x[0] not in ("S21", "S22", "S23")]
    return dict(score=(sc["main"] + sc["end"]) / 2, main=sc["main"], end=sc["end"],
                score_med=(medsc["main"] + medsc["end"]) / 2, cov_main=cov["main"], cov_end=cov["end"],
                g1=g1, g2=len(endH), g2_list=endH, worst=worst[0], worst_season=worst[1],
                fallback=dict(fb), n=dict((w, len(L)) for w, L in win.items()),
                score_no2123=None if not no2123 else st.mean(x[1] for x in no2123),
                byT={k: dict(n=len(v), mae=st.mean(abs(x[1]) for x in v), bias=st.mean(x[1] for x in v),
                             cov=st.mean(x[2] for x in v), w80=st.median([x[3] for x in v if x[3] is not None] or [0]),
                             own=sum(x[4] for x in v)) for k, v in byT.items()})


def select(res, Fres, order):
    """사전 규칙: g1~g4, score 최소, 0.3%p 안이면 단순한 쪽. res {이름: 평가}. 반환 (고른 이름, 기록)."""
    log = []
    ok = {}
    for name in order:
        r = res[name]
        why = []
        if r["g1"] > 0:
            why.append(f"g1 {r['g1']}")
        if r["g2"] > Fres["g2"]:
            why.append(f"g2 {r['g2']} > F {Fres['g2']}")
        if r["worst"] > Fres["worst"] + 0.005:
            why.append(f"g3 최악 {r['worst'] * 100:.2f} > F {Fres['worst'] * 100:.2f}+0.5")
        if r["cov_main"] < 0.70 or r["cov_end"] < 0.70:
            why.append(f"g4 포함 {r['cov_main']:.2f}/{r['cov_end']:.2f}")
        log.append(dict(name=name, score=round(r["score"] * 100, 3), fail=why))
        if not why:
            ok[name] = r["score"]
    ok["F"] = Fres["score"]   # F는 단순한 순서 맨 뒤 후보 (조건은 F 자신이 기준이라 늘 지킴)
    log.append(dict(name="F", score=round(Fres["score"] * 100, 3), fail=[]))
    best = min(ok.values())
    for name in list(order) + ["F"]:
        if name in ok and ok[name] <= best + NEAR:
            return name, log
    return "F", log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", action="append", required=True)
    ap.add_argument("--fmain", action="append", required=True)
    ap.add_argument("--fref", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = [c for p in a.b for c in jl(p)["cells"]]
    seen = set()
    B2 = []
    for c in reversed(B):   # 같은 칸이 여러 파일에 있으면 뒤 파일
        k = (c["season"], c["T"], c["q"], c["cand"])
        if k not in seen:
            seen.add(k)
            B2.append(c)
    B = B2
    F = load_f(a.fmain, a.fref)
    truth = A.truth()
    cands = [b + m for b in SIMPLE for m in MODS]
    out = {"about": "5.3 B 채점 (사전 고정 규칙)", "groups": {}}
    for gname, qs in (("1·3%", MAIN_Q), ("10%", (10.0,))):
        cells = {n: build(B, F, n) for n in cands}
        Fc = build_f(F, B)
        res = {n: evaluate(cells[n], truth, qs, TOL) for n in cands}
        Fres = evaluate(Fc, truth, qs, TOL)
        Flb = evaluate(build_flb(F, B), truth, qs, TOL)
        order = list(cands)
        pick, log = select(res, Fres, order)
        wres = {}
        # g4만 못 지킨 후보에 +W를 얹어 다시 (W는 그 꼴의 뒤 — 단순한 순서 맨 뒤 무리)
        if True:
            for n in cands:
                if any(f.startswith("g4") for f in next(x for x in log if x["name"] == n)["fail"]):
                    wres[n + "+W"] = evaluate(widen(cells[n], truth), truth, qs, TOL)
        pickW, logW = (None, None)
        if wres:
            allres = dict(res, **wres)
            orderW = order + [n + "+W" for n in order if n + "+W" in wres]
            pickW, logW = select(allres, Fres, orderW)
        final = pickW if wres else pick
        out["groups"][gname] = dict(pick=final, pick_noW=pick, log=logW or log, F=Fres, F_lb=Flb, cands=res, W=wres)
        print(f"== {gname}: 고름 {final} (W 없이 {pick})")
        print(f"   F      score {Fres['score'] * 100:.2f} (main {Fres['main'] * 100:.2f} · end {Fres['end'] * 100:.2f}) 중앙판 {Fres['score_med'] * 100:.2f} 포함 {Fres['cov_main']:.2f}/{Fres['cov_end']:.2f} g2 {Fres['g2']} 최악 {Fres['worst'] * 100:.2f} {Fres['worst_season']}")
        print(f"   F+하한 score {Flb['score'] * 100:.2f} (main {Flb['main'] * 100:.2f} · end {Flb['end'] * 100:.2f}) g2 {Flb['g2']} 최악 {Flb['worst'] * 100:.2f}")
        for n in cands + list(wres):
            r = res.get(n) or wres[n]
            print(f"   {n:14s} score {r['score'] * 100:.2f} (main {r['main'] * 100:.2f} · end {r['end'] * 100:.2f}) 중앙판 {r['score_med'] * 100:.2f} 포함 {r['cov_main']:.2f}/{r['cov_end']:.2f} "
                  f"g1 {r['g1']} g2 {r['g2']} 최악 {r['worst'] * 100:.2f} {r['worst_season']} F대신 {r['fallback']}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
