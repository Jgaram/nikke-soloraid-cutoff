"""분석 5.3 A — 탐색 (서술, 고르기 아님). docs/solo-raid-analysis.md 「5.3 계획」 A1~A6.

  python3 -I tools/solo_raid/lab6/e_53a.py a2 --out A2.json                 # 지난 시즌 비 R 표와 퍼짐 (시즌 전체 state, 몇십 초)
  python3 -I tools/solo_raid/lab6/e_53a.py tr --work W --out TR.json         # A1·A3·A4·A6 (T 절단 사본 — xval이 만든 W/trunc/{시즌}/trunc_T{T})

T 절단 사본은 F 다시 내기(xval --work W)가 만든 것을 그대로 읽는다 (없는 칸은 빠짐 — 셈에 적는다). 레포를 읽기만 한다.
"""
import argparse
import collections
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anchor as A  # noqa: E402

GRID_P = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
QS = [1.0, 3.0, 10.0]
DTS = [1.0, 3.0, 6.0]
# A2 지난 시즌 컷을 잇는 꼴: 두 점 꺾은선(seg, ±dt) · 국소 평면(fit, ±dt시간·% ×dp 안) — 2026-10-28 seg가 거의 못 재 fit을 더함
VARIANTS = [("seg", dt, 1.5) for dt in DTS] + [("fit", dt, dp) for dt in (12.0, 24.0) for dp in (2.0, 3.0)]
COORDS = ["h", "rem", "reset"]


def t_grid(s):
    _, H, _ = A.info(s)
    Ts = [float(t) for t in range(6, 121, 6) if t <= H]
    for t in (H - 12, H - 6, H):
        if t not in Ts:
            Ts.append(float(t))
    return sorted(Ts)


def sd(xs):
    return st.pstdev(xs) if len(xs) >= 2 else None


def spearman(xs, ys):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    if len(xs) < 4:
        return None
    a, b = rk(xs), rk(ys)
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else None


def a2(args):
    """지난 시즌 자리 (p, t) → 최종 q%의 log R. 시즌마다 그 시즌 자기 점·자기 정답으로 (끝난 시즌).
    퍼짐 = 시즌 간 표준편차 (log, ×100 ≈ %). coord는 맞힐 시즌을 120시간·05:00 시즌으로 볼 때 다른 시즌으로 옮기는 규칙이라
    여기서는 시즌 자기 시각 h로 재고(=h 좌표), rem·reset은 길이·시작이 다른 S19·S28·S38 줄만 바뀐다 (따로 적음)."""
    out = {"about": "5.3 A2 — 시즌 자기 점으로 잰 log R(자리 p, t → 최종 q). sd·n은 시즌 간", "cells": {}, "coord_note": {}}
    idx = {s: i for i, s in enumerate(A.DEV)}
    for how, dt, dp in VARIANTS:
        for q in QS:
            for p in GRID_P:
                for t in range(6, 121, 6):
                    L = {}
                    for s in A.DEV:
                        v = A.log_r(s, p, float(t), q, dt, how, dp)
                        if v is not None:
                            L[s] = v
                    if not L:
                        continue
                    vals = list(L.values())
                    out["cells"][f"{how}|dt{dt:g}|dp{dp:g}|q{q:g}|p{p:g}|t{t}"] = dict(
                        n=len(L), med=round(st.median(vals), 4), sd=None if sd(vals) is None else round(sd(vals), 4),
                        p10=round(A.quant(sorted(vals), 0.1), 4), p90=round(A.quant(sorted(vals), 0.9), 4),
                        rho_order=None if spearman([idx[s] for s in L], vals) is None else round(spearman([idx[s] for s in L], vals), 3),
                        by={s: round(v, 4) for s, v in L.items()})
    # 길이·시작이 다른 시즌: 같은 맞힐 자리(120시간 05:00 시즌의 t)를 옮겼을 때의 그 시즌 시각
    for s in A.DEV:
        c0, H, off = A.info(s)
        if H != 120 or off != 0:
            out["coord_note"][s] = dict(H=H, offset_from_0500=off,
                                        t60={c: round(A.to_season(60.0, "S41", s, c), 2) for c in COORDS},
                                        t114={c: round(A.to_season(114.0, "S41", s, c), 2) for c in COORDS})
    return out


def past_of(s):
    return [x for x in A.DEV if A.DEV.index(x) < A.DEV.index(s)]


def trunc_path(work, s, T):
    p = os.path.join(work, "trunc", s, f"trunc_T{T:g}", "state")
    return p if os.path.exists(os.path.join(p, "core.jsonl")) else None


def tr(args):
    """A1 기준선 · A3 닻 수 · A4 하한 · A6 미상 닻 치우침. 맞힐 시즌 S21~S42 × T (6시간마다 + H−12·H−6·H)."""
    TV = A.truth()
    cells, missing = [], collections.Counter()
    dt = 3.0
    for s in A.TARGETS:
        past = past_of(s)
        _, H, _ = A.info(s)
        for T in t_grid(s):
            stp = trunc_path(args.work, s, T)
            if stp is None:
                missing["no_trunc"] += 1
                continue
            rows = A.load(stp, s, T, unknown=True)
            for q in QS:
                tv = TV.get((s, "KR", q))
                if tv is None:
                    continue
                c = dict(season=s, T=T, H=H, q=q, truth=tv)
                lab = A.anchors(rows, unknown=False)
                unk = [o for o in A.anchors(rows, unknown=True) if not o["kr_label"]]
                c["n_anchor_label"] = len(lab)
                c["n_anchor_unknown"] = len(unk)
                c["n_person_label"] = len({o["person"] for o in lab})
                c["min_dist_label"] = None if not lab else round(min(abs(math.log(o["x"] / q)) for o in lab), 4)
                rec6 = [o for o in lab if T - 6 <= o["h"] <= T]
                c["n_anchor_label_6h"] = len(rec6)
                lb = A.lower_bound(rows, q, unknown=False)
                lbu = A.lower_bound(rows, q, unknown=True)
                c["lb"] = lb
                c["lb_unknown"] = lbu
                c["lb_gap"] = None if lb is None else round(math.log(tv / lb), 5)
                c["lb_gap_unknown"] = None if lbu is None else round(math.log(tv / lbu), 5)
                # A1 기준선 B0: 최근 6시간 라벨 닻 중 q에 가장 가까운 %(log) 하나 × 앞 시즌 R 중앙
                for name, pool in (("b0", rec6), ("b0_unknown", [o for o in unk if T - 6 <= o["h"] <= T])):
                    if not pool:
                        continue
                    a = min(pool, key=lambda o: (abs(math.log(o["x"] / q)), -o["h"], o["oi"]))
                    Rs = [v for v in (A.log_r(x, a["x"], A.to_season(a["h"], s, x, "h"), q, dt) for x in past) if v is not None]
                    if not Rs:
                        continue
                    d = A.dist([a["s"] * math.exp(v) for v in Rs])
                    c[name] = dict(pred=round(d["median"], 3), i80=[round(v, 3) for v in d["i80"]], n_r=len(Rs),
                                   a_x=a["x"], a_h=round(a["h"], 2), a_grade=a["grade"],
                                   err=round(d["median"] / tv - 1, 5), in80=d["i80"][0] <= tv <= d["i80"][1],
                                   below_lb=lb is not None and d["median"] < lb - 0.05)
                # c6 꼴: 최근 6시간 라벨 A·B 중 q를 사이에 둔 두 점을 이은 지금 컷 × 앞 시즌 (정답 ÷ 같은 때 지금 컷) 중앙
                cc = A.cut_at(rows_ab_label(rows), q, T - 3.0, 3.0)
                if cc is not None:
                    G = []
                    for x in past:
                        tvx = TV.get((x, "KR", q))
                        cx = A.cut_at(A.full(x), q, A.to_season(T - 3.0, s, x, "h"), 3.0)
                        if tvx is not None and cx:
                            G.append(math.log(tvx / cx))
                    if G:
                        d = A.dist([cc * math.exp(v) for v in G])
                        c["b0_bracket"] = dict(pred=round(d["median"], 3), i80=[round(v, 3) for v in d["i80"]], n_r=len(G), now=round(cc, 3),
                                               err=round(d["median"] / tv - 1, 5), in80=d["i80"][0] <= tv <= d["i80"][1])
                cells.append(c)
    return {"about": "5.3 A1·A3·A4·A6 (T 절단 사본, 지난 시즌 비는 앞 시즌만, dt 3시간, 시각 좌표 h)", "missing": dict(missing), "cells": cells}


def rows_ab_label(rows):
    return [o for o in rows if o["grade"] in "AB" and o["kr_label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["a2", "tr"])
    ap.add_argument("--work")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = a2(args) if args.cmd == "a2" else tr(args)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
