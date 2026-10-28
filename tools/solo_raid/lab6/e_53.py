"""분석 5.3 B — 후보 계산 (사전 고정 뒤). docs/solo-raid-analysis.md 「5.3 B 사전 고정」.

  python3 -I tools/solo_raid/lab6/e_53.py run --work W --out RES.json [--seasons S21-S42]

후보 (모두 결정적 — 뽑기 없음):
  B0      최근 6시간 A·B 닻 중 q에 가장 가까운 % 하나 × 기준 시즌 R 중앙
  N1      모든 A·B 닻 중 기준 시즌 log R 퍼짐 σ_a가 가장 작은 닻 하나
  N2m{m}  σ_a가 작은 닻 m개를 기준 시즌 공분산(줄임)으로 합침
  덧붙임: 미상 닻 허용(u1), 기준 시즌 최근 10(k10). 이름 = 후보[+u1][+k10]
공통: 하한 아래 자르기, 목표 % 사이 단조, 닻이 없으면 그 칸은 비움(none — 채점에서 F로 대신하고 따로 센다).
"""
import argparse
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anchor as A  # noqa: E402
import e_53a as EA  # noqa: E402

N_MIN_ABS = 3          # 닻 하나의 R을 잴 수 있는 기준 시즌 최소 수
N_MIN_FRAC = 0.3       # … 와 기준 시즌 수의 이 몫 중 큰 쪽 (2026-10-28 — 칸 비율만 보고 0.5 → 0.3)
SIGMA_FLOOR = 0.002    # σ_a 아래 끝 (log, 0.2%)
SHRINK = 0.5           # N2 공분산 줄임: Σ = (1−λ)S + λ·diag(S)
PCTS_RUN = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


def sigma(v):
    """퍼짐: 5개 이상이면 MAD×1.4826, 아니면 표준편차. 아래 끝 SIGMA_FLOOR."""
    if len(v) >= 5:
        m = st.median(v)
        s = st.median([abs(x - m) for x in v]) * 1.4826
    else:
        s = st.pstdev(v) if len(v) >= 2 else float("inf")
    return max(s, SIGMA_FLOOR)


def rvec(a, s, ref, q):
    """닻 a의 기준 시즌별 log R {시즌: 값}."""
    out = {}
    for x in ref:
        v = A.log_r(x, a["x"], A.to_season(a["h"], s, x, "h"), q, A.R_DT, A.R_HOW, A.R_DP)
        if v is not None:
            out[x] = v
    return out


def n_min(ref):
    return max(min(N_MIN_ABS, len(ref)), math.ceil(N_MIN_FRAC * len(ref)))


def scored_anchors(pool, s, ref, q, cache):
    out = []
    for a in pool:
        k = (s, a["x"], round(a["h"], 6), q, tuple(ref))   # R은 자리 (%, 시각)와 q·기준 시즌에만 달림 — 사본마다 줄 번호가 달라 값으로 연다
        if k not in cache:
            cache[k] = rvec(a, s, ref, q)
        r = cache[k]
        if len(r) >= n_min(ref):
            out.append((sigma(list(r.values())), a, r))
    out.sort(key=lambda z: (z[0], -z[1]["h"], abs(math.log(z[1]["x"] / q)), z[1]["oi"]))
    return out


def combine(sel, ref):
    """N2: 닻 m개의 log 예측 y_a,s = log x_a + r_a,s 를 시즌 공분산(줄임)으로 가중. 반환: 시즌별 합친 값 목록."""
    m = len(sel)
    common = [x for x in ref if all(x in r for _, _, r in sel)]
    if len(common) < 2 or m == 1:
        _, a, r = sel[0]
        return [math.log(a["s"]) + v for v in r.values()], {"m": 1}
    R = [[r[x] for x in common] for _, _, r in sel]
    mu = [st.mean(v) for v in R]
    S = [[sum((R[i][k] - mu[i]) * (R[j][k] - mu[j]) for k in range(len(common))) / len(common) for j in range(m)] for i in range(m)]
    for i in range(m):
        S[i][i] = max(S[i][i], SIGMA_FLOOR ** 2)
    Sig = [[(1 - SHRINK) * S[i][j] + (SHRINK * S[i][i] if i == j else 0.0) for j in range(m)] for i in range(m)]
    w = solve(Sig, [1.0] * m)
    if w is None:
        _, a, r = sel[0]
        return [math.log(a["s"]) + v for v in r.values()], {"m": 1, "singular": True}
    tot = sum(w)
    w = [x / tot for x in w]
    vals = []
    for x in ref:
        have = [i for i in range(m) if x in sel[i][2]]
        if not have:
            continue
        ws = sum(w[i] for i in have)
        if ws <= 0:
            continue
        vals.append(sum(w[i] * (math.log(sel[i][1]["s"]) + sel[i][2][x]) for i in have) / ws)
    return vals, {"m": m, "w": [round(x, 3) for x in w], "common": len(common)}


def solve(M, b):
    n = len(b)
    T = [M[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(T[r][c]))
        if abs(T[p][c]) < 1e-14:
            return None
        T[c], T[p] = T[p], T[c]
        for r in range(n):
            if r != c:
                f = T[r][c] / T[c][c]
                T[r] = [T[r][j] - f * T[c][j] for j in range(n + 1)]
    return [T[i][n] / T[i][i] for i in range(n)]


def predict_cell(rows, s, T, q, cand, ref, cache):
    """한 칸 (시즌·T·q) 예측. 반환 dict 또는 None (닻 없음)."""
    base, *mods = cand.split("+")
    unknown = "u1" in mods
    pool = A.anchors(rows, unknown=unknown)
    lb = A.lower_bound(rows, q, unknown=unknown)
    if base == "B0":
        rec = [o for o in pool if T - 6 <= o["h"] <= T]
        if not rec:
            return None
        a = min(rec, key=lambda o: (abs(math.log(o["x"] / q)), -o["h"], o["oi"]))
        k = (s, a["x"], round(a["h"], 6), q, tuple(ref))
        if k not in cache:
            cache[k] = rvec(a, s, ref, q)
        r = cache[k]
        if not r:
            return None
        vals = [math.log(a["s"]) + v for v in r.values()]
        info = {"a": [a["x"], round(a["h"], 2), a["grade"], a["kr_label"]], "n_r": len(r)}
    else:
        sc = scored_anchors(pool, s, ref, q, cache)
        if not sc:
            return None
        m = 1 if base == "N1" else int(base[3:])
        sel = sc[:m]
        vals, info = combine(sel, ref)
        info["a"] = [[a["x"], round(a["h"], 2), a["grade"], a["kr_label"], round(sg, 4)] for sg, a, _ in sel]
    d = A.dist([math.exp(v) for v in vals], lb)
    d.update(info)
    d["lb"] = lb
    return d


def run(args):
    seasons = A.TARGETS if not args.seasons else parse(args.seasons)
    cands = args.cands.split(",")
    TV = A.truth()
    out = {"about": "5.3 B 후보 계산", "cands": cands, "cells": []}
    cache = {}
    for s in seasons:
        past = EA.past_of(s)
        for T in EA.t_grid(s):
            stp = EA.trunc_path(args.work, s, T)
            if stp is None:
                continue
            rows = A.load(stp, s, T, unknown=True)
            for cand in cands:
                ref = past[-10:] if "k10" in cand.split("+") else past
                preds = {}
                for q in PCTS_RUN:
                    preds[q] = predict_cell(rows, s, T, q, cand, ref, cache)
                # 목표 % 사이 단조: q가 작을수록 컷이 높다 (중앙·구간 끝 각각 — 큰 q부터 누적 최댓값)
                for key in ("median",):
                    run_max = None
                    for q in sorted(PCTS_RUN, reverse=True):
                        d = preds[q]
                        if d is None:
                            continue
                        if run_max is not None and d[key] < run_max:
                            d[key] = run_max
                            d["mono_fix"] = True
                        run_max = d[key]
                for q, d in preds.items():
                    tv = TV.get((s, "KR", q))
                    rec = dict(season=s, T=T, q=q, cand=cand, truth=tv)
                    if d is not None:
                        rec.update(pred=round(d["median"], 3), i80=[round(x, 3) for x in d["i80"]], i95=[round(x, 3) for x in d["i95"]],
                                   n=d["n"], cut=d["cut"], lb=d["lb"], a=d.get("a"), m=d.get("m"), w=d.get("w"))
                        if tv is not None:
                            rec.update(err=round(d["median"] / tv - 1, 5), in80=d["i80"][0] <= tv <= d["i80"][1])
                    out["cells"].append(rec)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("wrote", args.out, len(out["cells"]))


def parse(spec):
    a, b = spec.split("-")
    return [f"S{n}" for n in range(int(a[1:]), int(b[1:]) + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seasons")
    ap.add_argument("--cands", default="B0,N1,N2m3,N2m5,B0+u1,N1+u1,N2m3+u1,N2m5+u1,B0+k10,N1+k10,N2m3+k10,N2m5+k10,"
                                        "B0+u1+k10,N1+u1+k10,N2m3+u1+k10,N2m5+u1+k10")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
