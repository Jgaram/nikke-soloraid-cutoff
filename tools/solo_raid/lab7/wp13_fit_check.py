"""WP13a 자료 재현 셈 R1~R5 (2026-11-04) — 칸 (시즌 × T)마다 T까지의 자료로 맞춘 곡면이 그 자료를 표현하는지 변형 단위로 판정한다.

  python3 -I tools/solo_raid/lab7/wp13_fit_check.py --pred NAME=PRED.json:FIT.jsonl.gz [--pred ...] \\
      --lb data/solo-raid/analysis/r7/lb_D1.json --obs-dir W/obs7/full --out OUT.json     (+ OUT.md)

  --pred   변형 하나: 이름 = m3.py predict의 예측 파일 : 같은 실행의 --save-fit 파일 (gzip JSONL, 칸 한 줄).
  --lb     하한 파일 (baseline.py lb). R4.
  --obs-dir 끝난 시즌 전체 표 (obs_table.py build --full → DIR/{시즌}.jsonl.gz, T = H). R5의 「막판 정답·관측 어긋남」 셈.
  --out    결과 JSON ({"variants": {이름: {R1..R5: {pass, ...}, all_pass, ...}}}) + 같은 이름의 .md 요약.

조건 (docs/solo-raid-method.md 「자료 재현 조건」, 문턱은 지난 측정을 근거로 작업 세션이 정함):
  R1 지금 컷 읽기   T − 6 < 촬영 ≤ T 인 A 줄의 PSIS-LOO PIT를 모은 것: [0.1, 0.9] 안 몫 0.70~0.90, < 0.1 몫 ≤ 0.15, > 0.9 몫 ≤ 0.15.
                    창 (초반 T 18~48 · 막판 H−12~H)·시즌별로 나눠 같은 문턱으로 보고 (줄 20개 이상인 묶음만 실패 목록에 넣는다).
  R2 사진 잔차 꼴   A 줄 (사후 p_kr ≥ 0.5 · p_contam < 0.5) 잔차 (log10 %, 관측 구간 가운데 − KR 성분 mu 사후 중앙)를 KST 3시간 칸 (05시 경계) × 일차 ×
                    관측 %대 (0.1–0.3 · 0.3–1 · 1–3 · 3–10 · 10–40) 로 묶은 중앙. n (칸·줄 합친 수) ≥ 20 이면 |중앙| ≤ 0.01, 적은 묶음은 보고만.
                    실패한 묶음과 그 묶음에 줄을 낸 (시즌, T) 목록.
                    판정 고침 (2026-11-04, WP13b 1차를 본 뒤 — docs 「자료 재현 조건」 R2 줄): 같은 사진이 여러 칸에 들어가 묶음당 사진은 중앙 7장뿐이고
                    사진 잔차 sd 0.027이라 중앙의 잡음 (약 0.009)이 문턱 0.01과 같다 — 편향 없는 모형도 101묶음 중 약 42개를 못 지난다 (모의).
                    그래서 문턱 0.01은 그대로 두고 잡음을 넣어 판정한다: 사진 단위 무리 붓스트랩 (사진을 다시 뽑고 그 사진의 칸·줄을 모두 넣어 같은 중앙, 400번, 씨앗 0)의
                    95% 구간이 [−0.01, 0.01]과 겹치지 않으면 실패 (구간 아래 > 0.01 또는 위 < −0.01). 검사 묶음은 n ≥ 20 이고 사진 ≥ 5장.
                    옛 판정 (|중앙| ≤ 0.01)은 R2_raw로 함께 낸다 (all_pass에는 쓰지 않음).
  R3 기록 쌍        같은 사람·같은 점수 쌍의 dL − 곡면 dP (사후 중앙)를 쌍 가운데 시각의 KST 6시간 칸 (05·11·17·23시 경계)으로 묶은 무게 1/var 평균.
                    쌍 ≥ 15인 칸이면 |평균| ≤ 0.01 log10. 일차별 분해도 보고.
  R4 알려진 하한    score.py g1과 같은 정의: 자기 칸 (own) · T ≥ 18 · 중앙 < lb − 0.05 인 칸. 1·3·10% 0칸이어야 한다 (그 밖 %는 보고만).
  R5 시즌 끝 읽기   T = H 칸 |e| (e = 중앙 ÷ 정답 − 1, 정답은 answer_key): 1·3% ±2% · 10% ±3% 밖인 시즌 0.
                    「막판 정답·관측 어긋남」 (m2.py score-extra (3)과 같은 셈: 정답 < lb·(1 − 0.002) 또는 정답 > ub·(1 + 0.002), lb = % ≥ q 인 KR A 줄의 최고 점수,
                    ub = 촬영 ≥ H − 6 이고 % ≤ q 인 KR A 줄의 최저 점수) 인 (시즌, %) 칸은 따로 적고 세지 않는다.
  덧붙임            contam_share_A (A 줄 가운데 사후 오염 확률 > 0.5 몫)가 사전 (1.5%)의 2배 (3%)를 넘는 칸 목록.
all_pass = R1~R5 모두 pass (검사할 자료가 없어 pass가 null인 R이 있으면 false, untested에 적는다).
공개 규칙: 시즌·T·시각·%·점수만 쓴다 (글 번호·사람 열쇠·사진 열쇠 없음).
"""
import argparse
import collections
import gzip
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "lab6"))
import common as C  # noqa: E402
import obs_model as OM  # noqa: E402
import obs_table as OT  # noqa: E402
import wp12_tables as WT  # noqa: E402
import anchor as AN  # noqa: E402

PCT_BANDS = ((0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 40.0))
R1_IN, R1_TAIL, R1_MIN_N = (0.70, 0.90), 0.15, 20
R2_TOL, R2_MIN_N = 0.01, 20
R2_MIN_PHOTOS, R2_BOOT, R2_SEED = 5, 400, 0
R3_TOL, R3_MIN_N = 0.01, 15
R4_PCTS = (1.0, 3.0, 10.0)
R5_TOL = {1.0: 0.02, 3.0: 0.02, 10.0: 0.03}
MIS_TOL, MIS_WIN = 0.002, 6.0
CONTAM_PRIOR, CONTAM_LIMIT = 0.015, 0.03


def read_fit(path):
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_pred(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d, [dict(c, T=float(c["T"]), pct=float(c["pct"])) for c in d["cells"]]


def tkey(h):
    return (h["season"], h["T"])


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def window_of(h):
    T, H = h["T"], h["H"]
    if 18 - 1e-9 <= T <= 48 + 1e-9:
        return "early"
    if H - 12 - 1e-9 <= T <= H + 1e-9:
        return "late"
    return "other"


# ---------------- R1 ----------------

def _pit_stats(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return {"n": 0}
    return {"n": n, "inside": _r(np.mean((p >= 0.1) & (p <= 0.9))), "below": _r((p < 0.1).mean()), "above": _r((p > 0.9).mean())}


def _r1_ok(st):
    return (R1_IN[0] <= st["inside"] <= R1_IN[1]) and st["below"] <= R1_TAIL and st["above"] <= R1_TAIL


def r1(fits):
    allp, byw, bys = [], collections.defaultdict(list), collections.defaultdict(list)
    for h in fits:
        ps = [x["pit"] for x in h["pit"] if x["pit"] is not None]
        allp += ps
        byw[window_of(h)] += ps
        bys[h["season"]] += ps
    pooled = _pit_stats(allp)
    out = {"pooled": pooled, "by_window": {k: _pit_stats(v) for k, v in sorted(byw.items())},
           "by_season": {k: _pit_stats(v) for k, v in sorted(bys.items(), key=lambda kv: int(kv[0][1:]))}}
    out["pass"] = None if pooled["n"] == 0 else bool(_r1_ok(pooled))
    fails = []
    for kind, dd in (("window", out["by_window"]), ("season", out["by_season"])):
        for k, st in dd.items():
            if st["n"] >= R1_MIN_N and not _r1_ok(st):
                fails.append({"group": f"{kind}:{k}", **st})
    out["fail_groups"] = fails
    out["limits"] = {"inside": list(R1_IN), "tail_max": R1_TAIL}
    return out


# ---------------- R2 ----------------

def _band(p):
    for i, (lo, hi) in enumerate(PCT_BANDS):
        if lo <= p < hi:
            return i
    return None


def r2(fits):
    """돌려줌: (고친 판정, 옛 판정). 고친 판정은 사진 단위 무리 붓스트랩 95% 구간 (위 docstring)."""
    g = collections.defaultdict(list)       # (bin3, day, band) -> [resid]
    gp = collections.defaultdict(list)      # (bin3, day, band) -> [사진 번호] (g와 같은 순서)
    who = collections.defaultdict(collections.Counter)
    pid = {}
    for h in fits:
        s = h["season"]
        rows = [x for x in h["rows_A"] if x["resid"] is not None and x["p_kr"] is not None and x["p_kr"] >= 0.5 and (x["p_contam"] or 0) < 0.5 and x["pct"]]
        if not rows:
            continue
        hh = np.array([x["h"] for x in rows], float)
        b3, dy = WT.bin3(s, hh), WT.day_of(s, hh)
        for x, b, d in zip(rows, np.atleast_1d(b3), np.atleast_1d(dy)):
            bd = _band(x["pct"])
            if bd is None:
                continue
            k = (int(b), int(d), bd)
            g[k].append(x["resid"])
            gp[k].append(pid.setdefault((s, x["h"], x["pct"], x.get("zmid")), len(pid)))   # 같은 시즌·시각·%·구간 = 같은 사진
            who[k][tkey(h)] += 1
    rng = np.random.default_rng(R2_SEED)
    groups, fails, groups_raw, fails_raw = [], [], [], []
    for k in sorted(g):
        b, d, bd = k
        v, ph = np.asarray(g[k], float), np.asarray(gp[k])
        med = float(np.median(v))
        up = np.unique(ph)
        e = {"kst3h": f"{(5 + 3 * b) % 24:02d}-{(8 + 3 * b) % 24:02d}", "day": d, "pct_band": f"{PCT_BANDS[bd][0]:g}-{PCT_BANDS[bd][1]:g}",
             "n": len(v), "n_photos": int(len(up)), "median": _r(med, 4)}
        cells = [{"season": s_, "T": T_, "n": n} for (s_, T_), n in sorted(who[k].items(), key=lambda kv: (int(kv[0][0][1:]), kv[0][1]))]
        er = dict(e)
        er["tested"] = len(v) >= R2_MIN_N
        er["ok"] = (abs(med) <= R2_TOL) if er["tested"] else None
        groups_raw.append(er)
        if er["tested"] and not er["ok"]:
            fails_raw.append({**er, "cells": cells})
        e["tested"] = len(v) >= R2_MIN_N and len(up) >= R2_MIN_PHOTOS
        if e["tested"]:
            idx = {p: np.flatnonzero(ph == p) for p in up}
            bs = np.empty(R2_BOOT)
            for t in range(R2_BOOT):
                pick = rng.choice(up, len(up))
                bs[t] = np.median(v[np.concatenate([idx[p] for p in pick])])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            e["ci95"] = [_r(float(lo), 4), _r(float(hi), 4)]
            e["ok"] = not (lo > R2_TOL or hi < -R2_TOL)
        else:
            e["ok"] = None
        groups.append(e)
        if e["tested"] and not e["ok"]:
            fails.append({**e, "cells": cells})
    n_rows = int(sum(len(v) for v in g.values()))
    tested = [e for e in groups if e["tested"]]
    tested_raw = [e for e in groups_raw if e["tested"]]
    new = {"pass": None if not tested else not fails, "n_groups": len(groups), "n_tested": len(tested), "n_rows": n_rows, "n_photos": len(pid),
           "fails": fails, "groups": groups, "limit": R2_TOL, "rule": "photo-cluster bootstrap 95% CI outside ±0.01 (2026-11-04 고침)",
           "min_photos": R2_MIN_PHOTOS, "boot": R2_BOOT, "seed": R2_SEED}
    raw = {"pass": None if not tested_raw else not fails_raw, "n_groups": len(groups_raw), "n_tested": len(tested_raw), "n_rows": n_rows,
           "fails": fails_raw, "groups": groups_raw, "limit": R2_TOL, "rule": "|median| <= 0.01 (계산 전 판정)"}
    return new, raw


# ---------------- R3 ----------------

def r3(fits):
    bins = collections.defaultdict(list)      # 6h 칸 -> [(resid, w)]
    days = collections.defaultdict(list)
    who = collections.defaultdict(collections.Counter)
    for h in fits:
        s = h["season"]
        ps = [p for p in h["pairs"] if p["dP_med"] is not None]
        if not ps:
            continue
        tm = np.array([(p["t1"] + p["t2"]) / 2 for p in ps], float)
        c0h = WT.season_info()[s]["c0h"]
        b6 = np.mod(np.floor((c0h + tm - 5.0) / 6.0 + 1e-9), 4).astype(int)
        dy = np.atleast_1d(WT.day_of(s, tm))
        for p, b, d in zip(ps, b6, dy):
            x = (p["dL"] - p["dP_med"], 1.0 / p["var"])
            bins[int(b)].append(x)
            days[int(d)].append(x)
            who[int(b)][tkey(h)] += 1

    def wm(v):
        r, w = np.array([x[0] for x in v]), np.array([x[1] for x in v])
        return float((r * w).sum() / w.sum())
    by_bin, fails = {}, []
    for b in sorted(bins):
        lab = f"{(5 + 6 * b) % 24:02d}-{(11 + 6 * b) % 24:02d}"
        e = {"n": len(bins[b]), "wmean": _r(wm(bins[b]), 4), "tested": len(bins[b]) >= R3_MIN_N}
        e["ok"] = (abs(e["wmean"]) <= R3_TOL) if e["tested"] else None
        by_bin[lab] = e
        if e["tested"] and not e["ok"]:
            fails.append({"kst6h": lab, **e, "cells": [{"season": s_, "T": T_, "n": n} for (s_, T_), n in sorted(who[b].items(), key=lambda kv: (int(kv[0][0][1:]), kv[0][1]))]})
    by_day = {str(d): {"n": len(v), "wmean": _r(wm(v), 4)} for d, v in sorted(days.items())}
    tested = [e for e in by_bin.values() if e["tested"]]
    return {"pass": None if not tested else not fails, "by_kst6h": by_bin, "by_day": by_day, "fails": fails, "n_pairs": int(sum(len(v) for v in bins.values())), "limit": R3_TOL}


# ---------------- R4 ----------------

def r4(cells, lb):
    bad = []
    byp = collections.Counter()
    for c in cells:
        if c.get("server", "KR") != "KR" or c["T"] < 18 or not bool(c.get("own", True)) or not c.get("pred"):
            continue
        v = lb.get((c["season"], c["T"], c["pct"]))
        if v is not None and c["pred"] < v - 0.05:
            bad.append({"season": c["season"], "T": c["T"], "pct": c["pct"], "pred": _r(c["pred"], 3), "lb": _r(v, 3)})
            byp[c["pct"]] += 1
    main = [b for b in bad if b["pct"] in R4_PCTS]
    n_checked = sum(1 for c in cells if c.get("server", "KR") == "KR" and c["T"] >= 18 and (c["season"], c["T"], c["pct"]) in lb)
    return {"pass": None if n_checked == 0 else not main, "n_checked": n_checked, "n_bad_1_3_10": len(main), "by_pct": {f"{k:g}": v for k, v in sorted(byp.items())},
            "bad_cells": bad}


# ---------------- R5 ----------------

_A_ROWS = {}


def _A_rows(obs_dir, s):
    if s not in _A_ROWS:
        path = os.path.join(obs_dir, f"{s}.jsonl.gz")
        if not os.path.exists(path):
            _A_ROWS[s] = None
        else:
            head, rows = OT.read_cell(path)
            _A_ROWS[s] = (float(head["H"]), [r for r in OM.select_rows(rows, head["H"], stage1=False) if r["tg"] == "A" and r["pct"] and r.get("server") == "KR"])
    return _A_ROWS[s]


def mismatch(obs_dir, s, q, tv):
    """막판 정답·관측 어긋남 (m2 score-extra (3)과 같은 셈, ub는 촬영 ≥ H − 6). -> (어긋남?, 세부)"""
    got = _A_rows(obs_dir, s)
    if got is None:
        return False, {"note": "표 없음"}
    H, A = got
    lbs = [r["score"] for r in A if r["pct"] >= q]
    ubs = [r["score"] for r in A if r["pct"] <= q and r["h"] >= H - MIS_WIN]
    lbv, ubv = (max(lbs) if lbs else None), (min(ubs) if ubs else None)
    below = bool(lbv and tv < lbv * (1 - MIS_TOL))
    above = bool(ubv and tv > ubv * (1 + MIS_TOL))
    return below or above, {"truth_below_lb": below, "truth_above_ub": above, "lb": _r(lbv, 3), "ub": _r(ubv, 3)}


def r5(cells, obs_dir):
    truth = AN.truth()
    rows, fails, excl = [], [], []
    for c in cells:
        s = c["season"]
        if c.get("server", "KR") != "KR" or not c.get("pred") or abs(c["T"] - C.info(s)[1]) > 1e-9:
            continue
        tv = truth.get((s, "KR", c["pct"]))
        if tv is None:
            continue
        e = c["pred"] / tv - 1
        row = {"season": s, "pct": c["pct"], "e": _r(e, 4)}
        if c["pct"] in R5_TOL:
            row["tol"] = R5_TOL[c["pct"]]
            row["out"] = bool(abs(e) > R5_TOL[c["pct"]] + 1e-12)
            if row["out"]:
                mis, det = mismatch(obs_dir, s, c["pct"], tv)
                row["mismatch"] = mis
                row["mismatch_detail"] = det
                (excl if mis else fails).append({"season": s, "pct": c["pct"], "e": row["e"], **({"detail": det} if mis else {})})
        rows.append(row)
    fseasons = sorted({f["season"] for f in fails}, key=lambda x: int(x[1:]))
    n = sum(1 for r in rows if "tol" in r)
    return {"pass": None if n == 0 else not fails, "n_cells": n, "fail_seasons": fseasons, "fail_cells": fails, "mismatch_excluded": excl,
            "cells": sorted(rows, key=lambda r: (int(r["season"][1:]), r["pct"])), "tol": {f"{k:g}": v for k, v in R5_TOL.items()}}


# ---------------- 합치기 ----------------

def check(name, pred_path, fit_path, lb, obs_dir):
    pd, cells = load_pred(pred_path)
    fits = read_fit(fit_path)
    contam = [{"season": h["season"], "T": h["T"], "contam_share_A": h["contam_share_A"], "n_A": h["n_A"]} for h in fits
              if h.get("contam_share_A") is not None and h["contam_share_A"] > CONTAM_LIMIT]
    r2n, r2raw = r2(fits)
    res = {"R1": r1(fits), "R2": r2n, "R3": r3(fits), "R4": r4(cells, lb), "R5": r5(cells, obs_dir)}
    untested = [k for k, v in res.items() if v["pass"] is None]
    res["all_pass"] = bool(all(v["pass"] is True for v in res.values()))
    res["R2_raw"] = r2raw
    res["all_pass_raw_R2"] = bool(res["all_pass"] and r2raw["pass"] is True) if r2n["pass"] is not None else False
    res["untested"] = untested
    res["contam_over_limit"] = contam
    res["n_cells"] = len(fits)
    res["variants_in_fit"] = sorted({(h["variant"], h["bind"]) for h in fits})
    res["pred_file_made"] = pd.get("made", {}).get("cmd")
    return res


def _flag(p):
    return "-" if p is None else ("통과" if p else "못 지남")


def md(out):
    L = ["# WP13 자료 재현 R1~R5", "", f"만든 날: {out['date']} · 하한 {out['lb']}", ""]
    L += ["| 변형 | 칸 | R1 | R2 | R3 | R4 | R5 | 전부 |", "|---|---|---|---|---|---|---|---|"]
    for n, v in out["variants"].items():
        p = v["R1"]["pooled"]
        r1s = f"{_flag(v['R1']['pass'])} (안 {p.get('inside')} · 아래 {p.get('below')} · 위 {p.get('above')} · n {p['n']})" if p["n"] else "-"
        r2s = f"{_flag(v['R2']['pass'])} (실패 {len(v['R2']['fails'])}/{v['R2']['n_tested']} 묶음; 옛 판정 {_flag(v['R2_raw']['pass'])} {len(v['R2_raw']['fails'])}/{v['R2_raw']['n_tested']})"
        r3s = f"{_flag(v['R3']['pass'])} (실패 {len(v['R3']['fails'])}칸, 쌍 {v['R3']['n_pairs']})"
        r4s = f"{_flag(v['R4']['pass'])} ({v['R4']['n_bad_1_3_10']}칸)"
        r5s = f"{_flag(v['R5']['pass'])} (밖 {len(v['R5']['fail_seasons'])}시즌, 어긋남 제외 {len(v['R5']['mismatch_excluded'])})"
        L.append(f"| {n} | {v['n_cells']} | {r1s} | {r2s} | {r3s} | {r4s} | {r5s} | {'예' if v['all_pass'] else '아니오'} |")
    for n, v in out["variants"].items():
        L += ["", f"## {n}", ""]
        for k, key in (("R1", "fail_groups"), ("R2", "fails"), ("R3", "fails")):
            xs = v[k][key]
            L.append(f"- {k} 못 지난 묶음 {len(xs)}개" + ("" if not xs else ": " + "; ".join(_short(k, x) for x in xs[:12]) + (" ..." if len(xs) > 12 else "")))
        L.append(f"- R3 일차별 (n, 가중 평균): " + ", ".join(f"{d}일 {x['n']}/{x['wmean']}" for d, x in v["R3"]["by_day"].items()))
        L.append(f"- R4 못 지난 칸 ({len(v['R4']['bad_cells'])}, % 따로 {v['R4']['by_pct']}): " + ", ".join(f"{b['season']} T{b['T']:g} {b['pct']:g}%" for b in v["R4"]["bad_cells"][:20]))
        L.append(f"- R5 밖 시즌: {v['R5']['fail_seasons']} · 어긋남이라 제외: " + ", ".join(f"{x['season']} {x['pct']:g}%" for x in v["R5"]["mismatch_excluded"]))
        L.append(f"- 오염 몫 3% 넘는 칸 ({len(v['contam_over_limit'])}): " + ", ".join(f"{x['season']} T{x['T']:g} ({x['contam_share_A']})" for x in v["contam_over_limit"][:20]))
        if v["untested"]:
            L.append(f"- 검사할 자료 없음: {v['untested']}")
    return "\n".join(L) + "\n"


def _short(k, x):
    if k == "R1":
        return f"{x['group']} 안 {x['inside']} 아래 {x['below']} 위 {x['above']} (n {x['n']})"
    if k == "R2":
        ci = f" 구간 {x['ci95'][0]}~{x['ci95'][1]}" if x.get("ci95") else ""
        return f"{x['day']}일 {x['kst3h']}시 {x['pct_band']}% 중앙 {x['median']}{ci} (n {x['n']} · 사진 {x['n_photos']}; " + ",".join(f"{c['season']}T{c['T']:g}" for c in x["cells"][:6]) + ")"
    return f"{x['kst6h']}시 평균 {x['wmean']} (n {x['n']})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", action="append", required=True, help="이름=예측파일:save-fit파일 (여러 번)")
    ap.add_argument("--lb", required=True)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.lb, encoding="utf-8") as f:
        lbd = json.load(f)
    lb = {(c["season"], float(c["T"]), float(c["pct"])): c["lb"] for c in lbd["cells"] if c.get("lb") is not None}
    out = {"about": "WP13 자료 재현 R1~R5 (wp13_fit_check.py)", "date": "2026-11-04", "lb": a.lb, "obs_dir": a.obs_dir, "variants": {}}
    for sp in a.pred:
        name, rest = sp.split("=", 1)
        pp, fp = rest.rsplit(":", 1)
        out["variants"][name] = check(name, pp, fp, lb, a.obs_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    base = a.out[:-5] if a.out.endswith(".json") else a.out
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(md(out))
    for n, v in out["variants"].items():
        print(n, {k: _flag(v[k]["pass"]) for k in ("R1", "R2", "R3", "R4", "R5")}, "all_pass", v["all_pass"], flush=True)


if __name__ == "__main__":
    main()
