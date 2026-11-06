"""M3 운영 보정 층 (docs/solo-raid-method.md 「WP7 정함」 3, 2026-11-07).

연구의 C1 `emp1` (tools/solo_raid/lab7/score.py calibrate_emp, mode "emp1")을 운영에 옮긴 것이다.
창 T (초반 18·24·36·48 · 막판 H−12·H−6·H) 칸은 score.py와 같은 수를 낸다 (check 명령으로 확인).

  python3 -I tools/solo_raid/m3_calib.py fit --pred PRED.json --target S43 --H 120 [--out calib.json]
  python3 -I tools/solo_raid/m3_calib.py check --pred PRED.json      # 창 칸이 score.py emp1과 같은지 (22시즌)

셈 (연구와 같음):
  칸 잔차 d = ln(정답 ÷ 중앙). 무리 g (1·3%, 10%) · 창 w (main, end)마다 풀 = 맞힐 시즌보다 앞선 시즌의 창 w 칸.
  퍼짐 s_T = 풀 안 같은 T 열쇠(tkey) 칸의 |d| 평균 (그 T의 풀 시즌이 2개 미만이면 풀 전체 평균 `_all`).
  z = d ÷ s_T를 풀에서 모아 유한 표본 분위 — 풀 n칸이면 작은 순으로 아래 ⌊(n+1)·0.1⌋번째 · 위 ⌈(n+1)·0.9⌉번째.
  보정 구간 = 중앙 × exp(s_T · [z10, z90]). 풀 10칸 미만이면 모형 구간 그대로.
운영 확장 (연구에서 검증하지 않은 칸):
  - 창 밖 T는 가까운 창 (T < (48 + H−12)/2 → main, 아니면 end)의 z10·z90을 쓰고, 퍼짐 s_T는 그 T 열쇠의 풀 칸 |d| 평균
    (풀 시즌이 2개 미만이면 그 창의 `_all`). 이 칸들은 z 풀에 들어가지 않는다 — 그래서 창 칸의 수는 연구와 같다.
  - 0.5·2%는 1·3% 무리, 5%는 10% 무리 값으로 보정한다.
"""
import argparse
import collections
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GROUPS = {"13": (1.0, 3.0), "10": (10.0,)}
APPLY = {0.5: "13", 1.0: "13", 2.0: "13", 3.0: "13", 5.0: "10", 10.0: "10"}
MAIN_T = (18.0, 24.0, 36.0, 48.0)
CALIB_MIN = 10


def snum(s):
    return int(str(s).lstrip("S"))


def window(T, H):
    """연구의 창 (score.window와 같음). 창 밖이면 None."""
    if T in MAIN_T:
        return "main"
    if T in (H - 12, H - 6, H):
        return "end"
    return None


def near_window(T, H):
    """창 밖 T가 빌릴 창 (운영 확장)."""
    w = window(T, H)
    if w:
        return w
    return "main" if T < (48.0 + H - 12.0) / 2.0 else "end"


def tkey(T, H):
    """score.tkey와 같음."""
    return "H" if T == H else ("H-6" if T == H - 6 else ("H-12" if T == H - 12 else f"{T:g}"))


def load_truth():
    """{(시즌, %): 정답} — answer_key.json의 한섭 줄."""
    with open(os.path.join(ROOT, "data", "solo-raid", "analysis", "answer_key.json"), encoding="utf-8") as f:
        ak = json.load(f)
    return {(r["season"], float(r["pct"])): r["truth"] for r in ak["rows"] if r["truth"] is not None and r["server"] == "KR"}


def season_H():
    """{시즌: H} — 시즌 state의 season.json (predict.season_info)."""
    sys.path.insert(0, HERE)
    import predict as PR  # noqa: E402
    out = {}
    base = os.path.join(ROOT, "data", "solo-raid")
    for d in os.listdir(base):
        st_ = os.path.join(base, d, "state")
        if d.startswith("S") and d[1:].isdigit() and os.path.exists(os.path.join(st_, "season.json")):
            try:
                out[d] = float(PR.season_info(st_)[2])
            except Exception:  # noqa: BLE001 — c_start 없는 옛 시즌은 건너뛴다
                pass
    return out


def residuals(cells, truth, H_of):
    """{(시즌, 무리): [(T, tkey, 창 또는 None, d)]} — 한섭 칸 중 정답·중앙이 있는 것."""
    out = collections.defaultdict(list)
    for c in cells:
        if c.get("server", "KR") != "KR":
            continue
        q = float(c["pct"])
        g = next((k for k, v in GROUPS.items() if q in v), None)
        tv, m, s = truth.get((c["season"], q)), c.get("pred"), c["season"]
        if g is None or tv is None or not m or m <= 0 or s not in H_of:
            continue
        T, H = float(c["T"]), H_of[s]
        out[(s, g)].append((T, tkey(T, H), window(T, H), math.log(tv / m)))
    return out


def fit(cells, truth, H_of, target, H_target):
    """맞힐 시즌 target의 보정 값. 풀 = 예측 파일 안의 target보다 앞선 시즌 전부.
    반환 {무리: {창: {z10, z90, n, pool, s_T: {tkey: 값, "_all": 값}}}}."""
    ds = residuals(cells, truth, H_of)
    pool_s = sorted({s for s, _ in ds if snum(s) < snum(target)}, key=snum)
    out = {}
    for g in GROUPS:
        out[g] = {}
        for w in ("main", "end"):
            pool = [(s, k, d) for s in pool_s for (T, k, ww, d) in ds[(s, g)] if ww == w]
            if len(pool) < CALIB_MIN:
                out[g][w] = dict(z10=None, z90=None, n=len(pool), pool=pool_s, s_T=None)
                continue
            allm = st.mean(abs(d) for _, _, d in pool)
            byk = collections.defaultdict(list)
            for s, k, d in pool:
                byk[k].append((s, abs(d)))
            sT = {k: (st.mean(x for _, x in v) if len({s for s, _ in v}) >= 2 else allm) for k, v in byk.items()}
            sT["_all"] = allm
            z = [d / sT[k] if sT[k] > 0 else 0.0 for s, k, d in pool]
            zs, nn = sorted(z), len(z)
            z10 = zs[max(1, math.floor((nn + 1) * 0.1)) - 1]
            z90 = zs[min(nn, math.ceil((nn + 1) * 0.9)) - 1]
            # 운영 확장: 창 밖 T 열쇠의 퍼짐 (z 풀에는 넣지 않는다)
            ext = collections.defaultdict(list)
            for s in pool_s:
                for (T, k, ww, d) in ds[(s, g)]:
                    if ww is None and near_window(T, H_of[s]) == w:
                        ext[k].append((s, abs(d)))
            sT_ext = {k: (st.mean(x for _, x in v) if len({s for s, _ in v}) >= 2 else allm) for k, v in ext.items() if k not in sT}
            out[g][w] = dict(z10=z10, z90=z90, n=nn, pool=pool_s,
                             s_T={k: round(v, 6) for k, v in sT.items()}, s_T_ext={k: round(v, 6) for k, v in sT_ext.items()})
    return {"target": target, "H": H_target, "rule": "emp1 + 운영 확장 (m3_calib.py)", "groups": out}


def apply(calib, T, pct, med, i80_model):
    """칸 하나 보정. 반환 (i80, 쓴 것 dict). 보정할 수 없으면 모형 구간 그대로."""
    H = float(calib["H"])
    g = APPLY.get(float(pct))
    if g is None or not med or med <= 0:
        return list(i80_model) if i80_model else None, {"calib": None}
    w = near_window(float(T), H)
    e = calib["groups"][g][w]
    if e.get("z10") is None:
        return list(i80_model) if i80_model else None, {"calib": None, "why": f"풀 {e.get('n')}칸 < {CALIB_MIN}"}
    k = tkey(float(T), H)
    if k in e["s_T"]:
        sp, src = e["s_T"][k], "s_T"
    elif k in (e.get("s_T_ext") or {}):
        sp, src = e["s_T_ext"][k], "s_T_ext"
    else:
        sp, src = e["s_T"]["_all"], "_all"
    lo, hi = med * math.exp(sp * e["z10"]), med * math.exp(sp * e["z90"])
    return [lo, hi], {"calib": "emp1", "group": g, "window": w, "in_window": window(float(T), H) is not None,
                      "s_T": sp, "s_T_src": src, "z10": e["z10"], "z90": e["z90"], "n": e["n"]}


def backtest(cells, truth, H_of):
    """예측 파일 칸 전부에 시즌마다 앞 시즌만으로 낸 보정을 단다 (웹 되돌려 보기용). 반환 (칸 목록, {시즌: calib})."""
    seasons = sorted({c["season"] for c in cells}, key=snum)
    cal = {s: fit(cells, truth, H_of, s, H_of[s]) for s in seasons if s in H_of}
    out = []
    for c in cells:
        s = c["season"]
        if c.get("server", "KR") != "KR" or s not in cal:
            out.append(c)
            continue
        iv, info = apply(cal[s], c["T"], c["pct"], c.get("pred"), c.get("i80"))
        out.append(dict(c, i80=iv, i80_model=c.get("i80"), calib=info.get("calib")))
    return out, cal


def cmd_check(a):
    """창 칸이 score.py --calib emp1과 같은 i80인지."""
    sys.path.insert(0, os.path.join(HERE, "lab7"))
    import score as SC  # noqa: E402
    with open(a.pred, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    truth, H_of = load_truth(), season_H()
    mine, _ = backtest(cells, truth, H_of)
    bad = n = 0
    for g, pcts in GROUPS.items():
        ref, _ = SC.calibrate_emp(cells, SC.A.truth(), pcts, "emp1")
        refk = {(c["season"], c["T"], c["pct"]): c for c in ref if c.get("server", "KR") == "KR" and c["pct"] in pcts}
        for c in mine:
            key = (c["season"], c["T"], c["pct"])
            if c.get("server", "KR") != "KR" or c["pct"] not in pcts or window(float(c["T"]), H_of[c["season"]]) is None:
                continue
            r = refk[key]
            n += 1
            if (r.get("i80") is None) != (c.get("i80") is None) or (r.get("i80") and max(abs(x - y) for x, y in zip(r["i80"], c["i80"])) > 1e-9):
                bad += 1
                if bad <= 5:
                    print("다름", key, r.get("i80"), c.get("i80"))
    print(json.dumps({"window_cells": n, "differ": bad}, ensure_ascii=False))
    sys.exit(1 if bad else 0)


def cmd_fit(a):
    with open(a.pred, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    out = fit(cells, load_truth(), season_H(), a.target, a.H)
    with open(a.pred, "rb") as f:
        import hashlib
        out["pool_pred_sha1"] = hashlib.sha1(f.read()).hexdigest()
    out["pool_pred"] = os.path.relpath(os.path.abspath(a.pred), ROOT)
    js = json.dumps(out, ensure_ascii=False, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(js + "\n")
    print(js if not a.out else f"→ {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--pred", required=True)
    f.add_argument("--target", required=True)
    f.add_argument("--H", type=float, required=True)
    f.add_argument("--out")
    c = sub.add_parser("check")
    c.add_argument("--pred", required=True)
    a = ap.parse_args()
    {"fit": cmd_fit, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
