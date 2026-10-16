"""솔로레이드 구간 보정 — 검증 결과(xval_*.json)의 80% 구간을 다른 시즌들의 오차로 넓히거나 좁힌다 (할 일 3.6 「구간 보정」, 2026-10-16, 모델 0, 요청 0).

python3 -I tools/solo_raid/calib.py data/solo-raid/analysis/xval_dev.json [...] [--server KR] [--out F.json]

criteria.json의 coverage 규칙: 주 80% 구간 포함률(T_main·1·3% 묶음)이 0.70 미만이면 「과신」 → 구간 넓히기를 먼저 한다.
- 줄마다 표준 오차 z = ln(정답/중앙) ÷ (정답 쪽 80% 반폭, 로그). |z| ≤ 1이면 80% 구간 안.
- 배율 c = 학습에 쓴 시즌들의 |z| 80% 분위 (시즌마다 같은 무게 — 시즌 안 줄 무게 합이 1). 80% 구간을 중앙 둘레로 로그 c배 한다.
- 맞힐 시즌의 c는 그 시즌 줄을 빼고 정한다 (정직한 보정 포함률).
  - forward(기본): 그 시즌보다 앞 번호 맞힐 시즌들의 줄로만 (앞 시즌 2개 이상일 때만, 아니면 보정 없이 c = 1).
  - loso·fixed: 그 시즌을 뺀 나머지 맞힐 시즌들의 줄로.
- 지점 묶음은 1·3%와 10%를 따로 잰다 (10%는 구간 모양이 다르다). T는 criteria T_main.
- 요약: 묶음마다 보정 전·후 포함률(시즌×지점×T), 시즌 단위 포함률 평균, 80% 폭 중앙, 전체 시즌으로 정한 c (앞으로 쓸 값의 후보).
시험 시즌은 xval이 넣지 않으므로 여기에도 없다. 결과는 --out(기본: 입력 이름 + _calib.json)에.
"""
import argparse
import collections
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AN = os.path.join(ROOT, "data", "solo-raid", "analysis")
GROUPS = {"1_3": (1.0, 3.0), "10": (10.0,)}


def snum(s):
    return int(str(s).lstrip("S"))


def zval(r):
    lo, hi = r["i80"]
    m, t = r["pred"], r["truth"]
    if not (m and t and lo and hi) or lo <= 0 or hi <= m or m <= lo:
        return None
    d = math.log(t / m)
    h = math.log(hi / m) if d >= 0 else math.log(m / lo)
    return d / h if h > 0 else None


def wq(items, q):
    """[(값, 무게)]의 가중 분위."""
    items = sorted(items)
    tot = sum(w for _, w in items)
    acc = 0.0
    for v, w in items:
        acc += w
        if acc >= q * tot - 1e-12:
            return v
    return items[-1][0]


def season_weighted(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["season"]].append(r)
    return [(abs(r["z"]), 1.0 / len(L)) for L in by.values() for r in L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xval", nargs="+")
    ap.add_argument("--server", default="KR")
    ap.add_argument("--q", type=float, default=0.80)
    ap.add_argument("--out")
    a = ap.parse_args()
    crit = json.load(open(os.path.join(AN, "criteria.json"), encoding="utf-8"))
    TM = set(crit["T_main"])
    for path in a.xval:
        x = json.load(open(path, encoding="utf-8"))
        res = {"about": "구간 보정 (tools/solo_raid/calib.py). c = 다른 시즌 |z| 80% 분위, 80% 구간을 로그 c배",
               "input": os.path.relpath(path, ROOT), "input_run": x.get("run", {}).get("tool_head"), "server": a.server,
               "q": a.q, "schemes": {}}
        for scheme in sorted({r["scheme"] for r in x["rows"]}):
            res["schemes"][scheme] = {}
            for g, pts in GROUPS.items():
                rows = []
                for r in x["rows"]:
                    if (r["scheme"] == scheme and r["server"] == a.server and r["pct"] in pts and r["T"] in TM
                            and r.get("truth") and not r.get("shaky")):
                        z = zval(r)
                        if z is not None:
                            rows.append(dict(r, z=z))
                seasons = sorted({r["season"] for r in rows}, key=snum)
                per = {}
                for s in seasons:
                    if scheme == "forward":
                        others = [r for r in rows if snum(r["season"]) < snum(s)]
                        ok = len({r["season"] for r in others}) >= 2
                    else:
                        others = [r for r in rows if r["season"] != s]
                        ok = len({r["season"] for r in others}) >= 2
                    c = wq(season_weighted(others), a.q) if ok else 1.0
                    mine = [r for r in rows if r["season"] == s]
                    per[s] = {"c": round(c, 3), "calibrated": ok, "n": len(mine),
                              "cov_raw": round(sum(abs(r["z"]) <= 1 for r in mine) / len(mine), 3),
                              "cov_cal": round(sum(abs(r["z"]) <= c for r in mine) / len(mine), 3),
                              "w80_raw": round(statistics.median(r["w80"] for r in mine), 4),
                              "w80_cal": round(statistics.median(r["w80"] * c for r in mine), 4)}
                cal = [s for s in seasons if per[s]["calibrated"]]

                def pooled(key, ss):
                    L = [r for r in rows if r["season"] in ss]
                    cc = {s: (per[s]["c"] if key == "cal" else 1.0) for s in ss}
                    return round(sum(abs(r["z"]) <= cc[r["season"]] for r in L) / len(L), 3) if L else None

                res["schemes"][scheme][g] = {
                    "seasons": per,
                    "pooled_cov_raw": pooled("raw", seasons), "pooled_cov_cal": pooled("cal", seasons),
                    "pooled_cov_raw_on_calibrated": pooled("raw", cal), "pooled_cov_cal_on_calibrated": pooled("cal", cal),
                    "season_mean_cov_cal": round(statistics.mean(per[s]["cov_cal"] for s in cal), 3) if cal else None,
                    "c_all": round(wq(season_weighted(rows), a.q), 3) if rows else None,
                    "n_rows": len(rows), "n_seasons": len(seasons)}
                v = res["schemes"][scheme][g]
                print(f"{os.path.basename(path)} {scheme} {g}: 포함 {v['pooled_cov_raw']} → 보정 {v['pooled_cov_cal']} "
                      f"(보정된 시즌 {len(cal)}개만: {v['pooled_cov_raw_on_calibrated']} → {v['pooled_cov_cal_on_calibrated']}) · "
                      f"c 전체 {v['c_all']} · " + " ".join(f"{s}:c{per[s]['c']}/{per[s]['cov_cal']}" for s in seasons))
        out = a.out or os.path.splitext(path)[0] + "_calib.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("→", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
