"""지난 시즌을 M3로 되돌려 본 결과를 웹용으로 줄인다 (docs/solo-raid-method.md 「WP7 정함」 5, 2026-11-07 사용자 결정 — 웹에는 M3만).

python3 -I tools/solo_raid/backtest_web_m3.py [--pred PRED.json] [--extra-dir DIR] [--out data/solo-raid/analysis/backtest_M3.json]

입력
- PRED: WP14b 22시즌 예측 (기본 data/solo-raid/analysis/r7/wp14/pred_k4flbudget_D1.json — 한섭, 시즌 × T × % 0.5·1·2·3·5·10, 모형 구간 `i80`).
- EXTRA-DIR: 시즌이 끝날 때마다 더해지는 그 시즌의 되돌려 보기 칸 `pred_S{n}.json` (같은 cells 꼴, 기본 data/solo-raid/analysis/m3/backtest/). 폴더나 파일이 없으면 건너뛴다.
보정 층은 m3_calib.backtest — 시즌마다 그 시즌보다 앞선 시즌의 실제 오차만으로 낸 emp1 (+ 창 밖 T 확장). 실제 컷은 answer_key.json (옛 backtest_F.json과 같은 출처).
웹은 1·3·10%만 보인다. 출력 꼴은 backtest_F.json과 같다(시즌 → 학습 시즌·H·c_start·서버 → 지점 → 정답·근거·허용 오차·줄 [T, 가운데, 80% 아래, 80% 위]) —
다만 신뢰도·기록 수 열은 없다(M3는 내지 않음). 종료 뒤 T(T > H)는 뺀다. 개발 검증이다(시험 시즌 없음).
"""
import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import m3_calib as MC  # noqa: E402

D = os.path.join(ROOT, "data", "solo-raid")
WEB_P = (1.0, 3.0, 10.0)
TOL = {1.0: 0.02, 3.0: 0.02, 10.0: 0.03}
FIRST_TRAIN = 19   # M3 1단계 학습은 S19부터 (stage1_M3_D1)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha1_of(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def r2(v):
    return None if v is None else round(float(v), 2)


def cells_of(p):
    o = jload(p)
    return o["cells"] if isinstance(o, dict) else o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=os.path.join(D, "analysis", "r7", "wp14", "pred_k4flbudget_D1.json"))
    ap.add_argument("--extra-dir", default=os.path.join(D, "analysis", "m3", "backtest"))
    ap.add_argument("--out", default=os.path.join(D, "analysis", "backtest_M3.json"))
    a = ap.parse_args()

    cells = [c for c in cells_of(a.pred)]
    extra = sorted(glob.glob(os.path.join(a.extra_dir, "pred_S*.json")), key=lambda p: MC.snum(os.path.basename(p)[5:-5]))
    have = {(c["season"], c["T"], c["pct"], c.get("server", "KR")) for c in cells}
    for p in extra:
        for c in cells_of(p):
            k = (c["season"], c["T"], c["pct"], c.get("server", "KR"))
            if k not in have:
                have.add(k)
                cells.append(c)
    truth, H_of = MC.load_truth(), MC.season_H()
    ak = jload(os.path.join(D, "analysis", "answer_key.json"))
    basis = {(r["season"], float(r["pct"])): r.get("basis") for r in ak["rows"] if r["server"] == "KR"}
    cal_cells, cal = MC.backtest(cells, truth, H_of)

    seasons = sorted({c["season"] for c in cal_cells}, key=MC.snum)
    out_seasons = {}
    for s in seasons:
        if s not in H_of:
            continue
        sj = jload(os.path.join(D, s, "state", "season.json"))
        pts = {}
        for q in WEB_P:
            R = {}
            for c in cal_cells:
                if c["season"] != s or c.get("server", "KR") != "KR" or float(c["pct"]) != q:
                    continue
                T, m, iv = float(c["T"]), c.get("pred"), c.get("i80")
                if T > H_of[s] or not m or not iv or None in iv or min(iv) <= 0:
                    continue
                R[T] = [round(T, 2), r2(m), r2(iv[0]), r2(iv[1])]
            if R:
                pts[f"{q:g}"] = {"truth": r2(truth.get((s, q))), "basis": basis.get((s, q)) if truth.get((s, q)) is not None else None,
                                 "tol": TOL[q], "rows": [R[t] for t in sorted(R)]}
        if pts:
            out_seasons[s] = {"train": [f"S{n}" for n in range(FIRST_TRAIN, MC.snum(s))], "c_start": sj["c_start"], "H": round(H_of[s]),
                              "calib": {g: {w: {"n": e["n"], "pool": len(e["pool"])} for w, e in ge.items()} for g, ge in cal[s]["groups"].items()},
                              "srv": {"KR": pts}}
    allT = sorted({r[0] for x in out_seasons.values() for p in x["srv"]["KR"].values() for r in p["rows"]})
    out = {"about": ("지난 시즌을 M3(k4fl·budget·emp1)로 되돌려 본 예측 — 시즌마다 그 시즌보다 앞선 시즌 자료만으로 낸 예측과 보정 구간. "
                     "개발 검증이다(22시즌, 시험 시즌 없음). tools/solo_raid/backtest_web_m3.py가 씀 — 웹(tools/publish/site.py)의 「지난 시즌」 탭이 읽는다."),
           "model": "M3", "T": allT, "pcts": list(WEB_P),
           "source": {"pred": {"file": os.path.relpath(os.path.abspath(a.pred), ROOT), "sha1": sha1_of(a.pred)},
                      "extra": [{"file": os.path.relpath(p, ROOT), "sha1": sha1_of(p)} for p in extra],
                      "calib": "tools/solo_raid/m3_calib.py backtest (emp1 + 운영 확장)", "truth": "data/solo-raid/analysis/answer_key.json"},
           "seasons": out_seasons}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"→ {os.path.relpath(a.out, ROOT)} ({os.path.getsize(a.out) / 1024:.0f} KB) · 시즌 {len(out_seasons)}개 · T {len(allT)}개")


if __name__ == "__main__":
    main()
