"""WP4 빼 보기 보고 (2026-10-31) — 변형별 조정 6시즌 채점을 한 표로 (score.py의 evaluate를 그대로 쓴다).

  python3 -I tools/solo_raid/lab7/wp4_report.py --ref F_tune.json --lb r7/lb_D1.json --out r7/wp4/diag_wp4_D1.json --md r7/wp4/report_D1.md \
      base=r7/pred_M3_tune_D1.json A=r7/wp4/pred_M3A_tune_D1.json ...

주 지표 (중앙값 쪽): score 1·3%·10% (½ 18~48h + ½ 막판), 두 창 따로, g1·g2, 시즌별 score(그 시즌 칸만), 본안과의 차.
보고만: 80% 포함 (두 창)·폭·CRPS. 수렴 칸 수·칸당 시간 (예측 파일 info).
"""
import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S  # noqa: E402

TUNE = ("S22", "S26", "S30", "S34", "S38", "S42")
GROUPS = (("1·3%", (1.0, 3.0)), ("10%", (10.0,)))


def summarize(cells, truth, lb, qs):
    r = S.evaluate(cells, truth, lb, qs, S.TOL, weights="half")   # WP4 결과는 옛 ½·½ 규칙 (score.py 기본이 ⅓·⅔로 바뀜, 2026-10-31)
    per = {}
    for s in TUNE:
        x = S.evaluate(cells, truth, lb, qs, S.TOL, seasons={s}, weights="half")
        per[s] = x["score"]
    keep = ("score", "main", "end", "g1", "g2", "g2_list", "worst", "worst_season", "cov_main", "cov_end", "crps", "w80")
    out = {k: r[k] for k in keep}
    out["per_season"] = per
    return out


def run_info(d):
    info = d.get("info") or []
    if not info:
        return None
    secs = [i["secs"] + i.get("map_secs", 0.0) for i in info]
    ns = [i["n"] for i in info]
    return {"rows_med": st.median(ns), "rows_sum": sum(ns), "cells": len(info), "conv": sum(1 for i in info if i["conv"]), "secs_med": round(st.median(secs), 1), "secs_max": round(max(secs), 1),
            "hours": round(sum(secs) / 3600, 2)}


def f2(x):
    return "—" if x is None else f"{100 * x:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--lb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("preds", nargs="+", help="이름=경로 (첫째가 본안 base여야 한다)")
    a = ap.parse_args()
    truth = S.A.truth()
    lbh, lb = S.load_lb(a.lb)
    items = [(p.split("=", 1)[0], p.split("=", 1)[1]) for p in a.preds]
    ref = S.load_pred(a.ref)
    res = {"about": "WP4 빼 보기 — M3 관측 층 변형, 조정 6시즌 (개발 검증)", "data_ver": lbh["data_ver"],
           "files": {n: S.sha12(p) for n, p in items + [("F", a.ref), ("lb", a.lb)]}, "groups": {}, "runs": {}}
    for gname, qs in GROUPS:
        g = {"F": summarize(ref["cells"], truth, lb, qs)}
        for n, p in items:
            d = S.load_pred(p)
            if d["data_ver"] != lbh["data_ver"]:
                sys.exit(f"{n}: 자료 판 {d['data_ver']} ≠ {lbh['data_ver']}")
            g[n] = summarize(d["cells"], truth, lb, qs)
        res["groups"][gname] = g
    for n, p in items:
        with open(p, encoding="utf-8") as f:
            res["runs"][n] = run_info(json.load(f))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    lines = []
    base = items[0][0]
    for gname, g in res["groups"].items():
        lines.append(f"### {gname} (단위 %, 차 = 변형 − {base})\n")
        lines.append("| 변형 | score | 차 | 18~48h | 막판 | g1 | g2 | 최악 시즌 | " + " | ".join(TUNE) + " | 80% 포함 (두 창) | 폭 중앙 | CRPS |")
        lines.append("|---|" + "---|" * (11 + len(TUNE)))
        b = g[base]["score"]
        for n in ["F"] + [x for x, _ in items]:
            r = g[n]
            lines.append(f"| {n} | {f2(r['score'])} | {'' if n in ('F', base) else f'{100 * (r['score'] - b):+.2f}'} | {f2(r['main'])} | {f2(r['end'])} | {r['g1']} | {r['g2']} | "
                         f"{f2(r['worst'])} {r['worst_season']} | " + " | ".join(f2(r["per_season"][s]) for s in TUNE) +
                         f" | {r['cov_main']:.2f} · {r['cov_end']:.2f} | {f2(r['w80']['main'])} · {f2(r['w80']['end'])} | {f2(r['crps'])} |")
        lines.append("")
    lines.append("### 계산 (예측 파일 info)\n")
    lines.append("| 변형 | 칸 | 칸당 줄 중앙 (합) | 수렴 | 칸당 초 중앙 · 최대 | 합 시간 |")
    lines.append("|---|---|---|---|---|---|")
    for n, r in res["runs"].items():
        if r:
            lines.append(f"| {n} | {r['cells']} | {r['rows_med']:g} ({r['rows_sum']}) | {r['conv']} | {r['secs_med']} · {r['secs_max']} | {r['hours']} |")
    md = "\n".join(lines) + "\n"
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)


if __name__ == "__main__":
    main()
