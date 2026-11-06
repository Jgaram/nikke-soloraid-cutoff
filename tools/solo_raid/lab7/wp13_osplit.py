"""WP13b 진단 나눔 O1·O2·O3 (2026-11-04) — 예측 파일 여럿을 같은 칸으로 맞춰 창·T·%·시즌별 |e|와 차를 낸다. 맞춤 없음, 다 낸 예측 파일만 읽는다.

  python3 -I tools/solo_raid/lab7/wp13_osplit.py --pred O1=A.json --pred O2=B.json [--pred F=F.json ...] [--diff O2-O1 ...] [--shape r7/wp13/shape_D1.json] --out OUT.json   (+ OUT.md)

계획: docs/solo-raid-method.md 「WP13b 진단 먼저」 — 셋을 창 (초반 T 18~48 · 막판 H−12~H) · T · % (1·3·10%) · 시즌별로, 첫날 (T 6~24)과 마지막 날 17시 (H−12)는 따로.
  e = 중앙 ÷ 정답 − 1 (score.py와 같은 정답 · KR 칸만). 표마다 평균 |e|와 부호 평균 e, 칸 수. 이름들이 모두 가진 칸만 쓴다 (같은 칸끼리 견줌).
  --diff A-B: 같은 칸의 평균 |e| 차 (A − B, %p). O2 − O1 = 수준 추정 몫, O2 − O3 = 이번 시즌 자료가 고친 몫.
  --shape를 주면 shape 예측 파일 info의 ln m 사후 중앙 − 참 수준 (꼴 파일 m_true, 진단용)을 창별로 낸다 (수준 추정이 얼마나 빗나가나).
공개 규칙: 시즌·T·%·점수만 (글·사람·사진 열쇠 없음).
"""
import argparse
import collections
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as SC  # noqa: E402

A = SC.A
PCTS = (1.0, 3.0, 10.0)
GROUPS = (("1·3%", (1.0, 3.0)), ("10%", (10.0,)), ("1%", (1.0,)), ("3%", (3.0,)))


def slot(s, T):
    """칸이 드는 묶음들: main (T 18~48), end (H−12~H), day1 (T 6~24), h12 (H−12)."""
    H = A.info(s)[1]
    out = []
    if T in SC.MAIN_T:
        out.append("main")
    if T in (H - 12, H - 6, H):
        out.append("end")
    if 6 <= T <= 24:
        out.append("day1")
    if T == H - 12:
        out.append("h12")
    return out


def summ(es):
    if not es:
        return None
    return {"n": len(es), "abs": round(st.mean(abs(e) for e in es), 5), "sgn": round(st.mean(es), 5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", action="append", required=True, help="이름=예측 파일")
    ap.add_argument("--diff", action="append", default=[], help="A-B (이름)")
    ap.add_argument("--shape", default=None, help="꼴 파일 (m_true 대조용)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    truth = A.truth()
    names, cells = [], {}
    info = {}
    for x in a.pred:
        n, p = x.split("=", 1)
        names.append(n)
        d = SC.load_pred(p)
        cells[n] = {(c["season"], c["T"], c["pct"]): c["pred"] for c in d["cells"] if c.get("server", "KR") == "KR" and c["pct"] in PCTS}
        info[n] = {(i["season"], float(i["T"])): i.get("shape_info") for i in SC.jl(p).get("info", [])}
    keys = set.intersection(*[set(v) for v in cells.values()])
    keys = sorted(k for k in keys if truth.get((k[0], "KR", k[2])) is not None)
    if not keys:
        sys.exit("같은 칸이 없다")
    E = {n: {k: cells[n][k] / truth[(k[0], "KR", k[2])] - 1 for k in keys} for n in names}
    out = {"about": "WP13b O 나눔 (lab7/wp13_osplit.py)", "names": names, "n_cells": len(keys),
           "seasons": sorted({k[0] for k in keys}, key=SC.snum), "groups": {}}
    for g, qs in GROUPS:
        G = {"slots": {}, "by_T": {}, "by_season": {}}
        for sl in ("main", "end", "day1", "h12"):
            G["slots"][sl] = {n: summ([E[n][k] for k in keys if k[2] in qs and sl in slot(k[0], k[1])]) for n in names}
        tk = sorted({SC.tkey(k[0], k[1]) for k in keys}, key=lambda t: (t.startswith("H"), float(t) if not t.startswith("H") else -float(t[2:] or 0)))
        for t in tk:
            G["by_T"][t] = {n: summ([E[n][k] for k in keys if k[2] in qs and SC.tkey(k[0], k[1]) == t]) for n in names}
        for s in out["seasons"]:
            G["by_season"][s] = {sl: {n: summ([E[n][k] for k in keys if k[0] == s and k[2] in qs and sl in slot(k[0], k[1])]) for n in names} for sl in ("main", "end")}
        dd = {}
        for df in a.diff:
            x, y = df.split("-", 1)
            dd[df] = {sl: (round(G["slots"][sl][x]["abs"] - G["slots"][sl][y]["abs"], 5) if G["slots"][sl][x] and G["slots"][sl][y] else None) for sl in G["slots"]}
        G["diff"] = dd
        out["groups"][g] = G
    if a.shape:
        sh = SC.jl(a.shape)["targets"]
        lv = {}
        for n in names:
            rows = collections.defaultdict(list)
            for (s, T), si in info[n].items():
                if not si or "lnm" not in si or s not in sh:
                    continue
                mt = sh[s]["flood"].get("m_true", {}).get("lnm")
                if mt is None:
                    continue
                for sl in slot(s, T) + ["all"]:
                    rows[sl].append(si["lnm"]["med"] - mt)
            if rows:
                lv[n] = {sl: {"n": len(v), "mean": round(st.mean(v), 4), "mean_abs": round(st.mean(abs(x) for x in v), 4)} for sl, v in rows.items()}
        out["level_vs_true"] = lv
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    md = [f"# WP13b O 나눔 — {', '.join(names)}", "", f"같은 칸 {len(keys)}개 · 시즌 {', '.join(out['seasons'])}. 값 = 평균 |e| (%, 부호 평균 e). 차는 %p.", ""]
    P = lambda v: "—" if v is None else f"{v['abs'] * 100:.2f} ({v['sgn'] * 100:+.2f})"  # noqa: E731
    for g, G in out["groups"].items():
        md += [f"## {g}", "", "| 묶음 | " + " | ".join(names) + " | " + " | ".join(a.diff) + " |", "|---" * (1 + len(names) + len(a.diff)) + "|"]
        for sl, lab in (("main", "초반 T18~48"), ("end", "막판 H−12~H"), ("day1", "첫날 T6~24"), ("h12", "H−12")):
            md.append(f"| {lab} | " + " | ".join(P(G["slots"][sl][n]) for n in names) + " | " + " | ".join("—" if G["diff"][d][sl] is None else f"{G['diff'][d][sl] * 100:+.2f}" for d in a.diff) + " |")
        md += ["", "T별: " + " · ".join(f"{t} " + "/".join(f"{G['by_T'][t][n]['abs'] * 100:.1f}" if G['by_T'][t][n] else "—" for n in names) for t in G["by_T"]), ""]
        md += ["| 시즌 | " + " | ".join(f"{n} 초반·막판" for n in names) + " |", "|---" * (1 + len(names)) + "|"]
        for s, v in G["by_season"].items():
            md.append(f"| {s} | " + " | ".join(f"{(v['main'][n] or {}).get('abs', float('nan')) * 100:.2f} · {(v['end'][n] or {}).get('abs', float('nan')) * 100:.2f}" for n in names) + " |")
        md.append("")
    if out.get("level_vs_true"):
        md += ["## ln m 사후 중앙 − 참 수준 (꼴 파일 m_true)", "", "| 이름 | 전체 | 초반 | 막판 | 첫날 |", "|---|---|---|---|---|"]
        for n, v in out["level_vs_true"].items():
            md.append(f"| {n} | " + " | ".join(f"{v[sl]['mean']:+.3f} (|·| {v[sl]['mean_abs']:.3f}, n {v[sl]['n']})" if sl in v else "—" for sl in ("all", "main", "end", "day1")) + " |")
    with open(os.path.splitext(a.out)[0] + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(a.out, len(keys), "칸")


if __name__ == "__main__":
    main()
