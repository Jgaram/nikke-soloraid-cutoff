"""WP11a 4.3 점검 보고 (2026-11-03) — 컷 사진 잔차의 KST 시각·일차 표: M3 기본판(h) 대 M3 `act` 1단계 곡선. 보고만.

  python3 -I tools/solo_raid/lab7/wp11_resid.py --obs-dir W/obs7/full --act-stage1 r7/wp11/stage1_M3act_D1.json \
      --out data/solo-raid/analysis/r7/wp11/resid_D1.json --md data/solo-raid/analysis/r7/wp11/resid_D1.md

wp10_diag.part_curve_photo와 같은 셈 (컷 사진마다 실제 남은 침수 log10(최종 %/지금 %) − 곡선 P, 정답 표 보간 — 정답을 본 진단):
  h:   P(h/H) · M3 기본판 1단계 (r7/stage1_M3_D1.json)
  act: P(u*(h)) · M3 act 1단계 (그 시즌 g_s)
05시 시작 120시간 시즌 (wp10_diag와 같은 칸) 표와, 시작 시각이 다른 S28·S38을 넣은 표를 함께 낸다. 6 ≤ h < H−2, 최종 % 0.5~15.
보는 것: 3~4일차 11~17시와 23~05시의 차가 줄었는가 (WP10a: 기본판 +0.017 대 −0.019 log10), 일차별로 남는 꼴.
"""
import argparse
import collections
import json
import math
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_table as OT  # noqa: E402
import wp10_diag as WD  # noqa: E402
import act_clock as AC  # noqa: E402

R7 = WD.R7


def curve_par(path, s, m3):
    v = WD.jload(path)
    if s not in v["seasons"]:
        return None
    nm = v["header"]["names"]
    l12 = np.array([v["seasons"][s]["mean"][nm.index(f"l{i}")] for i in range(m3.NL)])
    return np.linalg.solve(m3.BASIS["Bl12"], l12), v["seasons"][s]["mean"][nm.index("beta")]


def Pm3(m3, cl, u):
    B = m3.BASIS
    sg = B["sg"]
    f = np.exp(B["Blf"] @ cl) * 2 * sg
    seg = 0.5 * (f[1:] + f[:-1]) * B["ds"]
    Pg = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return float(np.interp(math.sqrt(max(u, 0.02)), sg, Pg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--act-stage1", required=True)
    ap.add_argument("--profile", default=os.path.join(R7, "wp11/act_profile_D1.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    import m3
    prof = AC.load(a.profile)
    S1 = {"h": os.path.join(R7, "stage1_M3_D1.json"), "act": a.act_stage1}
    tru = collections.defaultdict(dict)
    for (s, q), v in WD.TRUTH.items():
        tru[s][q] = v
    res = {"c05": collections.defaultdict(lambda: collections.defaultdict(list)), "all": collections.defaultdict(lambda: collections.defaultdict(list))}
    for n in range(21, 43):
        s = f"S{n}"
        if len(tru[s]) < 4:
            continue
        par = {k: curve_par(p, s, m3) for k, p in S1.items()}
        if any(v is None for v in par.values()):
            continue
        head, rows = OT.read_cell(os.path.join(a.obs_dir, f"{s}.jsonl.gz"))
        H = float(head["H"])
        c0h = prof["g"][s]["c0h"]
        for r in rows:
            if r["kind"] not in ("A", "B", "C") or not r.get("use") or not r.get("pct") or not r.get("score") or r.get("hp") is None:
                continue
            h = r["hp"]
            if r.get("server") not in (None, "KR") or not (6 <= h < H - 2):
                continue
            pf = WD.final_pct(tru[s], r["score"])
            if not pf or not (0.5 <= pf <= 15) or pf < r["pct"] * 0.98:
                continue
            act = math.log10(pf / r["pct"])
            clk = (c0h + h) % 24
            w = "05-11" if 5 <= clk < 11 else "11-17" if 11 <= clk < 17 else "17-23" if 17 <= clk < 23 else "23-05"
            day = "d2" if h < 48 else "d3-4" if h < 96 else "d5"
            uu = {"h": h / H, "act": float(AC.ustar(prof, s, [h], H)[0])}
            rr = {k: act - Pm3(m3, cl, uu[k]) * (pf / 3) ** (beta / math.log(10)) for k, (cl, beta) in par.items()}
            for part in (("all", "c05") if (abs(c0h - 5) < 1e-6 and H == 120) else ("all",)):
                for key in ("all", w, day, f"{day}|{w}"):
                    for k, v in rr.items():
                        res[part][key][k].append(v)
    order = lambda k: (k != "all", k.count("|"), k)  # noqa: E731
    tab = {part: {k: dict(n=len(v["h"]), **{m: round(st.median(x), 4) for m, x in v.items()}) for k, v in sorted(R.items(), key=lambda kv: order(kv[0]))}
           for part, R in res.items()}
    gap = {}
    for part in tab:
        t = tab[part]
        gap[part] = {m: (round(t["d3-4|11-17"][m] - t["d3-4|23-05"][m], 4) if "d3-4|11-17" in t and "d3-4|23-05" in t else None) for m in ("h", "act")}
    out = {"about": "WP11a 컷 사진 잔차 (log10, 실제 남은 침수 − 곡선). 정답 표를 본 진단 — 보고만", "table": tab, "gap_d34_11_17_minus_23_05": gap,
           "note": "c05 = 05시 시작 120시간 시즌 (wp10_diag와 같은 칸), all = S28·S38 포함. 잔차 > 0 = 곡선이 남은 침수를 작게 잡음"}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    L = ["# WP11a 컷 사진 잔차 — 시계 시간(h) 대 활동 시간(act) (2026-11-03, 보고만)", "",
         "실제 남은 침수 log10(최종 %/지금 %) − 곡선 P의 중앙. 정답 표를 본 진단. h = M3 기본판 1단계 · act = M3 act 1단계.", ""]
    for part, nm in (("c05", "05시 시작 120시간 시즌"), ("all", "모든 시즌 (S28·S38 포함)")):
        L += [f"## {nm}", "", f"3~4일차 11~17시 − 23~05시: h {gap[part]['h']} · act {gap[part]['act']}", "", "| 칸 | n | h | act |", "|---|---|---|---|"]
        for k, v in tab[part].items():
            L.append(f"| {k} | {v['n']} | {v['h']:+.4f} | {v['act']:+.4f} |")
        L.append("")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(json.dumps(gap, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
