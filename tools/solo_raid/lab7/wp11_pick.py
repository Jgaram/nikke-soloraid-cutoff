"""WP11b 고르기·보고 (2026-11-03) — 조정 6시즌 2×2 (pin·actpin·base·act)를 계획에 미리 적은 규칙으로 가린다.

  python3 -I tools/solo_raid/lab7/wp11_pick.py --diag data/solo-raid/analysis/r7/wp11/diag11b_D1.json \
      --pred pin=r7/wp11/pred_pin_tune_D1.json --pred actpin=… --pred base=… --pred act=… --ref r7/wp9/F_tune_D1.json \
      --out data/solo-raid/analysis/r7/wp11/pick11b_D1.json --md data/solo-raid/analysis/r7/wp11/diag11b_D1.md

규칙 (docs/solo-raid-method.md 「WP11b」, 계산 전에 고정): g5(조정 6시즌, 1·3%·10% 각각 같은 칸 F보다 score 0.3%p 넘게 나쁘지 않음)를 지나는 것 가운데
score 1·3%가 가장 낮은 것. 0.3%p 안이면 단순한 쪽 pin < actpin < base < act (--pred 순서). base가 고르면 WP11c 없이 「결론 부족」.
지나는 것이 없으면 F와의 차(1·3%·10% 중 큰 쪽)가 가장 작은 것을 적는다.
보고만: g1·g4(diag의 보정 구간 — emp_loso 어림), 조정 시즌 중 최근 셋(S34·S38·S42) score, 2×2로 가른 몫, T별 |e|·부호 평균, 수렴 비율·칸당 시간.
score.py가 낸 diag(--g5 --calib emp_loso)와 같은 예측 파일을 받는다.
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

NEAR = SC.NEAR
RECENT3 = ("S34", "S38", "S42")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--pred", action="append", required=True, help="이름=경로, 순서 = 단순한 쪽 먼저")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--lb", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "data/solo-raid/analysis/r7/lb_D1.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    D = SC.jl(a.diag)
    order = [p.split("=", 1)[0] for p in a.pred]
    paths = dict(p.split("=", 1) for p in a.pred)
    dnames = [n for n in D["groups"]["1·3%"]["cands"]]          # score.py 이름 (파일 순서와 같다)
    assert len(dnames) == len(order), (dnames, order)
    nm = dict(zip(order, dnames))
    truth = SC.A.truth()
    _, lb = SC.load_lb(a.lb)
    preds = {n: SC.load_pred(paths[n]) for n in order}
    ref = SC.load_pred(a.ref)
    res = {"order": order, "rule": "g5(조정 6시즌) 지나는 것 중 score 1·3% 최소, 0.3%p 안이면 앞선 쪽 (pin < actpin < base < act)", "cands": {}}
    for n in order:
        r = {}
        for g, qs in SC.GROUPS:
            G = D["groups"][g]
            c = G["cands"][nm[n]]
            F = G["F"]
            rec = SC.evaluate(preds[n]["cells"], truth, lb, qs, SC.TOL, seasons=set(RECENT3))
            recF = SC.evaluate(ref["cells"], truth, lb, qs, SC.TOL, seasons=set(RECENT3))
            byT = {k: {"mae": round(v["mae"] * 100, 2), "bias": round(v["bias"] * 100, 2)} for k, v in c["byT"].items() if k in ("18", "24", "36", "48", "H-12", "H-6", "H")}
            r[g] = {"score": c["score"], "F": F["score"], "g5_diff": c["score"] - F["score"], "main": c["main"], "end": c["end"],
                    "g1": c["g1"], "cov_main": c["cov_main"], "cov_end": c["cov_end"], "cov_model_main": c.get("cov_model_main"), "cov_model_end": c.get("cov_model_end"),
                    "recent3": rec["score"], "recent3_F": recF["score"], "byT": byT}
        info = preds[n].get("info", [])
        r["conv"] = f"{sum(1 for i in info if i.get('conv'))}/{len(info)}" if info else None
        r["secs_median"] = round(st.median([i["secs"] for i in info]), 1) if info else None
        r["g5_pass"] = all(r[g]["g5_diff"] <= NEAR + 1e-12 for g, _ in SC.GROUPS)
        res["cands"][n] = r
    ok = [n for n in order if res["cands"][n]["g5_pass"]]
    if ok:
        best = min(res["cands"][n]["1·3%"]["score"] for n in ok)
        pick = next(n for n in order if n in ok and res["cands"][n]["1·3%"]["score"] <= best + NEAR + 1e-12)
        res["pick"] = pick
        res["verdict"] = "base — WP11c 없이 「결론 부족」 (계획)" if pick == "base" else f"{pick} — WP11c로"
    else:
        gap = {n: max(res["cands"][n][g]["g5_diff"] for g, _ in SC.GROUPS) for n in order}
        res["pick"] = None
        res["closest"] = min(gap, key=gap.get)
        res["closest_gap"] = gap[res["closest"]]
        res["verdict"] = f"지나는 것 없음 — 가장 가까운 {res['closest']} (F와의 차 큰 쪽 {gap[res['closest']] * 100:+.2f}%p)"
    # 2×2 몫 (score, 무리별): act 몫 = ½[(act−base) + (actpin−pin)], pin 몫 = ½[(pin−base) + (actpin−act)], 겹침 = actpin − act − pin + base
    if all(n in res["cands"] for n in ("base", "act", "pin", "actpin")):
        two = {}
        for g, _ in SC.GROUPS:
            s = {n: res["cands"][n][g]["score"] for n in ("base", "act", "pin", "actpin")}
            two[g] = {"act": 0.5 * ((s["act"] - s["base"]) + (s["actpin"] - s["pin"])), "pin": 0.5 * ((s["pin"] - s["base"]) + (s["actpin"] - s["act"])),
                      "inter": s["actpin"] - s["act"] - s["pin"] + s["base"]}
        res["two_by_two"] = two
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    p = lambda x: "-" if x is None else f"{x * 100:.2f}"  # noqa: E731
    L = ["# WP11b 조정 6시즌 2×2 — 채점·고르기 (자료 판 D1, 121칸)", "", f"**판정: {res['verdict']}**", "",
         "| 변형 | 1·3% score | F | 차 | 10% score | F | 차 | g5 | 1·3% 초반/막판 | g1 (1·3%·10%) | 보정 포함 1·3% (초반/막판) | 모형 포함 | 최근 셋 1·3%·10% (F) | 수렴 | 칸당 초 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in order:
        r = res["cands"][n]
        a1, a10 = r["1·3%"], r["10%"]
        L.append(f"| {n} | {p(a1['score'])} | {p(a1['F'])} | {a1['g5_diff'] * 100:+.2f} | {p(a10['score'])} | {p(a10['F'])} | {a10['g5_diff'] * 100:+.2f} | {'지남' if r['g5_pass'] else '못 지남'} | "
                 f"{p(a1['main'])}/{p(a1['end'])} | {a1['g1']}·{a10['g1']} | {a1['cov_main']:.2f}/{a1['cov_end']:.2f} | {p(a1['cov_model_main'])}/{p(a1['cov_model_end'])} | "
                 f"{p(a1['recent3'])}·{p(a10['recent3'])} ({p(a1['recent3_F'])}·{p(a10['recent3_F'])}) | {r['conv']} | {r['secs_median']} |")
    if "two_by_two" in res:
        L += ["", "2×2로 가른 몫 (score %p, 음수 = 나아짐): " + " · ".join(f"{g} 활동 시간 {v['act'] * 100:+.2f} · 손잡이 줄이기 {v['pin'] * 100:+.2f} · 겹침 {v['inter'] * 100:+.2f}" for g, v in res["two_by_two"].items())]
    L += ["", "T별 |e| · 부호 평균 (%, 1·3%)", "", "| 변형 | " + " | ".join(("18", "24", "36", "48", "H-12", "H-6", "H")) + " |", "|---|" + "---|" * 7]
    for n in order:
        b = res["cands"][n]["1·3%"]["byT"]
        L.append(f"| {n} | " + " | ".join(f"{b[k]['mae']:.2f} ({b[k]['bias']:+.2f})" if k in b else "-" for k in ("18", "24", "36", "48", "H-12", "H-6", "H")) + " |")
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
