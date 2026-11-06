"""WP12 보고 뒤 더 본 것 (2026-11-04, 사용자 질문 「꼴의 모수도 시즌마다 바뀌나」): 시대별 (S19~S30 옛 · S31~S42 새) 합친 침수 곡선.
WP12 G1과 같은 부품 (wp12_flood_curve), κ는 21시즌 값 고정, 시대 안 시즌 붓스트랩 150. 하루 물결 꼴이 시대마다 다른가.
python3 -W ignore -I tools/solo_raid/lab7/wp12_era.py OBS_DIR . data/solo-raid/analysis/r7/wp12/era_D1.json   # 약 1분"""
import json
import sys

import numpy as np

OBS, REPO = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO + "/tools/solo_raid/lab7")
import wp12_flood_curve as FC  # noqa: E402

KAP = 0.316228
B = 150
O, PH, prof, s1m, s1a = FC.build_obs(OBS)
GRID = FC.GRID


def era_fit(seasons, seed):
    msk = np.isin(O["S"], [FC.SIDX[s] for s in seasons])
    Oe = FC.sub(O, msk)
    eidx = {s: i for i, s in enumerate(seasons)}
    Oe["si"] = np.array([eidx[FC.SEAS[i]] for i in Oe["S"]])
    ns = len(seasons)
    X = FC.design(FC.HH, Oe["ta"], Oe["tb"])
    DtD = FC.d2pen(X.shape[1])
    _, _, _, sig, _ = FC.fit_full(X, Oe["y"], Oe["wp"], Oe["isph"], Oe["si"], ns, 10.0, DtD, FC.Sg(0.03, 1.0), rounds=2)
    beta, m, r, sig, W0 = FC.fit_full(X, Oe["y"], Oe["wp"], Oe["isph"], Oe["si"], ns, KAP, DtD, sig, rounds=3)
    lam, Pc = FC.curves_from_beta(beta, FC.HH, GRID)
    rng = np.random.default_rng(seed)
    by = [np.where(Oe["si"] == j)[0] for j in range(ns)]
    bl = []
    for b in range(B):
        dr = rng.integers(0, ns, ns)
        rows = [by[d] for d in dr]
        idx = np.concatenate(rows)
        sj = np.concatenate([np.full(len(rr), j) for j, rr in enumerate(rows)])
        W0b = FC.W0of(Oe["isph"][idx], Oe["wp"][idx], sig)
        bb, _, _ = FC.fit_spl(X[idx], Oe["y"][idx], W0b, Oe["isph"][idx], sj, ns, KAP, DtD, sig, iters=10)
        bl.append(FC.curves_from_beta(bb, FC.HH, GRID)[0])
    return lam, np.array(bl), int((~Oe["isph"]).sum()), int(Oe["isph"].sum())


def shape_stats(lam):
    """2~4일차 (h 24~96) 같은 KST 시각 평균 → 하루 안 몫 (3시간 칸 8개, 05시부터) · 가장 빠른·느린 시각 · 비 · 일차 합."""
    m = (GRID >= 24 - 1e-9) & (GRID < 96 - 1e-9)
    seg = lam[m].reshape(3, 96).mean(axis=0)          # 0.25시간 × 96 = 하루
    sm = np.convolve(np.r_[seg[-4:], seg, seg[:4]], np.ones(9) / 9, mode="same")[4:-4]
    tot = seg.sum()
    share = [float(seg[12 * k:12 * k + 12].sum() / tot) for k in range(8)]
    kst = (5 + np.arange(96) * 0.25) % 24
    day = [float(np.trapezoid(lam[(GRID >= 24 * d) & (GRID <= 24 * d + 24)], GRID[(GRID >= 24 * d) & (GRID <= 24 * d + 24)])) for d in range(1, 5)]
    return {"share3h": share, "fast_kst": float(kst[np.argmax(sm)]), "slow_kst": float(kst[np.argmin(sm)]), "ratio": float(sm.max() / max(sm.min(), 1e-9)),
            "share_17_23": share[4] + share[5], "share_23_05": share[6] + share[7], "share_05_17": sum(share[:4]), "day_sums_2_5": day}


old = [s for s in FC.CLOCK if int(s[1:]) <= 30]
new = [s for s in FC.CLOCK if int(s[1:]) >= 31]
res = {}
for name, ss, sd in (("old", old, 11), ("new", new, 12)):
    lam, bl, npair, nph = era_fit(ss, sd)
    st = shape_stats(lam)
    bs = [shape_stats(x) for x in bl]
    band = {k: [float(np.percentile([b[k] for b in bs], 10)), float(np.percentile([b[k] for b in bs], 90))] for k in ("share_17_23", "share_23_05", "share_05_17", "ratio", "fast_kst", "slow_kst")}
    band["share3h"] = [[float(np.percentile([b["share3h"][k] for b in bs], q)) for q in (10, 90)] for k in range(8)]
    res[name] = {"seasons": ss, "pairs": npair, "photos": nph, "stat": st, "band": band, "_boot": bs}
# 차이 (새 − 옛) 붓스트랩: 두 시대를 따로 뽑아 짝
rng = np.random.default_rng(5)
for k in ("share_17_23", "share_23_05", "share_05_17"):
    d = np.array([res["new"]["_boot"][i][k] - res["old"]["_boot"][j][k] for i, j in zip(rng.integers(0, B, 2000), rng.integers(0, B, 2000))])
    print(f"diff new-old {k}: {res['new']['stat'][k] - res['old']['stat'][k]:+.3f}  10~90% [{np.percentile(d, 10):+.3f}, {np.percentile(d, 90):+.3f}]  P(diff<0) {np.mean(d < 0):.2f}")
for name in ("old", "new"):
    r = res[name]
    st, bd = r["stat"], r["band"]
    print(f"== {name} {len(r['seasons'])}시즌 쌍 {r['pairs']} 사진 {r['photos']}")
    print("  3h 몫 (05~):", [f"{v:.3f}" for v in st["share3h"]])
    print("  band       :", [f"{a:.2f}-{b:.2f}" for a, b in bd["share3h"]])
    print(f"  05~17 {st['share_05_17']:.3f} {bd['share_05_17']} · 17~23 {st['share_17_23']:.3f} {bd['share_17_23']} · 23~05 {st['share_23_05']:.3f} {bd['share_23_05']}")
    print(f"  가장 빠름 KST {st['fast_kst']:.2f} {bd['fast_kst']} · 가장 느림 {st['slow_kst']:.2f} {bd['slow_kst']} · 비 {st['ratio']:.2f} {bd['ratio']}")
    print(f"  2~5일차 합 {[round(x, 3) for x in st['day_sums_2_5']]}")
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "_boot"} for k, v in res.items()}, open(sys.argv[3] if len(sys.argv) > 3 else "/dev/null", "w"), ensure_ascii=False)
