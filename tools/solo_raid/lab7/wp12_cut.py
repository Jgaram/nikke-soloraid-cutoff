"""WP12 곡선 꼴 조사 — 컷 곡선 E (최종 컷 곡선의 꼴)·F (시즌 중 컷 곡선이 변하나). 서술만 — 예측·채점·고르기 없음, 새 곡면 적합 없음.

  python3 -I tools/solo_raid/lab7/wp12_cut.py --obs-dir W/obs7/full --out data/solo-raid/analysis/r7/wp12/cut_D1.json

계획: docs/solo-raid-method.md 「WP12」(볼 것 E·F). 공용 표는 wp12_tables.py. 읽기만 한다 (사이트 요청 0).
컷 곡선 좌표 y = ln 점수, x = log10 %. 기울기 s(x) = −d ln 점수 / d log10 % (양수).
방법: 시즌별 P-스플라인 (Eilers·Marx 1996 — 3차 B-스플라인, 2차 차분 벌점, 벌점 GCV), 칸별 최소제곱 기울기, 시즌 붓스트랩 (F),
      함수형 주성분 (Ramsay·Silverman — 평균 뺀 행렬의 SVD).
공개 규칙: 글 번호·원문·닉네임·원글 주소·사람 열쇠·사진 열쇠를 쓰지 않는다 (시각·%·점수·개수만).
"""
import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import wp12_tables as WT  # noqa: E402

SEASONS = WT.SEASONS
SNUM = {s: int(s[1:]) for s in SEASONS}
LN10 = math.log(10.0)
GRID_PCT = 10 ** np.linspace(-1.0, math.log10(40.0), 60)        # 0.1~40%, log 등간격 60점
GX = np.log10(GRID_PCT)
BOUNDS = [0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]  # 보상 구간 경계 (조각 나누기)
EDGES = [0.1] + BOUNDS + [50.0]                                  # 바깥 조각 [0.1, 0.5]·[30, 50] 포함
KINK_Q = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
BANDS = [(0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 40.0)]
BAND_NAME = ["0.1-0.3", "0.3-1", "1-3", "3-10", "10-40"]
B_BOOT = 300
FEW = 10                                                          # 사진 수가 이보다 적은 칸은 「적음」 표시
R5 = lambda v: None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), 5)  # noqa: E731


def rl(a):
    return [R5(v) for v in np.asarray(a, float).tolist()]


def pq(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else None


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else None


# ======================= P-스플라인 =======================

def _basis(x, K, lo, hi):
    dx = (hi - lo) / (K - 3)
    t = lo + dx * np.arange(-3, K + 1)
    return t, BSpline.design_matrix(np.clip(x, t[3], t[K]), t, 3).toarray()


def pspline(x, y):
    """3차 B-스플라인 K개 (K = clip(n/2, 7, 25), 매듭 등간격, 점이 덮은 범위), 2차 차분 벌점, log10 λ ∈ [−4, 6]의 GCV 최소."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    K = int(min(25, max(7, round(n / 2))))
    lo, hi = float(x.min()), float(x.max())
    t, Bm = _basis(x, K, lo, hi)
    D = np.diff(np.eye(K), 2, axis=0)
    DD = D.T @ D
    BtB, Bty = Bm.T @ Bm, Bm.T @ y
    best = None
    for lg in np.arange(-4.0, 6.01, 0.25):
        lam = 10 ** lg
        A = BtB + lam * DD
        H = Bm @ np.linalg.solve(A, Bm.T)
        edf = float(np.trace(H))
        res = y - H @ y
        rss = float(res @ res)
        if n - edf < 1.0:
            continue
        gcv = n * rss / (n - edf) ** 2
        if best is None or gcv < best[0]:
            best = (gcv, lg, edf, rss, np.linalg.solve(A, Bty))
    gcv, lg, edf, rss, coef = best
    spl = BSpline(t, coef, 3, extrapolate=False)
    return {"spl": spl, "dspl": spl.derivative(), "K": K, "log10_lambda": lg, "edf": edf, "rss": rss,
            "sd": math.sqrt(rss / max(n - edf, 1.0)), "lo": lo, "hi": hi, "n": n}


def season_fits():
    out = {}
    for s in SEASONS:
        ep = WT.end_points(s)
        p = np.array([a for a, _ in ep])
        sc = np.array([b for _, b in ep])
        x, y = np.log10(p), np.log(sc)
        ps = pspline(x, y)
        pc = PchipInterpolator(x, y)
        m = {"ps": ps, "spl": pc, "dspl": pc.derivative(), "lo": float(x.min()), "hi": float(x.max()), "n": len(x)}
        m.update({"p": p, "sc": sc, "x": x, "y": y})
        gaps = [(float(a), float(b)) for a, b in zip(p, p[1:]) if b / a > 1.5 and b - a > 0.5]
        m["gaps"] = gaps
        out[s] = m
    return out


def fval(m, x):
    return m["spl"](np.clip(x, m["lo"], m["hi"]))


def slope_of(m, x):
    return -m["dspl"](np.clip(x, m["lo"], m["hi"]))


def covered(m, x):
    x = np.asarray(x, float)
    return (x >= m["lo"] - 1e-9) & (x <= m["hi"] + 1e-9)


# ======================= E =======================

def piece_slope(m, lo_pct, hi_pct):
    """조각 [lo, hi] (양끝 포함)의 끝 화면 점 ln 점수 ~ log10 % 최소제곱 기울기 (양수 = 줄어드는 비탈). 점 2개 이상, 점이 덮은 폭이 조각 폭의 절반 넘을 때만."""
    x = m["x"]
    sel = (x >= math.log10(lo_pct) - 1e-9) & (x <= math.log10(hi_pct) + 1e-9)
    if sel.sum() < 2:
        return None
    xs, ys = x[sel], m["y"][sel]
    w = xs.max() - xs.min()
    if w <= 0.5 * (math.log10(hi_pct) - math.log10(lo_pct)):
        return None
    return float(-np.polyfit(xs, ys, 1)[0])


def part_E(fits, RT, J, plots, notes):
    E = {"tables": {}, "notes": notes, "extra": {}}
    # ---- 1. 시즌별 곡선 (본안 = 단조 보간 PCHIP, P-스플라인은 점검)
    rows = []
    for s in SEASONS:
        m = fits[s]
        dens = np.linspace(m["lo"], m["hi"], 400)
        dprime = -slope_of(m, dens)
        viol = dprime > 1e-6
        raw_up = int(np.sum(np.diff(m["y"]) > 1e-12))
        rows.append({"season": s, "n_points": m["n"], "pct_min": R5(m["p"][0]), "pct_max": R5(m["p"][-1]),
                     "mono_viol_frac": R5(float(viol.mean())), "raw_points_rising": raw_up,
                     "gaps_pct": [[R5(a), R5(b)] for a, b in m["gaps"]], "few_points": m["n"] < 15})
    E["tables"]["season_fit"] = rows
    nv = [r["season"] for r in rows if r["mono_viol_frac"] > 0 or r["raw_points_rising"] > 0]
    notes.append(f"E1 곡선 24시즌: 끝 화면 점 {sum(r['n_points'] for r in rows)}점 (시즌 {min(r['n_points'] for r in rows)}~{max(r['n_points'] for r in rows)}), 단조 보간 PCHIP (Fritsch·Carlson 1980) — 점을 그대로 지나가므로 잔차 0. 점 15 미만 시즌 {[r['season'] for r in rows if r['few_points']]}. 단조 어김(점이 오르거나 보간이 늘어남) 시즌 {nv}.")
    # P-스플라인 점검
    pcheck = []
    for s in SEASONS:
        m = fits[s]
        ps = m["ps"]
        dens = np.linspace(m["lo"], m["hi"], 400)
        t = ps["spl"].t
        fp = ps["spl"](np.clip(dens, t[3], t[-4]))
        d = fp - fval(m, dens)
        k = int(np.argmax(np.abs(d)))
        dpr = -(-ps["dspl"](np.clip(dens, t[3], t[-4])))
        pcheck.append({"season": s, "n_points": m["n"], "K": ps["K"], "log10_lambda": ps["log10_lambda"], "edf": R5(ps["edf"]), "resid_sd_ln": R5(ps["sd"]),
                       "max_abs_diff_vs_pchip": R5(abs(d[k])), "at_pct": R5(10 ** dens[k]), "pspline_mono_viol_frac": R5(float((dpr > 1e-6).mean()))})
    nlow = [r["season"] for r in pcheck if r["log10_lambda"] <= -4.0 + 1e-9]
    E["extra"]["pspline_check"] = {"label": "점검: 시즌별 P-스플라인 (3차 B-스플라인 K=clip(n/2,7,25), 2차 차분 벌점, GCV)과 PCHIP의 차. 본안은 PCHIP.",
                                   "seasons": pcheck, "gcv_at_lower_bound": nlow,
                                   "mono_viol_seasons": [r["season"] for r in pcheck if r["pspline_mono_viol_frac"] > 0]}
    top = sorted(pcheck, key=lambda r: -r["max_abs_diff_vs_pchip"])[:4]
    notes.append(f"E1 점검: 끝 화면 점이 거의 잡음 없어 P-스플라인 GCV가 벌점 하한(log10 λ=−4)에 {len(nlow)}/24시즌 붙었다 → 평활보다 보간 문제라 PCHIP을 본안으로 하고 P-스플라인은 extra.pspline_check에 둠. "
                 f"PCHIP과의 최대 차(ln 점수): " + ", ".join(f"{r['season']} {r['max_abs_diff_vs_pchip']:.3f}(%{r['at_pct']:.2f}, 점{r['n_points']})" for r in top) +
                 f"; P-스플라인 단조 위반 시즌 {E['extra']['pspline_check']['mono_viol_seasons']}.")

    # ---- 2. 기울기 격자
    S_mat = np.full((len(SEASONS), len(GX)), np.nan)
    F_mat = np.full_like(S_mat, np.nan)
    for i, s in enumerate(SEASONS):
        m = fits[s]
        c = covered(m, GX)
        S_mat[i, c] = slope_of(m, GX[c])
        F_mat[i, c] = fval(m, GX[c])
    ncov = np.sum(np.isfinite(S_mat), axis=0)
    s_med = np.nanmedian(S_mat, axis=0)
    s_lo, s_hi = np.nanpercentile(S_mat, 10, axis=0), np.nanpercentile(S_mat, 90, axis=0)
    ok = ncov >= 8
    E["tables"]["slope_grid"] = [{"pct": R5(GRID_PCT[j]), "n_seasons": int(ncov[j]), "median": R5(s_med[j]), "p10": R5(s_lo[j]), "p90": R5(s_hi[j])}
                                 for j in range(len(GX)) if ncov[j] > 0]
    E["tables"]["slope_by_season"] = [{"season": s, "pct": rl(GRID_PCT[np.isfinite(S_mat[i])]), "slope": rl(S_mat[i][np.isfinite(S_mat[i])])} for i, s in enumerate(SEASONS)]
    for q in [0.3, 0.5, 1, 2, 3, 5, 10, 20, 30]:
        j = int(np.argmin(np.abs(GRID_PCT - q)))
        if ncov[j]:
            notes.append(f"E2 기울기 s(PCHIP) {q}% 부근({GRID_PCT[j]:.2f}%): 시즌 중앙 {s_med[j]:.3f} [10~90% {s_lo[j]:.3f}~{s_hi[j]:.3f}], 시즌 {int(ncov[j])}개.")

    # ---- 3. 보상 경계와 조각 기울기
    tier_rows = []
    for s in SEASONS:
        tier_rows.append({"season": s, "has_table_screen": len(RT[s]) > 0, "tiers_pct": RT[s]})
    E["tables"]["reward_tiers_by_season"] = tier_rows
    allp = sorted({p for s in SEASONS for p in RT[s]})
    n_tab = sum(1 for s in SEASONS if RT[s])
    seen = []
    for p in allp:
        ss = [s for s in SEASONS if p in RT[s]]
        seen.append({"pct": p, "n_seasons_seen": len(ss), "first_season": ss[0], "last_season": ss[-1], "seasons": ss})
    E["tables"]["reward_tier_union"] = seen
    notes.append(f"E3 보상 구간 표 화면이 잡힌 시즌 {n_tab}/24 (나머지 {24 - n_tab}는 「표 화면 없음」). 잡힌 시즌에서도 한두 %만 잡힌 시즌이 있다(S40 [10.0] 등) — 「목록에 없음」이 「경계 아님」은 아니다.")
    for p in (0.5, 0.7):
        e = next((x for x in seen if x["pct"] == p), None)
        if e:
            notes.append(f"E3 {p}%는 표 화면 {e['n_seasons_seen']}시즌에서 보임 (처음 {e['first_season']}, 마지막 {e['last_season']}).")
    pieces = {}
    prow = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        v = {s: piece_slope(fits[s], a, b) for s in SEASONS}
        pieces[(a, b)] = v
        vals = [x for x in v.values() if x is not None]
        prow.append({"piece_pct": [a, b], "n_seasons": len(vals), "median": R5(med(vals)), "p25": R5(pq(vals, 25)), "p75": R5(pq(vals, 75)),
                     "few": len(vals) < 8})
    E["tables"]["piece_slope"] = prow
    E["extra"]["piece_slope_by_season"] = [{"season": s, **{f"{a}-{b}": R5(pieces[(a, b)][s]) for a, b in zip(EDGES[:-1], EDGES[1:])}} for s in SEASONS]
    E["extra"]["piece_rule"] = "조각 양끝 점 포함 (경계 점은 이웃 조각 둘에 들어간다). 바깥 조각 [0.1,0.5]·[30,50] 더함 — 카드에 없는 나눔이라 extra."

    # ---- 4. 꺾임
    krow = []
    for q in KINK_Q:
        i = EDGES.index(q)
        left, right = (EDGES[i - 1], q), (q, EDGES[i + 1])
        per = {}
        for s in SEASONS:
            l, r = pieces[left][s], pieces[right][s]
            if l is not None and r is not None and l > 0:
                per[s] = r / l
        v = list(per.values())
        isb = [per[s] for s in per if q in RT[s]]
        nb = [per[s] for s in per if RT[s] and q not in RT[s]]
        nt = [per[s] for s in per if not RT[s]]
        krow.append({"q_pct": q, "left_piece": list(left), "right_piece": list(right), "n_seasons": len(v), "median": R5(med(v)), "p25": R5(pq(v, 25)), "p75": R5(pq(v, 75)),
                     "n_tier_listed": len(isb), "median_tier_listed": R5(med(isb)), "n_table_not_listed": len(nb), "median_table_not_listed": R5(med(nb)),
                     "n_no_table_screen": len(nt), "median_no_table_screen": R5(med(nt)), "few": len(v) < 8})
    E["tables"]["kink_ratio"] = krow
    for r in krow:
        notes.append(f"E4 꺾임 {r['q_pct']}% (오른쪽÷왼쪽 조각 기울기): 중앙 {r['median']} [사분위 {r['p25']}~{r['p75']}], 시즌 {r['n_seasons']}; "
                     f"경계 표에 있음 {r['n_tier_listed']}시즌 중앙 {r['median_tier_listed']} · 표 화면 있고 목록엔 없음 {r['n_table_not_listed']}시즌 중앙 {r['median_table_not_listed']} · 표 화면 없음 {r['n_no_table_screen']}시즌 중앙 {r['median_no_table_screen']}.")
    # WP10a 10 견줌 (정답 표 기울기)
    def pm(a, b):
        v = [pieces[(a, b)][s] for s in SEASONS if pieces[(a, b)][s] is not None]
        return med(v), len(v)
    w1, w3, w10, w3l = pm(1.0, 2.0), pm(3.0, 5.0), pm(10.0, 15.0), pm(2.0, 3.0)
    notes.append(f"E4 WP10a 10 견줌 (정답 표 오른쪽 기울기 1% 0.126 · 3% 0.741 · 10% 1.254, 3% 왼쪽 0.18) 대 점 조각 중앙: 1~2% {w1[0]:.3f}(n{w1[1]}) · 3~5% {w3[0]:.3f}(n{w3[1]}) · 10~15% {w10[0]:.3f}(n{w10[1]}) · 2~3% {w3l[0]:.3f}(n{w3l[1]}).")

    # ---- 5. 시즌 사이 퍼짐 (3%에서 0)
    x3 = math.log10(3.0)
    sup_lo = max(fits[s]["lo"] for s in SEASONS)
    sup_hi = min(fits[s]["hi"] for s in SEASONS)
    okg = (GX >= sup_lo - 1e-9) & (GX <= sup_hi + 1e-9)
    gx = GX[okg]
    gp = GRID_PCT[okg]
    n_all = len(SEASONS)
    Cm = np.array([fval(fits[s], gx) - fval(fits[s], x3) for s in SEASONS])
    mean_c, sd_c = Cm.mean(0), Cm.std(0, ddof=1)
    Xc = Cm - mean_c
    U, Sg, Vt = np.linalg.svd(Xc, full_matrices=False)
    share = Sg ** 2 / np.sum(Sg ** 2)
    comps, scores = [], []
    for k in range(3):
        v = Vt[k] * Sg[k] / math.sqrt(n_all - 1)       # 성분 1 sd 꼴 (ln 점수 단위)
        sc = U[:, k] * Sg[k]
        if v[-1] < 0:
            v, sc = -v, -sc
        comps.append(v)
        scores.append(sc)
    def _pca(M):
        Xm = M - M.mean(0)
        _, Sx, _ = np.linalg.svd(Xm, full_matrices=False)
        return (Sx ** 2 / np.sum(Sx ** 2))[:3]
    Cps = np.array([fits[s_]["ps"]["spl"](np.clip(gx, fits[s_]["ps"]["spl"].t[3], fits[s_]["ps"]["spl"].t[-4])) - fits[s_]["ps"]["spl"](x3) for s_ in SEASONS])
    i27 = SEASONS.index("S27")
    sh_ps, sh_no27 = _pca(Cps), _pca(np.delete(Cm, i27, axis=0))
    E["extra"]["pca_variants"] = {"label": "더 본 것/점검: 같은 격자(24시즌 공통)에서 주성분 몫 1~3", "pchip_24": rl(share[:3]), "pspline_24": rl(sh_ps), "pchip_without_S27_23": rl(sh_no27)}
    notes.append(f"E5 주성분 몫 변형 (같은 격자): PCHIP 24시즌 {share[0]:.3f}/{share[1]:.3f}/{share[2]:.3f} · P-스플라인 24시즌 {sh_ps[0]:.3f}/{sh_ps[1]:.3f}/{sh_ps[2]:.3f} · PCHIP S27 뺀 23시즌 {sh_no27[0]:.3f}/{sh_no27[1]:.3f}/{sh_no27[2]:.3f}.")
    E["tables"]["spread_grid"] = [{"pct": R5(gp[j]), "mean_c": R5(mean_c[j]), "sd_c": R5(sd_c[j]), **{f"pc{k + 1}": R5(comps[k][j]) for k in range(3)}} for j in range(len(gx))]
    E["tables"]["pca_share"] = [{"k": k + 1, "share": R5(share[k]), "cum": R5(share[:k + 1].sum())} for k in range(5)]
    E["extra"]["spread_range"] = {"pct_lo": R5(10 ** sup_lo), "pct_hi": R5(10 ** sup_hi), "n_seasons": n_all, "n_grid": int(okg.sum()),
                                  "limit_lo": SEASONS[int(np.argmax([fits[s]['lo'] for s in SEASONS]))], "limit_hi": SEASONS[int(np.argmin([fits[s]['hi'] for s in SEASONS]))]}
    notes.append(f"E5 퍼짐: 24시즌이 모두 덮는 범위 {10 ** sup_lo:.2f}~{10 ** sup_hi:.2f}% (아래 한계 {E['extra']['spread_range']['limit_lo']}, 위 한계 {E['extra']['spread_range']['limit_hi']}), 격자 {int(okg.sum())}점. "
                 f"c_s의 시즌 sd: 0.5% {np.interp(math.log10(0.5), gx, sd_c):.3f} · 1% {np.interp(0.0, gx, sd_c):.3f} · 10% {np.interp(1.0, gx, sd_c):.3f}. 평균 꼴: 1% {np.interp(0.0, gx, mean_c):.3f} · 10% {np.interp(1.0, gx, mean_c):.3f}.")
    notes.append(f"E5 주성분 몫: 1 {share[0]:.3f} · 2 {share[1]:.3f} · 3 {share[2]:.3f} (누적 {share[:3].sum():.3f}), 시즌 24.")
    # 성분 점수와 시즌 속성
    info = WT.season_info()
    snum = np.array([SNUM[s] for s in SEASONS], float)
    era = np.array([1.0 if info[s]["era"] == "new" else 0.0 for s in SEASONS])
    spc = np.array([1.0 if info[s]["special"] else 0.0 for s in SEASONS])
    lc3 = np.array([float(fval(fits[s], x3)) for s in SEASONS])
    cor = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
    crow = []
    for k in range(3):
        crow.append({"pc": k + 1, "share": R5(share[k]), "r_season_no": R5(cor(scores[k], snum)), "r_era_new": R5(cor(scores[k], era)), "r_special": R5(cor(scores[k], spc)),
                     "r_lnC3": R5(cor(scores[k], lc3)), "n_seasons": n_all, "n_special": int(spc.sum())})
    E["tables"]["pca_score_corr"] = crow
    E["extra"]["pca_scores_by_season"] = [{"season": s, "pc1": R5(scores[0][i]), "pc2": R5(scores[1][i]), "pc3": R5(scores[2][i])} for i, s in enumerate(SEASONS)]
    for r in crow:
        notes.append(f"E5 성분{r['pc']} 점수 상관 (시즌 24, 특수 3): 시즌 번호 {r['r_season_no']} · 신시대 {r['r_era_new']} · 형식 특수 {r['r_special']} · 3% 수준 {r['r_lnC3']}.")
    t3 = [{"season": s, "lnC3_spline": R5(lc3[i]), "score3_spline": R5(math.exp(lc3[i]))} for i, s in enumerate(SEASONS)]
    E["tables"]["level_3pct"] = t3
    notes.append(f"E5 3% 수준 ln 점수: 시즌 번호와 상관 {cor(lc3, snum):.3f}, 구시대(S19~S30) 평균 {lc3[era == 0].mean():.3f} (n{int((era == 0).sum())}) 대 신시대 {lc3[era == 1].mean():.3f} (n{int((era == 1).sum())}); "
                 f"최소 {SEASONS[int(np.argmin(lc3))]} {lc3.min():.3f} 최대 {SEASONS[int(np.argmax(lc3))]} {lc3.max():.3f}.")

    # ---- 6. 견줄 꼴: M3 1단계·F 모양 사전
    mrows = []
    dm_all = []
    for i, s in enumerate(SEASONS):
        mc = WT.m3_lnC(s, gp) - WT.m3_lnC(s, 3.0)
        d = mc - Cm[i]
        dm_all.append(mc)
        mrows.append({"season": s, "n_grid": len(gp), "diff_sd": R5(d.std(ddof=1)), "diff_mean": R5(d.mean()), "diff_maxabs": R5(np.abs(d).max()), "at_pct_maxabs": R5(gp[int(np.argmax(np.abs(d)))])})
    E["tables"]["m3_vs_end"] = mrows
    M3m = np.mean(dm_all, 0)
    notes.append(f"E6 M3 1단계 c_s 대 끝 화면 PCHIP c_s (24시즌, 격자 {len(gp)}점): 차 sd 중앙 {med([r['diff_sd'] for r in mrows]):.3f} (최대 {max(r['diff_sd'] for r in mrows):.3f}), "
                 f"최대 |차| 중앙 {med([r['diff_maxabs'] for r in mrows]):.3f} (최대 {max(r['diff_maxabs'] for r in mrows):.3f}); 시즌 평균 M3 꼴과 끝 화면 평균 꼴 차 sd {float(np.std(M3m - mean_c, ddof=1)):.3f}.")
    sh = WT.F_shape()
    lnp, lr = np.array(sh["lnp"]), np.array(sh["lr"])
    s37 = [s for s in SEASONS if 37 <= SNUM[s] <= 41]
    lo37, hi37 = max(fits[s]["lo"] for s in s37), min(fits[s]["hi"] for s in s37)
    g37 = GX[(GX >= lo37 - 1e-9) & (GX <= hi37 + 1e-9)]
    C37 = np.array([fval(fits[s], g37) - fval(fits[s], x3) for s in s37])
    Fv = np.interp(g37 * LN10, lnp, lr)
    d37 = Fv - C37.mean(0)
    frow = {"seasons": s37, "n_seasons": len(s37), "pct_range": [R5(10 ** g37[0]), R5(10 ** g37[-1])], "n_grid": len(g37), "diff_sd": R5(d37.std(ddof=1)), "diff_mean": R5(d37.mean()),
            "diff_maxabs": R5(np.abs(d37).max()), "at_pct_maxabs": R5(10 ** g37[int(np.argmax(np.abs(d37)))]),
            "F_at": {str(q): R5(np.interp(math.log(q), lnp, lr)) for q in (0.5, 1, 2, 5, 10)},
            "S37_41_mean_at": {str(q): R5(np.interp(math.log10(q), g37, C37.mean(0))) for q in (0.5, 1, 2, 5, 10) if g37[0] <= math.log10(q) <= g37[-1]}}
    E["tables"]["F_shape_vs_S37_41"] = [frow]
    notes.append(f"E6 F 모양 사전 (S42 모델) 대 S37~S41 끝 화면 평균 꼴 ({frow['pct_range'][0]}~{frow['pct_range'][1]}%, 시즌 5): 차 sd {frow['diff_sd']}, 최대 |차| {frow['diff_maxabs']} ({frow['at_pct_maxabs']}%). "
                 f"1%: F {frow['F_at']['1']} 대 평균 {frow['S37_41_mean_at'].get('1')}; 10%: F {frow['F_at']['10']} 대 {frow['S37_41_mean_at'].get('10')}.")

    # ---- 7. 시대 둘
    eras = {"old": [i for i, s in enumerate(SEASONS) if SNUM[s] <= 30], "new": [i for i, s in enumerate(SEASONS) if SNUM[s] >= 31]}
    erow = []
    for j in range(len(gx)):
        r = {"pct": R5(gp[j])}
        for k, ix in eras.items():
            r[f"mean_c_{k}"] = R5(Cm[ix, j].mean())
        r["diff_new_minus_old"] = R5(Cm[eras["new"], j].mean() - Cm[eras["old"], j].mean())
        erow.append(r)
    E["tables"]["era_mean_c"] = erow
    # 기울기 중앙 차이 (격자, 시대별 3개 이상 시즌이 덮는 곳)
    srow = []
    for j in range(len(GX)):
        a = S_mat[eras["old"], j]
        b = S_mat[eras["new"], j]
        na, nb = int(np.isfinite(a).sum()), int(np.isfinite(b).sum())
        if na >= 3 and nb >= 3:
            srow.append({"pct": R5(GRID_PCT[j]), "n_old": na, "n_new": nb, "median_old": R5(np.nanmedian(a)), "median_new": R5(np.nanmedian(b)), "diff_new_minus_old": R5(np.nanmedian(b) - np.nanmedian(a))})
    E["tables"]["era_slope_median"] = srow
    for q in (0.5, 1, 2, 3, 5, 10):
        r = min(srow, key=lambda z: abs(z["pct"] - q))
        j = int(np.argmin(np.abs(gp - q)))
        notes.append(f"E7 시대 {q}% 부근({r['pct']}%): 기울기 중앙 구 {r['median_old']}(n{r['n_old']}) 신 {r['median_new']}(n{r['n_new']}); c_s 평균 구 {Cm[eras['old'], j].mean():.3f} 신 {Cm[eras['new'], j].mean():.3f}.")
    rrow = []
    for e, ix in (("old S19-S30", eras["old"]), ("new S31-S42", eras["new"])):
        r1, r10, r1p, r10p = [], [], [], []
        for i in ix:
            s = SEASONS[i]
            tt = WT.truth_table(s)
            if 1.0 in tt and 3.0 in tt:
                r1.append(tt[1.0] / tt[3.0])
            if 10.0 in tt and 3.0 in tt:
                r10.append(tt[10.0] / tt[3.0])
            m = fits[s]
            if m["lo"] <= 0 and m["hi"] >= 1:
                r1p.append(math.exp(float(fval(m, 0.0) - fval(m, x3))))
            if m["hi"] >= 1.0:
                r10p.append(math.exp(float(fval(m, 1.0) - fval(m, x3))))
        rrow.append({"era": e, "n_seasons": len(ix), "C1_C3_table_mean": R5(np.mean(r1)), "n_table_1": len(r1), "C10_C3_table_mean": R5(np.mean(r10)), "n_table_10": len(r10),
                     "C1_C3_spline_mean": R5(np.mean(r1p)), "n_spline_1": len(r1p), "C10_C3_spline_mean": R5(np.mean(r10p)), "n_spline_10": len(r10p),
                     "C1_C3_table_median": R5(med(r1)), "C10_C3_table_median": R5(med(r10))})
    E["tables"]["era_ratio"] = rrow
    notes.append(f"E7 점수 비 (정답 표) C1/C3 평균: 구 {rrow[0]['C1_C3_table_mean']} (n{rrow[0]['n_table_1']}) 신 {rrow[1]['C1_C3_table_mean']} (n{rrow[1]['n_table_1']}); "
                 f"C10/C3: 구 {rrow[0]['C10_C3_table_mean']} (n{rrow[0]['n_table_10']}) 신 {rrow[1]['C10_C3_table_mean']} (n{rrow[1]['n_table_10']}).")

    # ---- 8. 확장(extra): 더 넓은 범위 (0.3~25%)를 덮는 시즌만의 평균·sd
    wlo, whi = math.log10(0.3), math.log10(25.0)
    ss = [s for s in SEASONS if fits[s]["lo"] <= wlo and fits[s]["hi"] >= whi]
    gw = GX[(GX >= wlo - 1e-9) & (GX <= whi + 1e-9)]
    Cw = np.array([fval(fits[s], gw) - fval(fits[s], x3) for s in ss])
    E["extra"]["wide_0.3_25pct"] = {"label": "더 본 것: 0.3~25%를 덮는 시즌만의 퍼짐 (본안은 24시즌 공통 범위)", "n_seasons": len(ss), "seasons": ss,
                                    "pct": rl(10 ** gw), "mean_c": rl(Cw.mean(0)), "sd_c": rl(Cw.std(0, ddof=1))}
    notes.append(f"E 더 본 것: 0.3~25%를 덮는 시즌 {len(ss)}개만 모으면 c_s 시즌 sd가 25%에서 {Cw.std(0, ddof=1)[-1]:.3f}, 0.3%에서 {Cw.std(0, ddof=1)[0]:.3f}.")

    # ---- 그림
    sel_seas = []
    for i, s in enumerate(SEASONS):
        m = fits[s]
        c = covered(m, GX)
        sel_seas.append({"name": s, "kind": "line", "x": rl(GRID_PCT[c]), "y": rl(fval(m, GX[c]) - fval(m, x3)), "group": "season"})
    Ms = {"name": "M3 1단계 (시즌 평균)", "kind": "line", "x": rl(gp), "y": rl(M3m), "group": "M3"}
    Fs = {"name": "F 모양 사전 (S42 모델)", "kind": "line", "x": rl(np.exp(lnp[(lnp >= math.log(gp[0])) & (lnp <= math.log(gp[-1]))])), "y": rl(lr[(lnp >= math.log(gp[0])) & (lnp <= math.log(gp[-1]))]), "group": "F"}
    meanS = {"name": f"끝 화면 평균 (24시즌, {10 ** sup_lo:.2f}~{10 ** sup_hi:.2f}%)", "kind": "line", "x": rl(gp), "y": rl(mean_c), "group": "data"}
    plots.append({"id": "E1", "part": "E", "title": "시즌별 최종 컷 꼴 (3%를 0으로 맞춘 ln 점수)", "x": {"label": "최종 % (log)", "log": True, "reverse": False}, "y": {"label": "ln 점수 − ln 점수(3%)", "log": False},
                  "series": sel_seas + [meanS, Ms, Fs], "note": "끝 화면 점 PCHIP 보간, 시즌 24 · 점 1,281. 가는 선 = 시즌, 시즌마다 점이 덮은 범위만."})
    okS = ncov >= 8
    plots.append({"id": "E2", "part": "E", "title": "컷 곡선 기울기 s(x) — 시즌 중앙과 10~90%", "x": {"label": "최종 % (log)", "log": True, "reverse": False}, "y": {"label": "s = −d ln 점수 / d log10 %", "log": False},
                  "series": [{"name": "시즌 10~90%", "kind": "band", "x": rl(GRID_PCT[okS]), "y": rl(s_med[okS]), "lo": rl(s_lo[okS]), "hi": rl(s_hi[okS]), "group": "data"},
                             {"name": "시즌 중앙", "kind": "line", "x": rl(GRID_PCT[okS]), "y": rl(s_med[okS]), "group": "data"}],
                  "note": "시즌 8개 이상이 덮은 격자만. 보상 구간 경계 x 위치 (세로선 대신): " + ", ".join(str(b) for b in BOUNDS) + "%."})
    plots.append({"id": "E3", "part": "E", "title": "시즌 사이 퍼짐의 주성분 꼴 (1 sd)", "x": {"label": "최종 % (log)", "log": True, "reverse": False}, "y": {"label": "ln 점수 변화 (3%에서 0)", "log": False},
                  "series": [{"name": f"성분 {k + 1} (몫 {share[k]:.2f})", "kind": "line", "x": rl(gp), "y": rl(comps[k]), "group": "data"} for k in range(2)] +
                            [{"name": "시즌 sd", "kind": "line", "x": rl(gp), "y": rl(sd_c), "group": "M3"}],
                  "note": f"함수형 주성분 (평균 뺀 c_s 행렬 SVD), 시즌 24, 범위 {10 ** sup_lo:.2f}~{10 ** sup_hi:.2f}%. 성분 1~3 몫 {share[0]:.3f}/{share[1]:.3f}/{share[2]:.3f}."})
    return E, fits


# ======================= F =======================

def wmedian(v, w):
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    tot = cw[-1]
    if tot <= 0:
        return float("nan")
    k = int(np.searchsorted(cw, 0.5 * tot - 1e-12))
    if abs(cw[k] - 0.5 * tot) < 1e-9 and k + 1 < len(v):
        return 0.5 * (v[k] + v[k + 1])
    return float(v[k])


def prep_photos(obs_dir, kinds, ptkey="P_tab"):
    info = WT.season_info()
    out = []
    for r in WT.photos(obs_dir):
        if r["kind"] not in kinds or r[ptkey] is None:
            continue
        s = r["season"]
        H = info[s]["H"]
        h = r["h"]
        if h is None or h < 0 or h > H + 1e-9:
            continue
        d = int(WT.day_of(s, h))
        out.append({"s": s, "kind": r["kind"], "h": h, "p": r["pct"], "score": r["score"], "P": r[ptkey], "pf": r["pf_tab" if ptkey == "P_tab" else "pf_pts"], "day": d})
    return out


def band_of(p):
    for i, (a, b) in enumerate(BANDS):
        if a <= p < b or (i == len(BANDS) - 1 and p == b):
            return i
    return None


def day_center(s, d):
    info = WT.season_info()[s]
    off = (info["c0h"] - 5.0) % 24.0
    a, b = max(0.0, 24 * (d - 1) - off), min(info["H"], 24 * d - off)
    return 0.5 * (a + b)


def add_win(rows, wtype):
    info = WT.season_info()
    for r in rows:
        H = info[r["s"]]["H"]
        r["u"] = r["h"] / H
        if wtype == "6h":
            r["win"] = int(r["h"] // 6)
            r["hc"] = 6 * r["win"] + 3.0
        else:
            r["win"] = r["day"]
            r["hc"] = day_center(r["s"], r["day"])
        r["shift"] = float(WT.F_P(r["hc"] / H) - WT.F_P(r["u"]))      # P_obs + shift = 창 가운데로 옮긴 P
        r["Ps"] = r["P"] + r["shift"]
        r["lpc"] = math.log10(r["p"]) - r["shift"]                    # 창 가운데 시각으로 옮긴 log10 p (분리 꼴: log p_c = log p + F_P(h) − F_P(h_c))
    return rows


def window_table(rows, wtype, rng, boot=True):
    seasons = sorted({r["s"] for r in rows})
    sidx = {s: i for i, s in enumerate(seasons)}
    wins = sorted({r["win"] for r in rows})
    out = []
    cnts = []
    if boot:
        for _ in range(B_BOOT):
            c = np.bincount(rng.integers(0, len(seasons), len(seasons)), minlength=len(seasons)).astype(float)
            cnts.append(c)
    for w in wins:
        rw = [r for r in rows if r["win"] == w]
        s_arr = np.array([sidx[r["s"]] for r in rw])
        band = np.array([band_of(r["p"]) if band_of(r["p"]) is not None else -1 for r in rw])
        hc = float(np.median([r["hc"] for r in rw]))
        for key in ("Ps", "P"):
            v = np.array([r[key] for r in rw])
            wm_all = wmedian(v, np.ones(len(v)))
            dd = {}
            for b in range(len(BANDS)):
                m = band == b
                if not m.any():
                    dd[b] = None
                    continue
                pt = wmedian(v[m], np.ones(m.sum())) - wm_all
                lo = hi = None
                if boot:
                    bs = []
                    for c in cnts:
                        wt = c[s_arr]
                        if wt[m].sum() == 0 or wt.sum() == 0:
                            continue
                        bs.append(wmedian(v[m], wt[m]) - wmedian(v, wt))
                    if len(bs) >= 30:
                        lo, hi = float(np.percentile(bs, 10)), float(np.percentile(bs, 90))
                dd[b] = (pt, lo, hi, int(m.sum()), len(set(s_arr[m].tolist())))
            for b, z in dd.items():
                row = next((x for x in out if x["window"] == w and x["band"] == BAND_NAME[b]), None)
                if row is None:
                    row = {"window_type": wtype, "window": int(w), "center_h": R5(hc), "band": BAND_NAME[b], "n": 0, "n_seasons": 0}
                    out.append(row)
                if z is None:
                    continue
                row["n"], row["n_seasons"] = z[3], z[4]
                row["few"] = z[3] < FEW
                if key == "Ps":
                    row["diff_shifted"], row["p10_shifted"], row["p90_shifted"] = R5(z[0]), R5(z[1]), R5(z[2])
                else:
                    row["diff_unshifted"], row["p10_unshifted"], row["p90_unshifted"] = R5(z[0]), R5(z[1]), R5(z[2])
            out.append({"window_type": wtype, "window": int(w), "center_h": R5(hc), "band": "all", "n": len(rw), "n_seasons": len(set(s_arr.tolist())),
                        f"median_P_{'shifted' if key == 'Ps' else 'unshifted'}": R5(wm_all)})
    # 같은 (window, band 'all') 두 줄 합치기
    merged = {}
    for r in out:
        k = (r["window"], r["band"])
        merged.setdefault(k, {}).update(r)
    return [merged[k] for k in sorted(merged, key=lambda z: (z[0], BAND_NAME.index(z[1]) if z[1] in BAND_NAME else 99))]


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm = x.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        return None
    b = float(((x - xm) * (y - y.mean())).sum() / sxx)
    return b, float(y.mean() - b * xm)


def cross_section(rows, fits, wtype_day=True):
    """시즌·일차 창마다 단면 기울기 (ln 점수 ~ 옮긴 log10 p) 대 같은 시즌 최종 곡선 평균 기울기."""
    res = []
    keys = sorted({(r["s"], r["win"]) for r in rows})
    for s, d in keys:
        rw = [r for r in rows if r["s"] == s and r["win"] == d]
        if len(rw) < 8:
            continue
        for sh in (True, False):
            x = np.array([r["lpc"] if sh else math.log10(r["p"]) for r in rw])
            if x.max() - x.min() < 0.5:
                continue
            y = np.log([r["score"] for r in rw])
            fit = ols(x, y)
            if fit is None:
                continue
            m = fits[s]
            lpf = np.log10([r["pf"] for r in rw])
            lo, hi = max(lpf.min(), m["lo"]), min(lpf.max(), m["hi"])
            if hi - lo <= 0.05:
                continue
            sfin = -(float(fval(m, hi)) - float(fval(m, lo))) / (hi - lo)
            fit2 = ols(lpf, y)
            res.append({"s": s, "day": d, "shifted": sh, "n": len(rw), "spread": float(x.max() - x.min()), "slope_cross": -fit[0], "slope_final": sfin,
                        "slope_final_matched": -fit2[0] if fit2 else None, "ratio": (-fit[0]) / sfin if sfin > 0.02 else None,
                        "ratio_matched": (-fit[0]) / (-fit2[0]) if fit2 and -fit2[0] > 0.02 else None})
    return res


def local_fit(rw, q, hw=0.3, minside=2):
    lq = math.log10(q)
    pts = [(r["lpc"], math.log(r["score"])) for r in rw if abs(r["lpc"] - lq) <= hw]
    below = sum(1 for x, _ in pts if x < lq)
    above = sum(1 for x, _ in pts if x > lq)
    if below < minside or above < minside:
        return None
    x = np.array([a for a, _ in pts])
    y = np.array([b for _, b in pts])
    f = ols(x, y)
    return None if f is None else f[0] * lq + f[1]


def ratio_flow(rows, fits, hw=0.3):
    """창마다 관측 비 C_q/C3 (국소 선형) · 분리 꼴 기대 비 E_q = Cfin(q·10^P̄)/Cfin(3·10^P̄) (Cfin = 그 시즌 E의 PCHIP 곡선, P̄ = 창 A·B 사진의
    옮긴 P_obs 중앙 — 사진 8장 미만이면 F_P(창 가운데 h/H)) · 최종 비 (정답 표)."""
    info = WT.season_info()
    res, cover = [], {}
    keys = sorted({(r["s"], r["win"]) for r in rows})
    for s, d in keys:
        rw = [r for r in rows if r["s"] == s and r["win"] == d]
        tt = WT.truth_table(s)
        m = fits[s]
        f1, f3, f10 = local_fit(rw, 1.0, hw), local_fit(rw, 3.0, hw), local_fit(rw, 10.0, hw)
        if len(rw) >= 8:
            pbar, src = float(np.median([r["Ps"] for r in rw])), "photos"
        else:
            pbar, src = float(WT.F_P(rw[0]["hc"] / info[s]["H"])), "F_P"
        c = cover.setdefault(d, {"n_windows": 0, "c1_c3": 0, "c10_c3": 0, "no_truth": 0, "no_cover": 0})
        c["n_windows"] += 1
        for q, f in ((1.0, f1), (10.0, f10)):
            if f is None or f3 is None:
                continue
            if q not in tt or 3.0 not in tt:
                c["no_truth"] += 1
                continue
            xq, x3 = math.log10(q) + pbar, math.log10(3.0) + pbar
            if not (m["lo"] <= xq <= m["hi"] and m["lo"] <= x3 <= m["hi"]):
                c["no_cover"] += 1
                continue
            fin = math.log(tt[q] / tt[3.0])
            exp_ = float(fval(m, xq) - fval(m, x3))
            obs = f - f3
            res.append({"s": s, "day": d, "q": q, "ratio_over_final": math.exp(obs - fin), "ln_diff": obs - fin, "obs": math.exp(obs), "exp": math.exp(exp_),
                        "obs_over_exp": math.exp(obs - exp_), "pbar": pbar, "pbar_src": src})
            c["c1_c3" if q == 1.0 else "c10_c3"] += 1
    return res, cover


def part_F(obs_dir, fits, J, plots, notes):
    info = WT.season_info()
    F = {"tables": {}, "notes": notes, "extra": {}}
    rng = np.random.default_rng(12)
    ab = prep_photos(obs_dir, ("A", "B"))
    main = [r for r in ab if info[r["s"]]["clock_ok"]]
    t0 = time.time()
    for wt, label in (("6h", "F1_win6h"), ("day", "F1_winday")):
        rows = add_win([dict(r) for r in main], wt)
        F["tables"][label] = window_table(rows, wt, rng)
    print("F 창 표 (본안)", round(time.time() - t0, 1), "초", flush=True)
    F["extra"]["seasons_main"] = sorted({r["s"] for r in main})
    F["extra"]["photo_counts"] = {"AB_with_pf_tab_all_seasons": len(ab), "AB_main_clock_ok_120h_05h": len(main),
                                  "main_seasons": len({r["s"] for r in main}), "AB_with_pf_pts": len(prep_photos(obs_dir, ("A", "B"), "P_pts"))}
    # 자동 notes: 어느 %대·창에서 갈라지나
    for label, nm in (("F1_win6h", "6시간 창"), ("F1_winday", "일차 창")):
        tb = [r for r in F["tables"][label] if r["band"] != "all" and r.get("diff_shifted") is not None]
        full = [r for r in tb if r["n"] >= FEW]
        sig = [r for r in full if r["p10_shifted"] is not None and (r["p10_shifted"] > 0 or r["p90_shifted"] < 0)]
        mx = max(full, key=lambda r: abs(r["diff_shifted"])) if full else None
        ad = np.array([abs(r["diff_shifted"]) for r in full])
        notes.append(f"F1 {nm} (옮긴 판, 칸 {len(full)}개: 사진 {FEW}장 이상): |차| 중앙 {np.median(ad):.3f}, 최대 {ad.max():.3f} ({mx['band']}% · 창 가운데 h {mx['center_h']} · 사진 {mx['n']} · 시즌 {mx['n_seasons']}); "
                     f"시즌 붓스트랩 10~90%가 0을 벗어난 칸 {len(sig)}개.")
        for b in BAND_NAME:
            bb = [r for r in tb if r["band"] == b]
            few = [r["center_h"] for r in bb if r["n"] < FEW]
            allw = sorted({r["center_h"] for r in F["tables"][label] if r["band"] == "all"})
            miss = [h for h in allw if h not in [r["center_h"] for r in bb]]
            ss = [r for r in bb if r["n"] >= FEW and r["p10_shifted"] is not None and (r["p10_shifted"] > 0 or r["p90_shifted"] < 0)]
            notes.append(f"F1 {nm} {b}%: 사진 있는 창 {len(bb)}/{len(allw)} (비는 창 {len(miss)}, 사진 {FEW}장 미만 {len(few)}); 평균 차 {np.mean([r['diff_shifted'] for r in bb if r['n'] >= FEW]) if any(r['n'] >= FEW for r in bb) else float('nan'):+.3f}; "
                         f"0 벗어난 창 {[(r['center_h'], r['diff_shifted']) for r in ss][:8]}")
    # 옮김 안 한 판과 옮긴 판의 차이 한 줄
    t6 = [r for r in F["tables"]["F1_win6h"] if r["band"] != "all" and r["n"] >= FEW and r.get("diff_unshifted") is not None and r.get("diff_shifted") is not None]
    notes.append(f"F1 6시간 창에서 옮긴 판과 안 옮긴 판의 칸 차 |차| 평균 {np.mean([abs(r['diff_shifted'] - r['diff_unshifted']) for r in t6]):.4f} (칸 {len(t6)}개) — 창 안 시각 차를 옮겨도 거의 안 변한다." if t6 else "")

    # ---- extra: 05시 시작 120시간이 아닌 시즌 포함 h 축 그대로
    allrows = add_win([dict(r) for r in ab], "6h")
    F["extra"]["F1_win6h_all_seasons_h_axis"] = {"label": "더 본 것: 나머지 시즌(S19·S28·S38) 포함, h 축 그대로 6시간 창 (본안은 05시 시작 120시간 시즌)", "table": window_table(allrows, "6h", rng)}
    # C 참고, P_pts 참고
    cref = add_win([dict(r) for r in prep_photos(obs_dir, ("C",)) if info[r["s"]]["clock_ok"]], "6h")
    F["extra"]["F1_win6h_C_reference"] = {"label": "참고: C 사진 (hp = 게시 시각, 촬영의 상한) 6시간 창, 붓스트랩 없음", "table": window_table(cref, "6h", rng, boot=False)}
    ptsr = add_win([dict(r) for r in prep_photos(obs_dir, ("A", "B"), "P_pts") if info[r["s"]]["clock_ok"]], "6h")
    tp = window_table(ptsr, "6h", rng, boot=False)
    F["extra"]["F1_win6h_P_pts"] = {"label": "참고: 사진 P를 끝 화면 점 곡선(pf_pts)으로 잰 판, 붓스트랩 없음", "table": tp}
    cmp_ = []
    tbm = {(r["window"], r["band"]): r for r in F["tables"]["F1_win6h"]}
    for r in tp:
        k = (r["window"], r["band"])
        if r["band"] != "all" and r["n"] >= FEW and k in tbm and tbm[k].get("diff_shifted") is not None and r.get("diff_shifted") is not None and tbm[k]["n"] >= FEW:
            cmp_.append(abs(r["diff_shifted"] - tbm[k]["diff_shifted"]))
    notes.append(f"F1 P_pts판(끝 화면 점 곡선으로 잰 최종 %) 6시간 창: 사진 {len(ptsr)}장 (P_tab판 {len(main)}장), P_tab판과 칸 차이 |차| 평균 {np.mean(cmp_):.4f} (칸 {len(cmp_)}개)." if cmp_ else "F1 P_pts판 비교 칸 없음.")

    # ---- 단면 기울기 비
    dayrows = add_win([dict(r) for r in main], "day")
    cs = cross_section(dayrows, fits)
    dayall = add_win([dict(r) for r in ab], "day")
    cs_all = cross_section(dayall, fits)
    srows = []
    for sh in (True, False):
        for d in range(1, 7):
            v = [r for r in cs if r["day"] == d and r["shifted"] == sh and r["ratio"] is not None]
            vm = [r["ratio_matched"] for r in v if r["ratio_matched"] is not None]
            if not v:
                continue
            srows.append({"day": d, "shifted": sh, "n_windows": len(v), "n_seasons": len({r["s"] for r in v}), "median_ratio": R5(med([r["ratio"] for r in v])),
                          "p25": R5(pq([r["ratio"] for r in v], 25)), "p75": R5(pq([r["ratio"] for r in v], 75)),
                          "median_ratio_matched_final": R5(med(vm)), "median_slope_cross": R5(med([r["slope_cross"] for r in v])), "median_slope_final": R5(med([r["slope_final"] for r in v])),
                          "median_n_photos": int(med([r["n"] for r in v])), "few": len(v) < 5})
    F["tables"]["F2_cross_section_slope_ratio"] = srows
    F["extra"]["F2_all_seasons_h_axis"] = {"label": "더 본 것: S19·S28·S38 포함 (일차는 KST 05시 경계, 그 시즌 창 가운데)",
                                           "n_windows": len([r for r in cs_all if r["shifted"] and r["ratio"] is not None]),
                                           "median_ratio": R5(med([r["ratio"] for r in cs_all if r["shifted"] and r["ratio"] is not None]))}
    F["extra"]["F2_rule"] = ("창 = 시즌×일차, A·B 사진(pf_tab 있음) 8장 이상, 옮긴 log10 p 퍼짐 ≥ 0.5 decade. 단면 = ln 점수 ~ 옮긴 log10 p 최소제곱 기울기(양수 표기), "
                             "최종 = E의 PCHIP 곡선 평균 기울기 (사진들의 최종 % 범위를 곡선 범위로 자른 구간). matched = 같은 사진의 ln 점수 ~ log10 최종 %(pf_tab) 기울기와의 비.")
    for d in range(1, 6):
        z = next((r for r in srows if r["day"] == d and r["shifted"]), None)
        if z:
            notes.append(f"F2 {d}일차 단면÷최종 기울기: 중앙 {z['median_ratio']} [사분위 {z['p25']}~{z['p75']}], 창 {z['n_windows']}개 (시즌 {z['n_seasons']}, 창당 사진 중앙 {z['median_n_photos']}장); 같은 사진 최종% 기준 중앙 {z['median_ratio_matched_final']}.")
    ns_all = len([r for r in cs if r["shifted"]])
    notes.append(f"F2 조건(사진 8장 이상·퍼짐 0.5 decade 이상)을 채운 창 {ns_all}개 (본안 시즌).")

    # ---- C1/C3, C10/C3 시간 흐름
    rf, cover = ratio_flow(dayrows, fits)
    rfall, _ = ratio_flow(dayall, fits)
    rrows = []
    for q in (1.0, 10.0):
        for d in range(1, 7):
            v = [r for r in rf if r["q"] == q and r["day"] == d]
            cv = cover.get(d)
            if cv is None:
                continue
            g = lambda k: [r[k] for r in v]  # noqa: E731
            rrows.append({"q_pct": q, "day": d, "n_windows_day": cv["n_windows"], "n_windows_ratio": len(v), "n_seasons": len({r["s"] for r in v}),
                          "n_pbar_from_F_P": sum(r["pbar_src"] == "F_P" for r in v),
                          "obs_median": R5(med(g("obs"))), "exp_median": R5(med(g("exp"))),
                          "obs_over_exp_median": R5(med(g("obs_over_exp"))), "obs_over_exp_p25": R5(pq(g("obs_over_exp"), 25)), "obs_over_exp_p75": R5(pq(g("obs_over_exp"), 75)),
                          "median": R5(med(g("ratio_over_final"))), "p25": R5(pq(g("ratio_over_final"), 25)), "p75": R5(pq(g("ratio_over_final"), 75)),
                          "median_ln": R5(med(g("ln_diff"))), "empty": len(v) == 0, "few": 0 < len(v) < 5})
    F["tables"]["F3_ratio_over_final"] = rrows
    F["extra"]["F3_rule"] = "창 = 시즌×일차(본안 시즌), A·B 사진(pf_tab 있음), 옮긴 log10 p가 목표 %(1·3·10) ±0.3 decade 안인 사진으로 ln 점수 ~ log10 p 국소 최소제곱, 목표 위·아래 각 2장 이상. 관측 비 = 그 국소 적합의 C_q/C3. 기대 비 E_q = C_fin,s(q·10^P̄)/C_fin,s(3·10^P̄) (분리 꼴이 맞을 때의 비, C_fin = 그 시즌 E PCHIP 곡선, P̄ = 창 A·B 사진 옮긴 P_obs 중앙, 사진 8장 미만이면 F_P(창 가운데 h/H) — n_pbar_from_F_P에 개수), 곡선이 q·10^P̄ 또는 3·10^P̄를 덮지 않는 창은 뺌. 본선 = 관측 ÷ 기대. 「÷ 최종 비」(정답 표 C_q/C3)는 참고 — 분리 꼴이어도 1이 아니다."
    F["extra"]["F3_all_seasons_h_axis"] = {"label": "더 본 것: S19·S28·S38 포함", "n_C1": len([r for r in rfall if r["q"] == 1.0]), "n_C10": len([r for r in rfall if r["q"] == 10.0]),
                                           "median_C1": R5(med([r["ratio_over_final"] for r in rfall if r["q"] == 1.0])), "median_C10": R5(med([r["ratio_over_final"] for r in rfall if r["q"] == 10.0]))}
    for q, nm in ((1.0, "C1/C3"), (10.0, "C10/C3")):
        zz = [r for r in rrows if r["q_pct"] == q]
        txt = " · ".join(f"{r['day']}일차 관측 {r['obs_median']} 기대 {r['exp_median']} 관측÷기대 {r['obs_over_exp_median']} [{r['obs_over_exp_p25']}~{r['obs_over_exp_p75']}] (÷최종 {r['median']}; 창{r['n_windows_ratio']}/{r['n_windows_day']}, 시즌{r['n_seasons']})"
                         if not r["empty"] else f"{r['day']}일차 비었음(창{r['n_windows_day']})" for r in zz)
        notes.append(f"F3 {nm} 일차별 중앙 (본선 = 관측÷분리 꼴 기대, ÷최종은 참고): {txt}.")
    notes.append("F3 정정: 분리 꼴이어도 시각 h의 컷 비는 C_fin(q·10^P)/C_fin(3·10^P)라 최종 C_q/C3와 다르다 (최종 곡선이 3% 위에서 가팔라 초반에 크게 나옴). 그래서 「÷ 최종」은 분리 꼴 점검이 못 되고 관측÷기대가 점검이다.")
    notes.append("F2 「같은 사진을 최종 %로 잰 기준」(median_ratio_matched_final) = 같은 창·같은 사진의 (ln 점수 ~ 옮긴 log10 지금 %) 기울기 ÷ (ln 점수 ~ log10 최종 % pf_tab) 기울기, 창별 비의 일차 중앙. 분모가 E 곡선 평균 기울기가 아니라 그 사진들의 최종 % 곡선 그대로의 기울기다.")

    # ---- 종합 note (어느 %대·어느 때)
    for label, nm in (("F1_win6h", "6h"), ("F1_winday", "일차")):
        tb = [r for r in F["tables"][label] if r["band"] != "all" and r.get("diff_shifted") is not None and r["n"] >= FEW and r["p10_shifted"] is not None and (r["p10_shifted"] > 0 or r["p90_shifted"] < 0)]
        tb.sort(key=lambda r: -abs(r["diff_shifted"]))
        notes.append(f"F 종합 {nm} 창에서 분리 꼴과 가장 다른 칸 (사진 {FEW}장 이상, 붓스트랩이 0을 벗어남): " + "; ".join(f"{r['band']}% h{r['center_h']} {r['diff_shifted']:+.3f} (n{r['n']}, 시즌{r['n_seasons']})" for r in tb[:8]))

    # ---- 그림
    colors = BAND_NAME
    for pid, label, title, xl in (("F1", "F1_win6h", "창 × %대 P_obs 차 (6시간 창, 옮긴 판)", "창 가운데 h (시작부터 시간)"), ("F1d", "F1_winday", "창 × %대 P_obs 차 (일차 창, 옮긴 판)", "창 가운데 h (시작부터 시간)")):
        ser = []
        for b in BAND_NAME:
            rr = [r for r in F["tables"][label] if r["band"] == b and r.get("diff_shifted") is not None and r["n"] >= FEW]
            if not rr:
                continue
            ser.append({"name": f"{b}% (사진 {FEW}장 이상 창)", "kind": "line", "x": [r["center_h"] for r in rr], "y": [r["diff_shifted"] for r in rr], "group": "data"})
            lo_ok = [r for r in rr if r["p10_shifted"] is not None]
            if lo_ok:
                ser.append({"name": f"{b}% 10~90%", "kind": "band", "x": [r["center_h"] for r in lo_ok], "y": [r["diff_shifted"] for r in lo_ok], "lo": [r["p10_shifted"] for r in lo_ok], "hi": [r["p90_shifted"] for r in lo_ok], "group": "data"})
        plots.append({"id": pid, "part": "F", "title": title, "x": {"label": xl, "log": False, "reverse": False}, "y": {"label": "%대 중앙 − 창 전체 중앙 (log10 P_obs)", "log": False},
                      "series": ser, "note": f"A·B 사진, 한섭·미상, 05시 시작 120시간 시즌 {len({r['s'] for r in main})}개, 사진 {len(main)}장 (pf_tab 0.5~15% 안). 띠 = 시즌 붓스트랩 10~90% (B={B_BOOT}). 사진 {FEW}장 미만 칸은 그리지 않음."})
    ser = []
    for sh, nm in ((True, "옮긴 판"), (False, "안 옮긴 판")):
        rr = [r for r in srows if r["shifted"] == sh and not r["few"]]
        if rr:
            ser.append({"name": f"단면÷최종 중앙 ({nm})", "kind": "line", "x": [r["day"] for r in rr], "y": [r["median_ratio"] for r in rr], "group": "data"})
    rr = [r for r in srows if r["shifted"] and not r["few"]]
    if rr:
        ser.append({"name": "사분위 (옮긴 판)", "kind": "band", "x": [r["day"] for r in rr], "y": [r["median_ratio"] for r in rr], "lo": [r["p25"] for r in rr], "hi": [r["p75"] for r in rr], "group": "data"})
    pts = [r for r in cs if r["shifted"] and r["ratio"] is not None and r["ratio"] < 20]
    ser.append({"name": "창 하나씩 (옮긴 판)", "kind": "points", "x": [r["day"] for r in pts], "y": [R5(r["ratio"]) for r in pts], "group": "season"})
    plots.append({"id": "F2", "part": "F", "title": "단면 기울기 ÷ 최종 곡선 평균 기울기 (일차별)", "x": {"label": "일차", "log": False, "reverse": False}, "y": {"label": "비 (1 = 분리 꼴)", "log": False}, "series": ser,
                  "note": f"창 = 시즌×일차 (A·B 8장 이상·퍼짐 0.5 decade 이상), 창 {len(pts)}개 (비 20 미만만 점으로). 창 5개 미만 일차는 선에서 뺌."})
    ser = []
    for q, nm in ((1.0, "C1/C3"), (10.0, "C10/C3")):
        rr = [r for r in rrows if r["q_pct"] == q and not r["empty"]]
        if rr:
            ser.append({"name": f"{nm} 관측÷기대 중앙", "kind": "line", "x": [r["day"] for r in rr], "y": [r["obs_over_exp_median"] for r in rr], "group": "data"})
            ser.append({"name": f"{nm} 관측÷기대 사분위", "kind": "band", "x": [r["day"] for r in rr], "y": [r["obs_over_exp_median"] for r in rr], "lo": [r["obs_over_exp_p25"] for r in rr], "hi": [r["obs_over_exp_p75"] for r in rr], "group": "data"})
            ser.append({"name": f"{nm} 관측÷최종 (참고, 옅게)", "kind": "line", "x": [r["day"] for r in rr], "y": [r["median"] for r in rr], "group": "M3"})
    plots.append({"id": "F3", "part": "F", "title": "C1/C3·C10/C3 관측 ÷ 분리 꼴 기대 (일차별)", "x": {"label": "일차", "log": False, "reverse": False}, "y": {"label": "비 (1 = 분리 꼴)", "log": False}, "series": ser,
                  "note": "기대 = 시즌 최종 곡선을 P̄만큼 민 비. 옅은 선 = 관측 ÷ 최종 비 (참고, 분리 꼴이어도 1이 아님). 일차별 창 수·시즌 수는 표 F3_ratio_over_final."})
    return F


def git_head():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    fits = season_fits()
    RT = WT.reward_tiers(a.obs_dir)
    plots, nE, nF = [], [], []
    E, fits = part_E(fits, RT, None, plots, nE)
    print("E 끝", round(time.time() - t0, 1), "초", flush=True)
    F = part_F(a.obs_dir, fits, None, plots, nF)
    rules = {
        "시즌 곡선": "끝 점이 거의 잡음 없어 GCV가 하한에 붙음 → 단조 보간 PCHIP (Fritsch·Carlson 1980, scipy PchipInterpolator, (log10 %, ln 점수)) 본안. 기울기 s = PCHIP 미분. P-스플라인(3차 B-스플라인 K=clip(round(n/2),7,25), 2차 차분 벌점, log10 λ∈[−4,6] 0.25 간격 GCV)은 extra.pspline_check 점검.",
        "격자": "0.1~40% log 등간격 60점. 시즌 점이 덮은 범위(첫~마지막 점) 밖은 비움. 점 사이 빈 구간(이웃 % 비 1.5 초과 그리고 차 0.5 초과)은 막지 않고 시즌 표에 적음.",
        "조각 기울기": "경계 [0.5,0.7,1,2,3,5,10,15,20,30] + 바깥 [0.1,0.5]·[30,50], 양끝 포함, 점 2개 이상·덮은 폭 > 조각 폭 절반.",
        "주성분": "평균 뺀 c_s 행렬의 SVD (Ramsay·Silverman), 성분 꼴 = 1 sd 점수 × 단위 방향, 부호는 격자 끝 값이 양수가 되게.",
        "F 창": "A·B 사진, 한섭·미상, pf_tab 있음, 05시 시작 120시간 시즌 본안(S19·S28·S38 제외). F_P로 창 가운데로 옮김. %대 = 지금 %.",
        "붓스트랩": f"시즌 복원 추출 B={B_BOOT}, 시드 12, 10~90% 백분위. 칸 값 = 가중 중앙값 차(%대 중앙 − 창 전체 중앙).",
        "적은 칸": f"사진 {FEW}장 미만 칸 `few`; 끝 화면 점 15 미만 시즌 `few_points`.",
    }
    J = {"header": {"about": "WP12 컷 곡선 E(최종 컷 곡선 꼴)·F(시즌 중 컷 곡선이 변하나 — 분리 꼴 점검). 서술만 — 예측·채점·고르기 없음", "data_ver": C.DATA_VER,
                    "date": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d"), "code_head": git_head(), "secs": round(time.time() - t0, 1), "rules": rules},
         "E": E, "F": F, "plots": plots}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=1)
    print("쓴 파일", a.out, os.path.getsize(a.out), "바이트", round(time.time() - t0, 1), "초")


if __name__ == "__main__":
    main()
