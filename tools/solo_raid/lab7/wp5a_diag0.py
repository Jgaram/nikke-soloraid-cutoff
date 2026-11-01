"""WP5a 진단 0 — M3 예측 오차를 「1단계 곡선을 못 맞힌 몫」과 「1단계 목표 어긋남」으로 가른다
(docs/solo-raid-method.md 「WP3b M3 결과」). 새 적합 없이 이미 있는 파일만 읽는다.

  python3 -I tools/solo_raid/lab7/wp5a_diag0.py

입력 (읽기만): r7/stage1_M3_D1.json (1단계 24시즌, lnC_q10_50_90), r7/pred_M3_tune_D1.json (조정 6시즌 × T 격자),
              analysis/answer_key.json (정답). 시즌 길이 H와 창 정의는 score.py를 그대로 따른다.
출력: r7/wp5a/diag0_D1.json, r7/wp5a/diag0_D1.md

(A) 목표 어긋남 d = ln 정답 − 1단계 ln 컷 중앙 (24시즌 전체, 정답 있는 한섭 칸), z = d / sd1단계.
(B) 조정 칸 오차: e_tot = ln pred − ln 정답, e_fut = ln pred − 1단계 중앙, e_tgt = 1단계 중앙 − ln 정답 (= −d).
    s_pred = (ln i80[1] − ln i80[0]) / 2.5631. 숫자만 쓴다 (글·닉네임 없음).
"""
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, HERE)
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "lab6"))
import score as SC  # noqa: E402
import anchor as A  # noqa: E402

R7 = os.path.join(ROOT, "data", "solo-raid", "analysis", "r7")
OUT = os.path.join(R7, "wp5a")
Z80 = 1.2815515655446004
Z2 = 2.5631031310892007          # q90 − q10 = 2 × 1.2816 sd
PCTS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
GROUPS = (("1·3%", (1.0, 3.0)), ("10%", (10.0,)))
GROUPS_JSON = GROUPS + (("0.5·2·5%", (0.5, 2.0, 5.0)),)
WINS = ("main", "end", "other", "all")
EXPAND_FROM = "S21"


def r(x, n=5):
    return None if x is None else round(x, n)


def mean(v):
    return st.mean(v) if v else None


def rms(v):
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else None


def sd(v):
    return st.stdev(v) if len(v) > 1 else None


def med(v):
    return st.median(v) if v else None


def quant(v, p):
    """선형 보간 분위 (numpy 기본과 같음)."""
    s = sorted(v)
    if not s:
        return None
    k = (len(s) - 1) * p
    lo = int(math.floor(k))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def snum(s):
    return int(s[1:])


# ---------- 입력 ----------
stage1 = SC.jl(os.path.join(R7, "stage1_M3_D1.json"))["seasons"]
pred = SC.jl(os.path.join(R7, "pred_M3_tune_D1.json"))["cells"]
truth = A.truth()


def s1(s, q):
    """(ln 중앙, sd1단계)."""
    lo, m, hi = stage1[s]["lnC_q10_50_90"][f"{q:g}"]
    return m, (hi - lo) / Z2


# ---------- (A) ----------
def a_stats(rows):
    d = [x["d"] for x in rows]
    z = [x["z"] for x in rows]
    return dict(n=len(d), mean_d=r(mean(d)), sd_d=r(sd(d)), rms_d=r(rms(d)), med_abs_d=r(med([abs(x) for x in d])),
                med_abs_z=r(med([abs(x) for x in z])), pos_share=r(mean([1.0 if x > 0 else 0.0 for x in d]), 3))


A_rows = []
for s in sorted(stage1, key=snum):
    for q in PCTS:
        tv = truth.get((s, "KR", q))
        if tv is None:
            continue
        m, sg = s1(s, q)
        d = math.log(tv) - m
        A_rows.append(dict(season=s, pct=q, d=d, sd1=sg, z=d / sg))

A_by_pct = {f"{q:g}": a_stats([x for x in A_rows if x["pct"] == q]) for q in PCTS}
A_by_group = {g: a_stats([x for x in A_rows if x["pct"] in qs]) for g, qs in GROUPS_JSON}

# 시즌 순서로 늘려 가는 창: s보다 앞 시즌들의 d
seasons_all = sorted(stage1, key=snum)
A_expand = {}
for g, qs in GROUPS:
    rows = []
    for s in seasons_all:
        if snum(s) < snum(EXPAND_FROM):
            continue
        prev = [x["d"] for x in A_rows if x["pct"] in qs and snum(x["season"]) < snum(s)]
        cur = [x["d"] for x in A_rows if x["pct"] in qs and x["season"] == s]
        rows.append(dict(season=s, n_prev=len(prev), mean_prev=r(mean(prev)), rms_prev=r(rms(prev)),
                         sd_prev=r(sd(prev)), d_this=r(mean(cur)), n_this=len(cur)))
    A_expand[g] = rows

# 부호 쏠림: 시즌별 d 평균(그 묶음)의 부호, 이항 양쪽 p, 시즌 번호와의 상관
def binom_p(k, n):
    if n == 0:
        return None
    k = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def corr(x, y):
    if len(x) < 3:
        return None
    mx, my = mean(x), mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


A_sign = {}
A_season_tab = {}
for g, qs in GROUPS:
    per = []
    for s in seasons_all:
        v = [x["d"] for x in A_rows if x["season"] == s and x["pct"] in qs]
        if v:
            per.append((s, mean(v)))
    pos = sum(1 for _, m in per if m > 0)
    half = len(per) // 2
    A_sign[g] = dict(n_seasons=len(per), n_pos=pos, binom_p=r(binom_p(pos, len(per)), 4),
                     corr_with_season=r(corr([snum(s) for s, _ in per], [m for _, m in per]), 3),
                     mean_first_half=r(mean([m for _, m in per[:half]])), mean_second_half=r(mean([m for _, m in per[half:]])))
    A_season_tab[g] = {s: r(m) for s, m in per}

# ---------- (B) ----------
B_cells = []
conv_false = 0
for c in pred:
    if c.get("server", "KR") != "KR":
        continue
    s, T, q = c["season"], float(c["T"]), float(c["pct"])
    tv = truth.get((s, "KR", q))
    if tv is None or not c.get("pred") or not c.get("i80") or None in c["i80"]:
        continue
    m, sg = s1(s, q)
    lo, hi = c["i80"]
    sp = (math.log(hi) - math.log(lo)) / Z2
    lp, lt = math.log(c["pred"]), math.log(tv)
    w = SC.window(s, T) or "other"
    B_cells.append(dict(season=s, T=T, pct=q, win=w, conv=bool(c.get("conv", True)), e_tot=lp - lt, e_fut=lp - m,
                        e_tgt=m - lt, s_pred=sp, in_truth=lo <= tv <= hi, in_s1=lo <= math.exp(m) <= hi))
    if not c.get("conv", True):
        conv_false += 1


def covers(rows, sig_fn):
    return mean([1.0 if abs(x["e_tot"]) <= Z80 * sig_fn(x) else 0.0 for x in rows])


def need_factors(rows):
    z = [abs(x["e_tot"] / x["s_pred"]) for x in rows]
    c = quant(z, 0.8) / Z80
    add = None
    for i in range(0, 201):
        a = i * 0.0005
        if covers(rows, lambda x: math.sqrt(x["s_pred"] ** 2 + a * a)) >= 0.80 - 1e-12:
            add = a
            break
    return c, add


def b_stats(rows):
    if not rows:
        return dict(n=0)
    zf = [x["e_fut"] / x["s_pred"] for x in rows]
    zt = [x["e_tot"] / x["s_pred"] for x in rows]
    c, add = need_factors(rows)
    return dict(n=len(rows), n_conv_false=sum(1 for x in rows if not x["conv"]),
                rms_tot=r(rms([x["e_tot"] for x in rows])), rms_fut=r(rms([x["e_fut"] for x in rows])),
                rms_tgt=r(rms([x["e_tgt"] for x in rows])),
                bias_tot=r(mean([x["e_tot"] for x in rows])), bias_fut=r(mean([x["e_fut"] for x in rows])),
                bias_tgt=r(mean([x["e_tgt"] for x in rows])),
                cov_truth=r(mean([1.0 if x["in_truth"] else 0.0 for x in rows]), 3),
                cov_s1=r(mean([1.0 if x["in_s1"] else 0.0 for x in rows]), 3),
                med_s_pred=r(med([x["s_pred"] for x in rows])),
                sd_z_fut=r(sd(zf), 3), med_abs_z_fut=r(med([abs(x) for x in zf]), 3), sd_z_tot=r(sd(zt), 3),
                need_c=r(c, 3), need_sigma_add=add if add is None else r(add, 4))


B_tab = {}
for w in WINS:
    for g, qs in GROUPS_JSON:
        rows = [x for x in B_cells if x["pct"] in qs and (w == "all" or x["win"] == w)]
        B_tab[f"{w}|{g}"] = b_stats(rows)

T_tab = []
for T in sorted({x["T"] for x in B_cells}):
    rows = [x for x in B_cells if x["T"] == T and x["pct"] in (1.0, 3.0)]
    if not rows:
        continue
    T_tab.append(dict(T=T, n=len(rows), seasons=sorted({x["season"] for x in rows}, key=snum),
                      rms_tot=r(rms([x["e_tot"] for x in rows])), rms_fut=r(rms([x["e_fut"] for x in rows])),
                      rms_tgt=r(rms([x["e_tgt"] for x in rows])),
                      cov_truth=r(mean([1.0 if x["in_truth"] else 0.0 for x in rows]), 3),
                      cov_s1=r(mean([1.0 if x["in_s1"] else 0.0 for x in rows]), 3)))

result = dict(
    about="WP5a 진단 0 — M3 오차 가르기 (목표 어긋남 vs 곡선 맞히기). 새 적합 없음.",
    inputs=dict(stage1="r7/stage1_M3_D1.json", pred="r7/pred_M3_tune_D1.json", truth="answer_key.json"),
    conv=dict(cells=len(B_cells), conv_false=conv_false),
    A=dict(by_pct=A_by_pct, by_group=A_by_group, expand=A_expand, sign=A_sign, season_tab=A_season_tab,
           rows=[dict(season=x["season"], pct=x["pct"], d=r(x["d"]), sd1=r(x["sd1"]), z=r(x["z"], 3)) for x in A_rows]),
    B=dict(table=B_tab, by_T=T_tab),
)
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "diag0_D1.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=1, sort_keys=True)
    f.write("\n")


# ---------- 마크다운 ----------
def f(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"


L = []
L.append("# WP5a 진단 0 — M3 오차 가르기 (D1)\n")
L.append("`tools/solo_raid/lab7/wp5a_diag0.py`가 만든다. 모두 ln 단위. d = ln 정답 − 1단계 ln 컷 중앙, "
         "e_tot = ln 예측 − ln 정답, e_fut = ln 예측 − 1단계 중앙 (끝난 시즌 곡선 맞히기), e_tgt = 1단계 중앙 − ln 정답 (= −d). "
         "s_pred = (ln i80 상한 − ln i80 하한)/2.5631.\n")
L.append("## (A) 목표 어긋남 — 24시즌 (S19~S42)\n")
L.append("| 목표 | n | 평균 d | sd(d) | RMS(d) | 중앙 \\|d\\| | 중앙 \\|z\\| | 양수 몫 |")
L.append("|---|---|---|---|---|---|---|---|")
for k in [f"{q:g}" for q in PCTS]:
    v = A_by_pct[k]
    L.append(f"| {k}% | {v['n']} | {f(v['mean_d'])} | {f(v['sd_d'])} | {f(v['rms_d'])} | {f(v['med_abs_d'])} | {f(v['med_abs_z'], 2)} | {f(v['pos_share'], 2)} |")
for g, _ in GROUPS_JSON:
    v = A_by_group[g]
    L.append(f"| **{g}** | {v['n']} | {f(v['mean_d'])} | {f(v['sd_d'])} | {f(v['rms_d'])} | {f(v['med_abs_d'])} | {f(v['med_abs_z'], 2)} | {f(v['pos_share'], 2)} |")
L.append("")
L.append("**부호 쏠림** (시즌별 그 묶음 d 평균의 부호)\n")
L.append("| 묶음 | 시즌 | 양수 시즌 | 이항 p(양쪽) | 시즌번호와 상관 | 앞 절반 평균 | 뒤 절반 평균 |")
L.append("|---|---|---|---|---|---|---|")
for g, _ in GROUPS:
    v = A_sign[g]
    L.append(f"| {g} | {v['n_seasons']} | {v['n_pos']} | {f(v['binom_p'], 3)} | {f(v['corr_with_season'], 2)} | {f(v['mean_first_half'])} | {f(v['mean_second_half'])} |")
L.append("")
s13, s10 = A_sign["1·3%"], A_sign["10%"]
L.append(f"부호: 1·3%는 {s13['n_seasons']}시즌 중 {s13['n_pos']}시즌이 양수(이항 p {f(s13['binom_p'], 3)})지만 평균은 {f(A_by_group['1·3%']['mean_d'])}로 작다 — "
         f"3%는 대체로 양수, 1%는 전부 음수라 묶으면 상쇄된다. 10%는 {s10['n_pos']}시즌 양수, 시즌번호와 상관 {f(s10['corr_with_season'], 2)}, "
         f"앞 절반 {f(s10['mean_first_half'])} → 뒤 절반 {f(s10['mean_second_half'])}로 최근 시즌에 양수로 쏠린다.\n")
L.append(f"**시즌 순서로 늘려 가는 창** (그 시즌보다 앞 시즌들의 d, {EXPAND_FROM}부터; 「이번」은 그 시즌 자신의 d 평균 — 앞 시즌 값이 얼마나 맞았나)\n")
L.append("| 시즌 | 1·3% n | 평균 d | RMS(d) | 이번 d | 10% n | 평균 d | RMS(d) | 이번 d |")
L.append("|---|---|---|---|---|---|---|---|---|")
e13, e10 = A_expand["1·3%"], A_expand["10%"]
for a, b in zip(e13, e10):
    L.append(f"| {a['season']} | {a['n_prev']} | {f(a['mean_prev'])} | {f(a['rms_prev'])} | {f(a['d_this'])} | "
             f"{b['n_prev']} | {f(b['mean_prev'])} | {f(b['rms_prev'])} | {f(b['d_this'])} |")
L.append("")
L.append("**시즌별 d 평균** (묶음)\n")
L.append("| 시즌 | 1·3% | 10% |")
L.append("|---|---|---|")
for s in seasons_all:
    L.append(f"| {s} | {f(A_season_tab['1·3%'].get(s))} | {f(A_season_tab['10%'].get(s))} |")
L.append("")
L.append("## (B) 조정 6시즌 칸 오차 가르기\n")
L.append(f"한섭 정답 있는 칸 {len(B_cells)}개, 그중 conv false {conv_false}칸 (같이 센다). "
         "포함 = 80% 구간 포함률: 정답 기준(score.py와 같음) / 1단계 중앙 기준. "
         "c = s_pred에 곱할 배수, σ_add = √(s_pred²+σ²)로 80% 포함이 되는 가장 작은 σ (0~0.1, 0.0005 간격).\n")
L.append("| 창 | 묶음 | 칸 | conv× | RMS e_tot | RMS e_fut | RMS e_tgt | 평균 e_tot | 평균 e_fut | 평균 e_tgt | 포함(정답) | 포함(1단계) | 중앙 s_pred | sd z_fut | 중앙\\|z_fut\\| | sd z_tot | 필요 c | 필요 σ_add |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for w in WINS:
    for g, _ in GROUPS_JSON:
        v = B_tab[f"{w}|{g}"]
        if v["n"] == 0:
            continue
        L.append(f"| {w} | {g} | {v['n']} | {v['n_conv_false']} | {f(v['rms_tot'])} | {f(v['rms_fut'])} | {f(v['rms_tgt'])} | "
                 f"{f(v['bias_tot'])} | {f(v['bias_fut'])} | {f(v['bias_tgt'])} | {f(v['cov_truth'], 2)} | {f(v['cov_s1'], 2)} | "
                 f"{f(v['med_s_pred'])} | {f(v['sd_z_fut'], 2)} | {f(v['med_abs_z_fut'], 2)} | {f(v['sd_z_tot'], 2)} | "
                 f"{f(v['need_c'], 2)} | {f(v['need_sigma_add'], 4) if v['need_sigma_add'] is not None else '>0.1'} |")
L.append("")
L.append("**T별 (1·3% 칸)**\n")
L.append("| T | 칸 | RMS e_tot | RMS e_fut | RMS e_tgt | 포함(정답) | 포함(1단계) |")
L.append("|---|---|---|---|---|---|---|")
for v in T_tab:
    L.append(f"| {v['T']:g} | {v['n']} | {f(v['rms_tot'])} | {f(v['rms_fut'])} | {f(v['rms_tgt'])} | {f(v['cov_truth'], 2)} | {f(v['cov_s1'], 2)} |")
L.append("")
with open(os.path.join(OUT, "diag0_D1.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print("ok", len(A_rows), len(B_cells), conv_false)
