"""WP9a 진단 0 — M3 `dt`가 F에 지는 곳을 칸별로 가른다 (docs/solo-raid-method.md 「WP9a 진단」 0번).
새 적합 없음, 있는 파일만 읽는다. 채점 셈(정답·창·H·KR만·e = 중앙 ÷ 정답 − 1)은 score.py·anchor.py를 import해서 같이 쓴다.

  python3 -I tools/solo_raid/lab7/wp9_diag0.py

입력 (읽기만): r7/pred_M3dt_D1.json · r7/F_D1.json (M3 칸과 같은 (시즌,T,%)로 거름) · r7/lb_D1.json ·
              r7/wp5a/diag1_D1.json (cells[시즌]["base"]["full"][%]["z"] — 앞 시즌 사전 z) · r7/diag7.json (재현 확인)
출력: r7/wp9/diag0_D1.json, r7/wp9/diag0_D1.md (숫자만, 글·닉네임 없음)

정의
  칸 = (시즌, T, 목표 %), 창 main = T 18·24·36·48, end = T H−12·H−6·H. 무리 1·3% = pct 1·3, 10% = pct 10.
  e_M3, e_F = 중앙 ÷ 정답 − 1, d = |e_M3| − |e_F| (양수면 M3이 진다). 모든 평균은 칸 평균 (score.py와 같음).
  tune = S22·S26·S30·S34·S38·S42, rest = 그 밖 16시즌, all = 22시즌.
  몫 = rest 1·3% 초반 창의 (평균 |e_M3| − 평균 |e_F|). 「몫 줄어든 비율」 = 1 − (바꾼 뒤 차) ÷ (원래 차).
  conv = info의 (시즌, T) 수렴 여부 (R-hat·ESS 기준은 m3.py). 수렴 못 한 (시즌, T)의 모든 % 칸이 빠진다.
  사전 |z| = diag1 cells[시즌].base.full[칸의 %].z (1·3·10% 각각 그 %의 z).
  H3: ln(pred10/ans10) = ln(pred3/ans3) + [ln(pred10/pred3) − ln(ans10/ans3)] — 같은 (시즌, T)에 3%·10% 칸이 다 있을 때.
"""
import collections
import json
import math
import os
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
OUT = os.path.join(R7, "wp9")
GROUPS = (("1·3%", (1.0, 3.0)), ("10%", (10.0,)))
WINS = ("main", "end")
SPLITS = ("all", "tune", "rest")
TUNE = set(SC.TUNE)


def jl(p):
    with open(os.path.join(R7, p), encoding="utf-8") as f:
        return json.load(f)


def r(x, n=5):
    return None if x is None else round(float(x), n)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def in_split(s, sp):
    return sp == "all" or (sp == "tune") == (s in TUNE)


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return rk


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def zbin(z):
    if z is None:
        return "없음"
    a = abs(z)
    return "<1" if a < 1 else ("1~2" if a < 2 else ">=2")


def main():
    pm = jl("pred_M3dt_D1.json")
    pf = jl("F_D1.json")
    lbh, lb = SC.load_lb(os.path.join(R7, "lb_D1.json"))
    d1 = jl("wp5a/diag1_D1.json")
    d7 = jl("diag7.json")
    truth = A.truth()
    m3 = SC.load_pred(os.path.join(R7, "pred_M3dt_D1.json"))["cells"]
    keys = {(c["season"], c["server"], c["T"], c["pct"]) for c in m3}
    fall = SC.load_pred(os.path.join(R7, "F_D1.json"))["cells"]
    fc = [c for c in fall if (c["season"], c["server"], c["T"], c["pct"]) in keys]
    fmap = {(c["season"], c["T"], c["pct"]): c for c in fc}
    mmap = {(c["season"], c["T"], c["pct"]): c for c in m3 if c.get("server", "KR") == "KR"}
    info = {(i["season"], float(i["T"])): i for i in pm["info"]}
    missing_F = [k for k in mmap if k not in fmap]

    # ---- 0) 재현
    repro = {}
    maxdiff = 0.0
    for gname, qs in GROUPS:
        allr, sp = SC.evaluate_split(m3, truth, lb, qs, SC.TOL)
        allf, spf = SC.evaluate_split(fc, truth, lb, qs, SC.TOL)
        # diag7: all/tune/rest 각 파일의 groups[g]["cands"]["M3"] 와 ["F"]
        for split in SPLITS:
            src = d7[{"all": "all", "tune": "tune", "rest": "rest"}[split]]["groups"][gname]
            for w in WINS:
                mine_m = sp[split][w]
                mine_f = spf[split][w]
                ref_m = src["cands"]["M3"][w]
                ref_f = src["F"][w]
                repro[f"{gname}|{split}|{w}"] = dict(M3=r(mine_m, 6), M3_doc=r(ref_m, 6), F=r(mine_f, 6), F_doc=r(ref_f, 6))
                maxdiff = max(maxdiff, abs(mine_m - ref_m), abs(mine_f - ref_f))
    # 문서 표 숫자 (%)
    doc_tab = {("1·3%", "tune", "main"): (2.86, 3.15), ("1·3%", "tune", "end"): (1.21, 1.20),
               ("1·3%", "rest", "main"): (4.59, 3.63), ("1·3%", "rest", "end"): (1.01, 0.90),
               ("10%", "rest", "main"): (11.68, 6.81), ("10%", "rest", "end"): (3.76, 3.81),
               ("10%", "tune", "main"): (6.58, 4.60), ("10%", "tune", "end"): (3.78, 2.77)}
    doc_ok = True
    doc_rows = []
    for (g, sp_, w), (dm, df) in doc_tab.items():
        v = repro[f"{g}|{sp_}|{w}"]
        ok = abs(v["M3"] * 100 - dm) < 0.006 and abs(v["F"] * 100 - df) < 0.006
        doc_ok &= ok
        doc_rows.append(dict(group=g, split=sp_, win=w, M3=r(v["M3"] * 100, 3), M3_doc=dm, F=r(v["F"] * 100, 3), F_doc=df, ok=ok))

    # ---- 칸 표
    rows = []
    for (s, T, q), c in mmap.items():
        if q not in (1.0, 3.0, 10.0):
            continue
        w = SC.window(s, T)
        if w is None:
            continue
        f = fmap.get((s, T, q))
        tv = truth.get((s, "KR", q))
        if tv is None or f is None:
            continue
        ok = lambda x: bool(x.get("pred")) and bool(x.get("i80")) and None not in x["i80"]   # noqa: E731
        if not (ok(c) and ok(f)):
            continue
        inf = info[(s, T)]
        zr = (d1["cells"].get(s, {}).get("base", {}).get("full", {}).get(f"{q:g}") or {}).get("z")
        em, ef = c["pred"] / tv - 1, f["pred"] / tv - 1
        rows.append(dict(season=s, T=T, tk=SC.tkey(s, T), pct=q, win=w, group="10%" if q == 10.0 else "1·3%",
                         conv=bool(inf["conv"]), k=str(inf["k"]), n=inf["n"], z=zr, zb=zbin(zr),
                         em=em, ef=ef, am=abs(em), af=abs(ef), d=abs(em) - abs(ef),
                         pm=c["pred"], pf=f["pred"], ans=tv, tune=s in TUNE,
                         conv_cell=bool(c.get("conv", True))))
    conv_mismatch = sum(1 for x in rows if x["conv"] != x["conv_cell"])

    def sel(group, win, split):
        return [x for x in rows if x["group"] == group and x["win"] == win and in_split(x["season"], split)]

    # ---- 1) H5
    h5 = {}
    for gname, _ in GROUPS:
        for w in WINS:
            for sp in SPLITS:
                R = sel(gname, w, sp)
                cv = [x for x in R if x["conv"]]
                nc = [x for x in R if not x["conv"]]
                gap = mean(x["am"] for x in R) - mean(x["af"] for x in R)
                gap_ex = (mean(x["am"] for x in cv) - mean(x["af"] for x in cv)) if cv else None
                gap_sw = (sum(x["am"] for x in cv) + sum(x["af"] for x in nc)) / len(R) - mean(x["af"] for x in R)
                nst_all = len({(x["season"], x["T"]) for x in R})
                nst_nc = len({(x["season"], x["T"]) for x in nc})
                h5[f"{gname}|{w}|{sp}"] = dict(
                    n_cells=len(R), n_conv=len(cv), n_nonconv=len(nc), n_sT=nst_all, n_sT_nonconv=nst_nc,
                    mean_abs_M3=r(mean(x["am"] for x in R)), mean_abs_F=r(mean(x["af"] for x in R)), gap=r(gap),
                    mean_abs_M3_ex=r(mean(x["am"] for x in cv)) if cv else None,
                    mean_abs_F_ex=r(mean(x["af"] for x in cv)) if cv else None,
                    gap_exclude_nonconv=r(gap_ex),
                    shrink_exclude=r(1 - gap_ex / gap) if (gap_ex is not None and gap) else None,
                    gap_swap_nonconv_to_F=r(gap_sw), shrink_swap=r(1 - gap_sw / gap) if gap else None,
                    d_conv=r(mean(x["d"] for x in cv)) if cv else None,
                    d_nonconv=r(mean(x["d"] for x in nc)) if nc else None,
                    mean_abs_M3_conv=r(mean(x["am"] for x in cv)) if cv else None,
                    mean_abs_M3_nonconv=r(mean(x["am"] for x in nc)) if nc else None,
                    mean_abs_F_conv=r(mean(x["af"] for x in cv)) if cv else None,
                    mean_abs_F_nonconv=r(mean(x["af"] for x in nc)) if nc else None)
    # 시즌·T 수렴 칸 수 (info 전체, 창 T만)
    conv_sT = collections.Counter()
    for (s, T), inf in info.items():
        w = SC.window(s, T)
        if w:
            conv_sT[(w, bool(inf["conv"]))] += 1
    conv_sT = {f"{w}|{'conv' if c else 'nonconv'}": n for (w, c), n in sorted(conv_sT.items())}
    conv_all_sT = collections.Counter(bool(i["conv"]) for i in info.values())

    # ---- 2) d 가르기
    def agg(R):
        return dict(n=len(R), d=r(mean(x["d"] for x in R)), abs_M3=r(mean(x["am"] for x in R)), abs_F=r(mean(x["af"] for x in R)),
                    bias_M3=r(mean(x["em"] for x in R)), bias_F=r(mean(x["ef"] for x in R)))

    dims = {"T": lambda x: x["tk"], "season": lambda x: x["season"], "k": lambda x: x["k"],
            "conv": lambda x: "conv" if x["conv"] else "nonconv", "zbin": lambda x: x["zb"],
            "split": lambda x: "tune" if x["tune"] else "rest"}
    dsplit = {}
    for gname, _ in GROUPS:
        for w in WINS:
            R = sel(gname, w, "all")
            for dn, fn in dims.items():
                g = collections.defaultdict(list)
                for x in R:
                    g[fn(x)].append(x)
                dsplit[f"{gname}|{w}|{dn}"] = {k: agg(v) for k, v in sorted(g.items(), key=lambda kv: (len(kv[0]) if dn == "T" else 0, kv[0]))}
            # rest/tune 안에서 T, conv, zbin
            for sp in ("tune", "rest"):
                Rs = sel(gname, w, sp)
                for dn in ("T", "conv", "zbin", "k"):
                    g = collections.defaultdict(list)
                    for x in Rs:
                        g[dims[dn](x)].append(x)
                    dsplit[f"{gname}|{w}|{sp}|{dn}"] = {k: agg(v) for k, v in sorted(g.items())}
    # 시즌별 상관
    corr = {}
    for gname, _ in GROUPS:
        for w in WINS:
            bys = collections.defaultdict(list)
            for x in sel(gname, w, "all"):
                if x["z"] is not None:
                    bys[x["season"]].append(x)
            ss = sorted(bys)
            md = [mean(x["d"] for x in bys[s]) for s in ss]
            az = [mean(abs(x["z"]) for x in bys[s]) for s in ss]
            sz = [mean(x["z"] for x in bys[s]) for s in ss]
            corr[f"{gname}|{w}"] = dict(n_seasons=len(ss), pearson_abs_z=r(pearson(az, md), 3), spearman_abs_z=r(spearman(az, md), 3),
                                        pearson_signed_z=r(pearson(sz, md), 3), spearman_signed_z=r(spearman(sz, md), 3))
            # 칸 수준
            cz = [abs(x["z"]) for s in ss for x in bys[s]]
            cd = [x["d"] for s in ss for x in bys[s]]
            corr[f"{gname}|{w}"].update(cell_n=len(cd), cell_pearson_abs_z=r(pearson(cz, cd), 3), cell_spearman_abs_z=r(spearman(cz, cd), 3))
    season_tab = {}
    for gname, _ in GROUPS:
        for w in WINS:
            bys = collections.defaultdict(list)
            for x in sel(gname, w, "all"):
                bys[x["season"]].append(x)
            for s, v in bys.items():
                zs = [abs(x["z"]) for x in v if x["z"] is not None]
                season_tab.setdefault(s, {})[f"{gname}|{w}"] = dict(n=len(v), d=r(mean(x["d"] for x in v)), abs_M3=r(mean(x["am"] for x in v)),
                                                                   abs_F=r(mean(x["af"] for x in v)), abs_z=r(mean(zs)) if zs else None,
                                                                   n_nonconv=sum(1 for x in v if not x["conv"]))

    # ---- 3) H3
    by_sT = collections.defaultdict(dict)
    for x in rows:
        by_sT[(x["season"], x["T"])][x["pct"]] = x
    h3rows = []
    for (s, T), dct in sorted(by_sT.items()):
        if 3.0 in dct and 10.0 in dct:
            a3, a10 = dct[3.0], dct[10.0]
            out = dict(season=s, T=T, win=a3["win"], tune=a3["tune"], conv=a3["conv"])
            for nm, pk in (("M3", "pm"), ("F", "pf")):
                A3 = math.log(a3[pk] / a3["ans"])
                tot = math.log(a10[pk] / a10["ans"])
                B = tot - A3
                out[nm] = dict(A=A3, B=B, tot=tot, shape_pred=math.log(a10[pk] / a3[pk]), shape_ans=math.log(a10["ans"] / a3["ans"]))
            h3rows.append(out)
    h3 = {}
    for w in WINS:
        for sp in SPLITS:
            R = [x for x in h3rows if x["win"] == w and in_split(x["season"], sp)]
            if not R:
                continue
            ent = dict(n=len(R))
            for nm in ("M3", "F"):
                A_ = [x[nm]["A"] for x in R]
                B_ = [x[nm]["B"] for x in R]
                T_ = [x[nm]["tot"] for x in R]
                sst = sum(t * t for t in T_)
                saa = sum(a * a for a in A_)
                sbb = sum(b * b for b in B_)
                sab = 2 * sum(a * b for a, b in zip(A_, B_))
                rms = lambda v: math.sqrt(mean(t * t for t in v))   # noqa: E731
                ent[nm] = dict(mean_A=r(mean(A_)), rms_A=r(rms(A_)), mean_B=r(mean(B_)), rms_B=r(rms(B_)),
                               mean_tot=r(mean(T_)), rms_tot=r(rms(T_)),
                               mean_shape_pred=r(mean(x[nm]["shape_pred"] for x in R)), mean_shape_ans=r(mean(x[nm]["shape_ans"] for x in R)),
                               share_A=r(saa / sst), share_B=r(sbb / sst), share_cov=r(sab / sst), sumsq_tot=r(sst))
            # M3 − F 제곱합 차의 몫
            sm = sum(x["M3"]["tot"] ** 2 for x in R)
            sf = sum(x["F"]["tot"] ** 2 for x in R)
            ent["sumsq_diff_M3_minus_F"] = r(sm - sf)
            ent["diff_share_A"] = r((sum(x["M3"]["A"] ** 2 for x in R) - sum(x["F"]["A"] ** 2 for x in R)) / (sm - sf)) if sm != sf else None
            ent["diff_share_B"] = r((sum(x["M3"]["B"] ** 2 for x in R) - sum(x["F"]["B"] ** 2 for x in R)) / (sm - sf)) if sm != sf else None
            ent["diff_share_cov"] = r((2 * sum(x["M3"]["A"] * x["M3"]["B"] for x in R) - 2 * sum(x["F"]["A"] * x["F"]["B"] for x in R)) / (sm - sf)) if sm != sf else None
            h3[f"{w}|{sp}"] = ent

    # ---- 4) g1
    g1 = {}
    for nm, cells in (("M3", m3), ("F", fc)):
        for gname, qs in GROUPS:
            lst = []
            for c in cells:
                q, s, T = c["pct"], c["season"], c["T"]
                if c.get("server", "KR") != "KR" or q not in qs:
                    continue
                tv = truth.get((s, "KR", q))
                if tv is None or not c.get("pred") or not c.get("i80") or None in c["i80"]:
                    continue
                lbv = lb.get((s, T, q))
                if T >= 18 and bool(c.get("own", True)) and lbv is not None and c["pred"] < lbv - 0.05:
                    inf = info.get((s, T), {})
                    lst.append(dict(season=s, T=T, pct=q, pred=r(c["pred"], 3), lb=r(lbv, 3), ans=r(tv, 3), conv=inf.get("conv"), n=inf.get("n"),
                                    win=SC.window(s, T), other_method_pred=r((fmap if nm == "M3" else mmap).get((s, T, q), {}).get("pred"), 3)))
            ref = SC.evaluate(cells, truth, lb, qs, SC.TOL)["g1"]
            g1[f"{nm}|{gname}"] = dict(n=len(lst), n_score_py=ref, cells=sorted(lst, key=lambda z: (z["season"], z["T"], z["pct"])))

    res = dict(about="WP9a 진단 0 (새 적합 없음). 정의는 wp9_diag0.py 머리 참조. 비율·e는 소수 (0.01 = 1%).",
               inputs={"pred_M3dt_D1": pm["made"], "n_cells_M3": len(m3), "n_cells_F_filtered": len(fc), "F_missing_for_M3_cells": len(missing_F),
                       "n_window_rows": len(rows), "conv_mismatch_cell_vs_info": conv_mismatch},
               repro=dict(max_abs_diff_vs_diag7=maxdiff, doc_table_ok=doc_ok, doc_table=doc_rows, detail=repro),
               h5=h5, conv_counts_windowT=conv_sT, conv_counts_all_sT={"conv": conv_all_sT[True], "nonconv": conv_all_sT[False]},
               d_split=dsplit, z_corr=corr, season_table=season_tab, h3=h3, g1=g1)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "diag0_D1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    write_md(res)
    print("repro max diff", maxdiff, "doc ok", doc_ok)


def P(x, d=2):
    return "-" if x is None else f"{x * 100:.{d}f}"


def tbl(head, rows_):
    esc = lambda v: str(v).replace("|", "\\|")   # noqa: E731
    o = ["| " + " | ".join(esc(h) for h in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for rr in rows_:
        o.append("| " + " | ".join(esc(c) for c in rr) + " |")
    return "\n".join(o)


def write_md(res):
    h5, ds, corr, h3, g1 = res["h5"], res["d_split"], res["z_corr"], res["h3"], res["g1"]
    L = []
    k = h5["1·3%|main|rest"]
    k10 = h5["10%|main|rest"]
    L.append("# WP9a 진단 0 — M3 `dt` 대 F, 칸별 가르기 (D1, 22시즌, 새 적합 없음)\n")
    L.append("정의: 칸 = (시즌, T, 목표 %). 초반 창 main = T 18·24·36·48, 막판 창 end = T H−12·H−6·H. e = 중앙 ÷ 정답 − 1, d = |e_M3| − |e_F| (양수 = M3이 짐). "
             "평균은 칸 평균(score.py와 같음), 단위 %p는 숫자 × 100. tune = S22·S26·S30·S34·S38·S42, rest = 나머지 16시즌. "
             "몫 = rest 1·3% 초반 창의 평균 |e| 차 (M3 − F). 몫 줄어든 비율 = 1 − 바꾼 뒤 차 ÷ 원래 차. "
             "수렴 못 한 칸 = info의 (시즌, T) conv False — 그 (시즌, T)의 모든 % 칸. 사전 |z| = diag1 `cells[시즌].base.full[%].z`. "
             "코드 `tools/solo_raid/lab7/wp9_diag0.py`, 수치 전부 `diag0_D1.json`.\n")
    L.append("## 요약\n")
    L.append(f"1. 재현: diag7 대비 최대 차 {res['repro']['max_abs_diff_vs_diag7']:.2e}, 문서 표 숫자 8줄 {'모두 일치' if res['repro']['doc_table_ok'] else '불일치 있음 (아래 표)'} "
             f"(rest 1·3% 초반 M3 {P(k['mean_abs_M3'])} · F {P(k['mean_abs_F'])} · 차 {P(k['gap'])}%p).")
    L.append(f"2. **수렴 못 한 (시즌, T) 칸을 M3·F 둘 다에서 빼면 rest 1·3% 초반 몫이 {P(k['shrink_exclude'], 1)}% 준다 (음수 = 오히려 는다)** "
             f"(차 {P(k['gap'])} → {P(k['gap_exclude_nonconv'])}%p, 남은 칸 {k['n_conv']}/{k['n_cells']}). 못 한 칸의 M3를 F 값으로 바꾸면 {P(k['shrink_swap'], 1)}% 준다 (차 {P(k['gap_swap_nonconv_to_F'])}).")
    L.append(f"3. 10% rest 초반은 같은 셈으로 {P(k10['shrink_exclude'], 1)}% (차 {P(k10['gap'])} → {P(k10['gap_exclude_nonconv'])}), 바꾸기 {P(k10['shrink_swap'], 1)}%. "
             f"22시즌 1·3% 초반 {P(h5['1·3%|main|all']['shrink_exclude'], 1)}%, tune {P(h5['1·3%|main|tune']['shrink_exclude'], 1)}%.")
    c1, c2 = ds["1·3%|main|rest|conv"], ds["1·3%|main|rest|zbin"]
    L.append("4. rest 1·3% 초반 평균 d: " + ", ".join(f"{kk} {P(v['d'])}%p(n {v['n']})" for kk, v in c1.items()) + " / 사전 |z|: "
             + ", ".join(f"{kk} {P(v['d'])}%p(n {v['n']})" for kk, v in c2.items()) + f". 시즌 평균 d 대 |z| 상관(1·3% 초반) 피어슨 {corr['1·3%|main']['pearson_abs_z']} · 스피어만 {corr['1·3%|main']['spearman_abs_z']} (시즌 {corr['1·3%|main']['n_seasons']}).")
    a, b = h3["main|rest"], h3["end|rest"]
    L.append(f"5. 10% rest 초반 오차 제곱합(ln)의 몫: M3 3%오차 {P(a['M3']['share_A'], 0)}% · 3→10 모양 {P(a['M3']['share_B'], 0)}% · 공분산 {P(a['M3']['share_cov'], 0)}% / "
             f"F {P(a['F']['share_A'], 0)}·{P(a['F']['share_B'], 0)}·{P(a['F']['share_cov'], 0)}%. g1: M3 1·3% {g1['M3|1·3%']['n']}칸 · F {g1['F|1·3%']['n']}칸, 10% M3 {g1['M3|10%']['n']} · F {g1['F|10%']['n']}.\n")

    L.append("## 0. 재현 (%, 칸 평균 |e|)\n")
    L.append(tbl(["무리", "나눔", "창", "M3", "문서 M3", "F", "문서 F", "일치"],
                 [[x["group"], x["split"], x["win"], x["M3"], x["M3_doc"], x["F"], x["F_doc"], "o" if x["ok"] else "x"] for x in res["repro"]["doc_table"]]))
    L.append(f"\n입력: M3 칸 {res['inputs']['n_cells_M3']}, F를 같은 칸으로 거른 {res['inputs']['n_cells_F_filtered']} (M3 칸에 F 없음 {res['inputs']['F_missing_for_M3_cells']}), "
             f"창·정답 있는 1·3·10% 칸 {res['inputs']['n_window_rows']}, 칸 conv와 info conv 불일치 {res['inputs']['conv_mismatch_cell_vs_info']}. "
             f"info 전체 (시즌, T) 수렴 {res['conv_counts_all_sT']['conv']}/{res['conv_counts_all_sT']['conv'] + res['conv_counts_all_sT']['nonconv']}.\n")

    L.append("## 1. H5 수렴\n")
    rows_ = []
    for g in ("1·3%", "10%"):
        for w in WINS:
            for sp in SPLITS:
                h = h5[f"{g}|{w}|{sp}"]
                rows_.append([g, w, sp, h["n_cells"], f"{h['n_conv']}/{h['n_nonconv']}", f"{h['n_sT'] - h['n_sT_nonconv']}/{h['n_sT_nonconv']}",
                              P(h["mean_abs_M3"]), P(h["mean_abs_F"]), P(h["gap"]), P(h["gap_exclude_nonconv"]), P(h["shrink_exclude"], 1),
                              P(h["gap_swap_nonconv_to_F"]), P(h["shrink_swap"], 1)])
    L.append(tbl(["무리", "창", "나눔", "칸", "칸 수렴/못", "(시즌,T) 수렴/못", "|e_M3|", "|e_F|", "차", "못 칸 뺀 차", "줄어든 비율%", "바꾼 차", "줄어든 비율%"], rows_))
    L.append("\n수렴·못 한 칸의 평균 d, |e|:\n")
    rows_ = []
    for g in ("1·3%", "10%"):
        for w in WINS:
            for sp in SPLITS:
                h = h5[f"{g}|{w}|{sp}"]
                rows_.append([g, w, sp, P(h["d_conv"]), P(h["d_nonconv"]), P(h["mean_abs_M3_conv"]), P(h["mean_abs_F_conv"]), P(h["mean_abs_M3_nonconv"]), P(h["mean_abs_F_nonconv"])])
    L.append(tbl(["무리", "창", "나눔", "d 수렴", "d 못", "M3 수렴", "F 수렴", "M3 못", "F 못"], rows_))

    L.append("\n## 2. 칸별 d = |e_M3| − |e_F| 가르기 (%p, 22시즌 전체 칸)\n")

    def dtab(dn, g, w, label):
        dct = ds[f"{g}|{w}|{dn}"]
        return tbl([label, "n", "d", "|e_M3|", "|e_F|"], [[kk, v["n"], P(v["d"]), P(v["abs_M3"]), P(v["abs_F"])] for kk, v in dct.items()])

    for g in ("1·3%", "10%"):
        for w in WINS:
            L.append(f"\n### {g} {w}\n")
            for dn, lab in (("T", "T"), ("k", "창 k"), ("conv", "conv"), ("zbin", "사전 |z|"), ("split", "나눔")):
                L.append(dtab(dn, g, w, lab) + "\n")
            for sp in ("rest", "tune"):
                L.append(f"{sp} 안에서 (T, conv, 사전 |z|):\n")
                for dn in ("T", "conv", "zbin"):
                    dct = ds[f"{g}|{w}|{sp}|{dn}"]
                    L.append(f"- {dn}: " + ", ".join(f"{kk} {P(v['d'])} (n {v['n']})" for kk, v in dct.items()))
                L.append("")
    L.append("\n### 시즌별 (평균 d %p · 평균 사전 |z| · 못 수렴 칸 수)\n")
    st_ = res["season_table"]
    rows_ = []
    for s in sorted(st_):
        v = st_[s]
        def c(key):
            e = v.get(key)
            return ("-", "-", "-") if not e else (P(e["d"]), "-" if e["abs_z"] is None else f"{e['abs_z']:.2f}", e["n_nonconv"])
        a1, a2, a3, a4 = c("1·3%|main"), c("1·3%|end"), c("10%|main"), c("10%|end")
        rows_.append([s, "T" if False else ("t" if s in TUNE else "r"), *a1, *a2, *a3, *a4])
    L.append(tbl(["시즌", "tune(t)/rest(r)", "1·3% 초반 d", "|z|", "못수렴", "1·3% 막판 d", "|z|", "못수렴", "10% 초반 d", "|z|", "못수렴", "10% 막판 d", "|z|", "못수렴"], rows_))
    L.append("\n시즌 평균 d 대 시즌 평균 사전 |z| 상관:\n")
    L.append(tbl(["무리·창", "시즌 수", "피어슨 |z|", "스피어만 |z|", "피어슨 z(부호)", "스피어만 z(부호)", "칸 수", "칸 피어슨 |z|", "칸 스피어만 |z|"],
                 [[kk, v["n_seasons"], v["pearson_abs_z"], v["spearman_abs_z"], v["pearson_signed_z"], v["spearman_signed_z"], v["cell_n"], v["cell_pearson_abs_z"], v["cell_spearman_abs_z"]] for kk, v in corr.items()]))

    L.append("\n\n## 3. H3 10% 나누기 (ln 오차)\n")
    L.append("ln(pred10/ans10) = A + B, A = ln(pred3/ans3), B = ln(pred10/pred3) − ln(ans10/ans3). 같은 (시즌, T)에 3%·10% 칸이 다 있는 칸. "
             "제곱합 몫: A², B², 2AB를 전체 제곱합으로 나눈 값 (합 100%). 아래 `차 몫`은 M3 − F 제곱합 차를 A·B·공분산이 설명하는 비율.\n")
    rows_ = []
    for key_, e in h3.items():
        for nm in ("M3", "F"):
            m = e[nm]
            rows_.append([key_, e["n"], nm, f"{m['mean_A']:+.3f}", f"{m['rms_A']:.3f}", f"{m['mean_B']:+.3f}", f"{m['rms_B']:.3f}", f"{m['mean_tot']:+.3f}", f"{m['rms_tot']:.3f}",
                          P(m["share_A"], 0), P(m["share_B"], 0), P(m["share_cov"], 0)])
    L.append(tbl(["창|나눔", "n", "방법", "평균 A", "RMS A", "평균 B", "RMS B", "평균 합", "RMS 합", "A² %", "B² %", "2AB %"], rows_))
    L.append("\nM3 − F 제곱합 차의 몫:\n")
    L.append(tbl(["창|나눔", "제곱합 차", "A 몫", "B 몫", "공분산 몫"],
                 [[kk, e["sumsq_diff_M3_minus_F"], P(e["diff_share_A"], 0) if e["diff_share_A"] is not None else "-", P(e["diff_share_B"], 0) if e["diff_share_B"] is not None else "-",
                   P(e["diff_share_cov"], 0) if e["diff_share_cov"] is not None else "-"] for kk, e in h3.items()]))
    L.append("\n모양 항 평균 (ln): ")
    L.append("; ".join(f"{kk}: M3 예측 ln(p10/p3) {e['M3']['mean_shape_pred']:+.3f} · F {e['F']['mean_shape_pred']:+.3f} · 정답 {e['M3']['mean_shape_ans']:+.3f}" for kk, e in h3.items()))

    L.append("\n\n## 4. g1 (자기 칸, T ≥ 18, 중앙 < 하한 − 0.05)\n")
    for kk, v in g1.items():
        L.append(f"\n{kk}: {v['n']}칸 (score.py g1 = {v['n_score_py']})\n")
        if v["cells"]:
            L.append(tbl(["시즌", "T", "%", "pred", "lb", "정답", "conv", "n", "창", "다른 방법 pred"],
                         [[c["season"], f"{c['T']:g}", f"{c['pct']:g}", c["pred"], c["lb"], c["ans"], c["conv"], c["n"], c["win"], c["other_method_pred"]] for c in v["cells"]]))
    L.append("")
    with open(os.path.join(OUT, "diag0_D1.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


TUNE = set(SC.TUNE)

if __name__ == "__main__":
    main()
