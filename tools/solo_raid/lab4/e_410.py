"""분석 4.10 (2026-10-19) — 모양 갱신(--shape-update) 다시 재기. 사전 고정: docs/solo-raid-analysis.md 「4.10 사전 고정」 (이 파일과 같은 커밋).

predict.py·xval.py·backtest_web.py·model_S42.json은 고치지 않는다. 후보는 지금 F(R0)와 F + predict 옵션 shape_update=true(Z) 둘뿐이고,
실행 인자는 4.9 B의 R0와 같다 (e_49.e48_argv). 덧붙임은 기록(zrec)뿐 — 본 예측의 모양 기울임 추정 Ẑ·사후 sd를 줄에 남기고 예측값은 바꾸지 않는다.

  python3 -I tools/solo_raid/lab4/e_410.py run NAME main|ref --res RES --work XW [--seed 1] [--jobs 4]   # NAME = R0 | Z → RES/s{시드}/NAME/
  python3 -I tools/solo_raid/lab4/e_410.py todo RES                                                      # 남은 「시드 이름 부분」
  python3 -I tools/solo_raid/lab4/e_410.py sel RES --out diag410.json [--pts PTS --work XW]              # 고르기·보고

고르기 (사전 고정): 1·3%(main)와 10%(ref)를 따로, 지금 값 R0. 규칙은 4.9 B와 같다 (e_49.pick):
  (1) T_main 한섭 80% 포함(main은 M1 뒤) ≥ 0.70 (2) score = (T_main |e| + T_end |e|) ÷ 2 최소 (|e| = criteria_v2 평균 시즌 중앙 |e|)
  (3) 0.5%p 안이면 단순한 쪽 R0 (4) guard — T_end 포함 ≥ 0.70 · end_guard (T_end 가장 큰 지점 |e| ≤ R0) · late_guard (T72·96 포함 ≥ 0.70,
      |e|가 R0보다 0.3%p 넘게 나쁘지 않음) · early_guard (T12 포함 ≥ R0 − 0.05). 10%는 같은 규칙을 10% 줄로.
  운영에서 1·3%와 10%는 다른 실행(main·ref)이라 10%에 Z를 켜도 1·3% 값은 바뀌지 않는다.
  막판 다리 (4.9 C 메모를 보고 계산 전에 더함 — score는 시즌 중앙 |e|라 일부 시즌의 막판 실패를 못 본다): 위 규칙이 R0여도 Z가
  (가) T=H 허용 밖 시즌·지점 수(1·3% ±2%, 10% ±3%)를 2개 이상 줄이고 (나) T_end 시즌 평균 |e|(칸마다 시즌 평균, 칸 평균)를 0.5%p 넘게 줄이며
  (다) guard 넷을 지나고 (라) score가 R0보다 0.3%p 넘게 나쁘지 않으면 Z. 1·3%와 10% 따로.
보고만: 시드 2·3 (R0·Z, main·ref), e_49 report(Z 대 R0 — T별·창별·시즌별·연속성·summary_v2), Ẑ 진단(시즌별 Ẑ와 정답 비율의 관계).
"""
import argparse
import collections
import json
import math
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import predict as PR  # noqa: E402
import xval as XV  # noqa: E402
import e_48 as E48  # noqa: E402
import e_49 as E49  # noqa: E402

NAMES = ["R0", "Z"]                       # 단순한 순서 (0.5%p 안이면 앞)
TOL_END = {1.0: 0.02, 3.0: 0.02, 10.0: 0.03}  # 시즌 끝 허용 (4.9 C 메모와 같음)
TAIL_N = 2                                # 막판 다리 (가): 허용 밖 수를 이만큼 이상 줄임
TAIL_MAE = 0.005                          # (나): T_end 시즌 평균 |e|를 이보다 많이 줄임
TAIL_SCORE_TOL = 0.003                    # (라): score가 R0보다 이보다 많이 나쁘지 않음
SEEDS = (1, 2, 3)                         # 고르기는 시드 1, 2·3은 보고만
Z_OPT = "shape_update=true"


# ---------------- 기록 덧붙임: Ẑ ----------------

_Z = {"on": False, "last": None}


def apply_zrec():
    """predict.predict 안에서 info에 z_shape를 처음 남긴 Predictor.run(가운데 추정 — predict가 central일 때만 남김)의 값을 out.servers[s].z_shape로, 채점 줄에 z_shape로.
    예측값은 바꾸지 않는다 (기술 확인 (a))."""
    orig_run = PR.Predictor.run

    def run(self, rows, comps=("i", "ii", "iii"), draws=1000, seed=1, mode="pt"):
        out = orig_run(self, rows, comps, draws, seed, mode)
        info = out.get("info") or {}
        if _Z["on"] and _Z["last"] is None and any(v.get("z_shape") for v in info.values()):
            _Z["last"] = {s: v.get("z_shape") for s, v in info.items()}
        return out
    PR.Predictor.run = run

    orig_pred = PR.predict

    def predict(*a, **k):
        _Z["on"], _Z["last"] = True, None
        try:
            out = orig_pred(*a, **k)
        finally:
            _Z["on"] = False
        for s, sv in (out.get("servers") or {}).items():
            sv["z_shape"] = (_Z["last"] or {}).get(s)
        return out
    PR.predict = predict

    orig_score = XV.score

    def score(out, *a, **k):
        rows = orig_score(out, *a, **k)
        for r in rows:
            r["z_shape"] = ((out.get("servers") or {}).get(r["server"]) or {}).get("z_shape")
        return rows
    XV.score = score
    XV.SLIM_KEYS = tuple(XV.SLIM_KEYS) + ("z_shape",)


def cmd_inner(argv):
    """e_48.py run과 같은 인자 + zrec. run이 하위 프로세스로 부른다."""
    apply_zrec()
    rc = E48.cmd_run(argv)
    k = argv.index("--")
    xargs = argv[k + 1:]
    out = xargs[xargs.index("--out") + 1]
    res = E49.jl(out)
    res["run"]["lab_patches"] = dict(res["run"].get("lab_patches") or {}, zrec=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
        f.write("\n")
    return rc


def argv_for(name, part, targets, T, out, seed, work, jobs):
    if name not in NAMES:
        raise SystemExit(f"모르는 후보: {name}")
    a = E49.e48_argv(E49.spec("R0"), part, targets, T, out, seed, work, "-", jobs)[1:]      # 앞의 "run"을 뺀 e_48 run 인자
    if name == "Z":
        k = a.index("--work")
        a = a[:k] + ["--predict-opt", Z_OPT] + a[k:]
    return a


def cmd_run(argv):
    ap = argparse.ArgumentParser(prog="e_410.py run")
    ap.add_argument("name", choices=NAMES)
    ap.add_argument("part", choices=["main", "ref"])
    ap.add_argument("--res", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    d = E49.cand_dir(a.res, a.seed, a.name)
    os.makedirs(d, exist_ok=True)
    logd = os.path.join(a.res, "logs")
    os.makedirs(logd, exist_ok=True)
    for f, tg, T in E49.part_files(a.part):
        out = os.path.join(d, f)
        if os.path.exists(out):
            print("있음", out, flush=True)
            continue
        cmd = [sys.executable, "-I", os.path.abspath(__file__), "_inner"] + argv_for(a.name, a.part, tg, T, out + ".part", a.seed, a.work, a.jobs)
        lg = os.path.join(logd, f"s{a.seed}_{a.name}_{f[:-5]}.log")
        with open(lg, "w", encoding="utf-8") as fo:
            p = subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT)
        if p.returncode or not os.path.exists(out + ".part"):
            raise SystemExit(f"실패 {a.name} {f} (rc {p.returncode}) — {lg}")
        os.replace(out + ".part", out)
        print("→", out, flush=True)
    return 0


def need_list(R):
    return [(s, n, p) for s in SEEDS for n in NAMES for p in ("main", "ref") if not R.has(n, p, s)]


def cmd_todo(argv):
    ap = argparse.ArgumentParser(prog="e_410.py todo")
    ap.add_argument("res")
    a = ap.parse_args(argv)
    for s, n, p in need_list(E49.Res(a.res)):
        print(s, n, p)
    return 0


# ---------------- Ẑ 진단 (보고만) ----------------

def zdiag(R, seed=1):
    """시즌별 한섭 Ẑ(가운데 추정)와 정답 쪽 모양 차이.
    ref 줄: 10%/3% 예측 비(R0·Z)와 정답 비, main 줄: 1%/3%. 「모양 차이」 = ln(정답 비 / R0 예측 비) — Z가 따라가야 할 방향.
    T_end는 시즌의 H−12·H−6·H, T_main 18·24·36·48."""
    out = {}
    for part, (qa, qb) in (("ref", (10.0, 3.0)), ("main", (1.0, 3.0))):
        get = R.ref if part == "ref" else R.main
        rows = {n: get(n, seed)["rows"] for n in NAMES}
        cell = {n: {(r["season"], round(float(r["T"]), 2), float(r["pct"])): r for r in rows[n] if r["server"] == "KR"} for n in NAMES}
        per = {}
        for s in sorted({k[0] for k in cell["Z"]}, key=E49.snum):
            H = E48.season_H(s)
            ts = E48.tsets(H)
            row = {}
            for lab, T_list in (("T_main", E49.T_MAIN), ("T_end", ts["end"]), ("T=H", [round(H, 2)])):
                zs, d_r0, d_z, tr = [], [], [], None
                for t in T_list:
                    t = round(float(t), 2)
                    a0, b0 = cell["R0"].get((s, t, qa)), cell["R0"].get((s, t, qb))
                    a1, b1 = cell["Z"].get((s, t, qa)), cell["Z"].get((s, t, qb))
                    if not (a0 and b0 and a1 and b1):
                        continue
                    if a1.get("z_shape"):
                        zs.append(a1["z_shape"][0])
                    if a0.get("truth") and b0.get("truth"):
                        tr = a0["truth"] / b0["truth"]
                        d_r0.append(math.log(tr / (a0["pred"] / b0["pred"])))
                        d_z.append(math.log(tr / (a1["pred"] / b1["pred"])))
                row[lab] = {"z_mean": round(statistics.mean(zs), 3) if zs else None,
                            "ratio_truth": round(tr, 4) if tr else None,
                            "lnratio_gap_R0": round(statistics.mean(d_r0), 4) if d_r0 else None,
                            "lnratio_gap_Z": round(statistics.mean(d_z), 4) if d_z else None}
            per[s] = row
        summ = {}
        for lab in ("T_main", "T_end", "T=H"):
            P = [(v[lab]["z_mean"], v[lab]["lnratio_gap_R0"], v[lab]["lnratio_gap_Z"]) for v in per.values()
                 if v.get(lab) and v[lab]["z_mean"] is not None and v[lab]["lnratio_gap_R0"] is not None]
            if len(P) >= 3:
                zz, g0, g1 = zip(*P)
                summ[lab] = {"n_seasons": len(P), "corr_z_vs_gapR0": round(corr(zz, g0), 3),
                             "mean_abs_gap_R0": round(statistics.mean(abs(x) for x in g0), 4),
                             "mean_abs_gap_Z": round(statistics.mean(abs(x) for x in g1), 4),
                             "n_closer": sum(abs(b) < abs(a) for a, b in zip(g0, g1)),
                             "z_sd": round(statistics.pstdev(zz), 3), "n_z_capped": sum(abs(z) >= PR.Z_CAP - 1e-9 for z in zz)}
        out[f"{part}_{qa:g}over{qb:g}"] = {"summary": summ, "seasons": per}
    return out


def corr(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


# ---------------- 막판 지표·다리 ----------------

def tail(rows, qs):
    """한섭 T_end 칸 (T − H ∈ {−12, −6, 0}). out_TH = T=H에서 허용 밖 (시즌, 지점) 목록, mae_end_mean = 칸마다 시즌 평균 |e|의 칸 평균, max_end = T_end 최대 |e|."""
    R = [r for r in rows if r["server"] == "KR" and float(r["pct"]) in qs and r.get("err") is not None]
    cells = collections.defaultdict(list)
    out_TH, mx = [], (0.0, None)
    for r in R:
        k = E49.end_rel(r)
        if k is None:
            continue
        q = float(r["pct"])
        e = abs(r["err"])
        cells[(k, q)].append(e)
        if e > mx[0]:
            mx = (e, f"{r['season']} {q:g}% T{float(r['T']):g}")
        if abs(k) < 1e-9 and e > TOL_END[q] + 1e-12:
            out_TH.append(f"{r['season']} {q:g}% {r['err']:+.1%}")
    return {"n_out_TH": len(out_TH), "out_TH": sorted(out_TH, key=lambda x: (E49.snum(x.split()[0]), x)),
            "mae_end_mean": round(statistics.mean(statistics.mean(v) for v in cells.values()), 6) if cells else None,
            "max_end": [round(mx[0], 5), mx[1]], "n_cells": len(cells)}


def tail_leg(T, M):
    """막판 다리 (사전 고정): Z가 (가)~(라)를 모두 지나면 True."""
    why = []
    if T["Z"]["n_out_TH"] > T["R0"]["n_out_TH"] - TAIL_N:
        why.append(f"(가) T=H 허용 밖 {T['R0']['n_out_TH']} → {T['Z']['n_out_TH']} (2개 이상 줄지 않음)")
    if not T["Z"]["mae_end_mean"] < T["R0"]["mae_end_mean"] - TAIL_MAE - 1e-12:
        why.append(f"(나) T_end 시즌 평균 |e| {T['R0']['mae_end_mean']:.3%} → {T['Z']['mae_end_mean']:.3%} (0.5%p 넘게 줄지 않음)")
    g = E49.guards("Z", M, "R0", "R0")
    if g:
        why.append("(다) guard: " + " · ".join(g))
    if M["Z"]["score"] > M["R0"]["score"] + TAIL_SCORE_TOL + 1e-12:
        why.append(f"(라) score {M['R0']['score']:.3%} → {M['Z']['score']:.3%} (0.3%p 넘게 나쁨)")
    return {"ok": not why, "why_not": why}


# ---------------- 고르기 ----------------

def cmd_sel(argv):
    ap = argparse.ArgumentParser(prog="e_410.py sel")
    ap.add_argument("res")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pts", help="e_48 점 표 (하한 어김용, 없으면 그 지표만 뺀다)")
    ap.add_argument("--work")
    a = ap.parse_args(argv)
    R = E49.Res(a.res)
    need = need_list(R)
    D = {"about": ("분석 4.10 — 모양 갱신(--shape-update) 다시 (lab4/e_410.py sel). 사전 고정 docs 「4.10 사전 고정」. 개발 검증 (맞힐 시즌 S21~S41, 뽑기 2000). "
                   "1·3%는 main 줄 + M1(후보마다 앞 맞힐 시즌으로, T_end 칸 따로 — 4.9 B와 같음), 10%는 ref 줄. |e| = criteria_v2 평균 시즌 중앙 |e|."),
         "rule": __doc__.split("고르기 (사전 고정):")[1].strip(), "need": [list(x) for x in need]}
    if all(R.has(n, p, 1) for n in NAMES for p in ("main", "ref")):
        rank = {n: i for i, n in enumerate(NAMES)}
        Mm = {n: R.main(n)["M"] for n in NAMES}
        Mr = {n: R.ref(n)["M"] for n in NAMES}
        sm = E49.pick(NAMES, Mm, rank, "R0")
        sr = E49.pick(NAMES, Mr, rank, "R0")
        D["main_1_3"] = {"select": sm, "table": E49.table(NAMES, Mm), "metrics": {n: E49.brief(Mm[n]) for n in NAMES},
                         "guards_Z_ref_R0": E49.guards("Z", Mm, "R0", "R0")}
        D["ref_10"] = {"select": sr, "table": E49.table(NAMES, Mr), "metrics": {n: E49.brief(Mr[n]) for n in NAMES},
                       "guards_Z_ref_R0": E49.guards("Z", Mr, "R0", "R0")}
        Tm = {n: tail(R.main(n)["rows"], (1.0, 3.0)) for n in NAMES}
        Tr = {n: tail(R.ref(n)["rows"], (10.0,)) for n in NAMES}
        lm, lr = tail_leg(Tm, Mm), tail_leg(Tr, Mr)
        D["main_1_3"].update(tail=Tm, tail_leg=lm)
        D["ref_10"].update(tail=Tr, tail_leg=lr)
        fin = lambda sel, leg: "Z" if sel["pick"] == "Z" or leg["ok"] else "R0"  # noqa: E731
        D["final"] = {"main_1_3": fin(sm, lm), "ref_10": fin(sr, lr),
                      "how": {"main_1_3": "주 규칙" if sm["pick"] == "Z" else ("막판 다리" if lm["ok"] else "R0"),
                              "ref_10": "주 규칙" if sr["pick"] == "Z" else ("막판 다리" if lr["ok"] else "R0")}}
        for k, T_, L_ in (("1·3%", Tm, lm), ("10%", Tr, lr)):
            print(f"막판 {k}: 허용 밖 R0 {T_['R0']['n_out_TH']} · Z {T_['Z']['n_out_TH']} | T_end 시즌 평균 |e| {T_['R0']['mae_end_mean']:.3%} · {T_['Z']['mae_end_mean']:.3%} | 다리 {L_}")
        print("최종:", D["final"])
        E49.show(D["main_1_3"]["table"], f"1·3%: 고름 {sm['pick']} (0.5%p 안 {sm['within_0_5pp']})")
        print("   Z guard (참고):", D["main_1_3"]["guards_Z_ref_R0"])
        E49.show(D["ref_10"]["table"], f"10%: 고름 {sr['pick']} (0.5%p 안 {sr['within_0_5pp']})")
        print("   Z guard (참고):", D["ref_10"]["guards_Z_ref_R0"])
        D["report_Z_vs_R0"] = E49.report(R, "Z", "Z", a.pts, a.work)
        D["zdiag"] = zdiag(R)
        for k, v in D["zdiag"].items():
            print("Ẑ", k, json.dumps(v["summary"], ensure_ascii=False))
        seeds = {}
        for s in SEEDS[1:]:
            if all(R.has(n, p, s) for n in NAMES for p in ("main", "ref")):
                seeds[f"s{s}"] = {"main": {n: dict(E49.brief(R.main(n, s)["M"]), tail=tail(R.main(n, s)["rows"], (1.0, 3.0))) for n in NAMES},
                                  "ref": {n: dict(E49.brief(R.ref(n, s)["M"]), tail=tail(R.ref(n, s)["rows"], (10.0,))) for n in NAMES}}
        D["seeds"] = seeds
    D["source"] = R.src
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(D, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("→", a.out, "· 남은 실행", len(need))
    return 0


def main():
    cmds = {"run": cmd_run, "todo": cmd_todo, "sel": cmd_sel, "_inner": cmd_inner}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    return cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
