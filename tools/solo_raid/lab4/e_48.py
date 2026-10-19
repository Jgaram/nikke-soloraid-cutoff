"""분석 4.9 A (2026-10-19) — 옛 사진 무게 다시 짜기의 분석 단계 (기존 로직 그대로, 서술). 계획: docs/solo-raid-analysis.md 「4.9」.

predict.py·xval.py·backtest_web.py는 고치지 않는다 (S42 기본 동작). 창은 이 프로세스 안에서만 predict.predict를 감싸 넣고 xval.main을 부른다 (m5run.py와 같은 방식).

  python3 -I tools/solo_raid/lab4/e_48.py run --win W [--win-min N] [--patch no_interval ...] -- <xval.py 인자 --out F.json>
  python3 -I tools/solo_raid/lab4/e_48.py r0check --main MAIN.json --ref REF.json --out BT.json   # R0 = backtest_F 재현 확인
  python3 -I tools/solo_raid/lab4/e_48.py a1 RES --out diag48.json                                # A1 모으기 (diag48의 「a1」 열쇠만 고쳐 씀)

run — A1 창 R1: 촬영 시각 ≥ T − W인 점만 쓴다 (W 시간). 판정 시각은 예측이 % 변환에 쓰는 시각 h (load_obs의 h — A·B는 촬영 시각,
  게시 시각만 있는 C 줄은 구간 위 끝 h_hi = 게시 시각). 서버마다 창 안 사람(거르기 mono 뒤, 크게 어긋난 점 거르기 전)이 N명 미만이면
  그 서버는 모든 점(R0)으로 돌아간다 (기본 N 3, 0이면 늘 창). 수준 갱신(M4l, xval --level-update)은 xval이 predict 앞에서 전체 줄로 푼다 — 창은 predict 안에만 걸린다.
  줄마다 win(서버별 창 정보: 사람·줄·C 줄 수, 「확실히 창 안」 사람 = 촬영 구간 아래 끝 ≥ T − W, 돌아감 여부)을 남긴다. m5run의 warn은 늘 켠다.
a1 — RES/{R0,w6,w12,w24}/{MAIN,REF}.json(6시간 격자 T6~T120)과 같은 폴더의 {MAIN,REF}_S28.json·_S38.json(시즌 H 기준 T_end — 검토 보완 1)을 모은다.
  1·3%는 MAIN 줄 + M1(변형마다 그 변형의 앞 맞힐 시즌 줄로 — backtest_web.m1_for, 넓히기만), 10%는 REF 줄 그대로 (backtest_F와 같은 합침).
  창: T18+ = 6시간 격자의 18 ≤ T ≤ H와 T_end, T_main 18·24·36·48, T_end = 시즌마다 H−12·H−6·H. 「3명 이상 칸」 = 그 변형에서 창이 실제로 쓰인 칸(돌아가지 않음)만 R0와 견줌.
  관문 (검토 보완 2): 한섭 3% T18+ 3명 이상 칸의 평균 |e| 이득(R0 − 창) ≥ 0.27%p인 W가 하나라도 있으면 통과.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
sys.path.insert(0, HERE)
import predict as PR  # noqa: E402
import xval as XV  # noqa: E402
import m5run as M5  # noqa: E402

AN = os.path.join(ROOT, "data", "solo-raid", "analysis")
SERVERS = ("KR", "JP", "GL")
VARIANTS = ("R0", "w6", "w12", "w24")
T_GRID = tuple(float(t) for t in range(6, 121, 6))
T_MAIN = (18.0, 24.0, 36.0, 48.0)
GATE_PP = 0.0027            # 검토 보완 2: 4.8 이득 0.54%p의 절반
REF_48 = {"about": "4.8 근사 — 시즌 전체 state에서 게시 ≤ T인 줄, 중앙 추정만(뽑기 없음), 창 안 사람 3명 이상 칸 (4.8 표)",
          "KR3_T18_120": {"R0": 0.0236, "w6": 0.0182, "w12_R0": 0.0249, "w12": 0.0202, "n_w6": 389, "n_w12": 399},
          "KR3_TH": {"R0": 0.0094, "w6": 0.0042}, "KR10_T18_120": {"R0": 0.0484, "w6": 0.0433}}
EXTRA = {"S28": (118.5, 124.5, 130.5), "S38": (101.0, 107.0, 113.0)}   # 격자에 없는 T_end (H 130.5·113)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha1_of(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


# ---------------- run: 창 R1 ----------------

def servers_of(o):
    if o.get("unused"):
        return []
    if o.get("server"):
        return [o["server"]]
    return [s for s in (o.get("mix") or {}) if s in SERVERS]


def window_rows(rows, T, w, nmin):
    """→ (남길 줄, 서버별 창 정보). 줄 dict는 그대로 (복사하지 않음 — predict는 screen_z만 붙인다)."""
    lo = T - w
    info, fb = {}, {}
    for s in SERVERS:
        rs = [o for o in rows if s in servers_of(o)]
        if not rs:
            continue
        ins = [o for o in rs if o["h"] >= lo - 1e-9]
        strict = [o for o in rs if (o.get("span") or (o["h"], o["h"]))[0] >= lo - 1e-9]
        n_in = len({o["person"] for o in ins})
        fb[s] = n_in < nmin
        info[s] = {"w": w, "n_all": len({o["person"] for o in rs}), "n_in": n_in, "n_in_strict": len({o["person"] for o in strict}),
                   "rows_all": len(rs), "rows_in": len(ins), "rows_in_C": sum(1 for o in ins if o["grade"] == "C"),
                   "rows_in_mixed": sum(1 for o in ins if not o.get("server")), "fallback": fb[s]}
    keep = [o for o in rows if o.get("unused") or o["h"] >= lo - 1e-9 or any(fb.get(s) for s in servers_of(o))]
    return keep, info


def apply_window(w, nmin):
    orig = PR.predict

    def predict_w(state, prior, T, opts, draws=2000, seed=1, comps_split=True, priors=None, rows=None):
        if rows is None:
            _, c0, H = PR.season_info(state, allow_fallback=opts.get("allow_cstart_fallback", False))
            rows, _ = PR.load_obs(state, T, H, c0, opts.get("grades", "all"), opts.get("c_delay", 0.0),
                                  opts.get("server_mix"), opts.get("unknown"))
        keep, info = window_rows(rows, T, w, nmin)
        out = orig(state, prior, T, opts, draws, seed, comps_split, priors, keep)
        for s, d in info.items():
            if s in out["servers"]:
                out["servers"][s]["win"] = d
        return out
    PR.predict = predict_w
    orig_score = XV.score

    def score_win(out, akey, tgt, T, fold_name, train, crit):
        rows = orig_score(out, akey, tgt, T, fold_name, train, crit)
        for r in rows:
            r["win"] = (out["servers"].get(r["server"]) or {}).get("win")
        return rows
    XV.score = score_win
    XV.SLIM_KEYS = tuple(XV.SLIM_KEYS) + ("win",)


def cmd_run(argv):
    if "--" not in argv:
        raise SystemExit("run … -- <xval 인자>")
    k = argv.index("--")
    ap = argparse.ArgumentParser(prog="e_48.py run")
    ap.add_argument("--win", type=float, required=True, help="창 너비 W (시간): 촬영 시각 ≥ T − W인 점만")
    ap.add_argument("--win-min", type=int, default=3, help="서버별 창 안 사람이 이보다 적으면 그 서버는 R0 (0이면 늘 창)")
    ap.add_argument("--patch", action="append", default=[])
    a = ap.parse_args(argv[:k])
    patches = ["warn"] + [p for p in a.patch if p != "warn"]
    bad = [p for p in patches if p not in M5.PATCHES]
    if bad:
        raise SystemExit(f"모르는 patch: {bad}")
    xargs = argv[k + 1:]
    if "--out" not in xargs:
        raise SystemExit("--out을 준다 (analysis/에 쓰지 않는다)")
    M5.apply(patches)
    apply_window(a.win, a.win_min)
    sys.argv = ["xval.py"] + xargs
    rc = XV.main()
    out = xargs[xargs.index("--out") + 1]
    res = jl(out)
    res["run"]["lab_patches"] = {"patches": patches, "mixdir": None, "emdir": None,
                                 "window": {"w": a.win, "min_persons": a.win_min, "time": "h (C 줄은 h_hi = 게시 시각)", "per_server": True}}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
        f.write("\n")
    return rc


# ---------------- r0check: R0 = backtest_F ----------------

def cmd_r0check(argv):
    ap = argparse.ArgumentParser(prog="e_48.py r0check")
    ap.add_argument("--main", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True, help="backtest_web 출력 (세션 임시)")
    ap.add_argument("--against", default=os.path.join(AN, "backtest_F.json"))
    a = ap.parse_args(argv)
    old = jl(a.against)
    why = next((c.get("why") for c in old["source"].get("m1_check", []) if c.get("why")), None)
    cmd = [sys.executable, "-I", os.path.join(TOOLS, "backtest_web.py"), "--main", a.main, "--ref", a.ref, "--out", a.out]
    if why:
        cmd += ["--m1-known-diff", why]
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout.strip(), p.stderr.strip()[-500:])
    if p.returncode:
        raise SystemExit("backtest_web 실패")
    new = jl(a.out)
    diff = []
    for k in ("T", "pcts", "model"):
        if new[k] != old[k]:
            diff.append(k)
    nrows = 0
    for s in sorted(set(old["seasons"]) | set(new["seasons"])):
        so, sn = old["seasons"].get(s), new["seasons"].get(s)
        if so is None or sn is None:
            diff.append(f"{s}: 한쪽에만")
            continue
        for k in ("train", "c_start", "H", "m1"):
            if so[k] != sn[k]:
                diff.append(f"{s} {k}")
        for sv in sorted(set(so["srv"]) | set(sn["srv"])):
            po, pn = so["srv"].get(sv, {}), sn["srv"].get(sv, {})
            for q in sorted(set(po) | set(pn)):
                ro, rn = po.get(q), pn.get(q)
                if ro != rn:
                    bad = [x[0] for x, y in zip((ro or {}).get("rows", []), (rn or {}).get("rows", [])) if x != y]
                    diff.append(f"{s} {sv} {q}: {('T ' + ','.join(f'{t:g}' for t in bad)) if bad else '머리 또는 줄 수'}")
                nrows += len((rn or {}).get("rows", []))
    res = {"same": not diff, "rows_compared": nrows, "diff": diff[:50], "n_diff": len(diff),
           "against": {"file": os.path.relpath(a.against, ROOT), "sha1": sha1_of(a.against), "source": old["source"]},
           "new_source": new["source"]}
    print(json.dumps({k: res[k] for k in ("same", "rows_compared", "n_diff", "diff")}, ensure_ascii=False))
    with open(a.out + ".check.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return 0 if res["same"] else 1


# ---------------- a1: 모으기 ----------------

def season_H(s):
    sj = jl(os.path.join(ROOT, "data", "solo-raid", s, "state", "season.json"))
    c0 = dt.datetime.strptime(sj["c_start"][:16], "%Y-%m-%d %H:%M")
    end = dt.datetime.strptime(sj["end"][:16], "%Y-%m-%d %H:%M") + dt.timedelta(minutes=1)   # 04:59:59 → 05:00 (predict.season_info와 같음)
    return (end - c0).total_seconds() / 3600


def load_variant(d):
    """→ (main 줄, ref 줄, 원 파일 정보). 격자 실행 + S28·S38 T_end 실행, T ≤ H만."""
    src, main, ref = {}, [], []
    for part, acc in (("MAIN", main), ("REF", ref)):
        for suf in ("", "_S28", "_S38"):
            p = os.path.join(d, f"{part}{suf}.json")
            if not os.path.exists(p):
                raise SystemExit(f"없음: {p}")
            x = jl(p)
            ra = x["run"].get("args") or {}
            src[f"{part}{suf}"] = {"file": os.path.relpath(p, os.path.dirname(d)), "sha1": sha1_of(p),
                                   "tool_head": x["run"].get("tool_head"), "tools_dirty": x["run"].get("tools_dirty"),
                                   "lab_patches": x["run"].get("lab_patches"), "failures": len(x.get("failures") or []),
                                   "args": {k: ra.get(k) for k in ("scheme", "targets", "T", "draws", "seed", "boot", "slim",
                                                                    "predict_opt", "unknown_opt", "level_update")}}
            for r in x["rows"]:
                if r.get("scheme") != "recent5":
                    continue
                if suf and r["season"] != suf[1:]:
                    continue
                if not suf and r["season"] in EXTRA and float(r["T"]) in EXTRA[r["season"]]:
                    continue
                acc.append(r)
    return main, ref, src


def with_m1(main):
    """backtest_web와 같은 M1: 맞힐 시즌마다 앞 시즌의 격자 줄(T 12~96 일곱)로 배율 → 1·3% 줄(세 서버)의 80%를 넓힌다."""
    import backtest_web as BW
    grid = [r for r in main if float(r["T"]) in T_GRID]
    m1 = {s: BW.m1_for(grid, s) for s in sorted({r["season"] for r in main}, key=lambda x: int(x[1:]))}
    out = []
    for r in main:
        r = dict(r)
        tab = m1[r["season"]]
        c = PR.m1_factor({"m1": {"T": tab}}, float(r["T"])) if tab else 1.0
        if r.get("pred") and r.get("i80") and None not in r["i80"] and min(r["i80"]) > 0 and r["pct"] in (1.0, 3.0):
            m = r["pred"]
            lo, hi = r["i80"]
            lo, hi = m * (lo / m) ** c, m * (hi / m) ** c
            r["i80"] = [lo, hi]
            r["w80"] = hi / lo - 1
            if r.get("truth") is not None:
                r["in80"] = lo <= r["truth"] <= hi
        r["m1_c"] = c
        out.append(r)
    return out, m1


def tsets(H):
    end = [round(H - 12, 2), round(H - 6, 2), round(H, 2)]
    grid = [t for t in T_GRID if t <= H + 1e-9]
    return {"all": sorted(set(grid) | set(end)), "end": end, "grid": grid}


def label_T(r, H):
    t = float(r["T"])
    return {round(H - 12, 2): "H-12", round(H - 6, 2): "H-6", round(H, 2): "H"}.get(round(t, 2))


def stats(L, key_in=None):
    L = [r for r in L if r.get("err") is not None]
    if not L:
        return None
    ae = [abs(r["err"]) for r in L]
    g = collections.Counter(r.get("grade") for r in L)
    ins = [r["win"]["n_in"] for r in L if r.get("win")]
    out = {"n": len(L), "mean_abs_e": round(statistics.mean(ae), 5), "median_abs_e": round(statistics.median(ae), 5),
           "mean_e": round(statistics.mean(r["err"] for r in L), 5), "low_ratio": round(sum(r["err"] < 0 for r in L) / len(L), 3),
           "ok_ratio": round(sum(abs(r["err"]) <= r["tol"] for r in L) / len(L), 3),
           "cov80": round(sum(bool(r.get("in80")) for r in L) / len(L), 3),
           "med_w80": round(statistics.median(r["w80"] for r in L if r.get("w80") is not None), 4),
           "grades": {k: g[k] for k in sorted(g, key=str)},
           "cov80_AB": (round(sum(bool(r.get("in80")) for r in L if r.get("grade") in ("A", "B")) / sum(r.get("grade") in ("A", "B") for r in L), 3)
                        if any(r.get("grade") in ("A", "B") for r in L) else None),
           "med_persons": round(statistics.median(r.get("n_persons") or 0 for r in L), 1)}
    if ins:
        out["med_n_in"] = statistics.median(ins)
        out["n_fallback"] = sum(1 for r in L if r.get("win") and r["win"]["fallback"])
    return out


def cell_key(r):
    return (r["season"], round(float(r["T"]), 2), r["server"], float(r["pct"]))


def windows(rows, Hs, server, pct):
    """창별 줄 묶음: T18+ · T_main · T_end · T=H · T6~12."""
    R = [r for r in rows if r["server"] == server and float(r["pct"]) == pct and r.get("err") is not None]
    out = collections.defaultdict(list)
    for r in R:
        H = Hs[r["season"]]
        ts = tsets(H)
        t = round(float(r["T"]), 2)
        if t not in ts["all"]:
            continue
        if t >= 18:
            out["T18+"].append(r)
        if t in T_MAIN:
            out["T_main"].append(r)
        if t in ts["end"]:
            out["T_end"].append(r)
        if t == round(H, 2):
            out["T=H"].append(r)
        if t <= 12:
            out["T6-12"].append(r)
    return out


def by_T(rows, Hs, server, pct):
    R = [r for r in rows if r["server"] == server and float(r["pct"]) == pct and r.get("err") is not None]
    out = collections.defaultdict(list)
    for r in R:
        H = Hs[r["season"]]
        t = round(float(r["T"]), 2)
        if t not in tsets(H)["all"]:
            continue
        if t in T_GRID:
            out[f"T{t:g}"].append(r)
        lab = label_T(r, H)
        if lab:
            out[lab].append(r)
    order = [f"T{t:g}" for t in T_GRID] + ["H-12", "H-6", "H"]
    return {k: stats(out[k]) for k in order if out.get(k)}


def compare(r0, rv, Hs, server, pct, only_win):
    """같은 칸끼리 R0 vs 변형. only_win이면 변형에서 창이 실제로 쓰인 칸만 (창 안 사람 ≥ 문턱 — 4.8 「3명 이상 칸」)."""
    idx = {cell_key(r): r for r in r0}
    out = {}
    for wname, L in windows(rv, Hs, server, pct).items():
        pairs = []
        for r in L:
            if only_win and (not r.get("win") or r["win"]["fallback"]):
                continue
            b = idx.get(cell_key(r))
            if b is not None and b.get("err") is not None:
                pairs.append((b, r))
        if not pairs:
            continue
        sb, sv = stats([b for b, _ in pairs]), stats([r for _, r in pairs])
        out[wname] = {"n": len(pairs), "R0": sb, "var": sv, "gain_mean_abs_e": round(sb["mean_abs_e"] - sv["mean_abs_e"], 5),
                      "n_better": sum(abs(r["err"]) < abs(b["err"]) - 1e-12 for b, r in pairs),
                      "n_worse": sum(abs(r["err"]) > abs(b["err"]) + 1e-12 for b, r in pairs),
                      "n_seasons": len({r["season"] for _, r in pairs})}
    return out


def season_worst(r0, rv, Hs, server, pct):
    """시즌별 T18+ 평균 |e| (R0 · 변형) — 가장 나빠진 시즌 셋·나아진 셋."""
    a, b = windows(r0, Hs, server, pct)["T18+"], windows(rv, Hs, server, pct)["T18+"]
    ma = collections.defaultdict(list)
    mb = collections.defaultdict(list)
    for r in a:
        ma[r["season"]].append(abs(r["err"]))
    for r in b:
        mb[r["season"]].append(abs(r["err"]))
    d = {s: (statistics.mean(ma[s]), statistics.mean(mb[s])) for s in ma if s in mb}
    srt = sorted(d.items(), key=lambda kv: kv[1][1] - kv[1][0])
    f = lambda kv: {"season": kv[0], "R0": round(kv[1][0], 4), "var": round(kv[1][1], 4)}  # noqa: E731
    return {"most_better": [f(x) for x in srt[:3]], "most_worse": [f(x) for x in srt[::-1][:3]],
            "n_better": sum(v[1] < v[0] for v in d.values()), "n_seasons": len(d)}


def win_profile(rv, Hs):
    """창 정보 (한섭 3% 줄): 칸별 창 안 사람 분포·「확실히 창 안」 사람·C 줄 몫·돌아간 칸."""
    R = [r for r in rv if r["server"] == "KR" and float(r["pct"]) == 3.0 and r.get("win") and r.get("err") is not None
         and round(float(r["T"]), 2) in tsets(Hs[r["season"]])["all"] and float(r["T"]) >= 18]
    if not R:
        return None
    w = [r["win"] for r in R]
    rin = sum(x["rows_in"] for x in w)
    return {"n_cells_T18+": len(R), "n_fallback": sum(x["fallback"] for x in w),
            "n_in_median": statistics.median(x["n_in"] for x in w), "n_in_q10": sorted(x["n_in"] for x in w)[len(w) // 10],
            "n_in_strict_median": statistics.median(x["n_in_strict"] for x in w),
            "n_all_median": statistics.median(x["n_all"] for x in w),
            "rows_in_C_share": round(sum(x["rows_in_C"] for x in w) / rin, 3) if rin else None,
            "rows_in_mixed_share": round(sum(x["rows_in_mixed"] for x in w) / rin, 3) if rin else None,
            "cells_strict_lt3": sum(x["n_in_strict"] < 3 for x in w if not x["fallback"])}


def cmd_a1(argv):
    ap = argparse.ArgumentParser(prog="e_48.py a1")
    ap.add_argument("res")
    ap.add_argument("--out", required=True)
    ap.add_argument("--r0check", help="r0check 출력의 .check.json")
    a = ap.parse_args(argv)
    V = {}
    src = {}
    m1s = {}
    for v in VARIANTS:
        d = os.path.join(a.res, v)
        if not os.path.isdir(d):
            continue
        main, ref, src[v] = load_variant(d)
        main, m1s[v] = with_m1(main)
        V[v] = {"main": main, "ref": ref}
    if "R0" not in V:
        raise SystemExit("R0 없음")
    seasons = sorted({r["season"] for r in V["R0"]["main"]}, key=lambda x: int(x[1:]))
    Hs = {s: season_H(s) for s in seasons}
    pts = {"main": (1.0, 3.0), "ref": (10.0,)}
    res = {"about": ("분석 4.9 A1 — 4.8 「최근 사진만」을 T 절단 사본(tr)으로 다시. 기존 로직 그대로 창만 (lab4/e_48.py run). 개발 검증·서술 (고르기 아님). "
                     "R0 = backtest_F와 같은 설정 (1·3% F + 앞 시즌 M1, 10% M2a). 창 W: 촬영 시각 h ≥ T − W인 점만, 서버별 창 안 사람 3명 미만이면 R0로."),
           "seasons": seasons, "H": {s: round(h, 2) for s, h in Hs.items()},
           "windows_def": {"T18+": "6시간 격자의 18 ≤ T ≤ H와 T_end", "T_main": list(T_MAIN), "T_end": "시즌마다 H−12·H−6·H", "T=H": "시즌 H"},
           "source": src, "m1": {v: m1s[v] for v in m1s}, "ref_48": REF_48, "variants": {}}
    if a.r0check:
        c = jl(a.r0check)
        res["r0check"] = {k: c[k] for k in ("same", "rows_compared", "n_diff", "diff", "against")}
    for v, d in V.items():
        out = {}
        for part in ("main", "ref"):
            for q in pts[part]:
                for sv in SERVERS:
                    rows = d[part]
                    k = f"{sv}{q:g}"
                    W = windows(rows, Hs, sv, q)
                    e = {"windows": {w: stats(L) for w, L in W.items()}}
                    if sv == "KR":
                        e["by_T"] = by_T(rows, Hs, sv, q)
                    if v != "R0":
                        e["vs_R0_win_cells"] = compare(V["R0"][part], rows, Hs, sv, q, True)
                        e["vs_R0_all_cells"] = compare(V["R0"][part], rows, Hs, sv, q, False)
                        if sv == "KR":
                            e["season_T18+"] = season_worst(V["R0"][part], rows, Hs, sv, q)
                    out[k] = e
        if v != "R0":
            out["win_profile"] = win_profile(d["main"], Hs)
        res["variants"][v] = out
    # 관문 (검토 보완 2)
    g = {}
    for v in VARIANTS[1:]:
        if v not in res["variants"]:
            continue
        c = res["variants"][v]["KR3"]["vs_R0_win_cells"].get("T18+")
        ca = res["variants"][v]["KR3"]["vs_R0_all_cells"].get("T18+")
        g[v] = {"win_cells": None if not c else {"n": c["n"], "R0": c["R0"]["mean_abs_e"], "var": c["var"]["mean_abs_e"], "gain": c["gain_mean_abs_e"]},
                "all_cells": None if not ca else {"n": ca["n"], "R0": ca["R0"]["mean_abs_e"], "var": ca["var"]["mean_abs_e"], "gain": ca["gain_mean_abs_e"]}}
    best = max((x["win_cells"]["gain"], v) for v, x in g.items() if x["win_cells"]) if g else None
    res["gate"] = {"rule": "한섭 3% T18+ 「창이 쓰인 칸」 평균 |e| 이득(R0 − 창) ≥ 0.27%p인 W가 하나라도 있으면 통과 (검토 보완 2, 4.8과 같은 칸 정의)",
                   "threshold": GATE_PP, "by_w": g, "best": None if not best else {"w": best[1], "gain": best[0]},
                   "pass": bool(best and best[0] >= GATE_PP - 1e-12),
                   "pass_all_cells": any(x["all_cells"] and x["all_cells"]["gain"] >= GATE_PP - 1e-12 for x in g.values())}
    old = jl(a.out) if os.path.exists(a.out) else {}
    old["a1"] = res
    old.setdefault("about", "분석 4.9 A (docs/solo-raid-analysis.md 「4.9 A 결과」) — lab4/e_48.py가 씀. 개발 검증·서술. 열쇠: a1 (A2~A5는 이어 붙인다)")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
        f.write("\n")
    show(res)
    return 0


def pct(x):
    return "—" if x is None else f"{x * 100:.2f}"


def show(res):
    print("관문:", json.dumps(res["gate"], ensure_ascii=False))
    for k in ("KR3", "KR1", "KR10", "JP3", "GL3"):
        print(f"\n== {k}")
        for v, d in res["variants"].items():
            e = d.get(k)
            if not e:
                continue
            ws = e["windows"]
            line = " · ".join(f"{w} n{ws[w]['n']} |e| {pct(ws[w]['mean_abs_e'])} e {pct(ws[w]['mean_e'])} 포함 {ws[w]['cov80']:.2f} 폭 {pct(ws[w]['med_w80'])}"
                              for w in ("T18+", "T_main", "T_end", "T=H") if ws.get(w))
            print(f"  {v:4s} {line}")
            if v != "R0":
                c = e["vs_R0_win_cells"]
                print("       창 칸: " + " · ".join(f"{w} n{c[w]['n']} {pct(c[w]['R0']['mean_abs_e'])}→{pct(c[w]['var']['mean_abs_e'])} "
                                                  f"(나음 {c[w]['n_better']}) 포함 {c[w]['R0']['cov80']:.2f}→{c[w]['var']['cov80']:.2f}"
                                                  for w in ("T18+", "T_main", "T_end", "T=H") if c.get(w)))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "r0check", "a1"):
        raise SystemExit(__doc__)
    return {"run": cmd_run, "r0check": cmd_r0check, "a1": cmd_a1}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main() or 0)
