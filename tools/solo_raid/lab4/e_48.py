"""분석 4.9 A (2026-10-19) — 옛 사진 무게 다시 짜기의 분석 단계 (기존 로직 그대로, 서술). 계획: docs/solo-raid-analysis.md 「4.9」.

predict.py·xval.py·backtest_web.py는 고치지 않는다 (S42 기본 동작). 창은 이 프로세스 안에서만 predict.predict를 감싸 넣고 xval.main을 부른다 (m5run.py와 같은 방식).

  python3 -I tools/solo_raid/lab4/e_48.py run --win W [--win-min N] [--patch no_interval ...] -- <xval.py 인자 --out F.json>
  python3 -I tools/solo_raid/lab4/e_48.py r0check --main MAIN.json --ref REF.json --out BT.json   # R0 = backtest_F 재현 확인
  python3 -I tools/solo_raid/lab4/e_48.py a1 RES --out diag48.json                                # A1 모으기 (diag48의 「a1」 열쇠만 고쳐 씀)
  python3 -I tools/solo_raid/lab4/e_48.py pts --work XW --out PTS [--mode tr,full] [--jobs 4]      # A2~A5 점 표 (세션 임시, S19~S41)
  python3 -I tools/solo_raid/lab4/e_48.py a2 --pts PTS --work XW --sigma-out SIG --out diag48.json  # A2 + 맞힐 시즌별 σ_c 파일 SIG/S{n}.json
  python3 -I tools/solo_raid/lab4/e_48.py a3|a4 --pts PTS --work XW --out diag48.json              # A3 미상 줄 섞임 · A4 수준 갱신
  python3 -I tools/solo_raid/lab4/e_48.py a5 RES --pts PTS --work XW --out diag48.json             # A5 연속성 지표 (A1 원 결과)
  python3 -I tools/solo_raid/lab4/e_48.py bsum RES --out diag48.json                              # B 재료 요약 「b_inputs」 (M1 진단 포함)

run — A1 창 R1: 촬영 시각 ≥ T − W인 점만 쓴다 (W 시간). 판정 시각은 예측이 % 변환에 쓰는 시각 h (load_obs의 h — A·B는 촬영 시각,
  게시 시각만 있는 C 줄은 구간 위 끝 h_hi = 게시 시각). 서버마다 창 안 사람(거르기 mono 뒤, 크게 어긋난 점 거르기 전)이 N명 미만이면
  그 서버는 모든 점(R0)으로 돌아간다 (기본 N 3, 0이면 늘 창). 수준 갱신(M4l, xval --level-update)은 xval이 predict 앞에서 전체 줄로 푼다 — 창은 predict 안에만 걸린다.
  줄마다 win(서버별 창 정보: 사람·줄·C 줄 수, 「확실히 창 안」 사람 = 촬영 구간 아래 끝 ≥ T − W, 돌아감 여부)을 남긴다. m5run의 warn은 늘 켠다.
a1 — RES/{R0,w6,w12,w24}/{MAIN,REF}.json(6시간 격자 T6~T120)과 같은 폴더의 {MAIN,REF}_S28.json·_S38.json(시즌 H 기준 T_end — 검토 보완 1)을 모은다.
  1·3%는 MAIN 줄 + M1(변형마다 그 변형의 앞 맞힐 시즌 줄로 — backtest_web.m1_for, 넓히기만), 10%는 REF 줄 그대로 (backtest_F와 같은 합침).
  창: T18+ = 6시간 격자의 18 ≤ T ≤ H와 T_end, T_main 18·24·36·48, T_end = 시즌마다 H−12·H−6·H. 「3명 이상 칸」 = 그 변형에서 창이 실제로 쓰인 칸(돌아가지 않음)만 R0와 견줌.
  관문 (검토 보완 2): 한섭 3% T18+ 3명 이상 칸의 평균 |e| 이득(R0 − 창) ≥ 0.27%p인 W가 하나라도 있으면 통과.
pts·a2~a5·bsum — A2~A5 (2026-10-19, 서술). 점 표 위에서 R0와 같은 무게·변환의 중앙 추정(뽑기 없음, PAV 없음)으로 부분 집합 내재 컷을 다시 낸다
  (S30 T24·72·120 한섭 1·3·10%가 predict 중앙과 0.001% 안). 시즌마다 prior는 그 시즌을 맞힐 때 쓰는 앞 5시즌(A1 캐시), S19만 앞이 없어 S20.
  a2는 칸별 d = 시각 칸 내재 컷 − 최근 칸 내재 컷, σ_c 꼴 비교(학습 시즌 하나씩 빼기), 맞힐 시즌별 누수 없는 σ_c((i) prior 학습 시즌 · (ii) 앞 시즌 전부),
  창 구성·하루 시간대·C 줄 판정 시각. a3은 미상·라벨 줄 내재 컷 차이와 섞임 모형·U1, a4는 M4l(L0)·×2 안(L1)·끔(L2)과 모양 어긋남, a5는 연속성 지표.
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

_SEASON_H = {}


def season_H(s):
    if s not in _SEASON_H:
        _SEASON_H[s] = _season_H(s)
    return _SEASON_H[s]


def _season_H(s):
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


# ======================= A2~A5 (2026-10-19) — 점 표(pts) 위의 중앙 추정 셈 (뽑기 없음, 서술) =======================
# pts: 시즌 × T마다 (T 절단 사본 tr 또는 시즌 전체 state full의 게시 ≤ T) load_obs(mono) 줄과, 그 시즌을 맞힐 때 쓰는 prior
#      (앞 5시즌 recent5 — A1·운영과 같은 캐시, main = 구간 관측 뺌, ref = 그대로)로 옮긴 값(P·k·D·σ 항)을 세션 임시 파일에 둔다.
#      사람은 번호로만 남긴다. 창(A1 run과 같은 판정·3명 문턱)·M4l(전체·창·목표 % ×2 안 라벨 줄)·크게 어긋난 점 거르기도 여기서 낸다.
# 나머지 명령(a2~a5)은 이 표로 R0와 같은 무게(Predictor.weights)·변환(est_curve 중앙, Z = 0)의 「부분 집합 내재 컷」을 다시 낸다.

A_SEASONS = tuple(f"S{n}" for n in range(19, 42))
A_TARGETS = tuple(f"S{n}" for n in range(21, 42))
QS = (1.0, 3.0, 10.0)
QKIND = {1.0: "main", 3.0: "main", 10.0: "ref"}
WVARS = {"w6": (6.0, "hi"), "w12": (12.0, "hi"), "w24": (24.0, "hi"),
         "w6m": (6.0, "mid"), "w6l": (6.0, "lo"), "w12m": (12.0, "mid"), "w12l": (12.0, "lo")}
NEAR = {1.0: (0.5, 2.0), 3.0: (1.5, 6.0)}     # L1: 목표 % ×2 안 (중앙 변환 최종 %)
MIN_P = 3
LN10 = math.log(10)
R_KIND, R_P, R_GR, R_H, R_LO, R_HI, R_HP, R_X, R_S = range(9)        # pts 줄 열
C_P, C_UY, C_IN, C_K, C_D, C_SP, C_SF = range(7)                      # pts 옮긴 값 열 (+ vs q 셋)


def own_train(s):
    """그 시즌을 맞힐 때 쓰는 prior의 학습 시즌 = 앞 5시즌(recent5, 운영·A1과 같음). S20은 S19 하나, S19는 앞이 없어 뒤 시즌 S20 하나."""
    n = int(s[1:])
    if n == 19:
        return ("S20",), "backward"
    return tuple(f"S{k}" for k in range(max(19, n - 5), n)), "forward"


def prior_file(train, work, kind):
    """A1과 같은 prior 캐시 (xval.prior_for, boot 60·seed 1). main은 m5run no_interval처럼 구간 칸을 빼고 WORK/alt_noint에."""
    import fit_prior as FP
    if kind == "ref":
        return XV.prior_for(tuple(train), work, 60, 1, {}, False)[1]
    orig = FP.load_units

    def lu(state, *a, **k):
        season, H, obs, units = orig(state, *a, **k)
        return season, H, obs, [u for u in units if u["cens"] is None]
    FP.load_units = lu
    try:
        return XV.prior_for(tuple(train), os.path.join(work, "alt_noint"), 60, 1, {}, False)[1]
    finally:
        FP.load_units = orig


def a_Ts(s):
    H = season_H(s)
    return sorted({t for t in T_GRID if t <= H + 1e-9} | set(EXTRA.get(s, ())))


def kind_of(o):
    """L 라벨 한섭 · U 미상(mono가 한섭으로 둠) · X 미상(mono가 뺌) · J·G 라벨 일섭·글섭."""
    if o["server"]:
        return {"KR": "L", "JP": "J", "GL": "G"}[o["server"]]
    if not o["unused"] and (o.get("mix") or {}).get("KR"):
        return "U"
    return "X"


def jtime(o, how):
    """창 안 판정 시각: hi = 예측이 % 변환에 쓰는 h (A1, C 줄은 게시 시각), lo = 촬영 구간 아래 끝, mid = 가운데."""
    if how == "hi":
        return o["h"]
    lo, hi = o["span"]
    return lo if how == "lo" else 0.5 * (lo + hi)


def r6(v):
    return round(v, 6)


def pts_season(job):
    s, mode, work, outdir = job
    import backtest as BT
    import fit_prior_multi as FM
    import m_update as MU
    full = FM.state_of(s)
    train, how = own_train(s)
    ppath = {k: prior_file(train, work, k) for k in ("main", "ref")}
    pri = {k: jl(p) for k, p in ppath.items()}
    pers = {}
    res = {"season": s, "mode": mode, "train": list(train), "train_how": how,
           "prior": {k: os.path.relpath(p, work) for k, p in ppath.items()},
           "tau_p": {k: pri[k]["season_diff"]["tau_p"]["value"] for k in pri}, "T": {}}
    for T in a_Ts(s):
        st = BT.trunc_state(full, T, os.path.join(work, "trunc", s), XV.extra_of(full)) if mode == "tr" else full
        _, c0, H = PR.season_info(st)
        Tc = min(T, H)
        unk = {"mode": "mono", "servers": ["KR"], "borrow_state": [FM.state_of(x) for x in train], "borrow_kind": "label",
               "mix_file": None, "no_gate": False, "fallback": "site"}
        rows, cnt = PR.load_obs(st, Tc, H, c0, "all", 0.0, None, unk)
        kinds = [kind_of(o) for o in rows]
        R = [[kinds[i], pers.setdefault(o["person"], len(pers)), o["grade"], r6(o["h"]), r6(o["span"][0]), r6(o["span"][1]),
              r6(o["hp"]), o["x"], o["s"]] for i, o in enumerate(rows)]
        cand = [i for i, k in enumerate(kinds) if k in "LU"]
        u = cnt.get("unknown") or {}
        rec = {"Tc": Tc, "H": H, "rows": R, "unknown": {k: u.get(k) for k in ("rows_assigned", "rows_removed", "persons_removed")},
               "kind": {}}
        for kind, prior in pri.items():
            opts = {"eps_wrong": 0.0} if kind == "main" else {}
            lv = {}
            tau0 = prior["season_diff"]["tau_p"]["value"]

            def mp(sub):
                r = MU.m_posterior(prior, sub, H, tau0, unknown=False)
                return None if not r else [r["mu"], r["sd"], r.get("persons")]
            if kind == "main":
                lv["all"] = mp(rows)
                if lv["all"]:
                    opts["_lvl"] = (lv["all"][0], lv["all"][1])
            pr = PR.Predictor(prior, H, opts)
            tr = pr.tr_central(0)
            sh = pr.shapes["KR"]
            cols = [None] * len(rows)
            for i in cand:
                o = rows[i]
                P = PR.p_full(tr, "KR", o["h"], o["x"], H)
                uy = math.log(o["x"]) + P * LN10
                ins = sh.inside(uy)
                Dy = sh.D_at(uy)
                vs = [(pr.tau_shape * (sh.D_at(math.log(q)) - Dy)) ** 2 + (PR.KAPPA * (math.log(q) - uy)) ** 2 for q in QS]
                cols[i] = [r6(P), r6(uy), int(ins), r6(sh.k_at(uy)), r6(Dy), r6(math.sqrt(tr["s0"] ** 2 + (tr["rho"] * P) ** 2)),
                           r6(PR.sigma_form(prior["transform"], o["h"], H))] + [r6(v) for v in vs]
            if kind == "main":
                for vn in ("w6", "w12", "w24"):
                    lv[vn] = mp([o for o in rows if o["h"] >= Tc - WVARS[vn][0] - 1e-9])
                for q, (a, b) in NEAR.items():
                    lv[f"near{q:g}"] = mp([o for i, o in enumerate(rows) if cols[i] and a <= math.exp(cols[i][C_UY]) <= b])

            def scr(idx):
                O = [rows[i] for i in idx]
                if not O:
                    return []
                W, ys = pr.weights("KR", O, tr)
                _, drop = pr.screen("KR", O, tr, W, ys)
                ds = {id(o) for o in drop}
                return [i for i in idx if id(rows[i]) in ds]
            var = {"all": {"fb": False, "n_in": len({rows[i]["person"] for i in cand}), "keep": None, "drop": scr(cand)}}
            if mode == "tr":
                for vn, (w, jt) in WVARS.items():
                    ins = [i for i in cand if jtime(rows[i], jt) >= Tc - w - 1e-9]
                    n_in = len({rows[i]["person"] for i in ins})
                    fb = n_in < MIN_P
                    var[vn] = {"fb": fb, "n_in": n_in, "keep": None if fb else ins, "drop": scr(cand if fb else ins)}
            rec["kind"][kind] = {"cols": cols, "lvl": lv, "tau": pr.tau_p, "var": var}
        res["T"][f"{T:g}"] = rec
    os.makedirs(os.path.join(outdir, mode), exist_ok=True)
    p = os.path.join(outdir, mode, f"{s}.json")
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(res, f, separators=(",", ":"))
    os.replace(p + ".tmp", p)
    return s, mode


def cmd_pts(argv):
    ap = argparse.ArgumentParser(prog="e_48.py pts")
    ap.add_argument("--work", required=True, help="A1의 xval --work (T 절단 사본·prior 캐시)")
    ap.add_argument("--out", required=True, help="점 표 폴더 (세션 임시)")
    ap.add_argument("--mode", default="tr,full")
    ap.add_argument("--seasons", default="S19-S41")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    seasons = sorted(XV.parse_targets(a.seasons), key=lambda x: int(x[1:]))
    for s in seasons:   # prior를 먼저 (일꾼끼리 같은 prior를 맞추지 않게) — S19·S20의 prior만 새로 맞춘다
        for k in ("main", "ref"):
            print(s, k, os.path.relpath(prior_file(own_train(s)[0], a.work, k), a.work), flush=True)
    jobs = [(s, m, a.work, a.out) for m in a.mode.split(",") for s in seasons]
    for x in XV._each(pts_season, jobs, a.jobs):
        print("pts", *x, flush=True)
    return 0


# ---------------- 점 표 읽기·부분 집합 내재 컷 ----------------

_SHAPES = {}


class Pts:
    def __init__(self, path, work):
        self.d = jl(path)
        self.s = self.d["season"]
        self.shape = {}
        for k, p in self.d["prior"].items():
            if p not in _SHAPES:
                _SHAPES[p] = PR.Shape(jl(os.path.join(work, p))["shape"]["servers"]["KR"])
            self.shape[k] = _SHAPES[p]

    def Ts(self):
        return sorted(float(t) for t in self.d["T"])

    def rec(self, T):
        return self.d["T"][f"{T:g}"]


def load_pts(ptdir, mode, work, seasons=A_SEASONS):
    return {s: Pts(os.path.join(ptdir, mode, f"{s}.json"), work) for s in seasons
            if os.path.exists(os.path.join(ptdir, mode, f"{s}.json"))}


def r0_level(pt, T, kind, lv="all"):
    """R0의 (δ, τ): main은 M4l(라벨 줄 전체) 결과 — 사람 모자라 없으면 (0, prior τ_P), ref는 (0, prior τ_P)."""
    K = pt.rec(T)["kind"][kind]
    x = K["lvl"].get(lv) if kind == "main" else None
    return (x[0], x[1]) if x else (0.0, pt.d["tau_p"][kind])


def est(pt, T, kind, idx, q, delta, tau, alpha=None):
    """줄 번호 idx만으로 낸 q 내재 컷 (R0와 같은 무게·변환, 중앙, PAV 없음) → dict 또는 None.
    ln 컷, Σw, 사람 묶음 분산 V(ln², n/(n−1) 보정), 사람 수 n, 무게 평균 k·ln10(kb)·P(m 1, Pb)·h(hb). alpha: 미상(U) 줄 무게 배수."""
    rec = pt.rec(T)
    K = rec["kind"][kind]
    sh = pt.shape[kind]
    vi = 7 + QS.index(q)
    m = math.exp(delta)
    per = collections.defaultdict(list)
    sw = swc = swk = swp = swh = 0.0
    for i in idx:
        c = K["cols"][i]
        if not c or not c[C_IN]:
            continue
        r = rec["rows"][i]
        P = c[C_P]
        uy = math.log(r[R_X]) + LN10 * max(m * P, 0.0)
        if not sh.inside(uy):
            continue
        ci = math.log(r[R_S]) - sh.lr_at(uy)
        w = 1.0 / ((c[C_K] * LN10) ** 2 * (c[C_SP] ** 2 + (tau * P) ** 2 + c[C_SF] ** 2) + c[vi] + PR.V0)
        if alpha is not None and r[R_KIND] == "U":
            w *= alpha
        if w <= 0:
            continue
        per[r[R_P]].append((w, ci))
        sw += w
        swc += w * ci
        swk += w * c[C_K] * LN10
        swp += w * P
        swh += w * r[R_H]
    if sw <= 0:
        return None
    cb = swc / sw
    n = len(per)
    V = sum(sum(w * (ci - cb) for w, ci in L) ** 2 for L in per.values()) / sw ** 2 * (n / (n - 1) if n > 1 else 1.0)
    return {"ln": sh.lr_at(math.log(q)) + cb, "sw": sw, "V": V, "n": n, "kb": swk / sw, "Pb": swp / sw, "hb": swh / sw}


def idx_of(pt, T, kind, var="all", kinds="LU", extra=None):
    """창 변형 var의 남는 줄 (크게 어긋난 점 거르기 뒤)에서 종류 kinds만. extra(i, 줄) → bool로 더 거른다."""
    rec = pt.rec(T)
    V = rec["kind"][kind]["var"][var]
    drop = set(V["drop"])
    base = V["keep"] if V.get("keep") is not None else range(len(rec["rows"]))
    return [i for i in base if rec["rows"][i][R_KIND] in kinds and i not in drop and rec["kind"][kind]["cols"][i]
            and (extra is None or extra(i, rec["rows"][i]))]


def truth_map():
    ak = jl(os.path.join(AN, "answer_key.json"))["rows"]
    return {(r["season"], r["server"], float(r["pct"])): r["truth"] for r in ak if r.get("truth") is not None}


def kst_hour(h):
    return (5.0 + h) % 24.0      # c_start = 05:00 KST


# ---------------- A2: 시즌 공통 변환 오차의 시간 구조 ----------------
# 시즌·T마다 라벨 한섭 줄을 촬영 시각 칸 j(6시간, j = 0이 최근 [T−6, T])로 나눠 칸마다 내재 컷 ĉ_j(R0 무게·변환)를 내고 d_j = ĉ_j − ĉ_0을 잰다.
# 사람 3명 이상 칸만. 표본 잡음 V = V_j + V_0 (사람 묶음 분산)을 빼고 남는 퍼짐이 σ_c — R2가 칸마다 더할 「점 수로 나눠지지 않는」 공통 오차.
# 단위: ln(컷, q마다) 또는 P(log10 % 변환) = d ÷ (칸의 무게 평균 k·ln10).

LAGG = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 7), (8, 11), (12, 15), (16, 99))   # 시각 칸 j 묶음 (j 1 = 6~12시간 전)
TBAND = ((0, 24), (24.01, 48), (48.01, 72), (72.01, 96), (96.01, 999))
FORMS = ("const", "table", "pow", "pdiff", "pdiff0")
TAB_K, TAB_FLOOR = 10, 0.05   # table: 칸 분산을 시각 칸 묶음 값 쪽으로 기록 10개 몫만큼 당기고(묶음은 전체 쪽으로), 전체 분산의 5% 아래로 내리지 않는다
FORM_DOC = {"const": "σ = s (시각 무관)", "table": "σ(시각 칸 묶음 × T 띠) 칸별 표 — 칸 분산 = (n·잡음 뺀 적률 + 10·묶음 분산)/(n + 10), 묶음은 전체 쪽으로 같은 식, 바닥 전체 분산의 5%",
            "pow": "σ = a·(lag/24)^b, lag = T − 칸의 무게 평균 h (시간, 3 아래는 3)", "pdiff": "σ² = s0² + β²·ΔP², ΔP = 칸의 P̄ − 최근 칸 P̄ (prior 중앙 변환, h → T에서 오를 몫)",
            "pdiff0": "σ = β·ΔP"}
HOURS = ((5, "23-05"), (11, "05-11"), (17, "11-17"), (23, "17-23"))   # 칸이 끝나는 KST 시각


def lag_bin(Tc, h):
    return max(0, math.ceil((Tc - h) / 6.0 - 1e-9) - 1)


def lagg_of(j):
    return next(g for g, (a, b) in enumerate(LAGG) if a <= j <= b)


def tband_of(T):
    return next(b for b, (a, c) in enumerate(TBAND) if a <= T <= c)


def hour_class(t):
    """시각 t(챌린지 시작 후 시간)가 끝인 6시간 칸의 KST 이름 (가장 가까운 05·11·17·23시)."""
    k = kst_hour(t)
    return min(HOURS, key=lambda x: min(abs(k - x[0]), 24 - abs(k - x[0])))[1]


def a2_records(pt, kinds="L", lvar="L0", truth=None, counts=None):
    """→ d 기록 목록. counts(dict)를 주면 (q, j)별 칸 사람 수 분포도 모은다 (3명 미만 칸 포함)."""
    out = []
    for T in pt.Ts():
        rec = pt.rec(T)
        Tc = rec["Tc"]
        for q in QS:
            kind = QKIND[q]
            d0, t0 = r0_level(pt, T, kind) if lvar == "L0" else (0.0, pt.d["tau_p"][kind])
            by = collections.defaultdict(list)
            for i in idx_of(pt, T, kind, "all", kinds):
                by[lag_bin(Tc, rec["rows"][i][R_H])].append(i)
            if counts is not None:
                for j, ids in by.items():
                    counts[(q, min(j, 20))].append(len({rec["rows"][i][R_P] for i in ids}))
            b0 = est(pt, T, kind, by[0], q, d0, t0) if by.get(0) else None
            if not b0 or b0["n"] < MIN_P:
                continue
            tv = truth.get((pt.s, "KR", q)) if truth else None
            for j, ids in sorted(by.items()):
                if j == 0:
                    continue
                b = est(pt, T, kind, ids, q, d0, t0)
                if not b or b["n"] < MIN_P:
                    continue
                out.append({"s": pt.s, "T": T, "Tc": Tc, "H": rec["H"], "q": q, "j": j, "lag": Tc - b["hb"], "dP": abs(b["Pb"] - b0["Pb"]),
                            "kb": b["kb"], "d": b["ln"] - b0["ln"], "V": b["V"] + b0["V"], "n": b["n"], "n0": b0["n"],
                            "hc": hour_class(Tc - 6 * j), "Thc": hour_class(Tc),
                            "e": None if tv is None else math.exp(b["ln"]) / tv - 1, "e0": None if tv is None else math.exp(b0["ln"]) / tv - 1})
    return out


def in_unit(recs, unit):
    if unit == "ln":
        return recs
    return [dict(r, d=r["d"] / r["kb"], V=r["V"] / r["kb"] ** 2) for r in recs]


def s2_form(form, p, r):
    if form == "const":
        return math.exp(2 * p[0])
    if form == "pow":
        return math.exp(2 * p[0]) * (max(r["lag"], 3.0) / 24.0) ** (2 * p[1])
    if form == "pdiff":
        return math.exp(2 * p[0]) + math.exp(2 * p[1]) * r["dP"] ** 2
    if form == "pdiff0":
        return math.exp(2 * p[0]) * r["dP"] ** 2 + 1e-12
    raise ValueError(form)


def nelder_mead(f, x0, step=0.7, iters=300, tol=1e-8):
    n = len(x0)
    pts = [list(x0)] + [[x0[i] + (step if i == k else 0.0) for i in range(n)] for k in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = sorted(range(n + 1), key=lambda i: vals[i])
        pts, vals = [pts[i] for i in o], [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = [sum(p[i] for p in pts[:-1]) / n for i in range(n)]
        xr = [2 * cen[i] - pts[-1][i] for i in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [3 * cen[i] - 2 * pts[-1][i] for i in range(n)]
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = [0.5 * (cen[i] + pts[-1][i]) for i in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for k in range(1, n + 1):
                    pts[k] = [0.5 * (pts[0][i] + pts[k][i]) for i in range(n)]
                    vals[k] = f(pts[k])
    i = min(range(n + 1), key=lambda i: vals[i])
    return pts[i], vals[i]


def as_arr(recs):
    import numpy as np
    return {k: np.array([float(r[k]) for r in recs]) for k in ("d", "V", "lag", "dP")}


def s2_arr(form, p, A):
    import numpy as np
    if form == "const":
        return np.full_like(A["d"], math.exp(2 * p[0]))
    if form == "pow":
        return math.exp(2 * p[0]) * (np.maximum(A["lag"], 3.0) / 24.0) ** (2 * p[1])
    if form == "pdiff":
        return math.exp(2 * p[0]) + math.exp(2 * p[1]) * A["dP"] ** 2
    if form == "pdiff0":
        return math.exp(2 * p[0]) * A["dP"] ** 2 + 1e-12
    raise ValueError(form)


def nll_arr(A, s2):
    import numpy as np
    t = s2 + A["V"]
    return float(0.5 * np.sum(np.log(2 * math.pi * t) + A["d"] ** 2 / t))


def mom(L):
    return max(1e-8, statistics.mean(r["d"] ** 2 - r["V"] for r in L))


def fit_form(form, recs):
    """→ 모수 dict (σ 꼴은 FORM_DOC). 정규 최대 우도 (d ~ N(0, σ² + V)), table은 잡음 뺀 적률."""
    if not recs:
        return None
    if form == "table":
        glob = mom(recs)
        cell, lg = collections.defaultdict(list), collections.defaultdict(list)
        for r in recs:
            cell[(lagg_of(r["j"]), tband_of(r["Tc"]))].append(r)
            lg[lagg_of(r["j"])].append(r)
        tab = []
        raw = lambda L: statistics.mean(r["d"] ** 2 - r["V"] for r in L)  # noqa: E731
        for g in range(len(LAGG)):
            L = lg.get(g, [])
            sg = max(TAB_FLOOR * glob, (len(L) * raw(L) + TAB_K * glob) / (len(L) + TAB_K)) if L else glob
            for b in range(len(TBAND)):
                C = cell.get((g, b), [])
                v = max(TAB_FLOOR * glob, (len(C) * raw(C) + TAB_K * sg) / (len(C) + TAB_K)) if C else sg
                tab.append([g, b, round(math.sqrt(v), 6)])
        return {"cells": tab, "lagg": [list(x) for x in LAGG], "tband": [list(x) for x in TBAND], "n": len(recs)}
    s0 = math.log(math.sqrt(mom(recs)))
    starts = {"const": [[s0]], "pow": [[s0, 0.3], [s0 - 1, 1.0], [s0 + 0.5, -0.2]],
              "pdiff": [[s0, s0 + 1], [s0 - 2, s0 + 2], [s0 - 4, s0]], "pdiff0": [[s0 + 1], [s0 + 3]]}[form]
    best = None
    A = as_arr(recs)
    for x0 in starts:
        p, v = nelder_mead(lambda p: nll_arr(A, s2_arr(form, p, A)), x0)
        if best is None or v < best[1]:
            best = (p, v)
    p = best[0]
    out = {"const": lambda: {"s": math.exp(p[0])}, "pow": lambda: {"a": math.exp(p[0]), "b": p[1]},
           "pdiff": lambda: {"s0": math.exp(p[0]), "beta": math.exp(p[1])}, "pdiff0": lambda: {"beta": math.exp(p[0])}}[form]()
    out = {k: round(v, 6) for k, v in out.items()}
    out.update(n=len(recs), nll=round(best[1], 3), _p=p)
    return out


def sigma_c(form, par, lag=None, dP=None, j=None, T=None):
    """맞힐 시즌 σ_c 파일의 꼴 모수 → σ (단위는 그 맞춤의 단위). table은 시각 칸 j와 T, 나머지는 lag(시간)·ΔP."""
    if form == "table":
        g, b = lagg_of(max(j, 1)), tband_of(T)
        return next(c[2] for c in par["cells"] if c[0] == g and c[1] == b)
    if form == "const":
        return par["s"]
    if form == "pow":
        return par["a"] * (max(lag, 3.0) / 24.0) ** par["b"]
    if form == "pdiff":
        return math.sqrt(par["s0"] ** 2 + (par["beta"] * dP) ** 2)
    if form == "pdiff0":
        return par["beta"] * dP


def s2_of(form, par, r):
    if form == "table":
        return sigma_c("table", par, j=r["j"], T=r["Tc"]) ** 2
    if form == "none":
        return 0.0
    return s2_form(form, par["_p"], r)


def score_form(form, par, recs):
    import numpy as np
    if not recs or (par is None and form != "none"):
        return None
    A = as_arr(recs)
    if form in ("table", "none"):
        s2 = np.array([s2_of(form, par, r) for r in recs])
    else:
        s2 = s2_arr(form, par["_p"], A)
    ll = -nll_arr(A, s2) / len(recs)
    cov = float(np.mean(np.abs(A["d"]) <= 1.2816 * np.sqrt(s2 + A["V"])))
    return {"ll": round(ll, 4), "cov80": round(cov, 3), "n": len(recs)}


def loso_forms(recs, seasons):
    """학습 시즌 하나씩 빼기: 나머지로 맞춘 꼴로 뺀 시즌의 d를 잰다 (기록당 평균 로그 우도·80% 포함). 시즌 평균도."""
    out = {}
    for form in FORMS + ("none",):
        per, allr, cov = [], [], 0.0
        for s in seasons:
            te = [r for r in recs if r["s"] == s]
            if not te:
                continue
            par = None if form == "none" else fit_form(form, [r for r in recs if r["s"] != s])
            sc = score_form(form, par, te)
            per.append(sc["ll"])
            allr.append(sc["ll"] * sc["n"])
            cov += sc["cov80"] * sc["n"]
        n = sum(1 for r in recs if r["s"] in seasons)
        out[form] = {"ll_rec": round(sum(allr) / n, 4), "ll_season_mean": round(statistics.mean(per), 4), "cov80": round(cov / n, 3),
                     "n": n, "n_seasons": len(per)}
    return out


def strip(par):
    return None if par is None else {k: v for k, v in par.items() if k != "_p"}


SETUPS = (("ln", 1.0), ("ln", 3.0), ("ln", 10.0), ("P", 3.0))


def a2_sigma_tables(recs):
    """시각 칸 묶음 × T 띠의 잡음 뺀 σ (참고 표, 모든 시즌)와 칸 묶음별 기록 수·시즌 수."""
    out = {}
    for unit, q in SETUPS:
        R = in_unit([r for r in recs if r["q"] == q], unit)
        tab = {}
        for g, (a, b) in enumerate(LAGG):
            L = [r for r in R if a <= r["j"] <= b]
            if not L:
                continue
            row = {"all": {"sigma": round(math.sqrt(mom(L)), 5), "rms_d": round(math.sqrt(statistics.mean(r["d"] ** 2 for r in L)), 5),
                           "rms_noise": round(math.sqrt(statistics.mean(r["V"] for r in L)), 5), "mean_d": round(statistics.mean(r["d"] for r in L), 5),
                           "n": len(L), "n_seasons": len({r["s"] for r in L}), "lag_med": round(statistics.median(r["lag"] for r in L), 1)}}
            for bi, (ta, tb) in enumerate(TBAND):
                C = [r for r in L if ta <= r["Tc"] <= tb]
                if len(C) >= 5:
                    row[f"T{int(ta)}-{int(tb) if tb < 900 else 'H'}"] = {"sigma": round(math.sqrt(mom(C)), 5), "n": len(C), "n_seasons": len({r["s"] for r in C})}
            tab[f"j{a}-{b}" if a != b else f"j{a}"] = row
        out[f"{unit}_q{q:g}"] = tab
    return out


def a2_target_files(recs_by, outdir):
    """맞힐 시즌(S21~S41)마다 누수 없는 σ_c: (i) 그 시즌 prior의 학습 시즌(앞 5시즌), (ii) 앞 시즌 전부. 꼴마다 모수와 그 시즌 d로 잰 앞으로 확인(forward)."""
    os.makedirs(outdir, exist_ok=True)
    allf = {}
    fwd = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in A_TARGETS:
        sets = {"i": list(own_train(t)[0]), "ii": [s for s in A_SEASONS if int(s[1:]) < int(t[1:])]}
        res = {"target": t, "about": ("분석 4.9 A2 — 맞힐 시즌 σ_c (누수 없음: 그 시즌보다 앞 학습 시즌의 T 절단 사본 tr만, 각 학습 시즌은 자기 앞 5시즌 prior로 옮김 — "
                                      "S19만 앞이 없어 S20 prior). lab4/e_48.py a2가 씀. B의 R2 재료 (고르기 아님)."),
               "def": {"d": "칸 j(6시간, 0 = 최근 [T−6, T]) 내재 컷 ĉ_j − ĉ_0, 라벨 한섭 줄, 사람 3명 이상 칸", "sigma": "d ~ N(0, σ_c² + V_j + V_0) — 잡음 뺀 공통 오차 (1σ)",
                       "units": "ln_q*: ln 컷 (그 q), P_q3: log10 % 변환 (d ÷ 칸 무게 평균 k·ln10, 3% 무게)", "forms": FORM_DOC,
                       "lvar": "L0 = R0처럼 M4l(라벨 줄 전체) 수준 적용, L2 = 수준 갱신 끔 (δ 0, prior τ_P)",
                       "use": "sigma_c(form, par, lag=T−h̄_j, dP=|P̄_j − P̄_0|, j=j, T=T) (e_48.py)"},
               "sets": sets, "fits": {}, "forward": {}}
        for (kinds, lvar), recs in recs_by.items():
            key = f"{lvar}" if kinds == "L" else f"{lvar}_{kinds}"
            res["fits"][key], res["forward"][key] = {}, {}
            for unit, q in SETUPS:
                R = in_unit([r for r in recs if r["q"] == q], unit)
                te = [r for r in R if r["s"] == t]
                su = f"{unit}_q{q:g}"
                res["fits"][key][su], res["forward"][key][su] = {}, {}
                for sn, ss in sets.items():
                    tr = [r for r in R if r["s"] in ss]
                    fits = {f: fit_form(f, tr) for f in FORMS}
                    res["fits"][key][su][sn] = {"n_rec": len(tr), "n_seasons": len({r["s"] for r in tr}), **{f: strip(p) for f, p in fits.items()}}
                    sc = {f: score_form(f, fits[f], te) for f in FORMS}
                    sc["none"] = score_form("none", None, te)
                    res["forward"][key][su][sn] = sc
                    for f, v in sc.items():
                        if v:
                            fwd[(key, su, sn)][f].append((v["ll"], v["n"], v["cov80"]))
        p = os.path.join(outdir, f"{t}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        allf[t] = {"file": os.path.basename(p), "sha1": sha1_of(p), "n_train": {k: len(v) for k, v in sets.items()}}
    summ = {}
    for (key, su, sn), d in fwd.items():
        summ.setdefault(key, {}).setdefault(su, {})[sn] = {
            f: {"ll_rec": round(sum(a * n for a, n, _ in L) / sum(n for _, n, _ in L), 4), "cov80": round(sum(c * n for _, n, c in L) / sum(n for _, n, _ in L), 3),
                "n_targets": len(L)} for f, L in d.items()}
    return allf, summ


def a2_windows(P, truth):
    """창 구성 (A1 창 판정 — h, C 줄은 게시 시각)과 칸이 끝나는 KST 시각별 이득, C 줄 판정 시각(hi·mid·lo)별 창 안 사람과 창 내재 컷 치우침."""
    cells = []
    for s in A_TARGETS:
        pt = P[s]
        for T in pt.Ts():
            if T < 18:
                continue
            rec = pt.rec(T)
            Tc, H = rec["Tc"], rec["H"]
            rows = rec["rows"]
            d0, t0 = r0_level(pt, T, "main")
            c = {"s": s, "T": T, "hc": hour_class(Tc), "grid": T in T_GRID, "end": round(T, 2) in tsets(season_H(s))["end"], "TH": abs(T - season_H(s)) < 1e-6,
                 "main": T in T_MAIN}
            cand = [i for i, r in enumerate(rows) if r[R_KIND] in "LU"]
            win = [i for i in cand if rows[i][R_H] >= Tc - 6 - 1e-9]
            prev = [i for i in cand if Tc - 12 - 1e-9 <= rows[i][R_H] < Tc - 6 - 1e-9]
            g = collections.Counter(rows[i][R_GR] for i in win)
            c["comp"] = {"rows": len(win), "persons": len({rows[i][R_P] for i in win}), "A": g["A"], "B": g["B"], "C": g["C"],
                         "label_rows": sum(rows[i][R_KIND] == "L" for i in win), "label_persons": len({rows[i][R_P] for i in win if rows[i][R_KIND] == "L"}),
                         "strict_persons": len({rows[i][R_P] for i in win if rows[i][R_LO] >= Tc - 6 - 1e-9}),
                         "AB_persons": len({rows[i][R_P] for i in win if rows[i][R_GR] != "C"}),
                         "posted_rows": sum(1 for i in cand if rows[i][R_HP] >= Tc - 6 - 1e-9),
                         "prev_persons": len({rows[i][R_P] for i in prev})}
            for q in (1.0, 3.0):
                tv = truth.get((s, "KR", q))
                if tv is None:
                    continue
                e_all = est(pt, T, "main", idx_of(pt, T, "main", "all"), q, d0, t0)
                c[f"e_all_{q:g}"] = math.exp(e_all["ln"]) / tv - 1 if e_all else None
                for vn in ("w6", "w6m", "w6l", "w12", "w12m", "w12l", "w24"):
                    V = rec["kind"]["main"]["var"][vn]
                    e = est(pt, T, "main", idx_of(pt, T, "main", vn), q, d0, t0)
                    c.setdefault(vn, {"n_in": V["n_in"], "fb": V["fb"]})[f"e_{q:g}"] = math.exp(e["ln"]) / tv - 1 if e else None
                if q == 3.0:
                    for part, f in (("AB", lambda i, r: r[R_GR] != "C"), ("C", lambda i, r: r[R_GR] == "C")):
                        ids = [i for i in idx_of(pt, T, "main", "all", "LU", f) if rows[i][R_H] >= Tc - 6 - 1e-9]
                        e = est(pt, T, "main", ids, q, d0, t0)
                        c[f"w6_{part}"] = None if not e or e["n"] < MIN_P else math.exp(e["ln"]) / tv - 1
            cells.append(c)
    return cells


def summ_e(L, key):
    v = [x for x in L if x is not None]
    if not v:
        return None
    return {"n": len(v), "mean_abs_e": round(statistics.mean(abs(x) for x in v), 5), "mean_e": round(statistics.mean(v), 5)}


def a2_window_summary(cells, a1):
    out = {"by_hour": {}, "judge": {}}
    for _, hc in HOURS:
        L = [c for c in cells if c["hc"] == hc and c["grid"]]
        if not L:
            continue
        cp = [c["comp"] for c in L]
        rws = sum(x["rows"] for x in cp) or 1
        e_all = [c.get("e_all_3") for c in L]
        e_w6 = [c["w6"].get("e_3") if not c["w6"]["fb"] else c.get("e_all_3") for c in L]
        both = [(c["w6_AB"], c["w6_C"]) for c in L if c.get("w6_AB") is not None and c.get("w6_C") is not None]
        Ts = sorted({c["T"] for c in L})
        g1 = None
        if a1:
            r0, w6 = a1["variants"]["R0"]["KR3"]["by_T"], a1["variants"]["w6"]["KR3"]["by_T"]
            ks = [f"T{t:g}" for t in Ts if f"T{t:g}" in r0 and f"T{t:g}" in w6]
            g1 = {"T": [t for t in Ts], "R0": round(statistics.mean(r0[k]["mean_abs_e"] for k in ks), 5),
                  "w6": round(statistics.mean(w6[k]["mean_abs_e"] for k in ks), 5)} if ks else None
        out["by_hour"][hc] = {
            "n_cells": len(L), "T": Ts,
            "persons_med": statistics.median(x["persons"] for x in cp), "strict_persons_med": statistics.median(x["strict_persons"] for x in cp),
            "AB_persons_med": statistics.median(x["AB_persons"] for x in cp), "label_persons_med": statistics.median(x["label_persons"] for x in cp),
            "prev_persons_med": statistics.median(x["prev_persons"] for x in cp), "posted_rows_med": statistics.median(x["posted_rows"] for x in cp),
            "share_rows": {"A": round(sum(x["A"] for x in cp) / rws, 3), "B": round(sum(x["B"] for x in cp) / rws, 3), "C": round(sum(x["C"] for x in cp) / rws, 3),
                           "label": round(sum(x["label_rows"] for x in cp) / rws, 3)},
            "central_e3_all": summ_e(e_all, None), "central_e3_w6": summ_e(e_w6, None),
            "w6_C_minus_AB": None if not both else {"n": len(both), "mean": round(statistics.mean(math.log1p(b) - math.log1p(a) for a, b in both), 5)},
            "a1_draw_median_KR3": g1}
    for vn in ("w6", "w6m", "w6l", "w12", "w12m", "w12l", "w24"):
        res = {}
        for wn, f in (("T18+", lambda c: True), ("T_main", lambda c: c["main"]), ("T_end", lambda c: c["end"]), ("T=H", lambda c: c["TH"])):
            L = [c for c in cells if f(c)]
            if not L:
                continue
            ein = [c for c in L if not c[vn]["fb"]]
            res[wn] = {"n_cells": len(L), "n_fallback": len(L) - len(ein), "n_in_med": statistics.median(c[vn]["n_in"] for c in L),
                       "n_in_q10": sorted(c[vn]["n_in"] for c in L)[len(L) // 10],
                       "e3_win_cells": summ_e([c[vn].get("e_3") for c in ein], None), "e3_R0_same_cells": summ_e([c.get("e_all_3") for c in ein], None),
                       "e3_all_cells": summ_e([c[vn].get("e_3") if not c[vn]["fb"] else c.get("e_all_3") for c in L], None),
                       "e1_all_cells": summ_e([c[vn].get("e_1") if not c[vn]["fb"] else c.get("e_all_1") for c in L], None)}
        out["judge"][vn] = res
    out["R0_central"] = {wn: summ_e([c.get("e_all_3") for c in cells if f(c)], None)
                         for wn, f in (("T18+", lambda c: True), ("T_main", lambda c: c["main"]), ("T_end", lambda c: c["end"]), ("T=H", lambda c: c["TH"]))}
    return out


def a2_hours_of_day(P):
    """시즌 끝(T = H, tr) 한섭 후보 줄(라벨 + mono 미상)의 촬영(A·B)·게시 시각을 KST 6시간 칸으로 (21시즌 합)."""
    shot, post, postC = collections.Counter(), collections.Counter(), collections.Counter()
    for s in A_TARGETS:
        pt = P[s]
        rec = pt.rec(pt.Ts()[-1])
        for r in rec["rows"]:
            if r[R_KIND] not in "LU":
                continue
            if r[R_GR] != "C":
                shot[hour_class(6 * math.ceil(r[R_H] / 6 - 1e-9))] += 1
            post[hour_class(6 * math.ceil(r[R_HP] / 6 - 1e-9))] += 1
            if r[R_GR] == "C":
                postC[hour_class(6 * math.ceil(r[R_HP] / 6 - 1e-9))] += 1
    f = lambda c: {k: round(c[k] / max(1, sum(c.values())), 3) for _, k in HOURS} | {"n": sum(c.values())}  # noqa: E731
    return {"about": "칸 이름 = 그 6시간 칸이 끝나는 KST 시각 앞뒤 (23-05 = 23시~05시)", "shot_AB": f(shot), "post_all": f(post), "post_C": f(postC)}


def a2_resid_by_hour(recs, form="pow"):
    """꼴 하나(전체 시즌으로 맞춤) 뒤 표준화 잔차 z = d/√(σ²+V)를 칸이 끝나는 KST 시각별로 — 하루 주기 구조가 꼴 밖에 남는지."""
    out = {}
    for unit, q in SETUPS[:2] + SETUPS[3:]:
        R = in_unit([r for r in recs if r["q"] == q], unit)
        par = fit_form(form, R)
        z = collections.defaultdict(list)
        for r in R:
            z[r["hc"]].append(r["d"] / math.sqrt(s2_of(form, par, r) + r["V"]))
        out[f"{unit}_q{q:g}"] = {hc: {"n": len(z[hc]), "mean_z": round(statistics.mean(z[hc]), 3), "mean_z2": round(statistics.mean(x * x for x in z[hc]), 3)}
                                 for _, hc in HOURS if z.get(hc)}
    return out


def a2_bin_vs_truth(recs):
    """T = H(시즌 끝)에서 칸별 내재 컷의 정답 대비 e — 칸이 끝나는 KST 시각·시각 칸 묶음별 평균 e·rms (라벨 한섭 3%)."""
    out = {}
    R = [r for r in recs if r["q"] == 3.0 and r.get("e") is not None and abs(r["T"] - season_H(r["s"])) < 1e-6]
    for _, hc in HOURS:
        L = [r for r in R if r["hc"] == hc]
        if L:
            out[hc] = {"n": len(L), "mean_e": round(statistics.mean(r["e"] for r in L), 5), "rms_e": round(math.sqrt(statistics.mean(r["e"] ** 2 for r in L)), 5)}
    return out


def cmd_a2(argv):
    ap = argparse.ArgumentParser(prog="e_48.py a2")
    ap.add_argument("--pts", required=True, help="pts 폴더 (tr·full)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--sigma-out", required=True, help="맞힐 시즌별 σ_c 파일 폴더 (세션 임시)")
    ap.add_argument("--out", required=True, help="diag48.json (「a2」 열쇠만 고쳐 씀)")
    a = ap.parse_args(argv)
    truth = truth_map()
    P = load_pts(a.pts, "tr", a.work)
    PF = load_pts(a.pts, "full", a.work)
    old = jl(a.out) if os.path.exists(a.out) else {}
    a1 = old.get("a1")
    counts, counts_lu = collections.defaultdict(list), collections.defaultdict(list)
    recs = {}
    for kinds in ("L", "LU"):
        for lvar in ("L0", "L2"):
            R = []
            for s in A_SEASONS:
                if s in P:
                    R += a2_records(P[s], kinds, lvar, truth, (counts if kinds == "L" else counts_lu) if lvar == "L0" else None)
            recs[(kinds, lvar)] = R
            print(kinds, lvar, len(R), flush=True)
    recs_full = []
    for s in A_SEASONS:
        if s in PF:
            recs_full += a2_records(PF[s], "L", "L0", truth)
    seasons = [s for s in A_SEASONS if s in P]
    res = {"about": ("분석 4.9 A2 — 시즌 공통 변환 오차의 시간 구조 (lab4/e_48.py pts·a2). 개발 검증·서술 (고르기 아님). "
                     "d = 시각 칸(6시간) 내재 컷 − 최근 칸 [T−6, T] 내재 컷 (R0와 같은 무게·변환, 중앙, 라벨 한섭 줄, 사람 3명 이상 칸). "
                     "S19~S41 T 절단 사본(tr), 시즌마다 자기 앞 5시즌 prior (S19만 S20 prior). σ_c = d에서 사람 묶음 표본 분산을 뺀 퍼짐."),
           "forms": FORM_DOC, "lagg": [list(x) for x in LAGG], "tband": [list(x) for x in TBAND], "n_rec": {f"{k}_{l}": len(v) for (k, l), v in recs.items()},
           "n_rec_full": len(recs_full),
           "source": {"pts": "세션 임시 lab4/a2/pts/{tr,full}/S{n}.json (A1의 xw 사본·prior 캐시, S19·S20 T 절단 사본과 prior만 새로)",
                      "prior": {s: {"train": P[s].d["train"], "how": P[s].d["train_how"], **{k: os.path.basename(v) for k, v in P[s].d["prior"].items()}}
                                for s in seasons}}}
    res["bin_persons"] = {}
    for nm, cc in (("label", counts), ("label+unknown", counts_lu)):
        tab = {}
        for (q, j), L in sorted(cc.items()):
            if q != 3.0 or j > 16:
                continue
            tab[f"j{j}"] = {"n_cells": len(L), "median": statistics.median(L), "share_ge3": round(sum(x >= MIN_P for x in L) / len(L), 3)}
        res["bin_persons"][nm] = tab
    res["sigma_table"] = {f"{k}_{l}": a2_sigma_tables(v) for (k, l), v in recs.items() if l == "L0" or k == "L"}
    res["sigma_table_full_L_L0"] = a2_sigma_tables(recs_full)
    # tr vs full (같은 시즌·T·칸끼리)
    ix = {(r["s"], r["T"], r["q"], r["j"]): r for r in recs[("L", "L0")]}
    com = [(ix[(r["s"], r["T"], r["q"], r["j"])], r) for r in recs_full if (r["s"], r["T"], r["q"], r["j"]) in ix]
    res["tr_vs_full"] = {}
    for q in QS:
        C = [x for x in com if x[0]["q"] == q]
        if C:
            res["tr_vs_full"][f"q{q:g}"] = {"n_common": len(C), "sigma_tr": round(math.sqrt(mom([a for a, _ in C])), 5),
                                            "sigma_full": round(math.sqrt(mom([b for _, b in C])), 5),
                                            "n_tr_only": sum(1 for r in recs[("L", "L0")] if r["q"] == q) - len(C),
                                            "n_full_only": sum(1 for r in recs_full if r["q"] == q) - len(C),
                                            "pow_tr": strip(fit_form("pow", in_unit([r for r in recs[("L", "L0")] if r["q"] == q], "ln"))),
                                            "pow_full": strip(fit_form("pow", [r for r in recs_full if r["q"] == q]))}
    # 꼴 비교 (학습 시즌 하나씩 빼기)
    res["loso"] = {}
    for (kinds, lvar), R in recs.items():
        key = lvar if kinds == "L" else f"{lvar}_{kinds}"
        res["loso"][key] = {}
        for unit, q in SETUPS:
            res["loso"][key][f"{unit}_q{q:g}"] = loso_forms(in_unit([r for r in R if r["q"] == q], unit), seasons)
        print("loso", key, flush=True)
    res["fit_all"] = {}
    for (kinds, lvar), R in recs.items():
        res["fit_all"][lvar if kinds == "L" else f"{lvar}_{kinds}"] = {
            f"{unit}_q{q:g}": {f: strip(fit_form(f, in_unit([r for r in R if r["q"] == q], unit))) for f in FORMS if f != "table"} for unit, q in SETUPS}
    res["per_season"] = {}
    for kinds in ("L", "LU"):
        by = collections.defaultdict(list)
        for r in recs[(kinds, "L0")]:
            if r["q"] == 3.0:
                by[r["s"]].append(r)
        res["per_season"][kinds] = {s: {"n": len(L), "T_min": min(r["T"] for r in L), "sigma": round(math.sqrt(mom(L)), 4),
                                        "mean_d": round(statistics.mean(r["d"] for r in L), 4),
                                        "pdiff_beta": strip(fit_form("pdiff", L))["beta"] if len(L) >= 5 else None}
                                    for s, L in sorted(by.items(), key=lambda kv: int(kv[0][1:]))}
    res["resid_by_hour_pow"] = a2_resid_by_hour(recs[("L", "L0")])
    res["bin_vs_truth_T=H"] = a2_bin_vs_truth(recs[("L", "L0")])
    files, fwd = a2_target_files({k: v for k, v in recs.items()}, a.sigma_out)
    res["targets"] = {"dir": "세션 임시 lab4/a2/sigma_c/{시즌}.json", "files": files, "forward_summary": fwd,
                      "about": "맞힐 시즌 t마다 (i) t의 prior 학습 시즌(앞 5시즌), (ii) t보다 앞 시즌 전부로 맞춘 꼴 → t의 d로 앞으로 확인 (기록당 평균 로그 우도, 80% 포함)"}
    cells = a2_windows(P, truth)
    res["windows"] = a2_window_summary(cells, a1)
    res["hours_of_day"] = a2_hours_of_day(P)
    old["a2"] = res
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(os.path.join(a.sigma_out, "_records.json"), "w", encoding="utf-8") as f:   # d 기록 (세션 임시 — B가 다시 맞출 때)
        json.dump({f"{k}_{l}": v for (k, l), v in recs.items()}, f)
    print(json.dumps({k: res[k] for k in ("n_rec", "tr_vs_full")}, ensure_ascii=False, indent=1))
    for key, d in res["loso"].items():
        for su, x in d.items():
            print(key, su, " · ".join(f"{f} {v['ll_rec']:.3f}/{v['cov80']:.2f}" for f, v in x.items()))
    return 0


# ---------------- A3: 미상 줄 섞임 ----------------
# 같은 시각 묶음 안에서 라벨 한섭 줄(L)만 · mono 미상 줄(U)만으로 낸 내재 컷의 차이 Δ = ln Ĉ_U − ln Ĉ_L (R0 무게·변환, 사람 3명 이상씩).
# 섞임 모형: 미상 사람의 함축 3% 컷(마지막 24시간 점의 중앙값) − 라벨 사람 중앙값 z가 한섭 0 · 일섭 o_J · 글섭 o_G(그 시즌 정답 서버 비 ln) 셋의 섞임이라고 보고
# 몫 f_J·f_G를 EM으로 (퍼짐은 라벨 사람의 MAD). 섞임으로 설명되는 치우침 b = f_J·o_J + f_G·o_G.

REMAIN_47 = {"JP": ((24, 0.70), (48, 0.65), (96, 0.50), (120, 0.36)), "GL": ((24, 0.75), (48, 0.67), (96, 0.63), (120, 0.39))}  # 4.7: 거르기 뒤 남는 몫
KR_KEEP_47 = 0.975          # 4.7: 잘못 빠진 한섭 2~3%
ALPHAS = (1.0, 0.5, 0.25, 0.0)


def interp(tab, T):
    if T <= tab[0][0]:
        return tab[0][1]
    for (a, va), (b, vb) in zip(tab, tab[1:]):
        if T <= b:
            return va + (vb - va) * (T - a) / (b - a)
    return tab[-1][1]


def a3_groups(Tc):
    g = [(f"day{d + 1}", (24.0 * d, 24.0 * (d + 1))) for d in range(6) if 24 * d < Tc]
    return g + [("last6", (Tc - 6, 1e9)), ("last12", (Tc - 12, 1e9)), ("last24", (Tc - 24, 1e9)), ("all", (-1e9, 1e9))]


def tcls(s, T):
    """칸 묶음 이름: T=H · T_end · T_main · T18+ (겹침)."""
    H = season_H(s)
    out = ["T18+"] if T >= 18 else ["T6-12"]
    if T in T_MAIN:
        out.append("T_main")
    if round(T, 2) in tsets(H)["end"]:
        out.append("T_end")
    if abs(T - H) < 1e-6:
        out.append("T=H")
    return out


def point_imp(pt, T, kind, i, q, delta, tau):
    e = est(pt, T, kind, [i], q, delta, tau)
    return None if e is None else e["ln"]


def em_mix(z, sig, offs, iters=300):
    """z: 미상 사람 z, 성분 평균 (0, o_J, o_G), 같은 sd sig → 몫 (f_KR, f_J, f_G)."""
    f = [0.8, 0.1, 0.1]
    for _ in range(iters):
        acc = [0.0, 0.0, 0.0]
        for x in z:
            p = [fk * math.exp(-0.5 * ((x - o) / sig) ** 2) for fk, o in zip(f, offs)]
            t = sum(p) or 1e-300
            for k in range(3):
                acc[k] += p[k] / t
        f = [a / len(z) for a in acc]
    return f


def a3(P, truth):
    grp, u1, mix = [], [], []
    ratios = {}
    for s in A_TARGETS:
        k3 = truth.get((s, "KR", 3.0))
        ratios[s] = {sv: (math.log(truth[(s, sv, 3.0)] / k3) if truth.get((s, sv, 3.0)) and k3 else None) for sv in ("JP", "GL")}
    mean_o = {sv: statistics.mean(v[sv] for v in ratios.values() if v[sv] is not None) for sv in ("JP", "GL")}
    for s in A_TARGETS:
        pt = P[s]
        for T in pt.Ts():
            rec = pt.rec(T)
            Tc = rec["Tc"]
            rows = rec["rows"]
            d0, t0 = r0_level(pt, T, "main")
            for q in (1.0, 3.0):
                tv = truth.get((s, "KR", q))
                if tv is None:
                    continue
                for gname, (a, b) in a3_groups(Tc):
                    f = lambda i, r, a=a, b=b: a - 1e-9 <= r[R_H] < b  # noqa: E731
                    eL = est(pt, T, "main", idx_of(pt, T, "main", "all", "L", f), q, d0, t0)
                    eU = est(pt, T, "main", idx_of(pt, T, "main", "all", "U", f), q, d0, t0)
                    if eL and eU and eL["n"] >= MIN_P and eU["n"] >= MIN_P:
                        grp.append({"s": s, "T": T, "q": q, "g": gname, "eL": math.exp(eL["ln"]) / tv - 1, "eU": math.exp(eU["ln"]) / tv - 1,
                                    "D": eU["ln"] - eL["ln"], "nL": eL["n"], "nU": eU["n"], "cls": tcls(s, T)})
                for vn in ("all", "w6", "w12"):
                    ids = idx_of(pt, T, "main", vn)
                    full = est(pt, T, "main", ids, q, d0, t0)
                    lab = est(pt, T, "main", [i for i in ids if rows[i][R_KIND] == "L"], q, d0, t0)
                    r = {"s": s, "T": T, "q": q, "v": vn, "cls": tcls(s, T), "share_U": None if not full else round(1 - (lab["sw"] if lab else 0) / full["sw"], 4),
                         "nL": lab["n"] if lab else 0,
                         "fb": bool(rec["kind"]["main"]["var"][vn]["fb"]) if vn != "all" else False}
                    for al in ALPHAS:
                        e = est(pt, T, "main", ids, q, d0, t0, alpha=al) if al > 0 else lab
                        r[f"a{al:g}"] = None if not e else math.exp(e["ln"]) / tv - 1
                    u1.append(r)
            # 섞임 모형 (3%, 마지막 24시간)
            if T < 24 or ratios[s]["JP"] is None and ratios[s]["GL"] is None:
                continue
            ids = idx_of(pt, T, "main", "all", "LU", lambda i, r: r[R_H] >= Tc - 24 - 1e-9)
            per = collections.defaultdict(list)
            for i in ids:
                v = point_imp(pt, T, "main", i, 3.0, d0, t0)
                if v is not None:
                    per[(rows[i][R_KIND], rows[i][R_P])].append(v)
            zL = [statistics.median(v) for (k, _), v in per.items() if k == "L"]
            zU = [statistics.median(v) for (k, _), v in per.items() if k == "U"]
            if len(zL) < 5 or len(zU) < 5:
                continue
            mu = statistics.median(zL)
            sig = max(0.01, 1.4826 * statistics.median(abs(x - mu) for x in zL))
            oJ = ratios[s]["JP"] if ratios[s]["JP"] is not None else mean_o["JP"]
            oG = ratios[s]["GL"] if ratios[s]["GL"] is not None else mean_o["GL"]
            fk, fj, fg = em_mix([x - mu for x in zU], sig, (0.0, oJ, oG))
            rJ, rG = interp(REMAIN_47["JP"], Tc), interp(REMAIN_47["GL"], Tc)
            lab_p = collections.Counter()
            for r_ in rows:
                if r_[R_KIND] in "LJG":
                    lab_p[(r_[R_KIND], r_[R_P])] += 1
            nl = collections.Counter(k for k, _ in lab_p)
            mix.append({"s": s, "T": T, "cls": tcls(s, T), "nL": len(zL), "nU": len(zU), "sig": sig, "oJ": oJ, "oG": oG, "fJ": fj, "fG": fg,
                        "b_mix": fj * oJ + fg * oG, "D_obs": statistics.mean(zU) - mu, "D_obs_med": statistics.median(zU) - mu,
                        "pi_pre": (fj / rJ + fg / rG) / (fk / KR_KEEP_47 + fj / rJ + fg / rG),
                        "label_nonKR_share": (nl["J"] + nl["G"]) / max(1, nl["L"] + nl["J"] + nl["G"])})
    return grp, u1, mix


def cmean(L, key):
    v = [x[key] for x in L if x.get(key) is not None]
    return round(statistics.mean(v), 5) if v else None


def a3_summary(grp, u1, mix):
    out = {"about": ("Δ = ln Ĉ(미상 U만) − ln Ĉ(라벨 L만), 같은 시각 묶음·R0 무게·변환 (tr, mono). e는 정답 대비. "
                     "U1 = 미상 줄 무게 α배(α 0은 라벨 줄만) 중앙 추정의 정답 대비 e (그 묶음 라벨 사람 3명 이상 칸만). 섞임 모형은 마지막 24시간 사람별 함축 3% 컷 (머리말).")}
    # T = H 하루차별 (4.8 견줌)
    tH = {}
    for g in ("day1", "day2", "day3", "day4", "day5", "last6", "all"):
        L = [x for x in grp if x["q"] == 3.0 and x["g"] == g and "T=H" in x["cls"]]
        if L:
            tH[g] = {"n": len(L), "eL": cmean(L, "eL"), "eU": cmean(L, "eU"), "D": cmean(L, "D")}
    out["T=H_by_day_q3"] = tH
    out["T=H_last6_worst_q3"] = sorted(({"season": x["s"], "D": round(x["D"], 4), "eU": round(x["eU"], 4), "eL": round(x["eL"], 4)}
                                        for x in grp if x["q"] == 3.0 and x["g"] == "last6" and "T=H" in x["cls"]), key=lambda d: d["D"])[:5]
    byT = {}
    for T in T_GRID:
        e = {}
        for g in ("all", "last6", "last24"):
            L = [x for x in grp if x["q"] == 3.0 and x["g"] == g and x["T"] == T]
            if L:
                e[g] = {"n": len(L), "D": cmean(L, "D"), "eL": cmean(L, "eL"), "eU": cmean(L, "eU")}
        if e:
            byT[f"T{T:g}"] = e
    out["by_T_q3"] = byT
    win = {}
    for c in ("T18+", "T_main", "T_end", "T=H"):
        for g in ("all", "last6", "last12", "last24"):
            for q in (1.0, 3.0):
                L = [x for x in grp if x["q"] == q and x["g"] == g and c in x["cls"]]
                if L:
                    win[f"{c}/{g}/q{q:g}"] = {"n": len(L), "D": cmean(L, "D"), "eL": cmean(L, "eL"), "eU": cmean(L, "eU"),
                                              "share_D_neg": round(sum(x["D"] < 0 for x in L) / len(L), 3)}
    out["by_window"] = win
    # U1
    u = {}
    for vn in ("all", "w6", "w12"):
        for c in ("T18+", "T_main", "T_end", "T=H"):
            for q in (1.0, 3.0):
                L = [x for x in u1 if x["v"] == vn and c in x["cls"] and x["q"] == q and x["nL"] >= MIN_P and all(x.get(f"a{a:g}") is not None for a in ALPHAS)]
                if not L:
                    continue
                u[f"{vn}/{c}/q{q:g}"] = {"n": len(L), "share_U_med": statistics.median(x["share_U"] for x in L if x["share_U"] is not None),
                                         **{f"a{a:g}": {"mean_abs_e": round(statistics.mean(abs(x[f"a{a:g}"]) for x in L), 5),
                                                        "mean_e": round(statistics.mean(x[f"a{a:g}"] for x in L), 5)} for a in ALPHAS}}
    out["U1_central"] = u
    # 섞임 모형
    m = {}
    for c in ("T18+", "T_main", "T_end", "T=H"):
        L = [x for x in mix if c in x["cls"]]
        if not L:
            continue
        expl = [x["b_mix"] / x["D_obs"] for x in L if x["D_obs"] < -0.002]
        m[c] = {"n": len(L), "fJ_med": round(statistics.median(x["fJ"] for x in L), 3), "fG_med": round(statistics.median(x["fG"] for x in L), 3),
                "f_other_mean": round(statistics.mean(x["fJ"] + x["fG"] for x in L), 3), "b_mix_mean": cmean(L, "b_mix"), "D_obs_mean": cmean(L, "D_obs"),
                "D_obs_med_mean": cmean(L, "D_obs_med"), "explained_med": round(statistics.median(expl), 3) if expl else None, "n_D_neg": len(expl),
                "pi_pre_med": round(statistics.median(x["pi_pre"] for x in L), 3), "label_nonKR_share_med": round(statistics.median(x["label_nonKR_share"] for x in L), 3),
                "sig_med": round(statistics.median(x["sig"] for x in L), 4), "nL_med": statistics.median(x["nL"] for x in L), "nU_med": statistics.median(x["nU"] for x in L)}
    out["mixture"] = m
    out["mixture_T=H"] = [{"season": x["s"], "fJ": round(x["fJ"], 3), "fG": round(x["fG"], 3), "b_mix": round(x["b_mix"], 4), "D_obs": round(x["D_obs"], 4),
                           "nL": x["nL"], "nU": x["nU"]} for x in mix if "T=H" in x["cls"]]
    return out


def cmd_a3(argv):
    ap = argparse.ArgumentParser(prog="e_48.py a3")
    ap.add_argument("--pts", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    P = load_pts(a.pts, "tr", a.work, A_TARGETS)
    grp, u1, mix = a3(P, truth_map())
    res = a3_summary(grp, u1, mix)
    old = jl(a.out) if os.path.exists(a.out) else {}
    old["a3"] = res
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps({k: res[k] for k in ("T=H_by_day_q3", "by_window", "mixture")}, ensure_ascii=False, indent=0)[:6000])
    return 0


# ---------------- A4: 수준 갱신(M4l)이 반대로 가는 조건 ----------------

def pt_points(pt, T, kind, idx, q, delta, tau):
    """(사람, 무게, c, ln y, z = k·ln10·P, x) — est와 같은 셈의 점 단위."""
    rec = pt.rec(T)
    K = rec["kind"][kind]
    sh = pt.shape[kind]
    vi = 7 + QS.index(q)
    m = math.exp(delta)
    out = []
    for i in idx:
        c = K["cols"][i]
        if not c or not c[C_IN]:
            continue
        r = rec["rows"][i]
        uy = math.log(r[R_X]) + LN10 * max(m * c[C_P], 0.0)
        if not sh.inside(uy):
            continue
        w = 1.0 / ((c[C_K] * LN10) ** 2 * (c[C_SP] ** 2 + (tau * c[C_P]) ** 2 + c[C_SF] ** 2) + c[vi] + PR.V0)
        out.append((r[R_P], w, math.log(r[R_S]) - sh.lr_at(uy), uy, c[C_K] * LN10 * c[C_P], r[R_X]))
    return out


def wslope(xs, ys, ws):
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx, my = sum(w * x for w, x in zip(ws, xs)) / sw, sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    return None if sxx <= 0 else sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx


def wcorr(xs, ys, ws):
    sw = sum(ws)
    mx, my = sum(w * x for w, x in zip(ws, xs)) / sw, sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    syy = sum(w * (y - my) ** 2 for w, y in zip(ws, ys))
    return None if sxx <= 0 or syy <= 0 else sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / math.sqrt(sxx * syy)


def bisect0(f, a=-0.8, b=0.8, it=40):
    fa, fb = f(a), f(b)
    if fa is None or fb is None or fa * fb > 0:
        return None
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm is None:
            return None
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def spearman(xs, ys):
    pr = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pr) < 5:
        return None

    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    a, b = rk([p[0] for p in pr]), rk([p[1] for p in pr])
    return {"rho": round(wcorr(a, b, [1.0] * len(a)), 3), "n": len(pr)}


def a4(P, truth):
    cells, win = [], []
    for s in A_TARGETS:
        pt = P[s]
        sh = pt.shape["main"]
        tv = {q: truth.get((s, "KR", q)) for q in QS}
        lr = lambda q: sh.lr_at(math.log(q))  # noqa: E731
        D1 = math.log(tv[1.0] / tv[3.0]) - (lr(1.0) - lr(3.0)) if tv[1.0] and tv[3.0] else None
        D10 = math.log(tv[10.0] / tv[3.0]) - (lr(10.0) - lr(3.0)) if tv[10.0] and tv[3.0] else None
        tau0 = pt.d["tau_p"]["main"]
        for T in pt.Ts():
            rec = pt.rec(T)
            Tc = rec["Tc"]
            lv = rec["kind"]["main"]["lvl"]
            ids = idx_of(pt, T, "main", "all")
            lab = idx_of(pt, T, "main", "all", "L")
            pts0 = pt_points(pt, T, "main", lab, 3.0, 0.0, tau0)
            c = {"s": s, "T": T, "cls": tcls(s, T), "D1": D1, "D10": D10, "tau0": tau0,
                 "d_all": lv["all"][0] if lv.get("all") else None, "sd_all": lv["all"][1] if lv.get("all") else None,
                 "d_near3": lv["near3"][0] if lv.get("near3") else None, "d_near1": lv["near1"][0] if lv.get("near1") else None,
                 "n_lab": len({p[0] for p in pts0})}
            if pts0:
                ws = [p[1] for p in pts0]
                xs = [p[5] for p in pts0]
                c["band"] = {k: round(sum(w for w, x in zip(ws, xs) if a_ <= x < b_) / sum(ws), 3)
                             for k, (a_, b_) in (("<1", (0, 1)), ("1-2", (1, 2)), ("2-5", (2, 5)), ("5-10", (5, 10)), (">=10", (10, 1e9)))}
                c["corr_z_lny"] = wcorr([p[4] for p in pts0], [p[3] for p in pts0], ws) if len(pts0) >= 5 else None
            late_ids = [i for i in lab if rec["rows"][i][R_H] >= Tc - 12 - 1e-9]
            lp = pt_points(pt, T, "main", late_ids, 3.0, 0.0, tau0)
            c["beta_shape"] = (wslope([p[3] for p in lp], [p[2] for p in lp], [p[1] for p in lp])
                               if len({p[0] for p in lp}) >= 5 and statistics.pstdev([p[3] for p in lp]) > 0.3 else None)
            # 하루차별 기울기 δ (라벨 줄, 24시간 칸, 3명 이상 칸 3개 이상)
            days = collections.defaultdict(list)
            for i in lab:
                days[int(rec["rows"][i][R_H] // 24)].append(i)

            def day_slope(dl):
                xs, ys, ws = [], [], []
                for L in days.values():
                    b = est(pt, T, "main", L, 3.0, dl, tau0)
                    if b and b["n"] >= MIN_P:
                        xs.append(b["kb"] * b["Pb"] * math.exp(dl))
                        ys.append(b["ln"])
                        ws.append(b["n"])
                return wslope(xs, ys, ws) if len(xs) >= 3 else None
            c["d_day"] = bisect0(day_slope)
            for q in (1.0, 3.0):
                if not tv[q]:
                    continue

                def e(dl, tau, ids_=ids, q_=q):
                    x = est(pt, T, "main", ids_, q_, dl, tau)
                    return None if not x else math.exp(x["ln"]) / tv[q_] - 1
                L0 = lv.get("all")
                L1 = lv.get(f"near{q:g}")
                c[f"e_L0_{q:g}"] = e(L0[0], L0[1]) if L0 else e(0.0, tau0)
                c[f"e_L1_{q:g}"] = e(L1[0], L1[1]) if L1 else e(0.0, tau0)
                c[f"e_L2_{q:g}"] = e(0.0, tau0)
                tL0 = L0[1] if L0 else tau0
                c[f"dstar_{q:g}"] = bisect0(lambda dl: e(dl, tL0))
                for vn in ("w6", "w12", "w24"):
                    wids = idx_of(pt, T, "main", vn)
                    opt = {"full": L0, "win": lv.get(vn), "near": L1, "none": None}
                    r = {"s": s, "T": T, "q": q, "v": vn, "cls": tcls(s, T), "fb": rec["kind"]["main"]["var"][vn]["fb"]}
                    for on, ov in opt.items():
                        r[on] = e(ov[0], ov[1], wids) if ov else e(0.0, tau0, wids)
                    r["d_win"] = lv[vn][0] if lv.get(vn) else None
                    win.append(r)
            cells.append(c)
    return cells, win


def a4_summary(cells, win):
    out = {"about": ("L0 = R0(M4l, 라벨 줄 전체) · L1 = 목표 % ×2 안 라벨 줄로 M4l · L2 = 끔 (δ 0, prior τ_P). 한섭 1·3% 중앙 추정 (tr, R0 무게·변환). "
                     "δ* = 정답과 같아지는 δ (L0 τ로, 분석용), δ_day = 라벨 줄 하루(24시간) 칸 내재 3% 컷이 z̄ = k·ln10·P̄에 대해 평평해지는 δ (사전 없음), "
                     "D1·D10 = 정답 모양 ln(C1/C3)·ln(C10/C3) − prior 모양, beta_shape = 마지막 12시간 라벨 점의 c 대 ln y 기울기 (δ 0), "
                     "corr_z_lny = 라벨 점의 z와 ln y 상관 (3% 무게).")}
    for c in ("T18+", "T_main", "T_end", "T=H"):
        for q in (1.0, 3.0):
            L = [x for x in cells if c in x["cls"] and x.get(f"e_L0_{q:g}") is not None]
            if not L:
                continue
            wrong = [x for x in L if abs(x[f"e_L0_{q:g}"]) > abs(x[f"e_L2_{q:g}"]) + 1e-4]
            opp = [x for x in L if x.get("d_all") is not None and x.get(f"dstar_{q:g}") is not None and x["d_all"] * x[f"dstar_{q:g}"] < 0]
            out[f"{c}/q{q:g}"] = {"n": len(L), **{k: {"mean_abs_e": round(statistics.mean(abs(x[f"e_{k}_{q:g}"]) for x in L), 5),
                                                      "mean_e": round(statistics.mean(x[f"e_{k}_{q:g}"] for x in L), 5)} for k in ("L0", "L1", "L2")},
                                  "n_L0_worse_than_L2": len(wrong), "n_d_opposite_dstar": len(opp),
                                  "mean_abs_gap_L0_worse": round(statistics.mean(abs(x[f"e_L0_{q:g}"]) - abs(x[f"e_L2_{q:g}"]) for x in wrong), 5) if wrong else None}
    H = [x for x in cells if "T=H" in x["cls"]]
    out["T=H_seasons"] = [{"season": x["s"], "d_all": rnd(x["d_all"]), "d_near3": rnd(x["d_near3"]), "d_day": rnd(x["d_day"]), "dstar3": rnd(x.get("dstar_3")),
                           "e_L0_3": rnd(x.get("e_L0_3")), "e_L1_3": rnd(x.get("e_L1_3")), "e_L2_3": rnd(x.get("e_L2_3")), "D1": rnd(x["D1"]), "D10": rnd(x["D10"]),
                           "beta_shape": rnd(x["beta_shape"]), "corr_z_lny": rnd(x.get("corr_z_lny")), "band_lt2": rnd((x.get("band") or {}).get("<1", 0) + (x.get("band") or {}).get("1-2", 0)),
                           "n_lab": x["n_lab"]} for x in H]
    gap = lambda x: None if x.get("d_all") is None or x.get("dstar_3") is None else x["d_all"] - x["dstar_3"]  # noqa: E731
    gap2 = lambda x: None if x.get("d_all") is None or x.get("d_day") is None else x["d_all"] - x["d_day"]  # noqa: E731
    harm = lambda x: None if x.get("e_L0_3") is None else abs(x["e_L0_3"]) - abs(x["e_L2_3"])  # noqa: E731
    drv = {"D1": lambda x: x["D1"], "D10": lambda x: x["D10"], "beta_shape": lambda x: x["beta_shape"], "corr_z_lny": lambda x: x.get("corr_z_lny"),
           "band_lt2": lambda x: None if not x.get("band") else x["band"]["<1"] + x["band"]["1-2"], "d_day_minus_0": lambda x: x.get("d_day")}
    out["spearman_T=H"] = {f"{tn}~{dn}": spearman([t(x) for x in H], [d(x) for x in H])
                           for tn, t in (("d_all-dstar3", gap), ("d_all-d_day", gap2), ("harm_L0", harm)) for dn, d in drv.items()}
    T18 = [x for x in cells if "T18+" in x["cls"]]
    out["spearman_T18+"] = {f"{tn}~{dn}": spearman([t(x) for x in T18], [d(x) for x in T18])
                            for tn, t in (("d_all-dstar3", gap), ("harm_L0", harm)) for dn, d in drv.items()}
    out["dstar_vs_dday_T=H"] = spearman([x.get("dstar_3") for x in H], [x.get("d_day") for x in H])
    out["dstar_vs_dall_T=H"] = spearman([x.get("dstar_3") for x in H], [x.get("d_all") for x in H])
    out["dstar_vs_dnear3_T=H"] = spearman([x.get("dstar_3") for x in H], [x.get("d_near3") for x in H])
    # B (나): 창 + 수준 갱신 선택
    wb = {}
    for vn in ("w6", "w12", "w24"):
        for c in ("T18+", "T_main", "T_end", "T=H"):
            for q in (1.0, 3.0):
                L = [x for x in win if x["v"] == vn and c in x["cls"] and x["q"] == q and all(x.get(k) is not None for k in ("full", "win", "near", "none"))]
                if L:
                    wb[f"{vn}/{c}/q{q:g}"] = {"n": len(L), "n_fallback": sum(x["fb"] for x in L),
                                              **{k: {"mean_abs_e": round(statistics.mean(abs(x[k]) for x in L), 5), "mean_e": round(statistics.mean(x[k] for x in L), 5)}
                                                 for k in ("full", "win", "near", "none")}}
    out["window_level"] = wb
    return out


def rnd(v, k=4):
    return None if v is None else round(v, k)


def cmd_a4(argv):
    ap = argparse.ArgumentParser(prog="e_48.py a4")
    ap.add_argument("--pts", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    P = load_pts(a.pts, "tr", a.work, A_TARGETS)
    cells, win = a4(P, truth_map())
    res = a4_summary(cells, win)
    old = jl(a.out) if os.path.exists(a.out) else {}
    old["a4"] = res
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("T=H_seasons",)}, ensure_ascii=False)[:8000])
    return 0


# ---------------- A5: 연속성 지표 ----------------
# A1 원 결과(뽑기 중앙)로: T=H의 e · 마지막 날(T_end와 H−18 ~ H−12의 격자 T) 평균 |e| · 하한 어김 · T가 6시간 늘 때 |e|가 줄어든 칸 비율.
# 하한(4.6): 그 T의 tr에서 라벨 한섭 줄 중 시각 A·B이고 이미 q% 밖(지금 % ≥ q)인 점수의 최댓값 (종료 전 화면만 — load_obs가 거름). 예측 < 하한이면 어김.

def lastday_set(H):
    ts = tsets(H)
    return sorted(set(ts["end"]) | {t for t in ts["grid"] if H - 18 - 1e-9 <= t < H - 12 - 1e-9})


def a5_lb(pt, T, q):
    rec = pt.rec(T)
    v = [r[R_S] for r in rec["rows"] if r[R_KIND] == "L" and r[R_GR] in ("A", "B") and r[R_X] >= q - 1e-9]
    return max(v) if v else None


def cmd_a5(argv):
    ap = argparse.ArgumentParser(prog="e_48.py a5")
    ap.add_argument("res", help="A1 원 결과 폴더 RES/{R0,w6,w12,w24}")
    ap.add_argument("--pts", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    P = load_pts(a.pts, "tr", a.work, A_TARGETS)
    out = {"about": ("분석 4.9 A5 — 연속성 지표 (A1 원 결과의 뽑기 중앙, tr). last_day = 시즌마다 T_end(H−12·H−6·H)와 H−18 ≤ T < H−12인 격자 T. "
                     "하한 = 그 T의 tr에서 라벨 한섭·시각 A·B·이미 q% 밖(지금 % ≥ q)인 점수의 최댓값 (종료 전), 예측 < 하한이면 어김. "
                     "conv = 같은 시즌 T → T+6 (T ≥ 18, 둘 다 그 시즌 T 묶음 안) 짝에서 |e|가 줄어든 비율. 개발 검증·서술."), "variants": {}}
    for v in VARIANTS:
        d = os.path.join(a.res, v)
        if not os.path.isdir(d):
            continue
        main, ref, _ = load_variant(d)
        cell = {}
        for part, qs in ((main, (1.0, 3.0)), (ref, (10.0,))):
            for r in part:
                if r["server"] != "KR" or r.get("err") is None or float(r["pct"]) not in qs:
                    continue
                t = round(float(r["T"]), 2)
                if t in tsets(season_H(r["season"]))["all"]:
                    cell[(r["season"], t, float(r["pct"]))] = r
        res = {}
        for q in QS:
            C = {(s, t): r for (s, t, qq), r in cell.items() if qq == q}
            seasons = sorted({s for s, _ in C}, key=lambda x: int(x[1:]))
            TH = [C[(s, round(season_H(s), 2))] for s in seasons if (s, round(season_H(s), 2)) in C]
            LD = [C[(s, t)] for s in seasons for t in lastday_set(season_H(s)) if (s, t) in C]
            viol, viol_H, lbn, tru_bad, mags = 0, [], 0, 0, []
            for (s, t), r in C.items():
                if t < 18:
                    continue
                lb = a5_lb(P[s], t, q)
                if lb is None:
                    continue
                lbn += 1
                if r["pred"] < lb:
                    viol += 1
                    mags.append((lb - r["pred"]) / r["truth"])
                    if abs(t - season_H(s)) < 1e-6:
                        viol_H.append({"season": s, "pred": round(r["pred"], 2), "lb": round(lb, 2), "truth": r["truth"]})
                if r["truth"] < lb - 1e-9:
                    tru_bad += 1
            pairs, pairs_ld, jumps = [], [], []
            for s in seasons:
                H = season_H(s)
                ts = sorted(t for (ss, t) in C if ss == s and t >= 18)
                ld = set(lastday_set(H))
                for t1 in ts:
                    t2 = round(t1 + 6, 2)
                    if (s, t2) in C:
                        e1, e2 = abs(C[(s, t1)]["err"]), abs(C[(s, t2)]["err"])
                        pairs.append(e2 < e1)
                        if t1 in ld and t2 in ld:
                            pairs_ld.append(e2 < e1)
                            jumps.append(abs(C[(s, t2)]["err"] - C[(s, t1)]["err"]))
            res[f"KR{q:g}"] = {
                "T=H": {"n": len(TH), "mean_abs_e": round(statistics.mean(abs(r["err"]) for r in TH), 5), "mean_e": round(statistics.mean(r["err"] for r in TH), 5),
                        "max_abs_e": round(max(abs(r["err"]) for r in TH), 5)},
                "last_day": {"n": len(LD), "mean_abs_e": round(statistics.mean(abs(r["err"]) for r in LD), 5), "mean_e": round(statistics.mean(r["err"] for r in LD), 5)},
                "lb": {"n_cells_with_lb_T18+": lbn, "n_violation_T18+": viol, "violation_mag_med": round(statistics.median(mags), 5) if mags else None,
                       "n_violation_T=H": len(viol_H), "T=H": viol_H, "n_truth_below_lb": tru_bad},
                "conv": {"n_pairs": len(pairs), "share_decrease": round(sum(pairs) / len(pairs), 3) if pairs else None,
                         "n_pairs_last_day": len(pairs_ld), "share_decrease_last_day": round(sum(pairs_ld) / len(pairs_ld), 3) if pairs_ld else None,
                         "mean_abs_jump_last_day": round(statistics.mean(jumps), 5) if jumps else None}}
        out["variants"][v] = res
    old = jl(a.out) if os.path.exists(a.out) else {}
    old["a5"] = out
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
        f.write("\n")
    for v, res in out["variants"].items():
        for k, x in res.items():
            print(v, k, "T=H", x["T=H"]["mean_abs_e"], x["T=H"]["mean_e"], "| last", x["last_day"]["mean_abs_e"], "| lb", x["lb"]["n_violation_T18+"], "/",
                  x["lb"]["n_cells_with_lb_T18+"], "H", x["lb"]["n_violation_T=H"], "truth<lb", x["lb"]["n_truth_below_lb"],
                  "| conv", x["conv"]["share_decrease"], x["conv"]["share_decrease_last_day"], x["conv"]["mean_abs_jump_last_day"])
    return 0


# ---------------- B 재료 요약 (b_inputs) ----------------
# A2~A5 열쇠에서 B가 정할 것마다 숫자를 모으고, M1(구간 넓히기) 진단을 더한다: 변형마다 넓히기 전 |z|(정답 로그 거리 ÷ 그쪽 80% 반폭, freeze_model.zval)의
# 80% 분위(시즌 같은 무게)를 T96·T_end에서, 그리고 「T_end용 M1을 앞 맞힐 시즌의 T_end 줄로 따로」 냈을 때의 T_end 포함 (B 선택지 재료 — 고르기 아님).

def m1_diag(res_dir):
    import freeze_model as FZ
    out = {}
    for v in VARIANTS:
        d = os.path.join(res_dir, v)
        if not os.path.isdir(d):
            continue
        main, _, _ = load_variant(d)
        wid, _ = with_m1(main)
        grp = collections.defaultdict(list)
        for r, rw in zip(main, wid):
            if r["server"] != "KR" or float(r["pct"]) not in (1.0, 3.0) or r.get("err") is None:
                continue
            H = season_H(r["season"])
            t = round(float(r["T"]), 2)
            if t not in tsets(H)["all"]:
                continue
            z = FZ.zval(r)
            if z is None:
                continue
            for k in (["T_main"] if t in T_MAIN else []) + (["T96"] if t == 96.0 else []) + (["T_end"] if t in tsets(H)["end"] else []):
                grp[k].append((r["season"], abs(z), bool(rw.get("in80"))))
        e = {}
        for k, L in grp.items():
            cnt = collections.Counter(s for s, _, _ in L)
            e[k] = {"n": len(L), "q80_absz": round(FZ.wq([(z, 1.0 / cnt[s]) for s, z, _ in L], 0.8), 3),
                    "cov80_raw": round(sum(z <= 1 for _, z, _ in L) / len(L), 3), "cov80_A1_M1": round(sum(i for _, _, i in L) / len(L), 3)}
        # T_end용 M1을 앞 맞힐 시즌의 T_end 줄로 (3시즌 이상일 때, 넓히기만)
        L = grp.get("T_end", [])
        fw = []
        for s in A_TARGETS:
            prev = [(ss, z) for ss, z, _ in L if int(ss[1:]) < int(s[1:])]
            cnt = collections.Counter(ss for ss, _ in prev)
            c = 1.0 if len(cnt) < 3 else max(1.0, FZ.wq([(z, 1.0 / cnt[ss]) for ss, z in prev], 0.8))
            fw += [(z <= c, c) for ss, z, _ in L if ss == s]
        e["T_end_own_M1_forward"] = {"cov80": round(sum(a for a, _ in fw) / len(fw), 3) if fw else None,
                                     "factor_med": round(statistics.median(c for _, c in fw), 3) if fw else None}
        out[v] = e
    return out


def cmd_bsum(argv):
    ap = argparse.ArgumentParser(prog="e_48.py bsum")
    ap.add_argument("res", help="A1 원 결과 폴더")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    D = jl(a.out)
    a2, a3, a4, a5 = (D.get(k) for k in ("a2", "a3", "a4", "a5"))
    fa = a2["fit_all"]
    b = {"about": ("분석 4.9 B 재료 요약 (A2~A5에서 모음, lab4/e_48.py bsum). 서술 — 후보·규칙은 B가 사전 고정한다. "
                   "숫자는 a2~a5 열쇠의 값을 옮긴 것이고 m1_diag만 여기서 새로 낸다 (A1 원 결과, 넓히기 전 |z|)."),
         "sigma_c": {"loso_ll_rec": {k: {su: {f: x[f]["ll_rec"] for f in FORMS + ("none",)} for su, x in d.items()} for k, d in a2["loso"].items()},
                     "fit_all_pdiff": {k: {su: v.get("pdiff") for su, v in d.items()} for k, d in fa.items()},
                     "fit_all_pow": {k: {su: v.get("pow") for su, v in d.items()} for k, d in fa.items()},
                     "forward_ll_rec_L0_ln_q3": {sn: {f: x[f]["ll_rec"] for f in x} for sn, x in a2["targets"]["forward_summary"]["L0"]["ln_q3"].items()},
                     "target_files": a2["targets"]["files"], "tr_vs_full": {q: {k: x[k] for k in ("n_common", "sigma_tr", "sigma_full")} for q, x in a2["tr_vs_full"].items()},
                     "label_records_per_season": {s: x["n"] for s, x in a2["per_season"]["L"].items()},
                     "early_targets_i_records": {t: a2["targets"]["files"][t]["n_train"] for t in ("S21", "S22", "S23", "S24", "S25")}},
         "hours": {"bin_vs_truth_T=H": a2["bin_vs_truth_T=H"], "windows_by_hour": {h: {k: x[k] for k in ("persons_med", "strict_persons_med", "central_e3_all", "central_e3_w6")}
                                                                                    for h, x in a2["windows"]["by_hour"].items()}},
         "judge_time": {vn: {w: {k: x[k] for k in ("n_in_med", "n_fallback", "e3_all_cells", "e1_all_cells")} for w, x in d.items() if w in ("T18+", "T_end", "T=H")}
                        for vn, d in a2["windows"]["judge"].items()},
         "U1": {k: {al: v[al] for al in ("a1", "a0.5", "a0.25", "a0")} for k, v in a3["U1_central"].items() if "/q3" in k or "/q1" in k},
         "mixing": {"by_window": {k: a3["by_window"][k] for k in ("T18+/last6/q3", "T_main/last6/q3", "T_end/last6/q3", "T=H/last6/q3", "T18+/all/q3") if k in a3["by_window"]},
                    "mixture": a3["mixture"]},
         "level": {"R0": {k: a4[k] for k in a4 if k.count("/") == 1}, "window_level": a4["window_level"],
                   "spearman_T18+_gap_D1": a4["spearman_T18+"].get("d_all-dstar3~D1"), "dstar_vs": {k: a4[k] for k in ("dstar_vs_dday_T=H", "dstar_vs_dall_T=H", "dstar_vs_dnear3_T=H")}},
         "continuity": {v: {q: {k: x[k] for k in ("T=H", "last_day", "conv")} | {"lb_viol_T18+": x["lb"]["n_violation_T18+"], "lb_viol_T=H": x["lb"]["n_violation_T=H"]}
                            for q, x in r.items()} for v, r in a5["variants"].items()},
         "m1_diag": m1_diag(a.res)}
    D["b_inputs"] = b
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(D, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(b["m1_diag"], ensure_ascii=False, indent=0))
    return 0


CMDS = {"run": cmd_run, "r0check": cmd_r0check, "a1": cmd_a1, "pts": cmd_pts, "a2": cmd_a2, "a3": cmd_a3, "a4": cmd_a4, "a5": cmd_a5, "bsum": cmd_bsum}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        raise SystemExit(__doc__)
    return CMDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main() or 0)
