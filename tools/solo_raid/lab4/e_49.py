"""분석 4.9 B (2026-10-19) — 옛 사진 무게 후보 고르기. 사전 고정: docs/solo-raid-analysis.md 「4.9 B 사전 고정」 (이 파일과 같은 커밋).

predict.py·xval.py·backtest_web.py는 고치지 않는다 (S42 기본 동작). 후보 patch는 e_48.py run (R1 창 · R2 칸 합치기 · U1 · L1), 여기는 실행 묶음·M1·고르기·보고.

  python3 -I tools/solo_raid/lab4/e_49.py todo RES                                   # 다음에 돌릴 「시드 이름 부분」 줄 (사전 규칙대로 단계마다)
  python3 -I tools/solo_raid/lab4/e_49.py run NAME main|ref --res RES --work XW --sigma-dir SIG [--seed 1] [--jobs 4]
                                                                                     # 후보 하나의 격자(S21~S41 T6~T120) + S28·S38 T_end 실행 → RES/s{시드}/NAME/
  python3 -I tools/solo_raid/lab4/e_49.py sel RES --out diag49.json [--pts PTS --work XW]   # 있는 결과로 고르기·보고 (diag49 「B」)
  python3 -I tools/solo_raid/lab4/e_49.py same A.json B.json [--T 12,24]              # 두 xval 결과의 예측값(가운데·80%)만 견줌 — 정답은 보지 않는다
  python3 -I tools/solo_raid/lab4/e_49.py sigma --target S42 --records SIG/_records.json --out SIG/S42.json   # A2 d 기록으로 맞힐 시즌 σ_c ((i) pdiff)
  python3 -I tools/solo_raid/lab4/e_49.py s42 NAME --res RES --work XW --sigma-dir SIG --T 6,12,... --out F.json   # S42에 얹은 참고 실행 (정답 없음)
  python3 -I tools/solo_raid/lab4/e_49.py s42sum s42_R0.json [s42_이름.json ...] --out diag49.json   # S42 참고 실행을 R0와 견줘 diag49 「B」.「s42」에 (결과 뒤 더함, sel이 「B」를 새로 쓰므로 sel 뒤에)
  # sel --extra 이름,이름: 고르지 않은 후보도 report와 같은 표로 (보고만, 결과 뒤 더함)

후보 이름: R0 · R1w{W}n{N} (창 W시간, 서버별 창 안 사람 N명 미만이면 R0) · R2m{M} (σ_c 배수 M) · 덧붙임 +U{α} (미상 줄 무게) · +L2 (수준 갱신 끔) · +L1 (×2 안 줄로 수준).
고르기 (사전 고정): R 무리 먼저 → 고른 R 위에서 U·L 무리 하나씩 → 둘 다 바뀌면 함께 켬. 10%(ref)는 R0와 고른 R만.
  (1) T_main 한섭 1·3% 80% 포함(M1 뒤) ≥ 0.70 (2) |e| = (T_main 평균 시즌 중앙 |e| + T_end 평균 시즌 중앙 |e|) ÷ 2 최소 (3) 0.5%p 안이면 단순한 순서
  (4) guard — T_end 80% 포함 ≥ 0.70 · end_guard (T_end 3% |e| ≤ R0) · late_guard (T72·96 포함 ≥ 0.70, |e|가 그 단계 지금 값보다 0.3%p 넘게 나쁘지 않음)
      · early_guard (T12 포함이 지금 값 − 0.05 이상). 그 단계 지금 값(R 무리 R0, U·L 무리 고른 R)은 guard를 받지 않는다 — 모두 걸리면 지금 값.
M1 (구간 넓히기)은 후보마다 그 후보 자신의 앞 맞힐 시즌 줄로: T 12~96은 backtest_web.m1_for 그대로, T_end(H−12·H−6·H)는 같은 규칙으로 그 칸 줄에서 따로 (4.9 B에서 더함).
"""
import argparse
import collections
import hashlib
import json
import os
import re
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
import freeze_model as FZ  # noqa: E402
import backtest_web as BW  # noqa: E402
import e_48 as E48  # noqa: E402

AN = os.path.join(ROOT, "data", "solo-raid", "analysis")
TARGETS = "S21-S41"
T_GRID = ",".join(str(t) for t in range(6, 121, 6))
EXTRA = {"S28": "118.5,124.5,130.5", "S38": "101,107,113"}      # 격자에 없는 시즌 H 기준 T_end (검토 보완 1, A1과 같음)
T_MAIN = E48.T_MAIN
T_LATE = (72.0, 96.0)
T_EARLY = (12.0,)
END_REL = (-12.0, -6.0, 0.0)          # T_end 칸 = T − H
MARGIN = 0.005                        # 0.5%p (criteria_v2)
LATE_TOL = 0.003                      # 0.3%p (4.3 late_guard·함께 켬)
EARLY_TOL = 0.05                      # early_guard (4.3 마무리)
COV_MIN = 0.70
LEVEL_L0 = '{"unknown": false}'       # M4l (F)

R_NAMES = ["R0"] + [f"R1w{w}n{n}" for w in (24, 12, 6) for n in (10, 3)] + [f"R2m{m:g}" for m in (0.5, 1, 2)]   # 단순한 순서
U_ADD = ["U0.5", "U0.25"]             # U0 → U1 α 0.5 → 0.25
L_ORDER = ["L2", None, "L1"]          # L2(끔) → L0(지금, 덧붙임 없음) → L1
SEEDS_REPORT = (2, 3)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha1_of(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def snum(s):
    return int(s[1:])


# ---------------- 후보 이름 → 설정 ----------------

def spec(name):
    parts = name.split("+")
    s = {"name": name, "R": parts[0], "win": None, "nmin": 3, "r2": None, "u1": None, "level": "L0"}
    r = parts[0]
    m1 = re.fullmatch(r"R1w(\d+)n(\d+)", r)
    m2 = re.fullmatch(r"R2m([0-9.]+)", r)
    if r == "R0":
        pass
    elif m1:
        s["win"], s["nmin"] = float(m1.group(1)), int(m1.group(2))
    elif m2:
        s["r2"] = float(m2.group(1))
    else:
        raise SystemExit(f"모르는 R: {r}")
    for p in parts[1:]:
        mu = re.fullmatch(r"U([0-9.]+)", p)
        if mu:
            s["u1"] = float(mu.group(1))
        elif p in ("L1", "L2"):
            s["level"] = p
        else:
            raise SystemExit(f"모르는 덧붙임: {p}")
    return s


def cname(R, u=None, lv=None):
    return R + (f"+{u}" if u else "") + (f"+{lv}" if lv else "")


def e48_argv(sp, part, targets, T, out, seed, work, sigdir, jobs):
    """e_48.py run 인자 (사전 고정 설정). main = F의 1·3% (A1 MAIN과 같은 xval 인자), ref = M2a (A1 REF와 같음)."""
    if part == "ref" and (sp["u1"] is not None or sp["level"] != "L0"):
        raise SystemExit("U·L은 main만 (10%는 R만 견준다 — 사전 고정)")
    pre = []
    if sp["win"] is not None:
        pre += ["--win", f"{sp['win']:g}", "--win-min", str(sp["nmin"])]
    if sp["r2"] is not None:
        pre += ["--r2", f"{sp['r2']:g}", "--r2-part", part, "--sigma-dir", sigdir]
    if part == "main":
        pre += ["--patch", "no_interval"]
        if sp["u1"] is not None:
            pre += ["--u1", f"{sp['u1']:g}"]
        if sp["level"] == "L1":
            pre += ["--l1"]
    x = ["--targets", targets, "--T", T, "--jobs", str(jobs), "--draws", "2000", "--seed", str(seed), "--scheme", "recent5"]
    if part == "main":
        if sp["level"] in ("L0", "L1"):
            x += ["--level-update", LEVEL_L0]
        x += ["--predict-opt", "eps_wrong=0"]
    x += ["--work", work, "--out", out]
    return ["run"] + pre + ["--"] + x


def cand_dir(res, seed, name):
    return os.path.join(res, f"s{seed}", name)


def part_files(part):
    p = "MAIN" if part == "main" else "REF"
    return [(f"{p}.json", TARGETS, T_GRID)] + [(f"{p}_{s}.json", s, t) for s, t in EXTRA.items()]


def done(res, seed, name, part):
    return all(os.path.exists(os.path.join(cand_dir(res, seed, name), f)) for f, _, _ in part_files(part))


def cmd_run(argv):
    ap = argparse.ArgumentParser(prog="e_49.py run")
    ap.add_argument("name")
    ap.add_argument("part", choices=["main", "ref"])
    ap.add_argument("--res", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--sigma-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    sp = spec(a.name)
    d = cand_dir(a.res, a.seed, a.name)
    os.makedirs(d, exist_ok=True)
    logd = os.path.join(a.res, "logs")
    os.makedirs(logd, exist_ok=True)
    for f, tg, T in part_files(a.part):
        out = os.path.join(d, f)
        if os.path.exists(out):
            print("있음", out, flush=True)
            continue
        cmd = [sys.executable, "-I", os.path.join(HERE, "e_48.py")] + e48_argv(sp, a.part, tg, T, out + ".part", a.seed, a.work, a.sigma_dir, a.jobs)
        lg = os.path.join(logd, f"s{a.seed}_{a.name}_{f[:-5]}.log")
        with open(lg, "w", encoding="utf-8") as fo:
            p = subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT)
        if p.returncode or not os.path.exists(out + ".part"):
            raise SystemExit(f"실패 {a.name} {f} (rc {p.returncode}) — {lg}")
        os.replace(out + ".part", out)
        print("→", out, flush=True)
    return 0


# ---------------- 읽기 · M1 (T_end까지) ----------------

def load_part(d, part):
    rows, src = [], {}
    for f, tg, _ in part_files(part):
        p = os.path.join(d, f)
        x = jl(p)
        ra = x["run"].get("args") or {}
        lp = x["run"].get("lab_patches")
        if lp and (lp.get("r2") or {}).get("sigma_dir"):      # 세션 임시 경로는 남기지 않는다 ($S = 세션 임시 lab4/)
            lp = dict(lp, r2=dict(lp["r2"], sigma_dir="$S/" + "/".join(os.path.normpath(lp["r2"]["sigma_dir"]).split(os.sep)[-2:])))
        src[f[:-5]] = {"sha1": sha1_of(p), "tool_head": x["run"].get("tool_head"), "tools_dirty": x["run"].get("tools_dirty"),
                       "lab_patches": lp, "failures": len(x.get("failures") or []),
                       "args": {k: ra.get(k) for k in ("scheme", "targets", "T", "draws", "seed", "boot", "predict_opt", "level_update")}}
        for r in x["rows"]:
            if r.get("scheme") != "recent5":
                continue
            if tg in EXTRA and r["season"] != tg:
                continue
            if tg not in EXTRA and r["season"] in EXTRA and f"{float(r['T']):g}" in EXTRA[r["season"]].split(","):
                continue
            rows.append(r)
    return rows, src


def end_rel(r):
    """T_end 칸이면 T − H (−12·−6·0), 아니면 None."""
    H = E48.season_H(r["season"])
    t = round(float(r["T"]), 2)
    if t in E48.tsets(H)["end"]:
        return round(t - H, 2)
    return None


def m1_tables(main):
    """맞힐 시즌마다 {grid: {T: c} (backtest_web.m1_for 그대로), end: {T−H: c}} — 앞 맞힐 시즌의 한섭 1·3% |z| 80% 분위 (시즌 같은 무게, 3시즌 미만이면 1, 넓히기만)."""
    grid = [r for r in main if float(r["T"]) in BW.M1_T]
    by_end = collections.defaultdict(list)
    for r in main:
        if r.get("err") is None or r["server"] != "KR" or float(r["pct"]) not in FZ.M1_GROUP:
            continue
        k = end_rel(r)
        if k is None:
            continue
        z = FZ.zval(r)
        if z is not None:
            by_end[k].append((r["season"], abs(z)))
    out = {}
    for t in sorted({r["season"] for r in main}, key=snum):
        e = {}
        for k in END_REL:
            L = [(s, z) for s, z in by_end.get(k, []) if snum(s) < snum(t)]
            cnt = collections.Counter(s for s, _ in L)
            e[f"{k:g}"] = round(1.0 if len(cnt) < FZ.M1_MIN_PREV else max(1.0, FZ.wq([(z, 1.0 / cnt[s]) for s, z in L], FZ.M1_Q)), 4)
        out[t] = {"grid": BW.m1_for(grid, t), "end": e}
    return out


def m1_factor_ext(tab, T, H):
    """표의 점 = 격자 T(12~96) + 그 시즌의 H−12·H−6·H. 사이는 선형 보간, 밖은 끝값 (predict.m1_factor와 같은 꼴)."""
    pts = sorted([(float(k), v) for k, v in tab["grid"].items()] + [(round(H + float(k), 2), v) for k, v in tab["end"].items()])
    if not tab["grid"]:
        return 1.0      # 앞 시즌 줄이 없음 (S21) — backtest_web과 같이 넓히지 않는다
    if T <= pts[0][0]:
        return pts[0][1]
    if T >= pts[-1][0]:
        return pts[-1][1]
    for (t0, c0), (t1, c1) in zip(pts, pts[1:]):
        if t0 <= T <= t1:
            return c0 if t1 == t0 else c0 + (c1 - c0) * (T - t0) / (t1 - t0)
    return 1.0


def with_m1_ext(main):
    tabs = m1_tables(main)
    out = []
    for r in main:
        r = dict(r)
        c = m1_factor_ext(tabs[r["season"]], float(r["T"]), E48.season_H(r["season"]))
        if r.get("pred") and r.get("i80") and None not in r["i80"] and min(r["i80"]) > 0 and float(r["pct"]) in (1.0, 3.0):
            m = r["pred"]
            lo, hi = r["i80"]
            lo, hi = m * (lo / m) ** c, m * (hi / m) ** c
            r["i80"] = [lo, hi]
            r["w80"] = hi / lo - 1
            if r.get("truth") is not None:
                r["in80"] = lo <= r["truth"] <= hi
        r["m1_c"] = c
        out.append(r)
    return out, tabs


# ---------------- 지표 ----------------

def win(R, Ts, qs):
    return XV.window_v2(R, Ts, qs)


def metrics(rows, qs):
    """한섭 줄 → main(T_main)·end(T_end)·end_q(T_end, 가장 큰 지점 — main은 3%, ref는 10%)·late(T72·96)·t12, score = (main |e| + end |e|) ÷ 2.
    |e|는 criteria_v2의 「평균 시즌 중앙 |e|」 (칸 = (T, 지점) — T_end 칸은 (T − H, 지점))."""
    R = [r for r in rows if r["server"] == "KR" and float(r["pct"]) in qs and r.get("err") is not None]
    E = [dict(r, T=end_rel(r)) for r in R if end_rel(r) is not None]
    m = {"main": win(R, T_MAIN, qs), "end": win(E, END_REL, qs), "end_q": win(E, END_REL, (max(qs),)),
         "late": win(R, T_LATE, qs), "t12": win(R, T_EARLY, qs)}
    m["score"] = round((m["main"]["mean_med_abs_err"] + m["end"]["mean_med_abs_err"]) / 2, 6)
    return m


BRIEF_KEYS = ("n_rows", "n_seasons", "cov80", "mean_med_abs_err", "low_ratio", "med_w80", "cov80_grade_AB", "n_grade_AB",
              "season_cov80_min", "n_season_cov80_lt50", "mean_med_abs_err_b1", "beat_baseline1")


def brief(m):
    out = {"score": m["score"]}
    for k in ("main", "end", "end_q", "late", "t12"):
        x = m.get(k)
        out[k] = None if not x else {kk: x.get(kk) for kk in BRIEF_KEYS if x.get(kk) is not None}
    return out


# ---------------- 고르기 ----------------

def guards(n, M, base, r0):
    why = []
    e, eq, lt, t12 = M[n]["end"], M[n]["end_q"], M[n]["late"], M[n]["t12"]
    if e["cov80"] < COV_MIN:
        why.append(f"T_end 80% 포함 {e['cov80']:.3f} < 0.70")
    if eq["mean_med_abs_err"] > M[r0]["end_q"]["mean_med_abs_err"] + 1e-12:
        why.append(f"end_guard: T_end |e| {eq['mean_med_abs_err']:.4%} > R0 {M[r0]['end_q']['mean_med_abs_err']:.4%}")
    if lt["cov80"] < COV_MIN:
        why.append(f"late_guard: T72·96 80% 포함 {lt['cov80']:.3f} < 0.70")
    d = lt["mean_med_abs_err"] - M[base]["late"]["mean_med_abs_err"]
    if d > LATE_TOL + 1e-12:
        why.append(f"late_guard: T72·96 |e|가 {base}보다 {d:.4%}p 나쁨")
    if t12["cov80"] < M[base]["t12"]["cov80"] - EARLY_TOL - 1e-12:
        why.append(f"early_guard: T12 포함 {t12['cov80']:.3f} < {base} {M[base]['t12']['cov80']:.3f} − 0.05")
    return why


def pick(names, M, rank, base, r0="R0"):
    """사전 고정 규칙 (4.3 pick_rule 꼴): (1) 포함 ≥ 0.70 (2) score 최소 (3) 0.5%p 안 rank 순 (4) guard (지금 값 base는 받지 않음). 모두 걸리면 base."""
    names = [n for n in names if n in M]
    kept = [n for n in names if M[n]["main"]["cov80"] >= COV_MIN]
    step1 = "포함 0.70 이상"
    if not kept:
        bc = max(M[n]["main"]["cov80"] for n in names)
        kept = [n for n in names if M[n]["main"]["cov80"] == bc]
        step1 = "모두 0.70 미달 → 포함률이 가장 높은 것"
    best = min(M[n]["score"] for n in kept)
    near = sorted([n for n in kept if M[n]["score"] <= best + MARGIN + 1e-12], key=lambda n: (rank[n], M[n]["score"]))
    rest = sorted([n for n in kept if n not in near], key=lambda n: M[n]["score"])
    tried, got = [], None
    for n in near + rest:
        why = [] if n == base else guards(n, M, base, r0)
        tried.append({"name": n, "guard": why})
        if not why:
            got = n
            break
    return {"pick": got or base, "fallback_to_base": got is None, "step1": step1, "kept": kept, "best_score": best,
            "within_0_5pp": near, "tried": tried, "base": base}


def table(names, M):
    out = []
    for n in names:
        if n not in M:
            continue
        m = M[n]
        out.append({"name": n, "score": m["score"], "cov_main": m["main"]["cov80"], "mae_main": m["main"]["mean_med_abs_err"],
                    "cov_end": m["end"]["cov80"], "mae_end": m["end"]["mean_med_abs_err"], "mae_end_q": m["end_q"]["mean_med_abs_err"],
                    "cov_late": m["late"]["cov80"], "mae_late": m["late"]["mean_med_abs_err"], "cov_t12": m["t12"]["cov80"],
                    "low": m["main"]["low_ratio"], "season_cov_min": m["main"]["season_cov80_min"]})
    return out


def show(tab, title):
    print(f"\n{title}")
    print(f"  {'이름':22s} score  | main 포함 |e|    | end 포함 |e|    end_q  | late 포함 |e|   | T12 포함")
    for t in tab:
        print(f"  {t['name']:22s} {t['score']:.2%} | {t['cov_main']:.2f} {t['mae_main']:.2%} | {t['cov_end']:.2f} {t['mae_end']:.2%} {t['mae_end_q']:.2%} | "
              f"{t['cov_late']:.2f} {t['mae_late']:.2%} | {t['cov_t12']:.2f}")


class Res:
    """RES/s{시드}/{이름}의 결과를 읽어 M1·지표를 낸다 (한 번씩)."""

    def __init__(self, res):
        self.res = res
        self._m, self._r, self.src = {}, {}, {}

    def has(self, name, part, seed=1):
        return done(self.res, seed, name, part)

    def main(self, name, seed=1):
        k = (name, seed)
        if k not in self._m:
            rows, src = load_part(cand_dir(self.res, seed, name), "main")
            m1rows, tabs = with_m1_ext(rows)
            self._m[k] = {"raw": rows, "rows": m1rows, "m1": tabs, "M": metrics(m1rows, (1.0, 3.0))}
            self.src.setdefault(f"s{seed}/{name}", {})["main"] = src
        return self._m[k]

    def ref(self, name, seed=1):
        k = (name, seed)
        if k not in self._r:
            rows, src = load_part(cand_dir(self.res, seed, name), "ref")
            self._r[k] = {"rows": rows, "M": metrics(rows, (10.0,))}
            self.src.setdefault(f"s{seed}/{name}", {})["ref"] = src
        return self._r[k]


def stage_r(R):
    if not all(R.has(n, "main") for n in R_NAMES):
        return None
    M = {n: R.main(n)["M"] for n in R_NAMES}
    sel = pick(R_NAMES, M, {n: i for i, n in enumerate(R_NAMES)}, "R0")
    return {"names": R_NAMES, "select": sel, "table": table(R_NAMES, M), "metrics": {n: brief(M[n]) for n in R_NAMES}}


def stage_ul(R, base):
    names_u = [base] + [cname(base, u) for u in U_ADD]
    names_l = [cname(base, None, lv) for lv in L_ORDER]
    if not all(R.has(n, "main") for n in names_u + names_l):
        return None
    allm = {n: R.main(n)["M"] for n in set(names_u + names_l)}
    M = dict(allm, R0=R.main("R0")["M"])
    su = pick(names_u, M, {n: i for i, n in enumerate(names_u)}, base)
    sl = pick(names_l, M, {n: i for i, n in enumerate(names_l)}, base)
    changed = [(g, s["pick"]) for g, s in (("U", su), ("L", sl)) if s["pick"] != base]
    out = {"base": base, "U": {"names": names_u, "select": su, "table": table(names_u, M)},
           "L": {"names": names_l, "select": sl, "table": table(names_l, M)}, "changed": changed,
           "metrics": {n: brief(M[n]) for n in sorted(allm)}}
    if not changed:
        out["final"] = base
    elif len(changed) == 1:
        out["final"] = changed[0][1]
    else:
        u = su["pick"].split("+")[-1]
        lv = sl["pick"].split("+")[-1]
        combo = cname(base, u, lv)
        out["combo_name"] = combo
        if not R.has(combo, "main"):
            out["final"] = None
            out["need"] = combo
            return out
        M[combo] = R.main(combo)["M"]
        singles = [k for _, k in changed]
        best1 = min(singles, key=lambda k: M[k]["score"])
        gain = max(singles, key=lambda k: M[base]["score"] - M[k]["score"])
        why = []
        if M[combo]["main"]["cov80"] < COV_MIN:
            why.append(f"포함 {M[combo]['main']['cov80']:.3f} < 0.70")
        if M[combo]["score"] > M[best1]["score"] + LATE_TOL + 1e-12:
            why.append(f"|e|가 하나씩 중 가장 나은 {best1}보다 {M[combo]['score'] - M[best1]['score']:.4%}p 나쁨 (0.3%p 넘음)")
        why += guards(combo, M, base, "R0")
        out["combo"] = {"name": combo, "best_single": best1, "largest_gain_single": gain, "why_not": why, "ok": not why,
                        "table": table([base, combo] + singles, M), "metrics": brief(M[combo])}
        out["final"] = combo if not why else gain
    return out


def stage_10(R, rpick):
    names = ["R0"] + ([rpick] if rpick != "R0" else [])
    if not all(R.has(n, "ref") for n in names):
        return None
    M = {n: R.ref(n)["M"] for n in names}
    sel = pick(names, M, {n: i for i, n in enumerate(names)}, "R0")
    return {"names": names, "select": sel, "table": table(names, M), "metrics": {n: brief(M[n]) for n in names}}


def plan(R):
    """사전 규칙대로 지금까지의 결과에서 → (단계 결과 dict, 다음에 돌릴 [(시드, 이름, 부분)])."""
    st = {}
    need = [(1, n, p) for n in R_NAMES for p in ("main", "ref") if not R.has(n, p)]
    if need:
        return st, need
    st["R"] = stage_r(R)
    rp = st["R"]["select"]["pick"]
    st["P10"] = stage_10(R, rp)
    names_ul = [cname(rp, u) for u in U_ADD] + [cname(rp, None, lv) for lv in L_ORDER if lv]
    need = [(1, n, "main") for n in names_ul if not R.has(n, "main")]
    if need:
        return st, need
    st["UL"] = stage_ul(R, rp)
    if st["UL"].get("need"):
        return st, [(1, st["UL"]["need"], "main")]
    final = st["UL"]["final"]
    p10 = st["P10"]["select"]["pick"]
    st["final"] = {"main_1_3": final, "ref_10": p10}
    need = []
    for s in SEEDS_REPORT:
        for n, p in ((final, "main"), ("R0", "main"), (p10, "ref"), ("R0", "ref")):
            if not R.has(n, p, s) and (s, n, p) not in need:
                need.append((s, n, p))
    return st, need


def cmd_todo(argv):
    ap = argparse.ArgumentParser(prog="e_49.py todo")
    ap.add_argument("res")
    a = ap.parse_args(argv)
    _, need = plan(Res(a.res))
    for s, n, p in need:
        print(s, n, p)
    return 0


# ---------------- 보고 (계획 C 목록 + A5 연속성) ----------------

def continuity(main, ref, P):
    """A5와 같은 지표 (e_48 cmd_a5와 같은 셈): T=H e · 마지막 날 |e| · 하한 어김 · |e|가 준 비율 · 마지막 날 |Δe|. P = e_48 점 표(tr, 하한용)."""
    cell = {}
    for part, qs in ((main, (1.0, 3.0)), (ref, (10.0,))):
        for r in part:
            if r["server"] != "KR" or r.get("err") is None or float(r["pct"]) not in qs:
                continue
            t = round(float(r["T"]), 2)
            if t in E48.tsets(E48.season_H(r["season"]))["all"]:
                cell[(r["season"], t, float(r["pct"]))] = r
    res = {}
    for q in (1.0, 3.0, 10.0):
        C = {(s, t): r for (s, t, qq), r in cell.items() if qq == q}
        if not C:
            continue
        seasons = sorted({s for s, _ in C}, key=snum)
        TH = [C[(s, round(E48.season_H(s), 2))] for s in seasons if (s, round(E48.season_H(s), 2)) in C]
        LD = [C[(s, t)] for s in seasons for t in E48.lastday_set(E48.season_H(s)) if (s, t) in C]
        viol, viol_H, lbn = 0, [], 0
        if P:
            for (s, t), r in C.items():
                if t < 18 or s not in P:
                    continue
                lb = E48.a5_lb(P[s], t, q)
                if lb is None:
                    continue
                lbn += 1
                if r["pred"] < lb:
                    viol += 1
                    if abs(t - E48.season_H(s)) < 1e-6:
                        viol_H.append(s)
        pairs, pairs_ld, jumps = [], [], []
        for s in seasons:
            H = E48.season_H(s)
            ts = sorted(t for (ss, t) in C if ss == s and t >= 18)
            ld = set(E48.lastday_set(H))
            for t1 in ts:
                t2 = round(t1 + 6, 2)
                if (s, t2) in C:
                    e1, e2 = abs(C[(s, t1)]["err"]), abs(C[(s, t2)]["err"])
                    pairs.append(e2 < e1)
                    if t1 in ld and t2 in ld:
                        pairs_ld.append(e2 < e1)
                        jumps.append(abs(C[(s, t2)]["err"] - C[(s, t1)]["err"]))
        res[f"KR{q:g}"] = {"T=H": {"n": len(TH), "mean_abs_e": round(statistics.mean(abs(r["err"]) for r in TH), 5),
                                   "mean_e": round(statistics.mean(r["err"] for r in TH), 5)},
                           "last_day_mean_abs_e": round(statistics.mean(abs(r["err"]) for r in LD), 5) if LD else None,
                           "lb_violation_T18+": viol if P else None, "lb_cells": lbn if P else None, "lb_violation_T=H": viol_H if P else None,
                           "share_decrease": round(sum(pairs) / len(pairs), 3) if pairs else None,
                           "share_decrease_last_day": round(sum(pairs_ld) / len(pairs_ld), 3) if pairs_ld else None,
                           "mean_abs_jump_last_day": round(statistics.mean(jumps), 5) if jumps else None}
    return res


def early_info(rows):
    """이른 T(6~18) 한섭 3%: 쓴 사람 중앙, 창·R2가 R0로 돌아간 칸 수."""
    out = {}
    for T in (6.0, 12.0, 18.0):
        L = [r for r in rows if r["server"] == "KR" and float(r["pct"]) == 3.0 and float(r["T"]) == T]
        if not L:
            continue
        out[f"T{T:g}"] = {"n": len(L), "persons_med": statistics.median(r.get("n_persons") or 0 for r in L),
                          "win_fallback": sum(1 for r in L if (r.get("win") or {}).get("fallback")),
                          "r2_fallback": sum(1 for r in L if (r.get("r2") or {}).get("fallback")),
                          "grades": dict(collections.Counter(r.get("grade") for r in L))}
    return out


def report(R, name, p10, pts=None, work=None):
    Hs = {}
    mm, mr = R.main(name)["rows"], R.ref(p10)["rows"]
    m0, r0 = R.main("R0")["rows"], R.ref("R0")["rows"]
    for r in mm + m0:
        Hs.setdefault(r["season"], E48.season_H(r["season"]))
    P = E48.load_pts(pts, "tr", work, E48.A_TARGETS) if pts and work else None
    out = {"name": name, "ref_10": p10, "by_T": {}, "windows": {}, "season_T18+": {}, "continuity": {}, "early": {}, "summary_v2": {}}
    for lab, a_main, a_ref in (("pick", mm, mr), ("R0", m0, r0)):
        for part, rows, qs in (("main", a_main, (1.0, 3.0)), ("ref", a_ref, (10.0,))):
            for q in qs:
                for sv in E48.SERVERS:
                    k = f"{lab}/{sv}{q:g}"
                    W = E48.windows(rows, Hs, sv, q)
                    out["windows"][k] = {w: E48.stats(L) for w, L in W.items()}
                    if sv == "KR":
                        out["by_T"][k] = E48.by_T(rows, Hs, sv, q)
        out["continuity"][lab] = continuity(a_main, a_ref, P)
        out["early"][lab] = early_info(a_main)
        sv2 = XV.summarize_v2([dict(r, scheme="x") for r in a_main], jl(os.path.join(AN, "criteria_v2.json")))["x"]
        out["summary_v2"][lab] = {"main": {k: sv2["main"].get(k) for k in BRIEF_KEYS + ("season_bias", "season_cov80", "usable")},
                                  "by_volume_main": sv2.get("by_volume_main")}
    for q, (a, b) in ((1.0, (mm, m0)), (3.0, (mm, m0)), (10.0, (mr, r0))):
        out["season_T18+"][f"KR{q:g}"] = E48.season_worst(b, a, Hs, "KR", q)
    out["m1"] = {"pick": R.main(name)["m1"], "R0": R.main("R0")["m1"]}
    return out


def seed_report(R, final, p10):
    out = {}
    for s in SEEDS_REPORT:
        if not all(R.has(n, p, s) for n, p in ((final, "main"), ("R0", "main"), (p10, "ref"), ("R0", "ref"))):
            continue
        out[f"s{s}"] = {"main": {n: brief(R.main(n, s)["M"]) for n in dict.fromkeys((final, "R0"))},
                        "ref": {n: brief(R.ref(n, s)["M"]) for n in dict.fromkeys((p10, "R0"))}}
    return out


def cmd_sel(argv):
    ap = argparse.ArgumentParser(prog="e_49.py sel")
    ap.add_argument("res")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pts", help="e_48 점 표 (하한 어김용, 없으면 그 지표만 뺀다)")
    ap.add_argument("--work")
    ap.add_argument("--extra", help="보고만: 고르지 않은 후보 이름(쉼표)도 report와 같은 표로 (main·ref 모두 그 이름) — 고르기에는 쓰지 않는다 (4.9 B 결과를 본 뒤 더함)")
    a = ap.parse_args(argv)
    R = Res(a.res)
    st, need = plan(R)
    B = {"about": ("분석 4.9 B — 옛 사진 무게 후보 고르기 (lab4/e_49.py sel). 사전 고정 docs 「4.9 B 사전 고정」. 개발 검증 (맞힐 시즌 S21~S41, 뽑기 2000). "
                   "1·3%는 main 줄 + M1(후보마다 앞 맞힐 시즌으로, T_end 칸 따로), 10%는 ref 줄. |e| = criteria_v2 평균 시즌 중앙 |e|."),
         "rule": __doc__.split("고르기 (사전 고정):")[1].strip(), "stages": st, "need": [list(x) for x in need]}
    if "R" in st:
        show(st["R"]["table"], f"R 무리: 고름 {st['R']['select']['pick']} (0.5%p 안 {st['R']['select']['within_0_5pp']})")
        for t in st["R"]["select"]["tried"]:
            if t["guard"]:
                print("   guard:", t)
    if st.get("P10"):
        show(st["P10"]["table"], f"10%: 고름 {st['P10']['select']['pick']}")
    if st.get("UL"):
        for g in ("U", "L"):
            show(st["UL"][g]["table"], f"{g} 무리: 고름 {st['UL'][g]['select']['pick']}")
        if st["UL"].get("combo"):
            print("함께 켬:", st["UL"]["combo"]["name"], "ok" if st["UL"]["combo"]["ok"] else st["UL"]["combo"]["why_not"])
    if st.get("final"):
        f, p10 = st["final"]["main_1_3"], st["final"]["ref_10"]
        print("최종: 1·3%", f, "· 10%", p10)
        B["report"] = report(R, f, p10, a.pts, a.work)
        B["seeds"] = seed_report(R, f, p10)
    if a.extra:
        B["report_extra"] = {"about": "보고만 — 고르기와 무관. 4.9 B 결과를 본 뒤 C 재검토용으로 더함: score 최소 후보와 guard를 지나는 것 중 score 최소 후보 (main·ref 모두 그 R)",
                             **{n: report(R, n, n, a.pts, a.work) for n in a.extra.split(",") if R.has(n, "main") and R.has(n, "ref")}}
        Mm = {n: R.main(n)["M"] for n in R_NAMES if R.has(n, "main")}
        Mr = {n: R.ref(n)["M"] for n in R_NAMES if R.has(n, "ref")}
        B["report_extra"]["R_guards_all"] = {"about": "R 후보 모두에 guard 넷을 R0 기준으로 (고르기에서는 0.5%p 규칙으로 R0가 먼저 골라져 잰 적 없음)",
                                             "main": {n: guards(n, Mm, "R0", "R0") for n in Mm if n != "R0"},
                                             "ref10": {n: guards(n, Mr, "R0", "R0") for n in Mr if n != "R0"}}
        B["report_extra"]["ref10_R_table"] = table(R_NAMES, Mr)
        B["report_extra"]["ref10_R_metrics"] = {n: brief(Mr[n]) for n in Mr}
    B["source"] = R.src
    old = jl(a.out) if os.path.exists(a.out) else {"about": "분석 4.9 B (docs 「4.9 B 사전 고정」·「4.9 B 결과」) — lab4/e_49.py가 씀"}
    old["B"] = B
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("→", a.out, "· 남은 실행", len(need))
    return 0


# ---------------- 기술 확인: 예측값만 견줌 ----------------

def cmd_same(argv):
    ap = argparse.ArgumentParser(prog="e_49.py same")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--T", help="이 T만 (쉼표)")
    ap.add_argument("--seasons", help="이 시즌만 (쉼표)")
    ap.add_argument("--tol", type=float, default=1e-9, help="가운데·80% 끝의 상대 차이 허용")
    a = ap.parse_args(argv)
    Ts = {float(t) for t in a.T.split(",")} if a.T else None
    Ss = set(a.seasons.split(",")) if a.seasons else None
    key = lambda r: (r.get("scheme"), r["season"], float(r["T"]), r["server"], float(r["pct"]))  # noqa: E731
    use = lambda r: (Ts is None or float(r["T"]) in Ts) and (Ss is None or r["season"] in Ss)  # noqa: E731
    A = {key(r): r for r in jl(a.a)["rows"] if use(r)}
    B = {key(r): r for r in jl(a.b)["rows"] if use(r)}
    common = sorted(set(A) & set(B))
    worst, ndiff, ngrade = 0.0, 0, 0
    for k in common:
        x, y = A[k], B[k]
        vs = [(x.get("pred"), y.get("pred"))] + list(zip(x.get("i80") or [], y.get("i80") or []))
        d = max((abs(u / v - 1) for u, v in vs if u and v), default=0.0)
        worst = max(worst, d)
        ndiff += d > a.tol
        ngrade += x.get("grade") != y.get("grade")
    print(json.dumps({"rows_a": len(A), "rows_b": len(B), "common": len(common), "n_diff_gt_tol": ndiff, "max_rel_diff": worst,
                      "n_grade_diff": ngrade, "same": ndiff == 0 and len(A) == len(B) == len(common)}, ensure_ascii=False))
    return 0


# ---------------- σ_c 파일 (새 맞힐 시즌) ----------------

def cmd_sigma(argv):
    """A2 d 기록(_records.json)으로 맞힐 시즌 t의 (i) 맞춤만 낸다 — e_48.a2_target_files와 같은 셈 (학습 = t의 prior 학습 시즌, 앞 5시즌)."""
    ap = argparse.ArgumentParser(prog="e_49.py sigma")
    ap.add_argument("--target", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs = jl(a.records)
    t = a.target
    train = [f"S{k}" for k in range(max(19, snum(t) - 5), snum(t))]
    res = {"target": t, "about": "분석 4.9 B — e_49.py sigma: A2 d 기록에서 (i) 학습 시즌 맞춤만 (e_48 a2와 같은 꼴·같은 셈)", "sets": {"i": train},
           "records_sha1": sha1_of(a.records), "fits": {}}
    for rk, key in (("L_L0", "L0"), ("L_L2", "L2"), ("LU_L0", "L0_LU"), ("LU_L2", "L2_LU")):
        res["fits"][key] = {}
        for unit, q in E48.SETUPS:
            R = E48.in_unit([r for r in recs[rk] if r["q"] == q and r["s"] in train], unit)
            fits = {f: E48.fit_form(f, R) for f in E48.FORMS}
            res["fits"][key][f"{unit}_q{q:g}"] = {"i": {"n_rec": len(R), "n_seasons": len({r["s"] for r in R}), **{f: E48.strip(p) for f, p in fits.items()}}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("→", a.out, {k: v["ln_q3"]["i"]["pdiff"] for k, v in res["fits"].items()})
    return 0


# ---------------- S42 참고 실행 (정답 없음 — 계획 C 「S42에 얹었을 때의 차이」) ----------------

def cmd_s42(argv):
    """운영 모델 파일(model_S42.json)의 prior·빌릴 시즌으로 S42 T 절단 사본에 후보 하나를 얹는다. xval run_job과 같은 순서 (수준은 전체 줄로 먼저 풀고 predict).
    R(창·R2)은 main·ref 둘 다, U1·L1은 main만. M1은 그 후보의 xval main 줄로 맞힐 시즌 = S42 (앞 = S21~S41 전부)일 때의 표 (T_end 칸 포함).
    결과: T마다 세 서버 1·3%(main, M1 뒤)·10%(ref) 가운데·80%. 고른 1·3%와 10%의 R이 다르면 두 이름을 따로 돌려 쓴다. 정답이 없으니 R0와의 차이만 본다."""
    ap = argparse.ArgumentParser(prog="e_49.py s42")
    ap.add_argument("name")
    ap.add_argument("--res", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--sigma-dir", required=True)
    ap.add_argument("--T", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    import backtest as BT
    import fit_prior_multi as FM
    import m_update as MU
    sp = spec(a.name)
    model = PR.load_model(os.path.join(AN, "model_S42.json"))
    main_rows = Res(a.res).main(a.name)["raw"]
    tab = {"grid": BW.m1_for(main_rows, "S42"), "end": {}}
    by_end = collections.defaultdict(list)
    for r in main_rows:
        k = end_rel(r) if r["server"] == "KR" and float(r["pct"]) in FZ.M1_GROUP and r.get("err") is not None else None
        z = FZ.zval(r) if k is not None else None
        if z is not None:
            by_end[k].append((r["season"], abs(z)))
    for k in END_REL:
        L = by_end.get(k, [])
        cnt = collections.Counter(s for s, _ in L)
        tab["end"][f"{k:g}"] = round(1.0 if len(cnt) < FZ.M1_MIN_PREV else max(1.0, FZ.wq([(z, 1.0 / cnt[s]) for s, z in L], FZ.M1_Q)), 4)
    if sp["win"] is not None:
        E48.apply_window(sp["win"], sp["nmin"])
    if sp["r2"] is not None:
        E48.apply_r2(sp["r2"], "main", a.sigma_dir)
    plain = (PR.Predictor.weights, PR.Predictor.screen)
    if sp["u1"] is not None:
        E48.apply_u1(sp["u1"])
    u1_m = (PR.Predictor.weights, PR.Predictor.screen)
    pred_main = E48.l1_wrap(PR.predict) if sp["level"] == "L1" else PR.predict
    state = FM.state_of("S42")
    out = {"name": a.name, "model": {"file": os.path.relpath(model["_path"], ROOT), "sha1": model["_sha1"]}, "m1": tab, "T": {}}
    for T in [float(x) for x in a.T.split(",")]:
        st = BT.trunc_state(state, T, os.path.join(a.work, "trunc", "S42"), XV.extra_of(state))
        _, c0, H = PR.season_info(st)
        Tc = min(T, H)
        unk = {"mode": PR.UNKNOWN_DEFAULT, "servers": ["KR"], "borrow_state": model["_borrow"], "borrow_kind": "label",
               "mix_file": None, "no_gate": False, "fallback": "site"}
        base = {"grades": "all", "c_delay": 0.0, "p_update": False, "robust": "mean", "notes": [], "contam": (), "unknown": unk}
        rows, cnt = PR.load_obs(st, Tc, H, c0, "all", 0.0, None, unk)
        res = {}
        for part in ("main", "ref"):
            prior = model[part]["_prior"]
            o = dict(base, eps_wrong=model[part]["eps_wrong"])
            mu = None
            if part == "main" and sp["level"] in ("L0", "L1"):
                mu = MU.m_posterior(prior, rows, H, prior["season_diff"]["tau_p"]["value"], unknown=False)
                if mu:
                    o["_lvl"] = (mu["mu"], mu["sd"])
            E48.R2["part"] = part
            PR.Predictor.weights, PR.Predictor.screen = u1_m if part == "main" else plain
            x = (pred_main if part == "main" else PR.predict)(st, prior, Tc, o, a.draws, a.seed, comps_split=False, rows=rows)
            x["unknown"] = cnt.get("unknown")
            x["_mu"] = mu
            res[part] = x
        PR.Predictor.weights, PR.Predictor.screen = u1_m
        c = m1_factor_ext(tab, Tc, H)
        pts = {}
        for s in E48.SERVERS:
            for part, qs in (("main", ("1.0", "3.0")), ("ref", ("10.0",))):
                sv = res[part]["servers"].get(s) or {}
                for q in qs:
                    d = (sv.get("points") or {}).get(q)
                    if not d or not d.get("median"):
                        continue
                    m, (lo, hi) = d["median"], d["i80"]
                    if part == "main" and lo and hi and lo > 0:
                        lo, hi = m * (lo / m) ** c, m * (hi / m) ** c
                    e = {"median": m, "i80": [lo, hi], "i80_raw": d["i80"], "n_persons": sv.get("n_persons")}
                    if part == "main":
                        e["grade"] = PR.warning(Tc, s, sv, res[part]["unknown"], model[part]["_prior"])["level"]
                    pts[f"{s}{float(q):g}"] = e
        mu = res["main"]["_mu"]
        out["T"][f"{T:g}"] = {"Tc": Tc, "H": H, "m1_c": round(c, 4), "points": pts, "level": None if not mu else [mu["mu"], mu["sd"], mu.get("persons")],
                              "l1": res["main"].get("l1"), "r2": {k: (res[k].get("r2") or {}).get("src") for k in res}}
        k3 = pts.get("KR3", {})
        if k3:
            print(f"S42 T{T:g} {a.name}: 한섭 3% {k3['median']:.2f} [{k3['i80'][0]:.2f}~{k3['i80'][1]:.2f}] ×{c:.3f}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("→", a.out)
    return 0


def cmd_s42sum(argv):
    """S42 참고 실행 파일(s42 결과)을 R0와 견줘 diag49 「B」.「s42」에 적는다 — T마다 세 서버 1·3·10% 가운데의 R0 대비 상대 차이와 80% 폭(M1 뒤). 정답 없음 (4.9 B 결과를 본 뒤 더함)."""
    ap = argparse.ArgumentParser(prog="e_49.py s42sum")
    ap.add_argument("files", nargs="+", help="s42_{이름}.json — 첫 파일이 R0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    X = [jl(p) for p in a.files]
    base = X[0]
    if base["name"] != "R0":
        raise SystemExit("첫 파일은 R0")
    res = {"about": "S42 참고 실행 (정답 없음 — 계획 C 「S42에 얹었을 때의 차이」). model_S42 prior·빌릴 시즌, M1은 그 후보의 xval main 줄로 S42를 맞힐 때의 표 (커밋된 S42 예측의 M1 표와 다르다). "
                    "d = 가운데의 R0 대비 상대 차이, w = 80% 폭 (hi/lo − 1). 1·3%는 main(그 이름의 U·L 포함), 10%는 ref(그 이름의 R).",
           "source": {os.path.basename(p): sha1_of(p) for p in a.files}, "model": base.get("model"), "m1": {x["name"]: x["m1"] for x in X}, "T": {}}
    for T, b in base["T"].items():
        row = {}
        for k, e0 in b["points"].items():
            ent = {"R0": {"median": round(e0["median"], 3), "w": round(e0["i80"][1] / e0["i80"][0] - 1, 5)}}
            for x in X[1:]:
                e = x["T"].get(T, {}).get("points", {}).get(k)
                if e:
                    ent[x["name"]] = {"median": round(e["median"], 3), "d": round(e["median"] / e0["median"] - 1, 5), "w": round(e["i80"][1] / e["i80"][0] - 1, 5)}
            row[k] = ent
        res["T"][T] = {"m1_c": {x["name"]: x["T"].get(T, {}).get("m1_c") for x in X}, "points": row}
    old = jl(a.out)
    old["B"]["s42"] = res
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=1)
        f.write("\n")
    for T, r in res["T"].items():
        print(f"T{T}", " · ".join(f"{k} " + " ".join(f"{n} {v['d']:+.2%}/w{v['w']:.1%}" for n, v in ent.items() if n != "R0") + f" (R0 w{ent['R0']['w']:.1%})"
                                  for k, ent in r["points"].items() if k.startswith("KR")))
    print("→", a.out)
    return 0


CMDS = {"todo": cmd_todo, "run": cmd_run, "sel": cmd_sel, "same": cmd_same, "sigma": cmd_sigma, "s42": cmd_s42, "s42sum": cmd_s42sum}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        raise SystemExit(__doc__)
    return CMDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main() or 0)
