"""WP8b 숫자단 값어치 가늠 (docs/solo-raid-method.md 「WP8b 값어치 가늠」 카드, 2026-11-01 계산 전 고정, 1차).

그때 알 수 있던 한섭 순위표 2~7위 점수(숫자단)가 % 관측으로 만든 예측(상대 P = F_D1 또는 M3 본안)에
더해 한섭 최종 1·3% 컷을 더 맞히게 하나를 가늠한다. 인구는 쓰지 않는다. 공용 코드는 읽기만 한다
(obs_table.read_cell · common.t_grid/info · score.evaluate_split/window/tkey · anchor.truth).

  python3 -I tools/solo_raid/lab7/nr_value.py run --obs W/obs7 --out data/solo-raid/analysis/r7/nr [--jobs 4]

1. 특징 Z(s, T) — 관측 표 (시즌, T) 칸에서 kind ∈ {A, B}, exclude 비어 있음, screen == "rank_list", rank 2..7, h <= T 인 줄.
   k마다 「T까지 본 가장 높은 score」, Z = 그 ln의 평균 (k가 3개 이상일 때만). 두 판:
     lab = server == "KR"  ·  dc = server == "KR" 또는 (server 없음 and site == "dc").
   보고만: screen이 rank_list가 아닌데 rank 2..7인 줄을 더하면 Z 있는 칸이 몇 개 느는지 (lab 판 기준).
2. R — ln R(s,T,q) = Z(s,T) + mean_p [ln 정답(p,KR,q) − Z(p,T)], p = s보다 앞 시즌 가운데 Z(p,T)와 정답이 있는 것의 최근 6개 (3개 미만이면 R 없음).
   T 맞추기 (조율 세션 결정 2026-11-01): 격자 6..120 값은 숫자로, 막판 T (H−12·H−6·H)는 H에서의 거리로 맞춘다.
   p의 그 T가 p 격자에 없으면 (예: 시즌 끝 근처 격자 T가 p의 H를 넘음) p는 그 칸에서 빠진다.
3. 섞기 — 상대 예측 P의 칸 (s, T, q, KR)마다. iv(분산 역수): mse_X = 앞 시즌들 (P·R 둘 다 있고 정답 있는 것, 최근 6, 3개 이상)의
   같은 T·q (ln X/정답)² 평균, w_R = (1/mse_R)/(1/mse_R + 1/mse_P), 앞 시즌이 모자라면 ½. half: w_R = ½.
   ln pred = w_R ln R + (1−w_R) ln P. R 없는 칸은 P 그대로 (센다). i80·i95는 P의 것 × (새 pred / P pred). own·그 밖은 P 그대로.
4. 예측 파일 pred_{F|M3}+R_{iv|half}_{lab|dc}_D1.json (score.py 공통 꼴) → score.py --weights third → value.json·value.md.
5. 문턱 (카드): lab·iv 판, 상대 F에서 1·3% 초반 창 평균 |e|가 0.3%p 이상 작은가 (22시즌·16시즌 둘 다), 막판 창이 0.3%p 넘게 나빠지지 않는가, 전체 score 차.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import obs_table as O  # noqa: E402
import score as S  # noqa: E402

A = S.A
ROOT = C.ROOT
SEASONS_ALL = [f"S{n}" for n in range(19, 43)]    # Z를 만드는 시즌 (S19·S20은 앞 시즌 비율용)
PCTS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
RANKS = range(2, 8)
NPRIOR = 6
MINPRIOR = 3
TH = 0.003         # 문턱 0.3%p
VERS = ("lab", "dc")
WEIGHTS = ("iv", "half")
REL = "data/solo-raid/analysis/r7"


# ───────────────────────── 1. 특징 Z ─────────────────────────
def server_ok(r, ver):
    if r["server"] == "KR":
        return True
    return ver == "dc" and r["server"] is None and r["site"] == "dc"


def z_from_rows(rows, T, ver, need_screen=True):
    """표 줄 → (Z 또는 None, {k: 최대 점수}, 쓴 줄 수, 쓴 줄 가운데 게시 시각이 T보다 늦은 수)."""
    best = {}
    n = late = 0
    for r in rows:
        if r["kind"] not in ("A", "B") or r["exclude"]:
            continue
        if need_screen and r["screen"] != "rank_list":
            continue
        k = r["rank"]
        if k is None or k not in RANKS or r["h"] is None or r["h"] > T + 1e-9:
            continue
        if r["score"] is None or r["score"] <= 0 or not server_ok(r, ver):
            continue
        n += 1
        if r["hp"] is not None and r["hp"] > T + 1e-9:
            late += 1
        if k not in best or r["score"] > best[k]:
            best[k] = r["score"]
    z = sum(math.log(v) for v in best.values()) / len(best) if len(best) >= 3 else None
    return z, best, n, late


def _zjob(path):
    h, rows = O.read_cell(path)
    T = h["T"]
    out = {"season": h["season"], "T": T}
    for ver in VERS:
        z, best, n, late = z_from_rows(rows, T, ver)
        out[ver] = z
        out[ver + "_k"] = len(best)
        out[ver + "_n"] = n
        out[ver + "_late"] = late
    out["lab_any"] = z_from_rows(rows, T, "lab", need_screen=False)[0]
    out["dc_any"] = z_from_rows(rows, T, "dc", need_screen=False)[0]
    return out


def load_z(obs_dir, jobs):
    files = []
    for s in SEASONS_ALL:
        d = os.path.join(obs_dir, s)
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".jsonl.gz")]
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            res = pool.map(_zjob, files, chunksize=4)
    else:
        res = [_zjob(p) for p in files]
    Z = {}
    for r in res:
        Z[(r["season"], r["T"])] = r
    return Z, obs_files_sha(files)


def obs_files_sha(files):
    h = hashlib.sha1()
    for p in sorted(files):
        with open(p, "rb") as f:
            h.update(hashlib.sha1(f.read()).digest())
    return h.hexdigest()[:12], len(files)


# ───────────────────────── 2. T 맞추기 · R ─────────────────────────
def snum(s):
    return int(s[1:])


def t_match(s, T, p):
    """s의 T에 대응하는 p의 T (없으면 None). 막판 T (H−12·H−6·H)는 H에서의 거리, 나머지는 숫자."""
    Hs = C.info(s)[1]
    Hp = C.info(p)[1]
    grid = C.t_grid(p)
    if T in (Hs - 12, Hs - 6, Hs):
        Tp = Hp - (Hs - T)
    else:
        Tp = float(T)
    return Tp if Tp in grid else None


class Ratio:
    """R(s, T, q)와 그 앞 시즌 목록 — 메모."""

    def __init__(self, Z, ver, truth):
        self.Z, self.ver, self.truth = Z, ver, truth
        self.memo = {}

    def zval(self, s, T):
        r = self.Z.get((s, T))
        return None if r is None else r[self.ver]

    def lnR(self, s, T, q):
        """(ln R, 쓴 앞 시즌 수) 또는 (None, n)."""
        key = (s, T, q)
        if key in self.memo:
            return self.memo[key]
        out = (None, 0)
        z = self.zval(s, T)
        if z is not None:
            diffs = []
            for n in range(snum(s) - 1, snum("S19") - 1, -1):
                p = f"S{n}"
                tv = self.truth.get((p, "KR", q))
                if tv is None:
                    continue
                Tp = t_match(s, T, p)
                if Tp is None:
                    continue
                zp = self.zval(p, Tp)
                if zp is None:
                    continue
                diffs.append(math.log(tv) - zp)
                if len(diffs) == NPRIOR:
                    break
            out = (z + sum(diffs) / len(diffs), len(diffs)) if len(diffs) >= MINPRIOR else (None, len(diffs))
        self.memo[key] = out
        return out


# ───────────────────────── 3. 섞기 ─────────────────────────
def mix(P, ratio, truth, wmode):
    """상대 예측 P (load_pred 결과) → 섞은 칸 목록과 통계."""
    pcell = {(c["season"], c["T"], c["pct"]): c for c in P["cells"] if c.get("server", "KR") == "KR"}
    pseasons = sorted({c["season"] for c in pcell.values()}, key=snum)
    out, stat = [], collections.Counter()
    wlog = collections.defaultdict(list)    # tkey → [w_R] (R 있는 칸, 분산 역수 판의 계산된 것만 따로 센다)
    for c in P["cells"]:
        if c.get("server", "KR") != "KR":
            continue
        s, T, q = c["season"], c["T"], c["pct"]
        lnR, nprior = ratio.lnR(s, T, q)
        new = dict(c)
        stat["cells"] += 1
        if lnR is None or not c.get("pred"):
            new["has_R"] = False
            stat["no_R"] += 1
            out.append(new)
            continue
        w, informed = 0.5, False
        if wmode == "iv":
            eR, eP = [], []
            for p in reversed(pseasons):
                if snum(p) >= snum(s):
                    continue
                Tp = t_match(s, T, p)
                tv = truth.get((p, "KR", q))
                pc = pcell.get((p, Tp, q)) if Tp is not None else None
                if tv is None or pc is None or not pc.get("pred"):
                    continue
                lr, _ = ratio.lnR(p, Tp, q)
                if lr is None:
                    continue
                eR.append(lr - math.log(tv))
                eP.append(math.log(pc["pred"] / tv))
                if len(eR) == NPRIOR:
                    break
            if len(eR) >= MINPRIOR:
                mR = max(sum(x * x for x in eR) / len(eR), 1e-12)
                mP = max(sum(x * x for x in eP) / len(eP), 1e-12)
                w = (1 / mR) / (1 / mR + 1 / mP)
                informed = True
        lnp = w * lnR + (1 - w) * math.log(c["pred"])
        pred = math.exp(lnp)
        f = pred / c["pred"]
        new["pred"] = round(pred, 5)
        for k in ("i80", "i95"):
            if c.get(k) and None not in c[k]:
                new[k] = [round(c[k][0] * f, 5), round(c[k][1] * f, 5)]
        new["has_R"] = True
        new["w_R"] = round(w, 6)
        new["R"] = round(math.exp(lnR), 5)
        stat["with_R"] += 1
        if informed:
            stat["informed"] += 1
        wlog[S.tkey(s, T)].append(w if informed else None)
        out.append(new)
    return out, stat, wlog


# ───────────────────────── 채점 보조 ─────────────────────────
def abs_err(cells, truth, qs, seasons=None):
    """[(시즌, T, %, 창, e)] — KR, 정답 있는 칸, 두 창에 드는 칸만."""
    out = []
    for c in cells:
        if c.get("server", "KR") != "KR" or c["pct"] not in qs or not c.get("pred"):
            continue
        if seasons is not None and c["season"] not in seasons:
            continue
        tv = truth.get((c["season"], "KR", c["pct"]))
        if tv is None:
            continue
        w = S.window(c["season"], c["T"])
        out.append((c["season"], c["T"], c["pct"], w, c["pred"] / tv - 1))
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def sha12(path):
    return S.sha12(path)


def r4(x, d=4):
    return None if x is None else round(x, d)


def pc(x, d=2):
    return "-" if x is None else f"{x * 100:.{d}f}"


def sg(x, d=2):
    """부호 붙인 %p (개선이면 +, 나빠짐이면 -)."""
    return "-" if x is None else f"{x * 100:+.{d}f}"


# ───────────────────────── 실행 ─────────────────────────
def run(a):
    t0 = time.time()
    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)
    truth = A.truth()
    Z, (obs_sha, nfiles) = load_z(a.obs, a.jobs)
    print(f"관측 표 {nfiles}개에서 Z 계산 {time.time() - t0:.1f}초", flush=True)
    head = git_head()
    rel_out = os.path.relpath(out_dir, ROOT) if os.path.abspath(out_dir).startswith(ROOT) else out_dir
    cmd = f"python3 -I tools/solo_raid/lab7/nr_value.py run --obs <obs7> --out {rel_out} --jobs {a.jobs}"

    # Z 있는 칸 비율 (S21~S42 목표 시즌, 격자 칸) · 보고용
    targets = [f"S{n}" for n in range(21, 43)]
    cov = {}
    for ver in VERS + ("lab_any", "dc_any"):
        byT = collections.defaultdict(lambda: [0, 0])
        allc = [0, 0]
        for s in targets:
            for T in C.t_grid(s):
                r = Z.get((s, T))
                if r is None:
                    continue
                kk = S.tkey(s, T)
                byT[kk][1] += 1
                allc[1] += 1
                if r[ver] is not None:
                    byT[kk][0] += 1
                    allc[0] += 1
        cov[ver] = {"all": allc, "byT": {k: v for k, v in byT.items()}}
    missing_tab = [(s, T) for s in targets for T in C.t_grid(s) if (s, T) not in Z]
    late_rows = {ver: sum(r[ver + "_late"] for r in Z.values()) for ver in VERS}
    print("Z 칸 비율", {v: f"{cov[v]['all'][0]}/{cov[v]['all'][1]}" for v in cov}, "표 없는 칸", len(missing_tab), flush=True)

    # 원자료 확인용 한 칸
    sample = None
    for s, T in (("S35", 48.0),):
        p = os.path.join(a.obs, s, f"T{T:g}.jsonl.gz")
        if os.path.exists(p):
            h, rows = O.read_cell(p)
            z, best, n, late = z_from_rows(rows, T, "lab")
            sample = {"season": s, "T": T, "ver": "lab", "lines": n, "best_by_k": {str(k): v for k, v in sorted(best.items())}, "Z": z,
                      "lines_total_in_table": len(rows)}
    ratios = {v: Ratio(Z, v, truth) for v in VERS}

    bases = {"F": os.path.join(ROOT, REL, "F_D1.json"), "M3": os.path.join(ROOT, REL, "pred_M3_tune_D1.json")}
    lbp = os.path.join(ROOT, REL, "lb_D1.json")
    pred_files = {}       # 상대 → [(이름, 경로)]
    mix_cells = {}        # 이름 → 칸
    mix_stat, wlogs = {}, {}
    for base, bpath in bases.items():
        P = S.load_pred(bpath)
        pred_files[base] = [(P["method"], bpath)]
        for wmode in WEIGHTS:
            for ver in VERS:
                name = f"{base}+R_{wmode}_{ver}"
                cells, stat, wlog = mix(P, ratios[ver], truth, wmode)
                fn = os.path.join(out_dir, f"pred_{name}_D1.json")
                doc = {"about": f"WP8b 값어치 가늠 1차 — 상대 {base}에 한섭 순위표 2~7위 비율 예측 R을 섞음 (무게 {wmode}, 특징 판 {ver}). 개발 검증 자료라 낙관 위험",
                       "method": name, "data_ver": P["data_ver"],
                       "made": {"tool_head": head, "cmd": cmd,
                                "inputs": {"base": sha12(bpath), "obs7(표 파일 모음)": obs_sha, "n_obs_files": nfiles,
                                           "answer_key": sha12(os.path.join(ROOT, "data", "solo-raid", "analysis", "answer_key.json"))}},
                       "cells": cells}
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
                pred_files[base].append((name, fn))
                mix_cells[name] = cells
                mix_stat[name] = dict(stat)
                wlogs[name] = wlog
                print(f"{name}: 칸 {stat['cells']} R 있음 {stat['with_R']} (분산 역수 계산 {stat['informed']}) R 없음 {stat['no_R']}", flush=True)
    P_cells = {b: S.load_pred(bpath)["cells"] for b, bpath in bases.items()}
    P_by = {(c["season"], c["T"], c["pct"]): c for c in P_cells["F"] if c.get("server", "KR") == "KR"}

    # score.py (공용 채점) — 상대 원본 먼저, 섞은 것 뒤
    score_out = {}
    for base in bases:
        sp = os.path.join(out_dir, f"score_{base}.json")
        cmdl = [sys.executable, "-I", os.path.join(HERE, "score.py")]
        for _, fn in pred_files[base]:
            cmdl += ["--pred", fn]
        cmdl += ["--ref", bases["F"], "--lb", lbp, "--weights", "third", "--out", sp]
        r = subprocess.run(cmdl, capture_output=True, text=True, cwd=ROOT)
        if r.returncode:
            sys.exit(r.stderr)
        with open(os.path.join(out_dir, f"score_{base}.txt"), "w", encoding="utf-8") as f:
            f.write(r.stdout)
        with open(sp, encoding="utf-8") as f:
            score_out[base] = json.load(f)
        print(f"score.py {base} 묶음 → {sp}", flush=True)

    # score.py 출력에서 숫자 뽑기
    def score_names(base):
        names, seen = [], []
        for nm, _ in pred_files[base]:
            n = nm
            if n in seen or n == "F":     # 상대 F 원본은 ref 이름과 같아 score.py가 @파일이름을 붙인다
                n = f"{n}@{os.path.basename(pred_files[base][0][1])}"
            seen.append(n)
            names.append(n)
        return names
    table = {}
    for base in bases:
        sn = score_names(base)
        for gname in ("1·3%", "10%"):
            g = score_out[base]["groups"][gname]
            for (nm, _), key in zip(pred_files[base], sn):
                sp = g["split"][key]
                table.setdefault(base, {}).setdefault(gname, {})[nm] = {
                    sub: {k: sp[sub][k] for k in ("score", "main", "end", "cov_main", "cov_end", "crps", "n")} for sub in ("all", "tune", "rest")}
    # 자체 셈 일치 확인 (1·3% 초반 창)
    chk = []
    for base in bases:
        for nm, _ in pred_files[base]:
            cells = P_cells[base] if nm == pred_files[base][0][0] else mix_cells[nm]
            e = [abs(x[4]) for x in abs_err(cells, truth, (1.0, 3.0)) if x[3] == "main"]
            sc = table[base]["1·3%"][nm]["all"]["main"]
            chk.append(abs(mean(e) - sc) < 1e-9)
    assert all(chk), "자체 셈과 score.py 초반 창 평균이 다르다"

    # ── 보고용 셈
    TUNE = set(S.TUNE)
    seas_all = sorted({c["season"] for c in P_cells["F"]}, key=snum)
    seas_rest = [s for s in seas_all if s not in TUNE]

    def curve(cells, qs, seasons):
        g = collections.defaultdict(list)
        for s, T, q, w, e in abs_err(cells, truth, qs, set(seasons)):
            g[S.tkey(s, T)].append(abs(e))
        return {k: [mean(v), len(v)] for k, v in g.items()}

    def tsort(keys):
        def f(k):
            return (1, {"H-12": 0, "H-6": 1, "H": 2}[k]) if k.startswith("H") else (0, float(k))
        return sorted(keys, key=f)

    def per_season(cells, qs):
        g = collections.defaultdict(list)
        for s, T, q, w, e in abs_err(cells, truth, qs):
            if w == "main":
                g[s].append(abs(e))
        return {s: mean(v) for s, v in g.items()}

    rep = {}
    for base in bases:
        p0 = pred_files[base][0][0]
        seas = sorted({c["season"] for c in P_cells[base]}, key=snum)
        rep[base] = {"seasons": seas, "curve": {}, "per_season_main": {}}
        for gname, qs in (("1·3%", (1.0, 3.0)), ("10%", (10.0,))):
            rep[base]["curve"][gname] = {"base": curve(P_cells[base], qs, seas)}
            for nm, _ in pred_files[base][1:]:
                rep[base]["curve"][gname][nm] = curve(mix_cells[nm], qs, seas)
        rep[base]["per_season_main"]["1·3%"] = {"base": per_season(P_cells[base], (1.0, 3.0))}
        for nm, _ in pred_files[base][1:]:
            rep[base]["per_season_main"]["1·3%"][nm] = per_season(mix_cells[nm], (1.0, 3.0))
    # R 단독 오차 (1·3%, R이 있는 칸의 |R/정답 − 1|, 22시즌) — R이 얼마나 쓸 만한지 보는 참고
    ralone = {}
    for nm in ("F+R_iv_lab", "F+R_iv_dc"):
        g = collections.defaultdict(list)
        gp = collections.defaultdict(list)
        for c in mix_cells[nm]:
            tv = truth.get((c["season"], "KR", c["pct"]))
            if c["pct"] in (1.0, 3.0) and c.get("has_R") and tv:
                g[S.tkey(c["season"], c["T"])].append(abs(c["R"] / tv - 1))
                gp[S.tkey(c["season"], c["T"])].append(abs(next(x["pred"] for x in [P_by[(c["season"], c["T"], c["pct"])]]) / tv - 1))
        ralone[nm] = {k: [mean(g[k]), mean(gp[k]), len(g[k])] for k in tsort(g)}
    rep["R_alone"] = ralone
    # F를 M3와 같은 6시즌만으로 (견줄 때 쓴다)
    f6 = {}
    for gname, qs in (("1·3%", (1.0, 3.0)),):
        e = abs_err(P_cells["F"], truth, qs, TUNE)
        f6[gname] = {"main": mean(abs(x[4]) for x in e if x[3] == "main"), "end": mean(abs(x[4]) for x in e if x[3] == "end")}

    # w_R 분포 (T별 중앙, 분산 역수 판 · 계산된 것만) + ½로 떨어진 비율
    wsum = {}
    for nm, wl in wlogs.items():
        if "_iv_" not in nm:
            continue
        wsum[nm] = {}
        for k in tsort(wl):
            v = wl[k]
            inf = [x for x in v if x is not None]
            wsum[nm][k] = {"n_R": len(v), "n_informed": len(inf), "median": r4(st.median(inf)) if inf else None,
                           "min": r4(min(inf)) if inf else None, "max": r4(max(inf)) if inf else None}

    # R 칸 비율 (T별 — 상대 F의 칸 기준)
    rcov = {}
    for nm, cells in mix_cells.items():
        if not nm.startswith("F+R_iv_"):
            continue
        byT = collections.defaultdict(lambda: [0, 0])
        for c in cells:
            if c["pct"] in (1.0, 3.0):
                k = S.tkey(c["season"], c["T"])
                byT[k][1] += 1
                byT[k][0] += 1 if c["has_R"] else 0
        rcov[nm] = {k: byT[k] for k in tsort(byT)}

    # 문턱 판정
    def thr(base, wmode, ver):
        nm = f"{base}+R_{wmode}_{ver}"
        p0 = pred_files[base][0][0]
        t = table[base]["1·3%"]
        res = {}
        for sub in ("all", "rest", "tune"):
            b, m = t[p0][sub], t[nm][sub]
            if b["main"] is None:
                continue
            res[sub] = {"main_base": b["main"], "main_mix": m["main"], "main_gain": b["main"] - m["main"],
                        "end_base": b["end"], "end_mix": m["end"], "end_worse": m["end"] - b["end"],
                        "score_base": b["score"], "score_mix": m["score"], "score_gain": b["score"] - m["score"]}
        return res
    verdict = {}
    for base in bases:
        for wmode in WEIGHTS:
            for ver in VERS:
                verdict[f"{base}+R_{wmode}_{ver}"] = thr(base, wmode, ver)
    v = verdict["F+R_iv_lab"]
    ok_main = v["all"]["main_gain"] >= TH and v["rest"]["main_gain"] >= TH
    ok_end = v["all"]["end_worse"] <= TH and v["rest"]["end_worse"] <= TH
    ok_score = v["all"]["score_gain"] >= TH and v["rest"]["score_gain"] >= TH
    if ok_main and ok_end and ok_score:
        word = "문턱 통과, 전체 score도 0.3%p 이상 나음 → 8c를 계획한다 (1차 가늠)"
    elif ok_main and ok_end:
        word = "문턱 통과, 전체 score 0.3%p 미만 → 「초반에만 몫」, 8c는 사용자가 정한다 (1차 가늠)"
    else:
        word = "문턱 못 넘음 → 지금 자료로는 1~7위 점수의 값어치가 안 보인다 (1차 가늠)"
    verdict_text = (f"상대 F lab·iv: 1·3% 초반 창 |e| {pc(v['all']['main_base'])}→{pc(v['all']['main_mix'])}% (22시즌, {sg(v['all']['main_gain'])}%p) · "
                    f"{pc(v['rest']['main_base'])}→{pc(v['rest']['main_mix'])}% (16시즌, {sg(v['rest']['main_gain'])}%p) · "
                    f"막판 창 나빠짐 {sg(v['all']['end_worse'])}%p/{sg(v['rest']['end_worse'])}%p · 전체 score 개선 {sg(v['all']['score_gain'])}%p/{sg(v['rest']['score_gain'])}%p → {word}")
    print("판정:", verdict_text)

    res = {
        "about": "WP8b 값어치 가늠 1차 (lab7/nr_value.py). 상대 F는 % 관측 예측의 대역이고 개발 검증 자료라 낙관 위험. 조정 6시즌은 M3 묶음에서만 본다",
        "data_ver": "D1", "made": {"tool_head": head, "cmd": cmd, "obs_sha": obs_sha, "n_obs_files": nfiles,
                                    "seconds": round(time.time() - t0, 1)},
        "spec": {"ranks": [2, 7], "kinds": ["A", "B"], "screen": "rank_list", "min_k": 3, "nprior": NPRIOR, "min_prior": MINPRIOR, "threshold": TH,
                 "t_match": "숫자 격자는 숫자로, 막판 T(H−12·H−6·H)는 H에서의 거리로"},
        "verdict_text": verdict_text, "verdict": verdict, "threshold_pass": {"main": ok_main, "end": ok_end, "score": ok_score},
        "table": table, "F_on_tune6": f6,
        "coverage_Z": {v_: {"all": cov[v_]["all"], "byT": cov[v_]["byT"]} for v_ in cov},
        "coverage_R_1_3": rcov, "obs_table_missing": missing_tab, "late_posted_rows_used": late_rows,
        "mix_stat": mix_stat, "w_R": wsum, "report": rep, "sample_Z": sample,
        "wp5a": "WP5a 미정이라 1차에서 뺌",
    }
    with open(os.path.join(out_dir, "value.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    write_md(os.path.join(out_dir, "value.md"), res, tsort, pred_files)
    print(f"끝 {time.time() - t0:.1f}초 → {out_dir}/value.json · value.md")


def write_md(path, res, tsort, pred_files):
    L = []
    w = L.append
    t = res["table"]
    w("# WP8b 숫자단 값어치 가늠 — 1차 결과")
    w("")
    w(f"**판정 (한 줄).** {res['verdict_text']}")
    w("")
    w("> 1차 가늠이다. 상대 F는 「% 관측 예측」의 대역이고 (고르는 대상이 아님), 개발 검증 자료(D1)에서 본 것이라 낙관 위험이 있다.")
    w("> Z·R은 T 절단 사본의 서버 라벨·시각만 쓴다 (끝 라벨 낙관 없음). 인구는 쓰지 않는다. WP5a가 고른 변형은 아직 없어 **WP5a 미정이라 1차에서 뺌**.")
    w("")
    w(f"다시 만드는 명령: `{res['made']['cmd']}` (관측 표는 `obs_table.py build --work W --out W/obs7 --seasons S19-S42 --jobs 4`, 도구 {res['made']['tool_head']}, 표 {res['made']['n_obs_files']}개 sha {res['made']['obs_sha']})")
    w("")
    for base, title in (("F", "상대 F_D1 (22시즌 · 조정 뺀 16시즌)"), ("M3", "상대 M3 본안 (조정 6시즌만)")):
        w(f"## {title}")
        w("")
        w("1·3% 묶음, 평균 |e| (%). 초반 창 = T 18·24·36·48, 막판 창 = T H−12·H−6·H, score = ⅓ 초반 + ⅔ 막판.")
        w("")
        subs = ("all", "rest") if base == "F" else ("tune",)
        names = [n for n, _ in pred_files[base]]
        head = "| 예측 | " + " | ".join(f"{ {'all': '22시즌', 'rest': '16시즌', 'tune': '6시즌'}[s_] } 초반/막판/score" for s_ in subs) + " | 10% score |"
        w(head)
        w("|" + "---|" * (len(subs) + 2))
        for nm in names:
            cells = [f"{pc(t[base]['1·3%'][nm][s_]['main'])} / {pc(t[base]['1·3%'][nm][s_]['end'])} / {pc(t[base]['1·3%'][nm][s_]['score'])}" for s_ in subs]
            w(f"| {nm} | " + " | ".join(cells) + f" | {pc(t[base]['10%'][nm][subs[0]]['score'])} |")
        w("")
        if base == "F":
            w("문턱 (lab·iv): 초반 창 개선 ≥ 0.3%p (22·16시즌 둘 다), 막판 창 나빠짐 ≤ 0.3%p, 전체 score 개선 ≥ 0.3%p.")
            w("")
            w("| 판 | 시즌 | 초반 개선 | 막판 나빠짐 | score 개선 |")
            w("|---|---|---|---|---|")
            for nm in ("F+R_iv_lab", "F+R_half_lab", "F+R_iv_dc", "F+R_half_dc"):
                for s_ in ("all", "rest"):
                    x = res["verdict"][nm][s_]
                    w(f"| {nm} | {'22' if s_ == 'all' else '16'} | {sg(x['main_gain'])}%p | {sg(x['end_worse'])}%p | {sg(x['score_gain'])}%p |")
            w("")
        else:
            f6 = res["F_on_tune6"]["1·3%"]
            w(f"참고: 상대 F를 같은 6시즌만으로 보면 초반 {pc(f6['main'])} / 막판 {pc(f6['end'])}. M3 본안은 조정 6시즌뿐이라 앞 시즌 P가 모자라 분산 역수 무게가 대부분 ½이다 (아래 w_R).")
            w("")
    w("## 보고만")
    w("")
    w("### Z·R이 있는 칸 비율")
    w("")
    cv = res["coverage_Z"]
    w(f"- Z가 있는 (시즌, T) 칸 (S21~S42, 격자 칸 전체): " + ", ".join(f"{k} {v['all'][0]}/{v['all'][1]} ({v['all'][0] / max(v['all'][1], 1) * 100:.0f}%)" for k, v in cv.items()))
    la, lb_ = cv["lab_any"]["all"], cv["lab"]["all"]
    w(f"- screen이 rank_list가 아닌데 rank 2..7인 줄을 더하면 lab 판 Z 칸이 {lb_[0]} → {la[0]} (+{la[0] - lb_[0]}).")
    keys = tsort(cv["lab"]["byT"])
    w("")
    w("| T | " + " | ".join(keys) + " |")
    w("|---|" + "---|" * len(keys))
    for ver in ("lab", "dc"):
        w(f"| Z {ver} | " + " | ".join(f"{cv[ver]['byT'][k][0]}/{cv[ver]['byT'][k][1]}" for k in keys) + " |")
    rk = res["coverage_R_1_3"].get("F+R_iv_lab")
    if rk:
        w(f"| R lab (F 칸, 1·3%) | " + " | ".join(f"{rk[k][0]}/{rk[k][1]}" if k in rk else "-" for k in keys) + " |")
    rk = res["coverage_R_1_3"].get("F+R_iv_dc")
    if rk:
        w(f"| R dc (F 칸, 1·3%) | " + " | ".join(f"{rk[k][0]}/{rk[k][1]}" if k in rk else "-" for k in keys) + " |")
    w("")
    w("### T별 1·3% 평균 |e| 곡선 (%) — 상대 F vs 섞은 것 (22시즌)")
    w("")
    rep = res["report"]["F"]["curve"]["1·3%"]
    cols = ["base", "F+R_iv_lab", "F+R_half_lab", "F+R_iv_dc"]
    w("| T | " + " | ".join(cols) + " |")
    w("|---|" + "---|" * len(cols))
    for k in tsort(rep["base"]):
        w(f"| {k} | " + " | ".join(pc(rep[c][k][0]) if k in rep.get(c, {}) else "-" for c in cols) + " |")
    w("")
    w("R 단독 (R이 있는 칸만, 같은 칸의 F와 견줌 — 평균 |e| %, R / F):")
    w("")
    ra = res["report"]["R_alone"]
    ks = list(ra["F+R_iv_lab"].keys())
    w("| T | " + " | ".join(ks) + " |")
    w("|---|" + "---|" * len(ks))
    for nm in ("F+R_iv_lab", "F+R_iv_dc"):
        w(f"| {nm} | " + " | ".join(f"{pc(ra[nm][k][0], 1)} / {pc(ra[nm][k][1], 1)}" if k in ra[nm] else "-" for k in ks) + " |")
    w("")
    w("### 시즌별 초반 창 1·3% 평균 |e| (%) — 상대 F vs F+R_iv_lab")
    w("")
    ps = res["report"]["F"]["per_season_main"]["1·3%"]
    w("| 시즌 | F | F+R_iv_lab | 차 |")
    w("|---|---|---|---|")
    for s_ in res["report"]["F"]["seasons"]:
        b, m = ps["base"].get(s_), ps["F+R_iv_lab"].get(s_)
        w(f"| {s_}{'*' if s_ in S.TUNE else ''} | {pc(b)} | {pc(m)} | {sg(None if b is None or m is None else b - m)} |")
    w("")
    w("(* 조정 시즌)")
    w("")
    w("### 10% 묶음 (상대 F, score %)")
    w("")
    for nm in ("F", "F+R_iv_lab", "F+R_half_lab", "F+R_iv_dc", "F+R_half_dc"):
        key = nm if nm != "F" else pred_files["F"][0][0]
        x = t["F"]["10%"][key]
        w(f"- {nm}: 22시즌 초반 {pc(x['all']['main'])} / 막판 {pc(x['all']['end'])} / score {pc(x['all']['score'])} · 16시즌 score {pc(x['rest']['score'])}")
    w("")
    w("### 분산 역수 무게 w_R — T별 중앙 (R이 있고 앞 시즌 셋 이상으로 계산된 칸만)")
    w("")
    for nm in ("F+R_iv_lab", "M3+R_iv_lab"):
        ws = res["w_R"].get(nm)
        if not ws:
            continue
        tot_R = sum(v['n_R'] for v in ws.values())
        tot_i = sum(v['n_informed'] for v in ws.values())
        w(f"- {nm}: R 있는 칸 {tot_R}, 그중 계산된 칸 {tot_i} (나머지는 ½)")
        w("  - " + ", ".join(f"{k}: {v['median']:.2f}" for k, v in ws.items() if v["median"] is not None))
    w("")
    w("### 칸 수·기타")
    w("")
    for nm, ms in res["mix_stat"].items():
        w(f"- {nm}: 칸 {ms['cells']}, R 있음 {ms['with_R']}, R 없어 상대 그대로 {ms['no_R']}, 분산 역수 계산 {ms.get('informed', 0)}")
    w(f"- 게시 시각이 T보다 늦은데 Z에 쓰인 줄 (lab/dc): {res['late_posted_rows_used']['lab']} / {res['late_posted_rows_used']['dc']}")
    if res["obs_table_missing"]:
        w(f"- 관측 표가 없는 (시즌, T) 칸 {len(res['obs_table_missing'])}개")
    sm = res["sample_Z"]
    if sm:
        w(f"- 원자료 확인: {sm['season']} T{sm['T']:g} lab — 쓴 줄 {sm['lines']} (표 전체 {sm['lines_total_in_table']}줄), k별 최대 점수 " + ", ".join(f"{k}위 {v:.2f}" for k, v in sm['best_by_k'].items()) + f", Z = {sm['Z']:.4f}")
    w("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--obs", required=True, help="관측 표 폴더 (obs_table.py build --out)")
    r.add_argument("--out", required=True)
    r.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
