"""새 방법 연구 채점 틀 (docs/solo-raid-method.md WP0 d, 「2.1 성공 기준」). 5.3 B의 채점(`lab6/e_53s.py`)과 같은 셈을
방법 어느 것에나 쓰게 한 것이다 — 예측 파일 하나가 방법 하나.

  python3 -I tools/solo_raid/lab7/score.py --pred A.json [--pred B.json ...] --ref F.json --lb lb_D0.json [--out diag.json] [--weights third|half] [--g5] [--calib prev|loso]

--pred 순서가 「단순한 쪽 먼저」 순서다 (0.3%p 안이면 앞 방법을 고른다). --ref는 기준선 F — **참고 한 줄로만** 낸다
(2026-10-31 사용자 결정: 새 로직을 처음부터 만드는 연구라 F와 견주지 않는다. 고르기·지킬 조건에 안 쓴다). 옛 판(`--weights half`)에서만 F가 후보이자 g2·g3의 기준.
자료 판(data_ver)이 다른 파일은 견주지 않는다.

규칙 판 (--weights, 출력 JSON의 weights·rule에 적힘):
  third (기본, 2026-10-31) score = ⅓ × (초반 창 T 18·24·36·48 칸 평균 |e|) + ⅔ × (막판 창 T H−12·H−6·H 칸 평균 |e|).
        지킬 조건은 g1 (중앙 < 하한 − 0.05인 칸 0)과 g4 (80% 포함 두 창 각각 ≥ 0.70)뿐. g2(시즌 끝 허용 밖 수)·g3(최악 시즌)·F는 보고만.
        0.3%p 안이면 --pred 순서가 앞선 규칙. 지킨 규칙이 없으면 「없음」.
  half  (옛 판, 2026-10-31 이전) score = ½·½, 조건 g1 · g2 ≤ F · g3 최악 ≤ F + 0.5%p · g4, F도 후보 (5.3 B 사전 규칙). 옛 결과를 그대로 다시 낼 때만 쓴다
        (WP3b·WP4 결과 절의 숫자).

WP9 스위치 (2026-11-02, 「WP9」 — 둘 다 끄면 위 규칙 그대로, 출력 JSON도 바이트까지 같다. 켠 것만 키가 더해진다):
  --g5   지킬 조건 g5 「F만큼」 (사용자 2026-11-02): 무리(1·3%, 10%) 각각 후보 score − 같은 칸 F(--ref) score ≤ 0.003 (NEAR).
         22시즌 전체(split의 all)와 조정을 뺀 16시즌(split의 rest) **둘 다** 지나야 통과 — 채점 집합에 조정 밖 시즌이 없으면(rest 칸 없음) 전체만 본다.
         떨어지면 fail에 "g5 1·3% +0.26 (전체) / +0.39 (나머지)" (%p, 후보 − F). F는 기준이라 후보가 아니다 (third 판과 같음). score는 중앙값만의 셈이라 --calib과 무관.
  --calib {prev,loso}   구간 보정 층 — 시간 순 등각 예측(conformal) 꼴. 예측 파일 안 칸마다
         r = |ln(정답 ÷ 중앙)| ÷ |ln(정답 쪽 80% 끝 ÷ 중앙)| (정답이 중앙 위면 hi, 아래면 lo; 분모 0인 칸은 뺌). 칸 = 정답 있는 KR 창 칸(초반 main·막판 end).
         맞힐 시즌 s의 배수 k_s = 풀 칸들의 r의 80% 분위(선형 보간, numpy 기본과 같음) — 창(main·end) × 무리(1·3%, 10%)마다 따로.
           prev = 풀이 「s보다 앞 시즌」 (시즌 번호 순; 예측 파일 안의 모든 앞 시즌 — 채점 집합 거르기와 무관)
           loso = 풀이 「s 말고 모든 시즌」 (WP9b 보고용 한 시즌 빼기; 파일 안 시즌 전부. 조정 6시즌만 보려면 조정 6시즌 칸만 든 파일을 준다)
         풀 칸이 10개 미만이면 그 시즌·창·무리는 보정하지 않는다 (k 없음 = 모형 구간 그대로, 손해 보는 쪽).
         보정 구간 = 중앙 × (끝 ÷ 중앙)^k (lo·hi 각각, 창 칸만 — 창 밖 T 칸은 모형 구간). i95는 손대지 않고, CRPS는 모형 구간(i80)으로 센다.
         켜면 cov·w80(g4 포함)이 **보정 구간** 기준이 되고, 모형 구간의 포함률·폭 중앙(cov_model_*·w80_model)과 시즌별 k(groups[무리].calib)를 함께 낸다
         (「모형이 틀린 크기」 보고용). --ref F도 같은 보정을 해서 참고로 낸다.

예측 파일 (공통 꼴):
  {"about": str, "method": "F", "data_ver": "D0",
   "made": {"tool_head": git 짧은 sha, "cmd": str, "inputs": {이름: sha1[:12]}},
   "cells": [{"season": "S21", "T": 18.0, "pct": 1.0, "server": "KR", "pred": 123.4,
              "i80": [lo, hi], "i95": [lo, hi] 또는 null, "own": true}]}
  - own false = 그 방법이 값을 못 냈고 F 값으로 채운 칸 (fallback으로 센다, e_53s.build와 같음).
  - T는 실수 시간. 채점은 server KR만, 다른 서버 칸은 있어도 무시한다.
하한 파일 (모든 방법 공통, lab7/baseline.py lb):
  {"about", "data_ver", "cells": [{"season", "T", "pct", "lb"}]}

셈 (e_53s.evaluate와 같음):
  e = 중앙 ÷ 정답 − 1 (정답은 answer_key). 창 main = T 18·24·36·48, end = T H−12·H−6·H.
  score = w_main × (main 칸 평균 |e|) + w_end × (end 칸 평균 |e|), (w_main, w_end) = ⅓·⅔ (third) 또는 ½·½ (half). 칸 = 시즌 × T × 목표 %, 정답 있는 칸만.
  g1 = 자기 칸(own)에서 T ≥ 18이고 중앙 < 하한 − 0.05인 칸 수 · g2 = T = H에서 |e| > 허용인 (시즌, %) 수 ·
  worst = 시즌별 (두 창 칸 평균 |e|)의 최댓값 · cov = 창별 80% 구간 포함률.
덧붙임 (보고만, 고르기에 안 씀 — third 판에서는 g2·worst·F도 여기에 든다):
  crps = 창 칸 평균의 CRPS 분위 근사. 비율 척도 (중앙·분위 ÷ 정답, 정답 = 1)에서
         CRPS ≈ 2 × (분위 수준 τ ∈ {0.025, 0.1, 0.5, 0.9, 0.975}에 대한 pinball 손실 평균),
         pinball(τ) = (1 − x_τ) × (τ − 1[1 < x_τ]), x_τ = 그 분위 ÷ 정답. 분위: 0.025·0.975 = i95, 0.1·0.9 = i80, 0.5 = pred.
         i95가 없는 칸은 0.1·0.5·0.9 셋만 쓰고 crps_n3에 센다 (표시).
  w80 = 창별 80% 구간 폭 (hi/lo − 1) 중앙.
  split = 조정 시즌(S22·S26·S30·S34·S38·S42)과 나머지 시즌을 나눈 같은 지표 (계획 4.4: 22시즌 전체와 조정 뺀 16시즌을 함께 보고).
"""
import argparse
import collections
import hashlib
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "lab6"))
import anchor as A  # noqa: E402

MAIN_T = (18.0, 24.0, 36.0, 48.0)
TOL = {0.5: 0.03, 1.0: 0.02, 2.0: 0.03, 3.0: 0.02, 5.0: 0.03, 10.0: 0.03}
GROUPS = (("1·3%", (1.0, 3.0)), ("10%", (10.0,)))
NEAR = 0.003   # score 0.3%p 안이면 단순한 쪽
# 규칙 판 (2026-10-31): 이름 → (초반 창 무게, 막판 창 무게, 설명)
WEIGHTS = {
    "third": (1 / 3, 2 / 3, "2026-10-31: score = 1/3 초반 + 2/3 막판; 지킬 조건 g1·g4뿐, g2·g3·F는 보고만"),
    "half": (0.5, 0.5, "옛 판(2026-10-31 이전): score = 1/2·1/2; 조건 g1·g2<=F·g3<=F+0.5%p·g4, F도 후보"),
}
DEFAULT_WEIGHTS = "third"
TUNE = ("S22", "S26", "S30", "S34", "S38", "S42")   # 조정 시즌 (계획 4.4)
# CRPS 근사에 쓰는 분위 (수준, 어느 구간, 끝 번호) — 중앙은 pred
LEVELS95 = ((0.025, "i95", 0), (0.1, "i80", 0), (0.5, None, 0), (0.9, "i80", 1), (0.975, "i95", 1))
LEVELS80 = LEVELS95[1:4]


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha12(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def load_pred(path):
    """예측 파일 → 머리말 dict (cells는 같은 칸이면 뒤 것만 남기고, T·pct는 실수로)."""
    d = jl(path)
    seen = {}
    for c in d["cells"]:
        c = dict(c, T=float(c["T"]), pct=float(c["pct"]))
        seen[(c["season"], c.get("server", "KR"), c["T"], c["pct"])] = c
    return dict(d, cells=list(seen.values()))


def load_lb(path):
    """하한 파일 → (머리말, {(시즌, T, %): lb})."""
    d = jl(path)
    return d, {(c["season"], float(c["T"]), float(c["pct"])): c["lb"] for c in d["cells"] if c.get("lb") is not None}


def window(s, T):
    H = A.info(s)[1]
    if T in MAIN_T:
        return "main"
    if T in (H - 12, H - 6, H):
        return "end"
    return None


def tkey(s, T):
    H = A.info(s)[1]
    return "H" if T == H else ("H-6" if T == H - 6 else ("H-12" if T == H - 12 else f"{T:g}"))


def pinball(tau, x):
    """정답 = 1일 때 분위 x(정답 ÷ 비)의 pinball 손실."""
    return (1.0 - x) * (tau - (1.0 if 1.0 < x else 0.0))


def crps_cell(c, tv):
    """한 칸의 CRPS 근사 (비율 척도) → (값, 5분위 썼나)."""
    has95 = bool(c.get("i95")) and None not in c["i95"]
    tot = 0.0
    lv = LEVELS95 if has95 else LEVELS80
    for tau, key, j in lv:
        iv = c if key != "i80" else dict(c, i80=c.get("i80_m", c["i80"]))   # CRPS는 보정 전 모형 구간
        x = (c["pred"] if key is None else iv[key][j]) / tv
        tot += pinball(tau, x)
    return 2.0 * tot / len(lv), has95


def evaluate(cells, truth, lb, pcts, tol, seasons=None, weights=DEFAULT_WEIGHTS):
    """e_53s.evaluate와 같은 셈. cells = 예측 파일 칸 목록, truth = A.truth(), lb = {(시즌, T, %): 하한}.
    seasons를 주면 그 시즌 칸만 (조정/나머지 나눔용). weights = WEIGHTS의 이름 (기본 third ⅓·⅔, half = 옛 ½·½)."""
    w_main, w_end = WEIGHTS[weights][:2]
    win = collections.defaultdict(list)       # 창 → [(시즌, |e|, 포함, 폭, crps, 5분위)]
    per_season = collections.defaultdict(list)
    endH = []
    g1 = 0
    fb = collections.Counter()
    byT = collections.defaultdict(list)
    cellsE = []
    for c in cells:
        q, s, T = c["pct"], c["season"], c["T"]
        if c.get("server", "KR") != "KR" or q not in pcts or (seasons is not None and s not in seasons):
            continue
        tv = truth.get((s, "KR", q))
        if tv is None or not c.get("pred") or not c.get("i80") or None in c["i80"]:
            continue
        lo, hi = c["i80"]
        mlo, mhi = c.get("i80_m", c["i80"])
        own = bool(c.get("own", True))
        e = c["pred"] / tv - 1
        H = A.info(s)[1]
        w = window(s, T)
        wid = hi / lo - 1 if lo > 0 else None
        inside = lo <= tv <= hi
        cr, h95 = crps_cell(c, tv)
        byT[tkey(s, T)].append((s, e, inside, wid, own))
        lbv = lb.get((s, T, q))
        if T >= 18 and own and lbv is not None and c["pred"] < lbv - 0.05:
            g1 += 1
        if w is None:
            continue
        win[w].append((s, abs(e), inside, wid, cr, h95, mlo <= tv <= mhi, mhi / mlo - 1 if mlo > 0 else None))
        per_season[s].append(abs(e))
        cellsE.append((s, T, q, w, abs(e)))
        if not own:
            fb[w] += 1
        if T == H and abs(e) > tol[q] + 1e-12:
            endH.append(f"{s} {q:g}% {e * 100:+.1f}%")

    def mean(xs):
        return st.mean(xs) if xs else None

    sc = {w: mean([x[1] for x in win[w]]) for w in ("main", "end")}
    # 시즌 중앙 score: (%, T) 칸마다 시즌 중앙 |e|의 평균, 창별
    medsc = {}
    for w in ("main", "end"):
        g = collections.defaultdict(list)
        for s, T, q, ww, ae in cellsE:
            if ww == w:
                g[(q, tkey(s, T))].append(ae)
        medsc[w] = mean([st.median(v) for v in g.values()])
    worst = max(((st.mean(v), s) for s, v in per_season.items()), default=(None, None))
    cov = {w: mean([x[2] for x in win[w]]) for w in ("main", "end")}
    no2123 = [x for w in ("main", "end") for x in win[w] if x[0] not in ("S21", "S22", "S23")]
    both = lambda a, b: None if a is None or b is None else w_main * a + w_end * b   # noqa: E731
    allw = win["main"] + win["end"]
    res = dict(weights=weights, score=both(sc["main"], sc["end"]), main=sc["main"], end=sc["end"],
                score_med=both(medsc["main"], medsc["end"]), cov_main=cov["main"], cov_end=cov["end"],
                g1=g1, g2=len(endH), g2_list=endH, worst=worst[0], worst_season=worst[1],
                fallback=dict(fb), n=dict((w, len(win[w])) for w in ("main", "end")),
                score_no2123=mean([x[1] for x in no2123]),
                crps=mean([x[4] for x in allw]), crps_main=mean([x[4] for x in win["main"]]),
                crps_end=mean([x[4] for x in win["end"]]), crps_n3=sum(1 for x in allw if not x[5]),
                w80={w: st.median([x[3] for x in win[w] if x[3] is not None] or [0]) for w in ("main", "end")},
                byT={k: dict(n=len(v), mae=st.mean(abs(x[1]) for x in v), bias=st.mean(x[1] for x in v),
                             cov=st.mean(x[2] for x in v), w80=st.median([x[3] for x in v if x[3] is not None] or [0]),
                             own=sum(x[4] for x in v)) for k, v in byT.items()})
    if any("i80_m" in c for c in cells):   # --calib: 모형 구간(보정 전)의 포함률·폭 중앙을 함께
        res["cov_model_main"] = mean([x[6] for x in win["main"]])
        res["cov_model_end"] = mean([x[6] for x in win["end"]])
        res["w80_model"] = {w: st.median([x[7] for x in win[w] if x[7] is not None] or [0]) for w in ("main", "end")}
    return res


SPLIT_KEYS = ("score", "main", "end", "cov_main", "cov_end", "crps", "worst", "n", "w80")
CALIB_KEYS = ("cov_model_main", "cov_model_end", "w80_model")
CALIB_MIN = 10   # 보정 풀 칸이 이보다 적으면 보정하지 않는다 (WP9)
CALIB_Q = 0.8


def evaluate_split(cells, truth, lb, pcts, tol, weights=DEFAULT_WEIGHTS):
    """전체(22시즌) · 조정 6시즌 · 나머지 16시즌 — 앞 둘은 요약만."""
    allr = evaluate(cells, truth, lb, pcts, tol, weights=weights)
    tune = evaluate(cells, truth, lb, pcts, tol, seasons=set(TUNE), weights=weights)
    rest_s = {c["season"] for c in cells} - set(TUNE)
    rest = evaluate(cells, truth, lb, pcts, tol, seasons=rest_s, weights=weights)
    keys = SPLIT_KEYS + (CALIB_KEYS if "cov_model_main" in allr else ())
    pick = lambda r: {k: r[k] for k in keys}   # noqa: E731
    return allr, dict(all=pick(allr), tune=pick(tune), rest=pick(rest))


def quantile(xs, q):
    """선형 보간 분위 (numpy.percentile 기본과 같음)."""
    v = sorted(xs)
    h = (len(v) - 1) * q
    i = int(h)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (h - i)


def snum(s):
    return int(s[1:])


def calibrate(cells, truth, pcts, mode):
    """구간 보정 층 (WP9, 위 docstring). 반환 (칸 목록 — 창 칸의 i80은 보정 구간·i80_m에 모형 구간, 시즌별 k).
    k = {시즌: {"main": {"k": 배수 또는 None, "n": 풀 칸 수}, "end": ...}}."""
    rs = collections.defaultdict(list)       # (시즌, 창) → [r]
    for c in cells:
        s, T, q = c["season"], c["T"], c["pct"]
        if c.get("server", "KR") != "KR" or q not in pcts:
            continue
        r = cell_r(c, truth.get((s, "KR", q)))
        w = window(s, T)
        if r is not None and w is not None:
            rs[(s, w)].append(r)
    seasons = sorted({c["season"] for c in cells}, key=snum)
    kinfo = {}
    for s in seasons:
        kinfo[s] = {}
        for w in ("main", "end"):
            pool = [r for t in seasons if (snum(t) < snum(s) if mode == "prev" else t != s) for r in rs[(t, w)]]
            kinfo[s][w] = dict(k=quantile(pool, CALIB_Q) if len(pool) >= CALIB_MIN else None, n=len(pool))
    out = []
    for c in cells:
        s, T, q = c["season"], c["T"], c["pct"]
        w = window(s, T) if c.get("server", "KR") == "KR" and q in pcts else None
        k = kinfo[s][w]["k"] if w else None
        iv = c.get("i80")
        if w is None or not c.get("pred") or not iv or None in iv or iv[0] <= 0:
            out.append(c)
            continue
        m = c["pred"]
        k = 1.0 if k is None else k
        out.append(dict(c, i80=[m * (iv[0] / m) ** k, m * (iv[1] / m) ** k], i80_m=list(iv)))
    return out, kinfo


def cell_r(c, tv):
    """r = |ln(정답 ÷ 중앙)| ÷ |ln(정답 쪽 80% 끝 ÷ 중앙)|. 정답·구간이 없거나 분모가 0이면 None."""
    iv = c.get("i80")
    m = c.get("pred")
    if tv is None or not m or not iv or None in iv or iv[0] <= 0 or iv[1] <= 0 or m <= 0:
        return None
    den = abs(math.log((iv[1] if tv > m else iv[0]) / m))
    return None if den == 0 else abs(math.log(tv / m)) / den


def g5_fail(gname, split, Fsplit, name):
    """g5: 후보 − F score ≤ NEAR, 22시즌 전체(all)와 조정 뺀 나머지(rest) 둘 다. 반환 (fail 문구 목록, 차 dict)."""
    diffs = {}
    for part in ("all", "rest"):
        a, b = split[name][part]["score"], Fsplit[part]["score"]
        diffs[part] = None if a is None or b is None else a - b
    if split[name]["rest"]["n"]["main"] + split[name]["rest"]["n"]["end"] == 0 and Fsplit["rest"]["n"]["main"] + Fsplit["rest"]["n"]["end"] == 0:
        diffs["rest"] = None          # 조정 밖 시즌 없음 — 전체만 본다
        need = ("all",)
    else:
        need = ("all", "rest")
    if any(diffs[p] is None for p in need):
        return [f"g5 {gname} 칸 모자람"], diffs
    if all(diffs[p] <= NEAR + 1e-12 for p in need):
        return [], diffs
    txt = " / ".join(f"{diffs[p] * 100:+.2f} ({'전체' if p == 'all' else '나머지'})" for p in need)
    return [f"g5 {gname} {txt}"], diffs


def select(res, Fres, order, weights=DEFAULT_WEIGHTS, extra=None):
    """사전 규칙. third (기본, 2026-10-31): g1 0 · 80% 포함 두 창 ≥ 0.70 을 지킨 것 중 score 최소, 0.3%p 안이면 order에서 앞선 쪽.
    g2·worst는 기록(note)만, F는 고르기에 안 든다 — 지킨 것이 없으면 (None, 기록).
    half (옛 판, 5.3 B): g1 0 · g2 ≤ F · 최악 ≤ F + 0.5%p · 80% 포함 ≥ 0.70 을 지킨 것 중 score 최소, F도 후보(맨 뒤).
    extra = {이름: [탈락 문구]} (--g5: 켜면 g5 탈락 문구를 더한다, 없으면 기본 규칙 그대로).
    res {이름: 평가}. 반환 (고른 이름, 기록)."""
    old = weights == "half"
    log = []
    ok = {}
    for name in order:
        r = res[name]
        why = []
        note = []
        if r["score"] is None or r["worst"] is None or r["cov_main"] is None or r["cov_end"] is None:
            why.append("칸 모자람")
        else:
            if r["g1"] > 0:
                why.append(f"g1 {r['g1']}")
            if r["g2"] > Fres["g2"]:
                (why if old else note).append(f"g2 {r['g2']} > F {Fres['g2']}")
            if r["worst"] > Fres["worst"] + 0.005:
                (why if old else note).append(f"g3 최악 {r['worst'] * 100:.2f} > F {Fres['worst'] * 100:.2f}+0.5")
            if r["cov_main"] < 0.70 or r["cov_end"] < 0.70:
                why.append(f"g4 포함 {r['cov_main']:.2f}/{r['cov_end']:.2f}")
            why.extend((extra or {}).get(name, []))
        log.append(dict(name=name, score=None if r["score"] is None else round(r["score"] * 100, 3), fail=why, note=note))
        if not why:
            ok[name] = r["score"]
    if old:
        ok["F"] = Fres["score"]
        log.append(dict(name="F", score=round(Fres["score"] * 100, 3), fail=[], note=[]))
    if not ok:
        return None, log
    best = min(ok.values())
    for name in list(order) + (["F"] if old else []):
        if name in ok and ok[name] <= best + NEAR:
            return name, log
    return ("F" if old else None), log


def pct(x, d=2):
    return "  -  " if x is None else f"{x * 100:.{d}f}"


def line(name, r):
    fbs = sum(r["fallback"].values())
    return (f"   {name:10s} score {pct(r['score'])} (main {pct(r['main'])} · end {pct(r['end'])}) 중앙판 {pct(r['score_med'])} "
            f"포함 {pct(r['cov_main'], 0)}/{pct(r['cov_end'], 0)} g1 {r['g1']} g2 {r['g2']} 최악 {pct(r['worst'])} {r['worst_season'] or ''} "
            f"CRPS {pct(r['crps'])} 폭 {pct(r['w80']['main'], 1)}/{pct(r['w80']['end'], 1)} F대신 {fbs}"
            + (f" (5분위 아닌 칸 {r['crps_n3']})" if r["crps_n3"] else ""))


def main():
    ap = argparse.ArgumentParser(description="예측 파일 채점 (e_53s와 같은 셈)")
    ap.add_argument("--pred", action="append", required=True, help="예측 파일 (순서 = 단순한 쪽 먼저)")
    ap.add_argument("--ref", required=True, help="기준선 F 예측 파일")
    ap.add_argument("--lb", required=True, help="하한 파일")
    ap.add_argument("--out")
    ap.add_argument("--weights", choices=sorted(WEIGHTS), default=DEFAULT_WEIGHTS,
                    help="third = ⅓·⅔ 새 규칙(기본) · half = 옛 ½·½ 규칙(옛 결과 재현)")
    ap.add_argument("--g5", action="store_true", help="지킬 조건 g5 「F만큼」 (22시즌 전체·나머지 16시즌 둘 다) — WP9")
    ap.add_argument("--calib", choices=("prev", "loso"), help="구간 보정 층: prev = 앞 시즌만, loso = 그 시즌 빼고 전부 — WP9")
    a = ap.parse_args()
    ref = load_pred(a.ref)
    lbh, lb = load_lb(a.lb)
    preds = [load_pred(p) for p in a.pred]
    vers = {ref["data_ver"], lbh["data_ver"]} | {p["data_ver"] for p in preds}
    if len(vers) != 1:
        sys.exit(f"자료 판이 섞였다: {sorted(vers)} — 같은 판끼리만 견준다")
    names = []
    for p, path in zip(preds, a.pred):
        n = p["method"]
        if n in names or n == ref["method"]:
            n = f"{n}@{os.path.basename(path)}"
        names.append(n)
    truth = A.truth()
    out = {"about": "새 방법 채점 (lab7/score.py, e_53s와 같은 셈)", "data_ver": vers.pop(), "ref": ref["method"],
           "files": {os.path.basename(p): sha12(p) for p in a.pred + [a.ref, a.lb]},
           "weights": a.weights, "w_main": WEIGHTS[a.weights][0], "w_end": WEIGHTS[a.weights][1], "rule": WEIGHTS[a.weights][2], "groups": {}}
    if a.g5:
        out["g5"] = f"후보 score - F score <= {NEAR * 100:.1f}%p, 22시즌 전체와 조정 뺀 나머지 둘 다 (WP9)"
    if a.calib:
        out["calib"] = dict(mode=a.calib, min_cells=CALIB_MIN, q=CALIB_Q, about="구간 보정 층 (WP9, score.py 머리 docstring): cov·w80은 보정 구간, cov_model_*·w80_model은 모형 구간")
    for gname, qs in GROUPS:
        kinfo = {}
        rcells = ref["cells"]
        if a.calib:
            rcells, kinfo["F"] = calibrate(rcells, truth, qs, a.calib)
        Fres, Fsplit = evaluate_split(rcells, truth, lb, qs, TOL, a.weights)
        res, split = {}, {}
        for n, p in zip(names, preds):
            pcells = p["cells"]
            if a.calib:
                pcells, kinfo[n] = calibrate(pcells, truth, qs, a.calib)
            res[n], split[n] = evaluate_split(pcells, truth, lb, qs, TOL, a.weights)
        extra, g5d = None, {}
        if a.g5:
            extra = {}
            for n in names:
                extra[n], g5d[n] = g5_fail(gname, split, Fsplit, n)
        pick, log = select(res, Fres, names, a.weights, extra)
        if a.g5:
            for x in log:
                x["g5"] = {k: None if v is None else round(v * 100, 4) for k, v in g5d[x["name"]].items()}
        out["groups"][gname] = dict(pick=pick, log=log, F=Fres, cands=res, split=dict(F=Fsplit, **split))
        if a.calib:
            out["groups"][gname]["calib"] = kinfo
        print(f"== {gname}: 고름 {pick or '없음'}  (자료 판 {out['data_ver']}, 무게 {a.weights} {WEIGHTS[a.weights][0]:.3f}/{WEIGHTS[a.weights][1]:.3f}, 조건은 select 규칙)")
        print(line(ref["method"] + (" (참고)" if a.weights != "half" else ""), Fres))
        for n in names:
            print(line(n, res[n]))
        print("   -- 22시즌 전체 / 조정 6시즌 / 나머지 16시즌 (score · CRPS)")
        for n, sp in [(ref["method"], Fsplit)] + [(n, split[n]) for n in names]:
            print(f"   {n:10s} " + "  ".join(f"{k} {pct(sp[k]['score'])}·{pct(sp[k]['crps'])}" for k in ("all", "tune", "rest")))
        if a.calib:
            print(f"   -- 구간 보정 {a.calib} (칸 {CALIB_MIN}개 미만이면 보정 안 함) — 보정 포함·폭 대 모형 포함·폭 (초반/막판), 전체/조정/나머지")
            for n, r0 in [(ref["method"], Fres)] + [(n, res[n]) for n in names]:
                sp = Fsplit if n == ref["method"] else split[n]
                print(f"   {n:10s} " + "  ".join(
                    f"{k}: 포함 {pct(sp[k]['cov_main'], 2)}/{pct(sp[k]['cov_end'], 2)} (모형 {pct(sp[k]['cov_model_main'], 2)}/{pct(sp[k]['cov_model_end'], 2)}) "
                    f"폭 {pct(sp[k]['w80']['main'], 1)}/{pct(sp[k]['w80']['end'], 1)} (모형 {pct(sp[k]['w80_model']['main'], 1)}/{pct(sp[k]['w80_model']['end'], 1)})"
                    for k in ("all", "tune", "rest")))
            for n in [ref["method"]] + names:
                ki = kinfo["F" if n == ref["method"] else n]
                for w, wn in (("main", "초반"), ("end", "막판")):
                    ks = [(s, v[w]["k"]) for s, v in sorted(ki.items(), key=lambda t: snum(t[0]))]
                    got = [k for _, k in ks if k is not None]
                    rng = f"k {min(got):.2f}~{max(got):.2f} 중앙 {st.median(got):.2f}" if got else "k 없음"
                    print(f"   k {n:10s} {wn}: {rng} (보정 안 한 시즌 {len(ks) - len(got)}/{len(ks)}) " + " ".join(f"{s}:{'-' if k is None else format(k, '.2f')}" for s, k in ks))
        for x in log:
            if x["fail"]:
                print(f"   탈락 {x['name']}: {'; '.join(x['fail'])}")
            if x.get("note"):
                print(f"   보고만 {x['name']}: {'; '.join(x['note'])}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
