"""새 방법 연구 채점 틀 (docs/solo-raid-method.md WP0 d, 「2.1 성공 기준」). 5.3 B의 채점(`lab6/e_53s.py`)과 같은 셈을
방법 어느 것에나 쓰게 한 것이다 — 예측 파일 하나가 방법 하나.

  python3 -I tools/solo_raid/lab7/score.py --pred A.json [--pred B.json ...] --ref F.json --lb lb_D0.json [--out diag.json] [--weights third|half]

--pred 순서가 「단순한 쪽 먼저」 순서다 (0.3%p 안이면 앞 방법을 고른다). --ref는 기준선 F — **참고 한 줄로만** 낸다
(2026-10-31 사용자 결정: 새 로직을 처음부터 만드는 연구라 F와 견주지 않는다. 고르기·지킬 조건에 안 쓴다). 옛 판(`--weights half`)에서만 F가 후보이자 g2·g3의 기준.
자료 판(data_ver)이 다른 파일은 견주지 않는다.

규칙 판 (--weights, 출력 JSON의 weights·rule에 적힘):
  third (기본, 2026-10-31) score = ⅓ × (초반 창 T 18·24·36·48 칸 평균 |e|) + ⅔ × (막판 창 T H−12·H−6·H 칸 평균 |e|).
        지킬 조건은 g1 (중앙 < 하한 − 0.05인 칸 0)과 g4 (80% 포함 두 창 각각 ≥ 0.70)뿐. g2(시즌 끝 허용 밖 수)·g3(최악 시즌)·F는 보고만.
        0.3%p 안이면 --pred 순서가 앞선 규칙. 지킨 규칙이 없으면 「없음」.
  half  (옛 판, 2026-10-31 이전) score = ½·½, 조건 g1 · g2 ≤ F · g3 최악 ≤ F + 0.5%p · g4, F도 후보 (5.3 B 사전 규칙). 옛 결과를 그대로 다시 낼 때만 쓴다
        (WP3b·WP4 결과 절의 숫자).

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
        x = (c["pred"] if key is None else c[key][j]) / tv
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
        win[w].append((s, abs(e), inside, wid, cr, h95))
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
    return dict(weights=weights, score=both(sc["main"], sc["end"]), main=sc["main"], end=sc["end"],
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


SPLIT_KEYS = ("score", "main", "end", "cov_main", "cov_end", "crps", "worst", "n", "w80")


def evaluate_split(cells, truth, lb, pcts, tol, weights=DEFAULT_WEIGHTS):
    """전체(22시즌) · 조정 6시즌 · 나머지 16시즌 — 앞 둘은 요약만."""
    allr = evaluate(cells, truth, lb, pcts, tol, weights=weights)
    tune = evaluate(cells, truth, lb, pcts, tol, seasons=set(TUNE), weights=weights)
    rest_s = {c["season"] for c in cells} - set(TUNE)
    rest = evaluate(cells, truth, lb, pcts, tol, seasons=rest_s, weights=weights)
    pick = lambda r: {k: r[k] for k in SPLIT_KEYS}   # noqa: E731
    return allr, dict(all=pick(allr), tune=pick(tune), rest=pick(rest))


def select(res, Fres, order, weights=DEFAULT_WEIGHTS):
    """사전 규칙. third (기본, 2026-10-31): g1 0 · 80% 포함 두 창 ≥ 0.70 을 지킨 것 중 score 최소, 0.3%p 안이면 order에서 앞선 쪽.
    g2·worst는 기록(note)만, F는 고르기에 안 든다 — 지킨 것이 없으면 (None, 기록).
    half (옛 판, 5.3 B): g1 0 · g2 ≤ F · 최악 ≤ F + 0.5%p · 80% 포함 ≥ 0.70 을 지킨 것 중 score 최소, F도 후보(맨 뒤).
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
    for gname, qs in GROUPS:
        Fres, Fsplit = evaluate_split(ref["cells"], truth, lb, qs, TOL, a.weights)
        res, split = {}, {}
        for n, p in zip(names, preds):
            res[n], split[n] = evaluate_split(p["cells"], truth, lb, qs, TOL, a.weights)
        pick, log = select(res, Fres, names, a.weights)
        out["groups"][gname] = dict(pick=pick, log=log, F=Fres, cands=res, split=dict(F=Fsplit, **split))
        print(f"== {gname}: 고름 {pick or '없음'}  (자료 판 {out['data_ver']}, 무게 {a.weights} {WEIGHTS[a.weights][0]:.3f}/{WEIGHTS[a.weights][1]:.3f}, 조건은 select 규칙)")
        print(line(ref["method"] + (" (참고)" if a.weights != "half" else ""), Fres))
        for n in names:
            print(line(n, res[n]))
        print("   -- 22시즌 전체 / 조정 6시즌 / 나머지 16시즌 (score · CRPS)")
        for n, sp in [(ref["method"], Fsplit)] + [(n, split[n]) for n in names]:
            print(f"   {n:10s} " + "  ".join(f"{k} {pct(sp[k]['score'])}·{pct(sp[k]['crps'])}" for k in ("all", "tune", "rest")))
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
