"""솔로레이드 여러 시즌 검증 — 다른 시즌들로 만든 prior로 한 시즌을 T마다 맞혀 본다 (할 일 3.6, 2026-10-16, 모델 0, 요청 0).

python3 -I tools/solo_raid/xval.py [--seasons S38,S39,S40,S41] [--scheme forward,loso] [--T 12,18,...] [--draws 600]
        [--work DIR] [--out F.json] [--tag dev] [--prior-opt key=value ...] [--predict-opt key=value ...]
        [--targets S21-S41] [--jobs 4] [--slim] [--unknown-opt key=value ...] [--level-update '{}'] [--boot-single 200]

- 겹(fold). forward(기본): 시즌 번호 순으로, 앞 시즌들(학습 가능한 개발 시즌 2개 이상) → 다음 시즌. loso(비교): 그 시즌만 빼고 나머지 전부.
  4절 기준 시즌 규칙 (2026-10-17, 분석 4.0a·4.3 — criteria_v2.json simple_order):
  · prev (M0): 맞힐 시즌보다 앞인 학습 등급 full 시즌 가운데 가장 가까운 하나의 **한 시즌 prior** (fit_prior.fit, 운영 S41 prior와 같은 붓스트랩 --boot-single).
    빌릴 비율도 그 시즌. 「S41 고정」(지금 운영)과 달리 맞힐 시즌마다 직전 시즌을 쓴다.
  · recent{k} (M2a, 예: recent5): 앞 시즌 가운데 학습 가능한(full·shape_only) 가장 가까운 k개로 여러 시즌 prior. 앞 시즌이 k개 미만이면 있는 만큼 (full 1개 이상).
  · pairs (M0′ 진단 — 고르기에 쓰지 않음): 맞힐 시즌마다 다른 full 시즌 하나씩의 한 시즌 prior 전부. 겹이 많다 (23 × 22) — --jobs·--slim과 함께.
  · fixed: --fixed-prior 하나로 (맞힐 시즌에서 --fixed-train을 뺀다. 기본은 prior의 season — 「S38+S40」이면 둘 다).
  학습 시즌 = 등록부(registry.json) 몫 dev이고 학습 등급이 none이 아닌 시즌. 맞힐 시즌 = 정답 표(answer_key.json)에 채점 정답이 있는 개발 시즌.
  prior는 fit_prior_multi.fit_multi(그 겹의 학습 시즌만). 맞힐 시즌의 정답·자료는 prior에 들어가지 않는다.
- tr(T 절단 재처리)만: 맞힐 시즌 state에서 T까지 게시된 글·사진만 남긴 사본에 post_all.sh를 다시 돌리고 (backtest.trunc_state,
  시즌 전용 입력은 post_all_stamp.json의 extra) predict.predict로 낸다. 서버 미상 줄은 site 방식, 빌릴 비율은 그 겹의 학습 시즌들.
- 채점: answer_key.json의 truth(criteria.json 규칙 — 한섭 3·10% 집계표, 그 밖 곡선 strong)와 집계표·곡선 각각.
  e = 중앙값/정답 − 1, 80·50·95% 포함, 80% 폭, 기준선 0·1, 등급(predict warning level).
- 요약은 criteria.json(판 1)의 성공·고르기 규칙대로(summary — 웹 viz가 loso를 다시 계산해 견주므로 summarize는 바꾸지 않는다).
  criteria_v2.json(4절 고르기 기준)이 있으면 summary_v2도 낸다: T_main 18~48·T_late 72·96·T12 따로, 보고 지표(낮게 예측한 비율, 시즌별 80% 포함 최저·분포,
  자료량 셋 구간, 95% 포함, 10%), 그리고 select_v2(scheme끼리 고르기 규칙 — 결과 보기 전에 정한 순서 prev → recent{k} → forward).
  **시험 시즌은 넣지 않는다** (split 감시 — 판 2에는 잠긴 시즌이 없다).
- 덧붙임 (기본 끔): --level-update는 자체 일관성 수준 갱신(M4, m_update.py — 한 시즌 prior 위에서도 된다), --unknown-opt는 서버 미상 줄 설정의 일부만 바꾼다
  (--predict-opt unknown=…은 dict 전체를 바꿔 빌릴 시즌이 빠진다). contam(H2)은 4.1에서 「아님」이라 넣지 않았다.
- --jobs N: T 절단 사본·prior를 먼저 만들고 예측을 N 프로세스로 (결과 줄 순서는 1 프로세스와 같다).
  --slim: 한섭 줄만·요약에 쓰는 열만 남기고 들여쓰기 없이 쓴다 (M0′처럼 겹이 많을 때 — 23시즌 pairs가 수십 MB에서 몇 MB로).
  pairs면 pairs_v2(기준 시즌 × 맞힐 시즌의 T_main·한섭 1·3% 평균 e·|e|·포함)도 낸다.
출력(기본 data/solo-raid/analysis/xval_{tag}.json): 겹·T·서버·지점별 줄, 요약, 실행 정보(시즌 목록·state 도장·split 판·도구 HEAD·criteria 시각).
"""
import argparse
import collections
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import predict as PR  # noqa: E402
import backtest as BT  # noqa: E402
import fit_prior as FP  # noqa: E402
import fit_prior_multi as FM  # noqa: E402
import split as SP  # noqa: E402

DATA = os.path.join(ROOT, "data", "solo-raid")
AN = os.path.join(DATA, "analysis")
MIN_TRAIN = 2
CRIT2 = os.path.join(AN, "criteria_v2.json")
SLIM_KEYS = ("scheme", "train", "season", "T", "server", "pct", "tol", "pred", "i80", "grade", "n_persons", "n_persons_label",
             "truth", "err", "ok", "in50", "in80", "in95", "w80", "err_b0", "err_b1", "m_upd", "tool_note")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def single_scheme(scheme):
    """한 시즌 prior를 쓰는 scheme (prev·pairs)."""
    return scheme in ("prev", "pairs")


def folds(seasons, reg, scheme):
    """→ [(학습 시즌 튜플, 맞힐 시즌)]. scheme: forward · loso · prev · recent{k} · pairs (머리말)."""
    train_ok = [s for s in seasons if reg[s]["role"] == "dev" and reg[s]["tier"] != "none"]
    order = sorted(seasons, key=SP.snum)
    full = [s for s in sorted(train_ok, key=SP.snum) if reg[s]["tier"] == "full"]
    out = []
    for tgt in order:
        if scheme == "prev":
            c = [s for s in full if SP.snum(s) < SP.snum(tgt)]
            if c:
                out.append(((c[-1],), tgt))
            continue
        if scheme == "pairs":
            out += [((r,), tgt) for r in full if r != tgt]
            continue
        if scheme == "forward" or scheme.startswith("recent"):
            tr = [s for s in train_ok if SP.snum(s) < SP.snum(tgt)]
        elif scheme == "loso":
            tr = [s for s in train_ok if s != tgt]
        else:
            raise SystemExit(f"모르는 scheme: {scheme} (forward·loso·prev·recent{{k}}·pairs)")
        tr = sorted([s for s in tr if reg[s]["tier"] in ("full", "shape_only")], key=SP.snum)
        if scheme.startswith("recent"):
            tr = tr[-int(scheme[len("recent"):]):]
        if sum(1 for s in tr if reg[s]["tier"] == "full") < 1 or len(tr) < (1 if scheme.startswith("recent") else MIN_TRAIN):
            continue
        out.append((tuple(tr), tgt))
    return out


def stamp_digest(name):
    """시즌 state의 후처리 도장 digest — 등록부가 낡아도 지금 도장을 쓴다 (2026-10-17, 분석 4.0a)."""
    p = os.path.join(FM.state_of(name), "post_all_stamp.json")
    try:
        return jload(p)["digest"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def prior_for(train, work, boot, seed, popts, single=False):
    """겹의 prior (캐시). single이면 그 한 시즌의 fit_prior.fit (prev·pairs), 아니면 fit_multi.
    캐시 열쇠에 학습 시즌의 후처리 도장 digest, prior 도구 파일·집계표 파일 sha1을 넣는다 (등록부가 아니라 state의 도장 — 2026-10-17)."""
    tools = [os.path.join(HERE, f) for f in ("fit_prior.py", "fit_prior_multi.py", "explore_pct_transform.py", "predict.py")]
    key = hashlib.sha1(json.dumps([train, boot, seed, {} if single else popts, "single" if single else "multi",
                                   [stamp_digest(s) for s in train],
                                   [hashlib.sha1(open(t, "rb").read()).hexdigest() for t in tools if os.path.exists(t)],
                                   # 여러 시즌 prior는 집계표(seasons.json·server_cutoffs.csv)로 모양·서버 비를 잡는다 — 표를 고치면 다시 맞춘다 (2026-10-17)
                                   [hashlib.sha1(SP._real_open(t, "rb").read()).hexdigest() for t in (SP.SEASONS, SP.SERVER_CUTOFFS)]],
                                  sort_keys=True).encode()).hexdigest()[:12]
    name = "+".join(train) if len(train) <= 6 else f"{train[0]}..{train[-1]}_{len(train)}"
    p = os.path.join(work, "priors", f"prior_{'1_' if single else ''}{name}_{key}.json")
    if os.path.exists(p):
        return jload(p), p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if single:
        pr = FP.fit(FM.state_of(train[0]), boot=boot, seed=seed, quiet=True)
    else:
        pr = FM.fit_multi(list(train), boot=boot, seed=seed, quiet=True, **popts)
    pr = json.loads(json.dumps(PR.jsonable(pr)))
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(pr, f, ensure_ascii=False)
    os.replace(p + ".tmp", p)
    return pr, p


def extra_of(state):
    st = jload(os.path.join(state, "post_all_stamp.json"))
    return {k: os.path.join(ROOT, v["path"]) for k, v in st.get("extra", {}).items()}


def score(out, akey, tgt, T, fold_name, train, crit):
    rows = []
    tol = crit["tol"]
    for r in akey:
        if r["season"] != tgt:
            continue
        sv = out["servers"].get(r["server"])
        if not sv:
            continue
        d = sv["points"].get(str(float(r["pct"])))
        if not d or d.get("median") is None:
            continue
        q = float(r["pct"])
        tl = tol.get(f"{q:g}", tol["aux"])
        med = d["median"]
        rec = {"scheme": fold_name, "train": list(train), "season": tgt, "T": T, "server": r["server"], "pct": q, "tol": tl,
               "pred": round(med, 3), "i80": d["i80"], "i95": d["i95"], "grade": (sv.get("warning") or {}).get("level"),
               "basis_pred": sv.get("basis"), "n_persons": sv.get("n_persons"), "n_persons_label": sv.get("n_persons_label"),
               "truth": r["truth"], "truth_basis": r["basis"], "shaky": r["shaky"],
               "table": r["table"] and r["table"]["value"], "curve": r["curve"]["value"], "curve_grade": r["curve"]["grade"]}
        if r["truth"] is not None:
            tv = r["truth"]
            e = med / tv - 1
            rec.update(err=round(e, 5), ok=abs(e) <= tl,
                       in50=d["i50"][0] <= tv <= d["i50"][1], in80=d["i80"][0] <= tv <= d["i80"][1], in95=d["i95"][0] <= tv <= d["i95"][1],
                       w80=round(d["i80"][1] / d["i80"][0] - 1, 4),
                       err_b0=None if d.get("b0") is None else round(d["b0"] / tv - 1, 5),
                       err_b1=None if d.get("b1") is None else round(d["b1"] / tv - 1, 5))
        for src in ("table", "curve"):
            v = rec[src] if src == "table" else (r["curve"]["value"] if r["curve"]["grade"] in ("strong", "fragile") else None)
            rec["err_" + src] = None if v is None else round(med / v - 1, 5)
        rows.append(rec)
    return rows


def summarize(rows, crit):
    """scheme × T × 지점(한섭 주) 요약과 criteria 판정."""
    S = {}
    srv = crit["server_main"]
    for scheme in sorted({r["scheme"] for r in rows}):
        R = [r for r in rows if r["scheme"] == scheme and r["server"] == srv and r.get("err") is not None]
        by = {}
        for T in crit["T"]:
            for q in crit["points_main"]:
                L = [r for r in R if r["T"] == T and r["pct"] == q]
                if not L:
                    continue
                ae = [abs(r["err"]) for r in L]
                b1 = [abs(r["err_b1"]) for r in L if r["err_b1"] is not None]
                b0 = [abs(r["err_b0"]) for r in L if r["err_b0"] is not None]
                by[f"T{T:g}/{q:g}"] = {
                    "n_seasons": len(L), "seasons": {r["season"]: {"err": r["err"], "in80": r["in80"], "w80": r["w80"], "grade": r["grade"]} for r in L},
                    "med_abs_err": round(statistics.median(ae), 5), "usable_ratio": round(sum(r["ok"] for r in L) / len(L), 3),
                    "cov80": round(sum(r["in80"] for r in L) / len(L), 3), "cov95": round(sum(r["in95"] for r in L) / len(L), 3),
                    "med_w80": round(statistics.median(r["w80"] for r in L), 4),
                    "med_abs_err_b1": round(statistics.median(b1), 5) if b1 else None,
                    "med_abs_err_b0": round(statistics.median(b0), 5) if b0 else None}
        # 성공 판정 (T_main, 한섭 1·3%)
        Tm = crit["T_main"]
        M = [r for r in R if r["T"] in Tm and r["pct"] in (1.0, 3.0)]
        cov = sum(r["in80"] for r in M) / len(M) if M else None
        per_T = []
        per_T_b1 = []
        for T in Tm:
            for q in (1.0, 3.0):
                L = [r for r in R if r["T"] == T and r["pct"] == q]
                if L:
                    per_T.append(statistics.median(abs(r["err"]) for r in L))
                    b = [abs(r["err_b1"]) for r in L if r["err_b1"] is not None]
                    if b:
                        per_T_b1.append(statistics.median(b))
        AB = [r for r in M if r["grade"] in ("A", "B")]
        usable = {}
        for T in Tm:
            for q in crit["points_main"]:
                k = f"T{T:g}/{q:g}"
                if k in by:
                    usable[k] = by[k]["usable_ratio"] >= 2 / 3
        S[scheme] = {
            "by": by,
            "judge": {
                "n_rows_main": len(M), "seasons": sorted({r["season"] for r in M}),
                "cov80_main": None if cov is None else round(cov, 3),
                "coverage": None if cov is None else ("보정됨" if 0.70 <= cov <= 0.90 else "구간이 좁다(과신)" if cov < 0.70 else "구간이 넓다"),
                "mean_med_abs_err_main": round(statistics.mean(per_T), 5) if per_T else None,
                "mean_med_abs_err_b1": round(statistics.mean(per_T_b1), 5) if per_T_b1 else None,
                "beat_baseline1": (statistics.mean(per_T) < statistics.mean(per_T_b1)) if per_T and per_T_b1 else None,
                "cov80_grade_AB": round(sum(r["in80"] for r in AB) / len(AB), 3) if AB else None, "n_grade_AB": len(AB),
                "usable": usable,
            }}
    return S


# ---------------- 4절 고르기 기준 판 2 (criteria_v2.json) — summarize(판 1)는 그대로 둔다 ----------------

def _q(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo = int(math.floor(k))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def window_v2(R, Ts, pcts):
    """한섭 줄 R에서 시점 Ts·지점 pcts 묶음의 지표. 「평균 시즌 중앙 |e|」 = (T, 지점) 칸마다 시즌 중앙 |e|의 평균 (판 1과 같은 셈)."""
    M = [r for r in R if r["T"] in Ts and r["pct"] in pcts]
    if not M:
        return None
    cells = collections.defaultdict(list)
    for r in M:
        cells[(r["T"], r["pct"])].append(r)
    per = [statistics.median(abs(r["err"]) for r in L) for L in cells.values()]
    perb1 = [statistics.median(abs(r["err_b1"]) for r in L if r.get("err_b1") is not None)
             for L in cells.values() if any(r.get("err_b1") is not None for r in L)]
    seas = collections.defaultdict(list)
    for r in M:
        seas[r["season"]].append(r)
    scov = {s: sum(r["in80"] for r in L) / len(L) for s, L in seas.items()}
    sbias = {s: statistics.mean(r["err"] for r in L) for s, L in seas.items()}
    AB = [r for r in M if r.get("grade") in ("A", "B")]
    cov = sum(r["in80"] for r in M) / len(M)
    usable = {f"T{T:g}/{q:g}": sum(r["ok"] for r in L) / len(L) >= 2 / 3 for (T, q), L in sorted(cells.items())}
    return {"n_rows": len(M), "n_seasons": len(seas), "seasons": sorted(seas, key=SP.snum),
            "cov80": round(cov, 3), "cov95": round(sum(r["in95"] for r in M) / len(M), 3) if all("in95" in r for r in M) else None,
            "coverage": "보정됨" if 0.70 <= cov <= 0.90 else "구간이 좁다(과신)" if cov < 0.70 else "구간이 넓다",
            "mean_med_abs_err": round(statistics.mean(per), 5),
            "mean_med_abs_err_b1": round(statistics.mean(perb1), 5) if perb1 else None,
            "beat_baseline1": (statistics.mean(per) < statistics.mean(perb1)) if perb1 else None,
            "low_ratio": round(sum(r["err"] < 0 for r in M) / len(M), 3),
            "med_w80": round(statistics.median(r["w80"] for r in M), 4),
            "cov80_grade_AB": round(sum(r["in80"] for r in AB) / len(AB), 3) if AB else None, "n_grade_AB": len(AB),
            "season_cov80_min": round(min(scov.values()), 3), "n_season_cov80_lt50": sum(v < 0.5 for v in scov.values()),
            "season_cov80": {s: round(scov[s], 3) for s in sorted(scov, key=SP.snum)},
            "season_bias": {s: round(sbias[s], 4) for s in sorted(sbias, key=SP.snum)},
            "usable": usable}


def by_volume_v2(R, Ts, pcts, k=3):
    """자료량(그 줄의 한섭 관측 사람 수 n_persons) 셋 구간별 — 경계는 이 묶음 줄의 1/3·2/3 분위 (결과와 무관한 값으로만 나눈다)."""
    M = [r for r in R if r["T"] in Ts and r["pct"] in pcts and r.get("n_persons") is not None]
    if len(M) < k:
        return None
    xs = [r["n_persons"] for r in M]
    cuts = [_q(xs, j / k) for j in range(1, k)]
    out = []
    for j in range(k):
        lo = cuts[j - 1] if j else None
        hi = cuts[j] if j < k - 1 else None
        L = [r for r in M if (lo is None or r["n_persons"] > lo) and (hi is None or r["n_persons"] <= hi)]
        if not L:
            continue
        out.append({"band": j + 1, "n_persons_gt": None if lo is None else round(lo, 1), "n_persons_le": None if hi is None else round(hi, 1),
                    "n_rows": len(L), "n_seasons": len({r["season"] for r in L}),
                    "cov80": round(sum(r["in80"] for r in L) / len(L), 3),
                    "mean_abs_err": round(statistics.mean(abs(r["err"]) for r in L), 5),
                    "low_ratio": round(sum(r["err"] < 0 for r in L) / len(L), 3)})
    return out


def summarize_v2(rows, crit2):
    """scheme마다 criteria_v2의 창(T_main·T_late·T12)과 보고 지표. 고르기에 쓰는 것은 main(1·3%)뿐이다 (criteria_v2 selection)."""
    srv = crit2.get("server_main", "KR")
    Tm, Tl, T12 = tuple(crit2["T_main"]), tuple(crit2["T_late"]), tuple(crit2.get("T_report_only", []))
    S = {}
    for scheme in sorted({r["scheme"] for r in rows}):
        R = [r for r in rows if r["scheme"] == scheme and r["server"] == srv and r.get("err") is not None]
        if not R:
            continue
        S[scheme] = {"targets": sorted({r["season"] for r in R}, key=SP.snum),
                     "main": window_v2(R, Tm, (1.0, 3.0)), "late": window_v2(R, Tl, (1.0, 3.0)),
                     "t12": window_v2(R, T12, (1.0, 3.0)) if T12 else None,
                     "p10_main": window_v2(R, Tm, (10.0,)), "p10_late": window_v2(R, Tl, (10.0,)),
                     "by_volume_main": by_volume_v2(R, Tm, (1.0, 3.0))}
    return S


def pairs_v2(rows, crit2):
    """M0′ 쌍 표 (pairs scheme): 기준 시즌 r × 맞힐 시즌 t마다 T_main·한섭 1·3% 평균 e·|e|·80% 포함. 고르기에 쓰지 않는다 (분석 4.3 M0′)."""
    Tm = tuple(crit2["T_main"])
    g = collections.defaultdict(list)
    for r in rows:
        if r["scheme"] == "pairs" and r["server"] == crit2.get("server_main", "KR") and r.get("err") is not None \
                and r["T"] in Tm and r["pct"] in (1.0, 3.0):
            g[(r["train"][0], r["season"])].append(r)
    return [{"base": b, "target": t, "n": len(L), "mean_err": round(statistics.mean(r["err"] for r in L), 5),
             "mean_abs_err": round(statistics.mean(abs(r["err"]) for r in L), 5),
             "cov80": round(sum(r["in80"] for r in L) / len(L), 3)}
            for (b, t), L in sorted(g.items(), key=lambda kv: (SP.snum(kv[0][1]), SP.snum(kv[0][0])))]


SIMPLE_RANK = {"prev": 0, "forward": 2}   # recent{k} = 1 — criteria_v2 simple_order (M0 → M2a → M2b)


def simple_rank(name):
    base = name.split("+")[0]
    if base.startswith("recent"):
        r = 1
    else:
        r = SIMPLE_RANK.get(base)
    if r is None:
        return None
    return r + 3 * name.count("+")   # 덧붙임 하나마다 뒤로 (criteria_v2: 기준 + 덧붙임 하나 → 둘 이상)


def select_v2(cands, crit2):
    """cands: [(이름, summarize_v2의 한 scheme 값)]. criteria_v2 selection 규칙 (1)~(3)과 late_guard를 그대로 적용한다.
    이름은 scheme(prev·recent5·forward), 덧붙임은 「+」로 (예: prev+m1). 순서를 모르는 이름(loso·pairs·fixed)은 후보가 아니다."""
    C = [(n, simple_rank(n), v) for n, v in cands if simple_rank(n) is not None and v.get("main")]
    if len(C) < 2:   # 후보가 하나면 고르기가 아니다 (analyze_all의 forward·loso — loso는 후보 아님)
        return None
    note = []
    tg = {n: tuple(v["targets"]) for n, _, v in C}
    if len(set(tg.values())) > 1:
        note.append("맞힐 시즌이 후보마다 다르다 — criteria_v2 folds는 같은 시즌(S21~S41)으로 견준다 (--targets)")
    ok = [c for c in C if c[2]["main"]["cov80"] >= 0.70]
    step1 = "포함 0.70 이상" if ok else "모두 0.70 미달 → 포함률이 가장 높은 것"
    if not ok:
        best_cov = max(c[2]["main"]["cov80"] for c in C)
        ok = [c for c in C if c[2]["main"]["cov80"] == best_cov]
    best = min(c[2]["main"]["mean_med_abs_err"] for c in ok)
    near = sorted([c for c in ok if c[2]["main"]["mean_med_abs_err"] <= best + 0.005], key=lambda c: (c[1], c[2]["main"]["mean_med_abs_err"]))
    pick = near[0]
    guard = []
    lt = pick[2].get("late")
    if lt and lt["cov80"] < 0.70:
        guard.append(f"T_late 80% 포함 {lt['cov80']} < 0.70")
    for n, r, v in C:
        if r < pick[1] and v.get("late") and lt and lt["mean_med_abs_err"] - v["late"]["mean_med_abs_err"] > 0.003:
            guard.append(f"T_late |e|가 더 단순한 {n}보다 {lt['mean_med_abs_err'] - v['late']['mean_med_abs_err']:.2%}p 나쁨")
    return {"pick": pick[0], "step1": step1, "kept": [c[0] for c in ok], "best_mae": best,
            "within_0_5pp": [c[0] for c in near], "late_guard": guard, "notes": note,
            "table": [{"name": n, "rank": r, "cov80": v["main"]["cov80"], "mae": v["main"]["mean_med_abs_err"],
                       "late_cov80": (v.get("late") or {}).get("cov80"), "late_mae": (v.get("late") or {}).get("mean_med_abs_err")}
                      for n, r, v in sorted(C, key=lambda c: c[1])]}


def run_meta(seasons, reg, popts, pdopts, args):
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "tools/solo_raid"], capture_output=True, text=True).stdout.strip()
    sp = SP.load_split()
    crit = jload(os.path.join(AN, "criteria.json"))
    # 실행 시각은 넣지 않는다 (같은 입력 → 같은 파일). 시각은 git 커밋으로 본다
    meta = {"tool_head": head, "tools_dirty": bool(dirty), "split_version": sp["version"], "criteria_decided": crit["decided"],
            "seasons": {s: {"role": reg[s]["role"], "tier": reg[s]["tier"],
                            "post_all_digest": stamp_digest(s)} for s in seasons},   # 도장에서 (2026-10-17 — 전에는 등록부 값)
            "answer_key_sha1": hashlib.sha1(open(os.path.join(AN, "answer_key.json"), "rb").read()).hexdigest(),
            "prior_opts": popts, "predict_opts": pdopts,
            "args": {k: v for k, v in args.items() if k not in ("work", "out", "jobs")}}   # 프로세스 수는 결과를 바꾸지 않는다
    if os.path.exists(CRIT2):
        meta["criteria_v2_sha1"] = hashlib.sha1(open(CRIT2, "rb").read()).hexdigest()
    return meta


def parse_targets(spec):
    """「S21-S41,S19」 → 이름 집합."""
    out = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out |= {f"S{n}" for n in range(SP.snum(a), SP.snum(b) + 1)}
        else:
            out.add(f"S{SP.snum(part)}")
    return out


# ---------------- 실행 (1 프로세스 또는 --jobs) ----------------

_G = {}       # 부모가 채우고 fork로 물려준다 (akey·crit·설정)
_PRIORS = {}


def _prior(path):
    if path not in _PRIORS:
        _PRIORS[path] = jload(path)
    return _PRIORS[path]


def _trunc_all(job):
    """맞힐 시즌 하나(또는 early_level 학습 시즌)의 T 절단 사본을 모두 만든다."""
    name, Ts = job
    state = FM.state_of(name)
    ex = extra_of(state)
    for T in Ts:
        BT.trunc_state(state, T, os.path.join(_G["work"], "trunc", name), ex)
    return name


def _fit_prior(job):
    train, single = job
    _, p = prior_for(train, _G["work"], _G["boot_single"] if single else _G["boot"], _G["seed"], _G["popts"], single)
    return train, single, p


def run_job(job):
    """(scheme, 학습, 맞힐 시즌, T, prior 경로) → (줄, 한 줄 출력, 실패)."""
    scheme, train, tgt, T, ppath = job
    g = _G
    prior = _prior(ppath)
    state = FM.state_of(tgt)
    st = BT.trunc_state(state, T, os.path.join(g["work"], "trunc", tgt), extra_of(state))
    season, c0, H = PR.season_info(st)
    Tc = min(T, H)
    borrow = [FM.state_of(s) for s in train]
    opts = {"grades": "all", "c_delay": 0.0, "p_update": False, "robust": "mean", "eps_wrong": PR.EPS_WRONG,
            "notes": [], "contam": (),
            "unknown": {"mode": "site", "servers": ["KR"], "borrow_state": borrow, "borrow_kind": "label",
                        "mix_file": None, "no_gate": False, "fallback": "site"}}
    opts.update(g["pdopts"])
    if g["uopts"]:
        opts["unknown"] = dict(opts["unknown"], **g["uopts"])
    if opts.get("early_level") is True:   # 3.5e: 학습 시즌의 같은 T 절단 사본 (정답 없음, 수준 보정용)
        opts["early_level"] = {"train": [[s, BT.trunc_state(FM.state_of(s), T, os.path.join(g["work"], "trunc", s), extra_of(FM.state_of(s)))]
                                         for s in train]}
    rws, cnt = PR.load_obs(st, Tc, H, c0, opts["grades"], opts["c_delay"], None, opts["unknown"])
    mu = None
    if g["lvl"] is not None:   # M4 자체 일관성 수준 갱신 (m_update.py) — predict의 _lvl 자리
        import m_update as MU
        cfg = dict(g["lvl"])
        tau = cfg.pop("tau", None) or prior["season_diff"]["tau_p"]["value"]
        mu = MU.m_posterior(prior, rws, H, tau, **cfg)
        if mu:
            opts["_lvl"] = (mu["mu"], mu["sd"])
    try:
        out = PR.predict(st, prior, Tc, opts, g["draws"], g["seed"], comps_split=False, rows=rws)
    except ValueError as e:   # predict 가장자리 (서버 관측이 모두 걸러짐 등) — 실패로 적고 계속한다
        return [], f"!! [{scheme}] {'+'.join(train)} → {tgt} T={T:g}: 예측 실패 {e}", {"scheme": scheme, "train": list(train),
                                                                                   "target": tgt, "T": T, "error": repr(e)}
    out["unknown"] = cnt.get("unknown")
    for srv, sv in out["servers"].items():
        sv["warning"] = PR.warning(Tc, srv, sv, out["unknown"], prior)
    rs = score(out, g["akey"], tgt, T, scheme, train, g["crit"])
    if mu:
        for r in rs:
            r["m_upd"] = [mu["mu"], mu["sd"], mu.get("persons"), mu.get("phi")]
    if g["slim"]:   # 한섭 줄만, 요약에 쓰는 열만
        rs = [{k: r[k] for k in SLIM_KEYS if k in r} for r in rs if r["server"] == g["crit"]["server_main"]]
    k3 = next((r for r in rs if r["server"] == "KR" and r["pct"] == 3.0), None)
    line = None
    if k3:
        line = (f"[{scheme}] {'+'.join(train) if len(train) <= 6 else train[0] + '..' + train[-1]} → {tgt} T={T:>4g}: 한섭 3% {k3['pred']:.1f} "
                f"[{k3['i80'][0]:.1f}~{k3['i80'][1]:.1f}] e={k3.get('err', float('nan')):+.2%} 등급 {k3['grade']}")
    return rs, line, None


def _pool(n):
    import multiprocessing as mp
    return mp.get_context("fork").Pool(n)   # 감시(install_guard)와 _G를 그대로 물려준다


def _each(fn, items, jobs):
    if jobs <= 1 or len(items) <= 1:
        for it in items:
            yield fn(it)
        return
    with _pool(min(jobs, len(items))) as p:
        yield from p.imap(fn, items, chunksize=1)    # 순서 유지 — 1 프로세스와 같은 줄 순서


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons")
    ap.add_argument("--scheme", default="forward,loso", help="forward·loso·prev·recent{k}·pairs (쉼표)")
    ap.add_argument("--targets", help="맞힐 시즌만 (예: S21-S41 — criteria_v2는 모든 규칙을 같은 시즌으로 견준다)")
    ap.add_argument("--T")
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--boot", type=int, default=60, help="여러 시즌 prior 붓스트랩")
    ap.add_argument("--boot-single", type=int, default=200, help="한 시즌 prior 붓스트랩 (prev·pairs, 운영 fit_prior.py 기본과 같게)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--work", default=None, help="T 절단 사본·prior 캐시 (기본: 세션 임시 폴더 또는 /tmp)")
    ap.add_argument("--out")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--jobs", type=int, default=1, help="프로세스 수 (결과는 1과 같다)")
    ap.add_argument("--slim", action="store_true", help="한섭 줄만·요약에 쓰는 열만, 들여쓰기 없이 (M0′처럼 겹이 많을 때)")
    ap.add_argument("--prior-opt", action="append", default=[], help="fit_multi 옵션 (tiers 등)")
    ap.add_argument("--fixed-prior", help="겹마다 만들지 않고 이 prior 하나로 (예: 운영 S41 prior). scheme 이름 fixed")
    ap.add_argument("--fixed-train", default=None,
                    help="--fixed-prior를 만든 시즌 (맞힐 시즌에서 뺀다, 쉼표). 기본은 prior의 season (「S38+S40」이면 둘 다) — 2026-10-17, 전에는 언제나 S41")
    ap.add_argument("--predict-opt", action="append", default=[], help="predict 옵션 (tau_p=0.1, tau_shape=0, no_season_diff=true …)")
    ap.add_argument("--unknown-opt", action="append", default=[], help="서버 미상 줄 설정 중 일부만 (mode=none, borrow_kind=em …)")
    ap.add_argument("--level-update", default=None, metavar="JSON",
                    help="M4 자체 일관성 수준 갱신 (m_update.m_posterior 설정, 예: '{}' 또는 '{\"tau\": 0.1}') — 기본 끔")
    SP.add_final_arg(ap)
    a = ap.parse_args()
    if a.final:
        raise SystemExit("xval은 개발 시즌 검증이다 — 시험(3.7)은 따로 한다")
    SP.guard_args(a)
    # 커밋된 xval_dev.json은 analyze_all의 forward·loso 결과이고 웹 빌드(viz.py)가 loso 요약을 검사한다 (2026-10-17 레포 정리).
    # --tag 없이 다른 방식으로 돌려 덮어쓰면 push 뒤 공개 웹 빌드가 멈춘다.
    if os.path.abspath(a.out or os.path.join(AN, f"xval_{a.tag}.json")) == os.path.abspath(os.path.join(AN, "xval_dev.json")) \
            and (a.fixed_prior or "loso" not in a.scheme.split(",") or a.slim or a.level_update or a.unknown_opt or a.targets):
        raise SystemExit("xval_dev.json 자리는 analyze_all의 forward·loso 결과만 쓴다 — --tag나 --out을 따로 준다")
    reg = FM.registry()
    crit = jload(os.path.join(AN, "criteria.json"))
    crit2 = jload(CRIT2) if os.path.exists(CRIT2) else None
    akey = jload(os.path.join(AN, "answer_key.json"))["rows"]
    seasons = a.seasons.split(",") if a.seasons else sorted({r["season"] for r in akey if r["truth"] is not None}, key=SP.snum)
    seasons = [s for s in seasons if reg[s]["role"] == "dev" and reg[s].get("collect") == "완료"]   # 수집 끝난 시즌만 (2026-10-17, truth_grade와 같게)
    targets = parse_targets(a.targets) if a.targets else None
    Ts = [float(x) for x in a.T.split(",")] if a.T else [float(t) for t in crit["T"]]
    work = a.work or os.path.join(os.environ.get("CLAUDE_SCRATCH", "/tmp"), "solo_raid_xval")
    os.makedirs(work, exist_ok=True)
    popts = parse_kv(a.prior_opt)
    if "tiers" in popts and isinstance(popts["tiers"], str):
        popts["tiers"] = tuple(popts["tiers"].split(","))
    pdopts = parse_kv(a.predict_opt)
    lvl = None if a.level_update is None else (json.loads(a.level_update) or {})
    _G.update(akey=akey, crit=crit, work=work, draws=a.draws, seed=a.seed, boot=a.boot, boot_single=a.boot_single,
              popts=popts, pdopts=pdopts, uopts=parse_kv(a.unknown_opt), lvl=lvl, slim=a.slim)
    plan = []   # (scheme, train, tgt, single)
    if a.fixed_prior:
        fp = jload(a.fixed_prior)
        if a.fixed_train:
            ft = tuple(a.fixed_train.split(","))
        else:
            ft = tuple(str(fp.get("season", "")).split("+"))
            if not all(re.fullmatch(r"S\d+", x) for x in ft):
                raise SystemExit(f"prior의 season 「{fp.get('season')}」에서 학습 시즌을 읽지 못했다 — --fixed-train을 준다 (그 시즌이 자기 prior로 채점되면 누수)")
        plan = [("fixed", ft, t, None) for t in seasons if t not in ft]
    else:
        for scheme in a.scheme.split(","):
            plan += [(scheme, tr, t, single_scheme(scheme)) for tr, t in folds(seasons, reg, scheme)]
    if targets is not None:
        plan = [x for x in plan if x[2] in targets]
    if not plan:
        raise SystemExit("돌릴 겹이 없다 (시즌·scheme·--targets)")
    # 1) T 절단 사본 (시즌마다 따로 — 같은 폴더를 두 프로세스가 만들지 않게)
    need = sorted({x[2] for x in plan} | ({s for x in plan for s in x[1]} if pdopts.get("early_level") is True else set()), key=SP.snum)
    for name in _each(_trunc_all, [(n, Ts) for n in need], a.jobs):
        pass
    # 2) prior (겹마다, 같은 학습 시즌이면 한 번)
    ppath = {}
    if a.fixed_prior:
        ppath = {(x[1], None): os.path.abspath(a.fixed_prior) for x in plan}
    else:
        uniq = sorted({(x[1], x[3]) for x in plan}, key=lambda k: (k[1], [SP.snum(s) for s in k[0]]))
        for train, single, p in _each(_fit_prior, uniq, a.jobs):
            ppath[(train, single)] = p
    fold_info = []
    for scheme, train, tgt, single in plan:
        prior = _prior(ppath[(train, single)])
        sd = prior["season_diff"]
        fold_info.append({"scheme": scheme, "train": list(train), "target": tgt, "prior": "single" if single else "fixed" if a.fixed_prior else "multi",
                          "form": prior["transform"]["form"], "season_levels": prior["transform"].get("season_levels"),
                          "tau": {k: sd[k]["value"] for k in ("tau_p", "tau_shape", "tau_ratio", "tau_scale0")}})
    # 3) 예측·채점
    jobs = [(scheme, train, tgt, T, ppath[(train, single)]) for scheme, train, tgt, single in plan for T in Ts]
    rows, fails = [], []
    for rs, line, fail in _each(run_job, jobs, a.jobs):
        rows += rs
        if line:
            print(line, flush=True)
        if fail:
            fails.append(fail)
    res = {"about": "솔로레이드 여러 시즌 검증 결과 (tools/solo_raid/xval.py, 할 일 3.6·4절). 개발 시즌만 — 시험 시즌 결과가 아니다",
           "run": run_meta(seasons, reg, popts, pdopts, vars(a)), "folds": fold_info,
           "summary": summarize(rows, crit)}
    if crit2:
        res["summary_v2"] = summarize_v2(rows, crit2)
        sel = select_v2(list(res["summary_v2"].items()), crit2)
        if sel:
            res["select_v2"] = sel
        if any(x[0] == "pairs" for x in plan):
            res["pairs_v2"] = pairs_v2(rows, crit2)
    if fails:
        res["failures"] = fails
    res["rows"] = rows
    out = a.out or os.path.join(AN, f"xval_{a.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(PR.jsonable(res), f, ensure_ascii=False, indent=None if a.slim else 1)
        f.write("\n")
    for scheme, sm in res["summary"].items():
        j = sm["judge"]
        print(f"\n== {scheme} (판 1): 시즌 {j['seasons']} · 80% 포함(T_main 1·3%) {j['cov80_main']} ({j['coverage']}) · "
              f"평균 시즌 중앙 |e| {j['mean_med_abs_err_main']} vs 기준선1 {j['mean_med_abs_err_b1']} · A·B 포함 {j['cov80_grade_AB']} (n {j['n_grade_AB']})")
        if a.slim:
            continue
        for k, v in sm["by"].items():
            print(f"   {k:9s} n{v['n_seasons']} |e|중앙 {v['med_abs_err']:.3%} 쓸만 {v['usable_ratio']:.2f} 포함80 {v['cov80']:.2f} "
                  f"폭80 {v['med_w80']:.1%} b1 {v['med_abs_err_b1'] if v['med_abs_err_b1'] is None else format(v['med_abs_err_b1'], '.3%')} · "
                  + " ".join(f"{s}:{d['err']:+.1%}{'' if d['in80'] else '×'}" for s, d in v["seasons"].items()))
    for scheme, v in (res.get("summary_v2") or {}).items():
        parts = []
        for w in ("main", "late", "t12"):
            x = v.get(w)
            if x:
                parts.append(f"{w} 포함 {x['cov80']:.2f} |e| {x['mean_med_abs_err']:.2%} (b1 {x['mean_med_abs_err_b1'] if x['mean_med_abs_err_b1'] is None else format(x['mean_med_abs_err_b1'], '.2%')}) "
                             f"낮게 {x['low_ratio']:.2f} 시즌최저 {x['season_cov80_min']:.2f}")
        print(f"== {scheme} (판 2, 맞힐 {len(v['targets'])}시즌): " + " | ".join(parts))
    if res.get("select_v2"):
        sv = res["select_v2"]
        print(f"== 판 2 고르기: {sv['pick']} (남김 {sv['kept']}, 0.5%p 안 {sv['within_0_5pp']})"
              + (f" · late_guard: {'; '.join(sv['late_guard'])}" if sv["late_guard"] else "") + (f" · {'; '.join(sv['notes'])}" if sv["notes"] else ""))
    print("→", os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
