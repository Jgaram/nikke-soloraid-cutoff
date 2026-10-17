"""솔로레이드 정답 표 — 시즌·서버·% 지점마다 채점에 쓸 정답과 근거 등급 (할 일 3.2, 2026-10-16, 모델 0, 요청 0).

python3 -I tools/solo_raid/truth_grade.py [--seasons S38,S39,...] [--out F.json] [--final]

기본은 등록부(registry.json)에서 몫이 개발이고 정답 점이 있는 시즌. 시험 시즌은 --final에서만 (split 감시).
기본 출력: data/solo-raid/analysis/answer_key.json (개발 시즌만 — 시험 시즌을 넣으면 --out을 따로 줘야 한다).

지점마다 두 정답을 따로 낸다.
- 집계표(B): 한섭 3·10%는 seasons.json kr_cut_*, 그 밖은 server_cutoffs.csv, S41처럼 state/truth_table.json이 있으면 그 표.
  커뮤니티 집계·추정이고 공식 확정값이 아니다.
- 정답 곡선(A+A2, A만 따로): truth_points에서 best_record·mono_viol을 뺀 점의 (ln %, 점수) 꺾은선 (explore_pct_transform.Curve, truth.py와 같은 덮인 구간).
  덮인 구간 밖이어도 지점에서 ×1.05 안에 점이 있으면 그 점으로 낸다 (near).
  - 근거: 지점 [t/2, 2t] 안 점 수, 양쪽 이웃 점, 점 하나씩 빼기(LOO) 때 값이 가장 크게 바뀐 비율 loo.
  - 등급: strong = 값이 있고 loo ≤ 1% · fragile = 값은 있으나 loo > 1% (점 하나에 기댐) · none = 값 없음.
채점 정답 truth (criteria.json): 한섭 3·10% → 집계표, 없으면 곡선 strong. 그 밖 → 곡선 strong, 없으면 집계표(서버별 표).
  집계표와 곡선(strong·fragile)이 2% 넘게 어긋나면 shaky = true (「정답 흔들림」 — 따로 낸다).
"""
import argparse
import collections
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import split as SP  # noqa: E402

_spec = importlib.util.spec_from_file_location("explore_pct_transform", os.path.join(HERE, "explore_pct_transform.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

DATA = os.path.join(ROOT, "data", "solo-raid")
OUT = os.path.join(DATA, "analysis", "answer_key.json")   # 이름이 truth*가 아니다 — 개발 시즌만 담고, split 감시는 truth* 파일을 시즌으로 가린다
CRIT = os.path.join(DATA, "analysis", "criteria.json")
SERVERS = ("KR", "JP", "GL")
NEAR = math.log(1.05)
LOO_STRONG = 0.01
SHAKY = 0.02


def curve_value(pts, t):
    """(값, 방식) — 덮인 구간 꺾은선, 아니면 ×1.05 안 가까운 점, 아니면 (None, None)."""
    if not pts:
        return None, None
    c = E.Curve(pts)
    v = c.score_at(t)
    if v is not None:
        return v, "seg"
    near = min(pts, key=lambda q: abs(math.log(q[0] / t)))
    if abs(math.log(near[0] / t)) <= NEAR:
        return near[1], "near"
    return None, None


def grade_curve(pts, t):
    v, how = curve_value(pts, t)
    lo = max((q for q in pts if q[0] <= t), default=None, key=lambda q: q[0])
    hi = min((q for q in pts if q[0] >= t), default=None, key=lambda q: q[0])
    rec = {"value": None if v is None else round(v, 2), "how": how, "n_near": sum(1 for q in pts if t / 2 <= q[0] <= 2 * t),
           "lo": lo and [lo[0], round(lo[1], 2)], "hi": hi and [hi[0], round(hi[1], 2)]}
    if v is None:
        rec.update(grade="none", loo=None)
        return rec
    worst = 0.0
    for i in range(len(pts)):
        w, _ = curve_value(pts[:i] + pts[i + 1:], t)
        worst = max(worst, 1.0 if w is None else abs(w / v - 1))
    rec["loo"] = round(worst, 4)
    rec["grade"] = "strong" if worst <= LOO_STRONG else "fragile"
    return rec


def tables(name, sp, final, state):
    """{(server, pct): (점수, 출처)}"""
    out = {}
    tt = os.path.join(state, "truth_table.json")
    if os.path.exists(tt):
        d = json.load(open(tt, encoding="utf-8"))
        for r in d.get("table", {}).get("rows", []):
            out[(r["server"], float(r["pct"]))] = (float(r["score_eok"]), "truth_table.json")
    n = SP.snum(name)
    for s in SP.load_seasons(final=final, sp=sp):
        if SP.snum(s["name"]) == n:
            for p in (3, 10):
                if s.get(f"kr_cut_{p}") is not None:
                    out.setdefault(("KR", float(p)), (float(s[f"kr_cut_{p}"]), "seasons.json"))
    for r in SP.load_server_cutoffs(final=final, sp=sp):
        if SP.snum(r["season"]) == n and r.get("score_100m"):
            try:
                out.setdefault((r["server"], float(r["percentile"])), (float(r["score_100m"]), "server_cutoffs.csv"))
            except ValueError:
                pass
    return out


def build(seasons, final=False):
    sp = SP.load_split()
    crit = json.load(open(CRIT, encoding="utf-8"))
    targets = crit["points_main"] + crit["points_aux"]
    rows = []
    for name in seasons:
        st = os.path.join(DATA, name, "state")
        tab = tables(name, sp, final, st)
        for srv in SERVERS:
            pts_all = E.final_points(st, srv, ("A", "A2"))
            pts_a = E.final_points(st, srv, ("A",))
            for t in targets:
                g = grade_curve(pts_all, t)
                ga = grade_curve(pts_a, t)
                tb = tab.get((srv, float(t)))
                rec = {"season": name, "role": SP.role(name, sp), "server": srv, "pct": t,
                       "table": tb and {"value": tb[0], "src": tb[1]}, "curve": g, "curve_A": {k: ga[k] for k in ("value", "grade", "loo")},
                       "n_points": len(pts_all)}
                # 채점 정답
                truth, basis = None, None
                if srv == "KR" and t in (3.0, 10.0) and tb:
                    truth, basis = tb[0], "table"
                elif g["grade"] == "strong":
                    truth, basis = g["value"], "curve"
                elif tb:
                    truth, basis = tb[0], "table"
                rec["truth"], rec["basis"] = truth, basis
                cv = g["value"] if g["grade"] in ("strong", "fragile") else None
                rec["table_vs_curve"] = None if not (tb and cv) else round(cv / tb[0] - 1, 4)
                rec["shaky"] = bool(rec["table_vs_curve"] is not None and abs(rec["table_vs_curve"]) > SHAKY)
                rows.append(rec)
    return {"about": "솔로레이드 정답 표 (tools/solo_raid/truth_grade.py가 만든다, 손으로 고치지 않는다). 할 일 3.2",
            "rules": {"near": 1.05, "loo_strong": LOO_STRONG, "shaky": SHAKY, "criteria": os.path.relpath(CRIT, ROOT)},
            "split_version": sp["version"], "final": final, "seasons": seasons, "rows": rows}


def default_seasons():
    # 수집이 끝난 시즌만 (2026-10-17 레포 정리): 진행 중 시즌(S42 등)은 끝나기 전 정답 점이 없어도 목록에 들어가,
    # 종료 뒤 정답 점이 생기면 채점(score_live) 전에 xval이 그 시즌을 맞혀 보게 된다. collect_done.json은 채점 뒤에 쓴다 (solo-raid-live.md 「3. 끝난 뒤」).
    reg = json.load(open(os.path.join(DATA, "registry.json"), encoding="utf-8"))
    return [r["season"] for r in reg["seasons"]
            if r["role"] == "dev" and r.get("collect") == "완료" and (r.get("data") or {}).get("has_truth_points")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons")
    ap.add_argument("--out")
    SP.add_final_arg(ap)
    a = ap.parse_args()
    SP.guard_args(a)
    seasons = a.seasons.split(",") if a.seasons else default_seasons()
    if any(SP.is_test(s) for s in seasons) and not a.out:
        raise SystemExit("시험 시즌이 섞이면 --out을 따로 준다 (기본 출력은 개발 시즌만)")
    res = build(seasons, final=a.final)
    out = a.out or OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
        f.write("\n")
    # 요약 표: 한섭 주 지점
    print(f"{'시즌':5} {'서버':3} {'%':>5} {'집계표':>8} {'곡선':>8} {'등급':8} {'loo':>6} {'근처점':>4} {'채점':>8} {'근거':6} {'표-곡선':>7}")
    for r in res["rows"]:
        if r["pct"] not in (1.0, 3.0, 10.0) and not (r["server"] == "KR" and r["pct"] in (0.5, 2.0, 5.0)):
            continue
        c, tv = r["curve"], r["table_vs_curve"]
        print(f"{r['season']:5} {r['server']:3} {r['pct']:>5g} {str(r['table'] and r['table']['value']):>8} {str(c['value']):>8} "
              f"{c['grade']:8} {str(c['loo']):>6} {c['n_near']:>4} {str(r['truth']):>8} {str(r['basis']):6} "
              f"{'' if tv is None else f'{tv:+.1%}':>7}{' 흔들림' if r['shaky'] else ''}")
    print("→", os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
