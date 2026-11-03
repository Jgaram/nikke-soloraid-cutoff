"""WP9a 진단 2 — 묶어 보기 (docs/solo-raid-method.md 「WP9a 진단」 2번, 「진단 → 후보군」, 「몫 셈의 구체 정의」).
새 적합 없음. 도우미가 낸 칸 MAP 조각을 모아 읽기만 한다. 정답·창·H·키는 score.py(SC.A.truth(), SC.window)를 import해서 같이 쓴다.

  python3 -I tools/solo_raid/lab7/wp9_diag2.py            # 셈 → r7/wp9/diag2_D1.{json,md}
  python3 -I tools/solo_raid/lab7/wp9_diag2.py --check    # 조각이 다 모였는지 (판·시즌별 칸 수만 출력, 파일 안 씀)
  --parts DIR   조각 폴더 (기본 r7/wp9/parts)      --out-dir DIR   출력 폴더 (기본 r7/wp9)

입력 (읽기만): r7/wp9/parts/map_{bind}_S{n}.json (bind = none·l·r·a, S21~S41, 파일마다 7칸 — pred만, i80·i95 null)
              r7/pred_M3dt_D1.json (NUTS 중앙) · r7/F_D1.json
출력: r7/wp9/diag2_D1.json · diag2_D1.md (숫자만)

정의
  칸 = (시즌, T), 시즌 S21~S41, T = 18·24·36·48·H−12·H−6·H (7개). 칸 안의 목표 % 1·3·10 줄을 쓴다 (0.5·2·5는 MAP 대 NUTS 참고에만).
  채점 창 main = T 18·24·36·48, end = T H−12·H−6·H (score.py window). 무리 1·3% = pct 1·3, 10% = pct 10.
  e = 중앙 ÷ 정답 − 1 (정답은 채점에만). 모든 평균은 칸 평균 (score.py와 같은 쓰임).
  판 = none(①본안 MAP)·l(②침수 묶음)·r(③컷 모양 묶음)·a(④규모 묶음) + NUTS(pred_M3dt_D1 중앙) + F.
  쓰는 칸: 네 판 조각과 NUTS·F가 1·3·10% 줄을 다 가진 (시즌, T)만. 하나라도 빠진 (시즌, T)는 모든 판·F에서 같이 뺀다 (빠진 목록은 보고). 정답 없는 (시즌, %) 줄은 칸은 두고 그 줄만 모든 판에서 같이 뺀다.
  조정 tune = S22·S26·S30·S34·S38 (score.py TUNE에서 S42 빠짐 — 1단계 없음), 나머지 rest = S21~S41 중 16시즌, 전체 all = 21시즌.
  몫 (문서 「몫 셈의 구체 정의」) = (|e|none − |e|묶음) ÷ (|e|none − |e|F), rest 초반 창(main) 칸 평균. 분모가 0 이하면 몫 없음(그렇게 적음).
    분모는 NUTS가 아니라 ① MAP (같은 계산끼리 견주려고). NUTS − F 차도 함께 적는다.
  MAP 대 NUTS = none 판 MAP 중앙 ÷ NUTS 중앙 − 1.
  판정 표 (위에서부터, 맞는 줄이 첫 후보군·다음이 둘째): D 수렴 = 아님(진단 0, 몫 −88%) · C 과신 = 아님(진단 1, 사전만 12.89 > M3 4.59) ·
    A 남은 상승 = l 묶음 1·3% 몫 > 50% · B 컷 모양 = r 묶음 1·3% 몫 > 50% 또는 10% 몫 > 50%. 아무것도 아니면 「원인 못 좁힘」.
  보조(보고만) 10% 분해: A3 = ln(pred3/ans3), B = ln(pred10/ans10) − A3 (3→10% 모양 오차) — wp9_diag0.py h3와 같은 셈, 같은 (시즌, T)에 3·10% 줄이 다 있을 때.
"""
import argparse
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

R7 = os.path.join(ROOT, "data", "solo-raid", "analysis", "r7")
PARTS = os.path.join(R7, "wp9", "parts")
OUTD = os.path.join(R7, "wp9")

SEASONS = tuple(f"S{n}" for n in range(21, 42))
BINDS = ("none", "l", "r", "a")
METHODS = ("none", "l", "r", "a", "NUTS", "F")
PCTS = (1.0, 3.0, 10.0)
GROUPS = (("1·3%", (1.0, 3.0)), ("10%", (10.0,)))
WINS = ("main", "end")
TUNE = tuple(s for s in SC.TUNE if s in SEASONS)         # 조정 5
SETS = ("all", "tune", "rest")
SETNAME = {"all": "전체 21시즌", "tune": "조정 5", "rest": "나머지 16"}
BUNDLES = (("l", "② 침수 ℓ"), ("r", "③ 컷 모양 r"), ("a", "④ 규모 a"))
LABEL = {"none": "①본안 MAP", "l": "②l 묶음", "r": "③r 묶음", "a": "④a 묶음", "NUTS": "NUTS", "F": "F"}
MAP_NUTS_TOL = 0.003   # 0.3%


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def r(x, n=6):
    return None if x is None else round(float(x), n)


def in_set(s, sp):
    return sp == "all" or (sp == "tune") == (s in TUNE)


def t_list(s):
    H = SC.A.info(s)[1]
    return [18.0, 24.0, 36.0, 48.0, H - 12, H - 6, H]


def group_of(q):
    return "10%" if q == 10.0 else ("1·3%" if q in (1.0, 3.0) else None)


# ---------------------------------------------------------------- 입력
def load_parts(parts_dir):
    """{bind: {(시즌, T, %): pred}}, {bind: {(시즌, T): info}}, 경고 목록, 코드 판 집합."""
    pred = {b: {} for b in BINDS}
    info = {b: {} for b in BINDS}
    warn = []
    heads = collections.Counter()
    present = {}
    for s in SEASONS:
        for b in BINDS:
            p = os.path.join(parts_dir, f"map_{b}_{s}.json")
            if not os.path.exists(p):
                continue
            present[(b, s)] = True
            try:
                d = jl(p)
            except Exception as ex:   # noqa: BLE001
                warn.append(f"{os.path.basename(p)} 읽기 실패: {ex}")
                continue
            if d.get("bind") not in (None, b):
                warn.append(f"{os.path.basename(p)}: 파일 머리 bind={d.get('bind')} (이름은 {b})")
            if d.get("map_only") is not True:
                warn.append(f"{os.path.basename(p)}: map_only가 True가 아님 ({d.get('map_only')})")
            heads[(d.get("made") or {}).get("tool_head")] += 1
            for c in d.get("cells", []):
                if c.get("season") != s or c.get("server", "KR") != "KR":
                    continue
                if c.get("pred"):
                    pred[b][(s, float(c["T"]), float(c["pct"]))] = float(c["pred"])
            for i in d.get("info", []):
                if i.get("season") == s:
                    info[b][(s, float(i["T"]))] = i
    return pred, info, warn, dict(heads)


def load_ref():
    """NUTS·F 중앙 {(시즌, T, %): pred} (KR, S21~S41, T는 실수)."""
    out = {}
    for nm, fn in (("NUTS", "pred_M3dt_D1.json"), ("F", "F_D1.json")):
        d = SC.load_pred(os.path.join(R7, fn))
        out[nm] = {(c["season"], c["T"], c["pct"]): float(c["pred"]) for c in d["cells"]
                   if c.get("server", "KR") == "KR" and c["season"] in SEASONS and c.get("pred")}
    return out


def collect(parts_dir):
    truth = SC.A.truth()
    pred, info, warn, heads = load_parts(parts_dir)
    ref = load_ref()
    P = dict(pred)
    P["NUTS"], P["F"] = ref["NUTS"], ref["F"]
    expected = [(s, T) for s in SEASONS for T in t_list(s)]
    # 빠진 (시즌, 판, T): 그 판의 1·3·10% 줄 중 하나라도 없음
    miss_part = []
    for s, T in expected:
        for b in BINDS:
            lack = [q for q in PCTS if (s, T, q) not in pred[b]]
            if lack:
                miss_part.append(dict(season=s, bind=b, T=T, lack_pct=lack))
    miss_ref = []
    for s, T in expected:
        for nm in ("NUTS", "F"):
            lack = [q for q in PCTS if (s, T, q) not in ref[nm]]
            if lack:
                miss_ref.append(dict(season=s, method=nm, T=T, lack_pct=lack))
    miss_truth = sorted({(s, q) for s in SEASONS for q in PCTS if (s, "KR", q) not in truth})
    bad = {(m["season"], m["T"]) for m in miss_part} | {(m["season"], m["T"]) for m in miss_ref}
    valid = [(s, T) for s, T in expected if (s, T) not in bad]
    return dict(truth=truth, P=P, info=info, warn=warn, heads=heads, expected=expected, valid=valid,
                miss_part=miss_part, miss_ref=miss_ref, miss_truth=miss_truth)


# ---------------------------------------------------------------- --check
def check(parts_dir):
    pred, info, warn, heads = load_parts(parts_dir)
    print(f"조각 폴더 {parts_dir}")
    print("시즌별 칸 수 (7칸 중 1·3·10% 줄이 다 있는 칸)  none / l / r / a")
    tot = collections.Counter()
    full = 0
    for s in SEASONS:
        Ts = t_list(s)
        cnt = {b: sum(1 for T in Ts if all((s, T, q) in pred[b] for q in PCTS)) for b in BINDS}
        for b in BINDS:
            tot[b] += cnt[b]
        full += sum(1 for T in Ts if all(cnt_ok(pred, b, s, T) for b in BINDS))
        print(f"  {s}: " + "  ".join(f"{b} {cnt[b]}/7" for b in BINDS))
    print(f"합: " + "  ".join(f"{b} {tot[b]}/147" for b in BINDS) + f"   네 판이 다 있는 칸 {full}/147")
    done = (full == 147)
    print("다 모임" if done else f"덜 모임 — 빠진 판·칸 {sum(147 - tot[b] for b in BINDS)}개")
    for w in warn:
        print("경고:", w)
    if len(heads) > 1:
        print("경고: 코드 판(made.tool_head)이 여럿:", heads)
    return 0 if done else 1


def cnt_ok(pred, b, s, T):
    return all((s, T, q) in pred[b] for q in PCTS)


# ---------------------------------------------------------------- 셈
def main_calc(parts_dir):
    D = collect(parts_dir)
    truth, P, valid = D["truth"], D["P"], D["valid"]

    def e(m, s, T, q):
        return P[m][(s, T, q)] / truth[(s, "KR", q)] - 1

    def cells(group, win, sp, season=None):
        # 정답 없는 (시즌, %) 줄은 score.py처럼 모든 판에서 같이 빠진다 (칸 전체를 빼지는 않는다)
        qs = dict(GROUPS)[group]
        return [(s, T, q) for (s, T) in valid for q in qs if (s, "KR", q) in truth and SC.window(s, T) == win
                and in_set(s, sp) and (season is None or s == season)]

    def mabs(m, cs):
        return mean(abs(e(m, *c)) for c in cs)

    # ---- 1) MAP 대 NUTS (none 판)
    mvn = {}
    for sp in ("tune", "all"):
        for gname, qs in GROUPS:
            for win in WINS + ("both",):
                cs = [(s, T, q) for (s, T) in valid for q in qs
                      if (win == "both" or SC.window(s, T) == win) and in_set(s, sp)]
                rel = [(c, P["none"][c] / P["NUTS"][c] - 1) for c in cs]
                if not rel:
                    mvn[f"{sp}|{gname}|{win}"] = dict(n=0)
                    continue
                cmax, vmax = max(rel, key=lambda x: abs(x[1]))
                mvn[f"{sp}|{gname}|{win}"] = dict(
                    n=len(rel), mean_abs=r(mean(abs(v) for _, v in rel)), mean_signed=r(mean(v for _, v in rel)),
                    max_abs=r(abs(vmax)), max_cell=f"{cmax[0]} T{cmax[1]:g} {cmax[2]:g}%", max_signed=r(vmax),
                    n_over=sum(1 for _, v in rel if abs(v) > MAP_NUTS_TOL))
    over_cells = []
    allq = sorted({c[2] for c in P["none"]})
    for (s, T) in valid:
        for q in allq:
            k = (s, T, q)
            if k in P["none"] and k in P["NUTS"]:
                v = P["none"][k] / P["NUTS"][k] - 1
                if abs(v) > MAP_NUTS_TOL:
                    over_cells.append(dict(season=s, T=T, pct=q, rel=r(v), tune=s in TUNE, win=SC.window(s, T), in_scored_pcts=q in PCTS))
    over_cells.sort(key=lambda x: -abs(x["rel"]))
    n_pairs_all_pcts = sum(1 for (s, T) in valid for q in allq if (s, T, q) in P["none"] and (s, T, q) in P["NUTS"])

    # ---- 2) 판별 |e|
    abs_e = {}
    for gname, _ in GROUPS:
        for win in WINS:
            for sp in SETS:
                cs = cells(gname, win, sp)
                abs_e[f"{gname}|{win}|{sp}"] = dict(n=len(cs), **{m: r(mabs(m, cs)) for m in METHODS})

    # ---- 3) 몫
    def quota(bind, gname, win, sp):
        cs = cells(gname, win, sp)
        if not cs:
            return dict(n=0, quota=None, note="칸 없음")
        n0, nb, nf = mabs("none", cs), mabs(bind, cs), mabs("F", cs)
        num, den = n0 - nb, n0 - nf
        out = dict(n=len(cs), abs_none=r(n0), abs_bind=r(nb), abs_F=r(nf), num=r(num), den=r(den))
        if den <= 0:
            out.update(quota=None, note=f"분모 ≤ 0 (|e|none − |e|F = {den * 100:.2f}%p) — 몫 없음")
        else:
            out.update(quota=r(num / den), note="")
        return out

    quotas = {f"{b}|{g}|{w}|{sp}": quota(b, g, w, sp) for b, _ in BUNDLES for g, _ in GROUPS for w in WINS for sp in SETS}
    gap_none_F = {}
    gap_nuts_F = {}
    for gname, _ in GROUPS:
        for sp in SETS:
            cs = cells(gname, "main", sp)
            gap_none_F[f"{gname}|{sp}"] = r(mabs("none", cs) - mabs("F", cs)) if cs else None
            gap_nuts_F[f"{gname}|{sp}"] = r(mabs("NUTS", cs) - mabs("F", cs)) if cs else None

    # ---- 4) 판정
    def qv(b, g):
        return quotas[f"{b}|{g}|main|rest"]["quota"]

    ql13, qr13, qr10 = qv("l", "1·3%"), qv("r", "1·3%"), qv("r", "10%")
    gt = lambda x: x is not None and x > 0.5   # noqa: E731
    rows = [
        dict(line="D 수렴 먼저", hit=False, why="아님 — 진단 0: 수렴 못 한 칸을 빼면 몫 −88%"),
        dict(line="C 과신 줄이기", hit=False, why="아님 — 진단 1: 사전만 12.89 > M3 4.59 (초반 창 |e|)"),
        dict(line="A 남은 상승 (② l 묶음)", hit=gt(ql13), why=f"l 묶음 1·3% 몫 {pq(ql13)} (기준 > 50%)"),
        dict(line="B 컷 모양 (③ r 묶음)", hit=gt(qr13) or gt(qr10),
             why=f"r 묶음 1·3% 몫 {pq(qr13)} · 10% 몫 {pq(qr10)} (기준: 둘 중 하나 > 50%)"),
    ]
    hits = [x["line"] for x in rows if x["hit"]]
    verdict = dict(rows=rows, hits=hits,
                   first=hits[0] if hits else None, second=hits[1] if len(hits) > 1 else None,
                   text=("원인 못 좁힘 — 멈추고 사용자에게 보고" if not hits
                         else f"첫 후보군 {hits[0]}" + (f", 둘째 {hits[1]}" if len(hits) > 1 else "")))

    # ---- 5) 보조
    aux = {}
    # T별
    byT = {}
    for sp in ("all", "rest"):
        for gname, qs in GROUPS:
            g = collections.defaultdict(list)
            for (s, T) in valid:
                if in_set(s, sp):
                    for q in qs:
                        if (s, "KR", q) in truth:
                            g[SC.tkey(s, T)].append((s, T, q))
            order = ["18", "24", "36", "48", "H-12", "H-6", "H"]
            byT[f"{gname}|{sp}"] = {k: dict(n=len(g[k]), **{m: r(mabs(m, g[k])) for m in METHODS}) for k in order if k in g}
    # 시즌별 초반 창
    by_season = {}
    for gname, qs in GROUPS:
        t_ = {}
        for s in SEASONS:
            cs = cells(gname, "main", "all", season=s)
            if cs:
                t_[s] = dict(n=len(cs), tune=s in TUNE, **{m: r(mabs(m, cs)) for m in METHODS})
        by_season[gname] = t_
    # 10% 분해
    h3 = {}
    for sp in SETS:
        ent = {}
        sts = [(s, T) for (s, T) in valid if SC.window(s, T) == "main" and in_set(s, sp)
               and (s, "KR", 3.0) in truth and (s, "KR", 10.0) in truth]
        for m in METHODS:
            A_, B_, T_ = [], [], []
            for (s, T) in sts:
                a3 = math.log(P[m][(s, T, 3.0)] / truth[(s, "KR", 3.0)])
                tot = math.log(P[m][(s, T, 10.0)] / truth[(s, "KR", 10.0)])
                A_.append(a3)
                B_.append(tot - a3)
                T_.append(tot)
            if not sts:
                continue
            rms = lambda v: math.sqrt(mean(t * t for t in v))   # noqa: E731
            sst = sum(t * t for t in T_)
            ent[m] = dict(n=len(sts), rms_A=r(rms(A_)), rms_B=r(rms(B_)), mean_B=r(mean(B_)), rms_tot=r(rms(T_)),
                          share_B=r(sum(b * b for b in B_) / sst) if sst else None)
        h3[sp] = ent
    # map_ok·conv가 아닌 칸 (조각 info 전체 — 뺀 칸 포함)
    bad_cells = []
    for b in BINDS:
        for (s, T), i in sorted(D["info"][b].items()):
            if i.get("map_ok") is False or i.get("conv") is False:
                bad_cells.append(dict(bind=b, season=s, T=T, map_ok=i.get("map_ok"), conv=i.get("conv"),
                                      map_min_eig=r(i.get("map_min_eig"), 4), used=(s, T) in set(valid)))
    n_info = sum(len(D["info"][b]) for b in BINDS)

    res = dict(
        about="WP9a 진단 2 — 묶어 보기 (새 적합 없음, 칸 MAP 조각 읽기). 정의는 wp9_diag2.py 머리 참조. 비율·e는 소수 (0.01 = 1%).",
        inputs=dict(parts_dir=os.path.relpath(parts_dir, ROOT) if parts_dir.startswith(ROOT) else parts_dir,
                    n_cells_expected=len(D["expected"]), n_cells_used=len(valid), tool_heads=D["heads"], warnings=D["warn"],
                    seasons_used=sorted({s for s, _ in valid}), tune=list(TUNE),
                    n_rest_seasons=len({s for s, _ in valid if s not in TUNE}), n_tune_seasons=len({s for s, _ in valid if s in TUNE})),
        missing=dict(parts=D["miss_part"], ref=D["miss_ref"], truth=[f"{s} {q:g}%" for s, q in D["miss_truth"]],  # 정답 없는 줄: 칸은 두고 그 줄만 뺌
                     dropped_cells=sorted(f"{s} T{T:g}" for s, T in D["expected"] if (s, T) not in set(valid))),
        map_vs_nuts=dict(by=mvn, n_pairs_all_pcts=n_pairs_all_pcts, over_0p3pct_cells=over_cells, tol=MAP_NUTS_TOL),
        abs_e=abs_e, quota=quotas, gap_none_minus_F_main=gap_none_F, gap_NUTS_minus_F_main=gap_nuts_F,
        verdict=verdict,
        aux=dict(by_T=byT, by_season_main=by_season, h3_main=h3, not_ok_cells=bad_cells, n_info_cells=n_info))
    return res


def pq(x):
    return "몫 없음(분모 ≤ 0 또는 칸 없음)" if x is None else f"{x * 100:.1f}%"


# ---------------------------------------------------------------- md
def P1(x, d=2):
    return "-" if x is None else f"{x * 100:.{d}f}"


def tbl(head, rows_):
    esc = lambda v: str(v).replace("|", "\\|")   # noqa: E731
    o = ["| " + " | ".join(esc(h) for h in head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for rr in rows_:
        o.append("| " + " | ".join(esc(c) for c in rr) + " |")
    return "\n".join(o)


def write_md(res, path):
    inp, mv, ae, qt, vd, aux = res["inputs"], res["map_vs_nuts"], res["abs_e"], res["quota"], res["verdict"], res["aux"]
    L = []
    nu, ne = inp["n_cells_used"], inp["n_cells_expected"]
    t13, t10 = mv["by"].get("tune|1·3%|both", {}), mv["by"].get("tune|10%|both", {})
    a13, a10 = mv["by"].get("all|1·3%|both", {}), mv["by"].get("all|10%|both", {})
    L.append("# WP9a 진단 2 — 묶어 보기 (칸 MAP, S21~S41, D1)\n")
    L.append("## 요약\n")
    L.append(f"1. 자료: 칸(시즌, T) {nu}/{ne}개 사용" + (" (조각 다 모임)" if nu == ne else f" — 빠져 뺀 칸 {ne - nu}개: {', '.join(res['missing']['dropped_cells'][:12])}{' …' if ne - nu > 12 else ''}")
             + f". 시즌 {len(inp['seasons_used'])}개 (조정 {inp['n_tune_seasons']} · 나머지 {inp['n_rest_seasons']}).")
    if t13.get("n"):
        L.append(f"2. MAP 대 NUTS (none 판): 조정 시즌 평균 |차| 1·3% {P1(t13['mean_abs'])}% (최대 {P1(t13['max_abs'])}%, {t13['max_cell']}) · 10% {P1(t10.get('mean_abs'))}% (최대 {P1(t10.get('max_abs'))}%); "
                 f"0.3% 넘는 칸 조정 {t13['n_over'] + t10.get('n_over', 0)}/{t13['n'] + t10.get('n', 0)} · 전체 {a13.get('n_over', 0) + a10.get('n_over', 0)}/{a13.get('n', 0) + a10.get('n', 0)}.")
    else:
        L.append("2. MAP 대 NUTS: 조정 시즌 칸 없음.")
    qs_ = []
    for b, nm in BUNDLES:
        qs_.append(f"{b} {pq(qt[f'{b}|1·3%|main|rest']['quota'])} / {pq(qt[f'{b}|10%|main|rest']['quota'])}")
    den13 = qt["l|1·3%|main|rest"].get("den")
    den10 = qt["l|10%|main|rest"].get("den")
    L.append(f"3. 몫 (나머지 16시즌 초반 창, 1·3% / 10%): " + " · ".join(qs_)
             + f". 분모 |e|none − |e|F = 1·3% {P1(den13)}%p · 10% {P1(den10)}%p (NUTS − F: {P1(res['gap_NUTS_minus_F_main']['1·3%|rest'])} · {P1(res['gap_NUTS_minus_F_main']['10%|rest'])}%p).")
    L.append(f"4. **판정: {vd['text']}.** 맞은 줄 {', '.join(vd['hits']) if vd['hits'] else '없음'}.")
    k = ae["1·3%|main|rest"]
    k10 = ae["10%|main|rest"]
    L.append("5. 판별 |e| (rest 초반, %): 1·3% " + " ".join(f"{LABEL[m]} {P1(k[m])}" for m in METHODS)
             + " / 10% " + " ".join(f"{LABEL[m]} {P1(k10[m])}" for m in METHODS) + ".\n")

    L.append("## 정의\n")
    L.append("칸 = (시즌, T), 시즌 S21~S41, T = 18·24·36·48·H−12·H−6·H. 칸 안의 목표 % 줄 1·3·10을 쓴다. 창 main = T 18·24·36·48, end = H−12·H−6·H. 무리 1·3% = pct 1·3, 10% = pct 10. "
             "e = 중앙 ÷ 정답 − 1. 평균은 칸(시즌, T, %) 평균. 판 = ①none 본안 MAP · ②l 침수 묶음 · ③r 컷 모양 묶음 · ④a 규모 묶음 · NUTS(`pred_M3dt_D1` 중앙) · F. "
             "조정 5 = S22·S26·S30·S34·S38, 나머지 16 = S21~S41 중 그 밖, 전체 21. 쓴 칸은 네 판 조각·NUTS·F가 1·3·10% 줄을 다 가진 (시즌, T)뿐 — 하나라도 빠진 칸은 모든 판·F에서 같이 뺀다 (정답 없는 (시즌, %) 줄은 그 줄만 모든 판에서 같이 뺌). "
             "**몫** = (|e|none − |e|묶음) ÷ (|e|none − |e|F), 나머지 16시즌 초반 창 칸 평균 (문서 「몫 셈의 구체 정의」 — 분모는 NUTS가 아니라 ① MAP). 분모 ≤ 0이면 몫 없음. "
             "판정: D·C는 진단 0·1에서 이미 「아님」, A = l 묶음 1·3% 몫 > 50%, B = r 묶음 1·3% 몫 > 50% 또는 10% 몫 > 50% (④ a 묶음은 보고만). 코드 `tools/solo_raid/lab7/wp9_diag2.py`, 수치 전부 `diag2_D1.json`.\n")
    L.append(f"조각 코드 판(made.tool_head): {inp['tool_heads']}" + (f" — **여럿이면 판이 섞임**" if len(inp["tool_heads"]) > 1 else "")
             + (f". 경고: {'; '.join(inp['warnings'])}" if inp["warnings"] else "") + "\n")

    L.append("## 0. 빠진 조각\n")
    ms = res["missing"]
    if not ms["parts"] and not ms["ref"]:
        L.append("없음 — 147칸 모두 갖춤.\n")
        if ms["truth"]:
            L.append("정답 없는 (시즌, %) — 그 줄만 모든 판에서 같이 뺌: " + ", ".join(ms["truth"]) + "\n")
    else:
        L.append(f"뺀 칸 {len(ms['dropped_cells'])}개: {', '.join(ms['dropped_cells'])}\n")
        grp = collections.defaultdict(list)
        for m in ms["parts"]:
            grp[(m["season"], m["bind"])].append(f"T{m['T']:g}")
        if grp:
            L.append("빠진 (시즌, 판, T):\n")
            L.append(tbl(["시즌", "판", "빠진 T"], [[s, b, ", ".join(v)] for (s, b), v in sorted(grp.items())]) + "\n")
        if ms["ref"]:
            L.append("NUTS·F에 없는 칸: " + ", ".join(f"{m['season']} {m['method']} T{m['T']:g}" for m in ms["ref"]) + "\n")
        if ms["truth"]:
            L.append("정답 없는 (시즌, %) — 그 줄만 모든 판에서 같이 뺌: " + ", ".join(ms["truth"]) + "\n")

    L.append("## 1. MAP 대 NUTS (none 판 MAP 중앙 ÷ NUTS 중앙 − 1)\n")
    rows_ = []
    for sp in ("tune", "all"):
        for g, _ in GROUPS:
            for w in ("main", "end", "both"):
                x = mv["by"].get(f"{sp}|{g}|{w}", {})
                if not x.get("n"):
                    continue
                rows_.append([SETNAME[sp], g, {"both": "전체 창"}.get(w, w), x["n"], P1(x["mean_abs"]), P1(x["mean_signed"]), P1(x["max_abs"]), x["max_cell"], x["n_over"]])
    L.append(tbl(["집합", "무리", "창", "칸", "평균 |차|%", "평균 차%", "최대 |차|%", "최대 칸", "0.3% 넘는 칸"], rows_))
    oc = mv["over_0p3pct_cells"]
    L.append(f"\n0.3%를 넘는 칸 (모든 % 포함, 쌍 {mv['n_pairs_all_pcts']}개 중 {len(oc)}개; 큰 순 최대 25):\n")
    if oc:
        L.append(tbl(["시즌", "T", "%", "창", "조정", "차%", "채점 %"], [[x["season"], f"{x['T']:g}", f"{x['pct']:g}", x["win"] or "-", "o" if x["tune"] else "", P1(x["rel"]), "o" if x["in_scored_pcts"] else ""] for x in oc[:25]]))
    else:
        L.append("없음.")
    L.append("")

    L.append("## 2. 판별 |e| (칸 평균, %)\n")
    rows_ = []
    for g, _ in GROUPS:
        for w in WINS:
            for sp in SETS:
                x = ae[f"{g}|{w}|{sp}"]
                rows_.append([g, w, SETNAME[sp], x["n"]] + [P1(x[m]) for m in METHODS])
    L.append(tbl(["무리", "창", "집합", "칸"] + [LABEL[m] for m in METHODS], rows_) + "\n")

    L.append("## 3. 몫\n")
    L.append(f"분모 |e|none − |e|F (초반 창, %p): " + " · ".join(f"{g} {SETNAME[sp]} {P1(res['gap_none_minus_F_main'][f'{g}|{sp}'])}" for g, _ in GROUPS for sp in SETS)
             + "\n\nNUTS − F (같은 칸, %p): " + " · ".join(f"{g} {SETNAME[sp]} {P1(res['gap_NUTS_minus_F_main'][f'{g}|{sp}'])}" for g, _ in GROUPS for sp in SETS) + "\n")
    rows_ = []
    for b, nm in BUNDLES:
        for g, _ in GROUPS:
            for w in WINS:
                for sp in ("rest", "tune", "all"):
                    x = qt[f"{b}|{g}|{w}|{sp}"]
                    if not x.get("n"):
                        continue
                    rows_.append([f"{b} {nm}", g, w, SETNAME[sp], x["n"], P1(x["abs_none"]), P1(x["abs_bind"]), P1(x["abs_F"]), P1(x["den"]),
                                  pq(x["quota"]) if x["quota"] is not None else x["note"]])
    L.append(tbl(["묶음", "무리", "창", "집합", "칸", "|e|none%", "|e|묶음%", "|e|F%", "분모%p", "몫"], rows_))
    L.append("\n(판정에 쓰는 줄은 창 main · 집합 나머지 16. 나머지는 참고.)\n")

    L.append("## 4. 판정 표 적용\n")
    L.append(tbl(["줄", "맞음", "근거"], [[x["line"], "예" if x["hit"] else "아니오", x["why"]] for x in vd["rows"]]))
    L.append(f"\n**{vd['text']}.**\n")

    L.append("## 5. 보조 (보고만)\n")
    L.append("### T별 판별 |e| (%)\n")
    for key in ("1·3%|all", "1·3%|rest", "10%|all", "10%|rest"):
        g, sp = key.split("|")
        L.append(f"{g} · {SETNAME[sp]}:\n")
        L.append(tbl(["T", "칸"] + [LABEL[m] for m in METHODS], [[k, x["n"]] + [P1(x[m]) for m in METHODS] for k, x in aux["by_T"][key].items()]) + "\n")
    L.append("### 시즌별 초반 창 |e| (%)\n")
    for g, _ in GROUPS:
        L.append(f"{g}:\n")
        L.append(tbl(["시즌", "조정", "칸"] + [LABEL[m] for m in METHODS], [[s, "o" if x["tune"] else "", x["n"]] + [P1(x[m]) for m in METHODS] for s, x in aux["by_season_main"][g].items()]) + "\n")
    L.append("### 10% 오차 = 3% 오차 A + 3→10% 모양 오차 B (ln, 초반 창, % 단위 숫자 ×100)\n")
    for sp in SETS:
        ent = aux["h3_main"].get(sp, {})
        if not ent:
            continue
        L.append(f"{SETNAME[sp]}:\n")
        L.append(tbl(["판", "칸(시즌,T)", "rms A", "rms B", "평균 B", "rms 10%전체", "B의 제곱합 몫%"],
                     [[LABEL[m], x["n"], P1(x["rms_A"]), P1(x["rms_B"]), P1(x["mean_B"]), P1(x["rms_tot"]), P1(x["share_B"], 0)] for m, x in ent.items()]) + "\n")
    L.append("### map_ok·conv가 아닌 칸 (조각 info 전체)\n")
    bc = aux["not_ok_cells"]
    L.append(f"{len(bc)}/{aux['n_info_cells']}칸.\n")
    if bc:
        L.append(tbl(["판", "시즌", "T", "map_ok", "conv", "최소 고유값", "분석에 씀"], [[x["bind"], x["season"], f"{x['T']:g}", x["map_ok"], x["conv"], x["map_min_eig"], "o" if x["used"] else "x"] for x in bc]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="조각이 다 모였는지만 (판·시즌별 칸 수)")
    ap.add_argument("--parts", default=PARTS)
    ap.add_argument("--out-dir", default=OUTD)
    a = ap.parse_args()
    if a.check:
        sys.exit(check(a.parts))
    res = main_calc(a.parts)
    if not res["inputs"]["n_cells_used"]:
        print("쓸 수 있는 칸이 없다 (--check로 조각 확인).")
        sys.exit(2)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "diag2_D1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    write_md(res, os.path.join(a.out_dir, "diag2_D1.md"))
    print(f"칸 {res['inputs']['n_cells_used']}/{res['inputs']['n_cells_expected']} · {res['verdict']['text']} → {a.out_dir}")


if __name__ == "__main__":
    main()
