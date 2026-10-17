"""할 일 4.2 — 시즌 차이의 구조 (서술, 예측 성능을 주장하지 않는다). 2026-10-17 4.2 세션.

특징 목록·무리 경계·판정 조건은 계산 전에 문서에 고정했다 (docs/solo-raid-analysis.md 4.2 「사전 고정」, 커밋 f2703e6).

A 특징표 (끝난 뒤 값, 한섭) · B 무리 (형식·자료량, 경계는 자료와 무관) · C 초반 특징 (T24·T48 절단 사본, 정답 없음).
주 목표 ln m_s (prior_dev season_levels), 보조 ln(10%/3%)·ln(1%/3%) (answer_key 한섭 truth).
잣대: 시즌 하나씩 빼기(LOSO) 직선의 잔차 rms, 설명 몫 1 − (rms/rms_null)². 4.3 후보 조건: T24·T48 모두 m̂ 직선보다 rms 5% 넘게 작음.

돌리는 법 (레포에서 직접 돌리지 않는다 — lab4 README):
  S=$CLAUDE_SCRATCH/lab4; git clone -q . $S/wk
  (T24·T48 절단 사본 $S/xw/trunc/{시즌}/trunc_T{T}/state — mk_trunc.py의 TS를 24,48로)
  python3 tools/solo_raid/lab4/e_42.py [--out F.json]
"""
import argparse
import json
import math
import os
import random
import statistics
import sys

SCR = os.environ.get("LAB4_SCR") or os.path.join(os.environ.get("CLAUDE_SCRATCH", "/tmp"), "lab4")
WK = os.path.join(SCR, "wk")
XW = os.path.join(SCR, "xw")
sys.path.insert(0, os.path.join(WK, "tools/solo_raid"))
os.chdir(WK)
import split as SP  # noqa: E402

SP.install_guard()
import predict as PR  # noqa: E402
import fit_prior_multi as FM  # noqa: E402
import explore_early_level as EL  # noqa: E402

SEAS = [f"S{n}" for n in range(19, 42)]
TS = (24, 48)
SPECIAL = {"S26": "기간 변경", "S28": "재개", "S38": "재오픈"}   # A6·B1·⑤ (공지·등록부 형식, 결과와 무관)
MOCK = {"S41"}                                                   # 모의전 사전 개방 (개발 시즌에 하나)
AN = os.path.join(WK, "data", "solo-raid", "analysis")
BOOT_M = 200
SEED = 1
MARGIN = 0.95   # 4.3 후보 조건: m̂ 직선 rms의 95% 미만


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def trunc(s, T):
    return os.path.join(XW, "trunc", s, f"trunc_T{float(T):g}", "state")


# ---------------- 통계 도구 ----------------
def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 4:
        return None
    rx, ry = ranks(x), ranks(y)
    if statistics.pstdev(rx) == 0 or statistics.pstdev(ry) == 0:
        return None
    return statistics.correlation(rx, ry)


def ols(X, y):
    """X: 행마다 [1, x1, ...] → 계수 (정규방정식)."""
    p = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(p)] for i in range(p)]
    b = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(p)]
    return EL.solve(A, b)


def rms(v):
    return math.sqrt(sum(e * e for e in v) / len(v)) if v else None


def loso(rows, kind="lin", k=3):
    """rows = [(시즌, [x...], y)] → 뺀 시즌 잔차 목록 (시즌 순서 그대로). kind lin = 직선(특징 0개면 평균), knn = 가까운 k시즌 평균."""
    res = []
    for i, (s, x, y) in enumerate(rows):
        tr = [r for j, r in enumerate(rows) if j != i]
        if kind == "lin":
            X = [[1.0] + r[1] for r in tr]
            c = ols(X, [r[2] for r in tr])
            if c is None:
                return None
            pred = c[0] + sum(ci * xi for ci, xi in zip(c[1:], x))
        else:
            d = len(x)
            mu = [statistics.mean(r[1][q] for r in tr) for q in range(d)]
            sd = [statistics.pstdev([r[1][q] for r in tr]) or 1.0 for q in range(d)]
            z = lambda v: [(v[q] - mu[q]) / sd[q] for q in range(d)]
            zi = z(x)
            near = sorted(tr, key=lambda r: sum((a - b) ** 2 for a, b in zip(z(r[1]), zi)))[:k]
            pred = statistics.mean(r[2] for r in near)
        res.append(y - pred)
    return res


def boot_ratio(ra, rb, n=2000, seed=SEED):
    """같은 시즌 잔차 두 벌 → rms(a)/rms(b)의 시즌 붓스트랩 10·50·90% (안정성 서술용)."""
    rng = random.Random(seed)
    N = len(ra)
    out = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        b = rms([rb[i] for i in idx])
        if b and b > 0:
            out.append(rms([ra[i] for i in idx]) / b)
    out.sort()
    q = lambda p: round(out[int(p * (len(out) - 1))], 3)
    return [q(0.1), q(0.5), q(0.9)]


def evaluate(feat, target, keys, kind="lin"):
    """feat: {시즌: {이름: 값}}, target: {시즌: 값}, keys: 쓸 특징 이름들 → 요약."""
    S = [s for s in SEAS if s in target and target[s] is not None and all(feat.get(s, {}).get(k) is not None for k in keys)]
    if len(S) < 6:
        return {"n": len(S), "seasons": S}
    rows = [(s, [feat[s][k] for k in keys], target[s]) for s in S]
    null = loso([(s, [], y) for s, _, y in rows])
    r = loso(rows, kind)
    out = {"n": len(S), "seasons": S, "rms_null": round(rms(null), 4), "rms": round(rms(r), 4),
           "share": round(1 - rms(r) ** 2 / rms(null) ** 2, 3), "resid": {s: round(e, 4) for s, e in zip(S, r)}}
    if len(keys) == 1 and kind == "lin":
        out["spearman"] = round(spearman([x[0] for _, x, _ in rows], [y for _, _, y in rows]), 3)
    return out


# ---------------- A 특징표 ----------------
def table_A(pr):
    reg = FM.registry()
    ak = json.load(open(os.path.join(AN, "answer_key.json"), encoding="utf-8"))["rows"]
    tv = {(r["season"], r["server"], r["pct"]): r.get("truth") for r in ak}
    lv = pr["transform"]["season_levels"]
    th, form = pr["transform"]["theta"], pr["transform"]["form"]
    shape, shape_summ = EL.shape_check(SEAS, form, th, BOOT_M // 4, SEED)
    A = {}
    for s in SEAS:
        g = lambda srv, p: tv.get((s, srv, p))
        k1, k3, k10, j3, g3 = g("KR", 1.0), g("KR", 3.0), g("KR", 10.0), g("JP", 3.0), g("GL", 3.0)
        d = reg[s]["data"]
        A[s] = {
            "n": SP.snum(s),
            "ln_ms": round(math.log(lv[s]), 4) if s in lv else None,
            "ln_early_late": shape.get(s, {}).get("ln_ratio"),
            "ln_early_late_se": shape.get(s, {}).get("se"),
            "ln_1_3": round(math.log(k1 / k3), 4) if k1 and k3 else None,
            "ln_10_3": round(math.log(k10 / k3), 4) if k10 and k3 else None,
            "ln_jp_kr3": round(math.log(j3 / k3), 4) if j3 and k3 else None,
            "ln_gl_kr3": round(math.log(g3 / k3), 4) if g3 and k3 else None,
            "ln_kr3": round(math.log(k3), 4) if k3 else None,
            "special": 1 if s in SPECIAL else 0,
            "mock": 1 if s in MOCK else 0,
            "ln_kr_rows": round(math.log(d["use_by_server"].get("KR", 0)), 4) if d["use_by_server"].get("KR") else None,
            "ln_kr_label_persons": round(math.log(d["label_persons_pre_end"].get("KR", 0)), 4) if d["label_persons_pre_end"].get("KR") else None,
        }
    return A, shape_summ


A_KEYS = ["ln_ms", "ln_early_late", "ln_1_3", "ln_10_3", "ln_jp_kr3", "ln_gl_kr3", "ln_kr3", "special", "ln_kr_rows", "ln_kr_label_persons", "n"]


def corr_A(A):
    out = {}
    for i, a in enumerate(A_KEYS):
        for b in A_KEYS[i + 1:]:
            S = [s for s in SEAS if A[s].get(a) is not None and A[s].get(b) is not None]
            rho = spearman([A[s][a] for s in S], [A[s][b] for s in S])
            out[f"{a}~{b}"] = {"n": len(S), "rho": None if rho is None else round(rho, 3)}
    return out


# ---------------- B 무리 ----------------
def groups_B(A, targets):
    rows_kr = sorted(SEAS, key=lambda s: A[s]["ln_kr_rows"])
    med = statistics.median(A[s]["ln_kr_rows"] for s in SEAS)
    B2 = {s: 1 if A[s]["ln_kr_rows"] >= med else 0 for s in SEAS}
    out = {"B2_rule": f"기본 집합 한섭 줄 ≥ 23시즌 중앙값 {round(math.exp(med))}줄", "B2_hi": [s for s in SEAS if B2[s]], "B1_special": sorted(SPECIAL)}
    for gname, gm in (("B1", {s: A[s]["special"] for s in SEAS}), ("B2", B2)):
        out[gname] = {}
        for tname, tg in targets.items():
            feat = {s: {"g": gm[s]} for s in SEAS}
            ev = evaluate(feat, tg, ["g"])
            S = ev["seasons"]
            m1 = [tg[s] for s in S if gm[s]]
            m0 = [tg[s] for s in S if not gm[s]]
            ev["mean_in"] = round(statistics.mean(m1), 4) if m1 else None
            ev["mean_out"] = round(statistics.mean(m0), 4) if m0 else None
            ev["n_in"], ev["n_out"] = len(m1), len(m0)
            out[gname][tname] = ev
    # B3 모의전: S41이 다른 시즌 분포의 몇 번째인가 (판정 안 함)
    b3 = {}
    for k in ("ln_ms", "ln_early_late", "ln_1_3", "ln_10_3", "ln_jp_kr3", "ln_kr3"):
        v = A["S41"].get(k)
        oth = [A[s][k] for s in SEAS if s != "S41" and A[s].get(k) is not None]
        if v is not None and oth:
            b3[k] = {"S41": v, "below": sum(1 for o in oth if o < v), "n_other": len(oth),
                     "min": min(oth), "max": max(oth)}
    out["B3_S41"] = b3
    return out


# ---------------- C 초반 특징 ----------------
def early_rows(state, T, labeled_only):
    """explore_early_level.early_m과 같은 줄 고르기 (시점마다 하나)."""
    core = jl(os.path.join(state, "core.jsonl"))
    by = {}
    for c in core:
        if not (c.get("use") and c.get("src") == "photo" and c.get("pct") and c.get("score") and c.get("h") is not None):
            continue
        if c["t_grade"] not in ("A", "B") or c["h"] < 1 or c["h"] > T:
            continue
        if labeled_only and c.get("server") != "KR":
            continue
        if not labeled_only and c.get("server") not in ("KR", None):
            continue
        by.setdefault(c["moment"], c)
    return list(by.values()), core


def gap_1_3(rows, T, th, form, H):
    """② log10 x = a + b·z + c·z² − m·f(h) → h = T 곡선에서 지금 1%·3% 점수의 ln 차."""
    if len(rows) < 8:
        return None, None
    pts = [(math.log(c["score"]), PR.p_time(form, th, c["h"], H), math.log10(c["pct"])) for c in rows]
    z0 = statistics.mean(p[0] for p in pts)
    X = [[1.0, z - z0, (z - z0) ** 2, -f] for z, f, _ in pts]
    co = ols(X, [p[2] for p in pts])
    if co is None:
        return None, None
    a, b, c2, m = co
    fT = PR.p_time(form, th, T, H)
    zs = [p[0] for p in pts]
    lo, hi = min(zs), max(zs)
    grid = [lo + (hi - lo) * i / 2000 for i in range(2001)]
    L = [a + b * (z - z0) + c2 * (z - z0) ** 2 - m * fT for z in grid]

    def cross(v):
        hits = [i for i in range(2000) if (L[i] - v) * (L[i + 1] - v) <= 0 and L[i] != L[i + 1]]
        if len(hits) != 1:
            return None
        i = hits[0]
        return grid[i] + (grid[i + 1] - grid[i]) * (v - L[i]) / (L[i + 1] - L[i])
    z1, z3 = cross(0.0), cross(math.log10(3.0))
    if z1 is None or z3 is None:
        return None, {"z_range": [round(lo, 3), round(hi, 3)], "m_fit": round(m, 4)}
    return round(z1 - z3, 4), {"z_range": [round(lo, 3), round(hi, 3)], "m_fit": round(m, 4)}


def top_gamma(core, T, servers=("KR",)):
    """③ 한섭 라벨 순위표 1~7위: ln 점수 = 순위별 절편 + γ·ln h.
    ③′ servers=("KR", None) — 라벨 + 미상 줄 (사전 고정 뒤 더함: 순위표의 한섭 표시는 시즌 전체 글로 붙어 T48 사본에서도 12시즌만 남아서, 값 비교 전 개수만 보고)."""
    seen = set()
    by = {}
    for c in core:
        if c.get("src") != "photo" or c.get("screen") != "rank_list" or c.get("server") not in servers:
            continue
        if not c.get("rank") or not (1 <= c["rank"] <= 7) or not c.get("score") or c.get("h") is None:
            continue
        if c["t_grade"] not in ("A", "B") or c.get("exclude") or c["h"] < 1 or c["h"] > T:
            continue
        key = (c["rank"], round(c["score"], 1), round(c["h"], 2))
        if key in seen:
            continue
        seen.add(key)
        by.setdefault(c["rank"], []).append((math.log(c["h"]), math.log(c["score"])))
    rows = [p for v in by.values() for p in v]
    if len(rows) < 8:
        return None, {"rows": len(rows)}
    hs = [math.exp(x) for x, _ in rows]
    if max(hs) / min(hs) < 2:
        return None, {"rows": len(rows), "h_span": round(max(hs) / min(hs), 2)}
    num = den = 0.0
    for v in by.values():
        mx = statistics.mean(x for x, _ in v)
        my = statistics.mean(y for _, y in v)
        num += sum((x - mx) * (y - my) for x, y in v)
        den += sum((x - mx) ** 2 for x, _ in v)
    if den <= 0:
        return None, {"rows": len(rows)}
    return round(num / den, 4), {"rows": len(rows), "ranks": len(by), "h_span": round(max(hs) / min(hs), 2)}


def features_C(pr):
    th, form = pr["transform"]["theta"], pr["transform"]["form"]
    C = {}
    for T in TS:
        C[T] = {}
        for s in SEAS:
            st = trunc(s, T)
            season, c0, H = PR.season_info(st)
            Te = min(float(T), H)
            m_lu = EL.early_m(st, Te, th, form, H, False, BOOT_M, SEED)
            m_l = EL.early_m(st, Te, th, form, H, True, BOOT_M, SEED)
            rows, core = early_rows(st, Te, False)
            gap, gap_info = gap_1_3(rows, Te, th, form, H)
            gam, gam_info = top_gamma(core, Te)
            gamu, gamu_info = top_gamma(core, Te, ("KR", None))
            persons = {c["person"] for c in core if c.get("use") and c.get("pct")}
            C[T][s] = {
                "m_hat": math.log(m_lu["m"]) if m_lu.get("m") and m_lu["m"] > 0 else None,
                "m_hat_raw": m_lu,
                "m_hat_label": math.log(m_l["m"]) if m_l.get("m") and m_l["m"] > 0 else None,
                "m_hat_label_raw": m_l,
                "gap13": gap, "gap13_info": gap_info,
                "gamma_top": gam, "gamma_info": gam_info,
                "gamma_top_u": gamu, "gamma_u_info": gamu_info,
                "ln_persons": round(math.log(len(persons)), 4) if persons else None,
                "persons": len(persons),
                "special": 1 if s in SPECIAL else 0,
            }
            print(f"T{T} {s}: m̂ {m_lu.get('m')} ({m_lu.get('persons')}명) · 라벨 {m_l.get('m')} ({m_l.get('persons')}) · "
                  f"② {gap} · ③ {gam} {gam_info} · ③′ {gamu} {gamu_info} · ④ {len(persons)}", flush=True)
    return C


def compare(F, tg, keys, kind="lin"):
    """m̂ 직선과 같은 시즌 집합(m̂·특징·목표가 모두 있는 시즌)에서 견준다 → 모형 요약 + vs_m_hat(rms 비)·시즌 붓스트랩 10·50·90%."""
    S = [s for s in SEAS if tg.get(s) is not None and F.get(s, {}).get("m_hat") is not None
         and all(F[s].get(k) is not None for k in keys)]
    Fs = {s: F[s] for s in S}
    m = evaluate(Fs, tg, keys, kind)
    if "rms" not in m:
        return m
    b = evaluate(Fs, tg, ["m_hat"])
    m["m_hat_same_seasons_rms"] = b["rms"]
    m["vs_m_hat"] = round(m["rms"] / b["rms"], 3)
    m["vs_m_hat_boot"] = boot_ratio([m["resid"][x] for x in S], [b["resid"][x] for x in S])
    return m


def forward(F, tg, keys, kind="lin", min_train=8, recent=None):
    """규칙 밖 확인: 앞 시즌만으로 맞혀 다음 시즌 (4.3 기본 검증 방향). recent=k면 특징 없이 앞 k시즌 평균 (M2a 꼴)."""
    S = [s for s in SEAS if tg.get(s) is not None and all(F.get(s, {}).get(k) is not None for k in keys)]
    out = {}
    for i, s in enumerate(S):
        tr = S[:i]
        if len(tr) < min_train:
            continue
        if recent:
            out[s] = tg[s] - statistics.mean(tg[x] for x in tr[-recent:])
            continue
        rows = [(x, [F[x][k] for k in keys], tg[x]) for x in tr] + [(s, [F[s][k] for k in keys], tg[s])]
        r = loso(rows, kind)
        out[s] = r[-1]   # 마지막(맞힐 시즌)만 — loso가 그 시즌을 빼고 앞 시즌들로 맞춘 잔차
    return out


def eval_C(C, targets):
    out = {}
    others = ["gap13", "gamma_top", "gamma_top_u", "ln_persons", "special"]
    for T in TS:
        F = C[T]
        oT = out[f"T{T}"] = {}
        for tname, tg in targets.items():
            o = oT[tname] = {}
            o["m_hat"] = evaluate(F, tg, ["m_hat"])
            o["m_hat_label"] = evaluate(F, tg, ["m_hat_label"])
            o["knn_m_hat"] = compare(F, tg, ["m_hat"], "knn")
            for x in others:
                o[f"{x}_own"] = evaluate(F, tg, [x])      # 그 특징이 있는 시즌 모두 (m̂ 없는 시즌 포함)
                o[x] = compare(F, tg, [x])
                o[f"m_hat+{x}"] = compare(F, tg, ["m_hat", x])
                o[f"knn_m_hat+{x}"] = compare(F, tg, ["m_hat", x], "knn")
    return out


def era_check(C):
    """규칙 밖 확인: 초반 특징이 시즌 순서(시대)를 얼마나 따라가는지 — Spearman ρ(특징, 시즌 번호)."""
    out = {}
    for T in TS:
        for k in ("m_hat", "m_hat_label", "gap13", "gamma_top", "gamma_top_u", "ln_persons"):
            S = [s for s in SEAS if C[T][s].get(k) is not None]
            r = spearman([C[T][s][k] for s in S], [SP.snum(s) for s in S])
            out[f"T{T}/{k}"] = {"n": len(S), "rho_season_no": None if r is None else round(r, 3)}
    return out


def partial_rho(A, a, b, c="n"):
    """Spearman 순위로 a~b를 c(시즌 번호)로 뺀 부분 상관 (서술용)."""
    S = [s for s in SEAS if all(A[s].get(k) is not None for k in (a, b, c))]
    ra, rb, rc = (ranks([A[s][k] for s in S]) for k in (a, b, c))
    rab, rac, rbc = statistics.correlation(ra, rb), statistics.correlation(ra, rc), statistics.correlation(rb, rc)
    den = math.sqrt(max((1 - rac ** 2) * (1 - rbc ** 2), 1e-12))
    return {"n": len(S), "rho": round(rab, 3), "partial_given_" + c: round((rab - rac * rbc) / den, 3)}


def forward_shape(C, tg):
    """규칙 밖 확인 (보조 목표 ln(10%/3%)): ②·④가 「최근 5시즌 평균」(M2a 꼴)보다 더 아는지. 앞 시즌만으로."""
    out = {}
    for T in TS:
        F = C[T]
        R = {"recent5_mean": forward(F, tg, [], recent=5), "gap13": forward(F, tg, ["gap13"]),
             "ln_persons": forward(F, tg, ["ln_persons"]), "m_hat+gap13": forward(F, tg, ["m_hat", "gap13"])}
        # 최근 5시즌 평균에 남은 것을 ②로 (앞 시즌 잔차에 직선) — 「최근 시즌 규칙 위에 ②가 더하는가」
        S = [s for s in SEAS if tg.get(s) is not None]
        rec = {}
        for i, s in enumerate(S):
            if i >= 5:
                rec[s] = tg[s] - statistics.mean(tg[x] for x in S[i - 5:i])
        F2 = {s: dict(F[s]) for s in SEAS}
        R["recent5+gap13"] = {}
        rs = [s for s in S if s in rec and F[s].get("gap13") is not None]
        for i, s in enumerate(rs):
            tr = rs[:i]
            if len(tr) < 8:
                continue
            rows = [(x, [F[x]["gap13"]], rec[x]) for x in tr] + [(s, [F[s]["gap13"]], rec[s])]
            R["recent5+gap13"][s] = loso(rows)[-1]
        common = sorted(set.intersection(*[set(v) for v in R.values()]), key=SP.snum)
        out[f"T{T}"] = {"seasons": common, "n": len(common), "rms": {k: round(rms([v[s] for s in common]), 4) for k, v in R.items()}}
    return out


def label_subset(A, C):
    """규칙 밖 확인: 라벨 신호(m̂ 라벨·③)가 있는 시즌 안에서 ln m_s가 시즌 순서를 따르는지 (라벨 신호가 시대 대리인지)."""
    out = {}
    for T in TS:
        for k in ("m_hat_label", "gamma_top"):
            S = [s for s in SEAS if C[T][s].get(k) is not None and A[s].get("ln_ms") is not None]
            if len(S) < 5:
                continue
            out[f"T{T}/{k}"] = {"seasons": S, "rho_ms_feature": round(spearman([A[s]["ln_ms"] for s in S], [C[T][s][k] for s in S]), 3),
                                "rho_ms_season_no": round(spearman([A[s]["ln_ms"] for s in S], [SP.snum(s) for s in S]), 3)}
    return out


def forward_check(C, tg):
    """규칙 밖 확인 (주 목표 ln m_s): 통과한 유사 시즌 꼴이 「최근 시즌」(M2a, k = 5 — criteria_v2)보다 더 아는지. 같은 맞힐 시즌에서 rms."""
    out = {}
    for T in TS:
        F = C[T]
        models = {"m_hat": (["m_hat"], "lin"), "knn_m_hat": (["m_hat"], "knn"),
                  "knn_m_hat+gamma_top_u": (["m_hat", "gamma_top_u"], "knn"), "knn_m_hat+ln_persons": (["m_hat", "ln_persons"], "knn")}
        R = {k: forward(F, tg, kk, kind) for k, (kk, kind) in models.items()}
        R["recent5_mean"] = forward(F, tg, [], recent=5)
        R["recent5_mean_all"] = forward({s: {} for s in SEAS}, tg, [], recent=5)
        common = sorted(set.intersection(*[set(v) for k, v in R.items() if k != "recent5_mean_all"]), key=SP.snum)
        out[f"T{T}"] = {"seasons": common, "n": len(common),
                        "rms": {k: round(rms([v[s] for s in common]), 4) for k, v in R.items() if k != "recent5_mean_all"},
                        "resid": {k: {s: round(v[s], 4) for s in common} for k, v in R.items() if k != "recent5_mean_all"}}
    return out


def decide(EC):
    """4.3 후보 조건 (사전 고정): 주 목표 ln m_s에서 T24·T48 모두 rms가 m̂ 직선의 95% 미만. 형식(special)은 T48 한 번."""
    xs = ["gap13", "gamma_top", "gamma_top_u", "ln_persons"]
    keys = xs + [f"m_hat+{x}" for x in xs] + ["knn_m_hat"] + [f"knn_m_hat+{x}" for x in xs]
    out = {}
    for k in keys:
        v = [EC[f"T{T}"]["ln_ms"].get(k, {}).get("vs_m_hat") for T in TS]
        out[k] = {"vs_m_hat_T24_T48": v, "pass": all(x is not None and x < MARGIN for x in v)}
    for k in ("special", "m_hat+special", "knn_m_hat+special"):
        v = EC["T48"]["ln_ms"].get(k, {}).get("vs_m_hat")
        out[k] = {"vs_m_hat_T48": v, "pass": v is not None and v < MARGIN}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SCR, "diag42.json"))
    a = ap.parse_args()
    pr = json.load(open(os.path.join(AN, "prior_dev.json"), encoding="utf-8"))
    A, shape_summ = table_A(pr)
    targets = {"ln_ms": {s: A[s]["ln_ms"] for s in SEAS}, "ln_10_3": {s: A[s]["ln_10_3"] for s in SEAS},
               "ln_1_3": {s: A[s]["ln_1_3"] for s in SEAS}}
    res = {"about": "할 일 4.2 시즌 차이의 구조 (서술). 사전 고정 커밋 f2703e6 (docs/solo-raid-analysis.md 4.2). 한섭. "
                    "A 끝난 뒤 특징 · B 무리 · C 초반 특징(T24·T48 절단 사본, 정답 없음). 잣대 시즌 하나씩 빼기 직선 잔차 rms·설명 몫",
           "tool_head": os.popen("git -C " + WK + " rev-parse --short HEAD").read().strip(),
           "code_sha1": __import__("hashlib").sha1(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12],
           "prior": {"file": "analysis/prior_dev.json", "form": pr["transform"]["form"], "theta": pr["transform"]["theta"]},
           "A": A, "A_shape_summary": shape_summ, "A_corr": corr_A(A)}
    res["B"] = groups_B(A, targets)
    C = features_C(pr)
    res["C"] = {f"T{T}": C[T] for T in TS}
    res["C_eval"] = eval_C(C, targets)
    res["decide"] = decide(res["C_eval"])
    res["extra"] = {"about": "사전 고정 밖 확인 (해석·4.3에 넘길 근거, 판정에 쓰지 않음)",
                    "era": era_check(C), "forward_ln_ms": forward_check(C, targets["ln_ms"]),
                    "forward_ln_10_3": forward_shape(C, targets["ln_10_3"]), "label_subset": label_subset(A, C),
                    "partial": {"ln_ms~ln_10_3": partial_rho(A, "ln_ms", "ln_10_3"), "ln_ms~ln_kr_rows": partial_rho(A, "ln_ms", "ln_kr_rows"),
                                "ln_ms~ln_kr3": partial_rho(A, "ln_ms", "ln_kr3"), "ln_10_3~ln_kr_rows": partial_rho(A, "ln_10_3", "ln_kr_rows")}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("→", a.out)


if __name__ == "__main__":
    sys.exit(main())
