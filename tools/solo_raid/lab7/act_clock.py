"""활동 시간 축 (WP11a ⓐ, 2026-11-03) — 같은 사람·같은 점수 기록 쌍으로 하루 주기 활동 모양 g를 재고, 시간 좌표 u* = A(h)/A(H)를 낸다.

  python3 -I tools/solo_raid/lab7/act_clock.py fit --obs-dir W/obs7/full --out data/solo-raid/analysis/r7/wp11/act_profile_D1.json

계획: docs/solo-raid-method.md 「WP11」 ⓐ. 운영 시간(operational time, Clark 1973 · Ané·Geman 2000)과 호출 센터 「총량 × 하루 안 모양」(Weinberg·Brown·Stroud 2007)의 꼴.
  A(h) = ∫_0^h g(KST 시각(τ)) dτ,  u* = A(h)/A(H)  (곡면·제약은 그대로라 단조와 P(1) = 0이 저절로 선다. 하한 0.02는 u*에 같은 값)
  g = KST 3시간 칸 8개의 계단 함수 (05-08, 08-11, …, 02-05 — 05시 초기화에 맞춤), 하루 평균 1.
g 추정 (관측만, 정답 표 안 씀): 쌍 i (같은 pid·같은 점수, A·B 시각, 한섭·미상, use, 시각 차 1시간 이상 — wp10_diag.tracker_pairs와 같은 거르기, 시작 시각이 다른 시즌도 넣음)마다
  ΔL_i = log10(p2/p1) = m_s · Σ_{d,b} D_d · g_b · E_i[d,b]     (E = 겹친 시간, d = 시즌 일차(05시 경계), b = 시각 칸)
  m_s 시즌 배수 · D_d 일차 상수 (시즌 안 흐름 — 곡면 ℓ이 받으므로 버림) · g ≥ 0. 무게 1/sd (wp10_diag와 같은 판독 오차).
  벌점 λ1 Σ_b (g_b − g_{b+1})² (순환 이웃) + λ2 Σ_b (g_b − 1)² (평평 쪽). 번갈아 비음수 최소제곱 (D·g는 NNLS, m은 닫힌 꼴).
  벌점 크기 = 시즌 하나씩 빼는 교차검증 (뺀 시즌은 m만 다시 맞춰 예측 오차 제곱합) + 1-SE 규칙 (Breiman 외 1984 · Hastie 외 ESL 7.10):
  최소 교차검증 오차의 벌점과 견준 시즌별 오차 차의 합이 그 표준오차(차의 sd × √시즌 수, 짝지은 판) 안인 벌점 가운데 g의 유효 자유도가 가장 작은 (가장 매끈한) 것.
  (2026-11-03 조율 세션, 예측 결과를 보기 전 판단 — 최소값만 고르면 쌍이 적은 S21~S23·S27에서 벌점이 0 근처로 가 g가 들쭉날쭉했고,
   짝짓지 않은 표준오차(시즌 오차 합의 sd)는 시즌마다 쌍 수가 달라 너무 커서 S30까지 g = 1, 전체도 0.73~1.27로 눌렀다)
앞 시즌만 (expanding): 맞힐 시즌 s의 g_s = s보다 앞 시즌 쌍만 (벌점도 그 안의 교차검증). 앞 시즌이 2개 미만인 S19·S20은 전체 값 (WP1 지연과 같은 처리).
  쌍이 적은 앞쪽 시즌은 평평 쪽 벌점으로 g = 1에 가까워진다.
읽기: load(path) → 프로필, ustar(prof, season, t, H) → u* (numpy). obs_model.set_clock이 쓴다.
공개 규칙: 글 번호·원문·닉네임·원글 주소·사람 열쇠를 쓰지 않는다 (출력은 시즌·칸별 숫자만).
"""
import argparse
import collections
import datetime as dt
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

FMT = "%Y-%m-%d %H:%M:%S"
NB = 8                                              # 시각 칸 수 (3시간)
BW = 24.0 / NB
ND = 7                                              # 일차 상수 수 (S28 130시간이 0~5일차)
BINS = [f"{(5 + 3 * k) % 24:02d}-{(8 + 3 * k) % 24:02d}" for k in range(NB)]
L1_GRID = (0.0, 1e2, 1e3, 1e4, 1e5, 1e6)            # 순환 이웃 차 벌점
L2_GRID = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8)       # 평평 쪽 벌점 (1e8 ≈ g = 1)
U_LO = 0.02


def c0_hour(season):
    """시즌 시작 시각(KST)의 하루 안 시각 (시간, 소수). 관측 표 머리 c0와 같은 값 (state/season.json c_start)."""
    with open(os.path.join(C.ROOT, "data", "solo-raid", season, "state", "season.json"), encoding="utf-8") as f:
        c = dt.datetime.strptime(json.load(f)["c_start"], FMT)
    return c.hour + c.minute / 60 + c.second / 3600


# ---------- 시간 축 ----------

def _segments(c0h, H):
    """[0, H]를 시각 칸 경계(05시 + 3k)에서 자른 조각 (시작, 끝, 일차 d, 칸 b) 목록."""
    out = []
    t = 0.0
    while t < H - 1e-9:
        x = c0h + t - 5.0                          # 05시를 0으로 둔 KST 누적 시각
        k = math.floor(x / BW + 1e-9)
        end = min(H, (k + 1) * BW + 5.0 - c0h)
        d = math.floor(x / 24.0 + 1e-9) - math.floor((c0h - 5.0) / 24.0 + 1e-9)
        out.append((t, end, int(d), int(k % NB)))
        t = end
    return out


def exposure(c0h, H, t1, t2):
    """구간 [t1, t2]의 겹친 시간 E[ND, NB]."""
    E = np.zeros((ND, NB))
    for a, b, d, k in _segments(c0h, H):
        o = min(b, t2) - max(a, t1)
        if o > 0:
            E[min(d, ND - 1), k] += o
    return E


def A_of(g, c0h, H, t):
    """A(t) = ∫_0^t g(칸(τ)) dτ (numpy 배열 t, 0 ≤ t ≤ H 밖은 자른다)."""
    seg = _segments(c0h, H)
    bnd = np.array([0.0] + [s[1] for s in seg])
    cum = np.concatenate([[0.0], np.cumsum([(s[1] - s[0]) * g[s[3]] for s in seg])])
    return np.interp(np.clip(np.asarray(t, float), 0.0, H), bnd, cum)


def ustar_g(g, c0h, H, t):
    return np.clip(A_of(g, c0h, H, t) / A_of(g, c0h, H, [H])[0], U_LO, 1.0)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ustar(prof, season, t, H):
    """프로필의 g_season으로 u*. 시즌 시작 시각은 프로필에 적힌 c0h."""
    e = prof["g"][season]
    return ustar_g(np.asarray(e["g"], float), float(e["c0h"]), float(H), t)


# ---------- 기록 쌍 ----------

def pairs_of(obs_dir, seasons):
    """시즌마다 쌍 목록 [(t1, t2, ΔL, var)] (wp10_diag.tracker_pairs 거르기 + 시작 시각이 다른 시즌 포함, 시각은 [0, H])."""
    import obs_table as OT
    out = {}
    for s in seasons:
        head, rows = OT.read_cell(os.path.join(obs_dir, f"{s}.jsonl.gz"))
        H = float(head["H"])
        g = collections.defaultdict(list)
        for r in rows:
            if r["kind"] not in ("A", "B") or not r.get("pct") or not r.get("score") or not r.get("pid") or not r.get("use"):
                continue
            if r.get("server") not in (None, "KR") or r.get("h") is None:
                continue
            g[(r["pid"], round(r["score"], 2))].append((min(max(r["h"], 0.0), H), r["pct"]))
        P = []
        for L in g.values():
            L = sorted(set(L))
            for (t1, p1), (t2, p2) in zip(L, L[1:]):
                if t2 - t1 < 1.0 or t1 >= H:
                    continue
                dL = math.log10(p2 / p1)
                if dL < -0.02:
                    continue
                var = ((0.005 / p1) ** 2 + (0.005 / p2) ** 2) / math.log(10) ** 2 + 0.004 ** 2
                P.append((t1, t2, dL, var))
        out[s] = {"H": H, "c0h": c0_hour(s), "pairs": P}
    return out


def design(data, seasons):
    """쌍을 모은 배열: E[n, ND, NB], y[n], w[n], si[n] (seasons 안 번호)."""
    for s in seasons:                              # 시즌별 배열은 한 번만 만든다
        d = data[s]
        if "_E" not in d:
            P = d["pairs"]
            d["_E"] = np.array([exposure(d["c0h"], d["H"], t1, t2) for t1, t2, _, _ in P]).reshape(len(P), ND, NB)
            d["_y"] = np.array([p[2] for p in P], float)
            d["_w"] = np.array([1 / math.sqrt(p[3]) for p in P], float)
    E = np.concatenate([data[s]["_E"] for s in seasons]) if seasons else np.zeros((0, ND, NB))
    y = np.concatenate([data[s]["_y"] for s in seasons]) if seasons else np.zeros(0)
    w = np.concatenate([data[s]["_w"] for s in seasons]) if seasons else np.zeros(0)
    si = np.concatenate([np.full(len(data[s]["_y"]), j, int) for j, s in enumerate(seasons)]) if seasons else np.zeros(0, int)
    return E, y, w, si


# ---------- 맞추기 ----------

def fit(E, y, w, si, ns, l1, l2, iters=40, g0=None):
    """번갈아 NNLS. -> (g[NB] 평균 1, D[ND], m[ns] 기하평균 1, 가중 잔차 제곱합)."""
    g = np.ones(NB) if g0 is None else np.array(g0, float)
    D = np.full(ND, max(float((y * w).sum() / max((E.sum((1, 2)) * w).sum(), 1e-12)), 1e-6))
    m = np.ones(ns)
    Pn = np.zeros((NB, NB))                       # 순환 이웃 차 행렬
    for b in range(NB):
        Pn[b, b], Pn[b, (b + 1) % NB] = 1.0, -1.0
    used = E.sum((0, 2)) > 0                      # 쌍이 닿은 일차만
    for _ in range(iters):
        Xd = m[si, None] * np.einsum("idb,b->id", E, g)
        Dn = np.zeros(ND)
        if used.any():
            Dn[used], _ = nnls(Xd[:, used] * w[:, None], y * w)
        D = Dn
        Xg = m[si, None] * np.einsum("idb,d->ib", E, D)
        A = np.vstack([Xg * w[:, None], math.sqrt(l1) * Pn, math.sqrt(l2) * np.eye(NB)])
        b = np.concatenate([y * w, np.zeros(NB), math.sqrt(l2) * np.ones(NB)])
        g, _ = nnls(A, b)
        c = g.mean()
        if c <= 0:
            g = np.ones(NB)
            c = 1.0
        g, D = g / c, D * c
        base = np.einsum("idb,d,b->i", E, D, g)
        for j in range(ns):
            ix = si == j
            den = ((base[ix] * w[ix]) ** 2).sum()
            if den > 0:
                m[j] = (y[ix] * base[ix] * w[ix] ** 2).sum() / den
        gm = math.exp(np.mean(np.log(np.clip(m[[j for j in range(ns) if (si == j).any()]], 1e-9, None)))) if len(si) else 1.0
        m, D = m / gm, D * gm
    pred = m[si] * np.einsum("idb,d,b->i", E, D, g)
    return g, D, m, float((((y - pred) * w) ** 2).sum())


def cv_loss(data, seasons, l1, l2):
    """시즌 하나씩 빼기: 나머지로 g·D, 뺀 시즌은 m만 맞춰 가중 잔차 제곱합. -> (합, 쌍 수, 시즌별 오차 목록)."""
    tot, n, folds = 0.0, 0, []
    for s in seasons:
        rest = [x for x in seasons if x != s]
        E, y, w, si = design(data, rest)
        if len(y) == 0:
            continue
        g, D, _, _ = fit(E, y, w, si, len(rest), l1, l2)
        Eo, yo, wo, _ = design(data, [s])
        if len(yo) == 0:
            continue
        base = np.einsum("idb,d,b->i", Eo, D, g)
        den = ((base * wo) ** 2).sum()
        mo = (yo * base * wo ** 2).sum() / den if den > 0 else 1.0
        f = float((((yo - mo * base) * wo) ** 2).sum())
        tot += f
        folds.append(f)
        n += len(yo)
    return tot, n, folds


def g_df(E, y, w, si, m, D, l1, l2):
    """g 쪽 벌점 최소제곱의 유효 자유도 tr(X'X (X'X + λ1 P'P + λ2 I)^-1) (X = 고정한 m·D에서 g의 설계)."""
    Pn = np.zeros((NB, NB))
    for b in range(NB):
        Pn[b, b], Pn[b, (b + 1) % NB] = 1.0, -1.0
    X = (m[si, None] * np.einsum("idb,d->ib", E, D)) * w[:, None]
    G = X.T @ X
    return float(np.trace(G @ np.linalg.inv(G + l1 * Pn.T @ Pn + l2 * np.eye(NB))))


def fit_cv(data, seasons):
    """벌점 격자 교차검증 → 고른 벌점으로 전체 적합. 시즌 2개 미만이면 None."""
    have = [s for s in seasons if data[s]["pairs"]]
    if len(have) < 2:
        return None
    cv, folds, df, fits = {}, {}, {}, {}
    E, y, w, si = design(data, have)
    for l1 in L1_GRID:
        for l2 in L2_GRID:
            cv[(l1, l2)], _, folds[(l1, l2)] = cv_loss(data, have, l1, l2)
            fits[(l1, l2)] = fit(E, y, w, si, len(have), l1, l2)
            g_, D_, m_, _ = fits[(l1, l2)]
            df[(l1, l2)] = g_df(E, y, w, si, m_, D_, l1, l2)
    cmin = min(cv, key=cv.get)
    f0 = np.array(folds[cmin])
    sed = {k: float(np.std(np.array(folds[k]) - f0, ddof=1) * math.sqrt(len(f0))) if len(f0) >= 2 else 0.0 for k in cv}
    ok = [k for k in cv if cv[k] - cv[cmin] <= sed[k]]
    se = sed
    best = min(ok, key=lambda k: (df[k], -k[1], -k[0]))
    g, D, m, sse = fits[best]
    flat = cv[(L1_GRID[0], L2_GRID[-1])]
    return {"g": g.tolist(), "D": D.tolist(), "m": dict(zip(have, m.tolist())), "lam": list(best), "n_pairs": int(len(y)),
            "from": have, "cv_best": cv[best], "cv_min": cv[cmin], "lam_min": list(cmin), "cv_se_diff": se[best], "df": df[best], "df_min": df[cmin],
            "cv_flat": flat, "cv_gain": 1 - cv[best] / flat if flat > 0 else None}


def day_resid(data, seasons, g, D, m):
    """보고만: 2일차(h < 48) 대 3일차 뒤 쌍의 칸별 가중 평균 잔차 비 (관측 ÷ 적합) — 하루 모양이 일차마다 같은가."""
    out = {}
    for name, sel in (("d2", lambda t1, t2: t2 <= 48), ("d3plus", lambda t1, t2: t1 >= 48)):
        num, den = np.zeros(NB), np.zeros(NB)
        for s in seasons:
            d = data[s]
            for t1, t2, dL, var in d["pairs"]:
                if not sel(t1, t2):
                    continue
                Ei = exposure(d["c0h"], d["H"], t1, t2)
                pr = m.get(s, 1.0) * np.einsum("db,d,b->", Ei, np.asarray(D), np.asarray(g))
                if pr <= 0:
                    continue
                wb = Ei.sum(0) / max(Ei.sum(), 1e-9)
                num += wb * dL / var
                den += wb * pr / var
        out[name] = [round(float(a / b), 3) if b > 0 else None for a, b in zip(num, den)]
    return out


def post_shape(obs_dir, seasons):
    """보고만: 시각 칸별 게시량 (한섭·미상 A·B·C 사진 줄의 게시 시각 hp) — 하루 평균 1로."""
    import obs_table as OT
    cnt, expo = np.zeros(NB), np.zeros(NB)
    for s in seasons:
        head, rows = OT.read_cell(os.path.join(obs_dir, f"{s}.jsonl.gz"))
        c0h, H = c0_hour(s), float(head["H"])
        for a, b, d, k in _segments(c0h, H):
            expo[k] += b - a
        for r in rows:
            if r["kind"] in ("A", "B", "C") and r.get("use") and r.get("hp") is not None and r.get("server") in (None, "KR") and 0 <= r["hp"] < H:
                cnt[int(math.floor(((c0h + r["hp"] - 5) % 24) / BW))] += 1
    rate = cnt / np.maximum(expo, 1e-9)
    return (rate / rate.mean()).round(3).tolist()


def cmd_fit(a):
    t0 = time.time()
    seasons = C.parse_seasons(a.seasons)
    data = pairs_of(a.obs_dir, seasons)
    print("쌍 수:", {s: len(d["pairs"]) for s, d in data.items()}, flush=True)
    full = fit_cv(data, seasons)
    print(f"전체 g {np.round(full['g'], 2).tolist()} 벌점 {full['lam']} cv 이득 {full['cv_gain']:.3f} ({time.time() - t0:.0f}초)", flush=True)
    prof = {"header": {"about": "WP11a 활동 시간 g (KST 3시간 칸 8개, 하루 평균 1) — 같은 점수 기록 쌍, 앞 시즌만 (expanding). 정답 표 안 씀",
                       "data_ver": C.DATA_VER, "date": dt.date.today().isoformat(), "bins": BINS, "nd": ND,
                       "grid": {"l1": list(L1_GRID), "l2": list(L2_GRID)}, "u_lo": U_LO,
                       "rule": "g_s = s보다 앞 시즌 쌍 (벌점은 그 안의 시즌 빼기 교차검증 + 짝지은 1-SE 규칙 — 유효 자유도 최소). 앞 시즌 쌍이 있는 시즌이 2개 미만이면 전체 값 (S19·S20)"},
            "full": full, "g": {}, "ustar": {}}
    for s in seasons:
        prev = [x for x in seasons if int(x[1:]) < int(s[1:])]
        e = fit_cv(data, prev)
        rule = "expanding"
        if e is None:
            e, rule = full, "full"
        prof["g"][s] = {"g": [round(x, 5) for x in e["g"]], "lam": e["lam"], "lam_min": e["lam_min"], "df": round(e["df"], 2), "n_pairs": e["n_pairs"], "n_seasons": len(e["from"]),
                        "rule": rule, "c0h": data[s]["c0h"], "H": data[s]["H"], "cv_gain": e["cv_gain"]}
        H = data[s]["H"]
        T = [float(x) for x in range(0, int(H) + 1, 6)] + ([H] if H % 6 else [])
        prof["ustar"][s] = {"T": T, "u": [round(min(max(t / H, U_LO), 1.0), 4) for t in T],
                            "ustar": [round(float(x), 4) for x in ustar_g(np.asarray(e["g"]), data[s]["c0h"], H, T)]}
        print(s, rule, np.round(e["g"], 2).tolist(), e["lam"], e["n_pairs"], f"({time.time() - t0:.0f}초)", flush=True)
    prof["report"] = {"day_resid_full": day_resid(data, full["from"], full["g"], full["D"], full["m"]),
                      "post_shape": post_shape(a.obs_dir, seasons), "full_D": full["D"],
                      "g_by_season": {s: prof["g"][s]["g"] for s in seasons}, "secs": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(prof, f, ensure_ascii=False, indent=1)
    print("씀", a.out, f"{time.time() - t0:.0f}초", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--obs-dir", required=True, help="끝난 시즌 전체 관측 표 (obs_table.py build --full)")
    f.add_argument("--seasons", default="S19-S42")
    f.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_fit(a)


if __name__ == "__main__":
    main()
