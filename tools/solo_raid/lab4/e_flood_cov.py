"""할 일 4.2b 시즌 안 침수 속도 변화 — 두 시각 상관 지도 (서술, 2026-10-18).

사전 고정 커밋 1acf857 (docs/solo-raid-analysis.md 4.2b). 시간을 나누지 않고 「시즌이 두 시각에서 같은 쪽으로
벗어나는 정도」를 지도로 편다 — 희소 관측의 함수 주성분 분석 (PACE 꼴, Yao·Müller·Wang 2005).

- 자료: prior_dev season_levels 시즌, 한섭, fit_prior.load_units(state, ("A","A2"), "all")의 값 칸, h < H.
- 편차 e = P / g(h) − 1, g = prior_dev 공통 꼴 (시즌 배수 없음). 잡음 s = σ(h)/g(h), 무게 w = 1/s².
- 축 u(h) = 시즌마다 같은 무게로 모은 값 칸 h의 경험 분포. 평균 μ(u)·공분산 C(u,v) 모두 국소 선형 가우스 평활.
- C는 같은 시즌·다른 사람 짝만 (짝 무게 w_i·w_j를 시즌마다 합 1). 폭은 시즌 하나씩 빼기로 고른다.
- 가우스 커널이 두 좌표로 갈라지므로 짝 합은 시즌마다 행렬 곱 K_a·M·K_bᵀ로 정확히 낸다 (구간 묶기 없음).

lab 코드 (numpy). 운영 경로·Actions에 닿지 않는다. 레포를 읽기만 하고 결과 JSON 하나를 쓴다.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tools/solo_raid
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import split as SP  # noqa: E402

SP.install_guard()
import fit_prior as FP  # noqa: E402
import fit_prior_multi as FM  # noqa: E402
import predict as PR  # noqa: E402

AN = os.path.join(ROOT, "data", "solo-raid", "analysis")
PREREG = "1acf857"
BWS = [0.03, 0.05, 0.07, 0.10, 0.14, 0.20, 0.28]
NG = 40
UG = (np.arange(NG) + 0.5) / NG            # 0.0125 … 0.9875
H_R1 = [18, 42, 66, 90]
H_END = 108
FLAT_H = (6, 117)
SPECIAL = {"S19": "재오픈 (12:00 시작, 89시간)", "S26": "기간 변경", "S28": "재개 (18:30 시작, 130시간)",
           "S38": "재오픈 (12:00 시작, 113시간)", "S41": "모의전 사전 개방"}
HREP = 120                                 # R2의 g(h)를 그릴 때 쓰는 시즌 길이 (보통 시즌)


def snum(s):
    return int(s[1:])


# ---------------- 자료 ----------------

def load_raw(seasons):
    raw = {}
    for s in seasons:
        _, H, _, units = FP.load_units(FM.state_of(s), ("A", "A2"), "all")
        U = [u for u in units if u["server"] == "KR" and u["cens"] is None and u["h"] < H]
        pid = {p: k for k, p in enumerate(sorted({u["person"] for u in U}))}
        raw[s] = {"H": H, "h": np.array([u["h"] for u in U], float), "P": np.array([u["P"] for u in U], float),
                  "x": np.array([u["x"] for u in U], float), "conly": np.array([u["grade"] == "C" for u in U]),
                  "pid": np.array([pid[u["person"]] for u in U], int)}
    return raw


def prep(raw, tr, shift_c=0.0, keep=None, zmax=None):
    """시즌별 h·e·s·w·사람. shift_c: 등급 C뿐인 칸의 h를 당김 (S1). keep: 시즌별 남길 칸 bool (S2).
    zmax: |P − g| / σ가 이보다 큰 칸을 뺌 (사전 고정 밖 X2)."""
    form, th = tr["form"], tr["theta"]
    s0, rho = tr["sigma"]["s0"], tr["sigma"]["rho"]
    out = {}
    for s, R in raw.items():
        h = R["h"].copy()
        if shift_c:
            h[R["conly"]] = np.maximum(h[R["conly"]] - shift_c, 0.25)
        m = np.ones(len(h), bool) if keep is None else keep[s]
        h, P, pid, x = h[m], R["P"][m], R["pid"][m], R["x"][m]
        g = np.array([PR.p_time(form, th, v, R["H"]) for v in h])
        sd = np.sqrt(s0 ** 2 + (rho * g) ** 2) / g
        e = P / g - 1
        if zmax is not None:
            k = np.abs(e / sd) <= zmax
            h, x, pid, g, sd, e = h[k], x[k], pid[k], g[k], sd[k], e[k]
        out[s] = {"H": R["H"], "h": h, "x": x, "e": e, "s": sd, "w": 1 / sd ** 2, "pid": pid}
    return out


# ---------------- 축 ----------------

class Axis:
    """u(h) = 시즌 같은 무게 경험 분포 (가운데 순위, 선형 보간). kind="h"면 u = h 그대로 (S5)."""

    def __init__(self, D, kind="u"):
        self.kind = kind
        if kind == "h":
            return
        hs, ws = [], []
        S = len(D)
        for d in D.values():
            hs.append(d["h"])
            ws.append(np.full(len(d["h"]), 1 / (S * len(d["h"]))))
        h, w = np.concatenate(hs), np.concatenate(ws)
        o = np.argsort(h, kind="stable")
        h, w = h[o], w[o]
        F = np.cumsum(w) - w / 2
        self.hp = np.concatenate([[0.0], h, [max(h[-1], 130.0)]])
        self.Fp = np.concatenate([[0.0], F, [1.0]])

    def u(self, h):
        return np.asarray(h, float) if self.kind == "h" else np.interp(h, self.hp, self.Fp)

    def h(self, u):
        return np.asarray(u, float) if self.kind == "h" else np.interp(u, self.Fp, self.hp)


# ---------------- 평활 ----------------

def kern(z, u, b):
    d = u[None, :] - z[:, None]
    return np.exp(-0.5 * (d / b) ** 2), d


def ll1(z, u, y, w, b):
    K, d = kern(z, u, b)
    K = K * w[None, :]
    S0, S1, S2 = K.sum(1), (K * d).sum(1), (K * d * d).sum(1)
    T0, T1 = (K * y).sum(1), (K * d * y).sum(1)
    return (S2 * T0 - S1 * T1) / (S0 * S2 - S1 ** 2)


def mean_fit(D, names, b, sw=True):
    """μ의 자료 (u·e·무게). names는 중복 허용 (붓스트랩)."""
    u = np.concatenate([D[s]["u"] for s in names])
    y = np.concatenate([D[s]["e"] for s in names])
    w = np.concatenate([D[s]["w"] / (len(D[s]["e"]) if sw else 1.0) for s in names])
    return u, y, w


def pair_mats(d, r, sw=True):
    """시즌 하나의 짝 무게 M (다른 사람만)과 M∘(r rᵀ)."""
    M = np.outer(d["w"], d["w"]) * (d["pid"][:, None] != d["pid"][None, :])
    tot = M.sum()
    if tot <= 0:
        return None, None
    if sw:
        M = M / tot
    return M, M * np.outer(r, r)


def cov_moments(items, za, zb, b):
    """Σ_시즌 K_a^(k)·M·K_b^(l)ᵀ — 국소 선형 2차원에 쓰는 모멘트 (S 6개·T 3개)."""
    na, nb = len(za), len(zb)
    S = {k: np.zeros((na, nb)) for k in ("00", "10", "01", "20", "11", "02")}
    T = {k: np.zeros((na, nb)) for k in ("00", "10", "01")}
    for u, M, MY in items:
        Ka, da = kern(za, u, b)
        Kb, db = kern(zb, u, b)
        A = {0: Ka, 1: Ka * da, 2: Ka * da * da}
        B = {0: Kb, 1: Kb * db, 2: Kb * db * db}
        MB = {l: M @ B[l].T for l in (0, 1, 2)}
        for k in S:
            S[k] += A[int(k[0])] @ MB[int(k[1])]
        YB = {l: MY @ B[l].T for l in (0, 1)}
        for k in T:
            T[k] += A[int(k[0])] @ YB[int(k[1])]
    return S, T


def ll2(S, T):
    A = np.stack([np.stack([S["00"], S["10"], S["01"]], -1),
                  np.stack([S["10"], S["20"], S["11"]], -1),
                  np.stack([S["01"], S["11"], S["02"]], -1)], -2)
    y = np.stack([T["00"], T["10"], T["01"]], -1)
    with np.errstate(all="ignore"):
        det = np.linalg.det(A)
        ok = np.abs(det) > 1e-300
        A2 = np.where(ok[..., None, None], A, np.eye(3))
        sol = np.linalg.solve(A2, y[..., None])[..., 0]
    out = sol[..., 0]
    out[~ok] = np.nan
    return out


def estimate(D, names, b, za, sw=True):
    """μ(za)·C(za, za) (평활 그대로, 다듬기 전). names 중복 허용."""
    u, y, w = mean_fit(D, names, b, sw)
    mu_z = ll1(za, u, y, w, b)
    items = []
    for s in names:
        d = D[s]
        r = d["e"] - ll1(d["u"], u, y, w, b)
        M, MY = pair_mats(d, r, sw)
        if M is not None:
            items.append((d["u"], M, MY))
    S, T = cov_moments(items, za, za, b)
    C = ll2(S, T)
    return mu_z, C


def loso_err(D, names, b, sw=True):
    """시즌 하나씩 빼기: 뺀 시즌 짝 곱을 나머지 시즌 지도(μ도 나머지로)로 맞춘 무게 제곱오차의 합."""
    tot, bad = 0.0, 0
    for s in names:
        rest = [t for t in names if t != s]
        u, y, w = mean_fit(D, rest, b, sw)
        items = []
        for t in rest:
            d = D[t]
            r = d["e"] - ll1(d["u"], u, y, w, b)
            M, MY = pair_mats(d, r, sw)
            if M is not None:
                items.append((d["u"], M, MY))
        d = D[s]
        r = d["e"] - ll1(d["u"], u, y, w, b)
        M, _ = pair_mats(d, r, True)   # 뺀 시즌 오차는 늘 시즌 같은 무게로 더한다
        if M is None:
            continue
        S, T = cov_moments(items, d["u"], d["u"], b)
        Ch = ll2(S, T)
        R2 = (np.outer(r, r) - Ch) ** 2
        okm = np.isfinite(R2)
        bad += int((~okm & (M > 0)).sum())
        tot += float((M * np.where(okm, R2, 0.0)).sum())
    return tot, bad


# ---------------- 판독 ----------------

def interp_mat(z, grid):
    """z의 점마다 grid 위 선형 보간 무게 (끝 밖은 끝 값)."""
    z = np.clip(np.asarray(z, float), grid[0], grid[-1])
    L = np.zeros((len(z), len(grid)))
    j = np.clip(np.searchsorted(grid, z) - 1, 0, len(grid) - 2)
    f = (z - grid[j]) / (grid[j + 1] - grid[j])
    L[np.arange(len(z)), j] = 1 - f
    L[np.arange(len(z)), j + 1] = f
    return L


def psd(C, du):
    C = (C + C.T) / 2
    lam, V = np.linalg.eigh(C * du)
    o = np.argsort(lam)[::-1]
    lam, V = lam[o], V[:, o]
    neg = float(-lam[lam < 0].sum())
    lam = np.clip(lam, 0, None)
    Cp = (V * lam) @ V.T / du
    return Cp, lam, V / math.sqrt(du), neg


def zero_cross(hg, f, lo, hi):
    m = (hg >= lo) & (hg <= hi)
    x, y = hg[m], f[m]
    z = []
    for i in range(len(x) - 1):
        if y[i] == 0 or y[i] * y[i + 1] < 0:
            z.append(float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])))
    return z


def readouts(mu, C, ax, grid, tr, ref=None):
    du = grid[1] - grid[0]
    Cp, lam, phi, neg = psd(C, du)
    hg = ax.h(grid)
    sd = np.sqrt(np.clip(np.diag(Cp), 1e-300, None))
    rho = Cp / np.outer(sd, sd)

    def Cat(ha, hb):
        La, Lb = interp_mat(ax.u(ha), grid), interp_mat(ax.u(hb), grid)
        return La @ Cp @ Lb.T

    def rho_at(ha, hb):
        c = Cat(ha, hb)
        va = np.diag(Cat(ha, ha))
        vb = np.diag(Cat(hb, hb))
        return c / np.sqrt(np.outer(va, vb))

    r_end = (Cp @ interp_mat(ax.u([H_END]), grid).T)[:, 0] / (sd * math.sqrt(Cat([H_END], [H_END])[0, 0]))
    r1 = {str(h): float(rho_at([h], [H_END])[0, 0]) for h in H_R1}
    # R2 밀린 양 (log10 % 단위, g는 120시간 시즌 기준)
    g = lambda h: PR.p_time(tr["form"], tr["theta"], h, HREP)  # noqa: E731
    A = np.arange(12, 73, 1.0)
    pts = np.unique(np.concatenate([A, A + 24, [96.0]]))
    idx = {float(p): k for k, p in enumerate(pts)}
    CD = Cat(pts, pts) * np.outer([g(p) for p in pts], [g(p) for p in pts])
    k96 = idx[96.0]
    r2 = []
    for a in A:
        i, j = idx[float(a)], idx[float(a + 24)]
        cov = CD[i, k96] - CD[j, k96]
        vI = CD[i, i] + CD[j, j] - 2 * CD[i, j]
        r2.append(float(cov / math.sqrt(vI * CD[k96, k96])) if vI > 0 and CD[k96, k96] > 0 else float("nan"))
    # R3 주성분
    pos = lam[lam > 0]
    share = (lam / pos.sum()) if pos.size else lam * 0
    p1, p2 = phi[:, 0].copy(), phi[:, 1].copy()
    if ref is not None:
        if p1 @ ref["phi1"] < 0:
            p1 = -p1
        if p2 @ ref["phi2"] < 0:
            p2 = -p2
    else:
        if p1.sum() < 0:
            p1 = -p1
        if p2[-1] < 0:
            p2 = -p2
    mf = (hg >= FLAT_H[0]) & (hg <= FLAT_H[1])
    flat = float(p1[mf].min() / p1[mf].max()) if p1[mf].max() > 0 else float("nan")
    return {"Cp": Cp, "rho": rho, "hg": hg, "lam": lam, "phi": phi, "phi1": p1, "phi2": p2, "neg": neg,
            "mu": mu, "r_end": r_end, "r1": r1, "r2_a": A, "r2": np.array(r2), "share": share,
            "flat": flat, "z2": zero_cross(hg, p2, 3, 119)}


def pace_scores(D, names, mu_fn, R, ax, grid, sw=True):
    """제1·2 성분 점수 (조건부 기댓값). mu_fn(u) = μ."""
    lam, phi = R["lam"], R["phi"]
    out = {}
    for s in names:
        d = D[s]
        L = interp_mat(d["u"], grid)
        r = d["e"] - mu_fn(d["u"])
        SY = L @ R["Cp"] @ L.T + np.diag(d["s"] ** 2)
        a = np.linalg.solve(SY, r)
        sgn = [1.0 if phi[:, k] @ R["phi1" if k == 0 else "phi2"] >= 0 else -1.0 for k in (0, 1)]
        xi = [float(lam[k] * sgn[k] * (L @ phi[:, k]) @ a) for k in (0, 1)]
        out[s] = {"xi1": xi[0], "xi2": xi[1], "n": int(len(r)), "persons": int(len(set(d["pid"].tolist())))}
    return out


# ---------------- 한 판 ----------------

def run(D, names, b, ax, grid, tr, B, seed, sw=True, ref=None):
    for d in D.values():
        d["u"] = ax.u(d["h"])
    mu, C = estimate(D, names, b, grid, sw)
    R = readouts(mu, C, ax, grid, tr, ref)
    u, y, w = mean_fit(D, names, b, sw)
    R["scores"] = pace_scores(D, names, lambda z: ll1(z, u, y, w, b), R, ax, grid, sw)
    rng = np.random.default_rng(seed)
    boots = []
    refb = {"phi1": R["phi1"], "phi2": R["phi2"]}
    for _ in range(B):
        nb = [names[k] for k in rng.integers(0, len(names), len(names))]
        mub, Cb = estimate(D, nb, b, grid, sw)
        if not np.all(np.isfinite(Cb)):
            continue
        boots.append(readouts(mub, Cb, ax, grid, tr, refb))
    R["boots"] = boots
    return R


def q(v, p):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)])
    return float(np.quantile(v, p)) if v.size else None


def band(vals):
    return {"lo": q(vals, 0.1), "hi": q(vals, 0.9)}


def rd(x, k=3):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rd(v, k) for v in x]
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), k)


def summarize(R, full=True):
    bs = R["boots"]
    hg = R["hg"]
    o = {"bootstrap_ok": len(bs),
         "R1": {"rho_vs_108": {h: {"est": rd(v), **{k: rd(x) for k, x in band([b["r1"][h] for b in bs]).items()}}
                               for h, v in R["r1"].items()}},
         "R2": {"a12": {"est": rd(R["r2"][0]), **{k: rd(x) for k, x in band([b["r2"][0] for b in bs]).items()}}},
         "R3": {"share": [rd(x) for x in R["share"][:4]],
                "share1_band": {k: rd(x) for k, x in band([b["share"][0] for b in bs]).items()},
                "share2_band": {k: rd(x) for k, x in band([b["share"][1] for b in bs]).items()},
                "flat1": rd(R["flat"]), "flat1_band": {k: rd(x) for k, x in band([b["flat"] for b in bs]).items()},
                "phi2_zero_h": [rd(z, 1) for z in R["z2"]],
                "phi2_zero_h_boot": {"n_with_zero_in_84_96": sum(any(84 <= z <= 96 for z in b["z2"]) for b in bs),
                                     "n_any_zero": sum(bool(b["z2"]) for b in bs),
                                     "first_zero_q": {k: rd(q([b["z2"][0] for b in bs if b["z2"]], p), 1)
                                                      for k, p in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9))}},
                "neg_eig_mass": rd(R["neg"], 6)}}
    if full:
        o["grid"] = {"u": rd(UG, 4), "h": rd(hg, 2)}
        o["mu"] = {"est": rd(R["mu"], 4), "lo": rd([q([b["mu"][i] for b in bs], 0.1) for i in range(NG)], 4),
                   "hi": rd([q([b["mu"][i] for b in bs], 0.9) for i in range(NG)], 4)}
        o["sd_e"] = rd(np.sqrt(np.clip(np.diag(R["Cp"]), 0, None)), 4)
        o["R1"]["rho_map"] = {"est": rd(R["rho"], 3),
                              "lo": rd(np.quantile(np.stack([b["rho"] for b in bs]), 0.1, axis=0), 3),
                              "hi": rd(np.quantile(np.stack([b["rho"] for b in bs]), 0.9, axis=0), 3)}
        o["R1"]["rho_curve_108"] = {"h": rd(hg, 2), "est": rd(R["r_end"]),
                                    "lo": rd([q([b["r_end"][i] for b in bs], 0.1) for i in range(NG)]),
                                    "hi": rd([q([b["r_end"][i] for b in bs], 0.9) for i in range(NG)])}
        o["R2"]["curve"] = {"a": rd(R["r2_a"], 0), "est": rd(R["r2"]),
                            "lo": rd([q([b["r2"][i] for b in bs], 0.1) for i in range(len(R["r2_a"]))]),
                            "hi": rd([q([b["r2"][i] for b in bs], 0.9) for i in range(len(R["r2_a"]))])}
        for k in (1, 2):
            P = np.stack([b[f"phi{k}"] for b in bs])
            o["R3"][f"phi{k}"] = {"h": rd(hg, 2), "est": rd(R[f"phi{k}"]), "lo": rd(np.quantile(P, 0.1, axis=0)),
                                  "hi": rd(np.quantile(P, 0.9, axis=0))}
        o["R3"]["lam"] = rd(R["lam"][:6], 6)
    return o


def verdict(S):
    s1 = S["R3"]["share"][0]
    fl = S["R3"]["flat1"]
    lo18 = S["R1"]["rho_vs_108"]["18"]["lo"]
    if s1 >= 0.80 and fl >= 0.6 and lo18 is not None and lo18 >= 0.5:
        v = "시즌 배수 하나로 대부분 설명된다"
    elif s1 < 0.65 or fl < 0.4:
        v = "시즌 안 변화가 크다"
    else:
        v = "시즌 안 변화가 일부 있다"
    return {"sentence": v, "share1": s1, "flat1": fl, "rho_18_108_lo": lo18,
            "rule": "대부분: 몫 ≥ 0.80·평평함 ≥ 0.6·ρ(18,108) 띠 아래 끝 ≥ 0.5 (셋 다) / 크다: 몫 < 0.65 또는 평평함 < 0.4 / 그 사이 일부"}


def point(D, names, b, ax, grid, tr):
    for d in D.values():
        d["u"] = ax.u(d["h"])
    mu, C = estimate(D, names, b, grid)
    R = readouts(mu, C, ax, grid, tr)
    return {"b": b, "share": rd(R["share"][:3]), "flat1": rd(R["flat"]), "rho_vs_108": {h: rd(v) for h, v in R["r1"].items()},
            "R2_a12": rd(R["r2"][0]), "phi2_zero_h": [rd(z, 1) for z in R["z2"]]}


def windows(ax, b):
    """판독 점마다 커널 ±b가 h로 얼마인지 (u 축 폭 b가 h에서 얼마나 넓은지)."""
    return {str(h): [rd(float(ax.h(max(ax.u(h) - b, 0))), 1), rd(float(ax.h(min(ax.u(h) + b, 1))), 1)] for h in H_R1 + [H_END]}


def extra(raw, tr, D0, seasons, ax, bsel, cv, a):
    """사전 고정 밖 확인 (주 결과를 본 뒤 정함 — 판정에 쓰지 않음)."""
    ex = {"about": "사전 고정 밖 (주 판을 본 뒤 정함, 판정에 쓰지 않음). X1 폭 의존 — LOSO 오차가 후보 7개 모두 최소의 5% 안이라 "
                   "규칙이 가장 넓은 폭을 골랐다. X2 굵은 오차 빼기 — |P − g|/σ > 5 칸 (대부분 마지막 날 C 등급, 시각이 늦게 잡힌 것으로 보임)이 "
                   "짝 곱을 크게 흔든다."}
    ex["X1_sweep"] = [point(D0, seasons, b, ax, UG, tr) for b in BWS]
    barg = min(BWS, key=lambda b: cv[str(b)]["err"])
    R = run(D0, seasons, barg, ax, UG, tr, a.boot, a.seed)
    S = summarize(R, full=False)
    S["b"], S["would_say"] = barg, verdict(S)["sentence"]
    S["R3"]["phi1"] = {"h": rd(R["hg"], 1), "est": rd(R["phi1"])}
    S["R3"]["phi2"] = {"h": rd(R["hg"], 1), "est": rd(R["phi2"])}
    S["R1"]["rho_curve_108"] = {"h": rd(R["hg"], 1), "est": rd(R["r_end"])}
    S["R2"]["curve"] = {"a": rd(R["r2_a"], 0), "est": rd(R["r2"])}
    S["windows_h"] = windows(ax, barg)
    ex["X1_argmin"] = S
    print("X1", barg, S["R3"]["share"][:2], S["R3"]["flat1"], S["R1"]["rho_vs_108"]["18"], S["would_say"], flush=True)
    DX = prep(raw, tr, zmax=5)
    gone = {s: int(len(D0[s]["e"]) - len(DX[s]["e"])) for s in seasons}
    for d in DX.values():
        d["u"] = ax.u(d["h"])
    cvx = {str(b): loso_err(DX, seasons, b)[0] for b in BWS}
    ex_min = min(cvx.values())
    bx = max(b for b in BWS if cvx[str(b)] <= 1.05 * ex_min)
    R = run(DX, seasons, bx, ax, UG, tr, a.boot, a.seed)
    S = summarize(R, full=False)
    S["b"], S["would_say"] = bx, verdict(S)["sentence"]
    S["removed"] = {"total": sum(gone.values()), "by_season": {s: v for s, v in gone.items() if v}}
    S["loso_rel"] = {k: rd(v / ex_min, 4) for k, v in cvx.items()}
    S["R3"]["phi1"] = {"h": rd(R["hg"], 1), "est": rd(R["phi1"])}
    S["R3"]["phi2"] = {"h": rd(R["hg"], 1), "est": rd(R["phi2"])}
    S["R1"]["rho_curve_108"] = {"h": rd(R["hg"], 1), "est": rd(R["r_end"])}
    S["R2"]["curve"] = {"a": rd(R["r2_a"], 0), "est": rd(R["r2"])}
    S["windows_h"] = windows(ax, bx)
    S["sweep"] = [point(DX, seasons, b, ax, UG, tr) for b in BWS]
    ex["X2_trim5"] = S
    print("X2", bx, S["R3"]["share"][:2], S["R3"]["flat1"], S["R1"]["rho_vs_108"]["18"], S["would_say"], flush=True)
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(AN, "flood_cov.json"))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="", help="쉼표로 판 고르기 (main,S1..S6) — 시험용")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    pr = json.load(open(os.path.join(AN, "prior_dev.json"), encoding="utf-8"))
    tr = pr["transform"]
    seasons = sorted(tr["season_levels"], key=snum)
    raw = load_raw(seasons)
    D0 = prep(raw, tr)
    ax = Axis(D0)
    for d in D0.values():
        d["u"] = ax.u(d["h"])
    res = {"about": "할 일 4.2b 시즌 안 침수 속도 변화 — 두 시각 상관 지도 (서술). 사전 고정 커밋 " + PREREG +
                    " (docs/solo-raid-analysis.md 4.2b). 한섭 값 칸, e = P/g − 1 (g = prior_dev 공통 꼴), "
                    "같은 시즌·다른 사람 짝 곱의 국소 선형 평활, 축 = 시즌 같은 무게 자료 밀도 u(h).",
           "tool_head": subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "code_sha1": hashlib.sha1(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12],
           "prior": {"file": "analysis/prior_dev.json", "form": tr["form"], "theta": tr["theta"], "sigma": tr["sigma"]},
           "boot": a.boot, "seed": a.seed, "seasons": seasons,
           "data": {s: {"H": D0[s]["H"], "n": int(len(D0[s]["e"])), "persons": int(len(set(D0[s]["pid"].tolist()))),
                        "pairs_diff_person": int(((D0[s]["pid"][:, None] != D0[s]["pid"][None, :]).sum()) // 2),
                        "n_u_lt_0.1": int((D0[s]["u"] < 0.1).sum())} for s in seasons},
           "axis": {"h_at_u": {f"{p:.2f}": rd(float(ax.h(p)), 2) for p in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
                    "u_at_h": {str(h): rd(float(ax.u(h)), 4) for h in (6, 12, 18, 24, 42, 48, 66, 72, 90, 96, 108, 117)}}}
    # 폭 고르기 (주 판)
    cv = {}
    for b in BWS:
        e, bad = loso_err(D0, seasons, b)
        cv[str(b)] = {"err": e, "nan_pairs": bad}
        print("cv", b, e, bad, flush=True)
    emin = min(v["err"] for v in cv.values())
    bsel = max(b for b in BWS if cv[str(b)]["err"] <= 1.05 * emin)
    res["bandwidth"] = {"candidates": BWS, "loso": {k: {"err": rd(v["err"], 8), "rel": rd(v["err"] / emin, 4), "nan_pairs": v["nan_pairs"]}
                                                     for k, v in cv.items()},
                        "argmin": min(BWS, key=lambda b: cv[str(b)]["err"]), "chosen": bsel,
                        "rule": "LOSO 오차 최소의 5% 안에서 가장 넓은 폭"}
    print("b =", bsel, flush=True)
    runs = {}
    if not only or "main" in only:
        R = run(D0, seasons, bsel, ax, UG, tr, a.boot, a.seed)
        runs["main"] = R
        S = summarize(R)
        S["R4"] = {"scores": {s: {**{k: (rd(v, 4) if isinstance(v, float) else v) for k, v in R["scores"][s].items()},
                                  "ln_ms": rd(math.log(tr["season_levels"][s]), 4) if isinstance(tr["season_levels"][s], (int, float)) else None,
                                  "special": SPECIAL.get(s)} for s in seasons}}
        S["windows_h"] = windows(ax, bsel)
        res["main"] = S
        res["verdict"] = verdict(S)
        print("main", json.dumps(res["verdict"], ensure_ascii=False), flush=True)
    ref = {"phi1": runs["main"]["phi1"], "phi2": runs["main"]["phi2"]} if "main" in runs else None
    sens = {}
    # S1 C 시각 당기기
    if not only or "S1" in only:
        D1 = prep(raw, tr, shift_c=1.5)
        sens["S1"] = ("등급 C뿐인 칸 h −1.5시간 (0.25 바닥)", D1, seasons, bsel, ax, UG, True)
    if not only or "S2" in only:
        xe = np.concatenate([D0[s]["x"][D0[s]["u"] < 0.1] for s in seasons])
        lo, hi = np.quantile(xe, [0.1, 0.9])
        keep = {s: (raw[s]["x"] >= lo) & (raw[s]["x"] <= hi) for s in seasons}
        D2 = prep(raw, tr, keep=keep)
        res["S2_x_range"] = {"q10": rd(lo, 4), "q90": rd(hi, 4), "n_early": int(len(xe)),
                             "kept": int(sum(len(D2[s]["e"]) for s in seasons)), "all": int(sum(len(D0[s]["e"]) for s in seasons))}
        sens["S2"] = ("x를 이른 자료(u < 0.1) x 10~90% 분위 범위로 줄임", D2, seasons, bsel, ax, UG, True)
    if not only or "S3" in only:
        sens["S3"] = ("S28·S38 빼기", D0, [s for s in seasons if s not in ("S28", "S38")], bsel, ax, UG, True)
    if not only or "S4" in only:
        sens["S4"] = ("시즌 무게 없이 (짝 수 그대로)", D0, seasons, bsel, ax, UG, False)
    if not only or "S6" in only:
        sens["S6"] = ("S19 빼기 (탐색과 같은 22시즌)", D0, [s for s in seasons if s != "S19"], bsel, ax, UG, True)
    if not only or "S5" in only:
        axh = Axis(D0, "h")
        D5 = prep(raw, tr)
        for d in D5.values():
            d["u"] = d["h"]
        cv5 = {}
        for b in BWS:
            e, bad = loso_err(D5, seasons, b * 120)
            cv5[str(b * 120)] = {"err": e, "nan_pairs": bad}
            print("cv5", b * 120, e, bad, flush=True)
        e5 = min(v["err"] for v in cv5.values())
        b5 = max(b * 120 for b in BWS if cv5[str(b * 120)]["err"] <= 1.05 * e5)
        res["S5_bandwidth"] = {"loso": {k: {"err": rd(v["err"], 8), "rel": rd(v["err"] / e5, 4)} for k, v in cv5.items()}, "chosen_h": b5}
        sens["S5"] = (f"축 h 그대로 (폭 {b5:g}시간, 후보 × 120)", D5, seasons, b5, axh, UG * 120, True)
    res["sensitivity"] = {}
    for k in ("S1", "S2", "S3", "S4", "S5", "S6"):
        if k not in sens:
            continue
        what, D, names, b, axk, grid, sw = sens[k]
        refk = ref
        if k == "S5" and ref is not None:   # 격자 h가 다르므로 주 판 φ를 h로 옮겨 부호를 맞춘다
            hg5 = grid
            refk = {"phi1": np.interp(hg5, runs["main"]["hg"], ref["phi1"]), "phi2": np.interp(hg5, runs["main"]["hg"], ref["phi2"])}
        R = run(D, names, b, axk, grid, tr, a.boot, a.seed, sw, refk)
        S = summarize(R, full=False)
        S["R1"]["rho_curve_108"] = {"h": rd(R["hg"], 1), "est": rd(R["r_end"])}
        S["R3"]["phi1"] = {"h": rd(R["hg"], 1), "est": rd(R["phi1"])}
        S["R3"]["phi2"] = {"h": rd(R["hg"], 1), "est": rd(R["phi2"])}
        S["what"], S["b"], S["seasons_n"] = what, b, len(names)
        S["would_say"] = verdict(S)["sentence"]
        res["sensitivity"][k] = S
        print(k, what, S["R3"]["share"][:2], S["R3"]["flat1"], S["R1"]["rho_vs_108"]["18"], S["would_say"], flush=True)
    if not only or "X" in only:
        res["extra"] = extra(raw, tr, D0, seasons, ax, bsel, cv, a)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("→", a.out)


if __name__ == "__main__":
    main()
