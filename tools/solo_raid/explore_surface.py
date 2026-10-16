"""솔로레이드 2단계 탐색 2.2·2.3: 서버별 점수–% 곡면, 잡음 크기, 게시 지연 치우침을 잰다 (모델 판독 0, 요청 0).

python3 -I tools/solo_raid/explore_surface.py STATE [--out DIR] [--boot 300]

측정 설계용이다. 예측 모델이나 성능 주장을 만들지 않는다 (docs/solo-raid-analysis.md §2).
표준 라이브러리만 쓴다. 곡면 적합·강건 회귀는 직접 구현했다 (post_outlier.py가 이 파일의 함수를 가져다 쓴다).

입력 (읽기만): STATE/core.jsonl · obs.jsonl(같은 사진 img·순위 줄) · season.json(c_start·end) · truth_points.jsonl
출력: 표준 출력에 표. --out DIR이면 DIR/surface_KR_A.svg·surface_KR_AC.svg 그림.

정의
- L = log10(%) · S = 점수(억, core score = 0.1억 반올림)
- he = 곡면에 넣는 시간: min(h, H_END), 1 아래는 1. 확실한 종료 후(day_lo == "post")는 H_END. H_END = end − c_start (시간)
- 시간 모양 g(he): sqrt → sqrt(he/H_END) − 1, ln → ln(he/H_END), lin → he/H_END − 1 (모두 종료 = 0)
- 기본 곡면 (SurfL, 서버마다):  L = rcs(S/100; 점수 분위 매듭 5개) + c·g_sqrt(he)
  = 「같은 점수의 log %가 시간에 따라 c·g만큼 평행이동」. Tukey bisquare 강건 회귀
  비교용 SurfS: S = rcs(x) + 시간항 (x = L 또는 %), 시간항 add(g) · int(g, g·x) · full(g, g·x, g²)
- 적합 줄: 기본 집합 use · use_indep · 시각 등급 A · 점수 있음 · h ≥ 1
- 잡음 크기: 사람 하나씩 빼고 맞춘(LOPO) 잔차의 1.4826·MAD. 점수 단위(억)와 log % 단위(자릿수) 둘 다
- C 줄(게시 시각만, h = 상한)은 맞추지 않고 대어 본다. 지연 d시간 가정: he = max(h_hi − d, h_lo, 1)
- 상위 7줄 시각 구간(top7): 같은 서버 1~7위 점수는 시간에 따라 줄지 않는다. A 사진들의 1~7위와 견주어
  어긋남이 가장 적은 촬영 시각 구간을 구한다 (A 사진 하나씩 빼고 포함률로 방법을 확인)
"""
import argparse
import collections
import datetime as dt
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post_time import top7_bracket  # noqa: E402  (규칙 한 곳: post_time.py의 h_lo_top7·h_hi_top7과 같다)

TF = "%Y-%m-%d %H:%M:%S"


# ---------- 읽기 ----------

import importlib.util as _ilu  # noqa: E402
_sp = _ilu.spec_from_file_location("split", os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.py"))
split = _ilu.module_from_spec(_sp)
_sp.loader.exec_module(split)

def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load(st):
    core = jl(os.path.join(st, "core.jsonl"))
    obs = jl(os.path.join(st, "obs.jsonl"))
    with open(os.path.join(st, "season.json"), encoding="utf-8") as f:
        season = json.load(f)
    h_end = h_end_of(season)
    if len(core) and any(x.get("oi") is not None and x["oi"] >= len(obs) for x in core):
        raise SystemExit("core.jsonl의 oi가 obs.jsonl과 맞지 않는다")
    return core, obs, season, h_end


def h_end_of(season):
    cs = season.get("c_start") or ((dt.date.fromisoformat(season["start"][:10]) + dt.timedelta(days=2)).isoformat() + " 05:00:00")
    c0 = dt.datetime.strptime(cs, TF)   # c_start 없으면 start 날짜 + 2일 05:00 (2026-10-16, 다른 도구와 같게)
    end = dt.datetime.strptime(season["end"], TF)
    return round((end - c0).total_seconds() / 3600)   # 04:59:59 → 120


def h_eff(x, h_end, h=None):
    """곡면에 넣는 시간. 확실한 종료 후는 H_END, 그 밖은 min(h, H_END), 아래는 1로 막는다."""
    if h is None:
        if x.get("day_lo") == "post":
            return float(h_end)
        h = x["h"]
    return max(1.0, min(float(h), float(h_end)))


# ---------- 통계 ----------

def median(v):
    v = sorted(v)
    n = len(v)
    if not n:
        return float("nan")
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def quant(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = q * (len(v) - 1)
    i = int(math.floor(k))
    j = min(i + 1, len(v) - 1)
    return v[i] + (v[j] - v[i]) * (k - i)


def mad(r):
    m = median(r)
    return 1.4826 * median([abs(x - m) for x in r])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def solve(A, b):
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            raise ValueError("singular")
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                if f:
                    for k in range(c, n + 1):
                        M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def wls(X, y, w, ridge=1e-6):
    p = len(X[0])
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for xi, yi, wi in zip(X, y, w):
        if wi <= 0:
            continue
        for a in range(p):
            wa = wi * xi[a]
            b[a] += wa * yi
            Aa = A[a]
            for c in range(a, p):
                Aa[c] += wa * xi[c]
    for a in range(p):
        for c in range(a):
            A[a][c] = A[c][a]
        if a:
            A[a][a] += ridge
    return solve(A, b)


def robust_fit(X, y, c=4.685, iters=50):
    """LAD 근사(IRLS)로 시작해 Tukey bisquare로 맞춘다. (계수, 척도, 가중치)."""
    w = [1.0] * len(y)
    beta = wls(X, y, w)
    for _ in range(20):
        r = [yi - dot(xi, beta) for xi, yi in zip(X, y)]
        w = [1.0 / max(abs(x), 1e-4) for x in r]
        beta = wls(X, y, w)
    for _ in range(iters):
        r = [yi - dot(xi, beta) for xi, yi in zip(X, y)]
        s = max(mad(r), 1e-6)
        w = [(1 - (x / (c * s)) ** 2) ** 2 if abs(x) < c * s else 0.0 for x in r]
        nb = wls(X, y, w)
        done = max(abs(a - b) for a, b in zip(nb, beta)) < 1e-10
        beta = nb
        if done:
            break
    r = [yi - dot(xi, beta) for xi, yi in zip(X, y)]
    return beta, mad(r), w


def theil_sen(pairs, min_dx=0.05):
    sl = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            dx = pairs[j][0] - pairs[i][0]
            if abs(dx) > min_dx:
                sl.append((pairs[j][1] - pairs[i][1]) / dx)
    return (median(sl) if sl else None), len(sl)


# ---------- 곡면 ----------

def rcs(x, k):
    """제한 3차 스플라인 (Harrell). 열: x, 그리고 len(k)−2개."""
    t, t1 = k[-1], k[-2]
    d = (t - k[0]) ** 2
    cube = lambda u: u ** 3 if u > 0 else 0.0
    out = [x]
    for j in range(len(k) - 2):
        out.append((cube(x - k[j]) - cube(x - t1) * (t - k[j]) / (t - t1) + cube(x - t) * (t1 - k[j]) / (t - t1)) / d)
    return out


G = {
    "sqrt": (lambda he, H: math.sqrt(he / H) - 1.0, lambda g, H: H * (g + 1.0) ** 2 if g > -1 else 0.0),
    "ln": (lambda he, H: math.log(he / H), lambda g, H: H * math.exp(g)),
    "lin": (lambda he, H: he / H - 1.0, lambda g, H: H * (g + 1.0)),
}
QS = {3: (.1, .5, .9), 4: (.05, .35, .65, .95), 5: (.05, .275, .5, .725, .95)}


def pts_of(rows, h_end):
    return [(x["pct"], h_eff(x, h_end), x["score"]) for x in rows]


class SurfS:
    """비교용: S = rcs(x) + 시간항. x = L(log10 %) 또는 P(%)."""
    kind = "S"

    def __init__(self, axis="L", time="sqrt", terms="add", nk=5, H=120):
        self.axis, self.time, self.terms, self.nk, self.H = axis, time, terms, nk, H

    def x(self, pct):
        return math.log10(pct) if self.axis == "L" else pct

    def feat(self, pct, he):
        x = self.x(pct)
        g = G[self.time][0](he, self.H)
        f = [1.0] + rcs(x, self.knots) + [g]
        if self.terms in ("int", "full"):
            f.append(g * x)
        if self.terms == "full":
            f.append(g * g)
        return f

    def fit(self, pts):
        xs = [self.x(p) for p, _, _ in pts]
        self.knots = [quant(xs, q) for q in QS[self.nk]]
        self.beta, self.scale, self.w = robust_fit([self.feat(p, he) for p, he, _ in pts], [s for _, _, s in pts])
        return self

    def pred_S(self, pct, he, near=None):
        return dot(self.feat(pct, he), self.beta)

    def dS_dL(self, pct, he, e=0.01):
        a, b = 10 ** (math.log10(pct) - e), 10 ** (math.log10(pct) + e)
        return (self.pred_S(b, he) - self.pred_S(a, he)) / (2 * e)

    def resid(self, pct, S, he):
        """(점수 잔차, log % 잔차). log % 잔차는 점수 잔차 / 기울기(부호: 관측 %가 곡면보다 크면 +)."""
        eS = S - self.pred_S(pct, he)
        sl = self.dS_dL(pct, he)
        return eS, (-eS / sl if abs(sl) > 1 else None)


class SurfL:
    """기본: L = rcs(S/100) + c·g(he).  fixed_c를 주면 c를 고정하고 모양만 맞춘다."""
    kind = "L"

    def __init__(self, time="sqrt", nk=5, H=120, fixed_c=None, terms="add"):
        self.time, self.nk, self.H, self.fixed_c, self.terms = time, nk, H, fixed_c, terms

    def g(self, he):
        return G[self.time][0](he, self.H)

    def shape(self, S):
        return [1.0] + rcs(S / 100.0, self.knots)

    def feat(self, S, he):
        f = self.shape(S)
        if self.fixed_c is None:
            f.append(self.g(he))
            if self.terms == "int":
                f.append(self.g(he) * S / 100.0)
        return f

    def fit(self, pts):
        zs = [s / 100.0 for _, _, s in pts]
        self.knots = [quant(zs, q) for q in QS[self.nk]]
        X = [self.feat(s, he) for _, he, s in pts]
        y = [math.log10(p) - (self.fixed_c * self.g(he) if self.fixed_c is not None else 0.0) for p, he, _ in pts]
        self.beta, self.scale, self.w = robust_fit(X, y)
        self.c = self.fixed_c if self.fixed_c is not None else self.beta[len(self.shape(0))]
        self.S_rng = (min(s for _, _, s in pts), max(s for _, _, s in pts))
        self.h_rng = (min(he for _, he, _ in pts), max(he for _, he, _ in pts))
        self.h_sup = quant([he for _, he, _ in pts], 0.05)   # 시간 지지 범위 아래 끝 (맨 앞 몇 줄만으로 정해지는 곳은 뺀다)
        return self

    def F(self, S):
        return dot(self.shape(S), self.beta[:len(self.shape(S))])

    def pred_L(self, S, he):
        if self.terms == "int" and self.fixed_c is None:
            return dot(self.feat(S, he), self.beta)
        return self.F(S) + self.c * self.g(he)

    def pred_S(self, pct, he, near=None):
        """L(S, he) = log10 pct가 되는 S (격자 0~450억). 여러 개면 near에 가까운 것."""
        y = math.log10(pct)
        best, prev = None, None
        ref = near if near is not None else 200.0
        for k in range(0, 91):
            S = 5.0 * k
            v = self.pred_L(S, he) - y
            if prev is not None and (prev[1] >= 0) != (v >= 0):
                a, b, fa = prev[0], S, prev[1]
                for _ in range(25):
                    m = 0.5 * (a + b)
                    fm = self.pred_L(m, he) - y
                    if (fm >= 0) == (fa >= 0):
                        a, fa = m, fm
                    else:
                        b = m
                c = 0.5 * (a + b)
                if best is None or abs(c - ref) < abs(best - ref):
                    best = c
            prev = (S, v)
        return best

    def resid_L(self, pct, S, he):
        return math.log10(pct) - self.pred_L(S, he)

    def resid(self, pct, S, he):
        eL = math.log10(pct) - self.pred_L(S, he)
        sp = self.pred_S(pct, he, near=S)
        return (S - sp if sp is not None else None), eL

    def he_star(self, pct, S):
        """잔차 0이 되는 시각 (막지 않음, 1~H_END 밖이면 그 끝)."""
        if self.c <= 0:
            return None
        g = (math.log10(pct) - self.F(S)) / self.c
        g = max(G[self.time][0](1.0, self.H), min(0.0, g))
        return G[self.time][1](g, self.H)

    def band_resid(self, pct, S, lo, hi):
        """시각 구간 [lo, hi] 안에서 가장 잘 맞는 시각과 그때의 (점수 잔차, log % 잔차)."""
        lo, hi = max(1.0, min(lo, self.H)), max(1.0, min(hi, self.H))
        hs = self.he_star(pct, S)
        he = hi if hs is None else min(max(hs, lo), hi)
        eS, eL = self.resid(pct, S, he)
        return he, eS, eL

    def dS_dL(self, pct, he, e=0.01):
        a = self.pred_S(10 ** (math.log10(pct) - e), he)
        b = self.pred_S(10 ** (math.log10(pct) + e), he)
        return None if a is None or b is None else (b - a) / (2 * e)


def fit_rows(core, server, h_end, grades=("A",)):
    """곡면 적합 줄 (기본 집합·독립·점수 있음·등급)."""
    out = []
    for x in core:
        if not (x["use"] and x["use_indep"] and x["server"] == server and x["t_grade"] in grades):
            continue
        if x["score"] is None or not x["pct"] or x["pct"] <= 0:
            continue
        if x["day_lo"] != "post" and x["h"] < 1:
            continue
        out.append(x)
    return out


def lopo_models(rows, h_end, make):
    """{person: 그 사람을 뺀 곡면}"""
    out = {}
    for p in sorted(set(x["person"] for x in rows)):
        tr = [x for x in rows if x["person"] != p]
        try:
            out[p] = make().fit(pts_of(tr, h_end))
        except ValueError:
            out[p] = None
    return out


def lopo(rows, h_end, make):
    ms = lopo_models(rows, h_end, make)
    eS, eL = [], []
    for x in rows:
        m = ms[x["person"]]
        if m is None:
            eS.append(None)
            eL.append(None)
            continue
        a, b = m.resid(x["pct"], x["score"], h_eff(x, h_end))
        eS.append(a)
        eL.append(b)
    return eS, eL, ms


# ---------- top7 ----------

def top7_snapshots(core, obs):
    """사진별 top7 (시즌 in·exclude 없음·본인 줄 아님, 1~7위 점수 2개 이상)."""
    im = collections.defaultdict(list)
    for x in core:
        if x["src"] == "photo":
            im[obs[x["oi"]]["img"]].append(x)
    snaps = {}
    for img, v in im.items():
        if obs[v[0]["oi"]]["season"] != "in" or any(x["exclude"] for x in v):
            continue
        ranks = {}
        for x in v:
            o = obs[x["oi"]]
            if o["rank"] and o["score"] and 1 <= o["rank"] <= 7 and not o["self"]:
                ranks[o["rank"]] = o["score"]
        if len(ranks) < 2:
            continue
        x0 = v[0]
        snaps[img] = dict(server=x0["server"], grade=x0["t_grade"], h=x0["h"], h_lo=x0["h_lo"], h_hi=x0["h_hi"],
                          ranks=ranks, oi=[x["oi"] for x in v])
    return snaps


def top7_refs(snaps):
    return {sv: [(s["h"], s["ranks"]) for s in snaps.values() if s["server"] == sv and s["grade"] == "A"]
            for sv in set(s["server"] for s in snaps.values()) if sv}


# ---------- 그림 ----------

def svg_scatter(path, title, series, curves, xlab, ylab, xr, yr):
    """x는 log10 %. series: [(이름, 색, [(pct, S)], 모양 o|x)], curves: [(이름, 색, [(pct, S)])]"""
    W, Hh, ml, mr, mt, mb = 780, 500, 60, 180, 40, 50
    x0, x1 = xr
    y0, y1 = yr
    X = lambda v: ml + (math.log10(v) - x0) / (x1 - x0) * (W - ml - mr)
    Y = lambda v: Hh - mb - (v - y0) / (y1 - y0) * (Hh - mt - mb)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{Hh}" fill="#fff"/>', f'<text x="{ml}" y="22" font-size="14">{title}</text>',
         f'<rect x="{ml}" y="{mt}" width="{W-ml-mr}" height="{Hh-mt-mb}" fill="none" stroke="#888"/>']
    for t in (0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 20, 30):
        if x0 <= math.log10(t) <= x1:
            o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{mt}" y2="{Hh-mb}" stroke="#eee"/>')
            o.append(f'<text x="{X(t):.1f}" y="{Hh-mb+14}" text-anchor="middle">{t:g}</text>')
    for t in range(int(y0), int(y1) + 1, 50):
        o.append(f'<line x1="{ml}" x2="{W-mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="#eee"/>')
        o.append(f'<text x="{ml-4}" y="{Y(t)+4:.1f}" text-anchor="end">{t}</text>')
    o.append(f'<text x="{(ml+W-mr)/2}" y="{Hh-12}" text-anchor="middle">{xlab}</text>')
    o.append(f'<text x="14" y="{(mt+Hh-mb)/2}" transform="rotate(-90 14 {(mt+Hh-mb)/2})" text-anchor="middle">{ylab}</text>')
    ok = lambda a, b: x0 <= math.log10(a) <= x1 and y0 <= b <= y1
    for _, col, pts in curves:
        pts = [p for p in pts if ok(*p)]
        if len(pts) > 1:
            o.append(f'<polyline points="{" ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts)}" fill="none" '
                     f'stroke="{col}" stroke-width="1.6"/>')
    for _, col, pts, shape in series:
        for a, b in pts:
            if not ok(a, b):
                continue
            if shape == "o":
                o.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="3" fill="{col}" fill-opacity="0.75"/>')
            else:
                cx, cy = X(a), Y(b)
                o.append(f'<path d="M{cx-3:.1f},{cy-3:.1f}L{cx+3:.1f},{cy+3:.1f}M{cx-3:.1f},{cy+3:.1f}L{cx+3:.1f},{cy-3:.1f}" '
                         f'stroke="{col}" stroke-width="1.3"/>')
    ly = mt + 8
    for lab, col in [(c[0], c[1]) for c in curves] + [(s[0], s[1]) for s in series]:
        o.append(f'<rect x="{W-mr+10}" y="{ly-8}" width="10" height="10" fill="{col}"/>')
        o.append(f'<text x="{W-mr+24}" y="{ly+1}">{lab}</text>')
        ly += 15
    o.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(o))


# ---------- 보고 ----------

HBINS = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120.001)]
PBINS = [(0, 0.5), (0.5, 1), (1, 2), (2, 3.5), (3.5, 10), (10, 101)]
COL = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#111111"]


def fmt(v, d=1):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"


def desc(r, d=1):
    r = [x for x in r if x is not None]
    if not r:
        return "n=0"
    a = [abs(x) for x in r]
    return f"n={len(r):3d} 중앙 {median(r):+.{d}f} MAD {mad(r):.{d}f} |e|중앙 {median(a):.{d}f} |e|90% {quant(a, .9):.{d}f}"


def default_surface(H, nk=5, fixed_c=None):
    return SurfL("sqrt", nk, H, fixed_c)


def section_compare(rows, H):
    print("\n## 1. 곡면 모양 비교 (KR A 독립, 사람 하나씩 뺀 잔차; 점수 억 | log10 %)")
    core_i = [i for i, x in enumerate(rows) if h_eff(x, H) >= 24 and x["pct"] <= 10]
    print(f"  (핵심 영역 = he ≥ 24h · % ≤ 10: {len(core_i)}줄)")
    cands = [("L = F(S) + c·g  sqrt [기본]", lambda: SurfL("sqrt", 5, H)),
             ("L = F(S) + c·g  ln", lambda: SurfL("ln", 5, H)),
             ("L = F(S) + c·g  lin", lambda: SurfL("lin", 5, H)),
             ("L = F(S) + g + g·S  sqrt", lambda: SurfL("sqrt", 5, H, terms="int")),
             ("L = F(S) + c·g  sqrt 매듭4", lambda: SurfL("sqrt", 4, H)),
             ("S = F(L) + g  sqrt", lambda: SurfS("L", "sqrt", "add", 5, H)),
             ("S = F(L) + g + g·L  sqrt", lambda: SurfS("L", "sqrt", "int", 5, H)),
             ("S = F(L) + g + g·L + g²  ln", lambda: SurfS("L", "ln", "full", 5, H)),
             ("S = F(%) + g  sqrt", lambda: SurfS("P", "sqrt", "add", 5, H)),
             ("S = F(%) + g + g·%  sqrt", lambda: SurfS("P", "sqrt", "int", 5, H)),
             ("S = F(%) + g + g·% + g²  ln", lambda: SurfS("P", "ln", "full", 5, H))]
    out = {}
    for name, mk in cands:
        eS, eL, _ = lopo(rows, H, mk)
        out[name] = (eS, eL)
        aS = [abs(e) for e in eS if e is not None]
        aL = [abs(e) for e in eL if e is not None]
        cS = [abs(eS[i]) for i in core_i if eS[i] is not None]
        cL = [abs(eL[i]) for i in core_i if eL[i] is not None]
        print(f"  {name:30s} 억 |e|중앙 {median(aS):4.1f} 90% {quant(aS,.9):5.1f} (핵심 {median(cS):4.1f}/{quant(cS,.9):5.1f})"
              f" | log% |e|중앙 {median(aL):.3f} 90% {quant(aL,.9):.3f} (핵심 {median(cL):.3f}/{quant(cL,.9):.3f})")
    print("  % 구간별 |e| 중앙(억): x=log% (S = F(L)+g+g·L) / x=% (S = F(%)+g+g·%) / 기본")
    a, b, c = (out["S = F(L) + g + g·L  sqrt"][0], out["S = F(%) + g + g·%  sqrt"][0], out["L = F(S) + c·g  sqrt [기본]"][0])
    for lo, hi in PBINS:
        k = [i for i, x in enumerate(rows) if lo <= x["pct"] < hi]
        f = lambda e: fmt(median([abs(e[i]) for i in k if e[i] is not None]))
        print(f"    {lo:>4}~{hi:<4}%: n={len(k):2d}  {f(a):>5} / {f(b):>5} / {f(c):>5}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int, default=300)
    split.add_final_arg(ap)
    args = ap.parse_args()
    split.guard_args(args)   # 시험 시즌 정답 감시 (할 일 3.0b)
    core, obs, season, H = load(args.state)
    print(f"# {season['name']} · H_END={H}h")

    rows = fit_rows(core, "KR", H)
    rows_jp = fit_rows(core, "JP", H)
    rows_gl = fit_rows(core, "GL", H)
    print(f"적합 줄(A·독립·점수): KR {len(rows)}줄/{len(set(x['person'] for x in rows))}명 (종료 후 {sum(x['day_lo']=='post' for x in rows)}) · "
          f"JP {len(rows_jp)}/{len(set(x['person'] for x in rows_jp))} · GL {len(rows_gl)} (GL은 맞추지 않음)")
    hs = sorted(h_eff(x, H) for x in rows)
    print(f"  KR he 분포: <12h {sum(h<12 for h in hs)} · 12~24 {sum(12<=h<24 for h in hs)} · 24~48 {sum(24<=h<48 for h in hs)} · "
          f"48~72 {sum(48<=h<72 for h in hs)} · 72~96 {sum(72<=h<96 for h in hs)} · 96~120 {sum(96<=h<120 for h in hs)} · 120 {sum(h>=120 for h in hs)}")

    section_compare(rows, H)

    # 2. 기본 곡면
    S = default_surface(H).fit(pts_of(rows, H))
    eS, eL, ms = lopo(rows, H, lambda: default_surface(H))
    sS, sL = mad([e for e in eS if e is not None]), mad(eL)
    print(f"\n## 2. 기본 곡면 KR: L = F(S) + c·(sqrt(he/{H}) − 1)")
    print(f"  c = {S.c:.3f} 자릿수 · 매듭(억) {[round(k*100) for k in S.knots]} · 내부 척도 {S.scale:.3f} · 가중치 0 {sum(w == 0 for w in S.w)}줄")
    cs = [m.c for m in ms.values() if m]
    print(f"  LOPO c 범위 {min(cs):.3f}~{max(cs):.3f}")
    print("  같은 점수의 % 배율(종료 대비) 10^(c·g): " +
          " ".join(f"{h}h ×{10 ** (S.c * S.g(h)):.2f}" for h in (6, 12, 24, 48, 72, 96, 120)))
    print("  곡면 점수(억), 행 %, 열 h (적합 점수 범위 밖은 -):")
    hh = [12, 24, 48, 72, 96, 120]
    print("    %      " + " ".join(f"{h:>6}" for h in hh))
    for p in (0.3, 0.5, 1, 2, 3, 5, 10, 20):
        vals = []
        for h in hh:
            v = S.pred_S(p, h)
            vals.append(f"{v:6.1f}" if v is not None and S.S_rng[0] <= v <= S.S_rng[1] else "     -")
        print(f"    {p:<6} " + " ".join(vals))
    print("  곡면 기울기 dS/dlog10%(억/자릿수):")
    for p in (0.5, 1, 3, 10):
        print(f"    {p:<6} " + " ".join(f"{fmt(S.dS_dL(p, h)):>6}" for h in hh))
    tp = [p for p in jl(os.path.join(args.state, "truth_points.jsonl")) if p["server"] == "KR" and p["grade"] in ("A", "A2")]
    d = [S.resid(p["pct"], p["score"], H) for p in tp]
    print(f"  정답 점 KR A·A2 {len(tp)}개 − 곡면(he={H}): 억 {desc([a for a, _ in d])} | log% {desc([b for _, b in d], 3)}"
          f"  (정답 점 대부분이 적합 줄과 같은 관측 — 독립 검증 아님)")

    # 3. 단면
    print("\n## 3. 같은 시각 단면")
    im = collections.Counter(obs[x["oi"]]["img"] for x in core if x["src"] == "photo" and x["use"])
    print(f"  기본 집합에서 같은 사진에 % 줄이 2개 이상인 사진: {sum(1 for v in im.values() if v >= 2)}장 "
          f"(순위표 다른 줄은 1~7위 순위·점수뿐) → 사진 한 장 단면 기울기는 잴 수 없다. 대신 시간 창 안 A 줄의 Theil–Sen:")
    for a, b in HBINS + [(None, None)]:
        if a is None:
            v = [x for x in rows if x["day_lo"] == "post"]
            lab = "확실한 종료 후"
        else:
            v = [x for x in rows if a <= h_eff(x, H) < b and x["day_lo"] != "post"]
            lab = f"he {a:>3}~{b:>3.0f}h"
        v = [x for x in v if 0.3 <= x["pct"] <= 5]
        ts, npair = theil_sen([(math.log10(x["pct"]), x["score"]) for x in v])
        hm = median([h_eff(x, H) for x in v]) if v else None
        print(f"    {lab}: 0.3~5% A {len(v):2d}줄 {len(set(x['person'] for x in v)):2d}명 · Theil–Sen {fmt(ts):>6}억/자릿수 (쌍 {npair:3d})"
              f" · 곡면 1% 기울기(he 중앙 {fmt(hm, 0)}) {fmt(S.dS_dL(1, hm) if hm else None):>6}")

    # 4. 잡음
    print("\n## 4. 잡음 크기 (KR A, 사람 하나씩 뺀 잔차)")
    print(f"  전체: 억 {desc(eS)} | log% {desc(eL, 3)}  → log% MAD {sL:.3f} = % 배율 ×{10**sL:.2f}")
    for a, b in HBINS:
        k = [i for i, x in enumerate(rows) if a <= h_eff(x, H) < b and x["day_lo"] != "post"]
        print(f"    he {a:>3}~{b:>3.0f}h: 억 {desc([eS[i] for i in k])} | log% {desc([eL[i] for i in k], 3)}")
    k = [i for i, x in enumerate(rows) if x["day_lo"] == "post"]
    print(f"    종료 후     : 억 {desc([eS[i] for i in k])} | log% {desc([eL[i] for i in k], 3)}")
    for lo, hi in PBINS:
        k = [i for i, x in enumerate(rows) if lo <= x["pct"] < hi]
        print(f"    {lo:>4}~{hi:<4}%: 억 {desc([eS[i] for i in k])} | log% {desc([eL[i] for i in k], 3)}")
    print("  시간 해상도 = log% 잡음 / |dL/dh| (이만큼 시각이 틀려야 잡음 1배만큼 벗어난다):")
    print("    " + " ".join(f"{h}h {sL / (S.c * (S.g(h * 1.01) - S.g(h * 0.99)) / (0.02 * h)):.1f}h" for h in (12, 24, 48, 72, 96, 119)))
    # 판독 잡음 단서
    print("  판독 잡음 단서:")
    allp = [p for p in jl(os.path.join(args.state, "truth_points.jsonl")) if p.get("n", 1) >= 2]
    print(f"    종료 후 같은 정확 점수 묶음(n≥2) {len(allp)}개 중 %가 다르게 읽힌 묶음 {sum(1 for p in allp if len(set(p['pcts'])) > 1)}개")
    snaps = top7_snapshots(core, obs)
    refs = top7_refs(snaps)
    for sv in ("KR", "JP"):
        rr = sorted(refs.get(sv, []), key=lambda r: r[0])
        bad = tot = 0
        for i, (h, r) in enumerate(rr):
            for k_, v in r.items():
                tot += 1
                bef = [q[1][k_] for q in rr[:i] if k_ in q[1] and q[0] < h - 0.01]
                aft = [q[1][k_] for q in rr[i + 1:] if k_ in q[1] and q[0] > h + 0.01]
                if (bef and v < max(bef) and (not aft or v < min(aft))) or (aft and v > min(aft) and (not bef or v > max(bef))):
                    bad += 1
        print(f"    {sv} A 사진 top7 점수 {tot}개 중 앞뒤 사진과 단조 모순 {bad}개 (11자리 점수 판독 오류율 상한)")
    # 정답 점 % 판독 일치
    sp = collections.defaultdict(list)
    for x in core:
        if x["use"] and x["src"] == "photo" and x["day_lo"] == "post" and x["score_raw"] and x["score_raw"] > 50:
            sp[round(x["score_raw"], 4)].append(x["pct"])
    multi = [v for v in sp.values() if len(v) >= 2]
    print(f"    기본 집합 확실한 종료 후 같은 정확 점수 묶음 {len(multi)}개 · % 불일치 {sum(1 for v in multi if len(set(v)) > 1)}개")

    # 5. top7 시각 구간
    print(f"\n## 5. 상위 7줄(top7) 시각 구간: top7 사진 {len(snaps)}장 "
          f"{dict(sorted(collections.Counter(str(s['server']) + '/' + s['grade'] for s in snaps.values()).items()))}")
    br = {}
    for sv in ("KR", "JP"):
        ra = [(img, s) for img, s in snaps.items() if s["server"] == sv and s["grade"] == "A"]
        cov, wid = 0, []
        for img, s in ra:
            lo, hi, _ = top7_bracket(s["ranks"], [(t["h"], t["ranks"]) for j, t in ra if j != img])
            cov += lo - 0.01 <= s["h"] <= hi + 0.01
            wid.append(min(hi, H + 48) - max(lo, 0))
        print(f"  {sv} A 사진 하나씩 빼고: 실제 촬영 시각 포함 {cov}/{len(ra)} · 구간 폭 중앙 {median(wid):.1f}h · 90% {quant(wid, .9):.1f}h")
        cr = [(img, s) for img, s in snaps.items() if s["server"] == sv and s["grade"] in ("B", "C")]
        late, inside, early, viol, wd = [], 0, 0, 0, []
        for img, s in cr:
            lo, hi, v = top7_bracket(s["ranks"], refs.get(sv, []))
            br[img] = (lo, hi, v)
            viol += v > 0
            wd.append(min(hi, s["h_hi"]) - max(lo, 0))
            if s["h_hi"] > hi + 0.01:
                late.append((s["h_hi"] - hi, s["h_hi"]))
            elif s["h_hi"] < lo - 0.01:
                early += 1
            else:
                inside += 1
        dur = [1 for img, s in cr if s["h_hi"] <= H]
        print(f"  {sv} B·C 사진 {len(cr)}장(게시 종료 전 {len(dur)}): 게시 시각이 top7 구간 안 {inside} · 구간보다 늦음 {len(late)} "
              f"{[(round(a,1), round(b,1)) for a, b in sorted(late)]} (최소 지연, 게시 h) · 구간보다 이름(모순) {early} · 어긋남>0 {viol}"
              f" · 촬영 가능 구간 폭(게시 이전 부분) 중앙 {fmt(median(wd))}h")

    # 6. C 대어 보기
    print("\n## 6. C 줄을 A 곡면에 대어 보기 (KR, 기본 집합·독립). he = max(h_hi − d, h_lo, 1)")
    crow = [x for x in core if x["use"] and x["use_indep"] and x["server"] == "KR" and x["t_grade"] == "C"
            and x["score"] is not None and x["pct"]]
    during = [x for x in crow if x["h_hi"] <= H and x["day_lo"] != "post"]
    after = [x for x in crow if x["h_hi"] > H and x["day_lo"] != "post"]
    print(f"  C 독립 {len(crow)}줄: 게시 종료 전 {len(during)} · 게시 종료 후(촬영 미상) {len(after)} · 확실한 종료 후 {len(crow)-len(during)-len(after)}")

    def r_at(x, d):
        he = max(x["h_hi"] - d, x["h_lo"], 1.0)
        return S.resid(x["pct"], x["score"], min(he, H))
    a_dur = [i for i, x in enumerate(rows) if x["day_lo"] != "post"]
    print(f"  비교 A(종료 전, LOPO): 억 {desc([eS[i] for i in a_dur])} | log% {desc([eL[i] for i in a_dur], 3)}")
    print("  게시 종료 전 C:")
    best = None
    for d in (0, 0.5, 1, 2, 3, 4, 6, 8, 12, 24):
        rr = [r_at(x, d) for x in during]
        lm = median([b for _, b in rr])
        print(f"    d={d:>4}h: 억 {desc([a for a, _ in rr])} | log% {desc([b for _, b in rr], 3)}")
        if best is None or abs(lm - median([eL[i] for i in a_dur])) < best[1]:
            best = (d, abs(lm - median([eL[i] for i in a_dur])))
    # 중앙값을 A와 맞추는 d (사람 단위 부트스트랩)
    target = median([eL[i] for i in a_dur])

    def d_match(rows_c):
        lo_, hi_ = 0.0, 48.0
        f = lambda d: median([S.resid_L(x["pct"], x["score"], min(max(x["h_hi"] - d, x["h_lo"], 1.0), H)) for x in rows_c]) - target
        if f(0) >= 0:
            return 0.0
        if f(hi_) < 0:
            return hi_
        for _ in range(30):
            m = 0.5 * (lo_ + hi_)
            if f(m) < 0:
                lo_ = m
            else:
                hi_ = m
        return 0.5 * (lo_ + hi_)
    d0 = d_match(during)
    rng = random.Random(41)
    byp = collections.defaultdict(list)
    for x in during:
        byp[x["person"]].append(x)
    ps = sorted(byp)
    bs = []
    for _ in range(args.boot):
        smp = [x for p in (rng.choice(ps) for _ in ps) for x in byp[p]]
        bs.append(d_match(smp))
    print(f"  C 잔차 중앙값을 A와 맞추는 상수 지연 d = {d0:.1f}h (사람 단위 부트스트랩 {args.boot}회 90%: {quant(bs,.05):.1f}~{quant(bs,.95):.1f}h)")
    print("  시간대별 (d=0) C vs A LOPO, log%:")
    for a, b in HBINS:
        rc = [r_at(x, 0)[1] for x in during if a <= x["h_hi"] < b]
        ra = [eL[i] for i in a_dur if a <= h_eff(rows[i], H) < b]
        print(f"    {a:>3}~{b:>3.0f}h  C {desc(rc, 3)} | A {desc(ra, 3)}")
    print("  게시 종료 후(촬영 미상) C, he = H_END: " +
          f"억 {desc([S.resid(x['pct'], x['score'], H)[0] for x in after])} | log% {desc([S.resid(x['pct'], x['score'], H)[1] for x in after], 3)}")
    # 줄별 잔차 0 시각
    lag_c = [x["h_hi"] - S.he_star(x["pct"], x["score"]) for x in during]
    lag_a = [h_eff(rows[i], H) - ms[rows[i]["person"]].he_star(rows[i]["pct"], rows[i]["score"]) for i in a_dur]
    print("  줄별 「잔차 0 시각」으로 본 h − h* (A는 촬영 시각 기준이라 잡음만, C는 지연+잡음):")
    for nm, v in (("A", lag_a), ("C", lag_c)):
        print(f"    {nm}: n={len(v)} 중앙 {median(v):+.1f}h 25% {quant(v,.25):+.1f} 75% {quant(v,.75):+.1f} 90% {quant(v,.9):+.1f} · "
              f">6h {sum(x > 6 for x in v)} ({100*sum(x > 6 for x in v)/len(v):.0f}%) · >24h {sum(x > 24 for x in v)}")
    # 48~72h C 줄 상세
    bad = sorted([(r_at(x, 0)[1], x) for x in during], key=lambda t: t[0])[:8]
    print("  잔차가 가장 낮은(오래된 화면 같은) C 8줄: " + "; ".join(
        f"oi{x['oi']} {x['site']}{x['no']} h_hi {x['h_hi']:.1f} {x['pct']}%·{x['score']}억 eL {e:+.2f} h* {S.he_star(x['pct'], x['score']):.0f}h"
        for e, x in bad))

    # 7. JP
    print("\n## 7. JP (A 독립, 매듭 3)")
    for nm, mk in (("JP 자체 c", lambda: default_surface(H, 3)),
                   (f"KR c={S.c:.2f} 고정", lambda: default_surface(H, 3, fixed_c=S.c))):
        e1, e2, m2 = lopo(rows_jp, H, mk)
        mj = mk().fit(pts_of(rows_jp, H))
        print(f"  {nm}: c={mj.c:.2f} · LOPO 억 {desc(e1)} | log% {desc(e2, 3)}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        def bi(x):
            if x["day_lo"] == "post":
                return 5
            he = h_eff(x, H)
            return next((k for k, (a, b) in enumerate(HBINS) if a <= he < b), 4)
        names = [f"A {a}~{b:.0f}h" for a, b in HBINS] + ["A 종료 후"]
        series = [(names[k], COL[k], [(x["pct"], x["score"]) for x in rows if bi(x) == k], "o") for k in range(6)]
        curves = []
        for k, h in enumerate((12, 36, 60, 84, 108, 120)):
            pts = [(10 ** S.pred_L(s, h), s) for s in range(int(S.S_rng[0]), int(S.S_rng[1]) + 1)]
            curves.append((f"곡면 he={h}h", COL[k], pts))
        svg_scatter(os.path.join(args.out, "surface_KR_A.svg"), f"{season['name']} 한섭 A 줄과 기본 곡면 L = F(S) + c·g(h)",
                    series, curves, "상위 % (로그 축)", "점수(억)", (-1, 1.6), (0, 300))
        series = [("A (촬영 시각)", "#1f77b4", [(x["pct"], x["score"]) for x in rows], "o"),
                  ("C 게시 종료 전", "#d62728", [(x["pct"], x["score"]) for x in during], "x"),
                  ("C 게시 종료 후", "#ff7f0e", [(x["pct"], x["score"]) for x in after], "x")]
        svg_scatter(os.path.join(args.out, "surface_KR_AC.svg"), f"{season['name']} 한섭 A와 C (곡선: 종료 시각 곡면)",
                    series, [("곡면 he=120h", "#111", curves[-1][2])], "상위 % (로그 축)", "점수(억)", (-1, 1.6), (0, 300))
        print(f"\n그림: {args.out}/surface_KR_A.svg · surface_KR_AC.svg")


if __name__ == "__main__":
    main()
