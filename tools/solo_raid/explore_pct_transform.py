"""솔로레이드 2단계 2.5 탐색: % 변환 모양 — 「t시점 x%였던 점수 → 최종 y%」 (측정 설계용, 모델 아님, 요청 0).

python3 -I tools/solo_raid/explore_pct_transform.py STATE [--out DIR] [--grades A|all] [--interp lin|pchip|iso|linraw]
        [--with-c] [--with-best] [--server-person] [--json]

입력 (읽기만 한다. state/에 쓰지 않는다)
- STATE/season.json       c_start·end (챌린지 길이 H)
- STATE/core.jsonl        핵심 관측 표. use 줄, 같은 사람·같은 %는 한 시점 (dedupe_moments — core의 moment는 쓰지 않는다)
- STATE/truth_points.jsonl 정답 곡선 점. 등급 A(·A2), flag best_record는 뺀다 (truth.json curves와 같은 점, 점수는 score_raw)

두 표현
- P (% 기준): 관측 (점수 s, 시각 h, 그때 % x)에 최종 곡선의 역함수로 최종 % y(s)를 붙여 log10(y/x).
- S (점수 기준, 「컷 점수 비율」): 같은 % x의 최종 점수 F(x)와 지금 점수 s의 비 log10(F(x)/s).
- 관측 없는 % 구간에는 붙이지 않는다: 최종 곡선 이웃 점의 % 비가 1.5 넘고 차가 0.5 넘으면 빈 구간 (truth.py coverage와 같은 규칙),
  최소·최대 % 밖도 쓰지 않는다. P는 s가 덮인 구간의 점수 범위 안, S는 x가 덮인 % 구간 안일 때만 값이 있다.
- 단위 맞추기: S 값을 최종 곡선의 국소 기울기 k(x) = -d ln 점수 / d ln % 로 나누면 「% 단위로 옮긴 S」가 된다 (1차 근사).

기본 집합
- use · use_indep · 사진 · % 와 점수 있음 · 서버 KR/JP/GL · 종료 전 (A·B: h < H, C: --with-c일 때만, h(상한) < H)
- flag best_record는 뺀다 (--with-best로 넣음)
- --server-person: 같은 정확 점수 지문 사람(person P…)의 다른 줄에 서버가 하나만 있으면 그 서버를 쓴다 (점수 크기로 추정하는 것이 아님)

출력: 화면 표 (--json이면 요약 JSON도). --out DIR이면 DIR/pct_transform_points.jsonl (점마다 값)과 그림 SVG.
"""
import argparse
import bisect
from html import escape
import collections
import datetime as dt
import json
import math
import os
import random
import statistics

FMT = "%Y-%m-%d %H:%M:%S"
X_EDGES = [0, 0.5, 1, 2, 4, 8, 16, 101]
X_NAMES = ["<0.5", "0.5-1", "1-2", "2-4", "4-8", "8-16", ">=16"]


import importlib.util as _ilu  # noqa: E402
_sp = _ilu.spec_from_file_location("split", os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.py"))
split = _ilu.module_from_spec(_sp)
_sp.loader.exec_module(split)

def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def season_hours(state):
    s = json.load(open(os.path.join(state, "season.json"), encoding="utf-8"))
    c0 = dt.datetime.strptime(s["c_start"], FMT)
    end = dt.datetime.strptime(s["end"], FMT)
    return s, (end - c0).total_seconds() / 3600.0


# ---------------- 최종 곡선 ----------------

def final_points(state, server, grades):
    """(pct, score_raw) 목록, pct 순. 같은 %는 점수 평균."""
    pts = collections.defaultdict(list)
    for d in jl(os.path.join(state, "truth_points.jsonl")):
        if d["server"] != server or d["grade"] not in grades:
            continue
        if "best_record" in d["flags"] or "mono_viol" in d["flags"]:
            continue
        pts[d["pct"]].append(d["score_raw"] if d["score_raw"] is not None else d["score"])
    return sorted((p, statistics.mean(v)) for p, v in pts.items())


def pav_decreasing(xs, ys):
    """점수가 %에 대해 줄어들도록 단조 회귀 (PAV)."""
    blocks = []  # [sum, n, xs]
    for x, y in zip(xs, ys):
        blocks.append([y, 1, [x]])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, n, xx = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
            blocks[-1][2] += xx
    out = []
    for s, n, xx in blocks:
        for x in xx:
            out.append(s / n)
    return out


class Curve:
    """최종 곡선: % → 점수, 점수 → %. 축 u = ln %, 값 = 점수."""

    def __init__(self, pts, method="lin"):
        self.method = method
        self.p = [p for p, _ in pts]
        self.s = [s for _, s in pts]
        if method == "iso":
            self.s = pav_decreasing(self.p, self.s)
        self.u = [math.log(p) for p in self.p] if method != "linraw" else list(self.p)
        self.n = len(self.p)
        # 덮인 구간: 이웃 점 % 비 > 1.5 그리고 차 > 0.5 이면 빈 구간 (truth.py와 같음)
        self.seg_ok = [not (b / a > 1.5 and b - a > 0.5) for a, b in zip(self.p, self.p[1:])]
        if method == "pchip":
            self.d = self._pchip_slopes()

    def _pchip_slopes(self):
        u, s, n = self.u, self.s, self.n
        if n < 2:
            return [0.0] * n
        h = [u[i + 1] - u[i] for i in range(n - 1)]
        dl = [(s[i + 1] - s[i]) / h[i] for i in range(n - 1)]
        d = [0.0] * n
        if n == 2:
            return [dl[0], dl[0]]
        for k in range(1, n - 1):
            if dl[k - 1] * dl[k] <= 0:
                d[k] = 0.0
            else:
                w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
                d[k] = (w1 + w2) / (w1 / dl[k - 1] + w2 / dl[k])

        def end(h0, h1, m0, m1):
            v = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
            if v * m0 <= 0:
                return 0.0
            if m0 * m1 <= 0 and abs(v) > abs(3 * m0):
                return 3 * m0
            return v
        d[0] = end(h[0], h[1], dl[0], dl[1])
        d[-1] = end(h[-1], h[-2], dl[-1], dl[-2])
        return d

    def _seg_eval(self, i, uu):
        u0, u1 = self.u[i], self.u[i + 1]
        s0, s1 = self.s[i], self.s[i + 1]
        if self.method != "pchip":
            t = (uu - u0) / (u1 - u0)
            return s0 + t * (s1 - s0)
        hh = u1 - u0
        t = (uu - u0) / hh
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        return h00 * s0 + h10 * hh * self.d[i] + h01 * s1 + h11 * hh * self.d[i + 1]

    def _u(self, pct):
        return math.log(pct) if self.method != "linraw" else pct

    def _seg_of_pct(self, pct):
        if self.n < 2 or pct < self.p[0] or pct > self.p[-1]:
            return None
        i = bisect.bisect_right(self.p, pct) - 1
        i = min(i, self.n - 2)
        return i if self.seg_ok[i] else None

    def score_at(self, pct):
        """F(x): 덮인 구간 밖이면 None."""
        i = self._seg_of_pct(pct)
        if i is None:
            return None
        return self._seg_eval(i, self._u(pct))

    def pct_at(self, score):
        """F^-1(s): 점수가 덮인 구간의 점수 범위 안일 때만."""
        for i in range(self.n - 1):
            hi, lo = self.s[i], self.s[i + 1]
            if not self.seg_ok[i]:
                continue
            if lo - 0.005 <= score <= hi + 0.005:   # truth_points score_raw는 소수 4자리 — 0.005억 허용 (truth.py와 같음)
                if hi == lo:
                    uu = (self.u[i] + self.u[i + 1]) / 2
                else:
                    a, b = self.u[i], self.u[i + 1]
                    for _ in range(60):
                        m = (a + b) / 2
                        if self._seg_eval(i, m) > score:
                            a = m
                        else:
                            b = m
                    uu = (a + b) / 2
                return math.exp(uu) if self.method != "linraw" else uu
        return None

    def slope_k(self, pct):
        """k = -d ln 점수 / d ln % (유한 차분, 덮인 구간 안)."""
        e = 0.02
        a, b = self.score_at(pct * math.exp(-e)), self.score_at(pct * math.exp(e))
        if a is None or b is None:
            # 구간 끝: 한쪽 차분
            c = self.score_at(pct)
            if c is None:
                return None
            if b is not None:
                return -(math.log(b) - math.log(c)) / e
            if a is not None:
                return -(math.log(c) - math.log(a)) / e
            return None
        if a <= 0 or b <= 0:
            return None
        return -(math.log(b) - math.log(a)) / (2 * e)


# ---------------- 관측 ----------------

def dedupe_moments(use):
    """시점 대표 줄: 같은 사람·같은 %(0.005 안)는 한 시점으로 본다. 대표는 A·B > C, 그다음 h가 작은 줄(C면 더 좁은 상한).
    core.jsonl의 moment·use_indep를 쓰지 않는 까닭: S41에서 % 없는 줄을 거쳐 다른 % 줄이 한 시점으로 묶인 경우가 있다
    (P214#1 — 0.43·0.56·1.15·1.24·1.40%, P506#2 — 1.07·1.48%)."""
    g = collections.defaultdict(list)
    for r in use:
        if r["src"] != "photo" or r["pct"] is None:
            continue
        g[(r["person"], round(r["pct"] / 0.01))].append(r)
    out = []
    for L in g.values():
        L.sort(key=lambda r: (r["t_grade"] == "C", r["h"] if r["h"] is not None else 1e9, r["oi"]))
        out.append(L[0])
    out.sort(key=lambda r: r["oi"])
    return out


def load_obs(state, H, with_c=False, with_best=False, server_person=False):
    rows = jl(os.path.join(state, "core.jsonl"))
    use = [r for r in rows if r["use"]]
    psrv = collections.defaultdict(set)
    for r in use:
        if r["server"] and r["person"].startswith("P"):
            psrv[r["person"]].add(r["server"])
    out = []
    for r in dedupe_moments(use):
        if r["src"] != "photo" or r["pct"] is None or r["score"] is None:
            continue
        if r["day_lo"] == "post" or r["h"] is None:
            continue
        g = r["t_grade"]
        if g == "C" and not with_c:
            continue
        if r["h"] >= H:
            continue
        if not with_best and "best_record" in r["flag"]:
            continue
        srv = r["server"]
        if srv is None and server_person and len(psrv.get(r["person"], ())) == 1:
            srv = next(iter(psrv[r["person"]]))
        if srv not in ("KR", "JP", "GL"):
            continue
        s = r["score_raw"] if r["score_raw"] is not None else r["score"]
        out.append({"oi": r["oi"], "server": srv, "server_given": r["server"], "h": r["h"], "grade": g,
                    "x": r["pct"], "s": s, "person": r["person"], "moment": r["moment"],
                    "screen": r["screen"], "flag": r["flag"]})
    return out


def xbin(x):
    return bisect.bisect_right(X_EDGES, x) - 1


def attach(obs, curves):
    for o in obs:
        c = curves.get(o["server"])
        o["y"] = o["P"] = o["S"] = o["k"] = o["S_as_P"] = None
        if c is None:
            continue
        y = c.pct_at(o["s"])
        if y is not None and y > 0:
            o["y"] = y
            o["P"] = math.log10(y / o["x"])
        F = c.score_at(o["x"])
        if F is not None and F > 0:
            o["F"] = F
            o["S"] = math.log10(F / o["s"])
            k = c.slope_k(o["x"])
            o["k"] = k
            if k and k > 0:
                o["S_as_P"] = o["S"] / k
    return obs


# ---------------- 요약 ----------------

def collapse(points, key):
    """같은 사람의 같은 칸 값은 중앙값 하나로 (반복 행을 독립 표본으로 세지 않는다)."""
    g = collections.defaultdict(list)
    for o in points:
        if o.get(key) is not None:
            g[o["person"]].append(o)
    out = []
    for p, L in g.items():
        out.append({"person": p, "v": statistics.median(o[key] for o in L),
                    "h": statistics.median(o["h"] for o in L), "lx": statistics.median(math.log10(o["x"]) for o in L)})
    return out


def mad(v):
    if not v:
        return None
    m = statistics.median(v)
    return statistics.median(abs(a - m) for a in v)


def ols(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    if sxx <= 0:
        return None
    b = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / sxx
    return my - b * mx, b, mx


def boot_slope(pp, nb=400, seed=1):
    rnd = random.Random(seed)
    bs = []
    for _ in range(nb):
        smp = [pp[rnd.randrange(len(pp))] for _ in pp]
        r = ols([q["lx"] for q in smp], [q["v"] for q in smp])
        if r:
            bs.append(r[1])
    bs.sort()
    if len(bs) < 20:
        return None
    return bs[int(0.05 * len(bs))], bs[int(0.95 * len(bs)) - 1]


def day_of(h):
    return int(h // 24) + 1


def summarize(obs, H, server, reps=("P", "S", "S_as_P")):
    res = {"server": server, "n_obs": len(obs)}
    O = [o for o in obs if o["server"] == server]
    res["n_server"] = len(O)
    res["n_P"] = sum(o["P"] is not None for o in O)
    res["n_S"] = sum(o["S"] is not None for o in O)
    res["persons_P"] = len({o["person"] for o in O if o["P"] is not None})
    cells = {}
    days = sorted({day_of(o["h"]) for o in O})
    for rep in reps:
        for d in days:
            for b in range(len(X_NAMES)):
                pts = [o for o in O if day_of(o["h"]) == d and xbin(o["x"]) == b]
                pp = collapse(pts, rep)
                if pp:
                    v = [q["v"] for q in pp]
                    cells[(rep, d, b)] = {"n": len(pp), "med": statistics.median(v), "mad": mad(v)}
    res["cells"] = cells
    # 같은 일차 안의 모양: log 비율 ~ a + b log10 x (사람 묶음)
    shape = {}
    for rep in reps:
        for d in days:
            pts = [o for o in O if day_of(o["h"]) == d]
            pp = collapse(pts, rep)
            r = ols([q["lx"] for q in pp], [q["v"] for q in pp])
            if not r:
                continue
            a, b, mx = r
            level = statistics.median(q["v"] for q in pp)
            resid = [q["v"] - (a + b * q["lx"]) for q in pp]
            ci = boot_slope(pp)
            lx = [q["lx"] for q in pp]
            shape[(rep, d)] = {"n": len(pp), "level": level, "slope": b, "slope90": ci,
                               "rel_slope": (b / level) if level else None,
                               "resid_mad": mad(resid), "resid_mad0": mad([q["v"] for q in pp]),
                               "lx_range": (min(lx), max(lx))}
    res["shape"] = shape
    # h 수렴 모양: 비율이 끝에서 0(=1배)이 되도록 원점 지나는 적합 v = a·g(h), g = ln(H/h) 또는 (H-h)/H
    conv = {}
    for rep in reps:
        pts = [o for o in O if o[rep] is not None and o["h"] > 0]
        # 사람·일차 묶음
        g = collections.defaultdict(list)
        for o in pts:
            g[(o["person"], day_of(o["h"]))].append(o)
        pp = [(statistics.median(o["h"] for o in L), statistics.median(o[rep] for o in L),
               statistics.median(math.log10(o["x"]) for o in L)) for L in g.values()]
        if len(pp) < 5:
            continue
        out = {"n": len(pp)}
        for name, fn in (("lnHh", lambda h: math.log(H / h)), ("lin", lambda h: (H - h) / H),
                         ("sqrt", lambda h: math.sqrt((H - h) / H))):
            gx = [fn(h) for h, _, _ in pp]
            sxx = sum(a * a for a in gx)
            a = sum(a * v for a, (_, v, _) in zip(gx, pp)) / sxx
            resid = [v - a * q for q, (_, v, _) in zip(gx, pp)]
            out[name] = {"a": a, "resid_mad": mad(resid),
                         "rel": mad(resid) / statistics.median(abs(v) for _, v, _ in pp)}
            # x 의존 추가: v = g(h)·(a + b·log10 x)
            X1, X2, Y = gx, [q * lx for q, (_, _, lx) in zip(gx, pp)], [v for _, v, _ in pp]
            s11 = sum(a * a for a in X1); s22 = sum(a * a for a in X2); s12 = sum(a * b for a, b in zip(X1, X2))
            s1y = sum(a * y for a, y in zip(X1, Y)); s2y = sum(a * y for a, y in zip(X2, Y))
            det = s11 * s22 - s12 * s12
            if det > 0:
                aa = (s1y * s22 - s2y * s12) / det
                bb = (s2y * s11 - s1y * s12) / det
                r2 = [y - q1 * aa - q2 * bb for q1, q2, y in zip(X1, X2, Y)]
                out[name + "_x"] = {"a": aa, "b": bb, "resid_mad": mad(r2),
                                    "rel": mad(r2) / statistics.median(abs(v) for _, v, _ in pp)}
        conv[rep] = out
    res["conv"] = conv
    return res


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def print_cells(res, rep, title):
    cells = res["cells"]
    days = sorted({d for (r, d, b) in cells if r == rep})
    print(f"\n[{res['server']} · {title}] 칸 = 중앙값 log10 비율 (배수) / 사람 수")
    print("일차 | " + " | ".join(X_NAMES))
    for d in days:
        row = []
        for b in range(len(X_NAMES)):
            c = cells.get((rep, d, b))
            row.append("" if not c else f"{c['med']:+.2f}({10 ** c['med']:.2f}x)/{c['n']}")
        print(f"{d} | " + " | ".join(row))


def print_shape(res):
    print(f"\n[{res['server']}] 같은 일차 안 모양: v = a + b·log10 x (사람 묶음; rel = b/수준, 한 자릿수(10배) x당 수준 대비 변화)")
    print("표현 | 일차 | 사람 | 수준(중앙) | 기울기 b [90% 사람 부트] | rel | 흩어짐 MAD 적합 후/전 | log10 x 범위")
    for (rep, d), s in sorted(res["shape"].items()):
        ci = s["slope90"]
        cis = f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else "—"
        print(f"{rep} | {d} | {s['n']} | {s['level']:+.3f} | {s['slope']:+.3f} {cis} | {fmt(s['rel_slope'], 2)} | "
              f"{fmt(s['resid_mad'])}/{fmt(s['resid_mad0'])} | {s['lx_range'][0]:+.2f}~{s['lx_range'][1]:+.2f}")


def print_conv(res):
    print(f"\n[{res['server']}] h 수렴: v = a·g(h) (원점 지남; 끝에서 1배). rel = 잔차 MAD / |v| 중앙값")
    for rep, o in res["conv"].items():
        parts = []
        for name in ("lnHh", "lin", "sqrt"):
            if name in o:
                q = o[name]
                parts.append(f"{name}: a={q['a']:+.3f} rel={q['rel']:.2f}")
            if name + "_x" in o:
                q = o[name + "_x"]
                parts.append(f"{name}+x: a={q['a']:+.3f} b={q['b']:+.3f} rel={q['rel']:.2f}")
        print(f"{rep} (n={o['n']}): " + " · ".join(parts))


def loo_interp(state, server, grades, method):
    """정답 곡선 점 하나씩 빼고 그 점수에서 %를 보간 → log10 오차 (안쪽 점만, 덮인 구간만)."""
    pts = final_points(state, server, grades)
    errs = []
    for j in range(1, len(pts) - 1):
        rest = pts[:j] + pts[j + 1:]
        c = Curve(rest, method)
        y = c.pct_at(pts[j][1])
        if y is not None:
            errs.append((pts[j][0], math.log10(y / pts[j][0])))
    return errs


def svg_scatter(path, series, xlab, ylab, title, xlog=True, yzero=True):
    """series: [(name, color, [(x, y)])]. 손으로 만든 SVG."""
    W, Hh, L, R, T, B = 760, 440, 60, 140, 30, 50
    allp = [p for _, _, ps in series for p in ps]
    if not allp:
        return
    xs = [math.log10(p[0]) if xlog else p[0] for p in allp]
    ys = [p[1] for p in allp]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys + [0]), max(ys + [0])
    if x1 == x0:
        x1 += 1
    if y1 == y0:
        y1 += 1
    fx = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    fy = lambda v: T + (y1 - v) / (y1 - y0) * (Hh - T - B)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{Hh}" fill="white"/>',
         f'<text x="{L}" y="18" font-size="13">{escape(title)}</text>',
         f'<line x1="{L}" y1="{Hh - B}" x2="{W - R}" y2="{Hh - B}" stroke="#444"/>',
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{Hh - B}" stroke="#444"/>']
    if yzero:
        o.append(f'<line x1="{L}" y1="{fy(0)}" x2="{W - R}" y2="{fy(0)}" stroke="#aaa" stroke-dasharray="4 3"/>')
    nt = 6
    for i in range(nt + 1):
        v = x0 + (x1 - x0) * i / nt
        lab = f"{10 ** v:.2g}" if xlog else f"{v:.0f}"
        o.append(f'<text x="{fx(v)}" y="{Hh - B + 15}" text-anchor="middle">{lab}</text>')
        w = y0 + (y1 - y0) * i / nt
        o.append(f'<text x="{L - 5}" y="{fy(w) + 4}" text-anchor="end">{w:.2f}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{Hh - 12}" text-anchor="middle">{escape(xlab)}</text>')
    o.append(f'<text x="14" y="{(T + Hh - B) / 2}" transform="rotate(-90 14 {(T + Hh - B) / 2})" text-anchor="middle">{escape(ylab)}</text>')
    for k, (name, col, ps) in enumerate(series):
        for p in ps:
            xv = math.log10(p[0]) if xlog else p[0]
            o.append(f'<circle cx="{fx(xv):.1f}" cy="{fy(p[1]):.1f}" r="3" fill="{col}" fill-opacity="0.7"/>')
        o.append(f'<circle cx="{W - R + 12}" cy="{T + 10 + 16 * k}" r="4" fill="{col}"/>')
        o.append(f'<text x="{W - R + 20}" y="{T + 14 + 16 * k}">{escape(name)} ({len(ps)})</text>')
    o.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(o))


def build(state, grades, method, with_c, with_best, server_person):
    season, H = season_hours(state)
    curves = {srv: Curve(final_points(state, srv, grades), method) for srv in ("KR", "JP", "GL")}
    obs = attach(load_obs(state, H, with_c, with_best, server_person), curves)
    return season, H, curves, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out")
    ap.add_argument("--grades", default="all", choices=["A", "all"])
    ap.add_argument("--interp", default="lin", choices=["lin", "pchip", "iso", "linraw"])
    ap.add_argument("--with-c", action="store_true")
    ap.add_argument("--with-best", action="store_true")
    ap.add_argument("--server-person", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sens", action="store_true", help="등급×보간 민감도 표")
    split.add_final_arg(ap)
    a = ap.parse_args()
    split.guard_args(a)   # 시험 시즌 정답 감시 (할 일 3.0b)
    grades = ("A",) if a.grades == "A" else ("A", "A2")
    season, H, curves, obs = build(a.state, grades, a.interp, a.with_c, a.with_best, a.server_person)
    print(f"시즌 {season['name']} · H={H:.2f}시간 · 곡선 등급 {grades} · 보간 {a.interp} · C 포함 {a.with_c} · "
          f"best_record 포함 {a.with_best} · 사람 서버 {a.server_person}")
    for srv, c in curves.items():
        cov = [(round(p0, 2), round(p1, 2)) for (p0, p1), ok in zip(zip(c.p, c.p[1:]), c.seg_ok) if not ok]
        print(f"  곡선 {srv}: 점 {c.n}, % {c.p[0] if c.n else None}~{c.p[-1] if c.n else None}, 빈 구간 {cov}")
    for srv in ("KR", "JP", "GL"):
        res = summarize(obs, H, srv)
        print(f"\n===== {srv}: 관측 {res['n_server']}줄, P 값 {res['n_P']} (사람 {res['persons_P']}), S 값 {res['n_S']}")
        if res["n_server"] == 0:
            continue
        print_cells(res, "P", "P = log10(최종 y / 지금 x)")
        print_cells(res, "S", "S = log10(최종 F(x) / 지금 s)")
        print_cells(res, "S_as_P", "S를 % 단위로 (S / k)")
        print_shape(res)
        print_conv(res)
    # 보간 단독 오차 (정답 점 하나 빼기)
    print("\n정답 곡선 보간 오차 (점 하나 빼고 그 점수의 % 보간; log10, |중앙|·최대; 덮인 구간 안쪽 점)")
    for srv in ("KR", "JP"):
        for gs in (("A",), ("A", "A2")):
            parts = []
            for m in ("lin", "pchip", "iso", "linraw"):
                e = loo_interp(a.state, srv, gs, m)
                if e:
                    ab = [abs(v) for _, v in e]
                    parts.append(f"{m}: n={len(e)} med={statistics.median(ab):.3f} max={max(ab):.3f}")
            print(f"  {srv} {'+'.join(gs)}: " + " · ".join(parts))
    if a.sens:
        sens(a.state, a.with_c, a.with_best, a.server_person)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "pct_transform_points.jsonl"), "w", encoding="utf-8") as f:
            for o in obs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        cols = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
        for rep, lab in (("P", "log10(최종 y / 지금 x)"), ("S_as_P", "S / k (% 단위)"), ("S", "log10(F(x)/s)")):
            ser = []
            for d in range(1, 6):
                ps = [(o["x"], o[rep]) for o in obs if o["server"] == "KR" and day_of(o["h"]) == d and o[rep] is not None]
                ser.append((f"{d}일차", cols[d - 1], ps))
            svg_scatter(os.path.join(a.out, f"kr_{rep}_by_x.svg"), ser, "지금 % x (log)", lab,
                        f"KR {rep}: 지금 % x에 따른 변환 비율, 일차별 ({'+'.join(grades)}·{a.interp})")
            ser = []
            for b in range(len(X_NAMES)):
                ps = [(o["h"], o[rep]) for o in obs if o["server"] == "KR" and xbin(o["x"]) == b and o[rep] is not None]
                if ps:
                    ser.append((f"x {X_NAMES[b]}%", (cols + ["#8c564b", "#17becf"])[b], ps))
            svg_scatter(os.path.join(a.out, f"kr_{rep}_by_h.svg"), ser, "h (챌린지 시작 후 시간)", lab,
                        f"KR {rep}: h에 따른 변환 비율, x 구간별", xlog=False)
        print(f"\n점·그림: {a.out}")


def sens(state, with_c, with_best, server_person):
    """등급(A / A+A2) × 보간(lin·pchip·iso·linraw)으로 KR 칸 중앙값과 모양 지표가 얼마나 바뀌는지."""
    print("\n민감도 (KR): 기준 = A+A2·lin. 칸 중앙값 차 |Δlog10| 최대·중앙, 일차별 기울기 b")
    base = None
    for gs in (("A", "A2"), ("A",)):
        for m in ("lin", "pchip", "iso", "linraw"):
            season, H, curves, obs = build(state, gs, m, with_c, with_best, server_person)
            res = summarize(obs, H, "KR", reps=("P", "S"))
            if base is None:
                base = res
            out = []
            for rep in ("P", "S"):
                diffs = [abs(c["med"] - base["cells"][k]["med"]) for k, c in res["cells"].items()
                         if k[0] == rep and k in base["cells"]]
                miss = sum(1 for k in base["cells"] if k[0] == rep and k not in res["cells"])
                new = sum(1 for k in res["cells"] if k[0] == rep and k not in base["cells"])
                sl = " ".join(f"d{d}:{res['shape'][(rep, d)]['slope']:+.2f}" for d in range(1, 6) if (rep, d) in res["shape"])
                out.append(f"{rep} n={res['n_' + rep]} Δmax={max(diffs) if diffs else 0:.3f} Δmed={statistics.median(diffs) if diffs else 0:.3f} "
                           f"칸-{miss}/+{new} b[{sl}]")
            print(f"  {'+'.join(gs)}·{m}: " + " | ".join(out))


if __name__ == "__main__":
    main()
