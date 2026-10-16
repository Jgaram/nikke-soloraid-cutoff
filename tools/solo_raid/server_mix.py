"""솔로레이드 분석 층: 서버 미상 줄의 서버 확률 (모델 판독 0, 요청 0). server.jsonl은 덮어쓰지 않는다.

python3 -I tools/solo_raid/server_mix.py STATE [--cv | --gate] [--em] [--prior unit|flat|em] [--model off|offslope] [--alpha 0.5]

수집 단계의 「점수·% 크기로 서버를 정하지 않는다」 규칙은 server.jsonl(분류 층)에 그대로 남는다. 이 파일은 그 위의 별도 분석 층이다.
결과 p는 확률이며, 이 p로 정한 서버를 그 서버 곡선의 독립 근거처럼 쓰면 순환이다 (요약의 circularity 절 참고).

입력 (읽기만): STATE/core.jsonl · obs.jsonl · server.jsonl · post_time.jsonl · season.json(c_start·end)
출력
- STATE/post_server_mix.jsonl: core 사진 줄(oi)마다 한 줄
  {"oi", "p": {"KR":..,"JP":..,"GL":..,"OTHER":..} 또는 null, "best", "conf", "basis", "why", "unit", "n_unit"}
  미상 줄에는 민감도용 "p_alt"(다른 사전확률로 낸 p)와 "p_alt_prior"도 붙인다 (기본 em이면 라벨 unit 사전)
  - basis "label": server.jsonl에 서버가 붙은 줄. p = {그 서버: 1} (NA 등 곡면 없는 서버도 이름 그대로)
  - basis "surface": 기본 집합(use) 서버 미상 줄, 곡면 우도 + 사전확률 (+ 텍스트 후보·top7 있으면)
  - basis "prior": 기본 집합 서버 미상인데 점수·%가 없어 사전확률만
  - basis "not_use": 기본 집합 밖(제외·지난 시즌·트래커 등) 서버 미상 줄. p = null
  - OTHER = 곡면을 맞출 수 없는 서버 묶음(S41: NA 1줄뿐, SEA 0). 우도는 평평(log % 2자리 폭)하다
  - unit: 같은 서버여야 하는 줄 묶음(같은 사진 img ∪ 같은 사람 person ∪ server.jsonl 지문 묶음 cluster). 한 unit은 같은 p를 받는다
- STATE/post_server_mix_summary.json: 설정·근거별 기여·교차검증 지표·EM 비교·순환 영향

모형 (서버 s, 점수 S 억, L = log10 %, he = 곡면 시간, g = sqrt(he/H) − 1, H = end − c_start)
- 기준 서버 R(라벨 줄이 가장 많은 서버, S41은 KR): L = b0 + rcs(S/100; 분위 매듭) + c·g(he)  (explore_surface.SurfL과 같은 꼴)
- 다른 서버: L = 기준 곡면 + a_s (+ b_s·z, z = (S−200)/100을 그 서버 학습 점수 범위로 막음) — 모양과 시간 계수 c를 빌린다 (offslope, 적합 줄 8 미만이면 off)
- 적합 줄: 기본 집합 · use_indep · 점수·% 있음 · h ≥ 1. 시각 A·B·C 모두(C는 상한 h, docs 측정 규칙) — --fit-grades A로 바꾼다
- 잡음: 혼합 (1−ε)·정규(σ) + ε·평평(3자리 폭). σ는 학습 잔차의 가중 1.4826·MAD, he < 24와 그 뒤 따로 (앞이 5줄 미만이면 뒤 × 2.5),
  작은 서버는 오프셋 불확실성 1.25σ/√n을 더한다
- 시각 적분: A는 한 점, B는 [h_lo, h_hi] 고른 7점, 종료 전 게시 C는 he = max(h_hi − d, h_lo)를 게시 지연 d의 경험 분포
  (기본 집합 A 사진의 게시 − 촬영, 20 분위)로 평균. 확실한 종료 후(day_lo == post)는 he = H
- unit 우도: 시점(moment)마다 한 줄, log 우도 합 × m^(α−1) (m = 우도 있는 줄 수; α = 1이면 독립 가정, 기본 0.5로 같은 사람 줄의 상관을 깎는다)
- 사전확률 (기본 em) --prior unit: 라벨 unit 수(줄 수 아님)로 사이트별 디리클레 평활 (사이트 M = 5, 전체 +1).
  flat: 곡면 서버 같게, OTHER 2%. em: 라벨 곡면을 고정하고 미상 unit으로 사이트별 섞임 비율을 EM 추정 (라벨 쏠림 보정, Saerens 2002 방식,
  unit 사전을 디리클레 M = 5로 섞음). 기본을 em으로 둔 근거 (S41): 라벨 unit은 일섭 15·한섭 16·글섭 4라 일섭이 부풀었다
  (서버를 밝히는 글은 한섭 밖이 많다). 사전 없이(flat) 곡면만으로도 미상 줄의 95%가 한섭으로 가서 unit 사전(한섭 44%)과 맞지 않는다
- 텍스트 후보(server.jsonl cand, basis q·cmt·multi): 후보 서버 우도 × 2 (약한 근거, 라벨 줄에는 라벨 근거 자체라 검증 불가)
- top7: unit에 1~7위 점수 2개 이상인 사진이 있으면, 서버마다 그 서버 A 사진들의 k위 점수 꺾은선(시각 h)과 견준 log10 비 중앙값 d로
  우도 (1−ε)·정규(d; τ) + ε/3 (τ = A 사진 하나씩 빼고 잰 d의 1.4826·MAD, 같은 정확 점수 칸은 빼고 잼. A 사진 없는 서버는 평평).
  단조 어긋남 수만으로는 서버를 못 가른다 (점수가 높고 늦은 사진은 낮은 서버와도 어긋남 0). S41 기본 집합 미상 줄에는 해당 사진이 0장이다
- --em: 곡면 EM — 미상 줄을 p 가중으로 곡면 적합에 넣고 E·M을 되풀이 (8회 또는 p 변화 < 1e-3). 교차검증에서는 시험 묶음 줄을 넣지 않는다

--cv: 라벨 줄로 묶음 교차검증 (요약 cv 절). 설계 두 가지
- unit: unit 하나씩 빼기 (사진·사람·지문 묶음이 학습·시험에 걸치지 않음. S41 한섭 큰 묶음 c1 194줄도 통째로 빠진다 — 엄격)
- post: 글 번호 sha1 5겹, 시험 줄과 같은 사람·사진·글을 학습에서 지운다 (purge). 2단계 이음(c1 안 다른 사진)은 남는다
지표: 정확도·서버별 재현율·로그 손실·Brier·보정(conf 구간별 적중)·시간대(h 상한 24·48)·% 구간별, 부분집합(본인 줄·result/my_rank 화면·c1 밖).
같은 설정으로 두 번 돌리면 결과 파일이 같다 (무작위 없음).

--gate (2026-10-16): 진행 중 시즌(S42~)에 미상 줄 확률을 예측에 써도 되는지만 잰다 → 요약 "gate".
  unit 하나씩 빼기(위 unit 설계, 사전 --prior·곡면 --model, 텍스트 후보·top7 없음)에서 unit마다 「한섭인가 아닌가」가 맞는 비율.
  통과 = 한섭 곡면 있음 · 한섭 라벨 unit 5개 이상 · 그 비율 0.90 이상 (e1 제안 0.90 + 최소 unit 수는 관리 작업자가 덧붙임).
  predict.py --unknown mix는 통과일 때만 이 파일의 확률을 쓴다. --cv면 gate도 함께 잰다.
  곡면을 맞출 라벨 줄이 어느 서버에도 3줄 미만이면 멈추지 않고 미상 줄을 p = null, gate 미통과로 쓴다 (시즌 초반).
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explore_surface import G, QS, h_end_of, quant, rcs, wls  # noqa: E402

TF = "%Y-%m-%d %H:%M:%S"
OTHER = "OTHER"
W_FLAT = 3.0          # 잡음 혼합의 평평한 부분 폭 (log10 % 자릿수)
W_OTHER = 2.0         # OTHER 우도 폭
EPS = 0.05
EARLY = 24.0
N_DELAY = 20
B_GRID = 7
CAND_LR = 2.0
PRIOR_M = 5.0
NK_BY_N = ((40, 5), (20, 4), (0, 3))
MIN_SLOPE = 8
EM_ITERS = 8


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def ts(s):
    return dt.datetime.strptime(s, TF)


# ---------- 통계 ----------

def wmed(v, w):
    pr = sorted((a, b) for a, b in zip(v, w) if b > 0)
    if not pr:
        return float("nan")
    tot = sum(b for _, b in pr)
    acc = 0.0
    for a, b in pr:
        acc += b
        if acc >= 0.5 * tot - 1e-12:
            return a
    return pr[-1][0]


def wmad(r, w):
    m = wmed(r, w)
    return 1.4826 * wmed([abs(x - m) for x in r], w)


def rfit(X, y, pw, c=4.685):
    """가중 강건 회귀: LAD 근사(IRLS) → Tukey bisquare. pw = 사전 가중(EM의 p). (계수, 척도)"""
    p = len(X[0])
    beta = wls(X, y, pw, ridge=1e-6) if p > 1 else [wmed(y, pw)]
    if p == 1:
        for _ in range(30):
            r = [yi - beta[0] for yi in y]
            s = max(wmad(r, pw), 1e-6)
            w = [a * ((1 - (x / (c * s)) ** 2) ** 2 if abs(x) < c * s else 0.0) for a, x in zip(pw, r)]
            if sum(w) <= 0:
                break
            nb = sum(a * b for a, b in zip(w, y)) / sum(w)
            done = abs(nb - beta[0]) < 1e-10
            beta = [nb]
            if done:
                break
        r = [yi - beta[0] for yi in y]
        return beta, wmad(r, pw)
    for _ in range(10):
        r = [yi - sum(a * b for a, b in zip(xi, beta)) for xi, yi in zip(X, y)]
        w = [a / max(abs(x), 1e-3) for a, x in zip(pw, r)]
        beta = wls(X, y, w)
    for _ in range(30):
        r = [yi - sum(a * b for a, b in zip(xi, beta)) for xi, yi in zip(X, y)]
        s = max(wmad(r, pw), 1e-6)
        w = [a * ((1 - (x / (c * s)) ** 2) ** 2 if abs(x) < c * s else 0.0) for a, x in zip(pw, r)]
        nb = wls(X, y, w)
        done = max(abs(a - b) for a, b in zip(nb, beta)) < 1e-9
        beta = nb
        if done:
            break
    r = [yi - sum(a * b for a, b in zip(xi, beta)) for xi, yi in zip(X, y)]
    return beta, wmad(r, pw)


def npdf(e, s):
    return math.exp(-0.5 * (e / s) ** 2) / (s * math.sqrt(2 * math.pi))


# ---------- 자료 ----------

class NoSurface(Exception):
    """곡면을 맞출 라벨 줄(서버마다 3줄 이상)이 없다."""


class Data:
    def __init__(self, st, fit_grades="ABC"):
        self.core = jl(os.path.join(st, "core.jsonl"))
        self.obs = jl(os.path.join(st, "obs.jsonl"))
        self.srv = jl(os.path.join(st, "server.jsonl"))
        self.pt = jl(os.path.join(st, "post_time.jsonl"))
        with open(os.path.join(st, "season.json"), encoding="utf-8") as f:
            self.season = json.load(f)
        for name, rows in (("server.jsonl", self.srv), ("post_time.jsonl", self.pt)):
            if len(rows) != len(self.obs) or any((a["img"], a["row"]) != (b["img"], b["row"]) for a, b in zip(rows, self.obs)):
                raise SystemExit(f"{name}의 줄이 obs.jsonl과 맞지 않는다 — post_all.sh로 1단계를 다시 돌린다")
        self.H = float(h_end_of(self.season))
        self.fit_grades = fit_grades
        self.photos = [x for x in self.core if x["src"] == "photo"]
        self.cmap = {x["oi"]: x for x in self.photos}
        self.use = [x for x in self.photos if x["use"]]
        self.delays = self._delays()
        self._units()
        self.rows = [self._row(x) for x in self.use]
        self.by_oi = {r["oi"]: r for r in self.rows}
        names = collections.Counter(r["label"] for r in self.rows if r["label"] and r["fit"])
        self.modeled = sorted([s for s, n in names.items() if n >= 3], key=lambda s: (-names[s], s))
        if not self.modeled:   # 진행 중 시즌 초반: 곡면을 맞출 라벨 줄이 어느 서버에도 3줄 미만 (2026-10-16)
            raise NoSurface(dict(names))
        self.ref = self.modeled[0]
        self.classes = self.modeled + [OTHER]
        for r in self.rows:
            r["truth"] = None if not r["label"] else (r["label"] if r["label"] in self.modeled else OTHER)
        self.snaps = self._snaps()

    def _delays(self):
        d = []
        for x in self.use:
            if x["t_grade"] != "A" or x["h"] is None or not (0 <= x["h"] <= self.H):
                continue
            o, p = self.obs[x["oi"]], self.pt[x["oi"]]
            if o.get("post_t") and p.get("t"):
                d.append(max(0.0, (ts(o["post_t"]) - ts(p["t"])).total_seconds() / 3600))
        return [quant(d, (j + 0.5) / N_DELAY) for j in range(N_DELAY)] if d else [0.0]

    def _units(self):
        par = {}

        def f(a):
            while par.setdefault(a, a) != a:
                par[a] = par[par[a]]
                a = par[a]
            return a

        def un(a, b):
            ra, rb = f(a), f(b)
            if ra != rb:
                par[max(ra, rb)] = min(ra, rb)

        for x in self.use:
            r = "r%06d" % x["oi"]
            un(r, "i" + self.obs[x["oi"]]["img"])
            un(r, "p" + x["person"])
            c = self.srv[x["oi"]]["cluster"]
            if c:
                un(r, "c" + c)
        roots = sorted(set(f("r%06d" % x["oi"]) for x in self.use))
        first = {}
        for x in sorted(self.use, key=lambda x: x["oi"]):
            first.setdefault(f("r%06d" % x["oi"]), x["oi"])
        order = sorted(roots, key=lambda k: first[k])
        name = {k: "u%d" % i for i, k in enumerate(order)}
        self.unit_of = {x["oi"]: name[f("r%06d" % x["oi"])] for x in self.use}

    def _row(self, x):
        o, s = self.obs[x["oi"]], self.srv[x["oi"]]
        H = self.H
        L = math.log10(x["pct"]) if x["pct"] and x["pct"] > 0 else None
        S = x["score"]
        clip = lambda h: max(1.0, min(float(h), H))
        if x["day_lo"] == "post":
            hes, hfit = [H], H
        elif x["t_grade"] == "A":
            hes, hfit = [clip(x["h"])], clip(x["h"])
        elif x["t_grade"] == "B":
            lo = x["h_lo"] if x["h_lo"] is not None else x["h"]
            hi = x["h_hi"] if x["h_hi"] is not None else x["h"]
            hes = [clip(lo + (hi - lo) * j / (B_GRID - 1)) for j in range(B_GRID)]
            hfit = clip(x["h"])
        else:
            hi = x["h_hi"] if x["h_hi"] is not None else x["h"]
            lo = x["h_lo"] if x["h_lo"] is not None else 0.0
            hes = [clip(max(hi - d, lo)) for d in self.delays]
            hfit = clip(hi)
        ok = L is not None and S is not None
        fit = ok and x["use_indep"] and x["t_grade"] in self.fit_grades and (x["day_lo"] == "post" or (x["h"] is not None and x["h"] >= 1))
        hi = x["h_hi"] if x["h_hi"] is not None else x["h"]
        return dict(oi=x["oi"], site=x["site"], no=x["no"], img=o["img"], person=x["person"], moment=x["moment"],
                    indep=x["use_indep"], cluster=s["cluster"], label=x["server"], cand=list(s.get("cand") or []),
                    cand_basis=s.get("basis"), unit=self.unit_of[x["oi"]], L=L, S=S, hes=hes, hfit=hfit, ok=ok, fit=fit,
                    grade=x["t_grade"], h_hi=(H if x["day_lo"] == "post" else hi), post=(x["day_lo"] == "post"),
                    pct=x["pct"], self=x["self"], screen=x["screen"])

    def _snaps(self):
        """사진별 1~7위 점수 (본인 줄 아님, 2개 이상) — explore_surface.top7_snapshots와 같은 조건, 기본 집합 사진만."""
        im = collections.defaultdict(list)
        for k, o in enumerate(self.obs):
            im[o["img"]].append(k)
        use_img = collections.defaultdict(list)
        for r in self.rows:
            use_img[r["img"]].append(r)
        out = {}
        for img, rs in use_img.items():
            ks = im[img]
            if any(self.cmap[k]["exclude"] for k in ks if k in self.cmap):
                continue
            ranks = {}
            exact = set()
            for k in ks:
                o = self.obs[k]
                if o.get("rank") and o.get("score") and 1 <= o["rank"] <= 7 and not o.get("self"):
                    ranks[o["rank"]] = o["score"]
                if o.get("score"):
                    exact.add(o["score"])
            if len(ranks) < 2:
                continue
            r0 = rs[0]
            x0 = self.cmap[r0["oi"]]
            out[img] = dict(img=img, unit=r0["unit"], label=r0["label"], grade=x0["t_grade"], h=x0["h"],
                            lo=(x0["h_lo"] if x0["h_lo"] is not None else 0.0), hi=(x0["h_hi"] if x0["h_hi"] is not None else x0["h"]),
                            ranks=ranks, exact=exact, hes=r0["hes"])
        return out


# ---------- 곡면 ----------

class Model:
    """서버별 곡면 + 잡음. fit(pts) pts = [(server, L, S, he, w)]"""

    def __init__(self, ref, modeled, H, kind="offslope"):
        self.ref, self.modeled, self.H, self.kind = ref, modeled, H, kind

    def g(self, he):
        return G["sqrt"][0](he, self.H)

    def ref_feat(self, S, he):
        return [1.0] + rcs(S / 100.0, self.knots) + [self.g(he)]

    def ref_L(self, S, he):
        return sum(a * b for a, b in zip(self.ref_feat(S, he), self.beta))

    def pred(self, s, S, he):
        base = self.ref_L(S, he)
        if s == self.ref:
            return base
        a = self.off.get(s)
        if a is None:
            return None
        if len(a) == 1:
            return base + a[0]
        z = min(max((S - 200.0) / 100.0, self.zr[s][0]), self.zr[s][1])
        return base + a[0] + a[1] * z

    def fit(self, pts, knot_S=None):
        rp = [p for p in pts if p[0] == self.ref and p[4] > 0]
        ks = knot_S if knot_S is not None else [p[2] for p in rp]
        nk = next(k for n, k in NK_BY_N if len(ks) >= n)
        self.knots = [quant([s / 100.0 for s in ks], q) for q in QS[nk]]
        X = [self.ref_feat(p[2], p[3]) for p in rp]
        self.beta, _ = rfit(X, [p[1] for p in rp], [p[4] for p in rp])
        self.c = self.beta[-1]
        self.off, self.zr, self.n = {}, {}, {self.ref: sum(p[4] for p in rp)}
        for s in self.modeled:
            if s == self.ref:
                continue
            sp = [p for p in pts if p[0] == s and p[4] > 0]
            n = sum(p[4] for p in sp)
            self.n[s] = n
            if n < 1:
                continue
            e = [p[1] - self.ref_L(p[2], p[3]) for p in sp]
            w = [p[4] for p in sp]
            if self.kind == "offslope" and n >= MIN_SLOPE:
                zs = [(p[2] - 200.0) / 100.0 for p in sp]
                self.zr[s] = (min(zs), max(zs))
                b, _ = rfit([[1.0, z] for z in zs], e, w)
                self.off[s] = b
            else:
                b, _ = rfit([[1.0]] * len(e), e, w)
                self.off[s] = b
        # 잡음
        res, ws, early = [], [], []
        for p in pts:
            if p[4] <= 0:
                continue
            m = self.pred(p[0], p[2], p[3])
            if m is None:
                continue
            res.append(p[1] - m)
            ws.append(p[4])
            early.append(p[3] < EARLY)
        late = [(r, w) for r, w, e in zip(res, ws, early) if not e]
        ear = [(r, w) for r, w, e in zip(res, ws, early) if e]
        npar = len(self.beta) + sum(len(v) for v in self.off.values())
        ntot = sum(ws)
        infl = math.sqrt(ntot / max(ntot - npar, 1.0))
        self.sig_late = max(wmad([r for r, _ in late], [w for _, w in late]) * infl, 0.02)
        ne = sum(w for _, w in ear)
        self.sig_early = (max(wmad([r for r, _ in ear], [w for _, w in ear]) * infl, self.sig_late) if ne >= 5
                          else 2.5 * self.sig_late)
        self.n_early = ne
        # 시간 지지 범위: 서버마다 라벨 적합 줄 he의 둘째로 작은 값 (그 앞은 σ를 최대 2배까지 키운다)
        self.hsup = {}
        for s in self.modeled:
            hs = sorted(p[3] for p in pts if p[0] == s and p[5])
            self.hsup[s] = hs[1] if len(hs) > 1 else (hs[0] if hs else self.H)
        # 서버별 학습 잔차 (보고용)
        self.res_by = {}
        for s in self.modeled:
            rr = [(p[1] - self.pred(s, p[2], p[3]), p[4]) for p in pts if p[0] == s and p[4] > 0 and self.pred(s, p[2], p[3]) is not None]
            if rr:
                self.res_by[s] = (round(wmad([a for a, _ in rr], [b for _, b in rr]), 4), round(sum(b for _, b in rr), 2))
        return self

    def out_of_time(self, s, he):
        return he < self.hsup.get(s, 0.0)

    def sig(self, s, he):
        b = self.sig_early if he < EARLY else self.sig_late
        hs = self.hsup.get(s, 0.0)
        if he < hs:
            b *= 1.0 + (hs - he) / hs
        if s == self.ref:
            return b
        n = max(self.n.get(s, 0.0), 1.0)
        return math.sqrt(b * b + (1.25 * self.sig_late) ** 2 / n)

    def lik(self, r, s):
        """줄 r의 서버 s 우도 (시각 적분). 곡면 없으면 None."""
        if s == OTHER:
            return 1.0 / W_OTHER
        tot = 0.0
        for he in r["hes"]:
            m = self.pred(s, r["S"], he)
            if m is None:
                return None
            tot += (1 - EPS) * npdf(r["L"] - m, self.sig(s, he)) + EPS / W_FLAT
        return tot / len(r["hes"])

    def resid(self, r, s):
        """시각 적분 중앙의 잔차 (보고용): 지연 분포 중앙 시각."""
        he = r["hes"][len(r["hes"]) // 2]
        m = self.pred(s, r["S"], he)
        return None if m is None else r["L"] - m


def fit_points(rows, weights=None):
    """rows: 라벨 학습 줄 (server = label), weights: {oi: {s: w}} 미상 줄 EM 가중."""
    pts = []
    for r in rows:
        if r["fit"] and r["truth"] and r["truth"] != OTHER:
            pts.append((r["truth"], r["L"], r["S"], r["hfit"], 1.0, True))
    if weights:
        for r, ws in weights:
            if not r["fit"]:
                continue
            for s, w in sorted(ws.items()):
                if s != OTHER and w > 1e-4:
                    pts.append((s, r["L"], r["S"], r["hfit"], w, False))
    return pts


# ---------- 사전확률·추론 ----------

def unit_site(rows):
    return collections.Counter(r["site"] for r in rows).most_common(1)[0][0]


def prior_from_units(train_rows, classes, mode="unit"):
    """{site: {class: π}} + 전체. 라벨 unit 단위로 센다."""
    if mode == "flat":
        k = len(classes) - 1
        base = {c: (0.98 / k if c != OTHER else 0.02) for c in classes}
        return {"*": base}
    by_u = collections.defaultdict(list)
    for r in train_rows:
        if r["truth"]:
            by_u[r["unit"]].append(r)
    cnt = collections.Counter()
    site_cnt = collections.defaultdict(collections.Counter)
    for u, rs in by_u.items():
        t = collections.Counter(r["truth"] for r in rs).most_common(1)[0][0]
        cnt[t] += 1
        site_cnt[unit_site(rs)][t] += 1
    N = sum(cnt.values())
    glob = {c: (cnt[c] + 1.0) / (N + len(classes)) for c in classes}
    out = {"*": glob}
    for site, sc in site_cnt.items():
        n = sum(sc.values())
        out[site] = {c: (sc[c] + PRIOR_M * glob[c]) / (n + PRIOR_M) for c in classes}
    return out


def unit_loglik(model, rs, classes, alpha):
    """{class: log 우도} (사전 제외), 쓴 줄 수, 줄별 log 우도"""
    per = []
    for r in rs:
        if not (r["ok"] and r["indep"]):
            continue
        lk = {}
        for c in classes:
            v = model.lik(r, c)
            lk[c] = math.log(v) if v is not None else None
        per.append((r, lk))
    m = len(per)
    w = m ** (alpha - 1) if m else 1.0
    tot = {}
    for c in classes:
        vals = [lk[c] for _, lk in per]
        if any(v is None for v in vals):
            tot[c] = None
        else:
            tot[c] = w * sum(vals)
    return tot, m, per


def interp_rank(rf, k, t, skip):
    """서버 A 사진들의 k위 점수를 시각 t에 꺾은선으로 (양 끝은 끝값). skip = 시험 사진의 정확 점수(같은 사람 기록은 뺀다)."""
    pts = sorted((h, rk[k]) for h, rk, _ in rf if k in rk and rk[k] not in skip)
    if not pts:
        return None
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for (h0, v0), (h1, v1) in zip(pts, pts[1:]):
        if h0 <= t <= h1:
            return v0 if h1 == h0 else v0 + (v1 - v0) * (t - h0) / (h1 - h0)
    return pts[-1][1]


def top7_dev(sn, rf, t):
    """사진의 1~7위 점수와 서버 k위 꺾은선의 log10 비 중앙값 (없으면 None)."""
    d = []
    for k, v in sorted(sn["ranks"].items()):
        ref = interp_rank(rf, k, t, sn["exact"])
        if ref and ref > 0 and v > 0:
            d.append(math.log10(v / ref))
    if not d:
        return None
    d.sort()
    return d[len(d) // 2]


def top7_lr(snaps_in_unit, refs, classes, tau):
    """{class: log 우도} — 서버마다 사진 시각(곡면과 같은 시각 적분)에서 k위 꺾은선과의 log 비 중앙값 d,
    우도 = (1−ε)·정규(d; τ) + ε/W. refs {server: [(h, ranks, exact)]}. A 사진 없는 서버는 평평."""
    out = {c: 0.0 for c in classes}
    used = False
    for sn in snaps_in_unit:
        vals = {}
        for c in classes:
            rf = refs.get(c, [])
            if c == OTHER or not rf:
                vals[c] = None
                continue
            tot, n = 0.0, 0
            for t in sn["hes"]:
                d = top7_dev(sn, rf, t)
                if d is None:
                    continue
                tot += (1 - EPS) * npdf(d, tau) + EPS / W_FLAT
                n += 1
            vals[c] = tot / n if n else None
        got = [v for v in vals.values() if v is not None]
        if not got:
            continue
        used = True
        flat = 1.0 / W_OTHER
        for c in classes:
            out[c] += math.log(vals[c] if vals[c] is not None else flat)
    return out, used


def top7_tau(D, refs):
    """A 사진 하나씩 빼고 잰 d의 1.4826·MAD (서버 모두 모아서)."""
    d = []
    for s, rf in refs.items():
        for i, (h, rk, ex) in enumerate(rf):
            sn = {"ranks": rk, "exact": ex}
            others = rf[:i] + rf[i + 1:]
            v = top7_dev(sn, others, h)
            if v is not None:
                d.append(v)
    return max(wmad(d, [1.0] * len(d)), 0.005) if len(d) >= 5 else 0.05, len(d)


def posterior(logl, logp):
    ks = [c for c in logl if logl[c] is not None and logp.get(c, 0) > 0]
    if not ks:
        return None
    v = {c: logl[c] + math.log(logp[c]) for c in ks}
    mx = max(v.values())
    z = sum(math.exp(a - mx) for a in v.values())
    return {c: (math.exp(v[c] - mx) / z if c in v else 0.0) for c in logl}


class Infer:
    def __init__(self, D, model, prior, alpha, use_cand=True, use_t7=True, refs=None):
        self.D, self.model, self.prior, self.alpha = D, model, prior, alpha
        self.use_cand, self.use_t7, self.refs = use_cand, use_t7, refs or {}
        self.tau = top7_tau(D, self.refs)[0] if self.refs else 0.05

    def unit(self, rs):
        cl = self.D.classes
        ll, m, per = unit_loglik(self.model, rs, cl, self.alpha)
        parts = {"surface": dict(ll), "m": m}
        lp = self.prior.get(unit_site(rs), self.prior["*"])
        tot = dict(ll)
        if m == 0:
            tot = {c: 0.0 for c in cl}
        if self.use_cand:
            cs = set()
            for r in rs:
                if r["cand_basis"] in ("q", "cmt", "multi") and r["cand"]:
                    cs |= set(r["cand"])
            if cs:
                for c in cl:
                    if c in cs and tot[c] is not None:
                        tot[c] += math.log(CAND_LR)
                parts["cand"] = sorted(cs)
        if self.use_t7:
            sn = [self.D.snaps[i] for i in sorted(set(r["img"] for r in rs)) if i in self.D.snaps]
            if sn:
                lr, used = top7_lr(sn, self.refs, cl, self.tau)
                if used:
                    for c in cl:
                        if tot[c] is not None:
                            tot[c] += lr[c]
                    parts["top7"] = lr
        p = posterior(tot, lp)
        return p, parts, lp


def refs_from(D, rows):
    refs = collections.defaultdict(list)
    lab_imgs = set(r["img"] for r in rows if r["truth"])
    for img, sn in sorted(D.snaps.items()):
        if img in lab_imgs and sn["grade"] == "A" and sn["label"] in D.modeled:
            refs[sn["label"]].append((sn["h"], sn["ranks"], sn["exact"]))
    return refs


def group_units(rows):
    g = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: r["oi"]):
        g.setdefault(r["unit"], []).append(r)
    return g


def em_fit(D, train, pool, prior, alpha, kind, iters=EM_ITERS, refit=True, em_prior=False, knot_S=None):
    """라벨 학습 줄 + 미상 줄(pool). refit: 미상 줄을 p 가중으로 곡면 적합에 넣는다 (곡면 EM).
    em_prior: 미상 unit 사후확률로 사이트별 섞임 비율을 다시 정한다 (라벨 사전에 디리클레 M으로 고정). (model, {unit: p}, prior, 반복 수)"""
    model = Model(D.ref, D.modeled, D.H, kind).fit(fit_points(train), knot_S)
    units = group_units(pool)
    base = prior
    last = None
    it = 0
    for it in range(1, iters + 1):
        inf = Infer(D, model, prior, alpha, use_cand=False, use_t7=False)
        post = {u: inf.unit(rs)[0] for u, rs in units.items()}
        if em_prior:
            prior = em_prior_update(base, post, units, D.classes)
        if refit:
            weights = [(r, post[u]) for u, rs in units.items() for r in rs if post[u]]
            model = Model(D.ref, D.modeled, D.H, kind).fit(fit_points(train, weights), knot_S)
        if last is not None:
            dif = max([abs(post[u][c] - last[u][c]) for u in post if post[u] and last.get(u) for c in D.classes] or [0.0])
            if dif < 1e-3:
                break
        last = post
    return model, post, prior, it


def em_prior_update(prior, post, units, classes):
    """미상 unit의 사후확률 평균으로 사이트별 섞임 비율을 다시 정한다 (라벨 사전을 디리클레 M으로 섞음)."""
    by = collections.defaultdict(list)
    for u, rs in units.items():
        if post[u]:
            by[unit_site(rs)].append(post[u])
    out = {"*": prior["*"]}
    for site in sorted(set(list(prior) + list(by))):
        if site == "*":
            continue
        base = prior.get(site, prior["*"])
        ps = by.get(site, [])
        n = len(ps)
        out[site] = {c: (sum(p[c] for p in ps) + PRIOR_M * base[c]) / (n + PRIOR_M) for c in classes}
    return out


# ---------- 평가 ----------

def metrics(items, classes):
    """items: [(truth, p, row)]"""
    n = len(items)
    if not n:
        return {"n": 0}
    acc = sum(1 for t, p, _ in items if max(p, key=lambda c: (p[c], c)) == t) / n
    ll = sum(-math.log(max(p.get(t, 0.0), 1e-6)) for t, p, _ in items) / n
    br = sum(sum((p.get(c, 0.0) - (1.0 if c == t else 0.0)) ** 2 for c in classes) for t, p, _ in items) / n
    rec = {}
    for c in classes:
        sub = [(t, p) for t, p, _ in items if t == c]
        if sub:
            rec[c] = [sum(1 for t, p in sub if max(p, key=lambda k: (p[k], k)) == c), len(sub)]
    cal = []
    for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, .99), (.99, 1.01)):
        sub = [(t, p) for t, p, _ in items if lo <= max(p.values()) < hi]
        if sub:
            cal.append({"bin": [lo, min(hi, 1.0)], "n": len(sub), "conf": round(sum(max(p.values()) for _, p in sub) / len(sub), 3),
                        "hit": round(sum(1 for t, p in sub if max(p, key=lambda k: (p[k], k)) == t) / len(sub), 3)})
    ece = sum(b["n"] * abs(b["conf"] - b["hit"]) for b in cal) / n
    hi8 = [(t, p) for t, p, _ in items if max(p.values()) >= 0.8]
    return {"n": n, "acc": round(acc, 3), "logloss": round(ll, 3), "brier": round(br, 3), "recall": rec, "calib": cal,
            "ece": round(ece, 3), "p80": [sum(1 for t, p in hi8 if max(p, key=lambda k: (p[k], k)) == t), len(hi8)]}


def slices(items, classes, D):
    def pb(x):
        return "<1" if x < 1 else "1-3" if x < 3 else "3-10" if x < 10 else "10+"
    out = {"all": metrics(items, classes)}
    for name, f in (("h_le24", lambda r: r["h_hi"] is not None and r["h_hi"] <= 24),
                    ("h_le48", lambda r: r["h_hi"] is not None and r["h_hi"] <= 48),
                    ("h_gt48_pre", lambda r: not r["post"] and r["h_hi"] is not None and r["h_hi"] > 48),
                    ("post", lambda r: r["post"]),
                    ("self", lambda r: r["self"]),
                    ("result_myrank", lambda r: r["screen"] in ("result", "my_rank")),
                    ("not_c1", lambda r: r["cluster"] != "c1"),
                    ("self_not_c1", lambda r: r["self"] and r["cluster"] != "c1"),
                    ("grade_A", lambda r: r["grade"] == "A"), ("grade_C", lambda r: r["grade"] == "C")):
        out[name] = metrics([it for it in items if f(it[2])], classes)
    for b in ("<1", "1-3", "3-10", "10+"):
        out["pct" + b] = metrics([it for it in items if it[2]["pct"] is not None and pb(it[2]["pct"]) == b], classes)
    # unit 단위 (unit마다 한 표)
    seen = {}
    for t, p, r in items:
        seen.setdefault((r["unit"], r.get("_fold")), (t, p, r))
    out["units"] = metrics(list(seen.values()), classes)
    return out


def fold_post(r):
    return int(hashlib.sha1(f"{r['site']}:{r['no']}".encode()).hexdigest(), 16) % 5


def run_cv(D, design, variant, args):
    """variant: dict(prior, em, kind, alpha, t7) → items, 잔차 기록"""
    lab = [r for r in D.rows if r["truth"]]
    unl = [r for r in D.rows if not r["label"]]
    folds = collections.OrderedDict()
    if design == "unit":
        for r in lab:
            folds.setdefault(r["unit"], []).append(r)
    else:
        for r in sorted(lab, key=lambda r: r["oi"]):
            folds.setdefault(fold_post(r), []).append(r)
    items, resid = [], []
    for fk in sorted(folds, key=str):
        test = folds[fk]
        tp, ti, tn = set(r["person"] for r in test), set(r["img"] for r in test), set((r["site"], r["no"]) for r in test)
        tu = set(r["unit"] for r in test)
        if design == "unit":
            train = [r for r in lab if r["unit"] not in tu]
        else:
            train = [r for r in lab if not (r["oi"] in {x["oi"] for x in test} or r["person"] in tp or r["img"] in ti
                                             or (r["site"], r["no"]) in tn)]
        prior = prior_from_units(train, D.classes, "flat" if variant["prior"] == "flat" else "unit")
        if variant["prior"] == "em":
            _, _, prior, _ = em_fit(D, train, unl, prior, variant["alpha"], variant["kind"], iters=50, refit=False, em_prior=True)
        if variant["em"]:
            model, _, _, _ = em_fit(D, train, unl, prior, variant["alpha"], variant["kind"], refit=True)
        else:
            model = Model(D.ref, D.modeled, D.H, variant["kind"]).fit(fit_points(train))
        refs = refs_from(D, train)
        inf = Infer(D, model, prior, variant["alpha"], use_cand=False, use_t7=variant.get("t7", True), refs=refs)
        for u, rs in group_units(test).items():
            p, parts, _ = inf.unit(rs)
            if p is None:
                continue
            for r in rs:
                rr = dict(r)
                rr["_fold"] = fk
                items.append((r["truth"], p, rr))
                if r["ok"] and r["truth"] in D.modeled:
                    e = model.resid(r, r["truth"])
                    if e is not None:
                        resid.append((r, e))
    return items, resid


def top7_cv(D):
    """(d) top7 근거만으로 라벨 사진의 서버를 맞히는지. 자기 사진은 refs에서 빼고, 같은 정확 점수(같은 사람 기록) 칸은 꺾은선에서 뺀다."""
    lab_imgs = {r["img"] for r in D.rows if r["truth"]}
    refs = refs_from(D, [r for r in D.rows if r["truth"]])
    tau, ntau = top7_tau(D, refs)
    res = collections.Counter()
    margins = []
    for img, sn in sorted(D.snaps.items()):
        if img not in lab_imgs or sn["label"] not in D.modeled:
            continue
        rf = {s: [(h, rk, ex) for h, rk, ex in v if not (sn["grade"] == "A" and h == sn["h"] and rk == sn["ranks"])]
              for s, v in refs.items()}
        lr, used = top7_lr([sn], rf, D.classes, tau)
        if not used:
            res["no_refs"] += 1
            continue
        best = max(D.modeled, key=lambda c: (lr[c], c))
        res[("hit" if best == sn["label"] else "miss", sn["label"], sn["grade"])] += 1
        oth = max(lr[c] for c in D.modeled if c != sn["label"])
        margins.append(lr[sn["label"]] - oth)
    margins.sort()
    q = lambda a: round(quant(margins, a), 2) if margins else None
    return ({"|".join(map(str, k)) if isinstance(k, tuple) else k: v for k, v in sorted(res.items(), key=str)},
            {"tau_log10": round(tau, 4), "tau_n": ntau, "margin_true_minus_best_other_q05_25_50": [q(.05), q(.25), q(.5)]})


def resid_summary(resid, D):
    out = {}
    for s in D.modeled:
        for name, f in (("all", lambda r: True), ("h_le48", lambda r: r["h_hi"] is not None and r["h_hi"] <= 48),
                        ("A", lambda r: r["grade"] == "A")):
            v = [e for r, e in resid if r["truth"] == s and f(r)]
            if v:
                m = sorted(abs(x) for x in v)
                out[f"{s}:{name}"] = {"n": len(v), "med_abs": round(m[len(m) // 2], 4),
                                      "mad": round(wmad(v, [1.0] * len(v)), 4), "bias": round(sorted(v)[len(v) // 2], 4)}
    return out


# ---------- 순환 영향 ----------

def surface_report(D, model, label):
    grid = {}
    for s in D.modeled:
        for S in (150.0, 200.0, 250.0):
            for he in (24.0, 48.0, D.H):
                v = model.pred(s, S, he)
                grid[f"{s}:{int(S)}억@{int(he)}h"] = None if v is None else round(10 ** v, 3)
    return {"label": label, "c": round(model.c, 4), "off": {s: [round(a, 4) for a in v] for s, v in model.off.items()},
            "sig_late": round(model.sig_late, 4), "sig_early": round(model.sig_early, 4), "n_early_fit": round(model.n_early, 2),
            "n_fit": {s: round(v, 2) for s, v in model.n.items()}, "pct_grid": grid,
            "insample_mad_by_server": model.res_by, "h_support": {s: round(v, 1) for s, v in model.hsup.items()}}


def obs_counts(D, post):
    """서버별 관측 수: 라벨 moment 수, 미상 줄의 p 합(기대 줄 수)·p² 합(배정 불확실성을 깎은 유효 수)·p ≥ 0.8 줄 수.
    줄은 기본 집합 · 독립(use_indep) · 점수·% 있음. unit 단위(같은 unit은 한 표)도 함께. post = {unit: p}"""
    out = {}
    for s in D.modeled:
        for name, f in (("h_le24", lambda r: r["h_hi"] is not None and r["h_hi"] <= 24),
                        ("h_le48", lambda r: r["h_hi"] is not None and r["h_hi"] <= 48),
                        ("all", lambda r: True)):
            lab = set(r["moment"] for r in D.rows if r["truth"] == s and r["ok"] and f(r))
            labu = set(r["unit"] for r in D.rows if r["truth"] == s and r["ok"] and f(r))
            ws, uw = [], {}
            for r in D.rows:
                if r["label"] or not (r["ok"] and r["indep"] and f(r)):
                    continue
                p = post.get(r["unit"])
                if p:
                    ws.append(p[s])
                    uw[r["unit"]] = p[s]
            out[f"{s}:{name}"] = {"label_moments": len(lab), "label_units": len(labu),
                                  "unl_sum_p": round(sum(ws), 2), "unl_sum_p2": round(sum(w * w for w in ws), 2),
                                  "unl_rows_p80": sum(1 for w in ws if w >= 0.8),
                                  "unl_units_sum_p": round(sum(uw.values()), 2), "unl_units_sum_p2": round(sum(w * w for w in uw.values()), 2),
                                  "unl_units_p80": sum(1 for w in uw.values() if w >= 0.8)}
    return out


# ---------- 본체 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--prior", default="em", choices=["unit", "flat", "em"])
    ap.add_argument("--model", default="offslope", choices=["off", "offslope"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--em", action="store_true", help="곡면 EM (미상 줄 p 가중 적합)")
    ap.add_argument("--fit-grades", default="ABC")
    ap.add_argument("--cv", action="store_true", help="교차검증·EM 비교·순환 영향을 요약에 넣는다 (몇 분)")
    ap.add_argument("--out", default=None, help="출력 파일 앞부분 (기본 STATE/post_server_mix)")
    ap.add_argument("--gate", action="store_true",
                    help="S42 쓰기 기준만 잰다 (unit 하나씩 빼기, 기본 설정 하나 — --cv보다 빠름). --cv면 언제나 잰다")
    args = ap.parse_args()
    st = args.state
    out = args.out or os.path.join(st, "post_server_mix")
    try:
        D = Data(st, args.fit_grades)
    except NoSurface as e:
        write_no_surface(st, out, e.args[0], args)
        return
    lab = [r for r in D.rows if r["truth"]]
    unl = [r for r in D.rows if not r["label"]]
    prior0 = prior_from_units(lab, D.classes, "flat" if args.prior == "flat" else "unit")

    # 최종 적합 (라벨 전부)
    model_lab = Model(D.ref, D.modeled, D.H, args.model).fit(fit_points(lab))
    _, pe_post, prior_em, pe_it = em_fit(D, lab, unl, prior0, args.alpha, args.model, iters=50, refit=False, em_prior=True)
    prior = prior_em if args.prior == "em" else prior0
    em_model, em_post, _, em_it = em_fit(D, lab, unl, prior, args.alpha, args.model, refit=True, em_prior=False)
    model = em_model if args.em else model_lab
    refs = refs_from(D, lab)
    inf = Infer(D, model, prior, args.alpha, use_cand=True, use_t7=True, refs=refs)
    inf_nc = Infer(D, model, prior, args.alpha, use_cand=False, use_t7=True, refs=refs)
    alt_mode = "unit" if args.prior != "unit" else "em"
    inf_alt = Infer(D, model, prior0 if alt_mode == "unit" else prior_em, args.alpha, use_cand=True, use_t7=True, refs=refs)
    inf_flat = Infer(D, model, prior_from_units(lab, D.classes, "flat"), args.alpha, use_cand=False, use_t7=False, refs=refs)
    inf_surf_only = Infer(D, model, prior, args.alpha, use_cand=False, use_t7=False, refs=refs)

    unit_p, contrib = {}, collections.Counter()
    for u, rs in group_units(unl).items():
        p, parts, lp = inf.unit(rs)
        unit_p[u] = (p, parts, lp, rs, inf_alt.unit(rs)[0])
        p_nc = inf_nc.unit(rs)[0]
        p_fl = inf_flat.unit(rs)[0]
        p_so = inf_surf_only.unit(rs)[0]
        bk = lambda q: None if q is None else max(q, key=lambda c: (q[c], c))
        contrib["units"] += 1
        contrib["rows"] += len(rs)
        if parts["m"] == 0:
            contrib["units_no_surface"] += 1
        if "cand" in parts:
            contrib["units_cand"] += 1
            if bk(p) != bk(p_nc):
                contrib["units_cand_changed_best"] += 1
        if "top7" in parts:
            contrib["units_top7"] += 1
        if bk(p) != bk(p_fl):
            contrib["units_prior_changed_best_vs_flat"] += 1
        if bk(p_so) != bk(p):
            contrib["units_cand_or_top7_changed_best"] += 1

    by_oi = {}
    for x in D.photos:
        oi = x["oi"]
        if x["server"]:
            by_oi[oi] = {"oi": oi, "p": {x["server"]: 1.0}, "best": x["server"], "conf": 1.0, "basis": "label",
                         "why": f"server.jsonl {x['server_basis']}", "unit": D.unit_of.get(oi), "n_unit": None}
        elif not x["use"]:
            by_oi[oi] = {"oi": oi, "p": None, "best": None, "conf": None, "basis": "not_use",
                         "why": "기본 집합 밖: " + (",".join(x["exclude"]) or ",".join(f for f in x["flag"] if f) or "pct·시즌 조건"),
                         "unit": None, "n_unit": None}
    for u, (p, parts, lp, rs, p_alt) in unit_p.items():
        best = None if p is None else max(p, key=lambda c: (p[c], c))
        site = unit_site(rs)
        if parts["m"]:
            sl = parts["surface"]
            ref = sl.get(D.ref)
            lr = " ".join(f"{c}{(sl[c] - ref):+.1f}" for c in D.classes if c != D.ref and sl[c] is not None and ref is not None)
            why = f"unit {len(rs)}줄(우도 {parts['m']}); 곡면 logLR vs {D.ref}: {lr}; 사전 {site}"
            basis = "surface"
        else:
            why = f"unit {len(rs)}줄; 점수·% 없음 → 사전 {site}"
            basis = "prior"
        oot = sorted(set(c for r in rs if r["ok"] for c in D.modeled for he in r["hes"][len(r["hes"]) // 2:len(r["hes"]) // 2 + 1]
                         if model.out_of_time(c, he)))
        if oot:
            why += "; 시간 지지 밖(σ↑) " + "/".join(oot)
        if "cand" in parts:
            why += f"; 텍스트 후보 {'/'.join(parts['cand'])} ×{CAND_LR:g}"
        if "top7" in parts:
            why += "; top7 " + " ".join(f"{c}{v:+.1f}" for c, v in parts["top7"].items())
        for r in rs:
            by_oi[r["oi"]] = {"oi": r["oi"], "p": None if p is None else {c: round(p[c], 4) for c in D.classes},
                              "best": best, "conf": None if p is None else round(p[best], 4), "basis": basis, "why": why,
                              "unit": u, "n_unit": len(rs),
                              "p_alt": None if p_alt is None else {c: round(p_alt[c], 4) for c in D.classes}, "p_alt_prior": alt_mode}
    for r in D.rows:
        if r["label"]:
            by_oi[r["oi"]]["n_unit"] = None
    with open(out + ".jsonl", "w", encoding="utf-8") as f:
        for oi in sorted(by_oi):
            f.write(json.dumps(by_oi[oi], ensure_ascii=False, sort_keys=False) + "\n")

    # 요약
    ub = collections.Counter()
    conf_bins = collections.Counter()
    for oi, v in by_oi.items():
        if v["basis"] in ("surface", "prior"):
            ub[v["best"]] += 1
            c = v["conf"]
            conf_bins["<0.5" if c < .5 else "0.5-0.8" if c < .8 else "0.8-0.95" if c < .95 else "0.95+"] += 1
    sum_p = collections.defaultdict(float)
    for oi, v in by_oi.items():
        if v["basis"] in ("surface", "prior") and v["p"]:
            for c, a in v["p"].items():
                sum_p[c] += a
    summary = {
        "season": D.season.get("name"), "H": D.H, "generated_by": "tools/solo_raid/server_mix.py",
        "settings": {"prior": args.prior, "model": args.model, "alpha": args.alpha, "em": args.em, "fit_grades": args.fit_grades,
                     "eps": EPS, "cand_lr": CAND_LR, "prior_M": PRIOR_M},
        "classes": D.classes, "ref": D.ref, "other_means": "곡면 없는 서버 (S41 라벨: NA 1줄). 평평한 우도",
        "delay_quantiles_h": [round(d, 3) for d in D.delays],
        "rows": {"photo": len(D.photos), "use": len(D.rows), "use_labeled": len(lab) + sum(1 for r in D.rows if r["label"] and not r["truth"]),
                 "use_unlabeled": len(unl), "units": len(set(r["unit"] for r in D.rows)),
                 "units_labeled": len(set(r["unit"] for r in lab)), "units_unlabeled": len(set(r["unit"] for r in unl)),
                 "labeled_units_by_server": dict(collections.Counter(
                     collections.Counter(r["truth"] for r in rs).most_common(1)[0][0] for rs in group_units(lab).values()))},
        "prior": {k: {c: round(v, 3) for c, v in d.items()} for k, d in prior.items()},
        "prior_unit": {k: {c: round(v, 3) for c, v in d.items()} for k, d in prior0.items()},
        "prior_em_unlabeled": {k: {c: round(v, 3) for c, v in d.items()} for k, d in prior_em.items()},
        "prior_em_iters": pe_it,
        "unlabeled_best": dict(sorted(ub.items(), key=str)), "unlabeled_conf": dict(conf_bins),
        "unlabeled_sum_p": {c: round(v, 2) for c, v in sorted(sum_p.items())},
        "contrib": dict(contrib),
        "surface_labels": surface_report(D, model_lab, "labels only"),
        "surface_em": surface_report(D, em_model, f"labels + unlabeled p-weighted EM ({em_it} iters)"),
        "obs_counts_labels_model": obs_counts(D, {u: v[0] for u, v in unit_p.items()}),
        "obs_counts_labels_model_alt_prior": obs_counts(D, {u: v[4] for u, v in unit_p.items()}),
        "obs_counts_em_model": obs_counts(D, em_post),
        "shift": shift_report(D),
    }
    t7, t7d = top7_cv(D)
    summary["top7_check"] = {"result": t7, "scale": t7d}
    if args.gate or args.cv:
        summary["gate"] = gate_check(D, args)
    if args.cv:
        cv = {}
        variants = collections.OrderedDict([
            ("prior_only", None),
            ("surf_flat", dict(prior="flat", em=False, kind=args.model, alpha=args.alpha, t7=False)),
            ("surf_unitprior", dict(prior="unit", em=False, kind=args.model, alpha=args.alpha, t7=False)),
            ("surf_unitprior_t7", dict(prior="unit", em=False, kind=args.model, alpha=args.alpha, t7=True)),
            ("surf_unitprior_off", dict(prior="unit", em=False, kind="off", alpha=args.alpha, t7=False)),
            ("surf_unitprior_a1", dict(prior="unit", em=False, kind=args.model, alpha=1.0, t7=False)),
            ("surf_emprior", dict(prior="em", em=False, kind=args.model, alpha=args.alpha, t7=False)),
            ("em_surf_unitprior", dict(prior="unit", em=True, kind=args.model, alpha=args.alpha, t7=False)),
            ("em_surf_emprior", dict(prior="em", em=True, kind=args.model, alpha=args.alpha, t7=False)),
        ])
        for design in ("unit", "post"):
            for name, v in variants.items():
                if name == "prior_only":
                    items = prior_only_cv(D, design)
                    resid = []
                else:
                    items, resid = run_cv(D, design, v, args)
                sl = slices(items, D.classes, D)
                cv[f"{design}:{name}"] = {"metrics": sl, "resid": resid_summary(resid, D) if resid else None}
                print(f"cv {design}:{name} acc={sl['all'].get('acc')} ll={sl['all'].get('logloss')} "
                      f"brier={sl['all'].get('brier')} rec={sl['all'].get('recall')} p80={sl['all'].get('p80')}", file=sys.stderr)
        summary["cv"] = cv
    with open(out + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    print(f"server_mix: 사진 {len(D.photos)}줄 · 기본 집합 {len(D.rows)} (라벨 {len(D.rows) - len(unl)}, 미상 {len(unl)} → "
          f"best {dict(sorted(ub.items(), key=str))}, conf {dict(conf_bins)}) · unit {summary['rows']['units']} → {out}.jsonl")


# ---------- S42 쓰기 기준 (2026-10-16) ----------

GATE_ACC = 0.90       # e1 제안: unit 설계 교차검증의 한섭 unit 정확도 90% 미만이면 미상 줄을 쓰지 않는다
GATE_MIN_KR = 5       # 관리 작업자가 덧붙임: 한섭 라벨 unit이 5개 미만이면 정확도를 잴 수 없다고 보고 통과시키지 않는다


def gate_check(D, args):
    """진행 중 시즌에 서버 확률을 예측에 써도 되는지. 라벨 unit 하나씩 빼기 교차검증(기본 설정: 사전 args.prior·곡면 args.model,
    텍스트 후보·top7 없음 — e1의 surf_emprior와 같은 설정)으로 unit마다 「한섭인가 아닌가」 판정이 맞는 비율(한섭 unit 정확도)을 잰다.
    통과 = 한섭 곡면 있음 · 한섭 라벨 unit ≥ GATE_MIN_KR · 정확도 ≥ GATE_ACC. predict.py --unknown mix가 이 값을 읽는다."""
    lab = [r for r in D.rows if r["truth"]]
    units = group_units(lab)
    utruth = {u: collections.Counter(r["truth"] for r in rs).most_common(1)[0][0] for u, rs in units.items()}
    n_by = dict(collections.Counter(utruth.values()))
    g = {"rule": f"unit 하나씩 빼기 · 한섭 unit ≥ {GATE_MIN_KR} · 한섭/아님 정확도 ≥ {GATE_ACC}",
         "labeled_units": n_by, "modeled": D.modeled, "ref": D.ref, "acc_kr_binary": None, "kr_recall": None,
         "kr_precision": None, "acc_all": None}
    if "KR" not in D.modeled:
        g.update({"pass": False, "reason": "한섭 곡면 없음 (한섭 라벨 적합 줄 3줄 미만)"})
        return g
    if n_by.get("KR", 0) < GATE_MIN_KR:
        g.update({"pass": False, "reason": f"한섭 라벨 unit {n_by.get('KR', 0)}개 < {GATE_MIN_KR} (정확도를 잴 수 없음)"})
        return g
    v = dict(prior=args.prior, em=args.em, kind=args.model, alpha=args.alpha, t7=False)
    items, _ = run_cv(D, "unit", v, args)
    seen = {}
    for t, p, r in items:
        seen.setdefault(r["unit"], (t, p))
    best = lambda p: max(p, key=lambda c: (p[c], c))
    pairs = list(seen.values())
    n = len(pairs)
    hit = sum(1 for t, p in pairs if (best(p) == "KR") == (t == "KR"))
    kr_t = [(t, p) for t, p in pairs if t == "KR"]
    kr_p = [(t, p) for t, p in pairs if best(p) == "KR"]
    g["acc_kr_binary"] = round(hit / n, 3) if n else None
    g["kr_recall"] = [sum(1 for t, p in kr_t if best(p) == "KR"), len(kr_t)]
    g["kr_precision"] = [sum(1 for t, p in kr_p if t == "KR"), len(kr_p)]
    g["acc_all"] = round(sum(1 for t, p in pairs if best(p) == t) / n, 3) if n else None
    ok = n > 0 and hit / n >= GATE_ACC
    g.update({"pass": ok, "reason": f"한섭 unit 정확도 {hit}/{n}" + ("" if ok else f" < {GATE_ACC}")})
    return g


def write_no_surface(st, out, names, args):
    """곡면 없음: 라벨 줄은 그 서버, 기본 집합 미상 줄은 p = null(쓰지 않음)로 쓰고, 기준 미달을 요약에 남긴다."""
    core = [x for x in jl(os.path.join(st, "core.jsonl")) if x["src"] == "photo"]
    n_unl = 0
    with open(out + ".jsonl", "w", encoding="utf-8") as f:
        for x in sorted(core, key=lambda x: x["oi"]):
            if x["server"]:
                d = {"oi": x["oi"], "p": {x["server"]: 1.0}, "best": x["server"], "conf": 1.0, "basis": "label",
                     "why": f"server.jsonl {x['server_basis']}", "unit": None, "n_unit": None}
            elif not x["use"]:
                d = {"oi": x["oi"], "p": None, "best": None, "conf": None, "basis": "not_use", "why": "기본 집합 밖",
                     "unit": None, "n_unit": None}
            else:
                n_unl += 1
                d = {"oi": x["oi"], "p": None, "best": None, "conf": None, "basis": "no_surface",
                     "why": "곡면을 맞출 라벨 줄 없음 (서버마다 3줄 미만)", "unit": None, "n_unit": None}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(os.path.join(st, "season.json"), encoding="utf-8") as fs:
        season = json.load(fs)
    summary = {"season": season.get("name"), "generated_by": "tools/solo_raid/server_mix.py", "no_surface": True,
               "label_fit_rows_by_server": names, "use_unlabeled": n_unl,
               "gate": {"pass": False, "reason": f"곡면 없음 — 라벨 적합 줄 {names} (서버마다 3줄 필요)",
                        "rule": f"unit 하나씩 빼기 · 한섭 unit ≥ {GATE_MIN_KR} · 한섭/아님 정확도 ≥ {GATE_ACC}"}}
    with open(out + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    print(f"server_mix: 곡면 없음 (라벨 적합 줄 {names}) — 미상 {n_unl}줄은 p = null → {out}.jsonl")


def prior_only_cv(D, design):
    lab = [r for r in D.rows if r["truth"]]
    folds = collections.OrderedDict()
    for r in sorted(lab, key=lambda r: r["oi"]):
        folds.setdefault(r["unit"] if design == "unit" else fold_post(r), []).append(r)
    items = []
    for fk, test in folds.items():
        tu = set(r["unit"] for r in test)
        tp, ti, tn = set(r["person"] for r in test), set(r["img"] for r in test), set((r["site"], r["no"]) for r in test)
        if design == "unit":
            train = [r for r in lab if r["unit"] not in tu]
        else:
            to = {x["oi"] for x in test}
            train = [r for r in lab if not (r["oi"] in to or r["person"] in tp or r["img"] in ti or (r["site"], r["no"]) in tn)]
        pr = prior_from_units(train, D.classes, "unit")
        for u, rs in group_units(test).items():
            p = pr.get(unit_site(rs), pr["*"])
            for r in rs:
                rr = dict(r)
                rr["_fold"] = fk
                items.append((r["truth"], dict(p), rr))
    return items


def shift_report(D):
    """라벨 줄과 미상 줄의 분포 차이 (기본 집합)."""
    def desc(rs):
        pc = sorted(r["pct"] for r in rs if r["pct"])
        sc = sorted(r["S"] for r in rs if r["S"] is not None)
        hh = sorted(r["h_hi"] for r in rs if r["h_hi"] is not None)
        q = lambda v, a: round(quant(v, a), 2) if v else None
        return {"n": len(rs), "pct_q25_50_75": [q(pc, .25), q(pc, .5), q(pc, .75)],
                "score_q25_50_75": [q(sc, .25), q(sc, .5), q(sc, .75)], "h_hi_q25_50_75": [q(hh, .25), q(hh, .5), q(hh, .75)],
                "self": sum(1 for r in rs if r["self"]), "screen": dict(collections.Counter(r["screen"] for r in rs)),
                "grade": dict(collections.Counter(r["grade"] for r in rs)), "h_le48": sum(1 for r in rs if r["h_hi"] is not None and r["h_hi"] <= 48),
                "site": dict(collections.Counter(r["site"] for r in rs)), "c1": sum(1 for r in rs if r["cluster"] == "c1")}
    lab = [r for r in D.rows if r["label"]]
    unl = [r for r in D.rows if not r["label"]]
    return {"labeled": desc(lab), "unlabeled": desc(unl), "labeled_self_not_c1": desc([r for r in lab if r["self"] and r["cluster"] != "c1"])}


if __name__ == "__main__":
    main()
