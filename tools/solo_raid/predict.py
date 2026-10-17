"""솔로레이드 컷 예측: 시각 T까지 게시된 관측으로 서버별 최종 곡선(점수 → 최종 %)을 예측한다 (모델 판독 0, 요청 0).

python3 -I tools/solo_raid/predict.py STATE --prior PRIOR.json --at T [--unknown site|none|mix] [--borrow-state S41STATE]
        [--unknown-servers KR] [--borrow-kind label|em] [--mix-file F] [--no-gate] [--fallback site|none]
        [--server-mix PATH] [--grades AB|all] [--out DIR] [--draws 2000] [--seed 1] [--tau-p X] [--tau-shape X] [--tau-ratio X]
        [--tau-scale0 X] [--no-season-diff] [--p-update] [--c-delay 0] [--robust mean|wmedian] [--top-cut Q] [--eps-wrong 0.2] [--json]
        [--allow-cstart-fallback] [--contam 0.1,0.2]
진행 중 시즌은 tools/solo_raid/predict_run.sh STATE T 한 명령으로 돌린다 (post_all → (mix면) server_mix → 이것).
직접 돌려도 --borrow-state 기본값은 predict_run.sh와 같다 (이 레포의 data/solo-raid/S41/state, 없으면 경고). ''를 주면 빌리지 않는다.

**검증 안 된 예측이다.** S41 한 시즌으로 만들었고, S41 되돌아보기는 곡선 모양이 S41 정답 그 자체라 성능 근거가 아니다.
1·3% 구간 폭의 핵심(τ_P, 1~3% 모양 차)은 가정이다. 출력 첫머리에 이 말을 늘 찍는다.

서버 미상 줄 (--unknown, 기본 site — 2026-10-16 tr 점검으로 정함, docs/solo-raid-cutoff.md 「예측 도구」)
- none: 서버 라벨 줄만. S41 tr에서 48시간까지 라벨 한섭 사람 1~2명이었다
- site: 미상 줄을 사이트별 서버 구성 비율(서버 라벨 사람 수, 점수·%를 쓰지 않음)로 뽑기마다 서버에 배정한다. T까지 라벨 사람이
        10명 미만이면 --borrow-state(끝난 시즌, 기본 S41)의 시즌 전체 비율을 빌리고 「S41에서 빌림」으로 표시한다
- mix : server_mix.py 확률. 같은 state로 다시 만든 파일(core.jsonl보다 새것)이고 요약 gate가 통과일 때만 쓴다. 아니면 --fallback
- 어느 방식이든 미상 줄은 --unknown-servers(기본 한섭)에만 넣는다. 일섭·글섭은 라벨 줄만 쓴다 (넣으면 S41 tr 일섭 3% +12%·글섭 +25%)
- 미상 줄을 썼으면 「라벨 줄만 쓴 결과」를 1·3%에 나란히 낸다
- site 방식은 「미상 줄 ≈ 한섭」 가정이다. 미상 줄에 일섭·글섭이 섞이면 한섭 컷을 늘 아래로 끈다 (S41 T=24 모의: 섞임 약 10~20%에서 3% −2% 넘음).
  섞임 민감도(--contam, 기본 10·20%): 미상 사람 중 함축 3% 점수(중앙 변환으로 그 사람 점을 3%로 옮긴 값)가 가장 낮은 f 몫을 빼고
  다시 낸 중앙 추정의 변화. 주 구간은 넓히지 않고(섞임 없음 가정) 「섞임 가정」 줄과 JSON contam에 따로 낸다 (docs/solo-raid-cutoff.md 「예측 도구」)
출력의 구분: 맨 위 요약 4~5줄(한섭 1·3%·등급, 일섭·글섭 등급, T·c_start, 「검증 안 된 예측」, 등급 기준), 주 결과 1·3%, 보조 0.5·2%,
「참고 — 시즌 간 모양 가정에 기댐」 5·10·15·30%. 서버마다 등급 (GRADE_RULE, 2026-10-16 재검토 뒤 고침):
A 보통 · B 주의 · C 강한 주의(T < 18시간, 1·3% 80% 폭 > 15%, 쓴 사람 3명 미만 「관측 적음 — 예측 보류 수준」, 외삽) · D 숫자만 참고(그 서버 관측 0).

T: 챌린지 시작 후 시간(예: 24, 36.5) 또는 KST 시각('YYYY-MM-DD HH:MM'). 종료 뒤 값은 종료 시각으로 자른다.

입력 (읽기만 한다. **truth로 시작하는 파일은 열지 않는다** — 코드에서 막는다: 이 스크립트를 직접 돌리면 open() 자체를 감싼다)
- STATE/season.json  c_start·end (챌린지 길이 H). **c_start가 없으면 멈춘다.** --allow-cstart-fallback일 때만 start 날짜 + 2일 05:00으로
                     대신하고 출력 머리·JSON warnings에 크게 남긴다 (start를 챌린지 시작으로 잘못 넣으면 c_start가 48시간 늦어 C 줄이 망가진다)
- STATE/core.jsonl   핵심 관측 표 (build_core.py). 사진·use·% 와 점수 있음·best_record 아님·확실한 종료 후 아님
- STATE/obs.jsonl    사진 줄의 게시 시각 post_t. **게시 시각 ≤ T인 줄만 쓴다.** 관측 시각도 T 이하여야 한다
                     (A·B: h, C: 상한 h_hi − --c-delay. C는 post_time 규칙대로 상한으로 쓰고 보정하지 않는다. h_hi < 0인 C 줄은
                     챌린지 시작 전 화면이라 버리고 셈 drop_c_before_start에 남긴다)
- PRIOR.json         fit_prior.py가 끝난 시즌으로 만든 사전 정보 (% 변환 P, 최종 곡선 모양, 시즌 간 차이 기본값)
- --server-mix PATH  (기본 끔) 서버 미상 줄의 서버 확률. 한 줄 {"oi" 또는 "key", "p"/"probs": {"KR": .., "JP": .., ...}}.
                     줄마다 뽑기 때마다 서버를 확률대로 하나 뽑는다 (다중 대입). 서버가 붙은 줄은 그대로 쓴다

방법 (S41 탐색 2.5·2.6, docs/solo-raid-analysis.md 「0. 정할 것」)
1. 관측 (점수 s, 시각 h, 지금 % x)을 최종 %로 옮긴다: y = x · 10^P, P = m_서버 · P(h) (· (1 + bx·log10 x)).
   P(h)의 꼴·모수·잔차 분포는 prior. 같은 점수의 %는 줄지 않으므로 P ≥ 0으로 자른다.
2. 옮긴 점마다 목표 % q의 컷 점수를 하나씩 낸다: ln S(q) = ln s + ln r(q) − ln r(y). r = 최종 곡선 모양 (prior, s(q)/s(3%)).
   즉 규모 모수만 이 시즌 자료로 정한다. 목표 q마다 점들을 가중 평균한다 (가중치 = 1/예상 분산:
   변환 오차가 점수로 옮겨지는 크기 k(y)·ln10·σ_P(h), 시즌 간 모양 차이 τ_shape·|D(q) − D(y)|, 모양 보간 0.01·|ln q − ln y|).
   뽑기마다 q 순서로 단조(점수가 %에 대해 줄어듦)를 맞춘다 (PAV).
3. 구간 = 몬테카를로 뽑기. 세 출처를 따로도 낸다.
   (i)   관측 잡음·표본: 사람 단위 붓스트랩 + 사람마다 prior 잔차 분포에서 뽑은 변환 오차 (둘 다 넣어 보수적)
         + 그 서버 사람이 4명 미만이면 「점 하나가 틀림」 혼합항 (EPS_WRONG·SD_WRONG, 거르기가 꺼지는 경우)
   (ii)  prior 모수: fit_prior의 사람 단위 붓스트랩 복제본 하나를 뽑는다
   (iii) 시즌 간 차이 (S41 하나로는 잴 수 없다 → 가정, 출력에 표시): P 수준 배수 exp(N(0, τ_P)),
         모양 기울임 Z·τ_shape·D(u) (D = S41 일섭–한섭 모양 차이), 빌린 서버 비율 exp(N(0, τ_ratio)).
         --p-update: 그 시즌의 같은 점수 쌍(같은 사람, 시각 등급 A·B)으로 P 수준 배수를 갱신한다 (정답 없이 되는 갱신).
         기본 끔 (2026-10-16): S41 자신에 돌리면(참 배수 = 1) T 18~84에서 배수가 1.1~1.3으로 끌려가고 sd가 지나치게 작았다
         — 수준 차이보다 P 시간 모양의 어긋남을 잡는다. S41로는 이 갱신이 맞는지 검증할 수 없다
4. 관측 0인 서버: 관측 있는 서버(사람 수 많은 쪽)의 곡선 × S41 서버 비율 × exp(N(0, τ_ratio)) — 등급 「근거 없음」.
   어느 서버도 관측이 없으면 S41 곡선 × exp(N(0, τ_scale0)) — 등급 「근거 없음」.

출력: 표준 출력에 사람이 읽는 표. --out DIR이면 DIR/predict_T{T}.json·predict_T{T}.txt.
"""
import argparse
import bisect
import builtins
import collections
import datetime as dt
import json
import math
import os
import random
import sys

FMT = "%Y-%m-%d %H:%M:%S"
SERVERS = ("KR", "JP", "GL")
SNAME = {"KR": "한섭", "JP": "일섭", "GL": "글섭"}
MAIN = (1.0, 3.0, 10.0)
AUX = (0.5, 2.0, 5.0, 15.0, 30.0)
# 출력에서의 구분 (2026-10-16): 주 결과는 1·3%, 0.5·2%는 보조, 5·10·15·30%는 참고 — 시즌 간 모양 가정에 기댄다
OUT_MAIN = (1.0, 3.0)
OUT_AUX = (0.5, 2.0)
OUT_REF = (5.0, 10.0, 15.0, 30.0)
UNKNOWN_MODES = ("none", "site", "mix")
UNKNOWN_DEFAULT = "site"     # 2026-10-16 tr 점검으로 정함 (docs/solo-raid-cutoff.md 「예측 도구」)
SITE_MIN = 10                # T까지 서버 라벨 사람이 이보다 적으면 사이트 비율을 빌린다 (--borrow-state)
SITE_M = 5.0                 # 사이트 비율의 디리클레 평활 (server_mix.PRIOR_M과 같은 값)
TARGETS = tuple(sorted(MAIN + AUX))
LEVELS = (0.50, 0.80, 0.95)
V0 = 0.003 ** 2          # 점수 바닥 분산 (ln 단위, 판독·반올림)
KAPPA = 0.01             # 모양 보간 자체 잡음: |ln q − ln y| 1당 ln 점수 0.01 (가정)
LN10 = math.log(10)
# 「점 하나가 틀림」 혼합항 (2026-10-16, 반박 검토 3): 그 서버 사람이 N_SCREEN명 미만이면(크게 어긋난 점 거르기가 꺼지는 경우)
# 뽑기마다 사람마다 확률 EPS_WRONG로 그 사람의 규모(ln 점수)를 N(0, SD_WRONG)만큼 옮긴다. 사람이 적으면 붓스트랩이 아무 일도 하지 않아
# 구간이 정보량을 반영하지 못했다 (1명 80% 폭 4% < 11명 7%). N_SCREEN명 이상이면 거르기(|z| > 4)가 그 역할을 한다.
# 값 근거 (S41, 결과를 보고 정함 — 그만큼 낙관적): tr 절단 T=24·48에서 사람마다 관측 하나로 낸 한섭 3% 컷의
# |ln 오차| > 0.03이 10/37·20/94명(약 20~27%), 그 rms 0.22.
EPS_WRONG = 0.20
SD_WRONG = 0.22
N_SCREEN = 4
# 모양 갱신 (--shape-update, 할 일 3.5b, 2026-10-17 분석 ④ — 기본 끔): 시즌 간 모양 기울임 Z를 그 시즌 자료로 추정한다.
# 점마다 c_i = ln s_i − ln r(y_i) = 규모 + Z·τ_shape·D(y_i) + 잡음을 가중 최소제곱으로 풀고 Z의 prior N(0, 1)로 당긴다.
# 잡음 = 변환 오차가 점수로 옮겨진 크기 + 바닥 SHAPE_PT_SD (다른 서버 섞임·판독), 사람마다 무게 합 1. |Z| ≤ Z_CAP.
SHAPE_PT_SD = 0.03
Z_CAP = 3.0
N_FEW = 3                # 쓴 사람(유효)이 이보다 적으면 「관측 적음 — 예측 보류 수준」 (결과는 낸다)
# 등급 C 문턱 (2026-10-16 재검토 r2 뒤): S41 tr 모의에서 6h −38.7% · 9h −17.0% · 12h −8.1% · 15h −3.2% · 18~48h ±0.5%였다.
# 18시간 문턱은 S41 한 시즌에서 본 값이라 낙관적이다 (15h → 18h가 가파르다. 다른 시즌 초반 진행이 몇 시간 다르면 18h도 −3% 쪽일 수 있다).
T_TRUST = 18.0
WIDTH_C = 0.15           # 1·3% 중 하나라도 80% 폭/중앙 > 15%면 C
GRADE_RULE = ("등급 기준: D 그 서버 관측 0 · C T<18시간(S41에서 본 문턱, 낙관) 또는 1·3% 80% 폭>15% 또는 쓴 사람<3 또는 % 변환 외삽 · "
              "B 쓴 사람<10, 라벨 사람<3(미상 줄에 기댐), 비율 빌림 · A 그 밖 (사유 중 높은 쪽)")
CONTAM_DEFAULT = (0.10, 0.20)   # 섞임 민감도: 미상 사람 중 뺄 몫


# ---------------- 누수 방지 ----------------

def _forbidden(path):
    return os.path.basename(os.fsdecode(path)).startswith("truth")


def safe_open(path, *a, **k):
    if _forbidden(path):
        raise PermissionError(f"predict는 정답 파일을 읽지 않는다: {path}")
    return open(path, *a, **k)


def install_guard():
    """이 프로세스의 open()이 truth* 파일을 못 열게 한다 (predict.py를 직접 돌릴 때)."""
    real = builtins.open

    def guarded(file, *a, **k):
        if isinstance(file, (str, bytes, os.PathLike)) and _forbidden(file):
            raise PermissionError(f"predict는 정답 파일을 읽지 않는다: {file}")
        return real(file, *a, **k)
    builtins.open = guarded


def jl(p):
    with safe_open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------- 시간 ----------------

def cstart_fallback_note(c_start):
    return (f"경고 (c_start 대체): season.json에 c_start가 없어 start 날짜 + 2일 05:00 = {c_start}로 대신했다 "
            "(--allow-cstart-fallback). start가 이벤트 시작이 아니라 챌린지 시작이면 c_start가 48시간 늦어 결과가 망가진다")


def season_info(state, allow_fallback=True):
    """season.json → (dict, c_start, 챌린지 길이 H). c_start가 없을 때: allow_fallback이면 start + 2일 05:00으로 대신하고
    dict에 c_start_assumed=True를 남긴다 (끝난 시즌 도구 fit_prior·backtest용 기본). predict.py 본체와 predict_run.sh는 False로 부른다."""
    with safe_open(os.path.join(state, "season.json"), encoding="utf-8") as f:
        s = json.load(f)
    if s.get("c_start"):
        c0 = dt.datetime.strptime(s["c_start"], FMT)
    else:
        if not allow_fallback:
            raise SystemExit(f"{state}/season.json에 c_start가 없다 — 챌린지 시작(\"c_start\": \"YYYY-MM-DD 05:00:00\")을 넣는다. "
                             "start + 2일 05:00으로 대신하려면 --allow-cstart-fallback (권하지 않음)")
        # 수집 도구와 같은 규칙: 이벤트 시작일 + 2일 05:00 (3일차 05:00)
        st0 = dt.datetime.strptime(s["start"], FMT)
        c0 = (st0 + dt.timedelta(days=2)).replace(hour=5, minute=0, second=0)
        s = dict(s, c_start=c0.strftime(FMT), c_start_assumed=True)
        print(cstart_fallback_note(s["c_start"]), file=sys.stderr)
    end = dt.datetime.strptime(s["end"], FMT)
    H = round((end - c0).total_seconds() / 3600)   # 04:59:59 → 120
    return s, c0, H


def parse_at(at, c0, H, notes=None):
    """T(시간 수 또는 KST 시각) → 챌린지 시작 후 시간. notes(목록)를 주면 경고·주의를 거기에도 남긴다 (출력 파일에 들어간다)."""
    notes = notes if notes is not None else []
    try:
        T = float(at)
    except ValueError:
        t = None
        for f in (FMT, "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
            try:
                t = dt.datetime.strptime(at, f)
                break
            except ValueError:
                pass
        if t is None:
            raise SystemExit(f"--at을 읽지 못했다: {at} (시간 수 또는 KST 'YYYY-MM-DD HH:MM')")
        T = (t - c0).total_seconds() / 3600
    if T > H:
        notes.append(f"주의: T={T:.2f}시간이 챌린지 길이 {H}시간보다 커서 {H}로 자른다 (종료 후 자료는 쓰지 않는다)")
        print(notes[-1], file=sys.stderr)
        T = float(H)
    if T <= 0:
        notes.append(f"경고: T={T:.2f}시간 — 챌린지 시작(c_start {c0.strftime(FMT)}) 전이다. 쓸 관측이 없어 S41 곡선 그대로(근거 없음)가 나온다")
        print(notes[-1], file=sys.stderr)
    return T


def hours_of(ts, c0):
    return (dt.datetime.strptime(ts, FMT) - c0).total_seconds() / 3600


# ---------------- % 변환 P(h) ----------------

def p_time(form, th, h, H):
    """같은 점수의 log10(최종 % / 지금 %), 서버 배수·x 항 없음."""
    if form == "step":   # 일차별 상수 (기준선 1)
        return th[min(max(int(h // 24), 0), len(th) - 1)]
    t = min(max(h, 0.25), H) / H
    if form == "sqrt":
        return th[0] * (1 - math.sqrt(t))
    if form == "pow":
        return math.exp(th[0]) * (1 - t) ** th[1]
    if form == "f2pow":   # F(h) = 지금 %/최종 % = w t^p + (1−w) t^q, F(0) = 0, F(H) = 1
        w = 1 / (1 + math.exp(-th[0]))
        F = w * t ** math.exp(th[1]) + (1 - w) * t ** math.exp(th[2])
        return -math.log10(max(F, 1e-12))
    if form == "zero":
        return 0.0
    raise ValueError(form)


def p_full(tr, server, h, x, H, m_season=1.0):
    """tr = 변환 모수 dict (form·theta·bx·m). 서버 배수 m과 시즌 배수를 곱한다."""
    base = p_time(tr["form"], tr["theta"], h, H)
    if tr.get("bx"):
        base *= max(0.0, 1 + tr["bx"] * math.log10(x))
    return m_season * tr["m"].get(server, 1.0) * base


def sigma_p(tr, server, h, x, H):
    P = p_full(tr, server, h, x, H)
    return math.sqrt(tr["s0"] ** 2 + (tr["rho"] * P) ** 2)


def sigma_form(t, h, H):
    """꼴 외삽 불확실성 (log10, 1σ): 맞춘 칸보다 이른 h에서 세 꼴(sqrt·pow·f2pow) P 차이(최대−최소)."""
    hs = t.get("h_support")
    if not hs or h >= hs["h_ramp"]:
        return 0.0
    v = [p_time(f, th, h, H) for f, th in t["alt_forms"].items()]
    fac = 1.0 if h <= hs["h_min"] else (hs["h_ramp"] - h) / max(hs["h_ramp"] - hs["h_min"], 1e-6)
    return (max(v) - min(v)) * fac


# ---------------- 최종 곡선 모양 ----------------

class Shape:
    """ln % 격자 위의 ln r(u) = ln(s(u)/s(3%)), D(u) (시즌 간 모양 차이 방향), flag."""

    def __init__(self, d):
        self.u = d["lnp"]
        self.lr = d["lr"]
        self.D = d["D"]
        self.flag = d["flag"]
        self.n = len(self.u)
        self.du = self.u[1] - self.u[0]

    def _idx(self, u):
        f = (u - self.u[0]) / self.du
        i = int(math.floor(f))
        if i < 0:
            return 0, 0.0
        if i >= self.n - 1:
            return self.n - 2, 1.0
        return i, f - i

    def inside(self, u):
        return self.u[0] <= u <= self.u[-1]

    def lr_at(self, u):
        i, t = self._idx(u)
        return self.lr[i] + t * (self.lr[i + 1] - self.lr[i])

    def D_at(self, u):
        i, t = self._idx(u)
        return self.D[i] + t * (self.D[i + 1] - self.D[i])

    def flag_at(self, u):
        i, t = self._idx(u)
        return self.flag[i if t < 0.5 else i + 1]

    def k_at(self, u, w=0.15):
        """k = −d ln r / d ln y (점수 탄력), 창 ±w로 평평한 칸의 0을 피한다. 바닥 0.005."""
        a, b = max(self.u[0], u - w), min(self.u[-1], u + w)
        if b <= a:
            return 0.005
        return max(0.005, -(self.lr_at(b) - self.lr_at(a)) / (b - a))


# ---------------- 관측 ----------------

def read_mix(path):
    """서버 확률 파일 → {key: {서버: p, ("OTHER": p)}}. 곡선 없는 서버(OTHER·NA·SEA 등)의 몫은 「이번 뽑기에서 쓰지 않음」."""
    mix = {}
    for d in jl(path):
        k = d.get("key") or (f"p{d['oi']}" if d.get("oi") is not None else None)
        p = d.get("probs") or d.get("p") or d.get("prob")
        if k and isinstance(p, dict):
            q = {s: v for s, v in p.items() if s in SERVERS and v and v > 0}
            oth = sum(v for s, v in p.items() if s not in SERVERS and v and v > 0)
            tot = sum(q.values()) + oth
            if q and tot > 0:
                mix[k] = {s: v / tot for s, v in q.items()}
                if oth > 0:
                    mix[k]["OTHER"] = oth / tot
    return mix


def site_ratio(pers):
    """pers: {(site, server): 사람 수} → {"*": {..}, site: {..}} (서버 SERVERS + OTHER).
    전체는 +1 평활, 사이트는 전체 쪽으로 디리클레 SITE_M 평활 (server_mix.prior_from_units와 같은 꼴, 단위는 사람)."""
    cl = list(SERVERS) + ["OTHER"]
    cnt = collections.Counter()
    by = collections.defaultdict(collections.Counter)
    for (site, srv), n in pers.items():
        c = srv if srv in SERVERS else "OTHER"
        cnt[c] += n
        by[site][c] += n
    N = sum(cnt.values())
    glob = {c: (cnt[c] + 1.0) / (N + len(cl)) for c in cl}
    out = {"*": glob}
    for site, sc in by.items():
        n = sum(sc.values())
        out[site] = {c: (sc[c] + SITE_M * glob[c]) / (n + SITE_M) for c in cl}
    return out, N


def borrowed_ratio(bstate, kind="label"):
    """끝난 시즌(S41) state에서 빌리는 사이트 비율. label = 시즌 전체 서버 라벨 사람 수(점수·% 안 씀),
    em = server_mix 요약의 em 사전(미상 줄의 곡면 우도로 잰 섞임 비율 — 점수·%를 쓴 값)."""
    if kind == "em":
        with safe_open(os.path.join(bstate, "post_server_mix_summary.json"), encoding="utf-8") as f:
            pr = json.load(f)["prior"]
        return {k: {c: v for c, v in d.items()} for k, d in pr.items()}, None
    pers = collections.defaultdict(set)
    for st in (bstate if isinstance(bstate, (list, tuple)) else [bstate]):   # 여러 시즌에서 빌릴 수 있다 (3.6 xval, 2026-10-16)
        for r in jl(os.path.join(st, "core.jsonl")):
            if r["src"] == "photo" and r["use"] and r["server"] and r["pct"] is not None and r["score"] is not None:
                pers[(r["site"], r["server"])].add((st, r["person"]))
    return site_ratio({k: len(v) for k, v in pers.items()})


def restrict(p, allowed):
    """서버 확률에서 allowed 밖 서버의 몫을 OTHER(쓰지 않음)로 옮긴다."""
    q = {s: v for s, v in p.items() if s in allowed and s in SERVERS and v > 0}
    oth = sum(v for s, v in p.items() if not (s in allowed and s in SERVERS) and v > 0)
    tot = sum(q.values()) + oth
    if not q or tot <= 0:
        return None
    out = {s: v / tot for s, v in q.items()}
    if oth > 0:
        out["OTHER"] = oth / tot
    return out


def load_obs(state, T, H, c0, grades="all", c_delay=0.0, server_mix=None, unknown=None):
    """게시 시각 ≤ T인 기본 집합 사진 관측. 반환 (관측 목록, 셈 dict).
    server_mix: (예전 방식) 서버 확률 파일 경로를 그대로 모든 서버에 쓴다 (기준 검사 없음).
    unknown: 서버 미상 줄 처리 dict — mode none|site|mix, servers(미상 줄을 넣을 서버, 기본 KR만), borrow_state·borrow_kind,
             mix_file·no_gate·fallback. 쓴 방식과 경고는 셈 dict의 "unknown"에 남긴다."""
    core = jl(os.path.join(state, "core.jsonl"))
    obs = jl(os.path.join(state, "obs.jsonl"))
    mix = read_mix(server_mix) if server_mix else {}
    if mix:   # 예전 방식도 미상 줄은 허용 서버(기본 한섭)에만 넣는다 — 일섭·글섭 오염 방지 (반박 검토 2)
        allowed = tuple((unknown or {}).get("servers") or ("KR",))
        mix = {k: q for k, q in ((k, restrict(p, allowed)) for k, p in mix.items()) if q}
    cnt = collections.Counter()
    rows = []
    lab_pers = collections.defaultdict(set)
    for r in core:
        if r["src"] != "photo" or not r["use"] or r["pct"] is None or r["score"] is None:
            continue
        if "best_record" in r["flag"]:
            cnt["drop_best_record"] += 1
            continue
        if r["day_lo"] == "post" or r["h"] is None:
            continue
        o = obs[r["oi"]]
        if o.get("img") is not None and r.get("no") is not None and str(o.get("no")) != str(r["no"]):
            raise SystemExit("core.jsonl의 oi가 obs.jsonl과 맞지 않는다")
        if not o.get("post_t"):
            cnt["drop_no_post_t"] += 1
            continue
        hp = hours_of(o["post_t"], c0)
        if hp > T:
            cnt["after_T"] += 1
            continue
        g = r["t_grade"]
        if grades == "AB" and g == "C":
            cnt["drop_grade_C"] += 1
            continue
        if g == "C":
            hh = r["h_hi"] if r["h_hi"] is not None else r["h"]
            if hh is None or hh < 0:   # 상한(게시 시각)이 챌린지 시작 전 — 챌린지 관측이 아니다 (전에는 h=0으로 잘라 썼다)
                cnt["drop_c_before_start"] += 1
                continue
            lo = r["h_lo"] if r["h_lo"] is not None else 0.0
            he = max(hh - c_delay, lo, 0.0)
        else:
            he = r["h"]
        if he > T + 1e-9 or he >= H or he < 0:
            cnt["drop_time"] += 1
            continue
        s = r["score_raw"] if r["score_raw"] is not None else r["score"]
        srv = r["server"] if r["server"] in SERVERS else None
        if r["server"] is not None:
            lab_pers[(r["site"], r["server"])].add(r["person"])
        if r["server"] is not None and srv is None:
            cnt["drop_other_server"] += 1
            continue
        pm = None
        if srv is None:
            pm = mix.get(r["key"])
            if not pm:
                cnt["unlabeled"] += 1
                rows.append({"key": r["key"], "oi": r["oi"], "server": None, "mix": None, "s": s, "x": r["pct"],
                             "h": he, "hp": hp, "grade": g, "person": r["person"], "unused": True, "site": r["site"]})
                continue
        rows.append({"key": r["key"], "oi": r["oi"], "server": srv, "mix": pm, "s": s, "x": r["pct"], "h": he,
                     "hp": hp, "grade": g, "person": r["person"], "unused": False, "site": r["site"]})
    if unknown and not server_mix:
        cnt["unknown"] = assign_unknown(state, rows, {k: len(v) for k, v in lab_pers.items()}, unknown)
        cnt["unlabeled"] = sum(1 for o in rows if o["unused"])
    # 시점 대표: 같은 사람·같은 %(0.01 단위)는 한 시점 (explore_pct_transform.dedupe_moments와 같은 규칙)
    grp = collections.defaultdict(list)
    for o in rows:
        grp[(o["person"], round(o["x"] / 0.01))].append(o)
    out = []
    for L in grp.values():
        L.sort(key=lambda o: (o["grade"] == "C", o["h"], o["oi"]))
        out.append(L[0])
    out.sort(key=lambda o: o["oi"])
    cnt["rows_le_T"] = len(rows)
    cnt["moments"] = len(out)
    # 서버 미상 줄 수: 줄(rows_assigned) ≠ 시점(같은 사람·같은 % 묶은 뒤) — 출력에 둘 다 적는다
    cnt["unknown_moments"] = sum(1 for o in out if not o["server"] and o["mix"])
    return out, cnt


def assign_unknown(state, rows, lab_pers, u):
    """서버 미상 줄(rows의 unused)에 서버 확률을 붙인다. 반환: 쓴 방식·근거·경고 dict.
    - none: 붙이지 않는다 (라벨 줄만)
    - site: 사이트별 서버 구성 비율 (점수·%를 쓰지 않음). T까지 라벨 사람으로 재되, SITE_MIN명 미만이면 borrow_state에서 빌린다
    - mix : server_mix.py 확률 (STATE/post_server_mix.jsonl). 요약의 gate가 통과일 때만 (no_gate면 늘). 아니면 fallback 방식
    servers: 미상 줄을 넣을 서버 (기본 KR만 — 일섭·글섭은 라벨 줄만, e1 규칙). 나머지 몫은 「쓰지 않음」."""
    mode = u.get("mode", "none")
    allowed = tuple(u.get("servers") or ("KR",))
    info = {"requested": mode, "used": mode, "servers": list(allowed), "warnings": [], "ratio_source": None,
            "label_persons": sum(lab_pers.values()), "label_persons_by": {f"{a}:{b}": n for (a, b), n in sorted(lab_pers.items())}}
    if mode == "mix":
        mf = u.get("mix_file") or os.path.join(state, "post_server_mix.jsonl")
        sf = os.path.splitext(mf)[0] + "_summary.json"
        gate = None
        if os.path.exists(sf):
            with safe_open(sf, encoding="utf-8") as f:
                gate = json.load(f).get("gate")
        info["gate"] = gate
        ok = bool(gate and gate.get("pass")) or u.get("no_gate")
        if not os.path.exists(mf):
            ok = False
            info["warnings"].append(f"서버 확률 파일 없음 ({mf})")
        elif os.path.getmtime(mf) < os.path.getmtime(os.path.join(state, "core.jsonl")):
            ok = False   # 시즌 전체 등 다른 core로 만든 파일을 쓰지 않는다 (반박 검토 2: T까지 자료로 다시 만든 파일만)
            info["warnings"].append("서버 확률 파일이 core.jsonl보다 오래됨 — server_mix.py를 지금 state로 다시 돌려야 한다")
        if ok:
            if not (gate and gate.get("pass")):
                info["warnings"].append("서버 확률 기준(gate) 미통과·미확인인데 --no-gate로 씀")
            mix = read_mix(mf)
            n = 0
            for o in rows:
                if o["unused"] and mix.get(o["key"]):
                    p = restrict(mix[o["key"]], allowed)
                    if p:
                        o["mix"], o["unused"] = p, False
                        n += 1
            info["rows_assigned"] = n
            return info
        fb = u.get("fallback", "site")
        info["warnings"].append(f"서버 확률 기준 미통과 ({(gate or {}).get('reason', '요약에 gate 없음')}) → {fb} 방식으로 바꿈")
        info["used"] = mode = fb
    if mode == "none":
        info["rows_assigned"] = 0
        return info
    # site
    ratio, N = site_ratio(lab_pers)
    info["ratio_source"] = f"T까지 라벨 사람 {N}명"
    if N < SITE_MIN:
        if u.get("borrow_state"):
            ratio, _ = borrowed_ratio(u["borrow_state"], u.get("borrow_kind", "label"))
            info["ratio_source"] = f"S41에서 빌림 ({u.get('borrow_kind', 'label')}, {u['borrow_state']}) — T까지 라벨 사람 {N}명 < {SITE_MIN}"
            info["warnings"].append(f"사이트 비율을 끝난 시즌에서 빌림 (T까지 라벨 사람 {N}명)")
        else:
            info["warnings"].append(f"T까지 라벨 사람 {N}명 < {SITE_MIN}인데 빌릴 state가 없어 그 라벨로 잰 비율을 씀")
    info["ratio"] = {k: {c: round(v, 3) for c, v in d.items()} for k, d in ratio.items()}
    n = 0
    for o in rows:
        if o["unused"]:
            p = restrict(ratio.get(o["site"], ratio["*"]), allowed)
            if p:
                o["mix"], o["unused"] = p, False
                n += 1
    info["rows_assigned"] = n
    return info


def same_score_pairs(rows, H, tr):
    """같은 사람·같은 정확 점수의 이웃 시점 쌍 (A·B 시각만, 1시간 이상 떨어짐). 서버 미상도 쓴다 (P 모양이 서버 공통이라는 가정)."""
    g = collections.defaultdict(list)
    for o in rows:
        if o["grade"] in ("A", "B"):
            g[(o["person"], round(o["s"], 4))].append(o)
    out = []
    for L in g.values():
        L.sort(key=lambda o: o["h"])
        for a, b in zip(L, L[1:]):
            if b["h"] - a["h"] >= 1 and b["x"] != a["x"]:
                srv = a["server"] or b["server"] or "KR"
                dm = p_full(tr, srv, a["h"], a["x"], H) - p_full(tr, srv, b["h"], b["x"], H)
                out.append({"d_obs": math.log10(b["x"] / a["x"]), "d_model": dm, "h": (a["h"], b["h"])})
    return out


def p_level_posterior(pairs, tau, sig):
    """ln m ~ N(0, τ²) 사전, Δ_obs ~ t4(m·Δ_model, σ_pair). 격자 사후 (ln m 값, 누적)."""
    grid = [i * 0.005 for i in range(-300, 301)]
    lp = []
    for lm in grid:
        m = math.exp(lm)
        v = -0.5 * (lm / tau) ** 2 if tau > 0 else (0.0 if lm == 0 else -1e9)
        for p in pairs:
            z = (p["d_obs"] - m * p["d_model"]) / sig
            v += -2.5 * math.log(1 + z * z / 4)
        lp.append(v)
    mx = max(lp)
    w = [math.exp(v - mx) for v in lp]
    tot = sum(w)
    cdf, acc = [], 0.0
    for v in w:
        acc += v / tot
        cdf.append(acc)
    mean = sum(lm * v for lm, v in zip(grid, w)) / tot
    sd = math.sqrt(sum((lm - mean) ** 2 * v for lm, v in zip(grid, w)) / tot)
    return {"grid": grid, "cdf": cdf, "mean_lnm": mean, "sd_lnm": sd}


def draw_post(post, rng):
    u = rng.random()
    i = bisect.bisect_left(post["cdf"], u)
    return math.exp(post["grid"][min(i, len(post["grid"]) - 1)])


# ---------------- 예측 ----------------

def qgrid():
    g = set(TARGETS)
    lo, hi, n = math.log(0.3), math.log(40.0), 36
    for i in range(n + 1):
        g.add(round(math.exp(lo + (hi - lo) * i / n), 4))
    return sorted(g)


def pav_dec(v):
    """값이 줄어들도록 (q 순서) 단조 회귀."""
    blocks = []
    for y in v:
        blocks.append([y, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    out = []
    for s, n in blocks:
        out += [s / n] * n
    return out


def quant(v, q):
    v = sorted(v)
    if not v:
        return None
    k = q * (len(v) - 1)
    i = int(math.floor(k))
    j = min(i + 1, len(v) - 1)
    if math.isinf(v[i]) or math.isinf(v[j]):   # 범위 밖(±inf)이 섞이면 가까운 순위 값
        return v[i] if k - i < 0.5 else v[j]
    return v[i] + (v[j] - v[i]) * (k - i)


class Predictor:
    """prior(들)과 관측으로 서버별 곡선 뽑기를 만든다. priors: 접힘(fold)별 prior 목록 (기본 1개). 관측의 'fold'가 고른다."""

    def __init__(self, prior, H, opts, priors=None):
        self.prior = prior
        self.priors = priors or [prior]
        self.H = H
        self.o = opts
        self.shapes = {s: Shape(prior["shape"]["servers"][s]) for s in SERVERS}
        self.G = qgrid()
        self.uG = [math.log(q) for q in self.G]
        sd = prior["season_diff"]
        self.tau_p = opts.get("tau_p", sd["tau_p"]["value"])
        self.tau_shape = opts.get("tau_shape", sd["tau_shape"]["value"])
        self.tau_ratio = opts.get("tau_ratio", sd["tau_ratio"]["value"])
        self.tau_scale0 = opts.get("tau_scale0", sd["tau_scale0"]["value"])
        if opts.get("no_season_diff"):
            self.tau_p = self.tau_shape = self.tau_ratio = 0.0
        self.lvl = opts.get("_lvl")   # (ln 시즌 수준 평균, sd) — early_level (3.5e)
        if self.lvl:
            self.tau_p = self.lvl[1]

    def tr_central(self, fold=0):
        t = self.priors[fold]["transform"]
        return {"form": t["form"], "theta": t["theta"], "bx": t.get("bx", 0.0), "m": t["m"],
                "s0": t["sigma"]["s0"], "rho": t["sigma"]["rho"]}

    def tr_boot(self, fold, j):
        t = self.priors[fold]["transform"]
        b = t["boot"][j]
        return {"form": t["form"], "theta": b["theta"], "bx": b.get("bx", 0.0), "m": b["m"],
                "s0": b["s0"], "rho": b["rho"]}

    def mode_tr(self, mode, fold=0):
        """기준선: b0 = P 0 (지금 % 그대로), b1 = 일차별 중앙 배수만 (3.4)."""
        tr = self.tr_central(fold)
        if mode == "b0":
            tr = dict(tr, form="zero", bx=0.0)
        elif mode == "b1":
            t = self.priors[fold]["transform"]
            tr = dict(tr, form="step", theta=t["day_median"]["KR"], bx=0.0,
                      m={s: t["day_median_m"].get(s, 1.0) for s in SERVERS})
        return tr

    # 서버 한 곳, 관측 목록 O(이미 그 서버), 한 뽑기
    def est_curve(self, srv, O, trs, m_season, Z, counts, zs, W, xi=0.0, shift=None):
        sh = self.shapes[srv]
        H = self.H
        lrG = [sh.lr[0] for _ in self.G]
        for gi, u in enumerate(self.uG):
            lrG[gi] = sh.lr_at(u) + Z * self.tau_shape * sh.D_at(u)
        num = [0.0] * len(self.G)
        den = [0.0] * len(self.G)
        for i, ob in enumerate(O):
            c = counts[i]
            if c == 0:
                continue
            tr = trs[ob.get("fold", 0)]
            P = p_full(tr, srv, ob["h"], ob["x"], H, m_season)
            if xi:
                P += xi * tr["m"].get(srv, 1.0) * sigma_form(self.priors[ob.get("fold", 0)]["transform"], ob["h"], H)
            if zs is not None:
                P += math.sqrt(tr["s0"] ** 2 + (tr["rho"] * P) ** 2) * zs[i]
            P = max(P, 0.0)
            uy = math.log(ob["x"]) + P * LN10
            if not sh.inside(uy):
                continue
            ci = math.log(ob["s"]) - (sh.lr_at(uy) + Z * self.tau_shape * sh.D_at(uy))
            if shift is not None:
                ci += shift[i]
            wi = W[i]
            for gi in range(len(self.G)):
                w = wi[gi] * c
                num[gi] += w * ci
                den[gi] += w
        if not any(den):
            return None
        if self.o.get("robust") == "wmedian":
            est = self.wmedian_curve(srv, O, trs, m_season, Z, counts, zs, W, xi, lrG, shift)
            if est is not None:
                return pav_dec(est)
        est = [lrG[gi] + num[gi] / den[gi] for gi in range(len(self.G))]
        return pav_dec(est)

    def wmedian_curve(self, srv, O, trs, m_season, Z, counts, zs, W, xi, lrG, shift=None):
        """가중 평균 대신 가중 중앙값 (같은 가중치). 다른 서버 상위권이 섞일 때의 강건 처리 비교용 (--robust wmedian)."""
        sh = self.shapes[srv]
        H = self.H
        pts = []
        for i, ob in enumerate(O):
            c = counts[i]
            if c == 0:
                continue
            tr = trs[ob.get("fold", 0)]
            P = p_full(tr, srv, ob["h"], ob["x"], H, m_season)
            if xi:
                P += xi * tr["m"].get(srv, 1.0) * sigma_form(self.priors[ob.get("fold", 0)]["transform"], ob["h"], H)
            if zs is not None:
                P += math.sqrt(tr["s0"] ** 2 + (tr["rho"] * P) ** 2) * zs[i]
            P = max(P, 0.0)
            uy = math.log(ob["x"]) + P * LN10
            if not sh.inside(uy):
                continue
            pts.append((math.log(ob["s"]) - (sh.lr_at(uy) + Z * self.tau_shape * sh.D_at(uy))
                        + (shift[i] if shift is not None else 0.0), i, c))
        if not pts:
            return None
        pts.sort()
        est = []
        for gi in range(len(self.G)):
            ws = [W[i][gi] * c for _, i, c in pts]
            half = 0.5 * sum(ws)
            acc = 0.0
            for (ci, _, _), w in zip(pts, ws):
                acc += w
                if acc >= half:
                    est.append(lrG[gi] + ci)
                    break
        return est

    def z_update(self, srv, O, trs, m_season, counts, zs, xi):
        """그 서버 점들로 모양 기울임 Z를 추정 → (Ẑ, 사후 sd) 또는 None (3.5b, --shape-update)."""
        sh = self.shapes[srv]
        tau = self.tau_shape
        if tau <= 0:
            return None
        npr = collections.Counter(o["person"] for o, c in zip(O, counts) if c)
        Sw = SD = SDD = Sc = ScD = 0.0
        for i, ob in enumerate(O):
            c = counts[i]
            if not c:
                continue
            tr = trs[ob.get("fold", 0)]
            P = p_full(tr, srv, ob["h"], ob["x"], self.H, m_season)
            if xi:
                P += xi * tr["m"].get(srv, 1.0) * sigma_form(self.priors[ob.get("fold", 0)]["transform"], ob["h"], self.H)
            if zs is not None:
                P += math.sqrt(tr["s0"] ** 2 + (tr["rho"] * P) ** 2) * zs[i]
            P = max(P, 0.0)
            uy = math.log(ob["x"]) + P * LN10
            if not sh.inside(uy):
                continue
            k = sh.k_at(uy)
            sp = math.sqrt(tr["s0"] ** 2 + (tr["rho"] * P) ** 2)
            sf = sigma_form(self.priors[ob.get("fold", 0)]["transform"], ob["h"], self.H)
            v = (k * LN10) ** 2 * (sp ** 2 + sf ** 2) + V0 + SHAPE_PT_SD ** 2
            w = c / (v * npr[ob["person"]])
            d = tau * sh.D_at(uy)
            ci = math.log(ob["s"]) - sh.lr_at(uy)
            Sw += w
            SD += w * d
            SDD += w * d * d
            Sc += w * ci
            ScD += w * ci * d
        if Sw <= 0:
            return None
        var = max(SDD - SD * SD / Sw, 0.0)
        cov = ScD - Sc * SD / Sw
        prec = var + 1.0
        z = max(-Z_CAP, min(Z_CAP, cov / prec))
        return z, 1.0 / math.sqrt(prec)

    def weights(self, srv, O, tr):
        sh = self.shapes[srv]
        W, ys = [], []
        for ob in O:
            P = p_full(tr, srv, ob["h"], ob["x"], self.H)
            uy = math.log(ob["x"]) + P * LN10
            ys.append(uy)
            if not sh.inside(uy):
                W.append([0.0] * len(self.G))
                continue
            k = sh.k_at(uy)
            sp = math.sqrt(tr["s0"] ** 2 + (tr["rho"] * P) ** 2)
            sf = sigma_form(self.prior["transform"], ob["h"], self.H)
            vt = (k * LN10) ** 2 * (sp ** 2 + (self.tau_p * P) ** 2 + sf ** 2)
            Dy = sh.D_at(uy)
            row = []
            for gi, u in enumerate(self.uG):
                vs = (self.tau_shape * (sh.D_at(u) - Dy)) ** 2 + (KAPPA * (u - uy)) ** 2
                row.append(1.0 / (vt + vs + V0))
            W.append(row)
        return W, ys

    def screen(self, srv, O, tr, W, ys):
        """한 점씩 빼고 나머지의 가중 평균과 견줘 크게 어긋난 점(|z| > 4)을 뺀다 (사람 4명 이상일 때)."""
        sh = self.shapes[srv]
        if len({o["person"] for o in O}) < N_SCREEN:
            return O, []
        c = []
        for ob, uy in zip(O, ys):
            c.append(math.log(ob["s"]) - sh.lr_at(uy) if sh.inside(uy) else None)
        keep, drop = [], []
        for i, ob in enumerate(O):
            if c[i] is None:
                keep.append(ob)
                continue
            gi = min(range(len(self.G)), key=lambda g: abs(self.uG[g] - ys[i]))
            num = den = 0.0
            for j in range(len(O)):
                if j == i or c[j] is None or O[j]["person"] == ob["person"]:
                    continue
                num += W[j][gi] * c[j]
                den += W[j][gi]
            if den == 0:
                keep.append(ob)
                continue
            z = (c[i] - num / den) / math.sqrt(1 / W[i][gi] + 1 / den)
            (drop if abs(z) > 4 else keep).append(ob)
            if abs(z) > 4:
                ob["screen_z"] = round(z, 2)
        return keep, drop

    def run(self, rows, comps=("i", "ii", "iii"), draws=1000, seed=1, mode="pt"):
        """반환: {서버: {"draws": [곡선(ln S, G 순서)...], 기타}}"""
        rng = random.Random(seed)
        H = self.H
        trc = [self.mode_tr(mode, f) if mode != "pt" else self.tr_central(f) for f in range(len(self.priors))]
        by = {s: [] for s in SERVERS}
        mixed = []
        for ob in rows:
            if ob["unused"]:
                continue
            if ob["server"]:
                by[ob["server"]].append(ob)
            elif ob["mix"]:
                mixed.append(ob)
        # 서버 확률 줄은 그 서버 후보마다 넣고 뽑기마다 하나만 켠다
        cand = {s: list(by[s]) for s in SERVERS}
        for ob in mixed:
            for s in ob["mix"]:
                if s in cand:
                    cand[s].append(ob)
        info = {}
        prep = {}
        for s in SERVERS:
            O = cand[s]
            if not O:
                continue
            W, ys = self.weights(s, O, trc[0])
            topcut = []
            if self.o.get("top_cut"):
                # 꼭대기 제외: 옮긴 최종 %가 top_cut보다 작은(더 상위) 서버 미상 줄은 쓰지 않는다 (다른 서버 최상위가 섞이는 것 대비)
                keep = [k for k, (o, u) in enumerate(zip(O, ys)) if o["server"] or math.exp(u) >= self.o["top_cut"]]
                if len(keep) < len(O):
                    topcut = [O[k] for k in range(len(O)) if k not in set(keep)]
                    O = [O[k] for k in keep]
                    if not O:
                        continue
                    W, ys = self.weights(s, O, trc[0])
            if mode == "pt":
                O2, dropped = self.screen(s, O, trc[0], W, ys)
                if dropped:
                    W, ys = self.weights(s, O2, trc[0])
                    O = O2
            else:
                dropped = []
            persons = sorted({o["person"] for o in O})
            pidx = {p: k for k, p in enumerate(persons)}
            pf = {o["person"]: o.get("fold", 0) for o in O}
            prep[s] = {"O": O, "W": W, "ys": ys, "persons": persons, "pidx": pidx, "pfold": [pf[p] for p in persons]}
            info[s] = {"dropped": [{"key": o["key"], "z": o.get("screen_z")} for o in dropped], "top_cut": len(topcut),
                       "y_central": [math.exp(u) for u in ys],
                       "outside_shape": sum(1 for u in ys if not self.shapes[s].inside(u))}
        # 시즌 P 수준 갱신 (정답 없이: 같은 점수 쌍)
        post = None
        if mode == "pt" and "iii" in comps and self.o.get("p_update", False) and self.tau_p > 0:
            pairs = same_score_pairs([o for o in rows], H, trc[0])
            if pairs:
                post = p_level_posterior(pairs, self.tau_p, self.prior["transform"]["pair_sigma"])
                post["n_pairs"] = len(pairs)
        zres = [p["transform"]["z_resid"] for p in self.priors]
        nb = [len(p["transform"]["boot"]) for p in self.priors]
        res = {s: [] for s in SERVERS}
        central = mode != "pt" or draws == 0
        nd = 1 if central else draws
        for b in range(nd):
            xi = 0.0
            if not central and "ii" in comps:
                trs = [self.tr_boot(f, rng.randrange(nb[f])) for f in range(len(self.priors))]
                xi = rng.gauss(0, 1)
            else:
                trs = trc
            m_season, Z, rnoise = (math.exp(self.lvl[0]) if self.lvl else 1.0), 0.0, {}
            if not central and "iii" in comps:
                if self.lvl:
                    m_season = math.exp(self.lvl[0] + rng.gauss(0, self.lvl[1]))
                elif post:
                    m_season = draw_post(post, rng)
                elif self.tau_p > 0:
                    m_season = math.exp(rng.gauss(0, self.tau_p))
                Z = rng.gauss(0, 1)
                rnoise = {s: rng.gauss(0, self.tau_ratio) for s in SERVERS}
            # 서버 확률 줄: 이번 뽑기의 서버. 중앙 추정(central — 점 추정·기준선0·1·섞임 민감도)은 뽑지 않고 확률을 가중치로 쓴다
            # (전에는 한 번 무작위로 뽑아 시드·줄 순서에 따라 흔들렸다, 재검토 r2 6)
            pick = {}
            if mixed and not central:
                for ob in mixed:
                    u, acc = rng.random(), 0.0
                    for s, p in ob["mix"].items():
                        acc += p
                        if u <= acc:
                            pick[ob["key"]] = s
                            break
                    else:
                        pick[ob["key"]] = list(ob["mix"])[-1]
            for s, pp in prep.items():
                O = pp["O"]
                if central:
                    act = [1.0 if o["server"] else (o["mix"] or {}).get(s, 0.0) for o in O]
                else:
                    act = [1 if (o["server"] or pick.get(o["key"]) == s) else 0 for o in O]
                if not central and "i" in comps:
                    npers = len(pp["persons"])
                    pc = [0] * npers
                    for _ in range(npers):
                        pc[rng.randrange(npers)] += 1
                    zp = [rng.choice(zres[pp["pfold"][k]]) for k in range(npers)]
                    counts = [act[i] * pc[pp["pidx"][o["person"]]] for i, o in enumerate(O)]
                    zs = [zp[pp["pidx"][o["person"]]] for o in O]
                    shift = None
                    if self.o.get("eps_wrong", EPS_WRONG) > 0 and npers < N_SCREEN:
                        ps = [rng.gauss(0, SD_WRONG) if rng.random() < self.o.get("eps_wrong", EPS_WRONG) else 0.0
                              for _ in range(npers)]
                        shift = [ps[pp["pidx"][o["person"]]] for o in O]
                else:
                    counts, zs, shift = act, None, None
                if not any(counts):
                    counts = act
                Zs = Z
                if self.o.get("shape_update"):
                    zu = self.z_update(s, O, trs, m_season, counts, zs, xi)
                    if zu is not None:
                        Zs = zu[0] + zu[1] * Z
                        if central:
                            info[s]["z_shape"] = [round(zu[0], 3), round(zu[1], 3)]
                cur = self.est_curve(s, O, trs, m_season, Zs, counts, zs, pp["W"], xi, shift)
                if cur is not None:
                    res[s].append(cur)
            # 관측 없는 서버
            for s in SERVERS:
                if s in prep and res[s] and len(res[s]) == b + 1:
                    continue
                donors = [d for d in SERVERS if d in prep and len(res[d]) == b + 1 and d != s]
                if donors:
                    d = max(donors, key=lambda d: len(prep[d]["persons"]))
                    cur = [res[d][-1][gi] + math.log(self.ratio(s, d, q)) + rnoise.get(s, 0.0)
                           for gi, q in enumerate(self.G)]
                    if s in prep:   # 관측은 있는데 이번 뽑기에서 켜진 줄이 없음 (서버 확률 줄만 있을 때) — 근거 없음으로 치지 않는다
                        info[s]["donor_draws"] = info[s].get("donor_draws", 0) + 1
                    else:
                        info.setdefault(s, {})["donor"] = d
                else:
                    sh = self.shapes[s]
                    sc = math.log(self.prior["shape"]["s3_S41"][s])
                    e0 = 0.0 if central or "iii" not in comps else rng.gauss(0, self.tau_scale0)
                    cur = pav_dec([sc + sh.lr_at(u) + Z * self.tau_shape * sh.D_at(u) + e0 for u in self.uG])
                    if s in prep:
                        info[s]["donor_draws"] = info[s].get("donor_draws", 0) + 1
                    else:
                        info.setdefault(s, {})["donor"] = "S41"
                res[s].append(cur)
        return {"draws": res, "info": info, "post": post, "prep": prep}

    def ratio(self, s, d, q):
        """S41 서버 비율 s(q)_s / s(q)_d (모양 격자에서)."""
        a, b = self.shapes[s], self.shapes[d]
        u = math.log(q)
        return math.exp(a.lr_at(u) - b.lr_at(u)) * self.prior["shape"]["s3_S41"][s] / self.prior["shape"]["s3_S41"][d]


def summarize_draws(G, draws):
    """곡선 뽑기 → q마다 중앙값과 50·80·95% 구간 (억)."""
    out = {}
    for gi, q in enumerate(G):
        v = [math.exp(c[gi]) for c in draws]
        d = {"median": quant(v, 0.5)}
        for L in LEVELS:
            d[f"i{int(L * 100)}"] = [quant(v, (1 - L) / 2), quant(v, 1 - (1 - L) / 2)]
        out[q] = d
    return out


def pct_at_score(G, curve, S):
    """단조 곡선(ln S, q 순서)에서 점수 S의 최종 % (ln q 선형). 범위 밖이면 ±inf."""
    ls = math.log(S)
    if ls > curve[0]:
        return -math.inf
    if ls < curve[-1]:
        return math.inf
    for i in range(len(G) - 1):
        a, b = curve[i], curve[i + 1]
        if a >= ls >= b:
            if a == b:
                return G[i]
            t = (a - ls) / (a - b)
            return math.exp(math.log(G[i]) + t * (math.log(G[i + 1]) - math.log(G[i])))
    return math.inf


def score_table(G, draws, med):
    """점수 → 최종 % 표: 0.5%~30% 예측 중앙 점수 사이를 고르게 나눈 점수 줄."""
    hi, lo = med[0.5]["median"], med[30.0]["median"]
    step = max(1.0, round((hi - lo) / 14))
    step = min((1, 2, 5, 10, 20), key=lambda s: abs(s - step))
    rows = []
    S = math.floor(hi / step) * step + step
    while S >= lo - step and S > 0:   # 30% 중앙이 한 칸보다 작으면 0 이하로 내려가지 않게 (2026-10-16, 여러 시즌 prior 일섭)
        v = [pct_at_score(G, c, S) for c in draws]
        rows.append({"score": S, "median": quant(v, 0.5), "i80": [quant(v, 0.1), quant(v, 0.9)],
                     "i50": [quant(v, 0.25), quant(v, 0.75)]})
        S -= step
    return rows


def implied3(pr, s, O, tr):
    """사람 → 함축 ln 3% 점수: 그 사람 점을 중앙 변환으로 최종 %로 옮기고 S41 모양으로 3%에 댄 값(ln s − ln r(y) + ln r(3%))의 중앙값."""
    sh = pr.shapes[s]
    l3 = sh.lr_at(math.log(3.0))
    by = collections.defaultdict(list)
    for ob in O:
        P = max(p_full(tr, s, ob["h"], ob["x"], pr.H), 0.0)
        uy = math.log(ob["x"]) + P * LN10
        if sh.inside(uy):
            by[ob["person"]].append(math.log(ob["s"]) - sh.lr_at(uy) + l3)
    out = {}
    for p, v in by.items():
        v.sort()
        n = len(v)
        out[p] = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    return out


def contam_sensitivity(pr, rows, s, fracs, cen_curve, seed):
    """섞임 민감도 (재검토 r2 2): 서버 미상 줄에 다른 서버(일섭·글섭)가 f 몫 섞였다고 보고, 미상 사람 중 함축 3% 점수가 가장 낮은 f 몫을 뺀
    중앙 추정의 변화(뺀 것 / 전부 − 1)를 목표 %마다 낸다. 다른 서버 화면은 같은 %에서 점수가 낮아 한섭 컷을 늘 아래로 끈다 → 빼면 오른다.
    중앙 추정(뽑기 없음, 서버 확률을 가중치로)이라 결정적이고 몬테카를로 잡음이 없다. 섞임이 없으면 이 값은 위쪽으로 치우친 과대 보정이다."""
    tr = pr.tr_central(0)
    unk = [o for o in rows if not o["unused"] and not o["server"] and o.get("mix") and s in o["mix"]]
    imp = implied3(pr, s, unk, tr)
    persons = sorted(imp, key=lambda p: (imp[p], str(p)))
    n = len(persons)
    res = {"method": "미상 사람 중 함축 3% 점수가 가장 낮은 f 몫을 빼고 다시 낸 중앙 추정 (섞임은 늘 아래로 끈다)",
           "n_unknown_persons": n, "f": {}}
    for f in fracs:
        k = int(round(f * n))
        e = {"k": k, "points": {}}
        if k == 0:
            e["note"] = f"미상 사람 {n}명이라 {f * 100:g}% 몫이 0명"
            res["f"][f"{f:g}"] = e
            continue
        drop = set(persons[:k])
        rows2 = [o for o in rows if not (not o["server"] and o.get("mix") and o["person"] in drop)]
        c2 = pr.run(rows2, (), 0, seed)["draws"][s]
        if not c2:
            e["note"] = "뺀 뒤 쓸 점 없음"
            res["f"][f"{f:g}"] = e
            continue
        for gi, q in enumerate(pr.G):
            if q in TARGETS:
                e["points"][str(q)] = {"central": math.exp(c2[0][gi]), "shift": math.exp(c2[0][gi] - cen_curve[gi]) - 1}
        res["f"][f"{f:g}"] = e
    return res


EARLY_MIN_TRAIN = 5


def early_level_prior(prior, state, T, H, cfg):
    """초반 단면 침수로 시즌 수준을 정한다 (3.5e, 2026-10-17 분석 ④ — 기본 끔).
    cfg = {"train": [[시즌, 그 시즌의 T 절단 state]...]}. 학습 시즌마다 같은 T까지 자료로 m̂(explore_early_level.early_m, 라벨+미상 줄, 정답 없음)를 내고
    prior의 season_levels(m_s)와 ln m_s = α + β ln m̂로 맞춘다. 맞힐 시즌 m̂를 넣어 ln 수준의 평균·예측 sd(바닥 0.05)를 낸다.
    학습 시즌이 EARLY_MIN_TRAIN 미만이거나 m̂가 없으면 None (τ_P 그대로)."""
    import explore_early_level as EL
    t = prior["transform"]
    lv = t.get("season_levels") or {}
    th, form = t["theta"], t["form"]

    def mhat(st):
        _, c0_, H_ = season_info(st)
        r = EL.early_m(st, min(T, H_), th, form, H_, False, 0, 1)
        return r.get("m")
    pts = []
    for name, st in cfg.get("train", []):
        if name in lv:
            m = mhat(st)
            if m and m > 0:
                pts.append((math.log(m), math.log(lv[name])))
    m0 = mhat(state)
    info = {"n_train": len(pts), "m_hat": m0, "mu": None, "sd": None}
    if len(pts) < EARLY_MIN_TRAIN or not m0 or m0 <= 0:
        return info
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx > 0 else 0.0
    a = my - b * mx
    n = len(pts)
    s2 = sum((y - a - b * x) ** 2 for x, y in pts) / max(n - 2, 1)
    x0 = math.log(m0)
    sd = math.sqrt(s2 * (1 + 1 / n + ((x0 - mx) ** 2 / sxx if sxx > 0 else 0.0)))
    info.update(mu=round(a + b * x0, 5), sd=round(max(sd, 0.05), 5), alpha=round(a, 4), beta=round(b, 4))
    return info


def predict(state, prior, T, opts, draws=2000, seed=1, comps_split=True, priors=None, rows=None):
    """predict.py의 본체. backtest.py도 이 함수를 쓴다. truth* 파일은 safe_open이 막는다."""
    season, c0, H = season_info(state, allow_fallback=opts.get("allow_cstart_fallback", False))
    warns = list(opts.get("notes") or [])
    if season.get("c_start_assumed"):
        warns.insert(0, cstart_fallback_note(season["c_start"]))
    if rows is None:
        rows, cnt = load_obs(state, T, H, c0, opts.get("grades", "all"), opts.get("c_delay", 0.0),
                             opts.get("server_mix"), opts.get("unknown"))
    else:
        cnt = collections.Counter({"moments": len(rows)})
    lvl_info = None
    if opts.get("early_level"):
        lvl_info = early_level_prior(prior, state, T, H, opts["early_level"])
        if lvl_info.get("mu") is not None:
            opts = dict(opts, _lvl=(lvl_info["mu"], lvl_info["sd"]))
    pr = Predictor(prior, H, opts, priors)
    full = pr.run(rows, ("i", "ii", "iii"), draws, seed)
    comp = {}
    if comps_split:
        for c in ("i", "ii", "iii"):
            comp[c] = pr.run(rows, (c,), max(200, draws // 4), seed + 7)
    cen = pr.run(rows, (), 0, seed)
    # 라벨 줄만 쓴 결과를 나란히 (서버 미상 줄을 썼을 때, 반박 검토 2)
    lab_only = None
    if any(o.get("mix") and not o["server"] and not o["unused"] for o in rows):
        lab_only = pr.run([dict(o) for o in rows if o["server"]], ("i", "ii", "iii"), max(300, draws // 4), seed + 3)
    base = {m: pr.run(rows, (), 0, seed, mode=m) for m in ("b0", "b1")}
    out = {"season": season["name"], "T": T, "T_kst": (c0 + dt.timedelta(hours=T)).strftime("%Y-%m-%d %H:%M"),
           "c_start": season["c_start"], "c_start_assumed": bool(season.get("c_start_assumed")), "warnings": warns,
           "H": H, "draws": draws, "seed": seed,
           "settings": {"grades": opts.get("grades", "all"), "c_delay": opts.get("c_delay", 0.0),
                        "server_mix": opts.get("server_mix"), "p_update": opts.get("p_update", False),
                        "unknown": opts.get("unknown"), "robust": opts.get("robust", "mean"), "top_cut": opts.get("top_cut"),
                        "tau_p": pr.tau_p, "tau_shape": pr.tau_shape, "tau_ratio": pr.tau_ratio,
                        "tau_scale0": pr.tau_scale0, "prior_season": prior["season"],
                        "prior_form": prior["transform"]["form"]},
           "counts": {k: v for k, v in cnt.items() if k != "unknown"}, "unknown": cnt.get("unknown"),
           "p_update": None, "early_level": lvl_info, "servers": {}}
    if full["post"]:
        p = full["post"]
        out["p_update"] = {"n_pairs": p["n_pairs"], "m_mean": math.exp(p["mean_lnm"]), "sd_lnm": p["sd_lnm"],
                           "prior_tau_p": pr.tau_p}
    for s in SERVERS:
        D = full["draws"][s]
        summ = summarize_draws(pr.G, D)
        prep = full["prep"].get(s)
        n_obs = len(prep["O"]) if prep else 0
        # 사람 수: 라벨 줄의 사람 + 서버 확률 줄의 기대 사람 수(Σ 사람별 최대 p)
        n_lab = len({o["person"] for o in prep["O"] if o["server"]}) if prep else 0
        pm = collections.defaultdict(float)
        if prep:
            for o in prep["O"]:
                if not o["server"]:
                    pm[o["person"]] = max(pm[o["person"]], o["mix"].get(s, 0.0))
        n_pers = n_lab + sum(pm.values())
        inf = full["info"].get(s, {})
        donor = inf.get("donor")
        # 근거 종류 (basis). 믿을 정도의 「등급」은 warning(A~D) 하나뿐이다 (재검토 r2 6: 표기를 하나로)
        if n_pers >= 5:
            basis = "관측 기반"
        elif n_pers > 0:
            basis = "관측 적음"
        elif donor == "S41":
            basis = "근거 없음 (S41 곡선 그대로 + 가정 폭)"
        else:
            basis = f"근거 없음 ({SNAME.get(donor, donor)} 곡선 × S41 서버 비율)"
        ys = inf.get("y_central", [])
        if prep and all(o["h"] < pr.prior["transform"]["h_support"]["h_min"] for o in prep["O"]):
            basis += " · 모든 관측이 S41 % 변환 자료보다 이른 시각 (외삽)"
        sv = {"basis": basis, "n_moments": n_obs, "n_persons": round(n_pers, 1), "n_persons_label": n_lab,
              "n_persons_mix_expected": round(sum(pm.values()), 1), "donor": donor,
              "grades": dict(collections.Counter(o["grade"] for o in prep["O"])) if prep else {},
              "mixed_rows": sum(1 for o in prep["O"] if not o["server"]) if prep else 0,
              "h_range": [min(o["h"] for o in prep["O"]), max(o["h"] for o in prep["O"])] if prep else None,
              "x_range": [min(o["x"] for o in prep["O"]), max(o["x"] for o in prep["O"])] if prep else None,
              "y_central_range": [min(ys), max(ys)] if ys else None,
              "screened_out": inf.get("dropped", []), "outside_shape": inf.get("outside_shape", 0),
              "fill_draws_share": round(inf.get("donor_draws", 0) / max(1, len(D)), 3),
              "points": {}}
        sh = pr.shapes[s]
        for q in TARGETS:
            d = dict(summ[q])
            u = math.log(q)
            near = sum(1 for y in ys if abs(math.log(y) - u) <= math.log(2))
            d["near_obs"] = near
            d["shape_flag"] = sh.flag_at(u)
            notes = []
            if d["shape_flag"] != "own":
                notes.append(f"S41 모양 {d['shape_flag']}")
            if prep and near == 0:
                notes.append("옮긴 관측이 ×2 안에 없음 (모양 가정 의존)")
            d["notes"] = notes
            gi = pr.G.index(q)
            d["central"] = math.exp(cen["draws"][s][0][gi]) if cen["draws"][s] else None
            for m in ("b0", "b1"):
                d[m] = math.exp(base[m]["draws"][s][0][gi]) if base[m]["draws"][s] else None
            if comps_split:
                d["comp80"] = {}
                for c in ("i", "ii", "iii"):
                    cs = summarize_draws([q], [[cc[gi]] for cc in comp[c]["draws"][s]])[q]
                    d["comp80"][c] = cs["i80"]
            sv["points"][str(q)] = d
        if lab_only is not None:
            lo = summarize_draws(pr.G, lab_only["draws"][s])
            lp = lab_only["prep"].get(s)
            sv["label_only"] = {"n_persons": len(lp["persons"]) if lp else 0,
                                "donor": lab_only["info"].get(s, {}).get("donor"),
                                "points": {str(q): {"median": lo[q]["median"], "i80": lo[q]["i80"]} for q in OUT_MAIN + OUT_AUX}}
        sv["table"] = score_table(pr.G, D, summ)
        if sv["mixed_rows"] and not donor and cen["draws"][s] and s in ((out.get("unknown") or {}).get("servers") or ["KR"]):
            cs = contam_sensitivity(pr, rows, s, opts.get("contam", CONTAM_DEFAULT), cen["draws"][s][0], seed)
            for e in cs["f"].values():
                for q, pq in e["points"].items():
                    lo80, hi80 = sv["points"][q]["i80"]
                    pq["i80_ext"] = [lo80, hi80 * (1 + max(pq["shift"], 0.0))]   # 위로만 넓힌 80% (참고, 채점 안 함)
            sv["contam"] = cs
        sv["warning"] = warning(T, s, sv, out.get("unknown"), pr.prior)
        out["servers"][s] = sv
    out["assumptions"] = assumptions(prior, pr)
    return out


WARN_LABEL = {"A": "A 보통", "B": "B 주의", "C": "C 강한 주의", "D": "D 숫자만 참고 (근거 없음)"}


def warning(T, s, sv, unk, prior):
    """서버별 등급 (언제나 예측하되 믿을 정도를 분명히 — 사용자 결정 2026-10-16). 높은 등급 사유가 등급을 정한다. 기준은 GRADE_RULE 한 줄.
    기준 숫자(3·10명, 18시간, 폭 15%)는 S41 tr 점검을 본 뒤 정했다 (docs/solo-raid-cutoff.md 「예측 도구」) — 그만큼 낙관적이다.
    2026-10-16 재검토 r2 뒤: 12~18시간이 B였는데 S41 모의 12h −8.1%·15h −3.2%였다 → T < 18이면 C, 1·3% 80% 폭 > 15%여도 C."""
    R = []
    h_min = prior["transform"]["h_support"]["h_min"]
    if sv["donor"]:
        R.append(("D", "이 서버 관측 0 — " + ("S41 곡선 그대로" if sv["donor"] == "S41" else
                                             f"{SNAME.get(sv['donor'], sv['donor'])} 곡선 × S41 서버 비율")))
    if T < h_min:
        R.append(("C", f"T {T:.1f}시간 < {h_min:.1f}시간: S41 % 변환 자료보다 이른 시각 (외삽, 꼴 차이만큼 넓힘)"))
    elif sv["h_range"] and sv["h_range"][1] < h_min:
        R.append(("C", "모든 관측이 S41 % 변환 자료보다 이른 시각 (외삽)"))
    elif T < T_TRUST:
        R.append(("C", f"T {T:.1f}시간 < {T_TRUST:g}시간: 초반 % 변환 근거가 얇다 (S41 첫 24시간 한섭 A 8줄, 모의 12h −8.1%·15h −3.2%). "
                       f"{T_TRUST:g}시간 문턱은 S41에서 본 값이라 낙관적이다"))
    if not sv["donor"]:
        wide = []
        for q in OUT_MAIN:
            d = sv["points"].get(str(q)) or {}
            if d.get("median") and d.get("i80") and None not in d["i80"]:
                w = (d["i80"][1] - d["i80"][0]) / d["median"]
                if w > WIDTH_C:
                    wide.append(f"{q:g}% {w * 100:.0f}%")
        if wide:
            R.append(("C", f"80% 구간 폭이 {WIDTH_C * 100:.0f}%를 넘음 ({', '.join(wide)}) — 숫자보다 폭을 본다"))
        n = sv["n_persons"]
        if n < N_FEW:
            R.append(("C", f"관측 적음 — 예측 보류 수준: 쓴 사람(유효) {n}명 < {N_FEW}. 한두 사람의 화면이 결과를 정한다"
                           " (S41 tr 48시간에서 관측 하나로 낸 3% 컷은 5명 중 1명꼴로 5% 넘게 틀렸다)"))
        elif n < 10:
            R.append(("B", f"쓴 사람(유효) {n}명 < 10"))
        if sv["n_persons_label"] < N_FEW and sv["mixed_rows"]:
            R.append(("B", f"서버 라벨 사람 {sv['n_persons_label']}명 < {N_FEW} — 서버 미상 줄(사이트 비율·확률)에 기댄다. 아래 「라벨 줄만」과 견준다"))
    if sv.get("fill_draws_share", 0) > 0.1:
        R.append(("C", f"뽑기의 {sv['fill_draws_share'] * 100:.0f}%에서 쓸 점이 없어(옮긴 %가 모양 격자 밖 등) "
                       "다른 서버·S41 곡선(+가정 폭)으로 채웠다"))
    if unk:
        for w in unk.get("warnings", []):
            if s in unk.get("servers", []):
                R.append(("B", w))
    lvl = max([r[0] for r in R] or ["A"])
    return {"level": lvl, "label": WARN_LABEL[lvl], "reasons": [f"[{a}] {b}" for a, b in R]}


def assumptions(prior, pr):
    sd = prior["season_diff"]
    return [
        f"[가정] prior는 {prior['season']} 한 시즌에서 잰 값이다 (% 변환 꼴·모수, 최종 곡선 모양, 서버 비율).",
        f"[가정] 시즌 간 P 수준 차이 τ_P = {pr.tau_p:.3f} (ln 배수 1σ). 근거: {sd['tau_p']['basis']}",
        f"[가정] 시즌 간 곡선 모양 차이 τ_shape = {pr.tau_shape:.2f} × D(u). 근거: {sd['tau_shape']['basis']}",
        f"[가정] 관측 없는 서버의 서버 비율 차이 τ_ratio = {pr.tau_ratio:.3f}. 근거: {sd['tau_ratio']['basis']}",
        f"[가정] 관측이 하나도 없을 때 규모 차이 τ_scale0 = {pr.tau_scale0:.2f}. 근거: {sd['tau_scale0']['basis']}",
        "[가정] P(h)의 모양(시간 함수)은 시즌·서버에 공통이고 수준만 다를 수 있다. 일섭은 수준만 따로, 글섭은 한섭을 빌린다 (prior 참고).",
        "[가정] 시각 등급 C는 게시 시각(상한)을 관측 시각으로 쓴다 (보정 없음, --c-delay로 민감도).",
        f"[가정] 모양 보간 잡음 {KAPPA}·|ln q − ln y|, 점수 바닥 {math.sqrt(V0):.3f} (ln) — 근거 없이 정한 작은 값.",
    ]


def fmt_s(v):
    return "—" if v is None else f"{v:.1f}"


def fmt_i(d):
    return f"{fmt_s(d['median'])} [80% {fmt_s(d['i80'][0])}~{fmt_s(d['i80'][1])}]"


def contam_brief(sv, q="3.0"):
    """섞임 가정 한 토막: f별 q 중앙 추정 변화."""
    c = sv.get("contam")
    if not c:
        return ""
    parts, ext = [], None
    for f, e in c["f"].items():
        pq = e["points"].get(q)
        parts.append(f"f={float(f) * 100:g}% " + (f"{pq['shift'] * 100:+.1f}%" if pq else f"못 잼({e['k']}명)"))
        if pq:
            ext = (f, pq["i80_ext"][1])
    return (f"섞임 가정({float(q):g}% 중앙): " + " · ".join(parts)
            + (f" → f={float(ext[0]) * 100:g}%로 위로 넓힌 80% 위 끝 {fmt_s(ext[1])}" if ext else ""))


def summary_lines(out):
    """출력 맨 위 요약 (재검토 r2 8): T·c_start / 한섭 1·3%·등급 / 일섭·글섭 등급 / 검증 안 됨 / 등급 기준."""
    L = []
    L.append(f"[요약] {out['season']} · T = {out['T']:.1f}시간 = KST {out['T_kst']} · c_start {out['c_start']}"
             + (" (대체값! season.json에 c_start 없음)" if out.get("c_start_assumed") else "") + " · 게시 시각 ≤ T 자료만")
    kr = out["servers"]["KR"]
    w = kr.get("warning", {})
    L.append(f"[요약] 한섭 1% {fmt_i(kr['points']['1.0'])} · 3% {fmt_i(kr['points']['3.0'])} (억) · 등급 {w.get('label', '?')}"
             + (f" · {contam_brief(kr)}" if kr.get("contam") else ""))
    L.append("[요약] " + " · ".join(f"{SNAME[s]} 등급 {out['servers'][s].get('warning', {}).get('label', '?')}" for s in ("JP", "GL"))
             + " (일섭·글섭은 라벨 줄만 쓴다)")
    L.append("[요약] ※ 검증 안 된 예측이다 — S41 한 시즌으로 만든 도구, S41 되돌아보기는 성능 근거가 아니다. 구간에 섞임(미상 줄의 일섭·글섭)은 들어 있지 않다.")
    L.append("[요약] " + GRADE_RULE)
    return L


def render(out):
    L = summary_lines(out)
    for w in out.get("warnings") or []:
        L.append(f"!! {w}")
    L.append("")
    st = out["settings"]
    L.append(f"솔로레이드 컷 예측 — {out['season']} · T = {out['T']:.1f}시간 (KST {out['T_kst']}) · c_start {out['c_start']} · 게시 시각 ≤ T 자료만")
    u = out.get("unknown")
    um = "끔(라벨 줄만)" if not u else f"{u['used']}" + (f" (요청 {u['requested']})" if u["used"] != u["requested"] else "")
    L.append(f"prior {st['prior_season']} (P 꼴 {st['prior_form']}) · 시각 등급 {st['grades']} · C 지연 {st['c_delay']}시간 · "
             f"서버 미상 줄 {um}" + (f" · 서버 확률 파일 {st['server_mix']}" if st['server_mix'] else "") +
             f" · τ_shape {st['tau_shape']:.2f} · 뽑기 {out['draws']}"
             + (f" · 강건 {st['robust']}" if st.get("robust", "mean") != "mean" else "")
             + (f" · 꼭대기 제외 < {st['top_cut']}%" if st.get("top_cut") else ""))
    rm = out.get("run")
    if rm:
        ss = rm["state"]
        L.append(f"실행: 도구 HEAD {(rm['tool_head'] or '?')[:12]}" + (" (도구 파일에 커밋 안 된 수정 있음)" if rm.get("tool_dirty") else "")
                 + f" · prior sha1 {(rm['prior_sha1'] or '?')[:12]} · state obs {ss['obs_lines']}줄 (sha1 {(ss['obs_sha1'] or '?')[:12]})"
                 + f" · parsed.jsonl {'있음' if ss['parsed_jsonl'] else '없음'}")
    c = out["counts"]
    L.append(f"관측: T까지 게시된 기본 집합 줄 {c.get('rows_le_T', '?')} → 시점 {c.get('moments', '?')} "
             f"(서버 미상·안 씀 {c.get('unlabeled', 0)}, T 뒤 게시 {c.get('after_T', 0)}"
             + (f", 챌린지 시작 전 C 줄 버림 {c['drop_c_before_start']}" if c.get("drop_c_before_start") else "") + ")")
    if u:
        L.append(f"서버 미상 줄: {u.get('rows_assigned', 0)}줄(같은 사람·같은 % 묶은 시점 {c.get('unknown_moments', '?')})을 "
                 f"{'/'.join(SNAME.get(x, x) for x in u['servers'])} 후보로 씀 · "
                 f"비율 출처 {u.get('ratio_source') or '서버 확률 파일'} · T까지 서버 라벨 사람 {u['label_persons']}명")
        for w in u.get("warnings", []):
            L.append(f"  경고: {w}")
    if st["server_mix"] or (u and u["used"] == "mix"):
        L.append("주의: 서버 확률은 점수·%(곡면)로 낸 것이다. 그 줄을 같은 서버 곡선의 근거로 쓰면 일부 순환이다 — 라벨 줄만 쓴 결과와 함께 본다.")
    if out["p_update"]:
        p = out["p_update"]
        L.append(f"시즌 P 수준 갱신: 같은 점수 쌍 {p['n_pairs']}개 → 배수 {p['m_mean']:.3f} (ln sd {p['sd_lnm']:.3f}, 사전 τ_P {p['prior_tau_p']:.3f})")
    L.append("")
    L.append("※ 검증 안 된 예측이다. S41 한 시즌으로 만든 도구이고, S41 되돌아보기는 곡선 모양이 S41 정답 그 자체라 성능 근거가 아니다.")
    L.append("  1·3% 구간 폭의 핵심(시즌 간 % 변환 수준 τ_P, 1~3% 곡선 모양 차)은 가정이다 (아래 가정 목록).")
    L.append("읽는 법: ■ 주 결과는 1·3% 컷이다. 0.5·2%는 보조. 5·10·15·30%는 「참고 — 시즌 간 모양 가정에 기댐」(S41 곡선 모양을 그대로 빌림, 맞는지 잴 수 없음).")
    L.append("        등급 A 보통 · B 주의 · C 강한 주의 · D 숫자만 참고 (기준은 맨 위 요약). 구간은 S41 한 시즌으로 정한 가정을 포함한다 (아래 가정 목록).")
    L.append("        「라벨 줄만」은 60시간 전에는 라벨 사람이 1~2명이라 정보가 거의 없다 — 불확실성 점검이 아니라 참고다.")
    for s, sv in out["servers"].items():
        L.append("")
        w = sv.get("warning", {})
        L.append(f"■ {SNAME[s]} ({s}) — 등급 {w.get('label', '?')} · 근거 {sv['basis']} · 시점 {sv['n_moments']} · 사람 {sv['n_persons']}"
                 + (f" (라벨 {sv['n_persons_label']} + 서버 미상 기대 {sv['n_persons_mix_expected']})" if sv['mixed_rows'] else "")
                 + (f" · 시각 등급 {sv['grades']}" if sv['grades'] else "")
                 + (f" · 서버 미상 시점 {sv['mixed_rows']}" if sv['mixed_rows'] else ""))
        for r in w.get("reasons", []):
            L.append(f"  {r}")
        if sv["donor"]:
            L.append(f"  ▶ 근거 없음: 이 서버의 관측이 T까지 0개다. 아래 숫자는 "
                     + ("S41 곡선 그대로(+가정 폭)" if sv["donor"] == "S41" else f"{SNAME.get(sv['donor'], sv['donor'])} 예측 × S41 서버 비율")
                     + "일 뿐 이 시즌 자료로 낸 예측이 아니다.")
            d1, d3 = sv["points"]["1.0"], sv["points"]["3.0"]
            L.append(f"  (근거 없음) 1% {fmt_s(d1['median'])} [80% {fmt_s(d1['i80'][0])}~{fmt_s(d1['i80'][1])}] · "
                     f"3% {fmt_s(d3['median'])} [80% {fmt_s(d3['i80'][0])}~{fmt_s(d3['i80'][1])}] — 다른 지점은 JSON에만")
            continue
        if sv.get("label_only"):
            lo = sv["label_only"]
            L.append(f"  라벨 줄만 쓴 결과 (사람 {lo['n_persons']}" + (", 관측 0 — 근거 없음" if lo["donor"] else "") + "): " + " · ".join(
                f"{float(q):g}% {fmt_s(v['median'])} [80% {fmt_s(v['i80'][0])}~{fmt_s(v['i80'][1])}]"
                for q, v in lo["points"].items() if float(q) in OUT_MAIN))
        if sv.get("contam"):
            cs = sv["contam"]
            L.append(f"  섞임 가정 (site 방식은 「미상 줄 ≈ 한섭」 가정. 미상 줄에 일섭·글섭이 섞이면 컷을 늘 아래로 끈다 — 주 구간에 안 들어 있음): "
                     f"미상 사람 {cs['n_unknown_persons']}명 중 함축 3% 점수가 가장 낮은 f 몫을 빼면")
            for f, e in cs["f"].items():
                if not e["points"]:
                    L.append(f"    f={float(f) * 100:g}%: {e.get('note', '')}")
                    continue
                L.append(f"    f={float(f) * 100:g}% ({e['k']}명 뺌): " + " · ".join(
                    f"{float(q):g}% 중앙 추정 {pq['shift'] * 100:+.1f}% → 80% 위로 넓히면 {fmt_s(pq['i80_ext'][0])}~{fmt_s(pq['i80_ext'][1])}"
                    for q, pq in e["points"].items() if float(q) in OUT_MAIN))
        if sv["h_range"]:
            L.append(f"  관측 h {sv['h_range'][0]:.1f}~{sv['h_range'][1]:.1f}시간 · 지금 % {sv['x_range'][0]}~{sv['x_range'][1]} · "
                     f"옮긴 최종 % {sv['y_central_range'][0]:.2f}~{sv['y_central_range'][1]:.2f}"
                     + (f" · 크게 어긋나 뺀 점 {len(sv['screened_out'])}" if sv['screened_out'] else ""))
        hdr = "  최종 %  | 중앙(억) | 50% 구간      | 80% 구간      | 95% 구간      | 80%: (i)관측 (ii)prior (iii)시즌차 | 기준선0 기준선1 | 비고"
        for title, qs in (("주 결과", OUT_MAIN), ("보조", OUT_AUX), ("참고 — 시즌 간 모양 가정에 기댐", OUT_REF)):
            L.append(f"  [{title}]")
            L.append(hdr)
            for q in qs:
                d = sv["points"][str(q)]
                comp = d.get("comp80", {})
                cs = " ".join(f"{fmt_s(comp[k][0])}~{fmt_s(comp[k][1])}" for k in ("i", "ii", "iii")) if comp else ""
                L.append(f"   {q:>5g}% | {fmt_s(d['median']):>7} | {fmt_s(d['i50'][0])}~{fmt_s(d['i50'][1]):<6} | "
                         f"{fmt_s(d['i80'][0])}~{fmt_s(d['i80'][1]):<6} | {fmt_s(d['i95'][0])}~{fmt_s(d['i95'][1]):<6} | {cs} | "
                         f"{fmt_s(d['b0'])} {fmt_s(d['b1'])} | {'; '.join(d['notes'])}")
        L.append("  점수 → 최종 % (중앙 [80%], 10% 넘는 쪽은 참고):  " + " · ".join(
            f"{r['score']:g}억 {fp(r['median'])} [{fp(r['i80'][0])}~{fp(r['i80'][1])}]" for r in sv["table"]))
    L.append("")
    L.append("기준선0 = 지금 % 그대로 최종으로, 기준선1 = 일차별 S41 중앙 배수만 곱함 (둘 다 같은 모양·가중 평균, 점 추정만).")
    L.extend(out["assumptions"])
    return "\n".join(L)


def fp(v):
    if v is None:
        return "—"
    if v == -math.inf:
        return "<0.3%"
    if v == math.inf:
        return ">40%"
    return f"{v:.2f}%" if v < 10 else f"{v:.1f}%"


def main():
    install_guard()
    ap = argparse.ArgumentParser(description="솔로레이드 컷 예측 (게시 시각 ≤ T 자료만)")
    ap.add_argument("state")
    ap.add_argument("--prior", required=True)
    ap.add_argument("--at", required=True)
    ap.add_argument("--server-mix", help="(예전 방식) 서버 확률 파일을 기준 검사 없이 모든 서버에 쓴다. --unknown보다 앞선다")
    ap.add_argument("--unknown", choices=UNKNOWN_MODES, default=UNKNOWN_DEFAULT,
                    help=f"서버 미상 줄 처리 (기본 {UNKNOWN_DEFAULT}): none 라벨 줄만 · site 사이트별 서버 구성 비율(점수·% 안 씀) · "
                         "mix server_mix.py 확률(STATE/post_server_mix.jsonl, gate 통과일 때만)")
    ap.add_argument("--unknown-servers", default="KR", help="미상 줄을 넣을 서버 (쉼표, 기본 KR — 일섭·글섭은 라벨 줄만)")
    ap.add_argument("--borrow-state", default=None,
                    help="T까지 라벨 사람이 적을 때 사이트 비율을 빌릴 끝난 시즌 state. 기본은 predict_run.sh와 같은 "
                         "이 레포의 data/solo-raid/S41/state (없으면 경고). ''이면 빌리지 않는다")
    ap.add_argument("--borrow-kind", choices=["label", "em"], default="label",
                    help="빌리는 비율: label 서버 라벨 사람 수(기본) · em server_mix 요약의 em 사전(점수·%로 잰 값)")
    ap.add_argument("--mix-file", help="--unknown mix의 확률 파일 (기본 STATE/post_server_mix.jsonl, 요약은 같은 이름_summary.json)")
    ap.add_argument("--no-gate", action="store_true", help="--unknown mix에서 gate 미통과여도 쓴다 (민감도용)")
    ap.add_argument("--fallback", choices=["site", "none"], default="site", help="--unknown mix의 gate 미통과 때 쓸 방식")
    ap.add_argument("--robust", choices=["mean", "wmedian"], default="mean", help="점들을 합치는 방식 (비교용 wmedian)")
    ap.add_argument("--top-cut", type=float, help="옮긴 최종 %가 이보다 작은 서버 미상 줄은 쓰지 않음 (꼭대기 제외, 비교용)")
    ap.add_argument("--eps-wrong", type=float, default=EPS_WRONG, help=f"「점 하나가 틀림」 혼합 확률 (사람 {N_SCREEN}명 미만 서버만, 기본 {EPS_WRONG}, 0이면 끔)")
    ap.add_argument("--grades", default="all", choices=["AB", "all"])
    ap.add_argument("--out")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tau-p", type=float)
    ap.add_argument("--tau-shape", type=float)
    ap.add_argument("--tau-ratio", type=float)
    ap.add_argument("--tau-scale0", type=float)
    ap.add_argument("--no-season-diff", action="store_true")
    ap.add_argument("--p-update", action="store_true", help="같은 점수 쌍으로 시즌 P 수준 배수를 갱신 (기본 끔 — S41 점검에서 해로웠다)")
    ap.add_argument("--shape-update", action="store_true",
                    help="시즌 간 모양 기울임 Z를 그 시즌 자료로 추정 (3.5b, 기본 끔 — 2026-10-17 분석 ④에서 검증)")
    ap.add_argument("--c-delay", type=float, default=0.0)
    ap.add_argument("--json", action="store_true", help="표준 출력에 JSON도")
    ap.add_argument("--allow-cstart-fallback", action="store_true",
                    help="season.json에 c_start가 없을 때 start 날짜 + 2일 05:00으로 대신한다 (기본은 멈춤, 권하지 않음)")
    ap.add_argument("--contam", default=",".join(f"{f:g}" for f in CONTAM_DEFAULT),
                    help="섞임 민감도의 f 값들 (미상 사람 중 뺄 몫, 쉼표, 기본 0.1,0.2). ''이면 끔")
    a = ap.parse_args()
    with safe_open(a.prior, encoding="utf-8") as f:
        prior = json.load(f)
    notes = []
    season, c0, H = season_info(a.state, allow_fallback=a.allow_cstart_fallback)
    T = parse_at(a.at, c0, H, notes)
    borrow = a.borrow_state
    if borrow is None:   # predict_run.sh와 같은 기본값 (재검토 r2 6: 직접 실행이 비율을 다르게 냈다)
        borrow = DEFAULT_BORROW if os.path.isdir(DEFAULT_BORROW) else ""
        if not borrow and a.unknown != "none":
            notes.append(f"경고: --borrow-state 기본값({DEFAULT_BORROW})이 없어 비율을 빌리지 않는다 — predict_run.sh 결과와 다를 수 있다")
    opts = {"grades": a.grades, "c_delay": a.c_delay, "server_mix": a.server_mix, "p_update": a.p_update, "shape_update": a.shape_update,
            "no_season_diff": a.no_season_diff, "robust": a.robust, "top_cut": a.top_cut, "eps_wrong": a.eps_wrong,
            "allow_cstart_fallback": a.allow_cstart_fallback, "notes": notes,
            "contam": tuple(float(x) for x in a.contam.split(",") if x.strip()),
            "unknown": {"mode": a.unknown, "servers": [x.strip() for x in a.unknown_servers.split(",") if x.strip()],
                        "borrow_state": borrow or None, "borrow_kind": a.borrow_kind, "mix_file": a.mix_file,
                        "no_gate": a.no_gate, "fallback": a.fallback}}
    for k in ("tau_p", "tau_shape", "tau_ratio", "tau_scale0"):
        if getattr(a, k) is not None:
            opts[k] = getattr(a, k)
    out = predict(a.state, prior, T, opts, a.draws, a.seed)
    out["run"] = run_meta(a.state, a.prior, out)
    txt = render(out)
    print(txt)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = f"T{out['T']:g}"
        with open(os.path.join(a.out, f"predict_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(jsonable(out), f, ensure_ascii=False, indent=1)
        with open(os.path.join(a.out, f"predict_{tag}.txt"), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print(f"\n→ {a.out}/predict_{tag}.json · .txt")
    if a.json:
        print(json.dumps(jsonable(out), ensure_ascii=False))


DEFAULT_BORROW = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "solo-raid", "S41", "state"))


def sha1_of(path):
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except OSError:
        return None


def nlines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for l in f if l.strip())
    except OSError:
        return None


def run_meta(state, prior_path, out):
    """결과 JSON의 run: 도구 HEAD·prior sha1·입력 state 요약 (재검토 r2 4 — 커밋할 예측을 나중에 다시 만들 수 있게)."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*a):
        try:
            r = subprocess.run(["git", "-C", here, *a], capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    dirty = git("status", "--porcelain", "--", ".", "../../experiments/2026-10-15-solo-raid-low-cost/tools/server_classify.py")
    c = out.get("counts", {})
    u = out.get("unknown") or {}
    st = {"path": os.path.abspath(state),
          "season_json_sha1": sha1_of(os.path.join(state, "season.json")),
          "obs_lines": nlines(os.path.join(state, "obs.jsonl")), "obs_sha1": sha1_of(os.path.join(state, "obs.jsonl")),
          "claims_lines": nlines(os.path.join(state, "claims.jsonl")),
          "posts_min_lines": nlines(os.path.join(state, "posts_min.jsonl")),
          "core_sha1": sha1_of(os.path.join(state, "core.jsonl")),
          "parsed_jsonl": os.path.exists(os.path.join(state, "parsed.jsonl")),
          "rows_le_T": c.get("rows_le_T"), "moments": c.get("moments"),
          "unknown_rows": u.get("rows_assigned", c.get("unlabeled")), "unknown_moments": c.get("unknown_moments"),
          "label_persons": u.get("label_persons"), "label_persons_by": u.get("label_persons_by")}
    return {"tool_head": git("rev-parse", "HEAD"), "tool_dirty": (dirty != "") if dirty is not None else None,
            "prior_path": os.path.abspath(prior_path), "prior_sha1": sha1_of(prior_path), "state": st}


def jsonable(o):
    if isinstance(o, float):
        if o == math.inf:
            return "inf"
        if o == -math.inf:
            return "-inf"
        return round(o, 5)
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


if __name__ == "__main__":
    main()
