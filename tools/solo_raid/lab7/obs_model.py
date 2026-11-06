"""관측 층 (WP3b 1단계, 2026-10-29) — 베이지안 계층 모형 M1·M2·M3 공통. 곡면 꼴은 모르고, 곡면 값(mu·P)만 받아 줄 가능도를 낸다.

  python3 -I tools/solo_raid/lab7/obs_model.py        # 자체 점검 없음 — 시험은 check_obs.py

설계서 1절 그대로. docs/solo-raid-method.md 「WP3 명세」 2절. 쓰는 곳: m1.py·(M2·M3 세션)·check_obs.py.

쓰는 순서 (후보 모형 쪽)
  import obs_model as OM                  # sys.path.insert(0, HERE) 뒤. jax를 다른 데서 먼저 import하지 않는다
  OM.setup_host(4)                        # 실행 진입점에서, 모든 jax 계산 전에 (모듈 import 때는 안 부른다)
  head, rows = obs_table.read_cell(path)  # 관측 표
  err = OM.load_err()
  sel = OM.select_rows(rows, head["H"], stage1=True)
  obs = OM.prepare(rows, head["H"], head["season"], err, K=16, delay_key=None, stage1=True)  # numpy dict
  jobs = OM.to_jax(obs)                   # jnp dict (meta·counts 는 빠진다)
  def model(obs):                         # numpyro 모형
      ... theta 표집 ...
      mu, P = surface(theta, obs["y"], obs["u"])      # 곡면 함수 약속 ↓
      OM.observe(obs, mu, P, sigma0, rho)
  samples, info = OM.run_nuts(model, {"obs": jobs}, seed=0)      # samples[name] = [chains, draws, ...]
  diag = OM.diagnose(samples, names=[...])                       # R-hat·ESS·conv

곡면 함수 약속 (후보 쪽이 지킨다)
  입력 obs["y"][N] (ln 점수, 억), obs["u"][N,K] (시각 / 시즌 길이, 0.02~1).
  출력 mu[3,N,K], P[3,N,K] — 서버 성분 KR·JP·GL(0·1·2)마다, 그 줄·노드의 곡면 log10 % (= z_fin − P)와 그 자리 침수 P >= 0 (log10 단위).
  observe(obs, mu, P, sigma0, rho)에는 위 둘과 스칼라 sigma0·rho(잔차 sd = sqrt(sigma0^2 + (rho P)^2))를 넘긴다.
  eps(오염)·phi(사이트별 서버 몫)는 observe가 표집한다 — 후보 쪽은 건드리지 않는다.

줄 늘어놓기: prepare는 노드 하나뿐인 줄(A·post·한 노드 타이머)을 앞 obs["n1"]줄에, 노드 여럿 줄을 그 뒤에 놓고 각 칸을 채움 줄로 채운다 (FIX_N1·FIX_N2 칸).
  실제 줄 i(meta[i])는 행 obs["pos"][i]에 있다. row_loglik은 앞 n1줄은 노드 0번만 계산한다 (같은 값, 약 35% 빠름).
  obs는 모형 안에서 닫아(클로저·kwargs) 쓰고 jit 인자로 넘기지 않는다 (n1·fc_scale 같은 파이썬 값이 추적되어 깨진다).
  row_posteriors/flags가 돌려주는 줄 배열은 meta 순서 (pos로 이미 골랐다).
CPU 시간 감: S30 전체 표 441줄(N=512)에서 4사슬 x (500+500)는 M1 kuma 꼴로 약 4.5분 (check_obs.py 3번).
수치 약속: 모든 jax 계산은 float32. 구간 중도절단 정규는 logdiff_ndtr(안정식), 안 쓰는 노드의 log 무게는 -inf가 아니라 -1e4.
공개 규칙: 글 번호·원문·닉네임·원글 주소는 코드·출력에 쓰지 않는다 (줄 식별은 key·pid·mid만).
"""
import collections
import json
import os
import sys
import time

import numpy as np
from scipy import special as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import err_1b  # noqa: E402  (group_of만 쓴다)

ERR_PATH = os.path.join(C.ROOT, "data", "solo-raid", "analysis", "r7", "obs_error.json")
FIX_N1 = (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024)      # 노드 1개 줄 칸 (A·post·한 노드 타이머) + 채움
FIX_N2 = (64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048)  # 노드 여럿 줄 칸
NEG = -1e4
SITES = ("dc", "naver", "blabla")
SERVERS = ("KR", "JP", "GL")
DELAY_GROUPS = ("timer", "now", "clock")
ABC = ("A", "B", "C")
Q_TARGETS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)


def load_err(path=ERR_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- 실행 환경 ----------

def setup_host(n=4):
    """CPU 장치 수 지정 + float32 고정. 다른 jax 계산(배열 만들기 포함)보다 먼저 부른다 (import jax만은 괜찮다)."""
    import numpyro
    numpyro.set_host_device_count(n)
    import jax
    jax.config.update("jax_enable_x64", False)


# ---------- 시간 축 (WP11a ⓐ, 2026-11-03) ----------
# None = 시계 시간 u = h/H (지금 코드). set_clock("act", 프로필 경로) = 활동 시간 u* = A(h)/A(H) (act_clock.py — 그 시즌의 g_s).
# 켜면 prepare·simulate_pct의 u만 바뀌고 obs에 "clock" 키가 붙는다. 끄면 배열·키가 지금과 같다.
CLOCK = None
CLOCK_PROF = None


def set_clock(clock, profile=None):
    global CLOCK, CLOCK_PROF
    if clock in (None, "h"):
        CLOCK, CLOCK_PROF = None, None
        return
    if clock != "act":
        raise ValueError(f"모르는 clock {clock}")
    import act_clock
    CLOCK, CLOCK_PROF = "act", act_clock.load(profile)


def u_of(season, t, H):
    """노드 시각 t (시간) -> 곡면 시간 좌표 (0.02~1)."""
    if CLOCK is None:
        return np.clip(t / H, 0.02, 1.0)
    import act_clock
    return act_clock.ustar(CLOCK_PROF, season, t, H)


# ---------- 변형 (WP4 관측 층 빼 보기, 2026-10-31) ----------
# 이름 -> 설정. 없는 키는 본안 값. base·None은 get_var가 None으로 돌려 기존 코드 경로 그대로 간다.
VARIANTS = {
    "base": {},
    "A": {"kinds": {"A"}},                      # 정확한 점만
    "AB": {"kinds": {"A", "B"}},                # A + B
    "Cpost": {"c_post": True},                  # C를 게시 시각(he)에 둠, 시각 오차 0
    "text": {"text": True},                     # 본안 + 글로만 적은 %
    "nocontam": {"contam": False},              # 오염 성분 뺌
    "mono": {"server": "mono"},                 # 서버 미상 줄은 mono 거르기 + 한섭 확정
}
TEXT_EPS = 0.109                                # 글 줄 eps 고정 (1c 엄격 값, load_err 값으로 prepare가 덮는다)


def get_var(name):
    """변형 이름·설정 -> 설정 dict(name 포함). None·"base"는 None."""
    if name is None or name == "base":
        return None
    if isinstance(name, dict):
        return name
    if name not in VARIANTS:
        raise ValueError(f"모르는 변형 {name}; {sorted(VARIANTS)}")
    v = dict(VARIANTS[name])
    v["name"] = name
    return v


def _dec(v):
    """소수 자릿수 (0~3), 그 이상 9 (err_1c.dec과 같은 논리)."""
    for d in range(4):
        if abs(round(v, d) - v) < 1e-9:
            return d
    return 9


# ---------- 1.1 줄 고르기 ----------

def select_rows(rows, H, stage1=False, var=None, stats=None):
    """쓸 줄 목록 (복사본에 tg·post 추가). 기본 = kind A·B·C, lo_reason None, rep. stage1이면 종료 뒤 화면 줄도 더한다.
    var(변형)가 있을 때만 kind 거르기·mono·글 줄을 더 건드린다. stats = 세어 담을 dict (선택)."""
    var = get_var(var)
    out = []
    seen = set()
    for r in rows:
        if r["kind"] not in ABC:
            continue
        post = False
        if r["lo_reason"] is None and r["rep"]:
            pass
        elif stage1 and r["lo_reason"] == "skip_day_post" and r["use"] and r["pct"] is not None \
                and r["score"] is not None and "best_record" not in (r["flag"] or []) \
                and (r["server"] is None or r["server"] in SERVERS):
            k = (r["pid"], round(r["pct"], 2))
            if k in seen:
                continue
            seen.add(k)
            post = True
        else:
            continue
        if r["pct"] is None or r["score"] is None or r["score"] <= 0:
            continue
        if var:
            if not post and var.get("kinds") and r["kind"] not in var["kinds"]:
                continue
        x = dict(r)
        x["post"] = post
        if var and var.get("server") == "mono" and r["server"] is None:
            if post or r.get("mono_drop"):          # 종료 뒤 화면은 mono 정의 안 됨 / 거른 줄
                if stats is not None:
                    stats["drop_mono"] = stats.get("drop_mono", 0) + 1
                continue
            x["server"], x["server_basis"], x["mono_kr"] = "KR", "mono", True
        if post:
            x["tg"] = "post"
        elif r["kind"] == "A":
            x["tg"] = "A"
        elif r["kind"] == "B":
            g = err_1b.group_of(r["t_basis"])
            x["tg"] = g if g in DELAY_GROUPS else "delay"
        else:
            x["tg"] = "delay"
        out.append(x)
    if var and var.get("text"):
        out += _text_rows(rows, stage1, var, stats)
    return out


def _text_rows(rows, stage1, var, stats=None):
    """글로만 적은 %의 줄 (복사본에 tg·post·is_text). 사진과 같은 사람·같은 %(0.01 단위)인 글은 뺀다."""
    photo = {(r["pid"], round(r["pct"], 2)) for r in rows if r["kind"] in ABC and r["pct"] is not None}
    seen = set()
    out = []
    for r in rows:
        if r["kind"] != "text" or r["score"] is None or r["pct"] is None or r["score"] <= 0 or r["link"]:
            continue
        if r["server"] is not None and r["server"] not in SERVERS:
            continue
        k = (r["pid"], round(r["pct"], 2))
        if k in photo or k in seen or r["hp"] is None or r["hp"] < 0:
            continue
        post = r["day_lo"] == "post"
        if post and not stage1:
            continue
        seen.add(k)
        x = dict(r)
        x["post"], x["is_text"] = post, True
        x["tg"] = "post" if post else "delay"
        out.append(x)
    if stats is not None:
        stats["sel_text_rows"] = len(out)
    return out


# ---------- 1.2 시각 노드 ----------

def _mixcdf(x, comps):
    """comps = [(w, kind, a, b)] — lognorm: ln x ~ N(a,b) (x<=0 에서 0), norm: N(a,b)."""
    x = np.asarray(x, float)
    tot = np.zeros_like(x)
    for w, kind, a, b in comps:
        if kind == "lognorm":
            xs = np.maximum(x, 1e-300)
            c = np.where(x > 0, sps.ndtr((np.log(xs) - a) / b), 0.0)
        else:
            c = sps.ndtr((x - a) / b)
        tot = tot + w * c
    return tot


def _quad(comps, xa, xb, K):
    """[xa, xb]로 자른 분포의 (j-1/2)/K 분위 노드. (노드, 사유 또는 None)."""
    fa, fb = float(_mixcdf(np.array([xa]), comps)[0]), float(_mixcdf(np.array([xb]), comps)[0])
    j = (np.arange(1, K + 1) - 0.5) / K
    if fb - fa < 1e-9:
        return xa + j * (xb - xa), "flat_support"
    tau = fa + j * (fb - fa)
    lo = np.full(K, xa, float)
    hi = np.full(K, xb, float)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        up = _mixcdf(mid, comps) < tau
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi), None


def _pad(t, w, K):
    k = len(t)
    if k >= K:
        return t[:K], w[:K] / w[:K].sum()
    return np.concatenate([t, np.full(K - k, t[0])]), np.concatenate([w, np.zeros(K - k)])


def support(r, H):
    """(Lo, Up) 시간 지지 또는 None 항목. A·post는 쓰지 않는다."""
    tg = r["tg"]
    hp = r.get("hp")
    t7lo, t7hi = r.get("h_lo_top7"), r.get("h_hi_top7")
    ups = [H] + [v for v in (hp, t7hi) if v is not None]
    los = [0.0] + [v for v in (t7lo,) if v is not None]
    if tg == "delay" and r.get("h_lo") is not None:
        los.append(r["h_lo"])
    Up = min(ups) + (2.0 / 60 if tg in DELAY_GROUPS else 0.0)
    return max(los), Up


def nodes_detail(r, H, err, delay_key, K, var=None):
    """(t[K], w[K], 사유 또는 None) — 사유: bad_support·timer_fallback·flat_support. 지지가 비면 (None, None, "bad_support")."""
    tg = r["tg"]
    if var is not None and r["kind"] == "C" and tg == "delay":
        var = get_var(var)
        if var and var.get("c_post"):               # Cpost: C는 게시 시각(he)에 노드 하나, 시각 오차 0
            t0 = r.get("he") if r.get("he") is not None else r.get("hp")
            if t0 is None:
                return None, None, "bad_support"
            return _pad(np.array([min(max(float(t0), 0.0), float(H))]), np.array([1.0]), K) + (None,)
    if tg == "A":
        return _pad(np.array([r["h"]], float), np.array([1.0]), K) + (None,)
    if tg == "post":
        return _pad(np.array([float(H)]), np.array([1.0]), K) + (None,)
    if tg == "delay" and r.get("hp") is None:
        return None, None, "bad_support"
    if tg != "delay" and r.get("h") is None:
        return None, None, "bad_support"
    if tg in DELAY_GROUPS and r.get("hp") is None:
        return None, None, "bad_support"
    Lo, Up = support(r, H)
    if Lo > Up:
        return None, None, "bad_support"
    b1 = err["1b_stated_time"]
    h = r["h"]
    if tg == "timer":
        rare = b1["timer"]["rare"]
        pw = rare["pi"]["p"]
        g = rare["p_k_ge_2_given_wrong"]["p"]
        ks = np.arange(0, K)
        t = h - 24.0 * ks
        w = np.where(ks == 0, 1 - pw, pw * (1 - g) * g ** (np.maximum(ks, 1) - 1))
        ok = (t >= Lo - 1e-12) & (t <= Up + 1e-12)
        reason = None
        if not ok.any():
            t, w, reason = np.array([h]), np.array([1.0]), "timer_fallback"
        else:
            t, w = t[ok], w[ok] / w[ok].sum()
        return _pad(t, w, K) + (reason,)
    if tg == "now":
        p = b1["now"]["fit"]["params"]
        comps = [(1 - p["pi_tail"], "lognorm", p["mu1"], p["sigma1"]), (p["pi_tail"], "lognorm", p["mu2"], p["sigma2"])]
        xa, xb = max(0.0, (h - Up) * 60), (h - Lo) * 60
        if xb < xa:
            return None, None, "bad_support"
        x, reason = _quad(comps, xa, xb, K)
        return h - x / 60, np.full(K, 1.0 / K), reason
    if tg == "clock":
        p = b1["clock"]["fit"]["params"]
        comps = [(1 - p["pi_wide"], "norm", p["mu"], p["sigma_core"]), (p["pi_wide"], "norm", p["mu"], p["sigma_wide"])]
        xa, xb = (h - Up) * 60, (h - Lo) * 60
        x, reason = _quad(comps, xa, xb, K)
        return h - x / 60, np.full(K, 1.0 / K), reason
    # delay
    a1 = err["1a_post_delay"]
    p = a1["expanding"].get(delay_key) if delay_key else None
    p = p or a1["recommend"]["params"]
    comps = [(p["w_fast"], "lognorm", p["mu1"], p["sigma1"]), (1 - p["w_fast"], "lognorm", p["mu2"], p["sigma2"])]
    hp = r["hp"]
    xa, xb = max(0.0, hp - Up), hp - Lo
    x, reason = _quad(comps, xa, xb, K)
    return hp - x, np.full(K, 1.0 / K), reason


def nodes_for(r, H, err, delay_key, K, var=None):
    """줄 r (tg 붙은 것)의 촬영 시각 노드 (t[K'], w[K']), K' <= K. 지지가 비면 (None, None). 안 쓰는 칸은 무게 0 (K로 채움)."""
    t, w, _ = nodes_detail(r, H, err, delay_key, K, var)
    return t, w


# ---------- 1.3 묶음 ----------

ELAB_KEY = {"top7": "top7", "text": "text", "text+top7": "text+top7", "top7_1": "top7_1",
            "top7_post": "top7_post(+text)", "text+top7_post": "top7_post(+text)", "fix": "top7"}


def _elab(r, err):
    if r.get("mono_kr"):                           # mono: 한섭 확정
        return 0.0
    if r["server"] not in SERVERS:
        return 0.0
    key = ELAB_KEY.get(r["server_basis"], "text")
    return float(err["1e_server"]["label_error_by_basis"][key]["beta_central"]["mean"])


def _text_intervals(p, err):
    """글 % 값 구간 [(log10 하한, log10 상한, 무게)] 1~2개. 소수 둘째 자리 이상은 사진과 같은 구간, 한 자리·없음은 반올림·버림 혼합."""
    def lg(x):
        return float(np.log10(x)) if x > 0 else -30.0
    d = _dec(p)
    if d >= 2:
        return [(lg(p - 0.005), float(np.log10(p + 0.01)), 1.0)]
    w = err["1c_text_pct"]["digits_1"]["mixture"]["w_round"]
    u = 0.1 if d == 1 else 1.0
    return [(lg(p - u / 2), float(np.log10(p + u / 2)), w), (lg(p), float(np.log10(p + u)), 1.0 - w)]


def prepare(rows, H, season, err, K=16, delay_key=None, stage1=False, var=None, dep=None):
    """줄 고르기 + 노드 + 채움 → Obs (numpy dict). delay_key = 1a expanding 시즌 키 (맞힐 시즌, 앞 시즌 값) 또는 None(전체 값).
    var = 변형 이름(VARIANTS) 또는 None. None·base면 이전 코드와 같은 배열.
    dep = 시간 어긋남 변형(WP5a) "dt"·"dtp" 또는 None. None이면 이전 코드와 같은 배열(키도 안 더한다)."""
    if dep not in DEP_KINDS and dep is not None:
        raise ValueError(f"모르는 dep {dep}; {DEP_KINDS}")
    var = get_var(var)
    stats = {}
    sel = select_rows(rows, H, stage1, var, stats)
    cnt = collections.Counter(stats)
    nodes_reason = collections.Counter()
    keep = []
    for r in sel:
        t, w, why = nodes_detail(r, H, err, delay_key, K, var)
        if why:
            nodes_reason[why] += 1
        if t is None:
            cnt["drop_bad_support"] += 1
            continue
        keep.append((r, t, w))
    n = len(keep)
    nodes_n = [int((w > 0).sum()) for _, _, w in keep]
    single = [i for i in range(n) if nodes_n[i] <= 1]
    multi = [i for i in range(n) if nodes_n[i] > 1]
    N1 = next((m for m in FIX_N1 if m >= len(single)), ((len(single) + 1023) // 1024) * 1024)
    N2 = next((m for m in FIX_N2 if m >= len(multi)), ((len(multi) + 1023) // 1024) * 1024)
    N = N1 + N2
    pos = np.zeros(n, int)                      # keep[i]가 놓인 행 번호
    for j, i in enumerate(single):
        pos[i] = j
    for j, i in enumerate(multi):
        pos[i] = N1 + j
    y = np.zeros(N)
    zlo, zhi = np.full(N, -1.0), np.zeros(N)
    u, tt = np.full((N, K), 0.5), np.zeros((N, K))
    lw = np.full((N, K), NEG)
    lw[:, 0] = 0.0
    site, lab, elab = np.zeros(N, int), np.zeros(N, int), np.zeros(N)
    kind, mask = np.zeros(N, int), np.zeros(N)
    text = bool(var and var.get("text"))
    if text:
        zlo2, zhi2 = np.full(N, -1.0), np.zeros(N)
        lwi1, lwi2, eps_row = np.zeros(N), np.full(N, NEG), np.full(N, -1.0)
        eps_text = float(err["1c_text_pct"]["contamination"]["eps_strict_all_photos"]["p"])
    meta = []
    for i0, (r, t, w) in enumerate(keep):
        i = pos[i0]
        is_text = bool(r.get("is_text"))
        if is_text:
            sc = float(r["score"])
            y[i] = np.log(sc + 0.5) if sc.is_integer() else np.log(sc)     # 정수 억이면 [s, s+1)의 가운데
        else:
            y[i] = np.log(r["score"])
        p = r["pct"]
        if is_text:
            iv = _text_intervals(p, err)
            zlo[i], zhi[i] = iv[0][0], iv[0][1]
            zlo2[i], zhi2[i] = (iv[1][0], iv[1][1]) if len(iv) > 1 else (iv[0][0], iv[0][1])
            if len(iv) > 1:
                lwi1[i], lwi2[i] = np.log(iv[0][2]), np.log(iv[1][2])
            eps_row[i] = eps_text
        else:
            zlo[i] = np.log10(p - 0.005) if p - 0.005 > 0 else -30.0
            zhi[i] = np.log10(p + 0.01)
            if text:
                zlo2[i], zhi2[i] = zlo[i], zhi[i]
        tt[i] = t
        u[i] = np.clip(t / H, 0.02, 1.0) if CLOCK is None else u_of(season, t, H)
        lw[i] = np.where(w > 0, np.log(np.maximum(w, 1e-300)), NEG)
        site[i] = SITES.index(r["site"]) if r["site"] in SITES else 3
        lab[i] = SERVERS.index(r["server"]) if r["server"] in SERVERS else -1
        elab[i] = _elab(r, err)
        kind[i] = 3 if r["post"] else (4 if is_text else "ABC".index(r["kind"]))
        mask[i] = 1.0
        if r["tg"] == "A":
            Lo = Up = r["h"]
        elif r["tg"] == "post":
            Lo = Up = float(H)
        elif var and var.get("c_post") and r["kind"] == "C" and r["tg"] == "delay":
            Lo = Up = float(t[0])
        else:
            Lo, Up = support(r, H)
        meta.append({"key": r["key"], "pid": r["pid"], "kind": r["kind"], "tg": r["tg"], "server": r["server"],
                     "site": r["site"], "h": r["h"], "hp": r["hp"], "Lo": Lo, "Up": Up})
        if is_text:
            meta[-1]["is_text"] = True
            cnt["sel_text"] += 1
        cnt["sel_" + r["tg"]] += 1
    fill = mask == 0
    if n:
        y[fill] = y[~fill].mean()
    cnt["selected"] = n
    cnt["N"] = N
    cnt["N_single_block"] = N1
    dsite = err["1e_server"]["unknown_row_site_dirichlet"]
    dirichlet = np.array([[dsite[s]["alpha"][k] for k in SERVERS] for s in ("dc", "naver", "blabla", "*")])
    cd = err["1d_contamination"]
    obs = {"season": season, "H": float(H), "K": K, "n": n, "N": N, "n1": N1, "pos": pos,
            "y": y, "zlo": zlo, "zhi": zhi, "u": u, "lw": lw, "t": tt, "site": site, "lab": lab, "elab": elab,
            "kind": kind, "mask": mask, "meta": meta,
            "counts": {"rows": dict(cnt), "node_reasons": dict(nodes_reason)},
            "dirichlet": dirichlet,
            "eps_ab": (cd["contamination_weight_photo"]["prior_central"]["a"], cd["contamination_weight_photo"]["prior_central"]["b"]),
            "fc_scale": float(cd["contaminant_density"]["family_fits"]["laplace"]["params(shape..., loc, scale)"][-1])}
    if var:                                         # 변형 전용 키 (본안에는 없다)
        obs["var"] = var["name"]
        if text:
            obs.update(zlo2=zlo2, zhi2=zhi2, lwi1=lwi1, lwi2=lwi2, eps_row=eps_row)
        if var.get("contam") is False:
            obs["no_contam"] = True
    if CLOCK:                                       # WP11a: 활동 시간 축 (본안에는 키가 없다)
        obs["clock"] = CLOCK
        obs["counts"]["clock"] = CLOCK
    if dep:                                         # WP5a: 시간 어긋남 매듭 배열 (본안에는 없다)
        obs.update(_dep_arrays(dep, H, tt, [r["pid"] for r, _, _ in keep], pos, N))
        obs["counts"]["dep"] = {"kind": dep, "nk": obs["dep_nk"], "n_pid_effects": obs["dep_np"]}
    return obs


DEP_KINDS = ("dt", "dtp")
DEP_DT = 3.0                                        # 매듭 간격 (시간)
DEP_ARR = ("dep_j0", "dep_w1", "dep_pid")           # to_jax·MAP이 배열로 옮기는 dep 키


def _dep_arrays(dep, H, tt, pids, pos, N):
    """매듭 t_j = 0, 3, 6, ... (H를 넘는 첫 매듭까지) 위 선형 보간용 배열. 노드마다 (왼쪽 매듭 번호 j0, 오른쪽 무게 w1), 줄마다 사람 효과 번호.
    사람 효과는 고른 줄 안에서 줄이 2개 이상인 pid만 (번호 0..np-1, 나머지·채움 줄은 np = 0짜리 칸). dt면 np = 0."""
    nk = int(np.floor(H / DEP_DT)) + 2              # H가 매듭 위여도 H 오른쪽 매듭이 하나 더 있다 (w1 = 0 이어도 색인이 닿게)
    s = np.clip(tt, 0.0, None) / DEP_DT
    j0 = np.minimum(np.floor(s).astype(int), nk - 2)
    w1 = np.clip(s - j0, 0.0, 1.0)
    jH = int(np.floor(H / DEP_DT))
    out = {"dep": dep, "dep_nk": nk, "dep_j0": j0, "dep_w1": w1, "dep_jH": jH, "dep_wH": float(H / DEP_DT - jH)}
    ids = {}
    if dep == "dtp":
        cnt = collections.Counter(pids)
        ids = {p: i for i, p in enumerate(sorted(p for p, c in cnt.items() if c >= 2))}
    npid = len(ids)
    pi = np.full(N, npid, int)
    for i0, p in enumerate(pids):
        if p in ids:
            pi[pos[i0]] = ids[p]
    out["dep_np"] = npid
    out["dep_pid"] = pi
    return out


EXTRA = ("zlo2", "zhi2", "lwi1", "lwi2", "eps_row")
NUMERIC = ("y", "zlo", "zhi", "u", "lw", "t", "site", "lab", "elab", "kind", "mask", "dirichlet")


def to_jax(obs):
    """numpy Obs → jnp (float32/int32) dict. 배열만 옮기고 스칼라(eps_ab·fc_scale·K·N)는 파이썬 값으로 둔다."""
    import jax.numpy as jnp
    out = {}
    for k in NUMERIC:
        a = np.asarray(obs[k])
        out[k] = jnp.asarray(a, jnp.int32 if a.dtype.kind in "iu" else jnp.float32)
    for k in ("eps_ab", "fc_scale", "K", "N", "n", "n1", "H"):
        out[k] = obs[k]
    out["pos"] = np.asarray(obs["pos"])
    for k in EXTRA:                                 # 변형 전용 배열 (있을 때만)
        if k in obs:
            out[k] = jnp.asarray(np.asarray(obs[k]), jnp.float32)
    for k in ("no_contam", "var"):
        if k in obs:
            out[k] = obs[k]
    if "dep" in obs:                                # 시간 어긋남 변형 (WP5a)
        for k in DEP_ARR:
            a = np.asarray(obs[k])
            out[k] = jnp.asarray(a, jnp.int32 if a.dtype.kind in "iu" else jnp.float32)
        for k in ("dep", "dep_nk", "dep_np", "dep_jH", "dep_wH"):
            out[k] = obs[k]
    return out


def dep_add(obs, mu, delta, bp=None):
    """시간 어긋남 더하기 (WP5a). mu[3,N,K]의 한섭 성분(0번)에 노드 시각 보간 delta(t), 모든 서버 성분에 사람 효과 b_pid를 더한다.
    delta = 매듭 값 [nk] (log10 %), bp = 사람 효과 [np] 또는 None. 노드 값은 delta[j0](1 - w1) + delta[j0 + 1] w1."""
    import jax.numpy as jnp
    j0, w1 = obs["dep_j0"], obs["dep_w1"]
    dn = delta[j0] * (1.0 - w1) + delta[j0 + 1] * w1                # [N,K]
    mu = mu.at[0].add(dn)
    if bp is not None:
        b = jnp.concatenate([bp, jnp.zeros((1,), bp.dtype)])[obs["dep_pid"]]    # [N] (한 줄짜리·채움 줄은 0)
        mu = mu + b[None, :, None]
    return mu


# ---------- 1.4 가능도 ----------

def logdiff_ndtr(a, b):
    """log(Phi(b) - Phi(a)), a < b — float32 안정. a + b > 0이면 (-b, -a)로 뒤집어 log_ndtr(b) + log(-expm1(log_ndtr(a) - log_ndtr(b)))."""
    import jax.numpy as jnp
    from jax.scipy.special import log_ndtr
    flip = (a + b) > 0
    a2 = jnp.where(flip, -b, a)
    b2 = jnp.where(flip, -a, b)
    lb = log_ndtr(b2)
    d = jnp.minimum(log_ndtr(a2) - lb, -1e-30)   # 0 이하로 클립 (a=b여도 -inf가 아니라 약 -69)
    return lb + jnp.log(-jnp.expm1(d))


def _logpi(obs, phi):
    import jax.numpy as jnp
    phis = phi[obs["site"]]                                    # [N,3]
    lab = obs["lab"]
    labc = jnp.clip(lab, 0, 2)
    oh = jnp.arange(3)[None, :] == labc[:, None]
    ph_lab = jnp.take_along_axis(phis, labc[:, None], 1)
    other = jnp.maximum(phis.sum(1, keepdims=True) - ph_lab, 1e-12)
    el = obs["elab"][:, None]
    pi_lab = jnp.where(oh, 1.0 - el, el * phis / other)
    pi = jnp.where((lab < 0)[:, None], phis, pi_lab)
    return jnp.log(jnp.maximum(pi, 1e-30))


def _AB(zlo, zhi, mu, P, lw, sigma0, rho, fc_scale, mix=None):
    """노드 합 A = logsumexp_n(lw + logPr), B = logsumexp_n(lw + logfc) -> [3,N]. 입력: zlo·zhi [N], mu·P [3,N,K], lw [N,K].
    mix = (zlo2, zhi2, lwi1, lwi2) [N] — 값 구간 혼합(글 %): logPr = logaddexp(lwi1 + 구간1, lwi2 + 구간2)."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    sig = jnp.sqrt(sigma0 ** 2 + (rho * P) ** 2)
    zl, zh = zlo[None, :, None], zhi[None, :, None]
    logpr = logdiff_ndtr((zl - mu) / sig, (zh - mu) / sig)
    if mix is not None:
        zl2, zh2 = mix[0][None, :, None], mix[1][None, :, None]
        logpr2 = logdiff_ndtr((zl2 - mu) / sig, (zh2 - mu) / sig)
        logpr = jnp.logaddexp(mix[2][None, :, None] + logpr, mix[3][None, :, None] + logpr2)
    logfc = -jnp.abs(0.5 * (zl + zh) - mu) / fc_scale - jnp.log(2 * fc_scale) + jnp.log(zh - zl)
    return logsumexp(lw[None] + logpr, axis=2), logsumexp(lw[None] + logfc, axis=2), logpr, logfc


def row_loglik(obs, mu, P, sigma0, rho, eps, phi, parts=False):
    """줄마다 로그 가능도 ll[N] (채움 줄 0). parts=True면 dict(ll, p_contam, p_server, t_mean).
    obs["n1"]이 있으면 앞 n1줄은 노드 하나(0번)만 있는 줄이라 그 노드만 계산한다 (prepare가 그렇게 늘어놓는다; 시간 절약, 값은 같다)."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    s = obs["fc_scale"]
    n1 = obs.get("n1", 0)
    mix = (obs["zlo2"], obs["zhi2"], obs["lwi1"], obs["lwi2"]) if "zlo2" in obs else None   # 변형 text
    if mix is None:
        mx1 = mx2 = None
    else:
        mx1, mx2 = tuple(a[:n1] for a in mix), tuple(a[n1:] for a in mix)
    if parts or n1 == 0:
        A, B, logpr, logfc = _AB(obs["zlo"], obs["zhi"], mu, P, obs["lw"], sigma0, rho, s, mix)
    else:
        A1, B1, _, _ = _AB(obs["zlo"][:n1], obs["zhi"][:n1], mu[:, :n1, :1], P[:, :n1, :1], obs["lw"][:n1, :1], sigma0, rho, s, mx1)
        A2, B2, _, _ = _AB(obs["zlo"][n1:], obs["zhi"][n1:], mu[:, n1:], P[:, n1:], obs["lw"][n1:], sigma0, rho, s, mx2)
        A, B = jnp.concatenate([A1, A2], axis=1), jnp.concatenate([B1, B2], axis=1)
    if "eps_row" in obs:                                        # 줄마다 eps (글 줄은 고정 값, 사진은 표집 값)
        eps = jnp.where(obs["eps_row"] >= 0, obs["eps_row"], eps)
    le, l1e = jnp.log(eps), jnp.log1p(-eps)
    nocon = bool(obs.get("no_contam"))
    comp = A if nocon else jnp.logaddexp(l1e + A, le + B)
    lp = _logpi(obs, phi)                                       # [N,3]
    joint = lp.T + comp                                         # [3,N]
    ll = logsumexp(joint, axis=0) * obs["mask"]
    if not parts:
        return ll
    tot = logsumexp(joint, axis=0)
    p_c = jnp.exp(logsumexp(lp.T + le + B, axis=0) - tot)
    p_s = jnp.exp(joint - tot).T
    if "eps_row" in obs:
        le, l1e = le[None, :, None], l1e[None, :, None]
    nodell = logpr if nocon else jnp.logaddexp(l1e + logpr, le + logfc)   # [3,N,K]
    if nocon:
        p_c = jnp.zeros_like(p_c)
    lnode = obs["lw"] + logsumexp(lp.T[:, :, None] + nodell, axis=0)
    pn = jnp.exp(lnode - logsumexp(lnode, axis=1, keepdims=True))
    t_mean = (pn * obs["t"]).sum(1)
    return {"ll": ll, "p_contam": p_c, "p_server": p_s, "t_mean": t_mean}


def observe(obs, mu, P, sigma0, rho, keep_ll=False, eps=None, phi=None):
    """numpyro 모형 안에서: eps·phi 표집 + 관측 가능도 factor. eps·phi를 주면(사전 펴기 쪽에서 따로 다룰 때) 표집하지 않고 그 값을 쓴다."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    if eps is None:
        eps = numpyro.sample("eps", dist.Beta(*[jnp.float32(v) for v in obs["eps_ab"]]))
    if phi is None:
        phi = numpyro.sample("phi", dist.Dirichlet(obs["dirichlet"]))
    ll = row_loglik(obs, mu, P, sigma0, rho, eps, phi)
    numpyro.factor("obs_ll", ll.sum())
    if keep_ll:
        numpyro.deterministic("ll", ll)


# ---------- 1.5 실행 도구 ----------

def run_nuts(model, model_kwargs, seed=0, chains=4, warmup=500, samples=500, target_accept=0.9,
             max_tree_depth=10, dense=True, init=None):
    """NUTS (한 프로세스 chain_method='parallel'). (samples dict[name -> ndarray[chains, draws, ...]], info)."""
    import jax
    from numpyro.infer import MCMC, NUTS, init_to_median
    kern = NUTS(model, target_accept_prob=target_accept, max_tree_depth=max_tree_depth, dense_mass=dense,
                init_strategy=init if init is not None else init_to_median(num_samples=15))
    mcmc = MCMC(kern, num_warmup=warmup, num_samples=samples, num_chains=chains, chain_method="parallel",
                progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"), **model_kwargs)
    sm = mcmc.get_samples(group_by_chain=True)
    ex = mcmc.get_extra_fields(group_by_chain=True)
    jax.block_until_ready((sm, ex))   # jax는 비동기라 이걸 기다려야 걸린 시간이 맞다
    secs = time.time() - t0
    steps = np.asarray(ex["num_steps"])
    dep = np.ceil(np.log2(steps + 1)).astype(int)
    info = {"secs": round(secs, 1), "divergences": int(np.asarray(ex["diverging"]).sum()),
            "tree_cap_frac": float((steps >= 2 ** max_tree_depth - 1).mean()), "mean_leapfrog": float(steps.mean()),
            "depth_dist": {int(d): round(float((dep == d).mean()), 4) for d in np.unique(dep)},   # 나무 깊이별 몫 (뽑기 구간)
            "chains": chains, "warmup": warmup, "samples": samples, "target_accept": target_accept}
    return {k: np.asarray(v) for k, v in sm.items()}, info


def diagnose(samples, names=None, rhat_max=1.01, ess_min=400, divergences=0):
    """이름마다 분할 R-hat 최대·벌크 ESS 최소. conv = R-hat < 1.01 and 발산 0 and ESS >= 400 (발산은 호출자가 divergences로 준다)."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    names = names or [k for k in samples if k not in ("ll", "phi")]
    per = {}
    for k in names:
        a = np.asarray(samples[k])
        per[k] = {"rhat": float(np.nanmax(np.asarray(split_gelman_rubin(a)))),
                  "ess": float(np.nanmin(np.asarray(effective_sample_size(a))))}
    rmax = max(v["rhat"] for v in per.values())
    emin = min(v["ess"] for v in per.values())
    return {"per": per, "rhat_max": rmax, "ess_min": emin, "divergences": divergences,
            "conv": bool(rmax < rhat_max and divergences == 0 and emin >= ess_min)}


def fit_with_retry(model, model_kwargs, names=None, seed=0, chains=4, warmup=500, samples=500, target_accept=0.9,
                   max_tree_depth=10, dense=True, init=None, retry_warmup=1000, retry_ok=None):
    """run_nuts → diagnose → 못 지나면 target_accept 0.95·warmup retry_warmup(기본 1000)으로 한 번 더. (samples, info, diag) — info["retried"].
    retry_ok = 첫 뽑기 초(float)를 받아 재시도해도 되는지 답하는 함수 (기본 None = 항상 재시도). 안 되면 첫 뽑기를 그대로 두고 info["retry_skipped"] = True."""
    sm, info = run_nuts(model, model_kwargs, seed, chains, warmup, samples, target_accept, max_tree_depth, dense, init)
    dg = diagnose(sm, names, divergences=info["divergences"])
    info["retried"] = False
    if dg["conv"]:
        return sm, info, dg
    first = info["secs"]
    if retry_ok is not None and not retry_ok(first):
        info["retry_skipped"] = True
        return sm, info, dg
    sm, info = run_nuts(model, model_kwargs, seed + 1, chains, retry_warmup, samples, 0.95, max_tree_depth, dense, init)
    dg = diagnose(sm, names, divergences=info["divergences"])
    info["retried"] = True
    info["secs_first_try"] = first
    return sm, info, dg


def row_posteriors(obs, samples, surface_fn, max_draws=400):
    """사후 뽑기(최대 max_draws, 고르게)마다 row_loglik(parts=True)의 평균. surface_fn(draw: dict[name -> 한 뽑기 값], jobs) → (mu, P, sigma0, rho).
    samples에는 모형 표집 이름 + eps·phi가 있어야 한다. obs는 to_jax 결과에 prepare의 "pos"를 더한 것 (to_jax가 pos를 np 배열로 둔다).
    실제 줄(n개)만 meta 순서로 돌려준다."""
    import jax
    import jax.numpy as jnp
    flat = {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in samples.items() if k != "ll"}
    tot = len(next(iter(flat.values())))
    idx = np.unique(np.linspace(0, tot - 1, min(max_draws, tot)).astype(int))

    @jax.jit
    def one(draw):
        mu, P, s0, rho = surface_fn(draw, obs)
        return row_loglik(obs, mu, P, s0, rho, draw["eps"], draw["phi"], parts=True)

    acc = None
    for i in idx:
        r = one({k: jnp.asarray(v[i]) for k, v in flat.items()})
        r = {k: np.asarray(v) for k, v in r.items() if k != "ll"}
        acc = r if acc is None else {k: acc[k] + r[k] for k in r}
    pos = np.asarray(obs["pos"])
    return {k: v[pos] / len(idx) for k, v in acc.items()}


def flags(meta, rp, thr=0.5):
    """사후 오염 확률이 thr를 넘는 실제 줄 목록 (r7/flag_{M}_{판}.json 용). meta = prepare의 obs["meta"]."""
    out = []
    for i, m in enumerate(meta):
        if rp["p_contam"][i] > thr:
            out.append({"key": m["key"], "pid": m["pid"], "kind": m["kind"], "tg": m["tg"], "server": m["server"],
                        "p_contam": round(float(rp["p_contam"][i]), 4), "p_kr": round(float(rp["p_server"][i][0]), 4)})
    return out


def pred_cells(season, T, draws_lnC, conv, server="KR"):
    """score.py 예측 파일 꼴의 칸 목록. draws_lnC = {q(%): ln 컷 뽑기 배열}. pred = exp(중앙), i80 = 10·90%, i95 = 2.5·97.5%."""
    cells = []
    for q in sorted(draws_lnC):
        d = np.asarray(draws_lnC[q]).ravel()
        lo80, med, hi80, lo95, hi95 = np.quantile(d, [0.1, 0.5, 0.9, 0.025, 0.975])
        cells.append({"season": season, "T": float(T), "pct": float(q), "server": server, "pred": float(np.exp(med)),
                      "i80": [float(np.exp(lo80)), float(np.exp(hi80))], "i95": [float(np.exp(lo95)), float(np.exp(hi95))],
                      "own": True, "conv": bool(conv)})
    return cells


# ---------- 합성 % 뽑기 (SBC·PPC·시험 공용; 2026-10-29 더함, 기존 인자 안 바꿈) ----------

def simulate_pct(sel, H, err, mu_sd_fn, eps, phi_true, seed=0, delay_key=None, K=64, season=None):
    """select_rows가 고른 줄 배치(점수·종류·시각 정보·서버 라벨·사이트)를 그대로 두고 pct만 곡면에서 새로 뽑는다.
    mu_sd_fn(y[N], u[N]) -> (mu[3,N], sd[3,N]) numpy (서버 성분 KR·JP·GL의 log10 %와 잔차 sd). eps = 오염 확률, phi_true = [4,3] 사이트별 서버 몫.
    순서: 서버(라벨 줄은 elab 확률로 다른 서버, 미상 줄은 phi_true[사이트]) -> 촬영 시각(노드 K개 중 하나) -> 곡면 + 정규 잡음 ->
    오염(eps, 라플라스 0.34 더함) -> 표시(반올림/버림 반반, 0.01 단위). 100% 초과·0.01% 미만은 버린다.
    -> (rows [pct 바뀐 줄 목록], info {server_true, contam, t_shot, n_dropped})"""
    rng = np.random.default_rng(seed)
    keep, tshot, srv = [], [], []
    for r in sel:
        t, w, why = nodes_detail(r, H, err, delay_key, K)
        if t is None:
            continue
        tshot.append(float(rng.choice(t, p=w / w.sum())))
        si = SITES.index(r["site"]) if r["site"] in SITES else 3
        if r["server"] in SERVERS:
            lab = SERVERS.index(r["server"])
            if rng.random() < _elab(r, err):
                pr = np.array(phi_true[si], float)
                pr[lab] = 0
                srv.append(int(rng.choice(3, p=pr / pr.sum())))
            else:
                srv.append(lab)
        else:
            srv.append(int(rng.choice(3, p=np.asarray(phi_true[si], float) / np.sum(phi_true[si]))))
        keep.append(r)
    n = len(keep)
    if n == 0:
        return [], {"server_true": [], "contam": [], "t_shot": [], "n_dropped": 0}
    y = np.log([r["score"] for r in keep])
    tshot = np.array(tshot)
    u = np.clip(tshot / H, 0.02, 1.0) if CLOCK is None else u_of(season, tshot, H)   # 켠 시간 축으로 뽑는다 (season 필요)
    mu, sd = mu_sd_fn(y, u)
    srv = np.array(srv)
    z = mu[srv, np.arange(n)] + sd[srv, np.arange(n)] * rng.standard_normal(n)
    cont = rng.random(n) < eps
    z = np.where(cont, z + rng.laplace(0, 0.34, n), z)
    x100 = 10 ** np.clip(z, -6, 6) * 100
    pct = np.where(rng.random(n) < 0.5, np.round(x100), np.floor(x100)) / 100
    out, kk = [], []
    for i, (r, p) in enumerate(zip(keep, pct)):
        if p > 100.0 or p < 0.01:
            continue
        x = dict(r)
        x["pct"] = float(p)
        out.append(x)
        kk.append(i)
    kk = np.array(kk, int)
    return out, {"server_true": srv[kk].tolist(), "contam": cont[kk].tolist(), "t_shot": tshot[kk].tolist(), "n_dropped": n - len(out)}
