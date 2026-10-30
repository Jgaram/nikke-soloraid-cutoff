"""M2 HSGP 곡면 (WP3b M2) — 곡면 함수와 모형, MAP + 라플라스, 시간 재기, M1 최종 꼴 평균의 1단계 (2026-10-29). m1.py 꼴을 따른다.

  import m2 as M2        # sys.path.insert(0, HERE) 뒤
  python3 -I tools/solo_raid/lab7/m2.py stage1 --obs W/obs7/full --seasons S22,S26,... --m1-stage1 r7/wp3b/stage1_M1_kuma_b_r2_D1.json --out r7/wp3b_m2/stage1_M2_tune_D1.json --ll-dir W/m2/ll [--resume]
  python3 -I tools/solo_raid/lab7/m2.py test --which m1match --m1-stage1 r7/wp3b/stage1_M1_kuma_b_r2_D1.json --season S30 [--work W]   # M1과 같은 곡면인지 (f1 = f2 = 0)
  python3 -I tools/solo_raid/lab7/m2.py bench --work W --case a|b|c [--season S30] [--chains 4 --warmup 500 --samples 500]      # 시간 재기 (임시 평균, 옛 방식)
  python3 -I tools/solo_raid/lab7/m2.py test --work W --which basis|synth

명령 요약 (2026-10-30 더함 — 시즌 묶음 1단계·합치기·칸 예측):
  m2.py stage1 --seasons S19-S24 (범위·쉼표 모두) --out 시즌 묶음 파일 ...      # 코드 판(git head)·설정·M1 1단계 sha1이 파일에 적힌다
  m2.py merge-stage1 --in a.json b.json ... --out r7/stage1_M2_D1.json         # 코드 판·설정·M1 파일이 같아야 하고 시즌이 겹치면 오류
  m2.py predict --obs W/obs7 --seasons S42 --T 48 --m2-stage1 r7/stage1_M2_D1.json --m1-stage1 r7/stage1_M1_D1.json [--m1-hier r7/hier_M1_D1.json] --out 예측.json [--cache DIR --log FILE --resume]
            # 칸마다 score.py 예측 꼴 cells + fits (칸별 conv·시간·depth_dist). 사전은 맞힐 시즌 앞 시즌의 1단계만.

도우미 세션용 명령 (m2.py·관측 층을 고치지 않고 돌리기만 한다. 코드 판은 조율 세션이 알려 준 커밋 그대로. W = 세션 임시 작업 폴더, 한 컨테이너 4코어)
  pip install -r tools/solo_raid/lab7/requirements.txt                                          # 약 1분 40초
  python3 -I tools/solo_raid/lab7/obs_table.py build --out $W/obs7 --full                       # 끝난 시즌 전체 관측 표, 약 8초 -> $W/obs7/full/S{n}.jsonl.gz
  nohup python3 -I tools/solo_raid/lab7/m2.py stage1 --obs $W/obs7/full --seasons S19-S24 \
      --m1-stage1 data/solo-raid/analysis/r7/stage1_M1_D1.json \
      --out $W/m2/stage1_M2_D1_S19-S24.json --ll-dir $W/m2/ll --resume > $W/m2/S19-S24.log 2>&1 &   # 시즌당 약 45분, 끊기면 같은 명령을 다시 (--resume)
  결과 파일 이름: stage1_M2_D1_{첫 시즌}-{끝 시즌}.json (예: S19-S24, S25-S30, S31-S36, S37-S42). 돌려준 파일 + attempts.log(= $W/m2/attempts.log). 합치기는 조율 세션이:
  python3 -I tools/solo_raid/lab7/m2.py merge-stage1 --in stage1_M2_D1_S19-S24.json stage1_M2_D1_S25-S30.json ... --out data/solo-raid/analysis/r7/stage1_M2_D1.json
  (같은 --out으로 시즌 묶음을 달리 돌리면 안 된다. 묶음마다 --out이 다르다. 한 묶음이 한 프로세스, 같은 컨테이너에서 두 묶음을 동시에 돌리지 않는다.)

M1 최종 꼴(kuma 침수 + x항 beta + 2차 컷 기울기 r2)에 맞춤 (2026-10-29): 평균 r̄(w) = r0 + r1 w + r2 w^2, 침수 평균 = M1.flood(m = 1) (P(0.4) = 1), x항 P * exp(beta (z_fin - log10 3)).
  침수 평균은 u 격자 구간별 P의 정확한 감소분 dP로 들고 (M1 kuma의 beta < 1이면 -dP/du가 u = 1에서 발산해 점값 사다리꼴은 틀린다), 구간 질량 = dP * m * (구간 양끝 exp(f2) 평균).
  u 격자는 코사인 간격 (끝 간격 1e-4): 옛 제곱 간격은 u = 1 쪽이 0.0098로 성겨 P 보간이 막판 1%에서 틀렸다. eps·phi도 v에 붙여 MAP과 NUTS가 함께 다룬다 (m1 full=True: 오염 많은 시즌에서 MAP이 어긋났던 것).
  v 벡터: [a, lnm, lnsig0, lnrho, beta, dJP, dGL, pJP, pGL, kJP, kGL, (lnell1, lnalp1, lnell2, lnalp2), beta1[mw], beta2[mu], logit eps, phi stick-breaking 8개].
  mean_from_m1(theta, form) 검증은 test --which m1match (M1.surface·M1.lnC와 mu·P 차 4e-4 안, lnC 차 1.4e-4 안 — q 0.1~10%).

(아래는 첫 판 서술 — 평균·x항·격자 이외는 그대로다)
명세 「WP3 명세」 1·4·5·6절. 좌표 y = ln 점수, w = ln(q/3), u = h/H.
  최종 컷: ln C_k(w) = a_k - int_0^w exp(r_k(v)) dv,  r_k(w) = rbar(w) + f1(w) + kappa_k,  a_k = a + delta_k
           w 격자 120점 [ln(0.1/3), ln(40/3)] (w = 0이 격자 점 — 음쪽 69점 + 양쪽 52점으로 따로 등간격), 누적 사다리꼴.
           점수 y -> 최종 w_fin: ln C 격자를 역으로 보간, 격자 밖은 끝 기울기 exp(r_end)로 선형 외삽. z_fin = log10 3 + w_fin / ln 10.
  침수:    -dP/du = exp(l(u)),  l(u) = lbar(u) + ln m + psi_k + f2(u),  P(u) = int_u^1 exp(l) dv.
           u 격자 200점 [0.02, 1], 제곱 간격 u = 0.02 + 0.98 s^2 (s 등간격) — 앞쪽이 촘촘하고 끝 간격 약 0.0098.
           역방향 누적 사다리꼴, 관측 u[N,K]에서 선형 보간 (u = 1이면 정확히 0). f2는 u 격자 점 값의 평균을 빼 0으로 묶는다.
           (psi_k는 P 전체에 곱하는 exp(psi_k)로 정확히 같다.)
  f1·f2:   HSGP (Solin-Sarkka 2020), Matern-5/2, M = 20, c = 1.5. phi_j(x) = sqrt(1/L) sin(pi j (x - x0 + L)/(2L)),
           f = sum_j sqrt(s(sqrt_lam_j)) beta_j phi_j(x), beta ~ N(0,1) (비중심). 직접 쓴 순수 jnp (float64 MAP에서도 돈다).
  지금 %: mu = z_fin - P.  서버 k(KR·JP·GL): dJP,dGL (a), kJP,kGL (r에 더함), pJP,pGL (ln m에 더함) — m1과 같은 이름. KR은 0.

평균 함수 rbar[120]·lbar[200]은 인자다 (mean = {"rbar", "lbar"}; None이면 임시 평균):
  임시 평균 (M1 2단계 평균 곡선이 아직 없어서, 2026-10-29): rbar(w) = ln 0.3 + 1.0 w, lbar = M1 kuma eta=0 (F(u)=u)의 -dP/du 로그
  (= -ln u - ln(-ln 0.4)) 를 격자에서 P(0.4) = 1이 되게 다시 정규화.  default_mean() / mean_from_m1(theta, form)  (M1 theta에서 만드는 함수).

초모수 (f1·f2 각각 길이 ell, 크기 alpha):
  hyper=None (1단계): ell ~ InverseGamma(a, b) — 정의역 폭 W의 [W/10, W]가 1%~99% 분위가 되게 (scipy로 풀어 inv_gamma_params(W)),
                       alpha ~ HalfNormal(0.3). 로그 척도 (lnell1, lnalp1, lnell2, lnalp2)로 재매개화해 pre(라플라스)에 같이 넣는다.
  hyper = {"ell1","alpha1","ell2","alpha2"} (2단계): 고정값. hyper_center()가 사전 중앙값 (IG 중앙값, HalfNormal 중앙값 0.202).
  W: f1은 w 정의역 폭 5.99, f2는 u 정의역 폭 0.98.

  mu, P = surface(theta, y, u, G, mean)       # y[N], u[N,K] -> mu[3,N,K], P[3,N,K]  (obs_model 곡면 함수 약속). theta: a,lnm,dJP..kGL,beta1,beta2 + ell1,alpha1,ell2,alpha2
  model(obs, prior, hyper=None, mean=None, pre=None, keep_ll=False)   # numpyro 모형. prior = {"kind": "weak"} 또는 hier.prior_for 꼴 (names에 lnm·lnsig0·lnrho·d/p/k·a가 있어야 한다)
  pre = map_precond(jobs, prior, hyper, mean)  # float64 MAP + 라플라스. model(..., pre=pre)로 넘긴다. (pre 없으면 사전 평균 중심 대각 척도)
  표집 이름: z (불확실 벡터, 실제 값은 deterministic 아래 이름들), eps, phi. deterministic: a, lnm, lnsig0, lnrho, dJP..kGL,
             (hyper=None이면) lnell1, lnalp1, lnell2, lnalp2, beta1[20], beta2[20], lnC_{q} (q = 0.5 1 2 3 5 10, KR).
  v 벡터 순서: [a, lnm, lnsig0, lnrho, dJP, dGL, pJP, pGL, kJP, kGL, (lnell1, lnalp1, lnell2, lnalp2), beta1(20), beta2(20)].
공개 규칙: 글 번호·원문·닉네임은 쓰지 않는다.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_model as OM  # noqa: E402
import m1 as M1  # noqa: E402
import common as C  # noqa: E402
import hier as H  # noqa: E402

LN10 = math.log(10.0)
L10_3 = math.log10(3.0)
MB = 20            # 기저 수
CB = 1.5           # 경계 배수
NW_NEG, NW_POS = 69, 52     # w 격자: [lo, 0] 69점 + (0, hi] 51점 = 120점, 0이 격자 점
NU = 200
U_LO = 0.02
W_LO, W_HI = math.log(0.1 / 3.0), math.log(40.0 / 3.0)
CORE = ["lnm", "lnsig0", "lnrho", "beta"] + list(M1.OFF)   # a 다음 10개 (beta = x항, M1 최종 꼴 kuma + beta + r2를 따름 — 2026-10-29)
N_EP = 9           # v 꼬리: logit eps(1) + phi stick-breaking(8) — M1 map_precond(full=True)처럼 MAP·NUTS가 함께 다룬다
HYP = ["lnell1", "lnalp1", "lnell2", "lnalp2"]
ALPHA_SD = 0.3


# ---------- 격자 · 기저 (numpy 상수) ----------

def hsgp_basis(x, lo, hi, M=MB, c=CB):
    """-> (Phi[len(x), M], sqrt_lam[M], L). 정의역 [lo, hi]의 중심 x0, 반폭 S, L = c S."""
    x = np.asarray(x, float)
    x0, S = 0.5 * (lo + hi), 0.5 * (hi - lo)
    L = c * S
    j = np.arange(1, M + 1)
    phi = np.sqrt(1.0 / L) * np.sin(np.pi * j[None, :] * (x[:, None] - x0 + L) / (2 * L))
    return phi, np.pi * j / (2 * L), L


def matern52_spec(omega, ell, alpha):
    """1차원 Matern-5/2 스펙트럼 밀도 (jnp/np 둘 다)."""
    return alpha ** 2 * (16.0 / 3.0) * 5.0 ** 2.5 / ell ** 5 * (5.0 / ell ** 2 + omega ** 2) ** (-3.0)


def matern52_cov(d, ell, alpha):
    r = np.sqrt(5.0) * np.abs(d) / ell
    return alpha ** 2 * (1 + r + r ** 2 / 3.0) * np.exp(-r)


class Grids:
    """w·u 격자와 HSGP 기저 (numpy float64 상수). mw·mu = f1·f2 기저 수."""

    def __init__(self, mw=MB, mu=MB):
        wn = np.linspace(W_LO, 0.0, NW_NEG)
        wp = np.linspace(0.0, W_HI, NW_POS)[1:]
        self.w = np.concatenate([wn, wp])
        self.i0 = NW_NEG - 1
        assert self.w[self.i0] == 0.0 and len(self.w) == 120
        # u 격자 (2026-10-29 바꿈): 양끝이 촘촘한 코사인 간격. 옛 제곱 간격은 u = 1 쪽 간격이 0.0098이라, 침수가 (1-u)^beta (beta < 1, M1 kuma 적합이 0.4~0.5)처럼
        # u = 1에서 가팔라지는 곡선을 선형 보간하면 막판 1% 안 P가 크게 틀렸다. 코사인 간격은 끝 간격 약 1e-4, 가운데 약 0.0077 (옛 가운데와 같은 수준).
        s = np.linspace(0.0, 1.0, NU)
        self.u = U_LO + (1.0 - U_LO) * 0.5 * (1.0 - np.cos(np.pi * s))
        self.u[0], self.u[-1] = U_LO, 1.0
        self.mw, self.mu = mw, mu
        self.Phi_w, self.sl_w, _ = hsgp_basis(self.w, W_LO, W_HI, mw)
        self.Phi_u, self.sl_u, _ = hsgp_basis(self.u, U_LO, 1.0, mu)
        self.width_w = W_HI - W_LO
        self.width_u = 1.0 - U_LO


_G = {}


def grids(mw=MB, mu=MB):
    if (mw, mu) not in _G:
        _G[(mw, mu)] = Grids(mw, mu)
    return _G[(mw, mu)]


def min_len(width, M, c=CB):
    """Riutort-Mayol 진단: Matern-5/2를 기저 M개·경계 배수 c로 나타낼 수 있는 최소 길이 = 2.65 c / M * S (S = 정의역 반폭)."""
    return 2.65 * c / M * 0.5 * width


# ---------- 평균 함수 ----------

def _normalize_lbar(u, lbar):  # 옛 사다리꼴 정규화 (쓰지 않음)
    """격자 사다리꼴 P(0.4) = 1이 되게 lbar에 상수를 더한다 (numpy)."""
    e = np.exp(lbar)
    seg = 0.5 * (e[1:] + e[:-1]) * np.diff(u)
    P = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    p04 = np.interp(0.4, u, P)
    return lbar - math.log(p04)


def _mean_dict(G, rbar, P):
    """r̄[120]과 침수 P(u 격자 점, m = 1, P(0.4) = 1)에서 평균 dict. dP = 격자 구간별 P의 정확한 감소분 (곡선이 u = 1에서 가팔라도 오차 없음),
    lbar = ln(-dP/du) 격자 점 값 (구간 밀도의 이웃 평균 — 보고용)."""
    P = np.asarray(P, float)
    dP = P[:-1] - P[1:]
    dens = np.maximum(dP / np.diff(G.u), 1e-300)
    d_node = np.concatenate([[dens[0]], 0.5 * (dens[1:] + dens[:-1]), [dens[-1]]])
    return {"rbar": np.asarray(rbar, float), "lbar": np.log(d_node), "dP": dP}


def default_mean(G=None):
    """임시 평균: rbar = ln 0.3 + w, 침수 P(u) = ln u / ln 0.4 (M1 kuma eta=0, F(u)=u; P(0.4)=1)."""
    G = G or grids()
    return _mean_dict(G, math.log(0.3) + 1.0 * G.w, np.log(G.u) / math.log(0.4))


def mean_from_m1(theta, form="kuma", G=None):
    """M1 theta(dict: r0, r1, [r2], eta)에서 평균 곡선: rbar(w) = r0 + r1 w + r2 w^2 (m1 _r2_grid와 같은 뜻 — exp(rbar)가 컷 기울기),
    침수 P(u) = M1.flood(m = 1, eta, u) = p0(u)/p0(0.4) (P(0.4) = 1, u = 1에서 0)를 u 격자 점에서 float64로 내고 구간별 감소분을 dP로 둔다.
    lbar = ln(-dP/du)는 보고용 (surface는 dP를 쓴다 — M1 kuma의 beta < 1이면 -dP/du가 u = 1에서 발산해 점값 사다리꼴이 부정확)."""
    import jax
    import jax.numpy as jnp
    G = G or grids()
    r2 = float(theta.get("r2", 0.0))
    rbar = float(theta["r0"]) + float(theta["r1"]) * G.w + r2 * G.w ** 2
    with jax.enable_x64(True):
        eta = [jnp.asarray(float(e), jnp.float64) for e in M1._eta_of(theta, form)]
        P = np.asarray(M1.flood(jnp.asarray(1.0, jnp.float64), eta, jnp.asarray(G.u, jnp.float64), form), float)
    return _mean_dict(G, rbar, P)


# ---------- 초모수 사전 ----------

def inv_gamma_params(W):
    """InverseGamma(a, b): 1% 분위 = W/10, 99% 분위 = W 가 되는 (a, b)."""
    from scipy import optimize, stats
    lo, hi = W / 10.0, W

    def f(p):
        a, b = math.exp(p[0]), math.exp(p[1])
        return [stats.invgamma.ppf(0.01, a, scale=b) - lo, stats.invgamma.ppf(0.99, a, scale=b) - hi]
    s = optimize.fsolve(f, [math.log(3.0), math.log(0.5 * W)])
    return math.exp(s[0]), math.exp(s[1])


def hyper_center(G=None):
    """사전 중앙값 (길이 = IG 중앙값, 크기 = HalfNormal(0.3) 중앙값)."""
    from scipy import stats
    G = G or grids()
    out = {}
    for tag, W in (("1", G.width_w), ("2", G.width_u)):
        a, b = inv_gamma_params(W)
        out["ell" + tag] = float(stats.invgamma.ppf(0.5, a, scale=b))
        out["alpha" + tag] = float(stats.halfnorm.ppf(0.5, scale=ALPHA_SD))
    return out


def _hyper_logp(x4, igp):
    """x = ln(ell), ln(alpha) 척도의 로그 밀도 (야코비안 포함). x4 = [lnell1, lnalp1, lnell2, lnalp2]."""
    import jax.numpy as jnp
    from jax.scipy.special import gammaln
    tot = 0.0
    for i, (a, b) in enumerate(igp):
        lx = x4[2 * i]
        ell = jnp.exp(lx)
        tot = tot + a * math.log(b) - gammaln(a) - (a + 1.0) * lx - b / ell + lx
        la = x4[2 * i + 1]
        al = jnp.exp(la)
        tot = tot + math.log(2.0) - 0.5 * math.log(2 * math.pi) - math.log(ALPHA_SD) - 0.5 * (al / ALPHA_SD) ** 2 + la
    return tot


def _igps(G):
    return [inv_gamma_params(G.width_w), inv_gamma_params(G.width_u)]


# ---------- 곡면 ----------

def _hsgp_f(Phi, sl, ell, alpha, beta):
    import jax.numpy as jnp
    sp = jnp.sqrt(matern52_spec(jnp.asarray(sl, beta.dtype), ell, alpha))
    return jnp.asarray(Phi, beta.dtype) @ (sp * beta)


def curves(theta, G, mean):
    """-> dict(lc[3,120] 서버별 ln C 격자, Q[200] 침수 P 격자 (m, f2 포함; 서버 배수 전), rg[3,120])."""
    import jax.numpy as jnp
    dt = jnp.asarray(theta["a"]).dtype
    w = jnp.asarray(G.w, dt)
    f1 = _hsgp_f(G.Phi_w, G.sl_w, theta["ell1"], theta["alpha1"], theta["beta1"])
    f2 = _hsgp_f(G.Phi_u, G.sl_u, theta["ell2"], theta["alpha2"], theta["beta2"])
    f2 = f2 - f2.mean()
    z = jnp.zeros((), dt)
    dl = [z, theta["dJP"], theta["dGL"]]
    kp = [z, theta["kJP"], theta["kGL"]]
    lcs, rgs = [], []
    dw = jnp.diff(w)
    for k in range(3):
        r = jnp.clip(jnp.asarray(mean["rbar"], dt) + f1 + kp[k], -15.0, 10.0)
        e = jnp.exp(r)
        cum = jnp.concatenate([jnp.zeros(1, dt), jnp.cumsum(0.5 * (e[1:] + e[:-1]) * dw)])
        cum = cum - cum[G.i0]
        lcs.append(theta["a"] + dl[k] - cum)
        rgs.append(r)
    if "dP" in mean:
        # 구간 질량 = (M1 평균 곡선의 정확한 구간 P 감소분) x m x 구간 양끝 exp(f2)의 평균. f2 = 0이면 M1 침수와 같다.
        ef = jnp.exp(jnp.clip(f2, -10.0, 10.0))
        seg = jnp.asarray(mean["dP"], dt) * jnp.exp(theta["lnm"]) * 0.5 * (ef[1:] + ef[:-1])
    else:
        u = jnp.asarray(G.u, dt)
        el = jnp.exp(jnp.clip(jnp.asarray(mean["lbar"], dt) + theta["lnm"] + f2, -15.0, 10.0))
        seg = 0.5 * (el[1:] + el[:-1]) * jnp.diff(u)
    Q = jnp.concatenate([jnp.cumsum(seg[::-1])[::-1], jnp.zeros(1, dt)])
    return {"lc": jnp.stack(lcs), "rg": jnp.stack(rgs), "Q": Q, "f1": f1, "f2": f2}


def _inv_w(y, wg, lc, r):
    """y[N] -> 최종 w. lc는 w에 감소. 격자 밖은 끝 기울기 exp(r_end)로 선형 외삽."""
    import jax.numpy as jnp
    xp, fp = lc[::-1], wg[::-1]
    w = jnp.interp(y, xp, fp)
    w_hi = wg[-1] + (lc[-1] - y) / jnp.exp(r[-1])    # y < xp[0]: 점수가 40% 컷 밑
    w_lo = wg[0] + (lc[0] - y) / jnp.exp(r[0])       # y > xp[-1]: 점수가 0.1% 컷 위
    return jnp.where(y < xp[0], w_hi, jnp.where(y > xp[-1], w_lo, w))


def surface(theta, y, u, G=None, mean=None):
    """-> (mu[3,N,K], P[3,N,K]). theta 약속은 머리 참조. x항(theta["beta"]가 있으면): P * exp(beta * (z_fin - log10 3)) — m1.surface의 x_term과 같은 식."""
    import jax.numpy as jnp
    G = G or grids()
    mean = mean or default_mean(G)
    cv = curves(theta, G, mean)
    dt = cv["Q"].dtype
    wg = jnp.asarray(G.w, dt)
    Pb = jnp.interp(u, jnp.asarray(G.u, dt), cv["Q"])             # [N,K]
    ps = [jnp.zeros((), dt), theta["pJP"], theta["pGL"]]
    bt = theta.get("beta")
    mus, Ps = [], []
    for k in range(3):
        wf = _inv_w(y, wg, cv["lc"][k], cv["rg"][k])
        zf = L10_3 + wf / LN10
        P = Pb * jnp.exp(ps[k])
        if bt is not None:
            P = P * jnp.exp(bt * (wf / LN10))[:, None]
        mus.append(zf[:, None] - P)
        Ps.append(P)
    return jnp.stack(mus), jnp.stack(Ps)


def lnC(theta, q, G=None, mean=None):
    """KR 최종 컷 ln 점수 (q = 목표 %, 격자 범위 0.1~40 안)."""
    import jax.numpy as jnp
    G = G or grids()
    mean = mean or default_mean(G)
    cv = curves(theta, G, mean)
    dt = cv["Q"].dtype
    return jnp.interp(jnp.log(jnp.asarray(q, dt) / 3.0), jnp.asarray(G.w, dt), cv["lc"][0])


# ---------- 모형 ----------
# v 벡터: [a, lnm, lnsig0, lnrho, beta, dJP..kGL, (lnell1, lnalp1, lnell2, lnalp2), beta1[mw], beta2[mu], logit eps, phi stick-breaking 8개]

def _v_layout(hyper, G=None):
    G = G or grids()
    names = ["a"] + CORE + ([] if hyper is not None else HYP)
    return names, len(names) + G.mw + G.mu + N_EP


def _unpack(v, hyper, G):
    """v -> theta dict (jnp)."""
    import jax.numpy as jnp
    names, n0 = _v_layout(hyper, G)
    th = {nm: v[i] for i, nm in enumerate(names)}
    if hyper is None:
        th["ell1"], th["alpha1"], th["ell2"], th["alpha2"] = (jnp.exp(th[h]) for h in HYP)
    else:
        for k in ("ell1", "alpha1", "ell2", "alpha2"):
            th[k] = jnp.asarray(hyper[k], v.dtype)
    nh = len(names)
    th["beta1"] = v[nh:nh + G.mw]
    th["beta2"] = v[nh + G.mw:nh + G.mw + G.mu]
    return th


def _ep_unpack(tail, obs):
    """v 꼬리 -> (eps, phi[4,3], 로그 사전 밀도 — 야코비안 포함). m1.model의 full 꼴과 같다."""
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from numpyro.distributions.transforms import StickBreakingTransform
    sbt = StickBreakingTransform()
    eps_v = jax.nn.sigmoid(tail[0])
    sb = tail[1:].reshape(4, 2)
    phi_v = sbt(sb)
    ea, eb = obs["eps_ab"]
    lp = (dist.Beta(ea, eb).log_prob(eps_v) + jnp.log(eps_v) + jnp.log1p(-eps_v)
          + dist.Dirichlet(obs["dirichlet"]).log_prob(phi_v).sum() + sbt.log_abs_det_jacobian(sb, phi_v).sum())
    return eps_v, phi_v, lp


def _logprior(v, prior, hyper, G, igp, obs):
    """사전 로그 밀도 (dtype을 따른다 — float64 MAP에서도 쓴다). -> (lp, eps, phi)"""
    import jax.numpy as jnp
    names, n0 = _v_layout(hyper, G)
    lp = M1._prior_logp64(v[:1 + len(CORE)], CORE, prior)
    nh = len(names)
    nb = G.mw + G.mu
    if hyper is None:
        lp = lp + _hyper_logp(v[1 + len(CORE):nh], igp)
    b = v[nh:nh + nb]
    lp = lp - 0.5 * (b ** 2).sum() - 0.5 * len(b) * math.log(2 * math.pi)
    eps_v, phi_v, lpe = _ep_unpack(v[nh + nb:], obs)
    return lp + lpe, eps_v, phi_v


def _center(prior, hyper, G, obs=None):
    """v의 사전 중심 (loc)과 척도 (sd) — pre가 없을 때의 기본 (eps·phi 자리는 사전 평균)."""
    names, n0 = _v_layout(hyper, G)
    if prior.get("kind") == "weak":
        loc = [M1.WEAK["a"][0]] + [M1.WEAK[n][0] for n in CORE]
        sd = [M1.WEAK["a"][1]] + [M1.WEAK[n][1] for n in CORE]
    else:
        ix = [prior["names"].index(n) for n in CORE]
        loc = [prior["a"]["mean"]] + list(np.asarray(prior["mean"])[ix])
        sd = [math.sqrt(prior["a"]["var"])] + list(np.sqrt(np.diag(np.asarray(prior["cov"]))[ix]))
    if hyper is None:
        c = hyper_center(G)
        loc += [math.log(c["ell1"]), math.log(c["alpha1"]), math.log(c["ell2"]), math.log(c["alpha2"])]
        sd += [0.5, 0.7, 0.5, 0.7]
    loc += [0.0] * (G.mw + G.mu)
    sd += [1.0] * (G.mw + G.mu)
    if obs is not None:
        from numpyro.distributions.transforms import StickBreakingTransform
        e0 = obs["eps_ab"][0] / (obs["eps_ab"][0] + obs["eps_ab"][1])
        d = np.asarray(obs["dirichlet"], float)
        sb0 = np.asarray(StickBreakingTransform().inv(d / d.sum(1, keepdims=True))).reshape(-1)
        loc += [float(np.log(e0 / (1 - e0)))] + [float(x) for x in sb0]
    else:
        loc += [0.0] * N_EP
    sd += [1.0] * N_EP
    return np.asarray(loc, float), np.asarray(sd, float)


def model(obs, prior, hyper=None, mean=None, pre=None, keep_ll=False, mb=(MB, MB)):
    """numpyro 모형. 머리 docstring 참조. pre = map_precond 결과 {"loc","chol"}. mb = (f1 기저 수, f2 기저 수)."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import constraints
    G = grids(*mb)
    mean = mean or default_mean(G)
    names, n0 = _v_layout(hyper, G)
    if pre is None:
        loc, sd = _center(prior, hyper, G, obs)
        pre = {"loc": loc, "chol": np.diag(sd)}
    z = numpyro.sample("z", dist.ImproperUniform(constraints.real_vector, (), (n0,)))
    v = jnp.asarray(pre["loc"], jnp.float32) + jnp.asarray(pre["chol"], jnp.float32) @ z
    igp = _igps(G)
    lp, eps_v, phi_v = _logprior(v, prior, hyper, G, igp, obs)
    numpyro.factor("prior", lp)
    th = _unpack(v, hyper, G)
    for nm in names:
        numpyro.deterministic(nm, th[nm])
    numpyro.deterministic("beta1", th["beta1"])
    numpyro.deterministic("beta2", th["beta2"])
    numpyro.deterministic("eps", eps_v)
    numpyro.deterministic("phi", phi_v)
    mu, P = surface(th, obs["y"], obs["u"], G, mean)
    OM.observe(obs, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]), keep_ll, eps=eps_v, phi=phi_v)
    for q in OM.Q_TARGETS:
        numpyro.deterministic(f"lnC_{q:g}", lnC(th, q, G, mean))


def map_precond(obs, prior, hyper=None, mean=None, start=None, iters=1500, temps=(0.003, 0.03, 0.3, 1.0), two_step=True, mb=(MB, MB)):
    """float64 MAP + 라플라스 (m1.map_precond(full=True)와 같은 방식 — eps·phi도 함께 푼다). -> dict(loc, chol, info)
    hyper=None(초모수 표집)일 때 two_step=True면: (1) 초모수를 사전 중앙값에 고정하고 나머지(a·core·beta·eps·phi)의 MAP + 라플라스를 구하고,
    (2) 초모수 4개는 사전 중앙값 / 대각 사전 sd(0.5, 0.7, 0.5, 0.7)로 이어 붙인다. 초모수를 함께 푼 MAP은 beta-크기 능선 때문에 헤세 행렬이 불정부호가 되었다
    (2026-10-29 첫 시간 재기에서 min_eig -150). 남은 상관은 NUTS의 dense 질량 행렬이 예열에서 배운다."""
    G = grids(*mb)
    if hyper is None and two_step:
        hc = hyper_center()
        n_core = 1 + len(CORE)
        n = _v_layout(None, G)[1]
        keep = list(range(n_core)) + list(range(n_core + 4, n))          # 초모수 자리를 뺀 v 위치
        sub = map_precond(obs, prior, hc, mean, None if start is None else np.asarray(start, float)[keep], iters, temps, two_step=False, mb=mb)
        hp = [n_core + i for i in range(4)]
        cov = np.zeros((n, n))
        cov[np.ix_(keep, keep)] = sub["cov"]
        sdh = [0.5, 0.7, 0.5, 0.7]
        for i, j in enumerate(hp):
            cov[j, j] = sdh[i] ** 2
        loc = np.zeros(n)
        loc[keep] = sub["loc"]
        loc[hp] = [math.log(hc["ell1"]), math.log(hc["alpha1"]), math.log(hc["ell2"]), math.log(hc["alpha2"])]
        return {"loc": loc, "chol": np.linalg.cholesky(cov + 1e-12 * np.eye(n)), "cov": cov,
                "info": dict(sub["info"], note="초모수는 사전 중앙값에 고정한 MAP + 사전 sd")}
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    mean = mean or default_mean(G)
    igp = _igps(G)
    with jax.enable_x64(True):
        o = dict(obs)
        for k in OM.NUMERIC:
            a = np.asarray(obs[k])
            o[k] = jnp.asarray(a, jnp.int64 if a.dtype.kind in "iu" else jnp.float64)

        def negpost(v, temp):
            th = _unpack(v, hyper, G)
            mu, P = surface(th, o["y"], o["u"], G, mean)
            lp, eps_v, phi_v = _logprior(v, prior, hyper, G, igp, o)
            ll = OM.row_loglik(o, mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"]), eps_v, phi_v)
            return -(temp * ll.sum() + lp)

        vg = jax.jit(jax.value_and_grad(negpost))
        hess = jax.jit(jax.hessian(negpost))
        x = np.asarray(start if start is not None else _center(prior, hyper, G, obs)[0], float)
        nit = 0
        for temp in temps:
            def f(xx, temp=temp):
                val, g = vg(jnp.asarray(xx), temp)
                val, g = float(val), np.asarray(g, float)
                return (val, g) if np.isfinite(val) and np.isfinite(g).all() else (1e12, np.zeros_like(g))
            res = minimize(f, x, jac=True, method="L-BFGS-B", options={"maxiter": iters, "maxfun": 3 * iters})
            x, nit = res.x, nit + res.nit
        f1 = lambda xx: float(vg(jnp.asarray(xx), 1.0)[0])

        def polish(x):
            """뉴턴 다듬기 (m1과 같다): L-BFGS는 조건수가 큰 사후에서 일찍 선다."""
            n_new = 0
            for _ in range(60):
                fv, g = vg(jnp.asarray(x), 1.0)
                fv, g = float(fv), np.asarray(g, float)
                Hn = np.asarray(hess(jnp.asarray(x), 1.0), float)
                Hn = 0.5 * (Hn + Hn.T)
                wv, Qv = np.linalg.eigh(Hn)
                step = Qv @ ((Qv.T @ g) / np.maximum(np.abs(wv), 1e-3))
                t, ok = 1.0, False
                while t > 1e-4:
                    fn = f1(x - t * step)
                    if np.isfinite(fn) and fn < fv - 1e-9:
                        ok = True
                        break
                    t *= 0.5
                if not ok:
                    break
                x = x - t * step
                n_new += 1
                if fv - fn < 1e-6:
                    break
            return x, n_new

        x, n_newton = polish(x)
        n_saddle = 0
        for _ in range(3):   # 안장점: Hessian 최소 고윳값 <= 0이면 그 방향으로 +-0.5 밀어 다시 푼다 (m1과 같다)
            Hs = np.asarray(hess(jnp.asarray(x), 1.0), float)
            ws, Qs = np.linalg.eigh(0.5 * (Hs + Hs.T))
            if ws.min() > 1e-6:
                break
            n_saddle += 1
            best = (f1(x), x)
            for sgn in (1.0, -1.0):
                r2_ = minimize(lambda xx: f(xx, temp=1.0), x + sgn * 0.5 * Qs[:, 0], jac=True, method="L-BFGS-B",
                               options={"maxiter": iters, "maxfun": 3 * iters})
                nit += r2_.nit
                xn, nn = polish(r2_.x)
                n_newton += nn
                fn_ = f1(xn)
                if np.isfinite(fn_) and fn_ < best[0]:
                    best = (fn_, xn)
            x = best[1]
        res.fun = f1(x)
        H = np.asarray(hess(jnp.asarray(x), 1.0), float)
    H = 0.5 * (H + H.T)
    w, Q = np.linalg.eigh(H)
    cov = (Q / np.maximum(w, 1e-6)) @ Q.T
    chol = np.linalg.cholesky(0.5 * (cov + cov.T) + 1e-12 * np.eye(len(x)))
    return {"loc": x, "chol": chol, "cov": 0.5 * (cov + cov.T),
            "info": {"fun": float(res.fun), "nit": nit, "ok": bool(res.success), "min_eig": float(w.min()), "max_eig": float(w.max()),
                     "n_newton": n_newton, "n_saddle": n_saddle}}


# ---------- 다중 시작 MAP ----------

def multistart_map(obs, prior, hyper=None, mean=None, n=8, seed=0, scatter=0.5, mb=(MB, MB)):
    """시작점 n개의 MAP (m1.multistart_map과 같은 꼴): 첫째는 기본 시작, 나머지는 그 해에서 서버 보조 모수(dJP..kGL)만 N(0, scatter^2)로 흩음.
    -> info.fun 오름차순 [pre dict] (fun = -log 사후 상대값이므로 작을수록 높은 봉우리)."""
    names, _ = _v_layout(hyper, grids(*mb))
    base = map_precond(obs, prior, hyper, mean, mb=mb)
    out = [base]
    rng = np.random.default_rng(seed)
    idx = [i for i, nm in enumerate(names) if nm in M1.OFF]
    for _ in range(n - 1):
        st = np.array(base["loc"], float)
        st[idx] += rng.normal(0, scatter, len(idx))
        out.append(map_precond(obs, prior, hyper, mean, start=st, mb=mb))
    return sorted(out, key=lambda p: p["info"]["fun"])


def map_init(n):
    """NUTS 시작: z는 0(= MAP), 나머지(eps·phi)는 init_to_median (m1.map_init)."""
    return M1.map_init(n)


# ---------- 실행 도구 ----------

def _draws_lnc(vdraws, hyper, mean, mb=(MB, MB)):
    """v 뽑기 [S, n] -> {q: lnC [S]} (float32 vmap)."""
    import jax
    import jax.numpy as jnp
    G = grids(*mb)
    mean = mean or default_mean(G)

    def one(v):
        th = _unpack(v, hyper, G)
        return jnp.stack([lnC(th, q, G, mean) for q in OM.Q_TARGETS])
    out = np.asarray(jax.jit(jax.vmap(one))(jnp.asarray(vdraws, jnp.float32)))
    return {q: out[:, i] for i, q in enumerate(OM.Q_TARGETS)}


def laplace_summary(pre, hyper, mean, n=4000, seed=0, mb=(MB, MB)):
    """라플라스 사후에서 뽑아 core 모수·lnC_{1,3,10}의 평균·sd를 낸다."""
    rng = np.random.default_rng(seed)
    v = pre["loc"] + rng.standard_normal((n, len(pre["loc"]))) @ pre["chol"].T
    lc = _draws_lnc(v, hyper, mean, mb)
    names, _ = _v_layout(hyper, grids(*mb))
    out = {nm: {"mean": float(v[:, i].mean()), "sd": float(v[:, i].std())} for i, nm in enumerate(names)}
    for q in (1.0, 3.0, 10.0):
        out[f"lnC_{q:g}"] = {"mean": float(lc[q].mean()), "median": float(np.median(lc[q])), "sd": float(lc[q].std())}
    return out


def bench_case(work, case, season=None, chains=4, warmup=500, samples=500, K=16, target_accept=0.9, probe=False, mean_kind="temp", multistart=True):
    import jax
    import obs_table as O
    OM.setup_host(chains)
    err = OM.load_err()
    season = season or ("S30" if case in ("a", "b") else "S30")
    head, rows = O.read_cell(os.path.join(work, "obs7", "full", f"{season}.jsonl.gz"))
    H = head["H"]
    obs = OM.prepare(rows, H, season, err, K=K, stage1=True)
    jobs = OM.to_jax(obs)
    hyper = None if case == "a" else hyper_center()
    prior = {"kind": "weak"}
    mean = default_mean()
    mean_note = "임시 평균 (rbar = ln 0.3 + w, lbar = M1 kuma eta=0)"
    if mean_kind == "m1":
        # 진단용: 같은 시즌 자료에 M1 kuma를 MAP으로 맞춘 평균 곡선 (시간 재기 원인 가리기용 — 실제 쓰임이 아니다)
        p1 = M1.map_precond(jobs, {"kind": "weak"}, "kuma")
        nm1 = M1.theta_names("kuma")
        th1 = M1.theta_from_vec(list(p1["loc"][1:]), nm1, p1["loc"][0], "kuma")
        mean = mean_from_m1(th1, "kuma")
        mean_note = "진단용: 같은 시즌에 M1 kuma MAP을 맞춘 평균 곡선 (r0, r1, eta)"
        res_m1 = {"r0": float(th1["r0"]), "r1": float(th1["r1"]), "eta": [float(e) for e in th1["eta"]]}
    res = {"case": case, "season": season, "N": int(obs["N"]), "n_rows": int(obs["n"]), "counts": obs["counts"], "K": K,
           "hyper": "표집" if hyper is None else hyper, "mean": mean_note}
    if mean_kind == "m1":
        res["m1_map_theta"] = res_m1
    t0 = time.time()
    pre = map_precond(jobs, prior, hyper, mean)
    res["map_secs_incl_compile"] = round(time.time() - t0, 1)
    t0 = time.time()
    pre2 = map_precond(jobs, prior, hyper, mean)       # 같은 프로세스에서 두 번째 (jit 캐시가 안 맞으면 컴파일이 또 든다)
    res["map_secs_second_call"] = round(time.time() - t0, 1)
    if multistart:
        t0 = time.time()
        modes = multistart_map(jobs, prior, hyper, mean)
        res["multistart_secs"] = round(time.time() - t0, 1)
        res["modes_fun"] = [round(p["info"]["fun"], 3) for p in modes]
        pre = modes[0]
    res["map_info"] = pre["info"]
    lap = laplace_summary(pre, hyper, mean)
    res["laplace"] = lap
    print("MAP", res["map_secs_incl_compile"], res["map_secs_second_call"], pre["info"], flush=True)
    kw = {"obs": jobs, "prior": prior, "hyper": hyper, "mean": mean, "pre": pre}
    init = map_init(len(pre["loc"]))
    if probe:
        _, pi = OM.run_nuts(model, kw, seed=0, chains=chains, warmup=100, samples=50, target_accept=target_accept, init=init)
        print("PROBE", pi, flush=True)
        res["probe"] = pi
        return res
    # 컴파일 시간 추정: 아주 짧은 실행
    _, ci = OM.run_nuts(model, kw, seed=5, chains=chains, warmup=8, samples=4, target_accept=target_accept, init=init)
    res["compile_est_secs"] = ci["secs"]
    print("COMPILE_EST", ci["secs"], flush=True)
    sm, info = OM.run_nuts(model, kw, seed=0, chains=chains, warmup=warmup, samples=samples, target_accept=target_accept, init=init)
    res["nuts"] = info
    res["nuts_secs_excl_compile_est"] = round(info["secs"] - ci["secs"], 1)
    th_names = ["a"] + CORE + ([] if hyper is not None else HYP)
    lc_names = [f"lnC_{q:g}" for q in (1, 3, 10)]
    dg = OM.diagnose(sm, th_names + lc_names, divergences=info["divergences"])
    dgb = OM.diagnose(sm, ["beta1", "beta2"], divergences=info["divergences"])
    res["diag"] = {"rhat_max_theta_lnC": dg["rhat_max"], "ess_min_theta_lnC": dg["ess_min"],
                   "rhat_max_beta": dgb["rhat_max"], "ess_min_beta": dgb["ess_min"],
                   "conv_theta_lnC": dg["conv"], "per": dg["per"]}
    res["nuts_post"] = {nm: {"mean": float(sm[nm].mean()), "sd": float(sm[nm].std())} for nm in th_names}
    for nm in lc_names:
        v = sm[nm].ravel()
        res["nuts_post"][nm] = {"mean": float(v.mean()), "median": float(np.median(v)), "sd": float(v.std())}
    res["laplace_vs_nuts"] = {nm: {"median_diff_pct": round(100 * (math.exp(lap[nm]["median"] - float(np.median(sm[nm]))) - 1), 3),
                                   "sd_ratio": round(lap[nm]["sd"] / float(sm[nm].std()), 3)} for nm in lc_names}
    res["laplace_vs_nuts"]["core_sd_ratio_range"] = [round(min(lap[n]["sd"] / sm[n].std() for n in th_names), 3),
                                                      round(max(lap[n]["sd"] / sm[n].std() for n in th_names), 3)]
    res["total_secs"] = round(res["map_secs_incl_compile"] + info["secs"], 1)
    return res


# ---------- 시험 ----------

def test_basis():
    G = grids()
    out = {}
    have = True
    rng = np.random.default_rng(0)
    for tag, Phi, sl, lo, hi, W in (("w", G.Phi_w, G.sl_w, W_LO, W_HI, G.width_w), ("u", G.Phi_u, G.sl_u, U_LO, 1.0, G.width_u)):
        x = G.w if tag == "w" else G.u
        x0 = 0.5 * (lo + hi)
        mid = np.abs(x - x0) < 0.3 * (hi - lo) / 2 * 2      # 중앙 30% 폭
        for fr in (0.1, 0.5):
            ell, al = fr * W, 1.0
            S = matern52_spec(sl, ell, al)
            cov_h = (Phi * S[None, :]) @ Phi.T
            cov_e = matern52_cov(x[:, None] - x[None, :], ell, al)
            mm = mid[:, None] & mid[None, :]
            err = np.abs(cov_h - cov_e)[mm]
            # 표본 공분산 (20000 함수)
            b = rng.standard_normal((20000, MB))
            F = (b * np.sqrt(S)[None, :]) @ Phi.T
            ii = np.where(mid)[0]
            i0 = ii[len(ii) // 2]
            lag = [int(i0 + d) for d in (0, 3, 8, 15) if i0 + d < len(x)]
            samp = [float(np.mean(F[:, i0] * F[:, j])) for j in lag]
            exact = [float(cov_e[i0, j]) for j in lag]
            hs = [float(cov_h[i0, j]) for j in lag]
            out[f"{tag}_ell{fr}W"] = {"ell": ell, "max_abs_err_cov_center": float(err.max()),
                                      "var_center_hsgp_vs_exact": [float(np.diag(cov_h)[ii].mean()), 1.0],
                                      "lags_x": [float(x[j] - x[i0]) for j in lag], "sample": samp, "hsgp": hs, "exact": exact}
    try:
        from numpyro.contrib.hsgp.laplacian import eigenfunctions
        from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern
        import jax.numpy as jnp
        for tag, x, lo, hi, W, Phi, sl in (("w", G.w, W_LO, W_HI, G.width_w, G.Phi_w, G.sl_w), ("u", G.u, U_LO, 1.0, G.width_u, G.Phi_u, G.sl_u)):
            x0, S_, = 0.5 * (lo + hi), 0.5 * (hi - lo)
            L = CB * S_
            ph = np.asarray(eigenfunctions(jnp.asarray(x - x0), L, MB))
            ell, al = 0.3 * W, 0.7
            sd = np.asarray(diag_spectral_density_matern(2.5, al ** 2, ell, L, MB, 1))   # numpyro의 alpha는 분산 자리(곱하는 값)라 al**2
            ours = np.sqrt(matern52_spec(sl, ell, al))
            out[f"numpyro_{tag}"] = {"eigenfunction_max_abs_diff": float(np.abs(ph - Phi).max()),
                                     "sqrt_spec_max_rel_diff": float(np.max(np.abs(np.sqrt(sd).ravel() - ours) / ours))}
    except Exception as e:  # noqa
        out["numpyro_hsgp_cmp"] = f"비교 실패: {type(e).__name__}: {e}"
    return out


def truth_theta(a_mid, seed=11):
    """합성 시험의 참값: 작은 f1(길이 0.35 W, 크기 0.08)·f2(길이 0.3 W, 크기 0.12)를 HSGP 계수로 정해진 씨앗에서 뽑는다."""
    G = grids()
    rng = np.random.default_rng(seed)
    th = {"a": a_mid, "lnm": math.log(0.2), "lnsig0": math.log(0.01), "lnrho": math.log(0.13), "beta": 0.0, "dJP": math.log(0.8),
          "dGL": math.log(0.85), "pJP": 0.1, "pGL": -0.1, "kJP": 0.05, "kGL": -0.05,
          "ell1": 0.35 * G.width_w, "alpha1": 0.08, "ell2": 0.3 * G.width_u, "alpha2": 0.12,
          "beta1": rng.standard_normal(MB), "beta2": rng.standard_normal(MB)}
    return th


def synth_rows(rows, H, err, th, seed=1):
    """check_obs.synth_rows를 M2 곡면으로 바꾼 것: 실제 줄 배치(점수·종류·시각 정보·서버 라벨·사이트)에 th로 %를 뽑는다."""
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)
    sel = OM.select_rows(rows, H, stage1=True)
    keep, tshot, srv = [], [], []
    dm = err["1e_server"]["unknown_row_site_dirichlet"]
    phi_true = {s: np.array([dm[s]["mean"][k] for k in OM.SERVERS]) for s in dm}
    phi_true = {s: v / v.sum() for s, v in phi_true.items()}
    for r in sel:
        t, w, why = OM.nodes_detail(r, H, err, None, 64)
        if t is None:
            continue
        tshot.append(float(rng.choice(t, p=w / w.sum())))
        site = r["site"] if r["site"] in OM.SITES else "*"
        if r["server"] in OM.SERVERS:
            lab = OM.SERVERS.index(r["server"])
            el = OM._elab(r, err)
            if rng.random() < el:
                pr = phi_true[site].copy()
                pr[lab] = 0
                srv.append(int(rng.choice(3, p=pr / pr.sum())))
            else:
                srv.append(lab)
        else:
            srv.append(int(rng.choice(3, p=phi_true[site])))
        keep.append(r)
    n = len(keep)
    y = np.log([r["score"] for r in keep])
    u = np.clip(np.array(tshot) / H, 0.02, 1.0)
    thj = {k: (jnp.asarray(v, jnp.float32)) for k, v in th.items()}
    mu, P = surface(thj, jnp.asarray(y, jnp.float32), jnp.asarray(u[:, None], jnp.float32))
    mu, P = np.asarray(mu)[:, :, 0], np.asarray(P)[:, :, 0]
    s = np.array(srv)
    mu_row, P_row = mu[s, np.arange(n)], P[s, np.arange(n)]
    sd = np.sqrt(math.exp(th["lnsig0"]) ** 2 + (math.exp(th["lnrho"]) * P_row) ** 2)
    z = mu_row + sd * rng.standard_normal(n)
    cont = rng.random(n) < 0.015
    z = np.where(cont, z + rng.laplace(0, 0.34, n), z)
    x100 = 10 ** z * 100
    pct = np.where(rng.random(n) < 0.5, np.round(x100), np.floor(x100)) / 100
    out, dropped = [], 0
    for r, p in zip(keep, pct):
        if p > 100.0 or p < 0.01:
            dropped += 1
            continue
        x = dict(r)
        x["pct"] = float(p)
        out.append(x)
    return out, {"n_rows": len(out), "dropped_out_of_range": dropped, "n_contam": int(cont.sum()),
                 "server_counts": [int((s == k).sum()) for k in range(3)]}


def test_synth(work, season="S30", chains=4, warmup=500, samples=500, free_hyper=False):
    import jax.numpy as jnp
    import obs_table as O
    OM.setup_host(chains)
    err = OM.load_err()
    head, rows = O.read_cell(os.path.join(work, "obs7", "full", f"{season}.jsonl.gz"))
    H = head["H"]
    # a: w = 0 (3%)에 해당하는 ln 점수 = 3% 근처 줄의 ln 점수 평균
    near = [math.log(r["score"]) for r in rows if r["pct"] is not None and r["score"] and 2.5 <= r["pct"] <= 3.5 and r["kind"] in OM.ABC]
    a_mid = float(np.mean(near)) if near else math.log(100.0)
    th = truth_theta(a_mid)
    srows, tinfo = synth_rows(rows, H, err, th)
    obs = OM.prepare(srows, H, season, err, K=16, stage1=True)
    jobs = OM.to_jax(obs)
    hyper = None if free_hyper else {k: th[k] for k in ("ell1", "alpha1", "ell2", "alpha2")}
    mean = default_mean()
    t0 = time.time()
    pre = map_precond(jobs, {"kind": "weak"}, hyper, mean)
    t_map = time.time() - t0
    kw = {"obs": jobs, "prior": {"kind": "weak"}, "hyper": hyper, "mean": mean, "pre": pre}
    sm, info = OM.run_nuts(model, kw, seed=0, chains=chains, warmup=warmup, samples=samples)
    G = grids()
    names = ["a"] + CORE + ([] if hyper is not None else HYP)
    truth_v = {n: th[n] for n in ["a"] + CORE}
    if hyper is None:
        truth_v.update({"lnell1": math.log(th["ell1"]), "lnalp1": math.log(th["alpha1"]),
                        "lnell2": math.log(th["ell2"]), "lnalp2": math.log(th["alpha2"])})
    rows_out, cover = {}, 0
    lc_names = [f"lnC_{q:g}" for q in (1, 3, 10)]
    thj = {k: jnp.asarray(v, jnp.float32) for k, v in th.items()}
    for q in (1, 3, 10):
        truth_v[f"lnC_{q:g}"] = float(lnC(thj, float(q), G, mean))
    for nm in names + lc_names:
        v = sm[nm].ravel()
        lo, hi = np.quantile(v, [0.025, 0.975])
        ins = bool(lo <= truth_v[nm] <= hi)
        cover += ins
        rows_out[nm] = {"truth": truth_v[nm], "mean": float(v.mean()), "sd": float(v.std()), "lo95": float(lo), "hi95": float(hi), "inside": ins}
    for nm in lc_names:
        rows_out[nm]["rel_err_C_pct"] = round(100 * (math.exp(rows_out[nm]["mean"] - rows_out[nm]["truth"]) - 1), 3)
    # f1·f2 되찾기: 사후 뽑기에서 곡선을 만들어 참값 곡선이 95% 안에 드는 격자 점 비율 (자료가 있는 범위)
    b1 = sm["beta1"].reshape(-1, MB)
    b2 = sm["beta2"].reshape(-1, MB)
    idx = np.linspace(0, len(b1) - 1, 400).astype(int)
    al = {k: np.asarray(sm[k]).ravel() if hyper is None else None for k in ()}
    def curve(Phi, sl, ell, alpha, b):
        S = np.sqrt(matern52_spec(sl, ell, alpha))
        f = (b * S[None, :]) @ Phi.T
        return f
    if hyper is None:
        e1, a1, e2, a2 = (np.exp(sm[h]).ravel()[idx] for h in HYP)
        F1 = np.stack([curve(G.Phi_w, G.sl_w, e1[i], a1[i], b1[idx[i]][None])[0] for i in range(len(idx))])
        F2 = np.stack([curve(G.Phi_u, G.sl_u, e2[i], a2[i], b2[idx[i]][None])[0] for i in range(len(idx))])
    else:
        F1 = curve(G.Phi_w, G.sl_w, th["ell1"], th["alpha1"], b1[idx])
        F2 = curve(G.Phi_u, G.sl_u, th["ell2"], th["alpha2"], b2[idx])
    F2 = F2 - F2.mean(1, keepdims=True)
    T1 = curve(G.Phi_w, G.sl_w, th["ell1"], th["alpha1"], th["beta1"][None])[0]
    T2 = curve(G.Phi_u, G.sl_u, th["ell2"], th["alpha2"], th["beta2"][None])[0]
    T2 = T2 - T2.mean()
    # 자료 범위: 줄 y에서 나온 w (KR 참값 곡선 기준) 5~95%
    cv = curves(thj, G, mean)
    wf = np.asarray(_inv_w(jnp.asarray(np.log([r["score"] for r in srows]), jnp.float32), jnp.asarray(G.w, jnp.float32), cv["lc"][0], cv["rg"][0]))
    wlo, whi = np.quantile(wf, [0.05, 0.95])
    m1_ = (G.w >= wlo) & (G.w <= whi)
    uu = np.array([r.get("h", 0) for r in srows], float) / H
    ulo, uhi = np.quantile(uu[np.isfinite(uu)], [0.05, 0.95])
    m2_ = (G.u >= max(ulo, U_LO)) & (G.u <= uhi)
    q = lambda F: (np.quantile(F, 0.025, axis=0), np.quantile(F, 0.975, axis=0))
    l1, h1 = q(F1)
    l2, h2 = q(F2)
    fcov = {"f1_cover_frac_data_range": float(((T1 >= l1) & (T1 <= h1))[m1_].mean()), "f1_w_range": [float(wlo), float(whi)],
            "f1_rmse": float(np.sqrt(((F1.mean(0) - T1)[m1_] ** 2).mean())), "f1_truth_rms": float(np.sqrt((T1[m1_] ** 2).mean())),
            "f2_cover_frac_data_range": float(((T2 >= l2) & (T2 <= h2))[m2_].mean()), "f2_u_range": [float(ulo), float(uhi)],
            "f2_rmse": float(np.sqrt(((F2.mean(0) - T2)[m2_] ** 2).mean())), "f2_truth_rms": float(np.sqrt((T2[m2_] ** 2).mean()))}
    dg = OM.diagnose(sm, names + lc_names, divergences=info["divergences"])
    return {"season": season, "free_hyper": free_hyper, "synth": tinfo, "N": obs["N"], "n": obs["n"], "map_secs": round(t_map, 1),
            "nuts": info, "param_inside": f"{cover}/{len(names) + 3}", "params": rows_out, "f_recovery": fcov,
            "rhat_max": dg["rhat_max"], "ess_min": dg["ess_min"], "conv": dg["conv"]}


def compare_multistart(work, seasons, chains=4, warmup=500, samples=500, K=16, n=8):
    """경우 b(초모수 고정, 약한 사전, 임시 평균) 단일 MAP 대 시작점 n개 MAP: 봉우리별 log 사후, 사슬별 dJP·dGL 평균, 진단, 시간.
    NUTS는 둘 다 MAP(z=0)에서 시작한다 (차이는 MAP뿐)."""
    import obs_table as O
    OM.setup_host(chains)
    err = OM.load_err()
    hyper, prior, mean = hyper_center(), {"kind": "weak"}, default_mean()
    out = {"note": "임시 평균 (rbar = ln 0.3 + w, lbar = M1 kuma eta=0), 경우 b: 초모수 고정(사전 중앙값)·약한 사전. 2026-10-29",
           "chains": chains, "warmup": warmup, "samples": samples, "n_starts": n, "seasons": {}}
    for season in seasons:
        head, rows = O.read_cell(os.path.join(work, "obs7", "full", f"{season}.jsonl.gz"))
        obs = OM.prepare(rows, head["H"], season, err, K=K, stage1=True)
        jobs = OM.to_jax(obs)
        names, _ = _v_layout(hyper)
        off = [i for i, nm in enumerate(names) if nm in M1.OFF]
        r = {"N": int(obs["N"]), "n_rows": int(obs["n"])}
        for tag in ("single", "multi"):
            t0 = time.time()
            if tag == "single":
                modes = [map_precond(jobs, prior, hyper, mean)]
            else:
                modes = multistart_map(jobs, prior, hyper, mean, n=n)
            tmap = time.time() - t0
            pre = modes[0]
            init = map_init(len(pre["loc"]))
            kw = {"obs": jobs, "prior": prior, "hyper": hyper, "mean": mean, "pre": pre}
            sm, info = OM.run_nuts(model, kw, seed=0, chains=chains, warmup=warmup, samples=samples, init=init)
            watch = ["a"] + CORE + [f"lnC_{q:g}" for q in (1, 3, 10)]
            dg = OM.diagnose(sm, watch, divergences=info["divergences"])
            r[tag] = {
                "map_secs": round(tmap, 1), "nuts_secs": info["secs"],
                "modes": [{"logpost": round(-p["info"]["fun"], 2), "server": dict(zip([names[i] for i in off], [round(float(p["loc"][i]), 3) for i in off])),
                          "saddle": p["info"].get("n_saddle"), "ok": p["info"]["ok"]} for p in modes],
                "chain_mean": {nm: [round(float(sm[nm][c].mean()), 3) for c in range(chains)] for nm in ("dJP", "dGL")},
                "rhat_max": round(dg["rhat_max"], 3), "ess_min": round(dg["ess_min"], 1), "divergences": info["divergences"],
                "tree_cap_frac": info["tree_cap_frac"], "depth_dist": info["depth_dist"],
                "worst": sorted(((k, round(v["rhat"], 3)) for k, v in dg["per"].items()), key=lambda t: -t[1])[:3],
                "lnC_3_mean": round(float(sm["lnC_3"].mean()), 4)}
            print(season, tag, json.dumps(r[tag], ensure_ascii=False, default=float), flush=True)
        out["seasons"][season] = r
        d = os.path.join(work, "m2")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "multistart_partial.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    return out


# ---------- M1 같음 시험 (2026-10-29) ----------

def test_m1match(m1_stage1, season="S30", n_y=40):
    """M2 곡면에서 f1 = f2 = 0, 평균 = M1 최종 꼴(kuma + beta + r2)의 그 시즌 1단계 사후 평균 theta, lnm·beta·서버 모수 같게 두면
    m1.surface(x_term)·m1.lnC와 같아야 한다. 점수는 M1 컷 곡선 0.05~60% 구간, u는 0.02~1(0.99·0.999·0.9999·1 포함)."""
    import jax.numpy as jnp
    with open(m1_stage1, encoding="utf-8") as f:
        d = json.load(f)
    names = d["names"]
    m = dict(zip(names, d["seasons"][season]["mean"]))
    nm1 = M1.theta_names("kuma", True, True)
    th1 = M1.theta_from_vec([jnp.float32(m[n]) for n in nm1], nm1, jnp.float32(m["a"]), "kuma")
    G = grids()
    mean = mean_from_m1({"r0": m["r0"], "r1": m["r1"], "r2": m["r2"], "eta": [m["eta0"], m["eta1"]]}, "kuma", G)
    th2 = {k: jnp.float32(m[k]) for k in ["a"] + CORE}
    th2.update({"ell1": 1.0, "alpha1": 0.2, "ell2": 0.3, "alpha2": 0.2,
                "beta1": jnp.zeros(G.mw, jnp.float32), "beta2": jnp.zeros(G.mu, jnp.float32)})
    qs = np.exp(np.linspace(np.log(0.05), np.log(60.0), n_y))
    y = np.asarray(M1.lnC(th1, jnp.asarray(qs, jnp.float32)), np.float32)
    us = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999, 1.0], np.float32)
    u = np.broadcast_to(us[None, :], (n_y, len(us))).copy()
    mu1, P1 = (np.asarray(x) for x in M1.surface(th1, jnp.asarray(y), jnp.asarray(u), "kuma", True))
    mu2, P2 = (np.asarray(x) for x in surface(th2, jnp.asarray(y), jnp.asarray(u), G, mean))
    inr = (qs >= 0.1) & (qs <= 40.0)
    qg = np.exp(np.linspace(np.log(0.1), np.log(40.0), 60))
    c1 = np.asarray(M1.lnC(th1, jnp.asarray(qg, jnp.float32)))
    c2 = np.asarray(lnC(th2, qg, G, mean))
    dmu, dP = np.abs(mu2 - mu1), np.abs(P2 - P1)
    # 정확한 적분(scipy quad)과의 격자 오차: 두 곡면 모두 사다리꼴이라 컷 기울기가 가파른 10% 위쪽(S42 r2 = 0.39)에서 같이 벗어난다
    from scipy import integrate
    ex = np.array([m["a"] - integrate.quad(lambda v: math.exp(m["r0"] + m["r1"] * v + m["r2"] * v * v), 0, math.log(q / 3.0))[0] for q in qg])
    le10 = qg <= 10.0
    out = {"season": season, "m1_stage1": os.path.basename(m1_stage1), "theta_M1": {k: float(v) for k, v in m.items() if k in ["a"] + nm1},
           "max_abs_dlnC_q0.1to10": float(np.abs(c2 - c1)[le10].max()), "max_abs_dlnC_q0.1to40": float(np.abs(c2 - c1).max()),
           "grid_err_vs_exact_quad_q0.1to10": {"M1": float(np.abs(c1 - ex)[le10].max()), "M2": float(np.abs(c2 - ex)[le10].max())},
           "grid_err_vs_exact_quad_q0.1to40": {"M1": float(np.abs(c1 - ex).max()), "M2": float(np.abs(c2 - ex).max())}, "max_abs_dmu_log10_in_range": float(dmu[:, inr].max()),
           "max_abs_dmu_log10_all": float(dmu.max()), "max_abs_dP_log10": float(dP.max()),
           "max_abs_dP_by_u": {f"{float(a):g}": float(dP[:, i].max()) for i, a in enumerate(us)},
           "max_abs_dmu_by_server": [float(dmu[k, inr].max()) for k in range(3)],
           "P_scale_max": float(P1.max()),
           "note": "f1 = f2 = 0, 평균 = M1 theta의 r0 + r1 w + r2 w^2 와 kuma 침수; 서버 모수·beta 같음. 허용: mu·P 차 2e-3 (log10 단위), lnC 차 1.5e-3 (q 0.1~10%; 그 위는 둘 다 사다리꼴 오차)"}
    out["pass"] = bool(out["max_abs_dlnC_q0.1to10"] < 1.5e-3 and out["max_abs_dmu_log10_in_range"] < 2e-3 and out["max_abs_dP_log10"] < 2e-3)
    return out


# ---------- 1단계 (조정 시즌, M1 최종 꼴 평균) ----------
#   python3 -I tools/solo_raid/lab7/m2.py stage1 --obs W/obs7/full --seasons S22,S26,... --m1-stage1 r7/wp3b/stage1_M1_kuma_b_r2_D1.json
#           --out r7/wp3b_m2/stage1_M2_tune_D1.json --ll-dir W/m2/ll [--resume]

HYP_NAMES_HIER = ["a"] + CORE            # 2단계 hier에 넘길 모수 (초모수는 따로: 2단계에서 창 안 중앙을 그대로 쓴다)
SPOT_Q = (1.0, 3.0, 10.0)


def m1_mean_for(m1_json, season, G=None):
    """그 시즌 M1 1단계 사후 평균 theta_hat에서 평균 함수 (m1 kuma + beta + r2)."""
    m = dict(zip(m1_json["names"], m1_json["seasons"][season]["mean"]))
    assert m1_json["form"] == "kuma" and m1_json["x_term"] and m1_json["r2"], "M1 최종 꼴(kuma + beta + r2)이 아님"
    return mean_from_m1({"r0": m["r0"], "r1": m["r1"], "r2": m["r2"], "eta": [m["eta0"], m["eta1"]]}, "kuma", G), m


def theta_surface_fn(G, mean):
    """OM.row_posteriors용: draw(dict) -> (mu, P, sigma0, rho). draw에는 a·CORE·초모수(lnell1..)·beta1·beta2가 있다."""
    def fn(draw, jobs):
        import jax.numpy as jnp
        th = {k: draw[k] for k in ["a"] + CORE}
        th["ell1"], th["alpha1"], th["ell2"], th["alpha2"] = (jnp.exp(draw[h]) for h in HYP)
        th["beta1"], th["beta2"] = draw["beta1"], draw["beta2"]
        mu, P = surface(th, jobs["y"], jobs["u"], G, mean)
        return mu, P, jnp.exp(th["lnsig0"]), jnp.exp(th["lnrho"])
    return fn


def fit_one(rows, H, season, err, mean, mb, K=16, delay_key=None, seed=0, warmup=1000, samples=1000, retry_warmup=1500, max_tree_depth=10,
            prior=None, hyper=None, stage1=True, keep_ll=True):
    """한 칸 맞춤: prepare -> 시작점 8개 MAP(최고 봉우리가 둘 미만이면 8개 더) -> fit_with_retry. m1.fit_one과 같은 설정.
    1단계 (기본): prior = 약한 사전, hyper = None (초모수 표집), stage1=True (끝난 시즌 전체 표 — 종료 뒤 화면 줄 포함).
    칸 예측 (cmd_predict): prior = hier.prior_for 꼴, hyper = 고정 초모수 dict, stage1=False, keep_ll=False."""
    obs = OM.prepare(rows, H, season, err, K=K, delay_key=delay_key, stage1=stage1)
    jobs = OM.to_jax(obs)
    prior = prior or {"kind": "weak"}
    G = grids(*mb)
    names, _ = _v_layout(hyper, G)
    t0 = time.time()
    modes = multistart_map(jobs, prior, hyper, mean, mb=mb)
    rep = sum(1 for p in modes if p["info"]["fun"] - modes[0]["info"]["fun"] < 0.5)
    if rep < 2:
        modes = sorted(modes + multistart_map(jobs, prior, hyper, mean, seed=1, mb=mb), key=lambda p: p["info"]["fun"])
        rep = sum(1 for p in modes if p["info"]["fun"] - modes[0]["info"]["fun"] < 0.5)
    pre = modes[0]
    idx_off = [i for i, nm in enumerate(names) if nm in M1.OFF]
    pre_modes = [{"fun": round(p["info"]["fun"], 3), "server": [round(float(p["loc"][i]), 3) for i in idx_off]} for p in modes]
    map_secs = time.time() - t0
    watch = ["a"] + [n for n in CORE if n not in M1.OFF] + [f"lnC_{q:g}" for q in OM.Q_TARGETS]
    init = map_init(len(pre["loc"]))
    kw = {"obs": jobs, "prior": prior, "hyper": hyper, "mean": mean, "pre": pre, "keep_ll": keep_ll, "mb": mb}
    sm, info, dg = OM.fit_with_retry(model, kw, names=watch, seed=seed, init=init, max_tree_depth=max_tree_depth,
                                     warmup=warmup, samples=samples, retry_warmup=retry_warmup)
    info["map_secs"] = round(map_secs, 1)
    return {"samples": sm, "info": info, "diag": dg, "obs": obs, "jobs": jobs, "pre": pre, "modes": pre_modes, "n_starts": len(modes),
            "n_best_repeat": rep, "map": pre["info"], "mb": mb, "watch": watch}


def basis_check(sm, mb, G=None):
    """Riutort-Mayol: 사후 길이 중앙 대 그 기저의 최소 길이. -> dict(f1, f2: {ell_med, ell_q, min_len, short, ell_over_min}), short 목록."""
    G = G or grids(*mb)
    out = {}
    for tag, key, W, M in (("f1", "lnell1", G.width_w, mb[0]), ("f2", "lnell2", G.width_u, mb[1])):
        ell = np.exp(sm[key].reshape(-1))
        med = float(np.median(ell))
        ml = min_len(W, M)
        out[tag] = {"M": M, "ell_median": round(med, 4), "ell_q10_q90": [round(float(x), 4) for x in np.quantile(ell, [0.1, 0.9])],
                    "min_len": round(ml, 4), "short": bool(med < ml), "ell_over_min": round(med / ml, 3)}
    return out


def a_residuals(sm, obs, G, mean, nd=100):
    """4.3 점검 일부: 한섭(KR 라벨) A 줄의 로그10 % 잔차 = 구간 중점 - 사후 평균 곡면 값 (정답 표 안 씀). 전체와 % 구간별 평균·sd."""
    import jax
    import jax.numpy as jnp
    meta = obs["meta"]
    sel = [i for i, m_ in enumerate(meta) if m_["tg"] == "A" and m_["server"] == "KR"]
    if not sel:
        return {"n": 0}
    pos = np.asarray(obs["pos"])[sel]
    y = jnp.asarray(np.asarray(obs["y"])[pos], jnp.float32)
    u = jnp.asarray(np.asarray(obs["u"])[pos][:, :1], jnp.float32)
    zc = 0.5 * (np.asarray(obs["zlo"])[pos] + np.asarray(obs["zhi"])[pos])
    flat = {k: np.asarray(sm[k]).reshape((-1,) + np.asarray(sm[k]).shape[2:]) for k in ["a"] + CORE + HYP + ["beta1", "beta2"]}
    idx = np.unique(np.linspace(0, len(flat["a"]) - 1, nd).astype(int))
    dr = {k: jnp.asarray(v[idx], jnp.float32) for k, v in flat.items()}

    def one(d):
        th = {k: d[k] for k in ["a"] + CORE}
        th["ell1"], th["alpha1"], th["ell2"], th["alpha2"] = (jnp.exp(d[h]) for h in HYP)
        th["beta1"], th["beta2"] = d["beta1"], d["beta2"]
        mu, P = surface(th, y, u, G, mean)
        return mu[0, :, 0]
    mu = np.asarray(jax.vmap(one)(dr)).mean(0)
    res = zc - mu
    pct = 10 ** zc
    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 100)]
    out = {"n": int(len(res)), "mean": round(float(res.mean()), 4), "sd": round(float(res.std()), 4),
           "mad_sd": round(float(1.4826 * np.median(np.abs(res - np.median(res)))), 4), "median": round(float(np.median(res)), 4), "by_pct": {}}
    for lo, hi in bins:
        mk = (pct >= lo) & (pct < hi)
        if mk.sum() >= 3:
            out["by_pct"][f"{lo}-{hi}"] = {"n": int(mk.sum()), "mean": round(float(res[mk].mean()), 4), "sd": round(float(res[mk].std()), 4)}
    return out


def _season_record(r, season, mean_theta, m1_json, G, mean, ll_dir, tag):
    import arviz as az
    sm, obs, info, dg, jobs = r["samples"], r["obs"], r["info"], r["diag"], r["jobs"]
    mb = r["mb"]
    th_names = HYP_NAMES_HIER
    allv = th_names + HYP
    v_h = np.stack([sm[k].reshape(-1) for k in th_names], 1)
    v_all = np.stack([sm[k].reshape(-1) for k in allv], 1)
    pos = obs["pos"]
    ll = sm["ll"][:, :, pos]
    idata = az.from_dict(posterior={k: sm[k] for k in th_names}, log_likelihood={"y": ll})
    lo = az.loo(idata, pointwise=True)
    loo_i = np.asarray(lo.loo_i).reshape(-1)
    pk = np.asarray(lo.pareto_k).reshape(-1)
    keys = np.array([m_["key"] for m_ in obs["meta"]])
    os.makedirs(ll_dir, exist_ok=True)
    np.savez(os.path.join(ll_dir, f"{tag}_{season}.npz"), loo_i=loo_i, keys=keys, pareto_k=pk)
    rp = OM.row_posteriors(jobs, {k: x for k, x in sm.items() if k != "ll"}, theta_surface_fn(G, mean))
    fl = OM.flags(obs["meta"], rp)
    lq = {f"{q:g}": np.quantile(sm[f"lnC_{q:g}"].reshape(-1), [0.025, 0.1, 0.5, 0.9, 0.975]).tolist() for q in OM.Q_TARGETS}
    hyp = {}
    for h in HYP:
        x = sm[h].reshape(-1)
        hyp[h] = {"median": float(np.median(x)), "q10": float(np.quantile(x, 0.1)), "q90": float(np.quantile(x, 0.9)),
                  "exp_median": float(np.exp(np.median(x)))}
    core = {"rhat_max": max(dg["per"][k]["rhat"] for k in dg["per"]), "ess_min": min(dg["per"][k]["ess"] for k in dg["per"])}
    core["conv_core"] = bool(core["rhat_max"] < 1.01 and core["ess_min"] >= 400 and info["divergences"] == 0)
    off = OM.diagnose(sm, list(M1.OFF), divergences=0)
    hy = OM.diagnose(sm, HYP, divergences=0)
    cf = OM.diagnose(sm, ["beta1", "beta2"], divergences=0)
    core.update({"offsets_rhat_max": off["rhat_max"], "offsets_ess_min": off["ess_min"],
                 "hyper_rhat_max": hy["rhat_max"], "hyper_ess_min": hy["ess_min"],
                 "coef_rhat_max": cf["rhat_max"], "coef_ess_min": cf["ess_min"]})
    m1s = m1_json["seasons"][season]
    dif = {}
    for q in SPOT_Q:
        med2 = lq[f"{q:g}"][2]
        med1 = m1s["lnC_q"][f"{q:g}"][2]
        dif[f"{q:g}"] = {"M2_median": med2, "M1_median": med1, "dlnC": med2 - med1, "pct": 100 * (math.exp(med2 - med1) - 1),
                         "M2_sd": float(np.std(sm[f"lnC_{q:g}"].reshape(-1))),
                         "width80_M2": lq[f"{q:g}"][3] - lq[f"{q:g}"][1], "width80_M1": m1s["lnC_q"][f"{q:g}"][3] - m1s["lnC_q"][f"{q:g}"][1]}
    return {
        "n_rows": obs["n"], "N": obs["N"], "counts": obs["counts"], "delay_key": M1.delay_key_stage1(season),
        "basis": {"M_f1": mb[0], "M_f2": mb[1]},
        "mean": v_h.mean(0).tolist(), "cov": np.cov(v_h.T).tolist(),
        "mean_with_hyper": v_all.mean(0).tolist(), "cov_with_hyper": np.cov(v_all.T).tolist(),
        "hyper": hyp, "lnC_q": lq, "diag": {k: dg[k] for k in ("rhat_max", "ess_min", "conv", "divergences")}, "core": core,
        "worst": {"rhat": sorted(((v["rhat"], k) for k, v in dg["per"].items()), reverse=True)[:3],
                  "ess": sorted((v["ess"], k) for k, v in dg["per"].items())[:3]},
        "retried": info["retried"], "depth_dist": info.get("depth_dist"), "tree_cap_frac": info.get("tree_cap_frac"),
        "info": {k: info[k] for k in info}, "map": r["map"], "modes": r["modes"], "n_starts": r["n_starts"], "n_best_repeat": r["n_best_repeat"],
        "loo": {"elpd_loo": float(lo.elpd_loo), "se": float(lo.se), "p_loo": float(lo.p_loo), "pareto_k_gt_0.7": int((pk > 0.7).sum()), "n": int(len(pk))},
        "vs_M1": {"elpd_M1": m1s["loo"]["elpd_loo"], "se_M1": m1s["loo"]["se"], "n_M1": m1s["loo"]["n"],
                  "elpd_diff_M2_minus_M1": float(lo.elpd_loo) - m1s["loo"]["elpd_loo"], "lnC": dif},
        "check43": {"A_residual_KR": a_residuals(sm, obs, G, mean), "n_flags": len(fl), "flags": fl},
        "mean_M1_theta": {k: round(float(v), 5) for k, v in mean_theta.items() if k in ("a", "r0", "r1", "r2", "lnm", "eta0", "eta1", "beta")},
    }


def _git_dirty():
    """lab7 도구가 커밋과 다르면 True (도우미 세션은 고치지 않고 돌려야 한다)."""
    try:
        return bool(subprocess.check_output(["git", "-C", HERE, "status", "--porcelain", "--", HERE], text=True).strip())
    except Exception:
        return None


def _sha12(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


STAGE1_CFG_KEYS = ("warmup", "samples", "retry_warmup", "max_tree_depth", "refit_limit_secs", "K", "basis_start", "target_accept")


def cmd_stage1(a):
    import obs_table as O
    OM.setup_host(4)
    err = OM.load_err()
    with open(a.m1_stage1, encoding="utf-8") as f:
        m1j = json.load(f)
    seasons = C.parse_seasons(a.seasons)
    miss = [s for s in seasons if s not in m1j["seasons"]]
    if miss:
        raise SystemExit(f"--m1-stage1에 없는 시즌: {miss}")
    cfg = {"warmup": a.warmup, "samples": a.samples, "retry_warmup": a.retry_warmup, "max_tree_depth": a.max_tree_depth,
           "refit_limit_secs": a.refit_limit_secs, "K": 16, "basis_start": MB, "target_accept": 0.9}
    if _git_dirty():
        print("경고: lab7 도구가 커밋과 다르다 (git status). 합치기가 같은 코드 판인지 git head만 본다.", flush=True)
    out = {"about": ("M2 1단계 (조정 시즌): 끝난 시즌 전체 표를 약한 사전으로 맞춘 사후. 평균 함수 r̄·ℓ̄ = 그 시즌 M1(kuma + x항 β + 2차 컷 기울기 r2) 1단계 사후 평균 theta_hat에서 만든 것 "
                     "(명세에 1단계 평균이 적혀 있지 않아 사용자가 정함 — 2단계 평균을 1단계 theta_hat에서 만드는 것과 맞춤). a·lnm·β·lnsig0·lnrho·서버 모수: 약한 사전 (m1.WEAK), "
                     "f1·f2 길이 역감마·크기 반정규 표집, 계수 N(0,1) 비중심. 시작점 8개 MAP, NUTS 4사슬 x (예열 1000 + 뽑기 1000), target_accept 0.9, 재시도 0.95·예열 1500, 나무 깊이 상한 10. "
                     "hier에 넘길 mean/cov = names(= a, lnm, lnsig0, lnrho, beta, dJP, dGL, pJP, pGL, kJP, kGL) 사후 평균·공분산 (f1·f2 계수는 2단계에서 N(0,1)로 다시 맞추고 길이·크기는 창 안 중앙을 그대로 쓰므로 뺌). "
                     "mean_with_hyper/cov_with_hyper = 위에 초모수 로그 길이·로그 크기 4개를 더한 것. 정답 표 안 씀."),
           "data_ver": C.DATA_VER, "model": "M2", "m1_stage1": os.path.basename(a.m1_stage1), "m1_stage1_sha1": _sha12(a.m1_stage1), "config": cfg, "mean_source": "M1 kuma + beta + r2 1단계 사후 평균 theta_hat (같은 시즌)",
           "names": HYP_NAMES_HIER, "hyper_names": HYP, "delay_key_rule": M1.DELAY_RULE_NOTE, "grid_u": "코사인 간격 200점 [0.02, 1]", "tool_head": M1._git_head(), "tool_dirty": _git_dirty(), "seasons": {}}
    if a.resume and os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            out = json.load(f)
        if out.get("tool_head") != M1._git_head() or out.get("config") != cfg or out.get("m1_stage1_sha1") != _sha12(a.m1_stage1):
            raise SystemExit("--resume: 기존 --out의 코드 판·설정·M1 1단계 파일이 지금과 다르다. 다른 --out을 쓰거나 지우고 다시 돌린다.")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tag = a.tag
    log_dir = os.path.dirname(os.path.abspath(a.ll_dir.rstrip("/")))
    os.makedirs(log_dir, exist_ok=True)

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
        print(line, flush=True)
        with open(os.path.join(log_dir, "attempts.log"), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    for s in seasons:
        if s in out["seasons"] and a.resume:
            continue
        t0 = time.time()
        head, rows = O.read_cell(os.path.join(a.obs, f"{s}.jsonl.gz"))
        mb = (MB, MB)
        mean, m1theta = m1_mean_for(m1j, s)
        kw = dict(K=16, delay_key=M1.delay_key_stage1(s), warmup=a.warmup, samples=a.samples, retry_warmup=a.retry_warmup, max_tree_depth=a.max_tree_depth)
        r = fit_one(rows, head["H"], s, err, mean, mb, **kw)
        bc = basis_check(r["samples"], mb)
        t_first = time.time() - t0
        basis_log = {"first": {"mb": list(mb), "check": bc, "secs": round(t_first, 1), "conv": r["diag"]["conv"], "divergences": r["info"]["divergences"]}}
        log(f"{s} M={mb} {t_first:.0f}s conv {r['diag']['conv']} f1 ell {bc['f1']['ell_median']} (min {bc['f1']['min_len']}) f2 ell {bc['f2']['ell_median']} (min {bc['f2']['min_len']})")
        need = (MB * 2 if bc["f1"]["short"] else MB, MB * 2 if bc["f2"]["short"] else MB)
        if need != mb:
            if t_first > a.refit_limit_secs:
                basis_log["action"] = f"기저를 늘려야 하나(짧음) 첫 맞춤이 {t_first:.0f}초라 시간 상한({a.refit_limit_secs}초)으로 다시 맞추지 않음"
                log(f"{s} refit skipped (time)")
            else:
                # 다시 맞추다 시즌 60분 상한에 걸려도 첫 맞춤 결과는 남도록 먼저 저장한다 (first_fit_partial)
                try:
                    rec0 = _season_record(r, s, m1theta, m1j, grids(*mb), mean, a.ll_dir, tag + "_first")
                    rec0["secs_total"] = round(time.time() - t0, 1)
                    rec0["basis"]["log"] = basis_log
                    rec0["note"] = "첫 맞춤(기저 20/20) 결과 — 기저가 짧아 다시 맞추는 중에 저장. 최종 기록이 있으면 이 항목은 지워진다"
                    out.setdefault("first_fit_partial", {})[s] = rec0
                    with open(a.out + ".tmp", "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
                    os.replace(a.out + ".tmp", a.out)
                except Exception as e:  # noqa
                    log(f"{s} first-fit partial save failed: {type(e).__name__}: {e}")
                r_first = r
                mb = need
                r = fit_one(rows, head["H"], s, err, mean, mb, seed=100, **kw)
                bc2 = basis_check(r["samples"], mb)
                basis_log["action"] = f"기저 {list(r_first['mb'])} -> {list(mb)} 로 늘려 다시 맞춤"
                basis_log["refit"] = {"mb": list(mb), "check": bc2, "secs": round(time.time() - t0 - t_first, 1), "conv": r["diag"]["conv"],
                                      "divergences": r["info"]["divergences"], "lnC_3_median_first": float(np.median(r_first["samples"]["lnC_3"])),
                                      "lnC_3_median_refit": float(np.median(r["samples"]["lnC_3"]))}
                bc = bc2
                log(f"{s} refit M={mb} conv {r['diag']['conv']} f1 ell {bc['f1']['ell_median']} (min {bc['f1']['min_len']}) f2 ell {bc['f2']['ell_median']} (min {bc['f2']['min_len']})")
        else:
            basis_log["action"] = "기저 충분 (사후 길이 중앙이 최소 길이 이상) — 그대로"
        G = grids(*mb)
        rec = _season_record(r, s, m1theta, m1j, G, mean, a.ll_dir, tag)
        rec["basis"].update({"log": basis_log, "check_final": bc})
        rec["secs_total"] = round(time.time() - t0, 1)
        out["seasons"][s] = rec
        out.get("first_fit_partial", {}).pop(s, None)
        with open(a.out + ".tmp", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=float)
        os.replace(a.out + ".tmp", a.out)
        d = rec
        log(f"stage1 M2 {s} D1: {rec['secs_total']:.0f}s div {r['info']['divergences']} rhat {d['diag']['rhat_max']:.3f} ess {d['diag']['ess_min']:.0f} "
            f"conv {d['diag']['conv']} conv_core {d['core']['conv_core']} hyper rhat {d['core']['hyper_rhat_max']:.3f} ess {d['core']['hyper_ess_min']:.0f} "
            f"M {mb} retried {r['info']['retried']} elpd {d['loo']['elpd_loo']:.1f} (M1 {d['vs_M1']['elpd_M1']:.1f}) k>0.7 {d['loo']['pareto_k_gt_0.7']} "
            f"dlnC1/3/10 " + "/".join(f"{d['vs_M1']['lnC'][q]['pct']:+.2f}%" for q in ("1", "3", "10")))


# ---------- 1단계 합치기 ----------

def cmd_merge_stage1(a):
    """시즌 묶음 1단계 파일들을 하나로. 같은 코드 판(git head)·같은 설정·같은 M1 1단계 파일이어야 하고, 시즌이 겹치면 오류."""
    js = []
    for p in a.inputs:
        with open(p, encoding="utf-8") as f:
            js.append((p, json.load(f)))
    base_p, base = js[0]
    same = ("data_ver", "model", "m1_stage1", "m1_stage1_sha1", "config", "names", "hyper_names", "delay_key_rule", "grid_u", "tool_head", "mean_source")
    errs = []
    for p, j in js:
        if j.get("tool_dirty"):
            errs.append(f"{p}: tool_dirty = True (lab7 도구가 커밋과 다른 채 돌린 결과)")
        if j.get("first_fit_partial"):
            errs.append(f"{p}: first_fit_partial이 남아 있다 ({sorted(j['first_fit_partial'])}) — 다시 맞추다 멈춘 시즌. --resume으로 마저 돌린다")
        for k in same:
            if j.get(k) != base.get(k):
                errs.append(f"{p}: '{k}'가 {base_p}와 다르다 ({j.get(k)!r} vs {base.get(k)!r})")
    seen = {}
    for p, j in js:
        for s in j["seasons"]:
            if s in seen:
                errs.append(f"시즌 {s}이 {seen[s]}와 {p}에 겹친다")
            seen[s] = p
    if errs:
        raise SystemExit("합칠 수 없다:\n  " + "\n  ".join(errs))
    out = {k: v for k, v in base.items() if k not in ("seasons", "first_fit_partial", "run_notes", "tool_dirty")}
    out["seasons"] = {s: seen_j["seasons"][s] for seen_j in (j for _, j in js) for s in seen_j["seasons"]}
    out["seasons"] = dict(sorted(out["seasons"].items(), key=lambda kv: int(kv[0].lstrip("S"))))
    out["merged_from"] = [{"file": os.path.basename(p), "sha1": _sha12(p), "seasons": sorted(j["seasons"], key=lambda x: int(x.lstrip("S")))} for p, j in js]
    notes = {os.path.basename(p): j["run_notes"] for p, j in js if j.get("run_notes")}
    if notes:
        out["run_notes"] = notes
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    os.replace(a.out + ".tmp", a.out)
    nconv = sum(1 for v in out["seasons"].values() if v["diag"]["conv"])
    print(f"merge-stage1: {len(js)}개 파일 -> {a.out}: 시즌 {len(out['seasons'])}개 ({', '.join(out['seasons'])}), conv {nconv}/{len(out['seasons'])}, tool_head {out['tool_head']}")


# ---------- 칸 예측 (명세 3절 2단계 + 4절 M2) ----------
# 맞힐 시즌 s의 칸 (s, T)마다:
#   M1 2단계 사전 평균 theta -> 평균 곡선 r̄·ℓ̄ (mean_from_m1 — M1이 쓰는 함수 그대로)
#   M2 1단계 theta_hat_j (a 제외 모수 CORE) -> hier.prior_for (창 k 고르기 + NUTS 정규-정규) -> 정규 사전, a는 hier.fit_level·level_prior (prior_for 안)
#   초모수 = 창 안 시즌 1단계 사후 중앙(로그)의 중앙, 기저 수 = 그 길이로 Riutort-Mayol 진단을 지나는 최소 (20 -> 40)
#   f1·f2 계수 N(0,1) 비중심, 시작점 8개 MAP -> NUTS 4 x (500 + 500), 수렴 판정·재시도는 명세 6절 (R-hat < 1.01 · 발산 0 · ESS >= 400, 못 지나면 0.95·예열 1000으로 한 번 더)

def _snum(s):
    return int(s.lstrip("S"))


def stage1_dict(j):
    """1단계 파일 -> hier 입력 {시즌: {"names","mean","cov"}}."""
    return {s: {"names": j["names"], "mean": v["mean"], "cov": v["cov"]} for s, v in j["seasons"].items()}


def _pick_prior(hj, season):
    """M1 세션이 낸 hier 결과 파일에서 그 시즌의 사전 (hier.prior_for 꼴 dict: names·mean·a). {"targets": {s: ..}} (hier_M1_D1.json), {"seasons": {s: ..}}, {"priors": {s: ..}}, {s: ..} 어느 꼴이든, 값이 {"prior": ..}로 싸여 있어도 받는다."""
    for top in (hj.get("targets"), hj.get("seasons"), hj.get("priors"), hj):
        if isinstance(top, dict) and season in top:
            v = top[season]
            return v["prior"] if isinstance(v, dict) and "prior" in v else v
    raise KeyError(f"M1 hier 파일에 시즌 {season}이 없다")


def m1_prior(m1j, season, hier_json=None, cache=None):
    """맞힐 시즌의 M1 2단계 사전. hier_json이 있으면 거기서, 없으면 M1 1단계 파일로 hier.prior_for를 M1과 같은 인자로 부른다."""
    if hier_json is not None:
        return _pick_prior(hier_json, season)
    names = list(m1j["names"][1:])
    mu_loc, mu_scale, tau = M1.hier_args(names)
    return H.prior_for(stage1_dict(m1j), season, names, tau, mu_loc, mu_scale, cache)


def mean_from_m1_prior(prior, m1j, G=None):
    """M1 사전 평균 theta(= prior mean + a 평균)로 평균 곡선. M1 최종 꼴(kuma + x항 + r2)만 받는다. -> (mean, theta dict)."""
    assert m1j["form"] == "kuma" and m1j["x_term"] and m1j["r2"], "M1 최종 꼴(kuma + beta + r2)이 아님"
    th = dict(zip(prior["names"], (float(x) for x in prior["mean"])))
    th["a"] = float(prior["a"]["mean"])
    return mean_from_m1({"r0": th["r0"], "r1": th["r1"], "r2": th["r2"], "eta": [th["eta0"], th["eta1"]]}, "kuma", G), th


def m2_prior(m2j, season, cache=None):
    """맞힐 시즌의 M2 2단계 사전 (a 제외 모수 CORE는 정규, a는 국소 수준)."""
    mu_loc, mu_scale, tau = M1.hier_args(CORE)
    return H.prior_for(stage1_dict(m2j), season, CORE, tau, mu_loc, mu_scale, cache)


def hyper_for_window(m2j, window):
    """창 안 시즌들의 1단계 초모수 사후 중앙(로그)의 중앙 -> (hyper dict, mb). 기저 수는 그 길이가 Riutort-Mayol 최소 길이 이상인 최소 (20 -> 40, 40도 모자라면 40 + short 표시)."""
    med = {h: float(np.median([m2j["seasons"][j]["hyper"][h]["median"] for j in window])) for h in HYP}
    hy = {"ell1": math.exp(med["lnell1"]), "alpha1": math.exp(med["lnalp1"]), "ell2": math.exp(med["lnell2"]), "alpha2": math.exp(med["lnalp2"])}
    Ww, Wu = W_HI - W_LO, 1.0 - U_LO
    pick, short = [], {}
    for tag, W, ell in (("f1", Ww, hy["ell1"]), ("f2", Wu, hy["ell2"])):
        M = next((m for m in (MB, 2 * MB) if ell >= min_len(W, m)), 2 * MB)
        short[tag] = bool(ell < min_len(W, M))
        pick.append(M)
    return hy, (pick[0], pick[1]), short


def _atomic_dump(obj, path):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    os.replace(path + ".tmp", path)


def cmd_predict(a):
    import obs_table as O
    OM.setup_host(4)
    err = OM.load_err()
    with open(a.m1_stage1, encoding="utf-8") as f:
        m1j = json.load(f)
    with open(a.m2_stage1, encoding="utf-8") as f:
        m2j = json.load(f)
    hj = None
    if a.m1_hier:
        with open(a.m1_hier, encoding="utf-8") as f:
            hj = json.load(f)
    cfg = {"warmup": a.warmup, "samples": a.samples, "retry_warmup": a.retry_warmup, "max_tree_depth": a.max_tree_depth, "K": 16, "chains": 4,
           "target_accept": 0.9, "retry_target_accept": 0.95}
    inputs = {"m2_stage1": _sha12(a.m2_stage1), "m1_stage1": _sha12(a.m1_stage1)}
    if a.m1_hier:
        inputs["m1_hier"] = _sha12(a.m1_hier)
    out = {"about": ("M2 HSGP 곡면 칸 예측: 맞힐 시즌 s의 T 절단 관측 표로, 사전은 앞 시즌 1단계만 (M1 2단계 사전 평균 -> 평균 곡선 r̄·ℓ̄, M2 자신의 2단계 -> 모수 사전, "
                     "초모수는 창 안 시즌 1단계 사후 중앙의 중앙으로 고정). f1·f2 계수 N(0,1) 비중심, 시작점 8개 MAP, NUTS 4 x (예열+뽑기), 수렴 못 하면 0.95·긴 예열로 한 번 더. "
                     "conv false인 칸도 값은 낸다 (F로 채우지 않음). fits = 칸별 수렴·시간·나무 깊이 분포."),
           "method": "M2", "data_ver": C.DATA_VER,
           "made": {"tool_head": M1._git_head(), "tool_dirty": _git_dirty(), "cmd": "m2.py predict --seasons %s --T %s" % (a.seasons, a.T), "inputs": inputs},
           "config": cfg, "m1_prior_source": "hier 파일" if hj is not None else "M1 1단계 파일에서 hier.prior_for", "cells": [], "fits": []}
    if a.resume and os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("made", {}).get("inputs") != inputs or old.get("config") != cfg or old.get("made", {}).get("tool_head") != out["made"]["tool_head"]:
            raise SystemExit("--resume: 기존 --out의 입력 파일·설정·코드 판이 지금과 다르다.")
        out = old
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = {(f["season"], f["T"]) for f in out["fits"] if not f.get("error")}
    want_T = None if a.T == "all" else [float(x) for x in a.T.split(",")]
    priors = {}

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
        print(line, flush=True)
        if a.log:
            with open(a.log, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    for s in C.parse_seasons(a.seasons):
        if s not in priors:
            t0 = time.time()
            p1 = m1_prior(m1j, s, hj, a.cache)
            mean, m1th = mean_from_m1_prior(p1, m1j)
            p2 = m2_prior(m2j, s, a.cache)
            hy, mb, short = hyper_for_window(m2j, p2["window"])
            priors[s] = {"mean": mean, "prior": p2, "hyper": hy, "mb": mb,
                         "info": {"m1_window_k": p1["k"], "m1_window": p1["window"], "m2_window_k": p2["k"], "m2_window": p2["window"], "hyper": hy,
                                  "mb": list(mb), "basis_short_at_max": short, "m1_theta_mean": {k: round(v, 5) for k, v in m1th.items() if k in ("a", "r0", "r1", "r2", "lnm", "eta0", "eta1", "beta")},
                                  "prior_secs": round(time.time() - t0, 1)}}
            log(f"{s} 사전: M1 창 k={p1['k']} M2 창 k={p2['k']} 초모수 {{{', '.join(f'{k} {v:.3f}' for k, v in hy.items())}}} 기저 {mb} ({time.time() - t0:.0f}초)")
        pr = priors[s]
        Ts = want_T if want_T is not None else C.t_grid(s)
        for T in Ts:
            if (s, float(T)) in done:
                continue
            path = os.path.join(a.obs, s, f"T{T:g}.jsonl.gz")
            if not os.path.exists(path):
                log(f"{s} T{T:g}: 관측 표 없음 ({path}) — 건너뜀")
                continue
            t0 = time.time()
            rec = {"season": s, "T": float(T), "window": pr["info"]}
            try:
                head, rows = O.read_cell(path)
                dk = s if _snum(s) >= 21 else None
                r = fit_one(rows, head["H"], s, err, pr["mean"], pr["mb"], K=16, delay_key=dk, seed=0, warmup=a.warmup, samples=a.samples,
                            retry_warmup=a.retry_warmup, max_tree_depth=a.max_tree_depth, prior=pr["prior"], hyper=pr["hyper"], stage1=False, keep_ll=False)
                sm, info, dg, obs = r["samples"], r["info"], r["diag"], r["obs"]
                conv = bool(dg["conv"])
                cells = OM.pred_cells(s, T, {q: sm[f"lnC_{q:g}"] for q in OM.Q_TARGETS}, conv)
                for c in cells:
                    c["secs"] = round(time.time() - t0, 1)
                rec.update({"n_rows": obs["n"], "N": obs["N"], "counts": obs["counts"], "conv": conv, "secs": round(time.time() - t0, 1), "map_secs": info["map_secs"],
                            "nuts_secs": info["secs"], "retried": info["retried"], "divergences": info["divergences"], "rhat_max": dg["rhat_max"], "ess_min": dg["ess_min"],
                            "worst": {"rhat": sorted(((v["rhat"], k) for k, v in dg["per"].items()), reverse=True)[:3], "ess": sorted((v["ess"], k) for k, v in dg["per"].items())[:3]},
                            "depth_dist": info.get("depth_dist"), "tree_cap_frac": info.get("tree_cap_frac"), "mean_leapfrog": info.get("mean_leapfrog"),
                            "n_starts": r["n_starts"], "n_best_repeat": r["n_best_repeat"],
                            "lnC_q": {f"{q:g}": np.quantile(sm[f"lnC_{q:g}"].reshape(-1), [0.025, 0.1, 0.5, 0.9, 0.975]).tolist() for q in OM.Q_TARGETS}})
                out["cells"] = [c for c in out["cells"] if not (c["season"] == s and c["T"] == float(T))] + cells
                log(f"{s} T{T:g}: {rec['secs']:.0f}초 (map {rec['map_secs']}s nuts {rec['nuts_secs']}s) 줄 {obs['n']} conv {conv} rhat {dg['rhat_max']:.3f} ess {dg['ess_min']:.0f} "
                    f"div {info['divergences']} retried {info['retried']} depth {info.get('depth_dist')} 컷(1·3·10%) " +
                    "/".join(f"{math.exp(rec['lnC_q'][q][2]):.1f}" for q in ("1", "3", "10")))
            except Exception as e:  # noqa
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["secs"] = round(time.time() - t0, 1)
                log(f"{s} T{T:g}: 오류 {rec['error']}")
            out["fits"] = [f for f in out["fits"] if not (f["season"] == s and f["T"] == float(T))] + [rec]
            _atomic_dump(out, a.out)
    n = len(out["fits"])
    log(f"predict 끝: 칸 {n}개, 수렴 {sum(1 for f in out['fits'] if f.get('conv'))}/{n}, 오류 {sum(1 for f in out['fits'] if f.get('error'))} -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("bench")
    b.add_argument("--work", required=True)
    b.add_argument("--case", required=True, choices=["a", "b", "c"])
    b.add_argument("--season", default=None)
    b.add_argument("--chains", type=int, default=4)
    b.add_argument("--warmup", type=int, default=500)
    b.add_argument("--samples", type=int, default=500)
    b.add_argument("--probe", action="store_true")
    b.add_argument("--tag", default="")
    b.add_argument("--mean", default="temp", choices=["temp", "m1"])
    b.add_argument("--ta", type=float, default=0.9)
    b.add_argument("--multistart", type=int, default=1, help="1이면 시작점 8개 MAP(기본), 0이면 단일 MAP")
    c = sp.add_parser("compare")
    c.add_argument("--work", required=True)
    c.add_argument("--seasons", default="S34")
    c.add_argument("--warmup", type=int, default=500)
    c.add_argument("--samples", type=int, default=500)
    s1 = sp.add_parser("stage1")
    s1.add_argument("--obs", required=True)
    s1.add_argument("--seasons", required=True, help="S19-S24 · S19,S25 · S19-S24,S30 꼴")
    s1.add_argument("--m1-stage1", required=True)
    s1.add_argument("--out", required=True)
    s1.add_argument("--ll-dir", required=True)
    s1.add_argument("--resume", action="store_true")
    s1.add_argument("--tag", default="M2", help="ll-dir 안 줄별 LOO 파일 이름 앞부분")
    s1.add_argument("--warmup", type=int, default=1000)
    s1.add_argument("--samples", type=int, default=1000)
    s1.add_argument("--retry-warmup", type=int, default=1500)
    s1.add_argument("--max-tree-depth", type=int, default=10)
    s1.add_argument("--refit-limit-secs", type=int, default=1800, help="첫 맞춤이 이보다 오래 걸렸으면 기저를 늘려 다시 맞추지 않는다 (시즌 60분 상한)")
    mg = sp.add_parser("merge-stage1")
    mg.add_argument("--in", dest="inputs", nargs="+", required=True)
    mg.add_argument("--out", required=True)
    pd = sp.add_parser("predict")
    pd.add_argument("--obs", required=True, help="T 절단 관측 표 폴더 (obs_table.py build --work W --out DIR이 만든 DIR — DIR/{시즌}/T{T}.jsonl.gz)")
    pd.add_argument("--seasons", required=True, help="맞힐 시즌 S42 · S22-S26 · S22,S30")
    pd.add_argument("--T", default="all", help="all(= common.t_grid) 또는 48 · 24,48,96")
    pd.add_argument("--m2-stage1", required=True, help="M2 1단계 (r7/stage1_M2_D1.json — merge-stage1 결과)")
    pd.add_argument("--m1-stage1", required=True, help="M1 1단계 (r7/stage1_M1_D1.json)")
    pd.add_argument("--m1-hier", default=None, help="M1 2단계 결과 (r7/hier_M1_D1.json). 없으면 M1 1단계 파일에서 hier.prior_for를 M1과 같은 인자로 부른다")
    pd.add_argument("--out", required=True)
    pd.add_argument("--cache", default=None, help="hier 창 적합 캐시 폴더 (디스크 npz)")
    pd.add_argument("--log", default=None, help="진행 기록 파일 (덧붙임)")
    pd.add_argument("--resume", action="store_true")
    pd.add_argument("--warmup", type=int, default=500)
    pd.add_argument("--samples", type=int, default=500)
    pd.add_argument("--retry-warmup", type=int, default=1000)
    pd.add_argument("--max-tree-depth", type=int, default=10)
    t = sp.add_parser("test")
    t.add_argument("--work", default=None)
    t.add_argument("--m1-stage1", default=None)
    t.add_argument("--season", default="S30")
    t.add_argument("--which", required=True, choices=["basis", "synth", "m1match"])
    t.add_argument("--free-hyper", action="store_true")
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--samples", type=int, default=500)
    a = ap.parse_args()
    if a.cmd == "bench":
        OM.setup_host(a.chains)
        r = bench_case(a.work, a.case, a.season, a.chains, a.warmup, a.samples, target_accept=a.ta, probe=a.probe, mean_kind=a.mean, multistart=bool(a.multistart))
        name = f"bench_{a.case}{a.tag}.json"
    elif a.cmd == "compare":
        r = compare_multistart(a.work, a.seasons.split(","), 4, a.warmup, a.samples)
        name = "multistart.json"
    elif a.cmd == "stage1":
        cmd_stage1(a)
        return
    elif a.cmd == "merge-stage1":
        cmd_merge_stage1(a)
        return
    elif a.cmd == "predict":
        cmd_predict(a)
        return
    elif a.cmd == "test" and a.which == "m1match":
        r = test_m1match(a.m1_stage1, a.season)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        if a.work:
            os.makedirs(os.path.join(a.work, "m2"), exist_ok=True)
            with open(os.path.join(a.work, "m2", "test_m1match.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=1)
        return
    else:
        OM.setup_host(4)
        r = test_basis() if a.which == "basis" else test_synth(a.work, a.season, 4, a.warmup, a.samples, a.free_hyper)
        name = f"test_{a.which}{'_free' if a.free_hyper else ''}.json"
    d = os.path.join(a.work, "m2")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(r, ensure_ascii=False, indent=1, default=float)[:5000])


if __name__ == "__main__":
    main()
