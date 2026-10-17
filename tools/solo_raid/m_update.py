"""자체 일관성 수준 갱신 (M4 후보, 2026-10-17 분석 4.1에서 lab4/mupd.py로 처음 씀 → 4.0a에서 분석 도구로 옮김, 모델 0, 요청 0).

시즌 % 변환 수준 ln m을 그 시즌 T까지 자료로 prior와 함께 푼다. 한 시즌 prior 위에서도 된다
(predict의 early_level은 여러 시즌 prior의 season_levels·학습 5시즌 이상이 있어야 켜진다 — 분석 4.0a).
xval.py --level-update '{}' 가 이것을 불러 결과를 predict의 _lvl(early_level과 같은 자리)로 넣는다. predict.py는 고치지 않는다 (S42 기본 동작).

관측 i마다 c_i(δ) = ln s_i − ln r(u_i(δ)), u_i(δ) = ln x_i + LN10·e^δ·P_i (P_i = prior 중앙 변환).
수준이 맞으면 c_i는 시각과 무관하게 ln S3(규모) 근처다. 틀리면 c_i가 z_i = k(y_i)·LN10·P_i 쪽으로 기운다.
목적 J(a, δ) = Σ w_i (c_i(δ) − a)² / φ + (δ − μ0)² / τ²  (w_i = 사람마다 합 1이 되게 나눈 1/분산, φ = 과분산)
Gauss–Newton 몇 번. 반환 {mu, sd, persons, phi, …} 또는 None (사람 < min_persons).
4.1 단서: 직전 시즌 하나(M0) 위 T18~48 |e| 3.56 → 2.12%, 포함 0.76 → 0.73 (고르기 아님 — criteria_v2로 다시 잰다).
"""
import math
import collections

LN10 = math.log(10)


def m_posterior(prior, rows, H, tau, mu0=0.0, srv="KR", pt_sd=0.03, iters=4, cap=0.8, min_persons=3, phi_floor=1.0,
                use_sf=True):
    import predict as PR
    t = prior["transform"]
    tr = {"form": t["form"], "theta": t["theta"], "bx": t.get("bx", 0.0), "m": t["m"], "s0": t["sigma"]["s0"], "rho": t["sigma"]["rho"]}
    sh = PR.Shape(prior["shape"]["servers"][srv])
    O = []
    for o in rows:
        if o.get("unused"):
            continue
        if o["server"] == srv:
            p = 1.0
        elif not o["server"] and o.get("mix"):
            p = o["mix"].get(srv, 0.0)
        else:
            continue
        if p <= 0:
            continue
        P = PR.p_full(tr, srv, o["h"], o["x"], H, 1.0)
        if P <= 0:
            continue
        sp2 = tr["s0"] ** 2 + (tr["rho"] * P) ** 2
        sf = PR.sigma_form(t, o["h"], H) if use_sf else 0.0
        O.append((o, p, P, sp2 + sf ** 2))
    persons = collections.Counter(o["person"] for o, *_ in O)
    if len(persons) < min_persons:
        return None
    delta = mu0
    info = {}
    for it in range(iters):
        c, z, w = [], [], []
        for o, p, P, v in O:
            Pd = P * math.exp(delta)
            u = math.log(o["x"]) + LN10 * Pd
            if not sh.inside(u):
                continue
            k = sh.k_at(u)
            ci = math.log(o["s"]) - sh.lr_at(u)
            zi = k * LN10 * Pd                      # dc/dδ
            var = (k * LN10) ** 2 * v * math.exp(2 * delta) + PR.V0 + pt_sd ** 2
            c.append(ci)
            z.append(zi)
            w.append(p / (var * persons[o["person"]]))
        if len(c) < 3:
            return None
        Sw = sum(w)
        a = sum(wi * ci for wi, ci in zip(w, c)) / Sw
        zb = sum(wi * zi for wi, zi in zip(w, z)) / Sw
        # 잔차 e_i = c_i − a; dc/dδ = z_i, 프로파일: (z_i − z̄)
        r = [ci - a for ci in c]
        if it == 0:
            npers = len(persons)
            phi = sum(wi * ri * ri for wi, ri in zip(w, r)) / (npers - 2) if npers > 2 else 1.0
            phi = max(phi, phi_floor)
            info["phi"] = round(phi, 3)
        I = sum(wi * (zi - zb) ** 2 for wi, zi in zip(w, z)) / phi
        g = sum(wi * (zi - zb) * ri for wi, zi, ri in zip(w, z, r)) / phi   # ∂(−J/2)/∂δ 의 자료 부분 (부호: c가 δ로 커지면 r 줄어야)
        # J = Σ w (c(δ) − a)²/φ + (δ−μ0)²/τ² → dJ/dδ = 2Σ w r z'/φ + 2(δ−μ0)/τ²
        prec = I + 1 / tau ** 2
        step = -(g + (delta - mu0) / tau ** 2) / prec
        delta = max(mu0 - cap, min(mu0 + cap, delta + step))
        info.update(n=len(c), persons=len(persons), I=round(I, 3))
    sd = 1 / math.sqrt(I + 1 / tau ** 2)
    info.update(mu=round(delta, 5), sd=round(sd, 5), prior_mu=mu0, prior_tau=tau)
    return info
