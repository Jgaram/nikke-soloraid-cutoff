"""유사 시즌 수준 (분석 4.3 M2s 후보, 2026-10-18, 모델 0, 요청 0 — 기본 끔).

4.2 관문을 통과한 꼴 「가까운 3시즌 (m̂ + ④)」을 예측 시점에 쓸 수 있게 옮긴 것이다.
여러 시즌 prior(그 겹의 학습 시즌으로 만든 것) 위에서, 같은 T의 앞 시즌 절단 사본과 맞힐 시즌 절단 사본으로 정답 없이 특징 두 개를 낸다.
  ① m̂ — 초반 침수 크기 (explore_early_level.early_m, 라벨 + 미상 줄, 꼴 θ는 그 prior의 transform)
  ④ ln 관측 사람 수 — T까지 기본 집합에서 % 관측이 있는 사람 (모든 서버, 4.2 e_42.features_C와 같은 셈)
학습 시즌(prior의 season_levels가 있고 두 특징이 다 있는 시즌)의 평균·sd로 표준화한 거리로 가까운 k시즌을 골라
그 시즌 ln m_s의 평균을 시즌 수준 평균으로, 학습 시즌끼리 하나씩 빼기 잔차 rms(바닥 0.05)를 sd로 낸다.
xval.py --level-knn '{}' 가 이것을 불러 predict의 _lvl(early_level·m_update와 같은 자리)로 넣는다. predict.py는 고치지 않는다 (S42 기본 동작).
맞힐 시즌 m̂가 없거나 쓸 학습 시즌이 min_train 미만이면 None (prior 수준 그대로).
"""
import json
import math
import os
import statistics

FEATS = ("m_hat", "ln_persons")


def features(state, T, prior):
    """절단 사본 state 하나의 특징 (정답 없음)."""
    import predict as PR
    import explore_early_level as EL
    t = prior["transform"]
    _, _, H = PR.season_info(state)
    Te = min(float(T), H)
    r = EL.early_m(state, Te, t["theta"], t["form"], H, False, 0, 1)
    m = r.get("m")
    persons = set()
    with open(os.path.join(state, "core.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("use") and c.get("pct"):
                persons.add(c["person"])
    return {"m_hat": math.log(m) if m and m > 0 else None, "m_hat_persons": r.get("persons"),
            "ln_persons": math.log(len(persons)) if persons else None, "persons": len(persons)}


def _knn(train, x, k):
    """train = [(이름, [특징…], y)] → (예측, 고른 이름). 표준화는 train의 평균·모표준편차."""
    d = len(x)
    mu = [statistics.mean(r[1][q] for r in train) for q in range(d)]
    sd = [statistics.pstdev([r[1][q] for r in train]) or 1.0 for q in range(d)]
    z = lambda v: [(v[q] - mu[q]) / sd[q] for q in range(d)]
    zx = z(x)
    near = sorted(train, key=lambda r: (sum((a - b) ** 2 for a, b in zip(z(r[1]), zx)), r[0]))[:k]
    return statistics.mean(r[2] for r in near), [r[0] for r in near]


def knn_level(prior, state, T, train, k=3, min_train=5, sd_floor=0.05, feats=FEATS):
    """train = [[시즌, 그 시즌의 같은 T 절단 state]…] → {mu, sd, near, n_train, …} 또는 None."""
    lv = prior["transform"].get("season_levels") or {}
    x0 = features(state, T, prior)
    info = {"target": {f: x0[f] for f in feats}, "persons": x0["persons"]}
    if any(x0[f] is None for f in feats):
        return None
    rows = []
    for name, st in train:
        if name not in lv or not lv[name] or lv[name] <= 0:
            continue
        f = features(st, T, prior)
        if any(f[q] is None for q in feats):
            continue
        rows.append((name, [f[q] for q in feats], math.log(lv[name])))
    if len(rows) < min_train:
        return None
    mu, near = _knn(rows, [x0[f] for f in feats], k)
    res = []
    for i, (name, x, y) in enumerate(rows):
        others = rows[:i] + rows[i + 1:]
        res.append(y - _knn(others, x, k)[0])
    sd = max(sd_floor, math.sqrt(sum(e * e for e in res) / len(res)))
    info.update(mu=round(mu, 5), sd=round(sd, 5), near=near, n_train=len(rows), k=k)
    return info
