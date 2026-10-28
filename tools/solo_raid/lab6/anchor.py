"""분석 5.3 새 방법 (2026-10-28) 공통 함수 — 닻 옮기기.

docs/solo-raid-analysis.md 「5.3 계획」. 운영 도구(predict.py·xval.py·후처리)는 고치지 않고 읽기만 한다.
관측 읽기는 운영과 같은 predict.load_obs(서버 미상 줄 mono 거르기 포함)를 쓴다.

닻 a = (점수 x_a, % p_a, 시각 h_a): 시각 A·B 사진. 지난 시즌 s의 같은 자리 컷 C_{p_a,s}(h_a)를 그 시즌 자기 점으로 잇고
R_s = (s의 최종 q% 컷) ÷ C_{p_a,s}(h_a). 예측 분포 = {x_a × R_s}. 하한 = 「이미 q% 밖인(% > q) 가장 높은 점수」.
"""
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
import predict as PR  # noqa: E402

DEV = [f"S{n}" for n in range(19, 43)]
TARGETS = [f"S{n}" for n in range(21, 43)]
PCTS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0]
PCT_LO, PCT_HI = 0.005, 0.01   # % 표시 0.01%p — 반올림·버림을 다 덮는 구간 [p − 0.005, p + 0.01)

_info, _truth, _full = {}, None, {}


def state_of(s):
    return os.path.join(ROOT, "data", "solo-raid", s, "state")


def info(s):
    """(c_start, H, 05:00 갱신에서 c_start까지 시간)."""
    if s not in _info:
        _, c0, H = PR.season_info(state_of(s))
        off = (c0.hour + c0.minute / 60 - 5) % 24
        _info[s] = (c0, float(H), off)
    return _info[s]


def truth():
    """{(시즌, 서버, %): 정답 억} — answer_key.json (정답 없는 칸은 빠짐)."""
    global _truth
    if _truth is None:
        with open(os.path.join(ROOT, "data", "solo-raid", "analysis", "answer_key.json"), encoding="utf-8") as f:
            ak = json.load(f)
        _truth = {(r["season"], r["server"], float(r["pct"])): r["truth"] for r in ak["rows"] if r["truth"] is not None}
    return _truth


def is_kr(o, unknown):
    """한섭 줄인가 — 라벨 한섭, 또는 (unknown이면) mono를 지난 미상 줄."""
    if o["server"] == "KR":
        return True
    return bool(unknown and o.get("mix") and o["mix"].get("KR"))


def load(state, s, T, unknown=True):
    """T까지 게시된 기본 집합 사진 관측 (predict.load_obs, 같은 사람·같은 %는 한 시점). unknown이면 mono 거르기."""
    c0, H, _ = info(s)
    rows, _ = PR.load_obs(state, T, H, c0, unknown={"mode": "mono"} if unknown else None)
    for o in rows:
        o["kr_label"] = o["server"] == "KR"
    return rows


def full(s):
    """끝난 시즌 전체 state의 라벨 한섭 줄 (지난 시즌 비 R 표용 — 그 시즌은 끝났으므로 뒤 글을 봐도 된다)."""
    if s not in _full:
        _, H, _ = info(s)
        rows = load(state_of(s), s, H + 1, unknown=False)
        _full[s] = [o for o in rows if o["kr_label"]]
    return _full[s]


def to_season(h_a, s_a, s, coord):
    """맞힐 시즌 s_a의 시각 h_a를 지난 시즌 s의 같은 자리로. coord: h(시작부터) · rem(남은 시간) · reset(05:00 갱신 기준 시각)."""
    if coord == "h":
        return h_a
    _, Ha, offa = info(s_a)
    _, Hs, offs = info(s)
    if coord == "rem":
        return Hs - (Ha - h_a)
    if coord == "reset":
        return h_a + offa - offs
    raise ValueError(coord)


def cut_at(rows, p, t, dt):
    """그 시즌 점으로 잇는 컷 C_p(t): |h − t| ≤ dt인 A·B 줄 중 p를 사이에 둔 가장 가까운 두 점을 log %–점수 꺾은선으로.
    두 점이 p의 ÷1.5~×1.5 밖이면 None. 같은 %가 p면 그 점수들의 중앙."""
    near = [o for o in rows if o["grade"] in "AB" and abs(o["h"] - t) <= dt]
    eq = [o["s"] for o in near if abs(o["x"] - p) < 1e-9]
    if eq:
        return st.median(eq)
    ins = [o for o in near if o["x"] < p]
    outs = [o for o in near if o["x"] > p]
    if not ins or not outs:
        return None
    a = max(ins, key=lambda o: (o["x"], -abs(o["h"] - t)))
    b = min(outs, key=lambda o: (o["x"], abs(o["h"] - t)))
    if a["x"] < p / 1.5 or b["x"] > p * 1.5:
        return None
    w = (math.log(p) - math.log(a["x"])) / (math.log(b["x"]) - math.log(a["x"]))
    return a["s"] + (b["s"] - a["s"]) * w


def log_r(s, p, t, q, dt):
    """지난 시즌 s에서 자리 (p, t) → 최종 q%의 log 비. 잴 수 없으면 None."""
    tv = truth().get((s, "KR", float(q)))
    if tv is None:
        return None
    _, H, _ = info(s)
    if t < 0 or t > H:
        return None
    c = cut_at(full(s), p, t, dt)
    if c is None or c <= 0:
        return None
    return math.log(tv / c)


def lower_bound(rows, q, unknown=True):
    """하한: % 구간의 아래 끝이 q보다 큰(확실히 q% 밖) 한섭 줄의 가장 높은 점수. 시각 등급과 무관. 없으면 None."""
    out = [o["s"] for o in rows if is_kr(o, unknown) and o["x"] - PCT_LO > q]
    return max(out) if out else None


def anchors(rows, unknown=False):
    """닻 후보: 시각 A·B 한섭 줄."""
    return [o for o in rows if o["grade"] in "AB" and is_kr(o, unknown)]


def quant(xs, q):
    """선형 보간 분위 (xs 정렬됨)."""
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    i = int(math.floor(k))
    j = min(i + 1, len(xs) - 1)
    return xs[i] + (xs[j] - xs[i]) * (k - i)


def dist(vals, lb=None):
    """예측 분포 요약 (중앙·80%·95%). lb가 있으면 그 아래를 잘라 낸다 (남는 것이 없으면 lb 한 점)."""
    v = sorted(vals)
    cut = False
    if lb is not None:
        w = [x for x in v if x > lb]
        cut = len(w) < len(v)
        v = w or [lb]
    return {"median": quant(v, 0.5), "i80": [quant(v, 0.1), quant(v, 0.9)], "i95": [quant(v, 0.025), quant(v, 0.975)],
            "n": len(v), "cut": cut}
