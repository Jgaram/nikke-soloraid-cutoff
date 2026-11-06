"""WP13a 꼴 파일 합치기 (2026-11-04, 조율 세션) — 침수 꼴 (wp13_flood_shape.py)과 컷 꼴 (wp13_cut_shape.py)을 맞힐 시즌별로 묶어 m3.py --shape가 읽는 한 파일로.

  python3 -I tools/solo_raid/lab7/wp13_shape.py [--flood r7/wp13/flood_shape_D1.json] [--cut r7/wp13/cut_shape_D1.json] [--out r7/wp13/shape_D1.json]

계획: docs/solo-raid-method.md 「WP13」. 형식 (형식 v1): {"header": {...}, "targets": {시즌: {"flood": FLOOD, "cut": CUT}}}.
FLOOD·CUT 칸은 두 도구가 낸 맞힐 시즌 항목 그대로 (복사만, 값은 안 바꾼다). 두 파일의 맞힐 시즌이 다르거나, 학습에 맞힐 시즌 이후가 있거나, 침수 학습이 컷 학습 밖이면 멈춘다.
침수는 wp13_flood_emp.py가 낸 파일 (05시 시작 시즌은 WP12 경험 곡선 source emp, S28·S38은 HGAM source hgam — 재현 점검의 대체 규칙).
머리에는 두 입력의 sha1 12자·data_ver·만든 때를 적는다 — m3.py가 이 파일의 sha1을 칸 출력에 남긴다.
침수 칸에 tau_oos를 더한다 (WP13c): 맞힐 시즌 s 앞의 맞힐 시즌 j < s 가 낸 t_true[d]의 제곱평균 (사전 평균 0 둘레) 에서 s의 bootstrap 잡음 tau.noise[d]를 뺀 분산의 sd.
  표본 3 미만이면 sd 0.3 (ln m 사전과 같은 규칙). 뜻: 시즌 안 분산(tau.per_day)이 아니라 앞 시즌 하루 수준 t의 시즌 밖 흩어짐.
공개 규칙: 두 입력이 이미 시각·%·점수·개수만 담는다 (사람·사진 열쇠 없음).
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import math
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

R13 = os.path.join(C.ROOT, "data", "solo-raid", "analysis", "r7", "wp13")


def sha12(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def git_head():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def tau_oos(tf, s):
    """맞힐 시즌 s의 시즌 밖 tau: 앞 맞힐 시즌 j < s 의 t_true[d] 제곱평균 − s의 잡음 분산 (하한 0.01², 표본 < 3 이면 0.3²)."""
    n = int(s[1:])
    js = [j for j in sorted(tf, key=lambda x: int(x[1:])) if int(j[1:]) < n]
    assert all(int(j[1:]) < n for j in js), "tau_oos 누수"
    noise = [float(v) for v in tf[s]["tau"]["noise"]]
    per, cnt, raw = [], [], []
    for d in range(5):
        v = []
        for j in js:
            tt = tf[j].get("t_true")
            if tt is None:
                continue
            for x in (tt[d:] if d == 4 else tt[d:d + 1]):     # 5일째 이후 (S28 6일째)는 5일 칸에 묶음
                if x is not None:
                    v.append(float(x))
        cnt.append(len(v))
        if len(v) < 3:
            raw.append(None)
            per.append(0.3)
            continue
        r = sum(x * x for x in v) / len(v)
        raw.append(round(r, 6))
        per.append(round(math.sqrt(max(r - noise[d], 0.01 ** 2)), 5))
    return {"per_day": per, "n": cnt, "raw": raw, "noise": noise,
            "rule": "앞 맞힐 시즌 j<s 의 t_true[d] 제곱평균 − s의 tau.noise[d] (하한 0.01²), 표본 3 미만이면 0.3. 5일째 이후는 한 칸으로 묶음."}


def merge(flood_path, cut_path):
    fl, cu = jload(flood_path), jload(cut_path)
    tf, tc = fl["targets"], cu["targets"]
    if sorted(tf) != sorted(tc):
        sys.exit(f"맞힐 시즌이 다르다: 침수 {sorted(tf)} · 컷 {sorted(tc)}")
    for s in tf:
        # 학습 시즌은 맞힐 시즌보다 앞이어야 한다. 침수 경험 곡선 (source emp)은 05시 시작 시즌만 써서 컷보다 적을 수 있다 — 침수 ⊆ 컷이면 된다.
        n = int(s[1:])
        ff, cc = list(tf[s].get("train") or []), list(tc[s].get("train") or [])
        late = [x for x in ff + cc if int(x[1:]) >= n]
        if late:
            sys.exit(f"{s} 학습에 맞힐 시즌 이후가 있다: {sorted(set(late))}")
        if not set(ff) <= set(cc):
            sys.exit(f"{s} 침수 학습 시즌이 컷 학습 시즌 밖이다: 침수 {ff} · 컷 {cc}")
    head = {"about": "WP13 꼴 파일 (형식 v1) — 맞힐 시즌마다 앞 시즌만으로 낸 침수 꼴 P̄·컷 꼴 G와 편차 층 사전",
            "data_ver": C.DATA_VER, "format": 1,
            "made": {"tool_head": git_head(), "cmd": " ".join(["wp13_shape.py"] + sys.argv[1:]),
                     "at_kst": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")},
            "inputs": {"flood": {"file": os.path.basename(flood_path), "sha1": sha12(flood_path)},
                       "cut": {"file": os.path.basename(cut_path), "sha1": sha12(cut_path)}},
            "flood_rules": fl.get("header", {}).get("rules"), "cut_rules": cu.get("header", {}).get("rules"),
            "flood_source": {s: tf[s].get("source", "hgam") for s in sorted(tf, key=lambda x: int(x[1:]))}}
    for k, d in (("flood", fl), ("cut", cu)):
        dv = d.get("header", {}).get("data_ver")
        if dv is not None and dv != C.DATA_VER:
            sys.exit(f"{k} 파일 data_ver {dv} ≠ {C.DATA_VER}")
    return {"header": head, "targets": {s: {"flood": dict(copy.deepcopy(tf[s]), tau_oos=tau_oos(tf, s)), "cut": tc[s]} for s in sorted(tf, key=lambda x: int(x[1:]))}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--flood", default=os.path.join(R13, "flood_shape_D1.json"))
    ap.add_argument("--cut", default=os.path.join(R13, "cut_shape_D1.json"))
    ap.add_argument("--out", default=os.path.join(R13, "shape_D1.json"))
    a = ap.parse_args()
    out = merge(a.flood, a.cut)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, a.out)
    print(a.out, len(out["targets"]), "시즌", os.path.getsize(a.out), "바이트")


if __name__ == "__main__":
    main()
