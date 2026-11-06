"""WP14d D1 (2026-11-06): 꼴 파일의 ln m 사전 (flood.lnm_prior)을 최근 k시즌 창으로 바꾼 새 판을 쓴다.

맞힐 시즌마다 그 시즌 꼴 파일 칸의 lnm_train (앞 시즌만, 그 시즌 꼴로 맞춘 ln m과 se)만 쓴다.
k는 M3 2단계 창 고르기 (m3.choose_window_pca)와 같은 시간 순 교차검증으로 고른다 — 후보 4·6·10·all,
점수 = 학습 시즌 중 4번째부터 (s2) 그 앞 시즌들의 최근 k창으로 낸 사전에서 ln m_s2의 정규 로그 밀도 (분산 = 창 분산 + se_s2²) 합.
평가 시즌이 3개 미만이면 all. 창 분산은 지금 규칙 그대로 (J < 2 → 0.3², 아니면 max(표본 분산 − se² 평균, 0.01²)).
침수 꼴·컷 꼴·편차 층·m_true 등 다른 칸은 그대로 둔다. lnm_prior에 window = {k, scores, eval}가 더해진다.

  python3 -I tools/solo_raid/lab7/wp14_lnm_window.py --shape r7/wp13/shape_k4_D1.json --out r7/wp14/shape_k4w_D1.json
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import subprocess

import numpy as np

KS = (4, 6, 10, "all")


def snum(s):
    return int(s[1:])


def window_prior(vals):
    """vals = [(lnm, se)] -> lnm_prior 꼴 (wp13_flood_shape와 같은 규칙)."""
    lnm = np.array([v[0] for v in vals], float)
    se = np.array([v[1] for v in vals], float)
    J = len(lnm)
    raw = float(np.var(lnm, ddof=1)) if J >= 2 else None
    noise = float(np.mean(se ** 2))
    var = 0.3 ** 2 if J < 2 else max(raw - noise, 0.01 ** 2)
    return {"mean": round(float(lnm.mean()), 6), "var": round(var, 6), "raw_var": None if raw is None else round(raw, 6),
            "noise_var": round(noise, 6), "J": J}


def logn(x, mu, var):
    return -0.5 * (math.log(2 * math.pi * var) + (x - mu) ** 2 / var)


def choose(train):
    """train = {시즌: [lnm, se]} (모두 맞힐 시즌 앞). -> (k, scores, evals)"""
    before = sorted(train, key=snum)
    evals = [s for i, s in enumerate(before) if i >= 3]
    if len(evals) < 3:
        return "all", {}, evals
    scores = {}
    for k in KS:
        tot = 0.0
        for s2 in evals:
            prev = [s for s in before if snum(s) < snum(s2)]
            win = prev if k == "all" else prev[-int(k):]
            p = window_prior([train[s] for s in win])
            tot += logn(train[s2][0], p["mean"], p["var"] + train[s2][1] ** 2)
        scores[str(k)] = round(tot, 4)
    best = max(scores, key=scores.get)
    return (best if best == "all" else int(best)), scores, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.shape, "rb") as f:
        raw = f.read()
    src = json.loads(raw)
    out = copy.deepcopy(src)
    try:
        head = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        head = None
    out["header"]["about"] = src["header"]["about"] + " — WP14d: flood.lnm_prior만 최근 k시즌 창 (k는 앞 시즌만 시간 순 교차검증, 후보 4·6·10·all)"
    out["header"]["made"] = {"tool_head": head, "cmd": "wp14_lnm_window.py --shape " + a.shape + " --out " + a.out,
                             "at_kst": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")}
    out["header"]["inputs"] = {"base_shape": {"file": os.path.basename(a.shape), "sha1": hashlib.sha1(raw).hexdigest()[:12]}, **src["header"]["inputs"]}
    out["header"]["lnm_window_rules"] = {"ks": [str(k) for k in KS], "score": "sum over eval seasons s2 (4th training season on) of log N(lnm_s2; window mean, window var + se_s2^2)",
                                         "var": "J<2 -> 0.3^2 else max(sample var - mean se^2, 0.01^2) (wp13_flood_shape와 같음)", "fallback": "eval < 3 -> all"}
    rows = []
    for s in sorted(out["targets"], key=snum):
        fl = out["targets"][s]["flood"]
        train = {t: v for t, v in fl["lnm_train"].items() if snum(t) < snum(s)}
        assert len(train) == len(fl["lnm_train"]), f"{s}: lnm_train에 맞힐 시즌 뒤 시즌이 있다"
        old = fl["lnm_prior"]
        chk = window_prior(list(train.values()))
        if train and (abs(chk["mean"] - old["mean"]) > 2e-5 or abs(chk["var"] - old["var"]) > 2e-5):
            print(f"  주의 {s}: lnm_train으로 다시 낸 all 사전 {chk['mean']}/{chk['var']}가 파일 {old['mean']}/{old['var']}와 다르다")
        k, scores, evals = choose(train)
        win = sorted(train, key=snum)
        win = win if k == "all" else win[-k:]
        newp = window_prior([train[t] for t in win]) if win else dict(old)
        newp["window"] = {"k": str(k), "seasons": win, "scores": scores, "eval": evals}
        fl["lnm_prior"] = newp
        rows.append((s, len(train), str(k), old["mean"], math.sqrt(old["var"]), newp["mean"], math.sqrt(newp["var"])))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("시즌 J k | all 평균 sd | 창 평균 sd")
    for r in rows:
        print(f"{r[0]} {r[1]:2d} {r[2]:>3} | {r[3]:+.3f} {r[4]:.3f} | {r[5]:+.3f} {r[6]:.3f}")
    with open(a.out, "rb") as f:
        print("sha1", hashlib.sha1(f.read()).hexdigest()[:12])


if __name__ == "__main__":
    main()
