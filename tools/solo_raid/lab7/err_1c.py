"""WP1 1c (2026-10-28) — 글로만 적은 %(관측 표 kind "text")의 값 오차를 사진과 견줘 잰다.

  python3 -I tools/solo_raid/lab7/err_1c.py --obs DIR --out OUT.json [--md OUT.md] [--boot 2000]

DIR = 관측 표 끝난 시즌 전체 (DIR/full/S{19..42}.jsonl.gz, obs_table.py build --full).
정답(answer_key·truth·criteria)은 읽지 않는다. 오차는 글의 %와 같은 글에 붙은 사진의 %를 견줘서만 잰다.
core.jsonl은 (사이트, 글 열쇠)로 「같은 글의 사진」을 찾는 데만 쓴다 — 글 번호는 출력에 내지 않는다.

항목
  1 양        시즌·종류·t_kind·self별 줄 수, 점수가 있는 비율, 서버 라벨 비율, use 안팎
  2 검증 표본 link로 사진과 이어진 글 줄: 차이·정확·반올림/버림/대충·자릿수별·시각 차·self·pid
  2b 이음 편향 이음(post_link.py)은 「반올림·버림까지 맞아야」 잇는다 → 이어진 줄만 보면 큰 오차가 안 보인다.
              같은 글에 점수가 맞는 사진이 있는데 %가 안 맞아 이어지지 않은 줄(E 표본)로 꼬리를 본다.
  3 대표성    이어지지 않은 글 vs 이어진 글 (자릿수·값·t_kind·점수·끝자리)
  4 모형용 꼴 자릿수별 구간 관측 + 오염 혼합, 로그가능도·AIC
  5 값어치    점수 있는 글 줄이 사진 줄을 시즌×6시간 칸·목표 % 근처에서 얼마나 보태는지
  6 시즌 변동 시즌별 값, 이질성 검정, 늘어나는 창(앞 시즌만)
불확실성: 비율은 윌슨, 사진-글 짝에서 나온 비율·평균은 (시즌, pid) 묶음 부트스트랩(seed 고정).
"""
import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import obs_table as O  # noqa: E402

SEASONS = list(range(19, 43))
SEED = 20261028
EPS = 1e-9


def r4(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return r4(o)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def prop(k, n):
    """윌슨 구간 붙은 비율."""
    return {"k": int(k), "n": int(n), "p": (k / n if n else None), "ci": wilson(k, n)}


def dec(v):
    """소수 자릿수 (0, 1, 2, 3), 그 이상은 9."""
    for d in range(4):
        if abs(round(v, d) - v) < EPS:
            return d
    return 9


def rnd(v, d):
    """반올림(반이 올림)."""
    f = 10 ** d
    return math.floor(v * f + 0.5 + 1e-7) / f


def flo(v, d):
    f = 10 ** d
    return math.floor(v * f + 1e-7) / f


def num_match(claim, val, tol):
    """post_link.num_match와 같다: |차| ≤ tol 이거나 val을 claim 자리로 반올림·버림한 값이 같다."""
    if abs(claim - val) <= tol + EPS:
        return True
    d = dec(claim)
    f = 10 ** d
    return abs(round(val, d) - claim) < EPS or abs(math.floor(val * f + 1e-9) / f - claim) < EPS


class Clusters:
    """(시즌, pid) 묶음 부트스트랩. 행마다 묶음 번호를 주고, 묶음별 합을 다시 뽑는다."""

    def __init__(self, keys):
        ids = {}
        self.cid = np.array([ids.setdefault(k, len(ids)) for k in keys], dtype=int)
        self.m = len(ids)

    def sums(self, arr):
        return np.bincount(self.cid, weights=np.asarray(arr, float), minlength=self.m)


def boot_ratio(cl, num, den, B, seed):
    """sum(num)/sum(den)의 묶음 부트스트랩 95% 구간."""
    nu, de = cl.sums(num), cl.sums(den)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, cl.m, size=(B, cl.m))
    a, b = nu[idx].sum(1), de[idx].sum(1)
    ok = b > 0
    v = a[ok] / b[ok]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def ratio(cl, num, den, B, seed):
    n = float(np.sum(den))
    k = float(np.sum(num))
    return {"k": k, "n": n, "p": (k / n if n else None), "ci_boot": boot_ratio(cl, num, den, B, seed) if n else None,
            "ci_wilson": wilson(k, n)}


def boot_stat(cl, vals, fn, B, seed):
    """행 값 vals에 fn(배열)을 묶음 단위로 다시 뽑아 구간."""
    vals = np.asarray(vals, float)
    rows = [np.where(cl.cid == c)[0] for c in range(cl.m)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = rng.integers(0, cl.m, size=cl.m)
        ix = np.concatenate([rows[c] for c in pick])
        out.append(fn(vals[ix]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def qtile(vals, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    if len(vals) == 0:
        return None
    return {str(q): float(np.quantile(vals, q)) for q in qs}


def hetero(per):
    """per = {시즌: (k, n)}: 시즌별 비율 차이를 카이제곱으로. 묶음(같은 사람) 효과가 있으면 p가 너무 작게 나온다."""
    from scipy import stats
    items = [(s, k, n) for s, (k, n) in per.items() if n > 0]
    K = sum(k for _, k, _ in items)
    N = sum(n for _, _, n in items)
    p = K / N if N else 0
    if p in (0, 1) or len(items) < 2:
        return {"Q": None, "df": len(items) - 1, "p": None}
    Q = sum((k - n * p) ** 2 / (n * p * (1 - p)) for _, k, n in items)
    df = len(items) - 1
    sd_obs = float(np.sqrt(np.average([(k / n - p) ** 2 for _, k, n in items], weights=[n for _, _, n in items])))
    sd_exp = float(np.sqrt(np.average([p * (1 - p) / n for _, _, n in items], weights=[n for _, _, n in items])))
    return {"Q": Q, "df": df, "p": float(stats.chi2.sf(Q, df)), "pooled": p, "n_seasons": len(items),
            "sd_between_obs": sd_obs, "sd_expected_binomial": sd_exp}


def expanding(per, first=21, last=42):
    """늘어나는 창: 맞힐 시즌 s마다 s 앞 시즌만으로 잰 값. 흔들림과 s 자신의 값이 그 값에서 얼마나 떨어지는지(z)."""
    out = {}
    zs = []
    ests = []
    for s in range(first, last + 1):
        k = sum(per[t][0] for t in per if t < s)
        n = sum(per[t][1] for t in per if t < s)
        ks, ns = per.get(s, (0, 0))
        if n == 0:
            continue
        p = k / n
        ci = wilson(k, n)
        z = None
        if ns > 0 and 0 < p < 1:
            z = (ks / ns - p) / math.sqrt(p * (1 - p) / ns)
            zs.append(z)
        ests.append(p)
        out[f"S{s}"] = {"est": p, "n_prior": n, "ci": ci, "own": (ks / ns if ns else None), "own_n": ns, "z": z}
    summ = {"est_min": min(ests), "est_max": max(ests), "est_sd": float(np.std(ests)),
            "est_first": ests[0], "est_last": ests[-1],
            "z_abs_gt_1.96": int(sum(1 for z in zs if abs(z) > 1.96)), "n_z": len(zs),
            "z_sd": float(np.std(zs)) if zs else None}
    return {"by_season": out, "summary": summ}


# ---------------------------------------------------------------- 불러오기

def load(obs_dir):
    """시즌별 (머리, 줄 목록, 같은 글 열쇠 사전). 글 열쇠는 메모리 안에서만 쓴다."""
    D = {}
    for s in SEASONS:
        head, rows = O.read_cell(os.path.join(obs_dir, "full", f"S{s}.jsonl.gz"))
        core = {}
        with open(os.path.join(C.state_of(f"S{s}"), "core.jsonl"), encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    d = json.loads(l)
                    core[d["key"]] = (d["site"], d["no"])
        for r in rows:
            r["season"] = s
            r["post"] = core.get(r["key"])
            r["cl"] = (s, r["pid"])
        D[s] = (head, rows)
    return D


def digits_of(r):
    d = dec(r["pct"])
    return d if d in (1, 2) else 3


def consistent(t, v, dg):
    """글 값 t가 사진 값 v와 이음 규칙(반올림·버림·±0.01)으로 맞는가."""
    if dg == 2:
        return abs(t - v) <= 0.01 + EPS
    if dg == 1:
        return abs(t - rnd(v, 1)) < EPS or abs(t - flo(v, 1)) < EPS
    return abs(t - v) <= 0.01 + EPS


def klass(t, v, dg):
    """글-사진 짝의 분류. 2자리: exact·plus1·minus1·far. 1자리: both·round_only·floor_only·far."""
    d = t - v
    if dg == 2:
        if abs(d) < EPS:
            return "exact"
        if abs(d - 0.01) < EPS:
            return "plus1"
        if abs(d + 0.01) < EPS:
            return "minus1"
        return "far"
    a = abs(t - rnd(v, 1)) < EPS
    b = abs(t - flo(v, 1)) < EPS
    if a and b:
        return "both"
    if a:
        return "round_only"
    if b:
        return "floor_only"
    return "far"


def dt_bin(x):
    if x is None:
        return "na"
    if x <= 5 / 60 + EPS:
        return "le5min"
    if x <= 0.5 + EPS:
        return "5to30min"
    if x <= 2 + EPS:
        return "30min_2h"
    return "gt2h"


DT_BINS = ["le5min", "5to30min", "30min_2h", "gt2h"]


# ---------------------------------------------------------------- 1 양

def counts(D, B):
    T = {s: [r for r in D[s][1] if r["kind"] == "text"] for s in SEASONS}
    allt = [r for s in SEASONS for r in T[s]]
    out = {"n_text": len(allt), "n_all_rows": sum(len(D[s][1]) for s in SEASONS)}
    out["use_true"] = sum(1 for r in allt if r["use"])
    out["lo_reason"] = dict(collections.Counter(r["lo_reason"] for r in allt))
    out["ckind"] = dict(collections.Counter(r["ckind"] for r in allt))
    out["t_kind"] = dict(collections.Counter(r["t_kind"] for r in allt))
    out["t_grade"] = dict(collections.Counter(r["t_grade"] for r in allt))
    out["self"] = dict(collections.Counter(str(r["self"]) for r in allt))
    out["site"] = dict(collections.Counter(r["site"] for r in allt))
    out["digits"] = dict(collections.Counter(digits_of(r) for r in allt))
    out["with_score"] = prop(sum(1 for r in allt if r["score"] is not None), len(allt))
    out["server_label"] = prop(sum(1 for r in allt if r["server"] is not None), len(allt))
    out["server_label_values"] = dict(collections.Counter(r["server"] for r in allt))
    out["linked"] = prop(sum(1 for r in allt if r["link"]), len(allt))
    out["unique_moments"] = len({(r["season"], r["mid"]) for r in allt})
    out["unique_persons"] = len({(r["season"], r["pid"]) for r in allt})
    out["score_and_server"] = prop(sum(1 for r in allt if r["score"] is not None and r["server"] is not None), len(allt))
    out["with_hp"] = prop(sum(1 for r in allt if r["hp"] is not None), len(allt))
    out["by_digits_x_score"] = {str(d): prop(sum(1 for r in allt if digits_of(r) == d and r["score"] is not None),
                                            sum(1 for r in allt if digits_of(r) == d)) for d in (1, 2, 3)}
    sdc = collections.Counter(dec(r["score"]) for r in allt if r["score"] is not None)
    out["score_decimals"] = {str(k): v for k, v in sorted(sdc.items())}
    out["funnel"] = {
        "text": len(allt),
        "with_score": sum(1 for r in allt if r["score"] is not None),
        "with_score_unlinked": sum(1 for r in allt if r["score"] is not None and not r["link"]),
        "with_score_unlinked_d2": sum(1 for r in allt if r["score"] is not None and not r["link"] and digits_of(r) == 2),
        "with_score_unlinked_d2_server": sum(1 for r in allt if r["score"] is not None and not r["link"] and digits_of(r) == 2 and r["server"] is not None),
        "with_score_unlinked_score_decimal": sum(1 for r in allt if r["score"] is not None and not r["link"] and dec(r["score"]) >= 1),
        "with_score_unlinked_score_decimal_d2": sum(1 for r in allt if r["score"] is not None and not r["link"] and dec(r["score"]) >= 1 and digits_of(r) == 2),
    }
    # use 안팎: 글은 use=False 뿐이다. 같은 사람 사진 use 줄과의 관계는 5에서
    per = {}
    for s in SEASONS:
        t = T[s]
        per[f"S{s}"] = {"n": len(t), "linked": sum(1 for r in t if r["link"]), "with_score": sum(1 for r in t if r["score"] is not None),
                        "score_unlinked": sum(1 for r in t if r["score"] is not None and not r["link"]),
                        "server_label": sum(1 for r in t if r["server"] is not None),
                        "d1": sum(1 for r in t if digits_of(r) == 1), "d2": sum(1 for r in t if digits_of(r) == 2),
                        "stated": sum(1 for r in t if r["t_kind"] == "stated"),
                        "grade": dict(collections.Counter(r["t_grade"] for r in t)),
                        "n_photo_rows": sum(1 for r in D[s][1] if r["kind"] != "text"),
                        "n_photo_use": sum(1 for r in D[s][1] if r["kind"] != "text" and r["use"]),
                        "n_photo_use_indep": sum(1 for r in D[s][1] if r["kind"] != "text" and r["use_indep"])}
    out["by_season"] = per
    return out


# ---------------------------------------------------------------- 2 검증 표본

def pairs_of(D):
    P = []
    for s in SEASONS:
        by = {r["oi"]: r for r in D[s][1] if r["kind"] != "text"}
        for r in D[s][1]:
            if r["kind"] == "text" and r["link"]:
                p = by[r["link"][0]]
                dg = digits_of(r)
                dt = None if (r["hp"] is None or p["h"] is None) else r["hp"] - p["h"]
                P.append({"t": r, "p": p, "dg": dg, "cls": klass(r["pct"], p["pct"], dg), "dt": dt,
                          "dtb": dt_bin(dt) if p["kind"] in ("A", "B") else "na", "season": s,
                          "d": r["pct"] - p["pct"]})
    return P


def linked_stats(P, B):
    cl = Clusters([x["t"]["cl"] for x in P])
    out = {"n": len(P), "n_clusters": cl.m, "multi_link_rows": sum(1 for x in P if len(x["t"]["link"]) > 1),
           "photo_back_links": None}
    out["photo_kind"] = dict(collections.Counter(x["p"]["kind"] for x in P))
    out["photo_use"] = dict(collections.Counter(str(x["p"]["use"]) for x in P))
    out["photo_self"] = dict(collections.Counter(str(x["p"]["self"]) for x in P))
    pd = collections.Counter((x["dg"], dec(x["p"]["pct"])) for x in P)
    out["photo_digits_by_text_digits"] = {f"text{a}_photo{b}": v for (a, b), v in sorted(pd.items())}
    cnt = collections.Counter((x["season"], x["p"]["oi"]) for x in P)
    out["texts_per_photo"] = {"photos_linked": len(cnt), "photos_with_2plus_texts": sum(1 for v in cnt.values() if v > 1),
                              "max": max(cnt.values())}
    out["same_pid"] = prop(sum(1 for x in P if x["t"]["pid"] == x["p"]["pid"]), len(P))
    out["same_hp"] = prop(sum(1 for x in P if x["t"]["hp"] is not None and x["p"]["hp"] is not None
                              and abs(x["t"]["hp"] - x["p"]["hp"]) < 1e-5), len(P))
    out["score_pair"] = dict(collections.Counter((x["t"]["score"] is not None, x["p"]["score"] is not None) for x in P).most_common())
    out["score_pair"] = {str(k): v for k, v in out["score_pair"].items()}
    out["digits"] = {str(d): sum(1 for x in P if x["dg"] == d) for d in (1, 2, 3)}
    out["pid_per_cluster"] = qtile([sum(1 for x in P if x["t"]["cl"] == c) for c in set(x["t"]["cl"] for x in P)], (0.5, 0.9, 1.0))

    # 차이 분포 (글 − 사진), 비
    dif = {}
    for key, sel in (("all", lambda x: True), ("d2", lambda x: x["dg"] == 2), ("d1", lambda x: x["dg"] == 1)):
        xs = [x for x in P if sel(x)]
        d = np.array([x["d"] for x in xs])
        cc = Clusters([x["t"]["cl"] for x in xs])
        rt = np.array([x["t"]["pct"] / x["p"]["pct"] for x in xs if x["p"]["pct"] > 0])
        dif[key] = {"n": len(xs), "diff_quantiles": qtile(d), "diff_mean": float(d.mean()),
                    "diff_mean_ci_boot": boot_stat(cc, d, np.mean, B, SEED + 1),
                    "diff_abs_max": float(np.abs(d).max()), "ratio_quantiles": qtile(rt),
                    "ratio_n": int(len(rt)),
                    "diff_hist": {str(k): v for k, v in sorted(collections.Counter(round(float(v), 2) for v in d).items())}}
    out["diff"] = dif

    # 분류 (정확·±0.01 / 반올림·버림·둘 다)
    cls = {}
    for dg in (2, 1):
        xs = [x for x in P if x["dg"] == dg]
        cls[str(dg)] = dict(collections.Counter(x["cls"] for x in xs))
    out["class_by_digits"] = cls
    x2 = [x for x in P if x["dg"] == 2]
    c2 = Clusters([x["t"]["cl"] for x in x2])
    out["exact_d2"] = ratio(c2, [x["cls"] == "exact" for x in x2], [1] * len(x2), B, SEED + 2)
    out["plus1_d2"] = ratio(c2, [x["cls"] == "plus1" for x in x2], [1] * len(x2), B, SEED + 3)
    out["minus1_d2"] = ratio(c2, [x["cls"] == "minus1" for x in x2], [1] * len(x2), B, SEED + 4)
    x1 = [x for x in P if x["dg"] == 1]
    inf = [x for x in x1 if x["cls"] in ("round_only", "floor_only")]
    c1 = Clusters([x["t"]["cl"] for x in inf])
    out["d1_informative"] = {"n_both": sum(1 for x in x1 if x["cls"] == "both"), "n_informative": len(inf),
                             "round_share_of_informative": ratio(c1, [x["cls"] == "round_only" for x in inf], [1] * len(inf), B, SEED + 5)}
    # 「대충」: 1자리 글 값의 끝자리 분포 vs 사진 값을 반올림한 1자리의 끝자리 분포 (같은 짝)
    last_t = collections.Counter(int(round(x["t"]["pct"] * 10)) % 10 for x in x1)
    last_r = collections.Counter(int(round(rnd(x["p"]["pct"], 1) * 10)) % 10 for x in x1)
    last_f = collections.Counter(int(round(flo(x["p"]["pct"], 1) * 10)) % 10 for x in x1)
    out["d1_last_digit"] = {"text": dict(sorted(last_t.items())), "photo_rounded": dict(sorted(last_r.items())), "photo_floored": dict(sorted(last_f.items())),
                            "text_ends5": prop(last_t[5], len(x1)), "photo_round_ends5": prop(last_r[5], len(x1))}
    # 글이 정수·0.5 단위인 비율 (자릿수와 상관없이)
    out["half_unit"] = {"d1_ends5": prop(sum(1 for x in x1 if int(round(x["t"]["pct"] * 10)) % 5 == 0), len(x1)),
                        "d2_ends0_or5": prop(sum(1 for x in x2 if int(round(x["t"]["pct"] * 100)) % 5 == 0), len(x2)),
                        "d2_photo_ends0_or5": prop(sum(1 for x in x2 if int(round(x["p"]["pct"] * 100)) % 5 == 0), len(x2))}

    # 시각 차 (글 게시 hp − 사진 촬영 h; 사진 A·B만, C는 h=hp라 0)
    byd = {}
    for kind in ("A", "B"):
        sub = [x for x in P if x["p"]["kind"] == kind and x["dt"] is not None]
        row = {"n": len(sub), "dt_quantiles": qtile([x["dt"] for x in sub], (0.1, 0.5, 0.9, 0.99)), "bins": {}}
        for b in DT_BINS:
            xb = [x for x in sub if x["dtb"] == b]
            x2b = [x for x in xb if x["dg"] == 2]
            x1b = [x for x in xb if x["dg"] == 1 and x["cls"] in ("round_only", "floor_only")]
            row["bins"][b] = {"n": len(xb), "n_d2": len(x2b),
                              "d2_exact": prop(sum(1 for x in x2b if x["cls"] == "exact"), len(x2b)),
                              "n_d1_informative": len(x1b),
                              "d1_round_share": prop(sum(1 for x in x1b if x["cls"] == "round_only"), len(x1b)),
                              "abs_diff_mean": float(np.mean([abs(x["d"]) for x in xb])) if xb else None}
        # 시각 차와 |차| 상관 (스피어만)
        from scipy import stats
        if len(sub) > 5:
            rho = stats.spearmanr([x["dt"] for x in sub], [abs(x["d"]) for x in sub])
            row["spearman_dt_absdiff"] = {"rho": float(rho[0]), "p": float(rho[1]), "n": len(sub)}
        byd[kind] = row
    out["by_dt"] = byd

    # 본인/남: 글은 모두 본인(self) 글. 사진 쪽 self 별
    bys = {}
    for sv in (True, False):
        xs = [x for x in P if bool(x["p"]["self"]) == sv]
        x2s = [x for x in xs if x["dg"] == 2]
        x1s = [x for x in xs if x["dg"] == 1 and x["cls"] in ("round_only", "floor_only")]
        bys[str(sv)] = {"n": len(xs), "n_d2": len(x2s), "d2_exact": prop(sum(1 for x in x2s if x["cls"] == "exact"), len(x2s)),
                        "n_d1_informative": len(x1s), "d1_round_share": prop(sum(1 for x in x1s if x["cls"] == "round_only"), len(x1s)),
                        "with_score_both": prop(sum(1 for x in xs if x["t"]["score"] is not None and x["p"]["score"] is not None), len(xs)),
                        "photo_kind": dict(collections.Counter(x["p"]["kind"] for x in xs))}
    out["by_photo_self"] = bys
    out["text_self"] = dict(collections.Counter(str(x["t"]["self"]) for x in P))

    # 점수 차 (둘 다 점수가 있는 짝): 같은 기록인가
    sp = [x for x in P if x["t"]["score"] is not None and x["p"]["score"] is not None]
    sd = np.array([x["t"]["score"] - x["p"]["score"] for x in sp])
    out["score_diff_eok"] = {"n": len(sp), "quantiles": qtile(sd), "abs_gt_0.1": prop(int(np.sum(np.abs(sd) > 0.1 + EPS)), len(sp)),
                             "abs_gt_1": prop(int(np.sum(np.abs(sd) > 1.0)), len(sp))}
    return out


def link_by_from_raw(P):
    """post_claims의 link_by 가짓수 (시즌 상관없이 합). 글 번호는 쓰지 않는다."""
    cnt = collections.Counter()
    for s in SEASONS:
        pc = {}
        with open(os.path.join(C.state_of(f"S{s}"), "post_claims.jsonl"), encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    d = json.loads(l)
                    pc[d["i"]] = d
        for x in P:
            if x["season"] == s:
                d = pc[x["t"]["i"]]
                cnt[d["link_by"]] += 1
                cnt["n_link_all>1"] += 1 if len(d["link_all"]) > 1 else 0
    return dict(cnt)


def raw_claim_drops():
    """kind self 글 가운데 정수 %(int_pct)로 빠진 줄 — 표에 없는 「3퍼」 무리의 크기."""
    cnt = collections.Counter()
    for s in SEASONS:
        with open(os.path.join(C.state_of(f"S{s}"), "post_claims.jsonl"), encoding="utf-8") as f:
            for l in f:
                if not l.strip():
                    continue
                d = json.loads(l)
                if d["kind"] != "self":
                    continue
                cnt["self_total"] += 1
                if d["keep"]:
                    cnt["self_keep"] += 1
                for x in d["drop"]:
                    cnt["drop:" + x] += 1
                if d["pct"] is not None and d["pct"] == int(d["pct"]) and not d["keep"] and "post_before" not in d["drop"]:
                    cnt["int_pct_in_season"] += 1
    return dict(cnt)


# ---------------------------------------------------------------- 2b 이음 편향: 점수 맞는 같은 글 사진

def e_sample(D, self_only=False):
    """E 표본: 점수가 있는 글 줄 가운데 같은 글에 점수가 맞는 사진(점수·% 있음, exclude 없음)이 있는 줄.
    가장 가까운 % 의 사진을 짝으로. 이어진 줄과 안 이어진 줄을 함께 담는다. 점수가 정수 억이면 느슨(loose), 소수가 있으면 엄격(strict)."""
    E = []
    for s in SEASONS:
        rows = D[s][1]
        ph = collections.defaultdict(list)
        for r in rows:
            if r["kind"] != "text" and r["pct"] is not None and r["score"] is not None and r["post"] is not None \
                    and (r["self"] or not self_only):
                ph[r["post"]].append(r)
        for r in rows:
            if r["kind"] != "text" or r["score"] is None or r["post"] is None:
                continue
            cand = [p for p in ph.get(r["post"], []) if num_match(r["score"], p["score"], 0.1)]
            if not cand:
                continue
            dg = digits_of(r)
            best = min(cand, key=lambda p: (not consistent(r["pct"], p["pct"], dg), abs(p["pct"] - r["pct"])))
            sd = dec(r["score"])
            dt = None if (r["hp"] is None or best["h"] is None) else r["hp"] - best["h"]
            E.append({"t": r, "p": best, "dg": dg, "strict": sd in (1, 2), "cls": klass(r["pct"], best["pct"], dg),
                      "linked": bool(r["link"]), "d": r["pct"] - best["pct"], "season": s, "n_cand": len(cand),
                      "dtb": dt_bin(dt) if best["kind"] in ("A", "B") else "na", "dt": dt})
    return E


def mag_bin(a):
    if a <= 0.1 + EPS:
        return "<=0.1"
    if a <= 0.5:
        return "0.1-0.5"
    if a <= 2:
        return "0.5-2"
    return ">2"


def e_stats(E, B):
    out = {"n": len(E)}
    for name, sel in (("strict", lambda x: x["strict"]), ("loose", lambda x: True), ("loose_only", lambda x: not x["strict"])):
        xs = [x for x in E if sel(x)]
        row = {"n": len(xs), "linked": sum(1 for x in xs if x["linked"]), "unlinked": sum(1 for x in xs if not x["linked"])}
        for dg in (1, 2):
            xd = [x for x in xs if x["dg"] == dg]
            cc = Clusters([x["t"]["cl"] for x in xd]) if xd else None
            far = [x for x in xd if x["cls"] == "far"]
            r = {"n": len(xd), "far": ratio(cc, [x["cls"] == "far" for x in xd], [1] * len(xd), B, SEED + 10 + dg) if xd else None,
                 "far_mag": dict(collections.Counter(mag_bin(abs(x["d"])) for x in far)),
                 "far_sign_pos": prop(sum(1 for x in far if x["d"] > 0), len(far)),
                 "far_dt": dict(collections.Counter(x["dtb"] for x in far)),
                 "class": dict(collections.Counter(x["cls"] for x in xd))}
            row[f"d{dg}"] = r
        out[name] = row
    for name, sel in (("strict", lambda x: x["strict"]), ("loose", lambda x: True)):
        xd = [x for x in E if sel(x) and x["dg"] == 1]
        e5 = [x for x in xd if int(round(x["t"]["pct"] * 10)) % 5 == 0]
        ne = [x for x in xd if int(round(x["t"]["pct"] * 10)) % 5 != 0]
        out[f"d1_far_by_ends5_{name}"] = {"ends5": prop(sum(1 for x in e5 if x["cls"] == "far"), len(e5)),
                                          "other": prop(sum(1 for x in ne if x["cls"] == "far"), len(ne))}
    h5 = collections.defaultdict(collections.Counter)
    for x in E:
        if x["dg"] == 1 and not x["strict"]:
            k = "ends5" if int(round(x["t"]["pct"] * 10)) % 5 == 0 else "other"
            h5[k][min(10, max(-10, int(round(x["d"] * 10))))] += 1   # 0.1%p 칸, ±1 로 자름
    out["d1_loose_signed_diff_hist_0.1step"] = {k: {str(a): b for a, b in sorted(v.items())} for k, v in h5.items()}
    # 시각 차별 far 비율 (strict, 사진 A·B)
    byd = {}
    for b in DT_BINS:
        xs = [x for x in E if x["strict"] and x["dtb"] == b]
        byd[b] = prop(sum(1 for x in xs if x["cls"] == "far"), len(xs))
    out["strict_far_by_dt"] = byd
    out["strict_far_by_photo_self"] = {str(sv): prop(sum(1 for x in E if x["strict"] and bool(x["p"]["self"]) == sv and x["cls"] == "far"),
                                                     sum(1 for x in E if x["strict"] and bool(x["p"]["self"]) == sv)) for sv in (True, False)}
    out["far_by_n_cand"] = {"1": prop(sum(1 for x in E if x["strict"] and x["n_cand"] == 1 and x["cls"] == "far"), sum(1 for x in E if x["strict"] and x["n_cand"] == 1)),
                            "2+": prop(sum(1 for x in E if x["strict"] and x["n_cand"] > 1 and x["cls"] == "far"), sum(1 for x in E if x["strict"] and x["n_cand"] > 1))}
    return out


def post_photo_stats(D):
    """이어지지 않은 글 줄이 같은 글에 사진을 가졌는가 (이음이 안 된 까닭의 윤곽). 글 번호는 쓰지 않는다."""
    n = collections.Counter()
    for s in SEASONS:
        rows = D[s][1]
        ph = collections.defaultdict(list)
        for r in rows:
            if r["kind"] != "text" and r["pct"] is not None and r["post"] is not None:
                ph[r["post"]].append(r)
        for r in rows:
            if r["kind"] != "text" or r["link"]:
                continue
            n["unlinked"] += 1
            ps = ph.get(r["post"], []) if r["post"] is not None else []
            if ps:
                n["with_post_photo"] += 1
                dg = digits_of(r)
                if any(consistent(r["pct"], p["pct"], dg) for p in ps):
                    n["pct_consistent_photo_but_unlinked"] += 1
                else:
                    n["post_photo_pct_inconsistent"] += 1
                    m = min(abs(r["pct"] - p["pct"]) for p in ps)
                    n["incons_mag:" + mag_bin(m)] += 1
            if r["score"] is not None:
                n["unlinked_with_score"] += 1
                if ps:
                    n["unlinked_with_score_with_post_photo"] += 1
    return dict(n)


# ---------------------------------------------------------------- 4 모형 꼴

def ll_d1(xs, params):
    """1자리 글. 사진 값 v(둘째 자리 k). 반올림 t_r, 버림 t_f. 모형: (1-e)[w t_r + (1-w) t_f] + e · 균등(±0.5%p 안의 0.1 간격 11칸).
    반환 로그가능도."""
    w, e = params
    tot = 0.0
    for x in xs:
        t = x["t"]["pct"]
        v = x["p"]["pct"]
        tr, tf = rnd(v, 1), flo(v, 1)
        pr = (1 - e) * (w * (abs(t - tr) < EPS) + (1 - w) * (abs(t - tf) < EPS)) + e / 11.0 * (abs(t - tr) <= 0.5 + EPS)
        tot += math.log(max(pr, 1e-12))
    return tot


def ll_d2(xs, params):
    """2자리 글. (1-e)[(1-dl) 같음 + dl/2 (+1) + dl/2 (-1)] + e · 균등(±1%p 안 0.01 간격 201칸)."""
    dl, e = params
    tot = 0.0
    for x in xs:
        d = round(x["d"], 4)
        base = (1 - dl) * (abs(d) < EPS) + dl / 2 * (abs(abs(d) - 0.01) < EPS)
        pr = (1 - e) * base + e / 201.0 * (abs(d) <= 1.0 + EPS)
        tot += math.log(max(pr, 1e-12))
    return tot


def grid_fit(fn, xs, grids):
    best = None
    for p in grids:
        l = fn(xs, p)
        if best is None or l > best[0]:
            best = (l, p)
    return best


def model_fit(E, P):
    """E(점수로만 고른 표본)에서 모형 비교. 1자리: 반올림만·버림만·혼합·혼합+오염. 2자리: 정확만·±0.01·±0.01+오염."""
    out = {}
    for name, sel in (("strict", lambda x: x["strict"]), ("loose", lambda x: True)):
        xs1 = [x for x in E if sel(x) and x["dg"] == 1]
        xs2 = [x for x in E if sel(x) and x["dg"] == 2]
        r = {}
        if xs1:
            ws = [i / 20 for i in range(21)]
            es = [0.0] + [i / 200 for i in range(1, 80)]
            m = {}
            m["round_only"] = (ll_d1(xs1, (1.0, 0.0)), 0)
            m["floor_only"] = (ll_d1(xs1, (0.0, 0.0)), 0)
            l, p = grid_fit(ll_d1, xs1, [(w, 0.0) for w in ws])
            m["mix_round_floor"] = (l, 1, {"w_round": p[0]})
            l, p = grid_fit(ll_d1, xs1, [(w, e) for w in ws for e in es])
            m["mix_plus_contam"] = (l, 2, {"w_round": p[0], "eps": p[1]})
            l0, p0 = grid_fit(ll_d1, xs1, [(1.0, e) for e in es])
            m["round_plus_contam"] = (l0, 1, {"eps": p0[1]})
            r["d1"] = {"n": len(xs1), "models": {k: {"loglik": v[0], "k": v[1], "aic": -2 * v[0] + 2 * v[1], **(v[2] if len(v) > 2 else {})} for k, v in m.items()}}
        if xs2:
            ds = [i / 200 for i in range(0, 100)]
            es = [0.0] + [i / 200 for i in range(1, 80)]
            m = {}
            m["exact_only"] = (ll_d2(xs2, (0.0, 0.0)), 0)
            l, p = grid_fit(ll_d2, xs2, [(d, 0.0) for d in ds])
            m["exact_pm1"] = (l, 1, {"delta": p[0]})
            l, p = grid_fit(ll_d2, xs2, [(d, e) for d in ds for e in es])
            m["exact_pm1_plus_contam"] = (l, 2, {"delta": p[0], "eps": p[1]})
            r["d2"] = {"n": len(xs2), "models": {k: {"loglik": v[0], "k": v[1], "aic": -2 * v[0] + 2 * v[1], **(v[2] if len(v) > 2 else {})} for k, v in m.items()}}
        out[name] = r
    return out


# ---------------------------------------------------------------- 3 대표성

def lastdig_dist(rows, dg):
    sc = 10 if dg == 1 else 100
    c = collections.Counter(int(round(r["pct"] * sc)) % 10 for r in rows)
    n = sum(c.values())
    return {"n": n, "dist": {str(k): c.get(k, 0) for k in range(10)}}


def representativeness(D, B):
    from scipy import stats
    T = [r for s in SEASONS for r in D[s][1] if r["kind"] == "text"]
    L = [r for r in T if r["link"]]
    U = [r for r in T if not r["link"]]
    out = {"n_linked": len(L), "n_unlinked": len(U)}

    def share(rows, f):
        return len([r for r in rows if f(r)]) / len(rows)

    feats = {"d1": lambda r: digits_of(r) == 1, "d2": lambda r: digits_of(r) == 2, "stated": lambda r: r["t_kind"] == "stated",
             "grade_A": lambda r: r["t_grade"] == "A", "grade_B": lambda r: r["t_grade"] == "B", "grade_C": lambda r: r["t_grade"] == "C",
             "score": lambda r: r["score"] is not None, "server_label": lambda r: r["server"] is not None,
             "site_dc": lambda r: r["site"] == "dc", "day1": lambda r: r["day"] == 1, "day_none": lambda r: r["day"] is None,
             "pct_lt1": lambda r: r["pct"] < 1, "pct_1_3": lambda r: 1 <= r["pct"] < 3, "pct_3_10": lambda r: 3 <= r["pct"] < 10,
             "pct_ge10": lambda r: r["pct"] >= 10}
    comp = {}
    for k, f in feats.items():
        a, b = sum(1 for r in L if f(r)), sum(1 for r in U if f(r))
        tab = [[a, len(L) - a], [b, len(U) - b]]
        try:
            pv = float(stats.chi2_contingency(tab)[1]) if min(a + b, len(L) + len(U) - a - b) > 0 else None
        except ValueError:
            pv = None
        comp[k] = {"linked": prop(a, len(L)), "unlinked": prop(b, len(U)), "chi2_p": pv}
    out["share"] = comp
    out["ckind"] = {"linked": dict(collections.Counter(r["ckind"] for r in L)), "unlinked": dict(collections.Counter(r["ckind"] for r in U))}
    # 값 분포
    pl, pu = np.array([r["pct"] for r in L]), np.array([r["pct"] for r in U])
    ks = stats.ks_2samp(pl, pu)
    out["pct_value"] = {"linked_q": qtile(pl), "unlinked_q": qtile(pu), "ks_D": float(ks.statistic), "ks_p": float(ks.pvalue)}
    # 로그 값 분포 (낮은 %는 로그로)
    ks2 = stats.ks_2samp(np.log(np.clip(pl, 0.01, None)), np.log(np.clip(pu, 0.01, None)))
    out["pct_value"]["ks_log_D"] = float(ks2.statistic)
    out["pct_value"]["ks_log_p"] = float(ks2.pvalue)
    # 끝자리 (대충 여부)
    out["last_digit"] = {}
    for dg in (1, 2):
        out["last_digit"][str(dg)] = {"linked": lastdig_dist([r for r in L if digits_of(r) == dg], dg),
                                      "unlinked": lastdig_dist([r for r in U if digits_of(r) == dg], dg)}
        a = out["last_digit"][str(dg)]
        tab = np.array([[a["linked"]["dist"][str(k)] for k in range(10)], [a["unlinked"]["dist"][str(k)] for k in range(10)]])
        tab = tab[:, tab.sum(0) > 0]
        out["last_digit"][str(dg)]["chi2_p"] = float(stats.chi2_contingency(tab)[1])
    # 사진 use 줄의 끝자리 (참고)
    PH = [r for s in SEASONS for r in D[s][1] if r["kind"] != "text" and r["use"] and r["pct"] is not None]
    out["photo_use_last_digit_d2"] = lastdig_dist(PH, 2)
    ends = lambda rows, dg: sum(1 for r in rows if digits_of(r) == dg and int(round(r["pct"] * (10 if dg == 1 else 100))) % 5 == 0)
    out["heaping_ends_0_5"] = {
        "d1_linked": prop(ends(L, 1), sum(1 for r in L if digits_of(r) == 1)),
        "d1_unlinked": prop(ends(U, 1), sum(1 for r in U if digits_of(r) == 1)),
        "d2_linked": prop(ends(L, 2), sum(1 for r in L if digits_of(r) == 2)),
        "d2_unlinked": prop(ends(U, 2), sum(1 for r in U if digits_of(r) == 2)),
        "d2_photo_use": prop(ends(PH, 2) if False else sum(1 for r in PH if int(round(r["pct"] * 100)) % 5 == 0), len(PH))}
    # 이어진 비율이 시즌·점수 여부로 다른가
    out["linked_share_by_score"] = {"with_score": prop(sum(1 for r in T if r["score"] is not None and r["link"]), sum(1 for r in T if r["score"] is not None)),
                                    "no_score": prop(sum(1 for r in T if r["score"] is None and r["link"]), sum(1 for r in T if r["score"] is None))}
    return out


# ---------------------------------------------------------------- 5 값어치

BANDS = {"1pct": (0.7, 1.3), "3pct": (2.1, 3.9), "10pct": (7.0, 13.0)}


def value_part(D):
    out = {}
    # 사진 기준 집합: use_indep 이고 load_obs 거르기를 지난 줄 (lo_reason None). 글: 점수 있는 이어지지 않은 줄 (새 정보), 점수 없는 이어지지 않은 줄 따로.
    tot = collections.Counter()
    bins_all = collections.defaultdict(lambda: [0, 0, 0, 0])   # (시간칸) → [사진 줄, 글 새 점수, 글 새 점수 서버 라벨, 글 새 무점수]
    per_season = {}
    band_early = {b: [0, 0, 0] for b in BANDS}
    cell_zero = collections.Counter()
    grid = {}
    dup = collections.Counter()
    for s in SEASONS:
        H = D[s][0]["H"]
        nb = int(math.ceil(H / 6))
        rows = D[s][1]
        ph = [r for r in rows if r["kind"] != "text" and r["use_indep"] and r["lo_reason"] is None]
        ph_k = [r for r in ph if r["server"] is not None]
        tx = [r for r in rows if r["kind"] == "text" and not r["link"]]
        tx_s = [r for r in tx if r["score"] is not None]
        tx_s_k = [r for r in tx_s if r["server"] is not None]
        tx_n = [r for r in tx if r["score"] is None]
        # 같은 시즌의 다른 글에 있는 사진이 같은 기록을 이미 담았나 (점수 맞고 %가 이음 규칙으로 맞는 사진이 있으면 겹침)
        phs = [r for r in rows if r["kind"] != "text" and r["pct"] is not None and r["score"] is not None]
        phs.sort(key=lambda r: r["score"])
        sc = np.array([r["score"] for r in phs])
        for r in tx_s:
            sd = dec(r["score"])
            tol = 0.1 if sd >= 1 else 1.0
            lo, hi = np.searchsorted(sc, r["score"] - tol), np.searchsorted(sc, r["score"] + tol, side="right")
            hit = [p for p in phs[lo:hi] if num_match(r["score"], p["score"], 0.1) and consistent(r["pct"], p["pct"], digits_of(r))]
            key = "strict" if sd >= 1 else "loose"
            dup[f"text_score_{key}_n"] += 1
            if hit:
                dup[f"text_score_{key}_same_record_in_some_photo"] += 1
        cells = np.zeros((nb, 5), int)   # 사진, 사진(서버 라벨), 글 새 점수, 글 새 점수(서버 라벨), 글 새 무점수
        def b_of(h):
            return None if h is None else min(nb - 1, max(0, int(h // 6)))
        for r in ph:
            b = b_of(r["he"])
            if b is not None:
                cells[b, 0] += 1
                cells[b, 1] += 1 if r["server"] is not None else 0
        for r in tx_s:
            b = b_of(r["h"])
            if b is not None:
                cells[b, 2] += 1
                cells[b, 3] += 1 if r["server"] is not None else 0
        for r in tx_n:
            b = b_of(r["h"])
            if b is not None:
                cells[b, 4] += 1
        per_season[f"S{s}"] = {"H": H, "photo": cells[:, 0].tolist(), "photo_server": cells[:, 1].tolist(),
                               "text_new_score": cells[:, 2].tolist(), "text_new_score_server": cells[:, 3].tolist(),
                               "text_new_noscore": cells[:, 4].tolist()}
        tot["photo"] += len(ph)
        tot["photo_server"] += len(ph_k)
        tot["text_new_score"] += len(tx_s)
        tot["text_new_score_server"] += len(tx_s_k)
        tot["text_new_noscore"] += len(tx_n)
        tot["text_unlinked"] += len(tx)
        for b in range(nb):
            a = bins_all[b]
            a[0] += int(cells[b, 0]); a[1] += int(cells[b, 2]); a[2] += int(cells[b, 3]); a[3] += int(cells[b, 4])
            if cells[b, 0] == 0:
                cell_zero["photo_zero_cells"] += 1
                if cells[b, 2] > 0:
                    cell_zero["photo_zero_text_adds"] += 1
            if cells[b, 0] <= 2:
                cell_zero["photo_le2_cells"] += 1
                if cells[b, 2] > 0:
                    cell_zero["photo_le2_text_adds"] += 1
            cell_zero["cells"] += 1
        wins = {"early_0_24": (0, 24), "mid_24_96": (24, 96), "late_96_end": (96, 1e9)}
        for bn, (lo, hi) in BANDS.items():
            for wn, (w0, w1) in wins.items():
                p_ = sum(1 for r in ph if lo <= r["pct"] <= hi and w0 <= r["he"] < w1)
                t_ = sum(1 for r in tx_s if lo <= r["pct"] <= hi and r["h"] is not None and w0 <= r["h"] < w1)
                pk_ = sum(1 for r in ph_k if lo <= r["pct"] <= hi and w0 <= r["he"] < w1)
                tk_ = sum(1 for r in tx_s_k if lo <= r["pct"] <= hi and r["h"] is not None and w0 <= r["h"] < w1)
                g = grid.setdefault((bn, wn), {"cells": 0, "photo": 0, "text": 0, "photo_le2_cells": 0, "photo_le2_text": 0,
                                               "photo_zero_cells": 0, "photo_zero_text": 0, "text_adds_cells_le2": 0,
                                               "sv_cells": 0, "sv_photo_le2_cells": 0, "sv_text_adds_le2": 0, "sv_photo": 0, "sv_text": 0})
                g["cells"] += 1; g["photo"] += p_; g["text"] += t_
                g["sv_photo"] += pk_; g["sv_text"] += tk_
                if p_ <= 2:
                    g["photo_le2_cells"] += 1
                    g["photo_le2_text"] += t_
                    g["text_adds_cells_le2"] += 1 if t_ > 0 else 0
                if p_ == 0:
                    g["photo_zero_cells"] += 1
                    g["photo_zero_text"] += t_
                g["sv_cells"] += 1
                if pk_ <= 2:
                    g["sv_photo_le2_cells"] += 1
                    g["sv_text_adds_le2"] += 1 if tk_ > 0 else 0
        # 목표 % 근처 (초반 h<24 와 전체)
        for bn, (lo, hi) in BANDS.items():
            ph_b = [r for r in ph if lo <= r["pct"] <= hi]
            tx_b = [r for r in tx_s if lo <= r["pct"] <= hi]
            tx_bk = [r for r in tx_s_k if lo <= r["pct"] <= hi]
            per_season[f"S{s}"].setdefault("band", {})[bn] = {"photo": len(ph_b), "text_new_score": len(tx_b), "text_new_score_server": len(tx_bk),
                                                              "photo_h_lt24": sum(1 for r in ph_b if r["he"] < 24),
                                                              "text_h_lt24": sum(1 for r in tx_b if r["h"] is not None and r["h"] < 24)}
    out["totals"] = dict(tot)
    out["ratio_text_score_over_photo"] = tot["text_new_score"] / tot["photo"]
    out["ratio_text_score_server_over_photo_server"] = tot["text_new_score_server"] / tot["photo_server"] if tot["photo_server"] else None
    out["same_pid_check"] = dict(dup)
    out["cells"] = dict(cell_zero)
    out["by_time_bin"] = {f"{b * 6}-{b * 6 + 6}h": {"photo": v[0], "text_new_score": v[1], "text_new_score_server": v[2], "text_new_noscore": v[3],
                                                     "text_over_photo": (v[1] / v[0] if v[0] else None)} for b, v in sorted(bins_all.items())}
    # 초반 T6~24 합
    early = [sum(bins_all[b][i] for b in range(0, 4)) for i in range(4)]
    out["early_0_24h"] = {"photo": early[0], "text_new_score": early[1], "text_new_score_server": early[2], "text_new_noscore": early[3]}
    bt = {}
    for bn, (lo, hi) in BANDS.items():
        ps = [per_season[f"S{s}"]["band"][bn] for s in SEASONS]
        bt[bn] = {"range": [lo, hi], "photo": sum(p["photo"] for p in ps), "text_new_score": sum(p["text_new_score"] for p in ps),
                  "text_new_score_server": sum(p["text_new_score_server"] for p in ps),
                  "photo_h_lt24": sum(p["photo_h_lt24"] for p in ps), "text_h_lt24": sum(p["text_h_lt24"] for p in ps),
                  "seasons_text_adds_ge1_where_photo_le3": sum(1 for p in ps if p["photo"] <= 3 and p["text_new_score"] >= 1),
                  "seasons_photo_le3": sum(1 for p in ps if p["photo"] <= 3)}
    out["bands"] = bt
    out["band_window_grid"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(grid.items())}
    out["per_season"] = per_season
    return out


# ---------------------------------------------------------------- 6 시즌 변동

def season_part(P, E, D, B):
    out = {"by_season": {}}
    per_exact, per_round, per_far = {}, {}, {}
    per_d1share, per_score = {}, {}
    for s in SEASONS:
        xs = [x for x in P if x["season"] == s]
        x2 = [x for x in xs if x["dg"] == 2]
        x1 = [x for x in xs if x["dg"] == 1]
        inf = [x for x in x1 if x["cls"] in ("round_only", "floor_only")]
        es = [x for x in E if x["season"] == s and x["strict"]]
        tt = [r for r in D[s][1] if r["kind"] == "text"]
        per_exact[s] = (sum(1 for x in x2 if x["cls"] == "exact"), len(x2))
        per_round[s] = (sum(1 for x in inf if x["cls"] == "round_only"), len(inf))
        per_far[s] = (sum(1 for x in es if x["cls"] == "far"), len(es))
        per_d1share[s] = (sum(1 for x in xs if x["dg"] == 1), len(xs))
        per_score[s] = (sum(1 for r in tt if r["score"] is not None), len(tt))
        out["by_season"][f"S{s}"] = {"linked": len(xs), "d2_exact": prop(*per_exact[s]), "d1_round_share": prop(*per_round[s]),
                                     "E_strict_far": prop(*per_far[s]), "d1_share_of_linked": prop(*per_d1share[s]),
                                     "text_with_score": prop(*per_score[s]),
                                     "abs_diff_mean": float(np.mean([abs(x["d"]) for x in xs])) if xs else None}
    for name, per in (("d2_exact", per_exact), ("d1_round_share", per_round), ("E_strict_far", per_far),
                      ("d1_share_of_linked", per_d1share), ("text_with_score", per_score)):
        out[name] = {"hetero": hetero(per), "expanding": expanding(per)}
    return out


# ---------------------------------------------------------------- 확인용 표본

def check_rows(P, E):
    rows = []

    def fmt(x, note):
        t, p = x["t"], x["p"]
        return {"season": f"S{x['season']}", "text_key": t["key"], "photo_key": p["key"], "note": note,
                "text_pct": t["pct"], "photo_pct": p["pct"], "text_score": t["score"], "photo_score": p["score"],
                "digits": x["dg"], "class": x["cls"], "photo_kind": p["kind"], "photo_self": p["self"],
                "dt_h": None if x["dt"] is None else round(x["dt"], 3), "text_hp": t["hp"], "photo_h": p["h"], "pid": t["pid"]}
    def pick(lst, n, seed):
        rng = np.random.default_rng(seed)
        ix = rng.choice(len(lst), size=min(n, len(lst)), replace=False)
        return [lst[i] for i in ix]
    for x in pick([x for x in P if x["cls"] == "round_only"], 2, 1):
        rows.append(fmt(x, "1자리 글: 반올림만 맞음 (버림이면 달랐을 것)"))
    for x in pick([x for x in P if x["cls"] == "floor_only"], 2, 2):
        rows.append(fmt(x, "1자리 글: 버림만 맞음"))
    for x in pick([x for x in P if x["cls"] in ("plus1", "minus1")], 2, 3):
        rows.append(fmt(x, "2자리 글: 사진과 0.01 다름"))
    for x in pick([x for x in P if x["dg"] == 2 and x["cls"] == "exact" and not x["p"]["self"]], 1, 4):
        rows.append(fmt(x, "사진 쪽이 남(self 아님)인데 정확히 같음"))
    far = [x for x in E if x["strict"] and x["cls"] == "far" and not x["linked"]]
    for x in pick(far, 3, 5):
        rows.append(fmt(x, "E 표본: 같은 글에 점수 맞는 사진이 있으나 % 가 맞지 않아 이어지지 않음"))
    return rows


# ---------------------------------------------------------------- 주 흐름

def recommend(res):
    """모형용 꼴 제안 (숫자는 위 결과에서 그대로 옮긴다)."""
    L, M, E, ES = res["linked"], res["model_fit"]["strict"], res["E_sample"]["strict"], res["E_sample_self_photo_only"]["strict"]
    ri = L["d1_informative"]["round_share_of_informative"]
    far_all = E["d1"]["far"]["k"] + E["d2"]["far"]["k"]
    far_self = ES["d1"]["far"]["k"] + ES["d2"]["far"]["k"]
    return {
        "family": "구간 관측 + 떨어진 오류(오염) 혼합. 글 값 t, 숨은 참값 X, 사진 관측과 같은 단위 (X는 곡면의 %).",
        "digits_2": {"share_of_text_rows": 5802 / 9276,
                     "interval": "사진 % 와 같은 구간 [t−0.005, t+0.01) (사진 해석 그대로)",
                     "pm001": {"delta": L["plus1_d2"]["p"] + L["minus1_d2"]["p"], "n": L["digits"]["2"],
                               "plus": L["plus1_d2"]["p"], "minus": L["minus1_d2"]["p"],
                               "note": "정확히 같지 않은 2.5%는 ±0.01 한 칸. 구간을 양쪽으로 0.01 늘리거나 혼합 가중으로 둔다"}},
        "digits_1": {"share_of_text_rows": 3474 / 9276,
                     "mixture": {"w_round": ri["p"], "w_round_ci_boot": ri["ci_boot"], "n_informative": ri["n"],
                                 "round_interval": "X ∈ [t−0.05, t+0.05)", "floor_interval": "X ∈ [t, t+0.1)",
                                 "aic": {k: v["aic"] for k, v in M["d1"]["models"].items()}},
                     "heaping_ends5": {"share": L["half_unit"]["d1_ends5"]["p"], "uniform_expected": 1 / 9,
                                       "note": "끝자리 5(0.5 단위)는 느슨 표본에서 창 밖이 더 잦다 → 끝자리 5는 폭을 0.1 이상 (대충)으로 두거나 오염을 더 크게",
                                       "far_ends5_loose": res["E_sample"]["d1_far_by_ends5_loose"]["ends5"], "far_other_loose": res["E_sample"]["d1_far_by_ends5_loose"]["other"]},
                     "trailing_zero_note": "「1.20」을 「1.2」로 적은 정확한 베낌은 1자리 글에 섞여 있어 가를 수 없다 (이어진 1자리 글 362줄 중 사진 둘째 자리가 0인 줄 %d)" % L["photo_digits_by_text_digits"]["text1_photo1"]},
        "contamination": {"eps_strict_all_photos": prop(far_all, E["n"]), "eps_strict_self_photos": prop(far_self, ES["n"]),
                          "loose_d2": res["E_sample"]["loose"]["d2"]["far"], "loose_d1": res["E_sample"]["loose"]["d1"]["far"],
                          "shape": "창 밖 크기는 0.1~0.5%p가 중심, 0.5%p 넘는 것이 3분의 1 → 균등이나 넓은 라플라스형. 사전은 엄격 값 0.08~0.11 (구간 넓음), 모형이 터무니없다고 본 점은 오류 후보 목록으로",
                          "caveat": "이음 때문에 이어진 표본에서는 못 잰다. 오염에는 다른 기록의 글·판독 오류·글 시각 이동이 섞여 있다"},
        "score_note": "글의 점수는 83%가 정수 억 (버림에 가까움, 중앙 −0.31억) — 곡면 점으로 쓰려면 점수 구간 [s, s+1)억로 다뤄야 한다",
        "time_note": "글 시각은 5,880줄이 C(게시 시각 상한), 3,396줄 B(말한 시각)"}


def pc(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def ci(p):
    return "-" if p is None or p["ci"] is None else f"{100 * p['ci'][0]:.1f}~{100 * p['ci'][1]:.1f}%"


def write_md(res, path):
    c, L, E, U, M, V, S = res["counts"], res["linked"], res["E_sample"], res["unlinked_vs_linked"], res["model_fit"], res["value"], res["season"]
    es, el = E["strict"], E["loose"]
    sh = U["share"]
    f1 = M["strict"]["d1"]["models"]
    f2 = M["strict"]["d2"]["models"]
    pp = U["post_photo"]
    fn = c["funnel"]
    ex = L["exact_d2"]
    ri = L["d1_informative"]["round_share_of_informative"]
    L1 = []
    a = L1.append
    a("# WP1 1c — 글로만 적은 %의 값 오차 (자료 판 D0, S19~S42 끝난 시즌 전체 표, 2026-10-28)")
    a("")
    a(f"- **양.** 글 줄 {c['n_text']:,} (전부 ckind=self·use=False, 정수 %는 표에 없음 — 정수 %로 빠진 self 글 {c['raw_drop_reasons_kind_self']['int_pct_in_season']:,}줄은 별도). "
      f"t_kind upper {c['t_kind']['upper']:,}·stated {c['t_kind']['stated']:,}, 자릿수 1자리 {c['digits']['1']:,}·2자리 {c['digits']['2']:,}. "
      f"점수 있음 {c['with_score']['k']:,} ({pc(c['with_score']['p'])}, 83%는 정수 억), 서버 라벨 {c['server_label']['k']:,} ({pc(c['server_label']['p'])}), 사진과 이어짐 {c['linked']['k']:,} ({pc(c['linked']['p'])}).")
    a(f"- **검증 표본 n={L['n']:,}** (묶음 (시즌,pid) {L['n_clusters']:,}). 이음은 글 하나→사진 하나(후보 여럿이면 % 일치>사진 self>indep>|차| 작은 쪽, 후보 여럿이던 글 {L['link_by']['n_link_all>1']}줄). "
      f"이음 근거 pct {L['link_by']['pct']}·pct+score {L['link_by']['pct+score']}·pct_only {L['link_by']['pct_only']}. 같은 사람·같은 hp는 정의상 100%.")
    a(f"- **이음 편향(가장 중요).** 이음 규칙이 「반올림·버림까지 맞아야」 잇는다 → 이어진 줄은 정의상 창 안이다 (|글−사진| ≤ 0.01 또는 1자리 반올림·버림). "
      f"그래서 이어진 표본에서는 큰 오차가 안 보이고, 반올림/버림/정확만 가린다. 큰 오차(꼬리)는 E 표본(점수 맞는 같은 글 사진이 있으나 % 가 안 맞아 안 이어진 줄)으로만 본다.")
    a(f"- **2자리 글 (n={L['digits']['2']}):** 사진과 정확히 같음 {pc(ex['p'])} (묶음 부트스트랩 {100 * ex['ci_boot'][0]:.1f}~{100 * ex['ci_boot'][1]:.1f}%), +0.01 {pc(L['plus1_d2']['p'])}, −0.01 {pc(L['minus1_d2']['p'])}. 글−사진 평균 {L['diff']['d2']['diff_mean']:.4f}%p.")
    a(f"- **1자리 글 (n={L['digits']['1']}):** 반올림도 버림도 같은 줄 {L['class_by_digits']['1']['both']} (그중 사진 값의 둘째 자리가 0인 줄 {L['photo_digits_by_text_digits']['text1_photo1']} — 「1.20」을 「1.2」로 적은 정확한 베낌과 구별 안 됨), "
      f"가를 수 있는 {ri['n']}줄 중 반올림만 맞음 {int(ri['k'])} ({pc(ri['p'])}, 구간 {100 * ri['ci_boot'][0]:.0f}~{100 * ri['ci_boot'][1]:.0f}%) · 버림만 맞음 {int(ri['n'] - ri['k'])}. 「대충」: 끝자리 5가 {pc(L['half_unit']['d1_ends5']['p'])} (균등이면 11%). 이어지지 않은 1자리 글도 {pc(U['heaping_ends_0_5']['d1_unlinked']['p'])}로 같다.")
    a(f"- **시각 차·self·사람.** 시각 차(글 hp−사진 촬영 h, 사진 A {L['by_dt']['A']['n']}줄 중앙 {L['by_dt']['A']['dt_quantiles']['0.5'] * 60:.1f}분·90% {L['by_dt']['A']['dt_quantiles']['0.9'] * 60:.0f}분)는 |차|와 무관 (스피어만 ρ={L['by_dt']['A']['spearman_dt_absdiff']['rho']:.2f}, p={L['by_dt']['A']['spearman_dt_absdiff']['p']:.2f}). "
      f"사진 쪽 self 참 {L['by_photo_self']['True']['n']}줄 정확 {pc(L['by_photo_self']['True']['d2_exact']['p'])} · 거짓 {L['by_photo_self']['False']['n']}줄 {pc(L['by_photo_self']['False']['d2_exact']['p'])}. 글은 전부 본인 글, 사진 쪽만 갈린다. 점수는 글이 정수 억으로 적은 꼴(버림에 가까움) (둘 다 점수 있는 짝 {L['score_diff_eok']['n']}줄, 중앙 {L['score_diff_eok']['quantiles']['0.5']:.2f}억).")
    a(f"- **꼬리(E 표본, 점수 맞는 같은 글 사진 있는 점수 있는 글).** 엄격(점수에 소수 있음) n={es['n']}: % 가 창 밖 {int(es['d1']['far']['k'] + es['d2']['far']['k'])}줄 = "
      f"{pc((es['d1']['far']['k'] + es['d2']['far']['k']) / es['n'])} (윌슨 {100 * wilson(es['d1']['far']['k'] + es['d2']['far']['k'], es['n'])[0]:.0f}~{100 * wilson(es['d1']['far']['k'] + es['d2']['far']['k'], es['n'])[1]:.0f}%). "
      f"느슨(정수 억 포함) n={el['n']}: 2자리 {pc(el['d2']['far']['p'])}, 1자리 {pc(el['d1']['far']['p'])}, 크기 중앙 0.1~0.5%p, >0.5%p가 3분의 1. 시각 차·사진 self에 뚜렷한 관계 없음 (표본 작음). 1자리 글 끝자리 5는 느슨 표본에서 창 밖이 {pc(E['d1_far_by_ends5_loose']['ends5']['p'])} vs 다른 끝자리 {pc(E['d1_far_by_ends5_loose']['other']['p'])} → 「대충 x.5」가 실제로 있다.")
    eo = res["E_sample_self_photo_only"]
    so = eo["strict"]
    fo = so["d1"]["far"]["k"] + so["d2"]["far"]["k"]
    a(f"- **꼬리, 사진을 본인 줄로 한정.** 남의 줄과 점수가 우연히 맞는 경우를 빼면 엄격 n={so['n']}에서 창 밖 {fo:.0f}줄 = {pc(fo / so['n'])} (윌슨 {100 * wilson(fo, so['n'])[0]:.0f}~{100 * wilson(fo, so['n'])[1]:.0f}%), "
      f"느슨 n={eo['loose']['n']}에서 2자리 {pc(eo['loose']['d2']['far']['p'])}·1자리 {pc(eo['loose']['d1']['far']['p'])}. 위 전체 사진 표본의 오염에는 남의 줄 우연 일치가 섞여 있다.")
    a(f"- **이어지지 않은 글(n={U['n_unlinked']:,})은 다르다.** 1자리 {pc(sh['d1']['unlinked']['p'])} (이어진 글 {pc(sh['d1']['linked']['p'])}), 서버 라벨 {pc(sh['server_label']['unlinked']['p'])} (이어진 글 {pc(sh['server_label']['linked']['p'])} — 사진에서 온 라벨), "
      f"% 중앙 {U['pct_value']['unlinked_q']['0.5']:.2f} ({U['pct_value']['linked_q']['0.5']:.2f}), KS D={U['pct_value']['ks_D']:.2f}. 같은 글에 사진이 있는 안 이어진 글은 {pp['with_post_photo']:,}줄뿐, 그중 {pp['post_photo_pct_inconsistent']:,}줄은 사진 %와 안 맞음 (다른 기록의 글일 수 있음). "
      f"→ 검증 표본은 2자리·서버 라벨·낮은 %에 치우쳐 있어 1자리 글의 오차에는 대표성이 약하다. 끝자리 분포·t_kind·점수 비율은 비슷.")
    a(f"- **모형용 꼴 (엄격 E+이음 표본).** 2자리: 같음 {1 - f2['exact_pm1']['delta']:.3f} / ±0.01 {f2['exact_pm1']['delta']:.3f} + 오염 ε≈{f2['exact_pm1_plus_contam']['eps']:.2f} (AIC 정확만 {f2['exact_only']['aic']:.0f} → ±1 {f2['exact_pm1']['aic']:.0f} → +오염 {f2['exact_pm1_plus_contam']['aic']:.0f}). "
      f"1자리: 반올림 w≈{f1['mix_plus_contam']['w_round']:.2f}·버림 {1 - f1['mix_plus_contam']['w_round']:.2f} 혼합 + 오염 ε≈{f1['mix_plus_contam']['eps']:.2f} (AIC 반올림만 {f1['round_only']['aic']:.0f}, 버림만 {f1['floor_only']['aic']:.0f}, 혼합 {f1['mix_round_floor']['aic']:.0f}, 혼합+오염 {f1['mix_plus_contam']['aic']:.0f}). "
      f"구간 폭: 2자리 사진과 같은 구간(+0.01 겹침), 1자리 반올림 [t−0.05,t+0.05) · 버림 [t,t+0.1). 오염 ε 합계 {es['d1']['far']['k'] + es['d2']['far']['k']:.0f}/{es['n']}={(es['d1']['far']['k'] + es['d2']['far']['k']) / es['n']:.2f} (자릿수별 표본이 작아 1자리 {es['d1']['far']['k']:.0f}/{es['d1']['n']}, 2자리 {es['d2']['far']['k']:.0f}/{es['d2']['n']}는 따로 믿기 어려움); 다른 기록·판독·잘못 적음을 모두 뭉친 값이라 느슨 표본은 0.16~0.27, 창 밖 크기는 0.1~0.5%p가 중심이고 꼬리가 길다.")
    a(f"- **쓸 값어치.** 점수 있는 안 이어진 글 {fn['with_score_unlinked']:,}줄 (사진 독립 관측 {V['totals']['photo']:,}의 {pc(V['ratio_text_score_over_photo'], 0)}), 그중 2자리 {fn['with_score_unlinked_d2']:,}, 점수에 소수가 있는 줄 {fn['with_score_unlinked_score_decimal']}, "
      f"서버 라벨까지 있는 줄 {V['totals']['text_new_score_server']}(2자리 {fn['with_score_unlinked_d2_server']}) — 사진 서버 라벨 줄 {V['totals']['photo_server']:,}의 {pc(V['ratio_text_score_server_over_photo_server'], 0)}. "
      f"초반 0~24시간: 사진 {V['early_0_24h']['photo']:,} vs 글 {V['early_0_24h']['text_new_score']} (0~6시간 {V['by_time_bin']['0-6h']['photo']} vs {V['by_time_bin']['0-6h']['text_new_score']}). "
      f"시즌×6시간 칸 {V['cells']['cells']}개 중 사진 0줄 {V['cells'].get('photo_zero_cells', 0)}칸 (글이 채우는 칸 {V['cells'].get('photo_zero_text_adds', 0)}), ≤2줄 {V['cells'].get('photo_le2_cells', 0)}칸 (글이 보태는 칸 {V['cells'].get('photo_le2_text_adds', 0)}). 목표 % 근처 초반 24시간: 3% 근처 사진 {V['bands']['3pct']['photo_h_lt24']} vs 글 {V['bands']['3pct']['text_h_lt24']}, 10% 근처 {V['bands']['10pct']['photo_h_lt24']} vs {V['bands']['10pct']['text_h_lt24']}. "
      f"서버 라벨 있는 글만 쓰면 초반 0~24시간 {V['early_0_24h']['text_new_score_server']}줄이라 사실상 못 보태고, 서버 혼합 모형으로 미상을 쓸 때만 값어치가 있다 (글은 점수가 정수 억이라 곡면 점도 거칠다).")
    h = S["d2_exact"]["hetero"]
    h1 = S["d1_round_share"]["hetero"]
    he = S["E_strict_far"]["hetero"]
    a(f"- **시즌 변동.** 2자리 정확 비율 시즌 간 차이 없음 (Q={h['Q']:.1f}, df={h['df']}, p={h['p']:.2f}; 늘어나는 창 추정 {S['d2_exact']['expanding']['summary']['est_min']:.3f}~{S['d2_exact']['expanding']['summary']['est_max']:.3f}), "
      f"반올림 몫 p={h1['p']:.2f} (창 {S['d1_round_share']['expanding']['summary']['est_min']:.2f}~{S['d1_round_share']['expanding']['summary']['est_max']:.2f}), 오염 p={he['p']:.2f} (창 {S['E_strict_far']['expanding']['summary']['est_min']:.2f}~{S['E_strict_far']['expanding']['summary']['est_max']:.2f}, 초반 S21은 n 작아 흔들림). "
      f"반면 양은 시즌 따라 다르다: 이어진 글 중 1자리 비율 p={S['d1_share_of_linked']['hetero']['p']:.4f} (늘어나는 창 추정 38%→25%), 점수 있는 글 비율 p<0.001 (S42 55%).")
    a("- **한계.** 정답은 안 썼다. 사진을 참값 대신 쓴다 (사진 자체의 0.01 반올림/버림·판독 오류 포함). 이음 편향 때문에 이어진 표본으로 꼬리를 못 잰다; E 표본은 작고(엄격 119) 오염에 시각 차 이동·다른 행이 섞인다. "
      "글 값에서 「1.2」와 「1.20」을 못 가른다. 정수 %(「3퍼」)는 표에 없다. 글 시각은 대부분 게시 시각 상한(C)이다.")
    open(path, "w", encoding="utf-8").write("\n".join(L1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    B = a.boot
    D = load(a.obs)
    P = pairs_of(D)
    E = e_sample(D)
    res = {"item": "1c", "seed": SEED, "boot": B, "cluster_unit": "(시즌, pid)",
           "basis": "표본: kind==text 전부. 검증 표본: link가 있는 글 줄 (글 하나가 사진 하나에 이어진다). 사진 쪽은 표의 같은 시즌 oi. 정답은 쓰지 않음."}
    res["counts"] = counts(D, B)
    res["counts"]["raw_drop_reasons_kind_self"] = raw_claim_drops()
    res["linked"] = linked_stats(P, B)
    res["linked"]["link_by"] = link_by_from_raw(P)
    res["link_definition"] = ("post_link.py link_claims: 같은 글(site,no)의 사진 중 % 가 맞는 것 (|차| ≤ 0.01, 또는 사진 값을 글의 자릿수로 반올림·버림한 값이 같다. "
                              "정수 %는 |차| ≤ 0.01만); 둘 다 점수가 있으면 점수도 맞아야(|차| ≤ 0.1억 또는 글 자릿수 반올림·버림), 단 2자리 % 에서 |차| ≤ 0.005이면 점수가 달라도 pct_only로 잇는다. "
                              "후보가 여럿이면 % 일치 > 사진 self > 사진 indep > % 차 작은 쪽의 한 사진만 link_oi로 남는다. 글 하나 = 사진 하나, 한 사진에 여러 글이 이어질 수 있다. "
                              "→ 이어진 줄은 정의상 반올림·버림 창 안에 있다: 창 밖 큰 오차는 이어진 표본에서 보이지 않는다.")
    res["E_sample"] = e_stats(E, B)
    ES = e_sample(D, self_only=True)
    res["E_sample_self_photo_only"] = e_stats(ES, B)
    res["E_sample_self_photo_only"]["note"] = "같은 글 사진 후보를 사진 쪽 self=True 줄로만 한정 (남의 줄과 점수가 우연히 맞는 경우를 뺀다)"
    res["unlinked_vs_linked"] = representativeness(D, B)
    res["unlinked_vs_linked"]["post_photo"] = post_photo_stats(D)
    res["model_fit"] = model_fit(E, P)
    res["value"] = value_part(D)
    res["season"] = season_part(P, E, D, B)
    res["recommend"] = recommend(res)
    res["check_rows"] = check_rows(P, ES)
    res["seconds"] = time.time() - t0
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(clean(res), f, ensure_ascii=False, indent=1)
    if a.md:
        write_md(clean(res), a.md)
    print(f"끝 {time.time() - t0:.1f}초 → {a.out}")
    return res


if __name__ == "__main__":
    main()
