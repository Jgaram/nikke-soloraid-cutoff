"""WP10a 진단 (2026-11-03, 사용자 질문 「F보다 좋은 모델을 왜 못 찾나 — 저녁 시간대? 같은 점수를 계속 찍은 기록?」) — 새 적합 없음.

  python3 -I tools/solo_raid/lab7/wp10_diag.py --obs-dir W/obs7/full --out data/solo-raid/analysis/r7/wp10/diag_D1.json --md data/solo-raid/analysis/r7/wp10/diag_D1.md

입력 (모두 이미 있는 것): 끝난 시즌 전체 관측 표 (obs_table.py build --full, 약 17초), r7/F_D1.json · r7/pred_M3dt_D1.json (WP5b 22시즌),
r7/wp5a/diag1_D1.json (사전만 예측), r7/wp5a/stage1_M3dt_D1.json (1단계), r7/wp9/parts/map_{none,l,r,a}_S*.json (WP9a 묶음 MAP),
analysis/model_S42_main_prior.json (F의 시간 꼴, 최근 5시즌 S37~S41), r7/stage1_M3_D1.json (M3 기본판 1단계 — curve_photo),
answer_key.json (채점·치우침 보정·curve_photo의 최종 % 보간에만).

부분 (JSON 키):
  read_vs_future  T = H(남은 상승 없음) 칸의 |e|·부호 — 「지금 읽기」만 남는 자리
  bias            시즌·창·T·목표 %·KST 시각별 부호 오차 (e = 중앙 ÷ 정답 − 1)
  prior_pull      M3 초반 부호 오차 ~ 사전 중앙 오차 (앞 시즌 수준에 끌리는가)
  trackers        같은 사람·같은 점수 기록: 묶음 수, 6시간 칸 침수 속도(NNLS, 시즌 배수 함께), 하루 안 몫, %대 배수 (시즌 붓스트랩)
  smooth_vs_tr    F의 f2pow 꼴·M3 1단계 매끈한 P가 기록과 견줘 끝까지 남은 침수를 얼마나 담나, M3 `dt`의 δ(H)
  curve_photo     컷 사진마다 실제 남은 침수(정답 표 보간) − 매끈한 곡선 (F·M3 `dt`·M3 기본), 컷 기울기, 예측 오차의 하루 주기 (2026-11-03 추가)
  bind_bias       WP9a 묶음 MAP 판별 부호 오차
  bias_layer      앞 시즌만으로 낸 평균 치우침을 빼 본 진단 (예보의 MOS 꼴 — Glahn·Lowry 1972). 창은 결과를 보고 여럿 봤다 (이미 본 것)
  early_data      T까지 게시된 한섭·미상 사진 줄의 지금 %대 분포
시각 칸·하루 몫은 05:00 시작·120시간 시즌만 (S19·S28·S38 뺌 — 시각이 다른 h에 온다). 정답은 채점·보정 진단에만 쓰고 기록 분석에는 쓰지 않는다.
공개 규칙: 글 번호·원문·닉네임·원글 주소·사람 열쇠를 쓰지 않는다 (예시 기록은 시각·%만).
"""
import argparse
import collections
import datetime as dt
import glob
import json
import math
import os
import statistics as st
import sys

import numpy as np
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_table as OT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
R = os.path.join(ROOT, "data/solo-raid")
R7 = os.path.join(R, "analysis/r7")
FMT = "%Y-%m-%d %H:%M:%S"
SKIP_CLOCK = ("S19", "S28", "S38")
MAIN_T = (18.0, 24.0, 36.0, 48.0)
BIN, NB = 6, 20
DAYNAMES = ("05-11", "11-17", "17-23", "23-05")
BANDS = ((0.3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 10.0), (10.0, 40.0))


def jload(p):
    with open(p) as f:
        return json.load(f)


def seasons_info():
    out = {}
    for n in range(19, 43):
        f = os.path.join(R, f"S{n}/state/season.json")
        if os.path.exists(f):
            out[f"S{n}"] = jload(f)
    for s in jload(os.path.join(R, "seasons.json")):
        out.setdefault(s["name"], s)
    return out


SEAS = seasons_info()


def c0_H(s):
    c = dt.datetime.strptime(SEAS[s]["c_start"], FMT)
    e = dt.datetime.strptime(SEAS[s]["end"], FMT)
    return c, math.floor((e - c).total_seconds() / 3600 + 0.01)


def truth_table():
    ak = jload(os.path.join(R, "analysis/answer_key.json"))
    return {(r["season"], r["pct"]): r["truth"] for r in ak["rows"] if r["server"] == "KR" and r.get("truth")}


TRUTH = truth_table()


def load_pred(path):
    return {(c["season"], c["T"], c["pct"]): c for c in jload(path)["cells"] if c.get("server", "KR") == "KR"}


def e_of(c):
    return c["pred"] / TRUTH[(c["season"], c["pct"])] - 1


def r2(x, d=2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), d)


def mean(v):
    return st.mean(v) if v else float("nan")


# ---------------- 0·1 읽기 대 앞날, 치우침 ----------------

def win_cells(P, s, which):
    _, H = c0_H(s)
    Ts = MAIN_T if which == "main" else (H - 12, H - 6, H)
    return [c for (ss, T, q), c in P.items() if ss == s and any(abs(T - t) < 1e-6 for t in Ts)]


def part_read_bias(P):
    out = {"read_vs_future": {}, "bias": {"season": {}, "by_T": {}, "by_clock": {}}}
    for name, M in P.items():
        rv = {}
        for grp, pcts in (("1·3%", (1.0, 3.0)), ("10%", (10.0,))):
            v = [e_of(c) for (s, T, q), c in M.items() if q in pcts and (s, q) in TRUTH and abs(T - c0_H(s)[1]) < 1e-6]
            rv[grp] = {"n": len(v), "abs_e": r2(100 * mean([abs(x) for x in v])), "signed": r2(100 * mean(v))}
        out["read_vs_future"][name] = rv
    # 시즌별 창 부호 오차 (1·3%)
    for s in sorted({k[0] for k in P["M3dt"]}, key=lambda x: int(x[1:])):
        row = {}
        for name, M in P.items():
            for w in ("main", "end"):
                v = [e_of(c) for c in win_cells(M, s, w) if c["pct"] in (1.0, 3.0) and (s, c["pct"]) in TRUTH]
                row[f"{name}_{w}"] = r2(100 * mean(v))
        out["bias"]["season"][s] = row
    for name, M in P.items():
        lows = sum(1 for s, r in out["bias"]["season"].items() if r[f"{name}_main"] is not None and r[f"{name}_main"] < 0)
        out["bias"][f"{name}_main_low_seasons"] = lows
    # T × 목표 % (05:00 시작 시즌 아님도 포함 — T는 시작 후 시간)
    for name, M in P.items():
        d = {}
        for T in (12, 24, 48, 72, 96, 114):
            for q in (0.5, 1.0, 2.0, 3.0, 5.0, 10.0):
                v = [e_of(c) for (s, TT, qq), c in M.items() if TT == T and qq == q and (s, q) in TRUTH and T <= c0_H(s)[1]]
                if v:
                    d[f"T{T}|{q:g}"] = {"n": len(v), "signed": r2(100 * mean(v)), "n_low": sum(x < 0 for x in v)}
        out["bias"]["by_T"][name] = d
    # 일차 × KST 시각 (05:00 시작·120시간 시즌)
    for name, M in P.items():
        d = {}
        for grp, pcts in (("1·3%", (1.0, 3.0)), ("10%", (10.0,))):
            by = collections.defaultdict(list)
            for (s, T, q), c in M.items():
                if q not in pcts or (s, q) not in TRUTH or s in SKIP_CLOCK:
                    continue
                c0, H = c0_H(s)
                if T >= H:
                    continue
                by[f"d{int(T // 24) + 1}|{(c0 + dt.timedelta(hours=T)).hour:02d}"].append(e_of(c))
            d[grp] = {k: {"n": len(v), "signed": r2(100 * mean(v)), "abs": r2(100 * mean([abs(x) for x in v]))} for k, v in sorted(by.items())}
        out["bias"]["by_clock"][name] = d
    return out


# ---------------- 2 사전 끌림 ----------------

def part_prior_pull(M):
    d1 = jload(os.path.join(R7, "wp5a/diag1_D1.json"))
    xs, ys = [], []
    rows = {}
    for s in d1["targets"]:
        c = d1["cells"][s]["base"]["full"]
        pe = [math.exp(c[q]["med"]) / TRUTH[(s, float(q))] - 1 for q in ("1", "3") if (s, float(q)) in TRUTH]
        v = [e_of(cc) for cc in win_cells(M, s, "main") if cc["pct"] in (1.0, 3.0) and (s, cc["pct"]) in TRUTH]
        xs.append(100 * mean(pe))
        ys.append(100 * mean(v))
        rows[s] = {"prior_e": r2(xs[-1], 1), "m3_main": r2(ys[-1])}
    b = np.polyfit(xs, ys, 1)
    return {"seasons": rows, "slope": r2(b[0], 3), "corr": r2(np.corrcoef(xs, ys)[0, 1]),
            "prior_low_seasons": sum(x < 0 for x in xs), "n": len(xs)}


# ---------------- 3 같은 점수 기록 ----------------

def tracker_groups(obs_dir):
    """(시즌, 머리, {(pid, 점수): [(시각, %, 종류)]}) — 한섭 라벨 또는 미상, use 줄. A·B는 h, C는 게시 시각(상한)."""
    out = []
    for n in range(19, 43):
        s = f"S{n}"
        head, rows = OT.read_cell(os.path.join(obs_dir, f"{s}.jsonl.gz"))
        g = collections.defaultdict(list)
        for r in rows:
            if r["kind"] not in ("A", "B", "C") or not r.get("pct") or not r.get("score") or not r.get("pid") or not r.get("use"):
                continue
            if r.get("server") not in (None, "KR"):
                continue
            t = r["h"] if r["kind"] in ("A", "B") and r.get("h") is not None else r.get("hp")
            if t is None:
                continue
            g[(r["pid"], round(r["score"], 2))].append((t, r["pct"], r["kind"]))
        out.append((s, head, g))
    return out


def tracker_pairs(groups, lo=0.0, hi=1e9):
    pairs = []
    for s, head, g in groups:
        if s in SKIP_CLOCK:
            continue
        for L in g.values():
            L = sorted({(min(max(t, 0.0), 120.0), p) for t, p, k in L if k in ("A", "B") and lo <= p < hi})
            for (t1, p1), (t2, p2) in zip(L, L[1:]):
                if t2 - t1 < 1.0 or t1 >= 120:
                    continue
                dL = math.log10(p2 / p1)
                if dL < -0.02:      # 순서 모순 (다른 서버·판독 오류) — 버림
                    continue
                var = ((0.005 / p1) ** 2 + (0.005 / p2) ** 2) / math.log(10) ** 2 + 0.004 ** 2
                pairs.append((s, t1, t2, dL, var, p1))
    return pairs


def overlap(t1, t2):
    v = np.zeros(NB)
    for k in range(NB):
        v[k] = max(0.0, min((k + 1) * BIN, t2) - max(k * BIN, t1))
    return v


def fit_rate(pairs, band_of=None, nband=0):
    """ΔL = m_s · β_b · Σ_k λ_k·겹친 시간. λ ≥ 0 (NNLS), m_s 기하평균 1, β는 둘째 %대를 1로."""
    X = np.array([overlap(p[1], p[2]) for p in pairs])
    y = np.array([p[3] for p in pairs])
    w = 1 / np.sqrt([p[4] for p in pairs])
    seas = sorted({p[0] for p in pairs})
    si = np.array([seas.index(p[0]) for p in pairs])
    bi = np.array([band_of(p[5]) for p in pairs]) if band_of else np.zeros(len(pairs), int)
    m = np.ones(len(seas))
    beta = np.ones(max(nband, 1))
    for _ in range(40):
        sc = m[si] * beta[bi]
        lam, _ = nnls(X * (sc * w)[:, None], y * w)
        base = X @ lam
        if band_of:
            for j in range(nband):
                ix = bi == j
                den = ((base[ix] * m[si[ix]]) ** 2 * w[ix] ** 2).sum()
                if den > 0:
                    beta[j] = (y[ix] * base[ix] * m[si[ix]] * w[ix] ** 2).sum() / den
            beta /= beta[1]
        for j in range(len(seas)):
            ix = si == j
            den = ((base[ix] * beta[bi[ix]]) ** 2 * w[ix] ** 2).sum()
            if den > 0:
                m[j] = (y[ix] * base[ix] * beta[bi[ix]] * w[ix] ** 2).sum() / den
        m /= np.exp(np.mean(np.log(m)))
    return lam, beta, dict(zip(seas, m)), X.sum(0)


def boot_seasons(pairs, fn, B, seed):
    seas = sorted({p[0] for p in pairs})
    bys = {s: [p for p in pairs if p[0] == s] for s in seas}
    rng = np.random.default_rng(seed)
    return [fn([p for s in rng.choice(seas, len(seas)) for p in bys[s]]) for _ in range(B)]


def day_shares(lam):
    return [lam[4 * d:4 * d + 4] / max(lam[4 * d:4 * d + 4].sum(), 1e-12) for d in range(1, 5)]


def part_trackers(groups):
    n2 = collections.Counter()
    long_ = []
    for s, head, g in groups:
        for L in g.values():
            pts = sorted(set(L))
            if len({round(p, 2) for _, p, _ in pts}) >= 2:
                n2[s] += 1
                if len({round(p, 2) for _, p, _ in pts}) >= 4 and pts[-1][0] - pts[0][0] >= 24:
                    long_.append((s, head["H"], pts))
    long_.sort(key=lambda x: -len(x[2]))
    pairs = tracker_pairs(groups)
    lam, _, mseas, expo = fit_rate(pairs)
    boots = np.array(boot_seasons(pairs, lambda P: fit_rate(P)[0], 200, 1))
    bins = []
    for k in range(NB):
        lo, hi = np.percentile(boots[:, k], [10, 90])
        bins.append({"h": [k * BIN, (k + 1) * BIN], "kst": f"{(5 + k * BIN) % 24:02d}-{(5 + (k + 1) * BIN) % 24:02d}",
                     "rate_x1000": r2(1000 * lam[k]), "p10": r2(1000 * lo), "p90": r2(1000 * hi), "exposure_h": r2(expo[k], 0)})
    sh = day_shares(lam)
    bsh = np.array([np.mean(day_shares(b)[1:], 0) for b in boots])   # 3~5일차 평균
    shares = {f"day{d + 2}": {n: r2(x) for n, x in zip(DAYNAMES, v)} | {"total_log10": r2(lam[4 * (d + 1):4 * (d + 1) + 4].sum() * BIN, 3)}
              for d, v in enumerate(sh)}
    m35 = np.mean(sh[1:], 0)
    shares["day3_5_mean"] = {n: r2(x) for n, x in zip(DAYNAMES, m35)}
    shares["day3_5_boot_p10_p90"] = {n: [r2(np.percentile(bsh[:, i], 10)), r2(np.percentile(bsh[:, i], 90))] for i, n in enumerate(DAYNAMES)}

    def band_of(p):
        for i, (a, b) in enumerate(BANDS):
            if a <= p < b:
                return i
        return None
    bands = {}
    for name, sel in (("all", lambda p: True), ("start_lt48", lambda p: p[1] < 48), ("start_ge72", lambda p: p[1] >= 72)):
        P = [p for p in pairs if sel(p) and band_of(p[5]) is not None]
        _, beta, _, _ = fit_rate(P, band_of, len(BANDS))
        bb = np.array(boot_seasons(P, lambda Q: fit_rate(Q, band_of, len(BANDS))[1], 120, 2))
        cnt = collections.Counter(band_of(p[5]) for p in P)
        bands[name] = [{"band": list(BANDS[j]), "n": cnt.get(j, 0), "mult": r2(beta[j]),
                        "p10": r2(np.nanpercentile(bb[:, j], 10)), "p90": r2(np.nanpercentile(bb[:, j], 90))} for j in range(len(BANDS))]
    ex = [{"season": s, "H": H, "pts": [[r2(t, 1), p, k] for t, p, k in pts]} for s, H, pts in long_[:3]]
    return {"groups_ge2": dict(n2), "groups_ge2_total": sum(n2.values()), "long_ge4_ge24h": len(long_), "n_pairs": len(pairs),
            "rate_bins": bins, "day_shares": shares, "band_mult": bands, "examples": ex,
            "season_scale": {s: r2(v, 3) for s, v in mseas.items()}}, lam


# ---------------- 4 매끈한 곡선 대 기록 ----------------

def part_smooth(lam):
    tr = jload(os.path.join(R, "analysis/model_S42_main_prior.json"))["transform"]
    th, H = tr["theta"], 120.0

    def PF(h):
        t = min(max(h, 0.25), H) / H
        w = 1 / (1 + math.exp(-th[0]))
        return -math.log10(w * t ** math.exp(th[1]) + (1 - w) * t ** math.exp(th[2]))
    rem_tr = lambda T: float(sum(lam[k] * BIN for k in range(int(T // BIN), NB)))  # noqa: E731
    import m3
    B = m3.BASIS
    s1 = jload(os.path.join(R7, "wp5a/stage1_M3dt_D1.json"))
    li = [s1["header"]["names"].index(f"l{i}") for i in range(m3.NL)]

    def Pm3(cl, u):
        sg = B["sg"]
        f = np.exp(B["Blf"] @ cl) * 2 * sg
        seg = 0.5 * (f[1:] + f[:-1]) * B["ds"]
        Pg = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        return float(np.interp(math.sqrt(max(u, 0.02)), sg, Pg))
    Ts = (24, 36, 48, 60, 72, 84, 96, 108, 114)
    m3P = {T: [] for T in Ts}
    dH = {}
    for s, v in s1["seasons"].items():
        d = (v.get("dep_post") or {}).get("delta_H")
        if d:
            dH[s] = r2(d["mean"], 4)
        if s in SKIP_CLOCK:
            continue
        cl = np.linalg.solve(B["Bl12"], np.array([v["mean"][i] for i in li]))   # 1단계 l0..l11은 s12 격자 값 → 계수 (정방 B)
        for T in Ts:
            m3P[T].append(Pm3(cl, T / H))
    dHm = st.mean(dH.values())
    rows = []
    for T in Ts:
        a, f, m = rem_tr(T), PF(T), mean(m3P[T])
        rows.append({"T": T, "kst": f"{(5 + T) % 24:02d}", "tracker": r2(a, 3), "F": r2(f, 3), "F_over_tr": r2(f / a),
                     "M3_smooth": r2(m, 3), "M3_smooth_over_tr": r2(m / a), "M3_smooth_plus_dH": r2(m + dHm, 3)})
    inc = [{"h": [6 * k, 6 * k + 6], "kst": f"{(5 + 6 * k) % 24:02d}-{(11 + 6 * k) % 24:02d}", "tracker": r2(lam[k] * BIN, 4),
            "F": r2(PF(6 * k) - PF(6 * k + 6), 4)} for k in range(4, NB)]
    return {"remaining": rows, "bins6h": inc, "F_prior": {"basis": tr["basis"], "theta": th},
            "delta_H": {"by_season": dH, "mean": r2(dHm, 4), "n_pos": sum(x > 0 for x in dH.values()), "n": len(dH)}}


# ---------------- 4b 곡선 몫 — 컷 사진으로 (2026-11-03 추가, 사용자 질문 「곡선 영향이 있다고 보고 계획한 거야?」) ----------------

def cut_slopes():
    """컷 기울기 −d lnC / d log10 p 의 시즌 중앙 (한섭 정답). 남은 침수를 작게 잡으면 목표보다 오른쪽(큰 %)으로 가므로 오른쪽 기울기를 쓴다.
    보상 경계(1·3·10%) 바로 위가 가팔라 왼쪽·오른쪽이 크게 다르다 (3%: 2~3% 0.18 대 3~5% 0.74)."""
    by = collections.defaultdict(dict)
    for (s, q), v in TRUTH.items():
        by[s][q] = v

    def med(a, b):
        v = [math.log(d[a] / d[b]) / math.log10(b / a) for d in by.values() if a in d and b in d]
        return {"slope": r2(st.median(v), 3), "n": len(v)} if v else None
    return {"right": {"1": med(1.0, 2.0), "3": med(3.0, 5.0), "10": med(10.0, 15.0)},
            "left": {"1": med(0.5, 1.0), "3": med(2.0, 3.0), "10": med(5.0, 10.0)}}


def final_pct(tt, x):
    """정답 컷 표(한섭)에서 점수 x의 최종 % (log 점수 대 log %의 선형 보간, 표 밖이면 None)."""
    ps = sorted(tt)
    for i in range(len(ps) - 1):
        hi, lo = tt[ps[i]], tt[ps[i + 1]]
        if hi >= x >= lo and hi > lo:
            f = (math.log(hi) - math.log(x)) / (math.log(hi) - math.log(lo))
            return 10 ** (math.log10(ps[i]) + f * (math.log10(ps[i + 1]) - math.log10(ps[i])))
    return None


def part_curve_photo(obs_dir, J):
    """컷 사진(A·B·C, 한섭·미상)마다 실제 남은 침수 log10(최종 %/지금 %)에서 매끈한 곡선 P(h)를 뺀 잔차의 중앙 — KST 시각 창·일차·%대별.
    최종 %는 정답 표 보간 (정답을 본 진단). 곡선: F 꼴 (model_S42_main_prior), M3 1단계 ℓ (+ x항 β) — `dt`판과 기본판.
    05시 시작 120시간 시즌만, 6 ≤ h < 118, 최종 % 0.5~15."""
    import numpy as np
    import m3
    tr = jload(os.path.join(R, "analysis/model_S42_main_prior.json"))["transform"]
    th = tr["theta"]

    def PF(h, H=120.0):
        t = min(max(h, 0.25), H) / H
        w = 1 / (1 + math.exp(-th[0]))
        return -math.log10(w * t ** math.exp(th[1]) + (1 - w) * t ** math.exp(th[2]))
    B = m3.BASIS

    def Pm3(cl, u):
        sg = B["sg"]
        f = np.exp(B["Blf"] @ cl) * 2 * sg
        seg = 0.5 * (f[1:] + f[:-1]) * B["ds"]
        Pg = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        return float(np.interp(math.sqrt(max(u, 0.02)), sg, Pg))
    S1 = {"M3dt": jload(os.path.join(R7, "wp5a/stage1_M3dt_D1.json")), "M3base": jload(os.path.join(R7, "stage1_M3_D1.json"))}
    tru = collections.defaultdict(dict)
    for (s, q), v in TRUTH.items():
        tru[s][q] = v
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for n in range(21, 43):
        s = f"S{n}"
        if s in SKIP_CLOCK or len(tru[s]) < 4 or any(s not in v["seasons"] for v in S1.values()):
            continue
        par = {}
        for k, v in S1.items():
            nm = v["header"]["names"]
            l12 = np.array([v["seasons"][s]["mean"][nm.index(f"l{i}")] for i in range(m3.NL)])   # s12 격자 값 → 계수 (정방 B)
            par[k] = (np.linalg.solve(B["Bl12"], l12), v["seasons"][s]["mean"][nm.index("beta")])
        _, rows = OT.read_cell(os.path.join(obs_dir, f"{s}.jsonl.gz"))
        for r in rows:
            if r["kind"] not in ("A", "B", "C") or not r.get("use") or not r.get("pct") or not r.get("score") or r.get("hp") is None:
                continue
            if r.get("server") not in (None, "KR") or not (6 <= r["hp"] < 118):
                continue
            pf = final_pct(tru[s], r["score"])
            if not pf or not (0.5 <= pf <= 15) or pf < r["pct"] * 0.98:
                continue
            act, h = math.log10(pf / r["pct"]), r["hp"]
            clk = (5 + h) % 24
            w = "05-11" if 5 <= clk < 11 else "11-17" if 11 <= clk < 17 else "17-23" if 17 <= clk < 23 else "23-05"
            day = "d2" if h < 48 else "d3-4" if h < 96 else "d5"
            band = "<2%" if pf < 2 else "2-5%" if pf < 5 else "5%+"
            rr = {"F": act - PF(h)}
            for k, (cl, beta) in par.items():
                rr[k] = act - Pm3(cl, h / 120.0) * (pf / 3) ** (beta / math.log(10))
            for key in ("all", w, f"{day}|{w}", f"{band}|{w}"):
                for k, v in rr.items():
                    res[key][k].append(v)
    order = lambda k: (k != "all", k.count("|"), k)  # noqa: E731
    tab = {k: dict(n=len(v["F"]), **{m: r2(st.median(x), 3) for m, x in v.items()}) for k, v in sorted(res.items(), key=lambda kv: order(kv[0]))}
    # 실제 예측 오차에 하루 주기가 보이나 — by_clock (05시 시작 시즌)의 17시 시점 평균 − 05시 시점 평균
    key = {24: "d2|05", 36: "d2|17", 48: "d3|05", 60: "d3|17", 72: "d4|05", 84: "d4|17", 96: "d5|05", 108: "d5|17"}
    zig = {}
    for n in ("M3dt", "F"):
        for g in ("1·3%", "10%"):
            bc = J["bias"]["by_clock"][n][g]
            v17 = [bc[k]["signed"] for T, k in key.items() if k.endswith("17") and k in bc]
            v05 = [bc[k]["signed"] for T, k in key.items() if k.endswith("05") and k in bc]
            zig[f"{n}|{g}"] = {"at17": r2(mean(v17)), "at05": r2(mean(v05)), "diff": r2(mean(v17) - mean(v05))}
    return {"resid_median_log10": tab, "slopes": cut_slopes(), "forecast_zigzag": zig,
            "note": "잔차 > 0 = 곡선이 남은 침수를 작게 잡음. 컷 % 오차 ≈ −오른쪽 기울기 × 잔차 × 100. 정답을 본 진단 (이미 본 것)."}


# ---------------- 5 묶음 판 ----------------

def part_bind():
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for v in ("none", "l", "r", "a"):
        for f in glob.glob(os.path.join(R7, f"wp9/parts/map_{v}_S*.json")):
            for c in jload(f)["cells"]:
                if (c["season"], c["pct"]) in TRUTH:
                    res[("main" if c["T"] <= 48 else "end", c["pct"])][v].append(e_of(c))
    return {f"{w}|{q:g}": {v: {"signed": r2(100 * mean(L)), "n_low": sum(x < 0 for x in L), "n": len(L)} for v, L in d.items()}
            for (w, q), d in sorted(res.items())}


# ---------------- 6 치우침 보정 진단 ----------------

def bkey(s, T):
    H = c0_H(s)[1]
    return ("end", round(H - T)) if T >= H - 12 - 1e-6 else ("t", T)


def bias_correct(M, win, minprev=3):
    logs = collections.defaultdict(dict)
    for (s, T, q), c in M.items():
        if (s, q) in TRUTH:
            logs[(bkey(s, T), q)][s] = math.log(c["pred"] / TRUTH[(s, q)])
    out = {}
    for k, c in M.items():
        s, T, q = k
        prev = [v for ss, v in sorted(logs[(bkey(s, T), q)].items(), key=lambda kv: int(kv[0][1:])) if int(ss[1:]) < int(s[1:])]
        prev = prev[-win:] if win else prev
        c = dict(c)
        if len(prev) >= minprev:
            c["pred"] = c["pred"] * math.exp(-st.mean(prev))
        out[k] = c
    return out


def score(M, seasons, pcts):
    m, e, sg = [], [], []
    for (s, T, q), c in M.items():
        if s not in seasons or q not in pcts or (s, q) not in TRUTH:
            continue
        H = c0_H(s)[1]
        x = e_of(c)
        if T in MAIN_T:
            m.append(abs(x))
            sg.append(x)
        elif any(abs(T - t) < 1e-6 for t in (H - 12, H - 6, H)):
            e.append(abs(x))
            sg.append(x)
    return {"score": r2(100 * (mean(m) / 3 + 2 * mean(e) / 3)), "main": r2(100 * mean(m)), "end": r2(100 * mean(e)), "signed": r2(100 * mean(sg))}


def part_bias_layer(P):
    out = {"note": "앞 시즌(번호가 작은 시즌)만으로 같은 (T 자리, 목표 %) 칸의 평균 ln(중앙/정답)을 빼 봄. 앞 시즌 3개 미만이면 그대로. "
                   "창 all·10·6·4는 결과를 보고 여럿 본 것 — 방법이 아니라 진단 (이미 본 것). score는 칸 평균 |e| (⅓ 초반 + ⅔ 막판, score.py third와 같은 셈)."}
    for lo in (24, 29):
        ss = {f"S{n}" for n in range(lo, 43)}
        d = {}
        for name, M in P.items():
            for win in (None, 10, 6, 4):
                MM = M if win == -1 else bias_correct(M, win)
                d[f"{name}|{'all' if win is None else win}"] = {g: score(MM, ss, p) for g, p in (("1·3%", (1.0, 3.0)), ("10%", (10.0,)))}
            d[f"{name}|raw"] = {g: score(M, ss, p) for g, p in (("1·3%", (1.0, 3.0)), ("10%", (10.0,)))}
        out[f"S{lo}-S42"] = d
    return out


# ---------------- 7 초반 자료 ----------------

def part_early(groups_dir):
    edges = (0, 0.3, 0.6, 1, 2, 4, 10, 100)
    out = {}
    for T in (24, 48, 96):
        c = collections.Counter()
        for n in range(21, 43):
            s = f"S{n}"
            if s in ("S28", "S38"):
                continue
            _, rows = OT.read_cell(os.path.join(groups_dir, f"{s}.jsonl.gz"))
            for r in rows:
                if r["kind"] in ("A", "B", "C") and r.get("use") and r.get("pct") and r.get("hp") is not None \
                        and r.get("server") in (None, "KR") and r["hp"] <= T:
                    for i in range(len(edges) - 1):
                        if edges[i] <= r["pct"] < edges[i + 1]:
                            c[i] += 1
                            break
        n = sum(c.values())
        out[f"T{T}"] = {"n": n, "share": {f"{edges[i]}-{edges[i + 1]}": r2(c[i] / n) for i in range(len(edges) - 1)}}
    return out


# ---------------- 보고 ----------------

def md_report(J):
    L = ["# WP10a 진단 (2026-11-03) — 왜 F를 못 넘나", "",
         "`lab7/wp10_diag.py`가 만든 표. 새 적합 없음, 자료 판 D1. 22시즌 정답을 본 진단이라 WP10 결과의 낙관 위험에 적는다.", ""]
    rv = J["read_vs_future"]
    L += ["## 0. 지금 읽기 대 앞날 (T = H, 남은 상승 없음)", "", "| | 1·3% \\|e\\| (부호) | 10% \\|e\\| (부호) |", "|---|---|---|"]
    for n in ("F", "M3dt"):
        L.append(f"| {n} | {rv[n]['1·3%']['abs_e']} ({rv[n]['1·3%']['signed']:+}) | {rv[n]['10%']['abs_e']} ({rv[n]['10%']['signed']:+}) |")
    b = J["bias"]
    L += ["", f"## 1. 치우침", "", f"초반 창 1·3% 부호 평균이 음수인 시즌: M3 `dt` {b['M3dt_main_low_seasons']}/22 · F {b['F_main_low_seasons']}/22", "",
          "| T | M3 1% | 3% | 10% | F 1% | 3% | 10% |", "|---|---|---|---|---|---|---|"]
    for T in (24, 48, 72, 96, 114):
        row = [f"T{T}"]
        for n in ("M3dt", "F"):
            for q in ("1", "3", "10"):
                c = b["by_T"][n].get(f"T{T}|{q}")
                row.append(f"{c['signed']:+} ({c['n_low']}/{c['n']} 낮음)" if c else "-")
        L.append("| " + " | ".join(row) + " |")
    pp = J["prior_pull"]
    L += ["", f"## 2. 사전 끌림 — 아님", "", f"M3 초반 부호 오차 ~ 사전 중앙 오차: 기울기 {pp['slope']} · 상관 {pp['corr']} (사전이 낮은 시즌 {pp['prior_low_seasons']}/{pp['n']})"]
    t = J["trackers"]
    L += ["", "## 3. 같은 사람·같은 점수 기록", "",
          f"% 2개 이상 묶음 {t['groups_ge2_total']} · % 4개 이상·24시간 넘게 {t['long_ge4_ge24h']} · 침수 속도에 쓴 쌍(A·B, 05시 시작 시즌) {t['n_pairs']}", "",
          "하루(05시~다음 05시) 안 몫:", "", "| 일차 | 05-11 | 11-17 | 17-23 | 23-05 | 하루 합 (log10) |", "|---|---|---|---|---|---|"]
    for d in ("day2", "day3", "day4", "day5"):
        x = t["day_shares"][d]
        L.append(f"| {d[3:]} | {x['05-11']} | {x['11-17']} | {x['17-23']} | {x['23-05']} | {x['total_log10']} |")
    m = t["day_shares"]["day3_5_mean"]
    bt = t["day_shares"]["day3_5_boot_p10_p90"]
    L.append(f"| 3~5 평균 | {m['05-11']} | {m['11-17']} | {m['17-23']} {bt['17-23']} | {m['23-05']} {bt['23-05']} | |")
    L += ["", "%대 배수 (1~2% = 1, [시즌 붓스트랩 10~90%]):", ""]
    for name, rows in t["band_mult"].items():
        L.append(f"- {name}: " + " · ".join(f"{a}~{b_}% {x['mult']} [{x['p10']}, {x['p90']}] (쌍 {x['n']})" for x in rows for a, b_ in [x["band"]]))
    s = J["smooth_vs_tr"]
    L += ["", "## 4. 매끈한 시간 곡선 대 기록 — 끝까지 남은 침수 (log10)", "",
          "| T (KST) | 기록 | F 꼴 | M3 1단계 매끈 | 매끈/기록 | 매끈 + δ_H 평균 |", "|---|---|---|---|---|---|"]
    for r in s["remaining"]:
        L.append(f"| T{r['T']} ({r['kst']}시) | {r['tracker']} | {r['F']} | {r['M3_smooth']} | {r['M3_smooth_over_tr']} | {r['M3_smooth_plus_dH']} |")
    L += ["", f"M3 `dt` 1단계 δ(H): 평균 {s['delta_H']['mean']:+} log10, 양수 {s['delta_H']['n_pos']}/{s['delta_H']['n']} 시즌"]
    ce = J["curve_photo"]
    sl = ce["slopes"]
    L += ["", "## 4b. 컷 사진으로 본 곡선 — 실제 남은 침수 − 매끈한 곡선 (log10, 중앙. > 0 = 곡선이 작게 잡음)", "",
          "실제 남은 침수 = log10(정답 표로 보간한 최종 % ÷ 사진의 지금 %). 05시 시작 시즌, 6 ≤ h < 118, 최종 % 0.5~15.", "",
          "| 창 | 사진 | F 꼴 | M3 `dt` | M3 기본 |", "|---|---|---|---|---|"]
    for k, v in ce["resid_median_log10"].items():
        L.append(f"| {k} | {v['n']} | {v['F']:+} | {v['M3dt']:+} | {v['M3base']:+} |")
    L += ["", "컷 기울기 −d lnC/d log10 p (시즌 중앙): " + " · ".join(f"{q}% 오른쪽 {sl['right'][q]['slope']} / 왼쪽 {sl['left'][q]['slope']}" for q in ("1", "3", "10")) +
          " — 남은 침수를 작게 잡으면 오른쪽 기울기만큼 컷이 낮아진다.", "",
          "실제 예측 오차의 하루 주기 (by_clock, 17시 시점 평균 − 05시 시점 평균, %): " +
          " · ".join(f"{k} {v['diff']:+}" for k, v in ce["forecast_zigzag"].items())]
    bb = J["bind_bias"]
    L += ["", "## 5. WP9a 묶음 MAP 부호 오차 (%)", "", "| 창·% | 묶음 없음 | ℓ | r | a |", "|---|---|---|---|---|"]
    for k, d in bb.items():
        L.append(f"| {k} | " + " | ".join(f"{d[v]['signed']:+} ({d[v]['n_low']}/{d[v]['n']})" for v in ("none", "l", "r", "a")) + " |")
    bl = J["bias_layer"]
    L += ["", "## 6. 앞 시즌 평균 치우침 빼기 (진단 — 창은 결과를 보고 여럿 봄)", ""]
    for rng, d in bl.items():
        if rng == "note":
            continue
        L += [f"**{rng}**", "", "| 판 | 1·3% score (초반·막판, 부호) | 10% score (초반·막판, 부호) |", "|---|---|---|"]
        for k, v in d.items():
            L.append(f"| {k} | {v['1·3%']['score']} ({v['1·3%']['main']}·{v['1·3%']['end']}, {v['1·3%']['signed']:+}) | {v['10%']['score']} ({v['10%']['main']}·{v['10%']['end']}, {v['10%']['signed']:+}) |")
        L.append("")
    ed = J["early_data"]
    L += ["## 7. T까지 게시된 한섭·미상 사진 줄의 지금 %대", ""]
    for T, v in ed.items():
        L.append(f"- {T} ({v['n']}줄): " + " · ".join(f"{k}% {x}" for k, x in v["share"].items()))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    a = ap.parse_args()
    P = {"F": load_pred(os.path.join(R7, "F_D1.json")), "M3dt": load_pred(os.path.join(R7, "pred_M3dt_D1.json"))}
    J = {"about": "WP10a 진단 — 새 적합 없음. 정의는 lab7/wp10_diag.py 머리. 부호 오차·score는 %.", "data_ver": "D1"}
    J.update(part_read_bias(P))
    J["prior_pull"] = part_prior_pull(P["M3dt"])
    groups = tracker_groups(a.obs_dir)
    J["trackers"], lam = part_trackers(groups)
    J["smooth_vs_tr"] = part_smooth(lam)
    J["curve_photo"] = part_curve_photo(a.obs_dir, J)
    J["bind_bias"] = part_bind()
    J["bias_layer"] = part_bias_layer(P)
    J["early_data"] = part_early(a.obs_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(J, f, ensure_ascii=False, indent=1)
    if a.md:
        with open(a.md, "w") as f:
            f.write(md_report(J))
    print("wrote", a.out, a.md or "")


if __name__ == "__main__":
    main()
