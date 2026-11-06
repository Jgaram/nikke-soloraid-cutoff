"""WP12 곡선 꼴 조사 — 공용 표 (2026-11-04, 1단계) — 침수(A~D)·컷(E·F) 서술 계산이 함께 읽는 표. 새 적합·채점 없음.

  python3 -I tools/solo_raid/lab7/wp12_tables.py summary --obs-dir W/obs7/full --out data/solo-raid/analysis/r7/wp12/tables_D1.json

계획: docs/solo-raid-method.md 「WP12」. 공용 코드·시즌 state는 읽기만 한다 (새 사이트 요청 0).
모든 시각은 KST, h = 챌린지 시작(season.json c_start)부터 시간. 줄 단위 표는 import해서 쓰고, summary는 시즌별 개수만 파일에 쓴다.

읽기 (import 용):
  season_info() · kst_hour · day_of · bin3 · hours_left · overlap          시간 축
  pairs(obs_dir) · photos(obs_dir) · top7(obs_dir) · reward_tiers(obs_dir)  관측 표에서 (lru_cache — 반환값은 고치지 말 것)
  end_points(s) · truth_table(s)                                           끝 화면 점·정답 표 (한섭)
  F_P(u) · F_shape() · m3_P(s, u, pf) · m3_lnC(s, q) · act_g(s)            견줄 곡선 (F·M3 기본판 1단계·활동 g)
공개 규칙: 글 번호·원문·닉네임·원글 주소·사람 열쇠(pid·mid)·사진 열쇠(img·oi)를 결과에 쓰지 않는다 (시각·%·점수·개수만).
"""
import argparse
import collections
import datetime as dt
import functools
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import obs_table as OT  # noqa: E402
import wp10_diag as WD  # noqa: E402
import explore_pct_transform as EPT  # noqa: E402

R = WD.R
R7 = WD.R7
SEASONS = [f"S{n}" for n in range(19, 43)]
SPECIAL = ("S26", "S28", "S38")
_STATE = lambda s: os.path.join(R, s, "state")  # noqa: E731


def _jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---------------- 시즌 정보·시간 축 ----------------

@functools.lru_cache(None)
def season_info():
    """{시즌: c0(시작 KST 문자열) · c0h(시작의 하루 안 시각) · H(시간, wp10_diag.c0_H와 같은 셈) · dow(시작 요일) · era · special · clock_ok}."""
    out = {}
    for s in SEASONS:
        c0, H = WD.c0_H(s)
        c0h = c0.hour + c0.minute / 60 + c0.second / 3600
        n = int(s[1:])
        out[s] = {"c0": c0.strftime(WD.FMT), "c0h": c0h, "H": H, "dow": c0.strftime("%a"),
                  "era": "old" if n <= 30 else "new", "special": s in SPECIAL, "clock_ok": c0h == 5 and H == 120}
    return out


def _sc(v, fn):
    a = np.asarray(v, float)
    r = fn(a)
    return r.item() if r.ndim == 0 else r


def kst_hour(s, h):
    """시각 h(시작부터 시간)의 KST 하루 안 시각 [0, 24)."""
    c0h = season_info()[s]["c0h"]
    return _sc(h, lambda a: np.mod(c0h + a, 24.0))


def day_of(s, h):
    """일차 (KST 05시 경계, 첫날 = 1) — act_clock._segments의 d + 1과 같은 셈."""
    c0h = season_info()[s]["c0h"]
    base = math.floor((c0h - 5.0) / 24.0 + 1e-9)
    return _sc(h, lambda a: (np.floor((c0h + a - 5.0) / 24.0 + 1e-9) - base + 1).astype(int))


def bin3(s, h):
    """KST 3시간 칸 번호 0..7 (05-08 = 0, 08-11 = 1, …, 02-05 = 7)."""
    c0h = season_info()[s]["c0h"]
    return _sc(h, lambda a: np.mod(np.floor((c0h + a - 5.0) / 3.0 + 1e-9), 8).astype(int))


def hours_left(s, h):
    return _sc(h, lambda a: season_info()[s]["H"] - a)


def overlap(s, t1, t2, edges):
    """h 축 칸 경계 edges(오름차순) 사이마다 [t1, t2]와 겹친 시간 (길이 len(edges) − 1). s는 API 맞춤용(쓰지 않음)."""
    e = np.asarray(edges, float)
    return np.maximum(0.0, np.minimum(e[1:], t2) - np.maximum(e[:-1], t1))


# ---------------- 관측 표에서 ----------------

def _cell(obs_dir, s):
    return OT.read_cell(os.path.join(obs_dir, f"{s}.jsonl.gz"))


@functools.lru_cache(None)
def pairs(obs_dir):
    """같은 사람·같은 점수 기록 쌍 — act_clock.pairs_of와 같은 거르기.
    A·B 사진, use, pct·점수·h·pid 있음, 서버 한섭 라벨 또는 미상, 시각을 [0, H]로 자름, 같은 (사람, 점수) 안에서 (시각, %) 오름차순으로 이웃 둘,
    간격 ≥ 1시간, t1 < H, dL = log10(p2/p1) ≥ −0.02, var = 판독 오차 식. pid는 결과에 넣지 않는다."""
    out = []
    for s in SEASONS:
        head, rows = _cell(obs_dir, s)
        H = float(head["H"])
        g = collections.defaultdict(dict)
        for r in rows:
            if r["kind"] not in ("A", "B") or not r.get("pct") or not r.get("score") or not r.get("pid") or not r.get("use"):
                continue
            if r.get("server") not in (None, "KR") or r.get("h") is None:
                continue
            g[(r["pid"], round(r["score"], 2))].setdefault((min(max(r["h"], 0.0), H), r["pct"]), r["kind"])
        for (_, sc), d in g.items():
            L = sorted(d)
            for (t1, p1), (t2, p2) in zip(L, L[1:]):
                if t2 - t1 < 1.0 or t1 >= H:
                    continue
                dL = math.log10(p2 / p1)
                if dL < -0.02:
                    continue
                var = ((0.005 / p1) ** 2 + (0.005 / p2) ** 2) / math.log(10) ** 2 + 0.004 ** 2
                out.append({"season": s, "t1": t1, "t2": t2, "dL": dL, "var": var, "p1": p1, "p2": p2, "score": sc,
                            "k1": d[(t1, p1)], "k2": d[(t2, p2)]})
    return out


def truth_table(s):
    """answer_key 한섭 정답 {pct: 점수} (wp10_diag.truth_table과 같음)."""
    return {q: v for (ss, q), v in WD.TRUTH.items() if ss == s}


@functools.lru_cache(None)
def _end_curve(s):
    pts = EPT.final_points(_STATE(s), "KR", ("A", "A2"))
    return EPT.Curve(pts) if len(pts) >= 2 else None


def end_points(s):
    """끝 화면 점 [(pct, 점수)] — explore_pct_transform.final_points(state, "KR", ("A", "A2")) 그대로 (best_record·mono_viol 뺌)."""
    return EPT.final_points(_STATE(s), "KR", ("A", "A2"))


@functools.lru_cache(None)
def photos(obs_dir):
    """컷 사진 줄 — A·B·C, use, 한섭 라벨 또는 미상, pct·점수·hp 있음. h는 A·B 촬영 시각(없으면 hp), C는 hp(게시 시각 = 촬영의 상한).
    pf_tab = 정답 표 보간의 최종 %, pf_pts = 끝 화면 점 곡선의 최종 % (덮인 구간 밖이면 None), P_* = log10(pf/pct). 거르기는 쓰는 쪽."""
    out = []
    for s in SEASONS:
        _, rows = _cell(obs_dir, s)
        tt, cv = truth_table(s), _end_curve(s)
        for r in rows:
            if r["kind"] not in ("A", "B", "C") or not r.get("use") or not r.get("pct") or not r.get("score") or r.get("hp") is None:
                continue
            if r.get("server") not in (None, "KR"):
                continue
            h = r["h"] if r["kind"] in ("A", "B") and r.get("h") is not None else r["hp"]
            pft = WD.final_pct(tt, r["score"]) if tt else None
            pfp = cv.pct_at(r["score"]) if cv else None
            lg = lambda pf: None if not pf else math.log10(pf / r["pct"])  # noqa: E731
            out.append({"season": s, "kind": r["kind"], "h": h, "hp": r["hp"], "pct": r["pct"], "score": r["score"],
                        "pf_tab": pft, "P_tab": lg(pft), "pf_pts": pfp, "P_pts": lg(pfp)})
    return out


@functools.lru_cache(None)
def top7(obs_dir):
    """숫자단 1~7위 줄 — 점수 있음, server == "KR" (명시 라벨만, 근거는 server_basis). use는 따지지 않고 같은 (시즌, rank, 점수, h) 중복은 하나."""
    out, seen = [], set()
    for s in SEASONS:
        _, rows = _cell(obs_dir, s)
        for r in rows:
            rk = r.get("rank")
            if r["kind"] not in ("A", "B", "C") or rk is None or not (1 <= rk <= 7) or not r.get("score") or r.get("server") != "KR":
                continue
            h = r["h"] if r["kind"] in ("A", "B") and r.get("h") is not None else r.get("hp")
            if h is None:
                continue
            k = (s, rk, r["score"], h)
            if k in seen:
                continue
            seen.add(k)
            out.append({"season": s, "rank": rk, "score": r["score"], "h": h, "kind": r["kind"], "server": r["server"],
                        "server_basis": r.get("server_basis")})
    return out


@functools.lru_cache(None)
def reward_tiers(obs_dir):
    """{시즌: 정렬된 %} — exclude·flag에 "reward_table"이 든 줄 (보상 구간 표 화면)의 pct."""
    out = {}
    for s in SEASONS:
        _, rows = _cell(obs_dir, s)
        v = sorted({r["pct"] for r in rows if r.get("pct") and ("reward_table" in (r.get("exclude") or []) or "reward_table" in (r.get("flag") or []))})
        out[s] = v
    return out


# ---------------- 견줄 곡선 ----------------

@functools.lru_cache(None)
def _F_theta():
    return _jload(os.path.join(R, "analysis/model_S42_main_prior.json"))["transform"]["theta"]


def F_P(u):
    """F 침수 꼴 f2pow (S42 모델 = S37~S41): log10(최종 % ÷ 지금 %) = −log10(w·u^p + (1−w)·u^q), w = sigmoid(θ0), p = e^θ1, q = e^θ2.
    u = h/H는 [0.25/120, 1]로 자른다 (wp10_diag.PF와 같음). 배열 가능."""
    th = _F_theta()
    w = 1 / (1 + math.exp(-th[0]))
    t = np.clip(np.asarray(u, float), 0.25 / 120.0, 1.0)
    r = -np.log10(w * t ** math.exp(th[1]) + (1 - w) * t ** math.exp(th[2]))
    return r.item() if r.ndim == 0 else r


@functools.lru_cache(None)
def F_shape():
    """F 컷 모양 사전 (한섭): analysis/model_S42_main_prior.json의 shape.servers.KR에서 lnp(ln % 격자)와 lr(ln 점수 비 = ln(점수(%)/점수(3%)), 3%에서 0)을 그대로 꺼낸다.
    tools/solo_raid/predict.py의 Shape(d)가 self.u = d["lnp"], self.lr = d["lr"]로 읽는 것과 같다 (격자 사이는 선형 보간)."""
    d = _jload(os.path.join(R, "analysis/model_S42_main_prior.json"))["shape"]["servers"]["KR"]
    return {"lnp": list(d["lnp"]), "lr": list(d["lr"])}


@functools.lru_cache(None)
def _m3():
    import m3
    s1 = _jload(os.path.join(R7, "stage1_M3_D1.json"))
    return m3.BASIS, s1["header"]["names"], s1["seasons"]


@functools.lru_cache(None)
def _m3_par(s):
    B, nm, ss = _m3()
    mean = ss[s]["mean"]
    g = lambda pre, n: np.array([mean[nm.index(f"{pre}{i}")] for i in range(n)])  # noqa: E731
    cl = np.linalg.solve(B["Bl12"], g("l", B["Bl12"].shape[0]))      # 1단계 l0..l11 = s12 격자 값 → 계수 (정방 B)
    cr = np.linalg.solve(B["Br12"], g("r", B["Br12"].shape[0]))      # r0..r12 = w12 격자 값 → 계수
    return cl, cr, mean[nm.index("a")], mean[nm.index("beta")]


@functools.lru_cache(None)
def _m3_Pgrid(s):
    B, _, _ = _m3()
    cl = _m3_par(s)[0]
    f = np.exp(B["Blf"] @ cl) * 2 * B["sg"]
    seg = 0.5 * (f[1:] + f[:-1]) * B["ds"]
    return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])


def m3_P(s, u, pf=3.0):
    """M3 기본판 1단계 시즌 s의 P(u) (log10, P(1) = 0) × (pf/3)^(β/ln10) — wp10_diag.part_curve_photo와 같은 식. u는 0.02 아래를 자른다. 배열 가능."""
    B, _, _ = _m3()
    beta = _m3_par(s)[3]
    r = np.interp(np.sqrt(np.maximum(np.asarray(u, float), 0.02)), B["sg"], _m3_Pgrid(s)) * (np.asarray(pf, float) / 3.0) ** (beta / math.log(10))
    return r.item() if np.ndim(r) == 0 else r


def m3_lnC(s, q):
    """M3 기본판 1단계 시즌 s의 한섭 최종 컷 ln 점수 (m3.lnC를 numpy로): a − R(w), w = ln(q/3), R(0) = 0 맞춤, 격자 밖은 끝 기울기 직선. q는 % (배열 가능)."""
    B, _, _ = _m3()
    cr, a = _m3_par(s)[1], _m3_par(s)[2]
    e = np.exp(B["Brf"] @ cr)
    Rc = np.concatenate([[0.0], np.cumsum(0.5 * (e[1:] + e[:-1]) * B["dw"])])
    i0, f0 = B["i0"], B["f0"]
    Rg = Rc - ((1 - f0) * Rc[i0] + f0 * Rc[i0 + 1])
    wg = B["wg"]
    w = np.log(np.asarray(q, float) / 3.0)
    Rq = np.where(w < wg[0], Rg[0] + e[0] * (w - wg[0]), np.where(w > wg[-1], Rg[-1] + e[-1] * (w - wg[-1]), np.interp(w, wg, Rg)))
    r = a - Rq
    return r.item() if r.ndim == 0 else r


@functools.lru_cache(None)
def _act():
    return _jload(os.path.join(R7, "wp11/act_profile_D1.json"))


def act_g(s=None):
    """활동 시간 g (KST 3시간 칸 8개 05-08…02-05, 하루 평균 1). s None이면 전체(full), 아니면 그 시즌의 g (앞 시즌만 쓴 expanding 판)."""
    a = _act()
    return list(a["full"]["g"] if s is None else a["g"][s]["g"])


# ---------------- 요약 CLI ----------------

def _git_head():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def summary(obs_dir):
    info = season_info()
    P, PH, T7, RT = pairs(obs_dir), photos(obs_dir), top7(obs_dir), reward_tiers(obs_dir)
    per = {}
    for s in SEASONS:
        H = info[s]["H"]
        ps = [p for p in P if p["season"] == s]
        ph = [p for p in PH if p["season"] == s]
        ep = end_points(s)
        t7 = [t for t in T7 if t["season"] == s]
        per[s] = {
            "info": info[s],
            "pairs": {"n": len(ps), "first_day_h_lt_24": sum(p["t1"] < 24 for p in ps), "last_day_h_ge_Hm24": sum(p["t2"] > H - 24 for p in ps),
                      "last_6h": sum(p["t2"] > H - 6 for p in ps)},
            "photos": {"A": sum(p["kind"] == "A" for p in ph), "B": sum(p["kind"] == "B" for p in ph), "C": sum(p["kind"] == "C" for p in ph),
                       "n": len(ph), "pf_tab": sum(p["pf_tab"] is not None for p in ph), "pf_pts": sum(p["pf_pts"] is not None for p in ph)},
            "end_points": {"n": len(ep), "pct_min": ep[0][0] if ep else None, "pct_max": ep[-1][0] if ep else None},
            "top7": {"n": len(t7), "by_rank": {str(k): sum(t["rank"] == k for t in t7) for k in range(1, 8)},
                     "by_server_basis": dict(collections.Counter(str(t["server_basis"]) for t in t7))},
            "reward_tiers_pct": RT[s],
        }

    def tot(f):
        return sum(f(v) for v in per.values())
    total = {"pairs": tot(lambda v: v["pairs"]["n"]), "pairs_first_day": tot(lambda v: v["pairs"]["first_day_h_lt_24"]),
             "pairs_last_day": tot(lambda v: v["pairs"]["last_day_h_ge_Hm24"]), "pairs_last_6h": tot(lambda v: v["pairs"]["last_6h"]),
             "photos": tot(lambda v: v["photos"]["n"]), "photos_pf_tab": tot(lambda v: v["photos"]["pf_tab"]),
             "photos_pf_pts": tot(lambda v: v["photos"]["pf_pts"]), "end_points": tot(lambda v: v["end_points"]["n"]),
             "top7": tot(lambda v: v["top7"]["n"])}
    return {"header": {"about": "WP12 공용 표(wp12_tables.py)의 시즌별 개수 요약 — 줄 단위 값은 담지 않는다 (시각·%·점수·개수만)",
                       "data_ver": C.DATA_VER, "date": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d"),
                       "code_git_head": _git_head()},
            "total": total, "seasons": per}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("summary")
    a.add_argument("--obs-dir", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    J = summary(args.obs_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=1)
    print(json.dumps(J["total"], ensure_ascii=False))


if __name__ == "__main__":
    main()
