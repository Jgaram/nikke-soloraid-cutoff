"""솔로레이드 후처리 1.7: 서버별 점수–% 곡면에서 크게 벗어난 사진 줄을 표시한다 (지우지 않음, 모델 판독 0, 요청 0).

python3 -I tools/solo_raid/post_outlier.py STATE [--k 3.5] [--min-fit 12]

입력 (읽기만): core.jsonl은 읽지 않는다 — build_core.assemble()로 상류 파일(server·post_exclude·post_link·post_time …)에서
  같은 줄을 만든다 (surf 열 없음, 순환 없음). 그리고 obs.jsonl · season.json. 곡면 함수는 explore_surface.py.
  순서: post_all.sh에서 truth → post_outlier → build_core. build_core가 이 결과를 surf_z·surf_flag 열로 붙인다 (2026-10-16).
출력: STATE/post_outlier.jsonl — 사진 줄(oi)마다 한 줄, obs.jsonl과 같은 순서.
  STATE/post_outlier_summary.json — 잰 줄의 digest(core_digest, build_core가 맞춰 본다)와 요약
  {"oi", "resid", "resid_L", "z", "he", "flag": [...], "skip"}
  - resid: 점수 잔차(억, 관측 − 곡면, 0.1 반올림) · resid_L: log10 % 잔차(관측 − 곡면) · he: 잔차를 잰 곡면 시간(h)
  - z: 강건 z = 부호 × min(|resid|/척도_억, |resid_L|/척도_log%). 척도 = 그 서버 A 줄의 사람 하나씩 뺀(LOPO) 잔차 1.4826·MAD.
    두 축 모두에서 멀어야 크다 (곡선이 가파른 곳의 점수 차, 평평한 곳의 % 차를 과대평가하지 않으려고)
  - skip: 잔차를 재지 않은 이유 (not_use · no_score · no_surface(곡면 없는 서버) · server_unknown)
곡면 (explore_surface.SurfL): L = F(S) + c·(sqrt(he/H_END) − 1), 서버마다 A·독립 줄로 강건 적합 (KR 매듭 5, 그 밖 3).
  적합 줄이 --min-fit보다 적은 서버(S41은 GL)는 곡면을 만들지 않는다. 적합에 들어간 사람의 줄은 그 사람을 뺀 곡면으로 잰다.
시각: A는 촬영 시각. B·C는 구간 [h_lo, h_hi] 안에서 가장 잘 맞는 시각 (C의 h는 상한이다). 확실한 종료 후는 H_END.
  지지 범위: 점수는 적합 줄의 최소~최대, 시각은 적합 줄 he의 5% 분위 이후. 밖이면 곡면을 범위 끝으로 막아 한쪽 경계로만 판단한다
  (점수가 높을수록·시각이 이를수록 %는 작거나 같다 — 이 방향을 어길 때만 어긋남).
  같은 사진에 1~7위 점수가 있으면 top7 시각 구간(post_time.py의 h_lo_top7·h_hi_top7, 어긋남 0·구간과 겹칠 때만)으로 좁힌다.
flag
  - surface_low / surface_high: |z| > k. low = 그 %치고 점수가 낮다(= 그 점수치고 %가 크다), high = 반대
  - stale_fit: B·C 줄이 게시 시각(상한)에서는 |z| > k로 낮은데 구간 안 더 이른 시각에서는 맞는다 (늦게 올린 화면일 수 있음, 이상값 아님)
  - out_of_support: 점수가 그 서버 적합 줄의 점수 범위 밖 (잔차는 재지만 surface_* 표시는 하지 않는다)
  - no_server_fit: 서버 미상 줄이 곡면이 있는 어느 서버(S41: KR·JP)에도 |z| ≤ k로 맞지 않는다. 서버를 추정하지 않으며
    어느 서버에 가까운지도 남기지 않는다 (점수·%로 서버를 채우는 순환을 막는다). 곡면 없는 서버(GL 등)일 수 있다
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_core import assemble, digest  # noqa: E402
from explore_surface import SurfL, fit_rows, h_end_of, h_eff, lopo, mad, pts_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--k", type=float, default=3.5)
    ap.add_argument("--min-fit", type=int, default=12)
    args = ap.parse_args()
    st, K = args.state, args.k
    obs, core = assemble(st)
    dg = digest(core)
    with open(os.path.join(st, "season.json"), encoding="utf-8") as f:
        season = json.load(f)
    H = h_end_of(season)
    photos = [x for x in core if x["src"] == "photo"]
    if len(photos) != len(obs) or any(x["oi"] != k for k, x in enumerate(photos)):
        raise SystemExit("core.jsonl 사진 줄이 obs.jsonl과 맞지 않는다 — post_all.sh로 1단계를 다시 돌린다")

    surf = {}
    for sv in sorted(set(x["server"] for x in photos if x["server"])):
        rows = fit_rows(core, sv, H)
        if len(rows) < args.min_fit:
            continue
        nk = 5 if len(rows) >= 40 else 3
        mk = lambda nk=nk: SurfL("sqrt", nk, H)
        full = mk().fit(pts_of(rows, H))
        eS, eL, ms = lopo(rows, H, mk)
        sS = mad([e for e in eS if e is not None])
        sL = mad([e for e in eL if e is not None])
        surf[sv] = dict(full=full, lopo=ms, persons=set(x["person"] for x in rows), sS=sS, sL=sL, n=len(rows))
    def interval(x):
        if x["t_grade"] == "A" or x["day_lo"] == "post":
            return h_eff(x, H), h_eff(x, H)
        lo, hi = x["h_lo"], x["h_hi"]
        if x["h_lo_top7"] is not None:   # post_time이 이미 [h_lo, h_hi]와 겹친 구간만 남겼다
            lo, hi = x["h_lo_top7"], x["h_hi_top7"]
        return max(1.0, min(lo, H)), max(1.0, min(hi, H))

    def measure(x, sv, use_lopo):
        """(he, eS, eL, z, z_hi, support, verdict). verdict: fit · misfit · none(지지 범위 밖이라 판단 못 함)"""
        d = surf[sv]
        m = d["lopo"].get(x["person"]) if use_lopo and x["person"] in d["persons"] else None
        m = m or d["full"]
        lo, hi = interval(x)
        S = x["score"]
        s_lo, s_hi = m.S_rng
        h_min = m.h_sup
        support = s_lo <= S <= s_hi and hi >= h_min

        def zz(eS, eL):
            zs = [abs(eL) / d["sL"]] + ([abs(eS) / d["sS"]] if eS is not None else [])
            return (1 if eL > 0 else -1) * min(zs)   # eL > 0 (그 점수치고 % 큼) ⇔ eS > 0 (그 %치고 점수 높음)
        if support:
            he, eS, eL = m.band_resid(x["pct"], S, max(lo, h_min), hi)
            z = zz(eS, eL)
            z_hi = None
            if lo < hi:
                _, eS2, eL2 = m.band_resid(x["pct"], S, hi, hi)
                z_hi = zz(eS2, eL2)
            return he, eS, eL, z, z_hi, True, ("misfit" if abs(z) > K else "fit")
        # 지지 범위 밖: 범위 끝으로 막은 곡면은 한쪽 경계만 준다 (점수가 높을수록, 시각이 이를수록 %는 작거나 같다)
        Sc = min(max(S, s_lo), s_hi)
        he = max(hi, h_min)                     # 위쪽 경계(실제 % ≤ 경계 %)는 가장 늦은 시각에서 가장 느슨하다
        eL = m.resid_L(x["pct"], Sc, min(he, H))
        if S < s_lo and hi < h_min:
            allowed = None                      # 두 경계가 반대 방향 — 판단 못 함
        elif S < s_lo:
            allowed = -1                        # 실제 % ≥ 경계 % → %가 경계보다 확실히 작으면 어긋남
            eL = m.resid_L(x["pct"], Sc, max(lo, h_min) if hi >= h_min else h_min)
        else:
            allowed = +1                        # S > 최대 또는 시각이 이름 → 실제 % ≤ 경계 %
        z = eL / d["sL"]
        verdict = "misfit" if allowed is not None and z * allowed > K else "none"
        return min(he, H), None, eL, z, None, False, verdict

    out = []
    for x in photos:
        r = {"oi": x["oi"], "resid": None, "resid_L": None, "z": None, "he": None, "flag": []}
        if not x["use"]:
            r["skip"] = "not_use"
        elif x["score"] is None or not x["pct"]:
            r["skip"] = "no_score"
        elif x["server"] is None:
            r["skip"] = "server_unknown"
            v = [measure(x, sv, False) for sv in surf]
            if v and all(t[6] == "misfit" for t in v):
                r["flag"].append("no_server_fit")
            zs = [abs(t[3]) for t in v if t[6] != "none"]
            if zs:
                r["z"] = round(min(zs), 2)   # 판단할 수 있는 서버 곡면 중 가장 가까운 |z| (어느 서버인지는 남기지 않음)
        elif x["server"] not in surf:
            r["skip"] = "no_surface"
        else:
            he, eS, eL, z, z_hi, support, verdict = measure(x, x["server"], True)
            r.update(resid=None if eS is None else round(eS, 1), resid_L=round(eL, 3), z=round(z, 2), he=round(he, 2))
            if not support:
                r["flag"].append("out_of_support")
            if verdict == "misfit":
                r["flag"].append("surface_high" if z > 0 else "surface_low")
            if z_hi is not None and z_hi < -K and verdict == "fit":
                r["flag"].append("stale_fit")
        out.append(r)

    p = os.path.join(st, "post_outlier.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    summ = {"core_digest": dg, "k": K, "min_fit": args.min_fit,
            "surfaces": {sv: {"n": d["n"], "c": round(d["full"].c, 4), "scale_eok": round(d["sS"], 3),
                              "scale_logpct": round(d["sL"], 4)} for sv, d in surf.items()},
            "skip": dict(sorted(collections.Counter(r.get("skip", "-") for r in out).items())),
            "flag": dict(sorted(collections.Counter(f for r in out for f in r["flag"]).items()))}
    with open(os.path.join(st, "post_outlier_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{season['name']} 곡면: " + " · ".join(f"{sv} A {d['n']}줄 c={d['full'].c:.2f} 척도 {d['sS']:.1f}억/{d['sL']:.3f}"
                                                for sv, d in surf.items()) + f" · k={K}")
    print("skip:", dict(collections.Counter(r.get("skip", "-") for r in out)))
    print("flag:", dict(collections.Counter(f for r in out for f in r["flag"])))
    print(f"top7로 좁힌 B·C 줄: {sum(1 for x in photos if x['h_lo_top7'] is not None and x['use'] and x['t_grade'] in ('B', 'C'))}")
    print(f"→ {p}")


if __name__ == "__main__":
    main()
