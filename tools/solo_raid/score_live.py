"""솔로레이드 진행 중에 낸 예측을 끝난 뒤 채점한다 (2026-10-16, 모델 0, 요청 0).

python3 -I tools/solo_raid/score_live.py S{n}

- 읽는 것: data/solo-raid/S{n}/predict/T*/predict_T*.json (실시간 세션이 시점마다 커밋한 예측, predict_run.sh 결과 그대로 — F)
  와 m3_T*.json (M3 운영 예측, m3_live.py — S43부터 웹에 나가는 예측, WP7 2026-11-07). 둘은 방법(method)으로 나눠 적는다 (M3 줄은 「M3 …」).
- 정답: 정답 표 규칙(truth_grade.build — criteria.json 판 1: 한섭 3·10% 집계표, 그 밖 정답 곡선 strong). 채점은 xval.score와 같다.
  S42 채점은 criteria.json 판 1(sha1 80c63e2f)로 고정한다 (분석 4.0a). 4절 후보 고르기 기준(criteria_v2.json)은 여기서 쓰지 않는다.
- 쓰는 것: data/solo-raid/S{n}/predict/score.json, 표준 출력에 T × 한섭 1·3·10% 오차·80% 포함 표.
  T마다 그 예측을 낸 도구 커밋(run.tool_head)·prior·core sha1을 `runs`에 남긴다 — 진행 중에 도구가 바뀌었으면 tool_head별로 나눠 적는다 (분석 4.0).
  방법(줄의 method: 고정 모델 이름(model) 또는 「prior 하나」)도 T마다 남긴다 — S42는 T24까지 S41 한 시즌 prior, T30부터 모델 F (사용자 결정 2026-10-18). 같은 날 T6~T24를 F로 다시 냈다 (예측 JSON recalc — 재산정은 따로 읽는다).
  방법이 둘 이상이면 방법별로 나눠 읽는다 (같은 시즌이라도 다른 방법의 시험이다).
- **미래 시험 (split.json 판 2 v2.future_protocol, 사용자 2026-10-17):** S42부터 모든 새 시즌은 끝나면 도구·설정을 고치기 전에
  이 채점을 먼저 한 번 한다 → 수집 끝 표시 → 개발. 그래서 정답 표는 수집이 끝나기 전 시즌도 만든다(truth_grade.build는 등록부와 상관없이 이름으로 받는다).
  진행 중 예측은 그때의 기본 도구·prior로 낸 것이라 그 자체가 정직한 표본 밖 시험이다 — 결과를 보고 도구를 고치면 그 뒤 성능은 「미검증」이다.
- --final은 판 1 시험 시즌(test_v1)이 잠겨 있던 때의 것이다. 판 2에는 잠긴 시즌이 없어 쓸 일이 없다.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import split as SP  # noqa: E402

DATA = os.path.join(ROOT, "data", "solo-raid")
CRIT = os.path.join(DATA, "analysis", "criteria.json")


def fam_of(r):
    return "M3" if str(r.get("method", "")).startswith("M3") else "F"


def score_m3(files, akey, name, crit):
    """M3 운영 예측 (m3_T*.json, WP7 2026-11-07) 채점. 줄 꼴은 xval.score 줄과 같은 열쇠 (i50·등급은 없음) + i80_model·in80_model·src.
    대체(F) 칸 (status fallback)은 method 「M3 대체(F) …」, src "F" — 웹에 나간 그대로 센다."""
    tol = crit["tol"]
    rows, runs = [], []
    for p in files:
        o = json.load(open(p, encoding="utf-8"))
        T = float(o["T"])
        md = o.get("model") or {}
        st = o.get("status")
        method = f"M3 ({md.get('file')})" if st == "ok" else f"M3 대체(F) ({md.get('file')})"
        pts = ((o.get("servers") or {}).get("KR") or {}).get("points") or {}
        for r in akey:
            if r["season"] != name or r["server"] != "KR":
                continue
            d = pts.get(str(float(r["pct"])))
            if not d or d.get("median") is None or not d.get("i80"):
                continue
            q = float(r["pct"])
            tl = tol.get(f"{q:g}", tol["aux"])
            med = d["median"]
            rec = {"scheme": "live", "train": list(md.get("train") or []), "season": name, "T": T, "server": "KR", "pct": q, "tol": tl,
                   "pred": round(med, 3), "i80": d["i80"], "i80_model": d.get("i80_model"), "i95": d.get("i95"),
                   "truth": r["truth"], "truth_basis": r["basis"], "shaky": r["shaky"],
                   "table": r["table"] and r["table"]["value"], "curve": r["curve"]["value"], "curve_grade": r["curve"]["grade"],
                   "method": method, "src": "F" if d.get("src") == "F" or st != "ok" else "M3",
                   "tool_head": (o.get("run") or {}).get("tool_head"), "code": md.get("code")}
            if r["truth"] is not None:
                tv = r["truth"]
                e = med / tv - 1
                im = d.get("i80_model")
                rec.update(err=round(e, 5), ok=abs(e) <= tl, in80=d["i80"][0] <= tv <= d["i80"][1],
                           in80_model=None if not im or None in im else im[0] <= tv <= im[1],
                           in95=None if not d.get("i95") or None in d["i95"] else d["i95"][0] <= tv <= d["i95"][1],
                           w80=round(d["i80"][1] / d["i80"][0] - 1, 4))
            rows.append(rec)
        runs.append({"T": T, "file": os.path.relpath(p, ROOT), "tool_head": (o.get("run") or {}).get("tool_head"), "method": method,
                     "status": st, "fallback": (o.get("fallback") or {}).get("reason"), "model_sha1": md.get("sha1"), "code": md.get("code"),
                     "secs": (o.get("run") or {}).get("secs"), "conv": (o.get("fit") or {}).get("conv")})
    return rows, runs


def m3_summary(rows, name):
    """연구 score와 같은 셈 (lab7/score.py third 판): ⅓ × 초반 창 평균 |e| + ⅔ × 막판 창 평균 |e|, 무리 1·3%·10%. 첫 실전 시험 기록용."""
    import predict as PR
    H = float(PR.season_info(os.path.join(DATA, name, "state"))[2])
    out = {"about": "M3 미래 시험 (WP7 4) — 끝나기 전에 커밋된 예측을 끝난 뒤 채점. 셈은 lab7/score.py (third)", "H": H, "groups": {}}
    for g, pcts in (("1·3%", (1.0, 3.0)), ("10%", (10.0,))):
        rr = [r for r in rows if r["pct"] in pcts and r.get("err") is not None]
        mn = [r for r in rr if r["T"] in (18.0, 24.0, 36.0, 48.0)]
        en = [r for r in rr if r["T"] in (H - 12, H - 6, H)]
        if not mn or not en:
            continue
        am = sum(abs(r["err"]) for r in mn) / len(mn) * 100
        ae = sum(abs(r["err"]) for r in en) / len(en) * 100
        out["groups"][g] = {"score": round(am / 3 + ae * 2 / 3, 2), "main": round(am, 2), "end": round(ae, 2),
                            "cov_main": round(sum(r["in80"] for r in mn) / len(mn), 2), "cov_end": round(sum(r["in80"] for r in en) / len(en), 2),
                            "fallback_cells": sum(1 for r in mn + en if r.get("src") == "F"), "n": len(mn) + len(en)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("season")
    SP.add_final_arg(ap)
    a = ap.parse_args()
    SP.guard_args(a)
    name = a.season
    if SP.is_test(name) and not a.final:
        raise SystemExit(f"{name}은 잠긴 시험 시즌이다 (split.json test) — --final에서만")
    import truth_grade as TG   # 감시를 켠 뒤에 불러온다
    import xval as XV
    crit = json.load(open(CRIT, encoding="utf-8"))
    akey = TG.build([name], final=a.final)["rows"]
    files = sorted(glob.glob(os.path.join(DATA, name, "predict", "T*", "predict_T*.json")),
                   key=lambda p: float(re.search(r"predict_T([\d.]+)\.json$", p).group(1)))
    m3files = sorted(glob.glob(os.path.join(DATA, name, "predict", "T*", "m3_T*.json")),
                     key=lambda p: float(re.search(r"m3_T([\d.]+)\.json$", p).group(1)))
    if not files and not m3files:
        raise SystemExit(f"{name}: predict/T*/predict_T*.json·m3_T*.json 없음")
    rows, runs = [], []
    for p in files:
        out = json.load(open(p, encoding="utf-8"))
        T = float(out["T"]) if not isinstance(out["T"], str) else float(re.sub(r"[^\d.]", "", out["T"]))
        run = out.get("run") or {}
        prior = run.get("prior_sha1") or run.get("prior")
        md = out.get("model")
        method = f"model {md['name']} ({md.get('file')})" if md else "prior 하나"
        rs = XV.score(out, akey, name, T, "live", [f"prior {prior}"], crit)
        for r in rs:
            r["tool_head"] = run.get("tool_head")
            r["method"] = method
        rows += rs
        st = run.get("state") or {}
        runs.append({"T": T, "file": os.path.relpath(p, ROOT), "tool_head": run.get("tool_head"), "tool_dirty": run.get("tool_dirty"),
                     "prior_sha1": prior, "core_sha1": st.get("core_sha1"), "rows_le_T": st.get("rows_le_T"), "method": method,
                     "model_sha1": run.get("model_sha1"), "m1_c": (md or {}).get("m1_c")})
    m3_rows, m3_runs = score_m3(m3files, akey, name, crit)
    rows += m3_rows
    runs += m3_runs
    heads = []
    for r in runs:
        if r["tool_head"] not in heads:
            heads.append(r["tool_head"])
    with open(CRIT, "rb") as f:
        crit_sha1 = hashlib.sha1(f.read()).hexdigest()
    res = {"about": "진행 중 예측 채점 (tools/solo_raid/score_live.py). 줄 꼴은 xval.json rows와 같다 (+ tool_head). "
                    "runs = T마다 예측을 낸 도구 커밋·prior·core — 도구가 진행 중에 바뀌었으면 tool_heads별로 나눠 읽는다",
           "season": name, "final": a.final, "criteria": os.path.relpath(CRIT, ROOT), "criteria_sha1": crit_sha1,
           "tool_heads": heads, "methods": list(dict.fromkeys(r["method"] for r in runs)), "runs": runs,
           "files": [os.path.relpath(p, ROOT) for p in files + m3files], "rows": rows}
    if m3_rows:
        res["m3_summary"] = m3_summary(m3_rows, name)
    out_p = os.path.join(DATA, name, "predict", "score.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{name} 진행 중 예측 채점 (한섭, e = 예측/정답 − 1, [80% 포함])")
    for fam in ("M3", "F"):   # M3 운영(WP7)부터 둘을 따로 — M3가 웹에 나간 예측, F는 대비·참고
        fr = [r for r in rows if fam_of(r) == fam]
        if not fr:
            continue
        print(f"[{fam}]" + (" — 웹에 나간 예측 (대체(F) 칸 포함, 칸 끝에 F)" if fam == "M3" else ""))
        for T in sorted({r["T"] for r in fr}):
            cells = []
            for q in (1.0, 3.0, 10.0):
                r = next((r for r in fr if r["T"] == T and r["server"] == "KR" and r["pct"] == q), None)
                cells.append("—" if not r or r.get("err") is None else f"{r['err']:+.1%}[{'O' if r['in80'] else 'X'}] {r.get('grade') or ('F' if r.get('src') == 'F' else '')}")
            ru = next((r for r in runs if r["T"] == T and fam_of(r) == fam), {})
            print(f"  T={T:>4g}  1%: {cells[0]:16} 3%: {cells[1]:16} 10%: {cells[2]:16} 도구 {(ru.get('tool_head') or '?')[:8]} · {ru.get('method', '?')}")
    if res.get("m3_summary"):
        ms_ = res["m3_summary"]
        print("M3 score (연구와 같은 셈 — ⅓ 초반 T 18·24·36·48 + ⅔ 막판 H−12·H−6·H, |e| 평균 %): "
              + " · ".join(f"{g} {v['score']} (초반 {v['main']} · 막판 {v['end']}, 80% 포함 {v['cov_main']} · {v['cov_end']}, 대체 칸 {v['fallback_cells']})"
                           for g, v in ms_["groups"].items()))
    if len(heads) > 1:
        print(f"!! 예측을 낸 도구 커밋이 {len(heads)}개다 — tool_head별로 나눠 적는다: " + ", ".join((h or "?")[:8] for h in heads))
    ms = res["methods"]
    if len(ms) > 1:
        print(f"!! 방법이 {len(ms)}개다 — 방법별로 나눠 읽는다: " + " / ".join(ms))
    print("→", os.path.relpath(out_p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
