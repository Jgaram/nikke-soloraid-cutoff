"""솔로레이드 진행 중에 낸 예측을 끝난 뒤 채점한다 (2026-10-16, 모델 0, 요청 0).

python3 -I tools/solo_raid/score_live.py S{n}

- 읽는 것: data/solo-raid/S{n}/predict/T*/predict_T*.json (실시간 세션이 시점마다 커밋한 예측, predict_run.sh 결과 그대로).
- 정답: 정답 표 규칙(truth_grade.build — criteria.json 판 1: 한섭 3·10% 집계표, 그 밖 정답 곡선 strong). 채점은 xval.score와 같다.
  S42 채점은 criteria.json 판 1(sha1 80c63e2f)로 고정한다 (분석 4.0a). 4절 후보 고르기 기준(criteria_v2.json)은 여기서 쓰지 않는다.
- 쓰는 것: data/solo-raid/S{n}/predict/score.json, 표준 출력에 T × 한섭 1·3·10% 오차·80% 포함 표.
  T마다 그 예측을 낸 도구 커밋(run.tool_head)·prior·core sha1을 `runs`에 남긴다 — 진행 중에 도구가 바뀌었으면 tool_head별로 나눠 적는다 (분석 4.0).
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
    if not files:
        raise SystemExit(f"{name}: predict/T*/predict_T*.json 없음")
    rows, runs = [], []
    for p in files:
        out = json.load(open(p, encoding="utf-8"))
        T = float(out["T"]) if not isinstance(out["T"], str) else float(re.sub(r"[^\d.]", "", out["T"]))
        run = out.get("run") or {}
        prior = run.get("prior_sha1") or run.get("prior")
        rs = XV.score(out, akey, name, T, "live", [f"prior {prior}"], crit)
        for r in rs:
            r["tool_head"] = run.get("tool_head")
        rows += rs
        st = run.get("state") or {}
        runs.append({"T": T, "file": os.path.relpath(p, ROOT), "tool_head": run.get("tool_head"), "tool_dirty": run.get("tool_dirty"),
                     "prior_sha1": prior, "core_sha1": st.get("core_sha1"), "rows_le_T": st.get("rows_le_T")})
    heads = []
    for r in runs:
        if r["tool_head"] not in heads:
            heads.append(r["tool_head"])
    with open(CRIT, "rb") as f:
        crit_sha1 = hashlib.sha1(f.read()).hexdigest()
    res = {"about": "진행 중 예측 채점 (tools/solo_raid/score_live.py). 줄 꼴은 xval.json rows와 같다 (+ tool_head). "
                    "runs = T마다 예측을 낸 도구 커밋·prior·core — 도구가 진행 중에 바뀌었으면 tool_heads별로 나눠 읽는다",
           "season": name, "final": a.final, "criteria": os.path.relpath(CRIT, ROOT), "criteria_sha1": crit_sha1,
           "tool_heads": heads, "runs": runs, "files": [os.path.relpath(p, ROOT) for p in files], "rows": rows}
    out_p = os.path.join(DATA, name, "predict", "score.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{name} 진행 중 예측 채점 (한섭, e = 예측/정답 − 1, [80% 포함])")
    for T in sorted({r["T"] for r in rows}):
        cells = []
        for q in (1.0, 3.0, 10.0):
            r = next((r for r in rows if r["T"] == T and r["server"] == "KR" and r["pct"] == q), None)
            cells.append("—" if not r or r.get("err") is None else f"{r['err']:+.1%}[{'O' if r['in80'] else 'X'}] {r.get('grade') or ''}")
        th = next((r["tool_head"] for r in runs if r["T"] == T), None)
        print(f"  T={T:>4g}  1%: {cells[0]:16} 3%: {cells[1]:16} 10%: {cells[2]:16} 도구 {(th or '?')[:8]}")
    if len(heads) > 1:
        print(f"!! 예측을 낸 도구 커밋이 {len(heads)}개다 — tool_head별로 나눠 적는다: " + ", ".join((h or "?")[:8] for h in heads))
    print("→", os.path.relpath(out_p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
