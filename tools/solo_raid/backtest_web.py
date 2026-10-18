"""지난 시즌을 고정 모델 F로 되돌려 본 결과를 웹용으로 줄인다 (2026-10-18 사용자 요청 — 「S35를 S34 이하 자료만으로 예측했다면」, 모델 0, 요청 0).

python3 -I tools/solo_raid/backtest_web.py --main MAIN.json --ref REF.json [--out data/solo-raid/analysis/backtest_F.json] [--m1-known-diff WHY]

입력은 xval 결과 둘 (레포에 커밋하지 않는 큰 파일 — 세션 임시에 둔다. 출력에 sha1을 남긴다):
- MAIN: F의 1·3% 설정 (docs/solo-raid-analysis.md 「4.5」 근거 실행과 같고 T만 6시간마다)
    python3 -I tools/solo_raid/lab4/m5run.py run --patch no_interval -- --targets S21-S41 --T 6,12,...,120 --jobs 4 --draws 2000 --seed 1
            --scheme recent5 --level-update '{"unknown": false}' --predict-opt eps_wrong=0 --work W --out MAIN.json
- REF: F의 5~30% 설정 (M2a — 최근 5시즌 prior 그대로, 덧붙임 없음)
    python3 -I tools/solo_raid/xval.py --targets S21-S41 --T 6,12,...,120 --jobs 4 --draws 2000 --seed 1 --scheme recent5 --work W --out REF.json

운영(predict.py --model)과 같은 셈으로 합친다:
- 1·3%: MAIN 줄 + M1 (구간 넓히기). 배율은 맞힐 시즌보다 **앞** 시즌의 한섭 1·3% |z| 80% 분위 (freeze_model.m1_table — 표의 T는 12~96 일곱 개만,
  그 사이는 선형 보간·밖은 끝값, predict.m1_factor). 맞힐 시즌 = S42일 때의 표가 model_S42.json과 같은지 확인해 출력에 적는다.
- 10%: REF 줄 그대로.
- 맞힐 시즌마다 학습은 그 앞 5시즌(recent5)이고 T까지 게시된 글만 남긴 사본(tr)으로 냈다. 다만 F라는 방법 자체는 S21~S41 정답을 보고 골랐으므로 개발 검증이다.
출력: 시즌 → 학습 시즌·H·c_start·M1 배율·서버 → 지점 → 정답(근거)·허용 오차·줄 [T, 가운데, 80% 아래, 80% 위, 신뢰도 등급, 쓴 사람 수].
종료 뒤 T(T > H)는 종료 시각으로 잘린 같은 예측이라 뺀다.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import freeze_model as FZ  # noqa: E402
import predict as PR  # noqa: E402
import split as SP  # noqa: E402

D = os.path.join(ROOT, "data", "solo-raid")
M1_T = (12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0)   # model_S{n}.json의 M1 표와 같은 T
MAIN_P = (1.0, 3.0)
REF_P = (10.0,)
SERVERS = ("KR", "JP", "GL")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha1_of(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def r2(v):
    return None if v is None else round(float(v), 2)


def m1_for(rows, target):
    """맞힐 시즌의 M1 표 {T: 배율} — 앞 시즌의 표 T 줄만으로 (운영 모델 파일과 같은 규칙)."""
    tab = FZ.m1_table([r for r in rows if float(r["T"]) in M1_T], target)
    return {f"{T:g}": round(c, 4) for T, (c, _, _) in tab.items()}


def main():
    ap = argparse.ArgumentParser(description="F로 되돌려 본 지난 시즌 예측을 웹용으로 줄인다")
    ap.add_argument("--main", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default=os.path.join(D, "analysis", "backtest_F.json"))
    ap.add_argument("--m1-known-diff", metavar="WHY", help="운영 모델 파일과 M1 표가 다른 까닭을 알 때 (예: 그 모델을 고정한 뒤 앞 시즌 자료를 고침). "
                    "멈추지 않고 대조 결과에 까닭을 적는다. 운영 모델 파일은 바꾸지 않는다")
    a = ap.parse_args()
    xm, xr = jload(a.main), jload(a.ref)
    rm = [r for r in xm["rows"] if r.get("scheme") == "recent5"]
    rr = [r for r in xr["rows"] if r.get("scheme") == "recent5"]
    if not rm or not rr:
        raise SystemExit("recent5 줄이 없다")
    folds = {f["target"]: f["train"] for f in xm["folds"] if f.get("scheme") == "recent5"}
    folds_r = {f["target"]: f["train"] for f in xr["folds"] if f.get("scheme") == "recent5"}
    if folds != folds_r:
        raise SystemExit("MAIN과 REF의 학습 시즌이 다르다")
    Ts = sorted({float(r["T"]) for r in rm})
    if Ts != sorted({float(r["T"]) for r in rr}):
        raise SystemExit("MAIN과 REF의 T가 다르다")

    # 운영 모델 파일과 대조: 맞힐 시즌을 그 모델 시즌으로 두면 같은 M1 표가 나와야 한다 (앞 시즌 = 이 결과의 맞힐 시즌 전부일 때)
    checks = []
    for mp in sorted(p for p in os.listdir(os.path.join(D, "analysis")) if p.startswith("model_S") and p.count("_") == 1):
        md = jload(os.path.join(D, "analysis", mp))
        if not set(md["m1"]["source"]["seasons"]) <= set(folds):
            continue
        mine = m1_for([r for r in rm if r["season"] in md["m1"]["source"]["seasons"]], md["season"])
        same = all(abs(mine.get(k, 1.0) - v) < 1e-3 for k, v in md["m1"]["T"].items())
        checks.append({"model": mp, "same": same, **({"why": a.m1_known_diff} if not same and a.m1_known_diff else {})})
        print(f"M1 대조 {mp}: {'같음' if same else '다름 — ' + json.dumps(mine)}")
        if not same and not a.m1_known_diff:
            raise SystemExit("M1 표가 운영 모델 파일과 다르다 — 입력 xval 설정을 확인한다")

    out_seasons = {}
    for tgt in sorted(folds, key=SP.snum):
        sj = jload(os.path.join(D, tgt, "state", "season.json"))
        c0 = dt.datetime.strptime(sj["c_start"][:16], "%Y-%m-%d %H:%M")
        end = dt.datetime.strptime(sj["end"][:16], "%Y-%m-%d %H:%M")
        H = round((end - c0).total_seconds() / 3600)
        m1 = m1_for(rm, tgt)
        mdl = {"m1": {"T": m1}}
        srv = {}
        for sv in SERVERS:
            pts = {}
            for src, ps in ((rm, MAIN_P), (rr, REF_P)):
                for p in ps:
                    R = sorted((r for r in src if r["season"] == tgt and r.get("server", "KR") == sv and float(r["pct"]) == p),
                               key=lambda r: float(r["T"]))
                    if not R:
                        continue
                    rows = []
                    for r in R:
                        if float(r["T"]) > H:   # 종료 뒤 T는 종료 시각으로 잘린 같은 예측이라 싣지 않는다 (S38 H=113의 T114·120)
                            continue
                        m, iv = r.get("pred"), r.get("i80")
                        if not m or not iv or None in iv or min(iv) <= 0:
                            continue
                        lo, hi = iv
                        if p in MAIN_P:   # M1 — predict.widen과 같은 셈
                            c = PR.m1_factor(mdl, float(r["T"])) if m1 else 1.0   # 앞 시즌 줄이 없으면(S21) 넓히지 않는다
                            lo, hi = m * (lo / m) ** c, m * (hi / m) ** c
                        rows.append([round(float(r["T"]), 2), r2(m), r2(lo), r2(hi), r.get("grade"), round(r.get("n_persons") or 0)])
                    t = next((r for r in R if r.get("truth") is not None), None)
                    pts[f"{p:g}"] = {"truth": r2(t["truth"]) if t else None, "basis": t.get("truth_basis") if t else None,
                                     "tol": R[0].get("tol"), "rows": rows}
            if pts:
                srv[sv] = pts
        out_seasons[tgt] = {"train": folds[tgt], "c_start": sj["c_start"], "H": H, "m1": m1, "srv": srv}

    src = lambda p, x: {"file": os.path.basename(p), "sha1": sha1_of(p), "tool_head": (x.get("run") or {}).get("tool_head"),  # noqa: E731
                        "tools_dirty": (x.get("run") or {}).get("tools_dirty"), "lab_patches": (x.get("run") or {}).get("lab_patches")}
    out = {"about": ("지난 시즌을 고정 모델 F(분석 4.3 마무리 — 운영 predict.py --model과 같은 설정)로 되돌려 본 예측. 맞힐 시즌마다 그 앞 5시즌으로 prior를 만들고 "
                     "T까지 게시된 글만 남긴 사본으로 냈다. 1·3%는 앞 시즌만으로 정한 M1을 얹었고 10%는 ref(M2a) 실행. F 자체는 이 시즌들의 정답을 보고 골랐으므로 개발 검증이다. "
                     "tools/solo_raid/backtest_web.py가 씀 — 웹(tools/publish/site.py)의 「지난 시즌 살펴보기」가 읽는다."),
           "model": "F", "T": [round(T, 2) for T in Ts], "pcts": {"main": list(MAIN_P), "ref": list(REF_P)},
           "source": {"main": src(a.main, xm), "ref": src(a.ref, xr), "m1_check": checks},
           "seasons": out_seasons}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"→ {os.path.relpath(a.out, ROOT)} ({os.path.getsize(a.out) / 1024:.0f} KB) · 시즌 {len(out_seasons)}개 · T {len(Ts)}개")


if __name__ == "__main__":
    main()
