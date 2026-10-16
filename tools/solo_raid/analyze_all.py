"""솔로레이드 여러 시즌 분석을 한 명령으로 (할 일 3.A3, 2026-10-16, 모델 0, 요청 0).

python3 -I tools/solo_raid/analyze_all.py [--draws 600] [--boot 60] [--work DIR] [--skip-xval] [--viz DIR]

순서 (같은 입력이면 같은 결과 — 시드 고정, 결과 파일에 실행 시각을 넣지 않는다. 시각은 git 커밋으로 본다)
1. 후처리 최신 확인: 개발 시즌 state마다 post_stamp --check. 입력·도구가 바뀌었으면 post_all.sh를 그 도장의 시즌 전용 입력으로 다시 돈다.
   시험 시즌은 건드리지 않는다 (수집 세션이 돌린다). 수집이 끝나지 않은 시즌(registry.collect_status ≠ 완료)도 건드리지 않는다 —
   다른 세션이 쓰는 중이다 (여러 시즌 세션이 함께 돌 때, 2026-10-16).
   다른 세션과 생성 파일(registry.json·analysis/*.json·끝난 시즌의 state 후처리 파일)이 git에서 부딪히면 origin 쪽을 받고 이 스크립트를 다시 돌린다 (결정적이라 같은 입력이면 같은 결과).
2. 등록부: registry.py --write → data/solo-raid/registry.json
3. 정답 표: truth_grade.py → data/solo-raid/analysis/answer_key.json (개발 시즌만)
4. 여러 시즌 prior: 학습 가능한 개발 시즌 전부 → data/solo-raid/analysis/prior_dev.json
   (참고용. predict_run.sh 기본 prior는 바꾸지 않는다 — 기본을 바꾸는 판정은 criteria.json 규칙, 개발 시즌 8개 이상에서)
5. 검증: xval.py (forward·loso) → data/solo-raid/analysis/xval_dev.json
6. (--viz DIR일 때만) 시각화: viz.py --out DIR → DIR/solo-raid-analysis.html (세션 임시 폴더. 커밋하지 않는다)
끝나면 바뀐 파일을 커밋한다 (이 스크립트는 커밋하지 않는다).
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import split as SP  # noqa: E402
import post_stamp  # noqa: E402
import registry  # noqa: E402

DATA = os.path.join(ROOT, "data", "solo-raid")
AN = os.path.join(DATA, "analysis")


def py(*args):
    print("$", " ".join(os.path.relpath(a, ROOT) if os.path.isabs(a) else a for a in args), flush=True)
    r = subprocess.run([sys.executable, "-I", *args], cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"실패: {args[0]} (종료 코드 {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--boot", type=int, default=60)
    ap.add_argument("--work", default=None)
    ap.add_argument("--skip-xval", action="store_true")
    ap.add_argument("--viz", metavar="DIR", help="끝에 시각화 HTML을 DIR에 다시 만든다 (viz.py, 세션 임시 폴더)")
    a = ap.parse_args()
    sp = SP.load_split()
    prog = registry.progress_status()
    # 1
    for d in sorted(os.listdir(DATA), key=lambda x: (len(x), x)):
        st = os.path.join(DATA, d, "state")
        if not (d.startswith("S") and os.path.exists(os.path.join(st, "obs.jsonl"))):
            continue
        if SP.role(d, sp) == "test":
            print(f"{d}: 시험 시즌 — 건너뜀")
            continue
        cs = registry.collect_status(d, prog)
        if cs != "완료":
            print(f"{d}: 수집 {cs} — 건너뜀 (그 세션이 후처리한다)")
            continue
        s, why = post_stamp.check(st)
        print(f"{d}: 후처리 {s} {why}")
        if s != "fresh":
            if not os.path.exists(os.path.join(st, "obs.jsonl")):
                continue
            ex = []
            sp_ = os.path.join(st, post_stamp.STAMP)
            if os.path.exists(sp_):
                stamp = json.load(open(sp_, encoding="utf-8"))
                for k, v in stamp.get("extra", {}).items():
                    ex += [f"--{k}", os.path.join(ROOT, v["path"])]
                if stamp.get("opts"):
                    ex += stamp["opts"].split()
            r = subprocess.run(["bash", os.path.join(HERE, "post_all.sh"), st] + ex, cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(f"{d} post_all 실패: {r.stderr[-1500:]}")
            print(f"{d}: post_all 다시 돌림 → {post_stamp.check(st)[0]}")
    # 2·3
    py(os.path.join(HERE, "registry.py"), "--write")
    py(os.path.join(HERE, "truth_grade.py"))
    # 4
    reg = json.load(open(os.path.join(DATA, "registry.json"), encoding="utf-8"))["seasons"]
    train = [r["season"] for r in reg if r["role"] == "dev" and r["tier"] in ("full", "shape_only")]
    py(os.path.join(HERE, "fit_prior_multi.py"), ",".join(train), "--out", os.path.join(AN, "prior_dev.json"), "--boot", str(a.boot))
    # 5
    if not a.skip_xval:
        args = [os.path.join(HERE, "xval.py"), "--scheme", "forward,loso", "--draws", str(a.draws), "--boot", str(a.boot)]
        if a.work:
            args += ["--work", a.work]
        py(*args)
    # 6
    if a.viz:
        py(os.path.join(HERE, "viz.py"), "--out", os.path.abspath(a.viz))
    print("끝. 바뀐 파일: git status data/solo-raid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
