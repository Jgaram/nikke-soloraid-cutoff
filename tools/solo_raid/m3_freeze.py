"""M3 시즌 고정 모델 파일 만들기 (docs/solo-raid-method.md 「WP7 정함」 1·2·6, 2026-11-07).

  python3 -I tools/solo_raid/m3_freeze.py --season S43 [--out-dir data/solo-raid/analysis] [--force]

만드는 것 (맞힐 시즌 S{n}):
  analysis/m3/S{n}/shape.json   꼴 칸 (m3_shape.py — 없을 때만 만든다. 이미 있으면 그대로 쓴다)
  analysis/m3/S{n}/prior.json   m3.py prior --target S{n} (모델 스위치 그대로) — 2단계 사전
  analysis/m3/S{n}/pc/          prior cache (창 적합 npz). m3_live.py가 복사해 쓴다 → 칸 예측의 사전이 prior.json과 같다
  analysis/model_S{n}_m3.json   모델 파일 (아래 꼴). code = 지금 HEAD (origin/main에 있어야 한다, tools/solo_raid에 커밋 안 된 변경이 없어야 한다)
  analysis/model_S{n}.json      F 대비 모델 — 없으면 앞 시즌 F 모델 파일을 그대로 옮긴다 (season만 바꿈 — 순서표 8과 같은 방식)

보정 풀 = 연구 22시즌 되돌려 보기 (r7/wp14/pred_k4flbudget_D1.json) + 운영 뒤 시즌별 되돌려 보기 analysis/m3/backtest/pred_S*.json
  (끝난 시즌 보강이 더한다 — m3_backtest.py) 중 S{n}보다 앞선 시즌 전부. 셈은 m3_calib.fit.
1단계 = 모델 스위치의 --stage1 (기본: analysis/m3/stage1_through_S{n-1}.json이 있으면 그것, 아니면 analysis/m3/stage1.json, 없으면 r7/stage1_M3_D1.json). S{n}보다 앞선 시즌이 모두 들어 있어야 한다.

시즌 일정 가정: --H (기본 120)·--c0h (기본 5). 꼴 칸이 05시 시작 120시간 시즌용이라 다른 일정이면 멈춘다 (HGAM 길은 아직 없음).
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import m3_calib as MC  # noqa: E402

KST = dt.timezone(dt.timedelta(hours=9))
A = os.path.join(ROOT, "data", "solo-raid", "analysis")
R7 = os.path.join(A, "r7")
NAME, VARIANT = "M3", "k4fl·budget·emp1"
PRIOR_ARGS = ["--dev", "--dev-parts", "flood,cut", "--dev-tau", "oos"]
PREDICT_ARGS = ["--K", "32", "--pre-full", "--dev", "--dev-parts", "flood,cut", "--dev-tau", "oos", "--engine", "budget"]
LIMIT_MIN = 75.0
POOL_RESEARCH = os.path.join(R7, "wp14", "pred_k4flbudget_D1.json")
ENV = {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type=",
       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
VALIDATION = {
    "kind": "개발 검증 — 22시즌 S21~S42 (시즌마다 앞 시즌만으로 예측, 시험 시즌 없음 — 원칙 4)",
    "source": "docs/solo-raid-method.md 「WP14 결과」 (14b k4flbudget, C1 emp1)",
    "score_13": 1.49, "score_10": 3.53, "score_note": "score = ⅓ × 초반 창 (T 18·24·36·48) 평균 |e| + ⅔ × 막판 창 (H−12·H−6·H) 평균 |e|, % (F 1.82 · 4.42)",
    "cov80_13": [0.73, 0.85], "cov80_10": [0.75, 0.74], "cov80_note": "보정 80% 구간 포함률 (초반 · 막판)",
    "cell_worst_min": 55.2,
    "g6": "자료를 다 표현하지 못함 (R2 1일차 새벽·2일차 오후 1~3%대 · R4 마감 칸 · R5 S22 10%) · 10%는 조정 시즌에 맞춰졌을 수 있음 · 모형 구간은 아주 좁고 보정 층이 앞 시즌 실제 오차로 넓힌다",
}


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(os.path.abspath(p), ROOT)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=1)
        f.write("\n")


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)


def snum(s):
    return int(s.lstrip("S"))


def code_commit():
    head = git("rev-parse", "HEAD").stdout.strip()
    if git("diff", "--quiet", "HEAD", "--", "tools/solo_raid").returncode != 0:
        sys.exit("멈춤: tools/solo_raid에 커밋 안 된 변경이 있다 — 커밋·push한 뒤 고정한다 (모델 파일의 code가 그 커밋을 가리킨다)")
    git("fetch", "-q", "origin", "main")
    if git("merge-base", "--is-ancestor", head, "origin/main").returncode != 0:
        sys.exit(f"멈춤: HEAD {head[:8]}가 origin/main에 없다 — push한 뒤 고정한다 (6시간 실행이 그 커밋을 꺼낸다)")
    return head


def stage1_path(a):
    if a.stage1:
        return a.stage1
    # 끝난 시즌 보강(m3_post.py stage1)이 시즌마다 만드는 고정용 사본이 있으면 그것 — 나중에 stage1.json이 자라도 이 모델 파일의 sha1이 안 깨진다
    for base in (os.path.join(os.path.abspath(a.out_dir), "m3"), os.path.join(A, "m3")):
        snap = os.path.join(base, f"stage1_through_S{snum(a.season) - 1}.json")
        if os.path.exists(snap):
            return snap
    own = os.path.join(A, "m3", "stage1.json")
    return own if os.path.exists(own) else os.path.join(R7, "stage1_M3_D1.json")


def pool_cells(season):
    """보정 풀 칸 (S{n}보다 앞선 시즌) + 입력 파일 sha1."""
    files = [POOL_RESEARCH] + sorted(glob.glob(os.path.join(A, "m3", "backtest", "pred_S*.json")))
    cells, inputs, seen = [], {}, set()
    for p in files:
        cs = [c for c in jload(p)["cells"] if snum(c["season"]) < snum(season) and c.get("server", "KR") == "KR"]
        dup = {c["season"] for c in cs} & seen
        if dup:
            sys.exit(f"멈춤: 보정 풀 시즌이 겹친다 {sorted(dup)} ({rel(p)})")
        seen |= {c["season"] for c in cs}
        cells += cs
        inputs[rel(p)] = sha1(p)
    return cells, inputs, sorted(seen, key=snum)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", required=True)
    ap.add_argument("--H", type=float, default=120.0)
    ap.add_argument("--c0h", type=float, default=5.0)
    ap.add_argument("--stage1", help="1단계 파일 (기본: analysis/m3/stage1.json 또는 r7/stage1_M3_D1.json)")
    ap.add_argument("--shape", help="꼴 파일 (기본 analysis/m3/S{n}/shape.json — 없으면 m3_shape.py로 만든다)")
    ap.add_argument("--out-dir", default=A, help="모델 파일·부속 파일을 둘 analysis 폴더 (리허설은 임시 폴더)")
    ap.add_argument("--no-code-check", action="store_true", help="리허설용: code 확인을 건너뛰고 지금 HEAD를 적는다")
    ap.add_argument("--no-f", action="store_true", help="F 대비 모델 파일을 만들지 않는다")
    ap.add_argument("--force", action="store_true", help="이미 있는 모델 파일을 덮어쓴다 (시즌 시작 뒤에는 쓰지 않는다)")
    a = ap.parse_args()
    s = a.season
    if (a.H, a.c0h) != (120.0, 5.0):
        sys.exit("멈춤: 꼴 칸은 05시 시작 120시간 시즌용이다 — 다른 일정은 꼴 길(HGAM)을 먼저 만든다")
    code = git("rev-parse", "HEAD").stdout.strip() if a.no_code_check else code_commit()
    out_dir = os.path.abspath(a.out_dir)
    mpath = os.path.join(out_dir, f"model_{s}_m3.json")
    if os.path.exists(mpath) and not a.force:
        sys.exit(f"멈춤: {rel(mpath)}가 이미 있다 — 시즌 안에서는 모델 파일을 바꾸지 않는다 (--force)")
    sub = os.path.join(out_dir, "m3", s)
    os.makedirs(sub, exist_ok=True)
    env = dict(os.environ, **ENV)
    # 1단계
    s1 = stage1_path(a)
    s1_seasons = sorted(jload(s1)["seasons"], key=snum)
    need = [x for x in s1_seasons if snum(x) < snum(s)]
    if not need or snum(need[-1]) != snum(s) - 1:
        sys.exit(f"멈춤: 1단계 {rel(s1)}에 바로 앞 시즌 S{snum(s) - 1}이 없다 — 끝난 시즌 보강에서 1단계에 더한다 (m3_backtest.py stage1)")
    # 꼴
    shp = a.shape or os.path.join(sub, "shape.json")
    if not os.path.exists(shp):
        sys.exit(f"멈춤: 꼴 파일 {rel(shp)}이 없다 — m3_shape.py로 만든다 (끝난 시즌 보강의 m3_backtest.py next가 부른다)")
    # 사전 + cache
    pc = os.path.join(sub, "pc")
    if os.path.exists(pc):
        shutil.rmtree(pc)
    prior = os.path.join(sub, "prior.json")
    if os.path.exists(prior):
        os.remove(prior)
    cmd = [sys.executable, "-I", os.path.join(HERE, "lab7", "m3.py"), "prior", "--stage1", s1, "--target", s, "--shape", shp] + PRIOR_ARGS \
        + ["--prior-cache", pc, "--out", prior]
    print("$", " ".join(rel(x) if x.startswith(ROOT) else x for x in cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(prior):
        sys.exit(f"m3.py prior 실패:\n{r.stderr[-2000:]}")
    pr = jload(prior)
    cache_files = {}
    for dp, _, fns in os.walk(pc):
        for fn in fns:
            p = os.path.join(dp, fn)
            cache_files[os.path.relpath(p, pc)] = sha1(p)
    # 보정
    cells, pool_inputs, pool_seasons = pool_cells(s)
    calib = MC.fit(cells, MC.load_truth(), MC.season_H(), s, a.H)
    calib["pool_inputs"] = pool_inputs
    # 모델 파일
    obs_err = os.path.join(R7, "obs_error.json")
    m = {
        "about": f"M3 고정 모델 — {s} 운영 예측용 (docs/solo-raid-method.md 「WP7 정함」). 시즌 안에서는 바꾸지 않는다. 대비는 F model_{s}.json.",
        "name": NAME, "variant": VARIANT, "season": s, "data_ver": "D1", "code": code,
        "made": {"at_kst": dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M"), "tool": "tools/solo_raid/m3_freeze.py",
                 "cmd": "m3_freeze.py " + " ".join(sys.argv[1:])},
        "schedule": {"H": a.H, "c0h": a.c0h},
        "stage1": {"path": rel(s1), "sha1": sha1(s1), "seasons": need},
        "shape": {"path": rel(shp), "sha1": sha1(shp)},
        "prior": {"path": rel(prior), "sha1": sha1(prior), "cache": rel(pc), "cache_files": cache_files,
                  "window": pr.get("window"), "K": pr.get("K"), "k": pr.get("k")},
        "obs_error": {"path": rel(obs_err), "sha1": sha1(obs_err)},
        "predict_args": PREDICT_ARGS, "limit_min": LIMIT_MIN,
        "calib": calib, "calib_pool_seasons": pool_seasons,
        "validation": VALIDATION,
        "fallback": f"F — analysis/model_{s}.json (칸마다 함께 낸다, M3가 실패하거나 {LIMIT_MIN:g}분을 넘으면 그 칸은 F 한섭 값)",
    }
    jdump(mpath, m)
    print(f"→ {rel(mpath)} · prior window {pr.get('window')} K {pr.get('K')} · 보정 풀 {pool_seasons[0]}~{pool_seasons[-1]} ({len(pool_seasons)}시즌)")
    # F 대비 모델
    if not a.no_f:
        fpath = os.path.join(out_dir, f"model_{s}.json")
        prev = os.path.join(A, f"model_S{snum(s) - 1}.json")
        if os.path.exists(fpath):
            print(f"F {rel(fpath)} 있음 — 그대로")
        elif os.path.exists(prev):
            f = jload(prev)
            src_sha = sha1(prev)
            f["season"] = s
            f["about"] = (f.get("about") or "") + f" · {s}: 앞 시즌 F 설정 그대로 옮김 (M3 운영의 대비, 2026-11-07 WP7). 원본 {os.path.basename(prev)} sha1 {src_sha}"
            jdump(fpath, f)
            print(f"F → {rel(fpath)} (원본 {os.path.basename(prev)} {src_sha[:8]})")
        else:
            print(f"!! F 대비 모델을 못 만듦: {rel(prev)} 없음")


if __name__ == "__main__":
    main()
