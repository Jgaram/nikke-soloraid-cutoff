"""M3 운영 예측 한 칸 (docs/solo-raid-method.md 「WP7 정함」 4, 2026-11-07).

  python3 -I tools/solo_raid/m3_live.py STATE T --out DIR [--model MODEL.json] [--limit-min N] [--work W] [--keep]

  STATE  진행 중 시즌 state (data/solo-raid/S{n}/state — 사본이어도 된다, 리허설). season.json의 c_start·end를 쓴다.
  T      챌린지 시작 후 시간 (6시간 칸 경계, 예 24).
  --out  결과 폴더 (live: data/solo-raid/S{n}/predict/T{T}). m3_T{T}.json · m3_T{T}.txt · m3_run.log를 쓴다.
         같은 폴더에 F 예측 predict_T{T}.json이 먼저 있어야 대체(F)를 쓸 수 있다 (live_tick.sh finish 순서).
  --model  M3 고정 모델 파일. 기본 data/solo-raid/analysis/model_{시즌}_m3.json (m3_freeze.py가 만든다).
  --limit-min  M3 단계 전체 시간 한도 (기본: 모델 파일의 limit_min, 없으면 75분). 넘으면 끊고 대체(F).

하는 일 (연구 WP14b와 같은 길 — RUNPLAN_wp14bd.md 「준비」·「실행」):
  1. 모델 파일의 고정 커밋(code)으로 코드(CODE_PATHS — tools/와 post_all.sh가 부르는 experiments 도구)만 든 git worktree를 만들고, 그 안 data/solo-raid/는 이 레포 것을 가리키는 링크로 채운다
     (맞힐 시즌만 STATE의 시즌 폴더로). 그래서 main의 연구 코드가 바뀌어도 운영 예측은 고정 커밋 코드로 돈다.
  2. 고정 파일 확인: stage1·shape·prior·obs_error sha1, 시즌 일정 (H·c0h)이 모델 가정과 같은지. 다르면 대체(F).
  3. lab7/trunc.py (T까지 사본 + post_all.sh) → lab7/obs_table.py build → lab7/m3.py predict (모델의 스위치, --engine budget,
     prior cache는 고정 폴더를 복사해 씀 — 그래서 사전이 고정 prior.json과 같다).
  4. 보정 층 (m3_calib.apply, 모델 파일의 calib) → m3_T{T}.json.
  실패·시간 초과·고정 파일 불일치면 같은 폴더의 F 한섭 값을 status "fallback"으로 적는다 (사용자 결정 2026-11-07).

종료 코드: 0 = M3 예측 (status ok) · 3 = 대체(F)로 씀 · 1 = 아무것도 못 씀 (F도 없음).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import m3_calib as MC  # noqa: E402

# 고정 커밋에서 꺼내는 코드 (post_all.sh의 server_classify는 experiments 아래에 있다)
CODE_PATHS = ["tools", "experiments/2026-10-15-solo-raid-low-cost/tools"]
PCTS = ("0.5", "1.0", "2.0", "3.0", "5.0", "10.0")
ENV = {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type=",
       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
KST = dt.timezone(dt.timedelta(hours=9))
LOG = None


def now_kst():
    return dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M")


def log(msg):
    line = f"[{dt.datetime.now(KST).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ab(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(p, o):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, p)


class Fail(Exception):
    """M3를 낼 수 없음 — 대체(F)로 간다."""


def season_info(state):
    import predict as PR  # noqa: E402 — tools/solo_raid/predict.py (main 판 — 시즌 일정만 읽는다)
    sj, c0, H = PR.season_info(state)
    return (sj.get("name") if isinstance(sj, dict) else sj), c0, float(H)


def check_frozen(model):
    """고정 파일 sha1 확인. 다르면 Fail."""
    bad = []
    for k in ("stage1", "shape", "prior", "obs_error"):
        e = model.get(k) or {}
        p = ab(e["path"])
        if not os.path.exists(p):
            bad.append(f"{k} 없음 ({e['path']})")
        elif sha1(p) != e["sha1"]:
            bad.append(f"{k} sha1 다름 ({e['path']})")
    pc = ab(model["prior"]["cache"])
    for fn, h in (model["prior"].get("cache_files") or {}).items():
        p = os.path.join(pc, fn)
        if not os.path.exists(p) or sha1(p) != h:
            bad.append(f"prior cache {fn} 없음 또는 다름")
    if bad:
        raise Fail("고정 파일이 바뀜: " + "; ".join(bad))


def make_code(model, work, season, season_dir):
    """고정 커밋 tools/만 든 worktree + data/solo-raid 링크. 경로."""
    wt = os.path.join(work, "code")
    subprocess.run(["git", "-C", ROOT, "worktree", "prune"], check=False)
    subprocess.run(["git", "-C", ROOT, "worktree", "add", "-q", "--detach", "--no-checkout", wt, model["code"]], check=True)
    subprocess.run(["git", "-C", wt, "checkout", "-q", model["code"], "--"] + CODE_PATHS, check=True)
    d = os.path.join(wt, "data", "solo-raid")
    os.makedirs(d)
    src = os.path.join(ROOT, "data", "solo-raid")
    for name in os.listdir(src):
        if name != season:
            os.symlink(os.path.join(src, name), os.path.join(d, name))
    os.symlink(os.path.abspath(season_dir), os.path.join(d, season))
    return wt


def drop_code(work):
    wt = os.path.join(work, "code")
    if os.path.exists(wt):
        subprocess.run(["git", "-C", ROOT, "worktree", "remove", "--force", wt], check=False, capture_output=True)
    subprocess.run(["git", "-C", ROOT, "worktree", "prune"], check=False)


def ensure_deps(wt):
    probe = subprocess.run([sys.executable, "-I", "-c", "import numpyro, jax, scipy, arviz"], capture_output=True)
    if probe.returncode == 0:
        return
    log("lab7 패키지 설치 (pip, 약 2분)")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", os.path.join(wt, "tools", "solo_raid", "lab7", "requirements.txt")],
                   check=True, capture_output=True)


def run(cmd, deadline, logf):
    """하위 프로세스 하나 (자기 프로세스 묶음). 한도를 넘으면 묶음을 끊고 Fail."""
    left = deadline - time.time()
    if left <= 0:
        raise Fail("시간 한도를 넘음")
    env = dict(os.environ, **ENV)
    with open(logf, "a", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(cmd) + "\n")
        lf.flush()
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rc = p.wait(timeout=left)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise Fail("시간 한도를 넘음 — M3 계산을 끊음")
    if rc != 0:
        raise Fail(f"{os.path.basename(next(x for x in cmd if x.endswith('.py')))} 실패 (종료 {rc}) — m3_run.log")


def m3_cell(model, state, season, T, H, work, deadline):
    """M3 한 칸 → (칸 목록, info, prior 일치)."""
    wt = make_code(model, work, season, os.path.dirname(os.path.abspath(state)))
    ensure_deps(wt)
    L = os.path.join(wt, "tools", "solo_raid", "lab7")
    py = [sys.executable, "-u", "-I"]
    logf = LOG
    w = os.path.join(work, "w")
    os.makedirs(w, exist_ok=True)
    t0 = time.time()
    run(py + [os.path.join(L, "trunc.py"), "--work", w, "--jobs", "1", "--seasons", f"{season}-{season}", "--T", f"{T:g}"], deadline, logf)
    run(py + [os.path.join(L, "obs_table.py"), "build", "--work", w, "--out", os.path.join(w, "obs7"), "--seasons", f"{season}-{season}", "--jobs", "1"],
        deadline, logf)
    obs = os.path.join(w, "obs7", season, f"T{T:g}.jsonl.gz")
    if not os.path.exists(obs):
        raise Fail(f"관측 표가 없다 ({season} T{T:g}) — T가 시즌 T 격자에 없거나 사본을 못 만듦")
    t_prep = time.time() - t0
    pc = os.path.join(work, "pc")
    shutil.copytree(ab(model["prior"]["cache"]), pc)
    out = os.path.join(work, "pred.json")
    run(py + [os.path.join(L, "m3.py"), "predict", "--stage1", ab(model["stage1"]["path"]), "--obs-dir", os.path.join(w, "obs7"),
              "--seasons", season, "--T", f"{T:g}", "--shape", ab(model["shape"]["path"]), "--prior-cache", pc, "--out", out]
        + list(model["predict_args"]), deadline, logf)
    res = jload(out)
    cells = [c for c in res["cells"] if c["season"] == season and float(c["T"]) == float(T)]
    info = next((i for i in res["info"] if i["season"] == season and float(i["T"]) == float(T)), None)
    if not cells or info is None:
        raise Fail("m3.py가 칸을 내지 않음")
    fz = jload(ab(model["prior"]["path"]))
    match = res.get("priors", {}).get(season) == fz
    info = dict(info, prep_secs=round(t_prep, 1), tool_head_m3=res.get("made", {}).get("tool_head"))
    return cells, info, match


def points_m3(cells, model, T):
    pts = {}
    for c in cells:
        q = float(c["pct"])
        i80, ci = MC.apply(model["calib"], T, q, c["pred"], c.get("i80"))
        pts[f"{q:.1f}"] = {"median": c["pred"], "i80": i80, "i80_model": c.get("i80"), "i95": c.get("i95"), "conv": c.get("conv"), "calib": ci}
    return pts


def points_f(fpath):
    o = jload(fpath)
    kr = ((o.get("servers") or {}).get("KR") or {}).get("points") or {}
    return {k: {"median": v.get("median"), "i80": v.get("i80"), "i95": v.get("i95"), "src": "F"} for k, v in kr.items() if k in PCTS}


def fmt(v):
    return "-" if v is None else f"{v:.1f}"


def summary(o):
    pts = o["servers"]["KR"]["points"]
    parts = []
    for k in ("1.0", "3.0", "10.0"):
        p = pts.get(k)
        if p and p.get("median"):
            iv = p.get("i80") or [None, None]
            parts.append(f"{float(k):g}% {fmt(p['median'])} (80% {fmt(iv[0])}~{fmt(iv[1])})")
    head = "M3" if o["status"] == "ok" else "대체(F)"
    lines = [f"[요약] {o['season']} T{o['T']:g} 한섭 {head} — " + " · ".join(parts)]
    if o["status"] != "ok":
        lines.append(f"!! M3를 내지 못해 이 칸은 F 한섭 값 — {o['fallback']['reason']}")
    for wmsg in o.get("warnings", []):
        lines.append("!! " + wmsg)
    f = o.get("fit") or {}
    if o["status"] == "ok":
        lines.append(f"엔진 {f.get('engine')} · 수렴 {f.get('conv')} · rhat {f.get('rhat_max')} · 재시도 건너뜀 {f.get('retry_skipped')} · 55분 끊김 {f.get('timeout')}"
                     f" · 칸 {f.get('wall', {}).get('total')}초 · 준비 {f.get('prep_secs')}초")
    lines.append(f"모델 {o['model'].get('file')} (code {str(o['model'].get('code'))[:8]}) · 전체 {o['run']['secs']}초")
    return "\n".join(lines) + "\n"


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state")
    ap.add_argument("T", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--limit-min", type=float)
    ap.add_argument("--work")
    ap.add_argument("--keep", action="store_true", help="작업 폴더를 지우지 않는다 (점검용)")
    a = ap.parse_args()
    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)
    LOG = os.path.join(a.out, "m3_run.log")
    open(LOG, "w").close()
    T = float(a.T)
    season, c0, H = season_info(a.state)
    mpath = ab(a.model) if a.model else os.path.join(ROOT, "data", "solo-raid", "analysis", f"model_{season}_m3.json")
    work = a.work or tempfile.mkdtemp(prefix=f"m3live_{season}_T{T:g}_")
    os.makedirs(work, exist_ok=True)
    o = {"about": "M3 운영 예측 — 한섭 최종 컷 (tools/solo_raid/m3_live.py, docs/solo-raid-method.md 「WP7 정함」)",
         "season": season, "T": T, "T_kst": (c0 + dt.timedelta(hours=T)).strftime("%Y-%m-%d %H:%M"), "c_start": c0.strftime("%Y-%m-%d %H:%M:%S"), "H": H,
         "status": None, "fallback": None, "warnings": [], "model": {"file": os.path.relpath(mpath, ROOT)}, "servers": {"KR": {"points": {}}},
         "fit": None, "run": {}}
    log(f"M3 예측 — {season} T{T:g} (KST {o['T_kst']}) · 모델 {o['model']['file']} · 작업 {work}")
    try:
        if not os.path.exists(mpath):
            raise Fail(f"M3 모델 파일 없음 ({o['model']['file']}) — m3_freeze.py로 고정한다")
        model = jload(mpath)
        o["model"] = {"name": model.get("name"), "variant": model.get("variant"), "file": os.path.relpath(mpath, ROOT), "sha1": sha1(mpath),
                      "season": model.get("season"), "code": model.get("code"), "train": model.get("stage1", {}).get("seasons")}
        if model.get("season") != season:
            raise Fail(f"모델은 {model.get('season')}용인데 이 state는 {season} — 그 시즌 모델 파일이 필요하다")
        limit = a.limit_min or model.get("limit_min") or 75.0
        deadline = t_start + limit * 60
        sch = model.get("schedule") or {}
        c0h = c0.hour + c0.minute / 60
        if sch and (float(sch.get("H")) != H or float(sch.get("c0h")) != c0h):
            raise Fail(f"시즌 일정 (H {H:g}·시작 {c0h:g}시)이 모델 가정 (H {sch.get('H')}·시작 {sch.get('c0h')}시)과 다름 — 꼴을 다시 만들어야 한다")
        check_frozen(model)
        cells, info, match = m3_cell(model, a.state, season, T, H, work, deadline)
        if not match:
            o["warnings"].append("사전(prior)이 고정 prior.json과 다르다 — prior cache를 확인")
        o["servers"]["KR"]["points"] = points_m3(cells, model, T)
        o["fit"] = {k: info.get(k) for k in ("conv", "rhat_max", "ess_min", "divergences", "retried", "retry_skipped", "timeout", "approx", "engine",
                                             "secs", "map_secs", "wall", "n", "N", "K", "k", "prep_secs", "tool_head_m3")}
        o["status"] = "ok"
        rc = 0
    except Exception as e:  # noqa: BLE001 — 어떤 실패든 대체로 간다
        reason = str(e) if isinstance(e, Fail) else f"{type(e).__name__}: {e}"
        log(f"M3 실패 → 대체(F): {reason}")
        fpath = os.path.join(a.out, f"predict_T{T:g}.json")
        o["fallback"] = {"reason": reason, "source": "F", "file": os.path.basename(fpath)}
        if os.path.exists(fpath):
            o["servers"]["KR"]["points"] = points_f(fpath)
            o["status"] = "fallback"
            rc = 3
        else:
            o["status"] = "failed"
            o["fallback"]["reason"] += " · F 예측도 없음"
            rc = 1
    finally:
        if not a.keep:
            drop_code(work)
            if not a.work:
                shutil.rmtree(work, ignore_errors=True)
    o["run"] = {"tool_head": subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None,
                "state": os.path.relpath(os.path.abspath(a.state), ROOT), "obs_sha1": sha1(os.path.join(a.state, "obs.jsonl")) if os.path.exists(os.path.join(a.state, "obs.jsonl")) else None,
                "secs": round(time.time() - t_start, 1), "made_kst": now_kst()}
    jdump(os.path.join(a.out, f"m3_T{T:g}.json"), o)
    txt = summary(o)
    with open(os.path.join(a.out, f"m3_T{T:g}.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt, end="")
    log(f"끝 — status {o['status']} · {o['run']['secs']}초")
    sys.exit(rc)


if __name__ == "__main__":
    main()
