"""할 일 4.3 마무리 (2026-10-18) — M5 설정값 변형·10% 새 후보(N10)를 돌리는 lab 실행기.

predict.py·xval.py·fit_prior.py는 고치지 않는다 (S42 기본 동작 — 분석 4.0a). 이 프로세스 안에서만 함수를 바꿔 끼우고 xval.main을 부른다.
xval은 fork로 일꾼을 띄우므로 바꿔 끼운 것이 그대로 물려진다. 사전 고정: docs/solo-raid-analysis.md 「4.3 마무리 — 사전 고정」.

  python3 -I tools/solo_raid/lab4/m5run.py run [--patch NAME ...] [--mixdir D] [--emdir D] -- <xval.py 인자>
  python3 -I tools/solo_raid/lab4/m5run.py mkmix MIXDIR --work W --targets S21-S41 [--T 12,18,...] [--gate-min 5] [--jobs 4]
  python3 -I tools/solo_raid/lab4/m5run.py mkem EMDIR [--seasons S19-S41]

run의 patch (여럿 가능, 「warn」은 늘 켬):
  warn         한섭 줄에 등급 사유(warn)·서버 미상 줄 수(mixed_rows)·섞임 민감도 변화(c_shift)·×2 창 관측 수(near_obs)를 남긴다 (③·⑦ 사후 계산용)
  eps_all      ② 「점 하나가 틀림」을 모든 사람 수에 켠다. 크게 어긋난 점 거르기(screen)의 4명 문턱은 그대로 둔다
  sf_half      ⑥ % 변환 꼴 외삽 불확실성(sigma_form)을 절반으로 (predict·m_update 둘 다)
  no_interval  ⑥ prior 맞추기(fit_prior.load_units)에서 구간 칸을 뺀다. prior 캐시는 WORK/alt_noint에 따로 둔다
  near10       N10 — 목표 q ≥ 5% 칸마다 중앙 변환 최종 %가 q의 ×2 안인 점의 사람이 3명 이상이면 창 밖 점 무게를 10⁻⁶배로
  em_borrow    ① site-em — borrow_kind em을 여러 학습 시즌으로: EMDIR/{시즌}_summary.json의 em 사전을 사이트마다 같은 무게 평균
  mix          ① mix — unknown mode mix의 확률 파일을 MIXDIR/{시즌}/trunc_T{T}/post_server_mix.jsonl에서 읽는다 (mkmix로 만든 것)
섞임 민감도(⑦)는 patch가 아니라 xval의 --predict-opt contam=[0.2]로 켠다. 서버 미상 줄 방식은 --unknown-opt mode=…, borrow_kind=em.
결과 파일의 run.lab_patches에 patch·mixdir·emdir를 적는다.
"""
import argparse
import collections
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, TOOLS)
import predict as PR  # noqa: E402
import xval as XV  # noqa: E402
import fit_prior as FP  # noqa: E402
import fit_prior_multi as FM  # noqa: E402

LN2 = math.log(2)
NEAR_MIN_PERSONS = 3
NEAR_Q_MIN = 5.0
NEAR_FAR_W = 1e-6
PATCHES = ("warn", "eps_all", "sf_half", "no_interval", "near10", "em_borrow", "mix")


def season_of_state(st):
    """…/S40/state → S40, …/trunc/S40/trunc_T24/state → S40."""
    p = os.path.normpath(st)
    parts = p.split(os.sep)
    for x in reversed(parts):
        if x.startswith("S") and x[1:].isdigit():
            return x
    raise ValueError(f"시즌 이름을 못 읽음: {st}")


def apply(patches, mixdir=None, emdir=None):
    if "warn" in patches:
        orig_score = XV.score

        def score_w(out, akey, tgt, T, fold_name, train, crit):
            rows = orig_score(out, akey, tgt, T, fold_name, train, crit)
            for r in rows:
                sv = out["servers"].get(r["server"]) or {}
                r["warn"] = (sv.get("warning") or {}).get("reasons")
                r["mixed_rows"] = sv.get("mixed_rows")
                d = (sv.get("points") or {}).get(str(float(r["pct"]))) or {}
                r["near_obs"] = d.get("near_obs")
                cs = sv.get("contam")
                if cs and cs.get("f"):
                    r["c_shift"] = {f: round(e["points"][str(float(r["pct"]))]["shift"], 5)
                                    for f, e in cs["f"].items() if str(float(r["pct"])) in e.get("points", {})}
            return rows
        XV.score = score_w
        XV.SLIM_KEYS = tuple(XV.SLIM_KEYS) + ("warn", "mixed_rows", "near_obs", "c_shift")
    if "eps_all" in patches:
        orig_screen = PR.Predictor.screen

        def screen4(self, srv, O, tr, W, ys):
            old = PR.N_SCREEN
            PR.N_SCREEN = 4
            try:
                return orig_screen(self, srv, O, tr, W, ys)
            finally:
                PR.N_SCREEN = old
        PR.Predictor.screen = screen4
        PR.N_SCREEN = 10 ** 9   # run()의 「점 하나가 틀림」 조건 npers < N_SCREEN이 늘 참
    if "sf_half" in patches:
        orig_sf = PR.sigma_form
        PR.sigma_form = lambda t, h, H: 0.5 * orig_sf(t, h, H)
    if "no_interval" in patches:
        orig_lu = FP.load_units

        def lu(state, *a, **k):
            season, H, obs, units = orig_lu(state, *a, **k)
            return season, H, obs, [u for u in units if u["cens"] is None]
        FP.load_units = lu
        orig_pf = XV.prior_for

        def pf(train, work, *a, **k):
            return orig_pf(train, os.path.join(work, "alt_noint"), *a, **k)
        XV.prior_for = pf
    if "near10" in patches:
        orig_w = PR.Predictor.weights

        def weights_n10(self, srv, O, tr):
            W, ys = orig_w(self, srv, O, tr)
            sh = self.shapes[srv]
            for gi, (q, u) in enumerate(zip(self.G, self.uG)):
                if q < NEAR_Q_MIN - 1e-9:
                    continue
                near = {i for i, uy in enumerate(ys) if sh.inside(uy) and abs(uy - u) <= LN2}
                if len({O[i]["person"] for i in near}) < NEAR_MIN_PERSONS:
                    continue
                for i in range(len(O)):
                    if i not in near:
                        W[i][gi] *= NEAR_FAR_W
            return W, ys
        PR.Predictor.weights = weights_n10
    if "em_borrow" in patches:
        if not emdir:
            raise SystemExit("em_borrow에는 --emdir")
        orig_br = PR.borrowed_ratio

        def br(bstate, kind="label"):
            if kind != "em":
                return orig_br(bstate, kind)
            sts = bstate if isinstance(bstate, (list, tuple)) else [bstate]
            acc = collections.defaultdict(lambda: collections.defaultdict(float))
            cnt = collections.Counter()
            for st in sts:
                with open(os.path.join(emdir, f"{season_of_state(st)}_summary.json"), encoding="utf-8") as f:
                    pr = json.load(f)["prior"]
                for site, d in pr.items():
                    cnt[site] += 1
                    for c, v in d.items():
                        acc[site][c] += v
            return {site: {c: v / cnt[site] for c, v in d.items()} for site, d in acc.items()}, None
        PR.borrowed_ratio = br
    if "mix" in patches:
        if not mixdir:
            raise SystemExit("mix에는 --mixdir")
        orig_au = PR.assign_unknown

        def au(state, rows, lab_pers, u):
            if u.get("mode") == "mix":
                u = dict(u, mix_file=os.path.join(mixdir, season_of_state(state), os.path.basename(os.path.dirname(os.path.normpath(state))),
                                                  "post_server_mix.jsonl"))
            return orig_au(state, rows, lab_pers, u)
        PR.assign_unknown = au


def cmd_run(argv):
    if "--" not in argv:
        raise SystemExit("run … -- <xval 인자>")
    k = argv.index("--")
    ap = argparse.ArgumentParser(prog="m5run.py run")
    ap.add_argument("--patch", action="append", default=[])
    ap.add_argument("--mixdir")
    ap.add_argument("--emdir")
    a = ap.parse_args(argv[:k])
    patches = ["warn"] + [p for p in a.patch if p != "warn"]
    bad = [p for p in patches if p not in PATCHES]
    if bad:
        raise SystemExit(f"모르는 patch: {bad}")
    xargs = argv[k + 1:]
    if "--out" not in xargs:
        raise SystemExit("--out을 준다 (analysis/에 쓰지 않는다)")
    apply(patches, a.mixdir, a.emdir)
    sys.argv = ["xval.py"] + xargs
    rc = XV.main()
    out = xargs[xargs.index("--out") + 1]
    with open(out, encoding="utf-8") as f:
        res = json.load(f)
    res["run"]["lab_patches"] = {"patches": patches, "mixdir": a.mixdir, "emdir": a.emdir}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
        f.write("\n")
    return rc


def _mkmix_one(job):
    state, out, gate_min = job
    os.makedirs(os.path.dirname(out), exist_ok=True)
    code = ("import sys; sys.path.insert(0, %r); import server_mix as M; M.GATE_MIN_KR = %d; "
            "sys.argv = ['server_mix.py', %r, '--gate', '--out', %r]; M.main()") % (TOOLS, gate_min, state, out)
    p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    g = None
    try:
        with open(out + "_summary.json", encoding="utf-8") as f:
            g = json.load(f).get("gate")
    except OSError:
        pass
    return state, p.returncode, (g or {}).get("pass"), (p.stderr or "")[-300:]


def cmd_mkmix(argv):
    ap = argparse.ArgumentParser(prog="m5run.py mkmix")
    ap.add_argument("mixdir")
    ap.add_argument("--work", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--T", default="12,18,24,36,48,72,96")
    ap.add_argument("--gate-min", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    jobs = []
    for s in sorted(XV.parse_targets(a.targets), key=lambda x: int(x[1:])):
        for T in a.T.split(","):
            st = os.path.join(a.work, "trunc", s, f"trunc_T{float(T):g}", "state")
            if not os.path.exists(os.path.join(st, "core.jsonl")):
                raise SystemExit(f"절단 사본 없음: {st}")
            jobs.append((st, os.path.join(a.mixdir, s, f"trunc_T{float(T):g}", "post_server_mix"), a.gate_min))
    import multiprocessing as mp
    npass = 0
    with mp.get_context("fork").Pool(a.jobs) as p:
        for st, rc, ok, err in p.imap(_mkmix_one, jobs):
            npass += bool(ok)
            if rc:
                print("!!", st, rc, err)
    print(f"mkmix {a.mixdir}: {len(jobs)}칸, gate 통과 {npass} (최소 unit {a.gate_min})")


def cmd_mkem(argv):
    ap = argparse.ArgumentParser(prog="m5run.py mkem")
    ap.add_argument("emdir")
    ap.add_argument("--seasons", default="S19-S41")
    a = ap.parse_args(argv)
    os.makedirs(a.emdir, exist_ok=True)
    for s in sorted(XV.parse_targets(a.seasons), key=lambda x: int(x[1:])):
        p = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "server_mix.py"), FM.state_of(s), "--out", os.path.join(a.emdir, s)],
                           capture_output=True, text=True)
        print(s, p.returncode, (p.stdout or "").strip()[-160:], (p.stderr or "")[-200:])


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "mkmix", "mkem"):
        raise SystemExit(__doc__)
    return {"run": cmd_run, "mkmix": cmd_mkmix, "mkem": cmd_mkem}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main() or 0)
