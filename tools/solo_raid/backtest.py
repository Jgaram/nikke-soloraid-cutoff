"""솔로레이드 컷 예측 되돌아보기 점검: 끝난 시즌 자신에게 T마다 predict를 돌려 정답 곡선과 견준다 (모델 판독 0, 요청 0).

python3 -I tools/solo_raid/backtest.py STATE --prior PRIOR.json [--T 3,6,12,...,119] [--draws 1000] [--kfold 5]
        [--variants in,cf,tr] [--work DIR] [--out DIR] [--leak-check] [--boot 100]
        [--posts F --snap DIR --parsed F --alt F --runs DIR]   (tr 변형: post_all.sh에 넘길 시즌 전용 입력)
        [--predict-opt key=value ...]  (predict 옵션: tau_shape=0, no_season_diff=1, grades=AB, p_update=0, c_delay=2 ...)
        [--unknown none|site|mix] [--unknown-servers KR] [--borrow-state S] [--borrow-kind label|em] [--no-gate] [--fallback site|none]
        [--robust mean|wmedian] [--top-cut Q]   (서버 미상 줄 처리·강건 처리 비교, 2026-10-16. 기본 none = 예전과 같은 결과)

**표본 안 점검이다 — 이 수치를 예측 성능으로 인용하지 않는다** (반박 검토, 2026-10-16).
prior(최종 곡선 모양·% 변환)와 정답이 같은 시즌에서 나온다. 최종 곡선 모양은 정답 곡선 그 자체라(fit_prior.build_shape는 persons를
받아도 무시하고 정답 점 전부를 쓴다 — 정답 곡선은 사람 단위로 나눌 수 없다) 관측 하나만 있어도 규모만 맞추면 모든 % 지점이 맞는다.
T들은 누적·겹침이라 독립 시험이 아니다 (S41 tr T=18·24·36의 라벨 한섭 결과는 같은 관측 하나에서 나왔다).

변형 (누수를 줄이는 순서)
- in : prior를 시즌 전체로 만든 그대로 (가장 낙관적)
- cf : 교차 적합 — 사람을 K겹으로 나눠, 각 사람의 관측은 그 사람을 뺀 사람들로 맞춘 % 변환으로 옮긴다
       (꼴 고르기·붓스트랩·잔차 분포도 겹 안에서). **곡선 모양(정답 그 자체)·시즌 간 차이 기본값·가중치·거르기는 전체 prior(0겹)** 그대로다.
       in·cf는 시즌 전체로 만든 core의 server·person·use 열을 쓴다 — 지문 묶음이 나중 글로 서버를 얻는 간접 누수가 있다
       (S41 T=24 한섭 라벨 사람 in·cf 11명, tr 1명). 18~48시간 in·cf 수치는 그만큼 낙관적이다
- tr : cf + T 절단 재처리 — T까지 게시된 글·사진만 남긴 state 사본에 post_all.sh(서버 분류·제외·이음·시각 등급)를 다시 돌린 뒤 예측.
       서버 라벨(지문 묶음의 앵커 글)·사람 이음·제외 판단에 T 뒤 글이 섞이는 누수를 없앤다 (스냅숏 댓글도 T까지만).
       남는 누수: 시즌 전체를 보고 만든 손 판정 post_manual_season.json(S41 8장). 지우고 다시 돌리면 S41 T=96 사본의 core 8줄
       (글 …, prev_season ↔ season_unsure_mixed)이 다르지만 모두 use 밖이라 예측은 같다 (2026-10-16). 곡선 모양(위)은 tr에서도 정답 그대로다
정답: truth_points.jsonl의 A+A2 곡선(기본)과 A_only, 덮인 구간 안의 값만 (explore_pct_transform.Curve, (ln %, 점수) 꺾은선).
      커뮤니티 집계표 B(truth_table.json)는 보조로 따로.
쓸 만함 (결과 보기 전에 정함, docs/solo-raid-analysis.md 0): 중앙값 상대 오차 |e| ≤ 2% (1·3%), ≤ 3% (10%). 보조 지점은 3%를 쓴다.
기준선 0 = 지금 % 그대로, 기준선 1 = 일차별 S41 중앙 배수 (predict의 b0·b1, 점 추정).
누수 점검(--leak-check): truth* 파일을 지우고 T 뒤 게시 줄을 core·obs에서 지운 사본에 predict.py를 따로 프로세스로 돌려
(open() 감시 켬) 같은 seed의 프로세스 안 결과와 JSON이 같은지 본다.
"""
import argparse
import collections
import datetime as dt
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import predict as PR  # noqa: E402
import fit_prior as FP  # noqa: E402

E = FP.E
TS = (3, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 119)
TOL = {1.0: 0.02, 3.0: 0.02, 10.0: 0.03}
FMT = "%Y-%m-%d %H:%M:%S"


import importlib.util as _ilu  # noqa: E402
_sp = _ilu.spec_from_file_location("split", os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.py"))
split = _ilu.module_from_spec(_sp)
_sp.loader.exec_module(split)

def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- 정답 ----------------

def truth_values(state):
    out = {}
    for name, gs in (("AA2", ("A", "A2")), ("A", ("A",))):
        for s in PR.SERVERS:
            c = E.Curve(E.final_points(state, s, gs), "lin")
            for q in PR.TARGETS:
                v = c.score_at(q) if c.n >= 2 else None
                if v is not None:
                    out[(name, s, q)] = v
    tp = os.path.join(state, "truth_table.json")
    if os.path.exists(tp):
        for r in json.load(open(tp, encoding="utf-8")).get("table", {}).get("rows", []):
            if r["server"] in PR.SERVERS and float(r["pct"]) in PR.TARGETS:
                out[("B", r["server"], float(r["pct"]))] = r["score_eok"]
    return out


# ---------------- 사람 겹 ----------------

def person_folds(state, k, seed):
    core = jl(os.path.join(state, "core.jsonl"))
    persons = sorted({r["person"] for r in core if r["src"] == "photo" and r["use"]})
    rng = random.Random(seed)
    rng.shuffle(persons)
    return {p: i % k for i, p in enumerate(persons)}


def cf_priors(state, folds, k, boot, seed, cache):
    if cache and os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    pri = []
    for f in range(k):
        keep = {p for p, g in folds.items() if g != f}
        print(f"  교차 적합 prior {f + 1}/{k} (학습 사람 {len(keep)})", flush=True)
        pri.append(PR_jsonable(FP.fit(state, boot=boot, seed=seed + f, persons=keep, quiet=True)))
    if cache:
        with open(cache, "w", encoding="utf-8") as fo:
            json.dump(pri, fo, ensure_ascii=False)
    return pri


def PR_jsonable(p):
    return json.loads(json.dumps(PR.jsonable(p)))


# ---------------- T 절단 재처리 ----------------

def trunc_state(state, T, work, extra):
    """T까지 게시된 것만 남긴 state 사본을 만들고 post_all.sh를 돌린다. 사본 경로."""
    season, c0, H = PR.season_info(state)
    tabs = c0 + dt.timedelta(hours=T)
    d = os.path.join(work, f"trunc_T{T:g}")
    st = os.path.join(d, "state")
    if os.path.exists(os.path.join(st, "core.jsonl")) and os.path.exists(os.path.join(d, "DONE")):
        return st
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(state, st)
    for name in os.listdir(st):   # 1단계가 다시 만드는 파일과 정답 파일은 지운다 (남은 값이 섞이지 않게)
        if name.startswith(("truth.", "truth_points", "core", "post_", "server")) and name != "post_manual_season.json":
            os.remove(os.path.join(st, name))
    def le(ts):
        t = parse_t(ts)
        return t is not None and t <= tabs
    obs = [o for o in jl(os.path.join(state, "obs.jsonl")) if le(o.get("post_t"))]
    wjl(os.path.join(st, "obs.jsonl"), obs)
    wjl(os.path.join(st, "claims.jsonl"), [c for c in jl(os.path.join(state, "claims.jsonl")) if le(c.get("post_t"))])
    wjl(os.path.join(st, "posts_min.jsonl"), [p for p in jl(os.path.join(state, "posts_min.jsonl")) if le(p.get("t"))])
    args = []
    if extra.get("posts"):
        pf = os.path.join(d, "posts_extra.jsonl")
        wjl(pf, [p for p in jl(extra["posts"]) if le(p.get("t"))])
        args += ["--posts", pf]
    if extra.get("snap"):
        sd = os.path.join(d, "snap")
        os.makedirs(sd)
        for fn in os.listdir(extra["snap"]):
            j = json.load(open(os.path.join(extra["snap"], fn), encoding="utf-8"))
            if not le(j.get("published_at")):
                continue
            if isinstance(j.get("comments"), list):
                keep = []
                for cm in j["comments"]:
                    t = cm.get("t")
                    try:
                        ct = dt.datetime.strptime(f"{tabs.year}.{t}", "%Y.%m.%d %H:%M:%S") if t else None
                    except ValueError:
                        ct = None
                    if ct is None or ct <= tabs:
                        keep.append(cm)
                j["comments"] = keep
            with open(os.path.join(sd, fn), "w", encoding="utf-8") as fo:
                json.dump(j, fo, ensure_ascii=False)
        args += ["--snap", sd]
    for k in ("parsed", "alt", "runs"):
        if extra.get(k):
            args += [f"--{k}", extra[k]]
    r = subprocess.run([os.path.join(HERE, "post_all.sh"), st] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"post_all.sh 실패 (T={T}): {r.stderr[-2000:]}")
    open(os.path.join(d, "DONE"), "w").write("ok\n")
    return st


def mix_for(st):
    """T 절단 사본에 server_mix.py --gate를 돌린다 (그 시점 라벨만으로 곡면·사전·기준). 한 번만."""
    done = os.path.join(st, "post_server_mix_summary.json")
    if os.path.exists(done) and os.path.getmtime(done) >= os.path.getmtime(os.path.join(st, "core.jsonl")):
        return
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "server_mix.py"), st, "--gate"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"server_mix.py 실패 ({st}): {r.stderr[-2000:]}")


def parse_t(ts):
    """글·사진 시각 (여러 꼴). 못 읽으면 None (그 글은 절단 사본에서 뺀다)."""
    if not ts:
        return None
    for f in (FMT, "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(str(ts)[:19], f)
        except ValueError:
            pass
    return None


def orig_key_map(state, st):
    """절단 사본 oi → 원본 person (겹 배정용). (img, row)로 잇는다."""
    o_obs = jl(os.path.join(state, "obs.jsonl"))
    o_core = jl(os.path.join(state, "core.jsonl"))
    key = {(o["img"], o["row"]): k for k, o in enumerate(o_obs)}
    pers = {r["oi"]: r["person"] for r in o_core if r["src"] == "photo"}
    t_obs = jl(os.path.join(st, "obs.jsonl"))
    return {k: pers.get(key.get((o["img"], o["row"]))) for k, o in enumerate(t_obs)}


# ---------------- 점검 ----------------

def evaluate(out, truth, T, variant):
    rows = []
    for s, sv in out["servers"].items():
        for q in PR.TARGETS:
            d = sv["points"][str(q)]
            for src in ("AA2", "A", "B"):
                tv = truth.get((src, s, q))
                if tv is None:
                    continue
                med = d["median"]
                e = med / tv - 1
                tol = TOL.get(q, 0.02 if q < 3 else 0.03)
                r = {"variant": variant, "T": T, "server": s, "q": q, "truth_src": src, "truth": tv, "pred": med,
                     "err": e, "pass": abs(e) <= tol, "tol": tol, "basis": sv["basis"],
                     "n_persons": sv["n_persons"], "n_moments": sv["n_moments"], "near_obs": d["near_obs"],
                     "n_persons_label": sv["n_persons_label"], "mixed_rows": sv["mixed_rows"],
                     "warn": (sv.get("warning") or {}).get("level"),
                     "unknown_used": (out.get("unknown") or {}).get("used"), "notes": d["notes"]}
                for L in ("i50", "i80", "i95"):
                    r["in" + L[1:]] = d[L][0] <= tv <= d[L][1]
                    r["w" + L[1:]] = d[L][1] / d[L][0] - 1
                for b in ("b0", "b1", "central"):
                    r["err_" + b] = None if d.get(b) is None else d[b] / tv - 1
                if d.get("comp80"):
                    r["in80_noiii"] = None
                rows.append(r)
    return rows


def run_variant(variant, state, prior, priors_cf, folds, Ts, draws, seed, opts, work, extra, truth):
    res, outs = [], {}
    for T in Ts:
        st = state
        rows = None
        if variant == "tr":
            st = trunc_state(state, T, work, extra)
            if (opts.get("unknown") or {}).get("mode") == "mix":
                mix_for(st)
        season, c0, H = PR.season_info(st)
        rws, cnt = PR.load_obs(st, T, H, c0, opts.get("grades", "all"), opts.get("c_delay", 0.0),
                               opts.get("server_mix"), opts.get("unknown"))
        pri = None
        if variant in ("cf", "tr"):
            if variant == "tr":
                pm = orig_key_map(state, st)
                for o in rws:
                    o["fold"] = folds.get(pm.get(o["oi"]), 0)
            else:
                for o in rws:
                    o["fold"] = folds.get(o["person"], 0)
            pri = priors_cf
        out = PR.predict(st, prior, T, opts, draws, seed, comps_split=True, priors=pri, rows=rws)
        out["counts"] = {k: v for k, v in cnt.items() if k != "unknown"}
        out["unknown"] = cnt.get("unknown")
        for srv, sv in out["servers"].items():   # predict(rows=…)는 미상 줄 처리 내역을 모르므로 경고를 다시 단다
            sv["warning"] = PR.warning(T, srv, sv, out["unknown"], prior)
        # (iii) 없는 구간도 한 번 (시즌 간 차이 가정을 뺀 포함률)
        o2 = PR.predict(st, prior, T, dict(opts, no_season_diff=True), max(300, draws // 2), seed, comps_split=False,
                        priors=pri, rows=[dict(o) for o in rws])
        ev = evaluate(out, truth, T, variant)
        ev2 = {(r["server"], r["q"], r["truth_src"]): r for r in evaluate(o2, truth, T, variant)}
        for r in ev:
            r2 = ev2.get((r["server"], r["q"], r["truth_src"]))
            if r2:
                r["in80_noiii"] = r2["in80"]
                r["in50_noiii"] = r2["in50"]
                r["w80_noiii"] = r2["w80"]
        res += ev
        outs[T] = out
        kr = out["servers"]["KR"]["points"]["3.0"]
        print(f"  [{variant}] T={T:>5g}: 한섭 사람 {out['servers']['KR']['n_persons']:>3} · 3% {kr['median']:.1f} "
              f"[80% {kr['i80'][0]:.1f}~{kr['i80'][1]:.1f}]", flush=True)
    return res, outs


def leak_check(state, prior_path, Ts, draws, seed, work, opts):
    """정답·T 뒤 자료를 지운 사본에서 따로 돌린 predict.py와 결과가 같은지."""
    msgs = []
    with open(prior_path, encoding="utf-8") as f:
        prior = json.load(f)
    season, c0, H = PR.season_info(state)
    for T in Ts:
        d = os.path.join(work, f"leak_T{T:g}")
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        tabs = c0 + dt.timedelta(hours=T)
        obs = jl(os.path.join(state, "obs.jsonl"))
        keep_oi = set()
        new_obs = []
        for k, o in enumerate(obs):
            if o.get("post_t") and dt.datetime.strptime(o["post_t"], FMT) <= tabs:
                new_obs.append(o)
                keep_oi.add(k)
            else:
                new_obs.append({"no": o.get("no"), "img": None, "post_t": None})
        core = [r for r in jl(os.path.join(state, "core.jsonl")) if r["src"] == "photo" and r["oi"] in keep_oi]
        wjl(os.path.join(d, "obs.jsonl"), new_obs)
        wjl(os.path.join(d, "core.jsonl"), core)
        shutil.copy(os.path.join(state, "season.json"), d)
        assert not any(n.startswith("truth") for n in os.listdir(d))
        o_dir = os.path.join(d, "out")
        cmd = [sys.executable, "-I", os.path.join(HERE, "predict.py"), d, "--prior", prior_path, "--at", str(T),
               "--draws", str(draws), "--seed", str(seed), "--out", o_dir]
        u = opts.get("unknown") or {}
        cmd += ["--unknown", u.get("mode", "none")] + ["--borrow-state", u.get("borrow_state") or ""]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            msgs.append(f"T={T}: 사본 실행 실패 {r.stderr[-500:]}")
            continue
        sub = json.load(open(os.path.join(o_dir, f"predict_T{float(T):g}.json"), encoding="utf-8"))
        inp = PR_jsonable(PR.predict(state, prior, float(T), opts, draws, seed))
        for x in (sub, inp):
            x.pop("counts", None)
        same = json.dumps(sub["servers"], sort_keys=True) == json.dumps(inp["servers"], sort_keys=True)
        msgs.append(f"T={T}: 정답 파일·T 뒤 줄을 지운 사본(따로 프로세스, open 감시) 결과와 프로세스 안 결과 {'같음' if same else '다름!'}")
    # 감시가 실제로 막는지
    r = subprocess.run([sys.executable, "-I", "-c",
                        "import sys; sys.path.insert(0, %r); import predict as P; P.install_guard();\n"
                        "try:\n open(%r)\n print('열림!')\nexcept PermissionError as e:\n print('막힘')" %
                        (HERE, os.path.join(state, "truth.json"))], capture_output=True, text=True)
    msgs.append(f"감시 켠 프로세스에서 truth.json 열기: {r.stdout.strip() or r.stderr.strip()[-200:]}")
    return msgs


# ---------------- 표 ----------------

def fmt_e(e):
    return "—" if e is None else f"{e * 100:+.1f}%"


def summary_text(res, variants, Ts):
    L = []
    L.append("※ 표본 안 점검: prior와 정답이 같은 시즌(S41)이다. 최종 곡선 모양은 정답 곡선 그 자체라 1·10%의 모양 오차는 들어가지 않는다.")
    L.append("   쓸 만함: |오차| ≤ 2% (1·3%), ≤ 3% (10%). 구간: 50/80% 포함 (○ 포함 · × 밖). (iii)없음 = 시즌 간 차이 가정을 뺀 80% 구간.")
    by = {(r["variant"], r["T"], r["server"], r["q"], r["truth_src"]): r for r in res}
    for v in variants:
        L.append("")
        L.append(f"=== 변형 {v} ===")
        L.append("T(h) | 한섭 사람(유효, 라벨)·시점 경고 | 한섭 3% 오차 [50 80 | 80폭 | (iii)없음80] 통과 | 1% 오차 [80] | 10% 오차 [80] | 기준선0 3% · 기준선1 3% | "
                 "일섭 3%(A+A2) [80] · B | 글섭 3% B [80]")
        for T in Ts:
            k3 = by.get((v, T, "KR", 3.0, "AA2"))
            k1 = by.get((v, T, "KR", 1.0, "AA2"))
            k10 = by.get((v, T, "KR", 10.0, "AA2"))
            j3 = by.get((v, T, "JP", 3.0, "AA2"))
            j3b = by.get((v, T, "JP", 3.0, "B"))
            g3 = by.get((v, T, "GL", 3.0, "B"))
            if not k3:
                continue
            o = lambda b: "○" if b else ("×" if b is not None else "—")
            L.append(f"{T:>4g} | {k3['n_persons']:>5}(라벨 {k3.get('n_persons_label', '?')})·{k3['n_moments']:<3} {k3.get('warn') or ''} | {fmt_e(k3['err'])} [{o(k3['in50'])} {o(k3['in80'])} | "
                     f"{k3['w80'] * 100:4.0f}% | {o(k3.get('in80_noiii'))}] {'통과' if k3['pass'] else '—'} | "
                     f"{fmt_e(k1['err']) if k1 else '—'} [{o(k1['in80']) if k1 else '—'}] | "
                     f"{fmt_e(k10['err']) if k10 else '—'} [{o(k10['in80']) if k10 else '—'}] | "
                     f"{fmt_e(k3['err_b0'])} · {fmt_e(k3['err_b1'])} | "
                     f"{fmt_e(j3['err']) if j3 else '—'} [{o(j3['in80']) if j3 else '—'}] {j3['basis'][:5] if j3 else ''} · {fmt_e(j3b['err']) if j3b else '—'} | "
                     f"{fmt_e(g3['err']) if g3 else '—'} [{o(g3['in80']) if g3 else '—'}] {g3['basis'][:5] if g3 else ''}")
        # 요약: 주 지점별 포함률·통과율
        for (s, q, src) in (("KR", 1.0, "AA2"), ("KR", 3.0, "AA2"), ("KR", 10.0, "AA2"), ("KR", 3.0, "A"), ("KR", 3.0, "B"),
                            ("KR", 10.0, "B"), ("KR", 30.0, "AA2"), ("KR", 0.5, "AA2"), ("KR", 2.0, "AA2"),
                            ("JP", 3.0, "AA2"), ("JP", 2.0, "AA2"), ("JP", 3.0, "B"), ("JP", 10.0, "B"), ("GL", 3.0, "B"), ("GL", 10.0, "B")):
            rr = [by[(v, T, s, q, src)] for T in Ts if (v, T, s, q, src) in by]
            if not rr:
                continue
            for lab, sel in (("T≤24", [r for r in rr if r["T"] <= 24]), ("T 36~72", [r for r in rr if 36 <= r["T"] <= 72]),
                             ("T≥84", [r for r in rr if r["T"] >= 84])):
                if not sel:
                    continue
                n = len(sel)
                L.append(f"  {s} {q:g}% ({src}) {lab}: n={n} |오차| 중앙 {sorted(abs(r['err']) for r in sel)[n // 2] * 100:.1f}% · "
                         f"통과 {sum(r['pass'] for r in sel)}/{n} · 50% 포함 {sum(r['in50'] for r in sel)}/{n} · 80% 포함 {sum(r['in80'] for r in sel)}/{n} · "
                         f"95% 포함 {sum(r['in95'] for r in sel)}/{n} · 80% 폭 중앙 {sorted(r['w80'] for r in sel)[n // 2] * 100:.0f}% · "
                         f"(iii)없음 80% 포함 {sum(1 for r in sel if r.get('in80_noiii'))}/{n} · "
                         f"기준선0 |오차| 중앙 {sorted(abs(r['err_b0']) for r in sel if r['err_b0'] is not None)[n // 2] * 100:.1f}% · "
                         f"기준선1 {sorted(abs(r['err_b1']) for r in sel if r['err_b1'] is not None)[n // 2] * 100:.1f}%")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--prior", required=True)
    ap.add_argument("--T", default=",".join(str(t) for t in TS))
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--boot", type=int, default=100, help="교차 적합 prior의 붓스트랩 수")
    ap.add_argument("--variants", default="in,cf,tr")
    ap.add_argument("--work", help="절단 사본·교차 적합 prior 캐시 폴더 (기본 임시 폴더)")
    ap.add_argument("--out")
    ap.add_argument("--leak-check", action="store_true")
    for k in ("posts", "snap", "parsed", "alt", "runs"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--predict-opt", nargs="*", default=[])
    ap.add_argument("--unknown", choices=PR.UNKNOWN_MODES, default="none",
                    help="서버 미상 줄 처리 (predict.py --unknown과 같음). 이 점검의 기본은 none(예전 결과와 같게). "
                         "tr에서 mix면 절단 사본마다 server_mix.py --gate를 돌린다")
    ap.add_argument("--unknown-servers", default="KR")
    ap.add_argument("--borrow-state")
    ap.add_argument("--borrow-kind", choices=["label", "em"], default="label")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--fallback", choices=["site", "none"], default="site")
    ap.add_argument("--robust", choices=["mean", "wmedian"], default="mean")
    ap.add_argument("--top-cut", type=float)
    split.add_final_arg(ap)
    a = ap.parse_args()
    split.guard_args(a)   # 시험 시즌 정답 감시 (할 일 3.0b)
    Ts = [float(t) for t in a.T.split(",")]
    variants = a.variants.split(",")
    work = a.work or tempfile.mkdtemp(prefix="backtest_")
    os.makedirs(work, exist_ok=True)
    with open(a.prior, encoding="utf-8") as f:
        prior = json.load(f)
    opts = {}
    for kv in a.predict_opt:
        k, v = kv.split("=", 1)
        opts[k] = (v in ("1", "true")) if k in ("no_season_diff", "p_update") else (v if k in ("grades",) else float(v))
    if a.unknown != "none":
        opts["unknown"] = {"mode": a.unknown, "servers": a.unknown_servers.split(","), "borrow_state": a.borrow_state,
                           "borrow_kind": a.borrow_kind, "no_gate": a.no_gate, "fallback": a.fallback}
    if a.robust != "mean":
        opts["robust"] = a.robust
    if a.top_cut:
        opts["top_cut"] = a.top_cut
    truth = truth_values(a.state)
    print(f"정답 값 {len(truth)}개 (A+A2·A_only·B). T = {Ts}. 변형 {variants}. predict 옵션 {opts}. 작업 폴더 {work}")
    folds = person_folds(a.state, a.kfold, a.seed)
    priors_cf = None
    if any(v in ("cf", "tr") for v in variants):
        priors_cf = cf_priors(a.state, folds, a.kfold, a.boot, a.seed, os.path.join(work, f"cf_priors_k{a.kfold}_b{a.boot}.json"))
        print("  교차 적합 꼴: " + ", ".join(f"{p['transform']['form']}{'+x' if p['transform']['bx'] else ''}" for p in priors_cf))
    extra = {k: getattr(a, k) for k in ("posts", "snap", "parsed", "alt", "runs")}
    if extra["runs"] is None:
        extra["runs"] = os.path.join(a.state, "..", "runs")
    allres, allout = [], {}
    for v in variants:
        r, o = run_variant(v, a.state, prior, priors_cf, folds, Ts, a.draws, a.seed, opts, work, extra, truth)
        allres += r
        allout[v] = o
    txt = summary_text(allres, variants, Ts)
    lc = leak_check(a.state, a.prior, [24.0, 72.0], 300, a.seed, work, opts) if a.leak_check else []
    if lc:
        txt += "\n\n누수 점검\n" + "\n".join("  " + m for m in lc)
    print("\n" + txt)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "backtest.json"), "w", encoding="utf-8") as f:
            json.dump(PR.jsonable({"rows": allres, "leak_check": lc, "opts": opts, "Ts": Ts, "variants": variants}), f,
                      ensure_ascii=False, indent=1)
        with open(os.path.join(a.out, "backtest.txt"), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        with open(os.path.join(a.out, "backtest_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(PR.jsonable({v: {str(T): o for T, o in oo.items()} for v, oo in allout.items()}), f, ensure_ascii=False)
        print(f"→ {a.out}/backtest.txt · backtest.json")


if __name__ == "__main__":
    main()
