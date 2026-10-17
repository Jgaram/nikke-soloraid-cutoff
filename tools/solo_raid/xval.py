"""솔로레이드 여러 시즌 검증 — 다른 시즌들로 만든 prior로 한 시즌을 T마다 맞혀 본다 (할 일 3.6, 2026-10-16, 모델 0, 요청 0).

python3 -I tools/solo_raid/xval.py [--seasons S38,S39,S40,S41] [--scheme forward,loso] [--T 12,18,...] [--draws 600]
        [--work DIR] [--out F.json] [--tag dev] [--prior-opt key=value ...] [--predict-opt key=value ...]

- 겹(fold). forward(기본): 시즌 번호 순으로, 앞 시즌들(학습 가능한 개발 시즌 2개 이상) → 다음 시즌. loso(비교): 그 시즌만 빼고 나머지 전부.
  학습 시즌 = 등록부(registry.json) 몫 dev이고 학습 등급이 none이 아닌 시즌. 맞힐 시즌 = 정답 표(answer_key.json)에 채점 정답이 있는 개발 시즌.
  prior는 fit_prior_multi.fit_multi(그 겹의 학습 시즌만). 맞힐 시즌의 정답·자료는 prior에 들어가지 않는다.
- tr(T 절단 재처리)만: 맞힐 시즌 state에서 T까지 게시된 글·사진만 남긴 사본에 post_all.sh를 다시 돌리고 (backtest.trunc_state,
  시즌 전용 입력은 post_all_stamp.json의 extra) predict.predict로 낸다. 서버 미상 줄은 site 방식, 빌릴 비율은 그 겹의 학습 시즌들.
- 채점: answer_key.json의 truth(criteria.json 규칙 — 한섭 3·10% 집계표, 그 밖 곡선 strong)와 집계표·곡선 각각.
  e = 중앙값/정답 − 1, 80·50·95% 포함, 80% 폭, 기준선 0·1, 등급(predict warning level).
- 요약은 criteria.json의 성공·고르기 규칙대로. **시험 시즌은 넣지 않는다** (split 감시 — 3.7 시험은 따로).
출력(기본 data/solo-raid/analysis/xval_{tag}.json): 겹·T·서버·지점별 줄, 요약, 실행 정보(시즌 목록·state 도장·split 판·도구 HEAD·criteria 시각).
"""
import argparse
import collections
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import predict as PR  # noqa: E402
import backtest as BT  # noqa: E402
import fit_prior_multi as FM  # noqa: E402
import split as SP  # noqa: E402

DATA = os.path.join(ROOT, "data", "solo-raid")
AN = os.path.join(DATA, "analysis")
MIN_TRAIN = 2


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def folds(seasons, reg, scheme):
    train_ok = [s for s in seasons if reg[s]["role"] == "dev" and reg[s]["tier"] != "none"]
    order = sorted(seasons, key=SP.snum)
    out = []
    for tgt in order:
        if scheme == "forward":
            tr = [s for s in train_ok if SP.snum(s) < SP.snum(tgt)]
        else:
            tr = [s for s in train_ok if s != tgt]
        tr = [s for s in tr if reg[s]["tier"] in ("full", "shape_only")]
        if sum(1 for s in tr if reg[s]["tier"] == "full") < 1 or len(tr) < MIN_TRAIN:
            continue
        out.append((tuple(sorted(tr, key=SP.snum)), tgt))
    return out


def prior_for(train, work, boot, seed, popts):
    key = hashlib.sha1(json.dumps([train, boot, seed, popts, [FM.registry()[s].get("data", {}).get("stamp_digest") for s in train]],
                                  sort_keys=True).encode()).hexdigest()[:12]
    p = os.path.join(work, "priors", f"prior_{'+'.join(train)}_{key}.json")
    if os.path.exists(p):
        return jload(p), p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    pr = FM.fit_multi(list(train), boot=boot, seed=seed, quiet=True, **popts)
    pr = json.loads(json.dumps(PR.jsonable(pr)))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(pr, f, ensure_ascii=False)
    return pr, p


def extra_of(state):
    st = jload(os.path.join(state, "post_all_stamp.json"))
    return {k: os.path.join(ROOT, v["path"]) for k, v in st.get("extra", {}).items()}


def score(out, akey, tgt, T, fold_name, train, crit):
    rows = []
    tol = crit["tol"]
    for r in akey:
        if r["season"] != tgt:
            continue
        sv = out["servers"].get(r["server"])
        if not sv:
            continue
        d = sv["points"].get(str(float(r["pct"])))
        if not d or d.get("median") is None:
            continue
        q = float(r["pct"])
        tl = tol.get(f"{q:g}", tol["aux"])
        med = d["median"]
        rec = {"scheme": fold_name, "train": list(train), "season": tgt, "T": T, "server": r["server"], "pct": q, "tol": tl,
               "pred": round(med, 3), "i80": d["i80"], "i95": d["i95"], "grade": (sv.get("warning") or {}).get("level"),
               "basis_pred": sv.get("basis"), "n_persons": sv.get("n_persons"), "n_persons_label": sv.get("n_persons_label"),
               "truth": r["truth"], "truth_basis": r["basis"], "shaky": r["shaky"],
               "table": r["table"] and r["table"]["value"], "curve": r["curve"]["value"], "curve_grade": r["curve"]["grade"]}
        if r["truth"] is not None:
            tv = r["truth"]
            e = med / tv - 1
            rec.update(err=round(e, 5), ok=abs(e) <= tl,
                       in50=d["i50"][0] <= tv <= d["i50"][1], in80=d["i80"][0] <= tv <= d["i80"][1], in95=d["i95"][0] <= tv <= d["i95"][1],
                       w80=round(d["i80"][1] / d["i80"][0] - 1, 4),
                       err_b0=None if d.get("b0") is None else round(d["b0"] / tv - 1, 5),
                       err_b1=None if d.get("b1") is None else round(d["b1"] / tv - 1, 5))
        for src in ("table", "curve"):
            v = rec[src] if src == "table" else (r["curve"]["value"] if r["curve"]["grade"] in ("strong", "fragile") else None)
            rec["err_" + src] = None if v is None else round(med / v - 1, 5)
        rows.append(rec)
    return rows


def summarize(rows, crit):
    """scheme × T × 지점(한섭 주) 요약과 criteria 판정."""
    S = {}
    srv = crit["server_main"]
    for scheme in sorted({r["scheme"] for r in rows}):
        R = [r for r in rows if r["scheme"] == scheme and r["server"] == srv and r.get("err") is not None]
        by = {}
        for T in crit["T"]:
            for q in crit["points_main"]:
                L = [r for r in R if r["T"] == T and r["pct"] == q]
                if not L:
                    continue
                ae = [abs(r["err"]) for r in L]
                b1 = [abs(r["err_b1"]) for r in L if r["err_b1"] is not None]
                b0 = [abs(r["err_b0"]) for r in L if r["err_b0"] is not None]
                by[f"T{T:g}/{q:g}"] = {
                    "n_seasons": len(L), "seasons": {r["season"]: {"err": r["err"], "in80": r["in80"], "w80": r["w80"], "grade": r["grade"]} for r in L},
                    "med_abs_err": round(statistics.median(ae), 5), "usable_ratio": round(sum(r["ok"] for r in L) / len(L), 3),
                    "cov80": round(sum(r["in80"] for r in L) / len(L), 3), "cov95": round(sum(r["in95"] for r in L) / len(L), 3),
                    "med_w80": round(statistics.median(r["w80"] for r in L), 4),
                    "med_abs_err_b1": round(statistics.median(b1), 5) if b1 else None,
                    "med_abs_err_b0": round(statistics.median(b0), 5) if b0 else None}
        # 성공 판정 (T_main, 한섭 1·3%)
        Tm = crit["T_main"]
        M = [r for r in R if r["T"] in Tm and r["pct"] in (1.0, 3.0)]
        cov = sum(r["in80"] for r in M) / len(M) if M else None
        per_T = []
        per_T_b1 = []
        for T in Tm:
            for q in (1.0, 3.0):
                L = [r for r in R if r["T"] == T and r["pct"] == q]
                if L:
                    per_T.append(statistics.median(abs(r["err"]) for r in L))
                    b = [abs(r["err_b1"]) for r in L if r["err_b1"] is not None]
                    if b:
                        per_T_b1.append(statistics.median(b))
        AB = [r for r in M if r["grade"] in ("A", "B")]
        usable = {}
        for T in Tm:
            for q in crit["points_main"]:
                k = f"T{T:g}/{q:g}"
                if k in by:
                    usable[k] = by[k]["usable_ratio"] >= 2 / 3
        S[scheme] = {
            "by": by,
            "judge": {
                "n_rows_main": len(M), "seasons": sorted({r["season"] for r in M}),
                "cov80_main": None if cov is None else round(cov, 3),
                "coverage": None if cov is None else ("보정됨" if 0.70 <= cov <= 0.90 else "구간이 좁다(과신)" if cov < 0.70 else "구간이 넓다"),
                "mean_med_abs_err_main": round(statistics.mean(per_T), 5) if per_T else None,
                "mean_med_abs_err_b1": round(statistics.mean(per_T_b1), 5) if per_T_b1 else None,
                "beat_baseline1": (statistics.mean(per_T) < statistics.mean(per_T_b1)) if per_T and per_T_b1 else None,
                "cov80_grade_AB": round(sum(r["in80"] for r in AB) / len(AB), 3) if AB else None, "n_grade_AB": len(AB),
                "usable": usable,
            }}
    return S


def run_meta(seasons, reg, popts, pdopts, args):
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "tools/solo_raid"], capture_output=True, text=True).stdout.strip()
    sp = SP.load_split()
    crit = jload(os.path.join(AN, "criteria.json"))
    # 실행 시각은 넣지 않는다 (같은 입력 → 같은 파일). 시각은 git 커밋으로 본다
    return {"tool_head": head, "tools_dirty": bool(dirty), "split_version": sp["version"], "criteria_decided": crit["decided"],
            "seasons": {s: {"role": reg[s]["role"], "tier": reg[s]["tier"],
                            "post_all_digest": (reg[s].get("data") or {}).get("stamp_digest")} for s in seasons},
            "answer_key_sha1": hashlib.sha1(open(os.path.join(AN, "answer_key.json"), "rb").read()).hexdigest(),
            "prior_opts": popts, "predict_opts": pdopts,
            "args": {k: v for k, v in args.items() if k not in ("work", "out")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons")
    ap.add_argument("--scheme", default="forward,loso")
    ap.add_argument("--T")
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--boot", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--work", default=None, help="T 절단 사본·prior 캐시 (기본: 세션 임시 폴더 또는 /tmp)")
    ap.add_argument("--out")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--prior-opt", action="append", default=[], help="fit_multi 옵션 (tiers 등)")
    ap.add_argument("--fixed-prior", help="겹마다 만들지 않고 이 prior 하나로 (예: S41 한 시즌 prior — 표본 밖 첫 점검을 tr로 다시 잴 때). scheme 이름 fixed")
    ap.add_argument("--fixed-train", default="S41", help="--fixed-prior를 만든 시즌 (맞힐 시즌에서 뺀다, 쉼표)")
    ap.add_argument("--predict-opt", action="append", default=[], help="predict 옵션 (tau_p=0.1, tau_shape=0, no_season_diff=true …)")
    SP.add_final_arg(ap)
    a = ap.parse_args()
    if a.final:
        raise SystemExit("xval은 개발 시즌 검증이다 — 시험(3.7)은 따로 한다")
    SP.guard_args(a)
    # 커밋된 xval_dev.json은 analyze_all의 forward·loso 결과이고 웹 빌드(viz.py)가 loso 요약을 검사한다 (2026-10-17 레포 정리).
    # --tag 없이 다른 방식으로 돌려 덮어쓰면 push 뒤 공개 웹 빌드가 멈춘다.
    if os.path.abspath(a.out or os.path.join(AN, f"xval_{a.tag}.json")) == os.path.abspath(os.path.join(AN, "xval_dev.json")) \
            and (a.fixed_prior or "loso" not in a.scheme.split(",")):
        raise SystemExit("xval_dev.json 자리는 analyze_all의 forward·loso 결과만 쓴다 — --tag나 --out을 따로 준다")
    reg = FM.registry()
    crit = jload(os.path.join(AN, "criteria.json"))
    akey = jload(os.path.join(AN, "answer_key.json"))["rows"]
    seasons = a.seasons.split(",") if a.seasons else sorted({r["season"] for r in akey if r["truth"] is not None}, key=SP.snum)
    seasons = [s for s in seasons if reg[s]["role"] == "dev" and reg[s].get("collect") == "완료"]   # 수집 끝난 시즌만 (2026-10-17, truth_grade와 같게)
    Ts = [float(x) for x in a.T.split(",")] if a.T else [float(t) for t in crit["T"]]
    work = a.work or os.path.join(os.environ.get("CLAUDE_SCRATCH", "/tmp"), "solo_raid_xval")
    os.makedirs(work, exist_ok=True)
    popts = parse_kv(a.prior_opt)
    if "tiers" in popts and isinstance(popts["tiers"], str):
        popts["tiers"] = tuple(popts["tiers"].split(","))
    pdopts = parse_kv(a.predict_opt)
    rows, fold_info = [], []
    plan = []
    if a.fixed_prior:
        ft = tuple(a.fixed_train.split(","))
        plan = [("fixed", ft, t) for t in seasons if t not in ft]
    else:
        for scheme in a.scheme.split(","):
            plan += [(scheme, tr, t) for tr, t in folds(seasons, reg, scheme)]
    for scheme, train, tgt in plan:
        if a.fixed_prior:
            prior = jload(a.fixed_prior)
        else:
            prior, ppath = prior_for(train, work, a.boot, a.seed, popts)
        sd = prior["season_diff"]
        fold_info.append({"scheme": scheme, "train": list(train), "target": tgt,
                          "form": prior["transform"]["form"], "season_levels": prior["transform"].get("season_levels"),
                          "tau": {k: sd[k]["value"] for k in ("tau_p", "tau_shape", "tau_ratio", "tau_scale0")}})
        state = FM.state_of(tgt)
        ex = extra_of(state)
        tw = os.path.join(work, "trunc", tgt)
        os.makedirs(tw, exist_ok=True)
        borrow = [FM.state_of(s) for s in train]
        for T in Ts:
            st = BT.trunc_state(state, T, tw, ex)
            season, c0, H = PR.season_info(st)
            Tc = min(T, H)
            opts = {"grades": "all", "c_delay": 0.0, "p_update": False, "robust": "mean", "eps_wrong": PR.EPS_WRONG,
                    "notes": [], "contam": (),
                    "unknown": {"mode": "site", "servers": ["KR"], "borrow_state": borrow, "borrow_kind": "label",
                                "mix_file": None, "no_gate": False, "fallback": "site"}}
            opts.update(pdopts)
            if opts.get("early_level") is True:   # 3.5e: 학습 시즌의 같은 T 절단 사본 (정답 없음, 수준 보정용)
                opts["early_level"] = {"train": [[s, BT.trunc_state(FM.state_of(s), T, os.path.join(work, "trunc", s), extra_of(FM.state_of(s)))]
                                                 for s in train]}
            rws, cnt = PR.load_obs(st, Tc, H, c0, opts["grades"], opts["c_delay"], None, opts["unknown"])
            out = PR.predict(st, prior, Tc, opts, a.draws, a.seed, comps_split=False, rows=rws)
            out["unknown"] = cnt.get("unknown")
            for srv, sv in out["servers"].items():
                sv["warning"] = PR.warning(Tc, srv, sv, out["unknown"], prior)
            rs = score(out, akey, tgt, T, scheme, train, crit)
            rows += rs
            k3 = next((r for r in rs if r["server"] == "KR" and r["pct"] == 3.0), None)
            if k3:
                print(f"[{scheme}] {'+'.join(train)} → {tgt} T={T:>4g}: 한섭 3% {k3['pred']:.1f} "
                      f"[{k3['i80'][0]:.1f}~{k3['i80'][1]:.1f}] e={k3.get('err', float('nan')):+.2%} 등급 {k3['grade']}", flush=True)
    res = {"about": "솔로레이드 여러 시즌 검증 결과 (tools/solo_raid/xval.py, 할 일 3.6). 개발 시즌만 — 시험 시즌 결과가 아니다",
           "run": run_meta(seasons, reg, popts, pdopts, vars(a)), "folds": fold_info,
           "summary": summarize(rows, crit), "rows": rows}
    out = a.out or os.path.join(AN, f"xval_{a.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(PR.jsonable(res), f, ensure_ascii=False, indent=1)
        f.write("\n")
    for scheme, sm in res["summary"].items():
        j = sm["judge"]
        print(f"\n== {scheme}: 시즌 {j['seasons']} · 80% 포함(T_main 1·3%) {j['cov80_main']} ({j['coverage']}) · "
              f"평균 시즌 중앙 |e| {j['mean_med_abs_err_main']} vs 기준선1 {j['mean_med_abs_err_b1']} · A·B 포함 {j['cov80_grade_AB']} (n {j['n_grade_AB']})")
        for k, v in sm["by"].items():
            print(f"   {k:9s} n{v['n_seasons']} |e|중앙 {v['med_abs_err']:.3%} 쓸만 {v['usable_ratio']:.2f} 포함80 {v['cov80']:.2f} "
                  f"폭80 {v['med_w80']:.1%} b1 {v['med_abs_err_b1'] if v['med_abs_err_b1'] is None else format(v['med_abs_err_b1'], '.3%')} · "
                  + " ".join(f"{s}:{d['err']:+.1%}{'' if d['in80'] else '×'}" for s, d in v["seasons"].items()))
    print("→", os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
