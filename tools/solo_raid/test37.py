"""솔로레이드 3.7 시험 — 분석 ④가 고정한 선택(freeze_37.json)을 잠긴 시험 시즌에 한 번 돌린다 (할 일 3.7, 2026-10-17, 모델 0, 요청 0).

python3 -I tools/solo_raid/test37.py --final                 # 시험 세션만. 시험 시즌 정답을 연다 (split 감시 풀림)
python3 -I tools/solo_raid/test37.py --dry S38,S40 [--out F]  # 도구 점검용 — 개발 시즌으로 같은 길을 돌린다 (정답 잠금 그대로)

고정 파일 data/solo-raid/analysis/freeze_37.json (분석 ④가 쓰고 커밋, 시험 세션은 고치지 않는다)
- configs: 예측 설정 묶음. 묶음마다 prior 파일(sha1 확인), 빌릴 사이트 비율 시즌(borrow), predict 옵션, 맡는 % 지점(points).
  한 지점은 한 묶음만 맡는다. 등급(warning)은 main 묶음의 것을 쓴다.
- T·지표·성공 판정은 criteria.json 그대로 (xval.score·summarize — 3.6과 같은 계산).
- 시험 시즌 = split.json test. 정답 표는 truth_grade.build(시험 시즌, final=True) — criteria.json의 정답 규칙.
순서
1. 고정 파일·prior sha1·criteria sha1을 확인한다. 다르면 멈춘다 (고치지 않는다 — 분석 문서 3.0·3.7).
2. 시즌마다 후처리 도장이 fresh인지 본다 (post_stamp --check). stale이면 멈춘다 (고정 커밋에서 후처리를 다시 하면 그 결과는 「미검증」).
3. 정답 표를 만든다 (--final일 때만 시험 시즌).
4. 시즌 × T마다 T 절단 재처리(backtest.trunc_state, tr) → 묶음마다 predict → 지점별로 맡은 묶음의 값을 모아 채점.
5. data/solo-raid/analysis/test37.json(--final) 또는 --out(--dry)에 줄·요약·실행 정보를 쓴다. 실행 시각은 넣지 않는다.
결과를 보고 도구·설정을 고치면 그 시즌은 개발로 넘어가고 고친 뒤 성능은 「미검증」이다 (가이드 §4).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import split as SP  # noqa: E402

DATA = os.path.join(ROOT, "data", "solo-raid")
AN = os.path.join(DATA, "analysis")
FREEZE = os.path.join(AN, "freeze_37.json")


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha1(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def stop(msg):
    raise SystemExit(f"멈춤 (고치지 않는다 — 사용자에게 알린다): {msg}")


def main():
    ap = argparse.ArgumentParser()
    SP.add_final_arg(ap)
    ap.add_argument("--dry", help="도구 점검용 개발 시즌 (쉼표). --final과 함께 쓰지 않는다")
    ap.add_argument("--freeze", default=FREEZE)
    ap.add_argument("--work", default=None)
    ap.add_argument("--out")
    a = ap.parse_args()
    if bool(a.final) == bool(a.dry):
        raise SystemExit("--final(시험 세션) 또는 --dry 개발 시즌 하나만")
    SP.guard_args(a)
    import predict as PR  # noqa: E402  (감시를 켠 뒤 불러온다)
    import backtest as BT  # noqa: E402
    import xval as XV  # noqa: E402
    import post_stamp  # noqa: E402
    import truth_grade as TG  # noqa: E402
    import fit_prior_multi as FM  # noqa: E402

    fz = jload(a.freeze)
    crit_p = os.path.join(ROOT, fz["criteria"]["path"])
    if sha1(crit_p) != fz["criteria"]["sha1"]:
        stop(f"criteria.json sha1이 고정 때와 다르다 ({fz['criteria']['sha1']})")
    crit = jload(crit_p)
    sp = SP.load_split()
    if sp["version"] != fz["split_version"]:
        stop(f"split.json 판이 고정 때({fz['split_version']})와 다르다")
    priors = {}
    owner = {}
    for c in fz["configs"]:
        p = os.path.join(ROOT, c["prior"])
        if sha1(p) != c["prior_sha1"]:
            stop(f"{c['name']} prior {c['prior']} sha1이 고정 때와 다르다")
        priors[c["name"]] = jload(p)
        for q in c["points"]:
            if float(q) in owner:
                stop(f"지점 {q}를 두 묶음이 맡는다")
            owner[float(q)] = c["name"]
    main_name = fz["main_config"]
    seasons = sp["test"] if a.final else a.dry.split(",")
    seasons = [s if str(s).startswith("S") else f"S{s}" for s in seasons]
    if a.dry and any(SP.is_test(s, sp) for s in seasons):
        raise SystemExit("--dry에는 개발 시즌만")
    reg = FM.registry()
    # 2. 후처리 최신
    for s in seasons:
        st = FM.state_of(s)
        if not os.path.exists(os.path.join(st, "obs.jsonl")):
            stop(f"{s} state 없음 — 「쓸 수 없음」으로 적는다")
        stt, why = post_stamp.check(st)
        if stt != "fresh":
            stop(f"{s} 후처리 도장 {stt} ({why})")
    # 3. 정답 표
    akey = TG.build(seasons, final=bool(a.final))["rows"]
    Ts = [float(t) for t in crit["T"]]
    work = a.work or os.path.join(os.environ.get("CLAUDE_SCRATCH", "/tmp"), "solo_raid_test37")
    os.makedirs(work, exist_ok=True)
    rows = []
    for tgt in seasons:
        state = FM.state_of(tgt)
        ex = XV.extra_of(state)
        tw = os.path.join(work, "trunc", tgt)
        os.makedirs(tw, exist_ok=True)
        for T in Ts:
            st = BT.trunc_state(state, T, tw, ex)
            season, c0, H = PR.season_info(st)
            Tc = min(T, H)
            outs = {}
            for c in fz["configs"]:
                borrow = [FM.state_of(b) for b in c["borrow"]]
                opts = {"grades": "all", "c_delay": 0.0, "p_update": False, "robust": "mean", "eps_wrong": PR.EPS_WRONG,
                        "notes": [], "contam": (),
                        "unknown": {"mode": "site", "servers": ["KR"], "borrow_state": borrow, "borrow_kind": "label",
                                    "mix_file": None, "no_gate": False, "fallback": "site"}}
                opts.update(c.get("predict_opts") or {})
                rws, cnt = PR.load_obs(st, Tc, H, c0, opts["grades"], opts["c_delay"], None, opts["unknown"])
                out = PR.predict(st, priors[c["name"]], Tc, opts, fz["draws"], fz["seed"], comps_split=False, rows=rws)
                out["unknown"] = cnt.get("unknown")
                for srv, sv in out["servers"].items():
                    sv["warning"] = PR.warning(Tc, srv, sv, out["unknown"], priors[c["name"]])
                outs[c["name"]] = out
            comb = json.loads(json.dumps(PR.jsonable(outs[main_name])))
            for srv, sv in comb["servers"].items():
                for qk in list(sv["points"]):
                    o = owner.get(float(qk))
                    if o and o != main_name:
                        osv = outs[o]["servers"].get(srv)
                        sv["points"][qk] = json.loads(json.dumps(PR.jsonable(osv["points"].get(qk)))) if osv else None
                sv["points"] = {k: v for k, v in sv["points"].items() if v is not None}
            rs = XV.score(comb, akey, tgt, T, "test" if a.final else "dry", [main_name], crit)
            for r in rs:
                r["config"] = owner.get(r["pct"], main_name)
            rows += rs
            k3 = next((r for r in rs if r["server"] == "KR" and r["pct"] == 3.0), None)
            print(f"{tgt} T={T:>4g}: 한섭 3% 채점 줄 {'있음' if k3 and k3.get('err') is not None else '없음'}", flush=True)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = {"about": "솔로레이드 3.7 시험 결과 (tools/solo_raid/test37.py). 고정 선택 freeze_37.json을 시험 시즌에 한 번 돌린 것"
                    if a.final else "test37.py 도구 점검 (--dry, 개발 시즌 — 시험 결과가 아니다)",
           "run": {"tool_head": head, "freeze_sha1": sha1(a.freeze), "split_version": sp["version"], "seasons": seasons,
                   "post_all_digest": {s: (reg.get(s, {}).get("data") or {}).get("stamp_digest") for s in seasons}},
           "summary": XV.summarize(rows, crit), "rows": rows}
    out = a.out or (os.path.join(AN, "test37.json") if a.final else os.path.join(work, "test37_dry.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(PR.jsonable(res), f, ensure_ascii=False, indent=1)
        f.write("\n")
    for scheme, sm in res["summary"].items():
        j = sm["judge"]
        print(f"\n== {scheme}: 시즌 {j['seasons']} · 80% 포함(T_main 1·3%) {j['cov80_main']} ({j['coverage']}) · "
              f"평균 시즌 중앙 |e| {j['mean_med_abs_err_main']} vs 기준선1 {j['mean_med_abs_err_b1']} · A·B 포함 {j['cov80_grade_AB']} (n {j['n_grade_AB']})")
        for k, v in sm["by"].items():
            print(f"   {k:9s} n{v['n_seasons']} |e|중앙 {v['med_abs_err']:.3%} 쓸만 {v['usable_ratio']:.2f} 포함80 {v['cov80']:.2f} 폭80 {v['med_w80']:.1%}")
    print("→", os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
