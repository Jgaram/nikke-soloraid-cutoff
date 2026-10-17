"""진행 중 시즌 예측의 고정 모델 파일을 만든다 (분석 4.3 마무리의 F, 2026-10-18 — 모델 0, 요청 0).

python3 -I tools/solo_raid/freeze_model.py --season S42 --xval F.json [--train S37,S38,S39,S40,S41] [--boot 60] [--seed 1]
        [--out data/solo-raid/analysis/model_S42.json] [--check]

만드는 것 (모두 data/solo-raid/analysis/):
- model_{시즌}_main_prior.json — 최근 5시즌 여러 시즌 prior, prior 맞추기에서 구간 관측(cens 있는 칸)을 뺀 것 (분석 4.3 마무리 ⑥ — lab4/m5run.py no_interval과 같은 셈)
- model_{시즌}_ref_prior.json  — 같은 5시즌 여러 시즌 prior 그대로 (M2a, 5·10·15·30%)
- model_{시즌}.json — 모델 파일: 두 prior의 경로·sha1, 빌릴 시즌, 「점 하나가 틀림」(main 끔), 라벨 줄만 수준 갱신, M1 배율표, 만든 근거.
predict.py --model이 읽고, predict_run.sh가 시즌 이름으로 고른다. **시즌 안에서는 다시 만들지 않는다** (sha1이 다르면 predict가 멈춘다).

- 학습 시즌(--train 기본): 맞힐 시즌보다 앞이고 수집 끝난 개발 시즌 가운데 학습 등급 full·shape_only인 가장 가까운 5개 (xval recent5와 같은 규칙).
- prior 붓스트랩·시드는 xval 여러 시즌 prior 기본(60·1)과 같게 — 고른 검증이 그 값으로 돌았다.
- M1: --xval은 F 설정으로 돌린 xval 결과 (scheme recent5, --level-update '{"unknown": false}', --predict-opt eps_wrong=0, m5run no_interval, 뽑기 2000·시드 1,
  T 12~96). 한섭 1·3% 줄의 |z|(정답이 80% 구간 반폭의 몇 배 밖인가, lab4 e_43.zval)를 T마다 모아 시즌마다 무게 합 1로 80% 분위를 낸다.
  맞힐 시즌보다 앞의 모든 개발 시즌 (e_43.m1의 forward 규칙을 맞힐 시즌 = 이번 시즌으로). 넓히기만 (≥ 1), 앞 시즌 3개 미만이면 1.
- --check: 같은 xval 줄로 e_43.m1의 시즌별 배율을 다시 내 이 도구의 셈과 견준다 (lab4가 있을 때).
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import predict as PR  # noqa: E402
import fit_prior as FP  # noqa: E402
import fit_prior_multi as FM  # noqa: E402
import split as SP  # noqa: E402

AN = os.path.join(ROOT, "data", "solo-raid", "analysis")
K_RECENT = 5
M1_MIN_PREV = 3
M1_Q = 0.8
M1_GROUP = (1.0, 3.0)
M1_APPLY = (0.5, 1.0, 2.0, 3.0)   # main 실행의 지점 (검증은 1·3% — 0.5·2%는 같은 실행이라 같은 배율로 맞춘다)
MAIN_PCTS = (0.5, 1.0, 2.0, 3.0)
REF_PCTS = (5.0, 10.0, 15.0, 30.0)


def sha1_of(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def recent_train(season, k=K_RECENT):
    reg = FM.registry()
    ok = [s for s, r in reg.items() if r["role"] == "dev" and r.get("collect") == "완료" and r["tier"] in ("full", "shape_only")
          and SP.snum(s) < SP.snum(season)]
    tr = sorted(ok, key=SP.snum)[-k:]
    if not any(reg[s]["tier"] == "full" for s in tr):
        raise SystemExit(f"{season}: 학습 등급 full 시즌이 없다")
    return tr


def fit(train, boot, seed, no_interval):
    """xval.prior_for의 여러 시즌 prior와 같은 셈 (fit_multi → jsonable 왕복). no_interval이면 load_units에서 구간 칸을 뺀다."""
    orig = FP.load_units
    if no_interval:
        def lu(state, *a, **k):
            season, H, obs, units = orig(state, *a, **k)
            return season, H, obs, [u for u in units if u["cens"] is None]
        FP.load_units = lu
    try:
        pr = FM.fit_multi(list(train), boot=boot, seed=seed, quiet=True)
    finally:
        FP.load_units = orig
    return json.loads(json.dumps(PR.jsonable(pr)))


def zval(r):
    """lab4 e_43.zval과 같다: 정답의 로그 거리 ÷ 그 쪽 80% 반폭."""
    lo, hi = r["i80"]
    m, t = r["pred"], r["truth"]
    if not (m and t and lo and hi) or lo <= 0 or hi <= m or m <= lo:
        return None
    d = math.log(t / m)
    h = math.log(hi / m) if d >= 0 else math.log(m / lo)
    return d / h if h > 0 else None


def wq(items, q):
    items = sorted(items)
    tot = sum(w for _, w in items)
    acc = 0.0
    for v, w in items:
        acc += w
        if acc >= q * tot - 1e-12:
            return v
    return items[-1][0]


def m1_table(rows, target):
    """T → (배율, 시즌 수, 줄 수). 맞힐 시즌보다 앞 시즌의 한섭 1·3% 줄만."""
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("err") is None or r.get("server", "KR") != "KR" or r["pct"] not in M1_GROUP or SP.snum(r["season"]) >= SP.snum(target):
            continue
        z = zval(r)
        if z is not None:
            by[float(r["T"])].append((r["season"], abs(z)))
    out = {}
    for T, L in sorted(by.items()):
        cnt = collections.Counter(s for s, _ in L)
        c = 1.0 if len(cnt) < M1_MIN_PREV else max(1.0, wq([(z, 1.0 / cnt[s]) for s, z in L], M1_Q))
        out[T] = (c, len(cnt), len(L))
    return out


def check_with_lab(rows):
    sys.path.insert(0, os.path.join(HERE, "lab4"))
    import e_43 as E43
    _, fac = E43.m1(rows, groups=(M1_GROUP,))
    bad = 0
    for Tk, d in fac["1/3"].items():
        T = float(Tk[1:])
        for tgt, c in d.items():
            mine = m1_table(rows, tgt).get(T, (1.0,))[0]
            if abs(mine - c) > 1e-3:
                bad += 1
                print(f"!! 다름 {tgt} T{T:g}: e_43 {c} · 이 도구 {mine:.4f}")
    print(f"e_43.m1 대조: 다른 칸 {bad}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    ap.add_argument("--xval", required=True, help="F 설정 xval 결과 (머리말)")
    ap.add_argument("--train", help="학습 시즌 (쉼표). 기본: 앞 수집 끝난 개발 시즌 가운데 가장 가까운 5개")
    ap.add_argument("--boot", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--note", default="", help="모델 파일 about에 덧붙일 말")
    a = ap.parse_args()
    season = a.season
    train = a.train.split(",") if a.train else recent_train(season)
    out = a.out or os.path.join(AN, f"model_{season}.json")
    with open(a.xval, encoding="utf-8") as f:
        xv = json.load(f)
    rows = [r for r in xv["rows"] if r.get("scheme") == "recent5"]
    if not rows:
        raise SystemExit("xval 결과에 recent5 줄이 없다")
    if any(SP.snum(r["season"]) >= SP.snum(season) for r in rows):
        raise SystemExit(f"xval 줄에 {season}이나 그 뒤 시즌이 있다 — 맞힐 시즌의 정답은 넣지 않는다")
    run = xv.get("run") or {}
    if a.check:
        if check_with_lab(rows):
            raise SystemExit("e_43.m1과 배율이 다르다")
    tab = m1_table(rows, season)
    print(f"학습 시즌 {'+'.join(train)} · M1 배율 (한섭 1·3%, 앞 시즌 전부): "
          + " · ".join(f"T{T:g} {c:.3f} ({n}시즌)" for T, (c, n, _) in tab.items()))
    files = {}
    for part, noint in (("main", True), ("ref", False)):
        pr = fit(train, a.boot, a.seed, noint)
        p = os.path.join(os.path.dirname(out), f"model_{season}_{part}_prior.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(pr, f, ensure_ascii=False)
            f.write("\n")
        files[part] = p
        print(f"{part} prior → {os.path.relpath(p, ROOT)} ({pr['season']}, 구간 관측 {'뺌' if noint else '그대로'})")
    rel = lambda p: os.path.relpath(p, ROOT)  # noqa: E731
    model = {
        "about": ("진행 중 시즌 예측의 고정 모델 (tools/solo_raid/freeze_model.py가 씀, predict.py --model이 읽음). "
                  "분석 4.3 마무리에서 고른 F (docs/solo-raid-analysis.md 「4.3 마무리 결과」·「4.5 고정과 미래 시험」). 시즌 안에서는 바꾸지 않는다." + (" " + a.note if a.note else "")),
        "name": "F", "season": season, "borrow": train,
        "main": {"prior": rel(files["main"]), "prior_sha1": sha1_of(files["main"]), "pcts": list(MAIN_PCTS), "eps_wrong": 0.0,
                 "level_update": {"unknown": False},
                 "prior_fit": {"train": train, "boot": a.boot, "seed": a.seed, "interval_units": "뺌 (4.3 마무리 ⑥)"},
                 "what": "최근 5시즌 여러 시즌 prior(구간 관측 뺌) + 「점 하나가 틀림」 끔 + 라벨 줄만 자체 일관성 수준 갱신(M4l) + M1"},
        "ref": {"prior": rel(files["ref"]), "prior_sha1": sha1_of(files["ref"]), "pcts": list(REF_PCTS), "eps_wrong": PR.EPS_WRONG,
                "prior_fit": {"train": train, "boot": a.boot, "seed": a.seed, "interval_units": "그대로"},
                "what": "최근 5시즌 여러 시즌 prior 그대로, 덧붙임 없음 (M2a — 4.3 마무리 C)"},
        "m1": {"group": list(M1_GROUP), "apply": list(M1_APPLY), "q": M1_Q, "min_prev": M1_MIN_PREV,
               "T": {f"{T:g}": round(c, 4) for T, (c, _, _) in tab.items()},
               "seasons": {f"{T:g}": n for T, (_, n, _) in tab.items()}, "rows": {f"{T:g}": k for T, (_, _, k) in tab.items()},
               "rule": "T별 값 사이 선형 보간, 표 밖은 가장 가까운 끝값 (predict.m1_factor). 50·80·95% 구간 모두 같은 배율 (검증은 80%만)",
               "source": {"xval": os.path.basename(a.xval), "xval_sha1": sha1_of(a.xval), "tool_head": run.get("tool_head"),
                          "tools_dirty": run.get("tools_dirty"), "lab_patches": run.get("lab_patches"),
                          "seasons": sorted({r["season"] for r in rows}, key=SP.snum)}},
        "grade": "predict 등급 그대로 (문턱 옛 값, M1 넓히기 전 폭 — 4.3 마무리 ③)",
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"→ {rel(out)}")


if __name__ == "__main__":
    main()
