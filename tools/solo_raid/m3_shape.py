"""M3 운영 꼴 칸 (WP7, 2026-11-07) — 자료가 아직 없는 다음 시즌 S{n}을 맞힐 꼴 파일 (형식 v1, 맞힐 시즌 하나).

  python3 -I tools/solo_raid/m3_shape.py --target S43 --obs-dir W/obs7/full --out data/solo-raid/analysis/m3/S43/shape.json
  재현 확인: --target S42 --blind (S42 자료를 안 본 것처럼 같은 길로) --check data/solo-raid/analysis/r7/wp13/shape_k4_D1.json --out $SC/shape_S42.json

연구의 꼴 파일 (r7/wp13/shape_k4_D1.json)과 같은 규칙·같은 연구 함수 (lab7: wp13_flood_emp.recipe/post · wp13_cut_shape.make_shape/cut_obj · wp13_shape.tau_oos)를 그대로 불러 쓴다 (lab7 파일은 안 고친다).
  - 컷 꼴: 학습 = 시즌표(S19~)에서 target보다 앞선 전부, --share 0.99 --kmax 4 (k4 판).
  - 침수 꼴: 학습 = 시계 맞는 시즌(05시 시작 120시간, S28·S38 제외) 중 target보다 앞선 것 → source "emp".
    target이 05시 시작 120시간이 아니면 멈춘다 (HGAM 길은 여기 없다).
  - tau_oos: 연구 꼴 파일의 앞 맞힐 시즌 t_true (--prior-shape, 기본 shape_k4_D1.json의 flood 칸)로 wp13_shape.tau_oos 규칙 그대로.
  - target 시즌 자료가 필요한 칸 — m_true(진단·--bind용)·t_true·stability(전체 판과의 차)·cut.true_fit — 은 target이 자료 없는 시즌이면 만들지 않는다 (m_true는 lnm·se = null,
    t_true는 null). target 자료를 여는 코드(state_of 등)는 부르지 않는다: 관측 표는 target 이전 시즌만 쓴다.
  - --blind: target이 자료 있는 시즌(재현용)이어도 그 자료를 보지 않은 것처럼 같은 길을 탄다 (m_true·t_true 칸은 null). --check이 있으면 연구 칸과 견줘 숫자 차 최대값을 낸다.
공개 규칙: 시각·%·점수·개수만 (사람·사진 열쇠 없음).
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB7 = os.path.join(HERE, "lab7")
sys.path.insert(0, LAB7)
sys.path.insert(0, HERE)
import m3_extend as XT  # noqa: E402 — 연구 시즌 목록(S19~S42)을 늘리는 길. --extra-seasons가 없으면 아무것도 안 바뀐다
_EXTRA, _DROP = XT.argv_list(sys.argv, "--extra-seasons"), XT.argv_list(sys.argv, "--drop-seasons")
XT.pre_import(add=_EXTRA, drop=_DROP)

import numpy as np  # noqa: E402
import common as C  # noqa: E402
import wp12_tables as WT  # noqa: E402
import wp12_flood_curve as FC  # noqa: E402
import wp13_flood_shape as WF  # noqa: E402
import wp13_flood_emp as FE  # noqa: E402
import wp13_cut_shape as CS  # noqa: E402
import wp13_shape as SH  # noqa: E402
if _EXTRA or _DROP:
    XT.post_import()

R13 = os.path.join(C.ROOT, "data", "solo-raid", "analysis", "r7", "wp13")
SHARE, KMAX = 0.99, 4          # k4 판 (cut_shape_k4_D1.json: --share 0.99 --kmax 4)


def sha12(p):
    with open(p, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def git_head():
    try:
        return subprocess.check_output(["git", "-C", C.ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def flood_entry(tn):
    """target 침수 칸. target 자료가 필요한 칸 (m_true·t_true·stability)은 null (target 자료는 안 본다)."""
    s = f"S{tn}"
    train = [x for x in FC.CLOCK if int(x[1:]) < tn]
    if not train:
        sys.exit("침수 학습 시즌이 없다")
    rf = FE.recipe(FE.O_FC, train)
    full_rf = rf             # post가 쓰는 전체 판과의 차(stability)는 이 길에서 자기 자신 (0) — 어차피 아래서 지운다
    orig = (WF.m_fit, WF.level_fit)
    blind_ix = WF.SIDX.get(s)
    added = False
    if blind_ix is None:                          # 시즌표에 없는 시즌 (S43~)
        blind_ix = max(WF.SIDX.values()) + 1
        WF.SIDX[s] = blind_ix
        WF.SI[s] = {"H": 120.0}
        added = True
    nan_se = (float("nan"), float("nan"))

    def m_fit(obs, rws, *a, **k):
        if len(rws) == 0 or int(obs["S"][rws[0]]) == blind_ix:
            return nan_se
        return orig[0](obs, rws, *a, **k)

    def level_fit(obs, rws, *a, **k):
        if len(rws) == 0 or int(obs["S"][rws[0]]) == blind_ix:
            return {}
        return orig[1](obs, rws, *a, **k)

    WF.m_fit, WF.level_fit = m_fit, level_fit
    try:
        e = FE.post(tn, rf, full_rf, train)
    finally:
        WF.m_fit, WF.level_fit = orig
        if added:
            WF.SIDX.pop(s), WF.SI.pop(s)
    e["m_true"] = {"lnm": None, "se": None}
    e["stability"] = {"note": "자기 자신 (전체 판과의 차 없음)"}
    e["t_true"] = [None] * len(e["t_true"])
    return e


def cut_entry(tn, pts):
    """target 컷 칸 (k4). 학습 = 시즌표에서 target보다 앞선 전부. target 점이 필요한 칸 (true_fit·stability·v1/v2)은 만들지 않는다."""
    s = f"S{tn}"
    train = [x for x in WT.SEASONS if int(x[1:]) < tn]
    sh = CS.make_shape(pts, train, CS.RIDGE)
    c = CS.cut_obj(sh, train)
    for x in train:
        c["train_fit"][x]["n_pts"] = pts[x]["n"]
    c["self_check"] = CS.self_check(sh, s)
    return c


def jclean(o):
    """NaN → null (JSON)."""
    if isinstance(o, float):
        return None if math.isnan(o) else o
    if isinstance(o, dict):
        return {k: jclean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [jclean(v) for v in o]
    return o


def maxdiff(a, b, path="", out=None):
    """두 JSON 값의 숫자 차 최대값 (구조가 다르면 그 자리를 기록). 돌려줌: (최대 |차|, 자리)."""
    best = [0.0, ""]
    bad = []

    def rec(x, y, p):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in set(x) | set(y):
                if k not in x or k not in y:
                    bad.append(f"{p}/{k} 한쪽에만 있음")
                else:
                    rec(x[k], y[k], f"{p}/{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                bad.append(f"{p} 길이 {len(x)}≠{len(y)}")
            for i, (u, v) in enumerate(zip(x, y)):
                rec(u, v, f"{p}[{i}]")
        elif isinstance(x, bool) or isinstance(y, bool) or isinstance(x, str) or isinstance(y, str) or x is None or y is None:
            if x != y:
                bad.append(f"{p}: {x!r} ≠ {y!r}")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            d = abs(float(x) - float(y))
            if d > best[0]:
                best[0], best[1] = d, p
        else:
            bad.append(f"{p} 형이 다름")
    rec(a, b, path)
    return best[0], best[1], bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True, help="맞힐 시즌 (예: S43)")
    ap.add_argument("--obs-dir", required=True, help="끝난 시즌 전체 관측 표 폴더 (obs_table.py build --full 의 full/)의 부모가 아니라 obs7/full 자체")
    ap.add_argument("--prior-shape", default=os.path.join(R13, "shape_k4_D1.json"), help="tau_oos에 쓸 앞 맞힐 시즌들의 꼴 파일 (연구)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blind", action="store_true", help="target이 자료 있는 시즌이어도 안 본 것처럼 (재현용)")
    ap.add_argument("--check", default=None, help="연구 꼴 파일 — target 칸과 견줘 숫자 차 최대값을 낸다")
    ap.add_argument("--cmd-note", default=None)
    ap.add_argument("--extra-seasons", default=None, help="연구 시즌 목록에 더할 끝난 시즌 (예: S43 — 다음 시즌 S44 꼴을 만들 때). 자료(obs-dir)와 정답 표(answer_key.json)가 있어야 한다")
    ap.add_argument("--drop-seasons", default=None, help="리허설용: 연구 시즌 목록에서 뺀다 (뺀 시즌을 --extra-seasons로 다시 더해 같은 결과인지 본다)")
    a = ap.parse_args()
    tn = int(a.target[1:])
    s = a.target
    known = s in WF.SIDX
    if known and not a.blind:
        sys.exit(f"{s}는 자료가 있는 시즌이다 — 재현이면 --blind (자료 없는 시즌만 blind 없이)")
    if not known and a.blind:
        sys.exit(f"{s}는 자료 없는 시즌이라 --blind가 필요 없다")
    if known:
        si = WF.SI[s]
        if not si.get("clock_ok"):
            sys.exit(f"{s}는 05시 시작 120시간이 아니다 (clock_ok 아님)")
    FE.init_worker(a.obs_dir)
    prior = SH.jload(a.prior_shape)
    ptg = prior["targets"]
    # tau_oos는 target보다 앞 맞힐 시즌의 t_true만 쓴다 (SH.tau_oos가 j<s 만 고른다)
    # ---- 침수
    fe = flood_entry(tn)
    fe = jclean(fe)
    # ---- 컷
    CS.PC_SHARE, CS.PC_KMAX = SHARE, KMAX
    pts = CS.season_points()
    for x in WT.SEASONS:
        pts[x]["snum"] = int(x[1:])
    ce = cut_entry(tn, pts)
    # ---- 합침 (wp13_shape.py 규칙: 침수 학습 ⊆ 컷 학습, 학습은 target 앞)
    ff, cc = fe["train"], ce["train"]
    assert all(int(x[1:]) < tn for x in ff + cc) and set(ff) <= set(cc)
    tf = {j: v["flood"] for j, v in ptg.items() if int(j[1:]) < tn}
    tf[s] = fe
    fe = dict(fe, tau_oos=SH.tau_oos(tf, s))
    ents = {"flood": fe, "cut": ce}
    flood_in = os.path.join(R13, "flood_shape_D1.json")
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
    rules_src = prior["header"]
    cmd = " ".join(["python3 -I tools/solo_raid/m3_shape.py"] + sys.argv[1:])
    ncut = len(cc)
    head = {"about": "M3 운영 꼴 파일 (형식 v1) — 자료 없는 다음 시즌 하나를 맞힐 침수 꼴 P̄·컷 꼴 G·편차 층 사전 (앞 시즌만)",
            "data_ver": C.DATA_VER, "format": 1,
            "made": {"tool": "tools/solo_raid/m3_shape.py", "tool_head": git_head(), "cmd": cmd, "at_kst": now,
                     "lab7_tools": ["wp13_flood_emp.recipe/post", "wp13_cut_shape.make_shape/cut_obj (--share 0.99 --kmax 4)", "wp13_shape.tau_oos"]},
            "inputs": {"obs_full": {"dir": "obs7/full (obs_table.py build --full, S19~S42)",
                                    "sha1": {x: sha12(os.path.join(a.obs_dir, f"{x}.jsonl.gz")) for x in ["S" + str(n) for n in range(19, tn)] if os.path.exists(os.path.join(a.obs_dir, f"{x}.jsonl.gz"))}},
                       "prior_shape": {"file": os.path.relpath(a.prior_shape, C.ROOT) if os.path.isabs(a.prior_shape) else a.prior_shape, "sha1": sha12(a.prior_shape),
                                       "use": "tau_oos의 앞 맞힐 시즌 t_true (target 앞 시즌 칸만)"}},
            "flood_rules": rules_src.get("flood_rules"), "cut_rules": rules_src.get("cut_rules"),
            "flood_source": {s: "emp"},
            "rules": {"flood_rules·cut_rules": "원본 shape_k4_D1.json header의 것 그대로 복사",
                      "운영 WP7": f"target {s}, 앞 시즌 S19~S{tn - 1} (컷 학습 {ncut}시즌, 침수 학습 {len(ff)}시즌 {ff[0]}~{ff[-1]}, S28·S38 제외 — 05시 시작 120시간 시즌만, source emp)",
                      "자료 없는 target": "m_true(lnm·se)·t_true·stability·cut.true_fit 칸은 target 자료가 없어 null/없음. m3 --bind m 진단은 쓸 수 없다. 나머지 칸은 연구 shape_k4_D1.json과 같은 도구·같은 규칙"}}
    out = {"header": head, "targets": {s: ents}}
    if a.blind:
        head["rules"]["blind"] = f"재현용 {s} 자료를 안 본 길 (m_true·t_true 칸은 null)"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, a.out)
    print("wrote", a.out, os.path.getsize(a.out), "bytes")
    if a.check:
        ref = SH.jload(a.check)["targets"][s]
        skip = {("flood", "m_true"), ("flood", "t_true"), ("flood", "stability"), ("flood", "checks")}
        res = {}
        for part in ("flood", "cut"):
            mine = {k: v for k, v in ents[part].items() if (part, k) not in skip}
            theirs = {k: v for k, v in ref[part].items() if k in mine}
            extra = [k for k in ref[part] if k not in mine]
            d, where, bad = maxdiff(mine, theirs)
            res[part] = (d, where, bad, extra)
            print(f"[check {s}] {part}: 숫자 차 최대 {d:.3e} @ {where}; 구조 차 {len(bad)}; 연구에만 있는 칸 {extra}")
            for b in bad[:10]:
                print("   ", b)
        # checks 칸은 일부만 견줌 (target 자료 무관한 것)
        for k in ("P_monotone", "lam_nonneg", "lam_zero_points", "mean_lnm_train", "P_last", "A_day_sum_max_err"):
            x, y = ents["flood"]["checks"][k], ref["flood"]["checks"][k]
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                print(f"   checks.{k}: {abs(x - y):.3e}")
            elif x != y:
                print(f"   checks.{k}: {x!r} ≠ {y!r}")


if __name__ == "__main__":
    main()
