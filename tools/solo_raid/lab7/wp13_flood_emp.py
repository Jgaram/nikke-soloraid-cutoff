"""WP13a 침수 꼴 (대체판) — WP12 합친 경험 스플라인을 맞힐 시즌마다 앞 시즌만으로 다시 적합 (2026-11-04).

  python3 -I tools/solo_raid/lab7/wp13_flood_emp.py --obs-dir W/obs7/full [--hgam r7/wp13/flood_hgam_D1.json] [--out r7/wp13/flood_shape_D1.json] [--procs 2]
  (W = 관측 표 폴더의 부모. 경로는 data/solo-raid/analysis 아래 상대 또는 절대)

HGAM(wp13_flood_shape.py)은 재현 점검(|ΔP| ≤ 0.01, h 12~120)을 못 넘었다. 계획의 대체 규칙을 쓴다:
  "24시즌을 다 넣은 HGAM이 WP12 합친 경험 곡선과 |ΔP| ≤ 0.01 (h 12~120). 못 미치면 매듭을 늘리거나 벌점을 줄여 다시 본다.
   그래도 못 미치면 05시 시작 시즌은 WP12 경험 스플라인 (h 축)을 꼴로 쓰고 S28·S38만 HGAM — 어느 쪽을 썼는지 적는다."
- 05시 시작 120시간 시즌(S21~S42 중 S28·S38 제외): WP12 G1 「합친 곡선」 방법 그대로, 맞힐 시즌 앞의 시계 맞는 시즌만으로 다시 적합.
- S28·S38: HGAM 항목을 그대로 가져옴 (source = "hgam").
공용 코드(wp12_*.py, wp13_flood_shape.py)는 읽기만 한다 (가져다 쓴다).
공개 규칙: 글 번호·원문·닉네임·원글 주소·사람 열쇠·사진 열쇠를 쓰지 않는다 (시각·%·점수·개수만).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as CM  # noqa: E402
import wp12_tables as T  # noqa: E402
import wp12_flood_curve as FC  # noqa: E402
import wp13_flood_shape as W  # noqa: E402

SEAS, SIDX, SI = T.SEASONS, W.SIDX, W.SI
CLOCK = FC.CLOCK
R13 = os.path.join(T.R7, "wp13")
WP12_JSON = W.WP12_JSON
HH, DTc = 120.0, 0.05


class Cal120:
    """05시 시작 120시간 달력 (적분 0.05시간 칸). W.level_fit·m_fit이 쓰는 인터페이스만."""
    N, dt = int(round(HH / DTc)), DTc
    edges = W.day_edges_of(5.0, HH)

    def at(self, Ce, t):
        u = np.clip(np.asarray(t, float) / self.dt, 0.0, self.N)
        i = np.minimum(np.floor(u).astype(int), self.N - 1)
        f = u - i
        return Ce[i] * (1 - f) + Ce[i + 1] * f


CAL = Cal120()
GMID = (np.arange(CAL.N) + 0.5) * DTc
OUTG = FC.GRID                       # 0.25 h 격자 0..120

O_FC = None      # FC.build_obs 그대로 (적합용)
OBS = None       # W.load_obs (level_fit·m_fit용, 사진 일차 번호만 다시 셈)


def init_worker(obs_dir):
    global O_FC, OBS
    O_FC = FC.build_obs(obs_dir)[0]
    OBS = W.load_obs(obs_dir)


def recipe(Oall, seasons):
    """WP12 main() 1b 방법 그대로. seasons: 쓸 시계 맞는 시즌 목록. 돌려줌 dict."""
    ids = [SIDX[s] for s in seasons]
    msk = np.isin(Oall["S"], ids)
    Oc = FC.sub(Oall, msk)
    cidx = {s: i for i, s in enumerate(seasons)}
    Oc["si"] = np.array([cidx[SEAS[i]] for i in Oc["S"]])
    ns = len(seasons)
    Xc = FC.comb_X(Oc)
    DtD = FC.d2pen(Xc.shape[1])
    _, _, _, sig0, _ = FC.fit_full(Xc, Oc["y"], Oc["wp"], Oc["isph"], Oc["si"], ns, 10.0, DtD, FC.Sg(0.03, 1.0), rounds=2)
    err = FC.cv_kappa(Xc, Oc["y"], Oc["wp"], Oc["isph"], Oc["si"], ns, DtD, sig0, FC.KAPPA_GRID_C, np.random.default_rng(99))
    kap = FC.pick_kappa(err, FC.KAPPA_GRID_C)
    beta, m, r, sig, _ = FC.fit_full(Xc, Oc["y"], Oc["wp"], Oc["isph"], Oc["si"], ns, kap, DtD, sig0, rounds=3)
    return {"beta": beta, "m": m, "kap": kap, "sig": sig, "err": err, "n": Oc["n"], "seasons": seasons}


def shape_of(beta):
    """beta → (Ce 0.05시간 칸 누적, 0.25시간 격자 P, λ)."""
    lam = np.clip(FC.lam_pred(beta, HH, GMID), 0.0, None)
    Ce = np.concatenate([[0.0], np.cumsum(lam * DTc)])
    P = Ce[-1] - CAL.at(Ce, OUTG)
    return Ce, P, FC.lam_pred(beta, HH, OUTG)


def job(args):
    tn = args
    t0 = time.time()
    if tn is None:
        rf = recipe(O_FC, list(CLOCK))
        return None, {"rf": rf}, time.time() - t0
    train = [s for s in CLOCK if int(s[1:]) < tn]
    rf = recipe(O_FC, train)
    Ce, Pbar, lam_g = shape_of(rf["beta"])
    return tn, {"rf": rf, "train": train}, time.time() - t0


def post(tn, rf, full_rf, train):
    """HGAM target_job 뒤처리와 같은 셈, 꼴만 경험 스플라인."""
    sig = rf["sig"]
    Ce, Pbar, lam_g = shape_of(rf["beta"])
    Ce_f, Pf, _ = shape_of(full_rf["beta"])
    edges = CAL.edges
    Dn = len(edges) - 1
    hg = OUTG
    tseason = f"S{tn}"
    Ad = np.zeros((Dn, len(hg)))
    for d in range(Dn):
        Ad[d] = np.where(hg >= edges[d + 1] - 1e-9, 0.0, CAL.at(Ce, edges[d + 1]) - CAL.at(Ce, np.maximum(hg, edges[d])))
    J = len(train)
    lnm = np.zeros(J); se = np.zeros(J)
    for j, s in enumerate(train):
        rws = np.where(OBS["S"] == SIDX[s])[0]
        lnm[j], se[j] = W.m_fit(OBS, rws, CAL, Ce, sig)
    lnm_train = {s: [round(float(lnm[j]), 5), round(float(se[j]), 5)] for j, s in enumerate(train)}
    raw = float(np.var(lnm, ddof=1)) if J >= 2 else None
    noise = float(np.mean(se ** 2))
    var = 0.3 ** 2 if J < 2 else max(raw - noise, 0.01 ** 2)
    lnm_prior = {"mean": round(float(lnm.mean()), 6), "var": round(var, 6), "raw_var": None if raw is None else round(raw, 6),
                 "noise_var": round(noise, 6), "J": J}
    samples = {k: [] for k in range(1, 6)}
    noises = {k: [] for k in range(1, 6)}
    allv, alln, dropped = [], [], 0
    for j, s in enumerate(train):
        rws = np.where(OBS["S"] == SIDX[s])[0]
        for d, (L, seln) in W.level_fit(OBS, rws, CAL, Ce, sig, edges).items():
            if seln > 1.0:
                dropped += 1
                continue
            lt = math.log(L) - float(lnm[j])
            k = min(d + 1, 5)
            samples[k].append(lt); noises[k].append(seln ** 2)
            allv.append(lt); alln.append(seln ** 2)
    pooled_raw = float(np.var(allv, ddof=1)) if len(allv) >= 2 else 0.3 ** 2
    pooled_noise = float(np.mean(alln)) if alln else 0.0
    pooled = math.sqrt(max(pooled_raw - pooled_noise, 0.01 ** 2)) if len(allv) >= 2 else 0.3
    per_day, n_l, raw_l, noise_l = [], [], [], []
    for k in range(1, 6):
        v = samples[k]
        n_l.append(len(v))
        rw = float(np.var(v, ddof=1)) if len(v) >= 2 else None
        nz = float(np.mean(noises[k])) if v else None
        raw_l.append(None if rw is None else round(rw, 6)); noise_l.append(None if nz is None else round(nz, 6))
        per_day.append(round(math.sqrt(max(rw - nz, 0.01 ** 2)), 5) if len(v) >= 3 else round(pooled, 5))
    tau = {"per_day": per_day, "pooled": round(pooled, 5), "n": n_l, "raw": raw_l, "noise": noise_l,
           "pooled_raw": round(pooled_raw, 6), "pooled_noise": round(pooled_noise, 6), "dropped_se_gt_1": dropped,
           "note": "raw·noise는 분산 (ln t_{j,d}의 시즌 사이 표본 분산 · 델타법 se² 평균), per_day·pooled는 sd."}
    rws = np.where(OBS["S"] == SIDX[tseason])[0]
    lmt, lmt_se = W.m_fit(OBS, rws, CAL, Ce, sig)
    t_true = [None] * Dn
    for d, (L, _) in W.level_fit(OBS, rws, CAL, Ce, sig, edges).items():
        t_true[d] = round(math.log(L) - lmt, 5)
    mk = (hg >= 12) & (hg <= HH)
    mk2 = (hg >= 12) & (hg <= HH - 1)
    dl = np.log(np.maximum(Pbar[mk2], 1e-12)) - np.log(np.maximum(Pf[mk2], 1e-12))
    stab = {"max_dP_vs_full": round(float(np.abs(Pbar - Pf)[mk].max()), 5),
            "max_dlnP_shape_vs_full": round(float(np.abs(dl - dl.mean()).max()), 5)}
    chk = {"A_day_sum_max_err": float(np.abs(Ad.sum(axis=0) - Pbar).max()), "P_monotone": bool(np.all(np.diff(Pbar) <= 1e-12)),
           "lam_nonneg": bool(np.all(lam_g >= 0)), "lam_zero_points": int((lam_g <= 0).sum()),
           "mean_lnm_train": float(lnm.mean()), "P_last": float(Pbar[-1]),
           "day_edges_check": [int(W.day_c0(5.0, 0.5 * (edges[d] + edges[d + 1]))) for d in range(Dn)]}
    cv = {"cv": {f"{k:g}": round(float(e), 3) for k, e in zip(FC.KAPPA_GRID_C, rf["err"])}, "chosen": float(rf["kap"])}
    pen = {"kappa": float(rf["kap"]),
           "sigma": {"sigma_ph": float(sig), "phi_pair": float(sig.phi), "sigma_ph_day": [round(float(v), 5) for v in sig.sd]}}
    R = lambda a: [round(float(v), 6) for v in a]  # noqa: E731
    return {"train": train, "H": float(SI[tseason]["H"]), "Ht": HH, "c0h": 5.0, "h": R(hg), "P": R(Pbar), "lam": R(lam_g),
            "day_edges": [round(e, 4) for e in edges], "A_day": [R(a) for a in Ad], "lnm_prior": lnm_prior, "lnm_train": lnm_train,
            "tau": tau, "m_true": {"lnm": round(lmt, 5), "se": round(lmt_se, 5)}, "t_true": t_true, "stability": stab,
            "penalties": pen, "cv": cv, "checks": chk, "source": "emp"}


def git_head():
    try:
        return T._git_head()
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--hgam", default=os.path.join(R13, "flood_hgam_D1.json"))
    ap.add_argument("--out", default=None, help="기본: emp는 flood_shape_D1.json, 그 밖은 flood_smooth_{kind}_D1.json")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--kind", choices=["emp", "m3smooth", "f2pow"], default="emp", help="emp = 경험 스플라인(기본). m3smooth·f2pow = 매끈한 꼴 (WP13b 꼴 시험 비교용)")
    ap.add_argument("--ref", default=os.path.join(R13, "flood_shape_D1.json"), help="매끈한 꼴과 견줄 꼴 파일 (.md의 |ΔP|)")
    ap.add_argument("--stage1", default=os.path.join(T.R7, "stage1_M3_D1.json"))
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(R13, "flood_shape_D1.json" if a.kind == "emp" else f"flood_smooth_{a.kind}_D1.json")
    if a.kind != "emp":
        return main_smooth(a)
    t_all = time.time()
    rel = lambda p: p if os.path.isabs(p) else os.path.join(CM.ROOT, "data/solo-raid/analysis", p)  # noqa: E731
    hgam_path, out_path = rel(a.hgam), rel(a.out)

    def log(m):
        print(f"[{time.time() - t_all:7.1f}s] {m}", flush=True)

    hg = json.load(open(hgam_path, encoding="utf-8"))
    sha = hashlib.sha1(open(hgam_path, "rb").read()).hexdigest()[:12]
    tns = [n for n in range(21, 43) if f"S{n}" not in ("S28", "S38")]
    jobs = [None] + sorted(tns, reverse=True)       # 오래 걸리는 큰 시즌부터
    res = {}
    with mp.Pool(max(1, min(a.procs, 2)), initializer=init_worker, initargs=(a.obs_dir,)) as pool:
        for tn, r, secs in pool.imap_unordered(job, jobs):
            res[tn] = r
            log(f"  {'FULL' if tn is None else 'S%d' % tn}: κ={r['rf']['kap']:g} σ={float(r['rf']['sig']):.4f} n={r['rf']['n']} ({secs:.0f}s)")
    init_worker(a.obs_dir)
    # 작은 확인: 두 관측 표가 시계 시즌에서 같은 줄을 주는지
    for k in ("S", "ta", "tb", "y", "isph"):
        assert np.array_equal(O_FC[k], OBS[k]), k
    frf = res[None]["rf"]
    # ---- 전체 21시즌 점검
    tab = json.load(open(WP12_JSON, encoding="utf-8"))["G1"]["tables"]["combined_grid_0.25h"]
    th = np.array([t["h"] for t in tab]); tP = np.array([t["P"] for t in tab])
    Ce_f, Pf, lam_f = shape_of(frf["beta"])
    assert np.allclose(th, OUTG)
    _, Pcc = FC.curves_from_beta(frf["beta"], HH, OUTG)
    d_all = np.abs(Pcc - tP)
    d12 = d_all[OUTG >= 12]
    d_int = np.abs(Pf - tP)[OUTG >= 12]
    full_check = {"max_abs_dP_h0_120": round(float(d_all.max()), 6), "max_abs_dP_h12_120": round(float(d12.max()), 6),
                  "max_abs_dP_h12_120_ce_integral": round(float(d_int.max()), 6),
                  "note": "같은 코드·같은 관측이면 ~1e-4 안. 앞은 curves_from_beta(WP12 그대로), 뒤는 0.05시간 칸 누적(이 도구가 쓰는 셈)"}
    log(f"전체 점검: {full_check}")
    full = {"seasons": frf["seasons"], "kappa": float(frf["kap"]), "sigma_ph": float(frf["sig"]), "phi_pair": float(frf["sig"].phi),
            "sigma_ph_day": [round(float(v), 5) for v in frf["sig"].sd], "n_rows": frf["n"],
            "m": {s: round(float(frf["m"][i]), 5) for i, s in enumerate(frf["seasons"])},
            "h": [round(float(v), 2) for v in OUTG], "P": [round(float(v), 6) for v in Pf]}
    targets = {}
    for n in range(21, 43):
        s = f"S{n}"
        if n in tns:
            targets[s] = post(n, res[n]["rf"], frf, res[n]["train"])
        else:
            targets[s] = dict(hg["targets"][s]); targets[s]["source"] = "hgam"
    secs = time.time() - t_all
    cmd = "python3 -I tools/solo_raid/lab7/wp13_flood_emp.py --obs-dir W/obs7/full --procs 2"
    pack = {"header": {
        "about": "WP13a 침수 꼴 — 05시 시작 시즌은 WP12 합친 경험 스플라인(앞 시즌만, 늘려가며), S28·S38은 HGAM (flood_shape_D1)",
        "data_ver": CM.DATA_VER, "format": 1,
        "made": {"tool_head": git_head(), "cmd": cmd, "at_kst": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")},
        "secs": round(secs, 1),
        "rules": {
            "fallback": "계획 「재현 점검」: \"24시즌을 다 넣은 HGAM이 WP12 합친 경험 곡선과 |ΔP| ≤ 0.01 (h 12~120). 못 미치면 매듭을 늘리거나 벌점을 줄여 다시 본다. "
                        "그래도 못 미치면 05시 시작 시즌은 WP12 경험 스플라인 (h 축)을 꼴로 쓰고 S28·S38만 HGAM — 어느 쪽을 썼는지 적는다.\" HGAM은 시도한 모든 판에서 못 미쳐(최선 k16/cvmin 0.0111, h 12~18) 이 규칙을 쓴다.",
            "emp": "맞힐 시즌 s(05시 시작 120시간)마다: 시계 맞는 시즌 중 s보다 앞선 것(S20~)만으로 WP12 G1 합친 곡선 방법 그대로 — 쌍+사진 관측, 3차 B-스플라인(1시간 매듭, NNLS, 2차 차분 벌점), "
                   "시즌 배수 m(기하평균 1), Huber(1.345)+σ_ph·φ 되풀이, κ는 관측 무작위 5겹 CV(격자 10^[-2..6] 반 자리, default_rng(99)). λ(h)=Σβ B(h) ≥ 0, P(h)=∫_h^120 λ (0.05시간 칸 누적).",
            "hgam": "S28(18.5시 시작 130시간)·S38(12시 시작 113시간)은 HGAM 항목을 그대로 복사, source = \"hgam\". 나머지 source = \"emp\".",
            "post": "lnm_prior·tau·m_true·t_true는 꼴 고정 + wp13_flood_shape.m_fit·level_fit (HGAM과 같은 셈). emp 꼴은 학습 시즌 ln m 평균이 0이 아니다(lnm_prior.mean이 그 값).",
            "stability": "맞힐 시즌 꼴과 21시즌 전체 경험 꼴의 차 (h 12~120)"},
        "inputs": {"hgam": {"file": os.path.relpath(hgam_path, CM.ROOT), "sha1": sha}},
        "full_check": full_check,
        "notes": ["관측: FC.build_obs와 W.load_obs가 시계 시즌에서 같은 줄(시즌·구간·값)을 줌을 확인; 적합은 FC, level_fit·m_fit은 W.load_obs(사진 일차 번호만 다시 셈) 사용.",
                  "S21 맞힐 때 학습은 S20 하나 → m 고정 1, lnm_prior.var = 0.3².",
                  "λ가 정확히 0인 격자점 수는 checks.lam_zero_points."]},
        "full": full, "targets": targets}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, out_path)
    md = (out_path[:-5] if out_path.endswith(".json") else out_path) + ".md"
    with open(md + ".tmp", "w", encoding="utf-8") as f:
        f.write(make_md(pack, hg))
    os.replace(md + ".tmp", md)
    print("wrote", out_path, md, f"{secs:.0f}s", flush=True)


def make_md(pack, hg):
    H = pack["header"]
    L = [f"# WP13a 침수 꼴 (경험 스플라인 + S28·S38 HGAM) — 요약\n\n{H['made']['at_kst']} · data {H['data_ver']} · git {H['made']['tool_head']} · 걸린 시간 {H['secs']}초\n",
         "HGAM이 재현 점검(h12~120 \\|ΔP\\| ≤ 0.01)을 못 넘어 계획의 대체 규칙을 썼다: 05시 시작 시즌 = WP12 경험 스플라인(앞 시즌만), S28·S38 = HGAM.\n"]
    fc = H["full_check"]
    L.append(f"## 전체 21시즌 점검 (WP12 합친 곡선 대비)\n\nmax \\|ΔP\\| h0~120 {fc['max_abs_dP_h0_120']} · h12~120 {fc['max_abs_dP_h12_120']} · (0.05칸 누적 셈) h12~120 {fc['max_abs_dP_h12_120_ce_integral']}. "
             f"κ = {pack['full']['kappa']:g}, σ_ph = {pack['full']['sigma_ph']:.4f}\n")
    L.append("## 맞힐 시즌별\n\n| 시즌 | 출처 | 학습 J | κ | ln m 사전 sd | τ pooled | τ per_day | m_true | 안정(max ΔP vs 전체) |\n|---|---|---|---|---|---|---|---|---|")
    for s, t in pack["targets"].items():
        k = t["penalties"]["kappa"]
        k = f"{k:g}" if isinstance(k, (int, float)) else "HGAM"
        L.append(f"| {s} | {t['source']} | {t['lnm_prior']['J']} | {k} | {math.sqrt(t['lnm_prior']['var']):.3f} | {t['tau']['pooled']:.3f} | "
                 f"{', '.join('%.2f' % v for v in t['tau']['per_day'])} | {t['m_true']['lnm']:.3f} | {t['stability']['max_dP_vs_full']:.4f} |")
    L.append("\n## 경험 대 HGAM — P(h) 차 (emp 시즌만)\n\n| 시즌 | P emp h12 | h24 | h48 | h96 | \\|ΔP\\| h12 | h24 | h48 | h96 |\n|---|---|---|---|---|---|---|---|---|")
    for s, t in pack["targets"].items():
        if t["source"] != "emp":
            continue
        h = np.array(t["h"]); P = np.array(t["P"])
        g = hg["targets"][s]
        Pg = np.interp([12, 24, 48, 96], g["h"], g["P"]); Pe = np.interp([12, 24, 48, 96], h, P)
        L.append(f"| {s} | " + " | ".join(f"{v:.4f}" for v in Pe) + " | " + " | ".join(f"{abs(a - b):.4f}" for a, b in zip(Pe, Pg)) + " |")
    return "\n".join(L) + "\n"


# ======================= 매끈한 꼴 (m3smooth · f2pow) =======================
# 하루 물결 없는 꼴 P(u), u = t/Ht (시즌 자기 끝 Ht 기준 — 시작 시각과 무관). 맞힐 시즌 S21~S42 22개 모두.

class CalG:
    """시즌 하나의 달력 (c0h, Ht): 0.05시간 칸, 일차 경계. W.level_fit·m_fit이 쓰는 인터페이스."""

    def __init__(self, c0h, Ht):
        self.N = int(round(Ht / DTc)); self.dt = Ht / self.N
        self.edges = W.day_edges_of(c0h, Ht)

    def at(self, Ce, t):
        u = np.clip(np.asarray(t, float) / self.dt, 0.0, self.N)
        i = np.minimum(np.floor(u).astype(int), self.N - 1)
        f = u - i
        return Ce[i] * (1 - f) + Ce[i + 1] * f


_CALS = {}


def cal_of(s):
    if s not in _CALS:
        _CALS[s] = CalG(SI[s]["c0h"], W.HTRUE[s])
    return _CALS[s]


def ce_of(Pfun, s):
    """누적 침수 Ce(칸 끝) = P(0) − P(t) (log10). Pfun(u)."""
    cal = cal_of(s)
    t = np.arange(cal.N + 1) * cal.dt
    P = Pfun(t / W.HTRUE[s])
    return P[0] - P


def P_f2pow(th, u):
    w = 1 / (1 + math.exp(-th[0]))
    t = np.clip(np.asarray(u, float), 0.25 / 120.0, 1.0)
    return -np.log10(w * t ** math.exp(th[1]) + (1 - w) * t ** math.exp(th[2]))


def P_m3(cl, u):
    """m3.P_grid(numpy 판) + surface의 보간: √u 격자 위 P (log10, 끝에서 0)."""
    B = M3.BASIS
    sg = np.asarray(B["sg"])
    f = np.exp(np.asarray(B["Blf"]) @ cl) * 2.0 * sg
    seg = 0.5 * (f[1:] + f[:-1]) * B["ds"]
    Pg = np.concatenate([np.cumsum(seg[::-1])[::-1], np.zeros(1)])
    return np.interp(np.sqrt(np.clip(np.asarray(u, float), 0.0, 1.0)), sg, Pg)


M3 = None


def m3_cl(stage1, tn):
    """맞힐 시즌 tn의 M3 사전이 쓰는 창 평균 ℓ 계수 (m3.prior와 같은 셈, PCA_BLOCK 끔)."""
    target = f"S{tn}"
    before = [s for s in sorted(stage1, key=M3.HR.snum) if M3.HR.snum(s) < M3.HR.snum(target)]
    k, _ = M3.choose_window_pca(stage1, target)
    win = before if k == "all" else before[-int(k):]
    xv = [M3._xv(stage1[s]) for s in win]
    p = M3.pca_window(np.array([x for x, _ in xv]), np.array([v for _, v in xv]))
    cl = np.linalg.lstsq(M3.BASIS["Bl12"], p["mu"][M3.NE_R:], rcond=None)[0]
    return cl, k, win


def sig_from_m(Pfuns, train, sig, rounds=3):
    """꼴 고정, 시즌별 ln m(m_fit) → 잔차 → σ·φ(FC.sig_of) 되풀이."""
    Ces = {s: ce_of(Pfuns, s) for s in train}
    for _ in range(rounds):
        res = np.zeros(OBS["n"]); sel = np.zeros(OBS["n"], bool)
        for s in train:
            rws = np.where(OBS["S"] == SIDX[s])[0]
            lm, _ = W.m_fit(OBS, rws, cal_of(s), Ces[s], sig)
            Y = cal_of(s).at(Ces[s], OBS["tb"][rws]) - cal_of(s).at(Ces[s], OBS["ta"][rws])
            res[rws] = OBS["y"][rws] - math.exp(lm) * Y; sel[rws] = True
        a, b, c3 = FC.sig_of(res[sel], OBS["isph"][sel], OBS["wp"][sel])
        sig = FC.Sg(a, b, c3)
    return sig


def fit_f2pow(train, theta0=None, rounds=3):
    """f2pow 꼴 θ 셋 + 시즌 배수 m (기하평균 1) — 가중 Huber IRLS + scipy least_squares. 돌려줌 (θ, m dict, σ)."""
    from scipy.optimize import least_squares
    rows = np.where(np.isin(OBS["S"], [SIDX[s] for s in train]))[0]
    y, isph, wp = OBS["y"][rows], OBS["isph"][rows], OBS["wp"][rows]
    sj = np.array([train.index(SEAS[i]) for i in OBS["S"][rows]])
    Ht = np.array([W.HTRUE[s] for s in train])[sj]
    ua, ub = OBS["ta"][rows] / Ht, OBS["tb"][rows] / Ht
    J = len(train)

    def mprof(th, w):
        Y = P_f2pow(th, ua) - P_f2pow(th, ub)
        num = np.bincount(sj, w * y * Y, J); den = np.bincount(sj, w * Y * Y, J)
        m = np.where(den > 0, num / np.maximum(den, 1e-300), 1.0)
        m = np.maximum(m, 1e-3)
        m = m / math.exp(np.mean(np.log(m)))
        return m, Y

    sig = FC.Sg(0.03, 1.0)
    starts = [np.array(theta0)] if theta0 is not None else [np.array(v) for v in ([1.0, math.log(0.8), math.log(10.0)], [0.0, math.log(0.5), math.log(5.0)], [2.0, math.log(1.2), math.log(20.0)])]
    th = starts[0]; ps = np.ones(len(y)); m = np.ones(J)
    lo_b = np.array([-8.0, -5.0, -5.0]); hi_b = np.array([8.0, 6.0, 6.0])
    for rd in range(rounds):
        W0 = FC.W0of(isph, wp, sig)
        for it in range(8):
            w = W0 * ps
            fun = lambda t: np.sqrt(w) * (y - mprof(t, w)[0][sj] * (P_f2pow(t, ua) - P_f2pow(t, ub)))  # noqa: E731
            cand = starts if (rd == 0 and it == 0) else [th]
            best = None
            for st in cand:
                r_ = least_squares(fun, np.clip(st, lo_b + 1e-6, hi_b - 1e-6), bounds=(lo_b, hi_b), x_scale=1.0)
                if best is None or r_.cost < best.cost:
                    best = r_
            th = best.x
            m, Y = mprof(th, w)
            r = y - m[sj] * Y
            psn = FC.psi(r, W0)
            done = np.max(np.abs(psn - ps)) < 1e-4
            ps = psn
            if done:
                break
        a, b, c3 = FC.sig_of(r, isph, wp)
        sig = FC.Sg(a, b, c3)
    return th, {s: float(m[j]) for j, s in enumerate(train)}, sig


def smooth_job(args):
    kind, tn, stage1_path = args
    t0 = time.time()
    global M3
    if kind == "m3smooth" and M3 is None:
        sys.path.insert(0, os.path.dirname(HERE))
        import m3 as M3_  # noqa: E402
        M3 = M3_
    if tn is None:                                      # f2pow 전체(S19~S42) 판
        train = list(SEAS)
        th, m, sig = fit_f2pow(train)
        return None, {"theta": th, "sig": sig, "train": train}, time.time() - t0
    train = [s for s in SEAS if int(s[1:]) < tn]
    if kind == "f2pow":
        th, m, sig = fit_f2pow(train)
        return tn, {"theta": th, "sig": sig, "train": train, "m": m}, time.time() - t0
    s1 = M3.load_stage1(stage1_path)
    cl, k, win = m3_cl(s1, tn)
    Pf = lambda u: P_m3(cl, u)  # noqa: E731
    sig = sig_from_m(Pf, train, FC.Sg(0.03, 1.0))
    return tn, {"cl": cl, "k": k, "window": win, "sig": sig, "train": train}, time.time() - t0


def lam_wave(c0h, h, lam, Ht):
    """2~4일차 같은 KST 시각(15분 칸) λ 평균의 최고/최저 비."""
    d = W.day_c0(c0h, h)
    ok = (d >= 2) & (d <= 4) & (h < Ht - 1e-9)
    c = np.mod(c0h + h[ok], 24.0)
    b = np.floor(c * 4).astype(int) % 96
    num = np.bincount(b, lam[ok], 96); cnt = np.bincount(b, minlength=96)
    mk = cnt >= 2
    v = num[mk] / cnt[mk]
    return float(v.max() / v.min()) if v.min() > 0 else None


def post_smooth(kind, tn, r, th_full):
    sig, train = r["sig"], r["train"]
    tseason = f"S{tn}"
    if kind == "f2pow":
        Pf = lambda u: P_f2pow(r["theta"], u)  # noqa: E731
    else:
        Pf = lambda u: P_m3(r["cl"], u)  # noqa: E731
    cal = cal_of(tseason); c0h, Ht = SI[tseason]["c0h"], W.HTRUE[tseason]
    Ce = ce_of(Pf, tseason)
    edges = cal.edges; Dn = len(edges) - 1
    hg = W.out_grid(Ht)
    Pbar = Pf(hg / Ht)
    lam_g = -np.gradient(Pbar, hg)
    Ad = np.zeros((Dn, len(hg)))
    for d in range(Dn):
        Ad[d] = np.where(hg >= edges[d + 1] - 1e-9, 0.0, cal.at(Ce, edges[d + 1]) - cal.at(Ce, np.maximum(hg, edges[d])))
    # 위 Ad는 Ce(=P(0)−P) 기준이라 Σ_d A_day = P(h)−P(Ht) = P(h)
    J = len(train)
    lnm = np.zeros(J); se = np.zeros(J); Cej = {}
    for j, s in enumerate(train):
        Cej[s] = ce_of(Pf, s)
        rws = np.where(OBS["S"] == SIDX[s])[0]
        lnm[j], se[j] = W.m_fit(OBS, rws, cal_of(s), Cej[s], sig)
    lnm_train = {s: [round(float(lnm[j]), 5), round(float(se[j]), 5)] for j, s in enumerate(train)}
    raw = float(np.var(lnm, ddof=1)) if J >= 2 else None
    noise = float(np.mean(se ** 2))
    var = 0.3 ** 2 if J < 2 else max(raw - noise, 0.01 ** 2)
    lnm_prior = {"mean": round(float(lnm.mean()), 6), "var": round(var, 6), "raw_var": None if raw is None else round(raw, 6),
                 "noise_var": round(noise, 6), "J": J}
    samples = {k: [] for k in range(1, 6)}; noises = {k: [] for k in range(1, 6)}
    allv, alln, dropped = [], [], 0
    for j, s in enumerate(train):
        rws = np.where(OBS["S"] == SIDX[s])[0]
        cj = cal_of(s)
        for d, (L, seln) in W.level_fit(OBS, rws, cj, Cej[s], sig, cj.edges).items():
            if seln > 1.0:
                dropped += 1; continue
            lt = math.log(L) - float(lnm[j]); k = min(d + 1, 5)
            samples[k].append(lt); noises[k].append(seln ** 2); allv.append(lt); alln.append(seln ** 2)
    pooled_raw = float(np.var(allv, ddof=1)) if len(allv) >= 2 else 0.3 ** 2
    pooled_noise = float(np.mean(alln)) if alln else 0.0
    pooled = math.sqrt(max(pooled_raw - pooled_noise, 0.01 ** 2)) if len(allv) >= 2 else 0.3
    per_day, n_l, raw_l, noise_l = [], [], [], []
    for k in range(1, 6):
        v = samples[k]; n_l.append(len(v))
        rw = float(np.var(v, ddof=1)) if len(v) >= 2 else None
        nz = float(np.mean(noises[k])) if v else None
        raw_l.append(None if rw is None else round(rw, 6)); noise_l.append(None if nz is None else round(nz, 6))
        per_day.append(round(math.sqrt(max(rw - nz, 0.01 ** 2)), 5) if len(v) >= 3 else round(pooled, 5))
    tau = {"per_day": per_day, "pooled": round(pooled, 5), "n": n_l, "raw": raw_l, "noise": noise_l,
           "pooled_raw": round(pooled_raw, 6), "pooled_noise": round(pooled_noise, 6), "dropped_se_gt_1": dropped,
           "note": "raw·noise는 분산 (ln t_{j,d}의 시즌 사이 표본 분산 · 델타법 se² 평균), per_day·pooled는 sd. 일차 경계는 그 시즌 달력."}
    rws = np.where(OBS["S"] == SIDX[tseason])[0]
    lmt, lmt_se = W.m_fit(OBS, rws, cal, Ce, sig)
    t_true = [None] * Dn
    for d, (L, _) in W.level_fit(OBS, rws, cal, Ce, sig, edges).items():
        t_true[d] = round(math.log(L) - lmt, 5)
    if kind == "f2pow":
        Pfull = P_f2pow(th_full, hg / Ht)
        mk = (hg >= 12) & (hg <= Ht)
        stab = {"max_dP_vs_full": round(float(np.abs(Pbar - Pfull)[mk].max()), 5), "max_dlnP_shape_vs_full": None,
                "note": "S19~S42 전부로 맞춘 판과의 차 (h 12~Ht)"}
    else:
        stab = None
    chk = {"A_day_sum_max_err": float(np.abs(Ad.sum(axis=0) - Pbar).max()), "P_monotone": bool(np.all(np.diff(Pbar) <= 1e-12)),
           "lam_nonneg": bool(np.all(lam_g >= -1e-12)), "lam_zero_points": int((lam_g <= 1e-12).sum()),
           "mean_lnm_train": float(lnm.mean()), "P_last": float(Pbar[-1]),
           "day_edges_check": [int(W.day_c0(c0h, 0.5 * (edges[d] + edges[d + 1]))) for d in range(Dn)],
           "wave_ratio_days2_4": lam_wave(c0h, hg, lam_g, Ht)}
    if kind == "f2pow":
        pen = {"theta": [round(float(v), 5) for v in r["theta"]], "w": round(1 / (1 + math.exp(-r["theta"][0])), 5),
               "p": round(math.exp(r["theta"][1]), 5), "q": round(math.exp(r["theta"][2]), 5),
               "sigma": {"sigma_ph": float(sig), "phi_pair": float(sig.phi), "sigma_ph_day": [round(float(v), 5) for v in sig.sd]}}
        cv = None
    else:
        pen = {"window_k": r["k"], "window": r["window"], "sigma": {"sigma_ph": float(sig), "phi_pair": float(sig.phi), "sigma_ph_day": [round(float(v), 5) for v in sig.sd]}}
        cv = None
    R = lambda a: [round(float(v), 6) for v in a]  # noqa: E731
    return {"train": train, "H": float(SI[tseason]["H"]), "Ht": float(Ht), "c0h": float(c0h), "h": R(hg), "P": R(Pbar), "lam": R(lam_g),
            "day_edges": [round(e, 4) for e in edges], "A_day": [R(a) for a in Ad], "lnm_prior": lnm_prior, "lnm_train": lnm_train,
            "tau": tau, "m_true": {"lnm": round(lmt, 5), "se": round(lmt_se, 5)}, "t_true": t_true, "stability": stab,
            "penalties": pen, "cv": cv, "checks": chk, "source": kind}


def main_smooth(a):
    t_all = time.time()
    kind = a.kind
    rel = lambda p: p if os.path.isabs(p) else os.path.join(CM.ROOT, "data/solo-raid/analysis", p)  # noqa: E731
    out_path, ref_path, s1_path = rel(a.out), rel(a.ref), rel(a.stage1)

    def log(m):
        print(f"[{time.time() - t_all:7.1f}s] {m}", flush=True)

    tns = list(range(21, 43))
    jobs = [(kind, n, s1_path) for n in sorted(tns, reverse=True)]
    if kind == "f2pow":
        jobs = [(kind, None, s1_path)] + jobs
    global OBS
    OBS = W.load_obs(a.obs_dir if hasattr(a, "obs_dir") else a.obs_dir)
    res = {}
    with mp.Pool(max(1, min(a.procs, 2)), initializer=init_worker, initargs=(a.obs_dir,)) as pool:
        for tn, r, secs in pool.imap_unordered(smooth_job, jobs):
            res[tn] = r
            extra = f"θ={np.round(r['theta'], 3).tolist()}" if kind == "f2pow" else f"창 k={r['k']}"
            log(f"  {'FULL' if tn is None else 'S%d' % tn}: {extra} σ={float(r['sig']):.4f} ({secs:.0f}s)")
    th_full = res[None]["theta"] if kind == "f2pow" else None
    if kind == "m3smooth":
        sys.path.insert(0, os.path.dirname(HERE))
        import m3 as M3_
        globals()["M3"] = M3_
        M3_.load_stage1(s1_path)                       # 1단계 파일의 기저로 맞춤 (워커와 같게)
    targets = {f"S{n}": post_smooth(kind, n, res[n], th_full) for n in tns}
    ref = json.load(open(ref_path, encoding="utf-8"))
    s1 = json.load(open(s1_path, encoding="utf-8"))
    s1seas = sorted((s1["seasons"] if "seasons" in s1 else s1).keys(), key=lambda s: int(s[1:]))
    secs = time.time() - t_all
    cmd = f"python3 -I tools/solo_raid/lab7/wp13_flood_emp.py --kind {kind} --obs-dir W/obs7/full --procs {a.procs}"
    full = None
    if kind == "f2pow":
        th = res[None]["theta"]
        full = {"seasons": res[None]["train"], "theta": [round(float(v), 5) for v in th], "sigma_ph": float(res[None]["sig"])}
    rules = {
        "kind": ("M3 사전의 창 평균 ℓ: 맞힐 시즌 앞 시즌만, 창 k = m3.choose_window_pca, pca_window(PCA_BLOCK 끔)의 평균 μ의 ℓ 쪽 → lstsq(Bl12) → P_grid(√u 격자) → P(u)=interp(√u). 하루 물결 없음."
                 if kind == "m3smooth" else
                 "F f2pow: P(u) = −log10(w·u^p + (1−w)·u^q), w=sigmoid(θ0), p=e^θ1, q=e^θ2, u=t/Ht를 [0.25/120,1]로 자름 (wp12_tables.F_P와 같은 식). 맞힐 시즌 앞 시즌 전부(S19~, 시작 시각 무관)로 θ 셋+시즌 배수 m(기하평균 1) 적합: "
                 "쌍 ΔL=m_j[P(t1/Ht_j)−P(t2/Ht_j)], 사진 P_obs=m_j P(h/Ht_j), 가중 Huber(1.345)+σ_ph·φ 되풀이, scipy least_squares, 시작점 3개."),
        "u_axis": "u = h/Ht (Ht = 시즌 자기 사진 적분 끝 = htrue). P(Ht)=0.",
        "post": "emp와 같은 후처리 (lnm_prior·tau·m_true·t_true·A_day). 일차 경계는 그 시즌 달력(W.day_edges_of). lam = −dP/dh 수치 미분(0.25시간 격자). 꼴 고정 σ는 m3smooth는 m_fit 잔차 되풀이(3회), f2pow는 적합의 σ.",
        "stability": "m3smooth는 기준 없어 null; f2pow는 S19~S42 전부로 맞춘 판과 max ΔP (h 12~Ht)."}
    pack = {"header": {"about": f"WP13b 꼴 시험용 매끈한 침수 꼴 ({kind}) — 22개 맞힐 시즌 S21~S42", "data_ver": CM.DATA_VER, "format": 1,
                       "made": {"tool_head": git_head(), "cmd": cmd, "at_kst": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")},
                       "secs": round(secs, 1), "rules": rules,
                       "inputs": {"stage1_seasons": f"{s1seas[0]}~{s1seas[-1]} ({len(s1seas)}시즌)" if kind == "m3smooth" else None,
                                  "ref": os.path.relpath(ref_path, CM.ROOT)}},
            "full": full, "targets": targets}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(out_path + ".tmp", out_path)
    md = (out_path[:-5] if out_path.endswith(".json") else out_path) + ".md"
    with open(md + ".tmp", "w", encoding="utf-8") as f:
        f.write(make_md_smooth(pack, ref))
    os.replace(md + ".tmp", md)
    print("wrote", out_path, md, f"{secs:.0f}s", flush=True)


def make_md_smooth(pack, ref):
    H = pack["header"]; kind = H["about"].split("(")[1].split(")")[0]
    L = [f"# WP13b 매끈한 침수 꼴 ({kind}) — 요약\n\n{H['made']['at_kst']} · data {H['data_ver']} · git {H['made']['tool_head']} · 걸린 시간 {H['secs']}초\n",
         f"입력: 1단계 시즌 {H['inputs']['stage1_seasons']}\n" if H["inputs"]["stage1_seasons"] else "",
         "## 맞힐 시즌별\n\n| 시즌 | 학습 J | 꼴 | ln m 사전 sd | τ pooled | m_true | 물결비 2~4일차 | 안정 |\n|---|---|---|---|---|---|---|---|"]
    for s, t in pack["targets"].items():
        pn = t["penalties"]
        sh = f"θ={pn['theta']}" if "theta" in pn else f"창 k={pn['window_k']}"
        stb = "-" if t["stability"] is None else t["stability"]["max_dP_vs_full"]
        wv = t["checks"]["wave_ratio_days2_4"]
        L.append(f"| {s} | {t['lnm_prior']['J']} | {sh} | {math.sqrt(t['lnm_prior']['var']):.3f} | {t['tau']['pooled']:.3f} | {t['m_true']['lnm']:.3f} | {wv:.3f} | {stb} |")
    L.append("\n## 견줄 꼴(emp 또는 hgam, 파일의 source)과 P의 |ΔP|\n\n| 시즌 | 견줄 꼴 | h12 | h24 | h48 | h96 | H−2 | 견줄 꼴 물결비 |\n|---|---|---|---|---|---|---|---|")
    for s, t in pack["targets"].items():
        g = ref["targets"][s]; Ht = t["Ht"]
        hs = [12, 24, 48, 96, Ht - 2]
        d = [abs(float(np.interp(h, t["h"], t["P"])) - float(np.interp(h, g["h"], g["P"]))) for h in hs]
        wr = lam_wave(t["c0h"], np.array(g["h"]), np.array(g["lam"]), g["Ht"])
        L.append(f"| {s} | {g.get('source', '?')} | " + " | ".join(f"{v:.4f}" for v in d) + f" | {wr:.2f} |" if wr else f"| {s} | {g.get('source', '?')} | " + " | ".join(f"{v:.4f}" for v in d) + " | - |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
