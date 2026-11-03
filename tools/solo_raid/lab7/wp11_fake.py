"""WP11a 가짜 자료 되찾기 (2026-11-03) — 계단형 침수(활동 시간 g)로 뽑은 %를 h·act 두 좌표로 맞춰, 컷과 남은 침수를 되찾는지 본다.

  python3 -I tools/solo_raid/lab7/wp11_fake.py --obs-dir W/obs7/full --season S30 --reps 20 --out data/solo-raid/analysis/r7/wp11/fake_S30_D1.json [--jobs 4]

참 곡면: 그 시즌 실제 전체 표를 act 좌표(u* = A(h)/A(H), 그 시즌 g_s)로 1단계 MAP — 실제 자료에서 나온 계단형 침수 곡면.
뽑기 (m3.selftest_recovery와 같은 꼴): 고른 줄 배치(점수·종류·시각 정보·서버 라벨)를 그대로 두고, 노드 하나를 무게대로 뽑아 촬영 시각 →
  u* → 참 곡면 + 정규 잡음 sqrt(σ0² + (ρP)²) → 0.01 단위 반올림. 서버는 라벨(미상은 KR), 오염 없음.
맞추기: 같은 뽑은 표를 clock h와 act로 1단계 라플라스 (m3.laplace_fit, 약한 사전).
보는 것: 3%·10% 최종 컷 오차 (exp(중앙 − 참) − 1), 3% 점의 남은 침수 P(T) (log10, T = H−36 … H−6과 시각) 오차, 90% 구간 포함.
정답 표·다른 시즌은 안 쓴다. 공개 규칙: 글 번호·원문·닉네임·원글 주소를 쓰지 않는다.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

T_BACK = (36, 30, 24, 18, 12, 6)


def setup():
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type=")
    import obs_model as OM
    OM.setup_host(1)


def truth_theta(a):
    """참 곡면: act 좌표 1단계 MAP (실제 자료)."""
    setup()
    import m3
    import obs_model as OM
    import obs_table
    OM.set_clock("act", a.profile)
    err = OM.load_err()
    head, rows = obs_table.read_cell(os.path.join(a.obs_dir, f"{a.season}.jsonl.gz"))
    dk = a.season if a.season in err["1a_post_delay"]["expanding"] else None
    obs = OM.to_jax(OM.prepare(rows, head["H"], a.season, err, K=16, delay_key=dk, stage1=True))
    prob = m3.Stage1Prob()
    pre = m3.multistart_map(obs, prob)
    return np.asarray(pre["loc"], float)[:prob.n].tolist()


def P3(m3, th, u):
    """3% 점(w = 0)의 KR 침수 P(u) (log10). x항은 w_fin = 0이라 1."""
    import jax.numpy as jnp
    Pg = m3.P_grid(th["cl"])
    return float(jnp.interp(jnp.sqrt(jnp.asarray(u, jnp.float32)), jnp.asarray(m3.BASIS["sg"]), Pg))


def one(job):
    a, vt, rep = job
    setup()
    import jax.numpy as jnp
    import m3
    import obs_model as OM
    import obs_table
    import act_clock as AC
    prof = AC.load(a.profile)
    err = OM.load_err()
    head, rows = obs_table.read_cell(os.path.join(a.obs_dir, f"{a.season}.jsonl.gz"))
    H = float(head["H"])
    dk = a.season if a.season in err["1a_post_delay"]["expanding"] else None
    prob = m3.Stage1Prob()
    th = prob.theta(jnp.asarray(vt, jnp.float32))
    s0, rho = math.exp(vt[prob.it + 2]), math.exp(vt[prob.it + 3])
    rng = np.random.default_rng(1000 + rep)
    new = []
    for r in OM.select_rows(rows, H, stage1=True):
        t, w, why = OM.nodes_detail(r, H, err, dk, 16)
        if t is None:
            continue
        tt = float(rng.choice(t, p=w / w.sum()))
        k = OM.SERVERS.index(r["server"]) if r["server"] in OM.SERVERS else 0
        u = float(AC.ustar(prof, a.season, [tt], H)[0])
        mu, P = m3.surface(th, jnp.asarray([math.log(r["score"])], jnp.float32), jnp.asarray([[u]], jnp.float32))
        z = float(mu[k, 0, 0]) + math.sqrt(s0 ** 2 + (rho * float(P[k, 0, 0])) ** 2) * rng.standard_normal()
        p = round(10 ** z, 2)
        if not (0.01 <= p <= 100.0):
            continue
        x = dict(r)
        x["pct"] = p
        if k == 0 and r["server"] is None:
            x["server"] = "KR"
        new.append(x)
    Ts = [H - b for b in T_BACK]
    truth = {"lnC3": float(m3.lnC(th, 3.0)), "lnC10": float(m3.lnC(th, 10.0)),
             "P3": [P3(m3, th, float(AC.ustar(prof, a.season, [T], H)[0])) for T in Ts]}
    out = {"rep": rep, "rows": len(new), "truth": truth}
    for clock in ("h", "act"):
        OM.set_clock(clock, a.profile)
        sm, info, dg, obs = m3.laplace_fit(new, H, a.season, err, m3.Stage1Prob(), dk, ndraw=2000, seed=rep)
        v = sm["v"].reshape(-1, sm["v"].shape[-1])
        ufun = (lambda T: min(max(T / H, 0.02), 1.0)) if clock == "h" else (lambda T: float(AC.ustar(prof, a.season, [T], H)[0]))
        loc = jnp.asarray(info["pre"]["loc"][:prob.n], jnp.float32)
        thm = m3.Stage1Prob().theta(loc)
        Pd = []
        for j in range(0, len(v), 20):          # 100뽑기로 P 구간
            thj = m3.Stage1Prob().theta(jnp.asarray(v[j], jnp.float32))
            Pd.append([P3(m3, thj, ufun(T)) for T in Ts])
        Pd = np.array(Pd)
        res = {"conv": dg["conv"], "secs": info["secs"]}
        for q in (3, 10):
            d = sm[f"lnC_{q}"].ravel()
            lo, med, hi = np.quantile(d, [0.05, 0.5, 0.95])
            tv = truth[f"lnC{q}"]
            res[f"C{q}"] = {"err": float(math.exp(med - tv) - 1), "in90": bool(lo <= tv <= hi)}
        res["P3_map"] = [P3(m3, thm, ufun(T)) for T in Ts]
        lo, hi = np.quantile(Pd, [0.05, 0.95], axis=0)
        res["P3_err"] = (np.array(res["P3_map"]) - np.array(truth["P3"])).tolist()
        res["P3_in90"] = [bool(a_ <= t_ <= b_) for a_, t_, b_ in zip(lo, truth["P3"], hi)]
        out[clock] = res
    print(rep, out["rows"], {c: (round(out[c]["C3"]["err"] * 100, 2), round(out[c]["C10"]["err"] * 100, 2), round(out[c]["P3_err"][4], 4)) for c in ("h", "act")}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--season", default="S30")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--profile", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "data/solo-raid/analysis/r7/wp11/act_profile_D1.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as p:
        vt = p.apply(truth_theta, (a,))
    with ctx.Pool(a.jobs) as p:
        reps = p.map(one, [(a, vt, r) for r in range(a.reps)], chunksize=1)
    sm = {}
    for c in ("h", "act"):
        R = [x[c] for x in reps]
        sm[c] = {"C3_mean_err": float(np.mean([x["C3"]["err"] for x in R])), "C3_mae": float(np.mean([abs(x["C3"]["err"]) for x in R])),
                 "C10_mean_err": float(np.mean([x["C10"]["err"] for x in R])), "C10_mae": float(np.mean([abs(x["C10"]["err"]) for x in R])),
                 "C3_in90": float(np.mean([x["C3"]["in90"] for x in R])), "C10_in90": float(np.mean([x["C10"]["in90"] for x in R])),
                 "P3_mean_err_by_back": dict(zip([f"H-{b}" for b in T_BACK], np.mean([x["P3_err"] for x in R], 0).round(4).tolist())),
                 "P3_in90_by_back": dict(zip([f"H-{b}" for b in T_BACK], np.mean([x["P3_in90"] for x in R], 0).round(2).tolist())),
                 "conv": int(sum(x["conv"] for x in R))}
    H = reps[0]["truth"]
    with open(a.profile, encoding="utf-8") as f:
        prof = json.load(f)
    c0h = prof["g"][a.season]["c0h"]
    Hh = prof["g"][a.season]["H"]
    out = {"about": "WP11a 가짜 자료 되찾기 — 참 곡면 = act 1단계 MAP (실제 자료), 뽑은 %를 h·act로 1단계 라플라스", "season": a.season, "reps": a.reps,
           "truth_theta": vt, "truth_P3_by_back": dict(zip([f"H-{b}" for b in T_BACK], np.round(H["P3"], 4).tolist())),
           "kst_by_back": {f"H-{b}": f"{(c0h + Hh - b) % 24:04.1f}" for b in T_BACK},
           "summary": sm, "reps_detail": reps, "secs": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(json.dumps(sm, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
