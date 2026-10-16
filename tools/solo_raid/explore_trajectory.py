"""솔로레이드 2단계 2.6 탐색: 개인 궤적 — 같은 정확 점수로 이어진 반복 관측에서 %만 밀리는 모양 (측정 설계용, 모델 아님, 요청 0).

python3 -I tools/solo_raid/explore_trajectory.py STATE [--out DIR] [--grades A|all] [--interp lin|pchip|iso|linraw]

입력 (읽기만 한다. state/에 쓰지 않는다)
- STATE/season.json · core.jsonl · truth_points.jsonl
- 곡선·시점 묶기는 explore_pct_transform.py(같은 폴더)를 불러 쓴다

무엇을 재나
1) 궤적 표본: 같은 사람(person, 정확 점수 지문 P…)의 시점(같은 %는 한 시점)이 둘 이상인 사람. 종료 후 최종 점(truth_points A·A2·D)이 있으면 붙인다.
   최종 점이 있으면 서버 미상이어도 y(최종 %)를 직접 안다 (곡선 보간이 필요 없다).
2) 순서를 아는 이웃 쌍(앞 줄 h_hi ≤ 뒤 줄 h_lo; C 줄은 h_lo = 0이라 앞 줄로만 쓸 수 있다)에서 % 변화의 방향과 크기.
3) 시간 모양: A·B 시각 쌍의 r = Δln x / Δh(시간당)와 e = Δln x / Δln h, 사람 셋 이상 점의 ln x ~ h · ln h 적합 비교,
   05:00 갱신 경계(h = 24의 배수)를 넘는 쌍과 안 넘는 쌍의 r.
4) 2.5와 대조: 최종 y를 아는 궤적 점의 P = log10(y/x)를, 그 사람을 뺀 같은 서버 단면 자료의 같은 일차 P 중앙값과 견준다.
   서버 미상 궤적(D 최종)은 단면 자료(서버별)와 견줄 수 없어 따로 적는다.
5) 점수 갱신: 같은 글의 본인 줄에 서로 다른 정확 점수가 시각 순서와 함께 나온 경우 (지문은 같은 점수만 잇기 때문에
   다른 글 사이의 갱신은 이 자료로 찾을 수 없다 — 갱신 빈도의 하한도 아니고 「찾을 수 있었던 것」의 수다).
"""
import argparse
import collections
import importlib.util
import json
import math
import os
import statistics

_spec = importlib.util.spec_from_file_location(
    "explore_pct_transform", os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_pct_transform.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


import importlib.util as _ilu  # noqa: E402
_sp = _ilu.spec_from_file_location("split", os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.py"))
split = _ilu.module_from_spec(_sp)
_sp.loader.exec_module(split)

def final_by_oi(state):
    """oi → (서버, 등급, 최종 %, 점수). 등급 우선 A > A2 > D."""
    rank = {"A": 0, "A2": 1, "D": 2}
    out = {}
    for d in pt.jl(os.path.join(state, "truth_points.jsonl")):
        if d["grade"] not in rank or "best_record" in d["flags"]:
            continue
        for oi in d["src"]:
            if oi not in out or rank[d["grade"]] < rank[out[oi]["grade"]]:
                out[oi] = {"server": d["server"], "grade": d["grade"], "pct": d["pct"], "score": d["score_raw"]}
    return out


def ordered(a, b):
    return a["h_hi"] is not None and b["h_lo"] is not None and a["h_hi"] <= b["h_lo"] + 1e-9


def ols(xs, ys):
    r = pt.ols(xs, ys)
    if not r:
        return None
    a, b, _ = r
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    my = sum(ys) / len(ys)
    sst = sum((y - my) ** 2 for y in ys)
    return b, (1 - sum(e * e for e in res) / sst) if sst > 0 else None, max(abs(e) for e in res)


def cross_curve(cross, srv, person, h, H):
    """단면 P̄(h): 그 사람을 뺀 같은 서버 점의 일차별 중앙값(사람 묶음)을 일차 가운데 시각(12·36·…)에 두고,
    끝(h = H)에서 0으로 잇는 꺾은선. 일차 점이 3명 미만이면 그 일차는 건너뛴다."""
    knots = []
    for d in range(1, 6):
        pp = pt.collapse([o for o in cross if o["server"] == srv and o["person"] != person and o["P"] is not None
                          and pt.day_of(o["h"]) == d], "P")
        if len(pp) >= 3:
            knots.append((statistics.median(q["h"] for q in pp), statistics.median(q["v"] for q in pp)))
    if not knots:
        return None
    knots.append((H, 0.0))
    if h <= knots[0][0]:
        return knots[0][1]
    for (h0, v0), (h1, v1) in zip(knots, knots[1:]):
        if h0 <= h <= h1:
            return v0 + (v1 - v0) * (h - h0) / (h1 - h0)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out")
    ap.add_argument("--grades", default="all", choices=["A", "all"])
    ap.add_argument("--interp", default="lin", choices=["lin", "pchip", "iso", "linraw"])
    split.add_final_arg(ap)
    a = ap.parse_args()
    split.guard_args(a)   # 시험 시즌 정답 감시 (할 일 3.0b)
    season, H = pt.season_hours(a.state)
    grades = ("A",) if a.grades == "A" else ("A", "A2")
    rows = pt.jl(os.path.join(a.state, "core.jsonl"))
    use = [r for r in rows if r["use"] and r["src"] == "photo" and r["pct"] is not None]
    mom = pt.dedupe_moments(use)
    fin = final_by_oi(a.state)
    # 사람별 서버 (같은 지문 사람 줄 중 서버가 하나면 그것)
    psrv = collections.defaultdict(set)
    for r in use:
        if r["server"]:
            psrv[r["person"]].add(r["server"])
    persons = collections.defaultdict(list)
    for r in mom:
        persons[r["person"]].append(r)
    traj = {}
    for p, L in persons.items():
        if not p.startswith("P"):
            continue
        pre = [r for r in L if r["day_lo"] != "post" and r["h"] is not None and r["h"] < H]
        post = [r for r in L if r not in pre]
        f = None
        for r in L:
            if r["oi"] in fin:
                c = fin[r["oi"]]
                if f is None or {"A": 0, "A2": 1, "D": 2}[c["grade"]] < {"A": 0, "A2": 1, "D": 2}[f["grade"]]:
                    f = c
        if f is None and post:
            # 최종 점에 안 들어간 종료 후 줄 (점수 없음 등) — 참고만
            pass
        srv = next(iter(psrv[p])) if len(psrv[p]) == 1 else (None if not psrv[p] else "conflict")
        pre.sort(key=lambda r: (r["h"], r["oi"]))
        if len(pre) + (1 if f else 0) >= 2 and len(pre) >= 1:
            traj[p] = {"pre": pre, "final": f, "server": srv, "score": pre[0]["score_raw"] or pre[0]["score"]}
    print(f"시즌 {season['name']} · H={H:.2f}시간 · 곡선 {grades}·{a.interp}")
    multi = {p: t for p, t in traj.items() if len(t["pre"]) >= 2}
    withf = {p: t for p, t in traj.items() if t["final"]}
    print(f"\n[표본] 지문 사람 중 종료 전 시점 2개 이상 {len(multi)}명 · 종료 전 1개 이상 + 최종 점 {len(withf)}명 · "
          f"둘 다 {sum(1 for p in multi if p in withf)}명")
    c = collections.Counter((t["server"] or "미상", len(t["pre"]) >= 2, bool(t["final"])) for t in traj.values())
    print("  (서버, 종료 전 2개 이상, 최종 점 있음) →", dict(sorted(c.items(), key=str)))
    g = collections.Counter(tuple(sorted(collections.Counter(r["t_grade"] for r in t["pre"]).items())) for t in multi.values())
    print("  종료 전 2개 이상의 시각 등급 구성:", dict(g.most_common(8)))
    print("  종료 전 시점 수 분포:", dict(sorted(collections.Counter(len(t["pre"]) for t in traj.values()).items())))

    # 2) 순서를 아는 쌍
    pairs = []
    for p, t in traj.items():
        seq = list(t["pre"])
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                if ordered(seq[i], seq[j]):
                    pairs.append((p, seq[i], seq[j]))
                    break   # 바로 다음 순서 아는 점 하나만
    up = sum(1 for _, x, y in pairs if y["pct"] > x["pct"] + 0.005)
    dn = sum(1 for _, x, y in pairs if y["pct"] < x["pct"] - 0.005)
    print(f"\n[방향] 순서를 아는 종료 전 이웃 쌍 {len(pairs)}개 (사람 {len({p for p, _, _ in pairs})}): % 커짐 {up} · 같음 {len(pairs) - up - dn} · 작아짐 {dn}")
    for p, x, y in pairs:
        if y["pct"] < x["pct"] - 0.005:
            print(f"   작아짐: {p} {x['oi']}({x['t_grade']} h{x['h']:.1f} {x['pct']}%) → {y['oi']}({y['t_grade']} h{y['h']:.1f} {y['pct']}%) 점수 {x['score']}")
    fd = [(p, t["pre"][-1], t["final"]) for p, t in traj.items() if t["final"]]
    upf = sum(1 for _, r, f in fd if f["pct"] > r["pct"] + 0.005)
    dnf = sum(1 for _, r, f in fd if f["pct"] < r["pct"] - 0.005)
    print(f"  마지막 종료 전 점 → 최종: {len(fd)}명, 커짐 {upf} · 같음 {len(fd) - upf - dnf} · 작아짐 {dnf}")
    for p, r, f in fd:
        if f["pct"] < r["pct"] - 0.005:
            print(f"   작아짐: {p} {r['oi']}({r['t_grade']} h{r['h']:.1f} {r['pct']}%) → 최종 {f['pct']}% ({f['grade']})")

    # 3) 시간 모양 (A·B 시각만)
    ab = [(p, x, y) for p, x, y in pairs if x["t_grade"] != "C" and y["t_grade"] != "C" and y["h"] - x["h"] > 0.25]
    print(f"\n[시간 모양] A·B 시각 이웃 쌍 {len(ab)}개 (간격 15분 넘는 것, 사람 {len({p for p, _, _ in ab})})")
    print("  중간 h 구간 | 쌍 | r = Δln x/시간 중앙 [범위] | e = Δln x/Δln h 중앙 [범위] | 05:00 넘는 쌍")
    for lo, hi in ((0, 24), (24, 48), (48, 72), (72, 96), (96, 120)):
        q = [(x, y) for _, x, y in ab if lo <= (x["h"] + y["h"]) / 2 < hi]
        if not q:
            continue
        r = [math.log(y["pct"] / x["pct"]) / (y["h"] - x["h"]) for x, y in q]
        e = [math.log(y["pct"] / x["pct"]) / math.log(y["h"] / x["h"]) for x, y in q]
        cr = sum(1 for x, y in q if int(y["h"] // 24) > int(x["h"] // 24))
        print(f"  {lo}-{hi} | {len(q)} | {statistics.median(r):.4f} [{min(r):.4f},{max(r):.4f}] | "
              f"{statistics.median(e):.2f} [{min(e):.2f},{max(e):.2f}] | {cr}")
    # 05:00 경계: 같은 일차 안 vs 넘음 — 같은 중간 일차끼리
    print("  05:00 경계: 같은 일차 안 쌍 r 중앙 / 넘는 쌍 r 중앙 (중간 h 일차별)")
    for d in range(1, 6):
        inn = [math.log(y["pct"] / x["pct"]) / (y["h"] - x["h"]) for _, x, y in ab
               if int(x["h"] // 24) == int(y["h"] // 24) == d - 1]
        cr = [math.log(y["pct"] / x["pct"]) / (y["h"] - x["h"]) for _, x, y in ab
              if int(x["h"] // 24) != int(y["h"] // 24) and int(((x["h"] + y["h"]) / 2) // 24) == d - 1]
        if inn or cr:
            print(f"   {d}일차: 안 {len(inn)}쌍 {pt.fmt(statistics.median(inn) if inn else None, 4)} · "
                  f"넘음 {len(cr)}쌍 {pt.fmt(statistics.median(cr) if cr else None, 4)}")
    # 점 셋 이상 (A·B) — 최종 점(h = H)을 넣은 것과 뺀 것
    print("  사람별 적합 (A·B 시각 점 3개 이상): ln x ~ h 대 ln x ~ ln h, R²")
    fits = []
    for p, t in traj.items():
        q = [r for r in t["pre"] if r["t_grade"] != "C"]
        if len(q) >= 3 and q[-1]["h"] - q[0]["h"] > 2:
            hs = [r["h"] for r in q]
            lx = [math.log(r["pct"]) for r in q]
            f1, f2 = ols(hs, lx), ols([math.log(h) for h in hs], lx)
            fits.append((p, len(q), f1, f2, t))
    for p, n, f1, f2, t in fits:
        fs = f" 최종 {t['final']['pct']}%({t['final']['grade']})" if t["final"] else ""
        print(f"   {p} {t['server'] or '미상'} n={n} h {t['pre'][0]['h']:.1f}~{t['pre'][-1]['h']:.1f}: "
              f"선형 h R²={pt.fmt(f1[1], 3)} · ln h R²={pt.fmt(f2[1], 3)} (e={f2[0]:.2f}) · "
              f"% {[r['pct'] for r in t['pre'] if r['t_grade'] != 'C']}{fs}")

    # 4) 2.5 단면과 대조
    curves = {srv: pt.Curve(pt.final_points(a.state, srv, grades), a.interp) for srv in ("KR", "JP", "GL")}
    cross = pt.attach(pt.load_obs(a.state, H, with_c=True, with_best=False), curves)
    print("\n[2.5 대조] 궤적 점의 P = log10(y/x) − 단면 P̄(h) (그 사람 뺀 같은 서버 일차 중앙값을 잇은 꺾은선, 끝에서 0). y는 최종 점 직접값 > 곡선 보간")
    tab = collections.defaultdict(list)
    out_pts = []
    for p, t in traj.items():
        srv = t["server"]
        for r in t["pre"]:
            y, how = None, None
            if t["final"]:
                y, how = t["final"]["pct"], "direct"
            elif srv in curves:
                y = curves[srv].pct_at(t["score"])
                how = "interp" if y else None
            if not y:
                continue
            P = math.log10(y / r["pct"])
            d = pt.day_of(r["h"])
            ref = [o["P"] for o in cross if o["server"] == srv and o["person"] != p and o["P"] is not None
                   and pt.day_of(o["h"]) == d]
            pbar = cross_curve(cross, srv, p, r["h"], H)
            dev = P - pbar if pbar is not None and len(ref) >= 3 else None
            out_pts.append({"person": p, "server": srv, "h": r["h"], "grade": r["t_grade"], "x": r["pct"], "y": y,
                            "how": how, "P": P, "day": d, "ref_n": len(ref), "dev": dev, "multi": len(t["pre"]) >= 2})
            tab[(srv or "미상", d)].append((P, dev, len(t["pre"]) >= 2, how))
    print("  서버 | 일차 | 점 | P 중앙(배수) | 단면과 차 중앙 [최소,최대] (차를 잴 수 있는 점 수)")
    for (srv, d), L in sorted(tab.items(), key=str):
        P = [v for v, _, _, _ in L]
        dv = [v for _, v, _, _ in L if v is not None]
        dvs = f"{statistics.median(dv):+.3f} [{min(dv):+.3f},{max(dv):+.3f}] ({len(dv)})" if dv else "— (단면 없음)"
        print(f"  {srv} | {d} | {len(L)} | {statistics.median(P):+.3f} ({10 ** statistics.median(P):.2f}x) | {dvs}")
    # 같은 사람 안 P 감소 대 단면 감소 (두 시점 다 차를 잴 수 있는 사람)
    byp = collections.defaultdict(list)
    for o in out_pts:
        byp[o["person"]].append(o)
    within = []
    for p, L in byp.items():
        L = [o for o in L if o["dev"] is not None]
        if len(L) >= 2:
            L.sort(key=lambda o: o["h"])
            within.append((p, L[0], L[-1]))
    if within:
        print(f"  같은 사람 두 시점(첫·끝) 다 단면 기준이 있는 사람 {len(within)}명: 사람 안 ΔP, 단면 ΔP(일차 중앙값 차)")
        dd = []
        for p, o1, o2 in within:
            d_in = o2["P"] - o1["P"]
            d_cr = (o2["P"] - o2["dev"]) - (o1["P"] - o1["dev"])
            dd.append((d_in, d_cr))
            print(f"   {p} {o1['server']} h{o1['h']:.1f}({o1['grade']})→h{o2['h']:.1f}({o2['grade']}) x {o1['x']}→{o2['x']} "
                  f"y {o1['y']:.2f}({o1['how']}): 사람 안 {d_in:+.3f} · 단면 {d_cr:+.3f}")

        if within:
            r = [a_ / b_ for a_, b_ in dd if abs(b_) > 0.02]
            print(f"  요약: 사람 안 ΔP / 단면 ΔP 중앙 {statistics.median(r):.2f} [{min(r):.2f},{max(r):.2f}] (단면 ΔP 0.02 넘는 {len(r)}명), "
                  f"|차| 중앙 {statistics.median(abs(a_ - b_) for a_, b_ in dd):.3f}, 최대 {max(abs(a_ - b_) for a_, b_ in dd):.3f}")

    # 5) 점수 갱신
    print("\n[점수 갱신] 같은 글의 본인 줄에 서로 다른 정확 점수 (use 줄, 시각 순)")
    post = collections.defaultdict(list)
    for r in use:
        if r["self"] and r["score_raw"]:
            post[(r["site"], r["no"])].append(r)
    nmulti = 0
    for k, L in sorted(post.items(), key=str):
        sc = {}
        for r in L:
            sc.setdefault(round(r["score_raw"], 4), r)
        if len(sc) < 2:
            continue
        nmulti += 1
        S = sorted(sc.values(), key=lambda r: (r["h"], r["score_raw"]))
        same_t = len({r["h"] for r in S}) == 1
        desc = " → ".join(f"{r['score_raw']:.1f}억({r['pct']}%, {r['server'] or '미상'}, {r['t_grade']} h{r['h']:.1f}, {r['person']})" for r in S)
        print(f"  {k[0]} {k[1]}{' [같은 시각 — 순서·동일인 모름]' if same_t else ''}: {desc}")
    print(f"  본인 점수 줄이 있는 글 {len(post)}개 중 서로 다른 정확 점수가 둘 이상인 글 {nmulti}개")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "trajectory_points.jsonl"), "w", encoding="utf-8") as f:
            for o in out_pts:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        svg_traj(os.path.join(a.out, "trajectories.svg"), traj, H)
        print(f"\n점·그림: {a.out}")


def svg_traj(path, traj, H):
    W, Hh, L, R, T, B = 820, 480, 60, 150, 30, 50
    y0, y1 = math.log10(0.15), math.log10(40)
    fx = lambda h: L + h / (H + 6) * (W - L - R)
    fy = lambda v: T + (y1 - v) / (y1 - y0) * (Hh - T - B)
    col = {"KR": "#1f77b4", "JP": "#d62728", "GL": "#2ca02c", None: "#7f7f7f", "conflict": "#9467bd"}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{Hh}" fill="white"/>',
         f'<text x="{L}" y="18" font-size="13">같은 정확 점수의 % 궤적 (세로 log; 끝 점 = 최종, 속 빈 점 = C 상한 시각)</text>']
    for d in range(0, 6):
        o.append(f'<line x1="{fx(24 * d)}" y1="{T}" x2="{fx(24 * d)}" y2="{Hh - B}" stroke="#ddd"/>')
        o.append(f'<text x="{fx(24 * d)}" y="{Hh - B + 15}" text-anchor="middle">{24 * d}</text>')
    for v in (0.2, 0.5, 1, 2, 3, 5, 10, 20):
        o.append(f'<line x1="{L}" y1="{fy(math.log10(v))}" x2="{W - R}" y2="{fy(math.log10(v))}" stroke="#eee"/>')
        o.append(f'<text x="{L - 5}" y="{fy(math.log10(v)) + 4}" text-anchor="end">{v}%</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{Hh - 12}" text-anchor="middle">h (챌린지 시작 후 시간, 세로선 = 05:00)</text>')
    for p, t in traj.items():
        pts = [(r["h"], r["pct"], r["t_grade"]) for r in t["pre"]]
        if t["final"]:
            pts.append((H + 3, t["final"]["pct"], "F"))
        if len(pts) < 2:
            continue
        c = col.get(t["server"], "#7f7f7f")
        d = " ".join(f"{fx(h):.1f},{fy(math.log10(x)):.1f}" for h, x, _ in pts)
        o.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-opacity="0.5"/>')
        for h, x, g in pts:
            fill = "white" if g == "C" else c
            o.append(f'<circle cx="{fx(h):.1f}" cy="{fy(math.log10(x)):.1f}" r="2.6" fill="{fill}" stroke="{c}"/>')
    for k, (name, c) in enumerate((("KR", col["KR"]), ("JP", col["JP"]), ("GL", col["GL"]), ("미상", col[None]))):
        o.append(f'<circle cx="{W - R + 14}" cy="{T + 10 + 16 * k}" r="4" fill="{c}"/>')
        o.append(f'<text x="{W - R + 22}" y="{T + 14 + 16 * k}">{name}</text>')
    o.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(o))


if __name__ == "__main__":
    main()
