"""솔로레이드 2단계 탐색 2.4 시간 축: 05:00 갱신 계단이 있는지 본다 (모델 0, 요청 0, 표준 라이브러리만).

python3 -I tools/solo_raid/explore_timeaxis.py STATE [--out DIR] [--boot 2000] [--seed 1] [--win 6] [--night 1-8]

입력 (읽기만 한다): STATE/season.json(c_start·end), STATE/core.jsonl(기본 집합 use), STATE/obs.jsonl(게시 시각, 기준 활동량)

h 0 = 챌린지 시작(05:00)이므로 05:00 갱신 = h가 24의 배수. 시점(moment)마다 한 줄(use_indep 우선), flag best_record 줄은 뺀다.

(a) 같은 사람(정확 점수 지문 person, 점수 고정)의 이웃 시점 쌍. 둘 다 시각 등급 A·B, 둘 다 챌린지 안(h < 길이)인 쌍만.
    y = ln(p2/p1). 모형 y = c·X + J·n05 (절편 없음, 최소제곱), n05 = 사이에 낀 05:00 수,
    X = ∫ a(s)·(s + h0)^-β ds — a(s)는 KST 시각별 활동량(flat = 일정, act = 시즌 안 사진 글의 시각별 게시량을 평균 1로,
    night = --night 시각만 0.05 나머지 1 — 밤에 기록 갱신이 거의 멈춘다는 가정),
    β는 격자에서 잔차 제곱합 최소. J가 계단 크기(ln %, 05:00 하나당). 구간은 사람 단위 붓스트랩.
    짝 비교: 05:00을 끼는 쌍과 안 끼는 쌍의 시간당 ln % 변화 중앙값도 낸다.
(b) 같은 % 근처의 점수가 05:00 앞뒤로 뛰는지(단면 비교). 경계 B 앞 [B-w, B)와 뒤 [B, B+w)의 A·B 줄로
    ln 점수 = α_앞/뒤 + b·ln % (기울기 공통) → 점프 = α_뒤 - α_앞. 05:00 경계(24·48·72·96)와
    가짜 경계(6시간 간격의 나머지: 11·17·23시)를 같은 방법으로 견준다. 서버: 한섭만 / 전체(서버 무시) 두 가지.
(c) 기록 갱신 시점 단서. 개인 점수 갱신을 직접 이을 열쇠(작성자)가 state에 없어서 간접으로만 본다:
    - 결과 화면(result, 전투 직후) A 촬영 시각의 KST 시각 분포 vs 다른 화면 vs 시즌 안 사진 글 전체 게시량
    - 정확 점수 지문의 첫 관측 시각(새 기록이 처음 보인 때) KST 시각 분포
    - 같은 정확 점수가 05:00을 넘어 다시 보인 사람 수와 최대 간격 (점수 규칙 단서)
    - 일차별 점수 상단(최대·상위 5개 중앙값)과 1% 창(0.67–1.5%) 점수 중앙값 (누적이면 일차마다 크게 늘어야 한다)

출력: 표준 출력. --out DIR이면 timeaxis.json과 timeaxis.html(쌍 그림)을 쓴다. state/에는 쓰지 않는다.
"""
import argparse
import collections
import datetime as dt
import json
import math
import os
import random


import importlib.util as _ilu  # noqa: E402
_sp = _ilu.spec_from_file_location("split", os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.py"))
split = _ilu.module_from_spec(_sp)
_sp.loader.exec_module(split)

def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def quant(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


OFF = 5.0   # c_start의 KST 시각 (main에서 season.json으로 정함). 3일차 05:00 시즌이면 5 — S38(12:00)·S28(18:30)은 다르다 (할 일 3.A5)


def n05(h1, h2):
    """h1~h2 사이에 낀 KST 05:00 수."""
    return int((h2 + OFF - 5) // 24) - int((h1 + OFF - 5) // 24)


def integ(h1, h2, act, beta, h0=6.0, step=0.1):
    """∫_{h1}^{h2} act(KST 시각)·(s+h0)^-beta ds, 중점 규칙."""
    n = max(1, int(math.ceil((h2 - h1) / step)))
    d = (h2 - h1) / n
    s = 0.0
    for k in range(n):
        x = h1 + (k + 0.5) * d
        s += act[int((OFF + x) % 24)] * (x + h0) ** (-beta)
    return s * d


def ols2(X1, X2, y):
    """y = c·X1 + J·X2 (절편 없음). X2가 None이면 c만."""
    if X2 is None:
        sxx = sum(a * a for a in X1)
        c = sum(a * b for a, b in zip(X1, y)) / sxx
        sse = sum((b - c * a) ** 2 for a, b in zip(X1, y))
        return c, 0.0, sse
    s11 = sum(a * a for a in X1)
    s22 = sum(a * a for a in X2)
    s12 = sum(a * b for a, b in zip(X1, X2))
    s1y = sum(a * b for a, b in zip(X1, y))
    s2y = sum(a * b for a, b in zip(X2, y))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return None
    c = (s1y * s22 - s2y * s12) / det
    J = (s2y * s11 - s1y * s12) / det
    sse = sum((b - c * a - J * e) ** 2 for a, e, b in zip(X1, X2, y))
    return c, J, sse


BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def fit_pairs(pairs, act, use_J, xcache):
    best = None
    for beta in BETAS:
        X1 = [xcache[(id(p), beta, act["name"])] for p in pairs]
        X2 = [p["n05"] for p in pairs] if use_J else None
        r = ols2(X1, X2, [p["y"] for p in pairs])
        if r is None:
            continue
        c, J, sse = r
        if best is None or sse < best["sse"]:
            best = {"beta": beta, "c": c, "J": J, "sse": sse}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--win", type=float, default=6.0)
    ap.add_argument("--night", default="1-8", help="night 활동량 모형에서 거의 멈춘다고 보는 KST 시각 (포함)")
    split.add_final_arg(ap)
    a = ap.parse_args()
    split.guard_args(a)   # 시험 시즌 정답 감시 (할 일 3.0b)
    season = json.load(open(os.path.join(a.state, "season.json"), encoding="utf-8"))
    c0 = ts(season["c_start"])
    global OFF
    OFF = c0.hour + c0.minute / 60
    clen = round((ts(season["end"]) - c0).total_seconds() / 3600)
    core = jl(os.path.join(a.state, "core.jsonl"))
    obs = jl(os.path.join(a.state, "obs.jsonl"))
    U = [r for r in core if r["use"]]
    nbest = sum(1 for r in U if "best_record" in r["flag"])
    U = [r for r in U if "best_record" not in r["flag"]]
    mom = {}
    for r in U:
        if r["moment"] not in mom or (r["use_indep"] and not mom[r["moment"]]["use_indep"]):
            mom[r["moment"]] = r
    M = list(mom.values())
    out = {"season": season["name"], "n_moments": len(M), "dropped_best_record_rows": nbest}

    # 활동량 기준: 시즌 안 사진 글의 KST 게시 시각 (챌린지 중)
    hb = collections.Counter()
    seen = set()
    for o in obs:
        if o.get("season") != "in" or not o.get("post_t"):
            continue
        k = (o["site"], o["no"])
        hp = (ts(o["post_t"]) - c0).total_seconds() / 3600
        if 0 <= hp < clen and k not in seen:
            seen.add(k)
            hb[int((OFF + hp) % 24)] += 1
    mean = sum(hb.values()) / 24
    nlo, nhi = (int(x) for x in a.night.split("-"))
    ACTS = [{"name": "flat", "w": [1.0] * 24},
            {"name": "act", "w": [max(hb[h], 0.5) / mean for h in range(24)]},
            {"name": "night", "w": [0.05 if nlo <= h <= nhi else 1.0 for h in range(24)]}]
    out["activity_profile"] = [round(x, 3) for x in ACTS[1]["w"]]

    # ---------- (a) 같은 사람 쌍 ----------
    per = collections.defaultdict(list)
    for r in M:
        if r["person"].startswith("P"):
            per[r["person"]].append(r)
    allpairs, pairs = [], []
    for p, rs in per.items():
        rs.sort(key=lambda r: r["h"])
        for x, y in zip(rs, rs[1:]):
            q = {"person": p, "h1": x["h"], "h2": y["h"], "g1": x["t_grade"], "g2": y["t_grade"],
                 "p1": x["pct"], "p2": y["pct"], "score": x["score"],
                 "kr": "KR" in (x["server"], y["server"]), "y": math.log(y["pct"] / x["pct"])}
            allpairs.append(q)
            if q["g1"] != "C" and q["g2"] != "C" and q["h2"] < clen and q["h2"] > q["h1"]:
                q["n05"] = n05(q["h1"], q["h2"])
                pairs.append(q)
    print(f"(a) 같은 정확 점수 사람: 시점 2개 이상 {sum(1 for v in per.values() if len(v) > 1)}명, 이웃 쌍 {len(allpairs)}"
          f" → 둘 다 A·B·챌린지 안 {len(pairs)}쌍 ({len({q['person'] for q in pairs})}명), 05:00 낀 쌍 {sum(1 for q in pairs if q['n05'])}")
    neg = sum(1 for q in allpairs if q["y"] < 0)
    zero = sum(1 for q in allpairs if q["y"] == 0)
    print(f"    점수 고정인데 %가 내려간 쌍 {neg} · 그대로 {zero} · 오른 쌍 {len(allpairs) - neg - zero} (모든 등급 이웃 쌍)")
    out["a_counts"] = {"pairs_all": len(allpairs), "pairs_AB_in": len(pairs), "neg": neg, "zero": zero}

    # 짝 비교: 시간당 ln % 변화
    def rate_tab(sel, name):
        r = [q["y"] / (q["h2"] - q["h1"]) for q in sel]
        return f"{name}: n={len(r)} 시간당 중앙값 {median(r):.4f}" if r else f"{name}: n=0"
    short = [q for q in pairs if q["h2"] - q["h1"] <= 30]
    print("    시간당 ln % 변화 (Δh ≤ 30시간):", rate_tab([q for q in short if q["n05"] == 0], "05:00 안 낌"), "|",
          rate_tab([q for q in short if q["n05"] > 0], "05:00 낌"))
    # 05:00 낀 쌍 중 앞뒤 3시간 안에 양 끝이 있는 쌍
    tight = [q for q in pairs if q["n05"] == 1 and ((q["h1"] + OFF - 5) % 24) >= 18 and ((q["h2"] + OFF - 5) % 24) <= 6]
    print(f"    05:00 앞 6시간·뒤 6시간 안에 양 끝이 있는 쌍: {len(tight)}")
    for q in tight:
        print(f"      {q['person']} h {q['h1']:.1f}→{q['h2']:.1f}  {q['p1']}→{q['p2']}%  ln비 {q['y']:.3f}")

    # 모형 적합
    xcache = {}
    for q in pairs:
        for act in ACTS:
            for beta in BETAS:
                xcache[(id(q), beta, act["name"])] = integ(q["h1"], q["h2"], act["w"], beta)
    fits = {}
    print("    모형 y = c·∫a(s)(s+6)^-β ds + J·n05 (β 격자 최소 SSE) — 잔차 제곱합, J(ln %), exp(J)")
    for act in ACTS:
        for useJ in (False, True):
            f = fit_pairs(pairs, act, useJ, xcache)
            fits[f"{act['name']}/{'J' if useJ else 'noJ'}"] = f
            print(f"      a={act['name']:4} {'J 있음' if useJ else 'J 없음'}: β={f['beta']:.2f} c={f['c']:.4f} J={f['J']:+.3f}"
                  f" (×{math.exp(f['J']):.3f}) SSE={f['sse']:.3f}  n={len(pairs)}")
    # 사람 단위 붓스트랩 (J 있음 모형)
    rnd = random.Random(a.seed)
    byp = collections.defaultdict(list)
    for q in pairs:
        byp[q["person"]].append(q)
    plist = list(byp)
    boots = {}
    for act in ACTS:
        Js = []
        for _ in range(a.boot):
            samp = []
            for _ in plist:
                samp.extend(byp[rnd.choice(plist)])
            if sum(1 for q in samp if q["n05"]) == 0:
                continue
            f = fit_pairs(samp, act, True, xcache)
            if f:
                Js.append(f["J"])
        lo, hi = quant(Js, 0.025), quant(Js, 0.975)
        boots[act["name"]] = {"J_lo": lo, "J_hi": hi, "n": len(Js)}
        print(f"      a={act['name']:4} J 95% 사람 붓스트랩 구간 [{lo:+.3f}, {hi:+.3f}] (×{math.exp(lo):.3f}~×{math.exp(hi):.3f}), {len(Js)}회")
    # 한섭만
    kp = [q for q in pairs if q["kr"]]
    for act in ACTS:
        f = fit_pairs(kp, act, True, xcache) if sum(1 for q in kp if q["n05"]) else None
        if f:
            print(f"      한섭 쌍만 a={act['name']}: n={len(kp)} (05:00 낌 {sum(1 for q in kp if q['n05'])}) J={f['J']:+.3f} β={f['beta']}")
    out["a_fits"] = fits
    out["a_boot"] = boots
    out["a_pairs"] = [{k: v for k, v in q.items()} for q in pairs]

    # ---------- (b) 단면: 같은 % 근처 점수의 05:00 앞뒤 ----------
    print(f"\n(b) 단면 점프: 경계 앞 [B-{a.win:g}, B)·뒤 [B, B+{a.win:g}) A·B 줄, ln 점수 = α + b·ln % (b 공통)")
    ab = [r for r in M if r["t_grade"] != "C" and r["score"] and r["h"] < clen + a.win]

    def jump(B, srvsel):
        bef = [r for r in ab if B - a.win <= r["h"] < B and srvsel(r)]
        aft = [r for r in ab if B <= r["h"] < B + a.win and srvsel(r)]
        if len(bef) < 2 or len(aft) < 2:
            return None, len(bef), len(aft)
        rows = [(math.log(r["pct"]), math.log(r["score"]), 0) for r in bef] + \
               [(math.log(r["pct"]), math.log(r["score"]), 1) for r in aft]
        mx = [sum(x for x, _, g in rows if g == k) / sum(1 for *_, g in rows if g == k) for k in (0, 1)]
        my = [sum(y for _, y, g in rows if g == k) / sum(1 for *_, g in rows if g == k) for k in (0, 1)]
        sxx = sum((x - mx[g]) ** 2 for x, _, g in rows)
        if sxx < 1e-9:
            return None, len(bef), len(aft)
        b = sum((x - mx[g]) * (y - my[g]) for x, y, g in rows) / sxx
        al = [my[k] - b * mx[k] for k in (0, 1)]
        return al[1] - al[0], len(bef), len(aft)

    res_b = {}
    for sname, sel in (("KR", lambda r: r["server"] == "KR"), ("all", lambda r: True)):
        real, fake = [], []
        line = []
        for k in range(1, int(clen // 6)):
            B = 6 * k + (5 - OFF) % 6      # 6시간 칸 경계가 KST 05·11·17·23시에 오게
            j, nb, na = jump(B, sel)
            is05 = (B + OFF - 5) % 24 == 0
            tag = f"{int(OFF + B) % 24:02d}시 h{B:g}"
            line.append(f"{tag}{'*' if is05 else ''}:{'—' if j is None else f'{j:+.3f}'}({nb}/{na})")
            if j is not None:
                (real if is05 else fake).append(j)
        res_b[sname] = {"real_05": real, "placebo": fake}
        print(f"  {sname}: " + "  ".join(line))
        if real:
            fa = sorted(abs(x) for x in fake)
            print(f"    05:00 점프 |중앙값| {median([abs(x) for x in real]):.3f} (n={len(real)}) vs 가짜 경계 |중앙값| "
                  f"{median(fa) if fa else float('nan'):.3f} (n={len(fake)}, 90% {quant(fa, 0.9) if fa else float('nan'):.3f})")
    out["b"] = res_b

    # ---------- (c) 기록 갱신 시점 단서 ----------
    print("\n(c) 기록 갱신 단서 (KST 시각별, 챌린지 중)")
    hr_res, hr_oth = collections.Counter(), collections.Counter()
    for r in M:
        if r["t_grade"] == "A" and 0 <= r["h"] < clen:
            (hr_res if r["screen"] == "result" else hr_oth)[int((OFF + r["h"]) % 24)] += 1
    first = {}
    for r in M:
        if r["person"].startswith("P") and r["t_grade"] != "C" and 0 <= r["h"] < clen:
            if r["person"] not in first or r["h"] < first[r["person"]]:
                first[r["person"]] = r["h"]
    hr_first = collections.Counter(int((OFF + h) % 24) for h in first.values())
    print("시      " + "".join(f"{h:>4}" for h in range(24)))
    for name, c in (("결과화면A", hr_res), ("그외화면A", hr_oth), ("첫관측AB", hr_first), ("사진글전체", hb)):
        print(f"{name:7}" + "".join(f"{c[h]:>4}" for h in range(24)) + f"  계 {sum(c.values())}")
    win = lambda c, hs: sum(c[h] for h in hs)
    for name, c in (("결과화면A", hr_res), ("그외화면A", hr_oth), ("첫관측AB", hr_first), ("사진글전체", hb)):
        tot = sum(c.values()) or 1
        print(f"  {name}: 05~07시 {win(c, (5, 6, 7))} ({100 * win(c, (5, 6, 7)) / tot:.1f}%) · 08~10시 {win(c, (8, 9, 10))}"
              f" ({100 * win(c, (8, 9, 10)) / tot:.1f}%)")
    out["c_hours"] = {"result_A": [hr_res[h] for h in range(24)], "other_A": [hr_oth[h] for h in range(24)],
                      "first_seen_AB": [hr_first[h] for h in range(24)], "photo_posts": [hb[h] for h in range(24)]}

    # 같은 정확 점수가 05:00을 넘어 다시 보인 사람
    span = []
    for p, rs in per.items():
        if len(rs) < 2:
            continue
        hs = sorted(r["h"] for r in rs)
        hs_in = [h for h in hs if h < clen]
        span.append((p, hs[0], hs[-1], n05(hs[0], min(hs[-1], clen - 1e-6)), hs[-1] >= clen))
    cross = [s for s in span if s[3] > 0]
    print(f"\n  정확 점수 사람 {len(span)}명 중 같은 점수가 05:00을 넘어 다시 보인 사람 {len(cross)}명 "
          f"(2번 이상 넘김 {sum(1 for s in cross if s[3] >= 2)}명, 최대 간격 {max(s[2] - s[1] for s in span):.0f}시간), "
          f"종료 뒤까지 같은 점수 {sum(1 for s in span if s[4])}명 (C 상한 시각 포함)")
    out["c_same_score_span"] = {"persons": len(span), "cross_05": len(cross),
                                "cross_05_ge2": sum(1 for s in cross if s[3] >= 2)}

    # 일차별 점수 상단
    print("\n  일차별 점수(억) — 모든 등급(C는 상한 일차), 서버 무관 / 한섭")
    days = [1, 2, 3, 4, 5, "post"]
    tabd = {}
    for sname, sel in (("all", lambda r: True), ("KR", lambda r: r["server"] == "KR")):
        print(f"   {sname}: " + " | ".join(
            f"d{d}: 최대 {max(sc):.1f} 상위5중앙 {median(sorted(sc)[-5:]):.1f} 1%창 중앙 {median(w1) if w1 else float('nan'):.1f}(n{len(w1)})"
            for d in days
            for sc in [[r["score"] for r in M if r["day"] == d and r["score"] and sel(r)]]
            for w1 in [[r["score"] for r in M if r["day"] == d and r["score"] and sel(r) and 1 / 1.5 <= r["pct"] <= 1.5]]
            if sc))
        tabd[sname] = {str(d): sorted(r["score"] for r in M if r["day"] == d and r["score"] and sel(r))[-5:] for d in days}
    out["c_top_scores"] = tabd

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "timeaxis.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        write_html(os.path.join(a.out, "timeaxis.html"), pairs, allpairs, per, clen)
        print(f"\n쓴 파일: {a.out}/timeaxis.json · timeaxis.html")


def write_html(path, pairs, allpairs, per, clen):
    W, H, L, B = 900, 420, 50, 30
    hmax = clen + 8
    pmin, pmax = math.log(0.15), math.log(40)
    X = lambda h: L + (W - L - 10) * h / hmax
    Y = lambda p: H - B - (H - B - 10) * (math.log(p) - pmin) / (pmax - pmin)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="같은 정확 점수 사람의 % 궤적">']
    for d in range(0, int(hmax) + 1, 24):
        s.append(f'<line x1="{X(d)}" y1="10" x2="{X(d)}" y2="{H - B}" class="b05"/>'
                 f'<text x="{X(d)}" y="{H - B + 14}" class="ax">{d}h</text>')
    for d in range(12, int(hmax), 24):
        s.append(f'<line x1="{X(d)}" y1="10" x2="{X(d)}" y2="{H - B}" class="b17"/>')
    for p in (0.2, 0.5, 1, 2, 5, 10, 20):
        s.append(f'<text x="{L - 6}" y="{Y(p) + 4}" class="ay">{p:g}%</text><line x1="{L}" y1="{Y(p)}" x2="{W - 10}" y2="{Y(p)}" class="grid"/>')
    for p, rs in per.items():
        rs = sorted(rs, key=lambda r: r["h"])
        if len(rs) < 2:
            continue
        pts = [(r["h"], r["pct"], r["t_grade"]) for r in rs if r["h"] <= hmax and 0.15 <= r["pct"] <= 40]
        if len(pts) < 2:
            continue
        s.append('<polyline class="tr" points="' + " ".join(f"{X(h):.1f},{Y(q):.1f}" for h, q, _ in pts) + '"/>')
        for h, q, g in pts:
            s.append(f'<circle cx="{X(h):.1f}" cy="{Y(q):.1f}" r="2.6" class="{"pa" if g != "C" else "pc"}"/>')
    s.append("</svg>")
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S41 시간 축</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#4f8fc4;--c:#c98a2b;--b05:#c0392b;--grid:#e6e3dc}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#18191b;--fg:#e8e6e1;--mut:#9a9893;--line:#7fb0dc;--c:#e0a84a;--b05:#e06b5b;--grid:#2c2e31}}}}
:root[data-theme="dark"]{{--bg:#18191b;--fg:#e8e6e1;--mut:#9a9893;--line:#7fb0dc;--c:#e0a84a;--b05:#e06b5b;--grid:#2c2e31}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px}} main{{max-width:960px;margin:auto}}
text{{font-size:11px;fill:var(--mut)}} .ax{{text-anchor:middle}} .ay{{text-anchor:end}} .grid{{stroke:var(--grid)}}
.b05{{stroke:var(--b05);stroke-dasharray:4 3}} .b17{{stroke:var(--grid);stroke-dasharray:2 4}}
.tr{{fill:none;stroke:var(--line);stroke-opacity:.45}} .pa{{fill:var(--line)}} .pc{{fill:none;stroke:var(--c)}} p{{color:var(--mut)}}
</style></head><body><main><h1>같은 점수, 밀리는 %</h1>
<p>정확 점수 지문으로 이은 사람의 시점별 % (로그 축). 빨간 점선 = 05:00 갱신(h 24의 배수), 회색 = 17:00. 채운 점 = 시각 A·B, 빈 점 = C(게시 시각 상한).
05:00 계단이 있다면 빨간 선에서 선이 꺾여 뛰어야 한다.</p>{''.join(s)}</main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
