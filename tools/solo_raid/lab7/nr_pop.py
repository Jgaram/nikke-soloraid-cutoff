"""WP8a 인구 기록표 (2026-11-01) — 한섭 인구 N이 시즌 사이·시즌 안에서 어떻게 다르고, 커뮤니티 값을 얼마나 믿을 수 있나.

docs/solo-raid-method.md 「WP8a 인구 기록표」 카드대로. 공용 코드(common·obs_table 등)는 읽기만 한다.
정답(answer_key·truth*)은 읽지 않는다 — 6의 서술 하나만 예외로 prior_dev.json의 season_levels(m_s)를 쓴다.

  python3 -I tools/solo_raid/lab7/nr_pop.py cand    --w W            # 1. 후보 → W/cand.jsonl (원문 조각 — 임시 폴더에만)
  python3 -I tools/solo_raid/lab7/nr_pop.py bounds  --w W            # 3. 우리 자료로 낸 N 구간 (a)(b) → W/bounds.json
  python3 -I tools/solo_raid/lab7/nr_pop.py slope   --w W            # 4. 번짐 크기 → W/slope.json
  python3 -I tools/solo_raid/lab7/nr_pop.py summary --w W [--read 'W/read_*.jsonl']   # 5·6. r7/nr/pop.json·pop.md

  관측 표(--full)는 W/obs7full/full/{시즌}.jsonl.gz — 먼저
  python3 -I tools/solo_raid/lab7/obs_table.py build --out W/obs7full --full

공개 규칙: 커밋되는 결과(pop.json·pop.md·pop_prompt.md·이 코드)에 글 번호·원문·닉네임을 넣지 않는다.
글 열쇠는 sha1(「시즌|사이트|글 번호」)[:12]. 후보 창(원문 조각)은 W(세션 임시 폴더) 파일에만 둔다.
"""
import argparse
import collections
import glob
import gzip
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import obs_table as O  # noqa: E402

PR = C.PR
OUT = os.path.join(C.ROOT, "data", "solo-raid", "analysis", "r7", "nr")
SEASONS = C.DEV                      # S19~S42
WIN = 120                            # 후보 창: 맞은 곳 앞뒤 글자 수

# ---------------------------------------------------------------- 1. 후보 정규식 (계산 전 고정, 2026-11-01)
# 낱말 — 하나라도 있고
RE_WORD = re.compile(r"인구|참여자|참가자|참여\s*인원|모수|201\s*등|201\s*위|숫자단\s*컷")
# 숫자 — 하나라도 함께 있어야 한다: 「N만」(만원·만큼 제외) · 「N,NNN명」 · 「0.NN%·퍼」
RE_NUM = re.compile(
    r"\d+(?:\.\d+)?\s*만(?!\s*원|큼)"            # 11.8만 · 12만
    r"|\d{1,3}(?:,\d{3})+\s*명|\d{5,6}\s*명"      # 118,000명 · 118000명
    r"|\d*\.\d+\s*(?:%|％|퍼센트|퍼)"             # 0.17% · .17퍼
)
NEAR = WIN                           # 낱말과 숫자가 이만큼(글자) 안에 있어야 맞은 곳이다

CAND_COLS = ["k", "idx", "season", "site", "src", "t", "h", "win"]


def post_key(season, site, no):
    return hashlib.sha1(f"{season}|{site}|{no}".encode()).hexdigest()[:12]


def windows(text):
    """(낱말, 숫자)가 NEAR 글자 안에 함께 있는 곳의 창 [(시작, 끝)] — 앞뒤 WIN자를 붙이고 겹치면 합친다."""
    ws = [(m.start(), m.end()) for m in RE_WORD.finditer(text)]
    ns = [(m.start(), m.end()) for m in RE_NUM.finditer(text)]
    if not ws or not ns:
        return []
    spans = []
    for a, b in ws:
        for c, d in ns:
            gap = max(c - b, a - d, 0)
            if gap <= NEAR:
                spans.append((max(min(a, c) - WIN, 0), min(max(b, d) + WIN, len(text))))
    spans.sort()
    out = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def num_windows(text):
    """숫자(RE_NUM)만 있는 곳의 창 — 제목에 낱말이 있고 사진에 숫자만 있는 글(arca 「첫날 인구수」 재게시)용."""
    spans = sorted((max(m.start() - WIN, 0), min(m.end() + WIN, len(text))) for m in RE_NUM.finditer(text))
    out = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def ocr_text(o):
    return " ".join(it[1] for it in o.get("items", []) if len(it) > 1 and it[1])


def cmd_cand(a):
    os.makedirs(a.w, exist_ok=True)
    rows = []
    per = {}
    for s in SEASONS:
        st = C.state_of(s)
        c0, H = C.info(s)
        posts = {}
        for l in open(os.path.join(st, "posts_min.jsonl"), encoding="utf-8"):
            p = json.loads(l)
            posts[str(p["no"])] = p
        cnt = collections.Counter()
        for no, p in posts.items():
            text = (p.get("title") or "") + "\n" + (p.get("body") or "")
            for i, (x, y) in enumerate(windows(text)):
                cnt["post"] += 1
                rows.append({"k": post_key(s, p["site"], no), "idx": f"p{i}", "season": s, "site": p["site"], "src": "post",
                             "t": p["t"], "h": round(PR.hours_of(p["t"], c0), 3), "win": text[x:y]})
        for l in open(os.path.join(st, "ocr.jsonl"), encoding="utf-8"):
            o = json.loads(l)
            no, _, ord_ = o["id"].rpartition("_")
            p = posts.get(no)
            if p is None:
                cnt["ocr_no_post"] += 1
                continue
            text = ocr_text(o)
            h = round(PR.hours_of(p["t"], c0), 3)
            for i, (x, y) in enumerate(windows(text)):
                cnt["ocr"] += 1
                rows.append({"k": post_key(s, p["site"], no), "idx": f"o{ord_}.{i}", "season": s, "site": p["site"], "src": "ocr",
                             "t": p["t"], "h": h, "win": text[x:y]})
            # 확장 (카드 밖, 2026-11-01): 글 제목·본문에 낱말이 있고 그 글에 창이 없는데 사진 글자에 숫자가 있으면 — 낱말은 글에서, 숫자는 사진에서
            ptext = (p.get("title") or "") + "\n" + (p.get("body") or "")
            if RE_WORD.search(ptext) and not windows(ptext) and not windows(text):
                for i, (x, y) in enumerate(num_windows(text)):
                    cnt["ocr_title"] += 1
                    rows.append({"k": post_key(s, p["site"], no), "idx": f"t{ord_}.{i}", "season": s, "site": p["site"], "src": "ocr_title",
                                 "t": p["t"], "h": h, "win": "[글] " + ptext[:240] + " [사진] " + text[x:min(y, x + 2500)]})
        per[s] = cnt
    path = os.path.join(a.w, "cand.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tot = collections.Counter()
    print("시즌 post ocr ocr_title 글수(열쇠)")
    for s in SEASONS:
        ks = {r["k"] for r in rows if r["season"] == s}
        print(s, per[s]["post"], per[s]["ocr"], per[s]["ocr_title"], len(ks), "(ocr 글 없음 %d)" % per[s]["ocr_no_post"] if per[s]["ocr_no_post"] else "")
        tot.update(per[s])
    print("합계 창", len(rows), "post", tot["post"], "ocr", tot["ocr"], "ocr_title(확장)", tot["ocr_title"], "글 열쇠", len({(r['k']) for r in rows}))
    print("→", path)


# ---------------------------------------------------------------- 3·4. 우리 자료로 낸 구간·번짐 크기
# % 표시 구간 (1f 그대로): 화면 p는 실제 [p − 0.005, p + 0.01)
P_LO, P_HI = 0.005, 0.01
RANK_LO, RANK_HI = 120, 200          # (b)의 등수 줄 범위 (M3 범위 안)
WIDE_LO = 50                         # 범위를 넓힌 판의 수만 따로 센다
SLOPE_P = (0.08, 0.6)                # 4: 끝 24시간 % 범위
SLOPE_MIN_ROWS = 8
BOOT_N, BOOT_SEED = 2000, 20261101
BIN_H = 24.0                         # 구간 요약의 시간 칸


def load_kr(w, s):
    """한섭 사진 줄(kind A·B, exclude 빔, 점수·시각 있음) → (머리, 등수 줄, % 줄). 관측 표 --full 판."""
    hd, rows = O.read_cell(os.path.join(w, "obs7full", "full", f"{s}.jsonl.gz"))
    kr = [r for r in rows if r["server"] == "KR" and r["kind"] in ("A", "B") and not r["exclude"]
          and r["score"] is not None and r["h"] is not None]
    rk = [r for r in kr if r["rank"] is not None and r["pct"] is None and r["rank"] <= 200]
    pc = [r for r in kr if r["pct"] is not None and r["rank"] is None]
    return hd, rk, pc


def n_lo_of(r, p):
    """등수 r(이상)·% p에서 N의 하한: N > r ÷ ((p + 0.01)/100)."""
    return 100.0 * r / (p + P_HI)


def n_hi_of(r, p):
    """등수 r(미만)·% p에서 N의 상한: N < r ÷ ((p − 0.005)/100). p ≤ 0.005면 없음."""
    return 100.0 * r / (p - P_LO) if p - P_LO > 1e-9 else None


def bounds_season(w, s, lo=RANK_LO):
    """한 시즌의 (a)(b) 구간 줄 목록과 셈. 줄: dict(h, side lo|hi, N, basis, r, p, a=% 줄 pid, b=등수 줄 pid)."""
    hd, rk, pc = load_kr(w, s)
    ent, cnt = [], collections.Counter()
    by_pid = collections.defaultdict(list)
    for r in rk:
        by_pid[r["pid"]].append(r)
    mid = [r for r in rk if lo <= r["rank"] <= RANK_HI]
    cnt["rank_rows"] = len(rk)
    cnt["rank_rows_120_200"] = len(mid)
    cnt["rank_rows_50_200"] = sum(1 for r in rk if WIDE_LO <= r["rank"] <= RANK_HI)
    cnt["pct_rows"] = len(pc)
    for b in pc:
        x, p, t = b["score"], b["pct"], b["h"]
        # (a) 같은 사람·같은 점수의 등수 줄 (t1 ≤ t2). r = 200이면 「200위에서 밀린 순간」이라 201.
        best = None
        for a in by_pid.get(b["pid"], []):
            if abs(a["score"] - x) > 1e-6 * max(abs(x), 1.0) or a["h"] > t + 1e-9:
                continue
            reff = a["rank"] + 1 if a["rank"] == 200 else a["rank"]
            if best is None or reff > best[0]:
                best = (reff, a)
        if best:
            reff, a = best
            basis = "a201" if a["rank"] == 200 else "a"
            cnt["a_pairs"] += 1
            cnt["a201_pairs"] += basis == "a201"
            ent.append({"h": t, "side": "lo", "N": n_lo_of(reff, p), "basis": basis, "r": reff, "p": p, "a": b["pid"], "b": a["pid"]})
        # (b) 같은 시즌 r위 점수 S_r(t′)와 이 % 줄의 점수 x
        bl = bh = None
        for a in mid:
            if x < a["score"] and a["h"] <= t + 1e-9:
                if bl is None or a["rank"] > bl["rank"]:
                    bl = a
            elif x > a["score"] and a["h"] >= t - 1e-9:
                if bh is None or a["rank"] < bh["rank"]:
                    bh = a
        if bl:
            cnt["b_lo_pairs"] += 1
            ent.append({"h": t, "side": "lo", "N": n_lo_of(bl["rank"], p), "basis": "b", "r": bl["rank"], "p": p, "a": b["pid"], "b": bl["pid"]})
        if bh:
            cnt["b_hi_pairs"] += 1
            nh = n_hi_of(bh["rank"], p)
            if nh is not None:
                ent.append({"h": t, "side": "hi", "N": nh, "basis": "b", "r": bh["rank"], "p": p, "a": b["pid"], "b": bh["pid"]})
        # 확장 c201 (카드 밖): % 줄이 보였다 = 숫자단(200위) 밖 = 등수 ≥ 201 → N ≥ 201 ÷ ((p+0.01)/100). 커뮤니티 방법과 같은 가정.
        ent.append({"h": t, "side": "lo", "N": n_lo_of(201, p), "basis": "c201", "r": 201, "p": p, "a": b["pid"], "b": None})
    return hd, ent, cnt


def summarize_bins(ent, H):
    """24시간 칸마다 기본(카드 a·a201·b)과 확장(c201)의 가장 센 하한·상한, 줄 수, 어긋남(하한 > 상한)."""
    nb = int(math.ceil(H / BIN_H))
    out = []
    for k in range(nb):
        a0, a1 = k * BIN_H, (k + 1) * BIN_H
        E = [e for e in ent if a0 <= e["h"] < a1 or (k == nb - 1 and e["h"] >= a0)]
        card = [e for e in E if e["basis"] != "c201"]
        lo = max((e["N"] for e in card if e["side"] == "lo"), default=None)
        hi = min((e["N"] for e in card if e["side"] == "hi"), default=None)
        lo_ext = max((e["N"] for e in E if e["side"] == "lo"), default=None)
        out.append({"h0": a0, "h1": a1, "n_card": len(card), "n_lo": sum(1 for e in card if e["side"] == "lo"),
                    "n_hi": sum(1 for e in card if e["side"] == "hi"), "lo": lo, "hi": hi, "lo_ext": lo_ext,
                    "n_ext": sum(1 for e in E if e["basis"] == "c201"),
                    "conflict": bool(lo is not None and hi is not None and lo > hi)})
    return out


def cmd_bounds(a):
    res = {"method": "bounds", "p_display": [-P_LO, P_HI], "rank_range": [RANK_LO, RANK_HI], "seasons": {}}
    tot = collections.Counter()
    for s in SEASONS:
        hd, ent, cnt = bounds_season(a.w, s)
        _, ent_w, cnt_w = bounds_season(a.w, s, lo=WIDE_LO)
        card = [e for e in ent if e["basis"] != "c201"]
        # 줄 수 줄이기: 같은 (basis, side, % 줄) 중 가장 센 것만 이미 하나다. 보고용 센 값
        kinds = collections.Counter((e["basis"], e["side"]) for e in card)
        wide = collections.Counter((e["basis"], e["side"]) for e in ent_w if e["basis"] != "c201")
        res["seasons"][s] = {
            "H": hd["H"], "counts": dict(cnt), "kinds": {f"{k[0]}_{k[1]}": v for k, v in sorted(kinds.items())},
            "kinds_wide_50_200": {f"{k[0]}_{k[1]}": v for k, v in sorted(wide.items())},
            "bins": summarize_bins(ent, hd["H"]),
            "entries": [[round(e["h"], 3), e["side"], round(e["N"]), e["basis"], e["r"], e["p"], e["a"], e["b"]] for e in card],
            "ext": [[round(e["h"], 3), round(e["N"])] for e in ent if e["basis"] == "c201"],   # W 파일에만 (pop.json에는 안 넣는다)
        }
        tot.update({f"{k[0]}_{k[1]}": v for k, v in kinds.items()})
        tot["seasons_with_card"] += 1 if card else 0
        tot["seasons_with_lo"] += 1 if any(e["side"] == "lo" for e in card) else 0
        tot["seasons_with_hi"] += 1 if any(e["side"] == "hi" for e in card) else 0
        tot["seasons_with_a201"] += 1 if any(e["basis"] == "a201" for e in card) else 0
        tot["wide_b_lo"] += wide[("b", "lo")]
        tot["wide_b_hi"] += wide[("b", "hi")]
        tot["n120_200_rows"] += cnt["rank_rows_120_200"]
        tot["n50_200_rows"] += cnt["rank_rows_50_200"]
        tot["conflict_bins"] += sum(1 for b in res["seasons"][s]["bins"] if b["conflict"])
    res["total"] = dict(tot)
    res["premise"] = premise_check(a.w)
    res["entry_cols"] = ["h", "side", "N", "basis", "r", "p", "pct_row_pid", "rank_row_pid"]
    path = os.path.join(a.w, "bounds.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, separators=(",", ":"))
    print("근거 종류별(카드 a·a201·b, 시즌 합):", {k: v for k, v in tot.items() if k.startswith(("a", "b_"))})
    print("값이 있는 시즌", tot["seasons_with_card"], "하한", tot["seasons_with_lo"], "상한", tot["seasons_with_hi"], "200→% 사진(a201)이 있는 시즌", tot["seasons_with_a201"])
    print("120~200위 등수 줄", tot["n120_200_rows"], "| 50~200위(넓힌 판, 따로) 등수 줄", tot["n50_200_rows"], "넓힌 판 (b)하한", tot["wide_b_lo"], "상한", tot["wide_b_hi"])
    print("하한>상한 칸", tot["conflict_bins"], "→", path)
    return res


def premise_check(w):
    """④ 5.2 전제 점검 (서술).
    (i) 「% 줄이 보였다 = 숫자단(200위) 밖」: % 줄 점수 x가 같은 시즌 r위(120~200)의 나중 점수 S_r(t′ ≥ t)보다 크면 모순. 시험할 수 있는 짝 수와 모순 수.
    (ii) 같은 사람·같은 점수의 % 줄이 둘 이상이면, 시각 차 ≥ 1시간인 이웃 짝에서 p가 표시 폭(0.015)보다 더 줄어든 비율 —
         등수 밀림보다 인구가 빨리 느는 때. 시즌 앞 절반(h < H/2)과 뒤 절반을 나눈다."""
    import numpy as np
    test = viol = test_w = viol_w = 0
    pairs = collections.Counter()
    dl = {"early": [], "late": []}
    for s in SEASONS:
        hd, rk, pc = load_kr(w, s)
        H = hd["H"]
        for b in pc:
            for a in rk:
                if a["h"] >= b["h"] - 1e-9 and a["rank"] >= WIDE_LO:
                    if a["rank"] >= RANK_LO:
                        test += 1
                        viol += b["score"] > a["score"]
                    test_w += 1
                    viol_w += b["score"] > a["score"]
        grp = collections.defaultdict(list)
        for b in pc:
            grp[(b["pid"], round(b["score"], 4))].append(b)
        for L in grp.values():
            L.sort(key=lambda r: r["h"])
            for x, y in zip(L, L[1:]):
                if y["h"] - x["h"] < 1.0 or y["h"] == x["h"]:
                    continue
                half = "early" if x["h"] < H / 2 else "late"
                d = y["pct"] - x["pct"]
                pairs[(half, "drop" if d < -0.015 else "rise" if d > 0.015 else "flat")] += 1
                if x["pct"] > 0 and y["pct"] > 0:
                    dl[half].append(math.log(y["pct"] / x["pct"]) / (y["h"] - x["h"]) * 24)
    out = {"b_hi_testable_120_200": test, "b_hi_violations_120_200": viol, "b_hi_testable_50_200": test_w, "b_hi_violations_50_200": viol_w,
           "same_score_pairs": {f"{k[0]}_{k[1]}": v for k, v in sorted(pairs.items())}, "unit": "다음 % 줄과 표시 폭 0.015%p 넘게 달라진 짝"}
    for half, v in dl.items():
        out[f"dlnp_per_day_{half}"] = {"n": len(v), "median": round(float(np.median(v)), 4) if v else None}
    return out


def fit_slope(rows):
    """ln score를 ln p에 회귀 → −기울기. 반환 (−기울기, 줄 수)."""
    import numpy as np
    x = np.log([r["pct"] for r in rows])
    y = np.log([r["score"] for r in rows])
    xm, ym = x.mean(), y.mean()
    den = ((x - xm) ** 2).sum()
    if den <= 0:
        return None, len(rows)
    return float(-((x - xm) * (y - ym)).sum() / den), len(rows)


def slope_rows(w, s, servers=("KR",), kinds=("A", "B")):
    """시즌 s의 끝 24시간 % 줄 (사진, exclude 빔, 0.08 ≤ p ≤ 0.6, 서버가 servers 안). 같은 사람·같은 %는 한 줄로."""
    hd, rows = O.read_cell(os.path.join(w, "obs7full", "full", f"{s}.jsonl.gz"))
    H = hd["H"]
    seen, out = set(), []
    for r in rows:
        if (r["server"] not in servers or r["kind"] not in kinds or r["exclude"] or r["pct"] is None or r["rank"] is not None
                or r["score"] is None or r["h"] is None or r["score"] <= 0):
            continue
        if r["h"] >= H - 24 and SLOPE_P[0] <= r["pct"] <= SLOPE_P[1]:
            k = (r["pid"], round(r["pct"] / 0.01))
            if k not in seen:
                seen.add(k)
                out.append(r)
    return out


def slope_variant(w, servers, kinds, min_rows):
    import numpy as np
    per, vals = {}, []
    for s in SEASONS:
        rows = slope_rows(w, s, servers, kinds)
        sl, n = fit_slope(rows) if rows else (None, 0)
        ok = sl is not None and n >= min_rows
        per[s] = {"rows": n, "slope": round(sl, 4) if ok else None}
        if ok:
            vals.append(sl)
    v = np.array(vals)
    res = {"min_rows": min_rows, "servers": ["KR" if x == "KR" else "미상" for x in servers], "kinds": list(kinds), "n_seasons": len(v), "seasons": per}
    if len(v):
        res["median"] = round(float(np.median(v)), 4)
        res["min"], res["max"] = round(float(v.min()), 4), round(float(v.max()), 4)
        rng = np.random.default_rng(BOOT_SEED)
        meds = np.array([np.median(rng.choice(v, size=len(v), replace=True)) for _ in range(BOOT_N)])
        res["boot_median_90"] = [round(float(x), 4) for x in np.quantile(meds, [0.05, 0.95])]
        res["boot"] = {"n": BOOT_N, "seed": BOOT_SEED, "unit": "season"}
    return res


def cmd_slope(a):
    """카드 4 본안(한섭·A B·8줄 이상) + 어림 판 몇 개(줄 수 하한을 낮추거나 서버 미상·C 사진을 넣음 — 보고만)."""
    main = slope_variant(a.w, ("KR",), ("A", "B"), SLOPE_MIN_ROWS)
    res = {"method": "slope", "window": "h >= H-24", "p_range": list(SLOPE_P), "main": main, "variants": {
        "KR_AB_min5": slope_variant(a.w, ("KR",), ("A", "B"), 5),
        "KR_AB_min3": slope_variant(a.w, ("KR",), ("A", "B"), 3),
        "KR_ABC_min8": slope_variant(a.w, ("KR",), ("A", "B", "C"), 8),
        "KRorUnknown_AB_min8": slope_variant(a.w, ("KR", None), ("A", "B"), 8),
        "KRorUnknown_ABC_min8": slope_variant(a.w, ("KR", None), ("A", "B", "C"), 8),
    }}
    path = os.path.join(a.w, "slope.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for name, r in [("본안 KR_AB_min8", main)] + list(res["variants"].items()):
        print(name, "시즌", r["n_seasons"], "중앙", r.get("median"), "범위", r.get("min"), r.get("max"), "붓스트랩 90%", r.get("boot_median_90"))
    return res


# ---------------------------------------------------------------- 5·6. 정리 (판독 결과가 들어오면 ①②③, 없으면 빈칸)
SHAPES = ("linear", "log", "sat")    # ② 꼴 후보 (계산 전 고정): 직선 · ln(1 + u/τ) · 1 − e^(−u/τ)
TAU_GRID = [10 ** (-1.5 + 0.05 * i) for i in range(0, 61)]   # τ 0.03 ~ 30
WIN_HALF = 12.0                      # 글 시각 값의 때 창 ±12시간 (N은 하루 1% 안팎으로 변한다)
VALID_AT = ("post_time", "now", "day1", "yesterday", "end", "past_season", "unknown")


def shape_f(shape, u, tau):
    import numpy as np
    u = np.asarray(u, float)
    if shape == "linear":
        return u
    if shape == "log":
        return np.log1p(u / tau)
    return 1.0 - np.exp(-u / tau)


def load_reads(pattern, cand):
    """판독 결과 줄 → 값 목록 (후보 표의 시즌·글 시각과 짝). 반환 (값 목록, 없음 줄 수, 짝 못 찾은 줄 수)."""
    meta = {(r["k"], r["idx"]): r for r in cand}
    vals, none, lost = [], 0, 0
    for fn in sorted(glob.glob(pattern)):
        for l in open(fn, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            d = json.loads(l)
            m = meta.get((d.get("k"), d.get("idx")))
            if m is None:
                lost += 1
                continue
            if d.get("none"):
                none += 1
                continue
            d = dict(d)
            for key in ("pct", "n", "n_hi", "r", "season_ref"):   # 글자로 온 숫자("0.17%", "118,000", "S30")도 받는다
                v_ = d.get(key)
                if isinstance(v_, str):
                    m_ = re.search(r"\d[\d,]*\.?\d*", v_)
                    try:
                        x_ = float(m_.group().replace(",", "")) if m_ else None
                    except ValueError:
                        x_ = None
                    d[key] = (int(x_) if x_ is not None and key in ("n", "n_hi", "r", "season_ref") else x_)
            if d.get("basis") == "rank201_pct":
                # 보충 판독 (2026-11-01): N 없이 「201등 = p%」만 적은 측정 — N = r ÷ (pct/100)
                if d.get("pct") is None or not d.get("r") or d["pct"] <= 0:
                    none += 1
                    continue
                d["n"] = int(round(100.0 * d["r"] / d["pct"]))
                d["rel"] = d.get("rel") or "eq"
            elif d.get("n") is None:
                none += 1
                continue
            d["season"], d["h_post"], d["site"], d["src"] = m["season"], m["h"], m["site"], m["src"]
            ref = d.get("season_ref")
            d["sref"] = f"S{ref}" if isinstance(ref, int) else m["season"]
            vals.append(d)
    # 같은 (글, 시즌, 때, %)가 1차 줄(N을 적은 같은 측정)에도 있으면 보충 줄을 버리고 1차 줄을 남긴다
    first = {(v["k"], v["sref"], v.get("at"), v.get("pct")) for v in vals if v.get("basis") != "rank201_pct" and v.get("pct") is not None}
    keep, dup = [], 0
    for v in vals:
        if v.get("basis") == "rank201_pct" and (v["k"], v["sref"], v.get("at"), v.get("pct")) in first:
            dup += 1
            continue
        keep.append(v)
    return keep, none, lost, dup


def window_of(v, H):
    """값이 가리키는 때 [h0, h1] (시작부터 시간). 없으면 None."""
    at, own = v.get("at"), v["sref"] == v["season"]
    if v["sref"] != v["season"] or at == "past_season":
        return (0.0, H) if at not in ("day1", "end") else ((0.0, 24.0) if at == "day1" else (H - 24.0, H))
    if at == "day1":
        return (0.0, 24.0)
    if at == "end":
        return (H - 24.0, H)
    if at in ("post_time", "now"):
        return (v["h_post"] - WIN_HALF, v["h_post"] + WIN_HALF)
    if at == "yesterday":
        return (v["h_post"] - 24.0 - WIN_HALF, v["h_post"] - 24.0 + WIN_HALF)
    return None


def point_h(v, H):
    """② 시각 점 (시작부터 시간). day1은 0."""
    at = v.get("at")
    if v["sref"] != v["season"]:
        return 0.0 if at == "day1" else (H if at == "end" else None)
    return {"day1": 0.0, "end": H, "post_time": v["h_post"], "now": v["h_post"], "yesterday": v["h_post"] - 24.0}.get(at)


N_OTHER_SERVER = 250000              # 서버 미상인데 이 값 이상이면 다른 서버 규모 (일섭 약 44만 · 글섭 약 18만 — 18만은 한섭 범위와 겹쳐 못 거른다)


def select_server(vals, mode):
    """서버 거르기. main = server ∈ {KR, unknown} (명시 JP·GL·NA만 뺌) 이고 unknown인데 n ≥ 250,000이면 뺌 · KR_only = 명시 KR만.
    반환 (남은 값, 뺀 수 dict)."""
    keep, ex = [], collections.Counter()
    for v in vals:
        sv = v.get("server") or "unknown"
        if mode == "KR_only":
            if sv == "KR":
                keep.append(v)
            else:
                ex[f"server_{sv}"] += 1
        elif sv in ("JP", "GL", "NA"):
            ex[f"server_{sv}"] += 1
        elif sv != "KR" and v["n"] >= N_OTHER_SERVER:
            ex["unknown_n_ge_250000"] += 1
        else:
            keep.append(v)
    return keep, dict(ex)


def display_interval(v):
    """값 하나의 표시 단위 구간 (N의 [하한, 상한]). % 근거가 적혀 있으면 201 ÷ [p+0.01, p−0.005]%, 아니면 적은 자릿수의 반폭."""
    p = v.get("pct")
    if p is not None and p > P_LO + 1e-9 and v.get("basis") == "rank201_pct" and v.get("r"):
        return n_lo_of(v["r"], p), n_hi_of(v["r"], p), "pct"
    if p is not None and p > P_LO + 1e-9 and v.get("basis") in ("calc_201_pct", "pct_written"):
        return n_lo_of(201, p), n_hi_of(201, p), "pct"
    n = v["n"]
    unit = 1
    while unit < 100000 and n % (unit * 10) == 0:
        unit *= 10
    unit = max(unit, 1000)
    return n - unit / 2, n + unit / 2, "digits"


def bounds_in(bnd, s, win):
    """시즌 s, 때 창 win 안의 카드 구간 (lo, hi, 줄 수)과 확장 하한."""
    d = bnd["seasons"].get(s)
    if d is None or win is None:
        return None, None, 0, None
    E = [e for e in d["entries"] if win[0] <= e[0] <= win[1]]
    lo = max((e[2] for e in E if e[1] == "lo"), default=None)
    hi = min((e[2] for e in E if e[1] == "hi"), default=None)
    ext = max((e[1] for e in d.get("ext", []) if win[0] <= e[0] <= win[1]), default=None)
    return lo, hi, len(E), ext


def quant(v, qs=(0.25, 0.5, 0.75)):
    import numpy as np
    return [round(float(x), 4) for x in np.quantile(v, qs)] if len(v) else None


def level_stats(vals, Hs, ats=("day1",)):
    """① 한섭 첫날 값: 시즌마다 중앙, 시즌 사이 ln sd, 직전 시즌 대비 변화 sd."""
    import numpy as np
    by = collections.defaultdict(list)
    for v in vals:
        if v.get("rel", "eq") == "eq" and v.get("basis") != "assumed" and v.get("at") in ats:
            by[v["sref"]].append(v["n"])
    per = {s: {"n_values": len(x), "distinct": len(set(x)), "min": min(x), "median": float(np.median(x)), "max": max(x)} for s, x in by.items()}
    seasons = sorted(per, key=lambda t: int(t[1:]))
    ln = {s: math.log(per[s]["median"]) for s in seasons}
    out = {"per_season": {s: per[s] for s in seasons}, "n_seasons": len(seasons)}
    if len(seasons) >= 2:
        out["ln_sd_between_seasons"] = round(float(np.std(list(ln.values()), ddof=1)), 4)
        d = [ln[b] - ln[a] for a, b in zip(seasons, seasons[1:]) if int(b[1:]) == int(a[1:]) + 1]
        out["n_consecutive_pairs"] = len(d)
        if len(d) >= 2:
            out["ln_change_mean"] = round(float(np.mean(d)), 4)
            out["ln_change_sd"] = round(float(np.std(d, ddof=1)), 4)
    return out, ln


def shape_fit(vals, Hs, level_ln):
    """② 시즌 안 꼴. y = ln(N(u) ÷ 첫날 N), u = h/H. 시즌 s마다 크기 a_s, τ 공통. 점은 한섭 값(첫날이 아닌 때)."""
    import numpy as np
    pts = collections.defaultdict(list)
    for v in vals:
        if v.get("rel", "eq") != "eq" or v.get("basis") == "assumed" or v["sref"] not in level_ln:
            continue
        h = point_h(v, Hs[v["sref"]])
        if h is None or v.get("at") == "day1":
            continue
        u = h / Hs[v["sref"]]
        if 0.0 < u <= 1.0:
            pts[v["sref"]].append((u, math.log(v["n"]) - level_ln[v["sref"]]))
    n_pts = sum(len(x) for x in pts.values())
    out = {"n_points": n_pts, "n_seasons": len(pts), "points_per_season": {s: len(x) for s, x in sorted(pts.items())}}
    if n_pts == 0:
        return out, {}
    fits = {}
    for shape in SHAPES:
        best = None
        for tau in ([1.0] if shape == "linear" else TAU_GRID):
            sse, a = 0.0, {}
            for s, L in pts.items():
                u = np.array([x[0] for x in L])
                y = np.array([x[1] for x in L])
                f = shape_f(shape, u, tau)
                den = float((f * f).sum())
                a_s = float((f * y).sum() / den) if den > 0 else 0.0
                a[s] = a_s
                sse += float(((y - a_s * f) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, tau, a)
        sse, tau, a = best
        k = len(pts) + (0 if shape == "linear" else 1)
        aic = n_pts * math.log(max(sse, 1e-12) / n_pts) + 2 * k
        g = {s: a_s * float(shape_f(shape, [1.0], tau)[0]) for s, a_s in a.items()}
        fits[shape] = {"tau": None if shape == "linear" else round(tau, 4), "sse": round(sse, 5), "k": k, "aic": round(aic, 2),
                       "growth_at_end_ln": {s: round(x, 4) for s, x in sorted(g.items())},
                       "growth_sd_between_seasons": round(float(np.std(list(g.values()), ddof=1)), 4) if len(g) >= 2 else None,
                       "growth_median": round(float(np.median(list(g.values()))), 4)}
    out["fits"] = fits
    out["identifiable"] = bool(n_pts > len(pts) + 1 and any(len(x) >= 2 for x in pts.values()))
    out["best_by_aic"] = min(fits, key=lambda k: fits[k]["aic"]) if out["identifiable"] else None
    best = out["best_by_aic"] or "linear"
    return out, fits[best]["growth_at_end_ln"]


def trust_stats(vals, bnd, Hs):
    """③ 커뮤니티 값(한섭) 대 우리 자료 구간."""
    rows, bins = [], collections.Counter()
    for v in vals:
        if v.get("rel", "eq") == "le":
            continue
        win = window_of(v, Hs.get(v["sref"], 120.0))
        if win is None or v["sref"] not in bnd["seasons"]:
            bins["no_window_or_season"] += 1
            continue
        lo, hi, ne, ext = bounds_in(bnd, v["sref"], win)
        dlo, dhi, dkind = display_interval(v)
        r = {"sref": v["sref"], "at": v.get("at"), "timed": v.get("at") in ("day1", "now", "post_time", "end", "yesterday"), "basis": v.get("basis"), "copied": v.get("copied"), "n": v["n"], "rel": v.get("rel", "eq"),
             "card_lo": lo, "card_hi": hi, "card_n": ne, "ext_lo": ext, "disp": [round(dlo), round(dhi) if dhi else None], "disp_kind": dkind,
             "disp_ln_width": round(math.log(dhi / dlo), 4) if dhi and dlo > 0 else None}
        r["in_card"] = None if ne == 0 else bool((lo is None or v["n"] >= lo) and (hi is None or v["n"] <= hi))
        if lo is not None and v["n"] < lo:
            r["ln_out"] = round(math.log(lo / v["n"]), 4)
        elif hi is not None and v["n"] > hi:
            r["ln_out"] = round(math.log(v["n"] / hi), 4)
        else:
            r["ln_out"] = 0.0 if ne else None
        r["ge_ext"] = None if ext is None else bool(v["n"] >= ext)
        r["ln_over_ext"] = None if ext is None else round(math.log(v["n"] / ext), 4)
        rows.append(r)
    ev = [r for r in rows if r["rel"] == "eq"]
    ck = [r for r in ev if r["in_card"] is not None]
    ex = [r for r in ev if r["ge_ext"] is not None]
    out = {"n_values": len(rows), "n_eq": len(ev), "with_card_bounds": len(ck),
           "in_card_share": round(sum(r["in_card"] for r in ck) / len(ck), 4) if ck else None,
           "ln_out_quartiles": quant([r["ln_out"] for r in ck if r["ln_out"]]) if any(r["ln_out"] for r in ck) else None,
           "timed_only": {"with_card_bounds": sum(1 for r in ck if r["timed"]),
                          "in_card_share": round(sum(r["in_card"] for r in ck if r["timed"]) / max(sum(1 for r in ck if r["timed"]), 1), 4),
                          "ge_ext_share": round(sum(r["ge_ext"] for r in ex if r["timed"]) / max(sum(1 for r in ex if r["timed"]), 1), 4),
                          "note": "때 창이 좁은 값만 (past_season·unknown은 시즌 전체 창이라 N이 늘어난 끝쪽 하한과 견줘져 벗어남이 부풀 수 있다)"},
           "with_ext_lower": len(ex), "n_ge_ext_lower_share": round(sum(r["ge_ext"] for r in ex) / len(ex), 4) if ex else None,
           "ln_over_ext_quartiles": quant([r["ln_over_ext"] for r in ex]),
           "by_basis": {}, "rows": rows}
    for b in sorted({r["basis"] for r in ev}, key=str):
        L = [r for r in ev if r["basis"] == b]
        Lc = [r for r in L if r["in_card"] is not None]
        out["by_basis"][str(b)] = {"n": len(L), "in_card_share": round(sum(r["in_card"] for r in Lc) / len(Lc), 4) if Lc else None, "n_card": len(Lc)}
    cp = [r for r in ev if r["copied"] == "yes"]
    out["copied_yes"] = len(cp)
    dsp = [r["disp_ln_width"] for r in ev if r["disp_kind"] == "pct" and r["disp_ln_width"]]
    out["disp_pct_ln_width_median"] = round(float(sorted(dsp)[len(dsp) // 2]), 4) if dsp else None
    return out


def spearman(x, y):
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None
    if len(x) < 3:
        return {"rho": None, "n": len(x)}
    r = spearmanr(x, y)
    return {"rho": round(float(r.statistic), 4), "n": len(x), "p": round(float(r.pvalue), 4)}


def cmd_summary(a):
    cand = [json.loads(l) for l in open(os.path.join(a.w, "cand.jsonl"), encoding="utf-8")]
    bnd = json.load(open(os.path.join(a.w, "bounds.json"), encoding="utf-8"))
    slp = json.load(open(os.path.join(a.w, "slope.json"), encoding="utf-8"))
    Hs = {s: C.info(s)[1] for s in SEASONS}
    vals, none, lost, dup = load_reads(a.read, cand) if a.read else ([], 0, 0, 0)
    out = {"made": "nr_pop.py summary", "data_ver": C.DATA_VER,
           "cand": {"windows": len(cand), "posts": len({r["k"] for r in cand}),
                    "by_src": dict(collections.Counter(r["src"] for r in cand)),
                    "by_season": {s: sum(1 for r in cand if r["season"] == s) for s in SEASONS}},
           "reads": {"values": len(vals), "none_rows": none, "unmatched_rows": lost, "rank201_dup_dropped": dup,
                     "by_basis": dict(collections.Counter(v.get("basis") for v in vals)),
                     "by_server": dict(collections.Counter(v.get("server") or "unknown" for v in vals)),
                     "n_ge_250000_by_server": dict(collections.Counter(v.get("server") or "unknown" for v in vals if v["n"] >= N_OTHER_SERVER))}}
    out["bounds"] = {"total": bnd["total"], "premise": bnd.get("premise"), "p_display": bnd["p_display"], "rank_range": bnd["rank_range"], "seasons": {}}
    for s, d in bnd["seasons"].items():
        ent = d["entries"]
        keep = []
        for side in ("lo", "hi"):
            for basis in ("a201", "a", "b"):
                L = sorted((e for e in ent if e[1] == side and e[3] == basis), key=lambda e: -e[2] if side == "lo" else e[2])[:3]
                keep += L
        out["bounds"]["seasons"][s] = {"H": d["H"], "counts": d["counts"], "kinds": d["kinds"], "kinds_wide_50_200": d["kinds_wide_50_200"], "bins": d["bins"],
                                       "top_entries": keep}
    out["bounds"]["entry_cols"] = bnd["entry_cols"]
    out["slope"] = slp
    pd = json.load(open(os.path.join(C.ROOT, "data", "solo-raid", "analysis", "prior_dev.json"), encoding="utf-8"))["transform"]["season_levels"]

    def block(sel):
        lv, level_ln = level_stats(sel, Hs)
        sh, growth = shape_fit(sel, Hs, level_ln)
        b = {"n_values": len(sel), "per_season_values": dict(sorted(collections.Counter(v["sref"] for v in sel).items(), key=lambda t: int(t[0][1:]))),
             "level": lv, "level_any_time": level_stats(sel, Hs, ("day1", "past_season", "end", "now", "post_time"))[0], "shape": sh,
             "trust": trust_stats(sel, bnd, Hs)}
        # 6. 정답 쪽 값 (서술) — 고르기에 안 씀
        ss = [x for x in sorted(growth, key=lambda t: int(t[1:])) if x in pd]
        b["spearman_growth_vs_ln_m"] = spearman([growth[x] for x in ss], [math.log(pd[x]) for x in ss])
        b["spearman_growth_vs_ln_m"]["seasons"] = ss
        return b

    if vals:
        sel, ex = select_server(vals, "main")
        out["main"] = block(sel)
        out["main"]["excluded"] = ex
        selk, exk = select_server(vals, "KR_only")
        out["KR_only"] = block(selk)
        out["KR_only"]["excluded"] = exk
    else:
        out["main"] = out["KR_only"] = None
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(a.out or OUT, "pop.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out or OUT, "pop.md"), "w", encoding="utf-8") as f:
        f.write(render_md(out))
    print("→", os.path.join(a.out or OUT, "pop.json"), "·", os.path.join(a.out or OUT, "pop.md"))
    return out


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(o):
    L = ["# WP8a 인구 기록표 (2026-11-01, `lab7/nr_pop.py summary`)", "",
         "글 번호·원문·닉네임은 없다 (열쇠는 sha1 앞 12자). 판독 전 판이면 ①②③⑥은 빈칸이다.", ""]
    c = o["cand"]
    L += [f"## 후보 (정규식은 코드 머리에 고정)", "", f"창 {c['windows']}개, 글 {c['posts']}개, 출처별 {c['by_src']}", "",
          "시즌별 창 수: " + " ".join(f"{s[1:]}:{n}" for s, n in c["by_season"].items()), ""]
    r = o["reads"]
    L += [f"판독 값 {r['values']}줄 (없음 줄 {r['none_rows']}, 짝 못 찾은 줄 {r['unmatched_rows']})", ""]
    b = o["bounds"]
    t = b["total"]
    L += ["## 3. 우리 자료로 낸 N 구간 (한섭 사진 A·B, exclude 빔; 끝난 시즌 전체 표)", "",
          f"표시 구간 [p{b['p_display'][0]:+.3f}, p+{b['p_display'][1]:.2f}) · (b)의 등수 줄 {b['rank_range'][0]}~{b['rank_range'][1]}위", "",
          f"- 근거별 줄 수 (시즌 합): {({k: v for k, v in t.items() if k[:2] in ('a_', 'a2', 'b_')})}",
          f"- 값이 있는 시즌 {t.get('seasons_with_card')}/24 (하한 {t.get('seasons_with_lo')}, 상한 {t.get('seasons_with_hi')}), 200위→% 짝(a201)이 있는 시즌 {t.get('seasons_with_a201')}",
          f"- 120~200위 등수 줄 {t.get('n120_200_rows')}개 (50~200위로 넓히면 {t.get('n50_200_rows')}개, (b)하한 {t.get('wide_b_lo')} · 상한 {t.get('wide_b_hi')} — 보고만)",
          f"- 하한 > 상한 칸: {t.get('conflict_bins')}", ""]
    pr = b.get("premise")
    if pr:
        L += ["### ④ 전제 점검 (서술)", "",
              f"- 「% 줄 = 숫자단 밖」: 시험할 수 있는 짝 {pr['b_hi_testable_120_200']} (50~200위 {pr['b_hi_testable_50_200']}) 중 모순 {pr['b_hi_violations_120_200']} ({pr['b_hi_violations_50_200']})",
              f"- 같은 사람·같은 점수의 이웃 % 짝 ({pr['unit']}): {pr['same_score_pairs']}",
              f"- 같은 짝의 ln p 변화(하루당) 앞 절반 {pr['dlnp_per_day_early']} · 뒤 절반 {pr['dlnp_per_day_late']}", ""]
    L += ["### 시즌별 24시간 칸의 가장 센 하한 (단위 천 명; 카드 a·b / 확장 c201 = 모든 % 줄은 201위 밖)", "",
          "| 시즌 | 120~200 줄 | a201·a·b 줄 | 칸별 카드 하한 / 확장 하한 |", "|---|---|---|---|"]
    for s, d in b["seasons"].items():
        cells = " ".join(f"{(x['lo'] or 0) / 1000:.0f}/{(x['lo_ext'] or 0) / 1000:.0f}" for x in d["bins"])
        L.append(f"| {s[1:]} | {d['counts'].get('rank_rows_120_200')} | {d['kinds']} | {cells} |")
    L.append("")
    sl = o["slope"]
    m = sl["main"]
    L += ["## 4. 번짐 크기 (끝 24시간 % 0.08~0.6, −d ln 점수 / d ln %)", "",
          f"- 본안 (한섭·A B·8줄 이상): 시즌 {m['n_seasons']}개, 중앙 {fmt(m.get('median'), 4)}, 범위 {fmt(m.get('min'), 4)}~{fmt(m.get('max'), 4)}, 시즌 붓스트랩 90% {m.get('boot_median_90')}"]
    for k, v in sl["variants"].items():
        L.append(f"- 어림 {k}: 시즌 {v['n_seasons']}개, 중앙 {fmt(v.get('median'), 4)}, 범위 {fmt(v.get('min'), 4)}~{fmt(v.get('max'), 4)}, 90% {v.get('boot_median_90')}")
    L.append("")
    if o["main"]:
        L += ["## 서버 거르기 (정리 ①②③⑥에 쓰는 값)", "",
              "- **본안**: server ∈ {KR, unknown} (명시 JP·GL·NA만 뺌). 근거: WP1 1e에서 디시 서버 미상 사진 줄의 한섭 몫 93%, 모은 글이 한국 커뮤니티. "
              f"서버 미상인데 n ≥ {N_OTHER_SERVER:,}이면 다른 서버 규모(일섭 약 44만 등)라 뺀다. **글섭 약 18만이라는 값은 한섭 범위(11~14만대)와 겹쳐 서버가 안 적히면 거르지 못한다** — 본안에 일부 섞여 있을 수 있다.",
              "- **KR_only**: 명시 KR만 (같은 셈, 견줌).", "",
              f"판독 {o['reads']['values']}값 (보충 `rank201_pct`와 1차 같은 측정은 1차만 남김: 중복 {o['reads']['rank201_dup_dropped']}줄 버림). 근거별 {o['reads']['by_basis']}, 서버별 {o['reads']['by_server']}, "
              f"n ≥ 250,000 서버별 {o['reads']['n_ge_250000_by_server']}", ""]
        for name, key in (("본안 (KR + 미상)", "main"), ("KR_only", "KR_only")):
            B = o[key]
            lv, sh, tr, sp = B["level"], B["shape"], B["trust"], B["spearman_growth_vs_ln_m"]
            L += [f"## {name}", "", f"값 {B['n_values']}개 (뺀 것 {B['excluded']}), 시즌별 값 수: " + " ".join(f"{s[1:]}:{n}" for s, n in B["per_season_values"].items()), "",
                  f"### 5-①. 수준 (한섭 첫날 값, eq만)", "",
                  f"시즌 {lv['n_seasons']}개, 시즌 사이 ln sd {fmt(lv.get('ln_sd_between_seasons'), 4)}, 직전 시즌 대비 ln 변화 평균 {fmt(lv.get('ln_change_mean'), 4)} sd {fmt(lv.get('ln_change_sd'), 4)} (이웃 시즌 짝 {lv.get('n_consecutive_pairs')}개)",
                  f"참고 (때 불문 시즌 값): 시즌 {B['level_any_time']['n_seasons']}개, ln sd {fmt(B['level_any_time'].get('ln_sd_between_seasons'), 4)}, 변화 sd {fmt(B['level_any_time'].get('ln_change_sd'), 4)}", "",
                  "| 시즌 | 값 수 | 서로 다른 값 | 최소 | 중앙 | 최대 |", "|---|---|---|---|---|---|"]
            for s_, d in lv["per_season"].items():
                L.append(f"| {s_[1:]} | {d['n_values']} | {d['distinct']} | {d['min']:.0f} | {d['median']:.0f} | {d['max']:.0f} |")
            L += ["", "### 5-②. 시즌 안 꼴", "",
                  f"점 {sh['n_points']}개 ({sh['n_seasons']}시즌, 점 2개 이상인 시즌 {sum(1 for v in sh.get('points_per_season', {}).values() if v >= 2)}), 식별 가능 {sh.get('identifiable')}, AIC로 {sh.get('best_by_aic')}"]
            for k, f_ in (sh.get("fits") or {}).items():
                L.append(f"- {k}: τ {f_['tau']}, SSE {f_['sse']}, k {f_['k']}, AIC {f_['aic']}, 끝 증가 ln 중앙 {f_['growth_median']} sd {f_['growth_sd_between_seasons']}")
            L += ["", "### 5-③. 신뢰도", "",
                  f"값 {tr['n_values']}개 (eq {tr['n_eq']}), 카드 구간(=하한)이 있는 값 {tr['with_card_bounds']}개 중 안에 든(하한 이상) 비율 {fmt(tr['in_card_share'], 3)}, 벗어난 크기 ln 사분위 {tr['ln_out_quartiles']}",
                  f"확장 하한(201÷최소 %) 이상인 비율 {fmt(tr['n_ge_ext_lower_share'], 3)} ({tr['with_ext_lower']}개), ln(값÷하한) 사분위 {tr['ln_over_ext_quartiles']}",
                  f"때 창이 좁은 값만: 카드 구간이 있는 값 {tr['timed_only']['with_card_bounds']}개 중 하한 이상 {tr['timed_only']['in_card_share']}, 확장 하한 이상 {tr['timed_only']['ge_ext_share']} (past_season·unknown은 시즌 전체 창이라 끝쪽 하한과 견줘져 벗어남이 부풀 수 있다)",
                  f"근거별 {tr['by_basis']}", f"% 근거 값의 표시 한 칸 ln 폭 중앙 {fmt(tr['disp_pct_ln_width_median'], 4)}", "",
                  "### 6. 서술: 시즌 안 인구 증가 크기와 침수 크기 ln m_s (정답 쪽 값 — 고르기에 안 씀)", "", f"Spearman {sp}", ""]
    else:
        L += ["## 5-① ~ ③, 6", "", "판독 결과가 아직 없다 (`--read 'W/read_*.jsonl'`).", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cand", "bounds", "slope", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--w", required=True, help="세션 임시 폴더 (cand.jsonl·obs7full·bounds.json·slope.json)")
        if name == "summary":
            p.add_argument("--read", default=None, help="판독 결과 JSONL 묶음 (예: 'W/read_*.jsonl')")
            p.add_argument("--out", default=None, help="기본: r7/nr/")
    a = ap.parse_args()
    {"cand": cmd_cand, "bounds": cmd_bounds, "slope": cmd_slope, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
