"""OCR 원시 결과에서 순위·%·점수 줄과 화면 종류를 규칙으로 뽑는다 (모델 0).

python3 -I tools/parse_ocr.py OCR.jsonl OUT.jsonl [OCR2.jsonl]

OCR2(확대 두 번째 판)가 있으면 같은 사진의 두 결과 중 나은 쪽을 쓴다: 자신 있음 > 줄 수 많음.

한 줄 출력: id, screen, conf(자신 있음), why(자신 없는 이유), rows=[rank, pct, score, self], hit(코드 선별 신호)
화면 종류:
- season_end  시즌 최종 순위 (RAID RESULT) — %·점수
- result_pop  결과 팝업 (ENEMY DEFEATED, 나의 최고 기록) — 점수 + 순위(2ND·44TH) 또는 %
- my_rank     내 순위 화면 (상위 X%, N위)
- rank_list   랭킹 목록 (11~12자리 점수가 세로로 여럿)
- deck        덱별 결과 (SQUAD·DAMAGE·전투기록) → 음성
- mock        모의전·계산기 → 음성
  (순위 팝업이면 deck·mock으로 정하지 않는다: 「랭킹 정보」·「전투 결과에 따라」 글자, 또는 모의전 근거가 「모의」 버튼뿐이고
   「솔로레이드」 글자와 100억 이상 점수가 둘 이상. S41 F2에서 진짜 순위 팝업 4장 36줄이 이 때문에 빠졌다. 2026-10-16)
- other_raid  협동작전·유니온 레이드 → 음성
- none        모름
"""
import collections as C
import json
import re
import sys

PCT = re.compile(r"(?<![\d.])(\d{1,2}\.\d{1,2}|\d{1,3})\s*%")
SCORE = re.compile(r"(?<!\d)(\d{1,3}(?:[,.\s]\d{3}){2,3})(?![\d])")
STRICT = re.compile(r"^\d{1,3}(?:,\d{3}){3}$")  # 11~12자리 콤마 정상
EOK = re.compile(r"(\d{1,4}(?:\.\d{1,2})?)\s*억")
ORD = re.compile(r"^\(?(\d{1,6})\s*(?:ST|ND|RD|TH)$", re.I)
WI = re.compile(r"(?<!\d)(\d{1,6})\s*위")
TOP = re.compile(r"상위\s*(\d{1,2}(?:\.\d{1,2})?)\s*%")
OCR_MIN = 0.7

KW = {
    "mock": ("모의", "계산", "시뮬", "주기", "MOCK", "SIMUL", "억/초", "기여"),
    "other_raid": ("협동", "유니온", "UNION", "COOP"),
    "deck": ("SQUAD", "DAMAGE", "전투기록", "전투 기록", "총교전", "TOTAL"),
    "result_pop": ("DEFEATED", "최고기록", "최고 기록"),
    "rank": ("상위", "순위", "랭킹", "RANK", "기준", "솔로레이드"),
    "season_end": ("RAIDRESULT", "최종순위"),
    # 순위 팝업 고유 글자. 아래에 모의전 버튼·덱 정보가 함께 찍혀도 mock·deck으로 버리지 않는다 (2026-10-16, F2 놓친 줄)
    "rank_pop": ("랭킹정보", "전투결과에따라"),
}
BROKEN = re.compile(r"\d[\d,.\s]{6,}\d")  # 숫자 7자리 이상이 구분자와 섞인 덩어리


def norm(t):
    t = t.replace("，", ",").replace("．", ".").replace("％", "%")
    # 숫자 사이 O·o → 0
    t = re.sub(r"(?<=\d)[Oo](?=[\d,.])|(?<=[\d,.])[Oo](?=\d)", "0", t)
    return t


def has(text, key):
    return any(k in text for k in KW[key])


def lines(items):
    """세로 위치가 겹치는 글자 상자끼리 한 줄로 묶는다."""
    its = sorted(items, key=lambda x: (x[0][1] + x[0][3]) / 2)
    out = []
    for it in its:
        cy, h = (it[0][1] + it[0][3]) / 2, it[0][3] - it[0][1]
        if out and abs(cy - out[-1]["cy"]) <= max(h, out[-1]["h"]) * 0.6:
            out[-1]["items"].append(it)
        else:
            out.append({"cy": cy, "h": h, "items": [it]})
    for ln in out:
        ln["items"].sort(key=lambda x: x[0][0])
    return out


def score_of(t):
    """(정수 점수, 형식이 정상인가) 또는 None."""
    m = SCORE.search(t)
    if not m:
        return None
    raw = m.group(1)
    digits = re.sub(r"\D", "", raw)
    if not 10 <= len(digits) <= 12:
        return None
    return int(digits), bool(STRICT.match(raw))


def parse(rec):
    items = [[b, norm(t), float(c)] for b, t, c in rec.get("items") or []]
    text = " ".join(t for _, t, _ in items)
    flat = text.upper().replace(" ", "")
    hit = bool(PCT.search(text)) or any(len(re.sub(r"\D", "", t)) >= 9 for _, t, _ in items)
    out = {"id": rec["id"], "screen": "none", "conf": False, "why": [], "rows": [], "hit": hit}
    if not items:
        out["why"].append("no_text")
        return out
    # 순위 팝업: 고유 글자, 또는 (제목을 못 읽었을 때) 모의전 근거가 아래쪽 「모의전」 버튼뿐이고 솔로레이드 화면에 100억대 점수가 둘 이상
    big_n = sum(1 for _, t, _ in items if (score_of(t) or (0,))[0] >= 10 ** 10)
    btn_only = [k for k in KW["mock"] if k in text or k in flat] == ["모의"]
    pop = has(flat, "rank_pop") or (btn_only and "솔로레이드" in flat and big_n >= 2)
    if not pop and (has(text, "mock") or has(flat, "mock")):
        out["screen"] = "mock"
    elif has(text, "other_raid") or has(flat, "other_raid"):
        out["screen"] = "other_raid"
    elif has(flat, "season_end"):
        out["screen"] = "season_end"
    elif has(flat, "result_pop"):
        out["screen"] = "result_pop"
    elif TOP.search(text) or ("기준" in text and WI.search(text) and EOK.search(text)):
        out["screen"] = "my_rank"
    elif has(flat, "deck") and not PCT.search(text) and not pop:
        out["screen"] = "deck"
    if out["screen"] in ("mock", "other_raid", "deck"):
        out["conf"] = True
        return out

    broken = [t for _, t, _ in items if BROKEN.search(t) and not score_of(t) and not EOK.search(t)
              and len(re.sub(r"\D", "", t)) >= 7]
    low = [t for _, t, c in items if c < OCR_MIN and (PCT.search(t) or score_of(t) or ORD.match(t.strip()))]
    bad_fmt = False
    rows = []
    if out["screen"] in ("result_pop", "season_end"):
        sc = [score_of(t) for _, t, _ in items]
        sc = [s for s in sc if s]
        r = {"rank": None, "pct": None, "score": None, "self": True}
        if sc:
            r["score"] = sc[0][0]
            bad_fmt |= not sc[0][1] and not any(s[1] for s in sc)
            if len({s[0] for s in sc}) > 1:
                out["why"].append("score_disagree")
        for _, t, _ in items:
            m = ORD.match(t.strip())
            if m and r["rank"] is None:
                r["rank"] = int(m.group(1))
            m = PCT.search(t)
            if m and r["pct"] is None and "." in m.group(1):
                r["pct"] = float(m.group(1))
        if r["score"] is not None or r["pct"] is not None:
            rows.append(r)
    elif out["screen"] == "my_rank":
        r = {"rank": None, "pct": None, "score": None, "self": True, "score_eok": None}
        m = TOP.search(text)
        if m:
            r["pct"] = float(m.group(1))
        m = WI.search(text)
        if m:
            r["rank"] = int(m.group(1))
        sc = [score_of(t) for _, t, _ in items]
        sc = [s for s in sc if s]
        if sc:
            r["score"] = sc[0][0]
        else:
            m = EOK.search(text)
            if m:
                r["score"] = round(float(m.group(1)) * 1e8)
        rows.append(r)
    else:
        # 랭킹 목록: 점수 상자를 기준으로, 세로로 가장 많이 겹치는 왼쪽 순위·% 상자를 붙인다
        def ov(a, b):
            return max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(1, min(a[3] - a[1], b[3] - b[1]))
        W = rec.get("w", 850)
        ranks = []
        pcts = []
        for b, t, c in items:
            m = ORD.match(t.strip()) or re.fullmatch(r"#?(\d{1,6})", t.strip())
            if m and b[0] < W * 0.25:
                ranks.append((b, int(m.group(1))))
            m = PCT.search(t)
            if m and "." in m.group(1):
                pcts.append((b, float(m.group(1))))
        used = set()
        for b, t, c in items:
            sc = score_of(t)
            if not sc:
                continue
            bad_fmt |= not sc[1]
            r = {"rank": None, "pct": None, "score": sc[0], "self": False}
            for key, cand in (("rank", ranks), ("pct", pcts)):
                best = max(((ov(b, cb), i) for i, (cb, v) in enumerate(cand) if cb[0] < b[0]), default=(0, None))
                if best[0] >= 0.5 and (key, best[1]) not in used:
                    used.add((key, best[1]))
                    r[key] = cand[best[1]][1]
            r["self"] = r["pct"] is not None
            rows.append(r)
        for i, (cb, v) in enumerate(pcts):
            if ("pct", i) not in used:
                rows.append({"rank": None, "pct": v, "score": None, "self": True})
        # 같은 점수가 두 번(목록 줄 + 아래 내 줄)이면 하나로 합친다
        merged = {}
        for r in rows:
            k = r["score"] if r["score"] is not None else ("pct", r["pct"])
            if k in merged:
                for f in ("rank", "pct"):
                    merged[k][f] = merged[k][f] if merged[k][f] is not None else r[f]
                merged[k]["self"] |= r["self"]
            else:
                merged[k] = r
        rows = list(merged.values())
        scs = [r["score"] for r in rows if r["score"] is not None]
        all_sc = [score_of(t)[0] for _, t, _ in items if score_of(t)]
        if len(all_sc) >= 3 and max(C.Counter(all_sc).values()) >= 3:
            out["screen"] = "other_raid"  # 같은 점수가 여러 줄: 유니온 레이드 목록
            out["conf"] = True
            out["rows"] = []
            return out
        big = [r for r in rows if (r["score"] or 0) >= 10 ** 10]  # 솔로레이드 총점은 100억대 이상, 덱별 피해는 대개 10자리
        medal = bool(re.search(r"(?<![A-Z])(ST|ND|RD)(?![A-Z])", " ".join(t.upper() for _, t, _ in items)))
        if rows and (has(flat, "rank") or sum(1 for r in rows if r["rank"] is not None) >= 2
                     or len(big) >= 3 or (len(big) >= 2 and medal)):
            out["screen"] = "rank_list"
        elif not has(flat, "deck") and len(rows) <= 2 and any(r["pct"] is not None for r in rows) and big:
            # %와 11자리 점수만 있는 잘린 내 줄
            pr = [r for r in rows if r["pct"] is not None][0]
            if pr["score"] is None:
                pr["score"] = big[0]["score"]
            rows = [pr]
            out["screen"] = "my_rank"
        elif rows:
            out["why"].append("screen_unknown")
    out["rows"] = rows
    if out["screen"] == "none":
        out["why"].append("screen_unknown")
    if not rows and out["screen"] != "none":
        out["why"].append("no_rows")
    if broken:
        out["why"].append("score_broken")
    if bad_fmt:
        out["why"].append("score_format")
    if low:
        out["why"].append("low_ocr")
    out["conf"] = not out["why"]
    return out


def better(a, b):
    return b if (b["conf"], len(b["rows"])) > (a["conf"], len(a["rows"])) else a


def main():
    src, dst = sys.argv[1:3]
    second = {}
    if len(sys.argv) > 3:
        second = {r["id"]: r for r in map(json.loads, open(sys.argv[3], encoding="utf-8"))}
    with open(dst, "w", encoding="utf-8") as f:
        for l in open(src, encoding="utf-8"):
            rec = json.loads(l)
            p = parse(rec)
            if rec["id"] in second:
                q = parse(second[rec["id"]])
                p = better(p, q)
                p["pass2"] = p is q
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
