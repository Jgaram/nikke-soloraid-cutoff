"""0단계: 있는 자료로 계산만 한다 (모델 0, 요청 0).

python3 -I tools/step0.py   (실험 폴더에서)

입력: ../2026-10-15-s41-opus-blind/data/
출력: data/step0.json (지표), data/orig_diff.jsonl (1번 차이 줄),
      data/multi_pct_posts.jsonl (3번 여러 % 글), data/evalset.jsonl (4번)
"""
import collections as C
import hashlib
import json
import os
import random
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "..", "2026-10-15-s41-opus-blind", "data")
OUT = os.path.join(HERE, "data")
KEY_SCREENS = ("my_rank", "rank_list", "season_end")
EOK = 100_000_000


def load(name):
    return [json.loads(l) for l in open(os.path.join(SRC, name), encoding="utf-8") if l.strip()]


def base(i):
    return i.split("#")[0]


def write_jsonl(name, rows):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- 1. 원본 재판독이 필요한가 ----------

def merge_reads(rs):
    """조각(#p) 판독을 사진 하나로 합친다. 같은 (rank, pct, score) 줄은 하나만."""
    out = C.defaultdict(lambda: {"screens": [], "rows": []})
    for r in rs:
        k = base(r["id"])
        out[k]["screens"].append(r.get("screen"))
        seen = {(x.get("rank"), x.get("pct"), x.get("score")) for x in out[k]["rows"]}
        for x in r.get("rows") or []:
            t = (x.get("rank"), x.get("pct"), x.get("score"))
            if any(v is not None for v in t) and t not in seen:
                seen.add(t)
                out[k]["rows"].append(x)
    return out


def pair_rows(a, b):
    """같은 사진의 두 판독 줄을 짝짓는다: 순위 → (self, %) → 가까운 점수 순."""
    a, b = list(a), list(b)
    pairs = []
    for key in (lambda x: x.get("rank"),
                lambda x: (x.get("self"), x.get("pct")) if x.get("pct") is not None else None):
        for x in list(a):
            kx = key(x)
            if kx is None:
                continue
            m = [y for y in b if key(y) == kx]
            if len(m) == 1 and sum(1 for z in a if key(z) == kx) == 1:
                pairs.append((x, m[0]))
                a.remove(x)
                b.remove(m[0])
    for x in sorted(a, key=lambda r: -(r.get("score") or 0)):
        if x.get("score") is None:
            continue
        c = [y for y in b if y.get("score") is not None]
        if not c:
            break
        y = min(c, key=lambda y: abs(y["score"] - x["score"]))
        if abs(y["score"] - x["score"]) < 10 * EOK:  # 10억 넘게 다르면 다른 줄로 본다
            pairs.append((x, y))
            a.remove(x)
            b.remove(y)
    return pairs, a, b


def digit_pos(s1, s2):
    """두 정수에서 처음 다른 자리(오른쪽부터 1=일의 자리). 11자리면 11=백억 자리."""
    t1, t2 = str(s1), str(s2)
    if len(t1) != len(t2):
        return max(len(t1), len(t2))
    for i, (c1, c2) in enumerate(zip(t1, t2)):
        if c1 != c2:
            return len(t1) - i
    return 0


def step1(first, orig):
    fm, om = merge_reads(first), merge_reads(orig)
    need = {base(r["id"]) for r in first if r.get("need_orig")}
    st = C.Counter()
    pos = C.Counter()
    diffs = []
    for k in sorted(om):
        oscr = [s for s in om[k]["screens"] if s]
        if not any(s in KEY_SCREENS for s in oscr):
            continue
        st["images"] += 1
        st["images_need_orig"] += k in need
        if k not in fm:
            st["no_first"] += 1
            continue
        pairs, ua, ub = pair_rows(fm[k]["rows"], om[k]["rows"])
        st["rows_orig"] += len(om[k]["rows"])
        st["pairs"] += len(pairs)
        st["only_first"] += len(ua)
        st["only_orig"] += len(ub)
        for x, y in pairs:
            if x.get("rank") is not None and y.get("rank") is not None:
                st["rank_both"] += 1
                if x["rank"] != y["rank"]:
                    st["rank_diff"] += 1
                    diffs.append({"id": k, "what": "rank", "first": x.get("rank_txt"), "orig": y.get("rank_txt")})
            if x.get("pct") is not None and y.get("pct") is not None:
                st["pct_both"] += 1
                if abs(x["pct"] - y["pct"]) > 1e-9:
                    st["pct_diff"] += 1
                    diffs.append({"id": k, "need_orig": k in need, "what": "pct", "first": x.get("pct_txt"), "orig": y.get("pct_txt")})
            if x.get("score") is not None and y.get("score") is not None:
                st["score_both"] += 1
                d = abs(x["score"] - y["score"])
                if d == 0:
                    st["score_same"] += 1
                    continue
                p = digit_pos(x["score"], y["score"])
                pos[p] += 1
                if d >= EOK // 10:
                    st["score_ge_0.1eok"] += 1
                    st["score_ge_0.1eok_need_orig"] += k in need
                    diffs.append({"id": k, "what": "score", "first": x.get("score_txt"), "orig": y.get("score_txt"),
                                  "diff_eok": round(d / EOK, 3), "digit": p, "self": y.get("self"),
                                  "need_orig": k in need})
                else:
                    st["score_lt_0.1eok"] += 1
            elif y.get("score") is not None:
                st["score_orig_only"] += 1
    rate = st["score_ge_0.1eok"] / st["score_both"] if st["score_both"] else None
    return {
        "counts": dict(st),
        "digit_of_first_diff": {str(k): v for k, v in sorted(pos.items())},
        "score_ge_0.1eok_rate": rate,
        "verdict": "drop_orig" if rate is not None and rate <= 0.005 else "orig_only_need_orig",
    }, diffs


# ---------- 2. 덱별 피해량을 빼면 ----------

def step2(images):
    tab = C.Counter((i["tri"] or "-", i["kind"] or "-") for i in images)
    read = [i for i in images if i["read"]]
    deck = [i for i in read if i["kind"] == "deck_only"]
    by_tri = C.Counter(i["tri"] or "-" for i in deck)
    orig = [i for i in images if i["orig_read"]]
    orig_deck = [i for i in orig if i["kind"] == "deck_only"]
    # 지난 비용: 판독 1.37억 / 판독 사진, 원본 재판독 1.26억 / 원본 재판독 사진
    per_read, per_orig = 1.37e8 / len(read), 1.26e8 / len(orig)
    return {
        "tri_x_kind": {f"{t}|{k}": v for (t, k), v in sorted(tab.items())},
        "read": len(read),
        "read_deck_only": len(deck),
        "read_deck_only_by_tri": dict(by_tri),
        "read_without_deck": len(read) - len(deck),
        "read_cut_rate": len(deck) / len(read),
        "orig_read": len(orig),
        "orig_read_deck_only": len(orig_deck),
        "tokens_read_per_image": round(per_read),
        "tokens_saved_read": round(per_read * len(deck)),
        "tokens_saved_orig": round(per_orig * len(orig_deck)),
    }


# ---------- 3. %만 있는 관측 ----------

PCT_RE = re.compile(r"(\d{1,2}(?:\.\d{1,3})?)\s*(?:%|퍼(?!즐|스|펙|센트?가)|프로(?!필|젝|그))")


def step3(obs, claims, posts):
    keyobs = [o for o in obs if o["screen"] in KEY_SCREENS and not o["mock"] and not o.get("dup_of")]
    img_pct_only = [o for o in keyobs if o["pct"] is not None and o["score"] is None]
    img_pct_only_self = [o for o in img_pct_only if o["self"]]
    cl_pct_only = [c for c in claims if c["pct"] is not None and c["score_eok"] is None]
    cl_by_post = C.defaultdict(list)
    for c in claims:
        cl_by_post[c["no"]].append(c)
    multi = []
    for p in posts:
        txt = (p.get("title") or "") + "\n" + (p.get("body") or "")
        vals = []
        for m in PCT_RE.finditer(txt):
            v = float(m.group(1))
            if 0 < v <= 100 and v not in vals:
                vals.append(v)
        if len(vals) < 2:
            continue
        got = sorted({c["pct"] for c in cl_by_post.get(p["no"], []) if c["pct"] is not None})
        missing = [v for v in vals if v not in got]
        strict = sum(1 for v in vals if v != int(v)) >= 2  # 소수 % 둘 이상: 관측일 가능성이 높다
        multi.append({"no": p["no"], "t": p["t"], "strict": strict, "pcts": vals, "claimed": got, "missing": missing,
                      "title": p["title"][:80]})
    reduced = [m for m in multi if m["missing"]]
    return {
        "img_key_rows": len(keyobs),
        "img_pct_only_rows": len(img_pct_only),
        "img_pct_only_self": len(img_pct_only_self),
        "img_pct_only_images": len({o["img"] for o in img_pct_only}),
        "claims": len(claims),
        "claims_pct_only": len(cl_pct_only),
        "claims_pct_only_by_kind": dict(C.Counter(c["kind"] for c in cl_pct_only)),
        "posts_multi_pct": len(multi),
        "posts_multi_pct_reduced": len(reduced),
        "pcts_missing": sum(len(m["missing"]) for m in reduced),
        "posts_multi_decimal_pct": sum(1 for m in multi if m["strict"]),
        "posts_multi_decimal_pct_reduced": sum(1 for m in reduced if m["strict"] and any(v != int(v) for v in m["missing"])),
    }, multi


# ---------- 4. 평가 세트 ----------

ANS_FIELDS = ("screen", "mock", "server", "rank", "rank_txt", "pct", "pct_txt", "score", "score_txt", "self", "label")


def step4(images, obs):
    rnd = random.Random(20261015)
    by_img = C.defaultdict(list)
    for o in obs:
        by_img[base(o["img"])].append(o)
    allmock = {k for k, v in by_img.items() if v and all(o["mock"] for o in v)}
    im = {i["id"]: i for i in images}
    groups = {}
    for i in images:
        if i["kind"] == "key":
            groups[i["id"]] = "key"
        elif i["kind"] == "other_data":
            groups[i["id"]] = "other_data"
    mock_pool = sorted(k for k in allmock if k not in groups)
    for k in rnd.sample(mock_pool, min(200, len(mock_pool))):
        groups[k] = "mock"
    deck_pool = sorted(i["id"] for i in images if i["kind"] == "deck_only" and i["id"] not in allmock)
    for k in rnd.sample(deck_pool, 200):
        groups[k] = "deck_only"
    nd_pool = sorted(i["id"] for i in images if i["kind"] == "no_data")
    for k in rnd.sample(nd_pool, 100):
        groups[k] = "no_data"
    for i in images:
        if i["skip"] == "small" and max(i["w"] or 0, i["h"] or 0) >= 250 and (i["h"] or 0) >= 40:
            groups[i["id"]] = "bar"
    out = []
    for k in sorted(groups, key=lambda x: (im[x]["no"], im[x]["i"])):
        rows = by_img.get(k, [])
        if any(o["src"] == "orig" for o in rows):
            rows = [o for o in rows if o["src"] == "orig"]
            src = "orig"
        else:
            src = "first" if rows else None
        h = int(hashlib.sha1(k.encode()).hexdigest(), 16) % 5
        i = im[k]
        out.append({"id": k, "no": i["no"], "group": groups[k], "split": "holdout" if h == 0 else "dev",
                    "url": i["url"], "orig": i["orig"], "sha1": i["sha1"], "w": i["w"], "h": i["h"],
                    "ans_src": src, "rows": [{f: o[f] for f in ANS_FIELDS} for o in rows]})
    return out, {
        "images": len(out),
        "by_group": dict(C.Counter(r["group"] for r in out)),
        "holdout": dict(C.Counter(r["group"] for r in out if r["split"] == "holdout")),
        "ans_src": dict(C.Counter(f'{r["group"]}|{r["ans_src"]}' for r in out)),
        "allmock_images": len(allmock),
        "mock_pool": len(mock_pool),
    }


def main():
    images, obs = load("images.jsonl"), load("obs_images.jsonl")
    first, orig = load("reads_first.jsonl"), load("reads_orig.jsonl")
    claims, posts = load("claims_text.jsonl"), load("posts_min.jsonl")
    r1, diffs = step1(first, orig)
    r2 = step2(images)
    r3, multi = step3(obs, claims, posts)
    ev, r4 = step4(images, obs)
    write_jsonl("orig_diff.jsonl", diffs)
    write_jsonl("multi_pct_posts.jsonl", multi)
    write_jsonl("evalset.jsonl", ev)
    res = {"1_orig": r1, "2_deck": r2, "3_pct_only": r3, "4_evalset": r4}
    with open(os.path.join(OUT, "step0.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
