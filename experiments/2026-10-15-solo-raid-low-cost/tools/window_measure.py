"""세션 F1·F2: 기간 실행 결과(STATE/obs.jsonl·claims.jsonl)를 S41 재수집과 맞댄다 (모델 0, 요청 0).

python3 -I tools/window_measure.py STATE START END OUT.json [DIFF.jsonl]

- 대상: S41 재수집 글 중 글 시각이 [START, END]인 글.
- 사진: S41 핵심 줄(목록·내 순위·시즌 최종, 순위·%가 있는 결과, mock·지난 시즌 아님)을 이번 관측(season in)과
  같은 사진(id)끼리 step0.pair_rows로 짝짓는다. 재현율, 값이 다른 줄(%·순위 다름, 점수 0.1억 이상), 이번에만 있는 줄.
  지난번에 판독하지 않은 사진에서 나온 줄은 따로 센다.
- 글: S41 claims_text 중 % 또는 점수가 있는 주장을 같은 글의 이번 claims와 맞댄다 (% 같음, 점수만이면 0.5억 안).
- DIFF: 값이 다른 줄과 놓친 줄 (원본 확인용).
- S41 재수집의 긴 사진 조각 id(`6232406_0#p1of2`)는 `#` 앞으로 이번 사진 id와 잇는다. F2 측정에서 22줄을 「받지 않음」으로 잘못 셌다.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S41 = os.path.join(HERE, "..", "2026-10-15-s41-opus-blind", "data")
sys.path.insert(0, os.path.join(HERE, "tools"))
from step0 import pair_rows as _pair  # noqa: E402
from season_measure import key_row  # noqa: E402

def pair_rows(a, b):
    return _pair(a, b)[0]


KEY = ("rank_list", "my_rank", "season_end", "result", "tracker")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def main(state, start, end, out, diff_p=None):
    old = [o for o in jl(os.path.join(S41, "obs_images.jsonl")) if o["post_t"] and start <= o["post_t"] <= end]
    old_key = [o for o in old if key_row(o)]
    imgs41 = {i["id"]: i for i in jl(os.path.join(S41, "images.jsonl"))}
    obs = [o for o in jl(os.path.join(state, "obs.jsonl")) if o["post_t"] and start <= o["post_t"] <= end]
    new_in = [o for o in obs if o["season"] == "in"]
    new_un = [o for o in obs if o["season"] == "unsure"]
    new_key = [o for o in new_in if o["screen"] in KEY and (o["rank"] is not None or o["pct"] is not None)]
    st = collections.Counter()
    diffs = []
    by_old, by_new = collections.defaultdict(list), collections.defaultdict(list)
    for o in old_key:
        by_old[o["img"].split("#")[0]].append(o)   # 긴 사진 조각(`#p1of2`)은 같은 사진으로 묶는다 (2026-10-16)
    for o in new_in:
        by_new[o["img"]].append(o)
    route = {r["id"]: r for r in jl(os.path.join(state, "route.jsonl"))}
    for img, rows in by_old.items():
        st["old_key"] += len(rows)
        got = by_new.get(img, [])
        pairs = pair_rows(rows, got)
        st["old_key_found"] += len(pairs)
        for a, b in pairs:
            bad = [k for k in ("rank", "pct") if a[k] is not None and b[k] is not None and a[k] != b[k]]
            if a["score"] and b["score"] and abs(a["score"] - b["score"]) >= 1e7:
                bad.append("score")
            if bad:
                st["value_diff"] += 1
                diffs.append({"img": img, "kind": "diff", "fields": bad, "old": {k: a[k] for k in ("rank", "pct", "score")},
                              "new": {k: b[k] for k in ("rank", "pct", "score")}, "src": b["src"]})
        paired = {id(a) for a, _ in pairs}
        for a in rows:
            if id(a) not in paired:
                r = route.get(img, {})
                st[f"miss_route_{r.get('route', 'not_fetched')}"] += 1
                diffs.append({"img": img, "kind": "miss", "old": {k: a[k] for k in ("rank", "pct", "score")},
                              "route": r.get("route"), "screen_old": a["screen"],
                              "new_rows": len(got), "season_new": None})
    # unsure 줄: 지난번 핵심 줄과 짝지어지는 수(넣으면 재현이 느는 것)와 아닌 수
    by_un = collections.defaultdict(list)
    for o in new_un:
        by_un[o["img"]].append(o)
    for img, rows in by_un.items():
        st["unsure"] += len(rows)
        st["unsure_was_key"] += len(pair_rows(by_old.get(img, []), rows))
    # 이번 관측 중 시즌 거르기로 빠진 줄이 지난번 핵심 줄과 짝지어지는지
    out_rows = collections.defaultdict(list)
    for o in obs:
        if o["season"] == "out":
            out_rows[o["img"]].append(o)
    for img, rows in out_rows.items():
        st["filtered_out"] += len(rows)
        st["filtered_out_was_key"] += len(pair_rows(by_old.get(img, []), rows))
    old_imgs = set(by_old)
    for img, rows in by_new.items():
        k = [o for o in rows if o in new_key]
        if not k:
            continue
        if img in old_imgs:
            st["new_key_extra_same_img"] += max(0, len(k) - len(pair_rows(by_old[img], k)))
        else:
            i41 = imgs41.get(img)
            tag = "not_in_s41" if i41 is None else ("s41_unread" if not i41.get("read") else f"s41_{i41.get('kind')}")
            st[f"new_key_only_{tag}"] += len(k)
    st["new_key"] = len(new_key)
    st["new_pct_only"] = sum(1 for o in new_in if o["pct"] is not None and o["score"] is None)
    st["new_tracker"] = sum(1 for o in new_in if o["src"] == "tracker")
    st.update({f"src_{k}": v for k, v in collections.Counter(o["src"] for o in new_key).items()})
    # 글 주장
    oc = [c for c in jl(os.path.join(S41, "claims_text.jsonl")) if start <= (c.get("post_t") or "") <= end
          and (c["pct"] is not None or c["score_eok"] is not None)]
    nc = collections.defaultdict(list)
    for c in jl(os.path.join(state, "claims.jsonl")):
        nc[c["no"]].append(c)
    for c in oc:
        st["claims_old"] += 1
        hit = [x for x in nc.get(c["no"], []) if (c["pct"] is not None and x["pct"] is not None and abs(x["pct"] - float(c["pct"])) < 0.005)
               or (c["pct"] is None and x["score_eok"] is not None and abs(x["score_eok"] - float(c["score_eok"])) <= 0.5)]
        st["claims_found"] += bool(hit)
        st["claims_kind_same"] += bool(hit) and any(x["kind"] == c["kind"] for x in hit)
        if not hit:
            diffs.append({"no": c["no"], "kind": "claim_miss", "old": {"pct": c["pct"], "score_eok": c["score_eok"], "kind": c["kind"]},
                          "txt": c.get("txt", "")[:60]})
    res = dict(st)
    res["recall_key"] = round(st["old_key_found"] / max(1, st["old_key"]), 4)
    res["recall_claims"] = round(st["claims_found"] / max(1, st["claims_old"]), 4)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1, sort_keys=True)
    if diff_p:
        with open(diff_p, "w", encoding="utf-8") as f:
            for d in diffs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main(*sys.argv[1:6])
