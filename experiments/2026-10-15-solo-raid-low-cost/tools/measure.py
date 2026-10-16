"""파서 결과를 평가 세트 정답과 맞댄다.

python3 -I tools/measure.py PARSED.jsonl [dev|holdout|all] [OUT.json] [MISS.jsonl]

- 정답 줄: mock이 아니고, 화면이 my_rank·rank_list·season_end이거나 result 중 순위·%가 있는 줄.
- 짝짓기는 step0.py의 pair_rows (순위 → 내 줄 % → 가까운 점수).
- 점수 일치: 0.1억(1천만) 미만 차이.
"""
import collections as C
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step0 import pair_rows  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS = ("result_pop", "my_rank", "rank_list", "season_end")
NEG = ("deck_only", "mock", "other_data", "no_data")


def target(rows):
    out = []
    for r in rows:
        if r["mock"]:
            continue
        if r["screen"] in ("my_rank", "rank_list", "season_end") or (
                r["screen"] == "result" and (r["rank"] is not None or r["pct"] is not None)):
            if any(r[k] is not None for k in ("rank", "pct", "score")):
                out.append(r)
    return out


def main():
    parsed = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
    split = sys.argv[2] if len(sys.argv) > 2 else "dev"
    ev = [json.loads(l) for l in open(os.path.join(HERE, "data", "evalset.jsonl"), encoding="utf-8")]
    ev = [e for e in ev if e["id"] in parsed and (split == "all" or e["split"] == split)]
    st = C.Counter()
    why = C.Counter()
    neg = C.Counter()
    miss = []
    for e in ev:
        p = parsed[e["id"]]
        g = e["group"]
        st[f"n_{g}"] += 1
        if g == "key":
            st["key_hit"] += p["hit"]
            ok = p["conf"] and p["screen"] in POS
            st["key_conf"] += ok
            if not ok:
                why[p["screen"] + ":" + ",".join(p["why"])] += 1
                continue
            tg = target(e["rows"])
            pairs, ua, ub = pair_rows(p["rows"], tg)
            st["tg_rows"] += len(tg)
            st["pairs"] += len(pairs)
            st["extra_rows"] += len(ua)
            bad = []
            for x, y in pairs:
                for k in ("pct", "rank"):
                    if x.get(k) is not None and y.get(k) is not None:
                        st[f"{k}_both"] += 1
                        if abs(x[k] - y[k]) < 1e-9:
                            st[f"{k}_ok"] += 1
                        else:
                            bad.append((k, x[k], y[k]))
                if x.get("score") is not None and y.get("score") is not None:
                    st["score_both"] += 1
                    if abs(x["score"] - y["score"]) < 10_000_000:
                        st["score_ok"] += 1
                    else:
                        bad.append(("score", x["score"], y["score"]))
            if bad or ua or ub:
                miss.append({"id": e["id"], "screen": p["screen"], "bad": bad[:4], "extra": len(ua), "missed": len(ub)})
        elif g in NEG:
            fp = p["conf"] and p["screen"] in POS and bool(p["rows"])
            neg[f"{g}_fp"] += fp
            neg[f"{g}_hit"] += p["hit"]
            if fp:
                miss.append({"id": e["id"], "group": g, "fp_screen": p["screen"], "rows": p["rows"][:2]})
        elif g == "bar":
            st["bar_hit"] += p["hit"]
            st["bar_conf_rows"] += p["conf"] and bool(p["rows"])

    def r(a, b):
        return round(st[a] / st[b], 4) if st[b] else None

    nneg = sum(st[f"n_{g}"] for g in NEG)
    res = {
        "split": split, "images": len(ev),
        "coverage": r("key_conf", "n_key"),
        "code_triage_recall": r("key_hit", "n_key"),
        "pct_ok": r("pct_ok", "pct_both"), "score_ok_0.1eok": r("score_ok", "score_both"), "rank_ok": r("rank_ok", "rank_both"),
        "row_recall_in_conf": r("pairs", "tg_rows"),
        "neg_fp_rate": round(sum(neg[f"{g}_fp"] for g in NEG) / nneg, 4) if nneg else None,
        "neg": dict(neg), "counts": dict(st), "not_conf_why": dict(why.most_common(12)),
    }
    print(json.dumps(res, ensure_ascii=False))
    if len(sys.argv) > 3:
        json.dump(res, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if len(sys.argv) > 4:
        with open(sys.argv[4], "w", encoding="utf-8") as f:
            for m in miss:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
