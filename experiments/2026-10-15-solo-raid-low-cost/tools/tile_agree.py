"""2단계: 같은 시트를 두 모델이 읽은 결과를 서로 맞대고, 혼합 방식의 오류를 잰다 (모델 0).

python3 -I tools/tile_agree.py SHEETS.jsonl READS_A.jsonl READS_B.jsonl EVALSET.jsonl OUT.json [DIFF.jsonl]

- 사진마다 A·B 줄을 pair_rows로 짝짓는다. 세 값(순위·%·점수 0.1억)이 모두 같으면 "같음", 하나라도 다르거나
  한쪽에만 있거나 한쪽이 `?`로 비웠으면 "다름".
- key 사진은 정답과도 맞대: A만 쓸 때, B만 쓸 때, "같으면 그대로·다르면 B" 혼합, "같은 줄만 남김"의 틀린 값 수.
- 음성 오판은 묶음별로 센다 (`other_data`는 지난 시즌·다른 레이드의 같은 모양 화면이라 따로 본다).
- DIFF: 두 모델이 다른 줄 (원본 확인 대상).
"""
import collections as C
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import target  # noqa: E402
from step0 import pair_rows  # noqa: E402
from tile_measure import NEG, rows_of  # noqa: E402

K = ("rank", "pct", "score")


def same(x, y, k):
    a, b = x.get(k), y.get(k)
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < (10_000_000 if k == "score" else 1e-9)


def wrong(x, g):
    """x 줄이 정답 g 줄과 다른 값 수 (둘 다 있는 값만)."""
    return sum(1 for k in K if x.get(k) is not None and g.get(k) is not None and not same(x, g, k))


def main():
    sheets_f, ra_f, rb_f, ev_f, out_f = sys.argv[1:6]
    cell2id = {}
    for m in map(json.loads, open(sheets_f)):
        cell2id.update(m["cells"])
    reads = []
    for f in (ra_f, rb_f):
        d = C.defaultdict(list)
        for r in map(json.loads, open(f)):
            if r.get("cell") in cell2id:
                d[cell2id[r["cell"]]].append(r)
        reads.append(d)
    ev = {e["id"]: e for e in map(json.loads, open(ev_f))}
    st = C.Counter()
    diff = []
    for i in sorted(set(cell2id.values())):
        g = ev[i]["group"]
        a, b = rows_of(reads[0].get(i, [])), rows_of(reads[1].get(i, []))
        if g in NEG:
            st[f"fp_{g}_a"] += bool(a)
            st[f"fp_{g}_b"] += bool(b)
            st[f"fp_{g}_both"] += bool(a) and bool(b)
            st[f"n_{g}"] += 1
            continue
        pairs, ua, ub = pair_rows(a, b)
        agree = [(x, y) for x, y in pairs if all(same(x, y, k) for k in K) and not x["partial"] and not y["partial"]]
        st[f"{g}_rows_a"] += len(a)
        st[f"{g}_rows_b"] += len(b)
        st[f"{g}_agree"] += len(agree)
        st[f"{g}_differ"] += len(pairs) - len(agree) + len(ua) + len(ub)
        for x, y in pairs:
            if (x, y) not in agree:
                diff.append({"id": i, "group": g, "a": x, "b": y})
        for x in ua:
            diff.append({"id": i, "group": g, "a": x, "b": None})
        for y in ub:
            diff.append({"id": i, "group": g, "a": None, "b": y})
        if g != "key":
            continue
        gold = target(ev[i]["rows"])
        st["gold"] += len(gold)
        # 줄마다 각 방식의 값과 정답을 맞댄다
        mix = [y if (x, y) not in agree else x for x, y in pairs] + ub + ua   # 다르면 B, 한쪽에만 있으면 그 줄
        for name, rows in (("a", a), ("b", b), ("mix", mix), ("agree_only", [x for x, _ in agree])):
            gp, _, um = pair_rows(rows, gold)
            st[f"{name}_paired"] += len(gp)
            st[f"{name}_wrong_vals"] += sum(wrong(x, y) for x, y in gp)
            st[f"{name}_wrong_rows"] += sum(bool(wrong(x, y)) for x, y in gp)
            st[f"{name}_score_both"] += sum(x.get("score") is not None and y.get("score") is not None for x, y in gp)
        gp, _, _ = pair_rows([x for x, _ in agree], gold)
        for x, y in gp:
            if wrong(x, y):
                diff.append({"id": i, "group": g, "agree_but_gold_differs": True, "row": x, "gold": y})
    res = {"counts": dict(st)}
    for name in ("a", "b", "mix", "agree_only"):
        p = st[f"{name}_paired"]
        res[name] = {"recall": round(p / st["gold"], 4) if st["gold"] else None,
                     "wrong_rows": st[f"{name}_wrong_rows"], "paired": p,
                     "wrong_row_rate": round(st[f"{name}_wrong_rows"] / p, 4) if p else None}
    json.dump(res, open(out_f, "w"), ensure_ascii=False, indent=1)
    if len(sys.argv) > 6:
        with open(sys.argv[6], "w") as f:
            for d in diff:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "counts"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
