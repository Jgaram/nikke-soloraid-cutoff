"""2단계: 타일 판독 결과를 평가 세트 정답과 맞댄다 (모델 0).

python3 -I tools/tile_measure.py SHEETS.jsonl READS.jsonl EVALSET.jsonl PARSED.jsonl OUT.json [MISS.jsonl]

- READS: 에이전트가 쓴 칸별 JSONL (여러 파일이면 이어 붙여서 준다). 칸 → 사진 id는 SHEETS.jsonl.
- 한 사진의 여러 칸 줄을 합친 뒤 measure.py와 같은 정답 줄·짝짓기·일치 기준으로 잰다.
  같은 사진이 띠 여러 칸에 걸치면 겹친 줄이 둘 나올 수 있다 → (순위, %, 점수)가 같은 줄은 하나로 친다.
- key: 정답 줄 재현율, %·점수(0.1억)·순위 일치. 읽다 만 자리(`?`)는 null로 센다.
- 음성(덱별·모의전·그 밖·자료 없음): 순위·% 화면으로 보고 줄을 낸 사진 수.
- bar: 정답이 있는 줄만 같은 방식으로, 없으면 코드 판독(파서 rows)과 같은 값인지.
"""
import collections as C
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure import target  # noqa: E402
from step0 import pair_rows  # noqa: E402

NEG = ("deck_only", "mock", "other_data", "no_data")
OBS = ("rank_list", "my_rank", "result", "season_end", "tracker")  # tracker: 세션 F1


def num(s, kind):
    if s is None or "?" in str(s):
        return None
    s = str(s)
    if kind == "pct":
        m = re.search(r"(\d+(?:\.\d+)?)\s*%?", s)
        return float(m.group(1)) if m else None
    d = re.sub(r"[^\d]", "", s)
    if not d:
        return None
    if kind == "score":
        return int(d) if len(d) >= 9 else None
    return int(d)


def rows_of(reads):
    out, seen = [], set()
    for r in reads:
        if r.get("screen") not in OBS:
            continue
        for x in r.get("rows") or []:
            y = {"rank": num(x.get("rank_txt"), "rank"), "pct": num(x.get("pct_txt"), "pct"),
                 "score": num(x.get("score_txt"), "score"), "self": bool(x.get("self")),
                 "partial": any("?" in str(x.get(k) or "") for k in ("rank_txt", "pct_txt", "score_txt"))}
            k = (y["rank"], y["pct"], y["score"])
            if k == (None, None, None) or k in seen:
                continue
            seen.add(k)
            out.append(y)
    return out


def compare(st, got, gold, pre):
    pairs, ua, ub = pair_rows(got, gold)
    st[pre + "gold"] += len(gold)
    st[pre + "pairs"] += len(pairs)
    st[pre + "extra"] += len(ua)
    bad = []
    for x, y in pairs:
        for k in ("pct", "rank"):
            if x.get(k) is not None and y.get(k) is not None:
                st[f"{pre}{k}_both"] += 1
                if abs(x[k] - y[k]) < 1e-9:
                    st[f"{pre}{k}_ok"] += 1
                else:
                    bad.append((k, x[k], y[k]))
        if x.get("score") is not None and y.get("score") is not None:
            st[pre + "score_both"] += 1
            if abs(x["score"] - y["score"]) < 10_000_000:
                st[pre + "score_ok"] += 1
            else:
                bad.append(("score", x["score"], y["score"]))
    return bad, len(ua), len(ub)


def main():
    sheets_f, reads_f, ev_f, parsed_f, out_f = sys.argv[1:6]
    cell2id = {}
    for m in map(json.loads, open(sheets_f)):
        cell2id.update(m["cells"])
    by_id = C.defaultdict(list)
    st = C.Counter()
    for r in map(json.loads, open(reads_f)):
        i = cell2id.get(r.get("cell"))
        st["cells_read"] += 1
        if i is None:
            st["cells_unknown"] += 1
            continue
        by_id[i].append(r)
    st["cells_total"] = len(cell2id)
    ev = {e["id"]: e for e in map(json.loads, open(ev_f))}
    parsed = {p["id"]: p for p in map(json.loads, open(parsed_f))}
    miss = []
    for i in sorted(set(cell2id.values())):
        e, g = ev[i], ev[i]["group"]
        got = rows_of(by_id.get(i, []))
        st[f"n_{g}"] += 1
        st["partial_rows"] += sum(x["partial"] for x in got)
        if g == "key" or (g == "bar" and target(e["rows"])):
            pre = "" if g == "key" else "bar_"
            bad, extra, missed = compare(st, got, target(e["rows"]), pre)
            if bad or missed:
                miss.append({"id": i, "group": g, "bad": bad[:4], "extra": extra, "missed": missed})
        elif g == "bar":
            st["bar_nogold"] += 1
            st["bar_nogold_rows"] += bool(got)
            code = parsed.get(i, {}).get("rows") or []
            if got and code:
                pairs, _, _ = pair_rows(got, code)
                st["bar_code_pairs"] += len(pairs)
                st["bar_code_same"] += sum(all(x.get(k) == y.get(k) or None in (x.get(k), y.get(k))
                                               for k in ("pct", "rank")) and
                                           (x.get("score") is None or y.get("score") is None or
                                            abs(x["score"] - y["score"]) < 10_000_000) for x, y in pairs)
        elif g in NEG:
            fp = bool(got)
            st["neg_fp"] += fp
            if fp:
                miss.append({"id": i, "group": g, "fp": got[:2], "screens": [r.get("screen") for r in by_id[i]]})

    def r(a, b):
        return round(st[a] / st[b], 4) if st[b] else None

    res = {"row_recall": r("pairs", "gold"), "pct_ok": r("pct_ok", "pct_both"),
           "score_ok_0.1eok": r("score_ok", "score_both"), "rank_ok": r("rank_ok", "rank_both"),
           "neg_fp": f'{st["neg_fp"]}/{sum(st["n_" + g] for g in NEG)}', "counts": dict(st)}
    json.dump(res, open(out_f, "w"), ensure_ascii=False, indent=1)
    if len(sys.argv) > 6:
        with open(sys.argv[6], "w") as f:
            for m in miss:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "counts"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
