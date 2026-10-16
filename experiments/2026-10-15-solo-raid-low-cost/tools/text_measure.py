"""3단계 3번: 코드 추출(text_extract.py 결과)을 글 주장 정답·기준점 글 출처 행·여러 % 글과 맞댄다.

python3 -I tools/text_measure.py S41_DATA_DIR BASELINE_DIR EXTRACTED.jsonl MULTI_PCT.jsonl OUT_PREFIX

- 정답 범위: claims_text 중 %나 점수가 있고, 그 값이 posts_min의 해당 칸(제목·본문 300자)에 숫자로 보이는 주장.
- 짝: 같은 글에서 정답 %가 그 칸에 보이면 % 같음(차 0.005 미만), 아니면(문맥으로 붙인 %) 점수가 0.5억 안.
  % 짝의 점수는 따로 센다(같음·추출 쪽 없음·다름).
- 관측 종류: self·cutoff·other (기준점 비교 도구와 같다). 나머지 predict·question·prev_season은 관측 아님.
- 정밀도: 추출 줄 중 같은 글에 짝 정답이 있는 비율. `core`(소수 %이거나 %+점수) 관측 줄만 따로도 센다. 정답에 빠진 값(여러 % 글)도 오답으로 잡히니 하한이다.
- OUT_PREFIX.json에 지표, OUT_PREFIX_miss.jsonl에 놓친 정답·종류 틀린 줄·짝 없는 core 추출 관측 줄.
"""
import collections
import json
import os
import re
import sys

OBS = {"self", "cutoff", "other"}
RE_NUM = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def nums(t):
    return {float(x) for x in RE_NUM.findall(t or "")}


def peq(a, b):
    return a is not None and b is not None and abs(a - b) < 0.005


def seq(a, b):
    return a is not None and b is not None and abs(a - b) <= 0.5


def match(g, rows, ns):
    if g["pct"] is not None and (float(g["pct"]) in ns or g["score_eok"] is None):
        return [r for r in rows if peq(r["pct"], float(g["pct"]))]
    return [r for r in rows if r["pct"] is None and seq(r["score_eok"], float(g["score_eok"]))] or \
        [r for r in rows if seq(r["score_eok"], float(g["score_eok"]))]


def main():
    ddir, bdir, ext_p, multi_p, outp = sys.argv[1:6]
    posts = {p["no"]: p for p in jl(os.path.join(ddir, "posts_min.jsonl"))}
    ext = jl(ext_p)
    by = collections.defaultdict(list)
    for r in ext:
        by[(r["no"], r["src"])].append(r)
    by_post = collections.defaultdict(list)
    for r in ext:
        by_post[r["no"]].append(r)

    gold = []
    for c in jl(os.path.join(ddir, "claims_text.jsonl")):
        if c["pct"] is None and c["score_eok"] is None:
            continue
        field = (posts.get(c["no"]) or {}).get(c["src"]) or ""
        ns = nums(field)
        vals = [float(v) for v in (c["pct"], c["score_eok"]) if v is not None]
        c["in_text"] = any(v in ns for v in vals)
        gold.append(c)
    scope = [g for g in gold if g["in_text"]]

    m = collections.Counter()
    by_kind = collections.defaultdict(collections.Counter)
    conf = collections.Counter()
    miss = []
    used = set()
    for g in scope:
        rows = by_post[g["no"]]
        hit = match(g, rows, nums((posts.get(g["no"]) or {}).get(g["src"]) or ""))
        gk = g["kind"]
        by_kind[gk]["n"] += 1
        if not hit:
            m["miss"] += 1
            miss.append({"why": "miss", "no": g["no"], "src": g["src"], "kind": gk, "pct": g["pct"],
                         "score": g["score_eok"], "txt": g["txt"][:60]})
            continue
        m["hit"] += 1
        by_kind[gk]["hit"] += 1
        best = next((r for r in hit if r["kind"] == gk), hit[0])
        for r in hit:
            used.add(id(r))
        conf[(gk, best["kind"])] += 1
        m["kind_ok"] += best["kind"] == gk
        m["obs_ok"] += (best["kind"] in OBS) == (gk in OBS)
        if best["kind"] != gk:
            miss.append({"why": "kind", "no": g["no"], "src": g["src"], "gold": gk, "got": best["kind"],
                         "pct": g["pct"], "score": g["score_eok"], "txt": best["txt"][:60]})
        if g["pct"] is not None and g["score_eok"] is not None:
            m["pct_with_score"] += 1
            ss = [r["score_eok"] for r in hit if r["score_eok"] is not None]
            if any(seq(s, float(g["score_eok"])) for s in ss):
                m["score_same"] += 1
            elif not ss:
                m["score_none"] += 1
            else:
                m["score_diff"] += 1

    # 정밀도: 정답이 있는 칸이든 없든 모든 추출 줄
    gold_by = collections.defaultdict(list)
    for g in gold:
        gold_by[g["no"]].append(g)
    pm = collections.Counter()
    for r in ext:
        gs = gold_by[r["no"]]
        ok = any((g["pct"] is not None and peq(r["pct"], float(g["pct"]))) or
                 (g["pct"] is None and seq(r["score_eok"], float(g["score_eok"]))) or
                 (r["pct"] is None and g["score_eok"] is not None and seq(r["score_eok"], float(g["score_eok"])))
                 for g in gs)
        k = "obs" if r["kind"] in OBS else "nonobs"
        pm[k + "_n"] += 1
        pm[k + "_ok"] += ok
        if r.get("core") and r["kind"] in OBS:
            pm["core_obs_n"] += 1
            pm["core_obs_ok"] += ok
        if not ok and r["kind"] in OBS and r.get("core"):
            miss.append({"why": "extra_obs", "no": r["no"], "src": r["src"], "got": r["kind"], "pct": r["pct"],
                         "score": r["score_eok"], "txt": r["txt"][:60]})

    # 기준점 글 출처 행 (디시 제목·본문)
    base = [b for b in json.load(open(os.path.join(bdir, "data/outputs/graded.json"), encoding="utf-8"))
            if b["src"] == "dc" and b["loc"] in ("title", "body")]
    bm = collections.Counter()
    for b in base:
        no = int(b["post"])
        field = (posts.get(no) or {}).get(b["loc"]) or ""
        inr = b["pct"] is not None and float(b["pct"]) in nums(field)
        bm["n"] += 1
        if not inr:
            bm["out_of_text"] += 1
            continue
        bm["in_text"] += 1
        hit = [r for r in by_post[no] if peq(r["pct"], float(b["pct"]))]
        if not hit:
            bm["miss"] += 1
            miss.append({"why": "base_miss", "no": no, "src": b["loc"], "pct": b["pct"], "txt": (b.get("quote") or "")[:60]})
            continue
        bm["hit"] += 1
        bm["hit_obs"] += any(r["kind"] in OBS for r in hit)
        if b["score"] is not None:
            bm["score_n"] += 1
            bm["score_same"] += any(seq(r["score_eok"], b["score"] / 1e8) for r in hit)
        bm["time_n"] += bool(b.get("time_quote"))
        bm["time_hit"] += bool(b.get("time_quote")) and any(r["time"] and r["time"] in b["time_quote"].replace(" ", "") for r in hit)

    # 여러 %를 하나로 줄인 글 (0단계 strict)
    mm = collections.Counter()
    for x in jl(multi_p):
        if not x["strict"]:
            continue
        dec = [v for v in x["missing"] if v != int(v)]
        if not dec:
            continue
        mm["posts"] += 1
        got = [v for v in dec if any(peq(r["pct"], v) for r in by_post[x["no"]])]
        mm["values"] += len(dec)
        mm["values_got"] += len(got)
        mm["posts_all"] += len(got) == len(dec)
        mm["values_got_obs"] += sum(any(peq(r["pct"], v) and r["kind"] in OBS for r in by_post[x["no"]]) for v in dec)

    res = {
        "gold_all": len(gold), "gold_in_text": len(scope),
        "recall": round(m["hit"] / len(scope), 4),
        "recall_by_kind": {k: f"{v['hit']}/{v['n']}" for k, v in sorted(by_kind.items())},
        "kind_acc_6": round(m["kind_ok"] / m["hit"], 4),
        "kind_acc_obs": round(m["obs_ok"] / m["hit"], 4),
        "score_on_pct_match": {k: m[k] for k in ("pct_with_score", "score_same", "score_none", "score_diff")},
        "precision_obs": f"{pm['obs_ok']}/{pm['obs_n']}", "precision_obs_rate": round(pm["obs_ok"] / max(1, pm["obs_n"]), 4),
        "precision_core_obs": f"{pm['core_obs_ok']}/{pm['core_obs_n']}",
        "precision_core_obs_rate": round(pm["core_obs_ok"] / max(1, pm["core_obs_n"]), 4),
        "precision_all_rate": round((pm["obs_ok"] + pm["nonobs_ok"]) / max(1, pm["obs_n"] + pm["nonobs_n"]), 4),
        "extracted_rows": len(ext),
        "confusion": {f"{a}->{b}": n for (a, b), n in sorted(conf.items()) if a != b},
        "baseline": dict(bm), "multi_pct_strict": dict(mm),
    }
    json.dump(res, open(outp + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(outp + "_miss.jsonl", "w", encoding="utf-8") as f:
        for x in miss:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({k: res[k] for k in ("gold_in_text", "recall", "kind_acc_6", "kind_acc_obs", "precision_obs_rate",
                                           "precision_core_obs_rate", "precision_all_rate", "score_on_pct_match", "baseline", "multi_pct_strict")},
                     ensure_ascii=False))
    print(json.dumps(res["recall_by_kind"], ensure_ascii=False))
    print(collections.Counter(x["why"] for x in miss))


if __name__ == "__main__":
    main()
