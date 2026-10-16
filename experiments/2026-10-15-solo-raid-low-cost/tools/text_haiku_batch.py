"""3단계 2번 시범: 규칙 분류를 Haiku 한 배치(200줄)와 비교한다.

python3 -I tools/text_haiku_batch.py make S41_DATA_DIR EXTRACTED.jsonl IN.jsonl KEY.jsonl
python3 -I tools/text_haiku_batch.py score IN.jsonl KEY.jsonl OUT.jsonl RESULT.json

- make: 정답 주장과 짝이 맞은 추출 줄에서 200줄을 시드 20261015로 뽑는다. 정답 kind는 KEY에만 둔다.
  입력 줄: id, 제목, 그 칸 글(본문은 300자), 추출한 % · 점수, 근거 조각.
- score: Haiku kind와 규칙 kind를 각각 정답과 맞댄다 (6종, 관측 여부 self·cutoff·other).
"""
import collections
import json
import os
import random
import sys

OBS = {"self", "cutoff", "other"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def make(ddir, ext_p, in_p, key_p):
    posts = {p["no"]: p for p in jl(os.path.join(ddir, "posts_min.jsonl"))}
    by_post = collections.defaultdict(list)
    for r in jl(ext_p):
        by_post[r["no"]].append(r)
    pairs = []
    for g in jl(os.path.join(ddir, "claims_text.jsonl")):
        rows = by_post[g["no"]]
        if g["pct"] is not None:
            hit = [r for r in rows if r["pct"] is not None and abs(r["pct"] - float(g["pct"])) < 0.005]
        elif g["score_eok"] is not None:
            hit = [r for r in rows if r["score_eok"] is not None and abs(r["score_eok"] - float(g["score_eok"])) <= 0.5]
        else:
            continue
        if hit:
            pairs.append((g, hit[0]))
    random.seed(20261015)
    seen, pick = set(), []
    for g, r in random.sample(pairs, len(pairs)):
        k = (r["no"], r["src"], r["seg"], r["pct"], r["score_eok"])
        if k in seen:
            continue
        seen.add(k)
        pick.append((g, r))
        if len(pick) == 200:
            break
    with open(in_p, "w", encoding="utf-8") as fi, open(key_p, "w", encoding="utf-8") as fk:
        for i, (g, r) in enumerate(pick):
            p = posts[r["no"]]
            fi.write(json.dumps({"id": i, "t": p["t"], "title": p["title"], "body": (p["body"] or "")[:300],
                                 "src": r["src"], "seg": r["txt"], "pct": r["pct"], "score_eok": r["score_eok"]},
                                ensure_ascii=False) + "\n")
            fk.write(json.dumps({"id": i, "gold": g["kind"], "rule": r["kind"]}, ensure_ascii=False) + "\n")
    print(f"pairs {len(pairs)} picked {len(pick)}")


def score(in_p, key_p, out_p, res_p):
    key = {k["id"]: k for k in jl(key_p)}
    out = {o["id"]: o for o in jl(out_p)}
    c = collections.Counter()
    conf = collections.Counter()
    for i, k in key.items():
        h = (out.get(i) or {}).get("kind")
        c["n"] += 1
        c["answered"] += h is not None
        c["haiku_ok"] += h == k["gold"]
        c["rule_ok"] += k["rule"] == k["gold"]
        c["haiku_obs_ok"] += (h in OBS) == (k["gold"] in OBS)
        c["rule_obs_ok"] += (k["rule"] in OBS) == (k["gold"] in OBS)
        c["agree"] += h == k["rule"]
        c["agree_ok"] += h == k["rule"] == k["gold"]
        if h != k["gold"]:
            conf[f"{k['gold']}->{h}"] += 1
    res = dict(c)
    for a in ("haiku_ok", "rule_ok", "haiku_obs_ok", "rule_obs_ok"):
        res[a + "_rate"] = round(c[a] / c["n"], 4)
    res["agree_acc"] = round(c["agree_ok"] / max(1, c["agree"]), 4)
    res["haiku_confusion"] = dict(conf.most_common())
    json.dump(res, open(res_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "haiku_confusion"}, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(*sys.argv[2:6])
    else:
        score(*sys.argv[2:6])
