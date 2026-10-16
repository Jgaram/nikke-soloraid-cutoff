"""세션 F1: 시즌 거르기(season_filter.py)를 S41 재수집 관측과 세션 D·E 판독으로 잰다 (모델 0, 요청 0).

python3 -I tools/season_measure.py make OUTDIR            # 문맥 시범 입력 OUTDIR/ctx/in_NN.jsonl (150글씩)
python3 -I tools/season_measure.py score OUTDIR OUT.json  # OUTDIR/ctx/out_*.jsonl을 읽어 잰다

- 대상 글: S41 obs_images에 줄이 있는 글 + 세션 D·E 574장의 글 (posts_min에 있는 것, 본문은 300자뿐).
- 잴 것
  A. S41 obs_images 전체에 코드 판정을 댄다 (비교 줄 = S41 핵심 줄 자신, 같은 글 제외).
     핵심 줄 2,023줄 중 잘못 빼는 수, `file_date_out` 43줄 중 빼는 수, 다른 레이드 줄 622줄 중 빼는 수.
  B. 세션 D·E 합친 Haiku 판독(data/stage2e/final_*)의 사진별 줄에 코드 + 문맥을 댄다.
     음성 98장에서 나온 줄 중 빼는 수, key 사진에서 나온 줄 중 정답과 맞는 줄을 잘못 빼는 수. 정책 unsure·all 둘 다.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S41 = os.path.join(HERE, "..", "2026-10-15-s41-opus-blind", "data")
sys.path.insert(0, os.path.join(HERE, "tools"))
import merge_obs  # noqa: E402
import season_filter as sf  # noqa: E402
from measure import target  # noqa: E402

SEASON = {"name": "S41", "start": "2026-09-24 12:00:00", "end": "2026-10-01 04:59:59"}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def key_row(o):
    return (not o["mock"] and o["screen"] in ("my_rank", "rank_list", "season_end", "result")
            and (o["rank"] is not None or o["pct"] is not None) and not o["file_date_out"])


def make(outdir):
    obs = jl(os.path.join(S41, "obs_images.jsonl"))
    sh = jl(os.path.join(HERE, "data", "stage2e", "final_sheets.jsonl"))
    nos = {o["no"] for o in obs} | {int(i.split("_")[0]) for m in sh for i in m["cells"].values()}
    pm = [p for p in jl(os.path.join(S41, "posts_min.jsonl")) if p["no"] in nos]
    d = os.path.join(outdir, "ctx")
    os.makedirs(d, exist_ok=True)
    n = 0
    for k in range(0, len(pm), 150):
        n += 1
        with open(os.path.join(d, f"in_{n:02d}.jsonl"), "w", encoding="utf-8") as f:
            for p in pm[k:k + 150]:
                f.write(json.dumps({"no": p["no"], "t": p["t"], "title": p["title"], "body": p["body"], "files": []},
                                   ensure_ascii=False) + "\n")
    print("posts", len(pm), "batches", n)


def score(outdir, out):
    ctx = {}
    for f in sorted(os.listdir(os.path.join(outdir, "ctx"))):
        if f.startswith("out_"):
            for r in jl(os.path.join(outdir, "ctx", f)):
                ctx[r["no"]] = r.get("season")
    asked = sum(1 for f in os.listdir(os.path.join(outdir, "ctx")) if f.startswith("in_")
                for _ in open(os.path.join(outdir, "ctx", f)))
    res = {"ctx_posts": len(ctx), "ctx_asked": asked, "ctx_labels": collections.Counter(ctx.values())}

    # A. S41 관측 전체, 코드 판정
    obs = jl(os.path.join(S41, "obs_images.jsonl"))
    key = [o for o in obs if key_row(o)]
    ref = sf.Ref([(o["no"], o["pct"], o["rank"], o["score"]) for o in key])
    A = collections.Counter()
    for o in obs:
        g = "key" if key_row(o) else "file_date_out" if o["file_date_out"] else \
            "other_raid" if o["screen"] == "other_raid" else "mock" if o["mock"] else "rest"
        c, why = sf.code_check(o, SEASON, ref)
        A[(g, "n")] += 1
        A[(g, c)] += 1
        for pol in ("unsure", "all"):
            A[(g, "final_" + pol, sf.decide(c, ctx.get(o["no"]), pol))] += 1
        if g == "key" and c != "in":
            A[("key_why", why.split()[0])] += 1
    res["A_s41_obs"] = {" ".join(k): v for k, v in sorted(A.items())}

    # B. 세션 D·E Haiku 판독
    ev = {e["id"]: e for e in jl(os.path.join(HERE, "data", "evalset.jsonl"))}
    where = {c: i for m in jl(os.path.join(HERE, "data", "stage2e", "final_sheets.jsonl")) for c, i in m["cells"].items()}
    reads = collections.defaultdict(list)
    for r in jl(os.path.join(HERE, "data", "stage2e", "final_reads.jsonl")):
        if r.get("cell") in where:
            reads[where[r["cell"]]].append(r)
    meta = {}
    for o in obs:
        meta.setdefault(o["img"], o)
    pm = {p["no"]: p for p in jl(os.path.join(S41, "posts_min.jsonl"))}
    B = collections.Counter()
    for i, rs in reads.items():
        e = ev[i]
        _, rows = merge_obs.read_rows(rs)
        no = int(i.split("_")[0])
        m = meta.get(i, {})
        truth = target(e.get("rows") or []) if e["group"] == "key" else []
        for x in rows:
            o = {"no": no, "post_t": (pm.get(no) or {}).get("t") or m.get("post_t"), "file_date": m.get("file_date"),
                 "shot_t": m.get("shot_t"), **{k: x[k] for k in ("rank", "pct", "score")}}
            if e["group"] == "key":
                ok = any((t["pct"] is not None and x["pct"] is not None and abs(t["pct"] - x["pct"]) < 0.005) or
                         (t["score"] and x["score"] and abs(t["score"] - x["score"]) < 1e7) for t in truth)
                g = "key_true" if ok else "key_extra"
            elif e["group"] in ("deck_only", "mock", "other_data", "no_data"):
                g = "neg"
            else:
                g = e["group"]
            c, why = sf.code_check(o, SEASON, ref)
            B[(g, "n")] += 1
            B[(g, c)] += 1
            for pol in ("unsure", "all"):
                B[(g, "final_" + pol, sf.decide(c, ctx.get(no), pol))] += 1
            B[(g, "ctx", str(ctx.get(no)))] += 1
    res["B_stage2e"] = {" ".join(k): v for k, v in sorted(B.items())}
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]](*sys.argv[2:])
