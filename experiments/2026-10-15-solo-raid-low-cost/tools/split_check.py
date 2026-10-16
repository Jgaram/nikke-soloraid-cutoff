"""F2: 나눠 돌린 것 = 기간 하나로 돌린 것인지 (요청 0). python3 -I split_check.py TOOLS STATE CHK OUT.json

S40 시범(2026-10-16): 기간을 S41 고정값 대신 season.json(c_start 또는 e_start ~ e_end)에서 읽고, 구간 수를 2개로 가정하지 않는다.
"""
import collections, json, os, shutil, subprocess, sys
T, S, C, OUT = sys.argv[1:5]
sys.path.insert(0, T)
from sites import sort_key  # noqa: E402  세 사이트 글 번호(숫자·n…·b…)가 섞인다
runs = sorted(os.listdir(os.path.join(S, "runs")))
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
tsv = lambda p: {l.split("\t")[0]: l.split("\t")[1] for l in open(p).read().split("\n")[1:] if l}
# 1) 목록 합치기 → 기간 하나로 후보
lst = {}
for r in runs:
    for x in jl(os.path.join(S, "runs", r, "list.jsonl")):
        lst[x["no"]] = x
season = json.load(open(os.path.join(S, "season.json")))
lo, hi = season.get("c_start") or season["e_start"], season["e_end"]
one = sorted((x for x in lst.values() if lo <= x["t"] <= hi), key=lambda x: sort_key(x["no"]))
os.makedirs(C, exist_ok=True)
with open(os.path.join(C, "list.jsonl"), "w", encoding="utf-8") as f:
    for x in one: f.write(json.dumps(x, ensure_ascii=False) + "\n")
subprocess.run([sys.executable, "-I", os.path.join(T, "select_cand.py"), os.path.join(C, "list.jsonl"), os.path.join(C, "cand"),
                season["e_start"], season["e_end"]], check=True, stdout=subprocess.DEVNULL)
cand1 = tsv(os.path.join(C, "cand", "cand.tsv"))
cand_split = {}
for r in runs: cand_split.update(tsv(os.path.join(S, "runs", r, "cand", "cand.tsv")))
nos = [open(os.path.join(S, "runs", r, "nos.txt")).read().split() for r in runs]
res = {"list_one": len(one), "list_split_sum": sum(len(jl(os.path.join(S, "runs", r, "list.jsonl"))) for r in runs),
       "cand_one": len(cand1), "cand_split": len(cand_split),
       "cand_only_one": sorted(set(cand1) - set(cand_split))[:20], "cand_only_split": sorted(set(cand_split) - set(cand1))[:20],
       "cand_rule_diff": sum(1 for n in set(cand1) & set(cand_split) if cand1[n] != cand_split[n]),
       "nos_overlap": sum(len(n) for n in nos) - len(set().union(*nos)), "nos_union": len(set().union(*nos)),
       "processed_eq_cand_one": set().union(*nos) == set(cand1)}
# 2) 길: 같은 manifest·parsed로 처음부터 한 번에
S2 = os.path.join(C, "STATE"); R2 = os.path.join(S2, "runs", "one")
shutil.rmtree(S2, ignore_errors=True); os.makedirs(R2)
for f in ("manifest.jsonl", "parsed.jsonl"): shutil.copy(os.path.join(S, f), S2)
subprocess.run([sys.executable, "-I", os.path.join(T, "route.py"), S2, R2], check=True, stdout=subprocess.DEVNULL)
a = {r["id"]: r for r in jl(os.path.join(S, "route.jsonl"))}
b = {r["id"]: r for r in jl(os.path.join(S2, "route.jsonl"))}
man = jl(os.path.join(S, "manifest.jsonl"))
ids = [m["id"] for m in man]
diff = [i for i in a if i in b and (a[i]["route"], a[i].get("same_as")) != (b[i]["route"], b[i].get("same_as"))]
res.update(manifest=len(man), manifest_dup_id=len(ids) - len(set(ids)), route_split=len(a), route_one=len(b),
           route_diff=len(diff), route_diff_ex=[(i, a[i]["route"], b[i]["route"]) for i in diff[:10]],
           route_counts_one=dict(collections.Counter(r["route"] for r in b.values())),
           posts_files=len(os.listdir(os.path.join(S, "posts"))))
# 3) 판독 대상 칸(strip·orig·whole)
tgt = lambda d: {i for i, r in d.items() if r["route"] in ("strip", "orig", "whole", "code")}
res["target_split"], res["target_one"], res["target_diff"] = len(tgt(a)), len(tgt(b)), len(tgt(a) ^ tgt(b))
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.endswith("_ex")}, ensure_ascii=False))
