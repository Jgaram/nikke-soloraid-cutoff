"""4단계: 이번 실행에서 받은 글의 숫자 추출과 Haiku 입력 배치를 만든다 (모델 0, 요청 0). run_window.sh가 부른다.

python3 -I tools/text_prep.py STATE RUN

- RUN/posts_min.jsonl: 이번 실행에서 받은 글(RUN/nos.txt)의 번호·사이트·시각·제목·본문 앞 500자·첨부 이름.
  STATE/posts_min.jsonl에도 합쳐 둔다 (실행을 거쳐 쌓임, 다음 실행의 「이미 받은 글」 판단과 merge_obs.py의 글 시각).
- RUN/text_extracted.jsonl: text_extract.py 결과 (코드 추출 + 규칙 분류).
- RUN/text/in_NN.jsonl: 추출 줄 전부를 200줄씩 (`delta` 제외). Haiku가 `kind`만 고른다 (세션 B: 규칙과 다르면 Haiku를 쓴다).
- RUN/ctx/in_NN.jsonl: 판독할 사진(code·strip·orig·whole·same)이 있는 글을 150개씩. Haiku가 이번 시즌 자료인지 글 문맥으로 고른다.
- RUN/text_prompt.md·RUN/ctx_prompt.md: tools/의 지시문 틀에 STATE/season.json의 시즌 설명을 채운 것.
- 표준 출력: {"posts":…, "text_rows":…, "text_batches":…, "ctx_posts":…, "ctx_batches":…}
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import season_filter  # noqa: E402
from sites import parse_no, site_of  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
READ = {"code", "strip", "orig", "whole", "same"}
TEXT_PER, CTX_PER = 200, 150


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def batches(rows, per, d):
    os.makedirs(d, exist_ok=True)
    n = 0
    for k in range(0, len(rows), per):
        n += 1
        with open(os.path.join(d, f"in_{n:02d}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows[k:k + per]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return n


def main(state, run):
    season = json.load(open(os.path.join(state, "season.json"), encoding="utf-8"))
    desc = (f"{season['name']}(보스 {season['boss']}, 이벤트 {season['start'][:16]} ~ {season['end'][:16]} KST, "
            f"챌린지 랭킹은 {season_filter.c_start(season)[:16]}부터)")
    for tpl, out in (("text_kind_prompt.md", "text_prompt.md"), ("season_ctx_prompt.md", "ctx_prompt.md")):
        s = open(os.path.join(HERE, tpl), encoding="utf-8").read().replace("{SEASON_DESC}", desc)
        open(os.path.join(run, out), "w", encoding="utf-8").write(s)
    nos = [parse_no(x) for x in open(os.path.join(run, "nos.txt")).read().split()]
    pm = []
    for no in nos:
        p = os.path.join(state, "posts", f"{no}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p, encoding="utf-8"))
        if r.get("err"):
            continue
        pm.append({"no": no, "site": site_of(no), "t": r.get("t"), "title": r.get("title") or "", "body": (r.get("body") or "")[:500],
                   "files": [f["name"] for f in r.get("files") or []]})
    pm_f = os.path.join(run, "posts_min.jsonl")
    with open(pm_f, "w", encoding="utf-8") as f:
        for r in pm:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # STATE/posts_min.jsonl: 받은 글 전체의 줄인 본문. posts/ 대신 커밋한다 (state_sync.py). 같은 글은 새 줄로 바꾼다
    sp = os.path.join(state, "posts_min.jsonl")
    mine = {r["no"] for r in pm}
    keep = [r for r in jl(sp) if r["no"] not in mine]
    with open(sp + ".tmp", "w", encoding="utf-8") as f:
        for r in keep + pm:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(sp + ".tmp", sp)
    ext_f = os.path.join(run, "text_extracted.jsonl")
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "text_extract.py"), pm_f, ext_f], check=True,
                   stdout=subprocess.DEVNULL)
    byno = {r["no"]: r for r in pm}
    rows = []
    for k, x in enumerate(jl(ext_f)):
        if x["kind"] == "delta":
            continue
        p = byno[x["no"]]
        rows.append({"id": k, "t": p["t"], "title": p["title"], "body": p["body"], "src": x["src"],
                     "seg": x["txt"], "pct": x["pct"], "score_eok": x["score_eok"]})
    nt = batches(rows, TEXT_PER, os.path.join(run, "text"))
    run_id = os.path.basename(os.path.normpath(run))
    want = {r["no"] for r in jl(os.path.join(state, "route.jsonl")) if r.get("run") == run_id and r["route"] in READ}
    ctx = [{"no": r["no"], "t": r["t"], "title": r["title"], "body": r["body"], "files": r["files"][:12]}
           for r in pm if r["no"] in want]
    nc = batches(ctx, CTX_PER, os.path.join(run, "ctx"))
    print(json.dumps({"posts": len(pm), "text_rows": len(rows), "text_batches": nt,
                      "ctx_posts": len(ctx), "ctx_batches": nc}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
