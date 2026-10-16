"""4단계: 새로 받은 사진마다 갈 길을 정한다 (모델 0, 요청 0). run_window.sh가 부른다.

python3 -I tools/route.py STATE RUN

- 입력: STATE/manifest.jsonl(받은 사진), STATE/parsed.jsonl(1단계 파서), STATE/route.jsonl(앞 실행에서 정한 길),
  STATE/pending_orig.txt(앞 실행 판독에서 목록 화면으로 나온 띠 사진, merge_obs.py가 쓴다).
- STATE/route.jsonl에 없는 사진만 정해서 이어 쓴다. 한 번 정한 사진은 다시 보지 않는다.
  같은 RUN을 다시 돌리면(중간 실패 뒤) 그 RUN이 정한 줄을 지우고 다시 정한다.
- 길 (`route`)
  - `same`: 앞서 받은 사진과 sha1이 같다. 다시 읽지 않고 `same_as`의 판독을 새 글에 붙인다.
  - `skip_err`·`skip_gif`·`skip_small`: 받기 실패·움짤·mp4, 작은 사진(짧은 변 150px 미만이고 막대 모양도 아님).
  - `drop_notext`·`drop_neg`·`drop_nohit`: 글자 없음, 파서가 모의전·다른 레이드·덱별로 자신 있게 정함, 숫자 없음.
  - `code`: 파서가 자신 있게 읽은 결과 팝업·내 순위·시즌 최종. 코드 판독을 그대로 쓴다.
  - `strip`: 파서가 순위·결과 화면으로 봤지만 자신 없음, 또는 잘라 낸 막대(숫자를 못 읽었어도). 850px 사본 띠 칸으로 Haiku가 읽는다.
  - `orig`: 파서가 순위 목록(`rank_list`)으로 본 사진 (자신 있어도). 원본 첨부를 폭 1400px 통째로 읽는다.
  - `whole`: 화면을 못 정했지만 숫자가 있다. 원본 첨부를 폭 1400px 통째로 읽는다 (원본이 없으면 850px 사본).
    세션 E는 2x2 축소 1차 → 순위 화면만 2차였지만, 목록이면 결국 원본이 필요해 처음부터 원본으로 읽는다 (세션 F1).
- 출력: RUN/strip_ids.txt, RUN/orig_ids.txt, RUN/orig_todo.jsonl(id, no, orig), RUN/route_counts.json
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tile_sheets import is_bar  # noqa: E402

POS = ("result_pop", "my_rank", "season_end")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def decide(m, p):
    if m.get("err") == "skip gif" or m.get("ext") == "mp4":
        return "skip_gif"
    if m.get("err") or m.get("ext") not in ("jpg", "png", "webp"):
        return "skip_err"
    w, h = m.get("w", 0), m.get("h", 0)
    bar = is_bar(w, h)
    if min(w, h) < 150 and not bar:
        return "skip_small"
    if p is None:
        return "skip_err"
    if "no_text" in p["why"]:
        return "drop_notext"
    if p["screen"] in ("mock", "other_raid", "deck") and p["conf"]:
        return "drop_neg"
    if p["screen"] == "rank_list":
        return "orig"
    if bar:
        # 막대는 숫자를 못 읽어도 띠로 읽는다: OCR이 이름·레벨만 읽은 내 줄 막대가 drop_nohit으로 빠졌다 (2026-10-16)
        return "strip"
    if p["screen"] in POS:
        return "code" if p["conf"] else "strip"
    return "whole" if p["hit"] else "drop_nohit"


def main(state, run):
    man = jl(os.path.join(state, "manifest.jsonl"))
    parsed = {p["id"]: p for p in jl(os.path.join(state, "parsed.jsonl"))}
    run_id = os.path.basename(os.path.normpath(run))
    old = [r for r in jl(os.path.join(state, "route.jsonl")) if r.get("run") != run_id]
    with open(os.path.join(state, "route.jsonl"), "w", encoding="utf-8") as f:
        for r in old:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    done = {r["id"] for r in old}
    first = {}
    for r in old:
        if r.get("sha1") and r["route"] != "same":
            first.setdefault(r["sha1"], r["id"])
    new, cnt = [], collections.Counter()
    strip, orig = [], []
    for m in man:
        if m["id"] in done:
            continue
        done.add(m["id"])
        r = {"id": m["id"], "no": m["no"], "sha1": m.get("sha1"), "run": run_id}
        if m.get("sha1") and m["sha1"] in first:
            r.update(route="same", same_as=first[m["sha1"]])
        else:
            p = parsed.get(m["id"])
            r["route"] = decide(m, p)
            r["screen"] = p["screen"] if p else None
            if m.get("sha1"):
                first[m["sha1"]] = m["id"]
        cnt[r["route"]] += 1
        if r["route"] == "strip":
            strip.append(m["id"])
        elif r["route"] in ("orig", "whole"):
            orig.append(m)
        new.append(r)
    pend_f = os.path.join(state, "pending_orig.txt")
    used_f = os.path.join(run, "pending_orig_used.txt")
    pend = open(pend_f).read().split() if os.path.exists(pend_f) else []
    pend += open(used_f).read().split() if os.path.exists(used_f) else []
    pend = list(dict.fromkeys(pend))
    byid = {m["id"]: m for m in man}
    for i in pend:
        if i in byid and byid[i] not in orig:
            orig.append(byid[i])
            cnt["pending_orig"] += 1
    with open(os.path.join(state, "route.jsonl"), "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    open(used_f, "w").write("".join(i + "\n" for i in pend))
    if os.path.exists(pend_f):
        os.remove(pend_f)
    open(os.path.join(run, "strip_ids.txt"), "w").write("".join(i + "\n" for i in strip))
    open(os.path.join(run, "orig_ids.txt"), "w").write("".join(m["id"] + "\n" for m in orig))
    with open(os.path.join(run, "orig_todo.jsonl"), "w", encoding="utf-8") as f:
        for m in orig:
            f.write(json.dumps({"id": m["id"], "no": m["no"], "orig": m.get("orig")}, ensure_ascii=False) + "\n")
    json.dump(dict(cnt), open(os.path.join(run, "route_counts.json"), "w"))
    print(json.dumps(dict(cnt)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
