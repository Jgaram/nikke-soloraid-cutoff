"""순서표 3 「자료 고침」 반영 — review.json의 최종 제안(`kind` `s3_final`)을 discard.json에 옮긴다 (2026-10-20, 모델 0, 요청 0).

docs/solo-raid-todo.md 3 · docs/solo-raid-analysis.md 5.1. 시즌 경계에 S42 채점(순서표 1) 뒤에만 --write로 돌린다
(S42 동안 discard.json을 바꾸면 후처리 도장이 stale이 된다). 기본은 보기만 한다.

  python3 -I tools/solo_raid/lab5/s3_apply.py                       # 무엇을 옮길지 보기만
  python3 -I tools/solo_raid/lab5/s3_apply.py --write --by "…세션"   # discard.json에 쓰고 review.json decision을 바꿈
  python3 -I tools/solo_raid/lab5/s3_apply.py --root DIR            # 레포 사본에 (모의 실행)

- `s3_final`의 items는 4.2b 확신 높은 제안(`gross_42b`)과 2026-10-20 원인 조사(`premise_c2`·`premise_c1`·`s3_group`·`s3_new`, 기존 항목의 `s3` 칸)를 합치고,
  조사가 대신한 4.2b 제안(`superseded` 칸이 붙은 항목)을 뺀 것이다. 항목마다 `review`에 근거가 된 review.json 항목 id가 있다.
- reason은 근거 항목의 note(4.2b는 「/ 제안」 앞, 조사 항목은 s3.note)에서, ref는 근거 종류에 따라 붙인다.
- 같은 대상(scope·글·사진·줄)에 이미 항목이 있으면: 새 항목이 버림이면 옛 항목을 대신하고(`replaces`에 옛 action을 남김), 아니면 멈춘다.
  같은 내용이 이미 있으면 건너뛴다 (두 번 돌려도 같다).
- --write면 근거 항목의 decision을 한 일로 바꾼다 (줄 버림 row · 글 버림 discard · 서버 고침 server). 조사에서 오류 아님으로 본 항목(s3 verdict keep, 제안 없음)은 keep.
  4.2b가 대신된 항목은 decision을 `later` 그대로 두고 `superseded`를 남긴다. 옮긴 뒤 analyze_all.py로 다시 돈다 (순서표 5).
"""
import argparse
import glob
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REF_42B = "docs/solo-raid-analysis.md 4.2b 굵은 오차 36칸 점검"
REF_S3 = "docs/solo-raid-todo.md 3 · docs/solo-raid-analysis.md 5.1 (2026-10-20 원인 조사)"


def tkey(i):
    return (i["scope"], str(i["no"]) if i["scope"] == "post" else None, i.get("img"), i.get("row"))


def same(a, b):
    f = ("scope", "action", "server", "img", "row", "correct")
    return tkey(a) == tkey(b) and all(a.get(k, "discard" if k == "action" else None) == b.get(k, "discard" if k == "action" else None) for k in f)


def note_of(it):
    if it.get("s3", {}).get("note") and it["kind"] != "gross_42b":
        return it["s3"]["note"]
    if it["kind"] == "gross_42b":
        return it["note"].split(" / 제안")[0]
    return it.get("note") or it.get("s3", {}).get("note", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--decided", default=None, help="discard.json의 decided (기본: 오늘 KST 날짜)")
    ap.add_argument("--by", default="순서표 3 반영 (제안: 4.2b 점검 · 순서표 3 원인 조사 2026-10-20)")
    a = ap.parse_args()
    if a.decided is None:
        import datetime as dt
        a.decided = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d")
    spec = importlib.util.spec_from_file_location("discard", os.path.join(a.root, "tools", "solo_raid", "discard.py"))
    DC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DC)
    tot = {"add": 0, "replace": 0, "skip": 0}
    for rp in sorted(glob.glob(os.path.join(a.root, "data", "solo-raid", "S*", "review.json")), key=lambda p: int(p.split(os.sep)[-2][1:])):
        s = rp.split(os.sep)[-2]
        R = json.load(open(rp, encoding="utf-8"))
        byid = {x["id"]: x for x in R["items"]}
        fin = byid.get(f"s3_final:{s}")
        if not fin:
            continue
        dp = os.path.join(a.root, "data", "solo-raid", s, "discard.json")
        D = json.load(open(dp, encoding="utf-8")) if os.path.exists(dp) else {
            "about": "발견되면 버릴 것·고칠 것 (할 일 3.A2). 이 시즌 세션·점검 세션이 고친다. post_all.sh가 늘 다시 적용하고, 원자료는 고치지 않는다. "
                     "꼴과 action·scope의 뜻은 tools/solo_raid/discard.py 머리말.", "items": []}
        n0 = dict(tot)
        for it in fin["items"]:
            refs = it["review"]
            src = [byid[r] for r in refs if r in byid]
            g = next((x for x in src if x["kind"] == "gross_42b"), None)
            why = note_of(g) if g else (note_of(src[0]) if src else "")
            ref = REF_42B if g else REF_S3
            if g and any(x["kind"] != "gross_42b" for x in src):
                ref = f"{REF_42B} · {REF_S3}"
            x = {k: v for k, v in it.items() if k != "review"}
            x.update(reason=why[:600], decided=a.decided, by=a.by, ref=ref, review=refs)
            DC.check(x, s)
            old = [i for i in D["items"] if tkey(i) == tkey(x)]
            if old and same(old[0], x):
                tot["skip"] += 1
                continue
            if old:
                o = old[0]
                assert x.get("action", "discard") == "discard", f"{s} {tkey(x)}: 이미 다른 항목이 있다 — {o}"
                x["reason"] = f"{o.get('decided')} {o.get('action')} {o.get('server') or ''} 고침을 대신함 — " + x["reason"]
                x["replaces"] = {k: o.get(k) for k in ("action", "server", "decided", "by")}
                D["items"] = [i for i in D["items"] if tkey(i) != tkey(x)]
                tot["replace"] += 1
            else:
                tot["add"] += 1
            D["items"].append(x)
            print(f"  {s} {x['scope']}/{x.get('action', 'discard')}{'/' + x['server'] if x.get('server') else ''} "
                  f"{x.get('img') or x['no']}{'#' + str(x['row']) if 'row' in x else ''}{' (대신함)' if old else ''} ← {', '.join(refs)}")
            if a.write:
                dec = "server" if x.get("action") == "server" else ("discard" if x["scope"] == "post" else "row")
                for r in refs:
                    if r in byid and not byid[r].get("superseded"):
                        byid[r]["decision"] = dec
                        byid[r]["applied"] = a.decided
        if a.write:
            for x in R["items"]:
                s3 = x.get("s3")
                if s3 and s3.get("verdict") == "keep" and not s3.get("proposed") and x.get("decision") == "later":
                    x["decision"] = "keep"
            fin["decision"] = "applied"
            fin["applied"] = a.decided
            with open(dp, "w", encoding="utf-8") as f:
                json.dump(D, f, ensure_ascii=False, indent=1)
                f.write("\n")
            with open(rp, "w", encoding="utf-8") as f:
                json.dump(R, f, ensure_ascii=False, indent=1)
                f.write("\n")
        print(f"{s}: 더함 {tot['add'] - n0['add']} · 대신함 {tot['replace'] - n0['replace']} · 이미 있음 {tot['skip'] - n0['skip']}")
    print(f"합계 {tot} — {'썼다. 다음: python3 -I tools/solo_raid/analyze_all.py' if a.write else '보기만 (--write로 쓴다)'}")


if __name__ == "__main__":
    main()
