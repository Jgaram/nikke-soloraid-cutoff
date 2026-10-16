"""목록 JSONL에서 제목 규칙으로 후보를 고른다.

python3 -I select_cand.py LIST.jsonl OUTDIR [E_START E_END]

(S41 재수집 tools/에서 가져와 고침, 세션 F1 2026-10-15)
- D 추가: %·퍼 없이 소수만 있는 제목("0.7딱코", "엇 0.5는 안댓네"). S41에서 기준점 3글을 놓친 규칙.
- E 기간을 인자로 받는다 (기본은 S41 값). C 표본은 뽑지 않는다.
- OUTDIR/cand.tsv (no, rule)도 쓴다. 규칙 순서는 A → B → D → E (먼저 걸린 것 하나).

등급
- A: 솔로레이드를 직접 가리키는 말 (솔레·솔로레이드·리버렐리오·바디 등)
- B: 순위·점수 관련 말 (컷·등수·순위·상위·n%·n등·n위·n억·딜량·막트 등)
- C: 나머지 이미지 글 (icon_pic) — 표본만 뽑아 누락률을 잰다
- E: C 중 확장 규칙에 걸리는 글 (레이드 기간 09-24 12:00 ~ 10-02 12:00). 키워드 없는 이미지 글 표본 250건을
  판독해 보니 17건에 시즌41 자료가 있었고(제목: "기갱 안해도 되지?", "글섭 전광판 뭐임" 등), 그 말들로 만든 규칙.
  표본 기준 자료 글 17건 중 12건(순위·% 자료 6건 중 5건)을 잡고 250건 중 29건만 걸린다.
OUTDIR에 a.txt, b.txt, c_all.txt, e.txt, counts.json을 쓴다.
"""
import json
import os
import re
import sys

A = re.compile(r"솔레|솔로\s*레|솔로레|솔랭|리버렐|리버|바디|레이드|솔컷|작열")
B = re.compile(
    r"컷|등수|순위|랭킹|랭커|상위|하위|퍼센|\d\s*%|\d\s*퍼|[0-9,.]+\s*등(?![장록급대산교])|"
    r"[0-9,.]+\s*위(?![로에치])|[0-9.]+\s*억|[0-9.]+\s*조(?![합건])|만\s*등|천\s*등|백\s*등|"
    r"몇\s*등|딜량|딜\s*[0-9]|점수|막트|[0-9]\s*트(?![럭윗])|정산|프레임|보상|등급|커트")


D = re.compile(r"(?<![\d.])\d\.\d{1,2}(?![\d.]|\s*(?:%|퍼|주년|배|초|버전|ver|v\b))")
E = re.compile(r"기갱|갱신|계산기|전광판|택틱|덱|딜|방무|스쿼드|모의전|이론|트라이|[0-9]\s*(->|→)|막타|\d트|점수|등|위|퍼|억|%|컷|랭|"
               r"레이드|보스|칠걸|쳤|치고|친다|솔|리버|바디")
EWIN = ("2026-09-24 12:00:00", "2026-10-02 12:00:00")


def main():
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    out = sys.argv[2]
    ewin = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else EWIN
    os.makedirs(out, exist_ok=True)
    a, b, d, c = [], [], [], []
    for r in rows:
        t = r["title"]
        if A.search(t):
            a.append(r["no"])
        elif B.search(t):
            b.append(r["no"])
        elif D.search(t):
            d.append(r["no"])
        elif r["type"] in ("icon_pic", "icon_recomimg"):
            c.append(r["no"])
    byno = {r["no"]: r for r in rows}
    e = [n for n in c if ewin[0] <= byno[n]["t"] <= ewin[1] and E.search(byno[n]["title"])]
    for name, lst in (("a.txt", a), ("b.txt", b), ("d.txt", d), ("c_all.txt", c), ("e.txt", e)):
        open(os.path.join(out, name), "w").write("\n".join(map(str, lst)) + "\n")
    with open(os.path.join(out, "cand.tsv"), "w") as f:
        f.write("no\trule\n")
        for rule, lst in (("A", a), ("B", b), ("D", d), ("E", e)):
            for n in lst:
                f.write(f"{n}\t{rule}\n")
    counts = {"total": len(rows), "A": len(a), "B": len(b), "D": len(d), "C_img": len(c), "E": len(e)}
    json.dump(counts, open(os.path.join(out, "counts.json"), "w"))
    print(counts)


if __name__ == "__main__":
    main()
