"""(S41 재수집 tools/에서 가져옴, 세션 F1 2026-10-15: 임시 파일 위치만 바꿈)
니마갤 전체글 목록을 기간으로 잘라 JSONL로 저장한다.

python3 -I list_crawl.py START END OUT.jsonl   (시각은 KST 'YYYY-MM-DD HH:MM:SS')

1) 이분 탐색으로 END 직후가 들어 있는 페이지를 찾는다.
2) 그 페이지부터 과거로 넘기며 START 이전 글이 나올 때까지 받는다.
작성자 정보는 남기지 않는다 (번호·시각·글 종류·제목·댓글·추천·조회만).
"""
import json
import os
import sys

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcnet  # noqa: E402

BASE = "https://gall.dcinside.com/mgallery/board/lists/?id=gov&list_num=100&page={}"
TMP = "list_tmp.html"  # main에서 OUT 옆으로 옮긴다


def page(n):
    code, _ = dcnet.get(BASE.format(n), TMP)
    if code != 200:
        raise SystemExit(f"page {n}: http {code}")
    s = BeautifulSoup(open(TMP, encoding="utf-8").read(), "lxml")
    rows = []
    for r in s.select("tr.ub-content[data-no]"):
        if r.get("data-type") == "icon_notice":
            continue
        d = r.select_one(".gall_date")
        a = r.select_one(".gall_tit a")
        if not d or not a or not d.get("title"):
            continue
        rep = r.select_one(".reply_num")
        rows.append({
            "no": int(r["data-no"]),
            "t": d["title"],
            "type": r.get("data-type"),
            "title": a.get_text(" ", strip=True),
            "head": (r.select_one(".gall_subject").get_text(strip=True)
                     if r.select_one(".gall_subject") else ""),
            "rep": int(rep.get_text(strip=True).strip("[]").split("/")[0] or 0) if rep else 0,
            "rec": int(r.select_one(".gall_recommend").get_text(strip=True) or 0),
            "view": int((r.select_one(".gall_count").get_text(strip=True) or "0").replace(",", "") or 0),
        })
    return rows


def main():
    global TMP
    start, end, out = sys.argv[1], sys.argv[2], sys.argv[3]
    TMP = os.path.join(os.path.dirname(os.path.abspath(out)), "list_tmp.html")
    # 이분 탐색: 페이지의 가장 오래된 글 시각 <= end 인 첫 페이지
    # 글이 없는 쪽(게시판 끝 너머)은 END보다 오래된 쪽으로 친다 (S40 시범, 2026-10-16: 23개월 전을 찾다 65536쪽이 비어 멈췄다)
    def oldest(n):
        rows = page(n)
        return rows[-1]["t"] if rows else ""
    lo, hi = 1, 2
    while oldest(hi) > end:
        lo, hi = hi, hi * 2
    while lo < hi:
        mid = (lo + hi) // 2
        if oldest(mid) > end:
            lo = mid + 1
        else:
            hi = mid
    n = lo
    print("start page", n, flush=True)
    seen = {}
    while True:
        rows = page(n)
        for r in rows:
            if start <= r["t"] <= end:
                seen[r["no"]] = r
        if not rows or rows[-1]["t"] < start:
            break
        if n % 20 == 0:
            print(n, rows[-1]["t"], len(seen), flush=True)
        n += 1
    with open(out, "w", encoding="utf-8") as f:
        for no in sorted(seen):
            f.write(json.dumps(seen[no], ensure_ascii=False) + "\n")
    print("pages", n, "posts", len(seen))


if __name__ == "__main__":
    main()
