"""(S41 재수집 tools/에서 가져옴, 세션 F1 2026-10-15: 고치지 않음. 임시 파일은 현재 폴더에 생기므로 작업 폴더에서 실행한다)
글 번호 목록을 받아 본문·이미지 주소·댓글을 글마다 JSON으로 저장한다.

python3 -I post_fetch.py NOS.txt POSTDIR [--no-comments]

- 이미 받은 글(POSTDIR/{no}.json)은 건너뛴다 → 중단돼도 다시 돌리면 이어진다.
- 원본 HTML은 저장하지 않는다. 본문 텍스트, 이미지 주소, 첨부파일(이름·원본 링크), 댓글(작성자 정보 없이 시각·본문만)만 남긴다.
- 본문 이미지(viewimage.php)는 가로 850px로 줄인 표시용이다. 원본은 첨부파일 링크에만 있다.
- 댓글은 목록의 댓글 수와 상관없이 글 페이지를 받은 김에 요청한다 (쿠키·e_s_n_o 필요).
"""
import json
import os
import re
import sys

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcnet  # noqa: E402

VIEW = "https://gall.dcinside.com/mgallery/board/view/?id=gov&no={}"
CMT = "https://gall.dcinside.com/board/comment/"
TMP, JAR, CTMP = "view_tmp.html", "jar.txt", "cmt_tmp.json"


def text_of(html):
    return BeautifulSoup(html or "", "lxml").get_text(" ", strip=True)


def fetch(no, want_comments=True, tag=""):
    """tag: 여러 스레드가 같이 부를 때 임시 파일 이름을 가르는 꼬리 (collect.py 작업자 번호)."""
    TMP, JAR, CTMP = f"view_tmp{tag}.html", f"jar{tag}.txt", f"cmt_tmp{tag}.json"
    url = VIEW.format(no)
    if os.path.exists(JAR):
        os.remove(JAR)
    code, _ = dcnet.get(url, TMP, jar=JAR)
    if code != 200:
        return {"no": no, "err": f"http {code}"}
    s = BeautifulSoup(open(TMP, encoding="utf-8", errors="replace").read(), "lxml")
    body = s.select_one("div.write_div")
    if body is None:
        return {"no": no, "err": "no write_div (삭제?)"}
    imgs = []
    for im in body.select("img, video source"):
        u = im.get("data-original") or im.get("src") or ""
        if "dccon.php" in u or "gallview_loading" in u or not u:
            continue
        if "viewimage.php" in u or "dcimg" in u:
            imgs.append(u)
    movies = [f.get("src") for f in body.select("iframe") if f.get("src")]
    d = s.select_one(".gall_date")
    rec = {
        "no": no,
        "t": d.get("title") if d else "",
        "title": (s.select_one(".title_subject").get_text(strip=True)
                  if s.select_one(".title_subject") else ""),
        "body": body.get_text(" ", strip=True)[:3000],
        "imgs": list(dict.fromkeys(imgs)),
        "embeds": movies,
        # 첨부파일: 원본 해상도 링크(download.php, Referer 필요)와 올린 사람의 파일 이름.
        # 스크린샷 파일 이름에 찍은 시각이 남는 경우가 많다 (Screenshot_YYYYMMDD_HHMMSS…).
        "files": [{"name": a.get_text(strip=True), "href": a.get("href")}
                  for a in s.select(".appending_file a") if a.get("href")],
        "cmts": [],
    }
    tok = s.select_one("input[name=e_s_n_o]")
    if want_comments and tok:
        page, total = 1, None
        while True:
            data = (f"id=gov&no={no}&cmt_id=gov&cmt_no={no}&e_s_n_o={tok.get('value')}"
                    f"&comment_page={page}&_GALLTYPE_=M")
            c, _ = dcnet.get(CMT, CTMP, referer=url, data=data, jar=JAR,
                             headers=("X-Requested-With: XMLHttpRequest",))
            if c != 200:
                rec["cmt_err"] = f"http {c}"
                break
            try:
                j = json.load(open(CTMP, encoding="utf-8"))
            except ValueError:
                rec["cmt_err"] = "bad json"
                break
            total = int(j.get("total_cnt") or 0)
            for cm in j.get("comments") or []:
                if str(cm.get("no")) == "0":
                    continue
                memo = cm.get("memo") or ""
                rec["cmts"].append({
                    "d": cm.get("reg_date"),
                    "dep": int(cm.get("depth") or 0),
                    "m": text_of(memo)[:500],
                    "con": "dccon" in memo and not text_of(memo),
                })
            if page * 100 >= total:
                break
            page += 1
        rec["cmt_total"] = total
    return rec


def main():
    nos = [int(x) for x in open(sys.argv[1]).read().split()]
    outdir = sys.argv[2]
    want_c = "--no-comments" not in sys.argv
    os.makedirs(outdir, exist_ok=True)
    done = 0
    for no in nos:
        p = os.path.join(outdir, f"{no}.json")
        if os.path.exists(p):
            continue
        rec = fetch(no, want_c)
        json.dump(rec, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        done += 1
        if done % 25 == 0:
            print("fetched", done, flush=True)
    print("done", done)


if __name__ == "__main__":
    main()
