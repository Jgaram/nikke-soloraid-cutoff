"""네이버 게임 라운지·블라블라링크를 디시와 같은 모양(글 JSON·사진 목록)으로 받는다 (2026-10-15, 디시와 묶기 준비).

python3 -I other_sites.py list SITE START END OUT.jsonl     SITE = naver | blabla, 시각은 KST 'YYYY-MM-DD HH:MM:SS'
python3 -I other_sites.py posts LIST.jsonl NOS.txt POSTDIR   목록 줄에서 글 JSON을 만든다 (요청 0)
python3 -I other_sites.py collect NOS.txt POSTDIR IMGDIR MANIFEST.jsonl
python3 -I other_sites.py orig ORIG_TODO.jsonl IDS.txt ORIGDIR   (디시 줄은 건너뛴다 → orig_fetch.py)

- 글 번호는 사이트 머리글자를 붙인 문자열이다: 네이버 `n{feedId}`, 블라블라링크 `b{post_uuid}`. 디시 번호(숫자)와 겹치지 않는다.
- 두 사이트 모두 목록 응답에 본문과 사진 주소가 다 들어 있어 글을 따로 요청하지 않는다 (docs/community-access.md).
  목록 줄(LIST.jsonl)이 디시 list_crawl.py 줄(no·t·type·title)에 본문·사진·첨부 이름을 더한 모양이라 select_cand.py를 그대로 쓴다.
- 사진은 디시 850px 사본에 맞춰 줄인 것을 받고(`imgs`), 원본 주소는 `files[].href`에 둔다 → img_fetch.py가 목록 줄의 `orig`로 잇는다.
  - 네이버: `?type=w800` (w966은 404, 2026-10-15). 원본은 쿼리 없는 주소. 첨부 이름은 주소 끝 파일 이름이다 (스크린샷 시각).
  - 블라블라링크: `?imageMogr2/thumbnail/850x/format/jpg` (원본 PNG 3.4MB → 약 14만 바이트). 원본은 쿼리 없는 주소.
- 네이버 사진 서버는 연결이 응답 없이 끊기는 일이 잦다(상태 0). img_fetch.py가 4번까지 다시 요청한다.
- 요청은 dcnet.py로만 한다 (UA는 DCNET_UA, 호스트별 간격, 403·429면 멈춤). 작성자 정보는 남기지 않는다.
- collect: 사이트마다 사진 작업자 COLLECT_WORKERS개(기본 1)를 따로 돌린다. 두 사이트는 호스트가 달라 동시에 받는다.
"""
import json
import os
import re
import sys
import threading
import time
from html import unescape
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcnet  # noqa: E402
import img_fetch  # noqa: E402

NAVER = "https://comm-api.game.naver.com/nng_main/v1/community/lounge/nikke/feed?boardId=0&buffFilteringYN=N&limit=25&offset={}&order=NEW"
BLA = "https://api.blablalink.com/api/ugc/direct/standalonesite/Dynamics/GetPostListNew"
BLA_H = ("Content-Type: application/json", "x-channel-type: 2", "x-language: ko",
         'x-common-params: {"game_id":"","area_id":"","source":"pc_web","intl_game_id":"","language":"ko","env":"prod"}')
BLA_SMALL = "?imageMogr2/thumbnail/850x/format/jpg"
TMP = "other_tmp.json"


def kst(sec):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(sec) + 9 * 3600))


def html_text(h):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", h or ""))).strip()


def get_json(url, **kw):
    code, _ = dcnet.get(url, TMP, tries=4, **kw)
    if dcnet.BLOCKED:
        raise SystemExit(f"blocked {dcnet.BLOCKED[:3]}")
    if code != 200:
        raise SystemExit(f"http {code} {url}")
    return json.load(open(TMP, encoding="utf-8"))


def se_walk(x, texts, imgs):
    """네이버 SmartEditor JSON에서 글자와 본문 사진 주소를 모은다 (스티커·링크 미리보기 제외)."""
    if isinstance(x, dict):
        ct = x.get("@ctype")
        if ct == "textNode" and isinstance(x.get("value"), str):
            texts.append(x["value"])
        elif ct == "image" and "pstatic.net" in (x.get("src") or ""):
            imgs.append(x["src"].split("?")[0])
            return
        elif ct in ("sticker", "oglink", "oembed", "video"):
            return
        for v in x.values():
            se_walk(v, texts, imgs)
    elif isinstance(x, list):
        for v in x:
            se_walk(v, texts, imgs)


def row(no, t, title, body, base_imgs, small, site, url):
    """base_imgs: 쿼리 없는 원본 주소. 사이트 CDN이 아닌 주소(블라블라링크 글의 firebasestorage 등)는 쿼리가 열쇠라 그대로 둔다."""
    names = [unquote(os.path.basename(urlparse(u).path)) for u in base_imgs]
    own = lambda u: "pstatic.net" in u or "blablalink.com" in u  # noqa: E731
    return {"no": no, "t": t, "type": "icon_pic" if base_imgs else "icon_txt", "title": title, "site": site, "url": url,
            "body": body[:3000], "imgs": [u + small if own(u) else u for u in base_imgs],
            "files": [{"name": n, "href": u} for n, u in zip(names, base_imgs)]}


def naver_page(off):
    out = []
    for f in (get_json(NAVER.format(off)).get("content") or {}).get("feeds") or []:
        fd = f["feed"]
        d = fd["createdDate"]
        t = f"{d[:4]}-{d[4:6]}-{d[6:8]} {d[8:10]}:{d[10:12]}:{d[12:14]}"
        texts, imgs = [], []
        try:
            se_walk(json.loads(fd.get("contents") or "{}"), texts, imgs)
        except ValueError:
            texts.append(html_text(fd.get("contents")))
        out.append(row(f"n{fd['feedId']}", t, fd.get("title") or "", " ".join(x for x in texts if x.strip()),
                       list(dict.fromkeys(imgs)), "?type=w800", "naver",
                       f"https://game.naver.com/lounge/nikke/board/detail/{fd['feedId']}"))
    return out


def naver_list(start, end):
    """offset은 페이지 번호다. 고정글이 맨 앞에 섞일 수 있어 페이지의 가장 오래된 시각으로 찾는다."""
    oldest = lambda rows: min(r["t"] for r in rows) if rows else ""  # noqa: E731
    lo, hi = 0, 1
    while oldest(naver_page(hi)) > end:
        lo, hi = hi, hi * 2
    while lo < hi:
        mid = (lo + hi) // 2
        if oldest(naver_page(mid)) > end:
            lo = mid + 1
        else:
            hi = mid
    n, seen = lo, {}
    print("naver start page", n, flush=True)
    while True:
        rows = naver_page(n)
        for r in rows:
            if start <= r["t"] <= end:
                seen[r["no"]] = r
        if not rows or oldest(rows) < start:
            break
        n += 1
    print("naver pages", n - lo + 1, "posts", len(seen), flush=True)
    return seen


BLA_FIRST = "2024-12-19 21:13:53"  # 한국어 글 목록의 가장 오래된 글 (S40 시범 2026-10-16, 433쪽 끝에서 is_finish)


def bla_list(start, end):
    """한국어 글(regions ko)을 100개씩 최신부터 넘긴다. 커서라 건너뛸 수 없다.

    S40 시범(2026-10-16): 지난 시즌일수록 구간마다 수백 쪽을 넘긴다 (2024-12까지 433쪽, 31분, 약 200MB).
    환경 변수 BLA_CACHE(run_window.sh가 STATE/bla_cache.jsonl로 준다)가 있으면 넘긴 글 중 START 뒤 것을 모두 저장하고,
    다음 구간이 그 범위 안이면 다시 넘기지 않는다. 시즌 구간은 오래된 것부터 돌므로 첫 구간 한 번만 넘긴다.
    """
    cache = os.environ.get("BLA_CACHE")
    if end < BLA_FIRST:
        print("blabla pages 0 posts 0 (BLA_FIRST 전)", flush=True)
        return {}
    if cache and os.path.exists(cache):
        rows = [json.loads(l) for l in open(cache, encoding="utf-8") if l.strip()]
        meta = rows[0] if rows and "_cache" in rows[0] else None
        if meta and meta["from"] <= start and end <= meta["to"]:
            seen = {r["no"]: r for r in rows[1:] if start <= r["t"] <= end}
            print("blabla pages 0 posts", len(seen), "(cache)", flush=True)
            return seen
    cur, seen, pages = "", {}, 0
    keep, newest = {}, ""
    while True:
        body = {"search_type": 0, "plate_id": 46, "plate_unique_id": "recommend", "order_by": 1, "limit": "100",
                "regions": ["ko"], "next_page_cursor": cur}
        d = get_json(BLA, data=json.dumps(body), headers=BLA_H, timeout=120).get("data") or {}  # 100개 약 55만 바이트, 40초를 넘기기도 한다
        pages += 1
        lst = d.get("list") or []
        for x in lst:
            t = kst(x["created_on"])
            newest = max(newest, t)
            if start <= t <= end or (cache and start <= t):
                base = list(dict.fromkeys(u.split("?")[0] if "blablalink.com" in u else u for u in x.get("pic_urls") or []))
                r = row("b" + x["post_uuid"], t, x.get("title") or "", html_text(x.get("content")),
                        base, BLA_SMALL, "blabla", f"https://www.blablalink.com/post/{x['post_uuid']}")
                keep[r["no"]] = r
                if start <= t <= end:
                    seen[r["no"]] = r
        cur = (d.get("page_info") or {}).get("next_page_cursor")
        if not lst or not cur or d["page_info"].get("is_finish") or kst(lst[-1]["created_on"]) < start:
            break
    if cache:
        with open(cache + ".tmp", "w", encoding="utf-8") as f:
            f.write(json.dumps({"_cache": 1, "from": start, "to": newest}, ensure_ascii=False) + "\n")
            for r in keep.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(cache + ".tmp", cache)
    print("blabla pages", pages, "posts", len(seen), flush=True)
    return seen


def cmd_list(site, start, end, out):
    global TMP
    TMP = os.path.join(os.path.dirname(os.path.abspath(out)), f"{site}_tmp.json")
    seen = (naver_list if site == "naver" else bla_list)(start, end)
    with open(out, "w", encoding="utf-8") as f:
        for r in sorted(seen.values(), key=lambda r: r["t"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_posts(listf, nosf, postdir):
    rows = {r["no"]: r for r in map(json.loads, open(listf, encoding="utf-8"))}
    os.makedirs(postdir, exist_ok=True)
    n = 0
    for no in open(nosf).read().split():
        p = os.path.join(postdir, f"{no}.json")
        if no in rows and not os.path.exists(p):
            r = dict(rows[no], embeds=[], cmts=[])
            r.pop("type", None)
            json.dump(r, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(p + ".tmp", p)
            n += 1
    print("posts", n)


def cmd_collect(nosf, postdir, imgdir, man):
    os.makedirs(imgdir, exist_ok=True)
    w = max(1, int(os.environ.get("COLLECT_WORKERS") or 1))
    nos = open(nosf).read().split()
    have = img_fetch.load_have(man)
    lock, err, cnt = threading.Lock(), [], {"n": 0}

    class Out:
        def __init__(self, f):
            self.f = f

        def write(self, s):
            with lock:
                self.f.write(s)

        def flush(self):
            with lock:
                self.f.flush()

    def worker(q):
        try:
            while q and not dcnet.BLOCKED:
                with lock:
                    if not q:
                        return
                    no = q.pop(0)
                img_fetch.fetch_post(no, postdir, imgdir, out, have)
                with lock:
                    cnt["n"] += 1
                    if cnt["n"] % 50 == 0:
                        print("posts", cnt["n"], flush=True)
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    with open(man, "a", encoding="utf-8") as f:
        out = Out(f)
        ts = []
        for pre in ("n", "b"):
            q = [x for x in nos if x.startswith(pre)]
            ts += [threading.Thread(target=worker, args=(q,)) for _ in range(w)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    if err:
        raise err[0]
    if dcnet.BLOCKED:
        raise SystemExit(f"blocked {dcnet.BLOCKED[:3]}")
    print("done", cnt["n"], "workers", w, flush=True)


def cmd_orig(todo, idsf, origdir):
    os.makedirs(origdir, exist_ok=True)
    ids = set(open(idsf).read().split())
    log = os.path.join(origdir, "fetch.jsonl")
    done = {json.loads(l)["id"] for l in open(log)} if os.path.exists(log) else set()
    with open(log, "a") as out:
        for r in map(json.loads, open(todo, encoding="utf-8")):
            if r["id"] not in ids or r["id"] in done or not r.get("orig") or str(r["no"]).isdigit():
                continue   # 디시 글(숫자 번호)은 orig_fetch.py가 Referer를 붙여 받는다
            path = os.path.join(origdir, r["id"] + ".img")
            code, size = dcnet.get(r["orig"], path + ".tmp", tries=4, timeout=90)
            if dcnet.BLOCKED:
                raise SystemExit(f"blocked {dcnet.BLOCKED[:3]}")
            if code == 200 and size > 0:
                os.replace(path + ".tmp", path)
            out.write(json.dumps({"id": r["id"], "code": code, "size": size}) + "\n")
            out.flush()


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    {"list": cmd_list, "posts": cmd_posts, "collect": cmd_collect, "orig": cmd_orig}[c](*a)
