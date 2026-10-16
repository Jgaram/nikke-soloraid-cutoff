"""평가 세트 사진의 원본 첨부(`orig`)를 받아 IMGDIR/{id}.img로 저장한다.

python3 -I tools/orig_fetch.py EVALSET.jsonl IDS.txt IMGDIR

- 요청은 dcnet.py로만 한다 (UA Claude-User, 호스트별 1.2초, flock).
- 원본 첨부(`image.dcinside.com/download.php`)는 Referer에 글 주소가 있어야 받아진다. 없으면 200에 0바이트가 온다 (2026-10-15 확인).
- `orig`가 없는 사진은 건너뛴다 (850px 사본을 그대로 쓴다). 이미 받은 파일도 건너뛴다.
- 결과는 IMGDIR/fetch.jsonl (id, code, size).
- 디시 글(번호가 숫자)만 받는다. 네이버·블라블라링크 줄은 `other_sites.py orig`가 같은 목록에서 맡는다 (세 사이트 묶기, 2026-10-15).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcnet  # noqa: E402

VIEW = "https://gall.dcinside.com/mgallery/board/view/?id=gov&no={no}"


def main():
    ev, ids_f, imgdir = sys.argv[1:4]
    os.makedirs(imgdir, exist_ok=True)
    os.environ.setdefault("FETCHLOG", os.path.join(imgdir, "fetchlog.tsv"))
    dcnet.LOG = os.environ["FETCHLOG"]
    ids = set(open(ids_f).read().split())
    log = os.path.join(imgdir, "fetch.jsonl")
    done = {json.loads(l)["id"] for l in open(log)} if os.path.exists(log) else set()
    with open(log, "a") as out:
        for l in open(ev, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] not in ids or r["id"] in done or not r.get("orig") or not str(r["no"]).isdigit():
                continue
            path = os.path.join(imgdir, r["id"] + ".img")
            code, size = dcnet.get(r["orig"], path + ".tmp", referer=VIEW.format(no=r["no"]), tries=3, timeout=60)
            if dcnet.BLOCKED:
                raise SystemExit(f"blocked {dcnet.BLOCKED[:3]}")
            if code == 200 and size > 0:
                os.replace(path + ".tmp", path)
            out.write(json.dumps({"id": r["id"], "code": code, "size": size}) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
