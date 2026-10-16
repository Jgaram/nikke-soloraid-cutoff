"""글 JSON의 이미지 주소를 받아 IMGDIR/{no}_{i}.img로 저장하고 목록을 만든다.

(S41 재수집 tools/에서 가져와 고침, 세션 F1 2026-10-15)
- 사진은 확장자 대신 `.img`로 저장한다 (ocr_run.py·tile_sheets.py가 `.img`를 읽는다). 종류는 목록의 `ext`. mp4는 `.mp4`.
- 목록 줄에 첨부 이름(`fname`)과 원본 링크(`orig`)를 붙인다. 본문 사진 수와 첨부 수가 같을 때만 i번째끼리 잇는다.
- GIF·WEBP 첫 프레임 PNG는 따로 만들지 않는다 (OCR이 첫 프레임을 읽는다).

python3 -I img_fetch.py NOS.txt POSTDIR IMGDIR MANIFEST.jsonl

- 파일 종류는 헤더로 판별 (디시는 Content-Type이 octet-stream). mp4는 저장만 하고 판독 대상에서 뺀다.
- GIF는 첫 프레임을 PNG로 따로 저장한다 (판독용).
- 목록 한 줄: id, no, i, ext, w, h, sha1. 같은 sha1은 판독 단계에서 한 번만 본다.
- 이미 받은 파일은 건너뛴다.
"""
import hashlib
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcnet  # noqa: E402

# 본문 사진도 글 주소를 Referer로 보낸다. dcimg8은 Referer가 없으면 403 (S37 2026-10-16, Referer를 주니 200)
VIEW = "https://gall.dcinside.com/mgallery/board/view/?id=gov&no={no}"


def kind(b):
    if b[:3] == b"\xff\xd8\xff":
        return "jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if b[:4] == b"GIF8":
        return "gif"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if b[4:8] == b"ftyp":
        return "mp4"
    return "bin"


def load_have(man):
    if not os.path.exists(man):
        return set()
    return {json.loads(l)["id"] for l in open(man, encoding="utf-8")}


def fetch_post(no, postdir, imgdir, out, have):
    """글 하나의 이미지를 받아 목록(out)에 한 줄씩 쓴다. 받은 장 수 반환."""
    p = os.path.join(postdir, f"{no}.json")
    if not os.path.exists(p):
        return 0
    rec = json.load(open(p, encoding="utf-8"))
    imgs = rec.get("imgs") or []
    files = rec.get("files") or []
    paired = len(files) == len(imgs)
    names = [f["name"].lower() for f in files] if paired else [""] * len(imgs)
    n = 0
    for i, u in enumerate(imgs):
        iid = f"{no}_{i}"
        if iid in have:
            continue
        if names[i].endswith(".gif"):
            # 움짤은 크고(평균 4MB) 시범 47장에서 자료가 0건이라 받지 않는다 (2026-10-15)
            out.write(json.dumps({"id": iid, "no": no, "i": i, "err": "skip gif", "fname": files[i]["name"]},
                                 ensure_ascii=False) + "\n")
            out.flush()
            have.add(iid)
            continue
        tmp = os.path.join(imgdir, iid + ".part")
        code, size = dcnet.get(u, tmp, referer=VIEW.format(no=no), tries=4, timeout=60)
        row = {"id": iid, "no": no, "i": i}
        if paired:
            row["fname"] = files[i]["name"]
            row["orig"] = files[i]["href"]
        if code != 200 or size == 0:
            row["err"] = f"http {code}"
            if os.path.exists(tmp):
                os.remove(tmp)
        else:
            b = open(tmp, "rb").read()
            ext = kind(b[:16])
            dst = os.path.join(imgdir, f"{iid}.{'mp4' if ext == 'mp4' else 'img'}")
            os.replace(tmp, dst)
            row.update(ext=ext, sha1=hashlib.sha1(b).hexdigest(), bytes=len(b))
            if ext in ("jpg", "png", "gif", "webp"):
                try:
                    im = Image.open(dst)
                    row["w"], row["h"] = im.size
                except Exception as e:  # 깨진 파일
                    row["err"] = f"decode {e.__class__.__name__}"
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        have.add(iid)
        n += 1
    return n


def main():
    nos = [int(x) for x in open(sys.argv[1]).read().split()]
    postdir, imgdir, man = sys.argv[2], sys.argv[3], sys.argv[4]
    os.makedirs(imgdir, exist_ok=True)
    have = load_have(man)
    out = open(man, "a", encoding="utf-8")
    n = 0
    for no in nos:
        n += fetch_post(no, postdir, imgdir, out, have)
    print("done", n)


if __name__ == "__main__":
    main()
