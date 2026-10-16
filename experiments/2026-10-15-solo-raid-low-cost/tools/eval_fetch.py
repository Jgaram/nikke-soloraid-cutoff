"""평가 세트 사진(850px 사본 `url`)을 받아 IMGDIR/{id}.img로 저장하고 sha1을 확인한다.

python3 -I tools/eval_fetch.py data/evalset.jsonl IMGDIR

- 요청은 dcnet.py로만 한다 (UA Claude-User, 호스트별 1.2초, flock).
- 이미 받은 파일은 건너뛴다. 결과는 IMGDIR/fetch.jsonl (id, code, size, sha1_ok).
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcnet  # noqa: E402


def main():
    ev, imgdir = sys.argv[1], sys.argv[2]
    os.makedirs(imgdir, exist_ok=True)
    os.environ.setdefault("FETCHLOG", os.path.join(imgdir, "fetchlog.tsv"))
    dcnet.LOG = os.environ["FETCHLOG"]
    log = os.path.join(imgdir, "fetch.jsonl")
    done = {json.loads(l)["id"] for l in open(log)} if os.path.exists(log) else set()
    with open(log, "a") as out:
        for l in open(ev, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in done or not r.get("url"):
                continue
            path = os.path.join(imgdir, r["id"] + ".img")
            code, size = dcnet.get(r["url"], path + ".tmp", tries=4, timeout=60)
            ok = None
            if code == 200 and size > 0:
                h = hashlib.sha1(open(path + ".tmp", "rb").read()).hexdigest()
                ok = h == r.get("sha1")
                os.replace(path + ".tmp", path)
            out.write(json.dumps({"id": r["id"], "code": code, "size": size, "sha1_ok": ok}) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
