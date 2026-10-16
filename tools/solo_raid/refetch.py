"""솔로레이드 점검용 사진 다시 받기 — 시즌당 요청 상한 안에서 (2026-10-16, 사용자 결정: 시즌당 200번, 자기 이름 UA).

python3 -I tools/solo_raid/refetch.py data/solo-raid/S{n}/state OUTDIR ID [ID ...] [--max 200] [--user-request]

- ID는 manifest.jsonl의 사진 id (예: 5881317_0, n7969682_0, b9628780651271152930_1). 원본 주소(orig)를 받는다.
  디시는 글 주소를 Referer로 보낸다 (dcimg8은 Referer가 없으면 403, 2026-10-16).
- OUTDIR은 레포 밖(세션 임시 폴더)의 빈 폴더. 사진은 커밋하지 않는다. 이미 받은 파일은 다시 요청하지 않는다.
- 요청마다 data/solo-raid/S{n}/review_fetch.tsv에 한 줄 (시각 KST, id, 상태, 바이트, UA 종류). 이 파일 줄 수 + 이번 요청 수가
  --max(기본 200)를 넘으면 아무것도 받지 않고 멈춘다. 넘겨야 하면 사용자에게 묻는다.
- UA: 사람 없이 도는 점검이라 기본은 자기 이름 UA(dcnet.AUTO_UA). 사용자가 직접 요청한 확인이면 --user-request로 `Claude-User`
  (docs/community-access.md 공통 원칙).
- 403·429가 나오면 그 자리에서 멈춘다 (다시 시도하지 않는다).
"""
import argparse
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOWCOST = os.path.join(ROOT, "experiments", "2026-10-15-solo-raid-low-cost", "tools")
DATA = os.path.join(ROOT, "data", "solo-raid")
VIEW = "https://gall.dcinside.com/mgallery/board/view/?id=gov&no={no}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("out")
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--user-request", action="store_true")
    a = ap.parse_args()
    name = json.load(open(os.path.join(a.state, "season.json"), encoding="utf-8"))["name"]
    log = os.path.join(DATA, name, "review_fetch.tsv")
    used = sum(1 for _ in open(log, encoding="utf-8")) if os.path.exists(log) else 0
    man = {}
    for line in open(os.path.join(a.state, "manifest.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        if d["id"] in a.ids:
            man[d["id"]] = d
    miss = [i for i in a.ids if i not in man or not man[i].get("orig")]
    if miss:
        print("manifest에 원본 주소 없음:", " ".join(miss))
    todo = [i for i in a.ids if i in man and man[i].get("orig")
            and not os.path.exists(os.path.join(a.out, f"{i}.{man[i].get('ext') or 'img'}"))]
    if used + len(todo) > a.max:
        raise SystemExit(f"{name}: 점검 요청 {used}번 + 이번 {len(todo)}번 > 상한 {a.max}. 사용자에게 묻는다")
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, LOWCOST)
    import dcnet
    dcnet.UA = "Claude-User" if a.user_request else (os.environ.get("DCNET_UA") or dcnet.AUTO_UA)
    dcnet.LOG = os.path.join(a.out, "fetchlog.tsv")
    dcnet.LOCKDIR = a.out
    kind = "Claude-User" if a.user_request else "자기 이름"
    for i in todo:
        d = man[i]
        p = os.path.join(a.out, f"{i}.{d.get('ext') or 'img'}")
        ref = VIEW.format(no=d["no"]) if str(d["no"]).isdigit() else None
        code, size = dcnet.get(d["orig"], p, referer=ref)
        with open(log, "a", encoding="utf-8") as f:
            now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{now}\t{i}\t{code}\t{size}\t{kind}\n")
        print(i, code, size, p if code == 200 else "")
        if code in (403, 429):
            raise SystemExit(f"{code} — 멈춤. 다시 시도하지 않는다")
    print(f"{name}: 점검 요청 누적 {used + len(todo)}/{a.max}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
