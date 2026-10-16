"""(S41 재수집 tools/에서 가져와 고침, 세션 F1 2026-10-15)
글 본문과 사진을 동시에 받는다: 본문 작업자(gall.dcinside.com)가 글을 받는 대로 사진 작업자(dcimg)가 그 글의 사진을 받는다.

python3 -I collect.py NOS.txt POSTDIR IMGDIR MANIFEST.jsonl [--comments]

- 작업자 수: 환경 변수 COLLECT_WORKERS (기본 1 → 본문 1 · 사진 1). N이면 본문 N · 사진 N.
  요청 시작 간격은 호스트별로 dcnet.py가 지킨다 (DCNET_GAP, 기본 1.2초). 응답(요청당 약 2초)이 간격보다 길면
  작업자 여럿의 요청이 겹쳐서 빨라진다. 예: 작업자 4 · 간격 0.5초 → 호스트마다 평균 0.5초에 하나.
- 403·429가 오면(dcnet.BLOCKED) 새 글을 더 꺼내지 않고 멈춘 뒤 실패로 끝난다. 같은 인자로 다시 돌리면 이어진다.
- 글 JSON은 다 쓴 뒤 이름을 바꿔 저장한다. 목록(MANIFEST) 줄 순서는 작업자 순서라 글 번호 순이 아닐 수 있다.
"""
import json
import os
import queue
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcnet  # noqa: E402
import post_fetch  # noqa: E402
import img_fetch  # noqa: E402

DONE = object()


class Out:
    """여러 사진 작업자가 한 목록 파일에 줄 단위로 쓴다."""

    def __init__(self, f):
        self.f, self.lock = f, threading.Lock()

    def write(self, s):
        with self.lock:
            self.f.write(s)

    def flush(self):
        with self.lock:
            self.f.flush()


def post_worker(k, todo, postdir, want_c, q, err):
    try:
        while not dcnet.BLOCKED:
            try:
                no = todo.get_nowait()
            except queue.Empty:
                return
            p = os.path.join(postdir, f"{no}.json")
            if not os.path.exists(p):
                rec = post_fetch.fetch(int(no), want_c, tag=f"_{k}")
                if dcnet.BLOCKED:
                    return
                tmp = f"{p}.{k}.tmp"
                json.dump(rec, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
                os.replace(tmp, p)
            q.put(no)
    except BaseException as e:  # noqa: BLE001
        err.append(e)
    finally:
        q.put(DONE)


def img_worker(q, nposts, postdir, imgdir, out, have, cnt, err):
    try:
        while True:
            no = q.get()
            if no is DONE:
                q.put(DONE)   # 다른 사진 작업자에게도 알린다
                return
            if not dcnet.BLOCKED:
                img_fetch.fetch_post(int(no), postdir, imgdir, out, have)
            with cnt["lock"]:
                cnt["n"] += 1
                if cnt["n"] % 50 == 0:
                    print("posts", cnt["n"], flush=True)
    except BaseException as e:  # noqa: BLE001
        err.append(e)


def main():
    nosf, postdir, imgdir, man = sys.argv[1:5]
    want_c = "--comments" in sys.argv
    w = max(1, int(os.environ.get("COLLECT_WORKERS") or 1))
    os.makedirs(postdir, exist_ok=True)
    os.makedirs(imgdir, exist_ok=True)
    nos = open(nosf).read().split()
    todo = queue.Queue()
    for no in nos:
        todo.put(no)
    have = img_fetch.load_have(man)
    q, err = queue.Queue(), []
    cnt = {"n": 0, "lock": threading.Lock()}
    with open(man, "a", encoding="utf-8") as f:
        out = Out(f)
        posts = [threading.Thread(target=post_worker, args=(k, todo, postdir, want_c, q, err)) for k in range(w)]
        for t in posts:
            t.start()
        # 본문 작업자가 모두 끝나야 DONE이 사진 쪽으로 넘어간다: DONE 수를 세는 대신 본문 작업자 join 뒤 하나만 넣는다
        imgq = queue.Queue()

        def relay():
            left = w
            while left:
                x = q.get()
                if x is DONE:
                    left -= 1
                else:
                    imgq.put(x)
            imgq.put(DONE)
        r = threading.Thread(target=relay)
        r.start()
        imgs = [threading.Thread(target=img_worker, args=(imgq, len(nos), postdir, imgdir, out, have, cnt, err))
                for _ in range(w)]
        for t in imgs:
            t.start()
        for t in posts + [r] + imgs:
            t.join()
    if err:
        raise err[0]
    if dcnet.BLOCKED:
        raise SystemExit(f"blocked {dcnet.BLOCKED[:3]}")
    print("done", cnt["n"], "workers", w, flush=True)


if __name__ == "__main__":
    main()
