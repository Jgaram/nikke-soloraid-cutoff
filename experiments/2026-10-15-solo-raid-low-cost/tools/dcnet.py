"""디시 요청 공통 도구: curl 호출, 1.2초 간격, 재시도, 요청 기록.

- UA는 기본 `Claude-User` (사용자 요청을 받아 Claude가 가져오는 경우, docs/community-access.md 공통 원칙).
  사람 없이 도는 자동 실행은 환경 변수 `DCNET_UA`로 자기 이름 UA를 준다 (`AUTO_UA`, 세션 F1 2026-10-15).
- 실패한 요청도 간격 한 칸을 쓴다. curl이 오류로 끝나면(시간 제한·중간 끊김) 상태가 200이어도 실패로 보고 다시 요청한다.
- 간격은 호스트별로, 여러 프로세스가 같이 지킨다: DCNET_LOCKDIR(기본 /tmp)의 호스트별 잠금 파일에
  마지막 요청 시각을 적고 flock으로 순서를 잡는다. 이미지 호스트는 dcimg1~9를 한 호스트로 친다.
- 모든 요청을 fetchlog.tsv에 남긴다: 시각, 상태, 바이트, 시도 횟수, 주소, 응답 시간(초).
- 여러 스레드가 같이 써도 된다. 간격은 요청 시작 사이의 간격이라, 응답이 간격보다 길면 요청이 겹친다 (collect.py 작업자).
- 403·429가 한 번이라도 오면 `BLOCKED`를 켠다. collect.py 작업자는 이것을 보고 멈춘다 (세션 F1 2026-10-15).
  예외: 본문 사진 서버(dcimg*)의 403은 그 사진만 실패로 두고 `IMG403`에 모은다. 한 프로세스에서 IMG403_MAX(기본 20)번이
  쌓이면 그때 BLOCKED를 켠다. 429와 목록·본문의 403은 그대로 바로 멈춘다 (S37 2026-10-16: 한 글의 dcimg8 사진 10장이
  Referer가 없어 403이었고, 같은 때 다른 사진 서버는 200이었다. 사용자 결정).
"""
import fcntl
import os
import re
import subprocess
import time
from urllib.parse import urlparse

AUTO_UA = "nikke-community-research/0.1 (+https://github.com/jgaram/nikke-community)"
UA = os.environ.get("DCNET_UA") or "Claude-User"
GAP = float(os.environ.get("DCNET_GAP") or 1.2)  # 호스트별 요청 간격(초). 0.5초 시험은 2026-10-15 사용자 요청
LOG = os.environ.get("FETCHLOG", "fetchlog.tsv")


LOCKDIR = os.environ.get("DCNET_LOCKDIR", "/tmp")
BLOCKED = []
IMG403 = []
IMG403_MAX = int(os.environ.get("IMG403_MAX") or 20)
IMG_HOST = re.compile(r"^dcimg\d+\.dcinside\.(co\.kr|com)$")


def _wait(url):
    host = re.sub(r"^dcimg\d+\.", "dcimg.", urlparse(url).hostname or "x")
    path = os.path.join(LOCKDIR, f"dcnet_{host}.lock")
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            last = float(f.read().strip() or 0)
        except ValueError:
            last = 0.0
        d = time.time() - last
        if d < GAP:
            time.sleep(GAP - d)
        f.seek(0)
        f.truncate()
        f.write(str(time.time()))
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def get(url, out, referer=None, tries=3, timeout=40, data=None, headers=(), jar=None):
    """url을 out 파일로 받는다. (http 상태, 바이트) 반환. 실패면 상태 0."""
    code, size, n, sec = 0, 0, 0, 0.0
    for n in range(1, tries + 1):
        _wait(url)
        cmd = ["curl", "-sS", "-L", "--max-time", str(timeout), "-A", UA,
               "-o", out, "-w", "%{http_code} %{size_download} %{time_total}"]
        if referer:
            cmd += ["-e", referer]
        if jar:
            cmd += ["-c", jar, "-b", jar]
        for h in headers:
            cmd += ["-H", h]
        if data is not None:
            cmd += ["--data", data]
        cmd.append(url)
        p = subprocess.run(cmd, capture_output=True, text=True)
        try:
            c, s, tt = p.stdout.split()
            code, size, sec = int(c), int(float(s)), float(tt)
        except ValueError:
            code, size = 0, 0
        if p.returncode and code == 200:
            code = 0   # 시간 제한(28)·중간 끊김(18) 등: 200이 찍혀도 받은 파일이 잘렸다 (2026-10-15 블라블라링크 목록 40초)
        if code == 403 and IMG_HOST.match(urlparse(url).hostname or ""):
            IMG403.append(url)
            if len(IMG403) >= IMG403_MAX:
                BLOCKED.append((code, url))
        elif code in (403, 429):
            BLOCKED.append((code, url))
        if code == 200 and size > 0:
            break
        if code in (403, 404, 410):
            break
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{code}\t{size}\t{n}\t{url}\t{sec:.2f}\n")
    return code, size
