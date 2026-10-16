"""공개 레포 jgaram/nikke-soloraid-cutoff에 올릴 파일을 한 폴더에 모은다 (모델 0, 요청 0, 2026-10-16).

python3 -I tools/publish/build_public.py --out DIR [--site DIR2]

- DIR: 공개 레포 main 브랜치의 내용 전체 (코드·문서·README·LICENSE). .git은 건드리지 않는다.
- DIR2: 공개 웹(gh-pages 브랜치) — tools/publish/site.py 결과. 주지 않으면 만들지 않는다.

이 레포에서 무엇을 내보내는지는 아래 ALLOW가 정한다 (허용 목록 방식 — 목록에 없는 파일은 나가지 않는다).
내보내기 전에 손보는 것
- 공개하지 않는 파일로 가는 마크다운 링크는 링크를 풀고 글자만 남긴다 (AGENTS.md·experiments 기록·data 등).
- 특정 글을 가리키는 번호(디시 글 번호·주소, 첨부 파일명의 촬영 시각)는 가린다.
그 뒤 검사에 걸리면(사진·jsonl·큰 파일·글 번호가 남음) 아무것도 쓰지 않고 멈춘다. 규칙은 docs/publish.md.
"""
import argparse
import fnmatch
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

ALLOW = [
    "tools/solo_raid/*.py",
    "tools/solo_raid/*.sh",
    "tools/solo_raid/*.html",
    "tools/publish/*.py",
    "tools/publish/*.html",
    "experiments/2026-10-15-solo-raid-low-cost/README.md",
    "experiments/2026-10-15-solo-raid-low-cost/tools/*",
    "docs/solo-raid-*.md",
    "docs/community-access.md",
    "docs/publish.md",
    ".claude/agents/raid-vision.md",
    ".claude/workflows/raid-tile-read.js",
    ".claude/workflows/raid-window.js",
    "data/solo-raid/split.json",
    "data/solo-raid/analysis/criteria.json",
]
GENERATED = {"README.md": "public_README.md", "LICENSE": "LICENSE-CC0.txt"}

# 특정 글을 가리키는 것 → 가린다
MASK = [
    (re.compile(r"(board/gov/)\d{5,}"), r"\1…"),
    (re.compile(r"(\bno=)\d{5,}"), r"\1…"),
    (re.compile(r"(글 ?)\d{6,}"), r"\1…"),
    (re.compile(r"Screenshot_\d{8}_\d{6}"), "Screenshot_YYYYMMDD_HHMMSS"),
]
# 가린 뒤에도 남으면 멈춘다
BAD_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".webm", ".jsonl", ".csv", ".tsv", ".pyc"}
BAD_TEXT = [re.compile(p) for p in (r"board/gov/\d{5,}", r"\bno=\d{5,}", r"글 ?\d{6,}", r"Screenshot_\d{8}_\d{6}",
                                    r"viewimage\.php\?id=\w+&no=\w{20,}", r"download\.php\?no=\w{20,}")]
MAX_BYTES = 1_000_000
MD_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def select(files):
    return sorted(f for f in files if any(fnmatch.fnmatchcase(f, p) for p in ALLOW))


def fix_links(text, path, published):
    """공개하지 않는 파일로 가는 상대 링크는 글자만 남긴다."""
    base = posixpath.dirname(path)

    def sub(m):
        label, target = m.group(1), m.group(2)
        if re.match(r"^[a-z]+:|^#", target):
            return m.group(0)
        tgt = posixpath.normpath(posixpath.join(base, target.split("#")[0])) if target.split("#")[0] else path
        if tgt in published or any(p.startswith(tgt.rstrip("/") + "/") for p in published):
            return m.group(0)
        return label
    return MD_LINK.sub(sub, text)


def build(out):
    files = select(tracked())
    published = set(files) | set(GENERATED)
    staged = {}
    for f in files:
        with open(os.path.join(ROOT, f), "rb") as fh:
            raw = fh.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            staged[f] = raw
            continue
        if f.endswith(".md"):
            text = fix_links(text, f, published)
        for pat, rep in MASK:
            text = pat.sub(rep, text)
        staged[f] = text.encode("utf-8")
    for dst, src in GENERATED.items():
        with open(os.path.join(HERE, src), "rb") as fh:
            staged[dst] = fh.read()

    problems = []
    for f, b in staged.items():
        if os.path.splitext(f)[1].lower() in BAD_EXT:
            problems.append(f"{f}: 내보내지 않는 형식")
        if len(b) > MAX_BYTES:
            problems.append(f"{f}: {len(b):,}바이트 (1MB 넘음)")
        t = b.decode("utf-8", "replace")
        for pat in BAD_TEXT:
            m = pat.search(t)
            if m:
                problems.append(f"{f}: 글을 가리키는 값이 남음 «{m.group(0)[:40]}»")
    if problems:
        print("멈춤 — 공개 검사에 걸렸다:\n  " + "\n  ".join(problems), file=sys.stderr)
        raise SystemExit(1)

    # 공개 레포 내용을 통째로 바꾼다 (.git만 남김)
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        if name == ".git":
            continue
        p = os.path.join(out, name)
        shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
    for f, b in staged.items():
        p = os.path.join(out, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(b)
        if os.access(os.path.join(ROOT, f), os.X_OK):
            os.chmod(p, 0o755)
    return staged


def main():
    ap = argparse.ArgumentParser(description="공개 레포(nikke-soloraid-cutoff) 내용을 모은다")
    ap.add_argument("--out", required=True, help="공개 레포 main 작업 폴더")
    ap.add_argument("--site", help="공개 웹(gh-pages) 작업 폴더")
    a = ap.parse_args()
    staged = build(a.out)
    print(f"{a.out}: 파일 {len(staged)}개, {sum(len(b) for b in staged.values()) / 1024:.0f} KB")
    if a.site:
        # 지금 공개된 data.json을 site.py에 넘긴다 — 내용이 같으면 기준 시각을 그대로 둬서 gh-pages에 커밋이 생기지 않는다
        prev = None
        old = os.path.join(a.site, "data.json")
        if os.path.isfile(old):
            fd, prev = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            shutil.copyfile(old, prev)
        for name in os.listdir(a.site) if os.path.isdir(a.site) else []:
            if name != ".git":
                p = os.path.join(a.site, name)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        try:
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "site.py"), "--out", a.site] + (["--prev", prev] if prev else []), check=True)
        finally:
            if prev:
                os.remove(prev)
        for name in os.listdir(a.site):
            p = os.path.join(a.site, name)
            if os.path.isfile(p):
                with open(p, encoding="utf-8", errors="replace") as fh:
                    t = fh.read()
                hit = [m.group(0) for pat in BAD_TEXT for m in [pat.search(t)] if m]
                if hit or re.search(r"https?://(gall\.dcinside|m\.dcinside|game\.naver|www\.blablalink|api\.blablalink)", t):
                    raise SystemExit(f"멈춤 — 웹 파일 {name}에 원글을 가리키는 값이 있다: {hit[:2]}")


if __name__ == "__main__":
    main()
