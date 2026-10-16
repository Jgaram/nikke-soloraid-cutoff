"""솔로레이드 정기 수집 상태를 레포(data/solo-raid/S{n}/)와 STATE(세션 임시 폴더) 사이에서 옮긴다 (모델 0, 요청 0).

python3 -I tools/state_sync.py export STATE DEST      실행·합치기가 끝날 때마다. DEST 예: data/solo-raid/S40
python3 -I tools/state_sync.py import DEST STATE      새 세션에서 첫 실행 전에. STATE에 manifest.jsonl이 있으면 멈춘다 (--force로 덮어씀)

컨테이너는 세션마다 지워지므로, 다음 세션이 이어 가는 데 필요한 것만 커밋한다 (사용자 결정, 2026-10-15).
- DEST/state/: season.json, manifest.jsonl(사진 목록), ocr.jsonl·ocr2.jsonl(OCR 원시 결과, 다음 실행의 parse_ocr.py가 전체를 다시 읽는다),
  route.jsonl(사진의 길, `same` 판단의 sha1), reads.jsonl(칸 판독 누적), obs.jsonl·claims.jsonl·ctx.jsonl(관측 표),
  posts_min.jsonl(받은 글의 번호·시각·제목·본문 앞 500자·첨부 이름, posts/ 대신), runs.jsonl, pending_orig.txt, orig_fetch.jsonl(STATE/orig/fetch.jsonl).
- DEST/runs/{RUN}/: DONE·FAILED·run.log·merge.json·missing.txt·route_counts.json·fetchlog.tsv·nos*.txt·*_ids.txt·
  pending_orig_used.txt·cand/cand.tsv·cand/counts.json.
- 넣지 않는 것: 사진(img/·orig/·sheets/·origsrc/), 글 원문(posts/, 목록 list*.jsonl의 본문), 판독 입력·출력 배치(text/·ctx/·reads/,
  합친 결과가 state/에 있다), parsed.jsonl(OCR에서 다시 만든다), OCR venv·모델, dcnet 잠금 파일, last_run(절대 경로).
- export는 STATE에 없어진 state 파일(예: 다 처리한 pending_orig.txt)을 DEST에서도 지운다. runs/는 덧붙이기만 한다.

불러온 STATE의 한계
- 사진 파일이 없다. 앞 실행 사진을 다시 읽어야 하는 경우(pending_orig)는 원본을 새로 받고, 원본 주소가 없거나 못 받으면
  그 사진은 띠 판독 줄을 그대로 둔다 (run_window.sh의 orig_nosrc).
- 중간에 실패한 실행은 같은 세션에서 이어 돌린다. 세션이 바뀌었으면 그 기간을 새 실행으로 다시 돌린다
  (이미 받은 글은 posts_min.jsonl로 건너뛰므로, 실패한 실행이 text_prep 전에 멈췄다면 그 글들은 다시 받는다).
"""
import json
import os
import shutil
import sys

STATE_FILES = ["season.json", "manifest.jsonl", "ocr.jsonl", "ocr2.jsonl", "route.jsonl", "reads.jsonl", "obs.jsonl",
               "claims.jsonl", "ctx.jsonl", "posts_min.jsonl", "runs.jsonl", "pending_orig.txt"]
RENAME = {"orig_fetch.jsonl": os.path.join("orig", "fetch.jsonl")}   # DEST 이름 → STATE 경로
RUN_FILES = ["DONE", "FAILED", "run.log", "merge.json", "missing.txt", "route_counts.json", "fetchlog.tsv", "nos.txt",
             "nos_dc.txt", "nos_other.txt", "strip_ids.txt", "orig_ids.txt", "pass2_ids.txt", "pending_orig_used.txt",
             os.path.join("cand", "cand.tsv"), os.path.join("cand", "counts.json")]


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + ".tmp")
    os.replace(dst + ".tmp", dst)


def state_pairs(state, dest):
    for n in STATE_FILES:
        yield os.path.join(state, n), os.path.join(dest, "state", n)
    for n, sp in RENAME.items():
        yield os.path.join(state, sp), os.path.join(dest, "state", n)


def export(state, dest):
    if not os.path.isfile(os.path.join(state, "season.json")):
        raise SystemExit(f"{state}/season.json이 없다")
    n = rm = 0
    for s, d in state_pairs(state, dest):
        if os.path.exists(s):
            copy(s, d)
            n += 1
        elif os.path.exists(d):
            os.remove(d)
            rm += 1
    runs = 0
    rdir = os.path.join(state, "runs")
    for run in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
        got = 0
        for f in RUN_FILES:
            s = os.path.join(rdir, run, f)
            if os.path.isfile(s):
                copy(s, os.path.join(dest, "runs", run, f))
                got += 1
        runs += bool(got)
    print(json.dumps({"export": dest, "state_files": n, "removed": rm, "runs": runs}, ensure_ascii=False))


def do_import(dest, state, force=False):
    if not os.path.isfile(os.path.join(dest, "state", "season.json")):
        raise SystemExit(f"{dest}/state/season.json이 없다")
    if os.path.exists(os.path.join(state, "manifest.jsonl")) and not force:
        raise SystemExit(f"{state}에 이미 상태가 있다 (덮어쓰려면 --force)")
    n = 0
    for s, d in state_pairs(state, dest):
        if os.path.exists(d):
            copy(d, s)
            n += 1
    runs = 0
    rdir = os.path.join(dest, "runs")
    for run in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
        for root, _, files in os.walk(os.path.join(rdir, run)):
            for f in files:
                p = os.path.join(root, f)
                copy(p, os.path.join(state, "runs", os.path.relpath(p, rdir)))
        runs += 1
    for d in ("posts", "img", "orig", "models"):
        os.makedirs(os.path.join(state, d), exist_ok=True)
    print(json.dumps({"import": dest, "state": state, "state_files": n, "runs": runs}, ensure_ascii=False))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--force"]
    if len(a) != 3 or a[0] not in ("export", "import"):
        raise SystemExit(__doc__)
    if a[0] == "export":
        export(a[1], a[2])
    else:
        do_import(a[1], a[2], "--force" in sys.argv)
