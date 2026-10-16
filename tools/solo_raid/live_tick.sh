#!/usr/bin/env bash
# 솔로레이드 자동 수집(6시간 간격) 한 번의 코드 단계 (판독 모델 0). 2026-10-16 사용자 결정, docs/solo-raid-live.md 「자동 실행 한 번」.
# 예약 작업이 띄운 세션이 차례로 부른다. 판독(raid-window Workflow)만 세션이 하고 나머지는 여기서 한다.
#
#   bash tools/solo_raid/live_tick.sh plan              할 일 판정 → WORK/plan.json (live_tick.py plan)
#   bash tools/solo_raid/live_tick.sh collect           state 불러오기 → (칸 경계까지 기다림) → run_window.sh → WORK/workflow.json
#   bash tools/solo_raid/live_tick.sh collect --orig    (final만) 원본 대기 실행: [e_end − 1시간, e_end)를 한 번 더
#   bash tools/solo_raid/live_tick.sh merge             merge_obs.py. 종료 코드 3 = 빠진 배치(표준 출력 끝 줄이 Workflow skip 목록),
#                                                       5 = (final만) 원본 대기 실행이 필요함
#   bash tools/solo_raid/live_tick.sh finish            내보내기·커밋·push → run: 예측 T → 커밋·push / final: 후처리 → 커밋·push
#   bash tools/solo_raid/live_tick.sh fail STEP [NOTE]  받은 데까지 내보내고 실패를 기록·커밋·push (예측은 내지 않는다)
#
# 종료 코드: collect 4 = 403·429 (그 시즌 받기를 멈춘다, docs/solo-raid-backfill.md 「멈출 때」), 1 = 두 번 실패.
# 환경: LIVE_WORK(기본 ${TMPDIR:-/tmp}/solo_raid_live), 리허설용 LIVE_NOW·LIVE_DATA(live_tick.py)와 LIVE_DRY=1(커밋·push 안 함).
# 커밋·push는 이 스크립트가 한다 (main에 바로, AGENTS.md). push가 4번 실패하면 종료 코드 6.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
L=$REPO/experiments/2026-10-15-solo-raid-low-cost
WORK=${LIVE_WORK:-${TMPDIR:-/tmp}/solo_raid_live}
STATE=$WORK/state
PLAN=$WORK/plan.json
TICK="python3 -I $REPO/tools/solo_raid/live_tick.py"
export DCNET_UA="nikke-community-research/0.1 (+https://github.com/jgaram/nikke-community)" COLLECT_WORKERS=4 DCNET_GAP=0.5
mkdir -p "$WORK"
cmd=${1:?plan|collect|merge|finish|fail}
shift

J() { python3 -I -c 'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' "$PLAN" "$1"; }
need_plan() { [ -s "$PLAN" ] || { echo "$PLAN 없음 — plan부터" >&2; exit 2; }; }
kst() { TZ=Asia/Seoul date '+%F %T'; }

commit() {   # 메시지 — DEST와 live.json만 올린다
  if [ "${LIVE_DRY:-0}" = 1 ]; then echo "(리허설) 커밋 안 함: $1"; return 0; fi
  git -C "$REPO" add -A -- "$DEST_REL" data/solo-raid/live.json
  if git -C "$REPO" diff --cached --quiet; then echo "바뀐 것 없음: $1"; return 0; fi
  git -C "$REPO" commit -q -m "$1" -m "솔로레이드 자동 수집 (docs/solo-raid-live.md) · 세션 ${CLAUDE_CODE_SESSION_ID:-?}"
}
push() {
  if [ "${LIVE_DRY:-0}" = 1 ]; then return 0; fi
  local i
  for i in 1 2 3 4; do
    if git -C "$REPO" fetch -q origin main && git -C "$REPO" merge -q --no-edit origin/main && git -C "$REPO" push -q origin HEAD:main; then
      return 0
    fi
    sleep $((2 ** i))
  done
  echo "push 4번 실패" >&2
  exit 6
}
export_state() {
  [ -s "$STATE/manifest.jsonl" ] || [ -s "$STATE/runs.jsonl" ] || return 0
  python3 -I "$L/tools/state_sync.py" export "$STATE" "$DEST"
}

if [ "$cmd" = plan ]; then
  $TICK plan > "$PLAN.tmp"
  mv "$PLAN.tmp" "$PLAN"
  cat "$PLAN"
  exit 0
fi

need_plan
ACT=$(J action)
DEST=$(J dest)
NAME=$(J season)
T=$(J T)
DEST_REL=$(python3 -I -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$DEST" "$REPO")
case $ACT in run|final) ;; *) echo "할 일 없음 (action=$ACT): $(J reason)" >&2; exit 2 ;; esac
LABEL=$([ "$ACT" = run ] && echo "T$T" || echo "종료 후")

case $cmd in
collect)
  START=$(J start); END=$(J end_w)
  if [ "${1:-}" = --orig ]; then
    [ "$ACT" = final ] || { echo "--orig는 final에서만" >&2; exit 2; }
    END=$(J e_end)
    START=$(python3 -I -c 'import datetime as d,sys; print((d.datetime.strptime(sys.argv[1],"%Y-%m-%d %H:%M:%S")-d.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"))' "$END")
    touch "$WORK/orig_pass"
  fi
  # 컨테이너가 새것이면 커밋된 state를 불러온다 (첫 실행은 season.json뿐 — 있는 파일만 옮긴다)
  if [ ! -s "$STATE/season.json" ]; then
    mkdir -p "$STATE"
    python3 -I "$L/tools/state_sync.py" import "$DEST" "$STATE"
  fi
  W=$(J wait_sec)
  if [ "${1:-}" != --orig ] && [ -n "$W" ] && [ "$W" -gt 0 ] && [ "$W" -le 900 ]; then
    echo "칸 경계까지 ${W}초 기다림"
    sleep "$W"
  fi
  if [ "$(J collect)" != True ] && [ "${1:-}" != --orig ]; then
    echo '{"needed": false, "why": "받을 구간 없음"}' | tee "$WORK/workflow.json"
    exit 0
  fi
  echo "$(kst) 받기 $NAME $LABEL [$START, $END)"
  ok=0
  for attempt in 1 2; do
    if (cd "$L" && bash tools/run_window.sh "$STATE" "$START" "$END"); then ok=1; break; fi
    RUN=$(cat "$STATE/last_run" 2>/dev/null || echo "$STATE/runs/$(echo "${START}_${END}" | tr -d ' :-' | tr '_' '-')")
    if ! $TICK block "$RUN"; then
      echo "403·429 — 이 시즌 받기를 멈춘다" >&2
      exit 4
    fi
    echo "run_window 실패 ($(cat "$RUN/FAILED" 2>/dev/null || echo ?)) — 같은 인자로 한 번 더" >&2
  done
  [ $ok = 1 ] || exit 1
  RUN=$(cat "$STATE/last_run")
  $TICK block "$RUN" > /dev/null || { echo "403·429 기록 있음 — 이 시즌 받기를 멈춘다" >&2; exit 4; }
  python3 -I - "$RUN" "$L/tools/tile_prompt.md" "$WORK/workflow.json" <<'PY'
import json, sys
run, prompt, out = sys.argv[1:4]
d = json.loads(open(f"{run}/DONE").read().strip().splitlines()[-1])
args = {"run": run, "sheets": d.get("sheets", 0), "text": d.get("text_batches", 0), "ctx": d.get("ctx_batches", 0),
        "tile_prompt": prompt, "parts": 4}
w = {"needed": bool(args["sheets"] or args["text"] or args["ctx"]), "scriptPath": ".claude/workflows/raid-window.js",
     "args": args, "done": {k: d.get(k) for k in ("list", "new_posts", "sheets", "requests", "req_fail", "sec")}}
json.dump(w, open(out, "w"), ensure_ascii=False)
print(json.dumps(w, ensure_ascii=False))
PY
  ;;

merge)
  RUN=$(cat "$STATE/last_run")
  (cd "$L" && python3 -I tools/merge_obs.py "$STATE" "$RUN")
  if [ -s "$RUN/missing.txt" ]; then
    echo "빠진 배치: $(tr '\n' ' ' < "$RUN/missing.txt")" >&2
    $TICK skip "$RUN"
    exit 3
  fi
  if [ "$ACT" = final ] && [ ! -e "$WORK/orig_pass" ] && [ -s "$STATE/pending_orig.txt" ]; then
    echo "원본 대기 $(wc -l < "$STATE/pending_orig.txt")장 — collect --orig 한 번 더" >&2
    exit 5
  fi
  ;;

finish)
  RUN=$(cat "$STATE/last_run" 2>/dev/null || true)
  export_state
  commit "$NAME 받기 $LABEL"
  push
  if [ "$ACT" = run ]; then
    OUT=$DEST/predict/T$T
    # 다른 실행이 같은 칸을 먼저 올렸으면 겹쳐 쓰지 않는다
    if [ "${LIVE_DRY:-0}" != 1 ] && git -C "$REPO" cat-file -e "origin/main:$DEST_REL/predict/T$T/predict_T$T.json" 2>/dev/null; then
      echo "T$T 예측이 이미 main에 있다 — 예측을 건너뛴다"
      exit 0
    fi
    bash "$REPO/tools/solo_raid/predict_run.sh" "$DEST/state" "$T" --out "$OUT" > "$WORK/predict_stdout.txt" 2>&1 \
      || { tail -20 "$WORK/predict_stdout.txt" >&2; $TICK record "$PLAN" "예측 실패" "$RUN" "predict_run.sh" > /dev/null; commit "$NAME 예측 T$T 실패"; push; exit 1; }
    $TICK record "$PLAN" ok "$RUN" > /dev/null
    commit "$NAME 예측 T$T"
    push
    echo "=== 예측 T$T 요약 ($OUT/predict_T$T.txt)"
    grep -E '^(\[요약\]|!! )' "$OUT/predict_T$T.txt" || head -5 "$OUT/predict_T$T.txt"
  else
    note=""
    bash "$REPO/tools/solo_raid/post_all.sh" "$DEST/state" > "$WORK/post_all.txt" 2>&1 || { note="post_all.sh 실패 — $(tail -3 "$WORK/post_all.txt" | tr '\n' ' ')"; echo "$note" >&2; }
    $TICK record "$PLAN" ok "$RUN" "$note" > /dev/null
    printf '{"done": "%s", "by": "%s"}\n' "$(kst)" "${CLAUDE_CODE_SESSION_ID:-?}" > "$DEST/live/final_done.json"
    if [ "${LIVE_DRY:-0}" != 1 ]; then
      python3 -I - "$REPO/data/solo-raid/live.json" "$NAME" <<'PY'
import json, sys
f, name = sys.argv[1:3]
d = json.load(open(f, encoding="utf-8"))
if d.get("season") == name:
    d["season"] = None
    d["last"] = name
    json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(f, "a").write("\n")
PY
    fi
    commit "$NAME 종료 후 수집 끝 (자동 수집 끔)"
    push
    echo "=== $NAME 종료 후 수집 끝. 남은 일: 데이터 점검·수집 끝 표시·채점 (docs/solo-raid-live.md 「3. 끝난 뒤」 3~7)"
  fi
  ;;

fail)
  STEP=${1:-?}; NOTE=${2:-}
  RUN=$(cat "$STATE/last_run" 2>/dev/null || true)
  export_state || true
  $TICK record "$PLAN" "실패 ($STEP)" "$RUN" "$NOTE" > /dev/null
  commit "$NAME 자동 $LABEL 실패 ($STEP)"
  push
  ;;

*) echo "모르는 명령: $cmd" >&2; exit 2 ;;
esac
