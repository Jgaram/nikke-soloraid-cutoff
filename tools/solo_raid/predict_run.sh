#!/usr/bin/env bash
# 진행 중 시즌의 컷 예측을 한 명령으로: post_all.sh(후처리 1단계) → (--unknown mix면) server_mix.py --gate → predict.py
# (모델 판독 0, 요청 0, 몇 초~1분). 2026-10-16.
#
# tools/solo_raid/predict_run.sh STATE T [--out DIR] [--prior PRIOR.json] [--unknown site|none|mix] [--borrow-state S]
#                                [--draws N] [--skip-post] [--allow-cstart-fallback]
#                                [--posts F --snap DIR --parsed F --alt F --runs DIR] [-- predict.py 옵션...]
#
#   STATE  수집 도구가 만든 state 폴더 (예: data/solo-raid/S{n}/state). season.json의 c_start·end와 obs.jsonl 등만 쓴다.
#          **season.json에 c_start(챌린지 시작, "YYYY-MM-DD 05:00:00")가 없으면 멈춘다.**
#          --allow-cstart-fallback이면 start 날짜 + 2일 05:00으로 대신하고 run.log·결과 머리에 크게 경고한다 (권하지 않음).
#   T      챌린지 시작 후 시간(예: 24) 또는 KST 시각('YYYY-MM-DD HH:MM'). **지금까지 모은 자료의 끝 시각을 준다.**
#          (T보다 뒤에 게시된 글이 state에 있으면 서버 라벨·사람 잇기에 그 글이 섞인다. 지난 시각을 다시 재려면 backtest.py의 tr처럼
#           T까지만 남긴 사본을 만든다.)
#   --out  결과 폴더 (기본 ${TMPDIR:-/tmp}/solo_raid_predict/{시즌}_T{T}). predict_T{T}.txt·.json과 run.log·stdout.txt를 쓴다.
#          결과 JSON의 run에 도구 HEAD(git rev-parse)·수정 여부·prior sha1·입력 state 요약(줄 수·미상 줄·라벨 사람 수)이 들어간다.
#   --prior    기본 data/solo-raid/S41/state/prior.json
#   --unknown  서버 미상 줄 처리, 기본 site (docs/solo-raid-cutoff.md 「예측 도구」). mix면 server_mix.py --gate를 먼저 돌린다.
#   --borrow-state  T까지 서버 라벨 사람이 10명 미만일 때 사이트 비율을 빌릴 끝난 시즌 (기본 data/solo-raid/S41/state)
#   --skip-post  post_all.sh를 건너뛴다 (이미 돌렸을 때)
#   post_all.sh에 넘기는 시즌 전용 입력은 --posts·--snap·--parsed·--alt·--runs (S41처럼 state 밖 사본이 있을 때만)
#
# post_all.sh는 state 안의 후처리 파일(server*·post_*·core*·truth*)을 다시 쓴다. 원자료(obs·claims 등)는 읽기만 한다.
# 진행 중 시즌에는 종료 후 자료·truth_table.json이 없다. truth.py는 빈 정답을 쓰고 넘어가며, predict.py는 truth* 파일을 열지 않는다.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,25p' "$0"; exit 2; }
ST=$1; T=$2; shift 2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PRIOR=$ROOT/data/solo-raid/S41/state/prior.json
BORROW=$ROOT/data/solo-raid/S41/state
UNK=site; OUT=""; DRAWS=2000; SKIP=0; PA=(); PX=(); FB=()
while [ $# -gt 0 ]; do
  case $1 in
    --out) OUT=$2; shift 2 ;;
    --prior) PRIOR=$2; shift 2 ;;
    --unknown) UNK=$2; shift 2 ;;
    --borrow-state) BORROW=$2; shift 2 ;;
    --draws) DRAWS=$2; shift 2 ;;
    --skip-post) SKIP=1; shift ;;
    --allow-cstart-fallback) FB=(--allow-cstart-fallback); shift ;;
    --posts|--snap|--parsed|--alt|--runs) PA+=("$1" "$2"); shift 2 ;;
    --) shift; PX=("$@"); break ;;
    *) echo "모르는 옵션: $1" >&2; exit 2 ;;
  esac
done
[ -f "$ST/season.json" ] || { echo "$ST/season.json 없음" >&2; exit 1; }
[ -f "$ST/obs.jsonl" ] || { echo "$ST/obs.jsonl 없음" >&2; exit 1; }
[ -f "$PRIOR" ] || { echo "prior 없음: $PRIOR" >&2; exit 1; }
NAME=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("name","season"))' "$ST/season.json")
# c_start와 해석된 T(KST)를 먼저 확인한다 (재검토 r2 3: c_start가 없으면 start에 기대 조용히 망가졌다)
CS=$(python3 -I - "$ST/season.json" "$T" "${#FB[@]}" <<'PY'
import datetime as dt, json, sys
s = json.load(open(sys.argv[1], encoding="utf-8")); at = sys.argv[2]; fb = sys.argv[3] != "0"
F = "%Y-%m-%d %H:%M:%S"
if s.get("c_start"):
    c0, note = dt.datetime.strptime(s["c_start"], F), ""
elif fb:
    c0 = (dt.datetime.strptime(s["start"], F) + dt.timedelta(days=2)).replace(hour=5, minute=0, second=0)
    note = " !! c_start 대체값 (season.json에 없음, --allow-cstart-fallback)"
else:
    sys.exit(3)
try:
    T = float(at)
except ValueError:
    T = None
    for f in (F, "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            T = (dt.datetime.strptime(at, f) - c0).total_seconds() / 3600
            break
        except ValueError:
            pass
    if T is None:
        sys.exit(4)
print(f"c_start {c0.strftime(F)}{note} · T = {T:.2f}시간 = KST {(c0 + dt.timedelta(hours=T)).strftime('%Y-%m-%d %H:%M')}")
PY
) || { rc=$?; case $rc in
         3) echo "멈춤: $ST/season.json에 c_start가 없다. 챌린지 시작을 넣는다 (\"c_start\": \"YYYY-MM-DD 05:00:00\")." >&2
            echo "      start + 2일 05:00으로 대신하려면 --allow-cstart-fallback (권하지 않음)" >&2 ;;
         4) echo "멈춤: T를 읽지 못했다: $T (시간 수 또는 KST 'YYYY-MM-DD HH:MM')" >&2 ;;
         *) echo "멈춤: season.json을 읽지 못했다" >&2 ;;
       esac; exit 1; }
TAG=$(echo "$T" | tr ' :' '_-')
OUT=${OUT:-${TMPDIR:-/tmp}/solo_raid_predict/${NAME}_T${TAG}}
mkdir -p "$OUT"
LOG=$OUT/run.log
: > "$LOG"
echo "솔로레이드 예측 실행 — $NAME · T=$T · 미상 줄 $UNK · 결과 $OUT" | tee -a "$LOG"
echo "$CS" | tee -a "$LOG"
[ ${#FB[@]} -eq 0 ] || echo "!! 경고: c_start 없이 start + 2일 05:00으로 대신했다 — start가 챌린지 시작이면 48시간 어긋나 결과가 망가진다" | tee -a "$LOG" >&2
if [ $SKIP -eq 0 ]; then
  echo "[1/3] post_all.sh" | tee -a "$LOG"
  "$ROOT/tools/solo_raid/post_all.sh" "$ST" "${PA[@]}" >> "$LOG" 2>&1 || { echo "post_all.sh 실패 — $LOG" >&2; tail -20 "$LOG" >&2; exit 1; }
else
  echo "[1/3] post_all.sh 건너뜀" | tee -a "$LOG"
fi
if [ "$UNK" = mix ]; then
  echo "[2/3] server_mix.py --gate" | tee -a "$LOG"
  python3 -I "$ROOT/tools/solo_raid/server_mix.py" "$ST" --gate >> "$LOG" 2>&1 || { echo "server_mix.py 실패 — $LOG" >&2; exit 1; }
else
  echo "[2/3] server_mix 안 씀 (--unknown $UNK)" | tee -a "$LOG"
fi
echo "[3/3] predict.py" | tee -a "$LOG"
python3 -I "$ROOT/tools/solo_raid/predict.py" "$ST" --prior "$PRIOR" --at "$T" --unknown "$UNK" --borrow-state "$BORROW" \
  --draws "$DRAWS" --out "$OUT" "${FB[@]}" "${PX[@]}" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt" \
  || { cat "$OUT/stderr.txt" >&2; cat "$OUT/stderr.txt" >> "$LOG"; echo "predict.py 실패 — $LOG" >&2; exit 1; }
cat "$OUT/stdout.txt"
# run.log에도 맨 위 요약·경고와 stderr를 남긴다 (stderr에만 나오던 경고가 출력 파일에 없던 것, 재검토 r2 3)
grep -E '^(\[요약\]|!! )' "$OUT/stdout.txt" >> "$LOG" || true
[ -s "$OUT/stderr.txt" ] && { echo "[predict.py stderr]" >> "$LOG"; cat "$OUT/stderr.txt" >> "$LOG"; }
echo "→ $OUT" | tee -a "$LOG"
