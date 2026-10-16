#!/usr/bin/env bash
# 네이버·블라블라링크 기간 하나의 받기·OCR 단계 (모델 0). 2026-10-15, 디시와 묶기 전 시간 재기용.
# 묶은 뒤(2026-10-15 밤)에는 쓰지 않는다 → SITES="naver blabla" bash tools/run_window.sh … (시트·글 숫자·시즌 시작까지 한다).
# S41 시간 재기 기록으로 남겨 둔다.
#
#   DCNET_UA=... bash tools/run_other.sh STATE 'YYYY-MM-DD HH:MM:SS' 'YYYY-MM-DD HH:MM:SS'
#
# run_window.sh의 list → cand → collect+ocr1 → ocr2+parse → route → orig 까지를 두 사이트로 돈다.
# 시트(tile_sheets)·글 숫자(text_prep)는 아직 디시 번호(숫자)만 받으므로 묶을 때 고친다.
# - STATE/season.json은 run_window.sh와 같다. 디시와 같은 STATE를 써도 글 번호가 겹치지 않는다 (n…·b…).
# - 끝나면 RUN/DONE에 단계별 걸린 시간(초)·요청 수 요약 한 줄.
set -euo pipefail

STATE=$(mkdir -p "${1:?STATE}" && cd "$1" && pwd)
START=${2:?START}
END=${3:?END}
T=$(cd "$(dirname "$0")" && pwd)
[ -s "$STATE/season.json" ] || { echo "STATE/season.json이 없다" >&2; exit 2; }
RUN=$STATE/runs/other-$(echo "${START}_${END}" | tr -d ' :-' | tr '_' '-')
mkdir -p "$RUN" "$STATE/posts" "$STATE/img" "$STATE/orig" "$STATE/models"
rm -f "$RUN/DONE" "$RUN/FAILED"
STEP=start
trap 'echo "$STEP" > "$RUN/FAILED"' ERR
log() { echo "$(TZ=Asia/Seoul date +%H:%M:%S) $*" >> "$RUN/run.log"; }
export FETCHLOG=$RUN/fetchlog.tsv DCNET_LOCKDIR=$STATE
SEASON() { python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$STATE/season.json" "$1"; }
declare -A SEC
lap() { local now; now=$(date +%s); SEC[$1]=$(( now - L0 )); L0=$now; }
T0=$(date +%s); L0=$T0
log "run $START ~ $END UA=${DCNET_UA:-Claude-User}"

STEP=setup
VENV=$STATE/ocr
M=$STATE/models
HF=https://huggingface.co/PaddlePaddle
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q rapidocr-onnxruntime==1.2.3 pyyaml
fi
[ -s "$M/det_v5.onnx" ] || curl -sSLf -m 300 -o "$M/det_v5.onnx" "$HF/PP-OCRv5_mobile_det_onnx/resolve/main/inference.onnx"
[ -s "$M/ko_v5_inference.onnx" ] || curl -sSLf -m 300 -o "$M/ko_v5_inference.onnx" "$HF/korean_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.onnx"
[ -s "$M/ko_v5_inference.yml" ] || curl -sSLf -m 300 -o "$M/ko_v5_inference.yml" "$HF/korean_PP-OCRv5_mobile_rec_onnx/resolve/main/inference.yml"
[ -s "$M/ko_v5_keys.txt" ] || "$VENV/bin/python" -c "
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))['PostProcess']['character_dict']
open(sys.argv[2], 'w').write('\n'.join(d) + '\n')" "$M/ko_v5_inference.yml" "$M/ko_v5_keys.txt"
lap setup

STEP=list
if [ -s "$RUN/nos.txt" ]; then
  log "list·cand 건너뜀 (같은 RUN 다시 돌림)"
else
  python3 -I "$T/other_sites.py" list naver "$START" "$END" "$RUN/list_naver.jsonl" >> "$RUN/run.log" 2>&1 &
  P1=$!
  python3 -I "$T/other_sites.py" list blabla "$START" "$END" "$RUN/list_blabla.jsonl" >> "$RUN/run.log" 2>&1 &
  P2=$!
  wait $P1
  wait $P2
  cat "$RUN/list_naver.jsonl" "$RUN/list_blabla.jsonl" > "$RUN/list.jsonl"
  lap list
  STEP=cand
  python3 -I "$T/select_cand.py" "$RUN/list.jsonl" "$RUN/cand" "$(SEASON e_start)" "$(SEASON e_end)" >> "$RUN/run.log"
  python3 -I -c "
import os, sys
state, run = sys.argv[1:3]
nos = [l.split('\t')[0] for l in open(os.path.join(run, 'cand', 'cand.tsv')).read().split('\n')[1:] if l]
new = [n for n in nos if not os.path.exists(os.path.join(state, 'posts', n + '.json'))]
open(os.path.join(run, 'nos.txt'), 'w').write(''.join(n + '\n' for n in new))
print('cand', len(nos), 'new', len(new))" "$STATE" "$RUN" >> "$RUN/run.log"
  python3 -I "$T/other_sites.py" posts "$RUN/list.jsonl" "$RUN/nos.txt" "$STATE/posts" >> "$RUN/run.log"
  log "list $(wc -l < "$RUN/list.jsonl") $(tail -2 "$RUN/run.log" | head -1)"
  lap cand
fi

STEP=collect+ocr1
( cd "$RUN" && python3 -I "$T/other_sites.py" collect "$RUN/nos.txt" "$STATE/posts" "$STATE/img" "$STATE/manifest.jsonl" ) >> "$RUN/run.log" 2>&1 &
CPID=$!
touch "$STATE/ocr.jsonl" "$STATE/ocr2.jsonl"
while kill -0 "$CPID" 2>/dev/null; do
  "$VENV/bin/python" "$T/ocr_run.py" "$M" "$STATE/img" "$STATE/ocr.jsonl" 2>> "$RUN/ocr.err"
  sleep 20
done
wait "$CPID"
lap collect
"$VENV/bin/python" "$T/ocr_run.py" "$M" "$STATE/img" "$STATE/ocr.jsonl" 2>> "$RUN/ocr.err"
log "collect done, ocr1 $(wc -l < "$STATE/ocr.jsonl")"
lap ocr1_tail

STEP=ocr2+parse
python3 -I "$T/parse_ocr.py" "$STATE/ocr.jsonl" "$RUN/parsed1.jsonl"
python3 -I -c "
import json, sys
need = {'score_broken', 'score_format', 'low_ocr'}
ids = [p['id'] for p in map(json.loads, open(sys.argv[1])) if need & set(p['why'])]
open(sys.argv[2], 'w').write('\n'.join(ids) + '\n')" "$RUN/parsed1.jsonl" "$RUN/pass2_ids.txt"
OCR_SCALE=1.5 OCR_IDS=$RUN/pass2_ids.txt "$VENV/bin/python" "$T/ocr_run.py" "$M" "$STATE/img" "$STATE/ocr2.jsonl" 2>> "$RUN/ocr.err"
python3 -I "$T/parse_ocr.py" "$STATE/ocr.jsonl" "$STATE/parsed.jsonl" "$STATE/ocr2.jsonl"
rm -f "$RUN/parsed1.jsonl"
lap ocr2

STEP=route
python3 -I "$T/route.py" "$STATE" "$RUN" >> "$RUN/run.log"

STEP=orig
python3 -I "$T/other_sites.py" orig "$RUN/orig_todo.jsonl" "$RUN/orig_ids.txt" "$STATE/orig" >> "$RUN/run.log" 2>&1
lap orig

STEP=done
python3 -I -c "
import json, sys, os, collections
run, sec, laps = sys.argv[1], sys.argv[2], sys.argv[3]
req = [l.rstrip('\n').split('\t') for l in open(os.path.join(run, 'fetchlog.tsv'))]
host = collections.Counter(r[4].split('/')[2] for r in req)
man = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.dirname(run)), 'manifest.jsonl'))]
nos = open(os.path.join(run, 'nos.txt')).read().split()
d = {'run': run, 'list': sum(1 for _ in open(os.path.join(run, 'list.jsonl'))),
     'cand': json.load(open(os.path.join(run, 'cand', 'counts.json'))),
     'new_posts': {'naver': sum(n[0] == 'n' for n in nos), 'blabla': sum(n[0] == 'b' for n in nos)},
     'imgs': sum(1 for m in man if str(m['no']) in set(nos)),
     'route': json.load(open(os.path.join(run, 'route_counts.json'))),
     'requests': len(req), 'req_fail': sum(1 for r in req if r[1] != '200'), 'hosts': dict(host),
     'sec': int(sec), 'laps': dict(x.split('=') for x in laps.split())}
print(json.dumps(d, ensure_ascii=False))" "$RUN" "$(( $(date +%s) - T0 ))" "$(for k in "${!SEC[@]}"; do printf '%s=%s ' "$k" "${SEC[$k]}"; done)" > "$RUN/DONE"
log "done"
cat "$RUN/DONE"
