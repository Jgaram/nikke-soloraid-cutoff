#!/usr/bin/env bash
# 1단계 전체를 한 프로세스로 돌린다: 준비 → 사진 받기(+받는 대로 1배 OCR) → 깨진 사진만 1.5배 OCR → 파싱 → 측정.
#
#   bash tools/pipeline.sh WORK [EVALSET] [SPLIT]      (실험 폴더에서, Bash run_in_background로 하나만 띄운다)
#
# - WORK: 세션 임시 폴더 안의 작업 폴더. venv·모델·사진·OCR 결과가 모두 여기에 생긴다 (레포 밖).
# - 끝나면 WORK/DONE에 요약 한 줄을 쓰고 종료한다. 실패하면 WORK/FAILED에 단계 이름을 쓴다.
#   기다릴 때는 이 프로세스의 완료 알림이나 DONE·FAILED 파일만 본다. pgrep -f로 기다리지 않는다
#   (대기 명령줄 자신이 걸려 끝나지 않는다, 2026-10-15에 22분 헛기다림).
# - 이미 받은 사진·이미 한 OCR은 건너뛰므로 중간에 끊겨도 다시 돌리면 이어서 한다.
set -euo pipefail

WORK=${1:?WORK 폴더}
EVAL=${2:-data/evalset.jsonl}
SPLIT=${3:-holdout}
T=$(cd "$(dirname "$0")" && pwd)
VENV=$WORK/ocr
M=$WORK/models
HF=https://huggingface.co/PaddlePaddle
mkdir -p "$WORK" "$M" "$WORK/img"
rm -f "$WORK/DONE" "$WORK/FAILED"
STEP=start
trap 'echo "$STEP" > "$WORK/FAILED"' ERR
log() { echo "$(TZ=Asia/Seoul date +%H:%M:%S) $*" >> "$WORK/pipeline.log"; }

STEP=setup
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
log "setup ok"

STEP=fetch+ocr1
DCNET_LOCKDIR=$WORK python3 -I "$T/eval_fetch.py" "$EVAL" "$WORK/img" >> "$WORK/pipeline.log" 2>&1 &
FPID=$!
# 받는 동안 받은 것부터 OCR한다. 받기 프로세스는 PID로만 확인한다.
while kill -0 "$FPID" 2>/dev/null; do
  "$VENV/bin/python" "$T/ocr_run.py" "$M" "$WORK/img" "$WORK/ocr.jsonl" 2>> "$WORK/ocr.err"
  sleep 20
done
wait "$FPID"
"$VENV/bin/python" "$T/ocr_run.py" "$M" "$WORK/img" "$WORK/ocr.jsonl" 2>> "$WORK/ocr.err"
log "fetch $(wc -l < "$WORK/img/fetch.jsonl"), ocr1 $(wc -l < "$WORK/ocr.jsonl")"

STEP=ocr2
python3 -I "$T/parse_ocr.py" "$WORK/ocr.jsonl" "$WORK/parsed1.jsonl"
python3 -I -c "
import json, sys
need = {'score_broken', 'score_format', 'low_ocr'}
ids = [p['id'] for p in map(json.loads, open(sys.argv[1])) if need & set(p['why'])]
open(sys.argv[2], 'w').write('\n'.join(ids) + '\n')" "$WORK/parsed1.jsonl" "$WORK/pass2_ids.txt"
OCR_SCALE=1.5 OCR_IDS=$WORK/pass2_ids.txt "$VENV/bin/python" "$T/ocr_run.py" "$M" "$WORK/img" "$WORK/ocr2.jsonl" 2>> "$WORK/ocr.err"
touch "$WORK/ocr2.jsonl"
log "ocr2 $(wc -l < "$WORK/ocr2.jsonl")"

STEP=parse+measure
python3 -I "$T/parse_ocr.py" "$WORK/ocr.jsonl" "$WORK/parsed.jsonl" "$WORK/ocr2.jsonl"
python3 -I "$T/measure.py" "$WORK/parsed.jsonl" "$SPLIT" "$WORK/metrics.json" "$WORK/miss.jsonl" > "$WORK/measure.out"
python3 -I -c "
import json, sys
d = json.load(open(sys.argv[1]))
print({k: d[k] for k in ('split', 'images', 'coverage', 'pct_ok', 'score_ok_0.1eok', 'rank_ok', 'neg_fp_rate')})" "$WORK/metrics.json" > "$WORK/DONE"
log "done"
cat "$WORK/DONE"
