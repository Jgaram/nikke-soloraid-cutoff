#!/usr/bin/env bash
# 4단계: 기간 하나의 코드 단계를 한 프로세스로 돌린다 (모델 0). 세션 F1, 2026-10-15. 세 사이트 묶기 2026-10-15 밤.
#
#   DCNET_UA=... bash tools/run_window.sh STATE 'YYYY-MM-DD HH:MM:SS' 'YYYY-MM-DD HH:MM:SS'
#   (실험 폴더에서, Bash run_in_background로 하나만 띄운다)
#
# 사이트: 디시(니마갤)·네이버 게임 라운지·블라블라링크를 함께 돈다. 환경 변수 SITES로 고른다 (기본 "dc naver blabla").
#   네이버·블라블라링크 받기는 other_sites.py다. 글 번호는 디시 숫자, 네이버 n…, 블라블라링크 b…라 한 STATE에서 겹치지 않는다.
#   사이트마다 호스트가 달라 목록·본문·사진·원본을 동시에 받는다 (dcnet.py 간격은 호스트별).
# START가 챌린지 시작(season.json c_start, 없으면 start 날짜 + 2일 05:00) 전이면 c_start로 당긴다.
#   1~2일차 글은 받지 않는다 (사용자 결정, 2026-10-15). START를 '-'로 주면 c_start부터.
#
# 순서: 목록(list_crawl·other_sites list) → 후보(select_cand, 이미 받은 글 빼기) → 본문·사진(collect, 받는 대로 1배 OCR)
#       → 1.5배 OCR(깨진 사진만) → 파싱(전체) → 길 정하기(route) → 원본 첨부(orig_fetch) → 시트(tile_sheets)
#       → 글 숫자 추출·Haiku 입력 배치(text_prep).
# 그다음 조율자가 Workflow(raid-window) 한 번, merge_obs.py 한 번을 돌린다.
#
# - STATE: 실행 사이에 이어 가는 폴더 (레포 밖). 처음에 STATE/season.json을 만들어 둔다:
#     {"name": "S41", "boss": "리버렐리오 바디", "start": "2026-09-24 12:00:00", "end": "2026-10-01 04:59:59",
#      "e_start": "2026-09-24 12:00:00", "e_end": "2026-10-02 12:00:00"}
#   c_start(챌린지 시작)는 넣어 두는 것이 좋다 (예: S40 "2026-08-22 05:00:00").
#   STATE에 생기는 것: posts/ img/ orig/ manifest.jsonl ocr.jsonl ocr2.jsonl parsed.jsonl route.jsonl posts_min.jsonl(text_prep)
#   reads.jsonl(merge_obs) obs.jsonl·claims.jsonl(merge_obs) runs.jsonl ocr/ models/ runs/{RUN}/
# - 이미 받은 글(STATE/posts 또는 STATE/posts_min.jsonl)·사진·OCR은 건너뛴다. 기간을 겹쳐 돌려도 같은 글을 두 번 처리하지 않는다.
# - 세션이 바뀌면 state_sync.py import로 data/solo-raid/S{n}/에서 STATE를 불러오고, 실행·합치기가 끝날 때마다 export로 내보낸다.
# - 중간에 실패하면 같은 인자로 다시 돌린다. RUN/nos.txt가 있으면 목록·후보를 건너뛰고, 길 정하기는 그 RUN 것을 다시 한다.
# - 끝나면 RUN/DONE에 요약 JSON 한 줄, 실패하면 RUN/FAILED에 단계 이름. STATE/last_run에 RUN 경로.
# - UA: DCNET_UA가 없으면 Claude-User (사람이 시킬 때). 자동 실행은 자기 이름 UA (dcnet.AUTO_UA).
set -euo pipefail

STATE=$(mkdir -p "${1:?STATE}" && cd "$1" && pwd)
START=${2:?START}
END=${3:?END}
T=$(cd "$(dirname "$0")" && pwd)
[ -s "$STATE/season.json" ] || { echo "STATE/season.json이 없다" >&2; exit 2; }
SITES=${SITES:-dc naver blabla}
for s in $SITES; do case $s in dc|naver|blabla) ;; *) echo "SITES에 모르는 사이트: $s" >&2; exit 2;; esac; done
has() { case " $SITES " in *" $1 "*) return 0;; esac; return 1; }
C_START=$(python3 -I -c "
import json, sys
sys.path.insert(0, sys.argv[2])
import season_filter
print(season_filter.c_start(json.load(open(sys.argv[1], encoding='utf-8'))))" "$STATE/season.json" "$T")
if [ "$START" = "-" ] || [[ "$START" < "$C_START" ]]; then
  echo "START $START → 챌린지 시작 $C_START" >&2
  START=$C_START
fi
[[ "$START" < "$END" ]] || { echo "START($START)가 END($END)보다 늦다" >&2; exit 2; }
RUN=$STATE/runs/$(echo "${START}_${END}" | tr -d ' :-' | tr '_' '-')
mkdir -p "$RUN" "$STATE/posts" "$STATE/img" "$STATE/orig" "$STATE/models"
rm -f "$RUN/DONE" "$RUN/FAILED"
STEP=start
trap 'echo "$STEP" > "$RUN/FAILED"' ERR
log() { echo "$(TZ=Asia/Seoul date +%H:%M:%S) $*" >> "$RUN/run.log"; }
export FETCHLOG=$RUN/fetchlog.tsv DCNET_LOCKDIR=$STATE BLA_CACHE=$STATE/bla_cache.jsonl  # 블라블라링크 목록 재사용 (other_sites.bla_list, S40 시범 2026-10-16)
SEASON() { python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$STATE/season.json" "$1"; }
T0=$(date +%s)
log "run $START ~ $END sites=[$SITES] c_start=$C_START UA=${DCNET_UA:-Claude-User}"

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

STEP=list
if [ -s "$RUN/nos.txt" ]; then
  log "list·cand 건너뜀 (같은 RUN 다시 돌림)"
else
# 사이트마다 목록을 따로 받고(동시에) 합친다. 하나라도 실패하면 이 단계 실패다
LP=()
has dc && { python3 -I "$T/list_crawl.py" "$START" "$END" "$RUN/list_dc.jsonl" >> "$RUN/run.log" 2>&1 & LP+=($!); }
for s in naver blabla; do
  has $s && { python3 -I "$T/other_sites.py" list $s "$START" "$END" "$RUN/list_$s.jsonl" >> "$RUN/run.log" 2>&1 & LP+=($!); }
done
for p in "${LP[@]}"; do wait "$p"; done
cat $(for s in $SITES; do echo "$RUN/list_$s.jsonl"; done) > "$RUN/list.jsonl"
STEP=cand
python3 -I "$T/select_cand.py" "$RUN/list.jsonl" "$RUN/cand" "$(SEASON e_start)" "$(SEASON e_end)" >> "$RUN/run.log"
python3 -I -c "
import json, os, sys
sys.path.insert(0, sys.argv[3])
from sites import sort_key
state, run = sys.argv[1:3]
nos = [l.split('\t')[0] for l in open(os.path.join(run, 'cand', 'cand.tsv')).read().split('\n')[1:] if l]
pm = os.path.join(state, 'posts_min.jsonl')
seen = {str(json.loads(l)['no']) for l in open(pm, encoding='utf-8') if l.strip()} if os.path.exists(pm) else set()
new = [n for n in nos if n not in seen and not os.path.exists(os.path.join(state, 'posts', n + '.json'))]
open(os.path.join(run, 'nos.txt'), 'w').write(''.join(n + '\n' for n in sorted(new, key=sort_key)))
print('cand', len(nos), 'new', len(new))" "$STATE" "$RUN" "$T" >> "$RUN/run.log"
log "list $(wc -l < "$RUN/list.jsonl") $(tail -1 "$RUN/run.log")"
fi
# 네이버·블라블라링크는 목록 줄에 본문·사진 주소가 다 있어 글 JSON을 여기서 만든다 (요청 0). 다시 돌려도 있는 글은 건너뛴다
grep -v '^[0-9]' "$RUN/nos.txt" > "$RUN/nos_other.txt" || true
grep '^[0-9]' "$RUN/nos.txt" > "$RUN/nos_dc.txt" || true
[ -s "$RUN/nos_other.txt" ] && python3 -I "$T/other_sites.py" posts "$RUN/list.jsonl" "$RUN/nos_other.txt" "$STATE/posts" >> "$RUN/run.log"

STEP=collect+ocr1
# 디시(collect.py)와 두 사이트(other_sites.py collect)를 동시에 받는다. 둘 다 같은 manifest.jsonl에 줄 단위로 덧붙인다
CP=()
[ -s "$RUN/nos_dc.txt" ] && { ( cd "$RUN" && python3 -I "$T/collect.py" "$RUN/nos_dc.txt" "$STATE/posts" "$STATE/img" "$STATE/manifest.jsonl" ) >> "$RUN/run.log" 2>&1 & CP+=($!); }
[ -s "$RUN/nos_other.txt" ] && { ( cd "$RUN" && python3 -I "$T/other_sites.py" collect "$RUN/nos_other.txt" "$STATE/posts" "$STATE/img" "$STATE/manifest.jsonl" ) >> "$RUN/run.log" 2>&1 & CP+=($!); }
alive() { local p; for p in "${CP[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
touch "$STATE/ocr.jsonl" "$STATE/ocr2.jsonl"
while alive; do
  "$VENV/bin/python" "$T/ocr_run.py" "$M" "$STATE/img" "$STATE/ocr.jsonl" 2>> "$RUN/ocr.err"
  sleep 20
done
for p in "${CP[@]}"; do wait "$p"; done
"$VENV/bin/python" "$T/ocr_run.py" "$M" "$STATE/img" "$STATE/ocr.jsonl" 2>> "$RUN/ocr.err"
log "collect done, ocr1 $(wc -l < "$STATE/ocr.jsonl")"

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

STEP=route
python3 -I "$T/route.py" "$STATE" "$RUN" >> "$RUN/run.log"

STEP=orig
# 디시 원본(Referer 필요)은 orig_fetch.py, 두 사이트 원본은 other_sites.py orig. 같은 목록에서 자기 줄만 받는다 (동시에)
python3 -I "$T/orig_fetch.py" "$RUN/orig_todo.jsonl" "$RUN/orig_ids.txt" "$STATE/orig" >> "$RUN/run.log" 2>&1 &
O1=$!
python3 -I "$T/other_sites.py" orig "$RUN/orig_todo.jsonl" "$RUN/orig_ids.txt" "$STATE/orig" >> "$RUN/run.log" 2>&1 &
O2=$!
wait $O1
wait $O2
# 원본이 없거나 못 받은 사진은 850px 사본으로 대신한다.
# 둘 다 없는 사진(다른 세션에서 불러온 STATE의 앞 실행 사진)은 이번 판독에서 빼고, 띠 판독 줄을 그대로 둔다
mkdir -p "$RUN/origsrc"
: > "$RUN/orig_ids.ok"
while read -r i; do
  [ -n "$i" ] || continue
  if [ -s "$STATE/orig/$i.img" ]; then ln -sf "$STATE/orig/$i.img" "$RUN/origsrc/$i.img"
  elif [ -s "$STATE/img/$i.img" ]; then ln -sf "$STATE/img/$i.img" "$RUN/origsrc/$i.img"
  else echo "orig_nosrc $i" >> "$RUN/run.log"; continue; fi
  echo "$i" >> "$RUN/orig_ids.ok"
done < "$RUN/orig_ids.txt"
mv "$RUN/orig_ids.ok" "$RUN/orig_ids.txt"
if [ -s "$RUN/pending_orig_used.txt" ]; then
  grep -Fxf "$RUN/orig_ids.txt" "$RUN/pending_orig_used.txt" > "$RUN/pending_orig_used.ok" || true
  mv "$RUN/pending_orig_used.ok" "$RUN/pending_orig_used.txt"
fi
# 받았어도 PIL이 못 여는 원본(큰 움직이는 webp 등, 세션 F2)도 사본으로 대신한다
python3 -I -c "
import os, sys
from PIL import Image
run, state = sys.argv[1:3]
for i in open(os.path.join(run, 'orig_ids.txt')).read().split():
    p = os.path.join(run, 'origsrc', i + '.img')
    try:
        Image.open(p).convert('RGB')
    except Exception:
        os.remove(p)
        os.symlink(os.path.join(state, 'img', i + '.img'), p)
        print('orig_unreadable', i)" "$RUN" "$STATE" >> "$RUN/run.log"

STEP=sheets
rm -rf "$RUN/sheets"
TILE_W=850 TILE_NOWHOLE=1 python3 -I "$T/tile_sheets.py" "$RUN/strip_ids.txt" "$STATE/parsed.jsonl" "$STATE/ocr.jsonl" "$STATE/img" "$RUN/sheets" >> "$RUN/run.log"
N1=$(wc -l < "$RUN/sheets/sheets.jsonl")
TILE_W=1400 TILE_FULL=1 TILE_START=$N1 python3 -I "$T/tile_sheets.py" "$RUN/orig_ids.txt" "$STATE/parsed.jsonl" "$STATE/ocr.jsonl" "$RUN/origsrc" "$RUN/sheets" >> "$RUN/run.log"
NS=$(wc -l < "$RUN/sheets/sheets.jsonl")

STEP=text
TX=$(python3 -I "$T/text_prep.py" "$STATE" "$RUN")

STEP=done
python3 -I -c "
import collections, json, sys, os
run, state, tx, n1, ns, sec, tools = sys.argv[1:8]
sys.path.insert(0, tools)
from sites import site_of
req = [l.split('\t') for l in open(os.path.join(run, 'fetchlog.tsv'))] if os.path.exists(os.path.join(run, 'fetchlog.tsv')) else []
lst = collections.Counter(site_of(json.loads(l)['no']) for l in open(os.path.join(run, 'list.jsonl')))
nos = open(os.path.join(run, 'nos.txt')).read().split()
d = {'run': run, 'list': sum(lst.values()), 'list_by_site': dict(lst),
     'new_posts': len(nos), 'new_by_site': dict(collections.Counter(site_of(n) for n in nos)),
     'route': json.load(open(os.path.join(run, 'route_counts.json'))),
     'sheets': int(ns), 'sheets_850': int(n1), **json.loads(tx),
     'requests': len(req), 'req_fail': sum(1 for r in req if r[1] != '200'), 'sec': int(sec)}
print(json.dumps(d, ensure_ascii=False))" "$RUN" "$STATE" "$TX" "$N1" "$NS" "$(( $(date +%s) - T0 ))" "$T" > "$RUN/DONE"
echo "{\"run\": \"$RUN\", \"start\": \"$START\", \"end\": \"$END\", \"done\": \"$(TZ=Asia/Seoul date '+%F %T')\"}" >> "$STATE/runs.jsonl"
echo "$RUN" > "$STATE/last_run"
log "done"
cat "$RUN/DONE"
