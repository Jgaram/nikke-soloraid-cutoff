#!/usr/bin/env bash
# 솔로레이드 후처리 1단계 전체를 한 번에 다시 만든다 (모델 0, 요청 0, 몇 초).
#   server_classify → post_exclude → post_link → post_time → truth → post_outlier → build_core
#   (post_outlier는 core.jsonl을 읽지 않고 build_core.assemble()로 상류 파일에서 같은 줄을 만든다. build_core가 그 결과를
#    surf_z·surf_flag 열로 붙인다 — 순환 없음, 2026-10-16)
#
# tools/solo_raid/post_all.sh STATE [--posts F.jsonl] [--snap DIR] [--parsed F.jsonl] [--alt F.jsonl] [--runs DIR]
#                                   [--no-code-scores]
#
# 원자료(obs.jsonl·claims.jsonl 등 수집이 만든 파일)와 손 입력(post_manual_season.json·truth_table.json)은 읽기만 한다.
# 다시 쓰는 파일: server.jsonl·server_claims.jsonl·server_summary.json·post_exclude.jsonl·post_link.jsonl·post_claims.jsonl·
#   post_time.jsonl·post_time_claims.jsonl·truth.json·truth_points.jsonl·post_outlier.jsonl·post_outlier_summary.json·
#   core.jsonl·core_summary.json·post_all_stamp.json(입력·도구 sha1 도장, 할 일 3.A1 — 등록부가 최신인지 본다)
#
# 시즌 전용 입력 (해당 단계에만 넘긴다. 같은 값을 환경변수 POST_POSTS·POST_SNAP·POST_PARSED·POST_ALT·POST_RUNS로 줘도 된다)
#   --posts  글 텍스트를 더 준다 (no·title·body 줄)          → server_classify·post_exclude·post_time
#   --snap   기준점 글 스냅숏 폴더 (댓글 포함)              → server_classify·post_time
#   --parsed 코드 판독 줄 (없으면 STATE/parsed.jsonl)        → post_exclude
#   --alt    같은 사진의 다른 판독                           → post_exclude
#   --runs   실행 기록 폴더 (기본 STATE/../runs)             → post_time
#   --no-code-scores  코드 OCR 점수를 지문에 쓰지 않는다 (server_classify·post_link 둘 다. 기본은 쓴다)
#
# S41 (디시 글·판독 사본이 state에 없어 실험 사본을 준다):
#   tools/solo_raid/post_all.sh data/solo-raid/S41/state \
#     --posts experiments/2026-10-15-s41-opus-blind/data/posts_min.jsonl \
#     --snap experiments/2026-10-15-s41-collection-baseline/data/snapshot/posts \
#     --parsed experiments/2026-10-15-solo-raid-low-cost/data/parsed.jsonl \
#     --alt experiments/2026-10-15-s41-opus-blind/data/obs_images.jsonl
set -euo pipefail
[ $# -ge 1 ] || { sed -n '2,10p' "$0"; exit 2; }
ST=$1; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POSTS=${POST_POSTS:-}; SNAP=${POST_SNAP:-}; PARSED=${POST_PARSED:-}; ALT=${POST_ALT:-}; RUNS=${POST_RUNS:-}; CODE=()
while [ $# -gt 0 ]; do
  case $1 in
    --posts) POSTS=$2; shift 2 ;;
    --snap) SNAP=$2; shift 2 ;;
    --parsed) PARSED=$2; shift 2 ;;
    --alt) ALT=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --no-code-scores) CODE=(--no-code-scores); shift ;;
    *) echo "모르는 옵션: $1" >&2; exit 2 ;;
  esac
done
[ -f "$ST/obs.jsonl" ] || { echo "$ST/obs.jsonl 없음" >&2; exit 1; }
P=(); [ -n "$POSTS" ] && P=(--posts "$POSTS")
S=(); [ -n "$SNAP" ] && S=(--snap "$SNAP")
X=(); [ -n "$PARSED" ] && X=(--parsed "$PARSED")
A=(); [ -n "$ALT" ] && A=(--alt "$ALT")
R=(); [ -n "$RUNS" ] && R=(--runs "$RUNS")
py() { python3 -I "$ROOT/$1" "${@:2}"; }

echo "[1/7] server_classify"; py experiments/2026-10-15-solo-raid-low-cost/tools/server_classify.py "$ST" "${P[@]}" "${S[@]}" "${CODE[@]}"
echo "[2/7] post_exclude";    py tools/solo_raid/post_exclude.py "$ST" "${P[@]}" "${X[@]}" "${A[@]}"
echo "[3/7] post_link";       py tools/solo_raid/post_link.py "$ST" "${CODE[@]}"
echo "[4/7] post_time";       py tools/solo_raid/post_time.py "$ST" "${P[@]}" "${S[@]}" "${R[@]}"
echo "[5/7] truth";           py tools/solo_raid/truth.py "$ST"
echo "[6/7] post_outlier";    py tools/solo_raid/post_outlier.py "$ST"
echo "[7/7] build_core";      py tools/solo_raid/build_core.py "$ST"
py tools/solo_raid/post_stamp.py "$ST" --extra "posts=$POSTS" --extra "snap=$SNAP" --extra "parsed=$PARSED" --extra "alt=$ALT" --extra "runs=$RUNS" --opts="${CODE[*]:-}"
