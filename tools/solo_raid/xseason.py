"""앞 시즌에 이미 올라온 사진 (시즌을 건넌 재업로드, 2026-10-17 분석 ②, 모델 0, 요청 0).

같은 사진 파일(sha1)이 앞 시즌 받기 구간에 이미 있으면, 이번 시즌 화면일 수 없다 — 지난 시즌 기록을 다시 올린 것이다.
S31 글 …·4196485가 S30 글 …_1(그것도 S29 3810211_0의 재업로드)을 다시 올려 S31 기본 집합에 들어갔던 일에서 나왔다
(S30·S27 시즌 세션 알림). S30~S41에 9장.

앞 시즌 = 같은 data/solo-raid 아래 번호가 작은 시즌 중 수집이 끝난(collect_done.json) 시즌. 받는 중인 시즌은 파일이 바뀌므로 뺀다.
그 시즌 받기 끝(e_end)이 이번 시즌 받기 시작(e_start)보다 늦으면 구간이 겹쳐 뺀다.
값(점수·%)은 읽지 않는다 — route.jsonl의 sha1만 본다 (시험 시즌 정답을 열지 않음).
"""
import json
import os


def _season_num(state):
    try:
        name = json.load(open(os.path.join(state, "season.json"), encoding="utf-8")).get("name") or ""
        return int(name[1:])
    except (OSError, ValueError):
        return None


def earlier(state):
    """→ [(시즌 번호, 그 시즌 state 경로)] 이 규칙이 쓰는 앞 시즌."""
    n = _season_num(state)
    if n is None:
        return []
    base = os.path.dirname(os.path.dirname(os.path.abspath(state)))
    me = json.load(open(os.path.join(state, "season.json"), encoding="utf-8"))
    out = []
    for d in sorted(os.listdir(base)):
        if not (d.startswith("S") and d[1:].isdigit()) or int(d[1:]) >= n:
            continue
        st = os.path.join(base, d, "state")
        if not (os.path.exists(os.path.join(base, d, "collect_done.json")) and os.path.exists(os.path.join(st, "route.jsonl"))):
            continue
        try:
            other = json.load(open(os.path.join(st, "season.json"), encoding="utf-8"))
        except OSError:
            continue
        if other.get("e_end") and me.get("e_start") and other["e_end"] > me["e_start"]:
            continue
        out.append((int(d[1:]), st))
    return out


def photo_sha(state):
    """→ {sha1: "S{m}:{사진 id}"} 앞 시즌 받기 구간에 있던 사진 (가장 앞 시즌)."""
    seen = {}
    for m, st in sorted(earlier(state), reverse=True):
        for l in open(os.path.join(st, "route.jsonl"), encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("sha1"):
                    seen[r["sha1"]] = f"S{m}:{r['id']}"
    return seen


def stamp_paths(state):
    """도장(post_stamp)에 넣을 입력 파일 → [(이름, 경로)]."""
    return [(f"S{m}/state/route.jsonl", os.path.join(st, "route.jsonl")) for m, st in earlier(state)]
