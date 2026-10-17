"""앞 시즌에 이미 올라온 사진 (시즌을 건넌 재업로드, 2026-10-17 분석 ②, 모델 0, 요청 0).

같은 사진 파일(sha1)이 앞 시즌 받기 구간에 이미 있으면, 이번 시즌 화면일 수 없다 — 지난 시즌 기록을 다시 올린 것이다.
S31 글 …·4196485가 S30 글 …_1(그것도 S29 3810211_0의 재업로드)을 다시 올려 S31 기본 집합에 들어갔던 일에서 나왔다
(S30·S27 시즌 세션 알림). S30~S41에 9장.

앞 시즌 = 같은 data/solo-raid 아래 번호가 작은 시즌 중 수집이 끝난(collect_done.json) 시즌. 받는 중인 시즌은 파일이 바뀌므로 뺀다.
그 시즌 받기 끝(e_end)이 이번 시즌 받기 시작(e_start)보다 늦으면 구간이 겹쳐 뺀다.
photo_sha는 값(점수·%)을 읽지 않는다 — route.jsonl의 sha1만 본다.

score_seen (2026-10-17 분석 ③): 파일은 달라도(크기 바꿈·캡처·모음) 사진의 정확 점수(10자리 이상, 글자 판독과 맞는 것) 하나가
앞 시즌의 시즌 안(season in) 관측에 이미 있으면 그 사진은 앞 시즌 화면이다 — 점수는 시즌마다 0부터 쌓이므로 10자리 정확 점수가 시즌을 건너 같을 수 없다.
순위표 1~7위 줄(% 없음)도 쓴다. S26 3402860_1·3402833_0(S25 한섭 최종 순위표, 1위 22,679,377,987 — 시즌 세션은 「S26 초·중반 화면」으로 봄),
S31 4175575·4182214(S23·S26·S28 기록 회상), S28 3695740(S25 결과 화면 4장) 등 (S23·S26 「다음 시즌에 바꿀 것」).
앞 시즌 obs.jsonl의 원관측만 읽고 정답 파일(truth*)·집계표는 열지 않는다. 시험 시즌이 앞 시즌이어도 그 값은 맞대기에만 쓰고 어디에도 적지 않는다.
"""
import json
import os
import re

REPO_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "solo-raid")


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
    if not any(d.startswith("S") and d[1:].isdigit() for d in os.listdir(base)):
        # 검증의 T 절단 사본(작업 폴더)처럼 시즌 폴더 밖이면 레포의 data/solo-raid를 본다 (2026-10-17 분석 ④ — 전에는 사본에서 xseason이 꺼졌다)
        base = REPO_DATA
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


def exact(o):
    """정확 점수 (10자리 이상 정수, 글자 판독이 있으면 그것과 같을 때만. 코드 판독 줄은 글자 없이도). 아니면 None."""
    s = o.get("score")
    if not isinstance(s, (int, float)) or s != int(s) or s < 1e9:
        return None
    digits = re.sub(r"\D", "", o.get("score_txt") or "")
    if digits:
        return int(s) if int(digits) == int(s) else None
    return int(s) if o.get("src") == "code" else None


def score_seen(state):
    """→ {정확 점수: "S{m}:{사진 id}"} 앞 시즌의 시즌 안 관측 (가장 앞 시즌)."""
    seen = {}
    for m, st in sorted(earlier(state), reverse=True):
        p = os.path.join(st, "obs.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p, encoding="utf-8"):
            if l.strip():
                o = json.loads(l)
                e = exact(o)
                if e and o.get("season") == "in":
                    seen[e] = f"S{m}:{o['img']}"
    return seen


def stamp_paths(state):
    """도장(post_stamp)에 넣을 입력 파일 → [(이름, 경로)]."""
    out = []
    for m, st in earlier(state):
        out.append((f"S{m}/state/route.jsonl", os.path.join(st, "route.jsonl")))
        if os.path.exists(os.path.join(st, "obs.jsonl")):
            out.append((f"S{m}/state/obs.jsonl", os.path.join(st, "obs.jsonl")))
    return out
