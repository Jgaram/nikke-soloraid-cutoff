"""「발견되면 버릴 것·고칠 것」 기록 data/solo-raid/S{n}/discard.json을 읽는다 (할 일 3.A2, 2026-10-16).

시즌마다 파일 하나 (그 시즌 세션·점검 세션이 고친다. 여러 세션이 한 파일을 함께 고치면 충돌해서 시즌별로 나눔, 2026-10-16).
꼴: {"about": ..., "items": [항목, ...]}. 항목 필드: season·site·no·scope·action·reason·decided(날짜)·by(결정자)·ref, 그리고 아래 것.
원자료(obs.jsonl 등)는 고치지 않는다. post_all.sh가 매번 이 기록을 다시 적용한다.

- scope post, action discard(기본): 글을 버린다. server_classify는 이 글의 텍스트를 서버 근거로 쓰지 않고,
  post_exclude는 이 글의 줄과 같은 사진(sha1·dup_of) 사본 줄에 exclude `discard`를 붙인다.
- scope post, action server (+ `server`: KR·JP·GL·NA·SEA): 판독은 맞고 서버만 틀린 글. server_classify가 글 텍스트 대신 이 서버를 근거 `fix`로 쓴다
  (3.2b S39 5881317, 2026-10-16).
- scope row (+ `img`·`row`), action discard: 관측 줄 하나를 버린다 (판독이 틀린 줄 등). 같은 사진 사본의 같은 줄도 버린다.
  server_classify는 이 줄의 점수를 지문 잇기에 쓰지 않는다. 바른 값을 원본에서 읽었으면 `correct`({"pct":…, "score":…})에 적어 둔다 —
  적용하지는 않는다 (obs.jsonl을 읽는 도구가 많아 값 바꿔 넣기를 한곳에서 할 수 없다. 사례가 쌓이면 다시 정한다).

- scope row (+ `img`·`row`), action server (+ `server`): 그 줄 하나만 서버를 고친다 (이어붙인 사진처럼 한 사진에 여러 서버가 있을 때, 2026-10-16).
  server_classify가 그 줄에 근거 `fix`로 쓴다. 줄의 점수는 지문 잇기에 그대로 쓴다.
- scope img (+ `img`, row 없음), action server (+ `server`): 사진 한 장 전체의 서버를 고친다 (한 글에 서로 다른 서버 사진이 섞인 경우 —
  S33 4573600·4573011, 2026-10-16). server_classify가 그 사진(같은 sha1 사본 포함)의 모든 줄에 `fix`를 쓰고, 그 사진을 묶음의 앵커로 센다.
  그 글의 텍스트 서버는 이 사진의 앵커가 되지 않는다.

items(state) → {열쇠: 항목} (열쇠: post는 글 번호 문자열, row는 "img#row", img는 "img#"). posts(state) → post·discard, fixes(state) → post·server,
rows(state) → {(img, row): 항목} (버린 줄만), row_fixes(state) → {(img, row): 항목}, img_fixes(state) → {img: 항목}. 시즌은 STATE/season.json의 name. 글 번호 문자열은 obs.jsonl의 no와 같은 꼴 (dc는 숫자, naver·blabla는 수집 도구의 no).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "solo-raid")
SCOPES = ("post", "row", "img")
ACTIONS = ("discard", "server")
SERVERS = ("KR", "JP", "GL", "NA", "SEA")


def path_for(name):
    return os.path.join(DATA, name, "discard.json")


def season_name(state):
    with open(os.path.join(state, "season.json"), encoding="utf-8") as f:
        return json.load(f).get("name")


def check(it, name):
    """항목 하나의 꼴을 확인한다. 틀리면 AssertionError."""
    assert it.get("season") == name, f"{name} 파일에 다른 시즌 항목: {it}"
    assert it.get("site") and it.get("no") is not None and it.get("reason") and it.get("decided") and it.get("by"), it
    sc, ac = it.get("scope"), it.get("action", "discard")
    assert sc in SCOPES and ac in ACTIONS, it
    if ac == "server":
        assert it.get("server") in SERVERS, it
    if sc == "row":
        assert it.get("img") and isinstance(it.get("row"), int), it
    elif sc == "img":
        assert ac == "server" and it.get("img") and "row" not in it, it
    else:
        assert "img" not in it and "row" not in it, it
    if sc in ("row", "img"):
        assert str(it["img"]).split("_")[0] == str(it["no"]), f"img와 no가 다른 글: {it}"


def key(it):
    if it["scope"] == "post":
        return str(it["no"])
    return f"{it['img']}#{it.get('row', '')}"


def load(name):
    p = path_for(name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("items", [])


def items(state=None, name=None):
    name = name or season_name(state)
    out = {}
    for it in load(name):
        check(it, name)
        k = key(it)
        assert k not in out, f"{name} {k}: 같은 대상에 항목 하나만"
        out[k] = it
    return dict(sorted(out.items()))


def posts(state):
    return {k: it for k, it in items(state).items() if it["scope"] == "post" and it.get("action", "discard") == "discard"}


def fixes(state):
    return {k: it for k, it in items(state).items() if it["scope"] == "post" and it.get("action") == "server"}


def rows(state):
    return {(it["img"], it["row"]): it for it in items(state).values()
            if it["scope"] == "row" and it.get("action", "discard") == "discard"}


def row_fixes(state):
    return {(it["img"], it["row"]): it for it in items(state).values() if it["scope"] == "row" and it.get("action") == "server"}


def img_fixes(state):
    return {it["img"]: it for it in items(state).values() if it["scope"] == "img"}
