"""관측 줄에 서버를 붙인다 (모델 0, 요청 0). 합치기(merge_obs.py)가 끝난 뒤 시즌 전체에 따로 돌린다.

python3 -I tools/server_classify.py STATE [--posts FILE.jsonl ...] [--snap DIR ...] [--no-code-scores]

점수·%의 크기로는 서버를 정하지 않는다 (docs/solo-raid-cutoff.md, 순환 주의). 근거는 두 가지뿐이다.
- 글 텍스트 `text`: 제목·본문·작성자 댓글에 서버 이름이 하나만 나오면 그 글의 관측 전체에 붙인다.
  - 두 서버 이상이 나오면(`multi`), 서버 이름이 값을 묻는 문장("몇퍼"·"얼마")에만 나오면(`q`),
    남의 댓글에만 나오면(`cmt`) 후보(`cand`)로만 남긴다.
  - 텍스트는 STATE/posts_min.jsonl·claims.jsonl, 그리고 --posts(no·title·body 줄)·--snap(기준점 글 스냅숏, 댓글 포함)으로 더 준다.
- 사진 지문 `top7`: 정확한 점수(10자리 이상, 끝 6자리가 0이 아닌 값)가 같은 사진끼리 묶는다. 같은 점수는 같은 사람이라
  같은 서버다. 같은 사진(route.jsonl sha1)도 한 덩어리다. 상위 7명뿐 아니라 순위표 다른 줄·본인 줄·덱 점수도 잇는다.
  - 코드 OCR 줄(src code)은 score_txt 없이 점수만 남기므로 그 점수의 자릿수로 본다 (기본 켬, --no-code-scores로 끈다).
    S41에서 켜도 묶음이 두 서버에 걸리거나(앵커 엇갈림·conflict 0), 같은 점수의 %가 시간을 거슬러 줄거나,
    집계표·정답 곡선과 어긋나는 줄이 없어 기본으로 했다 (2026-10-16, docs/solo-raid-cutoff.md 「서버 분류」).
  - 묶음 이름은 그 안의 `text` 글(앵커)로 정한다: 앵커 글 2개 이상이고 한 서버가 75% 이상이면 그 서버.
    앵커 글이 1개면 `top7_1`: 그 묶음의 줄(지문·같은 사진으로 이어진 줄)에만 앵커 서버를 붙인다 (글 단위로 퍼뜨리지 않는다).
    줄의 글 텍스트 후보(q·cmt·multi)가 그 서버를 빼거나, 글이 다른 서버 묶음으로 가면 후보로만 남긴다.
    같은 사진(sha1)이 여러 글에 올라온 경우도 사진 하나의 묶음으로 보고 같은 규칙을 쓴다 (묶음 번호는 붙이지 않는다).
    2026-10-16부터 기본 (전에는 후보로만 남김, --no-single-anchor로 되돌린다). 같은 정확 점수는 같은 사람이라
    앵커 글의 텍스트를 그 글의 줄에 믿는 것과 같은 근거다. S41에서 시즌 안 % 관측 14줄이 더 붙었다
    (한섭 9·일섭 4·북미 1, 모든 줄 22줄. 서버가 바뀌는 줄 0). post_link의 글 단위 열쇠(post_self·post_one)로만 이어진 줄과
    top7_post로만 서버가 정해진 사진의 다른 글 사본에는 붙이지 않는다.
  - 서버 섞인 덩어리 가르기 (2026-10-16, S33 c1). 이어붙인 사진·사진 한 줄이 다른 서버 화면을 한 덩어리로 이으면 앵커 한 건 차이로
    덩어리 전체 이름이 바뀌었다 (S33: 앵커 한섭 24·글섭 9·일섭 1, 72.7%로 이름 없음 → use의 82%가 서버 미정).
    이음의 세기 = 두 사진이 함께 가진 정확 점수 개수. 앵커 서버가 둘 이상인 덩어리는 세기 t+1 이상으로 다시 나눠 보고(t = 1 → 4까지),
    그 안에 서로 다른 서버로 top7 이름이 서는 부분이 둘 이상이면 합치지 않는다 (t 상한 7 — 2026-10-16 S32에서 4 → 7).
    이어붙인 순위표 사진(한 사진 안에서 1위가 다른 점수로 두 번 이상)은 순위표마다 마디를 나눈다 (S32 4381843_0). 나머지 부분은 세기 t 이음으로 가장 가까운 top7 부분에
    붙고, 가장 가까운 부분들의 서버가 갈리면 다리(`bridge`, 이름 없음)로 남는다. 서로 다른 서버로 확실한 부분이 없으면 전과 같다
    (S35~S41은 줄 하나 바뀌지 않음). 같은 서버 순위표 사진끼리는 1~7위 점수를 여럿 함께 가져 세기 2 이상으로 이어진다.
    server.jsonl의 `cluster`는 가른 뒤 묶음, `link`는 가르기 전 덩어리(g1…)다. 시즌 판정(post_exclude)·묶음 이음(truth)은 link를 쓴다.
  - 글 단위 `top7_post`: 글의 사진들이 이름 붙은 묶음 하나로만 가면, 묶음 밖 사진(잘라 낸 점수 막대 등)과 글 주장도 그 서버다.
    글의 사진이 서로 다른 서버 묶음으로 가면 후보로만 남긴다. 사진 자신이 앵커 1개 묶음(top7_1)이고 그 서버가 글의 다른 사진 묶음에
    없으면 top7_post로 덮지 않는다 (한 글에 한섭·북미 순위표, S33 4555355, 2026-10-16).
- 최종 `server`: text와 지문이 같으면 `text+top7`(`text+top7_post`), 하나만 있으면 그것, 다르면 `conflict`(server 없음).
- 고침 `fix`: data/solo-raid/S{n}/discard.json의 action server 항목 (원본을 보고 서버만 고친 글, 2026-10-16). 글 텍스트 대신 쓰고,
  앵커로도 세며, 묶음 이름과 달라도 fix가 이긴다. action discard 글은 텍스트를 쓰지 않는다 (줄은 post_exclude가 뺀다).
  줄 하나를 버린 항목(scope row, 판독 틀림 등)의 점수는 지문 잇기에 쓰지 않는다.
  사진 단위 고침(scope img, action server)은 그 사진(같은 sha1 사본 포함)의 모든 줄에 fix를 쓰고 그 사진을 앵커 하나로 센다 (글 텍스트는
  그 사진의 앵커가 되지 않는다). 줄 단위 고침(scope row, action server)은 그 줄에만 fix를 쓴다 (2026-10-16, S33 4573600·4573011).

서버 이름: KR(한섭·한국 서버·KOR) · JP(일섭·일본 서버) · GL(글섭·글로벌) · NA(북미) · SEA(동남아). `무섭`·`타섭`·`본섭`은 서버가 아니다.
이름 바로 뒤가 가정·비교("한섭이였으면"·"한섭이랑"·"한섭보다")면 그 서버가 아니라는 뜻이라 세지 않는다.
이름 바로 앞이 「번외」·「덤으로」면 덧붙인 서버라 세지 않는다 (S32 4368058 「번외 북미섭」, 2026-10-16).

출력
- STATE/server.jsonl: obs.jsonl 한 줄마다 {site, no, img, row, server, basis, cand, cluster, link, quote}. 이음 열쇠는 (img, row).
  `quote`는 서버 이름 앞뒤 짧은 원문이다 (작성자 정보는 넣지 않는다).
- STATE/server_claims.jsonl: claims.jsonl 한 줄마다 {i(줄 순서), site, no, server, basis, cand, quote}. 글 단위 근거만 쓴다.
- STATE/server_summary.json: 근거별 줄 수, 묶음별 사진 수·앵커·이름, 시즌 안 % 관측의 서버별 줄 수.
- 표준 출력에 요약 한 줄.
obs.jsonl의 `server` 칸은 판독 단계 칸이라 건드리지 않는다. merge_obs.py가 관측을 다시 쓰면 이 스크립트도 다시 돌린다.
"""
import argparse
import ast
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools", "solo_raid"))
import discard  # noqa: E402  data/solo-raid/S{n}/discard.json (할 일 3.A2)

SRV = [
    ("KR", r"한국\s?서버|한국\s?섭|(?<![A-Za-z])(?:KR|KOR)(?![A-Za-z])"),
    ("JP", r"일본\s?서버|일본\s?섭|(?<![A-Za-z])(?:JP|JPN)(?![A-Za-z])|日本サーバ|日服"),
    ("GL", r"글로벌|(?<![A-Za-z])(?:GLOBAL|Global)(?![A-Za-z])"),
    ("NA", r"[북븜]미|(?<![A-Za-z])NA(?![A-Za-z])"),
    ("SEA", r"동남아|(?<![A-Za-z])SEA(?![A-Za-z])"),
]
SRV_RE = [(k, re.compile(p)) for k, p in SRV]
# 한섭·일섭·글섭과 일글섭처럼 붙여 쓴 것. 앞 글자는 따지지 않는다 (미친한섭·넘는한섭)
SHORT_RE = re.compile(r"([한일글]+)[섭썹](?![섭썹])")   # 섭섭은 뺀다
SHORT = {"한": "KR", "일": "JP", "글": "GL"}
URL_RE = re.compile(r"https?://\S+")
SENT_END = re.compile(r"[?!\n]+|(?<!\d)\.+|\.+(?!\d)")   # 0.53퍼의 점에서는 자르지 않는다
# 값을 묻는 문장만 질문으로 본다 ("140억이면 일섭 몇퍼?"). `ㅁㅌㅊ?`처럼 물음표만 있는 자기 인증은 질문이 아니다
Q_RE = re.compile(r"몇\s?(?:퍼|%|프로|등|위|점)|얼마")
# 서버 이름 바로 뒤가 가정·비교면 그 서버가 아니라는 뜻이라 근거에서 뺀다 (S41: "한섭이였으면 3퍼 명함도 못내밀"은 일섭 글,
# "보통 한섭이랑 30ㅡ35억정도 차이났"은 다른 서버 글)
NOT_RE = re.compile(r"(?:\s?(?:섭|썹|서버))?(?:(?:이?[였었]?으면|이?라면|이면|보다|이?랑|과(?!금)|와)|\s?(?:보다|대비|에 ?비해)"
                    r"|(?:으로|로|에)?\s?(?:갔|넘어갔|옮겼)었?으면|(?:으로|로|에)\s?가면)")   # 「글섭갔으면」(S33 4573593, 2026-10-16)
# 「번외 북미섭」처럼 덧붙인 서버는 글의 주 서버가 아니다 (S32 4368058, 2026-10-16)
EXTRA_RE = re.compile(r"번외|덤으로|덤\s*:")
TRACKER_NO_ANCHOR = True
ID_RE = re.compile(r"[A-Za-z0-9_]{6,}")   # 인용에서 닉네임·친구 코드처럼 보이는 긴 영숫자는 가린다


def jl(p):
    out = []
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                out.append(json.loads(l))
    return out


def mentions(text):
    """[(서버, 질문문장인가, 인용)]"""
    out = []
    text = URL_RE.sub(" ", text or "")
    cuts = [0] + [m.end() for m in SENT_END.finditer(text)] + [len(text)]
    for s in (text[i:j] for i, j in zip(cuts, cuts[1:])):
        if not s.strip():
            continue
        found = []
        for m in SHORT_RE.finditer(s):
            found += [(SHORT[c], m) for c in dict.fromkeys(m.group(1))]
        for k, rx in SRV_RE:
            found += [(k, m) for m in rx.finditer(s)]
        q = bool(Q_RE.search(s))
        found.sort(key=lambda x: x[1].start())
        neg, nxt = set(), None   # "일섭, 글섭이었으면"처럼 나열한 이름은 뒤의 가정을 함께 받는다
        for j in range(len(found) - 1, -1, -1):
            m = found[j][1]
            if NOT_RE.match(s, m.end()) or EXTRA_RE.search(s[max(0, m.start() - 6):m.start()]) or (nxt is not None and nxt in neg
                                            and re.fullmatch(r"[\s,·/및]*", s[m.end():found[nxt][1].start()])):
                neg.add(j)
            nxt = j
        for j, (k, m) in enumerate(found):
            if j in neg:
                continue
            a, b = max(0, m.start() - 15), min(len(s), m.end() + 15)
            out.append((k, q, ID_RE.sub("…", s[a:b]).strip()))
    return out


def load_texts(state, posts_files, snap_dirs):
    """글 번호(str) → [(부분, 글)]. 부분: title·body·op(작성자 댓글)·cmt(남의 댓글)·claim."""
    tx = collections.defaultdict(list)
    seen = collections.defaultdict(set)

    def add(no, part, t):
        if t and (part, t) not in seen[no]:
            seen[no].add((part, t))
            tx[no].append((part, t))
    for f in [os.path.join(state, "posts_min.jsonl")] + posts_files:
        for p in jl(f):
            add(str(p["no"]), "title", p.get("title"))
            add(str(p["no"]), "body", p.get("body"))
    for c in jl(os.path.join(state, "claims.jsonl")):
        add(str(c["no"]), "claim", c.get("txt"))
    for d in snap_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            p = json.load(open(f, encoding="utf-8"))
            no = {"naver": "n", "blabla": "b"}.get(p.get("source"), "") + str(p["id"])
            add(no, "title", p.get("title"))
            add(no, "body", p.get("body"))
            cm = p.get("comments") or []
            if isinstance(cm, str):
                try:
                    cm = ast.literal_eval(cm)
                except (ValueError, SyntaxError):
                    cm = []
            for c in cm:
                add(no, "op" if c.get("by_op") else "cmt", c.get("m"))
    return tx


def post_server(parts):
    """글 하나의 텍스트 근거 → (server|None, basis, cand, quote)."""
    main, side = [], []
    for part, t in parts:
        (side if part == "cmt" else main).extend(mentions(t))
    srv = list(dict.fromkeys(k for k, _, _ in main))
    if len(srv) == 1:
        sure = [x for x in main if not x[1]]
        if sure:
            return srv[0], "text", [], sure[0][2]
        return None, "q", srv, main[0][2]
    if len(srv) > 1:
        return None, "multi", srv, " / ".join(dict.fromkeys(x[2] for x in main))[:120]
    cs = list(dict.fromkeys(k for k, _, _ in side))
    if cs:
        return None, "cmt", cs, side[0][2]
    return None, None, [], None


def exact(r, code_scores=False):
    """정확 점수. code_scores면 코드 OCR 줄(src code, score_txt 없음)의 10자리 이상 점수도 쓴다 (post_link.py exact()와 같다)."""
    s = r.get("score")
    if not isinstance(s, (int, float)) or s != int(s):
        return None
    s = int(s)
    digits = re.sub(r"\D", "", r.get("score_txt") or "")
    if code_scores and not digits and r.get("src") == "code":
        digits = str(s)
    if s < 10 ** 9 or len(digits) < 10 or s % 10 ** 6 == 0:
        return None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--posts", action="append", default=[], help="no·title·body 줄의 jsonl (예: 재수집 posts_min)")
    ap.add_argument("--snap", action="append", default=[], help="글 스냅숏 폴더 (source·id·title·body·comments)")
    ap.add_argument("--code-scores", action=argparse.BooleanOptionalAction, default=True,
                    help="코드 OCR 줄(score_txt 없음)의 점수도 지문으로 쓴다 (기본 켬, 2026-10-16. post_link.py와 같은 기준)")
    ap.add_argument("--single-anchor", action=argparse.BooleanOptionalAction, default=True,
                    help="앵커 글이 1개인 묶음(top7_1)의 줄에 그 서버를 붙인다 (기본 켬, 2026-10-16. 끄면 후보로만)")
    ap.add_argument("--tracker-anchor", action=argparse.BooleanOptionalAction, default=False,
                    help="트래커 화면만 있는 사진도 글 텍스트 앵커로 센다 (2026-10-17 전 동작. 기본 끔)")
    a = ap.parse_args()
    state = a.state
    global TRACKER_NO_ANCHOR
    TRACKER_NO_ANCHOR = not a.tracker_anchor

    obs = jl(os.path.join(state, "obs.jsonl"))
    sha = {r["id"]: r["sha1"] for r in jl(os.path.join(state, "route.jsonl")) if r.get("sha1")}
    tx = load_texts(state, a.posts, a.snap)
    # 버릴 것(discard.json)의 글은 서버 근거로 쓰지 않는다 (앵커가 되지 않게). 줄 자체는 post_exclude가 뺀다
    for no in discard.posts(state):
        tx.pop(no, None)
    # 서버 고침(discard.json action server): 글 텍스트 대신 고친 서버를 근거 fix로 쓴다 (3.2b, 2026-10-16)
    fix = discard.fixes(state)
    bad_rows = set(discard.rows(state))   # 버린 줄(판독 틀림 등)의 점수는 지문으로 잇지 않는다
    img_fix = discard.img_fixes(state)    # 사진 단위 서버 고침 (scope img, 2026-10-16)
    row_fix = discard.row_fixes(state)    # 줄 단위 서버 고침 (scope row, action server)

    def text_server(no):
        if no in fix:
            return fix[no]["server"], "fix", [], ("고침: " + fix[no]["reason"])[:120]
        return post_server(tx.get(no, []))

    # 1) 글 텍스트
    post = {}
    for no in sorted({str(r["no"]) for r in obs}):   # 순서를 고정한다 (server_summary.json이 실행마다 같게)
        post[no] = text_server(no)

    # 2) 사진 지문 묶음: 마디 = 사진(sha1), 같은 정확 점수로 잇는다. 이음의 세기 = 두 사진이 함께 가진 정확 점수 개수
    # 이어붙인 순위표 사진: 한 사진 안에서 순위가 1위부터 다시 시작하면(1위 점수가 둘 이상) 순위표마다 마디를 나눈다
    # (S32 4381843_0: 한섭·일섭 순위표를 이어붙인 사진이 두 서버와 점수 5~7개씩 같아 덩어리가 갈리지 않았다, 2026-10-16).
    # 순위 없는 줄(본인 줄 등)은 바로 앞 순위표에 붙인다.
    seg = {}
    by_img = collections.defaultdict(list)
    for r in obs:
        by_img[r["img"]].append(r)
    collage = set()
    for img, rs in by_img.items():
        k, ranked, ones = 0, False, set()
        for r in sorted(rs, key=lambda r: r["row"]):
            if r.get("rank") == 1 and ranked:
                k += 1
            if r.get("rank") is not None:
                ranked = True
            if r.get("rank") == 1 and r.get("score"):
                ones.add(r["score"])
            seg[(img, r["row"])] = k
        if k and len(ones) >= 2:
            collage.add(img)

    def node(r):
        n = sha.get(r["img"], r["img"])
        return f"{n}#{seg[(r['img'], r['row'])]}" if r["img"] in collage else n
    nscores = collections.defaultdict(set)
    nposts = collections.defaultdict(set)
    for r in obs:
        n = node(r)
        nposts[n].add(str(r["no"]))
        s = None if (r["img"], r["row"]) in bad_rows else exact(r, a.code_scores)
        if s is not None:
            nscores[n].add(s)
    by_score = collections.defaultdict(set)
    for n, ss in nscores.items():
        for s in ss:
            by_score[s].add(n)
    adj = collections.defaultdict(collections.Counter)
    for s in sorted(by_score):
        ns = sorted(by_score[s])
        for x in ns:
            for y in ns:
                if x != y:
                    adj[x][y] += 1
    allnodes = sorted(nposts)
    # 트래커 화면(외부 집계·비교표)만 있는 마디는 글 텍스트 앵커가 되지 않는다 (2026-10-17 분석 ②, S30 3941671_0 —
    # 한국·일본·글로벌 열이 있는 컷 비교표가 제목 「일섭」으로 일섭 앵커가 되어 묶음이 섞였다). 사진 단위 고침(fix)은 그대로 센다
    tnode = collections.defaultdict(lambda: True)
    for r in obs:
        n = node(r)
        tnode[n] = tnode[n] and r.get("screen") == "tracker"
    # 사진 단위 서버 고침(scope img): 그 사진(마디)은 글 텍스트 대신 고친 서버의 앵커다
    nfix = {}
    for img, it in img_fix.items():
        for r in by_img.get(img, []):
            nfix[node(r)] = it["server"]
        nfix.setdefault(sha.get(img, img), it["server"])   # 같은 sha1 사본

    def anchors(nodes):
        """앵커 수: 글 텍스트(text·fix) 글은 글마다 하나 (그 글의 사진이 모두 사진 고침이면 세지 않는다), 사진 고침은 사진마다 하나."""
        c = collections.Counter()
        ps = collections.defaultdict(list)
        for n in nodes:
            if n in nfix:
                c[nfix[n]] += 1
            elif not (TRACKER_NO_ANCHOR and tnode[n]):
                for no in nposts[n]:
                    ps[no].append(n)
        for no in sorted(ps):
            if post[no][1] in ("text", "fix"):
                c[post[no][0]] += 1
        return c

    def name_rule(anc):
        n = sum(anc.values())
        if n:
            k, m = sorted(anc.items(), key=lambda x: (-x[1], x[0]))[0]
            if n >= 2 and m / n >= 0.75:
                return k, "top7"
            if n == 1:
                return k, "top7_1"
        return None, None

    def parts(nodes, t):
        """nodes 안에서 세기 t 이상 이음으로 이어진 덩어리들 (정렬된 목록)."""
        ns, seen, out = set(nodes), set(), []
        for x in sorted(ns):
            if x in seen:
                continue
            st, cc = [x], set()
            while st:
                y = st.pop()
                if y in cc:
                    continue
                cc.add(y)
                st.extend(z for z, w in adj[y].items() if w >= t and z in ns and z not in cc)
            seen |= cc
            out.append(frozenset(cc))
        return out

    # 서버 섞인 묶음 가르기 (2026-10-16, S33 c1 — 다음 시즌에 바꿀 것). 다른 서버 화면이 이어붙인 사진·사진 한 줄로 한 묶음이 되면
    # 앵커 한 건 차이로 묶음 전체 이름이 바뀌었다. 앵커 서버가 둘 이상인 덩어리는 더 센 이음(함께 가진 점수 t+1개 이상)으로 쪼개 보고,
    # 그 안에 서로 다른 서버로 이름이 서는(top7) 부분이 둘 이상이면 합치지 않는다. 나머지 부분은 세기 t 이음으로 가장 가까운
    # top7 부분에 붙인다 (가장 가까운 부분들의 서버가 갈리면 붙이지 않고 다리 `bridge`로 둔다 — 이름 없음).
    TMAX = 7   # 같은 서버 순위표끼리는 1~7위 점수를 최대 7개 함께 가진다 (4 → 7, S32 c1, 2026-10-16)
    bridge = set()

    def refine(nodes, t):
        out = []
        for c in parts(nodes, t):
            if t < TMAX and len(anchors(c)) >= 2:
                subs = refine(c, t + 1)
                lab = [name_rule(anchors(x)) for x in subs]
                conf = {i for i, (k, b) in enumerate(lab) if b == "top7"}
                if len({lab[i][0] for i in conf}) >= 2:
                    out += attach(c, subs, lab, conf, t)
                    continue
            out.append(c)
        return out

    def attach(c, subs, lab, conf, t):
        where = {n: i for i, x in enumerate(subs) for n in x}
        nb = collections.defaultdict(set)
        for n in c:
            for z, w in adj[n].items():
                if w >= t and z in where and where[z] != where[n]:
                    nb[where[n]].add(where[z])
        dist = {i: 0 for i in conf}
        got = {i: {lab[i][0]} for i in conf}
        front = sorted(conf)
        while front:   # 여러 출발점 너비 우선: 같은 거리에서 닿은 서버를 모두 모은다
            nxt = []
            for i in front:
                for j in sorted(nb[i]):
                    if j in conf:
                        continue
                    if j not in dist:
                        dist[j] = dist[i] + 1
                        got[j] = set()
                        nxt.append(j)
                    if dist[j] == dist[i] + 1:
                        got[j] |= got[i]
            front = sorted(set(nxt))
        grp = collections.defaultdict(set)
        out = []
        for i, x in enumerate(subs):
            g = got.get(i, set())
            if len(g) == 1:
                grp[next(iter(g))] |= x
            else:
                if len(g) >= 2:
                    bridge.update(x)
                out.append(x)
        return [frozenset(grp[k]) for k in sorted(grp)] + out

    link = parts(allnodes, 1)
    final = refine(allnodes, 1)
    cid = {n: i for i, x in enumerate(final) for n in x}
    lid = {n: i for i, x in enumerate(link) for n in x}
    linked = {i for i, x in enumerate(final) if any(nscores[n] for n in x)}
    comp_imgs = {i: set(x) for i, x in enumerate(final)}
    comp_posts = {i: set().union(*(nposts[n] for n in x)) for i, x in enumerate(final)}
    # 사진 둘 이상을 이은 묶음만 지문으로 쓴다
    comps = [i for i in comp_imgs if len(comp_imgs[i]) >= 2 and i in linked]
    comps.sort(key=lambda i: (-len(comp_imgs[i]), min(comp_imgs[i])))
    cname = {c: f"c{i + 1}" for i, c in enumerate(comps)}
    lk = [i for i, x in enumerate(link) if len(x) >= 2 and any(nscores[n] for n in x)]
    lk.sort(key=lambda i: (-len(link[i]), min(link[i])))
    lname = {c: f"g{i + 1}" for i, c in enumerate(lk)}
    clabel, csum = {}, []
    for c in comps:
        anc = anchors(comp_imgs[c])
        lab, basis = (None, None) if comp_imgs[c] & bridge else name_rule(anc)
        clabel[c] = (lab, basis)
        g = lid[min(comp_imgs[c])]
        csum.append({"cluster": cname[c], "images": len(comp_imgs[c]), "posts": len(comp_posts[c]),
                     "anchors": dict(sorted(anc.items())), "label": lab, "basis": basis, "link": lname.get(g),
                     "link_images": len(link[g]), **({"bridge": True} if comp_imgs[c] & bridge else {})})

    # 같은 사진이 여러 글에 올라온 사진 하나 덩어리: 앵커 글이 1개면 top7_1 (묶음 번호 없음, 요약 csum에도 넣지 않음)
    for c in sorted(comp_imgs):
        if c in clabel or len(comp_posts[c]) < 2 or comp_imgs[c] & bridge:
            continue
        anc = anchors(comp_imgs[c])
        if sum(anc.values()) == 1:
            clabel[c] = (next(iter(anc)), "top7_1")

    SINGLE = a.single_anchor

    def find(n):
        return cid[n]

    # 글 단위 지문: 글의 사진들이 이름 붙은 묶음 하나로만 가면, 묶음 밖 사진(잘라 낸 점수 막대 등)과 글 주장도 그 서버로 본다
    post_c = collections.defaultdict(set)
    for r in obs:
        lab, basis = clabel.get(find(node(r)), (None, None))
        if basis == "top7":
            post_c[str(r["no"])].add(lab)

    def combine(t, own, no):
        """t = post_server 결과, own = 사진 자신의 묶음 (이름, 근거) → (server, basis, cand)."""
        t_srv, t_basis, t_cand, _ = t
        cand = list(t_cand)
        c_lab, c_basis = own
        single = None
        if c_basis == "top7_1":
            cand = list(dict.fromkeys(cand + [c_lab]))
            single = c_lab
            c_lab = c_basis = None
        pc = post_c.get(no, set())
        if single and pc and single not in pc:
            # 사진 자신의 묶음(앵커 1개) 서버가 글의 다른 사진 묶음 어디에도 없으면 글 단위(top7_post)로 덮지 않는다
            # (S33 4555355·4560439: 한섭 순위표 + 북미 순위표 한 글, 2026-10-16). 그 서버가 다른 사진 묶음에도 있으면 전처럼 후보만
            cand = list(dict.fromkeys(cand + sorted(pc)))
            pc = set()
        if c_lab is None:
            if len(pc) == 1:
                c_lab, c_basis = next(iter(pc)), "top7_post"
            elif pc:
                cand = list(dict.fromkeys(cand + sorted(pc)))
        if t_basis == "fix":
            return t_srv, "fix", cand   # 사람·실무 세션이 원본을 보고 고친 것이 묶음보다 앞선다
        if t_srv and c_lab:
            if t_srv == c_lab:
                return t_srv, "text+" + c_basis, cand
            return None, "conflict", [t_srv, c_lab]
        if t_srv:
            return t_srv, "text", cand
        if c_lab:
            return c_lab, c_basis, cand
        if single and SINGLE and (not t_cand or single in t_cand) and not pc:
            return single, "top7_1", cand
        return None, t_basis or ("top7_1" if cand else None), cand

    # 3) 줄마다 합치기 (줄 단위 서버 고침은 같은 사진 사본의 같은 줄에도 쓴다)
    row_sha = {(sha.get(i, i), rw): it for (i, rw), it in row_fix.items()}
    out, cnt = [], collections.Counter()
    for r in obs:
        no = str(r["no"])
        c = find(node(r))
        rf = row_fix.get((r["img"], r["row"])) or row_sha.get((sha.get(r["img"], r["img"]), r["row"]))
        nf = nfix.get(node(r))
        if rf or nf:
            sv = rf["server"] if rf else nf
            srv, basis = sv, "fix"
            cand = [x for x in combine(post[no], clabel.get(c, (None, None)), no)[2] if x != sv]
            quote = ("고침: " + (rf or img_fix.get(r["img"]) or {"reason": "같은 사진 사본"})["reason"])[:120]
        else:
            srv, basis, cand = combine(post[no], clabel.get(c, (None, None)), no)
            quote = post[no][3]
        cnt[basis] += 1
        out.append({"site": r["site"], "no": r["no"], "img": r["img"], "row": r["row"], "server": srv,
                    "basis": basis, "cand": cand, "cluster": cname.get(c), "link": lname.get(lid[node(r)]), "quote": quote})

    # 4) 글 숫자 주장 (claims.jsonl 줄 순서 i로 잇는다)
    claims = jl(os.path.join(state, "claims.jsonl"))
    out_c, cnt_c = [], collections.Counter()
    for i, c in enumerate(claims):
        no = str(c["no"])
        if no not in post:
            post[no] = text_server(no)
        srv, basis, cand = combine(post[no], (None, None), no)
        cnt_c[basis] += 1
        out_c.append({"i": i, "site": c.get("site"), "no": c["no"], "server": srv, "basis": basis, "cand": cand,
                      "quote": post[no][3]})
    with open(os.path.join(state, "server_claims.jsonl"), "w", encoding="utf-8") as f:
        for o in out_c:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    with open(os.path.join(state, "server.jsonl"), "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    key = {(o["img"], o["row"]): o for o in out}
    core = [r for r in obs if r.get("season") == "in" and r.get("pct") is not None and r.get("screen") != "tracker"]
    summary = {
        "rows": len(out),
        "basis": dict(cnt),
        "server_all": dict(collections.Counter(o["server"] for o in out)),
        "core_rows": len(core),
        "core_server": dict(collections.Counter(key[(r["img"], r["row"])]["server"] for r in core)),
        "core_basis": dict(collections.Counter(key[(r["img"], r["row"])]["basis"] for r in core)),
        "claims_rows": len(out_c),
        "claims_basis": dict(cnt_c),
        "claims_server": dict(collections.Counter(o["server"] for o in out_c)),
        "posts_text": dict(collections.Counter(v[1] for v in post.values())),
        "posts_with_text": sum(1 for no in post if tx.get(no)),
        "posts": len(post),
        "collage_images": sorted(collage),
        "clusters": csum,
    }
    json.dump(summary, open(os.path.join(state, "server_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    big = ", ".join(f"{x['cluster']} {x['images']}장→{x['label'] or '?'}" for x in csum[:4])
    print(f"server: 줄 {len(out)} · % 관측 {len(core)}줄 중 서버 있음 "
          f"{sum(v for k, v in summary['core_server'].items() if k)} · 근거 {dict(cnt)} · 묶음 {len(csum)} ({big})"
          f" · 글 주장 {len(out_c)}줄 중 서버 있음 {sum(1 for o in out_c if o['server'])}")


if __name__ == "__main__":
    main()
