"""3단계 1·2번: 글 제목·본문에서 (시각 표현, %, 점수 억) 묶음을 코드로 뽑고 규칙으로 분류한다. 모델 0, 요청 0.

python3 -I tools/text_extract.py POSTS_MIN.jsonl OUT.jsonl

- 문장 조각(마침표·느낌표·물음표·줄바꿈·'/' 기준)마다 %와 점수를 찾는다.
  - %: `N%`·`N퍼`·`N프로`·`N펀데`, 그리고 % 없이 소수만 있는 값(`0.7딱코`, 정수부 한 자리 + 소수 1~2자리).
  - 점수: `N억`, 그리고 `억` 없이 100~400 사이 숫자(`10퍼컷은 173`). `A > B`·`A에서 B로`는 B만 남긴다.
- 한 조각에 %가 여러 개면 % 하나에 한 줄. 점수가 하나뿐이면 모든 %에 붙이고, 개수가 같으면 차례로 짝짓는다.
- %마다 바로 앞의 시각 표현(`13시`·`지금`·`어제` 등)을 붙인다.
- 같은 글에서 자기 기록 점수가 하나뿐이면 점수 없는 자기 기록 %에 그 점수를 붙인다(`score_from: "post"`).
- 분류: prev_season·question·cutoff·predict·other·self. 질문형 자기 값("지금 4퍼면 20퍼 안에 드냐?")은 self에 `q: true`.
- 계산기·모의전 이야기는 predict. 증가량(`10분에 0.01퍼`·`0.5퍼 차이`)은 kind `delta`로 관측에서 뺀다. `3.5 유입`·`3.5 뉴비`는 %가 아니다.
- `core`: 소수 %이거나 %와 점수가 함께 있는 줄. 정수 %만 있는 줄은 대개 구간 이름("3퍼 따고 싶다")이라 따로 본다.
"""
import json
import re
import sys

NUM = r"(?<![\d.])(\d{1,3}(?:\.\d{1,2})?)"
RE_PCT = re.compile(NUM + r"\s*(?:%|퍼|프로|펀|프(?![로레]))")
RE_DEC = re.compile(r"(?<![\d.])(\d\.\d{1,2})(?![\d.]|\s*(?:억|만|배|초|분|시|일|x|k|%|퍼|프|펀|등|위|트|천))")
RE_EOK = re.compile(NUM + r"\s*억")
RE_BARE = re.compile(r"(?<![\d.])([1-3]\d\d|400)(?:\.(\d))?(?![\d.]|\s*(?:억|만|천|원|등|위|시|분|일|명|개|번|%|퍼|프|펀|트|연|챠|뽑|레벨|렙|lv|k|초|층|회|판|장|스|패))")
RE_ARROW = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s*억?\s*(?:->|→|>|에서|=>|~>)\s*(\d{1,3}(?:\.\d{1,2})?)")
RE_TIME = re.compile(
    r"(\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?|\d{1,2}:\d\d|오전|오후|새벽|아침|점심|저녁|밤|지금|현재|방금|아까|어제|오늘|첫날|둘째\s*날|마감|막날|"
    r"[월화수목금토일]요일|\d\s*일\s*차)")
RE_SPLIT = re.compile(r"(?<=[?!？/])|(?<!\d)\.(?!\d)|\n")

PREV = re.compile(r"지난|저번|직전|전\s*시즌|전시즌|예전|작년|옛날|전번|S(?!41)\d\d|시즌\s*(?!41)\d\d|[가-힣]+때(?:는|에|도)?\s")
PRED = re.compile(r"예상|듯|거\s*같|것\s*같|려나|겠|일거|일걸|봐야|될까|내외|아닐까|가능할|일지|일까|나올|끊기|찍을|갈거|갈듯|않을까|"
                  r"대략|대충|대강|보인다|보이는데|ㄹ걸|할걸|목표|예측|못\s*넘|안\s*넘|안\s*될|설이|의견")
CALC = re.compile(r"계산기|모의전|렛도|렛츠도로|렛츠\s*도로")
# 「6시 27분」처럼 시 뒤의 분은 시각이라 뺀다.
DELTA_PRE = re.compile(r"(?:(?<![\d시])(?<!시\s)\d+\s*분|(?:한|두|세|몇)\s*분|(?:\d+|한|두|세|몇)\s*(?:시간|일)|하루|이틀|사흘|사이|그새)\s*(?:에|만에|동안|사이에)?\s*$")
DELTA_POST = re.compile(r"^\s*(?:%|퍼|프로|펀|프)?\s*(?:씩|차이|차|올라|떨어|밀려|빠졌|늘|줄|찼|가\s*찼)")
ANNIV = re.compile(r"^\s*(?:유입|뉴비|주년|복귀|버전)")
DONE = re.compile(r"찍었|찍힘|찍히|찍음|나옴|나왔|나오는데|나오네|기갱|됐|였|했|햇|쳤|쳐봤|넘노|넘음|떴|마무리|완$|인데|인대|임$|이네|현재|지금")
QEND = re.compile(r"\?|？|냐|나요|까요|인가|ㄴ가|되나|되냐|있삼|니\s*$|가능\s*$")
SELF_Q = re.compile(r"(?<![\d.])\d\.\d{1,2}\s*(?:%|퍼|프|펀|$|\s)|지금|현재|(?:^|\s)(?:나|내|난|본인)(?:\s|가|는|도)|억인데|억인대|억이고|억임")
ME = re.compile(r"(?:^|\s)(?:나|내|난|본인)(?:\s|가|는|도)|턱걸이|생존|입성|간당|딸깍|넘김|넘었|안착|찍|기갱")
RANK = re.compile(r"\d+\s*(?:등|위)(?!\s*컷)|in\s*\d+(?!\s*컷)")
CUT = re.compile(r"컷|커트")
HEARD = re.compile(r"다는데|라는데|라매|라더라|래더라|라며|던데\s*$|다더라|더라고")


def find_scores(seg):
    """(위치, 값) 목록. 화살표·에서 앞쪽 값은 버린다."""
    drop = {m.start(1) for m in RE_ARROW.finditer(seg)}
    out, taken = [], []
    for m in RE_EOK.finditer(seg):
        v = float(m.group(1))
        if m.start(1) in drop or v <= 0:
            continue
        out.append((m.start(1), v))
        taken.append((m.start(), m.end()))
    for m in RE_BARE.finditer(seg):
        if any(a <= m.start() < b for a, b in taken) or m.start(1) in drop:
            continue
        out.append((m.start(1), float(m.group(1) + ("." + m.group(2) if m.group(2) else ""))))
    return sorted(out)


def find_pcts(seg):
    out, taken = [], []
    for m in RE_PCT.finditer(seg):
        v = float(m.group(1))
        if 0 < v <= 100:
            out.append((m.start(1), v, m.end(1)))
            taken.append((m.start(), m.end()))
    for m in RE_DEC.finditer(seg):
        if any(a <= m.start() < b for a, b in taken):
            continue
        v = float(m.group(1))
        if 0 < v < 10 and not ANNIV.search(seg[m.end():]):
            out.append((m.start(1), v, m.end(1)))
    return sorted(out)


def is_delta(seg, pos, end):
    return bool(DELTA_PRE.search(seg[:pos]) or DELTA_POST.search(seg[end:]))


def classify(seg):
    s = seg.strip()
    q = bool(QEND.search(s))
    if PREV.search(s):
        return "prev_season", q
    if RANK.search(s) and not ME.search(s):
        return "other", q
    if CUT.search(s):
        if q:
            return "question", q
        if PRED.search(s):
            return "predict", q
        if ME.search(s):
            return "self", q
        return "cutoff", q
    if HEARD.search(s):
        return "cutoff", q
    if CALC.search(s) and not re.search(r"실전|실제|본게임|현시각|현재|지금", s):
        return "predict", q
    if q:
        return ("self", True) if SELF_Q.search(s) else ("question", True)
    if PRED.search(s) and not DONE.search(s):
        return "predict", q
    return "self", q


def time_before(seg, pos, lo):
    best = None
    for m in RE_TIME.finditer(seg):
        if lo <= m.start() < pos:
            best = re.sub(r"\s", "", m.group(1))
    return best


def extract_field(no, src, text):
    rows = []
    if not text:
        return rows
    segs = [s for s in RE_SPLIT.split(text) if s and s.strip()]
    for si, seg in enumerate(segs):
        pcts, scores = find_pcts(seg), find_scores(seg)
        if not pcts and not scores:
            continue
        kind, q = classify(seg)
        if pcts and len(scores) == len(pcts):
            pairs = list(zip(pcts, scores))
        elif pcts and len(scores) == 1:
            pairs = [(p, scores[0]) for p in pcts]
        else:
            pairs = [(p, None) for p in pcts] + [(None, s) for s in scores]
        lo = 0
        for p, s in pairs:
            t = time_before(seg, p[0], lo) if p else time_before(seg, s[0], 0)
            if p:
                lo = p[0]
            k = "delta" if p and is_delta(seg, p[0], p[2]) else kind
            rows.append({"no": no, "src": src, "seg": si, "kind": k, "q": q,
                         "pct": p[1] if p else None, "score_eok": s[1] if s else None,
                         "time": t, "txt": seg.strip()[:80]})
    return rows


def main():
    posts = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    n = 0
    with open(sys.argv[2], "w", encoding="utf-8") as out:
        for p in posts:
            rows = extract_field(p["no"], "title", p["title"] or "") + extract_field(p["no"], "body", p["body"] or "")
            own = {r["score_eok"] for r in rows if r["kind"] == "self" and r["score_eok"] is not None}
            for r in rows:
                r["score_from"] = "seg" if r["score_eok"] is not None else None
                if r["kind"] == "self" and r["pct"] is not None and r["score_eok"] is None and len(own) == 1:
                    r["score_eok"], r["score_from"] = next(iter(own)), "post"
                r["core"] = r["pct"] is not None and (r["pct"] != int(r["pct"]) or r["score_eok"] is not None)
                r["post_t"] = p["t"]
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"posts {len(posts)} rows {n}")


if __name__ == "__main__":
    main()
