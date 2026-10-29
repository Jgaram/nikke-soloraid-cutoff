"""관측 시각 등급 (솔로레이드 후처리 1.5). 모델 0, 요청 0. 원자료(state/의 기존 파일)는 읽기만 한다.

python3 -I tools/solo_raid/post_time.py STATE [--posts FILE.jsonl ...] [--snap DIR ...] [--runs DIR] [--summary OUT.json]

게시 시각을 촬영 시각으로 단정하지 않는다. 모든 줄에 시각 구간 [t_lo, t_hi]과 점 t, 그리고 t가 무엇인지(`t_kind`)를 붙인다.
제외 판단은 하지 않는다 (시즌 밖·지난 시즌 줄도 시각만 붙인다).

등급 (관측 줄, obs.jsonl 한 줄당 한 줄)
- A 기계 시각: 첨부 이름·epoch(`shot_t`·`shot_src`). 같은 파일(route.jsonl sha1)의 다른 사본에 이름 시각이 있으면 그것도 쓴다.
  t = t_lo = t_hi = 찍은 시각, t_kind "shot".
  강등(→ C, 이름 시각은 상한 t_hi로만): 이름이 남의 화면을 다시 찍은 것·옮긴 것(브라우저·갤러리·카카오톡 등, `SECOND`).
  이름 종류는 manifest fname, STATE/../runs/*/fetchlog.tsv 의 f_no, --snap 글 스냅숏의 첨부 이름으로 찾는다.
  표시만(등급 유지): 촬영이 게시보다 늦음(≤5분, 시계 차) `shot_after_post`, 게시까지 24시간 넘게 걸림 `late>24h`,
  같은 파일 사본끼리 이름 시각이 다름 `sha1_conflict`(가장 이른 것을 씀), 종료 후 화면인데 종료 전 시각 `end_screen_before_end`.
- B 사람·화면이 말한 시각:
  - 갱신 타이머 `갱신 시간 N시간 M분 남음`(판독 메모·OCR): 다음 05:00 − 남은 시간, 게시 직전 가장 가까운 것. [t−1분, t].
  - 화면 안 시계 `clock`(지금 판독 칸, S41은 모두 비어 있음): 게시 시각 기준 가장 가까운 과거.
  - 글 주장 이음: 같은 글의 글 주장(아래 B로 풀린 것, kind가 predict·prev_season·delta 아님)의 %가 이 사진의 % 줄과 같고,
    그 % 값이 그 글에서 사진 하나(같은 sha1은 하나로 셈)에만 있으면 그 사진의 모든 줄에 주장 시각을 붙인다.
    % 같음: 주장 소수 자리 d로 반올림 또는 버림이 같다. 정수 %는 사진 값이 정확히 같을 때만. 둘 다 점수가 있으면 1억 안.
- C 게시 시각만: t_hi = min(게시, 같은 파일이 더 먼저 올라온 게시, 강등된 이름 시각), t = t_hi, t_kind "upper".
  t_lo = 챌린지 시작(c_start)(t_hi가 그 전이면 비움). 종료 후 화면(`season_end`)이고 t_hi ≥ 종료면 t_lo = 종료.
  파일 날짜만 있으면(시각 없음) 그 날 00:00으로 올린다. `file_date_out`(시즌 전 파일)은 t_lo를 비우고 표시한다.

글 주장 (claims.jsonl 한 줄당 한 줄): `time` 칸(text_extract.py가 % 바로 앞에서 뽑은 시각 표현)을 원문 `txt`에서 다시 찾아 푼다.
- 지금·현재·방금 → [게시−30분, 게시] t=게시 (폭은 근거 없이 정함, 기준점 grade.py와 같음). 아까 → [게시−6시간, 게시].
- 오늘 → [그날 00:00(게시가 05시 전이면 전날 05:00), 게시]. 어제 → [전날 00:00(게시가 06시 전이면 6시간 더 앞), min(게시, 그날 06:00)].
- 요일 → 게시일 이전 가장 가까운 그 요일 하루. 새벽·아침·점심·오후·저녁·밤·오전 → 가장 가까운 과거의 그 시간대.
- 시각 `N시`·`N시M분`·`HH:MM`·`N시반`: 앞(같은 문장, 앞 시각 표현 뒤)의 날짜(`9.27`·`27일`)·어제·오늘·요일·오전/오후 등으로 후보를 줄이고,
  1~12시는 N과 N+12 둘 다 후보, 24~29시는 다음날 0~5시. 게시+5분 이전, 게시−4일 이후 후보 중 **가장 늦은 것**(가장 가까운 과거).
  폭: 분까지 [t, t+1분], 반 [t−15분, t+15분], 시만 [t−30분, t+60분], `쯤·넘어서` [t−60분, t+60분]. 모두 게시에서 자른다.
  12시간 다른 후보가 있으면 basis에 `alt=`. 날짜는 시각 바로 앞에 붙고 시즌 앞뒤 1주 안일 때만 쓴다(`5.1퍼 18시`의 5.1은 %).
- `N시간 전` → 게시 − N시간 ±30분. `N시간`(남음 등) → C.
- 요일·시간대·시각은 다음이면 C: 바로 뒤 `까지`·글에 `자정`(마감), 시각 뒤 ~ % 사이 조건(`이면`·`어야`·`퍼면`), 앞으로의 일(`겠`·`봐야`·`목표`·
  앞에 `내일`), 앞에 `마감·종료·갱신`(종료 기준 상대 시각), 풀린 시각이 c_start 전인데 글은 챌린지 중(`pre_c_start`).
- 첫날·막날·마감·N일차 → C (대개 친 때나 단계를 말하고 % 본 때가 아님). 풀리지 않으면 C. txt가 잘려 시각이 안 보이면 문맥 없이 푼다(`ctx_cut`).
- 시각 없는 주장은 C: t_hi = 게시, t_lo = c_start (게시가 그 전이면 비움).

열: t·t_lo·t_hi(KST 'YYYY-MM-DD HH:MM:SS'), h·h_lo·h_hi = c_start 후 시간(시), day = 05:00 갱신 기준 챌린지 일차
(1일차 = c_start ~ 다음 05:00, 정수). t가 종료(`end`) 뒤면 "post", c_start 전이면 "pre". day는 t 기준이다 (C는 상한의 일차).
day_lo = t_lo의 일차. 「확실히 종료 후」는 day_lo == "post"다 (C는 게시가 종료 뒤여도 day_lo가 1일 수 있다).
t_kind: "shot"(기계 시각 한 점) · "stated"(말한 시각, 구간 안의 점) · "upper"(t = t_hi, 실제는 그 전 어딘가).
basis: 등급 근거와 표시를 `;`로 잇는다.

top7 시각 구간 (별도 열, 2026-10-16. 등급·t_lo·t_hi·h_lo·h_hi는 바꾸지 않는다 — 분석에서 넣고 빼 본다)
- 같은 서버의 k위 점수는 시간에 따라 줄지 않는다. 그래서 B·C 사진의 1~7위 점수를 같은 서버 A 사진들(촬영 시각 h)의 1~7위와 견주면
  촬영 시각 구간이 나온다: A 점수 > 이 사진 점수면 촬영은 그 A 전, < 면 후. 어긋남이 가장 적은 구간을 고른다 (top7_bracket).
  S41 확인(explore_surface.py, 2.3): A 사진 하나씩 빼고 구한 구간이 촬영 시각을 포함 한섭 83/83·일섭 18/18, 한섭 B·C 70장 중 게시 시각과 모순 0.
- 사진 조건 (explore_surface.top7_snapshots와 같다): 시즌 in, 사진의 어느 줄도 post_exclude의 exclude 없음, 본인 줄이 아닌
  1~7위 점수 2개 이상. 서버는 server.jsonl만 쓴다 (사진 첫 줄의 server; 점수·%로 서버를 정하지 않는다).
- 붙이는 조건: B·C 사진, 같은 서버 A 사진이 있음, 어긋남 0, 그리고 줄의 [h_lo, h_hi]와 겹침 (비면 = 게시 시각·종료 화면과 모순 → 안 붙임).
  h_lo_top7·h_hi_top7 = 그 겹침 (h 단위). top7 = 근거 문자열 "{서버} refs={A 사진 수} raw=[lo,hi]" (열린 끝은 None).
  A 줄·조건이 안 맞는 줄은 셋 다 null. server.jsonl·post_exclude.jsonl이 없거나 줄이 obs와 안 맞으면 모두 null로 두고 알린다.

2026-10-29 WP2 고침 (C 줄만 올린다 — 이미 A·B인 줄은 그대로. 근거는 data/solo-raid/analysis/r7/wp1/1g.md, 가늠 코드는 lab7/err_1g.py)
- 갱신 타이머 느슨한 꼴 (C → B): 지금 `TIMER`가 못 읽는 「갱신 시간 → N시간 M분 남음」(화살표·기호 사이, TIMER_LOOSE)과, OCR에서 「갱신」 항목
  다음 항목이 「N시간 M분 남」으로 갈라진 꼴(TIMER_REM). 판독 메모·reads·ocr·ocr2 모두. 지금 규칙으로 읽히는 사진은 지금 것 그대로.
  남은 시간은 24시간 미만만. basis `B:timer_loose(ocr) '…'`. 시각은 지금 타이머와 같게 풀고 [t−1분, t].
- 이름 꼴 확장 (C → A): `name_time`이 못 읽는 이름 — 13자리·10자리 epoch(2026-05 앞, 1[67]로 시작)·공백·마침표·세미콜론 날짜시각(X_LOOSE)·맥 한글 스크린샷(X_MAC)·
  YYMMDDHHMM(SS)(X_YY). 우선순위는 err_1g parse_name과 같다. `gov-`로 시작하는 이름(디시 앱 업로드 이름)은 쓰지 않는다. 기존 `name_time`이 읽는 이름과
  obs shot_t가 있는 줄은 바꾸지 않고, shot_t가 비어 C가 될 줄에만 쓴다. 출처는 그 사진의 manifest fname. 남의 화면(SECOND) 이름이면 지금처럼 상한으로만.
  유효 창은 게시−4일 ≤ t ≤ 게시+5분. basis `A:name_ext(manifest)`.
- 첨부 짝 (C → A): manifest에 fname이 없는 사진은 글의 첨부 이름 목록(STATE/posts_min.jsonl의 files, --posts 파일의 files)에서 이름 시각(기존+확장 꼴)을 찾는다.
  (a) extpair: 확장자 묶음(jpeg=jpg) 안에서 첨부 수와 그 글의 사진 수(manifest 같은 글 번호·확장자)가 같으면 차례대로 짝지은 이름 하나의 시각, basis `A:name_pair(extpair)`.
  (b) tight: 첨부 이름 수 ≥ 그 글의 사진 수이고(이름 없는 사진이 남으면 안 씀 — WP2 세션 판단) 글 첨부의 유효 이름 시각이 모두 2분 안에 몰려 있으면 가장 이른 것, basis `A:name_pair(tight)`. 2~10분 묶음은 올리지 않고 C로 두고 센다(`tight_2_10`).
  애매하면 올리지 않는다. 새 A에도 shot_after_post·late>24h·edit_name 표시 규칙이 그대로 적용된다.
- 안전장치 (순위표 구간): 위 셋으로 새로 올릴 후보의 시각을 **기존 규칙만으로 정한 A 사진들**을 참조로 한 top7 구간(top7_bracket의 raw 구간, 열린 끝 허용)과 견준다.
  구간이 있고 후보 시각이 구간 밖(±0.1시간)이면 올리지 않고 C로 두며 basis에 `cand_out_top7:{근거}`를 적는다. 구간이 없으면 올린다.
  그다음 최종 등급으로 top7 열을 다시 계산한다 (순환 없이 두 단계: 기존 규칙 → 후보 점검·올림 → top7 열).
- 후보 이름이 SECOND인 manifest 이름이면 C로 두되 t_hi를 그 이름 시각으로 낮출 수 있다 (`secondhand:…` 표시, 지금 obs shot_t 경로와 같다).
- 요약(표준 출력·--summary)에 `wp2` 항목: 새 근거별 줄 수, 안전장치로 막힌 수, tight 2~10분이라 안 올린 수.

출력: STATE/post_time.jsonl {oi, site, img, row, grade, t, t_lo, t_hi, t_kind, h, h_lo, h_hi, day, day_lo, basis,
h_lo_top7, h_hi_top7, top7},
STATE/post_time_claims.jsonl {i, grade, t, t_lo, t_hi, t_kind, h, h_lo, h_hi, day, day_lo, basis}. 표준 출력에 요약(시즌 안 % 관측 기준 등급별 줄 수, A 게시 지연 분위수).
--summary로 요약 JSON을 따로 쓸 수 있다. 시즌 값은 STATE/season.json(c_start·end·start)에서 읽는다.
"""
import argparse
import ast
import collections
import datetime as dt
import glob
import json
import os
import re
import urllib.parse

F = "%Y-%m-%d %H:%M:%S"
P1 = re.compile(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ _\-T]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})")   # merge_obs.py와 같음
P2 = re.compile(r"(?<!\d)(17[89]\d{10})(?!\d)")
KST = dt.timezone(dt.timedelta(hours=9))
# 남의 화면을 다시 찍었거나 옮긴 이름: 그 시각은 관측의 상한일 뿐이다 (com.proximabeta는 게임 자체)
SECOND = re.compile(r"browser|chrome|gallery|kakao|discord|twitter|instagram|naver|whale|line_|telegram|"
                    r"com\.(?!proximabeta)", re.I)
# WP2 (2026-10-29): 느슨한 갱신 타이머 · 이름 꼴 확장 (lab7/err_1g.py와 같은 규칙)
TIMER_LOOSE = re.compile(r"갱신\s*시간?\s*[→>\-–:~=\s]*\s*(?:(\d+)\s*시간)?\s*(?:(\d+)\s*분)?\s*남")
TIMER_REM = re.compile(r"^\s*(?:(\d+)\s*시간)?\s*(?:(\d+)\s*분)?\s*남")
X_EPOCH13 = re.compile(r"(?<!\d)(1[67]\d{11})(?!\d)")
X_EPOCH10 = re.compile(r"(?<!\d)(1[67]\d{8})(?!\d)")
X_LOOSE = re.compile(r"(20\d{2})[-_. ]+(\d{2})[-_. ]+(\d{2})[ _\-T.]+(\d{2})[-_.:; ]+(\d{2})[-_.:; ]+(\d{2})")
X_MAC = re.compile(r"(20\d{2})-(\d{2})-(\d{2})\s*(오전|오후)\s*(\d{1,2})_(\d{2})_(\d{2})")
X_YY = re.compile(r"(?<!\d)(2[3-6])(\d{2})(\d{2})[_\-]?(\d{2})(\d{2})(\d{2})?(?!\d)")
TIGHT_MAX = dt.timedelta(minutes=2)        # 첨부 짝 tight: 유효 이름 시각이 이 안에 몰려야 올린다
TIGHT_LOOSE_MAX = dt.timedelta(minutes=10)  # 2~10분 묶음은 올리지 않고 센다
TOP7_MARGIN_H = 0.1                         # 안전장치 여유 (시간)
TIMER = re.compile(r"갱신\s*시간\s*(?:(\d+)\s*시간)?\s*(?:(\d+)\s*분)?\s*남")
RE_TIME = re.compile(   # text_extract.py와 같음
    r"(\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?|\d{1,2}:\d\d|오전|오후|새벽|아침|점심|저녁|밤|지금|현재|방금|아까|어제|오늘|첫날|둘째\s*날|마감|막날|"
    r"[월화수목금토일]요일|\d\s*일\s*차)")
CLOCK = re.compile(r"(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?|(\d{1,2}):(\d\d)")
WD = {c: i for i, c in enumerate("월화수목금토일")}
PERIOD = {"새벽": (0, 6), "아침": (6, 10), "오전": (0, 12), "점심": (11, 14), "오후": (12, 24), "저녁": (17, 21), "밤": (20, 27)}
AM = ("오전", "새벽", "아침")
PM = ("오후", "저녁", "밤")
COND = re.compile(r"이?어야|이면|라면|되면|이내면|안이면|안쪽이면|미만이면|(?:퍼|%|프로|억)\s*(?:이)?면(?![가-힣])")
FUTURE = re.compile(r"겠|봐야|해야지|려고|예정|목표|할\s*(?:거|듯|게)")
NOW_W = dt.timedelta(minutes=30)


def jl(p):
    out = []
    for f in sorted(glob.glob(p)):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                out.append(json.loads(line))
    return out


def ts(s):
    return dt.datetime.strptime(s, F) if s else None


def fmt(t):
    return t.strftime(F) if t else None


def name_time(name):
    m = P1.search(name or "")
    if m:
        try:
            return dt.datetime(*map(int, m.groups()))
        except ValueError:
            pass
    m = P2.search(name or "")
    if m:
        return dt.datetime.fromtimestamp(int(m.group(1)) / 1000, KST).replace(tzinfo=None)
    return None


def _dtc(*a):
    try:
        return dt.datetime(*a)
    except ValueError:
        return None


def parse_name(name):
    """이름 → [(시각, 꼴)]. 꼴: std(`name_time`이 읽는 꼴) · ep13·ep10·loose·mac·yy(2026-10-29 확장). err_1g.parse_name과 같다.
    `gov-`로 시작하는 이름(디시 앱 업로드 이름, 업로드 시각이라 촬영이 아님)은 아무것도 읽지 않는다."""
    out = []
    if not name or name.startswith("gov-"):
        return out
    m = P1.search(name)
    if m:
        t = _dtc(*map(int, m.groups()))
        if t:
            out.append((t, "std"))
    m = P2.search(name)
    if m:
        out.append((dt.datetime.fromtimestamp(int(m.group(1)) / 1000, KST).replace(tzinfo=None), "std"))
    if out:
        return out
    m = X_EPOCH13.search(name)
    if m:
        out.append((dt.datetime.fromtimestamp(int(m.group(1)) / 1000, KST).replace(tzinfo=None), "ep13"))
    m = X_EPOCH10.search(name)
    if m and not out:
        out.append((dt.datetime.fromtimestamp(int(m.group(1)), KST).replace(tzinfo=None), "ep10"))
    m = X_MAC.search(name)
    if m:
        y, mo, d, ap, hh, mi, ss = m.groups()
        t = _dtc(int(y), int(mo), int(d), int(hh) % 12 + (12 if ap == "오후" else 0), int(mi), int(ss))
        if t:
            out.append((t, "mac"))
    m = X_LOOSE.search(name)
    if m and not any(k == "mac" for _, k in out):
        t = _dtc(*map(int, m.groups()))
        if t:
            out.append((t, "loose"))
    if not out:
        m = X_YY.search(name)
        if m:
            g = m.groups()
            t = _dtc(2000 + int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5] or 0))
            if t:
                out.append((t, "yy"))
    return out


def valid_times(names, pt, skip_second=True):
    """이름들 → 유효 창(게시−4일 ≤ t ≤ 게시+5분, shot_time과 같음) 안의 [(t, 꼴, 이름)]. 남의 화면 이름(SECOND)은 기본으로 뺀다."""
    res = []
    if pt is None:
        return res
    for n in names:
        if not n or (skip_second and SECOND.search(n)):
            continue
        for t, k in parse_name(n):
            if pt - dt.timedelta(days=4) <= t <= pt + dt.timedelta(minutes=5):
                res.append((t, k, n))
    return res


def timer_loose_find(text, nxt=None):
    """느슨한 갱신 타이머 (C → B 후보). 반환 (원문 조각, 남은 분) 또는 None. 24시간 미만만.
    text 한 줄에서 TIMER_LOOSE, 없으면 text에 「갱신」이 있고 다음 항목(nxt)이 「N시간 M분 남」인 갈라진 꼴(TIMER_REM)."""
    m = TIMER_LOOSE.search(text or "")
    frag = m.group(0) if m else None
    if not (m and (m.group(1) or m.group(2))) and nxt is not None and "갱신" in (text or ""):
        m = TIMER_REM.search(nxt)
        frag = (text.strip()[-12:] + " | " + m.group(0).strip()) if m else None
    if m and (m.group(1) or m.group(2)):
        mins = int(m.group(1) or 0) * 60 + int(m.group(2) or 0)
        if mins < 24 * 60:
            return frag.replace(";", ",").replace("'", ""), mins
    return None


def ext_class(e):
    return {"jpeg": "jpg"}.get(e, e)


class Season:
    def __init__(self, s):
        # c_start 없으면 다른 도구와 같게 start 날짜 + 2일 05:00 (2026-10-16 고침, 전에는 start 날짜 05:00이라 어긋났다)
        self.c0 = ts(s.get("c_start") or ((dt.date.fromisoformat(s["start"][:10]) + dt.timedelta(days=2)).isoformat() + " 05:00:00"))
        self.end = ts(s["end"])
        self.start = ts(s.get("start") or s.get("e_start"))

    def h(self, t):
        return None if t is None else round((t - self.c0).total_seconds() / 3600, 3)

    def day(self, t):
        if t is None:
            return None
        if t > self.end:
            return "post"
        if t < self.c0:
            return "pre"
        return int((t - self.c0).total_seconds() // 86400) + 1


def pct_match(claim, obs):
    """주장 %(소수 자리 d)가 사진 %와 같은가: 반올림 또는 버림이 같다. 정수 주장은 정확히 같을 때만."""
    if claim is None or obs is None:
        return False
    s = repr(float(claim))
    d = len(s.split(".")[1].rstrip("0")) if "." in s else 0
    if d == 0:
        return abs(obs - claim) < 1e-9
    u = 10 ** -d
    return abs(round(obs, d) - claim) < u / 2 or (0 <= obs - claim < u - 1e-9)


# ---------- 글 주장 시각 풀기 ----------

def day_start(pt):
    d0 = dt.datetime.combine(pt.date(), dt.time())
    return d0 - dt.timedelta(hours=19) if pt.hour < 5 else d0   # 05시 전 '오늘'은 전날 05:00부터


def context(txt, a, b, se=None):
    """시각 표현 앞(같은 문장, 앞 시각 숫자 뒤)에서 날짜 제약·오전오후를 찾는다."""
    pre = txt[a:b]
    days, ampm, notes = None, None, []
    # 날짜는 시각 바로 앞에 붙은 것만 (`9.27 13:40`·`27일 1시`). 앞 %의 소수(`5.1퍼 18시`)를 날짜로 읽지 않는다
    md = re.search(r"(?<![\d.])(\d{1,2})\s*[./]\s*(\d{1,2})\s*$", pre)
    dd = re.search(r"(?<![\d.])(\d{1,2})\s*일\s*$", pre)
    if md and se is not None:   # 시즌 앞뒤 1주 안의 날짜만 (`7.09 8시`의 7.09는 %다)
        try:
            d = dt.datetime(se.c0.year, int(md.group(1)), int(md.group(2)))
            if not (se.c0 - dt.timedelta(days=7) <= d <= se.end + dt.timedelta(days=7)):
                md = None
        except ValueError:
            md = None
    if md:
        days = ("md", int(md.group(1)), int(md.group(2)))
    elif dd:
        days = ("d", int(dd.group(1)))
    wm = re.search(r"([월화수목금토일])(?:요일|욜)", pre)
    if wm:
        days = ("wd", WD[wm.group(1)])
    if "어제" in pre or "어젯" in pre:
        days = ("rel", -1)
    elif "그저께" in pre or "엊그제" in pre:
        days = ("rel", -2)
    elif "오늘" in pre:
        days = ("rel", 0)
    for k in AM + PM + ("점심",):
        if k in pre:
            ampm = k
    return days, ampm, notes


def allowed_dates(pt, days):
    base = pt.date()
    if days is None:
        return [base - dt.timedelta(days=k) for k in range(0, 5)]
    kind = days[0]
    if kind == "rel":
        d = base + dt.timedelta(days=days[1])
        if days[1] == 0 and pt.hour < 5:
            return [d, d - dt.timedelta(days=1)]
        return [d]
    if kind == "wd":
        return [base - dt.timedelta(days=k) for k in range(0, 7) if (base - dt.timedelta(days=k)).weekday() == days[1]]
    if kind == "md":
        try:
            return [dt.date(base.year, days[1], days[2])]
        except ValueError:
            return []
    if kind == "d":
        out = []
        for k in range(0, 5):
            d = base - dt.timedelta(days=k)
            if d.day == days[1]:
                out.append(d)
        return out
    return []


def resolve_clock(pt, hh, mm, ampm, days):
    """[(t, alt)] 가장 늦은 후보와 12시간 다른 후보."""
    if hh > 29 or (mm is not None and mm > 59):
        return None, None
    if hh > 12 and ampm is not None:
        hours = [hh]   # "오후 18시"처럼 24시간 표기면 그대로
    elif ampm in AM:
        hours = [hh % 12]
    elif ampm == "오후":
        hours = [hh % 12 + 12]
    elif ampm in ("저녁", "밤"):
        hours = [hh + 24] if hh <= 4 else ([24] if hh == 12 else [hh % 12 + 12])   # 밤 12시 = 자정, 밤 2시 = 다음날 02시
    elif ampm == "점심":
        hours = [hh if hh >= 11 else hh + 12]
    elif 1 <= hh <= 12:
        hours = [hh, hh + 12]
    else:
        hours = [hh]
    cands = []
    for d in allowed_dates(pt, days):
        for h in hours:
            t = dt.datetime.combine(d, dt.time()) + dt.timedelta(hours=h, minutes=mm or 0)
            if pt - dt.timedelta(days=4) <= t <= pt + dt.timedelta(minutes=5):
                cands.append(t)
    if not cands:
        return None, None
    cands.sort()
    t = cands[-1]
    alt = [c for c in cands[:-1] if dt.timedelta(hours=11) <= t - c <= dt.timedelta(hours=13)]
    return t, (alt[-1] if alt and len(hours) > 1 else None)


def resolve_claim(c, se):
    """(grade, t, lo, hi, t_kind, basis)"""
    pt = ts(c.get("post_t"))
    T = c.get("time")
    if pt is None:
        return "C", None, None, None, None, "C:no_post_t"

    def c_default(why):
        lo = se.c0 if pt >= se.c0 else None
        return "C", pt, lo, pt, "upper", "C:post" + (";" + why if why else "")
    if not T:
        return c_default("")
    txt = c.get("txt") or ""
    ms = [m for m in RE_TIME.finditer(txt) if re.sub(r"\s", "", m.group(1)) == T]
    m, pend = None, None
    if ms:   # 주장 %가 있는 위치 바로 앞의 것을 고른다
        if c.get("pct") is not None:
            v = c["pct"]
            num = (str(int(v)) if v == int(v) else repr(v).rstrip("0"))
            unit = r"(?![\d])" if "." in num else r"\s*(?:%|퍼|프|펀)"
            for x in re.finditer(r"(?<![\d.])" + re.escape(num) + unit, txt):
                before = [y for y in ms if y.end() <= x.start()]
                if before:
                    m, pend = before[-1], x.end()
                    break
        m = m or ms[-1]
    tag = f"'{T}'"
    if m is None:   # txt가 80자에서 잘려 시각이 안 보인다. 앞뒤 문맥 없이 푼다
        tag += ";ctx_cut"
        after, days, ampm, tail = "", None, None, ""
    else:
        after = txt[m.end():m.end() + 12]
        tail = txt[m.end():(pend or m.end() + 14) + 6]
        prev = [x for x in RE_TIME.finditer(txt) if x.end() <= m.start() and CLOCK.match(x.group(1))]
        a = max(prev[-1].end() if prev else 0, max(txt.rfind(ch, 0, m.start()) for ch in "?!\n") + 1)
        days, ampm, _ = context(txt, max(a, m.start() - 16), m.start(), se)
    if T in ("지금", "현재", "방금"):
        return "B", pt, pt - NOW_W, pt, "stated", f"B:now {tag}"
    if T == "아까":
        return "B", pt, pt - dt.timedelta(hours=6), pt, "upper", f"B:earlier {tag}"
    if T == "오늘":
        return "B", pt, day_start(pt), pt, "upper", f"B:today {tag}"
    if T == "어제":
        d0 = dt.datetime.combine(pt.date(), dt.time())
        lo = d0 - dt.timedelta(days=1) - (dt.timedelta(hours=6) if pt.hour < 6 else dt.timedelta())
        hi = min(pt, d0 + dt.timedelta(hours=6))
        return "B", hi, lo, hi, "upper", f"B:yesterday {tag}"
    if T in ("첫날", "막날", "마감") or re.fullmatch(r"\d일차|둘째날", T):
        return c_default(f"phase:{T}")
    # 여기부터(요일·시간대·시각)는 마감·가정·앞으로의 일을 거른다
    if re.match(r"\s*(?:쯤|정도|경)?\s*(?:에\s*)?까지", after) or "자정" in txt:
        return c_default(f"deadline:{T}")
    if COND.search(tail):
        return c_default(f"cond:{T}")
    if FUTURE.search(tail) or (m and re.search(r"내일|낼|모레", txt[max(0, m.start() - 8):m.start()])):
        return c_default(f"future:{T}")
    if re.search(r"마감|종료|끝나기|갱신", txt[max(0, m.start() - 8):m.start()] if m else ""):
        return c_default(f"before_end_rel:{T}")
    if re.fullmatch(r"[월화수목금토일]요일", T):
        ds = allowed_dates(pt, ("wd", WD[T[0]]))
        if not ds:
            return c_default(f"unresolved:{T}")
        lo = dt.datetime.combine(ds[0], dt.time())
        hi = min(pt, lo + dt.timedelta(days=1))
        return "B", hi, lo, hi, "upper", f"B:weekday {tag}"
    if T in PERIOD:
        a0, a1 = PERIOD[T]
        for d in allowed_dates(pt, days):
            lo = dt.datetime.combine(d, dt.time()) + dt.timedelta(hours=a0)
            if lo <= pt:
                hi = min(pt, dt.datetime.combine(d, dt.time()) + dt.timedelta(hours=a1))
                return "B", hi, lo, hi, "upper", f"B:period {tag}"
        return c_default(f"unresolved:{T}")
    cm = CLOCK.fullmatch(T)
    if not cm:
        return c_default(f"unparsed:{T}")
    if cm.group(1) is not None:
        hh, mm = int(cm.group(1)), (int(cm.group(2)) if cm.group(2) else None)
    else:
        hh, mm = int(cm.group(3)), int(cm.group(4))
    if after.startswith("간"):
        r = re.match(r"간\s*(?:정도\s*)?전", after)
        if r and mm is None:
            t = pt - dt.timedelta(hours=hh)
            return "B", t, t - NOW_W, min(pt, t + NOW_W), "stated", f"B:hours_ago {tag}간전"
        return c_default(f"duration:{T}간")
    half = mm is None and re.match(r"\s*반", after)
    if half:
        mm = 30
    t, alt = resolve_clock(pt, hh, mm, ampm, days)
    if t is None:
        return c_default(f"unresolved:{T}")
    if half:
        lo, hi = t - dt.timedelta(minutes=15), t + dt.timedelta(minutes=15)
    elif mm is not None:
        lo, hi = t, t + dt.timedelta(minutes=1)
    elif re.match(r"\s*(?:쯤|경|정도|넘)", after):
        lo, hi = t - dt.timedelta(minutes=60), t + dt.timedelta(minutes=60)
    else:
        lo, hi = t - NOW_W, t + dt.timedelta(minutes=60)
    hi = min(hi, pt)
    lo = min(lo, hi)
    t = min(t, pt)
    ctx = "".join(x for x in [ampm and f" {ampm}", days and f" {days[0]}"] if x)
    return "B", t, lo, hi, "stated", f"B:clock {tag}{ctx}" + (f";alt={fmt(alt)}" if alt else "")


def resolve_claim_checked(c, se):
    r = resolve_claim(c, se)
    pt = ts(c.get("post_t"))
    if r[0] == "B" and pt and pt >= se.c0 and r[3] < se.c0:   # 챌린지 중 글이 챌린지 전 시각을 말함: 지난 시즌·앞으로의 일
        lo = se.c0
        return "C", pt, lo, pt, "upper", f"C:post;pre_c_start:{fmt(r[1])} {r[5]}"
    return r


# ---------- 관측 줄 ----------

def timer_time(rem, pt):
    """남은 시간 → 다음 05:00 − rem 중 게시+2분 이전 가장 늦은 것."""
    best = None
    for k in range(-1, 5):
        d = pt.date() - dt.timedelta(days=k)
        t = dt.datetime.combine(d, dt.time(5)) - rem
        if t <= pt + dt.timedelta(minutes=2) and (best is None or t > best):
            best = t
    return best


def load_names(state, runs_dir, snaps):
    """이미지 id → 첨부 이름, 그리고 (글 번호, 이름 시각) → 이름, 이름 시각 → 이름(받은 원본 첨부, 글 모름)."""
    by_id, by_post, by_time = {}, collections.defaultdict(set), collections.defaultdict(set)
    for m in jl(os.path.join(state, "manifest.jsonl")):
        if m.get("fname"):
            by_id[m["id"]] = m["fname"]
    for f in glob.glob(os.path.join(runs_dir, "*", "fetchlog.tsv")):
        for line in open(f, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 4 and "f_no=" in parts[4]:
                n = urllib.parse.unquote_plus(parts[4].split("f_no=", 1)[1].split("&")[0])
                t = name_time(n)
                if t:
                    by_time[fmt(t)].add(n)
    for d in snaps:
        for f in glob.glob(os.path.join(d, "*.json")):
            p = json.load(open(f, encoding="utf-8"))
            ims = p.get("images") or []
            if isinstance(ims, str):
                try:
                    ims = ast.literal_eval(ims)
                except (ValueError, SyntaxError):
                    ims = []
            no = {"naver": "n", "blabla": "b"}.get(p.get("source"), "") + str(p.get("id"))
            for im in ims:
                n = im.get("filename") if isinstance(im, dict) else None
                t = name_time(n)
                if t:
                    by_post[(no, fmt(t))].add(n)
    return by_id, by_post, by_time


def top7_bracket(ranks, refs):
    """ranks: {k: 점수}, refs: [(h, {k: 점수})] (같은 서버 A 사진, 촬영 시각). 어긋남 최소인 촬영 시각 구간 (lo, hi, 어긋남 수).
    k위 점수는 줄지 않는다: ref 점수 > v면 촬영은 ref 전, < v면 후. 열린 끝은 ∓1e9.
    (explore_surface.py도 이 함수를 쓰고, post_outlier.py는 이 결과 열 h_lo_top7·h_hi_top7을 쓴다)"""
    hs = sorted(set(h for h, _ in refs))
    cand = [-1e9] + hs + [1e9]
    res = []
    for j in range(len(cand) - 1):
        lo, hi = cand[j], cand[j + 1]
        viol = 0
        for h, rr in refs:
            for k, v in ranks.items():
                if k in rr and ((h <= lo and rr[k] > v) or (h >= hi and rr[k] < v)):
                    viol += 1
        res.append((viol, lo, hi))
    best = min(v for v, _, _ in res)
    good = [(lo, hi) for v, lo, hi in res if v == best]
    return min(l for l, _ in good), max(h for _, h in good), best


def top7_snaps(st, obs, out_rows):
    """top7 구간의 재료: ({사진: 순위표 정보}, {서버: [A 사진 (h, 순위 점수)]}) 또는 (None, None) (server.jsonl·post_exclude.jsonl이 없거나 안 맞음).
    A 참조는 out_rows의 등급을 그대로 쓴다 (WP2: 후보를 올리기 전 단계에는 기존 규칙만의 등급을 준다)."""
    srv = jl(os.path.join(st, "server.jsonl"))
    px = jl(os.path.join(st, "post_exclude.jsonl"))
    ok = lambda rows: len(rows) == len(obs) and all((r["img"], r["row"]) == (o["img"], o["row"]) for r, o in zip(rows, obs))
    if not (ok(srv) and ok(px)):
        return None, None
    im = collections.OrderedDict()
    for k, o in enumerate(obs):
        im.setdefault(o["img"], []).append(k)
    snaps = {}
    for img, ks in im.items():
        if obs[ks[0]].get("season") != "in" or any(px[k]["exclude"] for k in ks):
            continue
        ranks = {}
        for k in ks:
            o = obs[k]
            if o.get("rank") and o.get("score") and 1 <= o["rank"] <= 7 and not o.get("self"):
                ranks[o["rank"]] = o["score"]
        if len(ranks) < 2:
            continue
        snaps[img] = dict(server=srv[ks[0]]["server"], grade=out_rows[ks[0]]["grade"], h=out_rows[ks[0]]["h"],
                          ranks=ranks, rows=ks)
    refs = collections.defaultdict(list)
    for s in snaps.values():
        if s["server"] and s["grade"] == "A" and s["h"] is not None:
            refs[s["server"]].append((s["h"], s["ranks"]))
    return snaps, refs


def top7_columns(st, obs, out_rows):
    """줄마다 (h_lo_top7, h_hi_top7, top7) 또는 None. 위 「top7 시각 구간」."""
    snaps, refs = top7_snaps(st, obs, out_rows)
    if snaps is None:
        print("top7: server.jsonl·post_exclude.jsonl이 없거나 obs와 줄이 맞지 않아 top7 열을 비운다")
        return [None] * len(obs), None
    col = [None] * len(obs)
    st_ = collections.Counter()
    for img, s in snaps.items():
        if s["grade"] not in ("B", "C") or not s["server"] or not refs.get(s["server"]):
            continue
        lo, hi, viol = top7_bracket(s["ranks"], refs[s["server"]])
        if viol:
            st_["viol"] += 1
            continue
        raw = f"[{None if lo < -1e8 else round(lo, 3)},{None if hi > 1e8 else round(hi, 3)}]"
        hit = False
        for k in s["rows"]:
            a, b = out_rows[k]["h_lo"], out_rows[k]["h_hi"]
            if a is None or b is None:
                continue
            l2, h2 = max(a, lo), min(b, hi)
            if l2 > h2:
                continue
            col[k] = (round(l2, 3), round(h2, 3), f"{s['server']} refs={len(refs[s['server']])} raw={raw}")
            hit = True
        st_["attached" if hit else "contradict"] += 1
    return col, dict(st_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--posts", nargs="*", default=[], help="글 시각을 더 줄 posts_min 형식(no·t) 파일")
    ap.add_argument("--snap", nargs="*", default=[], help="글 스냅숏 폴더(첨부 이름)")
    ap.add_argument("--runs", default=None, help="실행 기록 폴더 (기본 STATE/../runs)")
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    st = a.state
    se = Season(json.load(open(os.path.join(st, "season.json"), encoding="utf-8")))
    obs = jl(os.path.join(st, "obs.jsonl"))
    claims = jl(os.path.join(st, "claims.jsonl"))
    route = {r["id"]: r for r in jl(os.path.join(st, "route.jsonl"))}
    by_id, by_post, by_time = load_names(st, a.runs or os.path.join(st, "..", "runs"), a.snap)

    # WP2: 글의 첨부 이름 목록 (STATE/posts_min.jsonl의 files 먼저, 비었으면 --posts 파일의 files)과 글별 manifest 사진 (확장자 짝용)
    files_of = {}
    for f in [os.path.join(st, "posts_min.jsonl")] + a.posts:
        for p in jl(f):
            fl = [x if isinstance(x, str) else (x.get("name") if isinstance(x, dict) else None) for x in (p.get("files") or [])]
            if fl and not files_of.get(str(p["no"])):
                files_of[str(p["no"])] = fl
    man_of = collections.defaultdict(dict)
    for m in jl(os.path.join(st, "manifest.jsonl")):
        man_of[str(m["no"])][m["i"]] = m

    # 글 게시 시각 (같은 파일의 더 이른 게시를 찾는 데 쓴다)
    post_t = {}
    for o in obs:
        post_t.setdefault(str(o["no"]), o.get("post_t"))
    for c in claims:
        post_t.setdefault(str(c["no"]), c.get("post_t"))
    for f in [os.path.join(st, "posts_min.jsonl")] + a.posts:
        for p in jl(f):
            post_t.setdefault(str(p["no"]), p.get("t"))

    # 사진 단위 정보
    img = collections.OrderedDict()
    for o in obs:
        img.setdefault(o["img"], o)
    sha = {i: route.get(i, {}).get("sha1") for i in img}
    grp = collections.defaultdict(list)
    for i, r in route.items():
        if r.get("sha1"):
            grp[r["sha1"]].append(i)

    def name_of(i, o):
        if i in by_id:
            return by_id[i], "manifest"
        k = (str(o["no"]), o.get("shot_t"))
        if o.get("shot_t") and by_post.get(k):
            return sorted(by_post[k])[0], "snap"
        if o.get("shot_t") and by_time.get(o["shot_t"]):
            return sorted(by_time[o["shot_t"]])[0], "fetchlog"
        return None, None

    # 타이머 (판독 메모·reads 메모·OCR)
    timers = {}
    for o in obs:
        m = TIMER.search(o.get("note") or "")
        if m and (m.group(1) or m.group(2)):
            timers.setdefault(o["img"], (m.group(0), "note"))
    for r in jl(os.path.join(st, "reads.jsonl")):
        for x in r.get("reads") or []:
            m = TIMER.search(x.get("note") or "")
            if m and (m.group(1) or m.group(2)):
                timers.setdefault(r["id"], (m.group(0), "reads"))
    for f in ("ocr.jsonl", "ocr2.jsonl"):
        for r in jl(os.path.join(st, f)):
            for it in r.get("items") or []:
                m = TIMER.search(it[1] if len(it) > 1 else "")
                if m and (m.group(1) or m.group(2)):
                    timers.setdefault(r["id"], (m.group(0), "ocr"))

    # WP2: 느슨한 갱신 타이머 (지금 TIMER가 못 읽는 꼴). 지금 규칙으로 읽히는 사진에는 쓰지 않는다 (아래 후보 단계에서 거른다)
    timers_loose = {}
    for o in obs:
        r_ = timer_loose_find(o.get("note") or "")
        if r_:
            timers_loose.setdefault(o["img"], (r_[0], "note", r_[1]))
    for r in jl(os.path.join(st, "reads.jsonl")):
        for x in r.get("reads") or []:
            r_ = timer_loose_find(x.get("note") or "")
            if r_:
                timers_loose.setdefault(r["id"], (r_[0], "reads", r_[1]))
    for f in ("ocr.jsonl", "ocr2.jsonl"):
        for r in jl(os.path.join(st, f)):
            txt = [it[1] if len(it) > 1 else "" for it in (r.get("items") or [])]
            for k_, t_ in enumerate(txt):
                r_ = timer_loose_find(t_, txt[k_ + 1] if k_ + 1 < len(txt) else None)
                if r_:
                    timers_loose.setdefault(r["id"], (r_[0], "ocr", r_[1]))
                    break

    # 글 주장 시각
    cres = [resolve_claim_checked(c, se) for c in claims]

    # 사진별 시각
    info = {}
    stats = collections.Counter()
    for i, o in img.items():
        pt = ts(o.get("post_t"))
        flags = []
        members = grp.get(sha[i], [i]) if sha[i] else [i]
        g_post = [ts(post_t.get(str(route.get(x, {}).get("no", "")))) for x in members]
        g_post = [x for x in g_post if x]
        g_min = min(g_post) if g_post else pt
        # 이름 시각: 이 사진, 아니면 같은 파일 사본
        shot, src, via = ts(o.get("shot_t")), o.get("shot_src"), None
        if shot is None and len(members) > 1:
            cs = [(ts(img[x]["shot_t"]), img[x]["shot_src"], x) for x in members if x in img and img[x].get("shot_t")]
            if cs:
                shot, src, via = min(cs)
        if len(members) > 1:
            allshot = {img[x]["shot_t"] for x in members if x in img and img[x].get("shot_t")}
            if len(allshot) > 1:
                flags.append("sha1_conflict")
                shot = min(ts(x) for x in allshot)
        nm, nm_src = name_of(via or i, img.get(via or i, o))
        second = bool(nm and SECOND.search(nm))
        timer = timers.get(i)
        if not timer and len(members) > 1:
            timer = next((timers[x] for x in members if x in timers), None)
        res = None
        if shot is not None and not second:
            if pt and shot > pt:
                flags.append("shot_after_post")
            if pt and (pt - shot) > dt.timedelta(hours=24):
                flags.append("late>24h")
            if nm and nm.startswith("edit_"):
                flags.append("edit_name")
            b = f"A:{src}" + (f" via {via}" if via else "") + (f" name={nm_src}" if nm else "")
            res = ("A", shot, shot, shot, "shot", b)
        upper = min(x for x in [pt, g_min] if x) if pt else g_min
        if shot is not None and second:
            flags.append(f"secondhand:{SECOND.search(nm).group(0)}")
            upper = min(upper, shot) if upper else shot
        if res is None and timer and pt:
            mh = re.search(r"(\d+)\s*시간", timer[0])
            mm = re.search(r"(\d+)\s*분", timer[0])
            rem = dt.timedelta(hours=int(mh.group(1)) if mh else 0, minutes=int(mm.group(1)) if mm else 0)
            t = timer_time(rem, upper or pt)
            if t:
                res = ("B", t, t - dt.timedelta(minutes=1), t, "stated", f"B:timer({timer[1]}) '{timer[0]}'")
        if res is None and o.get("clock") and pt:
            cm = CLOCK.search(str(o["clock"]))
            if cm:
                hh = int(cm.group(1) or cm.group(3))
                mm = int(cm.group(2) or cm.group(4) or 0)
                t, _ = resolve_clock(upper or pt, hh, mm, None, None)
                if t:
                    res = ("B", t, t - dt.timedelta(minutes=1), t + dt.timedelta(minutes=1), "stated", f"B:clock '{o['clock']}'")
        info[i] = dict(res=res, flags=flags, upper=upper, pt=pt, shot=shot, second=second, nm=nm, timer=timer, members=members)

    # 글 주장 → 사진 잇기 (같은 글, 같은 %, 그 값이 사진 하나에만)
    rows_by_post = collections.defaultdict(list)
    for o in obs:
        rows_by_post[(o["site"], str(o["no"]))].append(o)
    link = collections.defaultdict(list)   # img → [(claim i, res)]
    val = []                                # A 사진에 이어진 주장으로 규칙을 확인
    for k, c in enumerate(claims):
        g, t, lo, hi, tk, b = cres[k]
        if g != "B" or c.get("pct") is None or c.get("kind") in ("predict", "prev_season", "delta"):
            continue
        rows = [o for o in rows_by_post.get((c.get("site"), str(c["no"])), []) if pct_match(c["pct"], o.get("pct"))]
        if c.get("score_eok") is not None:
            rows = [o for o in rows if o.get("score") is None or abs(o["score"] / 1e8 - c["score_eok"]) <= 1.0]
        ims = {sha.get(o["img"]) or o["img"]: o["img"] for o in rows}
        if len(ims) != 1:
            if len(ims) > 1:
                stats["claim_link_ambiguous"] += 1
            continue
        im = next(iter(ims.values()))
        same = [x for x in img if (sha.get(x) or x) == (sha.get(im) or im) and str(img[x]["no"]) == str(c["no"])]
        for x in same or [im]:
            link[x].append((k, cres[k]))
        stats["claim_linked"] += 1
        r = info[im]["res"]
        if r and r[0] == "A":
            val.append({"i": k, "img": im, "shot": fmt(r[1]), "lo": fmt(lo), "hi": fmt(hi), "basis": b,
                        "inside": lo <= r[1] <= hi, "off_min": round((r[1] - t).total_seconds() / 60, 1)})
    for i, lst in link.items():
        if info[i]["res"] is not None:
            continue
        lst.sort(key=lambda x: (x[1][3] - x[1][2]))   # 가장 좁은 구간
        k, (g, t, lo, hi, tk, b) = lst[0]
        up = info[i]["upper"]
        if up and hi > up:
            hi = up
            t = min(t, hi)
            lo = min(lo, hi)
        extra = ";claims=" + ",".join(str(x[0]) for x in lst) if len(lst) > 1 else ""
        info[i]["res"] = ("B", t, lo, hi, tk, f"B:claim#{k} {b[2:]}" + extra)

    # WP2 (2026-10-29): C로 남을 사진을 A(이름 꼴 확장·첨부 짝)·B(느슨한 타이머)로 올릴 수 있는지 본다. 이미 A·B인 사진은 건드리지 않는다.
    # 1단계 = 위의 기존 규칙, 2단계 = 후보 점검(기존 A만 참조한 top7 구간 안전장치) → 올림. 그다음 최종 등급으로 top7 열을 다시 계산한다.
    cand = collections.OrderedDict()   # 사진 → [(tag, res, flags)] (A 먼저)
    tight_2_10 = set()                 # 이름 시각이 2~10분에 걸려 올리지 않은 사진
    for i, x in info.items():
        if x["res"] is not None or x["pt"] is None:
            continue
        pt, o = x["pt"], img[i]
        cl = []
        if x["shot"] is None:
            nm_m = by_id.get(i)
            if nm_m:
                # (2) 이름 꼴 확장: manifest 이름이 name_time으로는 못 읽는 꼴일 때만
                if name_time(nm_m) is None:
                    vts = valid_times([nm_m], pt, skip_second=False)
                    if vts:
                        t = vts[0][0]
                        if SECOND.search(nm_m):   # 남의 화면: 지금처럼 상한으로만
                            x["flags"].append(f"secondhand:{SECOND.search(nm_m).group(0)}")
                            x["upper"] = min(x["upper"], t) if x["upper"] else t
                            x["second"] = True
                            x["nm"] = nm_m
                        else:
                            fl = []
                            if t > pt:
                                fl.append("shot_after_post")
                            if (pt - t) > dt.timedelta(hours=24):
                                fl.append("late>24h")
                            if nm_m.startswith("edit_"):
                                fl.append("edit_name")
                            cl.append(("name_ext", ("A", t, t, t, "shot", "A:name_ext(manifest)"), fl))
            else:
                # (3) 첨부 짝: manifest에 이름이 없는 사진 → 글의 첨부 이름 목록에서
                files = files_of.get(str(o["no"])) or []
                vts = valid_times(files, pt)
                if vts:
                    pair = None
                    mrow = man_of.get(str(o["no"]), {})
                    idx = int(i.rsplit("_", 1)[1])
                    fx = collections.defaultdict(list)
                    for fi, fn in enumerate(files):
                        fx[ext_class((fn or "").rsplit(".", 1)[-1].lower())].append(fi)
                    ix = collections.defaultdict(list)
                    for ii in sorted(mrow):
                        ix[ext_class((mrow[ii].get("ext") or "").lower())].append(ii)
                    me = ext_class((mrow.get(idx, {}).get("ext") or "").lower())
                    if me in fx and len(fx[me]) == len(ix.get(me, [])) and idx in ix[me]:
                        v1 = valid_times([files[fx[me][ix[me].index(idx)]]], pt)
                        if v1:
                            pair = ("extpair", v1[0][0])
                    if pair is None and len(files) >= len(mrow):   # 이름 없는 사진이 남는 글은 tight 안 씀 (그 사진은 다른 때일 수 있다)
                        tt = [t for t, _, _ in vts]
                        spread = max(tt) - min(tt)
                        if spread <= TIGHT_MAX:
                            pair = ("tight", min(tt))
                        elif spread <= TIGHT_LOOSE_MAX:
                            tight_2_10.add(i)
                    if pair:
                        t = pair[1]
                        fl = []
                        if t > pt:
                            fl.append("shot_after_post")
                        if (pt - t) > dt.timedelta(hours=24):
                            fl.append("late>24h")
                        cl.append((pair[0], ("A", t, t, t, "shot", f"A:name_pair({pair[0]})"), fl))
        # (1) 느슨한 갱신 타이머: 지금 규칙으로 읽히는 타이머가 없는 사진만
        if x["timer"] is None:
            tl = timers_loose.get(i) or next((timers_loose[m_] for m_ in x["members"] if m_ in timers_loose), None)
            if tl:
                t = timer_time(dt.timedelta(minutes=tl[2]), x["upper"] or pt)
                if t:
                    cl.append(("timer_loose", ("B", t, t - dt.timedelta(minutes=1), t, "stated",
                                               f"B:timer_loose({tl[1]}) '{tl[0]}'"), []))
        if cl:
            cand[i] = cl
    # 안전장치: 기존 규칙만으로 정한 A 사진들을 참조로 한 top7 구간 (열린 끝 허용)과 견준다
    wp2 = {"blocked": collections.Counter(), "blocked_left_c": 0, "cand": collections.Counter(), "cand_bracket": collections.Counter(),
           "cand_no_bracket": collections.Counter(), "cand_inside": collections.Counter()}
    snaps = refs = None
    if cand:
        pre = [{"grade": (info[o["img"]]["res"] or ("C",))[0], "h": se.h(info[o["img"]]["res"][1]) if info[o["img"]]["res"] else None}
               for o in obs]
        snaps, refs = top7_snaps(st, obs, pre)
    for i, cl in cand.items():
        x = info[i]
        br = None
        if snaps and i in snaps and snaps[i]["server"] and refs.get(snaps[i]["server"]):
            lo_, hi_, viol_ = top7_bracket(snaps[i]["ranks"], refs[snaps[i]["server"]])
            if not viol_:
                br = (lo_, hi_)
        blocked = []
        for tag, res_, fl in cl:
            h_ = se.h(res_[1])
            wp2["cand"][tag] += 1
            if br is None:
                wp2["cand_no_bracket"][tag] += 1
            else:
                wp2["cand_bracket"][tag] += 1
            if br is not None and (h_ < br[0] - TOP7_MARGIN_H or h_ > br[1] + TOP7_MARGIN_H):
                blocked.append(tag)
                continue
            if br is not None:
                wp2["cand_inside"][tag] += 1
            if x.get("wp2") is None:   # 첫 통과 후보만 쓴다 (뒤 후보는 통계로만 센다)
                x["res"] = res_
                x["flags"] = x["flags"] + fl
                x["wp2"] = tag
        if x.get("wp2") is None and blocked:   # 막혀서 C로 남은 사진만 표시한다
            for tag in blocked:
                x["flags"].append(f"cand_out_top7:{tag}")
                wp2["blocked"][tag] += 1
            wp2["blocked_left_c"] += 1

    # C
    for i, x in info.items():
        if x["res"] is not None:
            continue
        o = img[i]
        hi = x["upper"]
        lo = se.c0 if hi and hi >= se.c0 else None
        b = "C:post"
        if hi and x["pt"] and hi < x["pt"]:
            b = "C:earlier_copy" if not x["second"] else "C:secondhand_name"
        if o.get("file_date") and not o.get("file_date_out") and not x["shot"]:
            fd = dt.datetime.strptime(o["file_date"], "%Y-%m-%d")
            if lo is None or fd > lo:
                lo = min(fd, hi) if hi else fd
                b += ";file_date"
        if o.get("file_date_out"):
            lo = None
            x["flags"].append("file_date_out")
        x["res"] = ("C", hi, lo, hi, "upper", b)

    # 쓰기
    out_rows = []
    for o in obs:
        x = info[o["img"]]
        g, t, lo, hi, tk, b = x["res"]
        fl = list(x["flags"])
        if o.get("screen") == "season_end":   # 화면 종류는 줄마다 다를 수 있다
            if g == "C" and hi and hi >= se.end and (lo is None or lo < se.end):
                lo = se.end
                fl.append("end_screen")
            elif g in ("A", "B") and t and t <= se.end:
                fl.append("end_screen_before_end")
        if o.get("screen") == "tracker":
            fl.append("tracker")
        out_rows.append({"oi": len(out_rows), "site": o["site"], "img": o["img"], "row": o["row"], "grade": g,
                         "t": fmt(t), "t_lo": fmt(lo), "t_hi": fmt(hi), "t_kind": tk,
                         "h": se.h(t), "h_lo": se.h(lo), "h_hi": se.h(hi), "day": se.day(t), "day_lo": se.day(lo),
                         "basis": ";".join([b] + fl)})
    t7, t7_stat = top7_columns(st, obs, out_rows)
    for r, c in zip(out_rows, t7):
        r["h_lo_top7"], r["h_hi_top7"], r["top7"] = c if c else (None, None, None)
    with open(os.path.join(st, "post_time.jsonl"), "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(st, "post_time_claims.jsonl"), "w", encoding="utf-8") as f:
        for k, (g, t, lo, hi, tk, b) in enumerate(cres):
            f.write(json.dumps({"i": k, "grade": g, "t": fmt(t), "t_lo": fmt(lo), "t_hi": fmt(hi), "t_kind": tk,
                                "h": se.h(t), "h_lo": se.h(lo), "h_hi": se.h(hi), "day": se.day(t),
                                "day_lo": se.day(lo), "basis": b},
                               ensure_ascii=False) + "\n")

    # 요약: 시즌 안 % 관측 (server_classify.py와 같은 기준)
    core = [k for k, o in enumerate(obs) if o.get("season") == "in" and o.get("pct") is not None and o.get("screen") != "tracker"]
    cg = collections.Counter(out_rows[k]["grade"] for k in core)
    delays = sorted((info[i]["pt"] - info[i]["res"][1]).total_seconds() / 60
                    for i in {obs[k]["img"] for k in core} if info[i]["res"][0] == "A" and info[i]["pt"])
    def dsplit(after_end):
        return sorted((info[i]["pt"] - info[i]["res"][1]).total_seconds() / 60
                      for i in {obs[k]["img"] for k in core}
                      if info[i]["res"][0] == "A" and info[i]["pt"] and (info[i]["res"][1] > se.end) == after_end)
    d_in, d_post = dsplit(False), dsplit(True)
    delays_rows = sorted((ts(obs[k]["post_t"]) - ts(out_rows[k]["t"])).total_seconds() / 60
                         for k in core if out_rows[k]["grade"] == "A")

    def q(d, p):
        return round(d[min(len(d) - 1, int(p * (len(d) - 1) + 0.5))], 1) if d else None
    qs = (0, .1, .25, .5, .75, .9, .95, 1)
    self_t = [k for k, c in enumerate(claims) if c.get("kind") == "self" and c.get("pct") is not None and c.get("time")]
    summ = {
        "obs_all": dict(collections.Counter(r["grade"] for r in out_rows)),
        "core_rows": len(core), "core_grade": dict(cg),
        "core_basis": dict(collections.Counter(re.match(r"[ABC]:[a-z_]+", out_rows[k]["basis"]).group(0)
                                               + (":" + out_rows[k]["basis"].split(" ")[1] if "claim#" in out_rows[k]["basis"] else "")
                                               for k in core)),
        "core_flags": dict(collections.Counter(f.split(":")[0] for k in core for f in out_rows[k]["basis"].split(";")[1:])),
        "core_day": dict(collections.Counter(str(out_rows[k]["day"]) for k in core)),
        "delay_min_images": {"n": len(delays), "q": dict(zip(map(str, qs), (q(delays, p) for p in qs))),
                             "le10": sum(1 for d in delays if d <= 10), "gt60": sum(1 for d in delays if d > 60),
                             "gt1440": sum(1 for d in delays if d > 1440)},
        "delay_min_in_challenge": {"n": len(d_in), "q": dict(zip(map(str, qs), (q(d_in, p) for p in qs))),
                                   "le10": sum(1 for d in d_in if d <= 10), "gt60": sum(1 for d in d_in if d > 60)},
        "delay_min_after_end": {"n": len(d_post), "q": dict(zip(map(str, qs), (q(d_post, p) for p in qs)))},
        "delay_min_rows": {"n": len(delays_rows), "q": dict(zip(map(str, qs), (q(delays_rows, p) for p in qs)))},
        "claims_grade": dict(collections.Counter(r[0] for r in cres)),
        "claims_self_pct_time": {"n": len(self_t), "grade": dict(collections.Counter(cres[k][0] for k in self_t)),
                                 "basis": dict(collections.Counter(cres[k][5].split(" ")[0].split(";")[-1].split(":")[0]
                                                                   for k in self_t))},
        "claims_c_reason": dict(collections.Counter(r[5].split(";")[1].split(":")[0] for r in cres
                                                    if r[0] == "C" and ";" in r[5])),
        "link": dict(stats), "link_on_A": {"n": len(val), "inside": sum(v["inside"] for v in val), "rows": val},
        "timers": {k: v for k, v in timers.items()},
        "top7_images": t7_stat,
        "top7_core_rows": dict(collections.Counter(out_rows[k]["grade"] for k in core if out_rows[k]["top7"])),
    }
    # WP2 요약: 새 근거별 줄 수(전체·시즌 안 % 관측), 안전장치로 막힌 수, tight 2~10분이라 안 올린 수
    core_set = set(core)
    w_rows = {"all": collections.Counter(), "core": collections.Counter()}
    for k, o in enumerate(obs):
        x = info[o["img"]]
        if x.get("wp2"):
            tag = {"name_ext": "A:name_ext", "extpair": "A:name_pair(extpair)", "tight": "A:name_pair(tight)"}.get(x["wp2"])
            if tag is None:
                tag = "B:timer_loose(" + out_rows[k]["basis"].split("(", 1)[1].split(")", 1)[0] + ")"
            w_rows["all"][tag] += 1
            if k in core_set:
                w_rows["core"][tag] += 1
    blocked_rows = collections.Counter()
    t210_rows = collections.Counter()
    for k, o in enumerate(obs):
        x = info[o["img"]]
        for f_ in x["flags"]:
            if f_.startswith("cand_out_top7:"):
                blocked_rows["all"] += 1
                if k in core_set:
                    blocked_rows["core"] += 1
        if o["img"] in tight_2_10 and not x.get("wp2"):
            t210_rows["all"] += 1
            if k in core_set:
                t210_rows["core"] += 1
    summ["wp2"] = {
        "rows_new_all": dict(w_rows["all"]), "rows_new_core": dict(w_rows["core"]),
        "images_new": dict(collections.Counter(x["wp2"] for x in info.values() if x.get("wp2"))),
        "cand_images": dict(wp2["cand"]), "cand_with_bracket": dict(wp2["cand_bracket"]), "cand_no_bracket": dict(wp2["cand_no_bracket"]),
        "cand_inside_bracket": dict(wp2["cand_inside"]),
        "blocked_images": dict(wp2["blocked"]), "blocked_images_left_c": wp2["blocked_left_c"],
        "blocked_rows": dict(blocked_rows), "tight_2_10_images": sum(1 for i in tight_2_10 if not info[i].get("wp2")),
        "tight_2_10_rows": dict(t210_rows),
    }
    if a.summary:
        json.dump(summ, open(a.summary, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"obs {len(out_rows)} {summ['obs_all']} | core {len(core)} {dict(cg)} | A delay(min) n={len(delays)} "
          f"q50={q(delays, .5)} q75={q(delays, .75)} q90={q(delays, .9)} | claims {summ['claims_grade']} "
          f"self%time {len(self_t)} {summ['claims_self_pct_time']['grade']} | linked {stats['claim_linked']} | "
          f"top7 사진 {t7_stat} · 시즌 안 % 줄 {summ['top7_core_rows']}")
    w = summ["wp2"]
    print(f"WP2 새 근거 줄(전체) {w['rows_new_all']} · 시즌 안 % 줄 {w['rows_new_core']} | 안전장치로 막힌 사진 {w['blocked_images']} "
          f"(C로 남음 {w['blocked_images_left_c']}, 줄 {w['blocked_rows']}) | tight 2~10분이라 안 올린 사진 {w['tight_2_10_images']} (줄 {w['tight_2_10_rows']})")


if __name__ == "__main__":
    main()
