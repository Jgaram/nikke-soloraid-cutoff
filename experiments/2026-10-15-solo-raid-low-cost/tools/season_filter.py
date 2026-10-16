"""4단계: 관측 줄이 이번 시즌 솔로레이드 자료인지 거른다 (코드 + 글 문맥 두 겹). merge_obs.py가 부른다.

시즌 시작은 챌린지 시작 `c_start`(3일차 05:00 KST)다 (사용자 설명·결정, 2026-10-15). 1~2일차 순위·점수는 챌린지 컷 관측이 아니다.
season.json에 `c_start`가 없으면 `start` 날짜 + 2일 05:00으로 정한다 (`c_start()`).

코드 판정 `code_check(row, season, ref)` → ("in" | "out" | "unsure", 이유)
- `post_before`: 글 시각이 챌린지 시작 전 (시작 전 한 시간 목록에 걸린 글의 순위 화면은 지난 시즌이다).
- `file_date`·`shot_before`: 첨부 이름 날짜가 챌린지 시작일 전, 또는 첨부 이름 시각이 챌린지 시작 전.
- `label_season`: 판독 줄의 시즌 표기(`label`, 외부 트래커의 시즌별 목록)가 이번 시즌이 아님.
- `curve`: 같은 시즌 다른 글들의 (%, 점수) 또는 (순위, 점수)와 비교해 점수가 크게 벗어남 → unsure (out은 내지 않는다).
  비교 줄은 %가 ±25% 안(순위는 ×1.5 안)인 다른 글의 줄, 5줄·3글 이상일 때만 본다. 중앙값 대비 0.3배 미만·3배 초과.
  처음(0.6·1.6배 out)에는 S41 핵심 줄 45줄을 잘못 뺐다: 서버마다 같은 %의 점수가 2~3배 다르고,
  결과 화면 점수는 그 판 점수다. 협동작전(같은 %에서 약 0.1배)만 잡을 만큼 넓혔다 (세션 F1).
- `low_score`: 비교 줄이 모자라고 점수가 50억 미만 (협동작전 20~40억대 화면). unsure.
문맥 `ctx`: 글 단위 Haiku 판정 cur | prev | other | mixed (season_ctx_prompt.md).

최종 `decide(code, ctx, policy)` → in | unsure | out:
- code out → out.
- policy "unsure"(기본): code unsure이고 ctx가 prev·other이면 out, ctx가 cur·mixed·없음이면 unsure. code in이면 in.
  unsure는 관측에 남기되 분석 때 넣을지 고른다. S41 09-26 20~23시 시험에서 협동작전·유니온 화면 4줄이
  code unsure + ctx cur로 들어온 뒤 세 값으로 나눴다 (세션 F1).
  ctx `mixed`(이번 시즌과 지난 기록을 함께 올린 글)는 빼지 않는다. 빼면 S41 핵심 줄 손실이 3→4줄로 늘고
  음성은 거의 줄지 않았다 (세션 F1 시범). 분석 때 `ctx == "mixed"`로 따로 볼 수 있다.
- policy "all": ctx가 prev·other이면 code와 상관없이 out.
"""
import bisect
import collections
import datetime as dt
import re
import statistics

LO_UN, HI_UN = 0.3, 3.0


def c_start(season):
    """챌린지 시작 시각 'YYYY-MM-DD HH:MM:SS'. 예: S41 start 09-24 12:00 → 09-26 05:00."""
    if season.get("c_start"):
        return season["c_start"]
    d = dt.date.fromisoformat(season["start"][:10]) + dt.timedelta(days=2)
    return f"{d.isoformat()} 05:00:00"


class Ref:
    """같은 시즌 비교 줄. rows: (no, pct, rank, score)."""

    def __init__(self, rows):
        # 글 번호는 문자열로 맞춘다: 디시(숫자)와 네이버·블라블라링크(문자열)가 섞이면 같은 값끼리 정렬할 때 비교가 안 된다
        self.by_pct = sorted((p, str(no), s) for no, p, r, s in rows if p and s)
        self.by_rank = sorted((r, str(no), s) for no, p, r, s in rows if r and s)

    @staticmethod
    def _near(arr, lo, hi, no):
        i = bisect.bisect_left(arr, (lo,))
        out = []
        while i < len(arr) and arr[i][0] <= hi:
            if arr[i][1] != no:
                out.append(arr[i])
            i += 1
        return out

    def median(self, no, pct=None, rank=None):
        no = str(no)
        if pct:
            near = self._near(self.by_pct, pct / 1.25, pct * 1.25, no)
        else:
            near = self._near(self.by_rank, rank / 1.5, rank * 1.5, no)
        if len(near) < 5 or len({x[1] for x in near}) < 3:
            return None
        return statistics.median(x[2] for x in near)


def code_check(o, season, ref):
    start = c_start(season)
    if o.get("post_t") and o["post_t"] < start:
        return "out", "post_before"
    if o.get("file_date") and o["file_date"] < start[:10]:
        return "out", "file_date"
    if o.get("shot_t") and o["shot_t"] < start:
        return "out", "shot_before"
    m = re.search(r"S\s*(\d{2})", o.get("label") or "")
    if m and season.get("name") and f"S{m.group(1)}" != season["name"]:
        return "out", "label_season"
    s = o.get("score")
    if not s:
        return "in", ""
    med = None
    if o.get("pct"):
        med = ref.median(o["no"], pct=o["pct"])
    elif o.get("rank"):
        med = ref.median(o["no"], rank=o["rank"])
    if med:
        r = s / med
        if r < LO_UN or r > HI_UN:
            return "unsure", f"curve {r:.2f}"
        return "in", ""
    if s < 5_000_000_000:
        return "unsure", "low_score"
    return "in", ""


def decide(code, ctx, policy="unsure"):
    if code == "out":
        return "out"
    bad = ctx in ("prev", "other")
    if bad and (policy == "all" or code == "unsure"):
        return "out"
    return "unsure" if code == "unsure" else "in"


def annotate(obs, season, ctx, policy="unsure", ref_rows=None):
    """obs 줄마다 season_code·season_why·ctx·season을 붙인다. ref_rows가 없으면 obs 자신(코드 in인 줄)으로 비교한다."""
    if ref_rows is None:
        first = [o for o in obs if not o.get("mock") and o.get("screen") != "tracker"
                 and code_check(o, season, Ref([]))[0] != "out"]
        ref_rows = [(o["no"], o.get("pct"), o.get("rank"), o.get("score")) for o in first]
    ref = Ref(ref_rows)
    cnt = collections.Counter()
    for o in obs:
        c, why = code_check(o, season, ref)
        o["season_code"], o["season_why"] = c, why
        o["ctx"] = ctx.get(o["no"])
        o["season"] = decide(c, o["ctx"], policy)
        cnt[(c, o["season"])] += 1
    return cnt
