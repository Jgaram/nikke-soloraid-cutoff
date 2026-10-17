"""관측 줄마다 분석에서 뺄 사유(exclude)와 표시(flag), 쓸 값(pct_use·score_use)을 붙인다 (모델 0, 요청 0).
할 일 1.2 선수 기록이 아닌 화면 · 1.3 지난 시즌 화면 · 1.7 판독 의심 줄(일부) (docs/solo-raid-analysis.md).

python3 -I tools/solo_raid/post_exclude.py STATE [--posts F.jsonl ...] [--parsed F.jsonl] [--alt F.jsonl ...] [--manual F.json]

입력 (읽기만): STATE/obs.jsonl·season.json·server.jsonl(묶음 `cluster`)·route.jsonl(sha1)·posts_min.jsonl·claims.jsonl·
ocr.jsonl·ocr2.jsonl, 그리고 수동 판정 STATE/post_manual_season.json(있으면, --manual로 바꿈).
- --posts: 글 텍스트를 더 준다 (no·title·body 줄). S41은 디시 글이 state에 없어 재수집 posts_min을 준다.
- --parsed: 코드 판독 줄 ({id, rows:[{rank,pct,score}]}). STATE/parsed.jsonl이 있으면 기본으로 읽는다.
- --alt: 같은 사진의 다른 판독 ({img,row,pct,score}, 예: 재수집 obs_images.jsonl).
출력: STATE/post_exclude.jsonl — obs.jsonl 한 줄마다, 같은 순서로
  {oi, site, img, row, exclude:[…], flag:[…], pct_use, score_use, rank_use, why}
  oi = obs.jsonl 줄 번호(0부터). score_use는 억 단위, 0.1억 반올림. pct_use는 %.
  rank_use는 게임 화면 등수 1~200이고 %가 없는 줄만 (200을 넘거나 %가 함께 있으면 null + flag). 트래커 화면은 null.
표준 출력에 사유별 줄 수 (전체, 시즌 안 % 관측 = season in·pct 있음·tracker 아님).

exclude (기본으로 뺀다)
- season_out: 수집 단계 시즌 거르기가 이미 시즌 밖으로 뺀 줄 (why에 season_why·ctx).
- tracker: 판독 화면이 tracker (외부 사이트·집계표·덱 피해 트래커). 게임 안 선수 화면이 아니다.
- reward_table: 시즌 종료 보상 구간 표. season_end 화면이고 % 글자가 「~ 1.0%」처럼 물결로 시작하거나 메모에
  「보상 구간·보상표·보상 탭」. 구간 경계라 선수의 %가 아니다 (MY RANK 강조 줄도 구간의 끝값일 뿐이다).
- pct_over_100: %가 100을 넘는다.
- pct_no_sign: 사진 판독(code 아님)의 % 글자에 %·퍼가 없다. S41 두 줄(70·124)은 다른 실행의 같은 칸 판독이
  rank_txt로 읽었다 → 순위다.
- other_raid: 메모에 점수 아이콘이 「조준 모양」·「다름」(솔로레이드 아닌 레이드 화면. 수집 F1 「협동작전·유니온 화면」).
- mock: 판독이 모의전·계산기로 표시한 줄 (S41 obs에는 0줄, 판독 단계에서 이미 빠짐).
- prev_season: 시즌 판정이 지난 시즌 (아래).
- season_unsure: 수집 단계 season이 unsure(점수가 같은 시즌 곡선에서 0.3배 미만·3배 초과, 또는 50억 미만)이고
  이번 시즌 근거가 없다. 수집 F1 시험에서 unsure 줄은 모두 다른 레이드였다. 넣어 보려면 이 사유만 풀면 된다.
- ctx_prev·ctx_other: 시즌 안 줄인데 글 문맥(Haiku)이 prev·other이고 사진 단위 근거가 없다.
- discard: data/solo-raid/S{n}/discard.json(「발견되면 버릴 것」, 할 일 3.A2)에 있는 글의 줄, 그리고 같은 사진(sha1·dup_of)의 다른 글 사본.
  줄 하나 단위(scope row) 항목은 그 줄과 같은 사진 사본의 같은 줄만.
  why에 사유·결정자·날짜.

flag (빼지 않고 표시만)
- season_unsure_mixed: 글 문맥 mixed(이번 시즌과 지난 기록을 함께 올린 글)인데 사진을 가를 근거가 없다.
- ctx_overridden: 글 문맥은 prev·other·mixed지만 사진 근거로 이번 시즌이 됐다.
- season_conflict: 사진 근거끼리 이번·지난 시즌이 엇갈린다.
- curve_outlier: 수집 단계 unsure(곡선 밖)였지만 사진 근거로 이번 시즌이 됐다. 점수 판독을 의심한다.
- score_short: 게임 화면 점수가 9자리 이하 (자릿수가 빠졌을 수 있다).
- rank_over_200: 게임 화면(트래커 아님)의 등수가 200을 넘는다. 게임은 200위까지만 등수로 보여 주고 그 아래는 %다 (사용자 확인,
  2026-10-16). 배지 숫자 같은 다른 숫자를 등수로 읽은 줄로 보인다 — %·점수도 함께 잘못 읽었는지 review.py 후보 rank_over_200으로 다시 본다.
  rank_use는 null. S31~S41 obs 311줄(대부분 % 있는 순위표 줄). 등수 점검 세션(2026-10-16)이 S32~S41 265장을 원본으로 보니
  모두 아바타 아래 계정 레벨 배지였고 %·점수는 맞았다 (전부 keep).
- rank_with_pct: 게임 화면 한 줄에 1~200 등수와 %가 함께 있다. 게임은 한 줄에 둘 중 하나만 보여 주므로 등수 쪽은 200 이하 레벨 배지로 본다
  (2026-10-16, 사용자 결정). rank_use는 null, 점검 후보로 올리지 않는다 (위 점검 결과로 %·점수는 그대로 쓴다). S31~S41 obs 69줄.
- best_record: 판독 메모에 「나의 최고 기록」·BEST DAMAGE가 나오는 줄 (트래커 제외). 「나의 최고 기록」 줄의 %는 지금 %가 아니라
  그 기록을 세운 때의 %일 수 있다 (S41 oi 2039: 종료 후 올린 my_rank 1.40%·250.4억, 한섭 종료 3% 컷 250.35억). 2026-10-16.
- other_raid_maybe: 메모에 협동작전·유니온 레이드가 나오지만 확실하지 않다.
- verify_differ: 원본 판독과 코드 판독이 달랐던 사진의 줄 (점수·% 값이 있는 줄만). 같은 값을 다른 판독이 받치면
  원값을 쓴다(why에 근거).
  받침이 없으면 verify_unresolved도 붙인다.

시즌 판정 (1.3) — 점수·%의 크기로는 정하지 않는다. 근거는 사진 하나(img) 단위로 모은다.
- shot: 첨부 이름·epoch 시각(shot_t)이 챌린지 시작(c_start) 뒤 → cur.
- final_pre_end: 최종 결과 화면(screen season_end, 보상 구간 표 빼고)인데 글이 시즌 끝(season.json end) 전에 올라왔다 → prev.
  끝나기 전에는 이번 시즌 최종 %가 없다. 그 사진의 shot(cur) 근거는 지운다 (챌린지 중에 지난 시즌 결과를 찍은 것). 2026-10-16, S33 4550934 등
  (시즌 세션 점검이 「같은 사람 기록」으로 남긴 곡면 이탈 z 12~19 줄).
- xseason: 같은 사진 파일(sha1)이 앞 시즌(수집 끝난 시즌) 받기 구간에 이미 있다 → prev (`xseason.py`, 2026-10-17 분석 ②).
  S31 4183097·4196485가 S30 3940094_1(S29 3810211_0의 재업로드)을 다시 올린 일. 수동 판정처럼 다른 근거보다 앞서고, 묶음으로 퍼뜨리지 않는다.
  파일이 달라도 사진의 한 줄의 정확 점수(10자리 이상)가 앞 시즌 시즌 안 관측에 있으면 그 사진도 prev (`xseason.score_seen`, 2026-10-17 분석 ③).
- note_date: 판독 메모의 연·월·일이 모두 챌린지 시작 전 (예: 명예의 전당 「2024.02.09-02.16」) → prev.
- label: 판독 label의 시즌 표기가 이번 시즌 → cur.
- text: 글 제목·본문에서 이 줄의 %(와 점수)가 함께 나온 기록 조각을 찾는다. 조각에 이번 시즌 이름(S41)·보스 이름·
  챌린지 달("9월", "26년 9월", "26.09")이 있고 다른 시즌 번호·다른 달이 없으면 cur, 반대면 prev.
- manual: post_manual_season.json의 사진별 판정 (사유 필수). 다른 근거보다 앞선다.
- 묶음: 같은 지문 묶음(server.jsonl cluster, 정확 점수가 같은 사진들)과 같은 사진(sha1)은 한 시즌이다.
  묶음 안 사진들의 근거를 모아 묶음 전체에 붙인다 (예: S41 c9 = S37 한섭 순위표).
- 사진 근거와 묶음 근거를 합쳐 한 값이면 그 시즌, 엇갈리면 conflict, 없으면 미정.

판독 의심 (1.7 일부)
- score_use = round(score / 1e8, 1) (0.1억). 11자리 점수의 약 3%가 한 자리 다르다 (재수집 원본 재판독).
- verify == differ 줄: 원값의 점수가 (1) 코드 판독 (2) 다른 판독 (3) OCR 글자 (4) 다른 사진의 같은 정확 점수 중
  하나라도 받치고 %가 어긋나지 않으면 원값을 쓴다. 곡면에서 벗어난 줄 표시는 하지 않는다 (2단계).
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discard  # noqa: E402
import xseason  # noqa: E402

KST_MONTH = re.compile(r"(?<!\d)(?:(\d{2})\s?년\s?)?(\d{1,2})(?:\s?~\s?(\d{1,2}))?\s?월")
YYMM = re.compile(r"(?<![\d.])(\d{2})\.(\d{2})(?![\d.]|\s*(?:%|퍼))")
S_NUM = re.compile(r"(?<![A-Za-z0-9])[Ss]\s?(\d{2})(?!\d)")
PCT = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d{1,2})?)\s*(?:%|퍼)")
RANK_MAX = 200   # 게임 화면은 200위까지 등수, 그 아래는 % (사용자 확인, 2026-10-16)
EOK = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)\s*억")
EOK_AFTER = re.compile(r"\s*\)?\s*(\d{1,3}(?:\.\d{1,2})?)\s*억")
LONG = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3}){2,3})(?!\d)")
NOTE_DATE = re.compile(r"(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})")
OTHER_RAID = re.compile(r"조준|점수 아이콘이 다름")
OTHER_RAID_MAYBE = re.compile(r"협동\s?작전|유니온\s?레이드")
BEST_RECORD = re.compile(r"최고\s?기록|BEST\s?DAMAGE", re.I)
REWARD_NOTE = re.compile(r"보상\s?(?:구간|표|탭)")


def jl(p):
    out = []
    if p and os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                out.append(json.loads(l))
    return out


def eok1(score):
    return None if score is None else int(score / 1e7 + 0.5) / 10


class SeasonWords:
    """이번 시즌을 가리키는 말과 다른 시즌을 가리키는 말 (season.json에서)."""

    def __init__(self, season):
        self.name = season.get("name")                      # "S41"
        self.num = int(self.name[1:]) if self.name and self.name[1:].isdigit() else None
        boss = season.get("boss") or ""
        self.boss = re.compile(r"\s*".join(map(re.escape, boss.replace(" ", "")))) if boss else None
        a, b = season["c_start"][:7], (season.get("end") or season["c_start"])[:7]
        self.ym = {(int(a[2:4]), int(a[5:7])), (int(b[2:4]), int(b[5:7]))}
        self.months = {m for _, m in self.ym}
        self.years = {y for y, _ in self.ym}

    def judge(self, seg):
        cur, prev = [], []
        for m in S_NUM.finditer(seg):
            (cur if int(m.group(1)) == self.num else prev).append(m.group(0))
        if self.boss and self.boss.search(seg):
            cur.append("boss")
        for m in KST_MONTH.finditer(seg):
            y = int(m.group(1)) if m.group(1) else None
            ms = {int(m.group(2))} | ({int(m.group(3))} if m.group(3) else set())
            if all(1 <= x <= 12 for x in ms):
                ok = ms <= self.months and (y is None or y in self.years)
                (cur if ok else prev).append(m.group(0).strip())
        for m in YYMM.finditer(seg):
            y, mo = int(m.group(1)), int(m.group(2))
            if 20 <= y <= 40 and 1 <= mo <= 12:
                (cur if (y, mo) in self.ym else prev).append(m.group(0))
        if cur and not prev:
            return "cur", cur[0]
        if prev and not cur:
            return "prev", prev[0]
        return None, None


def records(text):
    """글 텍스트 → [(pct, 억 또는 None, 조각)]. 조각 = 앞 기록 끝부터 이 % 끝까지 (+ 바로 뒤 「N억」)."""
    out, pos = [], 0
    for m in PCT.finditer(text):
        if m.start() < pos:
            continue
        seg = text[pos:m.end()]
        sc = None
        after = EOK_AFTER.match(text, m.end())
        end = m.end()
        if after:
            sc, end = float(after.group(1)), after.end()
        else:
            e = list(EOK.finditer(text, pos, m.start()))
            if e:
                sc = float(e[-1].group(1))
            else:
                ln = list(LONG.finditer(text, pos, m.start()))
                if ln:
                    sc = int(ln[-1].group(1).replace(",", "")) / 1e8
        out.append((float(m.group(1)), sc, seg + text[m.end():end]))
        pos = end
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--posts", action="append", default=[])
    ap.add_argument("--parsed", default=None)
    ap.add_argument("--alt", action="append", default=[])
    ap.add_argument("--manual", default=None)
    a = ap.parse_args()
    st = a.state
    season = json.load(open(os.path.join(st, "season.json"), encoding="utf-8"))
    if not season.get("c_start"):
        import datetime as dt
        d = dt.date.fromisoformat(season["start"][:10]) + dt.timedelta(days=2)
        season["c_start"] = f"{d.isoformat()} 05:00:00"
    c0 = season["c_start"]
    words = SeasonWords(season)

    obs = jl(os.path.join(st, "obs.jsonl"))
    srv = jl(os.path.join(st, "server.jsonl"))
    if len(srv) != len(obs) or any((s["img"], s["row"]) != (o["img"], o["row"]) for s, o in zip(srv, obs)):
        raise SystemExit("server.jsonl이 obs.jsonl과 줄이 맞지 않는다 → server_classify.py를 다시 돌린다")
    sha = {r["id"]: r.get("sha1") for r in jl(os.path.join(st, "route.jsonl"))}
    man_p = a.manual or os.path.join(st, "post_manual_season.json")
    manual = {}
    if os.path.exists(man_p):
        for m in json.load(open(man_p, encoding="utf-8")).get("img", []):
            assert m["season"] in ("cur", "prev") and m.get("why"), m
            manual[m["img"]] = m

    # 글 텍스트: 제목 + 본문 (글 번호 문자열 → 텍스트 목록)
    text = collections.defaultdict(list)
    for f in [os.path.join(st, "posts_min.jsonl")] + a.posts:
        for p in jl(f):
            t = " ".join(x for x in (p.get("title"), p.get("body")) if x)
            if t and t not in text[str(p["no"])]:
                text[str(p["no"])].append(t)
    for c in jl(os.path.join(st, "claims.jsonl")):
        t = c.get("txt")
        no = str(c["no"])
        if t and not any(t in x for x in text[no]):
            text[no].append(t)
    recs = {no: [r for t in ts for r in records(t)] for no, ts in text.items()}

    # 1) 사진 단위 근거
    ev = collections.defaultdict(set)      # img -> {(season, basis, detail)}
    for i, o in enumerate(obs):
        img = o["img"]
        if o.get("shot_t") and o["shot_t"] >= c0:
            ev[img].add(("cur", "shot", o["shot_t"]))
        ds = [(int(y), int(m), int(d)) for y, m, d in NOTE_DATE.findall(o.get("note") or "")]
        if ds and o.get("screen") != "tracker" and all(f"{y:04d}-{m:02d}-{d:02d}" < c0[:10] for y, m, d in ds):
            ev[img].add(("prev", "note_date", "%d.%02d.%02d" % ds[0]))
        lm = S_NUM.search(o.get("label") or "")
        if lm and words.num is not None and int(lm.group(1)) == words.num:
            ev[img].add(("cur", "label", o["label"][:30]))
        if o.get("pct") is not None and o.get("screen") != "tracker":
            for p, sc, seg in recs.get(str(o["no"]), []):
                if abs(p - o["pct"]) > 0.005:
                    continue
                if sc is not None and o.get("score") and abs(sc - o["score"] / 1e8) > 1.0:
                    continue
                s, why = words.judge(seg)
                if s:
                    ev[img].add((s, "text", re.sub(r"\s+", " ", seg.strip())[-40:]))
    # 끝나기 전에 올라온 최종 결과 화면(season_end, 보상 구간 표 빼고)은 이번 시즌일 수 없다 → 지난 시즌 (2026-10-16, S33 4550934 등).
    # 챌린지 중에 찍은 지난 시즌 결과 화면이라 shot(cur) 근거는 지운다
    end = season.get("end")
    for o in obs:
        if (end and o.get("screen") == "season_end" and o.get("post_t") and o["post_t"] < end and o.get("pct") is not None
                and not (o.get("pct_txt") or "").strip().startswith("~") and not REWARD_NOTE.search(o.get("note") or "")):
            ev[o["img"]] = {e for e in ev[o["img"]] if e[1] != "shot"} | {("prev", "final_pre_end", o["post_t"][:16])}

    # 앞 시즌에 이미 올라온 같은 사진 → 지난 시즌 (2026-10-17, 시즌을 건넌 재업로드)
    # 사진 하나에만 정하고 묶음으로 퍼뜨리지 않는다 (지문 잇기가 이번 시즌 사진과 이어 묶음 전체가 conflict가 되지 않게)
    xs = xseason.photo_sha(st)
    xs_img = {o["img"]: "같은 사진 " + xs[sha[o["img"]]] for o in obs if sha.get(o["img"]) in xs}
    # 파일은 달라도 정확 점수 하나가 앞 시즌에 있으면 앞 시즌 화면 (2026-10-17 분석 ③, xseason.score_seen). 사진 전체를 prev로
    xp = xseason.score_seen(st)
    for o in obs:
        e = xseason.exact(o)
        if e and e in xp and o["img"] not in xs_img:
            xs_img[o["img"]] = f"같은 정확 점수 {xp[e]}"
    for o in obs:   # 같은 파일 사본도 같은 판정
        if o["img"] not in xs_img and o.get("dup_of") in xs_img:
            xs_img[o["img"]] = xs_img[o["dup_of"]]

    # 2) 묶음: 지문 묶음 이름, 없으면 같은 사진(sha1)
    group = {}
    for o, s in zip(obs, srv):
        # 이음 덩어리(link): 서버로 가르기 전의 지문 덩어리 (2026-10-16 server_classify가 섞인 묶음을 가른 뒤에도 시즌 판정은 같은 사람 잇기 그대로)
        group[o["img"]] = s.get("link") or s.get("cluster") or ("sha:" + (sha.get(o["img"]) or o["img"]))
    members = collections.defaultdict(set)
    for img, g in group.items():
        members[g].add(img)
    gev = collections.defaultdict(set)
    for img, g in group.items():
        for e in ev.get(img, ()):
            gev[g].add((e[0], e[1], img))
        if img in manual:
            gev[g].add((manual[img]["season"], "manual", img))

    def img_season(img):
        if img in manual:
            return manual[img]["season"], f"manual: {manual[img]['why'][:60]}"
        if img in xs_img:
            return "prev", f"xseason 앞 시즌에 {xs_img[img]}"
        own = ev.get(img, set())
        gs = {e for e in gev.get(group[img], set()) if e[2] != img}
        vals = {e[0] for e in own} | {e[0] for e in gs}
        parts = [f"{e[1]} {e[2]}" for e in sorted(own)][:2]
        if gs:
            g = group[img]
            gname = g if not g.startswith("sha:") else "같은 사진"
            gs_l = sorted(gs)
            parts.append(f"묶음 {gname}: " + ", ".join(f"{e[2]} {e[1]}={e[0]}" for e in gs_l[:2])
                         + (f" 외 {len(gs_l) - 2}" if len(gs_l) > 2 else ""))
        if not vals:
            return None, ""
        if len(vals) > 1:
            return "conflict", "; ".join(parts)
        return vals.pop(), "; ".join(parts)

    # 3) 판독 의심: verify differ 줄의 받침
    parsed_p = a.parsed or os.path.join(st, "parsed.jsonl")
    parsed = {p["id"]: p for p in jl(parsed_p)}
    alt = collections.defaultdict(list)
    for f in a.alt:
        for r in jl(f):
            alt[r["img"]].append(r)
    ocr_digits = collections.defaultdict(str)
    for f in ("ocr.jsonl", "ocr2.jsonl"):
        for r in jl(os.path.join(st, f)):
            ocr_digits[r["id"]] += " " + " ".join(re.sub(r"\D", "", t) for _, t, _ in r.get("items", []))
    exact_imgs = collections.defaultdict(set)
    for o in obs:
        if o.get("score") and o["score"] >= 10 ** 9 and o["score"] % 10 ** 6:
            exact_imgs[o["score"]].add(sha.get(o["img"]) or o["img"])

    def verify(o):
        """→ (받침 목록, % 어긋남 목록)"""
        sup, bad = [], []
        s, p = o.get("score"), o.get("pct")
        for name, rows in (("code", parsed.get(o["img"], {}).get("rows", [])), ("alt", alt.get(o["img"], []))):
            if s:
                m = [r for r in rows if r.get("score") and abs(r["score"] - s) < 1e7]
                if m:
                    sup.append(name)
                    if p is not None and m[0].get("pct") is not None and abs(m[0]["pct"] - p) > 0.005:
                        bad.append(f"{name} {m[0]['pct']}")
            elif p is not None and any(r.get("pct") is not None and abs(r["pct"] - p) < 0.005 for r in rows):
                sup.append(name)
        if s and str(s) in ocr_digits.get(o["img"], ""):
            sup.append("ocr")
        if s and len(exact_imgs.get(s, set()) - {sha.get(o["img"]) or o["img"]}) > 0:
            sup.append(f"같은 점수 다른 사진 {len(exact_imgs[s]) - 1}장")
        return sup, bad

    # 버릴 것: 그 글의 사진과 같은 사진(sha1·dup_of)
    drop = discard.posts(st)
    drop_img = {o["img"]: drop[str(o["no"])] for o in obs if str(o["no"]) in drop}
    drop_sha = {sha[i]: d for i, d in drop_img.items() if sha.get(i)}
    for o in obs:
        d = drop_sha.get(sha.get(o["img"])) or drop_img.get(o.get("dup_of"))
        if d and o["img"] not in drop_img:
            drop_img[o["img"]] = d
    # 줄 하나 단위(scope row): 그 줄, 같은 사진(sha1·dup_of) 사본의 같은 줄
    drop_row = discard.rows(st)
    row_sha = {(sha[i], r): d for (i, r), d in drop_row.items() if sha.get(i)}
    for o in obs:
        k = (o["img"], o["row"])
        d = drop_row.get(k) or row_sha.get((sha.get(o["img"]), o["row"])) or drop_row.get((o.get("dup_of"), o["row"]))
        if d and k not in drop_row:
            drop_row[k] = d

    # 4) 줄마다
    out, cnt, cnt_core = [], collections.Counter(), collections.Counter()
    for oi, o in enumerate(obs):
        ex, fl, why = [], [], []
        if o["img"] in drop_img:
            d = drop_img[o["img"]]
            ex.append("discard")
            why.append(f"버릴 것 {d['site']} {d['no']}{'' if str(o['no']) == str(d['no']) else ' 사본'}: "
                       f"{d['reason'][:40]} ({d['by']}, {d['decided']})")
        elif (o["img"], o["row"]) in drop_row:
            d = drop_row[(o["img"], o["row"])]
            ex.append("discard")
            why.append(f"버릴 줄 {d['img']}#{d['row']}{'' if o['img'] == d['img'] else ' 사본'}: "
                       f"{d['reason'][:40]} ({d['by']}, {d['decided']})")
        scr, ctx, sea = o.get("screen"), o.get("ctx"), o.get("season")
        pct_txt = o.get("pct_txt") or ""
        note = o.get("note") or ""
        if sea == "out":
            ex.append("season_out")
            why.append(f"수집 out ({o.get('season_why') or 'ctx ' + str(ctx)})")
        if o.get("mock"):
            ex.append("mock")
        if scr == "tracker":
            ex.append("tracker")
        if scr == "season_end" and (pct_txt.strip().startswith("~") or REWARD_NOTE.search(note)):
            ex.append("reward_table")
            why.append(f"보상 구간 표 「{pct_txt or note[:20]}」")
        if o.get("pct") is not None and o["pct"] > 100:
            ex.append("pct_over_100")
        if o.get("pct") is not None and pct_txt and scr != "tracker" and o.get("src") != "code" \
                and not re.search(r"%|퍼", pct_txt):
            ex.append("pct_no_sign")
            why.append(f"% 글자 「{pct_txt}」에 %가 없음(순위로 보임)")
        if OTHER_RAID.search(note):
            ex.append("other_raid")
            why.append("메모: 점수 아이콘이 다름(조준)")
        elif OTHER_RAID_MAYBE.search(note):
            fl.append("other_raid_maybe")
        if o.get("score") and o["score"] < 10 ** 9 and scr != "tracker":
            fl.append("score_short")
        if scr != "tracker" and BEST_RECORD.search(note):
            fl.append("best_record")
        rank = o.get("rank")
        rank_use = None
        if scr != "tracker" and isinstance(rank, (int, float)):
            if rank > RANK_MAX:
                fl.append("rank_over_200")
                why.append(f"등수 {rank}: 게임 화면 등수는 {RANK_MAX}위까지")
            elif o.get("pct") is not None:
                fl.append("rank_with_pct")
                why.append(f"등수 {rank}와 %가 한 줄: 등수 쪽은 레벨 배지로 봄")
            elif isinstance(rank, int) and rank >= 1:
                rank_use = rank

        if sea != "out":
            s, swhy = img_season(o["img"])
            if s == "prev":
                ex.append("prev_season")
                why.append(f"지난 시즌: {swhy}")
            elif s == "conflict":
                fl.append("season_conflict")
                why.append(f"시즌 근거 엇갈림: {swhy}")
                if sea == "unsure":
                    ex.append("season_unsure")
            elif s == "cur":
                if sea == "unsure":
                    fl.append("curve_outlier")
                if ctx in ("prev", "other", "mixed"):
                    fl.append("ctx_overridden")
                if sea == "unsure" or ctx in ("prev", "other", "mixed"):
                    why.append(f"이번 시즌: {swhy}")
            else:
                if sea == "unsure":
                    ex.append("season_unsure")
                    why.append(f"수집 unsure ({o.get('season_why')}), 이번 시즌 근거 없음")
                elif ctx == "mixed":
                    fl.append("season_unsure_mixed")
                elif ctx in ("prev", "other"):
                    ex.append("ctx_" + ctx)

        pct_use, score_use = o.get("pct"), eok1(o.get("score"))
        if o.get("verify") == "differ" and (o.get("score") or o.get("pct") is not None):
            fl.append("verify_differ")
            sup, bad = verify(o)
            if sup and not bad:
                why.append("판독 차이: 원값을 받침 " + "·".join(sup))
            else:
                fl.append("verify_unresolved")
                why.append("판독 차이: 받침 없음" + (f", % 어긋남 {bad}" if bad else "") + " → 원값 그대로")
        rec = {"oi": oi, "site": o.get("site"), "img": o["img"], "row": o["row"],
               "exclude": list(dict.fromkeys(ex)), "flag": list(dict.fromkeys(fl)),
               "pct_use": pct_use, "score_use": score_use, "rank_use": rank_use, "why": " / ".join(why)[:300]}
        out.append(rec)
        core = sea == "in" and o.get("pct") is not None and scr != "tracker"
        for k in rec["exclude"] or ["(none)"]:
            cnt["ex:" + k] += 1
            if core:
                cnt_core["ex:" + k] += 1
        for k in rec["flag"]:
            cnt["fl:" + k] += 1
            if core:
                cnt_core["fl:" + k] += 1
        cnt["rows"] += 1
        if core:
            cnt_core["rows"] += 1
            if not rec["exclude"]:
                cnt_core["kept"] += 1

    with open(os.path.join(st, "post_exclude.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    unsure_pct = [r for r, o in zip(out, obs) if o.get("season") == "unsure" and o.get("pct") is not None]
    print("post_exclude: 전체", dict(sorted(cnt.items())))
    print("시즌 안 % 관측", dict(sorted(cnt_core.items())))
    print("unsure % 줄", len(unsure_pct), dict(collections.Counter(
        ",".join(r["exclude"]) or "(남김:" + ",".join(r["flag"]) + ")" for r in unsure_pct)))


if __name__ == "__main__":
    main()
