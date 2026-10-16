"""솔로레이드 후처리 1.4·1.6: 관측 줄의 중복·같은 사람·같은 시점을 잇고, 글 주장을 사진 관측에 잇는다 (모델 0, 요청 0).

python3 -I tools/solo_raid/post_link.py STATE [--no-code-scores] [--shot-gap 10] [--kind kind|rule|either]

원자료(obs.jsonl·claims.jsonl·route.jsonl·season.json)는 읽기만 한다. 줄을 지우지 않고 열쇠와 사유만 붙인다.
제외 판단(선수 기록이 아닌 화면·지난 시즌)은 하지 않는다 — 모든 줄을 잇기만 한다.

출력 1) STATE/post_link.jsonl — obs.jsonl 한 줄마다 한 줄, 같은 순서. 열쇠 oi(줄 번호, 0부터)·site·img·row.
  - dup_canon: 같은 사진을 다시 올린 줄(dup_of)이면 원본 사진의 같은 row 줄 oi. 원본이면 null.
  - person / person_basis: 같은 사람 열쇠.
      fp        정확 점수(지문)로 이은 사람. 지문 기준은 server_classify.py의 exact()와 같다
                (10자리 이상 글자, 10억 이상, 끝 6자리가 0이 아님). 코드 OCR 줄(score_txt 없음)의 점수도 쓴다
                (기본 켬, 2026-10-16. server_classify.py도 같이 켰다. 끄려면 둘 다 --no-code-scores).
      post_one  지문 없는 본인 줄(self)인데, 같은 글의 본인 줄 지문이 한 사람뿐이라 그 사람에 붙임
      post_self 지문 없는 본인 줄 — 글 단위 임시 열쇠 (같은 글의 지문 없는 본인 줄은 글쓴이 한 사람으로 본다)
      row       지문 없는 남의 줄 — 줄 하나가 한 사람 (다른 줄과 잇지 않음)
    열쇠 이름은 P·T·R + 그 사람의 첫 oi다 (obs.jsonl이 바뀌지 않으면 그대로).
  - moment / rep / n_rep: 같은 사람 안의 시점 묶음. rep는 시각 순 1부터, n_rep는 그 사람의 시점 수.
    같은 시점(진짜 겹침): (같은 사진(sha1)이거나 두 사진이 「같은 화면」)이고 두 줄의 값이 모순 없음.
      이행적으로 잇되 한 시점 안의 어느 두 줄도 모순되지 않게 합친다 (같은 사진 쌍 먼저, 그다음 시각 차 작은 쌍. 2026-10-16 고침:
      전에는 % 없는 줄을 거쳐 서로 다른 % 줄이 한 시점으로 묶였다 — S41 P214·P506, 독립 관측 5개 적게 셈).
      - 값 모순 없음: 둘 다 있는 칸(%·순위·점수)이 같다 (% 0.005, 점수 0.1억 안).
      - 같은 화면: 두 사진에 함께 나온 사람들의 값이 모순 없고, 둘 다 지문 사람이 2명 이상이면 한쪽 명단이 다른 쪽에 들어가며,
        둘 다 촬영 시각(shot_t)이 있으면 차이가 --shot-gap분 이하. 게시 시각만 다른 것은 같은 화면의 재게시로 본다.
    값이 다르거나 순위표 명단이 다르면(다른 시각의 화면) 새 시점이다 — 지우지 않는다.
  - indep: 시점마다 대표 한 줄만 true (원본 사진 > 시즌 안 > % 있음 > 점수 있음 > 촬영 시각 있음 > 앞 줄). same_as: 대표가 아니면 대표 oi.
    부분 집합(예: 시즌 안 % 줄)의 독립 수는 indep 합이 아니라 그 줄들의 moment 가짓수로 센다.
  - self·screen·rank·pct·score는 확인용으로 obs에서 옮긴 값이다.
출력 2) STATE/post_claims.jsonl — claims.jsonl 한 줄마다 한 줄, 같은 순서. 열쇠 i(줄 번호, server_claims.jsonl의 i와 같음).
  - kind: claims의 kind(Haiku 값, 없으면 규칙 값)를 따른다 (--kind로 바꿔 민감도를 본다). kind_agree: kind_rule == kind_haiku (Haiku 없으면 null).
    근거: S41 200줄 시범에서 6종 정확도 규칙 70.0% · Haiku 80.0%, 관측 여부 78.5% · 86.0%
    (experiments/2026-10-15-solo-raid-low-cost/README.md 「결과 — 세션 B」).
  - drop(사유 목록, 비면 keep=true):
      kind:question·kind:predict·kind:prev_season·kind:cutoff·kind:other·kind:delta  본인 기록이 아님
      no_pct      %가 없음 (점수만)
      int_pct     정수 % — 대개 구간 이름("3퍼 따고 싶다", "in 10%", "3퍼 전문가 246억"). 점수가 함께 있어도 %는 점값이 아니다
      post_before 게시가 챌린지 시작(season.json c_start, 없으면 start 날짜+2일 05:00) 전
      same_claim  다른 사유 없이 남는 줄 중, 같은 글에서 같은 %·점수·시각 표현을 이미 적은 줄 (제목과 본문 반복). same_as_i에 첫 줄.
                  시각 표현 없음·지금·현재·방금은 같은 것으로 본다
  - flag(빼지 않는 표시): q(값을 넣은 질문형), kind_disagree(규칙과 Haiku가 다름), score_post(점수를 글의 다른 줄에서 가져옴)
  - link_oi: 같은 글의 사진 관측 중 값이 같은 줄 (% 일치 > 본인 줄 > indep 줄 > % 차이 작은 줄). link_all: 맞는 줄 전부.
      % 맞음: |차| ≤ 0.01, 또는 사진 값을 주장의 소수 자리로 반올림·버림한 값이 같다 (0.87 → "0.9퍼"·"0.8퍼"). 정수 %는 |차| ≤ 0.01만
      점수 맞음: |차| ≤ 0.1억, 또는 주장 자리로 반올림·버림이 같다 (254.7억 → "254억"·"255억"). 둘 다 점수가 있을 때만 본다
      link_by: pct+score(둘 다 맞음) · pct(사진이나 글에 점수 없음) · pct_only(소수 둘째 자리 %가 같고 점수만 다름 — 글의 점수가
      딴 이야기인 경우, "200억 못넘겼네") · score(점수만 있거나 정수 %인 주장이 점수로 맞음). % 한 자리 일치에 점수가 다르면 잇지 않는다
  - new: keep이고 link_oi가 없음 — 사진 관측과 따로 세는 새 정보.
표준 출력에 요약.
"""
import argparse
import collections
import datetime as dt
import json
import math
import os
import re


def jl(p):
    out = []
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                out.append(json.loads(l))
    return out


def exact(r, code_scores=False):
    """server_classify.py의 exact()와 같다. code_scores면 코드 OCR 줄(score_txt 없음)의 11자리 점수도 지문으로 쓴다."""
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


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S") if s else None


def c_start(season):
    if season.get("c_start"):
        return season["c_start"]
    d = ts(season["start"]).date() + dt.timedelta(days=2)
    return f"{d} 05:00:00"


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def compat(a, b):
    """두 줄의 값이 모순 없는가: 둘 다 있는 %·순위·점수가 같다."""
    if a["pct"] is not None and b["pct"] is not None and abs(a["pct"] - b["pct"]) > 0.005:
        return False
    if a["rank"] is not None and b["rank"] is not None and a["rank"] != b["rank"]:
        return False
    if a["score"] is not None and b["score"] is not None and abs(a["score"] - b["score"]) >= 1e7:
        return False
    return True


def decimals(v):
    for d in range(3):
        if abs(round(v, d) - v) < 1e-9:
            return d
    return 3


def num_match(claim, val, tol):
    """claim(글에 적은 값) ~ val(사진 값): |차| ≤ tol, 또는 val을 claim 자리로 반올림·버림한 값이 같다."""
    if abs(claim - val) <= tol + 1e-9:
        return True
    d = decimals(claim)
    f = 10 ** d
    return abs(round(val, d) - claim) < 1e-9 or abs(math.floor(val * f + 1e-9) / f - claim) < 1e-9


def link_obs(obs, sha, code_scores, shot_gap):
    n = len(obs)
    img_rows = collections.defaultdict(list)
    for i, r in enumerate(obs):
        img_rows[r["img"]].append(i)

    # 1) 같은 사진 재게시: dup_of 사진의 같은 row
    dup_canon = [None] * n
    for i, r in enumerate(obs):
        t = r.get("dup_of")
        if t:
            m = [j for j in img_rows.get(t, []) if obs[j]["row"] == r["row"]]
            dup_canon[i] = m[0] if m else None
    canon = [dup_canon[i] if dup_canon[i] is not None else i for i in range(n)]

    def cimg(i):   # 같은 사진 덩어리 (sha1, 없으면 img)
        return sha.get(obs[i]["img"], obs[i]["img"])

    # 2) 같은 사람: 지문 → 글의 본인 줄 → 줄 하나
    fp = [exact(r, code_scores) for r in obs]
    uf = UF()
    first = {}
    for i in range(n):
        uf.find(i)
        if canon[i] != i:
            uf.union(i, canon[i])
        if fp[i] is not None:
            if fp[i] in first:
                uf.union(i, first[fp[i]])
            else:
                first[fp[i]] = i
    has_fp = collections.defaultdict(bool)
    for i in range(n):
        if fp[i] is not None:
            has_fp[uf.find(i)] = True
    basis = [None] * n
    for i in range(n):
        if has_fp[uf.find(i)]:
            basis[i] = "fp"
    post_rows = collections.defaultdict(list)
    for i, r in enumerate(obs):
        if canon[i] == i:
            post_rows[(r["site"], str(r["no"]))].append(i)
    for key, rows in post_rows.items():
        strong = {uf.find(i) for i in rows if obs[i]["self"] and basis[i] == "fp"}
        weak = [i for i in rows if obs[i]["self"] and basis[i] is None]
        if not weak:
            continue
        if len(strong) == 1:
            s = next(iter(strong))
            for i in weak:
                uf.union(i, s)
                basis[i] = "post_one"
        else:
            for i in weak[1:]:
                uf.union(i, weak[0])
            for i in weak:
                basis[i] = "post_self"
    for i in range(n):
        if basis[i] is None and canon[i] == i:
            basis[i] = "row"
    for i in range(n):   # 재게시 줄은 원본 줄의 근거를 따른다
        if canon[i] != i and basis[i] is None:
            basis[i] = basis[canon[i]]
    comp = collections.defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(i)
    person = [None] * n
    pbasis = [None] * n
    for root, rows in comp.items():
        b = {basis[i] for i in rows}
        tag = "P" if "fp" in b else ("T" if "post_self" in b else "R")
        name = f"{tag}{min(rows)}"
        for i in rows:
            person[i] = name
            pbasis[i] = basis[i]

    # 3) 같은 시점
    # 사진 덩어리마다 사람 → 합친 값, 지문 사람 명단
    img_person = collections.defaultdict(dict)
    img_strong = collections.defaultdict(set)
    img_shot = {}
    for i, r in enumerate(obs):
        c = cimg(i)
        v = img_person[c].setdefault(person[i], {"pct": None, "rank": None, "score": None})
        for k in ("pct", "rank", "score"):
            if v[k] is None and r[k] is not None:
                v[k] = r[k]
        if person[i].startswith("P"):
            img_strong[c].add(person[i])
        if r.get("shot_t") and c not in img_shot:
            img_shot[c] = ts(r["shot_t"])
    memo = {}

    def same_screen(a, b):
        if a == b:
            return True
        k = (a, b) if a < b else (b, a)
        if k in memo:
            return memo[k]
        ok = True
        for p in set(img_person[a]) & set(img_person[b]):
            if not compat(img_person[a][p], img_person[b][p]):
                ok = False
                break
        sa, sb = img_strong[a], img_strong[b]
        if ok and len(sa) >= 2 and len(sb) >= 2 and not (sa <= sb or sb <= sa):
            ok = False
        if ok and a in img_shot and b in img_shot and abs((img_shot[a] - img_shot[b]).total_seconds()) > shot_gap * 60:
            ok = False
        memo[k] = ok
        return ok

    def t_of(i):
        r = obs[i]
        return r.get("shot_t") or r.get("post_t") or ""

    # 같은 시점은 이행적으로 잇되, 한 시점 안의 어느 두 줄도 값이 모순되면 안 된다 (2026-10-16 고침).
    # 전에는 쌍마다 union만 해서 % 없는 줄을 거쳐 서로 다른 % 줄이 한 시점으로 묶였다
    # (S41 P214: 0.43·0.56·1.15·1.24·1.40%, P506: 같은 사진의 1.07·1.48%).
    # 후보 쌍(같은 사진, 또는 값 모순 없는 같은 화면)을 같은 사진 먼저 → 시각 차 작은 순 → 줄 번호 순으로 보며,
    # 두 묶음의 모든 줄 쌍이 모순 없을 때만 합친다. 같은 사진이라도 값이 다르면(집계 화면의 한 사람 두 기록 등) 다른 시점이다.
    muf = UF()
    by_person = collections.defaultdict(list)
    for i in range(n):
        by_person[person[i]].append(i)

    def tsec(i):
        t = ts(t_of(i)) if t_of(i) else None
        return t.timestamp() if t else None

    for p, rows in by_person.items():
        edges = []
        for x in range(len(rows)):
            i = rows[x]
            muf.find(i)
            for y in range(x):
                j = rows[y]
                if not compat(obs[i], obs[j]):
                    continue
                ci, cj = cimg(i), cimg(j)
                if ci == cj or same_screen(ci, cj):
                    a, b = tsec(i), tsec(j)
                    gap = abs(a - b) if a is not None and b is not None else float("inf")
                    edges.append((ci != cj, gap, min(i, j), max(i, j)))
        if not edges:
            continue
        members = {i: [i] for i in rows}
        for _, _, i, j in sorted(edges):
            ri, rj = muf.find(i), muf.find(j)
            if ri == rj:
                continue
            if all(compat(obs[a], obs[b]) for a in members[ri] for b in members[rj]):
                merged = members.pop(ri) + members.pop(rj)
                muf.union(ri, rj)
                members[muf.find(ri)] = merged

    moments = collections.defaultdict(list)
    for i in range(n):
        moments[muf.find(i)].append(i)
    rep = [None] * n
    n_rep = [None] * n
    rep_oi = [None] * n
    for p, rows in by_person.items():
        ms = {}
        for i in rows:
            ms.setdefault(muf.find(i), []).append(i)
        order = sorted(ms.values(), key=lambda m: (min(t_of(i) for i in m), min(m)))
        for k, m in enumerate(order, 1):
            best = min(m, key=lambda i: (canon[i] != i, obs[i].get("season") != "in", obs[i]["pct"] is None,
                                         obs[i]["score"] is None,
                                         not obs[i].get("shot_t"), i))
            for i in m:
                rep[i], n_rep[i], rep_oi[i] = k, len(order), best

    out = []
    for i, r in enumerate(obs):
        out.append({"oi": i, "site": r["site"], "no": r["no"], "img": r["img"], "row": r["row"],
                    "dup_canon": dup_canon[i], "person": person[i], "person_basis": pbasis[i],
                    "moment": f"{person[i]}#{rep[i]}", "rep": rep[i], "n_rep": n_rep[i],
                    "indep": rep_oi[i] == i, "same_as": None if rep_oi[i] == i else rep_oi[i],
                    "fp": fp[i] is not None, "self": r["self"], "screen": r["screen"], "rank": r["rank"],
                    "pct": r["pct"], "score": r["score"]})
    return out


NOW = {None, "지금", "현재", "방금"}


def pick_kind(c, mode):
    if mode == "rule":
        return c.get("kind_rule")
    if mode == "either" and "self" in (c.get("kind_rule"), c.get("kind_haiku")):
        return "self"
    return c.get("kind")


def link_claims(claims, obs, links, season, kind_mode="kind"):
    cs = c_start(season)
    post_obs = collections.defaultdict(list)
    for i, r in enumerate(obs):
        post_obs[(r["site"], str(r["no"]))].append(i)
    seen = {}
    out = []
    for i, c in enumerate(claims):
        key = (c.get("site"), str(c["no"]))
        kind = pick_kind(c, kind_mode)
        drop, flag = [], []
        if kind != "self":
            drop.append(f"kind:{kind}")
        if c.get("pct") is None:
            drop.append("no_pct")
        elif c.get("pct") == int(c["pct"]):
            drop.append("int_pct")
        if (c.get("post_t") or "") < cs:
            drop.append("post_before")
        same_as_i = None
        if not drop:   # 남는 줄끼리만 비교한다. 시각 표현 없음·지금·현재·방금은 게시 시점으로 같게 본다
            t = c.get("time")
            sk = (key, c["pct"], c.get("score_eok"), None if t in NOW else t)
            if sk in seen:
                same_as_i = seen[sk]
                drop.append("same_claim")
            else:
                seen[sk] = i
        if c.get("q"):
            flag.append("q")
        agree = None if c.get("kind_haiku") is None else c.get("kind_rule") == c.get("kind_haiku")
        if agree is False:
            flag.append("kind_disagree")
        if c.get("score_from") == "post":
            flag.append("score_post")
        # 같은 글의 사진 관측과 잇기
        cand = []
        p, s = c.get("pct"), c.get("score_eok")
        for oi in post_obs.get(key, []):
            r = obs[oi]
            pm = p is not None and r["pct"] is not None and (
                num_match(p, r["pct"], 0.01) if p != int(p) else abs(p - r["pct"]) <= 0.01)   # 정수 %는 반올림·버림 없이
            sm = None
            if s is not None and r["score"] is not None:
                sm = num_match(s, r["score"] / 1e8, 0.1)
            if pm and sm is False and not (decimals(p) == 2 and abs(p - r["pct"]) <= 0.005):
                pm = False   # % 한 자리·반올림 일치는 점수가 다르면 다른 값으로 본다
            if pm:
                by = "pct+score" if sm else ("pct" if sm is None else "pct_only")
            elif sm and (p is None or p == int(p)):
                by = "score"   # 점수만 있거나 정수 %(구간 이름)와 점수: 점수로 잇는다
            else:
                continue
            cand.append((by == "score", not r["self"], not links[oi]["indep"], abs(p - r["pct"]) if pm else 0, oi, by))
        cand.sort()
        out.append({"i": i, "site": c.get("site"), "no": c["no"], "keep": not drop, "drop": drop, "flag": flag,
                    "kind": kind, "kind_agree": agree, "pct": p, "score_eok": s, "time": c.get("time"),
                    "same_as_i": same_as_i,
                    "link_oi": cand[0][4] if cand else None, "link_by": cand[0][5] if cand else None,
                    "link_all": [x[4] for x in cand],
                    "new": not drop and not cand})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--code-scores", action=argparse.BooleanOptionalAction, default=True,
                    help="코드 OCR 줄(score_txt 없음)의 점수도 지문으로 쓴다 (기본 켬, 2026-10-16. server_classify.py와 같은 기준)")
    ap.add_argument("--shot-gap", type=float, default=10, help="같은 화면으로 볼 촬영 시각 차이 상한(분)")
    ap.add_argument("--kind", choices=["kind", "rule", "either"], default="kind",
                    help="글 주장 종류: kind(Haiku, 없으면 규칙 — 기본) · rule(규칙만) · either(둘 중 하나라도 self면 self, 민감도 확인용)")
    a = ap.parse_args()
    st = a.state
    obs = jl(os.path.join(st, "obs.jsonl"))
    claims = jl(os.path.join(st, "claims.jsonl"))
    season = json.load(open(os.path.join(st, "season.json"), encoding="utf-8"))
    sha = {r["id"]: r["sha1"] for r in jl(os.path.join(st, "route.jsonl")) if r.get("sha1")}

    links = link_obs(obs, sha, a.code_scores, a.shot_gap)
    cl = link_claims(claims, obs, links, season, a.kind)
    for name, rows in (("post_link.jsonl", links), ("post_claims.jsonl", cl)):
        with open(os.path.join(st, name), "w", encoding="utf-8") as f:
            for o in rows:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # 요약: 모든 줄과 핵심 줄(server_classify의 core: 시즌 안 · % 있음 · 트래커 아님)
    def counts(ix):
        L = [links[i] for i in ix]
        return {"rows": len(L), "no_dup": sum(1 for x in L if x["dup_canon"] is None),
                "moments": len({x["moment"] for x in L}), "persons": len({x["person"] for x in L}),
                "persons_fp": len({x["person"] for x in L if x["person"].startswith("P")}),
                "persons_multi_moment": len({x["person"] for x in L if x["n_rep"] > 1})}
    core = [i for i, r in enumerate(obs) if r.get("season") == "in" and r.get("pct") is not None
            and r.get("screen") != "tracker"]
    fp_multi = collections.defaultdict(set)
    for i, r in enumerate(obs):
        if links[i]["fp"] and r["dup_of"] is None:
            fp_multi[links[i]["person"]].add((r["site"], str(r["no"])))
    selfpct = [x for x, c in zip(cl, claims) if c.get("kind") == "self" and c.get("pct") is not None]
    summ = {
        "all": counts(range(len(obs))), "core": counts(core),
        "person_basis": dict(collections.Counter(x["person_basis"] for x in links)),
        "fp_persons_in_2plus_posts": sum(1 for v in fp_multi.values() if len(v) >= 2),
        "claims": len(cl), "claims_keep": sum(x["keep"] for x in cl),
        "claims_keep_linked": sum(1 for x in cl if x["keep"] and x["link_oi"] is not None),
        "claims_new": sum(x["new"] for x in cl),
        "self_pct": len(selfpct), "self_pct_noscore": sum(1 for x in selfpct if x["score_eok"] is None),
        "self_pct_drop": dict(collections.Counter(d for x in selfpct for d in x["drop"])),
        "link_by": dict(collections.Counter(x["link_by"] for x in cl if x["keep"])),
    }
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
