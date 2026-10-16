"""할 일 1.9 정답 곡선: 시즌의 서버별 최종 곡선(최종 점수 → 최종 %)을 근거 등급과 함께 만든다 (모델 0, 요청 0).

python3 -I tools/solo_raid/truth.py STATE [--table FILE] [--exclude FILE | --no-exclude] [--c-hours 6] [--out-dir DIR]

입력 (읽기만 한다)
- STATE/season.json        `name`·`end`·`c_start`(없으면 start + 2일 05:00)
- STATE/obs.jsonl          사진 관측. 열쇠 oi = 줄 번호(0부터), 확인용 site·img·row
- STATE/server.jsonl       obs와 같은 줄 순서. 서버는 이 파일의 `server`만 쓴다 (점수·% 크기로 서버를 정하지 않는다)
- STATE/truth_table.json   손 입력: 커뮤니티 집계표(출처 파일·줄), 지난 시즌 화면 수동 제외, 공지 기준 시각. 없으면 표 비교만 빠진다
- STATE/post_exclude.jsonl 있으면 그 제외(exclude)를 따른다 (--exclude로 바꿈, --no-exclude면 아래 자체 거르기). 2026-10-16

근거 등급 (최종값 쪽: A·A2·D, 비교용: B·C)
- A  서버 있음 · 종료 후 촬영 근거: 첨부 이름·epoch 시각이 `end` 뒤, 또는 화면이 season_end(RAID RESULT)이고 글이 `end` 뒤
- A2 서버 있음 · 종료 후 게시, 촬영 시각 없음 (진행 중에 찍어 두었다 나중에 올린 화면일 수 있다)
- B  커뮤니티 집계표 (사용자 집계, 공식 확정값 아님)
- C  서버 있음 · 종료 전 마지막 --c-hours 시간 안에 찍힌 화면 (첨부 이름·epoch 시각). 최종값이 아니다
- D  서버 미상 · A나 A2의 시각 조건을 채운 화면. 곡선에 넣지 않고 따로 둔다

거르는 규칙
- 두 방식 모두: 화면 rank_list·my_rank·result·season_end, % 0 초과 100 이하, 점수 있음 (보상 구간 표·점수 가린 줄은 여기서 빠진다),
  truth_table.json `manual_exclude`의 사진은 뺀다
- post_exclude.jsonl이 있으면 (기본): 그 줄의 exclude가 비지 않았으면 뺀다 (`px:사유`). flag `season_unsure_mixed`도 뺀다
  (핵심 관측 표 core.jsonl의 기본 집합 규칙과 같음). 나머지 flag는 점에 옮겨 적는다. 아래 시즌·문맥·메모 규칙은 쓰지 않는다.
  자체 거르기와 달라지는 것(S41): 문맥 mixed·prev인데 글·수동 근거로 이번 시즌이 된 4줄(oi 2513·2561·2701·2718)이 들어간다
  (모두 서버 미상이라 D 점, 곡선 영향 없음)
- flag `best_record`(post_exclude: 메모에 「나의 최고 기록」)가 붙은 점은 그 %가 기록을 세운 때 값일 수 있어 곡선에 넣지 않고 표시만 한다
- post_exclude.jsonl이 없거나 --no-exclude면 자체 거르기 (post_exclude가 생기기 전 규칙):
  - 판독 메모에 보상표·「선수 기록 아님」·명예의 전당이 있으면 뺀다. mock이면 뺀다
  - 시즌 `in`만. `unsure`는 S41 묶음 이음이 있을 때만 넣고 표시한다
  - 문맥 mixed·prev·other 글은 종료 뒤 첨부 시각이 있거나 S41 묶음 이음이 있을 때만 넣는다
    S41 묶음 이음: 그 사진의 지문 묶음(server.jsonl cluster)에 다른 글의 사진으로, 문맥 cur·시즌 in·첨부 시각이 챌린지 기간 안인 것이 있다
  (manual_exclude S41: S37 한섭 순위표 사진 4장 — 묶음 이름은 server_classify 설정에 따라 c9 또는 c12)
- 같은 사진 재게시(dup_of)는 원본 사진 하나로 센다. 정확한 점수(10자리 이상)가 같은 줄은 같은 사람의 같은 최종 기록이라 한 점으로 묶는다

점검 (표시만 하고 지우지 않는다)
- 단조성: 같은 서버 최종 점(A·A2)에서 점수가 높은데 %도 높은 쌍. 위반이 많은 점부터 하나씩 빼며(동률이면 A2 먼저) 위반이 없어질 때까지 `mono_viol`을 붙인다
- 같은 정확 점수의 최종 줄끼리 %가 다르면 `pct_spread`
- 공지 기준 시각(checkpoints) 앞뒤로 같은 점수(정확 점수, 없으면 0.01억+같은 %)가 다시 보인 경우의 % 차이
- 종료 전 관측(시즌 in, 문맥 cur)과 최종 점의 정확 점수가 같으면 (t시점 %, 최종 %) 쌍을 남긴다 → 2단계 「% 변환」 확인용

출력
- STATE/truth.json          등급 정의·거른 수·서버별 곡선 점(pct 순)·단조성·관측 없는 % 구간·집계표 비교·C 비교·쌍·정산 점검.
                            `exclude_source`: 거르기 방식 (post_exclude 파일 경로 또는 "self")
- STATE/truth_points.jsonl  점 하나에 한 줄 (A·A2·B·C·D). 각 점의 src 줄 목록(oi·site·img·row·시각 근거)
- 표준 출력에 요약
"""
import argparse
import collections
import datetime as dt
import json
import math
import os
import re
import statistics

SCREENS = ("rank_list", "my_rank", "result", "season_end")
NOTE_BAD = re.compile(r"보상\s?표|보상 구간|선수 기록 아님|명예의 전당")
FMT = "%Y-%m-%d %H:%M:%S"
GRADE_ORDER = {"A": 0, "A2": 1, "D": 2}
EPS_PCT, EPS_EOK = 0.005, 0.005   # 단조성 허용: % 0.005, 점수 0.005억 (반올림 차이)


def jl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
    return out


def ts(s):
    return dt.datetime.strptime(s, FMT) if s else None


def c_start(season):
    if season.get("c_start"):
        return season["c_start"]
    d = dt.date.fromisoformat(season["start"][:10]) + dt.timedelta(days=2)
    return f"{d.isoformat()} 05:00:00"


def exact(o):
    """정확 점수: 10자리 이상이고 끝 6자리가 0이 아닌 값. server_classify.py는 score_txt 자릿수를 보지만,
    코드 판독(src code)은 score_txt 없이 11자리를 그대로 남기므로 여기서는 그것도 정확 점수로 본다."""
    s = o.get("score")
    if not isinstance(s, (int, float)) or s != int(s):
        return None
    s = int(s)
    digits = re.sub(r"\D", "", o.get("score_txt") or "")
    if s < 10 ** 9 or s % 10 ** 6 == 0 or (len(digits) < 10 and o.get("src") != "code"):
        return None
    return s


def interp_log(pts, pct):
    """비교용 선형 보간: (log %, 점수) 꺾은선. pts는 (pct, eok) pct 오름차순. 범위 밖이면 None."""
    for (p0, s0), (p1, s1) in zip(pts, pts[1:]):
        if p0 <= pct <= p1:
            if p1 == p0:
                return (s0 + s1) / 2
            w = (math.log(pct) - math.log(p0)) / (math.log(p1) - math.log(p0))
            return s0 + w * (s1 - s0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--table", help="기본 STATE/truth_table.json")
    ap.add_argument("--c-hours", type=float, default=6.0, help="C 등급: 종료 전 몇 시간 (기본 6)")
    ap.add_argument("--exclude", help="post_exclude 출력 (기본 STATE/post_exclude.jsonl, 있으면 따른다)")
    ap.add_argument("--no-exclude", action="store_true", help="post_exclude를 쓰지 않고 자체 거르기만 한다")
    ap.add_argument("--out-dir", help="출력 폴더 (기본 STATE)")
    a = ap.parse_args()
    st = a.state
    out_dir = a.out_dir or st
    season = json.load(open(os.path.join(st, "season.json"), encoding="utf-8"))
    END, C0 = season["end"], c_start(season)
    c_lo = (ts(END) - dt.timedelta(hours=a.c_hours)).strftime(FMT)
    obs = jl(os.path.join(st, "obs.jsonl"))
    srv = jl(os.path.join(st, "server.jsonl"))
    if len(obs) != len(srv) or any((o["img"], o["row"]) != (s["img"], s["row"]) for o, s in zip(obs, srv)):
        raise SystemExit("obs.jsonl와 server.jsonl 줄이 맞지 않는다 — server_classify.py를 다시 돌린다")
    tpath = a.table or os.path.join(st, "truth_table.json")
    tin = json.load(open(tpath, encoding="utf-8")) if os.path.exists(tpath) else {}
    if tin and tin.get("season") != season.get("name"):
        print(f"경고: {tpath}의 시즌 {tin.get('season')} ≠ {season.get('name')} — 손 입력을 쓰지 않는다")
        tin = {}
    man_ex = {x["img"]: x["reason"] for x in tin.get("manual_exclude", [])}
    px = None
    px_path = a.exclude or os.path.join(st, "post_exclude.jsonl")
    if not a.no_exclude and os.path.exists(px_path):
        px = jl(px_path)
        if len(px) != len(obs) or any((o["img"], o["row"]) != (x["img"], x["row"]) for o, x in zip(obs, px)):
            raise SystemExit(f"{px_path}가 obs.jsonl과 줄이 맞지 않는다 — post_exclude.py를 다시 돌린다")

    # S41 묶음 이음: 챌린지 기간 안에 찍힌(첨부 시각) 문맥 cur·시즌 in 사진이 있는 묶음 → 그 글 번호 집합
    linked = collections.defaultdict(set)
    for o, s in zip(obs, srv):
        if (s.get("cluster") and o["season"] == "in" and o.get("ctx") == "cur" and o.get("shot_t")
                and C0 <= o["shot_t"] <= END):
            linked[s["cluster"]].add(str(o["no"]))

    def is_linked(o, s):
        c = s.get("cluster")
        return bool(c and linked.get(c, set()) - {str(o["no"])})

    # 같은 사진 재게시(dup_of)의 원본 글 시각: 원본이 종료 전에 올라왔으면 재게시도 종료 전 화면이다
    first_post = {}
    for o in obs:
        if o.get("post_t") and (o["img"] not in first_post or o["post_t"] < first_post[o["img"]]):
            first_post[o["img"]] = o["post_t"]

    def self_filter(oi, o, s, ph, tb, flags):
        """post_exclude가 없을 때의 자체 거르기. 빼면 (None, None), 남기면 (S41 묶음 이음, 문맥). flags에 표시를 더한다."""
        def drop(k):
            excl[ph + ":" + k] += 1
            if ph == "final":
                excl_oi[ph + ":" + k].append(oi)
            return None, None
        if o.get("mock"):
            return drop("mock")
        if NOTE_BAD.search(o.get("note") or ""):
            return drop("note_not_player")
        lk = is_linked(o, s)
        if o["season"] == "unsure":
            if not lk:
                return drop("season_unsure")
            flags.append("season_unsure_linked")
        elif o["season"] != "in":
            excl[ph + ":season_" + o["season"]] += 1
            return None, None
        ctx = o.get("ctx")
        if ctx not in ("cur", None):
            if tb == "shot" or lk:
                flags.append(f"ctx_{ctx}")
            else:
                return drop(f"ctx_{ctx}_nolink")
        if o.get("verify") == "differ":
            flags.append("verify_differ")
        return lk, ctx

    excl = collections.Counter()
    excl_oi = collections.defaultdict(list)   # 종료 후 줄(final:)만 oi를 남긴다 — post_exclude와 맞춰 보기용
    final_rows, c_rows, pre_rows = [], [], []
    for oi, (o, s) in enumerate(zip(obs, srv)):
        if o.get("screen") not in SCREENS or o.get("pct") is None:
            continue
        pct = o["pct"]
        shot, post = o.get("shot_t"), o.get("post_t")
        if shot and shot > END:
            tb = "shot"
        elif not shot and post and post > END and o["screen"] == "season_end":
            tb = "screen"
        elif not shot and post and post > END and first_post.get(o.get("dup_of") or "", post) > END:
            tb = "post"
        elif shot and shot >= c_lo:
            tb = "pre_shot"      # C 후보 (종료 전 마지막 c_hours 안)
        else:
            tb = "pre"
        ph = "final" if tb in ("shot", "screen", "post") else "pre"
        if not (0 < pct <= 100):
            excl[ph + ":pct_range"] += 1
            if ph == "final":
                excl_oi[ph + ":pct_range"].append(oi)
            continue
        if not o.get("score"):
            excl[ph + ":no_score"] += 1
            if ph == "final":
                excl_oi[ph + ":no_score"].append(oi)
            continue
        if o["img"] in man_ex:
            excl[ph + ":manual_" + man_ex[o["img"]]] += 1
            continue
        flags = []
        if px is not None:
            x = px[oi]
            why = ["px:" + "+".join(x["exclude"])] if x["exclude"] else (
                ["px_flag:season_unsure_mixed"] if "season_unsure_mixed" in x["flag"] else [])
            if why:
                excl[ph + ":" + why[0]] += 1
                if ph == "final":
                    excl_oi[ph + ":" + why[0]].append(oi)
                continue
            flags = list(x["flag"])
            ctx = o.get("ctx")
            cur_ok = True   # post_exclude가 남긴 줄은 이번 시즌 (문맥 prev·other·mixed는 사진·글 근거가 있을 때만 남는다)
        else:
            lk, ctx = self_filter(oi, o, s, ph, tb, flags)
            if lk is None:
                continue
            cur_ok = ctx == "cur"
        r = {"oi": oi, "site": o["site"], "img": o["img"], "row": o["row"], "no": str(o["no"]),
             "uimg": o.get("dup_of") or o["img"], "server": s.get("server"), "basis": s.get("basis"),
             "cluster": s.get("cluster"), "pct": pct, "score": o["score"], "exact": exact(o),
             "rank": o.get("rank"), "screen": o["screen"], "src": o.get("src"), "t_basis": tb,
             "shot_t": shot, "post_t": post, "ctx": ctx, "self": o.get("self"), "flags": flags}
        if tb in ("shot", "screen", "post"):
            r["grade"] = ("A" if tb in ("shot", "screen") else "A2") if r["server"] else "D"
            final_rows.append(r)
        else:
            if tb == "pre_shot" and r["server"] and cur_ok:
                r["grade"] = "C"
                c_rows.append(r)
            if cur_ok and r["exact"]:
                pre_rows.append(r)

    def group(rows, grade_fn, by_pct=False):
        """서버별로 정확 점수(없으면 0.01억+%)로 묶어 점을 만든다. by_pct면 %가 다른 줄은 따로 둔다 (종료 전 C)."""
        g = collections.OrderedDict()
        for r in rows:
            k = (r["server"], ("x", r["exact"]) if r["exact"] else ("r", round(r["score"] / 1e6), r["pct"]))
            if by_pct and k[1][0] == "x":
                k = (k[0], ("x", r["exact"], r["pct"]))
            g.setdefault(k, []).append(r)
        pts = []
        for (server, key), mem in g.items():
            grade = grade_fn(mem)
            best = [m for m in mem if m["grade"] == grade]
            pcts = sorted({m["pct"] for m in mem})
            pct = statistics.median([m["pct"] for m in best])
            score = statistics.median([m["score"] for m in best])
            flags = sorted({f for m in mem for f in m["flags"]})
            if len(pcts) > 1:
                flags.append("pct_spread")
            ranks = sorted({m["rank"] for m in mem if m["rank"]})
            pts.append({
                "server": server, "pct": pct, "score": round(score / 1e8, 1), "score_raw": round(score / 1e8, 4),
                "score_int": key[1] if key[0] == "x" else None, "rank": ranks[0] if len(ranks) == 1 else (ranks or None),
                "grade": grade, "src": [m["oi"] for m in mem], "n": len({m["uimg"] for m in mem}),
                "n_posts": len({(m["site"], m["no"]) for m in mem}), "pcts": pcts,
                "t_basis": sorted({m["t_basis"] for m in mem}), "flags": flags,
                "src_detail": [{k: m[k] for k in ("oi", "site", "img", "row", "t_basis", "shot_t", "post_t", "ctx",
                                                  "screen", "src", "basis", "cluster")} for m in mem]})
        return pts

    final_pts = group(final_rows, lambda mem: min((m["grade"] for m in mem), key=GRADE_ORDER.get))
    # 종료 전에 같은 정확 점수·같은 %가 이미 보였으면 표시 (A2면 진행 중 화면을 나중에 올린 것일 수 있다)
    pre_seen = {(r["exact"], r["pct"]) for r in pre_rows}
    for p in final_pts:
        if p["score_int"] and any((p["score_int"], x) in pre_seen for x in p["pcts"]):
            p["flags"].append("same_pct_pre_end")
    c_pts = group(c_rows, lambda mem: "C", by_pct=True)
    known = [p for p in final_pts if p["grade"] in ("A", "A2")]
    # D 점의 정확 점수가 서버 있는 점과 같으면 적어만 둔다 (서버는 server.jsonl만 쓰므로 붙이지 않는다)
    kx = {p["score_int"]: p for p in known if p["score_int"]}
    for p in final_pts:
        if p["grade"] == "D" and p["score_int"] in kx:
            q = kx[p["score_int"]]
            p["same_exact_as"] = {"server": q["server"], "pct": q["pct"], "src": q["src"]}
            p["flags"].append("same_exact_as_known")

    # 단조성 (서버별, A·A2)
    mono = {}
    by_srv = collections.defaultdict(list)
    for p in known:
        by_srv[p["server"]].append(p)

    def viol(p, q):
        return ((p["score_raw"] > q["score_raw"] + EPS_EOK and p["pct"] > q["pct"] + EPS_PCT)
                or (q["score_raw"] > p["score_raw"] + EPS_EOK and q["pct"] > p["pct"] + EPS_PCT))

    for server, ps in by_srv.items():
        pairs = [(p, q) for i, p in enumerate(ps) for q in ps[i + 1:] if viol(p, q)]
        best = [p for p in ps if "best_record" in p["flags"]]   # 곡선에 넣지 않으므로 단조성 정리에서도 먼저 뺀다
        alive, removed = [p for p in ps if "best_record" not in p["flags"]], []
        while True:
            cnt = collections.Counter()
            for i, p in enumerate(alive):
                for q in alive[i + 1:]:
                    if viol(p, q):
                        cnt[id(p)] += 1
                        cnt[id(q)] += 1
            if not cnt:
                break
            worst = max(alive, key=lambda p: (cnt[id(p)], GRADE_ORDER[p["grade"]], -p["n"]))
            alive.remove(worst)
            removed.append(worst)
            worst["flags"].append("mono_viol")
        mono[server] = {"points": len(ps), "violating_pairs": len(pairs),
                        "violating_pairs_without_best_record": sum(1 for p, q in pairs if p not in best and q not in best),
                        "best_record": [{"pct": p["pct"], "score": p["score"], "grade": p["grade"], "src": p["src"]}
                                        for p in best],
                        "flagged": [{"pct": p["pct"], "score": p["score"], "grade": p["grade"], "src": p["src"]}
                                    for p in removed]}

    def off_curve(p):
        return "mono_viol" in p["flags"] or "best_record" in p["flags"]

    # 곡선: A + 표시 없는 A2 (mono_viol·best_record 제외). A만 곡선도 따로
    curves, coverage = {}, {}
    for server, ps in sorted(by_srv.items()):
        good = sorted((p for p in ps if not off_curve(p)), key=lambda p: (p["pct"], -p["score_raw"]))
        curves[server] = {
            "all": [[p["pct"], p["score"], p["grade"], p["n"]] for p in good],
            "A_only": [[p["pct"], p["score"]] for p in good if p["grade"] == "A"]}
        pc = sorted({p["pct"] for p in good})
        gaps = [[x, y] for x, y in zip(pc, pc[1:]) if y / x > 1.5 and y - x > 0.5]
        coverage[server] = {"n_points": len(good), "pct_min": pc[0] if pc else None, "pct_max": pc[-1] if pc else None,
                            "score_max": max((p["score"] for p in good), default=None),
                            "score_min": min((p["score"] for p in good), default=None),
                            "gaps_pct": gaps,
                            "no_obs": ([f"< {pc[0]}%"] if pc else []) + ([f"> {pc[-1]}%"] if pc else [])}

    # 집계표 B 점과 비교
    b_pts, tcmp = [], []
    trows = tin.get("table", {}).get("rows", [])
    tsrc = tin.get("table", {}).get("transcribed_from", "truth_table.json")
    for t in trows:
        b_pts.append({"server": t["server"], "pct": t["pct"], "score": round(t["score_eok"], 1),
                      "score_raw": t["score_eok"], "score_int": None, "rank": None, "grade": "B",
                      "src": [f"{tsrc.split(' ')[0]}:{t['line']}"], "n": 1, "n_posts": 1, "pcts": [t["pct"]],
                      "t_basis": ["table"], "flags": []})
        pts = [(p[0], p[1]) for p in curves.get(t["server"], {}).get("all", [])]
        exact_pts = [p for p in by_srv.get(t["server"], []) if not off_curve(p)]
        near = sorted((p for p in exact_pts if abs(p["pct"] - t["pct"]) <= max(0.15, 0.05 * t["pct"])),
                      key=lambda p: p["pct"])
        lo = max((p for p in exact_pts if p["pct"] <= t["pct"]), key=lambda p: p["pct"], default=None)
        hi = min((p for p in exact_pts if p["pct"] >= t["pct"]), key=lambda p: p["pct"], default=None)
        ip = interp_log(sorted(set(pts)), t["pct"])
        cn = [p for p in c_pts if p["server"] == t["server"] and abs(p["pct"] - t["pct"]) <= max(0.15, 0.05 * t["pct"])]
        tcmp.append({
            "server": t["server"], "pct": t["pct"], "table": t["score_eok"],
            "near_final": [[p["pct"], p["score_raw"], p["grade"], p["src"]] for p in near],
            "near_score_range": [min(p["score_raw"] for p in near), max(p["score_raw"] for p in near)] if near else None,
            "bracket": [[lo["pct"], lo["score_raw"], lo["grade"]] if lo else None,
                        [hi["pct"], hi["score_raw"], hi["grade"]] if hi else None],
            "interp_loglin": round(ip, 2) if ip is not None else None,
            "table_minus_interp": round(t["score_eok"] - ip, 2) if ip is not None else None,
            "near_C": [[p["pct"], p["score_raw"], p["src"]] for p in sorted(cn, key=lambda p: p["pct"])]})

    # C 점: 같은 서버 최종 곡선에서 보간한 최종 점수와 비교 (C의 %는 그 시각 %)
    for p in c_pts:
        pts = [(x[0], x[1]) for x in curves.get(p["server"], {}).get("all", [])]
        ip = interp_log(sorted(set(pts)), p["pct"])
        p["final_curve_score_at_pct"] = round(ip, 2) if ip is not None else None

    # 같은 정확 점수: 종료 전 관측 → 최종 % (2단계 「% 변환」 확인용)
    fin_by_exact = {(p["server"], p["score_int"]): p for p in known if p["score_int"] and not off_curve(p)}
    pairs = []
    for r in pre_rows:
        f = fin_by_exact.get((r["server"], r["exact"])) if r["server"] else None
        if not f:
            continue
        t = r["shot_t"] or r["post_t"]
        pairs.append({"server": r["server"], "score": round(r["exact"] / 1e8, 2), "oi": r["oi"], "img": r["img"],
                      "screen": r["screen"],
                      "t": t, "t_kind": "shot" if r["shot_t"] else "post",
                      "h_before_end": round((ts(END) - ts(t)).total_seconds() / 3600, 2),
                      "pct_t": r["pct"], "pct_final": f["pct"], "final_grade": f["grade"], "final_src": f["src"],
                      "flags": r["flags"]})
    seen, uniq = set(), []
    for x, r in sorted(zip(pairs, [None] * len(pairs)), key=lambda z: z[0]["oi"]):
        k = (x["server"], x["score"], x["pct_t"], x["t"])
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    pairs = sorted(uniq, key=lambda x: (x["server"], x["score"], x["t"]))

    # 등급끼리: A2·C·D 점을 같은 서버의 A만 곡선에 보간해 점수 차이를 본다 (D는 서버가 없어 뺀다)
    agree = {}
    for server, ps in sorted(by_srv.items()):
        a_pts = sorted({(p["pct"], p["score_raw"]) for p in ps if p["grade"] == "A" and not off_curve(p)})
        rows = []
        for p in sorted([q for q in ps if q["grade"] == "A2"] + [q for q in c_pts if q["server"] == server],
                        key=lambda q: q["pct"]):
            ip = interp_log(a_pts, p["pct"])
            if ip is not None:
                rows.append({"grade": p["grade"], "pct": p["pct"], "score": p["score_raw"], "A_interp": round(ip, 2),
                             "diff": round(p["score_raw"] - ip, 2), "src": p["src"], "flags": p["flags"]})
        d2 = [abs(r["diff"]) for r in rows if r["grade"] == "A2" and not off_curve(r)]
        dc = [r["diff"] for r in rows if r["grade"] == "C"]
        agree[server] = {"A_points": len(a_pts), "A_pct_range": [a_pts[0][0], a_pts[-1][0]] if a_pts else None,
                         "A2_in_range": len(d2), "A2_absdiff_median": round(statistics.median(d2), 2) if d2 else None,
                         "A2_absdiff_max": round(max(d2), 2) if d2 else None,
                         "C_in_range": len(dc), "C_diff_median": round(statistics.median(dc), 2) if dc else None,
                         "rows": rows}

    # 공지 기준 시각 앞뒤: 최종 줄끼리 점수가 0.015억 안이면 (서버가 둘 다 있으면 같을 때만) %를 비교한다.
    # 같은 점수의 최종 %는 같아야 하므로 사람이 같은지는 따지지 않는다. 기준 시각 뒤에 기록 말소가 있었으면 % 차이가 보인다
    settle = []
    t_of = lambda m: m["shot_t"] or m["post_t"]
    for cp in tin.get("checkpoints", []):
        cut = cp["t"]
        bef = [m for m in final_rows if t_of(m) < cut]
        aft = [m for m in final_rows if t_of(m) >= cut]
        for m in aft:
            near = [b for b in bef if abs(b["score"] - m["score"]) <= 1.5e6
                    and (not (b["server"] and m["server"]) or b["server"] == m["server"])]
            if near:
                b = min(near, key=lambda b: abs(b["score"] - m["score"]))
                settle.append({"checkpoint": cut, "score_after": round(m["score"] / 1e8, 4),
                               "score_before": round(b["score"] / 1e8, 4), "same_exact": bool(m["exact"] and m["exact"] == b["exact"]),
                               "server": [b["server"], m["server"]], "pct_before": b["pct"], "pct_after": m["pct"],
                               "t_before": t_of(b), "t_after": t_of(m), "oi": [b["oi"], m["oi"]],
                               "diff": round(m["pct"] - b["pct"], 3)})

    grades = {
        "A": "서버 있음 · 종료 후 촬영 근거 (첨부 이름·epoch 시각 > end, 또는 season_end 화면이고 글 > end)",
        "A2": "서버 있음 · 종료 후 게시, 촬영 시각 없음 (진행 중 화면을 나중에 올렸을 수 있음)",
        "B": "커뮤니티 집계표 (사용자 집계·추정, 공식 확정값 아님)",
        "C": f"서버 있음 · 종료 전 {a.c_hours:g}시간 안 촬영 (첨부 시각). 최종값 아님, 비교용",
        "D": "서버 미상 · A·A2 시각 조건. 곡선에 넣지 않음"}
    all_pts = sorted(final_pts + b_pts + c_pts, key=lambda p: (p["server"] or "~", p["grade"], p["pct"]))
    summary = {
        "season": season.get("name"), "end": END, "c_start": C0, "c_window": [c_lo, END],
        "generated_by": "tools/solo_raid/truth.py", "table_input": os.path.relpath(tpath) if tin else None,
        "exclude_source": os.path.relpath(px_path) if px is not None else "self",
        "grades": grades,
        "excluded": dict(excl),
        "excluded_final_oi": dict(excl_oi),
        "rows": {"final_rows": len(final_rows), "C_rows": len(c_rows), "pre_exact_rows": len(pre_rows)},
        "points_by_grade": dict(collections.Counter((p["server"] or "?", p["grade"]) for p in all_pts).most_common()),
        "points_by_grade_str": {f"{k[0]}/{k[1]}": v for k, v in collections.Counter(
            (p["server"] or "?", p["grade"]) for p in all_pts).items()},
        "curves": curves, "coverage": coverage, "monotone": mono,
        "table_compare": tcmp,
        "C_points": [{k: p[k] for k in ("server", "pct", "score", "score_raw", "src", "final_curve_score_at_pct")}
                     for p in sorted(c_pts, key=lambda p: (p["server"], p["pct"]))],
        "D_points": [[p["pct"], p["score"], p["t_basis"], p["src"]] for p in sorted(
            (p for p in final_pts if p["grade"] == "D"), key=lambda p: p["pct"])],
        "grade_agreement": agree,
        "same_score_pairs": pairs,
        "checkpoint_check": settle,
        "interp_note": ("보간 함수는 정하지 않았다. 후보: (log %, 점수) 꺾은선 · 같은 축의 단조 3차(PCHIP) · 단조 회귀 후 꺾은선. "
                        "table_compare.interp_loglin은 비교용 꺾은선 값이다. 관측 없는 구간(coverage.gaps_pct·no_obs)에서는 보간값을 정답으로 쓰지 않는다."),
    }
    del summary["points_by_grade"]
    summary["points"] = [{k: p[k] for k in ("server", "pct", "score", "score_raw", "score_int", "rank", "grade", "src",
                                             "n", "n_posts", "pcts", "t_basis", "flags")} for p in all_pts]
    with open(os.path.join(out_dir, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(os.path.join(out_dir, "truth_points.jsonl"), "w", encoding="utf-8") as f:
        for p in all_pts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"truth: 최종 줄 {len(final_rows)} · 점 {summary['points_by_grade_str']} · 단조 위반 표시 "
          f"{ {k: len(v['flagged']) for k, v in mono.items()} } · 거름 {dict(excl)}")


if __name__ == "__main__":
    main()
