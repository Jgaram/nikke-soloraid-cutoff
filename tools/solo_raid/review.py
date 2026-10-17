"""솔로레이드 시즌 데이터 점검 목록 — 스크립트가 예외 후보를 고르고, 세션(AI)이 원본을 보고 판단한다 (2026-10-16, 모델 0, 요청 0).

python3 -I tools/solo_raid/review.py data/solo-raid/S{n}/state [--write]

후처리(post_all.sh)가 끝난 state를 읽어 「다시 볼 것」을 고른다. --write면 data/solo-raid/S{n}/review.json에 쓴다.
다시 돌려도 항목 id가 같으면 세션이 적은 판단(decision·note)을 그대로 둔다. 이번에 안 나온 항목은 gone: true로 남긴다.
판단은 [예외 처리 규칙](../../docs/solo-raid-backfill.md#예외-처리-규칙)대로 하고, 고칠 것은 S{n}/discard.json에 적는다.

고르는 것 (kind)
- truth_flag: 서버가 붙은 정답 점(truth_points, 등급 A·A2·C)에 강한 플래그(mono_viol·verify_unresolved·curve_outlier)가 있음. 점마다 하나.
- truth_flag_same_score: truth_flag 중 플래그가 mono_viol뿐이고 같은 서버·같은 정확 점수가 다른 사진에도 정답 점으로 있는 것은 한 항목으로 묶는다
  (같은 사람이 시각이 다른 화면을 올려 %만 다른 꼴, S19 「다음 시즌에 바꿀 것」 2026-10-17). 묶음의 사진 몇 장을 보고 판단한다.
- truth_flag_group: 약한 플래그(ctx_overridden·season_conflict·verify_differ 등, best_record·rank_over_200·rank_with_pct 빼고)는 플래그마다 한 항목으로 묶는다 (사진 목록).
  묶음은 표본 몇 장을 보고 판단해도 된다 (S38 season_conflict처럼 수가 많을 때).
- rank_server: 순위표 1~3위 정확 점수가 같은데 그 사진들 글의 텍스트 서버(text·fix)가 다름 → 소수 쪽 글 (3.2b S39 5881317 같은 경우).
- cluster_mixed: 지문 묶음의 앵커 글이 두 서버 이상 (server_summary clusters). 소수 쪽 앵커 글을 적는다.
- server_count_text: 서버 라벨이 붙은 글인데 본문·제목이 「n서버」「전서버」「타서버」처럼 여러 서버를 말함 (3.2 S40 6066145 같은 경우).
- cross_season: 이 시즌 기본 집합 줄(season in·core use, 2026-10-16부터)의 정확 점수가 주로 다른 시즌 obs.jsonl에 나옴 → 지난 시즌 기록·화면일 수 있음 (3.2 S40 6069623 같은 경우).
  점수의 「주인」은 그 점수가 가장 많이 나온 시즌이다. 주인이 다른 시즌이고 그 시즌 횟수가 이 시즌의 2배 이상일 때만 고른다
  (순위표 상위권 점수는 그 시즌에 수십~수백 번 나온다).
- rank1_low: 기본 집합 줄이 있는 사진의 순위표 1위 점수가 그보다 먼저 올라온 같은 서버 순위표 7위(글 2개 이상에 나온 점수) 최댓값의 0.95배보다 낮음
  → 그 시각 화면일 수 없다 — 지난 시즌·다른 서버·오래전 화면 (k위 점수는 시즌 안에서 줄지 않는다. 1위는 핵 점수가 지워져 쓰지 않는다. S21 2544611 「솔레 성장」 글, 2026-10-17 분석 ④). 사진마다 하나.
- outlier: post_outlier가 곡면에서 크게 벗어났다고 본 줄 (surface_high·surface_low).
- rank_over_200: 시즌 안(season in)·빼지 않은 줄인데 게임 화면 등수가 200을 넘음 (post_exclude flag rank_over_200). 사진마다 하나.
  게임은 200위까지만 등수다 (2026-10-16) — 원본에서 그 숫자가 무엇인지(배지 등), 같은 줄의 %·점수도 잘못 읽혔는지 본다.
- shaky (개발 시즌만): 정답 표에서 집계표와 정답 곡선이 2% 넘게 어긋남 (truth_grade 규칙).

시험 시즌(split.json test): 집계표(seasons.json 컷·server_cutoffs.csv)와 견주지 않는다 (shaky를 고르지 않음). review.json에 % ·점수 값을 쓰지 않는다
(이음 열쇠·종류만). 정답 점 내부 일관성(truth_flag)은 데이터 문제라 같은 규칙으로 고른다.
판 2(2026-10-17)부터 잠근 시험 시즌이 없어 이 분기는 쓰이지 않는다 — 모든 시즌을 개발 시즌 규칙으로 고른다. 새 시즌의 미래 시험도
점검(예외 처리)은 같고, 도구 고침만 자기 예측 채점 뒤로 미룬다 (docs/solo-raid-live.md 「3. 끝난 뒤」). 판 1 시험 6시즌의 옛 review.json은 값 칸이 비어 있다.
"""
import argparse
import collections
import glob
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import split as SP  # noqa: E402

DATA = os.path.join(ROOT, "data", "solo-raid")
MULTI_SRV = re.compile(r"[2-9]\s*서버|[2-9]\s*섭|[이삼사오]\s*서버|전\s?서버|전섭|모든\s?서버|타\s?서버|타섭|다른\s?서버|서버\s?별|서버마다")
RANK1_LOW = 0.95   # rank1_low 문턱 (2026-10-17)
KEEP = ("decision", "note", "by", "decided", "seen")   # seen: 이 항목을 판단하며 원본으로 본 사진 id 목록 (2026-10-16)


def jl(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def exact(r):
    """server_classify.exact와 같은 정확 점수 (코드 OCR 줄 포함)."""
    s = r.get("score")
    if not isinstance(s, (int, float)) or s != int(s):
        return None
    s = int(s)
    digits = re.sub(r"\D", "", r.get("score_txt") or "")
    if not digits and r.get("src") == "code":
        digits = str(s)
    if s < 10 ** 9 or len(digits) < 10 or s % 10 ** 6 == 0:
        return None
    return s


def collect(state):
    name = json.load(open(os.path.join(state, "season.json"), encoding="utf-8"))["name"]
    sp = SP.load_split()
    test = SP.is_test(name, sp)
    obs = jl(os.path.join(state, "obs.jsonl"))
    srv = {(r["img"], r["row"]): r for r in jl(os.path.join(state, "server.jsonl"))}
    posts = {str(p["no"]): p for p in jl(os.path.join(state, "posts_min.jsonl"))}
    items = []

    def add(kind, iid, detail, **kw):
        items.append(dict({"id": f"{kind}:{iid}", "kind": kind}, **kw, detail=detail))

    def val(x):
        return None if test else x

    # truth_flag (강한 플래그는 점마다, 약한 플래그는 묶음)
    STRONG = {"mono_viol", "verify_unresolved", "curve_outlier"}
    group = collections.defaultdict(list)
    oxs = {(r["img"], r["row"]): exact(r) for r in obs}
    strong_pts = []
    for t in jl(os.path.join(state, "truth_points.jsonl")):
        if not (t.get("server") and t.get("flags") and t.get("grade") in ("A", "A2", "C")):
            continue
        for d in t.get("src_detail", []):
            strong = sorted(STRONG & set(t["flags"]))
            if strong:
                strong_pts.append((t, d))
            else:
                for f in t["flags"]:
                    if f not in ("best_record", "rank_over_200", "rank_with_pct"):   # rank_over_200은 아래 사진별 항목으로, rank_with_pct는 후보 아님
                        group[f].append(f"{d['img']}#{d['row']}")
    for f, lst in sorted(group.items()):
        add("truth_flag_group", f, f"정답 점 {len(lst)}개에 {f}", imgs=sorted(set(lst)))
    # 단조 위반만 있는 점 중 같은 서버·같은 정확 점수가 다른 사진에도 있으면(같은 사람이 시각이 다른 화면을 올려 %만 다름) 한 항목으로 묶는다
    # (S19 「다음 시즌에 바꿀 것」 — 후보 107개 중 90개 안팎이 이 꼴, 2026-10-17 분석 ④). 묶음 id는 첫 줄 열쇠라 값이 드러나지 않는다
    same = collections.defaultdict(list)
    for t, d in strong_pts:
        e = oxs.get((d["img"], d["row"]))
        if e and set(t["flags"]) & STRONG == {"mono_viol"}:
            same[(t["server"], e)].append((t, d))
    for k, lst in list(same.items()):
        if len({d["img"] for _, d in lst}) < 2:
            del same[k]
    grouped = {(d["img"], d["row"]) for lst in same.values() for _, d in lst}
    for (sv, _), lst in sorted(same.items(), key=lambda kv: min(f"{d['img']}#{d['row']}" for _, d in kv[1])):
        keys = sorted({f"{d['img']}#{d['row']}" for _, d in lst})
        add("truth_flag_same_score", keys[0], f"{sv} 단조 위반 — 같은 정확 점수 사진 {len(keys)}개가 다른 %로 (시각 차인지, 지난 화면 재게시인지 본다)",
            site=lst[0][1]["site"], imgs=keys, pcts=val(sorted({t["pct"] for t, _ in lst})))
    for t, d in strong_pts:
        if (d["img"], d["row"]) in grouped:
            continue
        add("truth_flag", f"{d['img']}#{d['row']}", f"{t['server']} {t['grade']} {','.join(t['flags'])}",
            site=d["site"], img=d["img"], row=d["row"], pct=val(t["pct"]), score=val(t["score"]))

    # rank_server: 순위표 1~3위 정확 점수별 글 서버
    by_score = collections.defaultdict(list)
    for r in obs:
        s = exact(r)
        if s and r.get("rank") in (1, 2, 3):
            sv = srv.get((r["img"], r["row"]), {})
            if sv.get("basis") in ("text", "fix") or str(sv.get("basis", "")).startswith("text+"):
                by_score[s].append((str(r["no"]), r["site"], r["img"], sv["server"]))
    seen = set()
    for s, lst in sorted(by_score.items()):
        cnt = collections.Counter(sv for no, sv in {(no, sv) for no, _, _, sv in lst})
        if len(cnt) < 2:
            continue
        major = cnt.most_common(1)[0][0]
        for no, site, img, sv in lst:
            if sv != major and no not in seen:
                seen.add(no)
                add("rank_server", no, f"글 서버 {sv}, 같은 1~3위 점수의 다른 글은 {dict(cnt)}", site=site, no=no, img=img,
                    title=(posts.get(no) or {}).get("title"))

    # cluster_mixed
    summ = json.load(open(os.path.join(state, "server_summary.json"), encoding="utf-8"))
    anc_posts = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in srv.values():
        if r.get("cluster") and (r.get("basis") in ("text", "fix", "conflict") or str(r.get("basis", "")).startswith("text+")):
            if r.get("basis") == "conflict":
                continue
            anc_posts[r["cluster"]][r["server"]].add(str(r["no"]))
    for c in summ.get("clusters", []):
        anc = c.get("anchors") or {}
        if len(anc) >= 2:
            major = max(anc, key=lambda k: anc[k])
            minor = {k: sorted(anc_posts[c["cluster"]].get(k, set())) for k in anc if k != major}
            add("cluster_mixed", c["cluster"], f"사진 {c['images']}장, 앵커 {anc}, 이름 {c.get('label')}",
                cluster=c["cluster"], minority_posts=minor)

    # server_count_text
    labeled = {}
    for r in srv.values():
        b = str(r.get("basis") or "")
        if r.get("server") and (b == "text" or b.startswith("text+")):   # fix(고친 글)는 다시 고르지 않는다
            labeled[str(r["no"])] = (r["site"], r["server"])
    for no, (site, sv) in sorted(labeled.items()):
        p = posts.get(no) or {}
        m = MULTI_SRV.search(f"{p.get('title') or ''} {p.get('body') or ''}")
        if m:
            add("server_count_text", no, f"라벨 {sv}, 글에 「{m.group(0)}」", site=site, no=no, title=p.get("title"))

    # cross_season: 이 시즌 줄의 정확 점수가 주로 다른 시즌에 나옴
    # 기본 집합(core use)에 든 줄만 후보로 올린다 — 이미 빠진 줄은 다시 볼 필요가 없다 (S34 「다음 시즌에 바꿀 것」, 2026-10-16)
    use_oi = {c["oi"] for c in jl(os.path.join(state, "core.jsonl")) if c.get("src") == "photo" and c.get("use")}
    mine = collections.defaultdict(set)
    occ = collections.defaultdict(collections.Counter)
    for oi, r in enumerate(obs):
        s = exact(r)
        if s:
            occ[s][name] += 1
            if r.get("season") == "in" and oi in use_oi:
                mine[s].add((str(r["no"]), r["site"]))
    for p in sorted(glob.glob(os.path.join(DATA, "S*", "state", "obs.jsonl"))):
        on = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if on == name:
            continue
        for r in jl(p):
            s = exact(r)
            if s in mine:
                occ[s][on] += 1
    hit = collections.defaultdict(lambda: [collections.Counter(), 0])
    for s, posts_ in mine.items():
        home, nh = occ[s].most_common(1)[0]
        if home == name or nh < 2 * occ[s][name]:
            continue
        for no, site in posts_:
            hit[(no, site)][0][home] += 1
            hit[(no, site)][1] += 1
    for (no, site), (homes, n) in sorted(hit.items()):
        add("cross_season", no, f"정확 점수 {n}개의 주인이 다른 시즌 {dict(homes)}", site=site, no=no,
            title=(posts.get(no) or {}).get("title"))

    # rank1_low: 순위표 1위 점수가 그보다 먼저 올라온 같은 서버 순위표 7위(글 2개 이상에 나온 점수)의 최댓값 × RANK1_LOW보다 낮은 사진 —
    # k위 점수는 시즌 안에서 줄지 않으므로 그 시각 화면일 수 없다. 지난 시즌·다른 서버·오래전에 찍은 화면이다
    # (S21 2544611 「솔레 성장」 글의 S19보다 앞 시즌 사진, 2026-10-17 분석 ④). 1위가 아니라 7위와 견주는 것은 핵 점수 1위가 나중에 지워지기 때문 (S27 일섭 580억).
    # 기본 집합(core use)에 든 줄이 있는 사진만. 서버는 server.jsonl의 서버(붙은 것만)
    px_ = jl(os.path.join(state, "post_exclude.jsonl"))
    ok_row = {(o["img"], o["row"]) for o, x in zip(obs, px_) if not x["exclude"]} if len(px_) == len(obs) else set()
    use_img = {obs[c["oi"]]["img"] for c in jl(os.path.join(state, "core.jsonl")) if c.get("src") == "photo" and c.get("use")}
    ev = []
    for r in obs:
        sv = srv.get((r["img"], r["row"]), {}).get("server")
        if (r.get("rank") in (1, 7) and r.get("score") and r.get("pct") is None and not r.get("self") and sv and r.get("season") == "in"
                and (r["img"], r["row"]) in ok_row):   # % 없는 순위표 줄만 (내 줄의 「Q1」을 1위로 읽은 S31 4188371_0)
            ev.append((r["post_t"] or "", r["rank"], r["score"], r["img"], sv, str(r["no"]), r["site"]))
    ev.sort()
    best, seen7 = {}, collections.defaultdict(set)
    for t, rk, sc, img, sv, no, site in ev:
        if rk == 7:
            seen7[(sv, sc)].add(no)
            if len(seen7[(sv, sc)]) >= 2 and (sv not in best or sc > best[sv][0]):
                best[sv] = (sc, img)
            continue
        b = best.get(sv)
        if b and sc < RANK1_LOW * b[0] and img in use_img:
            add("rank1_low", img, f"{sv} 순위표 1위가 먼저 올라온 같은 서버 7위({b[1]})의 {sc / b[0]:.0%} — 지난 시즌·다른 서버·오래전 화면인지 본다",
                site=site, no=no, img=img, title=(posts.get(no) or {}).get("title"))

    # outlier
    for o in jl(os.path.join(state, "post_outlier.jsonl")):
        fl = [f for f in o.get("flag", []) if f in ("surface_high", "surface_low")]
        if fl:
            r = obs[o["oi"]]
            add("outlier", f"{r['img']}#{r['row']}", f"{','.join(fl)} z={o.get('z')}", site=r["site"], img=r["img"], row=r["row"],
                pct=val(r.get("pct")), score=val(r.get("score")))

    # rank_over_200 (사진마다)
    px = jl(os.path.join(state, "post_exclude.jsonl"))
    over = collections.defaultdict(list)
    if len(px) == len(obs):
        for o, x in zip(obs, px):
            if "rank_over_200" in x["flag"] and not x["exclude"] and o.get("season") == "in":
                over[o["img"]].append(o)
    for img, rows in sorted(over.items()):
        r0 = rows[0]
        add("rank_over_200", img, f"등수 {sorted({r['rank'] for r in rows})} ({len(rows)}줄, 화면 {sorted({str(r.get('screen')) for r in rows})})",
            site=r0["site"], no=str(r0["no"]), img=img, rows=[r["row"] for r in rows])

    # shaky (개발 시즌만)
    if not test:
        spec = importlib.util.spec_from_file_location("truth_grade", os.path.join(HERE, "truth_grade.py"))
        TG = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(TG)
        for r in TG.build([name])["rows"]:
            if r["shaky"]:
                add("shaky", f"{r['server']}:{r['pct']:g}", f"집계표 {r['table']['value']} vs 곡선 {r['curve']['value']} ({r['table_vs_curve']:+.1%})",
                    server=r["server"], pct=r["pct"])
    return name, test, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    name, test, items = collect(a.state)
    out = os.path.join(DATA, name, "review.json")
    old = {}
    if os.path.exists(out):
        old = {x["id"]: x for x in json.load(open(out, encoding="utf-8")).get("items", [])}
    ids = set()
    for x in items:
        ids.add(x["id"])
        for k in KEEP:
            if old.get(x["id"], {}).get(k) is not None:
                x[k] = old[x["id"]][k]
    for i, x in old.items():
        if i not in ids and (x.get("decision") or not x.get("gone")):
            items.append(dict(x, gone=True))
    cnt = collections.Counter(x["kind"] for x in items if not x.get("gone"))
    todo = sum(1 for x in items if not x.get("gone") and not x.get("decision"))
    print(f"{name}{' (시험 시즌: 값 안 씀, 집계표와 안 견줌)' if test else ''} — 점검 후보 {sum(cnt.values())}개 {dict(cnt)}, 판단 안 한 것 {todo}개")
    for x in items:
        if not x.get("gone"):
            print(f"  [{x.get('decision') or ' '}] {x['id']}: {x['detail']}" + (f" 「{x['title'][:30]}」" if x.get("title") else ""))
    if a.write:
        res = {"about": "시즌 데이터 점검 목록 (tools/solo_raid/review.py가 고르고, 세션이 decision·note·by·decided를 채운다). "
                        "decision: keep(그대로) · discard(글 버림) · server(서버 고침) · row(줄 버림) · later(나중에, 사유를 note에). "
                        "고칠 것은 S{n}/discard.json에 적는다",
               "season": name, "test": test, "items": items}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("→", os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
