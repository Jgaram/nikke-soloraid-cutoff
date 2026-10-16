"""4단계: 기간 실행 하나의 판독·글 분류·시즌 문맥을 합쳐 관측 표를 갱신한다 (모델 0, 요청 0).

python3 -I tools/merge_obs.py STATE RUN [POLICY]

- 입력: RUN/sheets/sheets.jsonl, RUN/reads/reads_*.jsonl (raid-window Workflow 판독), RUN/text_extracted.jsonl,
  RUN/text/out_*.jsonl (Haiku kind), RUN/ctx/out_*.jsonl (시즌 문맥), STATE/route.jsonl·parsed.jsonl·manifest.jsonl·posts/.
- 사진 관측 `src`: code(파서) · haiku(850px 사본 띠·통째) · orig(원본 1400px) · tracker(판독 화면이 tracker).
  - 파서가 순위 목록으로 본 사진은 원본 판독만 남긴다. 파서가 자신 있게 읽었으면 원본과 맞대 `verify`(same·differ)를 적는다.
  - 띠 판독에서 순위 목록으로 나오고 순위·%가 있는 사진은 STATE/pending_orig.txt에 적는다 → 다음 실행이 원본으로 다시 읽고 줄을 바꾼다.
  - 같은 사진(`same`)은 먼저 읽은 사진의 줄을 새 글에 붙인다 (`dup_of`).
- 칸은 S41 재수집 obs_images.jsonl·claims_text.jsonl과 같게 펼치고 `run`·`season*`·`ctx`·`site`(dc·naver·blabla)를 더한다.
- 시즌 거르기(season_filter.py)는 매번 STATE 전체 관측에 다시 한다 (비교 줄이 실행마다 는다). POLICY 기본 "unsure".
- 출력: STATE/obs.jsonl, STATE/claims.jsonl, STATE/ctx.jsonl, STATE/reads.jsonl(칸 판독 누적), RUN/merge.json.
  표준 출력에 요약 한 줄. 빠진 칸·줄이 있으면 그 배치 출력 파일 이름을 RUN/missing.txt에 쓴다.
"""
import collections
import datetime as dt
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import season_filter  # noqa: E402
from sites import parse_no, site_of  # noqa: E402

KST = dt.timezone(dt.timedelta(hours=9))
# 첨부 이름의 찍은 시각 (S41 재수집 tools/consolidate.py·finalize.py와 같은 규칙)
P1 = re.compile(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ _\-T]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})")
P2 = re.compile(r"(?<!\d)(17[89]\d{10})(?!\d)")
OBS_SCREENS = ("rank_list", "my_rank", "result", "season_end", "tracker")


def jl(p):
    out = []
    for f in sorted(glob.glob(p)):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                try:
                    out.append(json.loads(l))
                except ValueError:
                    out.append({"_bad": True})
    return out


def shot_time(name, post_t):
    if not name or name.startswith("gov-") or not post_t:
        return None, None
    t, src = None, None
    m = P1.search(name)
    if m:
        try:
            t, src = dt.datetime(*map(int, m.groups())), "filename"
        except ValueError:
            t = None
    if t is None:
        m = P2.search(name)
        if m:
            t, src = dt.datetime.fromtimestamp(int(m.group(1)) / 1000, KST).replace(tzinfo=None), "epoch_ms"
    if t is None:
        return None, None
    pt = dt.datetime.strptime(post_t, "%Y-%m-%d %H:%M:%S")
    if t > pt + dt.timedelta(minutes=5) or t < pt - dt.timedelta(days=4):
        return None, None
    return t.strftime("%Y-%m-%d %H:%M:%S"), src


def file_date(name):
    if not name or name.startswith("gov-"):
        return None
    m = P1.search(name)
    if m:
        y, mo, d = (int(x) for x in m.groups()[:3])
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    m = P2.search(name)
    if m:
        return dt.datetime.fromtimestamp(int(m.group(1)) / 1000, KST).strftime("%Y-%m-%d")
    return None


def num(s, kind):
    """보이는 글자 → 값. `?`가 있으면 None (tile_measure.py와 같은 기준)."""
    if s is None or "?" in str(s):
        return None
    s = str(s)
    if kind == "pct":
        m = re.search(r"(\d+(?:\.\d+)?)\s*%?", s)
        return float(m.group(1)) if m else None
    d = re.sub(r"[^\d]", "", s)
    if not d:
        return None
    if kind == "score":
        return int(d) if len(d) >= 9 else None
    return int(d)


def read_rows(reads):
    """한 사진의 칸 판독 여러 개 → (화면들, 줄들). 같은 (순위, %, 점수) 줄은 하나로."""
    screens, rows, seen = [], [], set()
    for r in reads:
        screens.append(r.get("screen"))
        if r.get("screen") not in OBS_SCREENS:
            continue
        for x in r.get("rows") or []:
            row = {"rank": num(x.get("rank_txt"), "rank"), "rank_txt": x.get("rank_txt"),
                   "pct": num(x.get("pct_txt"), "pct"), "pct_txt": x.get("pct_txt"),
                   "score": num(x.get("score_txt"), "score"), "score_txt": x.get("score_txt"),
                   "self": bool(x.get("self")), "screen": r.get("screen"), "label": x.get("label"),
                   "note": (r.get("note") or "")[:200]}
            k = (row["rank"], row["pct"], row["score"])
            if not any(v is not None for v in k) and not any(row[f + "_txt"] for f in ("rank", "pct", "score")):
                continue
            if k in seen and any(v is not None for v in k):
                continue
            seen.add(k)
            rows.append(row)
    return screens, rows


def code_rows(p):
    scr = "result" if p["screen"] == "result_pop" else p["screen"]
    return [{"rank": r.get("rank"), "rank_txt": None, "pct": r.get("pct"), "pct_txt": None, "score": r.get("score"),
             "score_txt": None, "self": r.get("self"), "screen": scr, "note": ""} for r in p["rows"]]


def agree(code, orig):
    """코드 줄과 원본 줄이 맞는가: 같은 점수(0.1억 안) 줄끼리 순위·%가 둘 다 있을 때 같아야 하고, 코드 점수가 모두 원본에 있어야 한다.
    원본이 순위를 더 채운 것은 같다고 본다."""
    for c in code:
        if c["score"] is None:
            continue
        m = [o for o in orig if o["score"] and abs(o["score"] - c["score"]) < 1e7]
        if not m:
            return False
        if any(c[k] is not None and m[0][k] is not None and c[k] != m[0][k] for k in ("rank", "pct")):
            return False
    return True


def main(state, run, policy="unsure"):
    run_id = os.path.basename(os.path.normpath(run))
    season = json.load(open(os.path.join(state, "season.json"), encoding="utf-8"))
    c0 = season_filter.c_start(season)
    route = {r["id"]: r for r in jl(os.path.join(state, "route.jsonl"))}
    man = {m["id"]: m for m in jl(os.path.join(state, "manifest.jsonl"))}
    parsed = {p["id"]: p for p in jl(os.path.join(state, "parsed.jsonl"))}
    st = collections.Counter()
    missing = set()

    # 1) 칸 판독
    where, sheet_of = {}, {}
    for m in jl(os.path.join(run, "sheets", "sheets.jsonl")):
        for c, i in m["cells"].items():
            where[c] = i
            sheet_of[c] = m["sheet"]
    reads = collections.defaultdict(list)
    seen = set()
    for r in jl(os.path.join(run, "reads", "reads_*.jsonl")):
        if r.get("_bad") or r.get("cell") not in where:
            st["cell_bad"] += 1
            continue
        if r["cell"] in seen:
            continue
        seen.add(r["cell"])
        reads[where[r["cell"]]].append(r)
    st["cells"], st["cells_read"] = len(where), len(seen)
    miss_sheets = sorted({sheet_of[c] for c in set(where) - seen})
    if miss_sheets:
        missing.update(f"sheet {s}" for s in miss_sheets)
    keep = [r for r in jl(os.path.join(state, "reads.jsonl")) if r.get("run") != run_id]  # 같은 RUN을 다시 합치면 바꿔 쓴다
    with open(os.path.join(state, "reads.jsonl"), "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for i, rs in reads.items():
            f.write(json.dumps({"id": i, "run": run_id, "reads": rs}, ensure_ascii=False) + "\n")
    orig_ok = {json.loads(l)["id"] for l in open(os.path.join(state, "orig", "fetch.jsonl"))
               if json.loads(l)["code"] == 200} if os.path.exists(os.path.join(state, "orig", "fetch.jsonl")) else set()
    pend_used = set(open(os.path.join(run, "pending_orig_used.txt")).read().split()) \
        if os.path.exists(os.path.join(run, "pending_orig_used.txt")) else set()

    # 2) 사진별 줄
    img_rows = {}       # id -> (src, verify, rows)
    pending = []
    this = [i for i, r in route.items() if r.get("run") == run_id] + sorted(pend_used)
    for i in this:
        r = route[i]
        rt = "orig" if i in pend_used else r["route"]
        if rt == "code":
            img_rows[i] = ("code", None, code_rows(parsed[i]))
        elif rt == "strip":
            scr, rows = read_rows(reads.get(i, []))
            img_rows[i] = ("haiku", None, rows)
            if "rank_list" in scr and any(x["rank_txt"] or x["pct_txt"] for x in rows):
                pending.append(i)   # 순위·% 없는 카드형 덱 목록은 원본으로 다시 읽지 않는다
        elif rt in ("orig", "whole"):
            scr, rows = read_rows(reads.get(i, []))
            src = "orig" if i in orig_ok else "haiku"
            verify = None
            p = parsed.get(i)
            if rt == "orig" and p and p["conf"] and p["rows"]:
                verify = "same" if agree(code_rows(p), rows) else "differ"
                st[f"verify_{verify}"] += 1
            img_rows[i] = (src, verify, rows)
    st["pending_orig"] = len(pending)
    if pending:
        pf = os.path.join(state, "pending_orig.txt")
        have = open(pf).read().split() if os.path.exists(pf) else []
        with open(pf, "w") as f:
            f.write("".join(i + "\n" for i in dict.fromkeys(have + pending)))

    # 3) 관측 줄로 펼치기 (같은 사진은 먼저 읽은 사진의 줄)
    old = jl(os.path.join(state, "obs.jsonl"))
    old_by_img = collections.defaultdict(list)
    for o in old:
        old_by_img[o["img"]].append(o)
    posts = {}
    # 앞 실행의 글은 다른 세션이면 posts/에 없다 (state_sync.py로 불러온 상태). 시각은 STATE/posts_min.jsonl에서 찾는다
    pmin = {r["no"]: r for r in jl(os.path.join(state, "posts_min.jsonl"))}

    def post(no):
        if no not in posts:
            p = os.path.join(state, "posts", f"{no}.json")
            posts[no] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else pmin.get(no, {})
        return posts[no]

    new = []
    for i in this:
        r = route[i]
        if r["route"] == "same":
            base = img_rows.get(r["same_as"])
            if base is None:
                src_rows = [o for o in old_by_img.get(r["same_as"], [])]
                base = (src_rows[0]["src"], src_rows[0]["verify"], src_rows) if src_rows else None
            if base is None:
                continue
            src, verify, rows = base
            dup_of = r["same_as"]
        elif i in img_rows:
            src, verify, rows = img_rows[i]
            dup_of = None
        else:
            continue
        m = man.get(i, {})
        p = post(r["no"])
        st_, ssrc = shot_time(m.get("fname"), p.get("t"))
        fd = file_date(m.get("fname"))
        for k, x in enumerate(rows):
            new.append({
                "no": r["no"], "site": site_of(r["no"]), "img": i, "row": k,
                "src": "tracker" if x.get("screen") == "tracker" else src, "verify": verify, "dup_of": dup_of,
                "post_t": p.get("t"), "shot_t": st_, "shot_src": ssrc, "file_date": fd,
                "file_date_out": bool(fd and fd < c0[:10]), "clock": None,
                "screen": x.get("screen"), "boss": None, "server": None, "server_src": None, "mock": False,
                "rank": x.get("rank"), "rank_txt": x.get("rank_txt"), "pct": x.get("pct"), "pct_txt": x.get("pct_txt"),
                "score": x.get("score"), "score_txt": x.get("score_txt"), "self": x.get("self"),
                "label": x.get("label"), "conf": None, "note": x.get("note") or "", "run": run_id})
    redo = {o["img"] for o in new} | set(img_rows)
    obs = [o for o in old if o["img"] not in redo] + new
    st["obs_new"] = len(new)

    # 4) 글 숫자 주장
    ext = jl(os.path.join(run, "text_extracted.jsonl"))
    kinds = {}
    for r in jl(os.path.join(run, "text", "out_*.jsonl")):
        if not r.get("_bad") and "id" in r:
            kinds[r["id"]] = r.get("kind")
    want = {k for k, x in enumerate(ext) if x["kind"] != "delta"}
    st["text_rows"], st["text_read"] = len(want), len(want & set(kinds))
    if want - set(kinds):
        missing.add(f"text {len(want - set(kinds))} rows")
    run_nos = {parse_no(x) for x in open(os.path.join(run, "nos.txt")).read().split()}
    claims = [c for c in jl(os.path.join(state, "claims.jsonl")) if c["no"] not in run_nos]
    for k, x in enumerate(ext):
        kh = kinds.get(k)
        claims.append({"no": x["no"], "site": site_of(x["no"]), "post_t": x.get("post_t") or post(x["no"]).get("t"),
                       "kind": kh or x["kind"], "kind_rule": x["kind"], "kind_haiku": kh, "q": x.get("q"),
                       "server": None, "rank": None, "pct": x["pct"], "score_eok": x["score_eok"], "phase": None,
                       "src": x["src"], "time": x.get("time"), "core": x.get("core"), "score_from": x.get("score_from"),
                       "txt": x["txt"], "run": run_id})

    # 5) 시즌 문맥 + 거르기 (STATE 전체)
    ctx_rows = [c for c in jl(os.path.join(state, "ctx.jsonl")) if c["no"] not in run_nos]
    got = [r for r in jl(os.path.join(run, "ctx", "out_*.jsonl")) if not r.get("_bad") and "no" in r]
    asked = {r["no"] for r in jl(os.path.join(run, "ctx", "in_*.jsonl"))}
    if asked - {r["no"] for r in got}:
        missing.add(f"ctx {len(asked - {r['no'] for r in got})} posts")
    ctx_rows += [{"no": r["no"], "season": r.get("season"), "why": r.get("why"), "run": run_id} for r in got]
    ctx = {c["no"]: c["season"] for c in ctx_rows}
    cnt = season_filter.annotate(obs, season, ctx, policy)

    def dump(name, rows):
        with open(os.path.join(state, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dump("obs.jsonl", obs)
    dump("claims.jsonl", claims)
    dump("ctx.jsonl", ctx_rows)
    by_src = collections.Counter(o["src"] for o in obs if o["season"] == "in")
    by_site = collections.Counter(site_of(o["no"]) for o in obs if o["season"] == "in")
    summ = {"run": run_id, **st, "obs_total": len(obs), "obs_in": sum(1 for o in obs if o["season"] == "in"),
            "obs_unsure": sum(1 for o in obs if o["season"] == "unsure"),
            "obs_in_by_src": dict(by_src), "obs_in_by_site": dict(by_site), "season": {f"{a}->{b}": n for (a, b), n in cnt.items()},
            "claims_run": sum(1 for c in claims if c["run"] == run_id), "ctx_run": len(got),
            "missing": sorted(missing)}
    json.dump(summ, open(os.path.join(run, "merge.json"), "w"), ensure_ascii=False, indent=1)
    if missing:
        open(os.path.join(run, "missing.txt"), "w").write("\n".join(sorted(missing)) + "\n")
    elif os.path.exists(os.path.join(run, "missing.txt")):  # 빠진 배치를 다시 돌려 채웠으면 지난번 표시를 지운다 (S40 시범, 2026-10-16)
        os.remove(os.path.join(run, "missing.txt"))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
