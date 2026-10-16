"""솔로레이드 후처리 1.8: 1.1~1.7 결과를 합친 핵심 관측 표 core.jsonl을 만든다 (모델 0, 요청 0).

python3 -I tools/solo_raid/build_core.py STATE

한 명령으로 1단계 전체를 다시 만들려면 tools/solo_raid/post_all.sh STATE [옵션] (이 스크립트가 마지막 단계다).

입력 (읽기만 한다. 모두 obs.jsonl·claims.jsonl과 같은 줄 순서여야 한다 — 아니면 멈춘다)
- STATE/obs.jsonl · claims.jsonl            원자료
- STATE/server.jsonl · server_claims.jsonl  서버 (server_classify.py)
- STATE/post_exclude.jsonl                  제외 사유·표시·쓸 값 pct_use·score_use (post_exclude.py)
- STATE/post_link.jsonl · post_claims.jsonl 같은 사람·시점·대표 줄, 글 주장 keep·사진 이음 (post_link.py)
- STATE/post_time.jsonl · post_time_claims.jsonl  시각 등급·구간, top7 시각 구간 (post_time.py)
- STATE/post_outlier.jsonl · post_outlier_summary.json  곡면 이탈 표시 (post_outlier.py, 있으면)

순환 없이 곡면 이탈 표시를 붙이는 순서 (2026-10-16): post_outlier.py는 core.jsonl을 읽지 않고 이 파일의 assemble()로
같은 줄(surf 열 없음)을 만들어 잰다. 그 줄들의 digest를 post_outlier_summary.json에 남기고, 여기서 다시 만든 줄의 digest와 같을 때만
surf 열을 붙인다 (다르면 멈춘다 — post_all.sh로 다시 돌린다). post_outlier.jsonl이 없으면 surf 열은 null이다.

줄
- 사진 관측 `src: "photo"`: obs.jsonl 줄 전부 (시즌 밖·트래커·지난 시즌 포함, `exclude`로 표시). 열쇠 `oi`.
- 글 주장 `src: "claim"`: post_claims.jsonl에서 keep인 줄만. 열쇠 `i`.

열
- key: "p{oi}" 또는 "c{i}". src, oi, i, site, no
- server · server_basis: server*.jsonl의 server·basis
- h · h_lo · h_hi · t_kind · t_grade · day · day_lo: post_time*.jsonl의 h·h_lo·h_hi·t_kind·grade·day·day_lo
  (h = 챌린지 시작 후 시간. C는 t = 상한. 확실한 종료 후는 day_lo == "post")
- h_lo_top7 · h_hi_top7: post_time.jsonl의 top7 시각 구간 (B·C 사진만, 없으면 null. 주장은 null). h_lo·h_hi·t_grade와 따로 둔다
- pct: 사진은 post_exclude의 pct_use, 주장은 글에 적은 %. score: 0.1억 (사진은 score_use). score_raw: 억, 반올림 안 함
- rank: 등수 1~200 (사진은 post_exclude의 rank_use, 주장은 claims.jsonl의 rank). 게임은 200위까지 등수, 그 아래는 %로 보여 준다.
  200을 넘거나, 같은 줄에 %가 있거나(레벨 배지), 트래커 화면이면 null (사진은 flag rank_over_200·rank_with_pct).
  예측·use에는 아직 쓰지 않는다 (2026-10-16, 실어 두기만)
- screen · self: 사진은 obs 값. 주장은 screen null, self는 kind == "self"
- person · moment: 사진은 post_link 값. 사진과 이어진 주장(link)은 그 사진의 값. 이어지지 않은 주장은
  person = "W{site}:{no}"(같은 글의 주장은 글쓴이 한 사람으로 본다, 사진의 사람 열쇠와는 잇지 않음), moment = person + "#c{i}"
- indep: 사진은 post_link 값(시점마다 대표 한 줄, 모든 줄 기준). 주장은 사진과 이어졌으면 false(사진 쪽에서 센다), 아니면 true
- use_indep: use 줄 안에서 시점(moment)마다 한 줄만 true (post_link 대표 줄 > 앞 줄). 기본 집합의 독립 관측 수 = use_indep 합
- link: 사진은 이어진 keep 주장 i 목록, 주장은 link_oi (없으면 null)
- link_time: 사진만. 시각 등급 C인 사진에 이어진 주장 중 시각 등급 B가 있으면 {i, t_lo, t_hi, h_lo, h_hi} (첫 것).
  사진의 h·t_grade는 바꾸지 않는다 — 주장 시각은 근거로만 보탠다
- exclude: 사진은 post_exclude의 exclude. 주장은 [] (빠진 주장은 표에 없다)
- flag: 사진은 post_exclude의 flag. 주장은 post_claims의 flag + "claim_unverified" (+ 이어졌으면 "claim_linked")
- use: 기본 분석 집합 여부 (관리자 결정 2026-10-16, flag·exclude는 그대로 남겨 되돌릴 수 있다)
  - 사진: exclude가 비었고, pct 있음, obs season == "in", flag에 season_unsure_mixed 없음
    (season_unsure_mixed = 이번·지난 기록을 함께 올린 글인데 사진을 가를 근거가 없는 줄. 대부분 지난 시즌일 가능성이 커 뺀다.
     season unsure인데 사진 근거로 이번 시즌이 된 줄(flag curve_outlier)도 점수 판독을 의심해 뺀다)
  - 주장: 언제나 false. S41 새 정보 주장 40줄 표본의 정밀도가 약 절반이다 (지난 시즌 나열·목표치·협동작전·날짜 오독)

- surf_z · surf_flag: post_outlier.jsonl의 z·flag (사진만). 서버가 있고 곡면으로 잰 줄은 surf_z(부호 있음)와 flag 목록,
  서버 미상으로 판정만 한 줄은 surf_z null·flag 목록(no_server_fit 등), 재지 않은 줄(not_use·no_score·no_surface)은 둘 다 null.
  use·flag에는 섞지 않는다 (표시만, 2.2·1.7)

출력: STATE/core.jsonl, STATE/core_summary.json (기본 집합 줄 수, 서버·등급·일차별, 독립 관측·사람 수, 등수 줄 수). 표준 출력에 요약.
"""
import collections
import hashlib
import json
import os
import sys


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def same_rows(name, a, b, key):
    if len(a) != len(b) or any(key(x) != key(y) for x, y in zip(a, b)):
        raise SystemExit(f"{name}의 줄이 원자료와 맞지 않는다 — post_all.sh로 1단계를 다시 돌린다")


def digest(rows):
    """assemble() 줄의 지문 — post_outlier.py가 잰 줄과 지금 줄이 같은지 본다."""
    h = hashlib.sha1()
    for r in rows:
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def assemble(st):
    """core.jsonl 줄 (surf_z·surf_flag 열 없음). 반환 (obs, rows)."""
    obs = jl(os.path.join(st, "obs.jsonl"))
    claims = jl(os.path.join(st, "claims.jsonl"))
    srv = jl(os.path.join(st, "server.jsonl"))
    srv_c = jl(os.path.join(st, "server_claims.jsonl"))
    px = jl(os.path.join(st, "post_exclude.jsonl"))
    lk = jl(os.path.join(st, "post_link.jsonl"))
    pc = jl(os.path.join(st, "post_claims.jsonl"))
    tm = jl(os.path.join(st, "post_time.jsonl"))
    tm_c = jl(os.path.join(st, "post_time_claims.jsonl"))
    ik = lambda r: (r["img"], r["row"])
    for name, rows in (("server.jsonl", srv), ("post_exclude.jsonl", px), ("post_link.jsonl", lk),
                       ("post_time.jsonl", tm)):
        same_rows(name, rows, obs, ik)
    for name, rows in (("server_claims.jsonl", srv_c), ("post_claims.jsonl", pc), ("post_time_claims.jsonl", tm_c)):
        if name != "post_time_claims.jsonl":   # post_time_claims에는 글 번호가 없다 — 줄 수와 i만 본다
            same_rows(name, rows, claims, lambda r: str(r["no"]))
        if len(rows) != len(claims) or any(r["i"] != k for k, r in enumerate(rows)):
            raise SystemExit(f"{name}의 i가 줄 순서와 다르다")

    TCOLS = ("h", "h_lo", "h_hi", "t_kind", "h_lo_top7", "h_hi_top7")

    def tcols(t):
        d = {k: t.get(k) for k in TCOLS}
        d.update({"t_grade": t["grade"], "day": t["day"], "day_lo": t["day_lo"]})
        return d

    claim_by_oi = collections.defaultdict(list)
    for c in pc:
        if c["keep"] and c["link_oi"] is not None:
            claim_by_oi[c["link_oi"]].append(c["i"])

    out = []
    for oi, o in enumerate(obs):
        x, l, t, s = px[oi], lk[oi], tm[oi], srv[oi]
        use = (not x["exclude"] and x["pct_use"] is not None and o.get("season") == "in"
               and "season_unsure_mixed" not in x["flag"])
        link_time = None
        if t["grade"] == "C":
            for i in claim_by_oi.get(oi, []):
                tc = tm_c[i]
                if tc["grade"] == "B":
                    link_time = {"i": i, "t_lo": tc["t_lo"], "t_hi": tc["t_hi"], "h_lo": tc["h_lo"], "h_hi": tc["h_hi"]}
                    break
        r = {"key": f"p{oi}", "src": "photo", "oi": oi, "i": None, "site": o["site"], "no": o["no"],
             "server": s["server"], "server_basis": s["basis"]}
        r.update(tcols(t))
        r.update({"pct": x["pct_use"], "score": x["score_use"], "rank": x["rank_use"],
                  "score_raw": None if o.get("score") is None else o["score"] / 1e8,
                  "screen": o.get("screen"), "self": o.get("self"),
                  "person": l["person"], "moment": l["moment"], "indep": l["indep"], "use_indep": False,
                  "link": claim_by_oi.get(oi, []), "link_time": link_time,
                  "exclude": x["exclude"], "flag": x["flag"], "use": use})
        out.append(r)
    for c in pc:
        if not c["keep"]:
            continue
        i = c["i"]
        t, s = tm_c[i], srv_c[i]
        linked = c["link_oi"] is not None
        if linked:
            person, moment = lk[c["link_oi"]]["person"], lk[c["link_oi"]]["moment"]
        else:
            person = f"W{c['site']}:{c['no']}"
            moment = f"{person}#c{i}"
        sc = c["score_eok"]
        r = {"key": f"c{i}", "src": "claim", "oi": None, "i": i, "site": c["site"], "no": c["no"],
             "server": s["server"], "server_basis": s["basis"]}
        r.update(tcols(t))
        rk = claims[i].get("rank")
        r.update({"pct": c["pct"], "score": None if sc is None else int(sc * 10 + 0.5) / 10, "score_raw": sc,
                  "rank": rk if isinstance(rk, int) and 1 <= rk <= 200 else None,
                  "screen": None, "self": c["kind"] == "self",
                  "person": person, "moment": moment, "indep": not linked, "use_indep": False,
                  "link": c["link_oi"], "link_time": None,
                  "exclude": [], "flag": c["flag"] + ["claim_unverified"] + (["claim_linked"] if linked else []),
                  "use": False})
        out.append(r)

    # use 줄 안의 독립 관측: 시점마다 한 줄 (post_link 대표 줄 먼저, 그다음 앞 줄)
    first = {}
    for r in sorted((r for r in out if r["use"]), key=lambda r: (not r["indep"], r["oi"])):
        if r["moment"] not in first:
            first[r["moment"]] = r
            r["use_indep"] = True
    return obs, out


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.split("\n\n")[1])
    st = sys.argv[1]
    obs, out = assemble(st)
    dg = digest(out)
    po_p, ps_p = os.path.join(st, "post_outlier.jsonl"), os.path.join(st, "post_outlier_summary.json")
    surf = None
    if os.path.exists(po_p):
        po = jl(po_p)
        meta = json.load(open(ps_p, encoding="utf-8")) if os.path.exists(ps_p) else {}
        if meta.get("core_digest") != dg or len(po) != len(obs) or any(r["oi"] != k for k, r in enumerate(po)):
            raise SystemExit("post_outlier.jsonl이 지금 1단계 결과로 만든 것이 아니다 — post_all.sh로 다시 돌린다")
        surf = po
    else:
        print("core: post_outlier.jsonl 없음 — surf_z·surf_flag는 null")
    for r in out:
        r["surf_z"] = r["surf_flag"] = None
        if surf is not None and r["src"] == "photo":
            p = surf[r["oi"]]
            if p.get("skip") in (None, "server_unknown"):
                r["surf_flag"] = p["flag"]
                r["surf_z"] = p["z"] if p.get("skip") is None else None

    with open(os.path.join(st, "core.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    U = [r for r in out if r["use"]]
    I = [r for r in U if r["use_indep"]]
    cnt = lambda rows, k: dict(sorted(collections.Counter(str(r[k]) for r in rows).items()))
    photo_core = [r for r, o in zip(out, obs) if o.get("season") == "in" and o.get("pct") is not None
                  and o.get("screen") != "tracker"]
    summ = {
        "rows": len(out), "photo_rows": len(obs), "claim_rows": len(out) - len(obs),
        "photo_core_rows": len(photo_core),
        "photo_core_not_use": dict(sorted(collections.Counter(
            ",".join(r["exclude"]) or ("season_unsure_mixed" if "season_unsure_mixed" in r["flag"] else "other")
            for r in photo_core if not r["use"]).items())),
        "use": len(U), "use_indep": len(I),
        "photo_rank_rows": sum(1 for r, o in zip(out, obs) if o.get("season") == "in" and r["rank"] is not None),
        "photo_rank_over_200": sum(1 for r in out if "rank_over_200" in r["flag"]),
        "use_persons": len({r["person"] for r in U}),
        "use_persons_fp": len({r["person"] for r in U if r["person"].startswith("P")}),
        "use_persons_multi_moment": sum(1 for v in collections.Counter(
            p for p, m in {(r["person"], r["moment"]) for r in U}).values() if v > 1),
        "use_by_server": cnt(U, "server"), "use_by_t_grade": cnt(U, "t_grade"), "use_by_day": cnt(U, "day"),
        "use_by_day_lo": cnt(U, "day_lo"),
        "use_indep_by_server": cnt(I, "server"), "use_indep_by_t_grade": cnt(I, "t_grade"),
        "use_indep_by_day": cnt(I, "day"),
        "use_by_server_t_grade": dict(sorted(collections.Counter(
            f"{r['server']}/{r['t_grade']}" for r in U).items())),
        "use_post_end": sum(1 for r in U if r["day_lo"] == "post"),
        "use_link_time": sum(1 for r in U if r["link_time"]),
        "claims_linked": sum(1 for r in out if r["src"] == "claim" and r["link"] is not None),
        "claims_new": sum(1 for r in out if r["src"] == "claim" and r["link"] is None),
        "use_top7": dict(sorted(collections.Counter(r["t_grade"] for r in U if r["h_lo_top7"] is not None).items())),
        "use_surf_flag": dict(sorted(collections.Counter(f for r in U for f in (r["surf_flag"] or [])).items())),
        "use_surf_measured": sum(1 for r in U if r["surf_z"] is not None),
    }
    with open(os.path.join(st, "core_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("core:", json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
