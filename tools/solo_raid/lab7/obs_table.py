"""관측 표 (WP0 b, 2026-10-28) — (시즌, T, 자료 판) → 관측 줄 표. 새 방법 연구의 모든 후보·점검이 이 표를 읽는다.

  python3 -I tools/solo_raid/lab7/obs_table.py build --work W --out DIR [--seasons S21-S42] [--jobs N] [--full]
  python3 -I tools/solo_raid/lab7/obs_table.py check --work W --dir DIR [--out check.json]

T 절단 사본(W/trunc/{시즌}/trunc_T{T}/state, lab7/trunc.py가 만든 것)의 core.jsonl 줄 가운데 사진·글 **전부**를 줄로 둔다
(use=False도 — 고르는 것은 뒤 일). 파일: DIR/{시즌}/T{T}.jsonl.gz, 첫 줄은 머리 {season,T,data_ver,H,c0,n,src_key}.
--full은 끝난 시즌의 전체 state, T = H (DIR/full/{시즌}.jsonl.gz, 시즌 S19~S42) — 관측 오차를 재는 용도.

공개 규칙: 글 번호·원문·닉네임·원글 주소는 쓰지 않는다. 사람·시점은 불투명 열쇠(pid·mid)로, 시각 근거 문자열은 말 토큰만 남긴다.

칸 (줄 하나):
  kind         "A"·"B"·"C" (사진, 시각 등급) · "text" (글로만 적은 %)
  t_grade      core의 시각 등급 (글도 있다)        t_kind  upper(게시 시각이 상한)·shot(촬영)·stated(말한 시각)
  t_basis      시각 근거 (post_time.jsonl basis에서 글 번호·따옴표 원문·한글을 뺀 말 토큰). 글은 None
  score        모형에 쓰는 점수 (score_raw 있으면 그것, 없으면 score)   score_disp  화면에 보인 점수 (score)
  pct          % (글은 적은 값)
  h h_lo h_hi  챌린지 시작부터 시간. 촬영 시각 점·하한·상한 (등급 C는 h_hi가 게시 시각)
  h_lo_top7 h_hi_top7   top7 표에서 좁힌 구간
  hp           게시 시각 (시작부터 시간). 사진은 obs.post_t, 글은 claims.post_t. 없으면 None
  day day_lo   일차 (말한/하한)       site  dc·naver·blabla     self  본인 사진·글   screen 화면 종류   rank 순위
  server server_basis   서버 라벨과 근거 (None = 미상)
  shot_src     촬영 시각 출처 (obs.shot_src)       file_date_out  파일 날짜가 시즌 밖 (obs)    verify  판독 확인 same·differ·None
  oi i key     사진 번호(obs 줄) · 글 번호(claims 줄) · core 열쇠 (번호는 사본 안의 줄 번호일 뿐 글 번호가 아니다)
  link         이어진 쪽: 글이면 [사진 oi], 사진이면 [글 i]
  indep use_indep use   중복 제거 표시·기본 집합 표시     flag exclude   core의 표시 목록
  ckind        글의 종류 (claims.kind)
  pid mid      사람·시점 불투명 열쇠 = sha1(「시즌|원래 값」)[:12]. 같은 시즌 안에서만 같은 것을 가리킨다
  --- predict.load_obs(unknown={"mode":"mono"})가 T에서 이 줄을 어떻게 하는지 ---
  lo_reason    None이면 T·등급·시각 거르기를 지난 줄. 아니면 load_obs가 세는 이름:
               drop_best_record · drop_no_post_t · after_T · drop_c_before_start · drop_time · drop_other_server
               (load_obs가 세지 않고 지나치는 줄: claim · not_use(use 아님) · no_pct_score · skip_day_post · skip_no_h)
  he           load_obs가 쓰는 시각 (등급 C는 max(h_hi, h_lo, 0), A·B는 h). 계산할 수 없으면 None
  mono_drop    거르기를 지난 줄 가운데 서버 미상이고 mono_filter가 뺀 줄 (load_obs 출력에서 unused=True로 남는다)
  rep          같은 사람·같은 %(0.01 단위)의 대표로 load_obs 출력에 나오는 줄. 주의: mono_drop인 줄도 대표가 될 수 있다 (출력에는 unused로 남음)

load_obs_like(rows, T, H, unknown=True)는 위 칸만으로 load_obs 출력(사진 oi 목록)을 다시 셈한다.
"""
import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

PR = C.PR

COLS = ["kind", "t_grade", "t_kind", "t_basis", "score", "score_disp", "pct", "h", "h_lo", "h_hi", "h_lo_top7", "h_hi_top7",
        "hp", "day", "day_lo", "site", "self", "screen", "rank", "server", "server_basis", "shot_src", "file_date_out",
        "verify", "oi", "i", "key", "link", "indep", "use_indep", "use", "flag", "exclude", "ckind", "pid", "mid",
        "lo_reason", "he", "mono_drop", "rep"]
COUNTED = ("drop_best_record", "drop_no_post_t", "after_T", "drop_c_before_start", "drop_time", "drop_other_server")


def opaque(season, v):
    """사람·시점 값 → 불투명 열쇠 (원래 값에는 글 번호가 들어 있다)."""
    return None if v is None else hashlib.sha1(f"{season}|{v}".encode()).hexdigest()[:12]


def clean_basis(b):
    """시각 근거 문자열에서 글 번호(claim#N·via 사진열쇠)·따옴표 원문·한글을 뺀다. 말 토큰(now·clock·timer 등)은 남는다."""
    if b is None:
        return None
    b = re.sub(r"\bvia \S+", "via", b)
    b = re.sub(r"claim#\d+", "claim", b)
    b = re.sub(r"'[^']*'", "", b)
    b = re.sub(r"[^\x00-\x7f]+", "", b)
    return re.sub(r"\s+", " ", b).strip()


def _he(r):
    """load_obs가 쓰는 시각 (c_delay 0)."""
    if r["h"] is None:
        return None
    if r["t_grade"] == "C":
        hh = r["h_hi"] if r["h_hi"] is not None else r["h"]
        if hh is None or hh < 0:
            return None
        lo = r["h_lo"] if r["h_lo"] is not None else 0.0
        return max(hh, lo, 0.0)
    return r["h"]


def reason_of(r, T, H):
    """load_obs의 거르기 순서 그대로 — (lo_reason, he). 표 칸만 쓴다."""
    he = _he(r)
    if r["kind"] == "text":
        return "claim", None
    if not r["use"]:
        return "not_use", he
    if r["pct"] is None or r["score_disp"] is None:
        return "no_pct_score", he
    if "best_record" in r["flag"]:
        return "drop_best_record", he
    if r["day_lo"] == "post":
        return "skip_day_post", he
    if r["h"] is None:
        return "skip_no_h", he
    if r["hp"] is None:
        return "drop_no_post_t", he
    if r["hp"] > T:
        return "after_T", he
    if r["t_grade"] == "C" and he is None:
        return "drop_c_before_start", he
    if he > T + 1e-9 or he >= H or he < 0:
        return "drop_time", he
    if r["server"] is not None and r["server"] not in PR.SERVERS:
        return "drop_other_server", he
    return None, he


def replay(rows, T, H, unknown=True):
    """표 칸만으로 load_obs를 다시 돈다. 반환 (거르기를 지난 줄 [(줄 번호, dict)], 출력 dict 목록, assign_unknown 정보).
    dict 꼴은 predict.load_obs 줄과 같다 (person 자리에 pid). 미상 거르기는 PR.assign_unknown(mono)이 한다."""
    surv = []
    for n, r in enumerate(rows):
        why, he = reason_of(r, T, H)
        if why is not None:
            continue
        srv = r["server"] if r["server"] in PR.SERVERS else None
        d = {"key": r["key"], "oi": r["oi"], "server": srv, "mix": None, "s": r["score"], "x": r["pct"], "h": he,
             "hp": r["hp"], "grade": r["t_grade"], "person": r["pid"], "unused": srv is None, "site": r["site"],
             "span": PR.time_span(r, he)}
        surv.append((n, d))
    info = None
    if unknown:
        info = PR.assign_unknown(None, [d for _, d in surv], {}, {"mode": "mono"})
    grp = collections.defaultdict(list)
    for _, d in surv:
        grp[(d["person"], round(d["x"] / 0.01))].append(d)
    out = []
    for L in grp.values():
        L.sort(key=lambda o: (o["grade"] == "C", o["h"], o["oi"]))
        out.append(L[0])
    out.sort(key=lambda o: o["oi"])
    return surv, out, info


def load_obs_like(rows, T, H, unknown=True):
    """표 칸만으로 predict.load_obs(..., unknown={"mode":"mono"} 또는 None)의 출력 사진 oi 목록."""
    return [d["oi"] for d in replay(rows, T, H, unknown)[1]]


def counters(rows, T, H):
    """lo_reason 이름별 줄 수 (load_obs 셈 dict와 견주는 것만)."""
    c = collections.Counter(reason_of(r, T, H)[0] for r in rows)
    return {k: c.get(k, 0) for k in COUNTED}


def build(state, season, T):
    """state(절단 사본 또는 전체)의 core 사진·글 줄 전부 → 표 줄 목록 (core 순서)."""
    _, c0, H = PR.season_info(state)
    H = float(H)
    core = PR.jl(os.path.join(state, "core.jsonl"))
    obs = PR.jl(os.path.join(state, "obs.jsonl"))
    claims = PR.jl(os.path.join(state, "claims.jsonl"))
    pt = {d["oi"]: d for d in PR.jl(os.path.join(state, "post_time.jsonl"))}
    rows = []
    for r in core:
        text = r["src"] == "claim"
        if text:
            cl = claims[r["i"]]
            o = {}
            hp_t = cl.get("post_t")
            link = [] if r["link"] is None else [r["link"]]
            basis = None
        else:
            o = obs[r["oi"]]
            hp_t = o.get("post_t")
            link = list(r["link"] or [])
            basis = clean_basis((pt.get(r["oi"]) or {}).get("basis"))
        sc = r["score_raw"] if r["score_raw"] is not None else r["score"]
        row = {"kind": "text" if text else r["t_grade"], "t_grade": r["t_grade"], "t_kind": r["t_kind"], "t_basis": basis,
               "score": sc, "score_disp": r["score"], "pct": r["pct"], "h": r["h"], "h_lo": r["h_lo"], "h_hi": r["h_hi"],
               "h_lo_top7": r["h_lo_top7"], "h_hi_top7": r["h_hi_top7"],
               "hp": round(PR.hours_of(hp_t, c0), 6) if hp_t else None,
               "day": r["day"], "day_lo": r["day_lo"], "site": r["site"], "self": r["self"], "screen": r["screen"],
               "rank": r["rank"], "server": r["server"], "server_basis": r["server_basis"],
               "shot_src": o.get("shot_src"), "file_date_out": None if text else o.get("file_date_out"),
               "verify": o.get("verify"), "oi": r["oi"], "i": r["i"], "key": r["key"], "link": link,
               "indep": r["indep"], "use_indep": r["use_indep"], "use": r["use"], "flag": r["flag"], "exclude": r["exclude"],
               "ckind": cl.get("kind") if text else None,
               "pid": opaque(season, r["person"]), "mid": opaque(season, r["moment"])}
        rows.append(row)
    annotate(rows, T, H)
    return rows


def annotate(rows, T, H):
    """lo_reason·he·mono_drop·rep 칸을 채운다 (mono 방식)."""
    for r in rows:
        r["lo_reason"], r["he"] = reason_of(r, T, H)
        r["mono_drop"] = False
        r["rep"] = False
    surv, out, _ = replay(rows, T, H, True)
    for n, d in surv:
        rows[n]["mono_drop"] = bool(d["unused"])
    keep = {d["oi"] for d in out}
    for r in rows:
        r["rep"] = r["lo_reason"] is None and r["oi"] in keep
    return rows


def write_cell(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as g:
        g.write((json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        for r in rows:
            g.write((json.dumps([r[k] for k in COLS], ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    os.replace(tmp, path)


def read_cell(path):
    """(머리, 줄 dict 목록)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(f.readline())
        cols = header.get("cols", COLS)
        rows = [dict(zip(cols, json.loads(l))) for l in f if l.strip()]
    return header, rows


def _cell(job):
    kind, season, T, state, out, src_key = job
    t0 = time.time()
    rows = build(state, season, T)
    _, c0, H = PR.season_info(state)
    header = {"season": season, "T": T, "data_ver": C.DATA_VER, "H": float(H), "c0": c0.strftime("%Y-%m-%d %H:%M:%S"),
              "n": len(rows), "src_key": src_key, "cols": COLS}
    write_cell(out, header, rows)
    return {"season": season, "T": T, "path": out, "sec": time.time() - t0, "bytes": os.path.getsize(out), "n": len(rows),
            "rep": sum(1 for r in rows if r["rep"])}


def jobs_for(a):
    jobs, missing = [], 0
    if a.full:
        for s in (C.parse_seasons(a.seasons) if a.seasons else C.DEV):
            _, H = C.info(s)
            jobs.append(("full", s, H, C.state_of(s), os.path.join(a.out, "full", f"{s}.jsonl.gz"), "full"))
        return jobs, missing
    for s in C.parse_seasons(a.seasons or "S21-S42"):
        for T in C.t_grid(s):
            st = C.trunc_path(a.work, s, T)
            if st is None:
                missing += 1
                continue
            jobs.append(("trunc", s, T, st, os.path.join(a.out, s, f"T{T:g}.jsonl.gz"), C.done_key(a.work, s, T)))
    return jobs, missing


def cmd_build(a):
    jobs, missing = jobs_for(a)
    t0 = time.time()
    print(f"표 {len(jobs)}개 (사본 없어 건너뜀 {missing}), 프로세스 {a.jobs}", flush=True)
    if a.jobs <= 1:
        it = map(_cell, jobs)
    else:
        pool = mp.Pool(a.jobs)
        it = pool.imap_unordered(_cell, jobs, chunksize=1)
    res = []
    for k, r in enumerate(it, 1):
        res.append(r)
        print(f"[{k}/{len(jobs)}] {r['season']} T{r['T']:g} 줄 {r['n']} 대표 {r['rep']} {r['sec']:.2f}초 {r['bytes']:,}바이트", flush=True)
    if a.jobs > 1:
        pool.close()
        pool.join()
    tot = sum(r["bytes"] for r in res)
    print(f"끝: {len(res)}개 {tot:,}바이트 합 {sum(r['sec'] for r in res):.1f}초 (전체 {time.time() - t0:.1f}초)")


def _state_of_file(work, path, header):
    if os.path.basename(os.path.dirname(path)) == "full":
        return C.state_of(header["season"])
    return C.trunc_path(work, header["season"], header["T"])


def check_cell(work, path, base):
    """한 표 파일 점검. 반환 (요약 dict, 어긋남 목록)."""
    header, rows = read_cell(path)
    bad = []
    st = _state_of_file(work, path, header)
    s, T, H = header["season"], header["T"], header["H"]
    if st is None:
        return {"file": os.path.relpath(path, base), "n": len(rows)}, [f"{s} T{T:g}: 원본 사본이 없다"]
    if header.get("data_ver") != C.DATA_VER:
        bad.append(f"{s} T{T:g}: 자료 판 {header.get('data_ver')} ≠ {C.DATA_VER}")
    _, c0, H2 = PR.season_info(st)
    if float(H2) != H or len(rows) != header["n"]:
        bad.append(f"{s} T{T:g}: 머리와 다름 (H {H} vs {H2}, n {header['n']} vs {len(rows)})")
    cells = {}
    for name, unk in (("mono", {"mode": "mono"}), ("none", None)):
        ref, cnt = PR.load_obs(st, T, H, c0, unknown=unk)
        ref_oi = sorted(o["oi"] for o in ref)
        like = load_obs_like(rows, T, H, unknown=unk is not None)
        cell = {"ref_n": len(ref_oi), "like_n": len(like)}
        if like != ref_oi:
            bad.append(f"{s} T{T:g} {name}: load_obs_like {len(like)}줄 ≠ load_obs {len(ref_oi)}줄 (oi 집합 다름 {len(set(like) ^ set(ref_oi))})")
        if name == "mono":
            tab = sorted(r["oi"] for r in rows if r["rep"])
            if tab != ref_oi:
                bad.append(f"{s} T{T:g}: rep 칸 {len(tab)}줄 ≠ load_obs {len(ref_oi)}줄")
            ref_by = {o["oi"]: o for o in ref}
            for r in rows:
                if r["rep"]:
                    o = ref_by.get(r["oi"])
                    if o is None or (o["h"], o["s"], o["x"], o["grade"], o["unused"]) != (r["he"], r["score"], r["pct"], r["t_grade"], r["mono_drop"]):
                        bad.append(f"{s} T{T:g}: 대표 줄 oi={r['oi']} 값이 load_obs와 다르다")
                        break
            cm = collections.Counter(r["lo_reason"] for r in rows)
            for k in COUNTED:
                if cm.get(k, 0) != cnt.get(k, 0):
                    bad.append(f"{s} T{T:g}: {k} 표 {cm.get(k, 0)} ≠ load_obs {cnt.get(k, 0)}")
            if sum(1 for r in rows if r["lo_reason"] is None) != cnt["rows_le_T"]:
                bad.append(f"{s} T{T:g}: 거르기 지난 줄 수 {sum(1 for r in rows if r['lo_reason'] is None)} ≠ rows_le_T {cnt['rows_le_T']}")
            drop = (cnt.get("unknown") or {}).get("rows_removed")
            if drop is not None and drop != sum(1 for r in rows if r["mono_drop"]):
                bad.append(f"{s} T{T:g}: mono_drop 표 {sum(1 for r in rows if r['mono_drop'])} ≠ rows_removed {drop}")
        cells[name] = cell
    kinds = collections.Counter(r["kind"] for r in rows)
    with open(path, "rb") as f:
        sha = hashlib.sha1(f.read()).hexdigest()
    return {"file": os.path.relpath(path, base), "season": s, "T": T, "n": len(rows), "kinds": dict(kinds), "bytes": os.path.getsize(path), "sha1": sha,
            "rep": cells["mono"]["like_n"], "rep_none": cells["none"]["like_n"]}, bad


def cmd_check(a):
    files = []
    for root, _, fs in os.walk(a.dir):
        files += [os.path.join(root, f) for f in fs if f.endswith(".jsonl.gz")]
    files.sort()
    t0 = time.time()
    summ, bad = [], []
    for k, p in enumerate(files, 1):
        s, b = check_cell(a.work, p, a.dir)
        summ.append(s)
        bad += b
        print(f"[{k}/{len(files)}] {os.path.relpath(p, a.dir)} 줄 {s['n']} {'어긋남 ' + str(len(b)) if b else '맞음'}", flush=True)
    kinds = collections.Counter()
    for s in summ:
        kinds.update(s.get("kinds", {}))
    out = {"data_ver": C.DATA_VER, "cells_checked": len(summ), "mismatches": len(bad), "mismatch_list": bad,
           "rows_by_kind": dict(kinds), "total_rows": sum(s["n"] for s in summ), "total_bytes": sum(s.get("bytes", 0) for s in summ),
           "sec": round(time.time() - t0, 1), "files": summ}
    p = a.out or os.path.join(a.dir, "check.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for b in bad:
        print("어긋남:", b)
    print(f"점검 {len(summ)}개, 어긋남 {len(bad)}, 줄 {out['total_rows']:,}, 바이트 {out['total_bytes']:,} → {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--work", default=None, help="절단 사본 폴더 (--full이면 안 쓴다)")
    b.add_argument("--out", required=True)
    b.add_argument("--seasons", default=None, help="기본: 절단 S21-S42 · --full은 S19-S42")
    b.add_argument("--jobs", type=int, default=1)
    b.add_argument("--full", action="store_true", help="끝난 시즌 전체 state, T = H")
    c = sub.add_parser("check")
    c.add_argument("--work", default=None)
    c.add_argument("--dir", required=True)
    c.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        if not a.full and not a.work:
            ap.error("--work가 필요하다")
        cmd_build(a)
    else:
        cmd_check(a)


if __name__ == "__main__":
    main()
