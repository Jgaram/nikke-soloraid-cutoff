"""T 절단 사본을 한 명령으로 만든다 (WP0 c, 2026-10-28).

  python3 -I tools/solo_raid/lab7/trunc.py --work W [--seasons S21-S42] [--jobs 4] [--T 6,12]

맞힐 시즌 × T 격자(common.t_grid)마다 backtest.trunc_state(state, T, W/trunc/{시즌}, xval.extra_of(state))를 부른다.
xval.py와 같은 자리(W/trunc/{시즌}/trunc_T{T})에 만들고, DONE 열쇠가 같으면 건너뛰므로 다시 돌려도 된다.
셀(시즌, T) 하나가 일 하나다 (프로세스 --jobs개). 걸린 시간은 W/trunc_times.json에 합쳐 쓴다.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import backtest as BT  # noqa: E402
import xval as XV  # noqa: E402


def one(job):
    """절단 사본 하나. (시즌, T, 걸린 초, 새로 만들었나)."""
    s, T, work = job
    state = C.state_of(s)
    done = os.path.join(C.trunc_dir(work, s, T), "DONE")
    before = os.path.getmtime(done) if os.path.exists(done) else None
    t0 = time.time()
    BT.trunc_state(state, T, os.path.join(work, "trunc", s), XV.extra_of(state))
    sec = time.time() - t0
    after = os.path.getmtime(done) if os.path.exists(done) else None
    return s, T, sec, before != after


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="사본을 둘 폴더 (W/trunc/{시즌}/trunc_T{T})")
    ap.add_argument("--seasons", default="S21-S42")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--T", default=None, help="쉼표로 T만 (시험용). 없으면 격자 전부")
    a = ap.parse_args()
    seasons = C.parse_seasons(a.seasons)
    want = None if a.T is None else {float(x) for x in a.T.split(",")}
    jobs = []
    for s in seasons:
        for T in C.t_grid(s):
            if want is None or T in want:
                jobs.append((s, T, a.work))
    jobs.sort(key=lambda j: (-j[1], j[0]))   # 오래 걸리는 큰 T부터
    os.makedirs(os.path.join(a.work, "trunc"), exist_ok=True)
    t0 = time.time()
    cells = {}
    print(f"사본 {len(jobs)}개, 프로세스 {a.jobs}, 자료 판 {C.DATA_VER}", flush=True)
    if a.jobs <= 1:
        it = map(one, jobs)
    else:
        pool = mp.Pool(a.jobs)
        it = pool.imap_unordered(one, jobs, chunksize=1)
    for k, (s, T, sec, made) in enumerate(it, 1):
        cells[f"{s}/T{T:g}"] = {"sec": round(sec, 2), "built": made}
        print(f"[{k}/{len(jobs)}] {s} T{T:g} {sec:.1f}초 {'새로' if made else '있음'}", flush=True)
    if a.jobs > 1:
        pool.close()
        pool.join()
    elapsed = time.time() - t0
    p = os.path.join(a.work, "trunc_times.json")
    old = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            old = json.load(f).get("cells", {})
    old.update(cells)
    built = [v["sec"] for v in cells.values() if v["built"]]
    out = {"data_ver": C.DATA_VER, "last_run": {"elapsed_sec": round(elapsed, 1), "jobs": a.jobs, "cells": len(cells),
                                                "built": len(built), "built_sec_sum": round(sum(built), 1)},
           "cells": dict(sorted(old.items()))}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"끝: {len(cells)}개 (새로 {len(built)}개, 합 {sum(built):.0f}초) 전체 {elapsed:.0f}초 → {p}")


if __name__ == "__main__":
    main()
