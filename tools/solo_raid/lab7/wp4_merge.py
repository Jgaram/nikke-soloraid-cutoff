"""WP4·WP5 도우미 조각 합치기 (2026-10-31; WP5a부터 dep도 맞는지 본다) — 1단계 시즌 조각·칸 예측 조각을 파일 하나로.

  python3 -I tools/solo_raid/lab7/wp4_merge.py stage1 --out OUT.json PART.json [PART.json ...]
  python3 -I tools/solo_raid/lab7/wp4_merge.py pred   --out OUT.json PART.json [PART.json ...]

stage1: 머리(header)는 첫 조각 것, 시즌은 모두 모은다. 조각끼리 obs_var·basis가 다르거나 같은 시즌이 두 번 나오면 멈춘다.
pred: cells·info·priors를 모은다. obs_var·stage1 sha가 다르거나 같은 (시즌, T)가 두 번 나오면 멈춘다.
"""
import argparse
import json
import os
import sys


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _dump(p, o):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False)
    os.replace(p + ".tmp", p)


def merge_stage1(out, parts):
    res = None
    for p in parts:
        d = _load(p)
        if res is None:
            res = {"header": d["header"], "seasons": {}}
        h = d["header"]
        if any(h.get(k) != res["header"].get(k) for k in ("obs_var", "basis", "dep", "dep_fit")):
            sys.exit(f"{p}: obs_var·basis·dep가 첫 조각과 다르다")
        for s, e in d["seasons"].items():
            if s in res["seasons"]:
                sys.exit(f"{p}: {s}가 두 번")
            res["seasons"][s] = e
    res["seasons"] = dict(sorted(res["seasons"].items(), key=lambda kv: int(kv[0][1:])))
    res["header"]["merged_from"] = [os.path.basename(p) for p in parts]
    _dump(out, res)
    print("시즌", len(res["seasons"]), list(res["seasons"]))


def merge_pred(out, parts):
    res = None
    seen = set()
    for p in parts:
        d = _load(p)
        if res is None:
            res = {k: v for k, v in d.items() if k not in ("cells", "info", "priors")}
            res.update({"cells": [], "info": [], "priors": {}, "made_parts": []})
        if d.get("obs_var") != res.get("obs_var") or d.get("dep") != res.get("dep") or d["made"]["inputs"] != res["made"]["inputs"]:
            sys.exit(f"{p}: obs_var·dep·stage1 sha가 첫 조각과 다르다")
        for i in d["info"]:
            k = (i["season"], i["T"])
            if k in seen:
                sys.exit(f"{p}: {k}가 두 번")
            seen.add(k)
        res["cells"] += d["cells"]
        res["info"] += d["info"]
        res["priors"].update(d["priors"])
        res["made_parts"].append({"file": os.path.basename(p), "tool_head": d["made"].get("tool_head"), "cmd": d["made"].get("cmd")})
    _dump(out, res)
    print("칸", len(res["info"]), "줄", len(res["cells"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("stage1", "pred"))
    ap.add_argument("--out", required=True)
    ap.add_argument("parts", nargs="+")
    a = ap.parse_args()
    (merge_stage1 if a.what == "stage1" else merge_pred)(a.out, a.parts)


if __name__ == "__main__":
    main()
