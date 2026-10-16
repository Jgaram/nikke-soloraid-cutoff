"""솔로레이드 여러 시즌 분석의 학습·검증·시험 분리 (할 일 3.0, 모델 0, 요청 0).

python3 -I tools/solo_raid/split.py check            # split.json이 규칙대로인지 다시 계산해 본다
python3 -I tools/solo_raid/split.py role S35          # 시즌 몫 (dev | test)
python3 -I tools/solo_raid/split.py seen S35 "무엇을 봤는지" [--who 세션]   # 본 기록 한 줄 추가 (3.0c)

분석 도구에서 쓰는 것
- role(name) · is_test(name) · test_seasons() — 규칙: data/solo-raid/split.json
- install_guard(final=False) — 이 프로세스의 open()이 시험 시즌 정답을 못 열게 한다 (3.0b).
  · 시험 시즌 state의 truth* 파일 (state 사본도 같은 폴더의 season.json 이름으로 가린다)
  · data/solo-raid/seasons.json · server_cutoffs.csv는 통째로 막고, load_seasons()·load_server_cutoffs()로만 읽는다
    (시험 시즌의 kr_cut_* 열·행을 빼고 준다)
  final=True(도구의 --final)면 막지 않는다. 시험(3.7)은 모든 선택을 고정한 뒤 한 번만 한다.
- 수집·후처리(post_all.sh, truth.py)는 이 감시를 쓰지 않는다 — 시험 시즌도 평소대로 받고 정답 파일을 만든다.
"""
import argparse
import builtins
import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SPLIT = os.path.join(ROOT, "data", "solo-raid", "split.json")
SEASONS = os.path.join(ROOT, "data", "solo-raid", "seasons.json")
SERVER_CUTOFFS = os.path.join(ROOT, "experiments", "2026-10-14-solo-raid-cutoff", "server_cutoffs.csv")
TRUTH_KEYS = ("kr_cut_",)        # seasons.json에서 시험 시즌이면 빼는 열 (앞부분 일치)

_real_open = builtins.open


def snum(name):
    m = re.fullmatch(r"S?(\d+)", str(name).strip())
    if not m:
        raise ValueError(f"시즌 이름을 읽지 못했다: {name}")
    return int(m.group(1))


def load_split(path=SPLIT):
    with _real_open(path, encoding="utf-8") as f:
        return json.load(f)


def compute(rule):
    """규칙에서 시험 시즌 목록을 다시 계산한다 (split.json과 같아야 한다)."""
    rnd = random.Random(rule["seed"])
    test = []
    for lo, hi in rule["blocks"]:
        block = list(range(lo, hi + 1))
        rnd.shuffle(block)
        test += sorted(block[:rule["per_block"]])
    return sorted(test)


def future_role(n, rule):
    h = int(hashlib.sha256(f"{rule['future_salt']}:S{n}".encode()).hexdigest(), 16)
    return "test" if h % rule["future_mod"] == 0 else "dev"


def role(name, sp=None):
    sp = sp or load_split()
    n = snum(name)
    r = sp["rule"]
    if n in sp["test"]:
        return "test"
    if n in sp["dev_seen"] or r["candidates"][0] <= n <= r["candidates"][1]:
        return "dev"
    if n > r["candidates"][1] and n >= r["future_from"]:
        return future_role(n, r)
    return "dev"


def is_test(name, sp=None):
    return role(name, sp) == "test"


def test_seasons(sp=None):
    sp = sp or load_split()
    return sorted(sp["test"])


# ---------------- 감시 (3.0b) ----------------

def _season_of_dir(d):
    """state 폴더(또는 그 사본)의 season.json 이름 → 번호. 없으면 경로의 S{n}."""
    p = os.path.join(d, "season.json")
    try:
        with _real_open(p, encoding="utf-8") as f:
            return snum(json.load(f)["name"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    for part in reversed(os.path.normpath(d).split(os.sep)):
        if re.fullmatch(r"S\d+", part):
            return snum(part)
    return None


def _blocked(path, sp):
    p = os.path.realpath(os.fsdecode(path))
    if p == os.path.realpath(SEASONS):
        return "seasons.json은 split.load_seasons()로 읽는다 (시험 시즌 kr_cut_* 빼고)"
    if p == os.path.realpath(SERVER_CUTOFFS):
        return "server_cutoffs.csv는 split.load_server_cutoffs()로 읽는다 (시험 시즌 행 빼고)"
    if os.path.basename(p).startswith("truth"):
        n = _season_of_dir(os.path.dirname(p))
        if n is None:
            return "어느 시즌의 정답 파일인지 모른다 (같은 폴더에 season.json이 없음)"
        if n in sp["test"]:
            return f"S{n}은 시험 시즌이다 (data/solo-raid/split.json) — 정답은 --final에서만 연다"
    return None


def install_guard(final=False, sp=None):
    """open() 감시를 켠다. final이면 아무것도 막지 않고 표시만 남긴다."""
    sp = sp or load_split()
    if final:
        print(f"!! --final: 시험 시즌 {', '.join(f'S{n}' for n in sp['test'])}의 정답을 연다. "
              "시험은 선택을 모두 고정한 뒤 한 번만 한다 (할 일 3.7). 본 기록(split.py seen)을 남긴다", file=sys.stderr)
        return

    def guarded(file, *a, **k):
        mode = a[0] if a else k.get("mode", "r")
        if isinstance(file, (str, bytes, os.PathLike)) and not any(c in mode for c in "wax"):   # 읽기만 막는다
            why = _blocked(file, sp)
            if why:
                raise PermissionError(f"split 감시: {os.fsdecode(file)} — {why}")
        return _real_open(file, *a, **k)
    builtins.open = guarded


def add_final_arg(ap):
    ap.add_argument("--final", action="store_true",
                    help="시험 시즌(data/solo-raid/split.json)의 정답도 연다 — 할 일 3.7 시험에서만 (본 기록을 남긴다)")


def guard_args(a):
    """분석 도구 main에서 parse_args() 바로 뒤에 부른다."""
    install_guard(final=getattr(a, "final", False))


def load_seasons(final=False, sp=None):
    """seasons.json — 시험 시즌이면 kr_cut_* 열을 뺀다 (final이면 그대로)."""
    sp = sp or load_split()
    with _real_open(SEASONS, encoding="utf-8") as f:
        rows = json.load(f)
    if final:
        return rows
    out = []
    for r in rows:
        if snum(r["name"]) in sp["test"]:
            r = {k: v for k, v in r.items() if not k.startswith(TRUTH_KEYS)}
            r["truth_hidden"] = True
        out.append(r)
    return out


SV_CSV = {"GLOBAL": "GL"}


def load_server_cutoffs(final=False, sp=None):
    sp = sp or load_split()
    with _real_open(SERVER_CUTOFFS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:   # CSV는 글섭을 GLOBAL로 적는다 — 도구들은 GL로 찾는다 (2026-10-16, 그 전에는 글섭 집계표가 정답 표·prior에서 빠졌다)
        r["server"] = SV_CSV.get(r["server"], r["server"])
    if final:
        return rows
    return [r for r in rows if snum(r["season"]) not in sp["test"]]


# ---------------- 명령 ----------------

def cmd_check(_):
    sp = load_split()
    again = compute(sp["rule"])
    ok = again == sorted(sp["test"])
    print(f"규칙으로 다시 계산한 시험 시즌: {', '.join(f'S{n}' for n in again)} — split.json과 {'같다' if ok else '다르다!'}")
    overlap = set(sp["test"]) & set(sp["dev_seen"])
    if overlap:
        print(f"!! 이미 본 시즌이 시험에 있다: {sorted(overlap)}")
        ok = False
    fut = [f"S{n}:{future_role(n, sp['rule'])}" for n in range(sp["rule"]["future_from"], sp["rule"]["future_from"] + 12)]
    print("앞으로 들어올 시즌 몫 (규칙):", " ".join(fut))
    return 0 if ok else 1


def cmd_role(a):
    print(role(a.season))
    return 0


def cmd_seen(a):
    with _real_open(SPLIT, encoding="utf-8") as f:
        sp = json.load(f)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    names = [f"S{snum(s)}" for s in a.season.split(",")]
    sp["seen_log"].append({"when": now, "seasons": names, "what": a.what, "who": a.who,
                           "roles": {s: role(s, sp) for s in names}})
    with _real_open(SPLIT, "w", encoding="utf-8") as f:
        json.dump(sp, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"본 기록 추가: {names} — {a.what}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="솔로레이드 학습·검증·시험 분리")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("role")
    r.add_argument("season")
    s = sub.add_parser("seen")
    s.add_argument("season", help="S35 또는 S35,S36")
    s.add_argument("what")
    s.add_argument("--who", default="")
    a = ap.parse_args()
    return {"check": cmd_check, "role": cmd_role, "seen": cmd_seen}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
