"""후처리(post_all.sh)가 어떤 입력·도구로 돌았는지 남긴다 (할 일 3.A1, 2026-10-16, 모델 0, 요청 0).

python3 -I tools/solo_raid/post_stamp.py STATE [--extra 이름=경로 ...]          # post_all.sh 끝에서 부른다 → STATE/post_all_stamp.json
python3 -I tools/solo_raid/post_stamp.py STATE --check                          # 지금 입력이 도장과 같은지 (fresh·stale·never)

도장은 결정적이다 (시각을 넣지 않는다 — 두 번 돌려 결과 파일이 같아야 한다). 실행 시각은 등록부가 git 커밋 시각으로 본다.
- inputs: STATE의 입력 파일(아래 OUTPUTS·ANALYSIS가 아닌 것)마다 sha1, runs/*/fetchlog.tsv sha1 묶음, S{n}/discard.json(버릴 것·고칠 것) 항목,
  그리고 post_all에 준 시즌 전용 입력(--posts·--snap·--parsed·--alt·--runs, 폴더는 파일 sha1 묶음)
- tools: 후처리 도구 파일 sha1
- digest: 위 둘을 합친 sha1 (등록부·분석 결과에 남기는 값)
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import discard  # noqa: E402

STAMP = "post_all_stamp.json"
OUTPUTS = {"server.jsonl", "server_claims.jsonl", "server_summary.json", "post_exclude.jsonl", "post_link.jsonl",
           "post_claims.jsonl", "post_time.jsonl", "post_time_claims.jsonl", "truth.json", "truth_points.jsonl",
           "post_outlier.jsonl", "post_outlier_summary.json", "core.jsonl", "core_summary.json", STAMP}
ANALYSIS = {"prior.json", "post_server_mix.jsonl", "post_server_mix_summary.json"}   # 후처리 뒤 분석 층 (입력 아님)
TOOLS = ["tools/solo_raid/post_all.sh", "tools/solo_raid/post_exclude.py", "tools/solo_raid/post_link.py",
         "tools/solo_raid/post_time.py", "tools/solo_raid/truth.py", "tools/solo_raid/post_outlier.py",
         "tools/solo_raid/build_core.py", "tools/solo_raid/discard.py",
         "experiments/2026-10-15-solo-raid-low-cost/tools/server_classify.py"]


def sha1_file(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha1_tree(d):
    h = hashlib.sha1()
    for base, dirs, files in sorted(os.walk(d)):
        dirs.sort()
        for n in sorted(files):
            p = os.path.join(base, n)
            h.update(os.path.relpath(p, d).encode() + b"\0" + sha1_file(p).encode() + b"\n")
    return h.hexdigest()


def sha1_obj(o):
    return hashlib.sha1(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def compute(state, extra=None, opts=""):
    """→ 도장 dict. extra: {이름: 경로} (post_all의 시즌 전용 입력, 빈 값은 빼고). opts: post_all 옵션 (예: --no-code-scores)."""
    inputs = {}
    for n in sorted(os.listdir(state)):
        p = os.path.join(state, n)
        if os.path.isfile(p) and n not in OUTPUTS and n not in ANALYSIS:
            inputs[n] = sha1_file(p)
    extra = {k: v for k, v in (extra or {}).items() if v}

    def resolve(v):
        return os.path.abspath(v if os.path.isabs(v) or os.path.exists(v) else os.path.join(ROOT, v))
    runs = resolve(extra["runs"]) if "runs" in extra else os.path.join(state, "..", "runs")
    fl = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(runs) for f in fs if f == "fetchlog.tsv") if os.path.isdir(runs) else []
    inputs["runs/*/fetchlog.tsv"] = sha1_obj([(os.path.relpath(p, runs), sha1_file(p)) for p in fl])
    inputs["discard.json(시즌 항목)"] = sha1_obj(list(discard.items(state).values()))   # 열쇠 순서
    ex = {}
    for k, v in sorted(extra.items()):
        ap = resolve(v)
        sha = inputs["runs/*/fetchlog.tsv"] if k == "runs" else sha1_tree(ap) if os.path.isdir(ap) else sha1_file(ap)
        ex[k] = {"path": os.path.relpath(ap, ROOT), "sha1": sha}
    tools = {t: sha1_file(os.path.join(ROOT, t)) for t in TOOLS if os.path.exists(os.path.join(ROOT, t))}
    st = {"inputs": inputs, "extra": ex, "opts": opts.strip(), "tools": tools}
    st["input_digest"] = sha1_obj({"inputs": inputs, "extra": ex, "opts": opts.strip()})
    st["tools_digest"] = sha1_obj(tools)
    st["digest"] = sha1_obj(st)
    return st


def check(state):
    """→ (상태, 설명). fresh: 입력·도구가 도장과 같다 · stale_input · stale_tools · never(도장 없음)."""
    p = os.path.join(state, STAMP)
    if not os.path.exists(p):
        return "never", "도장 없음 (post_all.sh를 이 도구가 생기기 전에 돌렸거나 안 돌림)"
    with open(p, encoding="utf-8") as f:
        old = json.load(f)
    extra = {k: v["path"] for k, v in old.get("extra", {}).items()}
    now = compute(state, extra, old.get("opts", ""))
    if now["input_digest"] != old["input_digest"]:
        ch = [k for k in set(now["inputs"]) | set(old["inputs"]) if now["inputs"].get(k) != old["inputs"].get(k)]
        ch += [f"extra:{k}" for k in set(now["extra"]) | set(old["extra"]) if now["extra"].get(k) != old["extra"].get(k)]
        return "stale_input", "입력이 바뀜: " + ", ".join(sorted(ch)[:6])
    if now["tools_digest"] != old["tools_digest"]:
        ch = [k for k in set(now["tools"]) | set(old["tools"]) if now["tools"].get(k) != old["tools"].get(k)]
        return "stale_tools", "도구가 바뀜: " + ", ".join(os.path.basename(k) for k in sorted(ch))
    return "fresh", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--extra", action="append", default=[], help="이름=경로 (posts·snap·parsed·alt·runs)")
    ap.add_argument("--opts", default="", help="post_all 옵션 문자열 (예: --no-code-scores)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        s, why = check(a.state)
        print(s, why)
        return 0 if s == "fresh" else 1
    extra = dict(x.split("=", 1) for x in a.extra)
    st = compute(a.state, extra, a.opts)
    with open(os.path.join(a.state, STAMP), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")
    print(f"stamp: 입력 {len(st['inputs'])}개 · 시즌 전용 {len(st['extra'])}개 · digest {st['digest'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
