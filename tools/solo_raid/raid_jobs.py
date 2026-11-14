#!/usr/bin/env python3
"""판독 일감 목록 — Workflow 도구가 없을 때 .claude/workflows/raid-window.js 대신 (2026-11-14).

raid-window.js와 같은 규칙으로 일감(판독 시트 19장마다·글 분류·시즌 문맥 배치마다 하나)을 만들어
한 줄에 하나씩 JSON으로 찍는다: {"label", "phase", "out", "prompt"}.
조율 세션은 줄마다 Agent(subagent_type="raid-vision", model="haiku", prompt=그 prompt)를 띄우고
결과 파일을 열지 않은 채 `live_tick.sh merge`로 합친다 (docs/solo-raid-live.md 「2」의 3).

  python3 -I tools/solo_raid/raid_jobs.py WORK/workflow.json            모든 일감
  python3 -I tools/solo_raid/raid_jobs.py WORK/workflow.json '[...]'    merge 종료 3의 끝 줄(skip 목록)을 빼고
"""
import json
import sys


def jobs(args, skip=()):
    run = args["run"]
    per = args.get("batch") or 19
    pad = lambda n: f"{n:02d}"
    sheets = [f"{run}/sheets/strip_{pad(i)}.png" for i in range(1, (args.get("sheets") or 0) + 1)]
    out = []
    for k, i in enumerate(range(0, len(sheets), per), 1):
        b = sheets[i:i + per]
        o = f"{run}/reads/reads_{pad(k)}.jsonl"
        out.append({"label": f"판독 {pad(k)} ({len(b)}장)", "phase": "판독", "out": o,
                    "prompt": f"지시문 파일: {args['tile_prompt']}\n이 파일의 `---` 아래를 그대로 따른다. {{SHEETS}}와 {{OUT}}은 다음 값이다.\n"
                              f"{{SHEETS}}: {', '.join(b)}\n{{OUT}}: {o}"})
    for key, ph, f in (("text", "글 분류", "text_prompt.md"), ("ctx", "시즌 문맥", "ctx_prompt.md")):
        for i in range(1, (args.get(key) or 0) + 1):
            o = f"{run}/{key}/out_{pad(i)}.jsonl"
            out.append({"label": f"{ph} {pad(i)}", "phase": ph, "out": o,
                        "prompt": f"지시문 파일: {run}/{f}\n이 파일의 `---` 아래를 그대로 따른다. {{IN}}과 {{OUT}}은 다음 값이다.\n"
                                  f"{{IN}}: {run}/{key}/in_{pad(i)}.jsonl\n{{OUT}}: {o}"})
    skip = set(skip)
    return [j for j in out if j["out"] not in skip and j["out"][len(run) + 1:] not in skip]


def main():
    w = json.load(open(sys.argv[1]))
    if not w.get("needed"):
        print("판독할 것 없음 (needed false)", file=sys.stderr)
        return
    skip = json.loads(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].strip() else []
    js = jobs(w["args"], skip)
    for j in js:
        print(json.dumps(j, ensure_ascii=False))
    print(f"일감 {len(js)}개 (건너뜀 {len(skip)})", file=sys.stderr)


if __name__ == "__main__":
    main()
