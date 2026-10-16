"""판독 Workflow(.claude/workflows/raid-tile-read.js) 결과를 합치고 빠진 칸을 센다 (모델 0).

python3 -I tools/tile_check.py SHEETS.jsonl OUTDIR MERGED.jsonl

- OUTDIR/reads_*.jsonl을 배치 번호 순서로 합쳐 MERGED.jsonl에 쓴다. 이것이 tile_measure.py의 입력이다.
- 시트 칸 지도(SHEETS.jsonl)와 맞대 빠진 칸·없는 칸·겹친 칸·깨진 줄을 센다. 출력은 한 줄이다.
  빠진 칸이 있으면 그 칸이 든 시트 이름을 OUTDIR/missing_sheets.txt에 쓴다 (그 시트만 다시 돌린다).
"""
import glob
import json
import os
import sys


def main(sheets_path, outdir, merged_path):
    where = {}
    for line in open(sheets_path):
        m = json.loads(line)
        for cell in m["cells"]:
            where[cell] = m["sheet"]
    seen, rows, bad, dup = set(), [], 0, 0
    for fn in sorted(glob.glob(os.path.join(outdir, "reads_*.jsonl"))):
        for line in open(fn):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if r.get("cell") in seen:
                dup += 1
                continue
            seen.add(r.get("cell"))
            rows.append(line if line.endswith("\n") else line + "\n")
    with open(merged_path, "w") as f:
        f.writelines(rows)
    missing = sorted(set(where) - seen)
    extra = sorted(seen - set(where))
    redo = sorted({where[c] for c in missing})
    miss_path = os.path.join(outdir, "missing_sheets.txt")
    if redo:
        with open(miss_path, "w") as f:
            f.write("\n".join(redo) + "\n")
    elif os.path.exists(miss_path):
        os.remove(miss_path)
    print(f"칸 {len(where)} 읽음 {len(seen & set(where))} 빠짐 {len(missing)} (시트 {len(redo)})"
          f" 없는칸 {len(extra)} 겹침 {dup} 깨진줄 {bad} → {merged_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
