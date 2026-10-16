"""2단계: 코드가 자신 없는 사진을 칸으로 잘라 시트 몇 장에 모은다 (모델 0, 요청 0).

python3 -I tools/tile_sheets.py IDS.txt PARSED.jsonl OCR.jsonl IMGDIR OUTDIR

- 칸 종류
  - 띠(`strip`): 파서 화면이 순위·결과 화면이면 OCR이 찾은 숫자 상자(% 또는 5자리+ 숫자)의 세로 범위를 띠로 잡고,
    가까운 띠끼리 합쳐 한 칸으로 자른다. 가로는 사진 전체 폭. 잘라 낸 막대 모양 사진은 통째로 한 띠.
    순위 목록은 OCR이 못 찾은 줄이 빠지지 않게 첫 띠~마지막 띠를 한 칸으로 자른다 (시범 1차에서 중간 줄 6개가 빠짐).
  - 통째(`whole`): 화면을 못 정한 사진은 통째로 줄여 2x2 시트에 넣는다.
- 띠 시트는 가로 1100px로 키워(850px 사본이면 약 1.3배) 위에서 아래로 쌓는다. 시트 하나는 1.15MP 안
  (모델이 줄이지 않는다). 칸 위에 칸 이름(A1, A2 …)을 찍는다.
- `TILE_FULL=1`: 모든 사진을 통째로 한 띠로 보고 띠 시트에 쌓는다 (세션 E, 통째 칸 2차 판독용).
  2x2로 줄이지 않는다. 시트 높이를 넘는 사진은 여러 칸으로 나뉜다.
- `TILE_NOWHOLE=1`: 2x2 통째 칸을 만들지 않고, 띠를 못 잡은 사진은 통째로 한 띠에 쌓는다 (세션 F1, 기간 실행).
- `TILE_START=N`: 시트 번호를 N+1부터 매기고 OUTDIR/sheets.jsonl에 이어 쓴다. 폭이 다른 시트(850 사본·1400 원본)를
  한 폴더에 모아 판독 Workflow 한 번으로 읽을 때 쓴다 (세션 F1).
- OUTDIR/sheets.jsonl: 시트 파일, 종류, 칸 이름 → 사진 id. OUTDIR/*.png: 시트.
"""
import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

POS = ("result_pop", "my_rank", "rank_list", "season_end")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SW = int(os.environ.get("TILE_W", "1100"))   # 띠 시트 가로 (850px 사본을 이 폭으로 키운다)
SH = 1_150_000 // SW                          # 1.15MP 안
FULL = os.environ.get("TILE_FULL") == "1"
NOWHOLE = os.environ.get("TILE_NOWHOLE") == "1"
START = int(os.environ.get("TILE_START", "0"))
CW, CH = 540, 510            # 통째 칸 (2x2 → 1100x1040)
LAB = 26                     # 칸 이름 줄 높이
PAD = 10
NUM = re.compile(r"\d[\d,.]{3,}\d|\d+(?:\.\d+)?\s*%")


def is_bar(w, h):
    """잘라 낸 점수 막대 모양. 850px 사본에서 높이 161~202px인 내 줄 막대(F2 놓친 줄)를 넣으려고 4배 이상 넓으면 220px까지 본다 (2026-10-16)."""
    return max(w, h) >= 250 and h >= 40 and (h <= 150 and w >= 3 * h or h <= 220 and w >= 4 * h)


def bands(items, w, h, scale):
    """숫자 상자의 세로 범위 → 합친 띠 [(y0, y1)] (원래 사진 좌표)."""
    ys = []
    for box, txt, _ in items:
        if NUM.search(txt.replace(" ", "")) and sum(c.isdigit() for c in txt) >= 3:
            ys.append((box[1] / scale, box[3] / scale))
    ys.sort()
    out = []
    for a, b in ys:
        if out and a - out[-1][1] < max(40, (b - a) * 1.5):
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(max(0, int(a) - PAD * 2), min(h, int(b) + PAD * 2)) for a, b in out]


def main():
    ids_f, parsed_f, ocr_f, imgdir, outdir = sys.argv[1:6]
    os.makedirs(outdir, exist_ok=True)
    ids = open(ids_f).read().split()
    parsed = {p["id"]: p for p in map(json.loads, open(parsed_f))}
    ocr = {o["id"]: o for o in map(json.loads, open(ocr_f))}
    font = ImageFont.truetype(FONT, 20)
    strips, wholes = [], []
    for i in ids:
        path = os.path.join(imgdir, i + ".img")
        if not os.path.exists(path):
            path = next((os.path.join(imgdir, f) for f in os.listdir(imgdir) if f.startswith(i + ".")), None)
        if not path:
            continue
        im = Image.open(path).convert("RGB")
        w, h = im.size
        p = parsed.get(i, {})
        if FULL or is_bar(w, h):
            strips.append((i, im))
        elif p.get("screen") in POS and i in ocr:
            o = ocr[i]
            bs = bands(o["items"], w, h, o.get("scale", 1) * o["w"] / w if o.get("w") else 1)
            if bs and p["screen"] == "rank_list":
                # OCR이 점수 상자를 못 찾은 줄도 들어가게, 목록은 첫 띠~마지막 띠를 한 칸으로 (위아래 한 줄 여유)
                bs = [(max(0, bs[0][0] - 60), min(h, bs[-1][1] + 60))]
            if not bs:
                (strips if NOWHOLE else wholes).append((i, im))
            for a, b in bs:
                strips.append((i, im.crop((0, a, w, b))))
        else:
            (strips if NOWHOLE else wholes).append((i, im))

    manifest = []
    # 띠 시트
    sheet, y, cells, n = None, 0, {}, START

    def flush(kind, img, cells):
        nonlocal n
        n += 1
        fn = f"{kind}_{n:02d}.png"
        img.save(os.path.join(outdir, fn))
        manifest.append({"sheet": fn, "kind": kind, "cells": cells})

    k = 0
    for i, im in strips:
        s = SW / im.width
        cell = im.resize((SW, max(1, round(im.height * s))), Image.LANCZOS)
        if cell.height + LAB > SH:   # 너무 긴 띠는 잘라 여러 칸으로
            parts = [cell.crop((0, t, SW, min(cell.height, t + SH - LAB))) for t in range(0, cell.height, SH - LAB - 40)]
        else:
            parts = [cell]
        for part in parts:
            if sheet is None or y + LAB + part.height > SH or len(cells) >= 12:
                if sheet is not None:
                    flush("strip", sheet.crop((0, 0, SW, y)), cells)
                sheet, y, cells, k = Image.new("RGB", (SW, SH), (255, 255, 255)), 0, {}, 0
            k += 1
            name = f"S{n + 1}-{k}"
            d = ImageDraw.Draw(sheet)
            d.rectangle((0, y, SW, y + LAB - 2), fill=(200, 30, 30))
            d.text((8, y + 2), name, fill=(255, 255, 255), font=font)
            sheet.paste(part, (0, y + LAB))
            y += LAB + part.height + 4
            cells[name] = i
    if sheet is not None and cells:
        flush("strip", sheet.crop((0, 0, SW, y)), cells)
    # 통째 시트 (2x2)
    for s in range(0, len(wholes), 4):
        sheet = Image.new("RGB", (CW * 2 + 20, CH * 2 + 10), (60, 60, 60))
        d = ImageDraw.Draw(sheet)
        cells = {}
        for j, (i, im) in enumerate(wholes[s:s + 4]):
            x0, y0 = (j % 2) * (CW + 20), (j // 2) * (CH + 10)
            t = im.copy()
            t.thumbnail((CW, CH - LAB), Image.LANCZOS)
            name = f"W{n + 1}-{j + 1}"
            d.rectangle((x0, y0, x0 + CW, y0 + LAB - 2), fill=(30, 30, 200))
            d.text((x0 + 8, y0 + 2), name, fill=(255, 255, 255), font=font)
            sheet.paste(t, (x0, y0 + LAB))
            cells[name] = i
        flush("whole", sheet, cells)
    with open(os.path.join(outdir, "sheets.jsonl"), "a" if START else "w") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"strip_cells={len(strips)} whole={len(wholes)} sheets={len(manifest)}")


if __name__ == "__main__":
    main()
