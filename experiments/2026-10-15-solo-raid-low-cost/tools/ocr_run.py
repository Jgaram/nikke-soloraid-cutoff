"""평가 세트 사진에 OCR을 돌려 원시 결과를 JSONL로 쓴다 (모델 0, 요청 0).

세션 임시 폴더 venv의 python으로 실행한다 (rapidocr-onnxruntime 1.2.3 + PaddleOCR v5 korean rec ONNX, v5 mobile det).
    VENV/bin/python tools/ocr_run.py MODELDIR IMGDIR OUT.jsonl

- 모델 파일: MODELDIR/det_v5.onnx, ko_v5_inference.onnx, ko_v5_keys.txt
  (huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx, korean_PP-OCRv5_mobile_rec_onnx. 키는 inference.yml의 character_dict)
- 두 번째 판(선택): 환경 변수 OCR_SCALE(예 1.5)로 키워서 읽고, OCR_IDS(파일, 한 줄에 id 하나)에 있는 사진만 읽는다.
  w·h·상자 좌표는 키운 크기 기준이고 scale을 함께 적는다.
- 한 줄: id, w, h, sec(CPU 벽시계), items=[[x0,y0,x1,y1], 글자, 신뢰도]. 이미 한 id는 건너뛴다.
- 큰 사진: 픽셀이 OCR_MAX_PX(기본 1,600만)를 넘으면 줄여서 읽는다. w·h·상자는 줄인 크기 기준이고
  src_w·src_h·cap(줄인 비율)을 함께 적는다 (S36 2026-10-16: 1.02억 픽셀 사진에서 메모리 한도로 죽음).
  1배 판은 scale을 적지 않는다 (tile_sheets.py가 w/원본 너비로 되돌린다).
- 죽은 사진 건너뛰기: 읽는 중인 id를 OUT.cur에 적어 둔다. 같은 사진을 읽다 두 번 죽으면 다음 실행에서 err로 적고 넘어간다.
"""
import json
import math
import os
import sys
import time

import rapidocr_onnxruntime
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.ch_ppocr_v3_det.text_detect import TextDetector
from rapidocr_onnxruntime.ch_ppocr_v3_rec.text_recognize import TextRecognizer
from rapidocr_onnxruntime.utils import read_yaml
import numpy as np


def engine(mdir):
    eng = RapidOCR()
    cfg = read_yaml(os.path.join(os.path.dirname(rapidocr_onnxruntime.__file__), "config.yaml"))
    rc = dict(cfg["Rec"], model_path=os.path.join(mdir, "ko_v5_inference.onnx"),
              keys_path=os.path.join(mdir, "ko_v5_keys.txt"))
    eng.text_recognizer = TextRecognizer(rc)
    eng.text_detector = TextDetector(dict(cfg["Det"], model_path=os.path.join(mdir, "det_v5.onnx")))
    return eng


def main():
    mdir, imgdir, out = sys.argv[1:4]
    eng = engine(mdir)
    scale = float(os.environ.get("OCR_SCALE", "1"))
    only = set(open(os.environ["OCR_IDS"]).read().split()) if os.environ.get("OCR_IDS") else None
    max_px = int(os.environ.get("OCR_MAX_PX", "16000000"))
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    cur = out + ".cur"
    prev_id, prev_n = None, 0
    if os.path.exists(cur):
        parts = open(cur).read().split()
        if len(parts) == 2:
            prev_id, prev_n = parts[0], int(parts[1])
    with open(out, "a", encoding="utf-8") as f:
        if prev_id and prev_id not in done and prev_n >= 2:
            f.write(json.dumps({"id": prev_id, "err": f"killed {prev_n}x while reading, skipped"}) + "\n")
            f.flush()
            done.add(prev_id)
        for name in sorted(os.listdir(imgdir)):
            if not name.endswith(".img"):
                continue
            iid = name[:-4]
            if iid in done or (only is not None and iid not in only):
                continue
            with open(cur, "w") as c:
                c.write(f"{iid} {prev_n + 1 if iid == prev_id else 1}\n")
            rec = {"id": iid}
            try:
                im = Image.open(os.path.join(imgdir, name))
                im.seek(0)
                sw, sh = im.size
                s = scale
                if sw * sh * s * s > max_px:
                    s = math.sqrt(max_px / (sw * sh))
                    rec["src_w"], rec["src_h"], rec["cap"] = sw, sh, round(s / scale, 4)
                    im.draft("RGB", (round(sw * s), round(sh * s)))  # JPEG은 줄여서 풀어 메모리를 아낀다
                if s != 1:
                    im = im.convert("RGB").resize((round(sw * s), round(sh * s)), Image.LANCZOS)
                if scale != 1:
                    rec["scale"] = round(s, 4)
                arr = np.array(im.convert("RGB"))[:, :, ::-1].copy()
                rec["w"], rec["h"] = im.size
                t = time.time()
                r, _ = eng(arr)
                rec["sec"] = round(time.time() - t, 3)
                items = []
                for box, txt, conf in r or []:
                    xs, ys = [p[0] for p in box], [p[1] for p in box]
                    items.append([[round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))], txt, round(float(conf), 3)])
                rec["items"] = items
            except Exception as e:  # noqa: BLE001
                rec["err"] = repr(e)[:200]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    if os.path.exists(cur):
        os.remove(cur)


if __name__ == "__main__":
    main()
