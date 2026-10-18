"""웹에 싣는 시즌별 보스 그림을 받아 둔다 (2026-10-18 사용자 요청, 모델 0).

python3 -I tools/publish/boss_icons.py [--from DIR] [--check]

- 그림은 enikk(https://enikk.app)의 보스 그림을 사용자의 공개 레포 jgaram/nikke-analysis가 이미 받아 둔 것을 가져온다
  (256px WebP, `data/assets/icons/bosses/`). enikk는 한 사람이 운영하는 사이트라 거기에 다시 요청하지 않는다.
  시즌 ↔ 그림 이름은 그 레포의 `data/processed/soloraid_seasons.csv`(`season`·`boss_ko`·`boss_image`)가 정한다.
- 이 레포가 아는 시즌(seasons.json · registry.json · S*/state/season.json) 가운데 그림이 없는 것만 받아
  `data/solo-raid/bosses/S{n}.webp`에 쓰고, `bosses.json`에 출처를 적는다. 있는 그림은 다시 받지 않는다.
- 약점 속성(`weak_element` — Fire·Water·Electric·Iron·Wind)도 `bosses.json`의 `weak`에 적는다 (2026-10-18 사용자 요청,
  웹 지난 시즌 표의 꼬리표). 그림은 있는데 `weak`가 없는 시즌은 표만 다시 읽어 채운다.
- 보스 이름이 이 레포와 다르면(시즌 번호가 어긋났을 수 있음) 받지 않고 경고만 한다.
- --from: nikke-analysis 로컬 클론 (주지 않으면 GitHub raw에서 받는다)
- --check: 받지 않고 무엇이 없는지만 보인다

웹(site.py)은 그림이 있는 시즌에만 그림을 붙이고, 없으면 이름만 쓴다. 규칙은 docs/publish.md 「보스 그림」.
"""
import argparse
import csv
import glob
import io
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
D = os.path.join(ROOT, "data", "solo-raid")
OUT = os.path.join(D, "bosses")
RAW = "https://raw.githubusercontent.com/Jgaram/nikke-analysis/main/"
CSV = "data/processed/soloraid_seasons.csv"
IMG = "data/assets/icons/bosses/{}.webp"
UA = "nikke-community-research/0.1 (+https://github.com/jgaram/nikke-community)"
MAX_BYTES = 200_000


def fetch(src, rel):
    if src:
        with open(os.path.join(src, rel), "rb") as f:
            return f.read()
    req = urllib.request.Request(RAW + rel, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def known_seasons():
    """이 레포가 아는 시즌 → 보스 이름."""
    out = {}
    for s in json.load(open(os.path.join(D, "seasons.json"), encoding="utf-8")):
        out[s["name"]] = s.get("boss")
    for s in json.load(open(os.path.join(D, "registry.json"), encoding="utf-8"))["seasons"]:
        out.setdefault(s["season"], s.get("boss"))
    for p in glob.glob(os.path.join(D, "S*", "state", "season.json")):
        sj = json.load(open(p, encoding="utf-8"))
        out.setdefault(sj.get("name") or os.path.basename(os.path.dirname(os.path.dirname(p))), sj.get("boss"))
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))


def norm(s):
    return (s or "").replace(" ", "")


def main():
    ap = argparse.ArgumentParser(description="시즌별 보스 그림을 nikke-analysis에서 받아 둔다")
    ap.add_argument("--from", dest="src", help="nikke-analysis 로컬 클론 (없으면 GitHub raw)")
    ap.add_argument("--check", action="store_true", help="받지 않고 없는 것만 보인다")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    meta_p = os.path.join(OUT, "bosses.json")
    meta = json.load(open(meta_p, encoding="utf-8")) if os.path.exists(meta_p) else {}
    want = [s for s in known_seasons() if not os.path.exists(os.path.join(OUT, f"{s}.webp"))]
    no_weak = [s for s in known_seasons() if s not in want and not (meta.get(s) or {}).get("weak")]
    if not want and not no_weak:
        print("보스 그림: 모두 있음")
        return
    if a.check:
        print("보스 그림 없음: " + (", ".join(want) or "-") + " · 약점 없음: " + (", ".join(no_weak) or "-"))
        return
    rows = {f"S{int(r['season'])}": r for r in csv.DictReader(io.StringIO(fetch(a.src, CSV).decode("utf-8")))}
    names = known_seasons()
    got, miss = [], []
    for s in no_weak:
        r = rows.get(s)
        if r and r.get("weak_element") and (not names.get(s) or norm(r.get("boss_ko")) == norm(names[s])):
            meta.setdefault(s, {})["weak"] = r["weak_element"]
    for s in want:
        r = rows.get(s)
        if not r or not r.get("boss_image"):
            miss.append(f"{s}: nikke-analysis에 아직 없음")
            continue
        if names.get(s) and norm(r.get("boss_ko")) != norm(names[s]):
            miss.append(f"{s}: 보스 이름이 다름 (여기 {names[s]} · nikke-analysis {r.get('boss_ko')}) — 받지 않음")
            continue
        try:
            b = fetch(a.src, IMG.format(r["boss_image"]))
        except OSError as e:
            miss.append(f"{s}: {r['boss_image']} 받지 못함 ({e})")
            continue
        if not (b[:4] == b"RIFF" and b[8:12] == b"WEBP") or len(b) > MAX_BYTES:
            miss.append(f"{s}: {r['boss_image']}가 WebP가 아니거나 너무 큼 ({len(b):,}바이트)")
            continue
        with open(os.path.join(OUT, f"{s}.webp"), "wb") as f:
            f.write(b)
        meta[s] = {"boss": r.get("boss_ko"), "image": r["boss_image"], "from": "enikk.app (jgaram/nikke-analysis)"}
        if r.get("weak_element"):
            meta[s]["weak"] = r["weak_element"]
        got.append(s)
    meta = dict(sorted(meta.items(), key=lambda kv: int(kv[0][1:])))
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"보스 그림: 받음 {len(got)}개" + (f" ({got[0]}~{got[-1]})" if got else ""))
    for m in miss:
        print("  " + m, file=sys.stderr)


if __name__ == "__main__":
    main()
