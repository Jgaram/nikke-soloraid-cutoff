"""공개 웹(jgaram/nikke-soloraid-cutoff의 GitHub Pages)에 올릴 솔로레이드 데이터를 만든다 (모델 0, 요청 0, 2026-10-16).

python3 -I tools/publish/site.py --out DIR     # DIR/index.html · DIR/analysis.html · DIR/data.json

- index.html: ① 지난 시즌 커트라인 ② 진행 중 시즌 예측. tools/publish/site_template.html에 data.json을 넣은 것.
- analysis.html: tools/solo_raid/viz.py와 같은 화면(예측 방법·정확도). 위에 index로 가는 줄만 더한다.

공개 규칙 (docs/publish.md)
- 숫자와 개수만 내보낸다. 글 원문·사진·닉네임·원글 주소·글 번호는 넣지 않는다. 근거는 「커뮤니티 N건」으로만 쓴다.
- 시험 시즌(split.json)의 정답 컷은 넣지 않는다 — split 감시를 켜고 seasons.json은 split.load_seasons()로만 읽는다.
  시험 시즌은 「잠김」으로만 표시하고, 진행 중 예측의 채점(score.json)도 개발 시즌 것만 쓴다.
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SR = os.path.join(ROOT, "tools", "solo_raid")
D = os.path.join(ROOT, "data", "solo-raid")
sys.path.insert(0, SR)
import split  # noqa: E402

SERVERS = ["KR", "JP", "GL"]
PCTS = ["1", "3", "10"]
KST = dt.timezone(dt.timedelta(hours=9))


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def r2(v):
    return None if v is None else round(float(v), 2)


def past(viz_seasons, reg):
    """시즌마다 최종 컷. 분석된 개발 시즌은 정답 표(서버 셋), 그 밖 개발 시즌은 seasons.json 한섭 3·10%, 시험 시즌은 잠김."""
    table = {r["name"]: r for r in split.load_seasons()}   # 시험 시즌은 kr_cut_*가 빠져 온다
    out = []
    for s, r in sorted(reg.items(), key=lambda kv: split.snum(kv[0])):
        t = table.get(s, {})
        sj = os.path.join(D, s, "state", "season.json")
        start = t.get("start") or (jload(sj).get("start") if os.path.exists(sj) else None) or ""
        row = {"season": s, "n": split.snum(s), "boss": r["boss"], "start": start[:10],
               "locked": split.is_test(s), "cuts": {}, "n_obs": None, "src": None}
        if row["locked"]:
            out.append(row)
            continue
        if s in viz_seasons:
            v = viz_seasons[s]
            row["cuts"] = {sv: {p: r2(c["v"]) for p, c in v["key"].get(sv, {}).items() if p in PCTS} for sv in SERVERS}
            row["n_obs"] = v["n_all"]
            row["src"] = "analysis"
        elif t.get("kr_cut_3") or t.get("kr_cut_10"):
            row["cuts"] = {"KR": {p: r2(t.get(f"kr_cut_{p}")) for p in ("3", "10") if t.get(f"kr_cut_{p}")}}
            row["src"] = "table"
        out.append(row)
    return out


def live(reg):
    """predict/가 있는 시즌. 예측은 시험 시즌도 싣고, 채점(정답)은 개발 시즌만 싣는다."""
    now = dt.datetime.now(KST).replace(tzinfo=None)
    out = []
    for pdir in sorted(glob.glob(os.path.join(D, "S*", "predict")), key=lambda p: split.snum(os.path.basename(os.path.dirname(p)))):
        s = os.path.basename(os.path.dirname(pdir))
        sj = jload(os.path.join(D, s, "state", "season.json"))
        c0 = dt.datetime.strptime(sj["c_start"], "%Y-%m-%d %H:%M:%S")
        end = dt.datetime.strptime(sj["end"], "%Y-%m-%d %H:%M:%S")
        preds = []
        for p in glob.glob(os.path.join(pdir, "T*", "predict_T*.json")):
            o = jload(p)
            T = float(o["T"]) if not isinstance(o["T"], str) else float(re.sub(r"[^\d.]", "", o["T"]))
            srv = {}
            for sv in SERVERS:
                x = (o.get("servers") or {}).get(sv)
                if not x:
                    continue
                pts = {}
                for p_ in PCTS:
                    q = (x.get("points") or {}).get(f"{float(p_):.1f}")
                    if q and q.get("median") is not None:
                        pts[p_] = {"m": r2(q["median"]), "lo": r2(q["i80"][0]), "hi": r2(q["i80"][1])}
                w = x.get("warning") or {}
                srv[sv] = {"grade": w.get("level"), "label": w.get("label"), "n_persons": round(x.get("n_persons") or 0), "pts": pts}
            preds.append({"T": T, "T_kst": o.get("T_kst"), "srv": srv})
        preds.sort(key=lambda r: r["T"])
        truth = {}
        sc = os.path.join(pdir, "score.json")
        if not split.is_test(s) and os.path.exists(sc):
            for r in jload(sc).get("rows", []):
                p_ = f"{r['pct']:g}"
                if p_ in PCTS and r.get("truth") is not None:
                    truth.setdefault(r["server"], {})[p_] = r2(r["truth"])
        out.append({"season": s, "boss": sj.get("boss") or (reg.get(s) or {}).get("boss"), "locked": split.is_test(s),
                    "c_start": sj["c_start"][:16], "end": sj["end"][:16], "H": round((end - c0).total_seconds() / 3600),
                    "done": now > end, "preds": preds, "truth": truth})
    return out


def data_asof():
    """기준 시각 = 웹에 들어가는 자료·도구를 마지막으로 바꾼 커밋 시각 (KST). 같은 커밋이면 같은 웹이 나오게 한다."""
    import subprocess
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", "data/solo-raid", "tools/solo_raid", "tools/publish"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "data/solo-raid", "tools/solo_raid", "tools/publish"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        if out and not dirty:
            return dt.datetime.fromisoformat(out).astimezone(KST).strftime("%Y-%m-%d %H:%M")
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M")


def build():
    split.install_guard(final=False)
    import viz   # 감시를 켠 뒤에 불러온다
    vdata = viz.build(None)
    reg = {r["season"]: r for r in jload(os.path.join(D, "registry.json"))["seasons"]}
    asof = data_asof()
    data = {"asof": asof, "past": past(vdata["seasons"], reg), "live": live(reg),
            "analysis": {"seasons": vdata["season_list"], "stamp": vdata["stamp"]}}
    return data, vdata, asof


def main():
    ap = argparse.ArgumentParser(description="공개 웹 페이지(index.html·analysis.html·data.json)를 만든다")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, vdata, asof = build()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(os.path.join(HERE, "site_template.html"), encoding="utf-8") as f:
        html = f.read().replace("/*DATA*/", json.dumps(data, ensure_ascii=False))
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(SR, "viz_template.html"), encoding="utf-8") as f:
        tpl = f.read()
    nav = '<p class="muted"><a href="./">← 지난 시즌 컷 · 진행 중 예측</a></p>\n'
    ana = (tpl.replace("/*DATA*/", json.dumps(vdata, ensure_ascii=False)).replace("__ASOF__", asof)
           .replace("<main>", "<main>\n" + nav, 1))
    with open(os.path.join(a.out, "analysis.html"), "w", encoding="utf-8") as f:
        f.write(ana)
    with open(os.path.join(a.out, ".nojekyll"), "w") as f:
        f.write("")
    n_live = sum(len(x["preds"]) for x in data["live"])
    print(f"{a.out}: 지난 시즌 {len(data['past'])}개 (잠김 {sum(r['locked'] for r in data['past'])}) · "
          f"진행 중 예측 시즌 {len(data['live'])}개, 시점 {n_live}개 · 분석 {', '.join(vdata['season_list'])}")


if __name__ == "__main__":
    main()
