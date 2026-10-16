"""솔로레이드 2단계 탐색 2.1 커버리지 지도: 서버 × 시간대 × % 구간별 관측 수 (모델 0, 요청 0).

python3 -I tools/solo_raid/explore_coverage.py STATE [--out DIR] [--bins 0.5,1,2,3.5,7,15,40] [--at 24,48]

입력 (읽기만 한다)
- STATE/season.json  `c_start`(챌린지 시작), `end`
- STATE/core.jsonl   핵심 관측 표 (build_core.py). 기본 집합 `use`만 센다
- STATE/obs.jsonl    사진 줄의 게시 시각 post_t (예측 시점에 쓸 수 있었는지 = 게시 시각 ≤ T)

세는 법
- 시간: `h`(챌린지 시작 후 시간). 두 보기 — AB: 시각 등급 A·B만 (h = 촬영·말한 시각),
  all: 전체 (C는 h = 게시 시각 = 상한). 종료 뒤(h ≥ 챌린지 길이)는 따로: post(확실, day_lo == post)·post?(C 상한만 종료 뒤).
- 시간대 폭 6시간·12시간 두 가지. 6시간 칸은 05·11·17·23시에 끊긴다 (h 0 = 05:00).
- 칸마다 줄 수(`use`)와 독립 수(그 줄들의 `moment` 가짓수)를 센다. 서버는 core의 server만 쓴다 (미상을 채우지 않는다).
- 예측 시점 T까지 쓸 수 있는 관측: 게시 시각 ≤ T인 use 줄. 촬영 시각이 확실한 A·B와 상한만 아는 C를 나눠 센다.
- 목표 % 근처 창: 1%·3%·10%의 ×/÷1.5 (0.67–1.5, 2–4.5, 6.67–15).
- 시각대 주기: KST 시각(0~23시)별 — A 촬영 시각(독립), use 줄의 글 게시 시각(글 수), 시즌 안 사진 글 전체의 게시 시각(기준).

출력: 표준 출력에 표. --out DIR이면 coverage.json과 coverage.html(칸 색칠 그림)을 쓴다. state/에는 쓰지 않는다.
"""
import argparse
import collections
import datetime as dt
import json
import os

SERVERS = ("KR", "JP", "GL", None)
SNAME = {"KR": "한섭", "JP": "일섭", "GL": "글섭", None: "미상"}


import importlib.util as _ilu  # noqa: E402
_sp = _ilu.spec_from_file_location("split", os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.py"))
split = _ilu.module_from_spec(_sp)
_sp.loader.exec_module(split)

def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def pct_bin(p, edges):
    for k, e in enumerate(edges):
        if p < e:
            return k
    return len(edges)


def bin_labels(edges):
    lab = [f"<{edges[0]:g}"]
    lab += [f"{a:g}-{b:g}" for a, b in zip(edges, edges[1:])]
    lab.append(f"{edges[-1]:g}+")
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out")
    ap.add_argument("--bins", default="0.5,1,2,3.5,7,15,40")
    ap.add_argument("--at", default="24,48")
    split.add_final_arg(ap)
    a = ap.parse_args()
    split.guard_args(a)   # 시험 시즌 정답 감시 (할 일 3.0b)
    edges = [float(x) for x in a.bins.split(",")]
    plab = bin_labels(edges)
    at = [float(x) for x in a.at.split(",")]
    season = json.load(open(os.path.join(a.state, "season.json"), encoding="utf-8"))
    c0 = ts(season["c_start"])
    clen = (ts(season["end"]) - c0).total_seconds() / 3600
    clen = round(clen)  # 04:59:59 끝 → 120
    core = jl(os.path.join(a.state, "core.jsonl"))
    obs = jl(os.path.join(a.state, "obs.jsonl"))
    U = [r for r in core if r["use"]]
    for r in U:
        o = obs[r["oi"]] if r["src"] == "photo" else None
        r["_hp"] = (ts(o["post_t"]) - c0).total_seconds() / 3600 if o and o.get("post_t") else r["h"]
        r["_pb"] = pct_bin(r["pct"], edges)

    def tbin(r, w, view):
        if view == "AB" and r["t_grade"] == "C":
            return None
        h = r["h"]
        if h >= clen:
            return "post" if (r["t_grade"] != "C" or r["day_lo"] == "post") else "post?"
        return int(h // w)

    res = {"season": season["name"], "c_start": season["c_start"], "challenge_hours": clen,
           "pct_bins": plab, "n_use": len(U), "n_use_moments": len({r["moment"] for r in U}), "grid": {}}

    def table(w, view, srv):
        cells = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in U:
            if r["server"] != srv:
                continue
            b = tbin(r, w, view)
            if b is None:
                continue
            cells[b][r["_pb"]].append(r)
        tb = [k for k in range(int(clen // w))] + ["post", "post?"]
        out = {}
        for b in tb:
            row = []
            for k in range(len(plab)):
                rs = cells[b][k]
                row.append([len(rs), len({x["moment"] for x in rs})])
            out[b] = row
        return out

    def hlabel(b, w):
        if isinstance(b, str):
            return {"post": "종료후", "post?": "종료후?"}[b]
        s = int(b * w)
        clock = (5 + s) % 24
        return f"{s:>3}-{s + w:<3}h d{s // 24 + 1} {clock:02d}시"

    for w in (6, 12):
        for view in ("AB", "all"):
            for srv in SERVERS:
                t = table(w, view, srv)
                res["grid"][f"{w}h/{view}/{srv}"] = {str(k): v for k, v in t.items()}
                tot = sum(c[0] for v in t.values() for c in v)
                if tot == 0:
                    continue
                print(f"\n== {SNAME[srv]} · {w}시간 · {'A·B만' if view == 'AB' else '전체(C 상한)'} — 칸: 줄/독립")
                print(f"{'':22}" + "".join(f"{l:>9}" for l in plab) + f"{'계':>9}")
                for b, v in t.items():
                    if sum(c[0] for c in v) == 0 and b == "post?":
                        continue
                    line = "".join(f"{(str(c[0]) + '/' + str(c[1])) if c[0] else '.':>9}" for c in v)
                    print(f"{hlabel(b, w):22}{line}{sum(c[0] for c in v):>5}/{sum(c[1] for c in v):<3}")

    # 빈 칸 (첫 48시간, 6시간 칸, 독립 0) — A·B만 / 전체
    print("\n== 첫 48시간 6시간 칸 중 독립 관측 0인 % 구간 (서버별 · AB/전체)")
    empties = {}
    for srv in SERVERS:
        for view in ("AB", "all"):
            g = res["grid"][f"6h/{view}/{srv}"]
            e = []
            for b in range(8):
                z = [plab[k] for k, c in enumerate(g[str(b)]) if c[1] == 0]
                e.append((hlabel(b, 6).strip(), z))
            empties[f"{srv}/{view}"] = e
            nz = sum(len(z) for _, z in e)
            print(f"{SNAME[srv]} {view}: 빈 칸 {nz}/{8 * len(plab)}")
            for lab, z in e:
                if len(z) < len(plab):
                    print(f"   {lab}: 빔 {', '.join(z)}")
                else:
                    print(f"   {lab}: 전부 빔")
    res["empty_first48_6h"] = empties

    # 예측 시점 T까지 쓸 수 있는 관측
    targets = {"1%": (1 / 1.5, 1.5), "3%": (2.0, 4.5), "10%": (10 / 1.5, 15.0)}
    res["available"] = {}
    for T in at:
        print(f"\n== 게시 시각 ≤ {T:g}h ({(c0 + dt.timedelta(hours=T)):%m-%d %H시}) 에 쓸 수 있는 관측 — 독립 수 A·B / C(상한)")
        print(f"{'':6}" + "".join(f"{l:>10}" for l in plab) + "".join(f"{k + '창':>10}" for k in targets) + f"{'계':>10}")
        res["available"][str(T)] = {}
        for srv in SERVERS:
            rs = [r for r in U if r["server"] == srv and r["_hp"] <= T + 1e-9]
            ab = [r for r in rs if r["t_grade"] != "C"]
            cc = [r for r in rs if r["t_grade"] == "C"]
            mm = lambda xs: len({x["moment"] for x in xs})
            cols = []
            for k in range(len(plab)):
                cols.append((mm([x for x in ab if x["_pb"] == k]), mm([x for x in cc if x["_pb"] == k])))
            for lo, hi in targets.values():
                cols.append((mm([x for x in ab if lo <= x["pct"] <= hi]), mm([x for x in cc if lo <= x["pct"] <= hi])))
            cols.append((mm(ab), mm(cc)))
            res["available"][str(T)][str(srv)] = {"cols": plab + list(targets) + ["total"], "AB_C": cols,
                                                   "rows": len(rs)}
            print(f"{SNAME[srv]:6}" + "".join(f"{str(x) + '/' + str(y):>10}" for x, y in cols))

    # 시각대 주기 (챌린지 중 0 ≤ h < clen)
    hourA = collections.Counter()
    seenm = set()
    for r in U:
        if r["t_grade"] == "A" and 0 <= r["h"] < clen and r["moment"] not in seenm:
            seenm.add(r["moment"])
            hourA[int((5 + r["h"]) % 24)] += 1
    hourP = collections.Counter()
    seenp = set()
    for r in U:
        k = (r["site"], r["no"])
        if 0 <= r["_hp"] < clen and k not in seenp:
            seenp.add(k)
            hourP[int((5 + r["_hp"]) % 24)] += 1
    hourB = collections.Counter()
    seenb = set()
    for o in obs:
        if o.get("season") != "in" or not o.get("post_t"):
            continue
        k = (o["site"], o["no"])
        hp = (ts(o["post_t"]) - c0).total_seconds() / 3600
        if 0 <= hp < clen and k not in seenb:
            seenb.add(k)
            hourB[int((5 + hp) % 24)] += 1
    print(f"\n== 챌린지 중 KST 시각별: A 촬영(독립 {sum(hourA.values())}) · use 글 게시({sum(hourP.values())}글) · 시즌 안 사진 글 전체({sum(hourB.values())}글)")
    print("시   " + "".join(f"{h:>4}" for h in range(24)))
    for name, c in (("A촬영", hourA), ("use글", hourP), ("전체글", hourB)):
        print(f"{name:5}" + "".join(f"{c[h]:>4}" for h in range(24)))
    res["hour_of_day"] = {"A_shot_moments": [hourA[h] for h in range(24)],
                          "use_posts": [hourP[h] for h in range(24)],
                          "in_season_photo_posts": [hourB[h] for h in range(24)]}

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "coverage.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        write_html(os.path.join(a.out, "coverage.html"), res, plab, hlabel)
        print(f"\n쓴 파일: {a.out}/coverage.json · coverage.html")


def write_html(path, res, plab, hlabel):
    cw, ch, lw = 46, 18, 150
    parts = []
    for view in ("AB", "all"):
        for srv in SERVERS:
            g = res["grid"][f"12h/{view}/{srv}"]
            keys = [k for k in g if not (k == "post?" and sum(c[0] for c in g[k]) == 0)]
            W = lw + cw * len(plab) + 10
            H = 24 + ch * len(keys) + 6
            s = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">']
            for j, l in enumerate(plab):
                s.append(f'<text x="{lw + cw * j + cw / 2}" y="16" class="ax">{l}</text>')
            for i, k in enumerate(keys):
                y = 24 + ch * i
                lab = hlabel(int(k), 12) if k.isdigit() else hlabel(k, 12)
                s.append(f'<text x="{lw - 6}" y="{y + ch * 0.7}" class="lab">{lab}</text>')
                for j, c in enumerate(g[k]):
                    n = c[1]
                    lvl = 0 if n == 0 else 1 if n < 3 else 2 if n < 6 else 3 if n < 12 else 4
                    s.append(f'<rect x="{lw + cw * j + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" class="l{lvl}"/>')
                    if n:
                        s.append(f'<text x="{lw + cw * j + cw / 2}" y="{y + ch * 0.7}" class="v{lvl}">{n}</text>')
            s.append("</svg>")
            title = f"{SNAME[srv]} · {'A·B만' if view == 'AB' else '전체 (C는 게시 시각 상한)'}"
            parts.append(f'<figure><figcaption>{title}</figcaption>{"".join(s)}</figure>')
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>S41 커버리지 지도</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--l0:#efede8;--l1:#d6e4f0;--l2:#9fc3e0;--l3:#4f8fc4;--l4:#1f5b93}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#18191b;--fg:#e8e6e1;--mut:#9a9893;--l0:#26282b;--l1:#1f3550;--l2:#2d5680;--l3:#4f8fc4;--l4:#9fc3e0}}}}
:root[data-theme="dark"]{{--bg:#18191b;--fg:#e8e6e1;--mut:#9a9893;--l0:#26282b;--l1:#1f3550;--l2:#2d5680;--l3:#4f8fc4;--l4:#9fc3e0}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px}}
main{{max-width:1100px;margin:auto}} .grid{{display:flex;flex-wrap:wrap;gap:16px}} figure{{margin:0;overflow-x:auto;max-width:100%}}
figcaption{{font-weight:600;margin-bottom:4px}} text{{font-size:11px;fill:var(--fg)}} .ax{{text-anchor:middle;fill:var(--mut)}}
.lab{{text-anchor:end;fill:var(--mut)}} .v1,.v2,.v0{{text-anchor:middle}} .v3,.v4{{text-anchor:middle;fill:#fff}}
.l0{{fill:var(--l0)}} .l1{{fill:var(--l1)}} .l2{{fill:var(--l2)}} .l3{{fill:var(--l3)}} .l4{{fill:var(--l4)}} p{{color:var(--mut)}}
</style></head><body><main><h1>{res['season']} 커버리지 지도</h1>
<p>칸 = 12시간 시간대 × % 구간, 숫자 = 독립 관측 수(moment 가짓수). 기본 집합 use {res['n_use']}줄 · 독립 {res['n_use_moments']}.
h 0 = 챌린지 시작 {res['c_start']} KST. 색: 0 · 1–2 · 3–5 · 6–11 · 12+.</p>
<div class="grid">{''.join(parts)}</div></main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
