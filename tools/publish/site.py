"""공개 웹(jgaram/nikke-soloraid-cutoff의 GitHub Pages)에 올릴 솔로레이드 데이터를 만든다 (모델 0, 요청 0, 2026-10-16).

python3 -I tools/publish/site.py --out DIR [--prev 이전_data.json]     # DIR/index.html · DIR/analysis.html · DIR/data.json

- index.html: ① 최종 1·3·10% 컷 예측(진행 중 시즌) ② 지금 몇 %면 최종 몇 % 계산기 ③ 지난 시즌 컷.
  tools/publish/site_template.html에 data.json을 넣은 것.
- analysis.html: tools/solo_raid/viz.py와 같은 화면(예측 방법·정확도). 공개용으로 위에 안내를 더하고 비공개 경로·수집 상태를 뺀다.

공개 규칙 (docs/publish.md)
- 숫자와 개수만 내보낸다. 글 원문·사진·닉네임·원글 주소·글 번호는 넣지 않는다. 근거는 출처 종류(집계표·관측 곡선)와 사람 수로만 쓴다.
- 시험 시즌(split.json)의 정답 컷은 넣지 않는다 — split 감시를 켜고 seasons.json은 split.load_seasons()로만 읽는다.
  시험 시즌은 「잠김」으로만 표시하고, 진행 중 예측의 채점(score.json)도 개발 시즌 것만 쓴다.
- 기준 시각은 웹 내용이 바뀐 때다. --prev의 내용 지문(content)이 같으면 그 시각을 그대로 써서 결과 파일이 바뀌지 않게 한다.
- 시즌이 끝났는지는 만들 때 정하지 않는다 (브라우저가 종료 시각으로 판단한다).
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SR = os.path.join(ROOT, "tools", "solo_raid")
D = os.path.join(ROOT, "data", "solo-raid")
sys.path.insert(0, SR)
import split  # noqa: E402

# 두 페이지 맨 아래에 같이 붙이는 출처·권리 문구 (2026-10-16 사용자 결정, 합니다체)
FOOTER = (
    '    <p>코드와 수집 방법: <a href="https://github.com/jgaram/nikke-soloraid-cutoff">github.com/jgaram/nikke-soloraid-cutoff</a>'
    ' (CC0 1.0 — 이 프로젝트가 만든 코드와 숫자에만 해당합니다).</p>\n'
    '    <p>컷 정보는 커뮤니티 이용자분들이 공개로 올린 순위 화면에서 모았습니다. 예측값이라 실제 컷과 다를 수 있습니다.</p>\n'
    '    <p>「승리의 여신: 니케」와 게임 속 이름·내용의 권리는 SHIFT UP과 Level Infinite에 있습니다.'
    ' 이 사이트는 게임사와 관계없는 비공식 팬 프로젝트입니다.</p>\n'
)

SERVERS = ["KR", "JP", "GL"]
PCTS = ["1", "3", "10"]
# 지난 시즌 컷과 계산기가 고르는 지점 (2026-10-16 사용자 결정 — 서버를 정해 두지 않고 아래 기준을 넘는 지점만 그린다)
CAND = ["1", "3", "10", "30"]
PAST_MIN = 2   # 지난 시즌: 값이 있는(잠기지 않은) 시즌이 이만큼 있는 지점만 열·선
# 계산기: 최종이 그 지점의 ÷near~×near로 끝난 관측 칸으로 곡선을 대 본다 (곡선을 맞춘 그 자료라 표본 안 점검이다).
# 넘으면 실선, 관측이 있는데 못 넘으면 흐린 점선(근거 부족), 관측 0이면 그리지 않는다.
CONV_RULE = {"near": 1.3, "n": 30, "seasons": 4, "bias": 1.10, "cov": [0.65, 0.95]}
KST = dt.timezone(dt.timedelta(hours=9))
INPUTS = ["data/solo-raid", "tools/solo_raid", "tools/publish"]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def r2(v):
    return None if v is None else round(float(v), 2)


def warn(msg):
    """Actions 화면에 노란 경고로 남긴다 (웹은 계속 만든다)."""
    print(f"::warning::site.py: {msg}", flush=True)


def iso(s):
    """'YYYY-MM-DD HH:MM[:SS]' (KST) → 브라우저가 읽는 ISO 시각."""
    return s[:16].replace(" ", "T") + ":00+09:00"


def season_json(s):
    p = os.path.join(D, s, "state", "season.json")
    return jload(p) if os.path.exists(p) else {}


def past(viz_seasons, reg):
    """시즌마다 최종 컷과 출처 (지점은 CAND). 분석된 개발 시즌은 정답 표, 그 밖 개발 시즌은 seasons.json 한섭 3·10%와
    server_cutoffs.csv(서버 셋, 글섭 GLOBAL은 split이 GL로 바꿔 줌), 시험 시즌은 잠김. 출처(basis): table = 커뮤니티 집계표, curve = 커뮤니티 관측 점수로 그린 곡선."""
    table = {r["name"]: r for r in split.load_seasons()}   # 시험 시즌은 kr_cut_*가 빠져 온다
    csvc = {}   # 정답 표에 없는 시즌용 집계표 {(시즌 번호, 서버): {지점: 점수}} (시험 시즌 행은 빠져 온다)
    for c in split.load_server_cutoffs():
        if c["percentile"] in CAND and c.get("score_100m"):
            csvc.setdefault((split.snum(c["season"]), c["server"]), {}).setdefault(c["percentile"], c["score_100m"])
    out = []
    for s, r in sorted(reg.items(), key=lambda kv: split.snum(kv[0])):
        t = table.get(s, {})
        sj = season_json(s)
        start = t.get("start") or sj.get("start") or ""
        end = t.get("end") or sj.get("end") or ""
        row = {"season": s, "n": split.snum(s), "boss": r["boss"], "start": start[:10], "end": iso(end) if end else None,
               "locked": split.is_test(s), "cuts": {}, "basis": {}}
        if row["locked"]:
            out.append(row)
            continue
        if s in viz_seasons:
            key = viz_seasons[s]["key"]
            for sv in SERVERS:
                c = {p: c for p, c in key.get(sv, {}).items() if p in CAND}
                if c:
                    row["cuts"][sv] = {p: r2(v["v"]) for p, v in c.items()}
                    row["basis"][sv] = {p: v["basis"] for p, v in c.items()}
        else:
            for sv in SERVERS:
                c = {p: r2(v) for p, v in csvc.get((row["n"], sv), {}).items()}
                if sv == "KR":
                    c.update({p: r2(t[f"kr_cut_{p}"]) for p in ("3", "10") if t.get(f"kr_cut_{p}")})
                if c:
                    row["cuts"][sv] = c
                    row["basis"][sv] = {p: "table" for p in c}
        out.append(row)
    return out


def past_pcts(rows):
    """서버마다 지난 시즌 열·선을 넣을 지점: 값이 있는 시즌이 PAST_MIN개 이상. 그런 지점이 없으면 값이 있는 지점을 표로만."""
    out = {}
    for sv in SERVERS:
        n = {p: sum(1 for r in rows if p in r["cuts"].get(sv, {})) for p in CAND}
        out[sv] = [p for p in CAND if n[p] >= PAST_MIN] or [p for p in CAND if n[p]]
    return out


def method_label(o):
    """예측에 쓴 prior를 사람이 읽는 말로 (예측 JSON의 run.prior_path). 절대 경로는 내보내지 않는다."""
    p = ((o.get("run") or {}).get("prior_path") or "").replace("\\", "/")
    m = re.search(r"/(S\d+)/state/prior\.json$", p)
    if m:
        return f"{m.group(1)} 한 시즌 prior"
    if p.endswith("prior_dev.json"):
        return "여러 시즌 prior"
    return "예측 도구 기본값"


def default_method():
    """진행 중 예측이 아직 없을 때 쓰는 이름: predict_run.sh의 기본 prior (읽기만 한다)."""
    try:
        with open(os.path.join(SR, "predict_run.sh"), encoding="utf-8") as f:
            m = re.search(r"^PRIOR=\$ROOT/data/solo-raid/(S\d+)/state/prior\.json", f.read(), re.M)
        if m:
            return f"{m.group(1)} 한 시즌 prior"
    except OSError:
        pass
    return "예측 도구 기본값"


def one_pred(path):
    o = jload(path)
    T = float(o["T"])
    srv = {}
    for sv in SERVERS:
        x = (o.get("servers") or {}).get(sv)
        if not x:
            continue
        pts = {}
        for p_ in PCTS:
            q = (x.get("points") or {}).get(f"{float(p_):.1f}")
            if q and q.get("median") is not None and q.get("i80") and None not in q["i80"]:
                pts[p_] = {"m": r2(q["median"]), "lo": r2(q["i80"][0]), "hi": r2(q["i80"][1])}
        w = x.get("warning") or {}
        srv[sv] = {"grade": w.get("level"), "n_persons": round(x.get("n_persons") or 0), "pts": pts}
    return {"T": round(T, 2), "T_kst": o.get("T_kst"), "method": method_label(o), "srv": srv}


def live(reg):
    """predict/가 있는 시즌. 예측은 시험 시즌도 싣고, 채점(정답)은 개발 시즌만 싣는다.
    파일 하나가 깨져도 그 파일(또는 그 시즌)만 빼고 경고를 남긴다 — 나머지 웹은 계속 갱신된다."""
    out = []
    for pdir in sorted(glob.glob(os.path.join(D, "S*", "predict")), key=lambda p: split.snum(os.path.basename(os.path.dirname(p)))):
        s = os.path.basename(os.path.dirname(pdir))
        try:
            sj = season_json(s)
            c0 = dt.datetime.strptime(sj["c_start"][:16], "%Y-%m-%d %H:%M")
            end = dt.datetime.strptime(sj["end"][:16], "%Y-%m-%d %H:%M")
        except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as e:
            warn(f"{s}/state/season.json을 읽지 못해 진행 중 예측에서 뺐다 ({type(e).__name__}: {e})")
            continue
        preds = []
        for p in glob.glob(os.path.join(pdir, "T*", "predict_T*.json")):
            try:
                preds.append(one_pred(p))
            except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as e:
                warn(f"{os.path.relpath(p, ROOT)}를 읽지 못해 뺐다 ({type(e).__name__}: {e})")
        preds.sort(key=lambda r: r["T"])
        truth = {}
        sc = os.path.join(pdir, "score.json")
        if not split.is_test(s) and os.path.exists(sc):
            try:
                for r in jload(sc).get("rows", []):
                    p_ = f"{r['pct']:g}"
                    if p_ in PCTS and r.get("truth") is not None:
                        truth.setdefault(r["server"], {})[p_] = r2(r["truth"])
            except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as e:
                warn(f"{s}/predict/score.json을 읽지 못해 실제 컷 비교를 뺐다 ({type(e).__name__}: {e})")
                truth = {}
        out.append({"season": s, "boss": sj.get("boss") or (reg.get(s) or {}).get("boss"), "locked": split.is_test(s),
                    "c_start": iso(sj["c_start"]), "end": iso(sj["end"]), "H": round((end - c0).total_seconds() / 3600),
                    "preds": preds, "truth": truth})
    return out


def latest_season(reg):
    """state/season.json이 있는 가장 최근 시즌의 일정. 진행 중이면 계산기가 지금 시각을 그 시즌 기준으로 보여 준다 (예측이 아직 없어도).
    등록부(registry.json)는 분석 도구가 끝난 시즌만 올리므로, 진행 중 시즌은 폴더의 state/season.json으로 찾는다 (2026-10-16, S42)."""
    names = set(reg) | {os.path.basename(os.path.dirname(os.path.dirname(p))) for p in glob.glob(os.path.join(D, "S*", "state", "season.json"))}
    for s in sorted(names, key=split.snum, reverse=True):
        sj = season_json(s)
        if sj.get("c_start") and sj.get("end"):
            return {"season": s, "boss": sj.get("boss") or (reg.get(s) or {}).get("boss"), "c_start": iso(sj["c_start"]), "end": iso(sj["end"])}
    return None


def conv(vdata):
    """계산기용 % 변환 (viz의 여러 시즌 평균 곡선, 0.5시간 간격). 행 = [시작 후 시간, log10(최종 % ÷ 지금 %), 80% 구간용 sd]."""
    cv = vdata["conv"]
    seas = vdata["season_list"]   # viz가 개발 시즌만 쓴다 (시험 시즌이면 viz.build가 멈춤)
    # 지난 시즌 겹쳐 보기 (2026-10-16 사용자 요청): 시즌 수준 배수와, 최종 1·3·10% 근처(÷1.3~×1.3)로 끝난 관측 칸
    # 관측 = [시즌, 시작 후 시간, 그때 %, 최종 %] — 숫자만 (사람·글 정보 없음)
    # 지점마다 곡선 점검 (CONV_RULE): 관측 칸의 log10(최종 ÷ 그때)를 같은 시각 곡선과 견준다
    near = CONV_RULE["near"]
    obs = {sv: [] for sv in SERVERS}
    res = {(sv, p): [] for sv in SERVERS for p in CAND}
    for s in seas:
        for u in cv["units"][s]:
            if u["sv"] not in obs or u["P"] is None:
                continue
            f = u["x"] * 10 ** u["P"]
            hit = [p for p in CAND if float(p) / near <= f <= float(p) * near]
            if not hit:
                continue
            obs[u["sv"]].append([s, round(u["h"], 1), round(u["x"], 3), round(f, 3)])
            rows = cv["srv"][u["sv"]]
            P, sd = min(rows, key=lambda r: abs(r[0] - u["h"]))[1:3]
            for p in hit:
                res[(u["sv"], p)].append((s, u["P"] - P, sd))
    check = {sv: {} for sv in SERVERS}
    for (sv, p), L in res.items():
        if not L:
            continue
        bias = 10 ** statistics.median(d for _, d, _ in L)
        cov = sum(abs(d) <= 1.2816 * sd for _, d, sd in L) / len(L)
        nS = len({x[0] for x in L})
        ok = (len(L) >= CONV_RULE["n"] and nS >= CONV_RULE["seasons"] and 1 / CONV_RULE["bias"] <= bias <= CONV_RULE["bias"]
              and CONV_RULE["cov"][0] <= cov <= CONV_RULE["cov"][1])
        check[sv][p] = {"n": len(L), "seasons": nS, "bias": round(bias, 3), "cov": round(cov, 2), "ok": ok}
    return {"H": cv["H"], "seasons": seas, "rule": CONV_RULE, "check": check,
            "borrow": {sv: (cv["m_src"].get(sv) or "").startswith("borrow") for sv in SERVERS},
            "levels": {s: cv["levels"][s] for s in seas if s in cv["levels"]},
            "start": {s: iso(vdata["seasons"][s]["c_start"]) for s in seas},
            "obs": obs,
            "srv": {sv: [[r[0], r[1], r[2]] for i, r in enumerate(cv["srv"][sv]) if i % 2 == 0] for sv in SERVERS}}


def head_time():
    """지금 HEAD 커밋 시각 (KST). 웹 입력에 커밋 안 된 변경이 있으면(손으로 돌릴 때) 지금 시각."""
    import subprocess
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cI", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--"] + INPUTS, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        if out and not dirty:
            return dt.datetime.fromisoformat(out).astimezone(KST).strftime("%Y-%m-%d %H:%M")
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M")


def fingerprint(data, vdata):
    """웹 내용의 지문: 넣는 자료 + 페이지 틀. 이것이 같으면 웹도 같다."""
    h = hashlib.sha1()
    h.update(json.dumps({"data": data, "vdata": vdata}, ensure_ascii=False).encode("utf-8"))   # 순서는 입력 파일 순서대로라 같은 입력이면 같다
    for f in (os.path.join(HERE, "site_template.html"), os.path.join(SR, "viz_template.html"), os.path.abspath(__file__)):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def asof_for(content, prev):
    """기준 시각 = 웹 내용이 마지막으로 바뀐 때. 이전 웹과 지문이 같으면 이전 시각을 그대로 쓴다."""
    if prev and os.path.exists(prev):
        try:
            p = jload(prev)
            if p.get("content") == content and p.get("asof"):
                return p["asof"]
        except (OSError, ValueError):
            pass
    return head_time()


def build(prev=None):
    split.install_guard(final=False)
    import viz   # 감시를 켠 뒤에 불러온다
    vdata = viz.build(None)
    # 공개 분석 페이지에는 수집 진행 상태를 싣지 않는다
    for sv in vdata["notes"]:
        vdata["notes"][sv]["notes"] = [n for n in vdata["notes"][sv]["notes"] if not n.startswith("아직 분석에 들지 않은")]
    reg = {r["season"]: r for r in jload(os.path.join(D, "registry.json"))["seasons"]}
    lv = live(reg)
    methods = sorted({p["method"] for x in lv for p in x["preds"]})
    pst = past(vdata["seasons"], reg)
    data = {"past": pst, "past_pcts": past_pcts(pst), "live": lv, "latest": latest_season(reg), "conv": conv(vdata),
            "analysis": {"seasons": vdata["season_list"], "live_method": " · ".join(methods) if methods else default_method()}}
    content = fingerprint(data, vdata)
    data = {"asof": asof_for(content, prev), "content": content, **data}
    return data, vdata


def js(obj):
    """<script> 안에 넣는 JSON. 자료에 「</」가 있어도 스크립트가 끊기지 않게 한다."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def analysis_html(vdata, data):
    with open(os.path.join(SR, "viz_template.html"), encoding="utf-8") as f:
        tpl = f.read()
    seas = vdata["season_list"]
    m = data["analysis"]["live_method"]
    head = ('<p class="muted"><a href="./">← 첫 화면 (컷 예측 · 지금 몇 %면 최종 몇 % · 지난 시즌 컷)</a></p>\n'
            '<p class="note">이 페이지는 예측 방법을 검증하는 분석 화면입니다. '
            f'아래 ①은 첫 화면의 계산기와 같은 「여러 시즌 % 변환」이고, ②의 정확도도 <b>여러 시즌 prior</b>로 잰 것입니다. '
            f'첫 화면의 진행 중 컷 예측은 이와 다른 <b>{m}</b>로 내며, 그 방법의 정확도는 아직 검증 중입니다.</p>\n')
    sub = "개발 시즌 {}~{} ({}개) · {} KST 기준".format(seas[0], seas[-1], len(seas), data["asof"])
    stamp = "정답 표는 개발 시즌만 씁니다 — 시험 시즌(잠근 시즌)은 열지 않았습니다."
    foot = '  <footer class="muted" style="margin-top:28px">\n' + FOOTER + "  </footer>\n"
    tail = ("<script>\n"
            f'document.getElementById("sub").textContent = {js(sub)};\n'
            f'document.getElementById("stamp").textContent = {js(stamp)};\n'
            "</script>\n")
    return (tpl.replace("/*DATA*/", js(vdata)).replace("__ASOF__", data["asof"])
            .replace("<main>", "<main>\n" + head, 1).replace("</main>", foot + "</main>", 1)
            .replace("</body>", tail + "</body>", 1))


def main():
    ap = argparse.ArgumentParser(description="공개 웹 페이지(index.html·analysis.html·data.json)를 만든다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev", help="지금 공개된 data.json (내용이 같으면 기준 시각을 그대로 둔다)")
    a = ap.parse_args()
    data, vdata = build(a.prev)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(os.path.join(HERE, "site_template.html"), encoding="utf-8") as f:
        html = f.read().replace("/*DATA*/", js(data)).replace("<!--FOOTER-->\n", FOOTER, 1)
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(a.out, "analysis.html"), "w", encoding="utf-8") as f:
        f.write(analysis_html(vdata, data))
    with open(os.path.join(a.out, ".nojekyll"), "w") as f:
        f.write("")
    n_live = sum(len(x["preds"]) for x in data["live"])
    print(f"{a.out}: 기준 {data['asof']} (지문 {data['content']}) · 지난 시즌 {len(data['past'])}개 (잠김 {sum(r['locked'] for r in data['past'])}) · "
          f"진행 중 예측 시즌 {len(data['live'])}개, 시점 {n_live}개 · 분석 {', '.join(vdata['season_list'])}")


if __name__ == "__main__":
    main()
