"""공개 웹(jgaram/nikke-soloraid-cutoff의 GitHub Pages)에 올릴 솔로레이드 데이터를 만든다 (모델 0, 요청 0, 2026-10-16).

python3 -I tools/publish/site.py --out DIR [--prev 이전_data.json]     # DIR/index.html · DIR/analysis.html · DIR/data.json

- index.html: 탭 셋 (2026-10-18 사용자 결정 — 이번 시즌 것과 지난 시즌 것을 나누고, 긴 설명은 「방법」 탭에 모음).
  「이번 시즌」 = 최종 1·3·10% 컷 예측 · 계산기(최종 점수 ↔ 최종 %, 지금 % ↔ 최종 %).
  「지난 시즌」(#past) = 지난 시즌 커트라인 · 실제로 낸 예측(끝난 시즌 채점) · 되돌려 보기 · 시즌별 침수 비교(실제 관측 점).
  「방법」(#method) = 작업 흐름(단계별 코드 위치·공개 범위) · 컷 예측 · 계산기 · 지난 시즌 탭 · 검증과 한계. 데이터 탭에는 꼬리표만 둔다.
  tools/publish/site_template.html에 data.json을 넣은 것.
- analysis.html: tools/solo_raid/viz.py와 같은 화면(예측 방법·정확도). 공개용으로 위에 안내를 더하고 비공개 경로·수집 상태를 뺀다.

공개 규칙 (docs/publish.md)
- 숫자와 개수만 내보낸다. 글 원문·사진·닉네임·원글 주소·글 번호는 넣지 않는다. 근거는 출처 종류(집계표·관측 곡선)와 사람 수로만 쓴다.
- **끝난 시즌 기록은 빠짐없이 싣는다** (2026-10-17 공개 결정 A — 시험 몫이어도). 그래서 지난 시즌 컷은 split 몫과 상관없이
  seasons.json·server_cutoffs.csv 전체(split.load_seasons(final=True))에서 읽고, 진행 중 예측의 채점(score.json)도 있으면 싣는다.
  split 판 2(2026-10-17)에는 잠근 시험 시즌이 없다. 새 시즌의 독립 시험은 끝나기 전에 커밋된 예측을 끝난 뒤 채점하는 것이라 정답 공개와 부딪히지 않는다.
  아직 안 끝난 시즌은 정답이 없다 (지난 시즌 표는 브라우저가 종료 시각으로 숨긴다). split 감시는 그대로 켠다 (정답 파일 읽기 규칙 — 판 2에는 막는 시즌 없음).
- 기준 시각은 웹 내용이 바뀐 때다. --prev의 내용 지문(content)이 같으면 그 시각을 그대로 써서 결과 파일이 바뀌지 않게 한다.
- 시즌이 끝났는지는 만들 때 정하지 않는다 (브라우저가 종료 시각으로 판단한다).
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import math
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
    server_cutoffs.csv(서버 셋, 글섭 GLOBAL은 split이 GL로 바꿔 줌). 공개 결정 A(2026-10-17): 끝난 시즌은 몫과 상관없이 싣는다 (잠김 없음).
    출처(basis): table = 커뮤니티 집계표, curve = 커뮤니티 관측 점수로 그린 곡선."""
    table = {r["name"]: r for r in split.load_seasons(final=True)}
    csvc = {}   # 정답 표에 없는 시즌용 집계표 {(시즌 번호, 서버): {지점: 점수}}
    for c in split.load_server_cutoffs(final=True):
        if c["percentile"] in CAND and c.get("score_100m"):
            csvc.setdefault((split.snum(c["season"]), c["server"]), {}).setdefault(c["percentile"], c["score_100m"])
    out = []
    for s, r in sorted(reg.items(), key=lambda kv: split.snum(kv[0])):
        t = table.get(s, {})
        sj = season_json(s)
        start = t.get("start") or sj.get("start") or ""
        end = t.get("end") or sj.get("end") or ""
        row = {"season": s, "n": split.snum(s), "boss": r["boss"], "start": start[:10], "end": iso(end) if end else None,
               "cuts": {}, "basis": {}}
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


def model_label(md):
    """고정 모델(predict.py --model, 2026-10-18)을 사람이 읽는 말로."""
    tr = md.get("train") or []
    span = f"{tr[0]}~{tr[-1]}" if tr else "?"
    return f"최근 {len(tr)}시즌({span}) 곡선 · 1·3%는 이번 시즌 침수 세기 갱신·구간 넓히기 더함 (모델 {md.get('name', '?')})"


def method_label(o):
    """예측에 쓴 방법을 사람이 읽는 말로 (예측 JSON의 model, 없으면 run.prior_path). 절대 경로는 내보내지 않는다."""
    if o.get("model"):
        return model_label(o["model"])
    p = ((o.get("run") or {}).get("prior_path") or "").replace("\\", "/")
    m = re.search(r"/(S\d+)/state/prior\.json$", p)
    if m:
        return f"{m.group(1)} 한 시즌 곡선"
    if p.endswith("prior_dev.json"):
        return "여러 시즌 곡선"
    return "예측 도구 기본값"


def default_method():
    """진행 중 예측이 아직 없을 때 쓰는 이름: 가장 최근 시즌의 고정 모델 파일 (predict_run.sh 기본 — 읽기만 한다)."""
    ms = []
    for p in glob.glob(os.path.join(D, "analysis", "model_S*.json")):
        k = re.search(r"model_(S\d+)\.json$", p)
        if k:
            ms.append((split.snum(k.group(1)), p))
    if ms:
        try:
            m = jload(max(ms)[1])
            return model_label({"train": m.get("borrow"), "name": m.get("name")})
        except (OSError, ValueError, KeyError):
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
                pts[p_] = {"m": r2(q["median"]), "lo": r2(q["i80"][0]), "hi": r2(q["i80"][1]), "near": q.get("near_obs")}
        # 계산기용 곡선: 예측 JSON의 모든 지점 (% 오름차순) [%, 가운데, 80% 아래, 80% 위, 근처(×½~×2) 옮긴 관측 수] (2026-10-17)
        curve = []
        for k_, q in sorted((x.get("points") or {}).items(), key=lambda kv: float(kv[0])):
            if q and q.get("median") and q.get("i80") and None not in q["i80"] and min(q["i80"]) > 0:
                curve.append([float(k_), r2(q["median"]), r2(q["i80"][0]), r2(q["i80"][1]), q.get("near_obs")])
        w = x.get("warning") or {}
        # 어느 서버에도 관측이 없어 S41 곡선을 그대로 쓴 값은 싣지 않는다 (2026-10-17 사용자 결정 — 관측 1개 이상부터 표시).
        # 다른 서버 관측을 빌린 값(donor가 서버 이름)은 그대로 싣는다.
        nodata = x.get("donor") == "S41"
        if nodata:
            pts, curve = {}, []
        srv[sv] = {"grade": w.get("level"), "n_persons": round(x.get("n_persons") or 0), "nodata": nodata, "pts": pts, "curve": curve}
    return {"T": round(T, 2), "T_kst": o.get("T_kst"), "method": method_label(o), "srv": srv}


def live(reg):
    """predict/가 있는 시즌. 예측과, 끝난 뒤 채점(score.json)이 있으면 실제 컷도 싣는다 (공개 결정 A — 몫과 상관없이).
    파일 하나가 깨져도 그 파일(또는 그 시즌)만 빼고 경고를 남긴다 — 나머지 웹은 계속 갱신된다."""
    out = []
    for pdir in sorted(glob.glob(os.path.join(D, "S*", "predict")), key=lambda p: split.snum(os.path.basename(os.path.dirname(p)))):
        s = os.path.basename(os.path.dirname(pdir))
        try:
            sj = season_json(s)
            c0 = dt.datetime.strptime(sj["c_start"][:16], "%Y-%m-%d %H:%M")
            end = dt.datetime.strptime(sj["end"][:16], "%Y-%m-%d %H:%M")
            # 이벤트 종료(e_end, 챌린지 끝 +4일). 끝난 시즌 예측은 이때까지 「이번 시즌」 탭에 남는다 (2026-10-18 사용자 결정)
            e_end = sj.get("e_end") or (end + dt.timedelta(days=4)).strftime("%Y-%m-%d %H:%M")
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
        if os.path.exists(sc):
            try:
                for r in jload(sc).get("rows", []):
                    p_ = f"{r['pct']:g}"
                    if p_ in PCTS and r.get("truth") is not None:
                        truth.setdefault(r["server"], {})[p_] = r2(r["truth"])
            except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as e:
                warn(f"{s}/predict/score.json을 읽지 못해 실제 컷 비교를 뺐다 ({type(e).__name__}: {e})")
                truth = {}
        out.append({"season": s, "boss": sj.get("boss") or (reg.get(s) or {}).get("boss"),
                    "c_start": iso(sj["c_start"]), "end": iso(sj["end"]), "e_end": iso(e_end), "H": round((end - c0).total_seconds() / 3600),
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


def curve_source(lat):
    """계산기 곡선의 출처 (2026-10-18 사용자 결정 — 컷 예측과 같은 곡선으로). 가장 최근 시즌의 마지막 예측이 고정 모델이면
    그 모델의 main prior와 그 예측이 이번 시즌 관측으로 고친 침수 세기(level_update — predict가 모든 서버에 같은 배수로 곱함)를 쓴다.
    아니면 그 시즌(없으면 가장 최근) 고정 모델 파일의 main prior만 (세기는 모름 → prior의 τ_P 폭)."""
    s = lat["season"] if lat else None
    last = None
    if s:
        ps = []
        for p in glob.glob(os.path.join(D, s, "predict", "T*", "predict_T*.json")):
            try:
                o = jload(p)
                ps.append((float(o["T"]), o))
            except (KeyError, ValueError, TypeError, OSError) as e:
                warn(f"{os.path.relpath(p, ROOT)}를 계산기 곡선에 쓰지 못했다 ({type(e).__name__}: {e})")
        if ps:
            last = max(ps, key=lambda t: t[0])[1]
    lvl, meta, mpath = None, None, None
    if last and last.get("model"):
        mpath = os.path.join(ROOT, last["model"]["file"])
        lu = last.get("level_update") or {}
        r = lu.get("result")
        if r:
            lvl = {"mu": r["mu"], "sd": r["sd"], "persons": r.get("persons")}
        meta = {"season": s, "T": round(float(last["T"]), 2), "T_kst": last.get("T_kst"), "H": last.get("H")}
    if not mpath or not os.path.exists(mpath):
        own = os.path.join(D, "analysis", f"model_{s}.json") if s else None
        ms = sorted(glob.glob(os.path.join(D, "analysis", "model_S*.json")), key=lambda p: split.snum(re.search(r"model_(S\d+)", p).group(1)))
        ms = [p for p in ms if re.search(r"model_S\d+\.json$", p)]
        mpath = own if own and os.path.exists(own) else (ms[-1] if ms else None)
    if not mpath:
        return None
    model = jload(mpath)
    prior = jload(os.path.join(ROOT, model["main"]["prior"]))
    return {"model": model, "prior": prior, "level": lvl, "pred": meta}


def conv_row(PR, tr, sv, h, H, m_s, s_lvl):
    """[P, sd] — viz의 % 변환 행과 같은 셈 (σ(P)·꼴 외삽·시즌 수준 폭). m_s = 시즌 침수 세기 배수, s_lvl = 그 ln 폭."""
    P = PR.p_full(tr, sv, h, 1.0, H, m_s)
    sig = math.sqrt(tr["sigma"]["s0"] ** 2 + (tr["sigma"]["rho"] * P) ** 2)
    sf = PR.sigma_form(tr, h, H)
    return P, math.sqrt(sig ** 2 + (P * s_lvl) ** 2 + sf ** 2)


def conv(vdata, src, lat_H=None):
    """계산기용 % 변환, 0.5시간 간격. 행 = [시작 후 시간, log10(최종 % ÷ 지금 %), 80% 구간용 sd].
    곡선은 컷 예측과 같은 고정 모델(최근 5시즌 main prior × 이번 시즌 침수 세기, 2026-10-18 사용자 결정 — 전에는 viz의 23시즌 평균).
    이번 시즌 것만 싣는다. 지난 시즌 관측 점과 시즌마다의 곡선(23시즌 곡선 × 그 시즌 세기)은 「지난 시즌」 탭의 침수 비교(sink)로 옮겼다 (2026-10-18 사용자 결정)."""
    import predict as PR
    cv = vdata["conv"]
    seas = vdata["season_list"]   # 관측 점·시즌 세기는 viz 정답 표의 끝난 시즌 모두
    near = CONV_RULE["near"]
    if src:
        tr = src["prior"]["transform"]
        tau = src["prior"]["season_diff"]["tau_p"]["value"]
        H = int((src["pred"] or {}).get("H") or lat_H or cv["H"])
        lv = src["level"]
        m_live, s_live = (math.exp(lv["mu"]), lv["sd"]) if lv else (1.0, tau)
        train = list(src["model"].get("borrow") or [])
        base = {"name": src["model"].get("name"), "train": train, "season": src["model"].get("season")}
        rowsF = {sv: [[i / 2, *[round(v, 5) for v in conv_row(PR, tr, sv, i / 2, H, m_live, s_live)]] for i in range(H * 2 + 1)] for sv in SERVERS}
        at_base = lambda sv, h: conv_row(PR, tr, sv, h, H, 1.0, tau)  # noqa: E731
        m_src = tr.get("m_src", {})
    else:   # 고정 모델 파일이 없을 때만 (옛 웹과 같게 viz 곡선)
        H, train, base, lv, tau = cv["H"], list(seas), None, None, cv["tau_p"]
        rowsF = {sv: [[r[0], r[1], r[2]] for i, r in enumerate(cv["srv"][sv]) if i % 2 == 0] for sv in SERVERS}
        at_base = lambda sv, h: min(cv["srv"][sv], key=lambda r: abs(r[0] - h))[1:3]  # noqa: E731
        m_src = cv["m_src"]
    # 지점마다 곡선 점검 (CONV_RULE): 최종 1·3·10·30% 근처(÷1.3~×1.3)로 끝난 칸 중 그 곡선을 맞춘 학습 시즌의 것을
    # 이번 시즌 세기 없는 곡선과 견준다 (표본 안 점검)
    res = {(sv, p): [] for sv in SERVERS for p in CAND}
    for s in seas:
        if s not in train:
            continue
        for u in cv["units"][s]:
            if u["sv"] not in SERVERS or u["P"] is None:
                continue
            f = u["x"] * 10 ** u["P"]
            hit = [p for p in CAND if float(p) / near <= f <= float(p) * near]
            if not hit:
                continue
            P, sd = at_base(u["sv"], u["h"])
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
    live = None
    if src and src["pred"]:
        live = {**src["pred"], "level": round(math.exp(lv["mu"]), 4) if lv else None, "sd": round(lv["sd"], 4) if lv else None,
                "persons": lv.get("persons") if lv else None, "tau": round(tau, 4)}
    return {"H": H, "seasons": seas, "rule": CONV_RULE, "check": check, "check_seasons": train, "base": base, "live": live,
            "borrow": {sv: (m_src.get(sv) or "").startswith("borrow") for sv in SERVERS}, "srv": rowsF}


def sink(vdata, cvd, reg, src):
    """시즌별 침수 세기 비교 (2026-10-18 사용자 요청). 세기 = 여러 시즌 % 변환에서 그 시즌의 수준 배수(prior_dev season_levels —
    모든 시즌이 같은 모양이고 log 배율이 이만큼 크거나 작다고 본 값, 1 = 23시즌 평균). 이번 시즌은 컷 예측이 고친 세기가 있을 때만,
    같은 잣대(23시즌 곡선)로 옮겨 싣는다: 한섭 12시간~끝 곡선 비 log10 P_F·m ÷ P_dev의 기하 평균.
    계산기에 있던 지난 시즌 겹쳐 보기를 여기로 합쳤다 (2026-10-18 사용자 결정): dev = 23시즌 곡선(0.5시간 간격 [시간, log10 배율]),
    obs = 끝난 시즌 진행 중 관측 칸 모두 [시즌, 시작 후 시간, 그때 %, 최종 %] (숫자만 — 화면이 「지금 X%」 근처만 골라 찍는다)."""
    import predict as PR
    cv = vdata["conv"]
    pd = jload(os.path.join(D, "analysis", "prior_dev.json"))
    se = pd["transform"].get("season_level_se_ln", {})
    seas = vdata["season_list"]
    levels = {s: cv["levels"][s] for s in seas if s in cv["levels"]}
    rows = [{"season": s, "boss": (reg.get(s) or {}).get("boss"), "lev": v, "se": se.get(s)} for s, v in levels.items()]
    obs = {sv: [] for sv in SERVERS}
    for s in seas:
        for u in cv["units"][s]:
            if u["sv"] in obs and u["P"] is not None:
                obs[u["sv"]].append([s, round(u["h"], 1), round(u["x"], 3), round(u["x"] * 10 ** u["P"], 3)])
    live = None
    if src and src["level"] and src["pred"]:
        tr, trd = src["prior"]["transform"], pd["transform"]
        H = int(src["pred"].get("H") or cvd["H"])
        m = math.exp(src["level"]["mu"])
        L = [math.log(PR.p_full(tr, "KR", h, 1.0, H, m) / PR.p_full(trd, "KR", h, 1.0, H)) for h in range(12, H + 1)]
        live = {"season": src["pred"]["season"], "T": src["pred"]["T"], "T_kst": src["pred"].get("T_kst"),
                "lev": round(math.exp(sum(L) / len(L)), 4), "sd": round(src["level"]["sd"], 4), "persons": src["level"].get("persons")}
    ps = (pd.get("season") or "").split("+")
    return {"seasons": rows, "live": live, "prior": f"{ps[0]}~{ps[-1]} ({len(ps)}시즌)" if len(ps) > 1 else pd.get("season"),
            "dev": {sv: [[r[0], r[1]] for i, r in enumerate(cv["srv"][sv]) if i % 2 == 0] for sv in SERVERS},
            "borrow": {sv: (cv["m_src"].get(sv) or "").startswith("borrow") for sv in SERVERS},
            "start": {s: iso(vdata["seasons"][s]["c_start"]) for s in seas}, "obs": obs}


def back(reg):
    """지난 시즌 되돌려 보기 (2026-10-18 사용자 요청 — 「S35를 S34 이하 자료만으로」). backtest_web.py가 만든 analysis/backtest_F.json. 없으면 None."""
    p = os.path.join(D, "analysis", "backtest_F.json")
    if not os.path.exists(p):
        return None
    try:
        b = jload(p)
        seasons = [{"season": s, "boss": (reg.get(s) or {}).get("boss"), "train": x["train"], "H": x["H"], "c_start": iso(x["c_start"]), "srv": x["srv"]}
                   for s, x in sorted(b["seasons"].items(), key=lambda kv: split.snum(kv[0]))]
        return {"model": b.get("model"), "T": b["T"], "seasons": seasons}
    except (KeyError, ValueError, TypeError, OSError) as e:
        warn(f"analysis/backtest_F.json을 읽지 못해 「지난 시즌」 탭의 되돌려 보기를 뺐다 ({type(e).__name__}: {e})")
        return None


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
    lastp = [x["preds"][-1] for x in lv if x["preds"]]
    methods = [lastp[-1]["method"]] if lastp else []   # 가장 최근 예측의 방법 (S42 도중에 바뀜 — 2026-10-18)
    pst = past(vdata["seasons"], reg)
    lat = latest_season(reg)
    src = curve_source(lat)
    cvd = conv(vdata, src)
    data = {"past": pst, "past_pcts": past_pcts(pst), "live": lv, "latest": lat, "conv": cvd,
            "sink": sink(vdata, cvd, reg, src), "back": back(reg),
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
    head = ('<p class="muted"><a href="./">← 이번 시즌 (컷 예측 · 계산기)</a> · <a href="./#past">지난 시즌 (커트라인 · 되돌려 보기 · 침수 비교)</a>'
            ' · <a href="./#method">방법 (작업 흐름 · 로직)</a></p>\n'
            '<p class="note">이 페이지는 예측 방법을 고르기 전의 분석 화면입니다. '
            f'아래 ①은 끝난 시즌 {len(seas)}개의 평균 「% 변환」이고, ②의 정확도는 <b>여러 시즌 prior</b>(맞힐 시즌 하나만 빼고 나머지 전부)로 잰 것입니다. '
            f'첫 화면 「이번 시즌」 탭의 컷 예측과 계산기는 <b>{m}</b>로 냅니다 (지난 시즌 검증으로 고른 방법 — 아래 ①·②와는 설정이 다릅니다). '
            '이 방법으로 지난 시즌을 그 앞 시즌 자료만으로 되돌려 본 결과는 첫 화면의 「지난 시즌」 탭에 있습니다. '
            '독립 정확도는 새 시즌마다 끝나기 전에 낸 예측을 끝난 뒤 채점해 쌓습니다.</p>\n')
    sub = "개발 시즌 {}~{} ({}개) · {} KST 기준".format(seas[0], seas[-1], len(seas), data["asof"])
    stamp = "정답 표는 끝난 시즌을 모두 씁니다 — 이 시즌들로 방법을 고르고 다듬었으므로 개발 검증입니다. 독립 검증은 새 시즌마다 끝나기 전에 낸 예측을 끝난 뒤 채점해 쌓습니다."
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
    print(f"{a.out}: 기준 {data['asof']} (지문 {data['content']}) · 지난 시즌 {len(data['past'])}개 · "
          f"진행 중 예측 시즌 {len(data['live'])}개, 시점 {n_live}개 · 분석 {', '.join(vdata['season_list'])}")


if __name__ == "__main__":
    main()
