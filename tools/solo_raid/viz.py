"""솔로레이드 분석 시각화 HTML 한 장을 지금 분석 파일들로 다시 만든다 (모델 0, 요청 0, 2026-10-16).

python3 -I tools/solo_raid/viz.py --out DIR      # DIR/solo-raid-analysis.html

분석 결과가 바뀌면(정답 표·prior·검증을 다시 돌리면) 이 명령만 다시 돌린다. 손으로 고칠 값은 없다.
결과물은 레포에 커밋하지 않는다 — 세션 임시 폴더에 쓰고 채팅으로 보낸다 (AGENTS.md 「결과물」).

입력 (읽기만 한다. split 감시를 켠다 — 시즌은 정답 표에 든 개발 시즌만 쓴다. split 판 2(2026-10-17)에는 잠근 시험 시즌이 없어 끝난 시즌 모두)
- data/solo-raid/analysis/answer_key.json  시즌 목록·서버별 정답 컷
- data/solo-raid/analysis/xval_dev.json     시즌 하나씩 빼기 검증 (loso). 서버별 요약은 xval.summarize를 서버만 바꿔 다시 부른다
- data/solo-raid/analysis/prior_dev.json    여러 시즌 % 변환 P(h) · 시즌 수준 · 잔차 σ(h) · τ_P · 기준선 1(일차 중앙)
- data/solo-raid/analysis/criteria.json     허용 오차 · T_main · 판정에 필요한 시즌 수 문구
- data/solo-raid/registry.json              시즌 몫·보스·수집 상태
- data/solo-raid/S{n}/state/                fit_prior.load_units(배율 근거 점) · core.jsonl(서버 라벨 줄 수) · season.json

화면 (서버 한섭·일섭·글섭 전환)
① 지금 몇 %면 최종 몇 %? — 계산기, 최종 1·3·10%가 되는 지금 %, 근거(관측 배율), 시점별 표
② 예측 정확도 — 시점별 예측 vs 실제, 시즌별 오차, 시점별 요약 표
요약 · 읽는 법 — 아래 notes()가 숫자에서 문장을 만든다
"""
import argparse
import collections
import datetime as dt
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
D = os.path.join(ROOT, "data", "solo-raid")
sys.path.insert(0, HERE)
import split  # noqa: E402

SERVERS = ["KR", "JP", "GL"]
SRV = {"KR": "한섭", "JP": "일섭", "GL": "글섭"}
PCTS = [1.0, 3.0, 10.0]
Z80 = 1.2816
MIN_SEASONS = 8   # criteria.small_n_note: 개발 시즌 8개 이상에서 판정
# xval 판정 문구(검증 결과 파일에 저장된 값)를 화면의 다른 요약 줄처럼 명사형으로
COVERAGE = {"구간이 좁다(과신)": "구간이 좁음(과신)", "구간이 넓다": "구간이 넓음"}


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def eun(s):
    """시즌 이름 뒤 은/는 (S39 → 는, S41 → 은). 끝자리 읽기의 받침으로 고른다."""
    return s + ("는" if str(s)[-1] in "2459" else "은")


def pct_txt(v, d=1):
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v) * 100:.{d}f}%"


def build(args):
    import fit_prior as FP
    import predict as PR
    import xval as XV

    ak = jload(f"{D}/analysis/answer_key.json")
    xv = jload(f"{D}/analysis/xval_dev.json")
    prior = jload(f"{D}/analysis/prior_dev.json")
    crit = jload(f"{D}/analysis/criteria.json")
    reg = {r["season"]: r for r in jload(f"{D}/registry.json")["seasons"]}
    seasons = sorted(ak["seasons"], key=split.snum)
    for s in seasons:
        if split.is_test(s):
            raise SystemExit(f"정답 표에 시험 시즌 {s}가 있다 — 시각화는 개발 시즌만 쓴다")
    tr = prior["transform"]

    # 시즌 정보·정답 컷
    S = {}
    for s in seasons:
        sj = jload(f"{D}/{s}/state/season.json")
        c0 = dt.datetime.strptime(sj["c_start"], "%Y-%m-%d %H:%M:%S")
        end = dt.datetime.strptime(sj["end"], "%Y-%m-%d %H:%M:%S")
        key = {}
        for r in ak["rows"]:
            if r["season"] == s and r["truth"] is not None:
                key.setdefault(r["server"], {})[f'{r["pct"]:g}'] = {"v": r["truth"], "basis": r["basis"]}
        n_core = collections.Counter()
        n_all = 0
        for r in jl(f"{D}/{s}/state/core.jsonl"):
            if r.get("use") and r.get("score") is not None and r.get("pct") is not None:
                n_all += 1
                n_core[r.get("server")] += 1
        S[s] = {"n": split.snum(s), "boss": reg[s]["boss"], "c_start": sj["c_start"],
                "H": round((end - c0).total_seconds() / 3600), "key": key, "n_core": dict(n_core), "n_all": n_all}

    # 배율 근거 점: 같은 사람의 진행 중 %와 최종 %를 둘 다 아는 사람·일차 칸
    units = {}
    for s in seasons:
        _, H, _, U = FP.load_units(f"{D}/{s}/state", ["A", "A2"], "all")
        units[s] = [{"sv": u["server"], "h": round(u["h"], 2), "P": round(u["P"], 4), "x": round(u["x"], 3), "g": u["grade"]}
                    for u in U if u.get("P") is not None and u.get("cens") is None]

    # % 변환 곡선 (시즌 길이는 가장 흔한 값)
    H = collections.Counter(v["H"] for v in S.values()).most_common(1)[0][0]
    tau = prior["season_diff"]["tau_p"]["value"]
    conv = {"H": H, "tau_p": tau, "levels": tr["season_levels"], "m": tr["m"], "m_src": tr.get("m_src", {}),
            "day_median": tr["day_median"]["KR"], "day_median_m": tr.get("day_median_m", {}), "srv": {}, "units": units}
    for sv in SERVERS:
        rows = []
        for i in range(H * 4 + 1):
            h = i / 4
            P = PR.p_full(tr, sv, h, 1.0, H)
            sig = math.sqrt(tr["sigma"]["s0"] ** 2 + (tr["sigma"]["rho"] * P) ** 2)
            sf = PR.sigma_form(tr, h, H)
            rows.append([h, round(P, 5), round(math.sqrt(sig ** 2 + (P * tau) ** 2 + sf ** 2), 5), round(math.sqrt(sig ** 2 + sf ** 2), 5)])
        conv["srv"][sv] = rows

    # 검증: 서버별 요약을 같은 도구로
    summ = {sv: XV.summarize(xv["rows"], dict(crit, server_main=sv))["loso"] for sv in SERVERS}
    if summ["KR"] != xv["summary"]["loso"]:
        raise SystemExit("한섭 요약을 다시 계산한 값이 xval_dev.json과 다르다 — xval.py가 바뀌었으면 xval을 다시 돌린다")
    keep = ("season", "T", "server", "pct", "pred", "i80", "i95", "grade", "truth", "err", "err_b0", "err_b1", "in80", "ok", "tol", "w80")
    xrows = [{k: r.get(k) for k in keep} for r in xv["rows"] if r["scheme"] == "loso" and r["pct"] in PCTS]

    # 아직 분석에 안 든 개발 시즌 (state가 있는 것만, 시험 시즌은 이름만도 세지 않는다)
    pending = [s for s, r in sorted(reg.items(), key=lambda kv: split.snum(kv[0]))
               if r["role"] == "dev" and s not in seasons and os.path.isdir(f"{D}/{s}")]
    pending = [(s, reg[s]["collect"]) for s in pending]

    data = {"seasons": S, "season_list": seasons, "colors": colors(seasons), "xval": xrows, "summary": summ, "conv": conv,
            "crit": {"T_main": crit["T_main"], "tol": crit["tol"]},
            "stamp": {"xval_head": xv["run"]["tool_head"][:7], "answer_key_final": ak.get("final"), "prior": prior.get("season")}}
    data["notes"] = {sv: notes(sv, data, crit, pending) for sv in SERVERS}
    data["srv_note"] = {sv: srv_note(sv, data) for sv in SERVERS}
    return data


def colors(seasons):
    """색은 시즌 번호에 붙는다 (n mod 8 → 팔레트 칸). 최근 8개 시즌만 색, 그보다 오래된 시즌은 회색."""
    recent = set(seasons[-8:])
    return {s: (f"c{split.snum(s) % 8}" if s in recent else "other") for s in seasons}


def scored_pcts(sv, data, season=None):
    return [p for p in PCTS if any(r["server"] == sv and r["pct"] == p and r["truth"] is not None
                                    and (season is None or r["season"] == season) for r in data["xval"])]


def srv_note(sv, data):
    sp = scored_pcts(sv, data)
    ss = [s for s in data["season_list"] if scored_pcts(sv, data, s)]
    no = [f"{p:g}%" for p in PCTS if p not in sp]
    bits = [f"채점 {'·'.join(f'{p:g}%' for p in sp) or '없음'} · 시즌 {len(ss)}개"]
    if no:
        bits.append(f"{'·'.join(no)} 정답 없음")
    if data["conv"]["m_src"].get(sv, "").startswith("borrow"):
        bits.append("① 배율은 한섭 것을 빌림")
    return " · ".join(bits)


def notes(sv, data, crit, pending):
    """요약(맨 아래 불릿)과 읽는 법 문장을 숫자에서 만든다."""
    J = data["summary"][sv]["judge"]
    by = data["summary"][sv]["by"]
    seasons = data["season_list"]
    name = SRV[sv]
    rows = [r for r in data["xval"] if r["server"] == sv and r["err"] is not None]
    scored = [s for s in seasons if any(r["season"] == s for r in rows)]
    n_lab = sum(data["seasons"][s]["n_core"].get(sv, 0) for s in seasons)
    n_all = sum(data["seasons"][s]["n_all"] for s in seasons)
    n_unk = sum(data["seasons"][s]["n_core"].get(None, 0) for s in seasons)
    n_units = sum(1 for s in seasons for u in data["conv"]["units"][s] if u["sv"] == sv)
    main_pts = [p for p in (1.0, 3.0) if p in scored_pcts(sv, data)]
    summary = []
    summary.append(f"채점할 수 있는 시즌 <b>{len(scored)}개</b> ({' · '.join(scored) or '없음'})"
                   + (f", 나머지 {len(seasons) - len(scored)}개는 {name} 정답 근거 부족" if len(scored) < len(seasons) else ""))
    summary.append(f"{name} 라벨 관측 <b>{n_lab:,}줄</b> (점수 있는 줄, 서버 미상 제외) · ① 배율 근거 점 <b>{n_units}개</b>")
    if J.get("mean_med_abs_err_main") is not None:
        e, b = J["mean_med_abs_err_main"], J.get("mean_med_abs_err_b1")
        summary.append(f"예측 오차(18~96시간, {'·'.join(f'{p:g}%' for p in main_pts)} 평균) <b>{e * 100:.1f}%</b>"
                       + ("" if b is None else f" vs 기준선 1 {b * 100:.1f}% — <b>{'기준선보다 나음' if e < b else '아직 기준선보다 낫지 않음'}</b>"))
    if J.get("cov80_main") is not None:
        summary.append(f"80% 구간 포함률 <b>{J['cov80_main'] * 100:.0f}%</b> (목표 70~90%) — {COVERAGE.get(J['coverage'], J['coverage'])}"
                       + (" · 시즌 1개라 판단 불가" if len(J["seasons"]) < 2 else ""))

    N = []
    # 최종 3% 컷 차이
    c3 = [(s, data["seasons"][s]["key"].get(sv, {}).get("3")) for s in seasons]
    c3 = sorted([(s, v["v"]) for s, v in c3 if v], key=lambda t: -t[1])
    if len(c3) >= 2:
        lo, hi = c3[-1][1], c3[0][1]
        N.append(f"<b>최종 컷은 시즌(보스)마다 다릅니다.</b> {name} 3% 컷이 " + " · ".join(f"{s} {v:.0f}억" for s, v in c3)
                 + f" (최대·최소 {hi / lo:.2f}배). 지난 시즌 컷만으로는 맞히기 어렵고 진행 중 관측이 필요합니다.")
    elif len(c3) == 1:
        N.append(f"{name} 3% 정답 컷은 {c3[0][0]} {c3[0][1]:.0f}억 하나뿐입니다.")
    # 쓸 만함 시작 시점 · 늘 한쪽으로 치우친 시즌
    U = J.get("usable", {})
    Tm = crit["T_main"]
    for p in PCTS:
        tol = crit["tol"][f"{p:g}"]
        if p not in scored_pcts(sv, data):
            N.append(f"{p:g}%는 어느 시즌에도 {name} 정답 근거가 없어 채점하지 않았습니다.")
            continue
        flags = [U.get(f"T{T:g}/{p:g}") for T in Tm]
        start = next((Tm[i] for i in range(len(Tm)) if all(flags[i:])), None)
        ns = by.get(f"T{Tm[-1]:g}/{p:g}", {}).get("n_seasons", 0)
        if start is not None:
            s_txt = f"<b>{p:g}% 예측은 {start}시간부터 「쓸 만함」</b>(오차 ±{tol * 100:.0f}% 안인 시즌이 2/3 이상)"
        elif any(flags):
            s_txt = f"<b>{p:g}% 예측은 몇 시점만 「쓸 만함」</b>(" + "·".join(f"{T}" for T, f in zip(Tm, flags) if f) + "시간)"
        else:
            s_txt = f"<b>{p:g}% 예측은 어느 시점에서도 「쓸 만함」 기준(±{tol * 100:.0f}%)을 못 넘습니다</b>"
        s_txt += f" — 채점 시즌 {ns}개입니다."
        bias = []
        for s in (seasons if ns >= 2 else []):
            e = [r["err"] for r in rows if r["season"] == s and r["pct"] == p and r["T"] in Tm]
            if len(e) >= len(Tm) - 1 and (all(v > 0 for v in e) or all(v < 0 for v in e)) and statistics.median(abs(v) for v in e) > tol:
                bias.append(f"{eun(s)} 늘 {'높게' if e[0] > 0 else '낮게'}({pct_txt(min(e, key=abs))}~{pct_txt(max(e, key=abs))})")
        if bias:
            s_txt += " " + ", ".join(bias) + " 예측했습니다."
        early = [r for r in rows if r["pct"] == p and r["T"] in Tm and r["T"] <= 36]
        cd = sum(1 for r in early if r["grade"] in ("C", "D"))
        if early and cd / len(early) >= 0.5:
            s_txt += f" 다만 18~36시간 예측의 {cd}/{len(early)}개가 신뢰도 「매우 낮음」·「참고용」(관측이 적어 prior에 기댐)입니다."
        N.append(s_txt)
    cov = J.get("cov80_main")
    if cov is not None and len(J["seasons"]) >= 2:
        if cov < 0.70:
            N.append(f"<b>80% 구간이 너무 좁습니다</b>(실제 포함 {cov * 100:.0f}%). 화면의 구간보다 넓게 봐야 합니다.")
        elif cov > 0.90:
            N.append(f"80% 구간이 넓은 편입니다(실제 포함 {cov * 100:.0f}%).")
    # 배율 근거
    m = data["conv"]["m"].get(sv, 1.0)
    src = data["conv"]["m_src"].get(sv, "")
    if src.startswith("borrow"):
        N.append(f"{name}은 배율 근거 점이 {'없어' if n_units == 0 else '적어'} ① 배율을 한섭 것 그대로 빌립니다(가정).")
    elif sv != "KR":
        N.append(f"① 배율은 한섭 배율에 {name} 배수 {m:.2f}를 곱한 값입니다(근거 점 {n_units}개라 한섭보다 거칩니다).")
    lv = data["conv"]["levels"]
    if len(lv) >= 2:
        lo, hi = min(lv.items(), key=lambda t: t[1]), max(lv.items(), key=lambda t: t[1])
        N.append(f"시즌마다 밀리는 정도가 다릅니다: 배율 수준 {lo[0]} {lo[1]:.2f}배 ~ {hi[0]} {hi[1]:.2f}배. 새 시즌은 이 수준을 모르니 ①은 평균에 넓은 구간을 씁니다.")
    late = [s for s in seasons if not data["seasons"][s]["c_start"].endswith("05:00:00")]
    for s in late:
        N.append(f"{eun(s)} 챌린지가 {data['seasons'][s]['c_start'][11:16]}에 시작해 「시작 후 시간」이 다른 시즌과 다른 시각입니다(① 근거 그래프).")
    # 공통
    N.append(f"모든 수치는 개발 시즌 {len(seasons)}개({seasons[0]}~{seasons[-1]})의 "
             + ("<b>시험 운전</b>입니다 — 판정은 개발 시즌 8개 이상에서 합니다." if len(seasons) < MIN_SEASONS else "검증 결과입니다.")
             + " 이 시즌들로 방법을 고르고 다듬었으므로 개발 검증이고, 독립 검증은 새 시즌마다 끝나기 전에 낸 예측을 끝난 뒤 채점해 쌓습니다.")   # 2026-10-17 공개 결정 A·split 판 2 — 잠근 시험 시즌 없음
    if pending:
        N.append("아직 분석에 들지 않은 개발 시즌: " + " · ".join(f"{s}({c})" for s, c in pending) + ".")
    if n_all:
        N.append(f"서버 라벨이 없는 관측({n_unk / n_all * 100:.0f}%)은 ①의 점에 넣지 않았습니다. 예측(②)은 도구 기본값대로 서버 미상 줄을 "
                 "사이트별 서버 비율만큼 한섭 예측에만 넣고, 일섭·글섭은 라벨 붙은 줄만 씁니다.")
    return {"summary": summary, "notes": N}


def main():
    ap = argparse.ArgumentParser(description="솔로레이드 분석 시각화 HTML (분석 파일에서 다시 만든다)")
    ap.add_argument("--out", required=True, help="HTML을 쓸 폴더 (세션 임시 폴더. 레포에 커밋하지 않는다)")
    ap.add_argument("--name", default="solo-raid-analysis.html")
    ap.add_argument("--json", help="데이터를 따로 JSON으로도 쓴다 (점검용)")
    args = ap.parse_args()
    split.install_guard(final=False)
    data = build(args)
    with open(os.path.join(HERE, "viz_template.html"), encoding="utf-8") as f:
        tpl = f.read()
    kst = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
    html = tpl.replace("/*DATA*/", json.dumps(data, ensure_ascii=False)).replace("__ASOF__", kst)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, args.name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    print(f"{out} ({len(html) / 1024:.0f} KB) · 시즌 {', '.join(data['season_list'])} · xval {data['stamp']['xval_head']}")
    for sv in SERVERS:
        print(f"  {sv}: " + " / ".join(data["notes"][sv]["summary"]).replace("<b>", "").replace("</b>", ""))


if __name__ == "__main__":
    main()
