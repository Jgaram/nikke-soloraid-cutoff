"""WP1 1d·1e (2026-10-28) — 떨어진 오류 비율(오염 혼합의 사전)과 서버 착오율(서버 혼합의 사전).

  python3 -I tools/solo_raid/lab7/err_1de.py --tbl W/obs7/full --mirror W/wp1/mirror --out W/wp1/1de.json [--no-mirror]

정답(answer_key·truth)을 읽지 않는다. 읽는 것: 관측 표(끝난 시즌 전체) · 시즌 state의 obs·core·claims·server·server_claims
· S{n}/discard.json(버릴 것·고칠 것) · S{n}/review.json(점검 목록과 판단) · 서버 미상 줄 확률은 server_mix.py(운영 도구)를 repo 밖 임시 위치에 --out으로 다시 돌린 결과 (repo에는 S41 파일뿐이고 그것도 오래됨).
점검 기록(discard·review)의 글 번호·사진 이름·원문 이유는 줄을 잇는 데만 쓰고 출력·코드에 옮기지 않는다 — 집계만 낸다.

「점검 전 서버」는 서버 고침(action server)이 없었을 때의 서버 분류다. 운영 서버 분류기를 임시 사본(--mirror)에서
discard.json의 서버 고침 항목만 빼고 다시 돌려 얻는다 (같은 도구를 고침 항목이 있는 채로 돌리면 repo의 server.jsonl과
바이트까지 같음을 시즌마다 확인한다). 사본은 repo 밖에 있고 repo는 읽기만 한다.

단위: 줄(행) 비율은 같은 글의 줄이 서로 독립이 아니므로 윌슨 구간은 「줄 수 기준 참고」, 부트스트랩은 글((시즌, 글) 묶음)을
재표본 단위로 한다. 시즌별 값은 윌슨 구간. seed 고정.
"""
import argparse
import ast
import collections
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_table as O  # noqa: E402
import common as C  # noqa: E402

ROOT = C.ROOT
DATA = os.path.join(ROOT, "data", "solo-raid")
SEASONS = list(range(19, 43))
SEED = 20261028
CLASSIFIER = os.path.join(ROOT, "experiments", "2026-10-15-solo-raid-low-cost", "tools", "server_classify.py")
S41_AUX = {  # 서버 분류기에 S41만 더 주는 입력 (docs/solo-raid-cutoff.md 「서버 분류」)
    "posts": os.path.join(ROOT, "experiments", "2026-10-15-s41-opus-blind", "data", "posts_min.jsonl"),
    "snap": os.path.join(ROOT, "experiments", "2026-10-15-s41-collection-baseline", "data", "snapshot", "posts"),
}
SRV = ("KR", "JP", "GL", "NA", "SEA")


# ---------------------------------------------------------------- 공통 통계
def r4(x):
    return None if x is None else round(float(x), 4)


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [r4(max(0.0, c - h)), r4(min(1.0, c + h))]


def rate(k, n):
    return {"k": int(k), "n": int(n), "p": r4(k / n) if n else None, "wilson": wilson(k, n)}


def cluster_boot(pairs, B=2000, seed=SEED):
    """pairs = [(군집열쇠, 분자, 분모)]. 군집 재표본 비율 95% 구간."""
    g = collections.defaultdict(lambda: [0, 0])
    for key, k, n in pairs:
        g[key][0] += k
        g[key][1] += n
    a = np.array(list(g.values()), dtype=float)
    if len(a) < 2 or a[:, 1].sum() == 0:
        return {"clusters": len(a), "ci": [None, None]}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    num = a[idx, 0].sum(1)
    den = a[idx, 1].sum(1)
    ok = den > 0
    v = num[ok] / den[ok]
    return {"clusters": int(len(a)), "ci": [r4(np.percentile(v, 2.5)), r4(np.percentile(v, 97.5))]}


def beta_fit_from_rates(ks, ns):
    """시즌별 (k, n)에서 베타-이항 최대가능도 (a, b). 시즌마다 달라지는 정도를 담는다."""
    from scipy.optimize import minimize
    from scipy.special import betaln, gammaln
    ks = np.asarray(ks, float)
    ns = np.asarray(ns, float)
    ok = ns > 0
    ks, ns = ks[ok], ns[ok]
    if len(ks) < 3 or ks.sum() == 0:
        return None

    def nll(th):
        a, b = np.exp(th)
        ll = (gammaln(ns + 1) - gammaln(ks + 1) - gammaln(ns - ks + 1) + betaln(ks + a, ns - ks + b) - betaln(a, b))
        return -ll.sum()
    p0 = ks.sum() / ns.sum()
    best = None
    for conc in (3.0, 30.0, 300.0):
        r = minimize(nll, np.log([max(p0, 1e-4) * conc, (1 - p0) * conc]), method="Nelder-Mead")
        if best is None or r.fun < best.fun:
            best = r
    a, b = np.exp(best.x)
    if a + b > 1e5:
        return {"degenerate": "시즌 퍼짐이 이항 수준 이하(a+b 무한대) — 시즌 이질성 없음", "seasons": int(len(ks))}
    # 이항(시즌 같음) 대비 가능도
    p = ks.sum() / ns.sum()
    from scipy.stats import binom
    ll0 = binom.logpmf(ks, ns, p).sum()
    return {"a": r4(a), "b": r4(b), "mean": r4(a / (a + b)), "conc_a_plus_b": r4(a + b), "ll_betabinom": r4(-best.fun),
            "ll_binom": r4(ll0), "seasons": int(len(ks))}


def het_chi2(ks, ns):
    from scipy.stats import chi2
    ks = np.asarray(ks, float)
    ns = np.asarray(ns, float)
    ok = ns > 0
    ks, ns = ks[ok], ns[ok]
    p = ks.sum() / ns.sum()
    if p in (0, 1) or len(ks) < 2:
        return None
    e1, e0 = ns * p, ns * (1 - p)
    stat = (((ks - e1) ** 2) / e1).sum() + ((((ns - ks) - e0) ** 2) / e0).sum()
    df = len(ks) - 1
    return {"chi2": r4(stat), "df": int(df), "p": r4(chi2.sf(stat, df)), "pooled": r4(p)}


# ---------------------------------------------------------------- 자료 읽기
def jl(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run_classifier(mirror, s, variant):
    """임시 사본에서 서버 분류기를 다시 돌려 server.jsonl·server_claims.jsonl 문자열을 돌려준다.
    variant "intact": discard.json 그대로 (repo 파일과 같은지 확인용) · "nofix": 서버 고침 항목(action server)을 뺀다."""
    base = os.path.join(mirror, "data", "solo-raid", f"S{s}")
    work = os.path.join(mirror, f"work_{variant}", "data", "solo-raid", f"S{s}")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(os.path.dirname(work), exist_ok=True)
    shutil.copytree(os.path.join(DATA, f"S{s}", "state"), os.path.join(work, "state"))
    d = os.path.join(DATA, f"S{s}", "discard.json")
    if os.path.exists(d):
        j = json.load(open(d, encoding="utf-8"))
        if variant == "nofix":
            j["items"] = [x for x in j["items"] if x.get("action", "discard") != "server"]
        json.dump(j, open(os.path.join(work, "discard.json"), "w", encoding="utf-8"), ensure_ascii=False)
    root = os.path.join(mirror, f"work_{variant}")
    for rel in ("tools/solo_raid", "experiments/2026-10-15-solo-raid-low-cost/tools"):
        os.makedirs(os.path.join(root, rel), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "tools", "solo_raid", "discard.py"), os.path.join(root, "tools", "solo_raid", "discard.py"))
    shutil.copy(CLASSIFIER, os.path.join(root, "experiments", "2026-10-15-solo-raid-low-cost", "tools", "server_classify.py"))
    cmd = [sys.executable, "-I", os.path.join(root, "experiments", "2026-10-15-solo-raid-low-cost", "tools", "server_classify.py"),
           os.path.join(work, "state")]
    if s == 41:
        cmd += ["--posts", S41_AUX["posts"], "--snap", S41_AUX["snap"]]
    subprocess.run(cmd, check=True, capture_output=True)
    out = {}
    for f in ("server", "server_claims"):
        out[f] = jl(os.path.join(work, "state", f + ".jsonl"))
    shutil.rmtree(root)
    return out


def load_season(s, tbl_dir, mirror=None):
    st = os.path.join(DATA, f"S{s}", "state")
    head, rows = O.read_cell(os.path.join(tbl_dir, f"S{s}.jsonl.gz"))
    D = {"s": s, "head": head, "rows": rows,
         "obs": jl(os.path.join(st, "obs.jsonl")), "core": {r["key"]: r for r in jl(os.path.join(st, "core.jsonl"))},
         "claims": jl(os.path.join(st, "claims.jsonl")),
         "server": jl(os.path.join(st, "server.jsonl")), "server_claims": jl(os.path.join(st, "server_claims.jsonl")),
         "discard": [], "review": []}
    p = os.path.join(DATA, f"S{s}", "discard.json")
    if os.path.exists(p):
        D["discard"] = json.load(open(p, encoding="utf-8"))["items"]
    D["review"] = json.load(open(os.path.join(DATA, f"S{s}", "review.json"), encoding="utf-8"))["items"]
    D["pre"] = None
    if mirror:
        cp = os.path.join(mirror, "pre_cache", f"S{s}.json")
        if os.path.exists(cp):
            c = json.load(open(cp, encoding="utf-8"))
            D["repro_server"], D["pre"] = c["repro"], c["pre"]
        else:
            chk = run_classifier(mirror, s, "intact")
            D["repro_server"] = chk["server"] == D["server"] and chk["server_claims"] == D["server_claims"]
            D["pre"] = run_classifier(mirror, s, "nofix")
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            json.dump({"repro": D["repro_server"], "pre": D["pre"]}, open(cp, "w", encoding="utf-8"), ensure_ascii=False)
    return D


# ---------------------------------------------------------------- 점검 기록 ↔ 관측 표
TIME_RE = re.compile(r"시각 오류|촬영 시각|기기 시[계간]|시간대|기갱 전|종료 전|종료 \d+시간 전|이른 순위표|첫날|올리기 전|갱신 시간|기록 세운 때|캡처한 화면|시각이")
PAST_RE = re.compile(r"지난 (시즌|수냉|작열|속성|보스)|이전 (보스|시즌)|회상|기록 모음|모음|이어 ?붙인|콜라주|합성|편집|다른 시즌|시즌 (경계|번호)|재오픈|초기화 전|S\d\d[ )]|크챔|퀸")
READ_RE = re.compile(r"판독 (오류|틀림|이 줄을)|오독|점수 칸이|점수 앞 아이콘|잘못 (읽|이었)")
SERVER_RE = re.compile(r"서버 (오류|묶음|라벨)|서버만|서버가 다른|한섭 아님|글섭 화면")


def verdict_of(it, rv):
    """review 판단 id들에서 verdict(time·server·read) 하나를 찾는다 (없으면 None)."""
    for rid in it.get("review") or []:
        x = rv.get(rid)
        if not x:
            continue
        v = x.get("verdict")
        if v is None and x.get("s3"):
            s3 = x["s3"]
            if isinstance(s3, str):
                try:
                    s3 = ast.literal_eval(s3)
                except Exception:
                    s3 = {}
            v = (s3 or {}).get("verdict")
        if v in ("time", "server", "read"):
            return v
    return None


def cause_of(it, rv):
    """버림·고침 항목의 원인 종류 (대략): time(시각) · read(판독) · server · wrongscreen(지난 시즌·편집·모음 화면) · other.
    review의 verdict가 있으면 그것, 없으면 사유 글의 말 토큰으로 가른다 (경험칙 — 같은 항목이 둘에 걸리면 앞쪽 우선)."""
    if it.get("action") == "server":
        return "server"
    v = verdict_of(it, rv)
    if v:
        return {"time": "time", "read": "read", "server": "server"}[v]
    t = it.get("reason") or ""
    head = t[:60]
    if READ_RE.search(head):
        return "read"
    if re.search(r"시각 오류|촬영 시각|기기 시", head):
        return "time"
    if PAST_RE.search(t[:120]):
        return "wrongscreen"
    if TIME_RE.search(t[:160]):
        return "time"
    if SERVER_RE.search(head):
        return "server"
    return "other"


TZ_RE = re.compile(r"기기 시[계간]|시간대|UTC|동부|시계는|시계가|\+\s?\d+\s?시간")
STALE_RE = re.compile(r"올리기 전|이전|이른|기갱 전|종료 전|첫날|옛|전 화면|캡처|지난|오래|기록 세운|어제")


def time_sub(it):
    """시각 오류의 하위 종류 (사유 글의 말 토큰, 경험칙): tz = 기기 시계·시간대가 어긋남(A 파일 시각이 틀림) · stale = 올리기 전·옛 화면(촬영이 게시보다 훨씬 이름) · other."""
    t = it.get("reason") or ""
    if TZ_RE.search(t[:200]):
        return "tz"
    if STALE_RE.search(t[:200]):
        return "stale"
    return "other"


def tz_hours(it):
    m = re.search(r"\+\s?(\d{1,2})\s?시간", it.get("reason") or "")
    return int(m.group(1)) if m else None


def build_link(D):
    """discard 항목을 obs 줄(oi)·글 줄(i)에 잇는다. 반환 dict."""
    obs = D["obs"]
    by_ir = {(o["img"], str(o["row"])): oi for oi, o in enumerate(obs)}
    by_img = collections.defaultdict(list)
    by_no = collections.defaultdict(list)
    for oi, o in enumerate(obs):
        by_img[o["img"]].append(oi)
        by_no[str(o["no"])].append(oi)
    claim_by_no = collections.defaultdict(list)
    for i, c in enumerate(D["claims"]):
        claim_by_no[str(c["no"])].append(i)
    rv = {x["id"]: x for x in D["review"]}
    out = []
    for it in D["discard"]:
        sc, ac = it["scope"], it.get("action", "discard")
        if sc == "row":
            oi = [by_ir[(it["img"], str(it["row"]))]] if (it["img"], str(it["row"])) in by_ir else []
            ti = []
        elif sc == "img":
            oi, ti = list(by_img.get(it["img"], [])), []
        else:
            oi, ti = list(by_no.get(str(it["no"]), [])), list(claim_by_no.get(str(it["no"]), []))
        cz = cause_of(it, rv)
        out.append({"scope": sc, "action": ac, "cause": cz, "sub": time_sub(it) if cz == "time" else None,
                    "tz_hours": tz_hours(it) if cz == "time" else None, "oi": oi, "ti": ti,
                    "to": it.get("server"), "no": str(it["no"]), "site": it.get("site"), "by_review": bool(it.get("review")),
                    "has_correct": "correct" in it, "date": it.get("decided")})
    return out


# ---------------------------------------------------------------- 점검 후보(review) → 줄
def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except Exception:
            return []
    return list(v) if isinstance(v, (list, tuple)) else []


def review_cands(D):
    """review.json 항목마다 (kind, decision, 후보 oi 집합). 후보 oi는 그 항목이 가리키는 관측 줄.
    s3_final(시즌 경계에 옮긴 최종 제안)·shaky(집계표 견줌)·cluster_mixed(묶음 단위)는 줄로 풀지 않는다 (oi 비어 있음)."""
    obs = D["obs"]
    by_ir = {(o["img"], str(o["row"])): oi for oi, o in enumerate(obs)}
    by_img = collections.defaultdict(list)
    by_no = collections.defaultdict(list)
    for oi, o in enumerate(obs):
        by_img[o["img"]].append(oi)
        by_no[str(o["no"])].append(oi)
    out = []
    for x in D["review"]:
        k = x["kind"]
        oi = set()
        if k in ("s3_final", "shaky", "cluster_mixed"):
            out.append({"kind": k, "decision": x.get("decision"), "oi": oi, "gone": bool(x.get("gone"))})
            continue
        for ref in _as_list(x.get("imgs")) + _as_list(x.get("affected")):
            m = re.match(r"^(.+?)#(\d+)$", str(ref))
            if m and (m.group(1), m.group(2)) in by_ir:
                oi.add(by_ir[(m.group(1), m.group(2))])
            elif str(ref) in by_img:
                oi.update(by_img[str(ref)])
        img, row = x.get("img"), x.get("row")
        if img is not None and row is not None and (img, str(row)) in by_ir:
            oi.add(by_ir[(img, str(row))])
        elif img is not None and x.get("rows") is not None:
            for r in _as_list(x.get("rows")):
                if (img, str(r)) in by_ir:
                    oi.add(by_ir[(img, str(r))])
        elif img is not None and row is None and img in by_img and not oi:
            oi.update(by_img[img])
        if not oi and x.get("no") is not None and str(x["no"]) in by_no:
            oi.update(by_no[str(x["no"])])
        out.append({"kind": k, "decision": x.get("decision"), "oi": oi, "gone": bool(x.get("gone")), "verdict": x.get("verdict"),
                    "est": x.get("est_shot_time_kst")})
    return out


ERR_DEC = ("row", "server", "discard")
EXCL_PRIMARY = ["tracker", "season_out", "prev_season", "season_unsure", "ctx_other", "ctx_prev", "other_raid", "reward_table",
                "pct_over_100", "pct_no_sign", "mock", "discard"]
BASIS_GROUP = {"top7": "top7", "text+top7": "text+top7", "text": "text", "top7_1": "top7_1", "top7_post": "top7_post(+text)",
               "text+top7_post": "top7_post(+text)", "fix": "fix", "multi": "미상(multi)", "q": "미상(q·cmt)", "cmt": "미상(q·cmt)",
               "conflict": "미상(conflict)", None: "미상"}


def basis_group(b):
    return BASIS_GROUP.get(b, str(b))


def season_records(D, link):
    """사진·글 줄마다 한 레코드. 핵심: pool(점검 전 기본 집합 후보)·disc(버려짐)·fix(서버 고침)·cause."""
    rows = D["rows"]
    core = D["core"]
    direct = {}
    fixed_direct = {}
    # (위: 직접 이은 항목)
    for l in link:
        for oi in l["oi"]:
            (direct if l["action"] == "discard" else fixed_direct).setdefault(oi, l)
    text_disc = {}
    text_fix = {}
    for l in link:
        for ti in l["ti"]:
            (text_disc if l["action"] == "discard" else text_fix).setdefault(ti, l)
    out = []
    for r in rows:
        isT = r["kind"] == "text"
        z = (core.get(r["key"]) or {}).get("surf_z") if not isT else None
        disc = "discard" in r["exclude"] if not isT else (r["i"] in text_disc)
        l = (direct.get(r["oi"]) if not isT else text_disc.get(r["i"]))
        cause = None
        if disc:
            cause = l["cause"] if l else "copy"
        fx = (r["server_basis"] == "fix") if not isT else (r["i"] in text_fix)
        pool = (r["kind"] != "text" and r["pct"] is not None and (not r["exclude"] or r["exclude"] == ["discard"]))
        out.append({"s": D["s"], "key": r["key"], "kind": r["kind"], "screen": r["screen"], "site": r["site"],
                    "basis": r["server_basis"], "server": r["server"], "disc": disc, "fix": fx, "cause": cause,
                    "copy": bool(disc and l is None and not isT), "pool": pool, "z": z, "flag": r["flag"], "excl": r["exclude"],
                    "post": (D["obs"][r["oi"]]["no"] if not isT and r["oi"] is not None else
                             (D["claims"][r["i"]]["no"] if r["i"] is not None else None)),
                    "pct_none": r["pct"] is None, "oi": r["oi"], "t_grade": r["t_grade"], "h": r["h"], "hp": r["hp"],
                    "use": r["use"], "mono_drop": r["mono_drop"], "lo": r["lo_reason"],
                    "pass2": bool(l and (l["date"] or "") > "2026-10-17"), "sub": (l or {}).get("sub")})
    return out


# ---------------------------------------------------------------- 1d
def group_rates(recs, keyfn, num, den, boot=True):
    g = collections.defaultdict(list)
    for r in recs:
        if den(r):
            g[keyfn(r)].append(r)
    out = {}
    for k in sorted(g, key=lambda x: str(x)):
        L = g[k]
        n = len(L)
        kk = sum(1 for r in L if num(r))
        d = rate(kk, n)
        if boot and n >= 30:
            d["boot_post"] = cluster_boot([((r["s"], r["post"]), 1 if num(r) else 0, 1) for r in L], B=1000)
        out[str(k)] = d
    return out


def season_table(recs, num, den):
    out = {}
    for s in SEASONS:
        L = [r for r in recs if r["s"] == s and den(r)]
        out[f"S{s}"] = rate(sum(1 for r in L if num(r)), len(L))
    return out


def expanding(recs, num, den):
    """맞힐 시즌 s에서 쓸 수 있는 「s보다 앞 시즌들만」 풀 비율, 그 시즌 실제 비율, 앞 시즌 풀 n."""
    out = {}
    for s in range(SEASONS[0] + 1, SEASONS[-1] + 1):
        pre = [r for r in recs if r["s"] < s and den(r)]
        cur = [r for r in recs if r["s"] == s and den(r)]
        kp, npre = sum(1 for r in pre if num(r)), len(pre)
        kc, nc = sum(1 for r in cur if num(r)), len(cur)
        out[f"S{s}"] = {"prior_p": r4(kp / npre) if npre else None, "prior_n": npre, "prior_k": kp,
                        "season_p": r4(kc / nc) if nc else None, "season_n": nc, "season_k": kc}
    ps = [v["prior_p"] for v in out.values() if v["prior_p"] is not None]
    return {"by_season": out, "prior_p_range": [r4(min(ps)), r4(max(ps))] if ps else None,
            "prior_p_last5_sd": r4(float(np.std(ps[-5:], ddof=1))) if len(ps) >= 6 else None}


def hetero(recs, num, den):
    ks, ns = [], []
    for s in SEASONS:
        L = [r for r in recs if r["s"] == s and den(r)]
        ks.append(sum(1 for r in L if num(r)))
        ns.append(len(L))
    return {"chi2": het_chi2(ks, ns), "betabinom": beta_fit_from_rates(ks, ns)}


def beta_prior_from_pooled(k, n, conc):
    """합친 비율 p와 시즌 퍼짐에서 정한 농도(a+b)로 베타(a,b)."""
    p = k / n if n else 0
    return {"a": r4(p * conc), "b": r4((1 - p) * conc), "mean": r4(p), "conc": r4(conc)}


def zstats(vals):
    v = np.abs(np.array([x for x in vals if x is not None], float))
    if len(v) == 0:
        return {"n": 0}
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return {"n": int(len(v)), "abs_z_q10_25_50_75_90": [r4(x) for x in q],
            "frac_gt_3.5": r4((v > 3.5).mean()), "frac_gt_6": r4((v > 6).mean()), "frac_gt_10": r4((v > 10).mean())}


H_RE = re.compile(r"h\s*(-?\d+(?:\.\d+)?)\s*~\s*(-?\d+(?:\.\d+)?)")


def analyze_1d(Ds, links, recs, rcs):
    P = [r for r in recs if r["pool"]]
    T = [r for r in recs if r["kind"] == "text"]
    ABC = [r for r in recs if r["kind"] != "text"]
    isd = lambda r: r["disc"]
    isf = lambda r: r["fix"]
    isany = lambda r: r["disc"] or r["fix"]
    out = {}

    # --- 연결
    L_all = [(D["s"], l) for D in Ds for l in links[D["s"]]]
    ln = {"items": len(L_all), "unlinked_items": sum(1 for _, l in L_all if not l["oi"] and not l["ti"]),
          "by_scope_action": dict(collections.Counter(f"{l['scope']}/{l['action']}" for _, l in L_all)),
          "by_cause": dict(collections.Counter(f"{l['action']}:{l['cause']}" for _, l in L_all)),
          "items_tied_to_review": sum(1 for _, l in L_all if l["by_review"]),
          "items_with_correct_value": sum(1 for _, l in L_all if l["has_correct"]),
          "seasons_without_discard_json": [f"S{D['s']}" for D in Ds if not D["discard"]]}
    direct_rows = collections.Counter()
    for s, l in L_all:
        if l["action"] == "discard":
            direct_rows["photo_rows_direct"] += len(l["oi"])
            direct_rows["text_rows_direct"] += len(l["ti"])
    tbl_disc = sum(1 for r in ABC if r["disc"])
    ln.update({"table_photo_rows_excluded_discard": tbl_disc, "photo_rows_by_direct_link": direct_rows["photo_rows_direct"],
               "copies_via_same_image": sum(1 for r in ABC if r["copy"]),
               "table_text_rows_discarded_by_post": sum(1 for r in T if r["disc"]),
               "table_photo_rows_server_fix": sum(1 for r in ABC if r["fix"]),
               "table_text_rows_server_fix": sum(1 for r in T if r["fix"])})
    ln["note"] = ("잇는 법: discard 항목의 (img,row)→obs.jsonl 줄(oi)→관측 표 key p{oi}. scope img는 그 사진의 모든 줄, scope post는 그 글 번호의 모든 사진 줄과 "
                  "claims 줄(text). 표의 exclude `discard`는 같은 사진(sha1·dup_of) 사본 줄까지 포함하므로 직접 잇지 못한 줄은 사본이다. "
                  "항목 325 중 어느 줄에도 잇지 못한 것은 unlinked_items.")
    out["link"] = ln

    # --- 분모 정의
    out["scope_note"] = (
        "분모(pool) = 사진 줄(A·B·C) 가운데 %가 있고 자동 제외(exclude)가 없거나 `discard`뿐인 줄 = 점검 전 기본 집합 후보. "
        "분자 = 점검으로 버려진 줄(exclude `discard`: row·post·img 버림과 같은 사진 사본). 서버 고침(action server)은 판독·시각이 맞은 줄이라 별도 비율. "
        "text 분모 = 글로만 적은 %줄 전체, 분자 = 글째 버려진 줄. "
        "점검은 모든 줄을 본 것이 아니다: review.py가 고른 후보(곡면 이탈·정답 점 플래그·여러 시즌 점수 겹침·순위 200 초과 등)와 시즌 세션이 읽다 짚은 글만 원본을 봤다. "
        "후보 밖의 오류는 세지 못했으므로 모든 비율은 하한이다 (아래 coverage·inside_vs_outside). 점검 범위 근거: docs/solo-raid-backfill.md 「점검 세션 지시문」·"
        "review.py 머리말·docs/solo-raid-analysis.md 3.A6.")

    # --- 비율
    R = {"photo_pool_disc": rate(sum(isd(r) for r in P), len(P)),
         "photo_pool_fix": rate(sum(isf(r) for r in P), len(P)),
         "photo_pool_any": rate(sum(isany(r) for r in P), len(P)),
         "text_disc": rate(sum(isd(r) for r in T), len(T)),
         "text_fix": rate(sum(isf(r) for r in T), len(T))}
    R["photo_pool_disc"]["boot_post"] = cluster_boot([((r["s"], r["post"]), int(r["disc"]), 1) for r in P])
    R["photo_pool_any"]["boot_post"] = cluster_boot([((r["s"], r["post"]), int(isany(r)), 1) for r in P])
    # 글 단위: 풀 줄이 있는 글 가운데 버려진 줄이 있는 글
    posts = collections.defaultdict(lambda: [0, 0, 0])
    for r in P:
        v = posts[(r["s"], r["post"])]
        v[0] += 1
        v[1] += int(r["disc"])
        v[2] += int(r["fix"])
    R["posts_with_pool_rows"] = {"n": len(posts), "with_disc": sum(1 for v in posts.values() if v[1]),
                                 "with_fix": sum(1 for v in posts.values() if v[2]),
                                 "p_disc": rate(sum(1 for v in posts.values() if v[1]), len(posts)),
                                 "p_fix": rate(sum(1 for v in posts.values() if v[2]), len(posts)),
                                 "rows_per_disc_post_median": float(np.median([v[1] for v in posts.values() if v[1]])) if any(v[1] for v in posts.values()) else None}
    R["photo_pool_disc_second_pass_only"] = rate(sum(1 for r in P if r["disc"] and r["pass2"]), len(P))
    R["photo_pool_disc_first_pass_only"] = rate(sum(1 for r in P if r["disc"] and not r["pass2"]), len(P))
    R["second_pass_share_of_disc_rows"] = r4(sum(1 for r in P if r["disc"] and r["pass2"]) / max(1, sum(1 for r in P if r["disc"])))
    R["time_error_subtypes_rows"] = dict(collections.Counter(r["sub"] for r in P if r["disc"] and r["cause"] == "time"))
    R["time_error_tz_offset_hours_items"] = sorted(l["tz_hours"] for _, l in L_all if l["tz_hours"] is not None)
    R["second_pass_cause_counts"] = dict(collections.Counter(r["cause"] for r in P if r["disc"] and r["pass2"]))
    R["first_pass_cause_counts"] = dict(collections.Counter(r["cause"] for r in P if r["disc"] and not r["pass2"]))
    out["rates"] = R
    by = {}
    by["kind"] = group_rates(P, lambda r: r["kind"], isd, lambda r: True)
    by["kind_fix"] = group_rates(P, lambda r: r["kind"], isf, lambda r: True)
    by["server_basis"] = group_rates(P, lambda r: basis_group(r["basis"]), isd, lambda r: True)
    by["server_basis_fix"] = group_rates(P, lambda r: basis_group(r["basis"]), isf, lambda r: True)
    by["server_known"] = group_rates(P, lambda r: "서버있음" if r["server"] else "서버미상", isd, lambda r: True)
    by["screen"] = group_rates(P, lambda r: r["screen"], isd, lambda r: True)
    by["site"] = group_rates(P, lambda r: r["site"], isd, lambda r: True)
    by["kind_x_basis"] = group_rates(P, lambda r: f"{r['kind']}|{basis_group(r['basis'])}", isd, lambda r: True, boot=False)
    by["kind_x_screen"] = group_rates(P, lambda r: f"{r['kind']}|{r['screen']}", isd, lambda r: True, boot=False)
    by["season"] = season_table(P, isd, lambda r: True)
    by["season_fix"] = season_table(P, isf, lambda r: True)
    by["season_text"] = season_table(T, isd, lambda r: True)
    byc = {}
    causes = sorted({r["cause"] for r in P if r["cause"]})
    for c in causes:
        byc[c] = rate(sum(1 for r in P if r["cause"] == c), len(P))
    by["cause_share_of_pool"] = byc
    by["cause_x_kind"] = {c: {k: rate(sum(1 for r in P if r["cause"] == c and r["kind"] == k), sum(1 for r in P if r["kind"] == k))
                               for k in "ABC"} for c in causes}
    by["cause_post_counts"] = {c: len({(r["s"], r["post"]) for r in P if r["cause"] == c}) for c in causes}
    out["by"] = by

    # --- 자동 제외(점검 전에 이미 표시된 것)
    U = [r for r in ABC if not r["pct_none"]]
    ex = {}
    for k in "ABC":
        Uk = [r for r in U if r["kind"] == k]
        ex[k] = {"n_rows_with_pct": len(Uk),
                 "excluded_any_non_discard": rate(sum(1 for r in Uk if [e for e in r["excl"] if e != "discard"]), len(Uk)),
                 "by_reason": {e: rate(sum(1 for r in Uk if e in r["excl"]), len(Uk)) for e in EXCL_PRIMARY}}
    out["auto_exclusion_before_review"] = {"note": "%가 있는 사진 줄 중 후처리(post_exclude)가 점검 전에 자동 제외한 사유별 비율. 한 줄에 사유 여러 개 가능. "
                                           "이 줄들은 풀 밖이라 오염 혼합이 상대하는 줄이 아니다 (이미 걸러짐).", "by_kind": ex}
    fl = {}
    for f in sorted({f for r in P for f in r["flag"]}):
        Lf = [r for r in P if f in r["flag"]]
        fl[f] = rate(sum(isd(r) for r in Lf), len(Lf))
    fl["(플래그 없음)"] = rate(sum(isd(r) for r in P if not r["flag"]), sum(1 for r in P if not r["flag"]))
    out["flag_given_pool"] = {"note": "풀 줄에서 core 표시(flag)별 버려진 비율. 표시가 점검 후보를 고르는 데 쓰여 선택이 섞인다.", "rates": fl}

    # --- 곡면 이탈 크기 (관측끼리: core surf_z)
    ZP = [r for r in P if r["z"] is not None]
    out["surface_z"] = {"note": "surf_z = 서버별 곡면(적합 줄 사람 하나씩 뺌)에서 벗어난 강건 z (post_outlier, 관측만 사용 — 정답 안 씀). 서버 미상·곡면 없는 줄은 null.",
                        "disc_all": zstats([r["z"] for r in ZP if r["disc"]]),
                        "disc_by_cause": {c: zstats([r["z"] for r in ZP if r["disc"] and r["cause"] == c and not r["copy"]]) for c in causes},
                        "kept": zstats([r["z"] for r in ZP if not r["disc"]]),
                        "share_of_measured_rows_disc": rate(sum(1 for r in ZP if r["disc"]), len(ZP)),
                        "disc_given_abs_z": {"le3.5": rate(sum(1 for r in ZP if r["disc"] and abs(r["z"]) <= 3.5), sum(1 for r in ZP if abs(r["z"]) <= 3.5)),
                                              "gt3.5": rate(sum(1 for r in ZP if r["disc"] and abs(r["z"]) > 3.5), sum(1 for r in ZP if abs(r["z"]) > 3.5)),
                                              "gt6": rate(sum(1 for r in ZP if r["disc"] and abs(r["z"]) > 6), sum(1 for r in ZP if abs(r["z"]) > 6))},
                        "kept_abs_z_gt3.5": rate(sum(1 for r in ZP if not r["disc"] and abs(r["z"]) > 3.5), sum(1 for r in ZP if not r["disc"]))}
    out["contaminant_width"] = contaminant_width(P)
    # --- 점검 후보 범위
    cov = collect_review_cov(Ds, rcs, recs)
    out["review_coverage"] = cov
    # --- 값 폭 (고치기 전 원래 값)
    out["original_vs_corrected"] = correct_widths(Ds)
    out["time_shift_hours"] = time_shifts(Ds, rcs)
    # --- 시즌별 퍼짐·늘어나는 창
    out["hetero"] = {"photo_disc": hetero(P, isd, lambda r: True), "photo_fix": hetero(P, isf, lambda r: True),
                     "text_disc": hetero(T, isd, lambda r: True)}
    out["expanding_window"] = {"photo_disc": expanding(P, isd, lambda r: True), "text_disc": expanding(T, isd, lambda r: True)}
    return out


def collect_review_cov(Ds, rcs, recs):
    byk = collections.defaultdict(collections.Counter)
    for s in SEASONS:
        for c in rcs[s]:
            d = c["decision"]
            cls = "keep" if d == "keep" else ("error" if d in ERR_DEC else ("later" if d == "later" else (d or "none")))
            byk[c["kind"]][cls] += 1
    items = {}
    for k, v in sorted(byk.items()):
        n_dec = v["keep"] + v["error"]
        items[k] = {"keep": v["keep"], "error(row·server·discard)": v["error"], "later": v["later"],
                    "other": sum(x for kk, x in v.items() if kk not in ("keep", "error", "later")),
                    "error_rate_of_decided": rate(v["error"], n_dec)}
    P = {(r["s"], r["oi"]): r for r in recs if r["pool"] and r["oi"] is not None}
    cand_oi = collections.defaultdict(set)
    kind_oi = collections.defaultdict(lambda: collections.defaultdict(set))
    for s in SEASONS:
        for c in rcs[s]:
            for oi in c["oi"]:
                cand_oi[s].add(oi)
                kind_oi[c["kind"]][s].add(oi)
    inside = [r for (s, oi), r in P.items() if oi in cand_oi[s]]
    outside = [r for (s, oi), r in P.items() if oi not in cand_oi[s]]
    cov = {"pool_rows": len(P), "pool_rows_in_any_candidate": len(inside), "coverage_of_pool": r4(len(inside) / len(P)),
           "disc_inside_candidates": rate(sum(r["disc"] for r in inside), len(inside)),
           "disc_outside_candidates": rate(sum(r["disc"] for r in outside), len(outside)),
           "share_of_all_disc_rows_that_were_candidates": r4(sum(r["disc"] for r in inside) / max(1, sum(r["disc"] for r in P.values()))),
           "fix_inside_candidates": rate(sum(r["fix"] for r in inside), len(inside)),
           "fix_outside_candidates": rate(sum(r["fix"] for r in outside), len(outside))}
    kr = {}
    for k, per in kind_oi.items():
        rows = [P[(s, oi)] for s, st in per.items() for oi in st if (s, oi) in P]
        kr[k] = {"pool_rows": len(rows), "disc": rate(sum(r["disc"] for r in rows), len(rows)), "fix": rate(sum(r["fix"] for r in rows), len(rows))}
    return {"note": "review.json 항목(점검 후보)이 가리키는 풀 줄. 후보 밖 버림은 시즌 세션이 글을 읽다 짚은 것(지난 시즌 화면·기록 모음 등). "
                    "후보 안 비율은 후보를 곡면 이탈 등 값으로 골랐기 때문에 전체 오염율의 추정이 아니다.",
            "items_by_kind": items, "rows": cov, "by_kind_rows": kr}


def correct_widths(Ds):
    lp, ls = [], []
    for D in Ds:
        obs = D["obs"]
        by_ir = {(o["img"], str(o["row"])): o for o in obs}
        for it in D["discard"]:
            c = it.get("correct")
            if not c or it["scope"] != "row":
                continue
            o = by_ir.get((it["img"], str(it["row"])))
            if not o:
                continue
            if c.get("pct") and o.get("pct"):
                lp.append(math.log10(o["pct"] / c["pct"]))
            if c.get("score") and o.get("score"):
                ls.append(math.log10(o["score"] / c["score"]))
    f = lambda v: {"n": len(v), "log10_ratio_sorted": [r4(x) for x in sorted(v)], "abs_median": r4(float(np.median(np.abs(v)))) if v else None}
    return {"note": "discard.json `correct`(원본을 보고 적은 바른 값)가 있는 줄의 읽힌 값/바른 값. 표본이 아주 작다. 양수 = 읽힌 값이 더 큼.",
            "pct": f(lp), "score": f(ls)}


def time_shifts(Ds, rcs):
    sh = []
    for D in Ds:
        rows = {r["oi"]: r for r in D["rows"] if r["oi"] is not None}
        for c in rcs[D["s"]]:
            if c.get("verdict") != "time" or c["decision"] not in ERR_DEC or not c.get("est") or len(c["oi"]) != 1:
                continue
            m = H_RE.search(c["est"])
            if not m:
                continue
            oi = next(iter(c["oi"]))
            r = rows.get(oi)
            if not r or r["h"] is None:
                continue
            mid = (float(m.group(1)) + float(m.group(2))) / 2
            sh.append((r["h"] - mid, r["kind"], r["t_grade"]))
    v = [x[0] for x in sh]
    out = {"n": len(v), "note": "점검이 시각 오류로 본 줄(4.2b 굵은 오차·원인 조사)의 표 시각 h − 점검이 추정한 실제 촬영 시각 구간 중앙. 양수 = 표가 더 늦게 잡음. "
                                "est_shot_time_kst에 `h a~b`가 적힌 줄만."}
    if v:
        a = np.array(v)
        out.update({"shift_h_quantiles_10_25_50_75_90": [r4(x) for x in np.percentile(a, [10, 25, 50, 75, 90])],
                    "frac_shift_gt_6h": r4((np.abs(a) > 6).mean()), "frac_shift_gt_24h": r4((np.abs(a) > 24).mean()),
                    "by_kind": dict(collections.Counter(x[1] for x in sh))})
    return out


# ---------------------------------------------------------------- 1e
def label_rows(D, recs_s):
    """사진 줄마다 점검 전(pre)·후(post) 서버와 근거. 반환 dict oi → 칸."""
    pre = D["pre"]["server"]
    post = D["server"]
    pool = {r["oi"]: r["pool"] for r in recs_s if r["oi"] is not None}
    discd = {r["oi"]: r["disc"] for r in recs_s if r["oi"] is not None}
    use = {r["oi"]: r["use"] for r in recs_s if r["oi"] is not None}
    out = {}
    for oi, (a, b) in enumerate(zip(pre, post)):
        assert a["img"] == b["img"] and str(a["row"]) == str(b["row"])
        out[oi] = {"s": D["s"], "oi": oi, "post": D["obs"][oi]["no"], "pre_srv": a["server"], "pre_basis": a["basis"],
                   "post_srv": b["server"], "post_basis": b["basis"], "pool": pool.get(oi, False), "disc": discd.get(oi, False),
                   "use": use.get(oi, False), "site": D["obs"][oi]["site"], "img": D["obs"][oi]["img"]}
    return out


def analyze_1e_labels(Ds, labs, claim_labs):
    allr = [v for s in SEASONS for v in labs[s].values()]
    pool = [r for r in allr if r["pool"]]
    lab = lambda r: r["pre_srv"] is not None
    chg = lambda r: r["pre_srv"] is not None and r["post_srv"] != r["pre_srv"]
    gain = lambda r: r["pre_srv"] is None and r["post_srv"] is not None
    g = lambda r: basis_group(r["pre_basis"])
    out = {"note": "점검 전 서버 = 서버 고침 항목(action server)을 뺀 채 서버 분류기를 다시 돌린 결과. 착오 = 점검 전 라벨이 있었는데 고침으로 다른 서버가 된 줄. "
                   "점검이 후보 위주라 하한. 시즌 repro: " + ", ".join(f"S{D['s']}={'같음' if D.get('repro_server') else '다름'}" for D in Ds if not D.get('repro_server')) + " (비어 있으면 모두 같음)"}
    def table(universe, name):
        t = {}
        for key, fn in (("basis", g), ("site", lambda r: r["site"]), ("pre_server", lambda r: r["pre_srv"])):
            gr = collections.defaultdict(list)
            for r in universe:
                if lab(r):
                    gr[fn(r)].append(r)
            t[key] = {}
            for k in sorted(gr, key=str):
                L = gr[k]
                d = rate(sum(chg(r) for r in L), len(L))
                if len(L) >= 30:
                    d["boot_post"] = cluster_boot([((r["s"], r["post"]), int(chg(r)), 1) for r in L], B=1000)
                units = collections.defaultdict(int)
                for r in L:
                    units[(r["s"], r["post"])] |= int(chg(r))
                d["units_post"] = rate(sum(units.values()), len(units))
                t[key][str(k)] = d
        L = [r for r in universe if lab(r)]
        t["all_labeled"] = rate(sum(chg(r) for r in L), len(L))
        t["all_labeled"]["boot_post"] = cluster_boot([((r["s"], r["post"]), int(chg(r)), 1) for r in L])
        t["gained_label_by_fix"] = rate(sum(gain(r) for r in universe), sum(1 for r in universe if r["pre_srv"] is None))
        # 글(군집) 단위
        posts = collections.defaultdict(lambda: [0, 0])
        for r in L:
            posts[(r["s"], r["post"])][0] += 1
            posts[(r["s"], r["post"])][1] += int(chg(r))
        t["posts_with_labeled_rows"] = rate(sum(1 for v in posts.values() if v[1]), len(posts))
        tr = collections.Counter(f"{r['pre_srv']}->{r['post_srv']}" for r in universe if chg(r))
        t["transitions"] = dict(tr)
        t["by_season"] = {f"S{s}": rate(sum(chg(r) for r in universe if r["s"] == s and lab(r)), sum(1 for r in universe if r["s"] == s and lab(r)))
                          for s in SEASONS}
        ks = [sum(chg(r) for r in universe if r["s"] == s and lab(r)) for s in SEASONS]
        ns = [sum(1 for r in universe if r["s"] == s and lab(r)) for s in SEASONS]
        t["hetero"] = {"chi2": het_chi2(ks, ns), "betabinom": beta_fit_from_rates(ks, ns)}
        t["expanding_window"] = expanding(universe, chg, lab)
        return t
    out["pool_rows"] = table(pool, "pool")
    out["all_labeled_photo_rows"] = table(allr, "all")
    # 글(text) 라벨
    cl = [v for s in SEASONS for v in claim_labs[s]]
    cg = lambda r: basis_group(r["pre_basis"])
    clab = [r for r in cl if r["pre_srv"] is not None]
    cchg = lambda r: r["post_srv"] != r["pre_srv"]
    t = {"all_labeled": rate(sum(cchg(r) for r in clab), len(clab)), "by_basis": {}}
    for k in sorted({cg(r) for r in clab}):
        L = [r for r in clab if cg(r) == k]
        t["by_basis"][k] = rate(sum(cchg(r) for r in L), len(L))
    # 글 하나 단위 (한 글의 claims 여러 줄이 같은 라벨)
    posts = collections.defaultdict(lambda: [0, 0])
    for r in clab:
        posts[(r["s"], r["no"])][0] = 1
        posts[(r["s"], r["no"])][1] |= int(cchg(r))
    t["posts_labeled"] = rate(sum(v[1] for v in posts.values()), len(posts))
    out["claim_text_labels"] = t
    return out


def run_mix(s, outdir):
    """server_mix.py 기본 설정 (--cv 없음)을 임시 위치에 돌린다 (repo의 state는 읽기만, 출력은 --out). 몇 초."""
    base = os.path.join(outdir, f"S{s}")
    if os.path.exists(base + "_summary.json"):
        return base
    subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "solo_raid", "server_mix.py"),
                    os.path.join(DATA, f"S{s}", "state"), "--out", base], check=True, capture_output=True)
    return base


def load_mix(base):
    rows = {r["oi"]: r for r in jl(base + ".jsonl")}
    summ = json.load(open(base + "_summary.json", encoding="utf-8"))
    return rows, summ


def mono_hidden_test(D, T, hide_kr_frac, rng):
    """라벨을 가리고 mono 거르기(predict.mono_filter)를 다시 돌린다 (분석 4.7 줄 단위 시험의 재현).
    라벨 일섭·글섭 줄은 미상으로 바꾸고, 라벨 한섭은 사람 단위로 hide_kr_frac 만큼 미상으로 바꾼다 (나머지 한섭은 고정).
    반환 {라벨 구분: [빠진 줄, 전체 줄]} (줄 = 거르기를 지난 use 줄)."""
    H = D["head"]["H"]
    surv = []
    for r in D["rows"]:
        why, he = O.reason_of(r, T, H)
        if why is not None:
            continue
        srv = r["server"] if r["server"] in SRV else None
        surv.append((r, he, srv))
    kr_pers = sorted({r["pid"] for r, _, s in surv if s == "KR"})
    hide_pers = {p for p in kr_pers if rng.random() < hide_kr_frac}
    rows = []
    truth = []
    for r, he, srv in surv:
        hidden = srv in ("JP", "GL") or (srv == "KR" and r["pid"] in hide_pers)
        sv = None if (srv is None or hidden) else srv
        rows.append({"server": sv, "unused": sv is None, "person": r["pid"], "s": r["score"], "x": r["pct"],
                     "span": C.PR.time_span(r, he)})
        truth.append(srv if srv is not None else "UNK")
    drop, _ = C.PR.mono_filter(rows)
    cnt = collections.defaultdict(lambda: [0, 0])
    for i, t in enumerate(truth):
        if rows[i]["server"] is not None:
            continue   # 고정 한섭은 시험 밖
        cnt[t][1] += 1
        cnt[t][0] += int(i in drop)
    return dict(cnt)


def dirichlet_moments(vecs, classes):
    """사이트별 시즌 벡터(확률)에서 평균과 모멘트법 농도 α0 (성분별 α0 = m(1−m)/var − 1의 중앙값)."""
    a = np.array([[v[c] for c in classes] for v in vecs], float)
    if len(a) < 3:
        return None
    m = a.mean(0)
    var = a.var(0, ddof=1)
    est = [m[k] * (1 - m[k]) / var[k] - 1 for k in range(len(classes)) if var[k] > 1e-9 and 0 < m[k] < 1]
    a0 = float(np.median(est)) if est else None
    return {"mean": {c: r4(m[k]) for k, c in enumerate(classes)}, "sd": {c: r4(math.sqrt(var[k])) for k, c in enumerate(classes)},
            "alpha0_moment": r4(a0), "alpha": {c: r4(m[k] * a0) for k, c in enumerate(classes)} if a0 else None, "n_seasons": int(len(a))}


def analyze_1e_unknown(Ds, recs, mixes, cvs):
    out = {"seasons": {}}
    UK = []
    for D in Ds:
        s = D["s"]
        rows, summ = mixes[s]
        R = [r for r in recs if r["s"] == s and r["use"] and r["server"] is None and r["oi"] is not None]
        kept = [r for r in R if not r["mono_drop"]]
        drop = [r for r in R if r["mono_drop"]]
        def stats(L):
            ps = [rows[r["oi"]]["p"] for r in L if rows.get(r["oi"]) and rows[r["oi"]].get("p")]
            if not ps:
                return {"n_rows": len(L), "n_with_p": 0}
            pk = np.array([p.get("KR", 0) for p in ps])
            return {"n_rows": len(L), "n_with_p": len(ps), "mean_p_nonKR": r4(float(1 - pk.mean())),
                    "frac_pKR_lt_0.5": r4(float((pk < 0.5).mean())), "frac_pKR_lt_0.95": r4(float((pk < 0.95).mean())),
                    "expected_nonKR_rows": r4(float((1 - pk).sum()))}
        d = {"use_unknown_rows": len(R), "mono_dropped": len(drop), "mono_kept": len(kept),
             "kept": stats(kept), "dropped": stats(drop),
             "labeled_use_rows": summ["rows"]["use_labeled"], "unknown_units": summ["rows"]["units_unlabeled"],
             "prior_em_unlabeled": summ.get("prior_em_unlabeled"), "unlabeled_best": summ.get("unlabeled_best"),
             "unlabeled_conf": summ.get("unlabeled_conf"), "unlabeled_sum_p": summ.get("unlabeled_sum_p"),
             "labeled_units_by_server": summ["rows"].get("labeled_units_by_server"), "gate": {k: summ["gate"].get(k) for k in ("pass", "acc_kr_binary", "reason")}
             if summ.get("gate") else None}
        sit = collections.Counter(r["site"] for r in kept)
        d["kept_by_site"] = dict(sit)
        out["seasons"][f"S{s}"] = d
        UK += [(s, r) for r in kept]
    # --- 합침
    tot_n = sum(v["mono_kept"] for v in out["seasons"].values())
    tot_dropped = sum(v["mono_dropped"] for v in out["seasons"].values())
    k_exp = sum(v["kept"].get("expected_nonKR_rows", 0) or 0 for v in out["seasons"].values())
    n_p = sum(v["kept"].get("n_with_p", 0) for v in out["seasons"].values())
    out["pooled"] = {"mono_kept_rows": tot_n, "mono_dropped_rows": tot_dropped,
                     "kept_rows_with_p": n_p, "expected_nonKR_among_kept_by_post_server_mix": r4(k_exp),
                     "frac_expected_nonKR_among_kept": r4(k_exp / n_p) if n_p else None,
                     "note": "post_server_mix의 한섭 확률은 곡면이 한섭 라벨 위주로 배운 것이라 한섭 쏠림이 있다 (분석 server_mix 한계). 미상 줄의 진짜 서버가 아니다."}
    # --- cv 지표 (존재하는 시즌)
    cvout = {}
    pooled_bins = collections.defaultdict(lambda: [0.0, 0.0, 0])
    unit_bins = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for s, summ in cvs.items():
        c = {}
        for design in ("post:em_surf_emprior", "unit:em_surf_emprior"):
            m = (summ.get("cv", {}).get(design) or {}).get("metrics", {}).get("all")
            if not m:
                continue
            c[design] = {k: m.get(k) for k in ("n", "acc", "ece", "brier", "logloss", "recall")}
            if design == "post:em_surf_emprior":
                for b in m.get("calib", []):
                    key = tuple(b["bin"])
                    pooled_bins[key][0] += b["conf"] * b["n"]
                    pooled_bins[key][1] += b["hit"] * b["n"]
                    pooled_bins[key][2] += b["n"]
        mu = (summ.get("cv", {}).get("unit:em_surf_emprior") or {}).get("metrics", {}).get("units")
        if mu:
            c["unit_design_unit_level"] = {k: mu.get(k) for k in ("n", "acc", "ece", "brier", "logloss", "recall")}
            for b in mu.get("calib", []):
                key = tuple(b["bin"])
                unit_bins[key][0] += b["conf"] * b["n"]
                unit_bins[key][1] += b["hit"] * b["n"]
                unit_bins[key][2] += b["n"]
        if summ.get("gate"):
            c["gate"] = {k: summ["gate"].get(k) for k in ("pass", "acc_kr_binary", "reason")}
        cvout[f"S{s}"] = c
    out["cv_by_season"] = cvout
    out["cv_missing_seasons"] = [f"S{s}" for s in SEASONS if s not in cvs]
    out["cv_pooled_calibration_post_design"] = [{"bin": list(k), "mean_conf": r4(v[0] / v[2]) if v[2] else None, "hit": r4(v[1] / v[2]) if v[2] else None, "n": v[2],
                                                  "hit_wilson": wilson(round(v[1]), v[2])}
                                                 for k, v in sorted(pooled_bins.items())]
    out["cv_pooled_calibration_unit_design_unit_level"] = [{"bin": list(k), "mean_conf": r4(v[0] / v[2]) if v[2] else None, "hit": r4(v[1] / v[2]) if v[2] else None, "n": v[2],
                                                             "hit_wilson": wilson(round(v[1]), v[2])} for k, v in sorted(unit_bins.items())]
    out["cv_note"] = ("server_mix --cv (post: 글 번호 sha1 5겹, 같은 사람·사진·글을 학습에서 지움 / unit: 묶음 하나씩 빼기). CV는 라벨 줄로 한다 — 라벨 줄과 미상 줄은 분포가 다르다 "
                      "(미상은 결과 화면·48시간 전이 많다). 시즌 %d개의 요약이 있다: %s. 없는 시즌은 --cv가 멈췄거나(적합 줄 부족: 특이행렬·0 나누기) 아직 안 돌린 것이다." % (len(cvout), ",".join(sorted(cvout, key=lambda x: int(x[1:])))))
    # --- 사이트별 EM 사전 (디리클레 모멘트)
    classes = ["KR", "JP", "GL", "OTHER"]
    dir_ = {}
    for site in ("dc", "naver", "blabla", "*"):
        vecs = []
        for s in SEASONS:
            pr = (out["seasons"][f"S{s}"].get("prior_em_unlabeled") or {}).get(site)
            if pr:
                vecs.append({c: pr.get(c, 0.0) for c in classes})
        dir_[site] = dirichlet_moments(vecs, classes)
    out["site_prior_em_dirichlet_moments"] = dir_
    return out


def analyze_1e_mono(Ds, B_REP=5):
    rng = np.random.default_rng(SEED)
    out = {"note": "분석 4.7 줄 단위 시험을 표 칸으로 다시 돌린 것. 일섭·글섭 라벨 줄을 가린 채 mono_filter가 몇 %를 뺐는지 (뺀 비율 = 거르는 힘). "
                   "한섭 사람의 절반을 무작위로 가려 잘못 빼는 비율도 잰다 (사람 단위, 반복 " + str(B_REP) + "회 합산). T는 그 시점까지 게시된 줄."}
    per = {}
    for T in (24, 48, 96, "end"):
        agg = collections.defaultdict(lambda: [0, 0])
        for D in Ds:
            H = D["head"]["H"]
            TT = H if T == "end" else float(T)
            if TT > H:
                continue
            for rep in range(B_REP if T != "end" else B_REP):
                c = mono_hidden_test(D, TT, 0.5, rng)
                for k, (a, b) in c.items():
                    agg[k][0] += a
                    agg[k][1] += b
        per[str(T)] = {k: {"dropped": a, "n": b, "frac_dropped": r4(a / b) if b else None, "wilson_rows": wilson(a, b)} for k, (a, b) in sorted(agg.items())}
    out["by_T"] = per
    ps = {}
    for D in Ds:
        agg = collections.defaultdict(lambda: [0, 0])
        c = mono_hidden_test(D, D["head"]["H"], 0.5, np.random.default_rng(SEED + D["s"]))
        for k, (a, b) in c.items():
            agg[k] = [a, b]
        ps[f"S{D['s']}"] = {k: {"dropped": a, "n": b} for k, (a, b) in agg.items()}
    out["end_by_season"] = ps
    return out


DOC_4_7 = {"source": "docs/solo-raid-analysis.md 「### 4.7 점수 비교로 서버 미상 줄 거르기」 줄 단위 시험 (S19~S41 기본 집합, 라벨 줄을 정답으로, 2026-10-19 문서값 그대로)",
           "interval_as_is(C 구간 그대로 아님·게시 6시간 가정 없음)": {"JP_dropped_T24_T48_T96_end": [0.30, 0.35, 0.50, 0.64], "GL_dropped_T24_T48_T96_end": [0.25, 0.33, 0.37, 0.61],
                                                                        "KR_wrongly_dropped": "2~3%", "unknown_dropped": "1~3%"},
           "C_6h_assumption": {"JP": [0.32, 0.53, 0.73, 0.80], "GL": [0.62, 0.85, 0.89, 0.94], "KR_wrongly_dropped": "2~7%", "unknown_dropped": "3~8%"},
           "note": "표 칸으로 다시 센 값은 mono_hidden_label_test.by_T. 일섭·글섭 라벨을 가린 채 (한섭 사람 절반도 가림) 뺀 비율. 위 문서값은 다른 자료 판·한섭 라벨 고정이라 조금 다르다."}


def decided_buckets(Ds, links):
    """discard.json 항목을 결정 날짜로 나눈다: 첫 점검(~2026-10-17) · 둘째 걸이(2026-10-18~: 4.2b 굵은 오차·순서표 3 원인 조사 — 모형 잔차로 고른 칸에서 시작,
    2026-10-28에 discard.json에 옮김). 둘째 걸이가 첫 점검이 놓친 오류를 얼마나 더 찾았는지 = 첫 점검 놓침의 하한 (포획-재포획 눈치)."""
    out = collections.defaultdict(lambda: collections.Counter())
    for D in Ds:
        for l in links[D["s"]]:
            d = l["date"] or "?"
            b = "first_pass(decided<=10-17)" if d <= "2026-10-17" else "second_pass(decided>=10-18, applied 10-28)"
            out[b][f"{l['action']}/{l['cause']}"] += 1
            out[b]["items"] += 1
            out[b]["tied_to_review"] += int(l["by_review"])
    return {k: dict(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tbl", required=True, help="관측 표 폴더 (full/S{n}.jsonl.gz)")
    ap.add_argument("--mirror", required=True, help="repo 밖 작업 폴더 (서버 분류 재실행·server_mix 출력)")
    ap.add_argument("--cvdir", default=None, help="server_mix --cv 요약이 있는 폴더 (S{n}_summary.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mono-reps", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    mirror = a.mirror
    os.makedirs(os.path.join(mirror, "mix"), exist_ok=True)
    Ds = [load_season(s, a.tbl, mirror) for s in SEASONS]
    print(f"읽기·서버 분류 재실행 {time.time() - t0:.0f}초", flush=True)
    links = {D["s"]: build_link(D) for D in Ds}
    recs_by = {D["s"]: season_records(D, links[D["s"]]) for D in Ds}
    for s in SEASONS:
        dz = discard_z(s)
        for r in recs_by[s]:
            if r["disc"] and r["oi"] in dz:
                r["z"], r["z_support"], r["eL"] = dz[r["oi"]]
    recs = [r for s in SEASONS for r in recs_by[s]]
    rcs = {D["s"]: review_cands(D) for D in Ds}
    d1 = analyze_1d(Ds, links, recs, rcs)
    d1["decided_buckets"] = decided_buckets(Ds, links)
    print(f"1d {time.time() - t0:.0f}초", flush=True)

    labs = {D["s"]: label_rows(D, recs_by[D["s"]]) for D in Ds}
    claim_labs = {}
    for D in Ds:
        pre, post = D["pre"]["server_claims"], D["server_claims"]
        assert len(pre) == len(post)
        claim_labs[D["s"]] = [{"s": D["s"], "no": b["no"], "pre_srv": a_["server"], "pre_basis": a_["basis"], "post_srv": b["server"]}
                              for a_, b in zip(pre, post)]
    e1 = analyze_1e_labels(Ds, labs, claim_labs)
    e1["review_server_items"] = server_review_items(Ds, rcs)
    mixes = {}
    for D in Ds:
        mixes[D["s"]] = load_mix(run_mix(D["s"], os.path.join(mirror, "mix")))
    cvs = {}
    if a.cvdir:
        for s in SEASONS:
            p = os.path.join(a.cvdir, f"S{s}_summary.json")
            if os.path.exists(p):
                cvs[s] = json.load(open(p, encoding="utf-8"))
    e1["unknown_rows"] = analyze_1e_unknown(Ds, recs, mixes, cvs)
    print(f"1e 라벨·미상 {time.time() - t0:.0f}초", flush=True)
    e1["mono_hidden_label_test"] = analyze_1e_mono(Ds, a.mono_reps)
    print(f"1e mono {time.time() - t0:.0f}초", flush=True)
    e1["implied_nonKR_after_mono"] = implied_leak(e1["unknown_rows"], e1["mono_hidden_label_test"])

    d1 = dict({"item": "1d"}, **d1)
    e1 = dict({"item": "1e"}, **e1)
    e1["doc_4_7_reference"] = DOC_4_7
    res = {"1d": d1, "1e": e1, "meta": {"seed": SEED, "seasons": "S19-S42", "tbl": "자료 판 D0 끝난 시즌 전체", "bootstrap_unit": "(시즌, 글) 묶음",
                                         "seconds": r4(time.time() - t0)}}
    res["check_rows"] = check_rows(Ds, recs, labs, mixes)
    recommend(res)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_md(res, os.path.splitext(a.out)[0] + ".md")
    print("저장", a.out, f"{time.time() - t0:.0f}초")


def server_review_items(Ds, rcs):
    """점검 판단 중 서버와 관련된 것: decision server(서버 고침)인 항목을 종류별로, 그리고 후보 줄 수."""
    out = collections.defaultdict(lambda: {"server_fix_items": 0, "rows": 0})
    tot = collections.Counter()
    for D in Ds:
        for c in rcs[D["s"]]:
            tot[c["kind"]] += 1
            if c["decision"] == "server":
                out[c["kind"]]["server_fix_items"] += 1
                out[c["kind"]]["rows"] += len(c["oi"])
    return {"by_kind": {k: dict(v, items_of_kind=tot[k]) for k, v in sorted(out.items())},
            "total_server_fix_items": sum(v["server_fix_items"] for v in out.values())}


def check_rows(Ds, recs, labs, mixes):
    """원자료 확인용 표본: (시즌, key, 칸). 글 번호 없이. 조율 세션이 state의 obs·discard·server와 견준다."""
    rng = np.random.default_rng(SEED)
    out = []
    P = [r for r in recs if r["pool"]]
    for cause in ("time", "wrongscreen", "read", "other"):
        L = [r for r in P if r["disc"] and r["cause"] == cause and not r["copy"] and r["kind"] != "text"]
        for r in [L[i] for i in rng.choice(len(L), size=min(2, len(L)), replace=False)] if L else []:
            out.append({"what": f"버려진 줄 (원인 {cause})", "season": f"S{r['s']}", "key": r["key"], "kind": r["kind"], "screen": r["screen"], "site": r["site"],
                        "server": r["server"], "server_basis": r["basis"], "surf_z": r4(r["z"]), "h": r["h"], "hp": r["hp"],
                        "excl": r["excl"], "check": "state/obs.jsonl 이 oi 줄의 img·row가 S{n}/discard.json 항목(scope row/img/post)과 같은지"})
    ch = [v for s in SEASONS for v in labs[s].values() if v["pool"] and v["pre_srv"] is not None and v["post_srv"] != v["pre_srv"]]
    for v in [ch[i] for i in rng.choice(len(ch), size=min(3, len(ch)), replace=False)] if ch else []:
        out.append({"what": "서버가 고쳐진 줄", "season": f"S{v['s']}", "key": f"p{v['oi']}", "pre_server": v["pre_srv"], "pre_basis": v["pre_basis"],
                    "post_server": v["post_srv"], "post_basis": v["post_basis"],
                    "check": "state/server.jsonl 이 줄의 서버가 점검 전(pre_server)과 다르고, discard.json에 같은 img(row)·글의 action server 항목이 있는지 (근거 fix이거나 묶음 이름이 바뀐 줄)"})
    for s in (38, 41):
        rows, _ = mixes[s]
        L = [r for r in recs if r["s"] == s and r["use"] and r["server"] is None and not r["mono_drop"] and r["oi"] in rows and rows[r["oi"]].get("p")]
        L.sort(key=lambda r: rows[r["oi"]]["p"].get("KR", 1))
        for r in L[:1]:
            out.append({"what": "mono를 지난 미상 줄 중 한섭 확률이 가장 낮은 줄", "season": f"S{s}", "key": r["key"], "kind": r["kind"],
                        "p": {k: r4(v) for k, v in rows[r["oi"]]["p"].items()}, "basis": rows[r["oi"]].get("basis"), "mono_drop": r["mono_drop"],
                        "check": "state/post_server_mix(재실행)과 core.jsonl server null"})
    return out


def betaab(mean, conc):
    return {"a": r4(mean * conc), "b": r4((1 - mean) * conc), "mean": r4(mean), "conc_a_plus_b": r4(conc)}


TUNE_SEASONS = (22, 26, 30, 34, 38, 42)


def recommend(res):
    d1, e1 = res["1d"], res["1e"]
    R = d1["rates"]
    p_obs = R["photo_pool_disc"]["p"]
    hb = d1["hetero"]["photo_disc"]["betabinom"]
    conc_season = hb["conc_a_plus_b"] if hb else 100.0
    share2 = R["second_pass_share_of_disc_rows"]
    zq = d1["surface_z"]["disc_all"]
    vis = 1 - zq["frac_gt_3.5"] if zq.get("n") else None
    f_a = 1 / (1 - share2)
    f_b = 1 / zq["frac_gt_3.5"] if zq.get("n") and zq["frac_gt_3.5"] else None
    f_c = float(np.clip(round((f_a + (f_b or f_a)) / 2 * 2) / 2, 1.5, 3.0))   # 0.5 단위 반올림, 1.5~3
    conc = 100.0
    d1["recommend"] = {
        "contamination_weight_photo": {
            "observed_lower_bound": {"p": p_obs, "wilson": R["photo_pool_disc"]["wilson"], "n_rows": R["photo_pool_disc"]["n"], "k": R["photo_pool_disc"]["k"],
                                     "season_betabinom_fit": hb},
            "coverage_factor": {"from_second_pass": r4(f_a), "from_z_visibility": r4(f_b) if f_b else None, "chosen_central": f_c, "range": [1.0, 3.0],
                                "basis": "(1) 둘째 걸이(모형 잔차로 고른 칸, 전체 풀의 극히 일부)가 버려진 줄의 %.0f%%를 따로 찾았다 → 첫 점검 재현율 ≲ %.0f%% → ×%.2f. "
                                         "(2) 확인된 오류 줄 가운데 곡면 이탈 z가 3.5 넘는 것은 %.0f%%뿐 → z 기반 후보 선별이 보는 몫이 그 정도 → ×%.2f. "
                                         "두 값의 평균을 0.5 단위로 반올림해 중심값으로 하고 1.0(점검이 거의 다 찾았다)~3.0(시각 오류처럼 눈에 안 띄는 것이 더 많다)을 범위로 둔다. 판단이지 측정이 아니다."
                                         % (share2 * 100, (1 - share2) * 100, f_a, (zq["frac_gt_3.5"] if zq.get("n") else 0) * 100, f_b or 0)},
            "prior_lower_bound": betaab(p_obs, conc),
            "prior_central": betaab(min(p_obs * f_c, 0.5), conc),
            "prior_high": betaab(min(p_obs * 3.0, 0.5), conc),
            "concentration_note": "a+b = 100은 시즌 퍼짐(베타-이항 적합 a+b = %.0f)보다 약간 약하게 잡은 것 — 보정 계수의 불확실이 겹치기 때문. 표본 수 n 때문이 아니라 시즌 간 이질성이 퍼짐의 원천이다." % conc_season,
            "same_prior_for_kinds": "A %.4f %s·B %.4f %s·C %.4f %s — 구간이 겹쳐 A·B·C에 같은 사전을 쓰고, B는 표본(n=%d)이 작아 따로 못 정한다." % (
                d1["by"]["kind"]["A"]["p"], d1["by"]["kind"]["A"]["wilson"], d1["by"]["kind"]["B"]["p"], d1["by"]["kind"]["B"]["wilson"],
                d1["by"]["kind"]["C"]["p"], d1["by"]["kind"]["C"]["wilson"], d1["by"]["kind"]["B"]["n"]),
            "optional_multipliers_vs_overall": {
                "screen=season_end": r4(d1["by"]["screen"]["season_end"]["p"] / p_obs),
                "site=blabla": r4(d1["by"]["site"]["blabla"]["p"] / p_obs),
                "server_basis=top7_post(+text)": r4(d1["by"]["server_basis"]["top7_post(+text)"]["p"] / p_obs),
                "note": "표본이 작고(season_end n=%d·blabla n=%d·top7_post n=%d) 글 묶음 영향이 크다 — 참고용." % (
                    d1["by"]["screen"]["season_end"]["n"], d1["by"]["site"]["blabla"]["n"], d1["by"]["server_basis"]["top7_post(+text)"]["n"])},
        },
        "contamination_weight_text": {"p": R["text_disc"]["p"], "k": R["text_disc"]["k"], "n": R["text_disc"]["n"], "wilson": R["text_disc"]["wilson"],
                                      "note": "글로만 적은 %줄은 점검이 줄 단위로 본 적이 없다(글째 버림만 2줄). 이 비율은 정보가 없다 — 1c의 사진 견줌 결과로 사전을 따로 정한다."},
        "server_fix_rate_pool_rows": R["photo_pool_fix"],
        "contaminant_density": d1["contaminant_width"],
        "how_wrong": {"abs_z_quantiles_10_25_50_75_90": zq.get("abs_z_q10_25_50_75_90"), "frac_z_gt_3.5": zq.get("frac_gt_3.5"), "frac_z_gt_10": zq.get("frac_gt_10"),
                      "reading": "버려진 줄의 곡면 이탈은 「조금 어긋남」(|z|≤3.5가 %.0f%%, 주로 시각 오류·같은 사람 다른 화면)과 「완전히 엉뚱함」(지난 시즌 화면·판독 틀림·다른 서버)이 섞인 두 성분이다. "
                                 "엉뚱한 쪽만 곡면 이탈로 보인다." % ((vis or 0) * 100)},
    }
    # --- 늘어나는 창 (오염 비율)
    ew = d1["expanding_window"]["photo_disc"]["by_season"]
    d1["recommend"]["expanding_window_tune_seasons"] = {f"S{s}": ew[f"S{s}"] for s in TUNE_SEASONS if f"S{s}" in ew}
    # --- 1e
    pb = e1["pool_rows"]["basis"]
    lab = {}
    for b, v in pb.items():
        u = v["units_post"]
        mean_lb = (u["k"] + 0.5) / (u["n"] + 1)
        cc = float(min(u["n"] + 1, 100))
        lab[b] = {"units": {"k": u["k"], "n": u["n"], "wilson": u["wilson"]}, "rows": {"k": v["k"], "n": v["n"], "wilson": v["wilson"]},
                  "beta_lower_bound_jeffreys": {"a": r4(u["k"] + 0.5), "b": r4(u["n"] - u["k"] + 0.5)},
                  "beta_central": betaab(min(mean_lb * f_c, 0.5), cc)}
    gb = e1["pool_rows"]["hetero"]["betabinom"]
    e1["recommend"] = {
        "label_error_by_basis": lab,
        "note": "단위 = (시즌, 글): 한 글·한 묶음의 줄은 같은 라벨이라 줄 수가 아니라 글 수가 표본이다. 중심 사전 = 하한 평균 × 계수 %.1f, 농도 min(글 수+1, 100). "
                "season 이질성: 전체 착오율의 베타-이항 적합 a+b = %s (시즌마다 퍼짐 큼)." % (f_c, gb["conc_a_plus_b"] if gb else None),
        "claim_text_labels": e1["claim_text_labels"]["all_labeled"],
    }
    sp = e1["unknown_rows"]["site_prior_em_dirichlet_moments"]
    e1["recommend"]["unknown_row_site_dirichlet"] = {s: {"mean": v["mean"], "alpha0_moment": v["alpha0_moment"], "alpha": v["alpha"], "sd_across_seasons": v["sd"]}
                                                      for s, v in sp.items() if v}
    e1["recommend"]["unknown_row_site_dirichlet_note"] = ("미상 줄(mono 거르기 전)의 서버 섞임: 시즌 EM 사전 24개의 평균과 모멘트법 농도. dc는 거의 한섭(안정), naver·blabla는 시즌마다 크게 흔들린다. "
                                                          "EM 사전은 한섭 곡면에 기대어 낸 값이라 한섭 쪽으로 기운다 (server_mix 한계).")
    # 늘어나는 창: 미상 줄 비한섭 비율
    ser = {}
    for site in ("dc", "naver", "blabla"):
        ser[site] = {}
        for s in SEASONS:
            pr = e1["unknown_rows"]["seasons"][f"S{s}"].get("prior_em_unlabeled", {}).get(site)
            ser[site][s] = None if not pr else 1 - pr.get("KR", 0)
    win = {}
    for s in TUNE_SEASONS:
        row = {}
        for site in ("dc", "naver", "blabla"):
            prev = [ser[site][x] for x in SEASONS if x < s and ser[site][x] is not None]
            cur = ser[site][s]
            row[site] = {"prior_mean": r4(np.mean(prev)) if prev else None, "prior_sd": r4(np.std(prev, ddof=1)) if len(prev) > 2 else None, "n_prior_seasons": len(prev),
                         "season_value": r4(cur) if cur is not None else None}
        row["implied_nonKR_after_mono_season"] = e1["implied_nonKR_after_mono"]["by_season"].get(f"S{s}")
        win[f"S{s}"] = row
    e1["recommend"]["expanding_window_unknown_nonKR_share"] = win
    imp = e1["implied_nonKR_after_mono"]
    vals = [v for v in imp["by_season"].values() if v is not None]
    e1["recommend"]["nonKR_among_mono_passed_unknown"] = {
        "pooled_implied": imp["pooled_weighted_by_kept_rows"], "season_range": [r4(min(vals)), r4(max(vals))], "season_sd": r4(float(np.std(vals, ddof=1))),
        "post_server_mix_expected": e1["unknown_rows"]["pooled"]["frac_expected_nonKR_among_kept"],
        "beta_prior": betaab(imp["pooled_weighted_by_kept_rows"], 40.0),
        "note": f"mono를 지난 미상 줄 가운데 비한섭이 남는 몫. 두 추정(EM 사전 x mono가 못 거른 비율 / post_server_mix 기대값)을 견준다. 농도 40은 시즌 퍼짐(sd {float(np.std(vals, ddof=1)):.3f})으로 잡은 값."}
    ewl = e1["pool_rows"]["expanding_window"]["by_season"]
    e1["recommend"]["expanding_window_label_error_tune_seasons"] = {f"S{s}": ewl[f"S{s}"] for s in TUNE_SEASONS if f"S{s}" in ewl}
    e1["recommend"]["mono_filter_power_end"] = {k: e1["mono_hidden_label_test"]["by_T"]["end"][k] for k in ("JP", "GL", "KR", "UNK")}


# ---------------------------------------------------------------- 버려진 줄의 곡면 이탈 (원래 값이 얼마나 엉뚱했나)
def discard_z(s):
    """버려진 사진 줄(exclude `discard`)의 곡면 z — post_outlier.py와 같은 곡면(서버별 A·독립 줄, 사람 하나씩 뺀 척도)·같은 z 정의.
    버려진 줄은 core에서 use가 아니라 post_outlier가 재지 않는다. 곡면은 지금의 기본 집합(버린 뒤)으로 맞추고 버려진 줄을 거기에 댄다.
    반환 {oi: (z, 지지 안 여부)}; 서버 미상·곡면 없는 서버·점수 없음은 빠진다. 정답은 쓰지 않는다 (관측끼리)."""
    import explore_surface as ES
    st = os.path.join(DATA, f"S{s}", "state")
    core = jl(os.path.join(st, "core.jsonl"))
    season = json.load(open(os.path.join(st, "season.json"), encoding="utf-8"))
    H = ES.h_end_of(season)
    photos = [x for x in core if x["src"] == "photo"]
    surf = {}
    for sv in sorted({x["server"] for x in photos if x["server"]}):
        rows = ES.fit_rows(core, sv, H)
        if len(rows) < 12:
            continue
        nk = 5 if len(rows) >= 40 else 3
        mk = lambda nk=nk: ES.SurfL("sqrt", nk, H)
        full = mk().fit(ES.pts_of(rows, H))
        eS, eL, _ = ES.lopo(rows, H, mk)
        surf[sv] = dict(full=full, sS=ES.mad([e for e in eS if e is not None]), sL=ES.mad([e for e in eL if e is not None]))
    out = {}
    for x in photos:
        if "discard" not in (x["exclude"] or []) or x["server"] not in surf or x["score"] is None or not x["pct"]:
            continue
        d = surf[x["server"]]
        m = d["full"]
        if x["t_grade"] == "A" or x["day_lo"] == "post":
            lo = hi = ES.h_eff(x, H)
        else:
            lo, hi = x["h_lo"], x["h_hi"]
            if x["h_lo_top7"] is not None:
                lo, hi = x["h_lo_top7"], x["h_hi_top7"]
            if lo is None or hi is None:
                continue
            lo, hi = max(1.0, min(lo, H)), max(1.0, min(hi, H))
        S = x["score"]
        s_lo, s_hi = m.S_rng
        if s_lo <= S <= s_hi and hi >= m.h_sup:
            he, eS, eL = m.band_resid(x["pct"], S, max(lo, m.h_sup), hi)
            zs = [abs(eL) / d["sL"]] + ([abs(eS) / d["sS"]] if eS is not None else [])
            out[x["oi"]] = ((1 if eL > 0 else -1) * min(zs), True, eL)
        else:
            Sc = min(max(S, s_lo), s_hi)
            he = max(hi, m.h_sup)
            eL = m.resid_L(x["pct"], Sc, min(he, H))
            out[x["oi"]] = (eL / d["sL"], False, eL)
    return out


def time_top7(Ds, recs):
    """시각 오류로 버려진 A·B 줄이 top7 시각 구간 [h_lo_top7, h_hi_top7] 밖에 얼마나 떨어져 있나 (다른 사진들의 순위표로 좁힌 구간 — 관측끼리).
    기준: 버려지지 않은 A·B 풀 줄의 같은 지표."""
    rows = {}
    for D in Ds:
        for r in D["rows"]:
            if r["oi"] is not None:
                rows[(D["s"], r["oi"])] = r
    def dist(row):
        if row["h"] is None or row["h_lo_top7"] is None or row["h_hi_top7"] is None:
            return None
        return max(row["h_lo_top7"] - row["h"], row["h"] - row["h_hi_top7"], 0.0)
    a, b = [], []
    for r in recs:
        if not r["pool"] or r["kind"] not in ("A", "B") or r["oi"] is None:
            continue
        d = dist(rows[(r["s"], r["oi"])])
        if d is None:
            continue
        if r["disc"] and r["cause"] == "time":
            a.append(d)
        elif not r["disc"]:
            b.append(d)
    f = lambda v: {"n": len(v), "frac_outside_gt_1h": r4(float(np.mean(np.array(v) > 1.0))) if v else None,
                   "frac_outside_gt_6h": r4(float(np.mean(np.array(v) > 6.0))) if v else None,
                   "quantiles_50_75_90": [r4(x) for x in np.percentile(v, [50, 75, 90])] if v else None}
    return {"note": "A·B 풀 줄 중 top7 구간이 있는 것만. h − 구간 밖 거리(시간).", "time_error_discarded": f(a), "kept": f(b)}


def implied_leak(unk, mono):
    """mono를 지난 미상 줄에 남는 비한섭 비율의 추정. 사이트별 EM 사전(미상 줄의 서버 섞임)과 mono의 거르는 힘으로:
    통과 비한섭 ∝ π_서버 × (1 − 빠진 비율_서버), 통과 한섭 ∝ π_KR × (1 − 한섭 잘못 빠진 비율). 거르는 힘은 끝 시점 시험의 합친 값을 쓴다
    (NA·OTHER는 일섭·글섭 평균 힘으로 가정)."""
    e = mono["by_T"]["end"]
    leak = {"JP": 1 - e["JP"]["frac_dropped"], "GL": 1 - e["GL"]["frac_dropped"]}
    leak["NA"] = leak["OTHER"] = leak["SEA"] = (leak["JP"] + leak["GL"]) / 2
    kr_keep = 1 - e["KR"]["frac_dropped"]
    out = {"leak_by_server": {k: r4(v) for k, v in leak.items()}, "kr_keep": r4(kr_keep), "by_season": {}}
    num = den = 0.0
    for sname, v in unk["seasons"].items():
        pri = v.get("prior_em_unlabeled") or {}
        tot_n = tot_w = 0.0
        for site, n in v["kept_by_site"].items():
            pr = pri.get(site) or pri.get("*")
            if not pr:
                continue
            nk = pr.get("KR", 0) * kr_keep
            nn = sum(pr.get(c, 0) * leak.get(c, 0.5) for c in pr if c != "KR")
            if nk + nn <= 0:
                continue
            tot_n += n
            tot_w += n * nn / (nk + nn)
        out["by_season"][sname] = r4(tot_w / tot_n) if tot_n else None
        num += tot_w
        den += tot_n
    out["pooled_weighted_by_kept_rows"] = r4(num / den) if den else None
    return out


def contaminant_width(P):
    """버려진 줄의 log10 % 잔차 eL(관측 − 한섭 등 서버 곡면, 점수·시각 고정)의 분포와 널리 쓰는 꼴(정규·라플라스·스튜던트 t·코시) 맞춤.
    버려진 줄은 곡면에 대어 본 것이라 시각 구간 안에서 가장 잘 맞는 시각을 골라 준 값이다 (시각 오류 줄의 잔차는 작게 나온다) — 하한 쪽 폭이다."""
    from scipy import stats
    v = np.array([r["eL"] for r in P if r["disc"] and r.get("eL") is not None], float)
    out = {"n": int(len(v))}
    if len(v) < 10:
        return out
    a = np.abs(v)
    out["abs_eL_log10_quantiles_25_50_75_90_95"] = [r4(x) for x in np.percentile(a, [25, 50, 75, 90, 95])]
    out["frac_exact_fit_abs_eL_lt_0.001"] = r4(float((a < 1e-3).mean()))
    out["frac_abs_eL_gt_0.1"] = r4(float((a > 0.1).mean()))
    out["frac_abs_eL_gt_0.3"] = r4(float((a > 0.3).mean()))
    out["frac_abs_eL_gt_1"] = r4(float((a > 1.0).mean()))
    out["frac_positive_among_nonzero"] = r4(float((v[a >= 1e-3] > 0).mean()))
    w = v[a >= 1e-3]
    out["fit_sample"] = {"n_nonzero": int(len(w)), "note": "시각 구간 안에서 곡면과 정확히 맞는 줄(잔차 0)은 빼고 맞춘다. 위치 0 고정. 정확히 맞는 줄은 곡면으로 못 본다 (시각 오류·같은 사람 다른 화면)."}
    fits = {}
    for name, dist in (("normal", stats.norm), ("laplace", stats.laplace), ("student_t", stats.t), ("cauchy", stats.cauchy)):
        try:
            par = dist.fit(w, floc=0)
            ll = float(dist.logpdf(w, *par).sum())
            k = len(par) - 1
            fits[name] = {"params(shape..., loc, scale)": [r4(x) for x in par], "loglik": r4(ll), "aic": r4(2 * k - 2 * ll)}
        except Exception as ex:  # noqa: BLE001
            fits[name] = {"error": str(ex)[:60]}
    out["family_fits"] = fits
    ok = {k: x["aic"] for k, x in fits.items() if "aic" in x}
    out["best_by_aic"] = min(ok, key=ok.get) if ok else None
    qs = np.percentile(np.abs(w), [50, 90]) if len(w) else [None, None]
    out["nonzero_abs_eL_q50_q90"] = [r4(x) for x in qs]
    return out


def write_md(res, path):
    d1, e1 = res["1d"], res["1e"]
    R, B = d1["rates"], d1["by"]
    pc = lambda x: "-" if x is None else f"{x * 100:.2f}%"
    w = lambda r: f"{pc(r['p'])} [{pc(r['wilson'][0])}, {pc(r['wilson'][1])}] (k={r['k']}, n={r['n']})"
    rc = d1["recommend"]
    cw = rc["contamination_weight_photo"]
    z = d1["surface_z"]
    cov = d1["review_coverage"]["rows"]
    hb = d1["hetero"]["photo_disc"]
    L = []
    L.append("# WP1 1d 떨어진 오류 비율 · 1e 서버 착오율 (2026-10-28, 자료 판 D0, S19~S42)")
    L.append("")
    L.append("코드 `tools/solo_raid/lab7/err_1de.py` · JSON `1de.json`. 정답(answer_key·truth)은 읽지 않았다. 글 번호·사진 이름·점검 사유 원문은 줄을 잇는 데만 쓰고 출력하지 않았다.")
    L.append("")
    L.append("## 1d 오염 혼합의 사전")
    ln = d1["link"]
    L.append(f"- **잇기.** discard.json 항목 {ln['items']}개(S20은 파일 없음)를 (img,row)·사진·글 번호로 obs.jsonl 줄에 이어 관측 표 key p{{oi}}로 옮겼다. 못 이은 항목 {ln['unlinked_items']}개. "
             f"표의 exclude `discard` 사진 줄 {ln['table_photo_rows_excluded_discard']}개 = 직접 이은 {ln['photo_rows_by_direct_link']} + 같은 사진 사본 {ln['copies_via_same_image']}. 서버 고침 사진 줄 {ln['table_photo_rows_server_fix']}.")
    L.append(f"- **분모.** 점검 전 기본 집합 후보 = %가 있고 자동 제외가 없거나 `discard`뿐인 사진 줄 n={R['photo_pool_disc']['n']}. 점검은 후보(전체 풀의 {cov['coverage_of_pool'] * 100:.1f}%)와 세션이 읽다 짚은 글만 원본을 봤다 → **모든 비율은 하한**.")
    L.append(f"- **값.** 버려진 줄 {w(R['photo_pool_disc'])}, 글((시즌,글) 묶음) 부트스트랩 {pc(R['photo_pool_disc']['boot_post']['ci'][0])}~{pc(R['photo_pool_disc']['boot_post']['ci'][1])}. "
             f"서버만 고친 줄 {w(R['photo_pool_fix'])}. 종류 A {pc(B['kind']['A']['p'])} · B {pc(B['kind']['B']['p'])}(n={B['kind']['B']['n']}) · C {pc(B['kind']['C']['p'])} — 구간이 겹친다. text는 글째 버림 {R['text_disc']['k']}/{R['text_disc']['n']}뿐이라 정보 없음.")
    cc = B["cause_share_of_pool"]
    L.append(f"- **원인(경험칙 분류).** 시각 {pc(cc['time']['p'])} · 지난 시즌/모음/편집 화면 {pc(cc['wrongscreen']['p'])} · 사본 {pc(cc['copy']['p'])} · 서버 불신 {pc(cc['server']['p'])} · 판독 {pc(cc['read']['p'])} · 기타 {pc(cc['other']['p'])}. "
             f"시각 오류 줄의 하위 종류 {R['time_error_subtypes_rows']}.")
    L.append(f"- **누가 많이 걸리나.** screen season_end {pc(B['screen']['season_end']['p'])}(n={B['screen']['season_end']['n']}) · rank_list {pc(B['screen']['rank_list']['p'])} · result {pc(B['screen']['result']['p'])}; "
             f"site blabla {pc(B['site']['blabla']['p'])}(n={B['site']['blabla']['n']}) · dc {pc(B['site']['dc']['p'])} · naver {pc(B['site']['naver']['p'])}; 서버 근거 top7 {pc(B['server_basis']['top7']['p'])} · text {pc(B['server_basis']['text']['p'])} · top7_post {pc(B['server_basis']['top7_post(+text)']['p'])}(n={B['server_basis']['top7_post(+text)']['n']}) · 미상 {pc(B['server_basis']['미상']['p'])}.")
    L.append(f"- **하한 보정의 근거.** 후보 안 {pc(cov['disc_inside_candidates']['p'])} 대 밖 {pc(cov['disc_outside_candidates']['p'])}. 버려진 줄의 {R['second_pass_share_of_disc_rows'] * 100:.0f}%({R['photo_pool_disc_second_pass_only']['k']}줄, 대부분 시각 오류)는 "
             f"첫 점검이 아니라 둘째 걸이(모형 잔차로 고른 칸, 10-18~)가 찾았다. 확인된 오류 줄의 곡면 이탈 |z|>3.5는 {z['disc_all']['frac_gt_3.5'] * 100:.0f}%뿐(n={z['disc_all']['n']}), 중앙 |z|={z['disc_all']['abs_z_q10_25_50_75_90'][2]}, 90분위 {z['disc_all']['abs_z_q10_25_50_75_90'][4]}.")
    cwid = d1["contaminant_width"]
    L.append(f"- **엉뚱함의 폭.** 두 성분: 곡면과 구간 안에서 정확히 맞는 줄 {cwid['frac_exact_fit_abs_eL_lt_0.001'] * 100:.0f}%(눈에 안 띄는 오류: 시각·같은 사람 다른 화면) + 나머지는 log10 % 잔차 중앙 {cwid['nonzero_abs_eL_q50_q90'][0]}, 90분위 {cwid['nonzero_abs_eL_q50_q90'][1]} "
             f"(AIC 최선 {cwid['best_by_aic']}: {cwid['family_fits'][cwid['best_by_aic']]['params(shape..., loc, scale)']}, 정규 AIC {cwid['family_fits']['normal']['aic']}). `correct` 값이 남은 {d1['original_vs_corrected']['pct']['n']}줄 중 %가 맞고(로그비 0) 점수·귀속이 틀린 것 {sum(1 for x in d1['original_vs_corrected']['pct']['log10_ratio_sorted'] if x == 0)}줄, 나머지 %는 틀리게 읽힌 폭이 log10 {', '.join(str(x) for x in d1['original_vs_corrected']['pct']['log10_ratio_sorted'] if x != 0)}.")
    ts = d1["time_shift_hours"]
    L.append(f"- **시각 오류의 크기.** 점검이 실제 촬영 구간을 적은 {ts['n']}줄에서 표 시각이 실제보다 늦은 시간: 분위 10·25·50·75·90 = {ts['shift_h_quantiles_10_25_50_75_90']} (6시간 초과 {ts['frac_shift_gt_6h'] * 100:.0f}%). 기기 시계 시간대 어긋남은 +{R['time_error_tz_offset_hours_items']}시간 두 건. 즉 시각 오류는 「조금」이 아니라 하루 안팎의 큰 어긋남이다.")
    L.append(f"- **시즌 변동.** 시즌별 0~{max(v['p'] or 0 for v in B['season'].values()) * 100:.1f}%, 카이제곱 p<0.0001 (이질적), 베타-이항 a={hb['betabinom']['a']}, b={hb['betabinom']['b']} (a+b={hb['betabinom']['conc_a_plus_b']}). "
             f"늘어나는 창(앞 시즌 합친 비율)은 S22 {pc(rc['expanding_window_tune_seasons']['S22']['prior_p'])}에서 S26 {pc(rc['expanding_window_tune_seasons']['S26']['prior_p'])} · S38 {pc(rc['expanding_window_tune_seasons']['S38']['prior_p'])} · S42 {pc(rc['expanding_window_tune_seasons']['S42']['prior_p'])}로 수렴.")
    L.append(f"- **권하는 사전.** 사진 줄 오염 가중치 π ~ Beta: 하한 Beta({cw['prior_lower_bound']['a']}, {cw['prior_lower_bound']['b']}) (평균 {pc(cw['prior_lower_bound']['mean'])}) · "
             f"**중심 Beta({cw['prior_central']['a']}, {cw['prior_central']['b']}) (평균 {pc(cw['prior_central']['mean'])}, 보정 ×{cw['coverage_factor']['chosen_central']})** · 높음 Beta({cw['prior_high']['a']}, {cw['prior_high']['b']}). A·B·C 같은 사전. 오염 성분 꼴은 log % 라플라스(위치 0, 척도 ≈ {cwid['family_fits']['laplace']['params(shape..., loc, scale)'][1]}) 안팎 + 눈에 안 띄는 몫.")
    L.append("")
    L.append("## 1e 서버 착오율")
    pb = e1["pool_rows"]
    L.append(f"- **점검 전 서버**는 서버 고침 항목을 뺀 채 운영 서버 분류기를 임시 사본에서 다시 돌려 얻었다(고침 있는 채로 돌리면 repo의 server.jsonl과 24시즌 모두 바이트까지 같음을 확인). 착오 = 점검 전 라벨이 있었는데 다른 서버로 고쳐진 줄.")
    L.append(f"- **풀 줄 전체 {w(pb['all_labeled'])}**; 서버 근거별(글 단위 k/n): " + " · ".join(f"{b} {pc(v['units_post']['p'])}({v['units_post']['k']}/{v['units_post']['n']})" for b, v in pb["basis"].items()) + ".")
    L.append(f"  사진 줄 기준은 top7 {pc(pb['basis']['top7']['p'])}, text {pc(pb['basis']['text']['p'])}, top7_1 {pc(pb['basis']['top7_1']['p'])}. 이동: " + ", ".join(f"{k}:{v}" for k, v in sorted(pb["transitions"].items(), key=lambda kv: -kv[1])[:5]) + " (풀 밖 순위표 줄까지 합치면 한섭→글섭·북미 쪽이 크다).")
    L.append(f"  서버별 라벨 {', '.join(f'{k} {pc(v['p'])}' for k, v in pb['pre_server'].items())}. 시즌 변동: 카이제곱 p<0.0001, 베타-이항 a+b={pb['hetero']['betabinom']['conc_a_plus_b']}. 글 라벨(claims) 착오 {w(e1['claim_text_labels']['all_labeled'])}.")
    mh = e1["mono_hidden_label_test"]["by_T"]
    L.append(f"- **mono 거르기 재현(분석 4.7 줄 단위 시험, 표 칸으로 다시).** 일섭 라벨을 가리면 빠지는 비율 T48 {pc(mh['48']['JP']['frac_dropped'])} · 끝 {pc(mh['end']['JP']['frac_dropped'])}, 글섭 {pc(mh['48']['GL']['frac_dropped'])} · {pc(mh['end']['GL']['frac_dropped'])} "
             f"(문서 4.7: 일섭 30·35·50·64%, 글섭 25·33·37·61% — T24 {pc(mh['24']['JP']['frac_dropped'])}·T96 {pc(mh['96']['JP']['frac_dropped'])}와 글섭 끝은 조금 다름), 한섭 잘못 빠짐 {pc(mh['end']['KR']['frac_dropped'])}, 미상 {pc(mh['end']['UNK']['frac_dropped'])}. 즉 끝에도 일섭의 {100 - mh['end']['JP']['frac_dropped'] * 100:.0f}%, 글섭의 {100 - mh['end']['GL']['frac_dropped'] * 100:.0f}%가 걸리지 않고 통과.")
    imp = e1["recommend"]["nonKR_among_mono_passed_unknown"]
    L.append(f"- **mono 통과 미상 줄**(use {e1['unknown_rows']['pooled']['mono_kept_rows']}줄, 빠진 {e1['unknown_rows']['pooled']['mono_dropped_rows']}줄) 중 비한섭: 사이트별 EM 사전 x mono 통과율로 {pc(imp['pooled_implied'])}(시즌 {pc(imp['season_range'][0])}~{pc(imp['season_range'][1])}), "
             f"post_server_mix 기대값 {pc(imp['post_server_mix_expected'])}. 두 추정이 3~5% 안에서 일치. 사이트별 섞임(mono 전): " + " · ".join(f"{s} 한섭 {pc(v['mean']['KR'])}(sd {pc(v['sd']['KR'])})" for s, v in e1['unknown_rows']['site_prior_em_dirichlet_moments'].items() if v and s != '*') + ".")
    cp = e1["unknown_rows"]["cv_pooled_calibration_post_design"]
    L.append(f"- **서버 확률 보정(server_mix --cv, 요약이 있는 {len(e1['unknown_rows']['cv_by_season'])}시즌 — {','.join(e1['unknown_rows']['cv_missing_seasons'])}는 적합 줄이 모자라 --cv가 멈춤, 라벨 줄 교차검증).** 신뢰 0.99+ 적중 {pc(cp[-1]['hit'])}(n={cp[-1]['n']}), 0.95~0.99 {pc(cp[-2]['hit'])}(n={cp[-2]['n']}), 0.8~0.95 {pc(cp[-3]['hit'])}(n={cp[-3]['n']}, 신뢰 평균 {pc(cp[-3]['mean_conf'])} → 과신). "
             f"unit 단위 정확도 시즌별은 JSON `cv_by_season`.")
    rl = e1["recommend"]["label_error_by_basis"]
    L.append("- **권하는 사전.** 라벨 착오 Beta(글 단위 하한, Jeffreys): " + " · ".join(f"{b} ({v['beta_lower_bound_jeffreys']['a']}, {v['beta_lower_bound_jeffreys']['b']})" for b, v in rl.items())
             + f"; 보정 x{cw['coverage_factor']['chosen_central']}·농도 ≤100의 중심 평균은 " + " · ".join(f"{b} {pc(v['beta_central']['mean'])}" for b, v in rl.items()) + ". "
             f"미상 줄 사이트별 디리클레는 JSON `unknown_row_site_dirichlet`(dc α0≈{e1['recommend']['unknown_row_site_dirichlet']['dc']['alpha0_moment']:.0f}, naver·blabla α0≈16: 시즌마다 크게 흔들려 늘어나는 창 평균의 sd가 큼).")
    ew = e1["recommend"]["expanding_window_label_error_tune_seasons"]
    wn = e1["recommend"]["expanding_window_unknown_nonKR_share"]
    L.append("- **늘어나는 창(맞힐 시즌 s 앞 시즌들만).** 라벨 착오율(풀 줄) 앞 시즌 합친 값 → 그 시즌: " + " · ".join(f"S{s} {pc(ew[f'S{s}']['prior_p'])}→{pc(ew[f'S{s}']['season_p'])}" for s in TUNE_SEASONS) +
             ". 미상 줄의 비한섭 몫(dc, 앞 시즌 평균±sd → 그 시즌): " + " · ".join(f"S{s} {pc(wn[f'S{s}']['dc']['prior_mean'])}±{pc(wn[f'S{s}']['dc']['prior_sd'])}→{pc(wn[f'S{s}']['dc']['season_value'])}" for s in TUNE_SEASONS) + ".")
    L.append("")
    L.append("## 한계")
    L.append("- 점검은 후보 위주라 모든 비율이 하한이다. 보정 계수(×2, 1~3)는 측정이 아니라 판단이다. 후보 선별(곡면 이탈·정답 점 플래그)이 값에 기대므로 후보 안 비율은 전체 오염율 추정이 아니다.")
    L.append("- 원인 분류(시각·지난 시즌·판독·서버)는 review의 verdict가 있으면 그것, 없으면 사유 글의 말 토큰으로 가른 경험칙이다. 곡면 이탈 z는 버려진 줄을 지금 곡면에 댄 값(시각 구간 안 최적 시각)이라 시각 오류에 낮게 나온다.")
    L.append("- 서버 고침 후보는 cluster_mixed·rank_server·truth_flag·outlier 등 점검 종류에 따라 편향된다. post_server_mix의 CV 보정은 라벨 줄에서 쟀고 미상 줄은 분포가 다르다. S41 외 시즌의 post_server_mix 파일은 repo에 없어 server_mix.py를 임시 위치에 다시 돌렸다(repo 무변경).")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
