"""WP1 1g (2026-10-29) — WP2(공용 도구 고침)로 얻을 것 가늠. 고침 코드는 짜지 않고 줄 수와 크기 순서만 낸다.

  python3 -I tools/solo_raid/lab7/err_1g.py --obs DIR/full --out 1g.json [--md 1g.md]

입력: 관측 표 끝난 시즌 전체(S19~S42, 자료 판 D0) + 시즌 원자료 state의 obs·manifest·posts_min·ocr·claims·post_time_claims
(원본 HTML·이미지는 열지 않는다). 정답(answer_key·truth)은 읽지 않는다.

센 단위: **모형이 보는 줄** = 표의 rep 칸(load_obs 출력의 대표 줄) 가운데 사진. 그 안에서
  C 대표 줄 = 고침이 바꿀 수 있는 줄, 서버 집단은 한섭 라벨(KR) · 미상(mono 거르기가 뺀 줄은 따로) · 그 밖.
  초반 = 쓰는 시각 he < 48 (6시간 칸), 목표 % 근처 = % 가 1·3·10 의 ±30%.
고침 후보:
  4-3 첨부 이름 ↔ 사진 짝 (C → A): 사진 수와 첨부 수가 달라 이름이 안 붙은 줄(짝 못 맞춤), 이름은 붙었지만 시각 꼴을 못 읽는 줄(꼴 확장).
  4-4 글·사진 속 시각 (C → B): 본문·제목의 시각 말, OCR 갱신 타이머·작업표시줄 시계.
  4-2 순위표 시각 구간: h_lo_top7·h_hi_top7 이 있는 C 줄 (지금 load_obs 는 h 에 안 쓰고 mono 거르기의 span 에만 씀).
  4-1 서버 글자 근거: 글자 앵커만으로 한섭 라벨이 붙은 줄 (서버 오류 고침은 라벨을 빼는 쪽이라 미상 → 라벨은 거의 없다).
  4-5 판독 점검 후보: verify=differ 줄, 같은 사람·같은 점수에서 % 가 시간과 거꾸로 가는 줄.
공개 규칙: 글 번호·원문·닉네임·원글 주소·사진 이름을 출력에 넣지 않는다. 열쇠 (시즌, key) 만.
"""
import argparse
import collections
import datetime as dt
import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import obs_table as O  # noqa: E402

KST = dt.timezone(dt.timedelta(hours=9))
F = "%Y-%m-%d %H:%M:%S"
SEASONS = list(range(19, 43))
SEED = 20261029

# ---- 이름 시각 읽기 (표준 = post_time.name_time/merge_obs.shot_time 과 같은 꼴, 확장 = 이 가늠에서 더한 꼴) ----
P1 = re.compile(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ _\-T]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})")
P2 = re.compile(r"(?<!\d)(17[89]\d{10})(?!\d)")
SECOND = re.compile(r"browser|chrome|gallery|kakao|discord|twitter|instagram|naver|whale|line_|telegram|com\.(?!proximabeta)", re.I)
X_EPOCH13 = re.compile(r"(?<!\d)(1[67]\d{11})(?!\d)")
X_EPOCH10 = re.compile(r"(?<!\d)(1[67]\d{8})(?!\d)")
X_LOOSE = re.compile(r"(20\d{2})[-_. ]+(\d{2})[-_. ]+(\d{2})[ _\-T.]+(\d{2})[-_.:; ]+(\d{2})[-_.:; ]+(\d{2})")
X_MAC = re.compile(r"(20\d{2})-(\d{2})-(\d{2})\s*(오전|오후)\s*(\d{1,2})_(\d{2})_(\d{2})")
X_YY = re.compile(r"(?<!\d)(2[3-6])(\d{2})(\d{2})[_\-]?(\d{2})(\d{2})(\d{2})?(?!\d)")


def _dt(*a):
    try:
        return dt.datetime(*a)
    except ValueError:
        return None


def parse_name(name):
    """이름 → [(시각, 꼴)] . 꼴: std(지금 도구가 읽는 꼴) · ep13·ep10(2026-05 앞 epoch) · loose · mac · yy (확장)."""
    out = []
    if not name or name.startswith("gov-"):
        return out
    m = P1.search(name)
    if m:
        t = _dt(*map(int, m.groups()))
        if t:
            out.append((t, "std"))
    m = P2.search(name)
    if m:
        out.append((dt.datetime.fromtimestamp(int(m.group(1)) / 1000, KST).replace(tzinfo=None), "std"))
    if out:
        return out
    m = X_EPOCH13.search(name)
    if m:
        out.append((dt.datetime.fromtimestamp(int(m.group(1)) / 1000, KST).replace(tzinfo=None), "ep13"))
    m = X_EPOCH10.search(name)
    if m and not out:
        out.append((dt.datetime.fromtimestamp(int(m.group(1)), KST).replace(tzinfo=None), "ep10"))
    m = X_MAC.search(name)
    if m:
        y, mo, d, ap, hh, mi, ss = m.groups()
        hh = int(hh) % 12 + (12 if ap == "오후" else 0)
        t = _dt(int(y), int(mo), int(d), hh, int(mi), int(ss))
        if t:
            out.append((t, "mac"))
    m = X_LOOSE.search(name)
    if m and not any(k == "mac" for _, k in out):
        t = _dt(*map(int, m.groups()))
        if t:
            out.append((t, "loose"))
    if not out:
        m = X_YY.search(name)
        if m:
            g = m.groups()
            t = _dt(2000 + int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5] or 0))
            if t:
                out.append((t, "yy"))
    return out


def valid_times(names, post_t):
    """shot_time 규칙(게시 +5분 뒤 또는 4일 전보다 이르면 버림)을 지난 이름 시각들. 남의 화면 이름(SECOND)은 뺀다."""
    res = []
    if post_t is None:
        return res
    for n in names:
        if not n or SECOND.search(n):
            continue
        for t, k in parse_name(n):
            if t > post_t + dt.timedelta(minutes=5) or t < post_t - dt.timedelta(days=4):
                continue
            res.append((t, k, n))
    return res


# ---- 통계 도구 ----

def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def r4(x):
    return None if x is None else round(float(x), 4)


def quant(v, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v = sorted(v)
    if not v:
        return {str(q): None for q in qs}
    out = {}
    for q in qs:
        k = (len(v) - 1) * q
        a, b = int(math.floor(k)), int(math.ceil(k))
        out[str(q)] = r4(v[a] + (v[b] - v[a]) * (k - a))
    return out


def boot_ci(groups, stat, B=400, seed=SEED):
    """groups: 재표본 단위(시즌)별 값 목록 리스트. stat(합친 값 목록) 의 95% 구간."""
    rng = random.Random(seed)
    vals = []
    n = len(groups)
    if n == 0:
        return [None, None]
    for _ in range(B):
        pool = []
        for _ in range(n):
            pool += groups[rng.randrange(n)]
        s = stat(pool)
        if s is not None:
            vals.append(s)
    if not vals:
        return [None, None]
    vals.sort()
    return [r4(vals[int(0.025 * len(vals))]), r4(vals[min(len(vals) - 1, int(0.975 * len(vals)))])]


def band(p):
    if p is None:
        return None
    if 0.7 <= p <= 1.3:
        return 1
    if 2.1 <= p <= 3.9:
        return 3
    if 7 <= p <= 13:
        return 10
    return None


def jl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
    return out


def try_jl(path):
    try:
        return jl(path)
    except OSError:
        return []


# ---- 시즌 읽기 ----

TIMEWORD = {
    "clock": re.compile(r"\d{1,2}\s*시(?!간)(?:\s*\d{1,2}\s*분)?|\d{1,2}:\d\d"),
    "ago": re.compile(r"어제|아까|그제|\d+\s*(?:시간|분)\s*전|전날|며칠\s*전|몇\s*시간\s*전|찍은\s*거|캡처한"),
    "now": re.compile(r"지금|방금|현재"),
    "day": re.compile(r"\d\s*일\s*차|첫날|둘째\s*날|셋째\s*날|막날|마감|[월화수목금토일]요일"),
}
CLOCK_ITEM = re.compile(r"^\s*(?:오전|오후)?\s*\d{1,2}:\d{2}(?::\d{2})?\s*$")
DATE_ITEM = re.compile(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[./]\d{1,2}[./]\d{2,4}")


def load_season(n, objdir, a):
    s = f"S{n}"
    head, rows = O.read_cell(os.path.join(objdir, f"{s}.jsonl.gz"))
    st = C.state_of(s)
    c0 = dt.datetime.strptime(head["c0"], F)
    ob = jl(os.path.join(st, "obs.jsonl"))
    man = {}
    byno = collections.defaultdict(dict)
    for m in jl(os.path.join(st, "manifest.jsonl")):
        man[m["id"]] = m
        byno[str(m["no"])][m["i"]] = m
    posts = {}
    pfiles = [os.path.join(st, "posts_min.jsonl")]
    if s == "S41":   # 디시 글이 state 에 없어 재수집 posts_min(제목·본문·이미지별 이름 시각)을 준다
        pfiles.append(os.path.join(C.ROOT, "experiments", "2026-10-15-s41-opus-blind", "data", "posts_min.jsonl"))
    for f in pfiles:
        for p in try_jl(f):
            k = str(p["no"])
            d = posts.setdefault(k, {})
            for kk, vv in p.items():
                if vv not in (None, "") or kk not in d:
                    d.setdefault(kk, vv)
                    if vv not in (None, "", []):
                        d[kk] = vv
    return {"s": s, "n": n, "head": head, "rows": rows, "st": st, "c0": c0, "obs": ob, "man": man, "byno": byno, "posts": posts}


def server_group(r):
    if r["server"] == "KR":
        return "KR"
    if r["server"] is None:
        return "unk_drop" if r["mono_drop"] else "unk"
    return "other"


def post_dt(o):
    return dt.datetime.strptime(o["post_t"], F) if o.get("post_t") else None


def tag_rows(S):
    """사진 줄마다 고침 후보 표시를 붙인다 (이 시즌의 모든 사진 줄, 대표가 아니어도)."""
    rows, ob, byno, posts, c0 = S["rows"], S["obs"], S["byno"], S["posts"], S["c0"]
    for r in rows:
        r["g"] = server_group(r)
        r["band"] = band(r["pct"])
        r["x"] = {}
        if r["kind"] == "text":
            continue
        o = ob[r["oi"]]
        no = str(o["no"])
        idx = int(o["img"].split("_")[1])
        pt = post_dt(o)
        p = posts.get(no) or {}
        mrow = S["byno"].get(no, {})
        m = S["man"].get(o["img"])
        paired = bool(m and m.get("fname"))
        files = [f if isinstance(f, str) else (f.get("name") if isinstance(f, dict) else None) for f in (p.get("files") or [])]
        x = r["x"]
        x["site"] = o["site"]
        x["paired"] = paired
        x["nfiles"] = len(files)
        x["nimg"] = len(mrow)
        x["no"] = no
        x["idx"] = idx
        if r["kind"] != "C":
            continue
        # 4-3 : 이 사진에 붙일 수 있는 이름 시각
        if paired:
            names = [m["fname"]]
        else:
            names = files
        vt = valid_times(names, pt)
        if S["s"] == "S41" and o["site"] == "dc":   # S41 디시: 첨부 이름 목록이 없고 이미지별 이름 시각만 있다 (재수집 posts_min.shot_t)
            st41 = p.get("shot_t") or []
            x["nfiles"] = len(st41)
            if idx < len(st41) and st41[idx] and pt is not None:
                t41 = dt.datetime.strptime(st41[idx], F)
                if pt - dt.timedelta(days=4) <= t41 <= pt + dt.timedelta(minutes=5) and not x["paired"]:
                    vt = [(t41, "std", "")]
        x["names_n"] = len([n for n in names if n])
        x["vt"] = [(t, k) for t, k, _ in vt]
        x["second"] = any(n and SECOND.search(n) for n in names)
        # 짝 규칙(확장자 묶음 안에서 수가 같으면 차례대로)
        x["ext_pair"] = None
        if not paired and files and mrow:
            cls = lambda e: {"jpeg": "jpg"}.get(e, e)
            fx = collections.defaultdict(list)
            for fi, fn in enumerate(files):
                e = cls((fn or "").rsplit(".", 1)[-1].lower())
                fx[e].append(fi)
            ix = collections.defaultdict(list)
            for ii in sorted(mrow):
                e = cls((mrow[ii].get("ext") or "").lower())
                ix[e].append(ii)
            me = cls((mrow.get(idx, {}).get("ext") or "").lower())
            if me in fx and len(fx[me]) == len(ix.get(me, [])) and idx in ix[me]:
                fi = fx[me][ix[me].index(idx)]
                x["ext_pair"] = (fi, files[fi])
        # 4-4 : 본문·제목의 시각 말
        txt = ((p.get("title") or "") + " " + (p.get("body") or ""))
        x["tw"] = [k for k, rx in TIMEWORD.items() if rx.search(txt)]
        x["has_text"] = bool(txt.strip())
    return S


TIMER_STRICT = re.compile(r"갱신\s*시간\s*(?:(\d+)\s*시간)?\s*(?:(\d+)\s*분)?\s*남")   # post_time.py TIMER 와 같음
TIMER_LOOSE = re.compile(r"갱신\s*시간?\s*[→>\-–:~=\s]*\s*(?:(\d+)\s*시간)?\s*(?:(\d+)\s*분)?\s*남")
TIMER_REM = re.compile(r"^\s*(?:(\d+)\s*시간)?\s*(?:(\d+)\s*분)?\s*남")


def timer_time(rem, pt):
    """post_time.timer_time 과 같다: 다음 05:00 − 남은 시간 가운데 게시+2분 이전 가장 늦은 것."""
    best = None
    for k in range(-1, 5):
        d = pt.date() - dt.timedelta(days=k)
        t = dt.datetime.combine(d, dt.time(5)) - rem
        if t <= pt + dt.timedelta(minutes=2) and (best is None or t > best):
            best = t
    return best


def ocr_flags(S, ids):
    """OCR 항목에서 갱신 타이머 조각·작업표시줄 시계 꼴을 찾는다 (필요한 사진 id 만).
    반환 {id: dict(refresh: 「갱신」 글자 있음, strict: 지금 규칙이 읽는 타이머, loose: 화살표·갈라진 줄까지 읽는 타이머(분), clock: 오른쪽 아래 시계 꼴)}."""
    want = set(ids)
    items_by = collections.defaultdict(list)
    for f in ("ocr.jsonl", "ocr2.jsonl"):
        for r in try_jl(os.path.join(S["st"], f)):
            if r["id"] in want:
                items_by[r["id"]].append((r.get("w") or 1, r.get("h") or 1, r.get("items") or []))
    res = {}
    for iid, lst in items_by.items():
        refresh = strict = clock = False
        loose = None
        for w, h, items in lst:
            txt = [it[1] for it in items if len(it) > 1]
            refresh = refresh or any("갱신" in t for t in txt)
            for i, t in enumerate(txt):
                m = TIMER_STRICT.search(t)
                if m and (m.group(1) or m.group(2)):
                    strict = True
                m = TIMER_LOOSE.search(t)
                if not (m and (m.group(1) or m.group(2))) and "갱신" in t and i + 1 < len(txt):
                    m = TIMER_REM.search(txt[i + 1])
                if m and (m.group(1) or m.group(2)) and loose is None:
                    loose = int(m.group(1) or 0) * 60 + int(m.group(2) or 0)
            for it in items:
                if len(it) < 2:
                    continue
                (x0, y0, x1, y1), t = it[0], it[1]
                if y0 > 0.9 * h and x0 > 0.55 * w and (CLOCK_ITEM.match(t) or DATE_ITEM.search(t)):
                    clock = True
        res[iid] = {"refresh": refresh, "strict": strict, "loose": loose, "clock": clock}
    return res


def summ(L):
    """줄 목록 → 세는 값."""
    def cnt(f):
        return sum(1 for r in L if f(r))
    early = lambda r: r["he"] is not None and r["he"] < 48
    nt = lambda r: r["band"] is not None
    out = {"n": len(L), "KR": cnt(lambda r: r["g"] == "KR"), "unk": cnt(lambda r: r["g"] == "unk"),
           "unk_drop": cnt(lambda r: r["g"] == "unk_drop"), "other": cnt(lambda r: r["g"] == "other")}
    for g in ("KR", "unk"):
        out[f"{g}_early48"] = cnt(lambda r: r["g"] == g and early(r))
        out[f"{g}_early24"] = cnt(lambda r: r["g"] == g and r["he"] is not None and r["he"] < 24)
        out[f"{g}_target"] = cnt(lambda r: r["g"] == g and nt(r))
        out[f"{g}_early48_target"] = cnt(lambda r: r["g"] == g and early(r) and nt(r))
        out[f"{g}_early24_target"] = cnt(lambda r: r["g"] == g and r["he"] is not None and r["he"] < 24 and nt(r))
        for b in (1, 3, 10):
            out[f"{g}_early48_b{b}"] = cnt(lambda r: r["g"] == g and early(r) and r["band"] == b)
    return out


def bins6(L, grp="KR", target=False):
    c = collections.Counter()
    for r in L:
        if r["g"] != grp or r["he"] is None or r["he"] >= 120:
            continue
        if target and r["band"] is None:
            continue
        c[int(r["he"] // 6)] += 1
    return {f"T{6 * (k + 1)}": c.get(k, 0) for k in range(0, 8)}


def hrs(t, c0):
    return (t - c0).total_seconds() / 3600


# ---- 고침 후보 줄 만들기 ----

def collect(seasons):
    """모형이 보는 줄(rep) 가운데 C 사진마다 고침 후보 표시를 붙여 돌려준다. 반환 {고침 이름: [줄]}."""
    fx = collections.defaultdict(list)
    for n, S in seasons.items():
        c0 = S["c0"]
        # OCR 조각은 C 대표 줄의 사진만 읽는다
        ids = [S["obs"][r["oi"]]["img"] for r in S["rows"] if r["kind"] == "C" and r["rep"]]
        ocr = ocr_flags(S, ids)
        for r in S["rows"]:
            if r["kind"] != "C" or not r["rep"]:
                continue
            x = r["x"]
            r["season"] = n
            vt = x.get("vt") or []
            kinds = {k for _, k in vt}
            hs = [hrs(t, c0) for t, _ in vt]
            r["new_h"] = (min(hs) if hs else None)
            r["new_spread"] = (max(hs) - min(hs) if hs else None)
            r["shift"] = (r["he"] - min(hs)) if hs else None
            # --- 4-3 ---
            if not x["paired"] and "std" in kinds:
                r["f43_pair"] = True
                tight = r["new_spread"] <= 10 / 60
                ep = x.get("ext_pair")
                one = False
                if ep and not tight:
                    pt = post_dt(S["obs"][r["oi"]])
                    one = bool(valid_times([ep[1]], pt))
                r["f43_tier"] = "tight" if tight else ("extpair" if one else "ambiguous")
                if one and ep:
                    pt = post_dt(S["obs"][r["oi"]])
                    vt1 = valid_times([ep[1]], pt)
                    r["new_h"] = hrs(vt1[0][0], c0)
                    r["shift"] = r["he"] - r["new_h"]
                fx["4-3 짝 못 맞춤(이름 시각 있음)"].append(r)
            elif vt and "std" not in kinds:
                fx["4-3 이름 꼴 확장(ep13·loose·mac·yy)"].append(r)
                r["f43_ext"] = sorted(kinds)
            # --- 4-4 ---
            tw = x.get("tw") or []
            r["tw"] = tw
            if tw:
                fx["4-4 본문 시각 말(전체)"].append(r)
                for k in tw:
                    fx[f"4-4 본문 시각 말: {k}"].append(r)
            o = S["obs"][r["oi"]]
            of = ocr.get(o["img"])
            if of and of["refresh"]:
                fx["4-4 OCR 갱신 글자 있음"].append(r)
            if of and of["loose"] is not None and not of["strict"]:
                pt = post_dt(o)
                t = timer_time(dt.timedelta(minutes=of["loose"]), pt) if pt else None
                if t is not None:
                    r["timer_h"] = hrs(t, c0)
                    r["timer_shift"] = r["he"] - r["timer_h"]
                    fx["4-4 OCR 갱신 타이머(화살표·갈라진 줄)"].append(r)
            if of and of["clock"]:
                fx["4-4 OCR 작업표시줄 시계 꼴"].append(r)
            # --- 4-2 ---
            if r["h_lo_top7"] is not None and r["h_hi_top7"] is not None:
                fx["4-2 순위표 시각 구간 있음"].append(r)
    return fx


def tab(rows):
    return summ(rows)


def delay_samples(seasons):
    """A 사진(기계 시각)의 게시 지연 (hp − h, 시간) 과 쓰는 시각. 종류별·시각 칸별."""
    out = []
    for n, S in seasons.items():
        for r in S["rows"]:
            if r["kind"] != "A" or not r["use"] or r["h"] is None or r["hp"] is None or r["h"] < 0:
                continue
            out.append((n, r["h"], r["hp"] - r["h"], r["g"], r["rep"]))
    return out


WINS = [(0, 12), (12, 24), (24, 48), (48, 72), (72, 96), (96, 125)]


def gamma_fit(pts, lo, hi, pmin=0.3, pmax=20.0):
    """같은 시즌 안(시즌 고정효과)에서 ln 점수 ~ β ln % + γ h 를 시각 창 [lo,hi) 에서 맞춘다. 반환 (β, γ, 잔차 sd, n, 시즌 수)."""
    import numpy as np
    by = collections.defaultdict(list)
    for n, h, p, s in pts:
        if lo <= h < hi and pmin <= p <= pmax and s > 0:
            by[n].append((math.log(p), h, math.log(s)))
    X, Y = [], []
    ns = 0
    for n, L in by.items():
        if len(L) < 2:
            continue
        arr = np.array(L)
        arr = arr - arr.mean(0)
        X.append(arr[:, :2])
        Y.append(arr[:, 2])
        ns += 1
    if not X:
        return None
    X, Y = np.vstack(X), np.concatenate(Y)
    if len(Y) < 6:
        return None
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    res = Y - X @ coef
    return float(coef[0]), float(coef[1]), float(res.std(ddof=3)), int(len(Y)), ns


def gamma_table(seasons, B=300):
    pts = []
    for n, S in seasons.items():
        for r in S["rows"]:
            if r["kind"] in ("A", "B") and r["rep"] and r["g"] == "KR" and r["score"] and r["pct"] and r["pct"] > 0 and r["he"] is not None:
                pts.append((n, r["he"], r["pct"], r["score"]))
    out = {}
    rng = random.Random(SEED)
    sids = sorted(set(p[0] for p in pts))
    for lo, hi in WINS:
        f = gamma_fit(pts, lo, hi)
        if f is None:
            out[f"{lo}-{hi}"] = None
            continue
        gs = []
        bys = collections.defaultdict(list)
        for p in pts:
            bys[p[0]].append(p)
        for _ in range(B):
            samp = []
            for k, sid in enumerate(rng.choice(sids) for _ in sids):
                samp += [(f"{sid}#{k}",) + p[1:] for p in bys[sid]]
            g = gamma_fit(samp, lo, hi)
            if g:
                gs.append(g[1])
        gs.sort()
        out[f"{lo}-{hi}"] = {"beta": r4(f[0]), "gamma_per_h": r4(f[1]), "gamma_ci95": [r4(gs[int(0.025 * len(gs))]), r4(gs[int(0.975 * len(gs))])] if gs else None,
                             "resid_sd_ln": r4(f[2]), "n": f[3], "n_seasons": f[4]}
    return out, pts


def win_of(h):
    for lo, hi in WINS:
        if lo <= h < hi:
            return f"{lo}-{hi}"
    return f"{WINS[-1][0]}-{WINS[-1][1]}"


def weight(gam, delays, sig, center=None):
    """지연 표본 → 평균 무게 w = mean 1/(1+(γ(D−D0)/σ)²). D0 = 모형이 맞추는 중심(중앙값)."""
    if not delays:
        return None
    d0 = center if center is not None else sorted(delays)[len(delays) // 2]
    return sum(1.0 / (1.0 + (gam * (d - d0) / sig) ** 2) for d in delays) / len(delays)


def gain_for(rows, gam, delays, sig, new_sigma):
    """줄 목록 → (합 w_after − w_before, 시각 칸별 합). 줄마다 쓰는 시각 창의 γ."""
    tot, byw = 0.0, collections.Counter()
    for r in rows:
        w = win_of(r["he"])
        g = (gam.get(w) or {}).get("gamma_per_h")
        if g is None:
            continue
        g = abs(g)
        wb = weight(g, delays[w], sig)
        if wb is None:
            continue
        wa = 1.0 / (1.0 + (g * new_sigma / sig) ** 2) if new_sigma > 0 else 1.0
        gdiff = max(0.0, wa - wb)
        tot += gdiff
        byw[w] += gdiff
    return tot, {k: r4(v) for k, v in byw.items()}


def verify_reverse(seasons):
    """4-5: verify=differ 인 대표 줄, 같은 사람·같은 점수에서 %가 시간과 거꾸로 가는 줄(시각을 확실히 아는 쪽끼리 + 확실히 늦은 C)."""
    res = {"verify_differ": collections.Counter(), "reverse_rows": collections.Counter(), "reverse_pairs": 0}
    ex = []
    for n, S in seasons.items():
        by = collections.defaultdict(list)
        for r in S["rows"]:
            if r["kind"] == "text":
                continue
            if r["rep"] and r.get("verify") == "differ":
                res["verify_differ"][(r["kind"], r["g"])] += 1
            if r["use"] and r["score"] and r["pct"] is not None and r["pid"]:
                by[(r["pid"], round(r["score"], 3))].append(r)
        for k, L in by.items():
            if len(L) < 2:
                continue
            for a in L:
                for b in L:
                    if a is b:
                        continue
                    a_hi = a["h"] if a["t_grade"] != "C" else (a["h_hi"] if a["h_hi"] is not None else a["h"])
                    b_lo = b["h"] if b["t_grade"] != "C" else (b["h_lo"] if b["h_lo"] is not None else None)
                    if a_hi is None or b_lo is None or b_lo < a_hi + 1e-9:
                        continue
                    if b["pct"] < a["pct"] * 0.85 - 0.011:
                        res["reverse_pairs"] += 1
                        res["reverse_rows"][(b["kind"], b["g"], "rep" if b["rep"] else "notrep")] += 1
                        if len(ex) < 6:
                            ex.append({"season": n, "key_later": b["key"], "key_earlier": a["key"], "pct_earlier": a["pct"], "pct_later": b["pct"]})
    return res, ex


def server_basis_table(seasons):
    c = collections.Counter()
    fixrows = collections.Counter()
    for n, S in seasons.items():
        for r in S["rows"]:
            if r["kind"] == "text" or not r["rep"]:
                continue
            b = r["server_basis"]
            if r["server"] == "KR":
                c[(b, r["kind"])] += 1
            if b == "fix":
                fixrows[(r["server"], r["kind"])] += 1
    return c, fixrows


def claim_yield(seasons):
    """4-4 수율 가늠: 글 주장(% 있는 것)의 시각 말 종류별로 B 가 된 비율과 게시와의 시각 차(시간)."""
    cls = {k: [0, 0] for k in TIMEWORD}
    shifts = collections.defaultdict(list)
    for n, S in seasons.items():
        cl = try_jl(os.path.join(S["st"], "claims.jsonl"))
        ptc = try_jl(os.path.join(S["st"], "post_time_claims.jsonl"))
        c0 = S["c0"]
        for i, c in enumerate(cl):
            t = c.get("time")
            if not t or i >= len(ptc) or c.get("kind") != "self":
                continue
            kinds = [k for k, rx in TIMEWORD.items() if rx.search(str(t))]
            if not kinds:
                continue
            g = ptc[i]
            isb = g["grade"] == "B"
            for k in kinds:
                cls[k][1] += 1
                cls[k][0] += isb
            if isb and g.get("h") is not None and c.get("post_t"):
                hp = hrs(dt.datetime.strptime(c["post_t"], F), c0)
                b = (g.get("basis") or "").split(" ")[0].split(":")[-1]
                shifts[b].append(hp - g["h"])
    out = {k: {"B": v[0], "n": v[1], "rate": r4(v[0] / v[1]) if v[1] else None, "wilson95": wilson(v[0], v[1])} for k, v in cls.items()}
    sh = {b: {"n": len(v), "quantiles_h": quant(v), "share_gt_1h": r4(sum(1 for x in v if x > 1) / len(v)), "share_gt_6h": r4(sum(1 for x in v if x > 6) / len(v))} for b, v in shifts.items() if len(v) >= 5}
    return out, sh



def timer_validation(seasons):
    """느슨한 타이머 규칙이 읽은 시각을, 기계 시각을 아는 A 사진(같은 규칙으로 읽히는 것)과 견준다 (오차 = 타이머 시각 − 촬영 시각, 시간)."""
    errs, errs_loose_only = [], []
    for n, S in seasons.items():
        ids = {}
        for r in S["rows"]:
            if r["kind"] == "A" and r["rep"] and r["h"] is not None and r["hp"] is not None:
                ids[S["obs"][r["oi"]]["img"]] = r
        fl = ocr_flags(S, list(ids))
        for iid, of in fl.items():
            if of["loose"] is None:
                continue
            r = ids[iid]
            pt = post_dt(S["obs"][r["oi"]])
            t = timer_time(dt.timedelta(minutes=of["loose"]), pt) if pt else None
            if t is None:
                continue
            e = hrs(t, S["c0"]) - r["h"]
            errs.append(e)
            if not of["strict"]:
                errs_loose_only.append(e)
    def blk(v):
        if not v:
            return {"n": 0}
        ab = [abs(x) for x in v]
        k2 = sum(1 for x in ab if x <= 2 / 60 + 1e-9)
        k10 = sum(1 for x in ab if x <= 10 / 60)
        return {"n": len(v), "within_2min": r4(k2 / len(v)), "within_2min_wilson95": wilson(k2, len(v)), "within_10min": r4(k10 / len(v)),
                "abs_err_quantiles_h": quant(ab, (0.5, 0.9, 0.95))}
    return {"all_loose": blk(errs), "loose_only_(strict_misses)": blk(errs_loose_only)}


def share_block(cand, base_by_season, seasons_idx):
    """시즌별 후보 비율 + 이질성(카이제곱) + 늘어나는 창(앞 시즌들만으로 잰 비율)이 시즌마다 흔들리는 정도."""
    from scipy import stats
    per = {}
    ks, ns = [], []
    for n in seasons_idx:
        k = sum(1 for r in cand if r["season"] == n)
        b = base_by_season.get(n, 0)
        per[f"S{n}"] = {"k": k, "n": b, "share": r4(k / b) if b else None}
        if b:
            ks.append(k)
            ns.append(b)
    K, N = sum(ks), sum(ns)
    chi = None
    if K and len(ks) > 1:
        exp = [N and K * b / N for b in ns]
        chi2 = sum((k - e) ** 2 / e for k, e in zip(ks, exp) if e > 0) + sum(((b - k) - (b - e)) ** 2 / (b - e) for k, b, e in zip(ks, ns, exp) if b - e > 0)
        chi = {"chi2": r4(chi2), "df": len(ks) - 1, "p": float(f"{stats.chi2.sf(chi2, len(ks) - 1):.3g}")}
    # 늘어나는 창: 시즌 s 이전 시즌들만의 비율 (S21~S42 맞힐 시즌마다)
    win = {}
    for n in range(21, 43):
        kk = sum(per[f"S{m}"]["k"] for m in range(19, n))
        nn = sum(per[f"S{m}"]["n"] for m in range(19, n))
        win[f"S{n}"] = r4(kk / nn) if nn else None
    wv = [v for v in win.values() if v is not None]
    return {"total": {"k": K, "n": N, "share": r4(K / N) if N else None, "wilson95": wilson(K, N)}, "per_season": per, "heterogeneity": chi,
            "expanding_window_share_range": [min(wv), max(wv)] if wv else None, "expanding_window_share": win}


def shift_block(rows, key="shift"):
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return {"n": 0}
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            groups[r["season"]].append(r[key])
    med_ci = boot_ci(list(groups.values()), lambda p: sorted(p)[len(p) // 2] if p else None)
    return {"n": len(v), "quantiles_h": quant(v), "median_ci95_by_season_boot": med_ci,
            "share_gt_0.25h": r4(sum(1 for x in v if x > 0.25) / len(v)), "share_gt_1h": r4(sum(1 for x in v if x > 1) / len(v)),
            "share_gt_6h": r4(sum(1 for x in v if x > 6) / len(v)), "n_gt_0.25h": sum(1 for x in v if x > 0.25), "n_gt_1h": sum(1 for x in v if x > 1),
            "n_gt_6h": sum(1 for x in v if x > 6)}


def write_md(res, path):
    """사람이 읽는 요약 (한국어). 숫자는 JSON 에서 그대로 가져온다."""
    F = res["fixes"]
    base = res["base"]["rep_photo_rows_by_kind"]
    sup = res["base"]["exact_supply_by_6h_bin_all_seasons"]
    ab_t = sum(sup["A+B 한섭 목표 근처 6시간 칸"].values())
    c_t = sum(sup["C 한섭 목표 근처 6시간 칸"].values())
    u = res["union_of_conversions"]["eq_points"]
    ce = res["ceiling_if_all_C_exact"]
    gm = res["gamma_table_ln_cutoff_per_hour"]
    t43, t44 = F["4-3"], F["4-4"]["OCR 타이머"]
    tv = t44["timer_validation_on_A_photos"]["all_loose"]
    d42 = F["4-2"]
    L = []
    L.append("# 1g — WP2(공용 도구 고침)로 얻을 것 가늠 (자료 판 D0, S19~S42)")
    L.append("")
    L.append(f"**센 단위**: 모형이 보는 줄(rep) 사진. A {base['A']['n']:,} · B {base['B']['n']:,} · C {base['C']['n']:,} (한섭 라벨 {base['C']['KR']:,} · 미상 {base['C']['unk']:,}). "
             f"초반 = 쓰는 시각 he<48, 목표 근처 = pct 1·3·10 의 ±30%. 한섭 초반 목표 근처 정확한 점(A+B)은 24시즌 합 {ab_t}줄(시즌 중앙 {res['base']['median_AB_KR_early48_target_per_season']}), C 는 {c_t}줄.")
    L.append(f"**핵심**: 고침으로 C 가 A·B 가 되는 줄은 많지만(합집합 {res['union_of_conversions']['eq_points']['전체']['n_rows']:,}줄) **이동이 대부분 몇 분**이고, 한섭 초반 목표 근처는 {u['KR 초반48 목표근처']['n_rows']}줄뿐이다. "
             f"시각 모형 아래 환산하면 정확한 점 {u['KR 초반48 목표근처']['지연=C 후보 분포']:.1f}~{u['KR 초반48 목표근처']['지연=A 분포']:.1f}개분(24시즌 합, 기존 {ab_t}개 대비 {100 * u['KR 초반48 목표근처']['지연=C 후보 분포'] / ab_t:.0f}~{100 * u['KR 초반48 목표근처']['지연=A 분포'] / ab_t:.0f}%). "
             f"C 를 전부 정확하게 만들어도 {ce['KR 초반48 목표근처']['지연=C 후보 분포']['eq_points_if_all_C_became_exact']:.1f}~{ce['KR 초반48 목표근처']['지연=A 분포']['eq_points_if_all_C_became_exact']:.1f}개분이 천장이다.")
    L.append("")
    L.append("| 고침 | 바뀌는 줄(전체 / 한섭 / 한섭 초반 목표근처 / 미상 초반 목표근처) | 줄당 이동 | 시각 모형 아래 한섭 초반 목표근처 (A분포~C분포, 정확한 점 개수) | 비용 |")
    L.append("|---|---|---|---|---|")
    for t in res["table"]:
        eq = t.get("eq_exact_points_time_model_KR_early48_target_A_C")
        eqs = f"{eq[1]:.1f}~{eq[0]:.1f}" if eq and eq[0] is not None else "—"
        k = lambda v: "—" if v is None else (f"{v:,.0f}" if isinstance(v, (int, float)) and v == int(v) else f"{v:,.1f}")
        L.append(f"| {t['fix']} | {k(t.get('rows_changed'))} / {k(t.get('KR'))} / {k(t.get('KR_early48_target'))} / {k(t.get('unk_early48_target'))} | "
                 f"{'중앙 ' + format(res['fixes']['4-3']['shift_h']['quantiles_h']['0.5'] * 60, '.1f') + '분' if t['fix'].startswith('4-3') else ('중앙 ' + format(t44['shift_h']['quantiles_h']['0.5'] * 60, '.1f') + '분' if '타이머' in t['fix'] else '')} | {eqs} | {t.get('cost', '')} |")
    L.append("")
    L.append("**시각 → %**: 같은 점수 컷의 시간당 변화(ln 점수) γ = " + ", ".join(f"{w}h {v['gamma_per_h'] * 100:.1f}%/h" for w, v in gm.items() if v) +
             " (시즌 부트스트랩 구간은 JSON; 0~12h 는 30점·8시즌뿐이라 넓다). A 사진 게시 지연 중앙 " + f"{res['delay']['A_all']['quantiles_h']['0.5'] * 60:.1f}분·75% {res['delay']['A_all']['quantiles_h']['0.75'] * 60:.0f}분·90% {res['delay']['A_all']['quantiles_h']['0.9']:.1f}시간 → "
             f"초반 12시간 안에서 5분 늦음 = 컷 {gm['0-12']['gamma_per_h'] * 5 / 60 * 100:.1f}%, 90% 지연이면 {gm['0-12']['gamma_per_h'] * res['delay']['A_by_window']['0-12']['quantiles_h']['0.9'] * 100:.0f}%. 48시간 뒤에는 5분 = {gm['48-72']['gamma_per_h'] * 5 / 60 * 100:.2f}%.")
    L.append(f"**검증**: 느슨한 타이머 규칙은 기계 시각 A {tv['n']}장에서 촬영과 2분 안 {tv['within_2min'] * 100:.0f}% (윌슨 {tv['within_2min_wilson95'][0]:.2f}~{tv['within_2min_wilson95'][1]:.2f}), "
             f"순위표 구간과는 {t44['timer_vs_top7_bracket']['n_inside']}/{t44['timer_vs_top7_bracket']['n_with_bracket']}. 4-3 이름 시각도 순위표 구간 안 {F['4-3a']['top7_consistency']['n_inside']}/{F['4-3a']['top7_consistency']['n_with_bracket']}(짝 못 맞춤)·{F['4-3b']['top7_consistency']['n_inside']}/{F['4-3b']['top7_consistency']['n_with_bracket']}(꼴 확장).")
    L.append(f"**시즌 변동**: 4-3 후보 비율 시즌별 {min(v['share'] for v in t43['share_of_C_rep']['per_season'].values() if v['share'] is not None) * 100:.0f}~{max(v['share'] for v in t43['share_of_C_rep']['per_season'].values() if v['share'] is not None) * 100:.0f}% (카이제곱 p<1e-17 — 2026-05 앞 13자리 epoch 이름 때문에 S19~S38 에 몰림, S39~ 는 이미 A). "
             f"합집합 한섭 초반 목표근처는 시즌당 중앙 {sorted(res['union_KR_early48_target_per_season_sorted'])[12]}줄(24시즌 중 {sum(1 for v in res['union_KR_early48_target_per_season_sorted'] if v == 0)}시즌은 0줄).")
    L.append(f"**따로 얻는 것(도구 고침 아님)**: 순위표 시각 구간(4-2)은 관측 표에 열이 이미 있고 한섭 C {d42['rows']['KR']:,}줄({d42['share_of_C_rep_KR']['share'] * 100:.0f}%)에 붙어 있다. 관측 층이 읽어 지연을 구간 최대 지연으로 자르면 한섭 초반 목표근처 "
             f"{d42['eq_points_if_time_model_truncates_by_bracket']['지연=C 후보 분포']['KR_e48t']:.1f}~{d42['eq_points_if_time_model_truncates_by_bracket']['지연=A 분포']['KR_e48t']:.1f}개분 — 위 모든 도구 고침의 합과 같은 크기 이상이다. 도구로 h 를 내려 주는 줄은 35줄(1시간 넘게 17줄)뿐.")
    L.append("**한계**: ① γ 는 한섭 A·B 점으로 시즌 고정효과 회귀(초반 창 표본 작음) ② 지연 분포는 A 사진 것(C 가 더 길 수 있음)과 이름이 있는 C 후보 것(더 짧음) 두 가지로 가늠 — 어느 쪽이든 순서는 같다 ③ 본문 시각 말 수율은 글 주장 기준 상한 "
             "④ 13자리 epoch 이름은 업로드 시각에 가까울 수 있어(지연 중앙 1.7분 vs 파일 이름 2.6분) 4-3 꼴 확장의 실제 새 정보는 적을 수 있다 ⑤ 4-1·4-5 는 정보량이 아니라 오염·점검이라 환산하지 않았다.")
    L.append("**권하는 순서(결정은 조율 세션)**: " + " → ".join(f"{x['order']}. {x['fix']}" for x in res["recommend_order"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", required=True, help="관측 표 full 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--cache", default=None, help="읽은 시즌을 pickle 로 두는 파일 (되풀이할 때)")
    a = ap.parse_args()
    t0 = __import__("time").time()
    if a.cache and os.path.exists(a.cache):
        import pickle
        seasons = pickle.load(open(a.cache, "rb"))
    else:
        seasons = {n: tag_rows(load_season(n, a.obs, a)) for n in SEASONS}
        if a.cache:
            import pickle
            pickle.dump(seasons, open(a.cache, "wb"))
    fx = collect(seasons)
    idx = SEASONS

    # 0. 기준: 모형이 보는 사진 줄 (rep)
    rep = {k: [] for k in "ABC"}
    for n, S in seasons.items():
        for r in S["rows"]:
            if r["kind"] in rep and r["rep"]:
                r["season"] = n
                rep[r["kind"]].append(r)
    base = {k: summ(v) for k, v in rep.items()}
    base_c_season = {n: sum(1 for r in rep["C"] if r["season"] == n) for n in idx}
    base_c_KR_season = {n: sum(1 for r in rep["C"] if r["season"] == n and r["g"] == "KR") for n in idx}
    exact_supply = {
        "A+B 한섭 6시간 칸(he)": bins6(rep["A"] + rep["B"], "KR"),
        "A+B 한섭 목표 근처 6시간 칸": bins6(rep["A"] + rep["B"], "KR", True),
        "C 한섭 6시간 칸": bins6(rep["C"], "KR"),
        "C 한섭 목표 근처 6시간 칸": bins6(rep["C"], "KR", True),
        "A+B 미상 목표 근처 6시간 칸": bins6(rep["A"] + rep["B"], "unk", True),
        "C 미상 목표 근처 6시간 칸": bins6(rep["C"], "unk", True),
    }
    per_season_supply = {}
    for n in idx:
        L = [r for k in "AB" for r in rep[k] if r["season"] == n]
        Lc = [r for r in rep["C"] if r["season"] == n]
        f = lambda X, g: sum(1 for r in X if r["g"] == g and r["he"] is not None and r["he"] < 48 and r["band"] is not None)
        per_season_supply[f"S{n}"] = {"AB_KR_early48_target": f(L, "KR"), "C_KR_early48_target": f(Lc, "KR"), "AB_KR_all": sum(1 for r in L if r["g"] == "KR"), "C_KR_all": sum(1 for r in Lc if r["g"] == "KR")}
    ab_med = sorted(v["AB_KR_early48_target"] for v in per_season_supply.values())[len(per_season_supply) // 2]
    base_block = {"rep_photo_rows_by_kind": base, "exact_supply_by_6h_bin_all_seasons": exact_supply, "per_season": per_season_supply,
                  "median_AB_KR_early48_target_per_season": ab_med,
                  "note": "6시간 칸 T6 = 쓰는 시각 he 0~6, T12 = 6~12 …. 목표 근처 = pct 가 1·3·10 의 ±30%. he 는 C 에서 게시 시각."}

    # 1. 시각 감: γ(시각 창), 지연 분포
    gam, pts = gamma_table(seasons)
    dl = delay_samples(seasons)
    delays = {}
    for lo, hi in WINS:
        delays[f"{lo}-{hi}"] = [d for _, h, d, g, rp in dl if lo <= h < hi and d >= 0 and g in ("KR", "unk")]
    dq = {k: {"n": len(v), "quantiles_h": quant(v, (0.5, 0.75, 0.9, 0.95))} for k, v in delays.items()}
    allA = [d for _, h, d, g, rp in dl if d >= 0 and g in ("KR", "unk")]
    byseason_delay = collections.defaultdict(list)
    for n, h, d, g, rp in dl:
        if d >= 0 and g in ("KR", "unk"):
            byseason_delay[n].append(d)
    delay_block = {"A_all": {"n": len(allA), "quantiles_h": quant(allA, (0.5, 0.75, 0.9, 0.95)),
                             "median_ci95_by_season_boot": boot_ci(list(byseason_delay.values()), lambda p: sorted(p)[len(p) // 2]),
                             "p90_ci95_by_season_boot": boot_ci(list(byseason_delay.values()), lambda p: sorted(p)[int(len(p) * 0.9)])},
                   "A_by_window": dq,
                   "season_median_range": [min(sorted(v)[len(v) // 2] for v in byseason_delay.values()), max(sorted(v)[len(v) // 2] for v in byseason_delay.values())],
                   "season_p90_range": [min(sorted(v)[int(len(v) * 0.9)] for v in byseason_delay.values()), max(sorted(v)[int(len(v) * 0.9)] for v in byseason_delay.values())]}
    # 4-3 후보의 실제 지연 (C 줄 가운데 이름이 있는 것)
    c43 = fx["4-3 짝 못 맞춤(이름 시각 있음)"] + fx["4-3 이름 꼴 확장(ep13·loose·mac·yy)"]
    cdelay = [r["shift"] for r in c43 if r.get("shift") is not None and r["shift"] >= 0]
    delay_block["C_candidates_4-3"] = {"n": len(cdelay), "quantiles_h": quant(cdelay, (0.5, 0.75, 0.9, 0.95))}
    delays_C = {f"{lo}-{hi}": cdelay for lo, hi in WINS}
    # top7 구간으로 본 C 의 최대 지연 (he − h_lo_top7, 상한)
    br = fx["4-2 순위표 시각 구간 있음"]
    maxd = [r["he"] - r["h_lo_top7"] for r in br if r["he"] is not None]
    delay_block["C_top7_max_delay_upper_bound"] = {"n": len(maxd), "quantiles_h": quant(maxd, (0.1, 0.25, 0.5, 0.75, 0.9))}

    SIG = 0.01
    # 2. 고침별
    out_fx = {}

    def core(rows, label, shift_key="shift", new_sigma=0.0, extra=None, sc=("A", "C")):
        d = {"label": label, "rows": tab(rows), "share_of_C_rep": share_block(rows, base_c_season, idx),
             "share_of_C_rep_KR": share_block([r for r in rows if r["g"] == "KR"], base_c_KR_season, idx)["total"]}
        d["shift_h"] = shift_block(rows, shift_key)
        d["by_6h_bin_KR"] = bins6(rows, "KR")
        d["by_6h_bin_KR_target"] = bins6(rows, "KR", True)
        d["by_6h_bin_unk_target"] = bins6(rows, "unk", True)
        gains = {}
        for nm, dsc in (("지연=A 분포", delays), ("지연=C 후보 분포", delays_C)):
            for tag, subset in (("전체", rows), ("KR", [r for r in rows if r["g"] == "KR"]),
                                ("KR 초반48", [r for r in rows if r["g"] == "KR" and r["he"] is not None and r["he"] < 48]),
                                ("KR 초반48 목표근처", [r for r in rows if r["g"] == "KR" and r["he"] is not None and r["he"] < 48 and r["band"] is not None]),
                                ("KR 초반24 목표근처", [r for r in rows if r["g"] == "KR" and r["he"] is not None and r["he"] < 24 and r["band"] is not None]),
                                ("미상 초반48 목표근처", [r for r in rows if r["g"] == "unk" and r["he"] is not None and r["he"] < 48 and r["band"] is not None])):
                g, bw = gain_for(subset, gam, dsc, SIG, new_sigma)
                gains[f"{nm} | {tag}"] = {"eq_points": r4(g), "n_rows": len(subset)}
        d["eq_exact_points_gain_time_model"] = gains
        # 시각 모형 없이(C 를 게시 시각에 정확한 점으로 둠) 고침이 없애는 편향: 줄마다 γ × 이동량
        bias = []
        for r in rows:
            w = win_of(r["he"]) if r["he"] is not None else None
            g = (gam.get(w) or {}).get("gamma_per_h") if w else None
            sh = r.get(shift_key)
            if g is not None and sh is not None and r["g"] == "KR":
                bias.append(abs(g) * max(sh, 0))
        d["F_style_bias_removed_KR"] = {"n": len(bias), "quantiles_cutoff_rel": quant(bias, (0.5, 0.75, 0.9, 0.99)), "mean": r4(sum(bias) / len(bias)) if bias else None,
                                       "n_gt_0.01": sum(1 for b in bias if b > 0.01), "n_gt_0.03": sum(1 for b in bias if b > 0.03)}
        if extra:
            d.update(extra)
        return d

    c43p = fx["4-3 짝 못 맞춤(이름 시각 있음)"]
    tiers = collections.Counter(r["f43_tier"] for r in c43p)
    tier_by_g = collections.Counter((r["f43_tier"], r["g"]) for r in c43p)
    consist = [r for r in c43p if r["h_lo_top7"] is not None]
    cons_ok = [r for r in consist if r["h_lo_top7"] - 0.1 <= r["new_h"] <= r["h_hi_top7"] + 0.1]
    out_fx["4-3a"] = core(c43p, "4-3 짝 못 맞춤: 글 첨부 수 ≠ 사진 수라 이름이 안 붙었지만 같은 글 첨부에 쓸 수 있는 이름 시각이 있는 C 줄",
                          extra={"tiers": {k: v for k, v in tiers.items()}, "tiers_by_server": {f"{k[0]}|{k[1]}": v for k, v in tier_by_g.items()},
                                 "top7_consistency": {"n_with_bracket": len(consist), "n_inside": len(cons_ok), "share": r4(len(cons_ok) / len(consist)) if consist else None,
                                                      "wilson95": wilson(len(cons_ok), len(consist))}})
    c43x = fx["4-3 이름 꼴 확장(ep13·loose·mac·yy)"]
    kc = collections.Counter(k for r in c43x for k in r["f43_ext"])
    consist = [r for r in c43x if r["h_lo_top7"] is not None]
    cons_ok = [r for r in consist if r["h_lo_top7"] - 0.1 <= r["new_h"] <= r["h_hi_top7"] + 0.1]
    out_fx["4-3b"] = core(c43x, "4-3 이름 꼴 확장: 이름은 붙었는데 지금 정규식이 못 읽는 꼴(2026-05 앞 13자리 epoch, 공백·마침표 구분, 맥 한글 스크린샷, YYMMDD)",
                          extra={"by_form": dict(kc), "top7_consistency": {"n_with_bracket": len(consist), "n_inside": len(cons_ok), "share": r4(len(cons_ok) / len(consist)) if consist else None,
                                                                          "wilson95": wilson(len(cons_ok), len(consist))},
                                 "caveat": "13자리 epoch 이름은 앱이 올릴 때 붙이는 이름(업로드 시각)일 수 있다 — 촬영 시각이 아닐 위험. 2026-05 뒤 시즌(S39~)은 이미 A 로 쓴다."})
    both = {id(r): r for r in c43p + c43x}
    out_fx["4-3(a+b)"] = core(list(both.values()), "4-3 합")
    # 4-4
    yl, ysh = claim_yield(seasons)
    own43 = {id(r) for r in both.values()}
    med_shift = {"clock": (ysh.get("clock") or {}).get("quantiles_h", {}).get("0.5"),
                 "ago": (ysh.get("yesterday") or {}).get("quantiles_h", {}).get("0.5"),
                 "day": (ysh.get("weekday") or {}).get("quantiles_h", {}).get("0.5"), "now": 0.0}
    sigB = {"clock": 0.5, "ago": 1.0, "day": 2.0, "now": 0.5}
    prio = ["ago", "clock", "day", "now"]
    cls_rows = collections.defaultdict(list)
    for r in fx["4-4 본문 시각 말(전체)"]:
        if id(r) in own43:
            continue
        c = next(k for k in prio if k in r["tw"])
        r["tclass"] = c
        cls_rows[c].append(r)
    text_blocks = {}
    for c, L in cls_rows.items():
        y = (yl.get(c) or {}).get("rate") or 0.0
        sel_t = [r for r in L if r["g"] == "KR" and r["he"] is not None and r["he"] < 48 and r["band"] is not None]
        sel_k = [r for r in L if r["g"] == "KR"]
        def bias_n(X, thr):
            n_ = 0
            for r in X:
                w = win_of(r["he"])
                g = (gam.get(w) or {}).get("gamma_per_h")
                if g is not None and abs(g) * (med_shift[c] or 0) > thr:
                    n_ += 1
            return n_
        g_all, _ = gain_for(L, gam, delays, SIG, sigB[c])
        g_kt, _ = gain_for(sel_t, gam, delays, SIG, sigB[c])
        text_blocks[c] = {"rows": tab(L), "claim_yield_B": yl.get(c), "median_shift_if_B_h": med_shift[c], "sigma_B_assumed_h": sigB[c],
                          "expected_converted_rows": r4(len(L) * y), "expected_converted_KR": r4(len(sel_k) * y), "expected_converted_KR_early48_target": r4(len(sel_t) * y),
                          "expected_converted_unk_early48_target": r4(sum(1 for r in L if r["g"] == "unk" and r["he"] is not None and r["he"] < 48 and r["band"] is not None) * y),
                          "KR_rows_with_bias_gt_1pct_if_B": r4(bias_n(sel_k, 0.01) * y), "KR_rows_with_bias_gt_3pct_if_B": r4(bias_n(sel_k, 0.03) * y),
                          "eq_points_time_model_all": r4(g_all * y), "eq_points_time_model_KR_early48_target": r4(g_kt * y)}
    # 타이머: 화살표·갈라진 줄 (strict 규칙이 못 읽는 것)
    tim = fx["4-4 OCR 갱신 타이머(화살표·갈라진 줄)"]
    tim_ok = [r for r in tim if -0.05 <= r["timer_shift"] <= 96]
    ref_all = fx["4-4 OCR 갱신 글자 있음"]
    tv = timer_validation(seasons)
    d_tim = core(tim_ok, "OCR 에 「갱신 시간 → N시간 M분 남음」(화살표 또는 갈라진 두 줄)이 있어 느슨한 규칙으로 읽히지만 지금 규칙(post_time.TIMER)은 못 읽는 C 대표 줄",
                 shift_key="timer_shift", new_sigma=0.02,
                 extra={"timer_validation_on_A_photos": tv, "n_refresh_text_C": len(ref_all), "n_loose_parse_failed_range": len(tim) - len(tim_ok)})
    for r in tim_ok:
        r["shift"] = r["timer_shift"]
    tb = [r for r in tim_ok if r["h_lo_top7"] is not None and r["h_hi_top7"] is not None]
    tb_in = [r for r in tb if r["h_lo_top7"] - 0.1 <= r["timer_h"] <= r["h_hi_top7"] + 0.1]
    d_tim["timer_vs_top7_bracket"] = {"n_with_bracket": len(tb), "n_inside": len(tb_in), "share": r4(len(tb_in) / len(tb)) if tb else None, "wilson95": wilson(len(tb_in), len(tb)),
                                      "outside_rows": [{"season": f"S{r['season']}", "key": r["key"], "timer_shift_h": r4(r["timer_shift"])} for r in tb if r not in tb_in][:10]}
    big = [r for r in tim_ok if r["timer_shift"] > 6]
    d_tim["big_shift_rows_gt_6h"] = {"n": len(big), "with_bracket": sum(1 for r in big if r["h_lo_top7"] is not None),
                                     "inside": sum(1 for r in big if r["h_lo_top7"] is not None and r["h_lo_top7"] - 0.1 <= r["timer_h"] <= r["h_hi_top7"] + 0.1)}
    out_fx["4-4"] = {"label": "4-4 글·사진 속 시각 (C → B)", "본문 시각 말(4-3 후보 뺌, 가장 큰 이동 종류 하나로 배정)": text_blocks,
                     "글 주장 B 수율(시각 말 종류별)": yl, "글 주장 B 가 게시보다 이른 정도(근거별, 시간)": ysh,
                     "OCR 타이머": d_tim,
                     "OCR 작업표시줄 시계 꼴": {"rows": tab(fx["4-4 OCR 작업표시줄 시계 꼴"]), "note": "C 대표 줄에서 2줄뿐 — 무시해도 된다"}}
    tm_rows = tim_ok
    tw_rows = [r for c in ("ago", "clock", "day") for r in cls_rows[c]]
    out_fx["4-4"]["합(본문 ago·clock·day + OCR 타이머, 겹침 제거)"] = tab(list({id(r): r for r in tm_rows + tw_rows}.values()))
    # 4-2
    mx = []
    for r in br:
        r["bracket_w"] = r["h_hi_top7"] - r["h_lo_top7"]
        r["maxdelay"] = r["he"] - r["h_lo_top7"]
        r["shift_hi"] = r["he"] - r["h_hi_top7"]
    gam_gain = {}
    # 구간 상한으로 h 를 내리는 효과 (지금 load_obs 가 안 쓰는 부분)
    sh_hi = [r["shift_hi"] for r in br]
    d42 = core(br, "h_lo_top7·h_hi_top7 가 둘 다 있는 C 대표 줄", shift_key="shift_hi")
    d42["bracket_width_h"] = {"n": len(br), "quantiles_h": quant([r["bracket_w"] for r in br], (0.1, 0.25, 0.5, 0.75, 0.9)),
                              "share_le_1h": r4(sum(1 for r in br if r["bracket_w"] <= 1) / len(br)), "share_le_3h": r4(sum(1 for r in br if r["bracket_w"] <= 3) / len(br))}
    d42["max_delay_h"] = {"n": len(br), "quantiles_h": quant([r["maxdelay"] for r in br], (0.1, 0.25, 0.5, 0.75, 0.9)),
                          "share_le_0.25h": r4(sum(1 for r in br if r["maxdelay"] <= 0.25) / len(br)), "share_le_1h": r4(sum(1 for r in br if r["maxdelay"] <= 1) / len(br)),
                          "share_le_6h": r4(sum(1 for r in br if r["maxdelay"] <= 6) / len(br))}
    d42["h_hi_top7_lowers_h"] = {"n_gt_0.1h": sum(1 for r in br if r["shift_hi"] > 0.1), "n_gt_1h": sum(1 for r in br if r["shift_hi"] > 1), "n_gt_12h": sum(1 for r in br if r["shift_hi"] > 12)}
    # 구간을 시각 모형에 쓰면: 지연을 구간의 최대 지연 이하로 자른 뒤 남는 무게
    tg = {"지연=A 분포": {"all": 0.0, "KR": 0.0, "KR_e48t": 0.0, "KR_e24t": 0.0}, "지연=C 후보 분포": {"all": 0.0, "KR": 0.0, "KR_e48t": 0.0, "KR_e24t": 0.0}}
    for nm, dsc in (("지연=A 분포", delays), ("지연=C 후보 분포", delays_C)):
        for r in br:
            w = win_of(r["he"])
            g = (gam.get(w) or {}).get("gamma_per_h")
            if g is None:
                continue
            g = abs(g)
            wb = weight(g, dsc[w], SIG)
            cut = [d for d in dsc[w] if d <= max(r["maxdelay"], 0)]
            wa = weight(g, cut, SIG) if len(cut) >= 5 else 1.0
            if wb is None or wa is None:
                continue
            dg = max(0.0, wa - wb)
            tg[nm]["all"] += dg
            if r["g"] == "KR":
                tg[nm]["KR"] += dg
                if r["he"] < 48 and r["band"] is not None:
                    tg[nm]["KR_e48t"] += dg
                    if r["he"] < 24:
                        tg[nm]["KR_e24t"] += dg
    d42["eq_points_if_time_model_truncates_by_bracket"] = {nm: {k: r4(v) for k, v in dd.items()} for nm, dd in tg.items()}
    d42["eq_points_note"] = "표 칸 h_lo_top7·h_hi_top7 을 관측 층이 직접 읽으면 공용 도구 고침 없이 얻는다. core 의 eq_exact_points… 는 「전부 정확한 점이 된다」는 가정이라 4-2 에는 맞지 않는다 — 이 값을 쓴다"
    out_fx["4-2"] = d42
    # 4-1
    sb, fixrows = server_basis_table(seasons)
    kr_basis = collections.Counter()
    for (b, k), v in sb.items():
        kr_basis[b] += v
    anchor_text = sum(v for b, v in kr_basis.items() if b in ("text", "top7_1", "top7_post", "text+top7_post"))
    dis = collections.Counter()
    for n in idx:
        try:
            for it in json.load(open(os.path.join(C.ROOT, "data", "solo-raid", f"S{n}", "discard.json"), encoding="utf-8"))["items"]:
                if it.get("action") == "server":
                    dis[(it.get("scope"), n)] += 1
        except OSError:
            pass
    nfix_items = sum(dis.values())
    out_fx["4-1"] = {"label": "4-1 서버 글자 근거",
                     "한섭 라벨 대표 사진 줄(A·B·C)의 근거별": dict(kr_basis),
                     "글자 앵커에만 기댄 한섭 줄(text·top7_1·top7_post·text+top7_post)": {"n": anchor_text, "share_of_KR": r4(anchor_text / max(1, sum(kr_basis.values()))),
                                                                              "wilson95": wilson(anchor_text, sum(kr_basis.values()))},
                     "손으로 서버를 고친 줄(server_basis=fix) 대표 사진 줄, (서버, 종류)": {f"{k[0]}|{k[1]}": v for k, v in fixrows.items()},
                     "discard.json 서버 고침 항목 수(행·사진·글 단위, S19~S42)": nfix_items,
                     "새 라벨(미상→라벨)": "거의 없음 — 고침은 라벨을 빼는 쪽(앵커로 안 쓰는 문장 종류를 늘림). 정확한 수는 고침 코드 없이는 못 센다",
                     "효과 방향": "D0 에는 알려진 서버 오류가 discard 로 이미 빠졌다 → D0 분석에서 얻을 것은 남은 숨은 오류 몫(1e 의 착오율)이고, S43 이후 새 시즌의 오염 방지가 주 몫"}
    # 4-5
    vr, vex = verify_reverse(seasons)
    out_fx["4-5"] = {"label": "4-5 판독 점검 후보", "verify_differ 대표 사진 줄 (종류, 서버)": {f"{k[0]}|{k[1]}": v for k, v in vr["verify_differ"].items()},
                     "시간과 거꾸로 가는 % (같은 사람·같은 점수) 쌍": vr["reverse_pairs"],
                     "거꾸로 간 뒤쪽 줄 (종류, 서버, 대표 여부)": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in vr["reverse_rows"].items()},
                     "예": vex,
                     "note": "D0 에는 5.1 반영으로 순서 모순이 0 이어야 한다 — 남아 있으면 이 점검이 못 잡은 것이 아니라 5.1 뒤 새로 생긴 것"}
    # 2 (구조 정리)
    out_fx["2"] = {"label": "2 구조 정리 (실험 폴더 도구를 tools/solo_raid/ 로)", "바뀌는 줄": 0,
                   "비용": "옮기기만 해도 모든 시즌 도장이 stale → analyze_all.py 약 25분 (순서표 2·method WP2 카드). 4-1·4-3 이 같은 파일을 건드리므로 먼저 해야 한다"}

    # 3. 합집합·천장·민감도
    def subsets(rows):
        return {"전체": rows, "KR": [r for r in rows if r["g"] == "KR"],
                "KR 초반48": [r for r in rows if r["g"] == "KR" and r["he"] is not None and r["he"] < 48],
                "KR 초반48 목표근처": [r for r in rows if r["g"] == "KR" and r["he"] is not None and r["he"] < 48 and r["band"] is not None],
                "KR 초반24 목표근처": [r for r in rows if r["g"] == "KR" and r["he"] is not None and r["he"] < 24 and r["band"] is not None],
                "미상 초반48 목표근처": [r for r in rows if r["g"] == "unk" and r["he"] is not None and r["he"] < 48 and r["band"] is not None]}
    conv = {}
    for r in c43p + c43x + tm_rows + cls_rows["ago"] + cls_rows["clock"]:
        conv[id(r)] = r
    conv_rows = list(conv.values())
    ceiling = {}
    union = {}
    union_nt = {}
    nt_rows = list({id(r): r for r in c43p + c43x + tm_rows}.values())
    for tag, sub in subsets(nt_rows).items():
        uu = {"n_rows": len(sub)}
        for nm, dsc in (("지연=A 분포", delays), ("지연=C 후보 분포", delays_C)):
            uu[nm] = r4(gain_for(sub, gam, dsc, SIG, 0.0)[0])
        union_nt[tag] = uu
    for tag, sub in subsets(rep["C"]).items():
        cc = {}
        for nm, dsc in (("지연=A 분포", delays), ("지연=C 후보 분포", delays_C)):
            cc[nm] = {"n_C_rows": len(sub), "eq_points_if_all_C_became_exact": r4(gain_for(sub, gam, dsc, SIG, 0.0)[0])}
        ceiling[tag] = cc
    for tag, sub in subsets(conv_rows).items():
        uu = {"n_rows": len(sub)}
        for nm, dsc in (("지연=A 분포", delays), ("지연=C 후보 분포", delays_C)):
            uu[nm] = r4(gain_for(sub, gam, dsc, SIG, 0.0)[0])
        union[tag] = uu
    sens = {}
    for sg in (0.003, 0.01, 0.03):
        sens[str(sg)] = {"4-3(a+b) KR 초반48 목표근처(A 분포)": r4(gain_for(subsets(list(both.values()))["KR 초반48 목표근처"], gam, delays, sg, 0.0)[0]),
                         "합집합 KR 초반48 목표근처(A 분포)": r4(gain_for(subsets(conv_rows)["KR 초반48 목표근처"], gam, delays, sg, 0.0)[0]),
                         "C 전부 KR 초반48 목표근처(A 분포)": r4(gain_for(subsets(rep["C"])["KR 초반48 목표근처"], gam, delays, sg, 0.0)[0])}
    # 4. 표
    def eqp(d, tag, sc="지연=A 분포"):
        return (d["eq_exact_points_gain_time_model"].get(f"{sc} | {tag}") or {}).get("eq_points")
    def trow(name, d, cost, note, rows_override=None):
        R = rows_override or d["rows"]
        row = {"fix": name, "rows_changed": R["n"], "KR": R["KR"], "unk": R["unk"], "KR_early48": R["KR_early48"], "KR_early48_target": R["KR_early48_target"],
               "KR_early24_target": R["KR_early24_target"], "unk_early48_target": R["unk_early48_target"]}
        if "eq_exact_points_gain_time_model" in d:
            row["eq_exact_points_time_model_KR_early48_target_A_C"] = [eqp(d, "KR 초반48 목표근처"), eqp(d, "KR 초반48 목표근처", "지연=C 후보 분포")]
            row["eq_exact_points_time_model_KR_all_A_C"] = [eqp(d, "KR"), eqp(d, "KR", "지연=C 후보 분포")]
            n_ = (d["eq_exact_points_gain_time_model"].get("지연=A 분포 | KR 초반48 목표근처") or {}).get("n_rows")
            if n_:
                row["time_model_value_per_row_KR_early48_target_A"] = r4((eqp(d, "KR 초반48 목표근처") or 0) / n_)
            fb = d.get("F_style_bias_removed_KR") or {}
            row["F_style_mean_cutoff_bias_removed_KR"] = fb.get("mean")
        row["cost"] = cost
        row["note"] = note
        return row
    table = [
        trow("4-3 첨부 이름 ↔ 사진 짝 + 이름 꼴 (C→A)", out_fx["4-3(a+b)"], "post_time.py 안에서 (state 의 posts_min 첨부 목록·manifest 로 이름을 찾고 시각 꼴 확장) — 수집 파일 안 바꿈. analyze_all 약 25분 + 코드 반나절 추정",
             "짝 못 맞춤 319줄은 92%가 이름 시각이 10분 안에 모여 짝 규칙 없이 「글의 이름 시각 하나」로 충분. 이름 꼴 확장 872줄 중 688이 2026-05 앞 13자리 epoch"),
        trow("4-4 OCR 갱신 타이머 느슨한 규칙 (C→B)", out_fx["4-4"]["OCR 타이머"], "post_time.py TIMER 정규식 한 줄 + 갈라진 줄 잇기. analyze_all 약 25분. 코드 1시간 안팎 추정",
             "A 사진 208장으로 검증: 타이머 시각이 촬영과 2분 안 98%"),
        dict(trow("4-4 본문 시각 말 (C→B, ago·clock·day 기대값)", {}, "post_time.py 말 규칙 여럿 + 판독 확인. 수율은 글 주장 기준 상한 (ago 100%·clock 74%·day 6%)",
                  "now 는 이동 0 이라 얻을 것 없음(제외). ago·clock 은 이동이 몇 시간(중앙 7.5·4.2시간)이라 한 줄당 편향은 크지만 줄 수가 적다. B 의 σ 를 0.5~2시간으로 두면 초반 γ 에서 C 의 평균 무게보다 낫지 않아 시각 모형 아래 이득은 0 — 남는 것은 편향 제거",
                  rows_override={"n": r4(sum(text_blocks[c]["expected_converted_rows"] for c in ("ago", "clock", "day"))), "KR": r4(sum(text_blocks[c]["expected_converted_KR"] for c in ("ago", "clock", "day"))),
                                 "unk": None, "KR_early48": None,
                                 "KR_early48_target": r4(sum(text_blocks[c]["expected_converted_KR_early48_target"] for c in ("ago", "clock", "day"))), "KR_early24_target": None,
                                 "unk_early48_target": r4(sum(text_blocks[c]["expected_converted_unk_early48_target"] for c in ("ago", "clock", "day")))}),
             eq_exact_points_time_model_KR_early48_target_A_C=[0.0, 0.0]),
        dict(trow("4-2 순위표 시각 구간을 h 에 (시각 모형이 구간을 읽을 때의 값)", out_fx["4-2"], "관측 층이 표 칸을 읽으면 도구 고침 0. 도구를 고치면 post_time/load_obs 고침 — 운영 predict.load_obs 를 건드리고 h 를 내리는 줄은 35줄뿐",
                  "구간 열은 관측 표에 이미 있다. 한섭 라벨 줄에만 있다(미상 줄은 서버를 알아야 구간이 나온다)"),
             eq_exact_points_time_model_KR_early48_target_A_C=[out_fx["4-2"]["eq_points_if_time_model_truncates_by_bracket"]["지연=A 분포"]["KR_e48t"], out_fx["4-2"]["eq_points_if_time_model_truncates_by_bracket"]["지연=C 후보 분포"]["KR_e48t"]],
             eq_exact_points_time_model_KR_all_A_C=[out_fx["4-2"]["eq_points_if_time_model_truncates_by_bracket"]["지연=A 분포"]["KR"], out_fx["4-2"]["eq_points_if_time_model_truncates_by_bracket"]["지연=C 후보 분포"]["KR"]]),
        {"fix": "4-1 서버 글자 근거 (라벨 빼기)", "rows_changed": out_fx["4-1"]["글자 앵커에만 기댄 한섭 줄(text·top7_1·top7_post·text+top7_post)"]["n"], "KR": out_fx["4-1"]["글자 앵커에만 기댄 한섭 줄(text·top7_1·top7_post·text+top7_post)"]["n"],
         "note": "숫자는 후보 줄(글자 앵커에만 기댄 한섭 대표 사진 줄) — 이 중 실제 바뀌는 줄은 고침 없이는 못 센다. 알려진 오류는 D0 에서 이미 discard 로 빠졌다. 정보량이 아니라 오염 방지",
         "cost": "server_classify.py(실험 폴더) → 2 구조 정리 먼저. analyze_all 약 25분, S32 묶음 img 고침 필요 (순서표 4-1), 위험 중간"},
        {"fix": "4-5 판독 점검 후보", "rows_changed": sum(out_fx["4-5"]["verify_differ 대표 사진 줄 (종류, 서버)"].values()), "note": "verify=differ 대표 사진 줄 수. 거꾸로 가는 % 는 D0 에 0쌍", "cost": "review.py 에 kind 추가, 작음 (자료는 안 바뀜)"},
        {"fix": "2 구조 정리", "rows_changed": 0, "note": "줄 수 변화 없음. 4-1 의 선행 조건", "cost": "도장 전부 stale → analyze_all 약 25분"},
    ]
    recommend = [
        {"order": 1, "fix": "4-4 OCR 타이머 느슨한 규칙", "why": "비용이 가장 작고(정규식 한 줄 + 갈라진 줄 잇기, post_time.py) 검증이 가장 강하다(A 208장 2분 안 98%, 순위표 구간 175/177). 한섭 150줄을 B(±1분)로 올린다(한섭 C 의 7.1%). "
                                       "줄당 이동이 중앙 1.2분이라 정밀도 이득은 작다 — 이득보다 위험이 작아서 먼저. 순위표 구간 밖이면 올리지 않는 안전장치를 붙인다(이동 6시간 넘는 5줄 중 2줄이 구간 밖 = 하루 어긋남)."},
        {"order": 2, "fix": "4-3 이름 꼴 확장 + 짝 못 맞춤 (1과 같은 post_time.py 고침, analyze_all 한 번)", "why": "바뀌는 줄이 가장 많다(대표 1,191줄, 한섭 348·미상 756). 수집 파일을 안 바꾸고 post_time.py 가 state 의 posts_min 첨부 목록으로 이름을 찾으면 된다. "
                                                                                        "짝 못 맞춤 319줄은 92%(293)가 이름 시각이 10분 안에 모여 짝 규칙(sha1·확장자) 없이 「글의 이름 시각」으로 충분하다. 꼴 확장 872줄의 79%는 2026-05 앞 13자리 epoch(S39~ 는 이미 A) — 업로드 시각에 가까울 위험은 S39~ 가 이미 지고 있는 것과 같다. 정보량은 가장 크나 한섭 초반 목표근처 전체로는 1.4~4.1개분."},
        {"order": 3, "fix": "4-2 는 공용 도구 고침에서 뺀다 — 관측 층이 h_lo_top7·h_hi_top7 열을 직접 읽는다", "why": "열이 관측 표에 이미 있어 도구를 안 고쳐도 얻는다(한섭 초반 목표근처 1.9~7.1개분으로 도구 고침 전체와 비슷하거나 크다). 도구로 h 를 내려 주는 줄은 35줄(1시간 넘게 17줄)뿐이고 운영 predict.load_obs 를 건드려 위험만 있다."},
        {"order": 4, "fix": "4-4 본문 시각 말", "why": "한섭 초반 목표근처 기대 3줄. 「지금」은 이동 0 이라 득이 없고(B 의 ±30분이 오히려 C 의 평균 무게보다 느슨), ago·clock 은 이동이 몇 시간이라 한 줄당 편향(컷 3~10%)은 크지만 줄 수(한섭 기대 72)가 적다. 시각 모형이 지연 꼬리로 흡수한다."},
        {"order": 5, "fix": "4-1 (선행: 2)", "why": "정보를 더하는 일이 아니라 오염 방지. D0 에는 알려진 서버 오류(손으로 서버를 고친 대표 줄 16)가 이미 고쳐졌다. 글자 앵커에만 기댄 한섭 대표 줄 564(한섭의 14%)가 후보이고 S43 이후 새 시즌 대비로 따로 정한다. 하려면 server_classify.py 를 옮기는 2 가 먼저."},
        {"order": 6, "fix": "4-5", "why": "자료를 바꾸지 않는 점검 목록(verify=differ 대표 사진 121줄, 거꾸로 가는 % 는 D0 에 0쌍). 이 연구의 오차 크기에는 영향 없음."}]
    wp2_scope = {"proposal": "post_time.py 만 고치는 한 번(4-4 타이머 + 4-3 이름 꼴·짝) → analyze_all 약 25분 → e_premise 점검 → 자료 판 D1",
                 "skip": ["2 구조 정리", "4-1", "4-2 (도구)", "4-4 본문 시각 말", "4-5"],
                 "expected_gain_KR_early48_target_eq_points": {"A분포": union_nt["KR 초반48 목표근처"]["지연=A 분포"], "C분포": union_nt["KR 초반48 목표근처"]["지연=C 후보 분포"], "rows": union_nt["KR 초반48 목표근처"]["n_rows"]},
                 "expected_gain_KR_early24_target_eq_points": {"A분포": union_nt["KR 초반24 목표근처"]["지연=A 분포"], "C분포": union_nt["KR 초반24 목표근처"]["지연=C 후보 분포"], "rows": union_nt["KR 초반24 목표근처"]["n_rows"]},
                 "expected_gain_unk_early48_target_eq_points": {"A분포": union_nt["미상 초반48 목표근처"]["지연=A 분포"], "C분포": union_nt["미상 초반48 목표근처"]["지연=C 후보 분포"], "rows": union_nt["미상 초반48 목표근처"]["n_rows"]},
                 "reason": "합집합이 한섭 초반 목표근처 24시즌 합 정확한 점 약 2~5개분(기존 143개의 1~4%)이라 작다 — 비싼 고침(2·4-1)을 할 값이 이 연구의 오차 크기로는 안 보인다. 결정은 조율 세션."}

    # 5. 시각 → % 환산, 합집합의 분포, 확인용 줄
    t2p = {}
    for w, g in gam.items():
        if not g:
            continue
        dq_ = delay_block["A_by_window"].get(w, {}).get("quantiles_h", {})
        t2p[w] = {"gamma_per_h": g["gamma_per_h"], "gamma_ci95": g["gamma_ci95"], "n_points": g["n"],
                  "A_delay_h": dq_, "cutoff_rel_error_at_delay": {q: r4(abs(g["gamma_per_h"]) * v) for q, v in dq_.items() if v is not None},
                  "cutoff_rel_error_at_5min": r4(abs(g["gamma_per_h"]) * 5 / 60), "cutoff_rel_error_at_30min": r4(abs(g["gamma_per_h"]) * 0.5)}
    union_bins = {"KR": bins6(conv_rows, "KR"), "KR_target": bins6(conv_rows, "KR", True), "unk_target": bins6(conv_rows, "unk", True)}
    per_season_union = {}
    for n in idx:
        L = [r for r in conv_rows if r["season"] == n]
        per_season_union[f"S{n}"] = {"n": len(L), "KR": sum(1 for r in L if r["g"] == "KR"),
                                     "KR_early48_target": sum(1 for r in L if r["g"] == "KR" and r["he"] is not None and r["he"] < 48 and r["band"] is not None)}
    ps = sorted(v["KR_early48_target"] for v in per_season_union.values())
    check = []

    def pick(L, n_, key=None, rev=True):
        L2 = sorted(L, key=key, reverse=rev) if key else L
        return L2[:n_]

    def card(tag, r):
        d = {"why": tag, "season": f"S{r['season']}", "key": r["key"], "kind": r["kind"], "t_basis": r["t_basis"], "server": r["server"], "server_basis": r["server_basis"],
             "pct": r["pct"], "he": r["he"], "hp": r["hp"], "h_lo": r["h_lo"], "h_hi": r["h_hi"], "h_lo_top7": r["h_lo_top7"], "h_hi_top7": r["h_hi_top7"]}
        for k in ("new_h", "timer_h", "shift", "timer_shift", "f43_tier", "f43_ext", "tclass", "tw"):
            if r.get(k) is not None:
                d[k] = r[k] if not isinstance(r[k], float) else r4(r[k])
        d["x"] = {k: v for k, v in r["x"].items() if k in ("paired", "nfiles", "nimg", "names_n", "site", "ext_pair") and k != "ext_pair"}
        return d
    check += [card("4-3 짝 못 맞춤, 이름 시각이 게시보다 1시간 넘게 이름 (꼬리)", r) for r in pick([r for r in c43p if r["g"] == "KR"], 2, key=lambda r: r["shift"])]
    check += [card("4-3 짝 못 맞춤, 이름이 여럿이라 애매", r) for r in pick([r for r in c43p if r["f43_tier"] == "ambiguous"], 1)]
    check += [card("4-3 이름 꼴 확장 (13자리 epoch) + 순위표 구간이 있어 이름 시각이 구간 안인지 볼 수 있음", r) for r in pick([r for r in c43x if r["g"] == "KR" and r["h_lo_top7"] is not None and "ep13" in r["f43_ext"]], 2, key=lambda r: r["shift"])]
    check += [card("4-4 타이머 (화살표·갈라진 줄) — 이동이 큰 줄", r) for r in pick([r for r in tm_rows if r["g"] == "KR"], 2, key=lambda r: r["timer_shift"])]
    check += [card("4-4 타이머 — 이동이 보통인 줄", r) for r in pick([r for r in tm_rows if r["g"] == "KR" and r["he"] is not None and r["he"] < 48 and r["band"] is not None], 2, key=lambda r: r["timer_shift"], rev=False)]
    check += [card("4-2 순위표 구간 상한이 게시보다 12시간 넘게 이름", r) for r in pick(br, 2, key=lambda r: r["shift_hi"])]
    check += [card("4-4 본문 ago(어제·아까 …)", r) for r in pick([r for r in cls_rows["ago"] if r["g"] == "KR"], 1)]
    check += [card("4-4 본문 clock(N시)", r) for r in pick([r for r in cls_rows["clock"] if r["g"] == "KR"], 1)]
    out_fx["4-3"] = out_fx["4-3(a+b)"]
    res = {"item": "1g", "data_ver": C.DATA_VER, "seeds": SEED, "seasons": f"S{idx[0]}~S{idx[-1]}",
           "unit_note": "모형이 보는 줄 = 표의 rep 칸(load_obs 출력의 대표) 사진 줄. 서버 집단 KR=한섭 라벨, unk=미상(mono 거르기를 지난 것), unk_drop=mono 가 뺀 미상, other=일섭·글섭 등. 초반=he<48, 목표 근처=pct 1·3·10 의 ±30%.",
           "method_note": "eq_points = 줄마다 (고친 뒤 무게 − 고치기 전 무게) 의 합, 무게 w = mean_D 1/(1+(γ(D−D0)/σ)²) (D: 게시 지연 표본, D0: 그 중앙값=모형이 맞춘 몫, γ: 같은 점수 컷의 시간당 ln 변화, σ=0.01 ln 점수). 정확한 점 하나 = 1.",
           "base": base_block, "gamma_table_ln_cutoff_per_hour": gam, "delay": delay_block, "sigma_meas_ln": SIG,
           "fixes": out_fx, "ceiling_if_all_C_exact": ceiling, "union_of_conversions": {"rows": "4-3 a+b, 타이머, 본문 ago·clock (now·day 제외)", "eq_points": union}, "union_without_text": union_nt,
           "sigma_sensitivity": sens, "info_gain": union, "time_to_cutoff_error": t2p, "union_by_6h_bin": union_bins, "union_per_season": per_season_union,
           "union_KR_early48_target_per_season_sorted": ps, "check_rows": check, "table": table, "recommend_order": recommend, "wp2_scope_proposal": wp2_scope,
           "elapsed_sec": round(__import__("time").time() - t0, 1)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda o: sorted(o) if isinstance(o, (set, frozenset)) else str(o))
    if a.md:
        write_md(res, a.md)
    print("saved", a.out, res["elapsed_sec"], "sec")
    return res


if __name__ == "__main__":
    main()
