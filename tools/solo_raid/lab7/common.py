"""새 예측 방법 연구(WP0, 2026-10-28) 공통 — 시즌 목록·자료 판·T 격자·절단 사본 경로.

docs/solo-raid-method.md 「4.4 검증 설계」「WP0 기반」. 운영 도구는 고치지 않고 읽기만 한다.
T 격자는 5.3의 lab6/e_53a.py t_grid와 같다 (6시간마다 120까지 + H−12·H−6·H).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
import predict as PR  # noqa: E402

DATA_VER = "D0"   # 자료 판 (2026-10-28, 커밋 881e9c5 뒤의 자료). WP2가 공용 도구·자료를 고치면 올린다
DEV = [f"S{n}" for n in range(19, 43)]
TARGETS = [f"S{n}" for n in range(21, 43)]
TUNE = ["S22", "S26", "S30", "S34", "S38", "S42"]   # 조정 시즌 (4.4)

_info = {}


def parse_seasons(spec):
    """'S21-S42' · 'S21,S25,S30-S33' → ['S21', ...]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += [f"S{n}" for n in range(int(a.lstrip("S")), int(b.lstrip("S")) + 1)]
        else:
            out.append(part)
    return out


def state_of(s):
    return os.path.join(ROOT, "data", "solo-raid", s, "state")


def info(s):
    """(c0, H) — 챌린지 시작 시각과 길이(시간)."""
    if s not in _info:
        _, c0, H = PR.season_info(state_of(s))
        _info[s] = (c0, float(H))
    return _info[s]


def t_grid(s):
    """T 격자: 6..120 (6시간마다, H 이하) + H−12·H−6·H."""
    _, H = info(s)
    Ts = [float(t) for t in range(6, 121, 6) if t <= H]
    for t in (H - 12, H - 6, H):
        if t not in Ts:
            Ts.append(float(t))
    return sorted(Ts)


def trunc_dir(work, s, T):
    """backtest.trunc_state가 쓰는 사본 폴더 (xval과 같은 자리라 서로 쓴다)."""
    return os.path.join(work, "trunc", s, f"trunc_T{T:g}")


def trunc_path(work, s, T):
    """다 만들어진(DONE) 절단 사본의 state 경로, 없으면 None."""
    d = trunc_dir(work, s, T)
    p = os.path.join(d, "state")
    return p if os.path.exists(os.path.join(p, "core.jsonl")) and os.path.exists(os.path.join(d, "DONE")) else None


def done_key(work, s, T):
    """DONE 파일의 열쇠 (원본 도장·도구·T) — 표가 어느 사본에서 나왔는지 적는 용도."""
    try:
        with open(os.path.join(trunc_dir(work, s, T), "DONE"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None
