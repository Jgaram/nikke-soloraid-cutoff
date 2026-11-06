"""연구 도구(lab7)는 시즌 목록이 S19~S42로 박혀 있다 (wp12_tables.SEASONS). 운영에서 S43 이후를 같은 도구로 다루려고
lab7 모듈을 import하기 *전에* 그 목록을 제자리에서 늘린다 (lab7 파일은 안 고친다, WP7 2026-11-07).

  m3_extend.pre_import(add=["S43"])    wp12_tables.SEASONS에 더한다 (뒤에서 import하는 wp12·wp13 모듈이 이 목록을 그대로 쓴다)
  m3_extend.post_import()               새 시즌에 없는 곡선 입력 (옛 F·act 비교용 곡선 — 침수·컷 꼴에는 안 쓰임)을 NaN으로 두는 덧옷
  drop=[...]                            리허설용: 목록에서 뺀 뒤 add로 다시 더하면 (S42 빼고 더하기) 같은 길이 같은 결과를 내는지 본다

안 부르면 아무것도 바뀌지 않는다 (m3_shape.py의 S43 출력은 바이트까지 그대로).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB7 = os.path.join(HERE, "lab7")


def argv_list(argv, name):
    """--name a,b 또는 --name=a,b 에서 목록 (argparse 전에 읽는다)."""
    out = []
    for i, x in enumerate(argv):
        if x == name and i + 1 < len(argv):
            out += [v for v in argv[i + 1].split(",") if v]
        elif x.startswith(name + "="):
            out += [v for v in x.split("=", 1)[1].split(",") if v]
    return out


def pre_import(add=(), drop=()):
    for p in (HERE, LAB7):
        if p not in sys.path:
            sys.path.insert(0, p)
    if not add and not drop:
        return
    import wp12_tables as WT
    if any(m in sys.modules for m in ("wp12_flood", "wp13_flood_shape", "wp13_cut_shape")):
        raise RuntimeError("m3_extend.pre_import는 wp12_flood·wp13_* import보다 먼저 불러야 한다")
    for s in drop:
        if s in WT.SEASONS:
            WT.SEASONS.remove(s)
    for s in add:
        if s not in WT.SEASONS:
            WT.SEASONS.append(s)
    nums = [int(x[1:]) for x in WT.SEASONS]
    assert nums == sorted(nums) and len(set(nums)) == len(nums), f"시즌 목록이 오름차순이 아니다: {WT.SEASONS}"


def post_import():
    """새 시즌이 1단계 파일(r7/stage1_M3_D1.json 등)에 없어도 FL.build_photos가 안 죽게 한다 — 그 곡선 열(Mb·Ma)은 꼴 만들기에서 쓰지 않는다."""
    import numpy as np
    import wp12_flood as FL
    import act_clock as AC
    m3c, ust = FL.m3_curve, AC.ustar

    def m3_curve(s1, s, u, pf):
        if s not in s1[1]:
            return np.full(np.shape(u), np.nan)
        return m3c(s1, s, u, pf)

    def ustar(prof, season, t, H):
        if season not in prof["g"]:
            return np.full(np.shape(t), np.nan)
        return ust(prof, season, t, H)

    FL.m3_curve, AC.ustar = m3_curve, ustar
