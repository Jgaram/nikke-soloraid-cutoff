"""기준선 F·B0·Z0를 공통 예측 파일 꼴로 만들고, 하한 파일을 만든다 (docs/solo-raid-method.md WP0 e).
파일 꼴은 `score.py` 머리말. 모든 출력은 data_ver와 made(tool_head·명령·입력 sha1)를 단다.

  python3 -I tools/solo_raid/lab7/baseline.py f --main main.json [--main main_s28.json ...] --ref ref.json [--ref ref_s28.json ...] --out r7/F_D0.json
  python3 -I tools/solo_raid/lab7/baseline.py b0 --cells B0.json [--cells ...] --f r7/F_D0.json --out r7/B0_D0.json
  python3 -I tools/solo_raid/lab7/baseline.py z0 --f r7/F_D0.json --out r7/Z0_D0.json
  python3 -I tools/solo_raid/lab7/baseline.py lb --work W --out r7/lb_D0.json

- f: e_53s.load_f와 같다. recent5 한섭 줄, 0.5·1·2·3%는 main에 M1 넓히기 (앞 시즌 1·3% |z| 80% 분위, T별 —
  freeze_model.m1_table · predict.m1_factor), 80%뿐 아니라 95% 구간도 같은 배율로. 5·10%는 ref 그대로. 뒤 파일이 앞 파일을 덮는다.
- b0: `lab6/e_53.py run --cands B0` 결과에서 cand B0 칸. 값이 없는 칸은 F 값으로 채우고 own=false.
- z0 (사전 고정 정의, 조정하지 않는다): Z0 = 앞 시즌 최종 컷, T에 상관없이 일정.
  맞힐 시즌 s·목표 q: 중앙 = s보다 앞선 개발 시즌 중 한섭 정답이 있는 가장 가까운 시즌의 정답.
  구간: r = log(정답_x ÷ 정답_prev(x)), x와 prev(x)가 모두 s보다 앞이고 둘 다 한섭 q 정답이 있는 이웃 쌍 전부
  (prev(x) = x보다 앞선, q 정답이 있는 가장 가까운 개발 시즌). 80% = 중앙 × exp(r의 0.1·0.9 분위),
  95% = 중앙 × exp(r의 0.025·0.975 분위) (선형 보간 분위, anchor.quant). 비가 2개 미만이면 두 구간 모두 ±max|r|,
  하나도 없으면 [중앙, 중앙]. 칸 집합은 F와 같다. own=true.
- lb: 맞힐 시즌 S21~S42 × T 격자(e_53a.t_grid, T 절단 사본이 있는 것) × 0.5·1·2·3·5·10%의 하한
  anchor.lower_bound(anchor.load(사본, 시즌, T, unknown=True), q, unknown=True). 모든 방법이 같은 하한을 쓴다.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "lab6"))
sys.path.insert(0, HERE)
import freeze_model as FZ  # noqa: E402
import predict as PR  # noqa: E402
import anchor as A  # noqa: E402
import e_53a as EA  # noqa: E402
import score as SC  # noqa: E402

DATA_VER = "D1"
M1_T = (12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0)
M1_Q = (0.5, 1.0, 2.0, 3.0)
REF_Q = (5.0, 10.0)
LB_Q = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)


def made(inputs):
    """made 머리말: tool_head(git 짧은 sha) · 명령 · 입력 파일 sha1[:12]. inputs {이름: 경로 또는 이미 구한 sha}."""
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        head = None
    ins = {k: (SC.sha12(v) if os.path.isfile(v) else v) for k, v in inputs.items()}
    return {"tool_head": head, "cmd": "python3 -I " + " ".join(sys.argv), "inputs": ins}


def key(c):
    return (c["season"], float(c["T"]), float(c["pct"]))


def order(c):
    return (int(c["season"][1:]), c["T"], c["pct"])


def write(out, d):
    d["cells"].sort(key=order)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    print("wrote", out, len(d["cells"]), "칸")


def named(paths):
    """입력 이름 → 경로. 같은 이름이면 앞에 순번."""
    out = {}
    for p in paths:
        n = os.path.basename(p)
        out[n if n not in out else f"{len(out)}_{n}"] = p
    return out


def widen_iv(m, iv, c):
    """M1: 중앙 m을 두고 로그 반폭을 c배 (predict.widen과 같은 셈, 양수일 때만)."""
    if not iv or None in iv or iv[0] <= 0 or iv[1] <= 0 or m <= 0:
        return iv
    return [m * (iv[0] / m) ** c, m * (iv[1] / m) ** c]


def cmd_f(a):
    kr = lambda r: r.get("scheme") == "recent5" and r.get("server", "KR") == "KR"   # noqa: E731
    rm = [r for p in a.main for r in SC.jl(p)["rows"] if kr(r)]
    rr = [r for p in a.ref for r in SC.jl(p)["rows"] if kr(r)]
    cells = {}
    m1s = {}
    for r in rm:
        s = r["season"]
        if s not in m1s:
            tab = FZ.m1_table([x for x in rm if float(x["T"]) in M1_T], s)
            m1s[s] = {"m1": {"T": {f"{T:g}": c for T, (c, _, _) in tab.items()}}} if tab else None
        q = float(r["pct"])
        if q not in M1_Q or not r.get("pred") or not r.get("i80") or None in r["i80"]:
            continue
        m = r["pred"]
        i80, i95 = list(r["i80"]), r.get("i95")
        if m1s[s]:
            c = PR.m1_factor(m1s[s], float(r["T"]))
            i80 = [m * (i80[0] / m) ** c, m * (i80[1] / m) ** c]
            i95 = widen_iv(m, i95, c)
        cells[(s, float(r["T"]), q)] = dict(season=s, T=float(r["T"]), pct=q, server="KR", pred=m, i80=i80, i95=i95, own=True)
    for r in rr:
        q = float(r["pct"])
        if q not in REF_Q or not r.get("pred") or not r.get("i80") or None in r["i80"]:
            continue
        cells[(r["season"], float(r["T"]), q)] = dict(season=r["season"], T=float(r["T"]), pct=q, server="KR", pred=r["pred"],
                                                      i80=list(r["i80"]), i95=r.get("i95"), own=True)
    ins = named(a.main + a.ref)
    write(a.out, dict(about="F 운영 예측 (main 0.5·1·2·3% M1 넓힘 + ref 5·10%), recent5, 한섭 — 5.3 B와 같은 설정", method="F",
                      data_ver=a.data_ver, made=made(ins), cells=list(cells.values())))


def cmd_b0(a):
    F = SC.load_pred(a.f)
    if F["data_ver"] != a.data_ver:
        sys.exit(f"F 자료 판 {F['data_ver']} ≠ {a.data_ver}")
    B = {}
    for p in a.cells:
        for c in SC.jl(p)["cells"]:
            if c["cand"] == "B0":
                B[(c["season"], float(c["T"]), float(c["q"]))] = c   # 뒤 파일이 덮음
    cells = {}
    for k, c in B.items():
        if "pred" in c:
            cells[k] = dict(season=k[0], T=k[1], pct=k[2], server="KR", pred=c["pred"], i80=list(c["i80"]), i95=c.get("i95"), own=True)
    for f in F["cells"]:   # 값이 없는 칸은 F로 (own=false)
        k = key(f)
        if k not in cells:
            cells[k] = dict(f, own=False)
    ins = named(a.cells)
    ins["F"] = a.f
    write(a.out, dict(about="B0 — 최근 6시간 A·B 닻 하나 × 앞 시즌 비 중앙 (lab6/e_53.py, cand B0). 빈 칸은 F로(own=false)", method="B0",
                      data_ver=a.data_ver, made=made(ins), cells=list(cells.values())))


def z0_value(tr, s, q):
    """맞힐 시즌 s·목표 q의 Z0 (중앙, i80, i95). tr {(시즌, %): 정답}. 정의는 머리말."""
    seq = [x for x in A.DEV[:A.DEV.index(s)] if (x, q) in tr]
    if not seq:
        return None
    pred = tr[(seq[-1], q)]
    r = sorted(math.log(tr[(seq[i], q)] / tr[(seq[i - 1], q)]) for i in range(1, len(seq)))
    if len(r) >= 2:
        i80 = [pred * math.exp(A.quant(r, 0.1)), pred * math.exp(A.quant(r, 0.9))]
        i95 = [pred * math.exp(A.quant(r, 0.025)), pred * math.exp(A.quant(r, 0.975))]
    elif r:
        w = max(abs(x) for x in r)
        i80 = i95 = [pred * math.exp(-w), pred * math.exp(w)]
    else:
        i80 = i95 = [pred, pred]
    return pred, i80, i95


def cmd_z0(a):
    F = SC.load_pred(a.f)
    if F["data_ver"] != a.data_ver:
        sys.exit(f"F 자료 판 {F['data_ver']} ≠ {a.data_ver}")
    tr = {(s, q): v for (s, srv, q), v in A.truth().items() if srv == "KR"}
    cells = []
    memo = {}
    for f in F["cells"]:
        s, q = f["season"], f["pct"]
        if (s, q) not in memo:
            memo[(s, q)] = z0_value(tr, s, q)
        z = memo[(s, q)]
        if z is None:
            continue
        cells.append(dict(season=s, T=f["T"], pct=q, server="KR", pred=z[0], i80=z[1], i95=z[2], own=True))
    ins = {"F": a.f, "answer_key.json": os.path.join(ROOT, "data", "solo-raid", "analysis", "answer_key.json")}
    write(a.out, dict(about="Z0 — 앞 시즌 최종 컷 (T에 일정). 구간은 앞선 이웃 시즌 쌍의 log 비 분위 (사전 고정 정의)", method="Z0",
                      data_ver=a.data_ver, made=made(ins), cells=cells))


def cmd_lb(a):
    cells = []
    shas = []
    for s in A.TARGETS:
        for T in EA.t_grid(s):
            stp = EA.trunc_path(a.work, s, T)
            if stp is None:
                continue
            core = os.path.join(stp, "core.jsonl")
            shas.append(f"{s}/{T:g}:{SC.sha12(core)}")
            rows = A.load(stp, s, T, unknown=True)
            for q in LB_Q:
                v = A.lower_bound(rows, q, unknown=True)
                if v is not None:
                    cells.append(dict(season=s, T=float(T), pct=q, lb=v))
    ins = {"trunc_core.jsonl(모음)": hashlib.sha1("\n".join(shas).encode()).hexdigest()[:12], "n_trunc": str(len(shas))}
    write(a.out, dict(about="하한 — 한섭(라벨+mono 지난 미상) 줄 중 % − 0.005 > q인 가장 높은 점수. 모든 방법 공통", data_ver=a.data_ver,
                      made=made(ins), cells=cells))


def main():
    ap = argparse.ArgumentParser(description="기준선 예측 파일·하한 파일 만들기")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("f")
    p.add_argument("--main", action="append", required=True)
    p.add_argument("--ref", action="append", required=True)
    p = sub.add_parser("b0")
    p.add_argument("--cells", action="append", required=True)
    p.add_argument("--f", required=True)
    p = sub.add_parser("z0")
    p.add_argument("--f", required=True)
    p = sub.add_parser("lb")
    p.add_argument("--work", required=True)
    for p in sub.choices.values():
        p.add_argument("--out", required=True)
        p.add_argument("--data-ver", default=DATA_VER)
    a = ap.parse_args()
    {"f": cmd_f, "b0": cmd_b0, "z0": cmd_z0, "lb": cmd_lb}[a.cmd](a)


if __name__ == "__main__":
    main()
