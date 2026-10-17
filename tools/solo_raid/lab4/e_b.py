import sys; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); import lab as L
S = L.SEAS
full = [s for s in S if L.full_ok(s)]
print("full", len(full), "shape_only", [s for s in S if not L.full_ok(s)])
def prev(t):   # 직전 full 시즌
    c = [s for s in full if L.sn(s) < L.sn(t)]
    return c[-1] if c else None
tg = [t for t in S if L.sn(t) >= 21]
fw = {t: tuple(s for s in S if L.sn(s) < L.sn(t)) for t in tg}
r5 = {t: tuple([s for s in S if L.sn(s) < L.sn(t)][-5:]) for t in tg}
lo = {t: tuple(s for s in S if s != t) for t in S}
specs = [("multi", v) for v in list(fw.values()) + list(r5.values()) + list(lo.values())]
L.prefit(specs)
