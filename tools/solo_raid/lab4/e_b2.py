import sys; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); import lab as L
S = L.SEAS
full = [s for s in S if L.full_ok(s)]
def prev(t):
    c = [s for s in full if L.sn(s) < L.sn(t)]
    return c[-1] if c else None
tg = [t for t in S if L.sn(t) >= 21]
fw = {t: tuple(s for s in S if L.sn(s) < L.sn(t)) for t in tg}
r5 = {t: tuple([s for s in S if L.sn(s) < L.sn(t)][-5:]) for t in tg}
lo = {t: tuple(s for s in S if s != t) for t in S}
which = sys.argv[1].split(",") if len(sys.argv) > 1 else ["M0", "M2a", "M2b", "loso"]
popts = None
suffix = ""
if len(sys.argv) > 2 and sys.argv[2] == "mupd":
    popts = {"m_update": {}}; suffix = "_mupd"
plans = {
  "M0": [("M0", t, ("single", prev(t)), (prev(t),)) for t in tg],
  "M2a": [("M2a", t, ("multi", r5[t]), r5[t]) for t in tg],
  "M2b": [("M2b", t, ("multi", fw[t]), fw[t]) for t in tg],
  "loso": [("loso", t, ("multi", lo[t]), lo[t]) for t in S],
}
for k in which:
    rows = L.run("B_" + k + suffix, plans[k], popts=popts)
    L.show(k + suffix, rows)
    L.show(k + suffix + "+M1", L.calibrate(rows, loso=(k == "loso")))
