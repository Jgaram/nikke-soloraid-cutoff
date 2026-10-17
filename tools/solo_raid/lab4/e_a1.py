import sys; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); import lab as L, json, collections
P41 = ("file", L.WK + "/data/solo-raid/S41/state/prior.json")
plan = [("fixedS41", t, P41, ("S41",)) for t in L.SEAS if t != "S41"]
rows = L.run("A1_S41op", plan, nproc=2)
L.show("S41 운영 prior", rows)
# 기존 결과와 맞대기
old = {}
for fn in ["xval_fixedS41.json", "test37.json"]:
    for r in json.load(open(L.WK + "/data/solo-raid/analysis/" + fn))["rows"]:
        old[(r["season"], r["T"], r["server"], r["pct"])] = r["pred"]
d = [abs(r["pred"] / old[k] - 1) for r in rows for k in [(r["season"], r["T"], r["server"], r["pct"])] if k in old]
print("기존과 같은 칸", len(d), "최대 차", max(d), "차 > 0.1%", sum(x > 0.001 for x in d))
