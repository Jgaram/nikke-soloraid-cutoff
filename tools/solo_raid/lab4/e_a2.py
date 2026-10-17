import sys; sys.path.insert(0, sys.argv[0].rsplit('/',1)[0]); import lab as L
P41 = ("file", L.WK + "/data/solo-raid/S41/state/prior.json")
T22 = [t for t in L.SEAS if t != "S41"]
own = L.run("A2_own", [("own", t, ("single", t), (t,)) for t in L.SEAS])
ownT = L.run("A3_ownT_41shape", [("ownT", t, ("hyb", ("single", t), P41, P41), ("S41",)) for t in T22])
ownS = L.run("A4_41T_ownshape", [("ownS", t, ("hyb", P41, ("single", t), P41), ("S41",)) for t in T22])
own41b = L.run("A5_own_borrow41", [("own_b41", t, ("single", t), ("S41",)) for t in T22])
a1 = L.run("A1_S41op", None)
for name, r in [("S41 운영", a1), ("자기 시즌 전부(상한)", [x for x in own if x['season'] != 'S41']), ("자기 % 변환 + S41 모양", ownT), ("S41 % 변환 + 자기 모양", ownS), ("자기 prior + S41 비율 빌림", own41b)]:
    L.show(name, r)
