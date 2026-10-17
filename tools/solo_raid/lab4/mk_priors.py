import sys, os, json, time
from multiprocessing import Pool
WK = sys.argv[1]; OUT = sys.argv[2]; BOOT = int(sys.argv[3]) if len(sys.argv) > 3 else 60
sys.path.insert(0, os.path.join(WK, 'tools/solo_raid'))
os.chdir(WK)
import fit_prior as FP, fit_prior_multi as FM, predict as PR
SEAS = [f"S{n}" for n in range(19, 42)]
def job(s):
    p = os.path.join(OUT, f"{s}.json")
    if os.path.exists(p):
        return s, 0, 'cached'
    t = time.time()
    try:
        pr = FP.fit(FM.state_of(s), boot=BOOT, seed=1, quiet=True)
        pr = json.loads(json.dumps(PR.jsonable(pr)))
        json.dump(pr, open(p, 'w'), ensure_ascii=False)
        return s, round(time.time() - t, 1), 'ok'
    except BaseException as e:
        import traceback
        return s, round(time.time() - t, 1), traceback.format_exc()[-600:]
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    with Pool(int(os.environ.get('NP', '2'))) as p:
        for r in p.imap_unordered(job, SEAS):
            print(*r, flush=True)
