import sys, os, json, time
from multiprocessing import Pool
WK = sys.argv[1]
sys.path.insert(0, os.path.join(WK, 'tools/solo_raid'))
os.chdir(WK)
import backtest as BT, xval as XV, fit_prior_multi as FM
WORK = sys.argv[2]
SEAS = [f"S{n}" for n in range(19, 42)]
TS = [6, 12, 18, 24, 30, 36, 48, 60, 72, 84, 96, 108]
def job(a):
    s, T = a
    st = FM.state_of(s)
    t = time.time()
    try:
        BT.trunc_state(st, float(T), os.path.join(WORK, 'trunc', s), XV.extra_of(st))
        return (s, T, round(time.time() - t, 1), 'ok')
    except BaseException as e:
        return (s, T, round(time.time() - t, 1), repr(e)[:300])
if __name__ == '__main__':
    jobs = [(s, T) for T in TS for s in SEAS]
    with Pool(4) as p:
        for r in p.imap_unordered(job, jobs):
            print(*r, flush=True)
