"""Calibrate the free parameters and the classification threshold of every scheme
on a disjoint set of calibration seeds (1000-1009) in two scenarios (persistent
attackers and on-off attackers, 30% malicious, 500 vehicles, urban grid).
The objective is the mean F1 score. Output: results/calibration.json"""
import itertools, json, os
import numpy as np
from multiprocessing import Pool
import simulator

GRID = {
    "DLBCDTMS-PP": [dict(eta=e, lam=l, gamma=g) for e, l, g in itertools.product([1.0, 1.5, 2.0], [0.98, 0.99, 1.0], [1.0, 2.0])],
    "DLBTM": [dict(k=k) for k in (2.0, 4.0, 8.0)],
    "BDTM": [dict(kappa=k) for k in (0.05, 0.1, 0.2)],
    "BBC": [dict(delta=d) for d in (0.02, 0.05, 0.1)],
    "LTM": [dict()],
    "BIBM": [dict()],
}
THETAS = np.round(np.arange(0.05, 0.96, 0.01), 2)
SEEDS = range(1000, 1010)
SCEN = [dict(), dict(onoff=60)]


def job(args):
    name, kw, sc, seed = args
    o = simulator.run(seed=seed, schemes=[name], scheme_kw={name: kw}, return_scores=True, **sc)
    r = o["res"][name]
    mal = np.array(o["mal"]); s = np.array(r["scores"])
    ev = np.array(r["evidence"]) if r["evidence"] is not None else np.full(s.size, np.inf)
    f1s = []
    for th in THETAS:
        fl = (s < th) & (ev >= 2.0)
        tp = (fl & mal).sum(); fp = (fl & ~mal).sum(); fn = (~fl & mal).sum()
        p = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * p * rc / (p + rc) if p + rc else 0)
    return name, json.dumps(kw, sort_keys=True), f1s


def main():
    jobs = [(n, kw, sc, sd) for n, kws in GRID.items() for kw in kws for sc in SCEN for sd in SEEDS]
    acc = {}
    with Pool(2) as p:
        for name, kw, f1s in p.imap_unordered(job, jobs, chunksize=4):
            acc.setdefault(name, {}).setdefault(kw, []).append(f1s)
    best = {}
    for name, d in acc.items():
        cand = []
        for kw, lst in d.items():
            m = np.mean(lst, 0); i = int(np.argmax(m))
            cand.append((m[i], kw, float(THETAS[i])))
        cand.sort(reverse=True)
        f, kw, th = cand[0]
        best[name] = dict(json.loads(kw), theta=th, cal_f1=float(f))
        print(name, best[name])
    os.makedirs("results", exist_ok=True)
    json.dump(best, open("results/calibration.json", "w"), indent=1)


if __name__ == "__main__":
    main()
