"""Run all evaluation experiments with 30 independent seeds (0-29) using the
parameters calibrated on seeds 1000-1009. Output: results/runs.jsonl"""
import json, os, sys
from multiprocessing import Pool
import simulator

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = json.load(open(os.path.join(HERE, "results", "calibration.json")))
KW = {k: {a: b for a, b in v.items() if a != "cal_f1"} for k, v in CAL.items()}
SEEDS = range(30)
PROP = "DLBCDTMS-PP"


def exps():
    E = []
    for r in (0.1, 0.2, 0.3, 0.4, 0.5):
        E.append(("ratio", r, dict(ratio=r)))
    for n in (100, 250, 500, 1000):
        E.append(("scale", n, dict(N=n)))
    for m in ("manhattan", "highway", "rwp"):
        E.append(("mobility", m, dict(model=m)))
    for k in (4, 9, 16):
        E.append(("rsu", k, dict(rsu=k)))
    for v in (5, 10, 20, 30):
        E.append(("speed", v, dict(speed=float(v))))
    E.append(("attack", "Persistent, no collusion", dict(collusion=False)))
    E.append(("attack", "Persistent + collusion", dict()))
    E.append(("attack", "Selective (p=0.5)", dict(p_attack=0.5)))
    E.append(("attack", "On-off (60 s)", dict(onoff=60)))
    for name, kw in (("Sybil", dict(sybil=3)), ("Replay", dict(replay=0.5)),
                     ("Impersonation", dict(impersonate=0.5)), ("Rating flood (DoS)", dict(flood=10))):
        for d in (True, False):
            E.append(("security", f"{name}|{'on' if d else 'off'}", dict(kw, defend=d, schemes=[PROP])))
    abl = {"Full model": {}, "Rater weight = T only": dict(rmode="T"),
           "Rater weight = F only": dict(rmode="F"), "No distance weighting": dict(d0=1e9)}
    for sc_name, sc in (("persistent", {}), ("on-off", dict(onoff=60))):
        for an, akw in abl.items():
            E.append(("ablation", f"{an}|{sc_name}", dict(sc, schemes=[PROP], prop_kw=akw)))
        for eta in (1.0, 1.5, 2.0, 3.0):
            E.append(("sens_eta", f"{eta}|{sc_name}", dict(sc, schemes=[PROP], prop_kw=dict(eta=eta))))
        for lam in (0.95, 0.98, 0.99, 1.0):
            E.append(("sens_lam", f"{lam}|{sc_name}", dict(sc, schemes=[PROP], prop_kw=dict(lam=lam))))
    return E


def job(a):
    exp, val, cfg, seed = a
    cfg = dict(cfg)
    kw = {k: dict(v) for k, v in KW.items()}
    if "prop_kw" in cfg:
        kw[PROP].update(cfg.pop("prop_kw"))
    o = simulator.run(seed=seed, scheme_kw=kw, **cfg)
    keep_hist = (exp == "ratio" and val in (0.3,)) or (exp == "attack" and val == "On-off (60 s)")
    for k in o["res"]:
        if not keep_hist:
            o["res"][k].pop("hist_hon"); o["res"][k].pop("hist_mal")
    o["exp"], o["val"], o["seed"] = exp, val, seed
    return json.dumps(o)


def main():
    only = sys.argv[1:] or None
    jobs = [(e, v, c, s) for e, v, c in exps() for s in SEEDS if (only is None or e in only)]
    # long jobs first
    jobs.sort(key=lambda j: -(j[2].get("N", 500)) * (1 if "schemes" in j[2] else 6))
    out = os.path.join(HERE, "results", "runs.jsonl")
    mode = "a" if only else "w"
    with Pool(2) as p, open(out, mode) as fh:
        for i, line in enumerate(p.imap_unordered(job, jobs, chunksize=2)):
            fh.write(line + "\n")
            if i % 100 == 0:
                print(i, "/", len(jobs), flush=True)


if __name__ == "__main__":
    main()
