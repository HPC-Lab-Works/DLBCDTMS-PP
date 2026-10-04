"""Location-privacy evaluation of pseudonym-change strategies against a global
passive eavesdropper that links pseudonyms with a kinematic (constant-velocity)
predictor. Output: results/privacy.json"""
import json, os
import numpy as np
from multiprocessing import Pool
import mobility

STRATS = {
    "Static pseudonym": dict(change=False),
    "Periodic (60 s)": dict(change=True, fixed=60, silent=(0, 0), ctx=False),
    "Random interval": dict(change=True, fixed=None, silent=(0, 0), ctx=False),
    "Random + silent period": dict(change=True, fixed=None, silent=(3, 8), ctx=False),
    "Random + silent + context": dict(change=True, fixed=None, silent=(3, 8), ctx=True),
    "DLBCDTMS-PP (cooperative, synchronised)": dict(change=True, fixed=None, silent=(3, 8), ctx=True, sync=True),
}


def one(args):
    seed, N, coverage = args
    rng = np.random.default_rng(seed)
    T = 600
    pos = mobility.manhattan(rng, N, T).astype(np.float64)
    vel = np.diff(pos, axis=0, prepend=pos[:1])
    obs_noise = 3.0
    # number of neighbours within 100 m (context trigger)
    nb = np.zeros((T + 1, N), np.int16)
    for t in range(T + 1):
        d = np.hypot(pos[t, :, None, 0] - pos[t, None, :, 0], pos[t, :, None, 1] - pos[t, None, :, 1])
        nb[t] = (d < 100).sum(1) - 1
    # eavesdropper stations cover a fraction of the area (1.0 = global adversary)
    side = 2000 * np.sqrt(coverage)
    seen = (pos[:, :, 0] <= side) & (pos[:, :, 1] <= side) if coverage < 1 else np.ones((T + 1, N), bool)
    out = {}
    for name, st in STRATS.items():
        if not st["change"]:
            out[name] = dict(success=1.0, anon=1.0, entropy=0.0, track_frac=1.0, silent_frac=0.0, changes=0)
            continue
        r2 = np.random.default_rng(seed + 99)
        changes = []  # (vehicle, t_last, t_new)
        silent_total = 0
        if st.get("sync"):
            # cooperative change: when a vehicle is due and has >= 3 neighbours within 100 m
            # (or has waited 30 s), it and every neighbour whose pseudonym is at least 30 s old
            # change together; each then stays silent for an independent random period.
            nxt = r2.uniform(0, 60, N).astype(int); last = np.full(N, -1000); wait = np.zeros(N, int)
            busy_until = np.full(N, -1)
            for t in range(T - 20):
                due = np.where((t >= nxt) & (t > busy_until))[0]
                r2.shuffle(due)
                for v in due:
                    if t < nxt[v] or t <= busy_until[v]:
                        continue
                    if nb[t, v] < 3 and wait[v] < 30:
                        wait[v] += 1; continue
                    d = np.hypot(pos[t, :, 0] - pos[t, v, 0], pos[t, :, 1] - pos[t, v, 1])
                    grp = np.where((d < 100) & ((t - last) >= 30) & (t > busy_until))[0]
                    grp = np.union1d(grp, [v])
                    for g in grp:
                        s_ = int(r2.uniform(*st["silent"]))
                        changes.append((g, t, t + s_ + 1)); silent_total += s_
                        busy_until[g] = t + s_; last[g] = t + s_ + 1
                        nxt[g] = t + s_ + 1 + int(r2.uniform(30, 120)); wait[g] = 0
        for v in (range(N) if not st.get("sync") else []):
            t = r2.uniform(0, 60) if st["fixed"] is None else r2.uniform(0, st["fixed"])
            t = int(t)
            while t < T - 20:
                if st["ctx"]:
                    w = 0
                    while t + w < T - 20 and nb[t + w, v] < 3 and w < 30:
                        w += 1
                    t += w
                s = int(r2.uniform(*st["silent"])) if st["silent"][1] > 0 else 0
                changes.append((v, t, t + s + 1)); silent_total += s
                gap = st["fixed"] if st["fixed"] else int(r2.uniform(30, 120))
                t = t + s + 1 + gap
        changes = np.array(changes)
        # appearances of new pseudonyms, indexed by time
        app_t = changes[:, 2]; app_v = changes[:, 0]
        succ = []; anon = []; ent = []
        for (v, t1, t2) in changes:
            if not seen[t1, v]:
                continue  # change not observed by the adversary
            p1 = pos[t1, v] + rng.normal(0, obs_noise, 2); v1 = vel[t1, v]
            m = (app_t > t1) & (app_t <= t1 + 15)
            cv = app_v[m]; ct = app_t[m]
            ok = seen[ct, cv]
            cv, ct = cv[ok], ct[ok]
            if cv.size == 0:
                succ.append(0); anon.append(0); ent.append(0); continue
            pred = p1[None, :] + v1[None, :] * (ct - t1)[:, None]
            obs = pos[ct, cv] + rng.normal(0, obs_noise, (cv.size, 2))
            err = np.hypot(*(obs - pred).T)
            sig = 10.0 + 8.0 * (ct - t1)
            gate = err < 3 * sig
            like = np.exp(-0.5 * (err / sig) ** 2) * gate
            k = int(gate.sum()); anon.append(k)
            if like.sum() > 0:
                p = like / like.sum(); p = p[p > 0]
                ent.append(float(-(p * np.log2(p)).sum()))
            else:
                ent.append(0.0)
            j = int(np.argmin(np.where(gate, err / sig, np.inf)))
            succ.append(int(gate[j] and cv[j] == v))
        succ = np.array(succ)
        # tracking fraction computed from per-vehicle success sequences
        order = {}
        k = 0
        for (v, t1, t2) in changes:
            if not seen[t1, v]:
                continue
            order.setdefault(v, []).append((t1, succ[k])); k += 1
        fr = []
        for v in range(N):
            seq = order.get(v, [])
            tt = T
            for t1, s_ in seq:
                if not s_:
                    tt = t1; break
            fr.append(tt / T)
        out[name] = dict(success=float(succ.mean()) if succ.size else 0.0,
                         anon=float(np.mean(anon)) if anon else 0.0,
                         entropy=float(np.mean(ent)) if ent else 0.0,
                         track_frac=float(np.mean(fr)),
                         silent_frac=silent_total / (N * T), changes=int(len(changes)))
    return dict(seed=seed, N=N, coverage=coverage, res=out)


def main():
    jobs = [(s, N, c) for s in range(10) for N in (250, 500, 1000) for c in (1.0,)] + \
           [(s, 500, c) for s in range(10) for c in (0.25, 0.5)]
    with Pool(2) as p:
        res = p.map(one, jobs)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "results", "privacy.json"), "w"), indent=1)
    agg = {}
    for r in res:
        for k, v in r["res"].items():
            agg.setdefault((r["N"], r["coverage"], k), []).append(v)
    for key, lst in sorted(agg.items()):
        print(key, {m: round(float(np.mean([x[m] for x in lst])), 3) for m in ("success", "anon", "entropy", "track_frac", "silent_frac")})


if __name__ == "__main__":
    main()
