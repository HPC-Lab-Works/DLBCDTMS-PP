"""Discrete-time simulator of event reporting, credibility assessment, feedback
and blockchain-based trust update in a VANET.

One run generates a single mobility trace, malicious set and event stream, and
feeds the *same* stream to every scheme (identical settings).
"""
import numpy as np
import mobility
import schemes as S

DEFAULT = dict(
    model="manhattan", N=500, ratio=0.3, rsu=9, T=600, seed=0, speed=None,
    Rc=300.0,          # DSRC communication range (m)
    Rs=200.0,          # sensing range for witnessing an event (m)
    Rrsu=500.0,        # RSU coverage (m)
    lam_ev=1.0,        # events per second
    p_report=0.7,      # probability that a witness reports
    eps=0.05,          # sensing error of honest vehicles
    p_attack=1.0,      # probability that an active attacker lies
    collusion=True,    # bad-mouthing and ballot-stuffing by colluding raters
    onoff=0,           # on-off period in seconds (0 = always on)
    sybil=0,           # extra forged identities per attacker report
    replay=0.0,        # probability that an attacker replays an old honest report
    impersonate=0.0,   # probability that an attacker forges a report under an honest pseudonym
    flood=1,           # copies of each malicious rating (rating-flood DoS)
    defend=True,       # enable the DLBCDTMS-PP countermeasures (signature, freshness,
                       # non-overlapping pseudonym validity, de-duplication and quota)
    Tb=5.0,            # Layer-1 block interval (s)
    schemes=None,      # list of scheme names (None = all)
    scheme_kw=None,    # {scheme name: keyword overrides} (calibrated parameters)
    return_scores=False,
)


def run(**kw):
    cfg = dict(DEFAULT); cfg.update(kw)
    rng = np.random.default_rng(cfg["seed"])
    N, T = cfg["N"], cfg["T"]
    pos = mobility.generate(cfg["model"], rng, N, T, cfg["speed"])
    rsus = mobility.rsu_positions(cfg["model"], cfg["rsu"])
    nm = int(round(cfg["ratio"] * N))
    mal = np.zeros(N, bool); mal[rng.choice(N, nm, replace=False)] = True
    ks = cfg["sybil"]
    n_ids = N + ks * N  # forged identities are indexed after the real vehicles
    owner = np.concatenate([np.arange(N)] + [np.arange(N)] * ks)  # owner of each identity
    is_mal_id = mal[owner]

    # RSU coverage and the next time each vehicle can upload feedback
    dmin = np.full((T + 1, N), np.inf, dtype=np.float32)
    near = np.zeros((T + 1, N), dtype=np.int16)
    for k, r in enumerate(rsus):
        dk = np.hypot(pos[:, :, 0] - r[0], pos[:, :, 1] - r[1])
        near = np.where(dk < dmin, k, near)
        dmin = np.minimum(dmin, dk)
    cov = dmin <= cfg["Rrsu"]
    nxt = np.full((T + 2, N), np.inf)
    for t in range(T, -1, -1):
        nxt[t] = np.where(cov[t], t, nxt[t + 1])

    names = cfg["schemes"] or [c.name for c in S.ALL]
    objs = []
    for c in S.ALL:
        if c.name in names:
            o = c(n_ids, N, **((cfg["scheme_kw"] or {}).get(c.name, {})))
            objs.append(o)
    own_w = {"DLBCDTMS-PP": 1.0, "BDTM": np.log(0.95 / 0.05)}

    Tb = cfg["Tb"]; nblk = int(np.ceil(T / Tb))
    hist = {o.name: {"hon": [], "mal": []} for o in objs}
    first_flag = {o.name: np.full(N, np.inf) for o in objs}
    on_since = {o.name: np.full(N, np.inf) for o in objs}   # time of the latest transition into "flagged"
    fpr_t = {o.name: [] for o in objs}
    dec_ok = {o.name: [0, 0] for o in objs}
    lat = []  # feedback-to-commit latency of the proposed scheme (s)
    old_reports = []  # (reporter, value, time) captured by attackers for replay
    ev_id = 0
    stats = dict(ratings=0, reports=0, dropped_invalid=0, events=0, rating_tx=0, l2_records=0)
    l2_pairs = {}  # block -> set of (rsu, reporter) aggregate records
    blk_done = 0

    for t in range(T):
        ne = rng.poisson(cfg["lam_ev"])
        P = pos[t]
        for _ in range(ne):
            anchor = rng.integers(N)
            loc = P[anchor] + rng.uniform(-50, 50, 2)
            truth = int(rng.random() < 0.5)
            dist = np.hypot(P[:, 0] - loc[0], P[:, 1] - loc[1])
            wit = np.where((dist <= cfg["Rs"]) & (rng.random(N) < cfg["p_report"]))[0]
            if wit.size == 0:
                continue
            ev_id += 1; stats["events"] += 1
            active = mal[wit].copy()
            if cfg["onoff"] > 0:
                active &= ((t // cfg["onoff"]) % 2 == 0)
            lie = active & (rng.random(wit.size) < cfg["p_attack"])
            err = (~mal[wit]) & (rng.random(wit.size) < cfg["eps"])
            s = np.where(lie | err, 1 - truth, truth).astype(np.int8)
            rep, sv, rd = list(wit), list(s), list(dist[wit])
            # observation of honest witnesses (used as their own evidence)
            obs = np.full(N, -1, np.int8); obs[wit] = s
            # --- attacks on the message layer -------------------------------
            att = wit[active]
            if ks and att.size:
                for k in range(1, ks + 1):
                    fid = att + k * N  # forged identities
                    if cfg["defend"]:
                        stats["dropped_invalid"] += fid.size  # no valid certificate in this slot
                    else:
                        rep += list(fid); sv += [1 - truth] * fid.size; rd += list(dist[att])
            honest_near = np.where((~mal) & (dist <= cfg["Rc"]))[0]
            if cfg["replay"] > 0 and att.size and old_reports:
                for a in att:
                    if rng.random() < cfg["replay"]:
                        cand = [o for o in old_reports[-200:] if o[1] != truth]
                        if not cand:
                            continue
                        v, val, ts = cand[rng.integers(len(cand))]
                        if cfg["defend"]:
                            stats["dropped_invalid"] += 1  # stale timestamp / nonce
                        else:
                            rep.append(v); sv.append(val); rd.append(dist[a])
            if cfg["impersonate"] > 0 and att.size and honest_near.size:
                for a in att:
                    if rng.random() < cfg["impersonate"]:
                        v = honest_near[rng.integers(honest_near.size)]
                        if cfg["defend"]:
                            stats["dropped_invalid"] += 1  # signature verification fails
                        else:
                            rep.append(v); sv.append(1 - truth); rd.append(dist[a])
            for v, val in zip(wit[~mal[wit]], s[~mal[wit]]):
                old_reports.append((v, val, t))
            if len(old_reports) > 2000:
                old_reports = old_reports[-1000:]
            rep = np.array(rep, dtype=np.int64); sv = np.array(sv, dtype=float); rd = np.array(rd)
            stats["reports"] += rep.size
            rcv = np.where(dist <= cfg["Rc"])[0]
            if rcv.size == 0:
                continue
            self_mask = (owner[rep][None, :] == rcv[:, None])  # a receiver never rates itself
            own_s = np.where(obs[rcv] >= 0, obs[rcv], 0).astype(float)
            is_w = (obs[rcv] >= 0) & (~mal[rcv])
            mal_r = mal[rcv]
            deliver = nxt[t + 1, rcv] if t + 1 <= T else np.full(rcv.size, np.inf)
            ok = np.isfinite(deliver)
            blk = np.floor(np.where(ok, deliver, 0) / Tb).astype(np.int64)  # committed at end of that block
            for o in objs:
                W = o.weights(rcv, rep, rd)
                W[self_mask] = 0
                ow = own_w.get(o.name, 1.0)
                D, _ = _decide(W, sv, np.where(is_w, ow, 0.0), own_s)
                hon = ~mal_r
                dec_ok[o.name][0] += int((D[hon] == truth).sum()); dec_ok[o.name][1] += int(hon.sum())
                Rm = np.where(sv[None, :] == D[:, None], 1, -1).astype(np.int8)
                if cfg["collusion"]:
                    Rm[mal_r] = np.where(is_mal_id[rep][None, :], 1, -1)
                else:
                    Rm[mal_r] = -Rm[mal_r]
                Rm[self_mask] = 0
                o.observe_local(*_flat(rcv, rep, Rm, np.ones(rcv.size, bool)))
                rw = o.rater_weight(rcv)
                if cfg["flood"] > 1:
                    mult = np.where(mal_r & ~np.bool_(cfg["defend"]), cfg["flood"], 1)
                    rw = rw * mult
                    if cfg["defend"] and o is objs[0]:
                        # duplicated rating transactions rejected by FASC de-duplication and quota
                        stats["dropped_invalid"] += int((cfg["flood"] - 1) * (mal_r & ok).sum())
                fw = getattr(o, "fin_window", None)
                if fw is not None:
                    # FASC finalises an event's feedback at t_event + fw; later feedback is stale
                    okk = ok & (deliver <= t + fw)
                    blk_o = np.where(okk, int(np.floor((t + fw) / Tb)), -1)
                else:
                    okk, blk_o = ok, blk
                for b in np.unique(blk_o[okk]):
                    sel = okk & (blk_o == b)
                    rr, pp, vv = _flat(rcv[sel], rep, Rm[sel], np.ones(sel.sum(), bool))
                    ww = np.repeat(rw[sel], rep.size)[(Rm[sel] != 0).ravel()]
                    o.push(int(b), rr, pp, vv, np.full(rr.size, ev_id), ww)
                    if o.name == "DLBCDTMS-PP":
                        stats["ratings"] += rr.size
                        stats["rating_tx"] += int(sel.sum())
                        tt = min(int(b * Tb), T)
                        ru = near[np.minimum(deliver[sel].astype(int), T), rcv[sel]]
                        l2_pairs.setdefault(int(b), set()).update(
                            (int(a) * 10_000_000 + int(p_)) for a in np.unique(ru) for p_ in rep)
                        lat.extend([(b + 1) * Tb - t] * int(sel.sum()))
        # block boundaries
        while (blk_done + 1) * Tb <= t + 1:
            for o in objs:
                o.commit(blk_done)
                sc = o.score()[:N]
                hist[o.name]["hon"].append(float(sc[~mal].mean()))
                hist[o.name]["mal"].append(float(sc[mal].mean()) if nm else np.nan)
                fl = o.flagged()[:N]
                ff = first_flag[o.name]
                ff[fl & np.isinf(ff)] = (blk_done + 1) * Tb
                os_ = on_since[o.name]
                os_[fl & np.isinf(os_)] = (blk_done + 1) * Tb
                os_[~fl] = np.inf
                fpr_t[o.name].append(float(fl[~mal].mean()))
            blk_done += 1

    stats["l2_records"] = int(sum(len(v) for v in l2_pairs.values()))
    out = {"cfg": {k: v for k, v in cfg.items() if k not in ("schemes", "scheme_kw", "return_scores")}, "stats": stats,
           "lat_mean": float(np.mean(lat)) if lat else np.nan,
           "lat_p95": float(np.percentile(lat, 95)) if lat else np.nan, "res": {}}
    for o in objs:
        sc = o.score()[:N]; fl = o.flagged()[:N]
        tp = int((fl & mal).sum()); fp = int((fl & ~mal).sum())
        fn = int((~fl & mal).sum()); tn = int((~fl & ~mal).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        auc = _auc(-sc, mal)
        ff = first_flag[o.name][mal]
        det = float(np.mean(np.where(np.isinf(ff), T, ff))) if nm else np.nan
        so = on_since[o.name][mal]
        det_stable = float(np.mean(np.where(np.isinf(so), T, so))) if nm else np.nan
        fpr_avg = float(np.mean(fpr_t[o.name]))
        out["res"][o.name] = dict(precision=prec, recall=rec, f1=f1, fpr=fpr, auc=auc,
                                  acc_dec=dec_ok[o.name][0] / max(dec_ok[o.name][1], 1),
                                  det_time=det, det_stable=det_stable, fpr_avg=fpr_avg, tp=tp, fp=fp, fn=fn, tn=tn,
                                  hist_hon=hist[o.name]["hon"], hist_mal=hist[o.name]["mal"])
        if cfg["return_scores"]:
            out["res"][o.name]["scores"] = sc.tolist()
            out["res"][o.name]["evidence"] = (o.a + o.b)[:N].tolist() if hasattr(o, "a") else None
    if cfg["return_scores"]:
        out["mal"] = mal.tolist()
    return out


def _flat(rcv, rep, Rm, mask):
    J, R = Rm.shape
    rr = np.repeat(rcv, R); pp = np.tile(rep, J); vv = Rm.ravel()
    keep = vv != 0
    return rr[keep], pp[keep], vv[keep]


def _decide(W, S_, own_w, own_s):
    return S._decide(W, S_, own_w, own_s)


def _auc(score, y):
    y = np.asarray(y, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(score)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


if __name__ == "__main__":
    import time, json
    t0 = time.time()
    o = run(N=500, seed=1)
    print("time", time.time() - t0, o["stats"], o["lat_mean"])
    for k, v in o["res"].items():
        print(k, {a: round(b, 3) for a, b in v.items() if not a.startswith("hist")})
