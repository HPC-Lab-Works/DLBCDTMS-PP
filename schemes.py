"""Trust schemes evaluated under identical event, report and mobility streams.

Every scheme implements
    weights(rcv, rep, d)          -> per-receiver report weights (|J| x |R|)
    rater_weight(rcv)             -> credibility of raters
    ingest(block, rcv, rep, r, ev, w)  rating transactions delivered in a block
    commit(block)                 -> apply the block (trust update), at block end
    score()                       -> trust score per identity (higher = more trustworthy)
    flagged()                     -> boolean malicious classification per identity

DLBCDTMS-PP is the proposed scheme. The baselines are behavioural abstractions of
the core trust logic described in the cited papers (see Table "Baseline
abstractions" in the manuscript); they are not the authors' original code.
"""
import numpy as np


def _decide(W, S, own_w, own_s, rule="weighted"):
    """Return the binary decision of each receiver.
    W: |J| x |R| weights (0 for the receiver's own report), S: |R| reports in {0,1}.
    own_w/own_s: receiver's own observation weight and value (0 if not a witness)."""
    num = W @ S + own_w * own_s
    den = W.sum(1) + own_w
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(den > 0, num / den, 0.5)
    return (c >= 0.5).astype(np.int8), c


class Base:
    name = "base"
    rule = "weighted"
    theta = 0.5

    def __init__(self, n_ids, N):
        self.n, self.N = n_ids, N
        self.pending = {}

    def push(self, blk, rcv, rep, r, ev, w):
        self.pending.setdefault(blk, []).append((rcv, rep, r, ev, w))

    def pop(self, blk):
        items = self.pending.pop(blk, [])
        if not items:
            return None
        return [np.concatenate([it[k] for it in items]) for k in range(5)]

    def rater_weight(self, rcv):
        return np.ones(rcv.size)

    def observe_local(self, rcv, rep, r):
        pass

    def flagged(self):
        return self.score() < self.theta


class DLBCDTMS(Base):
    """Proposed: distance- and trust-weighted credibility, rater-credibility-weighted
    feedback normalised per (reporter, event), forgetting-factor beta reputation
    with asymmetric penalty (Eqs. in Section 4)."""
    name = "DLBCDTMS-PP"

    def __init__(self, n_ids, N, gamma=2.0, d0=150.0, lam=0.98, eta=2.0, theta=0.5, nmin=2.0, rmode="TF", fin_window=10.0):
        super().__init__(n_ids, N)
        self.fin_window = fin_window
        self.g, self.d0, self.lam, self.eta, self.theta, self.nmin = gamma, d0, lam, eta, theta, nmin
        self.a = np.zeros(n_ids); self.b = np.zeros(n_ids)
        self.c = np.zeros(n_ids); self.e = np.zeros(n_ids)   # feedback agreement / disagreement
        self.T = np.full(n_ids, 0.5); self.F = np.full(n_ids, 0.5)
        self.rmode = rmode

    def weights(self, rcv, rep, d):
        w = (self.T[rep] ** self.g) * np.exp(-d / self.d0)
        return np.broadcast_to(w, (rcv.size, rep.size)).copy()

    def rater_weight(self, rcv):
        if self.rmode == "T":
            return self.T[rcv] ** self.g
        if self.rmode == "F":
            return self.F[rcv] ** self.g
        return (self.T[rcv] * self.F[rcv]) ** self.g

    def commit(self, blk):
        self.a *= self.lam; self.b *= self.lam; self.c *= self.lam; self.e *= self.lam
        x = self.pop(blk)
        if x is not None:
            rcv, rep, r, ev, w = x
            # normalise rater weights per (reporter, event)
            key = rep.astype(np.int64) * 1_000_003 + ev
            uk, inv = np.unique(key, return_inverse=True)
            tot = np.bincount(inv, weights=w)
            wn = w / np.maximum(tot[inv], 1e-12)
            # credibility-weighted consensus rating of each (reporter, event)
            cons = np.sign(np.bincount(inv, weights=wn * r))
            agree = (np.sign(r) == cons[inv]) & (cons[inv] != 0)
            self.c += np.bincount(rcv, weights=agree.astype(float), minlength=self.n)
            self.e += np.bincount(rcv, weights=(~agree & (cons[inv] != 0)).astype(float), minlength=self.n)
            self.a += np.bincount(rep, weights=wn * (r > 0), minlength=self.n)
            self.b += self.eta * np.bincount(rep, weights=wn * (r < 0), minlength=self.n)
        self.T = (self.a + 1.0) / (self.a + self.b + 2.0)
        self.F = (self.c + 1.0) / (self.c + self.e + 2.0)

    def score(self):
        return self.T

    def flagged(self):
        return (self.T < self.theta) & (self.a + self.b >= self.nmin)


class DLBTM(Base):
    """Ruan et al. (2023) abstraction: double-layer chain; trust from a logistic
    function of the vehicle's evaluation history; raters weighted by the quality of
    their past evaluation behaviour; no forgetting, symmetric penalty."""
    name = "DLBTM"

    def __init__(self, n_ids, N, k=4.0, c=2.0, theta=0.5):
        super().__init__(n_ids, N)
        self.theta = theta
        self.k, self.c = k, c
        self.pos = np.zeros(n_ids); self.neg = np.zeros(n_ids)
        self.agree = np.zeros(n_ids); self.tot = np.zeros(n_ids)
        self.T = np.full(n_ids, 0.5)

    def weights(self, rcv, rep, d):
        return np.broadcast_to(self.T[rep], (rcv.size, rep.size)).copy()

    def rater_weight(self, rcv):
        return (self.agree[rcv] + 1) / (self.tot[rcv] + 2)

    def commit(self, blk):
        x = self.pop(blk)
        if x is not None:
            rcv, rep, r, ev, w = x
            key = rep.astype(np.int64) * 1_000_003 + ev
            uk, inv = np.unique(key, return_inverse=True)
            maj = np.sign(np.bincount(inv, weights=r.astype(float)))
            ag = (np.sign(r) == maj[inv]).astype(float)
            self.agree += np.bincount(rcv, weights=ag, minlength=self.n)
            self.tot += np.bincount(rcv, minlength=self.n)
            self.pos += np.bincount(rep, weights=w * (r > 0), minlength=self.n)
            self.neg += np.bincount(rep, weights=w * (r < 0), minlength=self.n)
        z = self.k * (self.pos - self.neg) / (self.pos + self.neg + self.c)
        self.T = 1.0 / (1.0 + np.exp(-z))

    def score(self):
        return self.T


class BDTM(Base):
    """Yang et al. (2019) abstraction: Bayesian-inference message credibility using
    the reporters' trust, unweighted +1/-1 ratings, RSU trust offsets added to the
    trust value (no rater weighting, no forgetting)."""
    name = "BDTM"

    def __init__(self, n_ids, N, kappa=0.1, theta=0.5):
        super().__init__(n_ids, N)
        self.theta = theta
        self.kappa = kappa
        self.T = np.full(n_ids, 0.5)

    def weights(self, rcv, rep, d):
        p = 0.5 + 0.45 * self.T[rep]
        llr = np.log(p / (1 - p))
        return np.broadcast_to(llr, (rcv.size, rep.size)).copy()

    def commit(self, blk):
        x = self.pop(blk)
        if x is not None:
            rcv, rep, r, ev, w = x
            s = np.bincount(rep, weights=r.astype(float), minlength=self.n)
            c = np.bincount(rep, minlength=self.n)
            off = np.where(c > 0, s / np.maximum(c, 1), 0.0)
            self.T = np.clip(self.T + self.kappa * off, 0, 1)

    def score(self):
        return self.T


class BBC(Base):
    """Alharthi et al. (2021) abstraction: identity-centric biometric blockchain;
    messages accepted by unweighted majority; reputation initialised at 1 and
    incremented or decremented by the net feedback ratio."""
    name = "BBC"

    def __init__(self, n_ids, N, delta=0.05, theta=0.5):
        super().__init__(n_ids, N)
        self.theta = theta
        self.delta = delta
        self.T = np.ones(n_ids)

    def weights(self, rcv, rep, d):
        return np.ones((rcv.size, rep.size))

    def commit(self, blk):
        x = self.pop(blk)
        if x is not None:
            rcv, rep, r, ev, w = x
            s = np.bincount(rep, weights=r.astype(float), minlength=self.n)
            c = np.bincount(rep, minlength=self.n)
            self.T = np.clip(self.T + self.delta * np.where(c > 0, s / np.maximum(c, 1), 0), 0, 1)

    def score(self):
        return self.T


class LTM(Base):
    """Junejo et al. (2020) abstraction: lightweight model combining a receiver's
    direct experience with RSU-provided indirect trust (equal weights); centralised
    RSU aggregation without consensus or rater weighting."""
    name = "LTM"

    def __init__(self, n_ids, N, theta=0.5):
        super().__init__(n_ids, N)
        self.theta = theta
        self.dp = np.zeros((N, n_ids), dtype=np.float32); self.dn = np.zeros((N, n_ids), dtype=np.float32)
        self.P = np.zeros(n_ids); self.Nn = np.zeros(n_ids)
        self.I = np.full(n_ids, 0.5)

    def weights(self, rcv, rep, d):
        D = (self.dp[np.ix_(rcv, rep)] + 1) / (self.dp[np.ix_(rcv, rep)] + self.dn[np.ix_(rcv, rep)] + 2)
        return 0.5 * D + 0.5 * self.I[rep][None, :]

    def observe_local(self, rcv, rep, r):
        np.add.at(self.dp, (rcv[r > 0], rep[r > 0]), 1)
        np.add.at(self.dn, (rcv[r < 0], rep[r < 0]), 1)

    def commit(self, blk):
        x = self.pop(blk)
        if x is not None:
            rcv, rep, r, ev, w = x
            self.P += np.bincount(rep, weights=(r > 0).astype(float), minlength=self.n)
            self.Nn += np.bincount(rep, weights=(r < 0).astype(float), minlength=self.n)
        self.I = (self.P + 1) / (self.P + self.Nn + 2)

    def score(self):
        return self.I


class BIBM(Base):
    """Ali et al. (2022, VABLOCK) abstraction: blockchain-recorded broadcast with
    unweighted majority validation; the ledger keeps one verdict per (reporter,
    event) obtained from the majority of received feedback."""
    name = "BIBM"

    def __init__(self, n_ids, N, theta=0.5):
        super().__init__(n_ids, N)
        self.theta = theta
        self.pos = np.zeros(n_ids); self.neg = np.zeros(n_ids)
        self.T = np.full(n_ids, 0.5)

    def weights(self, rcv, rep, d):
        return np.ones((rcv.size, rep.size))

    def commit(self, blk):
        x = self.pop(blk)
        if x is not None:
            rcv, rep, r, ev, w = x
            key = rep.astype(np.int64) * 1_000_003 + ev
            uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
            net = np.bincount(inv, weights=r.astype(float))
            reps = rep[first]
            self.pos += np.bincount(reps, weights=(net > 0).astype(float), minlength=self.n)
            self.neg += np.bincount(reps, weights=(net < 0).astype(float), minlength=self.n)
        self.T = (self.pos + 1) / (self.pos + self.neg + 2)

    def score(self):
        return self.T


ALL = [DLBCDTMS, DLBTM, BDTM, BBC, LTM, BIBM]
