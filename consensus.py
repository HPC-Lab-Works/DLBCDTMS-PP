"""Discrete-event emulation of PBFT consensus among M RSUs on the Layer-2 chain,
using the measured cryptographic costs (results/crypto.json), and comparison of
single-layer versus double-layer ledger organisations.
Output: results/consensus.json"""
import heapq, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "results", "crypto.json")))
T_SIGN = C["ecdsa_sign"]["mean_ms"] / 1e3
T_VER = C["ecdsa_verify"]["mean_ms"] / 1e3
T_HASH = C["sha256_200B"]["mean_ms"] / 1e3

# message and transaction sizes in bytes (IEEE 1609.2-style fields)
SIG, CERT_DIGEST, ECIES_CT, TS, NONCE, HID = 64, 8, 113, 8, 8, 8
REPORT_MSG = 32 + TS + NONCE + SIG + 125              # V2V event report with explicit certificate
RATING_TX_HDR = ECIES_CT + 8 + TS + NONCE + SIG + CERT_DIGEST   # batched rating tx header
RATING_ITEM = HID + 1                                  # per rated report
L2_RECORD = 32 + 16 + 16                               # TPID, evidence (P, N), agreement counts (A, B)
L2_TX_HDR = 8 + 8 + 32 + SIG                           # RSU id, epoch, Merkle root, signature
PBFT_MSG = 8 + 8 + 32 + SIG                            # view, seq, digest, signature
BLOCK_HDR = 32 + 32 + 8 + 8 + SIG


def pbft_latency(M, block_bytes, n_tx, rng, link_ms=2.0, jitter_ms=1.0, bw_gbps=1.0):
    """Return the time (s) until a correct replica commits a block (PBFT normal case)."""
    f = (M - 1) // 3
    bw = bw_gbps * 1e9 / 8
    q = 2 * f + 1
    busy = np.zeros(M)                 # CPU availability time of each replica
    ev = []                            # (time, kind, dst, src)
    prep = [set() for _ in range(M)]; comm = [set() for _ in range(M)]
    sent_commit = [False] * M; committed = [np.inf] * M

    def link():
        return (link_ms + rng.uniform(0, jitter_ms)) / 1e3

    # primary 0 assembles, hashes and signs the block, then multicasts PRE-PREPARE
    t0 = n_tx * T_HASH + T_SIGN
    for j in range(1, M):
        heapq.heappush(ev, (t0 + block_bytes / bw * j + link(), "pp", j, 0))  # serial uplink
    busy[0] = t0
    # primary also sends PREPARE implicitly (counted as its pre-prepare)
    for j in range(M):
        prep[j].add(0)

    def multicast(t, kind, src):
        for j in range(M):
            if j != src:
                heapq.heappush(ev, (t + PBFT_MSG / bw + link(), kind, j, src))

    while ev:
        t, kind, j, src = heapq.heappop(ev)
        if kind == "pp":
            # verify block: signature of primary, transaction signatures, Merkle hashing
            start = max(t, busy[j]); busy[j] = start + T_VER + n_tx * (T_VER + T_HASH) + T_SIGN
            prep[j].add(j)
            multicast(busy[j], "prep", j)
        elif kind == "prep":
            start = max(t, busy[j]); busy[j] = start + T_VER
            prep[j].add(src)
        elif kind == "comm":
            start = max(t, busy[j]); busy[j] = start + T_VER
            comm[j].add(src)
        if len(prep[j]) >= q and not sent_commit[j]:
            sent_commit[j] = True; busy[j] += T_SIGN; comm[j].add(j)
            multicast(busy[j], "comm", j)
        if len(comm[j]) >= q and np.isinf(committed[j]):
            committed[j] = busy[j]
    c = np.array(committed); c = c[np.isfinite(c)]
    return float(np.median(c)), int(M + 2 * M * (M - 1))


def main():
    rng = np.random.default_rng(7)
    sims = json.load(open(os.path.join(HERE, "results", "load.json")))
    out = {"sizes": dict(report=REPORT_MSG, rating_hdr=RATING_TX_HDR, rating_item=RATING_ITEM,
                         l2_record=L2_RECORD, l2_tx_hdr=L2_TX_HDR, pbft_msg=PBFT_MSG),
           "pbft_vs_M": [], "layers": []}
    # PBFT latency versus the number of RSUs and the block size
    for M in (4, 9, 16, 25, 36):
        for ntx in (100, 500, 1000, 2000):
            lats = [pbft_latency(M, BLOCK_HDR + ntx * 250, ntx, rng)[0] for _ in range(20)]
            out["pbft_vs_M"].append(dict(M=M, ntx=ntx, lat_ms=1e3 * float(np.mean(lats)),
                                         std_ms=1e3 * float(np.std(lats)),
                                         msgs=M + 2 * M * (M - 1),
                                         tps_capacity=ntx / float(np.mean(lats))))
    # single-layer versus double-layer organisation, using the measured load
    Tb = 5.0
    for row in sims:
        N, M = row["N"], row["rsu"]
        dur = row["T"]; nblk = dur / Tb
        tx_per_blk = row["rating_tx"] / nblk
        items_per_blk = row["ratings"] / nblk
        rec_per_blk = row["l2_records"] / nblk
        single_bytes = tx_per_blk * RATING_TX_HDR + items_per_blk * RATING_ITEM
        l2_tx = min(M, max(1, rec_per_blk))  # one commit transaction per RSU per epoch
        double_bytes = l2_tx * L2_TX_HDR + rec_per_blk * L2_RECORD
        lat_s = np.mean([pbft_latency(M, BLOCK_HDR + single_bytes, int(tx_per_blk), rng)[0] for _ in range(10)])
        lat_d = np.mean([pbft_latency(M, BLOCK_HDR + double_bytes, int(l2_tx), rng)[0] for _ in range(10)])
        hour = 3600 / Tb
        out["layers"].append(dict(
            N=N, M=M, tx_per_s=tx_per_blk / Tb, ratings_per_s=items_per_blk / Tb,
            single_lat_ms=1e3 * lat_s, double_lat_ms=1e3 * lat_d,
            single_storage_MB_h=hour * (BLOCK_HDR + single_bytes) / 1e6,       # per replica
            double_storage_MB_h_L2=hour * (BLOCK_HDR + double_bytes) / 1e6,      # per replica
            double_storage_MB_h_L1=hour * (M * BLOCK_HDR + single_bytes) / M / 1e6,  # per RSU (own zone)
            single_backhaul_MB_h=hour * (M - 1) * (BLOCK_HDR + single_bytes) / 1e6,
            double_backhaul_MB_h=hour * (M - 1) * (BLOCK_HDR + double_bytes) / 1e6,
        ))
    json.dump(out, open(os.path.join(HERE, "results", "consensus.json"), "w"), indent=1)
    for r in out["pbft_vs_M"]:
        print(r)
    for r in out["layers"]:
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()})


if __name__ == "__main__":
    main()
