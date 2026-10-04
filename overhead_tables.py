"""LaTeX tables and figures for overhead, smart contracts, consensus and privacy."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results"); TAB = os.path.join(HERE, "tables"); FIG = os.path.join(HERE, "figures")
C = json.load(open(os.path.join(R, "crypto.json")))
G = json.load(open(os.path.join(R, "gas.json")))
K = json.load(open(os.path.join(R, "consensus.json")))
P = json.load(open(os.path.join(R, "privacy.json")))
L = json.load(open(os.path.join(R, "load_all.json")))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42, "font.family": "serif"})


def w(name, lines):
    open(os.path.join(TAB, name), "w").write("\n".join(lines))


# ---- cryptographic costs and security overhead (before and after security operations)
sz = K["sizes"]
sign, ver = C["ecdsa_sign"]["mean_ms"], C["ecdsa_verify"]["mean_ms"]
enc, dec = C["ecies_enc"]["mean_ms"], C["ecies_dec"]["mean_ms"]
aes = C["aesgcm_enc_200B"]["mean_ms"]; sha = C["sha256_200B"]["mean_ms"]
rows = [("ECDSA P-256 signature generation", C["ecdsa_sign"]), ("ECDSA P-256 signature verification", C["ecdsa_verify"]),
        ("ECIES encryption of TPID (ECDH + HKDF + AES-GCM)", C["ecies_enc"]), ("ECIES decryption of TPID", C["ecies_dec"]),
        ("AES-128-GCM encryption, 200 bytes", C["aesgcm_enc_200B"]), ("SHA-256, 200 bytes", C["sha256_200B"])]
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Measured Cost of Cryptographic Operations (Python \texttt{cryptography} with OpenSSL, one core of an Intel Xeon at 2.1\,GHz; 5\% trimmed mean$\pm$standard deviation)}",
         r"\label{tab:crypto}", r"\scriptsize", r"\begin{tabular}{lr}", r"\toprule", r"Operation & Time ($\mu$s) \\", r"\midrule"]
for n, d in rows:
    lines.append(f"{n} & {1e3*d['mean_ms']:.2f}$\\pm${1e3*d['std_ms']:.2f}" + r" \\")
pw = C["pow"]
lines += [r"\midrule", r"\multicolumn{2}{l}{Proof-of-work for comparison (mean time per block, one core):} \\"]
for d in ("16", "18"):
    lines.append(f"\\quad difficulty {d} bits & {1e6*pw[d]['mean_s']:.0f}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("crypto.tex", lines)

# before / after security operations
plain_report, plain_rating = 48, 33 + 9  # unsigned payloads (no signature, certificate, encryption)
cap = 1e3 / (ver + dec)  # rating transactions per second per RSU core
default = [r for r in L if r.get("N") == 500 and r.get("rsu") == 9][0]
tx_rsu = default["rating_tx"] / 600 / 9
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Cost of the Security Operations: Without and With Signatures, Certificates and Encryption (500 vehicles, 9 RSUs)}",
         r"\label{tab:beforeafter}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{p{4.6cm}rr}", r"\toprule", r"Quantity & Without & With \\", r"\midrule",
         f"Report size (bytes) & {plain_report} & {sz['report']}" + r" \\",
         f"Rating transaction size, 10 rated reports (bytes) & {plain_rating + 9*9} & {sz['rating_hdr'] + 10*sz['rating_item']}" + r" \\",
         f"Sender processing per report ($\\mu$s) & 0 & {1e3*sign:.1f}" + r" \\",
         f"Receiver processing per report ($\\mu$s) & 0 & {1e3*ver:.1f}" + r" \\",
         f"Vehicle processing per rating transaction ($\\mu$s) & 0 & {1e3*(sign+enc):.1f}" + r" \\",
         f"RSU processing per rating transaction ($\\mu$s) & 0 & {1e3*(ver+dec):.1f}" + r" \\",
         f"RSU capacity (rating transactions per second per core) & -- & {cap:,.0f}".replace(",", "\\,") + r" \\",
         f"Measured load per RSU (rating transactions per second) & {tx_rsu:.1f} & {tx_rsu:.1f}" + r" \\",
         r"Precision, recall, F-measure and FPR without message-layer attacks & \multicolumn{2}{c}{identical} \\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("beforeafter.tex", lines)

# ---- gas
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Gas Consumption of the Smart Contracts (Solidity 0.8.24, optimiser enabled, py-evm)}",
         r"\label{tab:gas}", r"\scriptsize", r"\begin{tabular}{llr}", r"\toprule", r"Contract & Function & Gas \\", r"\midrule",
         f"Registry & deployment & {G['deploy_Registry']:,}" + r" \\",
         f"Registry & \\texttt{{registerVehicle}} & {G['registerVehicle']:,.0f}" + r" \\",
         f"Registry & \\texttt{{revokeVehicle}} & {G['revokeVehicle']:,}" + r" \\",
         f"FASC & deployment & {G['deploy_FASC']:,}" + r" \\",
         f"FASC & \\texttt{{submitRating}} (single) & {G['submitRating']:,.0f}" + r" \\",
         f"FASC & \\texttt{{submitBatch}}, per rating (batch of 100) & {G['submitBatch_100_per_rating']:,.0f}" + r" \\",
         f"TESC & deployment & {G['deploy_TESC']:,}" + r" \\",
         f"TESC & \\texttt{{commitEpoch}}, per vehicle (batch of 100) & {G['commitEpoch_update_100_per_vehicle']:,.0f}" + r" \\",
         f"TESC & \\texttt{{commitEpoch}}, per vehicle (batch of 200) & {G['commitEpoch_update_200_per_vehicle']:,.0f}" + r" \\",
         f"TESC & \\texttt{{trustOf}} (read-only call) & {G['trustOf_estimate']:,}" + r" \\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
w("gas.tex", [x.replace(",", "\\,") if "&" in x and "Gas" not in x else x for x in lines])

# ---- consensus figure: PBFT latency vs M and block size
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.2))
cols = ["#0072B2", "#E69F00", "#009E73", "#C0392B"]
mk = ["o", "s", "^", "D"]
for c, m_, ntx in zip(cols, mk, (100, 500, 1000, 2000)):
    rows_ = [r for r in K["pbft_vs_M"] if r["ntx"] == ntx]
    axs[0].errorbar([r["M"] for r in rows_], [r["lat_ms"] for r in rows_], yerr=[r["std_ms"] for r in rows_],
                    color=c, marker=m_, ms=4, lw=1.2, capsize=2, label=f"{ntx} tx per block")
axs[0].set_xlabel("Number of RSUs $M$"); axs[0].set_ylabel("PBFT commit latency (ms)")
axs[0].legend(fontsize=7)
lay = [r for r in K["layers"] if r["M"] == 9]
x = np.arange(len(lay)); wd = 0.38
axs[1].bar(x - wd / 2, [r["single_storage_MB_h"] for r in lay], wd, color="#7f8c8d", label="Single-layer (all ratings on the replicated chain)")
axs[1].bar(x + wd / 2, [r["double_storage_MB_h_L2"] for r in lay], wd, color="#C0392B", label="Double-layer (Layer-2 replica)")
axs[1].set_xticks(x); axs[1].set_xticklabels([str(r["N"]) for r in lay])
axs[1].set_xlabel("Number of vehicles ($M=9$)"); axs[1].set_ylabel("Replicated storage per RSU (MB/h)")
axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_consensus.pdf"), bbox_inches="tight"); plt.close(fig)

lines = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Single-Layer Versus Double-Layer Ledger Organisation (measured load from the simulations, $T_b=5$\,s, PBFT among $M$ RSUs; storage and backhaul per hour)}",
         r"\label{tab:layers}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
         r" & & & & \multicolumn{3}{c}{Single-layer} & \multicolumn{3}{c}{Double-layer (proposed)} \\",
         r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r"Vehicles & RSUs & Rating tx/s & Ratings/s & Latency (ms) & Storage per replica (MB) & Backhaul (MB) & Latency (ms) & Layer-2 storage per replica (MB) & Backhaul (MB) \\", r"\midrule"]
for r in K["layers"]:
    lines.append(f"{r['N']} & {r['M']} & {r['tx_per_s']:.1f} & {r['ratings_per_s']:.0f} & {r['single_lat_ms']:.1f} & {r['single_storage_MB_h']:.1f} & {r['single_backhaul_MB_h']:.0f} & {r['double_lat_ms']:.1f} & {r['double_storage_MB_h_L2']:.2f} & {r['double_backhaul_MB_h']:.1f}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
w("layers.tex", lines)

# ---- privacy
agg = {}
for r in P:
    for k, v in r["res"].items():
        agg.setdefault((r["N"], r["coverage"], k), []).append(v)
names = list(P[0]["res"].keys())
lines = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Location Privacy Against a Pseudonym-Linking Eavesdropper (urban grid, 600\,s, 10 runs; mean$\pm$standard deviation). Success: fraction of observed pseudonym changes that the adversary links correctly. Anonymity set: number of candidate pseudonyms inside the adversary's gate. Entropy: entropy of the adversary's linking distribution. Tracked: mean fraction of the trace followed before the first linking error. Silent: fraction of time without beacons.}",
         r"\label{tab:privacy}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llccccc}", r"\toprule",
         r"Setting & Strategy & Success & Anonymity set & Entropy (bits) & Tracked & Silent \\", r"\midrule"]
for (N, cov) in [(250, 1.0), (500, 1.0), (1000, 1.0), (500, 0.5), (500, 0.25)]:
    first = True
    for n in names:
        L_ = agg[(N, cov, n)]
        f = lambda m: (np.mean([x[m] for x in L_]), np.std([x[m] for x in L_], ddof=1))
        s_, a_, e_, t_, q_ = f("success"), f("anon"), f("entropy"), f("track_frac"), f("silent_frac")
        lab = (f"\\multirow{{{len(names)}}}{{*}}{{\\shortstack[l]{{{N} vehicles\\\\coverage {int(cov*100)}\\%}}}}") if first else ""
        first = False
        lines.append(f"{lab} & {n} & {s_[0]:.3f}$\\pm${s_[1]:.3f} & {a_[0]:.2f} & {e_[0]:.2f} & {t_[0]:.3f} & {q_[0]:.3f}" + r" \\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table*}"]
w("privacy.tex", lines)
print("ok")
