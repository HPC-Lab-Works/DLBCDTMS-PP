"""Aggregate the 30-seed runs: mean, standard deviation, 95% confidence interval,
paired Wilcoxon signed-rank tests (proposed vs. each baseline, Holm-corrected),
figures (PDF) and LaTeX tables for the manuscript."""
import json, os, collections
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(HERE, "tables"); os.makedirs(TAB, exist_ok=True)
PROP = "DLBCDTMS-PP"
ORDER = [PROP, "DLBTM", "BDTM", "BBC", "LTM", "BIBM"]
# validated categorical palette + secondary encoding (markers, dashes)
COL = dict(zip(ORDER, ["#C0392B", "#0072B2", "#E69F00", "#56B4E9", "#009E73", "#8E44AD"]))
MRK = dict(zip(ORDER, ["o", "s", "^", "v", "D", "P"]))
LS = dict(zip(ORDER, ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42, "font.family": "serif"})

runs = [json.loads(l) for l in open(os.path.join(R, "runs.jsonl"))]
D = collections.defaultdict(lambda: collections.defaultdict(dict))  # (exp,val) -> scheme -> seed -> res
STATS = collections.defaultdict(dict)
for o in runs:
    for s, r in o["res"].items():
        D[(o["exp"], o["val"])][s][o["seed"]] = r
    STATS[(o["exp"], o["val"])][o["seed"]] = dict(o["stats"], lat_mean=o["lat_mean"], lat_p95=o["lat_p95"])


def ms(exp, val, s, m):
    x = np.array([D[(exp, val)][s][k][m] for k in sorted(D[(exp, val)][s])], float)
    n = x.size; mu = x.mean(); sd = x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return mu, sd, h, x


def wilcoxon_holm(exp, val, m, better="greater"):
    """Paired two-sided Wilcoxon signed-rank tests of the proposed scheme vs each
    baseline across seeds; Holm correction within the family of baselines."""
    _, _, _, xp = ms(exp, val, PROP, m)
    ps = {}
    for s in ORDER[1:]:
        if s not in D[(exp, val)]:
            continue
        _, _, _, xb = ms(exp, val, s, m)
        d = xp - xb
        if np.allclose(d, 0):
            ps[s] = 1.0
        else:
            ps[s] = stats.wilcoxon(xp, xb, zero_method="zsplit").pvalue
    keys = sorted(ps, key=ps.get); k = len(keys); adj = {}; run = 0
    for i, s in enumerate(keys):
        run = max(run, min(1.0, (k - i) * ps[s])); adj[s] = run
    return adj


def fmt(mu, sd, digits=3):
    return f"{mu:.{digits}f} $\\pm$ {sd:.{digits}f}"


# ---------------------------------------------------------------- figures
def sweep_fig(exp, vals, xlabel, metrics, fname, xticks=None, logx=False):
    fig, axs = plt.subplots(1, len(metrics), figsize=(7.16, 2.0))
    for ax, (m, lab) in zip(axs, metrics):
        for s in ORDER:
            if s not in D[(exp, vals[0])]:
                continue
            mu = []; h = []
            for v in vals:
                a, _, c, _ = ms(exp, v, s, m); mu.append(a); h.append(c)
            ax.errorbar(vals, mu, yerr=h, color=COL[s], marker=MRK[s], ls=LS[s], lw=1.6 if s == PROP else 1.0,
                        ms=4, capsize=2, label=s, zorder=3 if s == PROP else 2)
        ax.set_xlabel(xlabel); ax.set_ylabel(lab)
        if logx: ax.set_xscale("log")
        if xticks is not None: ax.set_xticks(vals); ax.set_xticklabels(xticks)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), bbox_inches="tight"); plt.close(fig)


ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
sweep_fig("ratio", ratios, "Malicious vehicles (%)", [("precision", "Precision"), ("recall", "Recall"),
          ("f1", "F-measure"), ("fpr", "False positive rate")], "fig_ratio.pdf",
          xticks=["10", "20", "30", "40", "50"])
sweep_fig("scale", [100, 250, 500, 1000], "Number of vehicles", [("f1", "F-measure"), ("auc", "AUC"),
          ("det_stable", "Detection time (s)"), ("acc_dec", "Decision accuracy")], "fig_scale.pdf")
sweep_fig("speed", [5, 10, 20, 30], "Mean vehicle speed (m/s)", [("f1", "F-measure"), ("recall", "Recall"),
          ("det_stable", "Detection time (s)"), ("acc_dec", "Decision accuracy")], "fig_speed.pdf")


def trust_fig():
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3))
    for ax, key, title in ((axs[0], ("ratio", 0.3), "Persistent attackers with collusion (30%)"),
                           (axs[1], ("attack", "On-off (60 s)"), "On-off attackers (30%)")):
        for s in ORDER:
            H = np.array([D[key][s][k]["hist_hon"] for k in sorted(D[key][s])])
            M = np.array([D[key][s][k]["hist_mal"] for k in sorted(D[key][s])])
            t = (np.arange(H.shape[1]) + 1) * 5
            ax.plot(t, H.mean(0), color=COL[s], ls="-", lw=1.4 if s == PROP else 0.9)
            ax.plot(t, M.mean(0), color=COL[s], ls="--", lw=1.4 if s == PROP else 0.9)
        ax.set_title(title, fontsize=8); ax.set_xlabel("Time (s)"); ax.set_ylabel("Mean trust score")
        ax.set_ylim(0, 1.02)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=COL[s], lw=1.4) for s in ORDER] + \
         [Line2D([], [], color="k", ls="-"), Line2D([], [], color="k", ls="--")]
    fig.legend(hs, ORDER + ["honest vehicles", "malicious vehicles"], loc="upper center", ncol=8,
               bbox_to_anchor=(0.5, 1.08), fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_trust_time.pdf"), bbox_inches="tight"); plt.close(fig)


trust_fig()

# ---------------------------------------------------------------- tables
MET = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F-measure"), ("fpr", "FPR"), ("auc", "AUC"),
       ("acc_dec", "Dec. acc."), ("det_stable", "Det. time (s)"), ("fpr_avg", "Mean FPR")]


def table_group(exp, vals, labels, caption, label, metrics=MET):
    lines = [r"\begin{table*}[!t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{ll" + "c" * len(metrics) + "}", r"\toprule",
             "Setting & Scheme & " + " & ".join(m[1] for m in metrics) + r" \\", r"\midrule"]
    for v, lab in zip(vals, labels):
        tests = {m: wilcoxon_holm(exp, v, m) for m, _ in metrics}
        best = {}
        for m, _ in metrics:
            vals_ = {s: ms(exp, v, s, m)[0] for s in ORDER if s in D[(exp, v)]}
            best[m] = (min if m in ("fpr", "det_time", "det_stable", "fpr_avg") else max)(vals_.values())
        for i, s in enumerate([x for x in ORDER if x in D[(exp, v)]]):
            cells = []
            for m, _ in metrics:
                mu, sd, h, _ = ms(exp, v, s, m)
                d = 1 if m in ("det_time", "det_stable") else 3
                c = f"{mu:.{d}f}$\\pm${sd:.{d}f}"
                if abs(mu - best[m]) < 1e-9:
                    c = r"\textbf{" + c + "}"
                if s != PROP and tests[m].get(s, 1) >= 0.05:
                    c += r"$^{\dagger}$"
                cells.append(c)
            first = (r"\multirow{%d}{*}{%s}" % (len([x for x in ORDER if x in D[(exp, v)]]), lab)) if i == 0 else ""
            lines.append(f"{first} & {s} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(TAB, label.replace("tab:", "") + ".tex"), "w").write("\n".join(lines))


table_group("ratio", ratios, ["10\\%", "20\\%", "30\\%", "40\\%", "50\\%"],
            r"Detection performance versus the ratio of malicious vehicles (500 vehicles, urban grid, 9 RSUs, persistent attackers with collusion). Mean$\pm$standard deviation over 30 seeds; bold marks the best mean; $^{\dagger}$ marks baselines whose difference from DLBCDTMS-PP is \emph{not} significant (paired Wilcoxon signed-rank test, Holm-corrected, $\alpha=0.05$).",
            "tab:ratio")
table_group("mobility", ["manhattan", "highway", "rwp"], ["Urban grid", "Highway", "Random waypoint"],
            r"Detection performance under different mobility models (500 vehicles, 30\% malicious). Notation as in Table~\ref{tab:ratio}.",
            "tab:mobility", metrics=MET[:6] + [MET[7]])
table_group("rsu", [4, 9, 16], ["4 RSUs", "9 RSUs", "16 RSUs"],
            r"Detection performance under different RSU densities (urban grid, 500 vehicles, 30\% malicious). Notation as in Table~\ref{tab:ratio}.",
            "tab:rsu", metrics=[MET[2], MET[3], MET[4], MET[6], MET[7]])
table_group("attack", ["Persistent, no collusion", "Persistent + collusion", "Selective (p=0.5)", "On-off (60 s)"],
            ["No collusion", "Collusion", "Selective", "On-off"],
            r"Detection performance under different attacker behaviours (500 vehicles, 30\% malicious). Notation as in Table~\ref{tab:ratio}.",
            "tab:attack", metrics=[MET[0], MET[1], MET[2], MET[3], MET[4], MET[6], MET[7]])
table_group("scale", [100, 250, 500, 1000], ["100", "250", "500", "1000"],
            r"Scalability: detection performance versus the number of vehicles (urban grid, 9 RSUs, 30\% malicious). Notation as in Table~\ref{tab:ratio}.",
            "tab:scale", metrics=[MET[2], MET[3], MET[4], MET[5], MET[6], MET[7]])

# security ablation (countermeasures on/off)
sec_rows = []
for name in ("Sybil", "Replay", "Impersonation", "Rating flood (DoS)"):
    row = [name]
    for d in ("on", "off"):
        v = f"{name}|{d}"
        for m in ("f1", "fpr", "acc_dec"):
            mu, sd, _, _ = ms("security", v, PROP, m); row.append(f"{mu:.3f}$\\pm${sd:.3f}")
    xo = ms("security", f"{name}|on", PROP, "f1")[3]; xf = ms("security", f"{name}|off", PROP, "f1")[3]
    p = stats.wilcoxon(xo, xf, zero_method="zsplit").pvalue if not np.allclose(xo, xf) else 1.0
    row.append(f"{p:.1e}" if p < 0.001 else f"{p:.3f}")
    dropped = np.mean([STATS[("security", f"{name}|on")][k]["dropped_invalid"] for k in STATS[("security", f"{name}|on")]])
    row.append(f"{dropped:.0f}")
    sec_rows.append(row)
lines = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Security evaluation of DLBCDTMS-PP with its countermeasures enabled (on) and disabled (off) for each attack (500 vehicles, 30\% malicious, 30 seeds). $p$: paired Wilcoxon test on the F-measure (on vs.\ off). Rejected: mean number of malicious messages rejected per run by the countermeasure.}",
         r"\label{tab:security}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lcccccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{Countermeasures on} & \multicolumn{3}{c}{Countermeasures off} & & \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Attack & F-measure & FPR & Dec. acc. & F-measure & FPR & Dec. acc. & $p$ & Rejected \\", r"\midrule"]
lines += [" & ".join(r) + r" \\" for r in sec_rows]
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(os.path.join(TAB, "security.tex"), "w").write("\n".join(lines))

# ablation and sensitivity
lines = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Ablation and parameter sensitivity of DLBCDTMS-PP (500 vehicles, 30\% malicious, 30 seeds). F-measure and AUC as mean$\pm$standard deviation.}",
         r"\label{tab:ablation}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Persistent + collusion} & \multicolumn{2}{c}{On-off (60 s)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Variant & F-measure & AUC & F-measure & AUC \\", r"\midrule"]
for exp, vals, pref in (("ablation", ["Full model", "Rater weight = T only", "Rater weight = F only", "No distance weighting"], ""),
                        ("sens_eta", ["1.0", "1.5", "2.0", "3.0"], r"$\eta=$"),
                        ("sens_lam", ["0.95", "0.98", "0.99", "1.0"], r"$\lambda=$")):
    for v in vals:
        cells = []
        for sc in ("persistent", "on-off"):
            key = f"{v}|{sc}" if exp == "ablation" else f"{float(v)}|{sc}"
            for m in ("f1", "auc"):
                mu, sd, _, _ = ms(exp, key, PROP, m); cells.append(f"{mu:.3f}$\\pm${sd:.3f}")
        lab = v.replace("T only", r"$T$ only").replace("F only", r"$F$ only") if exp == "ablation" else pref + v
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table}"]
open(os.path.join(TAB, "ablation.tex"), "w").write("\n".join(lines))

# load for the consensus study (scale and RSU sweeps)
load = []
for exp, vals, keyname in (("scale", [100, 250, 500, 1000], "N"), ("rsu", [4, 9, 16], "rsu")):
    for v in vals:
        st = STATS[(exp, v)]
        row = dict(N=v if keyname == "N" else 500, rsu=v if keyname == "rsu" else 9, T=600)
        for k in ("ratings", "rating_tx", "l2_records", "reports", "events"):
            row[k] = float(np.mean([st[s][k] for s in st]))
        row["lat_mean"] = float(np.mean([st[s]["lat_mean"] for s in st]))
        row["lat_p95"] = float(np.mean([st[s]["lat_p95"] for s in st]))
        load.append(row)
for v in (5, 10, 20, 30):
    st = STATS[("speed", v)]
    load.append(dict(speed=v, lat_mean=float(np.mean([st[s]["lat_mean"] for s in st])),
                     lat_p95=float(np.mean([st[s]["lat_p95"] for s in st]))))
json.dump([r for r in load if "N" in r], open(os.path.join(R, "load.json"), "w"), indent=1)
json.dump(load, open(os.path.join(R, "load_all.json"), "w"), indent=1)

# compact numeric summary for writing the text
summ = {}
for (exp, val), d in D.items():
    for s in d:
        summ[f"{exp}|{val}|{s}"] = {m: round(ms(exp, val, s, m)[0], 4) for m in ("precision", "recall", "f1", "fpr", "auc", "acc_dec", "det_time", "det_stable", "fpr_avg")}
        summ[f"{exp}|{val}|{s}"].update({m + "_sd": round(ms(exp, val, s, m)[1], 4) for m in ("f1", "fpr")})
summ["tests_ratio"] = {str(v): {m: wilcoxon_holm("ratio", v, m) for m in ("f1", "fpr", "auc", "recall", "precision")} for v in ratios}
summ["tests_scale"] = {str(v): {m: wilcoxon_holm("scale", v, m) for m in ("f1", "auc")} for v in [100, 250, 500, 1000]}
summ["tests_mob"] = {str(v): {m: wilcoxon_holm("mobility", v, m) for m in ("f1", "auc")} for v in ["manhattan", "highway", "rwp"]}
summ["tests_rsu"] = {str(v): {m: wilcoxon_holm("rsu", v, m) for m in ("f1", "auc")} for v in [4, 9, 16]}
summ["tests_speed"] = {str(v): {m: wilcoxon_holm("speed", v, m) for m in ("f1",)} for v in [5, 10, 20, 30]}
summ["tests_attack"] = {v: {m: wilcoxon_holm("attack", v, m) for m in ("f1", "auc")} for v in
                        ["Persistent, no collusion", "Persistent + collusion", "Selective (p=0.5)", "On-off (60 s)"]}
json.dump(summ, open(os.path.join(R, "summary.json"), "w"), indent=1)
print("done", len(runs))
