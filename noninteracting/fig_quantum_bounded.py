"""SM figure fig_quantum_bounded.pdf: exact quantum dynamics with bounded random fields.
(a) distortion d and residual |r| against L at gamma = 1 (highstat.json);
(b) exponents of d and |r| across dephasing rates (gammascan.json)."""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 7.3, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.linewidth": 0.7, "mathtext.fontset": "cm", "savefig.dpi": 200})

hs = json.load(open("highstat.json")); Lh = np.array(sorted(int(k) for k in hs))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
a = ax[0]
for key, col, lab in (("d", "#1f5fa8", r"distortion $d$"), ("r2", "#1D9E75", r"blind $\cos2\pi x$: $|r|$"),
                      ("rrand", "#b33a1e", r"random blind state: $|r|$")):
    y = np.array([abs(hs[str(L)][key]["mean"]) for L in Lh]); e = np.array([hs[str(L)][key]["sem"] for L in Lh])
    s = np.polyfit(np.log(Lh), np.log(y), 1)[0]
    a.errorbar(Lh, y, yerr=e, fmt="o-", ms=3.5, lw=1.0, color=col, capsize=1.5, label=f"{lab} ({s:+.2f})")
a.set_xscale("log"); a.set_yscale("log")
a.set_xticks([24, 48, 96, 192]); a.set_xticklabels(["24", "48", "96", "192"])
a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xlabel(r"$L$"); a.set_ylabel("mean over samples")
a.set_title(r"(a) full quantum dynamics, $\gamma=1$", loc="left"); a.legend(frameon=False, loc="lower left")
a = ax[1]
gs = json.load(open("gammascan.json"))
gam = sorted(gs, key=float)
for key, col, lab in (("d", "#1f5fa8", "distortion $d$"), ("r2", "#1D9E75", r"blind $\cos2\pi x$")):
    ex, er = [], []
    for gm in gam:
        Lk = sorted(gs[gm], key=int); Lx = np.log([int(k) for k in Lk])
        yy = np.array([abs(gs[gm][k][key]) for k in Lk]); ee = np.array([gs[gm][k][key + "_e"] for k in Lk])
        ex.append(np.polyfit(Lx, np.log(yy), 1)[0]); rb = np.random.default_rng(2)
        er.append(np.std([np.polyfit(Lx, np.log(np.maximum(yy + rb.normal(0, ee), 1e-15)), 1)[0] for _ in range(1500)]))
    a.errorbar([float(g) for g in gam], ex, yerr=er, fmt="o-", ms=3.5, lw=1.0, color=col, capsize=1.5, label=lab)
a.axhline(-0.5, color="k", ls="--", lw=0.8, label=r"$-1/2$")
a.set_xscale("log"); a.set_ylim(-0.75, -0.25)
a.set_xlabel(r"dephasing rate $\gamma$"); a.set_ylabel(r"exponent over $L=24$--$96$")
a.set_title("(b) independent of dephasing", loc="left"); a.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("fig_quantum_bounded.pdf", bbox_inches="tight")
print("fig_quantum_bounded.pdf written")
