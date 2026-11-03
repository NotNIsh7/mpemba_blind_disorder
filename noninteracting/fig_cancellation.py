"""Fig. 1 (fig_concept.pdf), one heavy-tailed sample (L = 512, alpha = 1.25).
(a) deformed slow mode and the blind profile; (b) deformation times blind profile, whose area gives r."""

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blind_scaling import slow_mode

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
                     "mathtext.fontset": "cm", "savefig.dpi": 200})

L = 512
xs = (np.arange(L) + 0.5) / L
c1, c2 = np.cos(np.pi * xs), np.cos(2 * np.pi * xs)

# a sample with |r| close to 0.065
rng = np.random.default_rng(12)
best = None
for _ in range(200):
    rho = rng.uniform(1e-15, 1, L - 1) ** (-1 / 1.25)
    _, phi = slow_mode(1 / rho)
    phi = phi / ((c1 @ phi) / (c1 @ c1))
    r = (phi @ c2) / (phi @ c1)
    if 0.05 < abs(r) < 0.08 and (best is None or abs(abs(r) - 0.065) < abs(abs(best[1]) - 0.065)):
        best = (phi, r)
phi, r = best

plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7})
BLUE, GREEN, GREY = "#1f5fa8", "#1D9E75", "0.6"
fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.75))

a = ax[0]
a.plot(xs, c1, color=GREY, lw=1.6, alpha=0.6)
a.plot(xs, phi, color=BLUE, lw=0.9, label=r"slow mode $\phi_1$")
a.plot(xs, c2, color=GREEN, lw=1.1, ls="--", label="blind state")
a.set_ylim(-1.95, 1.35)
a.set_yticks([-1, 0, 1])
a.legend(frameon=False, loc="lower left", fontsize=6.3, handlelength=1.6,
         borderaxespad=0.2, labelspacing=0.25)
a.set_ylabel("density profile")

a = ax[1]
dphi = phi - c1
f = dphi * c2
a.fill_between(xs, 0, f, color="#D85A30", alpha=0.4, lw=0)
a.plot(xs, dphi, color="0.55", lw=0.8, label=r"$\phi_1-\cos\pi x$")
a.plot(xs, f, color="0.1", lw=0.7, label=r"$\times$ blind state")
a.text(0.5, 0.88, rf"$r={r:.3f}$", ha="center", transform=a.transAxes, fontsize=7)
lim = np.max(np.abs(dphi))
a.set_ylim(-2.0 * lim, 1.25 * lim)
a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
a.legend(frameon=False, loc="lower left", fontsize=6.3, handlelength=1.6,
         borderaxespad=0.2, labelspacing=0.25)
a.set_ylabel("deformation", labelpad=1)

for a, lab in zip(ax, "ab"):
    a.axhline(0, color="0.7", lw=0.5, zorder=0)
    a.set_xlabel(r"$x$", labelpad=1)
    a.set_title(f"({lab})", loc="left", fontsize=7.5, pad=2)
fig.tight_layout(w_pad=0.6, pad=0.2)
fig.savefig("fig_cancellation.pdf", bbox_inches="tight")
fig.savefig("fig_cancellation.png", bbox_inches="tight")
print(f"written; sample leftover r = {r:+.4f}")
