"""SM figures: fig_profile.pdf (paper: fig_profiles.pdf), the resistance profile u(x) - x and
the slow-mode deformation; fig_law.pdf (paper: fig_ks.pdf), (a) d against omega,
(b) Kolmogorov law, (c) heavy tails: omega against the largest bond's share."""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstwobign

from blind_scaling import slow_mode

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 7.3, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.linewidth": 0.7, "mathtext.fontset": "cm", "savefig.dpi": 200})

L = 2048
xs = (np.arange(L) + 0.5) / L
xb = np.arange(1, L) / L
c1 = np.cos(np.pi * xs)
cases = [("(a) resistance spread over many bonds", lambda r, m: 1 + r.uniform(0, 1, m),
          ["#1f5fa8", "#5b8fd1", "#9dbde6"]),
         (r"(b) a few bonds dominate ($\alpha=1.25$)",
          lambda r, m: r.uniform(1e-15, 1, m) ** (-1 / 1.25), ["#b33a1e", "#d9704f", "#eba68e"])]
fig, ax = plt.subplots(2, 2, figsize=(6.8, 3.9), sharex=True)
for j, (title, smp, cols) in enumerate(cases):
    rng = np.random.default_rng(3 + j)
    for s in range(3):
        rho = smp(rng, L - 1)
        u = np.cumsum(rho) / rho.sum()
        _, phi = slow_mode(1 / rho)
        phi = phi / ((c1 @ phi) / (c1 @ c1))
        ax[0, j].plot(xb, u - xb, color=cols[s], lw=0.9)
        ax[1, j].plot(xs, phi - c1, color=cols[s], lw=0.9)
    ax[0, j].set_title(title, loc="left")
    for a in ax[:, j]:
        a.axhline(0, color="0.7", lw=0.5, zorder=0)
    ax[1, j].set_xlabel(r"$x$")
ax[0, 0].set_ylabel(r"$u(x)-x$")
ax[1, 0].set_ylabel(r"$\phi_1-\cos\pi x$")
fig.tight_layout()
fig.savefig("fig_profile.pdf", bbox_inches="tight")
fig.savefig("fig_profile.png", bbox_inches="tight")

d = json.load(open("random_positions.json"))
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
rs = np.random.default_rng(0)

a = ax[0]
for xi, col, lab in ((6.0, "#1f5fa8", r"$\alpha=3$"), (3.0, "#7a1fa8", r"$\alpha=1.5$"),
                     (2.5, "#b33a1e", r"$\alpha=1.25$")):
    for L, mk, al in ((256, "o", 0.35), (8192, "s", 0.6)):
        r = d[f"positions|{xi}|{L}"]
        dd, om = np.array(r["d"]), np.array(r["omega"])
        i = rs.choice(len(dd), size=min(120, len(dd)), replace=False)
        a.loglog(om[i], dd[i], mk, ms=2.4, color=col, alpha=al, mew=0,
                 label=lab if L == 8192 else None)
lim = [3e-3, 1.5]
a.plot(lim, lim, "k--", lw=0.8)
a.set_xlim(*lim); a.set_ylim(*lim)
a.set_xlabel(r"$\omega=\max_x|u(x)-x|$")
a.set_ylabel(r"distortion $d$")
a.set_title("(a) one law for all tails", loc="left")
a.legend(frameon=False, loc="upper left", markerscale=2)
a.text(0.97, 0.05, "circles $L=256$\nsquares $L=8192$", transform=a.transAxes,
       ha="right", fontsize=6.8, color="0.35")

a = ax[1]
rng = np.random.default_rng(3)
Lk, vals = 4096, []
for _ in range(6000):
    rho = 1 + rng.uniform(0, 1, Lk - 1); rb = rho.mean()
    om = np.max(np.abs(np.cumsum(rho - rb))) / (Lk * rb)
    vals.append(np.sqrt(Lk) * om * rb / np.std(rho))
a.hist(vals, bins=45, density=True, color="#9dbde6", edgecolor="none", label="chain, $L=4096$")
z = np.linspace(0.2, 2.2, 300)
a.plot(z, kstwobign.pdf(z), "k-", lw=1.1, label="Kolmogorov")
a.set_xlabel(r"$\sqrt{L}\,\omega\,\langle\rho\rangle/\sigma_\rho$")
a.set_ylabel("density")
a.set_title("(b) many bonds: Kolmogorov law", loc="left")
a.legend(frameon=False, loc="upper right")

a = ax[2]
for xi, col, lab in ((6.0, "#1f5fa8", r"$\alpha=3$"), (2.5, "#b33a1e", r"$\alpha=1.25$")):
    r = d[f"positions|{xi}|8192"]
    om, sh = np.array(r["omega"]), np.array(r["share"])
    a.loglog(sh, om, "o", ms=2.4, color=col, alpha=0.5, mew=0, label=lab)
lim = [4e-4, 1.5]
a.plot(lim, lim, "k--", lw=0.8)
a.set_xlim(*lim); a.set_ylim(1e-3, 1.5)
a.set_xlabel(r"worst bond's share $\rho_{\max}/\sum\rho$")
a.set_ylabel(r"$\omega$")
a.set_title("(c) one bond: its share", loc="left")
a.legend(frameon=False, loc="upper left", markerscale=2)

fig.tight_layout(w_pad=0.9)
fig.savefig("fig_law.pdf", bbox_inches="tight")
fig.savefig("fig_law.png", bbox_inches="tight")
print("written")
