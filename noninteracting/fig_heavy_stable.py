"""SM figure fig_heavy_stable.pdf: stable-law statistics of r for heavy-tailed hoppings.
(a) N^(1 - 1/alpha) r at L = 1024 against the symmetric stable law;
(b), (c) win fractions and median reach (fig_mpemba.json) against the prediction."""

import json

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import integrate, special, stats

from blind_scaling import slow_mode, bonds_from_z
from fig_mpemba import r_of_profile

ALPHA = 1.25
Z = 1.0 / (2 * ALPHA)


def constants(alpha=ALPHA):
    """mu, sigma_alpha, ||g||_alpha for Pareto resistances with rho_min = 1 (scale-free ratios)."""
    mu = alpha / (alpha - 1)
    sig = (special.gamma(1 - alpha) * np.cos(np.pi * alpha / 2)) ** (1 / alpha)
    g = lambda x: 4 / 3 * np.sin(np.pi * x) * np.sin(2 * np.pi * x)
    gnorm = integrate.quad(lambda x: abs(g(x)) ** alpha, 0, 1, limit=400)[0] ** (1 / alpha)
    return mu, sig, gnorm


def zk(N, kappa, corrected, alpha=ALPHA):
    mu, sig, gnorm = constants(alpha)
    mplus = stats.levy_stable.ppf(0.5, alpha, 1.0) if corrected else 0.0
    return (mu * np.asarray(N, float) ** (1 - 1 / alpha) + sig * mplus) / (kappa * sig * gnorm)


def pwin(N, kappa, corrected):
    return 2 * stats.levy_stable.cdf(zk(N, kappa, corrected), ALPHA, 0.0) - 1


def reach(N, corrected):
    return zk(N, 1.0, corrected) / stats.levy_stable.ppf(0.75, ALPHA, 0.0)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
                         "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.linewidth": 0.7, "mathtext.fontset": "cm", "savefig.dpi": 200})
    mu, sig, gnorm = constants()
    s_a = sig * gnorm / mu
    col = "#b33a1e"

    # (a) distribution at L = 1024
    L = 1024; N = L - 1
    rng = np.random.default_rng(2026)
    R = []
    for _ in range(6000):
        _, phi = slow_mode(bonds_from_z(L, Z, rng))
        R.append(r_of_profile(phi, L))
    Y = N ** (1 - 1 / ALPHA) * np.array(R)
    mplus = stats.levy_stable.ppf(0.5, ALPHA, 1.0)
    f = mu / (mu + N ** (1 / ALPHA - 1) * sig * mplus)

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5))
    a = ax[0]
    xs = np.linspace(-1.2, 1.2, 400)
    a.hist(Y, bins=np.linspace(-1.2, 1.2, 61), density=True, color="#eba68e", edgecolor="white", lw=0.3,
           label=r"chain, $L=1024$")
    a.plot(xs, stats.levy_stable.pdf(xs, ALPHA, 0.0, scale=s_a), "k--", lw=1.0, label="stable law")
    a.plot(xs, stats.levy_stable.pdf(xs, ALPHA, 0.0, scale=s_a * f), "k-", lw=1.1, label="with correction")
    a.set_ylim(0, 2.3)
    a.set_xlabel(r"$N^{1-1/\alpha}\,r$"); a.set_ylabel("density")
    a.set_title("(a) residual distribution", loc="left")
    a.legend(frameon=False, loc="upper left", fontsize=6.8)

    # (b), (c) from the Fig. 3 cache
    db = json.load(open("fig_mpemba.json"))
    Lc = np.array([64, 128, 256, 512, 1024, 2048, 4096]); Nc = Lc - 1
    Lf = np.geomspace(64, 4096, 120); Nf = Lf - 1
    a = ax[1]
    for kappa, mk in ((20, "o"), (50, "s")):
        frac = [np.mean(np.array(db[f"cl|a1.25|{L}"]) * kappa < 1) for L in Lc]
        a.semilogx(Lc, frac, mk, color=col, ms=3.6, mfc="white" if kappa == 50 else col, mew=1.0,
                   label=rf"$\kappa={kappa}$")
        a.semilogx(Lf, pwin(Nf, kappa, False), "k--", lw=0.9)
        a.semilogx(Lf, pwin(Nf, kappa, True), "k-", lw=1.0)
    a.set_ylim(0, 0.75)
    a.set_xlabel(r"$L$"); a.set_ylabel("fraction where blind wins")
    a.set_title(r"(b) $P_{\rm win}$", loc="left")
    a.legend(frameon=False, loc="upper left", fontsize=6.8)

    a = ax[2]
    med = [1 / np.median(db[f"cl|a1.25|{L}"]) for L in Lc]
    a.loglog(Lc, med, "o", color=col, ms=3.6)
    a.loglog(Lf, reach(Nf, False), "k--", lw=0.9)
    a.loglog(Lf, reach(Nf, True), "k-", lw=1.0)
    a.set_xlabel(r"$L$"); a.set_ylabel(r"median reach $1/|r|$")
    a.set_title("(c) median reach", loc="left")
    for a in ax[1:]:
        a.set_xticks([64, 256, 1024, 4096]); a.set_xticklabels(["64", "256", "1024", "4096"])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].set_yticks([10, 20]); ax[2].set_yticklabels(["10", "20"])
    ax[2].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    fig.tight_layout(w_pad=0.8)
    fig.savefig("fig_heavy_stable.pdf", bbox_inches="tight")
    fig.savefig("fig_heavy_stable.png", bbox_inches="tight")

    print(f"s_alpha = {s_a:.4f}, mean of N^(1-1/a) r = {Y.mean():+.4f}, median |.| = {np.median(abs(Y)):.4f}")
    for kappa in (20, 50):
        d = [np.mean(np.array(db[f'cl|a1.25|{L}']) * kappa < 1) for L in Lc]
        p = pwin(Nc, kappa, True); p0 = pwin(Nc, kappa, False)
        print(f"kappa={kappa}: data {np.round(d,3)}\n   corrected {np.round(p,3)}  max|diff| L>=256: "
              f"{np.max(np.abs(np.array(d)-p)[2:]):.3f}\n   leading   {np.round(p0,3)}  range of (lead - data): "
              f"{np.min(p0-np.array(d)):.3f} .. {np.max(p0-np.array(d)):.3f}")
    rc = reach(Nc, True)
    print("reach data", np.round(med, 1), " corrected", np.round(rc, 1),
          " rel. diff", np.round(np.array(med) / rc - 1, 3))
