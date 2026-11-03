"""Fig. 2 (fig_scaling1d.pdf). (a) Exponent of the slow-mode distortion d against 1/alpha;
(b) random hoppings, L = 64-4096: exponents of the relaxation rate and of d."""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# bond disorder, J_i = u^z, L = 64..4096
Z = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
A_BOND = 1.0 / (2 * Z)
D_BOND = np.array([-0.493, -0.475, -0.443, -0.394, -0.322, -0.244, -0.175, -0.122])
LAM_BOND = np.array([-2.000, -2.000, -2.002, -2.004, -2.012, -2.026, -2.054, -2.087])

# bounded field disorder: 1/R bounded, so alpha = infinity
W_FIELD = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
D_FIELD = np.array([-0.495, -0.497, -0.498, -0.497, -0.496])
LAM_FIELD = np.array([-2.000, -2.001, -2.002, -2.003, -2.004])

# Pareto resistances, alpha chosen directly, L = 64..4096
A_PAR = np.array([4.0, 3.0, 2.5, 2.0, 1.8, 1.667, 1.5, 1.429, 1.3, 1.25, 1.111])
D_PAR = np.array([-0.479, -0.464, -0.443, -0.394, -0.358, -0.325,
                  -0.273, -0.247, -0.198, -0.179, -0.123])

# Pareto, asymptotic window L >= 4096 (from corrections.json)
LS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


def asymptotic_fits(path="corrections.json", Lmin=4096):
    db = json.load(open(path))
    alphas = sorted({float(k.split("|")[0]) for k in db}, reverse=True)
    out = []
    for a in alphas:
        L = np.array([l for l in LS if l >= Lmin and f"{a:g}|{l}" in db], float)
        if len(L) < 3:
            continue
        d = np.array([db[f"{a:g}|{int(l)}"]["d"] for l in L])
        e = np.array([db[f"{a:g}|{int(l)}"]["e"] for l in L])
        w = 1.0 / (e / d) ** 2
        A = np.vstack([np.log(L), np.ones_like(L)]).T
        cov = np.linalg.inv(A.T @ np.diag(w) @ A)
        p = cov @ A.T @ np.diag(w) @ np.log(d)
        out.append((a, p[0], np.sqrt(cov[0, 0])))
    return np.array(out)


def theory(inv_a):
    """max(-1/2, 1/alpha - 1) as a function of 1/alpha."""
    return np.maximum(-0.5, inv_a - 1.0)


def main():
    asy = asymptotic_fits()
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 6.6,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.8,
        "lines.linewidth": 1.3, "figure.dpi": 200, "savefig.dpi": 200,
        "mathtext.fontset": "dejavusans",
    })
    fig, ax = plt.subplots(2, 1, figsize=(3.36, 5.37),
                           gridspec_kw={"height_ratios": [1, 1], "hspace": 0.32})
    x = np.linspace(0, 1.0, 400)
    BLUE, GREEN = "#1f5fa8", "#1f7a4a"

    # (a)
    a = ax[0]
    a.axvspan(0.5, 1.0, color="0.93", zorder=0)
    a.plot(x, theory(x), "k-", lw=1.6, zorder=3, label=r"$\max(-1/2,\ 1/\alpha-1)$")
    a.axvline(0.5, color="0.45", ls="--", lw=1.0, zorder=1)
    a.plot(0.0, D_FIELD.mean(), marker="*", ms=11, color=BLUE, ls="none", zorder=6,
           label=r"bounded fields ($\alpha=\infty$)")
    a.plot(1.0 / A_BOND, D_BOND, "o", ms=5.4, color=BLUE, ls="none", zorder=5,
           label=r"random hoppings $J_i=v^z$")
    a.plot(1.0 / A_PAR, D_PAR, "s", ms=4.6, mfc="none", color="#c2492e", ls="none",
           zorder=5, label="Pareto resistances")
    a.errorbar(1.0 / asy[:, 0], asy[:, 1], yerr=asy[:, 2], fmt="D", ms=4.6, color="#7a1fa8",
               capsize=2, ls="none", zorder=7, label=r"Pareto, asymptotic window $L\geq4096$")
    a.text(0.53, -0.585, r"$\alpha<2$", color="0.35", fontsize=7.5)
    a.text(0.47, -0.585, r"$\alpha>2$", color="0.35", ha="right", fontsize=7.5)
    a.set_ylabel(r"exponent of $d=\Vert\phi_1-\cos\pi x\Vert$")
    a.set_xlabel(r"$1/\alpha$  (tail index of the resistance $\rho$)")
    a.set_ylim(-0.62, 0.10)
    a.legend(frameon=False, loc="upper left", handletextpad=0.4, borderaxespad=0.3,
             labelspacing=0.25, fontsize=6.2)

    # (b)
    b = ax[1]
    b.axvspan(0.5, 1.0, color="0.93", zorder=0)
    b.axvline(0.5, color="0.45", ls="--", lw=1.0, zorder=1)
    b.plot(x, theory(x), "k-", lw=1.6, zorder=3)
    b.axhline(-2.0, color="0.3", ls=":", lw=1.0)
    b.plot(1.0 / A_BOND, LAM_BOND, "o-", ms=4.6, color=GREEN, zorder=4,
           label=r"relaxation rate $\lambda_1$")
    b.plot(1.0 / A_BOND, D_BOND, "o-", ms=4.6, color=BLUE, zorder=4,
           label=r"slow-mode distortion $d$")
    b.plot(0.0, LAM_FIELD.mean(), marker="*", ms=11, color=GREEN, ls="none", zorder=5)
    b.plot(0.0, D_FIELD.mean(), marker="*", ms=11, color=BLUE, ls="none", zorder=5)
    b.text(0.02, -0.72, "degrades continuously", color=BLUE, fontsize=7.5)
    b.text(0.02, -1.92, "close to diffusive", color=GREEN, fontsize=7.5)
    b.set_xlabel(r"$1/\alpha$")
    b.set_ylabel("fitted exponent")
    b.set_ylim(-2.15, 0.05)
    b.legend(frameon=False, loc="center right", bbox_to_anchor=(0.98, 0.42),
             handletextpad=0.4, fontsize=6.6)

    for a, lab in zip(ax, "ab"):
        a.set_xlim(-0.05, 1.0)
        a.set_title(f"({lab})", loc="left", fontsize=8.5, pad=3)
    fig.align_ylabels(ax)
    fig.savefig("tail_index_law.pdf", bbox_inches="tight")
    fig.savefig("tail_index_law.png", bbox_inches="tight")
    print("wrote tail_index_law.pdf / .png")
    print("\nasymptotic-window fits used in panel (a):")
    for r in asy:
        print(f"   alpha={r[0]:>6.3f}   {r[1]:+.3f} +/- {r[2]:.3f}   "
              f"theory {max(-0.5, 1/r[0]-1):+.3f}")


if __name__ == "__main__":
    main()
