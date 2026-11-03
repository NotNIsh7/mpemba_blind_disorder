"""Fig. 4 (paper/figures/fig_interacting.pdf) from the tables in results/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figures", "fig_interacting.pdf")

INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#d9d8d2"
RAMP = {0.0: "#86b6ef", 1.0: "#3987e5", 2.0: "#1c5cab", 4.0: "#0d366b"} 
MARK = {0.0: "o", 1.0: "s", 2.0: "^", 4.0: "D"}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def read_pilot():
    """{V: [(L, value, err)]} for rms r * sqrt(L) from the exact-diagonalization tables."""
    out = {}
    for name, n in (("interacting_pilot_h4_L8-16.txt", 300), ("interacting_pilot_h4_L18.txt", 300),
                    ("interacting_pilot_h4_L20.txt", 100)):
        for line in open(os.path.join(RES, name)):
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            L, V, val = int(c[0]), float(c[1]), float(c[3])
            out.setdefault(V, []).append((L, val, val / np.sqrt(2 * n)))
    return out


def dmrg(names):
    return np.vstack([np.loadtxt(os.path.join(RES, f)) for f in names])


def boot_median(x, rng, nb=2000):
    m = np.median(x)
    b = np.median(rng.choice(x, (nb, len(x))), axis=1)
    return m, np.std(b)


def style(ax):
    ax.tick_params(colors=INK2, labelsize=7, length=3, width=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def panel_bounded(ax, title="(a) bounded fields"):
    """rms r * sqrt(L) against L, bounded fields, V = 0, 1, 2, 4."""
    pilot = read_pilot()
    for L, files in ((24, ["dmrg_V4_L24_seed1.txt"]), (32, ["dmrg_V4_L32_seed2.txt", "dmrg_V4_L32_seed3.txt"])):
        r = dmrg(files)[:, 3]
        val = np.sqrt(np.mean(r ** 2) * L)
        pilot[4.0].append((L, val, val / np.sqrt(2 * len(r))))
    for V in (0.0, 1.0, 2.0, 4.0):
        L, y, e = map(np.array, zip(*sorted(pilot[V])))
        ax.errorbar(L, y, e, color=RAMP[V], marker=MARK[V], ms=3.5, lw=1.3, capsize=1.5,
                    elinewidth=0.8, mec="white", mew=0.5, label=f"$V={V:g}$")
    ax.axvline(22, color=INK2, lw=0.6, ls=":")
    ax.text(22.6, 0.14, "DMRG", color=INK2, fontsize=6.5, va="top")
    ax.set_xlabel("$L$")
    ax.set_ylabel(r"rms $r\,\sqrt{L}$")
    ax.set_ylim(0, 0.15)
    ax.set_xticks([8, 16, 24, 32])
    ax.legend(ncol=2, fontsize=6.3, frameon=False, loc="lower left", handlelength=1.3,
              columnspacing=0.8, borderaxespad=0.2)
    ax.set_title(title, loc="left", fontsize=7.5, color=INK)
    style(ax)


def panel_bounded_scatter(ax, title="(b) $L=24,32$"):
    """Bounded fields: r against the averaged-rate resistor chain, DMRG at V = 4."""
    bd = dmrg(["dmrg_V4_L24_seed1.txt", "dmrg_V4_L32_seed2.txt", "dmrg_V4_L32_seed3.txt"])
    lim = 0.065
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls="--", zorder=1)
    sl = bd[:, 3] @ bd[:, 4] / (bd[:, 4] @ bd[:, 4])
    ax.plot([-lim / sl, lim / sl], [-lim, lim], color=RAMP[1.0], lw=1.0, zorder=2)
    ax.scatter(bd[:, 4], bd[:, 3], s=11, color=RAMP[4.0], marker="D", edgecolors="white", lw=0.4, zorder=3)
    ax.text(0.005, -0.055, f"slope {sl:.2f}", color=INK, fontsize=6.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$r$, resistor chain")
    ax.set_ylabel(r"$r$ at $V=4$")
    ax.set_title(title, loc="left", fontsize=7.5, color=INK)
    style(ax)


def panel_heavy_ratio(ax, rng, title=r"(c) heavy tails, $\alpha=1.25$"):
    """Heavy-tailed hoppings: median |r(V=4) / r(V=0)| on the same samples."""
    for h0, color, mk, label in ((0.0, ORANGE, "o", "hoppings only"), (4.0, AQUA, "s", "with fields")):
        pts = []
        for L in (8, 10, 12, 14, 16, 18):
            d = np.load(os.path.join(RES, f"heavy_tail_hopping_z0.4_h{h0}_L{L}.npz"))
            pts.append((L, *boot_median(np.abs(d["r_V4.0"] / d["r_V0.0"]), rng)))
        d = np.loadtxt(os.path.join(RES, f"heavy_tail_dmrg_h{int(h0)}_V4_L24.txt"))
        pts.append((24, *boot_median(np.abs(d[:, 3] / d[:, 5]), rng)))
        L, y, e = map(np.array, zip(*pts))
        ax.errorbar(L, y, e, color=color, marker=mk, ms=3.5, lw=1.3, capsize=1.5, elinewidth=0.8,
                    mec="white", mew=0.5, label=label)
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.axvline(21, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel("$L$")
    ax.set_ylabel(r"median $|r_{V=4}/r_{V=0}|$")
    ax.set_ylim(0.94, 1.01)
    ax.set_xticks([8, 12, 16, 20, 24])
    ax.legend(fontsize=6.3, frameon=False, loc="lower right", borderaxespad=0.2)
    ax.set_title(title, loc="left", fontsize=7.5, color=INK)
    style(ax)


def panel_heavy_scatter(ax, title="(d) $L=24$"):
    """Heavy-tailed hoppings: r against the averaged-rate resistor chain, DMRG."""
    ht = dmrg(["heavy_tail_dmrg_h0_V4_L24.txt"])
    lim = 0.6
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls="--", zorder=1)
    ax.scatter(ht[:, 4], ht[:, 3], s=11, color=ORANGE, marker="o", edgecolors="white", lw=0.4, zorder=3)
    sl = ht[:, 3] @ ht[:, 4] / (ht[:, 4] @ ht[:, 4])
    ax.text(0.05, -0.5, f"slope {sl:.2f}", color=INK, fontsize=6.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$r$, resistor chain")
    ax.set_ylabel(r"$r$ at $V=4$")
    ax.set_title(title, loc="left", fontsize=7.5, color=INK)
    style(ax)


def setup():
    plt.rcParams.update({"font.size": 7.5, "axes.labelcolor": INK, "text.color": INK,
                         "font.family": "serif", "mathtext.fontset": "cm"})


def main():
    setup()
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.3), constrained_layout=True)
    panel_bounded(axes[0, 0], title="(a)")
    panel_bounded_scatter(axes[0, 1], title="(b)")
    panel_heavy_ratio(axes[1, 0], np.random.default_rng(0), title="(c)")
    panel_heavy_scatter(axes[1, 1], title="(d)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
