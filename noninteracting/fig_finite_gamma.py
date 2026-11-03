"""SM figure fig_quantum_hopping.pdf: exponents at finite dephasing for random hoppings
(quantum density sector against the classical chain)."""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.linewidth": 0.7, "mathtext.fontset": "cm", "savefig.dpi": 200})

d = json.load(open("finite_gamma_heavy.json"))
LS = [48, 64, 96, 128, 192, 256]

# alpha = 1.25, L >= 96: use the higher-statistics rerun
try:
    more = json.load(open("finite_gamma_heavy_more.json"))
    rb0 = np.random.default_rng(0)
    for key, vals in more.items():
        v = np.asarray(vals)
        boots = [np.median(v[rb0.integers(0, len(v), len(v))]) for _ in range(800)]
        d[key] = {"median": float(np.median(v)), "err": float(np.std(boots)), "n": len(v)}
except FileNotFoundError:
    pass


def series(prefix):
    LL = [L for L in LS if f"{prefix}|{L}" in d]
    m = np.array([d[f"{prefix}|{L}"]["median"] for L in LL])
    e = np.array([d[f"{prefix}|{L}"]["err"] for L in LL])
    x = np.log(LL)
    a = np.polyfit(x, np.log(m), 1)[0]
    rb = np.random.default_rng(1)
    dr = [np.polyfit(x, np.log(np.maximum(m + rb.normal(0, e), 1e-9)), 1)[0] for _ in range(3000)]
    return np.array(LL), m, e, a, np.std(dr)


fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.6))
cols = {0.25: "#b33a1e", 1.0: "#7a1fa8", 4.0: "#1f5fa8"}
rows = []
for j, (z, title) in enumerate(((0.2, r"$\alpha=2.5$ (finite variance)"),
                                (0.4, r"$\alpha=1.25$ (infinite variance)"))):
    a = ax[j]
    L, m, e, s, se = series(f"classical|{z}")
    a.loglog(L, m, "k-", lw=1.4, label=rf"classical chain: ${s:+.2f}\pm{se:.2f}$")
    rows.append((z, "classical", s, se))
    for g, c in cols.items():
        pre = f"quantum|{z}|{g}"
        if not any(f"{pre}|{Lx}" in d for Lx in LS):
            continue
        L, m, e, s, se = series(pre)
        a.errorbar(L, m, yerr=e, fmt="o", ms=4, color=c, capsize=1.5, lw=0.8,
                   label=rf"quantum, $\gamma={g:g}$: ${s:+.2f}\pm{se:.2f}$")
        rows.append((z, f"gamma={g}", s, se))
    a.set_xlabel(r"$L$")
    a.set_ylabel(r"median $d=\|\phi_1-\cos\pi x\|$")
    a.set_title(f"({'ab'[j]}) random hoppings, {title}", loc="left")
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2),
             ncol=2, fontsize=6.9, columnspacing=1.0, handletextpad=0.4)
    a.set_xlim(42, 300)
    a.set_xticks([48, 64, 96, 128, 192, 256])
    a.set_xticklabels(["48", "64", "96", "128", "192", "256"])
    a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    a.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
fig.tight_layout()
fig.savefig("fig_finite_gamma.pdf", bbox_inches="tight")
fig.savefig("fig_finite_gamma.png", bbox_inches="tight")
for r in rows:
    print(f"alpha={1/(2*r[0]):.2f}  {r[1]:>11}  {r[2]:+.3f} +/- {r[3]:.3f}")
