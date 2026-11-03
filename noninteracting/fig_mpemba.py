"""Fig. 3 (fig_mpemba.pdf): Mpemba crossing and reach.
(a), (b) quantum dynamics (gamma = 1, L = 64) of the reference A cos(pi x) and the blind state
kappa A cos(2 pi x); (c), (d) median reach 1/|r| and win fractions over resistor-chain samples
(fig_mpemba.json), with the Gaussian and stable-law predictions."""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply

from density_sector import (single_particle_h, density_generator, gamma_to_Gamma,
                            slow_modes, density_profile, hydrodynamic_reference)
from bond_disorder import draw_bonds, bond_hamiltonian
from blind_scaling import slow_mode, bonds_from_W, bonds_from_z

GAMMA, A, KAPPA = 1.0, 0.2, 2.0
CACHE = "fig_mpemba.json"


def quantum_h(L, kind, rng):
    if kind == "bounded":
        h, _ = single_particle_h(L, J=1.0, W=2.0, rng=rng)
        return h
    return bond_hamiltonian(draw_bonds(L, {"a2.5": 0.2, "a1.25": 0.4}[kind], rng))


def r_of_profile(phi, L):
    xs = (np.arange(L) + 0.5) / L
    return (phi @ np.cos(2 * np.pi * xs)) / (phi @ np.cos(np.pi * xs))


def r_quantum(h, L):
    G = density_generator(h, gamma_to_Gamma(GAMMA))
    w, vr, _ = slow_modes(G, k=1, want_left=False)
    phi = density_profile(vr[:, 0], L, sign_ref=hydrodynamic_reference(L))
    return r_of_profile(phi, L), w[0].real, G


def r_classical(L, kind, rng):
    if kind == "bounded":
        R = bonds_from_W(L, 2.0, rng, gamma=GAMMA)
    else:
        R = bonds_from_z(L, {"a2.5": 0.2, "a1.25": 0.4}[kind], rng)
    _, phi = slow_mode(R)
    return r_of_profile(phi, L)


def stable_pwin(N, kappa, alpha=1.25):
    """Win fraction for heavy-tailed resistances from the symmetric stable law, with the
    finite-size correction (total resistance replaced by its median)."""
    from scipy import integrate, special, stats
    mu = alpha / (alpha - 1)                                  # Pareto, rho_min = 1
    sig = (special.gamma(1 - alpha) * np.cos(np.pi * alpha / 2)) ** (1 / alpha)
    g = lambda x: 4 / 3 * np.sin(np.pi * x) * np.sin(2 * np.pi * x)
    gnorm = integrate.quad(lambda x: abs(g(x)) ** alpha, 0, 1, limit=400)[0] ** (1 / alpha)
    mplus = stats.levy_stable.ppf(0.5, alpha, 1.0)            # median, totally skewed
    z = (mu * np.asarray(N, float) ** (1 - 1 / alpha) + sig * mplus) / (kappa * sig * gnorm)
    return 2 * stats.levy_stable.cdf(z, alpha, 0.0) - 1


def rms_density(V, L):
    dens = np.real(V.reshape(len(V), L, L, order="F")[:, np.arange(L), np.arange(L)])
    return np.sqrt(np.mean((dens - 0.5) ** 2, axis=1))


def trajectory(G, L, shape, amp, T, num=320, slow=None):
    """rms density deviation D(t); with slow=(lam, r1, l1) also D(t) without the slow-mode component."""
    xs = (np.arange(L) + 0.5) / L
    M0 = np.diag(0.5 + amp * shape(xs)).astype(complex).flatten(order="F")
    V = expm_multiply(G, M0, start=0.0, stop=T, num=num, endpoint=True)
    t = np.linspace(0, T, num)
    if slow is None:
        return t, rms_density(V, L)
    lam, r1, l1 = slow
    c1 = np.vdot(l1, M0)
    Vrem = V - c1 * np.exp(lam * t)[:, None] * r1[None, :]
    return t, rms_density(V, L), rms_density(Vrem, L)


def representative(L, kind, seed, ntry=60):
    """A sample whose leftover is close to the median, so the panel is typical."""
    rng = np.random.default_rng(seed)
    hs, rs = [], []
    for _ in range(ntry):
        h = quantum_h(L, kind, rng)
        r, _, _ = r_quantum(h, L)
        hs.append(h); rs.append(abs(r))
    i = int(np.argsort(rs)[len(rs) // 2])   # upper-middle sample (no median tie)
    return hs[i]


def ensemble():
    db = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    kinds = ("bounded", "a2.5", "a1.25")
    for kind in kinds:
        for L in (64, 128, 256, 512, 1024, 2048, 4096):
            key = f"cl|{kind}|{L}"
            if key not in db:
                rng = np.random.default_rng(31 + L)
                db[key] = [abs(r_classical(L, kind, rng)) for _ in range(1500)]
        for L in (32, 48, 64, 96, 128):
            key = f"qm|{kind}|{L}"
            if key not in db:
                rng = np.random.default_rng(71 + L)
                db[key] = [abs(r_quantum(quantum_h(L, kind, rng), L)[0]) for _ in range(150)]
        json.dump(db, open(CACHE, "w"))
    return db


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 6.2, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.linewidth": 0.6, "mathtext.fontset": "cm", "savefig.dpi": 300})
    fig, ax = plt.subplots(2, 2, figsize=(3.4, 3.7))
    L = 64
    c1 = lambda x: np.cos(np.pi * x)
    c2 = lambda x: np.cos(2 * np.pi * x)

    for j, (kind, title, seed) in enumerate((("bounded", r"bounded, $\alpha=\infty$", 5),
                                             ("a1.25", r"hoppings, $\alpha=1.25$", 9))):
        a = ax[0, j]
        h = representative(L, kind, seed)
        r, lam, G = r_quantum(h, L)
        w, vr, vl = slow_modes(G, k=1, want_left=True)
        T = 7.0 / abs(lam)
        t, Dn = trajectory(G, L, c1, A, T)
        _, Db, Db0 = trajectory(G, L, c2, KAPPA * A, T, slow=(w[0], vr[:, 0], vl[:, 0]))
        a.semilogy(t * abs(lam), Dn, color="0.35", lw=1.1, label="reference")
        a.semilogy(t * abs(lam), Db, color="#1D9E75", lw=1.3, label="blind")
        a.semilogy(t * abs(lam), Db0, color="#1D9E75", lw=0.9, ls="--",
                   label="blind, leftover removed")
        k = int(np.argmax(Db < Dn))
        print(f"{kind}: crossing at t|lam1| = {t[k]*abs(lam):.3f}, late-time kappa|r| = {KAPPA*abs(r):.4f}", flush=True)
        a.axvline(t[k] * abs(lam), color="0.6", lw=0.5, ls=":")
        a.text(t[k] * abs(lam) + 0.15, 3e-9, "crossing", fontsize=6, color="0.4", va="bottom")
        a.text(0.97, 0.22, rf"$\kappa|r|={KAPPA*abs(r):.3f}$",
               transform=a.transAxes, ha="right", fontsize=6.5)
        a.set_ylim(max(1e-9, Db0.min()), 0.3)
        a.set_xlabel(r"$t\,|\lambda_1|$")
        if j == 0:
            a.set_ylabel(r"$D(t)$")
            handles, labels = a.get_legend_handles_labels()
        a.set_title(f"({'ab'[j]})", loc="left")

    db = ensemble()
    Lc = np.array([64, 128, 256, 512, 1024, 2048, 4096])
    Lq = np.array([32, 48, 64, 96, 128])
    style = {"bounded": ("#1f5fa8", r"bounded, $\alpha=\infty$", 0.5),
             "a2.5": ("#7a1fa8", r"hoppings, $\alpha=2.5$", 0.5),
             "a1.25": ("#b33a1e", r"hoppings, $\alpha=1.25$", 0.2)}
    a = ax[1, 0]
    for kind, (col, lab, slope) in style.items():
        reach = np.array([1 / np.median(db[f"cl|{kind}|{L}"]) for L in Lc])
        fit = np.polyfit(np.log(Lc), np.log(reach), 1)[0]
        print(f"{kind}: median-reach slope {fit:.3f} (theory {slope})")
        a.loglog(Lc, reach, "o-", color=col, ms=2.8, lw=0.9, label=lab)
    a.set_xlabel(r"$L$")
    a.set_ylabel(r"median reach $1/|r|$")
    a.set_title("(c)", loc="left")
    a.set_ylim(7, 1500)
    a.legend(frameon=False, loc="upper left", handlelength=1.6, labelspacing=0.25,
             borderaxespad=0.2)

    a = ax[1, 1]
    for kappa, ls in ((20, "-"), (50, "--")):
        for kind, (col, lab, _) in style.items():
            frac = [np.mean(np.array(db[f"cl|{kind}|{L}"]) * kappa < 1) for L in Lc]
            a.semilogx(Lc, frac, ls, color=col, lw=0.9, marker="o", ms=2.2)
    a.set_ylim(0, 1.03)
    a.set_xlabel(r"$L$")
    a.set_ylabel("fraction of samples\n(blind wins)")
    # Gaussian prediction, bounded fields (W = 2, gamma = 1): sqrt(N) r ~ N(0, s^2), s = (2/3) sigma_lr / mu
    from scipy.special import erf
    s_pred = (2.0 / 3.0) * np.sqrt(0.2) / (5.0 / 6.0)
    Lf = np.geomspace(Lc[0], Lc[-1], 200)
    for kappa in (20, 50):
        a.semilogx(Lf, erf(np.sqrt(Lf - 1) / (np.sqrt(2) * kappa * s_pred)), ":", color="k", lw=1.0,
                   zorder=5)
    # stable-law prediction, alpha = 1.25:
    # N^(1-1/a) r -> s_a Z_a (symmetric stable), with the leading finite-size correction
    for kappa in (20, 50):
        a.semilogx(Lf, stable_pwin(Lf - 1, kappa), ":", color=style["a1.25"][0], lw=1.1, zorder=5)
        print(f"alpha=1.25 kappa={kappa}: data",
              [round(float(np.mean(np.array(db[f'cl|a1.25|{L}']) * kappa < 1)), 3) for L in Lc],
              " stable law", [round(float(stable_pwin(L - 1, kappa)), 3) for L in Lc])
    a.set_title("(d)", loc="left")

    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=1.1, rect=(0, 0, 1, 0.95))
    fig.savefig("fig_mpemba.pdf", bbox_inches="tight")
    fig.savefig("fig_mpemba.png", bbox_inches="tight")
    for kind in style:
        print(kind, "median reach:", [round(1 / np.median(db[f'cl|{kind}|{L}']), 1) for L in Lc],
              " quantum:", [round(1 / np.median(db[f'qm|{kind}|{L}']), 1) for L in Lq])
