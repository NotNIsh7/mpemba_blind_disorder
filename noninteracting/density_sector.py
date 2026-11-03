"""Exact density-sector dynamics of the dephased disordered XX chain.

After Jordan-Wigner, M[i,j] = <c^dag_j c_i> obeys dM/dt = -i[h, M] - Gamma (M - diag M),
with Gamma = 4 gamma, h[i,i+1] = 2J and h[i,i] = 2 h_i. This is an L^2-dimensional linear
equation, checked against the full Liouvillian in test_density_sector.py."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags, eye as sp_eye, kron as sp_kron
from scipy.sparse.linalg import eigs


def draw_fields(L, W=1.0, seed=None, dist="uniform", rng=None):
    """On-site fields: 'uniform' in [-W, W], 'gaussian', or 'cauchy'."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    if dist == "uniform":
        return rng.uniform(-W, W, size=L)
    if dist == "cauchy":
        return W * rng.standard_cauchy(L)
    if dist == "gaussian":
        return rng.normal(0.0, W, size=L)
    raise ValueError(dist)


def single_particle_h(L, J=1.0, W=1.0, seed=None, fields=None, pbc=False,
                      dist="uniform", rng=None):
    """Single-particle Hamiltonian after Jordan-Wigner. Returns (h, fields)."""
    if fields is None:
        fields = draw_fields(L, W=W, seed=seed, dist=dist, rng=rng)
    fields = np.asarray(fields, dtype=float)
    h = np.diag(2.0 * fields).astype(complex)
    for i in range(L - 1):
        h[i, i + 1] = h[i + 1, i] = 2.0 * J
    if pbc and L > 2:
        h[0, -1] = h[-1, 0] = 2.0 * J
    return h, fields


def density_generator(h, Gamma, sparse=True):
    """Generator of dM/dt with column stacking, vec(M)[i + L*j] = M[i, j]."""
    L = h.shape[0]
    I = sp_eye(L, format="csr", dtype=complex)
    G = -1j * (sp_kron(I, csr_matrix(h)) - sp_kron(csr_matrix(h.T), I))
    offdiag = (1.0 - np.eye(L)).flatten(order="F")
    G = G - Gamma * diags(offdiag, format="csr", dtype=complex)
    G = csr_matrix(G)
    return G if sparse else G.toarray()


def gamma_to_Gamma(gamma):
    """Spin dephasing rate -> fermionic number-dephasing rate."""
    return 4.0 * gamma



def slow_modes(G, k=1, n_zero=1, sigma=-1e-8, tol=1e-9, want_left=True):
    """The k slowest decaying modes: (evals, right, left), left biorthonormal to right.
    A small negative shift avoids the exact zero mode in shift-invert."""
    G = csr_matrix(G)
    want = min(k + n_zero + 2, G.shape[0] - 2)
    w, vr = eigs(G, k=want, sigma=sigma, which="LM")
    keep = np.abs(w) >= tol
    w, vr = w[keep], vr[:, keep]
    order = np.argsort(-w.real)[:k]
    w, vr = w[order], vr[:, order]

    vl = None
    if want_left:
        wl, vlall = eigs(G.conj().T.tocsr(), k=want, sigma=np.conj(sigma), which="LM")
        vl = np.zeros_like(vr)
        for i, lam in enumerate(w):
            j = np.argmin(np.abs(wl - np.conj(lam)))
            u = vlall[:, j]
            vl[:, i] = u / np.conj(u.conj() @ vr[:, i])
    return w, vr, vl


def density_profile(vec_mode, L, normalize=True, sign_ref=None):
    """
    Diagonal (density) part of a vectorized mode, sign-fixed against a
    reference so profiles from different realizations can be compared.
    """
    M = vec_mode.reshape((L, L), order="F")
    phi = np.real(np.diag(M))
    if normalize:
        n = np.linalg.norm(phi)
        if n > 0:
            phi = phi / n
    if sign_ref is not None and np.dot(phi, sign_ref) < 0:
        phi = -phi
    return phi


def hydrodynamic_reference(L):
    """First no-flux diffusive eigenfunction on [0, 1], discretized."""
    x = (np.arange(L) + 0.5) / L
    phi = np.cos(np.pi * x)
    return phi / np.linalg.norm(phi)



def bond_rates(fields, J=1.0, gamma=1.0):
    """R_i = 8 J^2 gamma / (4 gamma^2 + (h_i - h_{i+1})^2), one per bond."""
    d = np.diff(np.asarray(fields, dtype=float))
    return 8.0 * J ** 2 * gamma / (4.0 * gamma ** 2 + d ** 2)


def D_eff_harmonic(rates):
    """Series-resistor (harmonic) mean. Diverges if rates have weight near 0."""
    return 1.0 / np.mean(1.0 / np.asarray(rates, dtype=float))


def predicted_gap(D_eff, L):
    """Slowest mode of a uniform hopping chain of L sites with no-flux ends."""
    return -4.0 * D_eff * np.sin(np.pi / (2 * L)) ** 2



def neel_M(L, start_up=True):
    """Neel product state as a correlation matrix (diagonal, alternating)."""
    occ = np.array([1.0 if ((i % 2 == 0) == start_up) else 0.0 for i in range(L)])
    return np.diag(occ).astype(complex)


def imbalance_from_M(M):
    """I = (1/L) sum_i (-1)^i <sz_i>, with <sz_i> = 2 M_ii - 1."""
    L = M.shape[0]
    signs = (-1.0) ** np.arange(L)
    return float(np.real(signs @ (2 * np.diag(M) - 1)) / L)


def evolve(G, M0, times):
    """Imbalance trajectory by sparse exponential action."""
    from scipy.sparse.linalg import expm_multiply
    L = M0.shape[0]
    v0 = M0.flatten(order="F")
    out = []
    for t in times:
        v = expm_multiply(csr_matrix(G) * t, v0)
        out.append(imbalance_from_M(v.reshape((L, L), order="F")))
    return np.array(out)



def slow_profile_sample(L, J=1.0, W=1.0, gamma=1.0, seed=None,
                        dist="uniform", rng=None):
    """Slowest density mode of one disorder realization. Returns (lam, phi, fields)."""
    h, fields = single_particle_h(L, J=J, W=W, seed=seed, dist=dist, rng=rng)
    G = density_generator(h, gamma_to_Gamma(gamma))
    w, vr, _ = slow_modes(G, k=1, want_left=False)
    phi = density_profile(vr[:, 0], L, sign_ref=hydrodynamic_reference(L))
    return w[0], phi, fields


def overlap_deficit(L, n_samples=20, J=1.0, W=1.0, gamma=1.0, seed0=0,
                    against="hydro", dist="uniform"):
    """
    Mean overlap deficit 1 - |<phi_1, ref>|^2 of the slow density profile.

    against: 'hydro' (the homogenized cos profile), 'pairwise' (over sample
    pairs), or 'mean' (leave-one-out disorder average).
    """
    lams, profs, Ds = [], [], []
    rng = np.random.default_rng(seed0)
    for _ in range(n_samples):
        lam, phi, f = slow_profile_sample(L, J=J, W=W, gamma=gamma,
                                          dist=dist, rng=rng)
        lams.append(lam.real)
        profs.append(phi)
        Ds.append(D_eff_harmonic(bond_rates(f, J, gamma)))
    profs, lams = np.array(profs), np.array(lams)

    if against == "hydro":
        ref = hydrodynamic_reference(L)
        d = 1.0 - (profs @ ref) ** 2
    elif against == "pairwise":
        d = []
        for a in range(n_samples):
            for b in range(a + 1, n_samples):
                d.append(1.0 - (profs[a] @ profs[b]) ** 2)
        d = np.array(d)
    elif against == "mean":
        d = []
        for a in range(n_samples):
            others = np.delete(profs, a, axis=0).mean(axis=0)
            others /= np.linalg.norm(others)
            d.append(1.0 - (profs[a] @ others) ** 2)
        d = np.array(d)
    else:
        raise ValueError(against)

    return {"L": L, "deficit_mean": d.mean(), "deficit_std": d.std(),
            "deficit": d, "lam_mean": lams.mean(), "lam_std": lams.std(),
            "profiles": profs, "D_eff_mean": float(np.mean(Ds))}


def deficit_scan(Ls, n_samples=20, against="hydro", **kw):
    """kw is forwarded to overlap_deficit (W, gamma, dist, seed0, J)."""
    """Overlap deficit versus L, plus the L^2 * lambda_1 collapse."""
    rows = []
    for L in Ls:
        r = overlap_deficit(L, n_samples=n_samples, against=against, **kw)
        rows.append({"L": L, "deficit": r["deficit_mean"], "spread": r["deficit_std"],
                     "lam": r["lam_mean"], "lam_std": r["lam_std"],
                     "L2lam": L ** 2 * r["lam_mean"],
                     "rel_lam_spread": r["lam_std"] / abs(r["lam_mean"]),
                     "D_eff": r["D_eff_mean"]})
    return rows



if __name__ == "__main__":
    import time

    J, gamma, W = 1.0, 1.0, 2.0

    print("1. strong-dephasing homogenization: exact gap vs harmonic-mean prediction")
    print(f"{'L':>4} {'gamma':>7} {'exact lam1':>13} {'harmonic pred':>15} {'ratio':>8}")
    for g in (1.0, 4.0, 16.0):
        L = 48
        lam, phi, fields = slow_profile_sample(L, J=J, W=W, gamma=g, seed=1)
        pred = predicted_gap(D_eff_harmonic(bond_rates(fields, J, g)), L)
        print(f"{L:>4} {g:>7.1f} {lam.real:>13.6f} {pred:>15.6f} {lam.real/pred:>8.3f}")
    print("   ratio -> 1 as dephasing grows: the classical hopping reduction is")
    print("   the right description, and D_eff is the harmonic mean of bond rates.")

    print("\n2. does the slow density mode self-average? (32 samples per point)")
    for dist in ("uniform", "cauchy"):
        print(f"\n   {dist} fields")
        print(f"   {'L':>5} {'deficit':>11} {'spread':>11} {'L^2 lam1':>11} "
              f"{'lam relstd':>11} {'D_eff':>9} {'s':>6}")
        for L in (16, 32, 64, 128):
            t = time.time()
            r = deficit_scan([L], n_samples=32, W=W, gamma=gamma,
                             dist=dist, seed0=7)[0]
            print(f"   {L:>5} {r['deficit']:>11.3e} {r['spread']:>11.3e} "
                  f"{r['L2lam']:>11.4f} {r['rel_lam_spread']:>11.3e} "
                  f"{r['D_eff']:>9.4f} {time.time()-t:>6.1f}")
    print("\n   Bounded disorder: deficit falls as a power of L, lambda spread")
    print("   shrinks near the CLT rate, D_eff is L-independent. Self-averaging.")
    print("   Cauchy disorder: deficit saturates, spread stays O(1), and D_eff")
    print("   drifts toward zero with L, which is <1/R> diverging. Not self-averaging.")

    print("\n3. cost of one slow-mode solve")
    for L in (64, 128, 200):
        t = time.time()
        slow_profile_sample(L, W=W, gamma=gamma, seed=0)
        print(f"   L = {L:>4}  generator {L**2} x {L**2}"
              f"  (full Liouvillian would be {4.0**L:.0e})  {time.time()-t:.2f}s")
