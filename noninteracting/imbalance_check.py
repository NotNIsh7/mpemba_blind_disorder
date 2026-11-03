"""Weight of the slow density mode in the Neel state and in the imbalance, against a smooth
initial profile and observable."""

from __future__ import annotations

import numpy as np

from density_sector import (single_particle_h, density_generator, gamma_to_Gamma,
                            draw_fields, hydrodynamic_reference, bond_rates,
                            D_eff_harmonic)


def modes_dense(L, W, gamma, rng, dist="uniform"):
    """All density modes, biorthonormalized, sorted slowest-decaying first."""
    fields = draw_fields(L, W=W, dist=dist, rng=rng)
    h, _ = single_particle_h(L, fields=fields)
    G = density_generator(h, gamma_to_Gamma(gamma), sparse=False)
    w, vr = np.linalg.eig(G)
    vl = np.linalg.inv(vr).conj().T          # exact biorthonormal partner
    order = np.argsort(-w.real)
    return w[order], vr[:, order], vl[:, order], fields


def diag_of(vec, L):
    return np.diag(vec.reshape((L, L), order="F"))


def analyse(L, W=2.0, gamma=1.0, seed=0, dist="uniform"):
    rng = np.random.default_rng(seed)
    w, vr, vl, fields = modes_dense(L, W, gamma, rng, dist)

    # Neel deviation from the uniform steady state, as a density matrix
    dn = np.zeros(L)
    dn[0::2], dn[1::2] = 0.5, -0.5
    dM = np.diag(dn).astype(complex).flatten(order="F")

    # imbalance observable: I = (1/L) sum (-1)^i (2 M_ii - 1)
    Ovec = np.zeros((L, L), dtype=complex)
    for i in range(L):
        Ovec[i, i] = 2.0 * ((-1) ** i) / L
    Ovec = Ovec.flatten(order="F")

    c = vl.conj().T @ dM                      # <l_k, dn>
    o = (vr.conj().T @ Ovec).conj()           # Tr[O r_k]
    wt = c * o

    decay = np.abs(w) > 1e-9
    idx = np.where(decay)[0]                
    slow = idx[0]

    # smooth reference for comparison
    ref = hydrodynamic_reference(L)
    phi1 = np.real(diag_of(vr[:, slow], L))
    phi1 = phi1 / np.linalg.norm(phi1)
    if phi1 @ ref < 0:
        phi1 = -phi1

    # smooth initial deviation
    dn_smooth = ref / np.linalg.norm(ref) * 0.5
    dM_s = np.diag(dn_smooth).astype(complex).flatten(order="F")
    c_s = vl.conj().T @ dM_s

    # smooth observable: total staggering replaced by first cosine moment
    Osm = np.diag(2.0 * ref / L).astype(complex).flatten(order="F")
    o_s = (vr.conj().T @ Osm).conj()

    tot = np.abs(wt[idx]).sum()
    return {
        "L": L,
        "lam1": w[slow].real,
        "c1_neel": abs(c[slow]),
        "o1_imb": abs(o[slow]),
        "w1_neel_imb": abs(wt[slow]),
        "w1_frac": abs(wt[slow]) / tot if tot > 0 else np.nan,
        "argmax_weight_rank": int(np.argmax(np.abs(wt[idx]))),
        "lam_dominant": w[idx[np.argmax(np.abs(wt[idx]))]].real,
        "c1_smooth": abs(c_s[slow]),
        "o1_smooth": abs(o_s[slow]),
        "overlap_phi1_stag": abs(phi1 @ (np.array([(-1.0) ** i for i in range(L)])
                                         / np.sqrt(L))),
        "D_eff": D_eff_harmonic(bond_rates(fields, 1.0, gamma)),
    }


if __name__ == "__main__":
    print("Neel + imbalance vs the slowest hydrodynamic density mode")
    print("(W=2, gamma=1, uniform disorder; one sample per L)\n")
    hdr = (f"{'L':>4} {'lam_1':>11} {'|c1| Neel':>11} {'|o1| imb':>11} "
           f"{'w1 frac':>10} {'dom rank':>9} {'lam_dom':>11}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for L in (12, 16, 20, 24, 28, 32, 40):
        r = analyse(L, seed=5)
        rows.append(r)
        print(f"{L:>4} {r['lam1']:>11.3e} {r['c1_neel']:>11.3e} "
              f"{r['o1_imb']:>11.3e} {r['w1_frac']:>10.3e} "
              f"{r['argmax_weight_rank']:>9} {r['lam_dominant']:>11.3e}")

    print("\nSame slow mode, but with a SMOOTH preparation and observable:")
    hdr2 = f"{'L':>4} {'|c1| smooth':>13} {'|o1| smooth':>13} {'ratio to Neel/imb':>19}"
    print(hdr2); print("-" * len(hdr2))
    for r in rows:
        print(f"{r['L']:>4} {r['c1_smooth']:>13.3e} {r['o1_smooth']:>13.3e} "
              f"{r['c1_smooth']*r['o1_smooth']/max(r['w1_neel_imb'],1e-300):>19.3e}")

    Ls = np.array([r["L"] for r in rows], float)
    for k in ("c1_neel", "o1_imb", "w1_frac"):
        y = np.array([r[k] for r in rows])
        if (y > 0).all():
            print(f"\n  {k:>10} ~ L^({np.polyfit(np.log(Ls), np.log(y), 1)[0]:+.2f})")
