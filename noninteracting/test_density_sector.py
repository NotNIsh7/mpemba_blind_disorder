"""Checks of density_sector.py, including agreement with the full Liouvillian."""
import numpy as np
from dephased_xx import (build_hamiltonian, build_liouvillian, dephasing_ops,
                         dm_labels, solve_modes, neel_state, imbalance_op)
from density_sector import *


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


def main():
    L, J, W, gamma, seed = 4, 1.0, 4.0, 1.0, 3
    ok = []

    H, fields = build_hamiltonian(L, J=J, W=W, seed=seed, basis="pauli")
    Lv = build_liouvillian(H, dephasing_ops(L, gamma, "pauli"))
    ev_full = np.linalg.eigvals(Lv)

    h, f2 = single_particle_h(L, J=J, W=W, seed=seed)
    ok.append(check("same disorder realization as the spin code",
                    np.allclose(fields, f2), ""))

    G = density_generator(h, gamma_to_Gamma(gamma), sparse=False)
    ev_c = np.linalg.eigvals(G)

    worst = max(np.abs(ev_full - z).min() for z in ev_c)
    ok.append(check("two-point spectrum is a subset of the full Liouvillian",
                    worst < 1e-9, f"max distance {worst:.2e}"))

    slow_c = np.sort(ev_c[np.abs(ev_c) > 1e-9].real)[::-1][0]
    slow_f = np.sort(ev_full[np.abs(ev_full) > 1e-9].real)[::-1][0]
    ok.append(check("slowest decaying mode agrees with full ED",
                    abs(slow_c - slow_f) < 1e-9, f"{slow_c:.9f} vs {slow_f:.9f}"))

    nz = int((np.abs(ev_c) < 1e-9).sum())
    ok.append(check("exactly one zero mode (particle number)", nz == 1, f"got {nz}"))

    # imbalance trajectory against the full Liouvillian
    from dephased_xx import vec, unvec
    from scipy.linalg import expm
    times = np.linspace(0, 4, 5)
    I_c = evolve(density_generator(h, gamma_to_Gamma(gamma)), neel_M(L), times)
    rho0, O = neel_state(L), imbalance_op(L, basis="pauli")
    I_f = np.array([np.trace(O @ unvec(expm(Lv * t) @ vec(rho0), 2 ** L)).real
                    for t in times])
    e = np.abs(I_c - I_f).max()
    ok.append(check("imbalance trajectory matches full Liouvillian", e < 1e-9,
                    f"max dev {e:.2e}"))

    # sparse slow mode == dense slow mode
    w, vr, vl = slow_modes(density_generator(h, gamma_to_Gamma(gamma)), k=1)
    ok.append(check("sparse slow mode == dense", abs(w[0].real - slow_c) < 1e-9,
                    f"{w[0].real:.9f}"))
    ok.append(check("left mode biorthonormal", abs(vl[:, 0].conj() @ vr[:, 0] - 1) < 1e-8))

    # strong dephasing limit
    Lb = 64
    lam, _, fld = slow_profile_sample(Lb, W=2.0, gamma=32.0, seed=1)
    pred = predicted_gap(D_eff_harmonic(bond_rates(fld, 1.0, 32.0)), Lb)
    ratio = lam.real / pred
    ok.append(check("harmonic-mean gap prediction at strong dephasing",
                    abs(ratio - 1) < 0.02, f"ratio {ratio:.4f}"))

    # trace conservation under the generator
    Gs = density_generator(h, gamma_to_Gamma(gamma), sparse=False)
    tr = np.eye(L).flatten(order="F")
    ok.append(check("Tr M conserved (tr^dag G = 0)",
                    np.linalg.norm(tr @ Gs) < 1e-12, f"{np.linalg.norm(tr @ Gs):.2e}"))

    print(f"\n{sum(ok)}/{len(ok)} passed")


if __name__ == "__main__":
    main()
