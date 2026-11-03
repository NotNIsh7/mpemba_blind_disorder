"""Random hoppings J_i = u^z at finite dephasing (quantum density sector) against the
classical chain over the same L: exponent of d = ||phi_1 - cos(pi x)||, alpha = 2.5 and 1.25.
Writes finite_gamma_heavy.json."""

import json
import os
import time

import numpy as np

from density_sector import (density_generator, gamma_to_Gamma, slow_modes,
                            density_profile, hydrodynamic_reference)
from bond_disorder import draw_bonds, bond_hamiltonian
from blind_scaling import slow_mode

OUT = "finite_gamma_heavy.json"
ZS = (0.2, 0.4)                       # alpha = 2.5 and 1.25
GAMMAS = (0.25, 1.0, 4.0)
LS = [(48, 300), (64, 300), (96, 250), (128, 200), (192, 120), (256, 80)]


def d_quantum(L, z, gamma, rng):
    J = draw_bonds(L, z, rng)
    G = density_generator(bond_hamiltonian(J), gamma_to_Gamma(gamma))
    _, vr, _ = slow_modes(G, k=1, want_left=False)
    ref = hydrodynamic_reference(L)
    phi = density_profile(vr[:, 0], L, sign_ref=ref)
    return float(np.linalg.norm(phi - ref))


def d_classical(L, z, rng):
    J = draw_bonds(L, z, rng)
    _, phi = slow_mode(2.0 * J ** 2)       # gamma only rescales R here
    ref = hydrodynamic_reference(L)
    phi = phi / np.linalg.norm(phi)
    if phi @ ref < 0:
        phi = -phi
    return float(np.linalg.norm(phi - ref))


def summarise(ds):
    ds = np.asarray(ds)
    rb = np.random.default_rng(0)
    boots = [np.median(ds[rb.integers(0, len(ds), len(ds))]) for _ in range(500)]
    return {"median": float(np.median(ds)), "err": float(np.std(boots)), "n": len(ds)}


if __name__ == "__main__":
    db = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for z in ZS:
        for L, n in LS:
            key = f"classical|{z}|{L}"
            if key not in db:
                rng = np.random.default_rng(700 + L + int(100 * z))
                db[key] = summarise([d_classical(L, z, rng) for _ in range(3 * n)])
                json.dump(db, open(OUT, "w"), indent=1)
    for z in ZS:
        for g in GAMMAS:
            for L, n in LS:
                key = f"quantum|{z}|{g}|{L}"
                if key in db:
                    continue
                t = time.time()
                rng = np.random.default_rng(900 + L + int(100 * z) + int(10 * g))
                db[key] = summarise([d_quantum(L, z, g, rng) for _ in range(n)])
                json.dump(db, open(OUT, "w"), indent=1)
                print(f"z={z} gamma={g} L={L} median d={db[key]['median']:.4e}"
                      f"  [{time.time()-t:.0f}s]", flush=True)
    print("done", flush=True)
