"""Quantum exponents of d and r for bounded random fields across dephasing rates
(gammascan.json).

Usage: python gamma_scan.py <gamma>"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from density_sector import (single_particle_h, density_generator, gamma_to_Gamma,
                            slow_modes, density_profile, hydrodynamic_reference,
                            draw_fields, bond_rates, D_eff_harmonic, predicted_gap)

LS = [(24, 400), (32, 400), (48, 300), (64, 300), (96, 200)]
W, KMAX = 2.0, 8
OUT = "gammascan.json"


def unit(v):
    return v / np.linalg.norm(v)


def cosmode(L, k):
    x = (np.arange(L) + 0.5) / L
    return unit(np.cos(k * np.pi * x))


def run(L, n, gamma, seed):
    rng = np.random.default_rng(seed)
    ref = hydrodynamic_reference(L)
    basis = {k: cosmode(L, k) for k in range(2, KMAX + 1)}
    eta, d, lam, r2, rr, ratio_pred, mism = [], [], [], [], [], [], []
    wk = np.zeros(KMAX - 1)

    for j in range(n):
        f = draw_fields(L, W=W, dist="uniform", rng=rng)
        h, _ = single_particle_h(L, fields=f)
        G = density_generator(h, gamma_to_Gamma(gamma))
        w, vr, vl = slow_modes(G, k=1, want_left=True)

        phi = density_profile(vr[:, 0], L, sign_ref=ref)
        ld = np.real(np.diag(vl[:, 0].reshape((L, L), order="F")))
        if ld @ ref < 0:
            ld = -ld
        if j < 20:
            mism.append(1 - abs(unit(ld) @ phi))

        q = phi @ ref
        eta.append(1 - q ** 2)
        d.append(np.linalg.norm(phi - ref))
        lam.append(w[0].real)

        den = abs(ld @ ref)
        r2.append(abs(ld @ basis[2]) / den)
        rv = rng.normal(size=L)
        rr.append(abs(ld @ unit(rv - (rv @ ref) * ref)) / den)

        res = phi - ref
        wk += np.array([(res @ basis[k]) ** 2 for k in range(2, KMAX + 1)])

        # strong-dephasing (harmonic-mean) prediction of the rate
        ratio_pred.append(w[0].real /
                          predicted_gap(D_eff_harmonic(bond_rates(f, 1.0, gamma)), L))

    m = lambda a: float(np.mean(a))
    s = lambda a: float(np.std(a) / np.sqrt(len(a)))
    return {"n": n, "eta": m(eta), "eta_e": s(eta), "d": m(d), "d_e": s(d),
            "lam": m(lam), "L2lam": L ** 2 * m(lam),
            "r2": m(r2), "r2_e": s(r2), "rrand": m(rr), "rrand_e": s(rr),
            "wk": (wk / wk.sum()).tolist(),
            "harmonic_ratio": m(ratio_pred),
            "lr_mismatch": float(np.max(np.abs(mism)))}


if __name__ == "__main__":
    g = float(sys.argv[1])
    db = json.load(open(OUT)) if os.path.exists(OUT) else {}
    key = f"{g:g}"
    db.setdefault(key, {})
    for L, n in LS:
        t = time.time()
        db[key][str(L)] = run(L, n, g, seed=31337 + L + int(1000 * g))
        r = db[key][str(L)]
        print(f"  gamma={g:<6g} L={L:>4}  eta={r['eta']:.3e}  d={r['d']:.3e}  "
              f"r2={r['r2']:.3e}  L^2lam={r['L2lam']:+.3f}  "
              f"harm={r['harmonic_ratio']:.3f}  lr={r['lr_mismatch']:.0e}"
              f"  [{time.time()-t:.0f}s]", flush=True)
        json.dump(db, open(OUT, "w"), indent=1)
