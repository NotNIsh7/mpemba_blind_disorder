"""Quantum residuals at gamma = 1 with bounded random fields, L = 24-192 (highstat.json):
d, r for cos(2 pi x), and r for a random blind state, with bootstrap errors."""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from density_sector import (single_particle_h, density_generator, gamma_to_Gamma,
                            slow_modes, density_profile, hydrodynamic_reference,
                            draw_fields)

CONFIG = [(24, 3000), (32, 3000), (48, 2000), (64, 1500),
          (96, 800), (128, 500), (192, 250)]
W, GAMMA = 2.0, 1.0
KMAX = 10


def unit(v):
    return v / np.linalg.norm(v)


def cosmode(L, k):
    x = (np.arange(L) + 0.5) / L
    return unit(np.cos(k * np.pi * x))


def run_L(L, n, seed):
    rng = np.random.default_rng(seed)
    ref = hydrodynamic_reference(L)
    basis = {k: cosmode(L, k) for k in range(2, KMAX + 1)}

    rec = {"eta": [], "d": [], "lam": [], "lr_mismatch": [],
           "r2": [], "r3": [], "r4": [], "rrand": [],
           "wk": np.zeros(KMAX - 1)}

    for j in range(n):
        f = draw_fields(L, W=W, dist="uniform", rng=rng)
        h, _ = single_particle_h(L, fields=f)
        G = density_generator(h, gamma_to_Gamma(GAMMA))
        w, vr, vl = slow_modes(G, k=1, want_left=True)

        phi = density_profile(vr[:, 0], L, sign_ref=ref)
        ld = np.real(np.diag(vl[:, 0].reshape((L, L), order="F")))
        if ld @ ref < 0:
            ld = -ld
        ldn = unit(ld)

        if j < 20:
            rec["lr_mismatch"].append(1 - abs(ldn @ phi))

        q = phi @ ref
        rec["eta"].append(1 - q ** 2)
        rec["d"].append(np.linalg.norm(phi - ref))
        rec["lam"].append(w[0].real)

        den = abs(ld @ ref)
        rec["r2"].append(abs(ld @ basis[2]) / den)
        rec["r3"].append(abs(ld @ basis[3]) / den)
        rec["r4"].append(abs(ld @ basis[4]) / den)
        rr = rng.normal(size=L)
        rec["rrand"].append(abs(ld @ unit(rr - (rr @ ref) * ref)) / den)

        res = phi - ref
        rec["wk"] += np.array([(res @ basis[k]) ** 2 for k in range(2, KMAX + 1)])

    out = {}
    rb = np.random.default_rng(1234)
    for k in ("eta", "d", "lam", "r2", "r3", "r4", "rrand"):
        a = np.array(rec[k])
        idx = rb.integers(0, len(a), size=(2000, len(a)))
        bm = a[idx].mean(axis=1)
        out[k] = {"mean": float(a.mean()), "sem": float(bm.std()),
                  "lo": float(np.percentile(bm, 2.5)),
                  "hi": float(np.percentile(bm, 97.5))}
    out["wk"] = (rec["wk"] / rec["wk"].sum()).tolist()
    out["lr_mismatch"] = float(np.max(np.abs(rec["lr_mismatch"])))
    out["n"] = n
    return out


def fit(Ls, means, sems, nboot=5000, seed=7):
    rng = np.random.default_rng(seed)
    Ls, means, sems = map(np.asarray, (Ls, means, sems))
    a = np.polyfit(np.log(Ls), np.log(means), 1)[0]
    draws = [np.polyfit(np.log(Ls),
                        np.log(np.maximum(means + rng.normal(0, sems), 1e-300)), 1)[0]
             for _ in range(nboot)]
    return a, float(np.std(draws)), float(np.percentile(draws, 2.5)), \
        float(np.percentile(draws, 97.5))


if __name__ == "__main__":
    t0 = time.time()
    data = {}
    for L, n in CONFIG:
        t = time.time()
        data[L] = run_L(L, n, seed=90210 + L)
        print(f"L={L:>4} n={n:>5}  eta={data[L]['eta']['mean']:.4e}  "
              f"d={data[L]['d']['mean']:.4e}  r2={data[L]['r2']['mean']:.4e}  "
              f"rrand={data[L]['rrand']['mean']:.4e}  "
              f"[{time.time()-t:.0f}s]", flush=True)
        with open("highstat.json", "w") as fh:
            json.dump({str(k): v for k, v in data.items()}, fh, indent=1)

    Ls = [L for L, _ in CONFIG]
    print(f"\ntotal {time.time()-t0:.0f}s\n")
    print("max |1-<l,r>| over sampled modes:",
          max(data[L]["lr_mismatch"] for L in Ls))

    print(f"\n{'quantity':>10} {'exponent':>22} {'expected':>10}")
    exp = {"eta": -1.0, "d": -0.5, "r2": -0.5, "r3": -0.5,
           "r4": -0.5, "rrand": -1.0, "lam": -2.0}
    for k in ("eta", "d", "r2", "r3", "r4", "rrand", "lam"):
        m = [data[L][k]["mean"] for L in Ls]
        s = [data[L][k]["sem"] for L in Ls]
        if k == "lam":
            m = [abs(x) for x in m]
        a, sd, lo, hi = fit(Ls, m, s)
        print(f"{k:>10}   {a:+.3f} +/- {sd:.3f}  [{lo:+.3f},{hi:+.3f}] "
              f"{exp[k]:>10.1f}")

    print("\nlocal slopes (successive L pairs):")
    for k in ("d", "r2", "rrand"):
        m = np.array([data[L][k]["mean"] for L in Ls])
        sl = [np.log(m[i + 1] / m[i]) / np.log(Ls[i + 1] / Ls[i])
              for i in range(len(Ls) - 1)]
        print(f"  {k:>6}: " + " ".join(f"{x:+.2f}" for x in sl))

    print("\nresidual spectral weights vs [k/(k^2-1)]^2:")
    kk = np.arange(2, KMAX + 1)
    pred = (kk / (kk ** 2 - 1.0)) ** 2
    pred /= pred.sum()
    print("      k:  " + "  ".join(f"{k:>5d}" for k in kk))
    print("   pred:  " + "  ".join(f"{p:5.3f}" for p in pred))
    for L in (Ls[0], Ls[len(Ls) // 2], Ls[-1]):
        print(f"  L={L:>3}:  " + "  ".join(f"{p:5.3f}" for p in data[L]["wk"]))
