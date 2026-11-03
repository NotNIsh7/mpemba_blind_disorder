#!/usr/bin/env python3
"""Exponent of d for Pareto resistances with chosen alpha, L = 64-4096."""
import numpy as np
from blind_scaling import slow_mode, cos_basis

LS = [64, 128, 256, 512, 1024, 2048, 4096]
N = {64: 3000, 128: 3000, 256: 2000, 512: 1500,
     1024: 800, 2048: 400, 4096: 200}


def d_exponent(alpha, seed0=2024):
    ds = []
    for L in LS:
        rng = np.random.default_rng(seed0 + L)
        ref = cos_basis(L)[0]
        acc = []
        for _ in range(N[L]):
            rho = rng.uniform(1e-15, 1.0, size=L - 1) ** (-1.0 / alpha)
            _, phi = slow_mode(1.0 / rho)
            phi /= np.linalg.norm(phi)
            if phi @ ref < 0:
                phi = -phi
            acc.append(np.linalg.norm(phi - ref))
        ds.append(np.mean(acc))
    return np.polyfit(np.log(LS), np.log(ds), 1)[0], ds


if __name__ == "__main__":
    print(f"{'alpha':>7} {'d exponent':>12} {'asymptotic':>11} "
          f"{'matching z':>11} {'bond result':>12}")
    bond = {2.5: -0.443, 2.0: -0.394, 1.667: -0.322,
            1.429: -0.244, 1.25: -0.175, 1.111: -0.122}
    for a in (4.0, 3.0, 2.5, 2.0, 1.8, 1.667, 1.5, 1.429, 1.3, 1.25, 1.111):
        e, _ = d_exponent(a)
        pred = -0.5 if a > 2 else 1.0 / a - 1.0
        z = 1.0 / (2 * a)
        b = f"{bond[a]:+.3f}" if a in bond else "-"
        print(f"{a:>7.3f} {e:>+12.3f} {pred:>+11.3f} {z:>11.3f} {b:>12}")
    print("\nIdentical exponents at matched alpha from two unrelated disorder")
    print("models would establish the tail index as the control variable.")
