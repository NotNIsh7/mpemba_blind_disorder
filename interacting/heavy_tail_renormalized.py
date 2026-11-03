"""Resistor chain with configuration-averaged interacting rates, heavy-tailed hoppings,
L = 64-4096: residual exponent at V = 0 and V = 4.

Usage: python heavy_tail_renormalized.py z nsamp h0"""
import os
import sys
import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmrg_slow_mode import rate, residual


def averaged_conductances(h, V, Jb):
    """G_i = E[rate_i 1{hop}] under the uniform half-filled measure (large-L form:
    neighbours independent with occupation 1/2, bond active with probability 1/2)."""
    L = len(h)
    G = np.zeros(L - 1)
    for i in range(L - 1):
        sides = [s for s in (i - 1, i + 2) if 0 <= s < L]
        for occ in np.ndindex(*(2,) * len(sides)):
            s = dict(zip(sides, (2 * o - 1 for o in occ)))
            dE = 2 * (h[i + 1] - h[i]) + 2 * V * (s.get(i + 2, 0) - s.get(i - 1, 0))
            G[i] += 0.5 * 0.5 ** len(sides) * Jb[i] ** 2 * rate(dE)
    return G


def chain_residual(G):
    L = len(G) + 1
    d = np.r_[G, 0] + np.r_[0, G]
    w, v = eigh_tridiagonal(d, -G, select="i", select_range=(1, 1))
    return residual(v[:, 0])


if __name__ == "__main__":
    z, nsamp = float(sys.argv[1]), int(sys.argv[2])
    h0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    rng = np.random.default_rng(5)
    Ls = [64, 256, 1024, 4096]
    pred = max(-0.5, 2 * z - 1)    # predicted: max(-1/2, 1/alpha - 1), alpha = 1/(2z)
    print(f"# renormalized resistor chain, J = v^z, z={z}, alpha={1 / (2 * z):.3f}, "
          f"predicted exponent {pred:.3f}, h0={h0}, nsamp={nsamp}")
    print("#    L   median|r| V=0   median|r| V=4   median ratio")
    med = {0.0: [], 4.0: []}
    for L in Ls:
        r = {0.0: [], 4.0: []}
        for _ in range(nsamp):
            Jb = (1 - rng.uniform(0, 1, L - 1)) ** z
            h = rng.uniform(-h0, h0, L)
            for V in r:
                r[V].append(chain_residual(averaged_conductances(h, V, Jb)))
        a0, a4 = np.abs(r[0.0]), np.abs(r[4.0])
        med[0.0].append(np.median(a0)); med[4.0].append(np.median(a4))
        print(f"{L:6d}   {np.median(a0):.5f}   {np.median(a4):.5f}   {np.median(a4 / a0):.4f}", flush=True)
    for V in med:
        slope = np.polyfit(np.log(Ls), np.log(med[V]), 1)[0]
        print(f"# V={V}: fitted exponent {slope:.3f} (predicted {pred:.3f})")
