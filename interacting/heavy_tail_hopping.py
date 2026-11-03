"""Interacting strong-dephasing chain with heavy-tailed hoppings J_i = v_i^z (alpha = 1/(2z)),
by exact diagonalization. The same disorder samples are used at every V; optional fields
h_i uniform in [-h0, h0]. Raw residuals go to results/heavy_tail_hopping_z{z}_h{h0}_L{L}.npz.

Usage: python heavy_tail_hopping.py z nsamp L1,L2,... V1,V2,... [h0]"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interacting_pilot as ed

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def slow_residual(n, moves, w, occ, L):
    """Residual of the slowest mode (interacting_pilot.slow_profile)."""
    lam, m = ed.slow_profile(n, moves, w, occ)
    return ed.residual(m, L), lam


if __name__ == "__main__":
    z = float(sys.argv[1])
    nsamp = int(sys.argv[2])
    Ls = [int(a) for a in sys.argv[3].split(",")]
    Vs = [float(a) for a in sys.argv[4].split(",")]
    h0 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    rng = np.random.default_rng(int(1000 * z + h0))
    print(f"# heavy-tailed hoppings J = v^z, z={z}, alpha={1 / (2 * z):.3f}, "
          f"predicted exponent {2 * z - 1:.3f}; h0={h0}, gamma={ed.gamma}, nsamp={nsamp}")
    print("# L   V   median|r|  q25|r|  q75|r|  median(|r_V|/|r_0|)  corr(r,r_naive)")
    for L in Ls:
        N = L // 2
        states = np.sort(ed.basis(L, N))
        occ = np.array([[(s >> i) & 1 for i in range(L)] for s in states], float)
        moves = ed.bond_moves(states, occ, L)
        n = len(states)
        Js = 1 - rng.uniform(0, 1, (nsamp, L - 1))      # uniform on (0, 1]
        Js = Js ** z
        hs = rng.uniform(-h0, h0, (nsamp, L))
        out = {}
        for V in Vs:
            rs, rn, lam = [], [], []
            base = ed.rates(moves, ed.energies(occ, np.zeros(L), V))
            for J, h in zip(Js, hs):
                if h0 > 0:
                    base = ed.rates(moves, ed.energies(occ, h, V))
                w = [b * J[i] ** 2 for i, b in enumerate(base)]
                r, l = slow_residual(n, moves, w, occ, L)
                rs.append(r); lam.append(l)
                rn.append(ed.residual(ed.naive_resistor_r(L, n, w), L))
            out[V] = (np.array(rs), np.array(rn), np.array(lam))
            a = np.abs(out[V][0])
            ratio = np.median(a / np.abs(out[Vs[0]][0]))
            c = np.corrcoef(out[V][0], out[V][1])[0, 1]
            print(f"{L:3d} {V:4.1f}  {np.median(a):.5f}  {np.quantile(a, .25):.5f}  "
                  f"{np.quantile(a, .75):.5f}  {ratio:.3f}  {c:.3f}", flush=True)
        np.savez(os.path.join(RESULTS, f"heavy_tail_hopping_z{z}_h{h0}_L{L}.npz"), J=Js, h=hs,
                 **{f"r_V{V}": out[V][0] for V in Vs}, **{f"rnaive_V{V}": out[V][1] for V in Vs},
                 **{f"lam_V{V}": out[V][2] for V in Vs})
