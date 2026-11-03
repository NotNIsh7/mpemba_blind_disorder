"""Checks of dephased_xx.py."""

import numpy as np
from dephased_xx import *


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


def main():
    L, W, gamma, basis = 4, 4.0, 1.0, "pauli"
    d = 2 ** L
    H, fields = build_hamiltonian(L, W=W, seed=3, basis=basis)
    c_ops = dephasing_ops(L, gamma, basis)
    Lv = build_liouvillian(H, c_ops)
    dm = dm_labels(L)
    modes = solve_modes(Lv, dm=dm)
    rho0, O = neel_state(L), imbalance_op(L, basis=basis)
    res = analyze(modes, rho0, O)

    ok = []

    # 1. degeneracy of the steady manifold: strong U(1) gives L+1
    ns = len(modes.steady())
    ok.append(check("steady-state degeneracy = L+1", ns == L + 1, f"got {ns}, expect {L+1}"))

    # 2. all eigenvalues in the left half plane
    ok.append(check("Re(lambda) <= 0", modes.evals.real.max() < 1e-9,
                    f"max Re = {modes.evals.real.max():.2e}"))

    # 3. trace preservation: vec(I) is a left eigenvector with eigenvalue 0
    r = np.linalg.norm(vec(np.eye(d)) @ Lv)
    ok.append(check("trace preservation (I^dag L = 0)", r < 1e-10, f"|I L| = {r:.2e}"))

    # 4. biorthonormality actually achieved
    G = modes.left.conj().T @ modes.right
    err = np.abs(G - np.eye(d * d)).max()
    ok.append(check("biorthonormality l_i^dag r_j = delta_ij", err < 1e-6,
                    f"max dev {err:.2e}, cond(M) = {modes.biorth_cond:.2e}"))

    # 5. mode expansion reproduces direct exponentiation
    times = np.linspace(0, 8, 9)
    a = trajectory(res, times).real
    b = trajectory_direct(Lv, rho0, O, times)
    e = np.abs(a - b).max()
    ok.append(check("mode expansion == expm trajectory", e < 1e-8, f"max dev {e:.2e}"))

    # 6. I(0) equals the bare Neel imbalance
    i0 = np.trace(O @ rho0).real
    ok.append(check("I(0) matches direct trace", abs(a[0] - i0) < 1e-9,
                    f"{a[0]:.6f} vs {i0:.6f}"))

    # 7. the Neel state has strictly zero overlap with every coherence mode
    mx = np.abs(res["c"][res["dm"] != 0]).max()
    ok.append(check("Neel c_i = 0 for all Delta-m != 0", mx < 1e-12, f"max |c| = {mx:.2e}"))

    # 8. coherence rates are pinned at the analytic floor
    coh = modes.evals[(res["dm"] != 0)].real
    floor = -coherence_rate_floor(gamma, basis)
    ok.append(check("coherence floor matches analytics", abs(coh.max() - floor) < 1e-9,
                    f"{coh.max():.4f} vs {floor:.4f}"))

    # 9. blocks do not mix: modes live in a single Delta-m sector
    worst = 0.0
    for i in range(0, d * d, 37):
        w = np.abs(modes.right[:, i]) ** 2
        w /= w.sum()
        worst = max(worst, 1 - max(w[dm == s].sum() for s in np.unique(dm)))
    ok.append(check("each mode confined to one Delta-m block", worst < 1e-10,
                    f"max leakage {worst:.2e}"))

    # 10. block-restricted construction reproduces the full Delta-m = 0 spectrum
    Lb, idx = block_liouvillian(H, c_ops, L, dm_target=0, sparse=False)

    def canon(z):
        # lexsort on rounded (real, imag): sort_complex is unstable for near-degenerate real parts
        z = np.asarray(z)
        o = np.lexsort((np.round(z.imag, 9), np.round(z.real, 9)))
        return z[o]

    full0, blk0 = canon(modes.evals[modes.dm == 0]), canon(np.linalg.eigvals(Lb))
    e = np.abs(full0 - blk0).max()
    ok.append(check("Delta-m=0 block spectrum == full restriction", e < 1e-8,
                    f"max dev {e:.2e}, dim {len(idx)} vs {d*d}"))

    # 11. sparse slow modes agree with dense
    w, _ = slow_modes_sparse(Lb, k=6, n_discard=L + 1)
    ev = np.linalg.eigvals(Lb)
    dense_slow = np.sort(ev[np.abs(ev) >= 1e-9].real)[::-1][:6]
    e = np.abs(np.sort(w.real)[::-1] - dense_slow).max()
    ok.append(check("sparse Arnoldi slow modes == dense", e < 1e-8, f"max dev {e:.2e}"))

    print(f"\n{sum(ok)}/{len(ok)} passed")


if __name__ == "__main__":
    main()
