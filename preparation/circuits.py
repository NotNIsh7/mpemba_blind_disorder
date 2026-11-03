"""Gate-level checks of the preparation and measurement of the blind states, by state-vector simulation:
1 counter circuit, 2 random-Z dephasing, 3 four-qubit circuit, 4 heralded preparation,
5 affine states and SWAP loading, 6 measurement of r on the interacting lattice gas (L = 10).
Output: results/preparation_checks.txt.

Usage: python circuits.py"""
import itertools
import os
import sys
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interacting"))
import interacting_pilot as ed   # noqa: E402


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def configs(L, Q):
    return [eta for eta in itertools.product((0, 1), repeat=L) if sum(eta) == Q]


def target_state(w, Q):
    """Amplitudes of |Psi_b> on the full 2^L register (bit i = site i)."""
    L = len(w)
    psi = np.zeros(2 ** L)
    for eta in configs(L, Q):
        idx = sum(e << i for i, e in enumerate(eta))
        psi[idx] = np.sqrt(np.prod(np.asarray(w) ** np.asarray(eta)))
    return psi / np.linalg.norm(psi)


def partition_table(w, Q):
    """E[i][q] = sum over occupations of sites i..L-1 with q particles of prod w^eta."""
    L = len(w)
    E = np.zeros((L + 1, Q + 1))
    E[L, 0] = 1.0
    for i in range(L - 1, -1, -1):
        for q in range(Q + 1):
            E[i, q] = E[i + 1, q] + (w[i] * E[i + 1, q - 1] if q > 0 else 0.0)
    return E


# 1. counter circuit
def counter_circuit(w, Q):
    """Simulate W_L V_L ... W_1 V_1 on L site qubits plus an m-qubit counter."""
    L = len(w)
    m = max(1, int(np.ceil(np.log2(Q + 1))))
    E = partition_table(w, Q)
    dim = 2 ** (L + m)
    psi = np.zeros(dim)
    psi[Q << L] = 1.0                                   # |0^L>|Q>_C
    idx = np.arange(dim)
    cnt = idx >> L
    for i in range(L):
        # V_i: on each counter value q, rotate site i by theta_{i,q}
        new = psi.copy()
        for q in range(2 ** m):
            if q == 0 or q > Q or E[i, q] == 0:
                theta = 0.0
            else:
                p = w[i] * E[i + 1, q - 1] / E[i, q]
                theta = 2 * np.arcsin(np.sqrt(min(max(p, 0.0), 1.0)))
            sel = idx[(cnt == q) & (((idx >> i) & 1) == 0)]
            a0, a1 = psi[sel], psi[sel | (1 << i)]
            U = ry(theta)
            new[sel] = U[0, 0] * a0 + U[0, 1] * a1
            new[sel | (1 << i)] = U[1, 0] * a0 + U[1, 1] * a1
        psi = new
        # W_i: counter -> counter - eta_i (mod 2^m), a permutation of basis states
        eta = (idx >> i) & 1
        tgt = (idx & ((1 << L) - 1)) | ((((cnt - eta) % (2 ** m))) << L)
        new = np.zeros(dim)
        new[tgt] = psi
        psi = new
    return psi, m


# 3. four-qubit circuit
def gate(psi, U, targets, controls=(), L=4):
    """Apply a single-qubit U on `targets` conditioned on all `controls` being 1."""
    out = psi.copy()
    idx = np.arange(2 ** L)
    t = targets
    mask = np.ones(2 ** L, bool)
    for c in controls:
        mask &= ((idx >> c) & 1) == 1
    sel = idx[mask & (((idx >> t) & 1) == 0)]
    a0, a1 = psi[sel], psi[sel | (1 << t)]
    out[sel] = U[0, 0] * a0 + U[0, 1] * a1
    out[sel | (1 << t)] = U[1, 0] * a0 + U[1, 1] * a1
    return out


X = np.array([[0.0, 1.0], [1.0, 0.0]])


def four_qubit(a):
    """Compiled circuit of the notes for L = 4, Q = 2, f = (1,-1,-1,1)/sqrt(2)."""
    w = np.exp(a / np.sqrt(2))
    Z = w ** 2 + w ** -2 + 4
    th1 = 2 * np.arcsin(np.sqrt((w ** 2 + 2) / Z))
    th20 = 2 * np.arcsin(np.sqrt((w ** -2 + 1) / (w ** -2 + 2)))
    th21 = 2 * np.arcsin(np.sqrt(1 / (w ** 2 + 2)))
    beta = 2 * np.arcsin(np.sqrt(1 / (1 + w ** 2)))
    al, de = (th20 + th21) / 2, (th20 - th21) / 2
    psi = np.zeros(16)
    psi[0] = 1.0
    psi = gate(psi, ry(th1), 0)                       # site 1
    psi = gate(psi, ry(al), 1)                        # U2 = Ry(alpha) CNOT Ry(delta) CNOT
    psi = gate(psi, X, 1, (0,))
    psi = gate(psi, ry(de), 1)
    psi = gate(psi, X, 1, (0,))
    psi = gate(psi, ry(np.pi), 2)                     # U3 = Ry(pi), C-Ry(beta-pi) x2, CC-Ry(pi-2beta)
    psi = gate(psi, ry(beta - np.pi), 2, (0,))
    psi = gate(psi, ry(beta - np.pi), 2, (1,))
    psi = gate(psi, ry(np.pi - 2 * beta), 2, (0, 1))
    for c in (0, 1, 2):                               # parity onto site 4
        psi = gate(psi, X, 3, (c,))
    f = np.array([1, -1, -1, 1]) / np.sqrt(2)
    return psi, (th1, th20, th21, beta), np.exp(a * f)


def densities(psi, L):
    p = psi ** 2
    idx = np.arange(2 ** L)
    return np.array([p[((idx >> i) & 1) == 1].sum() for i in range(L)])


def z_dephase(psi, L):
    rho = np.outer(psi, psi)
    idx = np.arange(2 ** L)
    out = np.zeros_like(rho)
    for z in range(2 ** L):
        s = (-1.0) ** np.array([bin(k & z).count("1") for k in idx])
        out += np.outer(s, s) * rho
    return out / 2 ** L


# 5. affine states, SWAP loading
def swap_loading(eta):
    """Swaps taking |1^Q 0^{L-Q}> to |eta>: pair excess occupied sites with missing ones."""
    L, Q = len(eta), sum(eta)
    start = [1] * Q + [0] * (L - Q)
    excess = [i for i in range(L) if start[i] == 1 and eta[i] == 0]
    missing = [i for i in range(L) if start[i] == 0 and eta[i] == 1]
    state = start[:]
    for i, j in zip(excess, missing):
        state[i], state[j] = state[j], state[i]
    assert state == list(eta)
    return len(excess)


def main():
    rng = np.random.default_rng(11)
    print("# 1. counter circuit vs target |Psi_b>   (L, Q, bias): max amplitude error, counter weight off |0>")
    worst = 0.0
    for L, Q in ((4, 2), (6, 3), (7, 2), (8, 4), (9, 5)):
        x = (np.arange(L) + 0.5) / L
        for name, b in (("cos2pix a=0.7", 0.7 * np.cos(2 * np.pi * x)), ("random", rng.normal(0, 1.5, L))):
            w = np.exp(b)
            psi, m = counter_circuit(w, Q)
            off = np.sum(psi[np.arange(len(psi)) >> L != 0] ** 2)
            err = np.max(np.abs(psi[: 2 ** L] - target_state(w, Q)))
            worst = max(worst, err, off)
            print(f"   L={L} Q={Q} {name:14s}: {err:.1e}  {off:.1e}")
    print(f"# counter circuit: worst error {worst:.1e}")

    print("# 2. random-Z dephasing of |Psi_b> gives rho_b (max entrywise error)")
    for L, Q in ((4, 2), (6, 3)):
        w = np.exp(rng.normal(0, 1, L))
        psi = target_state(w, Q)
        rho = z_dephase(psi, L)
        print(f"   L={L} Q={Q}: {np.max(np.abs(rho - np.diag(psi ** 2))):.1e}")

    print("# 3. four-qubit circuit: max amplitude error vs Eq. (Psi_b) at a = -1, 0, 0.2, 1")
    for a in (-1.0, 0.0, 0.2, 1.0):
        psi, ang, w = four_qubit(a)
        err = np.max(np.abs(psi - target_state(w, 2)))
        rho = z_dephase(psi, 4)
        derr = np.max(np.abs(rho - np.diag(psi ** 2)))
        print(f"   a={a:+.1f}: amplitude {err:.1e}, dephased density matrix {derr:.1e}")
    psi, ang, _ = four_qubit(0.2)
    print("   angles at a=0.2 (theta1, theta2|0, theta2|1, beta): " + ", ".join(f"{t:.6f}" for t in ang))
    print("   densities at a=0.2: " + ", ".join(f"{n:.6f}" for n in densities(psi, 4)))

    print("# 4. heralded preparation: success probability and optimum")
    L, Q = 10, 5
    x = (np.arange(L) + 0.5) / L
    w = np.exp(0.5 * np.cos(2 * np.pi * x))
    ZQ = partition_table(w, Q)[0, Q]

    def herald(z):
        amp = np.ones(2 ** L)
        idx = np.arange(2 ** L)
        for i in range(L):
            p = z * w[i] / (1 + z * w[i])
            occ = ((idx >> i) & 1) == 1
            amp *= np.where(occ, np.sqrt(p), np.sqrt(1 - p))
        keep = np.array([bin(k).count("1") == Q for k in idx])
        return np.sum(amp[keep] ** 2), amp * keep
    for z in (0.5, 1.0, 2.0):
        ps, amp = herald(z)
        formula = z ** Q * ZQ / np.prod(1 + z * w)
        fid = abs(amp @ target_state(w, Q)) ** 2 / ps
        print(f"   z={z}: P_succ {ps:.6f} (formula {formula:.6f}), fidelity after herald {fid:.12f}")
    zopt = brentq(lambda z: np.sum(z * w / (1 + z * w)) - Q, 1e-3, 1e3)
    grid = np.exp(np.linspace(np.log(zopt) - 0.3, np.log(zopt) + 0.3, 61))
    zbest = grid[np.argmax([herald(z)[0] for z in grid])]
    print(f"   optimum: sum p_i = Q at z={zopt:.4f}; grid maximum at z={zbest:.4f}; "
          f"P_succ={herald(zopt)[0]:.4f}, [2 pi L nu(1-nu)]^(-1/2)={1 / np.sqrt(2 * np.pi * L * 0.25):.4f}")

    print("# 5. affine states P(1 + eps A_f)P/d: density linearity, sampler, SWAP loading")
    L, Q = 8, 4
    x = (np.arange(L) + 0.5) / L
    f = np.cos(2 * np.pi * x)
    nu = Q / L
    chiL = nu * (1 - nu) * L / (L - 1)
    cfg = np.array(configs(L, Q))
    A = cfg @ f - nu * f.sum()
    Mf = np.max(np.abs(A))
    for frac in (0.5, 1.0):
        eps = frac / Mf
        p = (1 + eps * A) / len(cfg)
        n = p @ cfg
        print(f"   eps = {frac} / M_f: min probability {p.min():.3f}, "
              f"max |<n_i> - nu - eps chi_L f_i| = {np.max(np.abs(n - nu - eps * chiL * f)):.1e}, "
              f"mean acceptance {np.mean((1 + eps * A) / (1 + eps * Mf)):.4f}")
    nswap = max(swap_loading(tuple(e)) for e in cfg)
    print(f"   SWAP loading: at most {nswap} swaps over all {len(cfg)} configurations (min(Q, L-Q) = {min(Q, L - Q)})")
    print(f"   M_f = {Mf:.4f} for L={L}, Q={Q} (sum of the Q largest entries of f)")

    print("# 6. measuring r: late-time ratio of density responses, interacting lattice gas")
    L, Q, V = 10, 5, 4.0
    states = np.sort(ed.basis(L, Q))
    occ = np.array([[(s >> i) & 1 for i in range(L)] for s in states], float)
    moves = ed.bond_moves(states, occ, L)
    x = (np.arange(L) + 0.5) / L
    A1 = occ @ np.cos(np.pi * x) - 0.5 * np.cos(np.pi * x).sum()
    A2 = occ @ np.cos(2 * np.pi * x) - 0.5 * np.cos(2 * np.pi * x).sum()
    rng2 = np.random.default_rng(4)
    for s in range(4):
        h = rng2.uniform(-4, 4, L)
        wts = ed.rates(moves, ed.energies(occ, h, V))
        M = ed.generator(len(states), moves, wts).toarray()
        lam1, m = ed.slow_profile(len(states), moves, wts, occ)
        r = ed.residual(m, L)
        ev, U = np.linalg.eigh(M)                       # M is symmetric: uniform stationary state
        order = np.argsort(-ev)
        ev, U = ev[order], U[:, order]
        gap = ev[1] - ev[2]
        out = []
        for eps_frac in (1.0, 0.3):
            ratios = []
            for tau in (2, 5, 10):                      # time in units of 1 / (lambda_1 - lambda_2)
                t = tau / gap
                prop = U @ np.diag(np.exp(ev * t)) @ U.T
                eps = eps_frac / max(np.abs(A1).max(), np.abs(A2).max())
                R = []
                for Ak in (A1, A2):
                    plus = prop @ ((1 + eps * Ak) / len(states))
                    minus = prop @ ((1 - eps * Ak) / len(states))
                    R.append((A1 @ plus - A1 @ minus) / (2 * eps))
                ratios.append(R[1] / R[0])
            out.append(ratios)
        print(f"   sample {s}: r = {r:+.9f}; R12/R11 at t = 2, 5, 10 /(lam1-lam2): "
              + ", ".join(f"{q:+.9f}" for q in out[0])
              + f"; same at eps/3: {', '.join(f'{q:+.9f}' for q in out[1])}")


if __name__ == "__main__":
    main()
