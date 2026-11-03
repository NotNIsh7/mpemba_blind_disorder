"""Interacting strong-dephasing chain by exact diagonalization (L <= 20).

Classical limit of the dephased XXZ chain: a particle hops across bond i at rate
32 J^2 gamma / (16 gamma^2 + dE^2), with dE the change of diagonal energy (fields and
interaction). The stationary state of each particle-number sector is uniform.
r = sum_i b_i m_i / sum_i c_i m_i, with m_i = E[f_1(eta) eta_i], b = cos(2 pi x), c = cos(pi x)."""
import itertools, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

J, gamma = 1.0, 4.0


def basis(L, N):
    states = []
    for occ in itertools.combinations(range(L), N):
        s = 0
        for o in occ:
            s |= 1 << o
        states.append(s)
    return np.array(states, dtype=np.int64)


def energies(occ, h, V):
    sz = 2 * occ - 1
    return V * np.sum(sz[:, :-1] * sz[:, 1:], axis=1) + sz @ h


def bond_moves(states, occ, L):
    """For each bond: indices of source states with a hop, and target indices."""
    moves = []
    for i in range(L - 1):
        src = np.nonzero(occ[:, i] != occ[:, i + 1])[0]
        tgt = np.searchsorted(states, states[src] ^ ((1 << i) | (1 << (i + 1))))
        moves.append((src, tgt))
    return moves


def rates(moves, E):
    out = []
    for src, tgt in moves:
        dE = E[tgt] - E[src]
        out.append(32 * J**2 * gamma / (16 * gamma**2 + dE**2))
    return out


def generator(n, moves, w):
    rows = np.concatenate([t for _, t in moves])
    cols = np.concatenate([s for s, _ in moves])
    vals = np.concatenate(w)
    M = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return M - sp.diags(np.asarray(M.sum(axis=0)).ravel())


def slow_profile(n, moves, w, occ):
    """Slowest nonstationary mode and its density profile (dense for small sectors,
    otherwise Lanczos started from the averaged-rate resistor-chain mode)."""
    M = generator(n, moves, w)
    if n <= 3000:
        ev, v = np.linalg.eigh(M.toarray())
        return ev[-2], occ.T @ v[:, -2] / n
    L = occ.shape[1]
    guess = occ @ naive_resistor_r(L, n, w)
    guess -= guess.mean()
    ev, v = sla.eigsh(M, k=4, which="LA", tol=1e-10, v0=guess, ncv=min(n, 40))
    order = np.argsort(-ev)
    f1 = v[:, order[1]]
    return ev[order[1]], occ.T @ f1 / n


def naive_resistor_r(L, n, w):
    G = np.array([x.sum() for x in w]) / n
    K = np.zeros((L, L))
    for i in range(L - 1):
        K[i, i] -= G[i]; K[i + 1, i + 1] -= G[i]
        K[i, i + 1] += G[i]; K[i + 1, i] += G[i]
    return np.linalg.eigh(K)[1][:, -2]


def residual(m, L):
    x = (np.arange(L) + 0.5) / L
    return np.dot(np.cos(2 * np.pi * x), m) / np.dot(np.cos(np.pi * x), m)


if __name__ == "__main__":
    h0 = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    Ls = [int(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else [8, 10, 12, 14, 16]
    Vs = [float(a) for a in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.0, 1.0, 4.0]
    rng = np.random.default_rng(1)
    print(f"# h0={h0} gamma={gamma} nsamp={nsamp}")
    print("# L V  rms_r  rms_r*sqrt(L)  L^2|lam1|  corr(r,r_naive)  slope  clean_r")
    for L in Ls:
        N = L // 2
        states = np.sort(basis(L, N))
        occ = np.array([[(s >> i) & 1 for i in range(L)] for s in states], float)
        moves = bond_moves(states, occ, L)
        n = len(states)
        hs = [rng.uniform(-h0, h0, L) for _ in range(nsamp)]
        for V in Vs:
            rs, rn, lams = [], [], []
            for h in hs:
                w = rates(moves, energies(occ, h, V))
                lam, m = slow_profile(n, moves, w, occ)
                rs.append(residual(m, L)); lams.append(lam)
                rn.append(residual(naive_resistor_r(L, n, w), L))
            rs, rn = np.array(rs), np.array(rn)
            wc = rates(moves, energies(occ, np.zeros(L), V))
            rc = residual(slow_profile(n, moves, wc, occ)[1], L)
            c = np.corrcoef(rs, rn)[0, 1]
            slope = np.dot(rs, rn) / np.dot(rn, rn)
            rms = np.sqrt(np.mean(rs**2))
            print(f"{L:3d} {V:4.1f} {rms:.4f} {rms*np.sqrt(L):.4f} "
                  f"{L**2*np.mean(np.abs(lams)):.3f} {c:.3f} {slope:.3f} {rc:.1e}", flush=True)
