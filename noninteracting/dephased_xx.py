"""Full Liouvillian of the dephased disordered XX chain (exact diagonalization, small L).

H = J sum_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1}) + sum_i h_i Sz_i, jump operators sqrt(gamma) Z_i.
Biorthonormal left/right modes, Delta-m sectors and trajectories.
Vectorization is column-stacking: vec(rho)[a + d*b] = rho[a, b]."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig, expm
from scipy.sparse import csr_matrix, identity as sp_identity, kron as sp_kron
from scipy.sparse.linalg import eigs


SIGMAX = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMAY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMAZ = np.array([[1, 0], [0, -1]], dtype=complex)
SX, SY, SZ = SIGMAX / 2, SIGMAY / 2, SIGMAZ / 2
ID2 = np.eye(2, dtype=complex)

# 'spin'  : S^a = sigma^a / 2   (textbook spin-1/2)
# 'pauli' : S^a = sigma^a
BASES = {"spin": (SX, SY, SZ), "pauli": (SIGMAX, SIGMAY, SIGMAZ)}


def site_op(op: np.ndarray, i: int, L: int) -> np.ndarray:
    """Embed a single-site operator at site i of an L-site chain."""
    out = np.array([[1.0 + 0j]])
    for k in range(L):
        out = np.kron(out, op if k == i else ID2)
    return out



def build_hamiltonian(L: int, J: float = 1.0, W: float = 1.0,
                      seed: int | None = None, pbc: bool = False,
                      fields: np.ndarray | None = None,
                      basis: str = "pauli"):
    """H = J sum (SxSx + SySy) + sum h_i Sz_i.  Returns (H, fields)."""
    X, Y, Z = BASES[basis]
    if fields is None:
        rng = np.random.default_rng(seed)
        fields = rng.uniform(-W, W, size=L)
    fields = np.asarray(fields, dtype=float)

    d = 2 ** L
    H = np.zeros((d, d), dtype=complex)

    bonds = range(L) if pbc else range(L - 1)
    for i in bonds:
        j = (i + 1) % L
        H += J * (site_op(X, i, L) @ site_op(X, j, L)
                  + site_op(Y, i, L) @ site_op(Y, j, L))
    for i in range(L):
        H += fields[i] * site_op(Z, i, L)
    return H, fields


def dephasing_ops(L: int, gamma: float = 1.0, basis: str = "pauli"):
    """Jump operators sqrt(gamma) Z_i."""
    Z = BASES[basis][2]
    return [np.sqrt(gamma) * site_op(Z, i, L) for i in range(L)]


def coherence_rate_floor(gamma: float = 1.0, basis: str = "pauli") -> float:
    """Analytic decay rate of the slowest possible coherence (n = 1 site)."""
    return gamma / 2 if basis == "spin" else 2 * gamma



def build_liouvillian(H: np.ndarray, c_ops) -> np.ndarray:
    """
    L[rho] = -i[H, rho] + sum_k ( L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho} )
    vectorized with column stacking.
    """
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    Lv = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for Lk in c_ops:
        LdL = Lk.conj().T @ Lk
        Lv += (np.kron(Lk.conj(), Lk)
               - 0.5 * np.kron(I, LdL)
               - 0.5 * np.kron(LdL.T, I))
    return Lv


def vec(rho: np.ndarray) -> np.ndarray:
    return rho.flatten(order="F")


def unvec(v: np.ndarray, d: int) -> np.ndarray:
    return v.reshape((d, d), order="F")



def magnetization_table(L: int) -> np.ndarray:
    """Sz eigenvalue of each computational basis state, in units of 1/2 (integers)."""
    states = np.arange(2 ** L)
    bits = ((states[:, None] >> np.arange(L - 1, -1, -1)[None, :]) & 1)
    # bit 0 -> spin up (+1/2), bit 1 -> spin down (-1/2); integer units of 1/2
    return (L - 2 * bits.sum(axis=1))


def dm_labels(L: int) -> np.ndarray:
    """
    Delta-m label for each vectorized basis element, in units of 1/2.
    Element k of vec corresponds to |a><b| with a = k % d, b = k // d.
    """
    d = 2 ** L
    m = magnetization_table(L)
    a = np.arange(d ** 2) % d
    b = np.arange(d ** 2) // d
    return m[a] - m[b]


def hamming_table(L: int) -> np.ndarray:
    """Number of sites on which |a> and |b> differ, for each vec element."""
    d = 2 ** L
    states = np.arange(d)
    a = np.arange(d ** 2) % d
    b = np.arange(d ** 2) // d
    x = states[a] ^ states[b]
    return np.array([bin(v).count("1") for v in x])



class Modes:
    """
    Biorthonormal spectral decomposition of a non-normal generator.

    evals[i]        eigenvalue lambda_i
    right[:, i]     vec(r_i)
    left[:, i]      vec(l_i),  normalized so that  Tr[l_i^dag r_j] = delta_ij
    dm[i]           Delta-m sector the mode lives in (units of 1/2), or None
    """

    def __init__(self, evals, right, left, dm=None, biorth_cond=None):
        self.evals = evals
        self.right = right
        self.left = left
        self.dm = dm
        self.biorth_cond = biorth_cond

    def steady(self, tol=1e-9):
        return np.where(np.abs(self.evals) < tol)[0]

    def decaying(self, tol=1e-9):
        idx = np.where(np.abs(self.evals) >= tol)[0]
        # slowest first: largest (least negative) real part
        return idx[np.argsort(-self.evals[idx].real)]


def solve_modes(Lv: np.ndarray, dm=None, biorth: bool = True) -> Modes:
    """Dense eigendecomposition with biorthonormalized left eigenvectors."""
    w, vl, vr = eig(Lv, left=True, right=True)

    cond = None
    if biorth:
        M = vl.conj().T @ vr
        cond = np.linalg.cond(M)
        vl = vl @ np.linalg.inv(M).conj().T

    dm_mode = None
    if dm is not None:
        dm_mode = np.array([_dominant_sector(vr[:, i], dm)
                            for i in range(vr.shape[1])])
    return Modes(w, vr, vl, dm_mode, cond)


def _dominant_sector(v: np.ndarray, dm: np.ndarray):
    """Delta-m sector carrying the weight of a mode (blocks do not mix)."""
    weight = np.abs(v) ** 2
    sectors = np.unique(dm)
    tot = np.array([weight[dm == s].sum() for s in sectors])
    return int(sectors[np.argmax(tot)])



def neel_state(L: int, start_up: bool = True) -> np.ndarray:
    """|up down up down ...><...| as a density matrix."""
    bits = [(0 if (i % 2 == 0) == start_up else 1) for i in range(L)]
    idx = int("".join(str(b) for b in bits), 2)
    rho = np.zeros((2 ** L, 2 ** L), dtype=complex)
    rho[idx, idx] = 1.0
    return rho


def imbalance_op(L: int, basis: str = "pauli") -> np.ndarray:
    """I = (1/L) sum_i (-1)^i Sz_i."""
    Z = BASES[basis][2]
    O = np.zeros((2 ** L, 2 ** L), dtype=complex)
    for i in range(L):
        O += ((-1) ** i) * site_op(Z, i, L)
    return O / L


def analyze(modes: Modes, rho0: np.ndarray, O: np.ndarray, tol: float = 1e-10):
    """
    Mode-by-mode visibility for the pair (rho0, O).

        c_i = Tr[l_i^dag rho0]      overlap of the initial state with mode i
        o_i = Tr[O r_i]             overlap of the observable with mode i

    A mode contributes to <O>(t) only when both are nonzero.
    Returns a dict of arrays indexed by mode.
    """
    v0, vO = vec(rho0), vec(O)
    c = modes.left.conj().T @ v0
    o = modes.right.conj().T @ vO
    o = o.conj()  # Tr[O r_i] = vec(O)^dag vec(r_i) for Hermitian O
    return {
        "evals": modes.evals,
        "c": c,
        "o": o,
        "weight": np.abs(c * o),
        "visible": (np.abs(c) > tol) & (np.abs(o) > tol),
        "dm": modes.dm,
    }


def relevant_gaps(res: dict, tol: float = 1e-9):
    """
    Slowest overall / slowest visible / slowest coherence decay rates.
    'coherence' means Delta-m != 0. Requires modes built with dm labels.
    """
    ev, vis, dm = res["evals"], res["visible"], res["dm"]
    dec = np.abs(ev) >= tol

    def slowest(mask):
        m = mask & dec
        return ev[m].real.max() if m.any() else np.nan

    out = {
        "slowest_overall": slowest(np.ones_like(dec)),
        "slowest_visible": slowest(vis),
    }
    if dm is not None:
        out["slowest_coherence"] = slowest(dm != 0)
        out["slowest_dm0"] = slowest(dm == 0)
    return out


def trajectory(res: dict, times: np.ndarray, O_inf: float = 0.0):
    """<O>(t) reconstructed from the mode expansion."""
    c, o, ev = res["c"], res["o"], res["evals"]
    return O_inf + (c * o)[None, :] @ np.exp(np.outer(ev, times)) if False else \
        O_inf + np.einsum("i,it->t", c * o, np.exp(np.outer(ev, times)))


def trajectory_direct(Lv, rho0, O, times):
    """<O>(t) by direct exponentiation, as an independent check."""
    d = O.shape[0]
    v0 = vec(rho0)
    return np.array([np.trace(O @ unvec(expm(Lv * t) @ v0, d)).real for t in times])



def block_liouvillian(H, c_ops, L, dm_target=0, sparse=True):
    """Liouvillian restricted to one Delta-m block (invariant under the dephasing)."""
    Lv = build_liouvillian(H, c_ops)
    dm = dm_labels(L)
    idx = np.where(dm == dm_target)[0]
    sub = Lv[np.ix_(idx, idx)]
    return (csr_matrix(sub) if sparse else sub), idx


def slow_modes_sparse(Lblock, k=6, n_discard=0, sigma=-1e-6, tol=1e-9):
    """The k slowest decaying modes of a block by shift-invert Arnoldi; n_discard zero modes
    (L + 1 in the Delta-m = 0 block) are dropped."""
    A = csr_matrix(Lblock)
    want = min(k + n_discard, A.shape[0] - 2)
    w, v = eigs(A, k=want, sigma=sigma, which="LM")
    keep = np.abs(w) >= tol
    w, v = w[keep], v[:, keep]
    order = np.argsort(-w.real)[:k]
    return w[order], v[:, order]



def table_row(L, W, gamma=1.0, J=1.0, seed=0, basis="pauli"):
    """One row of the relevant-gap table, for the (Neel, imbalance) pair."""
    H, fields = build_hamiltonian(L, J=J, W=W, seed=seed, basis=basis)
    Lv = build_liouvillian(H, dephasing_ops(L, gamma=gamma, basis=basis))
    modes = solve_modes(Lv, dm=dm_labels(L))
    res = analyze(modes, neel_state(L), imbalance_op(L, basis=basis))
    return (relevant_gaps(res), len(modes.steady()),
            int(res["visible"].sum()), modes.biorth_cond, fields)


if __name__ == "__main__":
    basis, gamma = "pauli", 1.0
    print(f"basis = {basis!r}, gamma = {gamma}, J = 1, open boundaries")
    print(f"coherence-rate floor (analytic) = "
          f"{-coherence_rate_floor(gamma, basis):+.4f}\n")

    header = (f"{'L':>2} {'W':>5} {'seed':>5} {'steady':>7} {'vis':>5} {'dim':>6} "
              f"{'slowest':>10} {'visible':>10} {'coherence':>10} {'dm=0':>10}")
    print(header)
    print("-" * len(header))
    for (L, W) in [(3, 1.0), (4, 1.0), (4, 4.0), (4, 10.0)]:
        for seed in (0, 1, 2):
            g, ns, nv, cond, _ = table_row(L, W, gamma=gamma, seed=seed, basis=basis)
            print(f"{L:>2} {W:>5.1f} {seed:>5} {ns:>7} {nv:>5} {4**L:>6} "
                  f"{g['slowest_overall']:>10.4f} {g['slowest_visible']:>10.4f} "
                  f"{g['slowest_coherence']:>10.4f} {g['slowest_dm0']:>10.4f}")
        print()
