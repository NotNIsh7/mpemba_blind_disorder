"""Random-hopping disorder J_i = u^z, u uniform on (0, 1], with zero on-site fields.
The bond resistance ~ 1/J_i^2 then has tail index alpha = 1/(2z)."""

import numpy as np


def draw_bonds(L, z, rng):
    """L - 1 hoppings J_i = u^z."""
    return rng.uniform(1e-12, 1.0, size=L - 1) ** z


def bond_hamiltonian(J):
    """Single-particle Hamiltonian with hoppings J_i and zero fields."""
    L = len(J) + 1
    h = np.zeros((L, L), dtype=complex)
    for i in range(L - 1):
        h[i, i + 1] = h[i + 1, i] = 2.0 * J[i]
    return h
