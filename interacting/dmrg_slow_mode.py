"""Slowest mode of the interacting strong-dephasing chain by DMRG (L = 24, 32).

A particle hops across bond (i, i+1) at rate 32 J^2 gamma / (16 gamma^2 + dE^2), with
dE = 2 (h_{i+1} - h_i) + 2 V (s_{i+2} - s_{i-1}) and s = 2n - 1 (0 beyond the ends).
H = -generator is real symmetric; its first excited state is the slow mode, and
r = sum_i cos(2 pi x_i) m_i / sum_i cos(pi x_i) m_i with m_i = <psi_1| n_i |psi_0>."""
import sys
import time
from math import comb
import numpy as np
from tenpy.networks.site import BosonSite
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

J, gamma = 1.0, 4.0
SHIFT = 1.0   # TeNPy's orthogonal_to gives excluded states energy 0; shift the spectrum below it


def rate(dE):
    return 32 * J**2 * gamma / (16 * gamma**2 + dE**2)


class Generator(CouplingMPOModel):
    def init_sites(self, model_params):
        site = BosonSite(Nmax=1, conserve="N")
        site.add_op("P0", site.Id - site.N)
        return site

    def init_terms(self, model_params):
        h = np.asarray(model_params["h"])
        V = model_params["V"]
        L = len(h)
        Jb = model_params.get("Jb", np.ones(L - 1))
        for i in range(L - 1):
            left = [(-1, "P0"), (1, "N")] if i > 0 else [(0, None)]
            right = [(-1, "P0"), (1, "N")] if i + 2 < L else [(0, None)]
            for sa, opa in left:
                for sd, opd in right:
                    w = Jb[i] ** 2 * rate(2 * (h[i + 1] - h[i]) + 2 * V * (sd - sa))
                    for mid, sign in ((["N", "P0"], 1), (["P0", "N"], 1),
                                      (["Bd", "B"], -1), (["B", "Bd"], -1)):
                        sites, ops = [i, i + 1], list(mid)
                        if opa:
                            sites, ops = [i - 1] + sites, [opa] + ops
                        if opd:
                            sites, ops = sites + [i + 2], ops + [opd]
                        self.add_multi_coupling_term(sign * w, sites, ops, ["Id"] * (len(sites) - 1))
        for i in range(L):
            self.add_onsite_term(-SHIFT / L, i, "Id")


def naive_residual(h, V, Jb=None):
    """Residual of the resistor chain with configuration-averaged rates (exact at V = 0)."""
    L, N = len(h), len(h) // 2
    Jb = np.ones(L - 1) if Jb is None else Jb
    G = np.zeros(L - 1)
    for i in range(L - 1):
        nb = [s for s in (i - 1, i + 2) if 0 <= s < L]
        for occ in np.ndindex(*(2,) * len(nb)):
            s = dict(zip(nb, (2 * o - 1 for o in occ)))
            dE = 2 * (h[i + 1] - h[i]) + 2 * V * (s.get(i + 2, 0) - s.get(i - 1, 0))
            # bond (i, i+1) holds one particle: 2 arrangements; neighbours fixed by occ
            p = 2 * comb(L - 2 - len(nb), N - 1 - sum(occ)) / comb(L, N)
            G[i] += p * Jb[i] ** 2 * rate(dE)
    K = np.diag(np.r_[G, 0] + np.r_[0, G]) - np.diag(G, 1) - np.diag(G, -1)
    return residual(np.linalg.eigh(K)[1][:, 1])


def slow_mode(h, V, chi_max=64, Jb=None):
    L = len(h)
    Jb = np.ones(L - 1) if Jb is None else Jb
    model = Generator({"L": L, "h": h, "V": V, "Jb": Jb, "bc_MPS": "finite"})
    init = [1, 0] * (L // 2)
    params = {"mixer": True, "max_E_err": 1e-11, "max_sweeps": 40, "min_sweeps": 6,
              "chi_list": {0: 16, 3: 32, 6: chi_max},
              "trunc_params": {"chi_max": chi_max, "svd_min": 1e-12},
              "lanczos_params": {"N_max": 60, "E_tol": 1e-15}}
    psi0 = MPS.from_product_state(model.lat.mps_sites(), init, bc="finite")
    E0 = dmrg.run(psi0, model, params)["E"]
    psi1 = MPS.from_product_state(model.lat.mps_sites(), init[::-1], bc="finite")
    E1, _ = dmrg.TwoSiteDMRGEngine(psi1, model, params, orthogonal_to=[psi0]).run()
    m = np.empty(L)
    for i in range(L):
        phi = psi0.copy()
        phi.apply_local_op(i, "N")
        m[i] = psi1.overlap(phi).real
    return E0 + SHIFT, E1 + SHIFT, m, max(psi1.chi)


def residual(m):
    L = len(m)
    x = (np.arange(L) + 0.5) / L
    return np.dot(np.cos(2 * np.pi * x), m) / np.dot(np.cos(np.pi * x), m)


if __name__ == "__main__":
    h0 = float(sys.argv[1])
    V = float(sys.argv[2])
    L = int(sys.argv[3])
    nsamp = int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    chi_max = int(sys.argv[6]) if len(sys.argv) > 6 else 64
    z = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0   # J_i = v^z, v ~ U(0, 1]; 0 = uniform
    rng = np.random.default_rng(seed)
    print(f"# h0={h0} V={V} L={L} gamma={gamma} seed={seed} z={z}")
    print(f"# chi_max={chi_max}")
    print("# sample  E0  |lambda1|*L^2  r  r_naive  r_V0(exact)  max_chi  seconds")
    for k in range(nsamp):
        h = rng.uniform(-h0, h0, L)
        # draw hoppings only for z > 0, so z = 0 keeps the stored samples
        Jb = (1 - rng.uniform(0, 1, L - 1)) ** z if z > 0 else np.ones(L - 1)
        t = time.time()
        E0, E1, m, chi = slow_mode(h, V, chi_max, Jb)
        print(f"{k:4d} {E0:.1e} {E1 * L**2:.5f} {residual(m):+.6f} {naive_residual(h, V, Jb):+.6f} "
              f"{naive_residual(h, 0.0, Jb):+.6f} {chi} {time.time() - t:.0f}", flush=True)
