"""Random site positions (exponential spacings, mean l) with J = exp(-x/xi), alpha = xi/(2l),
plus random fields h in [-W, W]; exponents, and d against omega sample by sample
(random_positions.json)."""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

from blind_scaling import slow_mode

OUT = "random_positions.json"
GAMMA, W = 1.0, 2.0
LS = [256, 512, 1024, 2048, 4096, 8192]
NS = {256: 800, 512: 800, 1024: 700, 2048: 500, 4096: 350, 8192: 220}


def rates_positions(L, xi, rng):
    x = rng.exponential(1.0, size=L - 1)            # spacings, mean l = 1
    J = np.exp(-x / xi)
    h = rng.uniform(-W, W, size=L)
    return 8 * J ** 2 * GAMMA / (4 * GAMMA ** 2 + np.diff(h) ** 2)


def rates_bonds(L, alpha, rng):
    u = rng.uniform(1e-300, 1.0, size=L - 1)
    return 2.0 * u ** (1.0 / alpha)                 # 1/R ~ u^(-1/alpha)


def measure(R):
    L = len(R) + 1
    rho = 1.0 / R
    u = np.concatenate([[0.0], np.cumsum(rho)]) / rho.sum()
    xb = np.arange(L) / (L - 1)
    omega = float(np.max(np.abs(u - xb)))
    _, phi = slow_mode(R)
    xs = (np.arange(L) + 0.5) / L
    ref = np.cos(np.pi * xs); ref /= np.linalg.norm(ref)
    phi /= np.linalg.norm(phi)
    if phi @ ref < 0:
        phi = -phi
    return float(np.linalg.norm(phi - ref)), omega, float(rho.max() / rho.sum())


def hill(rho, frac=0.01):
    s = np.sort(rho)[::-1]
    k = max(20, int(frac * len(s)))
    return 1.0 / np.mean(np.log(s[:k] / s[k]))


if __name__ == "__main__":
    xis = [float(a) for a in sys.argv[1:]] or [6.0, 5.0, 3.0, 2.5]
    db = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for xi in xis:
        alpha = xi / 2.0
        for model in ("positions", "bonds"):
            for L in LS:
                key = f"{model}|{xi}|{L}"
                if key in db:
                    continue
                rng = np.random.default_rng(int(1000 * xi) + L + (7 if model == "bonds" else 0))
                rows, pool = [], []
                for _ in range(NS[L]):
                    R = rates_positions(L, xi, rng) if model == "positions" else rates_bonds(L, alpha, rng)
                    rows.append(measure(R))
                    if len(pool) < 200000:
                        pool.extend((1.0 / R).tolist())
                d, om, sh = map(np.array, zip(*rows))
                db[key] = {"d": d.tolist(), "omega": om.tolist(), "share": sh.tolist(),
                           "hill": hill(np.array(pool))}
                json.dump(db, open(OUT, "w"))
        print(f"xi={xi} done", flush=True)
