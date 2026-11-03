"""More samples for alpha = 1.25 at L = 96-256, all gammas and the classical chain, fresh seeds
(finite_gamma_heavy_more.json, raw values)."""

import json
import os
import time

import numpy as np

from finite_gamma_heavy import d_quantum, d_classical

OUT = "finite_gamma_heavy_more.json"
Z = 0.4
LS = [(96, 600), (128, 500), (192, 320), (256, 220)]
GAMMAS = (4.0, 1.0, 0.25)

if __name__ == "__main__":
    db = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for L, n in LS:
        key = f"classical|{Z}|{L}"
        if key not in db:
            rng = np.random.default_rng(5000 + L)
            db[key] = [d_classical(L, Z, rng) for _ in range(3 * n)]
            json.dump(db, open(OUT, "w"))
    for g in GAMMAS:
        for L, n in LS:
            key = f"quantum|{Z}|{g}|{L}"
            if key in db:
                continue
            t = time.time()
            rng = np.random.default_rng(6000 + L + int(10 * g))
            db[key] = [d_quantum(L, Z, g, rng) for _ in range(n)]
            json.dump(db, open(OUT, "w"))
            print(f"gamma={g} L={L} n={n} median={np.median(db[key]):.4e} [{time.time()-t:.0f}s]",
                  flush=True)
    print("done", flush=True)
