"""Shape of the slow-mode residual above and below alpha = 2: mode coefficients c_k and the
largest density step across a single bond."""

import numpy as np
from blind_scaling import slow_mode

KMAX = 10


def sample(L, sampler, n, rng):
    xs = (np.arange(L) + 0.5) / L
    C = np.array([np.cos(k * np.pi * xs) for k in range(1, KMAX + 1)])
    norms = np.sum(C * C, axis=1)
    ck = np.empty((n, KMAX - 1))
    share = np.empty(n)
    at_max = np.empty(n, dtype=bool)
    xb = np.arange(1, L) / L
    for j in range(n):
        rho = sampler(rng, L - 1)
        _, phi = slow_mode(1.0 / rho)
        phi = phi / ((C[0] @ phi) / norms[0])        # amplitude 1 on cos(pi x)
        res = phi - C[0]
        ck[j] = (C[1:] @ res) / norms[1:]
        jumps = np.abs(np.diff(res))
        share[j] = jumps.max() / jumps.sum()
        # the step across a bond is resistance x current, and the current
        # vanishes at the ends like sin(pi x)
        at_max[j] = abs(int(np.argmax(jumps)) - int(np.argmax(rho * np.sin(np.pi * xb)))) <= 1
    return ck, share, at_max


if __name__ == "__main__":
    tails = {
        "bounded":    lambda r, m: 1.0 + r.uniform(0, 1, m),
        "alpha=4":    lambda r, m: r.uniform(1e-15, 1, m) ** (-1 / 4.0),
        "alpha=2.5":  lambda r, m: r.uniform(1e-15, 1, m) ** (-1 / 2.5),
        "alpha=1.5":  lambda r, m: r.uniform(1e-15, 1, m) ** (-1 / 1.5),
        "alpha=1.25": lambda r, m: r.uniform(1e-15, 1, m) ** (-1 / 1.25),
    }
    ks = np.arange(2, KMAX + 1)
    env = (ks / (ks ** 2 - 1.0)) / (2 / 3)

    for L, n in ((1024, 2000), (4096, 800)):
        print(f"\n===== L = {L}, {n} samples =====")
        print("P3a  median|c_k| / median|c_2|   (envelope k/(k^2-1) in last row)")
        print("      k:      " + " ".join(f"{k:>6d}" for k in ks))
        rows = {}
        for name, smp in tails.items():
            ck, share, at_max = sample(L, smp, n, np.random.default_rng(3 + L))
            med = np.median(np.abs(ck), axis=0)
            rows[name] = (med / med[0], ck, share, at_max)
            print(f"  {name:>10}: " + " ".join(f"{v:6.3f}" for v in med / med[0])
                  + f"   max dev {np.max(np.abs(med/med[0]/env - 1))*100:4.1f}%")
        print(f"  {'envelope':>10}: " + " ".join(f"{v:6.3f}" for v in env))

        print("\nP3b  fraction of samples where |c_2| is the largest |c_k|")
        for name, (_, ck, _, _) in rows.items():
            print(f"  {name:>10}: {np.mean(np.argmax(np.abs(ck), axis=1) == 0):.3f}")

        print("\nP3c  real space: largest single jump's share of the residual,"
              " and whether it sits at the largest resistance x current")
        for name, (_, _, share, at_max) in rows.items():
            print(f"  {name:>10}: median share {np.median(share):.4f}"
                  f"   at largest resistance x current: {np.mean(at_max):.3f}")
