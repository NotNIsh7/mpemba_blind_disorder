"""Check of the inverse-generator identity <g, K^{-1} f> = -sum_b rho_b F_b G_b and of the
first-order reduction: the error of the reduction scales as omega^2."""

import numpy as np
from blind_scaling import slow_mode

KMAX = 20


def one(L, rho):
    xs = (np.arange(L) + 0.5) / L
    xb = np.arange(1, L) / L
    C = np.array([np.cos(k * np.pi * xs) for k in range(1, KMAX + 1)])
    norms = np.sum(C * C, axis=1)
    _, phi = slow_mode(1.0 / rho)
    phi = phi / ((C[0] @ phi) / norms[0])
    exact = (C[1:] @ phi) / norms[1:]
    rbar = rho.mean()
    drho = rho - rbar
    ks = np.arange(2, KMAX + 1)
    first = np.array([(2 * k / (k * k - 1)) / (L * rbar)
                      * np.sum(drho * np.sin(k * np.pi * xb) * np.sin(np.pi * xb)) for k in ks])
    omega = np.max(np.abs(np.cumsum(drho))) / (L * rbar)
    return omega, np.linalg.norm(first), np.linalg.norm(exact - first)


if __name__ == "__main__":
    tails = {
        "bounded":    lambda r, m: 1.0 + r.uniform(0, 1, m),
        "alpha=4":    lambda r, m: r.uniform(1e-15, 1, m) ** (-1 / 4.0),
        "alpha=1.5":  lambda r, m: r.uniform(1e-15, 1, m) ** (-1 / 1.5),
        "alpha=1.25": lambda r, m: r.uniform(1e-15, 1, m) ** (-1 / 1.25),
    }
    Ls = (256, 1024, 4096)
    res = {}
    for name, smp in tails.items():
        for L in Ls:
            rng = np.random.default_rng(17 + L)
            n = 600 if L < 4096 else 250
            out = np.array([one(L, smp(rng, L - 1)) for _ in range(n)])
            res[(name, L)] = out

    print("L1  median( error / omega^2 )      -- should stay roughly constant")
    print(f"{'':>11}" + "".join(f"{'L='+str(L):>12}" for L in Ls))
    for name in tails:
        print(f"{name:>11}" + "".join(f"{np.median(res[(name,L)][:,2]/res[(name,L)][:,0]**2):>12.2f}" for L in Ls))

    print("\nL2  median( first-order size / omega )   -- should stay O(1)")
    for name in tails:
        print(f"{name:>11}" + "".join(f"{np.median(res[(name,L)][:,1]/res[(name,L)][:,0]):>12.3f}" for L in Ls))

    print("\nL3  median omega, and its exponent in L")
    for name, a in zip(tails, (None, 4.0, 1.5, 1.25)):
        om = [np.median(res[(name, L)][:, 0]) for L in Ls]
        e = np.polyfit(np.log(Ls), np.log(om), 1)[0]
        pred = -0.5 if (a is None or a > 2) else 1 / a - 1
        print(f"{name:>11}" + "".join(f"{o:>12.4f}" for o in om)
              + f"   exponent {e:+.3f}  (predicted {pred:+.3f})")

    print("\nfraction of samples inside the proven regime (omega < 3/(16 pi^2) = 0.019)")
    for name in tails:
        print(f"{name:>11}" + "".join(f"{np.mean(res[(name,L)][:,0] < 3/(16*np.pi**2)):>12.2f}" for L in Ls))
