"""For finite-variance disorder, sqrt(L) omega <rho> / sd(rho) follows the Kolmogorov
distribution; omega = max_x |u(x) - x|, u(x) = fraction of the total resistance left of x."""
import numpy as np
from scipy.stats import kstwobign
rng = np.random.default_rng(3)
for L in (256, 1024, 4096, 16384):
    v = []
    for _ in range(4000):
        rho = 1 + rng.uniform(0, 1, L - 1); rb = rho.mean()
        om = np.max(np.abs(np.cumsum(rho - rb))) / (L * rb)
        v.append(np.sqrt(L) * om * rb / np.std(rho))
    v = np.array(v)
    print(f"L={L:>6}  median {np.median(v):.4f} (Kolmogorov {kstwobign.median():.4f})"
          f"  mean {v.mean():.4f} (Kolmogorov {kstwobign.mean():.4f})")
