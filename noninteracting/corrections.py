#!/usr/bin/env python3
"""Pareto resistance chains up to L = 65536 (corrections.json; Fig. 2a diamonds).
Near alpha = 2 corrections to scaling decay slowly (relative order L^(1 - alpha/2)), so the
exponents are fitted over L >= 4096. Run via run_corrections.py."""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import curve_fit

from blind_scaling import slow_mode, cos_basis

LS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
NS = {64: 4000, 128: 4000, 256: 3000, 512: 2500, 1024: 2000, 2048: 1500,
      4096: 1000, 8192: 600, 16384: 400, 32768: 250, 65536: 150}
ALPHAS = [4.0, 3.0, 2.5, 2.0, 1.667, 1.429, 1.25, 1.111]
OUT = "corrections.json"


def point(args):
    alpha, L, n, seed = args
    rng = np.random.default_rng(seed)
    ref = cos_basis(L)[0]
    acc = np.empty(n)
    for i in range(n):
        rho = rng.uniform(1e-15, 1.0, size=L - 1) ** (-1.0 / alpha)
        _, phi = slow_mode(1.0 / rho)
        phi /= np.linalg.norm(phi)
        if phi @ ref < 0:
            phi = -phi
        acc[i] = np.linalg.norm(phi - ref)
    return alpha, L, float(acc.mean()), float(acc.std() / np.sqrt(n)), n


def collect(jobs_n=None):
    db = json.load(open(OUT)) if os.path.exists(OUT) else {}
    jobs = [(a, L, NS[L], 4321 + L + int(1000 * a))
            for a in ALPHAS for L in LS
            if f"{a:g}|{L}" not in db]
    if jobs:
        print(f"{len(jobs)} points to compute")
        t0 = time.time()
        with Pool(jobs_n or max(1, (os.cpu_count() or 2) - 1)) as pool:
            for i, (a, L, m, e, n) in enumerate(pool.imap_unordered(point, jobs), 1):
                db[f"{a:g}|{L}"] = {"d": m, "e": e, "n": n}
                json.dump(db, open(OUT, "w"))
                if i % 10 == 0 or i == len(jobs):
                    print(f"  [{i}/{len(jobs)}] {time.time()-t0:.0f}s", flush=True)
    return db


def analyse(db):
    print(f"\n{'alpha':>7} {'naive fit':>10} {'theta_inf':>18} {'omega':>8} "
          f"{'asymptotic':>11} {'verdict':>10}")
    rows = []
    for a in ALPHAS:
        L = np.array([l for l in LS if f"{a:g}|{l}" in db], float)
        d = np.array([db[f"{a:g}|{int(l)}"]["d"] for l in L])
        naive = np.polyfit(np.log(L), np.log(d), 1)[0]

        # local slopes at the geometric midpoint of each pair
        Lm = np.sqrt(L[:-1] * L[1:])
        th = np.log(d[1:] / d[:-1]) / np.log(L[1:] / L[:-1])

        def model(x, t_inf, c, om):
            return t_inf + c * x ** (-om)

        try:
            p, cov = curve_fit(model, Lm, th, p0=[-0.4, 0.5, 0.3],
                               bounds=([-1.5, -50, 0.02], [0.0, 50, 3.0]),
                               maxfev=40000)
            err = float(np.sqrt(np.diag(cov))[0])
        except Exception:
            p, err = [np.nan] * 3, np.nan

        asym = -0.5 if a > 2 else 1.0 / a - 1.0
        agree = "yes" if abs(p[0] - asym) < 3 * max(err, 0.02) else "no"
        rows.append((a, naive, p[0], err, p[2], asym, agree))
        print(f"{a:>7.3f} {naive:>+10.3f} {p[0]:>+11.3f}+/-{err:.3f} "
              f"{p[2]:>8.2f} {asym:>+11.3f} {agree:>10}")

    print("\nlocal slopes (L pairs, small to large):")
    for a in ALPHAS:
        L = np.array([l for l in LS if f"{a:g}|{l}" in db], float)
        d = np.array([db[f"{a:g}|{int(l)}"]["d"] for l in L])
        th = np.log(d[1:] / d[:-1]) / np.log(L[1:] / L[:-1])
        print(f"  a={a:<6.3f} " + " ".join(f"{x:+.3f}" for x in th))

    print("\ndirect fit  d = A L^(-theta) (1 + B L^(-omega)):")
    print(f"{'alpha':>7} {'theta':>16} {'asymptotic':>11}")
    for a in ALPHAS:
        L = np.array([l for l in LS if f"{a:g}|{l}" in db], float)
        d = np.array([db[f"{a:g}|{int(l)}"]["d"] for l in L])
        e = np.array([db[f"{a:g}|{int(l)}"]["e"] for l in L])

        def m2(x, A, t, B, om):
            return np.log(A) - t * np.log(x) + np.log1p(B * x ** (-om))

        try:
            p, cov = curve_fit(m2, L, np.log(d), sigma=e / d,
                               p0=[1.0, 0.45, 1.0, 0.3],
                               bounds=([1e-6, 0.0, -50, 0.02], [1e6, 1.5, 50, 3.0]),
                               maxfev=60000)
            s = float(np.sqrt(np.diag(cov))[1])
            asym = 0.5 if a > 2 else 1.0 - 1.0 / a
            print(f"{a:>7.3f} {-p[1]:>+11.3f}+/-{s:.3f} {-asym:>+11.3f}")
        except Exception as ex:
            print(f"{a:>7.3f}   fit failed ({type(ex).__name__})")


if __name__ == "__main__":
    analyse(collect())
