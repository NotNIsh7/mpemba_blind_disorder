#!/usr/bin/env python3
"""Classical resistor chain (strong-dephasing limit): slow mode, distortion d and residual r
for random fields and random hoppings, L = 64-4096.

    python blind_scaling.py verify      # against dense diagonalization and the quantum density sector
    python blind_scaling.py run --jobs 8
    python blind_scaling.py analyse

Results are cached; an interrupted run resumes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.linalg import eigh_tridiagonal

OUT = "blind_scaling.json"
KMAX = 8

# both disorder models return the L-1 bond conductances R_i

def bonds_from_z(L, z, rng):
    """Bond disorder: J_i = u^z on (0, 1], R_i proportional to J_i^2."""
    u = rng.uniform(1e-15, 1.0, size=L - 1)
    return 2.0 * (u ** z) ** 2


def bonds_from_W(L, W, rng, J=1.0, gamma=1.0):
    """Field disorder: R_i = 8 J^2 gamma / (4 gamma^2 + (h_i - h_{i+1})^2)."""
    h = rng.uniform(-W, W, size=L)
    return 8.0 * J ** 2 * gamma / (4.0 * gamma ** 2 + np.diff(h) ** 2)


def disorder_parameter(R):
    """<R><1/R>. Equals 1 with no disorder, diverges when <1/R> does."""
    return float(np.mean(R) * np.mean(1.0 / R))



def slow_mode(R):
    """Slowest nonuniform mode of the no-flux resistor chain with conductances R."""
    L = len(R) + 1
    diag = np.zeros(L)
    diag[:-1] -= R
    diag[1:] -= R
    w, v = eigh_tridiagonal(diag, R.copy(), select="i",
                            select_range=(L - 2, L - 2))
    return float(w[0]), v[:, 0]


def cos_basis(L, kmax=KMAX):
    x = (np.arange(L) + 0.5) / L
    B = np.stack([np.cos(k * np.pi * x) for k in range(1, kmax + 1)])
    return B / np.linalg.norm(B, axis=1, keepdims=True)


def measure(R, B, rng):
    """All scaling observables for one disorder realization."""
    lam, phi = slow_mode(R)
    L = len(phi)
    ref = B[0]
    phi = phi / np.linalg.norm(phi)
    if phi @ ref < 0:
        phi = -phi

    q = float(phi @ ref)
    res = phi - ref
    # the chain is symmetric, so the left mode equals the right one
    den = abs(phi @ ref)
    rv = rng.normal(size=L)
    rand = rv - (rv @ ref) * ref
    rand /= np.linalg.norm(rand)

    return {
        "lam": lam,
        "eta": 1.0 - q ** 2,
        "d": float(np.linalg.norm(res)),
        "r2": float(abs(phi @ B[1]) / den),
        "r3": float(abs(phi @ B[2]) / den),
        "r4": float(abs(phi @ B[3]) / den),
        "rrand": float(abs(phi @ rand) / den),
        "wk": [float((res @ B[k]) ** 2) for k in range(1, KMAX)],
        "dpar": disorder_parameter(R),
    }



def run_point(args):
    kind, par, L, n, seed = args
    rng = np.random.default_rng(seed)
    B = cos_basis(L)
    keys = ("lam", "eta", "d", "r2", "r3", "r4", "rrand", "dpar")
    acc = {k: np.empty(n) for k in keys}
    wk = np.zeros(KMAX - 1)

    for i in range(n):
        R = bonds_from_z(L, par, rng) if kind == "bond" else bonds_from_W(L, par, rng)
        m = measure(R, B, rng)
        for k in keys:
            acc[k][i] = m[k]
        wk += np.asarray(m["wk"])

    out = {"n": n, "L": L, "kind": kind, "par": par,
           "wk": (wk / wk.sum()).tolist()}
    rb = np.random.default_rng(seed + 1)
    for k in keys:
        a = acc[k]
        idx = rb.integers(0, n, size=(1000, n))
        out[k] = float(a.mean())
        out[k + "_e"] = float(a[idx].mean(axis=1).std())
    return out



def grid(quick=False):
    if quick:
        Ls = [(64, 400), (128, 300), (256, 200), (512, 150)]
        zs = [0.20, 0.35, 0.45]
        Ws = [2.0]
    else:
        Ls = [(64, 4000), (128, 4000), (256, 3000), (512, 2000),
              (1024, 1200), (2048, 600), (4096, 300)]
        zs = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        Ws = [1.0, 2.0, 4.0, 8.0, 16.0]
    jobs = []
    for L, n in Ls:
        for z in zs:
            jobs.append(("bond", z, L, n, 10_000 + L * 31 + int(1000 * z)))
        for W in Ws:
            jobs.append(("field", W, L, n, 90_000 + L * 31 + int(1000 * W)))
    return jobs


def key_of(kind, par, L):
    return f"{kind}:{par:g}:{L}"



def verify():
    """Check the tridiagonal solver against dense diagonalization and the quantum density sector."""
    from scipy.sparse import csr_matrix, diags, eye as sp_eye, kron as sp_kron
    from scipy.sparse.linalg import eigs
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    rng = np.random.default_rng(0)
    for L in (24, 40):
        R = bonds_from_z(L, 0.3, rng)
        lam, phi = slow_mode(R)
        diag = np.zeros(L); diag[:-1] -= R; diag[1:] -= R
        K = np.diag(diag) + np.diag(R, 1) + np.diag(R, -1)
        ev, evec = np.linalg.eigh(K)
        chk(f"L={L} tridiagonal vs dense eigenvalue",
            abs(lam - ev[-2]) < 1e-10, f"{lam:.12e} vs {ev[-2]:.12e}")
        chk(f"L={L} tridiagonal vs dense eigenvector",
            1 - abs(evec[:, -2] @ (phi / np.linalg.norm(phi))) < 1e-9)
        chk(f"L={L} zero mode is uniform",
            abs(ev[-1]) < 1e-10, f"lam_0 = {ev[-1]:.2e}")

    for L, g in ((16, 64.0), (20, 128.0)):
        J = rng.uniform(0.4, 1.0, size=L - 1)
        h = np.zeros((L, L), dtype=complex)
        for i in range(L - 1):
            h[i, i + 1] = h[i + 1, i] = 2.0 * J[i]
        G = -1j * (sp_kron(sp_eye(L, format="csr", dtype=complex), csr_matrix(h))
                   - sp_kron(csr_matrix(h.T), sp_eye(L, format="csr", dtype=complex)))
        Gam = 4.0 * g
        G = csr_matrix(G - Gam * diags((1.0 - np.eye(L)).flatten(order="F"),
                                       format="csr", dtype=complex))
        w, _ = eigs(G, k=4, sigma=-1e-9, which="LM")
        wq = np.sort(w[np.abs(w) > 1e-9].real)[::-1][0]
        lam_c, _ = slow_mode(2.0 * J ** 2 / g)
        chk(f"L={L} quantum (gamma={g:g}) vs classical chain",
            abs(wq / lam_c - 1) < 2e-3, f"ratio {wq/lam_c:.6f}")

    print("\n" + ("verification passed" if ok else "VERIFICATION FAILED"))
    return ok



def fit(Ls, m, e, nboot=4000, seed=0):
    rng = np.random.default_rng(seed)
    Ls, m, e = np.asarray(Ls, float), np.asarray(m), np.asarray(e)
    a = np.polyfit(np.log(Ls), np.log(m), 1)[0]
    dr = [np.polyfit(np.log(Ls),
                     np.log(np.maximum(m + rng.normal(0, e), 1e-300)), 1)[0]
          for _ in range(nboot)]
    return a, float(np.std(dr))


def analyse():
    db = json.load(open(OUT))
    rows = list(db.values())
    Ls = sorted({r["L"] for r in rows})
    kk = np.arange(2, KMAX + 1)
    pred_w = (kk / (kk ** 2 - 1.0)) ** 2
    pred_w /= pred_w.sum()

    for kind in ("bond", "field"):
        pars = sorted({r["par"] for r in rows if r["kind"] == kind})
        if not pars:
            continue
        print(f"\n=== {kind} disorder ===")
        lab = "z" if kind == "bond" else "W"
        print(f"{lab:>6} {'<R><1/R>':>9} {'lam exp':>9} {'d exp':>15} "
              f"{'cos2 exp':>15} {'2z-1':>7} {'wk dev':>8}")
        for p in pars:
            sel = [db[key_of(kind, p, L)] for L in Ls if key_of(kind, p, L) in db]
            if len(sel) < 3:
                continue
            LL = [r["L"] for r in sel]
            la, _ = fit(LL, [abs(r["lam"]) for r in sel],
                        [r["lam_e"] for r in sel])
            da, de = fit(LL, [r["d"] for r in sel], [r["d_e"] for r in sel])
            ra, re = fit(LL, [r["r2"] for r in sel], [r["r2_e"] for r in sel])
            dev = np.abs(np.asarray(sel[-1]["wk"]) - pred_w).sum()
            t = f"{2*p-1:+.2f}" if kind == "bond" else "-"
            print(f"{p:>6g} {sel[-1]['dpar']:>9.2f} {la:>+9.3f} "
                  f"{da:>+8.3f}+/-{de:.3f} {ra:>+8.3f}+/-{re:.3f} {t:>7} {dev:>8.3f}")

        print("\n  local slopes of d between successive L "
              "(reading A: settles near 2z-1; reading B: drifts to -0.50)")
        for p in pars:
            sel = [db[key_of(kind, p, L)] for L in Ls if key_of(kind, p, L) in db]
            if len(sel) < 3:
                continue
            LL = [r["L"] for r in sel]
            dd = [r["d"] for r in sel]
            s = [np.log(dd[i+1]/dd[i]) / np.log(LL[i+1]/LL[i])
                 for i in range(len(dd) - 1)]
            print(f"   {lab}={p:<5g} " + " ".join(f"{x:+.2f}" for x in s))

    print("\n  matching check: bond and field points with a similar <R><1/R>")
    print("  should share an exponent if that parameter controls the physics.")
    big = [r for r in rows if r["L"] == Ls[-1]]
    for r in sorted(big, key=lambda r: r["dpar"]):
        print(f"   {r['kind']:>5} {r['par']:>6g}  <R><1/R>={r['dpar']:>7.2f}  "
              f"d={r['d']:.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "run", "analyse"])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    if a.mode == "verify":
        sys.exit(0 if verify() else 1)
    if a.mode == "analyse":
        analyse()
        return

    db = json.load(open(OUT)) if os.path.exists(OUT) else {}
    # recompute cached points that have fewer samples than requested
    jobs = [j for j in grid(a.quick)
            if db.get(key_of(j[0], j[1], j[2]), {}).get("n", 0) < j[3]]
    print(f"{len(jobs)} points to compute, {len(db)} already done, "
          f"{a.jobs} processes")
    if not jobs:
        print("nothing to do")
        return
    jobs.sort(key=lambda j: j[2] * j[3])

    t0 = time.time()
    with Pool(a.jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(run_point, jobs), 1):
            db[key_of(res["kind"], res["par"], res["L"])] = res
            json.dump(db, open(OUT, "w"))
            el = time.time() - t0
            print(f"[{i}/{len(jobs)}] {res['kind']} par={res['par']:g} "
                  f"L={res['L']} n={res['n']}  d={res['d']:.4e}  "
                  f"({el/60:.1f} min elapsed, ~{el/i*(len(jobs)-i)/60:.0f} left)",
                  flush=True)
    print("\ndone. now run:  python blind_scaling.py analyse")


if __name__ == "__main__":
    main()
