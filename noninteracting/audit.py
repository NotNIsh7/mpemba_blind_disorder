"""Independent checks of the numerical pipeline, each by a route different from the production code.

A  sparse left/right modes against dense diagonalization
B  left and right slow modes coincide at strong dephasing
C  d^2 = 2 eta / (1 + q)
D  slow-mode amplitude from time evolution alone
E  exponent robustness to W, gamma and the blind state
F  Neel imbalance and the slow mode"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csr_matrix

from density_sector import (single_particle_h, density_generator, gamma_to_Gamma,
                            slow_modes, density_profile, hydrodynamic_reference,
                            draw_fields)

PASS, FAIL = [], []


def chk(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def unit(v):
    return v / np.linalg.norm(v)


def dense_modes(G):
    w, vr = np.linalg.eig(G)
    vl = np.linalg.inv(vr).conj().T
    o = np.argsort(-w.real)
    return w[o], vr[:, o], vl[:, o]


def blind(L, k=2):
    x = (np.arange(L) + 0.5) / L
    return unit(np.cos(k * np.pi * x))


def check_A():
    print("\nA. sparse left modes vs dense eig + inverse")
    for L in (16, 24):
        rng = np.random.default_rng(3)
        f = draw_fields(L, W=2.0, dist="uniform", rng=rng)
        h, _ = single_particle_h(L, fields=f)
        Gd = density_generator(h, gamma_to_Gamma(1.0), sparse=False)
        w, vr, vl = dense_modes(Gd)
        s = np.where(np.abs(w) > 1e-9)[0][0]
        ld_dense = np.real(np.diag(vl[:, s].reshape((L, L), order="F")))
        rd_dense = np.real(np.diag(vr[:, s].reshape((L, L), order="F")))

        ws, vrs, vls = slow_modes(csr_matrix(Gd), k=1, want_left=True)
        ld_sp = np.real(np.diag(vls[:, 0].reshape((L, L), order="F")))
        rd_sp = np.real(np.diag(vrs[:, 0].reshape((L, L), order="F")))

        chk(f"L={L} eigenvalue agrees", abs(ws[0] - w[s]) < 1e-9,
            f"{ws[0].real:.9f} vs {w[s].real:.9f}")
        b, ref = blind(L), hydrodynamic_reference(L)
        r_dense = abs(ld_dense @ b) / abs(ld_dense @ ref)
        r_sp = abs(ld_sp @ b) / abs(ld_sp @ ref)
        chk(f"L={L} suppression ratio agrees",
            abs(r_dense - r_sp) / r_dense < 1e-6,
            f"dense {r_dense:.6e} vs sparse {r_sp:.6e}")
        chk(f"L={L} right profile agrees",
            1 - abs(unit(rd_dense) @ unit(rd_sp)) < 1e-9)


def check_B():
    print("\nB. strong dephasing: left and right slow modes must coincide")
    L = 24
    for g in (1.0, 8.0, 64.0):
        rng = np.random.default_rng(3)
        f = draw_fields(L, W=2.0, dist="uniform", rng=rng)
        h, _ = single_particle_h(L, fields=f)
        G = density_generator(h, gamma_to_Gamma(g))
        w, vr, vl = slow_modes(G, k=1, want_left=True)
        rd = unit(np.real(np.diag(vr[:, 0].reshape((L, L), order="F"))))
        ld = unit(np.real(np.diag(vl[:, 0].reshape((L, L), order="F"))))
        mis = 1 - abs(rd @ ld)
        print(f"     gamma={g:>6.1f}   1-|<l,r>| = {mis:.3e}")
    chk("left/right converge as gamma grows", mis < 1e-3, f"final {mis:.2e}")


def check_C():
    print("\nC. algebraic identity d^2 = 2*eta/(1+q)")
    worst = 0.0
    for L in (16, 32, 64):
        rng = np.random.default_rng(9)
        for _ in range(5):
            f = draw_fields(L, W=2.0, dist="uniform", rng=rng)
            h, _ = single_particle_h(L, fields=f)
            G = density_generator(h, gamma_to_Gamma(1.0))
            w, vr, _ = slow_modes(G, k=1, want_left=False)
            ref = hydrodynamic_reference(L)
            phi = density_profile(vr[:, 0], L, sign_ref=ref)
            q = phi @ ref
            eta, d = 1 - q ** 2, np.linalg.norm(phi - ref)
            worst = max(worst, abs(d ** 2 - 2 * eta / (1 + q)))
    chk("identity holds exactly", worst < 1e-10, f"max dev {worst:.2e}")
    print("     => the deficit exponent alpha implies d ~ L^(-alpha/2) exactly;")
    print("        the factor-of-two exponent transfer is algebra, not an estimate.")


def check_D():
    print("\nD. c_1 from time evolution alone (no eigenvectors)")
    print("     late-time:  ||dM(t)|| * exp(-lam1*t) -> |c_1| * ||r_1||")
    for L in (16, 24):
        rng = np.random.default_rng(3)
        f = draw_fields(L, W=2.0, dist="uniform", rng=rng)
        h, _ = single_particle_h(L, fields=f)
        G = density_generator(h, gamma_to_Gamma(1.0))
        w, vr, vl = slow_modes(G, k=1, want_left=True)
        lam, r1 = w[0].real, vr[:, 0]
        nr1 = np.linalg.norm(r1)

        for label, prof in (("generic", hydrodynamic_reference(L)),
                            ("blind", blind(L))):
            v0 = np.diag(prof).astype(complex).flatten(order="F")
            c_eig = abs(vl[:, 0].conj() @ v0)
            T = 6.0 / abs(lam)
            vt = expm_multiply(csr_matrix(G) * T, v0)
            c_dyn = np.linalg.norm(vt) * np.exp(-lam * T) / nr1
            rel = abs(c_dyn - c_eig) / max(c_eig, 1e-300)
            chk(f"L={L} {label:>7}: eigen vs dynamical c_1", rel < 5e-2,
                f"{c_eig:.4e} vs {c_dyn:.4e}  ({rel*100:.2f}%)")


def check_E():
    print("\nE. robustness of the suppression exponent")
    Ls = [16, 24, 32, 48, 64]

    def ratio_scan(W, g, kmode, nsamp=30):
        out = []
        for L in Ls:
            rng = np.random.default_rng(777 + L)
            rs = []
            for _ in range(nsamp):
                f = draw_fields(L, W=W, dist="uniform", rng=rng)
                h, _ = single_particle_h(L, fields=f)
                G = density_generator(h, gamma_to_Gamma(g))
                w, vr, vl = slow_modes(G, k=1, want_left=True)
                ld = np.real(np.diag(vl[:, 0].reshape((L, L), order="F")))
                ref = hydrodynamic_reference(L)
                if ld @ ref < 0:
                    ld = -ld
                if kmode == "rand":
                    rr = rng.normal(size=L)
                    b = unit(rr - (rr @ ref) * ref)
                else:
                    b = blind(L, kmode)
                rs.append(abs(ld @ b) / abs(ld @ ref))
            out.append(np.mean(rs))
        return np.polyfit(np.log(Ls), np.log(out), 1)[0], out

    cases = [("W=2, g=1, cos2pix", 2.0, 1.0, 2),
             ("W=1, g=1, cos2pix", 1.0, 1.0, 2),
             ("W=4, g=1, cos2pix", 4.0, 1.0, 2),
             ("W=2, g=4, cos2pix", 2.0, 4.0, 2),
             ("W=2, g=1, cos3pix", 2.0, 1.0, 3),
             ("W=2, g=1, random  ", 2.0, 1.0, "rand")]
    exps = {}
    for name, W, g, k in cases:
        a, _ = ratio_scan(W, g, k)
        exps[name.strip()] = a
        print(f"     {name}:  ratio ~ L^({a:+.3f})")

    smooth = [v for kk, v in exps.items() if "cos" in kk]
    chk("smooth blind states near -1/2", all(abs(a + 0.5) < 0.15 for a in smooth),
        f"range [{min(smooth):+.3f}, {max(smooth):+.3f}]")
    chk("random blind state steeper than smooth ones",
        exps["W=2, g=1, random"] < max(smooth) - 0.1,
        f"random {exps['W=2, g=1, random']:+.3f} vs smooth max {max(smooth):+.3f}")


def check_F():
    print("\nF. imbalance: when does the slow mode take over, and at what size")
    Lw, fr = [], []
    for L in (16, 24, 32):
        rng = np.random.default_rng(5)
        f = draw_fields(L, W=2.0, dist="uniform", rng=rng)
        h, _ = single_particle_h(L, fields=f)
        Gd = density_generator(h, gamma_to_Gamma(1.0), sparse=False)
        w, vr, vl = dense_modes(Gd)
        dec = np.where(np.abs(w) > 1e-9)[0]
        s = dec[0]

        dn = np.zeros(L); dn[0::2], dn[1::2] = 0.5, -0.5
        v0 = np.diag(dn).astype(complex).flatten(order="F")
        O = np.zeros((L, L), dtype=complex)
        for i in range(L):
            O[i, i] = 2.0 * ((-1) ** i) / L
        Ov = O.flatten(order="F")

        c = vl.conj().T @ v0
        o = (vr.conj().T @ Ov).conj()
        wt = (c * o)[dec]
        lam = w[dec].real
        I0 = abs(wt.sum())
        w1 = abs(wt[0])

        # time at which the slow mode is the largest surviving contribution
        ts = np.logspace(-1, np.log10(30 / abs(lam[0])), 400)
        cross = None
        for t in ts:
            contrib = np.abs(wt) * np.exp(lam * t)
            if np.argmax(contrib) == 0:
                cross = t
                break
        Ic = abs(np.sum(wt * np.exp(lam * cross))) if cross else np.nan
        Lw.append(L); fr.append(w1 / I0)
        print(f"     L={L:>3}  lam_1={lam[0]:+.3e}  w_1/I(0)={w1/I0:.2e}  "
              f"takeover t={cross:.1f}  |I| there={Ic:.2e}")
    sl = np.polyfit(np.log(Lw), np.log(fr), 1)[0]
    chk("slow-mode share of imbalance falls with L", sl < -1.5,
        f"share ~ L^({sl:+.2f}), |I| at takeover (L={L}) = {Ic:.2e}")


if __name__ == "__main__":
    for fn in (check_A, check_B, check_C, check_D, check_E, check_F):
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  ")
