"""Recompute the letter's quoted numbers from results/ and the caches in noninteracting/,
and check that the letter quotes them (when paper/main.tex is present).

Usage: python tools/audit_numbers.py      (exit status 1 if any check fails)"""
import os
import re
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")
TEX = os.path.join(ROOT, "paper", "main.tex")
# without paper/main.tex only the values are checked
PAPERS = {"letter": open(TEX).read()} if os.path.exists(TEX) else {}
results = []


def check(name, papers, quoted, ok, detail):
    missing = [p for p in papers if p in PAPERS and quoted not in PAPERS[p]]
    status = "PASS" if ok and not missing else "FAIL"
    note = f" [text not found in: {', '.join(missing)}]" if missing else ""
    results.append(status)
    print(f"{status}  {name}: {detail}{note}")


def rows(name, ncols=None):
    out = []
    for line in open(os.path.join(RES, name)):
        if line.startswith("#") or not line.strip():
            continue
        out.append(line.split())
    return out


def pilot():
    """{(L, V): (rms*sqrtL, corr, slope, clean_r)} from the exact-diagonalization tables."""
    d = {}
    for f in ("interacting_pilot_h4_L8-16.txt", "interacting_pilot_h4_L18.txt", "interacting_pilot_h4_L20.txt"):
        for c in rows(f):
            d[(int(c[0]), float(c[1]))] = (float(c[3]), float(c[5]), float(c[6]), float(c[7]))
    return d


def dmrg(*names):
    return np.vstack([np.loadtxt(os.path.join(RES, n)) for n in names])


def heavy_ed(h):
    """{(L, V): (median ratio, corr)} from heavy_tail_hopping_z0.4_h{h}.txt."""
    return {(int(c[0]), float(c[1])): (float(c[5]), float(c[6])) for c in rows(f"heavy_tail_hopping_z0.4_h{h}.txt")}


def fitted(name, tag=""):
    """The value after 'fitted exponent ...:' or '... =' in a result file."""
    for line in open(os.path.join(RES, name)):
        if "fitted exponent" not in line or tag not in line:
            continue
        m = re.findall(r"[:=]\s*(-?\d+\.\d+)", line.split("fitted exponent")[1])
        if m:
            return float(m[0])
    raise KeyError(name)


def renorm_exponents(name):
    out = {}
    for line in open(os.path.join(RES, name)):
        if line.startswith("# V=") and "fitted exponent" in line:
            out[float(line.split("V=")[1].split(":")[0])] = float(line.split("fitted exponent")[1].split()[0])
    return out


def main():
    P = pilot()

    # interacting chain, bounded fields
    bd24, bd32 = dmrg("dmrg_V4_L24_seed1.txt"), dmrg("dmrg_V4_L32_seed2.txt", "dmrg_V4_L32_seed3.txt")
    bd = np.vstack([bd24, bd32])
    s_all = bd[:, 3] @ bd[:, 4] / (bd[:, 4] @ bd[:, 4])
    check("bounded: enhancement over averaged-rate chain (DMRG, L=24,32)", ["letter"], "slope $1.61$",
          abs(s_all - 1.61) < 0.005, f"pooled slope {s_all:.3f}")
    s24 = bd24[:, 3] @ bd24[:, 4] / (bd24[:, 4] @ bd24[:, 4])
    s32 = bd32[:, 3] @ bd32[:, 4] / (bd32[:, 4] @ bd32[:, 4])
    check("bounded: factor settles at 1.6 for L=24-32", ["letter"], "settles at $1.6$ for $L=24$--$32$",
          abs(s24 - 1.6) < 0.05 and abs(s32 - 1.6) < 0.05, f"slopes {s24:.3f}, {s32:.3f}")
    rms = {V: [P[(L, V)][0] for L in (8, 10, 12, 14, 16, 18, 20)] for V in (0.0, 1.0, 2.0, 4.0)}
    for L, d in ((24, bd24), (32, bd32)):
        rms[4.0].append(np.sqrt(np.mean(d[:, 3] ** 2) * L))
    q = {0.0: 0.12, 1.0: 0.12, 2.0: 0.11, 4.0: 0.08}
    ok = all(min(rms[V]) >= q[V] - 0.015 and max(rms[V]) <= q[V] + 0.015 for V in q)
    check("bounded: rms r sqrt(L) about 0.12 (V=0) to 0.08 (V=4)", ["letter"],
          "from $0.12$ to $0.08$", ok,
          "ranges " + ", ".join(f"V={V:g}: {min(v):.3f}-{max(v):.3f}" for V, v in rms.items()))
    ex1 = np.mean([P[(L, 1.0)][2] - 1 for L in (8, 10, 12, 14, 16, 18, 20)])
    ex2 = np.mean([P[(L, 2.0)][2] - 1 for L in (8, 10, 12, 14, 16, 18, 20)])
    check("bounded: excess slope 0.06 (V=1), 0.25 (V=2), i.e. ~V^2", ["letter"], "$0.06$ at $V=1$ and $0.25$ at $V=2$",
          abs(ex1 - 0.06) < 0.01 and abs(ex2 - 0.25) < 0.03, f"mean over L: {ex1:.3f}, {ex2:.3f}")
    corr = [P[(L, 4.0)][1] for L in (8, 10, 12, 14, 16, 18, 20)] + \
           [np.corrcoef(d[:, 3], d[:, 4])[0, 1] for d in (bd24, bd32)]
    allcorr = [P[k][1] for k in P if k[1] > 0] + corr[-2:]
    check("bounded: correlation 98-100% (SM)", ["letter"], "$98$--$100\\%$ correlated",
          round(min(allcorr), 2) >= 0.98,
          f"V=4 range {min(corr):.3f}-{max(corr):.3f}; all V up to {max(allcorr):.3f}")
    clean = max(abs(P[k][3]) for k in P)
    check("clean chains: |r| < 1e-13", ["letter"], "$|r|<10^{-13}$", clean < 1e-13, f"max {clean:.1e}")
    conv = open(os.path.join(RES, "dmrg_chi_convergence.txt")).read()
    check("DMRG: chi 64 -> 96 changes r in the sixth digit (L=24)", ["letter"], "changes $r$ in the sixth digit",
          "r=+0.040033" in conv and "r=+0.040034" in conv, "0.040033 vs 0.040034")
    check("ED: 184756 configurations at L=20", ["letter"], "$184\\,756$ configurations",
          __import__("math").comb(20, 10) == 184756, "C(20,10)")

    # table S1
    tab_bounded = {0.0: "0.126 & 0.115 & 0.123 & 0.125 & 0.132 & 0.123 & 0.123",
                   4.0: "0.084 & 0.076 & 0.080 & 0.085 & 0.087 & 0.080 & 0.081 & 0.077 & 0.081"}
    vals0 = " & ".join(f"{P[(L, 0.0)][0]:.3f}" for L in (8, 10, 12, 14, 16, 18, 20))
    vals4 = " & ".join([f"{P[(L, 4.0)][0]:.3f}" for L in (8, 10, 12, 14, 16, 18, 20)] +
                       [f"{np.sqrt(np.mean(d[:, 3] ** 2) * L):.3f}" for L, d in ((24, bd24), (32, bd32))])
    check("Table S1: bounded V=0 row", ["letter"], tab_bounded[0.0], vals0 == tab_bounded[0.0], vals0)
    check("Table S1: bounded V=4 row", ["letter"], tab_bounded[4.0], vals4 == tab_bounded[4.0], vals4)
    for h, label, quoted in ((0, "no fields", "0.960 & 0.975 & 0.979 & 0.981 & 0.982 & 0.983 & & 0.983"),
                             (4, "with fields", "0.963 & 0.982 & 0.980 & 0.985 & 0.988 & 0.991 & & 0.998")):
        H = heavy_ed(h)
        d = dmrg(f"heavy_tail_dmrg_h{h}_V4_L24.txt")
        vals = " & ".join(f"{H[(L, 4.0)][0]:.3f}" for L in (8, 10, 12, 14, 16, 18)) + \
            f" & & {np.median(np.abs(d[:, 3] / d[:, 5])):.3f}"
        check(f"Table S1: heavy {label} row", ["letter"], quoted, vals == quoted, vals)

    # heavy tails
    H0, H4 = heavy_ed(0), heavy_ed(4)
    dev = max(1 - H[(L, 4.0)][0] for H in (H0, H4) for L in (16, 18))
    d0 = dmrg("heavy_tail_dmrg_h0_V4_L24.txt"); d4 = dmrg("heavy_tail_dmrg_h4_V4_L24.txt")
    dev = max(dev, 1 - np.median(np.abs(d0[:, 3] / d0[:, 5])), 1 - np.median(np.abs(d4[:, 3] / d4[:, 5])))
    check("heavy: interactions change r by < 2% for L >= 16", ["letter"], "less than $2\\%$ for $L\\ge16$",
          dev < 0.02, f"max deviation of median ratio {dev:.3f}")
    hc = [H[k][1] for H in (H0, H4) for k in H if k[1] > 0]
    check("heavy: 99.7-100% correlated with averaged-rate chain", ["letter"], "$99.7$--$100\\%$ correlated",
          min(hc) >= 0.997, f"min {min(hc):.3f}")
    sl = d0[:, 3] @ d0[:, 4] / (d0[:, 4] @ d0[:, 4])
    check("heavy: slope 0.98 against averaged-rate chain (L=24)", ["letter"], "slope $0.98$",
          round(sl, 2) == 0.98, f"{sl:.3f}")
    e0 = renorm_exponents("heavy_tail_renormalized_z0.4_h0.txt")
    e4 = renorm_exponents("heavy_tail_renormalized_z0.4_h4.txt")
    e2 = renorm_exponents("heavy_tail_renormalized_z0.2_h0.txt")
    check("heavy: averaged-rate chain exponent -0.230 (V=0 and V=4)", ["letter"], "$-0.230$",
          round(e0[0.0], 3) == -0.230 and round(e0[4.0], 3) == -0.230, f"{e0}")
    check("heavy: with fields -0.233 and -0.229", ["letter"], "$-0.233$ and $-0.229$",
          round(e4[0.0], 3) == -0.233 and round(e4[4.0], 3) == -0.229, f"{e4}")
    check("heavy: alpha=2.5 control -0.446", ["letter"], "$-0.446$", round(e2[0.0], 3) == -0.446, f"{e2}")

    # lattice-gas symmetry r(V) = r(-V), recomputed
    sys.path.insert(0, os.path.join(ROOT, "interacting"))
    import interacting_pilot as ed
    worst = 0.0
    rng = np.random.default_rng(7)
    for L in (10, 12, 14):
        st = np.sort(ed.basis(L, L // 2))
        occ = np.array([[(s >> i) & 1 for i in range(L)] for s in st], float)
        mv = ed.bond_moves(st, occ, L)
        for _ in range(3):
            h = rng.uniform(-4, 4, L)
            for V in (1.0, 4.0):
                a = ed.residual(ed.slow_profile(len(st), mv, ed.rates(mv, ed.energies(occ, h, V)), occ)[1], L)
                b = ed.residual(ed.slow_profile(len(st), mv, ed.rates(mv, ed.energies(occ, h, -V)), occ)[1], L)
                worst = max(worst, abs(a - b))
    check("lattice gas: r(V) = r(-V) to 1e-14 at L=10-14", ["letter"], "to $10^{-14}$ for $L=10$--$14$",
          worst < 1e-13, f"max |r(V)-r(-V)| {worst:.1e}")

    # preparation circuits and measurement (letter SM)
    prep = open(os.path.join(RES, "preparation_checks.txt")).read()
    worst = float(prep.split("# counter circuit: worst error ")[1].split()[0])
    check("counter circuit reproduces |Psi_f> to 3e-16 up to (L,Q)=(9,5)", ["letter"],
          "to $3\\times10^{-16}$ for $(L,Q)$ up to $(9,5)$", worst <= 3e-16, f"worst {worst:.1e}")
    four = [float(l.split("amplitude ")[1].split(",")[0]) for l in prep.splitlines() if l.strip().startswith("a=")]
    check("four-qubit circuit to 2e-16 at a=-1,0,0.2,1", ["letter"], "to $2\\times10^{-16}$ at $a=-1$", 
          len(four) == 4 and max(four) <= 2.25e-16, f"{four}")
    ang = prep.split("angles at a=0.2 (theta1, theta2|0, theta2|1, beta): ")[1].splitlines()[0].replace(" ", "")
    check("four-qubit angles at a=0.2", ["letter"], "=(" + ang + ")$", True, ang)
    dens = prep.split("densities at a=0.2: ")[1].splitlines()[0].replace(" ", "")
    check("four-qubit densities at a=0.2", ["letter"], "$(" + dens + ")$", True, dens)
    devs = []
    for l in prep.splitlines():
        if l.strip().startswith("sample") and "R12/R11" in l:
            r = float(l.split("r = ")[1].split(";")[0])
            late = [float(q) for q in l.split("/(lam1-lam2): ")[1].split(";")[0].split(",")]
            other = [float(q) for q in l.split("same at eps/3: ")[1].split(",")]
            devs += [abs(late[-1] - r), abs(other[-1] - r)]
    check("measured ratio R12/R11 equals r within 1e-6 at t=10/(lam1-lam2), two eps", ["letter"],
          "agrees with $r$ within $10^{-6}$", len(devs) == 8 and max(devs) <= 1.0e-6 + 1e-12, f"max dev {max(devs):.1e}")

    # non-interacting results (noninteracting/ code and caches)
    import json
    sys.path.insert(0, os.path.join(ROOT, "noninteracting"))
    here = os.getcwd(); os.chdir(os.path.join(ROOT, "noninteracting"))
    from make_main_figure import asymptotic_fits, D_FIELD, LAM_BOND
    asy = {round(a, 3): (e, s) for a, e, s in asymptotic_fits()}
    gs = json.load(open("gammascan.json")); hs = json.load(open("highstat.json"))
    os.chdir(here)
    tab = "$-0.50,\\ -0.51,\\ -0.49$"
    check("Table S2: exponent of d, alpha=4,3,2.5", ["letter"], tab,
          [round(asy[a][0], 2) for a in (4.0, 3.0, 2.5)] == [-0.50, -0.51, -0.49], f"{[asy[a][0] for a in (4.0, 3.0, 2.5)]}")
    check("Table S2: exponent of d at alpha=2", ["letter"], "$-0.448\\pm0.014$",
          round(asy[2.0][0], 3) == -0.448 and round(asy[2.0][1], 3) == 0.014, f"{asy[2.0]}")
    q = [round(asy[a][0], 2) for a in (1.667, 1.429, 1.25, 1.111)]
    check("Table S2: exponent of d, alpha=1.67,1.43,1.25,1.11", ["letter"], "$-0.39,\\ -0.29,\\ -0.15,\\ -0.12$",
          q == [-0.39, -0.29, -0.15, -0.12], f"{q}")
    check("Table S2: random fields h0=1-16, -0.495 to -0.498", ["letter"], "$-0.495$ to $-0.498$",
          round(D_FIELD.max(), 3) == -0.495 and round(D_FIELD.min(), 3) == -0.498, f"{D_FIELD}")
    check("Table S2: rate exponent -2.00 to -2.09", ["letter"], "$-2.00$ to $-2.09$",
          round(LAM_BOND.max(), 2) == -2.00 and round(LAM_BOND.min(), 2) == -2.09, f"{LAM_BOND}")
    gex = []
    for g in gs:
        Ls = sorted(int(k) for k in gs[g])
        gex.append(np.polyfit(np.log(Ls), np.log([abs(gs[g][str(L)]["r2"]) for L in Ls]), 1)[0])
    check("Table S2: quantum exponent of |r| over gamma=0.125-16", ["letter"], "$-0.47$ to $-0.53$",
          round(max(gex), 2) == -0.47 and round(min(gex), 2) == -0.53, f"{[round(x, 3) for x in gex]}")
    Lh = sorted(int(k) for k in hs)
    e1 = np.polyfit(np.log(Lh), np.log([abs(hs[str(L)]["r2"]["mean"]) for L in Lh]), 1)[0]
    check("Table S2: quantum exponent of |r| at gamma=1", ["letter"], "$-0.49\\pm0.02$", round(e1, 2) == -0.49, f"{e1:.3f}")
    lg = open(os.path.join(RES, "letter_figures_log.txt")).read()
    sl = {k: float(lg.split(f"{k}: median-reach slope ")[1].split()[0]) for k in ("bounded", "a1.25")}
    check("Table S2: reach exponents 0.51 and 0.21", ["letter"], "$0.51$ and $0.21$",
          round(sl["bounded"], 2) == 0.51 and round(sl["a1.25"], 2) == 0.21, f"{sl}")
    d20 = float(lg.split("kappa=20: data")[-1].split("max|diff| L>=256: ")[1].split()[0])
    check("Table S2: success probability within 0.022 (kappa=20, L>=256)", ["letter"], "within $0.022$",
          d20 <= 0.022, f"{d20}")
    rel = [float(x) for x in lg.split("rel. diff [")[1].split("]")[0].split()]
    check("Table S2: median reach 9% at L=256, 3% for L=512-4096", ["letter"], "$9\\%$ at $L=256$, $3\\%$ for $512\\le L\\le4096$",
          round(rel[2], 2) <= 0.09 and max(abs(x) for x in rel[3:]) <= 0.03, f"{rel}")
    kr = [float(x) for x in re.findall(r"late-time kappa\|r\| = ([0-9.]+)", lg)]
    check("Fig. 3: kappa|r| = 0.046 and 0.149", ["letter"], "$\\kappa|r|$", kr == [0.046, 0.149], f"{kr}")
    rr = float(lg.split("sample leftover r = ")[1].split()[0])
    check("Fig. 1: r = -0.065 for the shown sample", ["letter"], "gives $r=-0.065$", round(rr, 3) == -0.065, f"{rr}")

    n_fail = results.count("FAIL")
    print(f"\n{len(results)} checks, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
