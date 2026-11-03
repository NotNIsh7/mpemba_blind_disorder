# Mpemba effect from disorder-blind state preparation: non-interacting code and data

Code and cached data for Figs. 1-3, the Supplemental Material figures, and the numerical-checks
table. The scripts import each other and read their data files by relative name, so run every
command from inside this folder. Requirements: `../requirements.txt`.

## Figures

`./make_figures.sh` runs every figure script, copies the outputs to `../paper/figures/`, and saves
the printed numbers to `../results/letter_figures_log.txt` (read by `../tools/audit_numbers.py`).
Each script takes under a minute with the cached data.

| Paper figure | Script | Data |
|---|---|---|
| Fig. 1 `fig_concept.pdf` | `fig_cancellation.py` | computed |
| Fig. 2 `fig_scaling1d.pdf` | `make_main_figure.py` | `corrections.json`; other exponents listed in the script (see below) |
| Fig. 3 `fig_mpemba.pdf` | `fig_mpemba.py` | `fig_mpemba.json` |
| Fig. 4 `fig_interacting.pdf` | `../interacting/make_fig_interacting.py` | `../results/` |
| SM `fig_profiles.pdf`, `fig_ks.pdf` | `fig_simple.py` | computed; `random_positions.json` |
| SM `fig_quantum_bounded.pdf` | `fig_quantum_bounded.py` | `highstat.json`, `gammascan.json` |
| SM `fig_quantum_hopping.pdf` | `fig_finite_gamma.py` | `finite_gamma_heavy.json`, `finite_gamma_heavy_more.json` |
| SM `fig_heavy_stable.pdf` | `fig_heavy_stable.py` | `fig_mpemba.json`; panel (a) computed |

The field, hopping and Pareto exponents of Fig. 2 for L = 64-4096 (`D_FIELD`, `D_BOND`, `LAM_BOND`,
`D_PAR`) are listed at the top of `make_main_figure.py`; they are the output of
`blind_scaling.py analyse` and `tail_index_test.py`. The diamonds (L >= 4096) are read from
`corrections.json`.

## Libraries

| File | Content |
|---|---|
| `dephased_xx.py` | Full 4^L Liouvillian of the dephased XX chain (small L) |
| `density_sector.py` | Exact correlation-matrix dynamics, dM/dt = -i[h, M] - 4γ(M - diag M) |
| `blind_scaling.py` | Classical resistor chain (strong dephasing): slow mode, distortion d, residual r |
| `bond_disorder.py` | Random hoppings J_i = u^z, α = 1/(2z) |

## Data

| File | Content | Produced by |
|---|---|---|
| `corrections.json` | Pareto chains, L = 64-65536: mean d and its error per (α, L) | `run_corrections.py` |
| `fig_mpemba.json` | absolute residual per resistor-chain sample (bounded, α = 2.5, 1.25), L = 64-4096 | `fig_mpemba.py` |
| `highstat.json` | quantum residuals at γ = 1, bounded fields, L = 24-192 | `highstat.py` (`resume.py L`) |
| `gammascan.json` | quantum d and r for γ = 0.125-16, bounded fields | `gamma_scan.py <gamma>` |
| `finite_gamma_heavy.json`, `finite_gamma_heavy_more.json` | median d (and raw values in `_more`) for random hoppings at finite γ and in the classical chain | `finite_gamma_heavy.py`, `finite_gamma_heavy_more.py` |
| `random_positions.json` | random positions plus fields: exponents, and d and ω per sample | `random_positions.py` |

## Tests and checks

| Script | Content |
|---|---|
| `test_dephased_xx.py`, `test_density_sector.py` | Self-tests; the density sector against the full Liouvillian |
| `audit.py` | Pipeline checks by independent routes (dense vs sparse modes, left = right slow mode, ...) |
| `reduction_lemma_check.py` | The inverse-generator identity and the first-order reduction |
| `kolmogorov_check.py` | √L ω μ/σ follows the Kolmogorov distribution for finite variance |
| `residual_shape.py` | Shape of the slow-mode residual above and below α = 2 |
| `imbalance_check.py` | The Néel state has only O(1/L) weight in the smooth slow mode |

## Notation

| Code | Paper |
|---|---|
| `R`, `R_i` | bond conductance G_i |
| `1/R` | resistance ρ_i |
| `W` | field strength h0 (fields uniform in [-h0, h0]) |
| `Gamma = 4*gamma` | coherence damping rate 4γ |
| `z` | J_i = v^z, α = 1/(2z) |
| `xi`, `l` | ξ and mean spacing ℓ, α = ξ/(2ℓ) |
