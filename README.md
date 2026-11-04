# Disorder-blind Mpemba acceleration of quantum state preparation: code and data

Code and data for the letter *Disorder-blind Mpemba acceleration of quantum state preparation*. The LaTeX source of the letter
will be added here on submission; until then `paper/` holds the figures only.

A preparation designed to cancel the slow mode of a clean dephased XX chain, a smooth density
modulation cos(2 pi x) orthogonal to cos(pi x), keeps only a small residual overlap r with
the slow mode of a disordered chain. At strong dephasing the density dynamics is a chain of
random resistors. |r| falls as L^-1/2 when the resistances have finite variance and as
L^(1/alpha - 1) for power-law tails with 1 < alpha < 2, while relaxation stays diffusive for
any finite mean. This repository reproduces every figure and every number quoted in the letter.

## Reproducing

    pip install -r requirements.txt
    ./reproduce.sh                      # tests, all figures, number audit (about 2 min)
    ./reproduce.sh --data               # also recompute results/ (days of CPU)

`./reproduce.sh` runs the self-tests of the quantum code, regenerates all figures into
`paper/figures/`, and runs `tools/audit_numbers.py`, which recomputes the letter's quoted
numbers from the data (36 checks). Once the LaTeX source is added it also builds the PDF and
checks that each number appears in the text.

## Where each figure comes from

| Figure | Script | Data |
|---|---|---|
| Fig. 1, Fig. 2, Fig. 3, SM figures (profiles, KS law, quantum bounded fields, quantum hoppings, heavy-tail stable law) | `noninteracting/make_figures.sh` (one script per figure, see `noninteracting/README.md`) | caches `noninteracting/*.json` |
| Fig. 4 (interacting chains) | `interacting/make_fig_interacting.py` | `results/` |
| SM Fig. S1 (preparation circuit) | drawn in LaTeX (quantikz); checked by `preparation/circuits.py` | `results/preparation_checks.txt` |

`results/README.md` maps the remaining quoted numbers (Table S1, the interacting chain, the
preparation section) to their scripts and data.

## Layout

- `paper/figures/` — the figures of the letter, as produced by the scripts.
- `noninteracting/` — the dephased XX chain without interactions: full Liouvillian and exact
  correlation-matrix dynamics, resistor chains to L = 65536, cached data, the figure scripts and
  independent checks.
- `interacting/` — the interacting chain at strong dephasing: exact diagonalization of the
  lattice gas, DMRG for L = 24 and 32 (TeNPy), heavy-tailed hoppings, Fig. 4.
- `preparation/` — gate-level checks of the circuits that prepare the blind and reference
  states, and of the protocol that measures r.
- `results/` — data for Fig. 4, Table S1 and the preparation section.
- `tools/audit_numbers.py` — recomputes each quoted number and checks that the letter quotes it.

## License

Code: MIT. Data, figures and text: CC BY 4.0. See `LICENSE`.
