#!/usr/bin/env bash
# Reproduce the figures and quoted numbers of "Mpemba effect from disorder-blind state preparation".
#
#   ./reproduce.sh          tests, all figures, the PDFs, and the number audit (about 3 min)
#   ./reproduce.sh --data   additionally recompute the interacting and preparation data in
#                           results/ (days of CPU; results/README.md lists the cost of each run)
#
# The non-interacting runs behind noninteracting/*.json are listed in noninteracting/README.md.
# Every computation is seeded; recomputed files match the stored ones.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MPLBACKEND=Agg
PY=${PYTHON:-python3}

if [[ "${1:-}" == "--data" ]]; then
  R=results
  # interacting lattice gas, exact diagonalization (bounded fields)
  $PY interacting/interacting_pilot.py 4.0 300 8,10,12,14,16 0,1,2,4 > $R/interacting_pilot_h4_L8-16.txt
  $PY interacting/interacting_pilot.py 4.0 300 18 0,1,2,4          > $R/interacting_pilot_h4_L18.txt
  $PY interacting/interacting_pilot.py 4.0 100 20 0,1,2,4          > $R/interacting_pilot_h4_L20.txt
  # DMRG, bounded fields and heavy-tailed hoppings (arguments: h0 V L nsamp seed chi_max [z])
  ( cd interacting
    $PY -W ignore dmrg_slow_mode.py 4.0 4.0 24 40 1 64      2>/dev/null > ../$R/dmrg_V4_L24_seed1.txt
    $PY -W ignore dmrg_slow_mode.py 4.0 4.0 32 15 2 64      2>/dev/null > ../$R/dmrg_V4_L32_seed2.txt
    $PY -W ignore dmrg_slow_mode.py 4.0 4.0 32 15 3 64      2>/dev/null > ../$R/dmrg_V4_L32_seed3.txt
    $PY -W ignore dmrg_slow_mode.py 0.0 4.0 24 40 11 64 0.4 2>/dev/null > ../$R/heavy_tail_dmrg_h0_V4_L24.txt
    $PY -W ignore dmrg_slow_mode.py 4.0 4.0 24 40 12 64 0.4 2>/dev/null > ../$R/heavy_tail_dmrg_h4_V4_L24.txt )
  # heavy-tailed hoppings: exact diagonalization and the averaged-rate chain to L = 4096
  $PY interacting/heavy_tail_hopping.py 0.4 400 8,10,12,14,16,18 0,2,4 0 > $R/heavy_tail_hopping_z0.4_h0.txt
  $PY interacting/heavy_tail_hopping.py 0.4 400 8,10,12,14,16,18 0,2,4 4 > $R/heavy_tail_hopping_z0.4_h4.txt
  $PY interacting/heavy_tail_renormalized.py 0.4 1000 0 > $R/heavy_tail_renormalized_z0.4_h0.txt
  $PY interacting/heavy_tail_renormalized.py 0.4 1000 4 > $R/heavy_tail_renormalized_z0.4_h4.txt
  $PY interacting/heavy_tail_renormalized.py 0.2 1000 0 > $R/heavy_tail_renormalized_z0.2_h0.txt
  # preparation circuits and the measurement protocol
  $PY preparation/circuits.py > $R/preparation_checks.txt
fi

# self-tests of the quantum code (full Liouvillian and density sector)
( cd noninteracting && $PY test_dephased_xx.py | tail -1 && $PY test_density_sector.py | tail -1 )
# Figs. 1-3 and the SM figures, from the caches in noninteracting/ (about two minutes)
./noninteracting/make_figures.sh > /dev/null
# Fig. 4, from results/
$PY interacting/make_fig_interacting.py > /dev/null
# the letter and its derivation manual
for tex in main Derivation_Manual; do
  [[ -f paper/$tex.tex ]] || continue
  ( cd paper && pdflatex -interaction=nonstopmode -halt-on-error $tex.tex > /dev/null
    if grep -q '\\bibdata' $tex.aux; then bibtex $tex > /dev/null; fi
    for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error $tex.tex > /dev/null; done )
  echo "built paper/$tex.pdf"
done
# every quoted number, checked against results/ and the caches
$PY tools/audit_numbers.py
