#!/usr/bin/env bash
# Figs. 1-3 and the SM figures from the cached data in this folder, copied to paper/figures/.
# Fig. 4 is made by interacting/make_fig_interacting.py.
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PY=${PYTHON:-python3}
FIG=../paper/figures
LOG=../results/letter_figures_log.txt
exec > >(tee "$LOG") 2>&1     # read by tools/audit_numbers.py
$PY fig_cancellation.py   && cp fig_cancellation.pdf "$FIG/fig_concept.pdf"
$PY make_main_figure.py   && cp tail_index_law.pdf   "$FIG/fig_scaling1d.pdf"
$PY fig_mpemba.py         && cp fig_mpemba.pdf       "$FIG/fig_mpemba.pdf"
$PY fig_simple.py         && cp fig_profile.pdf      "$FIG/fig_profiles.pdf" \
                          && cp fig_law.pdf          "$FIG/fig_ks.pdf"
$PY fig_quantum_bounded.py && cp fig_quantum_bounded.pdf "$FIG/fig_quantum_bounded.pdf"
$PY fig_finite_gamma.py   && cp fig_finite_gamma.pdf "$FIG/fig_quantum_hopping.pdf"
$PY fig_heavy_stable.py   && cp fig_heavy_stable.pdf "$FIG/fig_heavy_stable.pdf"
rm -f ./*.pdf ./*.png
echo "letter figures installed in paper/figures/"
