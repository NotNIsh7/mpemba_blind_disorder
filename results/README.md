# results/

Data behind Fig. 4, Table S1, the interacting-chain numbers and the preparation section of
the letter. Every file is produced by one command in `../reproduce.sh --data` (seeded, so a
rerun reproduces it). CPU times are single-core and approximate. The data behind Figs. 1-3,
the other SM figures and Table S2 are the caches in `../noninteracting/` (see its README).

| Item in the letter | Script | Data | Cost |
|---|---|---|---|
| Fig. 4(a), Table S1 bounded rows, factor 1.6, rms of sqrt(L) r | `interacting/interacting_pilot.py`, `interacting/dmrg_slow_mode.py` | `interacting_pilot_h4_L8-16.txt`, `_L18.txt`, `_L20.txt`; `dmrg_V4_L24_seed1.txt`, `dmrg_V4_L32_seed2.txt`, `dmrg_V4_L32_seed3.txt` | ED 1-3 h; DMRG 1.5-2 min per sample at L = 24, 4 min at L = 32 |
| Fig. 4(b), (d) | `interacting/dmrg_slow_mode.py` | DMRG files above; `heavy_tail_dmrg_h0_V4_L24.txt` | |
| Fig. 4(c), Table S1 heavy rows, "less than 2%" | `interacting/heavy_tail_hopping.py`, `interacting/dmrg_slow_mode.py` | `heavy_tail_hopping_z0.4_h{0.0,4.0}_L*.npz`, `heavy_tail_hopping_z0.4_h{0,4}.txt`, `heavy_tail_dmrg_h{0,4}_V4_L24.txt` | about 1 h per ensemble; DMRG 2 min per sample |
| Exponents -0.230, -0.233/-0.229, -0.446 (averaged-rate chain to L = 4096) | `interacting/heavy_tail_renormalized.py` | `heavy_tail_renormalized_z{0.4,0.2}_h*.txt` | minutes |
| DMRG convergence ("sixth digit") | `interacting/dmrg_slow_mode.py` at chi_max 24/48/96 | `dmrg_chi_convergence.txt` (collected from those runs) | |
| r(V) = r(-V) to 1e-14 (lattice gas) | recomputed by `tools/audit_numbers.py` | none | seconds |
| SM Sec. VIII: circuit accuracies, four-qubit angles and densities, measured ratio R12/R11 = r | `preparation/circuits.py` | `preparation_checks.txt` | seconds |
| Numbers printed by the figure scripts of Figs. 1-3 and the SM (checked by the audit) | `noninteracting/make_figures.sh` | `letter_figures_log.txt` | about 2 min |
