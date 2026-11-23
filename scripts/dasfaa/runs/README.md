# DASFAA 2027 experiment drivers

These are the shell and Python drivers that produced the numbers of the DASFAA 2027 version of QALSH4C, as they ran on
Centaurus (AMD Ryzen 9 5900X, 12 cores, 24 hardware threads, 121 GiB) on 2026-11-20 and 2026-11-21. Paths point to
that machine: results under `~/qalsh4c-results/dasfaa-20261120`, datasets under `~/datasets/qalsh4c`.

Build the binary with the release preset (`cmake --preset release`), which sets `-O3 -DNDEBUG -march=native`. A build
without `-march=native` makes the exact scan about 25% slower. The final runs used `build/release-soa`, configured with
the same flags.

## Order

1. `sweeps/screen.sh` screens candidate datasets with beta = 1 (exact NN files, one exact scan, one QALSH4C run).
2. `sweeps/mlsweep.sh` and `sweeps/mlsweep2.sh` compare explicit (m, l) and (w, m, l) choices with the beta = 1
   defaults. They ran on an early build without `-march=native`, so only their relative timings mean something.
   `sweeps/ml_grid.py` maps kappa(m, l) against a counting proxy.
3. `final/chain.sh` runs the final pipeline: `cost.sh` (exact scan three times, QALSH4C five times, FAISS, uniform and
   quadtree), `contrast.py` (relative contrast), `randomness.sh` (50 full reruns per dataset and norm), the summaries
   (`../summarize_cost.py`, `../summarize_failure.py`, `../time_accuracy.py`), and `extras.sh` (thread check, |A|
   scaling, HNSW, c sweep, stopping rules).
4. `final/exact12.sh` reruns every exact baseline at 12 threads, which is faster for the exact scan.
5. `final/validate.sh` times uniform sampling and quadtree weights at the equal-time budgets, and measures the stopping
   rules and c = 1.5 and 3 end to end. `final/recalib.py` turns these timings into the equal-time sample counts and
   errors of the paper.

`../make_tables.py` and `../csweep_table.py` print the LaTeX rows of the tables. `common.sh` holds the parameters:
beta = 1, (m, l) = (36, 18) for l1 and (17, 12) for l2 at c = 2, kappa_hat 2.136466 and 2.120719, T = 1,137 and 1,121.
