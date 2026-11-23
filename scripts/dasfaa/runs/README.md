# DASFAA 2027 experiment drivers

These shell and Python drivers produced the numbers of the DASFAA 2027 version of QALSH4C on Centaurus (AMD Ryzen 9
5900X, 12 cores, 24 hardware threads, 121 GiB). Paths point to that machine: results under
`~/qalsh4c-results/dasfaa-20261124-cpp`, datasets under `~/datasets/qalsh4c`.

The C++ binary does all of the estimator's work: crude distances, the certified kappa and T (`qalsh4c kappa`), the
importance samples and every estimate. The scripts run it, run the FAISS and HNSW baselines, and summarize the output.
Build the binary with the release preset (`cmake --workflow --preset release`, `-O3 -DNDEBUG -march=native`) and copy
it to `$R/bin/qalsh4c` (see `final/common.sh`).

## Pipeline (`final/chain.sh`)

1. **kappa.** `record_kappa` writes `kappa.txt`: the certified kappa, T(eps) and (m, l) for the continuous front at
   c = 1.5, 2 and 3 and for the two round rules of Sect. 4.2. Every later step reads T from this file.
2. **cost.sh.** Per dataset and norm: the exact distances d(a, B) (`ann --save-distances`, untimed), our exact scan
   three times at 12 and at 24 threads, QALSH4C five times at 24 threads with new projections each time, uniform
   sampling and the quadtree weights at the same T. `../summarize_cost.py` writes `cost_summary.json`; FAISS times
   (`../faiss_exact.py`) are reused from the runs of 2026-11-20/21, since they do not depend on QALSH4C.
3. **hnsw_summary.py.** HNSW crude distances (from `../hnsw_weights.py`, reused) as an empirical CrudeNN: graph time,
   D / CH, and the error of 400 estimates at T(eps = 0.1) formed by the binary.
4. **randomness.sh.** 50 full reruns per dataset and norm (`../failure_rate.py`): fresh projections, crude
   distances, and one estimate per eps in {0.3, 0.1}, all by the binary. `../summarize_failure.py` summarizes them.
5. **validate.sh.** Equal-time budgets of uniform sampling and quadtree weights (timed at T = 1 and near the
   budget), end-to-end totals of the three stopping rules with their own certified T on the same projections, and
   end-to-end totals for c = 1.5 and 3. `final/recalib.py` turns the timings into equal-time sample counts and
   errors, with the estimates formed by the binary (`../binary_estimates.py`).
6. **extras.sh.** Thread check (12 against 24 threads), |A| scaling at fixed B, and the c sweep (50 reruns per cell).

`../make_tables.py` and `../csweep_table.py` print the LaTeX rows of the tables. `final/contrast.py` computes the
relative contrast of Table 2 (reused).
