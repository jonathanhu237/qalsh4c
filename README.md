# QALSH4C: Chamfer Distance with a Certified Sample Size

This repository holds the code of QALSH4C, which estimates the directed Chamfer distance
CH(A -> B) = sum over a in A of d(a, B) by the importance-sampling estimator of Bakshi et al.:

1. **Crude distances.** QALSH-CrudeNN answers one query per source vector a and returns a crude distance
   D_a >= d(a, B). It scans the projection entries of all m QALSH lines in non-decreasing distance from the query's
   projection (the incremental scan of I-LSH, a "continuous front"), has no candidate budget, and returns as soon as
   the best distance is at most gamma times the radius of the front (gamma = c^1.5 by default).
2. **Sample size.** E[D_a] <= kappa d(a, B), where kappa = 1 + gamma * int_{1/gamma}^inf F(x) dx and F is a binomial tail of
   the QALSH collision probability. The program computes a certified upper bound on kappa from the QALSH parameters
   alone (interval arithmetic, `src/kappa.cc`) and the sample size T = ceil((kappa - 1) / (delta epsilon^2)).
3. **Sampling.** T source vectors are drawn with probability D_a / D, their exact nearest-neighbor distances are
   computed, and eta = (1/T) sum D d(x, B) / D_x is returned.

Everything that the estimator does runs in C++. The Python and shell files under `scripts/vldb` and `scripts/dasfaa`
only run the binary, the FAISS, HNSW and IVF baselines, and summarize results.

## Build

Requirements: CMake 3.28.3 or newer, Ninja, and GCC 13 or newer. CLI11, Eigen and nlohmann-json are vendored under
`third_party`. OpenMP is used when CMake finds it.

```bash
cmake --workflow --preset release   # build/release/qalsh4c, -O3 -DNDEBUG -march=native
cmake --workflow --preset dev       # build/dev/qalsh4c with ASan/UBSan and the tests
ctest --preset dev
```

The QALSH index and search live in [`src/qalsh`](src/qalsh/README.md) and are built and tested with the rest.

## Data layout

A dataset directory holds `A.bin` and `B.bin` (row-major float32 vectors) and `metadata.json`:

```json
{"num_dimensions": 300, "num_points": {"a": 1880, "b": 1178},
 "chamfer_distances": {"l1": {"ab": 37563.8, "ba": 22028.1}, "l2": {"ab": 2719.1, "ba": 1597.5}}}
```

The ground-truth fields are optional; when present, the program prints the relative error. `data/federalist` is a
small example used by the tests.

## Usage

```bash
# Certified kappa and T for l1 at c = 2 (the default continuous front, gamma = c^1.5).
qalsh4c kappa -p 1 -c 2 -e 0.3 0.1

# QALSH4C: T chosen from epsilon and delta (defaults 0.1 and 0.1), or given with -n.
qalsh4c -T 24 estimate -p 2 -d data/federalist --direction ab sampling qalsh -c 2
qalsh4c -T 24 estimate -p 2 -d data/federalist --direction ab sampling -n 469 qalsh -c 2

# Exact Chamfer distance by the blocked linear scan; optionally save d(a, B) of every source vector.
qalsh4c -T 24 estimate -p 2 -d data/federalist --direction ab ann --save-distances exact_l2.bin linear_scan

# Baselines of the paper: uniform sampling and the quadtree weights of Bakshi et al.'s code.
qalsh4c estimate -p 2 -d data/federalist sampling -n 469 uniform
qalsh4c estimate -p 2 -d data/federalist sampling -n 469 quadtree --max-levels 3
```

Options of `estimate ... sampling`:

| Option | Meaning |
|---|---|
| `-n` | Sample size T (otherwise QALSH weights take T from the certified kappa, `-e` and `--delta`) |
| `--use-cache` | Reuse or write the crude distances in the dataset directory (`l<p>_qalsh_weights_A.bin`) |
| `--repeats R` | Draw R independent samples from the same weights; each estimate is logged |
| `--exact-distances FILE` | Look up d(a, B) of the sampled vectors in a float64 file (from `--save-distances`) instead of computing them; the estimates are the same. Used by the accuracy experiments, which repeat the estimator many times |
| `precomputed -f FILE` | Weights from a float32 file, for example saved crude distances or HNSW distances |

Global options of the QALSH search: `-T` threads, `--use-fixed-seed` or `--seed N`, `--qalsh-continuous-ratio`
(the stopping ratio gamma of the continuous front; 0, the default, means c^1.5), and, for the stopping-rule
comparison of the paper, `--qalsh-stop-rule round --qalsh-scan-order quanta --qalsh-advance-check false` (rounds
R_t = R_0 c^t with end-of-round tests, gamma = c, with `--qalsh-mid-round-ratio 0` for pure end-of-round stops) plus
`--qalsh-stop-within-round` (QALSH's test on every new candidate, gamma = c^2). The round rules start from
`--initial-radius`, by default 2^-149 (at most any positive float32 distance). `kappa` takes the same options and
certifies the constant of the selected rule.

## Experiments

`scripts/vldb/v3.sh` runs every experiment of the PVLDB paper, one phase after another, into one results directory:
the cost table (`realcost.sh`), the 50-run randomness test (`randomness.sh`), ANN indexes and other sampling methods
(`ann.sh`, `anncmp.py`, `ann_weights.py`), equal time (`equaltime.sh`), the stopping-rule ablation (`ablation.sh`),
scaling (`dscaling.sh`, `bscaling.sh`), threads (`threads.sh`), the c sweep (`csweep.sh`), the eps sweep
(`epsweep.py`) and memory (`memory.sh`). `summarize_vldb.py` and the `make_*_figure.py` scripts turn the results into
the tables and figures. `scripts/dasfaa/runs/README.md` describes the earlier DASFAA pipeline, whose helpers the VLDB
scripts reuse.
