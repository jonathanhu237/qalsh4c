# QALSH for Chamfer Distance

Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH).

## Build Instructions

To build the project, you need to have the following dependencies installed:

- [CMake](https://cmake.org/): version 3.28.3 or higher
- [Ninja](https://ninja-build.org/): version 1.11.1 or higher
- [GCC](https://gcc.gnu.org/): version 13.3.0 or higher

Pinned header-only copies of CLI11, Eigen, and nlohmann-json are vendored under
`third_party`, so configure/build does not need network access for project
dependencies. OpenMP is optional and enabled automatically when CMake can find a
compatible local runtime.

QALSH is maintained as an internal module in [`src/qalsh`](src/qalsh/README.md).
No separate qalsh-lib checkout or installed QALSH package is required. Edit and
commit QALSH search/index changes together with the QALSH4C consumers in this
repository. The module's regression tests run with the normal CTest suite when
`BUILD_TESTING=ON`; it has no separate installation or release process.

CUDA ground-truth acceleration is optional. `QALSH4C_ENABLE_CUDA` accepts
`AUTO`, `ON`, or `OFF` and defaults to `AUTO`; `AUTO` builds CUDA support when a
CUDA compiler is available and otherwise builds a CPU-only binary. Use `ON` to
require CUDA at configure time:

```bash
cmake --preset release -DQALSH4C_ENABLE_CUDA=ON
cmake --build --preset release
```

Use the `dev` preset for day-to-day development. It builds
`./build/dev/qalsh4c` with debug symbols, warnings, ASan/UBSan, tests enabled,
and `compile_commands.json`:

```bash
cmake --workflow --preset dev
```

Run the dev test suite separately:

```bash
ctest --preset dev
```

Use the `release` preset for performance experiments and production runs. It
builds `./build/release/qalsh4c` with tests disabled and
`-O3 -DNDEBUG -march=native`; the resulting binary is optimized for the local
CPU and may not be portable to other machines:

```bash
cmake --workflow --preset release
```

The [runtime optimization report](docs/performance/2026-10-22-runtime.md) includes
paired before/after measurements, raw results, correctness checks, and commands
for reproducing the experiments on Centaurus.
The [follow-up query experiments](docs/performance/2026-10-22-query.md) document
additional candidates, their measured regressions, and why none were retained.

## Sample Datasets

The repository includes one committed canonical sample dataset:

- `data/federalist`: 300-dimensional Federalist sample with 1880 points in A and 1178 points in B, with all directed $\ell_1$/$\ell_2$ ground-truth fields populated.

## Dataset Preparation

Datasets are prepared through the C++ CLI. A dataset directory contains `A.bin`, `B.bin`, and `metadata.json`; point files store row-major float32 values, and metadata stores point counts plus optional directed ground truth fields. This is a breaking data-format fork: older double-encoded point files must be regenerated or reconverted.

```json
{
  "num_dimensions": 2,
  "num_points": {
    "a": 2,
    "b": 2
  },
  "chamfer_distances": {
    "l1": {
      "ab": null,
      "ba": null
    },
    "l2": {
      "ab": 2.0,
      "ba": null
    }
  }
}
```

Ground truth is not computed unless requested. Use a single comma-separated `--ground-truth` value with selectors from `l1:ab`, `l1:ba`, `l2:ab`, and `l2:ba`:

```bash
./build/dev/qalsh4c dataset generate gmm \
  --num-points-a 1000 \
  --num-points-b 1000 \
  --num-dimensions 256 \
  --outlier-fraction 0.1 \
  --outlier-scale 4096 \
  --seed 42 \
  --ground-truth l2:ab,l2:ba \
  -o data/generated
```

Existing datasets can be backfilled later:

```bash
./build/dev/qalsh4c dataset ground-truth \
  -d data/generated \
  --ground-truth l1:ab,l1:ba
```

Ground-truth computation supports `--ground-truth-device auto|cpu|cuda` on
generation, conversion, and backfill commands. The default `auto` mode uses the
optional CUDA backend when the binary was built with CUDA and the default CUDA
device can start the computation; otherwise it falls back to CPU before CUDA
work begins. Use `cpu` to force the existing CPU/OpenMP path, or `cuda` to
require CUDA and fail if CUDA support or device memory is unavailable. CUDA
acceleration only applies to exact directed Chamfer ground-truth calculation;
raw dataset parsing, shuffling, and writing remain CPU-side. The CUDA backend
uses float32 precision and batches source points from runtime free-memory
information while keeping target points resident on the default CUDA device.
Use `--cuda-batch-size N` to explicitly cap CUDA source batch size for
ground-truth computation; the default `0` keeps the automatic maximum batch.
Smaller batches can make progress updates more granular on large datasets, but
may add kernel-launch and transfer overhead.

Generation and conversion refuse to replace `A.bin`, `B.bin`, or `metadata.json` unless `--overwrite` is provided. Overwriting a dataset removes any stale `index/` directory. Ground-truth backfill also refuses to replace existing numeric values unless `--overwrite` is provided.

Known raw datasets can be converted with:

```bash
./build/dev/qalsh4c dataset convert mnist --raw-dir raw/mnist -o data/mnist
./build/dev/qalsh4c dataset convert p53 --raw-dir raw/p53 -o data/p53
./build/dev/qalsh4c dataset convert sift --raw-dir raw/sift -o data/sift
./build/dev/qalsh4c dataset convert gist --raw-dir raw/gist -o data/gist
./build/dev/qalsh4c dataset convert trevi --raw-dir raw/trevi -o data/trevi
```

Conversion defaults to shuffling points before a 50/50 A/B split. Use `--seed` for reproducible shuffling, `--no-shuffle` to preserve source order, and `--num-points-a` to choose the A split size. SIFT and GIST conversion read only `sift_base.fvecs` and `gist_base.fvecs`; `learn` and `query` files are kept as raw inputs but are not mixed into the prepared dataset. MNIST conversion reads `train-images-idx3-ubyte` and `t10k-images-idx3-ubyte` from the raw directory and does not download data. Trevi conversion uses an internal BMP reader for uncompressed 8-bit grayscale palette and 24-bit BGR BMP files, then extracts non-overlapping 64x64 grayscale patches.

## Estimate

We provide five methods for estimating the Chamfer distance:

- Linear Scan ANN
- QALSH ANN
- Uniform Sampling
- QALSH Sampling
- Quadtree Sampling

Non-disk QALSH uses an in-memory index. QALSH ANN and QALSH sampling build the
needed target-side hash tables inside the current process from `A.bin` and
`B.bin`. Configure the approximation ratio on the QALSH subcommand:

```bash
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction ab ann qalsh -c 2
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction ab sampling -n 100 qalsh -c 2
```

For QALSH sampling, omit `--num-samples` to let the implementation choose the sample count automatically. The automatic count uses the revised QALSH4C bound
`ceil((kappa_exp - 1) / (delta * epsilon^2))`, where
`kappa_exp = c^2 * (1 + (c - 1) * S_bound)`,
`S_bound = sum_{j=0}^J c^j * Delta_j + B_J`, and `B_J < 1e-10`.
The implementation computes `Delta_j` as the stable miss-probability tail
`Pr[Binomial(m, q_j) >= m - l + 1]`; for L1,
`q_j = 2/pi * atan(2 / (w * c^j))`, and for L2,
`q_j = erfc(w * c^j / (2 * sqrt(2)))`. Tune the estimator with
`--epsilon`, `--delta`, and the QALSH subcommand's `-c/--approximation-ratio`.

`sampling qalsh --use-cache` stores and reads raw float32 weight files such as `l2_qalsh_weights_A.bin`. Cache filenames are unchanged from earlier releases, but stale double-encoded caches are rejected by file-size validation and should be deleted.

`sampling quadtree` is an in-memory benchmark implementation of the deterministic quadtree/DFS importance weights from the Bakshi4C supplementary notebook. It supports L1 and L2 crude-distance weights, uses `--max-levels 3` by default, and preserves the notebook's stack-based DFS sibling order. Quadtree sampling works with `--direction ab`, `ba`, or `both`, but it does not support `--disk-based`:

```bash
./build/dev/qalsh4c estimate -p 1 -d ./data/federalist --direction ab sampling -n 20 quadtree
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction ab sampling -n 20 quadtree --max-levels 3
```

Pass `--use-cache` to reuse generated quadtree weights. Quadtree cache filenames include the norm order, method, max-levels value, and source point set, for example `l1_quadtree_l3_weights_A.bin`, and cache contents are raw float32 weights.

`estimate pool` avoids running QALSH for every source point. Each source point first gets a cheap upper bound,
its distance to a uniform subsample of `-s/--subsample-size` target points, and source points that coincide with
a target point are dropped because they contribute zero. The estimator then draws `-K/--pool-size` points in
proportion to these bounds, runs Modified QALSH only on the distinct pooled points, and resamples
`-n/--num-samples` pool draws in proportion to `D/U` for exact scans. `-n 0` reuses the QALSH4C automatic sample
count as a reference budget; it does not by itself carry the QALSH4C `(epsilon, delta)` guarantee over to the
pooled estimator. `-K 0` (the default) draws five times as many pool points as exact samples, and `--direct`
skips QALSH entirely and scans points drawn in proportion to the upper bounds, which serves as a baseline;
`--uniform` also skips the upper bounds and scans uniformly drawn source points from scratch. The estimate stays
unbiased because each stage divides by its own sampling probability. With `--disk-based`, the estimator streams
`A` once, probes the persistent QALSH index only for pooled points, and scans `B` sequentially for the exact
stage. Logs report the pool size, the number of distinct QALSH queries, and the time spent in each stage:

```bash
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction ab pool -K 300 -n 100 -s 100
```

`--plan model` and `--plan certified` choose `K` and `T` themselves, so they reject `-K` and `-n`. A pilot of draws in
proportion to the upper bounds runs Modified QALSH and an exact scan at every distinct drawn point. A model plan starts
with `--pilot-size` draws (128 by default) and doubles them, up to 4096, until both variance coefficients have a
relative standard error of at most `--pilot-relative-error` (0.25 by default); since plans scale with the
coefficients, this keeps the planning error to about 6% in relative variance. A certified plan keeps `--pilot-size`
fixed. Disjoint pairs of pilot draws give the two variance coefficients of the pooled estimate, whose
relative variance is `lambda0 / K + (1 - 1/K) lambda1 / T`, and the pilot's timings price one QALSH query and one
exact scan. The plan starts from the square-root rule for `K`, tries a few sizes around it using the expected number
of distinct pooled points, and pairs each with the smallest `T` that meets the target. `model` plugs in point
estimates and aims at an RMS relative error of `--target-error` without guaranteeing it. `certified` plugs in
empirical Bernstein upper bounds instead, which bound `P(|estimate / truth - 1| > --target-error)` by `--delta`, half
of it spent on the pilot. A pooled plan switches to querying every source point when that is predicted to finish
sooner, while `--full`, `--direct` and `--uniform` keep their design and only size `T` (`--uniform` with
`--plan model` only). Later stages reuse the pilot's QALSH and
exact results but not its draws, so the estimate stays unbiased. Plans run in memory only, and the log reports the
coefficients, costs and choice:

```bash
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction ab pool --plan model --target-error 0.01
```

The `estimate` command also accepts `--direction ab|ba|both` and defaults to `ab`. You can use linear scan on the Federalist sample dataset to check exact directed distances without building an index:

```bash
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction ab ann linear_scan
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction ba ann linear_scan
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction both ann linear_scan
```

Exact nearest neighbors, in linear scan and in the exact stage of every sampling estimator (in memory and
disk-based), are found in batches. A fast pass computes approximate distances for blocks of targets: blocked matrix
products through `||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>` for L2, and vectorized sums of absolute differences
shared by groups of queries for L1. A rounding-error bound decides which targets can still win, and every one of them
is rescored with the scalar kernel, so the distances are the same as a scalar scan's. Pass `--scalar-exact-scan`
before the subcommand to use the scalar kernel throughout, as an ablation.

Pass `--disk-based` at the `estimate` level to use an external-memory
implementation where available. Disk-based exact linear scan streams query
points from disk in batches and scans target points in blocks instead of loading
either point set in full. Disk-based uniform sampling samples query point ids
without materializing a full weight vector, then uses the same disk-based exact
scan for sampled points. Tune the target block size with `--disk-block-size` and
the query batch size with `--disk-query-batch-size`:

```bash
./build/dev/qalsh4c estimate --disk-based --disk-block-size 4096 --disk-query-batch-size 64 -p 2 -d ./data/federalist --direction ab ann linear_scan
```

Disk-based QALSH requires an explicit target-side index. The default target is
`b`, matching the default `--direction ab`; use `--target a` for `ba`, or
`--target both` before bidirectional estimates. Indexes are written under
`<dataset>/index/qalsh/<l1|l2>/<a|b>/` as a sidecar `meta.json` plus the
internal module’s immutable `index.qalsh` persistent file. The default layout is a
B+ tree; pass `--layout sorted-array` to choose the persistent sorted-array
layout. The file stores the actual projection vectors and layout metadata, so
query processes do not regenerate projections. Existing matching indexes are
reused; pass `--overwrite` to rebuild a selected target index. Historical flat
`dot_vectors.bin`/`tables.bin` indexes must be rebuilt. The build also writes
`fingerprints.bin`, one 64-bit hash per target point, which `estimate --disk-based pool` uses to detect source
points that coincide with a target point without reading `B` again; hash hits are still confirmed by comparing the
points. Running the build again on an existing index adds the file if it is missing, and estimates rehash `B` when
the file is absent or no longer matches the target points.

```bash
./build/dev/qalsh4c index qalsh -p 2 -d ./data/federalist
./build/dev/qalsh4c index qalsh -p 2 -d ./data/federalist --layout sorted-array --overwrite
./build/dev/qalsh4c index qalsh -p 2 -d ./data/federalist --target both --seed 42 --overwrite
./build/dev/qalsh4c estimate --disk-based -p 2 -d ./data/federalist --direction ab ann qalsh
./build/dev/qalsh4c estimate --disk-based -p 2 -d ./data/federalist --direction ab sampling --use-cache -n 100 qalsh
```

Disk QALSH estimates do not build indexes automatically and do not fall back to
linear scan when an index is missing or mismatched. If `-c/--approximation-ratio`
is omitted, disk QALSH uses the value stored in `meta.json`; if it is provided,
it must match the index metadata.

Disk QALSH opens the recorded index layout read-only and uses the shared
search engine for both B+ tree pages and sorted-array regions. The internal QALSH module owns
the persistent format and shares immutable metadata across searcher clones;
candidate vectors remain caller-owned and are read by point ID. The legacy `--disk-qalsh-table-block-size` and
`--disk-qalsh-table-cache-blocks` options remain accepted for workflow
compatibility, but traversal behavior is controlled by the index's stored
layout and page/region size. Pass `--disk-qalsh-profile` to print query,
logical projection-hit, candidate-read, and search-time counters; software
traversal counters are reported separately from physical I/O.

The top-level `-T, --threads` option controls OpenMP parallelism and defaults to the detected system processor count. Pass `-T N` before the subcommand to run with `N` threads, for example `./build/dev/qalsh4c -T 8 estimate ...`.
