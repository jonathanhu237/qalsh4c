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

We provide four methods for estimating the Chamfer distance:

- Linear Scan ANN
- QALSH ANN
- Uniform Sampling
- QALSH Sampling

Non-disk QALSH uses an in-memory index. QALSH ANN and QALSH sampling build the
needed target-side hash tables inside the current process from `A.bin` and
`B.bin`. Configure the approximation ratio on the QALSH subcommand:

```bash
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction ab ann qalsh -c 2
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction ab sampling -n 100 qalsh -c 2
```

For QALSH sampling, omit `--num-samples` to let the implementation choose the sample count automatically. The automatic count uses the QALSH4C bound
`ceil((c^2 * (1 - delta0) / (1 - c * delta0) - 1) / (delta * epsilon^2))`, where
`delta0 = exp(-m * KL((l - 1) / m, p1))`; tune the estimator with `--epsilon`, `--delta`, and the QALSH subcommand's `-c/--approximation-ratio`.

`sampling qalsh --use-cache` stores and reads raw float32 weight files such as `l2_qalsh_weights_A.bin`. Cache filenames are unchanged from earlier releases, but stale double-encoded caches are rejected by file-size validation and should be deleted.

The `estimate` command also accepts `--direction ab|ba|both` and defaults to `ab`. You can use linear scan on the Federalist sample dataset to check exact directed distances without building an index:

```bash
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction ab ann linear_scan
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction ba ann linear_scan
./build/dev/qalsh4c estimate -p 2 -d ./data/federalist --direction both ann linear_scan
```

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
`<dataset>/index/qalsh/<l1|l2>/<a|b>/` as `meta.json`, `dot_vectors.bin`, and
`tables.bin`. Existing matching indexes are reused; pass `--overwrite` to
rebuild a selected target index.

```bash
./build/dev/qalsh4c index qalsh -p 2 -d ./data/federalist
./build/dev/qalsh4c index qalsh -p 2 -d ./data/federalist --target both --seed 42 --overwrite
./build/dev/qalsh4c estimate --disk-based -p 2 -d ./data/federalist --direction ab ann qalsh
./build/dev/qalsh4c estimate --disk-based -p 2 -d ./data/federalist --direction ab sampling --use-cache -n 100 qalsh
```

Disk QALSH estimates do not build indexes automatically and do not fall back to
linear scan when an index is missing or mismatched. If `-c/--approximation-ratio`
is omitted, disk QALSH uses the value stored in `meta.json`; if it is provided,
it must match the index metadata.

Disk QALSH caches contiguous blocks from `tables.bin` inside each searcher
clone. This preserves the index file format while avoiding one `pread` per table
entry during ANN search. Tune the table cache with
`--disk-qalsh-table-block-size` (bytes per cached table block, default 65536)
and `--disk-qalsh-table-cache-blocks` (cached blocks per searcher, default 64).
The default table cache budget is 4 MiB per searcher clone.
These options do not change candidate point reads; candidate vectors are still
read as individual point records from the target point file. Pass
`--disk-qalsh-profile` to print aggregate query, table block, cache hit/miss,
candidate read, and search-time counters for disk QALSH ANN or disk QALSH
sampling weight generation.

The top-level `-T, --threads` option controls OpenMP parallelism and defaults to the detected system processor count. Pass `-T N` before the subcommand to run with `N` threads, for example `./build/dev/qalsh4c -T 8 estimate ...`.
