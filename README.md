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

The repository includes committed canonical sample datasets:

- `data/gmm`: synthetic Gaussian-mixture sample generated with 1024 points in A, 1024 points in B, 128 dimensions, seed `42`, and all directed $\ell_1$/$\ell_2$ ground-truth fields populated.
- `data/federalist`: 300-dimensional Federalist sample with 1880 points in A and 1178 points in B, with all directed $\ell_1$/$\ell_2$ ground-truth fields populated.

## Dataset Preparation

Datasets are prepared through the C++ CLI. A dataset directory contains `A.bin`, `B.bin`, and `metadata.json`; point files store row-major `double` values, and metadata stores point counts plus optional directed ground truth fields:

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
uses double precision and batches source points from runtime free-memory
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

## Index

To use QALSH for estimating the Chamfer distance, you first need to build an index using `index` command. The following command will index the `./data/federalist` dataset under $\ell_2$ norm:

```bash
./build/dev/qalsh4c index -p 2 -d data/federalist --direction both
```

The `--direction` option accepts `ab`, `ba`, or `both`. `ab` is the default and builds the target-side `B` index for `A -> B` estimation; `ba` builds the target-side `A` index; `both` builds both indexes. Once the index is built, you can find the index files in the `./data/federalist/index/l2` directory.

## Estimate

We provide four methods for estimating the Chamfer distance:

- Linear Scan ANN
- QALSH ANN
- Uniform Sampling
- QALSH Sampling

The `estimate` command also accepts `--direction ab|ba|both` and defaults to `ab`. You can use linear scan on the GMM sample dataset to check the exact directed distances without building an index:

```bash
./build/dev/qalsh4c estimate -p 2 -d ./data/gmm --direction ab ann linear_scan
./build/dev/qalsh4c estimate -p 2 -d ./data/gmm --direction ba ann linear_scan
./build/dev/qalsh4c estimate -p 2 -d ./data/gmm --direction both ann linear_scan
```
