# QALSH for Chamfer Distance

Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH).

## Build Instructions

To build the project, you need to have the following dependencies installed:

- [CMake](https://cmake.org/): version 3.28.3 or higher
- [Ninja](https://ninja-build.org/): version 1.11.1 or higher
- [GCC](https://gcc.gnu.org/): version 13.3.0 or higher

CMake downloads pinned versions of CLI11, Eigen, and nlohmann-json during the
first configure step for each build directory. OpenMP is optional and enabled
automatically when CMake can find a compatible local runtime.

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

## Toy Dataset

The repository includes a small `data/toy` dataset that uses the current metadata schema. It has known directed Chamfer distances under both $\ell_1$ and $\ell_2$:

- `A -> B`: `2.000000`
- `B -> A`: `3.000000`
- `A <-> B`: `5.000000`

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

Generation and conversion refuse to replace `A.bin`, `B.bin`, or `metadata.json` unless `--overwrite` is provided. Overwriting a dataset removes any stale `index/` directory. Ground-truth backfill also refuses to replace existing numeric values unless `--overwrite` is provided.

Known raw datasets can be converted with:

```bash
./build/dev/qalsh4c dataset convert mnist --images train-images-idx3-ubyte -o data/mnist
./build/dev/qalsh4c dataset convert p53 --raw-dir raw/p53 -o data/p53
./build/dev/qalsh4c dataset convert sift --raw-dir raw/sift -o data/sift
./build/dev/qalsh4c dataset convert gist --raw-dir raw/gist -o data/gist
./build/dev/qalsh4c dataset convert trevi --raw-dir raw/trevi -o data/trevi
```

Conversion defaults to shuffling points before a 50/50 A/B split. Use `--seed` for reproducible shuffling, `--no-shuffle` to preserve source order, and `--num-points-a` to choose the A split size. MNIST conversion reads a local IDX image file and does not download data. Trevi conversion uses an internal BMP reader for uncompressed 8-bit grayscale palette and 24-bit BGR BMP files, then extracts non-overlapping 64x64 grayscale patches.

## Index

To use QALSH for estimating the Chamfer distance, you first need to build an index using `index` command. The following command will index the `./data/toy` dataset under $\ell_2$ norm:

```bash
./build/dev/qalsh4c index -p 2 -d data/toy --direction both
```

The `--direction` option accepts `ab`, `ba`, or `both`. `ab` is the default and builds the target-side `B` index for `A -> B` estimation; `ba` builds the target-side `A` index; `both` builds both indexes. Once the index is built, you can find the index files in the `./data/toy/index/l2` directory.

## Estimate

We provide four methods for estimating the Chamfer distance:

- Linear Scan ANN
- QALSH ANN
- Uniform Sampling
- QALSH Sampling

The `estimate` command also accepts `--direction ab|ba|both` and defaults to `ab`. You can use linear scan on the toy dataset to check the exact directed distances without building an index:

```bash
./build/dev/qalsh4c estimate -p 2 -d ./data/toy --direction ab ann linear_scan
./build/dev/qalsh4c estimate -p 2 -d ./data/toy --direction ba ann linear_scan
./build/dev/qalsh4c estimate -p 2 -d ./data/toy --direction both ann linear_scan
```
