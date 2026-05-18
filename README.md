# QALSH for Chamfer Distance

Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH).

## Build Instructions

To build the project, you need to have the following dependencies installed:

- [CMake](https://cmake.org/): version 3.28.3 or higher
- [Ninja](https://ninja-build.org/): version 1.11.1 or higher
- [GCC](https://gcc.gnu.org/): version 13.3.0 or higher
- [Vcpkg](https://vcpkg.io/en/): version 2025-06-20 or higher

To build the project in release mode, run the following commands:

```bash
cmake --preset release && cmake --build --preset release-build
```

Besides the release build, you can also build the project in debug mode:

```bash
cmake --preset debug && cmake --build --preset debug-build
```

or in relWithDebInfo mode:

```bash
cmake --preset relWithDebInfo && cmake --build --preset relWithDebInfo-build
```

## Toy Dataset

The repository includes a small `data/toy` dataset that uses the current metadata schema. It has known directed Chamfer distances under both $\ell_1$ and $\ell_2$:

- `A -> B`: `2.000000`
- `B -> A`: `3.000000`
- `A <-> B`: `5.000000`

## Index

To use QALSH for estimating the Chamfer distance, you first need to build an index using `index` command. The following command will index the `./data/toy` dataset under $\ell_2$ norm:

```bash
./build/qalsh4c index -p 2 -d data/toy --direction both
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
./build/qalsh4c estimate -p 2 -d ./data/toy --direction ab ann linear_scan
./build/qalsh4c estimate -p 2 -d ./data/toy --direction ba ann linear_scan
./build/qalsh4c estimate -p 2 -d ./data/toy --direction both ann linear_scan
```
