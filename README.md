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

## Index

To use QALSH for estimating the Chamfer distance, you first need to build an index using `index` command. The following command will index the `./data/federalist` dataset under $\ell_2$ norm:

```bash
./build/qalsh4c index -p 2 -d data/federalist
```

Once the index is built, you can find the index files in the `./data/federalist/index/l2` directory.

## Estimate

We provide four methods for estimating the Chamfer distance:

- Linear Scan ANN
- QALSH ANN
- Uniform Sampling
- QALSH Sampling

You can use the following command to run the QALSH Sampling method to estimate the Chamfer distance between two point sets:

```bash
./build/qalsh4c estimate -p 2 -d ./data/federalist sampling qalsh
```
