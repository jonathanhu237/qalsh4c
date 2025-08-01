# QALSH for Chamfer Distance

Fast Chamfer Distance Approximation via Query-Aware Locality-Sensitive Hashing (QALSH).

## Build Instructions

To build the project, you need to have the following dependencies installed:

- [CMake](https://cmake.org/): version 3.28.3 or higher
- [Ninja](https://ninja-build.org/): version 1.11.1 or higher
- [GCC](https://gcc.gnu.org/): version 13.3.0 or higher

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

## Generate Dataset

The executable `qalsh_chamfer` can be used to generate the dataset for Chamfer distance estimation by running the following command:

```bash
./build/qalsh_chamfer generate_dataset -o ./data/toy --in-memory
```

The command above will generate a dataset in the `./data/toy` directory, containing two point sets. Each point set consists of 1000 points, with each point having 256 dimensions. The components of each point are sampled from a uniform distribution in the range of `[-1024, 1024]`.

The `--in-memory` flag will make the data to be generated in memory first and then written to a file. If your device has insufficient memory and you need to generate a large dataset, please remove the `--in-memory` flag.

To check other options for dataset generation, run:

```bash
./build/qalsh_chamfer generate_dataset -h
```

## Index

To use the disk version of QALSH for estimating the Chamfer distance, you first need to build an index using `index` command. The following command will index the `./data/toy` dataset:

```bash
./build/qalsh_chamfer index -d data/toy
```

By default, the approximation ratio is 2.0 and the page size is 4096 bytes. You can use the following command to see how to modify these parameters:

```bash
./build/qalsh_chamfer index -h
```

Once the index is built, you can find the index files in the `./data/toy/index` directory.

## Estimate

We provide four methods for estimating the Chamfer distance:

- Linear Scan ANN
- QALSH ANN
- Uniform Sampling
- QALSH Sampling

Each algorithm has both an disk and in-memory version. You can run the in-memory version by specifying the `--in-memory` flag. For example, the following command estimates the Chamfer distance for the `./data/toy` dataset using the in-memory version of QALSH Sampling:

```bash
./build/qalsh_chamfer estimate -d ./data/toy --in-memory sampling qalsh
```

If you remove the `--in-memory` flag from the command above, it will run the disk version of QALSH Sampling. In this case, you must build an index beforehand; otherwise, the algorithm will not run correctly.

For more options, please use the following command:

```bash
./build/qalsh_chamfer estimate -h
```