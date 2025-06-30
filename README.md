# QALSH for Chamfer Distance

The full version of README will be available soon ...

## Build Instructions

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

## Generate Dataset

```bash
./build/dataset-generator -n toy -N 1000 -D 64 -v
```