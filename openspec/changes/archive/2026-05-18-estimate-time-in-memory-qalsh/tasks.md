## 1. In-Memory QALSH Search

- [x] 1.1 Add an in-memory QALSH index structure that owns target points, QALSH config, dot vectors, and sorted projection arrays.
- [x] 1.2 Replace `QalshAnnSearcher` disk index loading and B+ tree searchers with shared in-memory index construction and sorted-array cursor scanning.
- [x] 1.3 Ensure `QalshAnnSearcher::Clone()` shares immutable index state across OpenMP clones and only duplicates mutable search state.
- [x] 1.4 Make QALSH ANN configuration use constructor-provided approximation ratio and estimate-time `QalshConfig::Regularize()`.

## 2. Estimate And CLI Behavior

- [x] 2.1 Remove the active `index` command from CLI usage and stop validating disk indexes before QALSH-backed estimates.
- [x] 2.2 Preserve direction-scoped construction so `ab` builds only B-side state, `ba` builds only A-side state, and `both` runs the directions sequentially.
- [x] 2.3 Update README examples and help expectations so users run QALSH estimates directly without a prebuilt index.

## 3. QALSH Sampling Weights

- [x] 3.1 Add `sampling qalsh -c/--approximation-ratio`.
- [x] 3.2 Replace implicit `sampling --use-cache` behavior for QALSH weights with `sampling qalsh --save-weights <dir>` and `--load-weights <dir>`.
- [x] 3.3 Implement the weights directory format with `metadata.json` and `weights.bin`, including validation for direction, norm, source/target metadata, file identity, and QALSH config.
- [x] 3.4 Use generated or loaded QALSH config for automatic sampling sample-count selection without reading `config.json`.
- [x] 3.5 Reject combined `--save-weights` and `--load-weights` usage.

## 4. Tests And Cleanup

- [x] 4.1 Update or remove index-specific CLI tests that expect disk index creation or missing-index failures.
- [x] 4.2 Add or update QALSH ANN tests to exercise estimate-time in-memory search without a prebuilt index.
- [x] 4.3 Add QALSH sampling tests for save/load weights and metadata mismatch handling.
- [x] 4.4 Update CMake target source lists if B+ tree or external sorter files are no longer needed by the main binary.
- [x] 4.5 Run OpenSpec validation and the relevant CMake tests using the Federalist dataset only for dataset-dependent local tests.
