## 1. Metadata And Shared Dataset Infrastructure

- [x] 1.1 Update `DatasetMetadata` to store optional `l1/l2` `ab/ba` ground-truth values and to load/save explicit JSON `null` values.
- [x] 1.2 Update `GroundTruth` lookup helpers so missing directed values are observable without terminating estimation.
- [x] 1.3 Add shared parsing for comma-separated ground-truth selectors limited to `l1:ab`, `l1:ba`, `l2:ab`, and `l2:ba`, with duplicate selectors deduplicated.
- [x] 1.4 Add reusable point-set writing for row-major `double` `A.bin` and `B.bin` files.
- [x] 1.5 Add output directory validation, `--overwrite` handling, and index invalidation for dataset generation/conversion.

## 2. Exact Ground Truth And Progress

- [x] 2.1 Implement exact directed Chamfer ground-truth computation for L1 and L2 using CPU linear scan.
- [x] 2.2 Parallelize CPU ground-truth computation with OpenMP when `QALSH4C_USE_OPENMP` is available.
- [x] 2.3 Add an internal stderr progress reporter with TTY single-line refresh, non-TTY low-frequency logs, elapsed/rate display, and `--no-progress`.
- [x] 2.4 Integrate progress reporting into ground-truth computation without per-thread stderr writes.
- [x] 2.5 Add `dataset ground-truth` to compute selected missing fields and support `--overwrite` for existing fields.

## 3. Dataset Generation

- [x] 3.1 Add `dataset generate gmm` CLI wiring and command object.
- [x] 3.2 Implement independent two-component GMM parameter generation for A and B.
- [x] 3.3 Implement Eigen-based multivariate normal sampling with core/outlier weights, outlier scale, requested point counts, dimensions, and explicit `--seed`.
- [x] 3.4 Write generated point sets and metadata, computing only requested ground-truth selectors.

## 4. Dataset Conversion Loaders

- [x] 4.1 Add shared conversion pipeline for full in-memory loading, optional shuffle, `--seed`, `--no-shuffle`, `--num-points-a`, and canonical output writing.
- [x] 4.2 Implement MNIST local IDX image loading without labels or downloads.
- [x] 4.3 Implement P53 `K9.data` loading, incomplete-row skipping, `?` filtering, and final-two-column dropping.
- [x] 4.4 Implement fvecs loading shared by `dataset convert sift` and `dataset convert gist`.
- [x] 4.5 Implement `dataset convert sift` using the expected SIFT base/learn/query fvecs files.
- [x] 4.6 Implement `dataset convert gist` using the expected GIST base/learn/query fvecs files.
- [x] 4.7 Implement a minimal uncompressed BMP reader for Trevi-supported BMP variants.
- [x] 4.8 Implement `dataset convert trevi` 64x64 non-overlapping grayscale patch extraction and unsupported-BMP errors.

## 5. CLI And Estimate Integration

- [x] 5.1 Add the `dataset` command group and all `generate`, `convert`, and `ground-truth` subcommands to `src/main.cc`.
- [x] 5.2 Update `estimate` output to print `Ground Truth: unavailable` and `Relative Error: unavailable` when selected metadata values are `null`.
- [x] 5.3 Preserve existing estimate output when selected metadata ground truth is numeric.
- [x] 5.4 Ensure command logs and progress use stderr while command result summaries remain on stdout.

## 6. Tests And Fixtures

- [x] 6.1 Add unit tests for ground-truth selector parsing, validation, and deduplication.
- [x] 6.2 Add metadata load/save tests covering numeric and `null` ground-truth fields.
- [x] 6.3 Add exact directed Chamfer tests for L1/L2 and A-to-B/B-to-A values on small point sets.
- [x] 6.4 Add CLI tests for `dataset generate gmm` output files, metadata shape, seeded reproducibility, and selected ground-truth computation.
- [x] 6.5 Add CLI tests for `dataset ground-truth` backfill, existing-field protection, and `--overwrite`.
- [x] 6.6 Add focused loader tests or CLI fixture tests for MNIST IDX, P53, fvecs, and supported/unsupported BMP inputs.
- [x] 6.7 Add CLI tests for overwrite protection and index directory removal on dataset overwrite.
- [x] 6.8 Add estimate tests for datasets with `null` ground truth and unavailable relative error output.

## 7. Documentation And Script Removal

- [x] 7.1 Update README dataset preparation examples to use the C++ `dataset` commands.
- [x] 7.2 Document ground-truth selector syntax, partial metadata, overwrite behavior, local-only MNIST conversion, and Trevi BMP support scope.
- [x] 7.3 Remove the Python dataset preparation scripts under `scripts/`.
- [x] 7.4 Remove or update any references to Python dataset scripts in tests or documentation.

## 8. Validation

- [x] 8.1 Run the relevant CMake build and test presets.
- [x] 8.2 Run OpenSpec validation for `add-cpp-dataset-pipeline`.
- [x] 8.3 Verify the repository contains no remaining `scripts/` dataset preparation workflow after migration.
