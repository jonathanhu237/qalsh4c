## Why

Dataset creation is currently split out into Python scripts, while the rest of the project is a C++ CLI. Moving generation, conversion, metadata writing, and ground-truth calculation into the C++ CLI makes dataset preparation a first-class, testable project capability and allows the repository to remove `scripts/`.

## What Changes

- Add a `dataset` command group to the C++ CLI.
- Add `dataset generate gmm` for synthetic Gaussian-mixture point-set generation.
- Add `dataset convert` subcommands for `mnist`, `p53`, `sift`, `gist`, and `trevi`.
- Add `dataset ground-truth` to compute or backfill selected exact directed Chamfer ground-truth fields for existing datasets.
- Add optional `--ground-truth` specs as a single comma-separated argument containing only `l1:ab`, `l1:ba`, `l2:ab`, and `l2:ba`; repeated expanded tasks are deduplicated.
- Allow generated or converted datasets to omit some or all ground-truth values by writing explicit `null` values in metadata.
- Update `estimate` output to report unavailable ground truth and relative error instead of failing when the selected metadata field is `null`.
- Add a small C++ progress reporter for long dataset and ground-truth work.
- Use CPU exact ground-truth calculation in the first change, parallelized with OpenMP when available.
- Support reproducible generation and conversion through explicit `--seed`; conversion defaults to shuffle plus 50/50 split unless overridden.
- Replace Python-only data loaders with C++ implementations for local IDX images, local fvecs, local P53 text data, and a minimal uncompressed BMP subset for Trevi patches.
- **BREAKING**: Remove `scripts/` after C++ dataset commands cover the current generation and conversion workflows.

## Capabilities

### New Capabilities

- `dataset-preparation`: C++ CLI dataset generation, conversion, ground-truth backfill, partial metadata emission, overwrite behavior, and progress reporting.

### Modified Capabilities

- `directed-chamfer-distance`: Dataset metadata ground-truth values may be `null`, and `estimate` must report unavailable ground truth and relative error without failing.

## Impact

- CLI: Adds `dataset generate`, `dataset convert`, and `dataset ground-truth` command paths.
- C++ data model: Metadata parsing and writing must support optional directed ground-truth values.
- C++ implementation: Adds dataset source loaders/generators, split/shuffle handling, binary point-set writing, exact ground-truth computation, metadata updates, and progress reporting.
- Build: Keeps first-stage dependencies to existing C++ dependencies; CUDA support is explicitly deferred to a later change.
- Data import: Uses local raw files only; C++ MNIST conversion does not download OpenML data.
- Trevi conversion: Supports a documented BMP subset instead of depending on OpenCV.
- Tests and docs: Adds CMake/CLI coverage for dataset commands, metadata with `null` ground truth, unavailable estimate output, overwrite behavior, and removal of Python script documentation.
