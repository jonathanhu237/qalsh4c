# dataset-preparation Specification

## Purpose
TBD - created by archiving change add-cpp-dataset-pipeline. Update Purpose after archive.
## Requirements
### Requirement: Dataset command group exposes preparation workflows
The system SHALL provide C++ CLI commands for dataset generation, dataset conversion, and ground-truth backfill.

#### Scenario: Generate GMM dataset command is available
- **WHEN** the user requests help for `dataset generate gmm`
- **THEN** the system shows options for synthetic GMM dataset creation

#### Scenario: Convert known dataset commands are available
- **WHEN** the user requests help for `dataset convert`
- **THEN** the system exposes conversion subcommands for `mnist`, `p53`, `sift`, `gist`, and `trevi`

#### Scenario: Ground truth backfill command is available
- **WHEN** the user requests help for `dataset ground-truth`
- **THEN** the system shows options for computing selected directed Chamfer ground-truth fields for an existing dataset

### Requirement: Dataset output uses the canonical C++ format
The system SHALL write prepared datasets as `A.bin`, `B.bin`, and `metadata.json` using row-major `double` point values and the current structured metadata schema.

#### Scenario: Prepared dataset contains canonical files
- **WHEN** a dataset generation or conversion command succeeds
- **THEN** the output directory contains `A.bin`, `B.bin`, and `metadata.json`

#### Scenario: Metadata records dimensions and point counts
- **WHEN** a dataset generation or conversion command succeeds
- **THEN** `metadata.json` records `num_dimensions`, `num_points.a`, and `num_points.b`

#### Scenario: Metadata records all directed ground-truth fields
- **WHEN** a dataset generation or conversion command succeeds
- **THEN** `metadata.json` contains `chamfer_distances.l1.ab`, `chamfer_distances.l1.ba`, `chamfer_distances.l2.ab`, and `chamfer_distances.l2.ba`

### Requirement: Ground-truth selection is explicit
The system SHALL compute exact ground truth only for explicitly requested `--ground-truth` entries.

#### Scenario: No ground truth requested
- **WHEN** the user runs a dataset generation or conversion command without `--ground-truth`
- **THEN** the system writes all ground-truth metadata fields as `null`

#### Scenario: Selected ground truth requested
- **WHEN** the user runs a dataset generation or conversion command with `--ground-truth l2:ab,l1:ba`
- **THEN** the system computes only `chamfer_distances.l2.ab` and `chamfer_distances.l1.ba`
- **AND** the unrequested ground-truth metadata fields are `null`

#### Scenario: Invalid ground truth selector is rejected
- **WHEN** the user passes a selector other than `l1:ab`, `l1:ba`, `l2:ab`, or `l2:ba`
- **THEN** the system exits with an error explaining the supported selector values

#### Scenario: Duplicate ground truth selectors are deduplicated
- **WHEN** the user passes duplicate selectors in the comma-separated `--ground-truth` argument
- **THEN** the system computes each unique selector at most once

### Requirement: Dataset ground-truth command updates existing metadata
The system SHALL allow users to compute selected exact directed Chamfer ground-truth fields for an existing canonical dataset.

#### Scenario: Backfill missing ground truth
- **WHEN** the user runs `dataset ground-truth -d <dataset> --ground-truth l2:ab` and `chamfer_distances.l2.ab` is `null`
- **THEN** the system computes the exact L2 A-to-B Chamfer component and writes it to metadata

#### Scenario: Existing ground truth is protected
- **WHEN** the user runs `dataset ground-truth -d <dataset> --ground-truth l2:ab` and `chamfer_distances.l2.ab` already has a numeric value
- **THEN** the system exits with an error unless `--overwrite` is provided

#### Scenario: Existing ground truth can be overwritten
- **WHEN** the user runs `dataset ground-truth -d <dataset> --ground-truth l2:ab --overwrite`
- **THEN** the system recomputes and replaces `chamfer_distances.l2.ab`

### Requirement: Exact ground truth uses directed Chamfer semantics
The system SHALL compute each requested ground-truth field as the exact directed Chamfer sum for its norm and direction.

#### Scenario: Compute A-to-B ground truth
- **WHEN** the user requests `l2:ab`
- **THEN** the system stores `sum_{a in A} min_{b in B} distance_l2(a,b)` in `chamfer_distances.l2.ab`

#### Scenario: Compute B-to-A ground truth
- **WHEN** the user requests `l1:ba`
- **THEN** the system stores `sum_{b in B} min_{a in A} distance_l1(b,a)` in `chamfer_distances.l1.ba`

#### Scenario: OpenMP build computes the same value
- **WHEN** the project is built with OpenMP support
- **THEN** ground-truth computation may run in parallel and produces the same value as the single-threaded exact scan

### Requirement: GMM generation creates synthetic datasets
The system SHALL generate synthetic datasets from independent two-component Gaussian mixtures for point sets A and B.

#### Scenario: Generate GMM point sets
- **WHEN** the user runs `dataset generate gmm` with point counts, dimensions, outlier fraction, outlier scale, and output directory
- **THEN** the system writes A and B point sets with the requested counts and dimensionality

#### Scenario: Seeded GMM generation is reproducible
- **WHEN** the user runs `dataset generate gmm` twice with the same explicit `--seed` and parameters
- **THEN** the generated point data is reproducible

### Requirement: Dataset conversion supports split and shuffle controls
The system SHALL convert loaded raw points into A and B using deterministic split controls.

#### Scenario: Default split
- **WHEN** the user converts a raw dataset without `--num-points-a`
- **THEN** the system places half of the loaded points in A after applying the selected shuffle behavior and places the remaining points in B

#### Scenario: Explicit A size
- **WHEN** the user converts a raw dataset with `--num-points-a <n>`
- **THEN** the system places `<n>` points in A after applying the selected shuffle behavior and places the remaining points in B

#### Scenario: Seeded shuffle is reproducible
- **WHEN** the user converts a raw dataset twice with the same explicit `--seed`
- **THEN** the A/B split is reproducible

#### Scenario: Shuffle can be disabled
- **WHEN** the user converts a raw dataset with `--no-shuffle`
- **THEN** the system splits points in their loaded order

### Requirement: MNIST conversion reads local IDX images
The system SHALL convert MNIST from a local IDX image file without downloading data.

#### Scenario: Convert local MNIST images
- **WHEN** the user runs `dataset convert mnist` with a local IDX image file and output directory
- **THEN** the system flattens each image into a double vector and writes the canonical dataset output

#### Scenario: MNIST labels are not required
- **WHEN** the user runs `dataset convert mnist`
- **THEN** the system does not require a labels file

### Requirement: P53 conversion reads local K9 data
The system SHALL convert P53 from a local `K9.data` file.

#### Scenario: Convert P53 rows
- **WHEN** the user runs `dataset convert p53` with a directory containing `K9.data`
- **THEN** the system parses complete numeric rows, skips rows containing `?`, drops the final two CSV columns, and writes the canonical dataset output

### Requirement: SIFT and GIST conversion read local fvecs files
The system SHALL convert SIFT and GIST from local fvecs files.

#### Scenario: Convert SIFT fvecs files
- **WHEN** the user runs `dataset convert sift` with a raw directory containing the expected SIFT fvecs files
- **THEN** the system loads base, learn, and query vectors and writes the canonical dataset output

#### Scenario: Convert GIST fvecs files
- **WHEN** the user runs `dataset convert gist` with a raw directory containing the expected GIST fvecs files
- **THEN** the system loads base, learn, and query vectors and writes the canonical dataset output

### Requirement: Trevi conversion reads a documented BMP subset
The system SHALL convert Trevi image patches from local BMP files using an internal minimal BMP reader.

#### Scenario: Convert supported BMP files
- **WHEN** the user runs `dataset convert trevi` with a raw directory containing supported uncompressed BMP files
- **THEN** the system extracts non-overlapping 64x64 grayscale patches as 4096-dimensional double vectors and writes the canonical dataset output

#### Scenario: Reject unsupported BMP files
- **WHEN** the user runs `dataset convert trevi` with an unsupported BMP variant
- **THEN** the system exits with an error explaining that the BMP encoding is unsupported

### Requirement: Output overwrite protects existing datasets
The system SHALL avoid overwriting existing dataset files unless explicitly requested.

#### Scenario: Existing dataset files are protected
- **WHEN** a dataset generation or conversion command targets a directory containing `A.bin`, `B.bin`, or `metadata.json`
- **THEN** the system exits with an error unless `--overwrite` is provided

#### Scenario: Overwrite replaces dataset files and invalidates indexes
- **WHEN** a dataset generation or conversion command runs with `--overwrite`
- **THEN** the system replaces `A.bin`, `B.bin`, and `metadata.json`
- **AND** removes any existing `index/` directory under the output directory

### Requirement: Dataset preparation reports progress
The system SHALL report progress for long-running dataset preparation work without polluting stdout.

#### Scenario: Interactive progress uses stderr
- **WHEN** a long-running dataset preparation or ground-truth command runs in an interactive terminal
- **THEN** the system displays progress on stderr with percent, current, total, rate, and elapsed time

#### Scenario: Non-interactive progress degrades to logs
- **WHEN** stderr is not an interactive terminal
- **THEN** the system reports progress as low-frequency stderr log lines

#### Scenario: Progress can be disabled
- **WHEN** the user passes `--no-progress`
- **THEN** the system suppresses progress reporting

### Requirement: Python dataset scripts are removed
The repository SHALL remove the Python dataset scripts after their workflows are available in the C++ CLI.

#### Scenario: Scripts directory is absent
- **WHEN** the change is complete
- **THEN** the repository no longer contains the Python dataset preparation scripts under `scripts/`

### Requirement: Ground-truth computation supports backend selection
The system SHALL allow dataset preparation commands that compute exact ground truth to select a computation backend using `--ground-truth-device auto|cpu|cuda`, defaulting to `auto`.

#### Scenario: Ground-truth backend option is available for generation
- **WHEN** the user requests help for `dataset generate gmm`
- **THEN** the help output includes `--ground-truth-device`

#### Scenario: Ground-truth backend option is available for conversion
- **WHEN** the user requests help for a `dataset convert` subcommand
- **THEN** the help output includes `--ground-truth-device`

#### Scenario: Ground-truth backend option is available for backfill
- **WHEN** the user requests help for `dataset ground-truth`
- **THEN** the help output includes `--ground-truth-device`

#### Scenario: CPU backend can be forced
- **WHEN** the user computes selected ground truth with `--ground-truth-device cpu`
- **THEN** the system computes exact directed Chamfer ground truth using the CPU implementation

#### Scenario: CUDA backend can be required
- **WHEN** the user computes selected ground truth with `--ground-truth-device cuda`
- **THEN** the system attempts to compute exact directed Chamfer ground truth using CUDA
- **AND** exits with an error if CUDA is not available or cannot start the computation

#### Scenario: Auto backend falls back before CUDA starts
- **WHEN** the user computes selected ground truth with `--ground-truth-device auto`
- **AND** CUDA is not available or cannot start the computation
- **THEN** the system computes exact directed Chamfer ground truth using the CPU implementation

### Requirement: CUDA ground-truth computation preserves exact Chamfer semantics
The CUDA backend SHALL compute the same exact directed Chamfer definition as the CPU backend for supported norms using double-precision point values.

#### Scenario: CUDA computes L1 directed Chamfer
- **WHEN** CUDA ground-truth computation is selected for `l1:ab`
- **THEN** the system stores `sum_{a in A} min_{b in B} distance_l1(a,b)` in `chamfer_distances.l1.ab`

#### Scenario: CUDA computes L2 directed Chamfer
- **WHEN** CUDA ground-truth computation is selected for `l2:ba`
- **THEN** the system stores `sum_{b in B} min_{a in A} distance_l2(b,a)` in `chamfer_distances.l2.ba`

#### Scenario: CUDA results match CPU within floating-point tolerance
- **WHEN** the same canonical dataset ground truth is computed with CPU and CUDA
- **THEN** the CUDA result matches the CPU result within an appropriate floating-point tolerance

### Requirement: CUDA ground-truth computation batches source points by available GPU memory
The CUDA backend SHALL derive a source-point batch size from runtime CUDA memory availability instead of requiring all source points to fit on the GPU at once.

#### Scenario: Source set exceeds remaining GPU memory
- **WHEN** the target point set fits on the CUDA device
- **AND** the full source point set would exceed the configured CUDA memory budget
- **THEN** the system processes source points in multiple CUDA batches

#### Scenario: Target set cannot fit on CUDA device in auto mode
- **WHEN** `--ground-truth-device auto` is selected
- **AND** the target point set plus at least one source point cannot fit within the CUDA memory budget
- **THEN** the system falls back to CPU before starting CUDA computation

#### Scenario: Target set cannot fit on CUDA device in cuda mode
- **WHEN** `--ground-truth-device cuda` is selected
- **AND** the target point set plus at least one source point cannot fit within the CUDA memory budget
- **THEN** the system exits with an error that indicates CUDA memory is insufficient

### Requirement: CUDA support is optional at build time
The system SHALL build and run without CUDA support while enabling CUDA acceleration when the build is configured with CUDA.

#### Scenario: Non-CUDA build accepts auto backend
- **WHEN** a binary built without CUDA support computes ground truth with `--ground-truth-device auto`
- **THEN** the system uses the CPU implementation

#### Scenario: Non-CUDA build rejects required CUDA backend
- **WHEN** a binary built without CUDA support computes ground truth with `--ground-truth-device cuda`
- **THEN** the system exits with an error explaining that CUDA support is not compiled in

#### Scenario: CUDA build uses the default CUDA device
- **WHEN** a binary built with CUDA support computes ground truth with `--ground-truth-device cuda`
- **THEN** the system uses the CUDA runtime default device
- **AND** does not require a command-line CUDA device id

