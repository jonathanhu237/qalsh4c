## Context

The repository currently has a C++ CLI for indexing and estimating Chamfer distance, but dataset preparation lives under `scripts/` in Python. Those scripts generate synthetic GMM data, convert several known raw datasets, write `A.bin`/`B.bin`, compute directed L1/L2 ground truth, and emit `metadata.json`.

The current C++ dataset format is the canonical format for this change: binary point sets are row-major `double` values, and metadata uses `num_dimensions`, `num_points.a/b`, and `chamfer_distances.l1/l2.ab/ba`. The migration does not need bit-for-bit compatibility with the Python implementation, and the existing `data/toy` schema is the target shape.

## Goals / Non-Goals

**Goals:**

- Make dataset generation, conversion, ground-truth calculation, and metadata writing available through the C++ CLI.
- Remove `scripts/` after C++ commands cover the current generation and conversion workflows.
- Support GMM synthetic data, MNIST local IDX images, P53 `K9.data`, SIFT/GIST local fvecs, and Trevi BMP patches.
- Allow datasets to be created without ground truth and backfilled later.
- Represent missing ground truth explicitly with `null` metadata values.
- Keep stdout suitable for command results while progress and logs go to stderr.
- Use OpenMP for CPU exact ground-truth parallelism when available.

**Non-Goals:**

- Bit-for-bit compatibility with Python, NumPy, scikit-learn, or PyTorch output.
- Automatic dataset downloads, including MNIST OpenML downloads.
- CUDA acceleration in this change.
- Full image decoding support or OpenCV dependency for Trevi.
- Streaming or external-shuffle import for very large raw datasets.
- Storing a separate bidirectional ground-truth field in metadata.

## Decisions

### Dataset CLI Shape

Add a top-level `dataset` command group with:

- `dataset generate gmm`
- `dataset convert mnist`
- `dataset convert p53`
- `dataset convert sift`
- `dataset convert gist`
- `dataset convert trevi`
- `dataset ground-truth`

Use dataset names as user-facing subcommands because they match current workflows. Internally, share format-specific implementations such as fvecs loading for SIFT/GIST and BMP patch extraction for Trevi.

Alternative considered: expose format names such as `fvecs` and `idx` directly. That would reduce CLI surface area but make common user workflows less obvious.

### Ground-Truth Selection

Use one optional `--ground-truth` argument containing a comma-separated list of exact directed tasks:

```text
l1:ab,l1:ba,l2:ab,l2:ba
```

Only `l1:ab`, `l1:ba`, `l2:ab`, and `l2:ba` are valid. Do not accept `both`; users express both directions by listing `ab` and `ba`. Deduplicate repeated tasks after parsing.

Generation and conversion do not compute ground truth unless `--ground-truth` is provided. `dataset ground-truth` accepts the same selector and updates existing metadata.

Alternative considered: repeatable `--ground-truth` options or accepting `both`. Both were rejected to keep the CLI small and aligned with metadata fields.

### Metadata With Optional Ground Truth

Keep the existing nested metadata schema but allow each ground-truth value to be a number or `null`:

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

Use `std::optional<double>` in the C++ metadata model. For `estimate --direction both`, ground truth is available only if both directed fields for the selected norm are present.

Alternative considered: omit missing fields. Explicit `null` was chosen because it keeps schema shape stable and makes missing values intentional.

### Estimate Behavior With Missing Ground Truth

When the requested ground truth is present, keep the current estimate output semantics. When it is unavailable, continue estimation and print:

```text
Ground Truth: unavailable
Relative Error: unavailable
```

This preserves the ability to use datasets that were generated or converted without exact ground truth.

### Dataset Sources

Implement first-stage sources as full in-memory loaders/generators:

- GMM: generate two independent two-component mixtures for A and B. Preserve the Python-level semantics of core/outlier weights, uniform means in `[-512, 512]`, covariance `R * R^T + I`, and outlier covariance scaling. Use Eigen Cholesky/LLT to sample multivariate normal points.
- MNIST: load local IDX image files only, flatten images as double vectors, and ignore labels.
- P53: parse local `K9.data`, skip blank rows and rows containing `?`, and drop the last two CSV columns as the Python loader does.
- SIFT/GIST: load local `.fvecs` files and combine the expected base/learn/query files for each dataset.
- Trevi: read local BMP files with a minimal uncompressed BMP subset and emit non-overlapping 64x64 grayscale patches as 4096-dimensional double vectors.

Full in-memory loading keeps the first implementation simple and close to current Python behavior. Streaming import and external shuffle can be introduced later if real datasets exceed memory limits.

### Split, Shuffle, And Reproducibility

Conversion defaults to shuffling all loaded points, splitting A/B 50/50 when `--num-points-a` is absent, and using the remainder for B when `--num-points-a` is provided. Add `--no-shuffle` to preserve source order. Add explicit `--seed` to generation and conversion, and let it control dataset output reproducibility independently of `--use-fixed-seed`.

### Output Overwrite Rules

For `dataset generate` and `dataset convert`:

- Create the output directory when absent.
- Fail by default if `A.bin`, `B.bin`, or `metadata.json` already exists.
- With `--overwrite`, replace those files and remove any existing `index/` directory because indexes become invalid when point data changes.

For `dataset ground-truth`:

- Write requested fields only when the existing metadata value is `null`.
- Fail by default when a requested field already has a value.
- With `--overwrite`, recompute and replace requested existing values.

### Exact Ground-Truth Computation

Compute exact directed Chamfer with CPU linear scan:

```text
sum over each source point of min distance to all target points
```

Parallelize over source points under `QALSH4C_USE_OPENMP` and reduce the directed sum. The first implementation keeps CPU as the only backend but should keep parsing and code organization open to a later CUDA backend change.

### Progress Reporting

Add a small internal progress reporter instead of a third-party progress library. It writes to stderr, keeps stdout for final command results, dynamically refreshes a single line when stderr is a TTY, and falls back to low-frequency log lines in non-interactive environments. Add `--no-progress`. Show percent, current, total, rate, and elapsed time; do not show ETA in the first version.

For OpenMP ground-truth work, worker threads update a shared completed counter and progress is refreshed at low frequency to avoid stderr contention.

### Documentation And Script Removal

Update README examples to use the new C++ dataset commands. Remove `scripts/` in the same change after tests cover the replacement workflows.

## Risks / Trade-offs

- Large raw datasets may use substantial memory because loaders are in-memory only -> Document the limitation and leave streaming import for a later change.
- CPU exact ground truth can run for a long time -> Require explicit `--ground-truth`, show progress, and parallelize with OpenMP when available.
- Minimal BMP support may reject valid BMP variants -> Document supported Trevi BMP subset and fail with clear errors for unsupported encodings.
- Removing Python scripts can break ad hoc workflows not captured in README -> Cover all known script capabilities with C++ commands and tests before deletion.
- Partial metadata changes an existing assumption that all ground truth fields are numeric -> Use `std::optional<double>` and add CLI tests for unavailable ground truth.
- OpenMP progress updates can become noisy or expensive -> Decouple computation from progress rendering through shared counters and throttled refreshes.
