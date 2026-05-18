## Why

QALSH estimation currently requires a separate disk-backed `index` workflow built around B+ tree files, which makes one-off experiments cumbersome and keeps large transient data on disk. This change makes QALSH estimation self-contained by building the required target-side ordered projection arrays in memory during `estimate`.

## What Changes

- **BREAKING**: Remove the user-facing requirement to run `qalsh4c index` before QALSH-backed estimation.
- **BREAKING**: Stop validating or reading disk-backed `data/<dataset>/index/...` QALSH indexes during `estimate`.
- Build the selected direction's target-side QALSH projection arrays in memory during `estimate`, using sorted arrays instead of B+ trees.
- Preserve direction-aware memory use: `ab` builds only B-side state, `ba` builds only A-side state, and `both` builds and releases each direction sequentially.
- Reuse each direction's generated dot vectors for all queries, OpenMP clones, and QALSH weight generation within that direction.
- Add explicit `sampling qalsh --save-weights <dir>` and `sampling qalsh --load-weights <dir>` workflows for long-running repeated sampling experiments.
- Add `sampling qalsh -c/--approximation-ratio` so saved weights record the QALSH approximation ratio used to generate them.

## Capabilities

### New Capabilities
- `qalsh-estimate-time-indexing`: QALSH estimation builds target-side in-memory ordered projection arrays during `estimate` and supports explicit QALSH weight save/load directories for sampling experiments.

### Modified Capabilities
- `directed-chamfer-distance`: QALSH-backed estimation no longer requires a prebuilt direction-specific disk index, and the `index` direction workflow is removed from the primary user flow.

## Impact

- CLI: remove or deprecate the `index` command path; update `estimate ann qalsh` and `estimate sampling qalsh` behavior; add `sampling qalsh` weight save/load flags.
- Core implementation: replace B+ tree searcher dependency in QALSH search with shared in-memory sorted projection arrays and per-searcher cursor state.
- Sampling: replace implicit `--use-cache` QALSH weight cache with explicit save/load directories and metadata validation.
- Tests and docs: update direction/index tests, QALSH ANN tests, sampling tests, and README usage examples.
