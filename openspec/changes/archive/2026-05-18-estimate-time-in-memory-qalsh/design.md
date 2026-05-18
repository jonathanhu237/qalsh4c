## Context

The current QALSH path has a two-step workflow: users run `index` to write target-side QALSH config, random dot vectors, and one B+ tree per hash table to disk, then `estimate` reads those files. `QalshAnnSearcher` is therefore coupled to disk paths and `BPlusTreeSearcher`, while `EstimateCommand` rejects QALSH-backed estimates when the selected direction's target index is missing.

The desired workflow is estimate-time indexing. A QALSH estimate should construct only the selected direction's target-side state in memory, use it for that directed estimate, and release it before building any reverse direction state. The in-memory structure is conceptually simpler than a B+ tree: each hash table is just a sorted array of `(dot_product, point_id)` pairs.

## Goals / Non-Goals

**Goals:**

- Remove the user-facing need to run `index` before QALSH-backed estimates.
- Build the selected direction's target-side QALSH state in memory during `estimate`.
- Keep `ab`, `ba`, and `both` memory use direction-aware; `both` runs the two directions sequentially.
- Share one target-side memory index across queries, OpenMP searcher clones, and QALSH weight generation for a directed estimate.
- Preserve QALSH sampling semantics: QALSH estimates weights, then sampled distances are computed with exact linear scan.
- Add explicit QALSH sampling weight save/load directories for repeated experiments.

**Non-Goals:**

- Persisting ANN QALSH indexes or sorted projection arrays.
- Replacing QALSH with linear scan.
- Loading both A-side and B-side QALSH state at once for `--direction both`.
- Removing the B+ tree source files in the same change unless they block the build after the CLI path is removed.
- Changing Chamfer sums to averages.

## Decisions

### In-Memory QALSH Index Shape

Introduce a target-side in-memory QALSH structure containing:

- `QalshConfig config`
- `std::vector<Point> points`
- `std::vector<Point> dot_vectors`
- `std::vector<std::vector<DotProductPointIdPair>> tables`

Each table is sorted with the existing `DotProductPointIdPairLess`. Search initializes per-query cursors with `std::ranges::lower_bound` and expands left/right while the dot product remains within the current bound.

Alternative considered: keeping a tree abstraction in memory. This was rejected because the required operation is ordered-array lower-bound plus adjacent scanning; a tree adds complexity without disk I/O benefits.

### Direction-Scoped Construction

Construct the in-memory index inside each directed `EstimateDistance(from, to, norm_order)` call, using only `to` as the target side. `EstimateCommand` already invokes the estimator separately for `ab` and `ba`; this naturally keeps `both` sequential and avoids holding both target indexes at once.

Alternative considered: pre-building both point-set indexes before estimation. This was rejected because single-direction estimates must not pay reverse-direction memory or construction cost.

### Dot Vector Reuse

Dot vectors are part of the in-memory index. Every query and every searcher clone for a directed estimate MUST reuse the same dot vectors and sorted tables. Different directions may generate independent dot vectors because their target-side indexes are independent.

### OpenMP Sharing

Make `QalshAnnSearcher::Clone()` cheap by sharing immutable target-side index state through `std::shared_ptr<const InMemoryQalshIndex>`. Each clone owns only mutable cursor/search state. This avoids rebuilding the same target-side sorted arrays per thread.

### QALSH Config Ownership

`QalshAnnSearcher` and `QalshWeightsGenerator` should accept the QALSH approximation ratio at construction. They regularize `QalshConfig` from the target point count and norm order during in-memory index construction. Sampling's automatic sample-count calculation reads the config from the generated or loaded weights context instead of `config.json`.

### Sampling Weight Persistence

Replace implicit cache reuse with explicit `sampling qalsh --save-weights <dir>` and `sampling qalsh --load-weights <dir>`.

The directory format is:

```text
<dir>/
  metadata.json
  weights.bin
```

`metadata.json` records a format identifier, direction, norm order, source/target point counts, dimensions, file size and modification time, and the regularized `QalshConfig`. `weights.bin` stores one `double` per source point. Loading validates the current command and dataset metadata against `metadata.json`, then skips QALSH weight generation and uses the stored config for automatic sample-count calculation.

`--save-weights` and `--load-weights` are mutually exclusive. They apply only to `sampling qalsh`; uniform sampling does not use persisted weights. Weight persistence is supported only for `--direction ab` or `--direction ba`; `--direction both` must be rejected because one directory contains one directed weight vector and one directed QALSH config.

### CLI Compatibility

Remove the active `index` command path from the primary CLI. Tests and documentation should stop teaching users to run `index`. B+ tree implementation files can remain temporarily as unused code while the new QALSH path stabilizes.

## Risks / Trade-offs

- Higher estimate-time startup cost -> Mitigate by making the cost explicit in estimate timing and allowing sampling weights to be saved for repeated experiments.
- Higher peak memory than disk-backed B+ tree search -> Mitigate by building only the selected direction's target-side index and releasing it before any reverse direction.
- Saved weights can be misapplied to a different dataset or parameter set -> Mitigate with metadata validation and hard errors on mismatch.
- Existing scripts that call `qalsh4c index` will break -> Mitigate by documenting the breaking change and updating README examples to call `estimate` directly.
- Random dot vectors move from a separate index command into estimate-time construction -> Mitigate by keeping dot vectors inside the shared in-memory index for the full directed estimate and preserving `--use-fixed-seed` for reproducible runs.
