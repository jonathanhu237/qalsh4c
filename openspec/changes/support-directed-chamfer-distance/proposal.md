## Why

The CLI currently hides directionality: the estimator API computes one directed Chamfer component, while the `estimate` command always sums `A -> B` and `B -> A`. Users need explicit control over directed and bidirectional Chamfer estimation, with metadata and indexing behavior that match the selected direction.

## What Changes

- **BREAKING**: Change the default Chamfer direction from bidirectional to `A -> B`.
- Add `--direction ab|ba|both` to `estimate`, with `ab` as the default.
- Add `--direction ab|ba|both` to `index`, with `ab` as the default.
- Make `index --direction ab` build only the target-side `B` index, `ba` build only the target-side `A` index, and `both` build both indexes.
- Report the selected direction, estimated distance, ground truth, and relative error in estimate output.
- Replace flat metadata with a structured schema containing directional ground truth for `l1` and `l2`.
- Add a small `data/toy` fixture using the new metadata schema.
- Update data-generation scripts to emit directional metadata.
- Add CLI and integration tests for direction behavior, metadata parsing, output values, and direction-aware index creation.

## Capabilities

### New Capabilities

- `directed-chamfer-distance`: Directed and bidirectional Chamfer distance selection for indexing, estimation, metadata, and CLI output.

### Modified Capabilities

- None.

## Impact

- CLI: `index` and `estimate` gain a shared `--direction` option and default to `ab`.
- C++ metadata parsing changes to the new structured schema only; legacy metadata is intentionally not supported in this change.
- Estimation output format changes to include direction and distance values.
- Indexing becomes direction-aware and may create only one point-set index.
- Scripts that create datasets must compute and write `ab` and `ba` ground truth values separately.
- Tests and examples should use `data/toy`; `data/federalist` migration is out of scope.
