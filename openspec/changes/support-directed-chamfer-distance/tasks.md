## 1. Direction Model And Metadata

- [ ] 1.1 Add a shared Chamfer direction representation for `ab`, `ba`, and `both`.
- [ ] 1.2 Update dataset metadata parsing to require the structured `num_points` and `chamfer_distances` schema.
- [ ] 1.3 Add helpers to resolve point-set metadata and ground truth values for the selected norm and direction.

## 2. CLI And Command Wiring

- [ ] 2.1 Add `--direction ab|ba|both` to the `index` command with default `ab`.
- [ ] 2.2 Add `--direction ab|ba|both` to the `estimate` command with default `ab`.
- [ ] 2.3 Pass the selected direction into `IndexCommand` and `EstimateCommand`.

## 3. Index Behavior

- [ ] 3.1 Update `IndexCommand` to build only the target-side B index for `ab`.
- [ ] 3.2 Update `IndexCommand` to build only the target-side A index for `ba`.
- [ ] 3.3 Keep `IndexCommand` building both A and B indexes for `both`.

## 4. Estimate Behavior

- [ ] 4.1 Update `EstimateCommand` to compute only A-to-B for `ab`.
- [ ] 4.2 Update `EstimateCommand` to compute only B-to-A for `ba`.
- [ ] 4.3 Update `EstimateCommand` to compute A-to-B plus B-to-A for `both`.
- [ ] 4.4 Add missing-index validation for QALSH-backed estimation paths and report the required `index --direction` command.
- [ ] 4.5 Update estimate output to include direction, estimated distance, ground truth, relative error, time, and memory.

## 5. Data Fixtures And Scripts

- [ ] 5.1 Add `data/toy/A.bin`, `data/toy/B.bin`, and structured `data/toy/metadata.json` with known directional values.
- [ ] 5.2 Update dataset generation utilities to compute A-to-B and B-to-A Chamfer components separately.
- [ ] 5.3 Update generated metadata writing to emit the new structured schema.

## 6. Tests And Documentation

- [ ] 6.1 Add exact linear-scan CLI tests for toy `ab`, `ba`, and `both` estimates.
- [ ] 6.2 Add CLI help tests covering `--direction` on `index` and `estimate`.
- [ ] 6.3 Add index integration tests verifying `ab` creates only B, `ba` creates only A, and `both` creates both indexes.
- [ ] 6.4 Update README examples to show the toy dataset and direction selection without migrating Federalist.
- [ ] 6.5 Run the relevant CMake test suite and OpenSpec validation.
