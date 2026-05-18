## REMOVED Requirements

### Requirement: Index command supports direction-aware indexing

**Reason**: QALSH-backed estimates now build the selected direction's target-side projection arrays in memory during `estimate`; users no longer prepare target indexes with a separate `index` command.

**Migration**: Run the desired `estimate --direction ab|ba|both` command directly. For repeated `sampling qalsh` experiments, use `--save-weights <dir>` once and `--load-weights <dir>` in later runs.

### Requirement: QALSH-backed estimation requires the selected direction index

**Reason**: QALSH-backed estimation no longer reads disk-backed direction indexes. Missing disk indexes are not an error because the required target-side QALSH state is built in memory for each directed estimate.

**Migration**: Remove pre-estimation `index` steps from scripts. Use `estimate ann qalsh` or `estimate sampling qalsh` directly.
