## Context

The estimator layer already models a directed Chamfer component with `EstimateDistance(from, to, norm_order)`: it sums nearest-neighbor distances from every point in `from` to the target set `to`. The `estimate` command currently hides that directionality by always computing `A -> B` and `B -> A`, then summing them. Dataset metadata also stores only one flat ground-truth value per norm, which represents the current bidirectional sum.

This change makes direction an explicit CLI concept shared by `index` and `estimate`. It intentionally introduces a breaking default: the default direction becomes `A -> B`, matching the estimator API and allowing users to request `B -> A` or bidirectional behavior when needed.

## Goals / Non-Goals

**Goals:**

- Support `--direction ab|ba|both` on `index` and `estimate`.
- Default both commands to `ab`.
- Keep bidirectional Chamfer defined as `ab + ba`.
- Use structured metadata that stores directional ground truth for each supported norm.
- Add a small `data/toy` dataset using the new metadata schema.
- Add CLI tests that verify directed estimates, bidirectional estimates, and direction-aware index creation.

**Non-Goals:**

- Migrating `data/federalist` to the new metadata schema.
- Supporting legacy flat metadata.
- Changing Chamfer distance from sums to means or averages.
- Making `estimate` implicitly build missing indexes.
- Falling back to linear scan when a QALSH-backed method is missing an index.

## Decisions

### Direction Model

Use a small explicit direction enum with values `ab`, `ba`, and `both`.

- `ab`: estimate `sum_{a in A} min_{b in B} d(a,b)`.
- `ba`: estimate `sum_{b in B} min_{a in A} d(b,a)`.
- `both`: estimate `ab + ba`.

Alternative considered: a boolean `--bidirectional` flag. This was rejected because it cannot express `B -> A`.

### CLI Placement

Add `--direction` only to the `index` and `estimate` parent commands.

Subcommands such as `ann`, `sampling`, `linear_scan`, and `qalsh` inherit the selected direction through the command object. This keeps direction a Chamfer-level concept rather than an algorithm-specific option.

### Direction-Aware Indexing

Interpret `index --direction` as the estimation direction, not the point set to index.

- `ab` builds only the `B` index because `A -> B` queries nearest neighbors in `B`.
- `ba` builds only the `A` index because `B -> A` queries nearest neighbors in `A`.
- `both` builds both indexes.

This reduces unnecessary index work while preserving an explicit way to prepare both directions.

### Missing Index Behavior

QALSH-backed estimation SHALL fail when the required target-side index is missing. It SHALL NOT build indexes implicitly or silently use another searcher. The error message should identify the missing point-set index and suggest the corresponding `index --direction` command.

### Metadata Schema

Use a structured metadata schema and require it for datasets used by the updated C++ code:

```json
{
  "num_dimensions": 2,
  "num_points": {
    "a": 2,
    "b": 2
  },
  "chamfer_distances": {
    "l1": {
      "ab": 2.0,
      "ba": 3.0
    },
    "l2": {
      "ab": 2.0,
      "ba": 3.0
    }
  }
}
```

Do not store `both`; derive it as `ab + ba` to avoid inconsistent redundant values.

Alternative considered: flat fields such as `chamfer_distance_l2_ab`. This was rejected because norm and direction are natural nested dimensions and the flat form will grow poorly.

### Output Format

`estimate` output should include:

- `Direction: A -> B`, `Direction: B -> A`, or `Direction: A <-> B`
- `Estimated Distance: <fixed 6 decimals>`
- `Ground Truth: <fixed 6 decimals>`
- `Relative Error: <fixed 2 decimals>%`
- existing timing and memory fields

Printing the distance and ground truth makes the selected direction observable and gives tests stable values to assert.

### Toy Fixture

Add `data/toy` with two 2D point sets:

- `A = [(0, 0), (2, 0)]`
- `B = [(1, 0), (4, 0)]`

For both L1 and L2:

- `ab = 2`
- `ba = 3`
- `both = 5`

The toy dataset is the source of truth for initial CLI tests and examples. `data/federalist` remains out of scope until a later migration.

## Risks / Trade-offs

- Breaking default behavior may surprise existing users -> Mitigate by documenting the new default and printing the selected direction in command output.
- Direction-aware indexing can be misread as "index A for ab" -> Mitigate with docs and error messages that explain target-side indexing.
- New metadata schema makes `data/federalist` temporarily unusable -> Mitigate by adding `data/toy` and explicitly leaving Federalist migration to a later change.
- Tests that depend on QALSH may be slower or sensitive to index state -> Mitigate by using `ann linear_scan` for exact estimate direction tests and separate lightweight index creation tests.
