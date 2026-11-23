# Internal QALSH module

This directory is owned and maintained by QALSH4C. It holds only the QALSH code
that the paper's experiments use: an in-memory index and one search loop.
Modify it here and commit it with its consumers; no external QALSH checkout,
package installation, or separate release is required.

- `include/qalsh/qalsh.h`: the public interface (index, search options,
  stopping-rule settings, results).
- `src/qalsh.cc`: implementation; `src/window_clamp.h`: private scan helper
  that clamps a sorted run of table entries to the current window.
- `tests/`: build, search, distance-kernel and SIMD/scalar window regression
  tests, registered in the parent CTest suite.
- `CMakeLists.txt`: internal `qalsh4c_qalsh` target. The `qalsh` C++ namespace
  and include paths remain to minimize changes to consumers.

## What the module does

`InMemoryIndex::Build` projects every point onto `m` random vectors (Gaussian
for L2, Cauchy for L1) and keeps one table per vector, sorted by projection,
with a separate id column for collision counting. Workers project the points
in parallel when the accessor allows concurrent calls
(`IndexConfig::concurrent_accessor`); the tables do not depend on the number
of workers. Two optional parts are written during the build:

- `compressed_filter`: a bfloat16 copy of the points. Before an exact bounded
  distance, the search computes a rigorous lower bound from the copy and skips
  the candidate only when that bound proves the exact distance would reject
  it, so results are unchanged.
- `point_projections`: every point's projections, point-major, which the
  projection-distance scan order needs.

`SearchEngine::search` runs the collision-count QALSH search: a point becomes
a candidate when it collides in `l` tables (one-byte counters, so `l <= 255`),
and each candidate's bounded distance to the query is computed at once.
`SearchOptions::scan_order` chooses how the window entries of a radius round
are visited:

- `quanta`: tables in turn, `scan_quantum` entries per visit (half on each
  side of the query's projection); needs `scan_quantum >= 2`.
- `projection_distance`: all tables in increasing distance from the query's
  projection, counted in `slabs_per_round` slabs per round, with candidates
  evaluated in increasing collision radius (ties by id). This is the paper's
  default.
- `projection_distance_reference`: a plain one-distance-at-a-time version of
  the same order, used by the tests as a reference.

`DefaultQalshStrategy` holds the stopping rules: the candidate budget, the
test `d_k <= c R` after each evaluation, at the end of each round and right
after the radius grows (each switchable), a stop inside a round once a
candidate lies within `ratio * R`, and the continuous stop of the
projection-distance orders (`d_k <= c` times the radius of the current front).
A search ends with `TerminationReason::strategy` when a rule holds and with
`scan_exhausted` when every table runs out of entries.

## Provenance

Imported from [jonathanhu237/qalsh-lib](https://github.com/jonathanhu237/qalsh-lib),
commit `174866e2650264d02200f1b83cca8f2146cfbfc4`, on 2026-10-22, and then
reduced to the parts above. The persistent B+ tree and sorted-array indexes,
the general strategy-driven search engine with its cursor API, and the
stand-alone distance functions of that snapshot were removed. The source
snapshot's MIT notice is preserved in [LICENSE](LICENSE). This is a
source-history reference, not an ongoing dependency.

The algorithm references are Huang et al., *Query-Aware Locality-Sensitive
Hashing for Approximate Nearest Neighbor Search* (PVLDB 2015), and
*Query-Aware Locality-Sensitive Hashing Scheme for l_p Norm* (VLDBJ 2017).
Author reference implementations are
[QALSH](https://github.com/HuangQiang/QALSH) and
[QALSH_Mem](https://github.com/HuangQiang/QALSH_Mem). This import comes from the
recorded qalsh-lib snapshot, not from those author repositories, and does not
establish algorithmic equivalence with them.
