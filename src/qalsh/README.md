# Internal QALSH module

This directory is owned and maintained by QALSH4C. Modify the index/search
implementation here and commit it with its consumers; no external QALSH
checkout, package installation, or separate release is required.

- `include/qalsh/qalsh.h`: index, search, strategy, and distance interfaces.
- `src/qalsh.cc`: implementation; `src/window_clamp.h`: private scan helper.
- `tests/`: existing algorithm, distance, persistence, dispatch, and SIMD/scalar
  regression coverage, registered in the parent CTest suite.
- `CMakeLists.txt`: internal `qalsh4c_qalsh` target. The `qalsh` C++ namespace
  and include paths remain to minimize changes to consumers.

## Provenance

Imported from [jonathanhu237/qalsh-lib](https://github.com/jonathanhu237/qalsh-lib),
commit `174866e2650264d02200f1b83cca8f2146cfbfc4`, on 2026-10-22.
Implementation headers/sources and tests were copied byte-for-byte. Build
integration was adapted to the parent project; standalone package exports,
installation, release automation, and external consumer scaffolding were not
imported. The source snapshot's MIT notice is preserved in [LICENSE](LICENSE).
This is a source-history reference, not an ongoing dependency.

The algorithm references are Huang et al., *Query-Aware Locality-Sensitive
Hashing for Approximate Nearest Neighbor Search* (PVLDB 2015), and
*Query-Aware Locality-Sensitive Hashing Scheme for l_p Norm* (VLDBJ 2017).
Author reference implementations are
[QALSH](https://github.com/HuangQiang/QALSH) and
[QALSH_Mem](https://github.com/HuangQiang/QALSH_Mem). This import comes from the
recorded qalsh-lib snapshot, not from those author repositories, and does not
establish algorithmic equivalence with them.

The import preserves the existing search behavior and persistent file format.
Existing QALSH4C indexes require no rebuild solely because of this move.
