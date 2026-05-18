## ADDED Requirements

### Requirement: QALSH estimates build target-side arrays at estimate time

The system SHALL build QALSH target-side projection state in memory during QALSH-backed `estimate` execution instead of reading a prebuilt disk index.

#### Scenario: A-to-B QALSH estimate builds only B-side state

- **WHEN** the user runs a QALSH-backed estimate with `--direction ab`
- **THEN** the system builds in-memory QALSH state only for point set B
- **AND** the system estimates the A-to-B directed Chamfer component using that B-side state

#### Scenario: B-to-A QALSH estimate builds only A-side state

- **WHEN** the user runs a QALSH-backed estimate with `--direction ba`
- **THEN** the system builds in-memory QALSH state only for point set A
- **AND** the system estimates the B-to-A directed Chamfer component using that A-side state

#### Scenario: Bidirectional QALSH estimate builds directions sequentially

- **WHEN** the user runs a QALSH-backed estimate with `--direction both`
- **THEN** the system estimates A-to-B with B-side in-memory QALSH state
- **AND** the system releases that directed state before estimating B-to-A with A-side in-memory QALSH state

### Requirement: In-memory QALSH search uses sorted projection arrays

The system SHALL represent each in-memory QALSH hash table as an ordered array of `(dot_product, point_id)` entries sorted by dot product and point id.

#### Scenario: Query initializes from the ordered array

- **WHEN** a QALSH search starts for a query point
- **THEN** the system computes the query dot product with each table's dot vector
- **AND** the system initializes left and right scan positions in the corresponding sorted projection array using lower-bound search

#### Scenario: Search expands from the lower-bound position

- **WHEN** QALSH search evaluates candidates for a radius bound
- **THEN** the system scans adjacent entries to the left and right while their dot products remain within the bound
- **AND** the system applies the existing collision threshold and candidate termination rules

### Requirement: Directed estimate reuses one dot-vector set

The system SHALL reuse the same generated dot vectors and sorted projection arrays for every query within one directed QALSH estimate.

#### Scenario: OpenMP clones share the directed QALSH state

- **WHEN** QALSH-backed estimation runs with OpenMP searcher clones
- **THEN** each clone shares the same immutable target-side points, dot vectors, config, and sorted projection arrays
- **AND** each clone maintains only its own query cursor state

#### Scenario: Reverse direction may use independent dot vectors

- **WHEN** the user runs `estimate --direction both`
- **THEN** the A-to-B direction and B-to-A direction may generate independent dot-vector sets

### Requirement: QALSH sampling supports explicit weight persistence

The system SHALL allow `sampling qalsh` users to save generated QALSH weights to a directory or load previously saved QALSH weights from a directory.

#### Scenario: Save QALSH weights

- **WHEN** the user runs `estimate ... sampling qalsh --save-weights <dir>`
- **THEN** the system generates weights using the directed in-memory QALSH state
- **AND** the system writes `<dir>/metadata.json` and `<dir>/weights.bin`

#### Scenario: Load QALSH weights

- **WHEN** the user runs `estimate ... sampling qalsh --load-weights <dir>`
- **THEN** the system reads weights from `<dir>/weights.bin`
- **AND** the system validates `<dir>/metadata.json` against the current direction, norm order, dataset point counts, dimensions, file identity, and QALSH parameters
- **AND** the system skips QALSH weight generation

#### Scenario: Save and load weights are mutually exclusive

- **WHEN** the user passes both `--save-weights` and `--load-weights` to `sampling qalsh`
- **THEN** the system exits with an error explaining that the options cannot be combined

#### Scenario: Weight persistence rejects bidirectional estimates

- **WHEN** the user passes `--direction both` with `sampling qalsh --save-weights <dir>` or `sampling qalsh --load-weights <dir>`
- **THEN** the system exits with an error explaining that QALSH weight persistence is only supported for a single directed estimate

### Requirement: QALSH sampling exposes approximation ratio

The system SHALL allow users to set the QALSH approximation ratio for `sampling qalsh` with `-c` or `--approximation-ratio`.

#### Scenario: Sampling QALSH uses explicit approximation ratio

- **WHEN** the user runs `estimate ... sampling qalsh -c 3.0`
- **THEN** the system regularizes QALSH config using approximation ratio `3.0`
- **AND** generated weight metadata records approximation ratio `3.0`
