# directed-chamfer-distance Specification

## Purpose
TBD - created by archiving change support-directed-chamfer-distance. Update Purpose after archive.
## Requirements
### Requirement: Estimate command supports directed Chamfer selection

The system SHALL allow users to select the Chamfer estimation direction with `--direction ab|ba|both` on the `estimate` command, defaulting to `ab`.

#### Scenario: Default estimate direction is A to B

- **WHEN** the user runs `estimate` without `--direction`
- **THEN** the system estimates the directed Chamfer component from point set A to point set B

#### Scenario: Estimate direction B to A

- **WHEN** the user runs `estimate --direction ba`
- **THEN** the system estimates the directed Chamfer component from point set B to point set A

#### Scenario: Estimate direction both

- **WHEN** the user runs `estimate --direction both`
- **THEN** the system estimates bidirectional Chamfer distance as the sum of the A-to-B and B-to-A directed components

### Requirement: Index command supports direction-aware indexing

The system SHALL allow users to select the estimation direction prepared by `index` with `--direction ab|ba|both`, defaulting to `ab`.

#### Scenario: Index for A to B

- **WHEN** the user runs `index --direction ab`
- **THEN** the system builds only the B point-set index required for A-to-B estimation

#### Scenario: Index for B to A

- **WHEN** the user runs `index --direction ba`
- **THEN** the system builds only the A point-set index required for B-to-A estimation

#### Scenario: Index for both directions

- **WHEN** the user runs `index --direction both`
- **THEN** the system builds both A and B point-set indexes

### Requirement: QALSH-backed estimation requires the selected direction index

The system SHALL fail QALSH-backed estimation when the index required by the selected direction is missing.

#### Scenario: Required target index is missing

- **WHEN** the user runs a QALSH-backed estimate for a direction whose target-side index does not exist
- **THEN** the system exits with an error that identifies the missing index and suggests the corresponding `index --direction` command

### Requirement: Metadata stores directional ground truth

The system SHALL read dataset metadata from a structured schema that stores point counts and optional directional ground truth values for each supported norm.

#### Scenario: Read directional ground truth

- **WHEN** the user runs `estimate --direction ab` with norm order 2 and `chamfer_distances.l2.ab` is numeric
- **THEN** the system reads `chamfer_distances.l2.ab` as the ground truth value

#### Scenario: Read reverse directional ground truth

- **WHEN** the user runs `estimate --direction ba` with norm order 1 and `chamfer_distances.l1.ba` is numeric
- **THEN** the system reads `chamfer_distances.l1.ba` as the ground truth value

#### Scenario: Derive bidirectional ground truth

- **WHEN** the user runs `estimate --direction both` and both matching `ab` and `ba` metadata values are numeric
- **THEN** the system derives ground truth as the sum of the matching `ab` and `ba` metadata values

#### Scenario: Missing directed ground truth is unavailable

- **WHEN** the user runs `estimate --direction ab` with norm order 2 and `chamfer_distances.l2.ab` is `null`
- **THEN** the system treats the ground truth as unavailable

#### Scenario: Missing bidirectional component makes bidirectional ground truth unavailable

- **WHEN** the user runs `estimate --direction both` and either matching directed metadata value is `null`
- **THEN** the system treats the bidirectional ground truth as unavailable

### Requirement: Estimate output reports direction and distance values

The system SHALL include the selected direction, estimated distance, ground truth availability, relative error availability, elapsed time, and memory usage in estimate output.

#### Scenario: Output for A-to-B estimate

- **WHEN** the user runs `estimate --direction ab` and the selected ground truth is numeric
- **THEN** the output includes `Direction: A -> B`, `Estimated Distance`, `Ground Truth`, and `Relative Error`

#### Scenario: Output for bidirectional estimate

- **WHEN** the user runs `estimate --direction both` and the selected ground truth is numeric
- **THEN** the output includes `Direction: A <-> B`, `Estimated Distance`, `Ground Truth`, and `Relative Error`

#### Scenario: Output when ground truth is unavailable

- **WHEN** the user runs `estimate` for a norm and direction whose metadata ground truth is `null`
- **THEN** the output includes `Estimated Distance`
- **AND** the output includes `Ground Truth: unavailable`
- **AND** the output includes `Relative Error: unavailable`

### Requirement: Data generation emits directional metadata

Dataset preparation utilities SHALL write separate A-to-B and B-to-A Chamfer ground truth fields for L1 and L2, using `null` for fields that were not computed.

#### Scenario: Generated metadata contains directional values

- **WHEN** a dataset is generated by the C++ dataset CLI with all four ground-truth selectors requested
- **THEN** metadata contains numeric values for `chamfer_distances.l1.ab`, `chamfer_distances.l1.ba`, `chamfer_distances.l2.ab`, and `chamfer_distances.l2.ba`

#### Scenario: Generated metadata contains null for uncomputed directional values

- **WHEN** a dataset is generated by the C++ dataset CLI without requesting some ground-truth selectors
- **THEN** metadata contains `null` for each uncomputed `chamfer_distances.l1.ab`, `chamfer_distances.l1.ba`, `chamfer_distances.l2.ab`, or `chamfer_distances.l2.ba` field

### Requirement: GMM sample dataset exercises the current schema

The repository SHALL include a committed `data/gmm` dataset using the current structured metadata schema.

#### Scenario: GMM sample has known directional distances

- **WHEN** tests run exact linear-scan estimates against `data/gmm`
- **THEN** the estimated directed Chamfer values match the corresponding committed metadata ground-truth values
