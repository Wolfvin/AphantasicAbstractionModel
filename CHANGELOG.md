# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [7.0.1] - 2026-05-05

### Fixed

- **CRITICAL**: Added missing PyO3 bindings for `mcts_query`, `run_reflection`, `consolidate`, `set_thinking_mode`, `verify`, and `spreading_activation` — previously these would crash with AttributeError when called from Python
- **CRITICAL**: Fixed `deps.rs` `FailureType` not deriving `Hash` — prevented HashMap usage at compile time
- **CRITICAL**: Fixed non-exhaustive match patterns in `deps.rs` — `classify_error` and `explain_error` now cover all `RsvsError` variants
- Replaced `unwrap()` in `mcts.rs` `extract_path()` with safe `if let Some` pattern
- Wired `NeuroSymVerifier` into `compose()` pipeline — compositions are now verified for self-reference and circular chains before creation
- Wired `DEPSPlanner` into `compose()` pipeline — failed compositions now include DEPS recovery hints in error messages
- Added `detect_composition_cycle()` method to `Rsvs` for transitive cycle detection

### Added

- New PyO3 binding classes: `PyMCTSResult`, `PyConsolidationResult`, `PyReflectionResult`
- New PyO3 methods: `mcts_query()`, `run_reflection()`, `consolidate()`, `set_thinking_mode()`, `verify()`, `spreading_activation()`
- All v7.0 module classes registered in `_rsvs` Python module

### Changed

- Bindings version comment updated from v6.0 to v7.0
- 175 tests passing (up from 113 in v6.2.0)

## [7.0.0] - 2026-05-04

### Added

- **ParadigmRouter** — Adaptive traversal paradigm selection (Direct → Shallow → Standard → Deep → MCTS)
  - Routes queries to the lightest traversal strategy that will succeed
  - Confidence-based baseline routing with structural complexity adjustment
  - Per-domain calibration: learns which paradigms work for which domains
  - Inspired by Losion's ParadigmRouter (DIRECT → CoT → ReAct → RAG → MCTS)

- **SpreadingActivation** — Network activation through composition edges
  - Activates related nodes through structural meaning connections
  - Energy decays per hop (configurable decay factor, default 0.5)
  - Additive accumulation: multiple paths reinforce activation
  - Targeted spread with grounding-adjusted initial energy
  - Inspired by Losion's EpisodicMemory spreading activation

- **DEPSPlanner** — Structured failure recovery (Describe-Explain-Plan-Select)
  - DESCRIBE: Classify failure type (SelfReference, CircularChain, TargetNotFound, etc.)
  - EXPLAIN: Generate human-readable root cause analysis
  - PLAN: Generate multiple recovery plans with estimated success rates
  - SELECT: Choose best plan based on 60% success_rate + 40% simplicity
  - Inspired by Losion's DEPS failure recovery system

### Changed

- **jaccard_sets()** optimized from O(n×m) to O(n+m) using HashSet for the second set
  - Previously used `b.contains(x)` on Vec which is O(m) per call
  - Now converts to HashSet once for O(1) lookups
  - This is a hot path for attention scoring and sense assignment

- Version bumped to v7.0.0 (major: new public modules with API surface)
- Schema badge updated from v5.0 to v7.0 in README

### Fixed

- Schema version badge in README was showing v5.0 instead of current version

## [6.5.0] - 2026-05-03

### Added

- Bug fixes, O(1) composition index, persistence improvements, API completeness, 20 new tests

## [6.4.0] - 2026-05-02

### Added

- Losion Cross-Pollination P0-P3 full implementation
- ThinkingToggle, CompositionIndex, ConsolidationEngine, SenseReflection
- MCTS traversal, Matryoshka traversal, Neuro-symbolic verification
- Ebbinghaus forgetting curve for sense lifecycle

## [4.2.0] - 2025-05-03

### Added

- v4.2 unified node model — eliminated Atom/Composite dualism, single `kind=node` with `CompressionState`
- Hard attention engine: `score = α·NPMI + β·Jaccard + γ·cooc` (sparse, deterministic, interpretable)
- Multi-sense disambiguation with incremental coherence O(n), fragile/mature lifecycle
- Autonomy engine: confidence EMA, tier promotion/demotion with hysteresis, global stability gate with rollback
- Policy engine: single owner governance, hysteresis anti-flip-flop, source trust weighting, quarantine, dedup gate
- Seed atom bootstrap: 24 primitive immutable atoms as axiomatic foundation
- PyO3 bindings for Rust↔Python bridge
- FastAPI server with OpenAPI docs (replaces BaseHTTPRequestHandler)
- 3D knowledge graph visualization with React Three Fiber
- Appraise and Relate mode UI panels
- Event timeline with play/pause/speed controls
- Comprehensive CLI with 11 subcommands
- Evaluation suite with 5 benchmarks
- CI/CD pipeline via GitHub Actions
- Docker and docker-compose support

### Changed

- Migrated Python server from BaseHTTPRequestHandler to FastAPI
- Proper Rust error types with `thiserror` replacing String errors
- Thread-safe Python singleton for Rust core access
- Frontend package name from generic template to `@rsvs/frontend`
- Strict TypeScript mode enabled (`noImplicitAny: true`)

### Fixed

- CLI version mismatch (0.8.0 → 4.2.0)
- Broken tempfile dependency in pyproject.toml
- Hardcoded personal paths in .env.example
- Mock data leaking into production code paths

## [4.1.0] - 2025-03-15

### Added

- Multi-sense framework with fragile/mature sense lifecycle
- Incremental coherence calculation O(n) per context
- Sense merge algorithm with Jaccard threshold
- Fragile sense pruning after inactivity timeout

### Changed

- Confidence update now uses EMA (η=0.10) with energy constraint
- Hysteresis gap widened to 0.15 (promote ≥ 0.75, demote < 0.60)

### Fixed

- Race condition in concurrent confidence updates
- Memory leak in sense manager during high-volume ingest

## [4.0.0] - 2025-01-20

### Added

- Rust core engine with PyO3 Python bindings
- Hard attention scoring (NPMI + Jaccard + Co-occurrence)
- Seed atom bootstrap (24 primitive atoms)
- Node lifecycle state machine (New → Candidate → Stable → Deprecated → Quarantine)
- Python bridge server with HTTP API
- Event stream with runtime correlation IDs
- Snapshot persistence with JSON serialization

### Changed

- **Breaking**: Unified node model replaces Atom/Composite dualism
- **Breaking**: Schema version bumped to v4.2 (no backward compatibility)
- Hard break policy for pre-v4.2 payloads

### Removed

- Legacy Python-only backend (all computation now in Rust)
- `_legacy_*` fallback functions from bridge server
- `render` metadata generation from bridge (moved to frontend)

## [3.0.0] - 2024-09-10

### Added

- Composite node model with render metadata
- Appraise mode for evaluating text against graph
- Relate mode for finding related concepts
- Artifact persistence (snapshots, events, reports)

### Changed

- Python server migrated from Flask to BaseHTTPRequestHandler
- Improved tokenization with stopword filtering

## [2.0.0] - 2024-05-15

### Added

- Atom promotion from text ingestion
- Co-occurrence statistics tracking
- Jaccard similarity for node comparison
- Basic web UI for graph visualization

## [1.0.0] - 2024-01-01

### Added

- Initial release with basic knowledge graph
- Simple text ingestion pipeline
- HTTP API with /run endpoint
- React frontend with D3.js visualization

[4.2.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v4.2.0
[4.1.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v4.1.0
[4.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v4.0.0
[3.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v3.0.0
[2.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v2.0.0
[1.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v1.0.0
