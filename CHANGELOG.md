# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
