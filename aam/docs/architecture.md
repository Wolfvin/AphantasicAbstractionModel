# Architecture

RSVS follows a three-tier architecture with strict separation of concerns. This page provides a high-level overview. For the full technical reference, see [ARCHITECTURE.md](https://github.com/Wolfvin/AphantasicAbstractionModel/blob/main/ARCHITECTURE.md) in the repository root.

---

## Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Frontend (Optional / Demo)                  │
│           Next.js 16 · React Three Fiber · shadcn/ui        │
├─────────────────────────────────────────────────────────────┤
│              Python Bridge (HTTP + Validation)               │
│            FastAPI · PyO3 FFI · CLI · Artifacts              │
├─────────────────────────────────────────────────────────────┤
│                Rust Core (All Computation)                   │
│      Graph · Attention · Autonomy · Sense · MCTS · Persist  │
└─────────────────────────────────────────────────────────────┘
```

### Rust Core

**Location**: `backend/crates/rsvs-core/src/`

All computational logic lives here — graph storage, attention scoring, sense management, autonomy lifecycle, pipeline orchestration, MCTS, consolidation, reflection, and persistence. The core has:

- Zero HTTP dependencies
- Zero file I/O
- Zero Python dependencies
- Compiles independently as a pure Rust crate
- Exposes 22+ modules with a clean Rust API

The Rust core is exposed to Python via PyO3 bindings (feature-gated behind `#[cfg(feature = "python")]`). When compiled as a Python extension via maturin, the bindings produce the `rsvs._rsvs` native module with 30+ Python-visible classes and methods.

### Python Bridge

**Location**: `python/rsvs/`

The Python layer provides the user-facing API. It adds no computational logic — all computation happens in the Rust core. The bridge provides:

- **PyO3 bindings**: The `rsvs._rsvs` native extension compiled from Rust
- **FastAPI server**: Optional HTTP API with auth, rate limiting, CORS
- **CLI tool**: 11 subcommands for command-line access
- **Validation**: Schema validation for API payloads
- **Artifact persistence**: JSON snapshots, JSONL event logs, reports
- **Type stubs**: `.pyi` files for IDE support (PEP 561)

### Frontend

**Location**: `frontend/`

An optional demo/visualization layer. It is not required for RSVS operation — the system is fully functional via the Python CLI or API alone. The frontend provides:

- 3D graph visualization with React Three Fiber
- Force-directed layout with physics-based positioning
- Interactive query, compose, appraise, and relate panels
- Event timeline for temporal navigation

---

## Key Design Decisions

### Compositional Senses, Not Embeddings

Every concept in RSVS is defined as a composition of other concepts. This means meaning is always traceable — you can follow the chain from any concept down to its constituent parts. Unlike embeddings, which compress meaning into opaque vectors, RSVS representations are human-readable and structurally precise.

### 24 Seed Atoms

The system bootstraps from 24 epistemological seed atoms (exists, entity, relation, state, change, time, space, cause, effect, context, signal, pattern, memory, attention, value, agent, goal, risk, trust, identity, language, meaning, action, feedback). These provide the axiomatic foundation for all subsequent compositions.

### Language-Agnostic Architecture

RSVS does not require linguistic metadata. The convergence engine detects when nodes from different languages have structurally equivalent compositions and creates bidirectional `LanguageLink` records automatically. This means the system works for any language without configuration.

### Autonomous Tiered Memory

Every node progresses through a lifecycle: New → Candidate → Stable → Deprecated. This lifecycle is managed by the autonomy engine using EMA confidence updates with hysteresis thresholds. The system self-corrects through consolidation (merge similar senses, prune weak edges) and reflection (CONFIRM/REVIEW/REVISE/RETIRE actions).

---

## Build and Distribution

RSVS uses maturin to build Python wheels from the Rust core. The build process:

1. Compile the Rust core with PyO3 bindings
2. Package as a Python wheel with `rsvs._rsvs` native extension
3. Include Python bridge code, type stubs, and CLI
4. Publish to PyPI for `pip install rsvs`

The wheel includes the compiled Rust code — no Rust toolchain needed at install time. The core library has zero Python dependencies.
