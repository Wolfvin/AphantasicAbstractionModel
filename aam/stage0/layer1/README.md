# RSVS — Recursive Symbolic Vector Space

Hybrid monorepo for RSVS:
- Rust core engine (`crates/rsvs-core`)
- Python package + CLI (`python/rsvs`)
- Python tests (`python/tests`)

## Repository Layout

- `crates/rsvs-core/src/`
  - core semantics: `types`, `graph`, `seed`, `sense`, `attention`, `autonomy`, `pipeline`
  - adapter/boundary: `persist`, `bindings`
  - smoke binary: `src/bin/rsvs-smoke.rs`
- `python/rsvs/`
  - package init, CLI, corpus ingest/eval helpers
- `python/tests/`
  - API/CLI/persistence/wiki ingestion tests
- `docs/`
  - architecture + workflow + archived skill artifacts

## Build and Test

```bash
make check-rust
make test-rust
make test-python
make test-all
```

## Python Package

`pyproject.toml` uses maturin with:
- `manifest-path = "crates/rsvs-core/Cargo.toml"`
- `python-source = "python"`
- module entry: `rsvs._rsvs`

## Compatibility Note (PyO3)

Current crate uses `pyo3 = 0.22`.
For Python versions newer than PyO3 max supported version, run with:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
```

This is enforced in CI and Makefile tasks.

## Attention Config File (v4.1 style)

You can override attention weights via JSON file and environment variable:

```bash
export RSVS_ATTENTION_CONFIG=/abs/path/attention.json
```

Example `attention.json`:

```json
{
  "alpha": 0.45,
  "beta": 0.35,
  "gamma": 0.20,
  "top_k": 12,
  "min_score": 0.05,
  "min_cooc": 2
}
```

## Design Contracts

See:
- `docs/architecture.md`
- `docs/dev-workflow.md`

## RSVS UI Subtree

Frontend interface drop is merged as an isolated app at `apps/rsvs-ui`.

Run locally:

```bash
cd apps/rsvs-ui
# use bun if available, otherwise npm/pnpm
bun install
bun run dev
```

This subtree is intentionally isolated to avoid affecting Rust/Python root workflows.
