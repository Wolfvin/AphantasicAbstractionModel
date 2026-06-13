# Contributing to RSVS

Thank you for your interest in contributing to RSVS! This guide will help you set up your development environment, understand our code style, and submit your changes.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Architecture Overview](#architecture-overview)
- [Code Style Guidelines](#code-style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Testing Requirements](#testing-requirements)
- [How to Add New Modes](#how-to-add-new-modes)
- [How to Extend the Rust Core](#how-to-extend-the-rust-core)
- [How to Add Frontend Components](#how-to-add-frontend-components)

---

## Development Environment Setup

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Rust | 1.75+ | [rustup.rs](https://rustup.rs/) |
| Python | 3.11+ | [python.org](https://python.org) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| maturin | latest | `pip install maturin` |

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Wolfvin/AphantasicAbstractionModel.git
cd AphantasicAbstractionModel

# 2. Build and test the Rust core
cd backend
cargo test --lib          # 114 unit tests
cargo clippy --all-targets  # Zero warnings required
cargo fmt -- --check       # Formatted code required

# 3. Build Python bindings and test
cd python
pip install maturin
maturin develop            # Compiles Rust core + creates Python wheel
pip install -e ".[dev]"    # Install dev dependencies
pytest tests/ -v           # Python test suite

# 4. Start the bridge server
python -m rsvs.fastapi_server

# 5. Start the frontend (new terminal)
cd frontend
npm install
npm run lint               # Zero errors required
npm run dev                # Development server

# 6. Run the smoke test binary
cd backend
cargo run --bin rsvs-smoke
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `RSVS_BRIDGE_HOST` | `127.0.0.1` | Bridge server bind address |
| `RSVS_BRIDGE_PORT` | `8000` | Bridge server bind port |
| `RSVS_ATOM_OUTPUT_DIR` | `../atom` | Artifact output directory |
| `RSVS_ATTENTION_CONFIG` | — | Path to JSON config for attention weights override |

---

## Architecture Overview

RSVS is a three-tier system. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical reference.

```
┌─────────────────────────────┐
│  Frontend (Next.js + R3F)   │  TypeScript · Zustand · shadcn/ui
├─────────────────────────────┤
│  Python Bridge (rsvs)       │  HTTP · Validation · Artifact I/O
├─────────────────────────────┤
│  Rust Core (rsvs-core)      │  Graph · Attention · Autonomy · Sense
└─────────────────────────────┘
```

### Key Boundaries

- **Rust core** (`backend/crates/rsvs-core/src/`): All computational logic. No HTTP, no file I/O, no Python.
- **Python bridge** (`python/rsvs/`): HTTP layer + artifact I/O only. No computation.
- **Frontend** (`frontend/`): Visualization + user interaction. No business logic.

### Module Map

**Rust Core:**

| Module | File | Responsibility |
|--------|------|---------------|
| types | `types.rs` | Unified node model, enums |
| graph | `graph.rs` | DAG storage, Jaccard similarity |
| seed | `seed.rs` | 24-atom bootstrap |
| attention | `attention.rs` | Hard attention scoring |
| sense | `sense.rs` | Multi-sense framework |
| autonomy | `autonomy.rs` | Confidence/tier lifecycle |
| pipeline | `pipeline.rs` | End-to-end orchestration |
| persist | `persist.rs` | JSON serialization |
| events | `events.rs` | Event stream |
| bindings | `bindings.rs` | PyO3 bindings (feature-gated) |

**Python Bridge:**

| Module | File | Responsibility |
|--------|------|---------------|
| fastapi_server | `fastapi_server.py` | FastAPI HTTP server (async, OpenAPI) |
| bridge_server | `bridge_server.py` | *(Deprecated)* Legacy HTTP handler |
| modes | `modes.py` | Mode dispatch (ingest/appraise/relate) |
| validation | `validation.py` | Schema validation |
| conversion | `conversion.py` | Format conversion |
| artifacts | `artifacts.py` | File persistence |
| rsvs_core | `rsvs_core.py` | Rust core wrapper |
| config | `config.py` | Configuration constants |
| exceptions | `exceptions.py` | Exception hierarchy |

---

## Code Style Guidelines

### Rust

- **Format**: Run `cargo fmt` before every commit. CI will reject unformatted code.
- **Lint**: Run `cargo clippy --all-targets -- -D warnings`. Zero warnings required.
- **Doc comments**: All public APIs must have `///` doc comments with examples.
- **Error handling**: Use `thiserror` for error types. Never use `String` as an error type.
- **Feature flags**: PyO3 bindings must be behind `#[cfg(feature = "python")]`.
- **Tests**: Use `#[cfg(test)] mod tests` within each module. Integration tests go in `tests/`.
- **Benchmarks**: Use criterion for performance-critical code. Add to `benches/`.

```rust
// ✅ Good — proper error type, doc comment, feature-gated
/// Ingest text into the knowledge graph.
///
/// # Arguments
/// * `text` - The input text to process
///
/// # Returns
/// `IngestStats` with processing metadata
#[cfg(feature = "python")]
pub fn ingest_with_meta_v1(&mut self, text: &str, domain_id: Option<u32>) -> IngestMeta {
    // ...
}

// ❌ Bad — no doc comment, String error
pub fn ingest(&mut self, text: &str) -> Result<(), String> {
    // ...
}
```

### Python

- **Format**: Run `ruff format .` before every commit.
- **Lint**: Run `ruff check .`. Zero errors required.
- **Type hints**: Required for all function signatures. Use `from __future__ import annotations`.
- **Docstrings**: Google-style docstrings for all public functions.
- **Exceptions**: Use custom exceptions from `exceptions.py`. Never raise bare `ValueError`.
- **Imports**: Use relative imports within the package (`from .config import ...`).

```python
# ✅ Good — type hints, docstring, custom exception
def _run_mode(mode: str, text: str, correlation_id: str, options: dict[str, Any] | None) -> dict[str, Any]:
    """Execute the given mode and return a structured envelope.

    Args:
        mode: One of 'ingest', 'appraise', 'relate'.
        text: Input text to process.
        correlation_id: Client-provided correlation ID.
        options: Mode-specific options.

    Returns:
        Structured response envelope with result, messages, and files.

    Raises:
        InvalidModeError: If the mode is not recognized.
    """
    ...

# ❌ Bad — no type hints, bare ValueError, no docstring
def run_mode(mode, text, corr_id, opts):
    if mode not in MODES:
        raise ValueError("bad mode")
    ...
```

### TypeScript

- **Format**: Run `npm run lint` before every commit. Zero errors required.
- **Strict mode**: Enabled (`noImplicitAny: true`). All code must pass strict checks.
- **Components**: Use shadcn/ui components. Avoid building from scratch.
- **State**: Use Zustand for client state. No prop drilling.
- **Imports**: Use `@/` alias for project imports.

```typescript
// ✅ Good — typed, uses shadcn component
interface AppraisePanelProps {
  result: AppraiseResult | null;
}

export function AppraisePanel({ result }: AppraisePanelProps) {
  if (!result) return null;
  return (
    <Card>
      <CardHeader>
        <Badge variant={result.verdict === 'agree' ? 'default' : 'destructive'}>
          {result.verdict}
        </Badge>
      </CardHeader>
    </Card>
  );
}

// ❌ Bad — any type, no props interface
export function AppraisePanel({ result }: any) {
  return <div>{result.verdict}</div>;
}
```

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types

| Type | Usage |
|------|-------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting (no code change) |
| `refactor` | Code restructuring (no behavior change) |
| `test` | Adding or updating tests |
| `chore` | Build process, tooling, dependencies |
| `perf` | Performance improvement |

### Examples

```
feat(attention): add configurable attention weights via env var

Adds RSVS_ATTENTION_CONFIG environment variable support for
overriding default α, β, γ weights from a JSON config file.

Closes #42
```

```
fix(autonomy): prevent confidence overflow in EMA update

Clamps evidence to [0, 1] before applying EMA formula
to prevent confidence values exceeding 1.0.

Fixes #89
```

```
docs(api): add complete API reference with request/response schemas
```

```
refactor(bridge): remove legacy Python fallback code

Removes _legacy_* functions and unified dispatch fallback.
All computation now goes through the Rust core.
```

### Scopes

Common scopes: `attention`, `autonomy`, `sense`, `graph`, `pipeline`, `bridge`, `frontend`, `api`, `cli`

---

## Pull Request Process

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feat/amazing-feature`
3. **Make your changes** following the code style guidelines above
4. **Run all tests**:
   ```bash
   cd backend && cargo test --lib && cargo clippy --all-targets -- -D warnings
   cd ../python && pytest tests/ -v && ruff check .
   cd frontend && npm run lint
   ```
5. **Write a clear PR description** using the PR template
6. **Ensure CI passes** (GitHub Actions runs on all PRs)
7. **Request review** from a maintainer
8. **Address review feedback** and push updates

### PR Checklist

- [ ] Code follows style guidelines (cargo fmt, ruff, eslint)
- [ ] Self-review completed
- [ ] Hard-to-understand code is commented
- [ ] Documentation updated if needed
- [ ] No new warnings generated
- [ ] Tests added for new features/bug fixes
- [ ] All tests pass locally

---

## Testing Requirements

### Rust Tests

```bash
# Unit tests (114 tests)
cd backend && cargo test --lib

# With output
cd backend && cargo test --lib -- --nocapture

# Specific module
cd backend && cargo test --lib sense

# Clippy (zero warnings)
cd backend && cargo clippy --all-targets -- -D warnings

# Format check
cd backend && cargo fmt -- --check
```

### Python Tests

```bash
# All tests
cd ../python && pytest tests/ -v

# Specific test file
cd ../python && pytest tests/test_conversion.py -v

# With coverage
cd ../python && pytest tests/ -v --cov=rsvs

# Lint
cd ../python && ruff check .
```

### Frontend Tests

```bash
# Lint (zero errors)
cd frontend && npm run lint

# Build check
cd frontend && npm run build
```

### Smoke Test

```bash
# Full pipeline smoke test (Rust binary)
cd backend && cargo run --bin rsvs-smoke
```

---

## How to Add New Modes

RSVS currently supports nine modes: `ingest`, `appraise`, `relate`, `compose`, `structural_similarity`, `substitution_analysis`, `grounding_info`, `context_query`, `context_similarity`. To add a new mode (e.g., `summarize`):

### 1. Add Rust Core Support

In `backend/crates/rsvs-core/src/pipeline.rs`:

```rust
/// Execute the summarize mode.
pub fn summarize(&self, text: &str) -> Result<SummarizeResult, RsvsError> {
    // 1. Tokenize input
    // 2. Find relevant nodes via attention
    // 3. Build summary from top-ranked nodes
    // 4. Return structured result
    todo!("Implement summarize mode")
}
```

Add the result type in `types.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeResult {
    pub summary: String,
    pub key_nodes: Vec<(NodeId, f32)>,
    pub coverage: f32,
}
```

Add PyO3 binding in `bindings.rs`:

```rust
#[pymethods]
impl RsvsPy {
    fn summarize(&mut self, text: &str) -> PyResult<SummarizeResultPy> {
        // ...
    }
}
```

### 2. Add Python Bridge Support

In `python/rsvs/modes.py`:

```python
def _run_summarize_rust(
    text: str,
    correlation_id: str,
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Summarize via the Rust core."""
    r = _get_rsvs()
    if r is None:
        raise RustCoreUnavailableError("Rust core required for summarize mode")
    # Call Rust core
    result = r.summarize(text)
    # Convert and return
    ...
```

Register the mode in the dispatch map:

```python
_MODE_HANDLERS = {
    "ingest": _run_ingest_rust,
    "appraise": _run_appraise_rust,
    "relate": _run_relate_rust,
    "summarize": _run_summarize_rust,  # ← Add here
}
```

Update `config.py`:

```python
VALID_MODES = {"ingest", "appraise", "relate", "summarize"}
```

### 3. Add Frontend Support

In `frontend/src/lib/types.ts`:

```typescript
export interface SummarizeResult {
  summary: string;
  keyNodes: Array<{ id: number; label: string; score: number }>;
  coverage: number;
}
```

Create `SummarizePanel.tsx` component and add it to `RightNodeDrawer.tsx`.

---

## How to Extend the Rust Core

### Adding a New Module

1. Create a new file: `backend/crates/rsvs-core/src/your_module.rs`
2. Register it in `lib.rs`:
   ```rust
   pub mod your_module;
   ```
3. Add any dependencies in `pipeline.rs` or other modules as needed
4. Add tests in `your_module.rs`:
   ```rust
   #[cfg(test)]
   mod your_module_tests {
       use super::*;
       // ...
   }
   ```

### Adding PyO3 Bindings

All PyO3 bindings must be feature-gated:

```rust
// bindings.rs
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymethods]
impl RsvsPy {
    fn your_new_method(&mut self, arg: &str) -> PyResult<String> {
        // Delegate to the actual Rust implementation
        Ok(self.rsvs.your_rust_method(arg).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })?)
    }
}
```

### Adding Benchmarks

1. Create a benchmark in `benches/rsvs_bench.rs`:
   ```rust
   fn bench_your_feature(c: &mut Criterion) {
       let mut group = c.benchmark_group("your_feature");
       group.bench_function("your_case", |b| {
           b.iter(|| {
               // benchmark code
           })
       });
       group.finish();
   }
   ```
2. Run: `cd backend && cargo bench`

### Important Rules

- **No Python in Rust**: The Rust core must compile without Python. All Python-facing code goes in `bindings.rs` behind `#[cfg(feature = "python")]`.
- **No I/O in core**: The Rust core does not do file I/O or HTTP. All persistence goes through `persist.rs` and the Python bridge.
- **No mock data**: Production code paths must never use mock/fallback data. If the Rust core is unavailable, raise `RustCoreUnavailableError`.
- **Deterministic**: Same inputs must always produce same outputs. No randomness in core algorithms.

---

## How to Add Frontend Components

### Adding a New Panel

1. Define types in `frontend/src/lib/types.ts`
2. Create the component in `frontend/src/components/rsvs/`
3. Use shadcn/ui components for consistent styling
4. Add Zustand store state if needed in `frontend/src/store/rsvsStore.ts`
5. Import and render in the appropriate parent component

### Adding Backend Communication

1. Add the API call in `frontend/src/lib/backendBridge.ts`
2. Call from the appropriate component (typically `LeftInputRail.tsx`)
3. Update the Zustand store with the response
4. Render the result in the relevant panel

### Component Guidelines

- Use `'use client'` directive for client components
- Use shadcn/ui components (Card, Badge, Button, etc.) for consistent UI
- Use framer-motion for animations (hover, entry transitions)
- Set `max-h-96 overflow-y-auto` for long lists with custom scrollbar styling
- Use responsive design (`sm:`, `md:`, `lg:` prefixes)
- Minimum 44px touch targets for interactive elements

---

## Getting Help

- **Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md) for technical reference, [docs/API.md](docs/API.md) for API docs
- **Issues**: [GitHub Issues](https://github.com/Wolfvin/AphantasicAbstractionModel/issues) for bugs and feature requests
- **Discussions**: [GitHub Discussions](https://github.com/Wolfvin/AphantasicAbstractionModel/discussions) for questions and ideas

Thank you for contributing to RSVS! 🚀
