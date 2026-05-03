# Contributing to RSVS

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites
- Rust 1.75+ (with cargo)
- Python 3.10+
- Node.js 18+

### Quick Start

```bash
git clone https://github.com/Wolfvin/SymbolicPuzzle3D.git
cd SymbolicPuzzle3D

# Rust tests
cd backend && cargo test

# Python setup
cd backend/python && pip install -e ".[dev]" && pytest

# Frontend
cd frontend && npm install && npm run dev
```

## Architecture Overview

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full architecture document.

Key boundaries:
- **Rust core** (`crates/rsvs-core/src/`): All computational logic
- **Python bridge** (`python/rsvs/`): HTTP layer + artifact I/O only
- **Frontend** (`frontend/`): Visualization + user interaction

## Code Style

### Rust
- Run `cargo fmt` before committing
- Run `cargo clippy` and fix all warnings
- All public APIs must have doc comments

### Python
- Run `ruff check .` and `ruff format .`
- Type hints required for all function signatures
- Google-style docstrings

### TypeScript
- Run `npm run lint`
- Strict mode enabled

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Ensure all tests pass
5. Submit a pull request

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new appraise evidence panel
fix: resolve seed node validation edge case
docs: update architecture diagram
test: add property tests for autonomy engine
refactor: remove legacy Python fallback code
```
