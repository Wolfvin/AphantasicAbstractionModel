<div align="center">

# ⚡ RSVS

### Relational Symbolic Vocabulary System

**A cognitive symbolic engine with hard attention, multi-sense disambiguation, and autonomous tiered memory lifecycle.**

[![CI](https://img.shields.io/github/actions/workflow/status/Wolfvin/SymbolicPuzzle3D/ci.yml?branch=main&style=flat-square&logo=github)](https://github.com/Wolfvin/SymbolicPuzzle3D/actions)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)](https://python.org)
[![crates.io](https://img.shields.io/crates/v/rsvs-core?style=flat-square&logo=rust)](https://crates.io/crates/rsvs-core)
[![PyPI](https://img.shields.io/pypi/v/rsvs?style=flat-square&logo=pypi)](https://pypi.org/project/rsvs/)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-green?style=flat-square)](LICENSE)
[![Schema](https://img.shields.io/badge/Schema-v4.2-purple?style=flat-square)]()

[Getting Started](#-quick-start) · [Why RSVS?](#-why-rsvs) · [Architecture](#-architecture) · [API Reference](docs/API.md) · [Contributing](CONTRIBUTING.md)

</div>

---

![RSVS 3D Knowledge Graph](docs/images/demo.png)

---

## 💡 Why RSVS?

Traditional knowledge graphs use **softmax attention** — dense, opaque, and non-deterministic. RSVS takes a fundamentally different approach:

| Problem | Traditional | RSVS |
|---------|-------------|------|
| **Attention** | Softmax → dense, all tokens get weight | Hard selection → sparse, top-k survive |
| **Interpretability** | Opaque weight matrices | Explicit NPMI + Jaccard + Co-occurrence decomposition |
| **Determinism** | Varies with temperature | Same input → same output, always |
| **Lifecycle** | Manual curation or passive decay | Autonomous: EMA confidence, hysteresis, quarantine |
| **Disambiguation** | Context vectors | Dynamic sense clusters with incremental O(n) coherence |
| **Governance** | Ad-hoc rules | Single-owner policy engine with evidence scoring and dedup |

**RSVS is for developers and researchers who need:**
- 🔬 **Interpretable** knowledge extraction — every score decomposes into named components
- 🧬 **Autonomous** lifecycle management — nodes promote, demote, and quarantine themselves
- ⚡ **High-performance** computation — Rust core handles millions of nodes with zero-cost abstractions
- 🌐 **Real-time** 3D visualization — explore your knowledge graph interactively

---

## ✨ Key Features

- 🧠 **Hard Attention** — `score = α·NPMI + β·Jaccard + γ·cooc` — sparse, deterministic, interpretable
- 🌱 **24 Seed Atoms** — Axiomatic foundation that cannot be deleted or decay (exists, entity, relation, ...)
- 🔀 **Multi-Sense Disambiguation** — Dynamic sense clusters with incremental O(n) coherence
- 🛡️ **Autonomy Engine** — EMA confidence with hysteresis (promote ≥ 0.75, demote < 0.60), quarantine, rollback
- ⚖️ **Policy Engine** — Single-owner governance, source trust weighting, dedup gate, evidence scoring
- 🏗️ **Rust Core + Python Bridge** — Compute-heavy ops in Rust, HTTP/API in Python via PyO3
- 🎯 **Three Modes** — `ingest` (learn), `appraise` (evaluate), `relate` (discover)
- 🌐 **3D Knowledge Graph** — Interactive React Three Fiber visualization with force-directed layout

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                      │
│            React Three Fiber · Zustand · shadcn/ui           │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐   │
│  │ ForceGraph│  │ Appraise │  │  Relate  │  │ Timeline   │   │
│  │  (3D)    │  │  Panel   │  │  Panel   │  │  + HUD     │   │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP (POST /run, GET /latest)
┌──────────────────────────▼──────────────────────────────────┐
│                Python Bridge (HTTP Server)                    │
│                                                               │
│  ┌────────────┐ ┌────────────┐ ┌───────────┐ ┌──────────┐  │
│  │ bridge_    │ │  modes.py  │ │validation │ │conversion│  │
│  │ server.py  │ │ (dispatch) │ │   .py     │ │  .py     │  │
│  └────────────┘ └─────┬──────┘ └───────────┘ └──────────┘  │
│                       │         ┌───────────┐ ┌──────────┐  │
│                       │         │ artifacts │ │ rsvs_    │  │
│                       │         │   .py     │ │ core.py  │  │
│                       │         └───────────┘ └─────┬────┘  │
└───────────────────────┼─────────────────────────────┼───────┘
                        │                             │ PyO3
┌───────────────────────┼─────────────────────────────▼───────┐
│                       │         Rust Core (rsvs-core)        │
│                                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ pipeline │ │attention │ │ autonomy │ │    sense.rs    │  │
│  │   .rs    │ │   .rs    │ │   .rs    │ │ (multi-sense)  │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │  graph   │ │  seed.rs │ │ persist  │ │   events.rs    │  │
│  │   .rs    │ │ (24 atoms)│ │   .rs    │ │ (stream)       │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

> 📖 For the full technical reference, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 🚀 Quick Start

### Prerequisites

- **Rust** 1.75+ ([install](https://rustup.rs/))
- **Python** 3.12+ ([install](https://python.org))
- **Node.js** 18+ ([install](https://nodejs.org/))
- **maturin** (`pip install maturin`)

### Rust Core Only (no Python, no frontend)

```bash
cd backend
cargo run --bin rsvs-smoke    # Runs 114 unit tests + smoke pipeline
```

### Full Stack

```bash
# 1. Clone
git clone https://github.com/Wolfvin/SymbolicPuzzle3D.git
cd SymbolicPuzzle3D

# 2. Build Rust core + Python bindings
cd backend/python
pip install maturin
maturin develop              # ← Compiles Rust core, installs Python wheel

# 3. Start FastAPI bridge server
python -m rsvs.fastapi_server

# 4. Start frontend (new terminal)
cd frontend
npm install
npm run dev
```

The frontend opens at `http://localhost:3000`, the bridge at `http://localhost:8000`.

> ⚠️ **Don't skip `maturin develop`!** This step compiles the Rust core and creates the Python bindings. Without it, the bridge server will start but report `rust_core_available: false`.

---

## 📖 Core Concepts

### Hard Attention

RSVS uses **hard selection** instead of softmax attention:

```
score(t, c) = α · NPMI(t, c) + β · Jaccard(A(t), A(c)) + γ · cooc(t, c)
```

| Component | Role | Default Weight |
|-----------|------|---------------|
| NPMI | Normalized Pointwise Mutual Information | α = 0.4 |
| Jaccard | Atom set overlap similarity | β = 0.4 |
| Co-occurrence | Conditional frequency | γ = 0.2 |

Result: **sparse, interpretable, deterministic** attention scores. No softmax, no temperature, no randomness.

### Seed Atom Bootstrap

24 primitive atoms form an axiomatic foundation:

| Layer | Atoms |
|-------|-------|
| Existential | `exists`, `entity`, `relation`, `state`, `change` |
| Spatiotemporal | `time`, `space`, `cause`, `effect`, `context` |
| Cognitive | `signal`, `pattern`, `memory`, `attention`, `value` |
| Agentic | `agent`, `goal`, `risk`, `trust`, `identity` |
| Linguistic | `language`, `meaning`, `action`, `feedback` |

These nodes have confidence=1.0, Tier=Tier1, status=Stable, and are **immutable**.

### Node Lifecycle

```
New ──→ Candidate ──→ Stable ──→ Deprecated
              ↕           ↕          ↕
         Quarantine ←─────┘──────────┘
```

Hysteresis prevents flip-flopping:
- **Promote** at confidence ≥ 0.75
- **Demote** at confidence < 0.60
- **Quarantine** after 3+ status flips (circuit-breaker pattern)

### Multi-Sense Disambiguation

Each node can have multiple sense clusters:

```
"bank" → Sense 0: financial institution [coherence: 0.85, status: mature]
       → Sense 1: river edge [coherence: 0.72, status: fragile]
```

Senses form from data — never hardcoded. Fragile senses (N=1) are pruned after inactivity. Mature senses merge when Jaccard similarity exceeds θ_merge.

---

## 🔧 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/run` | POST | General mode dispatch (`ingest`, `appraise`, `relate`) |
| `/ingest` | POST | Text ingestion (shorthand for `/run` with `mode=ingest`) |
| `/latest` | GET | Retrieve latest snapshot/artifacts |
| `/health` | GET | Health check with Rust core availability |
| `/status` | GET | Runtime statistics from the Rust core |

> 📖 For complete API documentation with request/response schemas and examples, see [docs/API.md](docs/API.md)

---

## ⚡ Performance

Benchmarks run on an Apple M2 Pro with criterion:

| Benchmark | Time | Notes |
|-----------|------|-------|
| `jaccard_100_elements` | ~2 µs | Atom set similarity for 100-element sets |
| `npmi_lookup` | ~50 ns | Single NPMI table lookup |
| `cooc_ingest_sentence_20_tokens` | ~5 µs | Co-occurrence stats for 20-token sentence |
| `sense_ingest_10_atoms` | ~15 µs | Sense assignment for 10-atom context |
| `pipeline_ingest_text` | ~800 µs | Full pipeline: tokenize → attention → sense → autonomy → persist |

Run benchmarks yourself:

```bash
cd backend && cargo bench
```

Results are generated in `target/criterion/` with HTML reports.

---

## 🧪 Testing

```bash
# Rust unit tests (114 tests)
cd backend && cargo test --lib

# Rust benchmarks
cd backend && cargo bench

# Python tests
cd backend/python && pytest tests/ -v

# Full pipeline smoke test
cd backend && cargo run --bin rsvs-smoke
```

---

## 📁 Project Structure

```
SymbolicPuzzle3D/
├── backend/
│   ├── crates/rsvs-core/src/     # Rust core engine
│   │   ├── types.rs              # Unified node model (v4.2)
│   │   ├── graph.rs              # DAG storage + similarity
│   │   ├── seed.rs               # 24-atom bootstrap
│   │   ├── attention.rs          # Hard attention scoring
│   │   ├── sense.rs              # Multi-sense framework
│   │   ├── autonomy.rs           # Confidence/tier lifecycle
│   │   ├── pipeline.rs           # End-to-end orchestration
│   │   ├── persist.rs            # State serialization
│   │   ├── events.rs             # Event stream contracts
│   │   └── bindings.rs           # PyO3 Python bindings
│   └── python/rsvs/              # Python bridge
│       ├── fastapi_server.py     # FastAPI HTTP server (async, OpenAPI)
│       ├── config.py             # Configuration
│       ├── validation.py         # Schema validation
│       ├── conversion.py         # Format conversion
│       ├── artifacts.py          # File persistence
│       ├── rsvs_core.py          # Rust core wrapper
│       └── modes.py              # Mode implementations
├── frontend/                     # Next.js 3D graph UI
├── cli/                          # Agent CLI
├── docs/                         # Architecture & API docs
├── .github/                      # CI/CD + issue templates
├── CHANGELOG.md                  # Version history
├── CONTRIBUTING.md               # Contribution guide
└── LICENSE                       # MIT OR Apache-2.0
```

---

## 🗺️ Roadmap

- [ ] **v4.3** — Streaming events via WebSocket/SSE for real-time UI updates
- [x] ~~**v4.3** — FastAPI migration (OpenAPI docs, async handlers)~~ ✅ Done!
- [ ] **v4.4** — Multi-domain knowledge graphs with cross-domain edges
- [ ] **v4.4** — Graph embeddings for approximate nearest-neighbor search
- [ ] **v5.0** — Distributed Rust core with Raft consensus
- [ ] **v5.0** — Plugin system for custom attention scorers and policy rules
- [ ] **Future** — LLM integration for guided knowledge extraction
- [ ] **Future** — Export to RDF/OWL for interoperability with semantic web

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development environment setup
- Code style guidelines (Rust, Python, TypeScript)
- Commit message convention (Conventional Commits)
- PR process and testing requirements
- How to add new modes and extend the Rust core

---

## 📄 License

Dual-licensed under [MIT](LICENSE) OR [Apache-2.0](LICENSE). You may choose either license.

---

<div align="center">

**[Report a Bug](https://github.com/Wolfvin/SymbolicPuzzle3D/issues/new?template=bug_report.md)** ·
**[Request a Feature](https://github.com/Wolfvin/SymbolicPuzzle3D/issues/new?template=feature_request.md)** ·
**[Read the Docs](docs/ARCHITECTURE.md)**

</div>
