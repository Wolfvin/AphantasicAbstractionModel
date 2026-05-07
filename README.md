[![PyPI](https://img.shields.io/pypi/v/rsvs?style=flat-square&logo=pypi&color=3775A9)](https://pypi.org/project/rsvs/)
[![CI](https://img.shields.io/github/actions/workflow/status/Wolfvin/SymbolicPuzzle3D/ci.yml?style=flat-square&logo=github&label=tests)](https://github.com/Wolfvin/SymbolicPuzzle3D/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-8A2BE2?style=flat-square&logo=materialformkdocs&logoColor=white)](https://wolfvin.github.io/SymbolicPuzzle3D/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?style=flat-square&logo=rust)](https://www.rust-lang.org/)

# RSVS — Recursive Symbolic Vocabulary System

**Compositional symbolic meaning, not embeddings. Traceable sense definitions with structural similarity.**

📖 **[Full documentation](https://wolfvin.github.io/SymbolicPuzzle3D/)** · 🚀 [Quick Start](#quick-start) · 📚 [Tutorials](https://wolfvin.github.io/SymbolicPuzzle3D/tutorials/) · 🔧 [API Reference](https://wolfvin.github.io/SymbolicPuzzle3D/api/)

---

## What is RSVS?

RSVS (Recursive Symbolic Vocabulary System) is a compositional symbolic meaning engine that builds structured knowledge graphs from raw text. Unlike vector embeddings that compress meaning into opaque floating-point arrays, RSVS represents every concept as a composition of other concepts -- and every composition is traceable. You can follow the chain from any concept down to its constituent parts, explain precisely why two concepts are related, and identify the exact substitution that transforms one into another. The system is designed to be an interpretation layer that works alongside Transformer models, adding symbolic traceability to dense vector representations.

At its core, RSVS ingests text and builds a knowledge graph composed of atoms, senses, and compositions. An atom is the smallest unit of meaning -- a token that has been promoted to entity status through co-occurrence statistics. A sense is a particular meaning of a concept, defined by which other senses it is composed from. A composition is a directed reference from one sense to another, forming a directed acyclic graph of meaning. When you define that "raja" (king) is composed of "tahta_tertinggi" (highest throne), "laki_laki" (male), and "kerajaan" (kingdom), you have created a precise, inspectable specification of what that word means -- not a statistical artifact, but a structural definition.

RSVS is built with a Rust core compiled to Python via PyO3 and maturin, giving you the safety and speed of Rust with the ergonomics of a Python library. It prioritizes Bahasa Indonesia as its primary language for development and testing, while supporting English and other languages through a fully language-agnostic architecture. The system includes an autonomous tiered memory lifecycle (New, Candidate, Stable, Deprecated), Monte Carlo Tree Search for complex reasoning paths, consolidation and reflection engines for self-maintenance, and an optional FastAPI server for production deployments. A Next.js demo frontend provides interactive 3D graph visualization.

---

## Why RSVS?

If you have used word embeddings or sentence transformers, you are familiar with the pattern: "raja and ratu have cosine similarity 0.87." But what does that number mean? Which aspects of meaning make them similar? What would you need to change to transform one into the other? Embeddings cannot answer these questions because they compress meaning into a single opaque vector. Knowledge graphs and ontologies offer more structure, but they require manual schema design and struggle with ambiguity, multiple senses, and the fluid nature of natural language meaning.

RSVS occupies a different position. It provides the structural precision of a knowledge graph without requiring upfront schema design, and the fuzzy similarity of embeddings without the opacity. When RSVS tells you that "raja" and "ratu" share two compositions (tahta_tertinggi and kerajaan) and differ in exactly one (laki_laki versus perempuan), you have an answer you can inspect, debug, and reason about. This structural approach enables substitution analysis (what transforms concept A into concept B?), context-aware queries (which sense of this word is active given these context atoms?), and compositional verification (are the compositions of this sense well-grounded in evidence?).

Compared to traditional knowledge graphs, RSVS does not require you to define a schema or ontology upfront. The system bootstraps from 24 seed atoms and induces senses automatically from text. Compared to ontologies, RSVS handles ambiguity natively through its multi-sense framework -- a single concept can have multiple senses, each with its own composition structure. Compared to embeddings, RSVS provides full traceability: every similarity score can be decomposed into shared and differing compositions, and every substitution can be named explicitly.

---

## Quick Install

Install the Python library from PyPI:

```bash
pip install rsvs
```

The core library has zero Python dependencies. The Rust engine is compiled into the wheel via PyO3, so there is no separate Rust toolchain needed at install time.

For the optional FastAPI server:

```bash
pip install rsvs[server]
```

For development (includes test tools, linters, and maturin):

```bash
pip install rsvs[dev]
```

To install everything:

```bash
pip install rsvs[all]
```

---

## Quick Start

```python
from rsvs import Rsvs

r = Rsvs(entity_promote_n=3, theta_assign=0.12, n_warm=20, eta=0.1)

r.ingest("Raja adalah pemimpin kerajaan laki-laki. "
         "Ratu adalah pemimpin kerajaan perempuan. "
         "Tahta tertinggi ada di kerajaan.")

r.compose("raja", [("tahta_tertinggi", 0), ("laki_laki", 0), ("kerajaan", 0)], lang="id")
r.compose("ratu", [("tahta_tertinggi", 0), ("perempuan", 0), ("kerajaan", 0)], lang="id")

sim = r.structural_similarity("raja", "ratu")
print(f"Structural similarity: {sim.structural_similarity:.3f}")  # 0.667

sub = r.substitution_analysis("raja", "ratu")
print(f"Substitution: {sub.substitution_labels(r)}")  # [("laki_laki", 0, "perempuan", 0)]
```

One swap -- `laki_laki` becomes `perempuan` -- is the entire semantic difference between king and queen, expressed as a precise structural transformation rather than a fuzzy vector distance.

---

## Core Concepts

### Atoms

An atom is the smallest unit of meaning in RSVS. When text is ingested, tokens are extracted and tracked for co-occurrence statistics. Tokens that appear frequently enough and with sufficient co-occurrence diversity are promoted to atom status. The system bootstraps with 24 seed atoms (basic semantic primitives like "laki_laki", "perempuan", "kekerasan", and others) that provide the initial vocabulary for compositional definitions. Promotion is controlled by the `entity_promote_n` parameter, which sets the minimum number of co-occurrence contexts required before a token becomes an atom.

### Senses

A sense is a particular meaning of a concept. Every node in the RSVS graph can have multiple senses, each representing a distinct usage. For example, "batu" (stone/rock) might have one sense for geological material and another for a gemstone. Each sense carries its own composition structure, grounding score, coherence metric, and status. Senses are induced automatically during ingestion -- the system identifies which atoms are active in the context of each token and uses them as the compositions of a new sense. Senses can also be defined explicitly via the `compose()` method.

### Compositions

A composition is a directed reference from one sense to another sense, forming the edges of the meaning graph. When you define "raja" as composed of ("tahta_tertinggi", sense 0), ("laki_laki", sense 0), and ("kerajaan", sense 0), you are creating three composition edges. Compositions are the fundamental building blocks of meaning in RSVS. They enable structural similarity (comparing shared and differing compositions between two concepts), substitution analysis (identifying the precise swaps that transform one concept into another), and compositional verification (checking whether a sense's compositions are well-grounded in evidence).

### Layers

Layers represent the depth of compositional structure. Seed atoms exist at layer 0. A concept composed entirely of layer-0 atoms exists at layer 1. A concept composed of at least one layer-1 sense exists at layer 2, and so on. Layers create a natural hierarchy: you cannot define a higher-layer concept without first having the lower-layer concepts it depends on. The layer system prevents circular definitions and enables the system to reason about compositional depth.

### Tiers and Autonomous Memory

RSVS implements an autonomous tiered memory lifecycle for all nodes. Every node progresses through four tiers based on its confidence and activity: New (just created, low confidence), Candidate (some evidence, gaining confidence), Stable (well-established, high confidence), and Deprecated (inactive, low confidence, scheduled for removal). This lifecycle is managed by the autonomy engine, which uses exponential moving averages (EMA) and hysteresis thresholds to prevent rapid oscillation between tiers. The `eta` parameter controls the EMA smoothing factor.

### Grounding

Grounding is the process of verifying that a sense's compositions are supported by evidence in the corpus. After a sense is formed, every subsequent ingestion that involves that sense updates its grounding score. Confirming contexts (where the sense's compositions are also active) boost the grounding score; contradicting contexts (where expected compositions are absent) penalize it. The asymmetric penalty (0.10) versus boost (0.05) ensures that poorly grounded compositions are caught quickly. A sense with a grounding score above 0.60 is considered well-grounded; below 0.20 it needs revision; and below that threshold it is a candidate for retirement.

---

## API Reference

### Core Operations

These are the primary operations for building and querying a knowledge graph.

| Method | Signature | Description |
|--------|-----------|-------------|
| `ingest` | `r.ingest(text: str) -> IngestStats` | Ingest text, update co-occurrence stats, promote atoms, induce senses. Returns stats on sentences processed, atoms promoted, senses created. |
| `query` | `r.query(concept: str, context: str) -> QueryResult \| None` | Context-aware query. Returns the active sense, its layer, grounding score, compositions, and scored atoms. |
| `context_query` | `r.context_query(concept, atoms, max_depth, gamma, halt_confidence, tau_relevance) -> ContextQueryResult \| None` | Depth-controlled lazy traversal with context atoms. Supports configurable depth limits, relevance thresholds, and halt conditions. |
| `compose` | `r.compose(label, compositions, lang) -> int` | Create a new compositional node. `compositions` is a list of `(label, sense_idx)` tuples. Returns the new node ID. |

### Analysis Operations

These methods compare concepts and analyze text against the knowledge graph.

| Method | Signature | Description |
|--------|-----------|-------------|
| `similarity` | `r.similarity(a: str, b: str) -> SimResult \| None` | Flat Jaccard similarity based on shared atoms. Returns Jaccard score, shared atoms, and atoms unique to each side. |
| `structural_similarity` | `r.structural_similarity(a: str, b: str) -> StructuralSimResult \| None` | Sense-level composition comparison. Returns the structural similarity score, shared compositions, and compositions unique to each sense. |
| `substitution_analysis` | `r.substitution_analysis(a: str, b: str) -> SubstitutionResult \| None` | Find the precise swaps that transform concept A into concept B. Returns paired substitutions and unpaired remainders. |
| `context_similarity` | `r.context_similarity(a: str, b: str, context: list[str]) -> float \| None` | Context-weighted similarity. Weights shared atoms by their relevance to the provided context. |
| `appraise` | `r.appraise(text: str) -> AppraiseResult` | Evaluate text plausibility against the graph. Returns agree/disagree percentages, verdict, and supporting evidence. |
| `relate` | `r.relate(concept: str) -> RelateResult \| None` | Find related nodes and edges via spreading activation along composition edges. |

### Composition Operations

| Method | Signature | Description |
|--------|-----------|-------------|
| `compose` | `r.compose(label, compositions, lang) -> int` | Create compositional node from label/sense references. |
| `compose_from_ids` | `r.compose_from_ids(label, atom_ids, lang) -> int` | Create compositional node from integer atom IDs. |

### Reasoning Operations

| Method | Signature | Description |
|--------|-----------|-------------|
| `mcts_query` | `r.mcts_query(label, simulations, exploration) -> MCTSResult \| None` | Monte Carlo Tree Search for complex disambiguation. Uses UCB1 selection and structural value functions. Returns active sense, scored atoms, depth reached, and halt reason. |
| `set_thinking_mode` | `r.set_thinking_mode(mode)` | Control traversal depth. `-1` = AUTO (router decides), `0` = NON_THINKING (shallow, fast), `1` = THINKING (deep, thorough). |
| `consolidate` | `r.consolidate() -> ConsolidationResult` | Periodic graph cleanup: merge similar senses, remove dead senses, prune weak edges, compact records. |
| `run_reflection` | `r.run_reflection() -> ReflectionResult` | Self-evaluate all senses. Produces CONFIRM, REVIEW, REVISE, or RETIRE actions based on grounding evidence. |
| `verify` | `r.verify() -> dict` | Neuro-symbolic composition verification. Checks five structural rules: no self-reference, layer consistency, grounding threshold, frequency threshold, and no circular chains. |

### Inspection Operations

| Method | Signature | Description |
|--------|-----------|-------------|
| `node_info` | `r.node_info(label) -> NodeInfo` | Detailed node information: label, layer, confidence, tier, status, composition state. |
| `senses` | `r.senses(concept) -> list[SenseInfo]` | All senses of a concept with grounding evidence, coherence, compositions, and status. |
| `nodes` | `r.nodes(include_seeds=False) -> list[str]` | List all known node labels. |
| `atoms` | `r.atoms(include_seeds=False) -> list[str]` | List all known atom labels. |
| `confidence_map` | `r.confidence_map() -> dict[str, float]` | Confidence scores for all nodes. |
| `entity_candidates` | `r.entity_candidates(top_k) -> list[tuple[str, float]]` | Unpromoted tokens with highest centrality scores. |
| `status` | `r.status() -> dict[str, float]` | System status including total nodes, atoms, contexts, and configuration parameters. |

### Persistence Operations

| Method | Signature | Description |
|--------|-----------|-------------|
| `save` | `r.save(path: str) -> None` | Serialize the entire knowledge graph to a JSON file. |
| `load` | `Rsvs.load(path: str) -> Rsvs` | Class method. Deserialize a knowledge graph from a JSON file. |
| `snapshot_v1` | `r.snapshot_v1() -> str` | Runtime snapshot for UI consumption. Returns JSON string. |
| `consume_events_v1` | `r.consume_events_v1(after_seq, limit) -> str` | Incremental event stream. Returns events with sequence numbers after `after_seq`. |
| `latest_seq_v1` | `r.latest_seq_v1() -> int` | Current monotonic event sequence number. |

---

## CLI Usage

RSVS installs a `rsvs` command-line tool after `pip install rsvs`. The CLI provides access to all core operations without writing Python code. State is persisted to a JSON file (default: `./rsvs.json`).

```bash
# Initialize a new knowledge graph
rsvs init --db my_graph.json

# Ingest text (literal string or file path)
rsvs ingest "Batu adalah material keras dari alam" --db my_graph.json
rsvs ingest corpus.txt --db my_graph.json

# Query a concept in context
rsvs query batu "material keras" --db my_graph.json

# Compute similarity between two concepts
rsvs similarity batu kayu --db my_graph.json

# Inspect a concept
rsvs info batu --db my_graph.json
rsvs senses batu --db my_graph.json

# List all atoms
rsvs atoms --db my_graph.json

# Show system status
rsvs status --db my_graph.json

# Ingest from embedded corpus (Bahasa Indonesia domains)
rsvs ingest-corpus --domains geology materials --db my_graph.json
rsvs ingest-corpus --all --db my_graph.json

# Run quality evaluation
rsvs eval --db my_graph.json --json

# Replay incremental event stream
rsvs replay-events --db my_graph.json --after-seq 100 --limit 50
```

All commands support `--json` for machine-readable output. The `init` command accepts tuning parameters: `--promote-n`, `--theta`, `--n-warm`, and `--eta` to configure the RSVS hyperparameters.

---

## FastAPI Server

RSVS includes an optional FastAPI server for HTTP access to all operations. Install with the `server` extra:

```bash
pip install rsvs[server]
```

Start the server:

```bash
# Development mode (auto-reload)
RSVS_DEV_RELOAD=1 python -m rsvs.fastapi_server

# Production mode
RSVS_API_KEY=your-secret-key RSVS_SESSION_SECRET=your-session-secret python -m rsvs.fastapi_server
```

The server runs on `0.0.0.0:8000` by default. Configure with environment variables: `RSVS_HOST`, `RSVS_PORT`, `RSVS_API_KEY`, `RSVS_SESSION_SECRET`, `RSVS_ALLOWED_ORIGINS`.

Key endpoints:

```bash
# Ingest text
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"text": "Raja adalah pemimpin kerajaan."}'

# Compose a node
curl -X POST http://localhost:8000/compose \
  -H "Content-Type: application/json" \
  -d '{"label": "raja", "compositions": [{"label": "tahta_tertinggi", "sense_id": 0}], "lang": "id"}'

# Structural similarity
curl "http://localhost:8000/structural-similarity?a=raja&b=ratu"

# Substitution analysis
curl "http://localhost:8000/substitution-analysis?a=raja&b=ratu"

# MCTS query
curl -X POST http://localhost:8000/mcts-query \
  -H "Content-Type: application/json" \
  -d '{"label": "batu", "simulations": 100}'

# Consolidate the graph
curl -X POST http://localhost:8000/consolidate \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

The server includes API-key-based rate limiting, CORS configuration, request size limits, and production fail-fast checks for required secrets. OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## Examples Gallery

### Comparing King and Queen

```python
from rsvs import Rsvs

r = Rsvs()
r.ingest("Raja adalah pemimpin kerajaan laki-laki. "
         "Ratu adalah pemimpin kerajaan perempuan. "
         "Tahta tertinggi ada di kerajaan.")

r.compose("raja", [("tahta_tertinggi", 0), ("laki_laki", 0), ("kerajaan", 0)], lang="id")
r.compose("ratu", [("tahta_tertinggi", 0), ("perempuan", 0), ("kerajaan", 0)], lang="id")

sim = r.structural_similarity("raja", "ratu")
print(f"Similarity: {sim.structural_similarity:.3f}")   # 0.667
print(f"Shared: {sim.shared_labels(r)}")                # [(tahta_tertinggi, 0), (kerajaan, 0)]

sub = r.substitution_analysis("raja", "ratu")
print(f"Substitution: {sub.substitution_labels(r)}")    # [(laki_laki, 0, perempuan, 0)]
```

### Context-Aware Querying

```python
from rsvs import Rsvs

r = Rsvs()
r.ingest("Batu adalah material keras. Tulang juga material keras. "
         "Batu ditemukan di alam. Tulang ada di tubuh.")

result = r.context_query("batu", ["material", "keras"], max_depth=5)
if result:
    print(f"Active sense: {result.active_sense_idx}")
    print(f"Depth reached: {result.depth_reached}")
    print(f"Scored atoms: {result.scored_atoms[:5]}")
```

### MCTS Reasoning

```python
from rsvs import Rsvs

r = Rsvs()
r.ingest("Batu adalah material keras dari alam. "
         "Batu digunakan untuk konstruksi. "
         "Mineral adalah komponen batu.")

result = r.mcts_query("batu", simulations=100)
if result:
    print(f"Active sense: {result.active_sense_idx}")
    print(f"Simulations: {result.simulations_run}")
    print(f"Depth: {result.depth_reached}")
    print(f"Halt reason: {result.halt_reason}")
    print(f"Best path: {result.best_path}")
```

### Appraising Text Plausibility

```python
from rsvs import Rsvs

r = Rsvs()
r.ingest("Batu adalah material keras. Kayu adalah material organik. "
         "Besi adalah logam keras.")

appraisal = r.appraise("Batu sangat keras")
print(f"Verdict: {appraisal.verdict}")           # "agree" or "disagree"
print(f"Agree: {appraisal.agree_pct:.0f}%")      # e.g. 85%
print(f"Evidence: {appraisal.evidence[:3]}")
```

### Persistence and Loading

```python
from rsvs import Rsvs

# Build a knowledge graph
r = Rsvs()
r.ingest("Kerajaan dipimpin oleh raja atau ratu.")
r.compose("kerajaan", [("raja", 0), ("negara", 0)], lang="id")

# Save to disk
r.save("my_graph.json")

# Load later in a different session
r2 = Rsvs.load("my_graph.json")
print(r2.status())
```

### Finding Related Concepts

```python
from rsvs import Rsvs

r = Rsvs()
r.ingest("Raja memimpin kerajaan. Ratu memimpin kerajaan. "
         "Kerajaan memiliki tahta. Tahta adalah simbol kekuasaan.")

related = r.relate("raja")
if related:
    print(f"Related nodes: {related.node_labels(r)[:5]}")
    print(f"Structural relations: {related.structural_labels(r)[:5]}")
```

---

## Architecture Overview

RSVS follows a three-tier architecture with strict separation of concerns:

- **Rust Core** (`backend/crates/rsvs-core/src/`): All computational logic lives here -- graph storage, attention scoring, sense management, autonomy lifecycle, pipeline orchestration, MCTS, consolidation, reflection, and persistence. The core has no HTTP, no file I/O, and no Python dependencies. It compiles independently and exposes a pure Rust API.

- **Python Bridge** (`python/rsvs/`): The Python layer provides the PyO3 bindings (compiled from Rust via maturin), FastAPI server, CLI tool, validation, and artifact persistence. No computation happens in Python -- it delegates everything to the Rust core. The Python package is typed (PEP 561) and ships with `.pyi` stubs for IDE support.

- **Frontend** (`frontend/`): A Next.js 16 application with React Three Fiber for 3D graph visualization, Zustand for state management, and shadcn/ui for UI components. The frontend communicates with the Python bridge via an API proxy that keeps the API key server-side.

For the full technical reference, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Performance

RSVS is designed for low-latency operations on knowledge graphs of moderate size. The Rust core eliminates the overhead of interpreted Python for all graph operations, and the PyO3 binding layer adds minimal overhead for cross-language calls.

Representative benchmarks (Apple M2 Pro, Criterion.rs):

| Operation | Time | Notes |
|-----------|------|-------|
| Jaccard similarity (100-element sets) | ~2 us | Atom set comparison |
| NPMI lookup | ~50 ns | Single table lookup |
| Co-occurrence ingest (20 tokens) | ~5 us | Per-sentence |
| Sense ingest (10 atoms) | ~15 us | Sense assignment |
| Full pipeline ingest | ~800 us | Tokenize through autonomy |
| Structural similarity | ~5 us | Compare two nodes' compositions |
| Substitution analysis | ~8 us | Find substitutions |

Run benchmarks yourself:

```bash
cd backend && cargo bench
```

For detailed benchmark methodology and scaling characteristics, see [BENCHMARKS.md](docs/BENCHMARKS.md).

---

## Contributing

We welcome contributions of all kinds -- bug reports, feature requests, documentation improvements, and code changes. RSVS follows Conventional Commits and requires all code to pass formatting checks, linting, and tests before merging.

Key guidelines:

- **Rust**: Run `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, and `cargo test --lib` before every commit.
- **Python**: Run `ruff format .`, `ruff check .`, and `pytest tests/ -v` before every commit.
- **TypeScript**: Run `npm run lint` before every commit.
- **PyO3 bindings**: All new Rust methods exposed to Python must be feature-gated behind `#[cfg(feature = "python")]`.
- **Tests**: Every new feature or bug fix must include tests.

For the full contribution guide, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Dual-licensed under [MIT](LICENSE) OR [Apache-2.0](LICENSE). You may choose either license at your option.

---

## Citation

If you use RSVS in your research, please cite it as follows:

```bibtex
@software{rsvs2026,
  title = {RSVS: Recursive Symbolic Vocabulary System},
  author = {Wolfvin},
  year = {2026},
  version = {8.3.0},
  url = {https://github.com/Wolfvin/SymbolicPuzzle3D}
}
```

The CITATION.cff file at the repository root contains the full citation metadata in CITATION.cff format.
