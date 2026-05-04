<div align="center">

# RSVS

### Relational Symbolic Vocabulary System

**Meaning is compositional — every sense is formed by other senses.**

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=flat-square&logo=typescript)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-green?style=flat-square)](LICENSE)
[![Schema](https://img.shields.io/badge/Schema-v7.0-purple?style=flat-square)]()

[Quick Start](#-quick-start) · [Key Insight](#-the-key-insight) · [Architecture](#-architecture-at-a-glance) · [Features](#-core-features) · [API](#-api-reference) · [Contributing](CONTRIBUTING.md)

</div>

---

## What is RSVS?

RSVS is a compositional symbolic meaning engine that builds structured knowledge graphs from raw text. Its core innovation is that **every sense of every word is defined by compositions** — references to specific senses of other words. This makes meaning fully traceable: you can follow the chain from any concept down to its constituent parts, explain precisely why two concepts are related, and identify the exact substitution that transforms one into another.

Unlike traditional embeddings that tell you "raja and ratu have cosine similarity 0.87," RSVS tells you they share two compositions (tahta_tertinggi, kerajaan) and differ in exactly one (laki_laki vs. perempuan). RSVS is not a replacement for Transformers — it's an interpretation layer on top of them, transforming opaque vector representations into symbolically referenceable ones.

---

## The Key Insight

**raja and ratu are related because they share compositions, not just co-occurrence statistics.**

```python
from rsvs import Rsvs

r = Rsvs()

# Ingest some text
r.ingest("Raja adalah pemimpin kerajaan laki-laki. "
         "Ratu adalah pemimpin kerajaan perempuan. "
         "Tahta tertinggi ada di kerajaan.")

# Define compositional meanings
r.compose("raja", [("tahta_tertinggi", 0), ("laki_laki", 0), ("kerajaan", 0)], "id")
r.compose("ratu", [("tahta_tertinggi", 0), ("perempuan", 0), ("kerajaan", 0)], "id")

# Structural similarity — WHY they're related
sim = r.structural_similarity("raja", "ratu")
print(f"Similarity: {sim.structural_similarity:.3f}")   # 0.667
print(f"Shared: {sim.shared_labels(r)}")                # [(tahta_tertinggi, 0), (kerajaan, 0)]

# Substitution analysis — WHAT transforms one into the other
sub = r.substitution_analysis("raja", "ratu")
print(f"Substitution: {sub.substitution_labels(r)}")    # [(laki_laki, 0, perempuan, 0)]
```

One swap: `laki_laki` → `perempuan`. That's the entire semantic difference between king and queen, expressed as a precise structural transformation.

---

## Quick Start

### Docker (Recommended)

```bash
docker compose up
```

Frontend at `http://localhost:3000`, API at `http://localhost:8000`.

### From Source

**Prerequisites:** Rust 1.75+, Python 3.12+, Node.js 18+, maturin

```bash
# Clone
git clone https://github.com/Wolfvin/SymbolicPuzzle3D.git
cd SymbolicPuzzle3D

# Build Rust core + Python bindings
cd backend/python
pip install maturin
maturin develop

# Start API server
python -m rsvs.fastapi_server

# Start frontend (new terminal)
cd ../../frontend
npm install && npm run dev
```

### Rust Only (No Python)

```bash
cd backend
cargo run --bin rsvs-smoke    # 114+ unit tests + smoke pipeline
```

### pip install

```bash
pip install rsvs
```

---

## Architecture at a Glance

```
┌───────────────────────────────────────────────────────────────┐
│                     Frontend (Next.js 16)                      │
│          React Three Fiber · Zustand · shadcn/ui               │
│                                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│   │  3D Force │  │ Compose  │  │ Appraise │  │  Timeline   │  │
│   │  Graph   │  │  Panel   │  │  Panel   │  │  + HUD      │  │
│   └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │
└─────────────────────────┬─────────────────────────────────────┘
                          │ HTTP (POST /run, GET /latest)
┌─────────────────────────▼─────────────────────────────────────┐
│               Python Bridge (FastAPI + PyO3)                    │
│                                                                 │
│   ┌────────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  │
│   │  fastapi_  │  │  modes.py │  │validation│  │conversion│  │
│   │  server.py │  │ (dispatch)│  │  .py     │  │  .py     │  │
│   └────────────┘  └─────┬─────┘  └──────────┘  └──────────┘  │
│                         │                                       │
│              ┌──────────▼──────────┐                            │
│              │   rsvs_core.py      │                            │
│              │ (Rust core wrapper) │                            │
│              └──────────┬──────────┘                            │
└─────────────────────────┼──────────────────────────────────────┘
                          │ PyO3 FFI
┌─────────────────────────▼──────────────────────────────────────┐
│                    Rust Core (rsvs-core)                         │
│                                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│   │ pipeline │  │attention │  │ autonomy │  │   sense.rs   │  │
│   │ compose  │  │  .rs     │  │  .rs     │  │ (composit-   │  │
│   │ query    │  │ NPMI +   │  │ EMA +    │  │  ional v5.0) │  │
│   │ ingest   │  │ Jaccard  │  │ hysteresis│  │              │  │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│   │ graph.rs │  │ seed.rs  │  │ persist  │  │  events.rs   │  │
│   │structural│  │(24 atoms)│  │  .rs     │  │ (stream)     │  │
│   │sim + sub │  │          │  │          │  │              │  │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Features

### Compositional Meaning

Every sense is defined by compositions — references to specific senses of other nodes. This is the structural definition of meaning: not a statistical artifact, but a precise specification of what a sense means in terms of other senses.

```rust
// Rust: Compose "raja" from explicit references
let compositions = vec![
    CompositionRef::new(tahta_tertinggi_id, 0),
    CompositionRef::new(laki_laki_id, 0),
    CompositionRef::new(kerajaan_id, 0),
];
rsvs.compose("raja", compositions, Some("id"))?;
```

```python
# Python: Same operation via PyO3
node_id = r.compose("raja", [
    ("tahta_tertinggi", 0),
    ("laki_laki", 0),
    ("kerajaan", 0)
], "id")
```

### Structural Similarity

Compare concepts by shared/differing compositions, not just co-occurrence statistics. Structural similarity produces an explicit decomposition: what they share, what differs, and the overall score.

```python
sim = r.structural_similarity("raja", "ratu")
# StructuralSimResult:
#   structural_similarity: 0.667
#   shared_compositions: [(tahta_tertinggi, 0), (kerajaan, 0)]
#   only_a_compositions: [(laki_laki, 0)]
#   only_b_compositions: [(perempuan, 0)]
#   layer_a: 2, layer_b: 2
```

### Substitution Analysis

Find the precise swap that transforms one concept into another. This goes beyond saying "they're similar" — it tells you exactly what needs to change.

```python
sub = r.substitution_analysis("raja", "ratu")
# SubstitutionResult:
#   structural_similarity: 0.667
#   substitutions: [(laki_laki, 0) → (perempuan, 0)]
#   unpaired_only_a: []
#   unpaired_only_b: []

# Get human-readable labels
labels = sub.substitution_labels(r)
# [("laki_laki", 0, "perempuan", 0)]
```

### Sense Induction

When text is ingested, RSVS doesn't just record co-occurrences — it induces compositional senses. For each token in context, the system identifies which senses of other tokens are active and uses them as the compositions of a new sense. This is the mechanism that creates structured meaning from raw text.

```python
stats = r.ingest("Raja adalah pemimpin kerajaan.")
# IngestStats(sentences=1, atoms_promoted=3, senses_created=3, compositions=6)

# Check induced senses
senses = r.senses("raja")
for s in senses:
    print(f"Sense {s.sense_idx}: layer={s.layer}, "
          f"grounding={s.grounding_score:.2f}, "
          f"compositions={s.compositions}")
```

### Composition Grounding

After a sense is formed, RSVS verifies its compositions against future evidence. Confirming contexts boost the grounding score; contradicting contexts penalize it. Compositions that are consistently contradicted become candidates for revision. The asymmetric penalty (0.10) vs. boost (0.05) ensures that poor compositions are caught quickly.

```python
# Grounding is updated automatically during ingestion
stats = r.ingest(more_text)

# Check grounding status
senses = r.senses("raja")
for s in senses:
    verdict = ("WellGrounded" if s.grounding_score >= 0.60
               else "NeedsReview" if s.grounding_score >= 0.20
               else "NeedsRevision")
    print(f"Sense {s.sense_idx}: grounding={s.grounding_score:.2f} ({verdict})")
```

### Transformer Bridge

RSVS is an interpretation layer on top of Transformer architecture. It doesn't replace Transformers — it adds symbolic traceability. Transformers provide the statistical signal; RSVS provides the structural explanation. You can use RSVS alongside any Transformer model.

| Transformer Concept | RSVS Equivalent |
|---------------------|-----------------|
| Attention weight | Hard-attention score (NPMI + Jaccard + Cooc) |
| Dense vectors | Sparse composition references |
| Multi-head attention | Multiple senses per node |
| Cosine similarity | Structural similarity (shared/differing compositions) |
| "They're similar" | "They share X, differ in Y, swap Z transforms A→B" |

### 3D Visualization

Explore your knowledge graph interactively with React Three Fiber. Nodes are rendered as spheres (size by tier, color by status), edges as lines (thickness by weight). Force-directed layout with tier-weighted repulsion keeps the graph readable.

```bash
cd frontend && npm run dev
```

---

## Example Usage

### Complete Working Example

```python
from rsvs import Rsvs

# 1. Create RSVS instance (bootstraps 24 seed atoms)
r = Rsvs(entity_promote_n=3, theta_assign=0.12, n_warm=20, eta=0.1)

# 2. Ingest text — builds co-occurrence stats, promotes entities, induces senses
r.ingest(
    "Raja adalah pemimpin tertinggi kerajaan. "
    "Raja adalah seorang laki-laki. "
    "Ratu adalah pemimpin perempuan kerajaan. "
    "Ratu duduk di tahta tertinggi. "
    "Kerajaan dipimpin oleh raja atau ratu. "
    "Tahta tertinggi simbol kekuasaan kerajaan."
)

# 3. Define compositional meanings explicitly
r.compose("raja", [
    ("tahta_tertinggi", 0),
    ("laki_laki", 0),
    ("kerajaan", 0)
], "id")

r.compose("ratu", [
    ("tahta_tertinggi", 0),
    ("perempuan", 0),
    ("kerajaan", 0)
], "id")

# 4. Query a concept in context
result = r.query("raja", "pemimpin kerajaan laki-laki")
print(f"Active sense: {result.sense_idx}")
print(f"Layer: {result.layer}")
print(f"Compositions: {result.compositions}")

# 5. Compute structural similarity
sim = r.structural_similarity("raja", "ratu")
print(f"\nStructural similarity: {sim.structural_similarity:.3f}")
print(f"Shared compositions: {sim.shared_labels(r)}")
print(f"Only in raja: {sim.only_a_compositions}")
print(f"Only in ratu: {sim.only_b_compositions}")

# 6. Substitution analysis — the precise swap
sub = r.substitution_analysis("raja", "ratu")
print(f"\nSubstitution: {sub.substitution_labels(r)}")
print(f"This single swap transforms 'king' into 'queen'.")

# 7. Appraise text against the graph
appraisal = r.appraise("Raja dan ratu memimpin kerajaan")
print(f"\nAppraisal: {appraisal.verdict} ({appraisal.agree_pct:.0f}% agree)")

# 8. Find related concepts
relations = r.relate("raja")
print(f"\nRelated nodes: {relations.node_labels(r)[:5]}")

# 9. Inspect node details
info = r.node_info("raja")
print(f"\nNode: {info.label}, Layer: {info.layer}, "
      f"Confidence: {info.confidence:.2f}, Status: {info.status}")

# 10. Persist state
r.save("my_knowledge_graph.json")
```

### cURL Examples

```bash
# Ingest text
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Raja adalah pemimpin kerajaan."}'

# Compose a node
curl -X POST http://localhost:8000/compose \
  -H "Content-Type: application/json" \
  -d '{
    "label": "raja",
    "compositions": [
      {"label": "tahta_tertinggi", "sense_id": 0},
      {"label": "laki_laki", "sense_id": 0},
      {"label": "kerajaan", "sense_id": 0}
    ],
    "lang": "id"
  }'

# Structural similarity
curl "http://localhost:8000/structural-similarity?a=raja&b=ratu"

# Substitution analysis
curl "http://localhost:8000/substitution-analysis?a=raja&b=ratu"
```

---

## Performance

Benchmarks on Apple M2 Pro with Criterion.rs:

| Benchmark | Time | Notes |
|-----------|------|-------|
| `jaccard_100_elements` | ~2 µs | Atom set similarity for 100-element sets |
| `npmi_lookup` | ~50 ns | Single NPMI table lookup |
| `cooc_ingest_sentence_20_tokens` | ~5 µs | Co-occurrence stats for 20-token sentence |
| `sense_ingest_10_atoms` | ~15 µs | Sense assignment for 10-atom context |
| `pipeline_ingest_text` | ~800 µs | Full pipeline: tokenize → attention → sense → autonomy |
| `structural_similarity` | ~5 µs | Compare two nodes' compositions |
| `substitution_analysis` | ~8 µs | Find substitutions between two nodes |

Run benchmarks yourself:

```bash
cd backend && cargo bench
```

### Complexity at a Glance

| Operation | Complexity | Parallelism |
|-----------|-----------|-------------|
| Ingest (per token) | O(S × K) | Sentence-level via rayon |
| Structural similarity | O(S_a × S_b × C) | — |
| Substitution analysis | O(S_a × S_b × C) | — |
| Relate | O(N × \|A\|) | Full rayon parallelism |
| Query | O(S × K) | — |

Where S = senses, K = core atoms, C = compositions, N = total nodes, A = atom set size.

---

## Roadmap

### v6.0 — Distributed Cognition
- [ ] Distributed Rust core with Raft consensus for multi-node deployments
- [ ] Plugin system for custom attention scorers and policy rules
- [ ] Streaming events via WebSocket/SSE for real-time UI updates
- [ ] Multi-domain knowledge graphs with cross-domain edges
- [ ] Graph embeddings for approximate nearest-neighbor search

### v7.0 — Language & Reasoning
- [ ] LLM integration for guided knowledge extraction
- [ ] Cross-lingual composition alignment (raja@id ↔ king@en ↔ roi@fr)
- [ ] Export to RDF/OWL for interoperability with semantic web
- [ ] Analogical reasoning via composition pattern matching
- [ ] Composition-driven text generation (controlled by structural constraints)

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development environment setup
- Code style guidelines (Rust, Python, TypeScript)
- Commit message convention (Conventional Commits)
- PR process and testing requirements
- How to add new modes and extend the Rust core

### Development Setup

```bash
# Rust unit tests (114+ tests)
cd backend && cargo test --lib

# Rust benchmarks
cd backend && cargo bench

# Python tests
cd backend/python && pytest tests/ -v

# Full pipeline smoke test
cd backend && cargo run --bin rsvs-smoke

# Frontend tests
cd frontend && npm test
```

---

## License

Dual-licensed under [MIT](LICENSE) OR [Apache-2.0](LICENSE). You may choose either license.

---

## Acknowledgments

RSVS was inspired by research in compositional semantics, symbolic AI, and the desire to make neural representations interpretable. Special thanks to:

- The Rust community for zero-cost abstractions that make compositional graph operations fast
- The PyO3 project for seamless Rust-Python interop
- The React Three Fiber team for making 3D visualization accessible
- The Indonesian language community for providing rich compositional examples (raja/ratu, tahta/kerajaan)

---

<div align="center">

**[Report a Bug](https://github.com/Wolfvin/SymbolicPuzzle3D/issues/new?template=bug_report.md)** ·
**[Request a Feature](https://github.com/Wolfvin/SymbolicPuzzle3D/issues/new?template=feature_request.md)** ·
**[Read the Architecture](ARCHITECTURE.md)** ·
**[Read the API Docs](docs/API.md)**

</div>
