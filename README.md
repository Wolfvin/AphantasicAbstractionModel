[![PyPI](https://img.shields.io/pypi/v/rsvs?style=flat-square&logo=pypi&color=3775A9)](https://pypi.org/project/rsvs/)
[![CI](https://img.shields.io/github/actions/workflow/status/Wolfvin/AphantasicAbstractionModel/ci.yml?style=flat-square&logo=github&label=tests)](https://github.com/Wolfvin/AphantasicAbstractionModel/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-8A2BE2?style=flat-square&logo=materialformkdocs&logoColor=white)](https://wolfvin.github.io/AphantasicAbstractionModel/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?style=flat-square&logo=rust)](https://www.rust-lang.org/)

# AphantasicAbstractionModel (AAM)

**Compositional symbolic meaning, not embeddings. Traceable sense definitions with structural similarity.**

*AAM v12 introduces a rule-based cognitive architecture with active enrichment loops, epistemic governance, and role-weighted structural similarity.*

AAM is inspired by **Aphantasia** — the cognitive condition where no visual imagery is stored, only relational structure. This is how AAM remembers: raw input → structured tuples → knowledge graph. No photos. Only relations.

📖 **[Full documentation](https://wolfvin.github.io/AphantasicAbstractionModel/)** · 🚀 [Quick Start](#quick-start) · 📚 [Tutorials](https://wolfvin.github.io/AphantasicAbstractionModel/tutorials/) · 🔧 [API Reference](#api-reference)

---

## What is RSVS?

RSVS (Recursive Symbolic Vector Space) is a compositional symbolic meaning engine that builds structured knowledge graphs from raw text — modeled after how human memory actually stores and retrieves knowledge.

Unlike vector embeddings that compress meaning into opaque floating-point arrays, RSVS represents every concept as a composition of other concepts — and every composition is traceable. You can follow the chain from any concept down to its constituent parts, explain precisely why two concepts are related, and identify the exact substitution that transforms one into another.

At its core, RSVS ingests text through a 14-transform DAG pipeline and builds a knowledge graph composed of semantic atoms, compositions, and typed edges. A **semantic atom** is the smallest unit of meaning — a token, event frame, or hidden meaning candidate unified into one type. A **composition** is a structured group of atoms playing semantic roles (agent, patient, cause, etc.), forming a directed acyclic graph of meaning. Every composition carries dual-axis status: **lifecycle** (structural maturity) and **epistemic** (truth confidence), governed by belief management rules.

RSVS is built with a Rust core compiled to Python via PyO3 and maturin, giving you the safety and speed of Rust with the ergonomics of a Python library. It prioritizes Bahasa Indonesia as its primary language for development and testing; verbalization currently supports Bahasa Indonesia, while the architecture is designed to be language-agnostic for future extension. The system includes an autonomous tiered memory lifecycle (New → Candidate → Stable → Deprecated, plus Quarantine), epistemic governance with seed alignment scores, active gap detection and enrichment loops, three cognitive modes (Reactive, Analytical, Reflective) with an executive orchestrator, and a compositional verbalization engine that generates zero-hallucination explanations.

---

## Why RSVS?

If you have used word embeddings or sentence transformers, you are familiar with the pattern: "raja and ratu have cosine similarity 0.87." But what does that number mean? Which aspects of meaning make them similar? What would you need to change to transform one into the other? Embeddings cannot answer these questions because they compress meaning into a single opaque vector. Knowledge graphs and ontologies offer more structure, but they require manual schema design and struggle with ambiguity, multiple senses, and the fluid nature of natural language meaning.

RSVS occupies a different position. It provides the structural precision of a knowledge graph without requiring upfront schema design, and the fuzzy similarity of embeddings without the opacity. When RSVS computes `similarity("raja", "ratu")` and returns a score, that score is derived from Jaccard overlap of composition neighborhoods plus spreading activation — and you can inspect exactly which compositions are shared and which differ. This structural approach enables semantic queries (find compositions by concept or role structure), path finding (trace the reasoning chain between two concepts), and compositional verification (are the compositions of this sense well-grounded in evidence?).

Compared to traditional knowledge graphs, RSVS does not require you to define a schema or ontology upfront. The system bootstraps from seed atoms and induces compositions automatically from text through a 14-transform pipeline. Compared to ontologies, RSVS handles ambiguity natively through its dual-axis status system — a single composition can exist in different lifecycle and epistemic states simultaneously. Compared to embeddings, RSVS provides full traceability: every similarity score can be decomposed into shared and differing compositions, and every connection can be verbalized as natural language.

---

## Quick Install

Install the Python library from PyPI:

```bash
pip install rsvs
```

The core library has zero Python dependencies. The Rust engine is compiled into the wheel via PyO3, so there is no separate Rust toolchain needed at install time.

For development (includes test tools, linters, and maturin):

```bash
pip install rsvs[dev]
```

To install everything:

```bash
pip install rsvs[all]
```

> **Note:** Python bindings are available via `PyV12Pipeline`. The v8.3 CLI and FastAPI server infrastructure may not reflect the current v12 API.

---

## Quick Start

```python
from rsvs import PyV12Pipeline

pipeline = PyV12Pipeline()
result = pipeline.v12_ingest("Raymond membuat aplikasi karena lambat")
print(f"Created {result.atoms_created} atoms, {result.compositions_created} compositions")

# Inspect the graph
for comp in pipeline.compositions():
    print(f"  {comp.id}: {comp.composition_type} (confidence={comp.confidence:.2f})")

# Detect gaps
gaps = pipeline.detect_gaps()
for gap in gaps:
    print(f"  Gap: {gap.gap_type} - {gap.description}")

# Explain a concept
explanation = pipeline.explain("aplikasi")
print(explanation.text)

# Find related concepts
related = pipeline.find_related("lambat", top_n=5)
for label, energy in related:
    print(f"  {label}: energy={energy:.3f}")

# Compute similarity
score = pipeline.similarity("raja", "ratu")
print(f"Similarity: {score:.3f}")
```

---

## Core Concepts

### SemanticAtom

A `SemanticAtom` is the universal ingest primitive — the only type that enters the RSVS graph. A token, an event frame, a hidden meaning candidate — these are all atoms with varying richness. A token is a sparse atom (no roles). An event frame is a rich atom (roles like Agent, Patient, Cause). A hidden meaning is a derived atom (roles like Problem, Solution). Atoms are classified by `AtomType` (Token, AmbiguousToken, Event, HiddenMeaning, Pattern, Hypothesis, Acquisition) and carry semantic role assignments, polarity, voice, and provenance. The v12 pipeline produces atoms through the Tokenize, ExtractFrame, and ReasonFrame transforms.

### Composition

A `Composition` is the universal structured grouping in the RSVS graph. When a `SemanticAtom` is ingested, it becomes a `Composition`: a group of nodes with typed roles, dual-axis status, and seed alignment scores. This replaces the separate EventFrame, HiddenMeaningCandidate, Pattern, Hypothesis, and SituationState types from earlier versions. Compositions are classified by `CompositionType` (Event, HiddenMeaning, Pattern, Situation, Hypothesis, Acquisition). Each member of a composition is a `CompositionMember` with a node ID, semantic role, confidence, and provenance.

### LifecycleState + EpistemicState

Every composition (and node) carries two orthogonal status axes:

- **LifecycleState** (structural maturity): `New → Candidate → Stable → Deprecated`, plus `Quarantine` for isolated entities. This replaces the old NodeStatus + Tier system.
- **EpistemicState** (truth confidence): `Observed → Inferred → Grounded`, plus `Hypothesis` and `Contradicted`. This replaces the old BeliefState + GroundingVerdict system.

These combine to express rich status: `(Candidate, Inferred)` means "rule-derived, under review"; `(Stable, Grounded)` means "well-established, repeatedly confirmed"; `(Quarantine, Hypothesis)` means "unconfirmed scenario, isolated."

### SemanticEdge

A `SemanticEdge` is a single typed triple with three dimensions: **relation** (what kind: Categorical, Causal, etc.), **role** (optional: if part of a composition), and **source** (provenance: where this edge came from). This replaces the separate RelationType, EdgeSource, SemanticRole, and ProvenanceSource systems from earlier versions with a single edge structure.

### Transform DAG

The v12 pipeline is not a hardcoded sequence — it is a directed acyclic graph of declarative transforms. Each transform declares what it consumes and produces, and the pipeline engine routes data through transforms in topological order with condition-gated execution. This replaces the fixed pipeline stages of earlier versions and enables feedback loops (EnrichComposition, ReExtractFrame) to be wired into the same DAG as the forward path.

### SeedPrimitive + seed_scores

Every composition carries a `seed_scores` map that aligns it against seed primitives — foundational semantic dimensions like AgentiveAction, CausalRelation, and TemporalEvent. These scores replace the old source trust weight system and provide the basis for epistemic governance: compositions with strong seed alignment are promoted faster through the lifecycle, while those with weak alignment require more independent evidence.

---

## Architecture Overview

### 14-Transform DAG Pipeline

RSVS v12 processes text through a condition-gated DAG of 14 transforms, executed in topological order:

```
Tokenize → ExtractFrame → ReasonFrame → IngestAtoms → GovernBeliefs → SeedAnchor
→ DetectGaps → SelectAcquisition → EnrichComposition / ReExtractFrame
→ TemporalDecay → SpreadingActivation → ConvergenceDetection → CompositionalVerbalize
```

| # | Transform | Dependencies | Condition | Purpose |
|---|-----------|-------------|-----------|---------|
| 1 | Tokenize | (none) | always | Extract tokens from raw text |
| 2 | ExtractFrame | Tokenize | is_sentence_like | Rule-based frame extraction (MD-1) |
| 3 | ReasonFrame | ExtractFrame | has_event_atoms | Pre-ingest reasoning (MD-2) |
| 4 | IngestAtoms | Tokenize, ReasonFrame | always | Create nodes and edges |
| 5 | GovernBeliefs | IngestAtoms | always | Lifecycle/epistemic governance (MD-4) |
| 6 | SeedAnchor | GovernBeliefs | always | Compute seed alignment scores (MD-4) |
| 7 | DetectGaps | SeedAnchor | gap_detection_enabled | Find knowledge gaps (MD-6) |
| 8 | SelectAcquisition | DetectGaps | has_gaps | Choose gap-filling strategy (MD-6) |
| 9 | EnrichComposition | SelectAcquisition | has_enrichment_requests | Fill gaps from graph recall |
| 10 | ReExtractFrame | SelectAcquisition | has_reextraction_requests | Re-extract with graph context |
| 11 | TemporalDecay | EnrichComposition | always | Apply Ebbinghaus-style decay |
| 12 | SpreadingActivation | GovernBeliefs | has_event_atoms | Propagate activation energy |
| 13 | ConvergenceDetection | EnrichComposition, TemporalDecay | always | Detect structurally equivalent compositions |
| 14 | CompositionalVerbalize | ConvergenceDetection | always | Generate zero-hallucination explanations |

### Three Cognitive Modes (MD-5)

The `ExecutiveOrchestrator` selects a cognitive mode for each input based on graph neighborhood health:

| Mode | Trigger | Behavior | Enrichment Rounds |
|------|---------|----------|-------------------|
| **Reactive** | No contradictions, no gaps, high confidence | Fast path — just extract and ingest | 0 |
| **Analytical** | Contradictions OR low confidence (<0.5) | Enrichment loop to fill gaps and resolve issues | 1 |
| **Reflective** | Deep contradictions (3+) | Extended reflection with finding analysis | 2 |

The orchestrator runs the enrichment loop (DetectGaps → SelectAcquisition → EnrichComposition → GovernBeliefs) for Analytical and Reflective modes, and produces reflection findings (PromotionCandidate, ContradictionResolvable, StagnantInferred, DecayedConfidence, OverlapDetected) in Reflective mode.

### Codebase Structure

- **Rust Core** (`stage0/layer1/crates/rsvs-core/src/`): All computational logic — graph storage, pipeline engine, transforms, governance, gap detection, convergence, spreading activation, temporal decay, verbalization, and persistence. No HTTP, no file I/O, no Python dependencies. Compiles independently.

- **Python Bridge** (`stage0/layer1/crates/rsvs-core/src/bindings.rs`): PyO3 bindings exposing `PyV12Pipeline` and all v12 types to Python. Compiled via maturin. No computation happens in Python — everything delegates to the Rust core.

- **Frontend** (`_archived/frontend/`): A Next.js application with React Three Fiber for 3D graph visualization.

For the full technical reference, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## API Reference

All operations are accessed through `PyV12Pipeline`, the main Python class wrapping the Rust pipeline engine.

### Core Operations

| Method | Signature | Description |
|--------|-----------|-------------|
| `v12_ingest` | `pipeline.v12_ingest(text: str) -> PyV12IngestResult` | Ingest text through the full 14-transform DAG. Returns stats on atoms created, compositions created, gaps detected, edges, enrichments, governance transitions, and selected cognitive mode. |
| `select_cognitive_mode` | `pipeline.select_cognitive_mode(text: str) -> str` | Select cognitive mode (Reactive/Analytical/Reflective) for the given input based on graph neighborhood health. |
| `save` | `pipeline.save(path: str) -> None` | Serialize the entire knowledge graph to a JSON file. |
| `load` | `pipeline.load(path: str) -> None` | Load a knowledge graph from a JSON file, replacing the current state. |

### Graph Inspection

| Method | Signature | Description |
|--------|-----------|-------------|
| `compositions` | `pipeline.compositions() -> list[PyComposition]` | All compositions in the graph, each with ID, type, members, lifecycle, epistemic, confidence, seed scores, and provenance. |
| `composition_count` | `pipeline.composition_count() -> int` | Total number of compositions. |
| `node_count` | `pipeline.node_count() -> int` | Total number of nodes. |
| `get_composition` | `pipeline.get_composition(id: str) -> PyComposition \| None` | Get a specific composition by its ID. |
| `find_weak_frames` | `pipeline.find_weak_frames() -> list[str]` | Low-confidence Event compositions missing expected roles. Returns composition IDs. |
| `snapshot_json` | `pipeline.snapshot_json() -> str` | JSON snapshot of current graph state for serialization or UI consumption. |
| `graph_summary` | `pipeline.graph_summary() -> str` | Human-readable summary: node count, composition count, lifecycle/epistemic distribution. |

### Gap Detection & Enrichment

| Method | Signature | Description |
|--------|-----------|-------------|
| `detect_gaps` | `pipeline.detect_gaps() -> list[PyKnowledgeGap]` | Detect knowledge gaps in the current graph. Each gap has a type (MissingRole, AmbiguousToken, etc.), description, confidence, and source composition. |
| `pending_gaps` | `pipeline.pending_gaps() -> list[PyKnowledgeGap]` | Get all pending knowledge gaps as structured objects (alias for detect_gaps). |
| `submit_answer` | `pipeline.submit_answer(gap_id: str, answer: str) -> bool` | Submit a user answer to fill a knowledge gap. Returns True if applied. |
| `set_gap_detection` | `pipeline.set_gap_detection(enabled: bool) -> None` | Enable or disable gap detection for subsequent ingest calls. |
| `run_enrichment_loop` | `pipeline.run_enrichment_loop() -> PyV12IngestResult` | Run the active enrichment loop: DetectGaps → SelectAcquisition → EnrichComposition → GovernBeliefs. |

### Semantic Query API

| Method | Signature | Description |
|--------|-----------|-------------|
| `query_concept` | `pipeline.query_concept(concept: str) -> list[tuple[PyComposition, float]]` | Find compositions where the concept appears as a member label. Results ranked by relevance score. |
| `query_structure` | `pipeline.query_structure(role_names: list[str]) -> list[PyComposition]` | Find compositions containing ALL specified semantic roles (e.g., `["Agent", "Cause"]` for causal events). |
| `similarity` | `pipeline.similarity(label_a: str, label_b: str) -> float` | Compute similarity between two concepts using Jaccard composition overlap (60%) + spreading activation cosine (40%). Returns 0.0–1.0. |
| `find_related` | `pipeline.find_related(label: str, top_n: int = 10) -> list[tuple[str, float]]` | Find related concepts using spreading activation. Returns top-N labels with activation energy. |
| `find_path` | `pipeline.find_path(label_from: str, label_to: str) -> list[str]` | Find a reasoning path between two concepts. Returns composition IDs forming the strongest bridge. |
| `explain_connection` | `pipeline.explain_connection(label_from: str, label_to: str) -> list[str]` | Combine find_path with verbalization to produce natural language explanation of why two concepts are related. |
| `compositions_for_label` | `pipeline.compositions_for_label(label: str) -> list[PyComposition]` | All compositions involving a specific node label. Exact match only. |

### Verbalization

| Method | Signature | Description |
|--------|-----------|-------------|
| `explain` | `pipeline.explain(query: str) -> PyVerbalizationResult` | Explain a concept using the Compositional Verbalization Engine. Traverses the graph, builds a reasoning path, and verbalizes each composition. Zero hallucination by design. |
| `verbalize_composition` | `pipeline.verbalize_composition(composition_id: str) -> str \| None` | Verbalize a single composition by its ID. Returns the sentence with epistemic qualifier. |
| `detect_convergence` | `pipeline.detect_convergence() -> list[PyConvergencePair]` | Detect structurally equivalent compositions (high overlap, low co-occurrence). |

### Training

| Method | Signature | Description |
|--------|-----------|-------------|
| `learn_corpus` | `pipeline.learn_corpus(sentences: list[str], priority: str = None) -> PyV12IngestResult` | Ingest multiple sentences with optional priority. High priority runs enrichment after each sentence. |
| `comprehension_check` | `pipeline.comprehension_check(topic: str) -> str` | Check how well the system "understands" a concept: composition count, average confidence, lifecycle distribution, related concepts. |

---

## Examples

### Ingesting and Exploring

```python
from rsvs import PyV12Pipeline

pipeline = PyV12Pipeline()

# Ingest a sentence
result = pipeline.v12_ingest("Raja memimpin kerajaan karena kebijakan")
print(f"Atoms: {result.atoms_created}, Compositions: {result.compositions_created}")
print(f"Cognitive mode: {result.cognitive_mode}")

# Inspect what was created
for comp in pipeline.compositions():
    print(f"  {comp.id}: {comp.composition_type}")
    print(f"    Lifecycle: {comp.lifecycle}, Epistemic: {comp.epistemic}")
    print(f"    Confidence: {comp.confidence:.2f}")
    for member in comp.members:
        print(f"    {member.role}: {member.label} (confidence={member.confidence:.2f})")

# Get a human-readable summary
print(pipeline.graph_summary())
```

### Semantic Similarity and Related Concepts

```python
from rsvs import PyV12Pipeline

pipeline = PyV12Pipeline()
pipeline.learn_corpus([
    "Raja adalah pemimpin kerajaan laki-laki",
    "Ratu adalah pemimpin kerajaan perempuan",
    "Tahta tertinggi ada di kerajaan",
    "Kerajaan dipimpin oleh raja atau ratu",
])

# Compute similarity
score = pipeline.similarity("raja", "ratu")
print(f"Similarity: {score:.3f}")

# Find related concepts
related = pipeline.find_related("raja", top_n=5)
for label, energy in related:
    print(f"  {label}: energy={energy:.3f}")

# Explain why two concepts are related
explanation = pipeline.explain_connection("raja", "kerajaan")
for sentence in explanation:
    print(f"  → {sentence}")
```

### Gap Detection and Enrichment

```python
from rsvs import PyV12Pipeline

pipeline = PyV12Pipeline()
pipeline.set_gap_detection(True)

result = pipeline.v12_ingest("Aplikasi dibuat karena lambat")
print(f"Gaps detected: {result.gaps_detected}")

# Inspect gaps
gaps = pipeline.detect_gaps()
for gap in gaps:
    print(f"  Gap: {gap.gap_type} - {gap.description}")
    if gap.missing_role:
        print(f"    Missing role: {gap.missing_role}")

# Fill a gap with a user answer
if gaps:
    pipeline.submit_answer(gaps[0].gap_id, "Raymond")

# Run the enrichment loop
enrichment = pipeline.run_enrichment_loop()
print(f"Enrichments applied: {enrichment.enrichments_applied}")
```

### Explaining Concepts

```python
from rsvs import PyV12Pipeline

pipeline = PyV12Pipeline()
pipeline.learn_corpus([
    "Raymond membuat aplikasi karena lambat",
    "Aplikasi mempercepat pekerjaan",
    "Lambat menyebabkan masalah",
], priority="high")

# Explain a concept
explanation = pipeline.explain("aplikasi")
print(explanation.text)
print(f"  Confidence: {explanation.avg_confidence:.2f}")
print(f"  Compositions used: {explanation.total_compositions}")
print(f"  Reasoning path: {' → '.join(explanation.path)}")

# Verbalize a specific composition
for comp in pipeline.compositions():
    text = pipeline.verbalize_composition(comp.id)
    if text:
        print(f"  {comp.id}: {text}")
```

### Finding Reasoning Paths

```python
from rsvs import PyV12Pipeline

pipeline = PyV12Pipeline()
pipeline.learn_corpus([
    "Obat menyembuhkan penyakit",
    "Penyakit disebabkan oleh virus",
    "Virus menyebar melalui udara",
])

# Find a reasoning path between two concepts
path = pipeline.find_path("obat", "virus")
print(f"Path: {' → '.join(path)}")

# Explain the connection with natural language
explanation = pipeline.explain_connection("obat", "virus")
for sentence in explanation:
    print(f"  → {sentence}")
```

### Persistence

```python
from rsvs import PyV12Pipeline

# Build a knowledge graph
pipeline = PyV12Pipeline()
pipeline.learn_corpus([
    "Kerajaan dipimpin oleh raja atau ratu",
    "Raja membuat kebijakan untuk rakyat",
])

# Save to disk
pipeline.save("my_graph.json")

# Load later in a different session
pipeline2 = PyV12Pipeline()
pipeline2.load("my_graph.json")
print(pipeline2.graph_summary())
```

### Comprehension Check

```python
from rsvs import PyV12Pipeline

pipeline = PyV12Pipeline()
pipeline.learn_corpus([
    "Batu adalah material keras dari alam",
    "Batu digunakan untuk konstruksi",
    "Mineral adalah komponen batu",
], priority="high")

check = pipeline.comprehension_check("batu")
print(check)
```

---

## Cognitive Foundations

RSVS is not designed from NLP literature alone. It is modeled after how human cognition actually works.

The core observation: humans receive information constantly, but most of it is not retained. Something must *trigger* the connection before a memory becomes accessible. This is not a bug in human cognition — it is a feature. Not everything deserves to be promoted to long-term memory.

RSVS implements this directly:

- **Seed anchoring** (GovernBeliefs + SeedAnchor) — compositions are promoted through the lifecycle only when they have sufficient seed alignment and independent evidence, just as new information only enters long-term human memory if it anchors to existing knowledge
- **Spreading activation** (SpreadingActivation transform) — retrieval works by activation spreading through composition edges, exactly as described by Collins & Loftus (1975) and Anderson (1983)
- **Epistemic governance** (GovernBeliefs) — dual-axis status (lifecycle + epistemic) mirrors how human memory tracks both structural maturity and truth confidence independently
- **Active enrichment** (DetectGaps → SelectAcquisition → EnrichComposition) — the system detects what it does not know and actively seeks to fill gaps, similar to curiosity-driven learning
- **Prediction layer** — RSVS is designed as the symbolic grounding layer for a prediction system (transformer), not as a replacement. The unconscious (RSVS graph) shapes what can be predicted before prediction happens

This is backed by established cognitive science: Predictive Coding (Friston), Global Workspace Theory (Baars 1988), State-Dependent Memory (Radulovic et al.), and recent involuntary memory research (Kobelt et al., 2025).

For the full theoretical foundation, see **[COGNITIVE_FOUNDATIONS.md](docs/COGNITIVE_FOUNDATIONS.md)**.

---

## Performance

RSVS is designed for low-latency operations on knowledge graphs of moderate size. The Rust core eliminates the overhead of interpreted Python for all graph operations, and the PyO3 binding layer adds minimal overhead for cross-language calls.

Representative benchmarks (Apple M2 Pro, Criterion.rs):

| Operation | Time | Notes |
|-----------|------|-------|
| Jaccard similarity (100-element sets) | ~2 us | Composition member comparison |
| NPMI lookup | ~50 ns | Single table lookup |
| Co-occurrence ingest (20 tokens) | ~5 us | Per-sentence |
| Full pipeline ingest | ~800 us | Tokenize through verbalization |
| Similarity (composition overlap) | ~5 us | Compare two nodes' compositions |
| Spreading activation | ~20 us | Single propagation step |

Run benchmarks yourself:

```bash
cd stage0/layer1 && cargo bench
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
  version = {12.0.0},
  url = {https://github.com/Wolfvin/AphantasicAbstractionModel}
}
```

The CITATION.cff file at the repository root contains the full citation metadata in CITATION.cff format.
