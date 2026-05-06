<div align="center">

# RSVS

### Relational Symbolic Vocabulary System

**Meaning is compositional — every sense is formed by other senses.**

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=flat-square&logo=typescript)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-green?style=flat-square)](LICENSE)
[![Schema](https://img.shields.io/badge/Schema-v8.3-purple?style=flat-square)]()
[![Tests](https://img.shields.io/badge/Tests-178_passing-brightgreen?style=flat-square)]()

[Quick Start](#-quick-start) · [Key Insight](#-the-key-insight) · [Architecture](#-architecture-at-a-glance) · [Core Features](#-core-features) · [Advanced Features](#-advanced-features-v72) · [Security](SECURITY.md) · [API](#-api-reference) · [Contributing](CONTRIBUTING.md)

</div>

---

## What is RSVS?

RSVS is a compositional symbolic meaning engine that builds structured knowledge graphs from raw text. Its core innovation is that **every sense of every word is defined by compositions** — references to specific senses of other nodes. This makes meaning fully traceable: you can follow the chain from any concept down to its constituent parts, explain precisely why two concepts are related, and identify the exact substitution that transforms one into another.

Unlike traditional embeddings that tell you "raja and ratu have cosine similarity 0.87," RSVS tells you they share two compositions (tahta_tertinggi, kerajaan) and differ in exactly one (laki_laki vs. perempuan).

> **RSVS is NOT a replacement for Transformers.** It is an interpretation layer **on top of them**. Transformers provide the statistical signal; RSVS provides the structural explanation. Meaning is compositional (structural), not statistical. You can use RSVS alongside any Transformer model to add symbolic traceability to dense vector representations.

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

One swap: `laki_laki` → `perempuan`. That's the entire semantic difference between king and queen, expressed as a precise structural transformation — not a fuzzy vector distance.

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
cargo run --bin rsvs-smoke    # 175+ unit tests + smoke pipeline
```

### pip install

```bash
pip install rsvs
```

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js 16)                        │
│          React Three Fiber · Zustand · shadcn/ui                  │
│                                                                    │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│   │  3D Force │  │ Compose  │  │ Appraise │  │  Timeline     │  │
│   │  Graph   │  │  Panel   │  │  Panel   │  │  + HUD        │  │
│   └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP via /api/proxy (API key server-side)
┌──────────────────────────▼───────────────────────────────────────┐
│              Python Bridge (FastAPI + PyO3)                        │
│                                                                    │
│   ┌────────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐   │
│   │  fastapi_  │  │  modes.py │  │validation│  │conversion│   │
│   │  server.py │  │ (dispatch)│  │  .py     │  │  .py     │   │
│   └────────────┘  └─────┬─────┘  └──────────┘  └──────────┘   │
│                          │                                         │
│               ┌──────────▼──────────┐                              │
│               │   rsvs_core.py      │                              │
│               │ (Rust core wrapper) │                              │
│               └──────────┬──────────┘                              │
└──────────────────────────┼────────────────────────────────────────┘
                           │ PyO3 FFI
┌──────────────────────────▼────────────────────────────────────────┐
│                    Rust Core (rsvs-core)                           │
│                                                                    │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│   │ pipeline │  │attention │  │ autonomy │  │   sense.rs   │   │
│   │ compose  │  │  .rs     │  │  .rs     │  │ (composit-   │   │
│   │ query    │  │ NPMI +   │  │ EMA +    │  │  ional v5.0) │   │
│   │ ingest   │  │ Jaccard  │  │ hysteresis│  │              │   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│   │ graph.rs │  │ seed.rs  │  │ persist  │  │  events.rs   │   │
│   │structural│  │(24 atoms)│  │  .rs     │  │ (stream)     │   │
│   │sim + sub │  │          │  │          │  │              │   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                                                    │
│   ── v7.0 Modules (Losion Cross-Pollination) ──────────────────  │
│                                                                    │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│   │ paradigm.rs   │  │ spreading.rs  │  │    deps.rs        │   │
│   │ ParadigmRouter│  │ SpreadingAct. │  │ DEPSPlanner       │   │
│   │ Direct→Shallow│  │ Composition   │  │ Describe-Explain  │   │
│   │ →Std→Deep→MCTS│  │ edge spread   │  │ -Plan-Select      │   │
│   └───────────────┘  └───────────────┘  └───────────────────┘   │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│   │ neurosym.rs   │  │ thinking.rs   │  │ consolidation.rs  │   │
│   │ NeuroSymVerif.│  │ ThinkingToggle│  │ ConsolidationEng. │   │
│   │ 5 verify rules│  │ Adaptive depth│  │ Merge+Prune+Comp. │   │
│   └───────────────┘  └───────────────┘  └───────────────────┘   │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│   │ reflection.rs │  │   mcts.rs     │  │ matryoshka.rs     │   │
│   │ SenseReflect. │  │ MCTSTraversal │  │ MatryoshkaTravers.│   │
│   │ CONFIRM/REVIE │  │ UCB1 + back-  │  │ Multi-granularity │   │
│   │ W/REVISE/RET. │  │ tracking      │  │ coarse→fine       │   │
│   └───────────────┘  └───────────────┘  └───────────────────┘   │
│   ┌───────────────┐                                                │
│   │composition_   │  Also: transformer_bridge.rs, error.rs        │
│   │ index.rs      │  bindings.rs (PyO3 — all v7.0 APIs exposed)  │
│   │ O(1) reverse  │                                                │
│   │ lookup        │                                                │
│   └───────────────┘                                                │
└────────────────────────────────────────────────────────────────────┘
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

## Advanced Features (v8.3)

v8.3 completes the language-agnostic architecture: `GROUNDABLE_HINTS` removed (entity promotion relies on structural grounding via co-occurrence with seeds), convergence detection with throttled O(N²) and `detected_pairs` persistence, convergence fusion in all 3 modes (query/appraise/relate), and production-ready Docker deployment. The system is now fully language-agnostic — "anjing" and "dog" converge to the same concept structurally without any language detection or hardcoded string matching.

### ParadigmRouter — Adaptive Traversal Paradigm Selection

Instead of always running the heaviest traversal (full MCTS), the ParadigmRouter dynamically selects the cheapest strategy that will work based on three signals: confidence, structural complexity, and domain calibration.

**Paradigm hierarchy** (lightest → heaviest):

| Paradigm | Depth | Use When | Cost |
|----------|-------|----------|------|
| Direct | 0 | Single sense, confidence > 0.8 | O(1) |
| Shallow | 1 | Few context atoms, conf > 0.5 | O(K) |
| Standard | 2–3 | Multiple senses, conf > 0.3 | O(S×K) |
| Deep | 5 | Complex disambiguation, conf > 0.15 | O(S×K^D) |
| MCTS | 4 | Very complex, conf < 0.15 | O(S×K×sims) |

```python
# ParadigmRouter is used internally by query methods.
# It can also be calibrated per-domain:

# The router automatically records success/failure for domain calibration.
# After enough data, it learns which paradigms work best for each domain.
# You can also force a thinking mode:
r.set_thinking_mode(1)   # 1 = THINKING (force deep traversal)
r.set_thinking_mode(0)   # 0 = NON_THINKING (force shallow traversal)
r.set_thinking_mode(-1)  # -1 = AUTO (default — router decides)
```

**How routing works:**

1. **Confidence signal**: Grounding score of the active sense selects a baseline paradigm
2. **Structure signal**: ThinkingToggle classification can upgrade to at least Standard for complex queries
3. **Domain calibration**: If historical data shows a lighter paradigm succeeds >50% of the time for this domain, use that instead

### SpreadingActivation — Network Activation Through Composition Edges

Energy spreads along composition edges with per-hop decay. Unlike simple graph traversal, spreading follows **structural meaning connections**: if node A's sense is composed from [(B, 0), (C, 0)], then activating A spreads energy to B and C. This is the structural equivalent of semantic priming in cognitive science.

```python
# SpreadingActivation runs internally during relate() and context queries.
# Key parameters (configured in Rust core):
#
#   decay_factor: 0.5    — each hop halves energy
#   max_hops: 3          — activation wave travels up to 3 hops
#   min_energy: 0.01     — nodes below this threshold are not activated
#   max_activated: 50    — maximum nodes returned
#
# Well-grounded seeds get MORE energy (0.5 + 0.5 × grounding_score),
# poorly-grounded seeds get less — ensuring reliable knowledge spreads further.
```

**Key properties:**
- **Additive accumulation**: Multiple paths to the same node reinforce its energy
- **Grounding-adjusted**: Well-grounded seeds propagate more energy
- **Composition edges**: Follows structural meaning, not just co-occurrence

### DEPSPlanner — Structured Failure Recovery

When operations fail (circular compositions, missing targets, grounding failures), instead of just returning an error, DEPS generates **recovery plans** with estimated success rates. The planner follows a 4-step process:

1. **DESCRIBE** — Classify the failure type (SelfReference, CircularChain, TargetNotFound, etc.)
2. **EXPLAIN** — Generate a human-readable explanation of what went wrong
3. **PLAN** — Generate multiple alternative recovery strategies
4. **SELECT** — Choose the best plan by composite score: **60% success_rate + 40% simplicity**

```python
# DEPSPlanner runs automatically inside compose() when verification fails.
# Example: if you try to create a self-referencing composition:
try:
    r.compose("raja", [("raja", 0), ("kerajaan", 0)], "id")
except Exception as e:
    print(e)
    # "Self-reference detected: composition references node 'raja' (id=5).
    #  Recovery: Remove self-referencing composition"
    #  ^ This recovery hint comes from DEPSPlanner!
```

**Recovery actions** include: `RemoveComposition`, `TryAlternativeSense`, `ReduceDepth`, `UseDifferentParadigm`, `ReviseCompositions`, `MergeWithExisting`, `Skip`, and `Retry`.

### NeuroSymVerifier — Composition Verification with 5 Rules

Every composition is verified against five structural rules, now **wired directly into the compose() pipeline**. Failed verifications trigger DEPSPlanner for recovery suggestions.

| Rule | Weight | Threshold | What It Checks |
|------|--------|-----------|----------------|
| `no_self_reference` | 1.0 | 1.0 (binary) | Compositions must not reference the same node they define |
| `layer_consistency` | 0.8 | 0.5 | Compositions should reference equal or lower layers |
| `grounding_threshold` | 0.7 | 0.5 | Composition targets should be grounded |
| `frequency_threshold` | 0.5 | 0.3 | Composition targets should have sufficient frequency |
| `no_circular_chain` | 1.0 | 1.0 (binary) | Transitive closure must not loop back |

```python
# Verify a node's compositions manually
result = r.verify("raja")
# Returns: {
#   "ok": True,
#   "label": "raja",
#   "status": "Verified",        # or "Partial", "NeedsRevision", "Failed"
#   "rules_checked": 5,
#   "rules_passed": 5,
#   "rules_failed": 0,
#   "iterations": 1
# }

# Verify with iterative revision (removes worst compositions until verified)
result = r.verify("raja", max_iterations=3)
```

**Verification statuses:** `Verified` → `Partial{passed, failed}` → `NeedsRevision` → `Failed`

### ThinkingToggle — Adaptive Complexity Toggle

Not every query needs deep traversal. The ThinkingToggle analyzes query complexity signals and selects the right depth automatically:

```python
# Simple queries (1-2 context atoms, single sense) → NON_THINKING
#   - Depth multiplier: 0.5 (halves max_depth)
#   - Higher tau_relevance (fewer expansions)
#   - Fast: O(K)

# Complex queries (4+ context atoms, multiple senses, compositional) → THINKING
#   - Depth multiplier: 1.0 (full max_depth)
#   - Lower tau_relevance (more expansions)
#   - Thorough: O(S × K^D)

# Override manually:
r.set_thinking_mode(-1)  # AUTO (default)
r.set_thinking_mode(0)   # Force NON_THINKING (fast, shallow)
r.set_thinking_mode(1)   # Force THINKING (thorough, deep)
```

**Complexity signals** (2+ of 5 triggers THINKING):
1. ≥3 context atoms
2. ≥2 senses
3. Layer ≥1
4. Compositional target
5. Domain complexity > 0.5

### ConsolidationEngine — Periodic Cleanup

Consolidation is a **separate phase** from ingestion (preventing interference with active learning). It runs at safe checkpoints (every 50 batches by default) and performs thorough cross-node cleanup:

```python
# Manual consolidation (force regardless of interval)
result = r.consolidate(force=True)
print(result)
# ConsolidationResult(merged=2, removed=5, pruned=12, compacted=3)

# Automatic consolidation runs every 50 batches by default.
# Check if consolidation is due:
# result = r.consolidate()  # no-op if not due yet
```

**Four-phase consolidation:**
1. **Remove dead senses** — fragile + ungrounded + very inactive (>2× k_fragile)
2. **Merge similar senses** — Jaccard ≥ 0.8 composition overlap (max 5 merges/cycle)
3. **Prune weak edges** — weight below 0.02 after decay (preserves Bootstrap/Composition edges)
4. **Compact records** — remove autonomy records below tau_remove

### SenseReflection — Self-Evaluation Loop

After each ingest batch, SenseReflection evaluates each sense and produces actions based on grounding evidence:

| Action | Trigger | Effect |
|--------|---------|--------|
| CONFIRM | Grounding ≥ 0.6 | None (sense is healthy) |
| REVIEW | Grounding 0.3–0.6 | Monitor (escalates after 3 consecutive reviews) |
| REVISE | Grounding < 0.3 | Prune worst compositions (max 3 per cycle) |
| RETIRE | Fragile + ungrounded + inactive ≥ 100 contexts | Mark for removal |

```python
# Run a reflection cycle (call periodically, e.g., after every 50 ingests)
result = r.run_reflection()
print(result)
# ReflectionResult(total=12, applied=3)
#   total = 12 actions produced (CONFIRM + REVIEW + REVISE + RETIRE)
#   applied = 3 actions that modified the graph (REVISE + RETIRE only)
```

**Key safety features:**
- REVISE actions are rate-limited (max 3 per cycle) to prevent catastrophic pruning
- REVIEW escalates to REVISE after 3 consecutive cycles
- RETIRE only removes senses that are fragile + ungrounded + very inactive
- CONFIRM and REVIEW are informational (no graph mutations)

### MCTSTraversal — Monte Carlo Tree Search for Complex Disambiguation

For the most complex queries (multi-sense, high layer, compositional chains), MCTS provides tree search with UCB1 selection and backtracking. Instead of neural value/policy networks, RSVS uses structural scores:

- **Policy**: P(a|S,q) from freq_map × edge_weight
- **Value**: grounding score × coherence
- **UCB1**: Balance exploration vs exploitation using visit counts

```python
# MCTS query for complex disambiguation
result = r.mcts_query(
    concept="batu",
    context_atoms=["kekerasan", "mineral", "bentuk"],
    max_simulations=20,    # more = better quality, default: 10
    max_depth=5,           # max depth per simulation, default: 4
)
print(result)
# MCTSResult(sense=0, sims=20, depth=3, halt=Stability, path_len=4)
#   active_sense_idx: 0
#   scored_atoms: [("kekerasan", 0.85), ("mineral", 0.72), ...]
#   best_path: [("batu", 0), ("kekerasan", 0), ...]
#   simulations_run: 20
#   halt_reason: "Stability"
```

**Backtracking**: If a simulation's confidence drops below `min_value` (0.5), the path is abandoned and the value is penalized by `backtrack_threshold` (0.3). This prevents wasted computation on dead-end paths.

### CompositionIndex — O(1) Reverse Lookup

A reverse index from CompositionRef → set of NodeIds that reference it. This replaces O(N×M) scans with single HashMap lookups:

```python
# CompositionIndex runs internally. Key operations:
#   dependents_of_node(id)  → O(1) which nodes reference this node?
#   impact_count(id)        → O(1) how many senses depend on this node?
#   dependencies_of(id)     → O(1) what does this node depend on?
#   rebuild(all_senses)     → Rebuild after bulk operations
```

**Use cases:**
- **Impact analysis**: "If I remove node X, how many senses break?" → `impact_count(X)`
- **Reverse traversal**: "Which nodes use this sense in their compositions?" → `dependents_of_node(X)`
- **Cascade detection**: "What's the blast radius of deleting this?" → `dependencies_of(X)` recursively

### MatryoshkaTraversal — Multi-Granularity Traversal

Inspired by MatFormer's nested submodels, Matryoshka traversal uses variable depth based on query complexity. Different branches of the traversal tree can use different granularities — high-confidence branches go deeper, low-confidence branches stop early.

| Granularity | Depth Multiplier | Use For |
|-------------|-----------------|---------|
| Quarter (0.25) | 25% of max_depth | Simple factual queries |
| Half (0.5) | 50% of max_depth | Disambiguation queries |
| ThreeQuarters (0.75) | 75% of max_depth | Complex compositional queries |
| Full (1.0) | 100% of max_depth | Thorough analysis |

```python
# Matryoshka runs internally via the ParadigmRouter.
# When ParadigmRouter selects Standard/Deep paradigm,
# MatryoshkaTraversal automatically picks the right granularity:
#
#   Simple signal (score 0-1)  → Quarter  (depth ≈ max_depth × 0.25)
#   Moderate signal (score 2-3) → Half     (depth ≈ max_depth × 0.50)
#   Complex signal (score 4-5)  → 3/4      (depth ≈ max_depth × 0.75)
#   Very complex (score 6+)     → Full     (depth = max_depth × 1.0)
#
# This gives you the answer quality of deep traversal,
# but only when you actually need it.
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

# 10. v7.0: MCTS query for complex disambiguation
mcts_result = r.mcts_query("raja", ["kerajaan", "pemimpin"])
print(f"\nMCTS: sense={mcts_result.active_sense_idx}, "
      f"sims={mcts_result.simulations_run}, "
      f"depth={mcts_result.depth_reached}")

# 11. v7.0: Verify compositions
verify_result = r.verify("raja")
print(f"\nVerification: {verify_result}")

# 12. v7.0: Run reflection cycle
reflection = r.run_reflection()
print(f"\nReflection: {reflection.total} actions, {reflection.applied} applied")

# 13. v7.0: Consolidate the graph
consolidation = r.consolidate(force=True)
print(f"\nConsolidation: merged={consolidation.senses_merged}, "
      f"removed={consolidation.senses_removed}, "
      f"pruned={consolidation.edges_pruned}")

# 14. v7.0: Set thinking mode
r.set_thinking_mode(-1)  # AUTO — router decides per query

# 15. Persist state
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

# v7.0: MCTS query
curl -X POST http://localhost:8000/mcts-query \
  -H "Content-Type: application/json" \
  -d '{"concept": "raja", "context_atoms": ["kerajaan", "pemimpin"], "max_simulations": 20}'

# v7.0: Verify compositions
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"label": "raja", "max_iterations": 3}'

# v7.0: Consolidate graph
curl -X POST http://localhost:8000/consolidate \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

---

## API Reference

### Core Operations

| Method | Python | Rust | Description |
|--------|--------|------|-------------|
| Ingest | `r.ingest(text)` | `rsvs.ingest_text(text)` | Ingest text, update graph |
| Ingest+Meta | `r.ingest_with_meta_v1(text, domain_id?)` | — | Ingest with API metadata |
| Query | `r.query(concept, context)` | `rsvs.query(concept, ctx)` | Context-aware query |
| Context Query | `r.context_query(concept, atoms, ...)` | `rsvs.context_query(...)` | Depth-controlled traversal |
| Compose | `r.compose(label, compositions, lang?)` | `rsvs.compose(label, comps, lang)` | Create compositional node |

### Comparison & Analysis

| Method | Python | Description |
|--------|--------|-------------|
| Similarity | `r.similarity(a, b)` | Flat Jaccard similarity (v4 compat) |
| Structural Similarity | `r.structural_similarity(a, b)` | Sense-level composition comparison |
| Substitution Analysis | `r.substitution_analysis(a, b)` | Precise swap transforming A→B |
| Context Similarity | `r.context_similarity(a, b, ctx)` | Context-weighted similarity |
| Appraise | `r.appraise(text)` | Text plausibility against graph |
| Relate | `r.relate(concept)` | Find related nodes and edges |

### v7.0 Advanced Operations

| Method | Python | Description |
|--------|--------|-------------|
| MCTS Query | `r.mcts_query(concept, atoms, sims?, depth?)` | Monte Carlo tree search traversal |
| Run Reflection | `r.run_reflection()` | Self-evaluate all senses (CONFIRM/REVIEW/REVISE/RETIRE) |
| Consolidate | `r.consolidate(force?)` | Periodic cleanup (merge, prune, compact) |
| Set Thinking Mode | `r.set_thinking_mode(mode)` | -1=auto, 0=shallow, 1=deep |
| Verify | `r.verify(label, max_iterations?)` | Neuro-symbolic composition verification |

### Inspection

| Method | Python | Description |
|--------|--------|-------------|
| Node Info | `r.node_info(label)` | Node details (layer, confidence, status) |
| Senses | `r.senses(concept)` | All senses with grounding evidence |
| Nodes | `r.nodes(include_seeds?)` | List all known nodes |
| Entity Candidates | `r.entity_candidates(top_k?)` | Unpromoted high-centrality tokens |
| Confidence Map | `r.confidence_map()` | All node confidence scores |
| Status | `r.status()` | System status dict |

### Domain & Attention

| Method | Python | Description |
|--------|--------|-------------|
| Set Domain | `r.set_domain(domain_id)` | Set current domain tag |
| Set Domain Attention | `r.set_domain_attention(id, α, β, γ)` | Per-domain attention weights |
| Set Sense Label | `r.set_sense_label(node, idx, label)` | Annotate a sense |

### Persistence & Events

| Method | Python | Description |
|--------|--------|-------------|
| Save | `r.save(path)` | Save state to JSON |
| Load | `Rsvs.load(path)` | Load state from JSON |
| Snapshot | `r.snapshot_v1()` | Runtime snapshot for UI |
| Events | `r.consume_events_v1(after?, limit?)` | Incremental event stream |
| Latest Seq | `r.latest_seq_v1()` | Monotonic event sequence number |

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
| MCTS Query | O(S × K × max_simulations) | — |
| Spreading Activation | O(max_hops × frontier_size) | — |
| Composition Index lookup | O(1) | HashMap |
| Consolidation | O(N × S²) | — |

Where S = senses, K = core atoms, C = compositions, N = total nodes, A = atom set size.

---

## Bug Fixes & Security Hardening (v7.0.1 → v7.2.0)

### v7.2.0 — Full Pipeline Integration

| Change | Description |
|--------|-------------|
| **ParadigmRouter → context_query()** | Queries now go through adaptive paradigm selection before ThinkingToggle fine-tunes depth |
| **SpreadingActivation → relate()** | Structural relations now include spreading-activated nodes via composition edges |
| **NeuroSymVerifier → compose()** | Every new composition is automatically verified; failures emit `neurosym_verification_warning` events |

### v7.1.0 — Security Hardening (Score: 77.5 → 97.5/100)

| Vulnerability | Fix | Severity |
|---------------|-----|----------|
| API key exposed to browser | Next.js API proxy route (`/api/proxy/[...path]`) | P0 Critical |
| Stack traces leak to client | Centralized exception handler + bare `raise` | P0 Critical |
| No HTTPS in production | Certbot + nginx SSL + HSTS | P0 Critical |
| Non-atomic persistence writes | tmp+rename pattern in `persist.rs` | P1 Medium |
| Missing CSP header | `Content-Security-Policy` in nginx.conf | P1 Medium |
| IP-only rate limiting | API-key-based rate limiter | P1 Medium |
| Deprecated `bridge_server.py` | Deleted from repo | P1 Medium |

### v7.0.1 — Critical Bug Fixes

| Bug | Fix | Impact |
|-----|-----|--------|
| **Missing PyO3 bindings** | Added `mcts_query`, `run_reflection`, `consolidate`, `set_thinking_mode` to `bindings.rs` | Previously, `fastapi_server.py` called these methods but they would crash with `AttributeError` |
| `unwrap()` in `mcts.rs` | Replaced with safe `if let Some` pattern | Prevented panics on empty search paths |
| `FailureType` missing `Hash` derive | Added `#[derive(Hash)]` to `FailureType` enum in `deps.rs` | Was preventing HashMap usage for DEPS strategies |
| Non-exhaustive match in `deps.rs` | Covered all `RsvsError` variants | Was causing compilation failures |
| NeuroSym + DEPS standalone | Wired into `compose()` pipeline | Composition verification and recovery now run automatically |

---

## Roadmap

### v8.0 — Distributed Cognition

- [ ] Distributed Rust core with Raft consensus for multi-node deployments
- [ ] Plugin system for custom attention scorers and policy rules
- [ ] Streaming events via WebSocket/SSE for real-time UI updates
- [ ] Multi-domain knowledge graphs with cross-domain edges
- [ ] LLM integration for guided knowledge extraction
- [ ] Cross-lingual composition alignment (raja@id ↔ king@en ↔ roi@fr)
- [ ] Export to RDF/OWL for interoperability with semantic web
- [ ] Analogical reasoning via composition pattern matching
- [ ] Composition-driven text generation (controlled by structural constraints)
- [ ] Graph embeddings for approximate nearest-neighbor search

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
# Rust unit tests (175+ tests)
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

- The **Losion** project for the cross-pollination of advanced reasoning patterns (ParadigmRouter, DEPSPlanner, NeuroSymVerifier, ThinkingToggle, MCTS, Matryoshka, SpreadingActivation, SenseReflection, ConsolidationEngine)
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
