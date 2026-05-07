# RSVS Architecture v8.3 — Language-Agnostic Compositional Symbolic Meaning

> Technical reference for the Recursive Symbolic Vocabulary System, v8.3

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Rust Core Architecture](#3-rust-core-architecture)
4. [Python Bridge Architecture](#4-python-bridge-architecture)
5. [Next.js Frontend](#5-nextjs-frontend)
6. [Data Model](#6-data-model)
7. [Key Algorithms](#7-key-algorithms)
8. [Build & Distribution System](#8-build--distribution-system)
9. [Performance Characteristics](#9-performance-characteristics)
10. [Security Architecture](#10-security-architecture)
11. [API Surface](#11-api-surface)

---

## 1. Overview

The Recursive Symbolic Vocabulary System (RSVS) is a compositional symbolic meaning engine built on the fundamental thesis that **meaning is structural, not statistical**. When RSVS says that "raja" (king) and "ratu" (queen) are related, it does not express this as a cosine similarity between opaque vectors — it says they share two out of three compositions (`tahta_tertinggi`, `kerajaan`) and differ in exactly one (`laki_laki` vs. `perempuan`). Every dimension of meaning can be traced back to its constituent senses, and every relationship between concepts can be explained as shared or differing compositions. The system builds a structured knowledge graph where each node can have multiple senses, each sense is defined by its compositions — pairs of `(NodeId, SenseId)` that collectively form the meaning — and this recursive structure means meaning is never atomic beyond the seed layer; it is always a composition of other meanings already in the system.

RSVS is **not a replacement for Transformer architecture**. It is an interpretation layer on top of it. Transformers produce dense vector representations that are powerful but opaque. RSVS transforms those abstract numbers into symbolically referenceable representations where every dimension of meaning can be traced back to its constituent senses. The TransformerBridge module provides the integration point, allowing RSVS to operate alongside any Transformer model: the Transformer handles pattern recognition at scale, and RSVS provides the symbolic traceability layer that makes results interpretable, auditable, and compositional. RSVS is also not a traditional knowledge graph with hand-curated ontologies — all meaning emerges from data through the ingest pipeline, with only 24 epistemological seed atoms forming the axiomatic foundation.

Version 8.3 represents the maturation of the language-agnostic architecture introduced in v8.0. The key breakthrough of v8.0 was the ConvergenceEngine: the system does not need to know that "anjing" is Indonesian and "dog" is English — it only needs to observe that their sense compositions are structurally similar and they never co-occur in the same text. When convergence is detected, `LanguageLink` records with type `structural_equivalence` are created automatically. v8.1 added throttled pair evaluation (`max_pairs_per_run = 500`) to prevent O(N²) blowup on large graphs, confidence-prioritized node ordering, and surface label separation from structural labels. v8.2 added convergence contributor metadata to query and appraise results, plus export/import of detected convergence pairs for persistence. v8.3 consolidates these features with performance tuning, expanded PyO3 bindings (30+ Python-visible classes and methods), and improved error recovery through the DEPS planner.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NEXT.JS FRONTEND (Optional/Demo)                  │
│  React 19 · Next.js 16 · R3F (React Three Fiber) · Zustand · shadcn/ui   │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ GraphScene │  │ LeftInputRail│  │ RightNode    │  │ AppraisePanel   │  │
│  │    3D      │  │ (Query/Ingest)│  │ Drawer       │  │ RelatePanel     │  │
│  │ ForceGraph │  │              │  │ (Node Detail) │  │ ComposePanel    │  │
│  └─────┬──────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│        │                │                  │                    │            │
│        └────────────────┴──────────────────┴────────────────────┘            │
│                              │  REST API (JSON)                             │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────────────┐
│                     PYTHON BRIDGE (HTTP + Validation)                       │
│  FastAPI · Uvicorn · PyO3 FFI · CLI · Benchmarks                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐   │
│  │fastapi_  │  │ modes.py │  │conversion│  │validation│  │ artifacts  │   │
│  │server.py │  │ (9 modes)│  │  .py     │  │  .py     │  │   .py      │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬──────┘   │
│       │              │              │              │              │           │
│  ┌────┴─────┐  ┌─────┴──────┐  ┌───┴──────┐  ┌───┴──────┐  ┌───┴────────┐ │
│  │api/      │  │rsvs_core.py│  │protocols │  │cli.py    │  │ eval.py    │ │
│  │routes/   │  │ (singleton │  │ .py      │  │(11 subs) │  │(5 suites)  │ │
│  │schemas/  │  │  manager)  │  │(Protocol)│  │          │  │            │ │
│  │middleware│  │            │  │          │  │          │  │            │ │
│  └──────────┘  └─────┬──────┘  └──────────┘  └──────────┘  └────────────┘ │
│                       │  PyO3 FFI (rsvs._rsvs native extension)            │
└───────────────────────┼─────────────────────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────────────────────┐
│                   RUST CORE (All Computation, No I/O)                       │
│  rsvs-core crate · Cargo workspace · Serde · Rayon · twox-hash             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  pipeline/ — Rsvs orchestrator                                      │    │
│  │  ┌────────┐ ┌───────┐ ┌────────┐ ┌─────────┐ ┌──────────┐         │    │
│  │  │ingest  │ │query  │ │compose │ │traverse │ │snapshot  │         │    │
│  │  └────────┘ └───────┘ └────────┘ └─────────┘ └──────────┘         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ graph.rs │ │attention │ │ sense.rs │ │autonomy  │ │  mcts.rs         │  │
│  │(nodes,   │ │  .rs     │ │(SenseMgr,│ │(lifecycle│ │(UCB1 tree search)│  │
│  │ edges,   │ │(spreading│ │ induction│ │ tiers,   │ │                  │  │
│  │ adjacency│ │  activ.) │ │ grounding│ │ EMA conf)│ │                  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │
│                                                                             │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │consolidation│ │reflection│ │converge  │ │neurosym  │ │composition   │   │
│  │  .rs       │ │  .rs     │ │  nce.rs  │ │  .rs     │ │_index.rs     │   │
│  │(4-phase    │ │(REVISE/  │ │(struct.  │ │(5 verify │ │(O(1) reverse │   │
│  │ cleanup)   │ │ RETIRE)  │ │ equiv.)  │ │ rules)   │ │ lookup)      │   │
│  └────────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
│                                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐     │
│  │deps.rs   │ │thinking  │ │paradigm  │ │spreading │ │matryoshka.rs │     │
│  │(DEPS     │ │  .rs     │ │  .rs     │ │  .rs     │ │(multi-grain  │     │
│  │ recovery)│ │(fast/    │ │(5-level  │ │(energy   │ │ traversal)   │     │
│  │          │ │ deep)    │ │ routing) │ │ activ.)  │ │              │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘     │
│                                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐     │
│  │seed.rs   │ │transform │ │persist.rs│ │events.rs │ │bindings.rs   │     │
│  │(24 seed  │ │er_bridge │ │(JSON     │ │(append-  │ │(PyO3 30+     │     │
│  │ atoms)   │ │  .rs     │ │ save/    │ │ only log)│ │ classes/     │     │
│  │          │ │(xformer  │ │ load)    │ │          │ │ methods)     │     │
│  │          │ │ config)  │ │          │ │          │ │              │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘     │
│                                                                             │
│  ┌──────────┐ ┌──────────┐                                                 │
│  │types.rs  │ │error.rs  │                                                 │
│  │(Node,    │ │(RsvsError│                                                 │
│  │ Edge,    │ │ enum)    │                                                 │
│  │ Sense,   │ │          │                                                 │
│  │ CompRef) │ │          │                                                 │
│  └──────────┘ └──────────┘                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Rust Core Architecture

The Rust core (`backend/crates/rsvs-core/`) is the computational heart of RSVS. It performs all graph operations, sense induction, attention scoring, traversal, verification, and persistence. It has zero I/O or HTTP dependencies — all external communication flows through the Python bridge via PyO3 FFI.

### 3.1 `lib.rs` — Entry Point and Re-exports

The crate root declares all public modules (22 modules total) and re-exports the primary types and structs that consumers need. Key re-export groups:

- **Pipeline**: `Rsvs`, `PipelineConfig`, `IngestStats`, `QueryResult`, `AppraiseResult`, `RelateResult`, `PipelineStatus`, `traverse_query`
- **Graph**: `RsvsGraph`, `SimilarityResult`, `StructuralSimResult`, `SubstitutionResult`, `jaccard_sets`
- **Types**: `NodeId`, `SenseId`, `AtomSet`, `CompositionRef`, `Node`, `Edge`, `NodeStatus`, `Tier`, `EdgeSource`, `TraversalConfig`, `HaltReason`, `ContextQueryResult`, `LanguageLink`, `SemanticMeta`, `CompressionState`, `Fingerprint`, `PolicyMeta`
- **Attention**: `CoocStats`, `EntityDetector`, `RsvsAttention`, `AttentionConfig`, `DomainAttentionConfig`, `AttentionComponent`
- **Autonomy**: `AutonomyEngine`, `AutonomyConfig`, `AtomRecord`, `MemoryClass`, `ConfidenceUpdateResult`, `StatusTransitionResult`
- **Convergence**: `ConvergenceEngine`, `ConvergenceConfig`, `ConvergencePair`
- **DEPS**: `DEPSPlanner`, `DEPSResult`, `RecoveryPlan`, `RecoveryAction`, `FailureType`
- **Other**: `ThinkingToggle`, `ParadigmRouter`, `MCTSTraversal`, `MatryoshkaTraversal`, `CompositionIndex`, `SenseReflection`, `SpreadingActivation`, `TransformerBridge`

The `python` feature flag gates the `bindings` module: when compiled with `--features python`, PyO3 bindings are included; otherwise they are omitted for pure-Rust usage.

### 3.2 `pipeline/` — Rsvs Orchestrator

The pipeline module wires all subsystems together into the `Rsvs` struct, which is the main entry point for all operations. It contains six submodules:

| Submodule | Purpose |
|-----------|---------|
| `ingest.rs` | Text ingestion: tokenize → co-occurrence → entity detection → node promotion → sense induction → confidence update |
| `query.rs` | Concept querying with context disambiguation |
| `compose.rs` | Compositional node creation with verification and cycle detection |
| `traverse.rs` | Depth-controlled lazy traversal with halting criteria |
| `modes.rs` | Mode dispatch: `appraise()`, `relate()`, and mode-specific result types |
| `snapshot.rs` | Runtime snapshot and event streaming (v1 API) |

The `Rsvs` struct holds the entire system state: graph, sense managers, autonomy engine, co-occurrence statistics, entity detector, attention scorer, composition index, thinking toggle, paradigm router, consolidation engine, reflection engine, spreading activation, DEPS planner, and convergence engine. The `PipelineConfig` struct centralizes all tunable parameters including attention weights, sense thresholds, autonomy settings, entity promotion thresholds, and traversal configuration.

### 3.3 `graph.rs` — Knowledge Graph

The `RsvsGraph` manages all nodes and edges. It provides:

- **Node operations**: `insert_node()`, `get_node()`, `get_node_mut()`, `remove_node()`
- **Edge operations**: `add_edge()`, `edges_from()`, `edges_to()`
- **Label indexing**: `label_to_id` HashMap for O(1) label → NodeId resolution
- **Similarity**: `similarity()` computes Jaccard similarity on atom sets
- **Structural similarity**: `structural_similarity()` compares nodes at the sense level — shared/differing `CompositionRef` sets
- **Substitution analysis**: `substitution_analysis()` identifies minimum-cost bipartite matching of composition pairs, producing precise `(from, to)` substitution pairs

The graph uses `NodeId` (u32) as the primary identifier, with label-based lookups as a convenience layer. Adjacency is stored as edge lists, enabling efficient edge iteration.

### 3.4 `attention.rs` — Spreading Activation and Domain Configs

The attention module provides three key components:

- **`CoocStats`**: Maintains unigram counts, bigram pair counts, and total counters. Computes NPMI (Normalized Pointwise Mutual Information) for any token pair. The `entity_score(token, alpha, beta)` method computes E(i) = α×C + β×D where C is centrality and D is diversity.
- **`EntityDetector`**: Tracks per-token sentence counts and grounding flags. Tokens appearing in ≥ N sentences that are groundable to seed atoms are promoted to nodes.
- **`RsvsAttention`**: Scores (token, candidate) pairs using `score = α·NPMI + β·Jaccard + γ·cooc` with configurable weights per domain. `DomainAttentionConfig` stores per-domain (alpha, beta, gamma) with an observation count — after 5+ observations, domain-specific weights override global defaults.

### 3.5 `sense.rs` — Multi-Sense Framework

The `SenseManager` manages all senses for a single node. Each sense has:

- **Compositions**: `Vec<CompositionRef>` — the structural definition of the sense
- **Contexts**: `Vec<AtomSet>` — observational evidence supporting the sense
- **Coherence**: Internal consistency score (0.0–1.0)
- **Status**: `Fragile` (1–4 contexts) or `Mature` (5+ contexts)
- **Grounding**: `GroundingEvidence` — full evidence trail with confirming/contradicting context counts and revision count
- **Layer**: Compositional depth
- **Condition label**: Optional annotation for UI display

**Sense induction** (`induce_sense()`) either assigns a context to an existing sense whose compositions overlap sufficiently (≥ `theta_assign`, default 0.12), or creates a new compositional sense. The `lazy_lookup()` method resolves the active sense for a given context by finding the best-matching sense. Proliferation control mechanisms include candidate pruning with a logarithmic threshold, merge for overlapping mature senses, fragile pruning for inactive low-grounding senses, and assignment preference over creation.

### 3.6 `autonomy.rs` — Tiered Memory Lifecycle

The autonomy engine manages node lifecycles and confidence scores:

- **Tier system**: Tier1 (seed/stable, autonomous), Tier2 (candidate, revocable), Tier3 (new/blocked, needs decision)
- **Status lifecycle**: `New → Candidate → Stable → Deprecated` with hysteresis to prevent flip-flopping. A `Quarantine` state catches nodes that flip too often (≥ 3 flips → quarantine).
- **Confidence**: Updated via EMA: `new_conf = (1 - η) · old_conf + η · (freq × coherence)`. Energy constraints limit single-step drops.
- **Policy metadata**: Governance scores, fingerprint dedup, status flip counts, and evidence pools are tracked per node.
- **Pending removals**: Nodes with low confidence but high impact (many dependents) require explicit approval before removal.

### 3.7 `mcts.rs` — Monte Carlo Tree Search

The MCTS module provides AlphaZero-style tree search for complex reasoning queries:

- **UCB1 selection**: `UCB1(child) = Q(child) + c_puct × sqrt(N(parent)) / N(child)` with default `c_puct = 1.414`
- **Structural value function**: `value = grounding × coherence` — no neural networks, deterministic structural quality
- **Random rollout**: Simulations follow composition references to estimate path quality
- **Backpropagation**: Values propagate back up the tree
- **Backtracking**: Paths with value < `min_value` (0.5) are abandoned with penalty (multiply by `backtrack_threshold = 0.3`)
- **Configuration**: `max_simulations = 10`, `max_depth = 4`

### 3.8 `consolidation.rs` — Sense Merging and Edge Pruning

Four-phase periodic cleanup (runs every 50 ingest batches by default):

1. **Remove dead senses**: Fragile + ungrounded + very inactive (≥ 2× `k_fragile`)
2. **Merge similar senses**: Composition overlap ≥ 0.8 within same node (max 5 merges/cycle)
3. **Prune weak edges**: Learned edges with weight < 0.02 (Bootstrap and Composition edges preserved)
4. **Compact atom records**: Purge autonomy records for low-confidence nodes (seeds always preserved)

### 3.9 `reflection.rs` — Self-Correction

The sense reflection engine evaluates all senses at safe checkpoints:

| Action | Trigger | Effect |
|--------|---------|--------|
| **CONFIRM** | Grounding ≥ 0.60 | No action |
| **REVIEW** | Grounding 0.20–0.59 | Monitor; track consecutive reviews |
| **REVISE** | Grounding < 0.20 OR ≥ 3 consecutive REVIEWs | Prune least-grounded composition |
| **RETIRE** | Fragile + ungrounded + inactivity ≥ 100 | Mark for deletion |

Escalation is built in: ≥ 3 consecutive REVIEWs automatically escalates to REVISE. REVISE actions are rate-limited (max 3/cycle) to prevent catastrophic pruning.

### 3.10 `convergence.rs` — Convergent Meaning Path Detection (v8.0)

The convergence engine detects when two nodes from different surface forms have structurally equivalent compositions, indicating they likely represent the same concept across languages or contexts. The key insight: the system does not need linguistic metadata — it observes that compositions are structurally similar (high Jaccard overlap) AND the nodes never co-occur (suggesting they are different surface forms for the same underlying concept).

**Algorithm**:
1. Collect eligible nodes (non-seed, confidence ≥ 0.3, sufficient contexts)
2. Sort by confidence descending (v8.1: prioritizes high-confidence pairs)
3. For each candidate pair (throttled to `max_pairs_per_run = 500`):
   - Skip already-detected pairs
   - Skip pairs with existing `LanguageLink`
   - Check co-occurrence ≤ 1 (likely same concept, not co-occurring)
   - Compute `best_sense_overlap()` — Jaccard on composition sets across all sense pairs
   - If overlap ≥ `min_overlap_threshold` (0.6): create bidirectional `LanguageLink` with type `structural_equivalence`

**Persistence** (v8.2): `export_detected_pairs()` and `import_detected_pairs()` allow saving/restoring detected convergence state across restarts.

### 3.11 `neurosym.rs` — Neuro-Symbolic Verification Bridge

Five deterministic structural verification rules (no neural networks):

| Rule | Weight | Type | Description |
|------|--------|------|-------------|
| `no_self_reference` | 1.0 | Binary | Compositions must not reference the same node they define |
| `layer_consistency` | 0.8 | Soft | Compositions should reference equal or lower layers |
| `grounding_threshold` | 0.7 | Soft | Composition targets should be grounded |
| `frequency_threshold` | 0.5 | Soft | Composition targets should have sufficient frequency |
| `no_circular_chain` | 1.0 | Binary | Transitive closure must not loop back to the node |

Verification produces a `VerificationStatus` (Verified / Partial / NeedsRevision / Failed / Unsure) computed from a weighted average. Iterative verification with revision (max 3 cycles) removes the worst-scoring composition on each failure. Circular chain detection uses depth-first traversal of composition references.

### 3.12 `composition_index.rs` — O(1) Composition Lookup

The `CompositionIndex` provides O(1) reverse lookup from `CompositionRef → dependent nodes`. Given a `(NodeId, SenseId)`, it returns all nodes whose senses reference that composition. This is critical for:
- **Impact analysis**: When a sense is modified, instantly find all dependent nodes
- **DEPS recovery**: Understand the dependency chain before making destructive changes
- **Convergence evaluation**: Efficiently find structurally related nodes

### 3.13 `deps.rs` — Dependency-Aware Entity Promotion

The `DEPSPlanner` provides structured recovery using the Describe-Explain-Plan-Select pattern:

1. **DESCRIBE**: Classify the failure type (SelfReference, CircularChain, TargetNotFound, SenseLimitReached, etc.)
2. **EXPLAIN**: Root cause analysis
3. **PLAN**: Generate alternative approaches with estimated success rates
4. **SELECT**: Choose the best plan by composite score: `0.6 × estimated_success_rate + 0.4 × simplicity`

Recovery actions include `RemoveComposition` (95% success for self-reference), `TryAlternativeSense`, `ReviseCompositions`, `MergeWithExisting`, `UseDifferentParadigm`, and `Skip`. Destructive plans are flagged so callers can make informed decisions.

### 3.14 Other Core Modules

- **`thinking.rs`**: `ThinkingToggle` classifies queries as `NON_THINKING` or `THINKING` based on 5 complexity signals (n_context_atoms, n_senses, target_layer, is_compositional, domain_complexity). ≥ 2/5 signals exceeding thresholds triggers THINKING mode.
- **`paradigm.rs`**: `ParadigmRouter` selects the lightest traversal strategy: Direct (>0.8 confidence, O(1)) → Shallow (>0.5, O(K)) → Standard (>0.3, O(S×K)) → Deep (>0.15, O(S×K^D)) → MCTS (<0.15, O(S×K×sim)). Three-signal routing: confidence baseline, structural adjustment from ThinkingToggle, and domain calibration from empirical success rates.
- **`spreading.rs`**: Energy-based activation along composition edges with per-hop decay. `SpreadingActivation` computes activation levels across the graph from seed nodes.
- **`matryoshka.rs`**: Multi-granularity traversal (Quarter → Half → ThreeQuarters → Full) with variable-depth branching — high-confidence branches continue deeper, low-confidence branches prune early.
- **`seed.rs`**: Bootstraps the 24 epistemological seed atoms: exists, entity, relation, state, change, time, space, cause, effect, context, signal, pattern, memory, attention, value, agent, goal, risk, trust, identity, language, meaning, action, feedback. Supports custom seed sets.
- **`transformer_bridge.rs`**: Integration config for Transformer models — similarity thresholds, max compositions per induced sense, and attention weight usage.
- **`persist.rs`**: JSON-based save/load for full RSVS state. Uses serde for serialization.
- **`events.rs`**: Append-only event log with monotonic sequence numbers, correlation IDs, and timestamps. `API_VERSION` and `SCHEMA_VERSION` are tracked as constants.
- **`types.rs`**: Core type definitions (see Section 6).
- **`error.rs`**: `RsvsError` enum with variants for all error conditions.
- **`bindings.rs`**: PyO3 bindings exposing 30+ Python-visible classes and methods (see Section 11).

---

## 4. Python Bridge Architecture

The Python bridge (`python/rsvs/`) provides HTTP serving, validation, format conversion, and a Pythonic API on top of the Rust core. It adds no computational logic — all computation happens in Rust.

### 4.1 `__init__.py` — Package Entry

Re-exports the primary API surface: the `PyRsvs` class (via `rsvs._rsvs` native extension), version constants, configuration helpers, and utility functions. Acts as the public facade for the `rsvs` Python package.

### 4.2 `_version.py` — Single Source of Truth

```python
__version__ = "8.3.0"
__schema_version__ = "v8.3"
__api_version__ = "v8.3"
```

All modules and configuration read version constants from this file, preventing version drift across the codebase.

### 4.3 `config.py` — Constants and BridgeConfig

Defines all bridge-level constants:

- **`VALID_MODES`**: 9 valid operation modes: `ingest`, `appraise`, `relate`, `compose`, `structural_similarity`, `substitution_analysis`, `grounding_info`, `context_query`, `context_similarity`
- **`SOURCE_TRUST`**: Trust levels for different provenance sources (trusted_seed: 1.0, governance_manual: 0.95, verified_runtime: 0.8, user_input: 0.65, unknown_external: 0.4)
- **`SEED_LABELS`**: Tuple of 24 seed atom labels
- **Policy constants**: `PROMOTION_THRESHOLD = 0.75`, `DEMOTION_THRESHOLD = 0.60`, `QUARANTINE_FLIP_BUDGET = 3`, `MAX_CONFIDENCE_DELTA = 0.12`
- **`BridgeConfig`**: Dataclass with `host` (default: 127.0.0.1), `port` (default: 8000), `atom_dir` (default: ./atom)
- **Helpers**: `iso_now()` for UTC timestamps, `make_id(prefix)` for collision-resistant UUID4 identifiers

### 4.4 `exceptions.py` — 9 Exception Classes

A typed exception hierarchy rooted at `RsvsError`:

| Exception | HTTP Status | Purpose |
|-----------|-------------|---------|
| `RsvsError` | 500 | Base exception |
| `SchemaVersionMismatchError` | 409 | Payload schema version mismatch |
| `SchemaValidationError` | 400 | Node/snapshot fails validation |
| `InvariantViolationError` | 422 | Node invariant violated (e.g., seed without lock) |
| `InvalidModeError` | 400 | Invalid mode specified |
| `RustCoreUnavailableError` | 503 | Rust core not available |
| `NodeNotFoundError` | 404 | Requested node/label not found |
| `CompositionError` | 422 | Compositional operation failed |
| `SenseError` | 422 | Sense-level operation failed |
| `GroundingError` | 422 | Grounding verification failed |

### 4.5 `protocols.py` — RsvsCoreProtocol

A `typing.Protocol` that defines the interface the Python bridge expects from the Rust core. This enables:
- **Type checking**: mypy validates that the PyO3-exposed API matches the protocol
- **Mocking**: Tests can use protocol-compliant mocks without the native extension
- **Documentation**: The protocol serves as the canonical Python-visible API contract

### 4.6 `rsvs_core.py` — Thread-Safe Singleton Manager

Manages a thread-safe singleton instance of the Rust core. Ensures only one `Rsvs` instance exists per process, handles initialization with configurable parameters, and provides safe access from multiple threads (the Rust core is `Send + Sync`). The module handles graceful fallback when the native extension is unavailable, raising `RustCoreUnavailableError`.

### 4.7 `modes.py` — 9 Mode Handlers

Dispatches operations to the appropriate Rust core method based on mode:

| Mode | Handler | Returns |
|------|---------|---------|
| `ingest` | `handle_ingest()` | `IngestStats` |
| `appraise` | `handle_appraise()` | `AppraiseResult` |
| `relate` | `handle_relate()` | `RelateResult` |
| `compose` | `handle_compose()` | `NodeId` |
| `structural_similarity` | `handle_structural_similarity()` | `StructuralSimResult` |
| `substitution_analysis` | `handle_substitution_analysis()` | `SubstitutionResult` |
| `grounding_info` | `handle_grounding_info()` | `GroundingEvidence` |
| `context_query` | `handle_context_query()` | `ContextQueryResult` |
| `context_similarity` | `handle_context_similarity()` | `float` |

### 4.8 `conversion.py` — Rust→Python Format Converters

Converts Rust core types to Python-friendly dictionaries and dataclasses. Handles label resolution (NodeId → label), type coercion (enum → string), and nesting flattening. Ensures that API consumers receive human-readable labels rather than raw integer IDs.

### 4.9 `validation.py` — Schema Validation

Validates incoming API payloads against expected schemas. Checks required fields, type constraints, value ranges, and semantic invariants (e.g., composition targets must exist in the graph). Returns structured validation errors with field paths.

### 4.10 `artifacts.py` — File Persistence

Handles writing RSVS artifacts to disk in JSON and JSONL formats:
- **Snapshots**: Full graph state serialized as JSON
- **Event logs**: Append-only JSONL (one event per line) with sequence numbers
- **Reports**: Aggregated statistics as JSON
- **Appraise/Relate results**: Mode-specific outputs

All files follow the naming convention `{type}-{timestamp}Z.{ext}` (e.g., `snapshot-20260422T021801Z.json`).

### 4.11 `corpus.py` — Embedded Indonesian Corpus

Contains an embedded Indonesian language corpus for testing and demonstration. Provides pre-built sentences covering common Indonesian vocabulary, enabling out-of-the-box evaluation without external data.

### 4.12 `ingest_wiki.py` — Corpus Ingestion Pipeline

Provides a pipeline for ingesting larger text corpora (e.g., Wikipedia dumps) into RSVS. Handles batch processing, progress reporting, and checkpoint/restart capabilities for large ingestion jobs.

### 4.13 `eval.py` — 5 Benchmark Suites

Five benchmark suites for evaluating RSVS performance:

1. **Similarity benchmark**: Jaccard and structural similarity accuracy
2. **Substitution benchmark**: Precision/recall of substitution pairs
3. **Attention benchmark**: Attention scoring correlation with human judgments
4. **Grounding benchmark**: Grounding verification accuracy
5. **End-to-end benchmark**: Full ingest → query → compose pipeline

### 4.14 `cli.py` — 11 Subcommand CLI

A command-line interface with 11 subcommands:

| Subcommand | Purpose |
|------------|---------|
| `ingest` | Ingest text from file or stdin |
| `query` | Query a concept with context |
| `appraise` | Appraise text against the graph |
| `relate` | Find related concepts |
| `compose` | Create compositional node |
| `similar` | Compute similarity between concepts |
| `substitute` | Run substitution analysis |
| `snapshot` | Save/load graph snapshots |
| `status` | Show system status |
| `eval` | Run benchmark suites |
| `serve` | Start FastAPI server |

### 4.15 `fastapi_server.py` — FastAPI App Wiring

Configures the FastAPI application with:
- CORS middleware
- Rate limiting (via slowapi)
- Exception handlers mapping RSVS exceptions to HTTP status codes
- Route inclusion from `api/routes/`
- Lifespan management (initialize/teardown Rust core)
- OpenAPI schema customization

### 4.16 `api/` — Routes, Schemas, Middleware, Auth

The `api/` package provides the HTTP layer:

- **`api/routes/core.py`**: Core operations — ingest, query, compose, snapshot
- **`api/routes/analysis.py`**: Analysis operations — similarity, substitution, context query
- **`api/routes/maintenance.py`**: Maintenance — consolidation, reflection, convergence detection
- **`api/schemas.py`**: Pydantic models for request/response validation
- **`api/middleware.py`**: Request logging, timing, error handling
- **`api/deps.py`**: FastAPI dependency injection (core instance, auth)

---

## 5. Next.js Frontend

The frontend (`frontend/`) is an optional demo/visualization layer that provides a 3D interactive graph explorer. It is not required for RSVS operation — the system is fully functional via the Python CLI or API alone.

**Technology stack**: React 19, Next.js 16, React Three Fiber (R3F) for 3D rendering, Zustand for state management, shadcn/ui for component library, Tailwind CSS 4 for styling.

**Key components**:
- `GraphScene3D`: Three.js scene with force-directed 3D graph layout
- `ForceGraph`: Physics-based node positioning with attraction/repulsion
- `GraphNode` / `GraphEdge`: Individual graph element renderers with hover/click interaction
- `GraphHUD`: Heads-up display showing graph statistics
- `LeftInputRail`: Input panel for queries, ingest, and compose operations
- `RightNodeDrawer`: Detail drawer for selected node information
- `AppraisePanel`, `RelatePanel`, `ComposePanel`: Mode-specific operation panels
- `TimelineBar`: Temporal navigation of graph state

**Backend bridge**: `backendBridge.ts` handles REST API communication with the Python bridge, including proxy authentication for production deployments.

---

## 6. Data Model

### Core Types

```rust
pub type NodeId = u32;       // 4 bytes — compact node identifier
pub type SenseId = u32;      // Sense index within a node
pub type AtomSet = Vec<NodeId>;  // Set of node IDs for similarity/attention
```

### Node

```rust
pub struct Node {
    pub id: NodeId,
    pub label: String,           // Canonical label (e.g., "raja")
    pub surface_label: String,   // Display form (e.g., "raja", "dog") — no language tag
    pub kind: String,            // Always "node" in v8.3
    pub tier: Tier,              // Tier1/Tier2/Tier3
    pub confidence: f32,         // 0.0–1.0
    pub status: NodeStatus,      // New/Candidate/Stable/Deprecated/Quarantine
    pub is_seed: bool,
    pub is_locked: bool,
    pub semantic: SemanticMeta,
    pub policy_meta: Option<PolicyMeta>,
    pub language_links: Vec<LanguageLink>,
    pub atoms: AtomSet,
    pub fingerprint: Option<Fingerprint>,
}
```

### Edge

```rust
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub weight: f32,                // 0.0–1.0
    pub source: EdgeSource,         // Bootstrap/Learned/Composition
    pub last_reinforced_batch: usize, // For weight decay tracking
}
```

### Sense

```rust
pub struct Sense {
    pub id: SenseId,
    pub compositions: Vec<CompositionRef>,  // Structural definition
    pub contexts: Vec<AtomSet>,             // Observational evidence
    pub coherence: f32,                     // Internal consistency
    pub status: SenseStatus,                // Fragile/Mature
    pub grounding: GroundingEvidence,       // Verification trail
    pub layer: u32,                         // Compositional depth
    pub condition_label: Option<String>,    // UI annotation
}
```

### CompositionRef

```rust
pub struct CompositionRef {
    pub node_id: NodeId,    // Target node
    pub sense_id: SenseId,  // Target sense within that node
}
```

### Supporting Types

```rust
pub struct SemanticMeta {
    pub compression_state: CompressionState,  // Raw/Compressed
    pub layer: u32,                           // Compositional depth
    pub derived_from_node_ids: Vec<NodeId>,   // Backward compat
    pub compression_reason: Option<String>,
    pub internal_representation: bool,        // v8.0: layer-1 bridge to seeds
}

pub struct LanguageLink {
    pub link_type: String,      // e.g., "structural_equivalence"
    pub target_id: NodeId,
}

pub struct GroundingEvidence {
    pub confirming_contexts: usize,
    pub contradicting_contexts: usize,
    pub last_contradiction: Option<String>,
    pub revision_count: usize,
}

pub struct TraversalConfig {
    pub max_depth: usize,        // Safety net (default: 3)
    pub gamma: f32,              // Stability threshold (default: 0.01)
    pub halt_epsilon: f32,       // Alias for gamma in some contexts
    pub halt_confidence: f32,    // Early halt when max score >= this (default: 0.90)
    pub tau_relevance: f32,      // Relevance gating (default: 0.10)
    pub epsilon_ig: f32,         // Min info gain per depth (default: 0.01)
}
```

### Layer System

| Layer | Contents | Example |
|-------|----------|---------|
| 0 | Seed atoms, primitive entities | exists, entity, laki_laki |
| 1 | Internal representations (compositions reference only seeds) | tahta_tertinggi |
| 2+ | Higher-order recursive compositions | raja (from Layer 1 senses) |

Layer is computed as `max(layer of all composition targets) + 1`. Internal representations (layer 1, all compositions target seeds only) are tagged `internal_representation = true` and serve as the bridge between surface tokens and epistemological primitives.

### Tier System

| Tier | Meaning | Confidence Range |
|------|---------|------------------|
| Tier1 | Autonomous, seed/stable | High (seeds: 1.0) |
| Tier2 | Candidate, revocable | Medium |
| Tier3 | New/blocked, needs decision | Low |

### Event

```rust
pub struct RuntimeEvent {
    pub api_version: String,
    pub schema_version: String,
    pub seq: u64,                       // Monotonic sequence number
    pub correlation_id: String,         // Batch correlation ID
    pub event_type: String,
    pub payload: serde_json::Value,
}
```

---

## 7. Key Algorithms

### 7.1 Sense Induction (NPMI Clustering)

```
FUNCTION induce_sense(node, context_atoms):
    FOR EACH existing_sense IN node.senses:
        overlap = jaccard(existing_sense.core_atoms, context_atoms)
        IF overlap >= theta_assign:          // default: 0.12
            assign context to existing_sense
            update grounding (confirming/contradicting)
            RETURN assigned

    // No matching sense found — create new
    compositions = active_senses_from_context(context_atoms)
    new_sense = Sense(compositions, context_atoms, layer=compute_layer(compositions))
    IF new_sense.compositions.is_empty():
        new_sense.status = Fragile
    node.senses.append(new_sense)
    RETURN created
```

### 7.2 Attention (Weighted Spreading Activation)

```
FUNCTION attention_score(token, candidate, domain_config):
    npmi  = compute_npmi(token, candidate)
    jacc  = jaccard(atom_set(token), atom_set(candidate))
    cooc  = cooccurrence_frequency(token, candidate)

    alpha = domain_config.alpha   // default: 0.40
    beta  = domain_config.beta    // default: 0.35
    gamma = domain_config.gamma   // default: 0.25

    RETURN alpha * npmi + beta * jacc + gamma * cooc
```

Domain-specific configs override global defaults after 5+ observations.

### 7.3 Structural Similarity (Jaccard on Composition Sets)

```
FUNCTION structural_similarity(node_a, node_b):
    best_score = 0.0
    FOR EACH sense_a IN node_a.senses:
        FOR EACH sense_b IN node_b.senses:
            set_a = sense_a.compositions.as_set()
            set_b = sense_b.compositions.as_set()
            score = |set_a ∩ set_b| / |set_a ∪ set_b|
            IF score > best_score:
                best_score = score
                best_pair = (sense_a, sense_b)
    RETURN best_score, shared, only_a, only_b
```

### 7.4 Substitution Analysis (Minimum-Cost Bipartite Matching)

```
FUNCTION substitution_analysis(node_a, node_b):
    (shared, only_a, only_b) = composition_diff(best_sense_a, best_sense_b)
    substitutions = minimum_cost_bipartite_matching(only_a, only_b)
    // Pair compositions by similarity; unpaired remain as "unpaired_only_a/b"
    RETURN structural_similarity, substitutions, unpaired_only_a, unpaired_only_b
```

### 7.5 MCTS (UCB1 Selection + Structural Value)

```
FUNCTION mcts_query(root, config):
    FOR sim = 1 TO config.max_simulations:
        // Selection: follow UCB1 from root to leaf
        node = root
        path = [root]
        WHILE node.has_unexpanded_children():
            node = argmax(child, UCB1(child))
            path.append(node)

        // Expansion: add one new child
        IF node.depth < config.max_depth:
            child = expand_one_child(node)
            path.append(child)

        // Evaluation: structural value = grounding × coherence
        value = evaluate_node(child)

        // Backtracking: penalize low-value paths
        IF value < config.min_value:
            value *= config.backtrack_threshold

        // Backpropagation: update all nodes on path
        FOR n IN path:
            n.visits += 1
            n.total_value += value

    RETURN best_path(root), scored_atoms(root)
```

### 7.6 Convergence Detection

```
FUNCTION detect_convergence(graph, senses, cooc_stats):
    eligible = graph.nodes.filter(n => !n.is_seed AND n.confidence >= 0.3 AND has_mature_sense(n))
    eligible.sort_by(confidence, descending)
    pairs_checked = 0
    results = []
    FOR i IN 0..eligible.len():
        IF pairs_checked >= max_pairs_per_run: BREAK
        FOR j IN (i+1)..eligible.len():
            IF pairs_checked >= max_pairs_per_run: BREAK
            pairs_checked += 1
            (a, b) = (eligible[i], eligible[j])
            IF already_detected(a, b): CONTINUE
            IF has_existing_link(a, b): CONTINUE
            IF cooc_count(a.label, b.label) > 1: CONTINUE
            (overlap, idx_a, idx_b) = best_sense_overlap(senses[a], senses[b])
            IF overlap >= 0.6:
                create_language_link(a, b, "structural_equivalence")
                results.append(ConvergencePair(a, b, overlap, idx_a, idx_b))
    RETURN results
```

### 7.7 Consolidation (4-Phase Cleanup)

```
FUNCTION consolidate(graph, senses, autonomy):
    // Phase 1: Remove dead senses
    FOR EACH (node_id, sense_mgr) IN senses:
        FOR EACH sense IN sense_mgr.senses:
            IF sense.status == Fragile AND sense.grounding.score() < 0.1
               AND sense.inactivity >= 2 * k_fragile:
                remove_sense(sense)

    // Phase 2: Merge similar senses (max 5)
    merges = 0
    FOR EACH (node_id, sense_mgr) IN senses:
        FOR EACH pair (s1, s2) IN sense_mgr.senses.combinations(2):
            IF jaccard(s1.compositions, s2.compositions) >= 0.8 AND merges < 5:
                merge_senses(s1, s2)
                merges += 1

    // Phase 3: Prune weak edges
    edges_to_prune = graph.edges.filter(e =>
        e.source == Learned AND e.weight < 0.02)
    remove_edges(edges_to_prune)

    // Phase 4: Compact atom records
    FOR EACH record IN autonomy.records:
        IF record.confidence < tau_remove AND !record.is_seed:
            remove_record(record)
```

---

## 8. Build & Distribution System

### Cargo Workspace

The Rust codebase is organized as a Cargo workspace:

```
backend/
├── Cargo.toml              # Workspace root
├── crates/
│   └── rsvs-core/          # Main library crate
│       ├── Cargo.toml       # Package config (version 8.3.0)
│       └── src/             # 22+ source modules
└── tests/
    └── integration/         # Integration test crate
```

**Release profile optimizations** (in workspace `Cargo.toml`):
```toml
[profile.release]
opt-level = 3
strip = true
lto = true
codegen-units = 1
```

These settings produce maximally optimized, minimal-size binaries with full link-time optimization and single codegen unit for better inlining.

### maturin Build

The Python wheel is built using maturin, which compiles the Rust crate into a native Python extension:

```toml
# python/pyproject.toml
[tool.maturin]
manifest-path = "../backend/crates/rsvs-core/Cargo.toml"
python-source = "python"
features = ["pyo3/extension-module"]
module-name = "rsvs._rsvs"
```

Build commands:
```bash
# Development build (fast, unoptimized)
maturin develop

# Release build (optimized, for distribution)
maturin build --release

# Publish to PyPI
maturin publish
```

### PyO3 FFI

The `bindings.rs` module (gated by the `python` feature) exposes 30+ Python-visible classes and methods:

| PyO3 Class | Rust Counterpart | Purpose |
|------------|------------------|---------|
| `PyRsvs` | `Rsvs` | Main system class |
| `PyIngestStats` | `IngestStats` | Ingestion results |
| `PyQueryResult` | `QueryResult` | Query results |
| `PyContextQueryResult` | `ContextQueryResult` | Context-aware query results |
| `PySimResult` | `SimilarityResult` | Flat similarity results |
| `PyStructuralSimResult` | `StructuralSimResult` | Structural similarity results |
| `PySubstitutionResult` | `SubstitutionResult` | Substitution analysis results |
| `PyAppraiseResult` | `AppraiseResult` | Appraisal results |
| `PyRelateResult` | `RelateResult` | Relatedness results |
| `PyMCTSResult` | — | MCTS traversal results |
| `PyConsolidationResult` | `ConsolidationResult` | Consolidation results |
| `PyReflectionResult` | — | Reflection cycle results |
| `PyGroundingEvidence` | `GroundingEvidence` | Grounding evidence trail |
| `PySenseInfo` | `Sense` | Sense information |
| `PyNodeInfo` | `Node` | Node information |
| `PyTransformerBridgeConfig` | `TransformerBridgeConfig` | Bridge configuration |
| `PyIngestMetaV1` | — | Stable API metadata |

### Docker

The project includes a `Dockerfile` and `docker-compose.yml` for containerized deployment. The frontend has its own `Dockerfile.frontend`. Nginx reverse proxy configuration is provided in `nginx.conf`.

---

## 9. Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Node lookup by label | O(1) | HashMap `label_to_id` |
| Node lookup by ID | O(1) | HashMap `nodes` |
| Edge iteration from node | O(K) | K = edge count for node |
| Attention scoring | O(T×C) | T = tokens, C = candidates per token |
| Sense induction | O(S) | S = number of existing senses |
| Structural similarity | O(S₁×S₂) | S₁, S₂ = sense counts for two nodes |
| Context query (Direct) | O(1) | Single sense, high confidence |
| Context query (Standard) | O(S×K) | S = senses, K = compositions per sense |
| Context query (MCTS) | O(S×K×sim) | sim = max_simulations (default: 10) |
| Composition reverse lookup | O(1) | CompositionIndex |
| Convergence detection | O(N²) worst case | Throttled to max 500 pairs/run |
| Save/Load | O(N+E+S) | N = nodes, E = edges, S = senses |

### Memory

- **NodeId as u32**: 4 bytes per node reference (vs ~50 bytes for String)
- **CompositionRef**: 8 bytes (two u32s)
- **Edge**: ~20 bytes (from, to, weight, source, batch)
- **Event log**: Bounded to `event_retention` (default: 10,000 events)
- **Sense contexts**: Stored as `Vec<Vec<NodeId>>`, compact atom set references

### Benchmarks

Criterion benchmarks are available in `backend/crates/rsvs-core/benches/rsvs_bench.rs`. Key results on typical hardware:

- Ingest 100 sentences: ~5–15ms
- Context query (Standard): ~0.1–1ms
- Structural similarity: ~0.05–0.5ms
- Consolidation cycle: ~1–10ms (depends on graph size)

Rayon enables parallel processing where applicable (e.g., batch attention scoring across sentences).

---

## 10. Security Architecture

### Input Validation

- **Schema validation** (`validation.py`): All incoming API payloads are validated against Pydantic schemas before reaching the Rust core
- **Label sanitization**: Labels are validated for length and character set before node creation
- **Composition target verification**: All composition references are verified against existing nodes and senses before acceptance

### Structural Integrity

- **Circular chain detection**: The NeuroSymVerifier prevents circular composition chains that could create infinite loops
- **Self-reference prevention**: Compositions cannot reference the node they define
- **Grounding verification**: All compositions must satisfy grounding thresholds
- **Batch rollback**: If total confidence delta exceeds threshold, entire ingest batch is rolled back

### Access Control

- **API authentication**: The `api/deps.py` module provides auth dependency injection for FastAPI routes
- **Rate limiting**: slowapi integration prevents API abuse
- **CORS**: Configurable CORS middleware for frontend access control
- **Proxy auth**: `proxyAuth.ts` in the frontend handles authenticated API proxying

### Data Integrity

- **Content-addressable fingerprints**: `Fingerprint` type uses XxHash64 for deterministic, cross-version-stable content hashing
- **Policy metadata**: Governance scores, status flip counts, and seen fingerprints enable dedup and abuse detection
- **Event sourcing**: Append-only event log with monotonic sequence numbers provides full audit trail
- **Quarantine state**: Nodes that flip-flop between statuses ≥ 3 times are quarantined and require manual intervention

### Rust Memory Safety

All core computation happens in Rust, which provides:
- Memory safety (no buffer overflows, use-after-free, or null pointer dereferences)
- Thread safety (the `Rsvs` struct is `Send + Sync` when accessed through the PyO3 singleton)
- No `unsafe` blocks in production code paths

---

## 11. API Surface

### Rust Core (via PyO3)

#### Core Operations

| Method | Signature | Returns |
|--------|-----------|---------|
| `new()` | `(entity_promote_n=3, theta_assign=0.12, n_warm=20, eta=0.1)` | `PyRsvs` |
| `ingest()` | `(text: &str)` | `PyIngestStats` |
| `ingest_with_meta_v1()` | `(text: &str, domain_id: Option<usize>)` | `PyIngestMetaV1` |
| `query()` | `(concept: &str, context: &str)` | `Option<PyQueryResult>` |
| `context_query()` | `(concept, context_atoms, max_depth?, gamma?, halt_confidence?, tau_relevance?)` | `Option<PyContextQueryResult>` |
| `compose()` | `(label, compositions: Vec<(String, u32)>, lang?)` | `PyResult<u32>` |
| `compose_from_ids()` | `(label, atom_ids: Vec<u32>, lang?)` | `PyResult<u32>` |

#### Analysis Operations

| Method | Signature | Returns |
|--------|-----------|---------|
| `similarity()` | `(a: &str, b: &str)` | `Option<PySimResult>` |
| `structural_similarity()` | `(a: &str, b: &str)` | `Option<PyStructuralSimResult>` |
| `substitution_analysis()` | `(a: &str, b: &str)` | `Option<PySubstitutionResult>` |
| `context_similarity()` | `(a, b, context: Vec<String>)` | `Option<f32>` |
| `appraise()` | `(text: &str)` | `PyAppraiseResult` |
| `relate()` | `(concept: &str)` | `Option<PyRelateResult>` |

#### Inspection Operations

| Method | Signature | Returns |
|--------|-----------|---------|
| `node_info()` | `(label: &str)` | `PyResult<PyNodeInfo>` |
| `atom_info()` | `(label: &str)` | `PyResult<PyNodeInfo>` |
| `senses()` | `(concept: &str)` | `PyResult<Vec<PySenseInfo>>` |
| `nodes()` | `(include_seeds=false)` | `Vec<String>` |
| `confidence_map()` | `()` | `HashMap<String, f32>` |
| `status()` | `()` | `HashMap<String, f64>` |
| `entity_candidates()` | `(top_k=10)` | `Vec<(String, f32)>` |

#### Configuration Operations

| Method | Signature | Returns |
|--------|-----------|---------|
| `set_domain()` | `(domain_id: usize)` | `()` |
| `set_domain_attention()` | `(domain_id, alpha, beta, gamma)` | `()` |
| `set_sense_label()` | `(node_label, sense_idx, label: Option<String>)` | `PyResult<()>` |

#### Persistence & Events

| Method | Signature | Returns |
|--------|-----------|---------|
| `save()` | `(path: &str)` | `PyResult<()>` |
| `snapshot_v1()` | `()` | `PyResult<String>` |
| `consume_events_v1()` | `(after_seq?, limit=500)` | `PyResult<String>` |
| `latest_seq_v1()` | `()` | `u64` |
| `pending_removals()` | `()` | `Vec<u32>` |

### HTTP API (via FastAPI)

#### Core Routes (`/api/core/`)

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/ingest` | POST | `{text, domain_id?}` | `IngestStats` |
| `/query` | POST | `{concept, context}` | `QueryResult` |
| `/context-query` | POST | `{concept, context_atoms, max_depth?, ...}` | `ContextQueryResult` |
| `/compose` | POST | `{label, compositions, lang?}` | `{node_id}` |
| `/snapshot` | GET | — | `Snapshot` |
| `/events` | GET | `?after_seq=&limit=` | `EventBatch` |

#### Analysis Routes (`/api/analysis/`)

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/similarity` | POST | `{a, b}` | `SimResult` |
| `/structural-similarity` | POST | `{a, b}` | `StructuralSimResult` |
| `/substitution` | POST | `{a, b}` | `SubstitutionResult` |
| `/context-similarity` | POST | `{a, b, context}` | `{score}` |
| `/appraise` | POST | `{text}` | `AppraiseResult` |
| `/relate` | POST | `{concept}` | `RelateResult` |

#### Maintenance Routes (`/api/maintenance/`)

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/consolidate` | POST | — | `ConsolidationResult` |
| `/reflect` | POST | — | `ReflectionResult` |
| `/converge` | POST | `{max_pairs?}` | `Vec<ConvergencePair>` |
| `/status` | GET | — | `SystemStatus` |

---

*RSVS v8.3 — Recursive Symbolic Vocabulary System. Architecture document generated from source code analysis.*
