# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [8.4.0] - 2026-05-12

### Added — Architecture Gap Closure (37 gaps resolved)

This release closes all 37 gaps identified in the AAM Architecture Gap Analysis Report v8.3.0.

#### Layer 0 — Perceptual Front-End
- **L0-01 [CRITICAL]**: Created `layer0/adapter.py` — bridge between Layer 0 and Layer 1. PerceptualObservation now converts to RSVS ingest operations via `ingest_observation()`.
- **L0-02 [CRITICAL]**: Added `RelationType` enum (Categorical/Differential/Functional/Spatial/Temporal/Causal) to Rust Edge struct in `types.rs`. Six relation types now have full representation in RSVS graph.
- **L0-03 [HIGH]**: Implemented `AudioAbstractor` (Whisper STT pipeline), `ImageAbstractor` (vision bridge), and `VideoAbstractor` (frame sampling + temporal linking). All 4 modalities now functional.
- **L0-04 [HIGH]**: TextAbstractor now has retry with exponential backoff (3 retries, 1s/2s/4s), improved noun-phrase fallback, and in-memory caching.
- **L0-05 [MEDIUM]**: Added 56 unit tests for Layer 0 covering all components.
- **L0-06 [LOW]**: Added `PerceptualTupleMeta` dataclass with `source_url`, `extraction_model`, `extraction_timestamp` fields. Backward compatible with dict input.

#### Layer 1 — Rust Core / RSVS
- **L1-05 [MEDIUM]**: Synchronized SCHEMA_VERSION to "v8.3" across Rust (`events.rs`), Python (`_version.py`), and artifacts. Added version compatibility check in `persist.rs::load()`.
- **L1-06 [MEDIUM]**: Added PyO3 bindings for DEPS Planner — `PyRecoveryPlan` and `PyDEPSResult` classes now accessible from Python.
- **L1-07 [MEDIUM]**: Added `EmbeddingProvider` trait and `embedding_similarity_fallback()`/`embedding_similarity_batch()` to `TransformerBridge`. Pluggable embedding backend support.
- **L1-08 [LOW]**: Verified `SessionGraph` already fully implemented in `session.rs` with isolated RSVS instance per context.

#### Layer 2 — Cognitive Runtime
- **L2-01 [CRITICAL]**: Added 11 missing bridge methods — `mcts_query()`, `consolidate()`, `run_reflection()`, `verify()`, `toggle_thinking()`, `route_paradigm()`, `deps_analyze()`, `matryoshka_traverse()`, `context_similarity()`, `structural_similarity()`, `substitution_analysis()`. All with fallback implementations.
- **L2-02 [HIGH]**: PredictiveEngine now uses `structural_similarity()` for prediction error and `mcts_query()` for complex predictions. Keyword matching remains as fallback.
- **L2-03 [HIGH]**: `_FallbackGraph` now has multi-sense support (`_FallbackSense`), composition references, grounding scores, and coherence calculation.
- **L2-04 [HIGH]**: PatternOutput uses spreading activation (`relate()`), structural similarity, and substitution analysis for pattern completion. 5 strategies instead of 3.
- **L2-05 [MEDIUM]**: Replaced noisy `"The {seed} relates to"` prefix with three-stage soft grounding: ingest_with_meta → appraise-based → minimal prefix.
- **L2-06 [MEDIUM]**: Active sense tracking now uses `consume_events_v1()` for precise timestamps and recency-weighted ranking.
- **L2-07 [MEDIUM]**: Context layer now filters search results with `appraise()` for consistency and `relate()` for relevance before ingestion.
- **L2-08 [LOW]**: Verified — no `rsvs_genius/` duplication exists. Marked resolved.

#### Layer 3 — Reasoning & Output
- **L3-01 [CRITICAL]**: Implemented full `ReasoningEngine` with 5-step deductive chain builder: Extract → Compose → Ground → Explore (MCTS) → Conclude. Produces `DeductiveChain` with full evidence traceability.
- **L3-02 [HIGH]**: PolicyEngine now integrates with RSVS `PolicyMeta` via `check_with_rsvs_policy()` — uses governance_score, status_flip_count, and seen_fingerprints.
- **L3-03 [HIGH]**: CoderLayer now has `analyze_with_rsvs()` method that represents code elements as RSVS nodes with composition references.
- **L3-04 [MEDIUM]**: Added `evidence_node_ids` (NodeId/SenseId tuples) and `grounding_scores` to `ReasoningStep` and `DeductiveStep`. Full traceability from output to evidence nodes.
- **L3-05 [LOW]**: Added 16 unit tests for Layer 3 covering ReasoningEngine, PolicyEngine RSVS integration, CoderLayer RSVS, and evidence traceability.

#### Pipeline — Integration
- **P-01 [HIGH]**: Defined structural data contracts: `PerceptualObservation` (L0→L1), `StructuralDelta` (L1→L2), `ReasoningRequest` (L2→L3). Data flows as typed objects, not raw strings.
- **P-02 [MEDIUM]**: Added Step 6 to `ask()` — appraise self-check. If verdict is "clash"/"disagree", lowers confidence and adds warning. New `appraise_warning` field in AamResponse.
- **P-03 [MEDIUM]**: Added `ask_stream()` async generator yielding `PipelineEvent` after each layer. Supports cancellation. `ask()` remains synchronous.
- **P-04 [MEDIUM]**: Added `source_provenance` parameter through pipeline. `appraise_with_provenance()` weights evidence by source trust.
- **P-05 [LOW]**: Added `maintenance()` and `force_maintenance()` methods. Auto-maintenance every N ingests (default 50).
- **P-06 [LOW]**: Defined `AamError` hierarchy (LayerError, IngestError, ReasoningError, BridgeError, MaintenanceError). Non-fatal errors collected in `AamResponse.errors`.

#### Frontend — UX
- **F-01 [HIGH]**: Replaced O(n²) force simulation with Barnes-Hut octree (O(n log n)). Added spatial grid hashing, frame budget limiting, delta time capping.
- **F-02 [CRITICAL]**: Added composition edges (curved cyan), convergence links (dashed purple), substitution pairs (dotted orange). Multi-sense badge, grounding evidence color dot, 5 new drawer sections for structural info.
- **F-03 [MEDIUM]**: Wrapped `GraphScene3D` with `ErrorBoundary` featuring WebGL context loss detection. Enhanced fallback UI.
- **F-04 [MEDIUM]**: Added label resolution in `backendBridge.ts` — numeric node IDs resolved to labels via API with 5-min TTL cache. Bridge `relate()` now returns (label, score, original_id).
- **F-05 [LOW]**: Re-enabled ESLint rules at "warn": `@typescript-eslint/no-explicit-any`, `@typescript-eslint/no-unused-vars`, `react-hooks/exhaustive-deps`, `no-undef`. Gradual enablement plan documented.

### Fixed
- Moved `import os` from method body to module level in `audio.py` and `image.py`
- Fixed mutable default argument `context: dict = {}` → `context: Optional[dict] = None` in `base.py`

## [8.3.1] - 2026-05-12

### Fixed
- **Infrastructure**: Created `backend → layer1` symlink to fix broken CI/CD, Docker, Makefile, and pip install
- **Security**: Changed default bind address from `0.0.0.0` to `127.0.0.1`
- **Security**: Hashed API keys in rate limiter storage to prevent key leakage
- **Security**: Fixed command injection risk in web_search.py subprocess fallback
- **Security**: Replaced external CDN favicon with local asset
- **Rust Core**: Fixed compose() adding `@en` language tag contradicting language-agnostic design
- **Rust Core**: Fixed hardcoded `tau=0.4` in traverse engine — now uses config value
- **Rust Core**: Fixed panicking `unwrap()` in PyO3 bindings
- **Rust Core**: Fixed composition index duplication on every ingest
- **Rust Core**: Optimized O(N²) similarity to O(N) using HashSet
- **Python**: Wrapped blocking Rust calls with `asyncio.to_thread()` in async handlers
- **Python**: Fixed request size limit bypass via chunked transfer encoding
- **Python**: Added API key authentication to CLI
- **Python**: Fixed test version mismatches (6.0.0 → 8.3.0)
- **Python**: Fixed PEP 621 violation in layer1/pyproject.toml
- **Frontend**: Added React Error Boundaries around major UI sections
- **Frontend**: Fixed demo mode auth bypass — added explicit RSVS_DEMO_MODE flag
- **Frontend**: Fixed setTimeout without cleanup in simulate functions
- **Frontend**: Consolidated duplicated constants (TIER_COLORS, lerp, RSVSMode)
- **Frontend**: Added batch position updates for force layout animation
- **Docs**: Updated all path references from `backend/` to `layer1/`
- **Docs**: Fixed stale filename references in CLI README
- **Docs**: Fixed schema version inconsistencies
- **Versions**: Synchronized all version numbers to 8.3.0

### Removed
- Deleted stale `rsvs_genius/` directory (v0.5.0 copy of `layer2/`)
- Deleted stale `layer1/python/rsvs/` directory (v6.0.0 copy of `python/rsvs/`)
- Removed external CDN dependency for favicon

## [8.3.0] - 2026-05-06

### Added

- **Python package restructured for PyPI publication**: The `python/` directory is now a proper installable package. Users can `pip install rsvs` and get the Rust core via maturin's compiled wheel, with no manual build step required.
- **maturin build system with proper configuration**: `pyproject.toml` now uses maturin as the build backend, ensuring seamless Rust→Python compilation during `pip install`. The `[project]` metadata follows PEP 621 with all required fields (name, version, description, dependencies, optional-dependencies).
- **FastAPI/uvicorn made optional**: Server dependencies are behind the `rsvs[server]` extra. A plain `pip install rsvs` gives you the core library without pulling in uvicorn, FastAPI, or their transitive dependencies. This keeps the library lightweight for programmatic use.
- **Type stubs (`__init__.pyi`) for IDE support**: Full type annotations for the PyO3-native `_rsvs` module are now shipped alongside the package. IDEs (VS Code, PyCharm) will offer autocomplete, type-checking, and inline documentation for all Rust-exposed methods without needing to inspect compiled code.
- **`_version.py` as single source of truth**: The package version is defined once in `python/rsvs/_version.py` and imported everywhere (CLI, server startup, PyO3 bindings). This eliminates the previous version chaos where `Cargo.toml`, `pyproject.toml`, CLI flags, and runtime each carried their own version string.
- **`RsvsCoreProtocol` for type-safe access**: A `typing.Protocol` class defines the complete public surface of the Rust core. Python code can type-hint against this protocol, enabling static analysis to catch attribute errors before runtime.
- **GitHub CI/CD workflows**: Automated testing on push/PR (Rust unit tests, Python integration tests, frontend build check), automated release on tag push, and maturin wheel builds for multiple platforms.
- **Professional README, ARCHITECTURE, CONTRIBUTING, CHANGELOG**: Project documentation rewritten for public consumption. README includes badges, quickstart, and architecture overview. ARCHITECTURE explains the Rust→Python→FastAPI→React layering. CONTRIBUTING defines PR workflow, code style, and testing expectations.
- **`CITATION.cff` for academic citation**: Researchers citing RSVS in papers get a BibTeX-ready citation file with DOI placeholder, author list, and version.
- **`examples/` directory with usage demos**: Minimal scripts demonstrating ingest, query, compose, appraise, relate, and MCTS query. Each example is self-contained and runnable against a fresh graph.

### Fixed

- **Version chaos unified to 8.3.0**: Previously, `Cargo.toml` reported one version, `pyproject.toml` another, the CLI printed a third, and the runtime singleton yet another. All now read from `_version.py`.
- **Validation rejecting `"composed"` compression_state**: The compression state enum was missing the `composed` variant, causing all compositional nodes to fail validation on load. The schema now recognizes `raw`, `compressed`, and `composed`.
- **`/similarity` null check on missing labels**: When querying similarity for a label not present in the graph, the server returned a 500 due to a null dereference. It now returns a proper 404 with a descriptive error message.
- **`pyproject.toml` dependencies format (PEP 621)**: Dependencies were listed as a flat string array instead of PEP 508 version specifiers. `dependencies = ["rsvs-core"]` became `dependencies = ["rsvs-core>=8.3.0"]`, and optional dependencies are under `[project.optional-dependencies]`.

### Changed

- **90%+ test coverage target for core logic**: The test suite now covers all Rust core modules (graph, sense, attention, autonomy, mcts, reflection, convergence, spreading, composition_index) and all Python API routes. Coverage is tracked in CI.

---

## [8.2.0] - 2026-05-05

### Added

- **Convergence contributors in query results**: `QueryResult` now includes a `convergence_contributors` field listing nodes and senses that contribute to convergent meaning paths. This allows callers to trace why two seemingly unrelated concepts are connected through shared deeper structures.
- **Convergence info in appraise scoring**: The `AppraiseResult` payload now carries `ConvergenceInfo`, which quantifies how much convergent evidence supports (or undermines) the appraisal verdict. A high convergence score means multiple independent paths confirm the same interpretation.
- **Convergence tendrils in 3D graph visualization**: The frontend `ForceGraph` component now renders convergence paths as semi-transparent "tendrils" — curved lines connecting nodes that share convergent meaning. These visual cues help users spot latent semantic connections that aren't direct edges.

---

## [8.1.0] - 2026-05-04

### Added

- **Cross-language sense alignment**: Nodes can now carry `LanguageLink` entries that map a sense in one language to an equivalent sense in another. For example, the English sense of "bank" (financial institution) can be linked to the Indonesian "bank" (same meaning) while keeping the river-bank sense unlinked to the Indonesian "tanggul".
- **`LanguageLink` type in nodes**: Each `LanguageLink` contains a `target_label` (the foreign-language label), `target_sense_id` (which sense of that label), `language` tag (ISO 639-1 code), and a `confidence` score indicating alignment quality.
- **Multi-language composition support**: The `compose()` method now accepts a `language` tag. When creating a compositional node from sources in different languages, the system records the language context per ingredient, enabling cross-lingual composition (e.g., composing "demokrasi" from Indonesian "rakyat" + Greek-derived "kratos").

---

## [8.0.0] - 2026-05-03

### Added

- **Convergence detection**: The system now automatically detects when multiple independent paths through the knowledge graph converge on the same semantic target. Convergence detection runs during `query()` and `appraise()` — when two or more traversal paths from different starting points arrive at the same node or sense, the system records this as a `ConvergenceInfo` event. Convergent paths are strong evidence that a concept is well-grounded in the graph's structure, not just a coincidence of edge overlap.
- **Language links**: Initial cross-language sense alignment infrastructure (expanded in v8.1.0). Nodes can store language tags and reference senses in other languages.
- **`ConvergenceInfo` in `AppraiseResult`**: Appraise verdicts now include convergence metadata — how many convergent paths were found, their depth, and the confidence boost they provide.
- **`ConvergenceContributors` in `QueryResult`**: Query results enumerate which nodes contributed to detected convergence, enabling downstream systems to weight convergent evidence more heavily than single-path evidence.

### Changed

- **Major version bump**: Convergence detection fundamentally changes the scoring model. Appraise and query results now incorporate convergent evidence, which can change verdicts and scores for existing graphs. This is a breaking change for any pipeline that assumes scores are purely additive from single-path traversal.

---

## [7.2.0] - 2026-05-05

### Added

- **175+ tests passing**: Comprehensive test coverage across Rust core, Python bindings, FastAPI server, and CLI. This is up from 113 tests in v6.2.0.
- **Race condition fixes in singleton manager**: The Python-side singleton that wraps the Rust core had a TOCTOU race when multiple threads tried to initialize simultaneously. Now uses a threading lock with double-checked locking pattern.
- **Improved error messages**: All Rust error types now include context (node ID, sense ID, operation name) in their `Display` impl. Python-side error handlers preserve this context rather than swallowing it into generic `str(e)`.
- **FastAPI middleware improvements**: Request logging, correlation ID propagation, and response time headers added to the middleware stack.
- **Security: rate limiting**: Requests are rate-limited per API key (or per IP if no key is present). Default: 100 requests/minute. Configurable via environment variable.
- **Security: CORS hardening**: CORS origins are now an explicit allowlist rather than `*`. Credentials are only sent to matched origins.
- **Security: request size limits**: Request bodies are capped at 10 MB by default. Oversized requests receive a 413 status code.

### Fixed

- **Race condition in singleton manager**: See above.
- **Error message context loss**: See above.

---

## [7.1.0] - 2026-05-05

### Added

- **`CompositionIndex` for fast composition lookup**: A secondary index that maps each atom node to the set of compositional nodes that reference it. Previously, finding all compositions containing a given atom required scanning every composition node — O(n). With the index, it's O(1) lookup + O(k) where k is the number of compositions. This dramatically speeds up `relate()`, `entity_candidates()`, and structural similarity calculations.
- **`composition_index_stats()` method**: Returns index health metrics — total entries, average fan-out, max fan-out, and orphaned entries (compositions referencing deleted atoms). Useful for monitoring and debugging.
- **Entity candidate improvements**: `entity_candidates()` now uses the composition index to quickly find structurally prominent nodes. Centrality scoring weights compositional nodes higher than raw atoms, reflecting that compositions represent richer semantic units.

---

## [7.0.0] - 2026-05-04

### Added

- **MCTS (Monte Carlo Tree Search) for reasoning path exploration**: A full MCTS implementation that treats the knowledge graph as a game tree. Each "move" is traversing an edge; the "reward" is the relevance score of the destination node. MCTS runs multiple simulations (default: 100) to explore the graph beyond the reach of greedy best-first search, discovering high-value paths that require initial low-value steps. The `mcts_query()` method accepts `simulations` and `exploration_constant` (cpuct) parameters.
- **`MCTSResult` data class**: Contains `active_sense` (the sense that MCTS settled on), `scored_atoms` (atoms ranked by MCTS visit count), `best_path` (the highest-reward path found), and `halt_reason` (why MCTS stopped — budget exhausted, stability reached, or confidence threshold met).
- **Reflection engine**: A periodic self-correction mechanism that reviews the graph's state and proposes `REVISE` and `RETIRE` actions. `REVISE` merges senses that have drifted together (Jaccard > threshold) or splits senses that have diverged. `RETIRE` removes senses with zero observations in the last N periods. The `run_reflection()` method executes one reflection cycle and returns a list of actions taken.
- **Thinking mode: "fast" vs "deep" query strategies**: `set_thinking_mode("fast")` uses direct lookup or shallow traversal. `set_thinking_mode("deep")` uses MCTS with full simulations. This mirrors the System 1 / System 2 distinction from cognitive science — fast mode for real-time responses, deep mode for careful reasoning.
- **Consolidation**: The `consolidate()` method merges duplicate senses, removes orphaned nodes, prunes low-weight edges, and compacts the atom store. It's the graph equivalent of garbage collection — run it periodically to keep the graph lean.
- **`verify()` method**: Runs a suite of structural invariants on the graph (no orphaned edges, no self-referencing compositions, layer consistency, confidence within [0,1]). Returns a list of violations. Useful for debugging and CI.
- **DEPS (Dependency-aware Entity Promotion System) recovery**: When a compose operation fails (e.g., circular chain detected), DEPS classifies the failure, explains the root cause, proposes recovery plans, and selects the best one. Failed operations now include DEPS recovery hints in their error messages.
- **Paradigm router for mode-specific behavior**: Routes queries to the lightest traversal strategy that will succeed: Direct → Shallow → Standard → Deep → MCTS. Simple lookups don't waste time on MCTS; complex queries automatically escalate to deeper paradigms.
- **Spreading activation module**: Activates related nodes through composition edges with configurable energy decay per hop. Multiple paths reinforce activation additively — the structural equivalent of semantic priming in cognitive science.
- **Neurosym verification bridge**: After creating a compositional node, the system automatically verifies structural invariants (no self-reference, layer consistency, grounding, frequency, no circular chains). Failed rules emit `neurosym_verification_warning` events.

### Changed

- **`jaccard_sets()` optimized from O(n×m) to O(n+m)**: Previously used `b.contains(x)` on `Vec` which is O(m) per call. Now converts to `HashSet` once for O(1) lookups. This is a hot path for attention scoring and sense assignment.

---

## [6.3.0] - 2026-05-03

### Added

- **Per-domain attention weights (alpha, beta, gamma)**: Each domain (e.g., "medicine", "law", "finance") can now have its own (α, β, γ) triple for the attention scoring formula `score = α·NPMI + β·Jaccard + γ·cooc`. This allows domain-specific tuning — medical terms might weight NPMI higher (precision matters), while general language might weight co-occurrence higher (breadth matters).
- **`set_domain_attention()` with auto-normalization**: Sets the attention weights for a domain. The method auto-normalizes the triple so that α + β + γ = 1.0, preventing accidental score inflation from unnormalized weights.
- **Observation-count-gated activation (5+ observations required)**: Nodes with fewer than 5 observations are excluded from spreading activation. This prevents rare or noisy terms from polluting the activation pattern. The threshold is configurable.

---

## [6.2.0] - 2026-05-02

### Added

- **`set_sense_label()` for condition annotations**: Senses can now carry arbitrary label annotations (e.g., "archaic", "technical", "regional"). These labels are stored alongside the sense and included in appraise/relate verdicts, giving downstream systems semantic context about *why* a sense scored the way it did.
- **`entity_candidates()` with centrality + diversity scoring**: Returns the most "important" nodes in the graph, scored by a combination of centrality (how many paths pass through this node) and diversity (how distinct this node is from other candidates). This is useful for building entity extraction pipelines and for highlighting key concepts in the 3D visualization.
- **Condition labels in appraise/relate verdicts**: When an appraise or relate operation returns a verdict, it now includes the condition labels of the senses involved. For example, an appraise verdict might note that the matched sense is labeled "technical", giving the caller confidence that the match is domain-specific.
- **Language tag support in `compose()`**: The compose method now accepts an optional `language` parameter, recording the language context of the composition. This is the foundation for cross-language composition support in v8.1.0.

---

## [6.1.0] - 2026-05-01

### Added

- **`context_query()` with depth-controlled lazy traversal**: A new query mode that starts from a set of seed atoms and traverses outward, scoring each encountered node with P(a|S,q) — the probability that atom a is relevant given the seed set S and query context q. Traversal is lazy: the engine expands the frontier one hop at a time, stopping when additional hops don't improve scores.
- **P(a|S,q) scoring for context-weighted atom relevance**: Instead of simple Jaccard overlap, context_query uses a probabilistic model that weights atoms by their relevance to both the seed set and the query. This produces more accurate rankings for ambiguous queries where the same atom appears in multiple contexts.
- **Cycle detection during traversal**: The traversal engine tracks visited nodes and refuses to revisit them, preventing infinite loops in cyclic graphs. This is essential for production use where the graph may contain bidirectional edges or composition cycles.
- **Adaptive halting criteria (stability, confidence, information gain)**: The traversal engine stops when one of three conditions is met: (1) score stability — the top-K ranked atoms haven't changed in the last two expansions; (2) confidence threshold — the cumulative score mass exceeds a configurable threshold; (3) information gain — the marginal information added by the last expansion falls below a threshold.
- **`TraversalConfig` data class**: Configurable parameters for context_query: `max_depth` (maximum hops from seed), `gamma` (decay factor for distance penalty), `halt_confidence` (cumulative score threshold for early stopping), `tau_relevance` (minimum score for an atom to be included in results).

---

## [6.0.0] - 2026-04-30

### Added

- **`compose()` PyO3 method with `(label, sense_id)` pairs**: The primary composition API now accepts human-readable (label, sense_id) pairs instead of raw node IDs. This makes composition creation self-documenting — callers specify which sense of each ingredient they mean. For example, `compose([("kucing", 0), ("hewan", 0)])` creates a composition from the first sense of "kucing" and the first sense of "hewan".
- **`compose_from_ids()` for backward compat**: The old ID-based composition API is preserved under `compose_from_ids()` for code that already has node IDs. Internally delegates to the same pipeline as `compose()`.
- **`StructuralSimResult` with shared/only_a/only_b compositions**: Structural similarity now returns a detailed result instead of a single float. The result breaks down which compositions are shared between two nodes, which are unique to A, and which are unique to B. This enables downstream analysis like "what transforms concept A into concept B" (the substitution analysis).
- **`SubstitutionResult` with mapped substitution pairs**: Given two nodes, the substitution analysis finds the minimal set of ingredient substitutions that would transform A's composition structure into B's. Each substitution pair maps an ingredient in A to its counterpart (or gap) in B. This is the foundation for analogical reasoning.
- **`PyGroundingEvidence` with `score()` method**: Python-accessible grounding evidence objects. Each evidence record indicates whether it's confirming or contradicting, and the `score()` method returns the net grounding score (confirming / total evidence). Scores below a threshold trigger verification warnings.
- **`SenseInfo` includes grounding_evidence and compositions**: When querying sense details, the result now includes the full grounding evidence chain and the list of compositions this sense participates in. This gives a complete picture of a sense's epistemic status.
- **`NodeInfo` includes `derived_from_node_ids` and `compression_reason`**: Node metadata now tracks where the node came from (which other nodes it was derived from, if any) and why it was compressed (e.g., "high_cooc", "manual", "auto_promote"). This is critical for debugging and for the reflection engine's RETIRE action.

### Changed

- **Major version bump (v6.0.0)**: The composition API changed from ID-based to label-based, and the similarity result type changed from `f64` to `StructuralSimResult`. These are breaking changes for any code calling `compose()` or `structural_similarity()` directly.

---

## [5.1.0] - 2026-04-28

### Added

- **`GroundingEvidence` type: confirming/contradicting contexts**: Each sense can now accumulate grounding evidence — observations from real data that either confirm or contradict the sense's existence. Confirming evidence is a context where the sense's atoms co-occur as expected. Contradicting evidence is a context where the sense's atoms fail to co-occur, suggesting the sense is spurious.
- **Composition verification pipeline**: After creating a composition, the system automatically checks grounding evidence for each ingredient sense. If any sense has a grounding score below threshold, the composition is flagged for review. This prevents the graph from accumulating poorly-grounded compositional nodes.
- **Grounding score derived from evidence ratio**: The grounding score for a sense is `confirming / (confirming + contradicting)`. A score of 1.0 means all evidence confirms the sense; 0.5 means equal evidence for and against. This simple ratio is surprisingly effective because contradictory evidence is rare for well-formed senses and common for spurious ones.

---

## [5.0.0] - 2026-04-25

### Added

- **Compositional sense definitions**: Senses are no longer flat atom sets — they can be defined as compositions of other senses. For example, the Malay word "raja" can be defined as the composition of "laki-laki" (male) and "kekuasaan" (power). This mirrors how humans understand many concepts as combinations of simpler ones.
- **`CompositionRef` type (`node_id`, `sense_id`)**: A typed reference to a specific sense of a specific node. Used as the ingredient type in compositional definitions. Each `CompositionRef` is a pair of (node_id, sense_id), enabling precise disambiguation — "bank" the financial institution (sense 0) vs. "bank" the river edge (sense 1).
- **Layer system (layer 0 = atoms, layer 1 = compositions, layer 2+ = recursive)**: Nodes are organized into layers based on their compositional depth. Layer 0 nodes are raw atoms with no ingredients. Layer 1 nodes are compositions of layer 0 nodes. Layer 2+ nodes are compositions that include at least one layer 1+ node. This layering prevents circular compositions and enables structured similarity comparison.
- **Structural similarity (compare composition structures, not just atom sets)**: Traditional Jaccard similarity compares the atom sets of two nodes. Structural similarity compares their composition structures — do they share the same ingredients in the same arrangement? Two nodes with low Jaccard but high structural similarity are analogies: different atoms, same relational pattern.
- **Substitution analysis (what transforms concept A into concept B)**: Given two structurally similar nodes, substitution analysis identifies the minimal set of ingredient substitutions needed to transform A into B. For example, "raja" → "ratu" requires substituting "laki-laki" → "perempuan" while keeping "kekuasaan" unchanged.
- **Compose API: explicit composition creation**: The `compose()` method creates a new compositional node from a list of `CompositionRef` ingredients. The system validates layer consistency (ingredients must be at least one layer below the new node), checks for self-reference, and records the composition in the graph.
- **Grounding evidence tracking per sense**: Each sense tracks how many times its ingredient atoms have been observed co-occurring in real data vs. failing to co-occur. This grounding evidence is used to validate that compositional senses reflect genuine patterns in the data, not just coincidental co-occurrence.

### Changed

- **Major version bump (v5.0.0)**: The introduction of compositional senses and the layer system fundamentally changes the graph's data model. Nodes can now participate in composition hierarchies, and the similarity model has expanded from Jaccard-only to include structural similarity and substitution analysis. Existing graphs can be migrated by assigning all pre-existing nodes to layer 0.

---

## [4.2.0] - 2025-05-03

### Added

- **Policy-driven node management**: A policy engine that governs how nodes are promoted, demoted, quarantined, and retired. Policies are configurable rules (e.g., "promote to Stable after 10 observations with confidence > 0.75") that replace ad-hoc hardcoded thresholds. Each domain can have its own policy set.
- **Auto-promotion rules**: Nodes that meet policy criteria are automatically promoted to the next tier without manual intervention. Promotion is gated by a hysteresis threshold — the promote threshold is higher than the demote threshold — to prevent flip-flopping in borderline cases.
- **Compression state tracking (`raw` → `compressed`)**: Nodes track whether their internal data is in raw form (full co-occurrence matrix) or compressed form (summary statistics). Compressed nodes use less memory but can't be decompressed back to raw. The transition is one-way and triggered by policy when the node reaches Stable tier.
- **Event sourcing with v1 schema**: All graph mutations are recorded as immutable events in a JSONL log. The event schema is versioned (currently v1) and includes a correlation ID for tracing related mutations across multiple events. This enables audit trails, replay-based debugging, and point-in-time graph reconstruction.
- **Seed atom bootstrap**: 24 primitive immutable atoms serve as the axiomatic foundation of the graph. These seeds (e.g., "entity", "action", "property") are always present and can't be deleted, ensuring that every composition has a well-defined grounding chain back to primitives.
- **PyO3 bindings for Rust↔Python bridge**: The Rust core is exposed to Python via PyO3, providing near-native performance for graph operations while maintaining Python's ergonomics for the API server and CLI.
- **FastAPI server with OpenAPI docs**: The HTTP API is built on FastAPI with automatic OpenAPI schema generation, request validation, and interactive docs at `/docs`. Replaces the previous `BaseHTTPRequestHandler`-based server.
- **3D knowledge graph visualization with React Three Fiber**: An interactive 3D force-directed graph rendered in the browser. Nodes are spheres colored by tier; edges are lines weighted by attention score. Compositional nodes display their ingredient structure on hover.
- **Appraise and Relate mode UI panels**: Appraise mode evaluates input text against the graph and returns a verdict with confidence. Relate mode finds structurally related concepts. Both are accessible from the frontend sidebar.
- **Event timeline with play/pause/speed controls**: A timeline bar at the bottom of the visualization that replays graph mutations over time. Users can scrub to any point, play forward at 1x–10x speed, and pause to inspect the graph state at that moment.
- **Comprehensive CLI with 11 subcommands**: `rsvs ingest`, `rsvs query`, `rsvs compose`, `rsvs appraise`, `rsvs relate`, `rsvs snapshot`, `rsvs events`, `rsvs stats`, `rsvs verify`, `rsvs consolidate`, `rsvs serve`.
- **Evaluation suite with 5 benchmarks**: Standardized benchmarks for similarity accuracy, composition correctness, attention precision, autonomy lifecycle fidelity, and query relevance.
- **CI/CD pipeline via GitHub Actions**: Automated testing, linting, and build on every push/PR. Release workflow publishes Python wheels and Docker images on tag push.
- **Docker and docker-compose support**: Multi-stage Dockerfile for the Rust+Python backend, separate Dockerfile for the Next.js frontend, and a docker-compose.yml that orchestrates both with nginx as reverse proxy.

### Changed

- **Migrated Python server from `BaseHTTPRequestHandler` to FastAPI**: Faster development cycle with auto-generated docs, type-safe request/response models, and async support.
- **Proper Rust error types with `thiserror`**: Replaced stringly-typed errors with structured error enums that carry context (node ID, operation, expected vs. actual). Python bindings convert these to typed exceptions.
- **Thread-safe Python singleton for Rust core access**: The Rust graph is wrapped in a `Mutex` inside a Python singleton, ensuring safe concurrent access from async FastAPI handlers.
- **Frontend package renamed to `@rsvs/frontend`**: From the generic template name, reflecting the project identity.
- **Strict TypeScript mode enabled (`noImplicitAny: true`)**: All frontend code is fully typed, catching a class of runtime errors at compile time.

### Fixed

- **CLI version mismatch (0.8.0 → 4.2.0)**: The CLI was reporting a hardcoded 0.8.0 instead of the actual version.
- **Broken `tempfile` dependency in `pyproject.toml`**: The dependency was listed as `tempfile` (a stdlib module) instead of being omitted.
- **Hardcoded personal paths in `.env.example`**: Replaced with placeholder values.
- **Mock data leaking into production code paths**: Mock data was imported at the module level in some API routes, causing test fixtures to appear in production responses.

---

## [4.1.0] - 2025-03-15

### Added

- **Multi-sense framework with fragile/mature sense lifecycle**: Nodes can carry multiple senses (polysemy support). Each sense has a lifecycle: Fragile (newly induced, low confidence) → Mature (well-confirmed by data). Fragile senses are pruned if they fail to accumulate confirming evidence within a timeout period.
- **Incremental coherence calculation O(n) per context**: Coherence scoring measures how well a sense's atom set fits a given context. The incremental algorithm updates coherence in O(n) when new observations arrive, instead of recomputing from scratch.
- **Sense merge algorithm with Jaccard threshold**: When two senses of the same node have Jaccard similarity above a threshold (default 0.7), they are automatically merged. This prevents sense proliferation in the face of noisy induction.
- **Fragile sense pruning after inactivity timeout**: Fragile senses that receive no confirming observations for a configurable period (default: 7 days equivalent in observation count) are automatically removed.

### Changed

- **Confidence update now uses EMA (η=0.10) with energy constraint**: Node confidence is updated using an exponential moving average with η=0.10, giving recent observations 10% weight. An energy constraint prevents confidence from exceeding the total observation energy, capping runaway confidence inflation.
- **Hysteresis gap widened to 0.15 (promote ≥ 0.75, demote < 0.60)**: The gap between promotion and demotion thresholds was widened to prevent nodes from rapidly oscillating between tiers when their confidence hovers near the boundary.

### Fixed

- **Race condition in concurrent confidence updates**: Two threads updating the same node's confidence could read the same base value and both write, losing one update. Now uses compare-and-swap semantics.
- **Memory leak in sense manager during high-volume ingest**: Sense objects were being retained in an internal cache even after pruning. The cache is now cleared on each pruning cycle.

---

## [4.0.0] - 2025-01-20

### Added

- **Rust core engine with PyO3 Python bindings**: The graph engine is rewritten in Rust for 10-100x performance improvement on hot paths (ingest, similarity, traversal). Python bindings via PyO3 provide transparent access — Python code calls the same API, but execution happens in Rust.
- **Hard attention scoring (NPMI + Jaccard + Co-occurrence)**: Attention scoring formula: `score = α·NPMI + β·Jaccard + γ·cooc`. Each component captures a different aspect of semantic relatedness: NPMI captures statistical association, Jaccard captures set overlap, and co-occurrence captures raw frequency. Weights are configurable per domain.
- **Seed atom bootstrap (24 primitive atoms)**: The graph is initialized with 24 seed atoms that serve as the foundation for all subsequent compositions. These are immutable and always present.
- **Node lifecycle state machine (New → Candidate → Stable → Deprecated → Quarantine)**: Every node progresses through a tiered lifecycle. New nodes start at "New" and are promoted based on confidence and observation count. Poorly-performing nodes are demoted to "Deprecated" and eventually removed. Quarantined nodes are flagged for review but not automatically deleted.
- **Python bridge server with HTTP API**: A Python HTTP server (initially BaseHTTPRequestHandler, later migrated to FastAPI) that exposes the Rust core over a REST API. This is the primary interface for the frontend and CLI.
- **Event stream with runtime correlation IDs**: All graph mutations emit events tagged with a correlation ID. Related mutations (e.g., creating a composition and updating its ingredients) share the same correlation ID, enabling end-to-end tracing.
- **Snapshot persistence with JSON serialization**: The entire graph state can be serialized to JSON and restored from a snapshot file. Snapshots are taken automatically before risky operations and on-demand via the CLI.

### Changed

- **Breaking: Unified node model replaces Atom/Composite dualism**: Previously, the graph had two node types (Atom and Composite) with different fields and different code paths. The unified model has a single `Node` type with a `compression_state` field that indicates whether the node's data is raw or compressed. This simplification eliminated an entire class of bugs where Atom-only operations were applied to Composite nodes.
- **Breaking: Schema version bumped to v4.2 (no backward compatibility)**: The data model change is too fundamental for migration. Pre-v4.2 data files cannot be loaded by v4.0+.
- **Hard break policy for pre-v4.2 payloads**: The server rejects requests that include pre-v4.2 schema fields, returning a 400 with a clear error message.

### Removed

- **Legacy Python-only backend (all computation now in Rust)**: The pure-Python graph engine has been completely removed. All computation goes through the Rust core via PyO3.
- **`_legacy_*` fallback functions from bridge server**: Temporary compatibility shims that wrapped the old Python API around the new Rust API have been removed.
- **`render` metadata generation from bridge (moved to frontend)**: Graph visualization metadata (positions, colors, sizes) is now computed client-side in the React Three Fiber frontend, not server-side.

---

## [3.0.0] - 2024-09-10

### Added

- **Tiered memory lifecycle: New → Candidate → Stable → Deprecated**: Nodes are no longer permanent — they progress through a tiered lifecycle based on their observed reliability and confidence score. New nodes start at "New" tier with minimal privileges. After accumulating sufficient confirming observations, they are promoted to "Candidate" and eventually "Stable". Poorly-performing nodes are demoted to "Deprecated" and eventually garbage-collected.
- **Confidence scoring with warm-up period**: Each node has a confidence score (0.0–1.0) that reflects how well its observed behavior matches expectations. New nodes start with a low confidence that increases during a warm-up period (first N observations). This prevents premature promotion of noisy or spurious nodes.
- **Automatic promotion and demotion of nodes**: Promotion and demotion are driven by policy rules, not manual intervention. The autonomy engine checks all nodes on each cycle and applies promotions/demotions as needed.
- **Quarantine and watchlist mechanisms**: Nodes that exhibit suspicious behavior (e.g., sudden confidence drop, contradictory evidence) are placed on a watchlist. If the behavior persists, they are quarantined — excluded from query results but retained in the graph for review. Quarantined nodes can be manually reinstated or permanently deleted.
- **Composite node model with render metadata**: Composite nodes (nodes composed of other nodes) now carry render metadata (position, color, size) for visualization. This decouples the visual representation from the semantic content.
- **Appraise mode for evaluating text against graph**: Given input text, appraise mode evaluates how well the text's concepts are represented in the graph and returns a verdict (strong match, weak match, no match) with a confidence score.
- **Relate mode for finding related concepts**: Given a concept, relate mode finds structurally similar concepts in the graph, ranked by similarity score. Relatedness is based on shared atom sets and, for compositional nodes, shared ingredient structures.
- **Artifact persistence (snapshots, events, reports)**: Graph artifacts can be saved to disk and restored later. Snapshots capture the full graph state, events capture the mutation log, and reports capture analysis results.

### Changed

- **Python server migrated from Flask to BaseHTTPRequestHandler**: The Flask dependency was removed in favor of a lighter-weight HTTP handler. This reduced the attack surface and deployment complexity, though it was later migrated again to FastAPI in v4.2.0.
- **Improved tokenization with stopword filtering**: Text ingestion now filters common stopwords before creating atoms, reducing graph noise from high-frequency low-information terms.

---

## [2.0.0] - 2024-05-15

### Added

- **Spreading activation for attention-weighted traversal**: Traversal through the graph is now weighted by an attention mechanism. When querying or relating, activation spreads from seed nodes through edges, with each hop applying a decay factor. Nodes that receive activation from multiple paths get reinforced (additive accumulation). This produces more nuanced query results than simple breadth-first or depth-first traversal.
- **Edge weight decay and reinforcement**: Edge weights are not static — they decay over time if unused and are reinforced when traversed successfully. This implements a use-it-or-lose-it policy that keeps the graph's attention weights aligned with actual query patterns.
- **Domain-aware attention configs**: Different domains (e.g., medical, legal, general) can have different attention parameters (decay rate, reinforcement rate, initial weight). This allows the system to adapt its traversal strategy to the nature of the content.
- **Atom promotion from text ingestion**: Raw text is ingested, tokenized, and converted into atom nodes. Co-occurring tokens create edges between their respective atoms. The ingestion pipeline handles deduplication, stopword filtering, and frequency tracking.
- **Co-occurrence statistics tracking**: The system tracks how often pairs of atoms appear together in the same context window. These co-occurrence statistics are the raw material for NPMI calculation and attention scoring.
- **Jaccard similarity for node comparison**: Node similarity is computed using Jaccard overlap of their atom sets. While simple, Jaccard provides a robust baseline that is interpretable and fast to compute.
- **Basic web UI for graph visualization**: A simple web interface that renders the knowledge graph as a force-directed layout. Users can click nodes to inspect details and search for specific concepts.

---

## [1.0.0] - 2024-01-01

### Added

- **Basic knowledge graph with atoms and edges**: The foundational data structure. Atoms are the smallest units of meaning (typically words or short phrases). Edges connect atoms that co-occur in the same context. The graph is stored in memory and can be persisted to JSON.
- **Simple ingest, query, similarity**: The three core operations. Ingest adds text to the graph, creating atoms and edges. Query searches the graph for nodes matching a given label or pattern. Similarity compares two nodes and returns a Jaccard-based overlap score.
- **Jaccard-based flat similarity**: Similarity between two nodes is the size of their atom set intersection divided by the size of their union. This flat (non-hierarchical) similarity is the baseline from which all later similarity models evolved.
- **HTTP API with `/run` endpoint**: A minimal HTTP server with a single `/run` endpoint that accepts JSON commands (ingest, query, similarity) and returns JSON results. The beginning of the API that would grow into the full FastAPI server.
- **React frontend with D3.js visualization**: A React application that renders the knowledge graph as an interactive D3.js force-directed layout. Users can drag nodes, zoom, and click to inspect details. This would later be replaced by the React Three Fiber 3D visualization in v4.2.0.

---

[8.3.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v8.3.0
[8.2.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v8.2.0
[8.1.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v8.1.0
[8.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v8.0.0
[7.2.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v7.2.0
[7.1.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v7.1.0
[7.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v7.0.0
[6.3.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v6.3.0
[6.2.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v6.2.0
[6.1.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v6.1.0
[6.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v6.0.0
[5.1.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v5.1.0
[5.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v5.0.0
[4.2.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v4.2.0
[4.1.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v4.1.0
[4.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v4.0.0
[3.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v3.0.0
[2.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v2.0.0
[1.0.0]: https://github.com/Wolfvin/SymbolicPuzzle3D/releases/tag/v1.0.0
