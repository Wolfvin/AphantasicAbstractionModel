# AGENT PROMPT — AAM v12.0 Rust Core Development
# Commit baseline: a85f903 (252 tests, all pass)
# Focus: Completing v12.0 cognitive pipeline implementation

## STATUS TERKINI
- 252 tests pass (102 cognitive lib + 8 integration + 142 integration)
- CrossType contradiction: fixed via predicate-label fallback strategy
- process_user_answer: verified semantically (P0 test with 4 proofs)
- Commutativity: documented as partial — structurally commutative, confidence-non-commutative ≤0.15
- 16 cognitive scenario tests covering all 5 conflict types + feedback loop + P0 pipeline

## YANG SUDAH DIIMPLEMENTASI

### Core Pipeline (layer1/crates/rsvs-core/src/v12/)
- **ExtractFrame** (MD-1): Token → SemanticAtom via rule-based frame compiler
- **ReasonFrame** (MD-2): ProblemSolutionRule, PolarityConflictRule, ReasoningContext
- **IngestAtoms** (MD-3): Atom → Composition → Graph insertion
- **GovernBeliefs** (MD-4): Lifecycle/Epistemic states, all 5 conflict types detected,
  promotions (New→Candidate→Stable, Inferred→Grounded), contradiction resolution
- **SeedAnchor** (MD-4): Seed-aligned confidence adjustment
- **DetectGaps** (MD-6): MissingRole, AmbiguousToken, SparseGraph, LowGrounding,
  UnresolvedContradiction, IncompleteHiddenMeaning, MissingCause, MissingPurpose
- **SelectAcquisition** (MD-6): PassiveRecall → ReExtraction → AskUser → Defer
  hierarchy, InquiryMemory cross-ingest tracking, process_user_answer/merge
- **PipelineEngine**: DAG-based transform pipeline with register_default_pipeline()

### Supporting Modules
- **ConvergenceDetection**: Jaccard-based structural equivalence (known limitation L1)
- **SpreadingActivation**: Energy-based activation propagation
- **ExecutiveOrchestrator**: Cognitive mode selection (stub run_enrichment_loop)
- **TemporalDecay**: Ebbinghaus forgetting curve (stub)
- **Persistence**: JSON save/load

### Test Coverage
- 16 cognitive scenario tests in `cognitive_tests.rs`
- 142 integration tests in `v12_validation.rs`
- All 5 conflict types: PolarityConflict, RoleReversal, PurposeConflict,
  SemanticContradiction (CrossType), EquivalenceMismatch
- P0 semantic verification: process_user_answer 4-proof pipeline test
- Commutativity: basic + feedback-loop variant
- Known limitations documented in `docs/v12-cognitive-test-report.md` (L1-L6)

## YANG BELUM DIIMPLEMENTASI (dari MD-1 s/d MD-6)

### HIGH PRIORITY
1. **EnrichComposition transform** — type/struct ada, execute() belum lengkap
   - Must: receive EnrichmentRequest → add CompositionMember → re-govern
   - Depends on: Graph mutation API, GovernBeliefs.re_govern_composition()
2. **ReExtractFrame transform** — skeleton ada, actual re-extraction logic missing
   - Must: receive ReExtractionRequest → re-tokenize with context hints → new atoms
   - Depends on: ExtractFrame with context parameter
3. **ExecutiveOrchestrator.run_enrichment_loop()** — stub, belum actual loop
   - Must: detect gaps → select strategy → execute enrichment → re-govern → check convergence
   - Depends on: EnrichComposition + ReExtractFrame above

### MEDIUM PRIORITY
4. **Convergence engine integration** — detect_convergence() ada tapi belum
   integrated ke pipeline. Need: trigger from ExecutiveOrchestrator after enrichment
5. **Spreading activation integration** — spread() ada tapi belum triggered dari
   executive. Need: activation as signal for gap prioritization
6. **Reflection (Reflect transform)** — struct ada, ReflectionFinding logic belum.
   Need: detect when composition is unstable → generate reflection finding

### LOW PRIORITY
7. **Python bindings (PyO3)** — bridge ke layer1 Rust belum update untuk v12 types.
   C1 AbstractionBridge alias still missing. 19/19 ImportError in Python tests.
8. **Layer 0 stubs** — image/audio/video perceptual adapters masih stub

### NOT STARTED
9. **Role-weighted structural similarity** (fix L1): `α × role_jaccard + (1-α) × node_jaccard`
10. **TemporalDecay implementation** (L4): actual decay curve + reinforcement
11. **Persistence integration** (L5): graph save/load in pipeline

## KNOWN LIMITATIONS (from test report)

| ID | Description | Proposed Fix |
|----|-------------|-------------|
| L1 | Jaccard node-overlap doesn't capture role equivalence | Role-weighted similarity (α≈0.6) |
| L2 | PassiveRecall for empty graph (self-referent) | Exclude self-composition candidates |
| L3 | ExecutiveOrchestrator not covered by cognitive tests | Implement enrichment loop first |
| L4 | TemporalDecay not covered | Implement Ebbinghaus decay |
| L5 | Persistence not covered | JSON save/load + roundtrip test |
| L6 | Commutativity partial (confidence ≤0.15 delta) | Expected behavior, not a bug |

## ARCHITECTURE

```
v12 DAG Pipeline:
  Tokenize → ExtractFrame → ReasonFrame → IngestAtoms
  → GovernBeliefs → SeedAnchor → DetectGaps → SelectAcquisition
  → ExecutiveOrchestrator → [EnrichComposition | ReExtractFrame]

Conflict Types (GovernBeliefs):
  PolarityConflict   — same predicate + same agent + XOR negation
  RoleReversal       — same predicate + swapped Agent/Patient
  PurposeConflict    — same predicate + same agent + different Purpose
  SemanticContradiction — HiddenMeaning vs Event (CrossType)
  EquivalenceMismatch  — same Problem + different Solution (HM)

Acquisition Hierarchy:
  PassiveRecall → ReExtraction → AskUser → Defer
  InquiryMemory prevents re-asking same gap across ingests
```

## RULES
- Jangan ubah existing test yang sudah pass
- Setiap implementasi baru harus disertai minimal 2 test kognitif
- Commit format: `feat(v12): [modul] — [deskripsi singkat]`
- Setelah setiap commit: `cargo test --features v12` harus 0 failures
- File output selalu ke `/home/z/my-project/download/`
- Semua Rust code ada di `layer1/crates/rsvs-core/src/v12/`

## KEY FILES

```
layer1/crates/rsvs-core/src/v12/
├── mod.rs              ← Module registration + re-exports
├── types.rs            ← Core types (SemanticAtom, Composition, etc.)
├── pipeline.rs         ← PipelineEngine + Graph + DAG execution
├── govern_beliefs.rs   ← GovernBeliefs + SeedAnchor (MD-4)
├── acquisition.rs      ← DetectGaps + SelectAcquisition (MD-6)
├── reason_frame.rs     ← ReasonFrame + ProblemSolutionRule (MD-2)
├── extract_frame.rs    ← ExtractFrame (MD-1)
├── executive.rs        ← ExecutiveOrchestrator (MD-5)
├── convergence.rs      ← ConvergenceDetection
├── spreading.rs        ← SpreadingActivation
├── temporal.rs         ← TemporalDecay
├── persistence.rs      ← JSON save/load
└── cognitive_tests.rs  ← 16 cognitive scenario tests

docs/
├── v12/MD1-6           ← Architecture specification documents
└── v12-cognitive-test-report.md  ← Test report with L1-L6 limitations
```
