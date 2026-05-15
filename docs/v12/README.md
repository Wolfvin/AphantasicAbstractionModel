# AAM v12.0 Design Documents

This directory contains the authoritative design specifications for the **Aphantasic Abstraction Model (AAM) v12.0** architecture.

## What is v12.0?

v12.0 is a major architecture refactor built on **6 Unified Abstractions** that replace the overlapping type systems accumulated in v8.3–v11.0:

1. **SemanticAtom** — Universal ingest primitive (replaces Token, EventFrame, HiddenMeaningCandidate)
2. **Composition** — Universal structured grouping (replaces EventFrame, HiddenMeaningCandidate, Pattern, Hypothesis)
3. **LifecycleState + EpistemicState** — Two orthogonal status axes (replaces 4 overlapping lifecycle enums)
4. **SemanticEdge** — Single typed triple (replaces 4 overlapping edge systems)
5. **Transform (DAG)** — Declarative transform graph (replaces hardcoded pipeline stages)
6. **Seed Anchoring** — Seed-driven epistemic confidence (replaces source trust weight system)

v12.0 also introduces a **Closed Feedback Loop**: `DetectGaps` → `SelectAcquisition` → `EnrichComposition`/`ReExtractFrame` → `GovernBeliefs` re-evaluation — enabling the system to structurally repair its understanding rather than just detect gaps.

## Document Index

| Document | Title | Purpose |
|----------|-------|---------|
| [MD1](MD1_semantic_frame_compiler.md) | Semantic Frame Compiler | Defines `ExtractFrame` transform: rule-based semantic frame extraction from text into `SemanticAtom(Event)` with role-based members |
| [MD2](MD2_pre_ingest_meaning_reasoner.md) | Pre-Ingest Meaning Reasoner | Defines `ReasonFrame` transform: pre-ingest reasoning on event atoms to derive `SemanticAtom(HiddenMeaning)` with problem/solution roles |
| [MD3](MD3_architecture_refactor.md) | Architecture Refactor | **Foundation document.** Defines the 6 Unified Abstractions, DAG-based `PipelineEngine`, `PipelineContext`, feedback loop types, and all core types |
| [MD4](MD4_epistemic_truth_governance.md) | Epistemic Truth Governance | Defines `GovernBeliefs` and `SeedAnchor` transforms: contradiction detection, promotion criteria, seed-driven confidence, contradiction resolution |
| [MD5](MD5_executive_cognition_layer.md) | Executive Cognition Layer | Defines 3 cognitive modes (Reactive, Analytical, Reflective), `ComputeBudget`, `StopCondition`, `Reflect` transform, enrichment loop |
| [MD6](MD6_epistemic_acquisition_hierarchy.md) | Epistemic Acquisition Hierarchy | Defines `DetectGaps` and `SelectAcquisition` transforms: per-atom-type gap detection, acquisition hierarchy (Remember/Study/Ask), graph-based disambiguation |
| [REVISION_NOTES](REVISION_NOTES.md) | Revision Notes | Documents the feedback loop closure, post-audit gap resolutions (3 rounds), and key design decisions across all MDs |

## Relationship to Implementation

These documents are the **authoritative design specs** that the v12.0 Rust module (`layer1/crates/rsvs-core/src/v12/`) implements. The Rust implementation is:

- **Additive** — v12.0 types do not replace existing v8.3 types (`Node`, `Edge`, `CompositionRef`)
- **Feature-flagged** — compiled only when the `v12` Cargo feature is enabled
- **Non-exhaustive** — enums that may grow in future versions are marked `#[non_exhaustive]`
- **Backward-compatible** — `#[serde(default)]` is used for forward-compatible deserialization
