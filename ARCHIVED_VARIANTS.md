# Archived Enum Variants

This document records enum variants that were removed during Phase 1 cleanup
because they had zero references in the Rust codebase. They were never produced
by the pipeline, never constructed in tests, and never matched in production code.

If any of these variants are needed in the future, they can be re-added to the
relevant enum. All enums use `#[non_exhaustive]`, so adding new variants is
not a breaking change for downstream consumers.

**Cleanup date**: 2026-05-21
**Phase**: Phase 1 — Dead Code Cleanup + Documentation-Reality Gap

---

## EdgeSource (types.rs)

Removed 8 variants (zero Rust references):

| Variant | Original Doc | Original Line | Rationale |
|---------|-------------|---------------|-----------|
| `Composition` | Created by explicit composition (compose API) | 62 | Never produced by pipeline |
| `GapDetection` | Created by gap detection (P1) — predicted but not observed compositions | 64 | Never produced by pipeline |
| `Discourse` | Created by discourse tracking (P3) — rhetorical/performative edges | 68 | Never produced by pipeline. Only referenced in docs/ |
| `Blending` | v10.0: Created by compositional blending — hybrid A∧B edges | 71 | Never produced by pipeline |
| `Synthesis` | v10.0: Created by cross-pathway synthesis — hidden meaning edges | 80 | Never produced by pipeline |
| `CompoundDiscovery` | v10.1: Created by compound discovery — multi-word expression edges | 83 | Never produced by pipeline |
| `EpistemicGovernance` | From MD-4: belief state transition | 91 | Never produced by pipeline. Was incorrectly listed as "active" in doc comment |
| `ExecutiveControl` | From MD-5: executive routing | 94 | Never produced by pipeline. Was incorrectly listed as "active" in doc comment |

**Kept variants** (2 design-intent variants in match arms):
- `Abductive` — referenced in govern_beliefs.rs match arm for `(CompositionType::Hypothesis, EdgeSource::Abductive)`. Also in test setup. Not yet produced by default pipeline.
- `PatternMining` — referenced in govern_beliefs.rs match arm for `(CompositionType::Pattern, EdgeSource::PatternMining)`. Also in cognitive_tests.rs. Not yet produced by default pipeline.

---

## RelationType (types.rs)

Removed 5 variants (zero Rust references):

| Variant | Original Doc | Rationale |
|---------|-------------|-----------|
| `Differential` | "X is more/less than Y in dimension D" — comparative | Never constructed |
| `Functional` | "X can do Y" / "X is used for Y" — functional | Never constructed |
| `Spatial` | "X is located at Y" — spatial | Never constructed |
| `Temporal` | "X occurs before/after Y" — temporal | Never constructed |
| `Discursive` | Discursive / rhetorical relation | Never constructed |

---

## HiddenMeaningType (types.rs)

Removed 5 variants (zero Rust references):

| Variant | Original Doc | Rationale |
|---------|-------------|-----------|
| `AffectiveDisguise` | The surface meaning masks a deeper affective truth | Never constructed |
| `SocialConcealment` | A social dynamic is hidden beneath the literal content | Never constructed |
| `PerformativeMask` | The utterance is a performative act disguised as something else | Never constructed |
| `TraumaPattern` | A trauma pattern underlies the surface expression | Never constructed |
| `PowerDynamic` | Power dynamics hidden in the communication | Never constructed |

---

## FrameSource (v12/types.rs)

Removed 3 variants (zero Rust references):

| Variant | Original Doc | Rationale |
|---------|-------------|-----------|
| `UdParse` | Phase 2: dependency parsing | Never constructed. Reserved for Phase 2 parser integration |
| `SrlLabel` | Phase 2: semantic role labeling | Never constructed. Reserved for Phase 2 parser integration |
| `AmrCompilation` | Phase 3: AMR graph compilation | Never constructed. Reserved for Phase 3 parser integration |

---

## PatternCategory (v12/types.rs)

Removed 4 variants (zero Rust references):

| Variant | Original Doc | Rationale |
|---------|-------------|-----------|
| `CausalChain` | Causal chain (A → B → C) | Never constructed |
| `GoalAction` | Goal-action pair | Never constructed |
| `RoleSubstitution` | Role substitution pattern | Never constructed |
| `TemporalSequence` | Temporal sequence pattern | Never constructed |

---

## AcquisitionSource (v12/types.rs)

Removed 2 variants (zero Rust references):

| Variant | Original Doc | Rationale |
|---------|-------------|-----------|
| `SelfStudy` | Self-directed study (Phase 2) | Never constructed. Reserved for Phase 2 acquisition |
| `ExternalSource` | External source (Phase 2) | Never constructed. Reserved for Phase 2 acquisition |

---

## EpistemicConflictType (v12/types.rs)

Removed 5 variants (zero Rust references):

| Variant | Original Doc | Rationale |
|---------|-------------|-----------|
| `AgentConflict` | Same predicate, different agent fillers | Never produced by detect_contradiction() |
| `PatientConflict` | Same predicate, different patient fillers | Never produced by detect_contradiction() |
| `CauseConflict` | Same predicate, different cause fillers | Never produced by detect_contradiction() |
| `TemporalConflict` | Temporal incompatibility between compositions | Never produced by detect_contradiction() |
| `LocationConflict` | Spatial incompatibility between compositions | Never produced by detect_contradiction() |

---

## EnrichmentSource (v12/types.rs)

Removed 2 variants (zero Rust references):

| Variant | Original Doc | Rationale |
|---------|-------------|-----------|
| `ReExtraction` | Enrichment from re-extraction with graph context | Never constructed |
| `HumanAssertion` | Enrichment from human assertion | Never constructed |

---

## Summary

| Enum | Total Before | Active After | Removed | Kept Design-Intent |
|------|-------------|-------------|---------|-------------------|
| EdgeSource | 20 | 12 | 8 | 2 (Abductive, PatternMining) |
| RelationType | 7 | 2 | 5 | 0 |
| HiddenMeaningType | 6 | 1 | 5 | 0 |
| FrameSource | 5 | 2 | 3 | 0 |
| PatternCategory | 5 | 1 | 4 | 0 |
| AcquisitionSource | 4 | 2 | 2 | 0 |
| EpistemicConflictType | 10 | 5 | 5 | 0 |
| EnrichmentSource | 4 | 2 | 2 | 0 |
| **Total** | **61** | **27** | **34** | **2** |

**Net reduction**: 61 → 27 enum variants (−56%)
