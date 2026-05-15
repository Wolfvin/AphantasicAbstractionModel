# !updates/ Revision Notes — Feedback Loop Closure

> Date: 2026-05-15
> Revision: v12.0 feedback loop closure across MD-1 through MD-6

## Problem Identified

The original MD-1 through MD-6 had a **broken feedback loop**:

- **MD-1** (`ExtractFrame`) produces `SemanticAtom(Event)` via blind rule-based extraction — it never knows if its understanding is correct
- **MD-6** (`DetectGaps`) detects weaknesses in those frames (missing roles, low confidence)
- But gaps are sent to `AskUser` or `Deferred` — **never back to the original Composition or ExtractFrame**
- User answers create separate `SemanticAtom(Acquisition)` atoms that **never merge** into the original Composition
- The system can detect that it doesn't understand, but **cannot structurally repair its understanding**

## What Was Changed

### MD-3 (Foundation) — 7 additions
1. `EdgeSource::EnrichmentFeedback` + `EdgeSource::ExtractionRepair` variants
2. `EnrichComposition` + `ReExtractFrame` in Core Transforms table
3. Feedback loop wiring in "Adding a New MD" subsection
4. **NEW section: "Feedback Loop — Closing the Detection-Repair Cycle"** with:
   - `EnrichmentRequest` struct + `EnrichmentSource` enum
   - `ReExtractionRequest` struct
   - `RecallAction` enum (the missing bridge)
   - `EnrichComposition` Transform
   - `ReExtractFrame` Transform
   - Closed-loop pipeline diagram
   - Enrichment Semantics subsection
5. Updated Full Architecture pipeline diagram with Composition Repair box
6. Migration Strategy Phase B items 7-8
7. Acceptance Criteria items 10-12

### MD-1 (ExtractFrame) — 8 additions
1. Updated prerequisite to mention feedback loop types
2. `source_composition_id` tracing comment in transform()
3. **NEW section: "Extraction Quality Tracking"** with:
   - `ExtractionQuality` struct (gap_rate, repair_rate, is_weak)
   - `ExtractionQualityTracker` (records extraction outcomes per rule)
4. **NEW section: "Feedback Integration — Closing the Loop"** with:
   - `re_extract_with_context()` method
   - `FrameSource::GraphAssisted` variant
   - Usage prose for graph-assisted frames
5. Updated Module Structure with `quality.rs`
6. Tests 7-9 (gap rate tracking, graph-context re-extraction, inferred member)
7. Acceptance Criteria 8-12
8. Updated Final Statement

### MD-6 (Acquisition) — 12 additions
1. Updated prerequisite for feedback loop types
2. `source_composition_id` + `source_atom_id` fields in `KnowledgeGap`
3. `GapSource::ExtractionFailure` variant
4. `action: Option<RecallAction>` field in `AcquisitionDecision`
5. **Revised `select_strategy()`** — produces concrete `RecallAction` for PassiveRecall
6. **NEW section: "Graph Role Candidate Lookup"** with `graph_find_role_candidate()`
7. **NEW section: "User Answer Merge"** with `process_user_answer_merge()`
8. **NEW section: "Feedback Loop — Closing the Gap Detection Cycle"** with closed-loop diagram
9. **Revised Executive Integration** — handles RecallAction::EnrichComposition and ReExtractFrame
10. Tests 9-13 (traceability, RecallAction production, merge, full cycle, fallback)
11. Acceptance Criteria 11-15
12. Updated Final Statement

### MD-4 (GovernBeliefs) — 5 additions
1. Updated prerequisite for feedback loop types
2. **NEW section: "Re-Governance After Enrichment"** with:
   - `re_govern_composition()` method
   - `GovernanceUpdate` struct
   - `is_sufficiently_complete()` helper
   - Enrichment-triggered transition rules table
3. SeedAnchor re-anchoring note after enrichment
4. Tests 8-12 (New→Candidate, Quarantine→Candidate, complete→Stable, no auto-ground, contradiction)
5. Acceptance Criteria 10-14

### MD-5 (Executive) — 8 additions
1. Updated prerequisite for feedback loop types
2. Analytical mode now includes gap detection + enrichment
3. Reflective mode includes gap detection + enrichment + weak frame re-extraction
4. `max_enrichment_rounds` in `ComputeBudget` (0/1/2 per mode)
5. **Revised Analytical ingest** — full enrichment loop with DetectGaps → SelectAcquisition → EnrichComposition/ReExtractFrame
6. **NEW section: "Enrichment Loop — Closing the Feedback Cycle"** with diagram
7. Tests 7-10 (Analytical enrichment, Reactive skip, Reflective rounds, early stop)
8. Acceptance Criteria 9-13

## Closed Loop Diagram

```
ExtractFrame → SemanticAtom(Event, conf=0.35)
                      ↓
            IngestAtoms → Composition(Event, missing Arg0Agent)
                      ↓
            GovernBeliefs → (New, Observed)
                      ↓
            [graph matures, user provides context]
                      ↓
            DetectGaps → KnowledgeGap(MissingFieldGap, source_comp=comp_42)
                      ↓
            SelectAcquisition
             ├── PassiveRecall → RecallAction::EnrichComposition
             │     ↓
             │   EnrichComposition → add Arg0Agent to comp_42
             │     ↓
             │   GovernBeliefs re-evaluation → (Candidate, Inferred)
             │     ↓
             │   confidence rises from 0.35 → 0.55
             │
             ├── AskUser → InquiryQuestion("Who performed this action?")
             │     ↓
             │   User: "Raymond"
             │     ↓
             │   process_user_answer_merge() → EnrichmentRequest
             │     ↓
             │   EnrichComposition → add Arg0Agent="Raymond" to comp_42
             │     ↓
             │   GovernBeliefs re-evaluation → (Candidate, Observed for new member)
             │
             └── Deferred → gap noted, no action (SelfStudy in Phase 2)
```

## Key Design Decisions

1. **Enriched members enter as EpistemicState::Inferred** — never auto-Observed. Only repeated independent evidence can transition to Grounded.
2. **Enrichment is promotional by default** — compositions that gain evidence get a chance to advance lifecycle.
3. **User answers merge into existing compositions** via EnrichmentRequest, not separate orphan atoms.
4. **ExtractionQualityTracker** identifies weak rules so ReExtractFrame knows which extractions to re-run with graph context.
5. **max_enrichment_rounds** bounds the loop per cognitive mode (0/1/2).
6. **source_composition_id** in KnowledgeGap enables traceability from gap back to the composition that needs repair.

## Post-Audit Fixes — 10-Gap Resolution

> Date: 2026-05-15 (same day, post-audit)
> All 10 gaps from user audit have been addressed.

### Compile Blockers (Fixed)

| Gap | Issue | Fix |
|-----|-------|-----|
| 4 | `Reflect` transform called but never defined | MD-5 now has full `Reflect` Transform with `ReflectConfig`, `ReflectionFinding`, `ReflectionAction`, and `transform()` implementation |
| 5 | `ReasoningState.update()` and `goal_met` undefined | MD-5 now defines `update()`, `check_goal_met()` per `ReasoningGoal` variant, `ReflectionLoopResult` |
| 6 | `graph_has_relevant_context()` and `graph_has_grounding_evidence()` undefined | MD-6 now has full implementations with per-gap-type logic and 2 grounding strategies |
| 10 | `recent_events` not in `PipelineContext` | MD-3 now has `recent_events: Vec<SemanticAtom>` with sliding window (50) and `record_event()` |

### Correctness Blockers (Fixed)

| Gap | Issue | Fix |
|-----|-------|-----|
| 1 | `PipelineEngine.ingest()` is hardcoded if-else, not DAG | MD-3 now has `TransformNode`, `register()`, `execute_dag()` with topological sort, `register_default_pipeline()` |
| 2 | `check_promotions()` has no criteria | MD-4 now has `can_promote_to_stable()`, `can_promote_to_grounded()`, `can_promote_hypothesis_to_inferred()` with explicit thresholds + `PromotionVerdict` |
| 9 | SeedAnchor gives free confidence boost | MD-4 now checks `has_alignment_data` — weight=0.0 when no data, proportional scaling otherwise |

### Medium Severity (Fixed)

| Gap | Issue | Fix |
|-----|-------|-----|
| 3 | `Contradicted→Grounded` no mechanism | MD-4 now has `ContradictionResolution` with `ResolutionType` enum and `resolve_contradiction()` |
| 7 | `KnowledgeGap` missing `source_composition_id` | MD-6 now has `source_composition_id: Option<CompositionId>` + `source_atom_id: Option<String>` |
| 8 | `AtomType::Token` excluded from gap detection | MD-3 defines `AtomType::AmbiguousToken`; MD-6 now has `AtomType::AmbiguousToken` match arm producing `AmbiguousReferenceGap` or `MissingFieldGap` based on graph context |

## Post-Audit Fixes Round 2 — 9-Gap Resolution

> Date: 2026-05-15 (same day, second audit)
> All 9 new gaps from fresh audit have been addressed.

### Compile Blockers (Fixed)

| Gap | Issue | Fix |
|-----|-------|-----|
| 2 | `original_text` not in Composition — `ReExtractionRequest` had no source | MD-3 adds `source_text: Option<String>` to Composition. MD-5's `run_enrichment_loop()` reads it via `comp.source_text` |
| 6 | `resolve_from_graph()` undefined in MD-6 | MD-6 now has `resolve_ambiguous_from_graph()` — recency-based pronoun resolution (most recent Arg0Agent/Patient) |
| 7 | `age_in_batches()`, `has_recent_contradiction()`, `provenance_source_count()` undefined | MD-3 adds `batch_seen: usize` + `contradiction_batches: Vec<usize>` to Composition + `impl Composition` with all 3 methods |

### Correctness Bugs (Fixed)

| Gap | Issue | Fix |
|-----|-------|-----|
| 3 | `re_govern_composition()` bypassed `can_promote_to_grounded()` criteria | MD-4 now uses `can_promote_to_grounded()` instead of raw confidence check. Enrichment from 1 source can no longer auto-Ground |
| 5 | `extract_missing_role_from_description()` parsed role from string — fragile | MD-6 adds `missing_role: Option<SemanticRole>` to KnowledgeGap (structured field). String parser removed. All consumers use `gap.missing_role` |
| 9 | Mode selection was global (`graph.has_contradictions()`) | MD-5 now uses `graph.neighborhood_for(&input_keywords)` for local scope. Only input-relevant contradictions trigger Reflective mode |

### Medium Severity (Fixed)

| Gap | Issue | Fix |
|-----|-------|-----|
| 1 | Enrichment loop copy-pasted identically in Analytical + Reflective | MD-5 extracts `run_enrichment_loop()` method — single implementation, called from both modes |
| 4 | `detect_contradiction()` only scanned Event compositions | MD-4 now checks same-type compositions + cross-type (HiddenMeaning vs Event) + equivalence mismatch for non-Event types |
| 8 | `is_voice_confusion()` couldn't distinguish duplicates from voice confusion | MD-4 adds provenance check: same roles + different origin_id = voice confusion; same origin_id = duplicate |
