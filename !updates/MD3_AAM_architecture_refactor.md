# MD-3 — AAM Architecture Refactor After Semantic Ingestion Upgrade (Adjusted for Implementation)

> **Adjustment Note (v11.0 alignment):** This document has been revised from a "full rebuild"
> directive into a **hybrid additive refactor** plan. Key changes from original spec:
> - Token-based ingest is NOT deprecated — it remains the foundation
> - Frame-based ingest is an **additive Layer 0.5**, not a Layer 0 replacement
> - Layer count stays at 4 (Layer 0-3), NOT expanded to 7
> - "Raw text is no longer primary semantic primitive" is REJECTED — it remains primary
> - All 1,081 existing tests must remain green at every phase
> - Migration is additive-only; no existing code is deleted or rewritten
> - Architectural shift is described as evolution, not revolution

---

## Context

Assume the following have been completed:

- MD-1: RSVS Semantic Frame Compiler (at least Phase 1: rule-based)
- MD-2: Pre-Ingest Meaning Reasoner (at least Phase 1: 3 core rules)

This document defines how the **AAM architecture** should evolve after these foundational upgrades.

This is an **evolutionary refactor**, not a ground-up rebuild.

---

## Executive Summary

AAM originally derived meaning through:

```text
tokens → co-occurrence → node promotion → sense induction → reasoning
```

MD-1 and MD-2 add a **parallel enrichment path**:

```text
sentences → event frames → hidden meaning candidates → enriched RSVS ingest
```

Both paths coexist. Token ingest remains the foundation. Frame ingest enriches it.

Previous (token-only):

```text
Text → tokens → RSVS → reasoning
```

Now (hybrid):

```text
Text → tokens → RSVS → reasoning
  └→ sentences → frames → hidden meanings → enriched RSVS → richer reasoning
```

---

## Core Architectural Principle

### What Changed

```text
Meaning CAN begin before ingest (when input is sentence-like).
```

### What Did NOT Change

```text
Token-based ingest is still the primary path.
Raw text is still a valid semantic primitive.
Node promotion and co-occurrence are still essential.
```

### Correct Model

```text
Pre-Ingest = semantic enrichment compiler / hypothesis generator
RSVS = structural long-term reasoning engine (unchanged)
Token Ingest = foundation (always active)
Frame Ingest = enhancement (active when sentence detected)
```

---

## Architecture — Hybrid 4-Layer Model

The architecture stays at 4 layers. New capabilities are ADDED within layers, not as new layers.

```text
Layer 0 — Perceptual Ingest
────────────────────────────────────────────────────
  [EXISTING] tokenizer, sentence splitter, co-occurrence
  [NEW]      sentence detection
  [NEW]      rule-based frame extraction (MD-1 Phase 1)
  [NEW]      pre-ingest reasoning (MD-2 Phase 1)
  [NEW]      frame ingest adapter

Layer 1 — RSVS Memory Core
────────────────────────────────────────────────────
  [EXISTING] graph, senses, attention, compositions, grounding
  [EXISTING] structural similarity, substitution, reflection
  [EXISTING] convergence, pattern memory, consolidation
  [NEW]      semantic role edges (SemanticRole type)
  [NEW]      extended HiddenMeaningType variants
  [NEW]      frame composition nodes

Layer 2 — Predictive & Situational Reasoning
────────────────────────────────────────────────────
  [EXISTING] predictive completion, situation modeling
  [EXISTING] latent signal synthesis, cross-pathway
  [NEW]      event-aware completion (uses frame structure)
  [NEW]      hidden candidate absorption into signals
  [DEFERRED] full event-driven prediction (needs Phase 2+)

Layer 3 — Deductive Reasoning & Narrative
────────────────────────────────────────────────────
  [EXISTING] deductive reasoning, conflict resolution
  [EXISTING] evidence chains, appraisal, conclusion
  [DEFERRED] event-level evidence chains (needs MD-4)
  [DEFERRED] narrative contract change (needs MD-4+)
```

---

## Layer 0 — Additive Enhancement (NOT Rebuild)

### Old Components (UNCHANGED)

```text
layer0/
  base/       # base abstractor
  text/       # text abstractor
  adapter/    # pipeline adapter
  ...
```

### New Components (ADDED)

```text
layer0/
  frame_compiler/        # MD-1: Semantic Frame Compiler
    mod.rs               # public API
    types.rs             # EventFrame, Polarity, Voice, FrameSource, SemanticRole
    rule_extractor.rs    # Phase 1: rule-based extraction
    sentence_detect.rs   # heuristic sentence detection
    ingest_adapter.rs    # EventFrame → RSVS bridge
    tests.rs

  pre_ingest_reasoning/  # MD-2: Pre-Ingest Meaning Reasoner
    mod.rs               # public API
    types.rs             # HiddenMeaningCandidate, etc.
    rules.rs             # reasoning rules
    scorer.rs            # confidence scoring
    mapper.rs            # candidate → RSVS mapping
    tests.rs
```

### How They Integrate

The existing `ingest_text()` in `pipeline/ingest.rs` is extended, not replaced:

```rust
fn ingest_text(&mut self, text: &str) -> IngestStats {
    // === EXISTING PATH (unchanged) ===
    let mut stats = self.tokenize_and_ingest(text);

    // === NEW PATH (additive) ===
    if self.config.frame_compiler_enabled && is_sentence_like(text) {
        if let Some(frame) = self.frame_compiler.extract(text) {
            let hidden = self.pre_ingest_reasoner.reason_on_frame(&frame, &self.context());
            stats.merge(self.frame_ingest_adapter.ingest_frame_with_candidates(&frame, &hidden));
        }
    }

    stats
}
```

This is a **wrapper**, not a rewrite. The existing token path runs first, always. Frame enrichment runs second, only when applicable.

---

## Layer 1 — RSVS Core Extension (NOT Refit)

RSVS remains the heart. Changes are additive.

### 1. Graph Core — New Edge Category

Add `SemanticRole` as a parallel edge type alongside existing `RelationType`. No existing edge handling code is modified.

```rust
// NEW — does not replace RelationType
pub enum SemanticRole {
    Predicate, Arg0Agent, Arg1Patient, Arg2Recipient,
    Cause, Purpose, Location, Time,
    SourceEvent, HiddenCandidate, PatternType,
}
```

### 2. Sense Engine — Extended Input Sources

Sense induction already uses contextual activation overlap. Frame-based compositions add a new input source:

```text
EXISTING sources:
  token co-occurrence
  contextual activation overlap
  composition patterns

NEW source:
  event frame structure (predicate + arg0 + arg1 + cause)
  hidden meaning candidate structure
```

The sense engine's `induce()` method gains an optional `frame_context` parameter. If absent, behavior is identical to current.

### 3. Grounding Engine — No Change Yet

Grounding already distinguishes confirming vs contradicting evidence via `GroundingEvidence`. No change needed until MD-4 adds epistemic governance.

### 4. Structural Similarity — Frame-Aware Extension

Current `structural_similarity()` compares composition structure. With `SemanticRole` edges, similarity can become role-aware:

```rust
fn structural_similarity(a: &Composition, b: &Composition) -> f32 {
    let mut score = existing_similarity(a, b);  // unchanged

    // NEW: bonus for role alignment
    if has_semantic_role_edges(a) && has_semantic_role_edges(b) {
        let role_alignment = compute_role_alignment(a, b);
        score = score * 0.7 + role_alignment * 0.3;
    }

    score
}
```

This is backward compatible — if compositions lack semantic role edges, the existing similarity formula is used unchanged.

### 5. Substitution Analysis — Role-Aware Extension

Same pattern: existing substitution analysis continues to work. Role-aware substitution is an enhancement:

```text
EXISTING: raja ↔ ratu (token substitution)
NEW:      membuat ↔ membangun (predicate substitution, same role structure)
          aplikasi ↔ sistem (patient substitution in same event structure)
```

### 6. Pattern Memory — Event-Aware Patterns

Pattern storage can now include:

```text
EXISTING: abstract composition co-occurrence patterns
NEW:      problem → solution patterns
          agent → action → tool patterns
          cause → action patterns
          goal → action patterns
```

New pattern types use `SemanticRole` edges. Existing pattern types unchanged.

---

## Layer 2 — Predictive Enhancement (NOT Refit)

### 1. Predictive Completion — Event Completion

Current prediction: pattern continuation from node activation.

New capability: event completion from partial frame.

```text
Input:  cause = "proses manual lambat" + action = "membuat" + patient = ???
Prediction: patient likely = tool/software/system
```

This is additive — only activates when frame context is available.

### 2. Situation Modeling — Extended Input

Situation aggregation can now include:

```text
EXISTING: nodes, senses, compositions, conflicts
NEW:      event frames, hidden candidates, goals, agents
```

### 3. Latent Signal Synthesis — Absorb MD-2 Outputs

MD-2 candidates (motivation, pain point, inefficiency) become latent signal sources.

```text
EXISTING signals: gap-based, pathway-based
NEW signals: hidden meaning candidates from pre-ingest reasoning
```

### 4. Hypothesis Expansion — Defer

Full hypothesis expansion from hidden meaning candidates is deferred to Phase 2. Current system already has `AbductiveHypothesis` and `AbductiveEngine`.

---

## Layer 3 — Reasoning Enhancement (Defer Major Changes)

Layer 3 changes are deferred until MD-4 (epistemic governance) is implemented. Current reasoning engines work fine with token-based evidence.

Future capabilities (after MD-4):

```text
- Event-level evidence chains (not just node-level)
- Typed conflict taxonomy (not just pathway-level)
- Role-aware contradiction detection
- Grounding-aware appraisal
```

---

## Existing Module Adjustments

### Abductive Reasoning — No Change

Current abductive engine works with seed overlap. It does NOT need to change to "event-role hypothesis generation" yet. That's a Phase 2+ enhancement.

### Pattern Mining — Extended Input

Current pattern mining works with composition frequency. It can be extended to also mine role-aware event patterns, but this is additive.

### Cross Pathway Synthesis — Extended Input

Current synthesis engine produces `SynthesisResult`. MD-2 candidates map INTO `SynthesisResult.hidden_meaning`. No architectural change needed.

### Reflection — No Change

Current reflection engine inspects belief stability and grounding. No change until MD-4.

### Convergence — Frame-Aware Extension

Current convergence merges similar compositions. With semantic role edges, convergence can recognize that active and passive versions of the same event should merge:

```text
"Raymond membuat aplikasi" ≈ "Aplikasi dibuat oleh Raymond"
```

This is an enhancement, not a rewrite.

---

## Migration Strategy — Additive Phases

### Phase 1 — Types + Frame Compiler (MD-1 Phase 1)

Introduce:

```text
EventFrame, Polarity, Voice, FrameSource
SemanticRole
FrameCompiler variant in EdgeSource
frame_compiler_enabled in PipelineConfig
```

**Tests affected**: 0 existing tests modified. New tests added for frame extraction.

### Phase 2 — Pre-Ingest Reasoner (MD-2 Phase 1)

Introduce:

```text
HiddenMeaningCandidate, MeaningNodeRef, CandidateStatus, CompositionHint
Extended HiddenMeaningType variants
Extended HiddenMeaning optional fields
Pre-ingest reasoning module
```

**Tests affected**: 0 existing tests modified. New tests added for reasoning rules.

### Phase 3 — Hybrid Pipeline Integration

Connect frame compiler + reasoner to ingest pipeline:

```text
Modify ingest_text() to add frame path alongside token path
Add frame_ingest_adapter to pipeline
Add pre_ingest_reasoner to pipeline
```

**Tests affected**: 0 existing tests modified. Existing tests still use token path. Frame path tested separately.

### Phase 4 — Sense + Similarity Enhancement

Extend sense induction and structural similarity to use frame context:

```text
Optional frame_context parameter in sense induction
Role-aware similarity bonus in structural_similarity
Event-aware pattern mining
```

**Tests affected**: 0 existing tests modified. All changes are additive with fallback to existing behavior.

### Phase 5 — Advanced Integration (after MD-4)

Event-level reasoning, typed conflicts, narrative contract change.

---

## Backward Compatibility — Absolute Rule

```text
ALL 1,081 EXISTING TESTS MUST REMAIN GREEN AT EVERY PHASE.

No existing test may be modified to accommodate new features.
No existing type may be changed in a breaking way.
No existing pipeline behavior may change without feature flag.
```

Techniques for guaranteed compatibility:

1. `#[non_exhaustive]` on all enums (already in place)
2. `Option<T>` for all new struct fields
3. Feature flags for new paths (`frame_compiler_enabled`)
4. Fallback to existing behavior when frame context absent
5. New modules are independent, not modifications of existing ones

---

## What "Raw Text as Semantic Primitive" Means

The original MD-3 stated:

```text
Raw text is no longer primary semantic primitive.
```

This is **rejected**. The adjusted position:

```text
Raw text / tokens remain the PRIMARY semantic primitive.
Event frames are a SECONDARY semantic primitive that ENRICHES
the graph when sentence-level structure is available.

Early-stage graphs are sparse. Token co-occurrence is the
foundation that builds the graph from nothing. Frame-based
reasoning operates on top of that foundation.
```

Correct relationship:

```text
Tokens  →  graph foundation  →  sense induction  →  meaning
Frames  →  event enrichment  →  hidden candidates →  deeper meaning
```

Both contribute. Neither replaces the other.

---

## Acceptance Criteria

Architecture evolution is acceptable if:

1. Token ingest path is UNCHANGED and always active
2. Frame ingest path is ADDITIVE and feature-flagged
3. All 1,081 existing tests remain green
4. New types are `#[non_exhaustive]` and backward compatible
5. New struct fields are `Option<T>` and backward compatible
6. Layer count remains at 4 (not expanded to 7)
7. No existing module is deleted or rewritten
8. `ingest_text()` wraps existing logic, does not replace it
9. Structural similarity falls back to existing formula when no frame context
10. Pipeline behavior is identical to v11.0 when `frame_compiler_enabled = false`

---

## Final Statement

The AAM refactor evolves the architecture from:

```text
token-driven symbolic reasoning (only)
```

to:

```text
token-driven + frame-enriched symbolic reasoning
```

This is additive evolution, not architectural revolution. The token foundation remains. Frame enrichment builds on top. Every downstream MD can leverage frame structure when available, while the system continues to function perfectly without it.
