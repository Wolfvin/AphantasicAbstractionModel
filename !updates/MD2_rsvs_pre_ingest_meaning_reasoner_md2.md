# MD-2 — Pre-Ingest Meaning Reasoner / Hidden Meaning Compiler (Adjusted for Implementation)

> **Adjustment Note (v11.0 alignment):** This document has been revised for implementation
> readiness. Key changes from original spec:
> - EXTENDS existing `HiddenMeaningType` (6 variants) instead of creating parallel hierarchy
> - EXTENDS existing `SynthesisResult` instead of replacing
> - Rule 5 negative markers made language-agnostic (graph-guided, not hardcoded word lists)
> - Module structure simplified from 8 files to 5
> - Phased: 3 core rules first, remaining rules in Phase 2
> - Explicit type alignment with existing `types.rs` types
> - All 1,081 existing tests must remain green

---

## Context

Assume **MD-1: RSVS Semantic Frame Compiler** has been implemented (at least Phase 1: rule-based extraction).

That means the system can transform raw text into deterministic semantic event frames without using an LLM.

Example input:

```text
Raymond membuat aplikasi untuk kantor karena proses manual terlalu lambat.
```

Expected MD-1 output:

```json
{
  "event_id": "e1",
  "predicate": "membuat",
  "arg0_agent": "Raymond",
  "arg1_patient": "aplikasi",
  "purpose": "kantor",
  "cause": "proses manual terlalu lambat",
  "polarity": "Positive",
  "voice": "Active",
  "source": "RuleBased"
}
```

MD-2 starts from this point.

---

## Mission

Build a **Pre-Ingest Meaning Reasoner**.

This layer runs **after Semantic Frame Compiler** and **before RSVS graph ingestion**.

Its job is to discover **hidden meaning candidates** that are not directly visible on the surface of the sentence but are implied by the relationship between:

- atoms
- semantic roles
- event frames
- existing RSVS senses
- context
- composition patterns
- cause-effect structure
- purpose/goal structure
- contradiction or tension between roles

Core statement:

> The Pre-Ingest Meaning Reasoner exists so RSVS does not only store what the text says, but also receives structured candidates for what the text implies.

---

## Non-Negotiable Constraint

Do **not** use LLMs.

This must be:

- deterministic
- auditable
- rule-guided
- graph-guided
- explainable
- testable

---

## High-Level Pipeline

```text
Raw Text
→ Semantic Frame Compiler        // MD-1
→ EventFrames
→ Pre-Ingest Meaning Reasoner    // THIS DOCUMENT (MD-2)
→ Hidden Meaning Candidates
→ RSVS Sense / Composition Ingest
→ Grounding + Reflection
```

Short form:

```text
Text → Frame → Hidden Meaning → RSVS
```

---

## Type Alignment with Existing Codebase

### Existing `HiddenMeaningType` (types.rs:559-573) — MUST EXTEND, NOT REPLACE

```rust
// EXISTING — DO NOT MODIFY
pub enum HiddenMeaningType {
    AffectiveDisguise,
    SocialConcealment,
    PerformativeMask,
    TraumaPattern,
    PowerDynamic,
    Emergent,
}
```

### Extended `HiddenMeaningType` — ADD new variants

```rust
#[non_exhaustive]  // Already non_exhaustive for forward compat
pub enum HiddenMeaningType {
    // === EXISTING (meaning-pathway focused) ===
    AffectiveDisguise,
    SocialConcealment,
    PerformativeMask,
    TraumaPattern,
    PowerDynamic,
    Emergent,

    // === NEW (event-structure focused, from MD-2) ===
    ProblemSolutionPattern,      // cause + action + object → problem-solution
    MotivationInference,         // cause-based motivation
    GoalInference,               // purpose-based goal
    AgentResponsibility,         // agent + active predicate → responsibility
    CauseEffectPattern,          // explicit cause-effect chain
    ToolUsePattern,              // tool created due to pain point
    InefficiencySignal,          // negative state in cause slot
    PolarityConflict,            // same event, opposite polarity
    PurposeConflict,             // same event, different purpose
    RoleAnomaly,                 // agent-patient reversal
}
```

The `#[non_exhaustive]` attribute already on the enum ensures this is a backward-compatible addition. All existing match statements will continue to compile (they must already have wildcard arms).

### Existing `SynthesisResult` (types.rs:526-541) — EXTEND, NOT REPLACE

```rust
// EXISTING
pub struct SynthesisResult {
    pub node_id: NodeId,
    pub sense_id: SenseId,
    pub gap: GapAnnotation,
    pub conflict: PathwayConflict,
    pub hidden_meaning: HiddenMeaning,
    pub confidence: f32,
    pub meaning_node_id: Option<NodeId>,
}
```

The reasoner produces `HiddenMeaningCandidate` (new struct), which maps INTO the existing `HiddenMeaning` field during RSVS ingest. Do NOT create a parallel type hierarchy.

### Existing `HiddenMeaning` (types.rs:544-556) — ADD fields

```rust
// EXISTING — add optional fields, backward compatible
pub struct HiddenMeaning {
    pub description: String,
    pub target_node: NodeId,
    pub seed_trace: Vec<NodeId>,
    pub meaning_type: HiddenMeaningType,
    pub evidence_strength: f32,
    // === NEW optional fields from MD-2 ===
    pub source_event_id: Option<String>,      // which frame produced this
    pub rule_id: Option<String>,               // which reasoning rule
    pub composition_hints: Option<Vec<CompositionHint>>, // ingest hints
    pub candidate_status: Option<CandidateStatus>,       // lifecycle state
}
```

All new fields are `Option<T>` — backward compatible. Existing code that constructs `HiddenMeaning` without these fields continues to work.

---

## New Types

### HiddenMeaningCandidate (NEW)

The primary output of the reasoner. Maps to `HiddenMeaning` during ingest.

```rust
/// Structured hidden meaning candidate produced by Pre-Ingest Reasoner.
/// Enters RSVS as provisional composition with candidate status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenMeaningCandidate {
    pub candidate_id: String,
    pub source_event_id: String,
    pub meaning_type: HiddenMeaningType,  // uses EXTENDED enum
    pub description: String,
    pub nodes: Vec<MeaningNodeRef>,
    pub composition_hints: Vec<CompositionHint>,
    pub evidence_roles: Vec<String>,      // "CAU_cause", "ARG1_patient", etc.
    pub rule_id: String,
    pub confidence: f32,
    pub status: CandidateStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeaningNodeRef {
    pub role: String,        // "problem", "solution", "agent", etc.
    pub label: String,       // node label in graph
    pub node_id: Option<NodeId>,  // resolved graph node (if exists)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CandidateStatus {
    Candidate,
    Confirmed,
    Contradicted,
    Deprecated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionHint {
    pub role: String,
    pub node_label: String,
    pub confidence: f32,
}
```

---

## Core Reasoning Rules — Phased Implementation

### Phase 1 — 3 Core Rules (IMMEDIATE)

These are the highest-value rules that unlock the most reasoning power.

#### Rule 1 — Cause + Action + Object → Problem-Solution Pattern

Trigger condition:

```text
frame.cause IS Some
AND frame.predicate IS Some
AND frame.arg1_patient IS Some
AND predicate belongs to action/create/fix/build/solve family
```

Predicate family detection is **graph-guided**, not hardcoded:

```rust
fn is_action_predicate(predicate: &str, graph: &Graph) -> bool {
    // 1. Check if predicate node has semantic edges to known action nodes
    // 2. Check if predicate's senses overlap with action/create categories
    // 3. Fallback: simple morphological check (me- prefix in Indonesian, -ify/-ize in English)
    graph_has_action_sense(predicate) || has_action_morphology(predicate)
}
```

Output:

```json
{
  "meaning_type": "ProblemSolutionPattern",
  "nodes": [
    { "role": "problem", "label": "proses manual terlalu lambat" },
    { "role": "solution", "label": "aplikasi" },
    { "role": "agent", "label": "Raymond" }
  ],
  "rule_id": "CAUSE_ACTION_PURPOSE_TO_PROBLEM_SOLUTION"
}
```

#### Rule 2 — Purpose Marker → Goal Inference

Trigger condition:

```text
frame.purpose IS Some
```

Output:

```json
{
  "meaning_type": "GoalInference",
  "nodes": [
    { "role": "goal", "label": "kantor" },
    { "role": "action", "label": "membuat" },
    { "role": "agent", "label": "Raymond" }
  ],
  "rule_id": "PURPOSE_TO_GOAL_INFERENCE"
}
```

#### Rule 3 — Same Event + Opposite Polarity → Direct Contradiction

Trigger condition:

```text
two frames with:
  same predicate
  same agent
  same patient
  opposite polarity
```

This requires comparing current frame against **recent frames in the session**.

Output:

```json
{
  "meaning_type": "PolarityConflict",
  "nodes": [
    { "role": "event_a", "label": "e1" },
    { "role": "event_b", "label": "e2" }
  ],
  "rule_id": "POLARITY_CONFLICT"
}
```

### Phase 2 — Additional Rules (AFTER Phase 1 is stable)

#### Rule 4 — Agent + Active Predicate → Agent Responsibility

Trigger: `voice == Active AND arg0_agent IS Some AND predicate IS Some`

#### Rule 5 — Cause Contains Negative Quality → Inefficiency Signal

Trigger: `cause IS Some AND cause has negative sentiment`

**Important**: Negative sentiment detection must be **graph-guided, not hardcoded**:

```rust
fn has_negative_sentiment(cause_text: &str, graph: &Graph) -> bool {
    // 1. Tokenize cause text
    // 2. For each token, check if graph has negative-valence sense
    // 3. If any token is grounded as negative, return true
    // 4. Fallback: small configurable seed list (NOT hardcoded in rule)
    let tokens = tokenize(cause_text);
    tokens.iter().any(|t| graph_has_negative_valence(t) || seed_negative_list.contains(t))
}
```

This avoids language-specific hardcoding. The seed list is configurable per-language, not baked into the rule.

#### Rule 6 — Tool/Object Created Due to Pain Point → Tool-Use Pattern

Trigger: `ProblemSolutionPattern exists AND arg1_patient is tool-like`

Tool-likeness is graph-guided (similar to action predicate detection).

#### Rule 7 — Passive Normalization

Trigger: `voice == Passive AND agent IS Some AND patient IS Some`

Normalize passive to active equivalent for convergence.

#### Rule 8 — Same Event + Different Purpose → Purpose Conflict

Trigger: `same predicate + same agent + same patient + different purpose`

#### Rule 9 — Agent-Patient Reversal → Role Anomaly

Trigger: `event_a.agent == event_b.patient AND event_a.patient == event_b.agent`

---

## Graph-Guided Enhancement

The reasoner should not rely only on hardcoded rules. It should query RSVS graph to improve confidence.

```rust
fn adjust_confidence_by_graph(
    candidate: &HiddenMeaningCandidate,
    graph: &Graph,
) -> f32 {
    let mut adjusted = candidate.confidence;

    // Check if role-filler nodes exist and have appropriate senses
    for node_ref in &candidate.nodes {
        if let Some(node_id) = node_ref.node_id {
            // If graph confirms the node plays the expected role
            if graph_confirms_role(graph, node_id, &node_ref.role) {
                adjusted += 0.05;
            }
            // If graph contradicts the role
            if graph_contradicts_role(graph, node_id, &node_ref.role) {
                adjusted -= 0.10;
            }
        }
    }

    adjusted.clamp(0.0, 1.0)
}
```

Graph-guided functions:

```text
role_score(node, role)           → how well node fits role
semantic_similarity(node, proto) → distance to prototype
sense_grounding(node)            → grounding strength
composition_overlap(node, proto) → structural overlap
```

---

## Confidence Scoring — Deterministic

Example for ProblemSolutionPattern:

```text
base = 0.40

+0.15 if frame.cause exists
+0.15 if frame.arg1_patient exists
+0.10 if frame.arg0_agent exists
+0.10 if predicate belongs to action family (graph-guided)
+0.10 if cause has negative sentiment (graph-guided)
-0.20 if polarity is negative
```

Clamp: `0.0 <= confidence <= 1.0`

Candidate confidence capped at frame confidence:

```text
candidate_confidence = min(rule_score, frame.confidence)
```

Then optionally adjusted by graph context.

---

## Integration With RSVS — Mapping to Existing Types

HiddenMeaningCandidate maps to existing RSVS structures:

```text
HiddenMeaningCandidate.meaning_type  → HiddenMeaning.meaning_type (extended enum)
HiddenMeaningCandidate.description    → HiddenMeaning.description
HiddenMeaningCandidate.source_event_id → HiddenMeaning.source_event_id (new field)
HiddenMeaningCandidate.rule_id        → HiddenMeaning.rule_id (new field)
HiddenMeaningCandidate.confidence     → HiddenMeaning.evidence_strength
HiddenMeaningCandidate.nodes          → HiddenMeaning.seed_trace (resolved NodeIds)
```

Composition hints become RSVS composition edges:

```text
hm1 --SemanticRole::PatternType--> problem_solution
hm1 --SemanticRole::Cause-->       proses_manual_terlalu_lambat
hm1 --SemanticRole::Arg1Patient--> aplikasi
hm1 --SemanticRole::Arg0Agent-->   Raymond
hm1 --SemanticRole::SourceEvent--> e1
```

All edges have `EdgeSource::FrameCompiler` (shared with MD-1).

Candidate enters RSVS as **provisional**:

```text
status = Candidate
confidence = computed_score
grounding = pending
source = FrameCompiler
```

---

## Module Structure (Simplified)

```text
layer0/
  pre_ingest_reasoning/
    mod.rs                  // public API: reason_on_frame(), reason_on_batch()
    types.rs                // HiddenMeaningCandidate, MeaningNodeRef, CandidateStatus, CompositionHint
    rules.rs               // All reasoning rules (PreIngestRule trait + implementations)
    scorer.rs              // Confidence scoring + graph-guided adjustment
    mapper.rs              // HiddenMeaningCandidate → RSVS ingest mapping
    tests.rs               // Unit tests for all rules
```

5 source files + 1 test file. NOT 8 files. Kept minimal.

The `PreIngestRule` trait:

```rust
pub trait PreIngestRule: Send + Sync {
    fn id(&self) -> &'static str;
    fn applies(&self, frame: &EventFrame, context: &ReasoningContext) -> bool;
    fn generate(&self, frame: &EventFrame, context: &ReasoningContext) -> Vec<HiddenMeaningCandidate>;
}

pub struct ReasoningContext {
    pub recent_frames: Vec<EventFrame>,    // for cross-frame comparison
    pub graph: Arc<Graph>,                 // for graph-guided confidence
}
```

---

## Required Tests

### Test 1 — Problem Solution Pattern (Phase 1, Rule 1)

Input frame:

```json
{
  "event_id": "e1",
  "predicate": "membuat",
  "arg0_agent": "Raymond",
  "arg1_patient": "aplikasi",
  "purpose": "kantor",
  "cause": "proses manual terlalu lambat"
}
```

Expected hidden meaning candidates:

```text
ProblemSolutionPattern   (Rule 1: cause + action + patient)
GoalInference            (Rule 2: purpose exists)
```

### Test 2 — Polarity Conflict (Phase 1, Rule 3)

Input events:

```text
Frame A: Raymond membuat aplikasi (polarity: Positive)
Frame B: Raymond tidak membuat aplikasi (polarity: Negative)
```

Expected:

```text
PolarityConflict
```

### Test 3 — Purpose Conflict (Phase 2, Rule 8)

Input events:

```text
Frame A: Raymond membuat aplikasi untuk kantor
Frame B: Raymond membuat aplikasi untuk sekolah
```

Expected:

```text
PurposeConflict
```

### Test 4 — Role Reversal (Phase 2, Rule 9)

Input events:

```text
Frame A: Raymond membuat aplikasi
Frame B: Aplikasi membuat Raymond
```

Expected:

```text
RoleAnomaly
```

### Test 5 — No Hidden Meaning from Incomplete Frame

Input frame:

```json
{
  "predicate": "membuat"
}
```

Expected: No hidden meaning candidates (no cause, no patient, no purpose).

### Test 6 — Confidence Capped at Frame Confidence

Frame confidence: 0.60
Rule computed score: 0.85

Expected candidate confidence: 0.60 (capped)

### Test 7 — Candidate Status Is Provisional

All candidates from reasoner must have `status: Candidate`, never `Confirmed` or `Grounded`.

---

## Alignment with Existing Codebase

| Existing Type | Relationship | Action |
|---------------|-------------|--------|
| `HiddenMeaningType` (6 variants) | EXTEND with 9 new variants | Backward compatible (`#[non_exhaustive]`) |
| `HiddenMeaning` (struct) | ADD optional fields | Backward compatible (all `Option<T>`) |
| `SynthesisResult` (struct) | Reasoner output maps INTO this | No modification needed |
| `CrossPathwaySynthesis` engine | Reasoner complements, not replaces | Independent module |
| `GapAnnotation`, `GapType` | Separate concern (predictive gaps vs hidden meaning) | No overlap |
| `EdgeSource` | Use `FrameCompiler` variant from MD-1 | Already planned |

---

## Acceptance Criteria

Phase 1 is acceptable if:

1. No LLMs used — all reasoning is deterministic
2. `HiddenMeaningType` extended with 9 new variants (backward compatible)
3. 3 core rules implemented: ProblemSolution, GoalInference, PolarityConflict
4. Each candidate includes: type, source event, evidence roles, rule id, confidence, composition hints
5. Confidence scoring is deterministic and auditable
6. Graph-guided enhancement adjusts confidence (not hardcoded)
7. Candidates map cleanly to existing `HiddenMeaning` during RSVS ingest
8. Negative sentiment detection is graph-guided, not hardcoded word lists
9. All 1,081 existing tests remain green
10. Module structure is 5 files + tests, not 8+

---

## Final Statement

MD-2 extends the existing meaning-pathway infrastructure with event-structure-aware hidden meaning detection. It builds on MD-1's EventFrame output and feeds into RSVS's existing synthesis and composition machinery. Every new type extends or maps to existing types — no parallel hierarchies, no duplicated functionality.
