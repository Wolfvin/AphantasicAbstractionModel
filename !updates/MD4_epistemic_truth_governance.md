# MD-4 — Epistemic Truth & Belief Governance Layer (Adjusted for Implementation)

> **Adjustment Note (v11.0 alignment):** This document has been revised for implementation
> readiness. Key changes from original spec:
> - Phase 1 implements BeliefState + ProvenanceChain ONLY; decay/arbitration deferred
> - UNIFY with existing `GroundingEvidence` instead of creating parallel system
> - Separate `EpistemicConflictType` from existing `ConflictType` (meaning-pathway vs epistemic)
> - Unify `SourceType` with existing `EdgeSource` where possible
> - Simplify `KnowledgeClaim` — start without heavy provenance/evidence/decay fields
> - Use existing `PolicyMeta` as governance entry point, not a new system
> - Hypothesis quarantine aligns with existing `NodeStatus::Quarantine`
> - All 1,081 existing tests must remain green

---

## Context

Assume completed:

- MD-1: RSVS Semantic Frame Compiler (at least Phase 1)
- MD-2: Pre-Ingest Meaning Reasoner (at least Phase 1)
- MD-3: AAM Architecture Refactor (hybrid additive approach)

This document defines the epistemic governance layer required to prevent knowledge pollution, false certainty, contradiction collapse, and unstable long-term reasoning.

---

## Core Problem

AAM now handles multiple knowledge classes:

```text
surface event facts        (from MD-1 EventFrame)
hidden meaning candidates  (from MD-2 Pre-Ingest Reasoner)
inferred hypotheses        (from abductive/predictive engines)
pattern-derived beliefs    (from pattern mining)
grounded confirmations     (from grounding engine)
contradictions             (from conflict detection)
reflections                (from reflection engine)
predictive projections     (from prediction engine)
```

Without governance:

```text
candidate inference may become treated as fact
temporary assumptions may persist forever
contradictions may poison graph state
weak evidence may dominate strong evidence
old beliefs may remain despite disproof
equivalent facts may fragment
```

Result: memory corruption, semantic pollution, belief drift, hallucinated certainty.

---

## Mission

Create a deterministic epistemic governance framework.

**Phase 1 responsibilities** (IMMEDIATE):

- truth state management (BeliefState transitions)
- provenance tracking (where did this belief come from?)
- hypothesis quarantine (prevent premature grounding)

**Phase 2 responsibilities** (DEFERRED):

- contradiction arbitration (trust comparison, evidence volume)
- temporal confidence decay (staleness management)
- semantic deduplication (equivalent claims unify)
- source trust evaluation (reliability weighting)

---

## Core Principle

RSVS must distinguish:

```text
what was observed     → BeliefState::Observed
what was inferred     → BeliefState::Inferred
what is hypothesized  → BeliefState::Hypothesis
what is weak          → BeliefState::Weak
what is contradicted  → BeliefState::Contradicted
what is deprecated    → BeliefState::Deprecated
```

Knowledge is not binary. Belief states govern lifecycle.

---

## Alignment with Existing Codebase

### Existing Governance Types

| Existing Type | Location | Fields/variants |
|---------------|----------|-----------------|
| `GroundingEvidence` | sense.rs:114 | confirming_contexts, contradicting_contexts, last_contradiction, revision_count |
| `GroundingVerdict` | sense.rs:174 | WellGrounded, NeedsReview, NeedsRevision |
| `NodeStatus` | types.rs:614 | New, Candidate, Stable, Deprecated, **Quarantine** |
| `PolicyMeta` | types.rs:769 | policy_version, governance_score, candidate_evidence_pool, status_flip_count, seen_fingerprints |
| `ConflictType` | types.rs:221 | AffectiveSocialMismatch, AffectivePragmaticMismatch, etc. (5 meaning-pathway variants) |
| `AtomRecord` | autonomy.rs:38 | governance_score, candidate_evidence_pool, status_flip_count, access_count |

**Key insight**: Governance infrastructure already exists at the node/sense level. MD-4 adds governance at the **knowledge claim** level — a layer above.

### Relationship: Not Replacement, Layering

```text
Level 3: KnowledgeClaim (belief state, provenance)       ← MD-4 NEW
Level 2: Sense + GroundingEvidence (sense grounding)      ← EXISTING
Level 1: Node + NodeStatus (node lifecycle)               ← EXISTING
Level 0: AtomRecord (activation governance)                ← EXISTING
```

MD-4 does NOT replace `GroundingEvidence` or `NodeStatus`. It adds a higher-level governance layer.

---

## Type Design — Phase 1

### BeliefState (NEW enum)

```rust
/// Epistemic state of a knowledge claim.
/// Governs lifecycle: observation → candidate → inference → grounding → deprecation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BeliefState {
    Observed,      // Direct extracted event (from MD-1)
    Candidate,     // Pre-ingest candidate (from MD-2)
    Inferred,      // Derived by reasoning rule
    Hypothesis,    // Unconfirmed scenario reasoning
    Grounded,      // Repeatedly supported by evidence
    Weak,          // Low evidence, at risk
    Contradicted,  // Evidence opposes
    Deprecated,    // No longer trusted
}
```

### BeliefState Transitions (IMMEDIATE)

```text
Observed → Candidate → Inferred → Grounded
                ↓           ↓
            Rejected    Contradicted → Deprecated
Hypothesis → Grounded (if confirmed)
Grounded   → Contradicted (if new contradiction)
Contradicted → Recovered (if contradiction resolved) [Phase 2]
```

Phase 1 implements these transitions:

```rust
impl BeliefState {
    pub fn can_transition_to(&self, target: &BeliefState) -> bool {
        match (self, target) {
            (Observed, Candidate) => true,
            (Candidate, Inferred) => true,
            (Candidate, Deprecated) => true,  // rejected
            (Inferred, Grounded) => true,
            (Inferred, Contradicted) => true,
            (Hypothesis, Grounded) => true,
            (Hypothesis, Contradicted) => true,
            (Grounded, Contradicted) => true,
            (Contradicted, Deprecated) => true,
            _ => false,
        }
    }
}
```

### ProvenanceChain (NEW struct — Phase 1: lightweight)

```rust
/// Tracks where a belief came from.
/// Phase 1: simple chain. Phase 2: full lineage with transformation records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceChain {
    pub source_type: ProvenanceSource,
    pub source_id: String,              // event_id, rule_id, etc.
    pub timestamp: String,              // ISO 8601
    pub parent_claim_id: Option<String>, // if derived from another claim
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProvenanceSource {
    DirectObservation,
    FrameExtraction,
    HiddenMeaningRule,
    PatternMining,
    AbductiveInference,
    DeductiveReasoning,
    PredictiveCompletion,
    Reflection,
    HumanAssertion,
}
```

Note: `ProvenanceSource` overlaps with `EdgeSource` conceptually but serves a different purpose. `EdgeSource` tracks graph edge origin. `ProvenanceSource` tracks knowledge claim origin. They coexist.

### KnowledgeClaim (NEW struct — Phase 1: minimal)

```rust
/// A governed knowledge claim in the epistemic layer.
/// Phase 1: belief state + provenance + confidence only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeClaim {
    pub claim_id: String,
    pub belief_state: BeliefState,
    pub confidence: f32,
    pub provenance: ProvenanceChain,
    pub created_at: String,
    pub updated_at: String,

    // Link to existing graph structures
    pub node_id: Option<NodeId>,          // if claim is about a node
    pub sense_id: Option<SenseId>,        // if claim is about a sense
    pub event_id: Option<String>,         // if claim originated from an event frame

    // Phase 2 fields (added later, all Option<T>)
    pub supporting_evidence: Option<Vec<EvidenceRef>>,
    pub contradicting_evidence: Option<Vec<EvidenceRef>>,
    pub decay_profile: Option<DecayProfile>,
    pub semantic_identity: Option<SemanticIdentity>,
}
```

All Phase 2 fields are `Option<T>` — backward compatible when added.

### EvidenceRef (Phase 2 — stub for now)

```rust
/// Reference to evidence supporting or contradicting a claim.
/// Phase 2 implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub evidence_id: String,
    pub evidence_type: String,  // "event", "sense", "pattern", etc.
    pub strength: f32,
}
```

---

## Type Design — Phase 2 (DEFERRED)

### EpistemicConflictType (NEW — separate from existing ConflictType)

The existing `ConflictType` in `types.rs` has 5 meaning-pathway variants:

```rust
// EXISTING — DO NOT MODIFY
pub enum ConflictType {
    AffectiveSocialMismatch,
    AffectivePragmaticMismatch,
    SocialPragmaticMismatch,
    AffectiveInternalConflict,
    ConnotativeLiteralMismatch,
}
```

MD-4 needs epistemic-level conflict types. These are a DIFFERENT classification axis:

```rust
/// Epistemic-level conflict taxonomy.
/// Different axis from meaning-pathway ConflictType.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EpistemicConflictType {
    PolarityConflict,      // same event, opposite polarity
    PurposeConflict,       // same event, different purpose
    AgentConflict,         // same event, different agent
    PatientConflict,       // same event, different patient
    CauseConflict,         // same event, different cause
    TemporalConflict,      // temporal inconsistency
    LocationConflict,      // location inconsistency
    SemanticContradiction, // general semantic clash
    RoleReversal,          // agent-patient swap
    EquivalenceMismatch,   // should be equivalent but differs
}
```

Rationale for separate type: meaning-pathway conflicts (existing) and epistemic conflicts (new) operate at different levels. A single claim can have both types. Merging them would break existing match statements.

### DecayProfile (Phase 2)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayProfile {
    Stable,       // historical facts, no decay
    SlowDecay,    // long-lived states
    MediumDecay,  // regular information
    FastDecay,    // current states, intentions
    Volatile,     // real-time observations
}
```

### SemanticIdentity (Phase 2)

For deduplicating equivalent claims:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticIdentity {
    pub canonical_form: String,          // normalized representation
    pub equivalent_forms: Vec<String>,   // passive/active, synonym variants
}
```

---

## Hypothesis Quarantine — Align with Existing NodeStatus::Quarantine

The existing `NodeStatus` enum already has a `Quarantine` variant:

```rust
pub enum NodeStatus {
    New, Candidate, Stable, Deprecated, Quarantine,
}
```

MD-4's hypothesis quarantine should USE this existing mechanism:

```text
Hypotheses enter the graph with:
  - node_status = Quarantine
  - belief_state = Hypothesis

Promotion conditions:
  - repeated support (multiple confirming events)
  - independent evidence (different sources confirm)
  - pattern consistency (fits established patterns)
  - no strong contradiction

After promotion:
  - node_status = Stable
  - belief_state = Grounded
```

No new quarantine system needed. Extend existing one.

---

## Integration with GroundingEvidence

`GroundingEvidence` already tracks confirming/contradicting contexts:

```rust
pub struct GroundingEvidence {
    pub confirming_contexts: usize,
    pub contradicting_contexts: usize,
    pub last_contradiction: Option<String>,
    pub revision_count: usize,
}
```

MD-4's `KnowledgeClaim` references grounding as evidence:

```text
KnowledgeClaim.supporting_evidence ← includes GroundingEvidence.confirming_contexts
KnowledgeClaim.contradicting_evidence ← includes GroundingEvidence.contradicting_contexts
```

The relationship:

```text
KnowledgeClaim (epistemic level)
  → references → Sense (has GroundingEvidence)
  → references → Node (has NodeStatus)
```

No duplication. `KnowledgeClaim` is a governance view over existing data.

---

## Confidence Governance — Phase 1 (Simple)

Phase 1 uses a simple formula:

```text
claim.confidence = provenance_source_weight
```

Source weights (defaults, configurable):

```text
DirectObservation       0.85
FrameExtraction         0.80
HiddenMeaningRule       0.65
PatternMining           0.60
AbductiveInference      0.50
PredictiveCompletion    0.45
Reflection              0.40
```

Phase 2 adds:

```text
confidence = base_source_weight
           + support_bonus
           - contradiction_penalty
           - decay_penalty
           + consistency_bonus
```

Clamp: 0.0 to 1.0

---

## Integration with MD-1 and MD-2

### From MD-1 (EventFrame)

```text
EventFrame → KnowledgeClaim
  belief_state = Observed
  provenance.source_type = FrameExtraction
  provenance.source_id = event_id
  confidence = frame.confidence
```

### From MD-2 (HiddenMeaningCandidate)

```text
HiddenMeaningCandidate → KnowledgeClaim
  belief_state = Candidate
  provenance.source_type = HiddenMeaningRule
  provenance.source_id = rule_id
  confidence = candidate.confidence
  node_status = Quarantine (hypothesis quarantine)
```

### Promotion Rules

```text
Candidate → Inferred:
  when candidate receives independent supporting evidence

Inferred → Grounded:
  when 3+ independent confirming contexts exist
  AND no strong contradiction

Hypothesis → Grounded:
  when pattern-consistent AND independently confirmed

Any → Contradicted:
  when strong contradiction with higher-trust source

Contradicted → Deprecated:
  when no recovery within N cycles
```

---

## Module Structure (Phase 1: Minimal)

```text
layer1/crates/rsvs-core/src/
  epistemic/
    mod.rs              // public API: register_claim(), promote(), contradict()
    types.rs            // BeliefState, ProvenanceChain, ProvenanceSource, KnowledgeClaim, EvidenceRef
    belief_state.rs     // BeliefState transition logic
    claim_store.rs      // HashMap<ClaimId, KnowledgeClaim> + query methods
    promotion.rs        // rules for belief state transitions
    tests.rs            // unit tests
```

6 files for Phase 1. Much simpler than the original 10-file structure.

---

## Required Tests

### Test 1 — Belief State Transitions Are Valid

```text
Observed → Candidate → Inferred → Grounded  ✓
Observed → Grounded  ✗ (skip not allowed)
Candidate → Deprecated  ✓ (rejected)
Grounded → Contradicted → Deprecated  ✓
```

### Test 2 — Direct Observation Creates Claim with Correct State

Input: EventFrame with source = FrameExtraction

Expected:

```text
claim.belief_state = Observed
claim.provenance.source_type = FrameExtraction
claim.confidence = frame.confidence
```

### Test 3 — Hidden Meaning Candidate Enters as Candidate + Quarantine

Input: HiddenMeaningCandidate

Expected:

```text
claim.belief_state = Candidate
node.node_status = Quarantine
```

### Test 4 — Promotion from Candidate to Inferred

Condition: 2+ independent supporting events

Expected: `belief_state` changes from `Candidate` to `Inferred`

### Test 5 — Contradiction Demotes Grounded Belief

Condition: strong contradiction from higher-trust source

Expected: `belief_state` changes from `Grounded` to `Contradicted`

### Test 6 — Prediction Does Not Become Grounded Automatically

Condition: claim from PredictiveCompletion source

Expected: `belief_state = Hypothesis`, never auto-promoted to Grounded

### Test 7 — Provenance Chain Is Traceable

Query: "Why do you believe X?"

Expected: chain from claim → source event → frame → rule

---

## Acceptance Criteria

Phase 1 is acceptable if:

1. `BeliefState` enum exists with valid transition logic
2. `ProvenanceChain` tracks claim origin
3. `KnowledgeClaim` has belief state + provenance + confidence
4. Hypothesis quarantine uses existing `NodeStatus::Quarantine`
5. No hypotheses can silently become facts (must pass through promotion rules)
6. Provenance is traceable (every claim knows where it came from)
7. `EpistemicConflictType` is separate from existing `ConflictType`
8. `KnowledgeClaim` references existing `GroundingEvidence` (no duplication)
9. All 1,081 existing tests remain green
10. Module structure is 6 files, not 10

---

## Final Statement

MD-4 adds epistemic governance as a layer ABOVE existing governance mechanisms. It does not replace `GroundingEvidence`, `NodeStatus`, or `PolicyMeta`. It adds belief lifecycle management at the knowledge claim level. Phase 1 implements the minimum viable governance: belief states, provenance, and quarantine. Phase 2 adds decay, arbitration, deduplication, and source trust evaluation.
