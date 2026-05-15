# MD-4 — Epistemic Truth & Belief Governance Layer

## Context

Assume completed:

- MD-1: RSVS Semantic Frame Compiler
- MD-2: Pre-Ingest Meaning Reasoner
- MD-3: AAM Architecture Refactor

This document defines the epistemic governance layer required to prevent knowledge pollution, false certainty, contradiction collapse, and unstable long-term reasoning.

---

# Core Problem

AAM now handles multiple knowledge classes:

- surface event facts
- hidden meaning candidates
- inferred hypotheses
- pattern-derived beliefs
- grounded confirmations
- contradictions
- reflections
- predictive projections

Without governance:

```text
candidate inference may become treated as fact
temporary assumptions may persist forever
contradictions may poison graph state
weak evidence may dominate strong evidence
old beliefs may remain despite disproof
equivalent facts may fragment
```

Result:

```text
memory corruption
semantic pollution
belief drift
hallucinated certainty
reasoning collapse
```

---

# Mission

Create a deterministic epistemic governance framework.

Responsibilities:

- truth state management
- uncertainty handling
- provenance tracking
- contradiction arbitration
- belief revision
- temporal confidence decay
- source trust evaluation
- hypothesis quarantine
- semantic deduplication
- evidence lineage

---

# Core Principle

RSVS must distinguish:

```text
what was observed
what was inferred
what is hypothesized
what is weak
what is contradicted
what is deprecated
```

Knowledge is not binary.

---

# Knowledge Classes

## 1. Observation

Direct extracted event.

Example:

```text
Raymond membuat aplikasi.
```

Status:

```text
Observed
```

---

## 2. Derived Inference

Rule-derived hidden meaning.

Example:

```text
Likely workflow automation initiative
```

Status:

```text
Inferred
```

---

## 3. Hypothesis

Unconfirmed scenario reasoning.

Example:

```text
Raymond may be solving office inefficiency.
```

Status:

```text
Hypothesis
```

---

## 4. Grounded Belief

Repeatedly supported.

Status:

```text
Grounded
```

---

## 5. Contradicted Belief

Evidence opposes belief.

Status:

```text
Contradicted
```

---

## 6. Deprecated Belief

No longer trusted.

Status:

```text
Deprecated
```

---

# Belief State Machine

```text
Observed
   ↓
Candidate
   ↓
Inferred
   ↓
Grounded
   ↓
Contradicted
   ↓
Deprecated
```

Alternative transitions:

```text
Candidate → Rejected
Hypothesis → Grounded
Grounded → Contradicted
Contradicted → Recovered
```

---

# Required Types

## BeliefState

```rust
pub enum BeliefState {
    Observed,
    Candidate,
    Inferred,
    Hypothesis,
    Grounded,
    Weak,
    Contradicted,
    Deprecated,
    Rejected,
    Recovered,
}
```

---

## KnowledgeClaim

```rust
pub struct KnowledgeClaim {
    pub claim_id: String,
    pub semantic_identity: SemanticIdentity,
    pub source_type: SourceType,
    pub belief_state: BeliefState,
    pub confidence: f32,
    pub provenance: ProvenanceChain,
    pub supporting_evidence: Vec<EvidenceRef>,
    pub contradicting_evidence: Vec<EvidenceRef>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub decay_profile: DecayProfile,
}
```

---

## SourceType

```rust
pub enum SourceType {
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

---

# Provenance Chain

Every claim must know:

```text
where it came from
what rule created it
what events support it
what transformations occurred
```

Example:

```text
Raw Text
→ Frame Extraction
→ Hidden Meaning Rule #CAUSE_ACTION_PATTERN
→ Candidate hm_12
→ Grounded by events e14/e29/e31
```

Never allow orphan beliefs.

---

# Confidence Governance

Confidence must not be static.

Confidence dimensions:

```text
source reliability
support count
contradiction count
recency
semantic consistency
grounding quality
```

Example formula:

```text
confidence =
base_source_weight
+ support_bonus
- contradiction_penalty
- decay_penalty
+ consistency_bonus
```

Clamp:

```text
0.0 to 1.0
```

---

# Source Trust Weights

Example defaults:

```text
DirectObservation       0.85
FrameExtraction         0.80
HiddenMeaningRule       0.65
PatternMining           0.60
AbductiveInference      0.50
PredictiveCompletion    0.45
Reflection              0.40
```

Configurable.

---

# Contradiction Arbitration

Contradictions must not instantly destroy beliefs.

Required process:

1. detect semantic equivalence
2. classify contradiction type
3. compare source trust
4. compare evidence volume
5. compare recency
6. compare consistency with world state

---

## Conflict Types

```rust
pub enum ConflictType {
    PolarityConflict,
    PurposeConflict,
    AgentConflict,
    PatientConflict,
    CauseConflict,
    TemporalConflict,
    LocationConflict,
    SemanticContradiction,
    RoleReversal,
    EquivalenceMismatch,
}
```

---

# Temporal Decay

Some beliefs age.

Examples:

```text
current location
temporary state
active intention
prediction
```

Decay profiles:

```rust
Stable
SlowDecay
MediumDecay
FastDecay
Volatile
```

Example:

```text
"Raymond is in office"
```

should decay faster than:

```text
"Raymond built software"
```

---

# Hypothesis Quarantine

Critical rule:

Hypotheses must not contaminate core grounded memory immediately.

Quarantine store:

```text
epistemic/hypotheses/
```

Conditions to promote:

```text
repeated support
independent evidence
pattern consistency
no strong contradiction
```

---

# Semantic Identity

Equivalent claims must unify.

Examples:

```text
Raymond membuat aplikasi
Aplikasi dibuat oleh Raymond
```

Same semantic identity.

Need canonical normalization.

---

# Evidence Lineage

Every belief should explain itself.

Query:

```text
Why do you believe X?
```

Response path:

```text
Claim X
supported by event e1
derived from frame f1
triggered by rule R12
reinforced by pattern P7
```

---

# Revision Protocol

When contradiction arrives:

```text
Do not overwrite blindly.
```

Protocol:

```text
detect
classify
compare trust
update confidence
change state
preserve provenance
```

---

# Reflection Integration

Reflection should audit:

```text
weak beliefs
stale beliefs
contradicted beliefs
duplicate beliefs
overconfident hypotheses
```

---

# Predictive Layer Rules

Predictions must enter as:

```text
Hypothesis
```

never:

```text
Grounded
```

unless independently confirmed.

---

# Pattern Mining Rules

Patterns discovered statistically:

```text
Inferred
```

not direct truth.

---

# Human Override

Human explicit corrections should support:

```text
promotion
demotion
forced contradiction
forced deprecation
manual truth anchoring
```

---

# Storage Architecture

Suggested:

```text
epistemic/
  claims/
  provenance/
  contradictions/
  hypotheses/
  decay/
  audits/
```

---

# Required Modules

```text
epistemic/
  mod.rs
  belief_state.rs
  confidence.rs
  provenance.rs
  arbitration.rs
  decay.rs
  semantic_identity.rs
  hypothesis_quarantine.rs
  revision.rs
  audit.rs
  tests.rs
```

---

# Required Tests

## Test 1

Direct observation + contradiction.

Expected:

confidence decreases
state changes

---

## Test 2

Equivalent passive/active claims unify.

---

## Test 3

Weak hypothesis remains quarantined.

---

## Test 4

Strong repeated evidence promotes belief.

---

## Test 5

Stale volatile belief decays.

---

## Test 6

Prediction does not become grounded automatically.

---

# Acceptance Criteria

System succeeds if:

- hypotheses cannot silently become facts
- contradictions are governed
- provenance is complete
- equivalent claims unify
- stale claims decay
- confidence is explainable
- reasoning can justify beliefs

---

# Final Statement

AAM without epistemic governance risks becoming structurally intelligent but epistemically unstable.

This layer ensures:

```text
memory integrity
belief discipline
truth governance
auditable reasoning
controlled uncertainty
```
