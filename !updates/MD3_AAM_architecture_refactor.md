# MD-3 — AAM Architecture Refactor After Semantic Ingestion Upgrade

## Context

Assume the following have already been completed:

- MD-1: RSVS Semantic Frame Compiler
- MD-2: Pre-Ingest Meaning Reasoner / Hidden Meaning Compiler

This document defines how the **entire AAM architecture** must be adjusted after these foundational upgrades.

This is not a feature patch.

This is an architectural refactor directive.

---

# Executive Summary

AAM originally assumed ingestion would primarily derive meaning through:

- token activation
- co-occurrence
- node promotion
- contextual activation
- sense induction from active nodes
- structural reasoning after ingest

That assumption is now obsolete.

Because MD-1 and MD-2 fundamentally change the semantic quality of incoming knowledge.

Previously:

```text
Text
→ tokens
→ promoted nodes
→ sense induction
→ reasoning
```

Now:

```text
Text
→ Semantic Frame Compiler
→ Event Frames
→ Pre-Ingest Meaning Reasoner
→ Hidden Meaning Candidates
→ RSVS semantic ingest
→ reasoning
```

This changes the role of nearly every layer.

---

# Core Architectural Shift

Old worldview:

```text
Meaning emerges mostly after ingest.
```

New worldview:

```text
Meaning begins before ingest.
```

Important clarification:

This does NOT replace RSVS reasoning.

Instead:

Pre-ingest reasoning provides structured semantic candidates so RSVS can reason with higher-quality primitives.

Correct model:

```text
Pre-Ingest = semantic compiler / hypothesis generator
RSVS = structural long-term reasoning engine
```

---

# Full AAM Architecture (New)

```text
Layer 0
────────────────────────
Raw Text
Semantic Frame Compiler
Pre-Ingest Meaning Reasoner
Semantic Candidate Generator
Frame Normalizer
Frame Validator

Layer 1
────────────────────────
RSVS Graph Core
Node Registry
Sense Engine
Composition Engine
Grounding Engine
Structural Similarity
Substitution Analysis
Reflection
Convergence
Pattern Memory

Layer 2
────────────────────────
Predictive Completion
Situation Modeling
Latent Signal Synthesis
Cross-Pathway Pattern Completion
Hypothesis Expansion

Layer 3
────────────────────────
Deductive Reasoning
Conflict Resolution
Evidence Chain Construction
Appraisal
Conclusion Formation

Layer 4
────────────────────────
Narrative Surface Generator
(diffusion body / renderer)
```

---

# Global Design Principle

AAM should no longer treat raw text as the canonical meaning source.

Canonical semantic primitive becomes:

```text
EventFrame
HiddenMeaningCandidate
Structured Composition
```

Not:

```text
raw token
flat co-occurrence
```

---

# MD-1 + MD-2 Artifacts That Must Become Global Types

## EventFrame

Canonical structured event meaning.

Required global type.

## HiddenMeaningCandidate

Canonical hidden semantic hypothesis.

Required global type.

## CompositionHint

Required shared ingest contract.

## RoleRef

Typed semantic role references.

## ConflictType

Unified contradiction/conflict taxonomy.

---

# Layer-by-Layer Refactor

# LAYER 0 — FULL REBUILD

## Old Role

Previously:

- token extraction
- sentence splitting
- co-occurrence preparation

## New Role

Layer 0 becomes semantic compilation infrastructure.

Pipeline:

```text
Raw Text
→ tokenizer
→ clause segmentation
→ syntax parsing
→ semantic frame extraction
→ normalization
→ hidden meaning generation
→ candidate validation
```

## New Components

Required:

```text
layer0/
  tokenizer
  clause_segmenter
  dependency_parser
  semantic_frame_compiler
  frame_normalizer
  frame_validator
  pre_ingest_reasoning
  hidden_candidate_generator
```

## Remove Assumption

No downstream layer should assume raw tokens are primary truth.

---

# LAYER 1 — RSVS CORE REFIT

RSVS remains the heart.

But ingestion semantics change dramatically.

---

## 1. Graph Core

### Old

Graph stores:

- nodes
- edges
- compositions

### New

Graph must support event-structured meaning.

Add typed edge categories:

```rust
Predicate
Arg0Agent
Arg1Patient
Arg2Recipient
Cause
Purpose
Location
Time
Instrument
Polarity
SourceEvent
HiddenCandidate
PatternType
```

Without typed semantic edges, event meaning collapses into flat relation soup.

---

## 2. Sense Engine

### Old Assumption

Sense emerges from contextual activation overlap.

### New Adjustment

Sense induction must support:

```text
surface semantic roles
hidden meaning candidates
structured event compositions
```

Meaning:

a sense can be induced from:

```text
predicate + arg0 + arg1 + cause
```

not merely token overlap.

---

### New Sense Sources

Sense induction input sources:

```text
explicit frame structure
hidden candidate structure
context activation
existing graph compositions
```

---

## 3. Grounding Engine

Major change required.

Old model risks:

absence = contradiction
co-occurrence-based confirmation
weak semantic specificity

New model:

```text
confirming evidence
contradicting evidence
neutral evidence
```

Grounding must understand role semantics.

Example:

```text
Raymond membuat aplikasi
Raymond tidak membuat aplikasi
```

This is contradiction.

But:

```text
Raymond membuat aplikasi
Raymond berjalan ke kantor
```

is neutral.

---

## 4. Structural Similarity

Must become role-aware.

Old risk:

```text
Raymond membuat aplikasi
Aplikasi membuat Raymond
```

token similarity appears high.

But semantics differ completely.

New similarity must compare:

```text
predicate alignment
role alignment
role reversal
semantic edge compatibility
```

---

## 5. Substitution Analysis

Must become semantic substitution analysis.

Example:

```text
raja ↔ ratu
```

still valid.

But event substitution also needed:

```text
Raymond membuat aplikasi
Raymond membangun sistem
```

Possible substitution:

```text
membuat ↔ membangun
aplikasi ↔ sistem
```

Role-aware substitution required.

---

## 6. Pattern Memory

Pattern storage must evolve.

Store patterns like:

```text
problem → solution
agent → action → tool
cause → action
goal → action
```

not only abstract composition co-occurrence.

---

# LAYER 2 — PREDICTIVE LAYER REFIT

This layer becomes much stronger.

---

## 1. Predictive Completion

Previously:

pattern continuation from node activation.

Now:

event completion.

Example:

Input:

```text
manual process too slow
Raymond builds software
```

Prediction:

```text
software likely solves manual inefficiency
```

---

## 2. Situation Modeling

This becomes much more important.

Situation should aggregate:

```text
events
hidden candidates
conflicts
goals
agents
environment
```

Situation becomes structured world-state.

---

## 3. Latent Signal Synthesis

This should absorb MD-2 outputs.

Examples:

```text
motivation
pain point
operational inefficiency
goal tension
role anomaly
```

---

## 4. Hypothesis Expansion

Should extend hidden meaning candidates into larger scenario hypotheses.

Example:

```text
manual pain
software creation
office beneficiary
```

Expanded hypothesis:

```text
workflow automation initiative
```

---

# LAYER 3 — REASONING REFIT

This layer requires major redesign.

---

## 1. Deductive Reasoning

Must shift from node reasoning to event reasoning.

Canonical unit:

```text
EventFrame
HiddenMeaningCandidate
SituationState
```

Not raw nodes.

---

## 2. Conflict Resolution

Must use typed conflict taxonomy.

Required conflicts:

```rust
PolarityConflict
PurposeConflict
AgentConflict
PatientConflict
CauseConflict
TemporalConflict
LocationConflict
RoleReversal
SemanticContradiction
```

---

## 3. Evidence Chains

Evidence chain should reference:

```text
source events
hidden candidates
grounded senses
supporting patterns
```

Not vague node activation.

---

## 4. Appraisal

Appraisal should classify:

```text
confirmed
weak
conflicted
speculative
unsupported
```

---

## 5. Conclusion Formation

Conclusion should emerge from:

```text
deduction
abduction
grounding
conflict resolution
situation state
```

---

# DIFFUSION BODY / NARRATIVE LAYER

Major contract change.

---

## Old Input

Likely:

```text
graph summary
reasoning chain
pattern outputs
```

## New Input

Should receive:

```json
{
  "events": [],
  "hidden_meanings": [],
  "situation_state": {},
  "evidence_chain": [],
  "conflicts": [],
  "confidence_summary": {}
}
```

Narrative body becomes renderer, not meaning discoverer.

---

# EXISTING MODULE ADJUSTMENTS

## Abductive Reasoning

Must evolve from:

```text
seed overlap
```

to:

```text
event-role hypothesis generation
```

---

## Pattern Mining

Must mine:

```text
role-aware event patterns
```

not only composition frequency.

---

## Cross Pathway Synthesis

Must synthesize:

```text
event contradictions
goal conflicts
role anomalies
multi-event causal chains
```

---

## Reflection

Should inspect:

```text
hidden candidate correctness
conflict consistency
grounding health
pattern stability
```

---

## Convergence

Should merge semantically equivalent events:

Example:

```text
Raymond membuat aplikasi
Aplikasi dibuat oleh Raymond
```

These should converge.

---

# MIGRATION STRATEGY

Recommended phases:

---

## Phase 1

Integrate MD-1 types.

Introduce:

```text
EventFrame
RoleRef
Typed semantic edges
```

---

## Phase 2

Integrate MD-2 hidden meaning candidates.

Introduce:

```text
HiddenMeaningCandidate
rule-based semantic hypotheses
```

---

## Phase 3

Refit Layer 1 reasoning.

Update:

- grounding
- structural similarity
- substitution
- convergence

---

## Phase 4

Refit Layer 2 predictive reasoning.

---

## Phase 5

Refit Layer 3 deductive reasoning.

---

## Phase 6

Refit narrative/diffusion contract.

---

# BACKWARD COMPATIBILITY

Decision needed.

Option A:

Full migration.

Old token-centric ingest deprecated.

Option B:

Hybrid compatibility mode.

Recommended:

```text
Hybrid transition mode first
then deprecate old path
```

---

# ACCEPTANCE CRITERIA

Architecture is considered successfully upgraded if:

1. Raw text is no longer primary semantic primitive.
2. EventFrame is canonical structured input.
3. HiddenMeaningCandidate becomes global reasoning primitive.
4. Structural similarity becomes role-aware.
5. Grounding distinguishes neutral vs contradiction.
6. Convergence merges semantic equivalents.
7. Pattern mining becomes event-aware.
8. Predictive layer uses situation state.
9. Deductive reasoning consumes event-level evidence.
10. Narrative layer acts as renderer, not discoverer.

---

# Final Statement

The AAM refactor changes the architecture from:

```text
token-driven symbolic reasoning
```

to:

```text
structured semantic event reasoning
```

This makes AAM capable of learning not merely:

```text
what co-occurred
```

but:

```text
what happened
why it happened
what it implies
how events relate
what conflicts
what patterns emerge
```

This is a fundamental architectural evolution.
