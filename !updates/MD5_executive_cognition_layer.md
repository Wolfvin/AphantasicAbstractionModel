# MD-5 — Executive Cognition / Meta-Control Layer

## Context

Assume completed:

- MD-1: Semantic Frame Compiler
- MD-2: Pre-Ingest Meaning Reasoner
- MD-3: AAM Architecture Refactor
- MD-4: Epistemic Truth & Belief Governance

This document defines the executive cognition layer for AAM.

This is the central control system.

Equivalent conceptual role:

```text
prefrontal cortex
executive controller
meta-cognitive governor
cognitive operating system
```

---

# Core Problem

By MD-4, AAM can:

- parse semantic structure
- infer hidden meaning
- reason structurally
- manage belief states
- detect contradictions
- govern truth confidence
- maintain provenance

But this is not enough.

Without executive control:

```text
reasoning may trigger unnecessarily
deep reasoning may waste compute
reflection may loop forever
conflicts may consume all resources
trivial tasks may use heavy reasoning
high-stakes tasks may use shallow reasoning
multiple goals may compete
attention may explode across graph
search branching may become unstable
```

Result:

```text
compute inefficiency
reasoning drift
decision instability
resource exhaustion
looping cognition
goal confusion
```

---

# Mission

Create a deterministic executive cognition layer.

Responsibilities:

- cognitive mode selection
- reasoning depth control
- attention routing
- compute budgeting
- goal arbitration
- reflection triggering
- verification escalation
- uncertainty-aware strategy switching
- stop condition enforcement
- self-monitoring
- task decomposition
- failure recovery

---

# Core Principle

AAM must not only reason.

AAM must decide:

```text
whether to reason
how to reason
how deeply to reason
when to stop reasoning
```

---

# Executive Architecture Position

New architecture:

```text
Layer 0
Semantic ingest

Layer 1
RSVS memory core

Layer 2
Prediction / situation modeling

Layer 3
Reasoning engines

Layer 4
Epistemic governance

Layer 5
Executive cognition (THIS DOCUMENT)

Layer 6
Narrative rendering
```

---

# Executive State Machine

```rust
pub enum ExecutiveState {
    Idle,
    Observe,
    Interpret,
    Route,
    Reason,
    Reflect,
    Verify,
    Arbitrate,
    Conclude,
    Commit,
    Suspend,
    Recover,
}
```

---

# Cognitive Modes

## Required Modes

```rust
pub enum CognitiveMode {
    Reactive,
    FastLookup,
    Analytical,
    Deductive,
    Abductive,
    Predictive,
    Reflective,
    ConflictResolution,
    Verification,
    Exploratory,
    MemoryRecovery,
    SituationModeling,
}
```

---

# Mode Descriptions

## Reactive

Use when:

```text
simple direct retrieval
low uncertainty
known pattern
```

Cheap path.

---

## FastLookup

Use when:

```text
graph memory already contains answer
```

No deep reasoning.

---

## Analytical

Use when:

```text
structured decomposition needed
```

---

## Deductive

Use when:

```text
facts are sufficient
rules available
```

---

## Abductive

Use when:

```text
hidden explanation required
```

---

## Predictive

Use when:

```text
future completion or projection needed
```

---

## Reflective

Use when:

```text
self-audit required
low confidence
internal inconsistency
```

---

## ConflictResolution

Use when:

```text
contradictions detected
```

---

## Verification

Use when:

```text
high stakes
high uncertainty
epistemic risk
```

---

## Exploratory

Use when:

```text
novel problem
weak prior structure
```

---

## MemoryRecovery

Use when:

```text
answer likely exists in memory
```

---

## SituationModeling

Use when:

```text
multi-event scenario reasoning needed
```

---

# Trigger System

Rules for mode activation.

---

## Trigger Examples

### High uncertainty

```text
confidence < threshold
```

Action:

```text
Reactive → Analytical
Analytical → Verification
```

---

### Contradiction detected

Trigger:

```text
conflict count > 0
```

Action:

```text
ConflictResolution
```

---

### Novel event

Trigger:

```text
semantic familiarity low
```

Action:

```text
Exploratory
```

---

### Pattern confidence high

Trigger:

```text
pattern familiarity high
uncertainty low
```

Action:

```text
FastLookup
```

---

### High stakes

Trigger:

```text
decision critical
irreversible
epistemic sensitivity
```

Action:

```text
Verification
```

---

# Attention Router

Critical subsystem.

Graph scale requires selective cognition.

Responsibilities:

```text
choose relevant nodes
choose relevant senses
choose event clusters
choose hypotheses
choose evidence
choose conflict targets
```

---

## Attention Scoring

Candidate score dimensions:

```text
semantic relevance
goal relevance
recency
grounding strength
uncertainty contribution
conflict significance
pattern familiarity
```

Formula example:

```text
attention_score =
semantic relevance
+ goal weight
+ uncertainty boost
+ contradiction boost
+ recency weight
```

---

# Compute Budget Manager

Executive must control cost.

---

## Budget Dimensions

```rust
pub struct ComputeBudget {
    pub max_reasoning_depth: usize,
    pub max_branching_factor: usize,
    pub max_hypothesis_count: usize,
    pub max_reflection_loops: usize,
    pub max_verification_cycles: usize,
    pub max_attention_targets: usize,
    pub time_budget_ms: u64,
}
```

---

## Example Policy

Low complexity:

```text
depth 2
branching 3
reflection 0
```

High complexity:

```text
depth 8
branching 12
reflection 3
verification enabled
```

---

# Goal Arbitration

AAM may face multiple competing goals.

Examples:

```text
answer query
resolve contradiction
repair belief
explore hypothesis
predict future
validate claim
```

Need prioritization.

---

## Goal Type

```rust
pub enum GoalType {
    AnswerQuestion,
    ResolveConflict,
    VerifyTruth,
    RecoverMemory,
    PredictOutcome,
    ExploreUnknown,
    RepairBelief,
    ImproveSituationModel,
}
```

---

## Arbitration Strategy

Priority dimensions:

```text
urgency
importance
epistemic risk
resource cost
user objective
stability impact
```

---

# Task Decomposition

Complex tasks should split.

Example:

```text
who did what, why, and what happens next?
```

Subtasks:

```text
identify event
infer motivation
simulate consequence
verify uncertainty
compose answer
```

---

# Stop Conditions

Mandatory.

Without stop conditions:

infinite cognition loops.

---

## Stop Triggers

```text
confidence sufficient
budget exhausted
no new evidence
conflict unresolved but diminishing returns
max reflection reached
goal satisfied
```

---

# Reflection Policy

Reflection must not run blindly.

---

## Trigger Reflection If

```text
confidence low
internal contradiction
high novelty
unstable hypothesis
overconfident weak evidence
```

---

## Reflection Questions

Executive should ask:

```text
Did I overfit?
Did I ignore contradictions?
Did I miss simpler explanation?
Is evidence weak?
Is current strategy appropriate?
```

---

# Verification Escalation

High-risk cognition must escalate.

Examples:

```text
belief conflict
critical conclusion
low-confidence decision
contradiction under high stakes
```

Verification actions:

```text
re-check evidence
re-run deduction
compare alternative hypothesis
demand stronger grounding
```

---

# Self Monitoring

Executive self-observation.

State metrics:

```rust
pub struct CognitiveHealth {
    pub uncertainty: f32,
    pub contradiction_load: f32,
    pub branching_pressure: f32,
    pub confidence_stability: f32,
    pub novelty_level: f32,
    pub loop_risk: f32,
}
```

---

## Failure Signals

Examples:

```text
loop risk high
contradictions growing
confidence oscillating
attention explosion
branching explosion
```

---

# Failure Recovery

Recovery strategies:

```text
reduce branching
switch to verification
switch to fast lookup
suspend weak hypotheses
prune low-value paths
fallback to stable evidence
```

---

# Executive Memory

Executive should remember recent cognitive context.

Short-term working memory:

```text
active goals
current hypotheses
active evidence
current strategy
visited reasoning paths
```

---

# Situation Routing

Executive should decide:

```text
single event reasoning?
multi-event scenario?
conflict arbitration?
prediction?
```

---

# Interaction with Layer 3

Executive does not replace reasoning engines.

Executive orchestrates them.

Relationship:

```text
Layer 3 = specialists
Layer 5 = conductor
```

---

# Interaction with Epistemic Layer

Executive must respect belief governance.

Cannot:

```text
promote weak hypothesis recklessly
ignore contradiction arbitration
```

---

# Required Modules

Suggested:

```text
executive/
  mod.rs
  state_machine.rs
  mode_selection.rs
  triggers.rs
  attention_router.rs
  budget.rs
  goal_arbitration.rs
  decomposition.rs
  reflection_policy.rs
  verification.rs
  self_monitoring.rs
  recovery.rs
  working_memory.rs
  tests.rs
```

---

# Required Traits

```rust
pub trait CognitiveStrategy {
    fn execute(&self, context: &ExecutiveContext) -> StrategyResult;
}
```

---

# Required Tests

## Test 1

Simple query.

Expected:

```text
FastLookup
```

---

## Test 2

Contradiction detected.

Expected:

```text
ConflictResolution
```

---

## Test 3

High uncertainty.

Expected:

```text
Verification
```

---

## Test 4

Novel problem.

Expected:

```text
Exploratory
```

---

## Test 5

Reflection loop prevention.

Expected:

```text
stop condition enforced
```

---

## Test 6

Budget exhaustion.

Expected:

```text
graceful halt
```

---

# Acceptance Criteria

Success if:

- AAM chooses reasoning strategy explicitly
- attention remains bounded
- compute remains governed
- contradictions trigger arbitration
- uncertainty changes behavior
- reflection is bounded
- failure recovery works
- goals are prioritized
- reasoning specialists are orchestrated cleanly

---

# Final Statement

Without executive cognition:

AAM has intelligence modules but no disciplined mind.

This layer transforms AAM from:

```text
collection of reasoning systems
```

into:

```text
coherent controlled cognition
```
