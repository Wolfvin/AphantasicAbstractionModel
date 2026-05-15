# MD-5 — Executive Cognition / Meta-Control Layer (Adjusted for Implementation)

> **Adjustment Note (v11.0 alignment):** This document has been revised for implementation
> readiness. Key changes from original spec:
> - Phase 1: 3 cognitive modes only (Reactive, Analytical, Reflective), not 12
> - Phase 1: Stop conditions + compute budget, not full executive architecture
> - Layer position: NOT a new Layer 5 — integrated into existing Layer 2/3 orchestration
> - ExecutiveState simplified from 12 states to 4
> - Goal arbitration, task decomposition, attention router DEFERRED to Phase 2
> - max_reasoning_depth aligned with current system (3-5), not 8
> - Failure recovery deferred — current system already has convergence + reflection safety
> - All 1,081 existing tests must remain green
> - Executive cognition is premature without mature specialists — Phase 1 is MINIMAL

---

## Context

Assume completed:

- MD-1: Semantic Frame Compiler (Phase 1)
- MD-2: Pre-Ingest Meaning Reasoner (Phase 1)
- MD-3: AAM Architecture Refactor (hybrid additive)
- MD-4: Epistemic Truth & Belief Governance (Phase 1)

This document defines the executive cognition layer for AAM.

**Important**: Executive cognition requires mature specialist engines (Predictive, Situation, Latent Signal, Cross-Pathway) to orchestrate. These are still evolving. Therefore Phase 1 is intentionally minimal — just enough meta-control to prevent runaway reasoning and wasted compute.

---

## Core Problem

By MD-4, AAM can:

```text
parse semantic structure
infer hidden meaning
reason structurally
manage belief states
detect contradictions
govern truth confidence
maintain provenance
```

But without meta-control:

```text
reasoning may trigger unnecessarily on trivial input
deep reasoning may waste compute on simple queries
reflection may loop without bound
contradictions may consume all resources
trivial tasks may use heavy reasoning
```

Phase 1 addresses the most critical of these: **unbounded reasoning** and **unnecessary deep analysis**.

---

## Mission — Phase 1 (MINIMAL)

Create a **minimal** deterministic executive control that:

1. Selects cognitive mode based on input complexity
2. Enforces stop conditions on reasoning
3. Governs compute budget

Everything else (attention routing, goal arbitration, task decomposition, failure recovery) is deferred.

---

## Architecture Position

Executive cognition is NOT a new layer. It is a **control overlay** that integrates into the existing pipeline orchestration.

```text
Layer 0: Perceptual Ingest
Layer 1: RSVS Memory Core
Layer 2: Predictive & Situational Reasoning
Layer 3: Deductive Reasoning & Narrative

Executive Control (cross-cutting):
  - mode selection (called before reasoning starts)
  - budget enforcement (called during reasoning)
  - stop conditions (called after each reasoning step)
```

Executive control is a **service** consumed by the pipeline, not a separate layer.

---

## Cognitive Modes — Phase 1: 3 Modes Only

### Reactive Mode

Use when:

```text
simple direct retrieval
low uncertainty (confidence > 0.8)
known pattern (pattern familiarity high)
no contradictions
```

Behavior:

```text
max_reasoning_depth = 2
no reflection
no deep analysis
fast lookup + shallow sense activation
```

### Analytical Mode

Use when:

```text
structured decomposition needed
moderate uncertainty (0.4 < confidence < 0.8)
new event frame detected
hidden meaning candidates present
```

Behavior:

```text
max_reasoning_depth = 4
1 reflection cycle allowed
sense induction + frame analysis + hidden meaning processing
```

### Reflective Mode

Use when:

```text
high uncertainty (confidence < 0.4)
contradictions detected
belief state instability
epistemic risk (hypothesis near promotion threshold)
```

Behavior:

```text
max_reasoning_depth = 5
2 reflection cycles allowed
full analysis + grounding review + belief state audit
```

### Mode Selection Logic

```rust
fn select_cognitive_mode(context: &ExecutiveContext) -> CognitiveMode {
    let has_contradiction = context.contradiction_count > 0;
    let avg_confidence = context.average_confidence;

    if has_contradiction || avg_confidence < 0.4 {
        return CognitiveMode::Reflective;
    }

    if avg_confidence < 0.8 || context.has_hidden_meaning_candidates {
        return CognitiveMode::Analytical;
    }

    CognitiveMode::Reactive
}
```

Simple, deterministic, auditable. No 12-mode complexity.

---

## Executive State — Simplified

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutiveState {
    Idle,       // no active reasoning
    Reasoning,  // actively reasoning (mode-specific depth)
    Stopped,    // hit stop condition
}
```

Not 12 states. 3 states. The pipeline already manages its own state transitions — executive control only governs **when to start, how deep to go, and when to stop**.

---

## Compute Budget — Phase 1

```rust
pub struct ComputeBudget {
    pub max_reasoning_depth: usize,       // default: 3 (current system range)
    pub max_reflection_loops: usize,      // default: 1
    pub max_branching_factor: usize,      // default: 5
    pub max_hypothesis_count: usize,      // default: 10
    pub time_budget_ms: u64,              // default: 5000
}
```

Mode-specific budgets:

```text
Reactive:   depth=2, reflection=0, branching=3, hypotheses=5,   time=1000ms
Analytical: depth=4, reflection=1, branching=5, hypotheses=10,  time=5000ms
Reflective: depth=5, reflection=2, branching=7, hypotheses=15,  time=10000ms
```

Note: max depth is 5, not 8. Current system operates at depth 3-5. Setting depth 8 would allow runaway reasoning before specialist engines are mature enough to produce useful results at that depth.

---

## Stop Conditions — CRITICAL

Stop conditions prevent infinite cognition loops.

```rust
pub struct StopCondition {
    pub confidence_sufficient: f32,      // stop if confidence > this (default: 0.9)
    pub no_new_evidence_loops: usize,    // stop after N loops with no new evidence (default: 2)
    pub budget_exhausted: bool,          // stop when budget depleted
    pub goal_satisfied: bool,            // stop when reasoning goal met
}

impl StopCondition {
    pub fn should_stop(&self, state: &ReasoningState) -> bool {
        if state.confidence >= self.confidence_sufficient {
            return true;  // confidence sufficient
        }
        if state.loops_without_new_evidence >= self.no_new_evidence_loops {
            return true;  // diminishing returns
        }
        if state.elapsed_ms >= state.budget.time_budget_ms {
            return true;  // budget exhausted
        }
        if state.goal_met {
            return true;  // goal satisfied
        }
        false
    }
}
```

This is the most important part of Phase 1. Without stop conditions, the system WILL loop.

---

## Executive Context

```rust
pub struct ExecutiveContext {
    // Input assessment
    pub input_complexity: InputComplexity,  // Simple, Moderate, Complex
    pub has_frame_context: bool,            // EventFrame available?
    pub has_hidden_meaning: bool,           // HiddenMeaningCandidate available?

    // Current reasoning state
    pub average_confidence: f32,
    pub contradiction_count: usize,
    pub active_hypotheses: usize,
    pub loops_completed: usize,

    // Budget
    pub budget: ComputeBudget,
    pub stop_condition: StopCondition,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InputComplexity {
    Simple,    // single token or short phrase
    Moderate,  // single sentence with structure
    Complex,   // multi-sentence or multi-event
}
```

---

## Integration with Pipeline

The executive control is called from the pipeline at specific points:

```rust
impl Rsvs {
    pub fn context_query(&mut self, query: &str) -> QueryResult {
        // 1. Assess input and select mode
        let mode = self.executive.select_mode(query, &self.graph);

        // 2. Apply budget
        let budget = self.executive.budget_for_mode(&mode);

        // 3. Run reasoning with stop conditions
        let mut result = self.reason_with_budget(query, &budget, &mode);

        // 4. Check stop conditions after each step
        while !self.executive.should_stop(&result.state) {
            result = self.reasoning_step(query, &mut result, &budget);
        }

        result
    }
}
```

Existing pipeline methods are WRAPPED, not replaced. When `executive_control_enabled = false`, pipeline behaves identically to v11.0.

---

## Integration with MD-4 (Epistemic Governance)

Executive must respect belief governance:

```text
- Cannot promote weak hypothesis recklessly
- Must not ignore contradiction arbitration
- Reflective mode triggered when belief states are unstable
- Stop conditions consider epistemic risk
```

But: executive does NOT override epistemic governance. It controls reasoning depth, not belief truth.

---

## Phase 2 — DEFERRED Features

These are architecturally important but premature without mature specialists:

### Additional Cognitive Modes (9 more)

```text
FastLookup, Deductive, Abductive, Predictive,
ConflictResolution, Verification, Exploratory,
MemoryRecovery, SituationModeling
```

### Attention Router

Selective attention over large graphs. Requires graph-scale benchmarking first.

### Goal Arbitration

Prioritizing competing goals. Requires multi-goal scenarios first.

### Task Decomposition

Splitting complex tasks. Requires evidence that current reasoning can't handle complexity.

### Self Monitoring + Failure Recovery

`CognitiveHealth` metrics and recovery strategies. Requires failure mode analysis first.

### Executive Working Memory

Short-term reasoning context. Current session state may suffice.

---

## Module Structure (Phase 1: Minimal)

```text
layer1/crates/rsvs-core/src/
  executive/
    mod.rs              // public API: select_mode(), should_stop(), budget_for_mode()
    types.rs            // CognitiveMode, ExecutiveState, ComputeBudget, StopCondition, ExecutiveContext
    mode_selection.rs   // deterministic mode selection logic
    budget.rs           // budget computation per mode
    stop.rs             // stop condition evaluation
    tests.rs            // unit tests
```

6 files for Phase 1. Not 14.

---

## Required Tests

### Test 1 — Simple Query → Reactive Mode

Input: single token query, high pattern familiarity

Expected: `CognitiveMode::Reactive`, depth=2

### Test 2 — Contradiction → Reflective Mode

Input: query where contradictions exist

Expected: `CognitiveMode::Reflective`, depth=5

### Test 3 — New Event Frame → Analytical Mode

Input: sentence with EventFrame, moderate confidence

Expected: `CognitiveMode::Analytical`, depth=4

### Test 4 — Stop Condition: Confidence Sufficient

Reasoning reaches confidence 0.92 (threshold 0.90)

Expected: stop condition triggered

### Test 5 — Stop Condition: Budget Exhausted

Reasoning exceeds time budget

Expected: stop condition triggered, graceful halt

### Test 6 — Stop Condition: No New Evidence

2 consecutive loops produce no new evidence

Expected: stop condition triggered

### Test 7 — Executive Disabled = Current Behavior

With `executive_control_enabled = false`, pipeline behaves exactly like v11.0

---

## Acceptance Criteria

Phase 1 is acceptable if:

1. 3 cognitive modes implemented: Reactive, Analytical, Reflective
2. Mode selection is deterministic and auditable
3. Compute budget is enforced per mode
4. Stop conditions prevent infinite reasoning loops
5. Max reasoning depth is 5 (aligned with current system)
6. Executive control is feature-flagged (can be disabled)
7. Pipeline behavior is identical to v11.0 when disabled
8. All 1,081 existing tests remain green
9. Module structure is 6 files, not 14
10. No 12-mode complexity, no 12-state state machine

---

## Final Statement

MD-5 Phase 1 introduces minimal executive control: mode selection, compute budgets, and stop conditions. This prevents the most dangerous failure modes (runaway reasoning, compute waste) without adding premature complexity. Full executive cognition (attention routing, goal arbitration, task decomposition) requires mature specialist engines that don't exist yet. Build the minimum, prove it works, then expand.
