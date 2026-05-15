# MD-5 — Executive Cognition (Elegant Architecture)

> **Prerequisite**: MD-3 defines Transform, SemanticAtom, Composition, LifecycleState,
> EpistemicState. MD-4 defines GovernBeliefs + SeedAnchor Transforms.
> This document defines executive control as a **Transform chain orchestrator**.

---

## Mission

Implement executive cognition as a **Transform chain orchestrator**: selects which
transforms to run, how deeply, and when to stop.

Executive cognition is NOT a new layer. It is a **control service** that wraps
the Transform engine with mode selection, budget enforcement, and stop conditions.

---

## Core Principle

AAM must decide:
- whether to reason (mode selection)
- how deeply to reason (budget)
- when to stop reasoning (stop conditions)

These decisions are deterministic, based on input atom type and graph state.

---

## Cognitive Modes — 3 Modes

### Reactive

```text
Trigger: simple input, high confidence, known pattern
Behavior:
  - Run Tokenize only (no frame extraction or reasoning)
  - Fast lookup from graph
  - max_depth = 2
  - 0 reflection cycles
```

### Analytical

```text
Trigger: structured input, moderate confidence, event frames present
Behavior:
  - Run Tokenize + ExtractFrame + ReasonFrame + IngestAtoms + GovernBeliefs + SeedAnchor
  - Standard reasoning depth
  - max_depth = 4
  - 1 reflection cycle
```

### Reflective

```text
Trigger: high uncertainty, contradictions, belief instability
Behavior:
  - Run full chain + extra reflection + contradiction review
  - Deep reasoning
  - max_depth = 5
  - 2 reflection cycles
  - Force GovernBeliefs re-evaluation
```

### Mode Selection

```rust
pub fn select_cognitive_mode(input: &str, graph: &Graph) -> CognitiveMode {
    let is_sentence = is_sentence_like(input);
    let has_contradictions = graph.has_contradictions();
    let avg_confidence = graph.average_confidence();

    if has_contradictions || avg_confidence < 0.4 {
        return CognitiveMode::Reflective;
    }

    if is_sentence || avg_confidence < 0.8 {
        return CognitiveMode::Analytical;
    }

    CognitiveMode::Reactive
}
```

---

## Compute Budget

```rust
pub struct ComputeBudget {
    pub max_reasoning_depth: usize,
    pub max_reflection_loops: usize,
    pub max_branching_factor: usize,
    pub max_hypothesis_count: usize,
    pub time_budget_ms: u64,
}

impl ComputeBudget {
    pub fn for_mode(mode: &CognitiveMode) -> Self {
        match mode {
            CognitiveMode::Reactive => ComputeBudget {
                max_reasoning_depth: 2,
                max_reflection_loops: 0,
                max_branching_factor: 3,
                max_hypothesis_count: 5,
                time_budget_ms: 1000,
            },
            CognitiveMode::Analytical => ComputeBudget {
                max_reasoning_depth: 4,
                max_reflection_loops: 1,
                max_branching_factor: 5,
                max_hypothesis_count: 10,
                time_budget_ms: 5000,
            },
            CognitiveMode::Reflective => ComputeBudget {
                max_reasoning_depth: 5,
                max_reflection_loops: 2,
                max_branching_factor: 7,
                max_hypothesis_count: 15,
                time_budget_ms: 10000,
            },
        }
    }
}
```

---

## Stop Conditions

```rust
pub struct StopCondition {
    pub confidence_threshold: f32,         // stop if confidence exceeds this
    pub max_loops_without_evidence: usize, // stop after N loops with no new evidence
    pub time_budget_ms: u64,              // stop when time exhausted
}

impl StopCondition {
    pub fn should_stop(&self, state: &ReasoningState) -> bool {
        // Confidence sufficient
        if state.confidence >= self.confidence_threshold {
            return true;
        }

        // Diminishing returns: no new evidence for N loops
        if state.loops_without_new_evidence >= self.max_loops_without_evidence {
            return true;
        }

        // Time budget exhausted
        if state.elapsed_ms >= self.time_budget_ms {
            return true;
        }

        // Goal satisfied
        if state.goal_met {
            return true;
        }

        false
    }
}
```

---

## Executive as Transform Chain Orchestrator

```rust
pub struct ExecutiveOrchestrator {
    mode: CognitiveMode,
    budget: ComputeBudget,
    stop: StopCondition,
}

impl ExecutiveOrchestrator {
    pub fn ingest(&self, text: &str, engine: &mut PipelineEngine) -> IngestResult {
        // 1. Always run Tokenize
        let mut atoms = engine.run::<Tokenize>(text);

        match self.mode {
            CognitiveMode::Reactive => {
                // Tokenize + Ingest only
                let delta = engine.run::<IngestAtoms>(&atoms);
                let governed = engine.run::<GovernBeliefs>(&delta);
                let anchored = engine.run::<SeedAnchor>(&governed);
                engine.apply(anchored)
            },

            CognitiveMode::Analytical => {
                // Tokenize + ExtractFrame + ReasonFrame + full chain
                if is_sentence_like(text) {
                    if let Some(frame) = engine.run::<ExtractFrame>(text) {
                        atoms.push(frame);
                        let hidden = engine.run::<ReasonFrame>(&atoms.last().unwrap());
                        atoms.extend(hidden);
                    }
                }
                let delta = engine.run::<IngestAtoms>(&atoms);
                let governed = engine.run::<GovernBeliefs>(&delta);
                let anchored = engine.run::<SeedAnchor>(&governed);
                engine.apply(anchored)
            },

            CognitiveMode::Reflective => {
                // Full chain + extra reflection + re-governance
                if is_sentence_like(text) {
                    if let Some(frame) = engine.run::<ExtractFrame>(text) {
                        atoms.push(frame);
                        let hidden = engine.run::<ReasonFrame>(&atoms.last().unwrap());
                        atoms.extend(hidden);
                    }
                }
                let delta = engine.run::<IngestAtoms>(&atoms);
                let governed = engine.run::<GovernBeliefs>(&delta);
                let anchored = engine.run::<SeedAnchor>(&governed);
                let result = engine.apply(anchored);

                // Extra: re-evaluate with stop conditions
                let mut state = ReasoningState::from(&result);
                let mut loops = 0;
                while !self.stop.should_stop(&state) && loops < self.budget.max_reflection_loops {
                    let reflection_delta = engine.run::<Reflect>(&state);
                    let re_governed = engine.run::<GovernBeliefs>(&reflection_delta);
                    let re_anchored = engine.run::<SeedAnchor>(&re_governed);
                    engine.apply(re_anchored);
                    state.update();
                    loops += 1;
                }

                result
            },
        }
    }
}
```

---

## Integration with v11.0 Pipeline

The executive orchestrator is feature-flagged:

```rust
impl Rsvs {
    pub fn ingest_text(&mut self, text: &str) -> IngestStats {
        if self.config.executive_enabled {
            let mode = select_cognitive_mode(text, &self.graph);
            let budget = ComputeBudget::for_mode(&mode);
            let orchestrator = ExecutiveOrchestrator {
                mode, budget,
                stop: StopCondition::default(),
            };
            orchestrator.ingest(text, &mut self.engine)
        } else {
            // v11.0 behavior: existing hardcoded pipeline
            self.v11_ingest_text(text)
        }
    }
}
```

When disabled: identical to v11.0. When enabled: uses executive control.

---

## Phase 2 — Deferred Features

| Feature | Why Deferred |
|---------|-------------|
| 9 more cognitive modes | Need mature specialist engines first |
| Attention Router | Need graph-scale benchmarking |
| Goal Arbitration | Need multi-goal scenarios |
| Task Decomposition | Need evidence that current system can't handle complexity |
| Failure Recovery | Need failure mode analysis |
| Executive Working Memory | Current session state may suffice |
| ExecutiveState (12 states) | 3 states sufficient; lifecycle is on Compositions |

---

## Module Structure

```text
layer1/crates/rsvs-core/src/
  executive/
    mod.rs              // ExecutiveOrchestrator + public API
    types.rs            // CognitiveMode, ComputeBudget, StopCondition, ReasoningState
    mode_selection.rs   // deterministic mode selection
    budget.rs           // budget per mode
    stop.rs             // stop condition evaluation
    tests.rs            // unit tests
```

6 files.

---

## Required Tests

### Test 1 — Simple Token → Reactive

Input: "raja" (not sentence-like)

Expected: Reactive mode, only Tokenize + IngestAtoms

### Test 2 — Sentence → Analytical

Input: "Raymond membuat aplikasi karena lambat"

Expected: Analytical mode, ExtractFrame + ReasonFrame + full chain

### Test 3 — Contradiction → Reflective

Input: any, but graph has existing contradictions

Expected: Reflective mode, extra reflection cycles

### Test 4 — Stop Condition: Confidence Sufficient

Reasoning reaches confidence 0.92 (threshold 0.90)

Expected: stop triggered

### Test 5 — Stop Condition: Budget Exhausted

Time budget exceeded

Expected: graceful halt

### Test 6 — Executive Disabled = v11.0

With executive disabled, pipeline behaves exactly like v11.0

---

## Acceptance Criteria

1. 3 cognitive modes: Reactive, Analytical, Reflective
2. Mode selection is deterministic and based on input + graph state
3. Compute budget enforced per mode (max depth 5, not 8)
4. Stop conditions prevent infinite loops
5. Executive is feature-flagged (can be disabled)
6. Pipeline identical to v11.0 when disabled
7. Reflective mode runs GovernBeliefs re-evaluation
8. All existing tests remain green

---

## Final Statement

MD-5 implements executive cognition as a Transform chain orchestrator. It selects mode,
enforces budget, and checks stop conditions — the minimum meta-control needed to prevent
runaway reasoning. Full executive features (attention, goals, decomposition) are deferred
until specialist engines are mature enough to benefit from them.
