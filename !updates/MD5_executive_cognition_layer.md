# MD-5 — Executive Cognition (Elegant Architecture)

> **Prerequisite**: MD-3 defines Transform, SemanticAtom, Composition, LifecycleState,
> EpistemicState. MD-3 now also defines EnrichmentRequest, RecallAction, EnrichComposition Transform,
> and ReExtractFrame Transform for the feedback loop. This document integrates the
> feedback loop into the executive orchestrator.
> MD-4 defines GovernBeliefs + SeedAnchor Transforms.
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
  - Run DetectGaps + SelectAcquisition (if gaps found)
  - For PassiveRecall decisions: run EnrichComposition + re-govern
  - For AskUser decisions: generate inquiry questions
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
  - Force gap detection + enrichment loop
  - Re-run ExtractFrame with graph context for weak frames
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
    pub max_enrichment_rounds: usize,  // NEW: max gap→enrich cycles per ingest
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
                max_enrichment_rounds: 0,
            },
            CognitiveMode::Analytical => ComputeBudget {
                max_reasoning_depth: 4,
                max_reflection_loops: 1,
                max_branching_factor: 5,
                max_hypothesis_count: 10,
                time_budget_ms: 5000,
                max_enrichment_rounds: 1,
            },
            CognitiveMode::Reflective => ComputeBudget {
                max_reasoning_depth: 5,
                max_reflection_loops: 2,
                max_branching_factor: 7,
                max_hypothesis_count: 15,
                time_budget_ms: 10000,
                max_enrichment_rounds: 2,
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
        // 1. Goal satisfied
        if state.goal_met {
            return true;
        }

        // 2. Confidence sufficient for goal type
        let confidence_threshold = match state.goal {
            ReasoningGoal::UnderstandInput => 0.8,
            ReasoningGoal::ResolveContradiction { .. } => 0.7,
            ReasoningGoal::FillGap { .. } => 0.6,
            ReasoningGoal::AnswerQuestion { .. } => 0.85,
        };
        if state.confidence >= confidence_threshold {
            return true;
        }

        // 3. Diminishing returns: no new evidence for N loops
        if state.loops_without_new_evidence >= self.max_loops_without_evidence {
            return true;
        }

        // 4. Time budget exhausted
        if state.elapsed_ms >= self.time_budget_ms {
            return true;
        }

        false
    }
}
```

---

## Reasoning State

Full state of a reasoning session. Used by `StopCondition` to decide when to halt,
and by `Reflect` to review. Defined in `types.rs`.

```rust
/// Full state of a reasoning session.
/// Used by StopCondition to decide when to halt, and by Reflect to review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningState {
    /// Current confidence of the reasoning result
    pub confidence: f32,

    /// Elapsed time since reasoning started
    pub elapsed_ms: u64,

    /// Number of reflection loops completed
    pub loops_completed: usize,

    /// Number of loops since the last new piece of evidence was found
    pub loops_without_new_evidence: usize,

    /// Whether the reasoning goal has been met
    pub goal_met: bool,

    /// The reasoning goal (what are we trying to determine?)
    pub goal: ReasoningGoal,

    /// Compositions modified during this reasoning session
    pub modified_compositions: Vec<CompositionId>,

    /// Evidence accumulated during this reasoning session
    pub evidence_count: usize,

    /// Evidence count at the start of the current loop
    pub evidence_at_loop_start: usize,
}

/// What the reasoning session is trying to accomplish.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningGoal {
    /// Understand the meaning of input text
    UnderstandInput,
    /// Resolve a specific contradiction
    ResolveContradiction { composition_id: CompositionId },
    /// Fill a specific knowledge gap
    FillGap { gap_id: String },
    /// Answer a user question
    AnswerQuestion { question: String },
}

impl ReasoningState {
    /// Create initial state for a reasoning session
    pub fn new(goal: ReasoningGoal) -> Self {
        Self {
            confidence: 0.0,
            elapsed_ms: 0,
            loops_completed: 0,
            loops_without_new_evidence: 0,
            goal_met: false,
            goal,
            modified_compositions: Vec::new(),
            evidence_count: 0,
            evidence_at_loop_start: 0,
        }
    }

    /// Update state after a reflection loop completes.
    /// This is the missing method that was called but never defined.
    pub fn update(&mut self, loop_result: &ReflectionLoopResult) {
        self.loops_completed += 1;
        self.confidence = loop_result.current_confidence;
        self.elapsed_ms += loop_result.elapsed_ms;

        // Track whether new evidence was found in this loop
        let new_evidence = loop_result.evidence_count - self.evidence_at_loop_start;
        if new_evidence > 0 {
            self.loops_without_new_evidence = 0;
            self.evidence_count = loop_result.evidence_count;
            self.evidence_at_loop_start = loop_result.evidence_count;
        } else {
            self.loops_without_new_evidence += 1;
        }

        // Track modified compositions
        self.modified_compositions.extend(loop_result.modified_compositions.iter().cloned());

        // Check if goal is met
        self.goal_met = self.check_goal_met(loop_result);
    }

    /// Determine if the reasoning goal has been satisfied.
    fn check_goal_met(&self, result: &ReflectionLoopResult) -> bool {
        match &self.goal {
            ReasoningGoal::UnderstandInput => {
                // Goal met when confidence is high and no gaps remain
                self.confidence >= 0.8 && !result.has_gaps
            },
            ReasoningGoal::ResolveContradiction { composition_id } => {
                // Goal met when the contradiction is resolved
                result.resolved_contradictions.contains(composition_id)
            },
            ReasoningGoal::FillGap { gap_id } => {
                // Goal met when the gap is filled
                result.filled_gaps.contains(gap_id)
            },
            ReasoningGoal::AnswerQuestion { .. } => {
                // Goal met when confidence exceeds threshold
                self.confidence >= 0.85
            },
        }
    }
}

/// Result of a single reflection loop (produced by the pipeline after
/// running GovernBeliefs + SeedAnchor on the reflection delta).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionLoopResult {
    pub current_confidence: f32,
    pub elapsed_ms: u64,
    pub evidence_count: usize,
    pub modified_compositions: Vec<CompositionId>,
    pub has_gaps: bool,
    pub resolved_contradictions: Vec<CompositionId>,
    pub filled_gaps: Vec<String>,
}
```

---

## Reflect Transform — Self-Review of Graph State

```rust
/// Reflect Transform
///
/// Reviews existing compositions in the graph for quality, consistency,
/// and improvement opportunities. Produces a GraphDelta with proposed
/// modifications (no destructive changes — those require explicit approval).
///
/// Input:  ReasoningState (current reasoning context + graph snapshot)
/// Output: GraphDelta (proposed modifications for re-governance)
pub struct Reflect {
    config: ReflectConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectConfig {
    /// Maximum compositions to review per reflection cycle
    pub max_review_count: usize,
    /// Minimum confidence to consider a composition "worth reviewing"
    pub review_confidence_threshold: f32,
    /// Whether to check for contradiction resolutions
    pub check_contradictions: bool,
}

impl Default for ReflectConfig {
    fn default() -> Self {
        Self {
            max_review_count: 20,
            review_confidence_threshold: 0.3,
            check_contradictions: true,
        }
    }
}

pub struct ReflectionFinding {
    pub composition_id: CompositionId,
    pub finding_type: ReflectionFindingType,
    pub description: String,
    pub proposed_action: ReflectionAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReflectionFindingType {
    /// Composition has been Inferred for a long time without reaching Grounded
    StagnantInferred,
    /// Composition is Candidate but has enough evidence for Stable
    PromotionCandidate,
    /// Composition has unresolved contradiction that may be resolvable now
    ContradictionResolvable,
    /// Composition has very low confidence and no recent supporting evidence
    DecayedConfidence,
    /// Composition overlaps significantly with another — possible merge
    OverlapDetected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReflectionAction {
    ProposePromotion { target_lifecycle: LifecycleState, target_epistemic: EpistemicState },
    ProposeContradictionResolution { resolution_type: ResolutionType },
    ProposeDeprecation { reason: String },
    ProposeMerge { other_composition_id: CompositionId },
    NoAction,
}

impl Transform for Reflect {
    type Input = ReasoningState;
    type Output = GraphDelta;

    fn id(&self) -> &'static str { "Reflect" }

    fn transform(&self, state: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        let mut delta = GraphDelta::new();
        let mut reviewed = 0;

        // 1. Review compositions in confidence order (low confidence first — most need attention)
        let mut compositions: Vec<_> = ctx.graph.compositions()
            .filter(|c| c.confidence < self.config.review_confidence_threshold
                || c.epistemic == EpistemicState::Contradicted)
            .collect();
        compositions.sort_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap());

        for comp in compositions.iter().take(self.config.max_review_count) {
            // Stagnant inferred: has been Inferred for many batches without Grounding
            if comp.epistemic == EpistemicState::Inferred && comp.age_in_batches() > 10 {
                delta.add_finding(ReflectionFinding {
                    composition_id: comp.id.clone(),
                    finding_type: ReflectionFindingType::StagnantInferred,
                    description: format!("Composition {} has been Inferred for {} batches",
                        comp.id, comp.age_in_batches()),
                    proposed_action: ReflectionAction::NoAction, // just flag it
                });
            }

            // Contradiction resolution: check if previously unresolvable contradictions
            // now have enough context to resolve
            if comp.epistemic == EpistemicState::Contradicted && self.config.check_contradictions {
                if let Some(resolution) = self.check_contradiction_resolution(comp, &ctx.graph) {
                    delta.add_finding(ReflectionFinding {
                        composition_id: comp.id.clone(),
                        finding_type: ReflectionFindingType::ContradictionResolvable,
                        description: format!("Contradiction resolvable: {:?}", resolution.resolution_type),
                        proposed_action: ReflectionAction::ProposeContradictionResolution {
                            resolution_type: resolution.resolution_type,
                        },
                    });
                }
            }

            // Decay: confidence has dropped very low with no recent support
            if comp.confidence < 0.15 && comp.age_in_batches() > 20 {
                delta.add_finding(ReflectionFinding {
                    composition_id: comp.id.clone(),
                    finding_type: ReflectionFindingType::DecayedConfidence,
                    description: format!("Confidence {:.2} after {} batches",
                        comp.confidence, comp.age_in_batches()),
                    proposed_action: ReflectionAction::ProposeDeprecation {
                        reason: "Sustained low confidence without recovery".into(),
                    },
                });
            }

            reviewed += 1;
        }

        // 2. Apply safe actions (non-destructive only)
        // Promotion proposals go into the delta for GovernBeliefs to evaluate
        // Deprecation proposals require explicit approval (not auto-applied)

        delta
    }
}
```

> **Note**: Reflect is a READ-ONLY review of graph state. It proposes changes but does not
> apply destructive ones. GovernBeliefs evaluates the proposals and decides which
> to apply. This separation ensures that reflection cannot damage the graph —
> it can only suggest improvements.

---

## Enrichment Loop — Closing the Feedback Cycle

The executive orchestrator now includes an enrichment loop that runs after initial
ingest. The loop: detect gaps → select acquisition → for PassiveRecall decisions,
enrich composition → re-govern → repeat until no more gaps or budget exhausted.

This closes the feedback cycle identified in MD-3: after atoms are ingested and
beliefs governed, the system can now look back at what it just learned, detect
what is missing, and proactively fill those gaps — either by recalling existing
knowledge from the graph (PassiveRecall via EnrichComposition) or by re-extracting
frames with enriched graph context (ReExtractFrame).

### Why This Is Bounded

The enrichment loop cannot run forever because:

- **max_enrichment_rounds** prevents infinite loops (hard cap per mode)
- Each round can only **add** members to compositions, not remove them (monotonic improvement)
- PassiveRecall becomes less productive over time — the best candidates are found first,
  subsequent rounds are increasingly unlikely to find high-quality matches
- **StopCondition** also applies: if confidence exceeds threshold, enrichment stops early

### Enrichment Loop Diagram

```text
Initial Ingest
  ↓
DetectGaps → gaps found?
  ├── No: done
  └── Yes: SelectAcquisition
        ↓
      For each decision:
        ├── PassiveRecall + EnrichComposition → re-govern → confidence update
        ├── AskUser → generate question → wait for answer → merge
        └── Deferred → note gap, no action
        ↓
      More enrichment rounds available?
        ├── No: done
        └── Yes: DetectGaps again (with updated graph)
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
                let result = engine.apply(anchored);

                // NEW: Gap detection + enrichment loop
                let mut enrichment_round = 0;
                while enrichment_round < self.budget.max_enrichment_rounds {
                    let snapshot = engine.snapshot();
                    let gaps = engine.run::<DetectGaps>(&snapshot);
                    if gaps.is_empty() { break; }

                    let decisions = engine.run::<SelectAcquisition>(&gaps);
                    let mut enriched_any = false;

                    for decision in &decisions {
                        match &decision.action {
                            Some(RecallAction::EnrichComposition { target_composition_id, role_to_fill, candidate_node_id }) => {
                                let request = EnrichmentRequest {
                                    target_composition_id: target_composition_id.clone(),
                                    role_to_fill: role_to_fill.clone(),
                                    candidate_node_id: *candidate_node_id,
                                    candidate_label: engine.graph().node_label(*candidate_node_id).unwrap_or_default(),
                                    source: EnrichmentSource::PassiveRecall,
                                    confidence: 0.7,
                                };
                                let delta = engine.run::<EnrichComposition>(&request);
                                let governed = engine.run::<GovernBeliefs>(&delta);
                                let anchored = engine.run::<SeedAnchor>(&governed);
                                engine.apply(anchored);
                                enriched_any = true;
                            },
                            Some(RecallAction::ReExtractFrame { target_composition_id, enriched_context }) => {
                                let request = ReExtractionRequest {
                                    original_text: /* from composition */,
                                    original_atom_id: /* from composition */,
                                    target_composition_id: target_composition_id.clone(),
                                    graph_context: enriched_context.clone(),
                                };
                                if let Some(improved) = engine.run::<ReExtractFrame>(&request) {
                                    // Merge improved frame into existing composition
                                    let delta = engine.run::<EnrichComposition>(&EnrichmentRequest::from_improved_atom(improved));
                                    let governed = engine.run::<GovernBeliefs>(&delta);
                                    let anchored = engine.run::<SeedAnchor>(&governed);
                                    engine.apply(anchored);
                                    enriched_any = true;
                                }
                            },
                            _ => {} // AskUser, Deferred, NoAction: handled externally
                        }
                    }

                    if !enriched_any { break; }
                    enrichment_round += 1;
                }

                result
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
                let mut state = ReasoningState::new(ReasoningGoal::UnderstandInput);
                let mut loops = 0;
                while !self.stop.should_stop(&state) && loops < self.budget.max_reflection_loops {
                    let reflection_delta = engine.run::<Reflect>(&state);
                    let re_governed = engine.run::<GovernBeliefs>(&reflection_delta);
                    let re_anchored = engine.run::<SeedAnchor>(&re_governed);
                    let loop_result = engine.apply_with_result(re_anchored);
                    state.update(&loop_result);
                    loops += 1;
                }

                // Force gap detection + enrichment loop (2 rounds)
                let mut enrichment_round = 0;
                while enrichment_round < self.budget.max_enrichment_rounds {
                    let snapshot = engine.snapshot();
                    let gaps = engine.run::<DetectGaps>(&snapshot);
                    if gaps.is_empty() { break; }

                    let decisions = engine.run::<SelectAcquisition>(&gaps);
                    let mut enriched_any = false;

                    for decision in &decisions {
                        match &decision.action {
                            Some(RecallAction::EnrichComposition { target_composition_id, role_to_fill, candidate_node_id }) => {
                                let request = EnrichmentRequest {
                                    target_composition_id: target_composition_id.clone(),
                                    role_to_fill: role_to_fill.clone(),
                                    candidate_node_id: *candidate_node_id,
                                    candidate_label: engine.graph().node_label(*candidate_node_id).unwrap_or_default(),
                                    source: EnrichmentSource::PassiveRecall,
                                    confidence: 0.7,
                                };
                                let delta = engine.run::<EnrichComposition>(&request);
                                let governed = engine.run::<GovernBeliefs>(&delta);
                                let anchored = engine.run::<SeedAnchor>(&governed);
                                engine.apply(anchored);
                                enriched_any = true;
                            },
                            Some(RecallAction::ReExtractFrame { target_composition_id, enriched_context }) => {
                                let request = ReExtractionRequest {
                                    original_text: /* from composition */,
                                    original_atom_id: /* from composition */,
                                    target_composition_id: target_composition_id.clone(),
                                    graph_context: enriched_context.clone(),
                                };
                                if let Some(improved) = engine.run::<ReExtractFrame>(&request) {
                                    let delta = engine.run::<EnrichComposition>(&EnrichmentRequest::from_improved_atom(improved));
                                    let governed = engine.run::<GovernBeliefs>(&delta);
                                    let anchored = engine.run::<SeedAnchor>(&governed);
                                    engine.apply(anchored);
                                    enriched_any = true;
                                }
                            },
                            _ => {} // AskUser, Deferred, NoAction: handled externally
                        }
                    }

                    if !enriched_any { break; }
                    enrichment_round += 1;
                }

                // Re-run ExtractFrame with graph context for weak frames
                let weak_frames = engine.find_weak_frames();
                for weak_frame in &weak_frames {
                    let request = ReExtractionRequest {
                        original_text: weak_frame.source_text().to_string(),
                        original_atom_id: weak_frame.atom_id(),
                        target_composition_id: weak_frame.composition_id().clone(),
                        graph_context: engine.snapshot().context_for(weak_frame),
                    };
                    if let Some(improved) = engine.run::<ReExtractFrame>(&request) {
                        let delta = engine.run::<EnrichComposition>(&EnrichmentRequest::from_improved_atom(improved));
                        let governed = engine.run::<GovernBeliefs>(&delta);
                        let anchored = engine.run::<SeedAnchor>(&governed);
                        engine.apply(anchored);
                    }
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
    types.rs            // CognitiveMode, ComputeBudget, StopCondition, ReasoningState, ReasoningGoal, ReflectionLoopResult
    mode_selection.rs   // deterministic mode selection
    budget.rs           // budget per mode
    stop.rs             // stop condition evaluation
    reflect.rs          // Reflect Transform (NEW)
    tests.rs            // unit tests
```

7 files.

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

### Test 7 — Analytical Mode Includes Gap Detection + Enrichment

Input: "membuat aplikasi karena lambat" (missing agent)
Mode: Analytical
Expected: ExtractFrame → Ingest → DetectGaps → SelectAcquisition → EnrichComposition → re-govern

### Test 8 — Reactive Mode Skips Gap Detection

Input: "raja"
Mode: Reactive
Expected: Tokenize → Ingest → Govern → Seed. No DetectGaps, no EnrichComposition.

### Test 9 — Reflective Mode Allows 2 Enrichment Rounds

Input: complex sentence with multiple gaps
Mode: Reflective
Expected: Up to 2 enrichment rounds, each detecting remaining gaps

### Test 10 — Enrichment Loop Stops When No More Gaps

After enrichment fills missing role, DetectGaps finds no new gaps
Expected: enrichment loop terminates before hitting max_enrichment_rounds

### Test 11 — Reflect Detects Stagnant Inferred Composition

Composition with epistemic=Inferred, age=15 batches
Expected: ReflectionFinding::StagnantInferred

### Test 12 — Reflect Proposes Contradiction Resolution

Composition with epistemic=Contradicted, opposing composition has voice confusion
Expected: ReflectionFinding::ContradictionResolvable

### Test 13 — Reflect Does Not Auto-Destruct

Composition with confidence < 0.1
Expected: ProposeDeprecation but NOT auto-applied. GovernBeliefs must approve.

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
9. Analytical mode includes gap detection + enrichment loop (1 round)
10. Reflective mode includes gap detection + enrichment loop (2 rounds)
11. Reactive mode does not run gap detection
12. Enrichment loop is bounded by max_enrichment_rounds
13. Enrichment loop stops early when no gaps remain
14. Reflect Transform is defined with config, finding types, and actions
15. Reflect is read-only — proposes changes but never applies destructive ones
16. ReasoningState.update() accepts ReflectionLoopResult and tracks evidence/gaps
17. StopCondition.should_stop() uses goal-based confidence thresholds
18. ReasoningGoal defines 4 goal types with distinct satisfaction criteria

---

## Final Statement

MD-5 implements executive cognition as a Transform chain orchestrator. It selects mode,
enforces budget, and checks stop conditions — the minimum meta-control needed to prevent
runaway reasoning. The enrichment loop closes the feedback cycle: after initial ingest,
the system detects gaps in its own knowledge and proactively fills them, bounded by
max_enrichment_rounds and diminishing returns. The Reflect Transform provides read-only
self-review of graph state, proposing improvements without destructive side effects.
ReasoningState tracks evidence accumulation and goal satisfaction, giving StopCondition
the data it needs to halt reasoning at the right time. Full executive features (attention,
goals, decomposition) are deferred until specialist engines are mature enough to benefit
from them.
