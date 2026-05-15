# MD-2 — Pre-Ingest Meaning Reasoner (Elegant Architecture)

> **Prerequisite**: MD-3 defines SemanticAtom, AtomType, AtomVariant, SemanticRole,
> Composition, Transform. MD-1 defines ExtractFrame producing SemanticAtom(Event).
> This document defines the `ReasonFrame` Transform that consumes Event atoms
> and produces HiddenMeaning atoms.

---

## Mission

Implement the `ReasonFrame` Transform: discovers hidden meaning candidates from
event structure before RSVS graph ingestion.

Input: `SemanticAtom(Event, ...)` — from ExtractFrame
Output: `Vec<SemanticAtom(HiddenMeaning, ...)>` — derived candidates

The reasoner does NOT use LLMs. All reasoning is deterministic, rule-guided, and
graph-guided.

---

## Transform Definition

```rust
/// ReasonFrame Transform
///
/// Input:  SemanticAtom (atom_type = Event)
/// Output: Vec<SemanticAtom> (atom_type = HiddenMeaning)
///
/// Discovers hidden meaning candidates from event structure.
pub struct ReasonFrame {
    rules: Vec<Box<dyn ReasoningRule>>,
    config: ReasonerConfig,
}

impl Transform for ReasonFrame {
    type Input = SemanticAtom;
    type Output = Vec<SemanticAtom>;

    fn id(&self) -> &'static str { "ReasonFrame" }

    fn transform(&self, event: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        debug_assert!(event.atom_type == AtomType::Event);

        let reasoning_ctx = ReasoningContext {
            event,
            recent_events: &ctx.recent_events,
            graph: &ctx.graph,
        };

        let mut candidates = Vec::new();

        for rule in &self.rules {
            if rule.applies(&reasoning_ctx) {
                let results = rule.generate(&reasoning_ctx);
                for result in results {
                    candidates.push(result.into_atom(ctx));
                }
            }
        }

        candidates
    }
}
```

---

## ReasoningRule Trait

```rust
pub trait ReasoningRule: Send + Sync {
    fn id(&self) -> &'static str;
    fn applies(&self, ctx: &ReasoningContext) -> bool;
    fn generate(&self, ctx: &ReasoningContext) -> Vec<ReasoningResult>;
}

pub struct ReasoningContext<'a> {
    pub event: &'a SemanticAtom,
    pub recent_events: &'a Vec<SemanticAtom>,
    pub graph: &'a Graph,
}

/// Internal result — converted to SemanticAtom by into_atom()
pub struct ReasoningResult {
    pub label: String,
    pub meaning_type: HiddenMeaningType,
    pub roles: HashMap<SemanticRole, String>,
    pub confidence: f32,
    pub rule_id: String,
    pub evidence_roles: Vec<SemanticRole>,
}

impl ReasoningResult {
    pub fn into_atom(self, ctx: &mut PipelineContext) -> SemanticAtom {
        SemanticAtom {
            id: format!("hm_{}", ctx.next_atom_id()),
            label: self.label,
            atom_type: AtomType::HiddenMeaning,
            roles: self.roles,
            polarity: None,
            voice: None,
            variant: Some(AtomVariant::MeaningVariant(self.meaning_type)),
            confidence: self.confidence,
            source: EdgeSource::HiddenMeaningRule,
        }
    }
}
```

---

## Core Rules — Phase 1: 3 Rules

### Rule 1 — Cause + Action + Patient → Problem-Solution Pattern

```rust
pub struct ProblemSolutionRule;

impl ReasoningRule for ProblemSolutionRule {
    fn id(&self) -> &'static str { "CAUSE_ACTION_PATIENT_TO_PROBLEM_SOLUTION" }

    fn applies(&self, ctx: &ReasoningContext) -> bool {
        ctx.event.roles.contains_key(&SemanticRole::Cause)
            && ctx.event.roles.contains_key(&SemanticRole::Arg1Patient)
            && is_action_predicate(&ctx.event.label, ctx.graph)
    }

    fn generate(&self, ctx: &ReasoningContext) -> Vec<ReasoningResult> {
        let cause = ctx.event.roles.get(&SemanticRole::Cause).unwrap();
        let patient = ctx.event.roles.get(&SemanticRole::Arg1Patient).unwrap();
        let agent = ctx.event.roles.get(&SemanticRole::Arg0Agent);

        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Problem, cause.clone());
        roles.insert(SemanticRole::Solution, patient.clone());
        if let Some(a) = agent {
            roles.insert(SemanticRole::Arg0Agent, a.clone());
        }
        roles.insert(SemanticRole::SourceEvent, ctx.event.id.clone());

        let confidence = self.compute_confidence(ctx);

        vec![ReasoningResult {
            label: "problem_solution".into(),
            meaning_type: HiddenMeaningType::ProblemSolutionPattern,
            roles,
            confidence,
            rule_id: self.id().to_string(),
            evidence_roles: vec![SemanticRole::Cause, SemanticRole::Arg1Patient],
        }]
    }
}
```

### Rule 2 — Purpose Marker → Goal Inference

```rust
pub struct GoalInferenceRule;

impl ReasoningRule for GoalInferenceRule {
    fn id(&self) -> &'static str { "PURPOSE_TO_GOAL_INFERENCE" }

    fn applies(&self, ctx: &ReasoningContext) -> bool {
        ctx.event.roles.contains_key(&SemanticRole::Purpose)
    }

    fn generate(&self, ctx: &ReasoningContext) -> Vec<ReasoningResult> {
        let purpose = ctx.event.roles.get(&SemanticRole::Purpose).unwrap();
        let agent = ctx.event.roles.get(&SemanticRole::Arg0Agent);

        let mut roles = HashMap::new();
        roles.insert(SemanticRole::ImpliedGoal, purpose.clone());
        roles.insert(SemanticRole::SourceEvent, ctx.event.id.clone());
        if let Some(a) = agent {
            roles.insert(SemanticRole::Arg0Agent, a.clone());
        }

        vec![ReasoningResult {
            label: "goal_inference".into(),
            meaning_type: HiddenMeaningType::GoalInference,
            roles,
            confidence: 0.60,
            rule_id: self.id().to_string(),
            evidence_roles: vec![SemanticRole::Purpose],
        }]
    }
}
```

### Rule 3 — Same Event + Opposite Polarity → Polarity Conflict

```rust
pub struct PolarityConflictRule;

impl ReasoningRule for PolarityConflictRule {
    fn id(&self) -> &'static str { "POLARITY_CONFLICT" }

    fn applies(&self, ctx: &ReasoningContext) -> bool {
        // Check if any recent event contradicts current event
        ctx.event.polarity == Some(Polarity::Negative)
            && ctx.recent_events.iter().any(|e| {
                e.atom_type == AtomType::Event
                && e.label == ctx.event.label
                && e.roles.get(&SemanticRole::Arg0Agent) == ctx.event.roles.get(&SemanticRole::Arg0Agent)
                && e.roles.get(&SemanticRole::Arg1Patient) == ctx.event.roles.get(&SemanticRole::Arg1Patient)
                && e.polarity == Some(Polarity::Positive)
            })
    }

    fn generate(&self, ctx: &ReasoningContext) -> Vec<ReasoningResult> {
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::SourceEvent, ctx.event.id.clone());
        // Find the contradicting event
        if let Some(other) = ctx.recent_events.iter().find(|e| {
            e.atom_type == AtomType::Event
            && e.label == ctx.event.label
            && e.polarity != ctx.event.polarity
        }) {
            roles.insert(SemanticRole::EquivalentOf, other.id.clone());
        }

        vec![ReasoningResult {
            label: "polarity_conflict".into(),
            meaning_type: HiddenMeaningType::PolarityConflict,
            roles,
            confidence: 0.90,
            rule_id: self.id().to_string(),
            evidence_roles: vec![SemanticRole::Arg0Agent, SemanticRole::Arg1Patient],
        }]
    }
}
```

---

## Phase 2 Rules (Deferred)

| Rule | Trigger | Output |
|------|---------|--------|
| AgentResponsibility | voice=Active + agent exists | AgentResponsibility |
| InefficiencySignal | cause has negative sentiment (graph-guided) | InefficiencySignal |
| ToolUsePattern | problem_solution + patient is tool-like | ToolUsePattern |
| PassiveNormalization | voice=Passive | normalized active equivalent |
| PurposeConflict | same event, different purpose | PurposeConflict |
| RoleAnomaly | agent-patient reversal | RoleAnomaly |

Negative sentiment and tool-likeness detection: **graph-guided, not hardcoded word lists**.

```rust
fn has_negative_sentiment(text: &str, graph: &Graph) -> bool {
    let tokens = tokenize(text);
    tokens.iter().any(|t| {
        // Check if graph has negative-valence sense for this token
        graph_has_negative_valence(t) || seed_negative_list.contains(t)
    })
}
```

---

## Graph-Guided Confidence Adjustment

```rust
fn adjust_confidence_by_graph(
    candidate: &ReasoningResult,
    graph: &Graph,
) -> f32 {
    let mut adjusted = candidate.confidence;

    for (role, label) in &candidate.roles {
        if let Some(node_id) = graph.find_node_by_label(label) {
            // If graph confirms the node plays the expected role
            if graph_confirms_role(graph, node_id, role) {
                adjusted += 0.05;
            }
            // If graph contradicts the role
            if graph_contradicts_role(graph, node_id, role) {
                adjusted -= 0.10;
            }
        }
    }

    // Cap at source event confidence
    adjusted.clamp(0.0, 1.0)
}
```

---

## Graph Integration

When `IngestAtoms` receives a `SemanticAtom(HiddenMeaning, ...)`, it creates:

1. A `Composition { composition_type: HiddenMeaning, ... }`
2. `SemanticEdge` per role with `source: HiddenMeaningRule`
3. `lifecycle: Quarantine` (hypothesis quarantine — not yet promoted)
4. `epistemic: Inferred` (derived by rule, not directly observed)

```rust
// Inside IngestAtoms
fn ingest_hidden_meaning_atom(&mut self, atom: &SemanticAtom, graph: &mut Graph) -> GraphDelta {
    // Same pattern as event atoms, but:
    // - composition_type = HiddenMeaning
    // - lifecycle = Quarantine (hypothesis quarantine)
    // - epistemic = Inferred
    // - source = HiddenMeaningRule

    let comp_id = CompositionId::new();
    let mut members = Vec::new();

    // Label node (e.g., "problem_solution")
    let label_id = graph.ensure_node(&atom.label);
    members.push(CompositionMember {
        node_id: label_id,
        role: SemanticRole::PatternType,
        confidence: atom.confidence,
    });

    // Role members
    for (role, label) in &atom.roles {
        let node_id = graph.ensure_node(label);
        members.push(CompositionMember {
            node_id,
            role: role.clone(),
            confidence: atom.confidence,
        });
    }

    // Composition with quarantine + inferred
    delta.add_composition(Composition {
        id: comp_id,
        composition_type: CompositionType::HiddenMeaning,
        members,
        lifecycle: LifecycleState::Quarantine,  // hypothesis quarantine
        epistemic: EpistemicState::Inferred,     // derived by rule
        confidence: atom.confidence,
        provenance: ProvenanceChain {
            origin: EdgeSource::HiddenMeaningRule,
            origin_id: atom.id.clone(),
            parent_composition_id: None,  // TODO: link to source event composition
            timestamp: now_iso8601(),
        },
        seed_scores: HashMap::new(),
        created_at: now_iso8601(),
        updated_at: now_iso8601(),
    });

    delta
}
```

---

## Cross-Type Reasoning: Event ↔ HiddenMeaning

Because both Event and HiddenMeaning are now Compositions with SemanticRole members,
cross-type reasoning becomes natural:

```text
Event(Cause="lambat") ←→ HiddenMeaning(Problem="lambat")
  → The cause in the event IS the problem in the hidden meaning
  → Detectable: same node appears with different roles in different compositions
  → structural_similarity() can compare across types
  → convergence can unify overlapping structures
```

This was impossible when EventFrame and HiddenMeaningCandidate were separate types.

---

## Module Structure

```text
layer0/
  pre_ingest_reasoning/
    mod.rs              // ReasonFrame Transform + public API
    types.rs            // ReasoningRule trait, ReasoningResult, ReasoningContext
    rules.rs            // All reasoning rule implementations
    scorer.rs           // confidence scoring + graph-guided adjustment
    tests.rs            // unit tests
```

5 files.

---

## Required Tests

### Test 1 — Problem Solution Pattern

Input: `SemanticAtom(Event, "membuat", {Arg0Agent: "Raymond", Arg1Patient: "aplikasi", Cause: "lambat"})`

Expected: `SemanticAtom(HiddenMeaning, "problem_solution", {Problem: "lambat", Solution: "aplikasi", Agent: "Raymond"})`

### Test 2 — Goal Inference

Input: `SemanticAtom(Event, "membuat", {Arg0Agent: "Raymond", Purpose: "kantor"})`

Expected: `SemanticAtom(HiddenMeaning, "goal_inference", {ImpliedGoal: "kantor", Agent: "Raymond"})`

### Test 3 — Polarity Conflict

Input: Event A (Positive) + Event B (same predicate/agent/patient, Negative)

Expected: `SemanticAtom(HiddenMeaning, "polarity_conflict", ...)`

### Test 4 — No Hidden Meaning from Incomplete Event

Input: `SemanticAtom(Event, "membuat", {})` (no agent, patient, cause, purpose)

Expected: No output from any rule.

### Test 5 — HiddenMeaning Composition Has Quarantine + Inferred

Verify ingested hidden meaning has `lifecycle=Quarantine, epistemic=Inferred`.

---

## Acceptance Criteria

1. `ReasonFrame` Transform implemented and registered
2. Produces `SemanticAtom(HiddenMeaning, ...)` for structured events
3. 3 core rules: ProblemSolution, GoalInference, PolarityConflict
4. Each candidate includes: meaning_type (via AtomVariant), evidence roles, rule_id
5. Confidence is deterministic and graph-adjustable
6. HiddenMeaning atoms ingested as `Composition(HiddenMeaning)` with quarantine
7. Negative sentiment detection is graph-guided (Phase 2)
8. All existing tests remain green

---

## Final Statement

MD-2 implements the second Transform in the elegant architecture. It consumes Event atoms
and produces HiddenMeaning atoms through deterministic rule-guided reasoning. Because both
are SemanticAtom and both become Composition in the graph, cross-type reasoning and
convergence are naturally enabled — no separate type hierarchies needed.
