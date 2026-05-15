# MD-3 — AAM Elegant Architecture: Unified Abstractions (v12.0 Target)

> **This is the FOUNDATION document.** All other MDs (MD1, MD2, MD4-MD6) reference the types
> and patterns defined here. This document describes the target architecture for AAM v12.0,
> built on 6 unified abstractions that replace the patchwork of overlapping types in v11.0.

---

## Why This Refactor Exists

v11.0 has accumulated overlapping type systems:

```text
4 overlapping lifecycle enums:   NodeStatus, CandidateStatus, BeliefState, GroundingVerdict
4 overlapping edge systems:      RelationType, EdgeSource, SemanticRole, ProvenanceSource
3 structured grouping types:     EventFrame, HiddenMeaningCandidate, Composition
2 ingest paths:                  token path + frame path (dual-track forever)
```

This creates confusion, maintenance burden, and prevents cross-type reasoning.
A Composition and an EventFrame are the same concept — structured groups of nodes with typed
relationships — yet they have separate type hierarchies and cannot be compared or merged.

The elegant architecture unifies these into **6 core abstractions**.

---

## The 6 Unified Abstractions

| # | Abstraction | Replaces | Purpose |
|---|------------|----------|---------|
| 1 | **SemanticAtom** | Token, EventFrame, HiddenMeaningCandidate | Universal ingest primitive |
| 2 | **Composition** | EventFrame, HiddenMeaningCandidate, Pattern, Hypothesis | Universal structured grouping |
| 3 | **LifecycleState + EpistemicState** | NodeStatus, CandidateStatus, BeliefState, GroundingVerdict | Two orthogonal status axes |
| 4 | **SemanticEdge** | Separate RelationType, EdgeSource, SemanticRole, ProvenanceSource | Single typed triple |
| 5 | **Transform (DAG)** | Hardcoded pipeline stages | Declarative transform graph |
| 6 | **Seed Anchoring** | Source trust weight system | Seed-driven epistemic confidence |

---

## Abstraction 1: SemanticAtom — Unified Ingest Primitive

Every piece of knowledge entering RSVS passes through one type: `SemanticAtom`.

A token, an event frame, a hidden meaning candidate — these are all atoms with varying
richness. A token is a sparse atom. An event frame is a rich atom. A hidden meaning is
a derived atom.

```rust
/// Universal ingest primitive.
/// Every piece of knowledge entering RSVS is a SemanticAtom.
/// Richness varies by atom_type: Token (sparse) vs Event (rich).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAtom {
    pub id: String,
    pub label: String,
    pub atom_type: AtomType,

    // Structural metadata (populated for Event, HiddenMeaning; empty for Token)
    pub roles: HashMap<SemanticRole, String>,   // role → target label
    pub polarity: Option<Polarity>,
    pub voice: Option<Voice>,

    // Type-specific classification
    pub variant: Option<AtomVariant>,

    // Provenance
    pub confidence: f32,
    pub source: EdgeSource,                      // unified with edge provenance
}

#[derive(Debug, Clone, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub enum AtomType {
    Token,          // simple token extraction (sparse: roles = {})
    AmbiguousToken, // token that requires disambiguation (pronouns, deictics)
    Event,          // semantic frame extraction (rich: roles = {Arg0, Arg1, Cause, ...})
    HiddenMeaning,  // pre-ingest reasoning output (rich: roles = {Problem, Solution, ...})
    Pattern,        // pattern mining output (structured: roles = {Cause, Action, ...})
    Hypothesis,     // abductive/predictive hypothesis
    Acquisition,    // externally acquired knowledge
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AtomVariant {
    /// For AtomType::HiddenMeaning — which kind of hidden meaning
    MeaningVariant(HiddenMeaningType),
    /// For AtomType::Event — how was this frame extracted
    FrameVariant(FrameSource),
    /// For AtomType::Pattern — what pattern category
    PatternVariant(PatternCategory),
    /// For AtomType::Acquisition — how was this acquired
    AcquisitionVariant(AcquisitionSource),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Polarity {
    Positive,
    Negative,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Voice {
    Active,
    Passive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FrameSource {
    RuleBased,        // Phase 1: deterministic rules
    UdParse,          // Phase 2: dependency parsing
    SrlLabel,         // Phase 2: semantic role labeling
    AmrCompilation,   // Phase 3: AMR graph compilation
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PatternCategory {
    EventPattern,
    CausalChain,
    GoalAction,
    RoleSubstitution,
    TemporalSequence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AcquisitionSource {
    PassiveRecall,
    SelfStudy,
    UserAnswer,
    ExternalSource,
}
```

### SemanticRole — Comprehensive Role Taxonomy

```rust
/// Unified semantic role taxonomy for all atom types.
/// Event roles, hidden meaning roles, pattern roles — one enum.
#[derive(Debug, Clone, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub enum SemanticRole {
    // === Event frame roles (from MD-1) ===
    Predicate,
    Arg0Agent,
    Arg1Patient,
    Arg2Recipient,
    Cause,
    Purpose,
    Location,
    Time,
    Instrument,

    // === Hidden meaning roles (from MD-2) ===
    Problem,
    Solution,
    Beneficiary,
    Tool,
    Motivation,
    PainPoint,
    ImpliedGoal,

    // === Pattern roles ===
    PatternType,
    Antecedent,
    Consequent,

    // === Structural roles ===
    SourceAtom,       // reference to producing atom
    SourceEvent,      // reference to source event
    EquivalentOf,    // semantic equivalence link
}
```

### How SemanticAtom Unifies Ingest

**Before (v11.0): Two separate paths:**
```text
Token path: text → tokens → nodes → senses
Frame path: text → EventFrame → ??? → nodes → senses
```

**After (v12.0): One unified path:**
```text
text → Atomizer → Vec<SemanticAtom> → IngestAtoms → graph
```

Examples:

```text
"raja"
→ SemanticAtom { atom_type: Token, label: "raja", roles: {} }

"Raymond membuat aplikasi karena lambat"
→ SemanticAtom { atom_type: Event, label: "membuat",
    roles: {Arg0Agent: "Raymond", Arg1Patient: "aplikasi", Cause: "lambat"} }

[pre-ingest reasoning on above event]
→ SemanticAtom { atom_type: HiddenMeaning, label: "problem_solution",
    roles: {Problem: "lambat", Solution: "aplikasi", Agent: "Raymond"},
    variant: MeaningVariant(ProblemSolutionPattern) }
```

All three enter through **one** ingest pipeline. No dual-track.

### Ambiguous Token Detection

Tokens like "dia", "itu", "mereka" are sources of major ambiguity, but the old
`DetectGaps` ignored them with `_ => {}`. The `AmbiguousToken` variant enables
gap detection for tokens that require context resolution.

When `is_ambiguous_token()` returns true, the Tokenize transform produces
`SemanticAtom { atom_type: AmbiguousToken, ... }` instead of `AtomType::Token`.
This enables DetectGaps in MD-6 to produce `KnowledgeGap(AmbiguousReferenceGap)`
for ambiguous tokens, which SelectAcquisition can then resolve via
PassiveRecall (graph disambiguation) or AskUser.

```rust
/// Heuristic: is this token ambiguous and in need of disambiguation?
/// Pronouns, deictics, and common ambiguous tokens need context resolution.
pub fn is_ambiguous_token(label: &str, graph: &Graph) -> bool {
    // 1. Known pronoun sets (language-agnostic where possible)
    let pronouns = [
        // Indonesian
        "dia", "ia", "mereka", "kita", "kami", "aku", "kamu", "ia", "ini", "itu",
        // English
        "he", "she", "it", "they", "we", "us", "them", "this", "that",
    ];
    if pronouns.contains(&label.to_lowercase().as_str()) {
        return true;
    }

    // 2. Graph-based: token has multiple senses (high ambiguity)
    if let Some(node) = graph.find_node_by_label(label) {
        if node.sense_count() >= 3 {
            return true;  // 3+ senses = ambiguous without context
        }
    }

    // 3. Graph-based: token never co-occurs with seeds (no grounding anchor)
    if let Some(node) = graph.find_node_by_label(label) {
        if node.confidence < 0.2 && node.status == NodeStatus::New {
            return true;  // ungrounded new token needs context
        }
    }

    false
}
```

Ambiguous token example:

```text
"dia"
→ SemanticAtom { atom_type: AmbiguousToken, label: "dia", roles: {} }
→ DetectGaps: AmbiguousReferenceGap ("dia" has 4 senses, needs context)
→ SelectAcquisition: PassiveRecall (graph resolves from recent context) or AskUser
```

---

## Abstraction 2: Composition — Universal Structured Grouping

When a SemanticAtom is ingested into the RSVS graph, it becomes a **Composition**:
a structured group of nodes with typed roles, lifecycle state, epistemic state, and
seed alignment scores.

This replaces the separate EventFrame, HiddenMeaningCandidate, Pattern, and
AbductiveHypothesis types with ONE grouping mechanism.

```rust
/// Universal structured grouping in the RSVS graph.
/// Every structured piece of knowledge is a Composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Composition {
    pub id: CompositionId,
    pub composition_type: CompositionType,

    // Members: which nodes participate, and in what role
    pub members: Vec<CompositionMember>,

    // Dual-axis status (Abstraction 3)
    pub lifecycle: LifecycleState,
    pub epistemic: EpistemicState,

    // Confidence and provenance
    pub confidence: f32,
    pub provenance: ProvenanceChain,

    // Seed alignment scores (Abstraction 6)
    pub seed_scores: HashMap<SeedPrimitive, f32>,

    // Source text — the original raw text that produced this composition.
    // Needed by ReExtractFrame to re-run extraction with graph context.
    // None for compositions not derived from text (e.g., Pattern, Hypothesis).
    pub source_text: Option<String>,

    // Batch tracking — how many ingest batches has this composition survived?
    // Incremented by GovernBeliefs at each ingest cycle.
    // Used by promotion criteria: Candidate → Stable requires age ≥ 3 batches.
    pub batch_seen: usize,

    // Contradiction history — tracks which batches had contradictions involving
    // this composition. Used by has_recent_contradiction() for promotion gating.
    pub contradiction_batches: Vec<usize>,

    // Metadata
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionMember {
    pub node_id: NodeId,
    pub role: SemanticRole,
    pub confidence: f32,
}

impl Composition {
    /// How many ingest batches has this composition existed for?
    /// GovernBeliefs increments batch_seen at the end of each ingest cycle.
    pub fn age_in_batches(&self) -> usize {
        self.batch_seen
    }

    /// Was this composition involved in a contradiction within the last N batches?
    /// Used by can_promote_to_grounded() to deny promotion if recent contradictions exist.
    pub fn has_recent_contradiction(&self, last_n: usize) -> bool {
        let threshold = self.batch_seen.saturating_sub(last_n);
        self.contradiction_batches.iter().any(|&b| b > threshold)
    }

    /// Count independent provenance sources contributing to this composition.
    /// A source is "independent" if it has a different EdgeSource origin.
    /// Two members from FrameCompiler count as 1 source; one from FrameCompiler
    /// and one from EnrichmentFeedback count as 2 independent sources.
    pub fn provenance_source_count(&self, graph: &Graph) -> usize {
        let mut origins: HashSet<EdgeSource> = HashSet::new();
        origins.insert(self.provenance.origin.clone());
        for member in &self.members {
            if let Some(edge) = graph.get_edge(self.id.clone(), member.node_id) {
                origins.insert(edge.source.clone());
            }
        }
        origins.len()
    }

    /// Find the member playing a specific semantic role in this composition.
    /// Returns None if no member has that role.
    ///
    /// This is the primary accessor for role-based lookup, used by:
    /// - detect_contradiction() to compare roles across compositions
    /// - graph_find_role_candidate() to find role fillers
    /// - resolve_ambiguous_from_graph() to find referents
    /// - has_equivalence_mismatch() to compare role fillers
    pub fn member_with_role(&self, role: &SemanticRole) -> Option<&CompositionMember> {
        self.members.iter().find(|m| m.role == *role)
    }

    /// Check if this composition has a member with a specific role.
    /// Returns true if any member has the given role.
    ///
    /// Overloaded variant: checks for role existence without comparing
    /// the member's value. Used by detect_contradiction() to filter
    /// compositions that have a Predicate role.
    pub fn has_member_with_role(&self, role: SemanticRole) -> bool {
        self.members.iter().any(|m| m.role == role)
    }

    /// Check if this composition has a member with a specific role AND
    /// whose node label matches the given predicate string.
    ///
    /// Used by graph_has_relevant_context() and graph_find_role_candidate()
    /// to find compositions with a specific predicate label.
    pub fn has_member_with_role_and_label(&self, role: SemanticRole, label: &str) -> bool {
        self.members.iter().any(|m| m.role == role && m.label() == label)
    }

    /// Get the opposing composition ID from a contradiction.
    ///
    /// After detect_contradiction() marks a composition as Contradicted,
    /// it stores the opposing composition ID in the Contradiction struct.
    /// This method retrieves that ID for use by check_contradiction_resolution()
    /// to find the opposing composition and attempt resolution.
    ///
    /// Returns None if the composition has no recorded contradiction,
    /// or if the contradiction doesn't specify an opposing composition.
    pub fn contradiction_opposing_id(&self) -> Option<CompositionId> {
        // The Contradiction is stored on the composition after governance.
        // In practice, this is a field on the composition that gets set
        // when GovernBeliefs marks it as Contradicted.
        self.contradiction.as_ref().map(|c| c.opposing_composition_id.clone())
    }
}

/// Extension: CompositionMember label lookup.
/// Allows `member.label()` to get the node label from the graph.
impl CompositionMember {
    /// Get the label for this member's node.
    /// Requires graph lookup — provided as a convenience for filtering.
    /// In practice, the label is often cached or available from context.
    pub fn label(&self) -> &str {
        // In a full implementation, this would look up the node label
        // from the graph. For design spec purposes, we assume the label
        // is either cached or the comparison is done via node_id.
        // The has_member_with_role_and_label() method uses this.
        "" // placeholder — real impl uses graph.get_node(self.node_id).label
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompositionType {
    Event,          // was EventFrame
    HiddenMeaning,  // was HiddenMeaningCandidate
    Pattern,        // was pattern from PatternMining
    Situation,      // was situation state fragment
    Hypothesis,     // was AbductiveHypothesis
    Acquisition,    // externally acquired knowledge
}
```

### Why This Unifies

| Old Type | Becomes | composition_type | Key Roles |
|----------|---------|-----------------|-----------|
| EventFrame | Composition | Event | Arg0Agent, Arg1Patient, Cause, Purpose |
| HiddenMeaningCandidate | Composition | HiddenMeaning | Problem, Solution, Agent |
| Pattern | Composition | Pattern | Antecedent, Consequent, PatternType |
| AbductiveHypothesis | Composition | Hypothesis | source nodes + inferred target |
| SituationState | Composition | Situation | event refs + environmental context |

### What This Enables

```text
structural_similarity() can compare:
  Event vs Event         — same event structure?
  HiddenMeaning vs Event — does this hidden meaning align with this event?
  Pattern vs Event       — does this event match a known pattern?
  HiddenMeaning vs HiddenMeaning — do these overlap?

convergence can merge:
  Active "Raymond membuat aplikasi" + Passive "Aplikasi dibuat oleh Raymond"
  → same Composition(Event), unified

cross-type reasoning:
  Event(Cause=lambat) + HiddenMeaning(Problem=lambat, Solution=aplikasi)
  → the cause in the event IS the problem in the hidden meaning
  → this is detectable because both are Compositions with role-based members
```

---

## Abstraction 3: Two Orthogonal Status Axes

v11.0 has 4 overlapping lifecycle enums. v12.0 has **2 orthogonal axes**.

```rust
/// Axis 1: Structural lifecycle — how mature is this entity in the graph?
/// Applies to: Nodes, Compositions, Senses
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum LifecycleState {
    #[default]
    New,         // just created
    Candidate,   // under consideration
    Stable,      // established
    Deprecated,  // no longer trusted
    Quarantine,  // isolated for review
}

/// Axis 2: Epistemic confidence — how confident are we in this knowledge?
/// Applies to: Compositions, Knowledge Claims
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum EpistemicState {
    #[default]
    Observed,      // directly extracted from input
    Inferred,      // derived by deterministic rule
    Hypothesis,    // unconfirmed scenario reasoning
    Grounded,      // repeatedly supported by independent evidence
    Contradicted,  // opposed by evidence
}
```

### Mapping from v11.0

| v11.0 Type | v11.0 Values | v12.0 Mapping |
|-----------|-------------|---------------|
| NodeStatus | New, Candidate, Stable, Deprecated, Quarantine | LifecycleState (identical) |
| CandidateStatus | Candidate, Confirmed, Contradicted, Deprecated | (Candidate, EpistemicState) pair |
| BeliefState | Observed, Candidate, Inferred, Hypothesis, Grounded, Weak, Contradicted, Deprecated, Rejected, Recovered | EpistemicState + LifecycleState |
| GroundingVerdict | WellGrounded, NeedsReview, NeedsRevision | Derived from (LifecycleState, EpistemicState) |

### Semantic Combinations

```text
(New, Observed)         = fresh direct observation
(Candidate, Inferred)   = rule-derived, under review
(Stable, Grounded)      = well-established, repeatedly confirmed
(Quarantine, Hypothesis)= unconfirmed scenario, isolated
(Stable, Contradicted)  = was established, now contradicted
(Deprecated, Contradicted) = abandoned belief
(Candidate, Observed)   = fresh observation, not yet promoted
```

"BeliefState::Weak" from v11.0 maps to `(Candidate, Inferred)` with low confidence.
"BeliefState::Rejected" maps to `(Deprecated, Contradicted)`.
"BeliefState::Recovered" maps to transition from `(Stable, Contradicted)` back to `(Stable, Grounded)`.

### Transition Rules

```text
Lifecycle transitions (structural maturity):
  New → Candidate → Stable → Deprecated
               ↓
           Quarantine → Candidate (recovered)

Epistemic transitions (truth confidence):
  Observed → Inferred → Grounded
  Hypothesis → Grounded (if confirmed)
  Any → Contradicted (if opposing evidence)
  Grounded → Contradicted → Grounded (recovery)

Cross-axis: lifecycle and epistemic transitions are INDEPENDENT.
  A node can become Stable while still Hypothesis.
  A composition can become Contradicted while still Stable.
```

### GroundingVerdict as Derived Function

```rust
fn grounding_verdict(lifecycle: &LifecycleState, epistemic: &EpistemicState, confidence: f32) -> GroundingVerdict {
    match (lifecycle, epistemic) {
        (Stable, Grounded) if confidence > 0.8 => GroundingVerdict::WellGrounded,
        (Stable | Candidate, Contradicted) => GroundingVerdict::NeedsRevision,
        _ => GroundingVerdict::NeedsReview,
    }
}
```

No separate GroundingVerdict enum needed — it's derived from the two axes.

---

## Abstraction 4: SemanticEdge — Typed Triple

v11.0 has 4 edge/classification systems. v12.0 has **1 edge structure** with 3 dimensions.

```rust
/// Single edge type with three orthogonal dimensions.
/// Every edge in the graph is a SemanticEdge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEdge {
    /// WHAT kind of semantic relation: Categorical, Causal, Functional, etc.
    /// Classifies the nature of the relationship.
    pub relation: RelationType,

    /// OPTIONAL: role if this edge is part of a structured composition.
    /// Only populated for edges linking composition members.
    /// Token-based edges have role = None.
    pub role: Option<SemanticRole>,

    /// WHERE this edge came from: provenance.
    /// Replaces separate ProvenanceSource — all provenance is EdgeSource.
    pub source: EdgeSource,
}
```

### RelationType (unchanged from v11.0)

```rust
#[non_exhaustive]
pub enum RelationType {
    Categorical,    // is-a, member-of
    Differential,   // contrast, negation
    Functional,     // enables, requires
    Spatial,        // near, contains
    Temporal,       // before, during
    Causal,         // causes, prevents
    Discursive,     // topic, reference
}
```

### EdgeSource (extended from v11.0)

```rust
#[non_exhaustive]
pub enum EdgeSource {
    // v11.0 existing
    Bootstrap,
    Learned,
    Composition,
    GapDetection,
    Discourse,
    Blending,
    Abductive,
    PatternMining,
    Synthesis,
    CompoundDiscovery,

    // v12.0 new (replaces ProvenanceSource)
    FrameCompiler,         // from MD-1: semantic frame extraction
    HiddenMeaningRule,     // from MD-2: pre-ingest reasoning
    EpistemicGovernance,   // from MD-4: belief state transition
    ExecutiveControl,      // from MD-5: executive routing
    AcquisitionRecall,     // from MD-6: passive recall
    AcquisitionSelfStudy,  // from MD-6: self-study
    AcquisitionUserAnswer, // from MD-6: user answer
    HumanAssertion,        // human override

    // v12.0 feedback loop (detection → repair cycle)
    EnrichmentFeedback,    // from feedback loop: composition enriched after gap detection
    ExtractionRepair,      // from feedback loop: frame re-extracted with graph context
}
```

`ProvenanceSource` from the adjusted MD4 is **eliminated** — its variants are absorbed
into `EdgeSource`. Provenance is a property of edges and compositions, tracked via
`EdgeSource`, not a separate classification axis.

### Examples

```text
Token-based edge:
  SemanticEdge { relation: Categorical, role: None, source: Bootstrap }

Event composition member edge:
  SemanticEdge { relation: Categorical, role: Some(Arg0Agent), source: FrameCompiler }

Hidden meaning composition member edge:
  SemanticEdge { relation: Causal, role: Some(Problem), source: HiddenMeaningRule }

Pattern composition member edge:
  SemanticEdge { relation: Causal, role: Some(Antecedent), source: PatternMining }

Enrichment feedback edge:
  SemanticEdge { relation: Categorical, role: Some(Arg0Agent), source: EnrichmentFeedback }

Re-extraction repair edge:
  SemanticEdge { relation: Categorical, role: Some(Cause), source: ExtractionRepair }
```

---

## Abstraction 5: Transform — DAG of Declarative Processing

v11.0 has a hardcoded pipeline: `ingest_text()` → tokenize → co-occurrence → promote → sense → batch → gap → discourse → etc.

v12.0 replaces this with a **declarative transform graph**: each processing step is a
`Transform` that declares its input and output types. The pipeline engine routes data
through available transforms based on type compatibility.

```rust
/// Declarative processing unit.
/// Each transform declares what it consumes and produces.
/// The pipeline engine routes data through transforms by type.
pub trait Transform: Send + Sync {
    type Input;
    type Output;

    fn id(&self) -> &'static str;
    fn transform(&self, input: Self::Input, ctx: &mut PipelineContext) -> Self::Output;
}
```

### Core Transforms (mapping from v11.0 pipeline stages)

```text
Transform           Input              Output              MD Source
─────────────────── ────────────────── ─────────────────── ─────────
Tokenize            RawText            Vec<SemanticAtom>   existing
ExtractFrame        RawText            Option<SemanticAtom> MD-1
ReasonFrame         SemanticAtom       Vec<SemanticAtom>   MD-2
IngestAtoms         Vec<SemanticAtom>  GraphDelta          existing
GovernBeliefs       GraphDelta         GovernedDelta       MD-4
SeedAnchor          GovernedDelta      AnchoredDelta       MD-4
RunPrediction       GraphSnapshot      Vec<SemanticAtom>   existing
RunSituation        GraphSnapshot      SituationUpdate     existing
DetectGaps          GraphSnapshot      Vec<KnowledgeGap>   MD-6
SelectAcquisition   Vec<KnowledgeGap>  AcquisitionDecision MD-6
RunReasoning        ReasoningRequest   ReasoningResult     existing
Appraise            ReasoningResult    AppraiseVerdict     existing
EnrichComposition   EnrichmentRequest  GraphDelta          revised
ReExtractFrame      ReExtractionRequest Option<SemanticAtom> revised
```

### DAG-Based Pipeline Engine

```rust
/// Transform node in the DAG.
/// Each node knows what it produces, and the engine routes data to it.
struct TransformNode {
    transform_id: TypeId,
    input_type: TypeId,
    output_type: TypeId,
    /// Which transforms must complete before this one can run
    dependencies: Vec<TypeId>,
    /// Condition: should this transform run given the current context?
    condition: Option<Box<dyn Fn(&PipelineContext) -> bool + Send + Sync>>,
}

/// DAG-based pipeline engine.
/// Transforms register with their types and dependencies.
/// The engine topologically sorts them and routes data automatically.
pub struct PipelineEngine {
    transforms: HashMap<TypeId, Box<dyn Transform>>,
    dag: Vec<TransformNode>,
    context: PipelineContext,
}

impl PipelineEngine {
    /// Register a transform with its dependency chain.
    /// No need to modify ingest() — the DAG routes data automatically.
    pub fn register<T: Transform>(
        &mut self,
        transform: T,
        dependencies: Vec<TypeId>,
        condition: Option<Box<dyn Fn(&PipelineContext) -> bool + Send + Sync>>,
    ) {
        let node = TransformNode {
            transform_id: TypeId::of::<T>(),
            input_type: TypeId::of::<T::Input>(),
            output_type: TypeId::of::<T::Output>(),
            dependencies,
            condition,
        };
        self.dag.push(node);
        self.transforms.insert(TypeId::of::<T>(), Box::new(transform));
    }

    /// Execute the DAG: topologically sort, then run each transform
    /// whose condition is met and whose dependencies have completed.
    pub fn execute_dag(&mut self, initial_input: &str) -> IngestResult {
        // 1. Topological sort the DAG (cached after registration)
        let sorted = self.topological_sort();

        // 2. Run each transform in order
        for node in &sorted {
            // Skip if condition not met
            if let Some(ref cond) = node.condition {
                if !cond(&self.context) { continue; }
            }

            // Skip if any dependency hasn't produced output yet
            if !self.dependencies_met(node) { continue; }

            // Run the transform
            self.run_node(node);
        }

        self.collect_result()
    }

    /// Convenience: full ingest pipeline using the DAG.
    /// This is the same as calling execute_dag() with the standard
    /// transform registration (see register_default_pipeline below).
    pub fn ingest(&mut self, text: &str) -> IngestResult {
        self.context.set_raw_text(text);
        self.execute_dag(text)
    }

    /// Apply a governed/anchored delta and return a ReflectionLoopResult.
    /// Used by Reflective mode's reflection loop to track evidence accumulation
    /// and goal satisfaction across iterations.
    ///
    /// Unlike plain apply(), this returns structured feedback that
    /// ReasoningState.update() can consume.
    pub fn apply_with_result(&mut self, anchored: AnchoredDelta) -> ReflectionLoopResult {
        let current_confidence = anchored.compositions.iter()
            .map(|c| c.confidence)
            .fold(0.0_f32, |a, b| a.max(b));
        let modified: Vec<CompositionId> = anchored.compositions.iter()
            .map(|c| c.id.clone())
            .collect();
        let has_gaps = anchored.compositions.iter()
            .any(|c| c.epistemic == EpistemicState::Inferred && c.confidence < 0.5);

        self.apply(anchored);

        ReflectionLoopResult {
            current_confidence,
            elapsed_ms: 0, // tracked externally by ExecutiveOrchestrator
            evidence_count: modified.len(),
            modified_compositions: modified,
            has_gaps,
            resolved_contradictions: vec![],  // filled by governance
            filled_gaps: vec![],              // filled by enrichment
        }
    }

    /// Find compositions in the graph that are candidates for re-extraction.
    /// A "weak frame" is one with low confidence AND missing expected roles.
    /// Used by Reflective mode to identify which frames to re-extract
    /// with graph context.
    pub fn find_weak_frames(&self) -> Vec<WeakFrame> {
        self.context.graph.compositions()
            .filter(|c| c.confidence < 0.5)
            .filter(|c| c.composition_type == CompositionType::Event)
            .filter(|c| !c.has_member_with_role(SemanticRole::Arg0Agent)
                      || !c.has_member_with_role(SemanticRole::Arg1Patient))
            .map(|c| WeakFrame {
                composition_id: c.id.clone(),
                atom_id: c.provenance.origin_id.clone(),
                source_text: c.source_text.clone(),
            })
            .collect()
    }

    /// Get a graph snapshot for the current state of the pipeline.
    pub fn snapshot(&self) -> GraphSnapshot {
        GraphSnapshot {
            graph: self.context.graph.clone(),
            recent_atoms: self.context.current_atoms.clone(),
        }
    }

    /// Get a reference to the graph.
    pub fn graph(&self) -> &Graph {
        &self.context.graph
    }
}

/// A weak frame identified for re-extraction.
/// Contains the info needed to construct a ReExtractionRequest.
pub struct WeakFrame {
    composition_id: CompositionId,
    atom_id: String,
    source_text: Option<String>,
}

impl WeakFrame {
    pub fn composition_id(&self) -> &CompositionId { &self.composition_id }
    pub fn atom_id(&self) -> &str { &self.atom_id }
    pub fn source_text(&self) -> Option<&str> { self.source_text.as_deref() }
    pub fn composition(&self) -> Option<Composition> { None /* resolved at call site from graph */ }
}

/// Graph snapshot for passing to DetectGaps.
pub struct GraphSnapshot {
    pub graph: Graph,
    pub recent_atoms: Vec<SemanticAtom>,
}

impl GraphSnapshot {
    /// Build graph context for re-extracting a weak frame.
    /// Returns (role, node_id, confidence) triples from compositions
    /// that share the same predicate, providing known role-fillers
    /// as hints for the rule-based re-extraction.
    pub fn context_for(&self, weak_frame: &WeakFrame) -> Vec<(SemanticRole, NodeId, f32)> {
        let target_comp = self.graph.get_composition(&weak_frame.composition_id);
        match target_comp {
            Some(comp) => {
                // Find the predicate of the weak frame
                let predicate = comp.member_with_role(&SemanticRole::Predicate)
                    .map(|m| m.node_id);

                match predicate {
                    Some(pred_id) => {
                        // Find other compositions with the same predicate
                        // and collect their role fillers as context hints
                        self.graph.compositions()
                            .filter(|c| c.id != comp.id)
                            .filter(|c| c.composition_type == CompositionType::Event)
                            .filter(|c| {
                                c.member_with_role(&SemanticRole::Predicate)
                                    .map(|m| m.node_id == pred_id)
                                    .unwrap_or(false)
                            })
                            .flat_map(|c| {
                                c.members.iter()
                                    .filter(|m| m.role != SemanticRole::Predicate)
                                    .map(|m| (m.role.clone(), m.node_id, m.confidence))
                                    .collect::<Vec<_>>()
                            })
                            .collect()
                    },
                    None => vec![],
                }
            },
            None => vec![],
        }
    }
}

/// Default pipeline registration.
/// This replaces the hardcoded ingest() — transforms declare themselves.
fn register_default_pipeline(engine: &mut PipelineEngine) {
    // Layer 0: Always run
    engine.register::<Tokenize>(Tokenize, vec![], None);

    // MD-1: Extract frame (only if sentence-like)
    engine.register::<ExtractFrame>(ExtractFrame, vec![TypeId::of::<Tokenize>()],
        Some(Box::new(|ctx| ctx.is_sentence_like())));

    // MD-2: Reason on frames (only if event atoms exist)
    engine.register::<ReasonFrame>(ReasonFrame, vec![TypeId::of::<ExtractFrame>()],
        Some(Box::new(|ctx| ctx.has_event_atoms())));

    // Core: Ingest all atoms
    engine.register::<IngestAtoms>(IngestAtoms,
        vec![TypeId::of::<Tokenize>(), TypeId::of::<ReasonFrame>()], None);

    // MD-4: Govern beliefs
    engine.register::<GovernBeliefs>(GovernBeliefs, vec![TypeId::of::<IngestAtoms>()], None);

    // MD-4: Seed anchor
    engine.register::<SeedAnchor>(SeedAnchor, vec![TypeId::of::<GovernBeliefs>()], None);

    // MD-6: Detect gaps (optional, controlled by executive)
    engine.register::<DetectGaps>(DetectGaps, vec![TypeId::of::<SeedAnchor>()],
        Some(Box::new(|ctx| ctx.gap_detection_enabled())));

    // MD-6: Select acquisition
    engine.register::<SelectAcquisition>(SelectAcquisition, vec![TypeId::of::<DetectGaps>()],
        Some(Box::new(|ctx| ctx.has_gaps())));

    // Feedback: EnrichComposition (only if acquisition produced enrichment)
    engine.register::<EnrichComposition>(EnrichComposition,
        vec![TypeId::of::<SelectAcquisition>()],
        Some(Box::new(|ctx| ctx.has_enrichment_requests())));

    // Feedback: ReExtractFrame (only if acquisition produced re-extraction)
    engine.register::<ReExtractFrame>(ReExtractFrame,
        vec![TypeId::of::<SelectAcquisition>()],
        Some(Box::new(|ctx| ctx.has_reextraction_requests())));
}

/// Adding a new MD = registering a transform.
/// No ingest() modification needed.
///
/// Example: adding MD-7 would be:
///   engine.register::<MD7Transform>(MD7Transform, vec![TypeId::of::<SeedAnchor>()], None);
///
/// The DAG automatically includes it in the execution order.
```

The old ingest() method (if-else chain) is replaced by the DAG engine. The
topological sort ensures transforms execute in dependency order. Conditions
gate whether a transform runs (e.g., ExtractFrame only for sentence-like input).
Adding a new MD now truly requires only `engine.register()` — no pipeline code changes.

**Old approach (kept for reference — the hardcoded if-else chain that the DAG replaces):**

```rust
// OLD: Hardcoded procedural ingest — required editing this method for every new MD.
// This is replaced by the DAG engine above.
//
// pub struct PipelineEngine {
//     transforms: HashMap<TypeId, Box<dyn Transform>>,
//     context: PipelineContext,
// }
//
// impl PipelineEngine {
//     pub fn ingest(&mut self, text: &str) -> IngestResult {
//         // 1. Tokenize (always)
//         let mut atoms = self.run::<Tokenize>(text);
//         // 2. Extract frame (if sentence-like)
//         if is_sentence_like(text) {
//             if let Some(frame_atom) = self.run::<ExtractFrame>(text) {
//                 atoms.push(frame_atom);
//             }
//         }
//         // 3. Reason on event atoms (if any)
//         let event_atoms: Vec<_> = atoms.iter()
//             .filter(|a| a.atom_type == AtomType::Event)
//             .cloned()
//             .collect();
//         for event in event_atoms {
//             let hidden = self.run::<ReasonFrame>(&event);
//             atoms.extend(hidden);
//         }
//         // 4. Ingest all atoms into graph
//         let delta = self.run::<IngestAtoms>(&atoms);
//         // 5. Govern beliefs
//         let governed = self.run::<GovernBeliefs>(&delta);
//         // 6. Seed-anchor confidence
//         let anchored = self.run::<SeedAnchor>(&governed);
//         // 7. Apply to graph
//         self.apply(anchored)
//     }
// }
```

### PipelineContext — Shared State

```rust
/// Shared state across all transforms in the pipeline.
/// Each transform reads from and writes to this context.
pub struct PipelineContext {
    // Raw input
    raw_text: Option<String>,

    // Atom accumulation (built up through the pipeline)
    current_atoms: Vec<SemanticAtom>,

    // Event history for cross-atom reasoning (MD-2 ReasonFrame)
    recent_events: Vec<SemanticAtom>,

    // Graph reference (for graph-guided operations)
    graph: Graph,

    // Extraction quality tracker (MD-1 feedback)
    extraction_quality: ExtractionQualityTracker,

    // Enrichment requests produced by SelectAcquisition (feedback loop)
    pending_enrichments: Vec<EnrichmentRequest>,
    pending_reextractions: Vec<ReExtractionRequest>,

    // Gap detection control
    gap_detection_enabled: bool,
    pending_gaps: Vec<KnowledgeGap>,

    // Atom ID counter
    next_atom_id: u64,
}

impl PipelineContext {
    /// Maximum recent events to keep in the sliding window.
    /// Prevents unbounded memory growth while preserving enough context
    /// for PolarityConflictRule and cross-atom reasoning.
    const RECENT_EVENTS_WINDOW: usize = 50;

    /// Add an event atom to recent_events with window management.
    pub fn record_event(&mut self, atom: SemanticAtom) {
        if atom.atom_type == AtomType::Event {
            self.recent_events.push(atom);
            // Sliding window: trim if exceeds limit
            if self.recent_events.len() > Self::RECENT_EVENTS_WINDOW {
                self.recent_events.remove(0);
            }
        }
    }

    /// Get recent events for ReasoningContext (MD-2).
    pub fn recent_events(&self) -> &Vec<SemanticAtom> {
        &self.recent_events
    }

    // Condition helpers for DAG gating
    pub fn is_sentence_like(&self) -> bool {
        self.raw_text.as_ref().map(|t| is_sentence_like(t)).unwrap_or(false)
    }

    pub fn has_event_atoms(&self) -> bool {
        self.current_atoms.iter().any(|a| a.atom_type == AtomType::Event)
    }

    pub fn gap_detection_enabled(&self) -> bool { self.gap_detection_enabled }
    pub fn has_gaps(&self) -> bool { !self.pending_gaps.is_empty() }
    pub fn has_enrichment_requests(&self) -> bool { !self.pending_enrichments.is_empty() }
    pub fn has_reextraction_requests(&self) -> bool { !self.pending_reextractions.is_empty() }

    pub fn set_raw_text(&mut self, text: &str) {
        self.raw_text = Some(text.to_string());
    }

    pub fn next_atom_id(&mut self) -> u64 {
        let id = self.next_atom_id;
        self.next_atom_id += 1;
        id
    }
}

/// Graph neighborhood for local mode selection.
/// Instead of scanning the entire graph for contradictions or low confidence,
/// mode selection evaluates only the neighborhood relevant to current input.
#[derive(Debug, Clone)]
pub struct GraphNeighborhood {
    pub compositions: Vec<Composition>,
}

impl GraphNeighborhood {
    /// Are there any contradicted compositions in this neighborhood?
    pub fn has_contradictions(&self) -> bool {
        self.compositions.iter().any(|c| c.epistemic == EpistemicState::Contradicted)
    }

    /// Average confidence across neighborhood compositions.
    pub fn average_confidence(&self) -> f32 {
        if self.compositions.is_empty() { return 1.0; }
        self.compositions.iter().map(|c| c.confidence).sum::<f32>()
            / self.compositions.len() as f32
    }
}

impl Graph {
    /// Extract the local neighborhood around a set of keywords.
    /// Returns compositions that are directly connected to nodes
    /// matching any of the input keywords.
    ///
    /// This scopes mode selection to input-relevant context only.
    /// Using global graph stats would force Reflective mode for every
    /// input just because 3 contradictions exist in an unrelated part
    /// of the graph.
    pub fn neighborhood_for(&self, keywords: &[String]) -> GraphNeighborhood {
        let relevant_node_ids: HashSet<NodeId> = keywords.iter()
            .filter_map(|kw| self.find_node_by_label(kw))
            .map(|n| n.id)
            .collect();

        let compositions: Vec<Composition> = self.compositions()
            .filter(|c| {
                // Composition is in the neighborhood if any of its members
                // reference a node matching an input keyword
                c.members.iter().any(|m| relevant_node_ids.contains(&m.node_id))
            })
            .cloned()
            .collect();

        GraphNeighborhood { compositions }
    }
}

/// Extract significant keywords from input text for neighborhood lookup.
/// Filters out stop words and short tokens, returning the remaining
/// words as candidate labels for graph node matching.
///
/// This is a simple rule-based extraction — no LLM needed.
/// For multilingual support, stop word lists are extended per language.
pub fn extract_keywords(input: &str) -> Vec<String> {
    let stop_words = [
        // Indonesian
        "yang", "di", "ke", "dari", "dan", "atau", "ini", "itu", "dengan",
        "untuk", "pada", "adalah", "akan", "telah", "oleh", "juga",
        // English
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after",
    ];
    input.split_whitespace()
        .map(|w| w.to_lowercase()
            .chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|w| w.len() > 2)
        .filter(|w| !stop_words.contains(&w.as_str()))
        .collect()
}
```

RECENT_EVENTS_WINDOW = 50 is chosen because:
- PolarityConflictRule needs to compare current event against recent events
- Most contradictions appear within 10-20 events of each other
- 50 provides ample margin without significant memory overhead
- Each SemanticAtom is small (~200 bytes), so 50 events ≈ 10 KB

### Adding a New MD = Adding a Transform

```text
To add MD-1:  implement ExtractFrame transform
To add MD-2:  implement ReasonFrame transform
To add MD-4:  implement GovernBeliefs + SeedAnchor transforms
To add MD-6:  implement DetectGaps + SelectAcquisition transforms
```

No pipeline modifications needed. New transforms are registered and called.

```text
Feedback loop:
  SelectAcquisition → RecallAction::EnrichComposition → EnrichComposition Transform
  SelectAcquisition → RecallAction::ReExtractFrame    → ReExtractFrame Transform
  process_user_answer_merge() → EnrichmentRequest     → EnrichComposition Transform
```

---

## Feedback Loop — Closing the Detection-Repair Cycle

### The Broken Loop Problem

In the original v11.0–v12.0 pipeline, MD-1 (ExtractFrame) produces SemanticAtoms via
rule-based extraction — it never knows if its understanding is correct. MD-6 (DetectGaps)
detects weaknesses in those frames (missing roles, low confidence), but the gap is sent to
AskUser or Deferred — **not** back to fix the original Composition or re-run extraction.
The loop is broken.

This means: gap detection produces knowledge about what's missing, but that knowledge never
flows back to repair the compositions that are missing it. The graph accumulates partial
compositions that never improve, even when the graph itself already contains the missing
information.

The feedback loop closes this gap by introducing two pathways:

1. **EnrichComposition**: When SelectAcquisition's PassiveRecall finds a candidate node in
   the graph that could fill a missing role, or when a user answer provides the missing
   information, an `EnrichmentRequest` is produced. This flows into the `EnrichComposition`
   transform, which patches the existing composition with the new member — no new atom
   needed, no re-extraction needed.

2. **ReExtractFrame**: When gap detection reveals systematic extraction failures (e.g.,
   ExtractFrame consistently misses the Cause role for certain verb patterns), a
   `ReExtractionRequest` is produced. This flows into the `ReExtractFrame` transform, which
   re-runs extraction with graph context as hints — the graph already knows likely
   role-fillers, and those hints improve the blind rule-based extraction.

Both pathways flow back through GovernBeliefs for re-evaluation, ensuring that enriched
or re-extracted compositions are properly governed before being committed to the RSVS
Memory Core.

### EnrichmentRequest

```rust
/// Request to enrich an existing Composition with a missing or inferred role.
/// Produced by SelectAcquisition when PassiveRecall finds a candidate in the graph,
/// or by process_user_answer_merge() when a user answer fills a gap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentRequest {
    pub target_composition_id: CompositionId,
    pub role_to_fill: SemanticRole,
    pub candidate_node_id: NodeId,
    pub candidate_label: String,
    pub source: EnrichmentSource,
    pub confidence: f32,
}

impl EnrichmentRequest {
    /// Construct an EnrichmentRequest from an improved atom produced
    /// by ReExtractFrame. The improved atom contains new/updated role
    /// fillers that should be merged into the target composition.
    ///
    /// Used by the enrichment loop in MD-5 when ReExtractFrame succeeds.
    pub fn from_improved_atom(atom: SemanticAtom) -> Self {
        // The improved atom's first role becomes the role to fill.
        // In practice, the atom may have multiple roles — each gets
        // its own EnrichmentRequest. For spec clarity, we show the
        // primary role extraction; the full implementation iterates.
        let (role, label) = atom.roles.iter().next()
            .map(|(r, l)| (r.clone(), l.clone()))
            .unwrap_or((SemanticRole::Arg0Agent, atom.label.clone()));

        EnrichmentRequest {
            target_composition_id: format!("comp_{}", atom.id),
            role_to_fill: role,
            candidate_node_id: 0, // resolved by EnrichComposition via graph.ensure_node(label)
            candidate_label: label,
            source: EnrichmentSource::PatternInference, // re-extraction is a form of pattern inference
            confidence: atom.confidence,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnrichmentSource {
    PassiveRecall,      // graph already knows the answer
    UserAnswerMerge,    // user provided the answer, merge into existing composition
    PatternInference,   // pattern mining suggests this role filler (Phase 2)
}
```

### ReExtractionRequest

```rust
/// Request to re-extract a frame with enriched graph context.
/// Used when gap detection reveals systematic extraction failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReExtractionRequest {
    pub original_text: String,
    pub original_atom_id: String,
    pub target_composition_id: CompositionId,
    pub graph_context: Vec<(SemanticRole, NodeId, f32)>,  // role → node → confidence from graph
}
```

### RecallAction

```rust
/// Action produced by SelectAcquisition when PassiveRecall finds a candidate.
/// This is the missing bridge from gap detection back to composition repair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecallAction {
    EnrichComposition {
        target_composition_id: CompositionId,
        role_to_fill: SemanticRole,
        candidate_node_id: NodeId,
    },
    ReExtractFrame {
        target_composition_id: CompositionId,
        enriched_context: Vec<(SemanticRole, NodeId, f32)>,
    },
    NoAction,  // gap noted but no graph context available
}
```

### EnrichComposition Transform

```rust
/// EnrichComposition Transform
///
/// Input:  EnrichmentRequest (composition_id + role + candidate node)
/// Output: GraphDelta (updated composition with new member + edge)
///
/// Closes the feedback loop: gap detection → graph recall → composition repair.
pub struct EnrichComposition;

impl Transform for EnrichComposition {
    type Input = EnrichmentRequest;
    type Output = GraphDelta;

    fn id(&self) -> &'static str { "EnrichComposition" }

    fn transform(&self, req: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        // 1. Find the target composition
        // 2. Add new CompositionMember with role and candidate node
        // 3. New member has EpistemicState::Inferred (not Observed — it came from graph)
        // 4. Add SemanticEdge from composition to candidate node
        // 5. Update composition confidence (blend existing + enrichment confidence)
        // 6. Trigger GovernBeliefs re-evaluation on this composition
    }
}
```

### ReExtractFrame Transform

```rust
/// ReExtractFrame Transform
///
/// Input:  ReExtractionRequest (original text + graph context)
/// Output: Option<SemanticAtom> (improved frame, or None if no improvement)
///
/// Re-runs ExtractFrame with graph context as hints.
/// Graph context provides known role-fillers that the blind rule-based
/// extraction missed.
pub struct ReExtractFrame {
    frame_compiler: ExtractFrame,
}

impl Transform for ReExtractFrame {
    type Input = ReExtractionRequest;
    type Output = Option<SemanticAtom>;

    fn id(&self) -> &'static str { "ReExtractFrame" }

    fn transform(&self, req: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        // 1. Run ExtractFrame on original text with graph context as hints
        // 2. Graph context provides: "for role Arg0Agent, node X is likely (conf 0.7)"
        // 3. If rule-based extraction finds nothing but graph context suggests a filler,
        //    use the graph suggestion with EpistemicState::Inferred
        // 4. If the re-extracted frame has more roles filled than the original,
        //    return it (it's better). Otherwise return None.
        // 5. The new atom references the target_composition_id for merge.
    }
}
```

### Closed-Loop Pipeline Diagram

```text
Raw Text
  ↓
┌─────────────────────────────────────────┐
│ Atomizer                                │
│   Tokenize      → Vec<SemanticAtom>     │
│   ExtractFrame  → Option<SemanticAtom>  │
│   ReasonFrame   → Vec<SemanticAtom>     │
└────────────┬────────────────────────────┘
             ↓ Vec<SemanticAtom>
┌─────────────────────────────────────────┐
│ IngestAtoms                             │
│   Create/update nodes for atom labels   │
│   Create Composition per atom           │
│   Add SemanticEdge per role             │
│   Trigger sense induction               │
└────────────┬────────────────────────────┘
             ↓ GraphDelta
┌─────────────────────────────────────────┐
│ GovernBeliefs                           │
│   Assign LifecycleState                 │
│   Assign EpistemicState                 │
│   Apply quarantine rules                │
└────────────┬────────────────────────────┘
             ↓ GovernedDelta
┌─────────────────────────────────────────┐
│ SeedAnchor                              │
│   Compute seed alignment per composition│
│   Adjust confidence via seed scores     │
└────────────┬────────────────────────────┘
             ↓ AnchoredDelta
┌─────────────────────────────────────────┐
│ RSVS Memory Core                        │
│   Nodes + Compositions + SemanticEdges  │
│   Sense profiles + Grounding evidence   │
│   Pattern memory + Convergence state    │
└────────────┬────────────────────────────┘
             ↓ GraphSnapshot
┌─────────────────────────────────────────┐
│ Reasoning Engines                       │
│   Prediction + Situation + Latent Signal│
│   Cross-Pathway + Abduction + Pattern   │
│   Reflection + Convergence              │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ Executive Control (if enabled)          │
│   Mode selection + Budget + Stop cond.  │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ Acquisition (if gap detected)           │
│   Passive Recall → Self Study → Ask User│
└────────────┬────────────────────────────┘
             ↓  ← FEEDBACK LOOP (NEW)
┌─────────────────────────────────────────┐
│ Composition Repair (NEW)                │
│   EnrichComposition → patch roles       │
│   ReExtractFrame    → re-extract better │
│   → GovernBeliefs re-evaluation         │
│   → Confidence update in existing comp  │
└────────────┬────────────────────────────┘
             ↓
         Back to RSVS Memory Core
```

### Enrichment Semantics

The feedback loop introduces new members and edges into existing compositions. The
semantics of these enrichments must be precise to maintain epistemic integrity:

- **Enriched members enter as `EpistemicState::Inferred`** — never auto-promoted to
  `Observed`. The enrichment came from graph recall, pattern inference, or user answer
  merge, not from direct extraction. Even when a user explicitly provides an answer,
  the new member enters the composition as `Inferred` because it was not part of the
  original extraction. Only the user's raw answer atom (if ingested separately) would
  be `Observed`.

- **Enrichment triggers GovernBeliefs re-evaluation** on the target composition. When a
  new member is added, the composition's lifecycle and epistemic state are re-assessed.
  This ensures that enriched compositions don't bypass governance.

- **If enrichment raises confidence above threshold**, the composition can transition
  lifecycle state (e.g., from `Candidate` to `Stable`). This is the mechanism by which
  partial compositions mature into well-grounded ones through the feedback loop.

- **User answer merge** enters as `(Candidate, Observed)` for the new member (the user
  directly observed/stated this information), but the overall composition's epistemic
  state is re-evaluated by GovernBeliefs — it may remain `Inferred` if other members
  are still inferred, or transition to `Grounded` if enough independent evidence
  converges.

- **Extraction quality tracking**: ExtractFrame tracks how often its extractions produce
  gaps (e.g., "for verb 'membuat', Cause role is missing 70% of the time"). This
  extraction quality metric influences future re-extraction decisions — compositions
  from low-quality extraction patterns are prioritized for ReExtractFrame when graph
  context becomes available.

---

## Abstraction 6: Seed-Driven Epistemic Anchoring

v11.0 already has 7 seed primitives working: value, risk, trust, identity, agent, goal, feedback.
These seeds are activated during reasoning and drive meaning pathways.

Instead of creating a separate "source trust weight" system, **use seeds as epistemic anchors**:

```rust
/// Evaluate confidence of a composition using seed alignment.
/// Seeds that are already proven in v11.0 become the foundation for belief evaluation.
pub fn seed_anchored_confidence(
    composition: &Composition,
    graph: &Graph,
    seed_engine: &SeedActivationEngine,
) -> f32 {
    let mut scores = HashMap::new();

    // For each seed primitive, compute how well this composition aligns
    for seed in SeedPrimitive::all() {
        let alignment = seed_engine.alignment(&composition.members, seed, graph);
        scores.insert(seed, alignment);
    }

    // Weighted combination — trust and risk dominate epistemic evaluation
    let trust  = scores.get(&SeedPrimitive::Trust).copied().unwrap_or(0.5);
    let risk   = scores.get(&SeedPrimitive::Risk).copied().unwrap_or(0.5);
    let value  = scores.get(&SeedPrimitive::Value).copied().unwrap_or(0.5);
    let goal   = scores.get(&SeedPrimitive::Goal).copied().unwrap_or(0.5);
    let identity = scores.get(&SeedPrimitive::Identity).copied().unwrap_or(0.5);

    trust * 0.30
    + (1.0 - risk) * 0.25
    + value * 0.20
    + goal * 0.15
    + identity * 0.10
}
```

### Why Seed-Driven Is Better Than Source Trust Weights

```text
Source trust weights (adjusted approach):
  - Separate system from reasoning
  - Static weights per source type
  - No connection to what the graph actually knows
  - External configuration

Seed-driven anchoring (elegant approach):
  - Uses same primitives as reasoning (seeds already drive meaning pathways)
  - Dynamic: alignment changes as graph matures
  - Grounded in graph structure, not configuration
  - A composition that aligns with trust seeds IS more trustworthy
  - A composition that triggers risk seeds IS more risky
  - Natural feedback loop: confidence → seed activation → reasoning → confidence update
```

### Seed Alignment Computation

```rust
impl SeedActivationEngine {
    /// How well does a set of composition members align with a seed?
    pub fn alignment(
        &self,
        members: &[CompositionMember],
        seed: SeedPrimitive,
        graph: &Graph,
    ) -> f32 {
        // 1. Find nodes in the graph that are activated by this seed
        let seed_nodes = self.nodes_for_seed(seed);

        // 2. Compute overlap between composition members and seed-activated nodes
        let mut total_alignment = 0.0;
        for member in members {
            if seed_nodes.contains(&member.node_id) {
                // Direct overlap — strong signal
                total_alignment += 0.5;
            } else {
                // Indirect: check if member node has senses close to seed-activated nodes
                let sense_overlap = graph.sense_overlap(member.node_id, &seed_nodes);
                total_alignment += sense_overlap * 0.3;
            }
        }

        // 3. Normalize by member count
        if members.is_empty() { return 0.5; }
        (total_alignment / members.len() as f32).clamp(0.0, 1.0)
    }
}
```

---

## ProvenanceChain — Lightweight, Reuses EdgeSource

```rust
/// Provenance chain for a composition.
/// Tracks where this knowledge came from.
/// Uses EdgeSource (not a separate ProvenanceSource).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceChain {
    pub origin: EdgeSource,                  // how this composition was created
    pub origin_id: String,                   // event_id, rule_id, etc.
    pub parent_composition_id: Option<CompositionId>,  // if derived from another
    pub timestamp: String,                   // ISO 8601
}
```

No separate `ProvenanceSource` enum. `EdgeSource` already captures where things come from.

---

## Full Architecture — Unified Pipeline

```text
Raw Text
  ↓
┌─────────────────────────────────────────┐
│ Atomizer                                │
│   Tokenize      → Vec<SemanticAtom>     │
│   ExtractFrame  → Option<SemanticAtom>  │
│   ReasonFrame   → Vec<SemanticAtom>     │
└────────────┬────────────────────────────┘
             ↓ Vec<SemanticAtom>
┌─────────────────────────────────────────┐
│ IngestAtoms                             │
│   Create/update nodes for atom labels   │
│   Create Composition per atom           │
│   Add SemanticEdge per role             │
│   Trigger sense induction               │
└────────────┬────────────────────────────┘
             ↓ GraphDelta
┌─────────────────────────────────────────┐
│ GovernBeliefs                           │
│   Assign LifecycleState                 │
│   Assign EpistemicState                 │
│   Apply quarantine rules                │
└────────────┬────────────────────────────┘
             ↓ GovernedDelta
┌─────────────────────────────────────────┐
│ SeedAnchor                              │
│   Compute seed alignment per composition│
│   Adjust confidence via seed scores     │
└────────────┬────────────────────────────┘
             ↓ AnchoredDelta
┌─────────────────────────────────────────┐
│ RSVS Memory Core                        │
│   Nodes + Compositions + SemanticEdges  │
│   Sense profiles + Grounding evidence   │
│   Pattern memory + Convergence state    │
└────────────┬────────────────────────────┘
             ↓ GraphSnapshot
┌─────────────────────────────────────────┐
│ Reasoning Engines                       │
│   Prediction + Situation + Latent Signal│
│   Cross-Pathway + Abduction + Pattern   │
│   Reflection + Convergence              │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ Executive Control (if enabled)          │
│   Mode selection + Budget + Stop cond.  │
└────────────┬────────────────────────────┘
             ↓
┌─────────────────────────────────────────┐
│ Acquisition (if gap detected)           │
│   Passive Recall → Self Study → Ask User│
└────────────┬────────────────────────────┘
             ↓  ← FEEDBACK LOOP (NEW)
┌─────────────────────────────────────────┐
│ Composition Repair (NEW)                │
│   EnrichComposition → patch roles       │
│   ReExtractFrame    → re-extract better │
│   → GovernBeliefs re-evaluation         │
│   → Confidence update in existing comp  │
└────────────┬────────────────────────────┘
             ↓
         Back to RSVS Memory Core
```

---

## HiddenMeaningType — Extended (Unified with AtomVariant)

```rust
#[non_exhaustive]
pub enum HiddenMeaningType {
    // v11.0 meaning-pathway variants (EXISTING)
    AffectiveDisguise,
    SocialConcealment,
    PerformativeMask,
    TraumaPattern,
    PowerDynamic,
    Emergent,

    // v12.0 event-structure variants (NEW)
    ProblemSolutionPattern,
    MotivationInference,
    GoalInference,
    AgentResponsibility,
    CauseEffectPattern,
    ToolUsePattern,
    InefficiencySignal,
    PolarityConflict,
    PurposeConflict,
    RoleAnomaly,
}
```

Used via `AtomVariant::MeaningVariant(HiddenMeaningType)`.

---

## Migration Strategy — From v11.0 to v12.0

### Phase A: Add New Types (Additive, Zero Breaking Changes)

```text
1. Add SemanticAtom, AtomType, AtomVariant, SemanticRole to types.rs
2. Add Composition, CompositionType, CompositionMember to types.rs
3. Add LifecycleState, EpistemicState to types.rs
4. Add SemanticEdge to types.rs
5. Add Transform trait to pipeline/
6. Add ProvenanceChain to types.rs
7. Extend EdgeSource with new variants
8. Extend HiddenMeaningType with new variants

All existing code UNCHANGED. All 1,081 tests green.
```

### Phase B: Implement Transforms (New Code Only)

```text
1. Implement Tokenize transform (wraps existing tokenizer)
2. Implement ExtractFrame transform (MD-1)
3. Implement ReasonFrame transform (MD-2)
4. Implement IngestAtoms transform (wraps existing ingest)
5. Implement GovernBeliefs transform (MD-4)
6. Implement SeedAnchor transform (MD-4)
7. Implement EnrichComposition transform (feedback loop)
8. Implement ReExtractFrame transform (feedback loop)

All existing code UNCHANGED. New transforms tested independently.
```

### Phase C: Bridge Existing → New Types

```text
1. Add conversion: NodeStatus → LifecycleState
2. Add conversion: BeliefState → (LifecycleState, EpistemicState)
3. Add conversion: EventFrame (if created) → SemanticAtom
4. Add conversion: HiddenMeaningCandidate → SemanticAtom
5. Pipeline can emit both old and new types simultaneously

All existing code UNCHANGED. Dual-emission ensures compatibility.
```

### Phase D: Switch Pipeline to Transform Engine

```text
1. PipelineEngine calls transforms instead of hardcoded stages
2. Existing stages become transform implementations
3. Feature flag: transform_pipeline_enabled

When disabled: identical to v11.0 behavior.
When enabled: uses new transform engine.
```

### Phase E: Deprecate Old Types (After Full Test Coverage)

```text
1. Mark NodeStatus as deprecated (→ LifecycleState)
2. Mark BeliefState as deprecated (→ EpistemicState)
3. Remove EventFrame, HiddenMeaningCandidate as separate types
4. Remove ProvenanceSource (→ EdgeSource)
5. Remove CandidateStatus (→ LifecycleState + EpistemicState)

Only after 100% test coverage on new types.
```

---

## Acceptance Criteria

The elegant architecture is accepted if:

1. **6 abstractions defined**: SemanticAtom, Composition, two-axis status, SemanticEdge, Transform, Seed Anchoring
2. **4 overlapping enums → 2 orthogonal axes**: no "Candidate" in 3 places
3. **4 edge systems → 1 typed triple**: no ProvenanceSource separate from EdgeSource
4. **3 grouping types → 1 Composition**: cross-type comparison and convergence enabled
5. **2 ingest paths → 1 unified path**: SemanticAtom handles all richness levels
6. **Pipeline extensible by adding transforms**: no pipeline modifications for new MDs
7. **Seed-driven confidence**: no separate source trust weight system
8. **All 1,081 tests survive migration**: additive changes first, deprecation last
9. **Feature-flagged at every phase**: can revert to v11.0 behavior at any point
10. **Feedback loop closed**: gap detection → graph recall → composition repair → re-governance
11. **Enrichment produces Inferred members**: never auto-promoted to Observed
12. **User answer merge**: answers flow back into existing compositions, not just new atoms

---

## Final Statement

This architecture replaces v11.0's patchwork of overlapping types with 6 unified abstractions.
Every piece of knowledge is a SemanticAtom entering through one path. Every structured group
is a Composition. Every entity has two status axes. Every edge is a typed triple. Every
processing step is a Transform. Every confidence score is seed-anchored.

The feedback loop closes the critical gap: gap detection no longer produces orphaned
knowledge that never repairs the compositions that need it. Passive recall, user answers,
and pattern inference all flow back through EnrichComposition and ReExtractFrame to
repair and improve existing compositions, which are then re-governed before being committed
to the RSVS Memory Core.

The result: **less code, more reasoning power, zero conceptual overlap, closed feedback loops**.
