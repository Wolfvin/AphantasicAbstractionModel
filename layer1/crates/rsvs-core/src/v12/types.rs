//! # v12.0 Unified Abstraction Type Definitions
//!
//! This module contains ALL v12.0 type definitions as specified in the AAM design
//! documents (MD-1 through MD-6). These types are the FOUNDATION of the v12.0
//! architecture — they are the ONLY type system. The old v8.3 types (`Node`, `Edge`,
//! `CompositionRef`) are legacy and only kept where still referenced by shared storage.
//!
//! ## Type Index
//!
//! | Type | MD Source | Abstraction | Purpose |
//! |------|-----------|-------------|---------|
//! | [`SemanticAtom`] | MD-3 §1 | 1 | Universal ingest primitive |
//! | [`AtomType`] | MD-3 §1 | 1 | Classification of atom richness |
//! | [`SemanticRole`] | MD-3 §1 | 1 | Comprehensive role taxonomy |
//! | [`Composition`] | MD-3 §2 | 2 | Universal structured grouping |
//! | [`LifecycleState`] | MD-3 §3 | 3 | Structural maturity axis |
//! | [`EpistemicState`] | MD-3 §3 | 3 | Truth confidence axis |
//! | [`SemanticEdge`] | MD-3 §4 | 4 | Single typed triple |
//! | [`Transform`] | MD-3 §5 | 5 | Declarative processing unit |
//! | [`SeedPrimitive`] | MD-3 §6 | 6 | Seed alignment score keys |
//!
//! ## Existing Types Reused from v8.3
//!
//! The following types are imported from `crate::types` (v8.3) and NOT redefined:
//! - `NodeId` — u32 node identifier
//! - `SenseId` — u32 sense identifier
//! - `RelationType` — semantic relation classification (Categorical, Causal, etc.)
//! - `EdgeSource` — provenance source (extended in v12.0 with new variants)
//! - `HiddenMeaningType` — hidden meaning classification

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// Reuse existing v8.3 types — do NOT redefine them.
use crate::types::{EdgeSource, HiddenMeaningType, NodeId, RelationType};

// ========================================================================
// v12.0 Node — Minimal Graph Node
// ========================================================================

/// Minimal node in the v12.0 graph.
///
/// Unlike the v8.3 `crate::types::Node` which carries 25+ fields accumulated
/// over versions 6–11, this struct contains only the fields actually used by
/// the v12.0 pipeline. All semantic structure is now expressed through
/// `Composition`s and `SemanticEdge`s, not through node fields.
///
/// # Fields actually used by v12
///
/// - `id` — unique identifier
/// - `label` — canonical label (used by `Graph::node_label()`)
/// - `surface_label` — display form (set by `Graph::ensure_node()`)
/// - `lifecycle` — structural maturity (replaces v8.3 `NodeStatus` + `Tier`)
/// - `confidence` — overall confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique integer ID.
    pub id: NodeId,
    /// Canonical label (e.g., "raja").
    pub label: String,
    /// Display-only surface form (e.g., "raja", "dog").
    pub surface_label: String,
    /// Structural lifecycle state (v12.0 replaces NodeStatus + Tier).
    pub lifecycle: LifecycleState,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            id: 0,
            label: String::new(),
            surface_label: String::new(),
            lifecycle: LifecycleState::New,
            confidence: 0.0,
        }
    }
}

impl Node {
    /// Create a new node with the given ID and label.
    pub fn new(id: NodeId, label: &str) -> Self {
        Self {
            id,
            label: label.to_string(),
            surface_label: label.to_string(),
            lifecycle: LifecycleState::New,
            confidence: 0.0,
        }
    }
}

// ========================================================================
// Abstraction 1: SemanticAtom — Unified Ingest Primitive
// ========================================================================

/// Universal ingest primitive (MD-3 §1).
///
/// Every piece of knowledge entering RSVS passes through one type: `SemanticAtom`.
/// A token, an event frame, a hidden meaning candidate — these are all atoms with
/// varying richness. A token is a sparse atom. An event frame is a rich atom.
/// A hidden meaning is a derived atom.
///
/// # Examples
///
/// ```text
/// "raja"
/// → SemanticAtom { atom_type: Token, label: "raja", roles: {} }
///
/// "Raymond membuat aplikasi karena lambat"
/// → SemanticAtom { atom_type: Event, label: "membuat",
///     roles: {Arg0Agent: "Raymond", Arg1Patient: "aplikasi", Cause: "lambat"} }
///
/// [pre-ingest reasoning on above event]
/// → SemanticAtom { atom_type: HiddenMeaning, label: "problem_solution",
///     roles: {Problem: "lambat", Solution: "aplikasi", Arg0Agent: "Raymond"},
///     variant: MeaningVariant(ProblemSolutionPattern) }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAtom {
    /// Unique identifier for this atom (e.g., "atom_42").
    pub id: String,
    /// Human-readable label (e.g., "membuat", "problem_solution").
    pub label: String,
    /// Classification of atom richness: Token (sparse) vs Event (rich).
    pub atom_type: AtomType,

    /// Structural metadata — populated for Event, HiddenMeaning; empty for Token.
    /// Maps semantic role → target label (e.g., Arg0Agent → "Raymond").
    #[serde(default)]
    pub roles: HashMap<SemanticRole, String>,

    /// Positive or negative polarity (None for neutral tokens).
    #[serde(default)]
    pub polarity: Option<Polarity>,

    /// Active or passive voice (None when not applicable).
    #[serde(default)]
    pub voice: Option<Voice>,

    /// Type-specific classification variant.
    /// E.g., MeaningVariant for HiddenMeaning, FrameVariant for Event.
    #[serde(default)]
    pub variant: Option<AtomVariant>,

    /// Confidence score (0.0–1.0) for this atom's extraction quality.
    pub confidence: f32,

    /// Provenance: where this atom came from (unified with edge provenance).
    pub source: EdgeSource,

    /// Set after ingest — links this atom to the Composition it created.
    /// `None` before ingest, `Some(comp_id)` after.
    #[serde(default)]
    pub composition_id: Option<CompositionId>,
}

impl Default for SemanticAtom {
    fn default() -> Self {
        Self {
            id: String::new(),
            label: String::new(),
            atom_type: AtomType::Token,
            roles: HashMap::new(),
            polarity: None,
            voice: None,
            variant: None,
            confidence: 0.0,
            source: EdgeSource::Bootstrap,
            composition_id: None,
        }
    }
}

/// Classification of SemanticAtom richness (MD-3 §1).
///
/// Determines how many roles and how much structural metadata an atom carries.
/// Token is sparse (roles = {}), Event is rich (roles = {Arg0, Arg1, Cause, ...}).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtomType {
    /// Simple token extraction (sparse: roles = {}).
    Token,
    /// Token that requires disambiguation (pronouns, deictics).
    /// Enables gap detection for tokens that need context resolution.
    AmbiguousToken,
    /// Semantic frame extraction (rich: roles = {Arg0, Arg1, Cause, ...}).
    Event,
    /// Pre-ingest reasoning output (rich: roles = {Problem, Solution, ...}).
    HiddenMeaning,
    /// Pattern mining output (structured: roles = {Cause, Action, ...}).
    Pattern,
    /// Abductive/predictive hypothesis.
    Hypothesis,
    /// Externally acquired knowledge (from user, self-study, or recall).
    Acquisition,
}

impl Default for AtomType {
    fn default() -> Self {
        AtomType::Token
    }
}

/// Positive or negative polarity (MD-3 §1).
///
/// Used for event-level polarity detection (e.g., "X did NOT cause Y").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Polarity {
    /// Affirmative / positive.
    Positive,
    /// Negative / negated.
    Negative,
}

impl Default for Polarity {
    fn default() -> Self {
        Polarity::Positive
    }
}

/// Active or passive voice (MD-3 §1).
///
/// Used for voice detection in event frames. Important for contradiction resolution:
/// active "X membuat Y" vs passive "Y dibuat oleh X" are the same event.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Voice {
    /// Subject performs the action.
    Active,
    /// Subject receives the action.
    Passive,
}

impl Default for Voice {
    fn default() -> Self {
        Voice::Active
    }
}

/// Type-specific classification variant (MD-3 §1).
///
/// Each variant is only meaningful for a specific `AtomType`:
/// - `MeaningVariant` → `AtomType::HiddenMeaning`
/// - `FrameVariant` → `AtomType::Event`
/// - `PatternVariant` → `AtomType::Pattern`
/// - `AcquisitionVariant` → `AtomType::Acquisition`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AtomVariant {
    /// For `AtomType::HiddenMeaning` — which kind of hidden meaning.
    MeaningVariant(HiddenMeaningType),
    /// For `AtomType::Event` — how was this frame extracted.
    FrameVariant(FrameSource),
    /// For `AtomType::Pattern` — what pattern category.
    PatternVariant(PatternCategory),
    /// For `AtomType::Acquisition` — how was this acquired.
    AcquisitionVariant(AcquisitionSource),
}

/// How a frame was extracted (MD-1, MD-3 §1).
///
/// Phase 1: RuleBased. Phase 2: UdParse/SrlLabel. Phase 3: AmrCompilation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrameSource {
    /// Phase 1: deterministic rules.
    RuleBased,
    /// Phase 2: dependency parsing.
    UdParse,
    /// Phase 2: semantic role labeling.
    SrlLabel,
    /// Phase 3: AMR graph compilation.
    AmrCompilation,
    /// v12.0: Frame re-extracted with graph context (feedback loop).
    GraphAssisted,
}

impl Default for FrameSource {
    fn default() -> Self {
        FrameSource::RuleBased
    }
}

/// Pattern category classification (MD-3 §1).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternCategory {
    /// Recurring event structure.
    EventPattern,
    /// Causal chain (A → B → C).
    CausalChain,
    /// Goal-action pair.
    GoalAction,
    /// Role substitution pattern.
    RoleSubstitution,
    /// Temporal sequence pattern.
    TemporalSequence,
}

impl Default for PatternCategory {
    fn default() -> Self {
        PatternCategory::EventPattern
    }
}

/// Acquisition source classification (MD-6, MD-3 §1).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcquisitionSource {
    /// Knowledge recalled from existing graph.
    PassiveRecall,
    /// Self-directed study (Phase 2).
    SelfStudy,
    /// Answer provided by the user.
    UserAnswer,
    /// External source (Phase 2).
    ExternalSource,
}

impl Default for AcquisitionSource {
    fn default() -> Self {
        AcquisitionSource::PassiveRecall
    }
}

// ========================================================================
// Abstraction 1b: SemanticRole — Comprehensive Role Taxonomy
// ========================================================================

/// Unified semantic role taxonomy for all atom types (MD-3 §1, MD-3 §1b).
///
/// Event roles, hidden meaning roles, pattern roles — one enum.
/// This replaces the ad-hoc role systems in v11.0 with a single comprehensive taxonomy.
///
/// Roles are grouped by their source:
/// - Event frame roles (from MD-1)
/// - Hidden meaning roles (from MD-2)
/// - Pattern roles
/// - Structural roles
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticRole {
    // === Event frame roles (from MD-1) ===
    /// The predicate/verb of an event frame.
    Predicate,
    /// ARG0: The agent/actor performing the action.
    Arg0Agent,
    /// ARG1: The patient/theme affected by the action.
    Arg1Patient,
    /// ARG2: The recipient/beneficiary of the action.
    Arg2Recipient,
    /// The cause/reason for the event.
    Cause,
    /// The purpose/goal of the event.
    Purpose,
    /// Where the event takes place.
    Location,
    /// When the event takes place.
    Time,
    /// The instrument/tool used in the event.
    Instrument,

    // === Hidden meaning roles (from MD-2) ===
    /// The problem in a problem-solution pattern.
    Problem,
    /// The solution in a problem-solution pattern.
    Solution,
    /// Who benefits from the action.
    Beneficiary,
    /// The tool used in a hidden meaning pattern.
    Tool,
    /// The motivation behind the action.
    Motivation,
    /// The pain point driving the need.
    PainPoint,
    /// The implied goal of the action.
    ImpliedGoal,

    // === Pattern roles ===
    /// What type of pattern this is.
    PatternType,
    /// The antecedent (if-then: the "if" part).
    Antecedent,
    /// The consequent (if-then: the "then" part).
    Consequent,

    // === Structural roles ===
    /// Reference to the producing atom.
    SourceAtom,
    /// Reference to the source event.
    SourceEvent,
    /// Semantic equivalence link.
    EquivalentOf,
}

impl Default for SemanticRole {
    fn default() -> Self {
        SemanticRole::Predicate
    }
}

// ========================================================================
// Abstraction 2: Composition — Universal Structured Grouping
// ========================================================================

/// Universal structured grouping in the RSVS graph (MD-3 §2).
///
/// When a `SemanticAtom` is ingested into the RSVS graph, it becomes a `Composition`:
/// a structured group of nodes with typed roles, lifecycle state, epistemic state,
/// and seed alignment scores.
///
/// This replaces the separate `EventFrame`, `HiddenMeaningCandidate`, `Pattern`,
/// and `AbductiveHypothesis` types with ONE grouping mechanism.
///
/// # Mapping from v11.0
///
/// | Old Type | Becomes | composition_type | Key Roles |
/// |----------|---------|-----------------|-----------|
/// | EventFrame | Composition | Event | Arg0Agent, Arg1Patient, Cause, Purpose |
/// | HiddenMeaningCandidate | Composition | HiddenMeaning | Problem, Solution, Arg0Agent |
/// | Pattern | Composition | Pattern | Antecedent, Consequent, PatternType |
/// | AbductiveHypothesis | Composition | Hypothesis | source nodes + inferred target |
/// | SituationState | Composition | Situation | event refs + environmental context |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Composition {
    /// Unique composition identifier.
    pub id: CompositionId,
    /// What kind of composition this is (Event, HiddenMeaning, Pattern, etc.).
    pub composition_type: CompositionType,

    /// Members: which nodes participate, and in what role.
    #[serde(default)]
    pub members: Vec<CompositionMember>,

    /// Dual-axis status — structural maturity (Abstraction 3).
    pub lifecycle: LifecycleState,
    /// Dual-axis status — epistemic confidence (Abstraction 3).
    pub epistemic: EpistemicState,

    /// Overall confidence score (0.0–1.0).
    pub confidence: f32,

    /// Provenance chain — where this composition came from.
    pub provenance: ProvenanceChain,

    /// Seed alignment scores (Abstraction 6).
    /// Maps each `SeedPrimitive` to its alignment score for this composition.
    #[serde(default)]
    pub seed_scores: HashMap<SeedPrimitive, f32>,

    /// Source text — the original raw text that produced this composition.
    /// Needed by `ReExtractFrame` to re-run extraction with graph context.
    /// `None` for compositions not derived from text (e.g., Pattern, Hypothesis).
    #[serde(default)]
    pub source_text: Option<String>,

    /// How many ingest batches has this composition survived?
    /// Incremented by `GovernBeliefs` at each ingest cycle.
    /// Used by promotion criteria: Candidate → Stable requires age ≥ 3 batches.
    #[serde(default)]
    pub batch_seen: usize,

    /// Which batches had contradictions involving this composition.
    /// Used by `has_recent_contradiction()` for promotion gating.
    #[serde(default)]
    pub contradiction_batches: Vec<usize>,

    /// Current contradiction, if any.
    #[serde(default)]
    pub contradiction: Option<Contradiction>,

    /// ISO 8601 timestamp when this composition was created.
    pub created_at: String,
    /// ISO 8601 timestamp when this composition was last updated.
    pub updated_at: String,
}

impl Default for Composition {
    fn default() -> Self {
        Self {
            id: String::new(),
            composition_type: CompositionType::Event,
            members: Vec::new(),
            lifecycle: LifecycleState::New,
            epistemic: EpistemicState::Observed,
            confidence: 0.0,
            provenance: ProvenanceChain::default(),
            seed_scores: HashMap::new(),
            source_text: None,
            batch_seen: 0,
            contradiction_batches: Vec::new(),
            contradiction: None,
            created_at: String::new(),
            updated_at: String::new(),
        }
    }
}

impl Composition {
    /// How many ingest batches has this composition existed for?
    /// `GovernBeliefs` increments `batch_seen` at the end of each ingest cycle.
    pub fn age_in_batches(&self) -> usize {
        self.batch_seen
    }

    /// Was this composition involved in a contradiction within the last N batches?
    /// Used by `can_promote_to_grounded()` to deny promotion if recent contradictions exist.
    pub fn has_recent_contradiction(&self, last_n: usize) -> bool {
        let threshold = self.batch_seen.saturating_sub(last_n);
        self.contradiction_batches.iter().any(|&b| b > threshold)
    }

    /// Find the member playing a specific semantic role in this composition.
    /// Returns `None` if no member has that role.
    ///
    /// This is the primary accessor for role-based lookup, used by:
    /// - `detect_contradiction()` to compare roles across compositions
    /// - `graph_find_role_candidate()` to find role fillers
    /// - `resolve_ambiguous_from_graph()` to find referents
    /// - `has_equivalence_mismatch()` to compare role fillers
    pub fn member_with_role(&self, role: &SemanticRole) -> Option<&CompositionMember> {
        self.members.iter().find(|m| m.role == *role)
    }

    /// Check if this composition has a member with a specific role.
    /// Returns `true` if any member has the given role.
    pub fn has_member_with_role(&self, role: SemanticRole) -> bool {
        self.members.iter().any(|m| m.role == role)
    }

    /// Check if this composition has a member with a specific role AND
    /// whose node label matches the given predicate string.
    ///
    /// Used by `graph_has_relevant_context()` and `graph_find_role_candidate()`
    /// to find compositions with a specific predicate label.
    pub fn has_member_with_role_and_label(&self, role: SemanticRole, label: &str) -> bool {
        self.members
            .iter()
            .any(|m| m.role == role && m.label() == label)
    }

    /// Get the opposing composition ID from a contradiction.
    ///
    /// After `detect_contradiction()` marks a composition as `Contradicted`,
    /// it stores the opposing composition ID in the `Contradiction` struct.
    /// This method retrieves that ID for use by `check_contradiction_resolution()`.
    pub fn contradiction_opposing_id(&self) -> Option<CompositionId> {
        self.contradiction
            .as_ref()
            .map(|c| c.opposing_composition_id.clone())
    }

    /// Count independent provenance sources contributing to this composition.
    /// A source is "independent" if it has a different `EdgeSource` origin.
    /// Two members from `FrameCompiler` count as 1 source; one from `FrameCompiler`
    /// and one from `EnrichmentFeedback` count as 2 independent sources.
    ///
    /// This is used by `can_promote_to_grounded()` in MD-4: the Inferred → Grounded
    /// transition requires ≥ 2 independent sources.
    pub fn provenance_source_count(&self, member_sources: &[EdgeSource]) -> usize {
        let mut origins: HashSet<EdgeSource> = HashSet::new();
        origins.insert(self.provenance.origin.clone());
        for source in member_sources {
            origins.insert(source.clone());
        }
        origins.len()
    }
}

/// A member of a Composition — a node playing a specific role (MD-3 §2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionMember {
    /// The node participating in this composition.
    pub node_id: NodeId,
    /// The semantic role this node plays in the composition.
    pub role: SemanticRole,
    /// Confidence that this node correctly fills this role (0.0–1.0).
    pub confidence: f32,
    /// Cached label for this member's node (avoids graph lookup for filtering).
    /// Set during ingest from `graph.get_node(self.node_id).label`.
    #[serde(default)]
    pub label: String,
}

impl CompositionMember {
    /// Get the label for this member's node.
    ///
    /// Returns the cached label set during ingest. This avoids a graph lookup
    /// for filtering operations like `has_member_with_role_and_label()`.
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Unique identifier for a Composition (MD-3 §2).
///
/// String-based to allow human-readable IDs like "comp_event_42" or
/// "comp_hm_problem_solution_7".
pub type CompositionId = String;

/// What kind of composition this is (MD-3 §2).
///
/// Replaces the separate EventFrame, HiddenMeaningCandidate, Pattern,
/// AbductiveHypothesis, and SituationState types.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompositionType {
    /// Was `EventFrame` — semantic event with agent, patient, cause, etc.
    Event,
    /// Was `HiddenMeaningCandidate` — pre-ingest reasoning output.
    HiddenMeaning,
    /// Was `Pattern` — recurring structural pattern from mining.
    Pattern,
    /// Was `SituationState` — situational context fragment.
    Situation,
    /// Was `AbductiveHypothesis` — abductive/predictive hypothesis.
    Hypothesis,
    /// Externally acquired knowledge.
    Acquisition,
}

impl Default for CompositionType {
    fn default() -> Self {
        CompositionType::Event
    }
}

// ========================================================================
// Abstraction 3: Two Orthogonal Status Axes
// ========================================================================

/// Axis 1: Structural lifecycle — how mature is this entity in the graph? (MD-3 §3)
///
/// Applies to: Nodes, Compositions, Senses.
/// This replaces `NodeStatus` from v11.0 (values are identical).
///
/// # Transitions
///
/// ```text
/// New → Candidate → Stable → Deprecated
///              ↓
///          Quarantine → Candidate (recovered)
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LifecycleState {
    /// Just created — no evaluation yet.
    New,
    /// Under consideration — not yet promoted.
    Candidate,
    /// Established — high confidence, trusted.
    Stable,
    /// No longer trusted — superseded or refuted.
    Deprecated,
    /// Isolated for review — unresolved conflict or hypothesis.
    Quarantine,
}

impl Default for LifecycleState {
    fn default() -> Self {
        LifecycleState::New
    }
}

/// Axis 2: Epistemic confidence — how confident are we in this knowledge? (MD-3 §3)
///
/// Applies to: Compositions, Knowledge Claims.
/// This replaces `CandidateStatus`, `BeliefState`, `GroundingVerdict` from v11.0.
///
/// # Transitions
///
/// ```text
/// Observed → Inferred → Grounded
/// Hypothesis → Grounded (if confirmed)
/// Any → Contradicted (if opposing evidence)
/// Grounded → Contradicted → Grounded (recovery)
/// ```
///
/// # Semantic Combinations
///
/// ```text
/// (New, Observed)         = fresh direct observation
/// (Candidate, Inferred)   = rule-derived, under review
/// (Stable, Grounded)      = well-established, repeatedly confirmed
/// (Quarantine, Hypothesis)= unconfirmed scenario, isolated
/// (Stable, Contradicted)  = was established, now contradicted
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicState {
    /// Directly extracted from input.
    Observed,
    /// Derived by deterministic rule.
    Inferred,
    /// Unconfirmed scenario reasoning.
    Hypothesis,
    /// Repeatedly supported by independent evidence.
    Grounded,
    /// Opposed by evidence.
    Contradicted,
}

impl Default for EpistemicState {
    fn default() -> Self {
        EpistemicState::Observed
    }
}

// ========================================================================
// Abstraction 4: SemanticEdge — Typed Triple
// ========================================================================

/// Single edge type with three orthogonal dimensions (MD-3 §4).
///
/// Every edge in the graph is a `SemanticEdge`. This replaces the separate
/// `RelationType`, `EdgeSource`, `SemanticRole`, and `ProvenanceSource`
/// systems from v11.0 with a single edge structure.
///
/// # Three Dimensions
///
/// 1. **relation**: WHAT kind of semantic relation (Categorical, Causal, etc.)
/// 2. **role**: OPTIONAL role if this edge is part of a composition
/// 3. **source**: WHERE this edge came from (provenance)
///
/// # Examples
///
/// ```text
/// Token-based edge:
///   SemanticEdge { relation: Categorical, role: None, source: Bootstrap }
///
/// Event composition member edge:
///   SemanticEdge { relation: Categorical, role: Some(Arg0Agent), source: FrameCompiler }
///
/// Hidden meaning composition member edge:
///   SemanticEdge { relation: Causal, role: Some(Problem), source: HiddenMeaningRule }
///
/// Enrichment feedback edge:
///   SemanticEdge { relation: Categorical, role: Some(Arg0Agent), source: EnrichmentFeedback }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEdge {
    /// WHAT kind of semantic relation: Categorical, Causal, Functional, etc.
    pub relation: RelationType,

    /// OPTIONAL: role if this edge is part of a structured composition.
    /// Only populated for edges linking composition members.
    /// Token-based edges have `role = None`.
    #[serde(default)]
    pub role: Option<SemanticRole>,

    /// WHERE this edge came from: provenance.
    /// Uses the extended `EdgeSource` with v12.0 variants.
    pub source: EdgeSource,
}

impl Default for SemanticEdge {
    fn default() -> Self {
        Self {
            relation: RelationType::Categorical,
            role: None,
            source: EdgeSource::Bootstrap,
        }
    }
}

// ========================================================================
// Abstraction 5: Transform — DAG of Declarative Processing
// ========================================================================

/// Declarative processing unit (MD-3 §5).
///
/// Each transform declares what it consumes and produces. The pipeline engine
/// routes data through transforms by type compatibility.
///
/// # Core Transforms
///
/// | Transform | Input | Output | MD Source |
/// |-----------|-------|--------|-----------|
/// | Tokenize | RawText | Vec\<SemanticAtom\> | existing |
/// | ExtractFrame | RawText | Option\<SemanticAtom\> | MD-1 |
/// | ReasonFrame | SemanticAtom | Vec\<SemanticAtom\> | MD-2 |
/// | IngestAtoms | Vec\<SemanticAtom\> | GraphDelta | existing |
/// | GovernBeliefs | GraphDelta | GovernedDelta | MD-4 |
/// | SeedAnchor | GovernedDelta | AnchoredDelta | MD-4 |
/// | EnrichComposition | EnrichmentRequest | GraphDelta | revised |
/// | ReExtractFrame | ReExtractionRequest | Option\<SemanticAtom\> | revised |
pub trait Transform: Send + Sync {
    /// The input type this transform consumes.
    type Input;
    /// The output type this transform produces.
    type Output;

    /// Unique identifier for this transform (e.g., "GovernBeliefs").
    fn id(&self) -> &'static str;

    /// Execute the transform on the given input, using the pipeline context.
    fn transform(&self, input: &Self::Input, ctx: &mut PipelineContext) -> Self::Output;
}

/// Shared state across all transforms in the pipeline (MD-3 §5, Gap 10 fix).
///
/// Each transform reads from and writes to this context. It provides:
/// - Raw input text
/// - Atom accumulation (built up through the pipeline)
/// - Event history for cross-atom reasoning (sliding window)
/// - Graph reference
/// - Extraction quality tracker
/// - Pending enrichment/reextraction requests
/// - Gap detection control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineContext {
    /// Raw input text for the current ingest cycle.
    #[serde(default)]
    pub raw_text: Option<String>,

    /// Atom accumulation — built up through the pipeline.
    #[serde(default)]
    pub current_atoms: Vec<SemanticAtom>,

    /// Event history for cross-atom reasoning (MD-2 ReasonFrame).
    /// Sliding window: max [`PipelineContext::RECENT_EVENTS_WINDOW`] entries.
    #[serde(default)]
    pub recent_events: Vec<SemanticAtom>,

    /// Extraction quality tracker (MD-1 feedback).
    #[serde(default)]
    pub extraction_quality: ExtractionQualityTracker,

    /// Enrichment requests produced by `SelectAcquisition` (feedback loop).
    #[serde(default)]
    pub pending_enrichments: Vec<EnrichmentRequest>,

    /// Re-extraction requests produced by `SelectAcquisition` (feedback loop).
    #[serde(default)]
    pub pending_reextractions: Vec<ReExtractionRequest>,

    /// Whether gap detection is enabled for this pipeline run.
    #[serde(default)]
    pub gap_detection_enabled: bool,

    /// Gaps detected so far in this pipeline run.
    #[serde(default)]
    pub pending_gaps: Vec<KnowledgeGapPlaceholder>,

    /// Atom ID counter — incremented for each new atom created.
    #[serde(default)]
    pub next_atom_id: u64,
}

/// Maximum recent events to keep in the sliding window.
/// Prevents unbounded memory growth while preserving enough context
/// for `PolarityConflictRule` and cross-atom reasoning.
impl PipelineContext {
    /// Maximum number of recent events in the sliding window.
    pub const RECENT_EVENTS_WINDOW: usize = 50;

    /// Add an event atom to `recent_events` with sliding window management.
    pub fn record_event(&mut self, atom: SemanticAtom) {
        if atom.atom_type == AtomType::Event {
            self.recent_events.push(atom);
            if self.recent_events.len() > Self::RECENT_EVENTS_WINDOW {
                self.recent_events.remove(0);
            }
        }
    }

    /// Get recent events for `ReasoningContext` (MD-2).
    pub fn recent_events(&self) -> &Vec<SemanticAtom> {
        &self.recent_events
    }

    /// DAG condition: is the raw input sentence-like?
    pub fn is_sentence_like(&self) -> bool {
        self.raw_text
            .as_ref()
            .map(|t| t.contains(' ') && t.len() > 10)
            .unwrap_or(false)
    }

    /// DAG condition: do we have any event atoms?
    pub fn has_event_atoms(&self) -> bool {
        self.current_atoms
            .iter()
            .any(|a| a.atom_type == AtomType::Event)
    }

    /// DAG condition: is gap detection enabled?
    pub fn gap_detection_enabled(&self) -> bool {
        self.gap_detection_enabled
    }

    /// DAG condition: do we have pending gaps?
    pub fn has_gaps(&self) -> bool {
        !self.pending_gaps.is_empty()
    }

    /// DAG condition: do we have enrichment requests?
    pub fn has_enrichment_requests(&self) -> bool {
        !self.pending_enrichments.is_empty()
    }

    /// DAG condition: do we have re-extraction requests?
    pub fn has_reextraction_requests(&self) -> bool {
        !self.pending_reextractions.is_empty()
    }

    /// Set the raw input text for this pipeline run.
    pub fn set_raw_text(&mut self, text: &str) {
        self.raw_text = Some(text.to_string());
    }

    /// Generate the next atom ID.
    pub fn next_atom_id(&mut self) -> u64 {
        let id = self.next_atom_id;
        self.next_atom_id += 1;
        id
    }
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self {
            raw_text: None,
            current_atoms: Vec::new(),
            recent_events: Vec::new(),
            extraction_quality: ExtractionQualityTracker::default(),
            pending_enrichments: Vec::new(),
            pending_reextractions: Vec::new(),
            gap_detection_enabled: false,
            pending_gaps: Vec::new(),
            next_atom_id: 0,
        }
    }
}

/// Tracks extraction quality per rule/pattern (MD-1 feedback loop).
///
/// Updated by the feedback loop when gap detection reveals extraction failures.
/// When a rule consistently produces frames with missing roles, the system
/// knows that rule is weak for certain patterns, and `ReExtractFrame` can use
/// graph context to compensate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionQuality {
    /// Which rule/pattern this quality record tracks.
    pub rule_id: String,
    /// Total number of extractions by this rule.
    pub total_extractions: usize,
    /// How many of those extractions produced gaps.
    pub gaps_detected: usize,
    /// How many gaps were successfully repaired by feedback loop.
    pub gaps_repaired: usize,
    /// Average confidence of extractions by this rule.
    pub avg_confidence: f32,
    /// The type of the most recent gap detected.
    pub last_gap_type: Option<String>,
}

impl ExtractionQuality {
    /// Gap rate: how often does this rule's extraction produce gaps?
    pub fn gap_rate(&self) -> f32 {
        if self.total_extractions == 0 { return 0.0; }
        self.gaps_detected as f32 / self.total_extractions as f32
    }

    /// Repair rate: how often are the gaps successfully repaired?
    pub fn repair_rate(&self) -> f32 {
        if self.gaps_detected == 0 { return 1.0; }
        self.gaps_repaired as f32 / self.gaps_detected as f32
    }

    /// Is this rule weak? (gap rate > 30% and repair rate < 50%)
    pub fn is_weak(&self) -> bool {
        self.gap_rate() > 0.30 && self.repair_rate() < 0.50
    }
}

impl Default for ExtractionQuality {
    fn default() -> Self {
        Self {
            rule_id: String::new(),
            total_extractions: 0,
            gaps_detected: 0,
            gaps_repaired: 0,
            avg_confidence: 0.0,
            last_gap_type: None,
        }
    }
}

/// Global tracker for extraction quality across all rules (MD-1 feedback).
///
/// Stored in `PipelineContext` and updated by the feedback loop.
/// When `ExtractionQualityTracker` marks a rule as weak (gap rate > 30%,
/// repair rate < 50%), `ReExtractionRequest` can be produced by
/// `SelectAcquisition`, and `ReExtractFrame` will use graph context
/// to compensate for the rule's weaknesses.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtractionQualityTracker {
    /// Quality records indexed by rule ID.
    #[serde(default)]
    pub quality_by_rule: HashMap<String, ExtractionQuality>,
    /// Number of frames successfully extracted (global).
    #[serde(default)]
    pub frames_extracted: usize,
    /// Number of frames that were low-confidence (< 0.5) (global).
    #[serde(default)]
    pub low_confidence_frames: usize,
    /// Average confidence of extracted frames (global).
    #[serde(default)]
    pub average_confidence: f32,
}

impl ExtractionQualityTracker {
    /// Record a successful extraction by a rule.
    pub fn record_extraction(&mut self, rule_id: &str, confidence: f32) {
        self.frames_extracted += 1;
        if confidence < 0.5 {
            self.low_confidence_frames += 1;
        }
        // Running average
        self.average_confidence = (self.average_confidence * (self.frames_extracted - 1) as f32
            + confidence) / self.frames_extracted as f32;

        let entry = self.quality_by_rule.entry(rule_id.to_string())
            .or_insert_with(|| ExtractionQuality {
                rule_id: rule_id.to_string(),
                ..Default::default()
            });
        entry.total_extractions += 1;
        entry.avg_confidence = (entry.avg_confidence * (entry.total_extractions - 1) as f32
            + confidence) / entry.total_extractions as f32;
    }

    /// Record that a gap was detected for a rule's extraction.
    pub fn record_gap(&mut self, rule_id: &str, gap_type: &str) {
        if let Some(entry) = self.quality_by_rule.get_mut(rule_id) {
            entry.gaps_detected += 1;
            entry.last_gap_type = Some(gap_type.to_string());
        }
    }

    /// Record that a gap was repaired for a rule's extraction.
    pub fn record_repair(&mut self, rule_id: &str) {
        if let Some(entry) = self.quality_by_rule.get_mut(rule_id) {
            entry.gaps_repaired += 1;
        }
    }

    /// Get weak rules — candidates for graph-assisted re-extraction.
    pub fn weak_rules(&self) -> Vec<&ExtractionQuality> {
        self.quality_by_rule.values()
            .filter(|q| q.is_weak())
            .collect()
    }
}

/// Placeholder for `KnowledgeGap` until the full MD-6 acquisition module is implemented.
///
/// This type is defined here so that `PipelineContext` can reference it. The full
/// definition (with `KnowledgeGapType`, `GapSource`, etc.) will be in the
/// acquisition module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGapPlaceholder {
    /// Unique gap identifier.
    pub gap_id: String,
    /// Human-readable description.
    pub description: String,
    /// Confidence of this gap detection (0.0–1.0).
    pub confidence: f32,
}

impl Default for KnowledgeGapPlaceholder {
    fn default() -> Self {
        Self {
            gap_id: String::new(),
            description: String::new(),
            confidence: 0.0,
        }
    }
}

// ========================================================================
// Abstraction 6: Seed Anchoring
// ========================================================================

/// Seed primitive types for alignment scoring (MD-3 §6, MD-4).
///
/// These are the 5 epistemological primitives that anchor meaning in RSVS.
/// Every composition carries `seed_scores: HashMap<SeedPrimitive, f32>`,
/// representing its position in meaning space relative to these primitives.
///
/// The seed scores are computed by `SeedAnchor` transform and used for:
/// - Confidence adjustment (seed-anchored confidence)
/// - Promotion criteria (seed alignment must be positive)
/// - Meaning similarity (compositions with similar seed profiles are related)
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SeedPrimitive {
    /// Trust seed — how much social trust does this align with?
    Trust,
    /// Risk seed — how much danger/threat does this relate to?
    Risk,
    /// Value seed — how much positive/negative valence?
    Value,
    /// Goal seed — how much does this relate to goals/purposes?
    Goal,
    /// Identity seed — how much does this relate to self/identity?
    Identity,
}

impl Default for SeedPrimitive {
    fn default() -> Self {
        SeedPrimitive::Trust
    }
}

// ========================================================================
// Supporting Types
// ========================================================================

/// Provenance chain — traces the origin of a composition (MD-3 §2, MD-4).
///
/// Every composition carries its provenance, enabling:
/// - Independent source counting (for `Inferred → Grounded` promotion)
/// - Voice confusion detection (same roles, different provenance)
/// - Extraction quality tracking (which extraction method produced this)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceChain {
    /// The primary source that created this composition.
    pub origin: EdgeSource,
    /// ID of the source extraction or reasoning step.
    #[serde(default)]
    pub origin_id: String,
    /// If this composition was derived from another composition (e.g., enrichment).
    #[serde(default)]
    pub parent_composition_id: Option<CompositionId>,
    /// ISO 8601 timestamp of creation.
    #[serde(default)]
    pub timestamp: String,
}

impl Default for ProvenanceChain {
    fn default() -> Self {
        Self {
            origin: EdgeSource::Bootstrap,
            origin_id: String::new(),
            parent_composition_id: None,
            timestamp: String::new(),
        }
    }
}

/// A recorded contradiction against a composition (MD-4).
///
/// When `GovernBeliefs.detect_contradiction()` finds two compositions
/// with conflicting roles, it creates a `Contradiction` that is stored
/// on the contradicted composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    /// What type of epistemic conflict was detected.
    pub conflict_type: EpistemicConflictType,
    /// The composition that opposes this one.
    pub opposing_composition_id: CompositionId,
    /// How strong the contradiction is (0.0–1.0).
    pub strength: f32,
}

/// Epistemic-level conflict taxonomy (MD-4).
///
/// Separate from v11.0's meaning-pathway `ConflictType`.
/// These conflicts are detected by `GovernBeliefs.detect_contradiction()`
/// by comparing role fillers across compositions with the same predicate.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicConflictType {
    /// Same predicate, opposite polarity (e.g., "X did" vs "X did NOT").
    PolarityConflict,
    /// Same predicate, conflicting purpose roles.
    PurposeConflict,
    /// Same predicate, different agent fillers.
    AgentConflict,
    /// Same predicate, different patient fillers.
    PatientConflict,
    /// Same predicate, different cause fillers.
    CauseConflict,
    /// Temporal incompatibility between compositions.
    TemporalConflict,
    /// Spatial incompatibility between compositions.
    LocationConflict,
    /// General semantic contradiction (cross-type).
    SemanticContradiction,
    /// Agent ↔ Patient swapped across two compositions.
    RoleReversal,
    /// Same structure but different role fillers (e.g., different solutions).
    EquivalenceMismatch,
}

impl Default for EpistemicConflictType {
    fn default() -> Self {
        EpistemicConflictType::SemanticContradiction
    }
}

// ========================================================================
// Delta Types — Pipeline Stage Outputs
// ========================================================================

/// Output of `IngestAtoms` transform (MD-3 §5).
///
/// Contains the new nodes, edges, and compositions created by ingesting
/// `SemanticAtom`s into the graph.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphDelta {
    /// New nodes created by ingest.
    #[serde(default)]
    pub new_nodes: Vec<NodeId>,
    /// New compositions created by ingest.
    #[serde(default)]
    pub new_compositions: Vec<Composition>,
    /// New edges created by ingest.
    #[serde(default)]
    pub new_edges: Vec<SemanticEdge>,
}

impl GraphDelta {
    /// Create an empty `GraphDelta`.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Output of `GovernBeliefs` transform (MD-4).
///
/// Same as `GraphDelta` plus lifecycle/epistemic assignments and transitions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GovernedDelta {
    /// Compositions with their lifecycle/epistemic states assigned.
    #[serde(default)]
    pub compositions: Vec<Composition>,
    /// Governance updates applied during this pass.
    #[serde(default)]
    pub updates: Vec<GovernanceUpdate>,
}

/// Output of `SeedAnchor` transform (MD-4).
///
/// Same as `GovernedDelta` plus seed alignment scores on each composition.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnchoredDelta {
    /// Compositions with seed scores and adjusted confidence.
    #[serde(default)]
    pub compositions: Vec<Composition>,
}

// ========================================================================
// Feedback Loop Types (MD-3, MD-6)
// ========================================================================

/// Request to enrich an existing composition with a new member (MD-3, MD-6).
///
/// Produced by `SelectAcquisition` when `PassiveRecall` finds a candidate
/// node in the graph to fill a missing role. The `EnrichComposition` transform
/// processes this request by adding the candidate as a new member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentRequest {
    /// The composition that needs a new member.
    pub target_composition_id: CompositionId,
    /// The role that needs to be filled.
    pub role_to_fill: SemanticRole,
    /// The node to add as the new member.
    pub candidate_node_id: NodeId,
    /// Human-readable label of the candidate node.
    #[serde(default)]
    pub candidate_label: String,
    /// Where the enrichment came from.
    #[serde(default)]
    pub source: EnrichmentSource,
    /// Confidence in this enrichment (0.0–1.0).
    pub confidence: f32,
}

impl Default for EnrichmentRequest {
    fn default() -> Self {
        Self {
            target_composition_id: String::new(),
            role_to_fill: SemanticRole::Arg0Agent,
            candidate_node_id: 0,
            candidate_label: String::new(),
            source: EnrichmentSource::PassiveRecall,
            confidence: 0.7,
        }
    }
}

/// Source of enrichment (MD-3, MD-6).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnrichmentSource {
    /// Enrichment from passive recall (graph-based).
    PassiveRecall,
    /// Enrichment from user answer merge.
    UserAnswerMerge,
    /// Enrichment from re-extraction with graph context.
    ReExtraction,
    /// Enrichment from human assertion.
    HumanAssertion,
}

impl Default for EnrichmentSource {
    fn default() -> Self {
        EnrichmentSource::PassiveRecall
    }
}

/// Request to re-extract a frame with graph context (MD-3, MD-6).
///
/// Produced by `SelectAcquisition` when `LowGroundingGap` is detected and
/// the graph has grounding evidence. The `ReExtractFrame` transform processes
/// this by re-running extraction with known role-fillers as hints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReExtractionRequest {
    /// The original text to re-extract from.
    pub original_text: String,
    /// ID of the original atom that produced the weak frame.
    #[serde(default)]
    pub original_atom_id: String,
    /// The composition that needs re-extraction.
    pub target_composition_id: CompositionId,
    /// Graph context: (role, node_id, confidence) triples from related compositions.
    #[serde(default)]
    pub graph_context: Vec<(SemanticRole, NodeId, f32)>,
}

impl Default for ReExtractionRequest {
    fn default() -> Self {
        Self {
            original_text: String::new(),
            original_atom_id: String::new(),
            target_composition_id: String::new(),
            graph_context: Vec::new(),
        }
    }
}

/// Concrete action to take after gap detection (MD-3, MD-6).
///
/// This replaces the old `PassiveRecall` mode which just returned a mode
/// without specifying WHAT to do. Now, `SelectAcquisition` produces concrete
/// actions that the pipeline can execute.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecallAction {
    /// Enrich an existing composition with a candidate from the graph.
    EnrichComposition {
        /// The composition to enrich.
        target_composition_id: CompositionId,
        /// The role to fill.
        role_to_fill: SemanticRole,
        /// The candidate node to add.
        candidate_node_id: NodeId,
    },
    /// Re-extract a weak frame with graph context.
    ReExtractFrame {
        /// The composition to re-extract.
        target_composition_id: CompositionId,
        /// Known role-fillers from the graph as hints.
        enriched_context: Vec<(SemanticRole, NodeId, f32)>,
    },
    /// Ask the user for clarification.
    AskUser {
        /// The question to ask.
        question: String,
    },
    /// No action — gap noted but deferred.
    NoAction,
}

impl Default for RecallAction {
    fn default() -> Self {
        RecallAction::NoAction
    }
}

// ========================================================================
// Graph Inspection Types (MD-3 §5, MD-5)
// ========================================================================

/// Graph neighborhood for local mode selection (MD-5).
///
/// Instead of scanning the entire graph for contradictions or low confidence,
/// mode selection evaluates only the neighborhood relevant to current input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNeighborhood {
    /// Compositions in this neighborhood.
    #[serde(default)]
    pub compositions: Vec<Composition>,
}

impl GraphNeighborhood {
    /// Are there any contradicted compositions in this neighborhood?
    pub fn has_contradictions(&self) -> bool {
        self.compositions
            .iter()
            .any(|c| c.epistemic == EpistemicState::Contradicted)
    }

    /// Average confidence across compositions in this neighborhood.
    pub fn average_confidence(&self) -> f32 {
        if self.compositions.is_empty() {
            return 0.0;
        }
        self.compositions.iter().map(|c| c.confidence).sum::<f32>() / self.compositions.len() as f32
    }

    /// Build a graph neighborhood for the given keywords (MD-5).
    ///
    /// Instead of scanning the entire graph for contradictions or low confidence,
    /// mode selection evaluates only the neighborhood relevant to current input.
    /// This finds compositions whose members' labels match any of the keywords.
    pub fn neighborhood_for(keywords: &[String], compositions: &[Composition]) -> Self {
        let relevant: Vec<Composition> = compositions.iter()
            .filter(|c| {
                // A composition is relevant if any member's label matches a keyword
                c.members.iter().any(|m| {
                    let label_lower = m.label.to_lowercase();
                    keywords.iter().any(|kw| label_lower.contains(&kw.to_lowercase()))
                })
            })
            .cloned()
            .collect();
        Self { compositions: relevant }
    }
}

impl Default for GraphNeighborhood {
    fn default() -> Self {
        Self {
            compositions: Vec::new(),
        }
    }
}

/// Graph snapshot for passing to `DetectGaps` (MD-3 §5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    /// Recent atoms from the current ingest.
    #[serde(default)]
    pub recent_atoms: Vec<SemanticAtom>,
    /// Compositions in the current snapshot.
    #[serde(default)]
    pub compositions: Vec<Composition>,
}

impl Default for GraphSnapshot {
    fn default() -> Self {
        Self {
            recent_atoms: Vec::new(),
            compositions: Vec::new(),
        }
    }
}

impl GraphSnapshot {
    /// Build graph context for re-extracting a weak frame.
    /// Returns `(SemanticRole, NodeId, f32)` triples from compositions
    /// that share the same predicate, providing known role-fillers
    /// as hints for the rule-based re-extraction.
    pub fn context_for(&self, weak_frame: &WeakFrame) -> Vec<(SemanticRole, NodeId, f32)> {
        let target_comp = self.compositions.iter()
            .find(|c| c.id == weak_frame.composition_id);

        match target_comp {
            Some(comp) => {
                let predicate = comp.member_with_role(&SemanticRole::Predicate)
                    .map(|m| m.node_id);

                match predicate {
                    Some(pred_id) => {
                        self.compositions.iter()
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

/// A weak frame identified for re-extraction (MD-3 §5, MD-5).
///
/// Contains the info needed to construct a `ReExtractionRequest`.
/// A "weak frame" is a composition with low confidence AND missing expected roles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakFrame {
    /// The composition that needs re-extraction.
    pub composition_id: CompositionId,
    /// The original atom that produced this frame.
    #[serde(default)]
    pub atom_id: String,
    /// The source text (if available) for re-extraction.
    #[serde(default)]
    pub source_text: Option<String>,
}

impl WeakFrame {
    /// Get the composition ID.
    pub fn composition_id(&self) -> &CompositionId {
        &self.composition_id
    }

    /// Get the atom ID.
    pub fn atom_id(&self) -> &str {
        &self.atom_id
    }

    /// Get the source text.
    pub fn source_text(&self) -> Option<&str> {
        self.source_text.as_deref()
    }
}

// ========================================================================
// Executive Types (MD-5)
// ========================================================================

/// Result of a single reflection loop (MD-5).
///
/// Produced by the pipeline after running `GovernBeliefs` + `SeedAnchor`
/// on the reflection delta. Used by `ReasoningState.update()` to track
/// evidence accumulation and goal satisfaction across iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionLoopResult {
    /// Current confidence after this loop.
    pub current_confidence: f32,
    /// Time spent in this loop (milliseconds).
    pub elapsed_ms: u64,
    /// Number of evidence items accumulated so far.
    pub evidence_count: usize,
    /// Compositions modified in this loop.
    #[serde(default)]
    pub modified_compositions: Vec<CompositionId>,
    /// Whether gaps remain after this loop.
    #[serde(default)]
    pub has_gaps: bool,
    /// Contradictions resolved in this loop.
    #[serde(default)]
    pub resolved_contradictions: Vec<CompositionId>,
    /// Gaps filled in this loop.
    #[serde(default)]
    pub filled_gaps: Vec<String>,
}

impl Default for ReflectionLoopResult {
    fn default() -> Self {
        Self {
            current_confidence: 0.0,
            elapsed_ms: 0,
            evidence_count: 0,
            modified_compositions: Vec::new(),
            has_gaps: false,
            resolved_contradictions: Vec::new(),
            filled_gaps: Vec::new(),
        }
    }
}

/// Full state of a reasoning session (MD-5).
///
/// Used by `StopCondition` to decide when to halt, and by `Reflect` to review.
/// Tracks confidence, evidence, and goal satisfaction across reflection loops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningState {
    /// Current confidence of the reasoning result.
    pub confidence: f32,
    /// Elapsed time since reasoning started (milliseconds).
    pub elapsed_ms: u64,
    /// Number of reflection loops completed.
    #[serde(default)]
    pub loops_completed: usize,
    /// Number of loops since the last new piece of evidence was found.
    #[serde(default)]
    pub loops_without_new_evidence: usize,
    /// Whether the reasoning goal has been met.
    #[serde(default)]
    pub goal_met: bool,
    /// The reasoning goal (what are we trying to determine?).
    pub goal: ReasoningGoal,
    /// Compositions modified during this reasoning session.
    #[serde(default)]
    pub modified_compositions: Vec<CompositionId>,
    /// Evidence accumulated during this reasoning session.
    #[serde(default)]
    pub evidence_count: usize,
    /// Evidence count at the start of the current loop.
    #[serde(default)]
    pub evidence_at_loop_start: usize,
}

impl ReasoningState {
    /// Create initial state for a reasoning session.
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
    pub fn update(&mut self, loop_result: &ReflectionLoopResult) {
        self.loops_completed += 1;
        self.confidence = loop_result.current_confidence;
        self.elapsed_ms += loop_result.elapsed_ms;

        // Track whether new evidence was found in this loop.
        let new_evidence = loop_result.evidence_count.saturating_sub(self.evidence_at_loop_start);
        if new_evidence > 0 {
            self.loops_without_new_evidence = 0;
            self.evidence_count = loop_result.evidence_count;
            self.evidence_at_loop_start = loop_result.evidence_count;
        } else {
            self.loops_without_new_evidence += 1;
        }

        // Track modified compositions.
        self.modified_compositions
            .extend(loop_result.modified_compositions.iter().cloned());

        // Check if goal is met.
        self.goal_met = self.check_goal_met(loop_result);
    }

    /// Determine if the reasoning goal has been satisfied.
    fn check_goal_met(&self, result: &ReflectionLoopResult) -> bool {
        match &self.goal {
            ReasoningGoal::UnderstandInput => {
                self.confidence >= 0.8 && !result.has_gaps
            }
            ReasoningGoal::ResolveContradiction { composition_id } => {
                result.resolved_contradictions.contains(composition_id)
            }
            ReasoningGoal::FillGap { gap_id } => {
                result.filled_gaps.contains(gap_id)
            }
            ReasoningGoal::AnswerQuestion { .. } => {
                self.confidence >= 0.85
            }
        }
    }
}

/// What the reasoning session is trying to accomplish (MD-5).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReasoningGoal {
    /// Understand the meaning of input text.
    UnderstandInput,
    /// Resolve a specific contradiction.
    ResolveContradiction {
        /// The composition that has the contradiction.
        composition_id: CompositionId,
    },
    /// Fill a specific knowledge gap.
    FillGap {
        /// The gap to fill.
        gap_id: String,
    },
    /// Answer a user question.
    AnswerQuestion {
        /// The question to answer.
        question: String,
    },
}

impl Default for ReasoningGoal {
    fn default() -> Self {
        ReasoningGoal::UnderstandInput
    }
}

// ========================================================================
// Governance Types (MD-4)
// ========================================================================

/// Result of re-governing a composition after enrichment (MD-4).
///
/// Contains the transitions to apply after `GovernBeliefs.re_govern_composition()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceUpdate {
    /// The composition being updated.
    pub composition_id: CompositionId,
    /// New lifecycle state (if changed).
    #[serde(default)]
    pub new_lifecycle: Option<LifecycleState>,
    /// New epistemic state (if changed).
    #[serde(default)]
    pub new_epistemic: Option<EpistemicState>,
    /// Contradiction detected during re-governance (if any).
    #[serde(default)]
    pub contradiction: Option<Contradiction>,
    /// Confidence adjustment (if any).
    #[serde(default)]
    pub confidence_adjustment: Option<f32>,
}

impl GovernanceUpdate {
    /// Create a new governance update for the given composition.
    pub fn new(id: CompositionId) -> Self {
        Self {
            composition_id: id,
            new_lifecycle: None,
            new_epistemic: None,
            contradiction: None,
            confidence_adjustment: None,
        }
    }

    /// Set the new lifecycle state.
    pub fn set_lifecycle(&mut self, state: LifecycleState) {
        self.new_lifecycle = Some(state);
    }

    /// Set the new epistemic state.
    pub fn set_epistemic(&mut self, state: EpistemicState) {
        self.new_epistemic = Some(state);
    }

    /// Mark this composition as contradicted.
    pub fn mark_contradicted(&mut self, contradiction: Contradiction) {
        self.contradiction = Some(contradiction);
        self.new_epistemic = Some(EpistemicState::Contradicted);
    }
}

/// Verdict from a promotion eligibility check (MD-4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PromotionVerdict {
    /// The promotion is approved.
    Approved,
    /// The promotion is denied, with a reason.
    Denied(String),
}

/// Result of seed anchoring evaluation (MD-4).
///
/// Contains both the computed seed confidence AND how strongly to apply it.
/// Critical fix: when no alignment data exists, weight = 0.0, meaning the
/// original confidence is preserved without adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedAdjustment {
    /// The seed-anchored confidence score.
    pub seed_confidence: f32,
    /// How strongly to apply this adjustment (0.0 = no adjustment, 0.4 = strong).
    pub weight: f32,
    /// How much actual alignment data drove this adjustment.
    pub alignment_strength: f32,
}

impl Default for SeedAdjustment {
    fn default() -> Self {
        Self {
            seed_confidence: 0.5,
            weight: 0.0,
            alignment_strength: 0.0,
        }
    }
}

/// Contradiction resolution status (MD-4).
///
/// Tracks whether a contradiction has been resolved and how.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionResolution {
    /// Unique identifier for this resolution record.
    #[serde(default)]
    pub contradiction_id: String,
    /// The opposing composition in this contradiction.
    pub opposing_composition_id: CompositionId,
    /// How the contradiction was resolved.
    pub resolution_type: ResolutionType,
    /// Whether the resolution is complete.
    pub resolved: bool,
}

/// How a contradiction was resolved (MD-4).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionType {
    /// One side was superseded by newer, stronger evidence.
    Superseded,
    /// The contradiction was based on incomplete context; new context resolves it.
    ContextResolved,
    /// Both sides are true in different contexts (scoped validity).
    ScopedValidity,
    /// One side was a misinterpretation (e.g., passive vs active voice).
    Misinterpretation,
    /// Resolution not yet possible.
    Unresolved,
}

impl Default for ResolutionType {
    fn default() -> Self {
        ResolutionType::Unresolved
    }
}

// ========================================================================
// Utility Functions
// ========================================================================

/// Extract keywords from input text for neighborhood-based mode selection (MD-5).
///
/// This is a simple keyword extraction heuristic for the executive cognition layer.
/// It tokenizes the input and filters out stop words, returning the remaining
/// tokens as keywords. Used by `Graph::neighborhood_for()` to find compositions
/// relevant to the current input.
pub fn extract_keywords(input: &str) -> Vec<String> {
    let stop_words = [
        // Indonesian
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
        "pada", "adalah", "akan", "telah", "sebuah", "seorang", "tidak", "bukan",
        "juga", "sudah", "oleh", "karena", "supaya", "agar", "sebab",
        // English
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some",
        "such", "no", "only", "own", "same", "than", "too", "very",
    ];

    input
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() > 2)
        .filter(|t| !stop_words.contains(&t.as_str()))
        .collect()
}
