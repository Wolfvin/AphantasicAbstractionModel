//! # v1.0.0 Unified Abstraction Type Definitions
//!
//! This module contains ALL type definitions for the AAM architecture
//! as specified in the design documents (MD-1 through MD-6). These types are the
//! FOUNDATION of the architecture — they are the ONLY type system. The old v8.3
//! types (`Node`, `Edge`, `CompositionRef`) are legacy and only kept where still
//! referenced by shared storage.
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
//! - `EdgeSource` — provenance source (extended with new variants)
//! - `HiddenMeaningType` — hidden meaning classification

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// Reuse existing v8.3 types — do NOT redefine them.
use crate::types::{EdgeSource, HiddenMeaningType, NodeId, RelationType};

// ========================================================================
// Named Constants — Audit v6 fix
// ========================================================================

/// Gap rate threshold for a rule to be considered weak (> 30%).
const WEAK_RULE_GAP_RATE_THRESHOLD: f32 = 0.30;
/// Repair rate floor for a rule to be considered weak (< 50%).
const WEAK_RULE_REPAIR_RATE_FLOOR: f32 = 0.50;

// ========================================================================
// v1.0.0 Node — Minimal Graph Node
// ========================================================================

/// Minimal node in the v1.0.0 graph.
///
/// Unlike the v8.3 `crate::types::Node` which carries 25+ fields accumulated
/// over versions 6–11, this struct contains only the fields actually used by
/// the pipeline. All semantic structure is now expressed through
/// `Composition`s and `SemanticEdge`s, not through node fields.
///
/// # Fields actually used by v12
///
/// - `id` — unique identifier
/// - `label` — canonical label (used by `Graph::node_label()`)
/// - `surface_label` — display form (set by `Graph::ensure_node()`)
/// - `lifecycle` — structural maturity (replaces v8.3 `NodeStatus` + `Tier`)
/// - `confidence` — overall confidence score
/// - `senses` — meaning variants (Phase J–P: sense layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique integer ID.
    pub id: NodeId,
    /// Canonical label (e.g., "raja").
    pub label: String,
    /// Display-only surface form (e.g., "raja", "dog").
    pub surface_label: String,
    /// Structural lifecycle state (replaces NodeStatus + Tier).
    pub lifecycle: LifecycleState,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// Sense variants for this node (Phase J–P).
    /// Each node can have multiple senses (meaning variants).
    /// Empty for fresh/unprocessed nodes.
    #[serde(default)]
    pub senses: Vec<Sense>,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            id: 0,
            label: String::new(),
            surface_label: String::new(),
            lifecycle: LifecycleState::New,
            confidence: 0.0,
            senses: Vec::new(),
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
            senses: Vec::new(),
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AtomType {
    /// Simple token extraction (sparse: roles = {}).
    #[default]
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

/// Positive or negative polarity (MD-3 §1).
///
/// Used for event-level polarity detection (e.g., "X did NOT cause Y").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Polarity {
    /// Affirmative / positive.
    #[default]
    Positive,
    /// Negative / negated.
    Negative,
}

/// Active or passive voice (MD-3 §1).
///
/// Used for voice detection in event frames. Important for contradiction resolution:
/// active "X membuat Y" vs passive "Y dibuat oleh X" are the same event.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Voice {
    /// Subject performs the action.
    #[default]
    Active,
    /// Subject receives the action.
    Passive,
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
/// # Active Variants
///
/// - `RuleBased` — Phase 1 deterministic rules (produced by ExtractFrame)
/// - `GraphAssisted` — frame re-extracted with graph context (produced by ReExtractFrame)
///
/// # Removed Variants (Phase 1 cleanup, see ARCHIVED_VARIANTS.md)
///
/// `UdParse`, `SrlLabel`, `AmrCompilation` — removed because they had zero
/// references in the Rust codebase. Reserved for Phase 2/3 parser integration.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum FrameSource {
    /// Phase 1: deterministic rules.
    #[default]
    RuleBased,
    /// Frame re-extracted with graph context (feedback loop).
    GraphAssisted,
}

/// Pattern category classification (MD-3 §1).
///
/// # Active Variants
///
/// - `EventPattern` — recurring event structure (default)
///
/// # Removed Variants (Phase 1 cleanup, see ARCHIVED_VARIANTS.md)
///
/// `CausalChain`, `GoalAction`, `RoleSubstitution`, `TemporalSequence` —
/// removed because they had zero references in the Rust codebase.
/// Never produced by the pipeline.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PatternCategory {
    /// Recurring event structure.
    #[default]
    EventPattern,
}

/// Acquisition source classification (MD-6, MD-3 §1).
///
/// # Active Variants
///
/// - `PassiveRecall` — knowledge recalled from existing graph (default)
/// - `UserAnswer` — answer provided by the user (produced by acquisition)
///
/// # Removed Variants (Phase 1 cleanup, see ARCHIVED_VARIANTS.md)
///
/// `SelfStudy`, `ExternalSource` — removed because they had zero
/// references in the Rust codebase. Reserved for Phase 2 acquisition.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AcquisitionSource {
    /// Knowledge recalled from existing graph.
    #[default]
    PassiveRecall,
    /// Answer provided by the user.
    UserAnswer,
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SemanticRole {
    // === Event frame roles (from MD-1) ===
    /// The predicate/verb of an event frame.
    #[default]
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

    // === Morphological roles ===
    /// Morphological root form of a token.
    ///
    /// When the Tokenize transform stems a token (e.g., "membuat" → "buat"),
    /// the root form is stored in this role so that downstream transforms
    /// can access the morphological lemma. If the token is already a root
    /// form (e.g., "raja"), this role is not set.
    RootForm,

    // === Morphological roles (extended for Morphological Sense Graph) ===
    /// Morphological prefix (awalan) of a derived word.
    MorphPrefix,
    /// Morphological suffix (akhiran) of a derived word.
    MorphSuffix,
    /// Morphological root (akar) in a morphology composition.
    MorphRoot,
    /// Morphological archimorpheme (arkhimorfem) underlying allomorphs.
    MorphArchimorpheme,
    /// Morphological allomorph (alomorf) — surface variant of an archimorpheme.
    MorphAllomorph,
    /// Morphological derived form — the surface word produced by derivation.
    MorphDerivedForm,
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
            id: CompositionId::new(String::new()),
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
    /// Provenance source for this member (which EdgeSource added it).
    #[serde(default)]
    pub source: Option<EdgeSource>,
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
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompositionId(String);

impl CompositionId {
    /// Create a new CompositionId from a String.
    pub fn new(id: String) -> Self {
        Self(id)
    }
    /// Get the inner string as a &str.
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// Consume this CompositionId and return the inner String.
    pub fn into_string(self) -> String {
        self.0
    }
    /// Check if this CompositionId is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Display for CompositionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for CompositionId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for CompositionId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl std::borrow::Borrow<str> for CompositionId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl std::borrow::Borrow<String> for CompositionId {
    fn borrow(&self) -> &String {
        &self.0
    }
}

impl Default for CompositionId {
    fn default() -> Self {
        Self(String::new())
    }
}

// ========================================================================
// Morphological Decomposition — Morphological Sense Graph Support
// ========================================================================

/// Position of an affix relative to the root.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum AffixPosition {
    #[default]
    Prefix,
    Suffix,
}

/// Information about one affix found during morphological decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffixInfo {
    /// Surface form of the affix (e.g., "mem").
    pub surface: String,
    /// The archimorpheme underlying this affix (e.g., "meN"), if any.
    pub archimorpheme: Option<String>,
    /// Position: prefix or suffix.
    pub position: AffixPosition,
}

impl Default for AffixInfo {
    fn default() -> Self {
        Self {
            surface: String::new(),
            archimorpheme: None,
            position: AffixPosition::Prefix,
        }
    }
}

/// Information about nasal assimilation applied during stemming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssimilationInfo {
    /// The archimorpheme that was assimilated (e.g., "meN").
    pub archimorpheme: String,
    /// The allomorph produced (e.g., "mem").
    pub allomorph: String,
    /// The phonological condition that triggered the assimilation.
    pub condition: String,
}

impl Default for AssimilationInfo {
    fn default() -> Self {
        Self {
            archimorpheme: String::new(),
            allomorph: String::new(),
            condition: String::new(),
        }
    }
}

/// Complete morphological decomposition of a word.
///
/// Produced by `GraphAwareStemmer::stem_detailed()` and used to create
/// `CompositionType::Morphology` compositions in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologicalDecomposition {
    /// The original surface form (e.g., "membuat").
    pub surface_form: String,
    /// The root after all affix stripping (e.g., "buat").
    pub root: String,
    /// Prefixes found, with archimorpheme links.
    pub prefixes: Vec<AffixInfo>,
    /// Suffixes found.
    pub suffixes: Vec<AffixInfo>,
    /// Nasal assimilation info, if applicable.
    pub assimilation: Option<AssimilationInfo>,
    /// Whether this word is a reduplication.
    pub is_reduplication: bool,
    /// Confidence score for this decomposition (0.0–1.0).
    pub confidence: f32,
}

impl Default for MorphologicalDecomposition {
    fn default() -> Self {
        Self {
            surface_form: String::new(),
            root: String::new(),
            prefixes: Vec::new(),
            suffixes: Vec::new(),
            assimilation: None,
            is_reduplication: false,
            confidence: 0.0,
        }
    }
}

/// What kind of composition this is (MD-3 §2).
///
/// Replaces the separate EventFrame, HiddenMeaningCandidate, Pattern,
/// AbductiveHypothesis, and SituationState types.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CompositionType {
    /// Was `EventFrame` — semantic event with agent, patient, cause, etc.
    #[default]
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
    /// Morphological decomposition — internal structure of a derived word.
    /// Members: {MorphDerivedForm, MorphPrefix*, MorphRoot, MorphSuffix*, MorphArchimorpheme*, MorphAllomorph*}
    Morphology,
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum LifecycleState {
    /// Just created — no evaluation yet.
    #[default]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EpistemicState {
    /// Directly extracted from input.
    #[default]
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
    /// Uses the extended `EdgeSource` with new variants.
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

    /// Last verbalization result from CompositionalVerbalize transform.
    /// Previously, the CVE transform computed the result but discarded it.
    /// Now stored for downstream access.
    #[serde(default)]
    pub last_verbalization_text: String,

    /// Last activation energy map from SpreadingActivation transform.
    /// Stored as HashMap<NodeId, f32> for downstream transforms
    /// (convergence, gap detection, attention).
    #[serde(default)]
    pub last_activation_energies: HashMap<NodeId, f32>,

    /// Audit v5 fix (DD5): Decay summary from TemporalDecay transform.
    #[serde(default)]
    pub last_decay_demoted: usize,

    /// Number of compositions deprecated by temporal decay.
    #[serde(default)]
    pub last_decay_deprecated: usize,

    /// Audit v5 fix (D14): Per-quality-level extraction tracker from ExtractFrame.
    /// Provides high/moderate/low/failed breakdown alongside the aggregate
    /// `extraction_quality` field.
    #[serde(default)]
    pub extraction_quality_ext: super::extract_frame::ExtractionQualityTrackerExt,

    /// Converged composition pairs from last ConvergenceDetection run.
    #[serde(default)]
    pub last_converged_pairs: Vec<(CompositionId, CompositionId)>,

    /// Number of reflection findings from last Reflective cycle.
    #[serde(default)]
    pub last_reflection_findings_count: usize,
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
        if self.total_extractions == 0 {
            return 0.0;
        }
        self.gaps_detected as f32 / self.total_extractions as f32
    }

    /// Repair rate: how often are the gaps successfully repaired?
    pub fn repair_rate(&self) -> f32 {
        if self.gaps_detected == 0 {
            return 1.0;
        }
        self.gaps_repaired as f32 / self.gaps_detected as f32
    }

    /// Is this rule weak? (gap rate > 30% and repair rate < 50%)
    pub fn is_weak(&self) -> bool {
        self.gap_rate() > WEAK_RULE_GAP_RATE_THRESHOLD && self.repair_rate() < WEAK_RULE_REPAIR_RATE_FLOOR
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
            + confidence)
            / self.frames_extracted as f32;

        let entry = self
            .quality_by_rule
            .entry(rule_id.to_string())
            .or_insert_with(|| ExtractionQuality {
                rule_id: rule_id.to_string(),
                ..Default::default()
            });
        entry.total_extractions += 1;
        entry.avg_confidence = (entry.avg_confidence * (entry.total_extractions - 1) as f32
            + confidence)
            / entry.total_extractions as f32;
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
        self.quality_by_rule
            .values()
            .filter(|q| q.is_weak())
            .collect()
    }
}

/// Placeholder for `KnowledgeGap` in `PipelineContext`.
///
/// Audit v4 fix: This type now carries the same critical fields as `KnowledgeGap`
/// so that downstream transforms (especially `SelectAcquisition`) can use the
/// gap data directly from `PipelineContext::pending_gaps` without re-detecting.
///
/// Previously, `KnowledgeGapPlaceholder` only had `gap_id`, `description`, and
/// `confidence` — dropping `gap_type`, `missing_role`, and `source_composition_id`.
/// This forced `SelectAcquisition` to re-run `DetectGaps` independently, which is
/// wasteful and may produce different results if the graph has changed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGapPlaceholder {
    /// Unique gap identifier.
    pub gap_id: String,
    /// Human-readable description.
    pub description: String,
    /// Confidence of this gap detection (0.0–1.0).
    pub confidence: f32,
    /// Gap type as a string (e.g., "MissingRole", "AmbiguousToken").
    /// Audit v4 fix: previously dropped from KnowledgeGap.
    #[serde(default)]
    pub gap_type: String,
    /// The specific role that's missing (for MissingRole/MissingCause/MissingPurpose gaps).
    /// Stored as a Debug-format string (e.g., "Arg0Agent", "Cause").
    /// Audit v4 fix: previously dropped from KnowledgeGap.
    #[serde(default)]
    pub missing_role: Option<String>,
    /// The composition that has this gap (if applicable).
    /// Audit v4 fix: previously dropped from KnowledgeGap.
    #[serde(default)]
    pub source_composition_id: Option<CompositionId>,
}

impl Default for KnowledgeGapPlaceholder {
    fn default() -> Self {
        Self {
            gap_id: String::new(),
            description: String::new(),
            confidence: 0.0,
            gap_type: String::new(),
            missing_role: None,
            source_composition_id: None,
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SeedPrimitive {
    /// Trust seed — how much social trust does this align with?
    #[default]
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
///
/// # Active Variants
///
/// - `PolarityConflict` — same predicate, opposite polarity
/// - `PurposeConflict` — same predicate, conflicting purpose roles
/// - `SemanticContradiction` — general semantic contradiction (cross-type, default)
/// - `RoleReversal` — Agent ↔ Patient swapped across two compositions
/// - `EquivalenceMismatch` — same structure but different role fillers
///
/// # Removed Variants (Phase 1 cleanup, see ARCHIVED_VARIANTS.md)
///
/// `AgentConflict`, `PatientConflict`, `CauseConflict`, `TemporalConflict`,
/// `LocationConflict` — removed because they had zero references in the
/// Rust codebase. Never produced by detect_contradiction().
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EpistemicConflictType {
    /// Same predicate, opposite polarity (e.g., "X did" vs "X did NOT").
    PolarityConflict,
    /// Same predicate, conflicting purpose roles.
    PurposeConflict,
    /// General semantic contradiction (cross-type).
    #[default]
    SemanticContradiction,
    /// Agent ↔ Patient swapped across two compositions.
    RoleReversal,
    /// Same structure but different role fillers (e.g., different solutions).
    EquivalenceMismatch,
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
            target_composition_id: CompositionId::default(),
            role_to_fill: SemanticRole::Arg0Agent,
            candidate_node_id: 0,
            candidate_label: String::new(),
            source: EnrichmentSource::PassiveRecall,
            confidence: 0.7,
        }
    }
}

/// Source of enrichment (MD-3, MD-6).
///
/// # Active Variants
///
/// - `PassiveRecall` — enrichment from graph-based recall (default)
/// - `UserAnswerMerge` — enrichment from user answer merge
///
/// # Removed Variants (Phase 1 cleanup, see ARCHIVED_VARIANTS.md)
///
/// `ReExtraction`, `HumanAssertion` — removed because they had zero
/// references in the Rust codebase.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EnrichmentSource {
    /// Enrichment from passive recall (graph-based).
    #[default]
    PassiveRecall,
    /// Enrichment from user answer merge.
    UserAnswerMerge,
}

/// Request to re-extract a frame with graph context (MD-3, MD-6).
///
/// Produced by `SelectAcquisition` when `LowGroundingGap` is detected and
/// the graph has grounding evidence. The `ReExtractFrame` transform processes
/// this by re-running extraction with known role-fillers as hints.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

/// Concrete action to take after gap detection (MD-3, MD-6).
///
/// This replaces the old `PassiveRecall` mode which just returned a mode
/// without specifying WHAT to do. Now, `SelectAcquisition` produces concrete
/// actions that the pipeline can execute.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
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
    #[default]
    NoAction,
}

// ========================================================================
// Graph Inspection Types (MD-3 §5, MD-5)
// ========================================================================

/// Graph neighborhood for local mode selection (MD-5).
///
/// Instead of scanning the entire graph for contradictions or low confidence,
/// mode selection evaluates only the neighborhood relevant to current input.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
        let relevant: Vec<Composition> = compositions
            .iter()
            .filter(|c| {
                // A composition is relevant if any member's label matches a keyword
                c.members.iter().any(|m| {
                    let label_lower = m.label.to_lowercase();
                    keywords
                        .iter()
                        .any(|kw| label_lower.contains(&kw.to_lowercase()))
                })
            })
            .cloned()
            .collect();
        Self {
            compositions: relevant,
        }
    }
}

/// Graph snapshot for passing to `DetectGaps` (MD-3 §5).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphSnapshot {
    /// Recent atoms from the current ingest.
    #[serde(default)]
    pub recent_atoms: Vec<SemanticAtom>,
    /// Compositions in the current snapshot.
    #[serde(default)]
    pub compositions: Vec<Composition>,
}

impl GraphSnapshot {
    /// Build graph context for re-extracting a weak frame.
    /// Returns `(SemanticRole, NodeId, f32)` triples from compositions
    /// that share the same predicate, providing known role-fillers
    /// as hints for the rule-based re-extraction.
    pub fn context_for(&self, weak_frame: &WeakFrame) -> Vec<(SemanticRole, NodeId, f32)> {
        let target_comp = self
            .compositions
            .iter()
            .find(|c| c.id == weak_frame.composition_id);

        match target_comp {
            Some(comp) => {
                let predicate = comp
                    .member_with_role(&SemanticRole::Predicate)
                    .map(|m| m.node_id);

                match predicate {
                    Some(pred_id) => self
                        .compositions
                        .iter()
                        .filter(|c| c.id != comp.id)
                        .filter(|c| c.composition_type == CompositionType::Event)
                        .filter(|c| {
                            c.member_with_role(&SemanticRole::Predicate)
                                .map(|m| m.node_id == pred_id)
                                .unwrap_or(false)
                        })
                        .flat_map(|c| {
                            c.members
                                .iter()
                                .filter(|m| m.role != SemanticRole::Predicate)
                                .map(|m| (m.role.clone(), m.node_id, m.confidence))
                                .collect::<Vec<_>>()
                        })
                        .collect(),
                    None => vec![],
                }
            }
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
    pub filled_gaps: Vec<CompositionId>,
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
        let new_evidence = loop_result
            .evidence_count
            .saturating_sub(self.evidence_at_loop_start);
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
            ReasoningGoal::UnderstandInput => self.confidence >= 0.8 && !result.has_gaps,
            ReasoningGoal::ResolveContradiction { composition_id } => {
                result.resolved_contradictions.contains(composition_id)
            }
            ReasoningGoal::FillGap { gap_id } => result.filled_gaps.contains(gap_id),
            ReasoningGoal::AnswerQuestion { .. } => self.confidence >= 0.85,
        }
    }
}

/// What the reasoning session is trying to accomplish (MD-5).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum ReasoningGoal {
    /// Understand the meaning of input text.
    #[default]
    UnderstandInput,
    /// Resolve a specific contradiction.
    ResolveContradiction {
        /// The composition that has the contradiction.
        composition_id: CompositionId,
    },
    /// Fill a specific knowledge gap.
    FillGap {
        /// The gap to fill.
        gap_id: CompositionId,
    },
    /// Answer a user question.
    AnswerQuestion {
        /// The question to answer.
        question: String,
    },
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
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
    #[default]
    Unresolved,
}

// ========================================================================
// Phase J–P: Sense Layer & Cognitive Extensions
// ========================================================================

/// A sense (meaning variant) of a node in the graph.
///
/// Each `Node` can carry multiple senses — different meanings the node
/// can represent depending on context. For example, the node "bank" might
/// have one sense for "financial institution" and another for "river edge".
///
/// # Phase J–P Fields
///
/// | Field | Phase | Purpose |
/// |-------|-------|---------|
/// | `label` | J | Human-readable sense label |
/// | `layer` | J | Abstraction layer (0=primitive, 1=derived, 2=situational, 3=conclusion) |
/// | `coherence` | K | How coherent this sense is with its neighbors (0.0–1.0) |
/// | `freq_map` | L | Weighted frequency per composition (HashMap for weighted Jaccard) |
/// | `composition_evidence` | M | Evidence for/against this sense from compositions |
/// | `is_utterance` | N | Whether this sense represents an utterance-level meaning |
/// | `grounding` | P | Current grounding level of this sense |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sense {
    /// Human-readable label for this sense (e.g., "financial institution").
    pub label: String,

    /// Abstraction layer:
    /// - 0 = primitive (direct token sense)
    /// - 1 = derived (from composition)
    /// - 2 = situational (cross-sentence context)
    /// - 3 = conclusion (abstract inference)
    ///
    ///   Hard cap: layer ≤ 3.
    pub layer: u32,

    /// How coherent this sense is with its neighboring senses (0.0–1.0).
    /// Updated by `compute_member_coherence_penalty()`.
    pub coherence: f32,

    /// Weighted frequency map: how often this sense appears in each composition.
    /// Key = composition ID, Value = confidence-weighted frequency.
    /// Used by weighted Jaccard similarity in LDCS (Phase L).
    #[serde(default)]
    pub freq_map: HashMap<CompositionId, f32>,

    /// Evidence accumulated for/against this sense from compositions.
    /// Updated by `update_sense_evidence()` in Phase M.
    #[serde(default)]
    pub composition_evidence: CompositionEvidence,

    /// Whether this sense represents an utterance-level meaning (Phase N).
    /// Utterance senses are attached to tokens that represent whole sentences
    /// or cross-sentence context.
    #[serde(default)]
    pub is_utterance: bool,

    /// Current grounding level of this sense (Phase P).
    /// Tracks how well-grounded this sense is in evidence.
    pub grounding: SenseGrounding,
}

impl Default for Sense {
    fn default() -> Self {
        Self {
            label: String::new(),
            layer: 0,
            coherence: 0.5,
            freq_map: HashMap::new(),
            composition_evidence: CompositionEvidence::default(),
            is_utterance: false,
            grounding: SenseGrounding::Fragile,
        }
    }
}

impl Sense {
    /// Create a new primitive sense (layer 0).
    pub fn new_primitive(label: &str) -> Self {
        Self {
            label: label.to_string(),
            layer: 0,
            ..Self::default()
        }
    }

    /// Create a new derived sense (layer 1+) from a composition.
    pub fn new_derived(label: &str, layer: u32) -> Self {
        Self {
            label: label.to_string(),
            layer: layer.min(3), // Hard cap at 3
            ..Self::default()
        }
    }

    /// Build the frequency map from graph compositions.
    ///
    /// For each composition that references this sense's node, compute a
    /// confidence-weighted frequency. Uses `comp.confidence` from the graph
    /// (not hardcoded 1.0).
    ///
    /// Fallback: compositions not found in graph get weight 0.5 (safe default
    /// for manually-created or cloned senses).
    pub fn build_freq_map(&mut self, node_id: NodeId, graph: &super::pipeline::Graph) {
        self.freq_map.clear();
        for comp in graph.compositions.values() {
            if comp.members.iter().any(|m| m.node_id == node_id) {
                let weight = comp.confidence;
                *self.freq_map.entry(comp.id.clone()).or_insert(0.0) += weight;
            }
        }
    }

    /// Is this sense at the primitive layer (0)?
    pub fn is_primitive(&self) -> bool {
        self.layer == 0
    }

    /// Is this sense a bridge between layers? (layer 1 = bridge by definition)
    ///
    /// A bridge sense sits at layer 1, connecting primitive (layer 0) seeds
    /// to derived (layer 2+) concepts. This is distinct from `Graph::is_bridge()`
    /// which checks whether a *node* has senses at 2+ different layers.
    pub fn is_bridge(&self) -> bool {
        self.layer == 1
    }

    /// Is this sense derived from compositions? (layer ≥ 1)
    pub fn is_derived(&self) -> bool {
        self.layer >= 1
    }

    /// Is this sense at utterance level? (layer ≥ 2 or is_utterance flag)
    pub fn is_utterance_level(&self) -> bool {
        self.layer >= 2 || self.is_utterance
    }
}

/// Evidence accumulated for/against a sense from compositions (Phase M).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompositionEvidence {
    /// Number of confirming observations.
    pub confirming: u32,
    /// Number of contradicting observations.
    pub contradicting: u32,
    /// IDs of compositions that contributed confirming evidence.
    #[serde(default)]
    pub confirming_sources: Vec<CompositionId>,
    /// IDs of compositions that contributed contradicting evidence.
    #[serde(default)]
    pub contradicting_sources: Vec<CompositionId>,
}

impl CompositionEvidence {
    /// Create empty evidence.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add confirming evidence from a composition.
    pub fn add_confirming(&mut self, composition_id: CompositionId) {
        self.confirming += 1;
        self.confirming_sources.push(composition_id);
    }

    /// Add contradicting evidence from a composition.
    pub fn add_contradicting(&mut self, composition_id: CompositionId) {
        self.contradicting += 1;
        self.contradicting_sources.push(composition_id);
    }

    /// Does this sense have any confirming evidence?
    pub fn has_confirming(&self) -> bool {
        self.confirming > 0
    }

    /// Is contradicting evidence stronger than confirming?
    pub fn is_contradicting_dominant(&self) -> bool {
        self.contradicting > self.confirming
    }
}

/// Grounding level for a sense (Phase P).
///
/// Senses progress through grounding levels as evidence accumulates:
/// Fragile → Tentative → Grounded → Mature
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SenseGrounding {
    /// No solid evidence yet — may be pruned.
    #[default]
    Fragile,
    /// Some confirming evidence but not enough to trust fully.
    Tentative,
    /// Well-grounded in multiple independent sources.
    Grounded,
    /// Long-established, high-coherence, multi-source sense.
    Mature,
}

/// Candidate for sense similarity comparison (Phase L — weighted Jaccard).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SenseCandidate {
    /// The sense being compared.
    pub sense_id: String,
    /// Properties extracted from the sense's composition context.
    /// Key = property name, Value = weight (importance).
    /// Uses HashMap instead of HashSet for weighted Jaccard.
    #[serde(default)]
    pub properties: HashMap<String, f32>,
}

impl SenseCandidate {
    /// Compute weighted Jaccard similarity between two candidates.
    ///
    /// Weighted Jaccard = sum(min(a_i, b_i)) / sum(max(a_i, b_i))
    /// for all keys in the union of both property sets.
    /// Missing keys in one set are treated as 0.0.
    pub fn weighted_jaccard(&self, other: &SenseCandidate) -> f32 {
        let all_keys: HashSet<&String> = self.properties.keys().chain(other.properties.keys()).collect();

        let mut sum_min = 0.0f32;
        let mut sum_max = 0.0f32;

        for key in &all_keys {
            let a = self.properties.get(*key).copied().unwrap_or(0.0);
            let b = other.properties.get(*key).copied().unwrap_or(0.0);
            sum_min += a.min(b);
            sum_max += a.max(b);
        }

        if sum_max == 0.0 {
            return 0.0;
        }

        sum_min / sum_max
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
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada", "adalah",
        "akan", "telah", "sebuah", "seorang", "tidak", "bukan", "juga", "sudah", "oleh", "karena",
        "supaya", "agar", "sebab", // English
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out", "off", "over", "under",
        "again", "further", "then", "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "either", "neither", "each", "every", "all", "any", "few", "more", "most", "other", "some",
        "such", "no", "only", "own", "same", "than", "too", "very",
    ];

    input
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() > 2)
        .filter(|t| !stop_words.contains(&t.as_str()))
        .collect()
}
