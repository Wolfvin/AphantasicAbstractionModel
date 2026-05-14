//! Core types for RSVS v10.0 — Emergent Reasoning Engines
//!
//! v10.0 builds on v9.0 Meaning Pathways with 4 Emergent Reasoning Engines:
//! - Compositional Blending: sense(A) + sense(B) → hybrid sense(A∧B)
//! - Abductive Reasoning: X→Y→Z pattern hypothesis from shared activation
//! - Pattern Mining: recurring composition pairs → named pattern nodes
//! - Cross-Pathway Synthesis: P1 gap + P2 conflict → deeper hidden meaning
//!
//! v6.1 builds on v6.0 with:
//! - `TraversalConfig`: Controls recursive composition expansion during queries
//! - `HaltReason`: Why a traversal stopped (stability, confidence, depth, relevance)
//! - `ContextQueryResult`: Scored atoms with P(a|S,q) from depth-controlled traversal
//! - Cycle detection via `HashSet<(NodeId, SenseId)>` during traversal
//! - Freq map per sense for weighted scoring P(a|S,q)
//! - Inactivity TTL for atom expiry
//!
//! v6.0: Every sense is formed by compositions — pairs of (ID, sense_id).
//! Relationships between IDs are structural, derived from shared/differing
//! compositions, not statistical co-occurrence alone.
//!
//! Key concepts:
//! - `CompositionRef`: A reference to a specific sense of a specific node.
//!   This is the fundamental unit of compositional meaning.
//! - `SenseId`: Identifies a sense within a node.
//! - `layer`: Compositional depth — 0 for primitives, N for nodes whose
//!   compositions reach layer N-1.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// -----------------------------------------------------------------------
// v9.0: Meaning Pathway Types
// -----------------------------------------------------------------------

/// A single gap annotation — detected meaning gap from Pathway 1.
/// Stored per-sense on the Node struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnnotation {
    /// What type of gap was detected.
    pub gap_type: GapType,
    /// Confidence of this gap detection (0.0–1.0).
    pub confidence: f32,
    /// The node that was expected but not present.
    pub target_node: NodeId,
    /// Trace back to seed primitives that motivated this gap.
    pub seed_trace: Vec<NodeId>,
}

/// Types of meaning gaps detected by Pathway 1 (Predictive Gap Detection).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GapType {
    /// Scalar implicature: "some" → ¬"all" (weaker item used, stronger unused).
    ScalarImplicature,
    /// Presupposition ungrounded: referenced concept has no grounding.
    PresuppositionUngrounded,
    /// Pragmatic divergence: actual compositions deviate from predicted.
    PragmaticDivergence,
    /// Affective mismatch: spreading from value seed doesn't match.
    AffectiveMismatch,
    /// Social mismatch: spreading from trust/identity seed doesn't match.
    SocialMismatch,
    /// Connotative absent: expected cultural association not present.
    ConnotativeAbsent,
    /// Expected composition: composition predicted by analogy but missing.
    ExpectedComposition,
}

impl Default for GapType {
    fn default() -> Self {
        GapType::ExpectedComposition
    }
}

/// Sense-level meaning profile from Pathway 2 (Affective-Social Seed Activation).
/// Per-sense because different senses of a polysemous node can have
/// very different affective and social profiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenseProfile {
    /// The sense this profile belongs to.
    pub sense_id: SenseId,
    /// Affective profile: valence, arousal, dominance.
    pub affective: AffectiveProfile,
    /// Social profile: distance, trust, power, politeness.
    pub social: SocialProfile,
    /// Connotative profile: cultural activations, connotation direction.
    pub connotative: ConnotativeProfile,
    /// Cross-pathway conflicts — where pathways contradict each other.
    /// This is a signal of hidden meaning (irony, sarcasm, gaslighting).
    #[serde(default)]
    pub conflicts: Vec<PathwayConflict>,
}

/// Affective profile: VAD (Valence, Arousal, Dominance) model.
/// Derived from spreading activation distance to affective seeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectiveProfile {
    /// Valence: how positive/negative (-1.0 to +1.0).
    /// Computed from spreading to "value" seed.
    pub valence: f32,
    /// Arousal: how intense/threatening (0.0 to 1.0).
    /// Computed from spreading to "risk" seed.
    pub arousal: f32,
    /// Dominance: how much control (0.0 to 1.0).
    /// Computed from spreading to "agent" seed.
    pub dominance: f32,
    /// Confidence of this profile (how much evidence supports it).
    pub profile_confidence: f32,
    /// Whether verified by more than one seed pathway.
    pub cross_verified: bool,
}

impl Default for AffectiveProfile {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
            dominance: 0.0,
            profile_confidence: 0.0,
            cross_verified: false,
        }
    }
}

/// Social profile: distance, trust, power.
/// Derived from spreading activation to social seeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialProfile {
    /// Social distance: 0.0 = self, 1.0 = other.
    /// Computed from spreading to "identity" seed.
    pub distance: f32,
    /// Trust level: 0.0 to 1.0.
    /// Computed from spreading to "trust" seed.
    pub trust: f32,
    /// Power direction: +1.0 = speaker dominant, -1.0 = addressee dominant.
    /// Computed from spreading to "agent" seed.
    pub power_direction: f32,
    /// Expected politeness level (Brown & Levinson: W = D + P + R).
    pub expected_politeness: f32,
    /// Confidence of this profile.
    pub profile_confidence: f32,
}

impl Default for SocialProfile {
    fn default() -> Self {
        Self {
            distance: 0.5,
            trust: 0.5,
            power_direction: 0.0,
            expected_politeness: 0.5,
            profile_confidence: 0.0,
        }
    }
}

/// Connotative profile: cultural associations and connotation direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnotativeProfile {
    /// Cultural area activations: cluster_id → activation energy.
    #[serde(default)]
    pub cultural_activations: HashMap<u64, f32>,
    /// Primary connotation direction.
    pub primary_connotation: ConnotationDirection,
    /// Secondary connotations: (activated_node, energy).
    #[serde(default)]
    pub secondary_connotations: Vec<(NodeId, f32)>,
    /// Confidence of this profile.
    pub profile_confidence: f32,
}

impl Default for ConnotativeProfile {
    fn default() -> Self {
        Self {
            cultural_activations: HashMap::new(),
            primary_connotation: ConnotationDirection::Neutral,
            secondary_connotations: Vec::new(),
            profile_confidence: 0.0,
        }
    }
}

/// Connotation direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum ConnotationDirection {
    #[default]
    Neutral,
    Positive,
    Negative,
    /// Positive and negative equally strong — IRONY/AMBIGUITY signal.
    Ambiguous,
    ContextDependent,
}

/// Cross-pathway conflict — hidden meaning signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathwayConflict {
    /// First pathway in conflict.
    pub pathway_a: SeedPathway,
    /// Second pathway in conflict.
    pub pathway_b: SeedPathway,
    /// Type of conflict.
    pub conflict_type: ConflictType,
    /// Strength of conflict (0.0–1.0).
    pub conflict_score: f32,
    /// Description of the structural conflict.
    pub description: StructuralConflictDescription,
}

/// Seed pathway categories.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SeedPathway {
    /// Affective seeds: value, risk.
    Affective,
    /// Social seeds: trust, identity, agent.
    Social,
    /// Pragmatic seeds: goal, feedback, action.
    Pragmatic,
}

/// Types of cross-pathway conflicts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConflictType {
    /// Affective positive but Social threatening → sarcasm.
    AffectiveSocialMismatch,
    /// Affective positive but Pragmatic manipulative → flattery.
    AffectivePragmaticMismatch,
    /// Social equal but Pragmatic dominant → hidden power.
    SocialPragmaticMismatch,
    /// Internal affective: valence positive + arousal high → ambiguity.
    AffectiveInternalConflict,
    /// Connotative contradicts literal meaning → euphemism.
    ConnotativeLiteralMismatch,
}

/// Structural description of a pathway conflict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralConflictDescription {
    /// Seed driving pathway A.
    pub seed_a: NodeId,
    /// Seed driving pathway B.
    pub seed_b: NodeId,
    /// Activation energy from seed A.
    pub activation_a: f32,
    /// Activation energy from seed B.
    pub activation_b: f32,
    /// Expected relation type.
    pub expected_relation: Option<RelationType>,
    /// Actual divergence score.
    pub actual_divergence: f32,
}

/// Discourse metadata for utterance nodes (Pathway 3).
/// Only present for nodes with `semantic.is_utterance = true`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseMeta {
    /// Speech act type (if node is an utterance).
    pub speech_act: Option<SpeechActType>,
    /// Felicity condition status.
    pub felicity: Option<FelicityStatus>,
    /// Centering state (updated per utterance).
    pub centering: Option<CenteringState>,
    /// Rhetorical relation to previous utterance.
    pub prev_relation: Option<(RhetoricalRelation, f32)>,
    /// Extensional referent set.
    pub extension: Option<ExtensionSet>,
}

impl Default for DiscourseMeta {
    fn default() -> Self {
        Self {
            speech_act: None,
            felicity: None,
            centering: None,
            prev_relation: None,
            extension: None,
        }
    }
}

/// Speech act type (Searle's taxonomy).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpeechActType {
    /// Claiming a fact: "Dia marah".
    Assertive,
    /// Requesting something: "Tolong duduk".
    Directive,
    /// Promising: "Aku akan datang".
    Commissive,
    /// Expressing feeling: "Wah!".
    Expressive,
    /// Declaring: "Ku nyatakan kamu suami istri".
    Declaration,
    /// Cannot be determined (insufficient context).
    Undetermined,
}

impl Default for SpeechActType {
    fn default() -> Self {
        SpeechActType::Undetermined
    }
}

/// Felicity condition check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FelicityStatus {
    /// Is propositional content condition met?
    pub propositional_content: bool,
    /// Is preparatory condition met?
    pub preparatory: bool,
    /// Is sincerity condition met?
    pub sincerity: bool,
    /// Is essential condition met?
    pub essential: bool,
    /// Overall: is this utterance felicitous?
    pub is_felicitous: bool,
    /// Details of each condition check.
    #[serde(default)]
    pub check_details: Vec<FelicityCheck>,
}

impl Default for FelicityStatus {
    fn default() -> Self {
        Self {
            propositional_content: true,
            preparatory: true,
            sincerity: true,
            essential: true,
            is_felicitous: true,
            check_details: Vec::new(),
        }
    }
}

/// A single felicity condition check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FelicityCheck {
    /// Name of the condition checked.
    pub condition_name: String,
    /// Whether the condition was found to be met.
    pub found: bool,
    /// Confidence of the check result.
    pub confidence: f32,
}

/// Centering state (Grosz, Joshi, Weinstein).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CenteringState {
    /// Backward-looking center: most salient entity from previous utterance.
    pub cb: Option<NodeId>,
    /// Forward-looking centers: entities that might be focus of next utterance.
    /// Sorted by salience (descending).
    pub cf: Vec<(NodeId, f32)>,
    /// Transition type from previous utterance.
    pub transition: TransitionType,
    /// Coherence score (1.0 = Continue, 0.2 = RoughShift).
    pub coherence: f32,
}

impl Default for CenteringState {
    fn default() -> Self {
        Self {
            cb: None,
            cf: Vec::new(),
            transition: TransitionType::Continue,
            coherence: 1.0,
        }
    }
}

/// Centering transition type (ordered by coherence: Continue > Retain > Smooth > Rough).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum TransitionType {
    /// Cb same, Cb ∈ Cf — most coherent.
    #[default]
    Continue,
    /// Cb same, Cb ∉ Cf — less coherent.
    Retain,
    /// Cb changed, Cb ∈ Cf — okay.
    SmoothShift,
    /// Cb changed, Cb ∉ Cf — least coherent.
    RoughShift,
}

/// Rhetorical relation (RST/SDRT).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RhetoricalRelation {
    // Nucleus-Satellite
    Elaboration,
    Background,
    Cause,
    Result,
    Concession,
    Condition,
    Interpretation,
    Evaluation,
    Evidence,
    Motivation,
    // Multi-nucleus
    Contrast,
    Conjunction,
    Disjunction,
    List,
    Sequence,
    // Unknown
    Unmarked,
}

impl Default for RhetoricalRelation {
    fn default() -> Self {
        RhetoricalRelation::Unmarked
    }
}

/// Extensional referent set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionSet {
    /// Node IDs that are referents of the utterance.
    pub referents: Vec<NodeId>,
    /// Quantifier type (if any).
    pub quantifier: Option<Quantifier>,
    /// Confidence of the extension computation.
    pub confidence: f32,
}

/// Quantifier type for extensional sets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Quantifier {
    Universal,
    Existential,
    Definite,
    Indefinite,
    Generic,
}

// -----------------------------------------------------------------------
// v10.0: Emergent Reasoning Engine Types
// -----------------------------------------------------------------------

/// A hybrid blend result from the Compositional Blending Engine.
///
/// When sense(A) and sense(B) are blended, the result is a new
/// compositional sense that merges shared compositions and marks
/// divergent ones. This is the structural basis for A+B→C emergence.
///
/// Example: sense(dikhianati) + sense(harga_diri) → sense(dikhianati∧harga_diri)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendResult {
    /// The two source nodes that were blended.
    pub source_a: NodeId,
    pub source_b: NodeId,
    /// Sense IDs from each source.
    pub sense_a: SenseId,
    pub sense_b: SenseId,
    /// Shared compositions (present in both A and B).
    pub shared_compositions: Vec<CompositionRef>,
    /// Divergent compositions from A (not in B).
    pub divergent_a: Vec<CompositionRef>,
    /// Divergent compositions from B (not in A).
    pub divergent_b: Vec<CompositionRef>,
    /// The newly created hybrid node ID (if committed to graph).
    pub hybrid_node_id: Option<NodeId>,
    /// Blend quality score (0.0–1.0): higher = more shared structure.
    pub blend_quality: f32,
    /// Emergence potential: how much this blend opens new meaning.
    /// High when divergent compositions from both sides are strong.
    pub emergence_potential: f32,
}

/// An abductive hypothesis from the Abductive Reasoning Engine.
///
/// When X and Y both activate Z, and X has a gap toward Y, the engine
/// hypothesizes that X→Y→Z is a single meaning pattern. This is the
/// mechanism behind discovering "dikhianati → harga_diri → trauma".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbductiveHypothesis {
    /// The starting node (X).
    pub node_x: NodeId,
    /// The intermediate node (Y) — linked by gap from X.
    pub node_y: NodeId,
    /// The convergent node (Z) — activated by both X and Y.
    pub node_z: NodeId,
    /// The gap that links X → Y.
    pub linking_gap: GapAnnotation,
    /// Shared seed activations between X and Z.
    pub shared_seeds_xz: Vec<(NodeId, f32)>,
    /// Shared seed activations between Y and Z.
    pub shared_seeds_yz: Vec<(NodeId, f32)>,
    /// Hypothesis confidence (0.0–1.0).
    pub confidence: f32,
    /// Whether this hypothesis created a pattern node in the graph.
    pub committed: bool,
}

/// A named pattern discovered by the Pattern Mining Engine.
///
/// When composition pairs (e.g., risk+identity) appear in multiple nodes
/// (dikhianati, trauma, sakit), the engine creates a "named pattern" node
/// that abstracts this recurring structure.
///
/// Example: (risk+identity) recurring → pattern = "kekerasan_terhadap_identitas"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedPattern {
    /// The pattern node ID in the graph.
    pub node_id: NodeId,
    /// Human-readable label for this pattern.
    pub label: String,
    /// The seed composition pair that defines this pattern.
    pub seed_composition: (NodeId, NodeId),
    /// Nodes that exhibit this pattern.
    pub exhibiting_nodes: Vec<NodeId>,
    /// How many nodes exhibit this pattern (≥ min_support required).
    pub support_count: usize,
    /// Pattern confidence (0.0–1.0).
    pub confidence: f32,
    /// The composition overlap that defines this pattern.
    pub defining_compositions: Vec<CompositionRef>,
}

/// A synthesis result from the Cross-Pathway Synthesis Engine.
///
/// When Pathway 1 finds a gap AND Pathway 2 finds a conflict on the
/// same node, the engine triggers a deeper search for hidden meaning.
///
/// Example: dikhianati has gap→harga_diri + conflict(AffectiveSocial) →
///   "makna tersembunyi: ini tentang harga diri"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResult {
    /// The node where the synthesis was triggered.
    pub node_id: NodeId,
    /// The sense where both gap and conflict were found.
    pub sense_id: SenseId,
    /// The gap annotation from Pathway 1.
    pub gap: GapAnnotation,
    /// The conflict from Pathway 2.
    pub conflict: PathwayConflict,
    /// The hidden meaning discovered by synthesis.
    pub hidden_meaning: HiddenMeaning,
    /// Synthesis confidence (0.0–1.0).
    pub confidence: f32,
    /// The target node that represents the hidden meaning (if created).
    pub meaning_node_id: Option<NodeId>,
}

/// Hidden meaning discovered by cross-pathway synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenMeaning {
    /// Human-readable description of the hidden meaning.
    pub description: String,
    /// The target node that the hidden meaning points to.
    pub target_node: NodeId,
    /// Seed trace supporting this hidden meaning.
    pub seed_trace: Vec<NodeId>,
    /// Type of hidden meaning discovered.
    pub meaning_type: HiddenMeaningType,
    /// Strength of evidence (0.0–1.0).
    pub evidence_strength: f32,
}

/// Types of hidden meaning that can be discovered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HiddenMeaningType {
    /// The surface meaning masks a deeper affective truth.
    AffectiveDisguise,
    /// A social dynamic is hidden beneath the literal content.
    SocialConcealment,
    /// The utterance is a performative act disguised as something else.
    PerformativeMask,
    /// A trauma pattern underlies the surface expression.
    TraumaPattern,
    /// Power dynamics hidden in the communication.
    PowerDynamic,
    /// General emergent meaning not fitting other categories.
    Emergent,
}

/// A node ID. u32 = 4 bytes vs ~50 bytes for a String.
pub type NodeId = u32;

/// A sense identifier — unique within a node's sense list.
/// Together with a NodeId, uniquely identifies any sense in the system.
pub type SenseId = u32;

/// A set of node IDs — used for similarity/attention.
pub type AtomSet = Vec<NodeId>;

/// A reference to a specific sense of a specific node.
///
/// This is the fundamental unit of compositional meaning in RSVS v6.0.
/// When sense S of node X is composed from [(A, s1), (B, s2), (C, s3)],
/// it means: "X in sense S means what A means in sense s1, AND what B
/// means in sense s2, AND what C means in sense s3."
///
/// Example:
///   raja.sense_1 = [(tahta_tertinggi, 0), (laki_laki, 0), (kerajaan, 0)]
///   ratu.sense_1 = [(tahta_tertinggi, 0), (perempuan, 0), (kerajaan, 0)]
///   They share 2/3 compositions → structural similarity = 0.667
///   Substitution: (laki_laki, 0) → (perempuan, 0) transforms raja → ratu
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct CompositionRef {
    /// The target node ID.
    pub node_id: NodeId,
    /// The target sense index within that node.
    pub sense_id: SenseId,
}

impl CompositionRef {
    /// Create a new composition reference.
    pub fn new(node_id: NodeId, sense_id: SenseId) -> Self {
        Self { node_id, sense_id }
    }
}

/// Node status lifecycle.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum NodeStatus {
    /// Newly created node.
    #[default]
    New,
    /// Node promoted from New, under evaluation.
    Candidate,
    /// Node with high confidence, trusted.
    Stable,
    /// Node that fell below demotion threshold.
    Deprecated,
    /// Node quarantined due to excessive status flips.
    Quarantine,
}

/// Compression state — indicates whether a node has compositional senses.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum CompressionState {
    /// Node has no compositional senses — it is a primitive.
    #[default]
    Raw,
    /// Node has at least one compositional sense — its meaning is
    /// derived from other senses.
    Compressed,
}

/// Tier for node autonomy.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum Tier {
    /// Tier1: Autonomous — high confidence, trusted.
    Tier1,
    /// Tier2: Flagged — revocable, under evaluation.
    Tier2,
    #[default]
    /// Tier3: Blocked — low confidence, needs decision.
    Tier3,
}

/// Source type for edges.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum EdgeSource {
    /// Created during seed bootstrap.
    #[default]
    Bootstrap,
    /// Created through learning from text ingestion.
    Learned,
    /// Created by explicit composition (compose API).
    Composition,
    /// Created by gap detection (P1) — predicted but not observed compositions.
    GapDetection,
    /// Created by discourse tracking (P3) — rhetorical/performative edges.
    Discourse,
    /// v10.0: Created by compositional blending — hybrid A∧B edges.
    Blending,
    /// v10.0: Created by abductive reasoning — hypothetical X→Y→Z edges.
    Abductive,
    /// v10.0: Created by pattern mining — named pattern edges.
    PatternMining,
    /// v10.0: Created by cross-pathway synthesis — hidden meaning edges.
    Synthesis,
    /// v10.1: Created by compound discovery — multi-word expression edges.
    CompoundDiscovery,
}

/// L0-02: Relation type for edges — mirrors Python Layer 0 RelationType.
///
/// Every edge in the graph now carries what kind of semantic relation it
/// represents. This information flows from Layer 0 (perceptual abstractors)
/// through the adapter into Layer 1 (RSVS graph). Default is Categorical
/// for backward compatibility with existing edges that have no relation type.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum RelationType {
    /// "X is a Y" — categorical / taxonomic relation.
    #[default]
    Categorical,
    /// "X is more/less than Y in dimension D" — comparative relation.
    Differential,
    /// "X can do Y" / "X is used for Y" — functional relation.
    Functional,
    /// "X is located at Y" — spatial relation.
    Spatial,
    /// "X occurs before/after Y" — temporal relation.
    Temporal,
    /// "X causes Y" / "X is caused by Y" — causal relation.
    Causal,
    /// Discursive / rhetorical relation between utterances.
    Discursive,
}

/// A language link between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageLink {
    /// Link type (e.g., "same_as").
    pub link_type: String,
    /// Target node ID.
    pub target_id: NodeId,
}

/// Semantic metadata for a node (v6.0 — compositional).
///
/// Key change from earlier versions: `layer` tracks compositional depth.
/// Layer 0 = primitive/seed, Layer N = at least one composition
/// references a layer N-1 sense.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMeta {
    /// Whether the node has compositional senses.
    pub compression_state: CompressionState,
    /// Compositional layer depth (0 = primitive/seed).
    pub layer: u32,
    /// Node IDs this node was derived from (backward compat).
    pub derived_from_node_ids: Vec<NodeId>,
    /// Why the node was compressed (None for raw nodes).
    pub compression_reason: Option<String>,
    /// v8.0: Whether this node is an "internal representation" — a layer 1
    /// node whose compositions reference ONLY layer 0 seed primitives.
    /// Such nodes serve as the bridge between surface tokens (layer 2+)
    /// and epistemological primitives (layer 0). They form the system's
    /// "internal language" — language-agnostic structural meanings that
    /// emerge from co-occurrence patterns with seed primitives.
    /// This field is enforced during ingest: if all composition targets
    /// are layer 0 seeds, the node is tagged `internal_representation = true`
    /// and its layer is forced to 1.
    pub internal_representation: bool,
    /// v9.0: Whether this node represents an utterance (sentence-level)
    /// rather than a token. Utterance nodes are created by Pathway 3
    /// (Discourse Structure Tracking) and have compositions that reference
    /// the token nodes that form the sentence.
    #[serde(default)]
    pub is_utterance: bool,
    /// v9.0: If this is an utterance node, references to its constituent
    /// token NodeIds. Empty for non-utterance nodes.
    #[serde(default)]
    pub utterance_tokens: Vec<NodeId>,
}

impl Default for SemanticMeta {
    fn default() -> Self {
        Self {
            compression_state: CompressionState::Raw,
            layer: 0,
            derived_from_node_ids: Vec::new(),
            compression_reason: None,
            internal_representation: false,
            is_utterance: false,
            utterance_tokens: Vec::new(),
        }
    }
}

/// Policy metadata for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyMeta {
    /// Version of the policy engine that created this metadata.
    pub policy_version: String,
    /// Governance score (0.0–1.0).
    pub governance_score: f32,
    /// Pool of accumulated candidate evidence.
    pub candidate_evidence_pool: f32,
    /// Number of status flip-flops detected.
    pub status_flip_count: u32,
    /// Content fingerprints already seen for dedup.
    pub seen_fingerprints: Vec<String>,
    /// ISO timestamp of last observation.
    pub last_seen_at: Option<String>,
}

impl Default for PolicyMeta {
    fn default() -> Self {
        Self {
            policy_version: "6.0".to_string(),
            governance_score: 0.0,
            candidate_evidence_pool: 0.0,
            status_flip_count: 0,
            seen_fingerprints: Vec::new(),
            last_seen_at: None,
        }
    }
}

/// A node in the RSVS graph (v6.0 — compositional).
///
/// A node represents an ID in the system. It can have multiple senses,
/// each of which is defined by its compositions (references to other
/// nodes' specific senses).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique integer ID.
    pub id: NodeId,
    /// Canonical label (e.g., "raja").
    pub label: String,
    /// v8.1: Display-only surface form (e.g., "raja", "dog").
    /// No language tag — the system is language-agnostic. Language tags
    /// are a presentation concern, not a structural one. Convergence
    /// detection handles cross-language equivalence structurally.
    pub surface_label: String,

    /// Node kind — always "node" in v6.0.
    pub kind: String,
    /// Autonomy tier.
    pub tier: Tier,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// Lifecycle status.
    pub status: NodeStatus,
    /// Whether this is a seed atom.
    pub is_seed: bool,
    /// Whether this node is locked from modification.
    pub is_locked: bool,

    /// Semantic metadata (compression state, layer, derivation).
    pub semantic: SemanticMeta,
    /// Policy metadata (governance, dedup).
    pub policy_meta: Option<PolicyMeta>,
    /// Cross-language links.
    pub language_links: Vec<LanguageLink>,

    /// Atom set for similarity/attention (retained for backward compat).
    pub atoms: AtomSet,

    /// Perceptual grounding fingerprint.
    pub fingerprint: Option<Fingerprint>,

    // === v9.0: Meaning Pathway Data ===

    /// Gap annotations per sense — detected meaning gaps from Pathway 1.
    /// Key = sense_id, Value = gaps found for that sense.
    /// Stored per-sense because different senses of a polysemous node
    /// can have different gaps (e.g., "bank" financial vs river).
    #[serde(default)]
    pub gap_annotations: HashMap<SenseId, Vec<GapAnnotation>>,

    /// Sense profiles per sense — affective/social/connotative from Pathway 2.
    /// Key = sense_id, Value = meaning profile for that sense.
    /// Per-sense because "bank" (financial) and "bank" (river) have
    /// very different affective and social profiles.
    #[serde(default)]
    pub sense_profiles: HashMap<SenseId, SenseProfile>,

    /// Discourse metadata — only present for utterance nodes (Pathway 3).
    /// Contains speech act type, felicity status, centering state,
    /// rhetorical relation, and extensional set.
    #[serde(default)]
    pub discourse_meta: Option<DiscourseMeta>,

    // === v10.0: Emergent Reasoning Engine Data ===

    /// Blend results per sense — from Compositional Blending Engine.
    /// Key = sense_id, Value = blend results for that sense.
    #[serde(default)]
    pub blend_results: HashMap<SenseId, Vec<BlendResult>>,

    /// Abductive hypotheses — from Abductive Reasoning Engine.
    /// Key = node_id of the hypothesis target (Z node).
    #[serde(default)]
    pub abductive_hypotheses: Vec<AbductiveHypothesis>,

    /// Named patterns this node participates in — from Pattern Mining Engine.
    #[serde(default)]
    pub pattern_memberships: Vec<NodeId>,

    /// Synthesis results — from Cross-Pathway Synthesis Engine.
    /// Key = sense_id, Value = synthesis results for that sense.
    #[serde(default)]
    pub synthesis_results: HashMap<SenseId, Vec<SynthesisResult>>,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            id: 0,
            label: String::new(),
            surface_label: String::new(),
            kind: "node".to_string(),
            tier: Tier::Tier3,
            confidence: 0.0,
            status: NodeStatus::New,
            is_seed: false,
            is_locked: false,
            semantic: SemanticMeta::default(),
            policy_meta: None,
            language_links: Vec::new(),
            atoms: Vec::new(),
            fingerprint: None,
            gap_annotations: HashMap::new(),
            sense_profiles: HashMap::new(),
            discourse_meta: None,
            blend_results: HashMap::new(),
            abductive_hypotheses: Vec::new(),
            pattern_memberships: Vec::new(),
            synthesis_results: HashMap::new(),
        }
    }
}

/// Content-addressable fingerprint for perceptual grounding.
///
/// Uses XxHash64 — a fast, deterministic, cross-version-stable hash
/// algorithm suitable for both in-session dedup and persistent
/// content-addressable storage across restarts and Rust compiler versions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fingerprint {
    hash: u64,
    algorithm: String,
}

impl Fingerprint {
    /// Create a new fingerprint from raw byte data.
    ///
    /// Uses `twox_hash::XxHash64` with seed 0 — a fast, deterministic,
    /// cross-version-stable hash algorithm.
    pub fn new(data: &[u8]) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = twox_hash::XxHash64::with_seed(0);
        data.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
            algorithm: "xxhash64".into(),
        }
    }

    /// Returns the hash value of this fingerprint.
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

/// A directed weighted edge: from node -> to node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source node ID.
    pub from: NodeId,
    /// Target node ID.
    pub to: NodeId,
    /// Edge weight (0.0–1.0).
    pub weight: f32,
    /// Whether this edge was created by bootstrap or learned.
    pub source: EdgeSource,
    /// v6.3: Batch number when this edge was last reinforced.
    /// Used for inactivity-based weight decay.
    /// Set to 0 for bootstrap edges (never decays — they're structural).
    #[serde(default)]
    pub last_reinforced_batch: usize,
    /// L0-02: Semantic relation type carried by this edge.
    /// Mirrors Python Layer 0 RelationType enum. Defaults to Categorical
    /// for backward compatibility with edges created before this field existed.
    #[serde(default)]
    pub relation_type: RelationType,
}

// ---------------------------------------------------------------------------
// v6.1: Depth-Controlled Lazy Traversal types
// ---------------------------------------------------------------------------

/// Configuration for depth-controlled lazy traversal (v6.1).
///
/// Controls how the query engine recursively expands `CompositionRef`s
/// during context-aware queries. Traversal stops when any halting
/// criterion is met — stability, confidence, or depth safety net.
///
/// # Halting Criteria
///
/// 1. **Stability**: ||h_{k+1} - h_k|| < gamma — score vector converges
/// 2. **Confidence**: max_score >= halt_confidence — found a strong enough answer
/// 3. **Safety net**: depth >= max_depth — prevent unbounded recursion
/// 4. **Relevance gating**: only expand nodes with similarity >= tau_relevance
///
/// # Example
///
/// ```ignore
/// // Shallow traversal for appraise — just active sense
/// let shallow = TraversalConfig { max_depth: 1, ..Default::default() };
/// // Medium traversal for relate — one hop
/// let medium = TraversalConfig { max_depth: 2, tau_relevance: 0.15, ..Default::default() };
/// // Deep traversal for grounding — full recursive
/// let deep = TraversalConfig { max_depth: 5, gamma: 0.005, ..Default::default() };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalConfig {
    /// Maximum recursion depth before forced stop (safety net).
    pub max_depth: usize,
    /// Stability halting: stop when ||h_{k+1} - h_k|| < gamma.
    pub gamma: f32,
    /// Epsilon for stability check (alias for gamma in some contexts).
    pub halt_epsilon: f32,
    /// Confidence threshold for early halt — stop when max score >= this.
    pub halt_confidence: f32,
    /// Relevance gating: only expand nodes with similarity(node, query_context) >= tau_relevance.
    pub tau_relevance: f32,
    /// v6.3: Minimum information gain per traversal depth.
    /// If IG(k) < epsilon_ig, traversal halts — going deeper adds no useful info.
    /// Set to 0.0 to disable IG halting (rely on stability + confidence only).
    /// Default: 0.01
    pub epsilon_ig: f32,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            gamma: 0.01,
            halt_epsilon: 0.001,
            halt_confidence: 0.90,
            tau_relevance: 0.10,
            epsilon_ig: 0.01,
        }
    }
}

/// Why a traversal halted (v6.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HaltReason {
    /// Reached max_depth safety net.
    MaxDepth,
    /// Stability: ||h_{k+1} - h_k|| < epsilon.
    Stability,
    /// Confidence: max score >= tau_confidence.
    Confidence,
    /// No more compositions to expand (leaf reached).
    LeafReached,
    /// Relevance gating: no children passed tau_relevance.
    RelevanceGate,
    /// v6.3: Information gain too small — traversal adds no useful information.
    InformationGain,
}

/// Result of a context-aware traversal query (v6.1).
///
/// Contains scored atoms with P(a|S,q) weighting, traversal metadata,
/// and cycle detection info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextQueryResult {
    /// The active sense index selected for the queried node.
    pub active_sense_idx: usize,
    /// Total number of senses for the node.
    pub total_senses: usize,
    /// Scored atoms: (label, P(a|S,q) score).
    pub scored_atoms: Vec<(String, f32)>,
    /// Compositional depth reached during traversal.
    pub depth_reached: usize,
    /// Which halting criterion stopped the traversal.
    pub halt_reason: HaltReason,
    /// Number of cycle detections encountered during traversal.
    pub cycles_detected: usize,
    /// Layer of the active sense.
    pub layer: u32,
    /// Grounding score of the active sense.
    pub grounding_score: f32,
}
