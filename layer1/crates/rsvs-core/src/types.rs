//! Core types for RSVS v6.1 — Compositional Architecture with Depth-Controlled Traversal
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
}

impl Default for SemanticMeta {
    fn default() -> Self {
        Self {
            compression_state: CompressionState::Raw,
            layer: 0,
            derived_from_node_ids: Vec::new(),
            compression_reason: None,
            internal_representation: false,
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
