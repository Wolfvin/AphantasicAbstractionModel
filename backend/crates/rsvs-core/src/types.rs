//! Core types for RSVS v6.0 — Compositional Architecture
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
}

impl Default for SemanticMeta {
    fn default() -> Self {
        Self {
            compression_state: CompressionState::Raw,
            layer: 0,
            derived_from_node_ids: Vec::new(),
            compression_reason: None,
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
    /// Surface form with language tag (e.g., "raja@id").
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
/// Uses a fixed, well-defined hashing algorithm (SipHash-1-3 via `DefaultHasher`)
/// for content deduplication. Note: `DefaultHasher` is not guaranteed stable
/// across Rust versions — it is suitable for runtime dedup within a session
/// but NOT for persistent content-addressable storage across restarts.
/// For cross-session persistence, prefer an explicit hasher (e.g., xxhash, ahash).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fingerprint {
    hash: u64,
    algorithm: String,
}

impl Fingerprint {
    /// Create a new fingerprint from raw byte data.
    ///
    /// Uses `std::collections::hash_map::DefaultHasher` (SipHash-1-3 variant).
    /// The algorithm field accurately reflects the hasher used.
    /// **Warning**: This hash is NOT stable across Rust compiler versions.
    /// Use only for in-session deduplication, not for persistent fingerprints.
    pub fn new(data: &[u8]) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        data.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
            algorithm: "std_default_hasher".into(),
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
}
