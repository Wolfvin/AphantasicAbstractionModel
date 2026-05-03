//! Core types for RSVS v4.2
//!
//! v4.2: Unified node model — no more Atom/Composite distinction.
//! All entities are "nodes". Compression is expressed as metadata.

use serde::{Deserialize, Serialize};

/// An node ID. u32 = 4 bytes vs ~50 bytes for a String.
pub type NodeId = u32;

/// A set of node IDs — used for similarity/attention (retained from v0.5).
pub type AtomSet = Vec<NodeId>;

/// Node status lifecycle (v4.2).
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

/// Compression state (v4.2: replaces Atom/Composite distinction).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum CompressionState {
    /// Node has not been aggregated from other nodes.
    #[default]
    Raw,
    /// Node was created by aggregating other nodes.
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
}

/// A language link between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageLink {
    /// Link type (e.g., "same_as").
    pub link_type: String,
    /// Target node ID.
    pub target_id: NodeId,
}

/// Semantic metadata for a node (v4.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMeta {
    /// Whether the node is raw or compressed.
    pub compression_state: CompressionState,
    /// Node IDs this node was derived from (empty for raw nodes).
    pub derived_from_node_ids: Vec<NodeId>,
    /// Why the node was compressed (None for raw nodes).
    pub compression_reason: Option<String>,
}

impl Default for SemanticMeta {
    fn default() -> Self {
        Self {
            compression_state: CompressionState::Raw,
            derived_from_node_ids: Vec::new(),
            compression_reason: None,
        }
    }
}

/// Policy metadata for a node (v4.2).
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
            policy_version: "4.2".to_string(),
            governance_score: 0.0,
            candidate_evidence_pool: 0.0,
            status_flip_count: 0,
            seen_fingerprints: Vec::new(),
            last_seen_at: None,
        }
    }
}

/// A node in the RSVS graph (v4.2: unified model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique integer ID.
    pub id: NodeId,
    /// Canonical label (e.g., "stone").
    pub label: String,
    /// Surface form with language tag (e.g., "stone@en").
    pub surface_label: String,

    /// Node kind — always "node" in v4.2.
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

    /// Semantic metadata (compression state, derivation).
    pub semantic: SemanticMeta,
    /// Policy metadata (governance, dedup).
    pub policy_meta: Option<PolicyMeta>,
    /// Cross-language links.
    pub language_links: Vec<LanguageLink>,

    /// Atom set for similarity/attention (retained from v0.5).
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fingerprint {
    hash: u64,
    algorithm: String,
}

impl Fingerprint {
    /// Create a new fingerprint from raw byte data using the default hasher.
    pub fn new(data: &[u8]) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        data.hash(&mut hasher);
        Self {
            hash: hasher.finish(),
            algorithm: "siphash".into(),
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
