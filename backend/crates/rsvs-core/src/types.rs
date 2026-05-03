//! Core types for RSVS v4.2
//!
//! v4.2: Unified node model — no more Atom/Composite distinction.
//! All entities are "nodes". Compression is expressed as metadata.

use serde::{Serialize, Deserialize};

/// An node ID. u32 = 4 bytes vs ~50 bytes for a String.
pub type NodeId = u32;

/// A set of node IDs — used for similarity/attention (retained from v0.5).
pub type AtomSet = Vec<NodeId>;

/// Node status lifecycle (v4.2)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeStatus {
    New,
    Candidate,
    Stable,
    Deprecated,
    Quarantine,
}

impl Default for NodeStatus {
    fn default() -> Self {
        NodeStatus::New
    }
}

/// Compression state (v4.2: replaces Atom/Composite distinction)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompressionState {
    Raw,
    Compressed,
}

impl Default for CompressionState {
    fn default() -> Self {
        CompressionState::Raw
    }
}

/// Tier for node autonomy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Tier {
    Tier1,  // Autonomous — high confidence, trusted
    Tier2,  // Flagged — revocable, under evaluation
    Tier3,  // Blocked — low confidence, needs decision
}

impl Default for Tier {
    fn default() -> Self {
        Tier::Tier3
    }
}

/// Source type for edges
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeSource {
    Bootstrap,
    Learned,
}

impl Default for EdgeSource {
    fn default() -> Self {
        EdgeSource::Bootstrap
    }
}

/// A language link between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageLink {
    pub link_type: String,  // "same_as", etc.
    pub target_id: NodeId,
}

/// Semantic metadata (v4.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMeta {
    pub compression_state: CompressionState,
    pub derived_from_node_ids: Vec<NodeId>,
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

/// Policy metadata (v4.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyMeta {
    pub policy_version: String,
    pub governance_score: f32,
    pub candidate_evidence_pool: f32,
    pub status_flip_count: u32,
    pub seen_fingerprints: Vec<String>,
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

/// A node in the RSVS graph (v4.2: unified model)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub label: String,
    pub surface_label: String,  // "stone@en" format

    pub kind: String,  // Always "node" in v4.2
    pub tier: Tier,
    pub confidence: f32,
    pub status: NodeStatus,
    pub is_seed: bool,
    pub is_locked: bool,

    pub semantic: SemanticMeta,
    pub policy_meta: Option<PolicyMeta>,
    pub language_links: Vec<LanguageLink>,

    /// Atom set for similarity/attention (retained from v0.5)
    pub atoms: AtomSet,

    /// Reserved slot for future perceptual grounding.
    pub fingerprint: Option<Fingerprint>,
}

/// Reserved — not implemented yet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fingerprint {
    pub image: Option<Vec<f32>>,
    pub audio: Option<Vec<f32>>,
    pub text: Option<Vec<f32>>,
    pub context: Option<Vec<f32>>,
}

/// A directed weighted edge: from node -> to node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub weight: f32,
    pub source: EdgeSource,
}
