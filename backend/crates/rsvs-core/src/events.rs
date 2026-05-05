//! Runtime event stream + snapshot contracts for external subscribers (v6.0)
//!
//! v6.0: Updated schema version. Added layer tracking to RuntimeNode.

use serde::{Deserialize, Serialize};

use crate::types::{CompositionRef, NodeId};

/// API version string for event contracts.
pub const API_VERSION: &str = "v1";
/// Schema version string for event contracts.
pub const SCHEMA_VERSION: &str = "v7.2";

/// Runtime node info (v6.0: compositional with layer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeNode {
    /// Unique integer node ID.
    pub id: NodeId,
    /// Canonical label string.
    pub label: String,
    /// Surface form with language tag (e.g., "raja@id").
    pub surface_label: String,
    /// Node kind — always "node" in v6.0.
    pub kind: String,
    /// Tier number (1, 2, or 3).
    pub tier: u8,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// Lifecycle status as string.
    pub status: String,
    /// Whether this is a seed node.
    pub is_seed: bool,
    /// Whether this node is locked from modification.
    pub is_locked: bool,
    /// Compression state ("raw" or "compressed").
    pub compression_state: String,
    /// Compositional layer depth (0 = primitive/seed).
    pub layer: u32,
    /// Node IDs this node was derived from.
    pub derived_from_node_ids: Vec<NodeId>,
    /// Number of senses for this node.
    pub sense_count: usize,
    /// Coherence of the primary sense (if any).
    pub coherence: Option<f32>,
    /// Grounding score of the primary sense (if any).
    pub grounding_score: Option<f32>,
    /// Compositions of the primary sense (if compositional).
    pub compositions: Vec<CompositionRef>,
}

/// Runtime edge info (v6.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEdge {
    /// Edge ID string (e.g., "1->2").
    pub id: String,
    /// Source node ID.
    pub source: NodeId,
    /// Target node ID.
    pub target: NodeId,
    /// Edge weight (0.0–1.0).
    pub weight: f32,
    /// Edge source type ("bootstrap", "learned", or "composition").
    pub source_type: String,
}

/// Full graph snapshot for external consumers (v6.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    /// API version string.
    pub api_version: String,
    /// Schema version string.
    pub schema_version: String,
    /// Latest event sequence number.
    pub latest_seq: u64,
    /// Total contexts processed.
    pub total_contexts: usize,
    /// All nodes in the snapshot.
    pub nodes: Vec<RuntimeNode>,
    /// All edges in the snapshot.
    pub edges: Vec<RuntimeEdge>,
}

/// A single runtime event in the event stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEvent {
    /// API version string.
    pub api_version: String,
    /// Schema version string.
    pub schema_version: String,
    /// Monotonic sequence number.
    pub seq: u64,
    /// Correlation ID linking related events.
    pub correlation_id: String,
    /// Event type (e.g., "node_created", "confidence_changed").
    pub event_type: String,
    /// Event payload as arbitrary JSON.
    pub payload: serde_json::Value,
}

/// A batch of runtime events returned from a poll.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventBatch {
    /// API version string.
    pub api_version: String,
    /// Schema version string.
    pub schema_version: String,
    /// Latest event sequence number at the time of the poll.
    pub latest_seq: u64,
    /// The events in this batch.
    pub events: Vec<RuntimeEvent>,
}
