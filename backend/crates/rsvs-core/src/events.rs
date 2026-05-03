//! Runtime event stream + snapshot contracts for external subscribers (v4.2)

use serde::{Deserialize, Serialize};

use crate::types::NodeId;

/// API version string for event contracts.
pub const API_VERSION: &str = "v1";
/// Schema version string for event contracts.
pub const SCHEMA_VERSION: &str = "v4.2";

/// Runtime node info (v4.2: unified model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeNode {
    /// Unique integer node ID.
    pub id: NodeId,
    /// Canonical label string.
    pub label: String,
    /// Surface form with language tag (e.g., "stone@en").
    pub surface_label: String,
    /// Node kind — always "node" in v4.2.
    pub kind: String,
    /// Tier number (1, 2, or 3).
    pub tier: u8,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// Lifecycle status as string ("new", "candidate", "stable", "deprecated", "quarantine").
    pub status: String,
    /// Whether this is a seed node.
    pub is_seed: bool,
    /// Whether this node is locked from modification.
    pub is_locked: bool,
    /// Compression state ("raw" or "compressed").
    pub compression_state: String,
    /// Node IDs this node was derived from.
    pub derived_from_node_ids: Vec<NodeId>,
    /// Number of senses for this node.
    pub sense_count: usize,
    /// Coherence of the primary sense (if any).
    pub coherence: Option<f32>,
}

/// Runtime edge info (v4.2).
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
    /// Edge source type ("bootstrap" or "learned").
    pub source_type: String,
}

/// Full graph snapshot for external consumers (v4.2).
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
