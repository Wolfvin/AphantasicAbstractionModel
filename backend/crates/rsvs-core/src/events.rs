//! Runtime event stream + snapshot contracts for external subscribers (v4.2)

use serde::{Deserialize, Serialize};

use crate::types::NodeId;

/// API version string for event contracts.
pub const API_VERSION: &str = "v1";
/// Schema version string for event contracts.
pub const SCHEMA_VERSION: &str = "v4.2";

/// Runtime node info (v4.2: unified model)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeNode {
    pub id: NodeId,
    pub label: String,
    pub surface_label: String,
    pub kind: String, // Always "node" in v4.2
    pub tier: u8,
    pub confidence: f32,
    pub status: String, // NodeStatus as string
    pub is_seed: bool,
    pub is_locked: bool,
    pub compression_state: String, // "raw" or "compressed"
    pub derived_from_node_ids: Vec<NodeId>,
    pub sense_count: usize,
    pub coherence: Option<f32>,
}

/// Runtime edge info (v4.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEdge {
    pub id: String,
    pub source: NodeId,
    pub target: NodeId,
    pub weight: f32,
    pub source_type: String,
}

/// Full graph snapshot for external consumers (v4.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub api_version: String,
    pub schema_version: String,
    pub latest_seq: u64,
    pub total_contexts: usize,
    pub nodes: Vec<RuntimeNode>,
    pub edges: Vec<RuntimeEdge>,
}

/// A single runtime event in the event stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEvent {
    pub api_version: String,
    pub schema_version: String,
    pub seq: u64,
    pub correlation_id: String,
    pub event_type: String,
    pub payload: serde_json::Value,
}

/// A batch of runtime events returned from a poll.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventBatch {
    pub api_version: String,
    pub schema_version: String,
    pub latest_seq: u64,
    pub events: Vec<RuntimeEvent>,
}
