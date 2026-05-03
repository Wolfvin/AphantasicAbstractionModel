//! Runtime event stream + snapshot contracts for external subscribers.

use serde::{Deserialize, Serialize};

use crate::types::NodeId;

pub const API_VERSION: &str = "v1";
pub const SCHEMA_VERSION: &str = "v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeNode {
    pub id: NodeId,
    pub label: String,
    pub kind: String,
    pub tier: u8,
    pub confidence: f32,
    pub sense_count: usize,
    pub coherence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEdge {
    pub id: String,
    pub source: NodeId,
    pub target: NodeId,
    pub weight: f32,
    pub source_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub api_version: String,
    pub schema_version: String,
    pub latest_seq: u64,
    pub total_contexts: usize,
    pub nodes: Vec<RuntimeNode>,
    pub edges: Vec<RuntimeEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEvent {
    pub api_version: String,
    pub schema_version: String,
    pub seq: u64,
    pub correlation_id: String,
    pub event_type: String,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventBatch {
    pub api_version: String,
    pub schema_version: String,
    pub latest_seq: u64,
    pub events: Vec<RuntimeEvent>,
}
