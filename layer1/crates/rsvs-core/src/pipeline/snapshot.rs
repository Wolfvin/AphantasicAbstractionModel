//! Snapshot and event helpers — RSVS v6.0
//!
//! Contains `snapshot_v1()`, `consume_events_v1()`, `latest_seq_v1()`, `status()`.

use super::Rsvs;
use crate::events::{EventBatch, RuntimeEdge, RuntimeNode, RuntimeSnapshot};
use crate::types::{CompressionState, EdgeSource, NodeStatus, RelationType, Tier};

// -----------------------------------------------------------------------
// PipelineStatus — key pipeline metrics
// -----------------------------------------------------------------------

/// Key pipeline metrics at a point in time.
#[derive(Debug)]
pub struct PipelineStatus {
    /// Total number of nodes in the graph.
    pub total_nodes: usize,
    /// Total number of tracked atoms (token-to-ID mappings).
    pub total_atoms: usize,
    /// Total number of contexts processed.
    pub total_contexts: usize,
    /// Whether the autonomy engine has completed warm-up.
    pub warmed_up: bool,
    /// Number of nodes on the removal watchlist.
    pub watchlist_count: usize,
    /// Number of entries in the changelog.
    pub changelog_count: usize,
    /// Current adaptive threshold for sense assignment.
    pub theta_assign: f32,
    /// Current adaptive threshold for sense merging.
    pub theta_merge: f32,
}

impl Rsvs {
    /// Return the latest event sequence number.
    pub fn latest_seq_v1(&self) -> u64 {
        self.latest_seq
    }

    /// Consume events after a given sequence number.
    pub fn consume_events_v1(&self, after_seq: Option<u64>, limit: usize) -> EventBatch {
        let after = after_seq.unwrap_or(0);
        let lim = limit.clamp(1, 5000);
        let events = self
            .events
            .iter()
            .filter(|e| e.seq > after)
            .take(lim)
            .cloned()
            .collect::<Vec<_>>();

        EventBatch {
            api_version: crate::events::API_VERSION.to_string(),
            schema_version: crate::events::SCHEMA_VERSION.to_string(),
            latest_seq: self.latest_seq,
            events,
        }
    }

    /// v6.0 snapshot with compositional architecture.
    ///
    /// Produces a `RuntimeSnapshot` containing all nodes and edges in the graph,
    /// with sense, composition, layer, and grounding metadata.
    pub fn snapshot_v1(&self) -> RuntimeSnapshot {
        let nodes = self
            .graph
            .nodes
            .values()
            .map(|n| {
                let sense = self.senses.get(&n.id);
                let primary_sense = sense.and_then(|s| s.senses.first());

                RuntimeNode {
                    id: n.id,
                    label: n.label.clone(),
                    surface_label: n.surface_label.clone(),
                    kind: n.kind.clone(),
                    tier: match n.tier {
                        Tier::Tier1 => 1,
                        Tier::Tier2 => 2,
                        Tier::Tier3 => 3,
                    },
                    confidence: self.autonomy.confidence(n.id).unwrap_or(n.confidence),
                    status: match self.autonomy.status(n.id).unwrap_or(&n.status) {
                        NodeStatus::New => "new",
                        NodeStatus::Candidate => "candidate",
                        NodeStatus::Stable => "stable",
                        NodeStatus::Deprecated => "deprecated",
                        NodeStatus::Quarantine => "quarantine",
                    }
                    .to_string(),
                    is_seed: n.is_seed,
                    is_locked: n.is_locked,
                    compression_state: match n.semantic.compression_state {
                        CompressionState::Raw => "raw",
                        CompressionState::Compressed => "compressed",
                    }
                    .to_string(),
                    layer: n.semantic.layer,
                    derived_from_node_ids: n.semantic.derived_from_node_ids.clone(),
                    sense_count: sense.map(|s| s.sense_count()).unwrap_or(0),
                    coherence: primary_sense.map(|x| x.coherence),
                    grounding_score: primary_sense.map(|x| x.grounding.score()),
                    compositions: primary_sense
                        .map(|x| x.compositions.clone())
                        .unwrap_or_default(),
                }
            })
            .collect::<Vec<_>>();

        let mut edges = Vec::new();
        for (from, list) in &self.graph.edges {
            for e in list {
                edges.push(RuntimeEdge {
                    id: format!("{}->{}", from, e.to),
                    source: e.from,
                    target: e.to,
                    weight: e.weight,
                    source_type: if e.source == EdgeSource::Bootstrap {
                        "bootstrap".into()
                    } else if e.source == EdgeSource::Composition {
                        "composition".into()
                    } else {
                        "learned".into()
                    },
                    relation_type: format!("{:?}", e.relation_type).to_lowercase(),
                });
            }
        }

        RuntimeSnapshot {
            api_version: crate::events::API_VERSION.to_string(),
            schema_version: crate::events::SCHEMA_VERSION.to_string(),
            latest_seq: self.latest_seq,
            total_contexts: self.total_contexts,
            nodes,
            edges,
        }
    }

    /// Return a status report with key pipeline metrics.
    pub fn status(&self) -> PipelineStatus {
        PipelineStatus {
            total_nodes: self.graph.node_count(),
            total_atoms: self.token_to_id.len(),
            total_contexts: self.total_contexts,
            warmed_up: self.autonomy.is_warmed_up(),
            watchlist_count: self.autonomy.watchlist_len(),
            changelog_count: self.autonomy.changelog_len(),
            theta_assign: self.autonomy.current_theta_assign(),
            theta_merge: self.autonomy.current_theta_merge(),
        }
    }
}
