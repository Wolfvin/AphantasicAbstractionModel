//! Seed graph bootstrap (v6.0)
//!
//! Loads seed atoms into the graph at startup. By default, uses 24
//! epistemological seeds, but callers can provide custom seed labels
//! for domain-specific bootstrapping.
//!
//! These nodes have confidence=1.0, Tier=Tier1, status=Stable,
//! is_seed=true, is_locked=true, and cannot be removed.
//! All nodes above the seed layer emerge from data.

use crate::error::RsvsError;
use crate::graph::RsvsGraph;
use crate::types::{CompressionState, Node, NodeId, NodeStatus, SemanticMeta, Tier};
use std::collections::HashMap;

/// Seed node definitions: label only (v6.0 — all seeds are equal, no layer distinction)
const SEED_ATOMS: &[&str] = &[
    "exists",
    "entity",
    "relation",
    "state",
    "change",
    "time",
    "space",
    "cause",
    "effect",
    "context",
    "signal",
    "pattern",
    "memory",
    "attention",
    "value",
    "agent",
    "goal",
    "risk",
    "trust",
    "identity",
    "language",
    "meaning",
    "action",
    "feedback",
];

/// Bootstrap the graph with seed nodes (v6.0 format).
///
/// If `custom_seeds` is provided, those labels are used instead of the
/// default 24 epistemological seeds. Returns a map of label → NodeId for
/// external reference.
///
/// # Errors
///
/// Returns `RsvsError::SeedInvariant` if the number of successfully seeded
/// nodes does not match the expected count.
pub fn bootstrap(
    graph: &mut RsvsGraph,
    custom_seeds: Option<&[String]>,
) -> Result<HashMap<String, NodeId>, RsvsError> {
    let labels: Vec<&str> = if let Some(seeds) = custom_seeds {
        seeds.iter().map(|s| s.as_str()).collect()
    } else {
        SEED_ATOMS.to_vec()
    };
    let expected_count = labels.len();

    let mut label_map = HashMap::new();

    for label in &labels {
        let node = Node {
            id: 0, // will be assigned by insert_node
            label: label.to_string(),
            surface_label: format!("{}@en", label),

            kind: "node".to_string(),
            tier: Tier::Tier1,
            confidence: 1.0,
            status: NodeStatus::Stable,
            is_seed: true,
            is_locked: true,

            semantic: SemanticMeta {
                compression_state: CompressionState::Raw,
                layer: 0, // Seeds are Layer 0 primitives
                derived_from_node_ids: vec![],
                compression_reason: None,
            },
            policy_meta: None,
            language_links: vec![],

            atoms: vec![],
            fingerprint: None,
        };

        let id = graph.insert_node(node)?;
        label_map.insert(label.to_string(), id);
    }

    if label_map.len() != expected_count {
        return Err(RsvsError::SeedInvariant(format!(
            "Seed node count mismatch — expected {}, got {}",
            expected_count,
            label_map.len()
        )));
    }

    Ok(label_map)
}

/// Public list of seed atom labels — used by pipeline for grounding checks.
pub const SEED_LABEL_LIST: &[&str] = &[
    "exists",
    "entity",
    "relation",
    "state",
    "change",
    "time",
    "space",
    "cause",
    "effect",
    "context",
    "signal",
    "pattern",
    "memory",
    "attention",
    "value",
    "agent",
    "goal",
    "risk",
    "trust",
    "identity",
    "language",
    "meaning",
    "action",
    "feedback",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_count_is_correct() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph, None).unwrap();
        assert_eq!(map.len(), 24);
        assert_eq!(graph.node_count(), 24);
    }

    #[test]
    fn all_seed_nodes_are_tier1_and_stable() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph, None).unwrap();
        for id in map.values() {
            let node = graph.get_node(*id).unwrap();
            assert_eq!(node.tier, Tier::Tier1);
            assert_eq!(node.confidence, 1.0);
            assert_eq!(node.status, NodeStatus::Stable);
            assert!(node.is_seed);
            assert!(node.is_locked);
        }
    }

    #[test]
    fn seed_nodes_have_surface_label_format() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph, None).unwrap();
        for id in map.values() {
            let node = graph.get_node(*id).unwrap();
            assert!(
                node.surface_label.ends_with("@en"),
                "surface_label '{}' should end with @en",
                node.surface_label
            );
        }
    }

    #[test]
    fn seed_nodes_have_correct_labels() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph, None).unwrap();
        assert!(map.contains_key("exists"));
        assert!(map.contains_key("entity"));
        assert!(map.contains_key("relation"));
        assert!(map.contains_key("feedback"));
    }

    #[test]
    fn seed_nodes_are_raw_compression() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph, None).unwrap();
        for id in map.values() {
            let node = graph.get_node(*id).unwrap();
            assert_eq!(node.semantic.compression_state, CompressionState::Raw);
            assert!(node.semantic.derived_from_node_ids.is_empty());
        }
    }
}
