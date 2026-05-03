//! Seed graph bootstrap.
//!
//! Loads the 24 Layer-0 and Layer-1 atoms into the graph at startup.
//! These atoms have confidence=1.0, Tier=1, and cannot be removed.
//! All atoms above Layer 1 (hard, round, hot, etc.) emerge from data.

use crate::types::{Node, NodeId, NodeKind, Tier};
use crate::graph::RsvsGraph;

/// Seed atom definitions: (label, layer)
const SEED_ATOMS: &[(&str, u8)] = &[
    // Layer 0 — Absolute Primitives (cannot be extended without Wolfvin)
    ("exists", 0), ("not", 0), ("one", 0), ("other", 0), ("if", 0),
    ("this", 0),   ("that", 0), ("i", 0),
    ("good", 0),   ("bad", 0),

    // Layer 1 — Grounded Primitives (extendable with concrete use case)
    ("see", 1),    ("hear", 1),   ("feel", 1),
    ("know", 1),   ("think", 1),  ("want", 1),
    ("do", 1),     ("can", 1),
    ("happen", 1),
    ("before", 1), ("after", 1),
    // Layer 1 extension for v4.1 baseline
    ("where", 1), ("when", 1), ("because", 1),
];

/// Bootstrap the graph with all 24 seed atoms.
/// Returns a map of label → NodeId for external reference.
pub fn bootstrap(graph: &mut RsvsGraph) -> std::collections::HashMap<String, NodeId> {
    let mut label_map = std::collections::HashMap::new();

    for (label, _layer) in SEED_ATOMS {
        let node = Node {
            id: 0, // will be assigned by insert_node
            kind: NodeKind::Atom,
            atoms: vec![],
            confidence: 1.0,
            tier: Tier::Tier1,
            label: Some(label.to_string()),
            fingerprint: None,
        };

        match graph.insert_node(node) {
            Ok(id) => {
                label_map.insert(label.to_string(), id);
            }
            Err(e) => {
                // Should never happen for atoms — panic is appropriate here
                panic!("Failed to seed atom '{}': {}", label, e);
            }
        }
    }

    assert_eq!(
        label_map.len(), SEED_ATOMS.len(),
        "Seed atom count mismatch — expected {}", SEED_ATOMS.len()
    );

    label_map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_count_is_correct() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph);
        assert_eq!(map.len(), 24); // 10 Layer0 + 14 Layer1
        assert_eq!(graph.node_count(), 24);
    }

    #[test]
    fn all_seed_atoms_are_tier1() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph);
        for id in map.values() {
            let node = graph.get_node(*id).unwrap();
            assert_eq!(node.tier, Tier::Tier1);
            assert_eq!(node.confidence, 1.0);
        }
    }

    #[test]
    fn seed_atoms_have_correct_labels() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph);
        assert!(map.contains_key("exists"));
        assert!(map.contains_key("feel"));
        assert!(map.contains_key("before"));
        assert!(map.contains_key("after"));
    }
}

/// Public list of seed atom labels — used by pipeline for grounding checks.
pub const SEED_LABEL_LIST: &[&str] = &[
    "exists", "not", "one", "other", "if",
    "this", "that", "i", "good", "bad",
    "see", "hear", "feel", "know", "think",
    "want", "do", "can", "happen", "before", "after",
    "where", "when", "because",
];
