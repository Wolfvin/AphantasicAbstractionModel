//! The RSVS graph — in-memory, DAG, integer-keyed (v4.2)
//!
//! v4.2: Unified node model. No more Atom/Composite distinction.
//! expand() checks CompressionState to decide expansion strategy.

use std::collections::HashMap;
use crate::types::{NodeId, Node, Edge, AtomSet, CompressionState};

#[derive(Debug)]
pub struct RsvsGraph {
    /// All nodes indexed by integer ID.
    pub nodes: HashMap<NodeId, Node>,

    /// Adjacency list: node → list of edges to other nodes.
    pub(crate) edges: HashMap<NodeId, Vec<Edge>>,

    /// Label → NodeId lookup (for input parsing only).
    pub(crate) label_to_id: HashMap<String, NodeId>,

    /// Next available ID.
    pub(crate) next_id: NodeId,
}

impl RsvsGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            label_to_id: HashMap::new(),
            next_id: 1,
        }
    }

    // ---------------------------------------------------------------
    // ID allocation
    // ---------------------------------------------------------------

    fn alloc_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // ---------------------------------------------------------------
    // Node insertion (v4.2)
    // ---------------------------------------------------------------

    /// Insert a node. Returns Err if circular reference detected.
    pub fn insert_node(&mut self, mut node: Node) -> Result<NodeId, String> {
        // Assign ID if not already set (0 = unassigned sentinel)
        if node.id == 0 {
            node.id = self.alloc_id();
        }

        // v4.2 DAG constraint: no self-reference in derived_from_node_ids
        if node.semantic.compression_state == CompressionState::Compressed {
            if node.semantic.derived_from_node_ids.contains(&node.id) {
                return Err(format!(
                    "Circular reference: ID {} appears in its own derived_from_node_ids",
                    node.id
                ));
            }
            // All referenced derived nodes must already exist
            for &derived_id in &node.semantic.derived_from_node_ids {
                if !self.nodes.contains_key(&derived_id) {
                    return Err(format!(
                        "Unknown node ID {} referenced in derived_from_node_ids of {}",
                        derived_id, node.id
                    ));
                }
            }
        }

        let id = node.id;
        // Register label in both label and surface_label lookups
        self.label_to_id.insert(node.label.clone(), id);
        if node.surface_label != node.label {
            self.label_to_id.insert(node.surface_label.clone(), id);
        }
        self.nodes.insert(id, node);
        Ok(id)
    }

    // ---------------------------------------------------------------
    // Edge insertion (v4.2: any node → any node)
    // ---------------------------------------------------------------

    pub fn insert_edge(&mut self, edge: Edge) -> Result<(), String> {
        // Both endpoints must exist
        if !self.nodes.contains_key(&edge.from) {
            return Err(format!("Edge source {} does not exist", edge.from));
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(format!("Edge target {} does not exist", edge.to));
        }
        self.edges.entry(edge.from).or_default().push(edge);
        Ok(())
    }

    // ---------------------------------------------------------------
    // Lookup
    // ---------------------------------------------------------------

    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn id_for_label(&self, label: &str) -> Option<NodeId> {
        self.label_to_id.get(label).copied()
    }

    pub fn edges_from(&self, node_id: NodeId) -> &[Edge] {
        self.edges.get(&node_id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }

    // ---------------------------------------------------------------
    // Expand (v4.2: based on CompressionState)
    //
    // If compressed: expand to derived_from_node_ids
    // If raw: expand to atoms, or just [id] if atoms is empty
    // ---------------------------------------------------------------

    pub fn expand(&self, id: NodeId) -> AtomSet {
        match self.nodes.get(&id) {
            None => vec![],
            Some(node) => {
                match node.semantic.compression_state {
                    CompressionState::Compressed => {
                        // Compressed node expands to its derivation
                        let mut expanded = node.semantic.derived_from_node_ids.clone();
                        if expanded.is_empty() {
                            expanded.push(id);
                        }
                        expanded
                    }
                    CompressionState::Raw => {
                        // Raw node expands to its atom set, or self
                        if node.atoms.is_empty() {
                            vec![id]
                        } else {
                            node.atoms.clone()
                        }
                    }
                }
            }
        }
    }

    pub fn similarity(&self, a: NodeId, b: NodeId) -> SimilarityResult {
        let atoms_a = self.expand(a);
        let atoms_b = self.expand(b);

        let mut shared   = vec![];
        let mut only_a   = vec![];
        let mut only_b   = atoms_b.clone();

        for &atom in &atoms_a {
            if atoms_b.contains(&atom) {
                shared.push(atom);
                only_b.retain(|&x| x != atom);
            } else {
                only_a.push(atom);
            }
        }

        let jaccard = jaccard_sets(&atoms_a, &atoms_b);

        SimilarityResult { shared, only_a, only_b, jaccard }
    }

    // ---------------------------------------------------------------
    // Jaccard between two atom sets (used by attention scoring)
    // ---------------------------------------------------------------

    pub fn jaccard_atom_sets(&self, a: NodeId, b: NodeId) -> f32 {
        let atoms_a = self.expand(a);
        let atoms_b = self.expand(b);
        jaccard_sets(&atoms_a, &atoms_b)
    }
}

/// Jaccard similarity between two atom sets.
/// |A ∩ B| / |A ∪ B|
pub fn jaccard_sets(a: &[NodeId], b: &[NodeId]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0; // undefined → 0 by convention
    }
    let intersection = a.iter().filter(|x| b.contains(x)).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
}

#[derive(Debug)]
pub struct SimilarityResult {
    pub shared:  Vec<NodeId>,
    pub only_a:  Vec<NodeId>,
    pub only_b:  Vec<NodeId>,
    pub jaccard: f32,
}
