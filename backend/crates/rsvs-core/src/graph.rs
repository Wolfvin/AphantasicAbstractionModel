//! The RSVS graph — in-memory, DAG, integer-keyed (v4.2)
//!
//! v4.2: Unified node model. No more Atom/Composite distinction.
//! expand() checks CompressionState to decide expansion strategy.

use crate::error::RsvsError;
use crate::types::{AtomSet, CompressionState, Edge, Node, NodeId};
use std::collections::HashMap;

#[derive(Debug)]
/// The RSVS knowledge graph — in-memory, DAG, integer-keyed (v4.2).
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

impl Default for RsvsGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl RsvsGraph {
    /// Create a new empty graph.
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

    /// Insert a new node into the graph.
    ///
    /// Returns the assigned `NodeId`. Returns `Err` if a circular reference is detected
    /// in `derived_from_node_ids` or if a referenced derived node does not exist.
    ///
    /// # Examples
    /// ```ignore
    /// let mut g = RsvsGraph::new();
    /// let id = g.insert_node(Node { id: 0, label: "test".into(), ..Default::default() })?;
    /// ```
    pub fn insert_node(&mut self, mut node: Node) -> Result<NodeId, RsvsError> {
        // Assign ID if not already set (0 = unassigned sentinel)
        if node.id == 0 {
            node.id = self.alloc_id();
        }

        // v4.2 DAG constraint: no self-reference in derived_from_node_ids
        if node.semantic.compression_state == CompressionState::Compressed {
            if node.semantic.derived_from_node_ids.contains(&node.id) {
                return Err(RsvsError::CircularRef {
                    from: node.id,
                    to: node.id,
                });
            }
            // All referenced derived nodes must already exist
            for &derived_id in &node.semantic.derived_from_node_ids {
                if !self.nodes.contains_key(&derived_id) {
                    return Err(RsvsError::NodeNotFound { id: derived_id });
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

    /// Insert a directed edge between two existing nodes.
    ///
    /// Returns `Err` if either endpoint does not exist in the graph.
    ///
    /// # Examples
    /// ```ignore
    /// let mut g = RsvsGraph::new();
    /// let a = g.insert_node(Node { id: 0, label: "a".into(), ..Default::default() })?;
    /// let b = g.insert_node(Node { id: 0, label: "b".into(), ..Default::default() })?;
    /// g.insert_edge(Edge { from: a, to: b, weight: 0.8, source: EdgeSource::Learned })?;
    /// ```
    pub fn insert_edge(&mut self, edge: Edge) -> Result<(), RsvsError> {
        // Both endpoints must exist
        if !self.nodes.contains_key(&edge.from) {
            return Err(RsvsError::NodeNotFound { id: edge.from });
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(RsvsError::NodeNotFound { id: edge.to });
        }
        self.edges.entry(edge.from).or_default().push(edge);
        Ok(())
    }

    // ---------------------------------------------------------------
    // Lookup
    // ---------------------------------------------------------------

    /// Look up a node by its integer ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Look up a node ID by its label string.
    pub fn id_for_label(&self, label: &str) -> Option<NodeId> {
        self.label_to_id.get(label).copied()
    }

    /// Return the outgoing edges from a node.
    pub fn edges_from(&self, node_id: NodeId) -> &[Edge] {
        self.edges
            .get(&node_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Return the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the total number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }

    // ---------------------------------------------------------------
    // Expand (v4.2: based on CompressionState)
    //
    // If compressed: expand to derived_from_node_ids
    // If raw: expand to atoms, or just [id] if atoms is empty
    // ---------------------------------------------------------------

    /// Expand a node into its atom set based on `CompressionState`.
    ///
    /// Compressed nodes expand to `derived_from_node_ids`; raw nodes expand to
    /// their `atoms` field, or just `[id]` if atoms is empty.
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

    /// Compute the full similarity breakdown between two nodes.
    pub fn similarity(&self, a: NodeId, b: NodeId) -> SimilarityResult {
        let atoms_a = self.expand(a);
        let atoms_b = self.expand(b);

        let mut shared = vec![];
        let mut only_a = vec![];
        let mut only_b = atoms_b.clone();

        for &atom in &atoms_a {
            if atoms_b.contains(&atom) {
                shared.push(atom);
                only_b.retain(|&x| x != atom);
            } else {
                only_a.push(atom);
            }
        }

        let jaccard = jaccard_sets(&atoms_a, &atoms_b);

        SimilarityResult {
            shared,
            only_a,
            only_b,
            jaccard,
        }
    }

    // ---------------------------------------------------------------
    // Jaccard between two atom sets (used by attention scoring)
    // ---------------------------------------------------------------

    /// Compute Jaccard similarity between the expanded atom sets of two nodes.
    ///
    /// # Examples
    /// ```ignore
    /// let score = graph.jaccard_atom_sets(node_a, node_b);
    /// ```
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
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

#[derive(Debug)]
/// Detailed similarity breakdown between two nodes.
pub struct SimilarityResult {
    /// Node IDs shared between both nodes.
    pub shared: Vec<NodeId>,
    /// Node IDs only in node A.
    pub only_a: Vec<NodeId>,
    /// Node IDs only in node B.
    pub only_b: Vec<NodeId>,
    /// Jaccard similarity coefficient.
    pub jaccard: f32,
}
