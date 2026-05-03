//! The RSVS graph — in-memory, DAG, integer-keyed (v5.0 — Compositional)
//!
//! v5.0: Compositional architecture. Nodes can have multiple senses,
//! each defined by compositions (references to specific senses of other nodes).
//!
//! Key additions:
//! - `structural_similarity()`: Compare two nodes based on shared/differing
//!   compositions at the sense level.
//! - `substitution_analysis()`: Find what composition substitutions transform
//!   one sense into another (e.g., raja→ratu by substituting laki-laki→perempuan).
//! - `expand()` now recurses through compositions, not just atom sets.

use crate::error::RsvsError;
use crate::sense::SenseManager;
use crate::types::{AtomSet, CompositionRef, CompressionState, Edge, Node, NodeId};
use std::collections::HashMap;

#[derive(Debug)]
/// The RSVS knowledge graph — in-memory, DAG, integer-keyed (v5.0).
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

    fn alloc_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Insert a new node into the graph.
    pub fn insert_node(&mut self, mut node: Node) -> Result<NodeId, RsvsError> {
        if node.id == 0 {
            node.id = self.alloc_id();
        }

        if node.semantic.compression_state == CompressionState::Compressed {
            if node.semantic.derived_from_node_ids.contains(&node.id) {
                return Err(RsvsError::CircularRef {
                    from: node.id,
                    to: node.id,
                });
            }
            for &derived_id in &node.semantic.derived_from_node_ids {
                if !self.nodes.contains_key(&derived_id) {
                    return Err(RsvsError::NodeNotFound { id: derived_id });
                }
            }
        }

        let id = node.id;
        self.label_to_id.insert(node.label.clone(), id);
        if node.surface_label != node.label {
            self.label_to_id.insert(node.surface_label.clone(), id);
        }
        self.nodes.insert(id, node);
        Ok(id)
    }

    /// Insert a directed edge between two existing nodes.
    pub fn insert_edge(&mut self, edge: Edge) -> Result<(), RsvsError> {
        if !self.nodes.contains_key(&edge.from) {
            return Err(RsvsError::NodeNotFound { id: edge.from });
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(RsvsError::NodeNotFound { id: edge.to });
        }
        self.edges.entry(edge.from).or_default().push(edge);
        Ok(())
    }

    /// Get a reference to a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Get a mutable reference to a node by ID.
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// Look up a node ID by its label.
    pub fn id_for_label(&self, label: &str) -> Option<NodeId> {
        self.label_to_id.get(label).copied()
    }

    /// Get all edges originating from a node.
    pub fn edges_from(&self, node_id: NodeId) -> &[Edge] {
        self.edges
            .get(&node_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Return the total number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the total number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }

    /// Expand a node into its atom set based on `CompressionState`.
    pub fn expand(&self, id: NodeId) -> AtomSet {
        match self.nodes.get(&id) {
            None => vec![],
            Some(node) => match node.semantic.compression_state {
                CompressionState::Compressed => {
                    let mut expanded = node.semantic.derived_from_node_ids.clone();
                    if expanded.is_empty() {
                        expanded.push(id);
                    }
                    expanded
                }
                CompressionState::Raw => {
                    if node.atoms.is_empty() {
                        vec![id]
                    } else {
                        node.atoms.clone()
                    }
                }
            },
        }
    }

    /// Compute the full similarity breakdown between two nodes (flat, v4 compat).
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

    /// Compute structural similarity between two nodes at the sense level (v5.0).
    ///
    /// This is the core of RSVS v5.0's compositional architecture. Two nodes
    /// are structurally similar if their senses share compositions.
    ///
    /// Example:
    ///   raja.sense_0 = [(tahta_tertinggi, 0), (laki_laki, 0), (kerajaan, 0)]
    ///   ratu.sense_0 = [(tahta_tertinggi, 0), (perempuan, 0), (kerajaan, 0)]
    ///   → shared: [(tahta_tertinggi, 0), (kerajaan, 0)]
    ///   → only_a: [(laki_laki, 0)]
    ///   → only_b: [(perempuan, 0)]
    ///   → structural_similarity: 2/3 = 0.667
    pub fn structural_similarity(
        &self,
        _a: NodeId,
        _b: NodeId,
        senses_a: &SenseManager,
        senses_b: &SenseManager,
    ) -> StructuralSimResult {
        let mut best_result: Option<StructuralSimResult> = None;

        for (idx_a, sense_a) in senses_a.senses.iter().enumerate() {
            for (idx_b, sense_b) in senses_b.senses.iter().enumerate() {
                if !sense_a.is_compositional() && !sense_b.is_compositional() {
                    continue;
                }

                let shared: Vec<CompositionRef> = sense_a
                    .compositions
                    .iter()
                    .filter(|c| sense_b.compositions.contains(c))
                    .cloned()
                    .collect();

                let only_a: Vec<CompositionRef> = sense_a
                    .compositions
                    .iter()
                    .filter(|c| !sense_b.compositions.contains(c))
                    .cloned()
                    .collect();

                let only_b: Vec<CompositionRef> = sense_b
                    .compositions
                    .iter()
                    .filter(|c| !sense_a.compositions.contains(c))
                    .cloned()
                    .collect();

                let union_len = sense_a.compositions.len()
                    + sense_b.compositions.len()
                    - shared.len();

                let score = if union_len == 0 {
                    0.0
                } else {
                    shared.len() as f32 / union_len as f32
                };

                let is_better = best_result
                    .as_ref()
                    .map(|r| score > r.structural_similarity)
                    .unwrap_or(true);

                if is_better {
                    best_result = Some(StructuralSimResult {
                        sense_idx_a: idx_a,
                        sense_idx_b: idx_b,
                        shared_compositions: shared,
                        only_a_compositions: only_a,
                        only_b_compositions: only_b,
                        structural_similarity: score,
                        layer_a: sense_a.layer,
                        layer_b: sense_b.layer,
                    });
                }
            }
        }

        best_result.unwrap_or_else(|| StructuralSimResult {
            sense_idx_a: 0,
            sense_idx_b: 0,
            shared_compositions: vec![],
            only_a_compositions: vec![],
            only_b_compositions: vec![],
            structural_similarity: 0.0,
            layer_a: 0,
            layer_b: 0,
        })
    }

    /// Analyze what substitution transforms one node's sense into another's (v5.0).
    ///
    /// This is the "why" of RSVS: it doesn't just say two things are
    /// related, it says exactly WHICH composition needs to change to
    /// transform one into the other.
    pub fn substitution_analysis(
        &self,
        a: NodeId,
        b: NodeId,
        senses_a: &SenseManager,
        senses_b: &SenseManager,
    ) -> Option<SubstitutionResult> {
        let sim = self.structural_similarity(a, b, senses_a, senses_b);

        if sim.only_a_compositions.is_empty() && sim.only_b_compositions.is_empty() {
            return None;
        }

        let substitutions: Vec<(CompositionRef, CompositionRef)> = sim
            .only_a_compositions
            .iter()
            .zip(sim.only_b_compositions.iter())
            .map(|(from, to)| (from.clone(), to.clone()))
            .collect();

        let unpaired_a: Vec<CompositionRef> = if sim.only_a_compositions.len()
            > sim.only_b_compositions.len()
        {
            sim.only_a_compositions[sim.only_b_compositions.len()..].to_vec()
        } else {
            vec![]
        };

        let unpaired_b: Vec<CompositionRef> = if sim.only_b_compositions.len()
            > sim.only_a_compositions.len()
        {
            sim.only_b_compositions[sim.only_a_compositions.len()..].to_vec()
        } else {
            vec![]
        };

        Some(SubstitutionResult {
            sense_idx_a: sim.sense_idx_a,
            sense_idx_b: sim.sense_idx_b,
            substitutions,
            unpaired_only_a: unpaired_a,
            unpaired_only_b: unpaired_b,
            structural_similarity: sim.structural_similarity,
        })
    }

    /// Compute Jaccard similarity between the expanded atom sets of two nodes.
    pub fn jaccard_atom_sets(&self, a: NodeId, b: NodeId) -> f32 {
        let atoms_a = self.expand(a);
        let atoms_b = self.expand(b);
        jaccard_sets(&atoms_a, &atoms_b)
    }
}

/// Jaccard similarity between two atom sets.
pub fn jaccard_sets(a: &[NodeId], b: &[NodeId]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
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
/// Detailed similarity breakdown between two nodes (flat, v4 compat).
pub struct SimilarityResult {
    pub shared: Vec<NodeId>,
    pub only_a: Vec<NodeId>,
    pub only_b: Vec<NodeId>,
    pub jaccard: f32,
}

/// Structural similarity result between two nodes at the sense level (v5.0).
#[derive(Debug, Clone)]
pub struct StructuralSimResult {
    /// Index of the best-matching sense in node A.
    pub sense_idx_a: usize,
    /// Index of the best-matching sense in node B.
    pub sense_idx_b: usize,
    /// Compositions shared by both senses.
    pub shared_compositions: Vec<CompositionRef>,
    /// Compositions only in sense A.
    pub only_a_compositions: Vec<CompositionRef>,
    /// Compositions only in sense B.
    pub only_b_compositions: Vec<CompositionRef>,
    /// Structural similarity score = shared / union.
    pub structural_similarity: f32,
    /// Layer of sense A.
    pub layer_a: u32,
    /// Layer of sense B.
    pub layer_b: u32,
}

/// Result of substitution analysis (v5.0).
#[derive(Debug, Clone)]
pub struct SubstitutionResult {
    /// Index of the sense in node A.
    pub sense_idx_a: usize,
    /// Index of the sense in node B.
    pub sense_idx_b: usize,
    /// Paired substitutions: (from_composition, to_composition).
    pub substitutions: Vec<(CompositionRef, CompositionRef)>,
    /// Unpaired compositions only in A.
    pub unpaired_only_a: Vec<CompositionRef>,
    /// Unpaired compositions only in B.
    pub unpaired_only_b: Vec<CompositionRef>,
    /// Overall structural similarity.
    pub structural_similarity: f32,
}
