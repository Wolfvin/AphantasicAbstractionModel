//! The RSVS graph — in-memory, DAG, integer-keyed (v6.0 — Compositional)
//!
//! v6.0: Compositional architecture. Nodes can have multiple senses,
//! each defined by compositions (references to specific senses of other nodes).
//!
//! Key additions:
//! - `structural_similarity()`: Compare two nodes based on shared/differing
//!   compositions at the sense level. Uses HashSet for O(1) lookups.
//! - `substitution_analysis()`: Find what composition substitutions transform
//!   one sense into another (e.g., raja→ratu by substituting laki-laki→perempuan).
//! - `expand()` now recurses through compositions, not just atom sets.

use crate::error::RsvsError;
use crate::sense::SenseManager;
use crate::types::{AtomSet, CompositionRef, CompressionState, Edge, EdgeSource, Node, NodeId, SenseId};
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
/// The RSVS knowledge graph — in-memory, DAG, integer-keyed (v6.0).
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

        // Use HashSet for O(1) lookup instead of Vec::contains (O(N))
        let set_b: HashSet<NodeId> = atoms_b.iter().copied().collect();

        let mut shared = vec![];
        let mut only_a = vec![];
        let mut only_b = atoms_b.clone();

        for &atom in &atoms_a {
            if set_b.contains(&atom) {
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

    /// Compute structural similarity between two nodes at the sense level (v6.0).
    ///
    /// This is the core of RSVS v6.0's compositional architecture. Two nodes
    /// are structurally similar if their senses share compositions.
    ///
    /// Uses HashSet for O(1) composition lookups instead of Vec::contains().
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

                // Use HashSet for O(1) lookups
                let set_a: HashSet<&CompositionRef> = sense_a.compositions.iter().collect();
                let set_b: HashSet<&CompositionRef> = sense_b.compositions.iter().collect();

                let shared: Vec<CompositionRef> =
                    set_a.intersection(&set_b).map(|c| (*c).clone()).collect();

                let only_a: Vec<CompositionRef> = set_a
                    .difference(&set_b)
                    .map(|c| (*c).clone())
                    .collect();

                let only_b: Vec<CompositionRef> = set_b
                    .difference(&set_a)
                    .map(|c| (*c).clone())
                    .collect();

                let union_len = set_a.union(&set_b).count();

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

    /// Analyze what substitution transforms one node's sense into another's (v6.0).
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

    /// v6.3: Apply inactivity-based decay to all learned edges.
    ///
    /// Called once per batch (from pipeline after ingest completes).
    /// Bootstrap edges (last_reinforced_batch == 0) and Composition edges
    /// are not decayed — only learned edges can weaken.
    ///
    /// Formula: weight_new = weight_old × decay_factor^(current_batch - last_reinforced)
    pub fn decay_edge_weights(
        &mut self,
        current_batch: usize,
        decay_factor: f32,
        grace_period: usize,
    ) -> usize {
        let mut decayed = 0;
        if (decay_factor - 1.0).abs() < 1e-6 {
            return 0; // Decay disabled (factor ≈ 1.0)
        }
        for edges in self.edges.values_mut() {
            for edge in edges.iter_mut() {
                // Don't decay bootstrap or composition edges
                if edge.last_reinforced_batch == 0 {
                    continue;
                }
                if matches!(edge.source, EdgeSource::Bootstrap | EdgeSource::Composition) {
                    continue;
                }

                let inactive_batches = current_batch.saturating_sub(edge.last_reinforced_batch);

                if inactive_batches <= grace_period {
                    continue; // Still within grace period
                }

                let batches_to_decay = inactive_batches - grace_period;
                let decay = decay_factor.powi(batches_to_decay as i32);
                edge.weight = (edge.weight * decay).clamp(0.01, 1.0);
                // Floor 0.01 — never zero, but can be very weak
                decayed += 1;
            }
        }
        decayed
    }

    /// v6.3: Reinforce an edge (called when attention selects this pair again).
    /// Resets the inactivity counter and applies EMA weight update.
    pub fn reinforce_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        evidence: f32,
        current_batch: usize,
        eta: f32,
    ) {
        if let Some(edges) = self.edges.get_mut(&from) {
            for edge in edges.iter_mut() {
                if edge.to == to {
                    // EMA update: weight_new = (1 - eta) * old + eta * evidence
                    edge.weight = ((1.0 - eta) * edge.weight + eta * evidence).clamp(0.0, 1.0);
                    edge.last_reinforced_batch = current_batch;
                    return;
                }
            }
        }
        // Edge doesn't exist — insert new
        let edge = Edge {
            from,
            to,
            weight: evidence,
            source: EdgeSource::Learned,
            last_reinforced_batch: current_batch,
        };
        let _ = self.insert_edge(edge);
    }

    /// v6.2: Context-weighted similarity between two nodes.
    ///
    /// Unlike `structural_similarity()` which compares compositions structurally
    /// (shared vs differing), this method weighs each composition based on its
    /// relevance to the `context_atoms`. This produces a context-aware similarity
    /// score that reflects how similar two nodes are WITHIN a given context.
    ///
    /// Formula: sim(A, B | q) = cosine_similarity(score_vec_A, score_vec_B)
    /// where score_vec[comp] = P(a|S,q) = freq_map[a] × edge_weight(a→q)
    ///
    /// Example: "batu" and "tulang" may have low structural similarity in general,
    /// but if context_atoms includes "kekerasan", both have high scores for the
    /// "hard" atom, producing a high context-weighted similarity.
    ///
    /// This method is the foundation for context-aware Appraise and Relate modes.
    pub fn context_weighted_similarity(
        &self,
        senses_a: &SenseManager,
        senses_b: &SenseManager,
        context_atoms: &[NodeId],
    ) -> f32 {
        // Convert slice to Vec for lazy_lookup compatibility (AtomSet = Vec<NodeId>)
        let context_vec: AtomSet = context_atoms.to_vec();
        // Select active sense for each node based on context
        let sense_a = match senses_a.lazy_lookup(&context_vec) {
            Some(idx) => &senses_a.senses[idx],
            None => return 0.0,
        };
        let sense_b = match senses_b.lazy_lookup(&context_vec) {
            Some(idx) => &senses_b.senses[idx],
            None => return 0.0,
        };

        // Collect all unique CompositionRefs from both senses
        let all_comps: HashSet<&CompositionRef> = sense_a.compositions.iter()
            .chain(sense_b.compositions.iter())
            .collect();

        if all_comps.is_empty() {
            return 0.0;
        }

        // Compute edge_weight per composition — proximity to context
        // Simple heuristic: if the composition's node_id is in context_atoms → 1.0,
        // otherwise → 0.1 (low relevance but not zero, to avoid division by zero)
        let context_set: HashSet<NodeId> = context_atoms.iter().copied().collect();
        let edge_weight = |comp: &CompositionRef| -> f32 {
            if context_set.contains(&comp.node_id) { 1.0 } else { 0.1 }
        };

        // Compute cosine similarity between score vectors
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for comp in &all_comps {
            let ew = edge_weight(comp);
            let score_a = sense_a.p_a_given_s_q(comp, ew);
            let score_b = sense_b.p_a_given_s_q(comp, ew);
            dot += score_a * score_b;
            norm_a += score_a * score_a;
            norm_b += score_b * score_b;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 { 0.0 } else { (dot / denom).clamp(0.0, 1.0) }
    }

    /// v6.3: Re-score nodes in relate result based on cross-activation coherence.
    ///
    /// For each candidate node, compute how many other already-activated nodes
    /// share compositions with it. Nodes that are more coherent with the
    /// activated set get a score boost.
    ///
    /// This is a single-pass operation (no iterative feedback loop).
    /// `boost_factor` controls how much each shared composition adds to the score.
    pub fn cross_activation_rescore(
        candidates: &mut Vec<(NodeId, f32)>,
        activated_senses: &HashMap<NodeId, SenseId>,
        all_senses: &HashMap<NodeId, SenseManager>,
        boost_factor: f32,
    ) {
        // Collect all compositions from activated senses
        let activated_comps: HashSet<CompositionRef> = activated_senses.iter()
            .filter_map(|(id, &sid)| {
                let sm = all_senses.get(id)?;
                let sense = sm.get_sense(sid as usize)?;
                Some(sense.compositions.clone())
            })
            .flatten()
            .collect();

        // Rescore candidates based on overlap with activated compositions
        for (node_id, score) in candidates.iter_mut() {
            if let Some(sm) = all_senses.get(node_id) {
                if let Some(&active_idx) = activated_senses.get(node_id) {
                    if let Some(sense) = sm.get_sense(active_idx as usize) {
                        let shared = sense.compositions.iter()
                            .filter(|c| activated_comps.contains(c))
                            .count();
                        let coherence_boost = shared as f32 * boost_factor;
                        *score += coherence_boost;
                    }
                }
            }
        }
        // Re-sort after rescore
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    }
}

/// Jaccard similarity between two atom sets.
///
/// v7.0: Optimized from O(n×m) to O(n+m) using HashSet for the second set.
/// Previously used `b.contains(x)` which is O(m) per call on a Vec,
/// making the total O(n×m). Now converts `b` to a HashSet once for O(1) lookups.
pub fn jaccard_sets(a: &[NodeId], b: &[NodeId]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let set_b: HashSet<NodeId> = b.iter().copied().collect();
    let intersection = a.iter().filter(|x| set_b.contains(x)).count();
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

/// Structural similarity result between two nodes at the sense level (v6.0).
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

/// Result of substitution analysis (v6.0).
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
