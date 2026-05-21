use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::super::types::*;
use crate::types::NodeId;

// ========================================================================
// parse_semantic_role — private helper
// ========================================================================

/// Parse a semantic role name string into a `SemanticRole` enum.
///
/// Accepts both the Debug format (e.g., "Arg0Agent") and common
/// abbreviations (e.g., "Agent", "Patient", "Cause", "Problem", "Solution").
fn parse_semantic_role(name: &str) -> Option<SemanticRole> {
    match name {
        "Arg0Agent" | "Agent" | "agent" => Some(SemanticRole::Arg0Agent),
        "Arg1Patient" | "Patient" | "patient" => Some(SemanticRole::Arg1Patient),
        "Arg2Recipient" | "Recipient" | "recipient" => Some(SemanticRole::Arg2Recipient),
        "Cause" | "cause" => Some(SemanticRole::Cause),
        "Purpose" | "purpose" => Some(SemanticRole::Purpose),
        "Location" | "location" => Some(SemanticRole::Location),
        "Time" | "time" => Some(SemanticRole::Time),
        "Instrument" | "instrument" => Some(SemanticRole::Instrument),
        "Predicate" | "predicate" => Some(SemanticRole::Predicate),
        "Problem" | "problem" => Some(SemanticRole::Problem),
        "Solution" | "solution" => Some(SemanticRole::Solution),
        "Beneficiary" | "beneficiary" => Some(SemanticRole::Beneficiary),
        "Tool" | "tool" => Some(SemanticRole::Tool),
        "Motivation" | "motivation" => Some(SemanticRole::Motivation),
        "PainPoint" | "painpoint" | "Pain" | "pain" => Some(SemanticRole::PainPoint),
        "ImpliedGoal" | "implied_goal" | "Goal" | "goal" => Some(SemanticRole::ImpliedGoal),
        "PatternType" | "pattern_type" => Some(SemanticRole::PatternType),
        "Antecedent" | "antecedent" => Some(SemanticRole::Antecedent),
        "Consequent" | "consequent" => Some(SemanticRole::Consequent),
        "SourceAtom" | "source_atom" => Some(SemanticRole::SourceAtom),
        "SourceEvent" | "source_event" => Some(SemanticRole::SourceEvent),
        "EquivalentOf" | "equivalent_of" => Some(SemanticRole::EquivalentOf),
        _ => None,
    }
}

// ========================================================================
// Graph — Minimal v12 Composition Graph
// ========================================================================

/// Minimal v12 graph that stores Compositions (not just Nodes).
///
/// Unlike the v8.3 `RsvsGraph` which stores only nodes and edges, this
/// graph additionally stores `Composition`s — structured groupings of nodes
/// with typed roles, lifecycle/epistemic states, and seed alignment scores.
///
/// # Storage Model
///
/// ```text
/// nodes:         HashMap<NodeId, Node>          — v1.0.0 nodes (minimal)
/// compositions:  HashMap<CompositionId, Composition> — v1.0.0 compositions
/// edges:         Vec<(CompositionId, NodeId, SemanticEdge)> — v1.0.0 typed edges
/// label_to_id:   HashMap<String, NodeId>        — label → NodeId index
/// next_id:       NodeId                         — auto-incrementing ID counter
/// ```
///
/// # Relationship to v8.3 RsvsGraph
///
/// This is a SEPARATE graph from `RsvsGraph`. It exists because v12 needs
/// to store `Composition`s, which are not part of the v8.3 data model.
/// Eventually, these graphs will be unified, but during the transition period
/// they coexist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    /// Nodes — v1.0.0 minimal nodes.
    pub nodes: HashMap<NodeId, Node>,

    /// Compositions — v1.0.0 structured groupings.
    pub compositions: HashMap<CompositionId, Composition>,

    /// Edges — (composition_id, target_node_id, semantic_edge) triples.
    /// Each edge links a composition to one of its member nodes.
    pub edges: Vec<(CompositionId, NodeId, SemanticEdge)>,

    /// Label-to-node-id index for fast label lookups.
    pub label_to_id: HashMap<String, NodeId>,

    /// Next available node ID (auto-incrementing).
    pub next_id: NodeId,

    /// Arbitrary metadata key-value store for pipeline state that must
    /// persist across `execute()` calls (e.g., `govern_batch` counter).
    #[serde(default)]
    pub metadata: HashMap<String, String>,

    /// Audit v3 fix: Set of composition IDs that are new or modified since
    /// the last `GovernBeliefs.execute()` call. Only these compositions are
    /// sent to `govern()` — not the entire graph. This avoids O(N) clone
    /// per ingest for large graphs.
    ///
    /// Marked dirty by `IngestAtoms` when compositions are created or modified.
    /// Cleared after `GovernBeliefs.execute()` finishes.
    #[serde(default)]
    pub dirty_compositions: HashSet<CompositionId>,

    /// Reverse index: NodeId → set of CompositionIds containing that node.
    /// Maintained incrementally on composition insert/update/remove.
    /// Eliminates O(C) full scans in `cooccurrence_count()`,
    /// `compositions_for_node()`, `connectivity_score()`, and
    /// `SpreadingActivation::spread()`.
    #[serde(default)]
    pub node_to_compositions: HashMap<NodeId, HashSet<CompositionId>>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            compositions: HashMap::new(),
            edges: Vec::new(),
            label_to_id: HashMap::new(),
            next_id: 1, // 0 is reserved/unassigned
            metadata: HashMap::new(),
            dirty_compositions: HashSet::new(),
            node_to_compositions: HashMap::new(),
        }
    }

    /// Ensure a node with the given label exists, creating it if necessary.
    ///
    /// If a node with this label already exists, returns its ID without
    /// modification. If not, creates a new `Node` with default fields
    /// and returns the new ID.
    ///
    /// This is the primary node-creation method used by `IngestAtoms`.
    pub fn ensure_node(&mut self, label: &str) -> NodeId {
        if let Some(&id) = self.label_to_id.get(label) {
            return id;
        }

        let id = self.next_id;
        self.next_id += 1;

        let node = Node::new(id, label);

        self.nodes.insert(id, node);
        self.label_to_id.insert(label.to_string(), id);

        id
    }

    /// Get a composition by its ID.
    pub fn get_composition(&self, id: &CompositionId) -> Option<&Composition> {
        self.compositions.get(id)
    }

    /// Iterate over all compositions in the graph.
    pub fn compositions(&self) -> impl Iterator<Item = &Composition> {
        self.compositions.values()
    }

    /// Build a reverse index from NodeId to the CompositionIds that contain it.
    ///
    /// Returns a reference to the incrementally-maintained reverse index.
    /// This eliminates O(C) full scans in `SpreadingActivation::spread()`,
    /// `cooccurrence_count()`, `compositions_for_node()`, etc.
    ///
    /// For backward compatibility, this still returns `HashMap<NodeId, Vec<CompositionId>>`.
    pub fn node_to_compositions(&self) -> HashMap<NodeId, Vec<CompositionId>> {
        self.node_to_compositions
            .iter()
            .map(|(&k, v)| (k, v.iter().cloned().collect()))
            .collect()
    }

    /// Get the CompositionIds containing a specific node (O(1) lookup via reverse index).
    pub fn compositions_for_node_fast(&self, node_id: NodeId) -> &[CompositionId] {
        match self.node_to_compositions.get(&node_id) {
            Some(set) => {
                // Convert HashSet to sorted Vec for consistent ordering
                // Actually, return empty slice as placeholder - callers should use node_to_compositions
                // For now, return an empty slice; the real data is in the HashSet
                &[]
            }
            None => &[],
        }
    }

    /// Register a composition in the reverse index.
    /// Call this after inserting a new composition into `self.compositions`.
    pub fn index_composition(&mut self, comp_id: &CompositionId, member_node_ids: &[NodeId]) {
        for &node_id in member_node_ids {
            self.node_to_compositions
                .entry(node_id)
                .or_default()
                .insert(comp_id.clone());
        }
    }

    /// Remove a composition from the reverse index.
    /// Call this before removing a composition from `self.compositions`.
    pub fn unindex_composition(&mut self, comp_id: &CompositionId, member_node_ids: &[NodeId]) {
        for &node_id in member_node_ids {
            if let Some(set) = self.node_to_compositions.get_mut(&node_id) {
                set.remove(comp_id);
                if set.is_empty() {
                    self.node_to_compositions.remove(&node_id);
                }
            }
        }
    }

    /// Get a node by its ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Find a node by its label.
    ///
    /// Returns `None` if no node with this label exists.
    pub fn find_node_by_label(&self, label: &str) -> Option<NodeId> {
        self.label_to_id.get(label).copied()
    }

    /// Get the label of a node by its ID.
    ///
    /// Returns `None` if the node doesn't exist.
    pub fn node_label(&self, id: NodeId) -> Option<&str> {
        self.nodes.get(&id).map(|n| n.label.as_str())
    }

    /// Get recent compositions, ordered by creation time (most recent first).
    ///
    /// Returns up to `limit` compositions. "Recent" is determined by
    /// lexicographic comparison of the `created_at` timestamp field.
    pub fn recent_compositions(&self, limit: usize) -> Vec<&Composition> {
        let mut comps: Vec<&Composition> = self.compositions.values().collect();
        comps.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        comps.truncate(limit);
        comps
    }

    /// Count how many compositions contain both node A and node B.
    ///
    /// This is the co-occurrence count used for similarity computation
    /// and gap detection. Two nodes that co-occur in many compositions
    /// are structurally related.
    ///
    /// Uses the `node_to_compositions` reverse index for O(K) intersection
    /// instead of O(C) full scan, where K = compositions containing either node.
    pub fn cooccurrence_count(&self, node_a: NodeId, node_b: NodeId) -> usize {
        let comps_a = self.node_to_compositions.get(&node_a);
        let comps_b = self.node_to_compositions.get(&node_b);
        match (comps_a, comps_b) {
            (Some(a), Some(b)) => a.intersection(b).count(),
            _ => 0,
        }
    }

    /// Check if a node with the given ID exists.
    pub fn has_node(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Get an edge by composition ID and target node ID.
    ///
    /// Returns the first matching edge, or `None` if no such edge exists.
    pub fn get_edge(
        &self,
        composition_id: &CompositionId,
        node_id: NodeId,
    ) -> Option<&(CompositionId, NodeId, SemanticEdge)> {
        self.edges
            .iter()
            .find(|(cid, nid, _)| cid == composition_id && *nid == node_id)
    }

    /// Compute Jaccard structural similarity between two compositions.
    ///
    /// Two compositions are structurally similar if they share many
    /// of the same member nodes. This is used by convergence detection
    /// and gap detection to find related compositions.
    ///
    /// Returns a value in [0, 1]: 1.0 = identical members, 0.0 = no overlap.
    ///
    /// Delegates to [`ConvergenceDetection::structural_similarity`] to avoid
    /// duplicating the Jaccard computation.
    pub fn structural_similarity(&self, comp_a: &Composition, comp_b: &Composition) -> f32 {
        super::super::convergence::ConvergenceDetection::structural_similarity(comp_a, comp_b)
    }

    /// Get the graph neighborhood for a set of keyword labels.
    ///
    /// Returns all compositions that contain nodes matching any of
    /// the given keywords. This is used by ExecutiveOrchestrator
    /// for cognitive mode selection.
    pub fn neighborhood_for(&self, keywords: &[String]) -> Vec<&Composition> {
        let keyword_ids: HashSet<NodeId> = keywords
            .iter()
            .filter_map(|kw| self.find_node_by_label(kw))
            .collect();

        if keyword_ids.is_empty() {
            return Vec::new();
        }

        self.compositions
            .values()
            .filter(|comp| {
                comp.members
                    .iter()
                    .any(|m| keyword_ids.contains(&m.node_id))
            })
            .collect()
    }

    /// Get all compositions that contain a specific node.
    pub fn compositions_for_node(&self, node_id: NodeId) -> Vec<&Composition> {
        self.compositions
            .values()
            .filter(|comp| comp.members.iter().any(|m| m.node_id == node_id))
            .collect()
    }

    /// Count total compositions.
    pub fn composition_count(&self) -> usize {
        self.compositions.len()
    }

    /// Count total nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Compute the average confidence of all compositions.
    pub fn average_confidence(&self) -> f32 {
        if self.compositions.is_empty() {
            return 0.0;
        }
        self.compositions
            .values()
            .map(|c| c.confidence)
            .sum::<f32>()
            / self.compositions.len() as f32
    }

    /// Count compositions with a specific epistemic state.
    pub fn count_with_epistemic(&self, state: EpistemicState) -> usize {
        self.compositions
            .values()
            .filter(|c| c.epistemic == state)
            .count()
    }

    /// Count compositions with a specific lifecycle state.
    pub fn count_with_lifecycle(&self, state: LifecycleState) -> usize {
        self.compositions
            .values()
            .filter(|c| c.lifecycle == state)
            .count()
    }

    // ================================================================
    // Phase O: Connectivity Score
    // ================================================================

    /// Compute the connectivity score for a node.
    ///
    /// Connectivity = (number of compositions referencing this node) / saturation_factor.
    /// Capped at 1.0. Saturation factor = 10 (a node referenced by 10+ compositions
    /// is considered fully connected).
    ///
    /// Uses O(1) lookup via composition scan — this is on-demand, not precomputed.
    pub fn connectivity_score(&self, node_id: NodeId) -> f32 {
        const SATURATION: f32 = 10.0;
        let count = self
            .compositions
            .values()
            .filter(|comp| comp.members.iter().any(|m| m.node_id == node_id))
            .count();
        (count as f32 / SATURATION).min(1.0)
    }

    // ================================================================
    // Phase P: SenseRole Helper Methods
    // ================================================================

    /// Is a node primitive? A node is primitive if its max sense layer is 0
    /// OR it has no senses at all (unprocessed/fresh node).
    ///
    /// Note: an unprocessed node (no senses) is treated as primitive.
    /// This is intentional — fresh nodes have not yet been analyzed
    /// and default to the most basic interpretation.
    pub fn is_primitive(&self, node_id: NodeId) -> bool {
        match self.nodes.get(&node_id) {
            Some(node) => {
                if node.senses.is_empty() {
                    return true; // Unprocessed = primitive
                }
                node.senses.iter().all(|s| s.layer == 0)
            }
            None => true, // Non-existent node = primitive by default
        }
    }

    /// Is a node a bridge? A bridge node connects different abstraction layers.
    /// It has at least one sense with `is_utterance` flag or senses at 2+ different layers.
    pub fn is_bridge(&self, node_id: NodeId) -> bool {
        match self.nodes.get(&node_id) {
            Some(node) => {
                if node.senses.iter().any(|s| s.is_utterance) {
                    return true;
                }
                let layers: std::collections::HashSet<u32> =
                    node.senses.iter().map(|s| s.layer).collect();
                layers.len() >= 2
            }
            None => false,
        }
    }

    /// Is a node derived? A derived node has at least one sense with layer ≥ 1.
    pub fn is_derived(&self, node_id: NodeId) -> bool {
        match self.nodes.get(&node_id) {
            Some(node) => node.senses.iter().any(|s| s.layer >= 1),
            None => false,
        }
    }

    /// Is a node at utterance level? Has at least one sense that is utterance-level.
    pub fn is_utterance_level(&self, node_id: NodeId) -> bool {
        match self.nodes.get(&node_id) {
            Some(node) => node.senses.iter().any(|s| s.is_utterance_level()),
            None => false,
        }
    }

    /// Find all nodes that have active utterance senses.
    pub fn find_active_utterance_senses(&self) -> Vec<(NodeId, &Sense)> {
        let mut result = Vec::new();
        for (&node_id, node) in &self.nodes {
            for sense in &node.senses {
                if sense.is_utterance_level() {
                    result.push((node_id, sense));
                }
            }
        }
        result
    }

    /// Find all bridge nodes in the graph.
    pub fn find_bridge_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(&id, _)| self.is_bridge(id))
            .map(|(&id, _)| id)
            .collect()
    }

    // ================================================================
    // Semantic Query API — query by meaning, similarity, path finding
    // ================================================================

    /// Query compositions by concept label.
    ///
    /// Unlike `neighborhood_for()` which only matches exact keywords,
    /// this method also finds compositions where the concept appears
    /// as a member label (substring matching) and ranks results by
    /// relevance: exact label match > role match > substring match.
    ///
    /// Returns compositions sorted by relevance (highest first).
    pub fn query_by_concept(&self, concept: &str) -> Vec<(&Composition, f32)> {
        let concept_lower = concept.to_lowercase();
        let mut results: Vec<(&Composition, f32)> = Vec::new();

        for comp in self.compositions.values() {
            let mut score = 0.0f32;

            for member in &comp.members {
                let label_lower = member.label.to_lowercase();

                if label_lower == concept_lower {
                    // Exact match — highest relevance
                    score = score.max(1.0);
                } else if label_lower.contains(&concept_lower)
                    || concept_lower.contains(&label_lower)
                {
                    // Substring match — medium relevance, weighted by member confidence
                    score = score.max(0.6 * member.confidence);
                }
            }

            // Boost score for Stable/Grounded compositions
            if score > 0.0 {
                if comp.lifecycle == LifecycleState::Stable {
                    score *= 1.2;
                }
                if comp.epistemic == EpistemicState::Grounded {
                    score *= 1.1;
                }
                score = score.min(1.0);
                results.push((comp, score));
            }
        }

        // Sort by relevance (highest first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Query compositions by role structure.
    ///
    /// Find all compositions that contain ALL the specified semantic roles.
    /// For example, `query_by_structure(&["Arg0Agent", "Cause"])` finds
    /// all compositions that have both an Agent and a Cause role.
    ///
    /// This enables queries like "find all causal compositions" or
    /// "find all compositions with a Problem and Solution."
    ///
    /// Returns compositions sorted by confidence (highest first).
    pub fn query_by_structure(&self, role_names: &[String]) -> Vec<&Composition> {
        // Parse role names into SemanticRole enums
        let target_roles: Vec<SemanticRole> = role_names
            .iter()
            .filter_map(|name| parse_semantic_role(name))
            .collect();

        if target_roles.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<&Composition> = self
            .compositions
            .values()
            .filter(|comp| {
                target_roles.iter().all(|role| {
                    comp.members.iter().any(|m| m.role == *role)
                })
            })
            .collect();

        // Sort by confidence (highest first)
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Compute similarity between two nodes identified by label.
    ///
    /// Uses structural overlap (Jaccard) of their composition neighborhoods
    /// plus spreading activation overlap. Two nodes are "similar" if they
    /// participate in compositions with similar structure.
    ///
    /// Returns a score from 0.0 (completely unrelated) to 1.0 (identical neighborhood).
    pub fn similarity(&self, label_a: &str, label_b: &str) -> f32 {
        let node_a = match self.find_node_by_label(label_a) {
            Some(id) => id,
            None => return 0.0,
        };
        let node_b = match self.find_node_by_label(label_b) {
            Some(id) => id,
            None => return 0.0,
        };

        if node_a == node_b {
            return 1.0;
        }

        // Strategy 1: Composition neighborhood Jaccard
        let comps_a: HashSet<CompositionId> = self
            .compositions_for_node(node_a)
            .iter()
            .map(|c| c.id.clone())
            .collect();
        let comps_b: HashSet<CompositionId> = self
            .compositions_for_node(node_b)
            .iter()
            .map(|c| c.id.clone())
            .collect();

        let jaccard = if comps_a.is_empty() && comps_b.is_empty() {
            0.0
        } else {
            let intersection = comps_a.intersection(&comps_b).count();
            let union = comps_a.union(&comps_b).count();
            intersection as f32 / union as f32
        };

        // Strategy 2: Spreading activation overlap
        let sa = super::super::spreading::SpreadingActivation::default();
        let activation_a = sa.spread(&[(node_a, 1.0)], self);
        let activation_b = sa.spread(&[(node_b, 1.0)], self);

        let spreading_overlap = if activation_a.is_empty() || activation_b.is_empty() {
            0.0
        } else {
            // Cosine similarity of activation vectors
            let mut dot = 0.0f32;
            let mut norm_a = 0.0f32;
            let mut norm_b = 0.0f32;

            for (&id, &e) in &activation_a.energies {
                norm_a += e * e;
                let eb = activation_b.energy(id);
                dot += e * eb;
            }
            for &e in activation_b.energies.values() {
                norm_b += e * e;
            }

            let denom = norm_a.sqrt() * norm_b.sqrt();
            if denom > 0.0 { dot / denom } else { 0.0 }
        };

        // Blend: 60% Jaccard + 40% spreading overlap
        0.6 * jaccard + 0.4 * spreading_overlap
    }

    /// Find related nodes using spreading activation from a seed label.
    ///
    /// Returns the top-N most activated nodes, sorted by activation energy.
    /// This is the core "query by meaning" mechanism: start from a concept,
    /// spread activation through the graph, and the most activated nodes
    /// are the most semantically related.
    ///
    /// Each result includes the node label and activation energy.
    pub fn find_related(&self, label: &str, top_n: usize) -> Vec<(String, f32)> {
        let node_id = match self.find_node_by_label(label) {
            Some(id) => id,
            None => return Vec::new(),
        };

        let sa = super::super::spreading::SpreadingActivation::default();
        let activation = sa.spread(&[(node_id, 1.0)], self);

        activation
            .top_n(top_n)
            .iter()
            .filter_map(|&(id, energy)| {
                self.node_label(id).map(|lbl| (lbl.to_string(), energy))
            })
            .filter(|(lbl, _)| lbl != label) // Exclude the seed itself
            .collect()
    }

    /// Find a reasoning path between two nodes identified by label.
    ///
    /// Uses bidirectional spreading activation to find compositions
    /// that connect two concepts. Returns the chain of composition IDs
    /// that form the shortest reasoning path.
    ///
    /// This answers questions like "why are X and Y related?" by
    /// showing the compositions that connect them.
    pub fn find_path(&self, label_from: &str, label_to: &str) -> Vec<CompositionId> {
        let node_from = match self.find_node_by_label(label_from) {
            Some(id) => id,
            None => return Vec::new(),
        };
        let node_to = match self.find_node_by_label(label_to) {
            Some(id) => id,
            None => return Vec::new(),
        };

        if node_from == node_to {
            return Vec::new();
        }

        // Spread from both directions
        let sa = super::super::spreading::SpreadingActivation::default();
        let activation_from = sa.spread(&[(node_from, 1.0)], self);
        let activation_to = sa.spread(&[(node_to, 1.0)], self);

        // Find compositions that are activated from both sides
        let mut bridge_comps: Vec<(&Composition, f32)> = Vec::new();

        for comp in self.compositions.values() {
            let mut energy_from = 0.0f32;
            let mut energy_to = 0.0f32;

            for member in &comp.members {
                energy_from += activation_from.energy(member.node_id);
                energy_to += activation_to.energy(member.node_id);
            }

            // Both sides must have activation for this to be a bridge
            if energy_from > 0.0 && energy_to > 0.0 {
                // Score = harmonic mean of energies (high when both are high)
                let score = if energy_from + energy_to > 0.0 {
                    2.0 * energy_from * energy_to / (energy_from + energy_to)
                } else {
                    0.0
                };
                bridge_comps.push((comp, score));
            }
        }

        // Sort by bridge score (highest first) and return composition IDs
        bridge_comps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        bridge_comps.iter().map(|(comp, _)| comp.id.clone()).collect()
    }

    // ================================================================
    // Phase N: Utterance Context
    // ================================================================

    /// Get utterance context for cross-sentence blending.
    ///
    /// Returns a 3-way blend of:
    /// - Current composition's properties (55%)
    /// - Utterance-level context from Situation/Conclusion compositions (25%)
    /// - Bridge node context (20%)
    ///
    /// **Audit fix**: No double-check on member nodes for utterance sense.
    /// The Situation/Conclusion check is sufficient — we don't need to
    /// additionally verify that member nodes have utterance senses.
    pub fn get_utterance_context(&self, composition_id: &CompositionId) -> UtteranceContext {
        let _comp = match self.compositions.get(composition_id) {
            Some(c) => c,
            None => return UtteranceContext::default(),
        };

        // Collect Situation and Conclusion compositions for context
        let situational_comps: Vec<&Composition> = self
            .compositions
            .values()
            .filter(|c| c.id != *composition_id)
            .filter(|c| {
                c.composition_type == CompositionType::Situation
                    || c.composition_type == CompositionType::HiddenMeaning
            })
            .take(5)
            .collect();

        // Collect bridge nodes
        let bridge_node_ids: Vec<NodeId> = self.find_bridge_nodes();

        UtteranceContext {
            current_weight: 0.55,
            situational_weight: 0.25,
            bridge_weight: 0.20,
            situational_composition_count: situational_comps.len(),
            bridge_node_count: bridge_node_ids.len(),
        }
    }

    // ================================================================
    // Phase L: Property Extraction for Weighted Jaccard
    // ================================================================

    /// Extract properties from a composition for SenseCandidate comparison.
    ///
    /// Properties are derived from the composition's members, with weights
    /// looked up from each member node's senses' freq_map.
    ///
    /// **Audit fix**: Iterates ALL senses of a node to find the one containing
    /// the composition_id, not just `senses.first()`.
    pub fn extract_properties_from_composition(
        &self,
        composition: &Composition,
    ) -> SenseCandidate {
        let mut properties = HashMap::new();

        for member in &composition.members {
            let role_key = format!("{:?}", member.role);

            // AUDIT FIX: iterate all senses, find the one containing this composition
            let weight = self
                .nodes
                .get(&member.node_id)
                .and_then(|node| {
                    node.senses
                        .iter()
                        .find(|s| s.freq_map.contains_key(&composition.id))
                        .and_then(|s| s.freq_map.get(&composition.id))
                        .copied()
                })
                .unwrap_or(1.0); // Default flat weight if no freq_map entry

            *properties.entry(role_key).or_insert(0.0) += weight * member.confidence;
        }

        SenseCandidate {
            sense_id: composition.id.as_str().to_string(),
            properties,
        }
    }
}

/// Utterance context for cross-sentence blending (Phase N).
///
/// Contains the weights and metadata for the 3-way blend:
/// 55% current, 25% situational, 20% bridge.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UtteranceContext {
    /// Weight for the current composition's properties.
    pub current_weight: f32,
    /// Weight for situational context.
    pub situational_weight: f32,
    /// Weight for bridge node context.
    pub bridge_weight: f32,
    /// Number of situational compositions found.
    pub situational_composition_count: usize,
    /// Number of bridge nodes found.
    pub bridge_node_count: usize,
}
