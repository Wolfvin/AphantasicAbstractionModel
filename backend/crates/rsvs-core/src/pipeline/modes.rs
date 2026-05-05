//! Appraise and Relate modes — RSVS v6.0
//!
//! Contains `appraise()` and `relate()` methods.
//! v6.0: relate() now includes structural similarity from compositions.

use super::Rsvs;
use crate::types::NodeId;
use rayon::prelude::*;

// -----------------------------------------------------------------------
// AppraiseResult — v6.0 appraise mode output
// -----------------------------------------------------------------------

/// Appraise mode output: agree/disagree percentages, verdict, and evidence.
#[derive(Debug, Clone)]
pub struct AppraiseResult {
    /// Percentage of tokens found in the graph.
    pub agree_pct: f32,
    /// Percentage of tokens NOT found in the graph.
    pub disagree_pct: f32,
    /// Verdict: "consistent", "partial", or "novel".
    pub verdict: String,
    /// Per-token evidence: (token, confidence) for matched tokens.
    pub evidence: Vec<(String, f32)>,
}

// -----------------------------------------------------------------------
// RelateResult — v6.0 relate mode output
// -----------------------------------------------------------------------

/// Relate mode output: related nodes and edges by overlap scoring.
#[derive(Debug, Clone)]
pub struct RelateResult {
    /// Related nodes: (node_id, overlap_score) sorted by score descending.
    pub related_nodes: Vec<(NodeId, f32)>,
    /// Related edges: (from, to, weight) sorted by weight descending.
    pub related_edges: Vec<(NodeId, NodeId, f32)>,
    /// Structural relationships found (v6.0): (node_id, structural_similarity).
    pub structural_relations: Vec<(NodeId, f32)>,
}

impl Rsvs {
    /// Evaluate text against the graph.
    pub fn appraise(&self, text: &str) -> AppraiseResult {
        let tokens = crate::attention::tokenize(text);
        if tokens.is_empty() {
            return AppraiseResult {
                agree_pct: 0.0,
                disagree_pct: 100.0,
                verdict: "novel".to_string(),
                evidence: vec![],
            };
        }

        let total = tokens.len() as f32;
        let mut found = 0usize;
        let mut evidence: Vec<(String, f32)> = Vec::new();

        for token in &tokens {
            if let Some(&id) = self.token_to_id.get(token.as_str()) {
                found += 1;
                let conf = self.autonomy.confidence(id).unwrap_or(0.0);
                evidence.push((token.clone(), conf));
            }
        }

        let agree_pct = (found as f32 / total) * 100.0;
        let disagree_pct = 100.0 - agree_pct;

        let verdict = if agree_pct >= 80.0 {
            "consistent"
        } else if agree_pct >= 40.0 {
            "partial"
        } else {
            "novel"
        }
        .to_string();

        evidence.sort_by(|a, b| b.1.total_cmp(&a.1));

        AppraiseResult {
            agree_pct,
            disagree_pct,
            verdict,
            evidence,
        }
    }

    /// Find nodes and edges related to the given concept.
    ///
    /// v7.2: Now uses SpreadingActivation for structural relation discovery.
    /// Spreading activation follows composition edges (structural meaning
    /// connections) to find related nodes that pure Jaccard overlap misses.
    /// This is the structural equivalent of semantic priming in cognitive
    /// science — activating "raja" spreads energy to "tahta_tertinggi",
    /// "kerajaan", and then onward to their composition neighbors.
    ///
    /// v6.0: Also includes structural relations based on composition overlap.
    pub fn relate(&self, concept: &str) -> Option<RelateResult> {
        let concept_id = *self.token_to_id.get(concept)?;

        // Find related nodes by Jaccard similarity (parallelized with rayon)
        let node_ids: Vec<NodeId> = self.graph.nodes.keys().copied().collect();
        let related_nodes: Vec<(NodeId, f32)> = node_ids
            .par_iter()
            .filter(|&&other_id| other_id != concept_id)
            .filter_map(|&other_id| {
                let jaccard = self.graph.jaccard_atom_sets(concept_id, other_id);
                if jaccard > 0.0 {
                    Some((other_id, jaccard))
                } else {
                    None
                }
            })
            .collect();

        let mut related_nodes = related_nodes;
        related_nodes.sort_by(|a, b| b.1.total_cmp(&a.1));
        related_nodes.truncate(20);

        // v7.2: Use SpreadingActivation for structural relation discovery
        // Spreading follows composition edges (structural meaning), not just
        // co-occurrence. This captures indirect relationships that Jaccard misses.
        let activation_result = self.spreading_activation.targeted_spread(
            concept_id,
            1.0,
            &self.senses,
            &self.composition_index,
        );

        // Merge spreading-activated nodes into structural relations
        let mut structural_relations: Vec<(NodeId, f32)> = Vec::new();

        // v6.0: Direct structural similarity from composition overlap
        if let Some(sm_concept) = self.senses.get(&concept_id) {
            for (&other_id, sm_other) in &self.senses {
                if other_id == concept_id {
                    continue;
                }
                let sim = self
                    .graph
                    .structural_similarity(concept_id, other_id, sm_concept, sm_other);
                if sim.structural_similarity > 0.0 {
                    structural_relations.push((other_id, sim.structural_similarity));
                }
            }
        }

        // v7.2: Add spreading-activated nodes that aren't already in structural_relations
        let existing_structural: std::collections::HashSet<NodeId> =
            structural_relations.iter().map(|(id, _)| *id).collect();
        for (node_id, energy) in &activation_result.activated {
            if *node_id == concept_id || existing_structural.contains(node_id) {
                continue;
            }
            // Normalize energy to [0, 1] range for compatibility with structural_similarity
            let normalized = (*energy).min(1.0);
            if normalized > 0.01 {
                structural_relations.push((*node_id, normalized));
            }
        }

        structural_relations.sort_by(|a, b| b.1.total_cmp(&a.1));
        structural_relations.truncate(20);

        // Find related edges involving this concept
        let mut related_edges: Vec<(NodeId, NodeId, f32)> = Vec::new();

        for e in self.graph.edges_from(concept_id) {
            related_edges.push((e.from, e.to, e.weight));
        }

        for (&from_id, edges) in &self.graph.edges {
            if from_id == concept_id {
                continue;
            }
            for e in edges {
                if e.to == concept_id {
                    related_edges.push((e.from, e.to, e.weight));
                }
            }
        }

        for &(node_id, _) in &related_nodes {
            for e in self.graph.edges_from(node_id) {
                if !related_edges
                    .iter()
                    .any(|(f, t, _)| *f == e.from && *t == e.to)
                {
                    related_edges.push((e.from, e.to, e.weight));
                }
            }
        }

        related_edges.sort_by(|a, b| b.2.total_cmp(&a.2));
        related_edges.truncate(30);

        Some(RelateResult {
            related_nodes,
            related_edges,
            structural_relations,
        })
    }
}
