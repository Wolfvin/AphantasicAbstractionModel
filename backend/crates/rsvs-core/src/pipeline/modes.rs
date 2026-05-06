//! Appraise and Relate modes — RSVS v7.3
//!
//! Contains `appraise()` and `relate()` methods.
//! v7.3: appraise() uses structural sense matching instead of shallow
//! token-presence check. Two concepts agree not just because they share
//! tokens, but because they share compositions (structural meaning).
//! v6.0: relate() now includes structural similarity from compositions.

use super::Rsvs;
use crate::types::NodeId;
use rayon::prelude::*;
use std::collections::HashSet;

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
    /// v8.2: Convergent nodes that contributed to the appraise score.
    /// Each entry is (label, convergence_boost) indicating how much
    /// a structurally equivalent node boosted the corroboration.
    pub convergence_info: Vec<(String, f32)>,
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
    /// Evaluate text against the graph using structural sense matching.
    ///
    /// v7.3: Instead of just checking if a token EXISTS in the graph
    /// (shallow token-presence check), we now evaluate whether the token's
    /// structural meaning (its compositions) is consistent with the graph.
    ///
    /// The evaluation works in two layers:
    /// 1. **Token presence**: Does the token exist at all? (baseline)
    /// 2. **Structural match**: For tokens with compositional senses, how
       ///    well do their compositions align with the other tokens in the text?
    ///
    /// A token that exists but whose compositions conflict with the rest of
    /// the text scores lower than one whose compositions are corroborated.
    /// This prevents false positives where "bank" (river) is counted as
    /// "consistent" when the text is about finance.
    ///
    /// Scoring:
    /// - Token exists but has no compositional sense → confidence × 0.7
    ///   (we know the token but can't verify structural meaning)
    /// - Token exists with compositional sense, compositions corroborated →
    ///   confidence × 1.0 (full structural agreement)
    /// - Token exists with compositional sense, compositions partially match →
    ///   confidence × (0.4 + 0.6 × overlap_ratio)
    /// - Token not found → 0.0
    pub fn appraise(&self, text: &str) -> AppraiseResult {
        let tokens = crate::attention::tokenize(text);
        if tokens.is_empty() {
            return AppraiseResult {
                agree_pct: 0.0,
                disagree_pct: 100.0,
                verdict: "novel".to_string(),
                evidence: vec![],
                convergence_info: vec![],
            };
        }

        let total = tokens.len() as f32;
        let mut structural_score = 0.0f32;
        let mut evidence: Vec<(String, f32)> = Vec::new();
        // v8.2: Track convergent nodes that contributed to appraise scoring
        let mut convergence_seen: std::collections::HashMap<String, f32> = std::collections::HashMap::new();

        // Build the set of all node IDs mentioned in the text for
        // cross-referencing compositions
        let text_node_ids: std::collections::HashSet<NodeId> = tokens
            .iter()
            .filter_map(|t| self.token_to_id.get(t.as_str()).copied())
            .collect();

        for token in &tokens {
            if let Some(&id) = self.token_to_id.get(token.as_str()) {
                let base_conf = self.autonomy.confidence(id).unwrap_or(0.0);

                // v7.3: Structural sense matching
                // Check if this token has compositional senses whose
                // compositions are corroborated by other tokens in the text
                // v8.1: Also consider convergent nodes' senses as corroborating
                // evidence — if "dog" converges with "anjing", and "anjing"'s
                // compositions match the text, that counts as corroboration for "dog".
                let structural_multiplier = if let Some(sm) = self.senses.get(&id) {
                    let best_sense_match = sm.senses.iter().map(|sense| {
                        if sense.compositions.is_empty() {
                            // No compositions — can't verify structurally,
                            // but token exists. Use reduced confidence.
                            0.7
                        } else {
                            // Count how many composition node_ids appear in
                            // the text's node set
                            let comp_node_ids: Vec<NodeId> =
                                sense.compositions.iter().map(|c| c.node_id).collect();
                            let overlap = comp_node_ids
                                .iter()
                                .filter(|cid| text_node_ids.contains(cid))
                                .count();
                            let overlap_ratio = overlap as f32 / comp_node_ids.len().max(1) as f32;

                            // Full corroboration → 1.0, partial → 0.4–1.0
                            0.4 + 0.6 * overlap_ratio
                        }
                    }).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.7);

                    // v8.1 → v8.2: Check convergent nodes for additional corroboration
                    // If this node has structural_equivalence links, check if
                    // the convergent nodes' compositions corroborate the text
                    // v8.2: Also track which convergent nodes contributed
                    let (conv_boost, conv_contributors) = self.convergence_boost_for_appraise(id, &text_node_ids);
                    for (label, boost) in conv_contributors {
                        convergence_seen.entry(label)
                            .and_modify(|existing| *existing = existing.max(boost))
                            .or_insert(boost);
                    }

                    (best_sense_match + conv_boost).min(1.0)
                } else {
                    // No sense manager — token exists but no sense data
                    0.7
                };

                let scored = base_conf * structural_multiplier;
                structural_score += scored;
                evidence.push((token.clone(), scored));
            }
            // Token not found → contributes 0.0 (implicit)
        }

        let agree_pct = (structural_score / total) * 100.0;
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

        // v8.2: Convert convergence tracking map to sorted vec
        let mut convergence_info: Vec<(String, f32)> = convergence_seen.into_iter().collect();
        convergence_info.sort_by(|a, b| b.1.total_cmp(&a.1));

        AppraiseResult {
            agree_pct,
            disagree_pct,
            verdict,
            evidence,
            convergence_info,
        }
    }

    /// Find nodes and edges related to the given concept.
    ///
    /// v7.3: SpreadingActivation is now FUSED into the relate score instead
    /// of being added as a separate list. The fusion formula is:
    ///   score = 0.6 × structural_similarity + 0.4 × spreading_energy
    ///
    /// This produces a single, unified ranking that combines:
    /// - Structural similarity (direct composition overlap)
    /// - Spreading activation (indirect connections via composition edges)
    ///
    /// Nodes with both high structural overlap AND strong spreading activation
    /// rank higher than nodes with only one signal. This is the structural
    /// equivalent of combining explicit memory with associative priming.
    ///
    /// v7.2: SpreadingActivation for structural relation discovery.
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

        // v7.3: FUSED SpreadingActivation + structural similarity
        // Instead of keeping them as separate lists, we combine them into
        // a single score: 0.6 × structural + 0.4 × spreading_energy
        let activation_result = self.spreading_activation.targeted_spread(
            concept_id,
            1.0,
            &self.senses,
            &self.composition_index,
        );

        // Build a map of node_id → spreading_energy for O(1) lookup
        let spreading_map: std::collections::HashMap<NodeId, f32> = activation_result
            .activated
            .iter()
            .filter(|(id, _)| *id != concept_id)
            .map(|(id, energy)| (*id, energy.min(1.0)))
            .collect();

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
                if sim.structural_similarity > 0.0 || spreading_map.contains_key(&other_id) {
                    // v7.3: Fuse structural + spreading scores
                    let structural_score = sim.structural_similarity;
                    let spreading_score = spreading_map.get(&other_id).copied().unwrap_or(0.0);
                    let fused = 0.6 * structural_score + 0.4 * spreading_score;
                    if fused > 0.01 {
                        structural_relations.push((other_id, fused));
                    }
                }
            }
        }

        // Add nodes that have spreading energy but no structural similarity
        let existing_structural: std::collections::HashSet<NodeId> =
            structural_relations.iter().map(|(id, _)| *id).collect();
        for (node_id, energy) in &spreading_map {
            if existing_structural.contains(node_id) {
                continue;
            }
            // These nodes have only spreading energy (no direct structural overlap)
            let fused = 0.4 * energy;
            if fused > 0.01 {
                structural_relations.push((*node_id, fused));
            }
        }

        structural_relations.sort_by(|a, b| b.1.total_cmp(&a.1));
        structural_relations.truncate(20);

        // v8.1: Convergence fusion — promote convergent nodes in the results.
        // If the queried concept has LanguageLinks, those convergent nodes
        // are structurally equivalent and should appear prominently in
        // the relate results. We boost their score to reflect this.
        if let Some(node) = self.graph.get_node(concept_id) {
            for link in &node.language_links {
                if link.link_type != "structural_equivalence" {
                    continue;
                }
                let conv_id = link.target_id;
                // Check if already in structural_relations
                if let Some(entry) = structural_relations.iter_mut().find(|(id, _)| *id == conv_id) {
                    // Boost existing score — convergence is strong evidence
                    entry.1 = (entry.1 + 0.5).min(1.0);
                } else {
                    // Add convergent node with a convergence-based score
                    let jaccard = self.graph.jaccard_atom_sets(concept_id, conv_id);
                    if jaccard > 0.0 {
                        structural_relations.push((conv_id, (jaccard + 0.5).min(1.0)));
                    } else {
                        // Even with 0 jaccard, convergence is meaningful
                        structural_relations.push((conv_id, 0.5));
                    }
                }
            }
            // Re-sort and re-truncate after convergence boost
            structural_relations.sort_by(|a, b| b.1.total_cmp(&a.1));
            structural_relations.truncate(20);
        }

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

    /// v8.1: Compute a convergence boost for appraise mode.
    ///
    /// When a node has LanguageLinks (structural_equivalence), we check
    /// if its convergent partners' compositions corroborate the text.
    /// If they do, this provides additional evidence that the node is
    /// structurally consistent — even if the node's own compositions
    /// don't directly overlap with the text.
    ///
    /// Returns a boost value in [0.0, 0.3] — capped to prevent
    /// convergence from overwhelming direct composition evidence.
    /// v8.2: Returns (boost_value, contributors) where contributors is a list
    /// of (label, boost) pairs for convergent nodes that corroborated the text.
    fn convergence_boost_for_appraise(
        &self,
        node_id: NodeId,
        text_node_ids: &HashSet<NodeId>,
    ) -> (f32, Vec<(String, f32)>) {
        let node = match self.graph.get_node(node_id) {
            Some(n) => n,
            None => return (0.0, vec![]),
        };

        let mut best_boost = 0.0f32;
        let mut contributors: Vec<(String, f32)> = Vec::new();

        for link in &node.language_links {
            if link.link_type != "structural_equivalence" {
                continue;
            }

            let conv_id = link.target_id;
            let conv_label = match self.graph.get_node(conv_id) {
                Some(n) => n.label.clone(),
                None => continue,
            };
            let conv_sm = match self.senses.get(&conv_id) {
                Some(sm) => sm,
                None => continue,
            };

            // Check convergent node's composition overlap with text
            let conv_best = conv_sm.senses.iter()
                .filter(|s| !s.compositions.is_empty())
                .map(|sense| {
                    let comp_ids: Vec<NodeId> = sense.compositions.iter().map(|c| c.node_id).collect();
                    let overlap = comp_ids.iter().filter(|id| text_node_ids.contains(id)).count();
                    overlap as f32 / comp_ids.len().max(1) as f32
                })
                .fold(0.0f32, f32::max);

            // Convert convergent corroboration to a boost (capped at 0.3)
            let boost = (conv_best * 0.3).min(0.3);
            if boost > 0.01 {
                contributors.push((conv_label, boost));
            }
            best_boost = best_boost.max(boost);
        }

        (best_boost, contributors)
    }
}
