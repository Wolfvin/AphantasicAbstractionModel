//! Query pipeline — RSVS v8.1 Compositional Architecture
//!
//! Contains `query()`, `similarity()`, `structural_similarity()`, and
//! `substitution_analysis()` methods.
//!
//! v8.1: query() now fuses convergent nodes' senses into scoring.
//! When a queried node has LanguageLinks (structural_equivalence),
//! the convergent nodes' scored atoms are included with a discount
//! factor based on the convergence overlap score. This means querying
//! "dog" will also surface senses from "anjing" if they converge.
//!
//! v7.3: query() now integrates ParadigmRouter for adaptive traversal
//! escalation and ThinkingToggle for depth adjustment.

use super::Rsvs;
use crate::types::{NodeId, SenseId};

// -----------------------------------------------------------------------
// QueryResult — output of a context-aware query
// -----------------------------------------------------------------------

/// Output of a context-aware query (v6.0).
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Index of the active sense used for the query.
    pub active_sense_idx: usize,
    /// Number of contexts in the active sense.
    pub active_sense_n: usize,
    /// Scored atoms: (label, score) sorted by score descending.
    pub scored_atoms: Vec<(String, f32)>,
    /// Layer of the active sense (v6.0).
    pub layer: u32,
    /// Grounding score of the active sense (v6.0).
    pub grounding_score: f32,
    /// Compositions of the active sense (v6.0).
    pub compositions: Vec<(String, SenseId)>,
    /// v8.1: Convergent nodes that contributed to this query result.
    /// Each entry is (label, convergence_discount) where discount
    /// indicates how strongly the convergent node's senses were
    /// incorporated (1.0 = full, 0.0 = none).
    pub convergence_contributors: Vec<(String, f32)>,
}

impl Rsvs {
    /// Context-aware lookup for a concept.
    ///
    /// v8.1: Convergence fusion — when a node has LanguageLinks
    /// (structural_equivalence), the convergent nodes' scored atoms are
    /// fused into the result with a convergence discount. This is the
    /// proof that the system truly "understands" equivalence: querying
    /// "dog" automatically includes senses from "anjing" if they
    /// structurally converge.
    ///
    /// The convergence discount is computed from the overlap score
    /// stored in the LanguageLink. A node converging with score 0.8
    /// contributes its atoms at 0.8 × their original score.
    ///
    /// v7.3: Now integrates ParadigmRouter for adaptive traversal escalation.
    /// For simple queries (high confidence, single sense), uses direct lookup.
    /// For complex queries (low confidence, multiple senses), escalates to
    /// deeper traversal with ThinkingToggle adjustment.
    pub fn query(&self, concept: &str, query_context: &str) -> Option<QueryResult> {
        let concept_id = *self.token_to_id.get(concept)?;
        let sense_mgr = self.senses.get(&concept_id)?;

        let query_tokens = crate::attention::tokenize(query_context);
        let query_atoms: Vec<NodeId> = query_tokens
            .iter()
            .filter_map(|t| self.token_to_id.get(t.as_str()).copied())
            .collect();

        let active_sense_idx = sense_mgr.lazy_lookup(&query_atoms).or_else(|| {
            if sense_mgr.sense_count() > 0 {
                Some(0)
            } else {
                None
            }
        })?;

        let sense = sense_mgr.get_sense(active_sense_idx)?;

        let tau = self.config.sense.tau_core;
        let core = sense.core(tau);

        // v7.3: Use ParadigmRouter + ThinkingToggle for adaptive scoring
        let confidence = sense.grounding.score();
        let signal = crate::thinking::ComplexitySignal {
            n_context_atoms: query_atoms.len(),
            n_senses: sense_mgr.senses.len(),
            target_layer: sense.layer,
            is_compositional: sense.is_compositional(),
            domain_complexity: 0.0,
        };

        // Route to optimal paradigm
        let paradigm = self.paradigm_router.route(confidence, &signal, self.config.current_domain);

        // v7.3: Apply ThinkingToggle to determine scoring depth
        let thinking_mode = self.thinking_toggle.classify(&signal);

        // Determine if we should use deep scoring based on paradigm + thinking mode
        let use_deep_scoring = paradigm >= crate::paradigm::TraversalParadigm::Standard
            || thinking_mode == crate::thinking::ThinkingMode::Thinking;

        let mut scored: Vec<(String, f32)> = core
            .iter()
            .filter_map(|&atom_id| {
                let label = self.graph.get_node(atom_id)?.label.clone();
                let freq = sense.freq(atom_id);

                if use_deep_scoring {
                    // v7.3: Deep scoring uses P(a|S,q) from compositions
                    let edge_score = self
                        .graph
                        .edges_from(atom_id)
                        .iter()
                        .filter(|e| query_atoms.contains(&e.to))
                        .map(|e| e.weight)
                        .fold(0.0f32, f32::max);
                    let comp_ref = crate::types::CompositionRef::new(atom_id, 0);
                    let p_score = sense.p_a_given_s_q(&comp_ref, if edge_score > 0.0 { edge_score } else { 1.0 });
                    let score = if p_score > 0.0 { p_score } else { freq };
                    Some((label, score))
                } else {
                    // Simple scoring for Direct/Shallow paradigms
                    let edge_score = self
                        .graph
                        .edges_from(atom_id)
                        .iter()
                        .filter(|e| query_atoms.contains(&e.to))
                        .map(|e| e.weight)
                        .fold(0.0f32, f32::max);
                    let score = if edge_score > 0.0 {
                        freq * edge_score
                    } else {
                        freq
                    };
                    Some((label, score))
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        // v8.1: Convergence fusion — include scored atoms from convergent nodes.
        // If this node has LanguageLinks with type "structural_equivalence",
        // the convergent nodes likely represent the same concept in a different
        // surface form. We fuse their atoms into the result with a discount
        // factor, so querying "dog" also surfaces "anjing"'s senses.
        //
        // The discount is proportional to the convergence overlap. Since we
        // don't store the overlap score in the LanguageLink directly, we
        // compute a proxy: the Jaccard similarity of the two nodes' atom sets.
        // This gives a reasonable approximation of structural equivalence.
        let mut convergence_contributors: Vec<(String, f32)> = Vec::new();
        if let Some(node) = self.graph.get_node(concept_id) {
            for link in &node.language_links {
                if link.link_type != "structural_equivalence" {
                    continue;
                }
                let conv_id = link.target_id;
                let conv_label = match self.graph.get_node(conv_id) {
                    Some(n) => n.label.clone(),
                    None => continue,
                };

                // v11.0: Use structural_similarity for convergence discount instead of deprecated jaccard_atom_sets
                let conv_discount = if let Some(sm_concept) = self.senses.get(&concept_id) {
                    if let Some(sm_conv) = self.senses.get(&conv_id) {
                        self.graph.structural_similarity(concept_id, conv_id, sm_concept, sm_conv).structural_similarity
                    } else {
                        0.3 // No senses for convergent node — moderate discount
                    }
                } else {
                    0.3 // No senses for concept — moderate discount
                };
                if conv_discount < 0.1 {
                    continue; // Too weak — don't fuse
                }

                // Score convergent node's atoms
                if let Some(conv_sm) = self.senses.get(&conv_id) {
                    let conv_sense_idx = conv_sm.lazy_lookup(&query_atoms)
                        .or_else(|| if conv_sm.sense_count() > 0 { Some(0) } else { None });
                    if let Some(idx) = conv_sense_idx {
                        if let Some(conv_sense) = conv_sm.get_sense(idx) {
                            let conv_core = conv_sense.core(self.config.sense.tau_core);
                            for &atom_id in &conv_core {
                                let label = match self.graph.get_node(atom_id) {
                                    Some(n) => n.label.clone(),
                                    None => continue,
                                };
                                // Skip atoms already in the primary result
                                if scored.iter().any(|(l, _)| l == &label) {
                                    continue;
                                }
                                let freq = conv_sense.freq(atom_id);
                                let fused_score = freq * conv_discount;
                                if fused_score > 0.01 {
                                    scored.push((label, fused_score));
                                }
                            }
                        }
                    }
                }

                convergence_contributors.push((conv_label, conv_discount));
            }
        }

        // Re-sort after convergence fusion
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Build composition labels (v6.0)
        let compositions: Vec<(String, crate::types::SenseId)> = sense
            .compositions
            .iter()
            .filter_map(|comp| {
                let label = self.graph.get_node(comp.node_id)?.label.clone();
                Some((label, comp.sense_id))
            })
            .collect();

        Some(QueryResult {
            active_sense_idx,
            active_sense_n: sense.context_count(),
            scored_atoms: scored,
            layer: sense.layer,
            grounding_score: sense.grounding.score(),
            compositions,
            convergence_contributors,
        })
    }

    /// Compute flat similarity between two concepts (v4 compat).
    pub fn similarity(&self, a: &str, b: &str) -> Option<crate::graph::SimilarityResult> {
        let id_a = *self.token_to_id.get(a)?;
        let id_b = *self.token_to_id.get(b)?;
        Some(self.graph.similarity(id_a, id_b))
    }

    /// Compute structural similarity between two concepts at the sense level (v6.0).
    ///
    /// This is the core of RSVS v6.0. Two concepts are structurally similar
    /// if their senses share compositions. This captures WHY they're related,
    /// not just THAT they're related.
    ///
    /// Example:
    ///   raja and ratu share 2/3 compositions → structural_similarity = 0.667
    pub fn structural_similarity(
        &self,
        a: &str,
        b: &str,
    ) -> Option<crate::graph::StructuralSimResult> {
        let id_a = *self.token_to_id.get(a)?;
        let id_b = *self.token_to_id.get(b)?;
        let sm_a = self.senses.get(&id_a)?;
        let sm_b = self.senses.get(&id_b)?;
        Some(self.graph.structural_similarity(id_a, id_b, sm_a, sm_b))
    }

    /// Analyze what substitution transforms one concept into another (v6.0).
    ///
    /// Returns the composition substitutions needed.
    /// Example: raja → ratu requires substituting (laki_laki, 0) → (perempuan, 0).
    pub fn substitution_analysis(
        &self,
        a: &str,
        b: &str,
    ) -> Option<crate::graph::SubstitutionResult> {
        let id_a = *self.token_to_id.get(a)?;
        let id_b = *self.token_to_id.get(b)?;
        let sm_a = self.senses.get(&id_a)?;
        let sm_b = self.senses.get(&id_b)?;
        self.graph.substitution_analysis(id_a, id_b, sm_a, sm_b)
    }

    /// v6.2: Context-weighted similarity between two concepts.
    ///
    /// Unlike `structural_similarity()` which compares compositions structurally
    /// (shared vs differing), this method weighs each composition based on its
    /// relevance to the `context` labels. This produces a context-aware similarity
    /// score that reflects how similar two concepts are WITHIN a given context.
    ///
    /// Formula: sim(A, B | q) = cosine_similarity(score_vec_A, score_vec_B)
    /// where score_vec[comp] = P(a|S,q) = freq_map[a] × edge_weight(a→q)
    ///
    /// Example: "batu" and "tulang" may have low structural similarity in general,
    /// but if context is ["kekerasan"], both score high for "hard" atom →
    /// context_weighted_similarity is high.
    pub fn context_similarity(
        &self,
        a: &str,
        b: &str,
        context: &[&str],
    ) -> Option<f32> {
        let id_a = *self.token_to_id.get(a)?;
        let id_b = *self.token_to_id.get(b)?;
        let sm_a = self.senses.get(&id_a)?;
        let sm_b = self.senses.get(&id_b)?;

        // Resolve context labels to node IDs
        let context_ids: Vec<NodeId> = context
            .iter()
            .filter_map(|t| self.token_to_id.get(*t).copied())
            .collect();

        Some(self.graph.context_weighted_similarity(sm_a, sm_b, &context_ids))
    }
}
