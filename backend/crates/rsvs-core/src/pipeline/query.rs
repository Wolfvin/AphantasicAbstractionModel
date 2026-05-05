//! Query pipeline — RSVS v6.0 Compositional Architecture
//!
//! Contains `query()`, `similarity()`, `structural_similarity()`, and
//! `substitution_analysis()` methods.

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
}

impl Rsvs {
    /// Context-aware lookup for a concept.
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

        let mut scored: Vec<(String, f32)> = core
            .iter()
            .filter_map(|&atom_id| {
                let label = self.graph.get_node(atom_id)?.label.clone();
                let freq = sense.freq(atom_id);
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
            })
            .collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Build composition labels (v6.0)
        let compositions: Vec<(String, SenseId)> = sense
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
