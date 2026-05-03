//! Query pipeline — RSVS v4.2
//!
//! Contains `query()` and `similarity()` methods.

use super::Rsvs;
use crate::types::NodeId;

// -----------------------------------------------------------------------
// QueryResult — output of a context-aware query
// -----------------------------------------------------------------------

/// Output of a context-aware query.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Index of the active sense used for the query.
    pub active_sense_idx: usize,
    /// Number of contexts in the active sense.
    pub active_sense_n: usize,
    /// Scored atoms: (label, score) sorted by score descending.
    pub scored_atoms: Vec<(String, f32)>,
}

impl Rsvs {
    /// Context-aware lookup for a concept.
    ///
    /// Given a concept label and a query context string, finds the most relevant
    /// sense and returns scored atoms from that sense's core.
    ///
    /// # Examples
    /// ```ignore
    /// let mut rsvs = Rsvs::new(PipelineConfig::default())?;
    /// rsvs.ingest_text("Stone is hard and solid.")?;
    /// if let Some(result) = rsvs.query("stone", "hard texture") {
    ///     println!("Found {} scored atoms", result.scored_atoms.len());
    /// }
    /// ```
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

        Some(QueryResult {
            active_sense_idx,
            active_sense_n: sense.context_count(),
            scored_atoms: scored,
        })
    }

    /// Compute similarity between two concepts in the graph.
    ///
    /// Returns `None` if either concept is not found in the token map.
    ///
    /// # Examples
    /// ```ignore
    /// if let Some(result) = rsvs.similarity("stone", "rock") {
    ///     println!("Jaccard: {:.3}", result.jaccard);
    /// }
    /// ```
    pub fn similarity(&self, a: &str, b: &str) -> Option<crate::graph::SimilarityResult> {
        let id_a = *self.token_to_id.get(a)?;
        let id_b = *self.token_to_id.get(b)?;
        Some(self.graph.similarity(id_a, id_b))
    }
}
