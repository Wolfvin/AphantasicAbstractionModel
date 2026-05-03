//! Appraise and Relate modes — RSVS v4.2
//!
//! Contains `appraise()` and `relate()` methods.

use super::Rsvs;
use crate::types::NodeId;
use rayon::prelude::*;

// -----------------------------------------------------------------------
// AppraiseResult — v4.2 appraise mode output
// -----------------------------------------------------------------------

/// v4.2 appraise mode output: agree/disagree percentages, verdict, and evidence.
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
// RelateResult — v4.2 relate mode output
// -----------------------------------------------------------------------

/// v4.2 relate mode output: related nodes and edges by overlap scoring.
#[derive(Debug, Clone)]
pub struct RelateResult {
    /// Related nodes: (node_id, overlap_score) sorted by score descending.
    pub related_nodes: Vec<(NodeId, f32)>,
    /// Related edges: (from, to, weight) sorted by weight descending.
    pub related_edges: Vec<(NodeId, NodeId, f32)>,
}

impl Rsvs {
    /// Evaluate text against the graph.
    ///
    /// Returns agree/disagree percentages, a verdict ("consistent", "partial", or "novel"),
    /// and per-token evidence with confidence scores.
    ///
    /// # Examples
    /// ```ignore
    /// let mut rsvs = Rsvs::new(PipelineConfig::default())?;
    /// rsvs.ingest_text("Stone is hard and solid.")?;
    /// let result = rsvs.appraise("Stone is hard");
    /// println!("Verdict: {}", result.verdict);
    /// ```
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

        // Sort evidence by confidence descending
        evidence.sort_by(|a, b| b.1.total_cmp(&a.1));

        AppraiseResult {
            agree_pct,
            disagree_pct,
            verdict,
            evidence,
        }
    }

    /// Find nodes and edges related to the given concept by overlap scoring.
    ///
    /// Uses Jaccard similarity (parallelized with rayon) to rank related nodes,
    /// and collects both outgoing and incoming edges involving the concept.
    ///
    /// # Examples
    /// ```ignore
    /// let mut rsvs = Rsvs::new(PipelineConfig::default())?;
    /// rsvs.ingest_text("Stone is hard. Rock is heavy.")?;
    /// if let Some(result) = rsvs.relate("stone") {
    ///     println!("Found {} related nodes", result.related_nodes.len());
    /// }
    /// ```
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

        // Find related edges involving this concept
        let mut related_edges: Vec<(NodeId, NodeId, f32)> = Vec::new();

        // Outgoing edges from concept
        for e in self.graph.edges_from(concept_id) {
            related_edges.push((e.from, e.to, e.weight));
        }

        // Incoming edges to concept
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

        // Also add edges from top related nodes
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
        })
    }
}
