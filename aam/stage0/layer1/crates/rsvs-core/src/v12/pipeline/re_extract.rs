use super::engine::{ErasedTransform, IngestResult};
use super::graph::Graph;
use super::super::types::*;
use crate::types::NodeId;

// ========================================================================
// ReExtractFrame — Graph-Assisted Re-Extraction
// ========================================================================

/// ReExtractFrame transform — re-extracts a frame using graph context.
///
/// For each `ReExtractionRequest` in `ctx.pending_reextractions`:
/// 1. Look up the target composition and its source text
/// 2. Use graph context (known role-fillers) as hints for re-extraction
/// 3. Re-run ExtractFrame with `graph_assisted = true`
/// 4. If the re-extracted frame has higher confidence, replace the old one
/// 5. Create a feedback edge with `EdgeSource::ExtractionRepair`
///
/// # Transform Signature
///
/// ```text
/// Input:  ReExtractionRequest — read from ctx.pending_reextractions
/// Output: Option<SemanticAtom> — new atom if re-extraction succeeded
/// ```
pub struct ReExtractFrame {
    /// Whether to always replace, even if re-extraction confidence is lower.
    /// Default: false (only replace if confidence improves).
    pub force_replace: bool,
}

impl ReExtractFrame {
    /// Create a new ReExtractFrame transform.
    pub fn new() -> Self {
        Self {
            force_replace: false,
        }
    }
}

impl Default for ReExtractFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl ErasedTransform for ReExtractFrame {
    fn id(&self) -> &'static str {
        "ReExtractFrame"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let requests = std::mem::take(&mut ctx.pending_reextractions);
        let mut enrichments_applied = 0;
        let mut edges_created = 0;

        for request in requests {
            // Get the source text for re-extraction.
            let source_text = if !request.original_text.is_empty() {
                request.original_text.clone()
            } else {
                // Try to get from the composition.
                match graph.get_composition(&request.target_composition_id) {
                    Some(comp) => match &comp.source_text {
                        Some(text) => text.clone(),
                        None => continue,
                    },
                    None => continue,
                }
            };

            // Get the target composition's current confidence.
            let current_confidence = graph
                .get_composition(&request.target_composition_id)
                .map(|c| c.confidence)
                .unwrap_or(0.0);

            // Build graph context hints as (role, node_id, confidence) triples.
            let mut context_hints: Vec<(SemanticRole, NodeId, f32)> = request.graph_context.clone();

            // Also add existing members as context.
            if let Some(comp) = graph.get_composition(&request.target_composition_id) {
                for member in &comp.members {
                    context_hints.push((member.role.clone(), member.node_id, member.confidence));
                }
            }

            // Re-extract using the enhanced context.
            let extractor = super::super::extract_frame::ExtractFrame::new();
            let re_result = extractor.re_extract_with_context_and_kb(&source_text, &context_hints, graph, &ctx.knowledge_base);

            match re_result {
                Some(re_atom) => {
                    let re_confidence = re_atom.confidence;

                    // Only replace if confidence improved (or force_replace).
                    if self.force_replace || re_confidence > current_confidence {
                        // Collect new members first (need immutable borrow for ensure_node).
                        let new_members: Vec<CompositionMember> = re_atom
                            .roles
                            .iter()
                            .map(|(role, label)| {
                                let node_id = graph.ensure_node(label);
                                CompositionMember {
                                    node_id,
                                    role: role.clone(),
                                    confidence: re_confidence * 0.95,
                                    label: label.clone(),
                                    source: None,
                                }
                            })
                            .collect();

                        // Create repair edges.
                        for member in &new_members {
                            graph.edges.push((
                                request.target_composition_id.clone(),
                                member.node_id,
                                SemanticEdge {
                                    relation: crate::types::RelationType::Categorical,
                                    role: Some(member.role.clone()),
                                    source: crate::types::EdgeSource::ExtractionRepair,
                                },
                            ));
                            edges_created += 1;
                        }

                        // Update the composition with re-extracted data.
                        if let Some(composition) =
                            graph.compositions.get_mut(&request.target_composition_id)
                        {
                            composition.members = new_members;
                            composition.confidence = re_confidence;
                            composition.provenance.origin =
                                crate::types::EdgeSource::ExtractionRepair;
                            enrichments_applied += 1;
                        }
                    }
                }
                None => {
                    // Re-extraction failed — no improvement possible.
                }
            }
        }

        IngestResult {
            atoms_created: 0,
            compositions_created: 0,
            edges_created,
            gaps_detected: 0,
            enrichments_applied,
            governance_transitions: 0,
        }
    }
}

/// Implement the `Transform` trait for `ReExtractFrame`.
impl Transform for ReExtractFrame {
    type Input = ReExtractionRequest;
    type Output = Option<SemanticAtom>;

    fn id(&self) -> &'static str {
        "ReExtractFrame"
    }

    fn transform(&self, input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        // Simplified: just return None (full logic requires Graph access).
        let _ = input;
        None
    }
}
