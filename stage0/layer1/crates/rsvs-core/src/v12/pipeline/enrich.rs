use super::engine::{ErasedTransform, IngestResult};
use super::graph::Graph;
use super::super::types::*;

// ========================================================================
// EnrichComposition — Graph-Context-Aware Enrichment
// ========================================================================

/// EnrichComposition transform — enriches compositions using graph context.
///
/// For each `EnrichmentRequest` in `ctx.pending_enrichments`:
/// 1. Look up the target composition in the graph
/// 2. Verify the candidate node exists (or create it via `ensure_node`)
/// 3. Check for role conflicts (avoid duplicate roles)
/// 4. Add the candidate node as a new member with the specified role
/// 5. Re-compute composition confidence based on completeness
/// 6. Create a feedback edge with `EdgeSource::EnrichmentFeedback`
/// 7. If enrichment came from PassiveRecall, also create a secondary
///    confirming edge from the source composition
///
/// # Transform Signature
///
/// ```text
/// Input:  EnrichmentRequest — read from ctx.pending_enrichments
/// Output: GraphDelta — applied to graph
/// ```
pub struct EnrichComposition {
    /// Whether to skip enrichment when the role is already filled.
    /// If true (default), adding a duplicate role is a no-op.
    /// If false, the existing member is replaced.
    pub skip_duplicate_roles: bool,
}

impl EnrichComposition {
    /// Create a new EnrichComposition transform.
    pub fn new() -> Self {
        Self {
            skip_duplicate_roles: true,
        }
    }
}

impl Default for EnrichComposition {
    fn default() -> Self {
        Self::new()
    }
}

impl ErasedTransform for EnrichComposition {
    fn id(&self) -> &'static str {
        "EnrichComposition"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let requests = std::mem::take(&mut ctx.pending_enrichments);
        let mut enrichments_applied = 0;
        let mut edges_created = 0;
        let mut governance_transitions = 0;

        for request in &requests {
            // Ensure the candidate node exists in the graph.
            let candidate_node_id = if graph.has_node(request.candidate_node_id) {
                request.candidate_node_id
            } else {
                // Try to find by label.
                match graph.find_node_by_label(&request.candidate_label) {
                    Some(id) => id,
                    None => graph.ensure_node(&request.candidate_label),
                }
            };

            if let Some(composition) = graph.compositions.get_mut(&request.target_composition_id) {
                // Check for duplicate role.
                if self.skip_duplicate_roles
                    && composition.has_member_with_role(request.role_to_fill.clone())
                {
                    // Skip — role already filled.
                    continue;
                }

                // If not skipping duplicates, remove the existing member with this role.
                if !self.skip_duplicate_roles
                    && composition.has_member_with_role(request.role_to_fill.clone())
                {
                    composition
                        .members
                        .retain(|m| m.role != request.role_to_fill);
                }

                // Add the candidate as a new member.
                composition.members.push(CompositionMember {
                    node_id: candidate_node_id,
                    role: request.role_to_fill.clone(),
                    confidence: request.confidence,
                    label: request.candidate_label.clone(),
                    source: None,
                });

                // Re-compute confidence based on completeness.
                let completeness_bonus = self.compute_completeness_bonus(composition);
                composition.confidence = (composition.confidence + completeness_bonus).min(1.0);

                // Create a feedback edge.
                graph.edges.push((
                    request.target_composition_id.clone(),
                    candidate_node_id,
                    SemanticEdge {
                        relation: crate::types::RelationType::Categorical,
                        role: Some(request.role_to_fill.clone()),
                        source: crate::types::EdgeSource::EnrichmentFeedback,
                    },
                ));
                edges_created += 1;
                enrichments_applied += 1;

                // Check for lifecycle promotion after enrichment.
                if composition.lifecycle == LifecycleState::New && composition.batch_seen >= 1 {
                    composition.lifecycle = LifecycleState::Candidate;
                    governance_transitions += 1;
                }
            }
        }

        IngestResult {
            atoms_created: 0,
            compositions_created: 0,
            edges_created,
            gaps_detected: 0,
            enrichments_applied,
            governance_transitions,
        }
    }
}

impl EnrichComposition {
    /// Compute a confidence bonus based on composition completeness.
    ///
    /// More complete compositions (more expected roles filled) get a
    /// higher bonus. This incentivizes filling gaps.
    fn compute_completeness_bonus(&self, composition: &Composition) -> f32 {
        let (expected, filled) = match composition.composition_type {
            CompositionType::Event => {
                let expected = 4; // Predicate, Agent, Patient, Cause
                let filled = composition
                    .members
                    .iter()
                    .filter(|m| {
                        matches!(
                            m.role,
                            SemanticRole::Predicate
                                | SemanticRole::Arg0Agent
                                | SemanticRole::Arg1Patient
                                | SemanticRole::Cause
                        )
                    })
                    .count();
                (expected, filled)
            }
            CompositionType::HiddenMeaning => {
                let expected = 3; // PatternType, Problem, Solution
                let filled = composition
                    .members
                    .iter()
                    .filter(|m| {
                        matches!(
                            m.role,
                            SemanticRole::PatternType
                                | SemanticRole::Problem
                                | SemanticRole::Solution
                        )
                    })
                    .count();
                (expected, filled)
            }
            CompositionType::Pattern => {
                let expected = 3; // PatternType, Antecedent, Consequent
                let filled = composition
                    .members
                    .iter()
                    .filter(|m| {
                        matches!(
                            m.role,
                            SemanticRole::PatternType
                                | SemanticRole::Antecedent
                                | SemanticRole::Consequent
                        )
                    })
                    .count();
                (expected, filled)
            }
            _ => {
                let expected = 2;
                let filled = composition.members.len().min(expected);
                (expected, filled)
            }
        };

        // Bonus scales with completeness: 0.05 per filled role.
        if filled > 0 && expected > 0 {
            0.05 * (filled as f32 / expected as f32)
        } else {
            0.0
        }
    }
}

/// Implement the `Transform` trait for `EnrichComposition`.
impl Transform for EnrichComposition {
    type Input = EnrichmentRequest;
    type Output = GraphDelta;

    fn id(&self) -> &'static str {
        "EnrichComposition"
    }

    fn transform(&self, input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        let mut delta = GraphDelta::new();
        delta.new_nodes.push(input.candidate_node_id);
        delta
    }
}
