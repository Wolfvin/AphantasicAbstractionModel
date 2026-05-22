//! # RAB Phase 4: Contextual Sense Disambiguation (CSD) Engine
//!
//! Uses Spreading Activation as graph-based attention to select the correct
//! sense of ambiguous words. This is AAM's answer to transformer self-attention.
//!
//! ## Algorithm
//!
//! ```text
//! 1. Look up candidate senses for word in SenseRegistry
//! 2. Spread activation from context nodes (surrounding words)
//! 3. For each candidate sense, sum activation energy of its representative nodes
//! 4. Select the highest-scoring sense
//! 5. Create DisambiguatedSense composition with evidence trail
//! ```

use std::collections::HashMap;

use super::pipeline::{ErasedTransform, Graph, IngestResult};
use super::spreading::{ActivationMap, SpreadingActivation};
use super::sense_registry::{DisambiguationResult, SenseEntry, SenseRegistry};
use super::types::*;
use crate::types::{EdgeSource, NodeId};

/// Minimum confidence for a sense to be considered resolved.
const CSD_MIN_CONFIDENCE: f32 = 0.3;
/// Minimum score difference between top 2 candidates to consider resolved.
const CSD_MIN_MARGIN: f32 = 0.2;

// ========================================================================
// CSDEngine — Contextual Sense Disambiguation
// ========================================================================

/// Contextual Sense Disambiguation engine.
///
/// Uses spreading activation from context nodes to score candidate senses
/// and select the most appropriate one.
#[derive(Debug, Clone)]
pub struct CSDEngine {
    /// The spreading activation engine.
    activation: SpreadingActivation,
    /// The sense registry.
    registry: SenseRegistry,
}

impl Default for CSDEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CSDEngine {
    /// Create a new CSD engine with bootstrap sense entries.
    pub fn new() -> Self {
        Self {
            activation: SpreadingActivation::new(),
            registry: SenseRegistry::with_bootstrap_entries(),
        }
    }

    /// Create with a custom sense registry.
    pub fn with_registry(registry: SenseRegistry) -> Self {
        Self {
            activation: SpreadingActivation::new(),
            registry,
        }
    }

    /// Get a reference to the sense registry.
    pub fn registry(&self) -> &SenseRegistry {
        &self.registry
    }

    /// Get a mutable reference to the sense registry.
    pub fn registry_mut(&mut self) -> &mut SenseRegistry {
        &mut self.registry
    }

    /// Disambiguate a word given its context nodes in the graph.
    ///
    /// # Algorithm
    ///
    /// 1. Look up candidate senses for `word` in the registry
    /// 2. If only one sense, return it immediately (unambiguous)
    /// 3. Spread activation from context nodes
    /// 4. For each candidate sense, sum activation energy of representative nodes
    /// 5. Select the highest-scoring sense
    /// 6. Return DisambiguationResult with evidence trail
    pub fn disambiguate(
        &self,
        word: &str,
        context_node_ids: &[NodeId],
        graph: &Graph,
    ) -> DisambiguationResult {
        let candidates = self.registry.senses_for(word);

        // Fast path: single sense or no candidates.
        if candidates.is_empty() {
            return DisambiguationResult {
                word: word.to_string(),
                ..DisambiguationResult::default()
            };
        }
        if candidates.len() == 1 {
            return DisambiguationResult {
                selected_sense: Some(candidates[0].clone()),
                confidence: 1.0,
                evidence: Vec::new(),
                candidate_scores: vec![(candidates[0].sense_id.clone(), 1.0)],
                word: word.to_string(),
            };
        }

        // Spread activation from context nodes.
        let context_seeds: Vec<(NodeId, f32)> = context_node_ids
            .iter()
            .map(|&id| (id, 1.0))
            .collect();

        let activation_map = if context_seeds.is_empty() {
            // No context — can't disambiguate.
            return DisambiguationResult {
                word: word.to_string(),
                candidate_scores: candidates.iter().map(|c| (c.sense_id.clone(), 0.0)).collect(),
                ..DisambiguationResult::default()
            };
        } else {
            self.activation.spread(&context_seeds, graph)
        };

        // Score each candidate by summing activation of its representative nodes.
        let mut candidate_scores: Vec<(String, f32)> = Vec::new();
        let mut best_sense: Option<&SenseEntry> = None;
        let mut best_score: f32 = -1.0;

        for candidate in candidates {
            let score = self.score_sense(candidate, &activation_map, graph);
            candidate_scores.push((candidate.sense_id.clone(), score));

            if score > best_score {
                best_score = score;
                best_sense = Some(candidate);
            }
        }

        // Sort scores descending.
        candidate_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute confidence: ratio of best score to total.
        let total_score: f32 = candidate_scores.iter().map(|(_, s)| *s).sum();
        let confidence = if total_score > 0.0 {
            best_score / total_score
        } else {
            0.0
        };

        // Check margin between top 2 candidates.
        let margin = if candidate_scores.len() >= 2 {
            candidate_scores[0].1 - candidate_scores[1].1
        } else {
            1.0
        };

        // Only resolve if confidence and margin are sufficient.
        let selected = if confidence >= CSD_MIN_CONFIDENCE && margin >= CSD_MIN_MARGIN {
            best_sense.cloned()
        } else if confidence > 0.0 {
            // Low confidence — still select best but mark as uncertain.
            best_sense.cloned()
        } else {
            None
        };

        DisambiguationResult {
            selected_sense: selected,
            confidence,
            evidence: activation_map.top_n(5),
            candidate_scores,
            word: word.to_string(),
        }
    }

    /// Score a sense entry by summing activation of its representative nodes.
    fn score_sense(
        &self,
        sense: &SenseEntry,
        activation_map: &ActivationMap,
        graph: &Graph,
    ) -> f32 {
        let mut score = 0.0f32;

        for label in &sense.representative_labels {
            // Look up the node by label.
            if let Some(node_id) = graph.label_to_id.get(label) {
                score += activation_map.energy(*node_id);
            }
        }

        score
    }

    /// Create a DisambiguatedSense composition from a disambiguation result.
    ///
    /// This is called after disambiguation succeeds to create the graph composition.
    pub fn create_disambiguated_composition(
        &self,
        result: &DisambiguationResult,
        target_node_id: NodeId,
        context_node_ids: &[NodeId],
        graph: &mut Graph,
    ) -> Option<CompositionId> {
        let sense = result.selected_sense.as_ref()?;

        // Create or find the sense-specific node.
        let sense_label = format!("{}_{}", result.word, sense.label);
        let sense_node_id = graph.ensure_node(&sense_label);

        // Build composition members.
        let mut members = vec![
            CompositionMember {
                node_id: target_node_id,
                role: SemanticRole::SenseTarget,
                confidence: result.confidence,
                label: result.word.clone(),
                source: Some(EdgeSource::SenseDisambiguation),
            },
            CompositionMember {
                node_id: sense_node_id,
                role: SemanticRole::SelectedSense,
                confidence: result.confidence,
                label: sense_label.clone(),
                source: Some(EdgeSource::SenseDisambiguation),
            },
        ];

        // Add context nodes.
        for &ctx_id in context_node_ids {
            if let Some(label) = graph.node_label(ctx_id) {
                members.push(CompositionMember {
                    node_id: ctx_id,
                    role: SemanticRole::SenseContext,
                    confidence: 0.7,
                    label: label.to_string(),
                    source: Some(EdgeSource::SenseDisambiguation),
                });
            }
        }

        let comp_id = CompositionId::new(format!(
            "comp_csd_{}_{}",
            result.word,
            sense.sense_id
        ));

        let composition = Composition {
            id: comp_id.clone(),
            composition_type: CompositionType::DisambiguatedSense,
            members,
            lifecycle: LifecycleState::Candidate,
            epistemic: EpistemicState::Observed,
            confidence: result.confidence,
            provenance: ProvenanceChain {
                origin: EdgeSource::SenseDisambiguation,
                origin_id: sense.sense_id.clone(),
                parent_composition_id: None,
                timestamp: chrono_now_iso(),
            },
            seed_scores: HashMap::new(),
            source_text: None,
            batch_seen: 0,
            contradiction_batches: Vec::new(),
            contradiction: None,
            correction_count: 0,
            last_correction_type: None,
            created_at: chrono_now_iso(),
            updated_at: chrono_now_iso(),
        };

        graph.compositions.insert(comp_id.clone(), composition);
        Some(comp_id)
    }
}

/// Simple ISO 8601 timestamp (no external chrono dependency).
fn chrono_now_iso() -> String {
    // Use a simple format — no external dependency needed.
    format!("{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs())
}

// ========================================================================
// CSDTransform — Pipeline Integration
// ========================================================================

/// Pipeline transform that performs sense disambiguation on ambiguous tokens.
///
/// Runs after IngestAtoms to check for multi-sense words and create
/// DisambiguatedSense compositions.
pub struct CSDTransform {
    engine: CSDEngine,
}

impl Default for CSDTransform {
    fn default() -> Self {
        Self::new()
    }
}

impl CSDTransform {
    pub fn new() -> Self {
        Self {
            engine: CSDEngine::new(),
        }
    }
}

impl std::fmt::Debug for CSDTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CSDTransform").finish()
    }
}

impl Clone for CSDTransform {
    fn clone(&self) -> Self {
        Self {
            engine: CSDEngine::new(),
        }
    }
}

impl ErasedTransform for CSDTransform {
    fn id(&self) -> &'static str {
        "CSD"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let mut compositions_created = 0;
        let atoms_created = 0;

        // Check all token atoms for ambiguous words.
        let ambiguous_atoms: Vec<(usize, String, NodeId)> = ctx
            .current_atoms
            .iter()
            .enumerate()
            .filter_map(|(idx, atom)| {
                if atom.atom_type == AtomType::Token {
                    if self.engine.registry().is_ambiguous(&atom.label) {
                        // Find the node ID for this token in the graph.
                        if let Some(&node_id) = graph.label_to_id.get(&atom.label) {
                            return Some((idx, atom.label.clone(), node_id));
                        }
                    }
                }
                None
            })
            .collect();

        for (_atom_idx, word, target_node_id) in ambiguous_atoms {
            // Collect context nodes from other atoms in the same ingest.
            let context_node_ids: Vec<NodeId> = ctx
                .current_atoms
                .iter()
                .filter(|a| a.atom_type == AtomType::Token && a.label != word)
                .filter_map(|a| graph.label_to_id.get(&a.label).copied())
                .collect();

            let result = self.engine.disambiguate(&word, &context_node_ids, graph);

            if result.is_resolved() {
                if let Some(_comp_id) = self.engine.create_disambiguated_composition(
                    &result,
                    target_node_id,
                    &context_node_ids,
                    graph,
                ) {
                    compositions_created += 1;
                }
            }
        }

        IngestResult {
            atoms_created,
            compositions_created,
            edges_created: 0,
            gaps_detected: 0,
            enrichments_applied: 0,
            governance_transitions: 0,
        }
    }
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csd_engine_unambiguous() {
        let engine = CSDEngine::new();
        let graph = Graph::new();
        let result = engine.disambiguate("ular", &[], &graph);
        // "ular" has no entry → empty candidates
        assert!(!result.is_resolved());
    }

    #[test]
    fn test_csd_engine_no_context() {
        let engine = CSDEngine::new();
        let graph = Graph::new();
        let result = engine.disambiguate("bisa", &[], &graph);
        // No context nodes → can't disambiguate
        assert!(!result.is_resolved());
    }

    #[test]
    fn test_disambiguation_result_defaults() {
        let result = DisambiguationResult::new();
        assert!(!result.is_resolved());
        assert!(result.word.is_empty());
    }

    #[test]
    fn test_csd_transform_creation() {
        let transform = CSDTransform::new();
        assert_eq!(transform.id(), "CSD");
    }
}
