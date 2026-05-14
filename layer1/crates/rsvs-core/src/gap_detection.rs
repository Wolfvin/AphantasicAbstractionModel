//! Pathway 1: Predictive Gap Detection — v9.0 Meaning Pathways
//!
//! Captures: Pragmatic, Implicature, Presupposition meaning types.
//!
//! Core algorithm: GAP = PREDICTED_COMPOSITIONS − ACTUAL_COMPOSITIONS
//!
//! Every time a node is ingested, RSVS predicts what compositions the node
//! SHOULD have based on the graph structure. Gaps between predicted and
//! actual = hidden meaning (implicature, presupposition, pragmatic divergence).
//!
//! Prediction strategies:
//! 1. Seed spreading: BatchSeedSpreading cache → O(1) per lookup
//! 2. Analogical: structurally similar nodes → missing compositions
//! 3. Scalar chain: differential edges → scalar implicature
//! 4. Grounding: ungrounded compositions → presupposition
//!
//! All predictions use existing graph data — NO LLM, NO training.

use crate::composition_index::CompositionIndex;
use crate::graph::RsvsGraph;
use crate::sense::SenseManager;
use crate::types::{
    CompositionRef, Edge, EdgeSource, GapAnnotation, GapType, NodeId, RelationType, SenseId,
};
use std::collections::{HashMap, HashSet};

/// One detected meaning gap.
#[derive(Debug, Clone)]
pub struct MeaningGap {
    /// What type of gap was detected.
    pub gap_type: GapType,
    /// Compositions that were expected but not found.
    pub expected: Vec<CompositionRef>,
    /// Evidence supporting why we expected these compositions.
    pub evidence: GapEvidence,
    /// Confidence of this gap detection (0.0–1.0).
    pub confidence: f32,
    /// The node where the gap was detected.
    pub source_node: NodeId,
    /// Structural description for Layer 3 reasoning.
    pub structural_description: StructuralDescription,
}

/// Evidence for why a gap was detected.
#[derive(Debug, Clone)]
pub enum GapEvidence {
    /// Gap from seed spreading activation.
    SeedActivation {
        seed: NodeId,
        activated_area: Vec<NodeId>,
        activation_energy: f32,
    },
    /// Gap from scalar chain (stronger item unused).
    ScalarChain {
        scale: Vec<NodeId>,
        used_index: usize,
        stronger_unused: Vec<NodeId>,
    },
    /// Gap from analogical reasoning (similar node has composition this node lacks).
    Analogical {
        similar_node: NodeId,
        similarity: f32,
        missing_composition_target: NodeId,
    },
    /// Gap from grounding check (referenced node not well-grounded).
    GroundingRequired {
        required_node_label: String,
        found: bool,
        accommodation_candidate: Option<NodeId>,
    },
    /// Gap from pattern divergence (actual ≠ predicted).
    PatternDivergence {
        predicted_pattern: Vec<CompositionRef>,
        actual_pattern: Vec<CompositionRef>,
        divergence_score: f32,
    },
}

/// Structural description for Layer 3 reasoning.
#[derive(Debug, Clone)]
pub struct StructuralDescription {
    /// Trace back to seed primitives.
    pub seed_trace: Vec<NodeId>,
    /// Relation type hint for gap edges.
    pub relation_hint: Option<RelationType>,
    /// Expected composition targets.
    pub expected_composition_targets: Vec<NodeId>,
}

/// A scalar scale for implicature detection.
#[derive(Debug, Clone)]
pub struct ScalarScale {
    /// Nodes ordered: strongest → weakest.
    pub nodes: Vec<NodeId>,
    /// Human-readable label for this scale.
    pub scale_label: String,
    /// The dimension along which items vary.
    pub dimension: String,
}

/// Cached scalar scale index for O(1) lookup.
pub struct ScalarScaleIndex {
    /// Node → (scale_index, position_in_scale)
    node_to_scale: HashMap<NodeId, (usize, usize)>,
    /// The scales themselves.
    scales: Vec<ScalarScale>,
}

impl ScalarScaleIndex {
    /// Create a new empty index.
    pub fn new() -> Self {
        Self {
            node_to_scale: HashMap::new(),
            scales: Vec::new(),
        }
    }

    /// O(1) lookup: is this node in a scalar scale?
    pub fn get_scale_position(&self, node_id: NodeId) -> Option<(usize, usize)> {
        self.node_to_scale.get(&node_id).copied()
    }

    /// Get a scale by index.
    pub fn get_scale(&self, idx: usize) -> Option<&ScalarScale> {
        self.scales.get(idx)
    }

    /// Rebuild index from discovered scales.
    pub fn rebuild(&mut self, scales: Vec<ScalarScale>) {
        self.node_to_scale.clear();
        for (scale_idx, scale) in scales.iter().enumerate() {
            for (pos, &node_id) in scale.nodes.iter().enumerate() {
                self.node_to_scale.insert(node_id, (scale_idx, pos));
            }
        }
        self.scales = scales;
    }

    /// Number of scales.
    pub fn len(&self) -> usize {
        self.scales.len()
    }

    /// Is the index empty?
    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }
}

/// Configuration for gap detection.
#[derive(Debug, Clone)]
pub struct GapDetectionConfig {
    /// Enable scalar implicature detection.
    pub enable_scalar: bool,
    /// Enable presupposition detection.
    pub enable_presupposition: bool,
    /// Enable pragmatic divergence detection.
    pub enable_pragmatic: bool,
    /// Enable affective mismatch detection.
    pub enable_affective: bool,
    /// Minimum activation energy to consider a prediction.
    pub min_activation_energy: f32,
    /// Minimum similarity for analogical prediction.
    pub min_analogical_similarity: f32,
    /// Minimum confidence to create a gap edge.
    pub min_gap_confidence_for_edge: f32,
    /// Maximum gaps per ingest batch.
    pub max_gaps_per_ingest: usize,
    /// Seed NodeIds for affective pathway.
    pub affective_seeds: Vec<NodeId>,
    /// Seed NodeIds for social pathway.
    pub social_seeds: Vec<NodeId>,
    /// Seed NodeIds for pragmatic pathway.
    pub pragmatic_seeds: Vec<NodeId>,
}

impl Default for GapDetectionConfig {
    fn default() -> Self {
        Self {
            enable_scalar: true,
            enable_presupposition: true,
            enable_pragmatic: true,
            enable_affective: true,
            min_activation_energy: 0.15,
            min_analogical_similarity: 0.4,
            min_gap_confidence_for_edge: 0.3,
            max_gaps_per_ingest: 20,
            affective_seeds: Vec::new(),
            social_seeds: Vec::new(),
            pragmatic_seeds: Vec::new(),
        }
    }
}

/// The gap detection engine.
pub struct GapDetector {
    /// Configuration.
    pub config: GapDetectionConfig,
    /// Discovered scalar scales.
    scalar_index: ScalarScaleIndex,
}

impl GapDetector {
    /// Create a new gap detector with the given configuration.
    pub fn new(config: GapDetectionConfig) -> Self {
        Self {
            config,
            scalar_index: ScalarScaleIndex::new(),
        }
    }

    /// Predict expected compositions for a node's sense using all strategies.
    ///
    /// Returns a list of (expected_composition, evidence) pairs.
    pub fn predict_expected_compositions(
        &self,
        node_id: NodeId,
        sense_id: SenseId,
        actual: &[CompositionRef],
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
        batch_cache: &crate::batch_spreading::BatchSeedSpreading,
    ) -> Vec<(CompositionRef, GapEvidence)> {
        let mut predictions = Vec::new();
        let actual_targets: HashSet<NodeId> = actual.iter().map(|c| c.node_id).collect();

        // Strategy 1: Seed spreading prediction (O(1) per lookup from cache)
        predictions.extend(self.predict_from_seeds(
            node_id, &actual_targets, batch_cache,
        ));

        // Strategy 2: Analogical prediction
        predictions.extend(self.predict_from_analogy(
            node_id, sense_id, &actual_targets, graph, senses, comp_index,
        ));

        // Strategy 3: Scalar chain prediction
        if self.config.enable_scalar {
            predictions.extend(self.predict_from_scalar(node_id, &actual_targets));
        }

        // Strategy 4: Grounding prediction
        if self.config.enable_presupposition {
            predictions.extend(self.predict_from_grounding(node_id, actual, graph));
        }

        predictions
    }

    /// Strategy 1: Predict from seed spreading cache.
    ///
    /// Nodes that are strongly activated by seeds but NOT in actual compositions
    /// are predicted gaps.
    fn predict_from_seeds(
        &self,
        node_id: NodeId,
        actual_targets: &HashSet<NodeId>,
        batch_cache: &crate::batch_spreading::BatchSeedSpreading,
    ) -> Vec<(CompositionRef, GapEvidence)> {
        let mut predictions = Vec::new();
        let all_seeds = batch_cache.all_seeds();

        for &seed_id in &all_seeds {
            let energy = batch_cache.get_energy(seed_id, node_id);
            if energy < self.config.min_activation_energy {
                continue;
            }

            // Find nodes activated by this seed that the node doesn't compose to
            // For simplicity, check if the seed itself is in actual
            if !actual_targets.contains(&seed_id) && energy >= self.config.min_activation_energy {
                let activated_area = vec![seed_id]; // Simplified
                predictions.push((
                    CompositionRef::new(seed_id, 0),
                    GapEvidence::SeedActivation {
                        seed: seed_id,
                        activated_area,
                        activation_energy: energy,
                    },
                ));
            }
        }

        predictions
    }

    /// Strategy 2: Predict from analogy — structurally similar nodes.
    fn predict_from_analogy(
        &self,
        node_id: NodeId,
        _sense_id: SenseId,
        actual_targets: &HashSet<NodeId>,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
    ) -> Vec<(CompositionRef, GapEvidence)> {
        let mut predictions = Vec::new();

        // Find dependents (nodes that share compositions with this node)
        let dependents = comp_index.dependents_of_node(node_id);
        for dep_id in dependents {
            if dep_id == node_id {
                continue;
            }

            // Retrieve SenseManagers for both nodes to match structural_similarity signature
            let (sm_a, sm_b) = match (senses.get(&node_id), senses.get(&dep_id)) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };

            let similarity = graph.structural_similarity(node_id, dep_id, sm_a, sm_b);
            if similarity.structural_similarity < self.config.min_analogical_similarity {
                continue;
            }

            // Get compositions of the similar node
            if let Some(sm) = senses.get(&dep_id) {
                for sense in &sm.senses {
                    for comp in &sense.compositions {
                        if !actual_targets.contains(&comp.node_id) {
                            predictions.push((
                                comp.clone(),
                                GapEvidence::Analogical {
                                    similar_node: dep_id,
                                    similarity: similarity.structural_similarity,
                                    missing_composition_target: comp.node_id,
                                },
                            ));
                        }
                    }
                }
            }

            // Only use top similar node to avoid explosion
            if predictions.len() >= 5 {
                break;
            }
        }

        predictions
    }

    /// Strategy 3: Predict from scalar scales.
    ///
    /// If a node is at position i in a scalar scale, items at positions < i
    /// are "stronger" — not using them = scalar implicature.
    fn predict_from_scalar(
        &self,
        node_id: NodeId,
        actual_targets: &HashSet<NodeId>,
    ) -> Vec<(CompositionRef, GapEvidence)> {
        let mut predictions = Vec::new();

        if let Some((scale_idx, used_index)) = self.scalar_index.get_scale_position(node_id) {
            if let Some(scale) = self.scalar_index.get_scale(scale_idx) {
                // Items before used_index are stronger (not used = implicature)
                let stronger_unused: Vec<NodeId> = scale.nodes.iter()
                    .take(used_index)
                    .filter(|&&id| !actual_targets.contains(&id))
                    .copied()
                    .collect();

                for &unused_id in &stronger_unused {
                    predictions.push((
                        CompositionRef::new(unused_id, 0),
                        GapEvidence::ScalarChain {
                            scale: scale.nodes.clone(),
                            used_index,
                            stronger_unused: stronger_unused.clone(),
                        },
                    ));
                }
            }
        }

        predictions
    }

    /// Strategy 4: Predict from grounding check.
    ///
    /// Compositions that reference nodes that don't exist or aren't
    /// well-grounded = presupposition gaps.
    fn predict_from_grounding(
        &self,
        _node_id: NodeId,
        actual: &[CompositionRef],
        graph: &RsvsGraph,
    ) -> Vec<(CompositionRef, GapEvidence)> {
        let mut predictions = Vec::new();

        for comp in actual {
            let target_exists = graph.get_node(comp.node_id).is_some();
            let target_grounded = graph.get_node(comp.node_id)
                .map(|n| n.confidence >= 0.3)
                .unwrap_or(false);

            if !target_exists {
                // Node doesn't exist — strong presupposition gap
                predictions.push((
                    comp.clone(),
                    GapEvidence::GroundingRequired {
                        required_node_label: format!("node_{}", comp.node_id),
                        found: false,
                        accommodation_candidate: None,
                    },
                ));
            } else if !target_grounded {
                // Node exists but not well-grounded — weaker presupposition
                predictions.push((
                    comp.clone(),
                    GapEvidence::GroundingRequired {
                        required_node_label: graph.get_node(comp.node_id)
                            .map(|n| n.label.clone())
                            .unwrap_or_default(),
                        found: true,
                        accommodation_candidate: None,
                    },
                ));
            }
        }

        predictions
    }

    /// Compute gaps between predicted and actual compositions.
    pub fn compute_gaps(
        &self,
        node_id: NodeId,
        sense_id: SenseId,
        actual: &[CompositionRef],
        predictions: Vec<(CompositionRef, GapEvidence)>,
        graph: &RsvsGraph,
    ) -> Vec<MeaningGap> {
        let actual_set: HashSet<CompositionRef> = actual.iter().cloned().collect();
        let mut gaps = Vec::new();

        for (predicted_comp, evidence) in predictions {
            // Skip if already in actual
            if actual_set.contains(&predicted_comp) {
                continue;
            }

            let gap_type = self.classify_gap(&evidence, graph);
            let confidence = self.compute_confidence(&evidence);
            let seed_trace = self.trace_to_seeds(&evidence);

            gaps.push(MeaningGap {
                gap_type: gap_type.clone(),
                expected: vec![predicted_comp.clone()],
                evidence,
                confidence,
                source_node: node_id,
                structural_description: StructuralDescription {
                    seed_trace,
                    relation_hint: self.infer_relation_type(&gap_type),
                    expected_composition_targets: vec![predicted_comp.node_id],
                },
            });

            if gaps.len() >= self.config.max_gaps_per_ingest {
                break;
            }
        }

        gaps
    }

    /// Classify a gap based on its evidence.
    fn classify_gap(&self, evidence: &GapEvidence, _graph: &RsvsGraph) -> GapType {
        match evidence {
            GapEvidence::SeedActivation { seed, activation_energy, .. } => {
                // Classify based on which seed pathway
                if self.config.affective_seeds.contains(seed) {
                    if *activation_energy > 0.5 {
                        GapType::AffectiveMismatch
                    } else {
                        GapType::PragmaticDivergence
                    }
                } else if self.config.social_seeds.contains(seed) {
                    GapType::SocialMismatch
                } else if self.config.pragmatic_seeds.contains(seed) {
                    GapType::PragmaticDivergence
                } else {
                    GapType::ExpectedComposition
                }
            }
            GapEvidence::ScalarChain { .. } => GapType::ScalarImplicature,
            GapEvidence::Analogical { .. } => GapType::ExpectedComposition,
            GapEvidence::GroundingRequired { found, .. } => {
                if *found {
                    GapType::PresuppositionUngrounded
                } else {
                    GapType::PresuppositionUngrounded
                }
            }
            GapEvidence::PatternDivergence { .. } => GapType::PragmaticDivergence,
        }
    }

    /// Compute confidence of a gap based on its evidence.
    fn compute_confidence(&self, evidence: &GapEvidence) -> f32 {
        match evidence {
            GapEvidence::SeedActivation { activation_energy, .. } => {
                (*activation_energy * 0.8).clamp(0.1, 0.9)
            }
            GapEvidence::ScalarChain { .. } => 0.7, // Scalar implicature is reliable
            GapEvidence::Analogical { similarity, .. } => {
                (similarity * 0.8).clamp(0.1, 0.9) // Discount for analogy
            }
            GapEvidence::GroundingRequired { found, .. } => {
                if *found { 0.5 } else { 0.8 } // Missing node = higher confidence
            }
            GapEvidence::PatternDivergence { divergence_score, .. } => {
                divergence_score.clamp(0.1, 0.9)
            }
        }
    }

    /// Trace evidence back to seed primitives.
    fn trace_to_seeds(&self, evidence: &GapEvidence) -> Vec<NodeId> {
        match evidence {
            GapEvidence::SeedActivation { seed, .. } => vec![*seed],
            GapEvidence::ScalarChain { scale, .. } => {
                // Trace to the strongest item in the scale
                scale.first().copied().map(|id| vec![id]).unwrap_or_default()
            }
            GapEvidence::Analogical { similar_node, .. } => vec![*similar_node],
            GapEvidence::GroundingRequired { .. } => Vec::new(),
            GapEvidence::PatternDivergence { .. } => Vec::new(),
        }
    }

    /// Infer relation type from gap type.
    fn infer_relation_type(&self, gap_type: &GapType) -> Option<RelationType> {
        match gap_type {
            GapType::ScalarImplicature => Some(RelationType::Differential),
            GapType::PresuppositionUngrounded => Some(RelationType::Categorical),
            GapType::PragmaticDivergence => Some(RelationType::Categorical),
            GapType::AffectiveMismatch => Some(RelationType::Categorical),
            GapType::SocialMismatch => Some(RelationType::Functional),
            GapType::ConnotativeAbsent => Some(RelationType::Categorical),
            GapType::ExpectedComposition => None,
        }
    }

    /// Discover scalar scales from differential edges in the graph.
    ///
    /// This is a periodic operation — not called on every ingest.
    /// Finds chains of Differential edges and builds ScalarScale entries.
    pub fn discover_scalar_scales(&mut self, graph: &RsvsGraph) {
        let mut scales = Vec::new();

        // Collect all differential edges
        let mut diff_edges: Vec<(NodeId, NodeId)> = Vec::new();
        for (&from_id, edges) in graph.edges.iter() {
            for edge in edges {
                if edge.relation_type == RelationType::Differential {
                    diff_edges.push((from_id, edge.to));
                }
            }
        }

        // Build adjacency for differential graph
        let mut diff_adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (from, to) in &diff_edges {
            diff_adj.entry(*from).or_default().push(*to);
        }

        // Find nodes with no incoming differential edges (chain starts)
        let has_incoming: HashSet<NodeId> = diff_edges.iter().map(|(_, to)| *to).collect();
        let starts: Vec<NodeId> = diff_adj.keys()
            .filter(|id| !has_incoming.contains(id))
            .copied()
            .collect();

        // Trace chains from each start
        let mut visited: HashSet<NodeId> = HashSet::new();
        for start in starts {
            if visited.contains(&start) {
                continue;
            }

            let mut chain = Vec::new();
            let mut current = Some(start);
            while let Some(node) = current {
                if visited.contains(&node) || chain.contains(&node) {
                    break;
                }
                visited.insert(node);
                chain.push(node);
                current = diff_adj.get(&node)
                    .and_then(|neighbors| neighbors.first().copied());
            }

            // Only keep chains with >= 3 items
            if chain.len() >= 3 {
                let scale_label = chain.iter()
                    .filter_map(|&id| graph.get_node(id).map(|n| n.label.clone()))
                    .collect::<Vec<_>>()
                    .join(" > ");

                scales.push(ScalarScale {
                    nodes: chain,
                    scale_label,
                    dimension: "quantity".to_string(),
                });
            }
        }

        self.scalar_index.rebuild(scales);
    }

    /// Get the scalar scale index.
    pub fn scalar_index(&self) -> &ScalarScaleIndex {
        &self.scalar_index
    }

    /// Process gap detection for all promoted nodes in a batch.
    ///
    /// This is the main entry point called from the ingest pipeline at
    /// batch-level (Step 5.6), AFTER BatchSeedSpreading has run.
    ///
    /// Returns gap annotations per (node_id, sense_id).
    pub fn process_batch(
        &self,
        promoted_nodes: &[NodeId],
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
        batch_cache: &crate::batch_spreading::BatchSeedSpreading,
    ) -> Vec<(NodeId, SenseId, Vec<GapAnnotation>)> {
        let mut results = Vec::new();

        for &node_id in promoted_nodes {
            let sm = match senses.get(&node_id) {
                Some(sm) => sm,
                None => continue,
            };

            for sense in &sm.senses {
                let sense_id = sense.id;
                let actual: Vec<CompositionRef> = sense.compositions.clone();

                if actual.is_empty() {
                    continue;
                }

                let predictions = self.predict_expected_compositions(
                    node_id, sense_id, &actual, graph, senses, comp_index, batch_cache,
                );

                let gaps = self.compute_gaps(node_id, sense_id, &actual, predictions, graph);

                if !gaps.is_empty() {
                    let annotations: Vec<GapAnnotation> = gaps.iter()
                        .filter(|g| g.confidence >= self.config.min_gap_confidence_for_edge)
                        .map(|g| GapAnnotation {
                            gap_type: g.gap_type.clone(),
                            confidence: g.confidence,
                            target_node: g.structural_description
                                .expected_composition_targets
                                .first()
                                .copied()
                                .unwrap_or(0),
                            seed_trace: g.structural_description.seed_trace.clone(),
                        })
                        .collect();

                    if !annotations.is_empty() {
                        results.push((node_id, sense_id, annotations));
                    }
                }
            }
        }

        results
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_detection_config_defaults() {
        let config = GapDetectionConfig::default();
        assert!(config.enable_scalar);
        assert!(config.enable_presupposition);
        assert!((config.min_activation_energy - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_scalar_scale_index() {
        let mut index = ScalarScaleIndex::new();
        assert!(index.is_empty());

        let scale = ScalarScale {
            nodes: vec![1, 2, 3, 4],
            scale_label: "all > most > many > some".to_string(),
            dimension: "quantity".to_string(),
        };

        index.rebuild(vec![scale]);
        assert_eq!(index.len(), 1);

        // Node 4 (some) is at position 3
        assert_eq!(index.get_scale_position(4), Some((0, 3)));
        // Node 1 (all) is at position 0
        assert_eq!(index.get_scale_position(1), Some((0, 0)));
        // Unknown node
        assert_eq!(index.get_scale_position(99), None);
    }

    #[test]
    fn test_gap_type_classification() {
        let config = GapDetectionConfig::default();
        let detector = GapDetector::new(config);

        let evidence = GapEvidence::ScalarChain {
            scale: vec![1, 2, 3],
            used_index: 2,
            stronger_unused: vec![1, 2],
        };
        let gap_type = detector.classify_gap(&evidence, &RsvsGraph::new());
        assert_eq!(gap_type, GapType::ScalarImplicature);
    }

    #[test]
    fn test_confidence_values() {
        let config = GapDetectionConfig::default();
        let detector = GapDetector::new(config);

        // Scalar implicature: fixed 0.7
        let evidence = GapEvidence::ScalarChain {
            scale: vec![1, 2],
            used_index: 1,
            stronger_unused: vec![1],
        };
        assert!((detector.compute_confidence(&evidence) - 0.7).abs() < 0.01);

        // Seed activation: energy * 0.8
        let evidence = GapEvidence::SeedActivation {
            seed: 1,
            activated_area: vec![2],
            activation_energy: 0.5,
        };
        assert!((detector.compute_confidence(&evidence) - 0.4).abs() < 0.01);
    }
}
