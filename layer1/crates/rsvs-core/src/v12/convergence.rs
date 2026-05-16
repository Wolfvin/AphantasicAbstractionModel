//! # Convergence Detection — Structural Equivalence Between Compositions
//!
//! Ported from v8.3 `convergence.rs` (652 lines), adapted for the v12
//! Composition-based graph model.
//!
//! ## Algorithm
//!
//! Structural equivalence via composition overlap + co-occurrence exclusion:
//! 1. Collect eligible compositions: non-seed, confidence >= 0.3
//! 2. For each pair (throttled to `max_pairs_per_run`):
//!    - Skip already-detected pairs
//!    - Skip co-occurring compositions (cooc > 1)
//!    - Compute member overlap via Jaccard similarity
//!    - If overlap >= `min_overlap_threshold` → CONVERGENCE DETECTED
//! 3. Create `EquivalentOf` member links between converged compositions
//!
//! ## v12 Adaptation
//!
//! In v8.3, convergence compared sense compositions (Jaccard of CompositionRef sets).
//! In v12, we compare Composition members directly (Jaccard of node_id sets).
//!
//! ## Usage
//!
//! ```ignore
//! let detector = ConvergenceDetection::default();
//! let pairs = detector.detect(&graph);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::pipeline::{ErasedTransform, Graph, IngestResult};
use super::types::*;
use crate::types::NodeId;

// ========================================================================
// ConvergencePair — A Detected Structural Equivalence
// ========================================================================

/// A detected structural equivalence between two compositions.
///
/// When two compositions have high member overlap but never co-occur
/// in the same context, they may be translation equivalents or
/// different expressions of the same underlying concept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergencePair {
    /// The first composition.
    pub composition_a: CompositionId,
    /// The second composition.
    pub composition_b: CompositionId,
    /// Jaccard similarity of their member node sets.
    pub overlap: f32,
    /// Co-occurrence count (should be 0 or 1 for true convergence).
    pub cooccurrence: usize,
    /// Confidence that this is a true convergence.
    pub confidence: f32,
}

// ========================================================================
// ConvergenceConfig — Configuration
// ========================================================================

/// Configuration for convergence detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    /// Minimum overlap (Jaccard) to consider as convergence (default: 0.6).
    pub min_overlap: f32,
    /// Maximum co-occurrence count for equivalence (default: 1).
    /// Compositions that frequently co-occur are NOT equivalents —
    /// they just share context.
    pub max_cooc_for_equivalence: usize,
    /// Minimum confidence for a composition to be eligible (default: 0.3).
    pub min_confidence: f32,
    /// Maximum pairs to check per run (default: 500).
    pub max_pairs_per_run: usize,
    /// Role weight for role-weighted similarity (default: 0.6).
    /// Formula: similarity = α × role_jaccard + (1-α) × node_jaccard
    /// Higher α means role structure matters more than raw node overlap.
    pub role_weight: Option<f32>,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            min_overlap: 0.6,
            max_cooc_for_equivalence: 1,
            min_confidence: 0.3,
            max_pairs_per_run: 500,
            role_weight: Some(0.6),
        }
    }
}

// ========================================================================
// ConvergenceDetection — The Engine
// ========================================================================

/// Convergence detection engine (ported from v8.3, adapted for v12).
///
/// Detects structural equivalence between compositions:
/// - Same composition type + high member overlap + low co-occurrence
///   → likely same concept expressed differently
/// - Useful for cross-linguistic detection (e.g., "merah" ≡ "red")
/// - Creates `EquivalentOf` links between converged compositions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceDetection {
    /// Configuration.
    pub config: ConvergenceConfig,
    /// Previously detected pairs (to avoid re-detection).
    pub detected_pairs: HashSet<(CompositionId, CompositionId)>,
}

impl Default for ConvergenceDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceDetection {
    /// Create a new convergence detection engine.
    pub fn new() -> Self {
        Self {
            config: ConvergenceConfig::default(),
            detected_pairs: HashSet::new(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: ConvergenceConfig) -> Self {
        Self {
            config,
            detected_pairs: HashSet::new(),
        }
    }

    /// Detect convergence pairs in the graph.
    ///
    /// Returns a list of detected convergence pairs. Each pair represents
    /// two compositions that are structurally equivalent but have low
    /// co-occurrence — suggesting they may represent the same underlying
    /// concept expressed differently.
    pub fn detect(&mut self, graph: &Graph) -> Vec<ConvergencePair> {
        // Collect eligible compositions.
        let eligible: Vec<&Composition> = graph
            .compositions
            .values()
            .filter(|c| c.confidence >= self.config.min_confidence)
            .filter(|c| c.lifecycle != LifecycleState::Deprecated)
            .collect();

        let mut pairs = Vec::new();
        let mut checked = 0;

        for i in 0..eligible.len() {
            if checked >= self.config.max_pairs_per_run {
                break;
            }
            for j in (i + 1)..eligible.len() {
                if checked >= self.config.max_pairs_per_run {
                    break;
                }
                checked += 1;

                let comp_a = eligible[i];
                let comp_b = eligible[j];

                // Must be same composition type for convergence.
                if comp_a.composition_type != comp_b.composition_type {
                    continue;
                }

                // Skip already-detected pairs.
                let pair_key = if comp_a.id < comp_b.id {
                    (comp_a.id.clone(), comp_b.id.clone())
                } else {
                    (comp_b.id.clone(), comp_a.id.clone())
                };
                if self.detected_pairs.contains(&pair_key) {
                    continue;
                }

                // Check co-occurrence.
                let nodes_a: HashSet<NodeId> = comp_a.members.iter().map(|m| m.node_id).collect();
                let nodes_b: HashSet<NodeId> = comp_b.members.iter().map(|m| m.node_id).collect();

                // Co-occurrence: count how many nodes they share.
                let shared_count = nodes_a.intersection(&nodes_b).count();

                // If they share too many nodes (high co-occurrence),
                // they're not convergence candidates — they just share context.
                let cooc = graph.cooccurrence_count(
                    *nodes_a.iter().next().unwrap_or(&0),
                    *nodes_b.iter().next().unwrap_or(&0),
                );

                // Actually, we want to use the intersection size as co-occurrence
                // for the convergence check. The graph.cooccurrence_count() is for
                // node-level co-occurrence; we need composition-level.
                if shared_count > self.config.max_cooc_for_equivalence + 1 {
                    continue;
                }

                // Compute role-weighted similarity.
                let similarity = self.role_weighted_similarity(comp_a, comp_b);

                if similarity >= self.config.min_overlap {
                    let confidence = (similarity * 0.8).min(0.9);
                    pairs.push(ConvergencePair {
                        composition_a: comp_a.id.clone(),
                        composition_b: comp_b.id.clone(),
                        overlap: similarity,
                        cooccurrence: cooc,
                        confidence,
                    });

                    self.detected_pairs.insert(pair_key);
                }
            }
        }

        pairs
    }

    /// Compute structural similarity between two compositions.
    ///
    /// Uses Jaccard similarity of their member node sets.
    pub fn structural_similarity(a: &Composition, b: &Composition) -> f32 {
        let nodes_a: HashSet<NodeId> = a.members.iter().map(|m| m.node_id).collect();
        let nodes_b: HashSet<NodeId> = b.members.iter().map(|m| m.node_id).collect();

        if nodes_a.is_empty() && nodes_b.is_empty() {
            return 1.0;
        }

        let intersection = nodes_a.intersection(&nodes_b).count();
        let union = nodes_a.union(&nodes_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Compute role-aware structural similarity.
    ///
    /// Unlike `structural_similarity()`, this considers both the node ID
    /// AND the role when computing overlap. Two compositions are similar
    /// if they have the same nodes in the same roles.
    pub fn role_aware_similarity(a: &Composition, b: &Composition) -> f32 {
        if a.composition_type != b.composition_type {
            return 0.0;
        }

        let roles_a: HashSet<(SemanticRole, NodeId)> = a
            .members
            .iter()
            .map(|m| (m.role.clone(), m.node_id))
            .collect();
        let roles_b: HashSet<(SemanticRole, NodeId)> = b
            .members
            .iter()
            .map(|m| (m.role.clone(), m.node_id))
            .collect();

        if roles_a.is_empty() && roles_b.is_empty() {
            return 1.0;
        }

        let intersection = roles_a.intersection(&roles_b).count();
        let union = roles_a.union(&roles_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Compute node-level Jaccard similarity between two compositions.
    ///
    /// Pure node overlap without considering roles.
    pub fn node_jaccard(&self, comp_a: &Composition, comp_b: &Composition) -> f32 {
        let nodes_a: HashSet<NodeId> = comp_a.members.iter().map(|m| m.node_id).collect();
        let nodes_b: HashSet<NodeId> = comp_b.members.iter().map(|m| m.node_id).collect();

        if nodes_a.is_empty() && nodes_b.is_empty() {
            return 1.0;
        }

        let intersection = nodes_a.intersection(&nodes_b).count();
        let union = nodes_a.union(&nodes_b).count();

        if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
    }

    /// Compute role-weighted structural similarity between two compositions.
    ///
    /// Formula: α × role_jaccard + (1-α) × node_jaccard
    /// Default α = 0.6 (role structure matters more than raw node overlap).
    ///
    /// role_jaccard: fraction of matching (role_type, label) pairs
    /// node_jaccard: fraction of shared node_ids
    ///
    /// This fixes Known Limitation L1: plain Jaccard node-overlap doesn't
    /// capture role equivalence (e.g., "dokter memeriksa pasien" vs
    /// "tabib memeriksa orang_sakit" — same structure, different nodes).
    pub fn role_weighted_similarity(
        &self,
        comp_a: &Composition,
        comp_b: &Composition,
    ) -> f32 {
        let alpha = self.config.role_weight.unwrap_or(0.6);

        // Node Jaccard (existing logic)
        let node_jaccard = self.node_jaccard(comp_a, comp_b);

        // Role Jaccard — match on (role_type,) to capture structural equivalence.
        // Two compositions with the same role types (Agent, Predicate, Patient)
        // but different labels (dokter/tabib) should score high on role structure.
        let roles_a: HashSet<String> = comp_a.members.iter()
            .map(|m| format!("{:?}", m.role))
            .collect();
        let roles_b: HashSet<String> = comp_b.members.iter()
            .map(|m| format!("{:?}", m.role))
            .collect();

        let role_intersection = roles_a.intersection(&roles_b).count() as f32;
        let role_union = roles_a.union(&roles_b).count() as f32;

        let role_jaccard = if role_union > 0.0 { role_intersection / role_union } else { 0.0 };

        alpha * role_jaccard + (1.0 - alpha) * node_jaccard
    }

    /// Blend seed scores between two converged compositions.
    ///
    /// When convergence is detected (e.g., "merah" ≡ "red"), blend
    /// their seed scores by averaging.
    pub fn blend_seed_scores(a: &Composition, b: &Composition) -> HashMap<SeedPrimitive, f32> {
        let mut blended = HashMap::new();

        // Collect all seed primitives from both.
        let mut all_seeds: HashSet<SeedPrimitive> = HashSet::new();
        for seed in a.seed_scores.keys() {
            all_seeds.insert(seed.clone());
        }
        for seed in b.seed_scores.keys() {
            all_seeds.insert(seed.clone());
        }

        for seed in all_seeds {
            let a_score = a.seed_scores.get(&seed).copied().unwrap_or(0.0);
            let b_score = b.seed_scores.get(&seed).copied().unwrap_or(0.0);
            blended.insert(seed, (a_score + b_score) / 2.0);
        }

        blended
    }
}

// ========================================================================
// ConvergenceDetectionTransform — Pipeline Integration
// ========================================================================

/// Pipeline transform that detects structural convergence.
///
/// This is an optional enrichment transform that can be registered
/// after `SeedAnchor` to detect cross-linguistic or cross-context
/// structural equivalences.
#[derive(Debug, Clone, Default)]
pub struct ConvergenceDetectionTransform {
    /// The underlying convergence detection engine.
    pub engine: ConvergenceDetection,
}

impl ErasedTransform for ConvergenceDetectionTransform {
    fn id(&self) -> &'static str {
        "ConvergenceDetection"
    }

    fn execute(&self, _ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let mut engine = self.engine.clone();
        let pairs = engine.detect(graph);

        // Create EquivalentOf links between converged compositions.
        let mut edges_created = 0;
        for pair in &pairs {
            // Find a node from comp_b to link to (immutable borrow first).
            let pred_node = graph.compositions.get(&pair.composition_b)
                .and_then(|comp_b| comp_b.member_with_role(&SemanticRole::Predicate))
                .map(|m| (m.node_id, m.label.clone()));

            // Add EquivalentOf member to composition A (mutable borrow second).
            if let Some((node_id, label)) = pred_node {
                if let Some(comp_a) = graph.compositions.get_mut(&pair.composition_a) {
                    comp_a.members.push(CompositionMember {
                        node_id,
                        role: SemanticRole::EquivalentOf,
                        confidence: pair.confidence,
                        label,
                    });
                    edges_created += 1;
                }
            }
        }

        IngestResult {
            edges_created,
            ..IngestResult::default()
        }
    }
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_composition(id: &str, node_ids: &[NodeId], comp_type: CompositionType) -> Composition {
        let mut comp = Composition::default();
        comp.id = id.to_string();
        comp.composition_type = comp_type;
        comp.confidence = 0.7;
        comp.members = node_ids
            .iter()
            .enumerate()
            .map(|(i, &nid)| CompositionMember {
                node_id: nid,
                role: if i == 0 { SemanticRole::Predicate } else { SemanticRole::Arg0Agent },
                confidence: 0.8,
                label: format!("node_{}", nid),
            })
            .collect();
        comp
    }

    #[test]
    fn test_structural_similarity_identical() {
        let a = make_composition("a", &[1, 2, 3], CompositionType::Event);
        let b = make_composition("b", &[1, 2, 3], CompositionType::Event);
        let sim = ConvergenceDetection::structural_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_structural_similarity_disjoint() {
        let a = make_composition("a", &[1, 2], CompositionType::Event);
        let b = make_composition("b", &[3, 4], CompositionType::Event);
        let sim = ConvergenceDetection::structural_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_structural_similarity_partial() {
        let a = make_composition("a", &[1, 2, 3], CompositionType::Event);
        let b = make_composition("b", &[2, 3, 4], CompositionType::Event);
        let sim = ConvergenceDetection::structural_similarity(&a, &b);
        // Intersection: {2, 3} = 2
        // Union: {1, 2, 3, 4} = 4
        // Jaccard = 0.5
        assert!((sim - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_role_aware_similarity() {
        let mut a = Composition::default();
        a.id = "a".to_string();
        a.composition_type = CompositionType::Event;
        a.members = vec![
            CompositionMember { node_id: 1, role: SemanticRole::Arg0Agent, confidence: 0.9, label: "x".to_string() },
            CompositionMember { node_id: 2, role: SemanticRole::Arg1Patient, confidence: 0.8, label: "y".to_string() },
        ];

        let mut b = Composition::default();
        b.id = "b".to_string();
        b.composition_type = CompositionType::Event;
        b.members = vec![
            CompositionMember { node_id: 1, role: SemanticRole::Arg1Patient, confidence: 0.9, label: "x".to_string() },
            CompositionMember { node_id: 2, role: SemanticRole::Arg0Agent, confidence: 0.8, label: "y".to_string() },
        ];

        let sim = ConvergenceDetection::role_aware_similarity(&a, &b);
        // No exact (role, node_id) matches → similarity = 0.0
        assert!((sim - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_blend_seed_scores() {
        let mut a = Composition::default();
        a.seed_scores.insert(SeedPrimitive::Trust, 0.8);
        a.seed_scores.insert(SeedPrimitive::Risk, 0.4);

        let mut b = Composition::default();
        b.seed_scores.insert(SeedPrimitive::Risk, 0.6);
        b.seed_scores.insert(SeedPrimitive::Value, 0.7);

        let blended = ConvergenceDetection::blend_seed_scores(&a, &b);
        assert!((blended.get(&SeedPrimitive::Trust).unwrap() - 0.4).abs() < 0.01);
        assert!((blended.get(&SeedPrimitive::Risk).unwrap() - 0.5).abs() < 0.01);
        assert!((blended.get(&SeedPrimitive::Value).unwrap() - 0.35).abs() < 0.01);
    }

    #[test]
    fn test_detect_no_convergence() {
        let mut detector = ConvergenceDetection::new();
        let graph = Graph::new();
        let pairs = detector.detect(&graph);
        assert!(pairs.is_empty());
    }
}
