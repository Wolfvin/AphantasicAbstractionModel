//! Engine 1: Compositional Blending — v10.0 Emergent Reasoning
//!
//! Core algorithm: BLEND(sense_A, sense_B) = shared_compositions ∪ divergent_A ∪ divergent_B
//!
//! When two interpretations (A and B) are both plausible for a node,
//! this engine blends their compositional structures to create a hybrid
//! sense A∧B that preserves shared structure and marks divergent parts.
//!
//! The key insight: blending A and B OPENS Possibility C — an emergent
//! meaning that was not visible from either A or B alone.
//!
//! Example:
//!   A: sense(dikhianati) = [(risk,0), (identity,0), (trust,0)]
//!   B: sense(harga_diri) = [(risk,0), (identity,0), (value,0)]
//!   Shared: [(risk,0), (identity,0)]
//!   Divergent_A: [(trust,0)]
//!   Divergent_B: [(value,0)]
//!   → Hybrid: sense(dikhianati∧harga_diri) = shared + divergent
//!   → Emergence: trust+value conflict → "dikhianatan terhadap harga diri"

use crate::batch_spreading::BatchSeedSpreading;
use crate::graph::RsvsGraph;
use crate::sense::SenseManager;
use crate::types::{
    BlendResult, CompositionRef, Edge, EdgeSource, Node, NodeId, NodeStatus, RelationType,
    SemanticMeta, SenseId, Tier,
};
use std::collections::{HashMap, HashSet};

/// Configuration for the Compositional Blending Engine.
#[derive(Debug, Clone)]
pub struct BlendingConfig {
    /// Minimum number of shared compositions to consider blending.
    pub min_shared_compositions: usize,
    /// Minimum blend quality to commit the hybrid to the graph.
    pub min_blend_quality: f32,
    /// Minimum emergence potential to flag as "opens C".
    pub min_emergence_potential: f32,
    /// Maximum blends per batch.
    pub max_blends_per_batch: usize,
}

impl Default for BlendingConfig {
    fn default() -> Self {
        Self {
            min_shared_compositions: 1,
            min_blend_quality: 0.2,
            min_emergence_potential: 0.3,
            max_blends_per_batch: 10,
        }
    }
}

/// The Compositional Blending Engine.
///
/// Merges sense compositions from two nodes to create hybrid senses.
/// This is the structural mechanism behind A+B→C emergence.
pub struct CompositionalBlendingEngine {
    /// Configuration.
    pub config: BlendingConfig,
}

impl CompositionalBlendingEngine {
    /// Create a new blending engine.
    pub fn new(config: BlendingConfig) -> Self {
        Self { config }
    }

    /// Blend two senses: merge compositions, compute quality and emergence.
    ///
    /// This is the core operation. Given sense_A of node_A and sense_B of
    /// node_B, it finds shared compositions (the structural overlap that
    /// makes blending meaningful) and divergent compositions (the tension
    /// that opens new meaning).
    pub fn blend_senses(
        &self,
        node_a: NodeId,
        sense_a: SenseId,
        comps_a: &[CompositionRef],
        node_b: NodeId,
        sense_b: SenseId,
        comps_b: &[CompositionRef],
    ) -> BlendResult {
        let set_a: HashSet<CompositionRef> = comps_a.iter().cloned().collect();
        let set_b: HashSet<CompositionRef> = comps_b.iter().cloned().collect();

        // Shared = intersection
        let shared: Vec<CompositionRef> = set_a.intersection(&set_b).cloned().collect();
        // Divergent from A = in A but not in B
        let divergent_a: Vec<CompositionRef> = set_a.difference(&set_b).cloned().collect();
        // Divergent from B = in B but not in A
        let divergent_b: Vec<CompositionRef> = set_b.difference(&set_a).cloned().collect();

        // Blend quality = shared / total_unique
        let total_unique = set_a.len() + set_b.len() - shared.len();
        let blend_quality = if total_unique > 0 {
            shared.len() as f32 / total_unique as f32
        } else {
            0.0
        };

        // Emergence potential = divergent from both sides
        // High when both A and B bring unique compositions to the blend
        let emergence_potential = if divergent_a.is_empty() || divergent_b.is_empty() {
            0.0 // No tension → no emergence
        } else {
            // Both sides have unique compositions → tension opens new meaning
            let divergent_ratio =
                (divergent_a.len() + divergent_b.len()) as f32 / total_unique.max(1) as f32;
            // Boost if divergent compositions come from different seed pathways
            (divergent_ratio * 1.5).min(1.0)
        };

        BlendResult {
            source_a: node_a,
            source_b: node_b,
            sense_a,
            sense_b,
            shared_compositions: shared,
            divergent_a,
            divergent_b,
            hybrid_node_id: None,
            blend_quality,
            emergence_potential,
        }
    }

    /// Find blend candidates among promoted nodes.
    ///
    /// Two nodes are blend candidates when:
    /// 1. Both have compositional senses
    /// 2. They share at least min_shared_compositions compositions
    /// 3. They have at least one divergent composition each
    pub fn find_blend_candidates(
        &self,
        promoted_nodes: &[NodeId],
        senses: &HashMap<NodeId, SenseManager>,
    ) -> Vec<(NodeId, SenseId, NodeId, SenseId)> {
        let mut candidates = Vec::new();

        // Collect all (node_id, sense_id, compositions) pairs
        let mut sense_data: Vec<(NodeId, SenseId, Vec<CompositionRef>)> = Vec::new();
        for &node_id in promoted_nodes {
            if let Some(sm) = senses.get(&node_id) {
                for sense in &sm.senses {
                    if !sense.compositions.is_empty() {
                        sense_data.push((node_id, sense.id, sense.compositions.clone()));
                    }
                }
            }
        }

        // Find pairs with shared compositions
        for i in 0..sense_data.len() {
            for j in (i + 1)..sense_data.len() {
                let (na, sa, ca) = &sense_data[i];
                let (nb, sb, cb) = &sense_data[j];
                if na == nb {
                    continue; // Don't blend same node
                }
                let set_a: HashSet<CompositionRef> = ca.iter().cloned().collect();
                let set_b: HashSet<CompositionRef> = cb.iter().cloned().collect();
                let shared_count = set_a.intersection(&set_b).count();
                if shared_count >= self.config.min_shared_compositions {
                    candidates.push((*na, *sa, *nb, *sb));
                }
            }
        }

        candidates
    }

    /// Process all promoted nodes for blending.
    ///
    /// Finds blend candidates, performs blending, and returns results.
    /// Does NOT modify the graph — that's done by the pipeline integration.
    pub fn process_batch(
        &self,
        promoted_nodes: &[NodeId],
        senses: &HashMap<NodeId, SenseManager>,
    ) -> Vec<BlendResult> {
        let candidates = self.find_blend_candidates(promoted_nodes, senses);
        let mut results = Vec::new();

        for (na, sa, nb, sb) in candidates {
            let comps_a = senses
                .get(&na)
                .and_then(|sm| sm.senses.iter().find(|s| s.id == sa))
                .map(|s| s.compositions.clone())
                .unwrap_or_default();
            let comps_b = senses
                .get(&nb)
                .and_then(|sm| sm.senses.iter().find(|s| s.id == sb))
                .map(|s| s.compositions.clone())
                .unwrap_or_default();

            if comps_a.is_empty() || comps_b.is_empty() {
                continue;
            }

            let blend = self.blend_senses(na, sa, &comps_a, nb, sb, &comps_b);

            if blend.blend_quality >= self.config.min_blend_quality {
                results.push(blend);
            }

            if results.len() >= self.config.max_blends_per_batch {
                break;
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
    fn blending_config_defaults() {
        let config = BlendingConfig::default();
        assert_eq!(config.min_shared_compositions, 1);
        assert!((config.min_blend_quality - 0.2).abs() < 0.01);
        assert!((config.min_emergence_potential - 0.3).abs() < 0.01);
    }

    #[test]
    fn blend_senses_finds_shared_and_divergent() {
        let engine = CompositionalBlendingEngine::new(BlendingConfig::default());

        let comps_a = vec![
            CompositionRef::new(1, 0), // risk
            CompositionRef::new(2, 0), // identity
            CompositionRef::new(3, 0), // trust (divergent)
        ];
        let comps_b = vec![
            CompositionRef::new(1, 0), // risk (shared)
            CompositionRef::new(2, 0), // identity (shared)
            CompositionRef::new(4, 0), // value (divergent)
        ];

        let result = engine.blend_senses(100, 0, &comps_a, 200, 0, &comps_b);

        // Shared: risk, identity = 2
        assert_eq!(result.shared_compositions.len(), 2);
        // Divergent A: trust = 1
        assert_eq!(result.divergent_a.len(), 1);
        // Divergent B: value = 1
        assert_eq!(result.divergent_b.len(), 1);
        // Blend quality = 2/4 = 0.5
        assert!((result.blend_quality - 0.5).abs() < 0.01);
        // Emergence potential > 0 (both sides have divergent compositions)
        assert!(result.emergence_potential > 0.0);
    }

    #[test]
    fn blend_no_emergence_when_one_side_no_divergent() {
        let engine = CompositionalBlendingEngine::new(BlendingConfig::default());

        // B is a subset of A — no divergent from B
        let comps_a = vec![
            CompositionRef::new(1, 0),
            CompositionRef::new(2, 0),
            CompositionRef::new(3, 0),
        ];
        let comps_b = vec![
            CompositionRef::new(1, 0),
            CompositionRef::new(2, 0),
        ];

        let result = engine.blend_senses(100, 0, &comps_a, 200, 0, &comps_b);

        assert_eq!(result.divergent_b.len(), 0);
        assert!((result.emergence_potential - 0.0).abs() < 0.01,
            "No emergence when one side has no divergent compositions");
    }

    #[test]
    fn blend_identical_senses_high_quality_no_emergence() {
        let engine = CompositionalBlendingEngine::new(BlendingConfig::default());

        let comps = vec![
            CompositionRef::new(1, 0),
            CompositionRef::new(2, 0),
        ];

        let result = engine.blend_senses(100, 0, &comps, 200, 0, &comps);

        assert!((result.blend_quality - 1.0).abs() < 0.01);
        assert_eq!(result.shared_compositions.len(), 2);
        assert_eq!(result.divergent_a.len(), 0);
        assert_eq!(result.divergent_b.len(), 0);
        assert!((result.emergence_potential - 0.0).abs() < 0.01,
            "Identical senses → no emergence");
    }
}
