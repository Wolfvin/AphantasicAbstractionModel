//! Engine 3: Pattern Mining — v10.0 Emergent Reasoning
//!
//! Core algorithm:
//!   1. Scan all composition structures in the graph
//!   2. Detect composition pairs that appear in ≥ min_support nodes
//!   3. Create "named pattern" nodes that abstract recurring structures
//!
//! This is the structural foundation for recognizing that certain
//! compositions "go together" across multiple concepts. When the
//! composition pair (risk, identity) appears in dikhianati, trauma,
//! sakit, and other nodes, the engine creates a named pattern node
//! like "kekerasan_terhadap_identitas" (violence against identity).
//!
//! The named pattern node becomes a first-class citizen in the graph,
//! with edges from all exhibiting nodes. This allows the system to
//! recognize new instances of the pattern by structural similarity.

use crate::batch_spreading::BatchSeedSpreading;
use crate::composition_index::CompositionIndex;
use crate::graph::RsvsGraph;
use crate::sense::SenseManager;
use crate::types::{
    CompositionRef, NamedPattern, Node, NodeId, NodeStatus, RelationType, SemanticMeta,
    SenseId, Tier, EdgeSource, Edge,
};
use std::collections::{HashMap, HashSet};

/// Configuration for the Pattern Mining Engine.
#[derive(Debug, Clone)]
pub struct PatternMiningConfig {
    /// Minimum number of nodes exhibiting a pattern to create a named pattern.
    pub min_support: usize,
    /// Minimum confidence for a named pattern.
    pub min_pattern_confidence: f32,
    /// Maximum patterns per batch.
    pub max_patterns_per_batch: usize,
    /// Only consider compositions that reference seed nodes.
    pub seed_compositions_only: bool,
}

impl Default for PatternMiningConfig {
    fn default() -> Self {
        Self {
            min_support: 2,
            min_pattern_confidence: 0.3,
            max_patterns_per_batch: 10,
            seed_compositions_only: true,
        }
    }
}

/// The Pattern Mining Engine.
///
/// Detects recurring composition structures and creates named pattern nodes.
pub struct PatternMiningEngine {
    /// Configuration.
    pub config: PatternMiningConfig,
    /// Previously discovered patterns: seed_pair → NamedPattern.
    pub discovered_patterns: HashMap<(NodeId, NodeId), NamedPattern>,
}

impl PatternMiningEngine {
    /// Create a new pattern mining engine.
    pub fn new(config: PatternMiningConfig) -> Self {
        Self {
            config,
            discovered_patterns: HashMap::new(),
        }
    }

    /// Extract all seed composition pairs from a node's senses.
    ///
    /// Returns pairs of (seed_id_a, seed_id_b) where both are present
    /// in the same sense's compositions. Only considers seed nodes if
    /// seed_compositions_only is enabled.
    pub fn extract_seed_pairs(
        &self,
        node_id: NodeId,
        senses: &HashMap<NodeId, SenseManager>,
        seed_node_ids: &HashSet<NodeId>,
    ) -> Vec<(NodeId, NodeId)> {
        let mut pairs = Vec::new();

        if let Some(sm) = senses.get(&node_id) {
            for sense in &sm.senses {
                // Collect seed compositions
                let seed_comps: Vec<NodeId> = if self.config.seed_compositions_only {
                    sense.compositions.iter()
                        .map(|c| c.node_id)
                        .filter(|id| seed_node_ids.contains(id))
                        .collect()
                } else {
                    sense.compositions.iter().map(|c| c.node_id).collect()
                };

                // Generate all pairs (sorted for consistency)
                for i in 0..seed_comps.len() {
                    for j in (i + 1)..seed_comps.len() {
                        let a = seed_comps[i].min(seed_comps[j]);
                        let b = seed_comps[i].max(seed_comps[j]);
                        pairs.push((a, b));
                    }
                }
            }
        }

        pairs.sort();
        pairs.dedup();
        pairs
    }

    /// Scan all promoted nodes for recurring composition pairs.
    ///
    /// Returns a map from (seed_a, seed_b) → list of nodes exhibiting that pair.
    pub fn scan_for_patterns(
        &self,
        promoted_nodes: &[NodeId],
        senses: &HashMap<NodeId, SenseManager>,
        seed_node_ids: &HashSet<NodeId>,
    ) -> HashMap<(NodeId, NodeId), Vec<NodeId>> {
        let mut pair_to_nodes: HashMap<(NodeId, NodeId), Vec<NodeId>> = HashMap::new();

        for &node_id in promoted_nodes {
            let pairs = self.extract_seed_pairs(node_id, senses, seed_node_ids);
            for pair in pairs {
                pair_to_nodes.entry(pair).or_default().push(node_id);
            }
        }

        // Filter by min_support
        pair_to_nodes.retain(|_, nodes| nodes.len() >= self.config.min_support);

        pair_to_nodes
    }

    /// Create a named pattern from a recurring composition pair.
    ///
    /// The pattern gets a label based on the seed labels, and is
    /// recorded as a first-class node in the graph.
    pub fn create_pattern(
        &mut self,
        seed_a: NodeId,
        seed_b: NodeId,
        exhibiting_nodes: &[NodeId],
        graph: &RsvsGraph,
    ) -> NamedPattern {
        // Generate label from seed node labels
        let label_a = graph
            .get_node(seed_a)
            .map(|n| n.label.clone())
            .unwrap_or_else(|| format!("seed_{}", seed_a));
        let label_b = graph
            .get_node(seed_b)
            .map(|n| n.label.clone())
            .unwrap_or_else(|| format!("seed_{}", seed_b));

        let label = format!("pola_{}_{}", label_a, label_b);

        // Defining compositions: the seed pair
        let defining_compositions = vec![
            CompositionRef::new(seed_a, 0),
            CompositionRef::new(seed_b, 0),
        ];

        // Confidence scales with support count
        let confidence = (0.3 + 0.1 * exhibiting_nodes.len() as f32).min(1.0);

        let pattern = NamedPattern {
            node_id: 0, // Will be assigned when committed to graph
            label,
            seed_composition: (seed_a, seed_b),
            exhibiting_nodes: exhibiting_nodes.to_vec(),
            support_count: exhibiting_nodes.len(),
            confidence,
            defining_compositions,
        };

        // Cache the pattern
        self.discovered_patterns.insert((seed_a, seed_b), pattern.clone());

        pattern
    }

    /// Process batch: scan for patterns, create named patterns.
    ///
    /// Returns newly discovered patterns (not yet committed to graph).
    pub fn process_batch(
        &mut self,
        promoted_nodes: &[NodeId],
        senses: &HashMap<NodeId, SenseManager>,
        seed_node_ids: &HashSet<NodeId>,
        graph: &RsvsGraph,
    ) -> Vec<NamedPattern> {
        let pair_map = self.scan_for_patterns(promoted_nodes, senses, seed_node_ids);

        let mut new_patterns = Vec::new();

        for ((seed_a, seed_b), nodes) in pair_map {
            // Skip if already discovered
            if self.discovered_patterns.contains_key(&(seed_a, seed_b)) {
                continue;
            }

            let pattern = self.create_pattern(seed_a, seed_b, &nodes, graph);

            if pattern.confidence >= self.config.min_pattern_confidence {
                new_patterns.push(pattern);
            }

            if new_patterns.len() >= self.config.max_patterns_per_batch {
                break;
            }
        }

        new_patterns
    }

    /// Get all discovered patterns.
    pub fn patterns(&self) -> &HashMap<(NodeId, NodeId), NamedPattern> {
        &self.discovered_patterns
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CompressionState;

    fn make_test_senses(
        node_id: NodeId,
        seed_comp_ids: Vec<NodeId>,
    ) -> (NodeId, SenseManager) {
        let compositions: Vec<CompositionRef> =
            seed_comp_ids.iter().map(|&id| CompositionRef::new(id, 0)).collect();
        let mut sm = SenseManager::new(crate::sense::SenseConfig::default());
        // Manually create a sense with these compositions
        sm.senses.push(crate::sense::Sense::new_compositional(
            0,
            compositions,
            vec![],
            1,
        ));
        (node_id, sm)
    }

    #[test]
    fn pattern_mining_config_defaults() {
        let config = PatternMiningConfig::default();
        assert_eq!(config.min_support, 2);
        assert!((config.min_pattern_confidence - 0.3).abs() < 0.01);
        assert!(config.seed_compositions_only);
    }

    #[test]
    fn extract_seed_pairs_finds_pairs() {
        let engine = PatternMiningEngine::new(PatternMiningConfig::default());
        let seed_ids: HashSet<NodeId> = vec![1, 2, 3, 4].into_iter().collect();

        let (_, sm) = make_test_senses(100, vec![1, 2, 3]); // risk, identity, trust
        let mut senses = HashMap::new();
        senses.insert(100, sm);

        let pairs = engine.extract_seed_pairs(100, &senses, &seed_ids);

        // Should find: (1,2), (1,3), (2,3)
        assert_eq!(pairs.len(), 3, "Should find 3 seed pairs from 3 seeds");
        assert!(pairs.contains(&(1, 2)));
        assert!(pairs.contains(&(1, 3)));
        assert!(pairs.contains(&(2, 3)));
    }

    #[test]
    fn scan_for_patterns_requires_min_support() {
        let mut engine = PatternMiningEngine::new(PatternMiningConfig {
            min_support: 2,
            ..PatternMiningConfig::default()
        });
        let seed_ids: HashSet<NodeId> = vec![1, 2].into_iter().collect();

        let mut senses = HashMap::new();
        let (_, sm1) = make_test_senses(100, vec![1, 2]);
        let (_, sm2) = make_test_senses(200, vec![1, 2]);
        let (_, sm3) = make_test_senses(300, vec![1]); // Only 1 seed, won't form pair
        senses.insert(100, sm1);
        senses.insert(200, sm2);
        senses.insert(300, sm3);

        let result = engine.scan_for_patterns(&[100, 200, 300], &senses, &seed_ids);

        // Pair (1,2) appears in nodes 100 and 200 → support = 2 ≥ min_support
        assert!(result.contains_key(&(1, 2)), "Pair (1,2) should be found with support 2");
        assert_eq!(result[&(1, 2)].len(), 2);
    }

    #[test]
    fn pattern_label_from_seed_names() {
        let mut engine = PatternMiningEngine::new(PatternMiningConfig::default());

        let mut graph = crate::graph::RsvsGraph::new();
        graph.insert_node(Node {
            label: "risk".to_string(),
            ..Node::default()
        }).unwrap();
        graph.insert_node(Node {
            label: "identity".to_string(),
            ..Node::default()
        }).unwrap();

        let pattern = engine.create_pattern(1, 2, &[100, 200], &graph);
        assert!(pattern.label.contains("risk") || pattern.label.contains("identity"),
            "Pattern label should reference seed names");
    }
}
