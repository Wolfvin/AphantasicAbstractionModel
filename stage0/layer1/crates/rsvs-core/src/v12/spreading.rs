//! # Spreading Activation — Energy Propagation Through Composition Edges
//!
//! Ported from v8.3 `spreading.rs` (351 lines) + `batch_spreading.rs` (302 lines),
//! adapted for the v12 Composition-based graph model.
//!
//! ## Algorithm
//!
//! Energy spreads along COMPOSITION edges (not just graph edges):
//! 1. Initialize seed nodes with `initial_energy`
//! 2. For each hop (up to `max_hops`):
//!    - `hop_energy = initial_energy × decay_factor^(hop+1)`
//!    - Find neighbors through composition membership
//!    - Unvisited nodes: add full `hop_energy`
//!    - Already-visited nodes: add `hop_energy × 0.5` (reinforcement)
//! 3. Sort by energy, truncate to `max_activated`, filter below `min_energy`
//!
//! ## v12 Adaptation
//!
//! In v8.3, spreading was node-to-node via `RsvsGraph::neighbors()`.
//! In v12, compositions connect nodes, so we spread through:
//! - Node → Composition → Node (2-hop per composition edge)
//! - Direct Composition → Composition (via shared nodes)
//!
//! ## Usage
//!
//! ```ignore
//! let activation = SpreadingActivation::default();
//! let seeds = vec![(node_id, 1.0), (another_node, 0.8)];
//! let energy_map = activation.spread(&seeds, &graph);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::pipeline::{ErasedTransform, Graph, IngestResult};
use super::types::*;
use crate::types::NodeId;

// ========================================================================
// ActivationMap — Result of Spreading Activation
// ========================================================================

/// Result of a spreading activation run.
///
/// Maps each activated `NodeId` to its energy level. Nodes with higher
/// energy are more strongly associated with the seed nodes.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActivationMap {
    /// Node ID → activation energy.
    pub energies: HashMap<NodeId, f32>,
    /// Which seed nodes were used (for provenance tracking).
    pub seeds: Vec<NodeId>,
    /// How many hops were computed.
    pub hops_computed: usize,
}

impl ActivationMap {
    /// Create a new empty activation map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the energy for a specific node.
    pub fn energy(&self, node_id: NodeId) -> f32 {
        self.energies.get(&node_id).copied().unwrap_or(0.0)
    }

    /// Get the top-N most activated nodes.
    pub fn top_n(&self, n: usize) -> Vec<(NodeId, f32)> {
        let mut entries: Vec<_> = self.energies.iter().map(|(&id, &e)| (id, e)).collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(n);
        entries
    }

    /// Get all nodes with energy above a threshold.
    pub fn above_threshold(&self, threshold: f32) -> Vec<(NodeId, f32)> {
        self.energies
            .iter()
            .filter(|(_, &e)| e >= threshold)
            .map(|(&id, &e)| (id, e))
            .collect()
    }

    /// Total number of activated nodes.
    pub fn len(&self) -> usize {
        self.energies.len()
    }

    /// Is the map empty?
    pub fn is_empty(&self) -> bool {
        self.energies.is_empty()
    }

    /// Merge another activation map into this one.
    /// For shared nodes, takes the maximum energy.
    pub fn merge_max(&mut self, other: &ActivationMap) {
        for (&node_id, &energy) in &other.energies {
            let entry = self.energies.entry(node_id).or_insert(0.0);
            *entry = (*entry).max(energy);
        }
    }

    /// Compute structural similarity (Jaccard) between two activation maps.
    /// Two nodes are "similar" if they activate similar neighborhoods.
    pub fn jaccard_similarity(&self, other: &ActivationMap) -> f32 {
        if self.energies.is_empty() && other.energies.is_empty() {
            return 1.0;
        }
        let set_a: std::collections::HashSet<NodeId> = self.energies.keys().copied().collect();
        let set_b: std::collections::HashSet<NodeId> = other.energies.keys().copied().collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        if union == 0 {
            return 0.0;
        }
        intersection as f32 / union as f32
    }
}

// ========================================================================
// SpreadingActivation — Configuration & Algorithm
// ========================================================================

/// Configuration for spreading activation (ported from v8.3).
///
/// Controls how energy propagates through the graph:
/// - `decay_factor`: How much energy is lost per hop (0.5 = half)
/// - `max_hops`: Maximum propagation distance
/// - `max_activated`: Maximum number of activated nodes to return
/// - `min_energy`: Minimum energy threshold for activation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadingConfig {
    /// Energy decay factor per hop (default: 0.5).
    pub decay_factor: f32,
    /// Maximum number of hops (default: 3).
    pub max_hops: usize,
    /// Maximum activated nodes to return (default: 50).
    pub max_activated: usize,
    /// Minimum energy for a node to be included (default: 0.01).
    pub min_energy: f32,
    /// Reinforcement factor for already-visited nodes (default: 0.5).
    pub reinforcement: f32,
}

impl Default for SpreadingConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.5,
            max_hops: 3,
            max_activated: 50,
            min_energy: 0.01,
            reinforcement: 0.5,
        }
    }
}

/// Spreading activation engine (ported from v8.3, adapted for v12).
///
/// Propagates activation energy through composition edges in the v12 graph.
/// This is the core mechanism for:
/// - Finding related nodes
/// - Computing structural similarity
/// - Guiding gap detection (nodes strongly activated by seeds but not in
///   actual compositions → predicted gaps)
/// - Attention scoring (which nodes are most relevant to current context)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadingActivation {
    /// Configuration for the activation algorithm.
    pub config: SpreadingConfig,
}

impl Default for SpreadingActivation {
    fn default() -> Self {
        Self::new()
    }
}

impl SpreadingActivation {
    /// Create a new spreading activation engine with default config.
    pub fn new() -> Self {
        Self {
            config: SpreadingConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: SpreadingConfig) -> Self {
        Self { config }
    }

    /// Run spreading activation from a set of seed nodes.
    ///
    /// # Arguments
    ///
    /// * `seeds` — Slice of (NodeId, initial_energy) pairs.
    /// * `graph` — The v12 graph to propagate through.
    ///
    /// # Returns
    ///
    /// An [`ActivationMap`] with energy levels for all activated nodes.
    pub fn spread(&self, seeds: &[(NodeId, f32)], graph: &Graph) -> ActivationMap {
        let mut energies: HashMap<NodeId, f32> = HashMap::new();
        let seed_ids: Vec<NodeId> = seeds.iter().map(|(id, _)| *id).collect();

        // Initialize seeds.
        for &(node_id, energy) in seeds {
            *energies.entry(node_id).or_insert(0.0) += energy;
        }

        // Propagate through composition edges.
        for hop in 0..self.config.max_hops {
            let hop_energy_base = self.config.decay_factor.powi((hop + 1) as i32);
            let mut new_energies: HashMap<NodeId, f32> = HashMap::new();

            // For each currently active node, find neighbors through compositions.
            for (&active_node, &current_energy) in &energies {
                let hop_energy = current_energy * hop_energy_base;
                if hop_energy < self.config.min_energy {
                    continue;
                }

                // Find all compositions this node participates in.
                for composition in graph.compositions.values() {
                    let is_member = composition.members.iter().any(|m| m.node_id == active_node);
                    if !is_member {
                        continue;
                    }

                    // Spread to all other members of this composition.
                    for member in &composition.members {
                        if member.node_id == active_node {
                            continue;
                        }

                        let boost = if energies.contains_key(&member.node_id) {
                            // Already activated: reinforcement.
                            hop_energy * self.config.reinforcement
                        } else {
                            // Newly activated: full energy.
                            hop_energy
                        };

                        *new_energies.entry(member.node_id).or_insert(0.0) += boost;
                    }
                }
            }

            // Merge new energies into the main map.
            for (node_id, energy) in new_energies {
                *energies.entry(node_id).or_insert(0.0) += energy;
            }
        }

        // Filter by minimum energy.
        energies.retain(|_, energy| *energy >= self.config.min_energy);

        // Truncate to max_activated.
        if energies.len() > self.config.max_activated {
            let mut entries: Vec<_> = energies.into_iter().collect();
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            entries.truncate(self.config.max_activated);
            energies = entries.into_iter().collect();
        }

        ActivationMap {
            energies,
            seeds: seed_ids,
            hops_computed: self.config.max_hops,
        }
    }

    /// Run targeted spreading with grounding score adjustment.
    ///
    /// Ported from v8.3 `targeted_spread()`: adjusts initial energy by
    /// the grounding score of each seed node.
    ///
    /// `adjusted_energy = base_energy × (0.5 + 0.5 × grounding_score)`
    pub fn targeted_spread(
        &self,
        seeds: &[(NodeId, f32, f32)], // (node_id, base_energy, grounding_score)
        graph: &Graph,
    ) -> ActivationMap {
        let adjusted_seeds: Vec<(NodeId, f32)> = seeds
            .iter()
            .map(|&(node_id, base_energy, grounding_score)| {
                let adjusted = base_energy * (0.5 + 0.5 * grounding_score);
                (node_id, adjusted)
            })
            .collect();
        self.spread(&adjusted_seeds, graph)
    }

    /// Compute the attention score for a node relative to the current context.
    ///
    /// Ported from v8.3 `attention.rs` hard selection formula:
    /// ```text
    /// score(t, c) = α × NPMI(t,c) + β × Jaccard(A(t), A(c)) + γ × cooc(t,c)
    /// ```
    ///
    /// Default weights: α=0.4, β=0.4, γ=0.2
    ///
    /// **Note**: This is an external API utility, not wired into the default pipeline.
    /// The CVE (Compositional Verbalization Engine) uses its own scoring in
    /// `build_reasoning_path()`. Wiring `attention_score()` into CVE would change
    /// existing scoring behavior and is thus a potential regression. Use this method
    /// for custom attention-based queries from external callers.
    ///
    /// **Audit v6 (P3-12)**: Not called by any internal code or PyO3 bindings.
    /// The Python FFI uses `pipeline.similarity()` instead which blends Jaccard
    /// with spreading activation internally. Retained for potential future FFI use.
    /// If unused after 2 releases, remove.
    pub fn attention_score(
        &self,
        target: NodeId,
        context_seeds: &[(NodeId, f32)],
        graph: &Graph,
    ) -> f32 {
        // Compute activation maps for target and context.
        let target_seeds = vec![(target, 1.0)];
        let target_map = self.spread(&target_seeds, graph);
        let context_map = self.spread(context_seeds, graph);

        // Jaccard component (β = 0.4).
        let jaccard = target_map.jaccard_similarity(&context_map);
        let beta = 0.4;

        // Co-occurrence component (γ = 0.2).
        let cooc = self.compute_cooccurrence(target, context_seeds, graph);
        let gamma = 0.2;

        // NPMI component (α = 0.4) — simplified as activation overlap.
        let target_energy = context_map.energy(target);
        let npm_i = if target_energy > 0.0 {
            target_energy
        } else {
            0.0
        };
        let alpha = 0.4;

        alpha * npm_i + beta * jaccard + gamma * cooc
    }

    /// Compute co-occurrence score between a target node and context seeds.
    fn compute_cooccurrence(
        &self,
        target: NodeId,
        context_seeds: &[(NodeId, f32)],
        graph: &Graph,
    ) -> f32 {
        let mut total_cooc = 0.0f32;
        let mut count = 0;

        for &(seed_id, _) in context_seeds {
            let cooc = graph.cooccurrence_count(target, seed_id);
            total_cooc += cooc as f32;
            count += 1;
        }

        if count == 0 {
            0.0
        } else {
            // Normalize to [0, 1] range.
            let avg = total_cooc / count as f32;
            (avg / (avg + 2.0)).min(1.0) // Softmax-style normalization
        }
    }
}

// ========================================================================
// SpreadingActivationTransform — Pipeline Integration
// ========================================================================

/// Pipeline transform that runs spreading activation after SeedAnchor.
///
/// This transform computes activation maps for all seed nodes and stores
/// the results in the PipelineContext for use by downstream transforms
/// (ConvergenceDetection, enhanced gap detection, attention scoring).
///
/// It is NOT in the default pipeline by default — it is an optional
/// enrichment transform that can be registered when needed.
#[derive(Debug, Clone, Default)]
pub struct SpreadingActivationTransform {
    /// The underlying spreading activation engine.
    pub engine: SpreadingActivation,
}

impl SpreadingActivationTransform {
    /// Create a new transform with default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom configuration.
    pub fn with_config(config: SpreadingConfig) -> Self {
        Self {
            engine: SpreadingActivation::with_config(config),
        }
    }
}

impl ErasedTransform for SpreadingActivationTransform {
    fn id(&self) -> &'static str {
        "SpreadingActivation"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        // Collect seed nodes with grounding score (confidence as proxy).
        let mut seeds_with_grounding: Vec<(NodeId, f32, f32)> = Vec::new();

        for composition in graph.compositions.values() {
            if composition.seed_scores.is_empty() {
                continue;
            }
            let avg_seed: f32 = composition.seed_scores.values().sum::<f32>()
                / composition.seed_scores.len() as f32;
            if avg_seed < 0.01 {
                continue;
            }

            // Use composition confidence as grounding score proxy.
            let grounding_score = composition.confidence;

            for member in &composition.members {
                seeds_with_grounding.push((member.node_id, avg_seed * member.confidence, grounding_score));
            }
        }

        if seeds_with_grounding.is_empty() {
            return IngestResult::new();
        }

        // Use targeted_spread() with grounding-score-adjusted energy.
        // Audit v4 fix: Store the activation map in PipelineContext
        // so downstream transforms can use it. Previously, the map was
        // computed but discarded.
        let activation_map = self.engine.targeted_spread(&seeds_with_grounding, graph);
        ctx.last_activation_energies = activation_map.energies;

        IngestResult {
            atoms_created: seeds_with_grounding.len(),
            compositions_created: 0,
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
    #![allow(clippy::field_reassign_with_default)]
    use super::*;

    #[test]
    fn test_empty_spread() {
        let engine = SpreadingActivation::new();
        let graph = Graph::new();
        let result = engine.spread(&[], &graph);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_seed_no_compositions() {
        let engine = SpreadingActivation::new();
        let mut graph = Graph::new();
        let node_id = graph.ensure_node("test");
        let result = engine.spread(&[(node_id, 1.0)], &graph);
        assert_eq!(result.len(), 1);
        assert!(result.energy(node_id) > 0.0);
    }

    #[test]
    fn test_spread_through_composition() {
        let engine = SpreadingActivation::new();
        let mut graph = Graph::new();

        // Create nodes.
        let node_a = graph.ensure_node("alpha");
        let node_b = graph.ensure_node("beta");
        let node_c = graph.ensure_node("gamma");

        // Create a composition connecting all three.
        let mut comp = Composition::default();
        comp.id = CompositionId::new("comp_test".to_string());
        comp.composition_type = CompositionType::Event;
        comp.members = vec![
            CompositionMember {
                node_id: node_a,
                role: SemanticRole::Arg0Agent,
                confidence: 0.9,
                label: "alpha".to_string(),
                source: None,
            },
            CompositionMember {
                node_id: node_b,
                role: SemanticRole::Arg1Patient,
                confidence: 0.8,
                label: "beta".to_string(),
                source: None,
            },
            CompositionMember {
                node_id: node_c,
                role: SemanticRole::Cause,
                confidence: 0.7,
                label: "gamma".to_string(),
                source: None,
            },
        ];
        graph.compositions.insert(CompositionId::new("comp_test".to_string()), comp);

        // Spread from node_a.
        let result = engine.spread(&[(node_a, 1.0)], &graph);

        // node_b and node_c should be activated.
        assert!(
            result.energy(node_b) > 0.0,
            "node_b should be activated through composition"
        );
        assert!(
            result.energy(node_c) > 0.0,
            "node_c should be activated through composition"
        );
        // node_a itself should have the highest energy.
        assert!(result.energy(node_a) >= result.energy(node_b));
    }

    #[test]
    fn test_activation_map_top_n() {
        let mut map = ActivationMap::new();
        map.energies.insert(1, 0.5);
        map.energies.insert(2, 0.9);
        map.energies.insert(3, 0.3);

        let top = map.top_n(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 2); // Highest energy first.
    }

    #[test]
    fn test_jaccard_similarity() {
        let mut map_a = ActivationMap::new();
        map_a.energies.insert(1, 0.5);
        map_a.energies.insert(2, 0.3);
        map_a.energies.insert(3, 0.1);

        let mut map_b = ActivationMap::new();
        map_b.energies.insert(2, 0.4);
        map_b.energies.insert(3, 0.2);
        map_b.energies.insert(4, 0.6);

        let sim = map_a.jaccard_similarity(&map_b);
        // Intersection: {2, 3} = 2 nodes
        // Union: {1, 2, 3, 4} = 4 nodes
        // Jaccard = 2/4 = 0.5
        assert!((sim - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_decay_reduces_energy() {
        let config = SpreadingConfig {
            decay_factor: 0.5,
            max_hops: 1,
            ..SpreadingConfig::default()
        };
        let engine = SpreadingActivation::with_config(config);
        let mut graph = Graph::new();

        let node_a = graph.ensure_node("a");
        let node_b = graph.ensure_node("b");

        let mut comp = Composition::default();
        comp.id = CompositionId::new("comp_ab".to_string());
        comp.members = vec![
            CompositionMember {
                node_id: node_a,
                role: SemanticRole::Predicate,
                confidence: 1.0,
                label: "a".to_string(),
                source: None,
            },
            CompositionMember {
                node_id: node_b,
                role: SemanticRole::Arg0Agent,
                confidence: 1.0,
                label: "b".to_string(),
                source: None,
            },
        ];
        graph.compositions.insert(CompositionId::new("comp_ab".to_string()), comp);

        let result = engine.spread(&[(node_a, 1.0)], &graph);

        // node_b should have less energy than node_a due to decay.
        assert!(result.energy(node_b) < result.energy(node_a));
    }
}
