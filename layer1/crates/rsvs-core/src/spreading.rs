//! Spreading Activation for RSVS v7.0 — Inspired by Losion's Episodic Memory
//!
//! Losion's EpisodicMemory uses spreading activation to retrieve related
//! experiences: activating one node spreads energy to its neighbors,
//! with decay based on distance. This enables accessing networks of
//! related experiences without explicit traversal.
//!
//! Adapted for RSVS's structural domain:
//! - Nodes are activated with initial energy
//! - Energy spreads along composition edges (not just graph edges)
//! - Each hop reduces energy by a decay factor
//! - Activation accumulates (multiple sources reinforce)
//! - Results: ranked list of activated nodes with their energy levels
//!
//! Key difference from Losion: RSVS spreads along COMPOSITION references
//! (structural meaning connections), not just co-occurrence edges.
//! This means spreading follows the "meaning is structural" principle.

use crate::composition_index::CompositionIndex;
use crate::sense::SenseManager;
use crate::types::{NodeId, CompositionRef};
use std::collections::{HashMap, HashSet};

// -----------------------------------------------------------------------
// SpreadingActivationConfig
// -----------------------------------------------------------------------

/// Configuration for the spreading activation engine.
#[derive(Debug, Clone)]
pub struct SpreadingActivationConfig {
    /// Energy decay factor per hop. Default: 0.5 (each hop halves energy).
    /// Higher = energy travels further. Lower = more localized activation.
    pub decay_factor: f32,
    /// Maximum number of hops from initial activation. Default: 3.
    /// Controls how far the activation wave travels.
    pub max_hops: usize,
    /// Minimum energy threshold. Nodes below this are not activated. Default: 0.01.
    pub min_energy: f32,
    /// Maximum number of activated nodes to return. Default: 50.
    pub max_activated: usize,
    /// Whether to use composition edges (true) or graph edges (false).
    /// Default: true (follow structural meaning, not just co-occurrence).
    pub use_composition_edges: bool,
}

impl Default for SpreadingActivationConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.5,
            max_hops: 3,
            min_energy: 0.01,
            max_activated: 50,
            use_composition_edges: true,
        }
    }
}

// -----------------------------------------------------------------------
// ActivationResult
// -----------------------------------------------------------------------

/// Result of a spreading activation query.
#[derive(Debug, Clone)]
pub struct ActivationResult {
    /// Activated nodes with their energy levels, sorted by energy descending.
    pub activated: Vec<(NodeId, f32)>,
    /// Total energy in the system (for diagnostics).
    pub total_energy: f32,
    /// Number of hops performed.
    pub hops_performed: usize,
}

// -----------------------------------------------------------------------
// SpreadingActivation
// -----------------------------------------------------------------------

/// Spreading activation engine — activates related nodes through composition edges.
///
/// Inspired by Losion's EpisodicMemory spreading activation, which enables
/// retrieving networks of related experiences without explicit traversal.
///
/// In RSVS, spreading follows COMPOSITION edges: if node A's sense is composed
/// from [(B, 0), (C, 0)], then activating A spreads energy to B and C.
/// This is the structural equivalent of semantic priming in cognitive science.
///
/// # Example
///
/// ```ignore
/// let sa = SpreadingActivation::new(SpreadingActivationConfig::default());
/// let result = sa.spread(
///     &[raja_id],
///     1.0,  // initial energy
///     &senses,
///     &composition_index,
/// );
/// // result.activated contains all nodes structurally related to "raja",
/// // ranked by their accumulated activation energy.
/// ```
pub struct SpreadingActivation {
    pub config: SpreadingActivationConfig,
}

impl SpreadingActivation {
    /// Create a new spreading activation engine.
    pub fn new(config: SpreadingActivationConfig) -> Self {
        Self { config }
    }

    /// Run spreading activation from a set of seed nodes.
    ///
    /// Each seed node receives `initial_energy`. Energy then spreads
    /// through composition edges, decaying by `decay_factor` per hop.
    /// Nodes accumulate energy from multiple sources (additive).
    ///
    /// Returns the activated nodes ranked by their total energy.
    pub fn spread(
        &self,
        seeds: &[NodeId],
        initial_energy: f32,
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
    ) -> ActivationResult {
        let mut energy_map: HashMap<NodeId, f32> = HashMap::new();
        let mut visited: HashSet<NodeId> = HashSet::new();

        // Phase 1: Initialize seed nodes
        for &seed in seeds {
            *energy_map.entry(seed).or_insert(0.0) += initial_energy;
            visited.insert(seed);
        }

        let mut current_frontier: Vec<NodeId> = seeds.to_vec();
        let mut hops_performed = 0;

        // Phase 2: Spread energy through composition edges
        for hop in 0..self.config.max_hops {
            if current_frontier.is_empty() {
                break;
            }

            let mut next_frontier: Vec<NodeId> = Vec::new();
            let hop_energy = initial_energy * self.config.decay_factor.powi((hop + 1) as i32);

            if hop_energy < self.config.min_energy {
                break; // Energy too low to continue
            }

            for &node_id in &current_frontier {
                // Find neighbors through composition edges
                let neighbors = if self.config.use_composition_edges {
                    self.composition_neighbors(node_id, senses, comp_index)
                } else {
                    // Fallback: use dependents from composition index
                    comp_index.dependents_of_node(node_id).into_iter().collect()
                };

                for neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        *energy_map.entry(neighbor).or_insert(0.0) += hop_energy;
                        visited.insert(neighbor);
                        next_frontier.push(neighbor);
                    } else if let Some(existing) = energy_map.get_mut(&neighbor) {
                        // Reinforcement: multiple paths to same node increase energy
                        *existing += hop_energy * 0.5; // Half energy for reinforcement
                    }
                }
            }

            current_frontier = next_frontier;
            hops_performed = hop + 1;
        }

        // Phase 3: Sort by energy and truncate
        let mut activated: Vec<(NodeId, f32)> = energy_map
            .into_iter()
            .filter(|(_, energy)| *energy >= self.config.min_energy)
            .collect();

        activated.sort_by(|a, b| b.1.total_cmp(&a.1));
        activated.truncate(self.config.max_activated);

        let total_energy: f32 = activated.iter().map(|(_, e)| *e).sum();

        ActivationResult {
            activated,
            total_energy,
            hops_performed,
        }
    }

    /// Get the neighbors of a node through composition edges.
    ///
    /// Two types of composition neighbors:
    /// 1. **Outgoing**: Nodes that this node's senses reference as compositions
    /// 2. **Incoming**: Nodes whose senses reference this node as a composition
    fn composition_neighbors(
        &self,
        node_id: NodeId,
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
    ) -> Vec<NodeId> {
        let mut neighbors = HashSet::new();

        // Outgoing: this node's compositions reference other nodes
        if let Some(sm) = senses.get(&node_id) {
            for sense in &sm.senses {
                for comp in &sense.compositions {
                    neighbors.insert(comp.node_id);
                }
            }
        }

        // Incoming: other nodes reference this node in their compositions
        let dependents = comp_index.dependents_of_node(node_id);
        for dep in dependents {
            neighbors.insert(dep);
        }

        neighbors.into_iter().collect()
    }

    /// Run targeted spreading from a single node with adaptive energy.
    ///
    /// The initial energy is adjusted based on the node's grounding score:
    /// well-grounded nodes get more energy (they're more reliable seeds).
    pub fn targeted_spread(
        &self,
        seed: NodeId,
        base_energy: f32,
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
    ) -> ActivationResult {
        // Adjust energy based on grounding score of the seed
        let grounding = senses
            .get(&seed)
            .and_then(|sm| sm.senses.first())
            .map(|s| s.grounding.score())
            .unwrap_or(0.5);

        // Well-grounded seeds get more energy; poorly-grounded get less
        let adjusted_energy = base_energy * (0.5 + 0.5 * grounding);

        self.spread(&[seed], adjusted_energy, senses, comp_index)
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CompositionRef;

    #[test]
    fn test_config_defaults() {
        let config = SpreadingActivationConfig::default();
        assert!((config.decay_factor - 0.5).abs() < 0.01);
        assert_eq!(config.max_hops, 3);
        assert!((config.min_energy - 0.01).abs() < 0.01);
    }

    #[test]
    fn test_spread_from_single_seed() {
        let sa = SpreadingActivation::new(SpreadingActivationConfig::default());

        // Create a simple sense graph: node 1 composed from [(2,0), (3,0)]
        let mut senses = HashMap::new();
        let mut sm1 = crate::sense::SenseManager::new(crate::sense::SenseConfig::default());
        let _ = sm1.ingest(vec![2, 3]);
        sm1.senses[0].compositions = vec![CompositionRef::new(2, 0), CompositionRef::new(3, 0)];
        senses.insert(1, sm1);

        let mut sm2 = crate::sense::SenseManager::new(crate::sense::SenseConfig::default());
        let _ = sm2.ingest(vec![1, 3]);
        senses.insert(2, sm2);

        let mut sm3 = crate::sense::SenseManager::new(crate::sense::SenseConfig::default());
        let _ = sm3.ingest(vec![1, 2]);
        senses.insert(3, sm3);

        // Build composition index
        let mut comp_index = CompositionIndex::new();
        comp_index.add(1, &[CompositionRef::new(2, 0), CompositionRef::new(3, 0)]);
        comp_index.add(2, &[CompositionRef::new(1, 0)]);
        comp_index.add(3, &[CompositionRef::new(1, 0)]);

        let result = sa.spread(&[1], 1.0, &senses, &comp_index);

        // Node 1 should have the highest energy (seed)
        assert!(!result.activated.is_empty());
        assert_eq!(result.activated[0].0, 1);
        assert!(result.hops_performed > 0);
    }

    #[test]
    fn test_spread_respects_max_hops() {
        let config = SpreadingActivationConfig {
            max_hops: 1,
            ..Default::default()
        };
        let sa = SpreadingActivation::new(config);

        let senses = HashMap::new();
        let comp_index = CompositionIndex::new();

        let result = sa.spread(&[1], 1.0, &senses, &comp_index);
        assert!(result.hops_performed <= 1);
    }

    #[test]
    fn test_spread_empty_seeds() {
        let sa = SpreadingActivation::new(SpreadingActivationConfig::default());
        let senses = HashMap::new();
        let comp_index = CompositionIndex::new();

        let result = sa.spread(&[], 1.0, &senses, &comp_index);
        assert!(result.activated.is_empty());
    }

    #[test]
    fn test_targeted_spread_adjusts_energy() {
        let sa = SpreadingActivation::new(SpreadingActivationConfig::default());

        // Create a well-grounded sense
        let mut senses = HashMap::new();
        let mut sm = crate::sense::SenseManager::new(crate::sense::SenseConfig::default());
        let _ = sm.ingest(vec![2, 3]);
        for _ in 0..10 {
            sm.senses[0].grounding.confirm();
        }
        senses.insert(1, sm);

        let comp_index = CompositionIndex::new();

        // Well-grounded seed should get more energy
        let result = sa.targeted_spread(1, 1.0, &senses, &comp_index);
        // The seed's energy should be > 0.5 * base (because grounding > 0.5)
        let seed_energy = result.activated.iter().find(|(id, _)| *id == 1).map(|(_, e)| *e);
        assert!(seed_energy.unwrap_or(0.0) > 0.5);
    }
}
