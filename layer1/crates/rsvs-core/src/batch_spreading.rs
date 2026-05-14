//! Batch Seed Spreading — v9.0 Meaning Pathways Shared Cache
//!
//! Runs spreading activation from 7 affective/social/pragmatic seeds ONCE
//! per ingest batch, caching all energy values in a HashMap for O(1) lookup.
//! All three pathways (Gap Detection, Seed Activation, Discourse Tracking)
//! read from this cache instead of running their own spreading computations.
//!
//! Key optimization: Instead of 7×N per-node spreading runs per batch,
//! we do 7 spreading runs total per batch, then O(1) lookups.
//!
//! Pathway seed mapping:
//! - Affective: value, risk
//! - Social: trust, identity, agent
//! - Pragmatic: goal, feedback, action

use crate::composition_index::CompositionIndex;
use crate::sense::SenseManager;
use crate::spreading::SpreadingActivation;
use crate::types::{NodeId, SeedPathway};
use std::collections::{HashMap, HashSet};

/// Batch seed spreading cache — shared across all meaning pathways.
///
/// Runs spreading activation from each pathway seed once per batch,
/// then provides O(1) energy lookups via nested HashMap.
pub struct BatchSeedSpreading {
    /// Cache: seed_id → {target_node_id → energy}
    /// HashMap inside HashMap = O(1) lookup per get_energy()
    pub cache: HashMap<NodeId, HashMap<NodeId, f32>>,

    /// Seeds per pathway
    affective_seeds: Vec<NodeId>,
    social_seeds: Vec<NodeId>,
    pragmatic_seeds: Vec<NodeId>,

    /// Reuse existing SpreadingActivation instance
    spreading: SpreadingActivation,

    /// Batch counter for tracking when cache was last rebuilt
    last_batch: usize,
}

impl BatchSeedSpreading {
    /// Create a new batch seed spreading engine.
    ///
    /// # Arguments
    /// * `spreading` — Existing SpreadingActivation instance to reuse
    /// * `affective_seeds` — NodeIds for value, risk seeds
    /// * `social_seeds` — NodeIds for trust, identity, agent seeds
    /// * `pragmatic_seeds` — NodeIds for goal, feedback, action seeds
    pub fn new(
        spreading: SpreadingActivation,
        affective_seeds: Vec<NodeId>,
        social_seeds: Vec<NodeId>,
        pragmatic_seeds: Vec<NodeId>,
    ) -> Self {
        Self {
            cache: HashMap::new(),
            affective_seeds,
            social_seeds,
            pragmatic_seeds,
            spreading,
            last_batch: 0,
        }
    }

    /// Run batch spreading — ONCE per ingest batch.
    ///
    /// Clears the cache and runs targeted_spread() from each seed,
    /// converting the results to HashMap for O(1) lookup.
    pub fn run_batch(
        &mut self,
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
    ) {
        self.cache.clear();

        let all_seeds: Vec<NodeId> = self.affective_seeds.iter()
            .chain(self.social_seeds.iter())
            .chain(self.pragmatic_seeds.iter())
            .cloned()
            .collect();

        for seed_id in all_seeds {
            let result = self.spreading.targeted_spread(
                seed_id, 1.0, // base_energy = 1.0 (full energy from seed)
                senses, comp_index,
            );

            // Convert Vec<(NodeId, f32)> → HashMap<NodeId, f32> for O(1) lookup
            let energy_map: HashMap<NodeId, f32> = result.activated
                .into_iter()
                .collect();

            self.cache.insert(seed_id, energy_map);
        }
    }

    /// Lookup energy from cache — O(1) via nested HashMap.
    ///
    /// Returns 0.0 if the seed or target is not in cache.
    pub fn get_energy(&self, seed_id: NodeId, target_id: NodeId) -> f32 {
        self.cache.get(&seed_id)
            .and_then(|energy_map| energy_map.get(&target_id))
            .copied()
            .unwrap_or(0.0)
    }

    /// Lookup aggregate energy per pathway — O(S) where S = seeds in pathway.
    ///
    /// Returns the average energy across all seeds in the pathway.
    pub fn get_pathway_energy(&self, pathway: &SeedPathway, target_id: NodeId) -> f32 {
        let seeds = match pathway {
            SeedPathway::Affective => &self.affective_seeds,
            SeedPathway::Social => &self.social_seeds,
            SeedPathway::Pragmatic => &self.pragmatic_seeds,
        };

        if seeds.is_empty() {
            return 0.0;
        }

        let total: f32 = seeds.iter()
            .map(|&s| self.get_energy(s, target_id))
            .sum();

        total / seeds.len() as f32
    }

    /// Incremental update: only recompute seeds affected by graph changes.
    ///
    /// Instead of clearing all cache and re-running all 7 seeds,
    /// only recompute the seeds whose activation areas overlap with
    /// the new/modified nodes.
    ///
    /// Complexity: O(k × (V+E)) where k = affected seeds (not S=7).
    /// Average case: k ≈ 1-2 per batch.
    pub fn incremental_update(
        &mut self,
        affected_seeds: &[NodeId],
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
    ) {
        for &seed_id in affected_seeds {
            let result = self.spreading.targeted_spread(
                seed_id, 1.0, senses, comp_index,
            );
            let energy_map: HashMap<NodeId, f32> = result.activated.into_iter().collect();
            self.cache.insert(seed_id, energy_map);
        }
    }

    /// Determine which seeds are affected by a new/modified node.
    ///
    /// A seed is "affected" if the new node is a neighbor of any node
    /// in the seed's activation area. This means the new node could
    /// change the spreading pattern from that seed.
    pub fn find_affected_seeds(&self, new_node_id: NodeId, graph: &crate::graph::RsvsGraph) -> Vec<NodeId> {
        let mut affected = Vec::new();
        let all_seeds: Vec<NodeId> = self.affective_seeds.iter()
            .chain(self.social_seeds.iter())
            .chain(self.pragmatic_seeds.iter())
            .cloned()
            .collect();

        for &seed_id in &all_seeds {
            if let Some(energy_map) = self.cache.get(&seed_id) {
                // Check if new node is a neighbor of any activated node
                let is_neighbor = graph.edges_from(new_node_id).iter().any(|e| {
                    energy_map.contains_key(&e.to)
                });
                if is_neighbor {
                    affected.push(seed_id);
                }
            }
        }
        affected
    }

    /// Get all seeds across all pathways.
    pub fn all_seeds(&self) -> Vec<NodeId> {
        self.affective_seeds.iter()
            .chain(self.social_seeds.iter())
            .chain(self.pragmatic_seeds.iter())
            .cloned()
            .collect()
    }

    /// Get affective seed NodeIds.
    pub fn affective_seeds(&self) -> &[NodeId] {
        &self.affective_seeds
    }

    /// Get social seed NodeIds.
    pub fn social_seeds(&self) -> &[NodeId] {
        &self.social_seeds
    }

    /// Get pragmatic seed NodeIds.
    pub fn pragmatic_seeds(&self) -> &[NodeId] {
        &self.pragmatic_seeds
    }

    /// Check if the cache is empty (not yet run).
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get the number of seeds in cache.
    pub fn cached_seed_count(&self) -> usize {
        self.cache.len()
    }

    /// Update the last batch counter.
    pub fn set_last_batch(&mut self, batch: usize) {
        self.last_batch = batch;
    }

    /// Get the last batch counter.
    pub fn last_batch(&self) -> usize {
        self.last_batch
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spreading::SpreadingActivationConfig;

    #[test]
    fn test_batch_spreading_cache_lookup() {
        let spreading = SpreadingActivation::new(SpreadingActivationConfig::default());
        let batch = BatchSeedSpreading::new(
            spreading,
            vec![1, 2],   // affective: value, risk
            vec![3, 4, 5], // social: trust, identity, agent
            vec![6, 7, 8], // pragmatic: goal, feedback, action
        );

        // Before run_batch, cache should be empty
        assert!(batch.is_empty());
        assert_eq!(batch.get_energy(1, 100), 0.0);
    }

    #[test]
    fn test_pathway_energy_averaging() {
        let spreading = SpreadingActivation::new(SpreadingActivationConfig::default());
        let mut batch = BatchSeedSpreading::new(
            spreading,
            vec![1, 2],
            vec![3],
            vec![],
        );

        // Manually populate cache for testing
        batch.cache.insert(1, {
            let mut m = HashMap::new();
            m.insert(100, 0.5);
            m
        });
        batch.cache.insert(2, {
            let mut m = HashMap::new();
            m.insert(100, 0.3);
            m
        });
        batch.cache.insert(3, {
            let mut m = HashMap::new();
            m.insert(100, 0.7);
            m
        });

        // Affective pathway = average of seed 1 and 2
        let affective = batch.get_pathway_energy(&SeedPathway::Affective, 100);
        assert!((affective - 0.4).abs() < 0.01);

        // Social pathway = seed 3 only
        let social = batch.get_pathway_energy(&SeedPathway::Social, 100);
        assert!((social - 0.7).abs() < 0.01);

        // Pragmatic = empty → 0.0
        let pragmatic = batch.get_pathway_energy(&SeedPathway::Pragmatic, 100);
        assert!((pragmatic - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_all_seeds() {
        let spreading = SpreadingActivation::new(SpreadingActivationConfig::default());
        let batch = BatchSeedSpreading::new(
            spreading,
            vec![1, 2],
            vec![3, 4, 5],
            vec![6, 7, 8],
        );

        let all = batch.all_seeds();
        assert_eq!(all.len(), 8);
    }
}
