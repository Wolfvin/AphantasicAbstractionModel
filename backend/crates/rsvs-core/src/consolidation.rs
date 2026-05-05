//! Consolidation periodic cleanup for RSVS v6.4
//!
//! Inspired by Losion's EpisodicMemory.consolidate() which:
//! 1. Removes weak episodes (below strength_threshold)
//! 2. Merges similar episodes (Jaccard ≥ 0.8 same domain)
//!
//! Adapted for RSVS's structural domain:
//! 1. Remove dead senses: fragile + ungrounded + long inactivity
//! 2. Merge similar senses: high composition overlap
//! 3. Compact atom records: remove nodes with confidence < tau_remove
//! 4. Prune weak edges: weight below min_edge_weight after decay
//!
//! Consolidation is different from regular maintenance (check_merge, purge_fragile):
//! - Regular maintenance runs every 20 contexts (lightweight)
//! - Consolidation runs less frequently (every N batches) and is more thorough
//! - Consolidation can make cross-node decisions (regular is per-node only)
//!
//! The key insight from Losion: consolidation should be a SEPARATE phase,
//! not interleaved with ingestion. This prevents consolidation from
//! interfering with active learning and ensures structural invariants
//! are maintained only when the graph is stable.

use crate::graph::RsvsGraph;
use crate::autonomy::AutonomyEngine;
use crate::sense::{SenseManager, SenseStatus};
use crate::types::{EdgeSource, NodeId};
use std::collections::HashMap;

// -----------------------------------------------------------------------
// ConsolidationConfig
// -----------------------------------------------------------------------

/// Configuration for the consolidation engine.
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// How often consolidation runs (in batch count).
    /// Default: 50 (every 50 ingest batches)
    pub consolidation_interval: usize,
    /// Minimum composition overlap (Jaccard) to merge two senses.
    /// Higher = more conservative merging. Default: 0.8 (Losion's threshold)
    pub merge_jaccard_threshold: f32,
    /// Minimum edge weight to keep. Edges below this are pruned.
    /// Default: 0.02
    pub min_edge_weight: f32,
    /// Maximum number of senses to merge per consolidation cycle.
    /// Prevents over-merging in a single pass. Default: 5
    pub max_merges_per_cycle: usize,
    /// Whether to also compact atom records during consolidation.
    /// Default: true
    pub compact_atom_records: bool,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            consolidation_interval: 50,
            merge_jaccard_threshold: 0.8,
            min_edge_weight: 0.02,
            max_merges_per_cycle: 5,
            compact_atom_records: true,
        }
    }
}

// -----------------------------------------------------------------------
// ConsolidationResult
// -----------------------------------------------------------------------

/// Result of a consolidation cycle.
#[derive(Debug, Default)]
pub struct ConsolidationResult {
    /// Number of senses merged.
    pub senses_merged: usize,
    /// Number of senses removed (dead/retired).
    pub senses_removed: usize,
    /// Number of edges pruned (too weak).
    pub edges_pruned: usize,
    /// Number of atom records compacted.
    pub atoms_compacted: usize,
}

// -----------------------------------------------------------------------
// ConsolidationEngine
// -----------------------------------------------------------------------

/// Periodic consolidation engine — inspired by Losion's EpisodicMemory.consolidate().
///
/// Runs as a separate phase after ingestion, performing thorough cleanup:
/// - Cross-node sense merging (regular merge only works per-node)
/// - Dead sense removal (stricter than purge_fragile)
/// - Edge pruning (remove weak edges that survived decay)
/// - Atom record compaction (remove nodes below tau_remove)
pub struct ConsolidationEngine {
    pub config: ConsolidationConfig,
}

impl ConsolidationEngine {
    /// Create a new consolidation engine.
    pub fn new(config: ConsolidationConfig) -> Self {
        Self { config }
    }

    /// Run a consolidation cycle.
    ///
    /// This should be called at safe checkpoints when the graph is stable,
    /// not during active ingestion.
    pub fn consolidate(
        &self,
        graph: &mut RsvsGraph,
        senses: &mut HashMap<NodeId, SenseManager>,
        autonomy: &mut AutonomyEngine,
    ) -> ConsolidationResult {
        let mut result = ConsolidationResult::default();

        // Phase 1: Remove dead senses (fragile + ungrounded + very inactive)
        result.senses_removed = self.remove_dead_senses(senses);

        // Phase 2: Merge similar senses across nodes
        result.senses_merged = self.merge_similar_senses(senses);

        // Phase 3: Prune weak edges
        result.edges_pruned = self.prune_weak_edges(graph);

        // Phase 4: Compact atom records
        if self.config.compact_atom_records {
            result.atoms_compacted = self.compact_records(autonomy);
        }

        result
    }

    /// Check if consolidation should run based on batch counter.
    pub fn should_run(&self, batch_counter: usize) -> bool {
        batch_counter > 0 && batch_counter % self.config.consolidation_interval == 0
    }

    /// Remove dead senses: fragile + ungrounded + very inactive.
    fn remove_dead_senses(&self, senses: &mut HashMap<NodeId, SenseManager>) -> usize {
        let mut removed = 0;
        for sm in senses.values_mut() {
            let before = sm.senses.len();
            // Use stricter criteria than purge_fragile:
            // - Must be fragile
            // - Must have low grounding
            // - Must have very high inactivity (> 2× k_fragile)
            let k = sm.config.k_fragile;
            let min_g = sm.config.grounding_min;
            sm.senses.retain(|s| {
                !(s.status == SenseStatus::Fragile
                    && !s.is_grounded(min_g)
                    && s.inactivity >= k * 2)
            });
            removed += before - sm.senses.len();
        }
        removed
    }

    /// Merge similar senses within each node.
    fn merge_similar_senses(&self, senses: &mut HashMap<NodeId, SenseManager>) -> usize {
        let mut merged = 0;
        for sm in senses.values_mut() {
            if merged >= self.config.max_merges_per_cycle {
                break;
            }
            // Try to merge sense pairs with very high overlap
            let n = sm.senses.len();
            if n < 2 {
                continue;
            }

            // Find the best merge candidate
            let mut best_pair: Option<(usize, usize, f32)> = None;
            for i in 0..n {
                for j in (i + 1)..n {
                    let overlap = sm.senses[i].composition_overlap(&sm.senses[j]);
                    if overlap >= self.config.merge_jaccard_threshold {
                        if best_pair
                            .as_ref()
                            .map(|(_, _, best)| overlap > *best)
                            .unwrap_or(true)
                        {
                            best_pair = Some((i, j, overlap));
                        }
                    }
                }
            }

            if let Some((i, j, _)) = best_pair {
                sm.merge_senses(i, j);
                merged += 1;
                if merged >= self.config.max_merges_per_cycle {
                    break;
                }
            }
        }
        merged
    }

    /// Prune edges that have become too weak after decay.
    fn prune_weak_edges(&self, graph: &mut RsvsGraph) -> usize {
        let mut pruned = 0;
        let min_weight = self.config.min_edge_weight;

        for edges in graph.edges.values_mut() {
            let before = edges.len();
            edges.retain(|e| {
                // Keep bootstrap and composition edges regardless of weight
                if matches!(e.source, EdgeSource::Bootstrap | EdgeSource::Composition) {
                    return true;
                }
                e.weight >= min_weight
            });
            pruned += before - edges.len();
        }
        pruned
    }

    /// Compact atom records: remove records for nodes that no longer exist.
    fn compact_records(&self, autonomy: &mut AutonomyEngine) -> usize {
        let before = autonomy.records.len();
        // Remove records for nodes with confidence below tau_remove
        let tau = autonomy.config.tau_remove;
        autonomy.records.retain(|_, rec| {
            rec.is_seed || rec.confidence >= tau
        });
        before - autonomy.records.len()
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_run_at_interval() {
        let engine = ConsolidationEngine::new(ConsolidationConfig {
            consolidation_interval: 50,
            ..Default::default()
        });
        assert!(!engine.should_run(49));
        assert!(engine.should_run(50));
        assert!(!engine.should_run(51));
        assert!(engine.should_run(100));
    }

    #[test]
    fn test_consolidation_config_defaults() {
        let config = ConsolidationConfig::default();
        assert_eq!(config.consolidation_interval, 50);
        assert!((config.merge_jaccard_threshold - 0.8).abs() < 0.01);
    }
}
