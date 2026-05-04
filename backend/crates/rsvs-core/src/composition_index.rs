//! CompositionIndex for O(1) reverse lookup — inspired by Losion's Engram hash-based retrieval
//!
//! Problem: Finding which senses reference a given (NodeId, SenseId) pair currently
//! requires scanning ALL senses in ALL nodes — O(N×M) where N = nodes, M = senses/node.
//! This is the bottleneck in `count_impact()`, `cross_activation_rescore()`, and
//! composition validation.
//!
//! Solution: A reverse index that maps (NodeId, SenseId) → set of NodeIds that
//! reference it in their compositions. This enables O(1) lookup for:
//! - Impact counting (how many senses depend on this node?)
//! - Reverse traversal (which nodes use this sense?)
//! - Cascade detection (if I remove this, what breaks?)
//!
//! Inspired by Losion's EngramMemory which uses hash-based O(1) retrieval
//! instead of linear scanning. The key insight: a hash table for reverse
//! lookups is trivially parallelizable and cache-friendly.

use crate::types::{NodeId, CompositionRef};
use std::collections::{HashMap, HashSet};

// -----------------------------------------------------------------------
// CompositionIndex — reverse lookup from CompositionRef → referencing nodes
// -----------------------------------------------------------------------

/// Reverse index from CompositionRef to the set of NodeIds whose senses
/// reference that composition. Enables O(1) reverse lookups.
///
/// This replaces the O(N×M) scan in `count_impact()` with a single HashMap lookup.
///
/// # Example
///
/// ```ignore
/// let mut index = CompositionIndex::new();
/// // Node 5 has a sense with compositions [(1,0), (2,0)]
/// index.add(5, &vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)]);
/// // Now we can ask: which nodes reference node 1?
/// let refs = index.dependents_of_node(1); // Returns {5}
/// ```
#[derive(Debug, Clone, Default)]
pub struct CompositionIndex {
    /// Maps (NodeId, SenseId) → set of NodeIds whose compositions reference this pair.
    /// This is the core reverse index for O(1) lookups.
    ref_to_dependents: HashMap<CompositionRef, HashSet<NodeId>>,

    /// Maps NodeId → set of NodeIds it depends on (forward index).
    /// Used for cascade detection: if I remove node X, what forward edges break?
    node_to_dependencies: HashMap<NodeId, HashSet<NodeId>>,

    /// v6.5: Secondary index for O(1) node-level lookup.
    /// Maps NodeId → set of NodeIds whose senses reference ANY sense of this node.
    /// Fixes the O(K) scan in dependents_of_node() — now true O(1).
    node_to_dependents: HashMap<NodeId, HashSet<NodeId>>,
}

impl CompositionIndex {
    /// Create a new empty composition index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Index a node's compositions. After this call, all CompositionRefs
    /// in the given compositions will map back to this node.
    ///
    /// # Arguments
    /// * `node_id` — the node whose compositions are being indexed
    /// * `compositions` — the list of CompositionRefs defining this node's sense
    pub fn add(&mut self, node_id: NodeId, compositions: &[CompositionRef]) {
        for comp in compositions {
            self.ref_to_dependents
                .entry(comp.clone())
                .or_default()
                .insert(node_id);
            self.node_to_dependencies
                .entry(node_id)
                .or_default()
                .insert(comp.node_id);
            // v6.5: Maintain secondary node-level index
            self.node_to_dependents
                .entry(comp.node_id)
                .or_default()
                .insert(node_id);
        }
    }

    /// Remove a node's compositions from the index.
    /// Call this when a sense is deleted or revised.
    pub fn remove(&mut self, node_id: NodeId, compositions: &[CompositionRef]) {
        for comp in compositions {
            if let Some(dependents) = self.ref_to_dependents.get_mut(comp) {
                dependents.remove(&node_id);
                if dependents.is_empty() {
                    self.ref_to_dependents.remove(comp);
                }
            }
            // v6.5: Update secondary node-level index
            if let Some(node_deps) = self.node_to_dependents.get_mut(&comp.node_id) {
                node_deps.remove(&node_id);
                if node_deps.is_empty() {
                    self.node_to_dependents.remove(&comp.node_id);
                }
            }
        }
        self.node_to_dependencies.remove(&node_id);
        self.node_to_dependents.remove(&node_id);
    }

    /// O(1) lookup: which nodes have senses that reference a specific (node_id, sense_id)?
    /// Returns a Vec of dependent NodeIds.
    pub fn dependents_of(&self, comp: &CompositionRef) -> Vec<NodeId> {
        self.ref_to_dependents
            .get(comp)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// O(1) lookup: which nodes have senses that reference ANY sense of node_id?
    /// Returns the union of all dependents across all senses of the given node.
    /// v6.5: Now uses secondary index for true O(1) lookup instead of O(K) scan.
    pub fn dependents_of_node(&self, node_id: NodeId) -> HashSet<NodeId> {
        self.node_to_dependents
            .get(&node_id)
            .cloned()
            .unwrap_or_default()
    }

    /// O(1) impact count: how many sense compositions reference this node?
    /// This replaces the O(N×M) `count_impact()` scan.
    pub fn impact_count(&self, node_id: NodeId) -> usize {
        self.dependents_of_node(node_id).len()
    }

    /// O(1) lookup: what nodes does this node directly depend on?
    pub fn dependencies_of(&self, node_id: NodeId) -> HashSet<NodeId> {
        self.node_to_dependencies
            .get(&node_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Rebuild the entire index from scratch from the given sense managers.
    /// Call this after bulk operations that may have invalidated the index.
    pub fn rebuild(&mut self, all_senses: &HashMap<NodeId, crate::sense::SenseManager>) {
        self.ref_to_dependents.clear();
        self.node_to_dependencies.clear();
        self.node_to_dependents.clear();
        for (&node_id, sm) in all_senses {
            for sense in &sm.senses {
                self.add(node_id, &sense.compositions);
            }
        }
    }

    /// Return the total number of reverse index entries.
    pub fn len(&self) -> usize {
        self.ref_to_dependents.len()
    }

    /// Return whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.ref_to_dependents.is_empty()
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_lookup() {
        let mut idx = CompositionIndex::new();
        idx.add(5, &[CompositionRef::new(1, 0), CompositionRef::new(2, 0)]);
        idx.add(6, &[CompositionRef::new(1, 0), CompositionRef::new(3, 0)]);

        let deps = idx.dependents_of_node(1);
        assert!(deps.contains(&5));
        assert!(deps.contains(&6));
        assert_eq!(deps.len(), 2);
    }

    #[test]
    fn test_impact_count() {
        let mut idx = CompositionIndex::new();
        idx.add(5, &[CompositionRef::new(1, 0)]);
        idx.add(6, &[CompositionRef::new(1, 0)]);
        idx.add(7, &[CompositionRef::new(2, 0)]);

        assert_eq!(idx.impact_count(1), 2); // Referenced by nodes 5 and 6
        assert_eq!(idx.impact_count(2), 1); // Referenced by node 7
        assert_eq!(idx.impact_count(3), 0); // Not referenced
    }

    #[test]
    fn test_remove() {
        let mut idx = CompositionIndex::new();
        idx.add(5, &[CompositionRef::new(1, 0)]);
        idx.remove(5, &[CompositionRef::new(1, 0)]);

        assert_eq!(idx.impact_count(1), 0);
    }

    #[test]
    fn test_dependencies_of() {
        let mut idx = CompositionIndex::new();
        idx.add(5, &[CompositionRef::new(1, 0), CompositionRef::new(2, 0)]);

        let deps = idx.dependencies_of(5);
        assert!(deps.contains(&1));
        assert!(deps.contains(&2));
    }

    #[test]
    fn test_rebuild() {
        let mut idx = CompositionIndex::new();
        let mut sm_map: HashMap<NodeId, crate::sense::SenseManager> = HashMap::new();
        let config = crate::sense::SenseConfig::default();

        let mut sm = crate::sense::SenseManager::new(config);
        let _ = sm.ingest(vec![1, 2, 3]);
        sm_map.insert(10, sm);

        idx.rebuild(&sm_map);
        // After rebuild, the index should reflect the sense managers
        // (but sense 0 of node 10 has no compositions, so no entries)
        assert_eq!(idx.len(), 0);
    }
}
