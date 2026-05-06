//! Convergence Detection — v8.0 PHASE 2
//!
//! Detects when two nodes from different surface forms (possibly different
//! languages) have structurally equivalent compositions, indicating they
//! likely represent the same concept.
//!
//! The key insight: the system does NOT need to know that "anjing" is
//! Indonesian and "dog" is English. It only needs to observe that their
//! sense compositions are structurally similar (high overlap of
//! CompositionRefs) AND they never co-occur in the same text (suggesting
//! they're different surface forms for the same underlying concept).
//!
//! When convergence is detected, a `LanguageLink` is created automatically
//! between the two nodes with type "structural_equivalence" and a score
//! representing the overlap ratio.

use crate::graph::RsvsGraph;
use crate::sense::SenseManager;
use crate::types::{LanguageLink, NodeId};
use std::collections::{HashMap, HashSet};

// -----------------------------------------------------------------------
// ConvergenceConfig — tunable parameters
// -----------------------------------------------------------------------

/// Configuration for the convergence detection engine.
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Minimum composition overlap ratio to consider two nodes equivalent.
    /// If the Jaccard similarity of their best sense pair exceeds this,
    /// they are candidates for convergence. Default: 0.6
    pub min_overlap_threshold: f32,

    /// Minimum number of contexts each node must have before it's eligible
    /// for convergence detection. Prevents premature linking of fragile nodes.
    /// Default: 3
    pub min_context_count: usize,

    /// Maximum co-occurrence count between two nodes for them to be considered
    /// "never co-occurring". If they co-occur more than this, they're likely
    /// different concepts that just happen to be similar. Default: 1
    pub max_cooc_for_equivalence: usize,

    /// Whether to automatically create LanguageLinks when convergence is detected.
    /// If false, convergence is only reported but not persisted. Default: true
    pub auto_link: bool,

    /// Minimum confidence both nodes must have for convergence. Default: 0.3
    pub min_confidence: f32,

    /// v8.1: Maximum number of candidate pairs to evaluate per `detect()` call.
    /// Prevents O(N²) blowup on large graphs. When the number of eligible
    /// pairs exceeds this limit, a stratified sample is taken (prioritizing
    /// higher-confidence nodes). Default: 500
    pub max_pairs_per_run: usize,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            min_overlap_threshold: 0.6,
            min_context_count: 3,
            max_cooc_for_equivalence: 1,
            auto_link: true,
            min_confidence: 0.3,
            max_pairs_per_run: 500,
        }
    }
}

// -----------------------------------------------------------------------
// ConvergenceResult — what was detected
// -----------------------------------------------------------------------

/// A detected convergence between two nodes.
#[derive(Debug, Clone)]
pub struct ConvergencePair {
    /// First node ID.
    pub node_a: NodeId,
    /// Second node ID.
    pub node_b: NodeId,
    /// Structural overlap ratio (Jaccard of best sense pair compositions).
    pub overlap_score: f32,
    /// Index of the best-matching sense in node A.
    pub sense_idx_a: usize,
    /// Index of the best-matching sense in node B.
    pub sense_idx_b: usize,
    /// Whether a LanguageLink was created.
    pub linked: bool,
}

// -----------------------------------------------------------------------
// ConvergenceEngine — main detection engine
// -----------------------------------------------------------------------

/// Engine for detecting structural convergence between nodes.
///
/// This is the core mechanism that makes RSVS truly language-agnostic:
/// "anjing" and "dog" will naturally converge to the same seed NodeIds
/// through their compositions, and this engine detects that convergence
/// and creates LanguageLinks to record the equivalence.
///
/// Should be called periodically (e.g., during consolidation) to detect
/// new convergence pairs as the graph grows.
pub struct ConvergenceEngine {
    /// Configuration for convergence detection.
    pub config: ConvergenceConfig,
    /// Set of already-detected convergence pairs to avoid duplicate work.
    /// Stored as (min_id, max_id) for canonical ordering.
    detected_pairs: HashSet<(NodeId, NodeId)>,
}

impl ConvergenceEngine {
    /// Create a new convergence engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(ConvergenceConfig::default())
    }

    /// Create a new convergence engine with custom configuration.
    pub fn with_config(config: ConvergenceConfig) -> Self {
        Self {
            config,
            detected_pairs: HashSet::new(),
        }
    }

    /// Run convergence detection over all eligible node pairs.
    ///
    /// This is an O(N²) operation in the worst case, but in practice
    /// most node pairs are filtered out early by the eligibility criteria.
    /// For large graphs, consider running this only during consolidation
    /// (every N batches) rather than on every ingest.
    ///
    /// Returns a list of detected convergence pairs.
    pub fn detect(
        &mut self,
        graph: &mut RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        cooc_stats: &crate::attention::CoocStats,
    ) -> Vec<ConvergencePair> {
        let mut results = Vec::new();

        // Collect eligible nodes: non-seed, sufficient contexts, sufficient confidence
        let eligible: Vec<NodeId> = graph
            .nodes
            .values()
            .filter(|n| !n.is_seed)
            .filter(|n| n.confidence >= self.config.min_confidence)
            .filter(|n| {
                senses
                    .get(&n.id)
                    .map(|sm| {
                        sm.senses
                            .iter()
                            .any(|s| s.context_count() >= self.config.min_context_count)
                    })
                    .unwrap_or(false)
            })
            .map(|n| n.id)
            .collect();

        // v8.1: Sort eligible nodes by confidence (descending) so that when
        // the throttle kicks in, we prioritize pairs involving high-confidence
        // nodes. This is a simple but effective heuristic — convergent pairs
        // involving low-confidence nodes are less reliable anyway.
        let mut eligible = eligible;
        eligible.sort_by(|&a, &b| {
            let conf_a = graph.get_node(a).map(|n| n.confidence).unwrap_or(0.0);
            let conf_b = graph.get_node(b).map(|n| n.confidence).unwrap_or(0.0);
            conf_b.partial_cmp(&conf_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Compare eligible pairs — v8.1: throttled to prevent O(N²) blowup
        let _total_pairs = eligible.len() * (eligible.len().saturating_sub(1)) / 2;
        let mut pairs_checked = 0usize;

        for i in 0..eligible.len() {
            if pairs_checked >= self.config.max_pairs_per_run {
                break;
            }
            for j in (i + 1)..eligible.len() {
                // v8.1: Throttle — stop evaluating pairs once we hit the budget
                if pairs_checked >= self.config.max_pairs_per_run {
                    break;
                }
                pairs_checked += 1;

                let a = eligible[i];
                let b = eligible[j];

                // Skip already-detected pairs
                let pair_key = if a < b { (a, b) } else { (b, a) };
                if self.detected_pairs.contains(&pair_key) {
                    continue;
                }

                // Check if they already have a language link to each other
                if self.has_existing_link(graph, a, b) {
                    self.detected_pairs.insert(pair_key);
                    continue;
                }

                // Check co-occurrence: if they co-occur too often, they're
                // likely different concepts, not translations
                let label_a = graph.get_node(a).map(|n| n.label.clone()).unwrap_or_default();
                let label_b = graph.get_node(b).map(|n| n.label.clone()).unwrap_or_default();
                let cooc = cooc_stats.pair_cooc_count(&label_a, &label_b);
                if cooc > self.config.max_cooc_for_equivalence {
                    continue; // They co-occur — likely different concepts
                }

                // Compute structural overlap
                let sm_a = match senses.get(&a) {
                    Some(sm) => sm,
                    None => continue,
                };
                let sm_b = match senses.get(&b) {
                    Some(sm) => sm,
                    None => continue,
                };

                let (overlap, idx_a, idx_b) =
                    best_sense_overlap(sm_a, sm_b);

                if overlap >= self.config.min_overlap_threshold {
                    // Convergence detected!
                    let linked = if self.config.auto_link {
                        self.create_language_link(graph, a, b, overlap)
                    } else {
                        false
                    };

                    self.detected_pairs.insert(pair_key);

                    results.push(ConvergencePair {
                        node_a: a,
                        node_b: b,
                        overlap_score: overlap,
                        sense_idx_a: idx_a,
                        sense_idx_b: idx_b,
                        linked,
                    });
                }
            }
        }

        results
    }

    /// Check if two nodes already have a LanguageLink between them.
    fn has_existing_link(&self, graph: &RsvsGraph, a: NodeId, b: NodeId) -> bool {
        if let Some(node_a) = graph.get_node(a) {
            for link in &node_a.language_links {
                if link.target_id == b {
                    return true;
                }
            }
        }
        if let Some(node_b) = graph.get_node(b) {
            for link in &node_b.language_links {
                if link.target_id == a {
                    return true;
                }
            }
        }
        false
    }

    /// Create bidirectional LanguageLinks between two converged nodes.
    fn create_language_link(
        &self,
        graph: &mut RsvsGraph,
        a: NodeId,
        b: NodeId,
        _score: f32,
    ) -> bool {
        let link_a = LanguageLink {
            link_type: "structural_equivalence".to_string(),
            target_id: b,
        };
        let link_b = LanguageLink {
            link_type: "structural_equivalence".to_string(),
            target_id: a,
        };

        let mut success = false;
        if let Some(node) = graph.get_node_mut(a) {
            // Only add if not already present
            if !node.language_links.iter().any(|l| l.target_id == b) {
                node.language_links.push(link_a);
                success = true;
            }
        }
        if let Some(node) = graph.get_node_mut(b) {
            if !node.language_links.iter().any(|l| l.target_id == a) {
                node.language_links.push(link_b);
                success = true;
            }
        }
        success
    }

    /// Clear the set of detected pairs (useful after a full re-index).
    pub fn reset(&mut self) {
        self.detected_pairs.clear();
    }

    /// Return the number of detected convergence pairs.
    pub fn detected_count(&self) -> usize {
        self.detected_pairs.len()
    }
}

// -----------------------------------------------------------------------
// Helper: Best sense overlap between two SenseManagers
// -----------------------------------------------------------------------

/// Find the pair of senses (one from each manager) with the highest
/// composition overlap (Jaccard similarity). Returns (score, idx_a, idx_b).
fn best_sense_overlap(
    sm_a: &SenseManager,
    sm_b: &SenseManager,
) -> (f32, usize, usize) {
    let mut best_score = 0.0f32;
    let mut best_idx_a = 0usize;
    let mut best_idx_b = 0usize;

    for (idx_a, sense_a) in sm_a.senses.iter().enumerate() {
        if !sense_a.is_compositional() {
            continue;
        }
        let set_a: HashSet<_> = sense_a.compositions.iter().collect();

        for (idx_b, sense_b) in sm_b.senses.iter().enumerate() {
            if !sense_b.is_compositional() {
                continue;
            }
            let set_b: HashSet<_> = sense_b.compositions.iter().collect();

            let intersection = set_a.intersection(&set_b).count();
            let union = set_a.union(&set_b).count();

            let score = if union == 0 {
                0.0
            } else {
                intersection as f32 / union as f32
            };

            if score > best_score {
                best_score = score;
                best_idx_a = idx_a;
                best_idx_b = idx_b;
            }
        }
    }

    (best_score, best_idx_a, best_idx_b)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sense::{Sense, SenseConfig, SenseStatus};
    use crate::types::{CompositionRef, CompressionState, Node, NodeStatus, SemanticMeta, Tier};

    #[test]
    fn test_convergence_detection_basic() {
        let mut graph = RsvsGraph::new();

        // First create seed nodes (IDs 1 and 2) so derived_from_node_ids are valid
        graph.insert_node(Node {
            id: 0, label: "exists".to_string(), surface_label: "exists".to_string(),
            is_seed: true, tier: Tier::Tier1, confidence: 1.0, status: NodeStatus::Stable,
            is_locked: true, semantic: SemanticMeta { internal_representation: false, ..SemanticMeta::default() },
            ..Node::default()
        }).unwrap();
        graph.insert_node(Node {
            id: 0, label: "entity".to_string(), surface_label: "entity".to_string(),
            is_seed: true, tier: Tier::Tier1, confidence: 1.0, status: NodeStatus::Stable,
            is_locked: true, semantic: SemanticMeta { internal_representation: false, ..SemanticMeta::default() },
            ..Node::default()
        }).unwrap();

        // Create two nodes: "dog" and "anjing" with identical compositions
        // pointing to seed nodes (exists=1, entity=2)
        let dog_id = graph
            .insert_node(Node {
                id: 0,
                label: "dog".to_string(),
                surface_label: "dog".to_string(),
                confidence: 0.7,
                is_seed: false,
                semantic: SemanticMeta {
                    compression_state: CompressionState::Compressed,
                    layer: 1,
                    derived_from_node_ids: vec![1, 2],
                    compression_reason: Some("test".to_string()),
                    internal_representation: true,
                },
                ..Node::default()
            })
            .unwrap();

        let anjing_id = graph
            .insert_node(Node {
                id: 0,
                label: "anjing".to_string(),
                surface_label: "anjing".to_string(),
                confidence: 0.7,
                is_seed: false,
                semantic: SemanticMeta {
                    compression_state: CompressionState::Compressed,
                    layer: 1,
                    derived_from_node_ids: vec![1, 2],
                    compression_reason: Some("test".to_string()),
                    internal_representation: true,
                },
                ..Node::default()
            })
            .unwrap();

        // Build sense managers with matching compositions
        let config = SenseConfig::default();
        let mut sm_dog = SenseManager::new(config.clone());
        let mut dog_sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2],
            1,
        );
        dog_sense.status = SenseStatus::Mature;
        dog_sense.contexts.push(vec![1, 2]); // Add enough contexts
        dog_sense.contexts.push(vec![1, 2]);
        sm_dog.senses.push(dog_sense);

        let mut sm_anjing = SenseManager::new(config);
        let mut anjing_sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2],
            1,
        );
        anjing_sense.status = SenseStatus::Mature;
        anjing_sense.contexts.push(vec![1, 2]);
        anjing_sense.contexts.push(vec![1, 2]);
        sm_anjing.senses.push(anjing_sense);

        let mut senses = HashMap::new();
        senses.insert(dog_id, sm_dog);
        senses.insert(anjing_id, sm_anjing);

        let cooc_stats = crate::attention::CoocStats::new();

        let mut engine = ConvergenceEngine::new();
        let results = engine.detect(&mut graph, &senses, &cooc_stats);

        assert_eq!(results.len(), 1);
        assert!(results[0].overlap_score >= 0.99); // Perfect overlap
        assert!(results[0].linked);

        // Check that LanguageLinks were created
        let dog_node = graph.get_node(dog_id).unwrap();
        assert!(dog_node
            .language_links
            .iter()
            .any(|l| l.target_id == anjing_id && l.link_type == "structural_equivalence"));
    }

    #[test]
    fn test_no_convergence_for_low_overlap() {
        let mut graph = RsvsGraph::new();

        // Create seed nodes
        graph.insert_node(Node {
            id: 0, label: "exists".to_string(), surface_label: "exists".to_string(),
            is_seed: true, tier: Tier::Tier1, confidence: 1.0, status: NodeStatus::Stable,
            is_locked: true, semantic: SemanticMeta { internal_representation: false, ..SemanticMeta::default() },
            ..Node::default()
        }).unwrap();
        graph.insert_node(Node {
            id: 0, label: "entity".to_string(), surface_label: "entity".to_string(),
            is_seed: true, tier: Tier::Tier1, confidence: 1.0, status: NodeStatus::Stable,
            is_locked: true, semantic: SemanticMeta { internal_representation: false, ..SemanticMeta::default() },
            ..Node::default()
        }).unwrap();
        graph.insert_node(Node {
            id: 0, label: "relation".to_string(), surface_label: "relation".to_string(),
            is_seed: true, tier: Tier::Tier1, confidence: 1.0, status: NodeStatus::Stable,
            is_locked: true, semantic: SemanticMeta { internal_representation: false, ..SemanticMeta::default() },
            ..Node::default()
        }).unwrap();

        let dog_id = graph
            .insert_node(Node {
                id: 0,
                label: "dog".to_string(),
                surface_label: "dog".to_string(),
                confidence: 0.7,
                is_seed: false,
                semantic: SemanticMeta {
                    compression_state: CompressionState::Compressed,
                    layer: 1,
                    derived_from_node_ids: vec![1, 2],
                    compression_reason: Some("test".to_string()),
                    internal_representation: false,
                },
                ..Node::default()
            })
            .unwrap();

        let cat_id = graph
            .insert_node(Node {
                id: 0,
                label: "cat".to_string(),
                surface_label: "cat".to_string(),
                confidence: 0.7,
                is_seed: false,
                semantic: SemanticMeta {
                    compression_state: CompressionState::Compressed,
                    layer: 1,
                    derived_from_node_ids: vec![1, 3], // Different from dog
                    compression_reason: Some("test".to_string()),
                    internal_representation: false,
                },
                ..Node::default()
            })
            .unwrap();

        let config = SenseConfig::default();
        let mut sm_dog = SenseManager::new(config.clone());
        let mut dog_sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2],
            1,
        );
        dog_sense.status = SenseStatus::Mature;
        dog_sense.contexts.push(vec![1, 2]);
        dog_sense.contexts.push(vec![1, 2]);
        sm_dog.senses.push(dog_sense);

        let mut sm_cat = SenseManager::new(config);
        let mut cat_sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(3, 0)],
            vec![1, 3],
            1,
        );
        cat_sense.status = SenseStatus::Mature;
        cat_sense.contexts.push(vec![1, 3]);
        cat_sense.contexts.push(vec![1, 3]);
        sm_cat.senses.push(cat_sense);

        let mut senses = HashMap::new();
        senses.insert(dog_id, sm_dog);
        senses.insert(cat_id, sm_cat);

        let cooc_stats = crate::attention::CoocStats::new();

        let mut engine = ConvergenceEngine::new();
        let results = engine.detect(&mut graph, &senses, &cooc_stats);

        // Overlap is 1/3 ≈ 0.33, below threshold of 0.6
        assert_eq!(results.len(), 0);
    }
}
