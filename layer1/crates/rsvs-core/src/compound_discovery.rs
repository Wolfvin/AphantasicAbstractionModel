//! Compound Discovery Engine — v10.1 Multi-Word Expression Detection
//!
//! Core insight: When two tokens ALWAYS co-occur with near-perfect NPMI,
//! they form a COMPOUND — a single meaning unit.
//!
//! Example:
//!   "harga" and "diri" always appear together → NPMI ≈ 1.0
//!   → They are NOT two separate meanings
//!   → They form ONE meaning: "harga_diri" (dignity/self-esteem)
//!
//! This engine discovers compounds FROM the graph's own co-occurrence
//! statistics. No dictionary needed — the evidence is structural:
//!
//! 1. NPMI(harga, diri) ≈ 1.0 — they always appear together
//! 2. Both are non-seed nodes with compositional senses
//! 3. Their joint composition creates a richer meaning than either alone
//!
//! The compound node gets:
//!   - label = "harga_diri" (underscore-joined)
//!   - compositions = union of both component senses' compositions
//!   - layer = max(layer_a, layer_b) + 1
//!   - EdgeSource::CompoundDiscovery edges from components

use crate::attention::CoocStats;
use crate::graph::RsvsGraph;
use crate::sense::SenseManager;
use crate::types::{
    CompositionRef, Edge, EdgeSource, Node, NodeId, NodeStatus, RelationType, SemanticMeta,
    SenseId, Tier, CompressionState,
};
use std::collections::{HashMap, HashSet};

/// Configuration for the Compound Discovery Engine.
#[derive(Debug, Clone)]
pub struct CompoundDiscoveryConfig {
    /// Minimum NPMI for two tokens to be considered a compound pair.
    /// NPMI ranges from -1.0 to 1.0. Near 1.0 = always co-occur.
    /// Default: 0.3 (lower than before because NPMI fails when P=1.0)
    pub min_npmi: f32,

    /// Minimum bidirectional conditional probability for compound detection.
    /// P(B|A) >= min_conditional AND P(A|B) >= min_conditional.
    /// This is the PRIMARY compound signal: "whenever A appears, B always appears too."
    /// Default: 0.8 (strong — they almost always co-occur)
    pub min_conditional: f32,

    /// Minimum raw co-occurrence count for a pair to be considered.
    /// Prevents spurious compounds from rare coincidences.
    /// Default: 3
    pub min_cooc_count: usize,

    /// Maximum compound length (number of component tokens).
    /// Currently only supports 2-word compounds.
    /// Default: 2
    pub max_compound_length: usize,

    /// Maximum compounds to discover per batch.
    /// Default: 10
    pub max_compounds_per_batch: usize,
}

impl Default for CompoundDiscoveryConfig {
    fn default() -> Self {
        Self {
            min_npmi: 0.3,
            min_conditional: 0.8,
            min_cooc_count: 3,
            max_compound_length: 2,
            max_compounds_per_batch: 10,
        }
    }
}

/// A discovered compound — two tokens that form one meaning unit.
#[derive(Debug, Clone)]
pub struct DiscoveredCompound {
    /// The compound label (e.g., "harga_diri")
    pub label: String,
    /// First component token
    pub component_a: String,
    /// Second component token
    pub component_b: String,
    /// NPMI between the two components
    pub npmi: f32,
    /// Raw co-occurrence count
    pub cooc_count: usize,
    /// NodeId of the first component (if promoted)
    pub node_id_a: Option<NodeId>,
    /// NodeId of the second component (if promoted)
    pub node_id_b: Option<NodeId>,
    /// NodeId of the newly created compound node
    pub compound_node_id: Option<NodeId>,
    /// Combined compositions from both components
    pub combined_compositions: Vec<CompositionRef>,
}

/// The Compound Discovery Engine.
///
/// Scans co-occurrence statistics for token pairs with near-perfect NPMI,
/// then creates compound nodes in the graph when such pairs are found.
pub struct CompoundDiscoveryEngine {
    /// Configuration.
    pub config: CompoundDiscoveryConfig,
    /// Previously discovered compounds (to avoid re-discovery)
    pub discovered: HashSet<String>,
}

impl CompoundDiscoveryEngine {
    /// Create a new compound discovery engine.
    pub fn new(config: CompoundDiscoveryConfig) -> Self {
        Self {
            config,
            discovered: HashSet::new(),
        }
    }

    /// Scan co-occurrence statistics for candidate compound pairs.
    ///
    /// Uses TWO signals:
    /// 1. **Conditional probability** (PRIMARY): P(B|A) and P(A|B) both high
    ///    → "whenever one appears, the other always appears too"
    /// 2. **NPMI** (SECONDARY): Statistical surprise of co-occurrence
    ///    → Fails when P(t)=1.0 (token appears in every sentence)
    ///
    /// A pair qualifies when:
    /// - P(B|A) >= min_conditional AND P(A|B) >= min_conditional
    /// - NPMI >= min_npmi (or NPMI ≈ 0 because P=1.0 for both)
    /// - cooc_count >= min_cooc_count
    ///
    /// Returns pairs sorted by compound_score = min(P(B|A), P(A|B)).
    pub fn scan_candidates(
        &self,
        stats: &CoocStats,
        token_to_id: &HashMap<String, NodeId>,
        seed_node_ids: &std::collections::HashSet<NodeId>,
    ) -> Vec<(String, String, f32, usize)> {
        let mut candidates = Vec::new();

        // Iterate over all pairs in the co-occurrence stats
        for ((token_a, token_b), count) in &stats.pair_count {
            if *count < self.config.min_cooc_count {
                continue;
            }

            // Both tokens must be promoted nodes
            let id_a = token_to_id.get(token_a).copied();
            let id_b = token_to_id.get(token_b).copied();
            if id_a.is_none() || id_b.is_none() {
                continue;
            }

            // CRITICAL: Skip pairs where either token is a seed node.
            // Seeds are epistemological primitives — they can't form compounds.
            // A compound is TWO NON-SEED tokens that form ONE meaning (e.g., harga+diri).
            if seed_node_ids.contains(&id_a.unwrap()) || seed_node_ids.contains(&id_b.unwrap()) {
                continue;
            }

            // Skip if already discovered as compound
            let compound_label = format!("{}_{}", token_a.min(token_b), token_a.max(token_b));
            if self.discovered.contains(&compound_label) {
                continue;
            }

            // Compute bidirectional conditional probability:
            // P(B|A) = cooc(A,B) / count(A) — how often B appears when A is present
            // P(A|B) = cooc(A,B) / count(B) — how often A appears when B is present
            let count_a = stats.token_count.get(token_a).copied().unwrap_or(0);
            let count_b = stats.token_count.get(token_b).copied().unwrap_or(0);

            if count_a == 0 || count_b == 0 {
                continue;
            }

            let p_b_given_a = *count as f32 / count_a as f32;
            let p_a_given_b = *count as f32 / count_b as f32;

            // PRIMARY signal: bidirectional conditional probability
            let min_conditional = p_b_given_a.min(p_a_given_b);

            if min_conditional < self.config.min_conditional {
                continue; // One of them appears without the other too often
            }

            // SECONDARY signal: NPMI
            // NPMI fails when P(t)=1.0 for both tokens (they appear in every sentence)
            // In that case, P(t,c) = P(t) = P(c) = 1.0 → PMI=0 → NPMI=0/0→0
            // This is actually a STRONG compound signal! If both always appear,
            // they're effectively one lexical unit.
            let npmi = stats.npmi(token_a, token_b);
            let both_omnipresent = count_a as f32 / stats.total_sentences as f32 > 0.9
                && count_b as f32 / stats.total_sentences as f32 > 0.9;

            if !both_omnipresent && npmi < self.config.min_npmi {
                continue; // Not statistically surprising and not omnipresent
            }

            // Compound score = min conditional probability (higher = stronger compound)
            let compound_score = min_conditional;

            candidates.push((token_a.clone(), token_b.clone(), compound_score, *count));
        }

        // Sort by compound score descending
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.max_compounds_per_batch);
        candidates
    }

    /// Create a compound node from two component nodes.
    ///
    /// The compound's sense is the UNION of both components' compositions,
    /// which is the key to understanding why they form one meaning:
    /// sense(harga) = [value] + sense(diri) = [identity]
    /// → sense(harga_diri) = [value, identity] = "dignity"
    pub fn create_compound_node(
        &mut self,
        label_a: &str,
        label_b: &str,
        node_id_a: NodeId,
        node_id_b: NodeId,
        graph: &mut RsvsGraph,
        senses: &mut HashMap<NodeId, SenseManager>,
    ) -> Option<DiscoveredCompound> {
        let compound_label = format!("{}_{}", label_a, label_b);

        // Skip if compound already exists in graph
        if graph.id_for_label(&compound_label).is_some() {
            return None;
        }

        // Collect compositions from both component nodes
        let comps_a: Vec<CompositionRef> = senses
            .get(&node_id_a)
            .map(|sm| sm.senses.iter().flat_map(|s| s.compositions.clone()).collect())
            .unwrap_or_default();

        let comps_b: Vec<CompositionRef> = senses
            .get(&node_id_b)
            .map(|sm| sm.senses.iter().flat_map(|s| s.compositions.clone()).collect())
            .unwrap_or_default();

        // Union of compositions (deduplicated)
        let mut comp_set: HashSet<CompositionRef> = HashSet::new();
        for c in &comps_a {
            comp_set.insert(c.clone());
        }
        for c in &comps_b {
            comp_set.insert(c.clone());
        }
        let combined_compositions: Vec<CompositionRef> = comp_set.into_iter().collect();

        // If both have no compositions, use the component node IDs themselves
        // as compositions (they become the structural bridge)
        let final_compositions = if combined_compositions.is_empty() {
            vec![
                CompositionRef::new(node_id_a, 0),
                CompositionRef::new(node_id_b, 0),
            ]
        } else {
            combined_compositions
        };

        // Compute layer
        let layer_a = graph.get_node(node_id_a).map(|n| n.semantic.layer).unwrap_or(0);
        let layer_b = graph.get_node(node_id_b).map(|n| n.semantic.layer).unwrap_or(0);
        let compound_layer = layer_a.max(layer_b) + 1;

        // Average confidence
        let conf_a = graph.get_node(node_id_a).map(|n| n.confidence).unwrap_or(0.5);
        let conf_b = graph.get_node(node_id_b).map(|n| n.confidence).unwrap_or(0.5);
        let avg_confidence = (conf_a + conf_b) / 2.0;

        // Create the compound node
        let node = Node {
            id: 0,
            label: compound_label.clone(),
            surface_label: compound_label.clone(),
            kind: "node".to_string(),
            tier: Tier::Tier2,
            confidence: avg_confidence,
            status: NodeStatus::Candidate,
            is_seed: false,
            is_locked: false,
            semantic: SemanticMeta {
                compression_state: CompressionState::Compressed,
                layer: compound_layer,
                derived_from_node_ids: vec![node_id_a, node_id_b],
                compression_reason: Some("compound_discovery".to_string()),
                internal_representation: false,
                is_utterance: false,
                utterance_tokens: Vec::new(),
            },
            policy_meta: None,
            language_links: vec![],
            atoms: vec![node_id_a, node_id_b],
            fingerprint: None,
            gap_annotations: HashMap::new(),
            sense_profiles: HashMap::new(),
            discourse_meta: None,
            blend_results: HashMap::new(),
            abductive_hypotheses: Vec::new(),
            pattern_memberships: Vec::new(),
            synthesis_results: HashMap::new(),
        };

        let compound_node_id = graph.insert_node(node).ok()?;

        // Create edges from components to compound (never decay)
        graph.insert_edge(Edge {
            from: node_id_a,
            to: compound_node_id,
            weight: 1.0,
            source: EdgeSource::CompoundDiscovery,
            last_reinforced_batch: 0,
            relation_type: RelationType::Categorical,
        }).ok()?;

        graph.insert_edge(Edge {
            from: node_id_b,
            to: compound_node_id,
            weight: 1.0,
            source: EdgeSource::CompoundDiscovery,
            last_reinforced_batch: 0,
            relation_type: RelationType::Categorical,
        }).ok()?;

        // Create a sense manager for the compound node
        // with compositional sense = union of both components
        let mut sm = SenseManager::new(crate::sense::SenseConfig::default());
        sm.create_compositional_sense(final_compositions.clone(), compound_layer);
        senses.insert(compound_node_id, sm);

        // Mark as discovered
        self.discovered.insert(compound_label.clone());

        Some(DiscoveredCompound {
            label: compound_label,
            component_a: label_a.to_string(),
            component_b: label_b.to_string(),
            npmi: 0.0, // Will be filled by caller
            cooc_count: 0, // Will be filled by caller
            node_id_a: Some(node_id_a),
            node_id_b: Some(node_id_b),
            compound_node_id: Some(compound_node_id),
            combined_compositions: final_compositions,
        })
    }

    /// Process a batch: scan for compounds and create compound nodes.
    ///
    /// Returns discovered compounds that were created in this batch.
    pub fn process_batch(
        &mut self,
        stats: &CoocStats,
        token_to_id: &HashMap<String, NodeId>,
        seed_node_ids: &std::collections::HashSet<NodeId>,
        graph: &mut RsvsGraph,
        senses: &mut HashMap<NodeId, SenseManager>,
    ) -> Vec<DiscoveredCompound> {
        let candidates = self.scan_candidates(stats, token_to_id, seed_node_ids);
        let mut results = Vec::new();

        for (token_a, token_b, npmi, cooc_count) in candidates {
            let node_id_a = token_to_id.get(&token_a).copied();
            let node_id_b = token_to_id.get(&token_b).copied();

            let (id_a, id_b) = match (node_id_a, node_id_b) {
                (Some(a), Some(b)) => (a, b),
                _ => continue,
            };

            // Determine compound label order: maintain surface order
            // If token_a appears before token_b in co-occurrence, keep that order
            let (label_a, label_b, nid_a, nid_b) = (token_a.clone(), token_b.clone(), id_a, id_b);

            if let Some(mut compound) = self.create_compound_node(
                &label_a, &label_b, nid_a, nid_b, graph, senses,
            ) {
                compound.npmi = npmi;
                compound.cooc_count = cooc_count;
                results.push(compound);
            }

            if results.len() >= self.config.max_compounds_per_batch {
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
    use crate::attention::CoocStats;

    #[test]
    fn compound_discovery_config_defaults() {
        let config = CompoundDiscoveryConfig::default();
        assert!((config.min_npmi - 0.3).abs() < 0.01);
        assert!((config.min_conditional - 0.8).abs() < 0.01);
        assert_eq!(config.min_cooc_count, 3);
        assert_eq!(config.max_compound_length, 2);
    }

    #[test]
    fn scan_finds_high_conditional_probability_pairs() {
        let mut stats = CoocStats::new();
        // "harga" and "diri" always co-occur → P(diri|harga) = 1.0
        for _ in 0..5 {
            stats.ingest_sentence(&["harga".into(), "diri".into(), "value".into()]);
        }
        // "batu" and "keras" sometimes co-occur
        stats.ingest_sentence(&["batu".into(), "keras".into()]);
        stats.ingest_sentence(&["batu".into(), "besar".into()]);

        let mut token_to_id = HashMap::new();
        token_to_id.insert("harga".to_string(), 100);
        token_to_id.insert("diri".to_string(), 101);
        token_to_id.insert("value".to_string(), 1);
        token_to_id.insert("batu".to_string(), 200);
        token_to_id.insert("keras".to_string(), 201);
        token_to_id.insert("besar".to_string(), 202);

        let engine = CompoundDiscoveryEngine::new(CompoundDiscoveryConfig {
            min_conditional: 0.8,
            min_cooc_count: 3,
            ..CompoundDiscoveryConfig::default()
        });

        let candidates = engine.scan_candidates(&stats, &token_to_id, &HashSet::new());

        // Should find (harga, diri) as a candidate (P(diri|harga) = 1.0, P(harga|diri) = 1.0)
        assert!(!candidates.is_empty(), "Should find at least one compound candidate");

        // Check that (harga, diri) pair is among the candidates
        let has_harga_diri = candidates.iter().any(|(a, b, score, count)| {
            let is_pair = (a == "harga" && b == "diri") || (a == "diri" && b == "harga");
            is_pair && *score >= 0.8 && *count >= 3
        });
        assert!(has_harga_diri, "Should find (harga, diri) as a compound candidate with high conditional probability");
    }

    #[test]
    fn scan_ignores_low_conditional_probability_pairs() {
        let mut stats = CoocStats::new();
        // "batu" appears with many different words → low conditional with any specific word
        stats.ingest_sentence(&["batu".into(), "keras".into()]);
        stats.ingest_sentence(&["batu".into(), "besar".into()]);
        stats.ingest_sentence(&["batu".into(), "hitam".into()]);
        stats.ingest_sentence(&["batu".into(), "berat".into()]);

        let mut token_to_id = HashMap::new();
        token_to_id.insert("batu".to_string(), 100);
        token_to_id.insert("keras".to_string(), 101);
        token_to_id.insert("besar".to_string(), 102);

        let engine = CompoundDiscoveryEngine::new(CompoundDiscoveryConfig {
            min_npmi: 0.7,
            min_cooc_count: 2,
            ..CompoundDiscoveryConfig::default()
        });

        let candidates = engine.scan_candidates(&stats, &token_to_id, &HashSet::new());
        // "batu" appears with many different words → NPMI is low for each pair
        assert!(candidates.is_empty(), "Low NPMI pairs should not be candidates");
    }

    #[test]
    fn create_compound_node_produces_correct_structure() {
        let mut engine = CompoundDiscoveryEngine::new(CompoundDiscoveryConfig::default());
        let mut graph = RsvsGraph::new();
        let mut senses = HashMap::new();

        // Create "harga" node with value composition
        let harga_id = graph.insert_node(Node {
            label: "harga".to_string(),
            semantic: SemanticMeta {
                layer: 1,
                compression_state: CompressionState::Compressed,
                ..SemanticMeta::default()
            },
            ..Node::default()
        }).unwrap();

        let mut sm_harga = SenseManager::new(crate::sense::SenseConfig::default());
        sm_harga.create_compositional_sense(
            vec![CompositionRef::new(1, 0)], // value seed
            1,
        );
        senses.insert(harga_id, sm_harga);

        // Create "diri" node with identity composition
        let diri_id = graph.insert_node(Node {
            label: "diri".to_string(),
            semantic: SemanticMeta {
                layer: 1,
                compression_state: CompressionState::Compressed,
                ..SemanticMeta::default()
            },
            ..Node::default()
        }).unwrap();

        let mut sm_diri = SenseManager::new(crate::sense::SenseConfig::default());
        sm_diri.create_compositional_sense(
            vec![CompositionRef::new(4, 0)], // identity seed
            1,
        );
        senses.insert(diri_id, sm_diri);

        // Create compound
        let result = engine.create_compound_node(
            "harga", "diri", harga_id, diri_id, &mut graph, &mut senses,
        );

        assert!(result.is_some(), "Should create compound node");
        let compound = result.unwrap();

        assert_eq!(compound.label, "harga_diri");
        assert_eq!(compound.component_a, "harga");
        assert_eq!(compound.component_b, "diri");

        // Check compound node in graph
        let compound_id = compound.compound_node_id.unwrap();
        let compound_node = graph.get_node(compound_id).unwrap();
        assert_eq!(compound_node.label, "harga_diri");
        assert_eq!(compound_node.semantic.layer, 2, "Compound should be layer 2");
        assert_eq!(compound_node.semantic.compression_reason, Some("compound_discovery".to_string()));
        assert_eq!(compound_node.semantic.derived_from_node_ids, vec![harga_id, diri_id]);

        // Check edges
        let edges_from_harga = graph.edges_from(harga_id);
        assert!(edges_from_harga.iter().any(|e| e.to == compound_id && e.source == EdgeSource::CompoundDiscovery));
        let edges_from_diri = graph.edges_from(diri_id);
        assert!(edges_from_diri.iter().any(|e| e.to == compound_id && e.source == EdgeSource::CompoundDiscovery));

        // Check sense: should have compositions from BOTH harga (value) and diri (identity)
        let compound_sm = senses.get(&compound_id).unwrap();
        assert!(!compound_sm.senses.is_empty(), "Compound should have a sense");
        let compound_sense = &compound_sm.senses[0];
        assert_eq!(compound_sense.compositions.len(), 2, "Compound sense should combine both compositions");

        let comp_node_ids: Vec<NodeId> = compound_sense.compositions.iter().map(|c| c.node_id).collect();
        assert!(comp_node_ids.contains(&1), "Should contain value seed from harga");
        assert!(comp_node_ids.contains(&4), "Should contain identity seed from diri");

        println!("\n=== COMPOUND DISCOVERY: harga + diri → harga_diri ===");
        println!("Component A: harga (node {}) → compositions: [value(1)]", harga_id);
        println!("Component B: diri (node {}) → compositions: [identity(4)]", diri_id);
        println!("Compound: harga_diri (node {}) → compositions: [value(1), identity(4)]", compound_id);
        println!("Layer: 2 (composed from two layer-1 nodes)");
        println!("Meaning: value + identity = 'dignity/self-esteem'");
    }

    #[test]
    fn already_discovered_compound_not_recreated() {
        let mut engine = CompoundDiscoveryEngine::new(CompoundDiscoveryConfig::default());
        let mut graph = RsvsGraph::new();
        let mut senses = HashMap::new();

        let harga_id = graph.insert_node(Node {
            label: "harga".to_string(),
            ..Node::default()
        }).unwrap();
        let diri_id = graph.insert_node(Node {
            label: "diri".to_string(),
            ..Node::default()
        }).unwrap();

        // Create first compound
        let _ = engine.create_compound_node(
            "harga", "diri", harga_id, diri_id, &mut graph, &mut senses,
        );

        // Try to create again — should return None
        let result = engine.create_compound_node(
            "harga", "diri", harga_id, diri_id, &mut graph, &mut senses,
        );
        assert!(result.is_none(), "Should not recreate already discovered compound");
    }
}
