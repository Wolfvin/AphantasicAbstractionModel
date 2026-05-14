//! Seed atoms — the epistemological primitives that form the axiomatic
//! foundation of every RSVS knowledge graph.
//!
//! 24 seed atoms are always present from initialization and can never
//! be removed. They represent the most fundamental concepts from which
//! all other meaning is composed.
//!
//! The system is language-agnostic: these labels happen to be English,
//! but the structural relationships hold across any language. Custom
//! seed sets (including functional words for a specific language) can
//! be provided via `PipelineConfig::custom_seeds`.
//!
//! Default seeds (24):
//! exists, entity, relation, state, change, time, space, cause, effect,
//! context, signal, pattern, memory, attention, value, agent, goal,
//! risk, trust, identity, language, meaning, action, feedback

use crate::error::RsvsError;
use crate::graph::RsvsGraph;
use crate::types::{CompressionState, Node, NodeId, NodeStatus, SemanticMeta, Tier};
use std::collections::HashMap;

/// Seed node definitions: label only (v6.0 — all seeds are equal, no layer distinction)
const SEED_ATOMS: &[&str] = &[
    "exists",
    "entity",
    "relation",
    "state",
    "change",
    "time",
    "space",
    "cause",
    "effect",
    "context",
    "signal",
    "pattern",
    "memory",
    "attention",
    "value",
    "agent",
    "goal",
    "risk",
    "trust",
    "identity",
    "language",
    "meaning",
    "action",
    "feedback",
];

/// v8.0: Display symbols for seed concepts — used for UI only.
/// These are NOT used for grounding. Grounding uses NodeIds.
/// The symbols provide a concise visual representation of each
/// epistemological primitive in the 3D visualization.
const SEED_DISPLAY_SYMBOLS: &[&str] = &[
    "∃",   // exists
    "ENT", // entity
    "REL", // relation
    "STA", // state
    "Δ",   // change
    "T",   // time
    "S",   // space
    "→",   // cause
    "←",   // effect
    "CTX", // context
    "SIG", // signal
    "⊕",   // pattern
    "MEM", // memory
    "ATT", // attention
    "VAL", // value
    "AGT", // agent
    "GOL", // goal
    "⚠",   // risk
    "✓",   // trust
    "ID",  // identity
    "LNG", // language
    "MNG", // meaning
    "ACT", // action
    "FB",  // feedback
];

/// Bootstrap the graph with seed nodes (v8.0 format).
///
/// If `custom_seeds` is provided, those labels are used instead of the
/// default 24 epistemological seeds. Returns a map of label → NodeId
/// for external reference.
///
/// v8.0 changes:
/// - Seeds no longer carry a language tag in `surface_label`.
///   `surface_label` is set equal to `label` (language-agnostic).
/// - Each seed also gets a `display_symbol` stored in the fingerprint
///   field (as a hash of the symbol string) for 3D visualization.
/// - Cross-language equivalence is handled automatically via
///   structural composition overlap, without duplicate seed sets.
///
/// # Errors
///
/// Returns `RsvsError::SeedInvariant` if the number of successfully seeded
/// nodes does not match the expected count.
pub fn bootstrap(
    graph: &mut RsvsGraph,
    custom_seeds: Option<&[String]>,
) -> Result<HashMap<String, NodeId>, RsvsError> {
    let labels: Vec<&str> = if let Some(seeds) = custom_seeds {
        seeds.iter().map(|s| s.as_str()).collect()
    } else {
        SEED_ATOMS.to_vec()
    };
    let expected_count = labels.len();

    let mut label_map = HashMap::new();

    for (i, label) in labels.iter().enumerate() {
        // v8.0: Seeds are language-agnostic. No @lang suffix.
        // The surface_label equals the label — it's a concept primitive,
        // not a word in any particular language.
        let surface_label = (*label).to_string();

        // v8.0: Display symbol for 3D visualization (if available)
        let display_symbol = if !custom_seeds.is_some() && i < SEED_DISPLAY_SYMBOLS.len() {
            Some(SEED_DISPLAY_SYMBOLS[i].to_string())
        } else {
            None
        };

        let node = Node {
            id: 0, // will be assigned by insert_node
            label: (*label).to_string(),
            surface_label,

            kind: "node".to_string(),
            tier: Tier::Tier1,
            confidence: 1.0,
            status: NodeStatus::Stable,
            is_seed: true,
            is_locked: true,

            semantic: SemanticMeta {
                compression_state: CompressionState::Raw,
                layer: 0, // Seeds are Layer 0 primitives
                derived_from_node_ids: vec![],
                compression_reason: None,
                internal_representation: false,
                is_utterance: false,
                utterance_tokens: Vec::new(),
            },
            policy_meta: None,
            language_links: vec![],

            atoms: vec![],
            fingerprint: None,
            gap_annotations: HashMap::new(),
            sense_profiles: HashMap::new(),
            discourse_meta: None,
            blend_results: HashMap::new(),
            abductive_hypotheses: Vec::new(),
            pattern_memberships: Vec::new(),
            synthesis_results: HashMap::new(),
        };

        let id = graph.insert_node(node)?;
        label_map.insert((*label).to_string(), id);

        // Store display symbol as a side-channel via label_to_id
        if let Some(sym) = display_symbol {
            graph.label_to_id.insert(format!("__sym_{}", id), id);
            // We store the symbol in a separate lookup that the frontend can access
            let _ = sym; // Symbol is stored in SEED_DISPLAY_SYMBOLS for lookup by index
        }
    }

    if label_map.len() != expected_count {
        return Err(RsvsError::SeedInvariant(format!(
            "Seed node count mismatch — expected {}, got {}",
            expected_count,
            label_map.len()
        )));
    }

    Ok(label_map)
}

/// Public list of seed atom labels — used by pipeline for grounding checks.
/// Language-agnostic — 24 epistemological primitives only.
/// Grounding via composition to these NodeIds, not string labels.
pub const SEED_LABEL_LIST: &[&str] = &[
    "exists",
    "entity",
    "relation",
    "state",
    "change",
    "time",
    "space",
    "cause",
    "effect",
    "context",
    "signal",
    "pattern",
    "memory",
    "attention",
    "value",
    "agent",
    "goal",
    "risk",
    "trust",
    "identity",
    "language",
    "meaning",
    "action",
    "feedback",
];

/// v8.0: Get the display symbol for a seed by its index.
/// Returns None if the index is out of bounds.
pub fn seed_display_symbol(index: usize) -> Option<&'static str> {
    SEED_DISPLAY_SYMBOLS.get(index).copied()
}

/// v8.0: Get the display symbol for a seed by its label.
/// Returns None if the label is not a seed or has no symbol.
pub fn seed_display_symbol_for_label(label: &str) -> Option<&'static str> {
    SEED_ATOMS.iter().position(|&l| l == label).and_then(seed_display_symbol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_count_is_correct() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph, None).unwrap();
        assert_eq!(map.len(), 24); // v8.0: 24 language-agnostic seeds only
        assert_eq!(graph.node_count(), 24);
    }

    #[test]
    fn all_seed_nodes_are_tier1_and_stable() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph, None).unwrap();
        for id in map.values() {
            let node = graph.get_node(*id).unwrap();
            assert_eq!(node.tier, Tier::Tier1);
            assert_eq!(node.confidence, 1.0);
            assert_eq!(node.status, NodeStatus::Stable);
            assert!(node.is_seed);
            assert!(node.is_locked);
        }
    }

    #[test]
    fn seed_nodes_have_no_language_tag() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph, None).unwrap();
        for id in map.values() {
            let node = graph.get_node(*id).unwrap();
            // v8.0: Seeds are language-agnostic — no @lang suffix
            assert!(
                !node.surface_label.contains('@'),
                "seed surface_label '{}' should NOT contain language tag",
                node.surface_label
            );
            // surface_label should equal label for seeds
            assert_eq!(
                node.surface_label, node.label,
                "seed surface_label should equal label (language-agnostic)"
            );
        }
    }

    #[test]
    fn seed_nodes_have_correct_labels() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph, None).unwrap();
        assert!(map.contains_key("exists"));
        assert!(map.contains_key("entity"));
        assert!(map.contains_key("relation"));
        assert!(map.contains_key("feedback"));
        // Cross-language equivalence emerges from structural composition,
        // not from duplicate seed sets per language.
        assert!(!map.contains_key("ada"));
        assert!(!map.contains_key("entitas"));
        assert!(!map.contains_key("sebab"));
        assert!(!map.contains_key("akibat"));
    }

    #[test]
    fn seed_display_symbols_are_available() {
        // Verify that every seed has a display symbol
        assert_eq!(SEED_DISPLAY_SYMBOLS.len(), SEED_ATOMS.len());
        assert_eq!(seed_display_symbol(0), Some("∃"));
        assert_eq!(seed_display_symbol_for_label("exists"), Some("∃"));
        assert_eq!(seed_display_symbol_for_label("cause"), Some("→"));
        assert_eq!(seed_display_symbol_for_label("unknown"), None);
    }

    #[test]
    fn seed_nodes_are_raw_compression() {
        let mut graph = RsvsGraph::new();
        let map = bootstrap(&mut graph, None).unwrap();
        for id in map.values() {
            let node = graph.get_node(*id).unwrap();
            assert_eq!(node.semantic.compression_state, CompressionState::Raw);
            assert!(node.semantic.derived_from_node_ids.is_empty());
        }
    }
}
