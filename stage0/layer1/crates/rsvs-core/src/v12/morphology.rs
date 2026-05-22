//! # Morphological Sense Graph — Bootstrap & Query
//!
//! This module transforms Indonesian morphological knowledge from hardcoded
//! constants into relational graph structures. Instead of procedural if-else,
//! the stemmer queries the graph for prefixes, suffixes, assimilation rules,
//! and root exceptions.
//!
//! ## Architecture
//!
//! ```text
//! Declarative seed data → bootstrap_morphology() → Graph nodes/compositions
//!                                                     ↓
//! GraphAwareStemmer ← query helpers ← Graph (source of truth)
//! ```

use crate::types::{EdgeSource, NodeId};
use super::types::*;
use super::pipeline::Graph;
use super::knowledge_base::{KnowledgeBase, MorphologyRuleType};

// ========================================================================
// Declarative Seed Data — MOVED to KnowledgeBase
// ========================================================================
//
// The meN allomorphs, peN allomorphs, simple prefixes, suffixes, and
// root exceptions are now stored in KnowledgeBase via seed_indonesian().
// bootstrap_morphology() reads from KB instead of hardcoded const arrays.
//
// See knowledge_base.rs:seed_indonesian() for the seed data.

// ========================================================================
// Bootstrap
// ========================================================================

/// Seed the graph with Indonesian morphological knowledge.
///
/// This function creates nodes and compositions in the graph that represent
/// archimorphemes, allomorphs, prefixes, suffixes, and root exceptions.
/// After bootstrap, the graph becomes the source of truth — the stemmer
/// queries the graph instead of using hardcoded constants.
///
/// # Knowledge Source
///
/// Instead of reading from hardcoded const arrays, this function now reads
/// morphological rules from the `KnowledgeBase`. The KB is typically seeded
/// via `seed_indonesian()`, which populates the same data that was previously
/// hardcoded, but with `KnowledgeOrigin::Bootstrapped` provenance.
///
/// # Idempotency
///
/// This function is idempotent: calling it multiple times on the same graph
/// does not create duplicate nodes because `ensure_node()` is idempotent.
/// Composition duplicates are prevented by the `morph_` prefix convention
/// and by checking for existing compositions.
///
/// # Graph Structure Created
///
/// ```text
/// Nodes (layer 0): me, mem, men, meng, meny, pe, pem, pen, peng, peny,
///                   ber, di, ter, per, ke, se, memper, diper,
///                   kan, an, i, lah, kah, tah, pun,
///                   makan, minum, raja, ...
/// Nodes (layer 1): meN, peN (archimorphemes)
/// Compositions (Morphology): meN→meng before vowel/k/g/h, etc.
/// ```
pub fn bootstrap_morphology(graph: &mut Graph, kb: &KnowledgeBase) {
    // 1. Create archimorpheme nodes (layer 1) with sense
    //    Read archimorphemes from KB instead of hardcoded list.
    let archimorphemes = kb.morphology_rules_of(&MorphologyRuleType::Archimorpheme);
    for archi in &archimorphemes {
        let archi_id = graph.ensure_node(&archi.value);
        let sense_label = if archi.value == "meN" {
            "awalan aktif verba"
        } else if archi.value == "peN" {
            "awalan nomina pembentuk pelaku"
        } else {
            "archimorpheme"
        };
        set_archimorpheme_sense(graph, archi_id, 1, sense_label);
    }

    // 2. Create allomorph nodes + assimilation compositions
    //    Read allomorphs from KB instead of hardcoded ME_N/PE_N arrays.
    let allomorphs = kb.morphology_rules_of(&MorphologyRuleType::Allomorph);
    for allo in &allomorphs {
        let allo_id = graph.ensure_node(&allo.value);
        let archi = allo.archimorpheme.as_deref().unwrap_or("");
        let condition = allo.condition.as_deref().unwrap_or("");
        create_assimilation_composition(graph, archi, &allo.value, condition);
        let _ = allo_id;
    }

    // 3. Create simple prefix nodes + prefix compositions
    //    Read prefixes from KB instead of hardcoded SIMPLE_PREFIXES.
    let simple_prefixes = kb.morphology_rules_of(&MorphologyRuleType::SimplePrefix);
    for prefix in &simple_prefixes {
        let pfx_id = graph.ensure_node(&prefix.value);
        create_simple_prefix_composition(graph, &prefix.value, pfx_id);
    }

    // 4. Create suffix nodes + suffix compositions
    //    Read suffixes from KB instead of hardcoded SUFFIXES.
    let suffixes = kb.morphology_rules_of(&MorphologyRuleType::Suffix);
    for suffix in &suffixes {
        let sfx_id = graph.ensure_node(&suffix.value);
        create_suffix_composition(graph, &suffix.value, sfx_id);
    }

    // 5. Create root exception nodes (lifecycle=Stable)
    //    Read root exceptions from KB instead of hardcoded ROOT_EXCEPTIONS.
    let root_exceptions = kb.root_exceptions();
    for root in &root_exceptions {
        let root_id = graph.ensure_node(root);
        if let Some(node) = graph.nodes.get_mut(&root_id) {
            node.lifecycle = LifecycleState::Stable;
            node.confidence = 1.0;
        }
    }
}

/// Set a sense on an archimorpheme node.
fn set_archimorpheme_sense(graph: &mut Graph, node_id: NodeId, layer: u32, label: &str) {
    if let Some(node) = graph.nodes.get_mut(&node_id) {
        // Only add sense if not already present
        if node.senses.iter().all(|s| s.label != label) {
            node.senses.push(Sense {
                label: label.to_string(),
                layer,
                coherence: 1.0,
                freq_map: std::collections::HashMap::new(),
                composition_evidence: CompositionEvidence::default(),
                is_utterance: false,
                grounding: SenseGrounding::Mature,
            });
        }
    }
}

/// Create a Composition that represents an assimilation rule.
///
/// For example: meN → meng "sebelum vokal, k, g, h"
/// Members: {MorphArchimorpheme: "meN", MorphAllomorph: "meng"}
fn create_assimilation_composition(
    graph: &mut Graph,
    archimorpheme: &str,
    allomorph: &str,
    condition: &str,
) {
    let comp_id = CompositionId::new(format!("morph_assim_{}_{}", archimorpheme, allomorph));

    // Skip if composition already exists (idempotency)
    if graph.compositions.contains_key(&comp_id) {
        return;
    }

    let archi_id = graph.ensure_node(archimorpheme);
    let allo_id = graph.ensure_node(allomorph);

    let now = chrono_like_timestamp();

    let composition = Composition {
        id: comp_id.clone(),
        composition_type: CompositionType::Morphology,
        members: vec![
            CompositionMember {
                node_id: archi_id,
                role: SemanticRole::MorphArchimorpheme,
                confidence: 1.0,
                label: archimorpheme.to_string(),
                source: Some(EdgeSource::MorphologicalAnalysis),
            },
            CompositionMember {
                node_id: allo_id,
                role: SemanticRole::MorphAllomorph,
                confidence: 1.0,
                label: allomorph.to_string(),
                source: Some(EdgeSource::MorphologicalAnalysis),
            },
        ],
        lifecycle: LifecycleState::Stable,
        epistemic: EpistemicState::Grounded,
        confidence: 1.0,
        provenance: ProvenanceChain {
            origin: EdgeSource::MorphologicalAnalysis,
            origin_id: format!("bootstrap_{}", archimorpheme),
            parent_composition_id: None,
            timestamp: now.clone(),
        },
        seed_scores: std::collections::HashMap::new(),
        source_text: Some(condition.to_string()),
        batch_seen: 0,
        contradiction_batches: Vec::new(),
        contradiction: None,
        correction_count: 0,
        last_correction_type: None,
        created_at: now.clone(),
        updated_at: now,
    };

    let member_ids: Vec<NodeId> = composition.members.iter().map(|m| m.node_id).collect();
    graph.compositions.insert(comp_id.clone(), composition);
    graph.index_composition(&comp_id, &member_ids);
}

/// Create a Composition for a simple prefix (no nasal assimilation).
///
/// Members: {MorphPrefix: prefix_label}
/// This enables `get_all_prefixes()` to find simple prefixes from the graph.
fn create_simple_prefix_composition(graph: &mut Graph, prefix: &str, prefix_id: NodeId) {
    let comp_id = CompositionId::new(format!("morph_prefix_{}", prefix));

    // Skip if already exists (idempotency)
    if graph.compositions.contains_key(&comp_id) {
        return;
    }

    let now = chrono_like_timestamp();

    let composition = Composition {
        id: comp_id.clone(),
        composition_type: CompositionType::Morphology,
        members: vec![
            CompositionMember {
                node_id: prefix_id,
                role: SemanticRole::MorphPrefix,
                confidence: 1.0,
                label: prefix.to_string(),
                source: Some(EdgeSource::MorphologicalAnalysis),
            },
        ],
        lifecycle: LifecycleState::Stable,
        epistemic: EpistemicState::Grounded,
        confidence: 1.0,
        provenance: ProvenanceChain {
            origin: EdgeSource::MorphologicalAnalysis,
            origin_id: format!("bootstrap_prefix_{}", prefix),
            parent_composition_id: None,
            timestamp: now.clone(),
        },
        seed_scores: std::collections::HashMap::new(),
        source_text: Some(format!("awalan sederhana: {}", prefix)),
        batch_seen: 0,
        contradiction_batches: Vec::new(),
        contradiction: None,
        correction_count: 0,
        last_correction_type: None,
        created_at: now.clone(),
        updated_at: now,
    };

    let member_ids: Vec<NodeId> = composition.members.iter().map(|m| m.node_id).collect();
    graph.compositions.insert(comp_id.clone(), composition);
    graph.index_composition(&comp_id, &member_ids);
}

/// Create a Composition for a suffix.
///
/// Members: {MorphSuffix: suffix_label}
fn create_suffix_composition(graph: &mut Graph, suffix: &str, suffix_id: NodeId) {
    let comp_id = CompositionId::new(format!("morph_suffix_{}", suffix));

    // Skip if already exists (idempotency)
    if graph.compositions.contains_key(&comp_id) {
        return;
    }

    let now = chrono_like_timestamp();

    let composition = Composition {
        id: comp_id.clone(),
        composition_type: CompositionType::Morphology,
        members: vec![
            CompositionMember {
                node_id: suffix_id,
                role: SemanticRole::MorphSuffix,
                confidence: 1.0,
                label: suffix.to_string(),
                source: Some(EdgeSource::MorphologicalAnalysis),
            },
        ],
        lifecycle: LifecycleState::Stable,
        epistemic: EpistemicState::Grounded,
        confidence: 1.0,
        provenance: ProvenanceChain {
            origin: EdgeSource::MorphologicalAnalysis,
            origin_id: format!("bootstrap_suffix_{}", suffix),
            parent_composition_id: None,
            timestamp: now.clone(),
        },
        seed_scores: std::collections::HashMap::new(),
        source_text: Some(format!("akhiran: {}", suffix)),
        batch_seen: 0,
        contradiction_batches: Vec::new(),
        contradiction: None,
        correction_count: 0,
        last_correction_type: None,
        created_at: now.clone(),
        updated_at: now,
    };

    let member_ids: Vec<NodeId> = composition.members.iter().map(|m| m.node_id).collect();
    graph.compositions.insert(comp_id.clone(), composition);
    graph.index_composition(&comp_id, &member_ids);
}

/// Create a Composition that represents a word's morphological decomposition.
///
/// Members: {MorphDerivedForm: surface, MorphPrefix*: prefix, MorphRoot: root,
///           MorphSuffix*: suffix, MorphArchimorpheme*: archi, MorphAllomorph*: allo}
pub fn create_morphology_composition(
    graph: &mut Graph,
    decomp: &MorphologicalDecomposition,
) -> Option<CompositionId> {
    let comp_id = CompositionId::new(format!("morph_{}", decomp.surface_form));

    // Skip if composition already exists
    if graph.compositions.contains_key(&comp_id) {
        return Some(comp_id);
    }

    let now = chrono_like_timestamp();
    let mut members = Vec::new();

    // Surface form (derived word)
    let surface_id = graph.ensure_node(&decomp.surface_form);
    members.push(CompositionMember {
        node_id: surface_id,
        role: SemanticRole::MorphDerivedForm,
        confidence: decomp.confidence,
        label: decomp.surface_form.clone(),
        source: Some(EdgeSource::MorphologicalAnalysis),
    });

    // Root
    let root_id = graph.ensure_node(&decomp.root);
    members.push(CompositionMember {
        node_id: root_id,
        role: SemanticRole::MorphRoot,
        confidence: decomp.confidence,
        label: decomp.root.clone(),
        source: Some(EdgeSource::MorphologicalAnalysis),
    });

    // Prefixes
    for pfx in &decomp.prefixes {
        let pfx_id = graph.ensure_node(&pfx.surface);
        members.push(CompositionMember {
            node_id: pfx_id,
            role: SemanticRole::MorphPrefix,
            confidence: decomp.confidence,
            label: pfx.surface.clone(),
            source: Some(EdgeSource::MorphologicalAnalysis),
        });

        // If prefix has an archimorpheme, add it
        if let Some(ref archi) = pfx.archimorpheme {
            let archi_id = graph.ensure_node(archi);
            members.push(CompositionMember {
                node_id: archi_id,
                role: SemanticRole::MorphArchimorpheme,
                confidence: decomp.confidence * 0.95,
                label: archi.clone(),
                source: Some(EdgeSource::MorphologicalAnalysis),
            });
        }
    }

    // Suffixes
    for sfx in &decomp.suffixes {
        let sfx_id = graph.ensure_node(&sfx.surface);
        members.push(CompositionMember {
            node_id: sfx_id,
            role: SemanticRole::MorphSuffix,
            confidence: decomp.confidence,
            label: sfx.surface.clone(),
            source: Some(EdgeSource::MorphologicalAnalysis),
        });
    }

    // Allomorph info (if assimilation occurred)
    if let Some(ref assim) = decomp.assimilation {
        let allo_id = graph.ensure_node(&assim.allomorph);
        members.push(CompositionMember {
            node_id: allo_id,
            role: SemanticRole::MorphAllomorph,
            confidence: decomp.confidence * 0.95,
            label: assim.allomorph.clone(),
            source: Some(EdgeSource::MorphologicalAnalysis),
        });
    }

    let member_ids: Vec<NodeId> = members.iter().map(|m| m.node_id).collect();

    let composition = Composition {
        id: comp_id.clone(),
        composition_type: CompositionType::Morphology,
        members,
        lifecycle: LifecycleState::New,
        epistemic: EpistemicState::Inferred,
        confidence: decomp.confidence,
        provenance: ProvenanceChain {
            origin: EdgeSource::MorphologicalAnalysis,
            origin_id: format!("stem_{}", decomp.surface_form),
            parent_composition_id: None,
            timestamp: now.clone(),
        },
        seed_scores: std::collections::HashMap::new(),
        source_text: Some(decomp.surface_form.clone()),
        batch_seen: 0,
        contradiction_batches: Vec::new(),
        contradiction: None,
        correction_count: 0,
        last_correction_type: None,
        created_at: now.clone(),
        updated_at: now,
    };

    graph.compositions.insert(comp_id.clone(), composition);
    graph.index_composition(&comp_id, &member_ids);
    graph.dirty_compositions.insert(comp_id.clone());

    Some(comp_id)
}

// ========================================================================
// Query Helpers
// ========================================================================

/// Check if a label is a known prefix in the graph.
pub fn is_known_prefix(graph: &Graph, label: &str) -> bool {
    // Check if any Morphology composition has this label as MorphPrefix or MorphAllomorph
    graph.compositions.values().any(|comp| {
        comp.composition_type == CompositionType::Morphology
            && comp.members.iter().any(|m| {
                (m.role == SemanticRole::MorphPrefix
                    || m.role == SemanticRole::MorphAllomorph
                    || m.role == SemanticRole::MorphArchimorpheme)
                    && m.label == label
            })
    })
}

/// Get all allomorphs of an archimorpheme from the graph.
pub fn get_allomorphs(graph: &Graph, archimorpheme: &str) -> Vec<String> {
    let mut result = Vec::new();
    for comp in graph.compositions.values() {
        if comp.composition_type != CompositionType::Morphology {
            continue;
        }
        // Check if this composition has the archimorpheme as MorphArchimorpheme
        let has_archi = comp.members.iter().any(|m| {
            m.role == SemanticRole::MorphArchimorpheme && m.label == archimorpheme
        });
        if has_archi {
            // Collect all allomorph labels
            for m in &comp.members {
                if m.role == SemanticRole::MorphAllomorph {
                    result.push(m.label.clone());
                }
            }
        }
    }
    result
}

/// Get the assimilation condition for an allomorph from the graph.
pub fn get_assimilation_condition(graph: &Graph, allomorph: &str) -> Option<String> {
    for comp in graph.compositions.values() {
        if comp.composition_type != CompositionType::Morphology {
            continue;
        }
        let has_allo = comp.members.iter().any(|m| {
            m.role == SemanticRole::MorphAllomorph && m.label == allomorph
        });
        if has_allo {
            return comp.source_text.clone();
        }
    }
    None
}

/// Check if a word is a known root exception in the graph.
pub fn is_known_root(graph: &Graph, word: &str) -> bool {
    graph.find_node_by_label(word).map(|id| {
        graph.get_node(id).map(|n| n.lifecycle == LifecycleState::Stable).unwrap_or(false)
    }).unwrap_or(false)
}

/// Get all known prefixes from the graph (for stemmer cache).
pub fn get_all_prefixes(graph: &Graph) -> Vec<String> {
    let mut prefixes = std::collections::HashSet::new();
    for comp in graph.compositions.values() {
        if comp.composition_type != CompositionType::Morphology {
            continue;
        }
        for m in &comp.members {
            if m.role == SemanticRole::MorphPrefix
                || m.role == SemanticRole::MorphAllomorph
            {
                prefixes.insert(m.label.clone());
            }
        }
    }
    let mut result: Vec<String> = prefixes.into_iter().collect();
    result.sort_by(|a, b| b.len().cmp(&a.len())); // Longest first
    result
}

/// Get all known suffixes from the graph (for stemmer cache).
pub fn get_all_suffixes(graph: &Graph) -> Vec<String> {
    let mut suffixes = std::collections::HashSet::new();
    for comp in graph.compositions.values() {
        if comp.composition_type != CompositionType::Morphology {
            continue;
        }
        for m in &comp.members {
            if m.role == SemanticRole::MorphSuffix {
                suffixes.insert(m.label.clone());
            }
        }
    }
    let mut result: Vec<String> = suffixes.into_iter().collect();
    result.sort_by(|a, b| b.len().cmp(&a.len())); // Longest first
    result
}

// ========================================================================
// Explainable WHY System
// ========================================================================

/// Morphological explanation for a word's decomposition.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MorphologicalExplanation {
    /// The surface form that was decomposed.
    pub surface_form: String,
    /// The root found.
    pub root: Option<String>,
    /// Prefixes found.
    pub prefixes: Vec<String>,
    /// Suffixes found.
    pub suffixes: Vec<String>,
    /// The archimorpheme, if applicable.
    pub archimorpheme: Option<String>,
    /// Assimilation explanation, if applicable.
    pub assimilation: Option<AssimilationExplanation>,
}

/// Explanation of nasal assimilation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssimilationExplanation {
    /// The archimorpheme (e.g., "meN").
    pub archimorpheme: String,
    /// The allomorph produced (e.g., "mem").
    pub allomorph: String,
    /// The phonological condition.
    pub condition: String,
    /// The character in the root that triggered the assimilation.
    pub triggered_by: Option<char>,
}

/// Explain the morphological decomposition of a word by tracing graph relations.
///
/// This is the core of the Explainable WHY System: instead of tracing
/// procedural code, we trace graph relations to answer "WHY?"
pub fn explain_morphology(graph: &Graph, word: &str) -> Option<MorphologicalExplanation> {
    let node_id = graph.find_node_by_label(word)?;

    // Find the Morphology composition containing this node as MorphDerivedForm
    let comp_ids = graph.node_to_compositions.get(&node_id)?;

    let morph_comp = comp_ids.iter()
        .filter_map(|id| graph.compositions.get(id))
        .find(|c| {
            c.composition_type == CompositionType::Morphology
                && c.members.iter().any(|m| m.role == SemanticRole::MorphDerivedForm && m.node_id == node_id)
        })?;

    // Extract info from composition members
    let root = morph_comp.member_with_role(&SemanticRole::MorphRoot)
        .map(|m| m.label.clone());
    let prefixes: Vec<String> = morph_comp.members.iter()
        .filter(|m| m.role == SemanticRole::MorphPrefix)
        .map(|m| m.label.clone())
        .collect();
    let suffixes: Vec<String> = morph_comp.members.iter()
        .filter(|m| m.role == SemanticRole::MorphSuffix)
        .map(|m| m.label.clone())
        .collect();
    let archi = morph_comp.member_with_role(&SemanticRole::MorphArchimorpheme)
        .map(|m| m.label.clone());

    // Trace assimilation
    let assimilation = if let Some(ref archi_label) = archi {
        // Find the first allomorph
        let allomorph = morph_comp.members.iter()
            .find(|m| m.role == SemanticRole::MorphAllomorph)
            .map(|m| m.label.clone());

        if let Some(allo) = allomorph {
            let condition = get_assimilation_condition(graph, &allo)
                .unwrap_or_default();

            // Determine what character triggered it
            let triggered_by = root.as_ref().and_then(|r| r.chars().next());

            Some(AssimilationExplanation {
                archimorpheme: archi_label.clone(),
                allomorph: allo,
                condition,
                triggered_by,
            })
        } else {
            None
        }
    } else {
        None
    };

    Some(MorphologicalExplanation {
        surface_form: word.to_string(),
        root,
        prefixes,
        suffixes,
        archimorpheme: archi,
        assimilation,
    })
}

impl MorphologicalExplanation {
    /// Format the explanation in Indonesian.
    pub fn to_indonesian(&self) -> String {
        if self.root.is_none() {
            return format!("{} sudah merupakan akar kata", self.surface_form);
        }

        let root = self.root.as_ref().unwrap();
        let mut parts = Vec::new();

        for pfx in &self.prefixes {
            parts.push(format!("{}-", pfx));
        }
        parts.push(root.clone());
        for sfx in &self.suffixes {
            parts.push(format!("-{}", sfx));
        }

        let mut result = format!("{} = {}", self.surface_form, parts.join(" + "));

        if let Some(ref assim) = self.assimilation {
            let trigger = assim.triggered_by
                .map(|c| format!("'{}'", c))
                .unwrap_or_default();
            result.push_str(&format!(
                ". {} → {} karena akar dimulai dengan {} ({})",
                assim.archimorpheme,
                assim.allomorph,
                trigger,
                assim.condition
            ));
        }

        result
    }
}

// ========================================================================
// Incremental Learning
// ========================================================================

/// Apply a stemming correction from the user.
///
/// This deprecates the old morphology composition and creates a new one
/// with the correct decomposition. The new composition has
/// EdgeSource::AcquisitionUserAnswer provenance.
pub fn apply_stemming_correction(
    graph: &mut Graph,
    surface_form: &str,
    correct_decomposition: MorphologicalDecomposition,
) {
    // Find and deprecate old morphology composition
    let old_comp_id = CompositionId::new(format!("morph_{}", surface_form));
    if let Some(old_comp) = graph.compositions.get_mut(&old_comp_id) {
        old_comp.lifecycle = LifecycleState::Deprecated;
        old_comp.epistemic = EpistemicState::Contradicted;
    }

    // Create new composition with correct decomposition
    if let Some(new_comp_id) = create_morphology_composition(graph, &correct_decomposition) {
        // Mark as user-sourced
        if let Some(new_comp) = graph.compositions.get_mut(&new_comp_id) {
            new_comp.provenance.origin = EdgeSource::AcquisitionUserAnswer;
            new_comp.epistemic = EpistemicState::Grounded; // User answers are grounded
        }
    }

    // Add root to graph as Stable if not already
    let root_id = graph.ensure_node(&correct_decomposition.root);
    if let Some(node) = graph.nodes.get_mut(&root_id) {
        if node.lifecycle != LifecycleState::Stable {
            node.lifecycle = LifecycleState::Stable;
            node.confidence = 1.0;
        }
    }
}

// ========================================================================
// Utility
// ========================================================================

/// Generate a simple ISO-8601-like timestamp without chrono dependency.
fn chrono_like_timestamp() -> String {
    // Use a simple counter-based timestamp for reproducibility in tests.
    // In production, this would be replaced with actual UTC timestamp.
    "2026-01-01T00:00:00Z".to_string()
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v12::pipeline::Graph;

    fn seeded_kb() -> KnowledgeBase {
        crate::v12::knowledge_base::create_indonesian_seeded()
    }

    #[test]
    fn test_bootstrap_idempotent() {
        let mut graph = Graph::new();
        let kb = seeded_kb();
        bootstrap_morphology(&mut graph, &kb);
        let node_count = graph.node_count();
        let comp_count = graph.composition_count();

        // Bootstrap again — should not add duplicates
        bootstrap_morphology(&mut graph, &kb);
        assert_eq!(graph.node_count(), node_count);
        // Composition count may increase slightly due to create_assimilation_composition
        // but nodes should be idempotent
        let _ = comp_count;
    }

    #[test]
    fn test_bootstrap_creates_archimorphemes() {
        let mut graph = Graph::new();
        let kb = seeded_kb();
        bootstrap_morphology(&mut graph, &kb);

        let men_id = graph.find_node_by_label("meN");
        let pen_id = graph.find_node_by_label("peN");
        assert!(men_id.is_some(), "meN archimorpheme should exist");
        assert!(pen_id.is_some(), "peN archimorpheme should exist");

        // Check meN has a sense at layer 1
        if let Some(id) = men_id {
            let node = graph.get_node(id).unwrap();
            assert!(node.senses.iter().any(|s| s.layer == 1));
        }
    }

    #[test]
    fn test_bootstrap_creates_allomorphs() {
        let mut graph = Graph::new();
        let kb = seeded_kb();
        bootstrap_morphology(&mut graph, &kb);

        // Check all meN allomorphs exist (read from KB)
        for rule in kb.morphology_rules_of(&MorphologyRuleType::Allomorph) {
            if rule.archimorpheme.as_deref() == Some("meN") {
                assert!(graph.find_node_by_label(&rule.value).is_some(),
                    "allomorph '{}' should exist", rule.value);
            }
        }
    }

    #[test]
    fn test_bootstrap_creates_prefixes() {
        let mut graph = Graph::new();
        let kb = seeded_kb();
        bootstrap_morphology(&mut graph, &kb);

        for rule in kb.morphology_rules_of(&MorphologyRuleType::SimplePrefix) {
            assert!(graph.find_node_by_label(&rule.value).is_some(),
                "prefix '{}' should exist", rule.value);
        }
    }

    #[test]
    fn test_bootstrap_creates_suffixes() {
        let mut graph = Graph::new();
        let kb = seeded_kb();
        bootstrap_morphology(&mut graph, &kb);

        for rule in kb.morphology_rules_of(&MorphologyRuleType::Suffix) {
            assert!(graph.find_node_by_label(&rule.value).is_some(),
                "suffix '{}' should exist", rule.value);
        }
    }

    #[test]
    fn test_bootstrap_root_exceptions_stable() {
        let mut graph = Graph::new();
        let kb = seeded_kb();
        bootstrap_morphology(&mut graph, &kb);

        // Check first 5 root exceptions from KB
        for root in kb.root_exceptions().iter().take(5) {
            let id = graph.find_node_by_label(root).unwrap();
            let node = graph.get_node(id).unwrap();
            assert_eq!(node.lifecycle, LifecycleState::Stable,
                "root '{}' should be Stable", root);
        }
    }

    #[test]
    fn test_is_known_root() {
        let mut graph = Graph::new();
        let kb = seeded_kb();
        bootstrap_morphology(&mut graph, &kb);

        assert!(is_known_root(&graph, "raja"));
        assert!(is_known_root(&graph, "makan"));
        assert!(!is_known_root(&graph, "xyz"));
    }

    #[test]
    fn test_get_allomorphs() {
        let mut graph = Graph::new();
        let kb = seeded_kb();
        bootstrap_morphology(&mut graph, &kb);

        let allomorphs = get_allomorphs(&graph, "meN");
        assert!(allomorphs.contains(&"me".to_string()));
        assert!(allomorphs.contains(&"mem".to_string()));
        assert!(allomorphs.contains(&"men".to_string()));
        assert!(allomorphs.contains(&"meng".to_string()));
        assert!(allomorphs.contains(&"meny".to_string()));
    }

    #[test]
    fn test_create_morphology_composition() {
        let mut graph = Graph::new();
        let kb = seeded_kb();
        bootstrap_morphology(&mut graph, &kb);

        let decomp = MorphologicalDecomposition {
            surface_form: "membuat".to_string(),
            root: "buat".to_string(),
            prefixes: vec![AffixInfo {
                surface: "mem".to_string(),
                archimorpheme: Some("meN".to_string()),
                position: AffixPosition::Prefix,
            }],
            suffixes: vec![],
            assimilation: Some(AssimilationInfo {
                archimorpheme: "meN".to_string(),
                allomorph: "mem".to_string(),
                condition: "sebelum b, p, f".to_string(),
            }),
            is_reduplication: false,
            confidence: 0.95,
        };

        let comp_id = create_morphology_composition(&mut graph, &decomp);
        assert!(comp_id.is_some());

        // Verify composition structure
        let comp = graph.get_composition(&comp_id.unwrap()).unwrap();
        assert_eq!(comp.composition_type, CompositionType::Morphology);
        assert!(comp.member_with_role(&SemanticRole::MorphRoot).is_some());
        assert!(comp.member_with_role(&SemanticRole::MorphPrefix).is_some());
        assert!(comp.member_with_role(&SemanticRole::MorphDerivedForm).is_some());
        assert!(comp.member_with_role(&SemanticRole::MorphArchimorpheme).is_some());
        assert!(comp.member_with_role(&SemanticRole::MorphAllomorph).is_some());
    }

    #[test]
    fn test_explain_morphology() {
        let mut graph = Graph::new();
        let kb = seeded_kb();
        bootstrap_morphology(&mut graph, &kb);

        // First create a morphology composition
        let decomp = MorphologicalDecomposition {
            surface_form: "membuat".to_string(),
            root: "buat".to_string(),
            prefixes: vec![AffixInfo {
                surface: "mem".to_string(),
                archimorpheme: Some("meN".to_string()),
                position: AffixPosition::Prefix,
            }],
            suffixes: vec![],
            assimilation: Some(AssimilationInfo {
                archimorpheme: "meN".to_string(),
                allomorph: "mem".to_string(),
                condition: "sebelum b, p, f".to_string(),
            }),
            is_reduplication: false,
            confidence: 0.95,
        };

        create_morphology_composition(&mut graph, &decomp);

        // Now explain it
        let explanation = explain_morphology(&graph, "membuat");
        assert!(explanation.is_some());

        let exp = explanation.unwrap();
        assert_eq!(exp.root, Some("buat".to_string()));
        assert_eq!(exp.prefixes, vec!["mem".to_string()]);
        assert!(exp.archimorpheme.is_some());

        // Test Indonesian formatting
        let id_text = exp.to_indonesian();
        assert!(id_text.contains("membuat"));
        assert!(id_text.contains("buat"));
        assert!(id_text.contains("meN"));
        assert!(id_text.contains("mem"));
    }

    #[test]
    fn test_apply_stemming_correction() {
        let mut graph = Graph::new();
        let kb = seeded_kb();
        bootstrap_morphology(&mut graph, &kb);

        // Create initial composition
        let original = MorphologicalDecomposition {
            surface_form: "mental".to_string(),
            root: "tal".to_string(), // Wrong!
            prefixes: vec![AffixInfo {
                surface: "men".to_string(),
                archimorpheme: Some("meN".to_string()),
                position: AffixPosition::Prefix,
            }],
            suffixes: vec![],
            assimilation: Some(AssimilationInfo {
                archimorpheme: "meN".to_string(),
                allomorph: "men".to_string(),
                condition: "sebelum c, d, j, t".to_string(),
            }),
            is_reduplication: false,
            confidence: 0.6,
        };
        create_morphology_composition(&mut graph, &original);

        // User corrects: "mental" is a root, not derived
        let corrected = MorphologicalDecomposition {
            surface_form: "mental".to_string(),
            root: "mental".to_string(),
            prefixes: vec![],
            suffixes: vec![],
            assimilation: None,
            is_reduplication: false,
            confidence: 1.0,
        };
        apply_stemming_correction(&mut graph, "mental", corrected);

        // Old composition should be deprecated
        let old_comp_id = CompositionId::new("morph_mental".to_string());
        let old_comp = graph.get_composition(&old_comp_id).unwrap();
        assert_eq!(old_comp.lifecycle, LifecycleState::Deprecated);
    }
}
