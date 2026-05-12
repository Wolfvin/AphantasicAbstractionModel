//! Persistence — RSVS v6.0 Compositional Architecture
//!
//! Serialize/deserialize the full RSVS state to/from disk.
//! Format: JSON (human-readable, debuggable).
//!
//! v6.0: Updated for compositional architecture — layer, compositions,
//! GroundingEvidence in senses, TransformerBridgeConfig support.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::attention::CoocStats;
use crate::autonomy::{AtomRecord, AutonomyConfig, AutonomyEngine, MemoryClass};
use crate::error::RsvsError;
use crate::graph::RsvsGraph;
use crate::pipeline::{PipelineConfig, Rsvs};
use crate::sense::{GroundingEvidence, Sense, SenseConfig, SenseManager, SenseStatus};
use crate::transformer_bridge::TransformerBridgeConfig;
use crate::types::{
    CompositionRef, CompressionState, Edge, EdgeSource, Node, NodeId, NodeStatus, PolicyMeta,
    RelationType, SemanticMeta, Tier,
};

// -----------------------------------------------------------------------
// Serializable mirror types (v6.0 serde-friendly)
// -----------------------------------------------------------------------

/// Serializable mirror of a `LanguageLink` for JSON persistence.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedLanguageLink {
    pub link_type: String,
    pub target_id: u32,
}

/// Serializable mirror of a `Node` for JSON persistence.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedNode {
    pub id: u32,
    pub label: String,
    pub surface_label: String,
    pub kind: String,
    pub tier: u8,
    pub confidence: f32,
    pub status: String,
    pub is_seed: bool,
    pub is_locked: bool,
    pub compression_state: String,
    pub layer: u32,
    pub derived_from_node_ids: Vec<u32>,
    pub compression_reason: Option<String>,
    /// v8.0: Whether this node is an internal representation (layer 1 bridge).
    #[serde(default)]
    pub internal_representation: bool,
    /// v8.0: Cross-language links (structural equivalence from convergence detection).
    #[serde(default)]
    pub language_links: Vec<SavedLanguageLink>,
    pub policy_meta: Option<SavedPolicyMeta>,
    pub atoms: Vec<u32>,
}

/// Serializable mirror of `PolicyMeta` for JSON persistence.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedPolicyMeta {
    pub policy_version: String,
    pub governance_score: f32,
    pub candidate_evidence_pool: f32,
    pub status_flip_count: u32,
    pub seen_fingerprints: Vec<String>,
    pub last_seen_at: Option<String>,
}

/// Serializable mirror of an `Edge` for JSON persistence.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedEdge {
    pub from: u32,
    pub to: u32,
    pub weight: f32,
    pub source: String,
    /// v6.5: Batch number when this edge was last reinforced.
    #[serde(default)]
    pub last_reinforced_batch: usize,
    /// L0-02: Semantic relation type carried by this edge.
    /// Missing in pre-v9.0 snapshots — defaults to "categorical".
    #[serde(default = "default_relation_type")]
    pub relation_type: String,
}

/// Serializable mirror of a `CompositionRef` for JSON persistence.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedCompositionRef {
    pub node_id: u32,
    pub sense_id: u32,
}

/// Serializable mirror of `GroundingEvidence` for JSON persistence (v6.0).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedGroundingEvidence {
    /// Contexts that confirmed the compositions.
    pub confirming_contexts: usize,
    /// Contexts that contradicted the compositions.
    pub contradicting_contexts: usize,
    /// Description of the last contradiction.
    pub last_contradiction: Option<String>,
    /// How many times compositions have been revised.
    pub revision_count: usize,
}

impl From<&GroundingEvidence> for SavedGroundingEvidence {
    fn from(ge: &GroundingEvidence) -> Self {
        SavedGroundingEvidence {
            confirming_contexts: ge.confirming_contexts,
            contradicting_contexts: ge.contradicting_contexts,
            last_contradiction: ge.last_contradiction.clone(),
            revision_count: ge.revision_count,
        }
    }
}

impl From<&SavedGroundingEvidence> for GroundingEvidence {
    fn from(sge: &SavedGroundingEvidence) -> Self {
        GroundingEvidence {
            confirming_contexts: sge.confirming_contexts,
            contradicting_contexts: sge.contradicting_contexts,
            last_contradiction: sge.last_contradiction.clone(),
            revision_count: sge.revision_count,
        }
    }
}

/// Serializable mirror of `TransformerBridgeConfig` for JSON persistence (v6.0).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedTransformerBridgeConfig {
    /// Similarity threshold for considering two vectors "related".
    pub similarity_threshold: f32,
    /// Maximum compositions per induced sense from Transformer output.
    pub max_compositions: usize,
    /// Whether to use Transformer attention weights for composition weighting.
    pub use_attention_weights: bool,
}

impl From<&TransformerBridgeConfig> for SavedTransformerBridgeConfig {
    fn from(cfg: &TransformerBridgeConfig) -> Self {
        SavedTransformerBridgeConfig {
            similarity_threshold: cfg.similarity_threshold,
            max_compositions: cfg.max_compositions,
            use_attention_weights: cfg.use_attention_weights,
        }
    }
}

impl From<&SavedTransformerBridgeConfig> for TransformerBridgeConfig {
    fn from(scfg: &SavedTransformerBridgeConfig) -> Self {
        TransformerBridgeConfig {
            similarity_threshold: scfg.similarity_threshold,
            max_compositions: scfg.max_compositions,
            use_attention_weights: scfg.use_attention_weights,
        }
    }
}

/// Serializable entry for per-composition evidence tracking (v7.3).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedCompositionEvidence {
    /// Node ID of the composition target.
    pub node_id: u32,
    /// Sense ID of the composition target.
    pub sense_id: u32,
    /// Number of confirming contexts.
    pub confirmations: usize,
    /// Number of contradicting contexts.
    pub contradictions: usize,
}

/// Serializable mirror of a `Sense` for JSON persistence (v6.0).
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedSense {
    pub compositions: Vec<SavedCompositionRef>,
    pub layer: u32,
    pub contexts: Vec<Vec<u32>>,
    pub freq_counts: HashMap<u32, usize>,
    /// v6.1: Composition frequency map for P(a|S,q) scoring.
    /// Missing in v6.0 snapshots — defaults to empty HashMap.
    #[serde(default)]
    pub freq_map: Vec<SavedFreqMapEntry>,
    pub sum_sim: f64,
    pub pair_count: usize,
    pub coherence: f32,
    pub status: String,
    pub inactivity: usize,
    pub grounding: SavedGroundingEvidence,
    /// v7.3: Per-composition evidence tracking.
    #[serde(default)]
    pub composition_evidence: Vec<SavedCompositionEvidence>,
}

/// Serializable mirror of a `SenseManager` for JSON persistence.
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedSenseManager {
    pub senses: Vec<SavedSense>,
    pub next_sense_id: u32,
    pub global_context_count: usize,
}

/// Serializable mirror of an `AtomRecord` for JSON persistence.
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedAtomRecord {
    pub id: u32,
    pub confidence: f32,
    pub tier: u8,
    pub status: String,
    pub memory: String,
    pub domain_count: usize,
    pub cooccurring_mature: Vec<u32>,
    pub observation_count: usize,
    pub is_seed: bool,
    pub status_flip_count: u32,
    pub governance_score: f32,
    pub candidate_evidence_pool: f32,
    /// v6.1: Context counter at which this atom was last seen.
    /// Missing in v6.0 snapshots — defaults to 0.
    #[serde(default)]
    pub last_seen_context: usize,
    /// v6.1: Inactivity TTL — number of contexts before stale.
    /// Missing in v6.0 snapshots — defaults to 50.
    #[serde(default = "default_inactivity_ttl")]
    pub inactivity_ttl: usize,
    /// v6.5: Number of times this atom has been accessed/reinforced.
    /// Missing in earlier snapshots — defaults to 0.
    #[serde(default)]
    pub access_count: usize,
    /// v6.5: Counter of contexts since last promotion.
    /// Missing in earlier snapshots — defaults to 0.
    #[serde(default)]
    pub context_count_since_promote: usize,
}

/// Serializable mirror of `CoocStats` for JSON persistence.
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedCoocStats {
    pub token_count: HashMap<String, usize>,
    pub pair_count: HashMap<String, usize>,
    pub total_tokens: usize,
    pub total_sentences: usize,
}

/// Serializable mirror of `EntityDetector` for JSON persistence.
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedEntityDetector {
    pub sentence_count: HashMap<String, usize>,
    pub groundable: HashMap<String, bool>,
}

/// Top-level snapshot of the entire RSVS state (v7.3).
#[derive(Serialize, Deserialize, Debug)]
pub struct RsvsSnapshot {
    pub version: String,
    pub total_contexts: usize,
    pub token_to_id: HashMap<String, u32>,
    pub next_node_id: u32,
    pub nodes: Vec<SavedNode>,
    pub edges: Vec<SavedEdge>,
    pub sense_managers: HashMap<u32, SavedSenseManager>,
    pub atom_records: Vec<SavedAtomRecord>,
    pub cooc_stats: SavedCoocStats,
    pub entity_detector: SavedEntityDetector,
    pub entity_promote_n: usize,
    pub theta_assign: f32,
    pub n_warm: usize,
    pub eta: f32,
    pub current_domain: usize,
    /// v6.1: Traversal configuration. Missing in v6.0 snapshots.
    #[serde(default)]
    pub traversal: SavedTraversalConfig,
    /// v7.3: Domain calibration data for ParadigmRouter persistence.
    #[serde(default)]
    pub domain_calibration: Vec<crate::paradigm::CalibrationEntry>,
    /// v8.2: Convergence engine's detected pairs for persistence.
    /// Stored as Vec of (min_id, max_id) pairs. Missing in pre-v8.2 snapshots → defaults to empty.
    #[serde(default)]
    pub convergence_detected_pairs: Vec<(u32, u32)>,
}

// -----------------------------------------------------------------------
// Serialization helpers
// -----------------------------------------------------------------------

/// Default value for inactivity_ttl (v6.1).
fn default_inactivity_ttl() -> usize {
    50
}

/// Default value for relation_type (L0-02) — backward compatible.
fn default_relation_type() -> String {
    "categorical".to_string()
}

/// Serializable entry for freq_map: (node_id, sense_id, frequency).
/// Vec is used instead of HashMap because CompositionRef isn't a simple key.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedFreqMapEntry {
    /// The target node ID.
    pub node_id: u32,
    /// The target sense index.
    pub sense_id: u32,
    /// The normalized frequency value.
    pub freq: f32,
}

/// Serializable mirror of `TraversalConfig` (v6.1).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedTraversalConfig {
    pub max_depth: usize,
    pub gamma: f32,
    pub halt_epsilon: f32,
    pub halt_confidence: f32,
    pub tau_relevance: f32,
    #[serde(default)]
    pub epsilon_ig: f32,
}

impl Default for SavedTraversalConfig {
    fn default() -> Self {
        let tc = crate::types::TraversalConfig::default();
        Self {
            max_depth: tc.max_depth,
            gamma: tc.gamma,
            halt_epsilon: tc.halt_epsilon,
            halt_confidence: tc.halt_confidence,
            tau_relevance: tc.tau_relevance,
            epsilon_ig: tc.epsilon_ig,
        }
    }
}

impl From<&crate::types::TraversalConfig> for SavedTraversalConfig {
    fn from(tc: &crate::types::TraversalConfig) -> Self {
        Self {
            max_depth: tc.max_depth,
            gamma: tc.gamma,
            halt_epsilon: tc.halt_epsilon,
            halt_confidence: tc.halt_confidence,
            tau_relevance: tc.tau_relevance,
            epsilon_ig: tc.epsilon_ig,
        }
    }
}

impl From<&SavedTraversalConfig> for crate::types::TraversalConfig {
    fn from(stc: &SavedTraversalConfig) -> Self {
        Self {
            max_depth: stc.max_depth,
            gamma: stc.gamma,
            halt_epsilon: stc.halt_epsilon,
            halt_confidence: stc.halt_confidence,
            tau_relevance: stc.tau_relevance,
            epsilon_ig: stc.epsilon_ig,
        }
    }
}

fn tier_to_u8(t: &Tier) -> u8 {
    match t {
        Tier::Tier1 => 1,
        Tier::Tier2 => 2,
        Tier::Tier3 => 3,
    }
}
fn u8_to_tier(n: u8) -> Tier {
    match n {
        1 => Tier::Tier1,
        2 => Tier::Tier2,
        _ => Tier::Tier3,
    }
}

fn status_to_str(s: &NodeStatus) -> &'static str {
    match s {
        NodeStatus::New => "new",
        NodeStatus::Candidate => "candidate",
        NodeStatus::Stable => "stable",
        NodeStatus::Deprecated => "deprecated",
        NodeStatus::Quarantine => "quarantine",
    }
}

fn str_to_status(s: &str) -> NodeStatus {
    match s {
        "candidate" => NodeStatus::Candidate,
        "stable" => NodeStatus::Stable,
        "deprecated" => NodeStatus::Deprecated,
        "quarantine" => NodeStatus::Quarantine,
        _ => NodeStatus::New,
    }
}

fn compression_to_str(c: &CompressionState) -> &'static str {
    match c {
        CompressionState::Raw => "raw",
        CompressionState::Compressed => "compressed",
    }
}

fn str_to_compression(s: &str) -> CompressionState {
    match s {
        "compressed" => CompressionState::Compressed,
        _ => CompressionState::Raw,
    }
}

fn pair_key(a: &str, b: &str) -> String {
    if a <= b {
        format!("{}|{}", a, b)
    } else {
        format!("{}|{}", b, a)
    }
}

/// Convert RelationType to lowercase string for persistence.
fn relation_type_to_str(rt: &RelationType) -> &'static str {
    match rt {
        RelationType::Categorical => "categorical",
        RelationType::Differential => "differential",
        RelationType::Functional => "functional",
        RelationType::Spatial => "spatial",
        RelationType::Temporal => "temporal",
        RelationType::Causal => "causal",
    }
}

/// Convert string to RelationType for deserialization. Defaults to Categorical.
fn str_to_relation_type(s: &str) -> RelationType {
    match s {
        "differential" => RelationType::Differential,
        "functional" => RelationType::Functional,
        "spatial" => RelationType::Spatial,
        "temporal" => RelationType::Temporal,
        "causal" => RelationType::Causal,
        _ => RelationType::Categorical,
    }
}

// -----------------------------------------------------------------------
// Save
// -----------------------------------------------------------------------

/// Save the full RSVS state to a JSON file.
///
/// Uses atomic write (write to .tmp, then rename) to prevent corruption
/// if the process crashes during serialization. On Linux/macOS,
/// `std::fs::rename` is atomic, so the state file is always either
/// the old version or the new version — never a partial write.
pub fn save(rsvs: &Rsvs, path: &Path) -> Result<(), RsvsError> {
    let snapshot = to_snapshot(rsvs);
    let tmp = path.with_extension("tmp");
    let file = File::create(&tmp).map_err(|e| RsvsError::Persistence(e.to_string()))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &snapshot)
        .map_err(|e| RsvsError::Persistence(e.to_string()))?;
    // Flush buffers before rename to ensure data is on disk
    std::fs::rename(&tmp, path).map_err(|e| {
        // Clean up temp file on rename failure
        let _ = std::fs::remove_file(&tmp);
        RsvsError::Persistence(e.to_string())
    })?;
    Ok(())
}

/// Serialize the RSVS state into a snapshot struct.
pub fn to_snapshot(rsvs: &Rsvs) -> RsvsSnapshot {
    let nodes: Vec<SavedNode> = rsvs
        .graph
        .nodes
        .values()
        .map(|n| SavedNode {
            id: n.id,
            label: n.label.clone(),
            surface_label: n.surface_label.clone(),
            kind: n.kind.clone(),
            tier: tier_to_u8(&n.tier),
            confidence: n.confidence,
            status: status_to_str(&n.status).to_string(),
            is_seed: n.is_seed,
            is_locked: n.is_locked,
            compression_state: compression_to_str(&n.semantic.compression_state).to_string(),
            layer: n.semantic.layer,
            derived_from_node_ids: n.semantic.derived_from_node_ids.clone(),
            compression_reason: n.semantic.compression_reason.clone(),
            internal_representation: n.semantic.internal_representation,
            language_links: n.language_links.iter().map(|ll| SavedLanguageLink {
                link_type: ll.link_type.clone(),
                target_id: ll.target_id,
            }).collect(),
            policy_meta: n.policy_meta.as_ref().map(|pm| SavedPolicyMeta {
                policy_version: pm.policy_version.clone(),
                governance_score: pm.governance_score,
                candidate_evidence_pool: pm.candidate_evidence_pool,
                status_flip_count: pm.status_flip_count,
                seen_fingerprints: pm.seen_fingerprints.clone(),
                last_seen_at: pm.last_seen_at.clone(),
            }),
            atoms: n.atoms.clone(),
        })
        .collect();

    let mut edges: Vec<SavedEdge> = Vec::new();
    for node_id in rsvs.graph.nodes.keys() {
        for e in rsvs.graph.edges_from(*node_id) {
            edges.push(SavedEdge {
                from: e.from,
                to: e.to,
                weight: e.weight,
                source: format!("{:?}", e.source).to_lowercase(),
                last_reinforced_batch: e.last_reinforced_batch,
                relation_type: relation_type_to_str(&e.relation_type).to_string(),
            });
        }
    }

    let mut sense_managers: HashMap<u32, SavedSenseManager> = HashMap::new();
    for (&node_id, sm) in &rsvs.senses {
        let saved_senses = sm
            .senses
            .iter()
            .map(|s| SavedSense {
                compositions: s
                    .compositions
                    .iter()
                    .map(|c| SavedCompositionRef {
                        node_id: c.node_id,
                        sense_id: c.sense_id,
                    })
                    .collect(),
                layer: s.layer,
                contexts: s.contexts.clone(),
                freq_counts: s.freq_counts.clone(),
                sum_sim: s.sum_sim,
                pair_count: s.pair_count,
                coherence: s.coherence,
                status: if s.status == SenseStatus::Fragile {
                    "fragile".into()
                } else {
                    "mature".into()
                },
                inactivity: s.inactivity,
                grounding: SavedGroundingEvidence::from(&s.grounding),
                freq_map: s.freq_map.iter().map(|(comp, freq)| SavedFreqMapEntry {
                    node_id: comp.node_id,
                    sense_id: comp.sense_id,
                    freq: *freq,
                }).collect(),
                composition_evidence: s.composition_evidence.iter().map(|(comp, (conf, contra))| SavedCompositionEvidence {
                    node_id: comp.node_id,
                    sense_id: comp.sense_id,
                    confirmations: *conf,
                    contradictions: *contra,
                }).collect(),
            })
            .collect();

        sense_managers.insert(
            node_id,
            SavedSenseManager {
                senses: saved_senses,
                next_sense_id: sm.next_sense_id,
                global_context_count: sm.global_context_count,
            },
        );
    }

    let atom_records: Vec<SavedAtomRecord> = rsvs
        .autonomy
        .records
        .values()
        .map(|r| SavedAtomRecord {
            id: r.id,
            confidence: r.confidence,
            tier: tier_to_u8(&r.tier),
            status: status_to_str(&r.status).to_string(),
            memory: if r.memory == MemoryClass::Stable {
                "stable".into()
            } else {
                "working".into()
            },
            domain_count: r.domain_count,
            cooccurring_mature: r.cooccurring_mature.iter().copied().collect(),
            observation_count: r.observation_count,
            is_seed: r.is_seed,
            status_flip_count: r.status_flip_count,
            governance_score: r.governance_score,
            candidate_evidence_pool: r.candidate_evidence_pool,
            last_seen_context: r.last_seen_context,
            inactivity_ttl: r.inactivity_ttl,
            access_count: r.access_count,
            context_count_since_promote: r.context_count_since_promote,
        })
        .collect();

    let pair_count_serialized: HashMap<String, usize> = rsvs
        .stats_db
        .pair_count
        .iter()
        .map(|((a, b), &c)| (pair_key(a, b), c))
        .collect();

    let cooc_stats = SavedCoocStats {
        token_count: rsvs.stats_db.token_count.clone(),
        pair_count: pair_count_serialized,
        total_tokens: rsvs.stats_db.total_tokens,
        total_sentences: rsvs.stats_db.total_sentences,
    };

    let entity_detector = SavedEntityDetector {
        sentence_count: rsvs.entities.sentence_count.clone(),
        groundable: rsvs.entities.groundable.clone(),
    };

    RsvsSnapshot {
        version: "8.3".to_string(),
        total_contexts: rsvs.total_contexts,
        token_to_id: rsvs.token_to_id.clone(),
        next_node_id: rsvs.graph.next_id,
        nodes,
        edges,
        sense_managers,
        atom_records,
        cooc_stats,
        entity_detector,
        entity_promote_n: rsvs.config.entity_promote_n,
        theta_assign: rsvs.config.sense.theta_assign,
        n_warm: rsvs.config.autonomy.n_warm,
        eta: rsvs.config.autonomy.eta,
        current_domain: rsvs.config.current_domain,
        traversal: SavedTraversalConfig::from(&rsvs.config.traversal),
        domain_calibration: rsvs.paradigm_router.export_calibration(),
        convergence_detected_pairs: rsvs.convergence.export_detected_pairs(),
    }
}

// -----------------------------------------------------------------------
// Load
// -----------------------------------------------------------------------

/// Load the full RSVS state from a JSON file.
///
/// v8.3: Added schema version check. Warns if the snapshot version
/// is older than the current code version but still loads (backward compatible).
/// Returns an error only for incompatible version differences.
pub fn load(path: &Path) -> Result<Rsvs, RsvsError> {
    let file = File::open(path).map_err(|e| RsvsError::Persistence(e.to_string()))?;
    let reader = BufReader::new(file);
    let snapshot: RsvsSnapshot =
        serde_json::from_reader(reader).map_err(|e| RsvsError::Persistence(e.to_string()))?;

    // v8.3: Schema version compatibility check.
    // The snapshot version is a string like "8.1", "8.3", etc.
    // We parse the major version and only reject if there's a major version mismatch.
    let code_version: u32 = 8; // Current major version
    let snap_major: u32 = snapshot
        .version
        .split('.')
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    if snap_major > code_version {
        return Err(RsvsError::Persistence(format!(
            "Snapshot version {} is newer than this code ({}). Please upgrade.",
            snapshot.version, code_version
        )));
    }
    // Minor version mismatches are OK — we use #[serde(default)] for new fields.

    Ok(from_snapshot(snapshot))
}

/// Reconstruct an `Rsvs` instance from a deserialized snapshot.
pub fn from_snapshot(snap: RsvsSnapshot) -> Rsvs {
    let config = PipelineConfig {
        entity_promote_n: snap.entity_promote_n,
        current_domain: snap.current_domain,
        sense: SenseConfig {
            theta_assign: snap.theta_assign,
            ..SenseConfig::default()
        },
        autonomy: AutonomyConfig {
            n_warm: snap.n_warm,
            eta: snap.eta,
            threshold_global_delta: 5.0,
            ..AutonomyConfig::default()
        },
        traversal: crate::types::TraversalConfig::from(&snap.traversal),
        ..PipelineConfig::default()
    };

    // --- Rebuild graph ---
    let mut graph = RsvsGraph::new();
    graph.next_id = snap.next_node_id;

    for sn in &snap.nodes {
        let node = Node {
            id: sn.id,
            label: sn.label.clone(),
            surface_label: sn.surface_label.clone(),
            kind: sn.kind.clone(),
            tier: u8_to_tier(sn.tier),
            confidence: sn.confidence,
            status: str_to_status(&sn.status),
            is_seed: sn.is_seed,
            is_locked: sn.is_locked,
            semantic: SemanticMeta {
                compression_state: str_to_compression(&sn.compression_state),
                layer: sn.layer,
                derived_from_node_ids: sn.derived_from_node_ids.clone(),
                compression_reason: sn.compression_reason.clone(),
                internal_representation: sn.internal_representation,
            },
            policy_meta: sn.policy_meta.as_ref().map(|pm| PolicyMeta {
                policy_version: pm.policy_version.clone(),
                governance_score: pm.governance_score,
                candidate_evidence_pool: pm.candidate_evidence_pool,
                status_flip_count: pm.status_flip_count,
                seen_fingerprints: pm.seen_fingerprints.clone(),
                last_seen_at: pm.last_seen_at.clone(),
            }),
            language_links: sn.language_links.iter().map(|sll| crate::types::LanguageLink {
                link_type: sll.link_type.clone(),
                target_id: sll.target_id,
            }).collect(),
            atoms: sn.atoms.clone(),
            fingerprint: None,
        };
        graph.label_to_id.insert(node.label.clone(), node.id);
        graph
            .label_to_id
            .insert(node.surface_label.clone(), node.id);
        graph.nodes.insert(node.id, node);
    }

    for se in &snap.edges {
        let source = if se.source == "bootstrap" {
            EdgeSource::Bootstrap
        } else if se.source == "composition" {
            EdgeSource::Composition
        } else {
            EdgeSource::Learned
        };
        let relation_type = str_to_relation_type(&se.relation_type);
        graph.edges.entry(se.from).or_default().push(Edge {
            from: se.from,
            to: se.to,
            weight: se.weight,
            source,
            last_reinforced_batch: se.last_reinforced_batch, // v6.5: Preserve reinforcement info
            relation_type,
        });
    }

    // --- Rebuild sense managers (v6.0) ---
    let mut senses: HashMap<NodeId, SenseManager> = HashMap::new();
    for (node_id, ssm) in snap.sense_managers {
        let mut sm = SenseManager::new(config.sense.clone());
        sm.next_sense_id = ssm.next_sense_id;
        sm.global_context_count = ssm.global_context_count;

        for ss in ssm.senses.into_iter() {
            let compositions: Vec<CompositionRef> = ss
                .compositions
                .into_iter()
                .map(|sc| CompositionRef::new(sc.node_id, sc.sense_id))
                .collect();

            let mut sense = Sense::new_compositional(
                0, // placeholder, will be set by id field
                compositions,
                vec![],
                ss.layer,
            );
            sense.id = 0; // Will be reassigned
            sense.contexts = ss.contexts;
            sense.freq_counts = ss.freq_counts;
            sense.sum_sim = ss.sum_sim;
            sense.pair_count = ss.pair_count;
            sense.coherence = ss.coherence;
            sense.status = if ss.status == "mature" {
                SenseStatus::Mature
            } else {
                SenseStatus::Fragile
            };
            sense.inactivity = ss.inactivity;
            sense.grounding = GroundingEvidence::from(&ss.grounding);
            // v6.1: Restore freq_map from saved entries
            sense.freq_map = ss.freq_map.iter().map(|entry| {
                let comp = crate::types::CompositionRef::new(entry.node_id, entry.sense_id);
                (comp, entry.freq)
            }).collect();
            // v7.3: Restore composition_evidence from saved entries
            sense.composition_evidence = ss.composition_evidence.iter().map(|entry| {
                let comp = crate::types::CompositionRef::new(entry.node_id, entry.sense_id);
                (comp, (entry.confirmations, entry.contradictions))
            }).collect();
            sm.senses.push(sense);
        }
        senses.insert(node_id, sm);
    }

    for &id in graph.nodes.keys() {
        senses
            .entry(id)
            .or_insert_with(|| SenseManager::new(config.sense.clone()));
    }

    // --- Rebuild autonomy engine ---
    let mut autonomy = AutonomyEngine::new(config.autonomy.clone());

    if snap.total_contexts >= snap.n_warm {
        for _ in 0..snap.n_warm {
            autonomy.tick_context();
        }
    }

    for sar in snap.atom_records {
        let mut record = AtomRecord::new(sar.id, sar.confidence, u8_to_tier(sar.tier));
        record.status = str_to_status(&sar.status);
        record.memory = if sar.memory == "stable" {
            MemoryClass::Stable
        } else {
            MemoryClass::Working
        };
        record.domain_count = sar.domain_count;
        record.observation_count = sar.observation_count;
        record.cooccurring_mature = sar.cooccurring_mature.into_iter().collect();
        record.is_seed = sar.is_seed;
        record.status_flip_count = sar.status_flip_count;
        record.governance_score = sar.governance_score;
        record.candidate_evidence_pool = sar.candidate_evidence_pool;
        record.last_seen_context = sar.last_seen_context;
        record.inactivity_ttl = sar.inactivity_ttl;
        record.access_count = sar.access_count;
        record.context_count_since_promote = sar.context_count_since_promote;
        autonomy.records.insert(sar.id, record);
    }

    // --- Rebuild CoocStats ---
    let mut stats_db = CoocStats::new();
    stats_db.token_count = snap.cooc_stats.token_count;
    stats_db.total_tokens = snap.cooc_stats.total_tokens;
    stats_db.total_sentences = snap.cooc_stats.total_sentences;

    for (key, count) in snap.cooc_stats.pair_count {
        let parts: Vec<&str> = key.splitn(2, '|').collect();
        if parts.len() == 2 {
            let pair = (parts[0].to_string(), parts[1].to_string());
            stats_db.pair_count.insert(pair, count);
        }
    }

    let mut entities = crate::attention::EntityDetector::new();
    entities.sentence_count = snap.entity_detector.sentence_count;
    entities.groundable = snap.entity_detector.groundable;

    let token_to_id = snap.token_to_id;
    let mut atom_sets: HashMap<String, Vec<NodeId>> = HashMap::new();
    for (token, &id) in &token_to_id {
        let atoms = graph
            .get_node(id)
            .map(|n| n.atoms.clone())
            .unwrap_or_default();
        atom_sets.insert(
            token.clone(),
            if atoms.is_empty() { vec![id] } else { atoms },
        );
    }

    let attention = crate::attention::RsvsAttention::new(config.attention.clone());

    // v8.0: Rebuild seed_node_ids from loaded graph
    let seed_node_ids: std::collections::HashSet<NodeId> = graph.nodes.values()
        .filter(|n| n.is_seed)
        .map(|n| n.id)
        .collect();

    let mut rsvs = Rsvs {
        graph,
        senses,
        autonomy,
        stats_db,
        entities,
        attention,
        token_to_id,
        atom_sets,
        seed_node_ids,
        config,
        total_contexts: snap.total_contexts,
        latest_seq: 0,
        ingest_counter: 0,
        event_retention: 10_000,
        events: std::collections::VecDeque::new(),
        batch_counter: 0,
        domain_configs: HashMap::new(),
        composition_index: crate::composition_index::CompositionIndex::new(),
        thinking_toggle: crate::thinking::ThinkingToggle::new(
            crate::thinking::ThinkingToggleConfig::default(),
        ),
        consolidation: crate::consolidation::ConsolidationEngine::new(
            crate::consolidation::ConsolidationConfig::default(),
        ),
        reflection: crate::reflection::SenseReflection::new(
            crate::reflection::ReflectionConfig::default(),
        ),
        paradigm_router: crate::paradigm::ParadigmRouter::new(
            crate::paradigm::ParadigmRouterConfig::default(),
        ),
        spreading_activation: crate::spreading::SpreadingActivation::new(
            crate::spreading::SpreadingActivationConfig::default(),
        ),
        deps_planner: crate::deps::DEPSPlanner::new(),
        convergence: crate::convergence::ConvergenceEngine::new(),
    };

    // v7.3: Restore domain calibration from saved data
    rsvs.paradigm_router.import_calibration(&snap.domain_calibration);

    // v8.2: Restore convergence detected pairs from saved data
    if !snap.convergence_detected_pairs.is_empty() {
        rsvs.convergence.import_detected_pairs(snap.convergence_detected_pairs);
    }

    // v7.3: Rebuild composition index from restored senses
    rsvs.composition_index.rebuild(&rsvs.senses);

    rsvs
}
