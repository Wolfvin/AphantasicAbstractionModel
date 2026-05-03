//! Persistence — RSVS v4.2
//!
//! Serialize/deserialize the full RSVS state to/from disk.
//! Format: JSON (human-readable, debuggable).
//!
//! v4.2: Updated for unified node model, NodeStatus, CompressionState,
//! SemanticMeta, PolicyMeta. No more NodeKind.

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
use crate::sense::{Sense, SenseConfig, SenseManager, SenseStatus};
use crate::types::{
    CompressionState, Edge, EdgeSource, Node, NodeId, NodeStatus, PolicyMeta, SemanticMeta, Tier,
};

// -----------------------------------------------------------------------
// Serializable mirror types (v4.2 serde-friendly)
// -----------------------------------------------------------------------

/// Serializable mirror of a `Node` for JSON persistence.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedNode {
    /// Unique integer node ID.
    pub id: u32,
    /// Canonical label string.
    pub label: String,
    /// Surface form with language tag.
    pub surface_label: String,
    /// Node kind — always "node" in v4.2.
    pub kind: String,
    /// Tier number (1, 2, or 3).
    pub tier: u8,
    /// Confidence score.
    pub confidence: f32,
    /// Lifecycle status as string.
    pub status: String,
    /// Whether this is a seed node.
    pub is_seed: bool,
    /// Whether this node is locked.
    pub is_locked: bool,
    /// Compression state ("raw" or "compressed").
    pub compression_state: String,
    /// Node IDs this node was derived from.
    pub derived_from_node_ids: Vec<u32>,
    /// Reason for compression, if any.
    pub compression_reason: Option<String>,
    /// Policy metadata, if present.
    pub policy_meta: Option<SavedPolicyMeta>,
    /// Atom set for similarity/attention.
    pub atoms: Vec<u32>,
}

/// Serializable mirror of `PolicyMeta` for JSON persistence.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedPolicyMeta {
    /// Policy engine version.
    pub policy_version: String,
    /// Governance score (0.0–1.0).
    pub governance_score: f32,
    /// Accumulated candidate evidence pool.
    pub candidate_evidence_pool: f32,
    /// Number of status flip-flops detected.
    pub status_flip_count: u32,
    /// Content fingerprints already seen for dedup.
    pub seen_fingerprints: Vec<String>,
    /// ISO timestamp of last observation.
    pub last_seen_at: Option<String>,
}

/// Serializable mirror of an `Edge` for JSON persistence.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedEdge {
    /// Source node ID.
    pub from: u32,
    /// Target node ID.
    pub to: u32,
    /// Edge weight.
    pub weight: f32,
    /// Edge source ("bootstrap" or "learned").
    pub source: String,
}

/// Serializable mirror of a `Sense` for JSON persistence.
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedSense {
    /// Context atom sets assigned to this sense.
    pub contexts: Vec<Vec<u32>>,
    /// Frequency counts per atom.
    pub freq_counts: HashMap<u32, usize>,
    /// Sum of pairwise similarities.
    pub sum_sim: f64,
    /// Number of pairs used in coherence.
    pub pair_count: usize,
    /// Cached coherence value.
    pub coherence: f32,
    /// Sense status ("fragile" or "mature").
    pub status: String,
    /// Contexts of inactivity since last assignment.
    pub inactivity: usize,
}

/// Serializable mirror of a `SenseManager` for JSON persistence.
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedSenseManager {
    /// The sense clusters.
    pub senses: Vec<SavedSense>,
    /// Next sense ID to allocate.
    pub next_sense_id: usize,
    /// Global context counter.
    pub global_context_count: usize,
}

/// Serializable mirror of an `AtomRecord` for JSON persistence.
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedAtomRecord {
    /// Node ID.
    pub id: u32,
    /// Confidence score.
    pub confidence: f32,
    /// Tier number.
    pub tier: u8,
    /// Status as string.
    pub status: String,
    /// Memory class ("stable" or "working").
    pub memory: String,
    /// Number of domains observed.
    pub domain_count: usize,
    /// Co-occurring mature node IDs.
    pub cooccurring_mature: Vec<u32>,
    /// Number of observations.
    pub observation_count: usize,
    /// Whether this is a seed node.
    pub is_seed: bool,
    /// Number of status flip-flops.
    pub status_flip_count: u32,
    /// Governance score.
    pub governance_score: f32,
    /// Candidate evidence pool.
    pub candidate_evidence_pool: f32,
}

/// Serializable mirror of `CoocStats` for JSON persistence.
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedCoocStats {
    /// Per-token count.
    pub token_count: HashMap<String, usize>,
    /// Pair count stored as "a|b" → count.
    pub pair_count: HashMap<String, usize>,
    /// Total tokens seen.
    pub total_tokens: usize,
    /// Total sentences seen.
    pub total_sentences: usize,
}

/// Serializable mirror of `EntityDetector` for JSON persistence.
#[derive(Serialize, Deserialize, Debug)]
pub struct SavedEntityDetector {
    /// Per-token sentence count.
    pub sentence_count: HashMap<String, usize>,
    /// Per-token groundable flag.
    pub groundable: HashMap<String, bool>,
}

/// Top-level snapshot of the entire RSVS state (v4.2).
#[derive(Serialize, Deserialize, Debug)]
pub struct RsvsSnapshot {
    /// Snapshot format version.
    pub version: String,
    /// Total contexts processed.
    pub total_contexts: usize,
    /// Token string to node ID mapping.
    pub token_to_id: HashMap<String, u32>,
    /// Next node ID to allocate.
    pub next_node_id: u32,
    /// All saved nodes.
    pub nodes: Vec<SavedNode>,
    /// All saved edges.
    pub edges: Vec<SavedEdge>,
    /// Per-node sense managers.
    pub sense_managers: HashMap<u32, SavedSenseManager>,
    /// All atom records.
    pub atom_records: Vec<SavedAtomRecord>,
    /// Co-occurrence statistics.
    pub cooc_stats: SavedCoocStats,
    /// Entity detector state.
    pub entity_detector: SavedEntityDetector,
    /// Entity promotion threshold.
    pub entity_promote_n: usize,
    /// Sense assignment threshold.
    pub theta_assign: f32,
    /// Number of warm-up contexts.
    pub n_warm: usize,
    /// EMA smoothing factor.
    pub eta: f32,
    /// Current domain tag.
    pub current_domain: usize,
}

// -----------------------------------------------------------------------
// Serialization helpers
// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------
// Save
// -----------------------------------------------------------------------

/// Save the full RSVS state to a JSON file.
///
/// # Examples
/// ```ignore
/// use rsvs::persist;
/// persist::save(&rsvs, Path::new("rsvs-state.json"))?;
/// ```
pub fn save(rsvs: &Rsvs, path: &Path) -> Result<(), RsvsError> {
    let snapshot = to_snapshot(rsvs);
    let file = File::create(path).map_err(|e| RsvsError::Persistence(e.to_string()))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &snapshot)
        .map_err(|e| RsvsError::Persistence(e.to_string()))?;
    Ok(())
}

/// Serialize the RSVS state into a snapshot struct (for programmatic use).
pub fn to_snapshot(rsvs: &Rsvs) -> RsvsSnapshot {
    // --- Nodes (v4.2) ---
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
            derived_from_node_ids: n.semantic.derived_from_node_ids.clone(),
            compression_reason: n.semantic.compression_reason.clone(),
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

    // --- Edges ---
    let mut edges: Vec<SavedEdge> = Vec::new();
    for node_id in rsvs.graph.nodes.keys() {
        for e in rsvs.graph.edges_from(*node_id) {
            edges.push(SavedEdge {
                from: e.from,
                to: e.to,
                weight: e.weight,
                source: format!("{:?}", e.source).to_lowercase(),
            });
        }
    }

    // --- Sense managers ---
    let mut sense_managers: HashMap<u32, SavedSenseManager> = HashMap::new();
    for (&node_id, sm) in &rsvs.senses {
        let saved_senses = sm
            .senses
            .iter()
            .map(|s| SavedSense {
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

    // --- Atom records (v4.2) ---
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
        })
        .collect();

    // --- CoocStats ---
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
        version: "4.2".to_string(),
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
    }
}

// -----------------------------------------------------------------------
// Load
// -----------------------------------------------------------------------

/// Load the full RSVS state from a JSON file.
///
/// # Examples
/// ```ignore
/// use rsvs::persist;
/// let rsvs = persist::load(Path::new("rsvs-state.json"))?;
/// ```
pub fn load(path: &Path) -> Result<Rsvs, RsvsError> {
    let file = File::open(path).map_err(|e| RsvsError::Persistence(e.to_string()))?;
    let reader = BufReader::new(file);
    let snapshot: RsvsSnapshot =
        serde_json::from_reader(reader).map_err(|e| RsvsError::Persistence(e.to_string()))?;
    Ok(from_snapshot(snapshot))
}

/// Reconstruct an `Rsvs` instance from a deserialized snapshot.
pub fn from_snapshot(snap: RsvsSnapshot) -> Rsvs {
    // Rebuild config from snapshot
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
        ..PipelineConfig::default()
    };

    // --- Rebuild graph (v4.2) ---
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
                derived_from_node_ids: sn.derived_from_node_ids.clone(),
                compression_reason: sn.compression_reason.clone(),
            },
            policy_meta: sn.policy_meta.as_ref().map(|pm| PolicyMeta {
                policy_version: pm.policy_version.clone(),
                governance_score: pm.governance_score,
                candidate_evidence_pool: pm.candidate_evidence_pool,
                status_flip_count: pm.status_flip_count,
                seen_fingerprints: pm.seen_fingerprints.clone(),
                last_seen_at: pm.last_seen_at.clone(),
            }),
            language_links: vec![],
            atoms: sn.atoms.clone(),
            fingerprint: None,
        };
        graph.label_to_id.insert(node.label.clone(), node.id);
        graph
            .label_to_id
            .insert(node.surface_label.clone(), node.id);
        graph.nodes.insert(node.id, node);
    }

    // Restore edges
    for se in &snap.edges {
        let source = if se.source == "bootstrap" {
            EdgeSource::Bootstrap
        } else {
            EdgeSource::Learned
        };
        graph.edges.entry(se.from).or_default().push(Edge {
            from: se.from,
            to: se.to,
            weight: se.weight,
            source,
        });
    }

    // --- Rebuild sense managers ---
    let mut senses: HashMap<NodeId, SenseManager> = HashMap::new();
    for (node_id, ssm) in snap.sense_managers {
        let mut sm = SenseManager::new(config.sense.clone());
        sm.next_sense_id = ssm.next_sense_id;
        sm.global_context_count = ssm.global_context_count;

        for (i, ss) in ssm.senses.into_iter().enumerate() {
            let mut sense = Sense::new(i, vec![]);
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
            sm.senses.push(sense);
        }
        senses.insert(node_id, sm);
    }

    for &id in graph.nodes.keys() {
        senses
            .entry(id)
            .or_insert_with(|| SenseManager::new(config.sense.clone()));
    }

    // --- Rebuild autonomy engine (v4.2) ---
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

    // --- Rebuild EntityDetector ---
    let mut entities = crate::attention::EntityDetector::new();
    entities.sentence_count = snap.entity_detector.sentence_count;
    entities.groundable = snap.entity_detector.groundable;

    // --- Rebuild token_to_id and atom_sets ---
    let token_to_id = snap.token_to_id;
    let mut atom_sets: HashMap<String, Vec<NodeId>> = HashMap::new();
    for (token, &id) in &token_to_id {
        atom_sets.insert(token.clone(), vec![id]);
    }

    let attention = crate::attention::RsvsAttention::new(config.attention.clone());

    Rsvs {
        graph,
        senses,
        autonomy,
        stats_db,
        entities,
        attention,
        token_to_id,
        atom_sets,
        config,
        total_contexts: snap.total_contexts,
        latest_seq: 0,
        ingest_counter: 0,
        event_retention: 10_000,
        events: std::collections::VecDeque::new(),
    }
}
