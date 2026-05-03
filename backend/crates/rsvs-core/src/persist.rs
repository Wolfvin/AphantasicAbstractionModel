//! Persistence — RSVS v0.7
//!
//! Serialize/deserialize the full RSVS state to/from disk.
//! Format: JSON (human-readable, debuggable) with optional
//! binary fallback via MessagePack-compatible layout.
//!
//! What gets saved:
//!   - All graph nodes (id, kind, atoms, confidence, tier, label)
//!   - All graph edges (from, to, weight, source)
//!   - All sense managers (per-ID: contexts, freq_map, coherence stats)
//!   - All autonomy records (confidence, tier, memory, domain_count, etc.)
//!   - CoocStats (token_count, pair_count, totals)
//!   - EntityDetector (sentence_count, groundable)
//!   - Pipeline config
//!   - total_contexts, token_to_id
//!
//! What does NOT get saved (re-derived on load):
//!   - atom_sets (re-derived from token_to_id + graph)
//!   - Adaptive threshold history (starts fresh, fallback used)

use std::collections::HashMap;
use std::path::Path;
use std::io::{self, BufReader, BufWriter};
use std::fs::File;

use serde::{Serialize, Deserialize};

use crate::types::{NodeId, NodeKind, Tier};
use crate::graph::RsvsGraph;
use crate::sense::{SenseManager, SenseConfig, Sense, SenseStatus};
use crate::autonomy::{AutonomyEngine, AutonomyConfig, MemoryClass};
use crate::attention::CoocStats;
use crate::pipeline::{Rsvs, PipelineConfig};

// -----------------------------------------------------------------------
// Serializable mirror types (serde-friendly)
// -----------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedNode {
    pub id:         u32,
    pub kind:       String,    // "atom" | "composite"
    pub atoms:      Vec<u32>,
    pub confidence: f32,
    pub tier:       u8,        // 1/2/3
    pub label:      Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SavedEdge {
    pub from:   u32,
    pub to:     u32,
    pub weight: f32,
    pub source: String, // "bootstrap" | "learned"
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SavedSense {
    pub contexts:   Vec<Vec<u32>>,
    pub freq_counts: HashMap<u32, usize>,
    pub sum_sim:    f64,
    pub pair_count: usize,
    pub coherence:  f32,
    pub status:     String, // "fragile" | "mature"
    pub inactivity: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SavedSenseManager {
    pub senses:        Vec<SavedSense>,
    pub next_sense_id: usize,
    pub global_context_count: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SavedAtomRecord {
    pub id:                  u32,
    pub confidence:          f32,
    pub tier:                u8,
    pub memory:              String, // "stable" | "working"
    pub domain_count:        usize,
    pub cooccurring_mature:  Vec<u32>,
    pub observation_count:   usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SavedCoocStats {
    pub token_count:     HashMap<String, usize>,
    pub pair_count:      HashMap<String, usize>, // "a|b" → count
    pub total_tokens:    usize,
    pub total_sentences: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SavedEntityDetector {
    pub sentence_count: HashMap<String, usize>,
    pub groundable:     HashMap<String, bool>,
}

/// Top-level snapshot of the entire RSVS state.
#[derive(Serialize, Deserialize, Debug)]
pub struct RsvsSnapshot {
    pub version:        String,
    pub total_contexts: usize,
    pub token_to_id:    HashMap<String, u32>,
    pub next_node_id:   u32,
    pub nodes:          Vec<SavedNode>,
    pub edges:          Vec<SavedEdge>,
    pub sense_managers: HashMap<u32, SavedSenseManager>, // node_id → manager
    pub atom_records:   Vec<SavedAtomRecord>,
    pub cooc_stats:     SavedCoocStats,
    pub entity_detector: SavedEntityDetector,
    // Config snapshots (for reference — not re-applied on load by default)
    pub entity_promote_n: usize,
    pub theta_assign:     f32,
    pub n_warm:           usize,
    pub eta:              f32,
    pub current_domain:   usize,
}

// -----------------------------------------------------------------------
// Serialization helpers
// -----------------------------------------------------------------------

fn tier_to_u8(t: &Tier) -> u8 {
    match t { Tier::Tier1 => 1, Tier::Tier2 => 2, Tier::Tier3 => 3 }
}
fn u8_to_tier(n: u8) -> Tier {
    match n { 1 => Tier::Tier1, 2 => Tier::Tier2, _ => Tier::Tier3 }
}
fn kind_to_str(k: &NodeKind) -> &'static str {
    match k { NodeKind::Atom => "atom", NodeKind::Composite => "composite" }
}
fn str_to_kind(s: &str) -> NodeKind {
    if s == "composite" { NodeKind::Composite } else { NodeKind::Atom }
}

// Pair key: always "min|max" for deterministic order
fn pair_key(a: &str, b: &str) -> String {
    if a <= b { format!("{}|{}", a, b) } else { format!("{}|{}", b, a) }
}

// -----------------------------------------------------------------------
// Save
// -----------------------------------------------------------------------

/// Serialize the full Rsvs state to a JSON file.
pub fn save(rsvs: &Rsvs, path: &Path) -> io::Result<()> {
    let snapshot = to_snapshot(rsvs);
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &snapshot)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(())
}

pub fn to_snapshot(rsvs: &Rsvs) -> RsvsSnapshot {
    // --- Nodes ---
    let nodes: Vec<SavedNode> = rsvs.graph.nodes.values().map(|n| SavedNode {
        id:         n.id,
        kind:       kind_to_str(&n.kind).to_string(),
        atoms:      n.atoms.clone(),
        confidence: n.confidence,
        tier:       tier_to_u8(&n.tier),
        label:      n.label.clone(),
    }).collect();

    // --- Edges ---
    let mut edges: Vec<SavedEdge> = Vec::new();
    for atom_id in rsvs.graph.nodes.keys() {
        for e in rsvs.graph.edges_from(*atom_id) {
            edges.push(SavedEdge {
                from:   e.from,
                to:     e.to,
                weight: e.weight,
                source: format!("{:?}", e.source).to_lowercase(),
            });
        }
    }

    // --- Sense managers ---
    let mut sense_managers: HashMap<u32, SavedSenseManager> = HashMap::new();
    for (&node_id, sm) in &rsvs.senses {
        let saved_senses = sm.senses.iter().map(|s| SavedSense {
            contexts:    s.contexts.clone(),
            freq_counts: s.freq_counts.clone(),
            sum_sim:     s.sum_sim,
            pair_count:  s.pair_count,
            coherence:   s.coherence,
            status:      if s.status == SenseStatus::Fragile { "fragile".into() }
                         else { "mature".into() },
            inactivity:  s.inactivity,
        }).collect();

        sense_managers.insert(node_id, SavedSenseManager {
            senses:               saved_senses,
            next_sense_id:        sm.next_sense_id,
            global_context_count: sm.global_context_count,
        });
    }

    // --- Atom records ---
    let atom_records: Vec<SavedAtomRecord> = rsvs.autonomy.records.values()
        .map(|r| SavedAtomRecord {
            id:                 r.id,
            confidence:         r.confidence,
            tier:               tier_to_u8(&r.tier),
            memory:             if r.memory == MemoryClass::Stable { "stable".into() }
                                else { "working".into() },
            domain_count:       r.domain_count,
            cooccurring_mature: r.cooccurring_mature.iter().copied().collect(),
            observation_count:  r.observation_count,
        })
        .collect();

    // --- CoocStats ---
    let pair_count_serialized: HashMap<String, usize> = rsvs.stats_db.pair_count
        .iter()
        .map(|((a, b), &c)| (pair_key(a, b), c))
        .collect();

    let cooc_stats = SavedCoocStats {
        token_count:     rsvs.stats_db.token_count.clone(),
        pair_count:      pair_count_serialized,
        total_tokens:    rsvs.stats_db.total_tokens,
        total_sentences: rsvs.stats_db.total_sentences,
    };

    // --- EntityDetector ---
    let entity_detector = SavedEntityDetector {
        sentence_count: rsvs.entities.sentence_count.clone(),
        groundable:     rsvs.entities.groundable.clone(),
    };

    RsvsSnapshot {
        version:          "0.7".to_string(),
        total_contexts:   rsvs.total_contexts,
        token_to_id:      rsvs.token_to_id.clone(),
        next_node_id:     rsvs.graph.next_id,
        nodes,
        edges,
        sense_managers,
        atom_records,
        cooc_stats,
        entity_detector,
        entity_promote_n: rsvs.config.entity_promote_n,
        theta_assign:     rsvs.config.sense.theta_assign,
        n_warm:           rsvs.config.autonomy.n_warm,
        eta:              rsvs.config.autonomy.eta,
        current_domain:   rsvs.config.current_domain,
    }
}

// -----------------------------------------------------------------------
// Load
// -----------------------------------------------------------------------

/// Deserialize RSVS state from a JSON file.
pub fn load(path: &Path) -> io::Result<Rsvs> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let snapshot: RsvsSnapshot = serde_json::from_reader(reader)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(from_snapshot(snapshot))
}

pub fn from_snapshot(snap: RsvsSnapshot) -> Rsvs {
    use crate::types::{Node, Edge, EdgeSource};
    use crate::autonomy::AtomRecord;

    // Rebuild config from snapshot
    let config = PipelineConfig {
        entity_promote_n: snap.entity_promote_n,
        current_domain:   snap.current_domain,
        sense: SenseConfig {
            theta_assign: snap.theta_assign,
            ..SenseConfig::default()
        },
        autonomy: AutonomyConfig {
            n_warm: snap.n_warm,
            eta:    snap.eta,
            threshold_global_delta: 5.0,
            ..AutonomyConfig::default()
        },
        ..PipelineConfig::default()
    };

    // --- Rebuild graph ---
    let mut graph = RsvsGraph::new();
    graph.next_id = snap.next_node_id;

    for sn in &snap.nodes {
        let node = Node {
            id:          sn.id,
            kind:        str_to_kind(&sn.kind),
            atoms:       sn.atoms.clone(),
            confidence:  sn.confidence,
            tier:        u8_to_tier(sn.tier),
            label:       sn.label.clone(),
            fingerprint: None,
        };
        // Insert directly into map (bypass DAG checks — we trust the snapshot)
        if let Some(label) = &node.label {
            graph.label_to_id.insert(label.clone(), node.id);
        }
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
            from: se.from, to: se.to, weight: se.weight, source,
        });
    }

    // --- Rebuild sense managers ---
    let mut senses: HashMap<NodeId, SenseManager> = HashMap::new();
    for (node_id, ssm) in snap.sense_managers {
        let mut sm = SenseManager::new(config.sense.clone());
        sm.next_sense_id        = ssm.next_sense_id;
        sm.global_context_count = ssm.global_context_count;

        for (i, ss) in ssm.senses.into_iter().enumerate() {
            let mut sense = Sense::new(i, vec![]); // placeholder
            sense.contexts    = ss.contexts;
            sense.freq_counts = ss.freq_counts;
            sense.sum_sim     = ss.sum_sim;
            sense.pair_count  = ss.pair_count;
            sense.coherence   = ss.coherence;
            sense.status      = if ss.status == "mature" {
                SenseStatus::Mature
            } else {
                SenseStatus::Fragile
            };
            sense.inactivity  = ss.inactivity;
            sm.senses.push(sense);
        }
        senses.insert(node_id, sm);
    }

    // Ensure every node has a SenseManager
    for &id in graph.nodes.keys() {
        senses.entry(id).or_insert_with(|| SenseManager::new(config.sense.clone()));
    }

    // --- Rebuild autonomy engine ---
    let mut autonomy = AutonomyEngine::new(config.autonomy.clone());

    // Restore warm-up state: if contexts > n_warm, mark complete
    if snap.total_contexts >= snap.n_warm {
        for _ in 0..snap.n_warm {
            autonomy.tick_context();
        }
    }

    for sar in snap.atom_records {
        let mut record = AtomRecord::new(sar.id, sar.confidence, u8_to_tier(sar.tier));
        record.memory = if sar.memory == "stable" {
            MemoryClass::Stable
        } else {
            MemoryClass::Working
        };
        record.domain_count       = sar.domain_count;
        record.observation_count  = sar.observation_count;
        record.cooccurring_mature = sar.cooccurring_mature.into_iter().collect();
        autonomy.records.insert(sar.id, record);
    }

    // --- Rebuild CoocStats ---
    let mut stats_db = CoocStats::new();
    stats_db.token_count     = snap.cooc_stats.token_count;
    stats_db.total_tokens    = snap.cooc_stats.total_tokens;
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
    entities.groundable     = snap.entity_detector.groundable;

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
