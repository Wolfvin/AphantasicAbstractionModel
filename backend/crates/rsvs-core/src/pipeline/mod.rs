//! End-to-end pipeline — RSVS v4.2
//!
//! Wires all modules together:
//!   text → CoocStats → EntityDetector → node promotion
//!   → SenseManager ingest → AutonomyEngine update → graph query
//!
//! v4.2: Unified node model, status lifecycle, policy engine.
//!   - ingest mode: produce v4.2 nodes with surface_label, semantic metadata, policy_meta
//!   - appraise mode: evaluate text against graph (agree/disagree %, verdict, evidence)
//!   - relate mode: find related nodes/edges by overlap scoring
//!   - query: context-aware lookup
//!   - snapshot_v1: produce v4.2 format snapshot
//!   - Seed bootstrap with new 24 atoms

pub mod compose;
pub mod ingest;
pub mod modes;
pub mod query;
pub mod snapshot;

use crate::attention::{CoocStats, EntityDetector, RsvsAttention};
use crate::autonomy::AutonomyEngine;
use crate::error::RsvsError;
use crate::events::{API_VERSION, SCHEMA_VERSION};
use crate::graph::RsvsGraph;
use crate::seed;
use crate::sense::SenseManager;
use crate::types::{NodeId, Tier};

use std::collections::HashMap;
use std::collections::VecDeque;

// Re-export everything from submodules
pub use ingest::IngestStats;
pub use modes::{AppraiseResult, RelateResult};
pub use query::QueryResult;
pub use snapshot::PipelineStatus;

// -----------------------------------------------------------------------
// PipelineConfig — all tunable knobs in one place
// -----------------------------------------------------------------------

/// All tunable knobs for the RSVS pipeline in one place.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Attention mechanism configuration.
    pub attention: crate::attention::AttentionConfig,
    /// Sense manager configuration.
    pub sense: crate::sense::SenseConfig,
    /// Autonomy engine configuration.
    pub autonomy: crate::autonomy::AutonomyConfig,

    /// N>= this to promote CANDIDATE_ID to node
    pub entity_promote_n: usize,

    /// Seed atom labels (for grounding check)
    pub seed_labels: Vec<String>,

    /// Domain tag for current ingestion batch
    pub current_domain: usize,

    /// Custom seed labels to use instead of the default 24 epistemological seeds.
    /// When provided, these become the Layer 1 atoms for the domain.
    pub custom_seeds: Option<Vec<String>>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        let mut attention = crate::attention::AttentionConfig::default();
        if let Ok(path) = std::env::var("RSVS_ATTENTION_CONFIG") {
            if let Ok(from_file) =
                crate::attention::AttentionConfig::from_json_file(std::path::Path::new(&path))
            {
                attention = from_file;
            }
        }
        Self {
            attention,
            sense: crate::sense::SenseConfig::default(),
            autonomy: crate::autonomy::AutonomyConfig::default(),
            entity_promote_n: 3,
            seed_labels: seed::SEED_LABEL_LIST
                .iter()
                .map(|s| s.to_string())
                .collect(),
            current_domain: 1,
            custom_seeds: None,
        }
    }
}

// -----------------------------------------------------------------------
// Rsvs — the main system struct (v4.2)
// -----------------------------------------------------------------------

/// The main RSVS system struct (v4.2).
///
/// Holds the knowledge graph, sense managers, autonomy engine, co-occurrence
/// statistics, entity detector, and attention scorer.
pub struct Rsvs {
    /// The knowledge graph.
    pub graph: RsvsGraph,
    /// Per-node sense managers.
    pub senses: HashMap<NodeId, SenseManager>,
    /// Autonomy engine for confidence and lifecycle management.
    pub autonomy: AutonomyEngine,
    /// Co-occurrence statistics database.
    pub stats_db: CoocStats,
    /// Entity detector for node promotion.
    pub entities: EntityDetector,
    /// Hard-attention scorer.
    pub attention: RsvsAttention,

    /// Token string → NodeId
    pub token_to_id: HashMap<String, NodeId>,

    /// NodeId → atom sets (for attention Jaccard)
    pub atom_sets: HashMap<String, Vec<NodeId>>,

    /// Pipeline configuration.
    pub config: PipelineConfig,
    /// Total contexts processed.
    pub total_contexts: usize,
    /// Latest event sequence number.
    pub latest_seq: u64,
    /// Ingest counter for correlation IDs.
    pub ingest_counter: u64,
    /// Maximum events to retain in memory.
    pub event_retention: usize,
    /// In-memory event stream.
    pub events: VecDeque<crate::events::RuntimeEvent>,
}

impl Rsvs {
    /// Create a new RSVS instance and bootstrap the 24 seed atoms.
    ///
    /// # Examples
    /// ```ignore
    /// let mut rsvs = Rsvs::new(PipelineConfig::default())?;
    /// ```
    pub fn new(config: PipelineConfig) -> Result<Self, RsvsError> {
        let mut graph = RsvsGraph::new();
        let mut autonomy = AutonomyEngine::new(config.autonomy.clone());
        let attention = RsvsAttention::new(config.attention.clone());

        // Bootstrap seed nodes (v4.2 format) — use custom seeds if provided
        let seed_map = seed::bootstrap(&mut graph, config.custom_seeds.as_ref().map(|v| &v[..]))?;
        let mut token_to_id: HashMap<String, NodeId> = HashMap::new();

        // Update seed_labels in config to match the actual seeds used
        let effective_seed_labels: Vec<String> = seed_map.keys().cloned().collect();

        for (label, &id) in &seed_map {
            autonomy.register_seed(id, 1.0, Tier::Tier1);
            token_to_id.insert(label.clone(), id);
        }

        // Each ID gets its own SenseManager
        let mut senses: HashMap<NodeId, SenseManager> = HashMap::new();
        for &id in seed_map.values() {
            senses.insert(id, SenseManager::new(config.sense.clone()));
        }

        let mut atom_sets: HashMap<String, Vec<NodeId>> = HashMap::new();
        for (label, &id) in &seed_map {
            atom_sets.insert(label.clone(), vec![id]);
        }

        let mut config = config;
        config.seed_labels = effective_seed_labels;

        Ok(Self {
            graph,
            senses,
            autonomy,
            stats_db: CoocStats::new(),
            entities: EntityDetector::new(),
            attention,
            token_to_id,
            atom_sets,
            config,
            total_contexts: 0,
            latest_seq: 0,
            ingest_counter: 0,
            event_retention: 10_000,
            events: VecDeque::new(),
        })
    }

    /// Generate the next correlation ID for an ingest batch.
    fn next_correlation_id(&mut self) -> String {
        self.ingest_counter += 1;
        format!("ingest_{:08}", self.ingest_counter)
    }

    /// Emit a runtime event into the in-memory event stream.
    fn emit_event(&mut self, correlation_id: &str, event_type: &str, payload: serde_json::Value) {
        self.latest_seq += 1;
        let evt = crate::events::RuntimeEvent {
            api_version: API_VERSION.to_string(),
            schema_version: SCHEMA_VERSION.to_string(),
            seq: self.latest_seq,
            correlation_id: correlation_id.to_string(),
            event_type: event_type.to_string(),
            payload,
        };
        self.events.push_back(evt);
        while self.events.len() > self.event_retention {
            let _ = self.events.pop_front();
        }
    }

    /// Save the full RSVS state to a JSON file.
    ///
    /// Delegates to [`crate::persist::save`].
    pub fn save(&self, path: &std::path::Path) -> Result<(), RsvsError> {
        crate::persist::save(self, path)
    }

    /// Load the full RSVS state from a JSON file.
    ///
    /// Delegates to [`crate::persist::load`].
    pub fn load(path: &std::path::Path) -> Result<Self, RsvsError> {
        crate::persist::load(path)
    }
}
