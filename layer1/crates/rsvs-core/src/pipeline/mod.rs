//! End-to-end pipeline — RSVS v6.0 Compositional Architecture
//!
//! Wires all modules together:
//!   text → CoocStats → EntityDetector → node promotion
//!   → SenseManager ingest (with composition induction) → AutonomyEngine update → graph query
//!
//! v6.0: Compositional architecture — every sense is formed by compositions.
//!   - ingest mode: induce senses with compositions from active context
//!   - compose mode: create compositional nodes with explicit (ID, sense) references
//!   - structural_similarity: compare nodes by shared/differing compositions
//!   - substitution_analysis: find what transforms one sense into another
//!   - Layer tracking: Layer 0 = primitive, Layer N = composed from Layer N-1

pub mod compose;
pub mod ingest;
pub mod modes;
pub mod query;
pub mod snapshot;
pub mod traverse;

use crate::attention::{CoocStats, EntityDetector, RsvsAttention};
use crate::autonomy::AutonomyEngine;
use crate::error::RsvsError;
use crate::events::{API_VERSION, SCHEMA_VERSION};
use crate::graph::RsvsGraph;
use crate::seed;
use crate::sense::SenseManager;
use crate::types::{AtomSet, ContextQueryResult, NodeId, NodeStatus, SenseId, Tier, TraversalConfig};

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

// Re-export everything from submodules
pub use ingest::IngestStats;
pub use modes::{AppraiseResult, AppraiseVerdict, RelateResult};
pub use query::QueryResult;
pub use snapshot::PipelineStatus;
pub use traverse::traverse as traverse_query;

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
    pub custom_seeds: Option<Vec<String>>,

    /// v6.1: Traversal configuration for depth-controlled lazy traversal.
    /// Controls how context-aware queries recursively expand CompositionRef trees.
    /// Default is TraversalConfig::default().
    pub traversal: TraversalConfig,

    /// v6.3: Threshold for learned entity promotion via E(i) = α×C + β×D.
    /// Tokens with entity_score > tau_entity_learned are candidates for promotion.
    /// Default: 0.15 (start conservative — needs empirical tuning).
    pub tau_entity_learned: f32,

    /// v6.3: Weight for centrality in entity score.
    pub alpha_entity: f32,

    /// v6.3: Weight for diversity in entity score.
    pub beta_entity: f32,

    /// v9.0: Master switch for all meaning pathways.
    /// When true, initializes BatchSeedSpreading, GapDetector,
    /// SeedActivationEngine, and DiscourseTracker.
    pub enable_meaning_pathways: bool,
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
            traversal: TraversalConfig::default(),
            tau_entity_learned: 0.15,
            alpha_entity: 0.5,
            beta_entity: 0.5,
            enable_meaning_pathways: true,
        }
    }
}

// -----------------------------------------------------------------------
// Rsvs — the main system struct (v6.0)
// -----------------------------------------------------------------------

/// The main RSVS system struct (v7.0 — Deep Losion Integration).
///
/// Holds the knowledge graph, sense managers (with compositions),
/// autonomy engine, co-occurrence statistics, entity detector, and
/// attention scorer.
///
/// v7.0 additions:
/// - `paradigm_router`: Adaptive traversal paradigm selection
/// - `spreading_activation`: Network activation through composition edges
/// - `deps_planner`: Structured failure recovery (Describe-Explain-Plan-Select)
///
/// v6.4 additions:
/// - `composition_index`: O(1) reverse lookup for CompositionRef → dependents
/// - `thinking_toggle`: Adaptive complexity toggle for traversal
/// - `consolidation`: Periodic consolidation engine
/// - `reflection`: Sense self-evaluation loop
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

    /// v8.0: Set of seed NodeIds for O(1) composition-based grounding checks.
    /// When a node's compositions reference any of these NodeIds, it is
    /// grounded to the seed layer. This replaces string-based grounding
    /// for composition-aware operations. The string-based `is_groundable_to_seeds`
    /// is retained for entity detection only.
    pub seed_node_ids: HashSet<NodeId>,

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
    /// v6.3: Batch counter for edge weight decay tracking.
    pub batch_counter: usize,
    /// v6.3: Per-domain attention config. Falls back to global config if domain not found.
    pub domain_configs: HashMap<usize, crate::attention::DomainAttentionConfig>,
    /// v6.4: Composition reverse index for O(1) lookup.
    pub composition_index: crate::composition_index::CompositionIndex,
    /// v6.4: Adaptive complexity toggle for traversal.
    pub thinking_toggle: crate::thinking::ThinkingToggle,
    /// v6.4: Periodic consolidation engine.
    pub consolidation: crate::consolidation::ConsolidationEngine,
    /// v6.4: Sense self-evaluation reflection engine.
    pub reflection: crate::reflection::SenseReflection,
    /// v7.0: Paradigm router — adaptive traversal paradigm selection.
    pub paradigm_router: crate::paradigm::ParadigmRouter,
    /// v7.0: Spreading activation engine — network activation through compositions.
    pub spreading_activation: crate::spreading::SpreadingActivation,
    /// v7.0: DEPS planner — structured failure recovery.
    pub deps_planner: crate::deps::DEPSPlanner,
    /// v8.0: Convergence detection engine — detects structural equivalence
    /// between nodes (e.g., "dog" ↔ "anjing") based on composition overlap.
    pub convergence: crate::convergence::ConvergenceEngine,

    // === v9.0: Meaning Pathways ===

    /// Master switch for all meaning pathways.
    pub enable_meaning_pathways: bool,

    /// v9.0: Batch seed spreading cache — shared across all pathways.
    /// Provides O(1) energy lookups from pre-computed spreading activation.
    pub batch_seed_spreading: Option<crate::batch_spreading::BatchSeedSpreading>,

    /// v9.0 Pathway 1: Gap detection — predicts missing compositions.
    pub gap_detector: Option<crate::gap_detection::GapDetector>,

    /// v9.0 Pathway 2: Affective-social seed activation — computes profiles.
    pub seed_activation_engine: Option<crate::seed_activation::SeedActivationEngine>,

    /// v9.0 Pathway 3: Discourse structure tracking — speech acts, centering.
    pub discourse_tracker: Option<crate::discourse_tracking::DiscourseTracker>,

    /// Sentence groups collected during per-sentence loop for P3.
    pub(crate) sentence_groups: Vec<Vec<NodeId>>,
}

impl Rsvs {
    /// Create a new RSVS instance and bootstrap the seed atoms.
    pub fn new(config: PipelineConfig) -> Result<Self, RsvsError> {
        let mut graph = RsvsGraph::new();
        let mut autonomy = AutonomyEngine::new(config.autonomy.clone());
        let attention = RsvsAttention::new(config.attention.clone());

        let seed_map = seed::bootstrap(&mut graph, config.custom_seeds.as_ref().map(|v| &v[..]))?;
        let mut token_to_id: HashMap<String, NodeId> = HashMap::new();
        let effective_seed_labels: Vec<String> = seed_map.keys().cloned().collect();

        for (label, &id) in &seed_map {
            autonomy.register_seed(id, 1.0, Tier::Tier1);
            token_to_id.insert(label.clone(), id);
        }

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

        // v9.0: Capture enable_meaning_pathways before config is moved
        let enable_meaning_pathways = config.enable_meaning_pathways;

        // v8.0: Build seed NodeId set for composition-based grounding
        let seed_node_ids: HashSet<NodeId> = seed_map.values().copied().collect();

        let mut rsvs = Self {
            graph,
            senses,
            autonomy,
            stats_db: CoocStats::new(),
            entities: EntityDetector::new(),
            attention,
            token_to_id,
            atom_sets,
            seed_node_ids,
            config,
            total_contexts: 0,
            latest_seq: 0,
            ingest_counter: 0,
            event_retention: 10_000,
            events: VecDeque::new(),
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

            // v9.0: Meaning Pathways initialization
            enable_meaning_pathways,
            batch_seed_spreading: None, // Initialized after bootstrap
            gap_detector: None,
            seed_activation_engine: None,
            discourse_tracker: None,
            sentence_groups: Vec::new(),
        };

        // v9.0: Initialize meaning pathway engines after bootstrap
        if rsvs.enable_meaning_pathways {
            rsvs.init_meaning_pathways()?;
        }

        Ok(rsvs)
    }

    /// v9.0: Initialize meaning pathway engines after bootstrap.
    ///
    /// Resolves seed NodeIds from the graph and creates all pathway engines.
    fn init_meaning_pathways(&mut self) -> Result<(), RsvsError> {
        use crate::batch_spreading::BatchSeedSpreading;
        use crate::gap_detection::{GapDetector, GapDetectionConfig};
        use crate::seed_activation::{SeedActivationEngine, SeedActivationConfig};
        use crate::discourse_tracking::{DiscourseTracker, DiscourseConfig};

        // Resolve seed NodeIds for each pathway
        let resolve_seeds = |labels: &[&str]| -> Vec<NodeId> {
            labels.iter()
                .filter_map(|l| self.graph.id_for_label(l))
                .collect()
        };

        let affective_seeds = resolve_seeds(&["value", "risk"]);
        let social_seeds = resolve_seeds(&["trust", "identity", "agent"]);
        let pragmatic_seeds = resolve_seeds(&["goal", "feedback", "action"]);

        // Create BatchSeedSpreading cache
        self.batch_seed_spreading = Some(BatchSeedSpreading::new(
            self.spreading_activation.clone_config(),
            affective_seeds.clone(),
            social_seeds.clone(),
            pragmatic_seeds.clone(),
        ));

        // Create GapDetector
        self.gap_detector = Some(GapDetector::new(GapDetectionConfig {
            affective_seeds: affective_seeds.clone(),
            social_seeds: social_seeds.clone(),
            pragmatic_seeds: pragmatic_seeds.clone(),
            ..GapDetectionConfig::default()
        }));

        // Create SeedActivationEngine
        self.seed_activation_engine = Some(SeedActivationEngine::new(
            SeedActivationConfig::default(),
            &self.graph,
        ));

        // Create DiscourseTracker
        self.discourse_tracker = Some(DiscourseTracker::new(DiscourseConfig::default()));

        Ok(())
    }
    /// Generate the next correlation ID for an ingest batch.
    fn next_correlation_id(&mut self) -> String {
        self.ingest_counter += 1;
        format!("ingest_{:08}", self.ingest_counter)
    }

    /// v7.2: Insert a token→id mapping into BOTH `token_to_id` and `graph.label_to_id`.
    ///
    /// This is the single source of truth for label→id registration.
    /// Previously, `token_to_id` and `graph.label_to_id` were updated
    /// independently in different places, which could cause them to
    /// diverge. All label→id insertions should go through this method.
    pub(crate) fn register_label(&mut self, label: &str, id: NodeId, surface_label: Option<&str>) {
        self.token_to_id.insert(label.to_string(), id);
        self.graph.label_to_id.insert(label.to_string(), id);
        if let Some(sl) = surface_label {
            if sl != label {
                self.graph.label_to_id.insert(sl.to_string(), id);
            }
        }
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
    pub fn save(&self, path: &std::path::Path) -> Result<(), RsvsError> {
        crate::persist::save(self, path)
    }

    /// Load the full RSVS state from a JSON file.
    pub fn load(path: &std::path::Path) -> Result<Self, RsvsError> {
        crate::persist::load(path)
    }

    /// Compute the layer for a new compositional node based on its composition targets.
    ///
    /// The layer is max(layer of all composition targets) + 1.
    /// If no compositions, layer = 0 (primitive).
    pub fn compute_layer(&self, composition_ids: &[NodeId]) -> u32 {
        if composition_ids.is_empty() {
            return 0;
        }

        let max_layer = composition_ids
            .iter()
            .filter_map(|&id| self.graph.get_node(id))
            .map(|n| n.semantic.layer)
            .max()
            .unwrap_or(0);

        max_layer + 1
    }

    /// v8.0: Check if a node is groundable via structural composition to seed NodeIds.
    ///
    /// This is the language-agnostic grounding check. A node is composition-groundable
    /// if ANY of its sense compositions reference a seed NodeId. This replaces the
    /// string-based `is_groundable_to_seeds()` for all composition-aware operations.
    ///
    /// The string-based check is retained ONLY for entity detection (which operates
    /// on raw tokens before they have compositions).
    pub fn is_groundable_via_composition(&self, node_id: NodeId) -> bool {
        if self.seed_node_ids.contains(&node_id) {
            return true; // It IS a seed
        }
        let sm = match self.senses.get(&node_id) {
            Some(sm) => sm,
            None => return false,
        };
        for sense in &sm.senses {
            for comp in &sense.compositions {
                if self.seed_node_ids.contains(&comp.node_id) {
                    return true;
                }
            }
        }
        false
    }

    /// v8.0: Check if a node qualifies as an "internal representation" (layer 1).
    ///
    /// An internal representation is a node whose ALL sense compositions reference
    /// ONLY layer 0 seed primitives. Such nodes serve as the bridge between surface
    /// tokens (layer 2+) and epistemological primitives (layer 0).
    ///
    /// Returns true if:
    /// - The node has at least one compositional sense
    /// - ALL composition targets across ALL senses are layer 0 seed nodes
    pub fn is_internal_representation(&self, node_id: NodeId) -> bool {
        if self.seed_node_ids.contains(&node_id) {
            return false; // Seeds themselves are not internal representations
        }
        let sm = match self.senses.get(&node_id) {
            Some(sm) => sm,
            None => return false,
        };

        let mut has_any_composition = false;
        for sense in &sm.senses {
            if sense.compositions.is_empty() {
                continue;
            }
            has_any_composition = true;
            for comp in &sense.compositions {
                // If ANY composition targets a non-seed node, this is NOT an internal repr
                if !self.seed_node_ids.contains(&comp.node_id) {
                    return false;
                }
            }
        }
        has_any_composition
    }

    /// v6.3: Get the active attention config for the current domain.
    /// Falls back to global config if domain has insufficient observations.
    pub fn active_attention_config(&self) -> crate::attention::AttentionConfig {
        if let Some(dc) = self.domain_configs.get(&self.config.current_domain) {
            if dc.observation_count >= 5 {
                let mut cfg = self.config.attention.clone();
                cfg.alpha = dc.alpha;
                cfg.beta = dc.beta;
                cfg.gamma = dc.gamma;
                return cfg;
            }
        }
        self.config.attention.clone()
    }

    /// v6.3: Return tokens that are entity candidates based on learned scoring.
    /// These are tokens that have high centrality (often targeted by attention)
    /// and high diversity (attend to many different concepts) but haven't been
    /// promoted to nodes yet.
    pub fn entity_candidates(&self, top_k: usize) -> Vec<(String, f32)> {
        let alpha_e = self.config.alpha_entity;
        let beta_e = self.config.beta_entity;
        let tau = self.config.tau_entity_learned;

        let mut candidates: Vec<(String, f32)> = self.stats_db
            .token_counts()
            .keys()
            .filter(|t| !self.token_to_id.contains_key(t.as_str()))
            .map(|t| {
                let score = self.stats_db.entity_score(t, alpha_e, beta_e);
                (t.clone(), score)
            })
            .filter(|(_, score)| *score >= tau)
            .collect();

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        candidates.truncate(top_k);
        candidates
    }

    /// Get the active sense for a node in a given context.
    ///
    /// Returns (sense_idx, sense_id) or None if the node has no senses.
    pub fn active_sense_for_node(
        &self,
        node_id: NodeId,
        context_node_ids: &[NodeId],
    ) -> Option<(usize, SenseId)> {
        let sm = self.senses.get(&node_id)?;
        if sm.senses.is_empty() {
            return None;
        }

        // Try lazy_lookup with context
        let context_atoms: Vec<NodeId> = context_node_ids.to_vec();
        if let Some(idx) = sm.lazy_lookup(&context_atoms) {
            return Some((idx, sm.senses[idx].id));
        }

        // Fallback: first sense
        Some((0, sm.senses[0].id))
    }

    /// v7.0: Detect if a composition from `start_node` would create a cycle
    /// that leads back to a node with the given `target_label`.
    ///
    /// This is used by the compose() method to prevent circular composition chains.
    /// A cycle exists if following the composition references from `start_node`
    /// leads back to any node that matches the target label.
    pub fn detect_composition_cycle(&self, start_node: NodeId, target_label: &str) -> bool {
        let target_id = self.token_to_id.get(target_label).copied();
        let mut visited = std::collections::HashSet::new();
        visited.insert(start_node);
        let mut stack = vec![start_node];

        while let Some(current) = stack.pop() {
            // Check if current node is the target
            if target_id == Some(current) {
                return true;
            }

            // Expand compositions of current node
            if let Some(sm) = self.senses.get(&current) {
                for sense in &sm.senses {
                    for comp in &sense.compositions {
                        if visited.contains(&comp.node_id) {
                            continue;
                        }
                        // Check if this composition target matches the target label
                        if let Some(n) = self.graph.get_node(comp.node_id) {
                            if n.label == target_label {
                                return true;
                            }
                        }
                        visited.insert(comp.node_id);
                        stack.push(comp.node_id);
                    }
                }
            }
        }
        false
    }

    // -------------------------------------------------------------------
    // v6.1: Context-Aware Query Endpoint
    // -------------------------------------------------------------------

    /// Context-aware query that uses depth-controlled lazy traversal.
    ///
    /// v7.2: Now uses ParadigmRouter for adaptive paradigm selection BEFORE
    /// ThinkingToggle fine-tunes depth. The router picks the lightest strategy
    /// (Direct → Shallow → Standard → Deep → MCTS) that will likely succeed,
    /// and the toggle adjusts within that paradigm.
    ///
    /// This is the v6.1 query endpoint (Point 4) that:
    /// 1. Resolves label to NodeId
    /// 2. Uses lazy_lookup to select active sense based on context
    /// 3. Uses ParadigmRouter to select optimal traversal strategy
    /// 4. Uses ThinkingToggle to fine-tune depth within the paradigm
    /// 5. Computes P(a|S,q) per atom using freq_map
    /// 6. Optionally recurses into compositions based on TraversalConfig
    /// 7. Returns ContextQueryResult
    ///
    /// # Arguments
    ///
    /// * `id_or_label` - The concept label (e.g., "raja") or node ID string
    /// * `context_atoms` - Context atom labels to disambiguate the query
    /// * `config` - Optional TraversalConfig (uses PipelineConfig.traversal if None)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let result = rsvs.context_query("raja", &vec!["kerajaan", "tahta"], None);
    /// ```
    pub fn context_query(
        &self,
        id_or_label: &str,
        context_atoms: &[&str],
        config: Option<&TraversalConfig>,
    ) -> Option<ContextQueryResult> {
        // 1. Resolve label to NodeId
        let start_node = *self.token_to_id.get(id_or_label)?;

        // 2. Convert context atom labels to NodeIds
        let context_ids: AtomSet = context_atoms
            .iter()
            .filter_map(|label| self.token_to_id.get(*label).copied())
            .collect();

        if context_ids.is_empty() {
            // No context atoms found — fall back to basic query
            return None;
        }

        // 3. Use provided config or fall back to pipeline config
        let base_config = config.cloned().unwrap_or_else(|| self.config.traversal.clone());

        // 4. v7.2: Build complexity signal for both router and toggle
        let signal = crate::thinking::ComplexitySignal {
            n_context_atoms: context_ids.len(),
            n_senses: self.senses.get(&start_node).map(|sm| sm.senses.len()).unwrap_or(0),
            target_layer: self.graph.get_node(start_node).map(|n| n.semantic.layer).unwrap_or(0),
            is_compositional: self.senses.get(&start_node)
                .and_then(|sm| sm.senses.first())
                .map(|s| s.is_compositional())
                .unwrap_or(false),
            domain_complexity: 0.0,
        };

        // 5. v7.2: Use ParadigmRouter to select optimal traversal strategy
        //    The router picks the lightest paradigm that will likely succeed,
        //    avoiding over-computation for simple queries.
        let confidence = self.senses.get(&start_node)
            .and_then(|sm| sm.senses.first())
            .map(|s| s.grounding.score())
            .unwrap_or(0.5);
        let paradigm = self.paradigm_router.route(confidence, &signal, self.config.current_domain);
        let paradigm_config = self.paradigm_router.to_traversal_config(paradigm, &base_config);

        // 6. v6.4: Use ThinkingToggle to fine-tune within the paradigm
        let mode = self.thinking_toggle.classify(&signal);
        let adjusted_config = self.thinking_toggle.adjust_traversal(&mode, &paradigm_config);

        // 7. Call the traversal engine with paradigm-adjusted config
        Some(traverse::traverse(
            &self.graph,
            &self.senses,
            start_node,
            &context_ids,
            &adjusted_config,
            &self.token_to_id,
            self.config.sense.tau_core as f64,
        ))
    }

    // -------------------------------------------------------------------
    // v8.2: Pruning for unbounded HashMap growth
    // -------------------------------------------------------------------

    /// Prune deprecated entries from `atom_sets`, `token_to_id`, and `senses`.
    ///
    /// Over time, nodes may become Deprecated (confidence fell below threshold,
    /// status transitioned to Deprecated by the autonomy engine). These nodes
    /// still occupy space in the main HashMaps. This method removes entries
    /// where the node has been deprecated, freeing memory.
    ///
    /// Returns the number of entries removed across all HashMaps.
    ///
    /// Should be called periodically (e.g., after every N ingests, or when
    /// total node count exceeds a threshold).
    pub fn prune_deprecated(&mut self) -> usize {
        // Collect deprecated node IDs from the autonomy engine
        let deprecated_ids: HashSet<NodeId> = self.autonomy.records
            .iter()
            .filter(|(_, rec)| matches!(rec.status, NodeStatus::Deprecated))
            .map(|(&id, _)| id)
            .collect();

        let mut removed = 0;

        for &id in &deprecated_ids {
            // Remove from senses
            if self.senses.remove(&id).is_some() {
                removed += 1;
            }

            // Remove from atom_sets (need label, look it up from graph)
            if let Some(node) = self.graph.get_node(id) {
                let label = node.label.clone();
                if self.atom_sets.remove(&label).is_some() {
                    removed += 1;
                }
                if self.token_to_id.remove(&label).is_some() {
                    removed += 1;
                }
            }
        }

        removed
    }
}
