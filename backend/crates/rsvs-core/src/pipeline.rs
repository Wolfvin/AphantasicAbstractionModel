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

use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::Path;
use crate::types::{NodeId, Tier, Node, NodeStatus, Edge, EdgeSource,
                   CompressionState, SemanticMeta, PolicyMeta};
use crate::graph::RsvsGraph;
use crate::seed;
use crate::sense::{SenseManager, SenseConfig, IngestResult};
use crate::attention::{
    CoocStats, RsvsAttention, AttentionConfig, EntityDetector,
    text_to_sentences, is_groundable_to_seeds,
};
use crate::autonomy::{AutonomyEngine, AutonomyConfig, ConfidenceUpdateResult};
use crate::events::{RuntimeSnapshot, RuntimeNode, RuntimeEdge, RuntimeEvent, EventBatch, API_VERSION, SCHEMA_VERSION};

// -----------------------------------------------------------------------
// PipelineConfig — all tunable knobs in one place
// -----------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub attention:  AttentionConfig,
    pub sense:      SenseConfig,
    pub autonomy:   AutonomyConfig,

    /// N>= this to promote CANDIDATE_ID to node
    pub entity_promote_n: usize,

    /// Seed atom labels (for grounding check)
    pub seed_labels: Vec<String>,

    /// Domain tag for current ingestion batch
    pub current_domain: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        let mut attention = AttentionConfig::default();
        if let Ok(path) = std::env::var("RSVS_ATTENTION_CONFIG") {
            if let Ok(from_file) = AttentionConfig::from_json_file(Path::new(&path)) {
                attention = from_file;
            }
        }
        Self {
            attention,
            sense:            SenseConfig::default(),
            autonomy:         AutonomyConfig::default(),
            entity_promote_n: 3,
            seed_labels:      seed::SEED_LABEL_LIST.iter().map(|s| s.to_string()).collect(),
            current_domain:   1,
        }
    }
}

// -----------------------------------------------------------------------
// IngestStats — what happened during one ingest call
// -----------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct IngestStats {
    pub sentences_processed: usize,
    pub atoms_promoted:      usize,
    pub sense_assigned:      usize,
    pub sense_created:       usize,
    pub confidence_updated:  usize,
    pub watchlist_additions: usize,
    pub frozen_batches:      usize,
}

// -----------------------------------------------------------------------
// QueryResult — output of a context-aware query
// -----------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub active_sense_idx: usize,
    pub active_sense_n:   usize,
    pub scored_atoms:     Vec<(String, f32)>,
}

// -----------------------------------------------------------------------
// AppraiseResult — v4.2 appraise mode output
// -----------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AppraiseResult {
    pub agree_pct:    f32,      // % of tokens found in graph
    pub disagree_pct: f32,      // % of tokens NOT found
    pub verdict:      String,   // "consistent", "partial", "novel"
    pub evidence:     Vec<(String, f32)>,  // (token, confidence) for matched tokens
}

// -----------------------------------------------------------------------
// RelateResult — v4.2 relate mode output
// -----------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RelateResult {
    pub related_nodes: Vec<(NodeId, f32)>,  // (node_id, overlap_score)
    pub related_edges: Vec<(NodeId, NodeId, f32)>,  // (from, to, weight)
}

// -----------------------------------------------------------------------
// Rsvs — the main system struct (v4.2)
// -----------------------------------------------------------------------

pub struct Rsvs {
    pub graph:    RsvsGraph,
    pub senses:   HashMap<NodeId, SenseManager>,
    pub autonomy: AutonomyEngine,
    pub stats_db: CoocStats,
    pub entities: EntityDetector,
    pub attention: RsvsAttention,

    /// token string → NodeId
    pub token_to_id: HashMap<String, NodeId>,

    /// NodeId → atom sets (for attention Jaccard)
    pub atom_sets: HashMap<String, Vec<NodeId>>,

    pub config: PipelineConfig,
    pub total_contexts: usize,
    pub latest_seq: u64,
    pub ingest_counter: u64,
    pub event_retention: usize,
    pub events: VecDeque<RuntimeEvent>,
}

impl Rsvs {
    /// Create a new RSVS instance and bootstrap seed nodes.
    pub fn new(config: PipelineConfig) -> Self {
        let mut graph   = RsvsGraph::new();
        let mut autonomy = AutonomyEngine::new(config.autonomy.clone());
        let attention   = RsvsAttention::new(config.attention.clone());

        // Bootstrap seed nodes (v4.2 format)
        let seed_map = seed::bootstrap(&mut graph);
        let mut token_to_id: HashMap<String, NodeId> = HashMap::new();

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

        Self {
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
        }
    }

    fn next_correlation_id(&mut self) -> String {
        self.ingest_counter += 1;
        format!("ingest_{:08}", self.ingest_counter)
    }

    fn emit_event(
        &mut self,
        correlation_id: &str,
        event_type: &str,
        payload: serde_json::Value,
    ) {
        self.latest_seq += 1;
        let evt = RuntimeEvent {
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

    pub fn latest_seq_v1(&self) -> u64 {
        self.latest_seq
    }

    pub fn consume_events_v1(&self, after_seq: Option<u64>, limit: usize) -> EventBatch {
        let after = after_seq.unwrap_or(0);
        let lim = limit.clamp(1, 5000);
        let events = self.events
            .iter()
            .filter(|e| e.seq > after)
            .take(lim)
            .cloned()
            .collect::<Vec<_>>();

        EventBatch {
            api_version: API_VERSION.to_string(),
            schema_version: SCHEMA_VERSION.to_string(),
            latest_seq: self.latest_seq,
            events,
        }
    }

    /// v4.2 snapshot with unified node model
    pub fn snapshot_v1(&self) -> RuntimeSnapshot {
        let nodes = self.graph.nodes.values().map(|n| {
            let sense = self.senses.get(&n.id);
            RuntimeNode {
                id: n.id,
                label: n.label.clone(),
                surface_label: n.surface_label.clone(),
                kind: n.kind.clone(),
                tier: match n.tier {
                    Tier::Tier1 => 1,
                    Tier::Tier2 => 2,
                    Tier::Tier3 => 3,
                },
                confidence: self.autonomy.confidence(n.id).unwrap_or(n.confidence),
                status: match self.autonomy.status(n.id).unwrap_or(&n.status) {
                    NodeStatus::New => "new",
                    NodeStatus::Candidate => "candidate",
                    NodeStatus::Stable => "stable",
                    NodeStatus::Deprecated => "deprecated",
                    NodeStatus::Quarantine => "quarantine",
                }.to_string(),
                is_seed: n.is_seed,
                is_locked: n.is_locked,
                compression_state: match n.semantic.compression_state {
                    CompressionState::Raw => "raw",
                    CompressionState::Compressed => "compressed",
                }.to_string(),
                derived_from_node_ids: n.semantic.derived_from_node_ids.clone(),
                sense_count: sense.map(|s| s.sense_count()).unwrap_or(0),
                coherence: sense.and_then(|s| s.senses.first().map(|x| x.coherence)),
            }
        }).collect::<Vec<_>>();

        let mut edges = Vec::new();
        for (from, list) in &self.graph.edges {
            for e in list {
                edges.push(RuntimeEdge {
                    id: format!("{}->{}", from, e.to),
                    source: e.from,
                    target: e.to,
                    weight: e.weight,
                    source_type: if e.source == EdgeSource::Bootstrap { "bootstrap".into() } else { "learned".into() },
                });
            }
        }

        RuntimeSnapshot {
            api_version: API_VERSION.to_string(),
            schema_version: SCHEMA_VERSION.to_string(),
            latest_seq: self.latest_seq,
            total_contexts: self.total_contexts,
            nodes,
            edges,
        }
    }

    // -----------------------------------------------------------------------
    // Main entry: ingest a block of text (v4.2 mode: ingest)
    // -----------------------------------------------------------------------

    pub fn ingest_text(&mut self, text: &str) -> IngestStats {
        let mut stats = IngestStats::default();
        let correlation_id = self.next_correlation_id();
        self.emit_event(
            &correlation_id,
            "ingest_started",
            serde_json::json!({
                "text_len": text.len(),
                "domain": self.config.current_domain,
                "mode": "ingest"
            }),
        );

        let sentences = text_to_sentences(text);
        if sentences.is_empty() { return stats; }

        // --- Step 1: Update co-occurrence statistics ---
        for tokens in &sentences {
            self.stats_db.ingest_sentence(tokens);
            stats.sentences_processed += 1;

            for token in tokens {
                let groundable = is_groundable_to_seeds(
                    token,
                    &self.config.seed_labels.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                );
                self.entities.record(token, groundable);
            }
        }

        // --- Step 2: Promote new entity candidates to nodes (v4.2 format) ---
        let candidates = self.entities.candidates(self.config.entity_promote_n);
        for token in &candidates {
            if self.token_to_id.contains_key(token.as_str()) { continue; }

            let id = self.graph.insert_node(Node {
                id: 0,
                label: token.clone(),
                surface_label: format!("{}@en", token),

                kind: "node".to_string(),
                tier: Tier::Tier2,
                confidence: 0.50,
                status: NodeStatus::Candidate,
                is_seed: false,
                is_locked: false,

                semantic: SemanticMeta {
                    compression_state: CompressionState::Raw,
                    derived_from_node_ids: vec![],
                    compression_reason: None,
                },
                policy_meta: Some(PolicyMeta {
                    policy_version: "4.2".to_string(),
                    governance_score: 0.0,
                    candidate_evidence_pool: 0.0,
                    status_flip_count: 0,
                    seen_fingerprints: vec![],
                    last_seen_at: None,
                }),
                language_links: vec![],

                atoms: vec![],
                fingerprint: None,
            }).unwrap();

            self.autonomy.register(id, 0.50, Tier::Tier2);
            self.token_to_id.insert(token.clone(), id);
            self.atom_sets.insert(token.clone(), vec![id]);
            self.senses.insert(id, SenseManager::new(self.config.sense.clone()));
            stats.atoms_promoted += 1;
            self.emit_event(
                &correlation_id,
                "node_created",
                serde_json::json!({
                    "id": id,
                    "label": token,
                    "surface_label": format!("{}@en", token),
                    "tier": 2,
                    "confidence": 0.5,
                    "status": "candidate",
                    "compression_state": "raw"
                }),
            );
        }

        // --- Step 2b: Build/update node atom sets from co-occurrence ---
        self.update_node_atoms(&candidates, &correlation_id);

        // --- Step 3: For each sentence, run attention + feed senses ---
        self.autonomy.begin_batch();
        let snapshot = self.autonomy.snapshot();

        for tokens in &sentences {
            self.total_contexts += 1;
            self.autonomy.tick_context();

            let selected = self.attention.select(
                tokens,
                &self.stats_db,
                &self.atom_sets,
            );

            for token in tokens {
                let token_id = match self.token_to_id.get(token.as_str()) {
                    Some(&id) => id,
                    None      => continue,
                };

                let context: Vec<NodeId> = if let Some(cands) = selected.get(token) {
                    cands.iter()
                        .filter_map(|c| self.token_to_id.get(c.token.as_str()).copied())
                        .collect()
                } else {
                    tokens.iter()
                        .filter(|t| *t != token)
                        .filter_map(|t| self.token_to_id.get(t.as_str()).copied())
                        .collect()
                };

                if context.is_empty() { continue; }

                let sense_mgr = self.senses.entry(token_id)
                    .or_insert_with(|| SenseManager::new(self.config.sense.clone()));

                let ingest_result = sense_mgr.ingest(context.clone());
                let mut sense_event: Option<serde_json::Value> = None;
                match ingest_result {
                    IngestResult::Assigned(idx) => {
                        stats.sense_assigned += 1;
                        sense_event = Some(serde_json::json!({
                            "id": token_id,
                            "sense_idx": idx,
                            "action": "assigned"
                        }));
                    }
                    IngestResult::Created(idx)  => {
                        stats.sense_created  += 1;
                        sense_event = Some(serde_json::json!({
                            "id": token_id,
                            "sense_idx": idx,
                            "action": "created"
                        }));
                    }
                }

                let active_coherence = sense_mgr.senses.first()
                    .map(|s| s.coherence)
                    .unwrap_or(0.5);

                let freq = 1.0f32;

                let co_ids: Vec<NodeId> = context.clone();
                let old_conf = self.autonomy.confidence(token_id).unwrap_or(0.0);
                let old_tier = self.autonomy.tier(token_id).cloned();
                let old_status = self.autonomy.status(token_id).cloned();
                let result = self.autonomy.update_confidence(
                    token_id,
                    freq,
                    active_coherence,
                    &co_ids,
                    self.config.current_domain,
                );
                if matches!(result, ConfidenceUpdateResult::Updated { .. }) {
                    stats.confidence_updated += 1;
                }
                if let Some(payload) = sense_event.take() {
                    self.emit_event(&correlation_id, "sense_changed", payload);
                }
                let new_conf = self.autonomy.confidence(token_id).unwrap_or(old_conf);
                if (new_conf - old_conf).abs() > f32::EPSILON {
                    self.emit_event(
                        &correlation_id,
                        "confidence_changed",
                        serde_json::json!({
                            "id": token_id,
                            "before": old_conf,
                            "after": new_conf
                        }),
                    );
                }
                let new_tier = self.autonomy.tier(token_id).cloned();
                if new_tier != old_tier {
                    let tier_num = |t: Option<Tier>| -> u8 {
                        match t.unwrap_or(Tier::Tier3) {
                            Tier::Tier1 => 1,
                            Tier::Tier2 => 2,
                            Tier::Tier3 => 3,
                        }
                    };
                    self.emit_event(
                        &correlation_id,
                        "tier_changed",
                        serde_json::json!({
                            "id": token_id,
                            "before": tier_num(old_tier),
                            "after": tier_num(new_tier)
                        }),
                    );
                }
                let new_status = self.autonomy.status(token_id).cloned();
                if new_status != old_status {
                    self.emit_event(
                        &correlation_id,
                        "status_changed",
                        serde_json::json!({
                            "id": token_id,
                            "before": format!("{:?}", old_status.unwrap_or(NodeStatus::New)),
                            "after": format!("{:?}", new_status.unwrap_or(NodeStatus::New))
                        }),
                    );
                }
            }

            // Periodic sense maintenance
            if self.total_contexts % 20 == 0 {
                for sense_mgr in self.senses.values_mut() {
                    sense_mgr.check_merge();
                    sense_mgr.purge_fragile();
                }
            }
        }

        // --- Step 4: Check global stability ---
        let stability = self.autonomy.check_global_stability();
        if matches!(stability, crate::autonomy::StabilityStatus::Frozen { .. }) {
            self.autonomy.rollback(&snapshot);
            stats.frozen_batches += 1;
            self.emit_event(
                &correlation_id,
                "confidence_changed",
                serde_json::json!({
                    "action": "rollback_freeze",
                    "frozen_batches": stats.frozen_batches
                }),
            );
        }

        stats.watchlist_additions = self.autonomy.watchlist_len();
        self.emit_event(
            &correlation_id,
            "ingest_completed",
            serde_json::json!({
                "sentences_processed": stats.sentences_processed,
                "atoms_promoted": stats.atoms_promoted,
                "sense_assigned": stats.sense_assigned,
                "sense_created": stats.sense_created,
                "confidence_updated": stats.confidence_updated,
                "frozen_batches": stats.frozen_batches,
                "watchlist_additions": stats.watchlist_additions
            }),
        );
        stats
    }

    // -----------------------------------------------------------------------
    // Build/update node atom sets from co-occurrence data (v4.2)
    // -----------------------------------------------------------------------

    fn update_node_atoms(&mut self, tokens: &[String], correlation_id: &str) {
        for token in tokens {
            let token_id = match self.token_to_id.get(token.as_str()) {
                Some(&id) => id,
                None      => continue,
            };

            // Skip seed nodes — they stay Raw
            if let Some(node) = self.graph.get_node(token_id) {
                if node.is_seed { continue; }
                if node.semantic.compression_state == CompressionState::Compressed && !node.atoms.is_empty() {
                    continue; // already built
                }
            }

            // Collect top co-occurring nodes from stats
            let mut cooc_nodes: Vec<(NodeId, f32)> = self.token_to_id
                .iter()
                .filter(|(t, _)| t.as_str() != token.as_str())
                .filter_map(|(t, &id)| {
                    let c = self.stats_db.cooc(token, t);
                    if c > 0.15 { Some((id, c)) } else { None }
                })
                .collect();

            if cooc_nodes.is_empty() { continue; }

            cooc_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            cooc_nodes.truncate(8);

            let atom_ids: Vec<NodeId> = cooc_nodes.iter().map(|(id, _)| *id).collect();

            // Update node: set atoms and mark as compressed if has derived nodes
            if let Some(node) = self.graph.nodes.get_mut(&token_id) {
                let old_atoms = node.atoms.clone();
                node.atoms = atom_ids.clone();

                // If node has derived atoms, mark as compressed
                if !atom_ids.is_empty() {
                    node.semantic.compression_state = CompressionState::Compressed;
                    node.semantic.derived_from_node_ids = atom_ids.clone();
                    node.semantic.compression_reason = Some("co-occurrence aggregation".to_string());
                }

                for removed in old_atoms.iter().filter(|a| !atom_ids.contains(a)) {
                    if let Some(list) = self.graph.edges.get_mut(removed) {
                        let before = list.len();
                        list.retain(|e| !(e.to == token_id && e.source == EdgeSource::Learned));
                        if list.len() < before {
                            self.emit_event(
                                correlation_id,
                                "edge_removed",
                                serde_json::json!({
                                    "source": *removed,
                                    "target": token_id
                                }),
                            );
                        }
                    }
                }

                for (source_id, weight) in &cooc_nodes {
                    let mut is_new = false;
                    let mut old_weight = 0.0f32;
                    let edges = self.graph.edges.entry(*source_id).or_default();
                    if let Some(existing) = edges.iter_mut()
                        .find(|e| e.to == token_id && e.source == EdgeSource::Learned)
                    {
                        old_weight = existing.weight;
                        existing.weight = *weight;
                    } else {
                        edges.push(Edge {
                            from: *source_id,
                            to: token_id,
                            weight: *weight,
                            source: EdgeSource::Learned,
                        });
                        is_new = true;
                    }

                    if is_new {
                        self.emit_event(
                            correlation_id,
                            "edge_created",
                            serde_json::json!({
                                "source": *source_id,
                                "target": token_id,
                                "weight": *weight
                            }),
                        );
                    } else if (old_weight - *weight).abs() > f32::EPSILON {
                        self.emit_event(
                            correlation_id,
                            "edge_weight_changed",
                            serde_json::json!({
                                "source": *source_id,
                                "target": token_id,
                                "before": old_weight,
                                "after": *weight
                            }),
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Query: context-aware lookup for a concept
    // -----------------------------------------------------------------------

    pub fn query(&self, concept: &str, query_context: &str) -> Option<QueryResult> {
        let concept_id = *self.token_to_id.get(concept)?;
        let sense_mgr  = self.senses.get(&concept_id)?;

        let query_tokens = crate::attention::tokenize(query_context);
        let query_atoms: Vec<NodeId> = query_tokens.iter()
            .filter_map(|t| self.token_to_id.get(t.as_str()).copied())
            .collect();

        let active_sense_idx = sense_mgr.lazy_lookup(&query_atoms)
            .or_else(|| if sense_mgr.sense_count() > 0 { Some(0) } else { None })?;

        let sense = sense_mgr.get_sense(active_sense_idx)?;

        let tau = self.config.sense.tau_core;
        let core = sense.core(tau);

        let mut scored: Vec<(String, f32)> = core.iter().filter_map(|&atom_id| {
            let label = self.graph.get_node(atom_id)?.label.clone();

            let freq = sense.freq(atom_id);

            let edge_score = self.graph.edges_from(atom_id).iter()
                .filter(|e| query_atoms.contains(&e.to))
                .map(|e| e.weight)
                .fold(0.0f32, f32::max);

            let score = if edge_score > 0.0 {
                freq * edge_score
            } else {
                freq
            };

            Some((label, score))
        }).collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Some(QueryResult {
            active_sense_idx,
            active_sense_n: sense.context_count(),
            scored_atoms: scored,
        })
    }

    // -----------------------------------------------------------------------
    // Similarity between two concepts
    // -----------------------------------------------------------------------

    pub fn similarity(&self, a: &str, b: &str) -> Option<crate::graph::SimilarityResult> {
        let id_a = *self.token_to_id.get(a)?;
        let id_b = *self.token_to_id.get(b)?;
        Some(self.graph.similarity(id_a, id_b))
    }

    // -----------------------------------------------------------------------
    // v4.2: Appraise — evaluate text against graph
    // -----------------------------------------------------------------------

    /// Evaluate text against the graph. Returns agree/disagree %, verdict, evidence.
    pub fn appraise(&self, text: &str) -> AppraiseResult {
        let tokens = crate::attention::tokenize(text);
        if tokens.is_empty() {
            return AppraiseResult {
                agree_pct: 0.0,
                disagree_pct: 100.0,
                verdict: "novel".to_string(),
                evidence: vec![],
            };
        }

        let total = tokens.len() as f32;
        let mut found = 0usize;
        let mut evidence: Vec<(String, f32)> = Vec::new();

        for token in &tokens {
            if let Some(&id) = self.token_to_id.get(token.as_str()) {
                found += 1;
                let conf = self.autonomy.confidence(id).unwrap_or(0.0);
                evidence.push((token.clone(), conf));
            }
        }

        let agree_pct = (found as f32 / total) * 100.0;
        let disagree_pct = 100.0 - agree_pct;

        let verdict = if agree_pct >= 80.0 {
            "consistent"
        } else if agree_pct >= 40.0 {
            "partial"
        } else {
            "novel"
        }.to_string();

        // Sort evidence by confidence descending
        evidence.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        AppraiseResult {
            agree_pct,
            disagree_pct,
            verdict,
            evidence,
        }
    }

    // -----------------------------------------------------------------------
    // v4.2: Relate — find related nodes/edges by overlap scoring
    // -----------------------------------------------------------------------

    /// Find nodes and edges related to the given concept by overlap scoring.
    pub fn relate(&self, concept: &str) -> Option<RelateResult> {
        let concept_id = *self.token_to_id.get(concept)?;

        // Find related nodes by Jaccard similarity
        let mut related_nodes: Vec<(NodeId, f32)> = Vec::new();
        for (&other_id, _) in &self.graph.nodes {
            if other_id == concept_id { continue; }
            let jaccard = self.graph.jaccard_atom_sets(concept_id, other_id);
            if jaccard > 0.0 {
                related_nodes.push((other_id, jaccard));
            }
        }
        related_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        related_nodes.truncate(20);

        // Find related edges involving this concept
        let mut related_edges: Vec<(NodeId, NodeId, f32)> = Vec::new();

        // Outgoing edges from concept
        for e in self.graph.edges_from(concept_id) {
            related_edges.push((e.from, e.to, e.weight));
        }

        // Incoming edges to concept
        for (&from_id, edges) in &self.graph.edges {
            if from_id == concept_id { continue; }
            for e in edges {
                if e.to == concept_id {
                    related_edges.push((e.from, e.to, e.weight));
                }
            }
        }

        // Also add edges from top related nodes
        for &(node_id, _) in &related_nodes {
            for e in self.graph.edges_from(node_id) {
                if !related_edges.iter().any(|(f, t, _)| *f == e.from && *t == e.to) {
                    related_edges.push((e.from, e.to, e.weight));
                }
            }
        }

        related_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        related_edges.truncate(30);

        Some(RelateResult {
            related_nodes,
            related_edges,
        })
    }

    // -----------------------------------------------------------------------
    // Status report
    // -----------------------------------------------------------------------

    pub fn status(&self) -> PipelineStatus {
        PipelineStatus {
            total_nodes:      self.graph.node_count(),
            total_atoms:      self.token_to_id.len(),
            total_contexts:   self.total_contexts,
            warmed_up:        self.autonomy.is_warmed_up(),
            watchlist_count:  self.autonomy.watchlist_len(),
            changelog_count:  self.autonomy.changelog_len(),
            theta_assign:     self.autonomy.current_theta_assign(),
            theta_merge:      self.autonomy.current_theta_merge(),
        }
    }
}

#[derive(Debug)]
pub struct PipelineStatus {
    pub total_nodes:     usize,
    pub total_atoms:     usize,
    pub total_contexts:  usize,
    pub warmed_up:       bool,
    pub watchlist_count: usize,
    pub changelog_count: usize,
    pub theta_assign:    f32,
    pub theta_merge:     f32,
}
