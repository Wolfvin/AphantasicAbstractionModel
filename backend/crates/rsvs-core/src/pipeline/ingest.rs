//! Ingest pipeline — RSVS v5.0 Compositional Architecture
//!
//! Contains `ingest_text()`, `update_node_atoms()`, and all ingest-related helpers.
//!
//! v5.0: When a new sense is induced from text, the system identifies which
//! (ID, sense) pairs are active in the context and uses them as the
//! compositions of the new sense. This is the INDUCTION mechanism.

use super::Rsvs;
use crate::attention::{is_groundable_to_seeds, text_to_sentences};
use crate::autonomy::ConfidenceUpdateResult;
use crate::error::RsvsError;
use crate::sense::IngestResult;
use crate::types::{
    CompressionState, Edge, EdgeSource, Node, NodeId, NodeStatus, PolicyMeta, SemanticMeta, Tier,
};
use rayon::prelude::*;

// -----------------------------------------------------------------------
// IngestStats — what happened during one ingest call
// -----------------------------------------------------------------------

/// Statistics returned from a single `ingest_text()` call.
#[derive(Debug, Default)]
pub struct IngestStats {
    /// Number of sentences processed.
    pub sentences_processed: usize,
    /// Number of atoms promoted to nodes.
    pub atoms_promoted: usize,
    /// Number of sense assignments to existing senses.
    pub sense_assigned: usize,
    /// Number of new senses created.
    pub sense_created: usize,
    /// Number of confidence updates applied.
    pub confidence_updated: usize,
    /// Number of watchlist additions.
    pub watchlist_additions: usize,
    /// Number of frozen batches (rollback triggered).
    pub frozen_batches: usize,
    /// Number of compositions induced (v5.0).
    pub compositions_induced: usize,
}

impl Rsvs {
    /// Ingest raw text into the knowledge graph.
    ///
    /// Runs the full pipeline: tokenize → co-occurrence → entity detection →
    /// node promotion → attention scoring → sense induction (with compositions) →
    /// confidence update → stability check.
    ///
    /// v5.0: When new senses are created, their compositions are induced from
    /// the active senses of context tokens. This is the INDUCTION mechanism
    /// (Problem 1 from the architecture spec).
    pub fn ingest_text(&mut self, text: &str) -> Result<IngestStats, RsvsError> {
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
        if sentences.is_empty() {
            return Ok(stats);
        }

        // --- Step 1: Update co-occurrence statistics ---
        let seed_refs: Vec<&str> = self.config.seed_labels.iter().map(|s| s.as_str()).collect();
        let entity_records: Vec<Vec<(String, bool)>> = sentences
            .par_iter()
            .map(|tokens| {
                tokens
                    .iter()
                    .map(|token| {
                        let groundable = is_groundable_to_seeds(token, &seed_refs);
                        (token.clone(), groundable)
                    })
                    .collect()
            })
            .collect();

        for (tokens, records) in sentences.iter().zip(entity_records.iter()) {
            self.stats_db.ingest_sentence(tokens);
            stats.sentences_processed += 1;

            for (token, groundable) in records {
                self.entities.record(token, *groundable);
            }
        }

        // --- Step 2: Promote new entity candidates to nodes ---
        let candidates = self.entities.candidates(self.config.entity_promote_n);
        for token in &candidates {
            if self.token_to_id.contains_key(token.as_str()) {
                continue;
            }

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
                    layer: 0,
                    derived_from_node_ids: vec![],
                    compression_reason: None,
                },
                policy_meta: Some(PolicyMeta {
                    policy_version: "5.0".to_string(),
                    governance_score: 0.0,
                    candidate_evidence_pool: 0.0,
                    status_flip_count: 0,
                    seen_fingerprints: vec![],
                    last_seen_at: None,
                }),
                language_links: vec![],

                atoms: vec![],
                fingerprint: None,
            })?;

            self.autonomy.register(id, 0.50, Tier::Tier2);
            self.token_to_id.insert(token.clone(), id);
            self.atom_sets.insert(token.clone(), vec![id]);
            self.senses.insert(
                id,
                crate::sense::SenseManager::new(self.config.sense.clone()),
            );
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
                    "compression_state": "raw",
                    "layer": 0
                }),
            );
        }

        // --- Step 2b: Build/update node atom sets from co-occurrence ---
        self.update_node_atoms(&candidates, &correlation_id);

        // --- Step 3: For each sentence, run attention + induce senses with compositions ---
        self.autonomy.begin_batch();
        let snapshot = self.autonomy.snapshot();

        for tokens in &sentences {
            self.total_contexts += 1;
            self.autonomy.tick_context();

            let selected = self
                .attention
                .select(tokens, &self.stats_db, &self.atom_sets);

            // Build context node IDs for this sentence
            let context_node_ids: Vec<NodeId> = tokens
                .iter()
                .filter_map(|t| self.token_to_id.get(t.as_str()).copied())
                .collect();

            for token in tokens {
                let token_id = match self.token_to_id.get(token.as_str()) {
                    Some(&id) => id,
                    None => continue,
                };

                let context: Vec<NodeId> = if let Some(cands) = selected.get(token) {
                    cands
                        .iter()
                        .filter_map(|c| self.token_to_id.get(c.token.as_str()).copied())
                        .collect()
                } else {
                    tokens
                        .iter()
                        .filter(|t| *t != token)
                        .filter_map(|t| self.token_to_id.get(t.as_str()).copied())
                        .collect()
                };

                if context.is_empty() {
                    continue;
                }

                // --- v5.0: Build active senses for composition induction ---
                let active_senses: Vec<(NodeId, crate::types::SenseId)> = context
                    .iter()
                    .filter_map(|&ctx_id| {
                        self.active_sense_for_node(ctx_id, &context_node_ids)
                            .map(|(_, sense_id)| (ctx_id, sense_id))
                    })
                    .collect();

                // Compute the layer for this induced sense
                let context_ids: Vec<NodeId> = active_senses.iter().map(|(id, _)| *id).collect();
                let layer = self.compute_layer(&context_ids);

                let sense_mgr = self
                    .senses
                    .entry(token_id)
                    .or_insert_with(|| crate::sense::SenseManager::new(self.config.sense.clone()));

                // Use compositional induction if we have active senses
                let ingest_result = if active_senses.is_empty() {
                    // Fallback to basic ingest (no compositions available)
                    sense_mgr.ingest(context.clone())
                } else {
                    // Compositional induction
                    let result = sense_mgr.induce_sense(context.clone(), &active_senses, layer);
                    if matches!(result, IngestResult::Created(_)) {
                        stats.compositions_induced += active_senses.len();
                    }
                    result
                };

                let sense_event: Option<serde_json::Value> = match ingest_result {
                    IngestResult::Assigned(idx) => {
                        stats.sense_assigned += 1;
                        Some(serde_json::json!({
                            "id": token_id,
                            "sense_idx": idx,
                            "action": "assigned"
                        }))
                    }
                    IngestResult::Created(idx) => {
                        stats.sense_created += 1;
                        // Update node layer if this sense has a higher layer
                        if let Some(node) = self.graph.get_node_mut(token_id) {
                            if layer > node.semantic.layer {
                                node.semantic.layer = layer;
                            }
                            if layer > 0 {
                                node.semantic.compression_state = CompressionState::Compressed;
                                node.semantic.derived_from_node_ids = context_ids.clone();
                                node.semantic.compression_reason =
                                    Some("compositional induction".to_string());
                            }
                        }
                        Some(serde_json::json!({
                            "id": token_id,
                            "sense_idx": idx,
                            "action": "created",
                            "layer": layer,
                            "compositions": active_senses.len()
                        }))
                    }
                };

                let active_coherence = sense_mgr.senses.first().map(|s| s.coherence).unwrap_or(0.5);

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
                if let Some(payload) = sense_event {
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
                            _ => 3,
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
            if self.total_contexts.is_multiple_of(20) {
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
                "compositions_induced": stats.compositions_induced
            }),
        );
        Ok(stats)
    }

    // -----------------------------------------------------------------------
    // Build/update node atom sets from co-occurrence data
    // -----------------------------------------------------------------------

    /// Update node atom sets from co-occurrence data.
    pub(super) fn update_node_atoms(&mut self, tokens: &[String], correlation_id: &str) {
        for token in tokens {
            let token_id = match self.token_to_id.get(token.as_str()) {
                Some(&id) => id,
                None => continue,
            };

            if let Some(node) = self.graph.get_node(token_id) {
                if node.is_seed {
                    continue;
                }
                if node.semantic.compression_state == CompressionState::Compressed
                    && !node.atoms.is_empty()
                {
                    continue;
                }
            }

            let mut cooc_nodes: Vec<(NodeId, f32)> = self
                .token_to_id
                .iter()
                .filter(|(t, _)| t.as_str() != token.as_str())
                .filter_map(|(t, &id)| {
                    let c = self.stats_db.cooc(token, t);
                    if c > 0.15 {
                        Some((id, c))
                    } else {
                        None
                    }
                })
                .collect();

            if cooc_nodes.is_empty() {
                continue;
            }

            cooc_nodes.sort_by(|a, b| b.1.total_cmp(&a.1));
            cooc_nodes.truncate(8);

            let atom_ids: Vec<NodeId> = cooc_nodes.iter().map(|(id, _)| *id).collect();

            // Compute layer BEFORE mutable borrow of graph
            let layer = if !atom_ids.is_empty() {
                self.compute_layer(&atom_ids)
            } else {
                0
            };

            if let Some(node) = self.graph.nodes.get_mut(&token_id) {
                let old_atoms = node.atoms.clone();
                node.atoms = atom_ids.clone();

                if !atom_ids.is_empty() {
                    node.semantic.compression_state = CompressionState::Compressed;
                    node.semantic.derived_from_node_ids = atom_ids.clone();
                    node.semantic.compression_reason =
                        Some("co-occurrence aggregation".to_string());
                    node.semantic.layer = layer;
                }

                self.atom_sets.insert(token.clone(), atom_ids.clone());

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
                    if let Some(existing) = edges
                        .iter_mut()
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
}
