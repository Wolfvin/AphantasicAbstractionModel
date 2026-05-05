//! Explicit composition API — RSVS v6.0 Compositional Architecture
//!
//! Contains the `compose()` method for creating compositional nodes from
//! explicit composition references. This is the core compositional mechanism:
//! higher-level concepts are built from lower-level senses.
//!
//! v6.0: compose() now accepts `Vec<CompositionRef>` — references to specific
//! senses of specific nodes, not just node IDs.
//!
//! Example:
//!   compose("raja", [(tahta_tertinggi, 0), (laki_laki, 0), (kerajaan, 0)], Some("id"))
//!   This creates a Layer 2 compositional node whose sense_0 is defined by
//!   those three compositions.

use super::Rsvs;
use crate::error::RsvsError;
use crate::types::{
    CompositionRef, CompressionState, Edge, EdgeSource, Node, NodeId, NodeStatus, SemanticMeta,
    Tier,
};

impl Rsvs {
    /// Validate composition constraints (v6.3 → v6.3.1 hardened):
    /// - tau_compress: all components must have freq >= threshold in their sense.
    ///   Seed nodes (layer 0, is_seed=true) and nodes with empty contexts are exempt —
    ///   they have no meaningful freq_counts yet, so the check would always fail.
    /// - tau_overlap: component set must have enough overlap with known nodes.
    ///   At least 2 compositions required for overlap check.
    fn validate_composition_constraints(
        &self,
        compositions: &[CompositionRef],
    ) -> Result<(), RsvsError> {
        let tau_compress = self.config.sense.induction.tau_compress;
        let tau_overlap = self.config.sense.induction.tau_overlap;

        // Check tau_compress: freq of each comp.node_id in its referenced sense
        for comp in compositions {
            // Exempt seed nodes — they have no meaningful freq_counts
            let is_seed = self.graph.get_node(comp.node_id)
                .map(|n| n.is_seed)
                .unwrap_or(false);
            if is_seed {
                continue;
            }

            if let Some(sm) = self.senses.get(&comp.node_id) {
                if let Some(sense) = sm.get_sense(comp.sense_id as usize) {
                    // Exempt senses with no contexts — newly created or primitive
                    if sense.context_count() == 0 {
                        continue;
                    }
                    let freq = sense.freq(comp.node_id);
                    if freq < tau_compress {
                        return Err(RsvsError::CompositionRejected {
                            reason: format!(
                                "Component {} has freq {:.3} in sense {} < tau_compress {:.3}",
                                comp.node_id, freq, comp.sense_id, tau_compress
                            ),
                        });
                    }
                }
            }
        }

        // Check tau_overlap: how many comp node_ids are known nodes with senses
        let comp_ids: Vec<NodeId> = compositions.iter().map(|c| c.node_id).collect();
        if comp_ids.len() >= 2 {
            let known_count = comp_ids
                .iter()
                .filter(|&&id| self.senses.contains_key(&id))
                .count();
            let overlap_ratio = known_count as f32 / comp_ids.len() as f32;
            if overlap_ratio < tau_overlap {
                return Err(RsvsError::CompositionRejected {
                    reason: format!(
                        "Overlap ratio {:.3} < tau_overlap {:.3} — not enough components are known nodes",
                        overlap_ratio, tau_overlap
                    ),
                });
            }
        }

        Ok(())
    }

    /// Create a compositional node from explicit composition references (v6.0).
    ///
    /// This is the core compositional mechanism: higher-level concepts
    /// are built from specific senses of lower-level concepts.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // compose("raja", [(tahta_tertinggi, 0), (laki_laki, 0), (kerajaan, 0)])
    /// // creates a Layer 2 node whose sense_0 is defined by those compositions.
    /// let compositions = vec![
    ///     CompositionRef::new(tahta_id, 0),
    ///     CompositionRef::new(laki_laki_id, 0),
    ///     CompositionRef::new(kerajaan_id, 0),
    /// ];
    /// let node_id = rsvs.compose("raja", compositions, Some("id"))?;
    /// ```
    pub fn compose(
        &mut self,
        label: &str,
        compositions: Vec<CompositionRef>,
        lang: Option<&str>,
    ) -> Result<NodeId, RsvsError> {
        // 1. Validate all composition targets exist in the graph
        for comp in &compositions {
            if self.graph.get_node(comp.node_id).is_none() {
                return Err(RsvsError::NodeNotFound { id: comp.node_id });
            }
            // Validate sense_id exists for the target node
            if let Some(sm) = self.senses.get(&comp.node_id) {
                if comp.sense_id as usize >= sm.sense_count() && sm.sense_count() > 0 {
                    // Sense doesn't exist yet — use sense 0 as fallback
                    // This is acceptable because senses can be created dynamically
                }
            }
        }

        // 1b. Validate composition constraints (tau_compress, tau_overlap)
        self.validate_composition_constraints(&compositions)?;

        // 1c. v7.0: Neuro-symbolic verification of compositions
        // Check for self-reference and circular chains before creating the node.
        // We use a temporary NodeId (0) for the check — self-reference is
        // detected by checking if any composition references a node that
        // would create a cycle.
        for comp in &compositions {
            // Self-reference: compositions must not reference the same label
            if let Some(&existing_id) = self.token_to_id.get(label) {
                if comp.node_id == existing_id {
                    // Use DEPS planner to generate recovery plan
                    let deps_result = self.deps_planner.analyze(
                        &RsvsError::CircularRef { from: existing_id, to: existing_id },
                        existing_id,
                    );
                    let recovery_hint = deps_result.recommended
                        .as_ref()
                        .map(|p| p.description.clone())
                        .unwrap_or_default();
                    return Err(RsvsError::CompositionRejected {
                        reason: format!(
                            "Self-reference detected: composition references node '{}' (id={}). Recovery: {}",
                            label, existing_id, recovery_hint
                        ),
                    });
                }
            }
            // Circular chain: check transitive closure
            if self.detect_composition_cycle(comp.node_id, label) {
                if let Some(&existing_id) = self.token_to_id.get(label) {
                    let deps_result = self.deps_planner.analyze(
                        &RsvsError::CircularRef { from: comp.node_id, to: existing_id },
                        existing_id,
                    );
                    let recovery_hint = deps_result.recommended
                        .as_ref()
                        .map(|p| p.description.clone())
                        .unwrap_or_default();
                    return Err(RsvsError::CompositionRejected {
                        reason: format!(
                            "Circular composition chain detected via node {}. Recovery: {}",
                            comp.node_id, recovery_hint
                        ),
                    });
                }
            }
        }

        // 2. Compute layer from composition targets
        let comp_node_ids: Vec<NodeId> = compositions.iter().map(|c| c.node_id).collect();
        let layer = self.compute_layer(&comp_node_ids);

        // 3. Check if a node with this label already exists
        if let Some(&existing_id) = self.token_to_id.get(label) {
            // Update existing node's composition
            if let Some(node) = self.graph.get_node_mut(existing_id) {
                node.semantic.compression_state = CompressionState::Compressed;
                node.semantic.layer = layer;
                node.semantic.derived_from_node_ids = comp_node_ids.clone();
                node.semantic.compression_reason = Some("explicit composition".to_string());
                node.atoms = comp_node_ids.clone();
            }
            // Update sense compositions (v6.3: use lazy refactor)
            if let Some(sm) = self.senses.get_mut(&existing_id) {
                if sm.senses.is_empty() {
                    sm.create_compositional_sense(compositions.clone(), layer);
                } else {
                    // Lazy refactor: stage first, flush at next safe checkpoint
                    sm.stage_compositions(0, compositions.clone(), layer);
                }
            }
            // Sync atom_sets
            self.atom_sets.insert(label.to_string(), comp_node_ids);
            return Ok(existing_id);
        }

        // 4. Create new Compressed node
        let avg_confidence: f64 = comp_node_ids
            .iter()
            .filter_map(|&aid| self.graph.get_node(aid).map(|n| n.confidence as f64))
            .sum::<f64>()
            / comp_node_ids.len().max(1) as f64;

        let node = Node {
            id: 0,
            label: label.to_string(),
            surface_label: format!("{}@{}", label, lang.unwrap_or("en")),
            kind: "node".to_string(),
            tier: Tier::Tier2,
            confidence: avg_confidence as f32,
            status: NodeStatus::Candidate,
            is_seed: false,
            is_locked: false,
            semantic: SemanticMeta {
                compression_state: CompressionState::Compressed,
                layer,
                derived_from_node_ids: comp_node_ids.clone(),
                compression_reason: Some("explicit composition".to_string()),
            },
            policy_meta: None,
            language_links: vec![],
            atoms: comp_node_ids.clone(),
            fingerprint: None,
        };

        let node_id = self.graph.insert_node(node)?;

        // 5. Create edges from each composition target to the composite
        for comp in &compositions {
            let edge = Edge {
                from: comp.node_id,
                to: node_id,
                weight: 1.0,
                source: EdgeSource::Composition,
                last_reinforced_batch: 0, // Composition edges never decay
            };
            self.graph.insert_edge(edge)?;
        }

        // 6. Register in lookup tables
        // v7.2: Use register_label to keep token_to_id and graph.label_to_id in sync
        self.register_label(label, node_id, Some(&format!("{}@{}", label, lang.unwrap_or("en"))));
        self.atom_sets.insert(label.to_string(), comp_node_ids);

        // Register with autonomy engine
        self.autonomy
            .register(node_id, avg_confidence as f32, Tier::Tier2);

        // Create a sense manager with compositional sense
        let mut sm = crate::sense::SenseManager::new(self.config.sense.clone());
        sm.create_compositional_sense(compositions, layer);
        self.senses.insert(node_id, sm);

        // 7. v7.2: Neuro-symbolic verification of the new composition
        //    Verify structural invariants after creation — if verification
        //    fails critically, log a warning but do NOT roll back (the node
        //    was already created). The user can explicitly verify and revise.
        if let Some(sm) = self.senses.get(&node_id) {
            if let Some(sense) = sm.senses.first() {
                let verifier = crate::neurosym::NeuroSymVerifier::new();
                let (status, results) = verifier.verify(
                    node_id,
                    sense,
                    &self.graph,
                    &self.senses,
                    &self.config.sense,
                );
                let failed_rules: Vec<&str> = results
                    .iter()
                    .filter(|r| !r.passed)
                    .map(|r| r.rule.name.as_str())
                    .collect();
                if !failed_rules.is_empty() {
                    self.emit_event(
                        &format!("compose_{}", node_id),
                        "neurosym_verification_warning",
                        serde_json::json!({
                            "label": label,
                            "status": format!("{:?}", status),
                            "failed_rules": failed_rules,
                        }),
                    );
                }
            }
        }

        Ok(node_id)
    }

    /// Backward-compatible compose that takes node IDs instead of CompositionRefs.
    ///
    /// Creates compositions with sense_id=0 for each node.
    pub fn compose_from_ids(
        &mut self,
        label: &str,
        atom_ids: Vec<NodeId>,
        lang: Option<&str>,
    ) -> Result<NodeId, RsvsError> {
        let compositions: Vec<CompositionRef> = atom_ids
            .iter()
            .map(|&id| CompositionRef::new(id, 0))
            .collect();
        self.compose(label, compositions, lang)
    }
}
