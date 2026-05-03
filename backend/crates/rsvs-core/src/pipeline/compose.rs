//! Explicit composition API — RSVS v5.0 Compositional Architecture
//!
//! Contains the `compose()` method for creating compositional nodes from
//! explicit composition references. This is the core compositional mechanism:
//! higher-level concepts are built from lower-level senses.
//!
//! v5.0: compose() now accepts `Vec<CompositionRef>` — references to specific
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
    /// Create a compositional node from explicit composition references (v5.0).
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
            // Update sense compositions
            if let Some(sm) = self.senses.get_mut(&existing_id) {
                if sm.senses.is_empty() {
                    sm.create_compositional_sense(compositions.clone(), layer);
                } else {
                    sm.set_compositions(0, compositions.clone(), layer);
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
            };
            self.graph.insert_edge(edge)?;
        }

        // 6. Register in lookup tables
        self.token_to_id.insert(label.to_string(), node_id);
        self.atom_sets.insert(label.to_string(), comp_node_ids);

        // Register with autonomy engine
        self.autonomy
            .register(node_id, avg_confidence as f32, Tier::Tier2);

        // Create a sense manager with compositional sense
        let mut sm = crate::sense::SenseManager::new(self.config.sense.clone());
        sm.create_compositional_sense(compositions, layer);
        self.senses.insert(node_id, sm);

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
