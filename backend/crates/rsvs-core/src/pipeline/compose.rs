//! Explicit composition API — RSVS v4.2
//!
//! Contains the `compose()` method for creating composite nodes from
//! explicit atom IDs. This is the core compositional mechanism:
//! higher-level concepts are built from lower-level atoms.

use super::Rsvs;
use crate::error::RsvsError;
use crate::types::{CompressionState, Edge, EdgeSource, Node, NodeId, NodeStatus, SemanticMeta, Tier};

impl Rsvs {
    /// Create a composite node from explicit atom IDs.
    ///
    /// This is the core compositional mechanism: higher-level concepts
    /// are built from lower-level atoms.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // compose("raja", [tahta_id, laki_laki_id, kerajaan_id])
    /// // creates a Compressed node whose derived_from_node_ids are those atoms.
    /// let node_id = rsvs.compose("raja", vec![tahta_id, laki_laki_id, kerajaan_id], Some("id"))?;
    /// ```
    pub fn compose(
        &mut self,
        label: &str,
        atom_ids: Vec<NodeId>,
        lang: Option<&str>,
    ) -> Result<NodeId, RsvsError> {
        // 1. Validate all atom_ids exist in the graph
        for &aid in &atom_ids {
            if self.graph.get_node(aid).is_none() {
                return Err(RsvsError::NodeNotFound { id: aid });
            }
        }

        // 2. Check if a node with this label already exists
        if let Some(&existing_id) = self.token_to_id.get(label) {
            // Update existing node's composition
            if let Some(node) = self.graph.get_node_mut(existing_id) {
                node.atoms = atom_ids.clone();
                node.semantic.compression_state = CompressionState::Compressed;
                node.semantic.derived_from_node_ids = atom_ids.clone();
                node.semantic.compression_reason = Some("explicit composition".to_string());
            }
            // Sync atom_sets
            self.atom_sets.insert(label.to_string(), atom_ids);
            return Ok(existing_id);
        }

        // 3. Create new Compressed node
        // Initial confidence from average of atom confidences
        let avg_confidence: f64 = atom_ids
            .iter()
            .filter_map(|&aid| self.graph.get_node(aid).map(|n| n.confidence as f64))
            .sum::<f64>()
            / atom_ids.len().max(1) as f64;

        let node = Node {
            id: 0, // will be assigned by insert_node
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
                derived_from_node_ids: atom_ids.clone(),
                compression_reason: Some("explicit composition".to_string()),
            },
            policy_meta: None,
            language_links: vec![],
            atoms: atom_ids.clone(),
            fingerprint: None,
        };

        let node_id = self.graph.insert_node(node)?;

        // 4. Create edges from each atom to the composite
        for &aid in &atom_ids {
            let edge = Edge {
                from: aid,
                to: node_id,
                weight: 1.0,
                source: EdgeSource::Composition,
            };
            self.graph.insert_edge(edge)?;
        }

        // 5. Register in lookup tables
        self.token_to_id.insert(label.to_string(), node_id);
        self.atom_sets.insert(label.to_string(), atom_ids);

        // Register with autonomy engine
        self.autonomy
            .register(node_id, avg_confidence as f32, Tier::Tier2);

        // Create a sense manager for the new node
        self.senses.insert(
            node_id,
            crate::sense::SenseManager::new(self.config.sense.clone()),
        );

        Ok(node_id)
    }
}
