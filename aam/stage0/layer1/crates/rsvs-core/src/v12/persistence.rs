//! # Persistence — Save/Load Graph to JSON
//!
//! Ported from v8.3 `persist.rs` (961 lines), simplified for v12.
//!
//! The v12 graph is entirely serializable via `serde`, so persistence
//! is straightforward: serialize the `Graph` to JSON and deserialize
//! it back.
//!
//! ## Usage
//!
//! ```ignore
//! let persistence = Persistence::new();
//! persistence.save(&graph, "/path/to/graph.json")?;
//! let loaded = persistence.load("/path/to/graph.json")?;
//! ```
//!
//! ## Format
//!
//! The JSON format includes:
//! - `nodes`: HashMap<NodeId, Node>
//! - `compositions`: HashMap<CompositionId, Composition>
//! - `edges`: Vec<(CompositionId, NodeId, SemanticEdge)>
//! - `label_to_id`: HashMap<String, NodeId>
//! - `next_id`: NodeId

use std::fs;
use std::io;
use std::path::Path;

use super::pipeline::Graph;

// ========================================================================
// PersistenceError — Errors During Save/Load
// ========================================================================

/// Errors that can occur during persistence operations.
#[derive(Debug)]
pub enum PersistenceError {
    /// I/O error (file not found, permission denied, etc.).
    Io(io::Error),
    /// Serialization error (JSON encoding failed).
    Serialization(String),
    /// Deserialization error (JSON decoding failed).
    Deserialization(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistenceError::Io(e) => write!(f, "I/O error: {}", e),
            PersistenceError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            PersistenceError::Deserialization(msg) => write!(f, "Deserialization error: {}", msg),
        }
    }
}

impl std::error::Error for PersistenceError {}

impl From<io::Error> for PersistenceError {
    fn from(e: io::Error) -> Self {
        PersistenceError::Io(e)
    }
}

// ========================================================================
// Persistence — Save/Load Engine
// ========================================================================

/// Persistence engine for the v12 graph.
///
/// Provides save/load functionality using JSON serialization.
/// The entire graph (nodes, compositions, edges, indices) is
/// serialized as a single JSON object.
pub struct Persistence {
    /// Whether to pretty-print the JSON output.
    pub pretty_print: bool,
}

impl Default for Persistence {
    fn default() -> Self {
        Self::new()
    }
}

impl Persistence {
    /// Create a new persistence engine.
    pub fn new() -> Self {
        Self { pretty_print: true }
    }

    /// Save the graph to a JSON file.
    ///
    /// Creates the file and any parent directories if they don't exist.
    pub fn save(&self, graph: &Graph, path: &Path) -> Result<(), PersistenceError> {
        // Create parent directories if needed.
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let json = if self.pretty_print {
            serde_json::to_string_pretty(graph)
                .map_err(|e| PersistenceError::Serialization(e.to_string()))?
        } else {
            serde_json::to_string(graph)
                .map_err(|e| PersistenceError::Serialization(e.to_string()))?
        };

        fs::write(path, json)?;
        Ok(())
    }

    /// Load a graph from a JSON file.
    pub fn load(&self, path: &Path) -> Result<Graph, PersistenceError> {
        let json = fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| PersistenceError::Deserialization(e.to_string()))
    }

    /// Serialize the graph to a JSON string.
    pub fn to_json(&self, graph: &Graph) -> Result<String, PersistenceError> {
        if self.pretty_print {
            serde_json::to_string_pretty(graph)
                .map_err(|e| PersistenceError::Serialization(e.to_string()))
        } else {
            serde_json::to_string(graph).map_err(|e| PersistenceError::Serialization(e.to_string()))
        }
    }

    /// Deserialize a graph from a JSON string.
    pub fn from_json(&self, json: &str) -> Result<Graph, PersistenceError> {
        serde_json::from_str(json).map_err(|e| PersistenceError::Deserialization(e.to_string()))
    }

    /// Get graph statistics without loading the full graph.
    ///
    /// Useful for quick inspections of saved graphs.
    pub fn stats(path: &Path) -> Result<GraphStats, PersistenceError> {
        let json = fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| PersistenceError::Deserialization(e.to_string()))?;

        let node_count = value
            .get("nodes")
            .and_then(|v| v.as_object())
            .map(|o| o.len())
            .unwrap_or(0);

        let composition_count = value
            .get("compositions")
            .and_then(|v| v.as_object())
            .map(|o| o.len())
            .unwrap_or(0);

        let edge_count = value
            .get("edges")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        Ok(GraphStats {
            node_count,
            composition_count,
            edge_count,
        })
    }
}

/// Quick statistics about a saved graph.
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Number of nodes in the graph.
    pub node_count: usize,
    /// Number of compositions.
    pub composition_count: usize,
    /// Number of edges.
    pub edge_count: usize,
}

// ========================================================================
// v8.3 → v12 Migration
// ========================================================================

/// Result of a v8.3 → v12 migration.
#[derive(Debug, Clone)]
pub struct MigrationResult {
    /// Number of v8.3 nodes migrated.
    pub nodes_migrated: usize,
    /// Number of v8.3 edges migrated.
    pub edges_migrated: usize,
    /// Number of v8.3 senses converted to compositions.
    pub senses_converted: usize,
    /// Number of nodes that could not be migrated.
    pub nodes_skipped: usize,
    /// Warnings encountered during migration.
    pub warnings: Vec<String>,
}

impl Persistence {
    /// Migrate a v8.3 snapshot JSON to a v12 Graph.
    ///
    /// v8.3 snapshots have a different structure:
    /// - `nodes` as a flat array of objects with `id`, `label`, `status`,
    ///   `observation_count`, and optionally `surface_label`, `is_seed`,
    ///   `senses`, `confidence`
    /// - `edges` as an array of objects with `from`, `to`, `weight`,
    ///   `source` fields, or as tuples `(from, to, weight)`
    /// - No compositions (senses become compositions in v12)
    ///
    /// This function performs a best-effort migration:
    /// 1. Each v8.3 node becomes a v12 Node (label, surface_label, lifecycle
    ///    from old NodeStatus, confidence from old observation_count normalized)
    /// 2. Co-occurring nodes (connected by edges) become v12 Event compositions
    /// 3. Each v8.3 sense becomes a v12 Composition
    /// 4. v8.3 edges become v12 SemanticEdges
    /// 5. Seed atoms get Stable lifecycle and EpistemicState::Grounded
    ///
    /// If the input is already a v12 graph, it is loaded directly.
    pub fn migrate_v83(path: &Path) -> Result<(Graph, MigrationResult), PersistenceError> {
        let json = fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| PersistenceError::Deserialization(e.to_string()))?;

        // Check if already v12 format
        if value.get("schema_version").and_then(|v| v.as_str()) == Some("v12") {
            let graph: Graph = serde_json::from_value(value)
                .map_err(|e| PersistenceError::Deserialization(e.to_string()))?;
            return Ok((graph, MigrationResult {
                nodes_migrated: 0,
                edges_migrated: 0,
                senses_converted: 0,
                nodes_skipped: 0,
                warnings: vec!["Input is already v12 format — no migration needed".to_string()],
            }));
        }

        let mut graph = Graph::new();
        let mut result = MigrationResult {
            nodes_migrated: 0,
            edges_migrated: 0,
            senses_converted: 0,
            nodes_skipped: 0,
            warnings: Vec::new(),
        };

        // Migrate nodes from v8.3 format.
        // Support both: nodes as array of objects OR nodes as object with NodeId keys.
        if let Some(nodes_val) = value.get("nodes") {
            // Try as object (flat map with NodeId keys) first.
            if let Some(nodes_obj) = nodes_val.as_object() {
                for (node_id_str, node_val) in nodes_obj {
                    if let Some(node_obj) = node_val.as_object() {
                        let label = node_obj.get("label")
                            .and_then(|v| v.as_str())
                            .unwrap_or(node_id_str)
                            .to_string();

                        if label.is_empty() {
                            result.nodes_skipped += 1;
                            continue;
                        }

                        let node_id = graph.ensure_node(&label);
                        if let Some(atom) = graph.nodes.get_mut(&node_id) {
                            // Migrate confidence from observation_count (normalized to 0-1).
                            if let Some(obs_count) = node_obj.get("observation_count").and_then(|v| v.as_u64()) {
                                atom.confidence = (obs_count as f32 / 10.0).min(1.0);
                            } else if let Some(conf) = node_obj.get("confidence").and_then(|v| v.as_f64()) {
                                atom.confidence = conf as f32;
                            }

                            // Migrate lifecycle from NodeStatus / status field.
                            let status_str = node_obj.get("status")
                                .or_else(|| node_obj.get("node_status"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("New");
                            atom.lifecycle = match status_str {
                                "Stable" | "Grounded" | "Tier1" => super::types::LifecycleState::Stable,
                                "Candidate" | "Tier2" => super::types::LifecycleState::Candidate,
                                "Deprecated" | "Refuted" => super::types::LifecycleState::Deprecated,
                                "Quarantine" => super::types::LifecycleState::Quarantine,
                                _ => super::types::LifecycleState::New,
                            };

                            // Override for seed nodes.
                            let is_seed = node_obj.get("is_seed")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            if is_seed {
                                atom.lifecycle = super::types::LifecycleState::Stable;
                                atom.confidence = atom.confidence.max(0.8);
                            }

                            // Migrate surface_label.
                            if let Some(surface) = node_obj.get("surface_label").and_then(|v| v.as_str()) {
                                atom.surface_label = surface.to_string();
                            }
                        }

                        result.nodes_migrated += 1;

                        // Migrate senses as compositions.
                        if let Some(senses) = node_obj.get("senses").and_then(|v| v.as_array()) {
                            for (sense_idx, sense_val) in senses.iter().enumerate() {
                                if let Some(sense_obj) = sense_val.as_object() {
                                    let comp_id = super::types::CompositionId::new(format!("{}_sense_{}", label, sense_idx));
                                    let mut comp = super::types::Composition::default();
                                    comp.id = comp_id.clone();
                                    comp.composition_type = super::types::CompositionType::Hypothesis;
                                    comp.lifecycle = super::types::LifecycleState::Candidate;
                                    comp.epistemic = super::types::EpistemicState::Inferred;
                                    comp.provenance = super::types::ProvenanceChain::default();
                                    comp.provenance.origin_id = format!("v83_sense_{}_{}", sense_idx, label);

                                    if let Some(coherence) = sense_obj.get("coherence").and_then(|v| v.as_f64()) {
                                        comp.confidence = coherence as f32;
                                    }

                                    // Migrate compositions as members.
                                    if let Some(comps) = sense_obj.get("compositions").and_then(|v| v.as_array()) {
                                        for comp_ref in comps {
                                            if let Some(ref_obj) = comp_ref.as_object() {
                                                let target_label = ref_obj.get("node_id")
                                                    .or_else(|| ref_obj.get("label"))
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("")
                                                    .to_string();

                                                if !target_label.is_empty() {
                                                    let target_id = graph.ensure_node(&target_label);
                                                    comp.members.push(super::types::CompositionMember {
                                                        node_id: target_id,
                                                        role: super::types::SemanticRole::Arg1Patient,
                                                        confidence: 0.7,
                                                        label: target_label,
                                                        source: None,
                                                    });
                                                }
                                            }
                                        }
                                    }

                                    // Add node itself as a member.
                                    comp.members.push(super::types::CompositionMember {
                                        node_id,
                                        role: super::types::SemanticRole::Arg0Agent,
                                        confidence: 1.0,
                                        label: label.clone(),
                                        source: None,
                                    });

                                    graph.compositions.insert(comp_id, comp);
                                    result.senses_converted += 1;
                                }
                            }
                        }
                    }
                }
            }
            // Fallback: try as array of node objects.
            else if let Some(nodes_arr) = nodes_val.as_array() {
                for node_val in nodes_arr {
                    if let Some(node_obj) = node_val.as_object() {
                        let label = node_obj.get("label")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        if label.is_empty() {
                            result.nodes_skipped += 1;
                            continue;
                        }

                        let node_id = graph.ensure_node(&label);
                        if let Some(atom) = graph.nodes.get_mut(&node_id) {
                            if let Some(obs_count) = node_obj.get("observation_count").and_then(|v| v.as_u64()) {
                                atom.confidence = (obs_count as f32 / 10.0).min(1.0);
                            } else if let Some(conf) = node_obj.get("confidence").and_then(|v| v.as_f64()) {
                                atom.confidence = conf as f32;
                            }

                            let is_seed = node_obj.get("is_seed")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);

                            if is_seed {
                                atom.lifecycle = super::types::LifecycleState::Stable;
                            } else {
                                let conf = atom.confidence;
                                atom.lifecycle = if conf >= 0.7 {
                                    super::types::LifecycleState::Candidate
                                } else {
                                    super::types::LifecycleState::New
                                };
                            }

                            if let Some(surface) = node_obj.get("surface_label").and_then(|v| v.as_str()) {
                                atom.surface_label = surface.to_string();
                            }
                        }

                        result.nodes_migrated += 1;

                        // Migrate senses as compositions (same as above).
                        if let Some(senses) = node_obj.get("senses").and_then(|v| v.as_array()) {
                            for (sense_idx, sense_val) in senses.iter().enumerate() {
                                if let Some(sense_obj) = sense_val.as_object() {
                                    let comp_id = super::types::CompositionId::new(format!("{}_sense_{}", label, sense_idx));
                                    let mut comp = super::types::Composition::default();
                                    comp.id = comp_id.clone();
                                    comp.composition_type = super::types::CompositionType::Hypothesis;
                                    comp.lifecycle = super::types::LifecycleState::Candidate;
                                    comp.epistemic = super::types::EpistemicState::Inferred;
                                    comp.provenance = super::types::ProvenanceChain::default();
                                    comp.provenance.origin_id = format!("v83_sense_{}_{}", sense_idx, label);

                                    if let Some(coherence) = sense_obj.get("coherence").and_then(|v| v.as_f64()) {
                                        comp.confidence = coherence as f32;
                                    }

                                    if let Some(comps) = sense_obj.get("compositions").and_then(|v| v.as_array()) {
                                        for comp_ref in comps {
                                            if let Some(ref_obj) = comp_ref.as_object() {
                                                let target_label = ref_obj.get("node_id")
                                                    .or_else(|| ref_obj.get("label"))
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("")
                                                    .to_string();

                                                if !target_label.is_empty() {
                                                    let target_id = graph.ensure_node(&target_label);
                                                    comp.members.push(super::types::CompositionMember {
                                                        node_id: target_id,
                                                        role: super::types::SemanticRole::Arg1Patient,
                                                        confidence: 0.7,
                                                        label: target_label,
                                                        source: None,
                                                    });
                                                }
                                            }
                                        }
                                    }

                                    comp.members.push(super::types::CompositionMember {
                                        node_id,
                                        role: super::types::SemanticRole::Arg0Agent,
                                        confidence: 1.0,
                                        label: label.clone(),
                                        source: None,
                                    });

                                    graph.compositions.insert(comp_id, comp);
                                    result.senses_converted += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Migrate edges from v8.3 format and create Event compositions for co-occurring nodes.
        // Track which node pairs have edges to create Event compositions.
        let mut edge_pairs: std::collections::HashMap<(crate::types::NodeId, crate::types::NodeId), Vec<f32>> = std::collections::HashMap::new();

        if let Some(edges) = value.get("edges").and_then(|v| v.as_array()) {
            for edge_val in edges {
                // Try as object first.
                if let Some(edge_obj) = edge_val.as_object() {
                    let from_label = edge_obj.get("from_label")
                        .or_else(|| edge_obj.get("from"))
                        .and_then(|v| {
                            if v.is_number() { None } else { v.as_str() }
                        })
                        .unwrap_or("")
                        .to_string();

                    let to_label = edge_obj.get("to_label")
                        .or_else(|| edge_obj.get("to"))
                        .and_then(|v| {
                            if v.is_number() { None } else { v.as_str() }
                        })
                        .unwrap_or("")
                        .to_string();

                    if from_label.is_empty() || to_label.is_empty() {
                        continue;
                    }

                    let from_id = graph.ensure_node(&from_label);
                    let to_id = graph.ensure_node(&to_label);

                    let weight = edge_obj.get("weight")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0) as f32;

                    let source_str = edge_obj.get("source")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Learned");

                    let edge_source = match source_str {
                        "Bootstrap" => crate::types::EdgeSource::Bootstrap,
                        "Composition" => crate::types::EdgeSource::FrameCompiler,
                        _ => crate::types::EdgeSource::Learned,
                    };

                    // Track edge pair for Event composition creation.
                    let pair_key = if from_id < to_id { (from_id, to_id) } else { (to_id, from_id) };
                    edge_pairs.entry(pair_key).or_default().push(weight);

                    // v8.3 edges are node-to-node; v12 edges are composition-to-node.
                    // Create a synthetic composition for co-occurring nodes.
                    let comp_id = super::types::CompositionId::new(format!("migrated_edge_{}", result.edges_migrated));
                    let mut comp = super::types::Composition::default();
                    comp.id = comp_id.clone();
                    comp.composition_type = super::types::CompositionType::Event;
                    comp.confidence = weight;
                    comp.provenance.origin_id = format!("migrated_{}", from_label);
                    comp.members.push(super::types::CompositionMember {
                        node_id: from_id,
                        role: super::types::SemanticRole::Arg0Agent,
                        confidence: 1.0,
                        label: from_label.clone(),
                        source: None,
                    });
                    comp.members.push(super::types::CompositionMember {
                        node_id: to_id,
                        role: super::types::SemanticRole::Arg1Patient,
                        confidence: weight,
                        label: to_label,
                        source: None,
                    });
                    comp.provenance = super::types::ProvenanceChain::default();
                    graph.compositions.insert(comp_id.clone(), comp);

                    let edge = super::types::SemanticEdge {
                        relation: crate::types::RelationType::Categorical,
                        role: None,
                        source: edge_source,
                    };
                    graph.edges.push((comp_id.clone(), to_id, edge));
                    result.edges_migrated += 1;
                }
                // Try as tuple (from, to, weight).
                else if let Some(edge_arr) = edge_val.as_array() {
                    if edge_arr.len() >= 2 {
                        let from_str = edge_arr[0].as_str().unwrap_or("");
                        let to_str = edge_arr[1].as_str().unwrap_or("");
                        let weight = edge_arr.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

                        if from_str.is_empty() || to_str.is_empty() {
                            continue;
                        }

                        let from_id = graph.ensure_node(from_str);
                        let to_id = graph.ensure_node(to_str);

                        let pair_key = if from_id < to_id { (from_id, to_id) } else { (to_id, from_id) };
                        edge_pairs.entry(pair_key).or_default().push(weight);

                        let comp_id = super::types::CompositionId::new(format!("migrated_edge_{}", result.edges_migrated));
                        let mut comp = super::types::Composition::default();
                        comp.id = comp_id.clone();
                        comp.composition_type = super::types::CompositionType::Event;
                        comp.confidence = weight;
                        comp.members.push(super::types::CompositionMember {
                            node_id: from_id,
                            role: super::types::SemanticRole::Arg0Agent,
                            confidence: 1.0,
                            label: from_str.to_string(),
                            source: None,
                        });
                        comp.members.push(super::types::CompositionMember {
                            node_id: to_id,
                            role: super::types::SemanticRole::Arg1Patient,
                            confidence: weight,
                            label: to_str.to_string(),
                            source: None,
                        });
                        graph.compositions.insert(comp_id.clone(), comp);

                        let edge = super::types::SemanticEdge {
                            relation: crate::types::RelationType::Categorical,
                            role: None,
                            source: crate::types::EdgeSource::Learned,
                        };
                        graph.edges.push((comp_id, to_id, edge));
                        result.edges_migrated += 1;
                    }
                }
            }
        }

        // Create Event compositions for co-occurring node pairs with multiple edges.
        // Nodes that appear together in multiple edges get a shared Event composition.
        for ((node_a, node_b), weights) in &edge_pairs {
            if weights.len() < 2 {
                continue; // Only create shared compositions for multi-edge pairs.
            }
            let avg_weight: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
            let comp_id = super::types::CompositionId::new(format!("migrated_cooc_{}_{}", node_a, node_b));
            let mut comp = super::types::Composition::default();
            comp.id = comp_id.clone();
            comp.composition_type = super::types::CompositionType::Event;
            comp.confidence = avg_weight;

            let label_a = graph.node_label(*node_a).unwrap_or("").to_string();
            let label_b = graph.node_label(*node_b).unwrap_or("").to_string();

            comp.members.push(super::types::CompositionMember {
                node_id: *node_a,
                role: super::types::SemanticRole::Arg0Agent,
                confidence: 1.0,
                label: label_a,
                source: None,
            });
            comp.members.push(super::types::CompositionMember {
                node_id: *node_b,
                role: super::types::SemanticRole::Arg1Patient,
                confidence: avg_weight,
                label: label_b,
                source: None,
            });

            // Add edges for this composition.
            graph.edges.push((
                comp_id.clone(),
                *node_a,
                super::types::SemanticEdge {
                    relation: crate::types::RelationType::Categorical,
                    role: Some(super::types::SemanticRole::Arg0Agent),
                    source: crate::types::EdgeSource::Learned,
                },
            ));
            graph.edges.push((
                comp_id.clone(),
                *node_b,
                super::types::SemanticEdge {
                    relation: crate::types::RelationType::Categorical,
                    role: Some(super::types::SemanticRole::Arg1Patient),
                    source: crate::types::EdgeSource::Learned,
                },
            ));

            graph.compositions.insert(comp_id, comp);
        }

        Ok((graph, result))
    }
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]
    use super::*;
    use crate::v12::types::*;

    #[test]
    fn test_save_load_roundtrip() {
        let persistence = Persistence::new();
        let mut graph = Graph::new();

        // Add some data.
        let node_a = graph.ensure_node("alpha");
        let node_b = graph.ensure_node("beta");

        let mut comp = Composition::default();
        comp.id = CompositionId::new("comp_test".to_string());
        comp.composition_type = CompositionType::Event;
        comp.confidence = 0.8;
        comp.members = vec![
            CompositionMember {
                node_id: node_a,
                role: SemanticRole::Arg0Agent,
                confidence: 0.9,
                label: "alpha".to_string(),
                source: None,
            },
            CompositionMember {
                node_id: node_b,
                role: SemanticRole::Arg1Patient,
                confidence: 0.8,
                label: "beta".to_string(),
                source: None,
            },
        ];
        graph.compositions.insert(CompositionId::new("comp_test".to_string()), comp);

        // Save to temp file.
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("v12_test_graph.json");

        persistence
            .save(&graph, &path)
            .expect("Save should succeed");
        let loaded = persistence.load(&path).expect("Load should succeed");

        // Verify roundtrip.
        assert_eq!(loaded.nodes.len(), 2);
        assert_eq!(loaded.compositions.len(), 1);
        assert_eq!(loaded.edges.len(), 0);

        // Clean up.
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_json_roundtrip() {
        let persistence = Persistence::new();
        let mut graph = Graph::new();
        graph.ensure_node("test_node");

        let json = persistence.to_json(&graph).expect("Serialize should work");
        let loaded = persistence
            .from_json(&json)
            .expect("Deserialize should work");

        assert_eq!(loaded.nodes.len(), 1);
    }

    #[test]
    fn test_stats() {
        let persistence = Persistence::new();
        let mut graph = Graph::new();
        graph.ensure_node("a");
        graph.ensure_node("b");

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("v12_test_stats.json");

        persistence
            .save(&graph, &path)
            .expect("Save should succeed");
        let stats = Persistence::stats(&path).expect("Stats should work");

        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.composition_count, 0);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let persistence = Persistence::new();
        let result = persistence.load(Path::new("/nonexistent/path/graph.json"));
        assert!(result.is_err());
    }
}
