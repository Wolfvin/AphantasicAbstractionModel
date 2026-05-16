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
        comp.id = "comp_test".to_string();
        comp.composition_type = CompositionType::Event;
        comp.confidence = 0.8;
        comp.members = vec![
            CompositionMember {
                node_id: node_a,
                role: SemanticRole::Arg0Agent,
                confidence: 0.9,
                label: "alpha".to_string(),
            },
            CompositionMember {
                node_id: node_b,
                role: SemanticRole::Arg1Patient,
                confidence: 0.8,
                label: "beta".to_string(),
            },
        ];
        graph.compositions.insert("comp_test".to_string(), comp);

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
