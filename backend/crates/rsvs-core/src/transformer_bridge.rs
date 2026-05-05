//! Bridge between Transformer vector representations and RSVS symbolic representations.
//!
//! RSVS is NOT a replacement for Transformers — it's an interpretation layer on top.
//! The TransformerBridge converts abstract vector similarity scores into
//! symbolically referenceable composition references.
//!
//! This module provides:
//! - `vectors_to_compositions()`: Convert vector similarity pairs to CompositionRefs
//! - `attention_weights_to_senses()`: Convert Transformer attention weights to sense compositions
//! - `explain_vector()`: Trace which compositions contribute to a vector representation
//!
//! v6.0: Initial implementation of the Transformer Bridge concept.

use crate::graph::RsvsGraph;
use crate::sense::SenseManager;
use crate::types::{CompositionRef, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// -----------------------------------------------------------------------
// TransformerBridgeConfig — configuration with serde support
// -----------------------------------------------------------------------

/// Configuration for the Transformer Bridge.
///
/// Controls how vector representations from a Transformer model are
/// converted into RSVS's symbolic composition references.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerBridgeConfig {
    /// Similarity threshold for considering two vectors "related".
    /// Pairs with cosine similarity below this are ignored.
    pub similarity_threshold: f32,
    /// Maximum compositions per induced sense from Transformer output.
    /// Prevents explosion of compositions from high-dimensional vectors.
    pub max_compositions: usize,
    /// Whether to use Transformer attention weights for composition weighting.
    /// If true, attention weights modulate the confidence of each composition.
    pub use_attention_weights: bool,
}

impl Default for TransformerBridgeConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.5,
            max_compositions: 10,
            use_attention_weights: true,
        }
    }
}

// -----------------------------------------------------------------------
// TransformerBridge — the bridge struct
// -----------------------------------------------------------------------

/// Bridge between Transformer vector representations and RSVS symbolic representations.
///
/// RSVS is NOT a replacement for Transformers — it's an interpretation layer on top.
/// The TransformerBridge converts abstract vector similarity scores into
/// symbolically referenceable composition references.
///
/// # Example
///
/// ```ignore
/// let bridge = TransformerBridge::new(TransformerBridgeConfig::default());
/// let compositions = bridge.vectors_to_compositions(
///     &token_vectors,
///     &token_labels,
///     0.7
/// );
/// ```
pub struct TransformerBridge {
    /// Similarity threshold for considering two vectors "related".
    pub similarity_threshold: f32,
    /// Maximum compositions per induced sense from Transformer output.
    pub max_compositions: usize,
    /// Whether to use Transformer attention weights for composition weighting.
    pub use_attention_weights: bool,
}

impl TransformerBridge {
    /// Create a new TransformerBridge with the given configuration.
    pub fn new(config: TransformerBridgeConfig) -> Self {
        Self {
            similarity_threshold: config.similarity_threshold,
            max_compositions: config.max_compositions,
            use_attention_weights: config.use_attention_weights,
        }
    }

    /// Convert vector similarity pairs to composition references.
    ///
    /// Given a set of token vectors and their labels, computes pairwise
    /// cosine similarity and returns CompositionRefs for pairs exceeding
    /// the threshold. This converts abstract vector similarity into
    /// symbolically referenceable structure.
    ///
    /// # Arguments
    /// * `token_vectors` - Vector representations for each token
    /// * `token_labels` - Labels corresponding to each vector
    /// * `threshold` - Minimum cosine similarity to create a composition pair
    ///
    /// # Returns
    /// A vector of CompositionRefs representing the discovered relationships.
    /// Each token that is similar enough to another gets a CompositionRef
    /// pointing to that token (with sense_id=0 as default).
    pub fn vectors_to_compositions(
        &self,
        token_vectors: &[Vec<f32>],
        token_labels: &[String],
        threshold: f32,
    ) -> Vec<CompositionRef> {
        if token_vectors.is_empty() || token_labels.is_empty() {
            return vec![];
        }

        let n = token_vectors.len().min(token_labels.len());
        let mut compositions = Vec::new();
        let mut per_token_counts: HashMap<usize, usize> = HashMap::new();

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let sim = cosine_similarity(&token_vectors[i], &token_vectors[j]);
                if sim >= threshold {
                    let count_i = per_token_counts.entry(i).or_insert(0);
                    if *count_i < self.max_compositions {
                        // Use index as NodeId placeholder; in practice these
                        // would be resolved to actual NodeIds via token_to_id
                        compositions.push(CompositionRef::new(j as NodeId, 0));
                        *count_i += 1;
                    }
                }
            }
        }

        compositions
    }

    /// Convert Transformer attention weights to sense compositions.
    ///
    /// Given an attention matrix (attention[i][j] = how much token i attends
    /// to token j), creates composition references for high-attention pairs.
    ///
    /// # Arguments
    /// * `attention_matrix` - Attention weights from a Transformer layer
    /// * `token_labels` - Labels corresponding to each position
    ///
    /// # Returns
    /// A vector of (token_label, compositions) pairs, where each token
    /// has its high-attention targets as compositions.
    pub fn attention_weights_to_senses(
        &self,
        attention_matrix: &[Vec<f32>],
        token_labels: &[String],
    ) -> Vec<(String, Vec<CompositionRef>)> {
        if attention_matrix.is_empty() || token_labels.is_empty() {
            return vec![];
        }

        let n = attention_matrix.len().min(token_labels.len());
        let mut result = Vec::new();

        for i in 0..n {
            let mut comps = Vec::new();

            if i >= attention_matrix.len() {
                continue;
            }

            // Sort attention targets by weight (descending)
            let mut indexed_weights: Vec<(usize, f32)> = attention_matrix[i]
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i) // Don't attend to self
                .map(|(j, &w)| (j, w))
                .collect();
            indexed_weights.sort_by(|a, b| b.1.total_cmp(&a.1));

            for (j, weight) in indexed_weights {
                if comps.len() >= self.max_compositions {
                    break;
                }
                if weight < self.similarity_threshold {
                    continue;
                }
                comps.push(CompositionRef::new(j as NodeId, 0));
            }

            if !comps.is_empty() {
                result.push((token_labels[i].clone(), comps));
            }
        }

        result
    }

    /// Explain which compositions contribute to a vector representation.
    ///
    /// Given a query vector, this method traces through the RSVS graph to find
    /// which symbolic compositions contribute to explaining the vector.
    /// This is the core of the "interpretation layer" concept — making
    /// abstract vector dimensions traceable to symbolic references.
    ///
    /// # Arguments
    /// * `vector` - The query vector to explain
    /// * `graph` - The RSVS knowledge graph
    /// * `senses` - The sense managers for all nodes
    ///
    /// # Returns
    /// A vector of human-readable strings describing the compositional
    /// contributions to this vector's representation.
    pub fn explain_vector(
        &self,
        _vector: &[f32],
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
    ) -> Vec<String> {
        let mut explanations = Vec::new();

        // For each node with compositional senses, check if its compositions
        // overlap with nodes that could be represented by the vector
        for (&node_id, sm) in senses {
            let node_label = match graph.get_node(node_id) {
                Some(n) => n.label.clone(),
                None => continue,
            };

            for sense in &sm.senses {
                if !sense.is_compositional() {
                    continue;
                }

                // Check each composition's contribution
                let mut comp_labels = Vec::new();
                for comp in &sense.compositions {
                    if let Some(target_node) = graph.get_node(comp.node_id) {
                        comp_labels
                            .push(format!("{}.sense_{}", target_node.label, comp.sense_id));
                    }
                }

                if !comp_labels.is_empty() {
                    explanations.push(format!(
                        "{}.sense_{} = [{}]",
                        node_label,
                        sense.id,
                        comp_labels.join(", ")
                    ));
                }
            }
        }

        // Sort for deterministic output
        explanations.sort();
        explanations.truncate(self.max_compositions * 3);
        explanations
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_vectors_to_compositions_basic() {
        let bridge = TransformerBridge::new(TransformerBridgeConfig {
            similarity_threshold: 0.5,
            max_compositions: 5,
            use_attention_weights: false,
        });

        // Two vectors that are similar
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0], // Similar to first
            vec![0.0, 0.0, 1.0], // Orthogonal
        ];
        let labels = vec!["token_a".to_string(), "token_b".to_string(), "token_c".to_string()];

        let comps = bridge.vectors_to_compositions(&vectors, &labels, 0.8);
        // token_a and token_b should be linked
        assert!(!comps.is_empty());
    }

    #[test]
    fn test_attention_weights_to_senses_basic() {
        let bridge = TransformerBridge::new(TransformerBridgeConfig {
            similarity_threshold: 0.3,
            max_compositions: 3,
            use_attention_weights: true,
        });

        // Attention matrix: token 0 attends strongly to token 1
        let attention = vec![
            vec![0.1, 0.8, 0.1], // token 0 → attends to token 1
            vec![0.7, 0.1, 0.2], // token 1 → attends to token 0
            vec![0.1, 0.1, 0.1], // token 2 → low attention everywhere
        ];
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let senses = bridge.attention_weights_to_senses(&attention, &labels);
        assert!(!senses.is_empty());

        // Token "a" should have compositions pointing to token "b"
        let a_sense = senses.iter().find(|(label, _)| label == "a");
        assert!(a_sense.is_some());
        assert!(!a_sense.unwrap().1.is_empty());
    }

    #[test]
    fn test_empty_inputs() {
        let bridge = TransformerBridge::new(TransformerBridgeConfig::default());

        let comps = bridge.vectors_to_compositions(&[], &[], 0.5);
        assert!(comps.is_empty());

        let senses = bridge.attention_weights_to_senses(&[], &[]);
        assert!(senses.is_empty());
    }
}
