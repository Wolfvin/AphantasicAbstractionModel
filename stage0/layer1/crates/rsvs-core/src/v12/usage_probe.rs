//! # RAB Phase S: Usage Discovery
//!
//! AAM doesn't just store definitions — it attempts to USE concepts in new
//! contexts and verifies whether the usage is consistent with its knowledge
//! graph. This is trial-and-error learning without retraining.
//!
//! ## Key Insight
//!
//! No LLM can "try" to use a concept and verify whether it succeeded.
//! AAM can: generate → validate → question → correct → learn.
//!
//! ## Architecture
//!
//! ```text
//! generate_usage_probe("raja") → GenerativeProbe { "raja memerintah kerajaan" }
//!       ↓
//! validate_usage_probe("raja menari balet") → ValidityProbe { valid: false, score: 0.2 }
//!       ↓
//! If invalid → generate question about inconsistency
//! ```

use serde::{Deserialize, Serialize};
use super::pipeline::Graph;
use super::spreading::{ActivationMap, SpreadingActivation};
use super::types::*;
use crate::types::NodeId;

// ========================================================================
// UsageProbe — AAM's Trial Usage of Knowledge
// ========================================================================

/// A probe that tests whether AAM can use its knowledge correctly.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageProbe {
    /// Unique probe identifier.
    #[serde(default)]
    pub probe_id: String,
    /// The node being probed.
    #[serde(default)]
    pub target_node_label: String,
    /// What type of probe this is.
    #[serde(default)]
    pub probe_type: ProbeType,
    /// When this probe was generated (epoch seconds).
    #[serde(default)]
    pub generated_at: u64,
}

// ========================================================================
// ProbeType — What kind of probe
// ========================================================================

/// What kind of usage probe this is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProbeType {
    /// AAM generates a template sentence from known compositions.
    GenerativeProbe {
        /// The template sentence AAM generated.
        #[serde(default)]
        generated_usage: String,
        /// Confidence in this usage (average of source compositions).
        #[serde(default)]
        confidence: f32,
        /// Which compositions were used to generate this.
        #[serde(default)]
        based_on_compositions: Vec<CompositionId>,
    },
    /// AAM validates a candidate usage against the graph.
    ValidityProbe {
        /// The candidate usage sentence to validate.
        #[serde(default)]
        candidate_usage: String,
        /// Whether AAM predicts the usage is valid.
        #[serde(default)]
        predicted_valid: bool,
        /// Evidence from spreading activation.
        #[serde(default)]
        spreading_evidence: Vec<(String, f32)>,
    },
    /// AAM checks if two compositions are consistent.
    ConsistencyProbe {
        /// First composition.
        #[serde(default)]
        composition_a: CompositionId,
        /// Second composition.
        #[serde(default)]
        composition_b: CompositionId,
        /// Whether they're predicted consistent.
        #[serde(default)]
        predicted_consistent: bool,
        /// Jaccard similarity score.
        #[serde(default)]
        jaccard_score: f32,
    },
}

impl Default for ProbeType {
    fn default() -> Self {
        ProbeType::GenerativeProbe {
            generated_usage: String::new(),
            confidence: 0.0,
            based_on_compositions: Vec::new(),
        }
    }
}

// ========================================================================
// ProbeResult — Output of probe validation
// ========================================================================

/// Result of validating a usage probe.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProbeResult {
    /// Whether the usage is valid/consistent.
    #[serde(default)]
    pub valid: bool,
    /// Validity score (0.0–1.0).
    #[serde(default)]
    pub score: f32,
    /// Evidence trail (top activated nodes).
    #[serde(default)]
    pub evidence: Vec<(String, f32)>,
    /// Why this result was produced.
    #[serde(default)]
    pub reason: String,
}

// ========================================================================
// UsageDiscoveryEngine
// ========================================================================

/// Engine for generating and validating usage probes.
#[derive(Debug, Clone)]
pub struct UsageDiscoveryEngine {
    activation: SpreadingActivation,
    /// Threshold for Jaccard similarity — below this = invalid usage.
    validity_threshold: f32,
}

impl Default for UsageDiscoveryEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl UsageDiscoveryEngine {
    /// Create a new usage discovery engine.
    pub fn new() -> Self {
        Self {
            activation: SpreadingActivation::new(),
            validity_threshold: 0.3,
        }
    }

    /// Create with custom validity threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            activation: SpreadingActivation::new(),
            validity_threshold: threshold,
        }
    }

    /// Generate a usage probe for a node label.
    ///
    /// Takes a node label, spreads activation from it, collects top-activated
    /// compositions, and generates a template sentence from their roles.
    ///
    /// This is NOT free-text generation — it reconstructs from templates
    /// derived from compositions AAM already knows.
    pub fn generate_usage_probe(
        &self,
        target_label: &str,
        graph: &Graph,
    ) -> UsageProbe {
        // Find the target node.
        let target_node_id = match graph.label_to_id.get(&target_label.to_lowercase()) {
            Some(&id) => id,
            None => {
                return UsageProbe {
                    probe_id: format!("probe_gen_{}", target_label),
                    target_node_label: target_label.to_string(),
                    probe_type: ProbeType::GenerativeProbe {
                        generated_usage: String::new(),
                        confidence: 0.0,
                        based_on_compositions: Vec::new(),
                    },
                    generated_at: now_epoch_secs(),
                };
            }
        };

        // Spread activation from the target node.
        let seeds = vec![(target_node_id, 1.0)];
        let activation_map = self.activation.spread(&seeds, graph);

        // Collect top activated compositions (N=3-5).
        let top_compositions = self.find_top_compositions(target_node_id, &activation_map, graph, 5);

        if top_compositions.is_empty() {
            return UsageProbe {
                probe_id: format!("probe_gen_{}", target_label),
                target_node_label: target_label.to_string(),
                probe_type: ProbeType::GenerativeProbe {
                    generated_usage: format!("[no compositions found for '{}']", target_label),
                    confidence: 0.0,
                    based_on_compositions: Vec::new(),
                },
                generated_at: now_epoch_secs(),
            };
        }

        // Generate template sentence from the best composition.
        let (best_comp_id, best_comp, best_score) = &top_compositions[0];
        let template = generate_template_from_composition(best_comp, target_label);
        let confidence = *best_score;

        let based_on: Vec<CompositionId> = top_compositions
            .iter()
            .map(|(id, _, _)| id.clone())
            .collect();

        UsageProbe {
            probe_id: format!("probe_gen_{}_{}", target_label, best_comp_id),
            target_node_label: target_label.to_string(),
            probe_type: ProbeType::GenerativeProbe {
                generated_usage: template,
                confidence,
                based_on_compositions: based_on,
            },
            generated_at: now_epoch_secs(),
        }
    }

    /// Validate a candidate usage against the graph.
    ///
    /// Spreads activation from nodes in the candidate usage and compares
    /// with baseline activation from known good compositions.
    /// Uses Jaccard similarity as the validity metric.
    pub fn validate_usage_probe(
        &self,
        candidate_usage: &str,
        graph: &Graph,
    ) -> (UsageProbe, ProbeResult) {
        // Find nodes mentioned in the candidate usage.
        let tokens: Vec<&str> = candidate_usage.split_whitespace().collect();
        let mut seed_node_ids: Vec<(NodeId, f32)> = Vec::new();

        for token in &tokens {
            let lower = token.to_lowercase();
            if let Some(&node_id) = graph.label_to_id.get(&lower) {
                seed_node_ids.push((node_id, 1.0));
            }
        }

        if seed_node_ids.is_empty() {
            return (
                UsageProbe {
                    probe_id: format!("probe_val_{}", now_epoch_secs()),
                    target_node_label: candidate_usage.to_string(),
                    probe_type: ProbeType::ValidityProbe {
                        candidate_usage: candidate_usage.to_string(),
                        predicted_valid: false,
                        spreading_evidence: Vec::new(),
                    },
                    generated_at: now_epoch_secs(),
                },
                ProbeResult {
                    valid: false,
                    score: 0.0,
                    evidence: Vec::new(),
                    reason: "No known nodes found in candidate usage".into(),
                },
            );
        }

        // Spread from candidate usage nodes.
        let candidate_map = self.activation.spread(&seed_node_ids, graph);

        // Get baseline from known compositions involving those nodes.
        let baseline_map = self.compute_baseline_activation(&seed_node_ids, graph);

        // Compute Jaccard similarity between candidate and baseline.
        let jaccard = candidate_map.jaccard_similarity(&baseline_map);

        // Get top evidence.
        let evidence: Vec<(String, f32)> = candidate_map
            .top_n(5)
            .into_iter()
            .filter_map(|(node_id, energy)| {
                graph.node_label(node_id).map(|label| (label.to_string(), energy))
            })
            .collect();

        let predicted_valid = jaccard >= self.validity_threshold;

        let reason = if predicted_valid {
            format!("Jaccard similarity {:.2} >= threshold {:.2} — usage consistent with graph", jaccard, self.validity_threshold)
        } else {
            format!("Jaccard similarity {:.2} < threshold {:.2} — usage inconsistent with graph", jaccard, self.validity_threshold)
        };

        (
            UsageProbe {
                probe_id: format!("probe_val_{}", now_epoch_secs()),
                target_node_label: candidate_usage.to_string(),
                probe_type: ProbeType::ValidityProbe {
                    candidate_usage: candidate_usage.to_string(),
                    predicted_valid,
                    spreading_evidence: evidence.clone(),
                },
                generated_at: now_epoch_secs(),
            },
            ProbeResult {
                valid: predicted_valid,
                score: jaccard,
                evidence,
                reason,
            },
        )
    }

    /// Check consistency between two compositions.
    pub fn check_consistency(
        &self,
        comp_a_id: &CompositionId,
        comp_b_id: &CompositionId,
        graph: &Graph,
    ) -> (UsageProbe, ProbeResult) {
        let comp_a = graph.compositions.get(comp_a_id);
        let comp_b = graph.compositions.get(comp_b_id);

        let (seeds_a, seeds_b) = match (comp_a, comp_b) {
            (Some(a), Some(b)) => {
                let seeds_a: Vec<(NodeId, f32)> = a.members.iter()
                    .map(|m| (m.node_id, 1.0))
                    .collect();
                let seeds_b: Vec<(NodeId, f32)> = b.members.iter()
                    .map(|m| (m.node_id, 1.0))
                    .collect();
                (seeds_a, seeds_b)
            }
            _ => {
                return (
                    UsageProbe {
                        probe_id: format!("probe_con_{}", now_epoch_secs()),
                        target_node_label: String::new(),
                        probe_type: ProbeType::ConsistencyProbe {
                            composition_a: comp_a_id.clone(),
                            composition_b: comp_b_id.clone(),
                            predicted_consistent: false,
                            jaccard_score: 0.0,
                        },
                        generated_at: now_epoch_secs(),
                    },
                    ProbeResult {
                        valid: false,
                        score: 0.0,
                        evidence: Vec::new(),
                        reason: "One or both compositions not found".into(),
                    },
                );
            }
        };

        let map_a = self.activation.spread(&seeds_a, graph);
        let map_b = self.activation.spread(&seeds_b, graph);
        let jaccard = map_a.jaccard_similarity(&map_b);

        let predicted_consistent = jaccard >= self.validity_threshold;

        (
            UsageProbe {
                probe_id: format!("probe_con_{}", now_epoch_secs()),
                target_node_label: String::new(),
                probe_type: ProbeType::ConsistencyProbe {
                    composition_a: comp_a_id.clone(),
                    composition_b: comp_b_id.clone(),
                    predicted_consistent,
                    jaccard_score: jaccard,
                },
                generated_at: now_epoch_secs(),
            },
            ProbeResult {
                valid: predicted_consistent,
                score: jaccard,
                evidence: Vec::new(),
                reason: format!("Jaccard similarity: {:.2}", jaccard),
            },
        )
    }

    // -------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------

    /// Find top N most-activated compositions involving the target node.
    fn find_top_compositions(
        &self,
        target_node_id: NodeId,
        activation_map: &ActivationMap,
        graph: &Graph,
        n: usize,
    ) -> Vec<(CompositionId, Composition, f32)> {
        let mut scored: Vec<(CompositionId, Composition, f32)> = Vec::new();

        for (comp_id, comp) in &graph.compositions {
            // Only consider compositions that include the target node.
            if !comp.members.iter().any(|m| m.node_id == target_node_id) {
                continue;
            }

            // Score: average activation of all member nodes.
            let total: f32 = comp.members.iter()
                .map(|m| activation_map.energy(m.node_id))
                .sum();
            let avg = if comp.members.is_empty() { 0.0 } else { total / comp.members.len() as f32 };

            scored.push((comp_id.clone(), comp.clone(), avg));
        }

        // Sort by score descending.
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    /// Compute baseline activation from known good compositions.
    fn compute_baseline_activation(
        &self,
        seeds: &[(NodeId, f32)],
        graph: &Graph,
    ) -> ActivationMap {
        // Collect member nodes from all compositions involving the seed nodes.
        let mut all_seeds: Vec<(NodeId, f32)> = seeds.to_vec();
        for seed_id in seeds.iter().map(|(id, _)| *id) {
            if let Some(comp_ids) = graph.node_to_compositions.get(&seed_id) {
                for comp_id in comp_ids {
                    if let Some(comp) = graph.compositions.get(comp_id) {
                        for member in &comp.members {
                            if !all_seeds.iter().any(|(id, _)| *id == member.node_id) {
                                all_seeds.push((member.node_id, 0.5)); // Lower energy for indirect
                            }
                        }
                    }
                }
            }
        }

        self.activation.spread(&all_seeds, graph)
    }
}

// ========================================================================
// Template Generation
// ========================================================================

/// Generate a template sentence from a composition.
///
/// This is NOT free-text generation — it reconstructs from templates
/// derived from compositions AAM already knows.
fn generate_template_from_composition(comp: &Composition, target_label: &str) -> String {
    match comp.composition_type {
        CompositionType::Event => {
            let agent = comp.member_with_role(&SemanticRole::Arg0Agent)
                .map(|m| m.label.as_str())
                .unwrap_or("?");
            let predicate = comp.member_with_role(&SemanticRole::Predicate)
                .map(|m| m.label.as_str())
                .unwrap_or(target_label);
            let patient = comp.member_with_role(&SemanticRole::Arg1Patient)
                .map(|m| m.label.as_str())
                .unwrap_or("?");

            format!("{} {} {}", agent, predicate, patient)
        }
        CompositionType::EquativeBinding => {
            let subject = comp.member_with_role(&SemanticRole::Subject)
                .map(|m| m.label.as_str())
                .unwrap_or("?");
            let complement = comp.member_with_role(&SemanticRole::Complement)
                .map(|m| m.label.as_str())
                .unwrap_or("?");

            format!("{} adalah {}", subject, complement)
        }
        CompositionType::PossessiveBinding => {
            let possessor = comp.member_with_role(&SemanticRole::Possessor)
                .map(|m| m.label.as_str())
                .unwrap_or("?");
            let possession = comp.member_with_role(&SemanticRole::Possession)
                .map(|m| m.label.as_str())
                .unwrap_or("?");

            format!("{} punya {}", possessor, possession)
        }
        _ => {
            // Generic: list members with roles.
            let parts: Vec<String> = comp.members.iter()
                .map(|m| format!("{:?}:{}", m.role, m.label))
                .collect();
            format!("[{}]", parts.join(", "))
        }
    }
}

/// Current epoch seconds.
fn now_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_usage_probe_empty_graph() {
        let engine = UsageDiscoveryEngine::new();
        let graph = Graph::new();
        let probe = engine.generate_usage_probe("raja", &graph);
        assert_eq!(probe.target_node_label, "raja");
    }

    #[test]
    fn test_validate_usage_probe_empty_graph() {
        let engine = UsageDiscoveryEngine::new();
        let graph = Graph::new();
        let (_probe, result) = engine.validate_usage_probe("raja memerintah kerajaan", &graph);
        assert!(!result.valid);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_usage_probe_default() {
        let probe = UsageProbe::default();
        assert!(probe.probe_id.is_empty());
    }

    #[test]
    fn test_probe_type_default() {
        let pt = ProbeType::default();
        assert!(matches!(pt, ProbeType::GenerativeProbe { .. }));
    }

    #[test]
    fn test_template_generation_event() {
        let comp = Composition {
            composition_type: CompositionType::Event,
            members: vec![
                CompositionMember { node_id: 0, role: SemanticRole::Arg0Agent, confidence: 0.8, label: "raja".into(), source: None },
                CompositionMember { node_id: 1, role: SemanticRole::Predicate, confidence: 0.8, label: "memerintah".into(), source: None },
                CompositionMember { node_id: 2, role: SemanticRole::Arg1Patient, confidence: 0.8, label: "kerajaan".into(), source: None },
            ],
            ..Composition::default()
        };

        let template = generate_template_from_composition(&comp, "memerintah");
        assert_eq!(template, "raja memerintah kerajaan");
    }

    #[test]
    fn test_template_generation_equative() {
        let comp = Composition {
            composition_type: CompositionType::EquativeBinding,
            members: vec![
                CompositionMember { node_id: 0, role: SemanticRole::Subject, confidence: 0.8, label: "ini".into(), source: None },
                CompositionMember { node_id: 1, role: SemanticRole::Complement, confidence: 0.8, label: "makanan".into(), source: None },
            ],
            ..Composition::default()
        };

        let template = generate_template_from_composition(&comp, "adalah");
        assert_eq!(template, "ini adalah makanan");
    }

    #[test]
    fn test_template_generation_possessive() {
        let comp = Composition {
            composition_type: CompositionType::PossessiveBinding,
            members: vec![
                CompositionMember { node_id: 0, role: SemanticRole::Possessor, confidence: 0.8, label: "raja".into(), source: None },
                CompositionMember { node_id: 1, role: SemanticRole::Possession, confidence: 0.8, label: "kerajaan".into(), source: None },
            ],
            ..Composition::default()
        };

        let template = generate_template_from_composition(&comp, "punya");
        assert_eq!(template, "raja punya kerajaan");
    }

    #[test]
    fn test_engine_with_custom_threshold() {
        let _engine = UsageDiscoveryEngine::with_threshold(0.5);
    }
}
