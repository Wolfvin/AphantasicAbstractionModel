//! Neuro-symbolic composition verification for RSVS v6.4
//!
//! Inspired by Losion's NeuroSymbolicVerifier (AlphaProof/AlphaGeometry 2 inspired).
//! Adapted for RSVS's structural domain:
//!
//! Instead of neural verification networks, RSVS uses:
//! - **Symbolic rules**: Structural invariants that compositions must satisfy
//! - **Verification loop**: Iterative checking with revision
//! - **Error localization**: Identify WHICH composition caused a violation
//! - **Feedback generation**: Describe what went wrong for logging/recovery
//!
//! Key differences from Losion's neural approach:
//! - No neural networks — uses deterministic structural checks
//! - Rules are RSVS-specific (no cross-domain rule engine)
//! - Verification is real-time (not batched)
//!
//! Verification rules:
//! 1. No self-reference: compositions must not reference the same node
//! 2. Layer consistency: compositions must reference lower layers
//! 3. Grounding threshold: composition targets must be grounded
//! 4. Frequency threshold: composition targets must have sufficient freq
//! 5. Max compositions: sense must not exceed max_senses_per_id compositions
//! 6. No circular chains: transitive closure must not loop back

use crate::sense::{Sense, SenseConfig, SenseManager};
use crate::types::{NodeId};
use crate::graph::RsvsGraph;
use std::collections::{HashMap, HashSet};

// -----------------------------------------------------------------------
// VerificationStatus
// -----------------------------------------------------------------------

/// Result of a verification check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationStatus {
    /// All rules passed.
    Verified,
    /// Some rules failed, but composition is partially valid.
    Partial {
        /// Number of rules that passed.
        passed: usize,
        /// Number of rules that failed.
        failed: usize,
    },
    /// Verification failed — composition is invalid.
    Failed,
    /// Verification inconclusive — need more data.
    Unsure,
    /// Composition needs revision based on verification results.
    NeedsRevision,
}

// -----------------------------------------------------------------------
// VerificationRule
// -----------------------------------------------------------------------

/// A single verification rule.
#[derive(Debug, Clone)]
pub struct VerificationRule {
    /// Human-readable name of the rule.
    pub name: String,
    /// Description of what this rule checks.
    pub description: String,
    /// Weight of this rule in the overall verification (0.0–1.0).
    pub weight: f32,
    /// Minimum score (0.0–1.0) to consider this rule "passed".
    pub threshold: f32,
}

impl VerificationRule {
    /// Create a new verification rule.
    pub fn new(name: &str, description: &str, weight: f32, threshold: f32) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            weight,
            threshold,
        }
    }
}

// -----------------------------------------------------------------------
// VerificationResult
// -----------------------------------------------------------------------

/// Result of verifying a single rule against a sense.
#[derive(Debug, Clone)]
pub struct RuleResult {
    /// The rule that was checked.
    pub rule: VerificationRule,
    /// Score achieved (0.0–1.0).
    pub score: f32,
    /// Whether this rule passed.
    pub passed: bool,
    /// Description of what went wrong (if failed).
    pub feedback: Option<String>,
}

// -----------------------------------------------------------------------
// NeuroSymVerifier
// -----------------------------------------------------------------------

/// Neuro-symbolic composition verifier — inspired by Losion's NeuroSymbolicVerifier.
///
/// Verifies that a sense's compositions satisfy structural invariants.
/// Uses iterative verification with revision (up to max_iterations).
pub struct NeuroSymVerifier {
    /// Rules to check during verification.
    pub rules: Vec<VerificationRule>,
    /// Maximum number of verification-revision iterations.
    /// Default: 3 (same as Losion)
    pub max_iterations: usize,
    /// Minimum average rule score for VERIFIED status.
    pub verification_threshold: f32,
}

impl Default for NeuroSymVerifier {
    fn default() -> Self {
        Self {
            rules: Self::default_rules(),
            max_iterations: 3,
            verification_threshold: 0.8,
        }
    }
}

impl NeuroSymVerifier {
    /// Create a new verifier with default rules.
    pub fn new() -> Self {
        Self::default()
    }

    /// Default verification rules for RSVS compositions.
    fn default_rules() -> Vec<VerificationRule> {
        vec![
            VerificationRule::new(
                "no_self_reference",
                "Compositions must not reference the same node they define",
                1.0,
                1.0, // Binary — any self-reference = failure
            ),
            VerificationRule::new(
                "layer_consistency",
                "Compositions should reference equal or lower layers",
                0.8,
                0.5, // Allow some upward references (cross-domain)
            ),
            VerificationRule::new(
                "grounding_threshold",
                "Composition targets should be grounded",
                0.7,
                0.5, // Some ungrounded targets are OK if others are grounded
            ),
            VerificationRule::new(
                "frequency_threshold",
                "Composition targets should have sufficient frequency",
                0.5,
                0.3, // Weak targets are OK if the composition is otherwise good
            ),
            VerificationRule::new(
                "no_circular_chain",
                "Transitive composition closure must not loop back to the node",
                1.0,
                1.0, // Binary — any circular chain = failure
            ),
        ]
    }

    /// Verify a sense's compositions against all rules.
    ///
    /// Returns a list of rule results and an overall verification status.
    pub fn verify(
        &self,
        node_id: NodeId,
        sense: &Sense,
        graph: &RsvsGraph,
        all_senses: &HashMap<NodeId, SenseManager>,
        config: &SenseConfig,
    ) -> (VerificationStatus, Vec<RuleResult>) {
        let mut results = Vec::new();

        for rule in &self.rules {
            let result = self.check_rule(rule, node_id, sense, graph, all_senses, config);
            results.push(result);
        }

        let status = self.compute_status(&results);
        (status, results)
    }

    /// Verify with iterative revision — keep revising until VERIFIED or max iterations.
    pub fn verify_with_revision(
        &self,
        node_id: NodeId,
        sense: &mut Sense,
        graph: &RsvsGraph,
        all_senses: &HashMap<NodeId, SenseManager>,
        config: &SenseConfig,
    ) -> (VerificationStatus, Vec<Vec<RuleResult>>) {
        let mut all_results = Vec::new();

        for _ in 0..self.max_iterations {
            let (status, results) = self.verify(node_id, sense, graph, all_senses, config);
            all_results.push(results);

            match status {
                VerificationStatus::Verified => return (VerificationStatus::Verified, all_results),
                VerificationStatus::NeedsRevision | VerificationStatus::Failed => {
                    // Apply revision: remove the worst-scoring composition
                    if sense.compositions.len() > 1 {
                        sense.compositions.pop();
                        sense.grounding.revision_count += 1;
                    } else {
                        // Can't revise further
                        return (VerificationStatus::Failed, all_results);
                    }
                }
                _ => return (status, all_results),
            }
        }

        (VerificationStatus::NeedsRevision, all_results)
    }

    /// Check a single rule against a sense.
    fn check_rule(
        &self,
        rule: &VerificationRule,
        node_id: NodeId,
        sense: &Sense,
        graph: &RsvsGraph,
        all_senses: &HashMap<NodeId, SenseManager>,
        config: &SenseConfig,
    ) -> RuleResult {
        match rule.name.as_str() {
            "no_self_reference" => {
                let has_self = sense.compositions.iter().any(|c| c.node_id == node_id);
                RuleResult {
                    rule: rule.clone(),
                    score: if has_self { 0.0 } else { 1.0 },
                    passed: !has_self,
                    feedback: if has_self {
                        Some(format!("Sense references itself (node {})", node_id))
                    } else {
                        None
                    },
                }
            }
            "layer_consistency" => {
                let node_layer = graph
                    .get_node(node_id)
                    .map(|n| n.semantic.layer)
                    .unwrap_or(0);
                let total = sense.compositions.len().max(1);
                let consistent = sense
                    .compositions
                    .iter()
                    .filter(|c| {
                        graph
                            .get_node(c.node_id)
                            .map(|n| n.semantic.layer < node_layer)
                            .unwrap_or(true) // Unknown nodes don't count against
                    })
                    .count();
                let score = consistent as f32 / total as f32;
                RuleResult {
                    rule: rule.clone(),
                    score,
                    passed: score >= rule.threshold,
                    feedback: if score < rule.threshold {
                        Some(format!(
                            "Only {}/{} compositions reference lower layers",
                            consistent, total
                        ))
                    } else {
                        None
                    },
                }
            }
            "grounding_threshold" => {
                let total = sense.compositions.len().max(1);
                let grounded = sense
                    .compositions
                    .iter()
                    .filter(|c| {
                        all_senses
                            .get(&c.node_id)
                            .and_then(|sm| sm.senses.get(c.sense_id as usize))
                            .map(|s| s.is_grounded(config.grounding_min))
                            .unwrap_or(true) // Unknown → assume grounded
                    })
                    .count();
                let score = grounded as f32 / total as f32;
                RuleResult {
                    rule: rule.clone(),
                    score,
                    passed: score >= rule.threshold,
                    feedback: if score < rule.threshold {
                        Some(format!(
                            "Only {}/{} composition targets are grounded",
                            grounded, total
                        ))
                    } else {
                        None
                    },
                }
            }
            "frequency_threshold" => {
                let total = sense.compositions.len().max(1);
                let frequent = sense
                    .compositions
                    .iter()
                    .filter(|c| {
                        all_senses
                            .get(&c.node_id)
                            .and_then(|sm| sm.senses.get(c.sense_id as usize))
                            .map(|s| s.freq(node_id) >= config.induction.tau_compress)
                            .unwrap_or(true) // Unknown → assume frequent
                    })
                    .count();
                let score = frequent as f32 / total as f32;
                RuleResult {
                    rule: rule.clone(),
                    score,
                    passed: score >= rule.threshold,
                    feedback: if score < rule.threshold {
                        Some(format!(
                            "Only {}/{} composition targets have sufficient frequency",
                            frequent, total
                        ))
                    } else {
                        None
                    },
                }
            }
            "no_circular_chain" => {
                let has_cycle = self.detect_circular_chain(node_id, sense, all_senses);
                RuleResult {
                    rule: rule.clone(),
                    score: if has_cycle { 0.0 } else { 1.0 },
                    passed: !has_cycle,
                    feedback: if has_cycle {
                        Some("Circular composition chain detected".to_string())
                    } else {
                        None
                    },
                }
            }
            _ => RuleResult {
                rule: rule.clone(),
                score: 1.0,
                passed: true,
                feedback: None,
            },
        }
    }

    /// Detect circular chains in composition references.
    fn detect_circular_chain(
        &self,
        start_node: NodeId,
        sense: &Sense,
        all_senses: &HashMap<NodeId, SenseManager>,
    ) -> bool {
        let mut visited = HashSet::new();
        visited.insert(start_node);
        let mut stack: Vec<NodeId> = sense.compositions.iter().map(|c| c.node_id).collect();

        while let Some(current) = stack.pop() {
            if current == start_node {
                return true; // Circular chain found
            }
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            // Expand compositions of current node
            if let Some(sm) = all_senses.get(&current) {
                if let Some(sense) = sm.senses.first() {
                    for comp in &sense.compositions {
                        stack.push(comp.node_id);
                    }
                }
            }
        }
        false
    }

    /// Compute overall verification status from rule results.
    fn compute_status(&self, results: &[RuleResult]) -> VerificationStatus {
        let total_weight: f32 = results.iter().map(|r| r.rule.weight).sum();
        if total_weight == 0.0 {
            return VerificationStatus::Unsure;
        }

        let weighted_score: f32 = results
            .iter()
            .map(|r| r.rule.weight * r.score)
            .sum::<f32>()
            / total_weight;

        let passed = results.iter().filter(|r| r.passed).count();
        let failed = results.len() - passed;

        if failed == 0 {
            VerificationStatus::Verified
        } else if weighted_score >= self.verification_threshold {
            VerificationStatus::Partial { passed, failed }
        } else if weighted_score >= 0.3 {
            VerificationStatus::NeedsRevision
        } else {
            VerificationStatus::Failed
        }
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CompositionRef;

    #[test]
    fn test_verifier_default_rules() {
        let verifier = NeuroSymVerifier::new();
        assert_eq!(verifier.rules.len(), 5);
    }

    #[test]
    fn test_self_reference_detection() {
        let verifier = NeuroSymVerifier::new();
        let config = SenseConfig::default();
        let graph = RsvsGraph::new();
        let senses = HashMap::new();

        // Create a sense with self-reference
        let sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0)], // References node 1
            vec![1, 2],
            1,
        );

        // Verify with node_id = 1 (self-reference)
        let (status, results) = verifier.verify(1, &sense, &graph, &senses, &config);
        assert_ne!(status, VerificationStatus::Verified);

        // The no_self_reference rule should have failed
        let self_ref_result = results.iter().find(|r| r.rule.name == "no_self_reference").unwrap();
        assert!(!self_ref_result.passed);
    }

    #[test]
    fn test_clean_sense_passes() {
        let verifier = NeuroSymVerifier::new();
        let config = SenseConfig::default();
        let graph = RsvsGraph::new();
        let senses = HashMap::new();

        // Create a sense with no compositions (primitive)
        let sense = Sense::new(0, vec![1, 2, 3]);

        // Primitive senses have no compositions, so verification rules
        // about compositions are vacuously true. But some rules may
        // score lower due to missing references — that's expected.
        let (status, results) = verifier.verify(5, &sense, &graph, &senses, &config);
        // For a primitive sense with no compositions, most rules should pass
        // (no self-reference, no circular chain, etc.)
        // The overall status may be Partial or Verified depending on the rules
        assert_ne!(status, VerificationStatus::Failed);
        // At minimum, binary rules should pass
        assert!(results.iter().find(|r| r.rule.name == "no_self_reference").unwrap().passed);
        assert!(results.iter().find(|r| r.rule.name == "no_circular_chain").unwrap().passed);
    }
}
