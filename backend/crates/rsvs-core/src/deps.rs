//! DEPS Failure Recovery for RSVS v7.0 — Inspired by Losion's DEPS Planner
//!
//! Losion's DEPS (Describe-Explain-Plan-Select) planner provides structured
//! recovery from failed agent actions. Instead of simple retry, it:
//!   1. DESCRIBE: What happened? Classify failure type
//!   2. EXPLAIN: Why did it fail? Root cause analysis
//!   3. PLAN: Generate multiple alternative approaches
//!   4. SELECT: Choose best plan based on success rate + simplicity
//!
//! Adapted for RSVS's structural domain, DEPS handles failed operations:
//! - Composition verification failures (circular chains, self-reference)
//! - Traversal failures (leaf reached, confidence too low)
//! - Sense induction failures (too many senses, insufficient divergence)
//! - Grounding failures (too many contradictions)
//!
//! Key insight: Instead of just returning an error, DEPS generates
//! RECOVERY PLANS with estimated success rates. The caller can then
//! choose the best recovery strategy instead of blindly retrying.

use crate::error::RsvsError;
use crate::types::{NodeId, CompositionRef};

// -----------------------------------------------------------------------
// FailureType
// -----------------------------------------------------------------------

/// Classification of operation failures in RSVS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureType {
    /// Composition references itself (self-reference rule violation).
    SelfReference,
    /// Circular chain detected in composition references.
    CircularChain,
    /// Composition target not found in the graph.
    TargetNotFound,
    /// Layer inconsistency (composition references higher layer).
    LayerInconsistency,
    /// Too many senses for this node.
    SenseLimitReached,
    /// Insufficient divergence for new sense induction.
    InsufficientDivergence,
    /// Traversal reached leaf with no useful results.
    TraversalLeafReached,
    /// Traversal confidence too low.
    LowConfidence,
    /// Grounding score too low — compositions need revision.
    GroundingFailure,
    /// General/unknown failure.
    General,
}

// -----------------------------------------------------------------------
// RecoveryPlan
// -----------------------------------------------------------------------

/// A recovery plan for a failed operation.
#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    /// Human-readable description of the plan.
    pub description: String,
    /// The type of action this plan suggests.
    pub action: RecoveryAction,
    /// Estimated success rate (0.0–1.0) based on heuristics.
    pub estimated_success_rate: f32,
    /// Simplicity score (0.0–1.0) — simpler plans are preferred.
    pub simplicity: f32,
    /// Whether this plan modifies the graph (vs. just retrying).
    pub is_destructive: bool,
}

impl RecoveryPlan {
    /// Compute a composite score for plan selection.
    /// Higher = better. Weighted: 60% success rate + 40% simplicity.
    pub fn score(&self) -> f32 {
        0.6 * self.estimated_success_rate + 0.4 * self.simplicity
    }
}

// -----------------------------------------------------------------------
// RecoveryAction
// -----------------------------------------------------------------------

/// Actions that can be taken to recover from a failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Remove the offending composition and retry.
    RemoveComposition { node_id: NodeId, comp_index: usize },
    /// Try a different sense for the composition target.
    TryAlternativeSense { node_id: NodeId, comp: CompositionRef, alt_sense: u32 },
    /// Reduce traversal depth and retry.
    ReduceDepth { new_depth: usize },
    /// Use a different traversal paradigm.
    UseDifferentParadigm,
    /// Revise compositions based on grounding evidence.
    ReviseCompositions { node_id: NodeId },
    /// Merge with an existing sense instead of creating new.
    MergeWithExisting { node_id: NodeId, target_sense_idx: usize },
    /// Skip this operation entirely.
    Skip,
    /// Retry the same operation (simple retry).
    Retry,
}

// -----------------------------------------------------------------------
// DEPSResult
// -----------------------------------------------------------------------

/// Result of a DEPS analysis.
#[derive(Debug, Clone)]
pub struct DEPSResult {
    /// What happened — the failure classification.
    pub failure_type: FailureType,
    /// Why it happened — root cause description.
    pub explanation: String,
    /// Available recovery plans, sorted by score (best first).
    pub plans: Vec<RecoveryPlan>,
    /// The recommended plan (highest score).
    pub recommended: Option<RecoveryPlan>,
}

// -----------------------------------------------------------------------
// DEPSPlanner
// -----------------------------------------------------------------------

/// DEPS (Describe-Explain-Plan-Select) failure recovery planner.
///
/// Inspired by Losion's DEPS planner which provides structured recovery
/// from failed agent actions. In RSVS, this handles failed operations
/// by generating recovery plans with estimated success rates.
///
/// # Example
///
/// ```ignore
/// let planner = DEPSPlanner::new();
/// let result = planner.analyze(&RsvsError::CircularRef { from: 5, to: 5 }, 5);
/// if let Some(plan) = result.recommended {
///     println!("Recovery: {} (success rate: {:.0}%)",
///              plan.description, plan.estimated_success_rate * 100.0);
/// }
/// ```
pub struct DEPSPlanner {
    /// Per-failure-type recovery strategies, ordered by priority.
    strategies: HashMap<FailureType, Vec<RecoveryStrategy>>,
}

use std::collections::HashMap;

impl Default for DEPSPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl DEPSPlanner {
    /// Create a new DEPS planner with default strategies.
    pub fn new() -> Self {
        let mut strategies = HashMap::new();

        // Self-reference recovery
        strategies.insert(FailureType::SelfReference, vec![
            RecoveryStrategy {
                description: "Remove self-referencing composition".into(),
                success_rate: 0.95,
                simplicity: 0.9,
                is_destructive: true,
            },
            RecoveryStrategy {
                description: "Replace self-reference with nearest neighbor".into(),
                success_rate: 0.7,
                simplicity: 0.5,
                is_destructive: true,
            },
        ]);

        // Circular chain recovery
        strategies.insert(FailureType::CircularChain, vec![
            RecoveryStrategy {
                description: "Break cycle by removing the weakest composition".into(),
                success_rate: 0.85,
                simplicity: 0.8,
                is_destructive: true,
            },
            RecoveryStrategy {
                description: "Replace circular reference with intermediate node".into(),
                success_rate: 0.6,
                simplicity: 0.3,
                is_destructive: true,
            },
            RecoveryStrategy {
                description: "Skip this composition operation".into(),
                success_rate: 1.0,
                simplicity: 1.0,
                is_destructive: false,
            },
        ]);

        // Target not found recovery
        strategies.insert(FailureType::TargetNotFound, vec![
            RecoveryStrategy {
                description: "Create the missing target node first".into(),
                success_rate: 0.8,
                simplicity: 0.6,
                is_destructive: false,
            },
            RecoveryStrategy {
                description: "Remove the invalid composition reference".into(),
                success_rate: 0.9,
                simplicity: 0.9,
                is_destructive: true,
            },
        ]);

        // Sense limit recovery
        strategies.insert(FailureType::SenseLimitReached, vec![
            RecoveryStrategy {
                description: "Merge with most similar existing sense".into(),
                success_rate: 0.75,
                simplicity: 0.7,
                is_destructive: true,
            },
            RecoveryStrategy {
                description: "Force-assign to best matching sense".into(),
                success_rate: 0.85,
                simplicity: 0.9,
                is_destructive: false,
            },
        ]);

        // Insufficient divergence recovery
        strategies.insert(FailureType::InsufficientDivergence, vec![
            RecoveryStrategy {
                description: "Assign to existing sense (no new sense needed)".into(),
                success_rate: 0.9,
                simplicity: 0.95,
                is_destructive: false,
            },
            RecoveryStrategy {
                description: "Relax divergence threshold for this context".into(),
                success_rate: 0.5,
                simplicity: 0.4,
                is_destructive: false,
            },
        ]);

        // Traversal failures
        strategies.insert(FailureType::TraversalLeafReached, vec![
            RecoveryStrategy {
                description: "Reduce depth and try again".into(),
                success_rate: 0.6,
                simplicity: 0.8,
                is_destructive: false,
            },
            RecoveryStrategy {
                description: "Use MCTS for deeper exploration".into(),
                success_rate: 0.7,
                simplicity: 0.4,
                is_destructive: false,
            },
        ]);

        strategies.insert(FailureType::LowConfidence, vec![
            RecoveryStrategy {
                description: "Use different traversal paradigm".into(),
                success_rate: 0.65,
                simplicity: 0.7,
                is_destructive: false,
            },
            RecoveryStrategy {
                description: "Ingest more context to build confidence".into(),
                success_rate: 0.8,
                simplicity: 0.5,
                is_destructive: false,
            },
        ]);

        // Grounding failure recovery
        strategies.insert(FailureType::GroundingFailure, vec![
            RecoveryStrategy {
                description: "Revise compositions (remove least grounded)".into(),
                success_rate: 0.7,
                simplicity: 0.8,
                is_destructive: true,
            },
            RecoveryStrategy {
                description: "Add confirming context to improve grounding".into(),
                success_rate: 0.6,
                simplicity: 0.5,
                is_destructive: false,
            },
        ]);

        // General fallback
        strategies.insert(FailureType::General, vec![
            RecoveryStrategy {
                description: "Retry the operation".into(),
                success_rate: 0.3,
                simplicity: 1.0,
                is_destructive: false,
            },
            RecoveryStrategy {
                description: "Skip this operation".into(),
                success_rate: 1.0,
                simplicity: 1.0,
                is_destructive: false,
            },
        ]);

        Self { strategies }
    }

    /// Analyze an error and produce a DEPS result.
    ///
    /// 1. DESCRIBE: Classify the failure type
    /// 2. EXPLAIN: Generate a human-readable explanation
    /// 3. PLAN: Generate recovery plans from strategies
    /// 4. SELECT: Choose the best plan
    pub fn analyze(&self, error: &RsvsError, node_id: NodeId) -> DEPSResult {
        let failure_type = Self::classify_error(error);
        let explanation = Self::explain_error(error, node_id);
        let plans = self.generate_plans(&failure_type, node_id);
        let recommended = plans.first().cloned();

        DEPSResult {
            failure_type,
            explanation,
            plans,
            recommended,
        }
    }

    /// Step 1: DESCRIBE — classify the error into a failure type.
    fn classify_error(error: &RsvsError) -> FailureType {
        match error {
            RsvsError::CircularRef { from, to } if from == to => FailureType::SelfReference,
            RsvsError::CircularRef { .. } => FailureType::CircularChain,
            RsvsError::NodeNotFound { .. } => FailureType::TargetNotFound,
            RsvsError::Pipeline(_) => FailureType::General,
        }
    }

    /// Step 2: EXPLAIN — generate a human-readable explanation.
    fn explain_error(error: &RsvsError, node_id: NodeId) -> String {
        match error {
            RsvsError::CircularRef { from, to } if from == to => {
                format!("Node {} references itself in its compositions, violating the no-self-reference rule", node_id)
            }
            RsvsError::CircularRef { from, to } => {
                format!("Circular composition chain detected: node {} → {} → ... → {}", from, to, from)
            }
            RsvsError::NodeNotFound { id } => {
                format!("Composition target node {} not found in the graph — it may have been removed or never created", id)
            }
            RsvsError::Pipeline(msg) => {
                format!("Pipeline error for node {}: {}", node_id, msg)
            }
        }
    }

    /// Step 3: PLAN — generate recovery plans from strategies.
    fn generate_plans(&self, failure_type: &FailureType, node_id: NodeId) -> Vec<RecoveryPlan> {
        let strategies = self.strategies.get(failure_type)
            .or_else(|| self.strategies.get(&FailureType::General))
            .cloned()
            .unwrap_or_default();

        let mut plans: Vec<RecoveryPlan> = strategies.into_iter().enumerate().map(|(i, s)| {
            let action = self.strategy_to_action(failure_type, node_id, i);
            RecoveryPlan {
                description: s.description,
                action,
                estimated_success_rate: s.success_rate,
                simplicity: s.simplicity,
                is_destructive: s.is_destructive,
            }
        }).collect();

        // Step 4: SELECT — sort by composite score (best first)
        plans.sort_by(|a, b| b.score().total_cmp(&a.score()));
        plans
    }

    /// Convert a strategy index to a concrete recovery action.
    fn strategy_to_action(&self, failure_type: &FailureType, node_id: NodeId, index: usize) -> RecoveryAction {
        match failure_type {
            FailureType::SelfReference => RecoveryAction::RemoveComposition {
                node_id,
                comp_index: 0, // Remove first (self-referencing) composition
            },
            FailureType::CircularChain => match index {
                0 => RecoveryAction::RemoveComposition { node_id, comp_index: 0 },
                1 => RecoveryAction::TryAlternativeSense {
                    node_id,
                    comp: CompositionRef::new(node_id, 0),
                    alt_sense: 1,
                },
                _ => RecoveryAction::Skip,
            },
            FailureType::TargetNotFound => match index {
                0 => RecoveryAction::Retry, // Create target first, then retry
                1 => RecoveryAction::RemoveComposition { node_id, comp_index: 0 },
                _ => RecoveryAction::Skip,
            },
            FailureType::SenseLimitReached => match index {
                0 => RecoveryAction::MergeWithExisting { node_id, target_sense_idx: 0 },
                _ => RecoveryAction::Retry,
            },
            FailureType::InsufficientDivergence => match index {
                0 => RecoveryAction::MergeWithExisting { node_id, target_sense_idx: 0 },
                _ => RecoveryAction::Retry,
            },
            FailureType::TraversalLeafReached => match index {
                0 => RecoveryAction::ReduceDepth { new_depth: 1 },
                _ => RecoveryAction::UseDifferentParadigm,
            },
            FailureType::LowConfidence => match index {
                0 => RecoveryAction::UseDifferentParadigm,
                _ => RecoveryAction::Retry,
            },
            FailureType::GroundingFailure => RecoveryAction::ReviseCompositions { node_id },
            FailureType::LayerInconsistency => RecoveryAction::RemoveComposition { node_id, comp_index: 0 },
            _ => RecoveryAction::Retry,
        }
    }
}

/// Internal strategy representation.
#[derive(Clone)]
struct RecoveryStrategy {
    description: String,
    success_rate: f32,
    simplicity: f32,
    is_destructive: bool,
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_self_reference() {
        let planner = DEPSPlanner::new();
        let error = RsvsError::CircularRef { from: 5, to: 5 };
        let result = planner.analyze(&error, 5);
        assert_eq!(result.failure_type, FailureType::SelfReference);
    }

    #[test]
    fn test_classify_circular_chain() {
        let planner = DEPSPlanner::new();
        let error = RsvsError::CircularRef { from: 5, to: 10 };
        let result = planner.analyze(&error, 5);
        assert_eq!(result.failure_type, FailureType::CircularChain);
    }

    #[test]
    fn test_classify_node_not_found() {
        let planner = DEPSPlanner::new();
        let error = RsvsError::NodeNotFound { id: 42 };
        let result = planner.analyze(&error, 5);
        assert_eq!(result.failure_type, FailureType::TargetNotFound);
    }

    #[test]
    fn test_deps_generates_plans() {
        let planner = DEPSPlanner::new();
        let error = RsvsError::CircularRef { from: 5, to: 5 };
        let result = planner.analyze(&error, 5);
        assert!(!result.plans.is_empty());
        assert!(result.recommended.is_some());
    }

    #[test]
    fn test_plans_sorted_by_score() {
        let planner = DEPSPlanner::new();
        let error = RsvsError::CircularRef { from: 5, to: 5 };
        let result = planner.analyze(&error, 5);

        for i in 1..result.plans.len() {
            assert!(result.plans[i - 1].score() >= result.plans[i].score());
        }
    }

    #[test]
    fn test_recovery_plan_score() {
        let plan = RecoveryPlan {
            description: "Test".into(),
            action: RecoveryAction::Skip,
            estimated_success_rate: 0.8,
            simplicity: 0.9,
            is_destructive: false,
        };
        let expected = 0.6 * 0.8 + 0.4 * 0.9;
        assert!((plan.score() - expected).abs() < 0.01);
    }

    #[test]
    fn test_explanation_is_human_readable() {
        let planner = DEPSPlanner::new();
        let error = RsvsError::NodeNotFound { id: 42 };
        let result = planner.analyze(&error, 5);
        assert!(result.explanation.contains("42"));
        assert!(result.explanation.len() > 20);
    }
}
