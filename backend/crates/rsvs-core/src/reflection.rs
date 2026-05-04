//! SenseReflection — self-evaluation loop for RSVS v6.4
//!
//! Inspired by Losion's ReflectionEngine which uses Reflexion + Self-Refine
//! patterns. Adapted for RSVS's structural (not verbal) domain:
//!
//! Instead of verbal feedback, RSVS's reflection operates on grounding evidence.
//! After each ingest batch, SenseReflection evaluates each sense and produces
//! an action: CONFIRM, REVIEW, REVISE, or RETIRE.
//!
//! Key differences from Losion:
//! - Losion uses natural language feedback; RSVS uses grounding scores
//! - Losion tracks tool trust; RSVS tracks composition trust (grounding_verdict)
//! - Losion's reflection is LLM-driven; RSVS's is deterministic + structural
//!
//! The reflection loop runs at safe checkpoints (periodic, not per-ingest)
//! and produces a list of actions to apply:
//! - CONFIRM: grounding ≥ 0.6, no action needed
//! - REVIEW: grounding 0.3–0.6, needs observation
//! - REVISE: grounding < 0.3, compositions should be pruned
//! - RETIRE: fragile + ungrounded for too long → safe to delete

use crate::sense::{GroundingVerdict, Sense, SenseConfig, SenseManager, SenseStatus};
use crate::types::NodeId;
use std::collections::HashMap;

// -----------------------------------------------------------------------
// ReflectionAction
// -----------------------------------------------------------------------

/// Action produced by reflection for a specific sense.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectionAction {
    /// Sense is well-grounded — no action needed.
    Confirm {
        /// Node ID.
        node_id: NodeId,
        /// Sense index.
        sense_idx: usize,
    },
    /// Sense has some contradictions — monitor closely.
    Review {
        node_id: NodeId,
        sense_idx: usize,
        /// Current grounding score.
        grounding_score: u32, // ×100 to avoid f32 in enum
    },
    /// Sense has many contradictions — revise compositions.
    Revise {
        node_id: NodeId,
        sense_idx: usize,
    },
    /// Sense is fragile + ungrounded for too long — safe to delete.
    Retire {
        node_id: NodeId,
        sense_idx: usize,
    },
}

// -----------------------------------------------------------------------
// SenseReflection
// -----------------------------------------------------------------------

/// Configuration for the sense reflection engine.
#[derive(Debug, Clone)]
pub struct ReflectionConfig {
    /// Maximum number of revise actions per reflection cycle.
    /// Prevents catastrophic pruning from a single bad batch.
    /// Default: 3
    pub max_revise_per_cycle: usize,
    /// How many consecutive REVIEW verdicts before escalating to REVISE.
    /// Default: 3
    pub review_escalation_threshold: usize,
    /// Maximum inactivity (contexts) for a sense before it's eligible for RETIRE.
    /// Default: 100
    pub retire_inactivity_threshold: usize,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            max_revise_per_cycle: 3,
            review_escalation_threshold: 3,
            retire_inactivity_threshold: 100,
        }
    }
}

/// Self-evaluation engine for senses — inspired by Losion's ReflectionEngine.
///
/// Runs at safe checkpoints and evaluates each sense based on:
/// 1. Grounding verdict (WellGrounded / NeedsReview / NeedsRevision)
/// 2. Inactivity duration
/// 3. Sense status (Fragile / Mature)
/// 4. Consecutive review count (escalation tracking)
pub struct SenseReflection {
    pub config: ReflectionConfig,
    /// Tracks how many consecutive REVIEW verdicts each sense has received.
    /// Maps (NodeId, sense_idx) → consecutive review count.
    review_counts: HashMap<(NodeId, usize), usize>,
}

impl SenseReflection {
    /// Create a new reflection engine with the given configuration.
    pub fn new(config: ReflectionConfig) -> Self {
        Self {
            config,
            review_counts: HashMap::new(),
        }
    }

    /// Run a reflection cycle over all senses, producing a list of actions.
    ///
    /// This is the main entry point — call it periodically (e.g., every 50 contexts).
    /// The actions should be applied after the cycle completes.
    pub fn reflect(
        &mut self,
        all_senses: &HashMap<NodeId, SenseManager>,
        sense_config: &SenseConfig,
    ) -> Vec<ReflectionAction> {
        let mut actions = Vec::new();
        let mut revise_count = 0;

        for (&node_id, sm) in all_senses {
            for (sense_idx, sense) in sm.senses.iter().enumerate() {
                let action = self.evaluate_sense(node_id, sense_idx, sense, sense_config);

                // Rate-limit REVISE actions
                if matches!(action, ReflectionAction::Revise { .. }) {
                    if revise_count >= self.config.max_revise_per_cycle {
                        // Downgrade to REVIEW if we've hit the limit
                        actions.push(ReflectionAction::Review {
                            node_id,
                            sense_idx,
                            grounding_score: (sense.grounding.score() * 100.0) as u32,
                        });
                        continue;
                    }
                    revise_count += 1;
                }

                actions.push(action);
            }
        }

        actions
    }

    /// Evaluate a single sense and produce a reflection action.
    fn evaluate_sense(
        &mut self,
        node_id: NodeId,
        sense_idx: usize,
        sense: &Sense,
        config: &SenseConfig,
    ) -> ReflectionAction {
        // Check for retirement: fragile + very inactive + ungrounded
        if sense.status == SenseStatus::Fragile
            && sense.inactivity >= self.config.retire_inactivity_threshold
            && !sense.is_grounded(config.grounding_min)
        {
            self.review_counts.remove(&(node_id, sense_idx));
            return ReflectionAction::Retire { node_id, sense_idx };
        }

        let verdict = sense.grounding_verdict();

        match verdict {
            GroundingVerdict::WellGrounded => {
                // Reset review counter on improvement
                self.review_counts.remove(&(node_id, sense_idx));
                ReflectionAction::Confirm { node_id, sense_idx }
            }
            GroundingVerdict::NeedsReview => {
                // Track consecutive reviews for escalation
                let count = self
                    .review_counts
                    .entry((node_id, sense_idx))
                    .or_insert(0);
                *count += 1;

                if *count >= self.config.review_escalation_threshold {
                    // Escalate to REVISE
                    self.review_counts.remove(&(node_id, sense_idx));
                    ReflectionAction::Revise { node_id, sense_idx }
                } else {
                    ReflectionAction::Review {
                        node_id,
                        sense_idx,
                        grounding_score: (sense.grounding.score() * 100.0) as u32,
                    }
                }
            }
            GroundingVerdict::NeedsRevision => {
                self.review_counts.remove(&(node_id, sense_idx));
                ReflectionAction::Revise { node_id, sense_idx }
            }
        }
    }

    /// Apply a list of reflection actions to the sense managers.
    /// Returns the number of actions actually applied.
    pub fn apply_actions(
        &self,
        all_senses: &mut HashMap<NodeId, SenseManager>,
        actions: &[ReflectionAction],
        sense_config: &SenseConfig,
    ) -> usize {
        let mut applied = 0;

        for action in actions {
            match action {
                ReflectionAction::Revise { node_id, sense_idx } => {
                    if let Some(sm) = all_senses.get_mut(node_id) {
                        if let Some(sense) = sm.senses.get_mut(*sense_idx) {
                            if sense.revise_compositions(sense_config.grounding_min) {
                                applied += 1;
                            }
                        }
                    }
                }
                ReflectionAction::Retire { node_id, sense_idx } => {
                    if let Some(sm) = all_senses.get_mut(node_id) {
                        if *sense_idx < sm.senses.len()
                            && sm.senses[*sense_idx].status == SenseStatus::Fragile
                        {
                            // Mark as very inactive — purge_fragile will clean it up
                            sm.senses[*sense_idx].inactivity =
                                sm.config.k_fragile + 100; // Well past the limit
                            applied += 1;
                        }
                    }
                }
                // CONFIRM and REVIEW are no-ops (informational only)
                _ => {}
            }
        }

        applied
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sense::GroundingEvidence;

    #[test]
    fn test_well_grounded_confirms() {
        let mut reflection = SenseReflection::new(ReflectionConfig::default());
        let config = SenseConfig::default();

        let mut sm = SenseManager::new(config.clone());
        let _ = sm.ingest(vec![1, 2, 3]);
        let _ = sm.ingest(vec![1, 2, 4]); // Mature now

        // Manually set good grounding
        for _ in 0..10 {
            sm.senses[0].grounding.confirm();
        }

        let mut senses = HashMap::new();
        senses.insert(1, sm);

        let actions = reflection.reflect(&senses, &config);
        let confirms = actions.iter().filter(|a| matches!(a, ReflectionAction::Confirm { .. })).count();
        assert!(confirms > 0);
    }

    #[test]
    fn test_needs_revision_triggers_revise() {
        let mut reflection = SenseReflection::new(ReflectionConfig::default());
        let config = SenseConfig::default();

        let mut sm = SenseManager::new(config.clone());
        let _ = sm.ingest(vec![1, 2, 3]);
        let _ = sm.ingest(vec![1, 2, 4]); // Mature

        // Manually set bad grounding
        for _ in 0..10 {
            sm.senses[0].grounding.contradict(Some("test".to_string()));
        }

        let mut senses = HashMap::new();
        senses.insert(2, sm);

        let actions = reflection.reflect(&senses, &config);
        let revises = actions.iter().filter(|a| matches!(a, ReflectionAction::Revise { .. })).count();
        assert!(revises > 0);
    }
}
