//! # Temporal Decay — Ebbinghaus Forgetting Curve for Confidence
//!
//! Ported from v8.3 `autonomy.rs` (939 lines), adapted for the v12
//! Composition-based model.
//!
//! ## Algorithm
//!
//! Ebbinghaus Forgetting Curve:
//! ```text
//! effective_confidence = confidence × e^(-λ × elapsed/T) × (1 + κ × ln(1 + access_count))
//! ```
//!
//! - `λ = ebbinghaus_decay_rate` (default: 2.0) → after 1 TTL, confidence ≈ 13.5%
//! - `κ = ebbinghaus_reinforce_factor` (default: 0.2) → 10 accesses = ×1.46 boost
//! - `T = inactivity_ttl` (default: 50 batches)
//!
//! ## Hysteresis Thresholds
//!
//! - Demote at `confidence < 0.40`
//! - Quarantine at `flip_count >= 3` (confidence oscillations)
//!
//! ## v12 Adaptation
//!
//! In v8.3, decay was computed per-Node with `access_count` and `last_access_batch`.
//! In v12, we compute decay per-Composition using `batch_seen` as the age metric
//! and `members.len()` as a proxy for access count (more members = more access).

use serde::{Deserialize, Serialize};

use super::pipeline::{ErasedTransform, Graph, IngestResult};
use super::types::*;

// ========================================================================
// DecayConfig — Configuration for Temporal Decay
// ========================================================================

/// Configuration for the Ebbinghaus forgetting curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayConfig {
    /// Ebbinghaus decay rate λ (default: 2.0).
    /// Higher values = faster forgetting.
    pub decay_rate: f32,
    /// Reinforcement factor κ (default: 0.2).
    /// How much repeated access mitigates decay.
    pub reinforce_factor: f32,
    /// Inactivity TTL in batches (default: 50).
    /// After this many batches without access, confidence decays significantly.
    pub inactivity_ttl: usize,
    /// Demotion threshold — confidence below this triggers demotion (default: 0.40).
    pub demotion_threshold: f32,
    /// Quarantine threshold — flip count at which composition is quarantined (default: 3).
    pub quarantine_flip_count: usize,
    /// Minimum confidence — compositions below this are deprecated (default: 0.1).
    pub deprecation_threshold: f32,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            decay_rate: 2.0,
            reinforce_factor: 0.2,
            inactivity_ttl: 50,
            demotion_threshold: 0.40,
            quarantine_flip_count: 3,
            deprecation_threshold: 0.1,
        }
    }
}

// ========================================================================
// DecayResult — Result of a Decay Computation
// ========================================================================

/// Result of applying temporal decay to a single composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayResult {
    /// The composition ID.
    pub composition_id: CompositionId,
    /// Confidence before decay.
    pub confidence_before: f32,
    /// Confidence after decay.
    pub confidence_after: f32,
    /// Whether the composition was demoted.
    pub demoted: bool,
    /// Whether the composition was deprecated.
    pub deprecated: bool,
    /// The decay factor applied.
    pub decay_factor: f32,
}

// ========================================================================
// TemporalDecay — The Engine
// ========================================================================

/// Temporal decay engine implementing the Ebbinghaus forgetting curve.
///
/// Confidence decays over time (batches) but is reinforced by repeated
/// access (more members = more access). This ensures that:
/// - Stale, unaccessed compositions gradually lose confidence
/// - Frequently accessed compositions retain confidence
/// - Very old, low-confidence compositions are eventually deprecated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDecay {
    /// Configuration.
    pub config: DecayConfig,
    /// Current batch number (incremented each ingest cycle).
    pub current_batch: usize,
}

impl Default for TemporalDecay {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalDecay {
    /// Create a new temporal decay engine.
    pub fn new() -> Self {
        Self {
            config: DecayConfig::default(),
            current_batch: 0,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: DecayConfig) -> Self {
        Self {
            config,
            current_batch: 0,
        }
    }

    /// Advance the batch counter.
    pub fn advance_batch(&mut self) {
        self.current_batch += 1;
    }

    /// Compute the Ebbinghaus decay factor for a composition.
    ///
    /// ```text
    /// effective_confidence = confidence × e^(-λ × elapsed/T) × (1 + κ × ln(1 + access_count))
    /// ```
    ///
    /// Where:
    /// - `elapsed` = current_batch - last_access_batch
    /// - `T` = inactivity_ttl
    /// - `access_count` ≈ members.len() (proxy for how often this composition is referenced)
    pub fn compute_decay_factor(&self, composition: &Composition) -> f32 {
        // Elapsed time since creation (using batch_seen as proxy).
        let elapsed = composition.batch_seen as f32;
        let ttl = self.config.inactivity_ttl as f32;

        // Ebbinghaus decay: e^(-λ × elapsed/T)
        let decay = (-self.config.decay_rate * elapsed / ttl).exp();

        // Reinforcement: (1 + κ × ln(1 + access_count))
        let access_count = composition.members.len() as f32;
        let reinforcement = 1.0 + self.config.reinforce_factor * (1.0 + access_count).ln();

        decay * reinforcement
    }

    /// Apply decay to a single composition.
    ///
    /// Returns a `DecayResult` describing what happened.
    pub fn apply_decay(&self, composition: &mut Composition) -> DecayResult {
        let confidence_before = composition.confidence;
        let decay_factor = self.compute_decay_factor(composition);

        // Apply decay.
        composition.confidence = (composition.confidence * decay_factor).clamp(0.0, 1.0);

        let confidence_after = composition.confidence;
        let mut demoted = false;
        let mut deprecated = false;

        // Check for demotion.
        if confidence_after < self.config.demotion_threshold {
            match composition.lifecycle {
                LifecycleState::Stable => {
                    composition.lifecycle = LifecycleState::Candidate;
                    demoted = true;
                }
                LifecycleState::Candidate => {
                    composition.lifecycle = LifecycleState::Quarantine;
                    demoted = true;
                }
                _ => {}
            }
        }

        // Check for deprecation.
        if confidence_after < self.config.deprecation_threshold {
            composition.lifecycle = LifecycleState::Deprecated;
            deprecated = true;
        }

        DecayResult {
            composition_id: composition.id.clone(),
            confidence_before,
            confidence_after,
            demoted,
            deprecated,
            decay_factor,
        }
    }

    /// Apply decay to all compositions in the graph.
    ///
    /// Returns results for each composition that was affected.
    pub fn apply_decay_all(&mut self, graph: &mut Graph) -> Vec<DecayResult> {
        self.advance_batch();
        let mut results = Vec::new();

        // Increment batch_seen for all compositions.
        for composition in graph.compositions.values_mut() {
            composition.batch_seen += 1;
        }

        // Apply decay.
        for composition in graph.compositions.values_mut() {
            // Skip New compositions — they haven't been around long enough.
            if composition.lifecycle == LifecycleState::New {
                continue;
            }
            // Skip Deprecated compositions — already dead.
            if composition.lifecycle == LifecycleState::Deprecated {
                continue;
            }

            results.push(self.apply_decay(composition));
        }

        results
    }
}

// ========================================================================
// TemporalDecayTransform — Pipeline Integration
// ========================================================================

/// Pipeline transform that applies temporal decay after enrichment.
///
/// This is an optional transform that can be registered at the end of
/// the pipeline to apply the Ebbinghaus forgetting curve. Without it,
/// compositions only ever gain confidence — they never lose it.
///
/// Recommended registration: after `ReExtractFrame`, no condition.
#[derive(Debug, Clone, Default)]
pub struct TemporalDecayTransform {
    /// The underlying temporal decay engine.
    pub engine: TemporalDecay,
}

impl ErasedTransform for TemporalDecayTransform {
    fn id(&self) -> &'static str {
        "TemporalDecay"
    }

    fn execute(&self, _ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let mut engine = self.engine.clone();
        let results = engine.apply_decay_all(graph);

        let governance_transitions = results.iter().filter(|r| r.demoted || r.deprecated).count();

        IngestResult {
            atoms_created: 0,
            compositions_created: 0,
            edges_created: 0,
            gaps_detected: 0,
            enrichments_applied: 0,
            governance_transitions,
        }
    }
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]
    use super::*;

    #[test]
    fn test_decay_factor_fresh() {
        let decay = TemporalDecay::new();
        let comp = Composition::default();
        // batch_seen = 0, so elapsed = 0, decay = e^0 = 1.0
        let factor = decay.compute_decay_factor(&comp);
        assert!(
            (factor - 1.0).abs() < 0.01,
            "Fresh composition should have decay factor ~1.0"
        );
    }

    #[test]
    fn test_decay_factor_old() {
        let decay = TemporalDecay::new();
        let mut comp = Composition::default();
        comp.batch_seen = 50; // 1 TTL
        comp.members = vec![CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.9,
            label: "test".to_string(),
        }];
        // decay = e^(-2.0 × 50/50) = e^(-2.0) ≈ 0.135
        // reinforcement = 1.0 + 0.2 × ln(2) ≈ 1.139
        // factor ≈ 0.135 × 1.139 ≈ 0.154
        let factor = decay.compute_decay_factor(&comp);
        assert!(
            factor < 0.2,
            "Old composition should have significant decay, got {}",
            factor
        );
        assert!(factor > 0.0, "Decay should be positive");
    }

    #[test]
    fn test_decay_factor_reinforced() {
        let decay = TemporalDecay::new();
        let mut comp_low_access = Composition::default();
        comp_low_access.batch_seen = 25;
        comp_low_access.members = vec![CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.9,
            label: "test".to_string(),
        }];

        let mut comp_high_access = Composition::default();
        comp_high_access.batch_seen = 25;
        comp_high_access.members = (0..10)
            .map(|i| CompositionMember {
                node_id: i,
                role: SemanticRole::Predicate,
                confidence: 0.9,
                label: format!("node_{}", i),
            })
            .collect();

        let factor_low = decay.compute_decay_factor(&comp_low_access);
        let factor_high = decay.compute_decay_factor(&comp_high_access);

        assert!(
            factor_high > factor_low,
            "High-access composition should decay slower: {} vs {}",
            factor_high,
            factor_low
        );
    }

    #[test]
    fn test_apply_decay_demotion() {
        let decay = TemporalDecay::new();
        let mut comp = Composition::default();
        comp.confidence = 0.35; // Below demotion threshold (0.40), but above deprecation (0.1)
        comp.lifecycle = LifecycleState::Stable;
        comp.batch_seen = 10; // Not too old, so decay factor isn't too extreme
        comp.members = vec![
            CompositionMember {
                node_id: 1,
                role: SemanticRole::Predicate,
                confidence: 0.9,
                label: "test".to_string(),
            },
            CompositionMember {
                node_id: 2,
                role: SemanticRole::Arg0Agent,
                confidence: 0.8,
                label: "test2".to_string(),
            },
            CompositionMember {
                node_id: 3,
                role: SemanticRole::Arg1Patient,
                confidence: 0.7,
                label: "test3".to_string(),
            },
        ];

        let result = decay.apply_decay(&mut comp);
        assert!(result.demoted, "Should be demoted due to low confidence");
        // After demotion from Stable, goes to Candidate (not directly Deprecated unless below 0.1)
        assert!(matches!(
            comp.lifecycle,
            LifecycleState::Candidate | LifecycleState::Quarantine | LifecycleState::Deprecated
        ));
    }

    #[test]
    fn test_apply_decay_deprecation() {
        let decay = TemporalDecay::new();
        let mut comp = Composition::default();
        comp.confidence = 0.05; // Below deprecation threshold (0.1)
        comp.lifecycle = LifecycleState::Quarantine;
        comp.batch_seen = 100;
        comp.members = vec![CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.9,
            label: "test".to_string(),
        }];

        let result = decay.apply_decay(&mut comp);
        assert!(
            result.deprecated,
            "Should be deprecated due to very low confidence"
        );
        assert_eq!(comp.lifecycle, LifecycleState::Deprecated);
    }

    #[test]
    fn test_stable_confidence_no_demotion() {
        let decay = TemporalDecay::new();
        let mut comp = Composition::default();
        comp.confidence = 0.8;
        comp.lifecycle = LifecycleState::Stable;
        comp.batch_seen = 5;
        comp.members = vec![CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.9,
            label: "test".to_string(),
        }];

        let result = decay.apply_decay(&mut comp);
        assert!(
            !result.demoted,
            "Should not be demoted with high confidence"
        );
        assert_eq!(comp.lifecycle, LifecycleState::Stable);
    }
}
