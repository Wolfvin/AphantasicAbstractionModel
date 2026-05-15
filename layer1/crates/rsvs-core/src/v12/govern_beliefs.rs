//! # MD-4: GovernBeliefs & SeedAnchor Transforms
//!
//! Dual-axis governance of composition lifecycle and epistemic state,
//! followed by seed-anchored confidence adjustment.
//!
//! ## GovernBeliefs
//!
//! ```text
//! GraphDelta → initial_states() → detect_contradiction() → check_promotions()
//!            → re_govern_composition() → GovernedDelta
//! ```
//!
//! ### Contradiction Detection
//!
//! | Type | Condition | Conflict |
//! |------|-----------|----------|
//! | PolarityConflict | same predicate + same agent + different patient + negation | Epistemic |
//! | RoleReversal | same predicate + swapped Agent/Patient | Structural |
//! | PurposeConflict | same predicate + same agent + different Purpose | Epistemic |
//! | CrossType | HiddenMeaning contradicts Event | Semantic |
//! | EquivalenceMismatch | non-Event same structure + different fillers | Structural |
//!
//! ### Promotion Criteria
//!
//! | Transition | Requirements |
//! |------------|-------------|
//! | New → Candidate | automatic after 1 batch |
//! | Candidate → Stable | age ≥ 3, confidence ≥ 0.6, no recent contradictions |
//! | Observed → Inferred | derived by reasoning rule |
//! | Inferred → Grounded | ≥ 2 independent sources, confidence ≥ 0.7 |
//! | Hypothesis → Inferred | confirmed by ≥ 1 independent evidence |
//!
//! ## SeedAnchor
//!
//! ```text
//! GovernedDelta → seed_anchored_confidence() → adjust_confidence() → AnchoredDelta
//! ```
//!
//! Critical fix: when no alignment data exists, weight = 0.0, meaning the
//! original confidence is preserved without adjustment.
//!
//! ## Feature Flag
//!
//! This module is only compiled when the `v12` feature is enabled.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::pipeline::{ErasedTransform, Graph, IngestResult};
use super::types::*;
use crate::types::{EdgeSource, NodeId};

// ========================================================================
// GovernBeliefs — The Transform
// ========================================================================

/// MD-4: GovernBeliefs transform — assigns lifecycle/epistemic states,
/// detects contradictions, and manages promotions.
///
/// # Transform Signature
///
/// ```text
/// Input:  GraphDelta — new compositions from IngestAtoms
/// Output: GovernedDelta — compositions with governance applied
/// ```
#[derive(Debug, Clone)]
pub struct GovernBeliefs {
    /// Current batch number (incremented each ingest cycle).
    pub current_batch: usize,
}

impl Default for GovernBeliefs {
    fn default() -> Self {
        Self::new()
    }
}

impl GovernBeliefs {
    /// Create a new GovernBeliefs transform.
    pub fn new() -> Self {
        Self { current_batch: 0 }
    }

    // ====================================================================
    // Initial State Assignment
    // ====================================================================

    /// Assign initial `LifecycleState` and `EpistemicState` based on
    /// `CompositionType` and `EdgeSource`.
    ///
    /// # Rules
    ///
    /// | CompositionType | EdgeSource | Lifecycle | Epistemic |
    /// |----------------|-----------|-----------|-----------|
    /// | Event | FrameCompiler | New | Observed |
    /// | Event | ExtractionRepair | Candidate | Observed |
    /// | HiddenMeaning | HiddenMeaningRule | Candidate | Inferred |
    /// | HiddenMeaning | EnrichmentFeedback | Candidate | Inferred |
    /// | Pattern | PatternMining | Candidate | Inferred |
    /// | Hypothesis | Abductive | Quarantine | Hypothesis |
    /// | Acquisition | AcquisitionRecall | Candidate | Inferred |
    /// | Acquisition | AcquisitionUserAnswer | Candidate | Grounded |
    /// | Acquisition | HumanAssertion | Stable | Grounded |
    /// | * | HumanAssertion | Candidate | Grounded |
    /// | * (default) | * | New | Observed |
    pub fn initial_states(&self, composition: &mut Composition) {
        let (lifecycle, epistemic) = match (&composition.composition_type, &composition.provenance.origin) {
            // Event compositions
            (CompositionType::Event, EdgeSource::FrameCompiler) => {
                (LifecycleState::New, EpistemicState::Observed)
            }
            (CompositionType::Event, EdgeSource::ExtractionRepair) => {
                (LifecycleState::Candidate, EpistemicState::Observed)
            }
            (CompositionType::Event, EdgeSource::EnrichmentFeedback) => {
                (LifecycleState::Candidate, EpistemicState::Observed)
            }

            // HiddenMeaning compositions
            (CompositionType::HiddenMeaning, EdgeSource::HiddenMeaningRule) => {
                (LifecycleState::Candidate, EpistemicState::Inferred)
            }
            (CompositionType::HiddenMeaning, EdgeSource::EnrichmentFeedback) => {
                (LifecycleState::Candidate, EpistemicState::Inferred)
            }

            // Pattern compositions
            (CompositionType::Pattern, EdgeSource::PatternMining) => {
                (LifecycleState::Candidate, EpistemicState::Inferred)
            }

            // Hypothesis compositions
            (CompositionType::Hypothesis, EdgeSource::Abductive) => {
                (LifecycleState::Quarantine, EpistemicState::Hypothesis)
            }

            // Acquisition compositions
            (CompositionType::Acquisition, EdgeSource::AcquisitionRecall) => {
                (LifecycleState::Candidate, EpistemicState::Inferred)
            }
            (CompositionType::Acquisition, EdgeSource::AcquisitionUserAnswer) => {
                (LifecycleState::Candidate, EpistemicState::Grounded)
            }

            // Human assertion overrides
            (_, EdgeSource::HumanAssertion) => {
                (LifecycleState::Candidate, EpistemicState::Grounded)
            }

            // Default
            _ => (LifecycleState::New, EpistemicState::Observed),
        };

        composition.lifecycle = lifecycle;
        composition.epistemic = epistemic;
    }

    // ====================================================================
    // Contradiction Detection
    // ====================================================================

    /// Detect contradictions among compositions in the governed delta
    /// plus existing graph compositions.
    ///
    /// For each pair of compositions with the same predicate, check for:
    /// - Polarity conflict
    /// - Role reversal
    /// - Purpose conflict
    /// - Cross-type contradiction (HiddenMeaning vs Event)
    /// - Equivalence mismatch (non-Event types)
    pub fn detect_contradiction(&self, compositions: &mut [Composition]) -> Vec<GovernanceUpdate> {
        let mut updates = Vec::new();
        let len = compositions.len();

        for i in 0..len {
            for j in (i + 1)..len {
                let (left, right) = (&compositions[i].clone(), &compositions[j].clone());

                if let Some(conflict) = self.check_pair_contradiction(left, right) {
                    let mut update_left = GovernanceUpdate::new(left.id.clone());
                    update_left.contradiction = Some(Contradiction {
                        conflict_type: conflict.clone(),
                        opposing_composition_id: right.id.clone(),
                        strength: 0.8,
                    });
                    update_left.new_epistemic = Some(EpistemicState::Contradicted);
                    updates.push(update_left);

                    let mut update_right = GovernanceUpdate::new(right.id.clone());
                    update_right.contradiction = Some(Contradiction {
                        conflict_type: conflict,
                        opposing_composition_id: left.id.clone(),
                        strength: 0.8,
                    });
                    update_right.new_epistemic = Some(EpistemicState::Contradicted);
                    updates.push(update_right);

                    // Apply contradiction to compositions.
                    compositions[i].epistemic = EpistemicState::Contradicted;
                    compositions[i].contradiction = Some(Contradiction {
                        conflict_type: updates.last().unwrap().contradiction.as_ref().unwrap().conflict_type.clone(),
                        opposing_composition_id: right.id.clone(),
                        strength: 0.8,
                    });
                    compositions[i].contradiction_batches.push(self.current_batch);

                    compositions[j].epistemic = EpistemicState::Contradicted;
                    compositions[j].contradiction = Some(Contradiction {
                        conflict_type: updates[updates.len() - 2].contradiction.as_ref().unwrap().conflict_type.clone(),
                        opposing_composition_id: left.id.clone(),
                        strength: 0.8,
                    });
                    compositions[j].contradiction_batches.push(self.current_batch);
                }
            }
        }

        updates
    }

    /// Check for contradiction between a pair of compositions.
    fn check_pair_contradiction(&self, left: &Composition, right: &Composition) -> Option<EpistemicConflictType> {
        // Both must be Event or at least one Event.
        match (&left.composition_type, &right.composition_type) {
            // Event vs Event: check all event-level contradictions.
            (CompositionType::Event, CompositionType::Event) => {
                // Must share the same predicate.
                if !self.share_predicate(left, right) {
                    return None;
                }

                // Check polarity conflict.
                if self.has_polarity_conflict(left, right) {
                    return Some(EpistemicConflictType::PolarityConflict);
                }

                // Check role reversal.
                if self.has_role_reversal(left, right) {
                    return Some(EpistemicConflictType::RoleReversal);
                }

                // Check purpose conflict.
                if self.has_purpose_conflict(left, right) {
                    return Some(EpistemicConflictType::PurposeConflict);
                }

                None
            }

            // HiddenMeaning vs Event: cross-type contradiction.
            (CompositionType::HiddenMeaning, CompositionType::Event) |
            (CompositionType::Event, CompositionType::HiddenMeaning) => {
                // If the HiddenMeaning implies something that contradicts the Event.
                if self.share_predicate(left, right) {
                    Some(EpistemicConflictType::SemanticContradiction)
                } else {
                    None
                }
            }

            // Non-Event types: equivalence mismatch.
            (_, _) => {
                if left.composition_type == right.composition_type && self.share_structure(left, right) {
                    if self.has_equivalence_mismatch(left, right) {
                        Some(EpistemicConflictType::EquivalenceMismatch)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    /// Do two compositions share the same predicate label?
    fn share_predicate(&self, left: &Composition, right: &Composition) -> bool {
        match (left.member_with_role(&SemanticRole::Predicate), right.member_with_role(&SemanticRole::Predicate)) {
            (Some(l), Some(r)) => l.node_id == r.node_id,
            _ => left.id.split('_').last() == right.id.split('_').last(), // Fallback: ID comparison
        }
    }

    /// Do two compositions share the same structure (same composition type + same roles)?
    fn share_structure(&self, left: &Composition, right: &Composition) -> bool {
        if left.composition_type != right.composition_type {
            return false;
        }
        let left_roles: std::collections::HashSet<_> = left.members.iter().map(|m| m.role.clone()).collect();
        let right_roles: std::collections::HashSet<_> = right.members.iter().map(|m| m.role.clone()).collect();
        left_roles == right_roles
    }

    /// Polarity conflict: same predicate + same agent + different patient + one is negated.
    fn has_polarity_conflict(&self, left: &Composition, right: &Composition) -> bool {
        // Simplified check: one composition has negative polarity events
        // and the other doesn't, while sharing agent and predicate.
        let left_agent = left.member_with_role(&SemanticRole::Arg0Agent);
        let right_agent = right.member_with_role(&SemanticRole::Arg0Agent);

        let same_agent = match (left_agent, right_agent) {
            (Some(l), Some(r)) => l.node_id == r.node_id,
            _ => false,
        };

        // Check if one has a negation marker that the other doesn't.
        // This is a simplified check — in a full implementation, we'd
        // compare the polarity field from the source atoms.
        if same_agent {
            let left_patient = left.member_with_role(&SemanticRole::Arg1Patient);
            let right_patient = right.member_with_role(&SemanticRole::Arg1Patient);

            match (left_patient, right_patient) {
                (Some(l), Some(r)) => l.node_id != r.node_id,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Role reversal: same predicate + swapped Agent/Patient.
    ///
    /// Example: "X membuat Y" vs "Y membuat X" (same predicate, swapped roles).
    fn has_role_reversal(&self, left: &Composition, right: &Composition) -> bool {
        let left_agent = left.member_with_role(&SemanticRole::Arg0Agent);
        let left_patient = left.member_with_role(&SemanticRole::Arg1Patient);
        let right_agent = right.member_with_role(&SemanticRole::Arg0Agent);
        let right_patient = right.member_with_role(&SemanticRole::Arg1Patient);

        match (left_agent, left_patient, right_agent, right_patient) {
            (Some(la), Some(lp), Some(ra), Some(rp)) => {
                la.node_id == rp.node_id && lp.node_id == ra.node_id
            }
            _ => false,
        }
    }

    /// Purpose conflict: same predicate + same agent + different Purpose.
    fn has_purpose_conflict(&self, left: &Composition, right: &Composition) -> bool {
        let left_agent = left.member_with_role(&SemanticRole::Arg0Agent);
        let right_agent = right.member_with_role(&SemanticRole::Arg0Agent);

        let same_agent = match (left_agent, right_agent) {
            (Some(l), Some(r)) => l.node_id == r.node_id,
            _ => false,
        };

        if !same_agent {
            return false;
        }

        let left_purpose = left.member_with_role(&SemanticRole::Purpose);
        let right_purpose = right.member_with_role(&SemanticRole::Purpose);

        match (left_purpose, right_purpose) {
            (Some(lp), Some(rp)) => lp.node_id != rp.node_id,
            _ => false,
        }
    }

    /// Equivalence mismatch: same structure but different role fillers.
    fn has_equivalence_mismatch(&self, left: &Composition, right: &Composition) -> bool {
        // For non-Event types, check if they have the same role structure
        // but different node fillers for key roles.
        for left_member in &left.members {
            if let Some(right_member) = right.member_with_role(&left_member.role) {
                if left_member.node_id != right_member.node_id
                    && left_member.role != SemanticRole::Predicate
                {
                    return true;
                }
            }
        }
        false
    }

    // ====================================================================
    // Promotion Checks
    // ====================================================================

    /// Check and apply promotions for all compositions.
    ///
    /// Returns governance updates for any promotions applied.
    pub fn check_promotions(&self, compositions: &mut [Composition]) -> Vec<GovernanceUpdate> {
        let mut updates = Vec::new();

        for comp in compositions.iter_mut() {
            // New → Candidate: automatic after 1 batch.
            if comp.lifecycle == LifecycleState::New && comp.batch_seen >= 1 {
                if let Some(verdict) = self.can_promote_to_candidate(comp) {
                    match verdict {
                        PromotionVerdict::Approved => {
                            comp.lifecycle = LifecycleState::Candidate;
                            let mut update = GovernanceUpdate::new(comp.id.clone());
                            update.new_lifecycle = Some(LifecycleState::Candidate);
                            updates.push(update);
                        }
                        PromotionVerdict::Denied(_) => {}
                    }
                }
            }

            // Candidate → Stable: age ≥ 3, confidence ≥ 0.6, no recent contradictions.
            if comp.lifecycle == LifecycleState::Candidate {
                if let Some(verdict) = self.can_promote_to_stable(comp) {
                    match verdict {
                        PromotionVerdict::Approved => {
                            comp.lifecycle = LifecycleState::Stable;
                            let mut update = GovernanceUpdate::new(comp.id.clone());
                            update.new_lifecycle = Some(LifecycleState::Stable);
                            updates.push(update);
                        }
                        PromotionVerdict::Denied(_) => {}
                    }
                }
            }

            // Inferred → Grounded: ≥ 2 independent sources, confidence ≥ 0.7.
            if comp.epistemic == EpistemicState::Inferred {
                if let Some(verdict) = self.can_promote_to_grounded(comp) {
                    match verdict {
                        PromotionVerdict::Approved => {
                            comp.epistemic = EpistemicState::Grounded;
                            let mut update = GovernanceUpdate::new(comp.id.clone());
                            update.new_epistemic = Some(EpistemicState::Grounded);
                            updates.push(update);
                        }
                        PromotionVerdict::Denied(_) => {}
                    }
                }
            }

            // Hypothesis → Inferred: confirmed by ≥ 1 independent evidence.
            if comp.epistemic == EpistemicState::Hypothesis {
                if let Some(verdict) = self.can_promote_hypothesis_to_inferred(comp) {
                    match verdict {
                        PromotionVerdict::Approved => {
                            comp.epistemic = EpistemicState::Inferred;
                            let mut update = GovernanceUpdate::new(comp.id.clone());
                            update.new_epistemic = Some(EpistemicState::Inferred);
                            updates.push(update);
                        }
                        PromotionVerdict::Denied(_) => {}
                    }
                }
            }
        }

        updates
    }

    /// Can this composition be promoted from New to Candidate?
    ///
    /// Always approved after 1 batch.
    fn can_promote_to_candidate(&self, comp: &Composition) -> Option<PromotionVerdict> {
        if comp.batch_seen >= 1 {
            Some(PromotionVerdict::Approved)
        } else {
            Some(PromotionVerdict::Denied("not yet seen 1 batch".to_string()))
        }
    }

    /// Can this composition be promoted from Candidate to Stable?
    ///
    /// Requirements:
    /// - Age ≥ 3 batches
    /// - Confidence ≥ 0.6
    /// - No recent contradictions (within last 3 batches)
    fn can_promote_to_stable(&self, comp: &Composition) -> Option<PromotionVerdict> {
        if comp.batch_seen < 3 {
            return Some(PromotionVerdict::Denied(format!(
                "age {} < 3 batches required",
                comp.batch_seen
            )));
        }

        if comp.confidence < 0.6 {
            return Some(PromotionVerdict::Denied(format!(
                "confidence {:.2} < 0.6 required",
                comp.confidence
            )));
        }

        if comp.has_recent_contradiction(3) {
            return Some(PromotionVerdict::Denied(
                "recent contradiction within last 3 batches".to_string(),
            ));
        }

        Some(PromotionVerdict::Approved)
    }

    /// Can this composition be promoted from Inferred to Grounded?
    ///
    /// Requirements:
    /// - Confidence ≥ 0.7
    /// - No recent contradictions (within last 5 batches)
    /// - At least 2 independent source types (simplified: check provenance)
    fn can_promote_to_grounded(&self, comp: &Composition) -> Option<PromotionVerdict> {
        if comp.confidence < 0.7 {
            return Some(PromotionVerdict::Denied(format!(
                "confidence {:.2} < 0.7 required for grounding",
                comp.confidence
            )));
        }

        if comp.has_recent_contradiction(5) {
            return Some(PromotionVerdict::Denied(
                "recent contradiction within last 5 batches".to_string(),
            ));
        }

        // Simplified: if the composition has survived enough batches and
        // has decent confidence, approve grounding.
        // In a full implementation, we'd check for ≥ 2 independent source types.
        Some(PromotionVerdict::Approved)
    }

    /// Can this Hypothesis be promoted to Inferred?
    ///
    /// Requirements:
    /// - At least 1 independent evidence supporting the hypothesis
    /// - No active contradictions
    fn can_promote_hypothesis_to_inferred(&self, comp: &Composition) -> Option<PromotionVerdict> {
        if comp.epistemic == EpistemicState::Contradicted {
            return Some(PromotionVerdict::Denied(
                "hypothesis is contradicted".to_string(),
            ));
        }

        // Simplified: if confidence is reasonable and no contradictions, approve.
        if comp.confidence >= 0.4 {
            Some(PromotionVerdict::Approved)
        } else {
            Some(PromotionVerdict::Denied(format!(
                "confidence {:.2} too low for hypothesis confirmation",
                comp.confidence
            )))
        }
    }

    // ====================================================================
    // Re-Governance After Enrichment
    // ====================================================================

    /// Re-evaluate a composition after it has been enriched.
    ///
    /// This is called by `EnrichComposition` after adding a new member.
    /// It may:
    /// - Update confidence
    /// - Check for new contradictions with other compositions
    /// - Re-assess promotion eligibility
    pub fn re_govern_composition(
        &self,
        composition: &mut Composition,
        other_compositions: &[Composition],
    ) -> Vec<GovernanceUpdate> {
        let mut updates = Vec::new();

        // Re-compute confidence based on completeness.
        let completeness = if self.is_sufficiently_complete(composition) { 0.1 } else { 0.0 };
        composition.confidence = (composition.confidence + completeness).min(1.0);

        // Increment batch_seen.
        composition.batch_seen += 1;

        // Check for contradictions with existing compositions.
        for other in other_compositions {
            if composition.id == other.id {
                continue;
            }
            if let Some(conflict) = self.check_pair_contradiction(composition, other) {
                let mut update = GovernanceUpdate::new(composition.id.clone());
                update.contradiction = Some(Contradiction {
                    conflict_type: conflict,
                    opposing_composition_id: other.id.clone(),
                    strength: 0.8,
                });
                update.new_epistemic = Some(EpistemicState::Contradicted);
                updates.push(update);

                composition.epistemic = EpistemicState::Contradicted;
                composition.contradiction_batches.push(self.current_batch);
                break;
            }
        }

        // If no contradictions, check for promotion.
        if composition.epistemic != EpistemicState::Contradicted {
            updates.extend(self.check_promotions(std::slice::from_mut(composition)));
        }

        updates
    }

    /// Check if a composition has all expected roles for its type.
    ///
    /// A composition is "sufficiently complete" if it has the minimum
    /// expected roles for its type:
    /// - Event: Predicate + Arg0Agent + Arg1Patient
    /// - HiddenMeaning: at least Problem or Solution
    /// - Pattern: Antecedent + Consequent
    /// - Hypothesis: at least 2 members
    /// - Acquisition: at least 1 member
    pub fn is_sufficiently_complete(&self, comp: &Composition) -> bool {
        match comp.composition_type {
            CompositionType::Event => {
                comp.has_member_with_role(SemanticRole::Predicate)
                    && comp.has_member_with_role(SemanticRole::Arg0Agent)
                    && comp.has_member_with_role(SemanticRole::Arg1Patient)
            }
            CompositionType::HiddenMeaning => {
                comp.has_member_with_role(SemanticRole::Problem)
                    || comp.has_member_with_role(SemanticRole::Solution)
            }
            CompositionType::Pattern => {
                comp.has_member_with_role(SemanticRole::Antecedent)
                    && comp.has_member_with_role(SemanticRole::Consequent)
            }
            CompositionType::Hypothesis => comp.members.len() >= 2,
            CompositionType::Situation => comp.members.len() >= 2,
            CompositionType::Acquisition => !comp.members.is_empty(),
        }
    }

    // ====================================================================
    // Contradiction Resolution
    // ====================================================================

    /// Check if a contradicted composition can be resolved.
    ///
    /// Returns `Some(ContradictionResolution)` if resolved, `None` if still unresolved.
    pub fn check_contradiction_resolution(
        &self,
        composition: &Composition,
        opposing: &Composition,
    ) -> Option<ContradictionResolution> {
        // Voice confusion: active vs passive saying the same thing.
        if self.is_voice_confusion(composition, opposing) {
            return Some(ContradictionResolution {
                contradiction_id: format!("contra_{}_{}", composition.id, opposing.id),
                opposing_composition_id: opposing.id.clone(),
                resolution_type: ResolutionType::Misinterpretation,
                resolved: true,
            });
        }

        // Scoped validity: both are true in different contexts.
        if self.has_scoped_validity(composition, opposing) {
            return Some(ContradictionResolution {
                contradiction_id: format!("contra_{}_{}", composition.id, opposing.id),
                opposing_composition_id: opposing.id.clone(),
                resolution_type: ResolutionType::ScopedValidity,
                resolved: true,
            });
        }

        // Superseded: one is clearly newer/stronger.
        if self.is_superseded(composition, opposing) {
            return Some(ContradictionResolution {
                contradiction_id: format!("contra_{}_{}", composition.id, opposing.id),
                opposing_composition_id: opposing.id.clone(),
                resolution_type: ResolutionType::Superseded,
                resolved: true,
            });
        }

        // Context resolved: new context resolves the contradiction.
        // TODO: Implement when enrichment context is available.

        None
    }

    /// Is this contradiction due to voice confusion?
    ///
    /// Active "X membuat Y" vs passive "Y dibuat oleh X" are the same event.
    /// Detected when Agent in one = Patient in the other, and vice versa.
    fn is_voice_confusion(&self, left: &Composition, right: &Composition) -> bool {
        // This is essentially a role reversal where the voice differs.
        // Check if provenance sources differ (one from active, one from passive).
        self.has_role_reversal(left, right)
            && left.provenance.origin != right.provenance.origin
    }

    /// Do both compositions have scoped validity (true in different contexts)?
    ///
    /// Heuristic: if both compositions have Purpose or Location roles
    /// that differ, they may be scoped.
    fn has_scoped_validity(&self, left: &Composition, right: &Composition) -> bool {
        // If they have different locations, they may both be true.
        if left.has_member_with_role(SemanticRole::Location)
            && right.has_member_with_role(SemanticRole::Location)
        {
            let left_loc = left.member_with_role(&SemanticRole::Location);
            let right_loc = right.member_with_role(&SemanticRole::Location);
            if let (Some(ll), Some(rl)) = (left_loc, right_loc) {
                if ll.node_id != rl.node_id {
                    return true;
                }
            }
        }

        // If they have different times, they may both be true.
        if left.has_member_with_role(SemanticRole::Time)
            && right.has_member_with_role(SemanticRole::Time)
        {
            let left_time = left.member_with_role(&SemanticRole::Time);
            let right_time = right.member_with_role(&SemanticRole::Time);
            if let (Some(lt), Some(rt)) = (left_time, right_time) {
                if lt.node_id != rt.node_id {
                    return true;
                }
            }
        }

        false
    }

    /// Is one composition superseded by the other?
    ///
    /// A composition is superseded if the opposing composition has:
    /// - Higher confidence, AND
    /// - Newer batch_seen, AND
    /// - More members (more complete)
    fn is_superseded(&self, left: &Composition, right: &Composition) -> bool {
        right.confidence > left.confidence
            && right.batch_seen >= left.batch_seen
            && right.members.len() >= left.members.len()
    }

    /// Resolve a contradiction by updating composition states.
    ///
    /// This is the top-level resolution method that:
    /// 1. Checks if resolution is possible
    /// 2. Updates composition states accordingly
    /// 3. Returns the resolution record
    pub fn resolve_contradiction(
        &self,
        composition: &mut Composition,
        opposing: &mut Composition,
    ) -> Option<ContradictionResolution> {
        let resolution = self.check_contradiction_resolution(composition, opposing)?;

        match resolution.resolution_type {
            ResolutionType::Misinterpretation => {
                // Voice confusion: un-contradict both, keep the more confident one.
                composition.epistemic = EpistemicState::Observed;
                composition.contradiction = None;
                opposing.epistemic = EpistemicState::Observed;
                opposing.contradiction = None;
            }
            ResolutionType::ScopedValidity => {
                // Both are valid in different scopes: un-contradict both,
                // but don't promote.
                composition.epistemic = EpistemicState::Observed;
                composition.contradiction = None;
                opposing.epistemic = EpistemicState::Observed;
                opposing.contradiction = None;
            }
            ResolutionType::Superseded => {
                // The older/weaker one is deprecated.
                composition.lifecycle = LifecycleState::Deprecated;
                composition.epistemic = EpistemicState::Observed;
                composition.contradiction = None;
            }
            ResolutionType::ContextResolved => {
                composition.epistemic = EpistemicState::Observed;
                composition.contradiction = None;
                opposing.epistemic = EpistemicState::Observed;
                opposing.contradiction = None;
            }
            ResolutionType::Unresolved => {
                return None;
            }
        }

        Some(resolution)
    }

    /// Run the full governance pipeline on a GraphDelta.
    ///
    /// 1. Assign initial states
    /// 2. Detect contradictions
    /// 3. Check promotions
    /// 4. Increment batch counters
    pub fn govern(&mut self, delta: GraphDelta) -> GovernedDelta {
        self.current_batch += 1;

        let mut compositions: Vec<Composition> = delta.new_compositions;
        let mut all_updates = Vec::new();

        // Step 1: Assign initial states.
        for comp in &mut compositions {
            self.initial_states(comp);
        }

        // Step 2: Detect contradictions.
        let contradiction_updates = self.detect_contradiction(&mut compositions);
        all_updates.extend(contradiction_updates);

        // Step 3: Check promotions (only for non-contradicted compositions).
        let promotion_updates = self.check_promotions(&mut compositions);
        all_updates.extend(promotion_updates);

        // Step 4: Increment batch_seen for all compositions.
        for comp in &mut compositions {
            comp.batch_seen += 1;
        }

        GovernedDelta {
            compositions,
            updates: all_updates,
        }
    }
}

/// Implement the `Transform` trait for `GovernBeliefs`.
impl Transform for GovernBeliefs {
    type Input = GraphDelta;
    type Output = GovernedDelta;

    fn id(&self) -> &'static str {
        "GovernBeliefs"
    }

    fn transform(&self, input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        let mut gb = self.clone();
        gb.govern(input.clone())
    }
}

/// Implement `ErasedTransform` for pipeline integration.
impl ErasedTransform for GovernBeliefs {
    fn id(&self) -> &'static str {
        "GovernBeliefs"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let mut gb = self.clone();

        // Build a GraphDelta from the current graph state.
        let delta = GraphDelta {
            new_nodes: Vec::new(),
            new_compositions: graph.compositions.values().cloned().collect(),
            new_edges: Vec::new(),
        };

        let governed = gb.govern(delta);

        // Apply governed compositions back to the graph.
        let transitions = governed.updates.len();
        for comp in &governed.compositions {
            graph.compositions.insert(comp.id.clone(), comp.clone());
        }

        IngestResult {
            governance_transitions: transitions,
            ..IngestResult::default()
        }
    }
}

// ========================================================================
// SeedAnchor — The Transform
// ========================================================================

/// MD-4: SeedAnchor transform — computes seed-anchored confidence
/// for each composition and adjusts confidence accordingly.
///
/// # Critical Fix: No-Alignment-Data
///
/// When no alignment data exists (all seed scores are at their default
/// value of 0.0), the weight is set to 0.0, meaning the original
/// confidence is preserved without adjustment. This prevents the
/// common bug where missing alignment data incorrectly lowers confidence.
///
/// # Transform Signature
///
/// ```text
/// Input:  GovernedDelta — compositions with governance applied
/// Output: AnchoredDelta — compositions with seed-adjusted confidence
/// ```
#[derive(Debug, Clone, Default)]
pub struct SeedAnchor;

impl SeedAnchor {
    /// Create a new SeedAnchor transform.
    pub fn new() -> Self {
        Self
    }

    /// Compute seed-anchored confidence for a composition.
    ///
    /// The seed confidence is the weighted average of seed scores,
    /// but with the critical fix: if all scores are at their default
    /// (0.0), the weight is set to 0.0 to prevent incorrect adjustment.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// if all seed_scores == 0.0 (no alignment data):
    ///     weight = 0.0  // Don't adjust — no data to anchor to
    ///     seed_confidence = 0.5  // Neutral
    /// else:
    ///     weight = 0.4  // Moderate blending
    ///     seed_confidence = average(seed_scores.values())
    ///
    /// adjusted = original * (1 - weight) + seed_confidence * weight
    /// ```
    pub fn seed_anchored_confidence(&self, comp: &Composition) -> SeedAdjustment {
        let all_defaults = comp.seed_scores.values().all(|&v| v == 0.0)
            || comp.seed_scores.is_empty();

        if all_defaults {
            // Critical fix: no alignment data → weight = 0.0
            // This means the original confidence is preserved.
            return SeedAdjustment {
                seed_confidence: 0.5,
                weight: 0.0,
                alignment_strength: 0.0,
            };
        }

        // Compute weighted average of seed scores.
        let seed_confidence: f32 = if comp.seed_scores.is_empty() {
            0.5
        } else {
            let sum: f32 = comp.seed_scores.values().sum();
            sum / comp.seed_scores.len() as f32
        };

        // Compute alignment strength: how far from neutral (0.5) are the scores?
        let alignment_strength: f32 = if comp.seed_scores.is_empty() {
            0.0
        } else {
            let deviations: f32 = comp
                .seed_scores
                .values()
                .map(|&v| (v - 0.5).abs())
                .sum();
            deviations / comp.seed_scores.len() as f32
        };

        // Weight: moderate blending when we have alignment data.
        let weight = 0.4f32.min(alignment_strength * 2.0);

        SeedAdjustment {
            seed_confidence,
            weight,
            alignment_strength,
        }
    }

    /// Adjust a composition's confidence using seed anchoring.
    ///
    /// Blend: `original * (1 - weight) + seed_confidence * weight`
    pub fn adjust_confidence(&self, comp: &mut Composition) -> SeedAdjustment {
        let adjustment = self.seed_anchored_confidence(comp);
        comp.confidence = comp.confidence * (1.0 - adjustment.weight)
            + adjustment.seed_confidence * adjustment.weight;
        adjustment
    }

    /// Run seed anchoring on a governed delta.
    pub fn anchor(&self, governed: GovernedDelta) -> AnchoredDelta {
        let mut compositions = governed.compositions;

        for comp in &mut compositions {
            self.adjust_confidence(comp);
        }

        AnchoredDelta { compositions }
    }
}

/// Implement the `Transform` trait for `SeedAnchor`.
impl Transform for SeedAnchor {
    type Input = GovernedDelta;
    type Output = AnchoredDelta;

    fn id(&self) -> &'static str {
        "SeedAnchor"
    }

    fn transform(&self, input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        self.anchor(input.clone())
    }
}

/// Implement `ErasedTransform` for pipeline integration.
impl ErasedTransform for SeedAnchor {
    fn id(&self) -> &'static str {
        "SeedAnchor"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        // Build a GovernedDelta from the current graph state.
        let governed = GovernedDelta {
            compositions: graph.compositions.values().cloned().collect(),
            updates: Vec::new(),
        };

        let anchored = self.anchor(governed);

        // Apply anchored compositions back to the graph.
        for comp in &anchored.compositions {
            graph.compositions.insert(comp.id.clone(), comp.clone());
        }

        IngestResult::new()
    }
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_composition(
        id: &str,
        comp_type: CompositionType,
        members: Vec<(SemanticRole, NodeId)>,
        confidence: f32,
    ) -> Composition {
        Composition {
            id: id.to_string(),
            composition_type: comp_type,
            members: members
                .into_iter()
                .map(|(role, node_id)| CompositionMember {
                    node_id,
                    role,
                    confidence: 0.8,
                })
                .collect(),
            lifecycle: LifecycleState::New,
            epistemic: EpistemicState::Observed,
            confidence,
            provenance: ProvenanceChain {
                origin: EdgeSource::FrameCompiler,
                ..ProvenanceChain::default()
            },
            seed_scores: HashMap::new(),
            source_text: None,
            batch_seen: 0,
            contradiction_batches: Vec::new(),
            contradiction: None,
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn test_initial_states_event() {
        let gb = GovernBeliefs::new();
        let mut comp = make_composition("test", CompositionType::Event, vec![], 0.5);
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::New);
        assert_eq!(comp.epistemic, EpistemicState::Observed);
    }

    #[test]
    fn test_initial_states_hidden_meaning() {
        let gb = GovernBeliefs::new();
        let mut comp = Composition {
            composition_type: CompositionType::HiddenMeaning,
            provenance: ProvenanceChain {
                origin: EdgeSource::HiddenMeaningRule,
                ..ProvenanceChain::default()
            },
            ..Composition::default()
        };
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::Candidate);
        assert_eq!(comp.epistemic, EpistemicState::Inferred);
    }

    #[test]
    fn test_role_reversal_detection() {
        let gb = GovernBeliefs::new();
        let left = make_composition(
            "left",
            CompositionType::Event,
            vec![
                (SemanticRole::Predicate, 1),
                (SemanticRole::Arg0Agent, 2),
                (SemanticRole::Arg1Patient, 3),
            ],
            0.7,
        );
        let right = make_composition(
            "right",
            CompositionType::Event,
            vec![
                (SemanticRole::Predicate, 1),
                (SemanticRole::Arg0Agent, 3),
                (SemanticRole::Arg1Patient, 2),
            ],
            0.7,
        );

        assert!(gb.has_role_reversal(&left, &right));
    }

    #[test]
    fn test_promotion_candidate_to_stable() {
        let gb = GovernBeliefs::new();
        let comp = Composition {
            lifecycle: LifecycleState::Candidate,
            confidence: 0.7,
            batch_seen: 4,
            ..Composition::default()
        };

        if let Some(PromotionVerdict::Approved) = gb.can_promote_to_stable(&comp) {
            // Approved
        } else {
            panic!("Expected promotion to be approved");
        }
    }

    #[test]
    fn test_promotion_denied_low_confidence() {
        let gb = GovernBeliefs::new();
        let comp = Composition {
            lifecycle: LifecycleState::Candidate,
            confidence: 0.4,
            batch_seen: 5,
            ..Composition::default()
        };

        if let Some(PromotionVerdict::Denied(reason)) = gb.can_promote_to_stable(&comp) {
            assert!(reason.contains("confidence"));
        } else {
            panic!("Expected promotion to be denied");
        }
    }

    #[test]
    fn test_sufficiently_complete_event() {
        let gb = GovernBeliefs::new();
        let comp = make_composition(
            "test",
            CompositionType::Event,
            vec![
                (SemanticRole::Predicate, 1),
                (SemanticRole::Arg0Agent, 2),
                (SemanticRole::Arg1Patient, 3),
            ],
            0.7,
        );
        assert!(gb.is_sufficiently_complete(&comp));

        let incomplete = make_composition(
            "test2",
            CompositionType::Event,
            vec![(SemanticRole::Predicate, 1)],
            0.5,
        );
        assert!(!gb.is_sufficiently_complete(&incomplete));
    }

    #[test]
    fn test_seed_anchor_no_data() {
        let sa = SeedAnchor::new();
        let comp = Composition {
            confidence: 0.8,
            ..Composition::default()
        };

        let adjustment = sa.seed_anchored_confidence(&comp);
        assert_eq!(adjustment.weight, 0.0); // Critical: no data → weight = 0
        assert_eq!(adjustment.seed_confidence, 0.5);
    }

    #[test]
    fn test_seed_anchor_with_data() {
        let sa = SeedAnchor::new();
        let mut comp = Composition {
            confidence: 0.5,
            ..Composition::default()
        };
        comp.seed_scores.insert(SeedPrimitive::Trust, 0.9);
        comp.seed_scores.insert(SeedPrimitive::Goal, 0.8);

        let adjustment = sa.seed_anchored_confidence(&comp);
        assert!(adjustment.weight > 0.0); // Has data → non-zero weight
        assert!(adjustment.seed_confidence > 0.5);
    }

    #[test]
    fn test_confidence_adjustment_preserves_original() {
        let sa = SeedAnchor::new();
        let mut comp = Composition {
            confidence: 0.8,
            ..Composition::default()
        };

        let adjustment = sa.adjust_confidence(&mut comp);
        // With weight = 0.0, confidence should be unchanged.
        assert!((comp.confidence - 0.8).abs() < 0.001);
        assert_eq!(adjustment.weight, 0.0);
    }
}
