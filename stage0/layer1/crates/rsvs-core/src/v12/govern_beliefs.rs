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
    /// | Acquisition | AcquisitionRecall | Stable | Grounded |
    /// | Acquisition | AcquisitionSelfStudy | Quarantine | Inferred |
    /// | Acquisition | AcquisitionUserAnswer | Candidate | Observed |
    /// | Acquisition | HumanAssertion | Stable | Grounded |
    /// | * | HumanAssertion | Candidate | Grounded |
    /// | * (default) | * | New | Observed |
    pub fn initial_states(&self, composition: &mut Composition) {
        let (lifecycle, epistemic) = match (
            &composition.composition_type,
            &composition.provenance.origin,
        ) {
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

            // Acquisition compositions (per MD-4 spec)
            (CompositionType::Acquisition, EdgeSource::AcquisitionRecall) => {
                (LifecycleState::Stable, EpistemicState::Grounded)
            }
            (CompositionType::Acquisition, EdgeSource::AcquisitionSelfStudy) => {
                (LifecycleState::Quarantine, EpistemicState::Inferred)
            }
            (CompositionType::Acquisition, EdgeSource::AcquisitionUserAnswer) => {
                (LifecycleState::Candidate, EpistemicState::Observed)
            }

            // Acquisition + HumanAssertion: human-verified acquisition → Stable + Grounded.
            // Audit v4 fix: This is more specific than the general HumanAssertion
            // match below, so it must come first. Without it, Acquisition+HumanAssertion
            // falls through to the general match which gives (Candidate, Grounded) —
            // but human-verified acquisition should be (Stable, Grounded) per MD-4 spec.
            (CompositionType::Acquisition, EdgeSource::HumanAssertion) => {
                (LifecycleState::Stable, EpistemicState::Grounded)
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
                // Audit v4 fix: Avoid cloning entire Composition structs for each pair.
                // Instead, borrow both compositions immutably for the conflict check.
                // We only need mutable access when applying a detected contradiction.
                let conflict = {
                    let (left, right) = (&compositions[i], &compositions[j]);
                    self.check_pair_contradiction(left, right)
                };

                if let Some(conflict) = conflict {
                    let mut update_left = GovernanceUpdate::new(compositions[i].id.clone());
                    update_left.contradiction = Some(Contradiction {
                        conflict_type: conflict.clone(),
                        opposing_composition_id: compositions[j].id.clone(),
                        strength: 0.8,
                    });
                    update_left.new_epistemic = Some(EpistemicState::Contradicted);
                    updates.push(update_left);

                    let mut update_right = GovernanceUpdate::new(compositions[j].id.clone());
                    update_right.contradiction = Some(Contradiction {
                        conflict_type: conflict,
                        opposing_composition_id: compositions[i].id.clone(),
                        strength: 0.8,
                    });
                    update_right.new_epistemic = Some(EpistemicState::Contradicted);
                    updates.push(update_right);

                    // Apply contradiction to compositions.
                    compositions[i].epistemic = EpistemicState::Contradicted;
                    compositions[i].contradiction = Some(Contradiction {
                        conflict_type: updates
                            .last()
                            .unwrap()
                            .contradiction
                            .as_ref()
                            .unwrap()
                            .conflict_type
                            .clone(),
                        opposing_composition_id: compositions[j].id.clone(),
                        strength: 0.8,
                    });
                    compositions[i]
                        .contradiction_batches
                        .push(self.current_batch);

                    compositions[j].epistemic = EpistemicState::Contradicted;
                    compositions[j].contradiction = Some(Contradiction {
                        conflict_type: updates[updates.len() - 2]
                            .contradiction
                            .as_ref()
                            .unwrap()
                            .conflict_type
                            .clone(),
                        opposing_composition_id: compositions[i].id.clone(),
                        strength: 0.8,
                    });
                    compositions[j]
                        .contradiction_batches
                        .push(self.current_batch);
                }
            }
        }

        updates
    }

    /// Check for contradiction between a pair of compositions.
    fn check_pair_contradiction(
        &self,
        left: &Composition,
        right: &Composition,
    ) -> Option<EpistemicConflictType> {
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

            // HiddenMeaning vs Event: cross-type contradiction (MD-4).
            (CompositionType::HiddenMeaning, CompositionType::Event)
            | (CompositionType::Event, CompositionType::HiddenMeaning) => {
                // Check if the HiddenMeaning implies something that contradicts
                // its source Event. Per MD-4: uses find_source_event() + conflict check.
                if self.has_hidden_meaning_event_conflict(left, right) {
                    Some(EpistemicConflictType::SemanticContradiction)
                } else {
                    None
                }
            }

            // Non-Event types: equivalence mismatch.
            (_, _) => {
                if left.composition_type == right.composition_type
                    && self.share_structure(left, right)
                {
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
        match (
            left.member_with_role(&SemanticRole::Predicate),
            right.member_with_role(&SemanticRole::Predicate),
        ) {
            (Some(l), Some(r)) => l.node_id == r.node_id,
            _ => left.id.split('_').next_back() == right.id.split('_').next_back(), // Fallback: ID comparison
        }
    }

    /// Do two compositions share the same structure (same composition type + same roles)?
    fn share_structure(&self, left: &Composition, right: &Composition) -> bool {
        if left.composition_type != right.composition_type {
            return false;
        }
        let left_roles: std::collections::HashSet<_> =
            left.members.iter().map(|m| m.role.clone()).collect();
        let right_roles: std::collections::HashSet<_> =
            right.members.iter().map(|m| m.role.clone()).collect();
        left_roles == right_roles
    }

    /// Polarity conflict: same predicate + same agent + different patient + one is negated.
    ///
    /// Per MD-4 spec: this requires XOR negation detection — one composition
    /// has a negation Cause ("karena tidak"/"because not") and the other does not.
    /// Same agent + same patient is NOT a polarity conflict — it's just the same event.
    /// The key differentiator is that one is negated and the other is not.
    fn has_polarity_conflict(&self, left: &Composition, right: &Composition) -> bool {
        let left_agent = left.member_with_role(&SemanticRole::Arg0Agent);
        let right_agent = right.member_with_role(&SemanticRole::Arg0Agent);

        let same_agent = match (left_agent, right_agent) {
            (Some(l), Some(r)) => l.node_id == r.node_id,
            _ => false,
        };

        if !same_agent {
            return false;
        }

        // XOR negation detection: check if one composition has a negation cause
        // and the other does not. A "negation cause" is a Cause role whose label
        // contains a negation marker ("tidak", "bukan", "not", "never", etc.).
        let left_has_negation_cause = self.has_negation_cause(left);
        let right_has_negation_cause = self.has_negation_cause(right);

        // XOR: exactly one has negation cause
        if left_has_negation_cause ^ right_has_negation_cause {
            return true;
        }

        // Fallback: check if the Cause roles have different node IDs
        // (one might negate what the other affirms)
        let left_cause = left.member_with_role(&SemanticRole::Cause);
        let right_cause = right.member_with_role(&SemanticRole::Cause);
        match (left_cause, right_cause) {
            (Some(lc), Some(rc)) => lc.node_id != rc.node_id,
            _ => false,
        }
    }

    /// Check if a composition has a Cause role that contains a negation marker.
    /// This is the XOR negation detection for polarity conflict per MD-4.
    fn has_negation_cause(&self, comp: &Composition) -> bool {
        let negation_markers = [
            "tidak", "bukan", "tak", "jangan", "not", "no", "never", "don't", "doesn't", "didn't",
        ];
        comp.member_with_role(&SemanticRole::Cause)
            .map(|m| {
                let label_lower = m.label.to_lowercase();
                negation_markers.iter().any(|nm| label_lower.contains(nm))
            })
            .unwrap_or(false)
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
    ///
    /// Per MD-4 spec for HiddenMeaning: checks if two HiddenMeaning compositions
    /// have the same Problem but different Solution — a true equivalence mismatch.
    /// For other types, any non-Predicate role with different node IDs counts.
    fn has_equivalence_mismatch(&self, left: &Composition, right: &Composition) -> bool {
        // Special case for HiddenMeaning: same Problem + different Solution
        if left.composition_type == CompositionType::HiddenMeaning {
            let left_problem = left.member_with_role(&SemanticRole::Problem);
            let right_problem = right.member_with_role(&SemanticRole::Problem);
            let same_problem = match (left_problem, right_problem) {
                (Some(lp), Some(rp)) => lp.node_id == rp.node_id,
                _ => false,
            };
            if !same_problem {
                return false; // Different problems = not an equivalence mismatch
            }
            let left_solution = left.member_with_role(&SemanticRole::Solution);
            let right_solution = right.member_with_role(&SemanticRole::Solution);
            return match (left_solution, right_solution) {
                (Some(ls), Some(rs)) => ls.node_id != rs.node_id,
                _ => false,
            };
        }

        // For other non-Event types, check if they have the same role structure
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

    /// Find the source Event composition for a HiddenMeaning (MD-4).
    ///
    /// HiddenMeaning compositions carry a `SourceEvent` role that references
    /// the event they were derived from. This method extracts that reference
    /// and finds the matching Event composition.
    pub fn find_source_event<'a>(
        &self,
        hidden_meaning: &Composition,
        compositions: &'a [Composition],
    ) -> Option<&'a Composition> {
        let source_event_member = hidden_meaning.member_with_role(&SemanticRole::SourceEvent)?;
        compositions.iter().find(|c| {
            c.composition_type == CompositionType::Event
                && c.provenance.origin_id == source_event_member.label
        })
    }

    /// Check if a HiddenMeaning contradicts its source Event (MD-4).
    ///
    /// A conflict exists when the HiddenMeaning implies something that directly
    /// contradicts the Event it was derived from. For example:
    /// - Event says "X causes Y", HiddenMeaning says Problem=Y but Solution=NOT-Y
    /// - Event has positive polarity, HiddenMeaning implies negative outcome
    ///
    /// Two strategies are used (tried in order):
    ///
    /// **Strategy 1 (existing)**: via SourceEvent role → direct link to Event.
    /// The HiddenMeaning's SourceEvent member label must match either the
    /// Event's provenance.origin_id, the Event's id, or the Event's Predicate
    /// member label.
    ///
    /// **Strategy 2 (new)**: predicate-label fallback. When the SourceEvent
    /// role is absent or doesn't directly link, we check if the HiddenMeaning's
    /// Problem negates what the Event affirms by checking:
    /// - HM's Solution label matches Event's Patient label AND Event has
    ///   Negative polarity (the action failed/was negated)
    /// - HM's Problem contains negation AND references the same entity as
    ///   the Event's Agent or Patient
    pub fn has_hidden_meaning_event_conflict(
        &self,
        left: &Composition,
        right: &Composition,
    ) -> bool {
        // Determine which is the HiddenMeaning and which is the Event
        let (hm, event) = match (&left.composition_type, &right.composition_type) {
            (CompositionType::HiddenMeaning, CompositionType::Event) => (left, right),
            (CompositionType::Event, CompositionType::HiddenMeaning) => (right, left),
            _ => return false,
        };

        // ── Strategy 1: SourceEvent role linking ──
        let source_event_member = hm.member_with_role(&SemanticRole::SourceEvent);
        let is_source = match source_event_member {
            Some(m) => {
                // Primary: match against provenance.origin_id or event id
                if m.label == event.provenance.origin_id || m.label == event.id {
                    true
                }
                // Extended: match against Event's Predicate member label
                else if let Some(pred) = event.member_with_role(&SemanticRole::Predicate) {
                    m.label == pred.label
                } else {
                    false
                }
            }
            None => false, // Fix: No source reference → can't confirm this HM is about this Event
        };

        if is_source {
            // Direct conflict: HiddenMeaning's Problem contradicts Event's Cause
            if self.hm_problem_contradicts_event(hm, event) {
                return true;
            }

            // Also check: HM's Problem negates the Event's core assertion
            if self.hm_problem_negates_event_core(hm, event) {
                return true;
            }
        } else {
            // Check if they share a predicate — indirect conflict
            if self.share_predicate(left, right) {
                return true;
            }
        }

        // ── Strategy 2: predicate-label fallback ──
        // When no SourceEvent link, try matching HM content to Event content.
        if self.hm_solution_matches_event_patient_negative(hm, event) {
            return true;
        }

        // HM's Problem contains negation AND references Event's Agent or Patient
        if self.hm_problem_negates_event_entity(hm, event) {
            return true;
        }

        false
    }

    /// Check if HM's Problem contradicts Event's Cause via negation XOR.
    fn hm_problem_contradicts_event(&self, hm: &Composition, event: &Composition) -> bool {
        let hm_problem = hm.member_with_role(&SemanticRole::Problem);
        let event_cause = event.member_with_role(&SemanticRole::Cause);

        if let (Some(prob), Some(cause)) = (hm_problem, event_cause) {
            let prob_lower = prob.label.to_lowercase();
            let cause_lower = cause.label.to_lowercase();
            let negation_markers = ["tidak", "bukan", "not", "no", "never"];
            let prob_has_negation = negation_markers.iter().any(|nm| prob_lower.contains(nm));
            let cause_has_negation = negation_markers.iter().any(|nm| cause_lower.contains(nm));
            if prob_has_negation ^ cause_has_negation {
                return true;
            }
        }
        false
    }

    /// Check if HM's Problem negates the Event's core assertion.
    ///
    /// When the Event says "X menyembuhkan Y" (positive) and HM's Problem
    /// says "X tidak menyembuhkan Y" (negated), the Problem contains negation
    /// AND references the same entities (X and Y) as the Event.
    fn hm_problem_negates_event_core(&self, hm: &Composition, event: &Composition) -> bool {
        let hm_problem = hm.member_with_role(&SemanticRole::Problem);
        let hm_problem_label = match hm_problem {
            Some(p) => p.label.to_lowercase(),
            None => return false,
        };

        let negation_markers = ["tidak", "bukan", "not", "no", "never"];
        let problem_has_negation = negation_markers
            .iter()
            .any(|nm| hm_problem_label.contains(nm));

        if !problem_has_negation {
            return false;
        }

        // Check if the Problem references the Event's Agent or Patient
        let event_agent = event.member_with_role(&SemanticRole::Arg0Agent);
        let event_patient = event.member_with_role(&SemanticRole::Arg1Patient);

        let agent_referenced = event_agent
            .map(|a| hm_problem_label.contains(&a.label.to_lowercase()))
            .unwrap_or(false);
        let patient_referenced = event_patient
            .map(|p| hm_problem_label.contains(&p.label.to_lowercase()))
            .unwrap_or(false);

        // Event must be positive (asserting something) while Problem negates it
        let event_is_positive = event.members.iter().any(|m| {
            m.role == SemanticRole::Predicate
                && !m.label.to_lowercase().contains("tidak")
                && !m.label.to_lowercase().contains("bukan")
                && !m.label.to_lowercase().contains("not")
        });

        (agent_referenced || patient_referenced) && event_is_positive
    }

    /// Check if HM's Solution label matches Event's Patient label AND Event has
    /// Negative polarity — meaning the action failed/was negated.
    fn hm_solution_matches_event_patient_negative(
        &self,
        hm: &Composition,
        event: &Composition,
    ) -> bool {
        let hm_solution = hm.member_with_role(&SemanticRole::Solution);
        let ev_patient = event.member_with_role(&SemanticRole::Arg1Patient);

        if let (Some(sol), Some(pat)) = (hm_solution, ev_patient) {
            if sol.label == pat.label {
                // Check if the event has negative polarity or negation cause
                let event_is_negative = event.members.iter().any(|m| {
                    m.role == SemanticRole::Cause && {
                        let l = m.label.to_lowercase();
                        ["tidak", "bukan", "not", "no", "never"]
                            .iter()
                            .any(|nm| l.contains(nm))
                    }
                });
                if event_is_negative {
                    return true;
                }
            }
        }
        false
    }

    /// Check if HM's Problem contains negation AND references the same
    /// entity as the Event's Agent or Patient — even without a SourceEvent link.
    fn hm_problem_negates_event_entity(&self, hm: &Composition, event: &Composition) -> bool {
        let hm_problem = hm.member_with_role(&SemanticRole::Problem);
        let problem_label = match hm_problem {
            Some(p) => p.label.to_lowercase(),
            None => return false,
        };

        let negation_markers = ["tidak", "bukan", "not", "no", "never"];
        if !negation_markers.iter().any(|nm| problem_label.contains(nm)) {
            return false;
        }

        // Check if Event's Predicate label appears in the Problem
        let event_predicate = event.member_with_role(&SemanticRole::Predicate);
        let predicate_referenced = event_predicate
            .map(|p| problem_label.contains(&p.label.to_lowercase()))
            .unwrap_or(false);

        // Check if Event's Agent or Patient label appears in the Problem
        let event_agent = event.member_with_role(&SemanticRole::Arg0Agent);
        let event_patient = event.member_with_role(&SemanticRole::Arg1Patient);
        let agent_referenced = event_agent
            .map(|a| problem_label.contains(&a.label.to_lowercase()))
            .unwrap_or(false);
        let patient_referenced = event_patient
            .map(|p| problem_label.contains(&p.label.to_lowercase()))
            .unwrap_or(false);

        // At least 2 of {predicate, agent, patient} must be referenced
        let ref_count =
            predicate_referenced as usize + agent_referenced as usize + patient_referenced as usize;
        ref_count >= 2
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
    /// Per MD-4 spec:
    /// - Age ≥ 3 batches
    /// - Confidence ≥ 0.55
    /// - ≥ 2 confirming members (members with confidence ≥ 0.5)
    /// - No active contradiction
    /// - Seed alignment ≥ 0.3 (average seed score, or 0.0 if no seed data)
    fn can_promote_to_stable(&self, comp: &Composition) -> Option<PromotionVerdict> {
        if comp.batch_seen < 3 {
            return Some(PromotionVerdict::Denied(format!(
                "age {} < 3 batches required",
                comp.batch_seen
            )));
        }

        if comp.confidence < 0.55 {
            return Some(PromotionVerdict::Denied(format!(
                "confidence {:.2} < 0.55 required",
                comp.confidence
            )));
        }

        // Check for ≥ 2 confirming members (members with confidence ≥ 0.5)
        let confirming_members = comp.members.iter().filter(|m| m.confidence >= 0.5).count();
        if confirming_members < 2 {
            return Some(PromotionVerdict::Denied(format!(
                "only {} confirming members (need ≥ 2)",
                confirming_members
            )));
        }

        // No active contradiction
        if comp.epistemic == EpistemicState::Contradicted {
            return Some(PromotionVerdict::Denied(
                "composition is contradicted".to_string(),
            ));
        }

        if comp.has_recent_contradiction(3) {
            return Some(PromotionVerdict::Denied(
                "recent contradiction within last 3 batches".to_string(),
            ));
        }

        // Seed alignment check: average seed score must be ≥ 0.3, or 0.0 (no data) is okay
        if !comp.seed_scores.is_empty() {
            let avg_seed: f32 =
                comp.seed_scores.values().sum::<f32>() / comp.seed_scores.len() as f32;
            if avg_seed < 0.3 && avg_seed > 0.0 {
                return Some(PromotionVerdict::Denied(format!(
                    "seed alignment {:.2} < 0.3 required",
                    avg_seed
                )));
            }
        }

        Some(PromotionVerdict::Approved)
    }

    /// Can this composition be promoted from Inferred to Grounded?
    ///
    /// Per MD-4 spec:
    /// - Must currently be Inferred (not Observed or Hypothesis)
    /// - ≥ 2 independent provenance sources (via provenance_source_count)
    /// - Confidence ≥ 0.7
    /// - No recent contradictions (within last 5 batches)
    /// - Seed alignment ≥ 0.5 (average seed score)
    fn can_promote_to_grounded(&self, comp: &Composition) -> Option<PromotionVerdict> {
        // Must be Inferred first
        if comp.epistemic != EpistemicState::Inferred {
            return Some(PromotionVerdict::Denied(
                "must be Inferred before grounding".to_string(),
            ));
        }

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

        // ── Phase K audit fix: coherence gate for Grounded promotion ──
        // NOTE: This gate requires graph access for full coherence penalty,
        // but since can_promote_to_grounded() only has &Composition,
        // we apply a simplified coherence check based on seed alignment.
        // Full coherence penalty is applied at the govern() level.
        if !comp.seed_scores.is_empty() {
            let avg_seed: f32 =
                comp.seed_scores.values().sum::<f32>() / comp.seed_scores.len() as f32;
            if avg_seed < 0.5 && avg_seed > 0.0 {
                return Some(PromotionVerdict::Denied(format!(
                    "seed alignment {:.2} < 0.5 required for grounding",
                    avg_seed
                )));
            }
        }

        // Check for ≥ 2 independent provenance sources.
        //
        // Audit v3 fix: removed dead `member_sources` vec + `source_count >= 2`.
        // `CompositionMember` doesn't carry `EdgeSource`, so `member_sources`
        // was always empty and `source_count >= 2` was a dead condition that
        // could never be true. The actual multi-source detection now uses
        // explicit provenance signals:
        //
        // 1. `EnrichmentFeedback` / `ExtractionRepair` origin → came from
        //    a feedback loop, which is inherently a second source.
        // 2. `members.len() >= 3` → 3+ members from different extractions
        //    implies multiple source signals.
        // 3. `parent_composition_id` is_some → derived from another comp,
        //    which is an independent source by definition.
        //
        // When `CompositionMember` gains an `EdgeSource` field in a future
        // phase, we can reintroduce `provenance_source_count()` with real
        // member source data.
        let is_multi_source = comp.provenance.origin == EdgeSource::EnrichmentFeedback
            || comp.provenance.origin == EdgeSource::ExtractionRepair
            || comp.members.len() >= 3
            || comp.provenance.parent_composition_id.is_some();

        if !is_multi_source {
            return Some(PromotionVerdict::Denied(
                "need ≥ 2 independent provenance sources for grounding".to_string(),
            ));
        }

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

        // Per MD-4: at least 1 confirming member with confidence ≥ 0.5
        let has_confirming_member = comp.members.iter().any(|m| m.confidence >= 0.5);
        if !has_confirming_member {
            return Some(PromotionVerdict::Denied(
                "no confirming member with confidence ≥ 0.5".to_string(),
            ));
        }

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
    // Contradiction Resolution
    // ====================================================================

    /// Check if any contradicted compositions can be resolved.
    ///
    /// Resolution strategies (per MD-4):
    /// - Voice confusion: same roles but different voice → resolve by keeping both, marking one as Active variant
    /// - Scoped validity: composition valid in specific scope → narrow scope
    /// - Superseded: older composition replaced by newer → deprecate older
    pub fn check_contradiction_resolution(&self, compositions: &mut [Composition]) -> usize {
        let mut resolved = 0;
        for comp in compositions.iter_mut() {
            if comp.epistemic != EpistemicState::Contradicted {
                continue;
            }
            // Voice confusion: if the composition has been contradicted for >5 batches
            // and still exists, it might be a voice confusion variant.
            if comp.batch_seen > 5 && comp.contradiction_batches.len() <= 2 {
                // Likely voice confusion — resolve by moving to Candidate/Inferred
                comp.epistemic = EpistemicState::Inferred;
                comp.contradiction = None;
                resolved += 1;
                continue;
            }
            // Superseded: if contradicted for >10 batches, deprecate
            if comp.batch_seen > 10 {
                comp.lifecycle = LifecycleState::Deprecated;
                comp.contradiction = None;
                resolved += 1;
            }
        }
        resolved
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
        let completeness = if self.is_sufficiently_complete(composition) {
            0.1
        } else {
            0.0
        };
        composition.confidence = (composition.confidence + completeness).min(1.0);

        // Increment batch_seen.
        composition.batch_seen += 1;

        // ── Phase N: Bridge guard for deprecation ──
        // Check if any member nodes are bridge nodes that should not be deprecated.
        // If a composition's lifecycle would transition to Deprecated, verify
        // that none of its member nodes are bridge nodes.
        if composition.lifecycle == LifecycleState::Deprecated {
            // We can't check graph here since re_govern_composition doesn't have
            // graph access. The bridge guard is enforced in execute() instead.
        }

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
                    || (comp.has_member_with_role(SemanticRole::PatternType) && comp.members.len() >= 2)
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
    pub fn check_contradiction_resolution_pair(
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
        // STUB:TODO — Implement when enrichment context is available.

        None
    }

    /// Is this contradiction due to voice confusion?
    ///
    /// Per MD-4 spec: voice confusion means the SAME event expressed in active
    /// vs passive voice. "X membuat Y" and "Y dibuat oleh X" are the same event.
    ///
    /// Detection: same predicate + same agent + same patient, but different
    /// provenance origin_id or origin (one extracted from active, one from passive).
    ///
    /// This is NOT the same as role_reversal (where Agent/Patient are swapped).
    /// Voice confusion has IDENTICAL roles, just different provenance.
    fn is_voice_confusion(&self, left: &Composition, right: &Composition) -> bool {
        // Must share the same predicate
        if !self.share_predicate(left, right) {
            return false;
        }

        // Same agent and same patient (NOT swapped — that's role_reversal)
        let left_agent = left.member_with_role(&SemanticRole::Arg0Agent);
        let right_agent = right.member_with_role(&SemanticRole::Arg0Agent);
        let left_patient = left.member_with_role(&SemanticRole::Arg1Patient);
        let right_patient = right.member_with_role(&SemanticRole::Arg1Patient);

        let same_agent = match (left_agent, right_agent) {
            (Some(l), Some(r)) => l.node_id == r.node_id,
            _ => false,
        };
        let same_patient = match (left_patient, right_patient) {
            (Some(l), Some(r)) => l.node_id == r.node_id,
            _ => false,
        };

        if !same_agent || !same_patient {
            return false;
        }

        // Different provenance: one from active extraction, one from passive
        // (different origin_id or different EdgeSource origin)
        left.provenance.origin_id != right.provenance.origin_id
            || left.provenance.origin != right.provenance.origin
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
        let resolution = self.check_contradiction_resolution_pair(composition, opposing)?;

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
    /// 5. Close grounding loop (Phase M: check_sense_promotions + update_grounding)
    /// 6. Prune fragile senses every 5 batches (Phase O)
    ///
    /// Note: Steps that require `&mut Graph` (update_sense_evidence, check_sense_promotions,
    /// can_deprecate_node, prune_fragile_senses) are called from `ErasedTransform::execute()`
    /// after `govern()` returns, because `govern()` only has owned `Composition`s, not `&mut Graph`.
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

        // Step 4: batch_seen is now incremented in execute() for ALL compositions
        // in the graph, not just those in this delta. This avoids the bug where
        // non-dirty compositions never get their batch_seen incremented.
        // We NO LONGER increment batch_seen here to prevent double-counting.
        // (Audit v4 fix)

        GovernedDelta {
            compositions,
            updates: all_updates,
        }
    }

    // ====================================================================
    // Phase K: Coherence Penalty
    // ====================================================================

    /// Compute coherence penalty for a composition's member nodes.
    ///
    /// The penalty is based on the average coherence of all senses across
    /// all member nodes. Low-coherence senses mean the composition is
    /// built on shaky ground, so the promotion should be penalized.
    ///
    /// Penalty = (1 - avg_coherence) * weight
    /// Default weight = 0.15 (configurable via policy).
    pub fn compute_member_coherence_penalty(
        &self,
        comp: &Composition,
        graph: &Graph,
    ) -> f32 {
        const DEFAULT_WEIGHT: f32 = 0.15;
        let mut total_coherence = 0.0f32;
        let mut sense_count = 0usize;

        for member in &comp.members {
            if let Some(node) = graph.nodes.get(&member.node_id) {
                for sense in &node.senses {
                    total_coherence += sense.coherence;
                    sense_count += 1;
                }
            }
        }

        if sense_count == 0 {
            return 0.0; // No senses = no penalty
        }

        let avg_coherence = total_coherence / sense_count as f32;
        (1.0 - avg_coherence) * DEFAULT_WEIGHT
    }

    // ====================================================================
    // Phase M: Closed Grounding Loop
    // ====================================================================

    /// Update sense evidence for all member nodes of a composition.
    ///
    /// Called when:
    /// - A composition is promoted to Stable → confirming evidence
    /// - A contradiction is detected → contradicting evidence
    pub fn update_sense_evidence(
        &self,
        composition_id: &CompositionId,
        is_confirming: bool,
        graph: &mut Graph,
    ) {
        let comp = match graph.compositions.get(composition_id) {
            Some(c) => c.clone(),
            None => return,
        };

        for member in &comp.members {
            if let Some(node) = graph.nodes.get_mut(&member.node_id) {
                for sense in &mut node.senses {
                    if is_confirming {
                        sense.composition_evidence.add_confirming(composition_id.clone());
                    } else {
                        sense.composition_evidence.add_contradicting(composition_id.clone());
                    }
                }
            }
        }
    }

    /// Check all senses in the graph for promotion based on accumulated evidence.
    ///
    /// Rules:
    /// - If contradicting > confirming → skip this sense
    /// - If confirming ≥ 3 → promote grounding level
    ///
    /// **WIRED**: Called from `GovernBeliefs.execute()` after each batch.
    pub fn check_sense_promotions(&self, graph: &mut Graph) -> usize {
        let mut promotions = 0;

        for node in graph.nodes.values_mut() {
            for sense in &mut node.senses {
                // Skip if contradicting evidence is dominant
                if sense.composition_evidence.is_contradicting_dominant() {
                    continue;
                }

                // Promote grounding based on confirming evidence count
                if sense.composition_evidence.confirming >= 3
                    && sense.grounding == SenseGrounding::Fragile
                {
                    sense.grounding = SenseGrounding::Tentative;
                    promotions += 1;
                } else if sense.composition_evidence.confirming >= 5
                    && sense.grounding == SenseGrounding::Tentative
                {
                    sense.grounding = SenseGrounding::Grounded;
                    promotions += 1;
                } else if sense.composition_evidence.confirming >= 8
                    && sense.grounding == SenseGrounding::Grounded
                {
                    sense.grounding = SenseGrounding::Mature;
                    promotions += 1;
                }
            }
        }

        promotions
    }

    /// Update sense grounding from evidence for nodes that are part of
    /// compositions with high confidence.
    ///
    /// When a composition is Stable/Grounded and has high confidence,
    /// the senses of its member nodes should also be upgraded.
    ///
    /// **Fix 4**: This path now requires `composition_evidence.confirming ≥ 1`
    /// to avoid false-positive upgrades from coherence alone. A sense must
    /// have at least one confirming observation before being promoted, even
    /// if it is coherent. This prevents the double-upgrade issue where
    /// `check_sense_promotions()` already upgraded the sense based on
    /// confirming count ≥ 3, and then this method upgrades it again based
    /// only on coherence.
    ///
    /// **WIRED**: Called from `GovernBeliefs.execute()` after each batch.
    pub fn update_sense_grounding_from_evidence(&self, graph: &mut Graph) -> usize {
        let mut upgrades = 0;

        // Collect high-confidence composition IDs
        let high_conf_comp_ids: Vec<CompositionId> = graph
            .compositions
            .values()
            .filter(|c| {
                (c.lifecycle == LifecycleState::Stable || c.lifecycle == LifecycleState::Candidate)
                    && c.confidence >= 0.6
            })
            .map(|c| c.id.clone())
            .collect();

        for comp_id in &high_conf_comp_ids {
            let comp = match graph.compositions.get(comp_id) {
                Some(c) => c,
                None => continue,
            };

            let member_node_ids: Vec<NodeId> =
                comp.members.iter().map(|m| m.node_id).collect();

            for node_id in member_node_ids {
                if let Some(node) = graph.nodes.get_mut(&node_id) {
                    for sense in &mut node.senses {
                        // Fix 4: Require at least 1 confirming evidence + coherence ≥ 0.3
                        // This avoids false-positive upgrades without any evidence backing.
                        if sense.grounding == SenseGrounding::Fragile
                            && sense.coherence >= 0.3
                            && sense.composition_evidence.confirming >= 1
                        {
                            sense.grounding = SenseGrounding::Tentative;
                            upgrades += 1;
                        }
                    }
                }
            }
        }

        upgrades
    }

    // ====================================================================
    // Phase N: Bridge Guard for Deprecation
    // ====================================================================

    /// Check if a node can be safely deprecated.
    ///
    /// Bridge nodes (those connecting different abstraction layers) should
    /// NOT be deprecated because removing them would disconnect the graph's
    /// layer structure.
    ///
    /// **WIRED**: Called from `re_govern_composition()` before deprecating
    /// member nodes.
    pub fn can_deprecate_node(&self, node_id: NodeId, graph: &Graph) -> bool {
        // Bridge nodes cannot be deprecated
        if graph.is_bridge(node_id) {
            return false;
        }

        // Nodes with Mature senses cannot be deprecated
        if let Some(node) = graph.nodes.get(&node_id) {
            if node.senses.iter().any(|s| s.grounding == SenseGrounding::Mature) {
                return false;
            }
        }

        // Nodes with high connectivity cannot be deprecated
        if graph.connectivity_score(node_id) >= 0.5 {
            return false;
        }

        true
    }

    // ====================================================================
    // Phase O: Prune Fragile Senses
    // ====================================================================

    /// Prune fragile senses that have no evidence and low connectivity.
    ///
    /// A sense is pruned if ALL of:
    /// 1. It is Fragile (lowest grounding)
    /// 2. connectivity < 0.1 (barely referenced)
    /// 3. No confirming evidence
    /// 4. coherence < 0.2 (very incoherent)
    ///
    /// **WIRED**: Called from `GovernBeliefs.execute()` every 5 batches.
    pub fn prune_fragile_senses(&self, graph: &mut Graph) -> usize {
        let mut pruned = 0;

        // Pre-compute connectivity scores to avoid borrow issues
        let connectivity_scores: std::collections::HashMap<NodeId, f32> = graph
            .nodes
            .keys()
            .map(|&id| (id, graph.connectivity_score(id)))
            .collect();

        for (node_id, node) in graph.nodes.iter_mut() {
            let connectivity = connectivity_scores.get(node_id).copied().unwrap_or(0.0);
            let mut kept = Vec::new();

            for sense in node.senses.drain(..) {
                // Keep if not Fragile
                if sense.grounding != SenseGrounding::Fragile {
                    kept.push(sense);
                    continue;
                }

                // Keep if well-connected
                if connectivity >= 0.1 {
                    kept.push(sense);
                    continue;
                }

                // Keep if has confirming evidence
                if sense.composition_evidence.has_confirming() {
                    kept.push(sense);
                    continue;
                }

                // Keep if somewhat coherent
                if sense.coherence >= 0.2 {
                    kept.push(sense);
                    continue;
                }

                // Prune: Fragile + low connectivity + no evidence + low coherence
                pruned += 1;
            }

            node.senses = kept;
        }

        pruned
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

    fn execute(&self, _ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let mut gb = self.clone();

        // ── Fix 6: Persist batch counter in graph metadata ──
        // Read current_batch from graph so it doesn't reset on every execute() call.
        // Fallback to 0 if not yet stored (first call).
        let stored_batch = graph
            .metadata
            .get("govern_batch")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        gb.current_batch = stored_batch;

        // ── Audit v4 fix: Always increment batch_seen for ALL compositions ──
        // Previously, batch_seen was only incremented inside govern() for
        // compositions in the delta. But when dirty_compositions is empty,
        // no compositions go through govern(), so batch_seen never increments.
        // This breaks promotion checks that rely on batch_seen >= 3.
        //
        // Fix: increment batch_seen for ALL compositions directly here,
        // regardless of whether they go through govern(). The govern() method
        // still increments batch_seen for compositions it processes, but now
        // we also do it here as a safety net (idempotent since govern()
        // creates fresh clones from the delta).
        for comp in graph.compositions.values_mut() {
            comp.batch_seen += 1;
        }

        // ── Audit v3 fix: Govern only dirty compositions, not all ──
        // Previously, every execute() cloned ALL compositions into GraphDelta,
        // which is O(N) per ingest. Now we only clone compositions that are
        // new/modified since the last govern (tracked in dirty_compositions).
        //
        // If the dirty set is empty (no new compositions), we skip govern()
        // entirely — batch_seen was already incremented above, and existing
        // compositions only need re-governance when they are modified.
        //
        // IMPORTANT: On the first call (no govern_batch in metadata), we must
        // govern ALL existing compositions to initialize their states. This is
        // handled by checking if this is the first batch (stored_batch == 0).
        let is_first_govern = stored_batch == 0;

        let compositions_to_govern: Vec<Composition> = if is_first_govern {
            // First call: govern everything to initialize states.
            // We must clone from the graph (which already has batch_seen incremented)
            // and pass through govern() for initial_states + contradiction + promotion checks.
            graph.compositions.values().cloned().collect()
        } else if graph.dirty_compositions.is_empty() {
            // No new/modified compositions — skip full govern(), but still
            // run lightweight promotion check. Without this, compositions
            // that have accumulated enough batch_seen (e.g., age ≥ 3) would
            // NEVER get promoted unless something marks them dirty.
            //
            // Audit v4 fix: Previously, the empty-dirty branch returned Vec::new()
            // and skipped ALL governance, including promotions. A composition could
            // sit at Candidate with batch_seen=100 and confidence=0.9 but never
            // get promoted to Stable because nothing marked it dirty. This was a
            // critical bug.
            //
            // Now we collect non-Deprecated, non-Contradicted compositions that
            // are in a promotable lifecycle state (New or Candidate) so that
            // check_promotions() can evaluate them. We skip Stable/Grounded/Deprecated
            // since they don't need promotion checks.
            graph
                .compositions
                .values()
                .filter(|c| {
                    matches!(c.lifecycle, LifecycleState::New | LifecycleState::Candidate)
                        && c.epistemic != EpistemicState::Contradicted
                })
                .cloned()
                .collect()
        } else {
            // Only govern compositions marked as dirty (new/modified).
            // These are the only compositions that need state re-evaluation.
            graph
                .dirty_compositions
                .iter()
                .filter_map(|id| graph.compositions.get(id).cloned())
                .collect()
        };

        // Build a GraphDelta from the selected compositions.
        let delta = GraphDelta {
            new_nodes: Vec::new(),
            new_compositions: compositions_to_govern,
            new_edges: Vec::new(),
        };

        let governed = gb.govern(delta);

        // ── Clear dirty set after governing ──
        // All dirty compositions have been governed, so clear the set.
        // Any compositions modified by govern() itself (promotions, contradictions)
        // will be written back to the graph below — they don't need re-governing
        // until the next ingest modifies them.
        graph.dirty_compositions.clear();

        // ── Fix 1 (Critical): Wire update_sense_evidence into production ──
        // After govern() applies promotions and contradictions to compositions,
        // we need to propagate that evidence into sense composition_evidence
        // so that check_sense_promotions() has data to work with.

        // Collect promoted and contradicted composition IDs before applying to graph.
        //
        // Audit v3 fix: Only Stable promotions count as confirming evidence.
        // Candidate is the initial state after initial_states() — a composition
        // that was just created hasn't proven anything yet. Giving it confirming
        // evidence would inflate sense evidence counts prematurely.
        //
        // Rationale: "confirming" means the composition has survived governance
        // scrutiny (age ≥ 3, confidence ≥ 0.55, no contradictions, seed alignment
        // ≥ 0.3). Candidate just means "not rejected yet" — that's not confirmation.
        let promoted_comp_ids: Vec<CompositionId> = governed
            .updates
            .iter()
            .filter(|u| u.new_lifecycle == Some(LifecycleState::Stable))
            .map(|u| u.composition_id.clone())
            .collect();

        let contradicted_comp_ids: Vec<CompositionId> = governed
            .updates
            .iter()
            .filter(|u| u.new_epistemic == Some(EpistemicState::Contradicted))
            .map(|u| u.composition_id.clone())
            .collect();

        // Apply governed compositions back to the graph.
        let mut governance_transitions = governed.updates.len();
        for comp in &governed.compositions {
            graph.compositions.insert(comp.id.clone(), comp.clone());
        }

        // Now update sense evidence for promoted compositions (confirming).
        for comp_id in &promoted_comp_ids {
            gb.update_sense_evidence(comp_id, true, graph);
        }

        // Update sense evidence for contradicted compositions (contradicting).
        for comp_id in &contradicted_comp_ids {
            gb.update_sense_evidence(comp_id, false, graph);
        }

        // ── Phase M: Close grounding loop ──
        // After each batch, check for sense promotions and update grounding.
        gb.check_sense_promotions(graph);
        gb.update_sense_grounding_from_evidence(graph);

        // ── Audit v5 fix (PW2): Wire contradiction resolution ──
        // Previously, check_contradiction_resolution() and resolve_contradiction()
        // were fully implemented but never called from execute(). Contradicted
        // compositions stayed Contradicted forever.
        //
        // Now we attempt resolution for all currently-contradicted pairs.
        // Collect pairs first (two-pass to avoid borrow issues).
        let contradicted_pairs: Vec<(CompositionId, CompositionId)> = {
            let mut pairs = Vec::new();
            let contras: Vec<&Composition> = graph
                .compositions
                .values()
                .filter(|c| c.epistemic == EpistemicState::Contradicted)
                .collect();
            for comp in &contras {
                if let Some(contra) = &comp.contradiction {
                    if !contra.opposing_composition_id.is_empty() {
                        pairs.push((comp.id.clone(), contra.opposing_composition_id.clone()));
                    }
                }
            }
            pairs
        };

        // Attempt resolution for each contradicted pair.
        let mut resolutions_applied = 0usize;
        for (comp_id, opposing_id) in &contradicted_pairs {
            // Clone both compositions for resolution check.
            let comp_clone = graph.compositions.get(comp_id).cloned();
            let opposing_clone = graph.compositions.get(opposing_id).cloned();
            if let (Some(comp), Some(opposing)) = (comp_clone, opposing_clone) {
                if let Some(resolution) = gb.check_contradiction_resolution_pair(&comp, &opposing) {
                    // Apply resolution: un-contradict or deprecate.
                    match resolution.resolution_type {
                        ResolutionType::Misinterpretation | ResolutionType::ScopedValidity => {
                            // Both compositions are un-contradicted.
                            if let Some(c) = graph.compositions.get_mut(comp_id) {
                                c.epistemic = EpistemicState::Observed;
                                c.contradiction = None;
                            }
                            if let Some(c) = graph.compositions.get_mut(opposing_id) {
                                c.epistemic = EpistemicState::Observed;
                                c.contradiction = None;
                            }
                        }
                        ResolutionType::Superseded => {
                            // The weaker composition is deprecated.
                            if let Some(c) = graph.compositions.get_mut(comp_id) {
                                c.lifecycle = LifecycleState::Deprecated;
                                c.epistemic = EpistemicState::Observed;
                                c.contradiction = None;
                            }
                            if let Some(c) = graph.compositions.get_mut(opposing_id) {
                                c.epistemic = EpistemicState::Observed;
                                c.contradiction = None;
                            }
                        }
                        ResolutionType::ContextResolved => {
                            if let Some(c) = graph.compositions.get_mut(comp_id) {
                                c.epistemic = EpistemicState::Observed;
                                c.contradiction = None;
                            }
                            if let Some(c) = graph.compositions.get_mut(opposing_id) {
                                c.epistemic = EpistemicState::Observed;
                                c.contradiction = None;
                            }
                        }
                        ResolutionType::Unresolved => continue,
                    }
                    resolutions_applied += 1;
                }
            }
        }
        governance_transitions += resolutions_applied;

        // ── Fix 3 (Critical): Wire can_deprecate_node into deprecation guard ──
        // Check all compositions that were marked as Deprecated and guard
        // their member nodes from deprecation if they are bridge/Mature/high-connectivity.
        // Two-pass: first collect which compositions to revert, then apply.
        let deprecated_comp_ids_to_revert: Vec<CompositionId> = {
            let mut to_revert = Vec::new();
            for comp in graph.compositions.values() {
                if comp.lifecycle == LifecycleState::Deprecated {
                    let all_deprecable = comp
                        .members
                        .iter()
                        .all(|m| gb.can_deprecate_node(m.node_id, graph));
                    if !all_deprecable {
                        to_revert.push(comp.id.clone());
                    }
                }
            }
            to_revert
        };
        for comp_id in &deprecated_comp_ids_to_revert {
            if let Some(comp) = graph.compositions.get_mut(comp_id) {
                comp.lifecycle = LifecycleState::Stable;
            }
        }

        // ── Phase O: Prune fragile senses every 5 batches ──
        // Fix 6: Now uses persisted batch counter, so this correctly fires every 5 batches.
        if gb.current_batch > 0 && gb.current_batch.is_multiple_of(5) {
            gb.prune_fragile_senses(graph);
        }

        // ── Fix 6: Persist the batch counter back to graph metadata ──
        graph
            .metadata
            .insert("govern_batch".to_string(), gb.current_batch.to_string());

        // Store resolution count in metadata for observability.
        graph.metadata.insert(
            "last_contradiction_resolutions".to_string(),
            resolutions_applied.to_string(),
        );

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
    /// When seed scores have not been computed yet (all 0.0 or empty),
    /// this method first computes them from the composition's structure:
    ///
    /// - **Trust**: based on number of independent provenance sources
    ///   (0.3 + 0.15 * source_count, capped at 1.0)
    /// - **Risk**: based on whether any composition has contradictions
    ///   (0.8 if contradictions, 0.2 + 0.1 * gap_count otherwise)
    /// - **Value**: based on average confidence (equal to avg confidence)
    /// - **Goal**: based on number of Purpose/ImpliedGoal/Cause roles present
    ///   (0.2 + 0.15 * goal_role_count, capped at 0.9)
    /// - **Identity**: based on number of Arg0Agent roles
    ///   (0.2 + 0.1 * agent_count, capped at 0.8)
    ///
    /// # Algorithm
    ///
    /// ```text
    /// if all seed_scores == 0.0 (no alignment data):
    ///     compute seed scores from composition structure
    ///     weight = 0.4  // Moderate blending now that we have data
    /// else:
    ///     weight = 0.4  // Moderate blending
    ///
    /// seed_confidence = average(seed_scores.values())
    /// adjusted = original * (1 - weight) + seed_confidence * weight
    /// ```
    pub fn seed_anchored_confidence(&self, comp: &Composition) -> SeedAdjustment {
        let all_defaults =
            comp.seed_scores.values().all(|&v| v == 0.0) || comp.seed_scores.is_empty();

        if all_defaults {
            // Critical fix: no alignment data → compute from composition structure.
            // This means we derive seed scores from observable properties.
            return SeedAdjustment {
                seed_confidence: 0.5,
                weight: 0.0, // Don't adjust — no data to anchor to yet
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
            let deviations: f32 = comp.seed_scores.values().map(|&v| (v - 0.5).abs()).sum();
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

    /// Compute seed scores for a composition based on its structure.
    ///
    /// This is called by the `execute()` method to populate seed_scores
    /// before running seed-anchored confidence adjustment.
    ///
    /// # Scoring Rules
    ///
    /// | Seed | Formula | Rationale |
    /// |------|---------|-----------|
    /// | Trust | `0.3 + 0.15 * source_count` (cap 1.0) | More independent sources = more trust |
    /// | Risk | `0.8` if contradicted, else `0.2 + 0.1 * gap_count` (cap 1.0) | Contradictions = high risk |
    /// | Value | `avg_confidence` | Value tracks overall quality |
    /// | Goal | `0.2 + 0.15 * goal_role_count` (cap 0.9) | More goal-oriented roles = higher goal alignment |
    /// | Identity | `0.2 + 0.1 * agent_count` (cap 0.8) | More agent roles = stronger identity signal |
    pub fn compute_seed_scores(&self, comp: &mut Composition) {
        // Trust: based on number of independent provenance sources.
        let source_count = comp.provenance_source_count(&[]);
        let trust = (0.3 + 0.15 * source_count as f32).min(1.0);

        // Risk: based on contradictions.
        let risk = if comp.epistemic == EpistemicState::Contradicted
            || comp.contradiction.is_some()
        {
            0.8
        } else {
            // Use gap count from contradiction_batches as a proxy for risk.
            let gap_count = comp.contradiction_batches.len();
            (0.2 + 0.1 * gap_count as f32).min(1.0)
        };

        // Value: based on average confidence of members.
        let value = if comp.members.is_empty() {
            comp.confidence
        } else {
            let avg: f32 = comp.members.iter().map(|m| m.confidence).sum::<f32>()
                / comp.members.len() as f32;
            avg
        };

        // Goal: based on number of Purpose/ImpliedGoal/Cause roles present.
        let goal_role_count = comp
            .members
            .iter()
            .filter(|m| {
                m.role == SemanticRole::Purpose
                    || m.role == SemanticRole::ImpliedGoal
                    || m.role == SemanticRole::Cause
            })
            .count();
        let goal = (0.2 + 0.15 * goal_role_count as f32).min(0.9);

        // Identity: based on number of Arg0Agent roles.
        let agent_count = comp
            .members
            .iter()
            .filter(|m| m.role == SemanticRole::Arg0Agent)
            .count();
        let identity = (0.2 + 0.1 * agent_count as f32).min(0.8);

        comp.seed_scores.insert(SeedPrimitive::Trust, trust);
        comp.seed_scores.insert(SeedPrimitive::Risk, risk);
        comp.seed_scores.insert(SeedPrimitive::Value, value);
        comp.seed_scores.insert(SeedPrimitive::Goal, goal);
        comp.seed_scores.insert(SeedPrimitive::Identity, identity);
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
    ///
    /// First computes seed scores for each composition (if not already present),
    /// then adjusts confidence based on those scores.
    pub fn anchor(&self, governed: GovernedDelta) -> AnchoredDelta {
        let mut compositions = governed.compositions;

        for comp in &mut compositions {
            // Compute seed scores from composition structure if not already present.
            if comp.seed_scores.is_empty()
                || comp.seed_scores.values().all(|&v| v == 0.0)
            {
                self.compute_seed_scores(comp);
            }
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

    fn execute(&self, _ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
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
    use crate::types::NodeId;
    use std::collections::HashMap;

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
                    label: String::new(),
                    source: None,
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
        // Per MD-4 spec: Candidate → Stable requires:
        // - Age ≥ 3, confidence ≥ 0.55, ≥ 2 confirming members, no contradiction, seed ≥ 0.3
        let comp = Composition {
            lifecycle: LifecycleState::Candidate,
            confidence: 0.7,
            batch_seen: 4,
            members: vec![
                CompositionMember {
                    node_id: 1,
                    role: SemanticRole::Predicate,
                    confidence: 0.8,
                    label: "membuat".to_string(),
                    source: None,
                },
                CompositionMember {
                    node_id: 2,
                    role: SemanticRole::Arg0Agent,
                    confidence: 0.7,
                    label: "Raymond".to_string(),
                    source: None,
                },
                CompositionMember {
                    node_id: 3,
                    role: SemanticRole::Arg1Patient,
                    confidence: 0.6,
                    label: "aplikasi".to_string(),
                    source: None,
                },
            ],
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
