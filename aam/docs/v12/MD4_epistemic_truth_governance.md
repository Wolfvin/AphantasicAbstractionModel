# MD-4 — Epistemic Truth & Belief Governance (Elegant Architecture)

> **Prerequisite**: MD-3 defines LifecycleState, EpistemicState (two orthogonal axes),
> Composition, SemanticEdge, ProvenanceChain, EdgeSource, Seed Anchoring.
> This document defines the `GovernBeliefs` and `SeedAnchor` Transforms.
>
> MD-3 now also defines EnrichmentRequest, EnrichmentSource, RecallAction,
> EnrichComposition Transform, and ReExtractFrame Transform for the feedback loop.
> This document defines how GovernBeliefs handles re-governance after enrichment.

---

## Mission

Implement belief governance as two Transforms:

1. **GovernBeliefs** — assigns and transitions LifecycleState + EpistemicState on Compositions
2. **SeedAnchor** — evaluates confidence via seed alignment, replacing source trust weights

These two transforms replace the entire patchwork of NodeStatus, CandidateStatus,
BeliefState, GroundingVerdict, and source trust weight systems.

---

## Core Principle

RSVS must distinguish **structural maturity** from **epistemic confidence**.

```text
LifecycleState:  how established is this entity in the graph?
EpistemicState:  how confident are we in this knowledge?

These are ORTHOGONAL. Changing one must not affect the other.
```

---

## GovernBeliefs Transform

```rust
/// GovernBeliefs Transform
///
/// Input:  GraphDelta (new nodes and compositions from ingest)
/// Output: GovernedDelta (same + lifecycle/epistemic assignments + transitions)
pub struct GovernBeliefs {
    config: GovernanceConfig,
}

impl Transform for GovernBeliefs {
    type Input = GraphDelta;
    type Output = GovernedDelta;

    fn id(&self) -> &'static str { "GovernBeliefs" }

    fn transform(&self, delta: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        let mut governed = GovernedDelta::from(delta.clone());

        for composition in &delta.new_compositions {
            // 1. Assign initial states based on composition type and source
            let (lifecycle, epistemic) = self.initial_states(composition);
            governed.set_states(composition.id.clone(), lifecycle, epistemic);

            // 2. Check for contradictions with existing graph
            if let Some(contradiction) = self.detect_contradiction(composition, &ctx.graph) {
                governed.mark_contradicted(composition.id.clone(), contradiction);
            }

            // 3. Check for promotion eligibility (existing compositions)
            self.check_promotions(&mut governed, &ctx.graph);
        }

        governed
    }
}
```

### Initial State Assignment

```rust
impl GovernBeliefs {
    fn initial_states(&self, comp: &Composition) -> (LifecycleState, EpistemicState) {
        match comp.composition_type {
            // Events from direct extraction
            CompositionType::Event => {
                match comp.provenance.origin {
                    EdgeSource::FrameCompiler => (LifecycleState::New, EpistemicState::Observed),
                    _ => (LifecycleState::New, EpistemicState::Inferred),
                }
            },

            // Hidden meanings from pre-ingest reasoning
            CompositionType::HiddenMeaning => {
                (LifecycleState::Quarantine, EpistemicState::Inferred)
            },

            // Patterns from mining
            CompositionType::Pattern => {
                (LifecycleState::Candidate, EpistemicState::Inferred)
            },

            // Hypotheses from abductive/predictive
            CompositionType::Hypothesis => {
                (LifecycleState::Quarantine, EpistemicState::Hypothesis)
            },

            // Externally acquired
            CompositionType::Acquisition => {
                match comp.provenance.origin {
                    EdgeSource::AcquisitionUserAnswer => (LifecycleState::Candidate, EpistemicState::Observed),
                    EdgeSource::AcquisitionSelfStudy => (LifecycleState::Quarantine, EpistemicState::Inferred),
                    EdgeSource::AcquisitionRecall => (LifecycleState::Stable, EpistemicState::Grounded),
                    _ => (LifecycleState::Candidate, EpistemicState::Inferred),
                }
            },

            // Situations
            CompositionType::Situation => {
                (LifecycleState::Candidate, EpistemicState::Inferred)
            },
        }
    }
}
```

### Contradiction Detection

```rust
pub struct Contradiction {
    pub conflict_type: EpistemicConflictType,
    pub opposing_composition_id: CompositionId,
    pub strength: f32,
}

/// Epistemic-level conflict taxonomy.
/// Separate from v11.0's meaning-pathway ConflictType.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EpistemicConflictType {
    PolarityConflict,
    PurposeConflict,
    AgentConflict,
    PatientConflict,
    CauseConflict,
    TemporalConflict,
    LocationConflict,
    SemanticContradiction,
    RoleReversal,
    EquivalenceMismatch,
}

impl GovernBeliefs {
    fn detect_contradiction(&self, comp: &Composition, graph: &Graph) -> Option<Contradiction> {
        // Find compositions with same predicate but conflicting roles.
        // NOT limited to Events — HiddenMeaning, Pattern, etc. can also contradict.
        let same_predicate: Vec<&Composition> = graph.compositions()
            .filter(|c| c.composition_type == comp.composition_type)  // same type
            .filter(|c| c.has_member_with_role(SemanticRole::Predicate, &comp))
            .collect();

        for other in same_predicate {
            // Polarity conflict
            if self.has_polarity_conflict(comp, other) {
                return Some(Contradiction {
                    conflict_type: EpistemicConflictType::PolarityConflict,
                    opposing_composition_id: other.id.clone(),
                    strength: 0.90,
                });
            }

            // Role reversal
            if self.has_role_reversal(comp, other) {
                return Some(Contradiction {
                    conflict_type: EpistemicConflictType::RoleReversal,
                    opposing_composition_id: other.id.clone(),
                    strength: 0.85,
                });
            }

            // Purpose conflict
            if self.has_purpose_conflict(comp, other) {
                return Some(Contradiction {
                    conflict_type: EpistemicConflictType::PurposeConflict,
                    opposing_composition_id: other.id.clone(),
                    strength: 0.70,
                });
            }
        }

        // Cross-type contradictions: HiddenMeaning vs Event
        // A HiddenMeaning that contradicts its source Event is a semantic contradiction.
        if comp.composition_type == CompositionType::HiddenMeaning {
            if let Some(source_event) = self.find_source_event(comp, graph) {
                if self.has_hidden_meaning_event_conflict(comp, &source_event) {
                    return Some(Contradiction {
                        conflict_type: EpistemicConflictType::SemanticContradiction,
                        opposing_composition_id: source_event.id.clone(),
                        strength: 0.75,
                    });
                }
            }
        }

        // Same-type non-Event contradictions (e.g., two ProblemSolution with
        // different Solutions for the same Problem)
        if comp.composition_type != CompositionType::Event {
            // FIX: Instead of .any() which discards which composition caused
            // the mismatch, use .find() to capture the opposing composition.
            if let Some(opposing) = graph.compositions()
                .filter(|c| c.composition_type == comp.composition_type)
                .filter(|c| c.id != comp.id)
                .find(|c| self.has_equivalence_mismatch(comp, c))
            {
                return Some(Contradiction {
                    conflict_type: EpistemicConflictType::EquivalenceMismatch,
                    opposing_composition_id: opposing.id.clone(),  // the OTHER composition
                    strength: 0.65,
                });
            }
        }

        None
    }

    /// Check if a HiddenMeaning contradicts its source Event.
    /// Example: Event says "X causes Y" but HiddenMeaning says "Y is solution for X"
    /// with inverted polarity — these are compatible, not contradictory.
    /// But if HiddenMeaning says "X is NOT a problem" while Event says "X caused Y"
    /// with negative polarity, that's a contradiction.
    fn has_hidden_meaning_event_conflict(&self, hm: &Composition, event: &Composition) -> bool {
        // HiddenMeaning's Problem should align with Event's cause or patient.
        // If HiddenMeaning says Problem=X but Event says Cause=Y (X != Y),
        // and the polarity is the same, it's a mismatch.
        let hm_problem = hm.member_with_role(&SemanticRole::Problem);
        let event_cause = event.member_with_role(&SemanticRole::Cause);

        if let (Some(hm_p), Some(ev_c)) = (hm_problem, event_cause) {
            // If the problem references a different node than the cause,
            // AND both have same polarity — they're talking about different things,
            // not contradictory. But if same predicate + different role filler — mismatch.
            return hm_p.node_id == ev_c.node_id
                && hm.members.len() > 1 && event.members.len() > 1
                && self.has_role_reversal(hm, event);
        }
        false
    }

    /// Check if two compositions of the same type have the same roles
    /// but different fillers — an equivalence mismatch.
    /// Example: ProblemSolution(Problem=lambat, Solution=aplikasi)
    /// vs       ProblemSolution(Problem=lambat, Solution=manual)
    fn has_equivalence_mismatch(&self, comp_a: &Composition, comp_b: &Composition) -> bool {
        // Same structure, but at least one role has a different filler
        let same_predicate = comp_a.member_with_role(&SemanticRole::Predicate)
            == comp_b.member_with_role(&SemanticRole::Predicate);

        if !same_predicate { return false; }

        // Check if Problem or PatternType matches but Solution or other role differs
        let same_problem = comp_a.member_with_role(&SemanticRole::Problem)
            == comp_b.member_with_role(&SemanticRole::Problem);
        let different_solution = comp_a.member_with_role(&SemanticRole::Solution)
            != comp_b.member_with_role(&SemanticRole::Solution);

        same_problem && different_solution
    }

    /// Find the source event for a HiddenMeaning composition.
    fn find_source_event(&self, hm: &Composition, graph: &Graph) -> Option<Composition> {
        if let Some(source_event_id) = hm.member_with_role(&SemanticRole::SourceEvent) {
            graph.get_composition_by_node_id(source_event_id.node_id)
        } else {
            None
        }
    }

    // === Contradiction Predicate Implementations ===
    //
    // These three functions are the core of detect_contradiction().
    // Without them, contradiction detection cannot function at all.
    // Each compares two compositions of the same type for specific
    // conflict patterns.

    /// Do two compositions have conflicting polarity?
    ///
    /// Two compositions with the SAME predicate but OPPOSITE polarity
    /// are in direct contradiction. For example:
    ///   Event("membuat", polarity=Positive) vs Event("membuat", polarity=Negative)
    ///   meaning: "X created Y" vs "X did NOT create Y"
    ///
    /// For Compositions, polarity is not a direct field (it lives on the
    /// SemanticAtom). Instead, we detect polarity conflict through role
    /// analysis: if both compositions have the same predicate + same agent
    /// but the Patient or Cause directly negates each other (one references
    /// a negation node), it's a polarity conflict.
    fn has_polarity_conflict(&self, comp_a: &Composition, comp_b: &Composition) -> bool {
        // Same predicate is required for any structural comparison
        let same_predicate = comp_a.member_with_role(&SemanticRole::Predicate)
            == comp_b.member_with_role(&SemanticRole::Predicate);
        if !same_predicate { return false; }

        // Same agent performing the action
        let same_agent = comp_a.member_with_role(&SemanticRole::Arg0Agent)
            == comp_b.member_with_role(&SemanticRole::Arg0Agent);

        // Different patient — one might be a negation
        let different_patient = comp_a.member_with_role(&SemanticRole::Arg1Patient)
            != comp_b.member_with_role(&SemanticRole::Arg1Patient);

        // One has a negation-related cause ("not", "bukan", "tidak")
        // This is detected by the cause referencing a negation node
        let has_negation_cause = comp_a.member_with_role(&SemanticRole::Cause)
            .map(|m| m.node_id)
            .xor(comp_b.member_with_role(&SemanticRole::Cause).map(|m| m.node_id))
            .is_some();

        same_agent && different_patient && has_negation_cause
    }

    /// Do two compositions have reversed roles?
    ///
    /// Role reversal means: composition A has X as Agent and Y as Patient,
    /// while composition B has Y as Agent and X as Patient. Same predicate,
    /// but the direction of the action is reversed.
    ///
    /// Example:
    ///   "Raymond membuat aplikasi" (Agent=Raymond, Patient=aplikasi)
    ///   vs "Aplikasi membuat Raymond" (Agent=aplikasi, Patient=Raymond)
    ///   — this is a role reversal, likely a contradiction or misinterpretation.
    fn has_role_reversal(&self, comp_a: &Composition, comp_b: &Composition) -> bool {
        let same_predicate = comp_a.member_with_role(&SemanticRole::Predicate)
            == comp_b.member_with_role(&SemanticRole::Predicate);
        if !same_predicate { return false; }

        let a_agent = comp_a.member_with_role(&SemanticRole::Arg0Agent);
        let a_patient = comp_a.member_with_role(&SemanticRole::Arg1Patient);
        let b_agent = comp_b.member_with_role(&SemanticRole::Arg0Agent);
        let b_patient = comp_b.member_with_role(&SemanticRole::Arg1Patient);

        // Agent in A == Patient in B AND Patient in A == Agent in B
        match (a_agent, a_patient, b_agent, b_patient) {
            (Some(aa), Some(ap), Some(ba), Some(bp)) => {
                aa.node_id == bp.node_id && ap.node_id == ba.node_id
            },
            _ => false,
        }
    }

    /// Do two compositions have conflicting purposes?
    ///
    /// Purpose conflict occurs when two compositions with the same predicate
    /// and agent have contradictory Purpose roles. For example:
    ///   "Raymond membuat aplikasi karena lambat" (Purpose=lambat)
    ///   vs "Raymond membuat aplikasi karena cepat" (Purpose=cepat)
    ///   — these have conflicting purposes for the same action.
    fn has_purpose_conflict(&self, comp_a: &Composition, comp_b: &Composition) -> bool {
        let same_predicate = comp_a.member_with_role(&SemanticRole::Predicate)
            == comp_b.member_with_role(&SemanticRole::Predicate);
        if !same_predicate { return false; }

        let same_agent = comp_a.member_with_role(&SemanticRole::Arg0Agent)
            == comp_b.member_with_role(&SemanticRole::Arg0Agent);
        if !same_agent { return false; }

        // Both must have a Purpose role, but with different fillers
        let a_purpose = comp_a.member_with_role(&SemanticRole::Purpose);
        let b_purpose = comp_b.member_with_role(&SemanticRole::Purpose);

        match (a_purpose, b_purpose) {
            (Some(ap), Some(bp)) => ap.node_id != bp.node_id,
            _ => false, // no conflict if one or both lack Purpose
        }
    }
}
```

### Promotion Criteria

Concrete promotion criteria. Every lifecycle and epistemic transition has explicit
thresholds. Without these, `check_promotions()` cannot be implemented.

```rust
/// Concrete promotion criteria.
/// Every lifecycle and epistemic transition has explicit thresholds.
impl GovernBeliefs {
    /// Can this composition promote from Candidate → Stable?
    fn can_promote_to_stable(&self, comp: &Composition, graph: &Graph) -> PromotionVerdict {
        // Criterion 1: Minimum confidence
        if comp.confidence < 0.55 {
            return PromotionVerdict::Denied("confidence below 0.55 threshold");
        }

        // Criterion 2: At least N confirming contexts (from sense grounding)
        let confirming = comp.members.iter()
            .filter(|m| m.confidence >= 0.5)
            .count();
        if confirming < 2 {
            return PromotionVerdict::Denied(format!(
                "only {} confirming members (need ≥ 2)", confirming));
        }

        // Criterion 3: No active contradictions
        if comp.epistemic == EpistemicState::Contradicted {
            return PromotionVerdict::Denied("active contradiction unresolved");
        }

        // Criterion 4: Has existed for at least K ingest batches (maturity)
        if comp.age_in_batches() < 3 {
            return PromotionVerdict::Denied(format!(
                "too young ({} batches, need ≥ 3)", comp.age_in_batches()));
        }

        // Criterion 5: Seed alignment is not negative
        let seed_conf = self.seed_anchored_confidence(&comp.seed_scores);
        if seed_conf < 0.3 {
            return PromotionVerdict::Denied("seed alignment too low (negative signal)");
        }

        PromotionVerdict::Approved
    }

    /// Can this composition's epistemic state promote from Inferred → Grounded?
    fn can_promote_to_grounded(&self, comp: &Composition, graph: &Graph) -> PromotionVerdict {
        // Criterion 1: Must be Inferred first (cannot skip from Observed to Grounded)
        if comp.epistemic != EpistemicState::Inferred {
            return PromotionVerdict::Denied("must be Inferred before Grounded");
        }

        // Criterion 2: Independent confirmation from ≥ 2 different sources
        let source_count = comp.provenance_source_count(graph);
        if source_count < 2 {
            return PromotionVerdict::Denied(format!(
                "only {} independent source(s), need ≥ 2", source_count));
        }

        // Criterion 3: High confidence (≥ 0.7)
        if comp.confidence < 0.7 {
            return PromotionVerdict::Denied("confidence below 0.7 for Grounded");
        }

        // Criterion 4: No contradictions in last K batches
        if comp.has_recent_contradiction(5) {
            return PromotionVerdict::Denied("recent contradiction within 5 batches");
        }

        // Criterion 5: Seed alignment confirms
        let seed_conf = self.seed_anchored_confidence(&comp.seed_scores);
        if seed_conf < 0.5 {
            return PromotionVerdict::Denied("seed alignment insufficient for Grounded");
        }

        PromotionVerdict::Approved
    }

    /// Can a Hypothesis promote to Inferred?
    fn can_promote_hypothesis_to_inferred(&self, comp: &Composition) -> PromotionVerdict {
        // Must have at least one confirming observation
        let confirming = comp.members.iter()
            .filter(|m| m.confidence >= 0.5)
            .count();
        if confirming < 1 {
            return PromotionVerdict::Denied("no confirming members");
        }

        // Confidence must be above base threshold
        if comp.confidence < 0.4 {
            return PromotionVerdict::Denied("confidence too low for Inferred");
        }

        PromotionVerdict::Approved
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PromotionVerdict {
    Approved,
    Denied(String),  // reason for denial
}
```

Promotion threshold reference table:

```text
Transition              Min Confidence   Min Confirming   Min Sources   Min Age
────────────────────────────────────────────────────────────────────────────────
New → Candidate         (automatic on first sense induction)
Candidate → Stable      0.55             2 members        —             3 batches
Inferred → Grounded     0.70             —                2 sources     —
Hypothesis → Inferred   0.40             1 member         —             —
Quarantine → Candidate  (enrichment provides evidence, see re_govern_composition)
```

### Lifecycle Transitions

```text
New → Candidate     (after first sense induction)
Candidate → Stable  (after grounding confirms — see Promotion Criteria)
Stable → Deprecated (if contradicted + not recovered)
Any → Quarantine    (if hypothesis detected or conflict unresolved)
Quarantine → Candidate (if evidence supports)
```

### Epistemic Transitions

```text
Observed → Inferred → Grounded   (as evidence accumulates — see Promotion Criteria)
Hypothesis → Inferred             (if confirming evidence — see Promotion Criteria)
Any → Contradicted                (if opposing evidence stronger)
Grounded → Contradicted           (if new contradiction)
Contradicted → Grounded           (if contradiction resolved — see Contradiction Resolution)
```

## Re-Governance After Enrichment

When EnrichComposition adds a new member to an existing composition, the composition's
lifecycle and epistemic states may need to transition. GovernBeliefs must support
re-evaluation of a single composition, not just initial assignment. During initial
ingest, GovernBeliefs assigns states once and checks for promotions across the graph.
After enrichment, the pipeline calls `re_govern_composition()` to re-evaluate the
specific composition that was repaired — without re-running the full governance pass.

```rust
impl GovernBeliefs {
    /// Re-govern a composition after it has been enriched.
    /// Called by the pipeline after EnrichComposition runs.
    ///
    /// Key principle: enrichment IMPROVES a composition, so transitions
    /// should be promotional (not demotional), unless contradictions emerge.
    pub fn re_govern_composition(
        &self,
        composition: &Composition,
        graph: &Graph,
    ) -> GovernanceUpdate {
        let mut update = GovernanceUpdate::new(composition.id.clone());

        // 1. Check if lifecycle should advance
        match composition.lifecycle {
            LifecycleState::New => {
                // Enrichment of a New composition promotes to Candidate
                // (it now has more evidence)
                update.set_lifecycle(LifecycleState::Candidate);
            },
            LifecycleState::Candidate => {
                // If enrichment fills all expected roles, promote to Stable
                if self.is_sufficiently_complete(composition) {
                    update.set_lifecycle(LifecycleState::Stable);
                }
                // Otherwise stays Candidate
            },
            LifecycleState::Quarantine => {
                // Enrichment of a quarantined composition is evidence FOR it
                // Transition to Candidate (give it a chance)
                update.set_lifecycle(LifecycleState::Candidate);
            },
            _ => {} // Stable, Deprecated: no lifecycle change from enrichment
        }

        // 2. Check if epistemic state should advance
        match composition.epistemic {
            EpistemicState::Observed => {
                // Enrichment adds inferred members → still Observed overall
                // (the original extraction was observed, the enrichment is inferred)
                // No change — the blend is correctly captured by member-level epistemic
            },
            EpistemicState::Inferred => {
                // Enrichment with graph context strengthens inference
                // BUT: must still pass full promotion criteria (independent sources,
                // no recent contradiction, seed alignment) — NOT just confidence.
                // This prevents enrichment from single-source compositions
                // from auto-grounding without independent verification.
                match self.can_promote_to_grounded(composition, graph) {
                    PromotionVerdict::Approved => {
                        update.set_epistemic(EpistemicState::Grounded);
                    },
                    PromotionVerdict::Denied(reason) => {
                        // Enrichment wasn't enough to ground — stays Inferred
                        // This is correct: enrichment from 1 source doesn't make it 2
                    },
                }
            },
            EpistemicState::Hypothesis => {
                // Enrichment provides evidence → promote to Inferred
                update.set_epistemic(EpistemicState::Inferred);
            },
            _ => {} // Grounded, Contradicted: special handling
        }

        // 3. Check for new contradictions (enrichment might introduce conflict)
        if let Some(contradiction) = self.detect_contradiction(composition, graph) {
            update.mark_contradicted(contradiction);
        }

        // 4. Update confidence
        // Enrichment confidence is blended: existing * 0.6 + new_member * 0.4
        // (done by EnrichComposition, GovernBeliefs just validates)

        update
    }

    /// A composition is sufficiently complete if it has all expected roles
    /// for its composition type filled.
    fn is_sufficiently_complete(&self, composition: &Composition) -> bool {
        match composition.composition_type {
            CompositionType::Event => {
                let has_predicate = composition.members.iter()
                    .any(|m| m.role == SemanticRole::Predicate);
                let has_agent = composition.members.iter()
                    .any(|m| m.role == SemanticRole::Arg0Agent);
                let has_patient = composition.members.iter()
                    .any(|m| m.role == SemanticRole::Arg1Patient);
                has_predicate && has_agent && has_patient
            },
            CompositionType::HiddenMeaning => {
                // Hidden meanings need at least PatternType + one other role
                let has_pattern = composition.members.iter()
                    .any(|m| m.role == SemanticRole::PatternType);
                has_pattern && composition.members.len() >= 2
            },
            _ => true, // other types: considered complete
        }
    }
}
```

```rust
/// Result of re-governing a composition after enrichment.
/// Contains the transitions to apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceUpdate {
    pub composition_id: CompositionId,
    pub new_lifecycle: Option<LifecycleState>,
    pub new_epistemic: Option<EpistemicState>,
    pub contradiction: Option<Contradiction>,
    pub confidence_adjustment: Option<f32>,
}

impl GovernanceUpdate {
    pub fn new(id: CompositionId) -> Self {
        Self {
            composition_id: id,
            new_lifecycle: None,
            new_epistemic: None,
            contradiction: None,
            confidence_adjustment: None,
        }
    }

    pub fn set_lifecycle(&mut self, state: LifecycleState) {
        self.new_lifecycle = Some(state);
    }

    pub fn set_epistemic(&mut self, state: EpistemicState) {
        self.new_epistemic = Some(state);
    }

    pub fn mark_contradicted(&mut self, contradiction: Contradiction) {
        self.contradiction = Some(contradiction);
        self.new_epistemic = Some(EpistemicState::Contradicted);
    }
}
```

Enrichment-triggered transition rules:

```text
Current State              After Enrichment        Transition
─────────────────────────────────────────────────────────────
(New, Observed)            role filled             → (Candidate, Observed)
(Candidate, Inferred)      all roles filled         → (Stable, Inferred)
(Quarantine, Inferred)     evidence supports        → (Candidate, Inferred)
(Quarantine, Hypothesis)   evidence supports        → (Candidate, Inferred)
(Candidate, Inferred)      can_promote_to_grounded()   → (Candidate, Grounded)
(Any, Contradicted)        enrichment contradicted   → stays Contradicted
(Stable, Grounded)         enrichment confirms       → stays (Stable, Grounded)
```

The key principle is that enrichment is promotional by default. A composition that
gains new evidence should be given a chance to advance. But enrichment that introduces
contradictions must be flagged. The system errs on the side of giving compositions
a chance to improve.

---

## Contradiction Resolution

MD-4's epistemic transitions include `Contradicted → Grounded`, but the previous
specification provided no transform or criteria for this. MD-2's PolarityConflictRule
detects contradictions but produces no resolution signal. This section defines the
mechanism.

```rust
/// Contradiction resolution status.
/// Tracks whether a contradiction has been resolved and how.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionResolution {
    pub contradiction_id: String,
    pub opposing_composition_id: CompositionId,
    pub resolution_type: ResolutionType,
    pub resolved: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionType {
    /// One side was superseded by newer, stronger evidence
    Superseded,
    /// The contradiction was based on incomplete context; new context resolves it
    ContextResolved,
    /// Both sides are true in different contexts (scoped validity)
    ScopedValidity,
    /// One side was a misinterpretation (e.g., passive vs active voice)
    Misinterpretation,
    /// Resolution not yet possible
    Unresolved,
}

impl GovernBeliefs {
    /// Check if a contradiction can be resolved.
    /// A contradiction is resolved when:
    /// 1. New evidence supports one side and contradicts the other
    /// 2. Context disambiguation shows the compositions are valid in different scopes
    /// 3. The contradicting composition was based on a misinterpretation (e.g., voice confusion)
    fn check_contradiction_resolution(
        &self,
        comp: &Composition,
        graph: &Graph,
    ) -> Option<ContradictionResolution> {
        // Find the opposing composition
        let opposing_id = comp.contradiction_opposing_id()?;
        let opposing = graph.get_composition(&opposing_id)?;

        // Strategy 1: Voice confusion — active vs passive same event
        if self.is_voice_confusion(comp, opposing) {
            return Some(ContradictionResolution {
                contradiction_id: format!("cr_{}", comp.id),
                opposing_composition_id: opposing_id,
                resolution_type: ResolutionType::Misinterpretation,
                resolved: true,
            });
        }

        // Strategy 2: Scoped validity — both true in different contexts
        if self.has_scoped_validity(comp, opposing, graph) {
            return Some(ContradictionResolution {
                contradiction_id: format!("cr_{}", comp.id),
                opposing_composition_id: opposing_id,
                resolution_type: ResolutionType::ScopedValidity,
                resolved: true,
            });
        }

        // Strategy 3: Superseded — one side has stronger evidence
        if self.is_superseded(comp, opposing, graph) {
            return Some(ContradictionResolution {
                contradiction_id: format!("cr_{}", comp.id),
                opposing_composition_id: opposing_id,
                resolution_type: ResolutionType::Superseded,
                resolved: true,
            });
        }

        None
    }

    /// Active "Raymond membuat aplikasi" + Passive "Aplikasi dibuat oleh Raymond"
    /// These are NOT contradictions — they're the same event in different voice.
    ///
    /// FIX: Same predicate + agent + patient alone is NOT sufficient to detect
    /// voice confusion — it would also match literal duplicates (same event said
    /// twice). We must also verify that the two compositions come from DIFFERENT
    /// extraction sources (one active, one passive provenance), OR that one
    /// composition's provenance indicates passive extraction.
    ///
    /// Without Voice field on Composition, we check provenance:
    /// - If both came from the same provenance origin_id, it's a DUPLICATE (not voice confusion)
    /// - If they came from different origin_ids with the same roles, it's likely voice confusion
    fn is_voice_confusion(&self, comp_a: &Composition, comp_b: &Composition) -> bool {
        let same_predicate = comp_a.member_with_role(&SemanticRole::Predicate)
            == comp_b.member_with_role(&SemanticRole::Predicate);
        let same_agent = comp_a.member_with_role(&SemanticRole::Arg0Agent)
            == comp_b.member_with_role(&SemanticRole::Arg0Agent);
        let same_patient = comp_a.member_with_role(&SemanticRole::Arg1Patient)
            == comp_b.member_with_role(&SemanticRole::Arg1Patient);

        if !(same_predicate && same_agent && same_patient) {
            return false;
        }

        // Same structural identity. Now distinguish:
        // DUPLICATE: same provenance (same extraction of the same sentence)
        // VOICE CONFUSION: different provenance (two different sentences, same event)
        let different_origin = comp_a.provenance.origin_id != comp_b.provenance.origin_id;
        let different_source = comp_a.provenance.origin != comp_b.provenance.origin;

        // Voice confusion requires: same roles BUT from different source sentences.
        // If they came from the same extraction (same origin_id), it's a duplicate.
        different_origin || different_source
    }

    /// Both compositions are valid but in different contexts (scoped).
    fn has_scoped_validity(&self, comp_a: &Composition, comp_b: &Composition, graph: &Graph) -> bool {
        // If the two compositions never co-occur in the same context,
        // they may both be true in different scopes.
        // This is detectable via graph co-occurrence analysis.
        let cooccurrence = graph.cooccurrence_count(
            comp_a.member_with_role(&SemanticRole::Predicate),
            comp_b.member_with_role(&SemanticRole::Predicate),
        );
        cooccurrence <= 1 // rarely co-occur → likely scoped
    }

    /// One side has accumulated much more evidence than the other.
    fn is_superseded(&self, comp_a: &Composition, comp_b: &Composition, graph: &Graph) -> bool {
        let conf_diff = (comp_a.confidence - comp_b.confidence).abs();
        let a_is_stronger = comp_a.confidence > comp_b.confidence
            && comp_a.members.len() > comp_b.members.len();
        let b_is_stronger = comp_b.confidence > comp_a.confidence
            && comp_b.members.len() > comp_a.members.len();
        conf_diff >= 0.3 && (a_is_stronger || b_is_stronger)
    }

    /// Resolve a contradiction: transition the superseded side to Deprecated,
    /// and the winning side to Grounded.
    fn resolve_contradiction(
        &self,
        resolution: &ContradictionResolution,
        graph: &mut Graph,
    ) -> GovernanceUpdate {
        match resolution.resolution_type {
            ResolutionType::Misinterpretation | ResolutionType::Superseded => {
                // The stronger composition transitions to Grounded
                // The weaker transitions to Deprecated
                // (determine which is which by confidence)
            },
            ResolutionType::ScopedValidity => {
                // Both remain, but scoped with context tags
                // Neither transitions — they're both valid in different scopes
            },
            ResolutionType::ContextResolved => {
                // The contradicted composition transitions from Contradicted → Inferred
                // It gets another chance, with new context
            },
            ResolutionType::Unresolved => {
                // No change
            },
        }
        // ... return appropriate GovernanceUpdate
    }
}
```

```text
NOTE: Contradiction resolution runs during Reflective mode's GovernBeliefs re-evaluation.
It is NOT automatic during initial ingest — contradictions must be reviewed
when the system has enough context to make a judgment. Premature resolution
is worse than leaving a contradiction in Quarantine.
```

---

## SeedAnchor Transform

Replaces source trust weight system with seed-driven confidence evaluation.

```rust
/// SeedAnchor Transform
///
/// Input:  GovernedDelta
/// Output: AnchoredDelta (same + seed_scores on each composition)
pub struct SeedAnchor {
    seed_engine: SeedActivationEngine,
}

impl Transform for SeedAnchor {
    type Input = GovernedDelta;
    type Output = AnchoredDelta;

    fn id(&self) -> &'static str { "SeedAnchor" }

    fn transform(&self, governed: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        let mut anchored = AnchoredDelta::from(governed.clone());

        for composition in &governed.compositions {
            // Compute seed alignment scores
            let seed_scores = self.compute_seed_scores(composition, &ctx.graph);
            anchored.set_seed_scores(composition.id.clone(), seed_scores.clone());

            // Adjust confidence based on seed anchoring
            // Uses SeedAdjustment which handles the no-alignment-data case correctly
            let adjustment = self.seed_anchored_confidence(&seed_scores);
            let adjusted = self.adjust_confidence(composition.confidence, &adjustment);
            anchored.adjust_confidence(composition.id.clone(), adjusted);
        }

        anchored
    }
}
```

### Seed-Driven Confidence

```rust
impl SeedAnchor {
    /// Seed-driven confidence evaluation.
    /// CRITICAL FIX: Only adjusts confidence when there is ACTUAL seed alignment data.
    /// Default/neutral scores (0.5) produce NO adjustment.
    fn seed_anchored_confidence(&self, scores: &HashMap<SeedPrimitive, f32>) -> SeedAdjustment {
        let trust   = scores.get(&SeedPrimitive::Trust).copied().unwrap_or(0.5);
        let risk    = scores.get(&SeedPrimitive::Risk).copied().unwrap_or(0.5);
        let value   = scores.get(&SeedPrimitive::Value).copied().unwrap_or(0.5);
        let goal    = scores.get(&SeedPrimitive::Goal).copied().unwrap_or(0.5);
        let identity = scores.get(&SeedPrimitive::Identity).copied().unwrap_or(0.5);

        // Compute raw score
        let raw = trust * 0.30
            + (1.0 - risk) * 0.25
            + value * 0.20
            + goal * 0.15
            + identity * 0.10;

        // KEY FIX: Measure how much actual alignment data we have.
        // If all scores are default (0.5), we have NO information and should NOT adjust.
        let has_alignment_data = scores.values().any(|&s| (s - 0.5).abs() > 0.05);

        if !has_alignment_data {
            // No alignment data → return neutral (no adjustment)
            return SeedAdjustment {
                seed_confidence: 0.5,
                weight: 0.0,  // zero weight means: don't blend, keep original
                alignment_strength: 0.0,
            };
        }

        // We have alignment data → compute adjustment strength
        let alignment_strength = scores.values()
            .map(|&s| (s - 0.5).abs())
            .sum::<f32>()
            / scores.len() as f32;

        // Weight scales with alignment strength:
        // Strong alignment (0.3+) → weight 0.4 (significant adjustment)
        // Weak alignment (0.05-0.3) → weight 0.1-0.4 (proportional)
        // No alignment (<0.05) → weight 0.0 (no adjustment)
        let weight = (alignment_strength * 1.33).clamp(0.0, 0.4);

        SeedAdjustment {
            seed_confidence: raw,
            weight,
            alignment_strength,
        }
    }
}

/// Result of seed anchoring evaluation.
/// Contains both the computed seed confidence AND how strongly to apply it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedAdjustment {
    pub seed_confidence: f32,
    pub weight: f32,             // 0.0 = no adjustment, 0.4 = strong adjustment
    pub alignment_strength: f32, // how much actual data drove this
}

impl SeedAnchor {
    fn adjust_confidence(&self, original: f32, adjustment: &SeedAdjustment) -> f32 {
        // When weight is 0.0 (no alignment data), result = original (no free boost)
        // When weight is 0.4 (strong alignment), result = original * 0.6 + seed * 0.4
        (original * (1.0 - adjustment.weight) + adjustment.seed_confidence * adjustment.weight)
            .clamp(0.0, 1.0)
    }
}

// NOTE: After enrichment, SeedAnchor should be re-run on the enriched composition.
// The new member may change seed alignment scores. For example:
// - Adding an Arg0Agent that is a known entity → trust seed alignment increases
// - Adding a Cause that references a risk-related node → risk seed alignment increases
// The pipeline ensures this by running GovernBeliefs → SeedAnchor after EnrichComposition.
```

Verification that the free-boost bug is fixed:

```text
Verification: with default scores (all 0.5):
  raw = 0.5, has_alignment_data = false → weight = 0.0
  adjust_confidence(0.1, {seed: 0.5, weight: 0.0}) = 0.1 * 1.0 + 0.5 * 0.0 = 0.1
  → NO free boost. Confidence stays at 0.1.

With strong trust alignment (trust=0.9, rest default):
  raw = 0.9*0.30 + 0.5*0.25 + 0.5*0.20 + 0.5*0.15 + 0.5*0.10 = 0.62
  has_alignment_data = true, alignment_strength ≈ 0.08
  weight = 0.08 * 1.33 ≈ 0.107
  adjust_confidence(0.1, {seed: 0.62, weight: 0.107}) = 0.1 * 0.893 + 0.62 * 0.107 ≈ 0.156
  → Small boost from genuine alignment data. Not a free ride.
```

### Why Seed-Driven Is Better

```text
Source trust weights:
  - Static per source type
  - No connection to graph content
  - External configuration
  - Separate from reasoning

Seed-driven anchoring:
  - Dynamic: changes as graph matures
  - Grounded in graph structure
  - Uses same primitives as reasoning
  - Natural feedback: confidence → seed activation → reasoning → confidence
  - A composition aligned with trust seeds IS more trustworthy
  - A composition triggering risk seeds IS more risky
```

---

## Integration with v11.0 Existing Governance

| v11.0 Type | v12.0 Equivalent | Migration |
|-----------|-----------------|-----------|
| `NodeStatus` | `LifecycleState` | Direct mapping (identical variants) |
| `BeliefState` | `(LifecycleState, EpistemicState)` | Split across two axes |
| `GroundingVerdict` | Derived function of (lifecycle, epistemic, confidence) | No separate type |
| `GroundingEvidence` | Kept as-is on Sense | Composition references Sense |
| `PolicyMeta` | Subsumed by Composition.seed_scores + provenance | Gradual migration |
| `CandidateStatus` | `(LifecycleState, EpistemicState)` | Eliminated as separate type |

---

## GroundingVerdict as Derived Function

No separate `GroundingVerdict` enum. It's computed from the two axes:

```rust
fn grounding_verdict(lifecycle: &LifecycleState, epistemic: &EpistemicState, confidence: f32) -> GroundingVerdict {
    match (lifecycle, epistemic) {
        (Stable, Grounded) if confidence > 0.8 => WellGrounded,
        (_, Contradicted) => NeedsRevision,
        (Quarantine, _) => NeedsRevision,
        _ => NeedsReview,
    }
}
```

---

## Module Structure

```text
layer1/crates/rsvs-core/src/
  epistemic/
    mod.rs              // GovernBeliefs + SeedAnchor Transforms
    types.rs            // EpistemicConflictType, Contradiction, GovernedDelta, AnchoredDelta
    governance.rs       // state assignment + transitions + contradiction detection
    seed_anchor.rs      // seed-driven confidence evaluation
    promotion.rs        // lifecycle + epistemic promotion rules
    resolution.rs       // contradiction resolution logic
    tests.rs            // unit tests
```

7 files.

---

## Required Tests

### Test 1 — Event → (New, Observed)

Event from FrameCompiler → lifecycle=New, epistemic=Observed

### Test 2 — HiddenMeaning → (Quarantine, Inferred)

HiddenMeaning from ReasonFrame → lifecycle=Quarantine, epistemic=Inferred

### Test 3 — Contradiction Detected → Epistemic Changes to Contradicted

Two compositions with polarity conflict → one becomes Contradicted

### Test 4 — Promotion: Candidate → Stable

Composition receives confirming evidence → lifecycle transitions to Stable

### Test 5 — Hypothesis Cannot Auto-Promote to Grounded

Hypothesis composition → must pass through Inferred first, cannot skip to Grounded

### Test 6 — Seed Anchor Adjusts Confidence

Composition aligned with trust seeds → confidence increases
Composition triggering risk seeds → confidence decreases

### Test 7 — Provenance Is Traceable

Query: "Why does the system believe X?"
→ ProvenanceChain: origin=HiddenMeaningRule, origin_id=hm_3, parent=evt_1

### Test 8 — Re-governance: New → Candidate After Enrichment

Composition(Event, New, Observed) enriched with missing Arg0Agent
Expected: lifecycle transitions to Candidate

### Test 9 — Re-governance: Quarantine → Candidate After Enrichment

Composition(HiddenMeaning, Quarantine, Inferred) enriched with supporting evidence
Expected: lifecycle transitions to Candidate

### Test 10 — Re-governance: Sufficiently Complete Promotes to Stable

Composition(Event, Candidate, Inferred) with Predicate + Agent but missing Patient
Then enriched with Patient → all roles filled
Expected: lifecycle transitions to Stable

### Test 11 — Enrichment Does Not Auto-Ground

Composition enriched with graph-recalled member (EpistemicState::Inferred)
Expected: epistemic stays Inferred, does NOT jump to Grounded
(Only repeated independent evidence can transition to Grounded)

### Test 12 — Enrichment That Introduces Contradiction

Composition enriched with a member that contradicts an existing member
Expected: contradiction detected, epistemic transitions to Contradicted

### Test 13 — Promotion Criteria: Candidate Denied for Low Confidence

Composition with confidence 0.4, age 5 batches, 3 confirming members
Expected: can_promote_to_stable returns Denied("confidence below 0.55 threshold")

### Test 14 — Promotion Criteria: Candidate Denied for Insufficient Confirming Members

Composition with confidence 0.6, age 5 batches, 1 confirming member
Expected: can_promote_to_stable returns Denied("only 1 confirming members (need ≥ 2)")

### Test 15 — Promotion Criteria: Candidate Denied for Active Contradiction

Composition with confidence 0.7, age 5 batches, 3 confirming members, but Contradicted
Expected: can_promote_to_stable returns Denied("active contradiction unresolved")

### Test 16 — Promotion Criteria: Candidate Denied for Immaturity

Composition with confidence 0.7, age 1 batch, 3 confirming members
Expected: can_promote_to_stable returns Denied("too young (1 batches, need ≥ 3)")

### Test 17 — Promotion Criteria: Inferred → Grounded Requires Independent Sources

Composition with confidence 0.8, epistemic=Inferred, but only 1 source
Expected: can_promote_to_grounded returns Denied("only 1 independent source(s), need ≥ 2")

### Test 18 — Contradiction Resolution: Voice Confusion

Active "Raymond membuat aplikasi" + Passive "Aplikasi dibuat oleh Raymond"
Expected: resolution_type = Misinterpretation, both compositions reconciled

### Test 19 — Contradiction Resolution: Scoped Validity

Two compositions with same predicate that never co-occur
Expected: resolution_type = ScopedValidity, both remain valid in different scopes

### Test 20 — Contradiction Resolution: Superseded

One composition with confidence 0.9, 5 members vs another with confidence 0.4, 2 members
Expected: resolution_type = Superseded, weaker transitions to Deprecated

### Test 21 — SeedAnchor: No Free Boost with Default Scores

Composition with confidence 0.1, all seed scores at 0.5
Expected: adjusted confidence remains 0.1 (weight = 0.0, no blending)

### Test 22 — SeedAnchor: Genuine Alignment Produces Proportional Boost

Composition with confidence 0.1, trust=0.9, rest default
Expected: small boost (≈0.156) proportional to alignment strength, not a free ride

---

## Acceptance Criteria

1. `GovernBeliefs` Transform assigns (LifecycleState, EpistemicState) correctly
2. `SeedAnchor` Transform computes seed_scores and adjusts confidence
3. Two axes replace four overlapping enums
4. No hypothesis can silently become fact (quarantine + promotion rules)
5. Provenance is traceable via ProvenanceChain (using EdgeSource)
6. Contradiction detection works for polarity, purpose, role reversal
7. `EpistemicConflictType` is separate from v11.0 `ConflictType`
8. GroundingVerdict is derived, not a separate enum
9. All existing tests remain green
10. GovernBeliefs supports re-governance of single compositions after enrichment
11. Enrichment-triggered transitions are promotional by default
12. Sufficiently complete compositions can promote from Candidate → Stable
13. Enriched members never auto-transition to Grounded (requires repeated evidence)
14. Contradictions from enrichment are detected and flagged
15. `check_promotions()` has concrete criteria with explicit thresholds for every transition
16. `PromotionVerdict` provides denial reasons (not just bool) for debugging and transparency
17. Contradiction resolution is defined with strategies: Misinterpretation, ScopedValidity, Superseded, ContextResolved
18. Contradiction resolution runs only in Reflective mode, not during initial ingest
19. `SeedAnchor` does not inflate low-confidence compositions when no alignment data exists
20. `SeedAdjustment.weight` is 0.0 when all scores are default (0.5), preventing free confidence boost

---

## Final Statement

MD-4 implements belief governance through two Transforms: GovernBeliefs (assigns and
transitions the two status axes) and SeedAnchor (evaluates confidence through seed
alignment). The two-axis model eliminates four overlapping enums. Seed-driven confidence
replaces static source trust weights with dynamic, graph-grounded evaluation. With
re-governance support, GovernBeliefs now handles the feedback loop from EnrichComposition,
ensuring that repaired compositions receive appropriate promotional transitions while
still guarding against contradictions introduced by enrichment.

Promotion criteria are explicit and testable: every transition has concrete thresholds
for confidence, confirming members, independent sources, and maturity. Contradiction
resolution provides a structured pathway from Contradicted back to Grounded via four
strategies (Misinterpretation, ScopedValidity, Superseded, ContextResolved), running
only in Reflective mode to avoid premature resolution. The SeedAnchor fix ensures that
default/neutral seed scores produce zero adjustment weight, eliminating the free
confidence boost that previously inflated unsupported compositions from 0.1 to 0.26.
