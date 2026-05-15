# MD-4 — Epistemic Truth & Belief Governance (Elegant Architecture)

> **Prerequisite**: MD-3 defines LifecycleState, EpistemicState (two orthogonal axes),
> Composition, SemanticEdge, ProvenanceChain, EdgeSource, Seed Anchoring.
> This document defines the `GovernBeliefs` and `SeedAnchor` Transforms.

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
        // Find compositions with same predicate but conflicting roles
        let same_predicate: Vec<&Composition> = graph.compositions()
            .filter(|c| c.composition_type == CompositionType::Event)
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

        None
    }
}
```

### Lifecycle Transitions

```text
New → Candidate     (after first sense induction)
Candidate → Stable  (after grounding confirms)
Stable → Deprecated (if contradicted + not recovered)
Any → Quarantine    (if hypothesis detected or conflict unresolved)
Quarantine → Candidate (if evidence supports)
```

### Epistemic Transitions

```text
Observed → Inferred → Grounded   (as evidence accumulates)
Hypothesis → Grounded             (if independently confirmed)
Any → Contradicted                (if opposing evidence stronger)
Grounded → Contradicted           (if new contradiction)
Contradicted → Grounded           (if contradiction resolved, Phase 2)
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
            let seed_confidence = self.seed_anchored_confidence(&seed_scores);
            let adjusted = (composition.confidence * 0.6 + seed_confidence * 0.4).clamp(0.0, 1.0);
            anchored.adjust_confidence(composition.id.clone(), adjusted);
        }

        anchored
    }
}
```

### Seed-Driven Confidence

```rust
impl SeedAnchor {
    fn seed_anchored_confidence(&self, scores: &HashMap<SeedPrimitive, f32>) -> f32 {
        let trust   = scores.get(&SeedPrimitive::Trust).copied().unwrap_or(0.5);
        let risk    = scores.get(&SeedPrimitive::Risk).copied().unwrap_or(0.5);
        let value   = scores.get(&SeedPrimitive::Value).copied().unwrap_or(0.5);
        let goal    = scores.get(&SeedPrimitive::Goal).copied().unwrap_or(0.5);
        let identity = scores.get(&SeedPrimitive::Identity).copied().unwrap_or(0.5);

        // Trust and risk dominate epistemic evaluation
        trust * 0.30
        + (1.0 - risk) * 0.25
        + value * 0.20
        + goal * 0.15
        + identity * 0.10
    }
}
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
    tests.rs            // unit tests
```

6 files.

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

---

## Final Statement

MD-4 implements belief governance through two Transforms: GovernBeliefs (assigns and
transitions the two status axes) and SeedAnchor (evaluates confidence through seed
alignment). The two-axis model eliminates four overlapping enums. Seed-driven confidence
replaces static source trust weights with dynamic, graph-grounded evaluation.
