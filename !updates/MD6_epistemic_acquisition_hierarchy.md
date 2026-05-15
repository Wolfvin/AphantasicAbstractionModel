# MD-6 — Epistemic Acquisition Hierarchy (Elegant Architecture)

> **Prerequisite**: MD-3 defines SemanticAtom, AtomType, Composition, Transform, LifecycleState,
> EpistemicState, SemanticEdge, EdgeSource. MD-4 defines GovernBeliefs + SeedAnchor.
> MD-5 defines ExecutiveOrchestrator.
> This document defines acquisition as **Transforms** that detect gaps and resolve them.

---

## Mission

Implement knowledge gap detection and resolution as Transforms:

1. **DetectGaps** — inspects SemanticAtoms and graph state for missing knowledge
2. **SelectAcquisition** — chooses resolution strategy (Passive Recall → Self Study → Ask User)
3. **AcquireUserAnswer** — processes user answers as new SemanticAtom(Acquisition)

Acquisition is standalone. It can be called from the pipeline directly, or from
the ExecutiveOrchestrator when enabled. It does NOT depend on MD-5.

---

## Core Doctrine

```text
Remember first.
Study second.
Ask last.
```

Minimize user burden. Maximize autonomous acquisition. Ask only when necessary.
Preserve epistemic integrity.

---

## DetectGaps Transform

```rust
/// DetectGaps Transform
///
/// Input:  GraphSnapshot (current graph state + recent atoms)
/// Output: Vec<KnowledgeGap>
///
/// Inspects SemanticAtoms and graph for missing knowledge.
pub struct DetectGaps {
    config: GapDetectionConfig,
}

impl Transform for DetectGaps {
    type Input = GraphSnapshot;
    type Output = Vec<KnowledgeGap>;

    fn id(&self) -> &'static str { "DetectGaps" }

    fn transform(&self, snapshot: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        let mut gaps = Vec::new();

        // 1. Check recent atoms for missing fields
        for atom in &snapshot.recent_atoms {
            gaps.extend(self.detect_atom_gaps(atom));
        }

        // 2. Check graph for sparsity in relevant areas
        gaps.extend(self.detect_graph_gaps(&snapshot.graph));

        // 3. Check for low-grounding compositions
        gaps.extend(self.detect_grounding_gaps(&snapshot.graph));

        gaps
    }
}
```

### Gap Types (Simplified for Phase 1)

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KnowledgeGapType {
    NoGap,
    SparseGraphGap,           // graph too sparse to reason
    AmbiguousReferenceGap,    // pronoun/reference unclear
    PrivateContextGap,        // user-specific context needed
    MissingFieldGap,          // SemanticAtom missing important role
    LowGroundingGap,          // composition grounding too weak
    UnresolvableGap,          // gap exists but no acquisition path can fix it
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGap {
    pub gap_id: String,
    pub gap_type: KnowledgeGapType,
    pub description: String,
    pub source: GapSource,
    pub confidence: f32,
    pub severity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GapSource {
    AtomMissingRole,        // SemanticAtom has missing role (e.g., no Arg0Agent)
    CandidateLowConfidence, // HiddenMeaning atom has confidence < threshold
    GroundingWeak,          // composition has low grounding score
    GraphSparse,            // not enough nodes in relevant area
    AmbiguousReference,     // pronoun or unclear reference in atom
}
```

### Detection Logic

```rust
impl DetectGaps {
    fn detect_atom_gaps(&self, atom: &SemanticAtom) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();

        match atom.atom_type {
            AtomType::Event => {
                // Events should have at least agent + patient
                if !atom.roles.contains_key(&SemanticRole::Arg0Agent) {
                    gaps.push(KnowledgeGap {
                        gap_id: format!("gap_{}", atom.id),
                        gap_type: KnowledgeGapType::MissingFieldGap,
                        description: format!("Event '{}' missing agent (ARG0)", atom.label),
                        source: GapSource::AtomMissingRole,
                        confidence: 0.9,
                        severity: 0.7,
                    });
                }

                if !atom.roles.contains_key(&SemanticRole::Arg1Patient) {
                    gaps.push(KnowledgeGap {
                        gap_id: format!("gap_{}", atom.id),
                        gap_type: KnowledgeGapType::MissingFieldGap,
                        description: format!("Event '{}' missing patient (ARG1)", atom.label),
                        source: GapSource::AtomMissingRole,
                        confidence: 0.85,
                        severity: 0.6,
                    });
                }

                // Low confidence frame
                if atom.confidence < 0.5 {
                    gaps.push(KnowledgeGap {
                        gap_type: KnowledgeGapType::LowGroundingGap,
                        description: format!("Event '{}' has low confidence ({:.2})", atom.label, atom.confidence),
                        source: GapSource::CandidateLowConfidence,
                        confidence: 0.8,
                        severity: 0.5,
                        ..Default::default()
                    });
                }
            },

            AtomType::HiddenMeaning => {
                // Hidden meanings with low confidence
                if atom.confidence < 0.4 {
                    gaps.push(KnowledgeGap {
                        gap_type: KnowledgeGapType::LowGroundingGap,
                        description: format!("HiddenMeaning '{}' has low confidence ({:.2})", atom.label, atom.confidence),
                        source: GapSource::CandidateLowConfidence,
                        confidence: 0.8,
                        severity: 0.5,
                        ..Default::default()
                    });
                }

                // Check if role-filler nodes exist in graph
                for (role, label) in &atom.roles {
                    if !graph.has_node(label) && *role != SemanticRole::SourceEvent {
                        gaps.push(KnowledgeGap {
                            gap_type: KnowledgeGapType::AmbiguousReferenceGap,
                            description: format!("Role '{}' references unknown node '{}'", 
                                format!("{:?}", role), label),
                            source: GapSource::AmbiguousReference,
                            confidence: 0.7,
                            severity: 0.4,
                            ..Default::default()
                        });
                    }
                }
            },

            _ => {} // Token and other atoms: no gap detection needed
        }

        gaps
    }

    fn detect_graph_gaps(&self, graph: &Graph) -> Vec<KnowledgeGap> {
        if graph.node_count() < SPARSE_THRESHOLD {
            return vec![KnowledgeGap {
                gap_type: KnowledgeGapType::SparseGraphGap,
                description: "Graph is too sparse for reliable reasoning".into(),
                source: GapSource::GraphSparse,
                confidence: 0.9,
                severity: 0.8,
                ..Default::default()
            }];
        }
        vec![]
    }

    fn detect_grounding_gaps(&self, graph: &Graph) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();
        for comp in graph.compositions() {
            if comp.epistemic == EpistemicState::Inferred && comp.confidence < 0.3 {
                gaps.push(KnowledgeGap {
                    gap_type: KnowledgeGapType::LowGroundingGap,
                    description: format!("Composition '{}' has low grounding", comp.id),
                    source: GapSource::GroundingWeak,
                    confidence: 0.7,
                    severity: 0.5,
                    ..Default::default()
                });
            }
        }
        gaps
    }
}
```

---

## SelectAcquisition Transform

```rust
/// SelectAcquisition Transform
///
/// Input:  Vec<KnowledgeGap>
/// Output: Vec<AcquisitionDecision>
///
/// Chooses resolution strategy for each gap.
pub struct SelectAcquisition {
    inquiry_memory: InquiryMemory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AcquisitionMode {
    PassiveRecall,   // use existing graph
    SelfStudy,       // research external sources (Phase 2)
    AskUser,         // inquire user
    Deferred,        // gap noted but no action (Phase 2: will be SelfStudy)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcquisitionDecision {
    pub gap_id: String,
    pub mode: AcquisitionMode,
    pub reason: String,
    pub confidence_before: f32,
    pub expected_gain: f32,
}

impl Transform for SelectAcquisition {
    type Input = Vec<KnowledgeGap>;
    type Output = Vec<AcquisitionDecision>;

    fn id(&self) -> &'static str { "SelectAcquisition" }

    fn transform(&self, gaps: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        gaps.iter()
            .filter(|gap| gap.gap_type != KnowledgeGapType::NoGap)
            .filter(|gap| self.inquiry_memory.should_ask(gap))
            .map(|gap| self.select_strategy(gap, &ctx.graph))
            .collect()
    }
}

impl SelectAcquisition {
    fn select_strategy(&self, gap: &KnowledgeGap, graph: &Graph) -> AcquisitionDecision {
        match gap.gap_type {
            KnowledgeGapType::NoGap => AcquisitionDecision {
                gap_id: gap.gap_id.clone(),
                mode: AcquisitionMode::PassiveRecall,
                reason: "No gap detected".into(),
                confidence_before: 1.0,
                expected_gain: 0.0,
            },

            KnowledgeGapType::SparseGraphGap => {
                if graph_has_relevant_context(graph, gap) {
                    AcquisitionDecision { mode: AcquisitionMode::PassiveRecall, .. }
                } else {
                    // Phase 1: Deferred (no SelfStudy yet)
                    // Phase 2: SelfStudy
                    AcquisitionDecision { mode: AcquisitionMode::Deferred, .. }
                }
            },

            KnowledgeGapType::AmbiguousReferenceGap |
            KnowledgeGapType::PrivateContextGap |
            KnowledgeGapType::MissingFieldGap => {
                AcquisitionDecision { mode: AcquisitionMode::AskUser, .. }
            },

            KnowledgeGapType::LowGroundingGap => {
                if graph_has_grounding_evidence(graph, gap) {
                    AcquisitionDecision { mode: AcquisitionMode::PassiveRecall, .. }
                } else {
                    AcquisitionDecision { mode: AcquisitionMode::AskUser, .. }
                }
            },

            KnowledgeGapType::UnresolvableGap => {
                AcquisitionDecision { mode: AcquisitionMode::Deferred, .. }
            },
        }
    }
}
```

---

## InquiryQuestion — Asking the User

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InquiryQuestion {
    pub question_id: String,
    pub gap_id: String,
    pub question_type: InquiryQuestionType,
    pub question_text: String,
    pub expected_answer_shape: ExpectedAnswerType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InquiryQuestionType {
    IdentityClarification,     // "Who does 'dia' refer to?"
    ReferenceClarification,   // "What does 'it' mean here?"
    GoalClarification,        // "What should be improved?"
    ConstraintClarification,  // "What are the limitations?"
    MissingFieldClarification, // "Who performed this action?"
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExpectedAnswerType {
    Entity, Definition, Constraint, Confirmation, FreeText,
}

impl SelectAcquisition {
    /// Generate an inquiry question for a gap that requires user input.
    pub fn generate_question(&self, gap: &KnowledgeGap) -> Option<InquiryQuestion> {
        match gap.gap_type {
            KnowledgeGapType::MissingFieldGap => {
                // Determine which field is missing
                if gap.description.contains("missing agent") {
                    Some(InquiryQuestion {
                        question_id: format!("q_{}", gap.gap_id),
                        gap_id: gap.gap_id.clone(),
                        question_type: InquiryQuestionType::MissingFieldClarification,
                        question_text: "Who performed this action?".into(),
                        expected_answer_shape: ExpectedAnswerType::Entity,
                    })
                } else if gap.description.contains("missing patient") {
                    Some(InquiryQuestion {
                        question_id: format!("q_{}", gap.gap_id),
                        gap_id: gap.gap_id.clone(),
                        question_type: InquiryQuestionType::MissingFieldClarification,
                        question_text: "What was affected by this action?".into(),
                        expected_answer_shape: ExpectedAnswerType::Entity,
                    })
                } else {
                    None
                }
            },

            KnowledgeGapType::AmbiguousReferenceGap => {
                Some(InquiryQuestion {
                    question_id: format!("q_{}", gap.gap_id),
                    gap_id: gap.gap_id.clone(),
                    question_type: InquiryQuestionType::ReferenceClarification,
                    question_text: format!("What does '{}' refer to in this context?", gap.description),
                    expected_answer_shape: ExpectedAnswerType::Entity,
                })
            },

            KnowledgeGapType::PrivateContextGap => {
                Some(InquiryQuestion {
                    question_id: format!("q_{}", gap.gap_id),
                    gap_id: gap.gap_id.clone(),
                    question_type: InquiryQuestionType::IdentityClarification,
                    question_text: "Can you clarify the context?".into(),
                    expected_answer_shape: ExpectedAnswerType::FreeText,
                })
            },

            _ => None,
        }
    }
}
```

---

## User Answer Processing

User answers are processed as new `SemanticAtom(Acquisition, ...)`:

```rust
/// Process a user answer into a SemanticAtom for ingestion.
pub fn process_user_answer(answer: &str, question: &InquiryQuestion) -> SemanticAtom {
    SemanticAtom {
        id: format!("acq_{}", question.question_id),
        label: answer.to_string(),
        atom_type: AtomType::Acquisition,
        roles: {
            let mut roles = HashMap::new();
            // Link to the gap this answer resolves
            roles.insert(SemanticRole::SourceAtom, question.gap_id.clone());
            roles
        },
        polarity: None,
        voice: None,
        variant: Some(AtomVariant::AcquisitionVariant(AcquisitionSource::UserAnswer)),
        confidence: 0.85,  // human assertion is high-confidence source
        source: EdgeSource::AcquisitionUserAnswer,
    }
}
```

When ingested, this becomes:

```rust
Composition {
    composition_type: CompositionType::Acquisition,
    lifecycle: LifecycleState::Candidate,      // not auto-promoted
    epistemic: EpistemicState::Observed,       // directly observed from user
    provenance: ProvenanceChain {
        origin: EdgeSource::AcquisitionUserAnswer,
        origin_id: question_id,
        parent_composition_id: None,
        timestamp: now_iso8601(),
    },
    // Seed scores computed by SeedAnchor
    ..
}
```

**Critical**: User answers enter as `(Candidate, Observed)` — NOT `(Stable, Grounded)`.
Human assertions about personal context get high seed trust alignment, but still
require the standard promotion path. Public factual claims from users still need
independent verification.

---

## Inquiry Memory — Prevent Repetition

```rust
pub struct InquiryMemory {
    asked_gaps: HashSet<String>,          // gap_ids that have been asked about
    resolved_gaps: HashSet<String>,       // gap_ids that have been resolved
    questions: HashMap<String, InquiryQuestion>,  // question_id → question
    answers: HashMap<String, UserAnswerRecord>,    // question_id → answer
}

impl InquiryMemory {
    pub fn should_ask(&self, gap: &KnowledgeGap) -> bool {
        // Don't ask about already-resolved gaps
        if self.resolved_gaps.contains(&gap.gap_id) {
            return false;
        }
        // Don't ask about the same gap twice
        if self.asked_gaps.contains(&gap.gap_id) {
            return false;
        }
        true
    }

    pub fn record_question(&mut self, question: &InquiryQuestion) {
        self.asked_gaps.insert(question.gap_id.clone());
        self.questions.insert(question.question_id.clone(), question.clone());
    }

    pub fn record_answer(&mut self, question_id: &str, answer: &str, gap_id: &str) {
        self.answers.insert(question_id.to_string(), UserAnswerRecord {
            answer: answer.to_string(),
            resolved_gaps: vec![gap_id.to_string()],
        });
        self.resolved_gaps.insert(gap_id.to_string());
    }
}
```

---

## Integration with Executive (Optional)

When ExecutiveOrchestrator is enabled:

```rust
// Inside ExecutiveOrchestrator, after Analytical or Reflective ingest
fn check_for_gaps(&self, engine: &mut PipelineEngine) -> Option<Vec<InquiryQuestion>> {
    let snapshot = engine.snapshot();
    let gaps = engine.run::<DetectGaps>(&snapshot);
    let decisions = engine.run::<SelectAcquisition>(&gaps);

    let questions: Vec<InquiryQuestion> = decisions.iter()
        .filter(|d| d.mode == AcquisitionMode::AskUser)
        .filter_map(|d| {
            let gap = gaps.iter().find(|g| g.gap_id == d.gap_id)?;
            engine.get::<SelectAcquisition>().generate_question(gap)
        })
        .collect();

    if questions.is_empty() { None } else { Some(questions) }
}
```

When Executive is NOT enabled, pipeline can call DetectGaps + SelectAcquisition directly.

---

## Phase 2 — Self Study (via Python Bridge)

```python
# layer2/acquisition/self_study.py

class SelfStudyProvider:
    """Phase 1: stub. Phase 2: web search integration."""

    def research(self, request: SelfStudyRequest) -> SelfStudyResult:
        # Phase 2: use z-ai-web-dev-sdk for web search
        # import ZAI
        # zai = ZAI.create()
        # results = zai.functions.invoke("web_search", { query: ..., num: ... })
        # Extract claims, assess source quality
        # Return SelfStudyResult with extracted claims as Candidates
        pass
```

Self-study results enter as `SemanticAtom(Acquisition, AcquisitionSource::SelfStudy)` →
`Composition(Acquisition, Quarantine, Inferred)` — never auto-Grounded.

---

## Module Structure

### Rust (layer1)

```text
layer1/crates/rsvs-core/src/
  acquisition/
    mod.rs              // DetectGaps + SelectAcquisition Transforms + public API
    types.rs            // KnowledgeGap, KnowledgeGapType, AcquisitionMode, AcquisitionDecision,
                        // InquiryQuestion, InquiryQuestionType, UserAnswerRecord, InquiryMemory
    gap_detect.rs       // gap detection logic
    strategy.rs         // acquisition strategy selection + question generation
    tests.rs            // unit tests
```

5 files.

### Python (layer2) — Phase 2

```text
layer2/
  acquisition/
    __init__.py
    self_study.py       // web search via z-ai-web-dev-sdk
    source_policy.py    // source trust and quality rules
    bridge.py           // FFI bridge from Rust
```

---

## Required Tests

### Test 1 — Missing Agent Role Detected

Input: `SemanticAtom(Event, "membuat", {})` (no Arg0Agent)

Expected: `KnowledgeGapType::MissingFieldGap` detected

### Test 2 — Low Confidence HiddenMeaning Detected

Input: `SemanticAtom(HiddenMeaning, ..., confidence: 0.2)`

Expected: `KnowledgeGapType::LowGroundingGap` detected

### Test 3 — Private Context Gap → Ask User

Input: ambiguous reference in atom

Expected: `AcquisitionMode::AskUser` selected

### Test 4 — Sparse Graph → Deferred (Phase 1)

Input: graph with few nodes

Expected: `AcquisitionMode::Deferred` (SelfStudy not yet available)

### Test 5 — No Gap → Passive Recall

Input: fully specified event atom, high confidence, dense graph

Expected: `AcquisitionMode::PassiveRecall`

### Test 6 — Inquiry Memory Prevents Repeat

Input: same gap detected twice

Expected: second gap filtered by InquiryMemory

### Test 7 — User Answer → SemanticAtom(Acquisition)

Input: "Raymond" as answer to "Who performed this action?"

Expected: `SemanticAtom { atom_type: Acquisition, source: AcquisitionUserAnswer }`

### Test 8 — User Answer Composition Has (Candidate, Observed)

Verify: `lifecycle=Candidate, epistemic=Observed` — NOT auto-Grounded

---

## Acceptance Criteria

1. `DetectGaps` Transform identifies gaps from atoms and graph state
2. `SelectAcquisition` Transform follows hierarchy: PassiveRecall → AskUser (SelfStudy Phase 2)
3. Gap types cover: missing fields, low confidence, sparse graph, ambiguous references
4. Questions are minimal and targeted (one question per gap)
5. InquiryMemory prevents repeat questions
6. User answers become `SemanticAtom(Acquisition)` → `Composition(Candidate, Observed)`
7. No user answer is auto-promoted to Grounded
8. Acquisition works WITHOUT ExecutiveOrchestrator
9. Acquisition CAN be called from Executive when available
10. All existing tests remain green

---

## Final Statement

MD-6 implements acquisition as Transforms that detect knowledge gaps and resolve them
through the hierarchy: Remember first, Study second, Ask last. Gap detection inspects
SemanticAtoms and graph state. Resolution produces either PassiveRecall decisions,
inquiry questions for users, or deferred SelfStudy requests. All acquired knowledge
enters as SemanticAtom(Acquisition) and passes through the same GovernBeliefs + SeedAnchor
pipeline — no special treatment, no bypassing epistemic governance.
