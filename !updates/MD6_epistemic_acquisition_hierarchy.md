# MD-6 — Epistemic Acquisition Hierarchy & Inquiry Layer (Adjusted for Implementation)

> **Adjustment Note (v11.0 alignment):** This document has been revised for implementation
> readiness. Key changes from original spec:
> - Standalone callable module, NOT dependent on MD-5 Executive Cognition
> - Can be called from pipeline directly OR from executive when available
> - Python bridge for self-study (z-ai-web-dev-sdk in backend layer2)
> - Phase 1: Passive Recall + gap detection + Ask User only
> - Phase 2: Self Study via web search bridge
> - Simplified KnowledgeGapType from 15 variants to 6 for Phase 1
> - Reduced module structure from 9 files to 5
> - No new validation gate in Phase 1 (AcquisitionDisciplineGate deferred)
> - All 1,081 existing tests must remain green

---

## Context

Assume completed:

- MD-1: Semantic Frame Compiler (Phase 1)
- MD-2: Pre-Ingest Meaning Reasoner (Phase 1)
- MD-3: AAM Architecture Refactor (hybrid additive)
- MD-4: Epistemic Truth & Belief Governance (Phase 1)
- MD-5: Executive Cognition (Phase 1, if available)

This document defines the epistemic acquisition hierarchy — how AAM acquires missing knowledge.

**Key independence**: MD-6 can function WITHOUT MD-5. Executive cognition is optional. The acquisition module can be called directly from the pipeline when knowledge gaps are detected.

---

## Core Doctrine

AAM must not guess when knowledge is insufficient.

But AAM must also not ask the user too early.

Correct acquisition order:

```text
1. Passive Recall  — check existing graph memory
2. Self Study      — research external sources (Phase 2)
3. Ask User        — inquire only when necessary
```

This is a hierarchy, not a flat mode selection.

---

## Why This Layer Exists

After MD-1 through MD-5, AAM has:

```text
semantic event understanding
hidden meaning inference
architecture with frame enrichment
truth governance
minimal executive control
```

But early-stage AAM will often have:

```text
sparse graph
immature senses
weak patterns
missing domain knowledge
unknown user context
ambiguous references
```

If AAM concludes too early → pollutes the graph.
If AAM asks too early → creates unnecessary friction.
If AAM searches too freely → may import low-quality information.

Therefore AAM needs an acquisition hierarchy.

---

## High-Level Flow

```text
Input
→ Semantic Frame Compiler (MD-1)
→ Pre-Ingest Meaning Reasoner (MD-2)
→ Knowledge Gap Detection           ← NEW
→ Acquisition Strategy Selection    ← NEW
    1. Passive Recall (graph lookup)
    2. Self Study (web search)      ← Phase 2
    3. Ask User (inquiry)
→ Evidence Assimilation
→ Epistemic Governance (MD-4)
→ RSVS Ingest / Reasoning
```

The acquisition module can be called from:

```text
1. Pipeline directly (when gap detected during ingest)
2. Executive cognition (MD-5, when available)
3. Query handler (when reasoning hits a gap)
```

---

## Acquisition Hierarchy — Phased

### Phase 1 — Passive Recall + Ask User

```text
Knowledge gap detected
→ Can existing graph resolve it?
    yes → Passive Recall / continue
    no  → Is the gap user-context dependent?
        yes → Ask User
        no  → Mark as "SelfStudyNeeded" (deferred to Phase 2)
```

Phase 1 does NOT include web search. It establishes the gap detection and inquiry infrastructure.

### Phase 2 — Self Study

```text
Add Self Study between Passive Recall and Ask User:
→ Can existing graph resolve it?
    yes → Passive Recall
    no  → Is missing knowledge public/researchable?
        yes → Self Study → Did it resolve?
            yes → assimilate, continue
            no  → Ask User
        no  → Ask User
```

Self Study uses Python bridge (z-ai-web-dev-sdk) in the backend layer.

---

## Type Design — Phase 1

### KnowledgeGapType (Simplified)

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KnowledgeGapType {
    NoGap,                    // no gap detected
    SparseGraphGap,           // graph too sparse to reason
    AmbiguousReferenceGap,    // pronoun/reference unclear
    PrivateContextGap,        // user-specific context needed
    MissingFieldGap,          // EventFrame missing required field
    LowGroundingGap,          // sense grounding too weak
    UnresolvableGap,          // gap exists but no acquisition path can fix it
}
```

7 variants, not 15. Phase 2 adds: PublicFactualGap, FreshnessGap, TechnicalDomainGap, etc.

### KnowledgeGap (Phase 1)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGap {
    pub gap_id: String,
    pub gap_type: KnowledgeGapType,
    pub description: String,
    pub source: GapSource,             // what detected this gap
    pub confidence: f32,               // how certain the gap exists
    pub severity: f32,                 // how much it blocks reasoning
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GapSource {
    FrameMissingField,       // MD-1 frame has missing ARG0, etc.
    CandidateLowConfidence,  // MD-2 candidate has low confidence
    GroundingWeak,           // MD-4 grounding below threshold
    GraphSparse,             // not enough nodes in relevant area
    AmbiguousReference,      // pronoun or unclear reference
}
```

### AcquisitionMode

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AcquisitionMode {
    PassiveRecall,   // use existing graph
    SelfStudy,       // research external sources (Phase 2)
    AskUser,         // inquire user
    Deferred,        // gap noted but no action taken yet
}
```

### AcquisitionDecision

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcquisitionDecision {
    pub mode: AcquisitionMode,
    pub gap_type: KnowledgeGapType,
    pub reason: String,
    pub confidence_before: f32,
    pub expected_gain: f32,           // expected information gain
}
```

### InquiryQuestion

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
    IdentityClarification,    // "Who does 'dia' refer to?"
    ReferenceClarification,  // "What does 'it' mean here?"
    GoalClarification,       // "What should be improved?"
    ConstraintClarification, // "What are the limitations?"
    MissingFieldClarification, // "Who performed this action?"
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExpectedAnswerType {
    Entity,
    Definition,
    Constraint,
    Confirmation,
    FreeText,
}
```

### UserAnswerEvent

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAnswerEvent {
    pub answer_id: String,
    pub question_id: String,
    pub raw_answer: String,
    pub resolved_gaps: Vec<String>,  // gap_ids resolved
    pub confidence: f32,
}
```

---

## Gap Detection Logic

### From MD-1 (EventFrame)

```rust
fn detect_frame_gaps(frame: &EventFrame) -> Vec<KnowledgeGap> {
    let mut gaps = Vec::new();

    if frame.arg0_agent.is_none() {
        gaps.push(KnowledgeGap {
            gap_type: MissingFieldGap,
            description: "Agent (ARG0) is missing from event frame".into(),
            source: FrameMissingField,
            confidence: 0.9,
            severity: 0.7,
            ..
        });
    }

    if frame.arg1_patient.is_none() && frame.predicate_has_expected_patient() {
        gaps.push(/* ... */);
    }

    if frame.confidence < 0.5 {
        gaps.push(/* Low confidence gap */);
    }

    gaps
}
```

### From MD-2 (HiddenMeaningCandidate)

```rust
fn detect_candidate_gaps(candidate: &HiddenMeaningCandidate, graph: &Graph) -> Vec<KnowledgeGap> {
    let mut gaps = Vec::new();

    if candidate.confidence < 0.4 {
        gaps.push(KnowledgeGap {
            gap_type: LowGroundingGap,
            description: format!("Hidden meaning candidate '{}' has low confidence", candidate.candidate_id),
            source: CandidateLowConfidence,
            confidence: 0.8,
            severity: 0.5,
            ..
        });
    }

    // Check if role-filler nodes exist in graph
    for node_ref in &candidate.nodes {
        if node_ref.node_id.is_none() && !graph.has_node(&node_ref.label) {
            gaps.push(/* node not in graph, reference gap */);
        }
    }

    gaps
}
```

### From Graph State

```rust
fn detect_graph_gaps(graph: &Graph, focus_nodes: &[NodeId]) -> Vec<KnowledgeGap> {
    let density = graph.local_density(focus_nodes);
    if density < SPARSE_THRESHOLD {
        return vec![KnowledgeGap {
            gap_type: SparseGraphGap,
            description: "Graph is too sparse in the relevant area".into(),
            source: GraphSparse,
            confidence: 0.9,
            severity: 0.8,
            ..
        }];
    }
    vec![]
}
```

---

## Strategy Selection

```rust
fn select_strategy(gap: &KnowledgeGap, graph: &Graph) -> AcquisitionDecision {
    match gap.gap_type {
        KnowledgeGapType::NoGap => AcquisitionDecision {
            mode: PassiveRecall,
            gap_type: gap.gap_type.clone(),
            reason: "No gap detected".into(),
            confidence_before: 1.0,
            expected_gain: 0.0,
        },

        KnowledgeGapType::SparseGraphGap => {
            // Check if graph has any relevant nodes at all
            if graph_has_relevant_context(graph, gap) {
                AcquisitionDecision { mode: PassiveRecall, .. }
            } else {
                // Phase 2: SelfStudy
                // Phase 1: Deferred
                AcquisitionDecision { mode: Deferred, .. }
            }
        },

        KnowledgeGapType::AmbiguousReferenceGap |
        KnowledgeGapType::PrivateContextGap |
        KnowledgeGapType::MissingFieldGap => {
            // User context is needed
            AcquisitionDecision { mode: AskUser, .. }
        },

        KnowledgeGapType::LowGroundingGap => {
            // Try passive recall first
            if graph_has_grounding_evidence(graph, gap) {
                AcquisitionDecision { mode: PassiveRecall, .. }
            } else {
                AcquisitionDecision { mode: AskUser, .. }
            }
        },

        KnowledgeGapType::UnresolvableGap => {
            AcquisitionDecision { mode: Deferred, .. }
        },
    }
}
```

---

## Integration with MD-4 (Epistemic Governance)

All acquired knowledge passes through epistemic governance:

### Passive Recall

```text
Uses grounded beliefs (BeliefState::Grounded) preferentially
Does NOT create new claims
```

### Ask User

```text
User answer → UserAnswerEvent
→ Semantic Frame Compiler (if sentence)
→ Epistemic Governance (register as claim)
  belief_state = Candidate (not auto-Grounded)
  provenance.source_type = HumanAssertion
  If user is source of personal context → high source trust (0.85)
  If user makes factual public claim → still Candidate, may need verification
```

### Self Study (Phase 2)

```text
External claim → register as Candidate
  provenance.source_type = ExternalEvidence
  belief_state = Candidate (NEVER auto-Grounded)
  Must pass through epistemic promotion rules
```

---

## Integration with MD-5 (Executive Cognition) — OPTIONAL

When executive cognition is available:

```text
Executive detects knowledge gap → calls acquisition module
Executive receives AcquisitionDecision → executes strategy
Executive respects acquisition hierarchy (recall → study → ask)
```

When executive cognition is NOT available:

```text
Pipeline detects gap during ingest → calls acquisition module directly
Pipeline executes acquisition decision
```

The acquisition module is **self-contained** and does not require executive cognition.

---

## Python Bridge for Self Study (Phase 2)

Self Study requires web search, which is only available via Python backend (z-ai-web-dev-sdk), not in Rust layer1.

Architecture:

```text
Rust layer1 (acquisition module)
  → FFI call to Python bridge
    → Python layer2 (web search via z-ai-web-dev-sdk)
      → Search results
    → Extract claims from results
  → Return SelfStudyResult to Rust
```

Python bridge stub (Phase 1):

```python
# layer2/acquisition/self_study.py

class SelfStudyProvider:
    """Phase 1: stub. Phase 2: web search integration."""

    def research(self, request: SelfStudyRequest) -> SelfStudyResult:
        # Phase 1: return empty result
        return SelfStudyResult(
            request_id=request.request_id,
            sources_used=[],
            extracted_claims=[],
            confidence=0.0,
            resolved_gap=False,
        )
```

Phase 2 implementation:

```python
import ZAI from 'z-ai-web-dev-sdk'

class SelfStudyProvider:
    def research(self, request: SelfStudyRequest) -> SelfStudyResult:
        zai = ZAI.create()
        results = zai.functions.invoke("web_search", {
            query: request.query,
            num: request.max_sources
        })
        # Extract claims from search results
        claims = self.extract_claims(results, request.source_policy)
        return SelfStudyResult(
            sources_used=results,
            extracted_claims=claims,
            confidence=self.compute_confidence(claims),
            resolved_gap=len(claims) > 0,
        )
```

---

## Inquiry Memory

AAM must remember previous questions and answers to avoid repetition:

```rust
pub struct InquiryMemory {
    pub asked_questions: HashMap<String, InquiryQuestion>,  // question_id → question
    pub received_answers: HashMap<String, UserAnswerEvent>,  // question_id → answer
    pub resolved_gaps: HashSet<String>,                      // gap_ids that have been resolved
}

impl InquiryMemory {
    pub fn should_ask(&self, gap: &KnowledgeGap) -> bool {
        // Don't ask about already-resolved gaps
        if self.resolved_gaps.contains(&gap.gap_id) {
            return false;
        }
        // Don't ask the same question twice
        for (_, q) in &self.asked_questions {
            if q.gap_id == gap.gap_id {
                return false;  // already asked about this gap
            }
        }
        true
    }
}
```

---

## Ask User Discipline

Questions must be minimal and high-value.

Bad:

```text
Can you clarify?
Tell me more.
```

Good:

```text
Who does "dia" refer to in this context?
What specific action made you call it betrayal?
Who performed this action? (ARG0 missing)
```

Rules:

```text
1. Ask the SMALLEST question needed to resolve the BIGGEST uncertainty
2. One question per gap, not compound questions
3. Never ask about something the graph already knows
4. Never repeat a question
5. User answers are acquisition events, not chat — process through MD-1 + MD-4
```

---

## Module Structure (Phase 1: Minimal)

### Rust (layer1)

```text
layer1/crates/rsvs-core/src/
  acquisition/
    mod.rs              // public API: detect_gaps(), select_strategy(), ask_user()
    types.rs            // KnowledgeGap, KnowledgeGapType, AcquisitionMode, AcquisitionDecision,
                        // InquiryQuestion, UserAnswerEvent, InquiryMemory
    gap_detector.rs     // gap detection from frame, candidate, graph state
    strategy.rs         // acquisition strategy selection
    inquiry.rs          // question generation + answer processing
    tests.rs            // unit tests
```

6 files for Phase 1.

### Python (layer2) — Phase 2

```text
layer2/
  acquisition/
    __init__.py
    self_study.py       // web search integration via z-ai-web-dev-sdk
    source_policy.py    // source trust and quality rules
    bridge.py           // FFI bridge from Rust
```

---

## Integration with Current Codebase

### Pipeline Integration

Add acquisition stage to pipeline (additive, feature-flagged):

```python
# pipeline.py — existing stages (unchanged)
_run_context_layer
_run_situation_layer

# NEW stage (additive)
if self.acquisition_enabled:
    _run_acquisition_layer

# existing stages (unchanged)
_run_reasoning_layer
_appraise
```

### Event Emissions

Add runtime events (additive):

```text
knowledge_gap_detected     // when gap detector finds a gap
acquisition_mode_selected  // when strategy is chosen
inquiry_question_created   // when user question is generated
user_answer_assimilated    // when user answer is processed
```

These use existing event emission infrastructure.

### Types (additive to types.rs)

```text
KnowledgeGap, KnowledgeGapType, AcquisitionMode, AcquisitionDecision,
InquiryQuestion, UserAnswerEvent, InquiryMemory, GapSource
```

All new types. No existing types modified.

---

## Required Tests

### Test 1 — Missing ARG0 Detected

Input: EventFrame with `arg0_agent = None`

Expected: `KnowledgeGapType::MissingFieldGap` detected

### Test 2 — Low Confidence Candidate Detected

Input: HiddenMeaningCandidate with confidence 0.2

Expected: `KnowledgeGapType::LowGroundingGap` detected

### Test 3 — Private Context Gap → Ask User

Input: ambiguous pronoun reference

Expected: `AcquisitionMode::AskUser` selected

### Test 4 — Sparse Graph → Deferred (Phase 1)

Input: graph with no relevant nodes

Expected: `AcquisitionMode::Deferred` (SelfStudy not available in Phase 1)

### Test 5 — No Gap → Passive Recall

Input: fully specified frame, high confidence

Expected: `AcquisitionMode::PassiveRecall`

### Test 6 — Inquiry Memory Prevents Repeat Questions

Input: same gap detected twice

Expected: `should_ask()` returns false for second occurrence

### Test 7 — User Answer Creates Claim with Correct Belief State

Input: UserAnswerEvent

Expected: `KnowledgeClaim` created with `belief_state = Candidate`, `provenance.source_type = HumanAssertion`

### Test 8 — User Answer Not Auto-Grounded

Input: UserAnswerEvent with factual claim

Expected: `belief_state = Candidate` (NOT Grounded)

---

## Phase 2 — Self Study Implementation

When Phase 2 is ready:

1. Implement `SelfStudyProvider` in Python with z-ai-web-dev-sdk
2. Add FFI bridge from Rust acquisition module to Python
3. Add `PublicFactualGap`, `FreshnessGap`, `TechnicalDomainGap` to KnowledgeGapType
4. Add `SourcePolicy` for self-study discipline
5. Add `AcquisitionDisciplineGate` to validation gates
6. Self-study results enter as `Candidate` claims (never auto-Grounded)

---

## Acceptance Criteria

Phase 1 is acceptable if:

1. Gap detection works for: missing frame fields, low-confidence candidates, sparse graph, ambiguous references
2. Strategy selection follows hierarchy: Passive Recall → Ask User (SelfStudy deferred)
3. User questions are minimal and targeted
4. Inquiry memory prevents repeat questions
5. User answers create KnowledgeClaims with Candidate status
6. No user answer is auto-promoted to Grounded
7. Acquisition module works WITHOUT MD-5 executive cognition
8. Acquisition module CAN be called from executive when available
9. All 1,081 existing tests remain green
10. Module structure is 6 Rust files + Python stub

---

## Final Statement

MD-6 transforms AAM from a system that waits for enough data into one that knows how to acquire missing context. Phase 1 establishes the gap detection and inquiry infrastructure without depending on executive cognition. Phase 2 adds autonomous self-study via web search. The acquisition hierarchy protects graph quality, minimizes user burden, and makes AAM capable of growing its own knowledge over time.

The hierarchy is simple:

```text
Remember first.
Study second.
Ask last.
```
