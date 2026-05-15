# MD-6 — Epistemic Acquisition Hierarchy & Inquiry Layer

## Context

Assume the following documents have already been implemented:

- MD-1: Semantic Frame Compiler
- MD-2: Pre-Ingest Meaning Reasoner / Hidden Meaning Compiler
- MD-3: AAM Architecture Refactor
- MD-4: Epistemic Truth & Belief Governance Layer
- MD-5: Executive Cognition / Meta-Control Layer

This document defines the next architectural layer:

```text
Epistemic Acquisition Hierarchy
```

This layer governs how AAM acquires missing knowledge before reasoning or committing new beliefs.

It must be integrated with the current AAM codebase and the newly implemented MD1–MD5 architecture.

---

# Core Doctrine

AAM must not guess when knowledge is insufficient.

But AAM must also not ask the user too early.

Correct acquisition order:

```text
1. Passive Recall
2. Self Study
3. Ask User
```

This is not a flat mode selection.

This is a hierarchy.

AAM should first attempt to solve the knowledge gap from memory.

If memory is insufficient, AAM should attempt autonomous research.

If autonomous research still cannot solve the gap, then AAM should ask the user.

---

# Central Statement

```text
Minimize user burden.
Maximize autonomous acquisition.
Ask only when necessary.
Preserve epistemic integrity.
```

---

# Why This Layer Exists

After MD1–MD5, AAM has:

- semantic event understanding
- hidden meaning inference
- architectural refactor
- truth governance
- executive control

But early-stage AAM will often have:

- sparse graph
- immature senses
- weak patterns
- missing domain knowledge
- unknown user context
- ambiguous references
- incomplete situational model

If AAM concludes too early, it pollutes the graph.

If AAM asks too early, it creates unnecessary friction.

If AAM searches too freely, it may import low-quality or irrelevant information.

Therefore AAM needs an acquisition hierarchy.

---

# High-Level Flow

```text
Input
→ Semantic Frame Compiler
→ Pre-Ingest Meaning Reasoner
→ Executive Cognition
→ Knowledge Gap Detection
→ Acquisition Strategy
    1. Passive Recall
    2. Self Study
    3. Ask User
→ Evidence Assimilation
→ Epistemic Governance
→ RSVS Ingest / Reasoning
```

---

# Acquisition Hierarchy

## Tier 1 — Passive Recall

Use existing AAM internal knowledge.

Sources:

```text
RSVS graph
grounded beliefs
sense memory
pattern memory
situation state
working memory
epistemic claims
previous user answers
```

Use when:

```text
graph confidence sufficient
ambiguity low
known domain
existing grounded belief available
no freshness requirement
```

Output:

```text
Continue reasoning without external acquisition.
```

---

## Tier 2 — Self Study

If passive recall is insufficient, AAM independently researches.

Sources:

```text
web search
official documentation
papers
repositories
uploaded files
trusted corpora
internal knowledge bases
```

Use when:

```text
public factual knowledge gap
technical unknown
fresh information required
domain unfamiliarity
objective external information needed
```

Output must be governed by MD-4.

Self-study evidence must enter as:

```text
ExternalEvidence
ObservedExternalClaim
InferredCandidate
```

Never automatically as Grounded truth.

---

## Tier 3 — Ask User

Ask the user only when passive recall and self-study cannot solve the gap.

Use when:

```text
private user context required
subjective meaning required
ambiguous pronoun/reference
personal preference required
goal unclear
relationship context unknown
user-owned project detail missing
```

Output:

```text
UserAnswerEvent
ResolvedAmbiguity
NewContextClaim
```

User answers should be remembered and governed by MD-4.

---

# Decision Tree

```text
Knowledge gap detected?
  no:
    Passive Recall / continue

  yes:
    Can existing graph solve it?
      yes:
        Passive Recall / continue

      no:
        Is the missing knowledge public or externally researchable?
          yes:
            Self Study
            Did self-study resolve the gap?
              yes:
                assimilate evidence, continue
              no:
                Ask User

          no:
            Ask User
```

---

# Knowledge Gap Types

Add this taxonomy.

```rust
pub enum KnowledgeGapType {
    NoGap,
    SparseGraphGap,
    PublicFactualGap,
    FreshnessGap,
    TechnicalDomainGap,
    RepositoryContextGap,
    FileContextGap,
    PrivateContextGap,
    SubjectiveMeaningGap,
    AmbiguousReferenceGap,
    GoalAmbiguityGap,
    RelationshipContextGap,
    ContradictionGap,
    MissingConstraintGap,
    LowGroundingGap,
}
```

---

# Acquisition Mode

```rust
pub enum AcquisitionMode {
    PassiveRecall,
    SelfStudy,
    AskUser,
    Hybrid,
}
```

---

# Acquisition Decision

```rust
pub struct AcquisitionDecision {
    pub mode: AcquisitionMode,
    pub gap_type: KnowledgeGapType,
    pub reason: String,
    pub confidence_before: f32,
    pub expected_information_gain: f32,
    pub user_burden_score: f32,
    pub research_cost_score: f32,
    pub epistemic_risk: f32,
}
```

---

# Core Types

## KnowledgeGap

```rust
pub struct KnowledgeGap {
    pub gap_id: String,
    pub gap_type: KnowledgeGapType,
    pub source_event_id: Option<String>,
    pub source_candidate_id: Option<String>,
    pub description: String,
    pub missing_fields: Vec<String>,
    pub ambiguity_targets: Vec<String>,
    pub confidence: f32,
    pub severity: f32,
}
```

---

## SelfStudyRequest

```rust
pub struct SelfStudyRequest {
    pub request_id: String,
    pub gap_id: String,
    pub query: String,
    pub source_policy: SourcePolicy,
    pub max_sources: usize,
    pub freshness_required: bool,
    pub expected_answer_type: ExpectedAnswerType,
}
```

---

## SelfStudyResult

```rust
pub struct SelfStudyResult {
    pub request_id: String,
    pub sources_used: Vec<SourceRef>,
    pub extracted_claims: Vec<ExternalClaim>,
    pub confidence: f32,
    pub resolved_gap: bool,
    pub provenance: ProvenanceChain,
}
```

---

## InquiryQuestion

```rust
pub struct InquiryQuestion {
    pub question_id: String,
    pub gap_id: String,
    pub question_type: InquiryQuestionType,
    pub question_text: String,
    pub expected_answer_shape: ExpectedAnswerType,
    pub information_gain_score: f32,
    pub user_burden_score: f32,
}
```

---

## UserAnswerEvent

```rust
pub struct UserAnswerEvent {
    pub answer_id: String,
    pub question_id: String,
    pub raw_answer: String,
    pub parsed_claims: Vec<KnowledgeClaim>,
    pub resolved_gaps: Vec<String>,
    pub confidence: f32,
}
```

---

# Inquiry Question Types

```rust
pub enum InquiryQuestionType {
    IdentityClarification,
    ReferenceClarification,
    GoalClarification,
    ConstraintClarification,
    PreferenceClarification,
    DefinitionClarification,
    RelationshipClarification,
    TemporalClarification,
    DomainClarification,
    ConflictResolutionQuestion,
    MissingEvidenceQuestion,
}
```

---

# Source Policy

Self-study must not ingest random low-quality information.

```rust
pub struct SourcePolicy {
    pub prefer_official_sources: bool,
    pub allow_web_search: bool,
    pub allow_repositories: bool,
    pub allow_papers: bool,
    pub allow_uploaded_files: bool,
    pub allow_low_quality_sources: bool,
    pub require_citations: bool,
}
```

Default:

```text
prefer official sources
require provenance
avoid low-quality sources
never treat self-study as direct truth
```

---

# Expected Answer Type

```rust
pub enum ExpectedAnswerType {
    Entity,
    Definition,
    Constraint,
    Preference,
    Time,
    Location,
    Cause,
    Purpose,
    Evidence,
    Confirmation,
    Rejection,
    FreeText,
}
```

---

# Integration With MD-1

MD-1 produces EventFrame.

MD-6 should inspect EventFrame for missing fields.

Examples:

```text
missing ARG0
missing ARG1
ambiguous predicate
low frame confidence
unknown reference
```

If missing field is public/domain knowledge:

```text
SelfStudy
```

If missing field is user-specific:

```text
AskUser
```

---

# Integration With MD-2

MD-2 produces HiddenMeaningCandidate.

MD-6 should inspect candidate quality.

Ask or self-study if:

```text
candidate confidence low
candidate depends on unknown node
hidden meaning type unfamiliar
candidate uses weak graph grounding
candidate conflicts with existing belief
```

Example:

```text
problem_solution_pattern detected
but "kantor" identity unknown
```

If “kantor” means user workplace, ask.

If “kantor” is a public organization, self-study may help.

---

# Integration With MD-3

MD-3 refactored AAM into structured event reasoning.

MD-6 must become part of the new global architecture:

```text
Layer 0: semantic ingest
Layer 1: RSVS memory
Layer 2: prediction / situation
Layer 3: reasoning
Layer 4: epistemic governance
Layer 5: executive cognition
Layer 6: epistemic acquisition
Layer 7: narrative rendering
```

Alternative:

MD-6 can be implemented as a subsystem under Executive Cognition.

Recommended:

```text
executive/acquisition/
```

because acquisition strategy is an executive decision.

---

# Integration With MD-4

All acquired knowledge must pass through epistemic governance.

## Passive Recall

Use grounded beliefs preferentially.

## Self Study

Must create:

```text
ExternalEvidence
ExternalClaim
ProvenanceChain
```

State:

```text
ObservedExternalClaim
```

or:

```text
Candidate
```

Never:

```text
Grounded
```

unless confirmed by independent evidence and governance rules.

## Ask User

User answers create:

```text
UserAnswerEvent
HumanAssertion
```

Still not always Grounded automatically.

If user is source of personal context, assign high source trust.

If user answer is factual public claim, still may need verification.

---

# Integration With MD-5

MD-5 Executive Cognition chooses reasoning strategy.

MD-6 adds acquisition strategy.

Executive should call MD-6 when:

```text
confidence low
graph sparse
ambiguity high
contradiction unresolved
missing constraints
domain unknown
freshness needed
```

Executive should receive:

```text
AcquisitionDecision
```

Then execute:

```text
PassiveRecall
SelfStudy
AskUser
```

---

# Integration With Current Codebase

Based on current AAM structure, suggested location:

```text
layer2/acquisition/
```

or:

```text
executive/acquisition/
```

Recommended final structure:

```text
executive/
  acquisition/
    mod.rs
    gap_detector.rs
    strategy_selector.rs
    passive_recall.rs
    self_study.rs
    inquiry.rs
    answer_assimilation.rs
    source_policy.rs
    tests.rs
```

Python bridge layer can mirror:

```text
layer2/acquisition.py
```

or:

```text
executive/acquisition.py
```

---

# Current Code Adjustments

## 1. `pipeline.py`

Current pipeline already has context, situation, predictive, pattern, reasoning, appraise.

Add acquisition stage before final reasoning when uncertainty is high:

```text
_run_context_layer
_run_situation_layer
_run_acquisition_layer
_run_reasoning_layer
_appraise
```

New stage:

```python
def _run_acquisition_layer(self, layer0_output, layer1_output, reasoning_request):
    gaps = self.acquisition.detect_gaps(...)
    decision = self.acquisition.select_strategy(gaps, ...)
    return self.acquisition.execute(decision)
```

---

## 2. `layer2/context.py`

Current context layer appears responsible for external search.

Refactor it into a self-study provider.

It should not decide everything by itself.

Executive/Acquisition decides whether to call it.

Context layer becomes:

```text
SelfStudyProvider
```

---

## 3. `layer2/situation.py`

Situation layer should expose missing context signals.

Add:

```python
detect_private_context_gap()
detect_ambiguous_reference()
detect_goal_ambiguity()
```

---

## 4. `layer1/rsvs-core`

Add acquisition metadata to runtime events.

Examples:

```text
knowledge_gap_detected
acquisition_decision
self_study_started
self_study_completed
user_question_generated
user_answer_assimilated
```

---

## 5. `events.rs`

Add new event payload types.

Required runtime events:

```text
knowledge_gap_detected
acquisition_mode_selected
self_study_request_created
external_claim_ingested
inquiry_question_created
user_answer_assimilated
```

---

## 6. `types.rs`

Add shared types:

```text
KnowledgeGap
KnowledgeGapType
AcquisitionMode
AcquisitionDecision
InquiryQuestion
SelfStudyResult
```

---

## 7. `validation_gates`

Add a new gate:

```text
AcquisitionDisciplineGate
```

Purpose:

Prevent bad behavior:

```text
asking too early
researching private context
using low-quality sources
skipping acquisition when needed
```

---

# Acquisition Strategy Selector

Pseudo logic:

```rust
fn select_strategy(gap: &KnowledgeGap, context: &ExecutiveContext) -> AcquisitionDecision {
    if gap.gap_type == KnowledgeGapType::NoGap {
        return PassiveRecall;
    }

    if graph_can_resolve(gap) {
        return PassiveRecall;
    }

    if is_public_or_researchable(gap) {
        return SelfStudy;
    }

    return AskUser;
}
```

---

# Self Study Discipline

Self-study should not be uncontrolled browsing.

It must answer:

```text
What exactly am I trying to learn?
Which sources are allowed?
What claim did I extract?
How reliable is the source?
Did this resolve the gap?
```

---

# Ask User Discipline

Questions must be minimal and high-value.

Bad:

```text
Can you clarify?
```

Good:

```text
Who does "dia" refer to here?
```

or:

```text
What specific action made you call it betrayal?
```

AAM should ask only the smallest question needed to resolve the biggest uncertainty.

---

# Inquiry Memory

AAM must remember:

```text
asked question
user answer
resolved gap
new belief created
assumption corrected
```

This prevents repeated questions.

---

# User Answer Assimilation

When user answers, run:

```text
Semantic Frame Compiler
Pre-Ingest Meaning Reasoner
Epistemic Governance
RSVS Ingest
```

User answers are not plain chat responses.

They are acquisition events.

---

# Example 1 — Public Knowledge Gap

Input:

```text
Explain quantum annealing.
```

Flow:

```text
Passive Recall: graph weak
Self Study: research public sources
Ask User: not needed
```

Output:

```text
External knowledge claims with provenance
```

---

# Example 2 — Private Context Gap

Input:

```text
He betrayed me.
```

Flow:

```text
Passive Recall: insufficient
Self Study: impossible
Ask User: required
```

Question:

```text
Who does "he" refer to, and what did he do?
```

---

# Example 3 — Project Architecture Gap

Input:

```text
Improve my AAM architecture.
```

Flow:

```text
Passive Recall: use graph memory
Self Study: inspect repo/docs if available
Ask User: only if constraints remain missing
```

---

# Example 4 — Freshness Gap

Input:

```text
What is the latest tax regulation?
```

Flow:

```text
Passive Recall: outdated risk
Self Study: required
Ask User: not needed unless local/personal constraint missing
```

---

# Example 5 — Ambiguous Goal

Input:

```text
Make it better.
```

Flow:

```text
Passive Recall: insufficient
Self Study: not enough because "it" is unclear
Ask User: required
```

Question:

```text
What exactly should be improved?
```

---

# Acceptance Criteria

Implementation succeeds if:

1. AAM detects knowledge gaps explicitly.
2. AAM tries passive recall first.
3. AAM self-studies before asking when the missing knowledge is public.
4. AAM asks user only when needed.
5. User answers become memory events.
6. Self-study outputs are governed by MD-4.
7. Acquisition decisions are recorded as runtime events.
8. The system avoids repeated questions.
9. The system does not research private/user-owned unknowns unnecessarily.
10. The system does not treat researched claims as immediately grounded truth.

---

# Non-Goals

Do not implement:

- open-ended autonomous internet crawling
- uncontrolled background research
- LLM-based hidden reasoning
- direct truth promotion from search result
- excessive questioning
- user interrogation loops

---

# Final Statement

MD-6 transforms AAM from:

```text
a reasoning system that waits for enough data
```

into:

```text
a knowledge-seeking cognitive system that knows how to acquire missing context
```

The acquisition hierarchy is:

```text
Remember first.
Study second.
Ask last.
```

This is the correct discipline for early-stage graph intelligence.

It protects graph quality, minimizes user burden, and makes AAM capable of growing its own knowledge structure over time.
