# MD-2 — Pre-Ingest Meaning Reasoner / Hidden Meaning Compiler

## Context

Assume **MD-1: RSVS Semantic Frame Compiler** has already been successfully implemented.

That means the system can already transform raw plain text into deterministic semantic event frames without using an LLM.

Example input:

```text
Raymond membuat aplikasi untuk kantor karena proses manual terlalu lambat.
```

Expected MD-1 output:

```json
{
  "event_id": "e1",
  "predicate": "membuat",
  "ARG0_agent": "Raymond",
  "ARG1_patient": "aplikasi",
  "PNC_purpose": "kantor",
  "CAU_cause": "proses manual terlalu lambat",
  "polarity": "positive",
  "voice": "active"
}
```

MD-2 starts from this point.

---

# Mission

Build a **Pre-Ingest Meaning Reasoner**.

This layer must run **after Semantic Frame Compiler** and **before RSVS graph ingestion**.

Its job is not merely to parse subject, predicate, object, cause, or purpose.

Its job is to discover **hidden meaning candidates** that are not directly visible on the surface of the sentence but are implied by the relationship between:

- atoms
- semantic roles
- event frames
- existing RSVS senses
- context
- situation
- composition patterns
- cause-effect structure
- purpose/goal structure
- contradiction or tension between roles

Core statement:

> The Pre-Ingest Meaning Reasoner exists so RSVS does not only store what the text says, but also receives structured candidates for what the text implies.

---

# Non-Negotiable Constraint

Do **not** use LLMs.

This must be:

- deterministic
- auditable
- rule-guided
- graph-guided
- explainable
- testable

No hidden model inference.

No prompt-based extraction.

No probabilistic black-box language model.

---

# High-Level Pipeline

```text
Raw Text
→ Semantic Frame Compiler        // already implemented in MD-1
→ Event Frames
→ Pre-Ingest Meaning Reasoner    // this MD-2
→ Hidden Meaning Candidates
→ RSVS Sense / Composition Ingest
→ Grounding + Reflection
```

Short form:

```text
Text → Frame → Hidden Meaning → RSVS
```

---

# Why This Layer Exists

A semantic frame captures surface structure.

Example:

```json
{
  "predicate": "membuat",
  "agent": "Raymond",
  "patient": "aplikasi",
  "purpose": "kantor",
  "cause": "proses manual terlalu lambat"
}
```

This is useful, but still incomplete.

The hidden meaning is:

```text
manual_process_lambat → pain_point
aplikasi → tool_solution
Raymond → problem_solver / agent
kantor → beneficiary / operational_context
membuat_aplikasi → response_to_inefficiency
```

Therefore the system should emit meaning candidates such as:

```json
{
  "type": "problem_solution_pattern",
  "problem": "proses manual terlalu lambat",
  "solution": "aplikasi",
  "agent": "Raymond",
  "beneficiary": "kantor",
  "confidence": 0.86,
  "evidence": ["CAU_cause", "ARG1_patient", "ARG0_agent", "PNC_purpose"]
}
```

---

# Terminology

## Surface Meaning

Meaning explicitly stated in the event frame.

Example:

```text
Raymond membuat aplikasi.
```

Surface meaning:

```text
agent = Raymond
predicate = membuat
patient = aplikasi
```

## Hidden Meaning

Meaning inferred from structural relationships.

Example:

```text
Raymond membuat aplikasi karena proses manual terlalu lambat.
```

Hidden meaning:

```text
The application is likely a solution to an operational inefficiency.
```

## Candidate, Not Fact

Pre-ingest reasoning must output **candidates**, not absolute truth.

A hidden meaning should not be committed as guaranteed truth immediately.

It should enter RSVS with:

- confidence
- evidence links
- source frame
- reasoning rule
- provisional status

---

# Required Output Format

The Pre-Ingest Meaning Reasoner should output:

```json
{
  "source_event_id": "e1",
  "surface_frame": {
    "predicate": "membuat",
    "ARG0_agent": "Raymond",
    "ARG1_patient": "aplikasi",
    "PNC_purpose": "kantor",
    "CAU_cause": "proses manual terlalu lambat",
    "polarity": "positive",
    "voice": "active"
  },
  "hidden_meaning_candidates": [
    {
      "candidate_id": "hm1",
      "type": "problem_solution_pattern",
      "description": "The application is likely a solution to a slow manual process.",
      "nodes": {
        "problem": "proses manual terlalu lambat",
        "solution": "aplikasi",
        "agent": "Raymond",
        "beneficiary": "kantor"
      },
      "composition_hints": [
        "problem:proses_manual_terlalu_lambat",
        "solution:aplikasi",
        "agent:Raymond",
        "beneficiary:kantor",
        "pattern:problem_solution"
      ],
      "confidence": 0.86,
      "evidence_roles": ["CAU_cause", "ARG1_patient", "ARG0_agent", "PNC_purpose"],
      "rule_id": "CAUSE_ACTION_PURPOSE_TO_PROBLEM_SOLUTION",
      "status": "candidate"
    }
  ]
}
```

---

# Core Data Structures

## EventFrame

This should already exist from MD-1.

```rust
pub struct EventFrame {
    pub event_id: String,
    pub predicate: String,
    pub arg0_agent: Option<String>,
    pub arg1_patient: Option<String>,
    pub arg2: Option<String>,
    pub cause: Option<String>,
    pub purpose: Option<String>,
    pub location: Option<String>,
    pub time: Option<String>,
    pub polarity: Polarity,
    pub voice: Voice,
    pub confidence: f32,
}
```

## HiddenMeaningCandidate

Add a new structure:

```rust
pub struct HiddenMeaningCandidate {
    pub candidate_id: String,
    pub source_event_id: String,
    pub meaning_type: HiddenMeaningType,
    pub description: String,
    pub nodes: Vec<MeaningNodeRef>,
    pub composition_hints: Vec<CompositionHint>,
    pub evidence_roles: Vec<RoleRef>,
    pub rule_id: String,
    pub confidence: f32,
    pub status: CandidateStatus,
}
```

## HiddenMeaningType

```rust
pub enum HiddenMeaningType {
    ProblemSolutionPattern,
    MotivationInference,
    GoalInference,
    OperationalContextInference,
    AgentResponsibility,
    BeneficiaryInference,
    CauseEffectPattern,
    ToolUsePattern,
    InefficiencySignal,
    ContradictionSignal,
    PolarityConflict,
    PurposeConflict,
    RoleAnomaly,
    Unknown,
}
```

## CandidateStatus

```rust
pub enum CandidateStatus {
    Candidate,
    Confirmed,
    Contradicted,
    Deprecated,
}
```

## CompositionHint

```rust
pub struct CompositionHint {
    pub role: String,
    pub node_label: String,
    pub confidence: f32,
}
```

---

# Core Reasoning Rules

## Rule 1 — Cause + Action + Object → Problem-Solution Pattern

Trigger condition:

```text
frame has CAU_cause
AND frame has predicate
AND frame has ARG1_patient
AND predicate belongs to action/create/fix/build/solve family
```

Example:

```text
Raymond membuat aplikasi karena proses manual terlalu lambat.
```

Output:

```json
{
  "type": "problem_solution_pattern",
  "problem": "proses manual terlalu lambat",
  "solution": "aplikasi",
  "agent": "Raymond"
}
```

Interpretation:

```text
The patient/object of the action is likely a solution to the cause.
```

---

## Rule 2 — Purpose Marker → Goal Inference

Trigger condition:

```text
frame has PNC_purpose
```

Example:

```text
Raymond membuat aplikasi untuk kantor.
```

Output:

```json
{
  "type": "goal_inference",
  "goal": "kantor",
  "action": "membuat",
  "agent": "Raymond"
}
```

Interpretation:

```text
The action is directed toward the purpose/beneficiary.
```

---

## Rule 3 — Agent + Active Predicate → Agent Responsibility

Trigger condition:

```text
voice = active
AND ARG0_agent exists
AND predicate exists
```

Output:

```json
{
  "type": "agent_responsibility",
  "agent": "Raymond",
  "action": "membuat",
  "patient": "aplikasi"
}
```

Interpretation:

```text
The agent is responsible for performing the action.
```

---

## Rule 4 — Passive Normalization

Trigger condition:

```text
voice = passive
AND patient exists
AND agent exists
```

Example:

```text
Aplikasi dibuat oleh Raymond.
```

Normalize to:

```json
{
  "predicate": "membuat",
  "agent": "Raymond",
  "patient": "aplikasi"
}
```

Interpretation:

```text
Passive and active versions should map to the same underlying event meaning.
```

This is important for RSVS convergence.

---

## Rule 5 — Cause Contains Negative Quality → Pain Point Signal

Trigger condition:

```text
CAU_cause exists
AND cause contains negative state markers
```

Negative state markers:

```text
lambat
gagal
salah
rusak
mahal
manual
tidak efisien
terlalu lama
berulang
repetitif
rawan error
```

Output:

```json
{
  "type": "inefficiency_signal",
  "pain_point": "proses manual terlalu lambat"
}
```

---

## Rule 6 — Tool/Object Created Due to Pain Point → Tool-Use Pattern

Trigger condition:

```text
problem_solution_pattern exists
AND ARG1_patient belongs to tool/software/system/document/process family
```

Output:

```json
{
  "type": "tool_use_pattern",
  "tool": "aplikasi",
  "problem": "proses manual terlalu lambat",
  "agent": "Raymond"
}
```

---

## Rule 7 — Same Event Structure + Opposite Polarity → Direct Contradiction

Trigger condition:

```text
same predicate
same agent
same patient
opposite polarity
```

Example:

```text
Raymond membuat aplikasi.
Raymond tidak membuat aplikasi.
```

Output:

```json
{
  "type": "polarity_conflict",
  "event_a": "e1",
  "event_b": "e2",
  "conflict": "same event but opposite polarity"
}
```

---

## Rule 8 — Same Event + Different Purpose → Purpose Conflict

Trigger condition:

```text
same predicate
same agent
same patient
different PNC_purpose
```

Example:

```text
Raymond membuat aplikasi untuk kantor.
Raymond membuat aplikasi untuk sekolah.
```

Output:

```json
{
  "type": "purpose_conflict",
  "event_a_purpose": "kantor",
  "event_b_purpose": "sekolah"
}
```

---

## Rule 9 — Agent-Patient Reversal → Role Anomaly

Trigger condition:

```text
event_a.agent == event_b.patient
AND event_a.patient == event_b.agent
AND same predicate
```

Example:

```text
Raymond membuat aplikasi.
Aplikasi membuat Raymond.
```

Output:

```json
{
  "type": "role_anomaly",
  "description": "Agent and patient roles are reversed for the same predicate."
}
```

This is critical because flat token similarity would falsely treat both as similar.

---

# Graph-Guided Enhancement

The reasoner should not rely only on hardcoded rules.

It should query RSVS graph to improve confidence.

Examples:

```text
Is "aplikasi" close to tool/software/system?
Is "proses manual terlalu lambat" close to inefficiency/pain_point?
Is "membuat" close to create/build/produce/action?
Is "kantor" close to beneficiary/organization/workplace?
```

If graph confirms role meaning, increase confidence.

If graph contradicts role meaning, reduce confidence.

---

# Confidence Scoring

Use deterministic scoring.

Example for problem-solution pattern:

```text
base = 0.40

+0.15 if CAU_cause exists
+0.15 if ARG1_patient exists
+0.10 if ARG0_agent exists
+0.10 if predicate belongs to create/build/fix/solve/action family
+0.10 if cause contains negative/pain marker
+0.10 if patient belongs to tool/system/software family
-0.20 if polarity is negative
```

Clamp:

```text
0.0 <= confidence <= 1.0
```

Example result:

```text
0.40 + 0.15 + 0.15 + 0.10 + 0.10 + 0.10 = 1.00
```

Then optionally cap candidate confidence at frame confidence:

```text
candidate_confidence = min(rule_score, frame.confidence)
```

If frame confidence is `0.94`, final confidence should not exceed `0.94`.

---

# Integration With RSVS

The Pre-Ingest Meaning Reasoner should emit candidates that can become RSVS compositions.

Example candidate:

```json
{
  "type": "problem_solution_pattern",
  "problem": "proses manual terlalu lambat",
  "solution": "aplikasi",
  "agent": "Raymond",
  "beneficiary": "kantor"
}
```

Map to RSVS composition:

```text
hidden_meaning_hm1
  composition:
    pattern:problem_solution
    problem:proses_manual_terlalu_lambat
    solution:aplikasi
    agent:Raymond
    beneficiary:kantor
```

Typed edges:

```text
hm1 --pattern--> problem_solution
hm1 --problem--> proses_manual_terlalu_lambat
hm1 --solution--> aplikasi
hm1 --agent--> Raymond
hm1 --beneficiary--> kantor
hm1 --source_event--> e1
```

Do not immediately treat hidden meaning as absolute truth.

Register as provisional candidate:

```text
status = candidate
confidence = computed_score
grounding = pending
source = pre_ingest_reasoning
```

---

# Required Modules

Suggested structure:

```text
pre_ingest_reasoning/
  mod.rs
  types.rs
  rules.rs
  scorer.rs
  graph_context.rs
  candidate_generator.rs
  contradiction_detector.rs
  mapper.rs
  tests.rs
```

## `types.rs`

Defines:

- HiddenMeaningCandidate
- HiddenMeaningType
- CandidateStatus
- CompositionHint
- RoleRef
- ReasoningEvidence
- RuleResult

## `rules.rs`

Contains deterministic rules.

Each rule should implement:

```rust
pub trait PreIngestRule {
    fn id(&self) -> &'static str;
    fn applies(&self, frame: &EventFrame, context: &ReasoningContext) -> bool;
    fn generate(&self, frame: &EventFrame, context: &ReasoningContext) -> Vec<HiddenMeaningCandidate>;
}
```

## `scorer.rs`

Contains deterministic confidence functions.

## `graph_context.rs`

Provides graph lookup helpers:

```rust
role_score(node, role)
semantic_similarity(node, prototype)
sense_grounding(node)
composition_overlap(node, prototype)
```

## `candidate_generator.rs`

Runs all rules on frames.

## `contradiction_detector.rs`

Compares current event frame against prior event frames.

Detects:

- polarity conflict
- purpose conflict
- role reversal
- cause conflict
- time conflict
- location conflict

## `mapper.rs`

Converts hidden meaning candidates into RSVS ingest-ready composition hints.

---

# Required Tests

## Test 1 — Problem Solution Pattern

Input frame:

```json
{
  "predicate": "membuat",
  "ARG0_agent": "Raymond",
  "ARG1_patient": "aplikasi",
  "PNC_purpose": "kantor",
  "CAU_cause": "proses manual terlalu lambat"
}
```

Expected hidden meaning:

```text
problem_solution_pattern
inefficiency_signal
tool_use_pattern
goal_inference
agent_responsibility
```

---

## Test 2 — Passive Normalization

Input:

```text
Aplikasi dibuat oleh Raymond.
```

Expected normalized event:

```json
{
  "predicate": "membuat",
  "agent": "Raymond",
  "patient": "aplikasi"
}
```

Expected hidden meaning:

```text
same as active equivalent
```

---

## Test 3 — Polarity Conflict

Input events:

```text
Raymond membuat aplikasi.
Raymond tidak membuat aplikasi.
```

Expected:

```text
polarity_conflict
```

---

## Test 4 — Purpose Conflict

Input events:

```text
Raymond membuat aplikasi untuk kantor.
Raymond membuat aplikasi untuk sekolah.
```

Expected:

```text
purpose_conflict
```

---

## Test 5 — Role Reversal

Input events:

```text
Raymond membuat aplikasi.
Aplikasi membuat Raymond.
```

Expected:

```text
role_anomaly
```

---

## Test 6 — Cause-Based Motivation

Input:

```text
Raymond membuat aplikasi karena input manual sering salah.
```

Expected:

```text
motivation_inference
pain_point = input manual sering salah
implied_goal = reduce errors / improve accuracy
```

---

# Acceptance Criteria

The implementation is acceptable if:

1. It does not use LLMs.
2. It accepts EventFrame outputs from MD-1.
3. It emits HiddenMeaningCandidate structures.
4. Each candidate includes:
   - type
   - source event
   - evidence roles
   - rule id
   - confidence
   - composition hints
5. It can detect at least:
   - problem_solution_pattern
   - motivation_inference
   - goal_inference
   - agent_responsibility
   - polarity_conflict
   - purpose_conflict
   - role_anomaly
6. It can map candidates into RSVS composition hints.
7. All reasoning is auditable through rule IDs and evidence roles.
8. Tests pass for active, passive, negation, cause, purpose, conflict, and role reversal cases.

---

# Non-Goals

Do not implement:

- LLM-based reasoning
- text generation
- large model training
- probabilistic black-box extraction
- UI
- diffusion model integration
- final answer generation

This layer only prepares hidden meaning candidates for RSVS ingestion.

---

# Important Design Principle

Do not replace RSVS reasoning.

This layer should not become the entire brain.

It should produce **structured semantic hints** so RSVS can perform better sense induction, grounding, reflection, and pattern completion.

Correct relationship:

```text
Pre-Ingest Meaning Reasoner = semantic compiler / candidate generator
RSVS = long-term structural memory + sense reasoning + grounding engine
```

---

# Final Statement

The purpose of MD-2 is to make RSVS ingest not only surface facts, but also structured candidate meanings implied by the relation between atoms, roles, senses, and situation.

This enables RSVS to learn:

```text
not only what appeared together,
but why those parts appeared together.
```
