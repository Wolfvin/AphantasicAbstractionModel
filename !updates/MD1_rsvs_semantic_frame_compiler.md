# MD-1 — RSVS Semantic Frame Compiler (Adjusted for Implementation)

> **Adjustment Note (v11.0 alignment):** This document has been revised from the original
> research-direction spec into an implementation-ready blueprint. Key changes:
> - Phased approach: rule-based frame extraction FIRST, UD/SRL deferred
> - Parallel ingest mode alongside existing token pipeline (not replacement)
> - Explicit type alignment with existing `types.rs`, `RelationType`, `EdgeSource`
> - Operational reality: early-stage graphs ingest short repeated tokens, not full sentences
> - EventFrame defined as NEW type to add, with migration bridge
> - All 1,081 existing tests must remain green — every addition is additive

---

## Objective

Design a **non-LLM semantic ingestion pipeline** for RSVS that can transform raw natural language text into structured graph representations suitable for compositional reasoning.

Core requirements:

- No LLM intervention
- Deterministic / auditable
- Compatible with RSVS compositional graph architecture
- Preserve semantic roles, relations, causality, and nested meaning
- **Additive to existing token pipeline** — not a replacement

---

## Problem Statement

A naive token/co-occurrence ingest pipeline loses structural meaning.

Example:

**Input:**

```text
Raymond membuat aplikasi untuk kantor karena proses manual terlalu lambat.
```

Naive token ingest:

```text
Raymond, membuat, aplikasi, kantor, proses, manual, lambat
```

This loses:

- who did the action
- what received the action
- why the action happened
- purpose of the action
- clause structure

RSVS can reason over relations, but only if relations are represented structurally.

### Operational Reality Check

Current RSVS v11.0 ingests **short repeated tokens** (`raja`, `ratu`, `keras`), not full sentences. The token-based ingest path will remain the primary path for node promotion and co-occurrence learning. Frame-based ingest is an **enhancement layer** that activates when input is a complete sentence or clause.

Therefore:

```text
Token ingest = foundation (always active)
Frame ingest = enhancement (active when sentence detected)
```

Both paths converge into the same RSVS graph.

---

## Implementation Phases

### Phase 1 — Rule-Based Frame Extraction (IMMEDIATE)

Implement EventFrame extraction using deterministic rules, WITHOUT requiring UD/SRL parsers.

Rule-based strategies:

```text
1. Predicate-first extraction
   - Identify likely predicate (verb) from sentence
   - Extract surrounding noun phrases as agent/patient
   - Detect "karena"/"because" → cause slot
   - Detect "untuk"/"for" → purpose slot
   - Detect "di"/"at" → location slot

2. Pattern matching for common clause structures
   - [Agent] [predicate] [Patient] (karena|because) [Cause]
   - [Agent] [predicate] [Patient] (untuk|for) [Purpose]
   - [Patient] (dibuat|di-verb) (oleh|by) [Agent]  → passive detection

3. Negation detection
   - "tidak"/"not" + predicate → polarity = negative

4. Voice detection
   - di- prefix on verb → passive voice
   - me- prefix on verb → active voice
```

This is deterministic, language-aware but not language-locked, and requires zero external models.

### Phase 2 — UD + SRL Integration (DEFERRED)

After Phase 1 proves stable and useful, integrate actual UD parsing and SRL.

Requires:

- Trained dependency parser for target languages (Indonesian, English)
- PropBank-style semantic role labeler
- These are external model dependencies — must be optional plugins

### Phase 3 — AMR-Style Full Semantic Graph (FUTURE)

Full AMR-style nested semantic graph compilation. This is the research vision, not the immediate implementation target.

---

## Research-Backed Architectural Direction (Reference)

The full research direction remains:

```text
Universal Dependencies
→ Semantic Role Labeling
→ AMR-style semantic graph
→ RSVS ingestion
```

Phase 1 implements the **spirit** of this direction using rule-based heuristics.
Phase 2 adds proper NLP tooling when available.
Phase 3 completes the vision.

---

## Target Structured Representation — EventFrame (NEW TYPE)

This type does NOT exist in the current v11.0 codebase. It must be added to `types.rs`.

```rust
/// Structured semantic event extracted from text.
/// Produced by Semantic Frame Compiler (Layer 0.5).
/// Enters RSVS via frame_ingest adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFrame {
    pub event_id: String,
    pub predicate: String,
    pub arg0_agent: Option<String>,
    pub arg1_patient: Option<String>,
    pub arg2: Option<String>,         // recipient/beneficiary/instrument
    pub cause: Option<String>,
    pub purpose: Option<String>,
    pub location: Option<String>,
    pub time: Option<String>,
    pub polarity: Polarity,
    pub voice: Voice,
    pub confidence: f32,
    pub source: FrameSource,           // RuleBased, UdParse, SrlLabel
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Polarity {
    Positive,
    Negative,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Voice {
    Active,
    Passive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FrameSource {
    RuleBased,        // Phase 1
    UdParse,          // Phase 2
    SrlLabel,         // Phase 2
    AmrCompilation,   // Phase 3
}
```

---

## Semantic Edge Types — EXTEND RelationType

Current `RelationType` in `types.rs` has 7 variants:

```rust
pub enum RelationType {
    Categorical, Differential, Functional,
    Spatial, Temporal, Causal, Discursive,
}
```

Add semantic role edges as a **new parallel edge category** rather than modifying `RelationType`:

```rust
/// Semantic role edges produced by Frame Compiler.
/// Stored alongside existing RelationType edges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SemanticRole {
    Predicate,       // event → predicate node
    Arg0Agent,       // event → agent node
    Arg1Patient,     // event → patient node
    Arg2Recipient,   // event → recipient/instrument
    Cause,           // event → cause node
    Purpose,         // event → purpose node
    Location,        // event → location node
    Time,            // event → time node
    SourceEvent,     // hidden_meaning → source event
    HiddenCandidate, // event → hidden meaning node
    PatternType,     // hidden_meaning → pattern classification
}
```

Rationale: `RelationType` is used pervasively across 1,081 tests. Adding variants there would cascade changes. A parallel `SemanticRole` type keeps frame-based edges cleanly separated while still being usable in the graph.

---

## RSVS Graph Mapping

Transform EventFrame to graph nodes and edges:

```text
event_e1 (composition node)
  --SemanticRole::Predicate--> membuat
  --SemanticRole::Arg0Agent--> Raymond
  --SemanticRole::Arg1Patient--> aplikasi
  --SemanticRole::Purpose-->    kantor
  --SemanticRole::Cause-->      proses_manual_terlalu_lambat
```

Each frame becomes a **composition node** with typed semantic edges to its role-fillers.

This is compatible with existing composition machinery. The composition node uses `EdgeSource::FrameCompiler` (new variant to add to `EdgeSource`).

---

## EdgeSource Extension

Add one new variant to the existing `EdgeSource` enum:

```rust
pub enum EdgeSource {
    // ... existing 10 variants ...
    FrameCompiler,    // NEW: edge created by Semantic Frame Compiler
}
```

This is a single additive change — backward compatible.

---

## Hybrid Ingest Pipeline

The frame compiler does NOT replace the token ingest path. Both coexist:

```text
Raw Text
├── Token Path (existing, unchanged)
│   → tokenizer
│   → co-occurrence
│   → node promotion
│   → attention + sense induction
│
└── Frame Path (NEW)
    → sentence detection
    → rule-based frame extraction
    → EventFrame construction
    → frame_ingest adapter
    → composition nodes with semantic edges
```

Decision logic:

```rust
fn ingest_text(text: &str) -> IngestResult {
    // Always run token path (existing behavior)
    let token_result = existing_token_ingest(text);

    // Run frame path only if input looks like a sentence
    if is_sentence_like(text) {
        if let Some(frame) = frame_compiler.extract(text) {
            let frame_result = frame_ingest_adapter.ingest(frame);
            return merge_results(token_result, frame_result);
        }
    }

    token_result
}
```

`is_sentence_like()` heuristic:

```text
- contains at least one verb-like token
- has 3+ tokens
- not purely repetitive tokens
```

This ensures short token inputs (`raja`, `ratu`, `keras`) skip frame extraction entirely and use the fast existing path.

---

## Frame Ingest Adapter

Bridges EventFrame → RSVS graph:

```rust
pub struct FrameIngestAdapter {
    graph: Arc<Graph>,
}

impl FrameIngestAdapter {
    pub fn ingest(&self, frame: &EventFrame) -> FrameIngestResult {
        // 1. Create event composition node
        let event_node_id = self.graph.add_node(frame.event_id.clone());

        // 2. Ensure role-filler nodes exist (or find existing)
        let predicate_id = self.ensure_node(&frame.predicate);
        let agent_id = frame.arg0_agent.as_ref().map(|a| self.ensure_node(a));
        let patient_id = frame.arg1_patient.as_ref().map(|p| self.ensure_node(p));
        // ... etc for all filled roles

        // 3. Add typed semantic edges
        self.graph.add_edge(event_node_id, predicate_id,
            EdgeSource::FrameCompiler, SemanticRole::Predicate);
        if let Some(aid) = agent_id {
            self.graph.add_edge(event_node_id, aid,
                EdgeSource::FrameCompiler, SemanticRole::Arg0Agent);
        }
        // ... etc

        // 4. Trigger sense induction on new composition
        // (reuses existing sense induction pipeline)

        FrameIngestResult {
            event_node_id,
            nodes_created: ...,
            edges_created: ...,
        }
    }

    fn ensure_node(&self, label: &str) -> NodeId {
        // Find existing node by label, or create new one
        self.graph.find_by_label(label)
            .unwrap_or_else(|| self.graph.add_node(label.to_string()))
    }
}
```

---

## Module Structure

```text
layer0/
  frame_compiler/
    mod.rs              // public API: extract(), is_sentence_like()
    types.rs            // EventFrame, Polarity, Voice, FrameSource, SemanticRole
    rule_extractor.rs   // Phase 1: rule-based frame extraction
    sentence_detect.rs  // heuristic sentence detection
    ingest_adapter.rs   // EventFrame → RSVS graph bridge
    tests.rs            // unit tests for frame extraction

  // Future (Phase 2+):
  // ud_parser.rs        // dependency parsing adapter
  // srl_labeler.rs      // semantic role labeling adapter
  // amr_compiler.rs     // AMR-style graph compilation
```

Note: This is 6 files, not 8+. Kept minimal for early stage.

---

## Required Tests

### Test 1 — Simple Active Sentence

Input:

```text
Raymond membuat aplikasi
```

Expected frame:

```json
{
  "predicate": "membuat",
  "arg0_agent": "Raymond",
  "arg1_patient": "aplikasi",
  "polarity": "Positive",
  "voice": "Active"
}
```

### Test 2 — Sentence with Cause

Input:

```text
Raymond membuat aplikasi karena proses manual terlalu lambat
```

Expected frame:

```json
{
  "predicate": "membuat",
  "arg0_agent": "Raymond",
  "arg1_patient": "aplikasi",
  "cause": "proses manual terlalu lambat",
  "polarity": "Positive",
  "voice": "Active"
}
```

### Test 3 — Passive Sentence

Input:

```text
Aplikasi dibuat oleh Raymond
```

Expected frame:

```json
{
  "predicate": "membuat",
  "arg0_agent": "Raymond",
  "arg1_patient": "aplikasi",
  "polarity": "Positive",
  "voice": "Passive"
}
```

### Test 4 — Negated Sentence

Input:

```text
Raymond tidak membuat aplikasi
```

Expected frame:

```json
{
  "predicate": "membuat",
  "arg0_agent": "Raymond",
  "arg1_patient": "aplikasi",
  "polarity": "Negative",
  "voice": "Active"
}
```

### Test 5 — Short Token Input (No Frame)

Input:

```text
raja
```

Expected: `is_sentence_like()` returns false, no frame extraction attempted.

### Test 6 — Frame Ingest Produces Correct Graph Structure

Frame:

```json
{
  "predicate": "membuat",
  "arg0_agent": "Raymond",
  "arg1_patient": "aplikasi"
}
```

Expected graph:

```text
event_e1 --SemanticRole::Predicate--> membuat
event_e1 --SemanticRole::Arg0Agent--> Raymond
event_e1 --SemanticRole::Arg1Patient--> aplikasi
```

All edges have `EdgeSource::FrameCompiler`.

### Test 7 — Hybrid Ingest Preserves Token Path

Input:

```text
Raymond membuat aplikasi
```

Assert: Both token nodes AND frame composition node exist in graph.

---

## Alignment with Existing Codebase

| Existing Type | Relationship | Action |
|---------------|-------------|--------|
| `RelationType` (7 variants) | Kept unchanged | Add parallel `SemanticRole` |
| `EdgeSource` (10 variants) | Add 1 variant | `FrameCompiler` |
| `NodeId`, `Graph` | Reused directly | No change |
| `SenseEngine`, sense induction | Triggered after frame ingest | No change to existing code |
| `PipelineConfig` | Add `frame_compiler_enabled: bool` | Additive config field |
| `ingest_text()` | Add frame path alongside token path | Wrapped, not replaced |
| `Composition` machinery | Frame events become compositions | Reused |

---

## Acceptance Criteria

Phase 1 is acceptable if:

1. EventFrame type exists in `types.rs`
2. SemanticRole type exists for typed edges
3. Rule-based extraction works for: active, passive, negated, cause, purpose
4. Short token inputs skip frame extraction entirely
5. Frame ingest adapter creates composition nodes with semantic edges
6. Existing token ingest path is UNCHANGED
7. All 1,081 existing tests remain green
8. Frame extraction is deterministic and auditable
9. `is_sentence_like()` correctly distinguishes sentences from tokens
10. Hybrid ingest merges both paths into the same graph

---

## What This Enables Downstream

- **MD2**: Pre-Ingest Meaning Reasoner operates on EventFrame outputs
- **MD3**: Hybrid ingest pipeline is the foundation for architecture refactor
- **MD4**: EventFrame provides structured input for epistemic governance
- **MD5**: Frame complexity can inform cognitive mode selection
- **MD6**: Missing frame fields trigger knowledge gap detection

---

## Final Statement

MD-1 introduces structured semantic ingestion as an **additive parallel path**, not a replacement. Token-based ingest remains the foundation. Frame-based ingest enhances the graph with structured event knowledge when input is sentence-like. This ensures zero regression while enabling all downstream MDs.
