# MD-1 — Semantic Frame Compiler (Elegant Architecture)

> **Prerequisite**: MD-3 defines the unified types (SemanticAtom, AtomType, SemanticRole,
> EdgeSource, Transform). This document defines the `ExtractFrame` Transform — the first
> enrichment step in the unified pipeline.

---

## Mission

Implement the `ExtractFrame` Transform: converts sentence-like text into
`SemanticAtom(Event, ...)` atoms with semantic role structure.

This Transform is the bridge from raw text to structured event knowledge. It sits
alongside `Tokenize` in the Atomizer stage. Both produce `SemanticAtom` — one sparse,
one rich. The downstream pipeline treats them uniformly.

---

## Core Constraint

No LLMs. Deterministic. Auditable. Rule-based in Phase 1.

---

## Transform Definition

```rust
/// ExtractFrame Transform
///
/// Input:  &str (raw text)
/// Output: Option<SemanticAtom> — Some if text is sentence-like and frame extracted
///
/// This transform runs AFTER Tokenize. Both produce SemanticAtom.
/// The pipeline decides whether to call ExtractFrame based on is_sentence_like().
pub struct ExtractFrame {
    config: FrameCompilerConfig,
}

impl Transform for ExtractFrame {
    type Input = &'static str;
    type Output = Option<SemanticAtom>;

    fn id(&self) -> &'static str { "ExtractFrame" }

    fn transform(&self, text: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        if !is_sentence_like(text) {
            return None;
        }

        let frame = self.extract_frame(text)?;

        Some(SemanticAtom {
            id: format!("evt_{}", ctx.next_atom_id()),
            label: frame.predicate,
            atom_type: AtomType::Event,
            roles: frame.roles,
            polarity: Some(frame.polarity),
            voice: Some(frame.voice),
            variant: Some(AtomVariant::FrameVariant(FrameSource::RuleBased)),
            confidence: frame.confidence,
            source: EdgeSource::FrameCompiler,
        })
    }
}
```

---

## Frame Extraction — Phase 1: Rule-Based

### ExtractionResult (internal, not a graph type)

```rust
/// Internal result of frame extraction.
/// Converted to SemanticAtom by the Transform.
struct ExtractionResult {
    predicate: String,
    roles: HashMap<SemanticRole, String>,
    polarity: Polarity,
    voice: Voice,
    confidence: f32,
}
```

### Rule-Based Strategies

```rust
impl ExtractFrame {
    fn extract_frame(&self, text: &str) -> Option<ExtractionResult> {
        // 1. Tokenize text
        let tokens = tokenize(text);

        // 2. Detect voice
        let voice = detect_voice(&tokens);

        // 3. Detect polarity
        let polarity = detect_polarity(&tokens);

        // 4. Extract predicate (verb-like token)
        let predicate = extract_predicate(&tokens)?;

        // 5. Extract roles based on clause patterns
        let mut roles = HashMap::new();

        // Agent/patient extraction
        if voice == Voice::Active {
            if let Some(agent) = extract_agent(&tokens, &predicate) {
                roles.insert(SemanticRole::Arg0Agent, agent);
            }
            if let Some(patient) = extract_patient(&tokens, &predicate) {
                roles.insert(SemanticRole::Arg1Patient, patient);
            }
        } else {
            // Passive: patient is grammatical subject, agent is "oleh" phrase
            if let Some(patient) = extract_passive_subject(&tokens, &predicate) {
                roles.insert(SemanticRole::Arg1Patient, patient);
            }
            if let Some(agent) = extract_by_phrase(&tokens) {
                roles.insert(SemanticRole::Arg0Agent, agent);
            }
        }

        // 6. Extract cause ("karena"/"because" clause)
        if let Some(cause) = extract_cause_clause(text) {
            roles.insert(SemanticRole::Cause, cause);
        }

        // 7. Extract purpose ("untuk"/"for" clause)
        if let Some(purpose) = extract_purpose_clause(text) {
            roles.insert(SemanticRole::Purpose, purpose);
        }

        // 8. Compute confidence
        let confidence = compute_frame_confidence(&roles, &polarity);

        Some(ExtractionResult { predicate, roles, polarity, voice, confidence })
    }
}
```

### Sentence Detection

```rust
/// Heuristic: is this text likely a sentence (vs a single token)?
pub fn is_sentence_like(text: &str) -> bool {
    let tokens: Vec<&str> = text.split_whitespace().collect();

    // Must have at least 3 tokens
    if tokens.len() < 3 {
        return false;
    }

    // Must contain at least one verb-like token
    let has_verb = tokens.iter().any(|t| looks_predicate_like(t));
    if !has_verb {
        return false;
    }

    // Not purely repetitive tokens
    let unique_count = tokens.iter().collect::<HashSet<_>>().len();
    if unique_count < 2 {
        return false;
    }

    true
}

fn looks_predicate_like(token: &str) -> bool {
    // Indonesian: me- prefix, ber- prefix, di- prefix
    // English: common verb suffixes
    let lower = token.to_lowercase();
    lower.starts_with("me") || lower.starts_with("ber") || lower.starts_with("di")
    || lower.ends_with("ify") || lower.ends_with("ize") || lower.ends_with("ate")
    || lower.ends_with("ing") || lower.ends_with("ed")
}
```

### Voice Detection

```rust
fn detect_voice(tokens: &[&str]) -> Voice {
    // Indonesian: di- prefix on predicate = passive
    // English: "was"/"were"/"is" + past participle = passive
    for token in tokens {
        let lower = token.to_lowercase();
        if lower.starts_with("di") && lower.len() > 3 {
            return Voice::Passive;
        }
    }
    Voice::Active
}
```

### Polarity Detection

```rust
fn detect_polarity(tokens: &[&str]) -> Polarity {
    // Indonesian: "tidak", "bukan", "tak", "jangan"
    // English: "not", "no", "never", "don't"
    let negation_markers = ["tidak", "bukan", "tak", "jangan", "not", "no", "never", "don't"];
    for token in tokens {
        if negation_markers.contains(&token.to_lowercase().as_str()) {
            return Polarity::Negative;
        }
    }
    Polarity::Positive
}
```

### Cause/Purpose Clause Extraction

```rust
fn extract_cause_clause(text: &str) -> Option<String> {
    // Split on "karena"/"because" and take the clause after it
    let markers = ["karena", "because", "since", "sebab"];
    for marker in markers {
        if let Some(pos) = text.to_lowercase().find(marker) {
            let clause = text[pos + marker.len()..].trim();
            if !clause.is_empty() {
                return Some(clause.to_string());
            }
        }
    }
    None
}

fn extract_purpose_clause(text: &str) -> Option<String> {
    let markers = ["untuk", "for", "agar", "supaya", "in order to"];
    for marker in markers {
        if let Some(pos) = text.to_lowercase().find(marker) {
            let clause = text[pos + marker.len()..].trim();
            if !clause.is_empty() {
                return Some(clause.to_string());
            }
        }
    }
    None
}
```

### Confidence Computation

```rust
fn compute_frame_confidence(roles: &HashMap<SemanticRole, String>, polarity: &Polarity) -> f32 {
    let mut score = 0.30; // base

    if roles.contains_key(&SemanticRole::Arg0Agent) { score += 0.15; }
    if roles.contains_key(&SemanticRole::Arg1Patient) { score += 0.15; }
    if roles.contains_key(&SemanticRole::Cause) { score += 0.10; }
    if roles.contains_key(&SemanticRole::Purpose) { score += 0.10; }

    if *polarity == Polarity::Negative { score -= 0.05; } // slight penalty

    score.clamp(0.0, 1.0)
}
```

---

## Graph Integration — IngestAtoms Transform

When `IngestAtoms` receives a `SemanticAtom(Event, ...)`, it creates:

1. **Nodes** for each role-filler label (or finds existing)
2. **A Composition** of type `CompositionType::Event`
3. **SemanticEdges** from the composition to each member node

```rust
// Inside IngestAtoms transform
fn ingest_event_atom(&mut self, atom: &SemanticAtom, graph: &mut Graph) -> GraphDelta {
    let mut delta = GraphDelta::new();

    // 1. Create composition node
    let comp_id = CompositionId::new();
    let mut members = Vec::new();

    // 2. Ensure predicate node exists
    let predicate_id = graph.ensure_node(&atom.label);
    members.push(CompositionMember {
        node_id: predicate_id,
        role: SemanticRole::Predicate,
        confidence: atom.confidence,
    });

    // 3. For each role, ensure node and add composition member
    for (role, label) in &atom.roles {
        let node_id = graph.ensure_node(label);
        members.push(CompositionMember {
            node_id,
            role: role.clone(),
            confidence: atom.confidence,
        });

        // Add SemanticEdge
        delta.add_edge(comp_id.clone(), node_id, SemanticEdge {
            relation: RelationType::Categorical,  // membership relation
            role: Some(role.clone()),
            source: atom.source.clone(),
        });
    }

    // 4. Create Composition
    delta.add_composition(Composition {
        id: comp_id,
        composition_type: CompositionType::Event,
        members,
        lifecycle: LifecycleState::New,
        epistemic: EpistemicState::Observed,
        confidence: atom.confidence,
        provenance: ProvenanceChain {
            origin: atom.source.clone(),
            origin_id: atom.id.clone(),
            parent_composition_id: None,
            timestamp: now_iso8601(),
        },
        seed_scores: HashMap::new(),  // filled by SeedAnchor transform
        created_at: now_iso8601(),
        updated_at: now_iso8601(),
    });

    delta
}
```

---

## Phase 2: UD + SRL Integration (Deferred)

When proper NLP models are available, `ExtractFrame` gains additional strategies:

```rust
impl ExtractFrame {
    fn extract_frame(&self, text: &str) -> Option<ExtractionResult> {
        // Phase 1: try rule-based
        if let Some(result) = self.rule_based_extract(text) {
            return Some(result);
        }

        // Phase 2: try UD + SRL (if configured)
        if self.config.use_ud_srl {
            if let Some(result) = self.ud_srl_extract(text) {
                return Some(result);
            }
        }

        None
    }
}
```

The `AtomVariant::FrameVariant` distinguishes: `RuleBased` vs `UdParse` vs `SrlLabel`.

---

## Module Structure

```text
layer0/
  frame_compiler/
    mod.rs              // ExtractFrame Transform + public API
    rules.rs            // rule-based extraction strategies
    sentence_detect.rs  // is_sentence_like() heuristic
    confidence.rs       // frame confidence computation
    tests.rs            // unit tests
```

5 files. Minimal.

---

## Required Tests

### Test 1 — Active Sentence

```text
Input:  "Raymond membuat aplikasi"
Output: SemanticAtom { atom_type: Event, label: "membuat",
         roles: {Arg0Agent: "Raymond", Arg1Patient: "aplikasi"},
         polarity: Positive, voice: Active }
```

### Test 2 — Sentence with Cause

```text
Input:  "Raymond membuat aplikasi karena proses manual terlalu lambat"
Output: SemanticAtom { ..., roles: {Arg0Agent: "Raymond", Arg1Patient: "aplikasi",
         Cause: "proses manual terlalu lambat"} }
```

### Test 3 — Passive Sentence

```text
Input:  "Aplikasi dibuat oleh Raymond"
Output: SemanticAtom { ..., voice: Passive,
         roles: {Arg1Patient: "Aplikasi", Arg0Agent: "Raymond"} }
```

### Test 4 — Negated Sentence

```text
Input:  "Raymond tidak membuat aplikasi"
Output: SemanticAtom { ..., polarity: Negative }
```

### Test 5 — Token Input (No Frame)

```text
Input:  "raja"
Output: None (is_sentence_like returns false)
```

### Test 6 — Frame Atom Ingested as Composition

Verify that `IngestAtoms` creates a `Composition { composition_type: Event, ... }`
with correct members and SemanticEdges.

---

## Acceptance Criteria

1. `ExtractFrame` Transform implemented and registered
2. Produces `SemanticAtom(Event, ...)` for sentence-like input
3. Returns `None` for token-like input
4. Rule-based extraction covers: active, passive, negated, cause, purpose
5. `is_sentence_like()` correctly distinguishes sentences from tokens
6. Frame atoms ingested as `Composition(Event)` with `SemanticEdge` per role
7. All existing tests remain green (Transform is additive)

---

## Final Statement

MD-1 implements the first enrichment Transform in the elegant architecture. It produces
`SemanticAtom(Event, ...)` atoms that flow through the same unified pipeline as token atoms.
No dual-track. No separate EventFrame type. One ingest path, varying richness.
