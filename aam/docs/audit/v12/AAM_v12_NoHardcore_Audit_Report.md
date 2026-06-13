# AAM v1.0.0 — Professional AI Architecture Audit Report

**Auditor**: AI Systems Architect (Independent)  
**Date**: 2026-05-22  
**Codebase**: `rsvs-core` v1.0.0 — 42 source files, ~15,000 LOC  
**Scope**: No-Hardcore Principle Compliance, Architectural Integrity, Emergent Learning Readiness  

---

## Executive Summary

The AAM v1.0.0 codebase demonstrates a **significant architectural evolution** from its prior hardcoded state. The introduction of `KnowledgeBase`, `KnowledgeOrigin`, `TeachingProtocol`, `SymbolicObserver`, and `AdaptiveParams` represents a genuine philosophical commitment to the "no-hardcore" principle — knowledge as data, not code.

However, this audit reveals that the transition is **incomplete and inconsistent**. The system exists in a **schizophrenic state**: the *infrastructure* for emergent learning is well-designed, but the *population* of that infrastructure still relies heavily on bootstrapped/hardcoded data. There are **7 critical violations**, **5 architectural weaknesses**, and **3 structural gaps** that prevent AAM from truly being a self-adaptive, observation-driven system.

**Severity Distribution**:
| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 CRITICAL | 7 | Hardcoded data that violates no-hardcore principle |
| 🟠 HIGH | 5 | Architectural weaknesses limiting self-adaptation |
| 🟡 MEDIUM | 3 | Structural gaps requiring design work |
| 🔵 LOW | 4 | Minor issues, code quality improvements |

---

## Part I: No-Hardcore Principle Compliance Audit

### The Principle

> "Everything must be created by we teach and AAM ask, by observation of AAM by relation of the symbolic, no hardcore — we must build this architect so it will adapt to all possibilities."

This means:
1. **Zero hardcoded linguistic knowledge** at `KnowledgeBase::new()`
2. All knowledge must enter via **Teaching**, **Asking**, or **Observation**
3. The system must be **language-agnostic** — Indonesian is a use case, not an architecture
4. Confidence thresholds, weights, and rules must be **self-calibrating**

### 🔴 H-1: `seed_indonesian()` Called Unconditionally in `PipelineEngine::new()`

**File**: `pipeline/engine.rs` line ~47  
**Severity**: CRITICAL — Violates blank-slate principle at the system's entry point  

```rust
pub fn new() -> Self {
    let mut context = PipelineContext::default();
    context.active_schemas = bootstrap_schemas();
    // No-Hardcore: Seed KnowledgeBase with Indonesian linguistic knowledge.
    super::super::knowledge_base::seed_indonesian(&mut context.knowledge_base);
    // ...
}
```

**Problem**: Every new `PipelineEngine` automatically receives Indonesian linguistic data with `KnowledgeOrigin::Bootstrapped`. There is no way to create a truly blank engine. The "blank slate" exists in `KnowledgeBase::new()` but is immediately violated by the engine constructor.

**Impact**: AAM cannot be instantiated for any language other than Indonesian without manually stripping the bootstrapped data. The architecture *says* it's blank-slate but *behaves* as if Indonesian is hardcoded.

**Recommendation**:
```rust
pub fn new() -> Self {
    // TRUE blank slate — no automatic seeding
    Self::new_blank()
}

pub fn new_with_locale(locale: &dyn Locale) -> Self {
    let mut engine = Self::new_blank();
    seed_from_locale(&mut engine.context.knowledge_base, locale);
    engine
}
```

---

### 🔴 H-2: `bootstrap_schemas()` Returns 5 Hardcoded Action Schemas

**File**: `action_schemas.rs` lines 349-429  
**Severity**: CRITICAL — Schemas are architectural DNA, not data  

The function creates 5 schemas (Copula, Possessive, Equative, Existential, Locative) with hardcoded:
- Trigger types (`SchemaTrigger::CopulaMarker`, etc.)
- Role bindings (`TokenBefore`, `TokenAfter`)
- Priority values (10, 9, 8, 7, 6)
- Composition types (`EquativeBinding`, `PossessiveBinding`)

**Problem**: These schemas encode **Indonesian-specific linguistic theory** (copula = equative binding, possessive = separate type). In other languages, copula might behave differently. The schema *structure* (trigger + roles + priority) should be learnable through observation.

**Impact**: AAM can never discover a new schema type — it's limited to the 5 pre-seeded categories. If a language has a schema that doesn't fit these 5, the system cannot represent it.

**Recommendation**: 
- Make `SchemaTrigger` fully data-driven (marker category → trigger, not enum variant)
- Allow schemas to be **induced** by `SymbolicObserver` when it detects repeated role patterns
- Add `SchemaInductionResult` to `ObservationResult` enum
- The 5 bootstrap schemas should be seeded via `TeachingProtocol`, not `bootstrap_schemas()`

---

### 🔴 H-3: `SenseRegistry::with_bootstrap_entries()` Hardcodes 10 Indonesian Homographs

**File**: `sense_registry.rs` lines 138-284  
**Severity**: CRITICAL — 10 hardcoded sense entries with 23 sense candidates  

The function hardcodes:
- `bisa` (venom/ability), `tahu` (know/tofu), `mangga` (fruit/please), `apis` (fire/possessive), `dapat` (obtain/can), `makan` (eat/consume), `tanam` (plant/bury), `buka` (open/event), `tinggal` (stay/remain/deceased), `kembali` (return/again)

Each with hardcoded `representative_labels` (context seeds for spreading activation).

**Problem**: These are **Indonesian-specific** and completely violate language-agnosticism. More critically, the sense entries are NOT stored with `KnowledgeOrigin` provenance — there's no way to trace HOW AAM learned that `bisa` means venom vs. ability.

**Impact**: 
1. The system "knows" Indonesian homographs without being taught — violates no-hardcore
2. No provenance tracking on sense entries (unlike markers which have `KnowledgeOrigin`)
3. Cannot be replaced by observation because `SenseRegistry` has no `KnowledgeOrigin` field

**Recommendation**:
- Add `KnowledgeOrigin` to `SenseEntry`
- Move all sense entries to `KnowledgeBase` with `Bootstrapped` provenance
- Implement sense discovery via `SymbolicObserver` (when a word appears in mutually exclusive contexts, it's likely ambiguous)

---

### 🔴 H-4: `IndonesianStemmer` Contains 3 Hardcoded `const` Arrays

**File**: `stemmer.rs` lines 36-59  
**Severity**: CRITICAL — ~50 root exceptions, 14 prefixes, 7 suffixes hardcoded as `const`  

```rust
const ROOT_EXCEPTIONS: &[&str] = &["makan", "minum", "tahu", ...]; // 49 entries
const PREFIXES_ORDERED: &[&str] = &["memper", "diper", "meng", ...]; // 14 entries
const SUFFIXES_ORDERED: &[&str] = &["kan", "an", "lah", ...]; // 7 entries
```

Plus allomorph data:
```rust
const ME_N_ALLOMORPHS_DATA: &[(&str, &str, &str)] = &[...]; // 5 entries
const PE_N_ALLOMORPHS_DATA: &[(&str, &str, &str)] = &[...]; // 5 entries
```

**Problem**: These are `const` arrays compiled into the binary. They cannot be:
- Taught at runtime
- Observed and induced
- Replaced for other languages
- Tracked with provenance

The `GraphAwareStemmer` partially addresses this (reads from graph), but falls back to the `const` arrays when the graph is empty.

**Impact**: The stemmer is the **deepest hardcode violation** — it's entirely procedural code with embedded linguistic data. This is the opposite of "knowledge as data."

**Recommendation**:
- Move ALL morphological data to `KnowledgeBase` (already partially done via `MorphologyRule`)
- Remove `const` arrays entirely
- `GraphAwareStemmer` should ONLY read from `KnowledgeBase`, never fallback to `const`
- Root exceptions should be learned: when AAM observes that stripping a prefix/suffix from "makan" produces "ak" (invalid), it should add "makan" as a root exception

---

### 🔴 H-5: `IndonesianLocale` Hardcodes 9 Marker Categories + Templates

**File**: `locale.rs` lines 149-245  
**Severity**: CRITICAL — All linguistic knowledge is in `&'static [&'static str]` returns  

The `IndonesianLocale` implementation hardcodes:
- 9 negation markers
- 4 core negation markers
- 2 cause markers
- 3 purpose markers
- 6 condition markers
- 6 verb prefixes
- 8 verbalization templates (Indonesian strings)
- 29 stopwords
- 6 epistemic qualifier strings

**Problem**: The `Locale` trait is an elegant abstraction, but its implementation is `&'static` — compile-time data that cannot be modified, observed, or taught. The Locale trait returns immutable static slices, so runtime learning is impossible through this interface.

**Impact**: The Locale system is a **dead end** for emergent learning — it's designed for compile-time i18n, not runtime observation. This directly contradicts the no-hardcore principle.

**Recommendation**:
- Replace `Locale` trait's `&'static [&'static str]` returns with runtime `Vec<String>` backed by `KnowledgeBase`
- Or: Make `Locale` a **seed source** that populates `KnowledgeBase` on initialization, then ALL lookups go through KB
- The verbalization templates should be in KB so they can be taught/observed too

---

### 🔴 H-6: 4 Reasoning Rules Hardcoded in `ReasonFrame::new()`

**File**: `reason_frame.rs` lines 737-748  
**Severity**: CRITICAL — Rules are code, not data  

```rust
pub fn new() -> Self {
    Self {
        rules: Arc::new(vec![
            Box::new(ProblemSolutionRule::new()),
            Box::new(GoalInferenceRule::new()),
            Box::new(PolarityConflictRule::new()),
            Box::new(ConditionConsequenceRule::new()),
        ]),
    }
}
```

**Problem**: The 4 reasoning rules are compiled into the binary. While the `ReasoningRule` trait allows custom rules, the default set is hardcoded. More critically, the rules themselves contain hardcoded logic:

- `ProblemSolutionRule`: Cause + Agent + Patient → ProblemSolutionPattern (confidence 0.85)
- `GoalInferenceRule`: Purpose → ImpliedGoal (confidence 0.80)
- `PolarityConflictRule`: Same predicate + opposite polarity (confidence 0.90)
- `ConditionConsequenceRule`: Antecedent + Consequent (confidence 0.90)

These confidence multipliers (0.85, 0.80, 0.90, 0.90) are **not** read from `AdaptiveParams`. They violate the self-calibrating principle.

**Impact**: 
1. AAM cannot discover new reasoning patterns through observation
2. The confidence multipliers are fixed and cannot be calibrated from feedback
3. The rule logic is procedural, not declarative — cannot be taught or modified

**Recommendation**:
- Read rule confidence multipliers from `KnowledgeBase.adaptive_params`
- Add `ReasoningRuleInduction` to `SymbolicObserver` — when certain role patterns consistently produce hidden meanings, propose a new rule
- Make rules serializable so they can be stored in the graph as data

---

### 🔴 H-7: `ExtractFrame` Contains Hardcoded Confidence Formula

**File**: `extract_frame.rs` lines 436-466  
**Severity**: CRITICAL — Formula is partially hardcoded  

```rust
pub fn compute_frame_confidence_with_kb(...) -> f32 {
    let mut confidence = kb.param("extract.base_confidence", 0.30);
    if roles.contains_key(&SemanticRole::Arg0Agent) {
        confidence += kb.param("extract.agent_bonus", 0.15);
    }
    // ... but:
    if roles.contains_key(&SemanticRole::Antecedent) {
        confidence += 0.10;  // ← HARDCODED! Not from KB
    }
    if roles.contains_key(&SemanticRole::Consequent) {
        confidence += 0.10;  // ← HARDCODED! Not from KB
    }
}
```

**Problem**: The `Antecedent` and `Consequent` bonuses are hardcoded as `0.10` instead of reading from `AdaptiveParams`. Additionally, the quality classification thresholds are hardcoded:

```rust
if confidence >= 0.70 && has_agent && has_patient { HighQuality }
else if confidence >= 0.45 { ModerateQuality }
```

The `0.70` and `0.45` thresholds are not from KB.

**Impact**: The confidence formula cannot self-calibrate for Antecedent/Consequent roles. Quality thresholds are fixed.

**Recommendation**: Move ALL bonus values and quality thresholds to `AdaptiveParams`.

---

## Part II: Architectural Weaknesses

### 🟠 A-1: `SymbolicObserver` Is Rudimentary — Only 3 Pattern Types

**File**: `knowledge_base.rs` lines 908-1014  
**Severity**: HIGH  

The `SymbolicObserver` currently implements only 3 pattern detectors:
1. Equative predicate → copula marker (threshold: 5 observations)
2. Possessive predicate → possessive marker (threshold: 5 observations)
3. Pre-verb word → verb-marking auxiliary (threshold: 10 observations, **but does nothing**)

**Critical gaps**:
- Pattern 3 observes but **never proposes** a new marker (dead code at line 998-1002)
- No morphological pattern induction (e.g., "words starting with 'me' are always verbs")
- No schema induction (e.g., "repeated Subject-Predicate-Object → Event schema")
- No reasoning rule induction (e.g., "when I see X+Y, I always derive Z")
- The observation threshold (5/10) is hardcoded, not from `AdaptiveParams`
- The observer is **not wired into the pipeline** — `observe_composition()` is never called by any transform

**Impact**: The entire "observation" learning path is essentially non-functional. AAM cannot learn anything by observing its own graph.

**Recommendation**:
- Wire `SymbolicObserver` into `IngestAtoms` or `GovernBeliefs` transform
- Implement all 6 pattern types from the architecture doc
- Read thresholds from `AdaptiveParams`
- Add `observe_atom()` for pre-composition observation

---

### 🟠 A-2: Dual Data Path — KnowledgeBase vs Locale vs SenseRegistry

**Severity**: HIGH — Three competing knowledge stores with no unification  

The system currently has **three separate knowledge stores**:

| Store | Data | Provenance | Mutable | Runtime-learning |
|-------|------|-----------|---------|-----------------|
| `KnowledgeBase` | Markers, morphology, params, stopwords, POS | ✅ `KnowledgeOrigin` | ✅ | ✅ |
| `IndonesianLocale` | Same markers + templates + stopwords | ❌ Compile-time | ❌ | ❌ |
| `SenseRegistry` | Sense entries, representative labels | ❌ None | ✅ | ✅ (via add_evidence) |

**Problem**: The same linguistic knowledge exists in multiple places:
- Negation markers: in `KnowledgeBase.markers[Negation]` AND `IndonesianLocale.negation_markers()`
- Verb prefixes: in `KnowledgeBase.markers[VerbPrefix]` AND `IndonesianLocale.verb_prefixes()`
- Stopwords: in `KnowledgeBase.stopwords` AND `IndonesianLocale.stopwords()`
- Sense entries: in `SenseRegistry` but NOT in `KnowledgeBase`

This creates **data synchronization risks**: if a marker is taught to `KnowledgeBase`, it won't appear in `Locale`, and vice versa.

**Recommendation**: 
- `Locale` should be a **one-time seed source** that populates `KnowledgeBase` on init
- After seeding, ALL lookups go through `KnowledgeBase` only
- `SenseRegistry` entries should be migrated to `KnowledgeBase` with `KnowledgeOrigin`
- Remove the `Locale` trait from runtime lookup paths

---

### 🟠 A-3: `SchemaTrigger` Enum is Closed — Cannot Discover New Trigger Types

**File**: `action_schemas.rs` lines 59-73  
**Severity**: HIGH  

```rust
pub enum SchemaTrigger {
    CopulaMarker,
    PossessiveMarker,
    EquativeMarker,
    ExistentialMarker,
    LocativeMarker,
    PredicatePattern(String),  // Only extensibility point
}
```

**Problem**: The 5 concrete variants encode linguistic theory. `PredicatePattern(String)` is the only extensibility point, but it uses substring matching — not marker-based. If AAM observes a new pattern (e.g., "comparative constructions using 'lebih'"), it cannot create a `SchemaTrigger::ComparativeMarker` at runtime.

**Impact**: The schema system is architecturally closed — AAM can only discover markers *within* existing trigger types, not new trigger types.

**Recommendation**:
- Replace enum variants with a single `MarkerTrigger(MarkerCategory)` variant
- Allow `MarkerCategory::Custom(String)` for runtime-discovered categories
- The `SymbolicObserver` should propose new `MarkerCategory` values when it detects novel patterns

---

### 🟠 A-4: `PipelineContext` Missing `SymbolicObserver` Field

**Severity**: HIGH — Observer cannot be called from transforms  

The `PipelineContext` struct (in `types.rs`) does not contain a `SymbolicObserver` field. This means:
- No transform can call `observer.observe_composition()`
- The observer has no integration point in the pipeline
- All observation logic is dead code

**Recommendation**: Add `pub symbolic_observer: SymbolicObserver` to `PipelineContext`.

---

### 🟠 A-5: Confidence Multipliers in Reasoning Rules Not From AdaptiveParams

**Severity**: HIGH — Rules cannot self-calibrate  

All 4 reasoning rules hardcode their confidence multipliers:
- `ProblemSolutionRule`: `0.85`
- `GoalInferenceRule`: `0.80`
- `PolarityConflictRule`: `0.90`
- `ConditionConsequenceRule`: `0.90`

And `apply_confidence_modulation()` hardcodes:
- Connectivity boost: `0.15` (line 246)
- Ambiguity penalty: `0.5` multiplier (line 253)
- Predicate count boost: `0.02` per instance, max `0.10` (lines 258-259)
- Contradiction penalty: `0.85` multiplier (line 265)
- Ambiguity threshold: `0.3` (line 252)
- Connectivity threshold: `0.5` (line 245)

None of these are from `AdaptiveParams`.

**Recommendation**: All magic numbers should be named parameters in `KnowledgeBase.adaptive_params`.

---

## Part III: Structural Gaps

### 🟡 S-1: No `TeachSchemas` Method in `TeachingProtocol`

**Severity**: MEDIUM  

The `TeachingProtocol` can teach markers, morphology, stopwords, and params — but **not schemas**. There's no way for a user to teach AAM a new action schema at runtime.

```rust
// Missing:
pub fn teach_schema(&self, kb: &mut KnowledgeBase, schema: ActionSchema) { ... }
```

**Impact**: Schemas can only come from `bootstrap_schemas()`. Users cannot teach new linguistic constructions.

---

### 🟡 S-2: No `TeachSense` Method in `TeachingProtocol`

**Severity**: MEDIUM  

While `SenseRegistry::add_sense()` exists, there's no `TeachingProtocol::teach_sense()`. More critically, `SenseEntry` lacks `KnowledgeOrigin`, so taught vs. bootstrapped senses are indistinguishable.

**Impact**: No provenance tracking for sense knowledge.

---

### 🟡 S-3: No Observation-to-Schema Pipeline

**Severity**: MEDIUM  

The `SymbolicObserver` can produce `ObservationResult::NewMarker` but cannot produce:
- `NewSchema` — induced from repeated composition patterns
- `NewReasoningRule` — induced from consistent hidden meaning derivations
- `NewSense` — induced from words appearing in mutually exclusive contexts

**Impact**: Observation is limited to markers only. The most powerful learning path (schema induction) doesn't exist.

---

## Part IV: Minor Issues

### 🔵 L-1: Deprecated Methods Create a `create_indonesian_seeded()` on Every Call

Multiple `#[deprecated]` methods create a fresh `KnowledgeBase` on every call:
```rust
#[deprecated(note = "Use matches_with_knowledge() which queries the KnowledgeBase")]
pub fn matches(&self, tokens: &[&str]) -> Option<usize> {
    let kb = crate::v12::knowledge_base::create_indonesian_seeded();
    self.matches_with_knowledge(tokens, &kb)
}
```

This is **O(N) allocation per call** and completely unnecessary. These should be removed in the next breaking release.

### 🔵 L-2: `classify_quality()` Thresholds Not From KB

The quality classification thresholds (`0.70` for HighQuality, `0.45` for ModerateQuality) are hardcoded in `ExtractFrame::classify_quality()`.

### 🔵 L-3: Test Coupling to Bootstrap Data

Many tests depend on `create_indonesian_seeded()` or `SenseRegistry::with_bootstrap_entries()`. If the bootstrap data changes, tests break. Tests should construct their own minimal KB for each scenario.

### 🔵 L-4: `DisambiguationResult::is_resolved()` Hardcodes 0.3 Threshold

```rust
pub fn is_resolved(&self) -> bool {
    self.selected_sense.is_some() && self.confidence > 0.3
}
```

The `0.3` threshold should come from `AdaptiveParams`.

---

## Part V: Hardcode Inventory — Complete Catalog

| ID | File | Line(s) | Type | Data | Count |
|----|------|---------|------|------|-------|
| H-1 | `engine.rs` | 47 | Seed call | `seed_indonesian()` | ~80 entries |
| H-2 | `action_schemas.rs` | 349-429 | Schemas | 5 action schemas | 5 |
| H-3 | `sense_registry.rs` | 138-284 | Senses | 10 homograph entries | 23 senses |
| H-4a | `stemmer.rs` | 36-49 | Roots | `ROOT_EXCEPTIONS` | 49 |
| H-4b | `stemmer.rs` | 52-54 | Prefixes | `PREFIXES_ORDERED` | 14 |
| H-4c | `stemmer.rs` | 57-59 | Suffixes | `SUFFIXES_ORDERED` | 7 |
| H-4d | `stemmer.rs` | 737-743 | Allomorphs | `ME_N_ALLOMORPHS_DATA` | 5 |
| H-4e | `stemmer.rs` | 746-752 | Allomorphs | `PE_N_ALLOMORPHS_DATA` | 5 |
| H-5 | `locale.rs` | 149-245 | Markers/Templates | 9 categories + templates | ~80 strings |
| H-6 | `reason_frame.rs` | 737-748 | Rules | 4 reasoning rules | 4 |
| H-7 | `extract_frame.rs` | 456-460 | Confidence | Antecedent/Consequent bonus | 2 values |

**Total hardcoded linguistic entries**: ~280 individual data points that should be learned, not compiled.

---

## Part VI: Architecture Health Scorecard

| Dimension | Score | Target | Gap |
|-----------|-------|--------|-----|
| Blank-Slate Compliance | 4/10 | 10/10 | 6 — Still auto-seeds Indonesian |
| Emergent Learning (Teach) | 7/10 | 10/10 | 3 — Can't teach schemas/senses |
| Emergent Learning (Ask) | 6/10 | 10/10 | 4 — Usage probes exist but disconnected |
| Emergent Learning (Observe) | 2/10 | 10/10 | 8 — Observer exists but rudimentary + unwired |
| Language-Agnosticism | 3/10 | 10/10 | 7 — Indonesian hardcoded everywhere |
| Self-Calibration | 4/10 | 10/10 | 6 — Many hardcoded multipliers |
| Provenance Tracking | 6/10 | 10/10 | 4 — Senses, schemas, rules lack origin |
| **Overall** | **4.6/10** | **10/10** | **5.4** |

---

## Part VII: Prioritized Remediation Roadmap

### Phase 1: Stop the Bleeding (1-2 days)
**Goal**: Make `PipelineEngine::new()` truly blank-slate

1. **[H-1]** Add `PipelineEngine::new_blank()` and `new_seeded(locale)`  
2. **[H-7]** Move Antecedent/Consequent bonuses to `AdaptiveParams`  
3. **[A-4]** Add `symbolic_observer: SymbolicObserver` to `PipelineContext`  
4. **[L-2/L-4]** Move quality thresholds and `is_resolved()` threshold to `AdaptiveParams`  

### Phase 2: Unify Knowledge Store (2-3 days)
**Goal**: Single source of truth for all linguistic knowledge

5. **[A-2]** Make `Locale` a one-time seed source → `KnowledgeBase`  
6. **[H-3]** Add `KnowledgeOrigin` to `SenseEntry`, migrate to `KnowledgeBase`  
7. **[H-5]** Replace `&'static` Locale returns with `KnowledgeBase` queries  
8. **[S-2]** Add `TeachProtocol::teach_sense()`  

### Phase 3: Kill the Constants (2-3 days)
**Goal**: Zero `const` arrays in the codebase

9. **[H-4]** Move all stemmer data to `KnowledgeBase.morphology_rules`  
10. **[H-4]** Remove `ROOT_EXCEPTIONS`, `PREFIXES_ORDERED`, `SUFFIXES_ORDERED`, `ME_N_ALLOMORPHS_DATA`, `PE_N_ALLOMORPHS_DATA`  
11. **[H-4]** `GraphAwareStemmer` reads ONLY from KB, no fallback to const  

### Phase 4: Schemas as Data (2-3 days)
**Goal**: Schemas learnable, not hardcoded

12. **[H-2]** Replace `bootstrap_schemas()` with `TeachProtocol::teach_schema()`  
13. **[A-3]** Replace `SchemaTrigger` enum variants with `MarkerTrigger(MarkerCategory)`  
14. **[S-1]** Add schema teaching API  
15. **[S-3]** Add `ObservationResult::NewSchema` to `SymbolicObserver`  

### Phase 5: Rules as Data (2-3 days)
**Goal**: Reasoning rules self-calibrate

16. **[H-6]** Read all confidence multipliers from `AdaptiveParams`  
17. **[A-5]** Move all `apply_confidence_modulation()` magic numbers to params  
18. **[H-6]** Add `ReasoningRuleInduction` to `SymbolicObserver`  
19. **[H-6]** Make reasoning rules serializable and storable in graph  

### Phase 6: Wire the Observer (1-2 days)
**Goal**: Observation path actually works

20. **[A-1]** Wire `SymbolicObserver` into `GovernBeliefs` transform  
21. **[A-1]** Implement all 6 pattern types  
22. **[A-1]** Read observation thresholds from `AdaptiveParams`  
23. **[A-1]** Implement Pattern 3 (verb-marking auxiliary) — currently dead code  

---

## Part VIII: The Deeper Question — Is Procedural Code Itself a Hardcode?

The most profound insight from this audit is that the "no-hardcore" principle, taken to its logical conclusion, challenges not just **data** hardcoding but **logic** hardcoding:

| Layer | Current State | No-Hardcore Ideal |
|-------|---------------|-------------------|
| **Data** | Some in KB, some hardcoded | ALL in KB with provenance |
| **Algorithms** | Procedural Rust code | **Learnable strategies** |
| **Control Flow** | Fixed pipeline DAG | **Self-organizing** pipeline |
| **Heuristics** | Hardcoded thresholds | **Self-calibrating** via feedback |

The current architecture addresses Layer 1 (data) partially. Layers 2-4 are entirely procedural. A truly self-adaptive AAM would need:

1. **Strategy objects** instead of hardcoded if-else chains (e.g., `ExtractFrame`'s role extraction should be a composable strategy, not procedural code)
2. **Pipeline self-organization** — transforms could be added/removed based on graph state
3. **Meta-learning** — AAM should learn WHICH strategies work, not just WHAT data they use

This is a **multi-year architectural trajectory**, not a single PR. But the current remediation roadmap (Phases 1-6) gets AAM from 4.6/10 to ~8/10 on the no-hardcore principle, which is a strong foundation for the deeper evolution.

---

*End of Audit Report*
