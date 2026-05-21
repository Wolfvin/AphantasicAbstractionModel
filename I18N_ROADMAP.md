# i18n Roadmap — AAM v12

## Current Status

AAM v12 is **Indonesian-only** by design. All NLP markers, verbalization
templates, and stop words are hardcoded in Bahasa Indonesia. This document
catalogs every hardcoded string and proposes a migration path for
multilingual support.

## Hardcoded String Inventory

### 1. NLP Markers (`extract_frame.rs`)

| Constant | Count | Strings | Purpose |
|----------|-------|---------|---------|
| `NEGATION_MARKERS` | 9 | tidak, bukan, belum, jangan, tak, nggak, enggak, ga, gak | Polarity flip detection |
| `CAUSE_MARKERS` | 2 | karena, sebab | Cause role extraction |
| `PURPOSE_MARKERS` | 3 | untuk, supaya, agar | Purpose role extraction |
| `CONDITION_MARKERS` | 6 | jika, apabila, kalau, bila, jikalau, bilamana | Condition-Consequent rule |
| `VERB_PREFIXES` | 6 | me, ber, di, ter, ke, pe | Predicate detection |
| Copula verbs | ~9 | ada, ialah, adalah, punya, mahu, hendak, boleh, perlu, harus | Equative sentence detection |

**Migration**: Replace with `NlpDictionary` trait — `fn negation_markers(lang: Lang) -> &[&str]`, etc.

### 2. Contradiction Detection (`govern_beliefs.rs`)

| Location | Strings | Purpose |
|----------|---------|---------|
| `has_negation_cause()` | tidak, bukan, tak, jangan, not, no, never, don't, doesn't, didn't | XOR negation for PolarityConflict |
| `hm_problem_contradicts_event()` | tidak, bukan, not, no, never | HM-Event conflict check |
| `hm_problem_negates_event_core()` | tidak, bukan, not | HM core negation check |
| `hm_solution_matches_event_patient_negative()` | tidak, bukan, not, no, never | HM negative polarity check |
| `hm_problem_negates_event_entity()` | tidak, bukan, not, no, never | HM entity negation check |

**Migration**: Reuse the same `NlpDictionary` trait as extract_frame.

### 3. Verbalization Templates (`verbalize.rs`)

| Method | Template | English Equivalent |
|--------|----------|-------------------|
| `insufficient()` | "Tidak ada informasi yang cukup untuk menjelaskan ini." | "Insufficient information to explain this." |
| `verbalize_event()` | ", karena {}", ", untuk {}", ", di {}", ", saat {}", ", dengan {}" | ", because {}", ", in order to {}", ", at {}", ", when {}", ", with {}" |
| `verbalize_hidden_meaning()` | "{} digunakan sebagai solusi untuk {}" | "{} is used as a solution for {}" |
| `verbalize_pattern()` | "Ketika {}, maka {}." | "When {}, then {}." |
| `verbalize_hypothesis()` | "Kemungkinan {} {}." | "Possibly {} {}." |
| `verbalize_situation()` | "Dalam konteks {}, {}{}." | "In the context of {}, {}{}." |
| `verbalize_acquisition()` | "Diketahui bahwa {} {}{}{}." | "It is known that {} {}{}{}." |
| `qualify()` | 8 epistemic qualifiers | See table below |
| Fallbacks | masalah ini, solusi, kondisi ini, hasil ini, konteks ini, ini, sumber, menyatakan, terjadi | this problem, solution, this condition, this result, this context, this, source, states, occurs |

#### Epistemic Qualifiers

| Lifecycle | Epistemic | Indonesian | English |
|-----------|-----------|------------|---------|
| Stable+Grounded (>0.8) | — | "" (no qualifier) | "" (no qualifier) |
| Stable+Grounded (≤0.8) | — | "Tampaknya, " | "Apparently, " |
| Candidate+Inferred | — | "Berdasarkan analisis, " | "Based on analysis, " |
| New+Observed | — | "Berdasarkan observasi, " | "Based on observation, " |
| *+Hypothesis | — | "Kemungkinan besar, " | "Most likely, " |
| *+Contradicted | — | "Meskipun ada kontradiksi, " | "Although contradicted, " |
| Quarantine | — | "Perlu ditinjau kembali, " | "Needs review, " |
| Deprecated | — | "Sebelumnya diyakini, " | "Previously believed, " |
| Default | — | "Kemungkinan, " | "Possibly, " |

**Migration**: Introduce `VerbalizeDictionary` trait with methods for each template.

### 4. Stop Words (`types.rs`)

| Location | Strings |
|----------|---------|
| `extract_keywords()` | yang, dan, di, ke, dari, ini, itu, dengan, untuk, pada, adalah, akan, telah, sebuah, seorang, tidak, bukan, juga, sudah, oleh, karena, supaya, agar, sebab |

**Migration**: `NlpDictionary::stop_words(lang: Lang) -> &[&str]`

## Proposed Architecture

```rust
/// Language identifier for i18n.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    /// Bahasa Indonesia (default)
    Id,
    /// English
    En,
}

/// Dictionary of NLP markers for a given language.
pub trait NlpDictionary: Send + Sync {
    fn lang(&self) -> Lang;
    fn negation_markers(&self) -> &[&str];
    fn cause_markers(&self) -> &[&str];
    fn purpose_markers(&self) -> &[&str];
    fn condition_markers(&self) -> &[&str];
    fn verb_prefixes(&self) -> &[&str];
    fn stop_words(&self) -> &[&str];
    fn copula_verbs(&self) -> &[&str];
}

/// Dictionary of verbalization templates for a given language.
pub trait VerbalizeDictionary: Send + Sync {
    fn lang(&self) -> Lang;
    fn insufficient_info(&self) -> &str;
    fn event_cause_connector(&self) -> &str;  // "karena" / "because"
    fn event_purpose_connector(&self) -> &str; // "untuk" / "in order to"
    // ... etc.
    fn qualify_stable_grounded_low(&self) -> &str;  // "Tampaknya, " / "Apparently, "
    fn qualify_candidate_inferred(&self) -> &str;    // "Berdasarkan analisis, " / "Based on analysis, "
    // ... etc.
}

/// Concrete Indonesian dictionary.
pub struct IndonesianDictionary;
impl NlpDictionary for IndonesianDictionary { /* ... */ }
impl VerbalizeDictionary for IndonesianDictionary { /* ... */ }

/// Concrete English dictionary.
pub struct EnglishDictionary;
impl NlpDictionary for EnglishDictionary { /* ... */ }
impl VerbalizeDictionary for EnglishDictionary { /* ... */ }
```

## Migration Steps

1. **Extract constants into `IndonesianDictionary`** — Zero-behavior-change refactor.
   Move all marker arrays into a struct that implements `NlpDictionary`.
   Replace direct `NEGATION_MARKERS` references with `dict.negation_markers()`.

2. **Extract templates into `IndonesianVerbalizeDictionary`** — Same approach.
   Replace `format!("karena {}", ...)` with `format!("{} {}", dict.event_cause_connector(), ...)`.

3. **Add `EnglishDictionary`** — Implement both traits for English.
   This is the proof that the abstraction works.

4. **Thread `Lang` through `PipelineContext`** — Add `lang: Lang` field.
   Each transform reads `ctx.lang` to get the correct dictionary.

5. **Add `--lang` CLI flag** — Allow users to select language at runtime.

## Why Not Now?

The i18n extraction is a **pure refactor** — it doesn't change any behavior
or fix any bug. The current Indonesian-only design serves its target audience
correctly. Deferring i18n avoids:
- Adding abstraction before the second language is actually needed
- Risking regressions in the carefully-tuned NLP markers
- Proliferating `dyn NlpDictionary` throughout the codebase prematurely

When a second language is needed, this roadmap provides the exact steps.
