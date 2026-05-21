//! # i18n Locale Abstraction
//!
//! Provides a [`Locale`] trait that abstracts language-dependent string resources
//! used by the v12 pipeline. This decouples hardcoded Bahasa Indonesia strings
//! from production logic, enabling future multi-language support.
//!
//! ## Usage
//!
//! ```ignore
//! let locale = IndonesianLocale;
//! let negation = locale.negation_markers();
//! assert!(negation.contains(&"tidak"));
//! ```
//!
//! ## Design Rationale
//!
//! Audit v6 identified 35+ hardcoded Indonesian strings across 4 files without
//! any abstraction. The Locale trait is the minimal viable i18n solution:
//!
//! - **Zero-cost at runtime**: Trait methods return `&'static [&'static str]`
//! - **Extensible**: Add new languages by implementing the trait
//! - **Injectable**: Store in `PipelineContext` for downstream transforms
//!
//! For a full i18n framework (runtime switching, Fluent messages), see the
//! `I18N_ROADMAP.md` document.

use std::fmt;

// ========================================================================
// Locale Trait
// ========================================================================

/// Abstraction for language-dependent string resources.
///
/// Each method returns static string slices, ensuring zero allocation
/// and enabling compile-time verification of language resources.
pub trait Locale: fmt::Debug + Clone + Send + Sync + 'static {
    // ── NLP Markers ──────────────────────────────────────────────

    /// Negation markers that flip polarity from Positive to Negative.
    ///
    /// Used by: `extract_frame`, `govern_beliefs`, `reason_frame`
    fn negation_markers(&self) -> &'static [&'static str];

    /// Short/formal negation markers (subset of `negation_markers`).
    ///
    /// Used for contradiction detection where only the core negation words
    /// are relevant (excluding colloquial forms).
    fn core_negation_markers(&self) -> &'static [&'static str];

    /// Causal conjunction markers.
    ///
    /// Used by: `extract_frame`
    fn cause_markers(&self) -> &'static [&'static str];

    /// Purpose conjunction markers.
    ///
    /// Used by: `extract_frame`
    fn purpose_markers(&self) -> &'static [&'static str];

    /// Conditional conjunction markers.
    ///
    /// Used by: `extract_frame` (ConditionConsequence extraction)
    fn condition_markers(&self) -> &'static [&'static str];

    /// Verb prefixes for predicate detection.
    ///
    /// Used by: `extract_frame`
    fn verb_prefixes(&self) -> &'static [&'static str];

    // ── Verbalization Templates ──────────────────────────────────

    /// Fallback text when insufficient information is available.
    fn insufficient_info_text(&self) -> &'static str;

    /// Default agent filler when no agent is found.
    fn default_agent(&self) -> &'static str;

    /// Default predicate filler when no predicate is found.
    fn default_predicate(&self) -> &'static str;

    /// Cause connector template (e.g., ", because {}").
    fn cause_connector(&self) -> &'static str;

    /// Purpose connector template (e.g., ", for {}").
    fn purpose_connector(&self) -> &'static str;

    /// Location connector template (e.g., ", at {}").
    fn location_connector(&self) -> &'static str;

    /// Time connector template (e.g., ", when {}").
    fn time_connector(&self) -> &'static str;

    /// Instrument connector template (e.g., ", with {}").
    fn instrument_connector(&self) -> &'static str;

    /// Agent "by" connector template (e.g., " by {}").
    fn agent_by_connector(&self) -> &'static str;

    /// Epistemic qualifiers mapped from `EpistemicState`.
    ///
    /// Returns (qualifier_prefix, qualifier_suffix) pairs for each state.
    /// Order must match: Observed, Inferred, Hypothesis, Grounded, Contradicted.
    fn epistemic_qualifiers(&self) -> EpistemicQualifiers;

    // ── Stopwords ────────────────────────────────────────────────

    /// Stopword list for `extract_keywords()`.
    fn stopwords(&self) -> &'static [&'static str];

    // ── Metadata ─────────────────────────────────────────────────

    /// ISO 639-1 language code (e.g., "id" for Indonesian, "en" for English).
    fn language_code(&self) -> &'static str;

    /// Human-readable language name.
    fn language_name(&self) -> &'static str;
}

// ========================================================================
// Epistemic Qualifiers
// ========================================================================

/// Epistemic qualifier strings for verbalization.
///
/// Each field is a prefix that indicates the epistemic status of a composition
/// in the verbalization output.
#[derive(Debug, Clone)]
pub struct EpistemicQualifiers {
    pub observed: &'static str,
    pub inferred: &'static str,
    pub hypothesis: &'static str,
    pub grounded: &'static str,
    pub contradicted: &'static str,
    pub default: &'static str,
}

// ========================================================================
// Indonesian Locale
// ========================================================================

/// Indonesian (Bahasa Indonesia) locale implementation.
///
/// This is the primary locale for the AAM system, as the project
/// explicitly prioritizes Bahasa Indonesia for development and testing.
#[derive(Debug, Clone, Default)]
pub struct IndonesianLocale;

impl Locale for IndonesianLocale {
    fn negation_markers(&self) -> &'static [&'static str] {
        &[
            "tidak",   // not (general negation)
            "bukan",   // not (identity negation)
            "belum",   // not yet
            "jangan",  // don't (prohibitive)
            "tak",     // not (short form)
            "nggak",   // not (colloquial)
            "enggak",  // not (colloquial variant)
            "ga",      // not (very colloquial)
            "gak",     // not (very colloquial variant)
        ]
    }

    fn core_negation_markers(&self) -> &'static [&'static str] {
        &["tidak", "bukan", "tak", "jangan"]
    }

    fn cause_markers(&self) -> &'static [&'static str] {
        &["karena", "sebab"]
    }

    fn purpose_markers(&self) -> &'static [&'static str] {
        &["untuk", "supaya", "agar"]
    }

    fn condition_markers(&self) -> &'static [&'static str] {
        &["jika", "apabila", "kalau", "bila", "jikalau", "bilamana"]
    }

    fn verb_prefixes(&self) -> &'static [&'static str] {
        &["me", "ber", "di", "ter", "ke", "pe"]
    }

    fn insufficient_info_text(&self) -> &'static str {
        "Tidak ada informasi yang cukup untuk menjelaskan ini."
    }

    fn default_agent(&self) -> &'static str {
        "Sesuatu"
    }

    fn default_predicate(&self) -> &'static str {
        "terjadi"
    }

    fn cause_connector(&self) -> &'static str {
        ", karena {}"
    }

    fn purpose_connector(&self) -> &'static str {
        ", untuk {}"
    }

    fn location_connector(&self) -> &'static str {
        ", di {}"
    }

    fn time_connector(&self) -> &'static str {
        ", saat {}"
    }

    fn instrument_connector(&self) -> &'static str {
        ", dengan {}"
    }

    fn agent_by_connector(&self) -> &'static str {
        " oleh {}"
    }

    fn epistemic_qualifiers(&self) -> EpistemicQualifiers {
        EpistemicQualifiers {
            observed: "",
            inferred: "Tampaknya, ",
            hypothesis: "Kemungkinan besar, ",
            grounded: "Berdasarkan observasi, ",
            contradicted: "Meskipun ada kontradiksi, ",
            default: "Kemungkinan, ",
        }
    }

    fn stopwords(&self) -> &'static [&'static str] {
        &["tidak", "bukan", "karena", "sebab", "untuk", "supaya", "agar",
           "jika", "kalau", "apabila", "bila", "dengan", "di", "ke", "dari",
           "yang", "ini", "itu", "adalah", "akan", "telah", "sudah", "belum",
           "jangan", "tak", "ada", "ia", "mereka", "kita", "kami", "anda"]
    }

    fn language_code(&self) -> &'static str {
        "id"
    }

    fn language_name(&self) -> &'static str {
        "Bahasa Indonesia"
    }
}

// ========================================================================
// English Locale
// ========================================================================

/// English locale implementation.
///
/// Provides English equivalents for all language-dependent resources.
/// Useful for testing and for English-language deployments.
#[derive(Debug, Clone, Default)]
pub struct EnglishLocale;

impl Locale for EnglishLocale {
    fn negation_markers(&self) -> &'static [&'static str] {
        &["not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't", "isn't", "aren't"]
    }

    fn core_negation_markers(&self) -> &'static [&'static str] {
        &["not", "no", "never", "don't"]
    }

    fn cause_markers(&self) -> &'static [&'static str] {
        &["because", "since", "as"]
    }

    fn purpose_markers(&self) -> &'static [&'static str] {
        &["to", "in order to", "so that", "for"]
    }

    fn condition_markers(&self) -> &'static [&'static str] {
        &["if", "when", "whenever", "unless", "provided that"]
    }

    fn verb_prefixes(&self) -> &'static [&'static str] {
        &[] // English doesn't use verb prefixes for predicate detection
    }

    fn insufficient_info_text(&self) -> &'static str {
        "There is not enough information to explain this."
    }

    fn default_agent(&self) -> &'static str {
        "Something"
    }

    fn default_predicate(&self) -> &'static str {
        "happened"
    }

    fn cause_connector(&self) -> &'static str {
        ", because {}"
    }

    fn purpose_connector(&self) -> &'static str {
        ", in order to {}"
    }

    fn location_connector(&self) -> &'static str {
        ", at {}"
    }

    fn time_connector(&self) -> &'static str {
        ", when {}"
    }

    fn instrument_connector(&self) -> &'static str {
        ", with {}"
    }

    fn agent_by_connector(&self) -> &'static str {
        " by {}"
    }

    fn epistemic_qualifiers(&self) -> EpistemicQualifiers {
        EpistemicQualifiers {
            observed: "",
            inferred: "Apparently, ",
            hypothesis: "Most likely, ",
            grounded: "Based on observation, ",
            contradicted: "Despite contradiction, ",
            default: "Possibly, ",
        }
    }

    fn stopwords(&self) -> &'static [&'static str] {
        &["not", "no", "never", "because", "since", "for", "to", "if", "when",
           "with", "at", "in", "on", "from", "the", "a", "an", "is", "are",
           "was", "were", "be", "been", "being", "have", "has", "had", "do",
           "does", "did", "will", "would", "could", "should", "may", "might"]
    }

    fn language_code(&self) -> &'static str {
        "en"
    }

    fn language_name(&self) -> &'static str {
        "English"
    }
}

// ========================================================================
// Default Locale Helper
// ========================================================================

/// Returns the default locale for the system.
///
/// Currently returns [`IndonesianLocale`] as the system is optimized
/// for Bahasa Indonesia. This will be configurable via `PipelineContext`
/// in a future release.
pub fn default_locale() -> IndonesianLocale {
    IndonesianLocale
}
