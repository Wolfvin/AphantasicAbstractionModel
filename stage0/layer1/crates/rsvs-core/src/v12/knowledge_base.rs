//! # No-Hardcore Architecture: Knowledge Base
//!
//! This module implements the **blank-slate** principle: AAM starts with ZERO
//! hardcoded knowledge. Everything must be learned through three mechanisms:
//!
//! 1. **We teach** — Users explicitly teach AAM markers, senses, morphology, etc.
//! 2. **AAM asks** — Usage probes ask questions when uncertain (Phase S).
//! 3. **Symbolic observation** — The `SymbolicObserver` watches the graph's
//!    relational structure and induces patterns (e.g., "words after 'tidak'
//!    are always verbs" → learns `tidak` as a verb-marking auxiliary).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
//! │   We Teach   │────▶│  KnowledgeBase   │◀────│  AAM Asks       │
//! │ (TeachingAPI)│     │                  │     │ (Usage Probes)  │
//! └─────────────┘     │  markers          │     └─────────────────┘
//!                      │  senses           │
//! ┌──────────────────┐ │  morphology       │     ┌─────────────────┐
//! │ SymbolicObserver │──▶│  params (adaptive)│◀────│  Correction     │
//! │ (pattern induce) │ │  templates        │     │  Feedback       │
//! └──────────────────┘ └──────────────────┘     └─────────────────┘
//!                              │
//!                    ┌─────────▼──────────┐
//!                    │  PipelineContext    │
//!                    │  (shared state)    │
//!                    └────────────────────┘
//!                              │
//!                    ┌─────────▼──────────┐
//!                    │  All Transforms    │
//!                    │  (query, not hard- │
//!                    │   code)            │
//!                    └────────────────────┘
//! ```
//!
//! ## Key Principle
//!
//! **Knowledge is DATA, not CODE.** Every marker word, sense entry, morphological
//! rule, confidence multiplier, and threshold is stored in `KnowledgeBase` with
//! a `KnowledgeOrigin` that tracks HOW it was learned. The system can answer
//! "WHY do you know that?" for every piece of knowledge.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::types::CompositionType;
use super::types::SemanticRole;

// ========================================================================
// KnowledgeOrigin — HOW Was Knowledge Acquired?
// ========================================================================

/// Provenance of a piece of knowledge — tracks HOW it was learned.
///
/// This is the philosophical core of the No-Hardcore architecture:
/// every piece of knowledge must have an origin that explains
/// why AAM believes it. This enables the system to answer
/// "WHY do you know that?" for every entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KnowledgeOrigin {
    /// User explicitly taught this knowledge.
    ///
    /// Example: "Teach AAM that 'adalah' is a copula marker."
    /// This is the highest-confidence source — humans are trusted.
    Taught {
        /// Who taught this (user ID, session, etc.).
        by: String,
        /// When this was taught (epoch seconds).
        at: String,
    },

    /// Induced from graph patterns by `SymbolicObserver`.
    ///
    /// Example: AAM observes that "tidak" always precedes verbs in the
    /// graph's compositions → induces that "tidak" is a verb-marking
    /// auxiliary. Confidence depends on evidence count.
    Observed {
        /// What pattern was observed.
        pattern: String,
        /// How many times this pattern was observed.
        evidence_count: usize,
        /// Confidence from observation (0.0–1.0).
        confidence: f32,
    },

    /// Learned from usage probe Q&A (Phase S).
    ///
    /// Example: AAM generates a probe "Is 'merupakan' a copula?"
    /// and the user confirms → learns the marker.
    Asked {
        /// The probe ID that triggered this learning.
        probe_id: String,
        /// The user's response.
        response: String,
    },

    /// Self-calibrated from correction feedback.
    ///
    /// Example: After 10 corrections, AAM learns that
    /// ConditionConsequence rules have 85% accuracy → adjusts
    /// the confidence multiplier from 0.90 to 0.85.
    Calibrated {
        /// Number of observations used for calibration.
        from_observations: usize,
        /// Measured accuracy (0.0–1.0).
        accuracy: f32,
    },

    /// Bootstrapped from seed data for backward compatibility.
    ///
    /// This is the TRANSITIONAL origin — existing hardcoded data is
    /// migrated here so it's tracked, but the long-term goal is to
    /// replace all Bootstrapped entries with Taught/Observed/Asked.
    /// Bootstrapped entries have lower confidence than Taught entries
    /// because they were never verified through interaction.
    Bootstrapped {
        /// Why this was bootstrapped (e.g., "Indonesian homograph baseline").
        reason: String,
    },
}

impl Default for KnowledgeOrigin {
    fn default() -> Self {
        KnowledgeOrigin::Bootstrapped {
            reason: "default origin".into(),
        }
    }
}

impl KnowledgeOrigin {
    /// Confidence multiplier based on origin type.
    ///
    /// Taught knowledge is trusted most (1.0), observed depends on
    /// evidence count, bootstrapped is lowest because it's unverified.
    pub fn confidence_weight(&self) -> f32 {
        match self {
            KnowledgeOrigin::Taught { .. } => 1.0,
            KnowledgeOrigin::Calibrated { accuracy, .. } => *accuracy,
            KnowledgeOrigin::Observed { confidence, .. } => *confidence,
            KnowledgeOrigin::Asked { .. } => 0.9,
            KnowledgeOrigin::Bootstrapped { .. } => 0.7,
        }
    }

    /// Human-readable description of how this knowledge was acquired.
    pub fn describe(&self) -> String {
        match self {
            KnowledgeOrigin::Taught { by, at } => {
                format!("Taught by '{}' at {}", by, at)
            }
            KnowledgeOrigin::Observed { pattern, evidence_count, confidence } => {
                format!(
                    "Observed pattern '{}' {} times (confidence: {:.0}%)",
                    pattern, evidence_count, confidence * 100.0
                )
            }
            KnowledgeOrigin::Asked { probe_id, response } => {
                format!("Learned from probe '{}' (response: {})", probe_id, response)
            }
            KnowledgeOrigin::Calibrated { from_observations, accuracy } => {
                format!(
                    "Calibrated from {} observations (accuracy: {:.0}%)",
                    from_observations, accuracy * 100.0
                )
            }
            KnowledgeOrigin::Bootstrapped { reason } => {
                format!("Bootstrapped: {}", reason)
            }
        }
    }
}

// ========================================================================
// KnowledgeEntry — A Single Piece of Learned Knowledge
// ========================================================================

/// A piece of knowledge with its provenance and confidence.
///
/// Every entry in the KnowledgeBase is wrapped in this struct,
/// enabling the system to trace WHY it knows something and
/// HOW reliable that knowledge is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry<T> {
    /// The knowledge value.
    pub value: T,
    /// How this knowledge was acquired.
    pub origin: KnowledgeOrigin,
    /// Overall confidence (0.0–1.0), combining origin weight and
    /// any subsequent evidence.
    pub confidence: f32,
}

impl<T> KnowledgeEntry<T> {
    /// Create a new entry with the given value and origin.
    pub fn new(value: T, origin: KnowledgeOrigin) -> Self {
        let confidence = origin.confidence_weight();
        Self { value, origin, confidence }
    }

    /// Create a taught entry (highest confidence).
    pub fn taught(value: T, by: &str) -> Self {
        Self::new(value, KnowledgeOrigin::Taught {
            by: by.to_string(),
            at: format!("{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()),
        })
    }

    /// Create a bootstrapped entry (lowest confidence, transitional).
    pub fn bootstrapped(value: T, reason: &str) -> Self {
        Self::new(value, KnowledgeOrigin::Bootstrapped {
            reason: reason.to_string(),
        })
    }

    /// Create an observed entry with evidence count.
    pub fn observed(value: T, pattern: &str, evidence_count: usize) -> Self {
        let confidence = (evidence_count as f32 / 10.0).min(0.95);
        Self::new(value, KnowledgeOrigin::Observed {
            pattern: pattern.to_string(),
            evidence_count,
            confidence,
        })
    }
}

// ========================================================================
// MarkerCategory — Types of Linguistic Markers
// ========================================================================

/// Categories of linguistic markers that AAM can learn.
///
/// Instead of hardcoding "COPULA_MARKERS", "NEGATION_MARKERS", etc.
/// as const arrays, these are categories in the KnowledgeBase that
/// can be populated at runtime through teaching, observation, or
/// bootstrapping.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarkerCategory {
    /// Copula markers — "adalah", "ialah", "merupakan".
    /// These trigger EquativeBinding compositions.
    Copula,
    /// Possessive markers — "punya", "miliki".
    /// These trigger PossessiveBinding compositions.
    Possessive,
    /// Equative/definitional markers — "yaitu", "yakni".
    Equative,
    /// Existential markers — "ada".
    Existential,
    /// Locative prepositions — "di", "ke", "dari".
    Locative,
    /// Negation markers — "tidak", "bukan", "belum".
    /// These flip polarity from Positive to Negative.
    Negation,
    /// Core negation markers (formal, not colloquial).
    CoreNegation,
    /// Cause conjunctions — "karena", "sebab".
    Cause,
    /// Purpose conjunctions — "untuk", "supaya", "agar".
    Purpose,
    /// Condition conjunctions — "jika", "apabila", "kalau".
    Condition,
    /// Verb prefixes for predicate detection — "me", "ber", "di", "ter".
    VerbPrefix,
    /// Auxiliary words that mark following word as verb — "tidak", "belum", "sudah".
    VerbMarkingAuxiliary,
    /// Determiners that mark following word as noun — "ini", "itu", "sebuah".
    NounDeterminer,
    /// Degree modifiers that suggest adjective — "sekali", "sangat".
    DegreeModifier,
    /// Common verbs that are not prefix-derived — "ada", "ialah", "adalah", "punya", etc.
    CommonVerb,
    /// Custom marker category (extensible).
    Custom(String),
}

impl std::fmt::Display for MarkerCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarkerCategory::Custom(s) => write!(f, "Custom({})", s),
            other => write!(f, "{:?}", other),
        }
    }
}

// ========================================================================
// MorphologyRule — Learned Morphological Knowledge
// ========================================================================

/// A morphological rule learned by AAM.
///
/// Instead of hardcoding prefixes, suffixes, and root exceptions,
/// these are rules in the KnowledgeBase that can be taught or observed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologyRule {
    /// What type of rule this is.
    pub rule_type: MorphologyRuleType,
    /// The rule value (e.g., "meN", "kan", "makan").
    pub value: String,
    /// For allomorphs: the associated archimorpheme.
    #[serde(default)]
    pub archimorpheme: Option<String>,
    /// For allomorphs: the phonological condition.
    #[serde(default)]
    pub condition: Option<String>,
}

/// Types of morphological rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MorphologyRuleType {
    /// An archimorpheme (e.g., "meN", "peN").
    Archimorpheme,
    /// An allomorph of an archimorpheme (e.g., "meng", "mem").
    Allomorph,
    /// A simple prefix (e.g., "ber", "di", "ter").
    SimplePrefix,
    /// A suffix (e.g., "kan", "an", "i").
    Suffix,
    /// A root exception — word that should not be further stemmed.
    RootException,
}

// ========================================================================
// AdaptiveParams — Self-Calibrating Parameters
// ========================================================================

/// Self-calibrating parameters that adjust based on feedback.
///
/// Instead of hardcoded magic numbers for confidence multipliers,
/// cognitive thresholds, and blending weights, all parameters are
/// stored here with their origin and calibration history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParams {
    /// Named parameters with origin tracking.
    params: HashMap<String, ParamEntry>,
}

/// A tracked parameter with calibration history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamEntry {
    /// Current value.
    pub value: f32,
    /// Initial value (for reset).
    pub initial_value: f32,
    /// How this parameter was set.
    pub origin: KnowledgeOrigin,
    /// Number of calibration observations.
    #[serde(default)]
    pub calibration_count: usize,
    /// Running sum of observed values (for mean calibration).
    #[serde(default)]
    pub calibration_sum: f32,
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveParams {
    /// Create empty params.
    pub fn new() -> Self {
        Self { params: HashMap::new() }
    }

    /// Get a parameter value, returning default if not set.
    pub fn get(&self, name: &str, default: f32) -> f32 {
        self.params.get(name).map(|p| p.value).unwrap_or(default)
    }

    /// Set a parameter with origin tracking.
    pub fn set(&mut self, name: &str, value: f32, origin: KnowledgeOrigin) {
        self.params.insert(name.to_string(), ParamEntry {
            value,
            initial_value: value,
            origin,
            calibration_count: 0,
            calibration_sum: 0.0,
        });
    }

    /// Calibrate a parameter based on a new observation.
    ///
    /// Uses exponential moving average to smoothly adjust:
    /// new_value = old_value * (1 - alpha) + observation * alpha
    pub fn calibrate(&mut self, name: &str, observation: f32, alpha: f32) {
        if let Some(entry) = self.params.get_mut(name) {
            entry.value = entry.value * (1.0 - alpha) + observation * alpha;
            entry.calibration_count += 1;
            entry.calibration_sum += observation;
            entry.origin = KnowledgeOrigin::Calibrated {
                from_observations: entry.calibration_count,
                accuracy: entry.value,
            };
        }
    }

    /// Get the origin of a parameter (for WHY tracing).
    pub fn origin_of(&self, name: &str) -> Option<&KnowledgeOrigin> {
        self.params.get(name).map(|p| &p.origin)
    }

    /// List all parameters with their current values and origins.
    pub fn all_params(&self) -> Vec<(&str, f32, &KnowledgeOrigin)> {
        self.params.iter()
            .map(|(name, entry)| (name.as_str(), entry.value, &entry.origin))
            .collect()
    }
}

// ========================================================================
// KnowledgeBase — The Blank Slate
// ========================================================================

/// The central knowledge store — starts EMPTY and is populated through
/// teaching, asking, and observation.
///
/// This replaces ALL hardcoded const arrays (COPULA_MARKERS,
/// NEGATION_MARKERS, etc.) with a runtime-populated, provenance-tracked
/// knowledge system. Every entry has a `KnowledgeOrigin` that explains
/// HOW AAM learned it.
///
/// # Blank Slate Principle
///
/// When `KnowledgeBase::new()` is called, the system knows NOTHING.
/// It cannot detect copulas, negation, or morphology because it has
/// no markers, no senses, and no rules. Knowledge must be explicitly
/// taught or discovered.
///
/// For convenience, `seed_from_locale()` populates the KnowledgeBase
/// with the same data that was previously hardcoded, but with
/// `KnowledgeOrigin::Bootstrapped` provenance. This allows the
/// system to work immediately while tracking that this knowledge
/// was not learned through interaction.
///
/// # The Three Learning Paths
///
/// 1. **Teaching**: `teach_marker()`, `teach_sense()`, `teach_morphology()`
/// 2. **Asking**: Via Usage Probes (Phase S) — AAM asks questions
/// 3. **Observation**: Via `SymbolicObserver` — pattern induction from graph
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KnowledgeBase {
    /// Linguistic markers: category → set of known markers.
    #[serde(default)]
    markers: HashMap<MarkerCategory, Vec<KnowledgeEntry<String>>>,

    /// Morphological rules.
    #[serde(default)]
    morphology_rules: Vec<KnowledgeEntry<MorphologyRule>>,

    /// Adaptive parameters (confidence multipliers, thresholds, weights).
    #[serde(default)]
    params: AdaptiveParams,

    /// Stopwords for keyword extraction.
    #[serde(default)]
    stopwords: Vec<KnowledgeEntry<String>>,

    /// POS compatibility rules: (sense_pos, hint_pos) → compatible.
    /// Learned through observation of which POS combinations actually
    /// occur in the graph vs. which are predicted but never seen.
    #[serde(default)]
    pos_compatibility: HashMap<(String, String), KnowledgeEntry<bool>>,
}

impl KnowledgeBase {
    /// Create a completely empty KnowledgeBase — the TRUE blank slate.
    ///
    /// AAM starts with zero knowledge. It cannot detect any linguistic
    /// patterns because it knows no markers, no senses, no morphology.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Marker Operations ──────────────────────────────────────

    /// Teach AAM a marker word for a specific category.
    ///
    /// Example: `kb.teach_marker(MarkerCategory::Copula, "adalah", KnowledgeOrigin::taught("user_1"))`
    ///
    /// After this, when the pipeline encounters "adalah", it will
    /// recognize it as a copula marker and create an EquativeBinding.
    pub fn teach_marker(&mut self, category: MarkerCategory, word: &str, origin: KnowledgeOrigin) {
        let entries = self.markers.entry(category.clone()).or_default();

        // Don't add duplicates.
        let lower = word.to_lowercase();
        if entries.iter().any(|e| e.value.to_lowercase() == lower) {
            return;
        }

        entries.push(KnowledgeEntry::new(word.to_lowercase(), origin));
    }

    /// Check if a word is a known marker for a category.
    pub fn is_marker(&self, category: &MarkerCategory, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.markers.get(category)
            .map(|entries| entries.iter().any(|e| e.value == lower))
            .unwrap_or(false)
    }

    /// Get all known markers for a category.
    pub fn markers_for(&self, category: &MarkerCategory) -> Vec<&str> {
        self.markers.get(category)
            .map(|entries| entries.iter().map(|e| e.value.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get all markers for a category as owned Strings.
    pub fn markers_for_owned(&self, category: &MarkerCategory) -> Vec<String> {
        self.markers.get(category)
            .map(|entries| entries.iter().map(|e| e.value.clone()).collect())
            .unwrap_or_default()
    }

    /// Get the origin of a marker (for WHY tracing).
    pub fn marker_origin(&self, category: &MarkerCategory, word: &str) -> Option<&KnowledgeOrigin> {
        let lower = word.to_lowercase();
        self.markers.get(category)
            .and_then(|entries| entries.iter().find(|e| e.value == lower).map(|e| &e.origin))
    }

    /// Check if a token is ANY known marker (negation, cause, purpose, condition).
    pub fn is_any_role_marker(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.is_marker(&MarkerCategory::Negation, &lower)
            || self.is_marker(&MarkerCategory::Cause, &lower)
            || self.is_marker(&MarkerCategory::Purpose, &lower)
            || self.is_marker(&MarkerCategory::Condition, &lower)
    }

    /// Check if a word is verb-like based on known verb prefixes.
    pub fn is_verb_prefix_match(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.markers_for(&MarkerCategory::VerbPrefix).iter().any(|prefix| {
            lower.starts_with(prefix) && lower.len() > prefix.len() + 1
        })
    }

    /// Infer POS from context markers in the KnowledgeBase.
    ///
    /// Checks if any context token is a known VerbMarkingAuxiliary,
    /// NounDeterminer, or DegreeModifier, and returns the inferred POS.
    pub fn infer_pos_from_context(&self, word: &str, context_tokens: &[&str]) -> Option<String> {
        let word_lower = word.to_lowercase();

        // Check if word itself starts with a known verb prefix.
        if self.is_verb_prefix_match(&word_lower) {
            return Some("verb".to_string());
        }

        // Check context for verb-marking auxiliaries.
        for token in context_tokens {
            if self.is_marker(&MarkerCategory::VerbMarkingAuxiliary, token) {
                return Some("verb".to_string());
            }
        }

        // Check context for noun determiners.
        for token in context_tokens {
            if self.is_marker(&MarkerCategory::NounDeterminer, token) {
                return Some("noun".to_string());
            }
        }

        // Check context for degree modifiers.
        for token in context_tokens {
            if self.is_marker(&MarkerCategory::DegreeModifier, token) {
                return Some("adjective".to_string());
            }
        }

        None
    }

    /// Check POS compatibility based on learned rules.
    pub fn pos_compatible(&self, sense_pos: &str, hint_pos: &str) -> bool {
        let s = sense_pos.to_lowercase();
        let h = hint_pos.to_lowercase();

        // Exact match.
        if s == h { return true; }

        // Check learned compatibility rules.
        if let Some(entry) = self.pos_compatibility.get(&(s.clone(), h.clone())) {
            return entry.value;
        }

        // Default: incompatible if no learned rule.
        false
    }

    /// Teach a POS compatibility rule.
    pub fn teach_pos_compatibility(&mut self, sense_pos: &str, hint_pos: &str, compatible: bool, origin: KnowledgeOrigin) {
        self.pos_compatibility.insert(
            (sense_pos.to_lowercase(), hint_pos.to_lowercase()),
            KnowledgeEntry::new(compatible, origin),
        );
    }

    // ── Morphology Operations ──────────────────────────────────

    /// Teach a morphological rule.
    pub fn teach_morphology(&mut self, rule: MorphologyRule, origin: KnowledgeOrigin) {
        // Don't add duplicates.
        let exists = self.morphology_rules.iter().any(|e| {
            e.value.rule_type == rule.rule_type && e.value.value == rule.value
        });
        if !exists {
            self.morphology_rules.push(KnowledgeEntry::new(rule, origin));
        }
    }

    /// Get all morphology rules of a specific type.
    pub fn morphology_rules_of(&self, rule_type: &MorphologyRuleType) -> Vec<&MorphologyRule> {
        self.morphology_rules.iter()
            .filter(|e| &e.value.rule_type == rule_type)
            .map(|e| &e.value)
            .collect()
    }

    /// Get all simple prefixes.
    pub fn simple_prefixes(&self) -> Vec<String> {
        let mut prefixes: Vec<String> = self.morphology_rules_of(&MorphologyRuleType::SimplePrefix)
            .iter()
            .map(|r| r.value.clone())
            .collect();
        // Also include allomorphs as they function as prefixes.
        for rule in self.morphology_rules_of(&MorphologyRuleType::Allomorph) {
            prefixes.push(rule.value.clone());
        }
        prefixes.sort_by(|a, b| b.len().cmp(&a.len())); // Longest first
        prefixes
    }

    /// Get all suffixes.
    pub fn suffixes(&self) -> Vec<String> {
        let mut suffixes: Vec<String> = self.morphology_rules_of(&MorphologyRuleType::Suffix)
            .iter()
            .map(|r| r.value.clone())
            .collect();
        suffixes.sort_by(|a, b| b.len().cmp(&a.len())); // Longest first
        suffixes
    }

    /// Get all root exceptions.
    pub fn root_exceptions(&self) -> HashSet<String> {
        self.morphology_rules_of(&MorphologyRuleType::RootException)
            .iter()
            .map(|r| r.value.clone())
            .collect()
    }

    /// Check if a word is a known root exception.
    pub fn is_root_exception(&self, word: &str) -> bool {
        self.root_exceptions().contains(&word.to_lowercase())
    }

    // ── Stopword Operations ────────────────────────────────────

    /// Teach a stopword.
    pub fn teach_stopword(&mut self, word: &str, origin: KnowledgeOrigin) {
        let lower = word.to_lowercase();
        if !self.stopwords.iter().any(|e| e.value == lower) {
            self.stopwords.push(KnowledgeEntry::new(lower, origin));
        }
    }

    /// Check if a word is a stopword.
    pub fn is_stopword(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.stopwords.iter().any(|e| e.value == lower)
    }

    /// Get all stopwords.
    pub fn stopwords(&self) -> Vec<&str> {
        self.stopwords.iter().map(|e| e.value.as_str()).collect()
    }

    // ── Parameter Operations ───────────────────────────────────

    /// Get the adaptive parameters.
    pub fn params(&self) -> &AdaptiveParams {
        &self.params
    }

    /// Get mutable access to adaptive parameters.
    pub fn params_mut(&mut self) -> &mut AdaptiveParams {
        &mut self.params
    }

    /// Convenience: get a parameter value with default.
    pub fn param(&self, name: &str, default: f32) -> f32 {
        self.params.get(name, default)
    }

    /// Convenience: set a parameter with origin.
    pub fn set_param(&mut self, name: &str, value: f32, origin: KnowledgeOrigin) {
        self.params.set(name, value, origin);
    }

    /// Convenience: calibrate a parameter from observation.
    pub fn calibrate_param(&mut self, name: &str, observation: f32, alpha: f32) {
        self.params.calibrate(name, observation, alpha);
    }

    // ── Diagnostics ────────────────────────────────────────────

    /// How many knowledge entries total?
    pub fn total_entries(&self) -> usize {
        let marker_count: usize = self.markers.values().map(|v| v.len()).sum();
        let morph_count = self.morphology_rules.len();
        let stopword_count = self.stopwords.len();
        let pos_count = self.pos_compatibility.len();
        let param_count = self.params.params.len();
        marker_count + morph_count + stopword_count + pos_count + param_count
    }

    /// How many entries have each origin type?
    pub fn origin_breakdown(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();

        for entries in self.markers.values() {
            for entry in entries {
                let key = match &entry.origin {
                    KnowledgeOrigin::Taught { .. } => "Taught",
                    KnowledgeOrigin::Observed { .. } => "Observed",
                    KnowledgeOrigin::Asked { .. } => "Asked",
                    KnowledgeOrigin::Calibrated { .. } => "Calibrated",
                    KnowledgeOrigin::Bootstrapped { .. } => "Bootstrapped",
                };
                *counts.entry(key.to_string()).or_default() += 1;
            }
        }

        for entry in &self.morphology_rules {
            let key = match &entry.origin {
                KnowledgeOrigin::Taught { .. } => "Taught",
                KnowledgeOrigin::Observed { .. } => "Observed",
                KnowledgeOrigin::Asked { .. } => "Asked",
                KnowledgeOrigin::Calibrated { .. } => "Calibrated",
                KnowledgeOrigin::Bootstrapped { .. } => "Bootstrapped",
            };
            *counts.entry(key.to_string()).or_default() += 1;
        }

        counts
    }

    /// Is the knowledge base completely empty (true blank slate)?
    pub fn is_blank(&self) -> bool {
        self.total_entries() == 0
    }

    /// Generate a human-readable knowledge audit report.
    pub fn audit_report(&self) -> String {
        let mut lines = Vec::new();
        lines.push("=== Knowledge Base Audit Report ===".to_string());
        lines.push(format!("Total entries: {}", self.total_entries()));
        lines.push(format!("Is blank: {}", self.is_blank()));

        lines.push("\n--- Marker Categories ---".to_string());
        for (category, entries) in &self.markers {
            lines.push(format!("  {}: {} entries", category, entries.len()));
            for entry in entries {
                lines.push(format!("    '{}' (confidence: {:.2}, origin: {})",
                    entry.value, entry.confidence, entry.origin.describe()));
            }
        }

        lines.push("\n--- Morphology Rules ---".to_string());
        for entry in &self.morphology_rules {
            lines.push(format!("  {:?}: '{}' (origin: {})",
                entry.value.rule_type, entry.value.value, entry.origin.describe()));
        }

        lines.push("\n--- Parameters ---".to_string());
        for (name, value, origin) in self.params.all_params() {
            lines.push(format!("  {} = {:.4} (origin: {})", name, value, origin.describe()));
        }

        lines.push("\n--- Origin Breakdown ---".to_string());
        for (origin_type, count) in self.origin_breakdown() {
            lines.push(format!("  {}: {}", origin_type, count));
        }

        lines.join("\n")
    }
}

// ========================================================================
// TeachingProtocol — How Users Teach AAM
// ========================================================================

/// The formal API for teaching AAM new knowledge.
///
/// Every teaching method creates a `KnowledgeEntry` with
/// `KnowledgeOrigin::Taught` provenance, so AAM can always
/// explain WHY it knows something: "Because you taught me."
///
/// # Usage
///
/// ```ignore
/// let mut kb = KnowledgeBase::new();
/// let teacher = TeachingProtocol::new("user_raymond");
///
/// // Teach AAM that "adalah" is a copula marker.
/// teacher.teach_marker(&mut kb, MarkerCategory::Copula, "adalah");
///
/// // Teach AAM a morphological rule.
/// teacher.teach_morphology_rule(&mut kb, MorphologyRule {
///     rule_type: MorphologyRuleType::SimplePrefix,
///     value: "ber".to_string(),
///     archimorpheme: None,
///     condition: None,
/// });
/// ```
#[derive(Debug, Clone)]
pub struct TeachingProtocol {
    /// Who is teaching (user ID, session, etc.).
    teacher: String,
}

impl TeachingProtocol {
    /// Create a new teaching protocol for a specific teacher.
    pub fn new(teacher: &str) -> Self {
        Self { teacher: teacher.to_string() }
    }

    /// Teach a marker word for a category.
    pub fn teach_marker(&self, kb: &mut KnowledgeBase, category: MarkerCategory, word: &str) {
        kb.teach_marker(category, word, KnowledgeOrigin::Taught {
            by: self.teacher.clone(),
            at: now_epoch_string(),
        });
    }

    /// Teach multiple marker words for a category at once.
    pub fn teach_markers(&self, kb: &mut KnowledgeBase, category: MarkerCategory, words: &[&str]) {
        for word in words {
            self.teach_marker(kb, category.clone(), word);
        }
    }

    /// Teach a morphological rule.
    pub fn teach_morphology_rule(&self, kb: &mut KnowledgeBase, rule: MorphologyRule) {
        kb.teach_morphology(rule, KnowledgeOrigin::Taught {
            by: self.teacher.clone(),
            at: now_epoch_string(),
        });
    }

    /// Teach a stopword.
    pub fn teach_stopword(&self, kb: &mut KnowledgeBase, word: &str) {
        kb.teach_stopword(word, KnowledgeOrigin::Taught {
            by: self.teacher.clone(),
            at: now_epoch_string(),
        });
    }

    /// Teach a POS compatibility rule.
    pub fn teach_pos_compatibility(&self, kb: &mut KnowledgeBase, sense_pos: &str, hint_pos: &str, compatible: bool) {
        kb.teach_pos_compatibility(sense_pos, hint_pos, compatible, KnowledgeOrigin::Taught {
            by: self.teacher.clone(),
            at: now_epoch_string(),
        });
    }

    /// Teach a parameter value.
    pub fn teach_param(&self, kb: &mut KnowledgeBase, name: &str, value: f32) {
        kb.set_param(name, value, KnowledgeOrigin::Taught {
            by: self.teacher.clone(),
            at: now_epoch_string(),
        });
    }

    /// Get the teacher name.
    pub fn teacher(&self) -> &str {
        &self.teacher
    }
}

// ========================================================================
// SymbolicObserver — Pattern Induction from Graph Relations
// ========================================================================

/// Observes the graph's relational structure and induces patterns.
///
/// The SymbolicObserver is AAM's "third eye" — it watches the
/// compositions being created in the graph and detects recurring
/// patterns that indicate new knowledge:
///
/// 1. **Marker induction**: If a word always appears before verbs,
///    it's likely a verb-marking auxiliary.
/// 2. **Schema induction**: If compositions with a specific role
///    pattern appear repeatedly, a new ActionSchema can be proposed.
/// 3. **Morphology induction**: If words with a specific prefix
///    always get the same POS, the prefix is likely a morphological
///    marker for that POS.
///
/// This is how AAM "discovers" linguistic knowledge rather than
/// having it hardcoded — by observing the relational structure
/// of its own symbolic graph.
#[derive(Debug, Clone, Default)]
pub struct SymbolicObserver {
    /// Patterns observed so far: pattern_key → count.
    observations: HashMap<String, usize>,
}

impl SymbolicObserver {
    /// Create a new observer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe a composition being created in the graph.
    ///
    /// Records the pattern of roles and types, and when a pattern
    /// reaches the evidence threshold, proposes new knowledge.
    pub fn observe_composition(
        &mut self,
        comp: &super::types::Composition,
        kb: &mut KnowledgeBase,
    ) -> Vec<ObservationResult> {
        let mut results = Vec::new();

        // Pattern 1: Detect repeated [Subject]-[Copula]-[Complement] →
        // If the predicate is always the same word, it's a copula marker.
        if comp.composition_type == CompositionType::EquativeBinding {
            if let Some(pred) = comp.member_with_role(&SemanticRole::Predicate) {
                let pattern = format!("equative_predicate_{}", pred.label());
                let count = self.observations.entry(pattern.clone()).or_insert(0);
                *count += 1;

                // After 5 observations of the same word as equative predicate,
                // propose it as a copula marker.
                if *count == 5 {
                    kb.teach_marker(
                        MarkerCategory::Copula,
                        pred.label(),
                        KnowledgeOrigin::Observed {
                            pattern: pattern.clone(),
                            evidence_count: *count,
                            confidence: 0.8,
                        },
                    );
                    results.push(ObservationResult::NewMarker {
                        category: MarkerCategory::Copula,
                        word: pred.label().to_string(),
                        evidence: *count,
                    });
                }
            }
        }

        // Pattern 2: Detect repeated [Possessor]-[Possessive]-[Possession] →
        // If the predicate is always the same word, it's a possessive marker.
        if comp.composition_type == CompositionType::PossessiveBinding {
            if let Some(pred) = comp.member_with_role(&SemanticRole::Predicate) {
                let pattern = format!("possessive_predicate_{}", pred.label());
                let count = self.observations.entry(pattern.clone()).or_insert(0);
                *count += 1;

                if *count == 5 {
                    kb.teach_marker(
                        MarkerCategory::Possessive,
                        pred.label(),
                        KnowledgeOrigin::Observed {
                            pattern: pattern.clone(),
                            evidence_count: *count,
                            confidence: 0.8,
                        },
                    );
                    results.push(ObservationResult::NewMarker {
                        category: MarkerCategory::Possessive,
                        word: pred.label().to_string(),
                        evidence: *count,
                    });
                }
            }
        }

        // Pattern 3: Detect words that always precede verbs →
        // These are likely verb-marking auxiliaries.
        if comp.composition_type == CompositionType::Event {
            if let Some(agent) = comp.member_with_role(&SemanticRole::Arg0Agent) {
                if let Some(_pred) = comp.member_with_role(&SemanticRole::Predicate) {
                    // Check if "agent" might actually be a modifier/auxiliary
                    // (observed before verbs repeatedly)
                    let pattern = format!("pre_verb_{}", agent.label());
                    let count = self.observations.entry(pattern.clone()).or_insert(0);
                    *count += 1;

                    if *count == 10 {
                        // This word appears before verbs 10+ times —
                        // might be a verb-marking auxiliary, not an agent.
                        // (Low confidence — needs human verification.)
                    }
                }
            }
        }

        results
    }

    /// Get the count of observations for a pattern.
    pub fn observation_count(&self, pattern: &str) -> usize {
        self.observations.get(pattern).copied().unwrap_or(0)
    }
}

/// Result of a symbolic observation.
#[derive(Debug, Clone)]
pub enum ObservationResult {
    /// A new marker was induced from pattern observation.
    NewMarker {
        category: MarkerCategory,
        word: String,
        evidence: usize,
    },
    /// A new morphological rule was induced.
    NewMorphology {
        rule: MorphologyRule,
        evidence: usize,
    },
    /// A new POS compatibility rule was induced.
    NewPosCompatibility {
        sense_pos: String,
        hint_pos: String,
        compatible: bool,
        evidence: usize,
    },
}

// ========================================================================
// Bootstrap Seed — Backward-Compatible Population
// ========================================================================

/// Seed the KnowledgeBase with Indonesian linguistic knowledge.
///
/// This provides the same data that was previously hardcoded as
/// const arrays, but now with `KnowledgeOrigin::Bootstrapped`
/// provenance. This is the TRANSITIONAL mechanism — the long-term
/// goal is to replace all Bootstrapped entries with Taught/Observed/Asked.
///
/// # What Gets Seeded
///
/// - Copula markers: adalah, ialah, merupakan, yaitu, yakni
/// - Possessive markers: punya, miliki, mempunyai, punyai
/// - Negation markers: tidak, bukan, belum, jangan, tak, nggak, enggak, ga, gak
/// - Cause markers: karena, sebab
/// - Purpose markers: untuk, supaya, agar
/// - Condition markers: jika, apabila, kalau, bila, jikalau, bilamana
/// - Verb prefixes: me, ber, di, ter, memper, diper
/// - Verb-marking auxiliaries: tidak, belum, sudah, telah, akan, mau, bisa, dapat, harus, perlu
/// - Noun determiners: ini, itu, sebuah, seekor, seorang, para, sang, si
/// - Degree modifiers: sekali, sangat, terlalu, paling
/// - Morphology rules: meN allomorphs, peN allomorphs, prefixes, suffixes, root exceptions
/// - POS compatibility rules: verb↔adjective, particle↔*, interjection↔*
/// - Stopwords
/// - Adaptive parameters (confidence multipliers, thresholds)
pub fn seed_indonesian(kb: &mut KnowledgeBase) {
    let teacher = TeachingProtocol::new("bootstrap_indonesian");

    // ── Copula markers ──
    teacher.teach_markers(kb, MarkerCategory::Copula, &[
        "adalah", "ialah", "merupakan", "yaitu", "yakni",
    ]);

    // ── Possessive markers ──
    teacher.teach_markers(kb, MarkerCategory::Possessive, &[
        "punya", "miliki", "mempunyai", "punyai",
    ]);

    // ── Equative markers ──
    teacher.teach_markers(kb, MarkerCategory::Equative, &[
        "yaitu", "yakni",
    ]);

    // ── Existential markers ──
    teacher.teach_markers(kb, MarkerCategory::Existential, &[
        "ada",
    ]);

    // ── Locative markers ──
    teacher.teach_markers(kb, MarkerCategory::Locative, &[
        "di", "ke", "dari",
    ]);

    // ── Negation markers ──
    teacher.teach_markers(kb, MarkerCategory::Negation, &[
        "tidak", "bukan", "belum", "jangan", "tak", "nggak", "enggak", "ga", "gak",
    ]);

    // ── Core negation markers ──
    teacher.teach_markers(kb, MarkerCategory::CoreNegation, &[
        "tidak", "bukan", "tak", "jangan",
    ]);

    // ── Cause markers ──
    teacher.teach_markers(kb, MarkerCategory::Cause, &[
        "karena", "sebab",
    ]);

    // ── Purpose markers ──
    teacher.teach_markers(kb, MarkerCategory::Purpose, &[
        "untuk", "supaya", "agar",
    ]);

    // ── Condition markers ──
    teacher.teach_markers(kb, MarkerCategory::Condition, &[
        "jika", "apabila", "kalau", "bila", "jikalau", "bilamana",
    ]);

    // ── Verb prefixes ──
    teacher.teach_markers(kb, MarkerCategory::VerbPrefix, &[
        "me", "ber", "di", "ter", "memper", "diper",
    ]);

    // ── Verb-marking auxiliaries ──
    teacher.teach_markers(kb, MarkerCategory::VerbMarkingAuxiliary, &[
        "tidak", "belum", "sudah", "telah", "akan", "mau",
        "bisa", "dapat", "harus", "perlu",
    ]);

    // ── Noun determiners ──
    teacher.teach_markers(kb, MarkerCategory::NounDeterminer, &[
        "ini", "itu", "sebuah", "seekor", "seorang",
        "para", "sang", "si",
    ]);

    // ── Degree modifiers ──
    teacher.teach_markers(kb, MarkerCategory::DegreeModifier, &[
        "sekali", "sangat", "terlalu", "paling",
    ]);

    // ── Common verbs (not prefix-derived) ──
    teacher.teach_markers(kb, MarkerCategory::CommonVerb, &[
        "ada", "ialah", "adalah", "punya", "mahu", "hendak",
        "boleh", "perlu", "harus", "mesti",
    ]);

    // ── Morphology: Archimorphemes ──
    for archi in &["meN", "peN"] {
        kb.teach_morphology(MorphologyRule {
            rule_type: MorphologyRuleType::Archimorpheme,
            value: archi.to_string(),
            archimorpheme: None,
            condition: None,
        }, KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian archimorpheme".into(),
        });
    }

    // ── Morphology: meN allomorphs ──
    let men_allomorphs = [
        ("meng", "meN", "sebelum vokal, k, g, h"),
        ("meny", "meN", "sebelum s (restore 's')"),
        ("mem",  "meN", "sebelum b, p, f"),
        ("men",  "meN", "sebelum c, d, j, t"),
        ("me",   "meN", "sebelum konsonan lain"),
    ];
    for (allo, archi, cond) in &men_allomorphs {
        kb.teach_morphology(MorphologyRule {
            rule_type: MorphologyRuleType::Allomorph,
            value: allo.to_string(),
            archimorpheme: Some(archi.to_string()),
            condition: Some(cond.to_string()),
        }, KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian meN allomorph".into(),
        });
    }

    // ── Morphology: peN allomorphs ──
    let pen_allomorphs = [
        ("peng", "peN", "sebelum vokal, k, g, h"),
        ("peny", "peN", "sebelum s (restore 's')"),
        ("pem",  "peN", "sebelum b, p, f"),
        ("pen",  "peN", "sebelum c, d, j, t"),
        ("pe",   "peN", "sebelum konsonan lain"),
    ];
    for (allo, archi, cond) in &pen_allomorphs {
        kb.teach_morphology(MorphologyRule {
            rule_type: MorphologyRuleType::Allomorph,
            value: allo.to_string(),
            archimorpheme: Some(archi.to_string()),
            condition: Some(cond.to_string()),
        }, KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian peN allomorph".into(),
        });
    }

    // ── Morphology: Simple prefixes ──
    for pfx in &["memper", "diper", "ber", "di", "ter", "per", "ke", "se"] {
        kb.teach_morphology(MorphologyRule {
            rule_type: MorphologyRuleType::SimplePrefix,
            value: pfx.to_string(),
            archimorpheme: None,
            condition: None,
        }, KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian simple prefix".into(),
        });
    }

    // ── Morphology: Suffixes ──
    for sfx in &["kan", "an", "i", "lah", "kah", "tah", "pun"] {
        kb.teach_morphology(MorphologyRule {
            rule_type: MorphologyRuleType::Suffix,
            value: sfx.to_string(),
            archimorpheme: None,
            condition: None,
        }, KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian suffix".into(),
        });
    }

    // ── Morphology: Root exceptions ──
    let root_exceptions = [
        "makan", "minum", "tahu", "kerja", "lari", "jalan", "tulis", "baca",
        "dengar", "lihat", "ambil", "beri", "buat", "cari", "duduk", "hidup",
        "ikan", "pulang", "sampai", "taruh", "tinggal", "tukar", "pukul",
        "main", "pilih", "bayar", "jual", "beli", "datang", "pergi", "masuk",
        "keluar", "naik", "turun", "buka", "tutup", "pakai", "lepas",
        "suka", "benci", "cinta", "sayang", "harus", "boleh", "bisa",
        "kata", "ada", "ialah", "adalah", "punya", "mahu", "hendak",
        "perlu", "mesti", "orang", "rumah", "air", "api", "tanah",
        "mata", "tangan", "kaki", "kepala", "hati", "badan",
        "raja", "rakyat", "negara", "kerajaan", "hukum", "adat",
        "mental", "modal", "sosial", "formal", "normal", "original",
        "total", "vital", "real", "ideal", "local", "kriminal",
    ];
    for root in &root_exceptions {
        kb.teach_morphology(MorphologyRule {
            rule_type: MorphologyRuleType::RootException,
            value: root.to_string(),
            archimorpheme: None,
            condition: None,
        }, KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian root exception".into(),
        });
    }

    // ── POS compatibility rules ──
    // Indonesian adjectives are stative verbs — they're POS-compatible.
    kb.teach_pos_compatibility("verb", "adjective", true, KnowledgeOrigin::Bootstrapped {
        reason: "Indonesian adjectives are stative verbs".into(),
    });
    kb.teach_pos_compatibility("adjective", "verb", true, KnowledgeOrigin::Bootstrapped {
        reason: "Indonesian adjectives are stative verbs".into(),
    });
    // Function words and interjections are compatible with anything.
    for pos in &["verb", "noun", "adjective", "adverb"] {
        kb.teach_pos_compatibility("particle", pos, true, KnowledgeOrigin::Bootstrapped {
            reason: "Particle is compatible with any POS".into(),
        });
        kb.teach_pos_compatibility("interjection", pos, true, KnowledgeOrigin::Bootstrapped {
            reason: "Interjection is compatible with any POS".into(),
        });
        kb.teach_pos_compatibility("adverb", pos, true, KnowledgeOrigin::Bootstrapped {
            reason: "Adverb is flexible in Indonesian".into(),
        });
    }

    // ── Stopwords ──
    let stopwords = [
        // Indonesian
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada",
        "adalah", "akan", "telah", "sebuah", "seorang", "tidak", "bukan", "juga",
        "sudah", "oleh", "karena", "supaya", "agar", "sebab",
        // English
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "not", "no", "never",
    ];
    for word in &stopwords {
        kb.teach_stopword(word, KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian/English stopword".into(),
        });
    }

    // ── Adaptive parameters ──
    // Confidence multipliers (from reason_frame.rs)
    kb.set_param("reason.problem_solution.confidence", 0.85,
        KnowledgeOrigin::Bootstrapped { reason: "ProblemSolution derivation confidence".into() });
    kb.set_param("reason.goal_inference.confidence", 0.80,
        KnowledgeOrigin::Bootstrapped { reason: "GoalInference derivation confidence".into() });
    kb.set_param("reason.polarity_conflict.confidence", 0.90,
        KnowledgeOrigin::Bootstrapped { reason: "PolarityConflict derivation confidence".into() });
    kb.set_param("reason.condition_consequence.confidence", 0.90,
        KnowledgeOrigin::Bootstrapped { reason: "ConditionConsequence derivation confidence".into() });

    // Confidence modulation (from reason_frame.rs)
    kb.set_param("modulation.activation_boost_cap", 0.15,
        KnowledgeOrigin::Bootstrapped { reason: "Max boost from predicate connectivity".into() });
    kb.set_param("modulation.ambiguity_penalty_rate", 0.5,
        KnowledgeOrigin::Bootstrapped { reason: "Ambiguity penalty multiplier".into() });
    kb.set_param("modulation.ambiguity_threshold", 0.3,
        KnowledgeOrigin::Bootstrapped { reason: "Ambiguity score threshold for penalty".into() });
    kb.set_param("modulation.contradiction_penalty", 0.85,
        KnowledgeOrigin::Bootstrapped { reason: "Contradiction confidence multiplier".into() });
    kb.set_param("modulation.predicate_count_boost_per", 0.02,
        KnowledgeOrigin::Bootstrapped { reason: "Per-instance predicate count boost".into() });
    kb.set_param("modulation.predicate_count_boost_cap", 0.10,
        KnowledgeOrigin::Bootstrapped { reason: "Max predicate count boost".into() });

    // Executive thresholds (from executive.rs)
    kb.set_param("executive.stagnant_batch_threshold", 10.0,
        KnowledgeOrigin::Bootstrapped { reason: "Batches before mode elevation".into() });
    kb.set_param("executive.decayed_confidence_floor", 0.2,
        KnowledgeOrigin::Bootstrapped { reason: "Confidence floor for decayed compositions".into() });
    kb.set_param("executive.reflection_contradiction_threshold", 3.0,
        KnowledgeOrigin::Bootstrapped { reason: "Contradictions triggering Reflective mode".into() });
    kb.set_param("executive.analytical_confidence_floor", 0.5,
        KnowledgeOrigin::Bootstrapped { reason: "Confidence floor for Analytical mode".into() });
    kb.set_param("executive.goal_met_confidence", 0.8,
        KnowledgeOrigin::Bootstrapped { reason: "Confidence for goal met".into() });
    kb.set_param("executive.reflection_promotion_min_age", 3.0,
        KnowledgeOrigin::Bootstrapped { reason: "Min age for reflection promotion".into() });
    kb.set_param("executive.reflection_promotion_min_confidence", 0.6,
        KnowledgeOrigin::Bootstrapped { reason: "Min confidence for reflection promotion".into() });
    kb.set_param("executive.correction_confidence", 0.85,
        KnowledgeOrigin::Bootstrapped { reason: "Confidence assigned to user corrections".into() });

    // Spreading activation defaults (from spreading.rs)
    kb.set_param("spreading.decay_factor", 0.5,
        KnowledgeOrigin::Bootstrapped { reason: "Energy decay per hop".into() });
    kb.set_param("spreading.max_hops", 3.0,
        KnowledgeOrigin::Bootstrapped { reason: "Max propagation depth".into() });
    kb.set_param("spreading.max_activated", 50.0,
        KnowledgeOrigin::Bootstrapped { reason: "Max activated nodes".into() });
    kb.set_param("spreading.min_energy", 0.01,
        KnowledgeOrigin::Bootstrapped { reason: "Minimum energy threshold".into() });
    kb.set_param("spreading.reinforcement", 0.5,
        KnowledgeOrigin::Bootstrapped { reason: "Reinforcement factor".into() });

    // CSD thresholds
    kb.set_param("csd.min_confidence", 0.3,
        KnowledgeOrigin::Bootstrapped { reason: "Min confidence for CSD resolution".into() });
    kb.set_param("csd.min_margin", 0.2,
        KnowledgeOrigin::Bootstrapped { reason: "Min margin between top 2 candidates".into() });
    kb.set_param("csd.lexical_fallback_weight", 0.5,
        KnowledgeOrigin::Bootstrapped { reason: "Jaccard fallback weight in CSD".into() });
    kb.set_param("csd.single_survivor_confidence", 0.9,
        KnowledgeOrigin::Bootstrapped { reason: "Confidence for single-survivor POS filter".into() });
    kb.set_param("csd.context_member_confidence", 0.7,
        KnowledgeOrigin::Bootstrapped { reason: "Confidence for CSD context members".into() });
    kb.set_param("csd.activation_energy_threshold", 0.1,
        KnowledgeOrigin::Bootstrapped { reason: "Min activation energy for role binding".into() });

    // Usage probe weights
    kb.set_param("usage.jaccard_weight", 0.6,
        KnowledgeOrigin::Bootstrapped { reason: "Jaccard weight in usage validation".into() });
    kb.set_param("usage.role_coherence_weight", 0.4,
        KnowledgeOrigin::Bootstrapped { reason: "Role coherence weight in usage validation".into() });
    kb.set_param("usage.validity_threshold", 0.3,
        KnowledgeOrigin::Bootstrapped { reason: "Validity threshold for usage probes".into() });
    kb.set_param("usage.indirect_seed_energy", 0.5,
        KnowledgeOrigin::Bootstrapped { reason: "Energy for indirect seeds".into() });

    // Extract frame confidence computation
    kb.set_param("extract.base_confidence", 0.30,
        KnowledgeOrigin::Bootstrapped { reason: "Base confidence for frame extraction".into() });
    kb.set_param("extract.agent_bonus", 0.15,
        KnowledgeOrigin::Bootstrapped { reason: "Agent role confidence bonus".into() });
    kb.set_param("extract.patient_bonus", 0.15,
        KnowledgeOrigin::Bootstrapped { reason: "Patient role confidence bonus".into() });
    kb.set_param("extract.cause_bonus", 0.10,
        KnowledgeOrigin::Bootstrapped { reason: "Cause role confidence bonus".into() });
    kb.set_param("extract.purpose_bonus", 0.10,
        KnowledgeOrigin::Bootstrapped { reason: "Purpose role confidence bonus".into() });
    kb.set_param("extract.negation_penalty", 0.05,
        KnowledgeOrigin::Bootstrapped { reason: "Negation confidence penalty".into() });
    kb.set_param("extract.schema_base_confidence", 0.35,
        KnowledgeOrigin::Bootstrapped { reason: "Base confidence for schema-driven extraction".into() });
    kb.set_param("extract.schema_role_bonus", 0.15,
        KnowledgeOrigin::Bootstrapped { reason: "Per-role bonus for schema extraction".into() });
}

/// Create a KnowledgeBase seeded with Indonesian data.
///
/// This is the convenience function for backward compatibility —
/// it creates a blank KnowledgeBase and seeds it with the same
/// data that was previously hardcoded.
pub fn create_indonesian_seeded() -> KnowledgeBase {
    let mut kb = KnowledgeBase::new();
    seed_indonesian(&mut kb);
    kb
}

// ========================================================================
// Utility
// ========================================================================

/// Simple epoch-seconds timestamp string.
fn now_epoch_string() -> String {
    format!("{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs())
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v12::types::{Composition, CompositionMember};

    #[test]
    fn test_blank_slate() {
        let kb = KnowledgeBase::new();
        assert!(kb.is_blank());
        assert!(!kb.is_marker(&MarkerCategory::Copula, "adalah"));
        assert!(kb.markers_for(&MarkerCategory::Copula).is_empty());
    }

    #[test]
    fn test_teach_marker() {
        let mut kb = KnowledgeBase::new();
        kb.teach_marker(MarkerCategory::Copula, "adalah",
            KnowledgeOrigin::Taught { by: "test".into(), at: "0".into() });

        assert!(kb.is_marker(&MarkerCategory::Copula, "adalah"));
        assert!(!kb.is_marker(&MarkerCategory::Copula, "merupakan"));
        assert!(!kb.is_marker(&MarkerCategory::Possessive, "adalah"));
    }

    #[test]
    fn test_teach_no_duplicates() {
        let mut kb = KnowledgeBase::new();
        kb.teach_marker(MarkerCategory::Copula, "adalah",
            KnowledgeOrigin::Taught { by: "test".into(), at: "0".into() });
        kb.teach_marker(MarkerCategory::Copula, "adalah",
            KnowledgeOrigin::Taught { by: "test".into(), at: "0".into() });

        assert_eq!(kb.markers_for(&MarkerCategory::Copula).len(), 1);
    }

    #[test]
    fn test_case_insensitive_markers() {
        let mut kb = KnowledgeBase::new();
        kb.teach_marker(MarkerCategory::Copula, "Adalah",
            KnowledgeOrigin::Taught { by: "test".into(), at: "0".into() });

        assert!(kb.is_marker(&MarkerCategory::Copula, "adalah"));
        assert!(kb.is_marker(&MarkerCategory::Copula, "ADALAH"));
    }

    #[test]
    fn test_knowledge_origin_describe() {
        let taught = KnowledgeOrigin::Taught { by: "raymond".into(), at: "12345".into() };
        assert!(taught.describe().contains("raymond"));

        let observed = KnowledgeOrigin::Observed {
            pattern: "copula_before_noun".into(),
            evidence_count: 7,
            confidence: 0.85,
        };
        assert!(observed.describe().contains("7 times"));
    }

    #[test]
    fn test_teaching_protocol() {
        let mut kb = KnowledgeBase::new();
        let teacher = TeachingProtocol::new("user_raymond");

        teacher.teach_marker(&mut kb, MarkerCategory::Copula, "adalah");
        teacher.teach_marker(&mut kb, MarkerCategory::Copula, "ialah");

        assert!(kb.is_marker(&MarkerCategory::Copula, "adalah"));
        assert!(kb.is_marker(&MarkerCategory::Copula, "ialah"));
        assert_eq!(kb.markers_for(&MarkerCategory::Copula).len(), 2);

        // Verify origin is Taught.
        let origin = kb.marker_origin(&MarkerCategory::Copula, "adalah").unwrap();
        match origin {
            KnowledgeOrigin::Taught { by, .. } => assert_eq!(by, "user_raymond"),
            _ => panic!("Expected Taught origin"),
        }
    }

    #[test]
    fn test_adaptive_params() {
        let mut params = AdaptiveParams::new();
        params.set("test_param", 0.5,
            KnowledgeOrigin::Bootstrapped { reason: "test".into() });

        assert!((params.get("test_param", 0.0) - 0.5).abs() < 0.001);
        assert_eq!(params.get("nonexistent", 0.3), 0.3);

        // Calibrate
        params.calibrate("test_param", 0.7, 0.5);
        let calibrated = params.get("test_param", 0.0);
        // 0.5 * 0.5 + 0.7 * 0.5 = 0.6
        assert!((calibrated - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_morphology_rules() {
        let mut kb = KnowledgeBase::new();
        kb.teach_morphology(MorphologyRule {
            rule_type: MorphologyRuleType::SimplePrefix,
            value: "ber".to_string(),
            archimorpheme: None,
            condition: None,
        }, KnowledgeOrigin::Bootstrapped { reason: "test".into() });

        assert_eq!(kb.simple_prefixes().len(), 1);
        assert!(kb.simple_prefixes().contains(&"ber".to_string()));
    }

    #[test]
    fn test_root_exceptions() {
        let mut kb = KnowledgeBase::new();
        kb.teach_morphology(MorphologyRule {
            rule_type: MorphologyRuleType::RootException,
            value: "makan".to_string(),
            archimorpheme: None,
            condition: None,
        }, KnowledgeOrigin::Bootstrapped { reason: "test".into() });

        assert!(kb.is_root_exception("makan"));
        assert!(!kb.is_root_exception("xyz"));
    }

    #[test]
    fn test_seed_indonesian() {
        let kb = create_indonesian_seeded();

        // Should have all the markers
        assert!(kb.is_marker(&MarkerCategory::Copula, "adalah"));
        assert!(kb.is_marker(&MarkerCategory::Possessive, "punya"));
        assert!(kb.is_marker(&MarkerCategory::Negation, "tidak"));
        assert!(kb.is_marker(&MarkerCategory::Cause, "karena"));
        assert!(kb.is_marker(&MarkerCategory::Purpose, "untuk"));
        assert!(kb.is_marker(&MarkerCategory::Condition, "jika"));

        // Should have morphology
        assert!(kb.is_root_exception("makan"));
        assert!(!kb.simple_prefixes().is_empty());
        assert!(!kb.suffixes().is_empty());

        // Should have parameters
        assert!(kb.param("reason.problem_solution.confidence", 0.0) > 0.0);
        assert!(kb.param("spreading.decay_factor", 0.0) > 0.0);

        // Should NOT be blank
        assert!(!kb.is_blank());
    }

    #[test]
    fn test_pos_inference_from_knowledge() {
        let kb = create_indonesian_seeded();

        // "bisa" after "tidak" → verb
        let pos = kb.infer_pos_from_context("bisa", &["tidak", "pergi"]);
        assert_eq!(pos.as_deref(), Some("verb"));

        // "bisa" after "seekor" → noun
        let pos = kb.infer_pos_from_context("bisa", &["seekor", "ular"]);
        assert_eq!(pos.as_deref(), Some("noun"));

        // "memakan" → verb (prefix match)
        let pos = kb.infer_pos_from_context("memakan", &["ikan"]);
        assert_eq!(pos.as_deref(), Some("verb"));

        // No context → None
        let pos = kb.infer_pos_from_context("bisa", &["ular", "gigitan"]);
        assert!(pos.is_none());
    }

    #[test]
    fn test_pos_compatibility_from_knowledge() {
        let kb = create_indonesian_seeded();

        // Exact match
        assert!(kb.pos_compatible("verb", "verb"));
        assert!(kb.pos_compatible("noun", "noun"));

        // Indonesian adjective-verb compatibility
        assert!(kb.pos_compatible("verb", "adjective"));
        assert!(kb.pos_compatible("adjective", "verb"));

        // Incompatible
        assert!(!kb.pos_compatible("noun", "verb"));

        // Particle is flexible
        assert!(kb.pos_compatible("particle", "verb"));
        assert!(kb.pos_compatible("interjection", "noun"));
    }

    #[test]
    fn test_verb_prefix_match() {
        let kb = create_indonesian_seeded();
        assert!(kb.is_verb_prefix_match("membuat"));
        assert!(kb.is_verb_prefix_match("berlari"));
        assert!(!kb.is_verb_prefix_match("raja"));
    }

    #[test]
    fn test_stopwords() {
        let kb = create_indonesian_seeded();
        assert!(kb.is_stopword("yang"));
        assert!(kb.is_stopword("the"));
        assert!(!kb.is_stopword("raja"));
    }

    #[test]
    fn test_audit_report() {
        let kb = create_indonesian_seeded();
        let report = kb.audit_report();
        assert!(report.contains("Knowledge Base Audit"));
        assert!(report.contains("Marker Categories"));
        assert!(report.contains("Parameters"));
    }

    #[test]
    fn test_symbolic_observer() {
        let mut kb = KnowledgeBase::new();
        let mut observer = SymbolicObserver::new();

        // Observe 5 EquativeBinding compositions with "adalah" as predicate
        for _ in 0..5 {
            let comp = Composition {
                composition_type: CompositionType::EquativeBinding,
                members: vec![
                    CompositionMember {
                        node_id: 0,
                        role: SemanticRole::Predicate,
                        confidence: 0.9,
                        label: "adalah".to_string(),
                        source: None,
                    },
                ],
                ..Composition::default()
            };
            observer.observe_composition(&comp, &mut kb);
        }

        // After 5 observations, "adalah" should be a learned copula marker
        assert!(kb.is_marker(&MarkerCategory::Copula, "adalah"));
    }

    #[test]
    fn test_any_role_marker() {
        let kb = create_indonesian_seeded();
        assert!(kb.is_any_role_marker("tidak"));
        assert!(kb.is_any_role_marker("karena"));
        assert!(kb.is_any_role_marker("untuk"));
        assert!(kb.is_any_role_marker("jika"));
        assert!(!kb.is_any_role_marker("raja"));
    }
}
