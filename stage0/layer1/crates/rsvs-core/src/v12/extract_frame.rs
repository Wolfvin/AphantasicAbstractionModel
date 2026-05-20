//! # MD-1: ExtractFrame Transform
//!
//! Rule-based semantic frame extraction from raw text. This is the Phase 1
//! extractor that converts sentence-like input into structured `SemanticAtom`s
//! of type `Event`.
//!
//! ## Extraction Pipeline
//!
//! ```text
//! Raw text → is_sentence_like? → detect voice → detect polarity
//!         → extract predicate → extract roles → compute confidence
//!         → SemanticAtom(Event)
//! ```
//!
//! ## Voice Detection
//!
//! In Malay/Indonesian, the "di-" prefix marks passive voice:
//! - Active: "Raymond **membuat** aplikasi" (Raymond makes an app)
//! - Passive: "Aplikasi **dibuat** oleh Raymond" (The app is made by Raymond)
//!
//! ## Polarity Detection
//!
//! Negation markers: "tidak", "bukan", "belum", "jangan", "tak", "nggak"
//!
//! ## Confidence Computation
//!
//! ```text
//! base = 0.30
//! + 0.15 if Agent present
//! + 0.15 if Patient present
//! + 0.10 if Cause present
//! + 0.10 if Purpose present
//! - 0.05 if Negative polarity
//! ```
//!
//! ## Feature Flag
//!
//! This module is only compiled when the `v12` feature is enabled.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::pipeline::{ErasedTransform, Graph, IngestResult};
use super::types::*;
use crate::types::{EdgeSource, NodeId};

// ========================================================================
// Negation Markers — Malay/Indonesian
// ========================================================================

/// Known negation markers for polarity detection.
///
/// These are Malay/Indonesian negation words that flip the polarity
/// of an event from Positive to Negative.
const NEGATION_MARKERS: &[&str] = &[
    "tidak",  // not (general negation)
    "bukan",  // not (identity negation)
    "belum",  // not yet
    "jangan", // don't (prohibitive)
    "tak",    // not (short form)
    "nggak",  // not (colloquial)
    "enggak", // not (colloquial variant)
    "ga",     // not (very colloquial)
    "gak",    // not (very colloquial variant)
];

/// Known causal/purpose markers for role extraction.
///
/// These markers signal Cause or Purpose roles in Malay/Indonesian:
/// - "karena" → Cause (because)
/// - "sebab" → Cause (because, more formal)
/// - "untuk" → Purpose (for/in order to)
/// - "supaya" → Purpose (so that)
/// - "agar" → Purpose (so that, more formal)
const CAUSE_MARKERS: &[&str] = &["karena", "sebab"];
const PURPOSE_MARKERS: &[&str] = &["untuk", "supaya", "agar"];

/// Conditional markers in Indonesian — trigger ConditionConsequence extraction.
///
/// When one of these markers appears, the text before it is the Antecedent
/// (condition) and the text after it is the Consequent (consequence).
///
/// - "jika" → if
/// - "apabila" → if/when (formal)
/// - "kalau" → if (informal)
/// - "bila" → if/when
/// - "jikalau" → if (archaic/formal)
/// - "bilamana" → whenever (formal)
const CONDITION_MARKERS: &[&str] = &["jika", "apabila", "kalau", "bila", "jikalau", "bilamana"];

/// Known verb prefixes in Malay/Indonesian for predicate detection.
const VERB_PREFIXES: &[&str] = &["me", "ber", "di", "ter", "ke", "pe"];

// ========================================================================
// ExtractionQuality — Quality Tracking
// ========================================================================

/// Quality classification of an extraction result (MD-1).
///
/// Categorizes the output of `ExtractFrame` to support quality tracking
/// and feedback loop decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum ExtractionQuality {
    /// High-confidence extraction with all key roles filled.
    HighQuality,
    /// Moderate confidence — some roles missing but predicate is clear.
    ModerateQuality,
    /// Low confidence — only predicate extracted, few or no roles.
    LowQuality,
    /// Failed extraction — input did not yield a frame.
    #[default]
    Failed,
}

/// Tracker for extraction quality across a pipeline run (MD-1).
///
/// Maintains running statistics on extraction quality, enabling the
/// feedback loop to identify systematic weaknesses and trigger
/// re-extraction for low-confidence frames.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtractionQualityTrackerExt {
    /// Number of high-quality extractions.
    #[serde(default)]
    pub high_quality: usize,
    /// Number of moderate-quality extractions.
    #[serde(default)]
    pub moderate_quality: usize,
    /// Number of low-quality extractions.
    #[serde(default)]
    pub low_quality: usize,
    /// Number of failed extractions (input was not sentence-like).
    #[serde(default)]
    pub failed: usize,
    /// Sum of confidence scores for average computation.
    #[serde(default)]
    pub confidence_sum: f32,
    /// Number of frames extracted (for averaging).
    #[serde(default)]
    pub frame_count: usize,
}

impl ExtractionQualityTrackerExt {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an extraction result.
    pub fn record(&mut self, quality: &ExtractionQuality, confidence: f32) {
        match quality {
            ExtractionQuality::HighQuality => self.high_quality += 1,
            ExtractionQuality::ModerateQuality => self.moderate_quality += 1,
            ExtractionQuality::LowQuality => self.low_quality += 1,
            ExtractionQuality::Failed => self.failed += 1,
        }
        if *quality != ExtractionQuality::Failed {
            self.confidence_sum += confidence;
            self.frame_count += 1;
        }
    }

    /// Compute average confidence across all non-failed extractions.
    pub fn average_confidence(&self) -> f32 {
        if self.frame_count == 0 {
            0.0
        } else {
            self.confidence_sum / self.frame_count as f32
        }
    }

    /// What fraction of extractions are low quality or failed?
    pub fn weak_fraction(&self) -> f32 {
        let total = self.high_quality + self.moderate_quality + self.low_quality + self.failed;
        if total == 0 {
            0.0
        } else {
            (self.low_quality + self.failed) as f32 / total as f32
        }
    }
}

// ========================================================================
// ExtractFrame — The Transform
// ========================================================================

/// MD-1: Rule-based semantic frame extraction transform.
///
/// Converts raw text into structured `SemanticAtom` events using deterministic
/// rules. This is Phase 1 extraction — future phases will add UD parsing,
/// SRL labeling, and AMR compilation.
///
/// # Transform Signature
///
/// ```text
/// Input:  &str (raw text, read from ctx.raw_text)
/// Output: Option<SemanticAtom> (Event atom if extraction succeeds)
/// ```
///
/// # Algorithm
///
/// 1. Check `is_sentence_like()` heuristic
/// 2. Detect voice (active/passive via "di-" prefix)
/// 3. Detect polarity (negation markers)
/// 4. Extract predicate (verb-like token)
/// 5. Extract roles (Agent, Patient, Cause, Purpose)
/// 6. Compute frame confidence
/// 7. Build and return `SemanticAtom`
#[derive(Debug, Clone, Default)]
pub struct ExtractFrame {
    /// Whether to use graph-assisted re-extraction when available.
    pub graph_assisted: bool,
}

impl ExtractFrame {
    /// Create a new ExtractFrame transform with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with graph-assisted mode enabled.
    pub fn with_graph_assist() -> Self {
        Self {
            graph_assisted: true,
        }
    }

    /// Heuristic: is the input text sentence-like?
    ///
    /// A text is sentence-like if:
    /// - It has >= 3 whitespace-separated tokens
    /// - It contains at least one verb-like token
    /// - It is not purely repetitive (all tokens identical)
    ///
    /// This is a more thorough check than `PipelineContext::is_sentence_like()`,
    /// which only checks for whitespace and minimum length.
    pub fn is_sentence_like(text: &str) -> bool {
        let tokens: Vec<&str> = text.split_whitespace().collect();

        // Must have at least 3 tokens.
        if tokens.len() < 3 {
            return false;
        }

        // Must not be purely repetitive.
        let first = tokens[0];
        if tokens.iter().all(|t| *t == first) {
            return false;
        }

        // Must have at least one verb-like token.
        let has_verb = tokens.iter().any(|t| is_verb_like(t));
        if !has_verb {
            return false;
        }

        true
    }

    /// Detect voice from the tokens.
    ///
    /// Returns `Some(Voice::Passive)` if any token starts with "di-"
    /// (the Malay/Indonesian passive prefix), otherwise `Some(Voice::Active)`.
    pub fn detect_voice(tokens: &[&str]) -> Voice {
        let has_passive = tokens.iter().any(|t| {
            let lower = t.to_lowercase();
            lower.starts_with("di")
                && lower.len() > 2
                && lower
                    .chars()
                    .nth(2)
                    .is_some_and(|c| c.is_ascii_alphabetic())
        });

        if has_passive {
            Voice::Passive
        } else {
            Voice::Active
        }
    }

    /// Detect polarity from the tokens.
    ///
    /// Returns `Some(Polarity::Negative)` if any token is a negation marker,
    /// otherwise `Some(Polarity::Positive)`.
    pub fn detect_polarity(tokens: &[&str]) -> Polarity {
        let has_negation = tokens.iter().any(|t| {
            let lower = t.to_lowercase();
            NEGATION_MARKERS.contains(&lower.as_str())
        });

        if has_negation {
            Polarity::Negative
        } else {
            Polarity::Positive
        }
    }

    /// Extract the predicate (verb-like token) from the tokens.
    ///
    /// Strategy: find the first verb-like token. In passive voice,
    /// the "di-" prefixed token is the predicate. In active voice,
    /// the "me-" prefixed token is preferred.
    pub fn extract_predicate<'a>(tokens: &'a [&str], voice: &Voice) -> Option<&'a str> {
        match voice {
            Voice::Passive => {
                // In passive, prefer "di-" prefixed token.
                tokens
                    .iter()
                    .find(|t| {
                        let lower = t.to_lowercase();
                        lower.starts_with("di") && lower.len() > 2
                    })
                    .copied()
            }
            Voice::Active => {
                // In active, prefer "me-" prefixed token.
                let me_verb = tokens.iter().find(|t| {
                    let lower = t.to_lowercase();
                    lower.starts_with("me") && lower.len() > 2
                });
                if let Some(v) = me_verb {
                    return Some(v);
                }
                // Fall back to any verb-like token.
                tokens.iter().find(|t| is_verb_like(t)).copied()
            }
        }
    }

    /// Extract semantic roles from the tokens.
    ///
    /// Uses positional heuristics and marker detection:
    /// - **Agent**: In active voice, the token before the predicate.
    ///   In passive voice, the token after "oleh" (by).
    /// - **Patient**: In active voice, the token after the predicate.
    ///   In passive voice, the first non-predicate token before "oleh".
    /// - **Cause**: The token(s) after a cause marker ("karena", "sebab").
    /// - **Purpose**: The token(s) after a purpose marker ("untuk", "supaya", "agar").
    pub fn extract_roles(
        tokens: &[&str],
        predicate: &str,
        voice: &Voice,
    ) -> HashMap<SemanticRole, String> {
        let mut roles = HashMap::new();
        let pred_idx = tokens.iter().position(|t| *t == predicate);

        if let Some(idx) = pred_idx {
            match voice {
                Voice::Active => {
                    // Agent: token before predicate (if exists and not a marker).
                    if idx > 0 {
                        let agent_candidate = tokens[idx - 1];
                        if !is_marker(agent_candidate) {
                            roles.insert(SemanticRole::Arg0Agent, agent_candidate.to_lowercase());
                        }
                    }

                    // Patient: token after predicate (if exists and not a marker).
                    if idx + 1 < tokens.len() {
                        let patient_candidate = tokens[idx + 1];
                        if !is_marker(patient_candidate)
                            && !is_cause_marker(patient_candidate)
                            && !is_purpose_marker(patient_candidate)
                        {
                            roles.insert(
                                SemanticRole::Arg1Patient,
                                patient_candidate.to_lowercase(),
                            );
                        }
                    }
                }
                Voice::Passive => {
                    // In passive voice, the patient is the subject (before predicate).
                    // Find the first non-negation token before the predicate.
                    for i in (0..idx).rev() {
                        let candidate = tokens[i];
                        if !is_negation_marker(candidate) && !is_verb_like(candidate) {
                            roles.insert(SemanticRole::Arg1Patient, candidate.to_lowercase());
                            break;
                        }
                    }

                    // Agent: token after "oleh" (by).
                    if let Some(oleh_idx) = tokens.iter().position(|t| t.to_lowercase() == "oleh") {
                        if oleh_idx + 1 < tokens.len() {
                            roles.insert(
                                SemanticRole::Arg0Agent,
                                tokens[oleh_idx + 1].to_lowercase(),
                            );
                        }
                    }
                }
            }

            // Cause: token(s) after cause markers.
            for (i, token) in tokens.iter().enumerate() {
                if is_cause_marker(token) && i + 1 < tokens.len() {
                    // Collect remaining tokens after the marker as the cause.
                    let cause_tokens: Vec<&str> = tokens[i + 1..].to_vec();
                    if !cause_tokens.is_empty() {
                        roles.insert(SemanticRole::Cause, cause_tokens.join(" ").to_lowercase());
                    }
                    break;
                }
            }

            // Purpose: token(s) after purpose markers.
            for (i, token) in tokens.iter().enumerate() {
                if is_purpose_marker(token) && i + 1 < tokens.len() {
                    let purpose_tokens: Vec<&str> = tokens[i + 1..].to_vec();
                    if !purpose_tokens.is_empty() {
                        roles.insert(
                            SemanticRole::Purpose,
                            purpose_tokens.join(" ").to_lowercase(),
                        );
                    }
                    break;
                }
            }

            // Condition/Consequence: detect conditional patterns.
            // When a condition marker is found, split into Antecedent/Consequent.
            for (i, token) in tokens.iter().enumerate() {
                if is_condition_marker(token) {
                    // Antecedent: tokens before the condition marker.
                    if i > 0 {
                        let ante_tokens: Vec<&str> = tokens[..i].to_vec();
                        if !ante_tokens.is_empty() {
                            roles.insert(
                                SemanticRole::Antecedent,
                                ante_tokens.join(" ").to_lowercase(),
                            );
                        }
                    }
                    // Consequent: tokens after the condition marker.
                    if i + 1 < tokens.len() {
                        let cons_tokens: Vec<&str> = tokens[i + 1..].to_vec();
                        if !cons_tokens.is_empty() {
                            roles.insert(
                                SemanticRole::Consequent,
                                cons_tokens.join(" ").to_lowercase(),
                            );
                        }
                    }
                    break;
                }
            }
        }

        roles
    }

    /// Compute frame confidence based on role coverage.
    ///
    /// ```text
    /// base = 0.30
    /// + 0.15 if Agent present
    /// + 0.15 if Patient present
    /// + 0.10 if Cause present
    /// + 0.10 if Purpose present
    /// + 0.10 if Antecedent present (conditional pattern)
    /// + 0.10 if Consequent present (conditional pattern)
    /// - 0.05 if Negative polarity
    /// ```
    ///
    /// The result is clamped to [0.0, 1.0].
    pub fn compute_frame_confidence(
        roles: &HashMap<SemanticRole, String>,
        polarity: &Polarity,
    ) -> f32 {
        let mut confidence = 0.30f32;

        if roles.contains_key(&SemanticRole::Arg0Agent) {
            confidence += 0.15;
        }
        if roles.contains_key(&SemanticRole::Arg1Patient) {
            confidence += 0.15;
        }
        if roles.contains_key(&SemanticRole::Cause) {
            confidence += 0.10;
        }
        if roles.contains_key(&SemanticRole::Purpose) {
            confidence += 0.10;
        }
        if roles.contains_key(&SemanticRole::Antecedent) {
            confidence += 0.10;
        }
        if roles.contains_key(&SemanticRole::Consequent) {
            confidence += 0.10;
        }
        if *polarity == Polarity::Negative {
            confidence -= 0.05;
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Classify extraction quality based on roles and confidence.
    pub fn classify_quality(
        roles: &HashMap<SemanticRole, String>,
        confidence: f32,
    ) -> ExtractionQuality {
        if confidence >= 0.70
            && roles.contains_key(&SemanticRole::Arg0Agent)
            && roles.contains_key(&SemanticRole::Arg1Patient)
        {
            ExtractionQuality::HighQuality
        } else if confidence >= 0.45
            && (roles.contains_key(&SemanticRole::Arg0Agent)
                || roles.contains_key(&SemanticRole::Arg1Patient))
        {
            ExtractionQuality::ModerateQuality
        } else if !roles.is_empty() {
            ExtractionQuality::LowQuality
        } else {
            ExtractionQuality::Failed
        }
    }

    /// Extract a semantic frame from raw text.
    ///
    /// This is the core extraction method. Returns `Some(SemanticAtom)` if
    /// the input is sentence-like and a predicate can be found, `None` otherwise.
    pub fn extract(&self, text: &str) -> Option<SemanticAtom> {
        if !Self::is_sentence_like(text) {
            return None;
        }

        let tokens: Vec<&str> = text.split_whitespace().collect();
        let voice = Self::detect_voice(&tokens);
        let polarity = Self::detect_polarity(&tokens);

        let predicate = Self::extract_predicate(&tokens, &voice)?;
        let roles = Self::extract_roles(&tokens, predicate, &voice);
        let confidence = Self::compute_frame_confidence(&roles, &polarity);

        Some(SemanticAtom {
            id: String::new(), // Will be assigned by PipelineContext
            label: predicate.to_lowercase(),
            atom_type: AtomType::Event,
            roles,
            polarity: Some(polarity),
            voice: Some(voice),
            variant: Some(AtomVariant::FrameVariant(FrameSource::RuleBased)),
            confidence,
            source: EdgeSource::FrameCompiler,
            composition_id: None,
        })
    }

    /// Graph-assisted re-extraction with context hints.
    ///
    /// When the feedback loop identifies a weak frame, it can request
    /// re-extraction with graph context (known role fillers from related
    /// compositions). This method re-runs extraction and merges any
    /// graph-provided context into the roles.
    ///
    /// The `FrameSource` is set to `GraphAssisted` to mark that this
    /// extraction used graph context.
    pub fn re_extract_with_context(
        &self,
        text: &str,
        graph_context: &[(SemanticRole, NodeId, f32)],
        graph: &Graph,
    ) -> Option<SemanticAtom> {
        let mut atom = self.extract(text)?;

        // Merge graph context: add roles that are missing from extraction
        // but present in graph context with sufficient confidence.
        for (role, node_id, confidence) in graph_context {
            if !atom.roles.contains_key(role) && *confidence >= 0.5 {
                if let Some(label) = graph.node_label(*node_id) {
                    atom.roles.insert(role.clone(), label.to_string());
                }
            }
        }

        // Recompute confidence with merged roles.
        let polarity = atom.polarity.as_ref().unwrap_or(&Polarity::Positive);
        atom.confidence = Self::compute_frame_confidence(&atom.roles, polarity);

        // Mark as graph-assisted.
        atom.variant = Some(AtomVariant::FrameVariant(FrameSource::GraphAssisted));

        Some(atom)
    }
}

/// Implement the `Transform` trait for `ExtractFrame`.
impl Transform for ExtractFrame {
    type Input = String;
    type Output = Option<SemanticAtom>;

    fn id(&self) -> &'static str {
        "ExtractFrame"
    }

    fn transform(&self, input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        self.extract(input)
    }
}

/// Implement `ErasedTransform` for pipeline integration.
impl ErasedTransform for ExtractFrame {
    fn id(&self) -> &'static str {
        "ExtractFrame"
    }

    fn execute(&self, ctx: &mut PipelineContext, _graph: &mut Graph) -> IngestResult {
        let text = match &ctx.raw_text {
            Some(t) => t.clone(),
            None => return IngestResult::new(),
        };

        let mut atoms_created = 0;

        // Check if the input is sentence-like before attempting extraction.
        if !Self::is_sentence_like(&text) {
            // Update quality tracker.
            ctx.extraction_quality.low_confidence_frames += 1;
            return IngestResult::new();
        }

        if let Some(mut atom) = self.extract(&text) {
            // Assign atom ID.
            atom.id = format!("atom_{}", ctx.next_atom_id());

            // Record in current_atoms.
            ctx.current_atoms.push(atom.clone());
            atoms_created += 1;

            // Update quality tracker.
            ctx.extraction_quality.frames_extracted += 1;
            ctx.extraction_quality.average_confidence = (ctx.extraction_quality.average_confidence
                * (ctx.extraction_quality.frames_extracted - 1) as f32
                + atom.confidence)
                / ctx.extraction_quality.frames_extracted as f32;

            if atom.confidence < 0.5 {
                ctx.extraction_quality.low_confidence_frames += 1;
            }
        }

        IngestResult {
            atoms_created,
            compositions_created: 0,
            edges_created: 0,
            gaps_detected: 0,
            enrichments_applied: 0,
            governance_transitions: 0,
        }
    }
}

// ========================================================================
// Helper Functions
// ========================================================================

/// Is a token verb-like?
///
/// A token is verb-like if it:
/// - Starts with a known Malay/Indonesian verb prefix (me-, ber-, di-, ter-), OR
/// - Is a known common verb, OR
/// - Is at least 4 characters and looks like a derived verb
fn is_verb_like(token: &str) -> bool {
    let lower = token.to_lowercase();

    // Check verb prefixes.
    for prefix in VERB_PREFIXES {
        if lower.starts_with(prefix) && lower.len() > prefix.len() + 1 {
            return true;
        }
    }

    // Check common verb list (very short for Phase 1).
    const COMMON_VERBS: &[&str] = &[
        "ada", "ialah", "adalah", "punya", "mahu", "hendak", "boleh", "perlu", "harus",
        "mesti",
    ];
    if COMMON_VERBS.contains(&lower.as_str()) {
        return true;
    }

    false
}

/// Is a token a marker (negation, cause, purpose, condition)?
fn is_marker(token: &str) -> bool {
    is_negation_marker(token)
        || is_cause_marker(token)
        || is_purpose_marker(token)
        || is_condition_marker(token)
}

/// Is a token a negation marker?
fn is_negation_marker(token: &str) -> bool {
    NEGATION_MARKERS.contains(&token.to_lowercase().as_str())
}

/// Is a token a cause marker?
fn is_cause_marker(token: &str) -> bool {
    CAUSE_MARKERS.contains(&token.to_lowercase().as_str())
}

/// Is a token a purpose marker?
fn is_purpose_marker(token: &str) -> bool {
    PURPOSE_MARKERS.contains(&token.to_lowercase().as_str())
}

/// Is a token a conditional marker (jika, apabila, kalau, etc.)?
fn is_condition_marker(token: &str) -> bool {
    CONDITION_MARKERS.contains(&token.to_lowercase().as_str())
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_sentence_like() {
        // Too short.
        assert!(!ExtractFrame::is_sentence_like("hello"));
        assert!(!ExtractFrame::is_sentence_like("dua kata"));

        // Repetitive.
        assert!(!ExtractFrame::is_sentence_like("satu satu satu"));

        // No verb.
        assert!(!ExtractFrame::is_sentence_like("kucing besar hitam"));

        // Valid sentence with verb.
        assert!(ExtractFrame::is_sentence_like(
            "Raymond membuat aplikasi karena lambat"
        ));
    }

    #[test]
    fn test_detect_voice_active() {
        let tokens: Vec<&str> = "Raymond membuat aplikasi".split_whitespace().collect();
        assert_eq!(ExtractFrame::detect_voice(&tokens), Voice::Active);
    }

    #[test]
    fn test_detect_voice_passive() {
        let tokens: Vec<&str> = "Aplikasi dibuat oleh Raymond".split_whitespace().collect();
        assert_eq!(ExtractFrame::detect_voice(&tokens), Voice::Passive);
    }

    #[test]
    fn test_detect_polarity_positive() {
        let tokens: Vec<&str> = "Raymond membuat aplikasi".split_whitespace().collect();
        assert_eq!(ExtractFrame::detect_polarity(&tokens), Polarity::Positive);
    }

    #[test]
    fn test_detect_polarity_negative() {
        let tokens: Vec<&str> = "Raymond tidak membuat aplikasi"
            .split_whitespace()
            .collect();
        assert_eq!(ExtractFrame::detect_polarity(&tokens), Polarity::Negative);
    }

    #[test]
    fn test_extract_predicate() {
        let tokens: Vec<&str> = "Raymond membuat aplikasi".split_whitespace().collect();
        let pred = ExtractFrame::extract_predicate(&tokens, &Voice::Active);
        assert_eq!(pred, Some("membuat"));
    }

    #[test]
    fn test_compute_frame_confidence() {
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "raymond".to_string());
        roles.insert(SemanticRole::Arg1Patient, "aplikasi".to_string());

        let confidence = ExtractFrame::compute_frame_confidence(&roles, &Polarity::Positive);
        // 0.30 + 0.15 + 0.15 = 0.60
        assert!((confidence - 0.60).abs() < 0.01);

        // With Cause: 0.30 + 0.15 + 0.15 + 0.10 = 0.70
        roles.insert(SemanticRole::Cause, "lambat".to_string());
        let confidence = ExtractFrame::compute_frame_confidence(&roles, &Polarity::Positive);
        assert!((confidence - 0.70).abs() < 0.01);

        // Negative: 0.70 - 0.05 = 0.65
        let confidence = ExtractFrame::compute_frame_confidence(&roles, &Polarity::Negative);
        assert!((confidence - 0.65).abs() < 0.01);
    }

    #[test]
    fn test_full_extraction() {
        let ef = ExtractFrame::new();
        let result = ef.extract("Raymond membuat aplikasi karena lambat");

        assert!(result.is_some());
        let atom = result.unwrap();
        assert_eq!(atom.atom_type, AtomType::Event);
        assert_eq!(atom.label, "membuat");
        assert_eq!(atom.voice, Some(Voice::Active));
        assert_eq!(atom.polarity, Some(Polarity::Positive));
        assert!(atom.roles.contains_key(&SemanticRole::Arg0Agent));
        assert!(atom.roles.contains_key(&SemanticRole::Cause));
    }

    #[test]
    fn test_quality_classification() {
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "a".to_string());
        roles.insert(SemanticRole::Arg1Patient, "b".to_string());

        assert_eq!(
            ExtractFrame::classify_quality(&roles, 0.75),
            ExtractionQuality::HighQuality
        );
        assert_eq!(
            ExtractFrame::classify_quality(&roles, 0.50),
            ExtractionQuality::ModerateQuality
        );

        roles.remove(&SemanticRole::Arg1Patient);
        assert_eq!(
            ExtractFrame::classify_quality(&roles, 0.35),
            ExtractionQuality::LowQuality
        );
    }
}
