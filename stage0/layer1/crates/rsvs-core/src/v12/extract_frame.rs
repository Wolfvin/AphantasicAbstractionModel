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

use super::knowledge_base::{KnowledgeBase, MarkerCategory};
use super::pipeline::{ErasedTransform, Graph, IngestResult};
use super::types::*;
use crate::types::{EdgeSource, NodeId};

// ========================================================================
// Named Constants — Audit v6 fix
// ========================================================================
// NOTE: BASE_EXTRACTION_CONFIDENCE and all marker/prefix const arrays have
// been migrated to KnowledgeBase. Use kb.param() and kb.is_marker() instead.

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
///
/// **Audit v5 fix (D14)**: This is now wired into `ExtractFrame::execute()`
/// alongside the simpler `ExtractionQualityTracker` in `PipelineContext`.
/// The `ExtractionQualityTracker` stores aggregate stats (frames_extracted,
/// average_confidence, low_confidence_frames) while this tracker provides
/// per-quality-level breakdown (high/moderate/low/failed counts).
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

    /// Heuristic: is the input text sentence-like? (KB-based)
    ///
    /// A text is sentence-like if:
    /// - It has >= 3 whitespace-separated tokens
    /// - It contains at least one verb-like token
    /// - It is not purely repetitive (all tokens identical)
    ///
    /// This is a more thorough check than `PipelineContext::is_sentence_like()`,
    /// which only checks for whitespace and minimum length.
    pub fn is_sentence_like_with_kb(text: &str, kb: &KnowledgeBase) -> bool {
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
        let has_verb = tokens.iter().any(|t| is_verb_like_with_kb(t, kb));
        if !has_verb {
            return false;
        }

        true
    }

    /// Backward-compatible wrapper using bootstrapped KB.
    #[deprecated(note = "Use is_sentence_like_with_kb instead")]
    pub fn is_sentence_like(text: &str) -> bool {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        Self::is_sentence_like_with_kb(text, &kb)
    }

    /// Detect voice from the tokens.
    ///
    /// Returns `Some(Voice::Passive)` if any token starts with "di-"
    /// (the Malay/Indonesian passive prefix), otherwise `Some(Voice::Active)`.
    pub fn detect_voice(tokens: &[&str]) -> Voice {
        let has_passive = tokens.iter().any(|t| {
            super::stemmer::IndonesianStemmer::is_passive_verb(t)
        });

        if has_passive {
            Voice::Passive
        } else {
            Voice::Active
        }
    }

    /// Detect polarity from the tokens using KnowledgeBase.
    pub fn detect_polarity_with_kb(tokens: &[&str], kb: &KnowledgeBase) -> Polarity {
        let has_negation = tokens.iter().any(|t| {
            kb.is_marker(&MarkerCategory::Negation, t)
        });

        if has_negation {
            Polarity::Negative
        } else {
            Polarity::Positive
        }
    }

    /// Backward-compatible wrapper using bootstrapped KB.
    #[deprecated(note = "Use detect_polarity_with_kb instead")]
    pub fn detect_polarity(tokens: &[&str]) -> Polarity {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        Self::detect_polarity_with_kb(tokens, &kb)
    }

    /// Extract the predicate (verb-like token) from the tokens using KnowledgeBase.
    pub fn extract_predicate_with_kb<'a>(tokens: &'a [&str], voice: &Voice, kb: &KnowledgeBase) -> Option<&'a str> {
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
                tokens.iter().find(|t| is_verb_like_with_kb(t, kb)).copied()
            }
        }
    }

    /// Backward-compatible wrapper using bootstrapped KB.
    #[deprecated(note = "Use extract_predicate_with_kb instead")]
    pub fn extract_predicate<'a>(tokens: &'a [&str], voice: &Voice) -> Option<&'a str> {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        Self::extract_predicate_with_kb(tokens, voice, &kb)
    }

    /// Extract semantic roles from the tokens using KnowledgeBase.
    pub fn extract_roles_with_kb(
        tokens: &[&str],
        predicate: &str,
        voice: &Voice,
        kb: &KnowledgeBase,
    ) -> HashMap<SemanticRole, String> {
        let mut roles = HashMap::new();
        let pred_idx = tokens.iter().position(|t| *t == predicate);

        if let Some(idx) = pred_idx {
            match voice {
                Voice::Active => {
                    // Agent: token before predicate (if exists and not a marker).
                    if idx > 0 {
                        let agent_candidate = tokens[idx - 1];
                        if !is_marker_with_kb(agent_candidate, kb) {
                            roles.insert(SemanticRole::Arg0Agent, agent_candidate.to_lowercase());
                        }
                    }

                    // Patient: token after predicate (if exists and not a marker).
                    if idx + 1 < tokens.len() {
                        let patient_candidate = tokens[idx + 1];
                        if !is_marker_with_kb(patient_candidate, kb)
                            && !is_cause_marker_with_kb(patient_candidate, kb)
                            && !is_purpose_marker_with_kb(patient_candidate, kb)
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
                        if !is_negation_marker_with_kb(candidate, kb) && !is_verb_like_with_kb(candidate, kb) {
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
                if is_cause_marker_with_kb(token, kb) && i + 1 < tokens.len() {
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
                if is_purpose_marker_with_kb(token, kb) && i + 1 < tokens.len() {
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
                if is_condition_marker_with_kb(token, kb) {
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

    /// Backward-compatible wrapper using bootstrapped KB.
    #[deprecated(note = "Use extract_roles_with_kb instead")]
    pub fn extract_roles(
        tokens: &[&str],
        predicate: &str,
        voice: &Voice,
    ) -> HashMap<SemanticRole, String> {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        Self::extract_roles_with_kb(tokens, predicate, voice, &kb)
    }

    /// Compute frame confidence based on role coverage using KnowledgeBase.
    ///
    /// All bonuses and penalties are read from the KnowledgeBase's adaptive
    /// parameters, enabling self-calibration over time.
    ///
    /// The result is clamped to [0.0, 1.0].
    pub fn compute_frame_confidence_with_kb(
        roles: &HashMap<SemanticRole, String>,
        polarity: &Polarity,
        kb: &KnowledgeBase,
    ) -> f32 {
        let mut confidence = kb.param("extract.base_confidence", 0.30);

        if roles.contains_key(&SemanticRole::Arg0Agent) {
            confidence += kb.param("extract.agent_bonus", 0.15);
        }
        if roles.contains_key(&SemanticRole::Arg1Patient) {
            confidence += kb.param("extract.patient_bonus", 0.15);
        }
        if roles.contains_key(&SemanticRole::Cause) {
            confidence += kb.param("extract.cause_bonus", 0.10);
        }
        if roles.contains_key(&SemanticRole::Purpose) {
            confidence += kb.param("extract.purpose_bonus", 0.10);
        }
        if roles.contains_key(&SemanticRole::Antecedent) {
            confidence += kb.param("extract.antecedent_bonus", 0.10);
        }
        if roles.contains_key(&SemanticRole::Consequent) {
            confidence += kb.param("extract.consequent_bonus", 0.10);
        }
        if *polarity == Polarity::Negative {
            confidence -= kb.param("extract.negation_penalty", 0.05);
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Backward-compatible wrapper using bootstrapped KB.
    #[deprecated(note = "Use compute_frame_confidence_with_kb instead")]
    pub fn compute_frame_confidence(
        roles: &HashMap<SemanticRole, String>,
        polarity: &Polarity,
    ) -> f32 {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        Self::compute_frame_confidence_with_kb(roles, polarity, &kb)
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

    /// Classify extraction quality based on roles and confidence, using KB thresholds.
    ///
    /// Quality thresholds are read from AdaptiveParams:
    /// - `extract.high_quality_threshold` (default 0.70)
    /// - `extract.moderate_quality_threshold` (default 0.45)
    pub fn classify_quality_with_kb(
        roles: &HashMap<SemanticRole, String>,
        confidence: f32,
        kb: &KnowledgeBase,
    ) -> ExtractionQuality {
        let high_threshold = kb.param("extract.high_quality_threshold", 0.70);
        let moderate_threshold = kb.param("extract.moderate_quality_threshold", 0.45);

        if confidence >= high_threshold
            && roles.contains_key(&SemanticRole::Arg0Agent)
            && roles.contains_key(&SemanticRole::Arg1Patient)
        {
            ExtractionQuality::HighQuality
        } else if confidence >= moderate_threshold
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

    /// Extract a semantic frame using Action Schemas when available (KB-based).
    ///
    /// This method checks Action Schemas (Phase 1) before falling back to
    /// generic Event extraction. When a schema's trigger matches, the
    /// composition type and role bindings come from the schema — producing
    /// EquativeBinding, PossessiveBinding, etc. instead of generic Event.
    ///
    /// # Algorithm
    ///
    /// 1. Tokenize input
    /// 2. Try schemas in priority order (highest first)
    /// 3. If a schema matches, create atom with schema's composition type + roles
    /// 4. If no schema matches, fall back to generic Event extraction
    pub fn extract_with_schemas_and_kb(
        &self,
        text: &str,
        schemas: &[super::action_schemas::ActionSchema],
        kb: &KnowledgeBase,
    ) -> Option<SemanticAtom> {
        if !Self::is_sentence_like_with_kb(text, kb) {
            return None;
        }

        let tokens: Vec<&str> = text.split_whitespace().collect();

        // Try schemas in priority order (highest first).
        let mut sorted_schemas: Vec<_> = schemas.iter().collect();
        sorted_schemas.sort_by(|a, b| b.priority.cmp(&a.priority));

        for schema in sorted_schemas {
            if let Some(trigger_idx) = schema.matches_tokens_with_knowledge(&tokens, kb) {
                let roles = schema.resolve_roles_with_knowledge(&tokens, trigger_idx, kb);
                if !roles.is_empty() {
                    let trigger_token = tokens[trigger_idx];

                    // Compute confidence based on role coverage using KB params.
                    let mut confidence = kb.param("extract.schema_base_confidence", 0.35);
                    confidence += kb.param("extract.schema_role_bonus", 0.15) * roles.len() as f32;
                    confidence = confidence.clamp(0.0, 1.0);

                    let roles_map: HashMap<SemanticRole, String> = roles.into_iter().collect();

                    // Determine if we need the Predicate role.
                    // For non-Event types, the trigger token is the binding marker,
                    // not a predicate in the traditional sense.
                    let atom_label = trigger_token.to_lowercase();

                    let atom_type = if schema.composition_type == CompositionType::Event {
                        AtomType::Event
                    } else {
                        // EquativeBinding, PossessiveBinding, etc. are still Event atoms
                        // in the pipeline (they carry roles), but their composition_type
                        // determines the final composition type.
                        AtomType::Event
                    };

                    let mut atom = SemanticAtom {
                        id: String::new(),
                        label: atom_label,
                        atom_type,
                        roles: roles_map,
                        polarity: Some(Polarity::Positive),
                        voice: Some(Voice::Active),
                        variant: Some(AtomVariant::FrameVariant(FrameSource::RuleBased)),
                        confidence,
                        source: EdgeSource::ActionSchemaExtraction,
                        composition_id: None,
                    };

                    // Store the intended composition type in a special role
                    // so IngestAtoms can create the correct composition type.
                    atom.roles.insert(
                        SemanticRole::PatternType,
                        format!("{:?}", schema.composition_type),
                    );

                    return Some(atom);
                }
            }
        }

        // No schema matched — fall back to generic Event extraction.
        self.extract_with_kb(text, kb)
    }

    /// Backward-compatible wrapper using bootstrapped KB.
    #[deprecated(note = "Use extract_with_schemas_and_kb instead")]
    pub fn extract_with_schemas(
        &self,
        text: &str,
        schemas: &[super::action_schemas::ActionSchema],
    ) -> Option<SemanticAtom> {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        self.extract_with_schemas_and_kb(text, schemas, &kb)
    }

    /// Extract a semantic frame from raw text using KnowledgeBase.
    ///
    /// This is the core extraction method. Returns `Some(SemanticAtom)` if
    /// the input is sentence-like and a predicate can be found, `None` otherwise.
    pub fn extract_with_kb(&self, text: &str, kb: &KnowledgeBase) -> Option<SemanticAtom> {
        if !Self::is_sentence_like_with_kb(text, kb) {
            return None;
        }

        let tokens: Vec<&str> = text.split_whitespace().collect();
        let voice = Self::detect_voice(&tokens);
        let polarity = Self::detect_polarity_with_kb(&tokens, kb);

        let predicate = Self::extract_predicate_with_kb(&tokens, &voice, kb)?;
        let roles = Self::extract_roles_with_kb(&tokens, predicate, &voice, kb);
        let confidence = Self::compute_frame_confidence_with_kb(&roles, &polarity, kb);

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

    /// Backward-compatible wrapper using bootstrapped KB.
    #[deprecated(note = "Use extract_with_kb instead")]
    pub fn extract(&self, text: &str) -> Option<SemanticAtom> {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        self.extract_with_kb(text, &kb)
    }

    /// Graph-assisted re-extraction with context hints (KB-based).
    ///
    /// When the feedback loop identifies a weak frame, it can request
    /// re-extraction with graph context (known role fillers from related
    /// compositions). This method re-runs extraction and merges any
    /// graph-provided context into the roles.
    ///
    /// The `FrameSource` is set to `GraphAssisted` to mark that this
    /// extraction used graph context.
    pub fn re_extract_with_context_and_kb(
        &self,
        text: &str,
        graph_context: &[(SemanticRole, NodeId, f32)],
        graph: &Graph,
        kb: &KnowledgeBase,
    ) -> Option<SemanticAtom> {
        let mut atom = self.extract_with_kb(text, kb)?;

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
        atom.confidence = Self::compute_frame_confidence_with_kb(&atom.roles, polarity, kb);

        // Mark as graph-assisted.
        atom.variant = Some(AtomVariant::FrameVariant(FrameSource::GraphAssisted));

        Some(atom)
    }

    /// Backward-compatible wrapper using bootstrapped KB.
    #[deprecated(note = "Use re_extract_with_context_and_kb instead")]
    pub fn re_extract_with_context(
        &self,
        text: &str,
        graph_context: &[(SemanticRole, NodeId, f32)],
        graph: &Graph,
    ) -> Option<SemanticAtom> {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        self.re_extract_with_context_and_kb(text, graph_context, graph, &kb)
    }
}

/// Implement the `Transform` trait for `ExtractFrame`.
impl Transform for ExtractFrame {
    type Input = String;
    type Output = Option<SemanticAtom>;

    fn id(&self) -> &'static str {
        "ExtractFrame"
    }

    fn transform(&self, input: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        self.extract_with_kb(input, &ctx.knowledge_base)
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
        if !Self::is_sentence_like_with_kb(&text, &ctx.knowledge_base) {
            // Update quality tracker.
            ctx.extraction_quality.low_confidence_frames += 1;
            // Audit v5 fix (D14): Also update the per-quality-level tracker.
            ctx.extraction_quality_ext.record(&ExtractionQuality::Failed, 0.0);
            return IngestResult::new();
        }

        // Try schema-driven extraction first, then fall back to generic.
        let atom_result = if !ctx.active_schemas.is_empty() {
            self.extract_with_schemas_and_kb(&text, &ctx.active_schemas, &ctx.knowledge_base)
        } else {
            self.extract_with_kb(&text, &ctx.knowledge_base)
        };

        if let Some(mut atom) = atom_result {
            atom.id = format!("atom_{}", ctx.next_atom_id());
            ctx.current_atoms.push(atom.clone());
            atoms_created += 1;

            ctx.extraction_quality.frames_extracted += 1;
            ctx.extraction_quality.average_confidence = (ctx.extraction_quality.average_confidence
                * (ctx.extraction_quality.frames_extracted - 1) as f32
                + atom.confidence)
                / ctx.extraction_quality.frames_extracted as f32;

            let quality = Self::classify_quality_with_kb(&atom.roles, atom.confidence, &ctx.knowledge_base);
            ctx.extraction_quality_ext.record(&quality, atom.confidence);

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
// Helper Functions — KnowledgeBase-based
// ========================================================================

/// Is a token verb-like? (KB-based)
///
/// A token is verb-like if it:
/// - Starts with a known verb prefix (from KnowledgeBase), OR
/// - Is a known common verb (from KnowledgeBase CommonVerb markers)
fn is_verb_like_with_kb(token: &str, kb: &KnowledgeBase) -> bool {
    let lower = token.to_lowercase();

    // Check verb prefixes from KB.
    if kb.is_verb_prefix_match(&lower) {
        return true;
    }

    // Check common verb list from KB.
    if kb.is_marker(&MarkerCategory::CommonVerb, &lower) {
        return true;
    }

    false
}

/// Backward-compatible wrapper using bootstrapped KB.
#[deprecated(note = "Use is_verb_like_with_kb instead")]
#[allow(dead_code)]
fn is_verb_like(token: &str) -> bool {
    let kb = crate::v12::knowledge_base::create_indonesian_seeded();
    is_verb_like_with_kb(token, &kb)
}

/// Is a token a marker (negation, cause, purpose, condition)? (KB-based)
fn is_marker_with_kb(token: &str, kb: &KnowledgeBase) -> bool {
    kb.is_any_role_marker(token)
}

/// Backward-compatible wrapper using bootstrapped KB.
#[deprecated(note = "Use is_marker_with_kb instead")]
#[allow(dead_code)]
fn is_marker(token: &str) -> bool {
    let kb = crate::v12::knowledge_base::create_indonesian_seeded();
    is_marker_with_kb(token, &kb)
}

/// Is a token a negation marker? (KB-based)
fn is_negation_marker_with_kb(token: &str, kb: &KnowledgeBase) -> bool {
    kb.is_marker(&MarkerCategory::Negation, token)
}

/// Backward-compatible wrapper using bootstrapped KB.
#[deprecated(note = "Use is_negation_marker_with_kb instead")]
#[allow(dead_code)]
fn is_negation_marker(token: &str) -> bool {
    let kb = crate::v12::knowledge_base::create_indonesian_seeded();
    is_negation_marker_with_kb(token, &kb)
}

/// Is a token a cause marker? (KB-based)
fn is_cause_marker_with_kb(token: &str, kb: &KnowledgeBase) -> bool {
    kb.is_marker(&MarkerCategory::Cause, token)
}

/// Backward-compatible wrapper using bootstrapped KB.
#[deprecated(note = "Use is_cause_marker_with_kb instead")]
#[allow(dead_code)]
fn is_cause_marker(token: &str) -> bool {
    let kb = crate::v12::knowledge_base::create_indonesian_seeded();
    is_cause_marker_with_kb(token, &kb)
}

/// Is a token a purpose marker? (KB-based)
fn is_purpose_marker_with_kb(token: &str, kb: &KnowledgeBase) -> bool {
    kb.is_marker(&MarkerCategory::Purpose, token)
}

/// Backward-compatible wrapper using bootstrapped KB.
#[deprecated(note = "Use is_purpose_marker_with_kb instead")]
#[allow(dead_code)]
fn is_purpose_marker(token: &str) -> bool {
    let kb = crate::v12::knowledge_base::create_indonesian_seeded();
    is_purpose_marker_with_kb(token, &kb)
}

/// Is a token a conditional marker? (KB-based)
fn is_condition_marker_with_kb(token: &str, kb: &KnowledgeBase) -> bool {
    kb.is_marker(&MarkerCategory::Condition, token)
}

/// Backward-compatible wrapper using bootstrapped KB.
#[deprecated(note = "Use is_condition_marker_with_kb instead")]
#[allow(dead_code)]
fn is_condition_marker(token: &str) -> bool {
    let kb = crate::v12::knowledge_base::create_indonesian_seeded();
    is_condition_marker_with_kb(token, &kb)
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kb() -> KnowledgeBase {
        crate::v12::knowledge_base::create_indonesian_seeded()
    }

    #[test]
    fn test_is_sentence_like() {
        let kb = make_kb();

        // Too short.
        assert!(!ExtractFrame::is_sentence_like_with_kb("hello", &kb));
        assert!(!ExtractFrame::is_sentence_like_with_kb("dua kata", &kb));

        // Repetitive.
        assert!(!ExtractFrame::is_sentence_like_with_kb("satu satu satu", &kb));

        // No verb.
        assert!(!ExtractFrame::is_sentence_like_with_kb("kucing besar hitam", &kb));

        // Valid sentence with verb.
        assert!(ExtractFrame::is_sentence_like_with_kb(
            "Raymond membuat aplikasi karena lambat",
            &kb
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
        let kb = make_kb();
        let tokens: Vec<&str> = "Raymond membuat aplikasi".split_whitespace().collect();
        assert_eq!(ExtractFrame::detect_polarity_with_kb(&tokens, &kb), Polarity::Positive);
    }

    #[test]
    fn test_detect_polarity_negative() {
        let kb = make_kb();
        let tokens: Vec<&str> = "Raymond tidak membuat aplikasi"
            .split_whitespace()
            .collect();
        assert_eq!(ExtractFrame::detect_polarity_with_kb(&tokens, &kb), Polarity::Negative);
    }

    #[test]
    fn test_extract_predicate() {
        let kb = make_kb();
        let tokens: Vec<&str> = "Raymond membuat aplikasi".split_whitespace().collect();
        let pred = ExtractFrame::extract_predicate_with_kb(&tokens, &Voice::Active, &kb);
        assert_eq!(pred, Some("membuat"));
    }

    #[test]
    fn test_compute_frame_confidence() {
        let kb = make_kb();
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "raymond".to_string());
        roles.insert(SemanticRole::Arg1Patient, "aplikasi".to_string());

        let confidence = ExtractFrame::compute_frame_confidence_with_kb(&roles, &Polarity::Positive, &kb);
        // 0.30 + 0.15 + 0.15 = 0.60
        assert!((confidence - 0.60).abs() < 0.01);

        // With Cause: 0.30 + 0.15 + 0.15 + 0.10 = 0.70
        roles.insert(SemanticRole::Cause, "lambat".to_string());
        let confidence = ExtractFrame::compute_frame_confidence_with_kb(&roles, &Polarity::Positive, &kb);
        assert!((confidence - 0.70).abs() < 0.01);

        // Negative: 0.70 - 0.05 = 0.65
        let confidence = ExtractFrame::compute_frame_confidence_with_kb(&roles, &Polarity::Negative, &kb);
        assert!((confidence - 0.65).abs() < 0.01);
    }

    #[test]
    fn test_full_extraction() {
        let ef = ExtractFrame::new();
        let kb = make_kb();
        let result = ef.extract_with_kb("Raymond membuat aplikasi karena lambat", &kb);

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

    #[test]
    fn test_extract_with_schemas_copula() {
        let ef = ExtractFrame::new();
        let schemas = super::super::action_schemas::bootstrap_schemas();
        let kb = make_kb();
        let result = ef.extract_with_schemas_and_kb("ini adalah makanan", &schemas, &kb);

        assert!(result.is_some());
        let atom = result.unwrap();
        // Should have Subject and Complement roles from the copula schema
        assert!(atom.roles.contains_key(&SemanticRole::Subject));
        assert!(atom.roles.contains_key(&SemanticRole::Complement));
        assert_eq!(atom.source, EdgeSource::ActionSchemaExtraction);
        // Should have PatternType role hinting at EquativeBinding
        assert_eq!(
            atom.roles.get(&SemanticRole::PatternType).map(|s| s.as_str()),
            Some("EquativeBinding")
        );
    }

    #[test]
    fn test_extract_with_schemas_possessive() {
        let ef = ExtractFrame::new();
        let schemas = super::super::action_schemas::bootstrap_schemas();
        let kb = make_kb();
        let result = ef.extract_with_schemas_and_kb("raja punya kerajaan", &schemas, &kb);

        assert!(result.is_some());
        let atom = result.unwrap();
        assert!(atom.roles.contains_key(&SemanticRole::Possessor));
        assert!(atom.roles.contains_key(&SemanticRole::Possession));
        assert_eq!(atom.source, EdgeSource::ActionSchemaExtraction);
        // Should have PatternType role hinting at PossessiveBinding
        assert_eq!(
            atom.roles.get(&SemanticRole::PatternType).map(|s| s.as_str()),
            Some("PossessiveBinding")
        );
    }

    #[test]
    fn test_extract_with_schemas_fallback() {
        let ef = ExtractFrame::new();
        let schemas = super::super::action_schemas::bootstrap_schemas();
        let kb = make_kb();
        // No schema matches "Raymond membuat aplikasi karena lambat"
        let result = ef.extract_with_schemas_and_kb("Raymond membuat aplikasi karena lambat", &schemas, &kb);

        assert!(result.is_some());
        let atom = result.unwrap();
        // Should fall back to generic Event extraction
        assert_eq!(atom.source, EdgeSource::FrameCompiler);
        assert!(atom.roles.contains_key(&SemanticRole::Arg0Agent));
        // Should NOT have a PatternType role (it's not schema-driven)
        assert!(!atom.roles.contains_key(&SemanticRole::PatternType));
    }

    #[test]
    fn test_extract_with_schemas_empty() {
        let ef = ExtractFrame::new();
        let kb = make_kb();
        // Empty schemas should fall back to generic extraction
        let result = ef.extract_with_schemas_and_kb("Raymond membuat aplikasi karena lambat", &[], &kb);
        assert!(result.is_some());
        let atom = result.unwrap();
        assert_eq!(atom.source, EdgeSource::FrameCompiler);
    }
}
