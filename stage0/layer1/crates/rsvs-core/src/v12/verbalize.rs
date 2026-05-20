//! # Compositional Verbalization Engine (CVE)
//!
//! Graph-Driven Self-Explanation without LLM.
//!
//! Instead of generating tokens one-by-one like an LLM, CVE traverses the
//! knowledge graph and converts each composition it encounters into a natural
//! language sentence using deterministic templates per `CompositionType`.
//!
//! ## Architecture
//!
//! ```text
//! Query → SpreadingActivation → Graph (relevant subset)
//!       → Traversal (reasoning path) → Verbalize → Text
//! ```
//!
//! ## Key Properties
//!
//! 1. **Zero hallucination by design** — CVE cannot produce text about anything
//!    not present in the graph. If the graph has no relevant information, the
//!    output is: "Tidak ada informasi yang cukup untuk menjelaskan ini."
//!
//! 2. **Fully replayable** — Output includes the composition path, so anyone can
//!    verify each sentence against the graph: `comp_4 → comp_1 → comp_2 → comp_3`
//!
//! 3. **Confidence-aware** — Sentences from `Stable/Grounded` compositions have
//!    no qualifier. Sentences from `Candidate/Inferred` get epistemic prefixes.
//!
//! 4. **No weights required** — No GPU, no API call. Runs in microseconds because
//!    it is pure graph traversal + template substitution.
//!
//! ## Templates (Bahasa Indonesia)
//!
//! | CompositionType | Template |
//! |-----------------|----------|
//! | Event | `[Agent] [Predicate] [Patient], karena [Cause], untuk [Purpose]` |
//! | HiddenMeaning | `[Solution] digunakan sebagai solusi untuk [Problem]` |
//! | Pattern | `Ketika [Antecedent], maka [Consequent]` |
//! | Hypothesis | `Kemungkinan [Patient] [Predicate]` |
//! | Situation | `Dalam konteks [Arg0Agent], [Predicate] [Arg1Patient]` |
//! | Acquisition | `Diketahui bahwa [Arg0Agent] [Predicate] [Arg1Patient]` |

use serde::{Deserialize, Serialize};

use super::pipeline::{ErasedTransform, Graph, IngestResult};
use super::spreading::{ActivationMap, SpreadingActivation, SpreadingConfig};
use super::types::*;
use crate::types::NodeId;

// ========================================================================
// VerbalizationResult — Output of a CVE query
// ========================================================================

/// The result of a compositional verbalization query.
///
/// Contains the generated explanation text, the reasoning path (ordered list
/// of composition IDs), aggregate confidence, and epistemic statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerbalizationResult {
    /// The generated explanation text (one or more sentences).
    pub text: String,
    /// Ordered composition IDs that were traversed to produce the text.
    /// This is the "reasoning path" — fully replayable for audit.
    pub path: Vec<CompositionId>,
    /// Average confidence across all compositions in the path.
    pub avg_confidence: f32,
    /// Number of compositions in Stable/Grounded state.
    pub stable_grounded_count: usize,
    /// Number of compositions in Candidate/Inferred state.
    pub candidate_inferred_count: usize,
    /// Total number of compositions used.
    pub total_compositions: usize,
}

impl Default for VerbalizationResult {
    fn default() -> Self {
        Self {
            text: String::new(),
            path: Vec::new(),
            avg_confidence: 0.0,
            stable_grounded_count: 0,
            candidate_inferred_count: 0,
            total_compositions: 0,
        }
    }
}

impl VerbalizationResult {
    /// Create an "insufficient information" result.
    ///
    /// Returned when the graph has no relevant compositions for the query.
    pub fn insufficient() -> Self {
        Self {
            text: "Tidak ada informasi yang cukup untuk menjelaskan ini.".to_string(),
            path: Vec::new(),
            avg_confidence: 0.0,
            stable_grounded_count: 0,
            candidate_inferred_count: 0,
            total_compositions: 0,
        }
    }
}

// ========================================================================
// VerbalizeConfig — Configuration
// ========================================================================

/// Configuration for the Compositional Verbalization Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerbalizeConfig {
    /// Minimum activation energy for a composition to be included in the path.
    /// Compositions whose activation energy falls below this threshold are
    /// considered irrelevant to the query. Default: 0.05.
    pub min_activation: f32,
    /// Maximum number of compositions to include in the reasoning path.
    /// Prevents overly long explanations. Default: 20.
    pub max_path_length: usize,
    /// Whether to include the audit footer (confidence, source count, path).
    pub include_audit_footer: bool,
    /// Spreading activation configuration (used internally for query activation).
    pub spreading_config: SpreadingConfig,
}

impl Default for VerbalizeConfig {
    fn default() -> Self {
        Self {
            min_activation: 0.05,
            max_path_length: 20,
            include_audit_footer: true,
            spreading_config: SpreadingConfig::default(),
        }
    }
}

// ========================================================================
// CompositionActivation — Scored composition for path ordering
// ========================================================================

/// A composition scored by activation energy for path ordering.
#[derive(Debug, Clone)]
struct CompositionActivation {
    comp_id: CompositionId,
    /// Composite score: activation + lifecycle/epistemic bonus.
    score: f32,
}

// ========================================================================
// CompositionalVerbalize — The Engine
// ========================================================================

/// Compositional Verbalization Engine (CVE).
///
/// Takes a query string, activates relevant nodes via SpreadingActivation,
/// traverses the graph to build a reasoning path, and verbalizes each
/// composition along the path using deterministic templates.
///
/// # Usage
///
/// ```ignore
/// let cve = CompositionalVerbalize::new();
/// let result = cve.explain("Kenapa aplikasi lambat", &graph);
/// println!("{}", result.text);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionalVerbalize {
    /// Configuration for the engine.
    pub config: VerbalizeConfig,
    /// The underlying spreading activation engine.
    pub spreading: SpreadingActivation,
}

impl Default for CompositionalVerbalize {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositionalVerbalize {
    /// Create a new CVE with default configuration.
    pub fn new() -> Self {
        Self {
            config: VerbalizeConfig::default(),
            spreading: SpreadingActivation::new(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: VerbalizeConfig) -> Self {
        let spreading = SpreadingActivation::with_config(config.spreading_config.clone());
        Self { config, spreading }
    }

    // ================================================================
    // Step 1: Query Activation
    // ================================================================

    /// Activate nodes relevant to the query string.
    ///
    /// Extracts keywords from the query, finds matching nodes in the graph,
    /// then runs SpreadingActivation from those seed nodes.
    fn activate_query(&self, query: &str, graph: &Graph) -> ActivationMap {
        let keywords = extract_keywords(query);
        let mut seeds: Vec<(NodeId, f32)> = Vec::new();

        for keyword in &keywords {
            if let Some(node_id) = graph.find_node_by_label(keyword) {
                seeds.push((node_id, 1.0));
            }
        }

        if seeds.is_empty() {
            // Fallback: try partial match on all node labels
            for keyword in &keywords {
                let keyword_lower = keyword.to_lowercase();
                for (label, &node_id) in &graph.label_to_id {
                    if label.to_lowercase().contains(&keyword_lower) {
                        seeds.push((node_id, 0.7));
                    }
                }
            }
        }

        if seeds.is_empty() {
            return ActivationMap::new();
        }

        self.spreading.spread(&seeds, graph)
    }

    // ================================================================
    // Step 2: Build Reasoning Path
    // ================================================================

    /// Build an ordered reasoning path from activated compositions.
    ///
    /// Compositions are scored by:
    /// - Activation energy (from SpreadingActivation)
    /// - Lifecycle bonus: Stable (+0.2), Candidate (+0.1)
    /// - Epistemic bonus: Grounded (+0.2), Inferred (+0.1)
    /// - Confidence bonus: proportion of confidence
    fn build_reasoning_path(
        &self,
        activation_map: &ActivationMap,
        graph: &Graph,
    ) -> Vec<CompositionActivation> {
        let mut scored: Vec<CompositionActivation> = Vec::new();

        for composition in graph.compositions.values() {
            // Compute maximum activation energy across all member nodes.
            let max_activation = composition
                .members
                .iter()
                .map(|m| activation_map.energy(m.node_id))
                .fold(0.0f32, f32::max);

            if max_activation < self.config.min_activation {
                continue;
            }

            // Composite score: activation + lifecycle/epistemic bonus + confidence
            let lifecycle_bonus = match composition.lifecycle {
                LifecycleState::Stable => 0.2,
                LifecycleState::Candidate => 0.1,
                LifecycleState::New => 0.05,
                LifecycleState::Deprecated => -0.1,
                LifecycleState::Quarantine => -0.2,
            };

            let epistemic_bonus = match composition.epistemic {
                EpistemicState::Grounded => 0.2,
                EpistemicState::Inferred => 0.1,
                EpistemicState::Observed => 0.05,
                EpistemicState::Hypothesis => 0.0,
                EpistemicState::Contradicted => -0.1,
            };

            let confidence_bonus = composition.confidence * 0.2;

            let score = max_activation + lifecycle_bonus + epistemic_bonus + confidence_bonus;

            scored.push(CompositionActivation {
                comp_id: composition.id.clone(),
                score,
            });
        }

        // Sort by score descending (most relevant first).
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to max path length.
        scored.truncate(self.config.max_path_length);

        scored
    }

    // ================================================================
    // Step 3: Verbalize per CompositionType
    // ================================================================

    /// Verbalize a single composition into a sentence.
    ///
    /// Uses deterministic templates per `CompositionType` in Bahasa Indonesia.
    /// Missing roles are handled gracefully with default fillers.
    fn verbalize_composition(&self, comp: &Composition) -> String {
        match comp.composition_type {
            CompositionType::Event => self.verbalize_event(comp),
            CompositionType::HiddenMeaning => self.verbalize_hidden_meaning(comp),
            CompositionType::Pattern => self.verbalize_pattern(comp),
            CompositionType::Hypothesis => self.verbalize_hypothesis(comp),
            CompositionType::Situation => self.verbalize_situation(comp),
            CompositionType::Acquisition => self.verbalize_acquisition(comp),
        }
    }

    /// Event template: "AGENT did ACTION to PATIENT because CAUSE"
    ///
    /// Falls back to Bahasa Indonesia: "[Agent] [Predicate] [Patient], karena [Cause]"
    fn verbalize_event(&self, comp: &Composition) -> String {
        let agent = comp
            .member_with_role(&SemanticRole::Arg0Agent)
            .map(|m| m.label.as_str())
            .unwrap_or("Sesuatu");
        let pred = comp
            .member_with_role(&SemanticRole::Predicate)
            .map(|m| m.label.as_str())
            .unwrap_or("terjadi");
        let patient = comp
            .member_with_role(&SemanticRole::Arg1Patient)
            .map(|m| format!(" {}", m.label))
            .unwrap_or_default();
        let cause = comp
            .member_with_role(&SemanticRole::Cause)
            .map(|m| format!(", karena {}", m.label))
            .unwrap_or_default();
        let purpose = comp
            .member_with_role(&SemanticRole::Purpose)
            .map(|m| format!(", untuk {}", m.label))
            .unwrap_or_default();
        let location = comp
            .member_with_role(&SemanticRole::Location)
            .map(|m| format!(", di {}", m.label))
            .unwrap_or_default();
        let time = comp
            .member_with_role(&SemanticRole::Time)
            .map(|m| format!(", saat {}", m.label))
            .unwrap_or_default();
        let instrument = comp
            .member_with_role(&SemanticRole::Instrument)
            .map(|m| format!(", dengan {}", m.label))
            .unwrap_or_default();

        format!(
            "{} {}{}{}{}{}{}{}.",
            agent, pred, patient, cause, purpose, location, time, instrument
        )
    }

    /// HiddenMeaning template: "PROBLEM led to SOLUTION by AGENT"
    ///
    /// Falls back to Bahasa: "[Solution] digunakan sebagai solusi untuk [Problem]"
    fn verbalize_hidden_meaning(&self, comp: &Composition) -> String {
        let problem = comp
            .member_with_role(&SemanticRole::Problem)
            .map(|m| m.label.as_str())
            .unwrap_or("masalah ini");
        let solution = comp
            .member_with_role(&SemanticRole::Solution)
            .map(|m| m.label.as_str())
            .unwrap_or("solusi");
        let agent = comp
            .member_with_role(&SemanticRole::Arg0Agent)
            .map(|m| format!(" oleh {}", m.label))
            .unwrap_or_default();
        let beneficiary = comp
            .member_with_role(&SemanticRole::Beneficiary)
            .map(|m| format!(", yang menguntungkan {}", m.label))
            .unwrap_or_default();
        let motivation = comp
            .member_with_role(&SemanticRole::Motivation)
            .map(|m| format!(", karena {}", m.label))
            .unwrap_or_default();

        format!(
            "{} digunakan sebagai solusi untuk {}{}{}{}.",
            problem, solution, agent, motivation, beneficiary
        )
    }

    /// Pattern template: "When ANTECEDENT then CONSEQUENT"
    fn verbalize_pattern(&self, comp: &Composition) -> String {
        let ante = comp
            .member_with_role(&SemanticRole::Antecedent)
            .map(|m| m.label.as_str())
            .unwrap_or("kondisi ini");
        let cons = comp
            .member_with_role(&SemanticRole::Consequent)
            .map(|m| m.label.as_str())
            .unwrap_or("hasil ini");
        let pattern_type = comp
            .member_with_role(&SemanticRole::PatternType)
            .map(|m| format!(" [{}]", m.label))
            .unwrap_or_default();

        format!(
            "Ketika {}, maka{} {}.",
            ante,
            pattern_type,
            cons
        )
    }

    /// Hypothesis template: "Kemungkinan [Patient] [Predicate]"
    fn verbalize_hypothesis(&self, comp: &Composition) -> String {
        let pred = comp
            .member_with_role(&SemanticRole::Predicate)
            .map(|m| m.label.as_str())
            .unwrap_or("terjadi");
        let patient = comp
            .member_with_role(&SemanticRole::Arg1Patient)
            .map(|m| m.label.as_str())
            .unwrap_or("ini");
        let cause = comp
            .member_with_role(&SemanticRole::Cause)
            .map(|m| format!(", karena {}", m.label))
            .unwrap_or_default();

        format!("Kemungkinan {} {}{}.", patient, pred, cause)
    }

    /// Situation template: "Dalam konteks [Agent], [Predicate] [Patient]"
    fn verbalize_situation(&self, comp: &Composition) -> String {
        let agent = comp
            .member_with_role(&SemanticRole::Arg0Agent)
            .map(|m| m.label.as_str())
            .unwrap_or("konteks ini");
        let pred = comp
            .member_with_role(&SemanticRole::Predicate)
            .map(|m| m.label.as_str())
            .unwrap_or("terjadi");
        let patient = comp
            .member_with_role(&SemanticRole::Arg1Patient)
            .map(|m| format!(" {}", m.label))
            .unwrap_or_default();

        format!("Dalam konteks {}, {}{}.", agent, pred, patient)
    }

    /// Acquisition template: "Diketahui bahwa [Agent] [Predicate] [Patient]"
    fn verbalize_acquisition(&self, comp: &Composition) -> String {
        let agent = comp
            .member_with_role(&SemanticRole::Arg0Agent)
            .map(|m| m.label.as_str())
            .unwrap_or("sumber");
        let pred = comp
            .member_with_role(&SemanticRole::Predicate)
            .map(|m| m.label.as_str())
            .unwrap_or("menyatakan");
        let patient = comp
            .member_with_role(&SemanticRole::Arg1Patient)
            .map(|m| format!(" {}", m.label))
            .unwrap_or_default();
        let tool = comp
            .member_with_role(&SemanticRole::Tool)
            .map(|m| format!(", melalui {}", m.label))
            .unwrap_or_default();

        format!("Diketahui bahwa {} {}{}{}.", agent, pred, patient, tool)
    }

    // ================================================================
    // Step 4: Epistemic Qualifier
    // ================================================================

    /// Apply an epistemic qualifier prefix based on lifecycle, epistemic, and confidence.
    ///
    /// | Lifecycle | Epistemic | Confidence | Qualifier |
    /// |-----------|-----------|------------|-----------|
    /// | Stable | Grounded | > 0.8 | "" (no prefix) |
    /// | Stable | Grounded | ≤ 0.8 | "Tampaknya, " |
    /// | Candidate | Inferred | any | "Berdasarkan analisis, " |
    /// | any | Hypothesis | any | "Kemungkinan besar, " |
    /// | any | Contradicted | any | "Meskipun ada kontradiksi, " |
    /// | Quarantine | any | any | "Perlu ditinjau kembali, " |
    /// | Deprecated | any | any | "Sebelumnya diyakini, " |
    /// | default | default | any | "Kemungkinan, " |
    fn qualify(&self, sentence: &str, comp: &Composition) -> String {
        let qualifier = match (&comp.lifecycle, &comp.epistemic, comp.confidence) {
            // Established, high confidence → no qualifier
            (LifecycleState::Stable, EpistemicState::Grounded, c) if c > 0.8 => "",
            // Stable + Grounded but lower confidence
            (LifecycleState::Stable, EpistemicState::Grounded, _) => "Tampaknya, ",
            // Still a candidate, derived by rule
            (LifecycleState::Candidate, EpistemicState::Inferred, _) => "Berdasarkan analisis, ",
            // New observation
            (LifecycleState::New, EpistemicState::Observed, _) => "Berdasarkan observasi, ",
            // Hypothesis
            (_, EpistemicState::Hypothesis, _) => "Kemungkinan besar, ",
            // Contradicted — still include for transparency
            (_, EpistemicState::Contradicted, _) => "Meskipun ada kontradiksi, ",
            // Quarantine — needs review
            (LifecycleState::Quarantine, _, _) => "Perlu ditinjau kembali, ",
            // Deprecated — was previously believed
            (LifecycleState::Deprecated, _, _) => "Sebelumnya diyakini, ",
            // Default fallback
            _ => "Kemungkinan, ",
        };
        format!("{}{}", qualifier, sentence)
    }

    // ================================================================
    // Step 5: Compose Output
    // ================================================================

    /// Compose the final explanation from a reasoning path.
    ///
    /// The output includes:
    /// - Each composition verbalized and qualified
    /// - An optional audit footer with confidence, source counts, and path
    fn compose_output(&self, path: &[CompositionActivation], graph: &Graph) -> VerbalizationResult {
        if path.is_empty() {
            return VerbalizationResult::insufficient();
        }

        let mut sentences: Vec<String> = Vec::new();
        let mut comp_ids: Vec<CompositionId> = Vec::new();
        let mut total_confidence = 0.0f32;
        let mut stable_grounded = 0usize;
        let mut candidate_inferred = 0usize;

        for ca in path {
            if let Some(comp) = graph.get_composition(&ca.comp_id) {
                let raw_sentence = self.verbalize_composition(comp);
                let qualified = self.qualify(&raw_sentence, comp);
                sentences.push(qualified);

                comp_ids.push(ca.comp_id.clone());
                total_confidence += comp.confidence;

                if comp.lifecycle == LifecycleState::Stable
                    && comp.epistemic == EpistemicState::Grounded
                {
                    stable_grounded += 1;
                }
                if comp.lifecycle == LifecycleState::Candidate
                    && comp.epistemic == EpistemicState::Inferred
                {
                    candidate_inferred += 1;
                }
            }
        }

        let total = comp_ids.len();
        if total == 0 {
            return VerbalizationResult::insufficient();
        }

        let avg_confidence = total_confidence / total as f32;

        // Build the text body.
        let mut text = sentences.join(" ");

        // Append audit footer if configured.
        if self.config.include_audit_footer && total > 0 {
            let avg_pct = (avg_confidence * 100.0) as usize;
            let path_str = comp_ids.join(" → ");
            text.push_str(&format!(
                "\n\n[Keyakinan rata-rata: {}%]\n[Sumber: {} komposisi, {} Stable/Grounded, {} Candidate/Inferred]\n[Dapat diaudit: {}]",
                avg_pct,
                total,
                stable_grounded,
                candidate_inferred,
                path_str,
            ));
        }

        VerbalizationResult {
            text,
            path: comp_ids,
            avg_confidence,
            stable_grounded_count: stable_grounded,
            candidate_inferred_count: candidate_inferred,
            total_compositions: total,
        }
    }

    // ================================================================
    // Public API
    // ================================================================

    /// Explain a query by traversing the graph.
    ///
    /// This is the main entry point for CVE. Given a query string and a graph,
    /// it:
    /// 1. Activates nodes relevant to the query via SpreadingActivation
    /// 2. Builds a reasoning path of scored compositions
    /// 3. Verbalizes each composition along the path
    /// 4. Applies epistemic qualifiers
    /// 5. Composes the final explanation with audit footer
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cve = CompositionalVerbalize::new();
    /// let result = cve.explain("Kenapa aplikasi lambat", &graph);
    /// assert!(!result.text.is_empty());
    /// assert!(!result.path.is_empty());
    /// ```
    pub fn explain(&self, query: &str, graph: &Graph) -> VerbalizationResult {
        // Step 1: Activate query-relevant nodes.
        let activation_map = self.activate_query(query, graph);

        if activation_map.is_empty() {
            return VerbalizationResult::insufficient();
        }

        // Step 2: Build reasoning path.
        let path = self.build_reasoning_path(&activation_map, graph);

        // Steps 3-5: Verbalize, qualify, compose.
        self.compose_output(&path, graph)
    }

    /// Explain a query with explicit seed nodes (advanced usage).
    ///
    /// When the caller already knows which nodes are relevant (e.g., from
    /// a previous spreading activation run), this avoids re-running the
    /// activation step.
    pub fn explain_with_seeds(
        &self,
        seeds: &[(NodeId, f32)],
        graph: &Graph,
    ) -> VerbalizationResult {
        let activation_map = self.spreading.spread(seeds, graph);
        let path = self.build_reasoning_path(&activation_map, graph);
        self.compose_output(&path, graph)
    }

    /// Verbalize a single composition without query context.
    ///
    /// Useful for debugging and for generating explanations of specific
    /// compositions that are already known to be relevant.
    pub fn verbalize_single(&self, comp: &Composition) -> String {
        let raw = self.verbalize_composition(comp);
        self.qualify(&raw, comp)
    }
}

// ========================================================================
// CompositionalVerbalizeTransform — Pipeline Integration
// ========================================================================

/// Pipeline transform that runs CVE after ConvergenceDetection.
///
/// This is an **optional** transform — it is NOT in the default pipeline by
/// default. It can be registered when you need the pipeline to produce
/// explanations as part of the ingest cycle.
///
/// When registered, it runs after `ConvergenceDetection` and generates
/// a verbalization of all compositions in the graph. The result is stored
/// in `PipelineContext` (future: as a side channel).
#[derive(Debug, Clone, Default)]
pub struct CompositionalVerbalizeTransform {
    /// The underlying CVE engine.
    pub engine: CompositionalVerbalize,
}

impl CompositionalVerbalizeTransform {
    /// Create a new transform with default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom configuration.
    pub fn with_config(config: VerbalizeConfig) -> Self {
        Self {
            engine: CompositionalVerbalize::with_config(config),
        }
    }
}

impl ErasedTransform for CompositionalVerbalizeTransform {
    fn id(&self) -> &'static str {
        "CompositionalVerbalize"
    }

    fn execute(&self, _ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        // Only run if there are compositions to verbalize.
        if graph.composition_count() == 0 {
            return IngestResult::new();
        }

        // Generate a default explanation of all compositions.
        // In pipeline context, we don't have a specific query, so we use
        // the raw text if available, otherwise we produce a summary.
        let _result = self.engine.explain("semua", graph);

        // The verbalization result is currently computed but not stored
        // in PipelineContext. Future: add `last_verbalization` field
        // to PipelineContext.

        IngestResult {
            atoms_created: 0,
            compositions_created: 0,
            edges_created: 0,
            gaps_detected: 0,
            enrichments_applied: 0,
            governance_transitions: 0,
        }
    }
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]
    use super::*;

    /// Helper: create a graph with a few compositions for testing.
    fn make_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Event: "Raymond membuat aplikasi karena lambat"
        let node_raymond = graph.ensure_node("Raymond");
        let node_membuat = graph.ensure_node("membuat");
        let node_aplikasi = graph.ensure_node("aplikasi");
        let node_lambat = graph.ensure_node("lambat");

        let mut comp_event = Composition::default();
        comp_event.id = "comp_event_1".to_string();
        comp_event.composition_type = CompositionType::Event;
        comp_event.lifecycle = LifecycleState::Stable;
        comp_event.epistemic = EpistemicState::Grounded;
        comp_event.confidence = 0.85;
        comp_event.members = vec![
            CompositionMember {
                node_id: node_raymond,
                role: SemanticRole::Arg0Agent,
                confidence: 0.9,
                label: "Raymond".to_string(),
            },
            CompositionMember {
                node_id: node_membuat,
                role: SemanticRole::Predicate,
                confidence: 0.9,
                label: "membuat".to_string(),
            },
            CompositionMember {
                node_id: node_aplikasi,
                role: SemanticRole::Arg1Patient,
                confidence: 0.8,
                label: "aplikasi".to_string(),
            },
            CompositionMember {
                node_id: node_lambat,
                role: SemanticRole::Cause,
                confidence: 0.7,
                label: "lambat".to_string(),
            },
        ];
        graph.compositions.insert(comp_event.id.clone(), comp_event);

        // HiddenMeaning: "cache digunakan sebagai solusi untuk lambat"
        let node_cache = graph.ensure_node("cache");
        let mut comp_hm = Composition::default();
        comp_hm.id = "comp_hm_1".to_string();
        comp_hm.composition_type = CompositionType::HiddenMeaning;
        comp_hm.lifecycle = LifecycleState::Candidate;
        comp_hm.epistemic = EpistemicState::Inferred;
        comp_hm.confidence = 0.72;
        comp_hm.members = vec![
            CompositionMember {
                node_id: node_cache,
                role: SemanticRole::Solution,
                confidence: 0.8,
                label: "cache".to_string(),
            },
            CompositionMember {
                node_id: node_lambat,
                role: SemanticRole::Problem,
                confidence: 0.7,
                label: "lambat".to_string(),
            },
        ];
        graph.compositions.insert(comp_hm.id.clone(), comp_hm);

        // Pattern: "Ketika database_penuh, maka lambat"
        let node_db_full = graph.ensure_node("database_penuh");
        let mut comp_pattern = Composition::default();
        comp_pattern.id = "comp_pattern_1".to_string();
        comp_pattern.composition_type = CompositionType::Pattern;
        comp_pattern.lifecycle = LifecycleState::Stable;
        comp_pattern.epistemic = EpistemicState::Grounded;
        comp_pattern.confidence = 0.9;
        comp_pattern.members = vec![
            CompositionMember {
                node_id: node_db_full,
                role: SemanticRole::Antecedent,
                confidence: 0.9,
                label: "database_penuh".to_string(),
            },
            CompositionMember {
                node_id: node_lambat,
                role: SemanticRole::Consequent,
                confidence: 0.85,
                label: "lambat".to_string(),
            },
        ];
        graph
            .compositions
            .insert(comp_pattern.id.clone(), comp_pattern);

        graph
    }

    #[test]
    fn test_verbalize_event_composition() {
        let graph = make_test_graph();
        let cve = CompositionalVerbalize::new();
        let comp = graph.get_composition(&"comp_event_1".to_string()).unwrap();
        let sentence = cve.verbalize_composition(comp);
        assert!(
            sentence.contains("Raymond"),
            "Event verbalization should contain agent 'Raymond': got '{}'",
            sentence
        );
        assert!(
            sentence.contains("membuat"),
            "Event verbalization should contain predicate 'membuat': got '{}'",
            sentence
        );
        assert!(
            sentence.contains("aplikasi"),
            "Event verbalization should contain patient 'aplikasi': got '{}'",
            sentence
        );
        assert!(
            sentence.contains("karena lambat"),
            "Event verbalization should contain cause 'karena lambat': got '{}'",
            sentence
        );
        eprintln!("Event verbalization: {}", sentence);
    }

    #[test]
    fn test_verbalize_hidden_meaning_composition() {
        let graph = make_test_graph();
        let cve = CompositionalVerbalize::new();
        let comp = graph.get_composition(&"comp_hm_1".to_string()).unwrap();
        let sentence = cve.verbalize_composition(comp);
        assert!(
            sentence.contains("cache"),
            "HiddenMeaning verbalization should contain 'cache': got '{}'",
            sentence
        );
        assert!(
            sentence.contains("solusi"),
            "HiddenMeaning verbalization should contain 'solusi': got '{}'",
            sentence
        );
        assert!(
            sentence.contains("lambat"),
            "HiddenMeaning verbalization should contain problem 'lambat': got '{}'",
            sentence
        );
        eprintln!("HiddenMeaning verbalization: {}", sentence);
    }

    #[test]
    fn test_verbalize_pattern_composition() {
        let graph = make_test_graph();
        let cve = CompositionalVerbalize::new();
        let comp = graph
            .get_composition(&"comp_pattern_1".to_string())
            .unwrap();
        let sentence = cve.verbalize_composition(comp);
        assert!(
            sentence.contains("Ketika"),
            "Pattern verbalization should start with 'Ketika': got '{}'",
            sentence
        );
        assert!(
            sentence.contains("database_penuh"),
            "Pattern verbalization should contain antecedent: got '{}'",
            sentence
        );
        assert!(
            sentence.contains("lambat"),
            "Pattern verbalization should contain consequent: got '{}'",
            sentence
        );
        eprintln!("Pattern verbalization: {}", sentence);
    }

    #[test]
    fn test_epistemic_qualifier_stable_grounded() {
        let graph = make_test_graph();
        let cve = CompositionalVerbalize::new();
        let comp = graph.get_composition(&"comp_event_1".to_string()).unwrap();
        // Stable + Grounded + confidence 0.85 > 0.8 → no qualifier
        let qualified = cve.qualify("Test sentence.", comp);
        assert_eq!(
            qualified, "Test sentence.",
            "Stable/Grounded/high-confidence should have NO qualifier, got: '{}'",
            qualified
        );
    }

    #[test]
    fn test_epistemic_qualifier_candidate_inferred() {
        let graph = make_test_graph();
        let cve = CompositionalVerbalize::new();
        let comp = graph.get_composition(&"comp_hm_1".to_string()).unwrap();
        // Candidate + Inferred → "Berdasarkan analisis, "
        let qualified = cve.qualify("Test sentence.", comp);
        assert!(
            qualified.starts_with("Berdasarkan analisis,"),
            "Candidate/Inferred should have 'Berdasarkan analisis,' qualifier, got: '{}'",
            qualified
        );
    }

    #[test]
    fn test_epistemic_qualifier_contradicted() {
        let cve = CompositionalVerbalize::new();
        let mut comp = Composition::default();
        comp.lifecycle = LifecycleState::Stable;
        comp.epistemic = EpistemicState::Contradicted;
        comp.confidence = 0.5;
        let qualified = cve.qualify("Test sentence.", &comp);
        assert!(
            qualified.starts_with("Meskipun ada kontradiksi,"),
            "Contradicted should have 'Meskipun ada kontradiksi,' qualifier, got: '{}'",
            qualified
        );
    }

    #[test]
    fn test_explain_query() {
        let graph = make_test_graph();
        let cve = CompositionalVerbalize::new();
        let result = cve.explain("lambat", &graph);

        assert!(
            !result.path.is_empty(),
            "Query 'lambat' should find relevant compositions"
        );
        assert!(!result.text.is_empty(), "Explanation should not be empty");
        assert!(
            result.text.contains("lambat"),
            "Explanation should contain 'lambat': got '{}'",
            result.text
        );
        assert!(
            result.avg_confidence > 0.0,
            "Average confidence should be positive"
        );
        eprintln!("Explanation for 'lambat':\n{}", result.text);
    }

    #[test]
    fn test_explain_empty_graph() {
        let graph = Graph::new();
        let cve = CompositionalVerbalize::new();
        let result = cve.explain("apa saja", &graph);

        assert!(
            result.path.is_empty(),
            "Empty graph should produce empty path"
        );
        assert!(
            result.text.contains("Tidak ada informasi"),
            "Empty graph should produce 'insufficient' message, got: '{}'",
            result.text
        );
    }

    #[test]
    fn test_explain_no_matching_nodes() {
        let graph = make_test_graph();
        let cve = CompositionalVerbalize::new();
        let result = cve.explain("xyzzy_nonexistent", &graph);

        // Either insufficient or no relevant compositions found
        assert!(
            result.path.is_empty() || result.text.contains("Tidak ada informasi"),
            "Query with no matching nodes should produce insufficient result"
        );
    }

    #[test]
    fn test_verbalize_single() {
        let graph = make_test_graph();
        let cve = CompositionalVerbalize::new();
        let comp = graph.get_composition(&"comp_event_1".to_string()).unwrap();
        let sentence = cve.verbalize_single(comp);

        assert!(
            sentence.contains("Raymond"),
            "Single verbalization should contain agent"
        );
        assert!(
            sentence.contains("membuat"),
            "Single verbalization should contain predicate"
        );
        eprintln!("Single verbalization: {}", sentence);
    }

    #[test]
    fn test_hypothesis_verbalization() {
        let mut graph = Graph::new();
        let node_patient = graph.ensure_node("server");
        let node_pred = graph.ensure_node("mengalami_kegagalan");

        let mut comp = Composition::default();
        comp.id = "comp_hypo_1".to_string();
        comp.composition_type = CompositionType::Hypothesis;
        comp.lifecycle = LifecycleState::Quarantine;
        comp.epistemic = EpistemicState::Hypothesis;
        comp.confidence = 0.4;
        comp.members = vec![
            CompositionMember {
                node_id: node_patient,
                role: SemanticRole::Arg1Patient,
                confidence: 0.5,
                label: "server".to_string(),
            },
            CompositionMember {
                node_id: node_pred,
                role: SemanticRole::Predicate,
                confidence: 0.4,
                label: "mengalami_kegagalan".to_string(),
            },
        ];

        let cve = CompositionalVerbalize::new();
        let sentence = cve.verbalize_single(&comp);

        assert!(
            sentence.contains("server"),
            "Hypothesis should contain patient"
        );
        assert!(
            sentence.contains("mengalami_kegagalan"),
            "Hypothesis should contain predicate"
        );
        assert!(
            sentence.contains("Kemungkinan besar"),
            "Hypothesis epistemic should have 'Kemungkinan besar' qualifier"
        );
        eprintln!("Hypothesis verbalization: {}", sentence);
    }

    #[test]
    fn test_situation_verbalization() {
        let mut graph = Graph::new();
        let node_agent = graph.ensure_node("tim_develop");
        let node_pred = graph.ensure_node("mengembangkan");
        let node_patient = graph.ensure_node("fitur_baru");

        let mut comp = Composition::default();
        comp.id = "comp_sit_1".to_string();
        comp.composition_type = CompositionType::Situation;
        comp.lifecycle = LifecycleState::Stable;
        comp.epistemic = EpistemicState::Observed;
        comp.confidence = 0.6;
        comp.members = vec![
            CompositionMember {
                node_id: node_agent,
                role: SemanticRole::Arg0Agent,
                confidence: 0.7,
                label: "tim_develop".to_string(),
            },
            CompositionMember {
                node_id: node_pred,
                role: SemanticRole::Predicate,
                confidence: 0.6,
                label: "mengembangkan".to_string(),
            },
            CompositionMember {
                node_id: node_patient,
                role: SemanticRole::Arg1Patient,
                confidence: 0.5,
                label: "fitur_baru".to_string(),
            },
        ];

        let cve = CompositionalVerbalize::new();
        let sentence = cve.verbalize_single(&comp);

        assert!(
            sentence.contains("Dalam konteks"),
            "Situation should start with 'Dalam konteks'"
        );
        assert!(
            sentence.contains("tim_develop"),
            "Situation should contain agent"
        );
        eprintln!("Situation verbalization: {}", sentence);
    }

    #[test]
    fn test_acquisition_verbalization() {
        let mut graph = Graph::new();
        let node_agent = graph.ensure_node("auditor");
        let node_pred = graph.ensure_node("menemukan");
        let node_patient = graph.ensure_node("pelanggaran");

        let mut comp = Composition::default();
        comp.id = "comp_acq_1".to_string();
        comp.composition_type = CompositionType::Acquisition;
        comp.lifecycle = LifecycleState::New;
        comp.epistemic = EpistemicState::Observed;
        comp.confidence = 0.7;
        comp.members = vec![
            CompositionMember {
                node_id: node_agent,
                role: SemanticRole::Arg0Agent,
                confidence: 0.8,
                label: "auditor".to_string(),
            },
            CompositionMember {
                node_id: node_pred,
                role: SemanticRole::Predicate,
                confidence: 0.7,
                label: "menemukan".to_string(),
            },
            CompositionMember {
                node_id: node_patient,
                role: SemanticRole::Arg1Patient,
                confidence: 0.7,
                label: "pelanggaran".to_string(),
            },
        ];

        let cve = CompositionalVerbalize::new();
        let sentence = cve.verbalize_single(&comp);

        assert!(
            sentence.contains("Diketahui bahwa"),
            "Acquisition should start with 'Diketahui bahwa'"
        );
        assert!(
            sentence.contains("auditor"),
            "Acquisition should contain agent"
        );
        eprintln!("Acquisition verbalization: {}", sentence);
    }

    #[test]
    fn test_audit_footer_present() {
        let graph = make_test_graph();
        let cve = CompositionalVerbalize::new();
        let result = cve.explain("lambat", &graph);

        assert!(
            result.text.contains("[Keyakinan rata-rata:"),
            "Audit footer should contain confidence summary"
        );
        assert!(
            result.text.contains("[Sumber:"),
            "Audit footer should contain source count"
        );
        assert!(
            result.text.contains("[Dapat diaudit:"),
            "Audit footer should contain audit path"
        );
    }

    #[test]
    fn test_audit_footer_disabled() {
        let graph = make_test_graph();
        let mut config = VerbalizeConfig::default();
        config.include_audit_footer = false;
        let cve = CompositionalVerbalize::with_config(config);
        let result = cve.explain("lambat", &graph);

        assert!(
            !result.text.contains("[Keyakinan rata-rata:"),
            "Audit footer should NOT be present when disabled"
        );
    }

    #[test]
    fn test_event_with_all_roles() {
        let mut graph = Graph::new();
        let node_agent = graph.ensure_node("tim");
        let node_pred = graph.ensure_node("mengoptimasi");
        let node_patient = graph.ensure_node("database");
        let node_cause = graph.ensure_node("keluhan");
        let node_purpose = graph.ensure_node("performa");
        let node_location = graph.ensure_node("server_room");
        let node_time = graph.ensure_node("malam");
        let node_instrument = graph.ensure_node("tool_monitoring");

        let mut comp = Composition::default();
        comp.id = "comp_full_event".to_string();
        comp.composition_type = CompositionType::Event;
        comp.lifecycle = LifecycleState::Stable;
        comp.epistemic = EpistemicState::Grounded;
        comp.confidence = 0.95;
        comp.members = vec![
            CompositionMember {
                node_id: node_agent,
                role: SemanticRole::Arg0Agent,
                confidence: 0.9,
                label: "tim".to_string(),
            },
            CompositionMember {
                node_id: node_pred,
                role: SemanticRole::Predicate,
                confidence: 0.9,
                label: "mengoptimasi".to_string(),
            },
            CompositionMember {
                node_id: node_patient,
                role: SemanticRole::Arg1Patient,
                confidence: 0.8,
                label: "database".to_string(),
            },
            CompositionMember {
                node_id: node_cause,
                role: SemanticRole::Cause,
                confidence: 0.7,
                label: "keluhan".to_string(),
            },
            CompositionMember {
                node_id: node_purpose,
                role: SemanticRole::Purpose,
                confidence: 0.7,
                label: "performa".to_string(),
            },
            CompositionMember {
                node_id: node_location,
                role: SemanticRole::Location,
                confidence: 0.6,
                label: "server_room".to_string(),
            },
            CompositionMember {
                node_id: node_time,
                role: SemanticRole::Time,
                confidence: 0.6,
                label: "malam".to_string(),
            },
            CompositionMember {
                node_id: node_instrument,
                role: SemanticRole::Instrument,
                confidence: 0.5,
                label: "tool_monitoring".to_string(),
            },
        ];

        let cve = CompositionalVerbalize::new();
        let sentence = cve.verbalize_single(&comp);

        assert!(sentence.contains("tim"), "Should contain agent");
        assert!(
            sentence.contains("mengoptimasi"),
            "Should contain predicate"
        );
        assert!(sentence.contains("database"), "Should contain patient");
        assert!(sentence.contains("karena keluhan"), "Should contain cause");
        assert!(
            sentence.contains("untuk performa"),
            "Should contain purpose"
        );
        assert!(
            sentence.contains("di server_room"),
            "Should contain location"
        );
        assert!(sentence.contains("saat malam"), "Should contain time");
        assert!(
            sentence.contains("dengan tool_monitoring"),
            "Should contain instrument"
        );

        eprintln!("Full event verbalization: {}", sentence);
    }

    #[test]
    fn test_deprecated_qualifier() {
        let cve = CompositionalVerbalize::new();
        let mut comp = Composition::default();
        comp.lifecycle = LifecycleState::Deprecated;
        comp.epistemic = EpistemicState::Grounded;
        comp.confidence = 0.3;
        let qualified = cve.qualify("Server down.", &comp);
        assert!(
            qualified.starts_with("Sebelumnya diyakini,"),
            "Deprecated should have 'Sebelumnya diyakini,' qualifier, got: '{}'",
            qualified
        );
    }

    #[test]
    fn test_quarantine_qualifier() {
        let cve = CompositionalVerbalize::new();
        let mut comp = Composition::default();
        comp.lifecycle = LifecycleState::Quarantine;
        comp.epistemic = EpistemicState::Inferred;
        comp.confidence = 0.4;
        let qualified = cve.qualify("Suspicious claim.", &comp);
        assert!(
            qualified.starts_with("Perlu ditinjau kembali,"),
            "Quarantine should have 'Perlu ditinjau kembali,' qualifier, got: '{}'",
            qualified
        );
    }
}
