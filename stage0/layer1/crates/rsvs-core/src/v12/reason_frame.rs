//! # MD-2: ReasonFrame Transform
//!
//! Pre-ingest reasoning over event atoms to derive hidden meaning atoms.
//! This transform applies deterministic reasoning rules to event frames
//! BEFORE they enter the graph, producing `SemanticAtom`s of type
//! `HiddenMeaning`.
//!
//! ## Reasoning Rules
//!
//! | # | Rule | Trigger | Output |
//! |---|------|---------|--------|
//! | 1 | ProblemSolutionRule | Cause + Action + Patient | ProblemSolutionPattern |
//! | 2 | GoalInferenceRule | Purpose marker | GoalInference |
//! | 3 | PolarityConflictRule | Same event + opposite polarity | PolarityConflict |
//! | 4 | ConditionConsequenceRule | Antecedent + Consequent | condition_consequence |
//!
//! ## Architecture
//!
//! ```text
//! Event atom → ReasoningContext → apply rules → Vec<ReasoningResult>
//!                                                     │
//!                                        each result → into_atom()
//! ```
//!
//! ## Feature Flag
//!
//! This module is only compiled when the `v12` feature is enabled.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use super::pipeline::{ErasedTransform, Graph, IngestResult};
use super::types::*;
use crate::types::{EdgeSource, NodeId};

// ========================================================================
// ReasoningRule — Trait for Deriving Hidden Meanings
// ========================================================================

/// A deterministic reasoning rule that derives hidden meaning from events.
///
/// Each rule implements two methods:
/// - `applies()` — check if the rule is relevant given the context
/// - `generate()` — produce hidden meaning atoms if the rule applies
///
/// Rules are stateless and deterministic. The same input always produces
/// the same output.
pub trait ReasoningRule: Send + Sync {
    /// Unique name for this rule (e.g., "ProblemSolutionRule").
    fn name(&self) -> &'static str;

    /// Check whether this rule applies to the given reasoning context.
    fn applies(&self, context: &ReasoningContext) -> bool;

    /// Generate hidden meaning atoms from the reasoning context.
    ///
    /// Called only when `applies()` returns `true`.
    fn generate(&self, context: &ReasoningContext) -> Vec<ReasoningResult>;
}

// ========================================================================
// ReasoningContext — Input to Reasoning Rules
// ========================================================================

/// Context provided to reasoning rules (MD-2).
///
/// Contains the current event atom plus surrounding context:
/// - The event being reasoned about
/// - Recent events from the sliding window (for cross-atom reasoning)
/// - Optional graph references for confidence adjustment
#[derive(Debug, Clone)]
pub struct ReasoningContext<'a> {
    /// The event atom being reasoned about.
    pub event: &'a SemanticAtom,
    /// Recent events from the sliding window.
    pub recent_events: &'a [SemanticAtom],
    /// Optional graph reference for confidence lookup.
    /// // STUB:TODO — Replace with actual Graph reference when available.
    pub graph_ref: Option<GraphContextRef>,
}

impl<'a> ReasoningContext<'a> {
    /// Create a new reasoning context.
    pub fn new(event: &'a SemanticAtom, recent_events: &'a [SemanticAtom]) -> Self {
        Self {
            event,
            recent_events,
            graph_ref: None,
        }
    }

    /// Create with graph context.
    pub fn with_graph(
        event: &'a SemanticAtom,
        recent_events: &'a [SemanticAtom],
        graph: GraphContextRef,
    ) -> Self {
        Self {
            event,
            recent_events,
            graph_ref: Some(graph),
        }
    }

    /// Find a recent event with the same predicate label.
    pub fn find_recent_with_predicate(&self, label: &str) -> Option<&SemanticAtom> {
        self.recent_events
            .iter()
            .find(|e| e.label == label && e.atom_type == AtomType::Event)
    }

    /// Find recent events with opposite polarity to the current event.
    pub fn find_opposite_polarity(&self) -> Vec<&SemanticAtom> {
        let current_polarity = match &self.event.polarity {
            Some(Polarity::Positive) => Polarity::Negative,
            Some(Polarity::Negative) => Polarity::Positive,
            None => return Vec::new(),
        };

        self.recent_events
            .iter()
            .filter(|e| {
                e.label == self.event.label
                    && e.atom_type == AtomType::Event
                    && e.polarity.as_ref() == Some(&current_polarity)
            })
            .collect()
    }
}

/// Lightweight graph context reference for reasoning rules.
///
/// Provides structural and activation information for rules to adjust
/// confidence based on graph structure and spreading activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphContextRef {
    // === Existing fields ===
    /// Number of compositions with the same predicate.
    #[serde(default)]
    pub same_predicate_count: usize,
    /// Whether a contradiction exists for this predicate.
    #[serde(default)]
    pub has_contradiction: bool,
    /// Average confidence of compositions with the same predicate.
    #[serde(default)]
    pub avg_confidence: f32,

    // === NEW: Activation data from SpreadingActivation (Phase T) ===
    /// Activation energy of the predicate node.
    /// Higher = more connected in the graph = more confident.
    #[serde(default)]
    pub activation_energy_for_predicate: f32,
    /// Activation energy per role node.
    /// Maps role → energy of the node filling that role.
    #[serde(default)]
    pub activation_energy_for_roles: HashMap<SemanticRole, f32>,
    /// Top N most-activated neighbor nodes.
    #[serde(default)]
    pub top_activated_neighbors: Vec<(NodeId, f32)>,
    /// Ambiguity score: how close are the top 2 interpretation candidates.
    /// > 0.3 means two interpretations are too close — flag for questioning.
    #[serde(default)]
    pub ambiguity_score: f32,
}

impl Default for GraphContextRef {
    fn default() -> Self {
        Self {
            same_predicate_count: 0,
            has_contradiction: false,
            avg_confidence: 0.0,
            activation_energy_for_predicate: 0.0,
            activation_energy_for_roles: HashMap::new(),
            top_activated_neighbors: Vec::new(),
            ambiguity_score: 0.0,
        }
    }
}

// ========================================================================
// ReasoningResult — Output of a Rule
// ========================================================================

/// Result of applying a reasoning rule (MD-2).
///
/// Contains the derived hidden meaning atom plus metadata about
/// which rule produced it and how confident we are in the derivation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResult {
    /// The reasoning rule that produced this result.
    pub rule_name: String,
    /// The derived hidden meaning atom.
    pub atom: SemanticAtom,
    /// Confidence in this derivation (0.0–1.0).
    pub derivation_confidence: f32,
}

impl ReasoningResult {
    /// Create a new reasoning result.
    pub fn new(rule_name: &str, atom: SemanticAtom, derivation_confidence: f32) -> Self {
        Self {
            rule_name: rule_name.to_string(),
            atom,
            derivation_confidence,
        }
    }

    /// Convert into the inner atom, discarding metadata.
    pub fn into_atom(self) -> SemanticAtom {
        self.atom
    }

    /// Get a reference to the derived atom.
    pub fn atom(&self) -> &SemanticAtom {
        &self.atom
    }
}

// ========================================================================
// Shared Confidence Modulation — Phase T: Activation-Aware Reasoning
// ========================================================================

/// Apply activation-aware confidence modulation to reasoning results.
///
/// This shared function is used by both `ReasonFrame::reason_with_graph()`
/// and `ReReasonFrame::execute()` to ensure consistent confidence adjustment.
///
/// # Modulation Rules
///
/// 1. **Predicate connectivity boost**: If the predicate is well-connected
///    in the graph (activation energy > 0.5), boost confidence by up to 15%.
/// 2. **Ambiguity penalty**: If two interpretations are too close
///    (ambiguity score > 0.3), reduce confidence proportionally.
/// 3. **Predicate count boost**: More existing compositions with the same
///    predicate increases confidence (up to 10% boost).
/// 4. **Contradiction penalty**: If a contradiction exists for this predicate,
///    reduce confidence by 15%.
fn apply_confidence_modulation(
    results: &mut [ReasoningResult],
    graph_ref: &GraphContextRef,
) {
    for result in results.iter_mut() {
        // Predicate connectivity boost.
        if graph_ref.activation_energy_for_predicate > 0.5 {
            let boost = graph_ref.activation_energy_for_predicate * 0.15;
            result.derivation_confidence = (result.derivation_confidence + boost).min(1.0);
            result.atom.confidence = result.derivation_confidence;
        }

        // Ambiguity penalty.
        if graph_ref.ambiguity_score > 0.3 {
            result.derivation_confidence *= 1.0 - graph_ref.ambiguity_score * 0.5;
            result.atom.confidence = result.derivation_confidence;
        }

        // Predicate count boost.
        if graph_ref.same_predicate_count > 0 {
            let boost = (graph_ref.same_predicate_count as f32 * 0.02).min(0.10);
            result.derivation_confidence = (result.derivation_confidence + boost).min(1.0);
            result.atom.confidence = result.derivation_confidence;
        }

        // Contradiction penalty.
        if graph_ref.has_contradiction {
            result.derivation_confidence *= 0.85;
            result.atom.confidence = result.derivation_confidence;
        }
    }
}

// ========================================================================
// Rule 1: ProblemSolutionRule
// ========================================================================

/// ProblemSolutionRule — derives ProblemSolutionPattern from Cause + Action + Patient.
///
/// When an event has a Cause role and an Agent performing an action on a Patient,
/// the Cause is the Problem and the action+Patient is the Solution.
///
/// # Trigger
///
/// Event must have:
/// - `Cause` role present
/// - `Arg0Agent` role present (the actor)
/// - `Arg1Patient` role present (what is acted upon)
///
/// # Output
///
/// ```text
/// SemanticAtom {
///     atom_type: HiddenMeaning,
///     label: "problem_solution",
///     variant: MeaningVariant(ProblemSolutionPattern),
///     roles: {
///         Problem: <Cause value>,
///         Solution: <Arg1Patient value>,
///         Arg0Agent: <Agent value>,
///     }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct ProblemSolutionRule;

impl ProblemSolutionRule {
    /// Create a new ProblemSolutionRule.
    pub fn new() -> Self {
        Self
    }
}

impl ReasoningRule for ProblemSolutionRule {
    fn name(&self) -> &'static str {
        "ProblemSolutionRule"
    }

    fn applies(&self, context: &ReasoningContext) -> bool {
        let event = context.event;
        event.atom_type == AtomType::Event
            && event.roles.contains_key(&SemanticRole::Cause)
            && event.roles.contains_key(&SemanticRole::Arg0Agent)
            && event.roles.contains_key(&SemanticRole::Arg1Patient)
    }

    fn generate(&self, context: &ReasoningContext) -> Vec<ReasoningResult> {
        let event = context.event;

        let problem = match event.roles.get(&SemanticRole::Cause) {
            Some(p) => p.clone(),
            None => return Vec::new(),
        };
        let solution = match event.roles.get(&SemanticRole::Arg1Patient) {
            Some(s) => s.clone(),
            None => return Vec::new(),
        };
        let agent = event
            .roles
            .get(&SemanticRole::Arg0Agent)
            .cloned()
            .unwrap_or_default();

        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Problem, problem);
        roles.insert(SemanticRole::Solution, solution);
        roles.insert(SemanticRole::Arg0Agent, agent);

        // Copy structural reference.
        roles.insert(SemanticRole::SourceEvent, event.id.clone());

        // Derivation confidence: based on event confidence with a small discount.
        let derivation_confidence = event.confidence * 0.85;

        let atom = SemanticAtom {
            id: String::new(), // Will be assigned by pipeline
            label: "problem_solution".to_string(),
            atom_type: AtomType::HiddenMeaning,
            roles,
            polarity: event.polarity.clone(),
            voice: None,
            variant: Some(AtomVariant::MeaningVariant(
                crate::types::HiddenMeaningType::Emergent,
            )),
            confidence: derivation_confidence,
            source: EdgeSource::HiddenMeaningRule,
            composition_id: None,
        };

        vec![ReasoningResult::new(
            self.name(),
            atom,
            derivation_confidence,
        )]
    }
}

// ========================================================================
// Rule 2: GoalInferenceRule
// ========================================================================

/// GoalInferenceRule — derives GoalInference from a Purpose marker.
///
/// When an event has a Purpose role, the agent's goal can be inferred
/// from the purpose content.
///
/// # Trigger
///
/// Event must have:
/// - `Purpose` role present
///
/// # Output
///
/// ```text
/// SemanticAtom {
///     atom_type: HiddenMeaning,
///     label: "goal_inference",
///     variant: MeaningVariant(Emergent),
///     roles: {
///         ImpliedGoal: <Purpose value>,
///         Arg0Agent: <Agent value>,
///     }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct GoalInferenceRule;

impl GoalInferenceRule {
    /// Create a new GoalInferenceRule.
    pub fn new() -> Self {
        Self
    }
}

impl ReasoningRule for GoalInferenceRule {
    fn name(&self) -> &'static str {
        "GoalInferenceRule"
    }

    fn applies(&self, context: &ReasoningContext) -> bool {
        let event = context.event;
        event.atom_type == AtomType::Event && event.roles.contains_key(&SemanticRole::Purpose)
    }

    fn generate(&self, context: &ReasoningContext) -> Vec<ReasoningResult> {
        let event = context.event;

        let purpose = match event.roles.get(&SemanticRole::Purpose) {
            Some(p) => p.clone(),
            None => return Vec::new(),
        };

        let mut roles = HashMap::new();
        roles.insert(SemanticRole::ImpliedGoal, purpose);

        if let Some(agent) = event.roles.get(&SemanticRole::Arg0Agent) {
            roles.insert(SemanticRole::Arg0Agent, agent.clone());
        }

        roles.insert(SemanticRole::SourceEvent, event.id.clone());

        // Derivation confidence: purpose is a strong signal, but we discount slightly.
        let derivation_confidence = event.confidence * 0.80;

        let atom = SemanticAtom {
            id: String::new(),
            label: "goal_inference".to_string(),
            atom_type: AtomType::HiddenMeaning,
            roles,
            polarity: event.polarity.clone(),
            voice: None,
            variant: Some(AtomVariant::MeaningVariant(
                crate::types::HiddenMeaningType::Emergent,
            )),
            confidence: derivation_confidence,
            source: EdgeSource::HiddenMeaningRule,
            composition_id: None,
        };

        vec![ReasoningResult::new(
            self.name(),
            atom,
            derivation_confidence,
        )]
    }
}

// ========================================================================
// Rule 3: PolarityConflictRule
// ========================================================================

/// PolarityConflictRule — detects polarity conflicts across events.
///
/// When the current event has the same predicate as a recent event
/// but opposite polarity, a polarity conflict is flagged as a
/// hidden meaning atom.
///
/// # Trigger
///
/// Current event + at least one recent event with:
/// - Same predicate label
/// - Opposite polarity
///
/// # Output
///
/// ```text
/// SemanticAtom {
///     atom_type: HiddenMeaning,
///     label: "polarity_conflict",
///     variant: MeaningVariant(Emergent),
///     roles: {
///         SourceEvent: <current event id>,
///         Problem: <conflict description>,
///     }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct PolarityConflictRule;

impl PolarityConflictRule {
    /// Create a new PolarityConflictRule.
    pub fn new() -> Self {
        Self
    }
}

impl ReasoningRule for PolarityConflictRule {
    fn name(&self) -> &'static str {
        "PolarityConflictRule"
    }

    fn applies(&self, context: &ReasoningContext) -> bool {
        if context.event.atom_type != AtomType::Event {
            return false;
        }

        // Must have polarity.
        if context.event.polarity.is_none() {
            return false;
        }

        // Must find a recent event with the same predicate and opposite polarity.
        !context.find_opposite_polarity().is_empty()
    }

    fn generate(&self, context: &ReasoningContext) -> Vec<ReasoningResult> {
        let event = context.event;
        let opposite_events = context.find_opposite_polarity();

        opposite_events
            .into_iter()
            .map(|opposite| {
                let conflict_desc = format!(
                    "{}: {} vs {}",
                    event.label,
                    match event.polarity {
                        Some(Polarity::Positive) => "positive",
                        Some(Polarity::Negative) => "negative",
                        None => "neutral",
                    },
                    match opposite.polarity {
                        Some(Polarity::Positive) => "positive",
                        Some(Polarity::Negative) => "negative",
                        None => "neutral",
                    }
                );

                let mut roles = HashMap::new();
                roles.insert(SemanticRole::SourceEvent, event.id.clone());
                roles.insert(SemanticRole::Problem, conflict_desc);

                // Derivation confidence: conflicts are high-signal.
                let derivation_confidence = (event.confidence + opposite.confidence) / 2.0 * 0.90;

                let atom = SemanticAtom {
                    id: String::new(),
                    label: "polarity_conflict".to_string(),
                    atom_type: AtomType::HiddenMeaning,
                    roles,
                    polarity: None,
                    voice: None,
                    variant: Some(AtomVariant::MeaningVariant(
                        crate::types::HiddenMeaningType::Emergent,
                    )),
                    confidence: derivation_confidence,
                    source: EdgeSource::HiddenMeaningRule,
                    composition_id: None,
                };

                ReasoningResult::new(self.name(), atom, derivation_confidence)
            })
            .collect()
    }
}

// ========================================================================
// Rule 4: ConditionConsequenceRule
// ========================================================================

/// ConditionConsequenceRule — derives if-then patterns from Antecedent/Consequent roles.
///
/// When an event has both Antecedent and Consequent roles (extracted by
/// `ExtractFrame` from conditional markers like "jika", "apabila"),
/// this rule produces a HiddenMeaning atom representing the conditional
/// relationship as a structured rule.
///
/// This is the foundation for the tax rule compiler: regulations like
/// "Wajib Pajak dengan PKP di atas Rp500 juta dikenakan tarif 30%"
/// are extracted as Antecedent + Consequent and then compiled into PolicyRules.
///
/// # Trigger
///
/// Event must have:
/// - `Antecedent` role present (the "if" part)
/// - `Consequent` role present (the "then" part)
///
/// # Output
///
/// ```text
/// SemanticAtom {
///     atom_type: HiddenMeaning,
///     label: "condition_consequence",
///     variant: MeaningVariant(Emergent),
///     roles: {
///         Antecedent: <condition text>,
///         Consequent: <consequence text>,
///         PatternType: "if_then",
///     }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct ConditionConsequenceRule;

impl ConditionConsequenceRule {
    /// Create a new ConditionConsequenceRule.
    pub fn new() -> Self {
        Self
    }
}

impl ReasoningRule for ConditionConsequenceRule {
    fn name(&self) -> &'static str {
        "ConditionConsequenceRule"
    }

    fn applies(&self, context: &ReasoningContext) -> bool {
        let event = context.event;
        event.atom_type == AtomType::Event
            && event.roles.contains_key(&SemanticRole::Antecedent)
            && event.roles.contains_key(&SemanticRole::Consequent)
    }

    fn generate(&self, context: &ReasoningContext) -> Vec<ReasoningResult> {
        let event = context.event;

        let antecedent = match event.roles.get(&SemanticRole::Antecedent) {
            Some(a) => a.clone(),
            None => return Vec::new(),
        };
        let consequent = match event.roles.get(&SemanticRole::Consequent) {
            Some(c) => c.clone(),
            None => return Vec::new(),
        };

        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Antecedent, antecedent);
        roles.insert(SemanticRole::Consequent, consequent);
        roles.insert(SemanticRole::PatternType, "if_then".to_string());
        roles.insert(SemanticRole::SourceEvent, event.id.clone());

        // Copy agent/patient if present for richer context.
        if let Some(agent) = event.roles.get(&SemanticRole::Arg0Agent) {
            roles.insert(SemanticRole::Arg0Agent, agent.clone());
        }
        if let Some(patient) = event.roles.get(&SemanticRole::Arg1Patient) {
            roles.insert(SemanticRole::Arg1Patient, patient.clone());
        }

        // Derivation confidence: conditional patterns are high-signal.
        let derivation_confidence = event.confidence * 0.90;

        let atom = SemanticAtom {
            id: String::new(), // Will be assigned by pipeline
            label: "condition_consequence".to_string(),
            atom_type: AtomType::HiddenMeaning,
            roles,
            polarity: event.polarity.clone(),
            voice: None,
            variant: Some(AtomVariant::MeaningVariant(
                crate::types::HiddenMeaningType::Emergent,
            )),
            confidence: derivation_confidence,
            source: EdgeSource::HiddenMeaningRule,
            composition_id: None,
        };

        vec![ReasoningResult::new(
            self.name(),
            atom,
            derivation_confidence,
        )]
    }
}

// ========================================================================
// ReasonFrame — The Transform
// ========================================================================

/// MD-2: Pre-ingest reasoning transform.
///
/// Applies deterministic reasoning rules to event atoms to derive
/// hidden meaning atoms. The rules are:
///
/// 1. `ProblemSolutionRule` — Cause + Action + Patient → ProblemSolutionPattern
/// 2. `GoalInferenceRule` — Purpose marker → GoalInference
/// 3. `PolarityConflictRule` — same event + opposite polarity → PolarityConflict
/// 4. `ConditionConsequenceRule` — Antecedent + Consequent → if_then pattern
///
/// # Transform Signature
///
/// ```text
/// Input:  SemanticAtom (Event) — read from ctx.current_atoms
/// Output: Vec<SemanticAtom> (HiddenMeaning) — appended to ctx.current_atoms
/// ```
///
/// # Rule Sharing (G4)
///
/// The rules are stored in an `Arc<Vec<Box<dyn ReasoningRule>>>` so that
/// `ReReasonFrame` can share the same rule set without duplicating the
/// 4 rule constructions. Rules are stateless and deterministic, so sharing
/// is safe.
pub struct ReasonFrame {
    /// The reasoning rules to apply, in order.
    /// Wrapped in Arc for sharing with ReReasonFrame (G4).
    rules: Arc<Vec<Box<dyn ReasoningRule>>>,
}

impl std::fmt::Debug for ReasonFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReasonFrame")
            .field("rules_count", &self.rules.len())
            .finish()
    }
}

impl Clone for ReasonFrame {
    fn clone(&self) -> Self {
        // Arc clone is O(1) — no rule duplication.
        Self { rules: Arc::clone(&self.rules) }
    }
}

impl Default for ReasonFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasonFrame {
    /// Create a ReasonFrame with all default rules.
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

    /// Create with custom rules.
    pub fn with_rules(rules: Vec<Box<dyn ReasoningRule>>) -> Self {
        Self { rules: Arc::new(rules) }
    }

    /// Get a reference-counted handle to the rules (for sharing with ReReasonFrame).
    pub fn shared_rules(&self) -> Arc<Vec<Box<dyn ReasoningRule>>> {
        Arc::clone(&self.rules)
    }

    /// Apply all rules to an event atom.
    ///
    /// Returns a vector of reasoning results (may be empty if no rules apply).
    pub fn reason(
        &self,
        event: &SemanticAtom,
        recent_events: &[SemanticAtom],
    ) -> Vec<ReasoningResult> {
        if event.atom_type != AtomType::Event {
            return Vec::new();
        }

        let context = ReasoningContext::new(event, recent_events);
        let mut results = Vec::new();

        for rule in self.rules.iter() {
            if rule.applies(&context) {
                results.extend(rule.generate(&context));
            }
        }

        results
    }

    /// Apply reasoning with graph context for confidence adjustment.
    ///
    /// When graph context is available, the derivation confidence may be
    /// adjusted based on how well the derived meaning aligns with existing
    /// graph structure.
    pub fn reason_with_graph(
        &self,
        event: &SemanticAtom,
        recent_events: &[SemanticAtom],
        graph_ref: &GraphContextRef,
    ) -> Vec<ReasoningResult> {
        if event.atom_type != AtomType::Event {
            return Vec::new();
        }

        let context = ReasoningContext::with_graph(event, recent_events, graph_ref.clone());
        let mut results = Vec::new();

        for rule in self.rules.iter() {
            if rule.applies(&context) {
                let mut rule_results = rule.generate(&context);
                apply_confidence_modulation(&mut rule_results, &graph_ref);
                results.extend(rule_results);
            }
        }

        results
    }

    /// Build a GraphContextRef from the graph for the given event.
    ///
    /// This method does NOT have access to activation energies; use
    /// [`build_graph_context_with_activation`] when activation data is available.
    fn build_graph_context(event: &SemanticAtom, graph: &Graph) -> GraphContextRef {
        let mut same_predicate_count = 0usize;
        let mut has_contradiction = false;
        let mut total_confidence = 0.0f32;
        let mut count = 0usize;

        for comp in graph.compositions.values() {
            if let Some(pred) = comp.member_with_role(&SemanticRole::Predicate) {
                if pred.label == event.label {
                    same_predicate_count += 1;
                    total_confidence += comp.confidence;
                    count += 1;
                }
            }
            if comp.epistemic == EpistemicState::Contradicted {
                has_contradiction = true;
            }
        }

        GraphContextRef {
            same_predicate_count,
            has_contradiction,
            avg_confidence: if count > 0 { total_confidence / count as f32 } else { 0.0 },
            // Activation fields default to zero — no activation data available.
            activation_energy_for_predicate: 0.0,
            activation_energy_for_roles: HashMap::new(),
            top_activated_neighbors: Vec::new(),
            ambiguity_score: 0.0,
        }
    }

    /// Build enriched GraphContextRef with activation data (Phase T).
    ///
    /// After SpreadingActivation runs, `ctx.last_activation_energies` contains
    /// the activation energy for each node in the graph. This method enriches
    /// the `GraphContextRef` with that data, enabling activation-aware
    /// confidence modulation in reasoning rules.
    pub fn build_graph_context_with_activation(
        event: &SemanticAtom,
        graph: &Graph,
        activation_energies: &HashMap<NodeId, f32>,
    ) -> GraphContextRef {
        let mut same_predicate_count = 0usize;
        let mut has_contradiction = false;
        let mut total_confidence = 0.0f32;
        let mut count = 0usize;

        // Existing logic: scan compositions for predicate stats.
        for comp in graph.compositions.values() {
            if let Some(pred) = comp.member_with_role(&SemanticRole::Predicate) {
                if pred.label == event.label {
                    same_predicate_count += 1;
                    total_confidence += comp.confidence;
                    count += 1;
                }
            }
            if comp.epistemic == EpistemicState::Contradicted {
                has_contradiction = true;
            }
        }

        // NEW: Populate activation energy for predicate node.
        let mut activation_energy_for_predicate = 0.0f32;
        if let Some(&pred_node_id) = graph.label_to_id.get(&event.label) {
            activation_energy_for_predicate = activation_energies.get(&pred_node_id).copied().unwrap_or(0.0);
        }

        // NEW: Populate activation energy for role nodes.
        let mut activation_energy_for_roles = HashMap::new();
        for (role, label) in &event.roles {
            if let Some(&node_id) = graph.label_to_id.get(label) {
                let energy = activation_energies.get(&node_id).copied().unwrap_or(0.0);
                activation_energy_for_roles.insert(role.clone(), energy);
            }
        }

        // NEW: Get top activated neighbors.
        let mut energy_pairs: Vec<(NodeId, f32)> = activation_energies.iter()
            .map(|(&id, &e)| (id, e))
            .collect();
        energy_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        energy_pairs.truncate(5);
        let top_activated_neighbors = energy_pairs;

        // NEW: Compute ambiguity score.
        // Ambiguity is HIGH when top-2 role energies are SIMILAR (gap is small).
        // Previous version was inverted — it reported the gap, not the similarity.
        // Fix: ambiguity = 1 - gap, so similar energies → high ambiguity.
        let mut role_energies: Vec<f32> = activation_energy_for_roles.values().copied().collect();
        role_energies.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let ambiguity_score = if role_energies.len() >= 2 {
            let gap = (role_energies[0] - role_energies[1]).min(1.0);
            1.0 - gap // Inverted: small gap → high ambiguity
        } else {
            0.0 // Single role → no ambiguity
        };

        GraphContextRef {
            same_predicate_count,
            has_contradiction,
            avg_confidence: if count > 0 { total_confidence / count as f32 } else { 0.0 },
            activation_energy_for_predicate,
            activation_energy_for_roles,
            top_activated_neighbors,
            ambiguity_score,
        }
    }
}

/// Implement the `Transform` trait for `ReasonFrame`.
impl Transform for ReasonFrame {
    type Input = SemanticAtom;
    type Output = Vec<SemanticAtom>;

    fn id(&self) -> &'static str {
        "ReasonFrame"
    }

    fn transform(&self, input: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        let results = self.reason(input, ctx.recent_events());
        results.into_iter().map(|r| r.into_atom()).collect()
    }
}

/// Implement `ErasedTransform` for pipeline integration.
impl ErasedTransform for ReasonFrame {
    fn id(&self) -> &'static str {
        "ReasonFrame"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let mut atoms_created = 0;

        // Collect event atoms from current_atoms.
        let event_atoms: Vec<SemanticAtom> = ctx
            .current_atoms
            .iter()
            .filter(|a| a.atom_type == AtomType::Event)
            .cloned()
            .collect();

        let recent = ctx.recent_events().clone();

        for event in &event_atoms {
            // Build graph context for graph-guided reasoning.
            let graph_ref = Self::build_graph_context(event, graph);
            let results = self.reason_with_graph(event, &recent, &graph_ref);

            for result in results {
                let mut atom = result.into_atom();
                atom.id = format!("atom_{}", ctx.next_atom_id());
                ctx.current_atoms.push(atom);
                atoms_created += 1;
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
// ReReasonFrame — Post-Spreading Re-Evaluation (Phase T)
// ========================================================================

/// Post-spreading re-evaluation transform (Phase T).
///
/// After SpreadingActivation runs and populates `ctx.last_activation_energies`,
/// this transform re-evaluates event atoms with the enriched `GraphContextRef`.
/// This is the "System 2" deliberate reasoning that uses graph-based attention.
///
/// When `ambiguity_score > 0.3`, it generates an `InquiryQuestion` to surface
/// the ambiguity to the user.
///
/// # Rule Sharing (G4)
///
/// Shares the same `Arc<Vec<Box<dyn ReasoningRule>>>` as `ReasonFrame`.
/// This eliminates the duplication of constructing 4 identical rule objects.
pub struct ReReasonFrame {
    rules: Arc<Vec<Box<dyn ReasoningRule>>>,
}

impl std::fmt::Debug for ReReasonFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReReasonFrame")
            .field("rules_count", &self.rules.len())
            .finish()
    }
}

impl Clone for ReReasonFrame {
    fn clone(&self) -> Self {
        Self { rules: Arc::clone(&self.rules) }
    }
}

impl Default for ReReasonFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl ReReasonFrame {
    /// Create a new ReReasonFrame with the standard rule set.
    /// Shares rules with ReasonFrame (G4).
    pub fn new() -> Self {
        Self {
            rules: ReasonFrame::new().shared_rules(),
        }
    }

    /// Create with custom shared rules.
    pub fn with_shared_rules(rules: Arc<Vec<Box<dyn ReasoningRule>>>) -> Self {
        Self { rules }
    }
}

impl ErasedTransform for ReReasonFrame {
    fn id(&self) -> &'static str {
        "ReReasonFrame"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let mut atoms_created = 0;
        let activation_energies = ctx.last_activation_energies.clone();

        if activation_energies.is_empty() {
            // No activation data yet — skip.
            return IngestResult::new();
        }

        let event_atoms: Vec<SemanticAtom> = ctx
            .current_atoms
            .iter()
            .filter(|a| a.atom_type == AtomType::Event)
            .cloned()
            .collect();

        let recent = ctx.recent_events().clone();

        for event in &event_atoms {
            // Build enriched GraphContextRef with activation data.
            let graph_ref = ReasonFrame::build_graph_context_with_activation(
                event, graph, &activation_energies,
            );

            // Apply rules with enriched context.
            let context = ReasoningContext::with_graph(
                event, &recent, graph_ref.clone(),
            );

            for rule in self.rules.iter() {
                if rule.applies(&context) {
                    let mut rule_results = rule.generate(&context);

                    // Phase T: Activation-aware confidence modulation.
                    apply_confidence_modulation(&mut rule_results, &graph_ref);

                    // Generate clarification questions for high-ambiguity results.
                    if graph_ref.ambiguity_score > 0.5 {
                        let question = super::acquisition::InquiryQuestion {
                            question_id: format!("q_amb_{}", event.id),
                            question_text: format!(
                                "'{}' di sini memiliki beberapa interpretasi. Mana yang dimaksud?",
                                event.label
                            ),
                            gap_id: format!("amb_{}", event.id),
                            target_role: None,
                            target_composition_id: None,
                            question_type: super::acquisition::QuestionType::ChoiceBetween,
                        };
                        ctx.pending_questions.push(question);
                    }

                    for result in rule_results {
                        let mut atom = result.into_atom();
                        atom.id = format!("atom_{}", ctx.next_atom_id());
                        ctx.current_atoms.push(atom);
                        atoms_created += 1;
                    }
                }
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
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(
        label: &str,
        roles: HashMap<SemanticRole, String>,
        polarity: Option<Polarity>,
    ) -> SemanticAtom {
        SemanticAtom {
            id: "atom_test".to_string(),
            label: label.to_string(),
            atom_type: AtomType::Event,
            roles,
            polarity,
            voice: Some(Voice::Active),
            variant: Some(AtomVariant::FrameVariant(FrameSource::RuleBased)),
            confidence: 0.75,
            source: EdgeSource::FrameCompiler,
            composition_id: None,
        }
    }

    #[test]
    fn test_problem_solution_rule_applies() {
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "Raymond".to_string());
        roles.insert(SemanticRole::Arg1Patient, "aplikasi".to_string());
        roles.insert(SemanticRole::Cause, "lambat".to_string());

        let event = make_event("membuat", roles, Some(Polarity::Positive));
        let ctx = ReasoningContext::new(&event, &[]);

        let rule = ProblemSolutionRule::new();
        assert!(rule.applies(&ctx));

        let results = rule.generate(&ctx);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].atom.atom_type, AtomType::HiddenMeaning);
        assert_eq!(results[0].atom.label, "problem_solution");
        assert_eq!(
            results[0].atom.roles.get(&SemanticRole::Problem),
            Some(&"lambat".to_string())
        );
        assert_eq!(
            results[0].atom.roles.get(&SemanticRole::Solution),
            Some(&"aplikasi".to_string())
        );
    }

    #[test]
    fn test_problem_solution_rule_no_cause() {
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "Raymond".to_string());
        roles.insert(SemanticRole::Arg1Patient, "aplikasi".to_string());

        let event = make_event("membuat", roles, Some(Polarity::Positive));
        let ctx = ReasoningContext::new(&event, &[]);

        let rule = ProblemSolutionRule::new();
        assert!(!rule.applies(&ctx));
    }

    #[test]
    fn test_goal_inference_rule() {
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "Raymond".to_string());
        roles.insert(SemanticRole::Purpose, "mempercepat pekerjaan".to_string());

        let event = make_event("membuat", roles, Some(Polarity::Positive));
        let ctx = ReasoningContext::new(&event, &[]);

        let rule = GoalInferenceRule::new();
        assert!(rule.applies(&ctx));

        let results = rule.generate(&ctx);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].atom.label, "goal_inference");
        assert_eq!(
            results[0].atom.roles.get(&SemanticRole::ImpliedGoal),
            Some(&"mempercepat pekerjaan".to_string())
        );
    }

    #[test]
    fn test_polarity_conflict_rule() {
        let mut roles1 = HashMap::new();
        roles1.insert(SemanticRole::Arg0Agent, "Raymond".to_string());
        let event_positive = make_event("membuat", roles1, Some(Polarity::Positive));

        let mut roles2 = HashMap::new();
        roles2.insert(SemanticRole::Arg0Agent, "Raymond".to_string());
        let event_negative = make_event("membuat", roles2, Some(Polarity::Negative));

        let recent = vec![event_negative];
        let ctx = ReasoningContext::new(&event_positive, &recent);

        let rule = PolarityConflictRule::new();
        assert!(rule.applies(&ctx));

        let results = rule.generate(&ctx);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].atom.label, "polarity_conflict");
    }

    #[test]
    fn test_reason_frame_full() {
        let rf = ReasonFrame::new();

        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "Raymond".to_string());
        roles.insert(SemanticRole::Arg1Patient, "aplikasi".to_string());
        roles.insert(SemanticRole::Cause, "lambat".to_string());

        let event = make_event("membuat", roles, Some(Polarity::Positive));
        let results = rf.reason(&event, &[]);

        // Should produce ProblemSolutionRule result.
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.atom.label == "problem_solution"));
    }
}
