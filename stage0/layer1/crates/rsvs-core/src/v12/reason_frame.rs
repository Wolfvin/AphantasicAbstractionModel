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

use super::pipeline::{ErasedTransform, Graph, IngestResult};
use super::types::*;
use crate::types::EdgeSource;

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
    /// TODO: Replace with actual Graph reference when available.
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
/// Provides just enough information for rules to adjust confidence
/// based on graph structure, without requiring a full `Graph` reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphContextRef {
    /// Number of compositions with the same predicate.
    #[serde(default)]
    pub same_predicate_count: usize,
    /// Whether a contradiction exists for this predicate.
    #[serde(default)]
    pub has_contradiction: bool,
    /// Average confidence of compositions with the same predicate.
    #[serde(default)]
    pub avg_confidence: f32,
}

impl Default for GraphContextRef {
    fn default() -> Self {
        Self {
            same_predicate_count: 0,
            has_contradiction: false,
            avg_confidence: 0.0,
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
pub struct ReasonFrame {
    /// The reasoning rules to apply, in order.
    rules: Vec<Box<dyn ReasoningRule>>,
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
        // Rules are stateless deterministic — recreate them with the same default set.
        Self::new()
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
            rules: vec![
                Box::new(ProblemSolutionRule::new()),
                Box::new(GoalInferenceRule::new()),
                Box::new(PolarityConflictRule::new()),
                Box::new(ConditionConsequenceRule::new()),
            ],
        }
    }

    /// Create with custom rules.
    pub fn with_rules(rules: Vec<Box<dyn ReasoningRule>>) -> Self {
        Self { rules }
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

        for rule in &self.rules {
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

        for rule in &self.rules {
            if rule.applies(&context) {
                let mut rule_results = rule.generate(&context);

                // Graph-guided confidence adjustment stub.
                // If the graph has many compositions with the same predicate,
                // boost confidence slightly. If there's a contradiction,
                // reduce confidence.
                for result in &mut rule_results {
                    if graph_ref.same_predicate_count > 0 {
                        let boost = (graph_ref.same_predicate_count as f32 * 0.02).min(0.10);
                        result.derivation_confidence =
                            (result.derivation_confidence + boost).min(1.0);
                        result.atom.confidence = result.derivation_confidence;
                    }
                    if graph_ref.has_contradiction {
                        result.derivation_confidence *= 0.85;
                        result.atom.confidence = result.derivation_confidence;
                    }
                }

                results.extend(rule_results);
            }
        }

        results
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

    fn execute(&self, ctx: &mut PipelineContext, _graph: &mut Graph) -> IngestResult {
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
            let results = self.reason(event, &recent);

            for result in results {
                let mut atom = result.into_atom();
                atom.id = format!("atom_{}", ctx.next_atom_id());
                ctx.current_atoms.push(atom);
                atoms_created += 1;
            }
        }

        IngestResult {
            atoms_created,
            ..IngestResult::default()
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
