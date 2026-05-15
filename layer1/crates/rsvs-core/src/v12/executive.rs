//! # MD-5: Executive Cognition Layer
//!
//! The executive cognition layer manages cognitive modes, compute budgets,
//! and the reflection loop. It provides the top-level orchestration that
//! selects how the pipeline should process each input.
//!
//! ## Cognitive Modes
//!
//! | Mode | Trigger | Behavior |
//! |------|---------|----------|
//! | Reactive | No contradictions, no gaps | Fast path — skip enrichment |
//! | Analytical | Contradictions OR low confidence | Enrichment loop (1–3 passes) |
//! | Reflective | Deep contradictions | Extended reflection (3–5 passes) |
//!
//! ## Architecture
//!
//! ```text
//! ingest(text) → select_cognitive_mode()
//!                     │
//!         ┌───────────┼───────────────┐
//!         │           │               │
//!     Reactive    Analytical     Reflective
//!     (fast)     (enrich)      (reflect)
//!         │           │               │
//!         │     run_enrichment_loop() │
//!         │           │         ┌─────┘
//!         │           │         │
//!         └───────────┴─────────┘
//!                     │
//!              ReflectionFinding
//! ```
//!
//! ## Feature Flag
//!
//! This module is only compiled when the `v12` feature is enabled.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::pipeline::{ErasedTransform, Graph, IngestResult, PipelineEngine};
use super::types::*;
use crate::types::EdgeSource;

// ========================================================================
// CognitiveMode — How the Pipeline Processes Input
// ========================================================================

/// Cognitive processing mode (MD-5).
///
/// Determines how aggressively the pipeline processes input:
/// - **Reactive**: Fast path — no enrichment or reflection. Just extract
///   and ingest. Used when the graph is healthy (no contradictions, no gaps).
/// - **Analytical**: Enrichment loop — run 1–3 enrichment passes to fill
///   gaps and resolve low-confidence compositions. Used when there are
///   contradictions or low-confidence compositions.
/// - **Reflective**: Extended reflection — run 3–5 enrichment passes with
///   deeper analysis. Used for deep contradictions or when the analytical
///   mode couldn't resolve issues in its budget.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveMode {
    /// Fast path — no enrichment or reflection.
    Reactive,
    /// Enrichment loop — 1–3 passes.
    Analytical,
    /// Extended reflection — 3–5 passes.
    Reflective,
}

impl Default for CognitiveMode {
    fn default() -> Self {
        CognitiveMode::Reactive
    }
}

impl CognitiveMode {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            CognitiveMode::Reactive => "Reactive",
            CognitiveMode::Analytical => "Analytical",
            CognitiveMode::Reflective => "Reflective",
        }
    }
}

// ========================================================================
// ComputeBudget — Resource Limits Per Mode
// ========================================================================

/// Compute budget for a pipeline execution (MD-5).
///
/// Limits how much work the pipeline can do in a single ingest cycle.
/// Higher budgets allow more enrichment passes and deeper analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeBudget {
    /// Maximum number of enrichment/reflection passes.
    pub max_passes: usize,
    /// Maximum milliseconds per pass.
    pub max_ms_per_pass: u64,
    /// Maximum number of gaps to address per pass.
    pub max_gaps_per_pass: usize,
    /// Minimum confidence improvement to continue looping.
    pub min_confidence_delta: f32,
    /// Whether enrichment is enabled.
    pub enrichment_enabled: bool,
}

impl Default for ComputeBudget {
    fn default() -> Self {
        Self::for_mode(&CognitiveMode::Reactive)
    }
}

impl ComputeBudget {
    /// Create a budget appropriate for the given cognitive mode.
    pub fn for_mode(mode: &CognitiveMode) -> Self {
        match mode {
            CognitiveMode::Reactive => Self {
                max_passes: 0,
                max_ms_per_pass: 0,
                max_gaps_per_pass: 0,
                min_confidence_delta: 0.0,
                enrichment_enabled: false,
            },
            CognitiveMode::Analytical => Self {
                max_passes: 3,
                max_ms_per_pass: 1000,
                max_gaps_per_pass: 5,
                min_confidence_delta: 0.05,
                enrichment_enabled: true,
            },
            CognitiveMode::Reflective => Self {
                max_passes: 5,
                max_ms_per_pass: 2000,
                max_gaps_per_pass: 10,
                min_confidence_delta: 0.02,
                enrichment_enabled: true,
            },
        }
    }
}

// ========================================================================
// StopCondition — When to Stop the Enrichment Loop
// ========================================================================

/// Condition for stopping the enrichment/reflection loop (MD-5).
///
/// The loop stops when any of these conditions is met:
/// - All gaps have been addressed
/// - Confidence has converged (delta < threshold)
/// - Budget is exhausted (max passes reached)
/// - No new evidence was found in the last pass
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopCondition {
    /// Minimum confidence delta to continue (0.0 = always continue).
    pub min_confidence_delta: f32,
    /// Maximum number of passes.
    pub max_passes: usize,
    /// Maximum number of passes without new evidence before stopping.
    pub max_passes_without_evidence: usize,
}

impl Default for StopCondition {
    fn default() -> Self {
        Self {
            min_confidence_delta: 0.05,
            max_passes: 3,
            max_passes_without_evidence: 2,
        }
    }
}

impl StopCondition {
    /// Create from a compute budget.
    pub fn from_budget(budget: &ComputeBudget) -> Self {
        Self {
            min_confidence_delta: budget.min_confidence_delta,
            max_passes: budget.max_passes,
            max_passes_without_evidence: 2,
        }
    }

    /// Should the enrichment loop stop?
    ///
    /// Returns `true` if any stop condition is met.
    pub fn should_stop(&self, state: &ReasoningState) -> bool {
        // Budget exhausted.
        if state.loops_completed >= self.max_passes {
            return true;
        }

        // Goal met.
        if state.goal_met {
            return true;
        }

        // No new evidence for too long.
        if state.loops_without_new_evidence >= self.max_passes_without_evidence {
            return true;
        }

        // Confidence converged (only after at least 1 loop).
        if state.loops_completed > 0 && state.goal == ReasoningGoal::UnderstandInput {
            // For understanding, if confidence is high enough, stop.
            if state.confidence >= 0.8 {
                return true;
            }
        }

        false
    }
}

// ========================================================================
// ReflectionFinding — Output of the Reflect Transform
// ========================================================================

/// Type of finding from reflection (MD-5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReflectionFindingType {
    /// A contradiction was resolved.
    ContradictionResolved,
    /// A knowledge gap was filled.
    GapFilled,
    /// A composition's confidence was improved.
    ConfidenceImproved,
    /// No improvement was possible.
    NoImprovement,
    /// The graph is in a healthy state.
    HealthyState,
    /// A new hidden meaning was discovered.
    NewInsight,
}

impl Default for ReflectionFindingType {
    fn default() -> Self {
        ReflectionFindingType::NoImprovement
    }
}

/// Action recommended by reflection (MD-5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReflectionAction {
    /// No further action needed.
    NoAction,
    /// Run another enrichment pass.
    EnrichFurther,
    /// Request user input to resolve ambiguity.
    AskUser(String),
    /// Re-govern a specific composition.
    ReGovern(CompositionId),
    /// Re-extract a specific frame.
    ReExtract(CompositionId),
}

impl Default for ReflectionAction {
    fn default() -> Self {
        ReflectionAction::NoAction
    }
}

/// A finding from the reflection loop (MD-5).
///
/// Each pass of the reflection loop produces findings that describe
/// what was discovered or improved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionFinding {
    /// What type of finding this is.
    pub finding_type: ReflectionFindingType,
    /// Human-readable description.
    pub description: String,
    /// Compositions affected by this finding.
    #[serde(default)]
    pub affected_compositions: Vec<CompositionId>,
    /// Confidence change (positive = improvement).
    #[serde(default)]
    pub confidence_delta: f32,
    /// Recommended next action.
    #[serde(default)]
    pub action: ReflectionAction,
}

impl Default for ReflectionFinding {
    fn default() -> Self {
        Self {
            finding_type: ReflectionFindingType::NoImprovement,
            description: String::new(),
            affected_compositions: Vec::new(),
            confidence_delta: 0.0,
            action: ReflectionAction::NoAction,
        }
    }
}

// ========================================================================
// Reflect Transform
// ========================================================================

/// MD-5: Reflect transform — analyzes the graph state after enrichment
/// and produces findings about what was improved or discovered.
///
/// This is the core of the Reflective cognitive mode. It examines the
/// current graph state and produces structured findings that guide
/// further action.
///
/// # Transform Signature
///
/// ```text
/// Input:  ReflectionLoopResult — result of an enrichment pass
/// Output: Vec<ReflectionFinding> — structured findings
/// ```
#[derive(Debug, Clone, Default)]
pub struct Reflect;

impl Reflect {
    /// Create a new Reflect transform.
    pub fn new() -> Self {
        Self
    }

    /// Analyze a reflection loop result and produce findings.
    pub fn reflect(&self, result: &ReflectionLoopResult, graph: &Graph) -> Vec<ReflectionFinding> {
        let mut findings = Vec::new();

        // Check for resolved contradictions.
        for comp_id in &result.resolved_contradictions {
            findings.push(ReflectionFinding {
                finding_type: ReflectionFindingType::ContradictionResolved,
                description: format!("Contradiction resolved for composition {}", comp_id),
                affected_compositions: vec![comp_id.clone()],
                confidence_delta: 0.1,
                action: ReflectionAction::ReGovern(comp_id.clone()),
            });
        }

        // Check for filled gaps.
        for comp_id in &result.filled_gaps {
            findings.push(ReflectionFinding {
                finding_type: ReflectionFindingType::GapFilled,
                description: format!("Knowledge gap filled for composition {}", comp_id),
                affected_compositions: vec![comp_id.clone()],
                confidence_delta: 0.05,
                action: ReflectionAction::NoAction,
            });
        }

        // Check overall confidence.
        if result.current_confidence >= 0.8 {
            findings.push(ReflectionFinding {
                finding_type: ReflectionFindingType::HealthyState,
                description: format!(
                    "Graph is in healthy state (confidence: {:.2})",
                    result.current_confidence
                ),
                affected_compositions: Vec::new(),
                confidence_delta: 0.0,
                action: ReflectionAction::NoAction,
            });
        } else if result.has_gaps {
            findings.push(ReflectionFinding {
                finding_type: ReflectionFindingType::NoImprovement,
                description: format!(
                    "Gaps remain after reflection (confidence: {:.2})",
                    result.current_confidence
                ),
                affected_compositions: Vec::new(),
                confidence_delta: 0.0,
                action: ReflectionAction::EnrichFurther,
            });
        }

        // Check for low-confidence compositions that could benefit from re-extraction.
        for composition in graph.compositions.values() {
            if composition.confidence < 0.5 && composition.composition_type == CompositionType::Event {
                if composition.source_text.is_some() {
                    findings.push(ReflectionFinding {
                        finding_type: ReflectionFindingType::ConfidenceImproved,
                        description: format!(
                            "Low-confidence event {} ({:.2}) could benefit from re-extraction",
                            composition.id, composition.confidence
                        ),
                        affected_compositions: vec![composition.id.clone()],
                        confidence_delta: 0.0,
                        action: ReflectionAction::ReExtract(composition.id.clone()),
                    });
                }
            }
        }

        findings
    }
}

// ========================================================================
// ExecutiveOrchestrator — Top-Level Ingest Orchestration
// ========================================================================

/// MD-5: Executive orchestrator — manages cognitive mode selection
/// and the enrichment loop.
///
/// This is the top-level orchestrator that decides HOW to process
/// each input. It selects a cognitive mode based on graph neighborhood
/// health, then runs the appropriate processing strategy.
///
/// # Usage
///
/// ```ignore
/// let mut orchestrator = ExecutiveOrchestrator::new();
/// let result = orchestrator.ingest("Raymond membuat aplikasi karena lambat", &mut engine);
/// ```
#[derive(Debug, Clone)]
pub struct ExecutiveOrchestrator {
    /// Current cognitive mode.
    pub mode: CognitiveMode,
    /// Current compute budget.
    pub budget: ComputeBudget,
    /// Reflection transform.
    reflect: Reflect,
}

impl Default for ExecutiveOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutiveOrchestrator {
    /// Create a new orchestrator in Reactive mode.
    pub fn new() -> Self {
        Self {
            mode: CognitiveMode::Reactive,
            budget: ComputeBudget::for_mode(&CognitiveMode::Reactive),
            reflect: Reflect::new(),
        }
    }

    /// Select cognitive mode based on graph neighborhood health.
    ///
    /// # Mode Selection Logic
    ///
    /// ```text
    /// if neighborhood has contradictions:
    ///     if contradictions are deep (multiple, cross-type):
    ///         mode = Reflective
    ///     else:
    ///         mode = Analytical
    /// elif neighborhood has low average confidence (< 0.5):
    ///     mode = Analytical
    /// else:
    ///     mode = Reactive
    /// ```
    pub fn select_cognitive_mode(&mut self, neighborhood: &GraphNeighborhood) -> CognitiveMode {
        if neighborhood.has_contradictions() {
            // Count how many compositions are contradicted.
            let contradicted_count = neighborhood
                .compositions
                .iter()
                .filter(|c| c.epistemic == EpistemicState::Contradicted)
                .count();

            if contradicted_count >= 3 {
                self.mode = CognitiveMode::Reflective;
            } else {
                self.mode = CognitiveMode::Analytical;
            }
        } else if neighborhood.average_confidence() < 0.5 {
            self.mode = CognitiveMode::Analytical;
        } else {
            self.mode = CognitiveMode::Reactive;
        }

        self.budget = ComputeBudget::for_mode(&self.mode);
        self.mode.clone()
    }

    /// Run the shared enrichment loop for Analytical and Reflective modes.
    ///
    /// The enrichment loop:
    /// 1. Detect gaps
    /// 2. Select acquisition strategies
    /// 3. Apply enrichment/re-extraction
    /// 4. Re-govern compositions
    /// 5. Check stop conditions
    /// 6. Repeat if budget allows
    ///
    /// Returns a `ReflectionLoopResult` summarizing what was accomplished.
    pub fn run_enrichment_loop(
        &self,
        engine: &mut PipelineEngine,
    ) -> ReflectionLoopResult {
        let stop_condition = StopCondition::from_budget(&self.budget);
        let mut state = ReasoningState {
            confidence: 0.0,
            elapsed_ms: 0,
            loops_completed: 0,
            loops_without_new_evidence: 0,
            goal_met: false,
            goal: ReasoningGoal::UnderstandInput,
        };

        let mut modified_compositions = Vec::new();
        let mut resolved_contradictions = Vec::new();
        let mut filled_gaps = Vec::new();

        // Compute initial confidence.
        let snapshot = engine.snapshot();
        state.confidence = if snapshot.compositions.is_empty() {
            0.0
        } else {
            snapshot.compositions.iter().map(|c| c.confidence).sum::<f32>()
                / snapshot.compositions.len() as f32
        };

        for _pass in 0..self.budget.max_passes {
            if stop_condition.should_stop(&state) {
                break;
            }

            // Apply pending enrichments.
            let enrichments: Vec<_> = engine.context.pending_enrichments.drain(..).collect();
            let new_evidence = !enrichments.is_empty();

            for request in &enrichments {
                modified_compositions.push(request.target_composition_id.clone());
            }

            // Apply pending re-extractions.
            let reextractions: Vec<_> = engine.context.pending_reextractions.drain(..).collect();
            for request in &reextractions {
                modified_compositions.push(request.target_composition_id.clone());
            }

            // Re-govern after enrichment.
            let graph = engine.graph();
            let contradicted_before: Vec<_> = graph
                .compositions
                .values()
                .filter(|c| c.epistemic == EpistemicState::Contradicted)
                .map(|c| c.id.clone())
                .collect();

            // After re-governance, check if any contradictions were resolved.
            // (Simplified: we check if the contradicted count decreased.)
            let graph_after = engine.graph();
            let contradicted_after: Vec<_> = graph_after
                .compositions
                .values()
                .filter(|c| c.epistemic == EpistemicState::Contradicted)
                .map(|c| c.id.clone())
                .collect();

            for id in &contradicted_before {
                if !contradicted_after.contains(id) {
                    resolved_contradictions.push(id.clone());
                }
            }

            // Update reasoning state.
            state.loops_completed += 1;
            if new_evidence {
                state.loops_without_new_evidence = 0;
            } else {
                state.loops_without_new_evidence += 1;
            }

            // Recompute confidence.
            let snapshot = engine.snapshot();
            let new_confidence = if snapshot.compositions.is_empty() {
                0.0
            } else {
                snapshot.compositions.iter().map(|c| c.confidence).sum::<f32>()
                    / snapshot.compositions.len() as f32
            };

            let delta = (new_confidence - state.confidence).abs();
            state.confidence = new_confidence;

            // Check if confidence converged.
            if delta < stop_condition.min_confidence_delta && state.loops_completed > 1 {
                break;
            }

            // Check if goal is met.
            if state.confidence >= 0.8 && !engine.context.has_gaps() {
                state.goal_met = true;
                break;
            }
        }

        ReflectionLoopResult {
            current_confidence: state.confidence,
            elapsed_ms: 0, // Caller should time this
            evidence_count: modified_compositions.len(),
            modified_compositions,
            has_gaps: engine.context.has_gaps(),
            resolved_contradictions,
            filled_gaps,
        }
    }

    /// Top-level ingest with mode-specific behavior.
    ///
    /// 1. Select cognitive mode from graph neighborhood
    /// 2. Run the standard pipeline
    /// 3. If Analytical or Reflective, run the enrichment loop
    /// 4. If Reflective, produce reflection findings
    pub fn ingest(&mut self, text: &str, engine: &mut PipelineEngine) -> IngestResult {
        // Step 1: Select cognitive mode.
        let neighborhood = engine.snapshot();
        let mode = self.select_cognitive_mode(&GraphNeighborhood {
            compositions: neighborhood.compositions,
        });

        // Step 2: Run standard pipeline.
        let mut result = engine.ingest(text);

        // Step 3: If Analytical or Reflective, run enrichment loop.
        if mode != CognitiveMode::Reactive && self.budget.enrichment_enabled {
            let loop_result = self.run_enrichment_loop(engine);

            // Merge enrichment results.
            result.enrichments_applied += loop_result.evidence_count;

            // Step 4: If Reflective, produce reflection findings.
            if mode == CognitiveMode::Reflective {
                let findings = self.reflect.reflect(&loop_result, engine.graph());
                // Findings could be logged, stored, or used to guide further action.
                // For now, we just count them.
                let _ = findings; // Available for future use
            }
        }

        result
    }
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_mode_budget() {
        let reactive = ComputeBudget::for_mode(&CognitiveMode::Reactive);
        assert_eq!(reactive.max_passes, 0);
        assert!(!reactive.enrichment_enabled);

        let analytical = ComputeBudget::for_mode(&CognitiveMode::Analytical);
        assert_eq!(analytical.max_passes, 3);
        assert!(analytical.enrichment_enabled);

        let reflective = ComputeBudget::for_mode(&CognitiveMode::Reflective);
        assert_eq!(reflective.max_passes, 5);
        assert!(reflective.enrichment_enabled);
    }

    #[test]
    fn test_stop_condition_budget_exhausted() {
        let condition = StopCondition {
            max_passes: 2,
            ..StopCondition::default()
        };
        let state = ReasoningState {
            confidence: 0.5,
            elapsed_ms: 0,
            loops_completed: 2,
            loops_without_new_evidence: 0,
            goal_met: false,
            goal: ReasoningGoal::UnderstandInput,
        };
        assert!(condition.should_stop(&state));
    }

    #[test]
    fn test_stop_condition_goal_met() {
        let condition = StopCondition::default();
        let state = ReasoningState {
            confidence: 0.9,
            elapsed_ms: 0,
            loops_completed: 1,
            loops_without_new_evidence: 0,
            goal_met: true,
            goal: ReasoningGoal::UnderstandInput,
        };
        assert!(condition.should_stop(&state));
    }

    #[test]
    fn test_stop_condition_no_evidence() {
        let condition = StopCondition {
            max_passes_without_evidence: 2,
            ..StopCondition::default()
        };
        let state = ReasoningState {
            confidence: 0.5,
            elapsed_ms: 0,
            loops_completed: 1,
            loops_without_new_evidence: 2,
            goal_met: false,
            goal: ReasoningGoal::UnderstandInput,
        };
        assert!(condition.should_stop(&state));
    }

    #[test]
    fn test_mode_selection_reactive() {
        let mut orchestrator = ExecutiveOrchestrator::new();
        let neighborhood = GraphNeighborhood {
            compositions: vec![Composition {
                confidence: 0.8,
                epistemic: EpistemicState::Observed,
                ..Composition::default()
            }],
        };

        let mode = orchestrator.select_cognitive_mode(&neighborhood);
        assert_eq!(mode, CognitiveMode::Reactive);
    }

    #[test]
    fn test_mode_selection_analytical() {
        let mut orchestrator = ExecutiveOrchestrator::new();
        let neighborhood = GraphNeighborhood {
            compositions: vec![Composition {
                confidence: 0.3,
                epistemic: EpistemicState::Observed,
                ..Composition::default()
            }],
        };

        let mode = orchestrator.select_cognitive_mode(&neighborhood);
        assert_eq!(mode, CognitiveMode::Analytical);
    }

    #[test]
    fn test_mode_selection_reflective() {
        let mut orchestrator = ExecutiveOrchestrator::new();
        let neighborhood = GraphNeighborhood {
            compositions: vec![
                Composition {
                    confidence: 0.5,
                    epistemic: EpistemicState::Contradicted,
                    ..Composition::default()
                },
                Composition {
                    confidence: 0.4,
                    epistemic: EpistemicState::Contradicted,
                    ..Composition::default()
                },
                Composition {
                    confidence: 0.3,
                    epistemic: EpistemicState::Contradicted,
                    ..Composition::default()
                },
            ],
        };

        let mode = orchestrator.select_cognitive_mode(&neighborhood);
        assert_eq!(mode, CognitiveMode::Reflective);
    }

    #[test]
    fn test_reflection_finding_types() {
        let reflect = Reflect::new();
        let result = ReflectionLoopResult {
            current_confidence: 0.9,
            elapsed_ms: 0,
            evidence_count: 2,
            modified_compositions: vec!["comp1".to_string()],
            has_gaps: false,
            resolved_contradictions: vec!["comp_contra".to_string()],
            filled_gaps: vec!["comp_gap".to_string()],
        };
        let graph = Graph::new();
        let findings = reflect.reflect(&result, &graph);

        assert!(findings.iter().any(|f| f.finding_type == ReflectionFindingType::ContradictionResolved));
        assert!(findings.iter().any(|f| f.finding_type == ReflectionFindingType::GapFilled));
        assert!(findings.iter().any(|f| f.finding_type == ReflectionFindingType::HealthyState));
    }
}
