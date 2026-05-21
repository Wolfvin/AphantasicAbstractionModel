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

use super::acquisition::{AcquisitionStrategy, DetectGaps, SelectAcquisition};
use super::govern_beliefs::GovernBeliefs;
use super::pipeline::{EnrichComposition, ErasedTransform, Graph, IngestResult, PipelineEngine};
use super::types::*;
use crate::types::NodeId;
use serde::{Deserialize, Serialize};

// ========================================================================
// Named Constants — Audit v6 fix
// ========================================================================

/// Batch threshold above which an Inferred composition is considered stagnant.
const STAGNANT_BATCH_THRESHOLD: usize = 10;
/// Confidence below which a composition is considered decayed.
const DECAYED_CONFIDENCE_FLOOR: f32 = 0.2;
/// Batch threshold above which decayed-confidence detection kicks in.
const DECAYED_BATCH_THRESHOLD: usize = 5;
/// Number of contradicted compositions that triggers Reflective mode.
const REFLECTIVE_CONTRADICTION_THRESHOLD: usize = 3;
/// Confidence threshold below which Analytical mode is triggered.
const ANALYTICAL_CONFIDENCE_FLOOR: f32 = 0.5;
/// Confidence threshold for goal-met detection in enrichment loop.
const GOAL_MET_CONFIDENCE: f32 = 0.8;
/// Minimum batch_seen and confidence for reflection-driven promotion.
const REFLECTION_PROMOTION_MIN_AGE: usize = 3;
const REFLECTION_PROMOTION_MIN_CONFIDENCE: f32 = 0.6;

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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CognitiveMode {
    /// Fast path — no enrichment or reflection.
    #[default]
    Reactive,
    /// Enrichment loop — 1–3 passes.
    Analytical,
    /// Extended reflection — 3–5 passes.
    Reflective,
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

/// Compute budget for cognitive mode execution (MD-5 spec).
///
/// Controls resource limits for different cognitive modes:
/// - Reactive: minimal (0 enrichment rounds, 1 reasoning depth)
/// - Analytical: moderate (1 enrichment round, 3 reasoning depth)
/// - Reflective: generous (2 enrichment rounds, 5 reasoning depth)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeBudget {
    /// Maximum depth of reasoning chain.
    pub max_reasoning_depth: usize,
    /// Maximum number of reflection loops.
    pub max_reflection_loops: usize,
    /// Maximum branching factor for hypothesis exploration.
    pub max_branching_factor: usize,
    /// Maximum number of hypotheses to maintain.
    pub max_hypothesis_count: usize,
    /// Time budget in milliseconds.
    pub time_budget_ms: u64,
    /// Maximum enrichment rounds (Reactive=0, Analytical=1, Reflective=2).
    pub max_enrichment_rounds: usize,
}

impl Default for ComputeBudget {
    fn default() -> Self {
        Self::analytical()
    }
}

impl ComputeBudget {
    /// Budget for Reactive mode: minimal resources.
    pub fn reactive() -> Self {
        Self {
            max_reasoning_depth: 1,
            max_reflection_loops: 0,
            max_branching_factor: 1,
            max_hypothesis_count: 1,
            time_budget_ms: 100,
            max_enrichment_rounds: 0,
        }
    }

    /// Budget for Analytical mode: moderate resources.
    pub fn analytical() -> Self {
        Self {
            max_reasoning_depth: 3,
            max_reflection_loops: 1,
            max_branching_factor: 3,
            max_hypothesis_count: 5,
            time_budget_ms: 1000,
            max_enrichment_rounds: 1,
        }
    }

    /// Budget for Reflective mode: generous resources.
    pub fn reflective() -> Self {
        Self {
            max_reasoning_depth: 5,
            max_reflection_loops: 3,
            max_branching_factor: 5,
            max_hypothesis_count: 10,
            time_budget_ms: 5000,
            max_enrichment_rounds: 2,
        }
    }

    /// Create a budget appropriate for the given cognitive mode.
    pub fn for_mode(mode: &CognitiveMode) -> Self {
        match mode {
            CognitiveMode::Reactive => Self::reactive(),
            CognitiveMode::Analytical => Self::analytical(),
            CognitiveMode::Reflective => Self::reflective(),
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
            min_confidence_delta: 0.05,
            max_passes: budget.max_enrichment_rounds,
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

/// Types of findings from reflection (MD-5 spec).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReflectionFindingType {
    /// Composition has been Inferred for >10 batches without progress.
    StagnantInferred,
    /// Composition meets promotion criteria.
    PromotionCandidate,
    /// Contradiction can potentially be resolved.
    ContradictionResolvable,
    /// Confidence has decayed below threshold after many batches.
    DecayedConfidence,
    /// Two compositions have overlapping structure.
    OverlapDetected,
}

/// Actions proposed by reflection findings (MD-5 spec).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ReflectionAction {
    /// Propose promoting a composition to a higher lifecycle/epistemic state.
    ProposePromotion(CompositionId),
    /// Propose resolving a contradiction between two compositions.
    ProposeContradictionResolution(CompositionId, CompositionId),
    /// Propose deprecating a composition that has decayed.
    ProposeDeprecation(CompositionId),
    /// Propose merging two overlapping compositions.
    ProposeMerge(CompositionId, CompositionId),
    /// No action needed.
    #[default]
    NoAction,
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
            finding_type: ReflectionFindingType::StagnantInferred,
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
///
/// # Performance
///
/// The overlap detection step is O(N²) in the number of compositions.
/// To prevent runaway computation on large graphs, the `max_overlap_pairs`
/// field caps the number of composition pairs checked per reflection pass.
/// Default: 500 (matching `ConvergenceConfig::max_pairs_per_run`).
#[derive(Debug, Clone)]
pub struct Reflect {
    /// Maximum number of composition pairs to check for overlap per reflection pass.
    /// Prevents O(N²) blow-up on large graphs.
    pub max_overlap_pairs: usize,
}

impl Default for Reflect {
    fn default() -> Self {
        Self {
            max_overlap_pairs: 500,
        }
    }
}

impl Reflect {
    /// Create a new Reflect transform with default throttle (500 pairs).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a Reflect transform with a custom overlap throttle.
    pub fn with_max_overlap_pairs(max: usize) -> Self {
        Self {
            max_overlap_pairs: max,
        }
    }

    /// Analyze a reflection loop result and produce findings.
    pub fn reflect(&self, result: &ReflectionLoopResult, graph: &Graph) -> Vec<ReflectionFinding> {
        let mut findings = Vec::new();

        // Check for resolved contradictions — now ContradictionResolvable.
        for comp_id in &result.resolved_contradictions {
            findings.push(ReflectionFinding {
                finding_type: ReflectionFindingType::ContradictionResolvable,
                description: format!("Contradiction can be resolved for composition {}", comp_id),
                affected_compositions: vec![comp_id.clone()],
                confidence_delta: 0.1,
                action: ReflectionAction::NoAction,
            });
        }

        // Check for filled gaps — compositions with improved confidence are PromotionCandidate.
        for comp_id in &result.filled_gaps {
            findings.push(ReflectionFinding {
                finding_type: ReflectionFindingType::PromotionCandidate,
                description: format!("Composition {} improved after gap filling", comp_id),
                affected_compositions: vec![comp_id.clone()],
                confidence_delta: 0.05,
                action: ReflectionAction::ProposePromotion(comp_id.clone()),
            });
        }

        // Check for stagnant inferred compositions.
        for composition in graph.compositions.values() {
            if composition.epistemic == EpistemicState::Inferred && composition.batch_seen > STAGNANT_BATCH_THRESHOLD {
                findings.push(ReflectionFinding {
                    finding_type: ReflectionFindingType::StagnantInferred,
                    description: format!(
                        "Composition {} has been Inferred for {} batches without progress",
                        composition.id, composition.batch_seen
                    ),
                    affected_compositions: vec![composition.id.clone()],
                    confidence_delta: 0.0,
                    action: ReflectionAction::ProposeDeprecation(composition.id.clone()),
                });
            }
        }

        // Check for decayed confidence compositions.
        for composition in graph.compositions.values() {
            if composition.confidence < DECAYED_CONFIDENCE_FLOOR && composition.batch_seen > DECAYED_BATCH_THRESHOLD {
                findings.push(ReflectionFinding {
                    finding_type: ReflectionFindingType::DecayedConfidence,
                    description: format!(
                        "Composition {} has decayed confidence ({:.2}) after {} batches",
                        composition.id, composition.confidence, composition.batch_seen
                    ),
                    affected_compositions: vec![composition.id.clone()],
                    confidence_delta: 0.0,
                    action: ReflectionAction::ProposeDeprecation(composition.id.clone()),
                });
            }
        }

        // Check for overlapping compositions.
        // O(N²) nested loop — throttled by max_overlap_pairs to prevent
        // runaway computation on large graphs. This mirrors the throttle
        // pattern used in ConvergenceDetection::detect().
        let comp_list: Vec<_> = graph.compositions.values().collect();
        let mut checked: usize = 0;
        for i in 0..comp_list.len() {
            if checked >= self.max_overlap_pairs { break; }
            for j in (i + 1)..comp_list.len() {
                if checked >= self.max_overlap_pairs { break; }
                checked += 1;
                let a = comp_list[i];
                let b = comp_list[j];
                if a.composition_type == b.composition_type {
                    let a_nodes: std::collections::HashSet<NodeId> =
                        a.members.iter().map(|m| m.node_id).collect();
                    let b_nodes: std::collections::HashSet<NodeId> =
                        b.members.iter().map(|m| m.node_id).collect();
                    let overlap = a_nodes.intersection(&b_nodes).count();
                    if overlap > 0 && overlap > a_nodes.len().min(b_nodes.len()) / 2 {
                        findings.push(ReflectionFinding {
                            finding_type: ReflectionFindingType::OverlapDetected,
                            description: format!(
                                "Compositions {} and {} have overlapping structure",
                                a.id, b.id
                            ),
                            affected_compositions: vec![a.id.clone(), b.id.clone()],
                            confidence_delta: 0.0,
                            action: ReflectionAction::ProposeMerge(a.id.clone(), b.id.clone()),
                        });
                    }
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
    /// Uses `extract_keywords()` on the input text and `neighborhood_for()`
    /// to build a relevant `GraphNeighborhood`, then applies:
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
    pub fn select_cognitive_mode(
        &mut self,
        input: &str,
        compositions: &[Composition],
    ) -> CognitiveMode {
        let keywords = extract_keywords(input);
        let neighborhood = GraphNeighborhood::neighborhood_for(&keywords, compositions);

        // If no relevant neighborhood found, fall back to all compositions
        // so that mode selection still works based on overall graph health.
        let neighborhood = if neighborhood.compositions.is_empty() {
            GraphNeighborhood {
                compositions: compositions.to_vec(),
            }
        } else {
            neighborhood
        };

        if neighborhood.has_contradictions() {
            // Count how many compositions are contradicted.
            let contradicted_count = neighborhood
                .compositions
                .iter()
                .filter(|c| c.epistemic == EpistemicState::Contradicted)
                .count();

            if contradicted_count >= REFLECTIVE_CONTRADICTION_THRESHOLD {
                self.mode = CognitiveMode::Reflective;
            } else {
                self.mode = CognitiveMode::Analytical;
            }
        } else if neighborhood.average_confidence() < ANALYTICAL_CONFIDENCE_FLOOR {
            self.mode = CognitiveMode::Analytical;
        } else {
            self.mode = CognitiveMode::Reactive;
        }

        self.budget = ComputeBudget::for_mode(&self.mode);
        self.mode.clone()
    }

    /// Run the active enrichment loop for Analytical and Reflective modes.
    ///
    /// The enrichment loop actively:
    /// 1. Snapshot the graph and detect gaps via `DetectGaps`
    /// 2. Select acquisition strategies via `SelectAcquisition`
    /// 3. Convert `PassiveRecall` decisions into `EnrichmentRequest`s
    /// 4. Apply enrichments via `EnrichComposition`
    /// 5. Re-govern affected compositions via `GovernBeliefs`
    /// 6. Update state tracking (evidence_count, confidence, stagnation)
    /// 7. Check stop conditions and repeat if budget allows
    ///
    /// Returns a `ReflectionLoopResult` summarizing what was accomplished.
    pub fn run_enrichment_loop(&self, engine: &mut PipelineEngine) -> ReflectionLoopResult {
        let stop_condition = StopCondition::from_budget(&self.budget);
        let mut state = ReasoningState {
            confidence: 0.0,
            elapsed_ms: 0,
            loops_completed: 0,
            loops_without_new_evidence: 0,
            goal_met: false,
            goal: ReasoningGoal::UnderstandInput,
            modified_compositions: Vec::new(),
            evidence_count: 0,
            evidence_at_loop_start: 0,
        };

        let mut modified_compositions = Vec::new();
        let resolved_contradictions = Vec::new();
        let mut filled_gaps = Vec::new();

        // Compute initial confidence.
        let snapshot = engine.snapshot();
        state.confidence = if snapshot.compositions.is_empty() {
            0.0
        } else {
            snapshot
                .compositions
                .iter()
                .map(|c| c.confidence)
                .sum::<f32>()
                / snapshot.compositions.len() as f32
        };

        for pass in 0..self.budget.max_enrichment_rounds {
            if stop_condition.should_stop(&state) {
                break;
            }

            state.evidence_at_loop_start = state.evidence_count;

            // ── Step 1: Detect gaps actively ──
            let snapshot = engine.snapshot();
            let gaps = {
                let mut dg = DetectGaps::new();
                dg.detect_all(&snapshot)
            };

            if gaps.is_empty() {
                // No gaps → nothing to enrich → stop.
                break;
            }

            // ── Step 2: Select strategies for each gap ──
            let mut sa = SelectAcquisition::new();
            let mut enrichments_queued = 0;

            for gap in &gaps {
                let decision = sa.select_strategy(gap, engine.graph());
                match &decision.strategy {
                    AcquisitionStrategy::PassiveRecall {
                        candidate_node_id,
                        candidate_label,
                        confidence,
                    } => {
                        // Queue enrichment for gap with known candidate.
                        if let Some(comp_id) = &gap.source_composition_id {
                            engine.context.pending_enrichments.push(EnrichmentRequest {
                                target_composition_id: comp_id.clone(),
                                role_to_fill: gap
                                    .missing_role
                                    .clone()
                                    .unwrap_or(SemanticRole::Arg0Agent),
                                candidate_node_id: *candidate_node_id,
                                candidate_label: candidate_label.clone(),
                                confidence: *confidence,
                                source: EnrichmentSource::PassiveRecall,
                            });
                            enrichments_queued += 1;
                            filled_gaps.push(CompositionId::new(gap.gap_id.clone()));
                        }
                    }
                    AcquisitionStrategy::AskUser { .. }
                    | AcquisitionStrategy::ReExtraction { .. }
                    | AcquisitionStrategy::Defer => {
                        // These require external input or are deferred — skip for now.
                    }
                }
            }

            if enrichments_queued == 0 {
                // No actionable enrichments → increment stagnation counter.
                state.loops_without_new_evidence += 1;
            }

            // ── Step 3: Apply enrichments via EnrichComposition ──
            // Track enrichment targets before they are drained.
            let enrichment_targets: Vec<_> = engine
                .context
                .pending_enrichments
                .iter()
                .map(|r| r.target_composition_id.clone())
                .collect();

            let result = {
                let enrich = EnrichComposition::new();
                let (ctx, graph) = engine.context_and_graph_mut();
                enrich.execute(ctx, graph)
            };

            // Track which compositions were enriched.
            for comp_id in &enrichment_targets {
                if !modified_compositions.contains(comp_id) {
                    modified_compositions.push(comp_id.clone());
                }
            }

            // Track evidence from enrichments that were actually applied.
            if result.enrichments_applied > 0 {
                state.evidence_count += result.enrichments_applied;
            }

            // ── Step 4: Re-govern affected compositions ──
            {
                let gb = GovernBeliefs::new();
                // Collect IDs of compositions that were modified.
                let affected_ids: Vec<_> = modified_compositions.clone();
                // Gather all affected compositions into a mutable vec for batch check.
                let mut affected_comps: Vec<Composition> = affected_ids
                    .iter()
                    .filter_map(|id| engine.graph().compositions.get(id).cloned())
                    .collect();
                gb.check_promotions(&mut affected_comps);
                // Write back any promoted compositions.
                for comp in affected_comps {
                    engine
                        .graph_mut()
                        .compositions
                        .insert(comp.id.clone(), comp);
                }
            }

            // ── Step 5: Update state tracking ──
            state.loops_completed = pass + 1;

            // Recompute confidence.
            // Audit v6 fix: Use average_confidence() instead of snapshot2
            // to avoid a second full graph clone per enrichment pass.
            let prev_confidence = state.confidence;
            let new_confidence = engine.average_confidence();
            state.confidence = new_confidence;

            // Check for new evidence.
            if state.evidence_count == state.evidence_at_loop_start {
                state.loops_without_new_evidence += 1;
            } else {
                state.loops_without_new_evidence = 0;
            }

            // Check if confidence converged (only after the first pass).
            let delta = (new_confidence - prev_confidence).abs();
            if state.loops_completed > 1 && delta < stop_condition.min_confidence_delta {
                break;
            }

            // Check if goal is met.
            if state.confidence >= GOAL_MET_CONFIDENCE && !engine.context.has_gaps() {
                state.goal_met = true;
                break;
            }
        }

        ReflectionLoopResult {
            current_confidence: state.confidence,
            elapsed_ms: 0, // Caller should time this
            evidence_count: state.evidence_count,
            modified_compositions,
            has_gaps: engine.context.has_gaps(),
            resolved_contradictions,
            filled_gaps,
        }
    }

    /// Execute actions proposed by reflection findings.
    fn execute_reflection_actions(&self, findings: &[ReflectionFinding], engine: &mut PipelineEngine) {
        for finding in findings {
            match &finding.action {
                ReflectionAction::ProposePromotion(comp_id) => {
                    // Try to promote the composition.
                    if let Some(comp) = engine.graph_mut().compositions.get_mut(comp_id) {
                        if comp.lifecycle == LifecycleState::Candidate && comp.batch_seen >= REFLECTION_PROMOTION_MIN_AGE && comp.confidence >= REFLECTION_PROMOTION_MIN_CONFIDENCE {
                            comp.lifecycle = LifecycleState::Stable;
                        }
                    }
                }
                ReflectionAction::ProposeDeprecation(comp_id) => {
                    if let Some(comp) = engine.graph_mut().compositions.get_mut(comp_id) {
                        comp.lifecycle = LifecycleState::Deprecated;
                    }
                }
                ReflectionAction::ProposeContradictionResolution(a_id, b_id) => {
                    // Mark both as Inferred (voice confusion resolution).
                    for comp_id in [a_id, b_id] {
                        if let Some(comp) = engine.graph_mut().compositions.get_mut(comp_id) {
                            if comp.epistemic == EpistemicState::Contradicted {
                                comp.epistemic = EpistemicState::Inferred;
                                comp.contradiction = None;
                            }
                        }
                    }
                }
                ReflectionAction::ProposeMerge(a_id, b_id) => {
                    // For now, deprecate the lower-confidence one.
                    let (_keep, remove) = {
                        let graph = engine.graph();
                        let comp_a = graph.get_composition(a_id);
                        let comp_b = graph.get_composition(b_id);
                        match (comp_a, comp_b) {
                            (Some(a), Some(b)) if a.confidence >= b.confidence => (a_id.clone(), b_id.clone()),
                            (Some(_), Some(_)) => (b_id.clone(), a_id.clone()),
                            _ => continue,
                        }
                    };
                    if let Some(comp) = engine.graph_mut().compositions.get_mut(&remove) {
                        comp.lifecycle = LifecycleState::Deprecated;
                    }
                }
                ReflectionAction::NoAction => {}
            }
        }
    }

    /// Top-level ingest with mode-specific behavior.
    ///
    /// 1. Select cognitive mode from graph neighborhood
    /// 2. Run the standard pipeline
    /// 3. If Analytical or Reflective, run the enrichment loop
    /// 4. If Reflective, produce reflection findings
    pub fn ingest(&mut self, text: &str, engine: &mut PipelineEngine) -> IngestResult {
        // Step 1: Select cognitive mode using extract_keywords + neighborhood_for.
        let snapshot = engine.snapshot();
        let mode = self.select_cognitive_mode(text, &snapshot.compositions);

        // Step 2: Run standard pipeline.
        let mut result = engine.ingest(text).unwrap_or_else(|e| {
            eprintln!("[ExecutiveOrchestrator] Pipeline error: {}", e);
            IngestResult::new()
        });

        // Step 3: If Analytical or Reflective, run enrichment loop.
        if mode != CognitiveMode::Reactive && self.budget.max_enrichment_rounds > 0 {
            let loop_result = self.run_enrichment_loop(engine);

            // Merge enrichment results.
            result.enrichments_applied += loop_result.evidence_count;

            // Step 4: If Reflective, produce reflection findings.
            if mode == CognitiveMode::Reflective {
                let findings = self.reflect.reflect(&loop_result, engine.graph());
                // Audit v5 fix (DD3): Previously, findings were discarded with
                // `let _ = findings`. Now we store the count and execute actions.
                engine.context.last_reflection_findings_count = findings.len();
                // Execute reflection actions.
                self.execute_reflection_actions(&findings, engine);
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
        assert_eq!(reactive.max_enrichment_rounds, 0);
        assert_eq!(reactive.max_reasoning_depth, 1);

        let analytical = ComputeBudget::for_mode(&CognitiveMode::Analytical);
        assert_eq!(analytical.max_enrichment_rounds, 1);
        assert_eq!(analytical.max_reasoning_depth, 3);

        let reflective = ComputeBudget::for_mode(&CognitiveMode::Reflective);
        assert_eq!(reflective.max_enrichment_rounds, 2);
        assert_eq!(reflective.max_reasoning_depth, 5);
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
            modified_compositions: Vec::new(),
            evidence_count: 0,
            evidence_at_loop_start: 0,
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
            modified_compositions: Vec::new(),
            evidence_count: 0,
            evidence_at_loop_start: 0,
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
            modified_compositions: Vec::new(),
            evidence_count: 0,
            evidence_at_loop_start: 0,
        };
        assert!(condition.should_stop(&state));
    }

    #[test]
    fn test_mode_selection_reactive() {
        let mut orchestrator = ExecutiveOrchestrator::new();
        let compositions = vec![Composition {
            confidence: 0.8,
            epistemic: EpistemicState::Observed,
            ..Composition::default()
        }];

        let mode = orchestrator.select_cognitive_mode("test input", &compositions);
        assert_eq!(mode, CognitiveMode::Reactive);
    }

    #[test]
    fn test_mode_selection_analytical() {
        let mut orchestrator = ExecutiveOrchestrator::new();
        let compositions = vec![Composition {
            confidence: 0.3,
            epistemic: EpistemicState::Observed,
            ..Composition::default()
        }];

        let mode = orchestrator.select_cognitive_mode("test input", &compositions);
        assert_eq!(mode, CognitiveMode::Analytical);
    }

    #[test]
    fn test_mode_selection_reflective() {
        let mut orchestrator = ExecutiveOrchestrator::new();
        let compositions = vec![
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
        ];

        let mode = orchestrator.select_cognitive_mode("test input", &compositions);
        assert_eq!(mode, CognitiveMode::Reflective);
    }

    #[test]
    fn test_reflection_finding_types() {
        let reflect = Reflect::new();
        let result = ReflectionLoopResult {
            current_confidence: 0.9,
            elapsed_ms: 0,
            evidence_count: 2,
            modified_compositions: vec!["comp1".into()],
            has_gaps: false,
            resolved_contradictions: vec!["comp_contra".into()],
            filled_gaps: vec!["comp_gap".into()],
        };
        let graph = Graph::new();
        let findings = reflect.reflect(&result, &graph);

        assert!(findings
            .iter()
            .any(|f| f.finding_type == ReflectionFindingType::ContradictionResolvable));
        assert!(findings
            .iter()
            .any(|f| f.finding_type == ReflectionFindingType::PromotionCandidate));
    }
}
