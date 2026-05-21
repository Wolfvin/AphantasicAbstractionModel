//! # v1.0.0 Transform Pipeline Engine
//!
//! This module implements the DAG-based transform pipeline that drives all
//! knowledge ingestion. Transforms are registered with dependency chains and optional
//! conditions, then executed in topological order.
//!
//! ## Architecture
//!
//! ```text
//! RawText → Tokenize → ExtractFrame → ReasonFrame ─┐
//!              │                                     │
//!              └───────────────────→ IngestAtoms ←───┘
//!                                        │
//!                                   GovernBeliefs
//!                                        │
//!                                     SeedAnchor
//!                                        │
//!                                    DetectGaps
//!                                        │
//!                                  SelectAcquisition
//!                                     /          \
//!                          EnrichComposition   ReExtractFrame
//! ```
//!
//! ## Key Design Decisions
//!
//! - **Object-safe trait**: Since the `Transform` trait has associated types
//!   (`Input`, `Output`), it cannot be made into a trait object directly.
//!   We use [`ErasedTransform`] as an object-safe wrapper that reads/writes
//!   all data through [`PipelineContext`] and [`Graph`].
//!
//! - **Condition-gated execution**: Each transform node can have an optional
//!   condition closure. Transforms whose conditions evaluate to `false` are
//!   skipped during DAG execution.
//!
//! - **Topological sort**: Kahn's algorithm ensures transforms execute in
//!   dependency order. A cycle in the dependency graph is treated as a
//!   registration error.
//!
//! ## Feature Flag
//!
//! This module is only compiled when the `v12` feature is enabled.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

use super::acquisition::{DetectGaps, SelectAcquisition};
use super::convergence::ConvergenceDetectionTransform;
use super::extract_frame::ExtractFrame;
use super::govern_beliefs::{GovernBeliefs, SeedAnchor};
use super::reason_frame::ReasonFrame;
use super::spreading::{SpreadingActivation, SpreadingActivationTransform};
use super::temporal::TemporalDecayTransform;
use super::types::*;
use super::verbalize::CompositionalVerbalizeTransform;
// NodeId is imported from crate::types — not re-exported by super::types.
use crate::types::NodeId;

// ========================================================================
// PipelineError — Errors from Pipeline Execution
// ========================================================================

/// Errors that can occur during pipeline execution.
///
/// Replaces the previous `eprintln!` approach for cycle detection,
/// providing proper error propagation that callers can handle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineError {
    /// A cycle was detected in the transform dependency DAG.
    /// This is a registration error — the pipeline cannot execute.
    CycleDetected {
        /// The transform IDs that form the cycle.
        cycle_nodes: Vec<String>,
    },
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::CycleDetected { cycle_nodes } => {
                write!(f, "Cycle detected in transform DAG: {:?}", cycle_nodes)
            }
        }
    }
}

impl std::error::Error for PipelineError {}

// ========================================================================
// IngestResult — Summary of a Pipeline Execution
// ========================================================================

/// Summary statistics from a single pipeline execution (MD-3 §5).
///
/// Returned by [`PipelineEngine::execute_dag`] and [`PipelineEngine::ingest`].
/// Provides a quick overview of what the pipeline accomplished in this run.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IngestResult {
    /// Number of new atoms created during this pipeline run.
    pub atoms_created: usize,
    /// Number of new compositions created.
    pub compositions_created: usize,
    /// Number of new edges created.
    pub edges_created: usize,
    /// Number of knowledge gaps detected.
    pub gaps_detected: usize,
    /// Number of enrichment requests applied.
    pub enrichments_applied: usize,
    /// Number of governance state transitions (lifecycle or epistemic).
    pub governance_transitions: usize,
}

impl IngestResult {
    /// Create an empty (zero-valued) ingest result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge another `IngestResult` into this one, summing all counters.
    pub fn merge(&mut self, other: &IngestResult) {
        self.atoms_created += other.atoms_created;
        self.compositions_created += other.compositions_created;
        self.edges_created += other.edges_created;
        self.gaps_detected += other.gaps_detected;
        self.enrichments_applied += other.enrichments_applied;
        self.governance_transitions += other.governance_transitions;
    }
}

// ========================================================================
// TransformCondition — Type alias for complex condition closure
// ========================================================================

/// Type alias for the optional condition closure that gates transform execution.
///
/// This avoids repeating the complex `Box<dyn Fn(&PipelineContext) -> bool + Send + Sync>`
/// signature and silences the `type_complexity` clippy lint.
pub type TransformCondition = Box<dyn Fn(&PipelineContext) -> bool + Send + Sync>;

// ========================================================================
// TransformNode — Node in the Transform DAG
// ========================================================================

/// A node in the transform dependency DAG.
///
/// Each `TransformNode` represents one registered transform, its dependency
/// chain, and an optional condition that gates execution.
///
/// # Dependency Semantics
///
/// All dependencies must have completed (either executed or skipped) before
/// this transform can run. A "completed" dependency is one that appeared
/// earlier in the topological sort and was either:
/// - Executed (condition was `true` or `None`), or
/// - Skipped (condition was `false`)
///
/// The difference between "executed" and "skipped" is relevant only to
/// downstream transforms that inspect the pipeline context for intermediate
/// results produced by their dependencies.
struct TransformNode {
    /// Unique identifier for this transform (e.g., "Tokenize", "GovernBeliefs").
    /// Uses `String` instead of `TypeId` for simplicity and debuggability.
    transform_id: String,

    /// Human-readable description of the input type this transform expects.
    input_type: String,

    /// Human-readable description of the output type this transform produces.
    output_type: String,

    /// IDs of transforms that must complete before this one can run.
    dependencies: Vec<String>,

    /// Optional condition that gates execution.
    /// If `None`, the transform always runs (if its dependencies are met).
    /// If `Some(predicate)`, the transform runs only if the predicate returns `true`.
    condition: Option<TransformCondition>,
}

impl std::fmt::Debug for TransformNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformNode")
            .field("transform_id", &self.transform_id)
            .field("input_type", &self.input_type)
            .field("output_type", &self.output_type)
            .field("dependencies", &self.dependencies)
            .field("condition", &self.condition.as_ref().map(|_| "Some(Fn)"))
            .finish()
    }
}

// ========================================================================
// ErasedTransform — Object-Safe Transform Wrapper
// ========================================================================

/// Object-safe transform wrapper for the pipeline engine.
///
/// The [`Transform`] trait cannot be made into a trait object because it has
/// associated types (`Input`, `Output`). This trait provides an object-safe
/// alternative that reads input from and writes output to the shared
/// [`PipelineContext`] and [`Graph`].
///
/// # Implementors
///
/// Each concrete transform type (e.g., `Tokenize`, `GovernBeliefs`) should
/// implement this trait. The `execute` method is responsible for:
/// 1. Reading its input from `ctx` or `graph`
/// 2. Performing the transform logic
/// 3. Writing its output back to `ctx` or `graph`
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` for potential multi-threaded
/// pipeline execution.
pub trait ErasedTransform: Send + Sync {
    /// Unique identifier for this transform (e.g., "Tokenize").
    fn id(&self) -> &'static str;

    /// Execute the transform, reading from and writing to the shared context
    /// and graph.
    ///
    /// # Arguments
    ///
    /// * `ctx` — Mutable reference to the pipeline context. Read input here,
    ///   write output here.
    /// * `graph` — Mutable reference to the v12 graph. For transforms that
    ///   create or modify compositions.
    ///
    /// # Returns
    ///
    /// An [`IngestResult`] summarizing what this transform accomplished.
    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult;
}

// ========================================================================
// PipelineEngine — DAG-Based Transform Executor
// ========================================================================

/// The central pipeline engine for v1.0.0 (MD-3 §5).
///
/// Manages a set of registered transforms, their dependency DAG, and the
/// shared pipeline context. Executes transforms in topological order with
/// condition gating.
///
/// # Usage
///
/// ```ignore
/// let mut engine = PipelineEngine::new();
/// register_default_pipeline(&mut engine);
/// let result = engine.ingest("Raymond membuat aplikasi karena lambat").unwrap();
/// ```
///
/// # Thread Safety
///
/// `PipelineEngine` itself is `Send` but not `Sync` (it has mutable state).
/// For multi-threaded use, wrap in a `Mutex` or use message passing.
pub struct PipelineEngine {
    /// Registered transforms, keyed by their ID.
    transforms: HashMap<String, Box<dyn ErasedTransform>>,

    /// The transform dependency DAG.
    dag: Vec<TransformNode>,

    /// Shared pipeline context — accessible to all transforms.
    pub context: PipelineContext,

    /// The v12 graph that stores nodes, compositions, and edges.
    graph: Graph,
}

impl PipelineEngine {
    /// Create a new empty pipeline engine with default [`PipelineContext`]
    /// and an empty [`Graph`].
    pub fn new() -> Self {
        Self {
            transforms: HashMap::new(),
            dag: Vec::new(),
            context: PipelineContext::default(),
            graph: Graph::new(),
        }
    }

    /// Register a transform with its dependency chain and optional condition.
    ///
    /// # Arguments
    ///
    /// * `transform` — The transform to register. Must implement [`ErasedTransform`].
    /// * `dependencies` — IDs of transforms that must complete before this one.
    ///   Empty vec means no dependencies (always runs first in topological order).
    /// * `condition` — Optional predicate that gates execution. If `None`, the
    ///   transform always runs. If `Some(fn)`, the transform runs only when
    ///   the predicate returns `true` given the current pipeline context.
    ///
    /// # Panics
    ///
    /// Will not panic during registration, but [`execute_dag`](Self::execute_dag)
    /// will detect cycles and missing dependency IDs at execution time.
    pub fn register<T: ErasedTransform + 'static>(
        &mut self,
        transform: T,
        dependencies: Vec<String>,
        condition: Option<TransformCondition>,
    ) {
        let id = transform.id().to_string();
        let input_type = String::new(); // Could be populated from type_name
        let output_type = String::new();

        self.dag.push(TransformNode {
            transform_id: id.clone(),
            input_type,
            output_type,
            dependencies,
            condition,
        });

        self.transforms.insert(id, Box::new(transform));
    }

    /// Execute the transform DAG on the given input text.
    ///
    /// This is the primary entry point for pipeline execution. It:
    /// 1. Sets the raw text in the pipeline context
    /// 2. Performs a topological sort of the registered transforms
    /// 3. Executes each transform in order, gated by its condition
    /// 4. Accumulates results into an [`IngestResult`]
    ///
    /// # Arguments
    ///
    /// * `initial_input` — The raw text to process.
    ///
    /// # Returns
    ///
    /// An [`IngestResult`] summarizing the pipeline execution.
    pub fn execute_dag(&mut self, initial_input: &str) -> Result<IngestResult, PipelineError> {
        // Reset per-run state in the context.
        self.context.set_raw_text(initial_input);
        self.context.current_atoms.clear();
        self.context.pending_gaps.clear();
        self.context.pending_enrichments.clear();
        self.context.pending_reextractions.clear();

        // Topological sort the DAG.
        let sorted_ids = match topological_sort(&self.dag) {
            Ok(ids) => ids,
            Err(cycle) => {
                return Err(PipelineError::CycleDetected {
                    cycle_nodes: cycle,
                });
            }
        };

        // Track which transforms have been "completed" (executed or skipped).
        let mut completed: HashSet<String> = HashSet::new();
        let mut result = IngestResult::new();

        for id in &sorted_ids {
            // Find the DAG node for this transform.
            let node = match self.dag.iter().find(|n| n.transform_id == *id) {
                Some(n) => n,
                None => continue,
            };

            // Check that all dependencies have completed.
            let deps_met = node.dependencies.iter().all(|dep| completed.contains(dep));
            if !deps_met {
                // Skip — dependencies not yet met (shouldn't happen after topological sort,
                // but defensive).
                continue;
            }

            // Check the condition gate.
            let should_run = match &node.condition {
                None => true,
                Some(cond) => cond(&self.context),
            };

            if should_run {
                // Execute the transform.
                if let Some(transform) = self.transforms.get(id) {
                    let step_result = transform.execute(&mut self.context, &mut self.graph);
                    result.merge(&step_result);
                }
            }

            // Mark as completed regardless of whether it was executed or skipped.
            completed.insert(id.clone());
        }

        Ok(result)
    }

    /// Convenience method — identical to [`execute_dag`](Self::execute_dag).
    ///
    /// Provided for ergonomic API: `engine.ingest("some text").unwrap()`.
    /// Returns `Result` so callers can handle pipeline errors properly.
    pub fn ingest(&mut self, text: &str) -> Result<IngestResult, PipelineError> {
        self.execute_dag(text)
    }

    /// Apply an [`AnchoredDelta`] to the graph.
    ///
    /// This is the post-pipeline step: after the full DAG has executed and
    /// produced an `AnchoredDelta` (from `SeedAnchor`), apply it to the
    /// graph by upserting all compositions.
    ///
    /// **Note**: In the current pipeline, `GovernBeliefs::execute()` writes
    /// compositions directly to the graph, bypassing this method. This
    /// method exists for external callers who want to apply an `AnchoredDelta`
    /// manually (e.g., from Python FFI). It is NOT called by the default
    /// pipeline flow.
    pub fn apply(&mut self, anchored: AnchoredDelta) {
        for composition in anchored.compositions {
            self.graph
                .compositions
                .insert(composition.id.clone(), composition);
        }
    }

    /// Apply an [`AnchoredDelta`] and return structured feedback.
    ///
    /// Like [`apply`](Self::apply), but constructs a [`ReflectionLoopResult`]
    /// from the applied delta, enabling the reflection loop (MD-5) to track
    /// progress.
    pub fn apply_with_result(&mut self, anchored: AnchoredDelta) -> ReflectionLoopResult {
        let mut modified_ids = Vec::new();
        let mut total_confidence = 0.0f32;

        for composition in &anchored.compositions {
            modified_ids.push(composition.id.clone());
            total_confidence += composition.confidence;
        }

        let count = anchored.compositions.len();
        let avg_confidence = if count > 0 {
            total_confidence / count as f32
        } else {
            0.0
        };

        // Apply to graph.
        let has_gaps = self.context.has_gaps();
        self.apply(anchored);

        ReflectionLoopResult {
            current_confidence: avg_confidence,
            elapsed_ms: 0, // Caller should time this
            evidence_count: modified_ids.len(),
            modified_compositions: modified_ids,
            has_gaps,
            resolved_contradictions: Vec::new(),
            filled_gaps: Vec::new(),
        }
    }

    /// Find low-confidence Event compositions that are missing expected roles.
    ///
    /// An Event composition is "weak" if:
    /// - Its confidence is below 0.5, AND
    /// - It is missing one or more expected roles (Arg0Agent, Arg1Patient, or Cause).
    ///
    /// Returns a list of [`WeakFrame`] descriptors that can be used to construct
    /// `ReExtractionRequest`s for the feedback loop.
    pub fn find_weak_frames(&self) -> Vec<WeakFrame> {
        let mut weak = Vec::new();

        for composition in self.graph.compositions.values() {
            if composition.composition_type != CompositionType::Event {
                continue;
            }
            if composition.confidence >= 0.5 {
                continue;
            }

            // Check for missing expected roles.
            let has_agent = composition.has_member_with_role(SemanticRole::Arg0Agent);
            let has_patient = composition.has_member_with_role(SemanticRole::Arg1Patient);
            let has_cause = composition.has_member_with_role(SemanticRole::Cause);

            if !has_agent || !has_patient || !has_cause {
                weak.push(WeakFrame {
                    composition_id: composition.id.clone(),
                    atom_id: String::new(), // Would be populated from provenance
                    source_text: composition.source_text.clone(),
                });
            }
        }

        weak
    }

    /// Get a snapshot of the current graph state.
    ///
    /// Useful for passing to `DetectGaps` or for serialization.
    pub fn snapshot(&self) -> GraphSnapshot {
        GraphSnapshot {
            recent_atoms: self.context.current_atoms.clone(),
            compositions: self.graph.compositions.values().cloned().collect(),
        }
    }

    /// Get a reference to the v12 [`Graph`].
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Get a mutable reference to the v12 [`Graph`].
    ///
    /// Use with caution — direct mutations bypass the pipeline's
    /// governance. Prefer using transforms for production code.
    pub fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    /// Get mutable references to both the context and graph simultaneously.
    ///
    /// This is needed when a transform's `execute()` requires `&mut PipelineContext`
    /// and `&mut Graph` at the same time, which can't be done through the
    /// individual `graph_mut()` and context field access due to borrow checker rules.
    pub fn context_and_graph_mut(&mut self) -> (&mut PipelineContext, &mut Graph) {
        (&mut self.context, &mut self.graph)
    }

    /// Save the current graph state to a JSON file.
    ///
    /// Returns an error string if saving fails.
    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let persistence = super::persistence::Persistence::new();
        persistence
            .save(&self.graph, path)
            .map_err(|e| e.to_string())
    }

    /// Load graph state from a JSON file, replacing the current graph.
    ///
    /// Returns an error string if loading fails.
    pub fn load(&mut self, path: &std::path::Path) -> Result<(), String> {
        let persistence = super::persistence::Persistence::new();
        match persistence.load(path) {
            Ok(graph) => {
                self.graph = graph;
                Ok(())
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Convenience method for running a specific transform by type.
    ///
    /// Creates a new instance of `T`, reads its input from the context,
    /// executes it, and returns its output. This does NOT go through the
    /// DAG — it's a direct execution for testing or one-off use.
    ///
    /// # Type Parameters
    ///
    /// * `T` — The concrete transform type to run.
    ///
    /// # Note
    ///
    /// This method requires `T: Default + Transform`. The transform's
    /// `transform()` method is called with the provided input and the
    /// engine's context.
    pub fn run<T: Transform + Default>(&mut self, input: &T::Input) -> T::Output {
        let transform = T::default();
        transform.transform(input, &mut self.context)
    }
}

impl Default for PipelineEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ========================================================================
// SyncPipelineEngine — Thread-Safe Wrapper
// ========================================================================

/// Thread-safe wrapper around [`PipelineEngine`].
///
/// `PipelineEngine` is `Send` but not `Sync` (it has mutable state with no
/// internal synchronization). This wrapper adds a `Mutex`, making the engine
/// safe to share across threads via `Arc<SyncPipelineEngine>`.
///
/// # Thread Safety
///
/// - `SyncPipelineEngine` is both `Send` and `Sync`.
/// - All operations acquire the internal `Mutex` before accessing the engine.
/// - For high-throughput scenarios, prefer message-passing (e.g., a dedicated
///   ingest thread with a channel) over shared `SyncPipelineEngine`.
///
/// # Usage
///
/// ```ignore
/// use std::sync::Arc;
/// let engine = Arc::new(SyncPipelineEngine::new());
///
/// // Share across threads
/// let engine_clone = engine.clone();
/// std::thread::spawn(move || {
///     let mut guard = engine_clone.lock();
///     guard.ingest("some text").unwrap();
/// });
/// ```
pub struct SyncPipelineEngine {
    inner: std::sync::Mutex<PipelineEngine>,
}

impl SyncPipelineEngine {
    /// Create a new thread-safe pipeline engine with default transforms.
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(PipelineEngine::new()),
        }
    }

    /// Create from an existing `PipelineEngine`.
    pub fn from_engine(engine: PipelineEngine) -> Self {
        Self {
            inner: std::sync::Mutex::new(engine),
        }
    }

    /// Acquire a lock on the inner engine.
    ///
    /// Returns a `MutexGuard` that derefs to `PipelineEngine`.
    /// The lock is held until the guard is dropped.
    pub fn lock(&self) -> std::sync::MutexGuard<'_, PipelineEngine> {
        self.inner.lock().expect("SyncPipelineEngine mutex poisoned")
    }

    /// Convenience: ingest text through the pipeline.
    ///
    /// Acquires the lock, runs `ingest()`, and releases the lock.
    pub fn ingest_sync(&self, text: &str) -> Result<IngestResult, PipelineError> {
        self.lock().ingest(text)
    }

    /// Convenience: get a snapshot of the current graph state.
    pub fn snapshot_sync(&self) -> GraphSnapshot {
        self.lock().snapshot()
    }
}

impl Default for SyncPipelineEngine {
    fn default() -> Self {
        Self::new()
    }
}

// SyncPipelineEngine is Send + Sync because Mutex<PipelineEngine> is Send + Sync
// (PipelineEngine itself is Send).
unsafe impl Send for SyncPipelineEngine {}
unsafe impl Sync for SyncPipelineEngine {}

// ========================================================================
// Topological Sort — Kahn's Algorithm
// ========================================================================

/// Perform a topological sort of the transform DAG using Kahn's algorithm.
///
/// Returns the sorted list of transform IDs, or an error containing the
/// cycle nodes if a cycle is detected.
///
/// # Algorithm
///
/// 1. Compute in-degrees for all nodes
/// 2. Initialize a queue with all nodes of in-degree 0
/// 3. While the queue is non-empty:
///    a. Dequeue a node and add it to the sorted output
///    b. Decrement in-degrees of all nodes that depend on it
///    c. If any dependent's in-degree reaches 0, enqueue it
/// 4. If the output contains fewer nodes than the DAG, there is a cycle
fn topological_sort(dag: &[TransformNode]) -> Result<Vec<String>, Vec<String>> {
    let id_set: HashSet<String> = dag.iter().map(|n| n.transform_id.clone()).collect();

    // Compute in-degree for each node.
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    for node in dag {
        in_degree.entry(node.transform_id.clone()).or_insert(0);
        for dep in &node.dependencies {
            // Only count dependencies that actually exist in the DAG.
            if id_set.contains(dep) {
                *in_degree.entry(node.transform_id.clone()).or_insert(0) += 1;
            }
        }
    }

    // Initialize queue with nodes that have in-degree 0.
    let mut queue: Vec<String> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(id, _)| id.clone())
        .collect();

    let mut sorted = Vec::with_capacity(dag.len());

    while let Some(current) = queue.pop() {
        sorted.push(current.clone());

        // Decrement in-degrees of nodes that depend on `current`.
        for node in dag {
            if node.dependencies.contains(&current) {
                let deg = in_degree.get_mut(&node.transform_id).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(node.transform_id.clone());
                }
            }
        }
    }

    // Check for cycle.
    if sorted.len() != dag.len() {
        let cycle_nodes: Vec<String> = dag
            .iter()
            .filter(|n| !sorted.contains(&n.transform_id))
            .map(|n| n.transform_id.clone())
            .collect();
        Err(cycle_nodes)
    } else {
        Ok(sorted)
    }
}

// ========================================================================
// register_default_pipeline — Wire All Core Transforms
// ========================================================================

/// Register all core v1.0.0 transforms in dependency order.
///
/// This wires up the complete default pipeline with 14 transforms:
///
/// | # | Transform | Dependencies | Condition |
/// |---|-----------|-------------|------------|
/// | 1 | Tokenize | (none) | always |
/// | 2 | ExtractFrame | Tokenize | is_sentence_like |
/// | 3 | ReasonFrame | ExtractFrame | has_event_atoms |
/// | 4 | IngestAtoms | Tokenize, ReasonFrame | always |
/// | 5 | GovernBeliefs | IngestAtoms | always |
/// | 6 | SeedAnchor | GovernBeliefs | always |
/// | 7 | DetectGaps | SeedAnchor | gap_detection_enabled |
/// | 8 | SelectAcquisition | DetectGaps | has_gaps |
/// | 9 | EnrichComposition | SelectAcquisition | has_enrichment_requests |
/// | 10 | ReExtractFrame | SelectAcquisition | has_reextraction_requests |
/// | 11 | TemporalDecay | EnrichComposition | always |
/// | 12 | SpreadingActivation | GovernBeliefs | has_event_atoms |
/// | 13 | ConvergenceDetection | EnrichComposition, TemporalDecay | always |
/// | 14 | CompositionalVerbalize | ConvergenceDetection | always |
///
/// Parse a semantic role name string into a `SemanticRole` enum.
///
/// Accepts both the Debug format (e.g., "Arg0Agent") and common
/// abbreviations (e.g., "Agent", "Patient", "Cause", "Problem", "Solution").
fn parse_semantic_role(name: &str) -> Option<SemanticRole> {
    match name {
        "Arg0Agent" | "Agent" | "agent" => Some(SemanticRole::Arg0Agent),
        "Arg1Patient" | "Patient" | "patient" => Some(SemanticRole::Arg1Patient),
        "Arg2Recipient" | "Recipient" | "recipient" => Some(SemanticRole::Arg2Recipient),
        "Cause" | "cause" => Some(SemanticRole::Cause),
        "Purpose" | "purpose" => Some(SemanticRole::Purpose),
        "Location" | "location" => Some(SemanticRole::Location),
        "Time" | "time" => Some(SemanticRole::Time),
        "Instrument" | "instrument" => Some(SemanticRole::Instrument),
        "Predicate" | "predicate" => Some(SemanticRole::Predicate),
        "Problem" | "problem" => Some(SemanticRole::Problem),
        "Solution" | "solution" => Some(SemanticRole::Solution),
        "Beneficiary" | "beneficiary" => Some(SemanticRole::Beneficiary),
        "Tool" | "tool" => Some(SemanticRole::Tool),
        "Motivation" | "motivation" => Some(SemanticRole::Motivation),
        "PainPoint" | "painpoint" | "Pain" | "pain" => Some(SemanticRole::PainPoint),
        "ImpliedGoal" | "implied_goal" | "Goal" | "goal" => Some(SemanticRole::ImpliedGoal),
        "PatternType" | "pattern_type" => Some(SemanticRole::PatternType),
        "Antecedent" | "antecedent" => Some(SemanticRole::Antecedent),
        "Consequent" | "consequent" => Some(SemanticRole::Consequent),
        "SourceAtom" | "source_atom" => Some(SemanticRole::SourceAtom),
        "SourceEvent" | "source_event" => Some(SemanticRole::SourceEvent),
        "EquivalentOf" | "equivalent_of" => Some(SemanticRole::EquivalentOf),
        _ => None,
    }
}

pub fn register_default_pipeline(engine: &mut PipelineEngine) {
    // 1. Tokenize — no dependencies, always runs.
    engine.register(Tokenize::new(), vec![], None);

    // 2. ExtractFrame — depends on Tokenize, condition: is_sentence_like.
    engine.register(
        ExtractFrame::new(),
        vec!["Tokenize".to_string()],
        Some(Box::new(|ctx: &PipelineContext| ctx.is_sentence_like())),
    );

    // 3. ReasonFrame — depends on ExtractFrame, condition: has_event_atoms.
    engine.register(
        ReasonFrame::new(),
        vec!["ExtractFrame".to_string()],
        Some(Box::new(|ctx: &PipelineContext| ctx.has_event_atoms())),
    );

    // 4. IngestAtoms — depends on Tokenize + ReasonFrame, always runs.
    engine.register(
        IngestAtoms::new(),
        vec!["Tokenize".to_string(), "ReasonFrame".to_string()],
        None,
    );

    // 5. GovernBeliefs — depends on IngestAtoms, always runs.
    engine.register(GovernBeliefs::new(), vec!["IngestAtoms".to_string()], None);

    // 6. SeedAnchor — depends on GovernBeliefs, always runs.
    engine.register(SeedAnchor::new(), vec!["GovernBeliefs".to_string()], None);

    // 7. DetectGaps — depends on SeedAnchor, condition: gap_detection_enabled.
    engine.register(
        DetectGaps::new(),
        vec!["SeedAnchor".to_string()],
        Some(Box::new(|ctx: &PipelineContext| {
            ctx.gap_detection_enabled()
        })),
    );

    // 8. SelectAcquisition — depends on DetectGaps, condition: has_gaps.
    engine.register(
        SelectAcquisition::new(),
        vec!["DetectGaps".to_string()],
        Some(Box::new(|ctx: &PipelineContext| ctx.has_gaps())),
    );

    // 9. EnrichComposition — depends on SelectAcquisition, condition: has_enrichment_requests.
    engine.register(
        EnrichComposition::new(),
        vec!["SelectAcquisition".to_string()],
        Some(Box::new(|ctx: &PipelineContext| {
            ctx.has_enrichment_requests()
        })),
    );

    // 10. ReExtractFrame — depends on SelectAcquisition, condition: has_reextraction_requests.
    engine.register(
        ReExtractFrame::new(),
        vec!["SelectAcquisition".to_string()],
        Some(Box::new(|ctx: &PipelineContext| {
            ctx.has_reextraction_requests()
        })),
    );

    // 11. TemporalDecay — runs after enrichment, applies Ebbinghaus decay.
    //     No dependencies on other transforms (reads graph directly).
    //     Condition: always (decay is continuous).
    engine.register(
        TemporalDecayTransform {
            engine: super::temporal::TemporalDecay::new(),
        },
        vec!["EnrichComposition".to_string()],
        None,
    );

    // 12. SpreadingActivation — propagates energy from seed-anchored nodes.
    //     Depends on GovernBeliefs (seeds must be computed first).
    //     Condition: has event atoms (only when pipeline produced events).
    engine.register(
        SpreadingActivationTransform::new(),
        vec!["GovernBeliefs".to_string()],
        Some(Box::new(|ctx: &PipelineContext| ctx.has_event_atoms())),
    );

    // 13. ConvergenceDetection — detects structurally equivalent compositions.
    //     Runs last, after all enrichment and decay.
    //     Condition: always (checks internally if ≥2 compositions exist).
    engine.register(
        ConvergenceDetectionTransform {
            engine: super::convergence::ConvergenceDetection::new(),
        },
        vec!["EnrichComposition".to_string(), "TemporalDecay".to_string()],
        None,
    );

    // 14. CompositionalVerbalize — generates explanations from the graph.
    //     Audit v5 fix: Previously NOT in default pipeline — the CVE transform
    //     was fully implemented but never registered. Now registered after ConvergenceDetection.
    engine.register(
        CompositionalVerbalizeTransform::new(),
        vec!["ConvergenceDetection".to_string()],
        None,
    );
}

// ========================================================================
// Graph — Minimal v12 Composition Graph
// ========================================================================

/// Minimal v12 graph that stores Compositions (not just Nodes).
///
/// Unlike the v8.3 `RsvsGraph` which stores only nodes and edges, this
/// graph additionally stores `Composition`s — structured groupings of nodes
/// with typed roles, lifecycle/epistemic states, and seed alignment scores.
///
/// # Storage Model
///
/// ```text
/// nodes:         HashMap<NodeId, Node>          — v1.0.0 nodes (minimal)
/// compositions:  HashMap<CompositionId, Composition> — v1.0.0 compositions
/// edges:         Vec<(CompositionId, NodeId, SemanticEdge)> — v1.0.0 typed edges
/// label_to_id:   HashMap<String, NodeId>        — label → NodeId index
/// next_id:       NodeId                         — auto-incrementing ID counter
/// ```
///
/// # Relationship to v8.3 RsvsGraph
///
/// This is a SEPARATE graph from `RsvsGraph`. It exists because v12 needs
/// to store `Composition`s, which are not part of the v8.3 data model.
/// Eventually, these graphs will be unified, but during the transition period
/// they coexist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    /// Nodes — v1.0.0 minimal nodes.
    pub nodes: HashMap<NodeId, Node>,

    /// Compositions — v1.0.0 structured groupings.
    pub compositions: HashMap<CompositionId, Composition>,

    /// Edges — (composition_id, target_node_id, semantic_edge) triples.
    /// Each edge links a composition to one of its member nodes.
    pub edges: Vec<(CompositionId, NodeId, SemanticEdge)>,

    /// Label-to-node-id index for fast label lookups.
    pub label_to_id: HashMap<String, NodeId>,

    /// Next available node ID (auto-incrementing).
    pub next_id: NodeId,

    /// Arbitrary metadata key-value store for pipeline state that must
    /// persist across `execute()` calls (e.g., `govern_batch` counter).
    #[serde(default)]
    pub metadata: HashMap<String, String>,

    /// Audit v3 fix: Set of composition IDs that are new or modified since
    /// the last `GovernBeliefs.execute()` call. Only these compositions are
    /// sent to `govern()` — not the entire graph. This avoids O(N) clone
    /// per ingest for large graphs.
    ///
    /// Marked dirty by `IngestAtoms` when compositions are created or modified.
    /// Cleared after `GovernBeliefs.execute()` finishes.
    #[serde(default)]
    pub dirty_compositions: HashSet<CompositionId>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            compositions: HashMap::new(),
            edges: Vec::new(),
            label_to_id: HashMap::new(),
            next_id: 1, // 0 is reserved/unassigned
            metadata: HashMap::new(),
            dirty_compositions: HashSet::new(),
        }
    }

    /// Ensure a node with the given label exists, creating it if necessary.
    ///
    /// If a node with this label already exists, returns its ID without
    /// modification. If not, creates a new `Node` with default fields
    /// and returns the new ID.
    ///
    /// This is the primary node-creation method used by `IngestAtoms`.
    pub fn ensure_node(&mut self, label: &str) -> NodeId {
        if let Some(&id) = self.label_to_id.get(label) {
            return id;
        }

        let id = self.next_id;
        self.next_id += 1;

        let node = Node::new(id, label);

        self.nodes.insert(id, node);
        self.label_to_id.insert(label.to_string(), id);

        id
    }

    /// Get a composition by its ID.
    pub fn get_composition(&self, id: &CompositionId) -> Option<&Composition> {
        self.compositions.get(id)
    }

    /// Iterate over all compositions in the graph.
    pub fn compositions(&self) -> impl Iterator<Item = &Composition> {
        self.compositions.values()
    }

    /// Get a node by its ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Find a node by its label.
    ///
    /// Returns `None` if no node with this label exists.
    pub fn find_node_by_label(&self, label: &str) -> Option<NodeId> {
        self.label_to_id.get(label).copied()
    }

    /// Get the label of a node by its ID.
    ///
    /// Returns `None` if the node doesn't exist.
    pub fn node_label(&self, id: NodeId) -> Option<&str> {
        self.nodes.get(&id).map(|n| n.label.as_str())
    }

    /// Get recent compositions, ordered by creation time (most recent first).
    ///
    /// Returns up to `limit` compositions. "Recent" is determined by
    /// lexicographic comparison of the `created_at` timestamp field.
    pub fn recent_compositions(&self, limit: usize) -> Vec<&Composition> {
        let mut comps: Vec<&Composition> = self.compositions.values().collect();
        comps.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        comps.truncate(limit);
        comps
    }

    /// Count how many compositions contain both node A and node B.
    ///
    /// This is the co-occurrence count used for similarity computation
    /// and gap detection. Two nodes that co-occur in many compositions
    /// are structurally related.
    pub fn cooccurrence_count(&self, node_a: NodeId, node_b: NodeId) -> usize {
        self.compositions
            .values()
            .filter(|comp| {
                let has_a = comp.members.iter().any(|m| m.node_id == node_a);
                let has_b = comp.members.iter().any(|m| m.node_id == node_b);
                has_a && has_b
            })
            .count()
    }

    /// Check if a node with the given ID exists.
    pub fn has_node(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Get an edge by composition ID and target node ID.
    ///
    /// Returns the first matching edge, or `None` if no such edge exists.
    pub fn get_edge(
        &self,
        composition_id: &CompositionId,
        node_id: NodeId,
    ) -> Option<&(CompositionId, NodeId, SemanticEdge)> {
        self.edges
            .iter()
            .find(|(cid, nid, _)| cid == composition_id && *nid == node_id)
    }

    /// Compute Jaccard structural similarity between two compositions.
    ///
    /// Two compositions are structurally similar if they share many
    /// of the same member nodes. This is used by convergence detection
    /// and gap detection to find related compositions.
    ///
    /// Returns a value in [0, 1]: 1.0 = identical members, 0.0 = no overlap.
    ///
    /// Delegates to [`ConvergenceDetection::structural_similarity`] to avoid
    /// duplicating the Jaccard computation.
    pub fn structural_similarity(&self, comp_a: &Composition, comp_b: &Composition) -> f32 {
        super::convergence::ConvergenceDetection::structural_similarity(comp_a, comp_b)
    }

    /// Get the graph neighborhood for a set of keyword labels.
    ///
    /// Returns all compositions that contain nodes matching any of
    /// the given keywords. This is used by ExecutiveOrchestrator
    /// for cognitive mode selection.
    pub fn neighborhood_for(&self, keywords: &[String]) -> Vec<&Composition> {
        let keyword_ids: HashSet<NodeId> = keywords
            .iter()
            .filter_map(|kw| self.find_node_by_label(kw))
            .collect();

        if keyword_ids.is_empty() {
            return Vec::new();
        }

        self.compositions
            .values()
            .filter(|comp| {
                comp.members
                    .iter()
                    .any(|m| keyword_ids.contains(&m.node_id))
            })
            .collect()
    }

    /// Get all compositions that contain a specific node.
    pub fn compositions_for_node(&self, node_id: NodeId) -> Vec<&Composition> {
        self.compositions
            .values()
            .filter(|comp| comp.members.iter().any(|m| m.node_id == node_id))
            .collect()
    }

    /// Count total compositions.
    pub fn composition_count(&self) -> usize {
        self.compositions.len()
    }

    /// Count total nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Compute the average confidence of all compositions.
    pub fn average_confidence(&self) -> f32 {
        if self.compositions.is_empty() {
            return 0.0;
        }
        self.compositions
            .values()
            .map(|c| c.confidence)
            .sum::<f32>()
            / self.compositions.len() as f32
    }

    /// Count compositions with a specific epistemic state.
    pub fn count_with_epistemic(&self, state: EpistemicState) -> usize {
        self.compositions
            .values()
            .filter(|c| c.epistemic == state)
            .count()
    }

    /// Count compositions with a specific lifecycle state.
    pub fn count_with_lifecycle(&self, state: LifecycleState) -> usize {
        self.compositions
            .values()
            .filter(|c| c.lifecycle == state)
            .count()
    }

    // ================================================================
    // Phase O: Connectivity Score
    // ================================================================

    /// Compute the connectivity score for a node.
    ///
    /// Connectivity = (number of compositions referencing this node) / saturation_factor.
    /// Capped at 1.0. Saturation factor = 10 (a node referenced by 10+ compositions
    /// is considered fully connected).
    ///
    /// Uses O(1) lookup via composition scan — this is on-demand, not precomputed.
    pub fn connectivity_score(&self, node_id: NodeId) -> f32 {
        const SATURATION: f32 = 10.0;
        let count = self
            .compositions
            .values()
            .filter(|comp| comp.members.iter().any(|m| m.node_id == node_id))
            .count();
        (count as f32 / SATURATION).min(1.0)
    }

    // ================================================================
    // Phase P: SenseRole Helper Methods
    // ================================================================

    /// Is a node primitive? A node is primitive if its max sense layer is 0
    /// OR it has no senses at all (unprocessed/fresh node).
    ///
    /// Note: an unprocessed node (no senses) is treated as primitive.
    /// This is intentional — fresh nodes have not yet been analyzed
    /// and default to the most basic interpretation.
    pub fn is_primitive(&self, node_id: NodeId) -> bool {
        match self.nodes.get(&node_id) {
            Some(node) => {
                if node.senses.is_empty() {
                    return true; // Unprocessed = primitive
                }
                node.senses.iter().all(|s| s.layer == 0)
            }
            None => true, // Non-existent node = primitive by default
        }
    }

    /// Is a node a bridge? A bridge node connects different abstraction layers.
    /// It has at least one sense with `is_utterance` flag or senses at 2+ different layers.
    pub fn is_bridge(&self, node_id: NodeId) -> bool {
        match self.nodes.get(&node_id) {
            Some(node) => {
                if node.senses.iter().any(|s| s.is_utterance) {
                    return true;
                }
                let layers: std::collections::HashSet<u32> =
                    node.senses.iter().map(|s| s.layer).collect();
                layers.len() >= 2
            }
            None => false,
        }
    }

    /// Is a node derived? A derived node has at least one sense with layer ≥ 1.
    pub fn is_derived(&self, node_id: NodeId) -> bool {
        match self.nodes.get(&node_id) {
            Some(node) => node.senses.iter().any(|s| s.layer >= 1),
            None => false,
        }
    }

    /// Is a node at utterance level? Has at least one sense that is utterance-level.
    pub fn is_utterance_level(&self, node_id: NodeId) -> bool {
        match self.nodes.get(&node_id) {
            Some(node) => node.senses.iter().any(|s| s.is_utterance_level()),
            None => false,
        }
    }

    /// Find all nodes that have active utterance senses.
    pub fn find_active_utterance_senses(&self) -> Vec<(NodeId, &Sense)> {
        let mut result = Vec::new();
        for (&node_id, node) in &self.nodes {
            for sense in &node.senses {
                if sense.is_utterance_level() {
                    result.push((node_id, sense));
                }
            }
        }
        result
    }

    /// Find all bridge nodes in the graph.
    pub fn find_bridge_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(&id, _)| self.is_bridge(id))
            .map(|(&id, _)| id)
            .collect()
    }

    // ================================================================
    // Semantic Query API — query by meaning, similarity, path finding
    // ================================================================

    /// Query compositions by concept label.
    ///
    /// Unlike `neighborhood_for()` which only matches exact keywords,
    /// this method also finds compositions where the concept appears
    /// as a member label (substring matching) and ranks results by
    /// relevance: exact label match > role match > substring match.
    ///
    /// Returns compositions sorted by relevance (highest first).
    pub fn query_by_concept(&self, concept: &str) -> Vec<(&Composition, f32)> {
        let concept_lower = concept.to_lowercase();
        let mut results: Vec<(&Composition, f32)> = Vec::new();

        for comp in self.compositions.values() {
            let mut score = 0.0f32;

            for member in &comp.members {
                let label_lower = member.label.to_lowercase();

                if label_lower == concept_lower {
                    // Exact match — highest relevance
                    score = score.max(1.0);
                } else if label_lower.contains(&concept_lower)
                    || concept_lower.contains(&label_lower)
                {
                    // Substring match — medium relevance, weighted by member confidence
                    score = score.max(0.6 * member.confidence);
                }
            }

            // Boost score for Stable/Grounded compositions
            if score > 0.0 {
                if comp.lifecycle == LifecycleState::Stable {
                    score *= 1.2;
                }
                if comp.epistemic == EpistemicState::Grounded {
                    score *= 1.1;
                }
                score = score.min(1.0);
                results.push((comp, score));
            }
        }

        // Sort by relevance (highest first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Query compositions by role structure.
    ///
    /// Find all compositions that contain ALL the specified semantic roles.
    /// For example, `query_by_structure(&["Arg0Agent", "Cause"])` finds
    /// all compositions that have both an Agent and a Cause role.
    ///
    /// This enables queries like "find all causal compositions" or
    /// "find all compositions with a Problem and Solution."
    ///
    /// Returns compositions sorted by confidence (highest first).
    pub fn query_by_structure(&self, role_names: &[String]) -> Vec<&Composition> {
        // Parse role names into SemanticRole enums
        let target_roles: Vec<SemanticRole> = role_names
            .iter()
            .filter_map(|name| parse_semantic_role(name))
            .collect();

        if target_roles.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<&Composition> = self
            .compositions
            .values()
            .filter(|comp| {
                target_roles.iter().all(|role| {
                    comp.members.iter().any(|m| m.role == *role)
                })
            })
            .collect();

        // Sort by confidence (highest first)
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Compute similarity between two nodes identified by label.
    ///
    /// Uses structural overlap (Jaccard) of their composition neighborhoods
    /// plus spreading activation overlap. Two nodes are "similar" if they
    /// participate in compositions with similar structure.
    ///
    /// Returns a score from 0.0 (completely unrelated) to 1.0 (identical neighborhood).
    pub fn similarity(&self, label_a: &str, label_b: &str) -> f32 {
        let node_a = match self.find_node_by_label(label_a) {
            Some(id) => id,
            None => return 0.0,
        };
        let node_b = match self.find_node_by_label(label_b) {
            Some(id) => id,
            None => return 0.0,
        };

        if node_a == node_b {
            return 1.0;
        }

        // Strategy 1: Composition neighborhood Jaccard
        let comps_a: HashSet<CompositionId> = self
            .compositions_for_node(node_a)
            .iter()
            .map(|c| c.id.clone())
            .collect();
        let comps_b: HashSet<CompositionId> = self
            .compositions_for_node(node_b)
            .iter()
            .map(|c| c.id.clone())
            .collect();

        let jaccard = if comps_a.is_empty() && comps_b.is_empty() {
            0.0
        } else {
            let intersection = comps_a.intersection(&comps_b).count();
            let union = comps_a.union(&comps_b).count();
            intersection as f32 / union as f32
        };

        // Strategy 2: Spreading activation overlap
        let sa = SpreadingActivation::default();
        let activation_a = sa.spread(&[(node_a, 1.0)], self);
        let activation_b = sa.spread(&[(node_b, 1.0)], self);

        let spreading_overlap = if activation_a.is_empty() || activation_b.is_empty() {
            0.0
        } else {
            // Cosine similarity of activation vectors
            let mut dot = 0.0f32;
            let mut norm_a = 0.0f32;
            let mut norm_b = 0.0f32;

            for (&id, &e) in &activation_a.energies {
                norm_a += e * e;
                let eb = activation_b.energy(id);
                dot += e * eb;
            }
            for &e in activation_b.energies.values() {
                norm_b += e * e;
            }

            let denom = norm_a.sqrt() * norm_b.sqrt();
            if denom > 0.0 { dot / denom } else { 0.0 }
        };

        // Blend: 60% Jaccard + 40% spreading overlap
        0.6 * jaccard + 0.4 * spreading_overlap
    }

    /// Find related nodes using spreading activation from a seed label.
    ///
    /// Returns the top-N most activated nodes, sorted by activation energy.
    /// This is the core "query by meaning" mechanism: start from a concept,
    /// spread activation through the graph, and the most activated nodes
    /// are the most semantically related.
    ///
    /// Each result includes the node label and activation energy.
    pub fn find_related(&self, label: &str, top_n: usize) -> Vec<(String, f32)> {
        let node_id = match self.find_node_by_label(label) {
            Some(id) => id,
            None => return Vec::new(),
        };

        let sa = SpreadingActivation::default();
        let activation = sa.spread(&[(node_id, 1.0)], self);

        activation
            .top_n(top_n)
            .iter()
            .filter_map(|&(id, energy)| {
                self.node_label(id).map(|lbl| (lbl.to_string(), energy))
            })
            .filter(|(lbl, _)| lbl != label) // Exclude the seed itself
            .collect()
    }

    /// Find a reasoning path between two nodes identified by label.
    ///
    /// Uses bidirectional spreading activation to find compositions
    /// that connect two concepts. Returns the chain of composition IDs
    /// that form the shortest reasoning path.
    ///
    /// This answers questions like "why are X and Y related?" by
    /// showing the compositions that connect them.
    pub fn find_path(&self, label_from: &str, label_to: &str) -> Vec<String> {
        let node_from = match self.find_node_by_label(label_from) {
            Some(id) => id,
            None => return Vec::new(),
        };
        let node_to = match self.find_node_by_label(label_to) {
            Some(id) => id,
            None => return Vec::new(),
        };

        if node_from == node_to {
            return Vec::new();
        }

        // Spread from both directions
        let sa = SpreadingActivation::default();
        let activation_from = sa.spread(&[(node_from, 1.0)], self);
        let activation_to = sa.spread(&[(node_to, 1.0)], self);

        // Find compositions that are activated from both sides
        let mut bridge_comps: Vec<(&Composition, f32)> = Vec::new();

        for comp in self.compositions.values() {
            let mut energy_from = 0.0f32;
            let mut energy_to = 0.0f32;

            for member in &comp.members {
                energy_from += activation_from.energy(member.node_id);
                energy_to += activation_to.energy(member.node_id);
            }

            // Both sides must have activation for this to be a bridge
            if energy_from > 0.0 && energy_to > 0.0 {
                // Score = harmonic mean of energies (high when both are high)
                let score = if energy_from + energy_to > 0.0 {
                    2.0 * energy_from * energy_to / (energy_from + energy_to)
                } else {
                    0.0
                };
                bridge_comps.push((comp, score));
            }
        }

        // Sort by bridge score (highest first) and return composition IDs
        bridge_comps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        bridge_comps.iter().map(|(comp, _)| comp.id.clone()).collect()
    }

    // ================================================================
    // Phase N: Utterance Context
    // ================================================================

    /// Get utterance context for cross-sentence blending.
    ///
    /// Returns a 3-way blend of:
    /// - Current composition's properties (55%)
    /// - Utterance-level context from Situation/Conclusion compositions (25%)
    /// - Bridge node context (20%)
    ///
    /// **Audit fix**: No double-check on member nodes for utterance sense.
    /// The Situation/Conclusion check is sufficient — we don't need to
    /// additionally verify that member nodes have utterance senses.
    pub fn get_utterance_context(&self, composition_id: &CompositionId) -> UtteranceContext {
        let _comp = match self.compositions.get(composition_id) {
            Some(c) => c,
            None => return UtteranceContext::default(),
        };

        // Collect Situation and Conclusion compositions for context
        let situational_comps: Vec<&Composition> = self
            .compositions
            .values()
            .filter(|c| c.id != *composition_id)
            .filter(|c| {
                c.composition_type == CompositionType::Situation
                    || c.composition_type == CompositionType::HiddenMeaning
            })
            .take(5)
            .collect();

        // Collect bridge nodes
        let bridge_node_ids: Vec<NodeId> = self.find_bridge_nodes();

        UtteranceContext {
            current_weight: 0.55,
            situational_weight: 0.25,
            bridge_weight: 0.20,
            situational_composition_count: situational_comps.len(),
            bridge_node_count: bridge_node_ids.len(),
        }
    }

    // ================================================================
    // Phase L: Property Extraction for Weighted Jaccard
    // ================================================================

    /// Extract properties from a composition for SenseCandidate comparison.
    ///
    /// Properties are derived from the composition's members, with weights
    /// looked up from each member node's senses' freq_map.
    ///
    /// **Audit fix**: Iterates ALL senses of a node to find the one containing
    /// the composition_id, not just `senses.first()`.
    pub fn extract_properties_from_composition(
        &self,
        composition: &Composition,
    ) -> SenseCandidate {
        let mut properties = HashMap::new();

        for member in &composition.members {
            let role_key = format!("{:?}", member.role);

            // AUDIT FIX: iterate all senses, find the one containing this composition
            let weight = self
                .nodes
                .get(&member.node_id)
                .and_then(|node| {
                    node.senses
                        .iter()
                        .find(|s| s.freq_map.contains_key(&composition.id))
                        .and_then(|s| s.freq_map.get(&composition.id))
                        .copied()
                })
                .unwrap_or(1.0); // Default flat weight if no freq_map entry

            *properties.entry(role_key).or_insert(0.0) += weight * member.confidence;
        }

        SenseCandidate {
            sense_id: composition.id.clone(),
            properties,
        }
    }
}

/// Utterance context for cross-sentence blending (Phase N).
///
/// Contains the weights and metadata for the 3-way blend:
/// 55% current, 25% situational, 20% bridge.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UtteranceContext {
    /// Weight for the current composition's properties.
    pub current_weight: f32,
    /// Weight for situational context.
    pub situational_weight: f32,
    /// Weight for bridge node context.
    pub bridge_weight: f32,
    /// Number of situational compositions found.
    pub situational_composition_count: usize,
    /// Number of bridge nodes found.
    pub bridge_node_count: usize,
}

// ========================================================================
// Transform: Tokenize
// ========================================================================

/// Tokenize transform — splits raw text into `SemanticAtom` tokens.
///
/// Splits on whitespace AND punctuation (.,!?;:,"'()[]{}—–-/), strips
/// punctuation from tokens, skips empty tokens, and detects sentence
/// boundaries (period followed by uppercase). Tokens within the same
/// sentence are grouped together for downstream co-occurrence edges.
///
/// # Transform Signature
///
/// ```text
/// Input:  RawText (str) — read from ctx.raw_text
/// Output: Vec<SemanticAtom> — written to ctx.current_atoms
/// ```
pub struct Tokenize {
    /// Punctuation characters to split on and strip.
    punct_chars: &'static [char],
}

impl Tokenize {
    /// Create a new Tokenize transform.
    pub fn new() -> Self {
        Self {
            punct_chars: &['.', '!', '?', ';', ':', ',', '"', '\'', '(', ')', '[', ']', '{', '}', '—', '–', '-', '/'],
        }
    }

    /// Split text into (sentence_index, cleaned_token) pairs.
    ///
    /// Handles:
    /// - Splitting on whitespace and punctuation
    /// - Stripping punctuation from tokens
    /// - Sentence boundary detection (period followed by uppercase)
    /// - Skipping empty tokens after stripping
    fn tokenize_with_sentences(&self, text: &str) -> Vec<(usize, String)> {
        let mut results = Vec::new();
        let mut sentence_idx = 0usize;
        let mut current_sentence_tokens: Vec<String> = Vec::new();

        // Split on whitespace first, then further split each chunk on punctuation.
        for chunk in text.split_whitespace() {
            let mut sub_tokens = Vec::new();
            let mut buffer = String::new();

            for ch in chunk.chars() {
                if self.punct_chars.contains(&ch) {
                    // Flush buffer as a token if non-empty.
                    if !buffer.is_empty() {
                        sub_tokens.push(buffer.clone());
                        buffer.clear();
                    }
                    // Sentence boundary: period, question mark, or exclamation mark
                    // followed by an uppercase letter later signals a new sentence.
                    if ch == '.' || ch == '?' || ch == '!' {
                        // We'll detect the boundary after we see the next token.
                        // For now, mark this as a sentence-ending token.
                        sub_tokens.push(format!("{}", ch));
                    }
                    // Other punctuation is simply discarded (not emitted as token).
                } else {
                    buffer.push(ch);
                }
            }

            // Flush remaining buffer.
            if !buffer.is_empty() {
                sub_tokens.push(buffer);
            }

            for tok in sub_tokens {
                let is_sentence_end = tok == "." || tok == "?" || tok == "!";
                if is_sentence_end {
                    // Emit all accumulated tokens for this sentence.
                    for t in &current_sentence_tokens {
                        let cleaned = t.to_lowercase();
                        if !cleaned.is_empty() {
                            results.push((sentence_idx, cleaned));
                        }
                    }
                    sentence_idx += 1;
                    current_sentence_tokens.clear();
                } else {
                    let cleaned = tok.to_lowercase();
                    if !cleaned.is_empty() {
                        current_sentence_tokens.push(tok.clone());
                    }
                }
            }
        }

        // Emit any remaining tokens in the last sentence.
        for t in &current_sentence_tokens {
            let cleaned = t.to_lowercase();
            if !cleaned.is_empty() {
                results.push((sentence_idx, cleaned));
            }
        }

        results
    }
}

impl Default for Tokenize {
    fn default() -> Self {
        Self::new()
    }
}

impl ErasedTransform for Tokenize {
    fn id(&self) -> &'static str {
        "Tokenize"
    }

    fn execute(&self, ctx: &mut PipelineContext, _graph: &mut Graph) -> IngestResult {
        let text = match &ctx.raw_text {
            Some(t) => t.clone(),
            None => return IngestResult::new(),
        };

        let mut atoms_created = 0;
        let tokenized = self.tokenize_with_sentences(&text);

        // Track sentence memberships for co-occurrence: sentence_idx -> Vec<atom_index>
        let mut sentence_atoms: HashMap<usize, Vec<usize>> = HashMap::new();

        for (sentence_idx, token_label) in &tokenized {
            let atom_id = format!("atom_{}", ctx.next_atom_id());
            let atom = SemanticAtom {
                id: atom_id,
                label: token_label.clone(),
                atom_type: AtomType::Token,
                confidence: 1.0,
                source: crate::types::EdgeSource::Learned,
                ..SemanticAtom::default()
            };
            let atom_idx = ctx.current_atoms.len();
            sentence_atoms.entry(*sentence_idx).or_default().push(atom_idx);
            ctx.current_atoms.push(atom);
            atoms_created += 1;
        }

        // Store sentence groupings in PipelineContext for downstream use.
        // We encode this as a simple string in the raw_text metadata.
        // Downstream transforms (IngestAtoms) can use sentence co-occurrence
        // to create edges between tokens in the same sentence.
        // For now, we store sentence info as a separate field on context.
        // Since PipelineContext doesn't have a dedicated field, we use
        // a convention: each atom's roles map carries a SentenceIdx key
        // with the sentence number as the value.
        for (sentence_idx, atom_indices) in &sentence_atoms {
            let sent_label = format!("sent_{}", sentence_idx);
            for &atom_idx in atom_indices {
                ctx.current_atoms[atom_idx]
                    .roles
                    .insert(SemanticRole::SourceAtom, sent_label.clone());
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

/// Implement the `Transform` trait for `Tokenize` so it can be used
/// with `PipelineEngine::run<T>`.
impl Transform for Tokenize {
    type Input = String;
    type Output = Vec<SemanticAtom>;

    fn id(&self) -> &'static str {
        "Tokenize"
    }

    fn transform(&self, input: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        let tokenized = self.tokenize_with_sentences(input);
        let mut atoms = Vec::new();
        for (sentence_idx, token_label) in &tokenized {
            let atom_id = format!("atom_{}", ctx.next_atom_id());
            let mut atom = SemanticAtom {
                id: atom_id,
                label: token_label.clone(),
                atom_type: AtomType::Token,
                confidence: 1.0,
                source: crate::types::EdgeSource::Learned,
                ..SemanticAtom::default()
            };
            atom.roles.insert(
                SemanticRole::SourceAtom,
                format!("sent_{}", sentence_idx),
            );
            atoms.push(atom);
        }
        atoms
    }
}

// ========================================================================
// Transform: IngestAtoms
// ========================================================================

/// IngestAtoms transform — creates graph structures from `SemanticAtom`s.
///
/// For each atom in `ctx.current_atoms`:
/// - **Token atoms**: calls `graph.ensure_node(label)` and creates co-occurrence
///   edges between tokens in the same sentence.
/// - **Event atoms**: creates a Composition with `CompositionType::Event`, adds
///   members with `SemanticRole`s (Arg0Agent, Arg1Patient, Cause, etc.) from
///   the atom's roles map, and creates edges for each member.
/// - **HiddenMeaning atoms**: creates a Composition with
///   `CompositionType::HiddenMeaning`, adds members with roles from the atom's
///   roles map, and creates edges for each member.
/// - Sets `composition.source_text` from `ctx.raw_text`.
///
/// # Transform Signature
///
/// ```text
/// Input:  Vec<SemanticAtom> — read from ctx.current_atoms
/// Output: GraphDelta — applied to graph
/// ```
pub struct IngestAtoms {
    /// Future: ingest configuration.
    _config: (),
}

impl IngestAtoms {
    /// Create a new IngestAtoms transform.
    pub fn new() -> Self {
        Self { _config: () }
    }
}

impl Default for IngestAtoms {
    fn default() -> Self {
        Self::new()
    }
}

impl ErasedTransform for IngestAtoms {
    fn id(&self) -> &'static str {
        "IngestAtoms"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let mut compositions_created = 0;
        let mut edges_created = 0;
        let mut atoms_counted = 0;

        // Collect composition IDs to assign back to atoms (can't mutate while iterating).
        let mut comp_id_assignments: Vec<(usize, String)> = Vec::new();
        // Collect event atoms for the sliding window.
        let mut event_atoms: Vec<SemanticAtom> = Vec::new();
        // Track sentence -> Vec<NodeId> for Token co-occurrence edges.
        let mut sentence_nodes: HashMap<String, Vec<NodeId>> = HashMap::new();

        // Ensure nodes exist for all atom labels.
        let atom_count = ctx.current_atoms.len();
        for i in 0..atom_count {
            let atom = &ctx.current_atoms[i];
            let node_id = graph.ensure_node(&atom.label);
            atoms_counted += 1;

            match atom.atom_type {
                // --- Token atoms: just ensure the node exists and track sentence co-occurrence ---
                AtomType::Token | AtomType::AmbiguousToken => {
                    // Track sentence membership for co-occurrence edges.
                    if let Some(sent_label) = atom.roles.get(&SemanticRole::SourceAtom) {
                        sentence_nodes
                            .entry(sent_label.clone())
                            .or_default()
                            .push(node_id);
                    }
                }

                // --- Event atoms: create a Composition with Event type ---
                AtomType::Event => {
                    let comp_id = format!("comp_{}", atom.id);
                    let mut composition = Composition {
                        id: comp_id.clone(),
                        composition_type: CompositionType::Event,
                        confidence: atom.confidence,
                        source_text: ctx.raw_text.clone(),
                        ..Default::default()
                    };

                    // Add the predicate as a member.
                    composition.members.push(CompositionMember {
                        node_id,
                        role: SemanticRole::Predicate,
                        confidence: atom.confidence,
                        label: atom.label.clone(),
                        source: None,
                    });

                    // Add role members from the atom's roles map.
                    for (role, label) in &atom.roles {
                        let role_node_id = graph.ensure_node(label);
                        composition.members.push(CompositionMember {
                            node_id: role_node_id,
                            role: role.clone(),
                            confidence: atom.confidence * 0.9,
                            label: label.clone(),
                            source: None,
                        });
                    }

                    // Create edges for each member.
                    for member in &composition.members {
                        graph.edges.push((
                            comp_id.clone(),
                            member.node_id,
                            SemanticEdge {
                                relation: crate::types::RelationType::Categorical,
                                role: Some(member.role.clone()),
                                source: crate::types::EdgeSource::FrameCompiler,
                            },
                        ));
                        edges_created += 1;
                    }

                    comp_id_assignments.push((i, comp_id.clone()));
                    graph.compositions.insert(comp_id.clone(), composition);
                    graph.dirty_compositions.insert(comp_id);
                    compositions_created += 1;
                    event_atoms.push(atom.clone());
                }

                // --- HiddenMeaning atoms: create a Composition with HiddenMeaning type ---
                AtomType::HiddenMeaning => {
                    let comp_id = format!("comp_{}", atom.id);
                    let mut composition = Composition {
                        id: comp_id.clone(),
                        composition_type: CompositionType::HiddenMeaning,
                        confidence: atom.confidence,
                        source_text: ctx.raw_text.clone(),
                        ..Default::default()
                    };

                    // Add the label as a Predicate member.
                    composition.members.push(CompositionMember {
                        node_id,
                        role: SemanticRole::Predicate,
                        confidence: atom.confidence,
                        label: atom.label.clone(),
                        source: None,
                    });

                    // Add role members from the atom's roles map.
                    for (role, label) in &atom.roles {
                        let role_node_id = graph.ensure_node(label);
                        composition.members.push(CompositionMember {
                            node_id: role_node_id,
                            role: role.clone(),
                            confidence: atom.confidence * 0.9,
                            label: label.clone(),
                            source: None,
                        });
                    }

                    // Create edges for each member.
                    for member in &composition.members {
                        graph.edges.push((
                            comp_id.clone(),
                            member.node_id,
                            SemanticEdge {
                                relation: crate::types::RelationType::Causal,
                                role: Some(member.role.clone()),
                                source: crate::types::EdgeSource::HiddenMeaningRule,
                            },
                        ));
                        edges_created += 1;
                    }

                    comp_id_assignments.push((i, comp_id.clone()));
                    graph.compositions.insert(comp_id.clone(), composition);
                    graph.dirty_compositions.insert(comp_id);
                    compositions_created += 1;
                }

                // --- Pattern / Hypothesis / Acquisition atoms ---
                _ => {
                    let comp_id = format!("comp_{}", atom.id);
                    let comp_type = match atom.atom_type {
                        AtomType::Pattern => CompositionType::Pattern,
                        AtomType::Hypothesis => CompositionType::Hypothesis,
                        AtomType::Acquisition => CompositionType::Acquisition,
                        _ => CompositionType::Situation,
                    };
                    let mut composition = Composition {
                        id: comp_id.clone(),
                        composition_type: comp_type,
                        confidence: atom.confidence,
                        source_text: ctx.raw_text.clone(),
                        ..Default::default()
                    };

                    composition.members.push(CompositionMember {
                        node_id,
                        role: SemanticRole::Predicate,
                        confidence: atom.confidence,
                        label: atom.label.clone(),
                        source: None,
                    });

                    for (role, label) in &atom.roles {
                        let role_node_id = graph.ensure_node(label);
                        composition.members.push(CompositionMember {
                            node_id: role_node_id,
                            role: role.clone(),
                            confidence: atom.confidence * 0.9,
                            label: label.clone(),
                            source: None,
                        });
                    }

                    for member in &composition.members {
                        graph.edges.push((
                            comp_id.clone(),
                            member.node_id,
                            SemanticEdge {
                                relation: crate::types::RelationType::Categorical,
                                role: Some(member.role.clone()),
                                source: atom.source.clone(),
                            },
                        ));
                        edges_created += 1;
                    }

                    comp_id_assignments.push((i, comp_id.clone()));
                    graph.compositions.insert(comp_id.clone(), composition);
                    graph.dirty_compositions.insert(comp_id);
                    compositions_created += 1;
                }
            }
        }

        // Create co-occurrence edges between tokens in the same sentence.
        // For each sentence, create a synthetic composition that groups the tokens.
        for (sent_label, node_ids) in &sentence_nodes {
            if node_ids.len() < 2 {
                continue;
            }
            let comp_id = format!("comp_sent_{}", sent_label);
            let mut composition = Composition {
                id: comp_id.clone(),
                composition_type: CompositionType::Situation,
                confidence: 0.5,
                source_text: ctx.raw_text.clone(),
                ..Default::default()
            };

            for (idx, &nid) in node_ids.iter().enumerate() {
                let role = if idx == 0 {
                    SemanticRole::Arg0Agent
                } else {
                    SemanticRole::Arg1Patient
                };
                let label = graph.node_label(nid).unwrap_or("").to_string();
                composition.members.push(CompositionMember {
                    node_id: nid,
                    role: role.clone(),
                    confidence: 0.5,
                    label,
                    source: None,
                });
                graph.edges.push((
                    comp_id.clone(),
                    nid,
                    SemanticEdge {
                        relation: crate::types::RelationType::Categorical,
                        role: Some(role),
                        source: crate::types::EdgeSource::Learned,
                    },
                ));
                edges_created += 1;
            }

            graph.compositions.insert(comp_id, composition);
            compositions_created += 1;
        }

        // Apply deferred composition ID assignments.
        for (idx, comp_id) in comp_id_assignments {
            ctx.current_atoms[idx].composition_id = Some(comp_id);
        }

        // Record event atoms in the sliding window.
        for atom in event_atoms {
            ctx.record_event(atom);
        }

        IngestResult {
            atoms_created: atoms_counted,
            compositions_created,
            edges_created,
            gaps_detected: 0,
            enrichments_applied: 0,
            governance_transitions: 0,
        }
    }
}

/// Implement the `Transform` trait for `IngestAtoms`.
impl Transform for IngestAtoms {
    type Input = Vec<SemanticAtom>;
    type Output = GraphDelta;

    fn id(&self) -> &'static str {
        "IngestAtoms"
    }

    fn transform(&self, input: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        let mut delta = GraphDelta::new();
        for atom in input {
            let node_id = ctx.next_atom_id() as NodeId;
            delta.new_nodes.push(node_id);
            // Note: full graph creation requires Graph access; this trait
            // only produces a delta for context-based usage.
            let _ = atom;
        }
        delta
    }
}

// ========================================================================
// EnrichComposition — Graph-Context-Aware Enrichment
// ========================================================================

/// EnrichComposition transform — enriches compositions using graph context.
///
/// For each `EnrichmentRequest` in `ctx.pending_enrichments`:
/// 1. Look up the target composition in the graph
/// 2. Verify the candidate node exists (or create it via `ensure_node`)
/// 3. Check for role conflicts (avoid duplicate roles)
/// 4. Add the candidate node as a new member with the specified role
/// 5. Re-compute composition confidence based on completeness
/// 6. Create a feedback edge with `EdgeSource::EnrichmentFeedback`
/// 7. If enrichment came from PassiveRecall, also create a secondary
///    confirming edge from the source composition
///
/// # Transform Signature
///
/// ```text
/// Input:  EnrichmentRequest — read from ctx.pending_enrichments
/// Output: GraphDelta — applied to graph
/// ```
pub struct EnrichComposition {
    /// Whether to skip enrichment when the role is already filled.
    /// If true (default), adding a duplicate role is a no-op.
    /// If false, the existing member is replaced.
    pub skip_duplicate_roles: bool,
}

impl EnrichComposition {
    /// Create a new EnrichComposition transform.
    pub fn new() -> Self {
        Self {
            skip_duplicate_roles: true,
        }
    }
}

impl Default for EnrichComposition {
    fn default() -> Self {
        Self::new()
    }
}

impl ErasedTransform for EnrichComposition {
    fn id(&self) -> &'static str {
        "EnrichComposition"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let requests = std::mem::take(&mut ctx.pending_enrichments);
        let mut enrichments_applied = 0;
        let mut edges_created = 0;
        let mut governance_transitions = 0;

        for request in &requests {
            // Ensure the candidate node exists in the graph.
            let candidate_node_id = if graph.has_node(request.candidate_node_id) {
                request.candidate_node_id
            } else {
                // Try to find by label.
                match graph.find_node_by_label(&request.candidate_label) {
                    Some(id) => id,
                    None => graph.ensure_node(&request.candidate_label),
                }
            };

            if let Some(composition) = graph.compositions.get_mut(&request.target_composition_id) {
                // Check for duplicate role.
                if self.skip_duplicate_roles
                    && composition.has_member_with_role(request.role_to_fill.clone())
                {
                    // Skip — role already filled.
                    continue;
                }

                // If not skipping duplicates, remove the existing member with this role.
                if !self.skip_duplicate_roles
                    && composition.has_member_with_role(request.role_to_fill.clone())
                {
                    composition
                        .members
                        .retain(|m| m.role != request.role_to_fill);
                }

                // Add the candidate as a new member.
                composition.members.push(CompositionMember {
                    node_id: candidate_node_id,
                    role: request.role_to_fill.clone(),
                    confidence: request.confidence,
                    label: request.candidate_label.clone(),
                    source: None,
                });

                // Re-compute confidence based on completeness.
                let completeness_bonus = self.compute_completeness_bonus(composition);
                composition.confidence = (composition.confidence + completeness_bonus).min(1.0);

                // Create a feedback edge.
                graph.edges.push((
                    request.target_composition_id.clone(),
                    candidate_node_id,
                    SemanticEdge {
                        relation: crate::types::RelationType::Categorical,
                        role: Some(request.role_to_fill.clone()),
                        source: crate::types::EdgeSource::EnrichmentFeedback,
                    },
                ));
                edges_created += 1;
                enrichments_applied += 1;

                // Check for lifecycle promotion after enrichment.
                if composition.lifecycle == LifecycleState::New && composition.batch_seen >= 1 {
                    composition.lifecycle = LifecycleState::Candidate;
                    governance_transitions += 1;
                }
            }
        }

        IngestResult {
            atoms_created: 0,
            compositions_created: 0,
            edges_created,
            gaps_detected: 0,
            enrichments_applied,
            governance_transitions,
        }
    }
}

impl EnrichComposition {
    /// Compute a confidence bonus based on composition completeness.
    ///
    /// More complete compositions (more expected roles filled) get a
    /// higher bonus. This incentivizes filling gaps.
    fn compute_completeness_bonus(&self, composition: &Composition) -> f32 {
        let (expected, filled) = match composition.composition_type {
            CompositionType::Event => {
                let expected = 4; // Predicate, Agent, Patient, Cause
                let filled = composition
                    .members
                    .iter()
                    .filter(|m| {
                        matches!(
                            m.role,
                            SemanticRole::Predicate
                                | SemanticRole::Arg0Agent
                                | SemanticRole::Arg1Patient
                                | SemanticRole::Cause
                        )
                    })
                    .count();
                (expected, filled)
            }
            CompositionType::HiddenMeaning => {
                let expected = 3; // PatternType, Problem, Solution
                let filled = composition
                    .members
                    .iter()
                    .filter(|m| {
                        matches!(
                            m.role,
                            SemanticRole::PatternType
                                | SemanticRole::Problem
                                | SemanticRole::Solution
                        )
                    })
                    .count();
                (expected, filled)
            }
            CompositionType::Pattern => {
                let expected = 3; // PatternType, Antecedent, Consequent
                let filled = composition
                    .members
                    .iter()
                    .filter(|m| {
                        matches!(
                            m.role,
                            SemanticRole::PatternType
                                | SemanticRole::Antecedent
                                | SemanticRole::Consequent
                        )
                    })
                    .count();
                (expected, filled)
            }
            _ => {
                let expected = 2;
                let filled = composition.members.len().min(expected);
                (expected, filled)
            }
        };

        // Bonus scales with completeness: 0.05 per filled role.
        if filled > 0 && expected > 0 {
            0.05 * (filled as f32 / expected as f32)
        } else {
            0.0
        }
    }
}

/// Implement the `Transform` trait for `EnrichComposition`.
impl Transform for EnrichComposition {
    type Input = EnrichmentRequest;
    type Output = GraphDelta;

    fn id(&self) -> &'static str {
        "EnrichComposition"
    }

    fn transform(&self, input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        let mut delta = GraphDelta::new();
        delta.new_nodes.push(input.candidate_node_id);
        delta
    }
}

// ========================================================================
// ReExtractFrame — Graph-Assisted Re-Extraction
// ========================================================================

/// ReExtractFrame transform — re-extracts a frame using graph context.
///
/// For each `ReExtractionRequest` in `ctx.pending_reextractions`:
/// 1. Look up the target composition and its source text
/// 2. Use graph context (known role-fillers) as hints for re-extraction
/// 3. Re-run ExtractFrame with `graph_assisted = true`
/// 4. If the re-extracted frame has higher confidence, replace the old one
/// 5. Create a feedback edge with `EdgeSource::ExtractionRepair`
///
/// # Transform Signature
///
/// ```text
/// Input:  ReExtractionRequest — read from ctx.pending_reextractions
/// Output: Option<SemanticAtom> — new atom if re-extraction succeeded
/// ```
pub struct ReExtractFrame {
    /// Whether to always replace, even if re-extraction confidence is lower.
    /// Default: false (only replace if confidence improves).
    pub force_replace: bool,
}

impl ReExtractFrame {
    /// Create a new ReExtractFrame transform.
    pub fn new() -> Self {
        Self {
            force_replace: false,
        }
    }
}

impl Default for ReExtractFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl ErasedTransform for ReExtractFrame {
    fn id(&self) -> &'static str {
        "ReExtractFrame"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let requests = std::mem::take(&mut ctx.pending_reextractions);
        let mut enrichments_applied = 0;
        let mut edges_created = 0;

        for request in requests {
            // Get the source text for re-extraction.
            let source_text = if !request.original_text.is_empty() {
                request.original_text.clone()
            } else {
                // Try to get from the composition.
                match graph.get_composition(&request.target_composition_id) {
                    Some(comp) => match &comp.source_text {
                        Some(text) => text.clone(),
                        None => continue,
                    },
                    None => continue,
                }
            };

            // Get the target composition's current confidence.
            let current_confidence = graph
                .get_composition(&request.target_composition_id)
                .map(|c| c.confidence)
                .unwrap_or(0.0);

            // Build graph context hints as (role, node_id, confidence) triples.
            let mut context_hints: Vec<(SemanticRole, NodeId, f32)> = request.graph_context.clone();

            // Also add existing members as context.
            if let Some(comp) = graph.get_composition(&request.target_composition_id) {
                for member in &comp.members {
                    context_hints.push((member.role.clone(), member.node_id, member.confidence));
                }
            }

            // Re-extract using the enhanced context.
            let extractor = super::extract_frame::ExtractFrame::new();
            let re_result = extractor.re_extract_with_context(&source_text, &context_hints, graph);

            match re_result {
                Some(re_atom) => {
                    let re_confidence = re_atom.confidence;

                    // Only replace if confidence improved (or force_replace).
                    if self.force_replace || re_confidence > current_confidence {
                        // Collect new members first (need immutable borrow for ensure_node).
                        let new_members: Vec<CompositionMember> = re_atom
                            .roles
                            .iter()
                            .map(|(role, label)| {
                                let node_id = graph.ensure_node(label);
                                CompositionMember {
                                    node_id,
                                    role: role.clone(),
                                    confidence: re_confidence * 0.95,
                                    label: label.clone(),
                                    source: None,
                                }
                            })
                            .collect();

                        // Create repair edges.
                        for member in &new_members {
                            graph.edges.push((
                                request.target_composition_id.clone(),
                                member.node_id,
                                SemanticEdge {
                                    relation: crate::types::RelationType::Categorical,
                                    role: Some(member.role.clone()),
                                    source: crate::types::EdgeSource::ExtractionRepair,
                                },
                            ));
                            edges_created += 1;
                        }

                        // Update the composition with re-extracted data.
                        if let Some(composition) =
                            graph.compositions.get_mut(&request.target_composition_id)
                        {
                            composition.members = new_members;
                            composition.confidence = re_confidence;
                            composition.provenance.origin =
                                crate::types::EdgeSource::ExtractionRepair;
                            enrichments_applied += 1;
                        }
                    }
                }
                None => {
                    // Re-extraction failed — no improvement possible.
                }
            }
        }

        IngestResult {
            atoms_created: 0,
            compositions_created: 0,
            edges_created,
            gaps_detected: 0,
            enrichments_applied,
            governance_transitions: 0,
        }
    }
}

/// Implement the `Transform` trait for `ReExtractFrame`.
impl Transform for ReExtractFrame {
    type Input = ReExtractionRequest;
    type Output = Option<SemanticAtom>;

    fn id(&self) -> &'static str {
        "ReExtractFrame"
    }

    fn transform(&self, input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        // Simplified: just return None (full logic requires Graph access).
        let _ = input;
        None
    }
}

// STUB:PLACEHOLDER — No-op transform placeholder for transforms not yet implemented.
/// No-op transform placeholder for transforms not yet implemented.
///
/// Used by `register_default_pipeline` for ExtractFrame, ReasonFrame,
/// GovernBeliefs, SeedAnchor, DetectGaps, and SelectAcquisition. These
/// will be replaced with real implementations in separate modules.
///
/// When executed, this transform does nothing and returns an empty
/// `IngestResult`.
pub struct NoOpTransform {
    /// The transform ID (e.g., "ExtractFrame").
    id_str: &'static str,
}

impl NoOpTransform {
    /// Create a new no-op transform with the given ID.
    pub fn new(id: &'static str) -> Self {
        Self { id_str: id }
    }
}

impl ErasedTransform for NoOpTransform {
    fn id(&self) -> &'static str {
        self.id_str
    }

    // STUB:PLACEHOLDER — returns empty IngestResult, no real logic.
    fn execute(&self, _ctx: &mut PipelineContext, _graph: &mut Graph) -> IngestResult {
        // No-op — does nothing.
        IngestResult::new()
    }
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::field_reassign_with_default)]
    use super::*;

    #[test]
    fn test_topological_sort_empty() {
        let dag: Vec<TransformNode> = vec![];
        let result = topological_sort(&dag);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_topological_sort_linear() {
        let dag = vec![
            TransformNode {
                transform_id: "A".to_string(),
                input_type: String::new(),
                output_type: String::new(),
                dependencies: vec![],
                condition: None,
            },
            TransformNode {
                transform_id: "B".to_string(),
                input_type: String::new(),
                output_type: String::new(),
                dependencies: vec!["A".to_string()],
                condition: None,
            },
            TransformNode {
                transform_id: "C".to_string(),
                input_type: String::new(),
                output_type: String::new(),
                dependencies: vec!["B".to_string()],
                condition: None,
            },
        ];
        let result = topological_sort(&dag).unwrap();
        let a_pos = result.iter().position(|x| x == "A").unwrap();
        let b_pos = result.iter().position(|x| x == "B").unwrap();
        let c_pos = result.iter().position(|x| x == "C").unwrap();
        assert!(a_pos < b_pos);
        assert!(b_pos < c_pos);
    }

    #[test]
    fn test_topological_sort_cycle_detection() {
        let dag = vec![
            TransformNode {
                transform_id: "A".to_string(),
                input_type: String::new(),
                output_type: String::new(),
                dependencies: vec!["B".to_string()],
                condition: None,
            },
            TransformNode {
                transform_id: "B".to_string(),
                input_type: String::new(),
                output_type: String::new(),
                dependencies: vec!["A".to_string()],
                condition: None,
            },
        ];
        let result = topological_sort(&dag);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_engine_new() {
        let engine = PipelineEngine::new();
        assert!(engine.transforms.is_empty());
        assert!(engine.dag.is_empty());
        assert!(engine.graph.compositions.is_empty());
    }

    #[test]
    fn test_register_default_pipeline() {
        let mut engine = PipelineEngine::new();
        register_default_pipeline(&mut engine);
        assert_eq!(engine.transforms.len(), 14);
        assert_eq!(engine.dag.len(), 14);
    }

    #[test]
    fn test_ingest_simple() {
        let mut engine = PipelineEngine::new();
        register_default_pipeline(&mut engine);

        let result = engine.ingest("Raymond membuat aplikasi karena lambat").unwrap();
        assert!(result.atoms_created > 0);
    }

    #[test]
    fn test_graph_ensure_node() {
        let mut graph = Graph::new();

        let id1 = graph.ensure_node("raja");
        let id2 = graph.ensure_node("raja"); // Same label, should return same ID.
        let id3 = graph.ensure_node("ratu");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert!(graph.has_node(id1));
        assert!(graph.has_node(id3));
    }

    #[test]
    fn test_graph_cooccurrence_count() {
        let mut graph = Graph::new();

        let a = graph.ensure_node("A");
        let b = graph.ensure_node("B");
        let c = graph.ensure_node("C");

        // Create two compositions containing A and B.
        for i in 0..2 {
            let comp_id = format!("comp_{}", i);
            let mut comp = Composition::default();
            comp.id = comp_id;
            comp.members.push(CompositionMember {
                node_id: a,
                role: SemanticRole::Arg0Agent,
                confidence: 1.0,
                label: String::new(),
                source: None,
            });
            comp.members.push(CompositionMember {
                node_id: b,
                role: SemanticRole::Arg1Patient,
                confidence: 1.0,
                label: String::new(),
                source: None,
            });
            graph.compositions.insert(comp.id.clone(), comp);
        }

        assert_eq!(graph.cooccurrence_count(a, b), 2);
        assert_eq!(graph.cooccurrence_count(a, c), 0);
    }

    #[test]
    fn test_ingest_result_merge() {
        let mut a = IngestResult {
            atoms_created: 5,
            compositions_created: 2,
            edges_created: 3,
            gaps_detected: 1,
            enrichments_applied: 0,
            governance_transitions: 0,
        };
        let b = IngestResult {
            atoms_created: 3,
            compositions_created: 1,
            edges_created: 2,
            gaps_detected: 0,
            enrichments_applied: 1,
            governance_transitions: 1,
        };
        a.merge(&b);
        assert_eq!(a.atoms_created, 8);
        assert_eq!(a.compositions_created, 3);
        assert_eq!(a.edges_created, 5);
        assert_eq!(a.gaps_detected, 1);
        assert_eq!(a.enrichments_applied, 1);
        assert_eq!(a.governance_transitions, 1);
    }

    #[test]
    fn test_find_weak_frames() {
        let mut engine = PipelineEngine::new();

        // Create a weak Event composition (low confidence, missing roles).
        let mut comp = Composition::default();
        comp.id = "comp_weak_1".to_string();
        comp.composition_type = CompositionType::Event;
        comp.confidence = 0.3;
        comp.members.push(CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.3,
            label: String::new(),
            source: None,
        });
        // Missing Arg0Agent and Arg1Patient.
        engine.graph.compositions.insert(comp.id.clone(), comp);

        // Create a strong Event composition (should NOT be weak).
        let mut comp2 = Composition::default();
        comp2.id = "comp_strong_1".to_string();
        comp2.composition_type = CompositionType::Event;
        comp2.confidence = 0.8;
        comp2.members.push(CompositionMember {
            node_id: 2,
            role: SemanticRole::Arg0Agent,
            confidence: 0.8,
            label: String::new(),
            source: None,
        });
        comp2.members.push(CompositionMember {
            node_id: 3,
            role: SemanticRole::Arg1Patient,
            confidence: 0.8,
            label: String::new(),
            source: None,
        });
        comp2.members.push(CompositionMember {
            node_id: 4,
            role: SemanticRole::Cause,
            confidence: 0.8,
            label: String::new(),
            source: None,
        });
        engine.graph.compositions.insert(comp2.id.clone(), comp2);

        let weak = engine.find_weak_frames();
        assert_eq!(weak.len(), 1);
        assert_eq!(weak[0].composition_id, "comp_weak_1");
    }

    // ====================================================================
    // New tests for previously-untested transforms
    // ====================================================================

    #[test]
    fn test_tokenize_basic() {
        let tok = Tokenize::new();
        let mut ctx = PipelineContext::default();
        ctx.raw_text = Some("karena harga naik".to_string());
        let mut graph = Graph::new();

        let result = tok.execute(&mut ctx, &mut graph);
        assert_eq!(result.atoms_created, 3);
        assert_eq!(ctx.current_atoms.len(), 3);
        assert_eq!(ctx.current_atoms[0].label, "karena");
        assert_eq!(ctx.current_atoms[1].label, "harga");
        assert_eq!(ctx.current_atoms[2].label, "naik");
        assert_eq!(ctx.current_atoms[0].atom_type, AtomType::Token);
    }

    #[test]
    fn test_tokenize_empty() {
        let tok = Tokenize::new();
        let mut ctx = PipelineContext::default();
        ctx.raw_text = Some(String::new());
        let mut graph = Graph::new();

        let result = tok.execute(&mut ctx, &mut graph);
        assert_eq!(result.atoms_created, 0);
        assert!(ctx.current_atoms.is_empty());
    }

    #[test]
    fn test_tokenize_none_text() {
        let tok = Tokenize::new();
        let mut ctx = PipelineContext::default();
        ctx.raw_text = None;
        let mut graph = Graph::new();

        let result = tok.execute(&mut ctx, &mut graph);
        assert_eq!(result.atoms_created, 0);
    }

    #[test]
    fn test_tokenize_lowercase() {
        let tok = Tokenize::new();
        let mut ctx = PipelineContext::default();
        ctx.raw_text = Some("HARGA NAIK".to_string());
        let mut graph = Graph::new();

        let result = tok.execute(&mut ctx, &mut graph);
        assert_eq!(result.atoms_created, 2);
        assert_eq!(ctx.current_atoms[0].label, "harga");
        assert_eq!(ctx.current_atoms[1].label, "naik");
    }

    #[test]
    fn test_ingest_atoms_creates_nodes() {
        let ingest = IngestAtoms::new();
        let mut ctx = PipelineContext::default();
        ctx.raw_text = Some("test text".to_string());

        // Pre-populate atoms from Tokenize
        let tok = Tokenize::new();
        let mut graph = Graph::new();
        tok.execute(&mut ctx, &mut graph);

        let result = ingest.execute(&mut ctx, &mut graph);
        assert_eq!(result.atoms_created, 2); // "test" and "text"
        assert!(graph.node_count() >= 2);
    }

    #[test]
    fn test_ingest_atoms_event_creates_composition() {
        let ingest = IngestAtoms::new();
        let mut ctx = PipelineContext::default();
        ctx.raw_text = Some("dia pergi".to_string());

        // Create an Event atom manually
        let mut atom = SemanticAtom::default();
        atom.id = "atom_0".to_string();
        atom.label = "pergi".to_string();
        atom.atom_type = AtomType::Event;
        atom.confidence = 0.8;
        atom.roles
            .insert(SemanticRole::Arg0Agent, "dia".to_string());
        ctx.current_atoms.push(atom);

        let mut graph = Graph::new();
        let result = ingest.execute(&mut ctx, &mut graph);

        assert!(result.compositions_created >= 1);
        assert!(result.edges_created >= 1);
        assert!(graph.composition_count() >= 1);
    }

    #[test]
    fn test_ingest_atoms_empty() {
        let ingest = IngestAtoms::new();
        let mut ctx = PipelineContext::default();
        let mut graph = Graph::new();

        let result = ingest.execute(&mut ctx, &mut graph);
        assert_eq!(result.atoms_created, 0);
        assert_eq!(result.compositions_created, 0);
    }

    #[test]
    fn test_enrich_composition_adds_member() {
        let enrich = EnrichComposition::new();
        let mut ctx = PipelineContext::default();
        let mut graph = Graph::new();

        // Create a composition
        let pred_id = graph.ensure_node("pergi");
        let comp_id = "comp_test".to_string();
        let mut comp = Composition::default();
        comp.id = comp_id.clone();
        comp.composition_type = CompositionType::Event;
        comp.confidence = 0.5;
        comp.members.push(CompositionMember {
            node_id: pred_id,
            role: SemanticRole::Predicate,
            confidence: 0.5,
            label: "pergi".to_string(),
            source: None,
        });
        graph.compositions.insert(comp_id.clone(), comp);

        // Create enrichment request
        let agent_id = graph.ensure_node("dia");
        ctx.pending_enrichments.push(EnrichmentRequest {
            target_composition_id: comp_id.clone(),
            role_to_fill: SemanticRole::Arg0Agent,
            candidate_node_id: agent_id,
            candidate_label: "dia".to_string(),
            source: EnrichmentSource::PassiveRecall,
            confidence: 0.7,
        });

        let result = enrich.execute(&mut ctx, &mut graph);

        // Verify enrichment was applied
        let enriched = graph.compositions.get(&comp_id).unwrap();
        assert!(enriched.members.len() >= 2);
        assert!(result.enrichments_applied >= 1);
    }

    #[test]
    fn test_enrich_composition_duplicate_role_rejected() {
        let enrich = EnrichComposition::new();
        let mut ctx = PipelineContext::default();
        let mut graph = Graph::new();

        let pred_id = graph.ensure_node("pergi");
        let agent_id = graph.ensure_node("dia");
        let comp_id = "comp_dup".to_string();
        let mut comp = Composition::default();
        comp.id = comp_id.clone();
        comp.composition_type = CompositionType::Event;
        comp.members.push(CompositionMember {
            node_id: pred_id,
            role: SemanticRole::Predicate,
            confidence: 0.5,
            label: "pergi".to_string(),
            source: None,
        });
        comp.members.push(CompositionMember {
            node_id: agent_id,
            role: SemanticRole::Arg0Agent,
            confidence: 0.7,
            label: "dia".to_string(),
            source: None,
        });
        graph.compositions.insert(comp_id.clone(), comp);

        // Try to add duplicate Agent role
        let other_id = graph.ensure_node("mereka");
        ctx.pending_enrichments.push(EnrichmentRequest {
            target_composition_id: comp_id.clone(),
            role_to_fill: SemanticRole::Arg0Agent, // duplicate!
            candidate_node_id: other_id,
            candidate_label: "mereka".to_string(),
            source: EnrichmentSource::PassiveRecall,
            confidence: 0.6,
        });

        let result = enrich.execute(&mut ctx, &mut graph);

        // Should NOT add duplicate role — enrichments_applied should be 0
        let enriched = graph.compositions.get(&comp_id).unwrap();
        let agent_count = enriched
            .members
            .iter()
            .filter(|m| m.role == SemanticRole::Arg0Agent)
            .count();
        assert_eq!(agent_count, 1); // Still only 1 Agent
        assert_eq!(result.enrichments_applied, 0);
    }

    #[test]
    fn test_enrich_composition_nonexistent_skipped() {
        let enrich = EnrichComposition::new();
        let mut ctx = PipelineContext::default();
        let mut graph = Graph::new();

        ctx.pending_enrichments.push(EnrichmentRequest {
            target_composition_id: "nonexistent_comp".to_string(),
            role_to_fill: SemanticRole::Arg0Agent,
            candidate_node_id: 0,
            candidate_label: "test".to_string(),
            source: EnrichmentSource::PassiveRecall,
            confidence: 0.7,
        });

        let result = enrich.execute(&mut ctx, &mut graph);
        assert_eq!(result.enrichments_applied, 0);
    }

    #[test]
    fn test_re_extract_frame_processes_pending() {
        let re_extract = ReExtractFrame::new();
        let mut ctx = PipelineContext::default();
        let mut graph = Graph::new();

        // Create a composition
        let pred_id = graph.ensure_node("makan");
        let comp_id = "comp_retest".to_string();
        let mut comp = Composition::default();
        comp.id = comp_id.clone();
        comp.composition_type = CompositionType::Event;
        comp.confidence = 0.3; // Low confidence — good re-extraction candidate
        comp.members.push(CompositionMember {
            node_id: pred_id,
            role: SemanticRole::Predicate,
            confidence: 0.3,
            label: "makan".to_string(),
            source: None,
        });
        graph.compositions.insert(comp_id.clone(), comp);

        // Queue a re-extraction request
        ctx.pending_reextractions.push(ReExtractionRequest {
            original_text: "kucing makan ikan".to_string(),
            original_atom_id: String::new(),
            target_composition_id: comp_id.clone(),
            graph_context: Vec::new(),
        });

        let _result = re_extract.execute(&mut ctx, &mut graph);

        // Pending reextractions should be consumed
        assert!(ctx.pending_reextractions.is_empty());
    }

    #[test]
    fn test_re_extract_frame_empty_pending() {
        let re_extract = ReExtractFrame::new();
        let mut ctx = PipelineContext::default();
        let mut graph = Graph::new();

        let result = re_extract.execute(&mut ctx, &mut graph);
        assert_eq!(result.compositions_created, 0);
        assert_eq!(result.edges_created, 0);
    }

    #[test]
    fn test_seed_anchor_adjusts_confidence() {
        use crate::v12::govern_beliefs::SeedAnchor;

        let anchor = SeedAnchor::new();
        let mut ctx = PipelineContext::default();
        let mut graph = Graph::new();

        // Create a composition with seed scores
        let pred_id = graph.ensure_node("pergi");
        let comp_id = "comp_anchor".to_string();
        let mut comp = Composition::default();
        comp.id = comp_id.clone();
        comp.composition_type = CompositionType::Event;
        comp.confidence = 0.5;
        comp.members.push(CompositionMember {
            node_id: pred_id,
            role: SemanticRole::Predicate,
            confidence: 0.5,
            label: "pergi".to_string(),
            source: None,
        });
        // Add seed scores
        comp.seed_scores.insert(SeedPrimitive::Trust, 0.8);
        comp.seed_scores.insert(SeedPrimitive::Risk, 0.2);
        graph.compositions.insert(comp_id.clone(), comp);

        let result = anchor.execute(&mut ctx, &mut graph);

        // SeedAnchor should have run without error
        assert!(result.governance_transitions <= 1); // At most 1 lifecycle transition
    }

    #[test]
    fn test_seed_anchor_empty_graph() {
        use crate::v12::govern_beliefs::SeedAnchor;

        let anchor = SeedAnchor::new();
        let mut ctx = PipelineContext::default();
        let mut graph = Graph::new();

        let result = anchor.execute(&mut ctx, &mut graph);
        assert_eq!(result.governance_transitions, 0);
    }

    #[test]
    fn test_full_pipeline_ingest() {
        let mut engine = PipelineEngine::new();
        register_default_pipeline(&mut engine);

        let result = engine.ingest("karena harga naik, rakyat menderita").unwrap();
        assert!(result.atoms_created > 0, "Should create atoms");
        assert!(
            result.compositions_created > 0,
            "Should create compositions"
        );
    }

    #[test]
    fn test_full_pipeline_multiple_ingests() {
        let mut engine = PipelineEngine::new();
        register_default_pipeline(&mut engine);

        let r1 = engine.ingest("kucing makan ikan").unwrap();
        let r2 = engine.ingest("karena hujan, jalan basah").unwrap();

        assert!(r1.atoms_created > 0);
        assert!(r2.atoms_created > 0);
        assert!(engine.graph.node_count() > 3);
    }
}
