use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

use super::graph::Graph;
use super::super::types::*;

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
pub(crate) struct TransformNode {
    /// Unique identifier for this transform (e.g., "Tokenize", "GovernBeliefs").
    /// Uses `String` instead of `TypeId` for simplicity and debuggability.
    pub transform_id: String,

    /// Human-readable description of the input type this transform expects.
    pub input_type: String,

    /// Human-readable description of the output type this transform produces.
    pub output_type: String,

    /// IDs of transforms that must complete before this one can run.
    pub dependencies: Vec<String>,

    /// Optional condition that gates execution.
    /// If `None`, the transform always runs (if its dependencies are met).
    /// If `Some(predicate)`, the transform runs only if the predicate returns `true`.
    pub condition: Option<TransformCondition>,
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
    pub(crate) transforms: HashMap<String, Box<dyn ErasedTransform>>,

    /// The transform dependency DAG.
    pub(crate) dag: Vec<TransformNode>,

    /// Shared pipeline context — accessible to all transforms.
    pub context: PipelineContext,

    /// The v12 graph that stores nodes, compositions, and edges.
    pub(crate) graph: Graph,
}

impl PipelineEngine {
    /// Create a new empty pipeline engine with default [`PipelineContext`]
    /// and an empty [`Graph`].
    ///
    /// The context is initialized with bootstrap Action Schemas from
    /// RAB Phase 1, enabling schema-driven extraction for copula ("adalah"),
    /// possessive ("punya"), and other linguistic patterns.
    pub fn new() -> Self {
        let mut context = PipelineContext::default();
        context.active_schemas = super::super::action_schemas::bootstrap_schemas();
        Self {
            transforms: HashMap::new(),
            dag: Vec::new(),
            context,
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
    ///
    /// **Audit v6 (P3-12)**: Not called by any internal code or PyO3 bindings.
    /// The Python FFI uses `ingest()` and `context_and_graph_mut()` instead.
    /// Retained for potential future FFI use. If unused after 2 releases, remove.
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
            pending_questions: Vec::new(),
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

    /// Compute the average confidence of all compositions without cloning.
    ///
    /// Audit v6 fix: This avoids the expensive `snapshot()` call when only
    /// the average confidence is needed (e.g., in the enrichment loop).
    /// Returns 0.0 if there are no compositions.
    pub fn average_confidence(&self) -> f32 {
        let comps = &self.graph.compositions;
        if comps.is_empty() {
            return 0.0;
        }
        comps.values().map(|c| c.confidence).sum::<f32>() / comps.len() as f32
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
        let persistence = super::super::persistence::Persistence::new();
        persistence
            .save(&self.graph, path)
            .map_err(|e| e.to_string())
    }

    /// Load graph state from a JSON file, replacing the current graph.
    ///
    /// Returns an error string if loading fails.
    pub fn load(&mut self, path: &std::path::Path) -> Result<(), String> {
        let persistence = super::super::persistence::Persistence::new();
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
    ///
    /// **Audit v6 (P3-12)**: Not called by any internal code or PyO3 bindings.
    /// The pipeline DAG executes transforms via `execute_dag()` instead.
    /// Retained for potential future single-transform FFI calls.
    /// If unused after 2 releases, remove.
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

// Audit v6 fix: Removed `unsafe impl Send/Sync for SyncPipelineEngine`.
// The compiler auto-derives Send + Sync for Mutex<T> when T: Send.
// PipelineEngine is Send (all fields are Send-safe: HashMap, Vec, Graph, PipelineContext).
// If PipelineEngine ever gains a non-Send field, the compiler will now correctly
// reject the auto-impl instead of silently allowing UB via the old unsafe impl.

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
pub(super) fn topological_sort(dag: &[TransformNode]) -> Result<Vec<String>, Vec<String>> {
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
// NoOpTransform — Placeholder transform
// ========================================================================

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
