//! # v12.0 Transform Pipeline Engine
//!
//! This module implements the DAG-based transform pipeline that drives all v12.0
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
//! - **Condition-gated execution**: Each [`TransformNode`] can have an optional
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

use super::types::*;
use super::extract_frame::ExtractFrame;
use super::reason_frame::ReasonFrame;
use super::govern_beliefs::{GovernBeliefs, SeedAnchor};
use super::acquisition::{DetectGaps, SelectAcquisition};
use crate::types::NodeId;

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
    condition: Option<Box<dyn Fn(&PipelineContext) -> bool + Send + Sync>>,
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

/// The central pipeline engine for v12.0 (MD-3 §5).
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
/// let result = engine.ingest("Raymond membuat aplikasi karena lambat");
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
        condition: Option<Box<dyn Fn(&PipelineContext) -> bool + Send + Sync>>,
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
    pub fn execute_dag(&mut self, initial_input: &str) -> IngestResult {
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
                // In production, this would log an error. For now, return empty.
                eprintln!(
                    "[PipelineEngine] Cycle detected in transform DAG: {:?}",
                    cycle
                );
                return IngestResult::new();
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
            let deps_met = node
                .dependencies
                .iter()
                .all(|dep| completed.contains(dep));
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

        result
    }

    /// Convenience method — identical to [`execute_dag`](Self::execute_dag).
    ///
    /// Provided for ergonomic API: `engine.ingest("some text")`.
    pub fn ingest(&mut self, text: &str) -> IngestResult {
        self.execute_dag(text)
    }

    /// Apply an [`AnchoredDelta`] to the graph.
    ///
    /// This is the post-pipeline step: after the full DAG has executed and
    /// produced an `AnchoredDelta` (from `SeedAnchor`), apply it to the
    /// graph by upserting all compositions.
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

/// Register all core v12.0 transforms in dependency order.
///
/// This wires up the complete default pipeline with 10 transforms:
///
/// | # | Transform | Dependencies | Condition |
/// |---|-----------|-------------|-----------|
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
///
/// Note: ExtractFrame, ReasonFrame, GovernBeliefs, SeedAnchor, DetectGaps,
/// and SelectAcquisition are registered as no-op stubs. They will be
/// implemented in separate modules and registered here via their
/// `ErasedTransform` implementations when available.
pub fn register_default_pipeline(engine: &mut PipelineEngine) {
    // 1. Tokenize — no dependencies, always runs.
    engine.register(
        Tokenize::new(),
        vec![],
        None,
    );

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
    engine.register(
        GovernBeliefs::new(),
        vec!["IngestAtoms".to_string()],
        None,
    );

    // 6. SeedAnchor — depends on GovernBeliefs, always runs.
    engine.register(
        SeedAnchor::new(),
        vec!["GovernBeliefs".to_string()],
        None,
    );

    // 7. DetectGaps — depends on SeedAnchor, condition: gap_detection_enabled.
    engine.register(
        DetectGaps::new(),
        vec!["SeedAnchor".to_string()],
        Some(Box::new(|ctx: &PipelineContext| ctx.gap_detection_enabled())),
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
        Some(Box::new(|ctx: &PipelineContext| ctx.has_enrichment_requests())),
    );

    // 10. ReExtractFrame — depends on SelectAcquisition, condition: has_reextraction_requests.
    engine.register(
        ReExtractFrame::new(),
        vec!["SelectAcquisition".to_string()],
        Some(Box::new(|ctx: &PipelineContext| ctx.has_reextraction_requests())),
    );
}

// ========================================================================
// Graph — Minimal v12 Composition Graph
// ========================================================================

/// Minimal v12 graph that stores Compositions (not just Nodes).
///
/// Unlike the v8.3 `RsvsGraph` which stores only nodes and edges, the v12
/// graph additionally stores `Composition`s — structured groupings of nodes
/// with typed roles, lifecycle/epistemic states, and seed alignment scores.
///
/// # Storage Model
///
/// ```text
/// nodes:         HashMap<NodeId, Node>          — v8.3 nodes (reused)
/// compositions:  HashMap<CompositionId, Composition> — v12.0 compositions
/// edges:         Vec<(CompositionId, NodeId, SemanticEdge)> — v12.0 typed edges
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
    /// Nodes — reused from v8.3 `Node` type.
    pub nodes: HashMap<NodeId, crate::types::Node>,

    /// Compositions — v12.0 structured groupings.
    pub compositions: HashMap<CompositionId, Composition>,

    /// Edges — (composition_id, target_node_id, semantic_edge) triples.
    /// Each edge links a composition to one of its member nodes.
    pub edges: Vec<(CompositionId, NodeId, SemanticEdge)>,

    /// Label-to-node-id index for fast label lookups.
    pub label_to_id: HashMap<String, NodeId>,

    /// Next available node ID (auto-incrementing).
    pub next_id: NodeId,
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

        let mut node = crate::types::Node::default();
        node.id = id;
        node.label = label.to_string();
        node.surface_label = label.to_string();

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
    pub fn get_node(&self, id: NodeId) -> Option<&crate::types::Node> {
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
}

// ========================================================================
// Placeholder Transform: Tokenize
// ========================================================================

/// Tokenize transform — splits raw text into `SemanticAtom` tokens.
///
/// This is a minimal stub implementation. The full implementation will:
/// - Split input text into tokens (whitespace + punctuation)
/// - Create a `SemanticAtom` of type `AtomType::Token` for each token
/// - Append atoms to `ctx.current_atoms`
///
/// # Transform Signature
///
/// ```text
/// Input:  RawText (str) — read from ctx.raw_text
/// Output: Vec<SemanticAtom> — written to ctx.current_atoms
/// ```
pub struct Tokenize {
    /// Placeholder — future: tokenizer configuration.
    _config: (),
}

impl Tokenize {
    /// Create a new Tokenize transform.
    pub fn new() -> Self {
        Self { _config: () }
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

        // Simple whitespace tokenization.
        // TODO: Replace with proper tokenizer (punctuation handling, etc.)
        for token in text.split_whitespace() {
            let atom_id = format!("atom_{}", ctx.next_atom_id());
            let atom = SemanticAtom {
                id: atom_id,
                label: token.to_lowercase(),
                atom_type: AtomType::Token,
                confidence: 1.0,
                source: crate::types::EdgeSource::Learned,
                ..SemanticAtom::default()
            };
            ctx.current_atoms.push(atom);
            atoms_created += 1;
        }

        IngestResult {
            atoms_created,
            ..IngestResult::default()
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
        let mut atoms = Vec::new();
        for token in input.split_whitespace() {
            let atom_id = format!("atom_{}", ctx.next_atom_id());
            atoms.push(SemanticAtom {
                id: atom_id,
                label: token.to_lowercase(),
                atom_type: AtomType::Token,
                confidence: 1.0,
                source: crate::types::EdgeSource::Learned,
                ..SemanticAtom::default()
            });
        }
        atoms
    }
}

// ========================================================================
// Placeholder Transform: IngestAtoms
// ========================================================================

/// IngestAtoms transform — creates graph structures from `SemanticAtom`s.
///
/// This is a minimal stub implementation. The full implementation will:
/// - For each atom in `ctx.current_atoms`, call `graph.ensure_node()`
/// - Create `Composition`s for Event/HiddenMeaning atoms
/// - Create `SemanticEdge`s linking compositions to their member nodes
/// - Return a `GraphDelta` with the new structures
///
/// # Transform Signature
///
/// ```text
/// Input:  Vec<SemanticAtom> — read from ctx.current_atoms
/// Output: GraphDelta — applied to graph
/// ```
pub struct IngestAtoms {
    /// Placeholder — future: ingest configuration.
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

        // Ensure nodes exist for all atom labels.
        let atom_count = ctx.current_atoms.len();
        for i in 0..atom_count {
            let atom = &ctx.current_atoms[i];
            let node_id = graph.ensure_node(&atom.label);
            atoms_counted += 1;

            // For Event atoms, create a Composition.
            if atom.atom_type == AtomType::Event {
                let comp_id = format!("comp_{}", atom.id);
                let mut composition = Composition::default();
                composition.id = comp_id.clone();
                composition.composition_type = CompositionType::Event;
                composition.confidence = atom.confidence;
                composition.source_text = ctx.raw_text.clone();

                // Add the predicate as a member.
                composition.members.push(CompositionMember {
                    node_id,
                    role: SemanticRole::Predicate,
                    confidence: atom.confidence,
                    label: atom.label.clone(),
                });

                // Add role members.
                for (role, label) in &atom.roles {
                    let role_node_id = graph.ensure_node(label);
                    composition.members.push(CompositionMember {
                        node_id: role_node_id,
                        role: role.clone(),
                        confidence: atom.confidence * 0.9,
                        label: label.clone(),
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

                // Queue composition ID assignment.
                comp_id_assignments.push((i, comp_id.clone()));

                graph.compositions.insert(comp_id, composition);
                compositions_created += 1;

                // Queue for sliding window recording.
                event_atoms.push(atom.clone());
            }
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
            ..IngestResult::default()
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

    fn transform(&self, input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        // TODO: Full implementation that creates GraphDelta from atoms.
        let mut delta = GraphDelta::new();
        for _atom in input {
            delta.new_nodes.push(0); // Placeholder node IDs
        }
        delta
    }
}

// ========================================================================
// Placeholder Transform: EnrichComposition
// ========================================================================

/// EnrichComposition transform — adds a new member to an existing composition.
///
/// This is a minimal stub implementation. The full implementation will:
/// - For each `EnrichmentRequest` in `ctx.pending_enrichments`:
///   - Look up the target composition in the graph
///   - Add the candidate node as a new member with the specified role
///   - Re-govern the composition via `GovernBeliefs.re_govern_composition()`
///   - Create a feedback edge with `EdgeSource::EnrichmentFeedback`
///
/// # Transform Signature
///
/// ```text
/// Input:  EnrichmentRequest — read from ctx.pending_enrichments
/// Output: GraphDelta — applied to graph
/// ```
pub struct EnrichComposition {
    /// Placeholder — future: enrichment configuration.
    _config: (),
}

impl EnrichComposition {
    /// Create a new EnrichComposition transform.
    pub fn new() -> Self {
        Self { _config: () }
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

        for request in &requests {
            if let Some(composition) = graph.compositions.get_mut(&request.target_composition_id) {
                // Add the candidate as a new member.
                composition.members.push(CompositionMember {
                    node_id: request.candidate_node_id,
                    role: request.role_to_fill.clone(),
                    confidence: request.confidence,
                    label: request.candidate_label.clone(),
                });

                // Create a feedback edge.
                graph.edges.push((
                    request.target_composition_id.clone(),
                    request.candidate_node_id,
                    SemanticEdge {
                        relation: crate::types::RelationType::Categorical,
                        role: Some(request.role_to_fill.clone()),
                        source: crate::types::EdgeSource::EnrichmentFeedback,
                    },
                ));
                edges_created += 1;
                enrichments_applied += 1;
            }
        }

        IngestResult {
            enrichments_applied,
            edges_created,
            ..IngestResult::default()
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

    fn transform(&self, _input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        // TODO: Full implementation.
        GraphDelta::new()
    }
}

// ========================================================================
// Placeholder Transform: ReExtractFrame
// ========================================================================

/// ReExtractFrame transform — re-extracts a frame using graph context.
///
/// This is a minimal stub implementation. The full implementation will:
/// - For each `ReExtractionRequest` in `ctx.pending_reextractions`:
///   - Use the graph context (known role-fillers) as hints
///   - Re-run frame extraction on the original text
///   - If a better frame is found, replace the old composition
///   - Create a feedback edge with `EdgeSource::ExtractionRepair`
///
/// # Transform Signature
///
/// ```text
/// Input:  ReExtractionRequest — read from ctx.pending_reextractions
/// Output: Option<SemanticAtom> — new atom if re-extraction succeeded
/// ```
pub struct ReExtractFrame {
    /// Placeholder — future: re-extraction configuration.
    _config: (),
}

impl ReExtractFrame {
    /// Create a new ReExtractFrame transform.
    pub fn new() -> Self {
        Self { _config: () }
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

    fn execute(&self, ctx: &mut PipelineContext, _graph: &mut Graph) -> IngestResult {
        // TODO: Full implementation that re-extracts frames with graph context.
        // For now, just drain the pending re-extractions (no-op).
        let _count = ctx.pending_reextractions.len();
        ctx.pending_reextractions.clear();

        IngestResult {
            // Re-extraction didn't create new atoms/compositions yet.
            ..IngestResult::default()
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

    fn transform(&self, _input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        // TODO: Full implementation.
        None
    }
}

// ========================================================================
// NoOpTransform — Placeholder for Unimplemented Transforms
// ========================================================================

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
        assert_eq!(engine.transforms.len(), 10);
        assert_eq!(engine.dag.len(), 10);
    }

    #[test]
    fn test_ingest_simple() {
        let mut engine = PipelineEngine::new();
        register_default_pipeline(&mut engine);

        let result = engine.ingest("Raymond membuat aplikasi karena lambat");
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
            });
            comp.members.push(CompositionMember {
                node_id: b,
                role: SemanticRole::Arg1Patient,
                confidence: 1.0,
                label: String::new(),
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
        });
        comp2.members.push(CompositionMember {
            node_id: 3,
            role: SemanticRole::Arg1Patient,
            confidence: 0.8,
            label: String::new(),
        });
        comp2.members.push(CompositionMember {
            node_id: 4,
            role: SemanticRole::Cause,
            confidence: 0.8,
            label: String::new(),
        });
        engine.graph.compositions.insert(comp2.id.clone(), comp2);

        let weak = engine.find_weak_frames();
        assert_eq!(weak.len(), 1);
        assert_eq!(weak[0].composition_id, "comp_weak_1");
    }
}
