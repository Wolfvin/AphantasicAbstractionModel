//! PyO3 bindings for the v12.0 AAM pipeline engine and types.
//!
//! This module exposes the v12.0 unified abstraction types and the DAG-based
//! pipeline engine to Python. The old v8.3 bindings (PyRsvs) have been removed.
//!
//! ## Python Classes
//!
//! | Class | Wraps | Purpose |
//! |-------|-------|---------|
//! | `PySemanticAtom` | `SemanticAtom` | Universal ingest primitive |
//! | `PyComposition` | `Composition` | Universal structured grouping |
//! | `PyCompositionMember` | `CompositionMember` | A node in a composition |
//! | `PyKnowledgeGap` | `KnowledgeGap` | A detected knowledge gap |
//! | `PyAcquisitionDecision` | `AcquisitionDecision` | How to fill a gap |
//! | `PyInquiryQuestion` | `InquiryQuestion` | A question to ask the user |
//! | `PyV12Pipeline` | `PipelineEngine` | The main v12 pipeline engine |
//! | `PyV12IngestResult` | `IngestResult` | Result of v12 ingest |

use pyo3::prelude::*;

use crate::v12 as v12;

// ========================================================================
// PySemanticAtom — Python wrapper for v12::SemanticAtom
// ========================================================================

/// Python wrapper for `v12::SemanticAtom` — the universal ingest primitive.
///
/// Every piece of knowledge entering RSVS passes through one type:
/// `SemanticAtom`. A token, an event frame, a hidden meaning candidate
/// -- these are all atoms with varying richness.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PySemanticAtom {
    /// Unique identifier for this atom (e.g., "atom_42").
    pub id: String,
    /// Human-readable label (e.g., "membuat", "problem_solution").
    pub label: String,
    /// Classification of atom richness: "Token", "Event", "HiddenMeaning", etc.
    pub atom_type: String,
    /// Semantic role assignments as (role_name, label) pairs.
    pub roles: Vec<(String, String)>,
    /// Positive or negative polarity: "Positive", "Negative", or None.
    pub polarity: Option<String>,
    /// Active or passive voice: "Active", "Passive", or None.
    pub voice: Option<String>,
    /// Atom variant (e.g., specific sub-type), if applicable.
    pub variant: Option<String>,
    /// Confidence score (0.0-1.0) for this atom's extraction quality.
    pub confidence: f32,
    /// Provenance: where this atom came from (EdgeSource as string).
    pub source: String,
    /// ID of the composition this atom belongs to, if already assigned.
    pub composition_id: Option<String>,
}

#[pymethods]
impl PySemanticAtom {
    fn __repr__(&self) -> String {
        let comp = self
            .composition_id
            .as_ref()
            .map(|c| format!(", composition_id='{}'", c))
            .unwrap_or_default();
        format!(
            "SemanticAtom(id='{}', label='{}', type='{}', confidence={:.2}, roles={}{})",
            self.id,
            self.label,
            self.atom_type,
            self.confidence,
            self.roles.len(),
            comp
        )
    }
}

impl From<&v12::SemanticAtom> for PySemanticAtom {
    fn from(atom: &v12::SemanticAtom) -> Self {
        let roles: Vec<(String, String)> = atom
            .roles
            .iter()
            .map(|(role, label)| (format!("{:?}", role), label.clone()))
            .collect();

        PySemanticAtom {
            id: atom.id.clone(),
            label: atom.label.clone(),
            atom_type: format!("{:?}", atom.atom_type),
            roles,
            polarity: atom.polarity.as_ref().map(|p| format!("{:?}", p)),
            voice: atom.voice.as_ref().map(|v| format!("{:?}", v)),
            variant: atom.variant.as_ref().map(|v| format!("{:?}", v)),
            confidence: atom.confidence,
            source: format!("{:?}", atom.source),
            composition_id: atom.composition_id.clone(),
        }
    }
}

// ========================================================================
// PyCompositionMember — Python wrapper for v12::CompositionMember
// ========================================================================

/// Python wrapper for `v12::CompositionMember` — a node playing a role
/// in a Composition.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyCompositionMember {
    /// The node ID participating in this composition.
    pub node_id: u32,
    /// The semantic role this node plays (as string).
    pub role: String,
    /// Confidence that this node correctly fills this role (0.0-1.0).
    pub confidence: f32,
    /// Cached label for this member's node.
    pub label: String,
}

#[pymethods]
impl PyCompositionMember {
    fn __repr__(&self) -> String {
        format!(
            "CompositionMember(node_id={}, role='{}', label='{}', confidence={:.2})",
            self.node_id, self.role, self.label, self.confidence
        )
    }
}

impl From<&v12::CompositionMember> for PyCompositionMember {
    fn from(member: &v12::CompositionMember) -> Self {
        PyCompositionMember {
            node_id: member.node_id,
            role: format!("{:?}", member.role),
            confidence: member.confidence,
            label: member.label.clone(),
        }
    }
}

// ========================================================================
// PyComposition — Python wrapper for v12::Composition
// ========================================================================

/// Python wrapper for `v12::Composition` — the universal structured grouping.
///
/// When a `SemanticAtom` is ingested into the RSVS graph, it becomes a
/// `Composition`: a structured group of nodes with typed roles, lifecycle
/// state, epistemic state, and seed alignment scores.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyComposition {
    /// Unique composition identifier.
    pub id: String,
    /// What kind of composition: "Event", "HiddenMeaning", "Pattern", etc.
    pub composition_type: String,
    /// Members: which nodes participate, and in what role.
    pub members: Vec<PyCompositionMember>,
    /// Structural maturity: "New", "Candidate", "Stable", "Deprecated", "Quarantine".
    pub lifecycle: String,
    /// Epistemic confidence: "Observed", "Inferred", "Hypothesis", "Grounded", "Contradicted".
    pub epistemic: String,
    /// Overall confidence score (0.0-1.0).
    pub confidence: f32,
    /// Provenance origin as a string (e.g., "UserInput", "Inferred").
    pub provenance: String,
    /// Seed alignment scores: (seed_name, score) pairs.
    pub seed_scores: Vec<(String, f32)>,
    /// Source text that produced this composition, if available.
    pub source_text: Option<String>,
    /// How many ingest batches this composition has survived.
    pub batch_seen: usize,
    /// Contradiction conflict type, if this composition is contradicted.
    pub contradiction: Option<String>,
    /// ISO 8601 timestamp when this composition was created.
    pub created_at: String,
    /// ISO 8601 timestamp when this composition was last updated.
    pub updated_at: String,
}

#[pymethods]
impl PyComposition {
    fn __repr__(&self) -> String {
        let contra = self
            .contradiction
            .as_ref()
            .map(|c| format!(", contradiction='{}'", c))
            .unwrap_or_default();
        format!(
            "Composition(id='{}', type='{}', lifecycle='{}', epistemic='{}', confidence={:.2}, members={}, seed_scores={}{})",
            self.id,
            self.composition_type,
            self.lifecycle,
            self.epistemic,
            self.confidence,
            self.members.len(),
            self.seed_scores.len(),
            contra
        )
    }
}

impl From<&v12::Composition> for PyComposition {
    fn from(comp: &v12::Composition) -> Self {
        PyComposition {
            id: comp.id.clone(),
            composition_type: format!("{:?}", comp.composition_type),
            members: comp.members.iter().map(PyCompositionMember::from).collect(),
            lifecycle: format!("{:?}", comp.lifecycle),
            epistemic: format!("{:?}", comp.epistemic),
            confidence: comp.confidence,
            provenance: format!("{:?}", comp.provenance.origin),
            seed_scores: comp
                .seed_scores
                .iter()
                .map(|(k, v)| (format!("{:?}", k), *v))
                .collect(),
            source_text: comp.source_text.clone(),
            batch_seen: comp.batch_seen,
            contradiction: comp
                .contradiction
                .as_ref()
                .map(|c| format!("{:?}", c.conflict_type)),
            created_at: comp.created_at.clone(),
            updated_at: comp.updated_at.clone(),
        }
    }
}

// ========================================================================
// PyKnowledgeGap — Python wrapper for v12::KnowledgeGap
// ========================================================================

/// Python wrapper for `v12::KnowledgeGap` — a detected knowledge gap.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyKnowledgeGap {
    /// Unique gap identifier.
    pub gap_id: String,
    /// Gap type: "MissingRole", "AmbiguousToken", "SparseGraph", etc.
    pub gap_type: String,
    /// Human-readable description of what's missing.
    pub description: String,
    /// Confidence that this is a real gap (0.0-1.0).
    pub confidence: f32,
    /// Severity of the gap (derived from confidence: 1.0 - confidence).
    pub severity: f32,
    /// The specific role that's missing (for MissingRole gaps).
    pub missing_role: Option<String>,
    /// The composition that has this gap, if applicable.
    pub source_composition_id: Option<String>,
}

#[pymethods]
impl PyKnowledgeGap {
    fn __repr__(&self) -> String {
        format!(
            "KnowledgeGap(id='{}', type='{}', confidence={:.2}, missing_role={:?})",
            self.gap_id,
            self.gap_type,
            self.confidence,
            self.missing_role
        )
    }
}

impl From<&v12::KnowledgeGap> for PyKnowledgeGap {
    fn from(gap: &v12::KnowledgeGap) -> Self {
        PyKnowledgeGap {
            gap_id: gap.gap_id.clone(),
            gap_type: format!("{:?}", gap.gap_type),
            description: gap.description.clone(),
            confidence: gap.confidence,
            severity: 1.0 - gap.confidence,
            missing_role: gap.missing_role.as_ref().map(|r| format!("{:?}", r)),
            source_composition_id: gap.source_composition_id.clone(),
        }
    }
}

// ========================================================================
// PyAcquisitionDecision — Python wrapper for v12::AcquisitionDecision
// ========================================================================

/// Python wrapper for `v12::AcquisitionDecision` — how to fill a gap.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyAcquisitionDecision {
    /// The gap this decision addresses.
    pub gap_id: String,
    /// Acquisition mode: "PassiveRecall", "ReExtraction", "AskUser", "Defer".
    pub mode: String,
    /// Human-readable reason for this strategy choice.
    pub reason: String,
    /// Confidence before applying this strategy (estimated from gap).
    pub confidence_before: f32,
    /// Expected confidence gain if this strategy succeeds.
    pub expected_gain: f32,
}

#[pymethods]
impl PyAcquisitionDecision {
    fn __repr__(&self) -> String {
        format!(
            "AcquisitionDecision(gap='{}', mode='{}', expected_gain={:.2})",
            self.gap_id, self.mode, self.expected_gain
        )
    }
}

impl From<&v12::AcquisitionDecision> for PyAcquisitionDecision {
    fn from(decision: &v12::AcquisitionDecision) -> Self {
        let (mode, reason) = match &decision.strategy {
            v12::AcquisitionStrategy::PassiveRecall {
                candidate_label,
                confidence,
                ..
            } => (
                "PassiveRecall".to_string(),
                format!(
                    "Graph has candidate '{}' (confidence: {:.2})",
                    candidate_label, confidence
                ),
            ),
            v12::AcquisitionStrategy::ReExtraction {
                target_composition_id,
                ..
            } => (
                "ReExtraction".to_string(),
                format!(
                    "Re-extract with graph context for composition '{}'",
                    target_composition_id
                ),
            ),
            v12::AcquisitionStrategy::AskUser { question } => (
                "AskUser".to_string(),
                format!("Ask user: '{}'", question.question_text),
            ),
            v12::AcquisitionStrategy::Defer => (
                "Defer".to_string(),
                "Gap noted but not actionable now".to_string(),
            ),
        };

        PyAcquisitionDecision {
            gap_id: decision.gap_id.clone(),
            mode,
            reason,
            confidence_before: 0.0,
            expected_gain: decision.expected_confidence_delta,
        }
    }
}

// ========================================================================
// PyInquiryQuestion — Python wrapper for v12::InquiryQuestion
// ========================================================================

/// Python wrapper for `v12::InquiryQuestion` — a question to ask the user.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyInquiryQuestion {
    /// Unique question identifier.
    pub question_id: String,
    /// The gap this question addresses.
    pub gap_id: String,
    /// Question type (derived from gap type).
    pub question_type: String,
    /// The question text to present to the user.
    pub question_text: String,
    /// What shape the answer should take (derived from target role).
    pub expected_answer_shape: String,
}

#[pymethods]
impl PyInquiryQuestion {
    fn __repr__(&self) -> String {
        format!(
            "InquiryQuestion(id='{}', gap='{}', text='{}')",
            self.question_id, self.gap_id, self.question_text
        )
    }
}

impl From<&v12::InquiryQuestion> for PyInquiryQuestion {
    fn from(q: &v12::InquiryQuestion) -> Self {
        let question_type = q
            .target_role
            .as_ref()
            .map(|r| format!("{:?}", r))
            .unwrap_or_else(|| "General".to_string());

        let expected_answer_shape = q
            .target_role
            .as_ref()
            .map(|r| format!("Entity filling the {:?} role", r))
            .unwrap_or_else(|| "Any relevant information".to_string());

        PyInquiryQuestion {
            question_id: q.question_id.clone(),
            gap_id: q.gap_id.clone(),
            question_type,
            question_text: q.question_text.clone(),
            expected_answer_shape,
        }
    }
}

// ========================================================================
// PyV12IngestResult — Result of v12 ingest
// ========================================================================

/// Result of ingesting text through the v12 pipeline.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyV12IngestResult {
    /// Number of new atoms created during this pipeline run.
    pub atoms_created: usize,
    /// Number of new compositions created.
    pub compositions_created: usize,
    /// Number of knowledge gaps detected.
    pub gaps_detected: usize,
    /// Number of new edges created during this pipeline run.
    pub edges_created: usize,
    /// Number of enrichments applied to existing compositions.
    pub enrichments_applied: usize,
    /// Number of governance state transitions applied.
    pub governance_transitions: usize,
    /// Cognitive mode selected for this input.
    pub cognitive_mode: String,
}

#[pymethods]
impl PyV12IngestResult {
    fn __repr__(&self) -> String {
        format!(
            "V12IngestResult(atoms={}, compositions={}, gaps={}, edges={}, enrichments={}, governance={}, mode='{}')",
            self.atoms_created,
            self.compositions_created,
            self.gaps_detected,
            self.edges_created,
            self.enrichments_applied,
            self.governance_transitions,
            self.cognitive_mode
        )
    }
}

// ========================================================================
// PyV12Pipeline — Main class wrapping v12::PipelineEngine
// ========================================================================

/// The main v12.0 pipeline engine, exposed to Python.
///
/// Wraps `PipelineEngine` with a DAG-based transform pipeline that
/// processes text through: Tokenize, ExtractFrame, ReasonFrame,
/// IngestAtoms, GovernBeliefs, SeedAnchor, DetectGaps,
/// SelectAcquisition, EnrichComposition, ReExtractFrame.
///
/// # Usage from Python
///
/// ```python
/// from rsvs import PyV12Pipeline
///
/// pipeline = PyV12Pipeline()
/// result = pipeline.v12_ingest("Raymond membuat aplikasi karena lambat")
/// print(f"Created {result.atoms_created} atoms, {result.compositions_created} compositions")
///
/// # Inspect the graph
/// for comp in pipeline.compositions():
///     print(f"  {comp.id}: {comp.composition_type} (confidence={comp.confidence:.2f})")
///
/// # Detect gaps
/// gaps = pipeline.detect_gaps()
/// for gap in gaps:
///     print(f"  Gap: {gap.gap_type} - {gap.description}")
/// ```
#[pyclass]
pub struct PyV12Pipeline {
    engine: v12::PipelineEngine,
    orchestrator: v12::ExecutiveOrchestrator,
}

#[pymethods]
impl PyV12Pipeline {
    /// Create a new v12 pipeline engine with the default transform DAG.
    #[new]
    fn new() -> PyResult<Self> {
        let mut engine = v12::PipelineEngine::new();
        v12::register_default_pipeline(&mut engine);
        Ok(Self {
            engine,
            orchestrator: v12::ExecutiveOrchestrator::new(),
        })
    }

    /// Ingest text using the v12 pipeline (DAG-based, with ExtractFrame,
    /// ReasonFrame, GovernBeliefs, DetectGaps, etc.).
    fn v12_ingest(&mut self, text: &str) -> PyResult<PyV12IngestResult> {
        let snapshot = self.engine.snapshot();
        let mode = self.orchestrator.select_cognitive_mode(text, &snapshot.compositions);
        let result = self.engine.ingest(text);

        Ok(PyV12IngestResult {
            atoms_created: result.atoms_created,
            compositions_created: result.compositions_created,
            gaps_detected: result.gaps_detected,
            edges_created: result.edges_created,
            enrichments_applied: result.enrichments_applied,
            governance_transitions: result.governance_transitions,
            cognitive_mode: mode.name().to_string(),
        })
    }

    /// Select cognitive mode for the given input text.
    fn select_cognitive_mode(&mut self, text: &str) -> String {
        let snapshot = self.engine.snapshot();
        let mode = self.orchestrator.select_cognitive_mode(text, &snapshot.compositions);
        mode.name().to_string()
    }

    /// Get all compositions in the v12 graph.
    fn compositions(&self) -> Vec<PyComposition> {
        self.engine
            .graph()
            .compositions()
            .map(PyComposition::from)
            .collect()
    }

    /// Detect gaps in the current graph state.
    fn detect_gaps(&self) -> Vec<PyKnowledgeGap> {
        let snapshot = self.engine.snapshot();
        let mut detector = v12::DetectGaps::new();
        let gaps = detector.detect_all(&snapshot);
        gaps.iter().map(PyKnowledgeGap::from).collect()
    }

    /// Get the number of compositions in the graph.
    fn composition_count(&self) -> usize {
        self.engine.graph().compositions.len()
    }

    /// Get the number of nodes in the graph.
    fn node_count(&self) -> usize {
        self.engine.graph().nodes.len()
    }

    /// Get a specific composition by its ID.
    fn get_composition(&self, id: &str) -> Option<PyComposition> {
        self.engine
            .graph()
            .get_composition(&id.to_string())
            .map(PyComposition::from)
    }

    /// Find weak frames — low-confidence Event compositions missing expected roles.
    fn find_weak_frames(&self) -> Vec<String> {
        self.engine
            .find_weak_frames()
            .iter()
            .map(|wf| wf.composition_id.clone())
            .collect()
    }

    /// Get a JSON snapshot of the current graph state.
    fn snapshot_json(&self) -> String {
        let snapshot = self.engine.snapshot();
        serde_json::to_string(&snapshot)
            .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
    }

    /// Enable or disable gap detection for subsequent ingest calls.
    fn set_gap_detection(&mut self, enabled: bool) {
        self.engine.context.gap_detection_enabled = enabled;
    }

    /// Check whether gap detection is currently enabled.
    fn gap_detection_enabled(&self) -> bool {
        self.engine.context.gap_detection_enabled
    }

    /// Save the current graph to a file path.
    fn save(&self, path: &str) -> PyResult<()> {
        self.engine
            .save(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }

    /// Load graph from a file path (replaces current graph).
    fn load(&mut self, path: &str) -> PyResult<()> {
        self.engine
            .load(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
    }

    /// Run the active enrichment loop for the current graph state.
    ///
    /// This is the core feedback loop: DetectGaps → SelectAcquisition →
    /// EnrichComposition → GovernBeliefs (re-evaluate). Runs for max rounds
    /// determined by the current cognitive mode.
    fn run_enrichment_loop(&mut self) -> PyResult<PyV12IngestResult> {
        let result = self.orchestrator.run_enrichment_loop(&mut self.engine);
        Ok(PyV12IngestResult {
            atoms_created: 0,
            compositions_created: 0,
            gaps_detected: 0,
            edges_created: 0,
            enrichments_applied: result.evidence_count,
            governance_transitions: result.modified_compositions.len(),
            cognitive_mode: "enrichment_loop".to_string(),
        })
    }

    /// Get all pending knowledge gaps as structured objects.
    ///
    /// Returns gaps with their type, source composition, and suggested strategy.
    fn pending_gaps(&self) -> Vec<PyKnowledgeGap> {
        let snapshot = self.engine.snapshot();
        let mut detector = v12::DetectGaps::new();
        detector.detect_all(&snapshot)
            .iter()
            .map(PyKnowledgeGap::from)
            .collect()
    }

    /// Submit a user answer to fill a knowledge gap.
    ///
    /// gap_id: the gap ID from PyKnowledgeGap.gap_id
    /// answer: the user's answer text (e.g., "Raymond")
    ///
    /// Returns True if the answer was applied to a composition, False if gap not found.
    fn submit_answer(&mut self, gap_id: &str, answer: &str) -> bool {
        // Find the gap
        let snapshot = self.engine.snapshot();
        let mut detector = v12::DetectGaps::new();
        let gaps = detector.detect_all(&snapshot);

        let gap = match gaps.iter().find(|g| g.gap_id == gap_id) {
            Some(g) => g.clone(),
            None => return false,
        };

        // Generate a question for this gap and apply the answer
        let sa = v12::SelectAcquisition::new();
        let question = sa.generate_question(&gap);

        // Apply the answer
        let (ctx, graph) = self.engine.context_and_graph_mut();
        match sa.process_user_answer_merge(&question, answer, graph) {
            Some(request) => {
                ctx.pending_enrichments.push(request);
                true
            }
            None => false,
        }
    }

    /// Get a human-readable summary of the current graph state.
    fn graph_summary(&self) -> String {
        let graph = self.engine.graph();
        let stable = graph.compositions.values()
            .filter(|c| c.lifecycle == v12::LifecycleState::Stable)
            .count();
        let candidate = graph.compositions.values()
            .filter(|c| c.lifecycle == v12::LifecycleState::Candidate)
            .count();
        let contradicted = graph.compositions.values()
            .filter(|c| c.epistemic == v12::EpistemicState::Contradicted)
            .count();
        format!(
            "Graph: {} nodes, {} compositions ({} stable, {} candidate, {} contradicted)",
            graph.nodes.len(),
            graph.compositions.len(),
            stable,
            candidate,
            contradicted,
        )
    }
}

// ========================================================================
// Python Module Registration
// ========================================================================

/// Register all v12 PyO3 classes with the Python module.
///
/// The module name `_rsvs` matches `module-name = "rsvs._rsvs"` in pyproject.toml,
/// so Python imports it as `from rsvs._rsvs import PyV12Pipeline`.
#[pymodule]
fn _rsvs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyV12Pipeline>()?;
    m.add_class::<PySemanticAtom>()?;
    m.add_class::<PyComposition>()?;
    m.add_class::<PyCompositionMember>()?;
    m.add_class::<PyKnowledgeGap>()?;
    m.add_class::<PyAcquisitionDecision>()?;
    m.add_class::<PyInquiryQuestion>()?;
    m.add_class::<PyV12IngestResult>()?;
    Ok(())
}
