//! PyO3 bindings for the v12.0 module — AAM pipeline engine and types.
//!
//! This module exposes the v12.0 unified abstraction types and the DAG-based
//! pipeline engine to Python. All types are feature-gated with
//! `#[cfg(all(feature = "python", feature = "v12"))]`.
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

#![cfg(all(feature = "python", feature = "v12"))]

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
    /// Confidence score (0.0-1.0) for this atom's extraction quality.
    pub confidence: f32,
    /// Provenance: where this atom came from (EdgeSource as string).
    pub source: String,
}

#[pymethods]
impl PySemanticAtom {
    fn __repr__(&self) -> String {
        format!(
            "SemanticAtom(id='{}', label='{}', type='{}', confidence={:.2}, roles={})",
            self.id,
            self.label,
            self.atom_type,
            self.confidence,
            self.roles.len()
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
            confidence: atom.confidence,
            source: format!("{:?}", atom.source),
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
    /// Source text that produced this composition, if available.
    pub source_text: Option<String>,
    /// How many ingest batches this composition has survived.
    pub batch_seen: usize,
    /// ISO 8601 timestamp when this composition was created.
    pub created_at: String,
}

#[pymethods]
impl PyComposition {
    fn __repr__(&self) -> String {
        format!(
            "Composition(id='{}', type='{}', lifecycle='{}', epistemic='{}', confidence={:.2}, members={})",
            self.id,
            self.composition_type,
            self.lifecycle,
            self.epistemic,
            self.confidence,
            self.members.len()
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
            source_text: comp.source_text.clone(),
            batch_seen: comp.batch_seen,
            created_at: comp.created_at.clone(),
        }
    }
}

// ========================================================================
// PyKnowledgeGap — Python wrapper for v12::KnowledgeGap
// ========================================================================

/// Python wrapper for `v12::KnowledgeGap` — a detected knowledge gap.
///
/// Represents something that the system doesn't know but could potentially
/// learn. Each gap has a type, a source composition, and a description
/// of what's missing.
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
///
/// Maps each gap to a specific acquisition strategy with expected
/// confidence improvement.
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
            confidence_before: 0.0, // Not tracked in current AcquisitionDecision
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
///
/// Provides a summary of what the pipeline accomplished in a single run:
/// how many atoms and compositions were created, how many gaps were
/// detected, and which cognitive mode was selected.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyV12IngestResult {
    /// Number of new atoms created during this pipeline run.
    pub atoms_created: usize,
    /// Number of new compositions created.
    pub compositions_created: usize,
    /// Number of knowledge gaps detected.
    pub gaps_detected: usize,
    /// Cognitive mode selected for this input.
    pub cognitive_mode: String,
}

#[pymethods]
impl PyV12IngestResult {
    fn __repr__(&self) -> String {
        format!(
            "V12IngestResult(atoms={}, compositions={}, gaps={}, mode='{}')",
            self.atoms_created,
            self.compositions_created,
            self.gaps_detected,
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
        })
    }

    /// Ingest text using the v12 pipeline (DAG-based, with ExtractFrame,
    /// ReasonFrame, GovernBeliefs, DetectGaps, etc.).
    ///
    /// Returns a `PyV12IngestResult` with summary statistics.
    fn v12_ingest(&mut self, text: &str) -> PyResult<PyV12IngestResult> {
        // Select cognitive mode before ingest.
        let snapshot = self.engine.snapshot();
        let mut orchestrator = v12::ExecutiveOrchestrator::new();
        let mode = orchestrator.select_cognitive_mode(text, &snapshot.compositions);

        // Run the pipeline.
        let result = self.engine.ingest(text);

        Ok(PyV12IngestResult {
            atoms_created: result.atoms_created,
            compositions_created: result.compositions_created,
            gaps_detected: result.gaps_detected,
            cognitive_mode: mode.name().to_string(),
        })
    }

    /// Select cognitive mode for the given input text.
    ///
    /// Returns one of: "Reactive", "Analytical", "Reflective".
    /// - Reactive: no contradictions, no gaps (fast path)
    /// - Analytical: contradictions or low confidence (enrichment loop)
    /// - Reflective: deep contradictions (extended reflection)
    fn select_cognitive_mode(&self, text: &str) -> String {
        let snapshot = self.engine.snapshot();
        let mut orchestrator = v12::ExecutiveOrchestrator::new();
        let mode = orchestrator.select_cognitive_mode(text, &snapshot.compositions);
        mode.name().to_string()
    }

    /// Get all compositions in the v12 graph.
    ///
    /// Returns a list of `PyComposition` objects representing the
    /// current state of the v12 knowledge graph.
    fn compositions(&self) -> Vec<PyComposition> {
        self.engine
            .graph()
            .compositions()
            .map(PyComposition::from)
            .collect()
    }

    /// Detect gaps in the current graph state.
    ///
    /// Runs the `DetectGaps` transform on a snapshot of the current
    /// graph and returns a list of `PyKnowledgeGap` objects.
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
    ///
    /// Returns `None` if no composition with the given ID exists.
    fn get_composition(&self, id: &str) -> Option<PyComposition> {
        self.engine
            .graph()
            .get_composition(&id.to_string())
            .map(PyComposition::from)
    }

    /// Find weak frames — low-confidence Event compositions missing
    /// expected roles.
    ///
    /// Returns a list of composition IDs that may benefit from
    /// re-extraction with graph context.
    fn find_weak_frames(&self) -> Vec<String> {
        self.engine
            .find_weak_frames()
            .iter()
            .map(|wf| wf.composition_id.clone())
            .collect()
    }

    /// Get a JSON snapshot of the current graph state.
    ///
    /// Useful for debugging or serialization. Returns a JSON string
    /// containing all compositions and their members.
    fn snapshot_json(&self) -> String {
        let snapshot = self.engine.snapshot();
        serde_json::to_string(&snapshot).unwrap_or_else(|_| "{}".to_string())
    }

    /// Enable or disable gap detection for subsequent ingest calls.
    ///
    /// When enabled, the DetectGaps transform will run after SeedAnchor
    /// and detect missing roles, ambiguous tokens, and other knowledge gaps.
    fn set_gap_detection(&mut self, enabled: bool) {
        self.engine.context.gap_detection_enabled = enabled;
    }

    /// Check whether gap detection is currently enabled.
    fn gap_detection_enabled(&self) -> bool {
        self.engine.context.gap_detection_enabled
    }
}
