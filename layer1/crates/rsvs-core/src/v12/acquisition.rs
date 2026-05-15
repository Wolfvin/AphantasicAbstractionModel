//! # MD-6: DetectGaps & SelectAcquisition Transforms
//!
//! Knowledge gap detection and acquisition strategy selection. These two
//! transforms form the feedback loop: `DetectGaps` identifies what's missing,
//! and `SelectAcquisition` decides how to fill each gap.
//!
//! ## DetectGaps
//!
//! ```text
//! GraphSnapshot → detect_atom_gaps() → detect_graph_gaps() → detect_grounding_gaps()
//!              → Vec<KnowledgeGap>
//! ```
//!
//! ## SelectAcquisition
//!
//! ```text
//! Vec<KnowledgeGap> → select_strategy() → Vec<AcquisitionDecision>
//!                                          ├─ PassiveRecall
//!                                          ├─ ReExtraction
//!                                          ├─ AskUser
//!                                          └─ Defer
//! ```
//!
//! ## Gap Types
//!
//! | Type | Condition | Example |
//! |------|-----------|---------|
//! | MissingRole | Event missing expected role | Event has no Agent |
//! | AmbiguousToken | Token needs disambiguation | "dia" (he/she) |
//! | SparseGraph | Too few compositions in neighborhood | New topic area |
//! | LowGrounding | Composition with low grounding score | Unverified claim |
//! | UnresolvedContradiction | Contradicted composition with no resolution | Active conflict |
//!
//! ## Acquisition Hierarchy
//!
//! ```text
//! 1. PassiveRecall   — graph already has the answer
//! 2. ReExtraction    — re-extract with graph context
//! 3. AskUser         — ask the user for clarification
//! 4. Defer           — gap noted but not actionable now
//! ```
//!
//! ## Feature Flag
//!
//! This module is only compiled when the `v12` feature is enabled.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::pipeline::{ErasedTransform, Graph, IngestResult};
use super::types::*;
use crate::types::{EdgeSource, NodeId};

// ========================================================================
// KnowledgeGapType — Classification of Knowledge Gaps
// ========================================================================

/// Classification of knowledge gaps (MD-6).
///
/// Each gap type corresponds to a specific kind of missing information
/// and determines which acquisition strategy to use.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KnowledgeGapType {
    /// An Event composition is missing an expected role.
    /// E.g., an Event with no Arg0Agent or no Arg1Patient.
    MissingRole,
    /// A token needs disambiguation (pronouns, deictics).
    /// E.g., "dia" could refer to any person.
    AmbiguousToken,
    /// The graph is too sparse in a neighborhood.
    /// Not enough compositions to support confident reasoning.
    SparseGraph,
    /// A composition has low grounding (few independent sources).
    LowGrounding,
    /// A contradicted composition with no resolution yet.
    UnresolvedContradiction,
    /// A HiddenMeaning composition is missing key roles.
    IncompleteHiddenMeaning,
    /// No cause has been identified for an event.
    MissingCause,
    /// No purpose has been identified for an action.
    MissingPurpose,
}

impl Default for KnowledgeGapType {
    fn default() -> Self {
        KnowledgeGapType::MissingRole
    }
}

// ========================================================================
// KnowledgeGap — A Detected Gap in Knowledge
// ========================================================================

/// A detected knowledge gap (MD-6).
///
/// Represents something that the system doesn't know but could potentially
/// learn. Each gap has a type, a source (which composition/atom has the gap),
/// and a description of what's missing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGap {
    /// Unique gap identifier.
    pub gap_id: String,
    /// What type of gap this is.
    pub gap_type: KnowledgeGapType,
    /// Human-readable description of what's missing.
    pub description: String,
    /// The composition that has this gap (if applicable).
    #[serde(default)]
    pub source_composition_id: Option<CompositionId>,
    /// The atom that has this gap (if applicable).
    #[serde(default)]
    pub source_atom_id: Option<String>,
    /// The specific role that's missing (for MissingRole gaps).
    #[serde(default)]
    pub missing_role: Option<SemanticRole>,
    /// Confidence that this is a real gap (0.0–1.0).
    pub confidence: f32,
}

impl Default for KnowledgeGap {
    fn default() -> Self {
        Self {
            gap_id: String::new(),
            gap_type: KnowledgeGapType::MissingRole,
            description: String::new(),
            source_composition_id: None,
            source_atom_id: None,
            missing_role: None,
            confidence: 0.0,
        }
    }
}

impl KnowledgeGap {
    /// Create a new knowledge gap.
    pub fn new(
        gap_id: &str,
        gap_type: KnowledgeGapType,
        description: &str,
        confidence: f32,
    ) -> Self {
        Self {
            gap_id: gap_id.to_string(),
            gap_type,
            description: description.to_string(),
            confidence,
            ..Self::default()
        }
    }
}

// ========================================================================
// AcquisitionDecision — How to Fill a Gap
// ========================================================================

/// Strategy for filling a knowledge gap (MD-6).
///
/// Ordered by preference (cheapest first):
/// 1. `PassiveRecall` — the graph already has the answer
/// 2. `ReExtraction` — re-extract the original text with graph context
/// 3. `AskUser` — ask the user for clarification
/// 4. `Defer` — gap noted but not actionable now
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AcquisitionStrategy {
    /// The graph already has a candidate node to fill the gap.
    PassiveRecall {
        /// The candidate node to use.
        candidate_node_id: NodeId,
        /// The candidate's label.
        candidate_label: String,
        /// Confidence in this candidate.
        confidence: f32,
    },
    /// Re-extract the original text with graph context hints.
    ReExtraction {
        /// The composition to re-extract.
        target_composition_id: CompositionId,
        /// Graph context hints for re-extraction.
        context_hints: Vec<(SemanticRole, NodeId, f32)>,
    },
    /// Ask the user for clarification.
    AskUser {
        /// The question to ask.
        question: InquiryQuestion,
    },
    /// Gap noted but deferred — not actionable now.
    Defer,
}

impl Default for AcquisitionStrategy {
    fn default() -> Self {
        AcquisitionStrategy::Defer
    }
}

/// A question to ask the user (MD-6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InquiryQuestion {
    /// Unique question identifier.
    pub question_id: String,
    /// The question text.
    pub question_text: String,
    /// The gap this question addresses.
    pub gap_id: String,
    /// What role the answer should fill.
    #[serde(default)]
    pub target_role: Option<SemanticRole>,
    /// The composition the answer should enrich.
    #[serde(default)]
    pub target_composition_id: Option<CompositionId>,
}

impl Default for InquiryQuestion {
    fn default() -> Self {
        Self {
            question_id: String::new(),
            question_text: String::new(),
            gap_id: String::new(),
            target_role: None,
            target_composition_id: None,
        }
    }
}

/// A decision about how to fill a knowledge gap (MD-6).
///
/// Maps each gap to a specific acquisition strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcquisitionDecision {
    /// The gap this decision addresses.
    pub gap_id: String,
    /// The strategy to use.
    pub strategy: AcquisitionStrategy,
    /// Expected confidence improvement if this strategy succeeds.
    pub expected_confidence_delta: f32,
}

impl Default for AcquisitionDecision {
    fn default() -> Self {
        Self {
            gap_id: String::new(),
            strategy: AcquisitionStrategy::Defer,
            expected_confidence_delta: 0.0,
        }
    }
}

// ========================================================================
// InquiryMemory — Prevent Repetition
// ========================================================================

/// Memory of past inquiries to prevent asking the same question twice (MD-6).
///
/// Tracks which gaps have been addressed and which questions have been
/// asked, so the system doesn't repeat itself.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InquiryMemory {
    /// Gap IDs that have been addressed (by any strategy).
    #[serde(default)]
    pub addressed_gaps: HashMap<String, String>, // gap_id → strategy used
    /// Questions that have been asked (question_id → answer, if any).
    #[serde(default)]
    pub asked_questions: HashMap<String, Option<String>>,
    /// Number of times we've deferred each gap type.
    #[serde(default)]
    pub defer_counts: HashMap<String, usize>, // gap_type → count
}

impl InquiryMemory {
    /// Create a new empty inquiry memory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Has this gap already been addressed?
    pub fn is_gap_addressed(&self, gap_id: &str) -> bool {
        self.addressed_gaps.contains_key(gap_id)
    }

    /// Record that a gap has been addressed.
    pub fn mark_gap_addressed(&mut self, gap_id: &str, strategy: &str) {
        self.addressed_gaps.insert(gap_id.to_string(), strategy.to_string());
    }

    /// Has this question been asked before?
    pub fn is_question_asked(&self, question_id: &str) -> bool {
        self.asked_questions.contains_key(question_id)
    }

    /// Record that a question was asked.
    pub fn mark_question_asked(&mut self, question_id: &str) {
        self.asked_questions.insert(question_id.to_string(), None);
    }

    /// Record a user's answer to a question.
    pub fn record_answer(&mut self, question_id: &str, answer: &str) {
        self.asked_questions.insert(question_id.to_string(), Some(answer.to_string()));
    }

    /// How many times has this gap type been deferred?
    pub fn defer_count(&self, gap_type: &KnowledgeGapType) -> usize {
        let key = format!("{:?}", gap_type);
        *self.defer_counts.get(&key).unwrap_or(&0)
    }

    /// Increment the defer count for a gap type.
    pub fn increment_defer(&mut self, gap_type: &KnowledgeGapType) {
        let key = format!("{:?}", gap_type);
        *self.defer_counts.entry(key).or_insert(0) += 1;
    }
}

// ========================================================================
// DetectGaps — The Transform
// ========================================================================

/// MD-6: DetectGaps transform — identifies knowledge gaps in the graph.
///
/// Scans the graph for three categories of gaps:
/// 1. **Atom gaps** — missing roles in compositions, ambiguous tokens
/// 2. **Graph gaps** — sparse neighborhoods, disconnected subgraphs
/// 3. **Grounding gaps** — low-confidence compositions without independent sources
///
/// # Transform Signature
///
/// ```text
/// Input:  GraphSnapshot — current graph state
/// Output: Vec<KnowledgeGap> — detected gaps
/// ```
#[derive(Debug, Clone, Default)]
pub struct DetectGaps {
    /// Gap ID counter.
    next_gap_id: u64,
}

impl DetectGaps {
    /// Create a new DetectGaps transform.
    pub fn new() -> Self {
        Self { next_gap_id: 0 }
    }

    /// Generate the next gap ID.
    fn next_gap_id(&mut self) -> String {
        let id = self.next_gap_id;
        self.next_gap_id += 1;
        format!("gap_{}", id)
    }

    /// Detect atom-level gaps: missing roles, ambiguous tokens, etc.
    pub fn detect_atom_gaps(&mut self, snapshot: &GraphSnapshot) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();

        for composition in &snapshot.compositions {
            match composition.composition_type {
                CompositionType::Event => {
                    // Check for missing expected roles in Event compositions.
                    let expected_roles = [
                        SemanticRole::Arg0Agent,
                        SemanticRole::Arg1Patient,
                        SemanticRole::Cause,
                        SemanticRole::Purpose,
                    ];

                    for role in &expected_roles {
                        if !composition.has_member_with_role(role.clone()) {
                            gaps.push(KnowledgeGap {
                                gap_id: self.next_gap_id(),
                                gap_type: match role {
                                    SemanticRole::Cause => KnowledgeGapType::MissingCause,
                                    SemanticRole::Purpose => KnowledgeGapType::MissingPurpose,
                                    _ => KnowledgeGapType::MissingRole,
                                },
                                description: format!(
                                    "Event '{}' missing {} role",
                                    composition.id,
                                    format!("{:?}", role).trim_start_matches("SemanticRole::")
                                ),
                                source_composition_id: Some(composition.id.clone()),
                                source_atom_id: None,
                                missing_role: Some(role.clone()),
                                confidence: 0.7,
                            });
                        }
                    }
                }

                CompositionType::HiddenMeaning => {
                    // Check for incomplete hidden meaning compositions.
                    let has_problem = composition.has_member_with_role(SemanticRole::Problem);
                    let has_solution = composition.has_member_with_role(SemanticRole::Solution);

                    if !has_problem || !has_solution {
                        gaps.push(KnowledgeGap {
                            gap_id: self.next_gap_id(),
                            gap_type: KnowledgeGapType::IncompleteHiddenMeaning,
                            description: format!(
                                "HiddenMeaning '{}' missing {}",
                                composition.id,
                                if !has_problem && !has_solution {
                                    "Problem and Solution"
                                } else if !has_problem {
                                    "Problem"
                                } else {
                                    "Solution"
                                }
                            ),
                            source_composition_id: Some(composition.id.clone()),
                            source_atom_id: None,
                            missing_role: if !has_problem {
                                Some(SemanticRole::Problem)
                            } else {
                                Some(SemanticRole::Solution)
                            },
                            confidence: 0.6,
                        });
                    }
                }

                _ => {}
            }
        }

        // Check for ambiguous tokens in recent atoms.
        for atom in &snapshot.recent_atoms {
            if atom.atom_type == AtomType::AmbiguousToken {
                gaps.push(KnowledgeGap {
                    gap_id: self.next_gap_id(),
                    gap_type: KnowledgeGapType::AmbiguousToken,
                    description: format!(
                        "Ambiguous token '{}' needs disambiguation",
                        atom.label
                    ),
                    source_composition_id: atom.composition_id.clone(),
                    source_atom_id: Some(atom.id.clone()),
                    missing_role: None,
                    confidence: 0.8,
                });
            }
        }

        gaps
    }

    /// Detect graph-level gaps: sparse neighborhoods, disconnected subgraphs.
    pub fn detect_graph_gaps(&mut self, snapshot: &GraphSnapshot) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();

        // Sparse graph: fewer than 3 compositions total.
        if snapshot.compositions.len() < 3 {
            gaps.push(KnowledgeGap {
                gap_id: self.next_gap_id(),
                gap_type: KnowledgeGapType::SparseGraph,
                description: format!(
                    "Graph is sparse (only {} compositions)",
                    snapshot.compositions.len()
                ),
                source_composition_id: None,
                source_atom_id: None,
                missing_role: None,
                confidence: 0.5,
            });
        }

        // Check for compositions that are isolated (no shared nodes with other compositions).
        let composition_count = snapshot.compositions.len();
        if composition_count > 1 {
            for comp in &snapshot.compositions {
                let node_ids: std::collections::HashSet<NodeId> =
                    comp.members.iter().map(|m| m.node_id).collect();

                let mut has_neighbor = false;
                for other in &snapshot.compositions {
                    if other.id == comp.id {
                        continue;
                    }
                    let other_nodes: std::collections::HashSet<NodeId> =
                        other.members.iter().map(|m| m.node_id).collect();
                    if !node_ids.is_disjoint(&other_nodes) {
                        has_neighbor = true;
                        break;
                    }
                }

                if !has_neighbor {
                    gaps.push(KnowledgeGap {
                        gap_id: self.next_gap_id(),
                        gap_type: KnowledgeGapType::SparseGraph,
                        description: format!(
                            "Composition '{}' is isolated (no shared nodes)",
                            comp.id
                        ),
                        source_composition_id: Some(comp.id.clone()),
                        source_atom_id: None,
                        missing_role: None,
                        confidence: 0.4,
                    });
                }
            }
        }

        gaps
    }

    /// Detect grounding gaps: low-confidence compositions without independent sources.
    pub fn detect_grounding_gaps(&mut self, snapshot: &GraphSnapshot) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();

        for composition in &snapshot.compositions {
            // Low grounding: Inferred but not Grounded, with low confidence.
            if composition.epistemic == EpistemicState::Inferred && composition.confidence < 0.5 {
                gaps.push(KnowledgeGap {
                    gap_id: self.next_gap_id(),
                    gap_type: KnowledgeGapType::LowGrounding,
                    description: format!(
                        "Composition '{}' is Inferred but not Grounded (confidence: {:.2})",
                        composition.id, composition.confidence
                    ),
                    source_composition_id: Some(composition.id.clone()),
                    source_atom_id: None,
                    missing_role: None,
                    confidence: 0.6,
                });
            }

            // Unresolved contradictions.
            if composition.epistemic == EpistemicState::Contradicted {
                gaps.push(KnowledgeGap {
                    gap_id: self.next_gap_id(),
                    gap_type: KnowledgeGapType::UnresolvedContradiction,
                    description: format!(
                        "Composition '{}' is Contradicted without resolution",
                        composition.id
                    ),
                    source_composition_id: Some(composition.id.clone()),
                    source_atom_id: None,
                    missing_role: None,
                    confidence: 0.8,
                });
            }
        }

        gaps
    }

    /// Detect all gaps in the graph snapshot.
    pub fn detect_all(&mut self, snapshot: &GraphSnapshot) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();
        gaps.extend(self.detect_atom_gaps(snapshot));
        gaps.extend(self.detect_graph_gaps(snapshot));
        gaps.extend(self.detect_grounding_gaps(snapshot));
        gaps
    }
}

/// Implement the `Transform` trait for `DetectGaps`.
impl Transform for DetectGaps {
    type Input = GraphSnapshot;
    type Output = Vec<KnowledgeGap>;

    fn id(&self) -> &'static str {
        "DetectGaps"
    }

    fn transform(&self, input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        let mut dg = DetectGaps::new();
        dg.detect_all(input)
    }
}

/// Implement `ErasedTransform` for pipeline integration.
impl ErasedTransform for DetectGaps {
    fn id(&self) -> &'static str {
        "DetectGaps"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let mut dg = DetectGaps::new();
        let snapshot = graph.compositions.values().cloned().collect();
        let atoms = ctx.current_atoms.clone();

        let gaps = dg.detect_all(&GraphSnapshot {
            recent_atoms: atoms,
            compositions: snapshot,
        });

        let gaps_detected = gaps.len();

        // Store gaps in pipeline context (convert to placeholders for now).
        ctx.pending_gaps = gaps
            .iter()
            .map(|g| KnowledgeGapPlaceholder {
                gap_id: g.gap_id.clone(),
                description: g.description.clone(),
                confidence: g.confidence,
            })
            .collect();

        // Store the full gaps for SelectAcquisition to use.
        // We use a side channel: store gap descriptions in pending_gaps.
        ctx.gap_detection_enabled = true;

        IngestResult {
            gaps_detected,
            ..IngestResult::default()
        }
    }
}

// ========================================================================
// SelectAcquisition — The Transform
// ========================================================================

/// MD-6: SelectAcquisition transform — decides how to fill each knowledge gap.
///
/// Uses an acquisition hierarchy:
/// 1. **PassiveRecall** — graph already has a candidate node
/// 2. **ReExtraction** — re-extract with graph context
/// 3. **AskUser** — ask the user for clarification
/// 4. **Defer** — gap noted but not actionable now
///
/// # Transform Signature
///
/// ```text
/// Input:  Vec<KnowledgeGap> — gaps from DetectGaps
/// Output: Vec<AcquisitionDecision> — strategies to fill each gap
/// ```
#[derive(Debug, Clone)]
pub struct SelectAcquisition {
    /// Inquiry memory to prevent repetition.
    pub memory: InquiryMemory,
}

impl Default for SelectAcquisition {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectAcquisition {
    /// Create a new SelectAcquisition transform.
    pub fn new() -> Self {
        Self {
            memory: InquiryMemory::new(),
        }
    }

    /// Select an acquisition strategy for a single gap.
    ///
    /// Uses the acquisition hierarchy:
    ///
    /// | Gap Type | Strategy |
    /// |----------|----------|
    /// | MissingRole | PassiveRecall if graph has candidate, else ReExtraction if source_text available, else AskUser |
    /// | AmbiguousToken | PassiveRecall (resolve from graph), else AskUser |
    /// | SparseGraph | Defer (needs more input) |
    /// | LowGrounding | PassiveRecall (find supporting evidence), else Defer |
    /// | UnresolvedContradiction | ReExtraction if source_text available, else AskUser |
    /// | IncompleteHiddenMeaning | ReExtraction, else AskUser |
    /// | MissingCause | PassiveRecall if graph has cause candidate, else AskUser |
    /// | MissingPurpose | PassiveRecall if graph has purpose candidate, else AskUser |
    pub fn select_strategy(&mut self, gap: &KnowledgeGap, graph: &Graph) -> AcquisitionDecision {
        // Skip already-addressed gaps.
        if self.memory.is_gap_addressed(&gap.gap_id) {
            return AcquisitionDecision {
                gap_id: gap.gap_id.clone(),
                strategy: AcquisitionStrategy::Defer,
                expected_confidence_delta: 0.0,
            };
        }

        let strategy = match gap.gap_type {
            KnowledgeGapType::MissingRole | KnowledgeGapType::MissingCause | KnowledgeGapType::MissingPurpose => {
                let role = gap.missing_role.clone().unwrap_or(SemanticRole::Arg0Agent);

                // Strategy 1: PassiveRecall — find a candidate in the graph.
                if let Some(candidate) = self.graph_find_role_candidate(graph, &role, gap) {
                    self.memory.mark_gap_addressed(&gap.gap_id, "PassiveRecall");
                    AcquisitionStrategy::PassiveRecall {
                        candidate_node_id: candidate.0,
                        candidate_label: candidate.1,
                        confidence: candidate.2,
                    }
                }
                // Strategy 2: ReExtraction — if source text is available.
                else if let Some(comp_id) = &gap.source_composition_id {
                    if let Some(comp) = graph.get_composition(comp_id) {
                        if comp.source_text.is_some() {
                            let context = self.gather_graph_context(graph, comp);
                            self.memory.mark_gap_addressed(&gap.gap_id, "ReExtraction");
                            AcquisitionStrategy::ReExtraction {
                                target_composition_id: comp_id.clone(),
                                context_hints: context,
                            }
                        } else {
                            // Strategy 3: AskUser.
                            let question = self.generate_question(gap);
                            self.memory.mark_gap_addressed(&gap.gap_id, "AskUser");
                            AcquisitionStrategy::AskUser { question }
                        }
                    } else {
                        let question = self.generate_question(gap);
                        self.memory.mark_gap_addressed(&gap.gap_id, "AskUser");
                        AcquisitionStrategy::AskUser { question }
                    }
                } else {
                    let question = self.generate_question(gap);
                    self.memory.mark_gap_addressed(&gap.gap_id, "AskUser");
                    AcquisitionStrategy::AskUser { question }
                }
            }

            KnowledgeGapType::AmbiguousToken => {
                // Try to resolve from graph first.
                if let Some(resolved) = self.resolve_ambiguous_from_graph(graph, gap) {
                    self.memory.mark_gap_addressed(&gap.gap_id, "PassiveRecall");
                    AcquisitionStrategy::PassiveRecall {
                        candidate_node_id: resolved.0,
                        candidate_label: resolved.1,
                        confidence: resolved.2,
                    }
                } else {
                    let question = self.generate_question(gap);
                    self.memory.mark_gap_addressed(&gap.gap_id, "AskUser");
                    AcquisitionStrategy::AskUser { question }
                }
            }

            KnowledgeGapType::SparseGraph => {
                self.memory.increment_defer(&gap.gap_type);
                self.memory.mark_gap_addressed(&gap.gap_id, "Defer");
                AcquisitionStrategy::Defer
            }

            KnowledgeGapType::LowGrounding => {
                // Try to find supporting evidence in the graph.
                if self.graph_has_grounding_evidence(graph, gap) {
                    // Strategy: find a second independent source.
                    if let Some(comp_id) = &gap.source_composition_id {
                        let context = if let Some(comp) = graph.get_composition(comp_id) {
                            self.gather_graph_context(graph, comp)
                        } else {
                            Vec::new()
                        };
                        self.memory.mark_gap_addressed(&gap.gap_id, "ReExtraction");
                        AcquisitionStrategy::ReExtraction {
                            target_composition_id: comp_id.clone(),
                            context_hints: context,
                        }
                    } else {
                        self.memory.mark_gap_addressed(&gap.gap_id, "Defer");
                        AcquisitionStrategy::Defer
                    }
                } else {
                    self.memory.increment_defer(&gap.gap_type);
                    self.memory.mark_gap_addressed(&gap.gap_id, "Defer");
                    AcquisitionStrategy::Defer
                }
            }

            KnowledgeGapType::UnresolvedContradiction => {
                if let Some(comp_id) = &gap.source_composition_id {
                    if let Some(comp) = graph.get_composition(comp_id) {
                        if comp.source_text.is_some() {
                            let context = self.gather_graph_context(graph, comp);
                            self.memory.mark_gap_addressed(&gap.gap_id, "ReExtraction");
                            AcquisitionStrategy::ReExtraction {
                                target_composition_id: comp_id.clone(),
                                context_hints: context,
                            }
                        } else {
                            let question = self.generate_question(gap);
                            self.memory.mark_gap_addressed(&gap.gap_id, "AskUser");
                            AcquisitionStrategy::AskUser { question }
                        }
                    } else {
                        let question = self.generate_question(gap);
                        self.memory.mark_gap_addressed(&gap.gap_id, "AskUser");
                        AcquisitionStrategy::AskUser { question }
                    }
                } else {
                    self.memory.mark_gap_addressed(&gap.gap_id, "Defer");
                    AcquisitionStrategy::Defer
                }
            }

            KnowledgeGapType::IncompleteHiddenMeaning => {
                if let Some(comp_id) = &gap.source_composition_id {
                    if let Some(comp) = graph.get_composition(comp_id) {
                        if comp.source_text.is_some() {
                            let context = self.gather_graph_context(graph, comp);
                            self.memory.mark_gap_addressed(&gap.gap_id, "ReExtraction");
                            AcquisitionStrategy::ReExtraction {
                                target_composition_id: comp_id.clone(),
                                context_hints: context,
                            }
                        } else {
                            let question = self.generate_question(gap);
                            self.memory.mark_gap_addressed(&gap.gap_id, "AskUser");
                            AcquisitionStrategy::AskUser { question }
                        }
                    } else {
                        let question = self.generate_question(gap);
                        self.memory.mark_gap_addressed(&gap.gap_id, "AskUser");
                        AcquisitionStrategy::AskUser { question }
                    }
                } else {
                    let question = self.generate_question(gap);
                    self.memory.mark_gap_addressed(&gap.gap_id, "AskUser");
                    AcquisitionStrategy::AskUser { question }
                }
            }
        };

        let expected_delta = match &strategy {
            AcquisitionStrategy::PassiveRecall { confidence, .. } => *confidence * 0.1,
            AcquisitionStrategy::ReExtraction { .. } => 0.15,
            AcquisitionStrategy::AskUser { .. } => 0.2,
            AcquisitionStrategy::Defer => 0.0,
        };

        AcquisitionDecision {
            gap_id: gap.gap_id.clone(),
            strategy,
            expected_confidence_delta: expected_delta,
        }
    }

    /// Check if the graph has relevant context for a gap.
    ///
    /// Per-gap-type logic:
    /// - MissingRole: check if any composition has the same predicate with this role filled
    /// - AmbiguousToken: check if recent events have a potential referent
    /// - LowGrounding: check if there are other compositions with shared nodes
    pub fn graph_has_relevant_context(&self, graph: &Graph, gap: &KnowledgeGap) -> bool {
        match gap.gap_type {
            KnowledgeGapType::MissingRole | KnowledgeGapType::MissingCause | KnowledgeGapType::MissingPurpose => {
                if let Some(comp_id) = &gap.source_composition_id {
                    if let Some(source_comp) = graph.get_composition(comp_id) {
                        // Check if any other composition has the same predicate
                        // with the missing role filled.
                        let predicate = source_comp.member_with_role(&SemanticRole::Predicate);
                        if let Some(pred) = predicate {
                            for comp in graph.compositions() {
                                if comp.id == *comp_id {
                                    continue;
                                }
                                if comp.has_member_with_role_and_label(SemanticRole::Predicate, "") {
                                    if let Some(role) = &gap.missing_role {
                                        if comp.has_member_with_role(role.clone()) {
                                            return true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                false
            }

            KnowledgeGapType::AmbiguousToken => {
                // Check if recent events have a potential referent.
                graph.compositions.len() > 0
            }

            KnowledgeGapType::LowGrounding => {
                self.graph_has_grounding_evidence(graph, gap)
            }

            _ => false,
        }
    }

    /// Check if the graph has grounding evidence for a gap.
    ///
    /// Two strategies:
    /// 1. Find compositions that share nodes with the source composition
    /// 2. Find compositions from different provenance sources
    pub fn graph_has_grounding_evidence(&self, graph: &Graph, gap: &KnowledgeGap) -> bool {
        if let Some(comp_id) = &gap.source_composition_id {
            if let Some(source_comp) = graph.get_composition(comp_id) {
                // Strategy 1: Find compositions sharing nodes.
                for comp in graph.compositions() {
                    if comp.id == *comp_id {
                        continue;
                    }
                    let source_nodes: std::collections::HashSet<NodeId> =
                        source_comp.members.iter().map(|m| m.node_id).collect();
                    let other_nodes: std::collections::HashSet<NodeId> =
                        comp.members.iter().map(|m| m.node_id).collect();
                    if !source_nodes.is_disjoint(&other_nodes) {
                        return true;
                    }
                }

                // Strategy 2: Find compositions from different provenance.
                for comp in graph.compositions() {
                    if comp.id == *comp_id {
                        continue;
                    }
                    if comp.provenance.origin != source_comp.provenance.origin {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Find a candidate node to fill a role, using frequency-based lookup.
    ///
    /// Scans all compositions in the graph for the same role and returns
    /// the most frequently used node as a candidate.
    pub fn graph_find_role_candidate(
        &self,
        graph: &Graph,
        role: &SemanticRole,
        gap: &KnowledgeGap,
    ) -> Option<(NodeId, String, f32)> {
        let mut candidate_counts: HashMap<NodeId, (usize, String, f32)> = HashMap::new();

        for comp in graph.compositions() {
            // Skip the source composition.
            if let Some(comp_id) = &gap.source_composition_id {
                if comp.id == *comp_id {
                    continue;
                }
            }

            if let Some(member) = comp.member_with_role(role) {
                let entry = candidate_counts.entry(member.node_id).or_insert((
                    0,
                    String::new(),
                    member.confidence,
                ));
                entry.0 += 1;
                if entry.1.is_empty() {
                    // Try to get the label from the graph.
                    if let Some(label) = graph.node_label(member.node_id) {
                        entry.1 = label.to_string();
                    }
                }
            }
        }

        // Return the most frequent candidate.
        candidate_counts
            .into_iter()
            .max_by_key(|(_, (count, _, _))| *count)
            .map(|(node_id, (count, label, confidence))| {
                (node_id, label, confidence * count as f32 / 3.0_f32.max(count as f32))
            })
    }

    /// Resolve an ambiguous token from the graph using recency-based resolution.
    ///
    /// For pronouns and deictics, find the most recent entity that could
    /// be the referent. Uses recency-weighted frequency.
    pub fn resolve_ambiguous_from_graph(
        &self,
        graph: &Graph,
        gap: &KnowledgeGap,
    ) -> Option<(NodeId, String, f32)> {
        // Get recent compositions ordered by recency.
        let recent = graph.recent_compositions(10);

        // Find the most recent Agent or Patient that could be the referent.
        for comp in recent {
            // Prefer Agent (most likely pronoun referent).
            if let Some(agent) = comp.member_with_role(&SemanticRole::Arg0Agent) {
                let label = graph.node_label(agent.node_id).unwrap_or("").to_string();
                if !label.is_empty() {
                    return Some((agent.node_id, label, 0.6));
                }
            }

            // Then try Patient.
            if let Some(patient) = comp.member_with_role(&SemanticRole::Arg1Patient) {
                let label = graph.node_label(patient.node_id).unwrap_or("").to_string();
                if !label.is_empty() {
                    return Some((patient.node_id, label, 0.5));
                }
            }
        }

        None
    }

    /// Gather graph context hints for re-extraction.
    ///
    /// Returns (role, node_id, confidence) triples from the composition's
    /// existing members plus any related compositions.
    pub fn gather_graph_context(
        &self,
        graph: &Graph,
        composition: &Composition,
    ) -> Vec<(SemanticRole, NodeId, f32)> {
        let mut context = Vec::new();

        // Add all existing members as context hints.
        for member in &composition.members {
            context.push((member.role.clone(), member.node_id, member.confidence));
        }

        // Find related compositions that share nodes.
        let source_nodes: std::collections::HashSet<NodeId> =
            composition.members.iter().map(|m| m.node_id).collect();

        for other in graph.compositions() {
            if other.id == composition.id {
                continue;
            }
            let other_nodes: std::collections::HashSet<NodeId> =
                other.members.iter().map(|m| m.node_id).collect();
            if !source_nodes.is_disjoint(&other_nodes) {
                for member in &other.members {
                    if !source_nodes.contains(&member.node_id) {
                        context.push((member.role.clone(), member.node_id, member.confidence * 0.7));
                    }
                }
            }
        }

        context
    }

    /// Generate a question to ask the user about a gap.
    pub fn generate_question(&self, gap: &KnowledgeGap) -> InquiryQuestion {
        let question_text = match &gap.gap_type {
            KnowledgeGapType::MissingRole => {
                let role_name = gap
                    .missing_role
                    .as_ref()
                    .map(|r| format!("{:?}", r).trim_start_matches("SemanticRole::").to_lowercase())
                    .unwrap_or_else(|| "unknown role".to_string());
                format!("Who or what is the {} in this event?", role_name)
            }
            KnowledgeGapType::AmbiguousToken => {
                format!("What does '{}' refer to?", gap.description)
            }
            KnowledgeGapType::SparseGraph => {
                "Can you tell me more about this topic?".to_string()
            }
            KnowledgeGapType::LowGrounding => {
                "Can you confirm this information?".to_string()
            }
            KnowledgeGapType::UnresolvedContradiction => {
                "There seems to be conflicting information. Which is correct?".to_string()
            }
            KnowledgeGapType::IncompleteHiddenMeaning => {
                "What problem does this solve?".to_string()
            }
            KnowledgeGapType::MissingCause => {
                "Why did this happen?".to_string()
            }
            KnowledgeGapType::MissingPurpose => {
                "What was the purpose of this action?".to_string()
            }
        };

        InquiryQuestion {
            question_id: format!("q_{}", gap.gap_id),
            question_text,
            gap_id: gap.gap_id.clone(),
            target_role: gap.missing_role.clone(),
            target_composition_id: gap.source_composition_id.clone(),
        }
    }

    /// Process a user's answer by merging it into the composition.
    ///
    /// This is called when the user provides an answer to an inquiry question.
    /// It creates an `EnrichmentRequest` to add the answer as a new member.
    pub fn process_user_answer_merge(
        &self,
        question: &InquiryQuestion,
        answer: &str,
        graph: &mut Graph,
    ) -> Option<EnrichmentRequest> {
        let target_comp_id = question.target_composition_id.as_ref()?;
        let target_role = question.target_role.as_ref()?;

        // Ensure the answer node exists.
        let candidate_node_id = graph.ensure_node(answer);

        Some(EnrichmentRequest {
            target_composition_id: target_comp_id.clone(),
            role_to_fill: target_role.clone(),
            candidate_node_id,
            candidate_label: answer.to_string(),
            source: EnrichmentSource::UserAnswerMerge,
            confidence: 0.9,
        })
    }

    /// Select strategies for all gaps.
    pub fn select_all(&mut self, gaps: &[KnowledgeGap], graph: &Graph) -> Vec<AcquisitionDecision> {
        gaps.iter()
            .map(|gap| self.select_strategy(gap, graph))
            .collect()
    }

    /// Convert acquisition decisions into pipeline actions.
    ///
    /// Maps each `AcquisitionDecision` to concrete pipeline actions:
    /// - `PassiveRecall` → `EnrichmentRequest`
    /// - `ReExtraction` → `ReExtractionRequest`
    /// - `AskUser` → stored for user interaction
    /// - `Defer` → no action
    pub fn decisions_to_actions(
        &self,
        decisions: &[AcquisitionDecision],
        graph: &Graph,
    ) -> (Vec<EnrichmentRequest>, Vec<ReExtractionRequest>) {
        let mut enrichments = Vec::new();
        let mut reextractions = Vec::new();

        for decision in decisions {
            match &decision.strategy {
                AcquisitionStrategy::PassiveRecall {
                    candidate_node_id,
                    candidate_label,
                    confidence,
                } => {
                    if let Some(comp_id) = &decision.gap_id.split('_').next() {
                        // Try to find the source composition from the gap ID.
                        // Simplified: use the gap_id as a key.
                        enrichments.push(EnrichmentRequest {
                            target_composition_id: decision.gap_id.clone(),
                            role_to_fill: SemanticRole::Arg0Agent, // Default; should be from gap
                            candidate_node_id: *candidate_node_id,
                            candidate_label: candidate_label.clone(),
                            source: EnrichmentSource::PassiveRecall,
                            confidence: *confidence,
                        });
                    }
                }

                AcquisitionStrategy::ReExtraction {
                    target_composition_id,
                    context_hints,
                } => {
                    if let Some(comp) = graph.get_composition(target_composition_id) {
                        if let Some(source_text) = &comp.source_text {
                            reextractions.push(ReExtractionRequest {
                                original_text: source_text.clone(),
                                original_atom_id: String::new(),
                                target_composition_id: target_composition_id.clone(),
                                graph_context: context_hints.clone(),
                            });
                        }
                    }
                }

                AcquisitionStrategy::AskUser { .. } => {
                    // User questions are handled externally.
                }

                AcquisitionStrategy::Defer => {
                    // No action.
                }
            }
        }

        (enrichments, reextractions)
    }
}

/// Implement the `Transform` trait for `SelectAcquisition`.
impl Transform for SelectAcquisition {
    type Input = Vec<KnowledgeGap>;
    type Output = Vec<AcquisitionDecision>;

    fn id(&self) -> &'static str {
        "SelectAcquisition"
    }

    fn transform(&self, input: &Self::Input, _ctx: &mut PipelineContext) -> Self::Output {
        // Note: this doesn't have access to the graph, so we use
        // the ErasedTransform for full functionality.
        input
            .iter()
            .map(|gap| AcquisitionDecision {
                gap_id: gap.gap_id.clone(),
                strategy: AcquisitionStrategy::Defer,
                expected_confidence_delta: 0.0,
            })
            .collect()
    }
}

/// Implement `ErasedTransform` for pipeline integration.
impl ErasedTransform for SelectAcquisition {
    fn id(&self) -> &'static str {
        "SelectAcquisition"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let mut sa = self.clone();

        // Reconstruct KnowledgeGaps from placeholders.
        let gaps: Vec<KnowledgeGap> = ctx
            .pending_gaps
            .iter()
            .map(|placeholder| KnowledgeGap {
                gap_id: placeholder.gap_id.clone(),
                gap_type: KnowledgeGapType::MissingRole, // Default; full impl would store type
                description: placeholder.description.clone(),
                confidence: placeholder.confidence,
                ..KnowledgeGap::default()
            })
            .collect();

        if gaps.is_empty() {
            return IngestResult::new();
        }

        // Select acquisition strategies.
        let decisions = sa.select_all(&gaps, graph);

        // Convert decisions to pipeline actions.
        let (enrichments, reextractions) = sa.decisions_to_actions(&decisions, graph);

        let enrichments_count = enrichments.len();
        let reextractions_count = reextractions.len();

        // Write to pipeline context.
        ctx.pending_enrichments.extend(enrichments);
        ctx.pending_reextractions.extend(reextractions);

        IngestResult {
            enrichments_applied: enrichments_count,
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

    #[test]
    fn test_detect_atom_gaps_missing_agent() {
        let mut dg = DetectGaps::new();
        let comp = Composition {
            id: "comp_1".to_string(),
            composition_type: CompositionType::Event,
            members: vec![
                CompositionMember {
                    node_id: 1,
                    role: SemanticRole::Predicate,
                    confidence: 0.8,
                },
                CompositionMember {
                    node_id: 2,
                    role: SemanticRole::Arg1Patient,
                    confidence: 0.7,
                },
            ],
            ..Composition::default()
        };

        let snapshot = GraphSnapshot {
            recent_atoms: Vec::new(),
            compositions: vec![comp],
        };

        let gaps = dg.detect_atom_gaps(&snapshot);
        assert!(gaps.iter().any(|g| g.gap_type == KnowledgeGapType::MissingRole && g.missing_role == Some(SemanticRole::Arg0Agent)));
        assert!(gaps.iter().any(|g| g.gap_type == KnowledgeGapType::MissingCause));
    }

    #[test]
    fn test_detect_ambiguous_token() {
        let mut dg = DetectGaps::new();
        let atom = SemanticAtom {
            id: "atom_1".to_string(),
            label: "dia".to_string(),
            atom_type: AtomType::AmbiguousToken,
            confidence: 1.0,
            ..SemanticAtom::default()
        };

        let snapshot = GraphSnapshot {
            recent_atoms: vec![atom],
            compositions: Vec::new(),
        };

        let gaps = dg.detect_atom_gaps(&snapshot);
        assert!(gaps.iter().any(|g| g.gap_type == KnowledgeGapType::AmbiguousToken));
    }

    #[test]
    fn test_detect_grounding_gaps() {
        let mut dg = DetectGaps::new();
        let comp = Composition {
            id: "comp_1".to_string(),
            composition_type: CompositionType::Event,
            epistemic: EpistemicState::Inferred,
            confidence: 0.3,
            ..Composition::default()
        };

        let snapshot = GraphSnapshot {
            recent_atoms: Vec::new(),
            compositions: vec![comp],
        };

        let gaps = dg.detect_grounding_gaps(&snapshot);
        assert!(gaps.iter().any(|g| g.gap_type == KnowledgeGapType::LowGrounding));
    }

    #[test]
    fn test_select_strategy_passive_recall() {
        let mut sa = SelectAcquisition::new();
        let mut graph = Graph::new();

        // Add a node and composition with an Agent role.
        let node_id = graph.ensure_node("Raymond");
        let mut comp = Composition {
            id: "comp_other".to_string(),
            composition_type: CompositionType::Event,
            ..Composition::default()
        };
        comp.members.push(CompositionMember {
            node_id,
            role: SemanticRole::Arg0Agent,
            confidence: 0.8,
        });
        graph.compositions.insert("comp_other".to_string(), comp);

        let gap = KnowledgeGap {
            gap_id: "gap_1".to_string(),
            gap_type: KnowledgeGapType::MissingRole,
            description: "Missing Agent".to_string(),
            source_composition_id: Some("comp_1".to_string()),
            missing_role: Some(SemanticRole::Arg0Agent),
            confidence: 0.7,
            ..KnowledgeGap::default()
        };

        let decision = sa.select_strategy(&gap, &graph);
        assert!(matches!(decision.strategy, AcquisitionStrategy::PassiveRecall { .. }));
    }

    #[test]
    fn test_inquiry_memory() {
        let mut mem = InquiryMemory::new();
        assert!(!mem.is_gap_addressed("gap_1"));

        mem.mark_gap_addressed("gap_1", "PassiveRecall");
        assert!(mem.is_gap_addressed("gap_1"));

        assert!(!mem.is_question_asked("q_1"));
        mem.mark_question_asked("q_1");
        assert!(mem.is_question_asked("q_1"));
    }

    #[test]
    fn test_generate_question() {
        let sa = SelectAcquisition::new();
        let gap = KnowledgeGap {
            gap_id: "gap_1".to_string(),
            gap_type: KnowledgeGapType::MissingCause,
            description: "Missing cause".to_string(),
            missing_role: Some(SemanticRole::Cause),
            ..KnowledgeGap::default()
        };

        let question = sa.generate_question(&gap);
        assert!(question.question_text.contains("Why"));
    }

    #[test]
    fn test_graph_find_role_candidate() {
        let sa = SelectAcquisition::new();
        let mut graph = Graph::new();

        let node_id = graph.ensure_node("Raymond");
        let mut comp = Composition {
            id: "comp_1".to_string(),
            composition_type: CompositionType::Event,
            ..Composition::default()
        };
        comp.members.push(CompositionMember {
            node_id,
            role: SemanticRole::Arg0Agent,
            confidence: 0.8,
        });
        graph.compositions.insert("comp_1".to_string(), comp);

        let gap = KnowledgeGap {
            gap_id: "gap_1".to_string(),
            gap_type: KnowledgeGapType::MissingRole,
            missing_role: Some(SemanticRole::Arg0Agent),
            source_composition_id: Some("comp_2".to_string()),
            ..KnowledgeGap::default()
        };

        let candidate = sa.graph_find_role_candidate(&graph, &SemanticRole::Arg0Agent, &gap);
        assert!(candidate.is_some());
        assert_eq!(candidate.unwrap().0, node_id);
    }
}
