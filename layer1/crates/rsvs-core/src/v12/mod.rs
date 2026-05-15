//! # v12.0 — Unified Abstraction Types & Pipeline Engine
//!
//! This module contains the v12.0 type system and pipeline engine for the
//! AphantasicAbstractionModel (AAM) architecture. It is the FOUNDATION for the
//! entire v12.0 architecture as defined in the design documents (MD-1 through MD-6).
//!
//! ## The 6 Unified Abstractions (MD-3)
//!
//! | # | Abstraction | Replaces | Purpose |
//! |---|------------|----------|---------|
//! | 1 | [`SemanticAtom`] | Token, EventFrame, HiddenMeaningCandidate | Universal ingest primitive |
//! | 2 | [`Composition`] | EventFrame, HiddenMeaningCandidate, Pattern, Hypothesis | Universal structured grouping |
//! | 3 | [`LifecycleState`] + [`EpistemicState`] | NodeStatus, CandidateStatus, BeliefState, GroundingVerdict | Two orthogonal status axes |
//! | 4 | [`SemanticEdge`] | Separate RelationType, EdgeSource, SemanticRole, ProvenanceSource | Single typed triple |
//! | 5 | [`Transform`] | Hardcoded pipeline stages | Declarative transform graph |
//! | 6 | [`SeedPrimitive`] + seed_scores | Source trust weight system | Seed-driven epistemic confidence |
//!
//! ## Pipeline Engine
//!
//! The [`PipelineEngine`] executes transforms in topological order based on
//! their dependency DAG. Use [`register_default_pipeline`] to wire all 10 core
//! transforms, then call [`PipelineEngine::ingest`] to process text.
//!
//! ## MD-Specific Transforms
//!
//! | MD | Transform | Module | Purpose |
//! |----|-----------|--------|---------|
//! | MD-1 | [`ExtractFrame`] | [`extract_frame`] | Rule-based frame extraction |
//! | MD-2 | [`ReasonFrame`] | [`reason_frame`] | Pre-ingest reasoning rules |
//! | MD-4 | [`GovernBeliefs`] + [`SeedAnchor`] | [`govern_beliefs`] | Lifecycle/epistemic governance |
//! | MD-5 | [`ExecutiveOrchestrator`] | [`executive`] | Cognitive mode & reflection |
//! | MD-6 | [`DetectGaps`] + [`SelectAcquisition`] | [`acquisition`] | Gap detection & acquisition |
//!
//! ## Design Principles
//!
//! - **Additive**: v12.0 types are ADDITIVE — they don't replace existing v8.3 types.
//!   The existing `Node`, `Edge`, `CompositionRef` remain unchanged.
//! - **Feature-flagged**: This module is only compiled when the `v12` feature is enabled,
//!   ensuring safe incremental adoption.
//! - **Non-exhaustive**: Enums that may grow in future versions are marked `#[non_exhaustive]`.
//! - **Backward-compatible**: `#[serde(default)]` is used where appropriate for
//!   forward-compatible deserialization.

pub mod acquisition;
pub mod executive;
pub mod extract_frame;
pub mod govern_beliefs;
pub mod pipeline;
pub mod reason_frame;
pub mod types;

// Re-export key types for convenience.
// Users can import from `rsvs::v12::SemanticAtom` instead of `rsvs::v12::types::SemanticAtom`.
pub use types::{
    // --- Abstraction 1: SemanticAtom ---
    SemanticAtom, AtomType, Polarity, Voice, AtomVariant, FrameSource, PatternCategory,
    AcquisitionSource,
    // --- Abstraction 1b: SemanticRole ---
    SemanticRole,
    // --- Abstraction 2: Composition ---
    Composition, CompositionId, CompositionType, CompositionMember,
    // --- Abstraction 3: Two Orthogonal Status Axes ---
    LifecycleState, EpistemicState,
    // --- Abstraction 4: SemanticEdge ---
    SemanticEdge,
    // --- Abstraction 5: Transform ---
    Transform, PipelineContext,
    // --- Abstraction 6: Seed Anchoring ---
    SeedPrimitive,
    // --- Supporting types ---
    ProvenanceChain, Contradiction, EpistemicConflictType,
    // --- Delta types ---
    GraphDelta, GovernedDelta, AnchoredDelta,
    // --- Feedback loop types (MD-3, MD-6) ---
    EnrichmentRequest, EnrichmentSource, ReExtractionRequest, RecallAction,
    ExtractionQualityTracker,
    // --- Graph inspection types ---
    GraphNeighborhood, GraphSnapshot, WeakFrame,
    // --- Executive types (MD-5) ---
    ReflectionLoopResult, ReasoningState, ReasoningGoal,
    // --- Governance types (MD-4) ---
    GovernanceUpdate, PromotionVerdict, SeedAdjustment, ContradictionResolution, ResolutionType,
    // --- Utility functions ---
    extract_keywords,
};

// Re-export pipeline types for convenience.
pub use pipeline::{
    // --- Core pipeline types ---
    PipelineEngine, ErasedTransform, IngestResult, register_default_pipeline,
    // --- v12 Graph ---
    Graph,
    // --- Placeholder transforms ---
    Tokenize, IngestAtoms, EnrichComposition, ReExtractFrame, NoOpTransform,
};

// Re-export MD-1: ExtractFrame types.
pub use extract_frame::{
    ExtractFrame,
    ExtractionQuality,
    ExtractionQualityTrackerExt,
};

// Re-export MD-2: ReasonFrame types.
pub use reason_frame::{
    ReasonFrame,
    ReasoningRule,
    ReasoningContext,
    ReasoningResult,
    GraphContextRef,
    ProblemSolutionRule,
    GoalInferenceRule,
    PolarityConflictRule,
};

// Re-export MD-4: GovernBeliefs & SeedAnchor types.
pub use govern_beliefs::{
    GovernBeliefs,
    SeedAnchor,
};

// Re-export MD-5: Executive types.
pub use executive::{
    CognitiveMode,
    ComputeBudget,
    StopCondition,
    ExecutiveOrchestrator,
    Reflect,
    ReflectionFinding,
    ReflectionFindingType,
    ReflectionAction,
};

// Re-export MD-6: Acquisition types.
pub use acquisition::{
    DetectGaps,
    SelectAcquisition,
    KnowledgeGap,
    KnowledgeGapType,
    AcquisitionDecision,
    AcquisitionStrategy,
    InquiryQuestion,
    InquiryMemory,
};
