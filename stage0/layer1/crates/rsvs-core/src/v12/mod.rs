//! # v1.0.0 — Unified Abstraction Types & Pipeline Engine
//!
//! This module contains the v1.0.0 type system and pipeline engine for the
//! AphantasicAbstractionModel (AAM) architecture. It is the FOUNDATION for the
//! entire architecture as defined in the design documents (MD-1 through MD-6).
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
//! their dependency DAG. Use [`register_default_pipeline`] to wire all 14 core
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
//! - **Unified**: v1.0.0 types are the ONLY architecture — the old v8.3 types
//!   (`Node`, `Edge`, `CompositionRef`) are legacy and only kept where still
//!   referenced by the v12 graph's `HashMap<NodeId, Node>` storage.
//! - **Always compiled**: The `v12` feature flag exists but is enabled by default.
//!   This module is always compiled in practice.
//! - **Non-exhaustive**: Enums that may grow in future versions are marked `#[non_exhaustive]`.
//! - **Backward-compatible**: `#[serde(default)]` is used where appropriate for
//!   forward-compatible deserialization.

pub mod acquisition;
pub mod convergence;
pub mod executive;
pub mod extract_frame;
pub mod govern_beliefs;
pub mod locale;
pub mod persistence;
pub mod pipeline;
pub mod reason_frame;
pub mod spreading;
pub mod temporal;
pub mod types;
pub mod verbalize;

#[cfg(test)]
mod cognitive_tests;

// Re-export key types for convenience.
// Users can import from `rsvs::v12::SemanticAtom` instead of `rsvs::v12::types::SemanticAtom`.
pub use types::{
    // --- Utility functions ---
    extract_keywords,
    AcquisitionSource,
    AnchoredDelta,
    AtomType,
    AtomVariant,
    // --- Abstraction 2: Composition ---
    Composition,
    CompositionEvidence,
    CompositionId,
    CompositionMember,
    CompositionType,
    Contradiction,
    ContradictionResolution,
    // --- Feedback loop types (MD-3, MD-6) ---
    EnrichmentRequest,
    EnrichmentSource,
    EpistemicConflictType,
    EpistemicState,
    // --- Quality tracking (MD-1) ---
    ExtractionQuality as ExtractionQualityStats,
    ExtractionQualityTracker,
    FrameSource,
    // --- Governance types (MD-4) ---
    GovernanceUpdate,
    GovernedDelta,
    // --- Delta types ---
    GraphDelta,
    // --- Graph inspection types ---
    GraphNeighborhood,
    GraphSnapshot,
    KnowledgeGapPlaceholder,
    // --- Abstraction 3: Two Orthogonal Status Axes ---
    LifecycleState,
    // --- v1.0.0 Node ---
    Node,
    PatternCategory,
    PipelineContext,
    Polarity,
    PromotionVerdict,
    // --- Supporting types ---
    ProvenanceChain,
    ReExtractionRequest,
    ReasoningGoal,
    ReasoningState,
    RecallAction,
    // --- Executive types (MD-5) ---
    ReflectionLoopResult,
    ResolutionType,
    SeedAdjustment,
    // --- Abstraction 6: Seed Anchoring ---
    SeedPrimitive,
    // --- Abstraction 1: SemanticAtom ---
    SemanticAtom,
    // --- Abstraction 4: SemanticEdge ---
    SemanticEdge,
    // --- Abstraction 1b: SemanticRole ---
    SemanticRole,
    // --- Phase J–P: Sense Layer ---
    Sense,
    SenseCandidate,
    SenseGrounding,
    // --- Abstraction 5: Transform ---
    Transform,
    Voice,
    WeakFrame,
};

// Re-export pipeline types for convenience.
pub use pipeline::{
    register_default_pipeline,
    EnrichComposition,
    ErasedTransform,
    // --- v12 Graph ---
    Graph,
    IngestAtoms,
    IngestResult,
    NoOpTransform,
    // --- Core pipeline types ---
    PipelineEngine,
    PipelineError,
    ReExtractFrame,
    // --- Placeholder transforms (STUB:PLACEHOLDER) ---
    SyncPipelineEngine,
    Tokenize,
};

// Re-export MD-1: ExtractFrame types.
// Audit v6 fix: Removed confusing alias `ExtractionQuality as ExtractionQualityLevel`.
// Both names are now exported independently:
//   - `ExtractionQuality` (from extract_frame) — per-rule quality tracker
//   - `ExtractionQualityStats` (from types, re-exported above) — aggregate quality stats
pub use extract_frame::{
    ExtractFrame, ExtractionQuality, ExtractionQualityTrackerExt,
};

// Audit v6: Backward compat alias — `ExtractionQualityLevel` was the old name.
// New code should use `ExtractionQuality` directly.
pub use extract_frame::ExtractionQuality as ExtractionQualityLevel;

// Re-export MD-2: ReasonFrame types.
pub use reason_frame::{
    GoalInferenceRule, GraphContextRef, PolarityConflictRule, ProblemSolutionRule, ReasonFrame,
    ReasoningContext, ReasoningResult, ReasoningRule,
};

// Re-export MD-4: GovernBeliefs & SeedAnchor types.
pub use govern_beliefs::{GovernBeliefs, SeedAnchor};

// Re-export MD-5: Executive types.
pub use executive::{
    CognitiveMode, ComputeBudget, ExecutiveOrchestrator, Reflect, ReflectionAction,
    ReflectionFinding, ReflectionFindingType, StopCondition,
};

// Re-export MD-6: Acquisition types.
pub use acquisition::{
    AcquisitionDecision, AcquisitionStrategy, DetectGaps, InquiryMemory, InquiryQuestion,
    KnowledgeGap, KnowledgeGapType, SelectAcquisition,
};

// Re-export Spreading Activation types.
pub use spreading::{
    ActivationMap, SpreadingActivation, SpreadingActivationTransform, SpreadingConfig,
};

// Re-export Convergence Detection types.
pub use convergence::{
    ConvergenceConfig, ConvergenceDetection, ConvergenceDetectionTransform, ConvergencePair,
};

// Re-export Temporal Decay types.
pub use temporal::{DecayConfig, DecayResult, TemporalDecay, TemporalDecayTransform};

// Re-export Persistence types.
pub use persistence::{GraphStats, Persistence, PersistenceError};

// Re-export Compositional Verbalization Engine types.
pub use verbalize::{
    CompositionalVerbalize, CompositionalVerbalizeTransform, VerbalizationResult, VerbalizeConfig,
};

// Re-export i18n Locale types.
pub use locale::{default_locale, EpistemicQualifiers, EnglishLocale, IndonesianLocale, Locale};
