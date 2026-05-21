//! Shared types used by both v8.3 legacy and the unified architecture.
//!
//! This file was trimmed as part of the C4+C5 refactoring (Node migration).
//! It now contains only the types that the unified architecture still imports
//! from `crate::types`:
//!
//! - `NodeId` — u32 node identifier
//! - `SenseId` — u32 sense identifier
//! - `EdgeSource` — provenance source (extended with new variants)
//! - `RelationType` — semantic relation classification
//! - `HiddenMeaningType` — hidden meaning classification
//!
//! All other v8.3 types (Node, Edge, CompositionRef, GapAnnotation, SenseProfile,
//! DiscourseMeta, BlendResult, AbductiveHypothesis, NamedPattern, SynthesisResult,
//! Fingerprint, TraversalConfig, etc.) have been removed. The unified `Node` is
//! defined in `v12::types::Node` with only the fields actually used.

use serde::{Deserialize, Serialize};

// -----------------------------------------------------------------------
// Shared Type Aliases
// -----------------------------------------------------------------------

/// A node ID. u32 = 4 bytes vs ~50 bytes for a String.
pub type NodeId = u32;

/// A sense identifier — unique within a node's sense list.
/// Together with a NodeId, uniquely identifies any sense in the system.
pub type SenseId = u32;

// -----------------------------------------------------------------------
// Shared Enums
// -----------------------------------------------------------------------

/// Source type for edges.
///
/// # Active Variants (produced by the v12 pipeline)
///
/// - `Bootstrap` — seed bootstrap edges
/// - `Learned` — token edges from text ingestion
/// - `FrameCompiler` — MD-1 semantic frame extraction
/// - `HiddenMeaningRule` — MD-2 pre-ingest reasoning
/// - `AcquisitionRecall` — MD-6 passive recall
/// - `AcquisitionSelfStudy` — MD-6 self-study
/// - `AcquisitionUserAnswer` — MD-6 user answer
/// - `HumanAssertion` — human override
/// - `EnrichmentFeedback` — feedback loop (composition enriched after gap detection)
/// - `ExtractionRepair` — feedback loop (frame re-extracted with graph context)
///
/// # Design-Intent Variants (in match arms, not yet produced by default pipeline)
///
/// - `Abductive` — for future `CompositionType::Hypothesis` pipeline output
/// - `PatternMining` — for future `CompositionType::Pattern` pipeline output
///
/// # Removed Variants (Phase 1 cleanup, see ARCHIVED_VARIANTS.md)
///
/// `Composition`, `GapDetection`, `Discourse`, `Blending`, `Synthesis`,
/// `CompoundDiscovery`, `EpistemicGovernance`, `ExecutiveControl` —
/// removed because they had zero references in the Rust codebase.
/// See `ARCHIVED_VARIANTS.md` for the full list with documentation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EdgeSource {
    /// Created during seed bootstrap.
    #[default]
    Bootstrap,
    /// Created through learning from text ingestion.
    /// Used by pipeline.rs for token atoms and persistence.rs for loaded graphs.
    Learned,

    // Provenance sources from MD-1 through MD-6 (all active in v12)
    /// From MD-1: semantic frame extraction.
    FrameCompiler,
    /// From MD-2: pre-ingest reasoning.
    HiddenMeaningRule,
    /// Design-intent: for future CompositionType::Pattern pipeline output.
    /// Referenced in govern_beliefs.rs match arm.
    PatternMining,
    /// Design-intent: for future CompositionType::Hypothesis pipeline output.
    /// Referenced in govern_beliefs.rs match arm.
    Abductive,
    /// From MD-6: passive recall.
    AcquisitionRecall,
    /// From MD-6: self-study.
    AcquisitionSelfStudy,
    /// From MD-6: user answer.
    AcquisitionUserAnswer,
    /// Human override.
    HumanAssertion,
    /// Feedback loop — composition enriched after gap detection.
    EnrichmentFeedback,
    /// Feedback loop — frame re-extracted with graph context.
    ExtractionRepair,
    /// From morphological analysis (Morphological Sense Graph).
    MorphologicalAnalysis,
}

/// L0-02: Relation type for edges — mirrors Python Layer 0 RelationType.
///
/// Every edge in the graph now carries what kind of semantic relation it
/// represents. This information flows from Layer 0 (perceptual abstractors)
/// through the adapter into Layer 1 (RSVS graph). Default is Categorical
/// for backward compatibility with existing edges that have no relation type.
///
/// # Active Variants
///
/// - `Categorical` — "X is a Y" (default, most common)
/// - `Causal` — "X causes Y" (used by hidden meaning and causal compositions)
///
/// # Removed Variants (Phase 1 cleanup, see ARCHIVED_VARIANTS.md)
///
/// `Differential`, `Functional`, `Spatial`, `Temporal`, `Discursive` —
/// removed because they had zero references in the Rust codebase.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum RelationType {
    /// "X is a Y" — categorical / taxonomic relation.
    #[default]
    Categorical,
    /// "X causes Y" / "X is caused by Y" — causal relation.
    Causal,
}

/// Types of hidden meaning that can be discovered.
///
/// # Active Variants
///
/// - `Emergent` — general emergent meaning (produced by ReasonFrame)
///
/// # Removed Variants (Phase 1 cleanup, see ARCHIVED_VARIANTS.md)
///
/// `AffectiveDisguise`, `SocialConcealment`, `PerformativeMask`,
/// `TraumaPattern`, `PowerDynamic` — removed because they had zero
/// references in the Rust codebase. Never produced by the pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HiddenMeaningType {
    /// General emergent meaning not fitting other categories.
    Emergent,
}
