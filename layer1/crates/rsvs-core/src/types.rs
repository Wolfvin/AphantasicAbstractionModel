//! Shared types used by both v8.3 legacy and v12.0.
//!
//! This file was trimmed as part of the C4+C5 refactoring (v12.0 Node migration).
//! It now contains only the types that v12.0 still imports from `crate::types`:
//!
//! - `NodeId` — u32 node identifier
//! - `SenseId` — u32 sense identifier
//! - `EdgeSource` — provenance source (extended in v12.0 with new variants)
//! - `RelationType` — semantic relation classification
//! - `HiddenMeaningType` — hidden meaning classification
//!
//! All other v8.3 types (Node, Edge, CompositionRef, GapAnnotation, SenseProfile,
//! DiscourseMeta, BlendResult, AbductiveHypothesis, NamedPattern, SynthesisResult,
//! Fingerprint, TraversalConfig, etc.) have been removed. The v12.0 `Node` is
//! defined in `v12::types::Node` with only the fields actually used by v12.

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
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EdgeSource {
    /// Created during seed bootstrap.
    #[default]
    Bootstrap,
    /// Created through learning from text ingestion.
    Learned,
    /// Created by explicit composition (compose API).
    Composition,
    /// Created by gap detection (P1) — predicted but not observed compositions.
    GapDetection,
    /// Created by discourse tracking (P3) — rhetorical/performative edges.
    Discourse,
    /// v10.0: Created by compositional blending — hybrid A∧B edges.
    Blending,
    /// v10.0: Created by abductive reasoning — hypothetical X→Y→Z edges.
    Abductive,
    /// v10.0: Created by pattern mining — named pattern edges.
    PatternMining,
    /// v10.0: Created by cross-pathway synthesis — hidden meaning edges.
    Synthesis,
    /// v10.1: Created by compound discovery — multi-word expression edges.
    CompoundDiscovery,

    // v12.0: New provenance sources (replaces ProvenanceSource from MD-4)
    /// v12.0: From MD-1: semantic frame extraction.
    FrameCompiler,
    /// v12.0: From MD-2: pre-ingest reasoning.
    HiddenMeaningRule,
    /// v12.0: From MD-4: belief state transition.
    EpistemicGovernance,
    /// v12.0: From MD-5: executive routing.
    ExecutiveControl,
    /// v12.0: From MD-6: passive recall.
    AcquisitionRecall,
    /// v12.0: From MD-6: self-study.
    AcquisitionSelfStudy,
    /// v12.0: From MD-6: user answer.
    AcquisitionUserAnswer,
    /// v12.0: Human override.
    HumanAssertion,
    /// v12.0: Feedback loop — composition enriched after gap detection.
    EnrichmentFeedback,
    /// v12.0: Feedback loop — frame re-extracted with graph context.
    ExtractionRepair,
}

/// L0-02: Relation type for edges — mirrors Python Layer 0 RelationType.
///
/// Every edge in the graph now carries what kind of semantic relation it
/// represents. This information flows from Layer 0 (perceptual abstractors)
/// through the adapter into Layer 1 (RSVS graph). Default is Categorical
/// for backward compatibility with existing edges that have no relation type.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum RelationType {
    /// "X is a Y" — categorical / taxonomic relation.
    #[default]
    Categorical,
    /// "X is more/less than Y in dimension D" — comparative relation.
    Differential,
    /// "X can do Y" / "X is used for Y" — functional relation.
    Functional,
    /// "X is located at Y" — spatial relation.
    Spatial,
    /// "X occurs before/after Y" — temporal relation.
    Temporal,
    /// "X causes Y" / "X is caused by Y" — causal relation.
    Causal,
    /// Discursive / rhetorical relation between utterances.
    Discursive,
}

/// Types of hidden meaning that can be discovered.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HiddenMeaningType {
    /// The surface meaning masks a deeper affective truth.
    AffectiveDisguise,
    /// A social dynamic is hidden beneath the literal content.
    SocialConcealment,
    /// The utterance is a performative act disguised as something else.
    PerformativeMask,
    /// A trauma pattern underlies the surface expression.
    TraumaPattern,
    /// Power dynamics hidden in the communication.
    PowerDynamic,
    /// General emergent meaning not fitting other categories.
    Emergent,
}
