#![warn(missing_docs)]

//! RSVS Core — v8.0 — Language-Agnostic Seeds + Convergence Detection + Internal Representation
//!
//! v6.1 builds on v6.0 with:
//! - `TraversalConfig`: Controls recursive composition expansion during queries
//! - `HaltReason`: Why a traversal stopped (stability, confidence, depth, relevance)
//! - `ContextQueryResult`: Scored atoms with P(a|S,q) from depth-controlled traversal
//! - Cycle detection via `HashSet<(NodeId, SenseId)>` during traversal
//! - Freq map per sense for weighted scoring P(a|S,q)
//! - Inactivity TTL for atom expiry
//!
//! v6.0: Every sense is formed by compositions — pairs of (ID, sense_id).
//! Relationships between IDs are structural, derived from shared/differing
//! compositions, not statistical co-occurrence alone.
//!
//! Key concepts:
//! - `CompositionRef`: Reference to a specific sense of a specific node
//! - `layer`: Compositional depth (0 = primitive, N = composed from layer N-1)
//! - `structural_similarity`: Compare nodes by shared/differing compositions
//! - `substitution_analysis`: Find what transforms one sense into another
//! - `TransformerBridge`: Interpretation layer ON TOP of Transformer output
//! - `GroundingEvidence`: Full evidence trail for composition verification
//! - `SenseInductionConfig`: Tunable parameters for sense formation

pub mod attention;
pub mod autonomy;
#[cfg(feature = "python")]
pub mod bindings;
pub mod composition_index;
pub mod convergence;
pub mod consolidation;
pub mod error;
pub mod events;
pub mod graph;
pub mod matryoshka;
pub mod mcts;
pub mod persist;
pub mod pipeline;
pub mod reflection;
pub mod seed;
pub mod sense;
pub mod tests;
pub mod thinking;
pub mod transformer_bridge;
pub mod types;
pub mod neurosym;
pub mod paradigm;
pub mod spreading;
pub mod deps;

pub use attention::{
    is_groundable_to_seeds, text_to_sentences, tokenize, AttentionComponent, AttentionConfig,
    CoocStats, DomainAttentionConfig, EntityDetector, RsvsAttention,
};
pub use autonomy::{
    AtomRecord, AutonomyConfig, AutonomyEngine, ConfidenceUpdateResult, MemoryClass,
    RemovalDecision, StabilityStatus, StatusTransitionResult, WarmUpState, count_impact,
};
pub use composition_index::CompositionIndex;
pub use convergence::{ConvergenceConfig, ConvergenceEngine, ConvergencePair};
pub use error::RsvsError;
pub use events::{EventBatch, RuntimeEvent, RuntimeSnapshot, API_VERSION, SCHEMA_VERSION};
pub use graph::{
    jaccard_sets, RsvsGraph, SimilarityResult, StructuralSimResult, SubstitutionResult,
};
pub use matryoshka::MatryoshkaTraversal;
pub use mcts::MCTSTraversal;
pub use pipeline::{
    AppraiseResult, IngestStats, PipelineConfig, PipelineStatus, QueryResult, RelateResult, Rsvs,
    traverse_query,
};
pub use reflection::SenseReflection;
pub use sense::{
    GroundingEvidence, GroundingVerdict, IngestResult, Sense, SenseConfig, SenseInductionConfig,
    SenseManager, SenseStatus,
};
pub use thinking::{ThinkingMode, ThinkingToggle, ThinkingToggleConfig, ComplexitySignal};
pub use transformer_bridge::{TransformerBridge, TransformerBridgeConfig};
pub use paradigm::{ParadigmRouter, ParadigmRouterConfig, TraversalParadigm, CalibrationEntry};
pub use spreading::{SpreadingActivation, SpreadingActivationConfig, ActivationResult};
pub use deps::{DEPSPlanner, DEPSResult, RecoveryPlan, RecoveryAction, FailureType};
pub use types::*;
