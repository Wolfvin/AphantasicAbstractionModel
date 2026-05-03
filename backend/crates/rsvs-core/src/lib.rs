#![warn(missing_docs)]

//! RSVS Core — v6.0 Compositional Architecture
//!
//! Every sense is formed by compositions — pairs of (ID, sense_id).
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
pub mod error;
pub mod events;
pub mod graph;
pub mod persist;
pub mod pipeline;
pub mod seed;
pub mod sense;
pub mod tests;
pub mod transformer_bridge;
pub mod types;

pub use attention::{
    is_groundable_to_seeds, text_to_sentences, tokenize, AttentionConfig, CoocStats,
    EntityDetector, RsvsAttention,
};
pub use autonomy::{
    AtomRecord, AutonomyConfig, AutonomyEngine, ConfidenceUpdateResult, MemoryClass,
    RemovalDecision, StabilityStatus, StatusTransitionResult, WarmUpState,
};
pub use error::RsvsError;
pub use events::{EventBatch, RuntimeEvent, RuntimeSnapshot, API_VERSION, SCHEMA_VERSION};
pub use graph::{
    jaccard_sets, RsvsGraph, SimilarityResult, StructuralSimResult, SubstitutionResult,
};
pub use pipeline::{
    AppraiseResult, IngestStats, PipelineConfig, PipelineStatus, QueryResult, RelateResult, Rsvs,
};
pub use sense::{
    GroundingEvidence, GroundingVerdict, IngestResult, Sense, SenseConfig, SenseInductionConfig,
    SenseManager, SenseStatus,
};
pub use transformer_bridge::{TransformerBridge, TransformerBridgeConfig};
pub use types::*;
