#![deny(missing_docs)]

//! RSVS Core — v4.2
//!
//! v4.2: Unified node model, status lifecycle, policy engine in Rust
//!
//! v0.5: End-to-end pipeline (Rsvs struct)
//!   text → CoocStats → EntityDetector → atom promotion
//!   → RSVS Attention → SenseManager ingest
//!   → AutonomyEngine confidence update
//!   → context-aware query + similarity
//!
//! v0.4: AutonomyEngine — confidence, tiered autonomy, adaptive thresholds
//! v0.3: RSVS Attention — NPMI+Jaccard+cooc, text pipeline, entity detection
//! v0.2: Multi-sense — SenseManager, coherence, lazy lookup, merge
//! v0.1: DAG, integer IDs, circular ref, Jaccard, seed graph

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
pub use graph::{jaccard_sets, RsvsGraph, SimilarityResult};
pub use pipeline::{
    AppraiseResult, IngestStats, PipelineConfig, PipelineStatus, QueryResult, RelateResult, Rsvs,
};
pub use sense::{IngestResult, Sense, SenseConfig, SenseManager, SenseStatus};
pub use types::*;
