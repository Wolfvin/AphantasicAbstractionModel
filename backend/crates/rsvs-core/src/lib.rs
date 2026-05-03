//! RSVS Core — v0.5
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

pub mod types;
pub mod graph;
pub mod seed;
pub mod sense;
pub mod attention;
pub mod autonomy;
pub mod pipeline;
pub mod persist;
pub mod events;
pub mod bindings;
pub mod tests;

pub use types::*;
pub use graph::{RsvsGraph, SimilarityResult, jaccard_sets};
pub use sense::{Sense, SenseManager, SenseConfig, SenseStatus, IngestResult};
pub use attention::{
    AttentionConfig, CoocStats, RsvsAttention, EntityDetector,
    text_to_sentences, tokenize, is_groundable_to_seeds,
};
pub use autonomy::{
    AutonomyConfig, AutonomyEngine, AtomRecord, MemoryClass,
    ConfidenceUpdateResult, RemovalDecision, StabilityStatus, WarmUpState,
};
pub use pipeline::{Rsvs, PipelineConfig, IngestStats, QueryResult, PipelineStatus};
pub use events::{API_VERSION, SCHEMA_VERSION, RuntimeEvent, RuntimeSnapshot, EventBatch};
