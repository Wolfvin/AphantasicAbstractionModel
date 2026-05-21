//! # v1.0.0 Transform Pipeline Engine
//!
//! This module implements the DAG-based transform pipeline that drives all
//! knowledge ingestion. Transforms are registered with dependency chains and optional
//! conditions, then executed in topological order.
//!
//! ## Architecture
//!
//! ```text
//! RawText → Tokenize → ExtractFrame → ReasonFrame ─┐
//!              │                                     │
//!              └───────────────────→ IngestAtoms ←───┘
//!                                        │
//!                                   GovernBeliefs
//!                                        │
//!                                     SeedAnchor
//!                                        │
//!                                    DetectGaps
//!                                        │
//!                                  SelectAcquisition
//!                                     /          \
//!                          EnrichComposition   ReExtractFrame
//! ```
//!
//! ## Key Design Decisions
//!
//! - **Object-safe trait**: Since the `Transform` trait has associated types
//!   (`Input`, `Output`), it cannot be made into a trait object directly.
//!   We use [`ErasedTransform`] as an object-safe wrapper that reads/writes
//!   all data through [`PipelineContext`] and [`Graph`].
//!
//! - **Condition-gated execution**: Each transform node can have an optional
//!   condition closure. Transforms whose conditions evaluate to `false` are
//!   skipped during DAG execution.
//!
//! - **Topological sort**: Kahn's algorithm ensures transforms execute in
//!   dependency order. A cycle in the dependency graph is treated as a
//!   registration error.
//!
//! ## Feature Flag
//!
//! This module is only compiled when the `v12` feature is enabled.

pub mod engine;
pub mod enrich;
pub mod graph;
pub mod ingest_atoms;
pub mod re_extract;
pub mod registry;
pub mod tokenize;

#[cfg(test)]
mod tests;

// Re-export all public items so that existing `use super::pipeline::{…}` paths
// continue to work unchanged.
pub use engine::{
    ErasedTransform, IngestResult, NoOpTransform, PipelineEngine, PipelineError,
    SyncPipelineEngine, TransformCondition,
};
pub use enrich::EnrichComposition;
pub use graph::{Graph, UtteranceContext};
pub use ingest_atoms::IngestAtoms;
pub use re_extract::ReExtractFrame;
pub use registry::register_default_pipeline;
pub use tokenize::Tokenize;
