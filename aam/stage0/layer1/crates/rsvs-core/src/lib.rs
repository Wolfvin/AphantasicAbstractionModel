//! RSVS Core — v1.0.0 — Unified Abstraction Architecture
//!
//! This is the v1.0.0 architecture of the AphantasicAbstractionModel (AAM).
//! The old v8.3 flat pipeline (Rsvs, RsvsGraph, SenseManager, etc.) has been
//! replaced by the unified DAG-based transform pipeline.
//!
//! ## The 6 Unified Abstractions (MD-3)
//!
//! | # | Abstraction | Replaces | Purpose |
//! |---|------------|----------|---------|
//! | 1 | [`v12::SemanticAtom`] | Token, EventFrame, HiddenMeaningCandidate | Universal ingest primitive |
//! | 2 | [`v12::Composition`] | EventFrame, HiddenMeaningCandidate, Pattern, Hypothesis | Universal structured grouping |
//! | 3 | [`v12::LifecycleState`] + [`v12::EpistemicState`] | NodeStatus, CandidateStatus, BeliefState, GroundingVerdict | Two orthogonal status axes |
//! | 4 | [`v12::SemanticEdge`] | Separate RelationType, EdgeSource, SemanticRole, ProvenanceSource | Single typed triple |
//! | 5 | [`v12::Transform`] | Hardcoded pipeline stages | Declarative transform graph |
//! | 6 | [`v12::SeedPrimitive`] + seed_scores | Source trust weight system | Seed-driven epistemic confidence |
//!
//! ## Pipeline Engine
//!
//! The [`v12::PipelineEngine`] executes transforms in topological order based on
//! their dependency DAG. Use [`v12::register_default_pipeline`] to wire all 13 core
//! transforms, then call [`v12::PipelineEngine::ingest`] to process text.
//!
//! ## MD-Specific Transforms
//!
//! | MD | Transform | Module | Purpose |
//! |----|-----------|--------|---------|
//! | MD-1 | [`v12::ExtractFrame`] | [`v12::extract_frame`] | Rule-based frame extraction |
//! | MD-2 | [`v12::ReasonFrame`] | [`v12::reason_frame`] | Pre-ingest reasoning rules |
//! | MD-4 | [`v12::GovernBeliefs`] + [`v12::SeedAnchor`] | [`v12::govern_beliefs`] | Lifecycle/epistemic governance |
//! | MD-5 | [`v12::ExecutiveOrchestrator`] | [`v12::executive`] | Cognitive mode & reflection |
//! | MD-6 | [`v12::DetectGaps`] + [`v12::SelectAcquisition`] | [`v12::acquisition`] | Gap detection & acquisition |

// Shared infrastructure — used by v12 and by Python bindings.
pub mod error;
pub mod types;

// v1.0.0 architecture — the ONLY pipeline engine.
pub mod v12;

// PyO3 bindings — always compiled when `python` feature is enabled.
// Exposes PyV12Pipeline and v12 types to Python.
#[cfg(feature = "python")]
pub mod bindings;
