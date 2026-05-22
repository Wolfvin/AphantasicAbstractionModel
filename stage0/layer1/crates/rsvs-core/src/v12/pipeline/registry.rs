use super::engine::PipelineEngine;
use super::enrich::EnrichComposition;
use super::ingest_atoms::IngestAtoms;
use super::morphological_analysis::MorphologicalAnalysis;
use super::re_extract::ReExtractFrame;
use super::super::types::*;
use super::tokenize::Tokenize;

use super::super::acquisition::{DetectGaps, SelectAcquisition};
use super::super::convergence::ConvergenceDetectionTransform;
use super::super::extract_frame::ExtractFrame;
use super::super::govern_beliefs::{GovernBeliefs, SeedAnchor};
use super::super::reason_frame::ReasonFrame;
use super::super::spreading::SpreadingActivationTransform;
use super::super::temporal::TemporalDecayTransform;
use super::super::verbalize::CompositionalVerbalizeTransform;
use super::super::csd::CSDTransform;

// ========================================================================
// register_default_pipeline — Wire All Core Transforms
// ========================================================================

/// Register all core v1.0.0 transforms in dependency order.
///
/// This wires up the complete default pipeline with 15 transforms:
///
/// | # | Transform | Dependencies | Condition |
/// |---|-----------|-------------|------------|
/// | 1 | Tokenize | (none) | always |
/// | 2 | ExtractFrame | Tokenize | is_sentence_like |
/// | 3 | ReasonFrame | ExtractFrame | has_event_atoms |
/// | 4 | IngestAtoms | Tokenize, ReasonFrame | always |
/// | 4b | MorphologicalAnalysis | Tokenize | always |
/// | 5 | GovernBeliefs | IngestAtoms | always |
/// | 6 | SeedAnchor | GovernBeliefs | always |
/// | 7 | DetectGaps | SeedAnchor | gap_detection_enabled |
/// | 8 | SelectAcquisition | DetectGaps | has_gaps |
/// | 9 | EnrichComposition | SelectAcquisition | has_enrichment_requests |
/// | 10 | ReExtractFrame | SelectAcquisition | has_reextraction_requests |
/// | 11 | TemporalDecay | EnrichComposition | always |
/// | 12 | SpreadingActivation | GovernBeliefs | has_event_atoms |
/// | 13 | ConvergenceDetection | EnrichComposition, TemporalDecay | always |
/// | 14 | CompositionalVerbalize | ConvergenceDetection | always |
pub fn register_default_pipeline(engine: &mut PipelineEngine) {
    // 1. Tokenize — no dependencies, always runs.
    engine.register(Tokenize::new(), vec![], None);

    // 2. ExtractFrame — depends on Tokenize, condition: is_sentence_like.
    engine.register(
        ExtractFrame::new(),
        vec!["Tokenize".to_string()],
        Some(Box::new(|ctx: &PipelineContext| ctx.is_sentence_like())),
    );

    // 3. ReasonFrame — depends on ExtractFrame, condition: has_event_atoms.
    engine.register(
        ReasonFrame::new(),
        vec!["ExtractFrame".to_string()],
        Some(Box::new(|ctx: &PipelineContext| ctx.has_event_atoms())),
    );

    // 4. IngestAtoms — depends on Tokenize + ReasonFrame, always runs.
    engine.register(
        IngestAtoms::new(),
        vec!["Tokenize".to_string(), "ReasonFrame".to_string()],
        None,
    );

    // 4b. MorphologicalAnalysis — decomposes stemmed tokens into graph structures.
    //     Depends on Tokenize (needs atoms with RootForm role).
    //     Always runs: even non-sentence input may have stemmed tokens.
    engine.register(
        MorphologicalAnalysis::new(),
        vec!["Tokenize".to_string()],
        None,
    );

    // 5. GovernBeliefs — depends on IngestAtoms, always runs.
    engine.register(GovernBeliefs::new(), vec!["IngestAtoms".to_string()], None);

    // 6. SeedAnchor — depends on GovernBeliefs, always runs.
    engine.register(SeedAnchor::new(), vec!["GovernBeliefs".to_string()], None);

    // 7. DetectGaps — depends on SeedAnchor, condition: gap_detection_enabled.
    engine.register(
        DetectGaps::new(),
        vec!["SeedAnchor".to_string()],
        Some(Box::new(|ctx: &PipelineContext| {
            ctx.gap_detection_enabled()
        })),
    );

    // 8. SelectAcquisition — depends on DetectGaps, condition: has_gaps.
    engine.register(
        SelectAcquisition::new(),
        vec!["DetectGaps".to_string()],
        Some(Box::new(|ctx: &PipelineContext| ctx.has_gaps())),
    );

    // 9. EnrichComposition — depends on SelectAcquisition, condition: has_enrichment_requests.
    engine.register(
        EnrichComposition::new(),
        vec!["SelectAcquisition".to_string()],
        Some(Box::new(|ctx: &PipelineContext| {
            ctx.has_enrichment_requests()
        })),
    );

    // 10. ReExtractFrame — depends on SelectAcquisition, condition: has_reextraction_requests.
    engine.register(
        ReExtractFrame::new(),
        vec!["SelectAcquisition".to_string()],
        Some(Box::new(|ctx: &PipelineContext| {
            ctx.has_reextraction_requests()
        })),
    );

    // 11. TemporalDecay — runs after enrichment, applies Ebbinghaus decay.
    //     No dependencies on other transforms (reads graph directly).
    //     Condition: always (decay is continuous).
    engine.register(
        TemporalDecayTransform {
            engine: super::super::temporal::TemporalDecay::new(),
        },
        vec!["EnrichComposition".to_string()],
        None,
    );

    // 12. SpreadingActivation — propagates energy from seed-anchored nodes.
    //     Depends on GovernBeliefs (seeds must be computed first).
    //     Condition: has event atoms (only when pipeline produced events).
    engine.register(
        SpreadingActivationTransform::new(),
        vec!["GovernBeliefs".to_string()],
        Some(Box::new(|ctx: &PipelineContext| ctx.has_event_atoms())),
    );

    // 12b. CSD — contextual sense disambiguation using spreading activation.
    //      Depends on SpreadingActivation (needs activation energies).
    engine.register(
        CSDTransform::new(),
        vec!["SpreadingActivation".to_string()],
        Some(Box::new(|ctx: &PipelineContext| ctx.has_event_atoms())),
    );

    // 13. ConvergenceDetection — detects structurally equivalent compositions.
    //     Runs last, after all enrichment and decay.
    //     Condition: always (checks internally if ≥2 compositions exist).
    engine.register(
        ConvergenceDetectionTransform {
            engine: super::super::convergence::ConvergenceDetection::new(),
        },
        vec!["EnrichComposition".to_string(), "TemporalDecay".to_string()],
        None,
    );

    // 14. CompositionalVerbalize — generates explanations from the graph.
    //     Audit v5 fix: Previously NOT in default pipeline — the CVE transform
    //     was fully implemented but never registered. Now registered after ConvergenceDetection.
    engine.register(
        CompositionalVerbalizeTransform::new(),
        vec!["ConvergenceDetection".to_string()],
        None,
    );
}
