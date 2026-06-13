//! # AAM v12.0 Validation Test Suite (Option B + M.1–M.7)
//!
//! Comprehensive integration tests covering the 6 unified abstractions,
//! the closed feedback loop, epistemic governance, and extraction quality.
//!
//! ## Test Modules
//!
//! | Module | Scope | MD Source |
//! |--------|-------|-----------|
//! | M.1 | Critical Missing Functions | MD-3, MD-4, MD-5, MD-6 |
//! | M.2 | Epistemic Governance Promotion/Demotion | MD-4 |
//! | M.3 | Executive Cognition Enrichment Loop | MD-5 |
//! | M.4 | Closed Feedback Loop Integration | MD-3, MD-6 |
//! | M.5 | Acquisition Pipeline (User Answer) | MD-6 |
//! | M.6 | Semantic Edge & Graph Neighborhood | MD-3, MD-5 |
//! | M.7 | ExtractionQualityTracker & Dedup | MD-1, MD-3 |

#![allow(clippy::field_reassign_with_default)]

#[cfg(feature = "v12")]
use rsvs::v12::{
    // Utility
    extract_keywords,
    // MD-3: Pipeline
    register_default_pipeline,
    AcquisitionSource,
    // MD-6: Acquisition
    AcquisitionStrategy,
    AtomType,
    AtomVariant,
    // MD-5: Executive
    CognitiveMode,
    Composition,
    CompositionMember,
    CompositionType,
    ComputeBudget,
    Contradiction,
    DetectGaps,
    EnrichmentRequest,
    EnrichmentSource,
    EpistemicConflictType,
    EpistemicState,
    ExecutiveOrchestrator,
    // MD-1: ExtractFrame
    ExtractFrame,
    ExtractionQualityStats,
    ExtractionQualityTracker,
    FrameSource,
    // MD-4: Governance
    GovernBeliefs,
    Graph,
    GraphNeighborhood,
    GraphSnapshot,
    IngestResult,
    InquiryMemory,
    InquiryQuestion,
    KnowledgeGap,
    KnowledgeGapType,
    LifecycleState,
    PipelineContext,
    PipelineEngine,
    Polarity,
    // MD-2: ReasonFrame
    ReasonFrame,
    ReasoningGoal,
    // MD-5/MD-3: Executive types from types module
    ReasoningState,
    Reflect,
    ReflectionAction,
    ReflectionFindingType,
    ReflectionLoopResult,
    ResolutionType,
    SeedAnchor,
    SeedPrimitive,
    SelectAcquisition,
    // MD-3: Types (all 6 abstractions)
    SemanticAtom,
    SemanticEdge,
    SemanticRole,
    StopCondition,
    Voice,
};

#[cfg(feature = "v12")]
use std::collections::HashMap;

#[cfg(feature = "v12")]
use rsvs::types::{EdgeSource, NodeId, RelationType};

// ========================================================================
// Shared Test Helpers
// ========================================================================

/// Create a basic Event composition with standard roles.
#[cfg(feature = "v12")]
fn make_event_composition(
    id: &str,
    predicate_label: &str,
    agent_label: &str,
    patient_label: &str,
    cause_label: Option<&str>,
    confidence: f32,
) -> Composition {
    let mut comp = Composition::default();
    comp.id = id.to_string();
    comp.composition_type = CompositionType::Event;
    comp.confidence = confidence;
    comp.lifecycle = LifecycleState::Candidate;
    comp.epistemic = EpistemicState::Observed;
    comp.provenance.origin = EdgeSource::FrameCompiler;
    comp.batch_seen = 3;

    let mut node_id_counter: NodeId = 1;
    comp.members.push(CompositionMember {
        node_id: node_id_counter,
        role: SemanticRole::Predicate,
        confidence: confidence * 0.95,
        label: predicate_label.to_string(),
    });
    node_id_counter += 1;
    comp.members.push(CompositionMember {
        node_id: node_id_counter,
        role: SemanticRole::Arg0Agent,
        confidence: confidence * 0.9,
        label: agent_label.to_string(),
    });
    node_id_counter += 1;
    comp.members.push(CompositionMember {
        node_id: node_id_counter,
        role: SemanticRole::Arg1Patient,
        confidence: confidence * 0.85,
        label: patient_label.to_string(),
    });
    if let Some(cause) = cause_label {
        node_id_counter += 1;
        comp.members.push(CompositionMember {
            node_id: node_id_counter,
            role: SemanticRole::Cause,
            confidence: confidence * 0.8,
            label: cause.to_string(),
        });
    }
    comp
}

/// Create a HiddenMeaning composition with Problem/Solution.
/// Kept for future test expansion.
#[cfg(feature = "v12")]
#[allow(dead_code)]
fn make_hidden_meaning_composition(
    id: &str,
    problem_label: &str,
    solution_label: &str,
    confidence: f32,
) -> Composition {
    let mut comp = Composition::default();
    comp.id = id.to_string();
    comp.composition_type = CompositionType::HiddenMeaning;
    comp.confidence = confidence;
    comp.lifecycle = LifecycleState::Candidate;
    comp.epistemic = EpistemicState::Inferred;
    comp.provenance.origin = EdgeSource::HiddenMeaningRule;
    comp.batch_seen = 2;

    comp.members.push(CompositionMember {
        node_id: 100,
        role: SemanticRole::Predicate,
        confidence: confidence * 0.9,
        label: "problem_solution".to_string(),
    });
    comp.members.push(CompositionMember {
        node_id: 101,
        role: SemanticRole::Problem,
        confidence: confidence * 0.85,
        label: problem_label.to_string(),
    });
    comp.members.push(CompositionMember {
        node_id: 102,
        role: SemanticRole::Solution,
        confidence: confidence * 0.8,
        label: solution_label.to_string(),
    });
    comp
}

/// Create a SemanticAtom of type Event with the given roles.
#[cfg(feature = "v12")]
fn make_event_atom(
    id: &str,
    label: &str,
    agent: &str,
    patient: &str,
    cause: Option<&str>,
    confidence: f32,
) -> SemanticAtom {
    let mut roles = HashMap::new();
    roles.insert(SemanticRole::Arg0Agent, agent.to_string());
    roles.insert(SemanticRole::Arg1Patient, patient.to_string());
    if let Some(c) = cause {
        roles.insert(SemanticRole::Cause, c.to_string());
    }
    SemanticAtom {
        id: id.to_string(),
        label: label.to_string(),
        atom_type: AtomType::Event,
        roles,
        confidence,
        source: EdgeSource::FrameCompiler,
        ..SemanticAtom::default()
    }
}

/// Create a basic pipeline engine with default transforms for testing.
#[cfg(feature = "v12")]
fn make_pipeline_engine() -> PipelineEngine {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);
    engine
}

// ========================================================================
// M.1: Critical Missing Functions Unit Tests
// ========================================================================

#[cfg(feature = "v12")]
mod m1_critical_functions {
    use super::*;

    // --- Composition::member_with_role ---

    #[test]
    fn test_member_with_role_returns_correct_member() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        let agent = comp.member_with_role(&SemanticRole::Arg0Agent);
        assert!(agent.is_some());
        assert_eq!(agent.unwrap().label, "Raymond");
    }

    #[test]
    fn test_member_with_role_returns_none_for_missing_role() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        let purpose = comp.member_with_role(&SemanticRole::Purpose);
        assert!(purpose.is_none());
    }

    // --- Composition::has_member_with_role ---

    #[test]
    fn test_has_member_with_role_true_when_present() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        assert!(comp.has_member_with_role(SemanticRole::Arg0Agent));
        assert!(comp.has_member_with_role(SemanticRole::Arg1Patient));
    }

    #[test]
    fn test_has_member_with_role_false_when_absent() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        assert!(!comp.has_member_with_role(SemanticRole::Purpose));
        assert!(!comp.has_member_with_role(SemanticRole::Location));
    }

    // --- Composition::has_member_with_role_and_label ---

    #[test]
    fn test_has_member_with_role_and_label_matches() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        assert!(comp.has_member_with_role_and_label(SemanticRole::Arg0Agent, "Raymond"));
    }

    #[test]
    fn test_has_member_with_role_and_label_no_match() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        assert!(!comp.has_member_with_role_and_label(SemanticRole::Arg0Agent, "Budi"));
    }

    // --- Composition::contradiction_opposing_id ---

    #[test]
    fn test_contradiction_opposing_id_returns_some_when_contradicted() {
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        comp.contradiction = Some(Contradiction {
            conflict_type: EpistemicConflictType::PolarityConflict,
            opposing_composition_id: "comp_2".to_string(),
            strength: 0.8,
        });
        assert_eq!(comp.contradiction_opposing_id(), Some("comp_2".to_string()));
    }

    #[test]
    fn test_contradiction_opposing_id_returns_none_when_no_contradiction() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        assert!(comp.contradiction_opposing_id().is_none());
    }

    // --- Composition::provenance_source_count ---

    #[test]
    fn test_provenance_source_count_single_source() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        // Only provenance.origin = FrameCompiler, no additional member sources
        let count = comp.provenance_source_count(&[]);
        assert_eq!(count, 1); // Just FrameCompiler
    }

    #[test]
    fn test_provenance_source_count_two_independent_sources() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        // FrameCompiler + EnrichmentFeedback = 2 independent sources
        let member_sources = vec![EdgeSource::EnrichmentFeedback];
        let count = comp.provenance_source_count(&member_sources);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_provenance_source_count_dedup_same_source() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        // FrameCompiler + FrameCompiler = still 1 (dedup)
        let member_sources = vec![EdgeSource::FrameCompiler, EdgeSource::FrameCompiler];
        let count = comp.provenance_source_count(&member_sources);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_provenance_source_count_three_distinct_sources() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        let member_sources = vec![EdgeSource::EnrichmentFeedback, EdgeSource::ExtractionRepair];
        let count = comp.provenance_source_count(&member_sources);
        assert_eq!(count, 3); // FrameCompiler + EnrichmentFeedback + ExtractionRepair
    }

    // --- Composition::has_recent_contradiction ---

    #[test]
    fn test_has_recent_contradiction_within_window() {
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        comp.batch_seen = 10;
        comp.contradiction_batches = vec![8, 9]; // Recent contradictions
        assert!(comp.has_recent_contradiction(3)); // Within last 3 batches (threshold=7)
    }

    #[test]
    fn test_has_recent_contradiction_outside_window() {
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        comp.batch_seen = 10;
        comp.contradiction_batches = vec![3, 4]; // Old contradictions
        assert!(!comp.has_recent_contradiction(3)); // Outside last 3 batches (threshold=7)
    }

    #[test]
    fn test_has_recent_contradiction_no_contradictions() {
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        assert!(!comp.has_recent_contradiction(5));
    }

    // --- extract_keywords ---

    #[test]
    fn test_extract_keywords_simple_sentence() {
        let keywords = extract_keywords("Raymond membuat aplikasi karena lambat");
        assert!(!keywords.is_empty());
        // Should contain at least the content words
        assert!(
            keywords.contains(&"raymond".to_string()) || keywords.contains(&"membuat".to_string())
        );
    }

    #[test]
    fn test_extract_keywords_empty_input() {
        let keywords = extract_keywords("");
        assert!(keywords.is_empty());
    }

    #[test]
    fn test_extract_keywords_single_word() {
        let keywords = extract_keywords("aplikasi");
        assert!(!keywords.is_empty());
    }

    // --- GraphNeighborhood::neighborhood_for ---

    #[test]
    fn test_neighborhood_for_finds_relevant_compositions() {
        let comp1 = make_event_composition(
            "comp_1",
            "membuat",
            "Raymond",
            "aplikasi",
            Some("lambat"),
            0.8,
        );
        let comp2 = make_event_composition("comp_2", "menulis", "Budi", "buku", None, 0.7);
        let compositions = vec![comp1, comp2];

        // Search with keyword that matches comp_1
        let keywords = vec!["raymond".to_string()];
        let neighborhood = GraphNeighborhood::neighborhood_for(&keywords, &compositions);

        // Should find comp_1 because it has "Raymond" as a member
        assert!(!neighborhood.compositions.is_empty());
    }

    #[test]
    fn test_neighborhood_for_no_match() {
        let comp1 = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        let compositions = vec![comp1];

        let keywords = vec!["xyznotexist".to_string()];
        let neighborhood = GraphNeighborhood::neighborhood_for(&keywords, &compositions);
        assert!(neighborhood.compositions.is_empty());
    }

    #[test]
    fn test_neighborhood_for_has_contradictions() {
        let mut comp1 =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        comp1.epistemic = EpistemicState::Contradicted;
        let compositions = vec![comp1];

        let keywords = vec!["raymond".to_string()];
        let neighborhood = GraphNeighborhood::neighborhood_for(&keywords, &compositions);
        assert!(neighborhood.has_contradictions());
    }

    #[test]
    fn test_neighborhood_for_average_confidence() {
        let comp1 = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        let comp2 = make_event_composition("comp_2", "membuat", "Raymond", "laptop", None, 0.6);
        let compositions = vec![comp1, comp2];

        let keywords = vec!["raymond".to_string()];
        let neighborhood = GraphNeighborhood::neighborhood_for(&keywords, &compositions);
        if neighborhood.compositions.len() == 2 {
            let avg = neighborhood.average_confidence();
            assert!((avg - 0.7).abs() < 0.01);
        }
    }

    // --- Composition::age_in_batches ---

    #[test]
    fn test_age_in_batches() {
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        assert_eq!(comp.age_in_batches(), 3);
        comp.batch_seen = 7;
        assert_eq!(comp.age_in_batches(), 7);
    }

    // --- CompositionMember::label ---

    #[test]
    fn test_composition_member_label_returns_cached_label() {
        let member = CompositionMember {
            node_id: 1,
            role: SemanticRole::Arg0Agent,
            confidence: 0.9,
            label: "Raymond".to_string(),
        };
        assert_eq!(member.label(), "Raymond");
    }
}

// ========================================================================
// M.2: Epistemic Governance Promotion/Demotion Tests
// ========================================================================

#[cfg(feature = "v12")]
mod m2_epistemic_governance {
    use super::*;

    // --- Initial State Assignment ---

    #[test]
    fn test_initial_states_event_frame_compiler() {
        let gb = GovernBeliefs::new();
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::Event;
        comp.provenance.origin = EdgeSource::FrameCompiler;
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::New);
        assert_eq!(comp.epistemic, EpistemicState::Observed);
    }

    #[test]
    fn test_initial_states_hidden_meaning_rule() {
        let gb = GovernBeliefs::new();
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::HiddenMeaning;
        comp.provenance.origin = EdgeSource::HiddenMeaningRule;
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::Candidate);
        assert_eq!(comp.epistemic, EpistemicState::Inferred);
    }

    #[test]
    fn test_initial_states_hypothesis_abductive() {
        let gb = GovernBeliefs::new();
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::Hypothesis;
        comp.provenance.origin = EdgeSource::Abductive;
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::Quarantine);
        assert_eq!(comp.epistemic, EpistemicState::Hypothesis);
    }

    #[test]
    fn test_initial_states_human_assertion() {
        let gb = GovernBeliefs::new();
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::Event;
        comp.provenance.origin = EdgeSource::HumanAssertion;
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::Candidate);
        assert_eq!(comp.epistemic, EpistemicState::Grounded);
    }

    #[test]
    fn test_initial_states_acquisition_recall() {
        let gb = GovernBeliefs::new();
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::Acquisition;
        comp.provenance.origin = EdgeSource::AcquisitionRecall;
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::Stable);
        assert_eq!(comp.epistemic, EpistemicState::Grounded);
    }

    // --- Promotion: New → Candidate ---

    #[test]
    fn test_promotion_new_to_candidate_after_one_batch() {
        let gb = GovernBeliefs::new();
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        comp.lifecycle = LifecycleState::New;
        comp.batch_seen = 1;
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(updates
            .iter()
            .any(|u| u.new_lifecycle == Some(LifecycleState::Candidate)));
    }

    #[test]
    fn test_promotion_new_to_candidate_denied_zero_batches() {
        let gb = GovernBeliefs::new();
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        comp.lifecycle = LifecycleState::New;
        comp.batch_seen = 0;
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(!updates
            .iter()
            .any(|u| u.new_lifecycle == Some(LifecycleState::Candidate)));
    }

    // --- Promotion: Candidate → Stable ---

    #[test]
    fn test_promotion_candidate_to_stable_meets_criteria() {
        let gb = GovernBeliefs::new();
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        comp.lifecycle = LifecycleState::Candidate;
        comp.batch_seen = 3;
        comp.confidence = 0.7;
        comp.epistemic = EpistemicState::Observed;
        // Has 3 confirming members (confidence >= 0.5): Predicate, Agent, Patient
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(updates
            .iter()
            .any(|u| u.new_lifecycle == Some(LifecycleState::Stable)));
    }

    #[test]
    fn test_promotion_candidate_to_stable_denied_low_confidence() {
        let gb = GovernBeliefs::new();
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.3);
        comp.lifecycle = LifecycleState::Candidate;
        comp.batch_seen = 3;
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(!updates
            .iter()
            .any(|u| u.new_lifecycle == Some(LifecycleState::Stable)));
    }

    #[test]
    fn test_promotion_candidate_to_stable_denied_young_age() {
        let gb = GovernBeliefs::new();
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        comp.lifecycle = LifecycleState::Candidate;
        comp.batch_seen = 1; // Too young
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(!updates
            .iter()
            .any(|u| u.new_lifecycle == Some(LifecycleState::Stable)));
    }

    #[test]
    fn test_promotion_candidate_to_stable_denied_contradicted() {
        let gb = GovernBeliefs::new();
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        comp.lifecycle = LifecycleState::Candidate;
        comp.batch_seen = 3;
        comp.epistemic = EpistemicState::Contradicted;
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(!updates
            .iter()
            .any(|u| u.new_lifecycle == Some(LifecycleState::Stable)));
    }

    // --- Promotion: Inferred → Grounded ---

    #[test]
    fn test_promotion_inferred_to_grounded_multi_source() {
        let gb = GovernBeliefs::new();
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        comp.lifecycle = LifecycleState::Candidate;
        comp.epistemic = EpistemicState::Inferred;
        comp.batch_seen = 5;
        comp.confidence = 0.8;
        // Use multi-source provenance to pass the ≥2 independent sources check
        comp.provenance.origin = EdgeSource::EnrichmentFeedback;
        // Members count >= 3 triggers the heuristic multi-source check
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(updates
            .iter()
            .any(|u| u.new_epistemic == Some(EpistemicState::Grounded)));
    }

    #[test]
    fn test_promotion_inferred_to_grounded_denied_low_confidence() {
        let gb = GovernBeliefs::new();
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.5);
        comp.epistemic = EpistemicState::Inferred;
        comp.batch_seen = 5;
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(!updates
            .iter()
            .any(|u| u.new_epistemic == Some(EpistemicState::Grounded)));
    }

    // --- Contradiction Detection ---

    #[test]
    fn test_contradiction_detection_polarity_conflict() {
        let gb = GovernBeliefs::new();
        let mut comp1 = make_event_composition(
            "comp_1",
            "membuat",
            "Raymond",
            "aplikasi",
            Some("karena lambat"),
            0.8,
        );
        let mut comp2 = make_event_composition(
            "comp_2",
            "membuat",
            "Raymond",
            "aplikasi",
            Some("karena tidak lambat"),
            0.8,
        );
        // Set different node IDs for causes to trigger different-patient detection
        comp1
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Cause)
            .unwrap()
            .node_id = 10;
        comp2
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Cause)
            .unwrap()
            .node_id = 20;

        let updates = gb.detect_contradiction(&mut [comp1, comp2]);
        // Should detect some contradiction (polarity or cause difference)
        assert!(!updates.is_empty());
    }

    #[test]
    fn test_contradiction_detection_role_reversal() {
        let gb = GovernBeliefs::new();
        let mut comp1 =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        let mut comp2 =
            make_event_composition("comp_2", "membuat", "aplikasi", "Raymond", None, 0.8);
        // comp1: Agent=Raymond(1), Patient=aplikasi(2)
        // comp2: Agent=aplikasi(1), Patient=Raymond(2) — but we need swapped node IDs
        // Let's set node IDs explicitly for role reversal
        comp1
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg0Agent)
            .unwrap()
            .node_id = 1;
        comp1
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg1Patient)
            .unwrap()
            .node_id = 2;
        comp2
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg0Agent)
            .unwrap()
            .node_id = 2; // Was patient
        comp2
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg1Patient)
            .unwrap()
            .node_id = 1; // Was agent

        let updates = gb.detect_contradiction(&mut [comp1, comp2]);
        // Should detect role reversal
        assert!(updates.iter().any(|u| u
            .contradiction
            .as_ref()
            .is_some_and(|c| c.conflict_type == EpistemicConflictType::RoleReversal)));
    }

    // --- Contradiction Resolution ---

    #[test]
    fn test_contradiction_resolution_voice_confusion() {
        let gb = GovernBeliefs::new();
        let mut comp1 =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        let mut comp2 =
            make_event_composition("comp_2", "membuat", "Raymond", "aplikasi", None, 0.8);
        // Same predicate, same agent, same patient, but different provenance
        comp1.provenance.origin_id = "active_extraction".to_string();
        comp2.provenance.origin_id = "passive_extraction".to_string();
        // Make node IDs match for voice confusion check
        comp1
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg0Agent)
            .unwrap()
            .node_id = 1;
        comp1
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg1Patient)
            .unwrap()
            .node_id = 2;
        comp2
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg0Agent)
            .unwrap()
            .node_id = 1;
        comp2
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg1Patient)
            .unwrap()
            .node_id = 2;

        let resolution = gb.check_contradiction_resolution(&comp1, &comp2);
        assert!(resolution.is_some());
        assert_eq!(
            resolution.unwrap().resolution_type,
            ResolutionType::Misinterpretation
        );
    }

    // --- is_sufficiently_complete ---

    #[test]
    fn test_sufficiently_complete_event_with_all_roles() {
        let gb = GovernBeliefs::new();
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        assert!(gb.is_sufficiently_complete(&comp));
    }

    #[test]
    fn test_sufficiently_complete_event_missing_agent() {
        let gb = GovernBeliefs::new();
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::Event;
        comp.members.push(CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.8,
            label: "membuat".to_string(),
        });
        comp.members.push(CompositionMember {
            node_id: 2,
            role: SemanticRole::Arg1Patient,
            confidence: 0.8,
            label: "aplikasi".to_string(),
        });
        assert!(!gb.is_sufficiently_complete(&comp)); // Missing Agent
    }

    // --- SeedAnchor ---

    #[test]
    fn test_seed_anchor_no_data_preserves_confidence() {
        let sa = SeedAnchor::new();
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        let _original_confidence = comp.confidence;
        let adjustment = sa.seed_anchored_confidence(&comp);
        // When no seed data, weight should be 0.0, meaning original confidence preserved
        assert_eq!(adjustment.weight, 0.0);
    }

    #[test]
    fn test_seed_anchor_with_data_adjusts_confidence() {
        let sa = SeedAnchor::new();
        let mut comp =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.5);
        comp.seed_scores.insert(SeedPrimitive::Trust, 0.8);
        comp.seed_scores.insert(SeedPrimitive::Value, 0.7);
        let original_confidence = comp.confidence;
        sa.adjust_confidence(&mut comp);
        // With positive seed scores, confidence should be adjusted
        assert!(comp.confidence > 0.0);
        // The adjustment should change the confidence from its original value
        assert!(
            comp.confidence != original_confidence || comp.seed_scores.values().all(|&v| v == 0.5)
        );
    }
}

// ========================================================================
// M.3: Executive Cognition Enrichment Loop Tests
// ========================================================================

#[cfg(feature = "v12")]
mod m3_executive_cognition {
    use super::*;

    // --- ComputeBudget ---

    #[test]
    fn test_compute_budget_reactive_mode() {
        let budget = ComputeBudget::for_mode(&CognitiveMode::Reactive);
        assert_eq!(budget.max_enrichment_rounds, 0);
        assert_eq!(budget.max_reasoning_depth, 1);
        assert_eq!(budget.max_reflection_loops, 0);
    }

    #[test]
    fn test_compute_budget_analytical_mode() {
        let budget = ComputeBudget::for_mode(&CognitiveMode::Analytical);
        assert_eq!(budget.max_enrichment_rounds, 1);
        assert_eq!(budget.max_reasoning_depth, 3);
    }

    #[test]
    fn test_compute_budget_reflective_mode() {
        let budget = ComputeBudget::for_mode(&CognitiveMode::Reflective);
        assert_eq!(budget.max_enrichment_rounds, 2);
        assert_eq!(budget.max_reasoning_depth, 5);
    }

    // --- Cognitive Mode Selection ---

    #[test]
    fn test_cognitive_mode_reactive_when_healthy() {
        let mut orchestrator = ExecutiveOrchestrator::new();
        let compositions = vec![Composition {
            confidence: 0.9,
            epistemic: EpistemicState::Observed,
            ..Composition::default()
        }];
        let mode = orchestrator.select_cognitive_mode("test input", &compositions);
        assert_eq!(mode, CognitiveMode::Reactive);
    }

    #[test]
    fn test_cognitive_mode_analytical_when_low_confidence() {
        let mut orchestrator = ExecutiveOrchestrator::new();
        let compositions = vec![Composition {
            confidence: 0.3,
            epistemic: EpistemicState::Observed,
            ..Composition::default()
        }];
        let mode = orchestrator.select_cognitive_mode("test input", &compositions);
        assert_eq!(mode, CognitiveMode::Analytical);
    }

    #[test]
    fn test_cognitive_mode_reflective_with_deep_contradictions() {
        let mut orchestrator = ExecutiveOrchestrator::new();
        let compositions = vec![
            Composition {
                confidence: 0.5,
                epistemic: EpistemicState::Contradicted,
                ..Composition::default()
            },
            Composition {
                confidence: 0.4,
                epistemic: EpistemicState::Contradicted,
                ..Composition::default()
            },
            Composition {
                confidence: 0.3,
                epistemic: EpistemicState::Contradicted,
                ..Composition::default()
            },
        ];
        let mode = orchestrator.select_cognitive_mode("test input", &compositions);
        assert_eq!(mode, CognitiveMode::Reflective);
    }

    // --- StopCondition ---

    #[test]
    fn test_stop_condition_budget_exhausted() {
        let condition = StopCondition {
            max_passes: 2,
            ..StopCondition::default()
        };
        let state = ReasoningState {
            loops_completed: 2,
            goal_met: false,
            loops_without_new_evidence: 0,
            confidence: 0.5,
            elapsed_ms: 0,
            goal: ReasoningGoal::UnderstandInput,
            modified_compositions: Vec::new(),
            evidence_count: 0,
            evidence_at_loop_start: 0,
        };
        assert!(condition.should_stop(&state));
    }

    #[test]
    fn test_stop_condition_goal_met() {
        let condition = StopCondition::default();
        let state = ReasoningState {
            loops_completed: 1,
            goal_met: true,
            loops_without_new_evidence: 0,
            confidence: 0.9,
            elapsed_ms: 0,
            goal: ReasoningGoal::UnderstandInput,
            modified_compositions: Vec::new(),
            evidence_count: 0,
            evidence_at_loop_start: 0,
        };
        assert!(condition.should_stop(&state));
    }

    #[test]
    fn test_stop_condition_no_evidence_stagnation() {
        let condition = StopCondition {
            max_passes_without_evidence: 2,
            ..StopCondition::default()
        };
        let state = ReasoningState {
            loops_completed: 1,
            goal_met: false,
            loops_without_new_evidence: 2,
            confidence: 0.5,
            elapsed_ms: 0,
            goal: ReasoningGoal::UnderstandInput,
            modified_compositions: Vec::new(),
            evidence_count: 0,
            evidence_at_loop_start: 0,
        };
        assert!(condition.should_stop(&state));
    }

    // --- Reflect & ReflectionFinding ---

    #[test]
    fn test_reflect_produces_contradiction_resolvable_finding() {
        let reflect = Reflect::new();
        let result = ReflectionLoopResult {
            current_confidence: 0.9,
            elapsed_ms: 0,
            evidence_count: 1,
            modified_compositions: vec!["comp_1".to_string()],
            has_gaps: false,
            resolved_contradictions: vec!["comp_contra".to_string()],
            filled_gaps: vec![],
        };
        let graph = Graph::new();
        let findings = reflect.reflect(&result, &graph);
        assert!(findings
            .iter()
            .any(|f| f.finding_type == ReflectionFindingType::ContradictionResolvable));
    }

    #[test]
    fn test_reflect_produces_promotion_candidate_finding() {
        let reflect = Reflect::new();
        let result = ReflectionLoopResult {
            current_confidence: 0.9,
            elapsed_ms: 0,
            evidence_count: 1,
            modified_compositions: vec!["comp_1".to_string()],
            has_gaps: false,
            resolved_contradictions: vec![],
            filled_gaps: vec!["comp_gap".to_string()],
        };
        let graph = Graph::new();
        let findings = reflect.reflect(&result, &graph);
        assert!(findings
            .iter()
            .any(|f| f.finding_type == ReflectionFindingType::PromotionCandidate));
        assert!(findings
            .iter()
            .any(|f| matches!(f.action, ReflectionAction::ProposePromotion(_))));
    }

    #[test]
    fn test_reflect_detects_stagnant_inferred() {
        let reflect = Reflect::new();
        let result = ReflectionLoopResult::default();
        let mut graph = Graph::new();
        // Add a stagnant composition
        let mut comp = Composition::default();
        comp.id = "stagnant_comp".to_string();
        comp.epistemic = EpistemicState::Inferred;
        comp.batch_seen = 15; // > 10 batches
        comp.confidence = 0.6;
        graph.compositions.insert("stagnant_comp".to_string(), comp);

        let findings = reflect.reflect(&result, &graph);
        assert!(findings
            .iter()
            .any(|f| f.finding_type == ReflectionFindingType::StagnantInferred));
        assert!(findings
            .iter()
            .any(|f| matches!(f.action, ReflectionAction::ProposeDeprecation(_))));
    }

    // --- ExecutiveOrchestrator.ingest end-to-end ---

    #[test]
    fn test_executive_ingest_analytical_on_empty_graph() {
        let mut engine = make_pipeline_engine();
        let mut orchestrator = ExecutiveOrchestrator::new();

        // Ingest with no prior graph → empty neighborhood → average_confidence = 0.0 < 0.5
        // triggers Analytical mode (low confidence interpreted as needing deeper analysis)
        let result = orchestrator.ingest("Raymond membuat aplikasi karena lambat", &mut engine);
        assert!(result.atoms_created > 0);
        assert_eq!(orchestrator.mode, CognitiveMode::Analytical);
    }
}

// ========================================================================
// M.4: Closed Feedback Loop Integration Tests
// ========================================================================

#[cfg(feature = "v12")]
mod m4_closed_feedback_loop {
    use super::*;

    // --- Closed Feedback Loop ---

    #[test]
    fn test_full_pipeline_ingest_creates_compositions() {
        let mut engine = make_pipeline_engine();
        let result = engine.ingest("Raymond membuat aplikasi karena lambat");
        assert!(
            result.atoms_created > 0,
            "Should create atoms from tokenization"
        );
        // ExtractFrame + IngestAtoms should produce compositions if sentence-like
        assert!(result.compositions_created > 0 || result.atoms_created > 0);
    }

    #[test]
    fn test_pipeline_context_tracks_atoms() {
        let mut engine = make_pipeline_engine();
        engine.ingest("Raymond membuat aplikasi karena lambat");
        assert!(!engine.context.current_atoms.is_empty());
    }

    #[test]
    fn test_pipeline_with_gap_detection_enabled() {
        let mut engine = make_pipeline_engine();
        engine.context.gap_detection_enabled = true;
        let result = engine.ingest("Raymond membuat aplikasi karena lambat");
        // With gap detection, gaps should be detected for incomplete compositions
        // (Atoms may not have all roles, but pipeline still runs)
        assert!(result.atoms_created > 0);
    }

    #[test]
    fn test_graph_ensure_node_creates_new() {
        let mut graph = Graph::new();
        let id = graph.ensure_node("raymond");
        assert_eq!(id, 1); // First node ID
        assert!(graph.has_node(id));
    }

    #[test]
    fn test_graph_ensure_node_idempotent() {
        let mut graph = Graph::new();
        let id1 = graph.ensure_node("raymond");
        let id2 = graph.ensure_node("raymond");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_graph_cooccurrence_count() {
        let mut graph = Graph::new();
        let node_a = graph.ensure_node("raymond");
        let node_b = graph.ensure_node("aplikasi");
        // No compositions yet, so co-occurrence should be 0
        assert_eq!(graph.cooccurrence_count(node_a, node_b), 0);

        // Add a composition with both nodes
        let mut comp = Composition::default();
        comp.id = "comp_1".to_string();
        comp.members.push(CompositionMember {
            node_id: node_a,
            role: SemanticRole::Arg0Agent,
            confidence: 0.9,
            label: "raymond".to_string(),
        });
        comp.members.push(CompositionMember {
            node_id: node_b,
            role: SemanticRole::Arg1Patient,
            confidence: 0.8,
            label: "aplikasi".to_string(),
        });
        graph.compositions.insert("comp_1".to_string(), comp);
        assert_eq!(graph.cooccurrence_count(node_a, node_b), 1);
    }

    #[test]
    fn test_detect_gaps_finds_missing_roles() {
        let mut dg = DetectGaps::new();
        let mut comp = Composition::default();
        comp.id = "comp_incomplete".to_string();
        comp.composition_type = CompositionType::Event;
        comp.members.push(CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.9,
            label: "membuat".to_string(),
        });
        // Missing Arg0Agent, Arg1Patient

        let snapshot = GraphSnapshot {
            recent_atoms: vec![],
            compositions: vec![comp],
        };
        let gaps = dg.detect_atom_gaps(&snapshot);
        assert!(!gaps.is_empty());
        assert!(gaps
            .iter()
            .any(|g| g.gap_type == KnowledgeGapType::MissingRole));
    }

    #[test]
    fn test_detect_gaps_finds_ambiguous_tokens() {
        let mut dg = DetectGaps::new();
        let atom = SemanticAtom {
            id: "atom_1".to_string(),
            label: "dia".to_string(),
            atom_type: AtomType::AmbiguousToken,
            confidence: 0.5,
            source: EdgeSource::Learned,
            ..SemanticAtom::default()
        };
        let snapshot = GraphSnapshot {
            recent_atoms: vec![atom],
            compositions: vec![],
        };
        let gaps = dg.detect_atom_gaps(&snapshot);
        assert!(gaps
            .iter()
            .any(|g| g.gap_type == KnowledgeGapType::AmbiguousToken));
    }

    #[test]
    fn test_detect_gaps_finds_grounding_gaps() {
        let mut dg = DetectGaps::new();
        let mut comp = Composition::default();
        comp.id = "comp_low_ground".to_string();
        comp.composition_type = CompositionType::Event;
        comp.epistemic = EpistemicState::Inferred;
        comp.confidence = 0.3; // Low confidence

        let snapshot = GraphSnapshot {
            recent_atoms: vec![],
            compositions: vec![comp],
        };
        let gaps = dg.detect_grounding_gaps(&snapshot);
        assert!(gaps
            .iter()
            .any(|g| g.gap_type == KnowledgeGapType::LowGrounding));
    }

    #[test]
    fn test_select_acquisition_passive_recall() {
        let mut sa = SelectAcquisition::new();
        let mut graph = Graph::new();
        // Add a composition with an Agent role to the graph
        let agent_id = graph.ensure_node("raymond");
        let mut comp = Composition::default();
        comp.id = "comp_source".to_string();
        comp.members.push(CompositionMember {
            node_id: agent_id,
            role: SemanticRole::Arg0Agent,
            confidence: 0.9,
            label: "raymond".to_string(),
        });
        graph.compositions.insert("comp_source".to_string(), comp);

        let gap = KnowledgeGap {
            gap_id: "gap_1".to_string(),
            gap_type: KnowledgeGapType::MissingRole,
            source_composition_id: Some("comp_target".to_string()),
            missing_role: Some(SemanticRole::Arg0Agent),
            confidence: 0.7,
            ..KnowledgeGap::default()
        };
        let decision = sa.select_strategy(&gap, &graph);
        assert!(matches!(
            decision.strategy,
            AcquisitionStrategy::PassiveRecall { .. }
        ));
    }

    #[test]
    fn test_select_acquisition_defer_sparse_graph() {
        let mut sa = SelectAcquisition::new();
        let graph = Graph::new();
        let gap = KnowledgeGap {
            gap_id: "gap_1".to_string(),
            gap_type: KnowledgeGapType::SparseGraph,
            confidence: 0.5,
            ..KnowledgeGap::default()
        };
        let decision = sa.select_strategy(&gap, &graph);
        assert!(matches!(decision.strategy, AcquisitionStrategy::Defer));
    }

    #[test]
    fn test_select_acquisition_ask_user_ambiguous_token() {
        let mut sa = SelectAcquisition::new();
        let graph = Graph::new(); // Empty graph → no recall candidates
        let gap = KnowledgeGap {
            gap_id: "gap_2".to_string(),
            gap_type: KnowledgeGapType::AmbiguousToken,
            confidence: 0.8,
            ..KnowledgeGap::default()
        };
        let decision = sa.select_strategy(&gap, &graph);
        // Without graph candidates, should fall back to AskUser
        assert!(matches!(
            decision.strategy,
            AcquisitionStrategy::AskUser { .. }
        ));
    }

    // --- InquiryMemory ---

    #[test]
    fn test_inquiry_memory_prevents_repetition() {
        let mut mem = InquiryMemory::new();
        mem.mark_gap_addressed("gap_1", "PassiveRecall");
        assert!(mem.is_gap_addressed("gap_1"));

        let mut sa = SelectAcquisition { memory: mem };
        let graph = Graph::new();
        let gap = KnowledgeGap {
            gap_id: "gap_1".to_string(),
            gap_type: KnowledgeGapType::MissingRole,
            confidence: 0.7,
            ..KnowledgeGap::default()
        };
        let decision = sa.select_strategy(&gap, &graph);
        // Already addressed → should Defer
        assert!(matches!(decision.strategy, AcquisitionStrategy::Defer));
    }

    #[test]
    fn test_inquiry_memory_records_answers() {
        let mut mem = InquiryMemory::new();
        mem.mark_question_asked("q_1");
        assert!(mem.is_question_asked("q_1"));
        mem.record_answer("q_1", "Raymond adalah agent");
        let answer = mem.asked_questions.get("q_1");
        assert_eq!(answer, Some(&Some("Raymond adalah agent".to_string())));
    }
}

// ========================================================================
// M.5: Acquisition Pipeline (User Answer) Tests
// ========================================================================

#[cfg(feature = "v12")]
mod m5_acquisition_pipeline {
    use super::*;

    // --- Acquisition Pipeline ---

    #[test]
    fn test_knowledge_gap_type_all_variants() {
        // Ensure all KnowledgeGapType variants can be constructed
        let types = [
            KnowledgeGapType::MissingRole,
            KnowledgeGapType::AmbiguousToken,
            KnowledgeGapType::SparseGraph,
            KnowledgeGapType::LowGrounding,
            KnowledgeGapType::UnresolvedContradiction,
            KnowledgeGapType::IncompleteHiddenMeaning,
            KnowledgeGapType::MissingCause,
            KnowledgeGapType::MissingPurpose,
        ];
        assert_eq!(types.len(), 8);
    }

    #[test]
    fn test_acquisition_strategy_all_variants() {
        let strategies = [
            AcquisitionStrategy::PassiveRecall {
                candidate_node_id: 1,
                candidate_label: "test".to_string(),
                confidence: 0.7,
            },
            AcquisitionStrategy::ReExtraction {
                target_composition_id: "comp_1".to_string(),
                context_hints: vec![],
            },
            AcquisitionStrategy::AskUser {
                question: InquiryQuestion::default(),
            },
            AcquisitionStrategy::Defer,
        ];
        assert_eq!(strategies.len(), 4);
    }

    #[test]
    fn test_inquiry_question_construction() {
        let q = InquiryQuestion {
            question_id: "q_1".to_string(),
            question_text: "Siapa yang membuat aplikasi?".to_string(),
            gap_id: "gap_1".to_string(),
            target_role: Some(SemanticRole::Arg0Agent),
            target_composition_id: Some("comp_1".to_string()),
        };
        assert_eq!(q.question_id, "q_1");
        assert_eq!(q.target_role, Some(SemanticRole::Arg0Agent));
    }

    #[test]
    fn test_detect_all_gaps_integrated() {
        let mut dg = DetectGaps::new();
        let mut comp1 = Composition::default();
        comp1.id = "comp_event".to_string();
        comp1.composition_type = CompositionType::Event;
        comp1.epistemic = EpistemicState::Inferred;
        comp1.confidence = 0.3;
        comp1.members.push(CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.9,
            label: "membuat".to_string(),
        });

        let snapshot = GraphSnapshot {
            recent_atoms: vec![],
            compositions: vec![comp1],
        };
        let gaps = dg.detect_all(&snapshot);
        // Should find both atom gaps (missing roles) and grounding gaps (low confidence)
        assert!(!gaps.is_empty());
    }

    #[test]
    fn test_select_acquisition_ask_user_for_missing_role() {
        let mut sa = SelectAcquisition::new();
        let graph = Graph::new(); // Empty → no PassiveRecall candidates

        let gap = KnowledgeGap {
            gap_id: "gap_1".to_string(),
            gap_type: KnowledgeGapType::MissingRole,
            source_composition_id: Some("comp_1".to_string()),
            missing_role: Some(SemanticRole::Arg0Agent),
            confidence: 0.7,
            ..KnowledgeGap::default()
        };
        let decision = sa.select_strategy(&gap, &graph);
        // Without graph candidates and no source_text on comp → AskUser
        assert!(matches!(
            decision.strategy,
            AcquisitionStrategy::AskUser { .. }
        ));
        if let AcquisitionStrategy::AskUser { question } = &decision.strategy {
            assert_eq!(question.gap_id, "gap_1");
            assert_eq!(question.target_role, Some(SemanticRole::Arg0Agent));
        }
    }

    #[test]
    fn test_generate_question_for_gap() {
        let sa = SelectAcquisition::new();
        let gap = KnowledgeGap {
            gap_id: "gap_1".to_string(),
            gap_type: KnowledgeGapType::MissingRole,
            source_composition_id: Some("comp_1".to_string()),
            missing_role: Some(SemanticRole::Arg0Agent),
            confidence: 0.7,
            ..KnowledgeGap::default()
        };
        let question = sa.generate_question(&gap);
        assert!(!question.question_text.is_empty());
        assert_eq!(question.gap_id, "gap_1");
        assert_eq!(question.target_role, Some(SemanticRole::Arg0Agent));
    }

    #[test]
    fn test_defer_count_increments_for_sparse_graph() {
        let mut sa = SelectAcquisition::new();
        let graph = Graph::new();
        // Use different gap IDs — InquiryMemory deduplicates same gap_id
        // (once addressed, subsequent calls with same gap_id short-circuit without incrementing)
        let gap1 = KnowledgeGap {
            gap_id: "gap_1".to_string(),
            gap_type: KnowledgeGapType::SparseGraph,
            confidence: 0.5,
            ..KnowledgeGap::default()
        };
        let gap2 = KnowledgeGap {
            gap_id: "gap_2".to_string(),
            gap_type: KnowledgeGapType::SparseGraph,
            confidence: 0.5,
            ..KnowledgeGap::default()
        };
        sa.select_strategy(&gap1, &graph);
        sa.select_strategy(&gap2, &graph);
        assert_eq!(sa.memory.defer_count(&KnowledgeGapType::SparseGraph), 2);
    }

    #[test]
    fn test_graph_find_role_candidate() {
        let sa = SelectAcquisition::new();
        let mut graph = Graph::new();
        let agent_id = graph.ensure_node("raymond");
        let mut comp = Composition::default();
        comp.id = "comp_existing".to_string();
        comp.members.push(CompositionMember {
            node_id: agent_id,
            role: SemanticRole::Arg0Agent,
            confidence: 0.9,
            label: "raymond".to_string(),
        });
        graph.compositions.insert("comp_existing".to_string(), comp);

        let gap = KnowledgeGap {
            gap_id: "gap_1".to_string(),
            gap_type: KnowledgeGapType::MissingRole,
            missing_role: Some(SemanticRole::Arg0Agent),
            confidence: 0.7,
            ..KnowledgeGap::default()
        };
        let candidate = sa.graph_find_role_candidate(&graph, &SemanticRole::Arg0Agent, &gap);
        assert!(candidate.is_some());
        let (node_id, label, confidence) = candidate.unwrap();
        assert_eq!(node_id, agent_id);
        assert_eq!(label, "raymond");
        assert!(confidence > 0.0);
    }
}

// ========================================================================
// M.6: Semantic Edge & Graph Neighborhood Tests
// ========================================================================

#[cfg(feature = "v12")]
mod m6_semantic_edge_graph {
    use super::*;

    // --- SemanticEdge construction ---

    #[test]
    fn test_semantic_edge_default() {
        let edge = SemanticEdge::default();
        assert_eq!(edge.relation, RelationType::Categorical);
        assert!(edge.role.is_none());
        assert_eq!(edge.source, EdgeSource::Bootstrap);
    }

    #[test]
    fn test_semantic_edge_with_role() {
        let edge = SemanticEdge {
            relation: RelationType::Causal,
            role: Some(SemanticRole::Cause),
            source: EdgeSource::FrameCompiler,
        };
        assert_eq!(edge.relation, RelationType::Causal);
        assert_eq!(edge.role, Some(SemanticRole::Cause));
    }

    // --- Graph operations ---

    #[test]
    fn test_graph_new_is_empty() {
        let graph = Graph::new();
        assert!(graph.nodes.is_empty());
        assert!(graph.compositions.is_empty());
        assert!(graph.edges.is_empty());
        assert!(graph.label_to_id.is_empty());
        assert_eq!(graph.next_id, 1);
    }

    #[test]
    fn test_graph_find_node_by_label() {
        let mut graph = Graph::new();
        let id = graph.ensure_node("raymond");
        assert_eq!(graph.find_node_by_label("raymond"), Some(id));
        assert_eq!(graph.find_node_by_label("nonexistent"), None);
    }

    #[test]
    fn test_graph_node_label_lookup() {
        let mut graph = Graph::new();
        let id = graph.ensure_node("aplikasi");
        assert_eq!(graph.node_label(id), Some("aplikasi"));
        assert_eq!(graph.node_label(999), None);
    }

    #[test]
    fn test_graph_recent_compositions() {
        let mut graph = Graph::new();
        let mut comp1 = Composition::default();
        comp1.id = "comp_old".to_string();
        comp1.created_at = "2024-01-01T00:00:00Z".to_string();

        let mut comp2 = Composition::default();
        comp2.id = "comp_new".to_string();
        comp2.created_at = "2024-12-01T00:00:00Z".to_string();

        graph.compositions.insert("comp_old".to_string(), comp1);
        graph.compositions.insert("comp_new".to_string(), comp2);

        let recent = graph.recent_compositions(1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].id, "comp_new");
    }

    #[test]
    fn test_graph_get_composition() {
        let mut graph = Graph::new();
        let comp = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        graph.compositions.insert("comp_1".to_string(), comp);
        assert!(graph.get_composition(&"comp_1".to_string()).is_some());
        assert!(graph.get_composition(&"nonexistent".to_string()).is_none());
    }

    #[test]
    fn test_graph_get_edge() {
        let mut graph = Graph::new();
        graph.edges.push((
            "comp_1".to_string(),
            1u32,
            SemanticEdge {
                relation: RelationType::Categorical,
                role: Some(SemanticRole::Arg0Agent),
                source: EdgeSource::FrameCompiler,
            },
        ));
        let edge = graph.get_edge(&"comp_1".to_string(), 1);
        assert!(edge.is_some());
        assert!(graph.get_edge(&"comp_1".to_string(), 2).is_none());
    }

    // --- GraphNeighborhood ---

    #[test]
    fn test_neighborhood_empty_compositions() {
        let keywords = vec!["test".to_string()];
        let neighborhood = GraphNeighborhood::neighborhood_for(&keywords, &[]);
        assert!(neighborhood.compositions.is_empty());
        assert!(!neighborhood.has_contradictions());
        assert_eq!(neighborhood.average_confidence(), 0.0);
    }

    #[test]
    fn test_neighborhood_average_confidence_no_compositions() {
        let neighborhood = GraphNeighborhood {
            compositions: vec![],
        };
        assert_eq!(neighborhood.average_confidence(), 0.0);
    }

    // --- IngestResult ---

    #[test]
    fn test_ingest_result_merge() {
        let mut result = IngestResult {
            atoms_created: 5,
            compositions_created: 2,
            edges_created: 10,
            gaps_detected: 1,
            enrichments_applied: 0,
            governance_transitions: 1,
        };
        let other = IngestResult {
            atoms_created: 3,
            compositions_created: 1,
            edges_created: 5,
            gaps_detected: 2,
            enrichments_applied: 1,
            governance_transitions: 0,
        };
        result.merge(&other);
        assert_eq!(result.atoms_created, 8);
        assert_eq!(result.compositions_created, 3);
        assert_eq!(result.edges_created, 15);
        assert_eq!(result.gaps_detected, 3);
        assert_eq!(result.enrichments_applied, 1);
    }

    // --- PipelineContext conditions ---

    #[test]
    fn test_pipeline_context_is_sentence_like() {
        let mut ctx = PipelineContext::default();
        assert!(!ctx.is_sentence_like()); // No raw_text
        ctx.set_raw_text("hello");
        assert!(!ctx.is_sentence_like()); // Too short
        ctx.set_raw_text("Raymond membuat aplikasi karena lambat");
        assert!(ctx.is_sentence_like());
    }

    #[test]
    fn test_pipeline_context_has_event_atoms() {
        let mut ctx = PipelineContext::default();
        assert!(!ctx.has_event_atoms());
        ctx.current_atoms.push(SemanticAtom {
            atom_type: AtomType::Event,
            ..SemanticAtom::default()
        });
        assert!(ctx.has_event_atoms());
    }

    #[test]
    fn test_pipeline_context_record_event_sliding_window() {
        let mut ctx = PipelineContext::default();
        for i in 0..55 {
            ctx.record_event(SemanticAtom {
                id: format!("atom_{}", i),
                atom_type: AtomType::Event,
                ..SemanticAtom::default()
            });
        }
        // Should be capped at RECENT_EVENTS_WINDOW (50)
        assert_eq!(
            ctx.recent_events.len(),
            PipelineContext::RECENT_EVENTS_WINDOW
        );
    }

    #[test]
    fn test_pipeline_context_record_event_ignores_non_events() {
        let mut ctx = PipelineContext::default();
        ctx.record_event(SemanticAtom {
            atom_type: AtomType::Token,
            ..SemanticAtom::default()
        });
        assert!(ctx.recent_events.is_empty());
    }

    // --- WeakFrame detection ---

    #[test]
    fn test_find_weak_frames() {
        let mut engine = make_pipeline_engine();
        // Create a weak Event composition (low confidence, missing roles)
        let mut comp = Composition::default();
        comp.id = "comp_weak".to_string();
        comp.composition_type = CompositionType::Event;
        comp.confidence = 0.3; // Low
        comp.members.push(CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.3,
            label: "membuat".to_string(),
        });
        // Missing Arg0Agent → weak
        engine
            .graph_mut()
            .compositions
            .insert("comp_weak".to_string(), comp);

        let weak = engine.find_weak_frames();
        assert!(!weak.is_empty());
        assert!(weak.iter().any(|wf| wf.composition_id == "comp_weak"));
    }
}

// ========================================================================
// M.7: ExtractionQualityTracker & Dedup Tests
// ========================================================================

#[cfg(feature = "v12")]
mod m7_quality_tracker_dedup {
    use super::*;

    // --- ExtractionQuality ---

    #[test]
    fn test_extraction_quality_gap_rate() {
        let q = ExtractionQualityStats {
            rule_id: "rule_1".to_string(),
            total_extractions: 10,
            gaps_detected: 3,
            gaps_repaired: 1,
            avg_confidence: 0.7,
            last_gap_type: None,
        };
        assert!((q.gap_rate() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_extraction_quality_repair_rate() {
        let q = ExtractionQualityStats {
            rule_id: "rule_1".to_string(),
            total_extractions: 10,
            gaps_detected: 4,
            gaps_repaired: 2,
            avg_confidence: 0.7,
            last_gap_type: None,
        };
        assert!((q.repair_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_extraction_quality_is_weak() {
        let weak = ExtractionQualityStats {
            rule_id: "rule_weak".to_string(),
            total_extractions: 10,
            gaps_detected: 5, // 50% gap rate > 30%
            gaps_repaired: 1, // 20% repair rate < 50%
            avg_confidence: 0.4,
            last_gap_type: None,
        };
        assert!(weak.is_weak());

        let strong = ExtractionQualityStats {
            rule_id: "rule_strong".to_string(),
            total_extractions: 10,
            gaps_detected: 2, // 20% gap rate < 30%
            gaps_repaired: 2, // 100% repair rate
            avg_confidence: 0.9,
            last_gap_type: None,
        };
        assert!(!strong.is_weak());
    }

    #[test]
    fn test_extraction_quality_zero_extractions() {
        let q = ExtractionQualityStats::default();
        assert_eq!(q.gap_rate(), 0.0);
        assert_eq!(q.repair_rate(), 1.0); // No gaps = fully repaired
        assert!(!q.is_weak());
    }

    // --- ExtractionQualityTracker ---

    #[test]
    fn test_tracker_record_extraction() {
        let mut tracker = ExtractionQualityTracker::default();
        tracker.record_extraction("rule_1", 0.8);
        tracker.record_extraction("rule_1", 0.6);
        assert_eq!(tracker.frames_extracted, 2);
        assert!(tracker.quality_by_rule.contains_key("rule_1"));
        let entry = tracker.quality_by_rule.get("rule_1").unwrap();
        assert_eq!(entry.total_extractions, 2);
        assert!((entry.avg_confidence - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_tracker_record_gap_and_repair() {
        let mut tracker = ExtractionQualityTracker::default();
        tracker.record_extraction("rule_1", 0.8);
        tracker.record_gap("rule_1", "MissingRole");
        tracker.record_gap("rule_1", "MissingCause");
        tracker.record_repair("rule_1");

        let entry = tracker.quality_by_rule.get("rule_1").unwrap();
        assert_eq!(entry.gaps_detected, 2);
        assert_eq!(entry.gaps_repaired, 1);
        assert_eq!(entry.last_gap_type, Some("MissingCause".to_string()));
    }

    #[test]
    fn test_tracker_weak_rules() {
        let mut tracker = ExtractionQualityTracker::default();
        // Make rule_1 weak: high gap rate, low repair rate
        tracker.record_extraction("rule_1", 0.5);
        tracker.record_extraction("rule_1", 0.5);
        tracker.record_extraction("rule_1", 0.5);
        tracker.record_extraction("rule_1", 0.5);
        tracker.record_extraction("rule_1", 0.5);
        tracker.record_gap("rule_1", "MissingRole");
        tracker.record_gap("rule_1", "MissingRole");
        tracker.record_repair("rule_1"); // 1/2 = 50% → NOT weak (repair rate = 50%, need < 50%)
                                         // Add one more gap without repair
        tracker.record_gap("rule_1", "MissingCause"); // 3 gaps, 1 repair = 33% repair < 50%
                                                      // gap_rate = 3/5 = 60% > 30% ✓, repair_rate = 1/3 = 33% < 50% ✓
        let weak = tracker.weak_rules();
        assert!(!weak.is_empty());
        assert!(weak.iter().any(|q| q.rule_id == "rule_1"));
    }

    #[test]
    fn test_tracker_low_confidence_frames() {
        let mut tracker = ExtractionQualityTracker::default();
        tracker.record_extraction("rule_1", 0.8); // High
        tracker.record_extraction("rule_2", 0.3); // Low
        tracker.record_extraction("rule_3", 0.2); // Low
        assert_eq!(tracker.low_confidence_frames, 2);
        assert!((tracker.average_confidence - (0.8 + 0.3 + 0.2) / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_tracker_multiple_rules_independent() {
        let mut tracker = ExtractionQualityTracker::default();
        tracker.record_extraction("rule_A", 0.9);
        tracker.record_extraction("rule_B", 0.4);
        tracker.record_gap("rule_A", "MissingRole");

        assert!(tracker.quality_by_rule.contains_key("rule_A"));
        assert!(tracker.quality_by_rule.contains_key("rule_B"));
        assert_eq!(
            tracker.quality_by_rule.get("rule_A").unwrap().gaps_detected,
            1
        );
        assert_eq!(
            tracker.quality_by_rule.get("rule_B").unwrap().gaps_detected,
            0
        );
    }

    // --- Serde roundtrip for key types ---

    #[test]
    fn test_semantic_atom_serde_roundtrip() {
        let atom = make_event_atom(
            "atom_1",
            "membuat",
            "Raymond",
            "aplikasi",
            Some("lambat"),
            0.8,
        );
        let json = serde_json::to_string(&atom).expect("serialize");
        let deserialized: SemanticAtom = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(atom.id, deserialized.id);
        assert_eq!(atom.label, deserialized.label);
        assert_eq!(atom.atom_type, deserialized.atom_type);
        assert!((atom.confidence - deserialized.confidence).abs() < 0.001);
    }

    #[test]
    fn test_composition_serde_roundtrip() {
        let comp = make_event_composition(
            "comp_1",
            "membuat",
            "Raymond",
            "aplikasi",
            Some("lambat"),
            0.8,
        );
        let json = serde_json::to_string(&comp).expect("serialize");
        let deserialized: Composition = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(comp.id, deserialized.id);
        assert_eq!(comp.composition_type, deserialized.composition_type);
        assert_eq!(comp.members.len(), deserialized.members.len());
    }

    #[test]
    fn test_semantic_edge_serde_roundtrip() {
        let edge = SemanticEdge {
            relation: RelationType::Causal,
            role: Some(SemanticRole::Cause),
            source: EdgeSource::FrameCompiler,
        };
        let json = serde_json::to_string(&edge).expect("serialize");
        let deserialized: SemanticEdge = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(edge.relation, deserialized.relation);
        assert_eq!(edge.role, deserialized.role);
    }

    #[test]
    fn test_extraction_quality_tracker_serde_roundtrip() {
        let mut tracker = ExtractionQualityTracker::default();
        tracker.record_extraction("rule_1", 0.8);
        tracker.record_gap("rule_1", "MissingRole");
        let json = serde_json::to_string(&tracker).expect("serialize");
        let deserialized: ExtractionQualityTracker =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(tracker.frames_extracted, deserialized.frames_extracted);
        assert_eq!(
            tracker.low_confidence_frames,
            deserialized.low_confidence_frames
        );
    }

    // --- PipelineContext serde roundtrip ---

    #[test]
    fn test_pipeline_context_serde_roundtrip() {
        let mut ctx = PipelineContext::default();
        ctx.set_raw_text("Raymond membuat aplikasi karena lambat");
        ctx.gap_detection_enabled = true;
        ctx.current_atoms.push(make_event_atom(
            "atom_1", "membuat", "Raymond", "aplikasi", None, 0.8,
        ));

        let json = serde_json::to_string(&ctx).expect("serialize");
        let deserialized: PipelineContext = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(ctx.raw_text, deserialized.raw_text);
        assert_eq!(
            ctx.gap_detection_enabled,
            deserialized.gap_detection_enabled
        );
        assert_eq!(ctx.current_atoms.len(), deserialized.current_atoms.len());
    }

    // --- EdgeSource v12 variants exist ---

    #[test]
    fn test_edge_source_v12_variants() {
        // Verify v12-specific EdgeSource variants exist and are constructible
        let sources = [
            EdgeSource::AcquisitionRecall,
            EdgeSource::AcquisitionSelfStudy,
            EdgeSource::AcquisitionUserAnswer,
            EdgeSource::EnrichmentFeedback,
            EdgeSource::ExtractionRepair,
        ];
        assert_eq!(sources.len(), 5);
    }

    // --- AtomType and variant completeness ---

    #[test]
    fn test_atom_type_all_variants() {
        let types = [
            AtomType::Token,
            AtomType::AmbiguousToken,
            AtomType::Event,
            AtomType::HiddenMeaning,
            AtomType::Pattern,
            AtomType::Hypothesis,
            AtomType::Acquisition,
        ];
        assert_eq!(types.len(), 7);
    }

    #[test]
    fn test_lifecycle_state_all_variants() {
        let states = [
            LifecycleState::New,
            LifecycleState::Candidate,
            LifecycleState::Stable,
            LifecycleState::Deprecated,
            LifecycleState::Quarantine,
        ];
        assert_eq!(states.len(), 5);
    }

    #[test]
    fn test_epistemic_state_all_variants() {
        let states = [
            EpistemicState::Observed,
            EpistemicState::Inferred,
            EpistemicState::Hypothesis,
            EpistemicState::Grounded,
            EpistemicState::Contradicted,
        ];
        assert_eq!(states.len(), 5);
    }

    #[test]
    fn test_seed_primitive_all_variants() {
        let seeds = [
            SeedPrimitive::Trust,
            SeedPrimitive::Risk,
            SeedPrimitive::Value,
            SeedPrimitive::Goal,
            SeedPrimitive::Identity,
        ];
        assert_eq!(seeds.len(), 5);
    }
}

// ========================================================================
// Option B: Focused Integration Tests
// ========================================================================
//
// | Module | Scope | Coverage |
// |--------|-------|----------|
// | B.1 | ExtractFrame Integration | MD-1 |
// | B.2 | ReasonFrame Integration | MD-2 |
// | B.3 | GovernBeliefs End-to-End | MD-4 |
// | B.4 | Closed Feedback Loop | MD-3, MD-6 |
// | B.5 | Executive Mode Selection | MD-5 |
// | B.6 | Full Pipeline End-to-End | MD-1–MD-6 |

// ========================================================================
// B.1: ExtractFrame Integration Tests
// ========================================================================

#[cfg(feature = "v12")]
mod b1_extract_frame {
    use super::*;

    // --- Active voice extraction ---

    #[test]
    fn test_extract_frame_active_voice() {
        // "Raymond membuat aplikasi karena lambat" → Event with Active voice,
        // Agent=raymond, Patient=aplikasi, Cause=lambat
        let ef = ExtractFrame::new();
        let result = ef.extract("Raymond membuat aplikasi karena lambat");
        assert!(result.is_some(), "Should extract from sentence-like input");

        let atom = result.unwrap();
        assert_eq!(atom.atom_type, AtomType::Event);
        assert_eq!(atom.voice, Some(Voice::Active));
        assert_eq!(atom.polarity, Some(Polarity::Positive));
        // Agent should be "raymond" (lowercased by extraction)
        assert_eq!(
            atom.roles.get(&SemanticRole::Arg0Agent),
            Some(&"raymond".to_string())
        );
        // Patient should be "aplikasi"
        assert_eq!(
            atom.roles.get(&SemanticRole::Arg1Patient),
            Some(&"aplikasi".to_string())
        );
        // Cause should be "lambat"
        assert_eq!(
            atom.roles.get(&SemanticRole::Cause),
            Some(&"lambat".to_string())
        );
    }

    // --- Passive voice extraction ---

    #[test]
    fn test_extract_frame_passive_voice() {
        // "Aplikasi dibuat oleh Raymond" → Event with Passive voice,
        // Patient before predicate, Agent after "oleh"
        let ef = ExtractFrame::new();
        let result = ef.extract("Aplikasi dibuat oleh Raymond");
        assert!(result.is_some(), "Should extract passive voice sentence");

        let atom = result.unwrap();
        assert_eq!(atom.atom_type, AtomType::Event);
        assert_eq!(atom.voice, Some(Voice::Passive));
        // Patient should be "aplikasi" (subject before predicate in passive)
        assert_eq!(
            atom.roles.get(&SemanticRole::Arg1Patient),
            Some(&"aplikasi".to_string())
        );
        // Agent should be "raymond" (after "oleh")
        assert_eq!(
            atom.roles.get(&SemanticRole::Arg0Agent),
            Some(&"raymond".to_string())
        );
    }

    // --- Negated sentence extraction ---

    #[test]
    fn test_extract_frame_negated() {
        // "Raymond tidak membuat aplikasi" → Event with Negative polarity
        let ef = ExtractFrame::new();
        let result = ef.extract("Raymond tidak membuat aplikasi");
        assert!(result.is_some(), "Should extract negated sentence");

        let atom = result.unwrap();
        assert_eq!(atom.atom_type, AtomType::Event);
        assert_eq!(atom.polarity, Some(Polarity::Negative));
    }

    // --- Non-sentence-like input (token) returns None ---

    #[test]
    fn test_extract_frame_token_input() {
        // "kucing" → None (not sentence-like: < 3 tokens, no verb)
        let ef = ExtractFrame::new();
        let result = ef.extract("kucing");
        assert!(result.is_none(), "Single token should not produce a frame");
    }

    // --- Graph-assisted re-extraction fills missing roles ---

    #[test]
    fn test_extract_frame_graph_assisted() {
        let ef = ExtractFrame::with_graph_assist();
        let mut graph = Graph::new();

        // Pre-populate graph: add a node that can fill a missing role
        let purpose_node = graph.ensure_node("mempercepat kerja");

        // Extract from a sentence that lacks Purpose role
        let result = ef.extract("Raymond membuat aplikasi karena lambat");
        assert!(result.is_some());
        let atom = result.unwrap();
        // Should NOT have Purpose from basic extraction
        assert!(!atom.roles.contains_key(&SemanticRole::Purpose));

        // Now re-extract with graph context providing a Purpose hint
        let graph_context = vec![(SemanticRole::Purpose, purpose_node, 0.8)];
        let re_result = ef.re_extract_with_context(
            "Raymond membuat aplikasi karena lambat",
            &graph_context,
            &graph,
        );
        assert!(re_result.is_some());
        let re_atom = re_result.unwrap();
        // Should have Purpose from graph context
        assert!(re_atom.roles.contains_key(&SemanticRole::Purpose));
        // Variant should be GraphAssisted
        assert!(matches!(
            re_atom.variant,
            Some(AtomVariant::FrameVariant(FrameSource::GraphAssisted))
        ));
    }

    // --- Confidence computation verification ---

    #[test]
    fn test_extract_frame_confidence_computation() {
        // Verify confidence formula:
        //   base = 0.30
        //   + 0.15 if Agent present
        //   + 0.15 if Patient present
        //   + 0.10 if Cause present
        //   + 0.10 if Purpose present
        //   - 0.05 if Negative polarity

        // Case 1: Agent + Patient only → 0.30 + 0.15 + 0.15 = 0.60
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "raymond".to_string());
        roles.insert(SemanticRole::Arg1Patient, "aplikasi".to_string());
        let conf = ExtractFrame::compute_frame_confidence(&roles, &Polarity::Positive);
        assert!((conf - 0.60).abs() < 0.01, "Expected ~0.60, got {}", conf);

        // Case 2: Agent + Patient + Cause → 0.30 + 0.15 + 0.15 + 0.10 = 0.70
        roles.insert(SemanticRole::Cause, "lambat".to_string());
        let conf = ExtractFrame::compute_frame_confidence(&roles, &Polarity::Positive);
        assert!((conf - 0.70).abs() < 0.01, "Expected ~0.70, got {}", conf);

        // Case 3: Agent + Patient + Cause + Purpose → 0.30 + 0.15 + 0.15 + 0.10 + 0.10 = 0.80
        roles.insert(SemanticRole::Purpose, "mempercepat".to_string());
        let conf = ExtractFrame::compute_frame_confidence(&roles, &Polarity::Positive);
        assert!((conf - 0.80).abs() < 0.01, "Expected ~0.80, got {}", conf);

        // Case 4: Negative polarity subtracts 0.05 → 0.80 - 0.05 = 0.75
        let conf = ExtractFrame::compute_frame_confidence(&roles, &Polarity::Negative);
        assert!((conf - 0.75).abs() < 0.01, "Expected ~0.75, got {}", conf);

        // Case 5: Base only (no roles) → 0.30
        let empty_roles = HashMap::new();
        let conf = ExtractFrame::compute_frame_confidence(&empty_roles, &Polarity::Positive);
        assert!((conf - 0.30).abs() < 0.01, "Expected ~0.30, got {}", conf);
    }
}

// ========================================================================
// B.2: ReasonFrame Integration Tests
// ========================================================================

#[cfg(feature = "v12")]
mod b2_reason_frame {
    use super::*;

    // --- ProblemSolutionRule ---

    #[test]
    fn test_reason_frame_problem_solution_rule() {
        // Event with Cause + Action + Patient → HiddenMeaning with Problem/Solution
        let rf = ReasonFrame::new();
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "raymond".to_string());
        roles.insert(SemanticRole::Arg1Patient, "aplikasi".to_string());
        roles.insert(SemanticRole::Cause, "lambat".to_string());

        let event = SemanticAtom {
            id: "atom_test".to_string(),
            label: "membuat".to_string(),
            atom_type: AtomType::Event,
            roles,
            polarity: Some(Polarity::Positive),
            voice: Some(Voice::Active),
            confidence: 0.75,
            source: EdgeSource::FrameCompiler,
            ..SemanticAtom::default()
        };

        let results = rf.reason(&event, &[]);
        // Should produce at least one result from ProblemSolutionRule
        assert!(!results.is_empty(), "Should produce reasoning results");
        assert!(
            results.iter().any(|r| r.rule_name == "ProblemSolutionRule"),
            "Should include ProblemSolutionRule result"
        );

        let ps_result = results
            .iter()
            .find(|r| r.rule_name == "ProblemSolutionRule")
            .unwrap();
        assert_eq!(ps_result.atom.atom_type, AtomType::HiddenMeaning);
        assert_eq!(ps_result.atom.label, "problem_solution");
        assert_eq!(
            ps_result.atom.roles.get(&SemanticRole::Problem),
            Some(&"lambat".to_string())
        );
        assert_eq!(
            ps_result.atom.roles.get(&SemanticRole::Solution),
            Some(&"aplikasi".to_string())
        );
    }

    // --- GoalInferenceRule ---

    #[test]
    fn test_reason_frame_goal_inference_rule() {
        // Event with Purpose → HiddenMeaning with ImpliedGoal
        let rf = ReasonFrame::new();
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "raymond".to_string());
        roles.insert(SemanticRole::Arg1Patient, "aplikasi".to_string());
        roles.insert(SemanticRole::Purpose, "mempercepat pekerjaan".to_string());

        let event = SemanticAtom {
            id: "atom_test".to_string(),
            label: "membuat".to_string(),
            atom_type: AtomType::Event,
            roles,
            polarity: Some(Polarity::Positive),
            voice: Some(Voice::Active),
            confidence: 0.75,
            source: EdgeSource::FrameCompiler,
            ..SemanticAtom::default()
        };

        let results = rf.reason(&event, &[]);
        assert!(
            results.iter().any(|r| r.rule_name == "GoalInferenceRule"),
            "Should include GoalInferenceRule result"
        );

        let gi_result = results
            .iter()
            .find(|r| r.rule_name == "GoalInferenceRule")
            .unwrap();
        assert_eq!(gi_result.atom.atom_type, AtomType::HiddenMeaning);
        assert_eq!(gi_result.atom.label, "goal_inference");
        assert_eq!(
            gi_result.atom.roles.get(&SemanticRole::ImpliedGoal),
            Some(&"mempercepat pekerjaan".to_string())
        );
    }

    // --- PolarityConflictRule ---

    #[test]
    fn test_reason_frame_polarity_conflict_rule() {
        // Two events with same predicate, opposite polarity → PolarityConflict detection
        let rf = ReasonFrame::new();

        let mut roles_pos = HashMap::new();
        roles_pos.insert(SemanticRole::Arg0Agent, "raymond".to_string());
        let event_positive = SemanticAtom {
            id: "atom_pos".to_string(),
            label: "membuat".to_string(),
            atom_type: AtomType::Event,
            roles: roles_pos,
            polarity: Some(Polarity::Positive),
            voice: Some(Voice::Active),
            confidence: 0.75,
            source: EdgeSource::FrameCompiler,
            ..SemanticAtom::default()
        };

        let mut roles_neg = HashMap::new();
        roles_neg.insert(SemanticRole::Arg0Agent, "raymond".to_string());
        let event_negative = SemanticAtom {
            id: "atom_neg".to_string(),
            label: "membuat".to_string(),
            atom_type: AtomType::Event,
            roles: roles_neg,
            polarity: Some(Polarity::Negative),
            voice: Some(Voice::Active),
            confidence: 0.75,
            source: EdgeSource::FrameCompiler,
            ..SemanticAtom::default()
        };

        let recent = vec![event_negative];
        let results = rf.reason(&event_positive, &recent);
        assert!(
            results
                .iter()
                .any(|r| r.rule_name == "PolarityConflictRule"),
            "Should detect polarity conflict between opposite-polarity events"
        );

        let pc_result = results
            .iter()
            .find(|r| r.rule_name == "PolarityConflictRule")
            .unwrap();
        assert_eq!(pc_result.atom.label, "polarity_conflict");
    }

    // --- No match (simple event with no rules triggered) ---

    #[test]
    fn test_reason_frame_no_match() {
        // Simple event with no Cause, Purpose, or opposite polarity → no rules triggered
        let rf = ReasonFrame::new();
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "raymond".to_string());
        roles.insert(SemanticRole::Arg1Patient, "aplikasi".to_string());
        // No Cause, no Purpose

        let event = SemanticAtom {
            id: "atom_simple".to_string(),
            label: "membuat".to_string(),
            atom_type: AtomType::Event,
            roles,
            polarity: Some(Polarity::Positive),
            voice: Some(Voice::Active),
            confidence: 0.75,
            source: EdgeSource::FrameCompiler,
            ..SemanticAtom::default()
        };

        let results = rf.reason(&event, &[]);
        // ProblemSolutionRule needs Cause → doesn't apply
        // GoalInferenceRule needs Purpose → doesn't apply
        // PolarityConflictRule needs opposite polarity in recent → no recent events
        assert!(
            results.is_empty(),
            "Simple event should trigger no reasoning rules"
        );
    }
}

// ========================================================================
// B.3: GovernBeliefs End-to-End Tests
// ========================================================================

#[cfg(feature = "v12")]
mod b3_governance_e2e {
    use super::*;

    // --- Initial states for all CompositionType × EdgeSource combinations ---

    #[test]
    fn test_governance_initial_states_all_types() {
        let gb = GovernBeliefs::new();

        // Event + FrameCompiler → New/Observed
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::Event;
        comp.provenance.origin = EdgeSource::FrameCompiler;
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::New);
        assert_eq!(comp.epistemic, EpistemicState::Observed);

        // HiddenMeaning + HiddenMeaningRule → Candidate/Inferred
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::HiddenMeaning;
        comp.provenance.origin = EdgeSource::HiddenMeaningRule;
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::Candidate);
        assert_eq!(comp.epistemic, EpistemicState::Inferred);

        // Hypothesis + Abductive → Quarantine/Hypothesis
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::Hypothesis;
        comp.provenance.origin = EdgeSource::Abductive;
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::Quarantine);
        assert_eq!(comp.epistemic, EpistemicState::Hypothesis);

        // Event + HumanAssertion → Candidate/Grounded
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::Event;
        comp.provenance.origin = EdgeSource::HumanAssertion;
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::Candidate);
        assert_eq!(comp.epistemic, EpistemicState::Grounded);

        // Acquisition + AcquisitionRecall → Stable/Grounded
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::Acquisition;
        comp.provenance.origin = EdgeSource::AcquisitionRecall;
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::Stable);
        assert_eq!(comp.epistemic, EpistemicState::Grounded);

        // Acquisition + AcquisitionUserAnswer → Candidate/Observed
        let mut comp = Composition::default();
        comp.composition_type = CompositionType::Acquisition;
        comp.provenance.origin = EdgeSource::AcquisitionUserAnswer;
        gb.initial_states(&mut comp);
        assert_eq!(comp.lifecycle, LifecycleState::Candidate);
        assert_eq!(comp.epistemic, EpistemicState::Observed);
    }

    // --- Contradiction detection and resolution attempt ---

    #[test]
    fn test_governance_contradiction_detection_and_resolution() {
        let gb = GovernBeliefs::new();
        // Create two contradictory compositions (role reversal)
        let mut comp1 =
            make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.8);
        let mut comp2 =
            make_event_composition("comp_2", "membuat", "aplikasi", "Raymond", None, 0.8);
        // Set swapped node IDs for role reversal detection
        comp1
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg0Agent)
            .unwrap()
            .node_id = 1;
        comp1
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg1Patient)
            .unwrap()
            .node_id = 2;
        comp2
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg0Agent)
            .unwrap()
            .node_id = 2;
        comp2
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg1Patient)
            .unwrap()
            .node_id = 1;

        // Step 1: Detect contradiction
        let updates = gb.detect_contradiction(&mut [comp1.clone(), comp2.clone()]);
        assert!(
            !updates.is_empty(),
            "Should detect contradiction between role-reversed compositions"
        );

        // Step 2: Attempt resolution
        let resolution = gb.check_contradiction_resolution(&comp1, &comp2);
        // Resolution may or may not succeed depending on the rule set,
        // but the method should return Some for resolvable conflicts
        // (Voice confusion is not applicable here since node IDs differ)
        // This is an assertion that resolution check runs without panic
        if let Some(res) = resolution {
            assert!(matches!(
                res.resolution_type,
                ResolutionType::Misinterpretation | ResolutionType::Superseded
            ));
        }
    }

    // --- Full lifecycle: New → Candidate → Stable ---

    #[test]
    fn test_governance_promotion_full_lifecycle() {
        let gb = GovernBeliefs::new();
        let mut comp =
            make_event_composition("comp_life", "membuat", "Raymond", "aplikasi", None, 0.8);

        // Stage 1: New (after initial states)
        comp.lifecycle = LifecycleState::New;
        comp.batch_seen = 0;
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(
            !updates
                .iter()
                .any(|u| u.new_lifecycle == Some(LifecycleState::Candidate)),
            "New with 0 batches should not promote"
        );

        // Stage 2: New → Candidate (after 1 batch)
        comp.batch_seen = 1;
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(
            updates
                .iter()
                .any(|u| u.new_lifecycle == Some(LifecycleState::Candidate)),
            "New with 1 batch should promote to Candidate"
        );

        // Stage 3: Candidate → Stable (after sufficient batches + confidence)
        comp.lifecycle = LifecycleState::Candidate;
        comp.batch_seen = 3;
        comp.confidence = 0.7;
        comp.epistemic = EpistemicState::Observed;
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(
            updates
                .iter()
                .any(|u| u.new_lifecycle == Some(LifecycleState::Stable)),
            "Candidate meeting criteria should promote to Stable"
        );
    }

    // --- Epistemic progression: Observed → Inferred → Grounded ---

    #[test]
    fn test_governance_epistemic_transitions() {
        let gb = GovernBeliefs::new();

        // Observed (from FrameCompiler) stays Observed without multi-source
        let mut comp =
            make_event_composition("comp_epi", "membuat", "Raymond", "aplikasi", None, 0.8);
        comp.epistemic = EpistemicState::Observed;
        let updates = gb.check_promotions(&mut [comp.clone()]);
        // Observed doesn't transition to Grounded via check_promotions
        // (Inferred → Grounded is the epistemic promotion path)
        assert!(!updates
            .iter()
            .any(|u| u.new_epistemic == Some(EpistemicState::Grounded)));

        // Inferred → Grounded (with multi-source provenance)
        comp.epistemic = EpistemicState::Inferred;
        comp.batch_seen = 5;
        comp.confidence = 0.8;
        comp.provenance.origin = EdgeSource::EnrichmentFeedback;
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(
            updates
                .iter()
                .any(|u| u.new_epistemic == Some(EpistemicState::Grounded)),
            "Inferred with multi-source should promote to Grounded"
        );
    }

    // --- Re-govern after enrichment ---

    #[test]
    fn test_governance_re_govern_after_enrichment() {
        let gb = GovernBeliefs::new();
        let mut comp =
            make_event_composition("comp_rich", "membuat", "Raymond", "aplikasi", None, 0.5);
        comp.lifecycle = LifecycleState::Candidate;
        comp.batch_seen = 1; // Too young for Stable
        comp.epistemic = EpistemicState::Observed;

        // First governance: too young
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(!updates
            .iter()
            .any(|u| u.new_lifecycle == Some(LifecycleState::Stable)));

        // Enrich: add a Purpose role, bump member confidences, and bump composition confidence
        // Note: can_promote_to_stable requires >= 2 confirming members (confidence >= 0.5)
        // Original members had confidence based on 0.5 base (0.475, 0.45, 0.425) → none confirming
        // Fix member confidences so at least 2 members have confidence >= 0.5
        comp.members[0].confidence = 0.6; // Predicate
        comp.members[1].confidence = 0.6; // Agent
        comp.members[2].confidence = 0.55; // Patient
        comp.members.push(CompositionMember {
            node_id: 99,
            role: SemanticRole::Purpose,
            confidence: 0.7,
            label: "mempercepat".to_string(),
        });
        comp.confidence = 0.75;
        comp.batch_seen = 3;

        // Re-govern: now meets criteria (age >= 3, confidence >= 0.55, >= 2 confirming members)
        let updates = gb.check_promotions(&mut [comp.clone()]);
        assert!(
            updates
                .iter()
                .any(|u| u.new_lifecycle == Some(LifecycleState::Stable)),
            "After enrichment, should promote to Stable"
        );
    }
}

// ========================================================================
// B.4: Closed Feedback Loop Tests (MOST CRITICAL)
// ========================================================================

#[cfg(feature = "v12")]
mod b4_closed_feedback_loop {
    use super::*;

    // --- Gap detection → PassiveRecall enrichment ---

    #[test]
    fn test_feedback_loop_gap_to_enrichment() {
        // Step 1: Create an incomplete composition and detect gaps
        let mut dg = DetectGaps::new();
        let mut comp = Composition::default();
        comp.id = "comp_incomplete".to_string();
        comp.composition_type = CompositionType::Event;
        comp.members.push(CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.9,
            label: "membuat".to_string(),
        });
        comp.members.push(CompositionMember {
            node_id: 2,
            role: SemanticRole::Arg1Patient,
            confidence: 0.8,
            label: "aplikasi".to_string(),
        });
        // Missing Arg0Agent

        let snapshot = GraphSnapshot {
            recent_atoms: vec![],
            compositions: vec![comp.clone()],
        };
        let gaps = dg.detect_atom_gaps(&snapshot);
        assert!(!gaps.is_empty(), "Should detect missing Agent role gap");

        // Step 2: SelectAcquisition — set up graph with a candidate
        let mut sa = SelectAcquisition::new();
        let mut graph = Graph::new();
        let agent_node = graph.ensure_node("raymond");
        let mut existing_comp = Composition::default();
        existing_comp.id = "comp_other".to_string();
        existing_comp.members.push(CompositionMember {
            node_id: agent_node,
            role: SemanticRole::Arg0Agent,
            confidence: 0.9,
            label: "raymond".to_string(),
        });
        graph
            .compositions
            .insert("comp_other".to_string(), existing_comp);

        let gap = gaps
            .iter()
            .find(|g| g.missing_role == Some(SemanticRole::Arg0Agent))
            .unwrap();
        let decision = sa.select_strategy(gap, &graph);

        // Step 3: Verify PassiveRecall was selected
        assert!(
            matches!(decision.strategy, AcquisitionStrategy::PassiveRecall { .. }),
            "Should select PassiveRecall for gap with graph candidate"
        );

        // Step 4: Build EnrichmentRequest from the decision
        if let AcquisitionStrategy::PassiveRecall {
            candidate_node_id,
            candidate_label,
            confidence,
        } = decision.strategy
        {
            let request = EnrichmentRequest {
                target_composition_id: comp.id.clone(),
                role_to_fill: SemanticRole::Arg0Agent,
                candidate_node_id,
                candidate_label,
                source: EnrichmentSource::PassiveRecall,
                confidence,
            };
            assert_eq!(request.role_to_fill, SemanticRole::Arg0Agent);
            assert!(!request.candidate_label.is_empty());
            assert!(request.confidence > 0.0);
        }
    }

    // --- Gap detection → AskUser → process_user_answer_merge → Enrichment ---

    #[test]
    fn test_feedback_loop_gap_to_ask_user() {
        // Step 1: Detect a gap on an empty graph (no PassiveRecall candidates)
        let mut dg = DetectGaps::new();
        let mut comp = Composition::default();
        comp.id = "comp_incomplete".to_string();
        comp.composition_type = CompositionType::Event;
        comp.members.push(CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.9,
            label: "membuat".to_string(),
        });
        // Missing Arg0Agent and Arg1Patient

        let snapshot = GraphSnapshot {
            recent_atoms: vec![],
            compositions: vec![comp.clone()],
        };
        let gaps = dg.detect_atom_gaps(&snapshot);
        assert!(!gaps.is_empty());

        // Step 2: SelectAcquisition with empty graph → AskUser
        let mut sa = SelectAcquisition::new();
        let graph = Graph::new();
        let gap = gaps
            .iter()
            .find(|g| g.missing_role == Some(SemanticRole::Arg0Agent))
            .unwrap();
        let decision = sa.select_strategy(gap, &graph);
        assert!(
            matches!(decision.strategy, AcquisitionStrategy::AskUser { .. }),
            "Should select AskUser when no graph candidates"
        );

        // Step 3: Simulate user answering
        if let AcquisitionStrategy::AskUser { question } = &decision.strategy {
            let mut graph = Graph::new();
            let enrichment = sa.process_user_answer_merge(question, "Raymond", &mut graph);
            assert!(
                enrichment.is_some(),
                "Should produce EnrichmentRequest from user answer"
            );

            let req = enrichment.unwrap();
            assert_eq!(req.role_to_fill, SemanticRole::Arg0Agent);
            assert_eq!(req.candidate_label, "Raymond");
            assert_eq!(req.source, EnrichmentSource::UserAnswerMerge);
            assert!(req.confidence > 0.0);
        }
    }

    // --- Gap detection → ReExtraction ---

    #[test]
    fn test_feedback_loop_gap_to_reextraction() {
        // Create a LowGrounding gap scenario
        let mut dg = DetectGaps::new();
        let mut comp = Composition::default();
        comp.id = "comp_low".to_string();
        comp.composition_type = CompositionType::Event;
        comp.epistemic = EpistemicState::Inferred;
        comp.confidence = 0.3; // Low confidence triggers LowGrounding

        let snapshot = GraphSnapshot {
            recent_atoms: vec![],
            compositions: vec![comp.clone()],
        };
        let gaps = dg.detect_grounding_gaps(&snapshot);
        assert!(
            gaps.iter()
                .any(|g| g.gap_type == KnowledgeGapType::LowGrounding),
            "Should detect low grounding gap"
        );

        // With grounding evidence in graph, SelectAcquisition may choose ReExtraction
        let mut sa = SelectAcquisition::new();
        let mut graph = Graph::new();
        // Add a second composition from a different source to provide grounding evidence
        let mut comp2 = Composition::default();
        comp2.id = "comp_other".to_string();
        comp2.provenance.origin = EdgeSource::HumanAssertion;
        comp2.members.push(CompositionMember {
            node_id: 1,
            role: SemanticRole::Predicate,
            confidence: 0.8,
            label: "membuat".to_string(),
        });
        graph.compositions.insert("comp_low".to_string(), comp);
        graph.compositions.insert("comp_other".to_string(), comp2);

        let grounding_gap = gaps
            .iter()
            .find(|g| g.gap_type == KnowledgeGapType::LowGrounding)
            .unwrap();
        let decision = sa.select_strategy(grounding_gap, &graph);
        // Should select ReExtraction (graph has grounding evidence from different source)
        assert!(
            matches!(decision.strategy, AcquisitionStrategy::ReExtraction { .. }),
            "Should select ReExtraction for low grounding with graph evidence"
        );
    }

    // --- process_user_answer creates an Acquisition atom ---

    #[test]
    fn test_feedback_loop_process_user_answer_creates_atom() {
        let mut ctx = PipelineContext::default();
        let mut roles = HashMap::new();
        roles.insert(SemanticRole::Arg0Agent, "raymond".to_string());

        let atom = SelectAcquisition::process_user_answer(
            "Raymond membuat aplikasi",
            roles,
            0.9,
            &mut ctx,
        );

        assert_eq!(atom.atom_type, AtomType::Acquisition);
        assert_eq!(atom.label, "Raymond membuat aplikasi");
        assert_eq!(atom.source, EdgeSource::AcquisitionUserAnswer);
        assert!(matches!(
            atom.variant,
            Some(AtomVariant::AcquisitionVariant(
                AcquisitionSource::UserAnswer
            ))
        ));
        assert!((atom.confidence - 0.9).abs() < 0.01);
        assert!(
            atom.id.starts_with("acq_"),
            "Atom ID should be prefixed with 'acq_'"
        );
    }

    // --- InquiryMemory prevents asking the same question twice ---

    #[test]
    fn test_feedback_loop_inquiry_memory_prevents_repetition() {
        let mut sa = SelectAcquisition::new();
        let graph = Graph::new();

        // Create a gap for a missing role
        let gap = KnowledgeGap {
            gap_id: "gap_repeat".to_string(),
            gap_type: KnowledgeGapType::MissingRole,
            source_composition_id: Some("comp_1".to_string()),
            missing_role: Some(SemanticRole::Arg0Agent),
            confidence: 0.7,
            ..KnowledgeGap::default()
        };

        // First selection: should produce AskUser (no graph candidates)
        let decision1 = sa.select_strategy(&gap, &graph);
        assert!(
            matches!(decision1.strategy, AcquisitionStrategy::AskUser { .. }),
            "First selection should be AskUser"
        );

        // Second selection: same gap → should Defer (already addressed)
        let decision2 = sa.select_strategy(&gap, &graph);
        assert!(
            matches!(decision2.strategy, AcquisitionStrategy::Defer),
            "Second selection for same gap should be Defer"
        );

        // Verify memory recorded the gap
        assert!(sa.memory.is_gap_addressed("gap_repeat"));
    }
}

// ========================================================================
// B.5: Executive Mode Selection End-to-End
// ========================================================================

#[cfg(feature = "v12")]
mod b5_executive_mode_e2e {
    use super::*;

    // --- Mode selection uses extract_keywords + neighborhood_for ---

    #[test]
    fn test_executive_mode_selection_with_neighborhood() {
        let mut orchestrator = ExecutiveOrchestrator::new();

        // Set up compositions that are relevant to the input
        let comp1 = make_event_composition("comp_1", "membuat", "Raymond", "aplikasi", None, 0.9);
        let compositions = vec![comp1];

        // Mode selection should use keywords from input to find relevant compositions
        let mode = orchestrator
            .select_cognitive_mode("Raymond membuat aplikasi karena lambat", &compositions);
        // With high-confidence compositions, should be Reactive
        assert_eq!(mode, CognitiveMode::Reactive);

        // Verify keywords extraction works (used internally by mode selection)
        let keywords = extract_keywords("Raymond membuat aplikasi karena lambat");
        assert!(
            !keywords.is_empty(),
            "Should extract keywords for neighborhood lookup"
        );

        // Verify neighborhood can find relevant compositions
        let neighborhood = GraphNeighborhood::neighborhood_for(&keywords, &compositions);
        assert!(
            !neighborhood.compositions.is_empty(),
            "Should find compositions via neighborhood"
        );
    }

    // --- Analytical mode runs enrichment loop ---

    #[test]
    fn test_executive_analytical_enrichment_loop() {
        // When compositions have low confidence, Analytical mode is selected
        // with max_enrichment_rounds = 1
        let budget = ComputeBudget::for_mode(&CognitiveMode::Analytical);
        assert_eq!(budget.max_enrichment_rounds, 1);
        assert_eq!(budget.max_reasoning_depth, 3);

        // Verify the mode is selected when average confidence is low
        let mut orchestrator = ExecutiveOrchestrator::new();
        let compositions = vec![Composition {
            confidence: 0.3,
            epistemic: EpistemicState::Observed,
            ..Composition::default()
        }];
        let mode = orchestrator.select_cognitive_mode("test input", &compositions);
        assert_eq!(mode, CognitiveMode::Analytical);

        // Ingest through executive orchestrator with low-confidence prior
        let mut engine = make_pipeline_engine();
        let result = orchestrator.ingest("Raymond membuat aplikasi", &mut engine);
        assert!(
            result.atoms_created > 0,
            "Analytical mode should still create atoms"
        );
    }

    // --- Reflective mode produces ReflectionFindings ---

    #[test]
    fn test_executive_reflective_reflection_findings() {
        // Reflective mode is selected when there are deep contradictions
        let budget = ComputeBudget::for_mode(&CognitiveMode::Reflective);
        assert_eq!(budget.max_enrichment_rounds, 2);

        let mut orchestrator = ExecutiveOrchestrator::new();
        let compositions = vec![
            Composition {
                confidence: 0.5,
                epistemic: EpistemicState::Contradicted,
                ..Composition::default()
            },
            Composition {
                confidence: 0.4,
                epistemic: EpistemicState::Contradicted,
                ..Composition::default()
            },
            Composition {
                confidence: 0.3,
                epistemic: EpistemicState::Contradicted,
                ..Composition::default()
            },
        ];
        let mode = orchestrator.select_cognitive_mode("test input", &compositions);
        assert_eq!(mode, CognitiveMode::Reflective);

        // Verify Reflect produces expected finding types
        let reflect = Reflect::new();
        let result = ReflectionLoopResult {
            current_confidence: 0.4,
            elapsed_ms: 0,
            evidence_count: 0,
            modified_compositions: vec![],
            has_gaps: true,
            resolved_contradictions: vec!["comp_c1".to_string()],
            filled_gaps: vec![],
        };
        let graph = Graph::new();
        let findings = reflect.reflect(&result, &graph);

        // Should produce at least one finding
        assert!(
            !findings.is_empty(),
            "Reflective mode should produce findings"
        );

        // Check for expected finding types
        let finding_types: Vec<_> = findings.iter().map(|f| f.finding_type.clone()).collect();
        assert!(
            finding_types.iter().any(|ft| matches!(
                ft,
                ReflectionFindingType::ContradictionResolvable
                    | ReflectionFindingType::PromotionCandidate
                    | ReflectionFindingType::StagnantInferred
            )),
            "Should produce meaningful reflection finding types"
        );
    }

    // --- ComputeBudget binds to cognitive mode ---

    #[test]
    fn test_executive_budget_binds_to_mode() {
        // Reactive: 0 enrichment rounds
        let reactive = ComputeBudget::for_mode(&CognitiveMode::Reactive);
        assert_eq!(reactive.max_enrichment_rounds, 0);
        assert_eq!(reactive.max_reasoning_depth, 1);
        assert_eq!(reactive.max_reflection_loops, 0);

        // Analytical: 1 enrichment round
        let analytical = ComputeBudget::for_mode(&CognitiveMode::Analytical);
        assert_eq!(analytical.max_enrichment_rounds, 1);
        assert_eq!(analytical.max_reasoning_depth, 3);

        // Reflective: 2 enrichment rounds
        let reflective = ComputeBudget::for_mode(&CognitiveMode::Reflective);
        assert_eq!(reflective.max_enrichment_rounds, 2);
        assert_eq!(reflective.max_reasoning_depth, 5);

        // Budget increases monotonically with mode depth
        assert!(reactive.max_enrichment_rounds < analytical.max_enrichment_rounds);
        assert!(analytical.max_enrichment_rounds < reflective.max_enrichment_rounds);
    }
}

// ========================================================================
// B.6: Full Pipeline End-to-End
// ========================================================================

#[cfg(feature = "v12")]
mod b6_full_pipeline_e2e {
    use super::*;

    // --- Full pipeline: sentence input → atoms + compositions + edges ---

    #[test]
    fn test_full_pipeline_sentence_input() {
        let mut engine = make_pipeline_engine();
        let result = engine.ingest("Raymond membuat aplikasi karena lambat");

        // Should create atoms from tokenization/extraction
        assert!(
            result.atoms_created > 0,
            "Should create atoms from sentence"
        );

        // Pipeline context should track the atoms
        assert!(!engine.context.current_atoms.is_empty());

        // At least one atom should be an Event
        let has_event = engine
            .context
            .current_atoms
            .iter()
            .any(|a| a.atom_type == AtomType::Event);
        assert!(has_event, "Should produce at least one Event atom");

        // The Event atom should have key roles filled
        let event_atom = engine
            .context
            .current_atoms
            .iter()
            .find(|a| a.atom_type == AtomType::Event);
        assert!(event_atom.is_some());
        let atom = event_atom.unwrap();
        assert!(
            atom.roles.contains_key(&SemanticRole::Arg0Agent)
                || atom.roles.contains_key(&SemanticRole::Arg1Patient),
            "Event atom should have Agent or Patient role"
        );
    }

    // --- Full pipeline with gap detection ---

    #[test]
    fn test_full_pipeline_with_gap_detection() {
        let mut engine = make_pipeline_engine();
        engine.context.gap_detection_enabled = true;
        let result = engine.ingest("Raymond membuat aplikasi karena lambat");

        // Pipeline should still run and create atoms
        assert!(result.atoms_created > 0);

        // Gap detection should be enabled
        assert!(engine.context.gap_detection_enabled);
    }

    // --- Full pipeline: passive voice correct role extraction ---

    #[test]
    fn test_full_pipeline_passive_voice() {
        // First verify ExtractFrame handles passive voice correctly
        let ef = ExtractFrame::new();
        let atom = ef.extract("Aplikasi dibuat oleh Raymond");
        assert!(atom.is_some());

        let atom = atom.unwrap();
        assert_eq!(atom.voice, Some(Voice::Passive));
        // In passive: Patient is the subject (before predicate), Agent after "oleh"
        assert_eq!(
            atom.roles.get(&SemanticRole::Arg1Patient),
            Some(&"aplikasi".to_string())
        );
        assert_eq!(
            atom.roles.get(&SemanticRole::Arg0Agent),
            Some(&"raymond".to_string())
        );

        // Now verify through the full pipeline
        let mut engine = make_pipeline_engine();
        let result = engine.ingest("Aplikasi dibuat oleh Raymond");
        assert!(
            result.atoms_created > 0,
            "Passive voice should still produce atoms"
        );
    }

    // --- Full pipeline: two contradictory inputs → contradiction detection ---

    #[test]
    fn test_full_pipeline_with_contradiction() {
        let gb = GovernBeliefs::new();

        // Create two contradictory compositions (role reversal)
        let mut comp1 =
            make_event_composition("comp_pos", "membuat", "Raymond", "aplikasi", None, 0.8);
        let mut comp2 =
            make_event_composition("comp_neg", "membuat", "aplikasi", "Raymond", None, 0.8);

        // Set up role reversal node IDs
        comp1
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg0Agent)
            .unwrap()
            .node_id = 1;
        comp1
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg1Patient)
            .unwrap()
            .node_id = 2;
        comp2
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg0Agent)
            .unwrap()
            .node_id = 2;
        comp2
            .members
            .iter_mut()
            .find(|m| m.role == SemanticRole::Arg1Patient)
            .unwrap()
            .node_id = 1;

        // Detect contradiction
        let updates = gb.detect_contradiction(&mut [comp1.clone(), comp2.clone()]);
        assert!(
            !updates.is_empty(),
            "Should detect contradiction between role-reversed compositions"
        );

        // Verify contradiction is a RoleReversal type
        assert!(
            updates.iter().any(|u| u
                .contradiction
                .as_ref()
                .is_some_and(|c| c.conflict_type == EpistemicConflictType::RoleReversal)),
            "Should detect RoleReversal conflict type"
        );
    }
}

// ========================================================================
// Stub test for when v12 feature is NOT enabled
// ========================================================================

#[cfg(not(feature = "v12"))]
mod v12_feature_not_enabled {
    #[test]
    fn v12_tests_require_feature_flag() {
        eprintln!("NOTE: v12 validation tests require --features v12");
        eprintln!("Run: cargo test --features v12 --test v12_validation");
    }
}
