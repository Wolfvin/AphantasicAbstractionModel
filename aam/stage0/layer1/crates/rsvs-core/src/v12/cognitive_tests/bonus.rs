use super::helpers::*;

// ========================================================================
// BONUS: ReasonFrame PolarityConflictRule (cross-atom reasoning)
// ========================================================================

#[test]
fn test_bonus_reason_frame_polarity_conflict() {
    let mut roles1 = HashMap::new();
    roles1.insert(SemanticRole::Arg0Agent, "obat".to_string());
    roles1.insert(SemanticRole::Arg1Patient, "penyakit".to_string());

    let event_positive = make_event_atom(
        "atom_pos",
        "menyembuhkan",
        roles1.clone(),
        Some(Polarity::Positive),
    );
    let event_negative =
        make_event_atom("atom_neg", "menyembuhkan", roles1, Some(Polarity::Negative));

    let recent = vec![event_negative];
    let context = ReasoningContext::new(&event_positive, &recent);

    let rule = PolarityConflictRule::new();
    assert!(
        rule.applies(&context),
        "PolarityConflictRule should fire for same predicate + opposite polarity"
    );

    let results = rule.generate(&context);
    assert_eq!(
        results.len(),
        1,
        "Should produce exactly 1 polarity_conflict atom"
    );
    assert_eq!(results[0].atom.label, "polarity_conflict");
    assert_eq!(results[0].atom.atom_type, AtomType::HiddenMeaning);
    assert!(results[0].atom.roles.contains_key(&SemanticRole::Problem));

    eprintln!("✅ BONUS TEST PASSED: ReasonFrame PolarityConflictRule detects cross-atom polarity conflict");
}

// ========================================================================
// BONUS: Full Pipeline End-to-End
// ========================================================================

#[test]
fn test_bonus_full_pipeline_e2e() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    let result = engine.ingest("Raymond membuat aplikasi karena lambat").unwrap();

    assert!(result.atoms_created > 0, "Pipeline should create atoms");
    assert!(engine.graph().node_count() > 0, "Graph should have nodes");

    eprintln!(
        "  → Pipeline result: atoms={}, compositions={}, edges={}, gaps={}",
        result.atoms_created,
        result.compositions_created,
        result.edges_created,
        result.gaps_detected
    );

    let result2 = engine.ingest("Aplikasi mempercepat pekerjaan tim").unwrap();
    assert!(
        result2.atoms_created > 0,
        "Second ingest should also create atoms"
    );
    assert!(engine.graph().node_count() > 1, "Graph should grow");

    eprintln!(
        "  → After 2 ingests: nodes={}, compositions={}",
        engine.graph().node_count(),
        engine.graph().composition_count()
    );

    eprintln!("✅ BONUS TEST 2 PASSED: Full pipeline end-to-end with multiple ingests");
}
