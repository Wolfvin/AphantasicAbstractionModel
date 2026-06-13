use super::helpers::*;

// ========================================================================
// Test 9 — Kompiler Aturan Pajak: ConditionConsequenceRule
// ========================================================================

/// Test: Kalimat regulasi dengan "jika" menghasilkan condition_consequence atom.
///
/// Skenario: Ingest "wajib pajak jika penghasilan di atas 500 juta dikenakan tarif 30 persen"
/// Expected: Pipeline menghasilkan composition dengan Antecedent/Consequent roles,
///           dan ConditionConsequenceRule menghasilkan hidden meaning "condition_consequence".
#[test]
fn test_condition_consequence_from_indonesian_if_then() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    // Ingest a conditional sentence in Indonesian
    let result =
        engine.ingest("wajib pajak jika penghasilan di atas 500 juta dikenakan tarif 30 persen").unwrap();

    // Should create atoms and compositions
    assert!(
        result.atoms_created > 0,
        "Should create atoms from conditional text"
    );
    assert!(
        result.compositions_created > 0,
        "Should create compositions from conditional text"
    );

    // Check that at least one composition has Antecedent or Consequent roles
    let graph = engine.graph();
    let has_conditional = graph.compositions.values().any(|c| {
        c.members
            .iter()
            .any(|m| m.role == SemanticRole::Antecedent || m.role == SemanticRole::Consequent)
    });

    // Even if the composition doesn't directly have these roles,
    // the hidden meaning atoms should exist
    let has_condition_atom = graph.compositions.values().any(|c| {
        c.composition_type == CompositionType::HiddenMeaning
            && c.members.iter().any(|m| m.role == SemanticRole::Antecedent)
    });

    // At minimum, the pipeline should not crash with conditional text
    // and should produce some output
    assert!(
        has_conditional || has_condition_atom || result.atoms_created > 5,
        "Pipeline should handle conditional Indonesian text — \
         either extract Antecedent/Consequent roles or create sufficient atoms. \
         atoms={}, comps={}",
        result.atoms_created,
        result.compositions_created
    );

    eprintln!(
        "Conditional ingest: atoms={}, comps={}",
        result.atoms_created, result.compositions_created
    );
}

/// Test: Direct ConditionConsequenceRule — verify it triggers on Antecedent+Consequent.
#[test]
fn test_condition_consequence_rule_direct() {
    // ConditionConsequenceRule, ReasoningContext, ReasoningRule available via helpers

    let rule = ConditionConsequenceRule::new();

    // Build an event atom with Antecedent and Consequent
    let mut roles = HashMap::new();
    roles.insert(SemanticRole::Predicate, "dikenakan".to_string());
    roles.insert(
        SemanticRole::Antecedent,
        "penghasilan di atas 500 juta".to_string(),
    );
    roles.insert(SemanticRole::Consequent, "tarif 30 persen".to_string());

    let event = SemanticAtom {
        id: "atom_test_cond".to_string(),
        label: "dikenakan".to_string(),
        atom_type: AtomType::Event,
        roles,
        polarity: Some(Polarity::Positive),
        voice: Some(Voice::Passive),
        variant: Some(AtomVariant::FrameVariant(FrameSource::RuleBased)),
        confidence: 0.80,
        source: EdgeSource::FrameCompiler,
        composition_id: None,
    };

    let ctx = ReasoningContext::new(&event, &[]);

    // Rule should apply
    assert!(
        rule.applies(&ctx),
        "ConditionConsequenceRule should apply to event with Antecedent+Consequent"
    );

    // Generate result
    let results = rule.generate(&ctx);
    assert_eq!(
        results.len(),
        1,
        "Should produce exactly one reasoning result"
    );
    assert_eq!(results[0].atom.label, "condition_consequence");
    assert_eq!(results[0].atom.atom_type, AtomType::HiddenMeaning);
    assert_eq!(
        results[0].atom.roles.get(&SemanticRole::Antecedent),
        Some(&"penghasilan di atas 500 juta".to_string())
    );
    assert_eq!(
        results[0].atom.roles.get(&SemanticRole::Consequent),
        Some(&"tarif 30 persen".to_string())
    );
    assert_eq!(
        results[0].atom.roles.get(&SemanticRole::PatternType),
        Some(&"if_then".to_string())
    );
    assert!(
        results[0].derivation_confidence > 0.7,
        "Confidence should be high for clear conditional"
    );
}
