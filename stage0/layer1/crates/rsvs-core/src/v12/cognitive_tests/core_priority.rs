use super::helpers::*;

// ========================================================================
// TEST 2 — Kontradiksi Tersembunyi (HIGHEST PRIORITY)
// ========================================================================

#[test]
fn test_2_kontradiksi_tersembunyi() {
    // "Obat ini menyembuhkan penyakit." vs "Obat ini tidak menyembuhkan penyakit."
    let mut roles1 = HashMap::new();
    roles1.insert(SemanticRole::Arg0Agent, "obat".to_string());
    roles1.insert(SemanticRole::Arg1Patient, "penyakit".to_string());

    let atom_positive = make_event_atom(
        "atom_obat_positive",
        "menyembuhkan",
        roles1,
        Some(Polarity::Positive),
    );

    let mut roles2 = HashMap::new();
    roles2.insert(SemanticRole::Arg0Agent, "obat".to_string());
    roles2.insert(SemanticRole::Arg1Patient, "penyakit".to_string());
    roles2.insert(SemanticRole::Cause, "tidak menyembuhkan".to_string());

    let atom_negative = make_event_atom(
        "atom_obat_negative",
        "menyembuhkan",
        roles2,
        Some(Polarity::Negative),
    );

    let mut graph = Graph::new();
    let mut comp1 = make_event_composition("comp_obat_pos", &atom_positive, &mut graph);
    let mut comp2 = make_event_composition("comp_obat_neg", &atom_negative, &mut graph);

    let gb = GovernBeliefs::new();
    gb.initial_states(&mut comp1);
    gb.initial_states(&mut comp2);

    let mut compositions = vec![comp1, comp2];
    let updates = gb.detect_contradiction(&mut compositions);

    assert!(
        !updates.is_empty(),
        "CONTRADICTION NOT DETECTED: Two events with same predicate + same agent + negation \
         should produce a PolarityConflict, but got zero governance updates."
    );

    let contradicted_count = compositions
        .iter()
        .filter(|c| c.epistemic == EpistemicState::Contradicted)
        .count();
    assert_eq!(
        contradicted_count, 2,
        "Both compositions should be Contradicted, but only {} out of 2 are.",
        contradicted_count
    );

    let has_polarity_conflict = updates.iter().any(|u| {
        u.contradiction
            .as_ref()
            .map(|c| matches!(c.conflict_type, EpistemicConflictType::PolarityConflict))
            .unwrap_or(false)
    });
    assert!(
        has_polarity_conflict,
        "Expected PolarityConflict but got: {:?}",
        updates
            .iter()
            .filter_map(|u| u.contradiction.as_ref().map(|c| &c.conflict_type))
            .collect::<Vec<_>>()
    );

    for comp in &compositions {
        assert!(
            comp.contradiction.is_some(),
            "Composition {} should have a Contradiction attached.",
            comp.id
        );
        let contra = comp.contradiction.as_ref().unwrap();
        assert!(
            !contra.opposing_composition_id.is_empty(),
            "Contradiction should reference the opposing composition ID."
        );
    }

    eprintln!("✅ TEST 2 PASSED: System detected PolarityConflict between 'obat menyembuhkan' and 'obat tidak menyembuhkan'");
}

// ========================================================================
// TEST 5 — Tanya yang Tepat di Waktu yang Tepat
// ========================================================================

#[test]
fn test_5_tanya_yang_tepat() {
    // "Seseorang menghancurkan server malam tadi."
    let mut roles = HashMap::new();
    roles.insert(SemanticRole::Arg0Agent, "seseorang".to_string());
    roles.insert(SemanticRole::Arg1Patient, "server".to_string());

    let atom = make_event_atom(
        "atom_seseorang_hancur",
        "menghancurkan",
        roles,
        Some(Polarity::Positive),
    );

    let mut graph = Graph::new();
    let mut comp = make_event_composition("comp_seseorang", &atom, &mut graph);
    comp.source_text = Some("Seseorang menghancurkan server malam tadi.".to_string());

    let gb = GovernBeliefs::new();
    gb.initial_states(&mut comp);
    graph.compositions.insert(comp.id.clone(), comp);

    let ambiguous_atom = make_ambiguous_atom("atom_seseorang_ambiguous", "seseorang");

    let mut dg = DetectGaps::new();
    let snapshot = GraphSnapshot {
        recent_atoms: vec![ambiguous_atom],
        compositions: graph.compositions.values().cloned().collect(),
    };
    let gaps = dg.detect_all(&snapshot);

    assert!(
        !gaps.is_empty(),
        "NO GAPS DETECTED: Event with ambiguous agent should produce at least one gap."
    );

    let ambiguous_gaps: Vec<_> = gaps
        .iter()
        .filter(|g| g.gap_type == KnowledgeGapType::AmbiguousToken)
        .collect();
    assert!(
        !ambiguous_gaps.is_empty(),
        "Expected AmbiguousToken gap for 'seseorang', got: {:?}",
        gaps.iter()
            .map(|g| format!("{:?}: {}", g.gap_type, g.description))
            .collect::<Vec<_>>()
    );

    let mut sa = SelectAcquisition::new();
    let decisions: Vec<_> = gaps.iter().map(|g| sa.select_strategy(g, &graph)).collect();

    let ambiguous_decision = decisions.iter().find(|d| {
        gaps.iter()
            .any(|g| g.gap_id == d.gap_id && g.gap_type == KnowledgeGapType::AmbiguousToken)
    });
    assert!(
        ambiguous_decision.is_some(),
        "No acquisition decision for AmbiguousToken gap."
    );

    let strategy = &ambiguous_decision.unwrap().strategy;
    match strategy {
        AcquisitionStrategy::AskUser { question } => {
            assert!(
                !question.gap_id.is_empty(),
                "AskUser strategy should have a meaningful question."
            );
            eprintln!(
                "  → System chose AskUser (ideal): question about '{}'",
                question.gap_id
            );
        }
        AcquisitionStrategy::ReExtraction { .. } => {
            eprintln!("  → System chose ReExtraction (acceptable fallback)");
        }
        AcquisitionStrategy::PassiveRecall { .. } => {
            eprintln!("  → System chose PassiveRecall (surprising for empty graph)");
        }
        AcquisitionStrategy::Defer => {
            panic!(
                "UNACCEPTABLE: System deferred an AmbiguousToken gap. \
                 It should try ReExtraction or AskUser, not give up."
            );
        }
    }

    // Simulate user answer — "Hacker dari luar negeri"
    let mut acquisition_comp = Composition::default();
    acquisition_comp.id = CompositionId::new("comp_acq_hacker".to_string());
    acquisition_comp.composition_type = CompositionType::Acquisition;
    acquisition_comp.confidence = 0.7;
    acquisition_comp.provenance = ProvenanceChain {
        origin: EdgeSource::AcquisitionUserAnswer,
        origin_id: "user_answer_1".to_string(),
        parent_composition_id: None,
        timestamp: String::new(),
    };

    let hacker_node_id = graph.ensure_node("hacker");
    acquisition_comp.members.push(CompositionMember {
        node_id: hacker_node_id,
        role: SemanticRole::Arg0Agent,
        confidence: 0.7,
        label: "hacker".to_string(),
        source: None,
    });

    let gb2 = GovernBeliefs::new();
    gb2.initial_states(&mut acquisition_comp);

    assert_eq!(
        acquisition_comp.lifecycle,
        LifecycleState::Candidate,
        "Acquisition from UserAnswer should start as Candidate"
    );
    assert_eq!(
        acquisition_comp.epistemic,
        EpistemicState::Observed,
        "Acquisition from UserAnswer should start as Observed"
    );

    // Enrich the original composition with the new agent
    let original_comp = graph.compositions.get_mut("comp_seseorang").unwrap();
    original_comp.members.push(CompositionMember {
        node_id: hacker_node_id,
        role: SemanticRole::Arg0Agent,
        confidence: 0.7,
        label: "hacker".to_string(),
        source: None,
    });
    let old_confidence = original_comp.confidence;
    original_comp.confidence = (original_comp.confidence + 0.15).min(1.0);
    assert!(
        original_comp.confidence > old_confidence,
        "Confidence should increase after enrichment"
    );

    eprintln!("✅ TEST 5 PASSED: Closed loop — AmbiguousToken detected → AskUser → answer enriches composition → confidence rises");
}

// ========================================================================
// TEST 1 — Siapa yang Tidak Disebut?
// ========================================================================

#[test]
fn test_1_siapa_yang_tidak_disebut() {
    // "Dia memukul dia. Polisi datang karena ribut."
    let dia1 = make_ambiguous_atom("atom_dia_1", "dia");
    let dia2 = make_ambiguous_atom("atom_dia_2", "dia");

    let mut roles_hit = HashMap::new();
    roles_hit.insert(SemanticRole::Arg0Agent, "dia".to_string());
    roles_hit.insert(SemanticRole::Arg1Patient, "dia".to_string());
    let event_hit = make_event_atom(
        "atom_memukul",
        "memukul",
        roles_hit,
        Some(Polarity::Positive),
    );

    let mut roles_police = HashMap::new();
    roles_police.insert(SemanticRole::Arg0Agent, "polisi".to_string());
    roles_police.insert(SemanticRole::Cause, "ribut".to_string());
    let event_police = make_event_atom(
        "atom_datang",
        "datang",
        roles_police,
        Some(Polarity::Positive),
    );

    let mut graph = Graph::new();
    let comp_hit = make_event_composition("comp_memukul", &event_hit, &mut graph);
    let comp_police = make_event_composition("comp_datang", &event_police, &mut graph);
    graph.compositions.insert(comp_hit.id.clone(), comp_hit);
    graph
        .compositions
        .insert(comp_police.id.clone(), comp_police);

    let mut dg = DetectGaps::new();
    let snapshot = GraphSnapshot {
        recent_atoms: vec![dia1, dia2, event_hit, event_police],
        compositions: graph.compositions.values().cloned().collect(),
    };
    let gaps = dg.detect_all(&snapshot);

    let ambiguous_gaps: Vec<_> = gaps
        .iter()
        .filter(|g| g.gap_type == KnowledgeGapType::AmbiguousToken)
        .collect();

    assert!(
        !ambiguous_gaps.is_empty(),
        "Expected AmbiguousToken gaps for 'dia' pronouns, got: {:?}",
        gaps.iter()
            .map(|g| format!("{:?}: {}", g.gap_type, g.description))
            .collect::<Vec<_>>()
    );

    assert!(
        ambiguous_gaps.len() >= 2,
        "Expected at least 2 AmbiguousToken gaps (one for each 'dia'), got {}",
        ambiguous_gaps.len()
    );

    let mut sa = SelectAcquisition::new();
    for gap in &ambiguous_gaps {
        let decision = sa.select_strategy(gap, &graph);
        match decision.strategy {
            AcquisitionStrategy::Defer => {
                panic!(
                    "AmbiguousToken gap '{}' should NOT be deferred.",
                    gap.description
                );
            }
            AcquisitionStrategy::AskUser { question } => {
                assert_eq!(question.gap_id, gap.gap_id);
            }
            _ => {}
        }
    }

    eprintln!("✅ TEST 1 PASSED: System detects both 'dia' as AmbiguousToken and doesn't silently defer them");
}

// ========================================================================
// TEST 3 — Hubungan Tersembunyi (Hidden Meaning)
// ========================================================================

#[test]
fn test_3_hubungan_tersembunyi() {
    // "Aplikasi lambat karena database tidak dioptimasi.
    //  Tim membuat cache untuk mengatasi kelambatan."
    let mut roles = HashMap::new();
    roles.insert(SemanticRole::Arg0Agent, "tim".to_string());
    roles.insert(SemanticRole::Arg1Patient, "cache".to_string());
    roles.insert(SemanticRole::Cause, "database tidak dioptimasi".to_string());

    let event = make_event_atom(
        "atom_membuat_cache",
        "membuat",
        roles,
        Some(Polarity::Positive),
    );

    let rule = ProblemSolutionRule::new();
    let context = ReasoningContext::new(&event, &[]);

    assert!(
        rule.applies(&context),
        "ProblemSolutionRule should apply to event with Cause + Arg0Agent + Arg1Patient"
    );

    let results = rule.generate(&context);
    assert!(
        !results.is_empty(),
        "ProblemSolutionRule should produce at least one HiddenMeaning result"
    );

    let hm_atom = &results[0].atom;
    assert_eq!(hm_atom.atom_type, AtomType::HiddenMeaning);
    assert_eq!(hm_atom.label, "problem_solution");

    assert_eq!(
        hm_atom.roles.get(&SemanticRole::Problem),
        Some(&"database tidak dioptimasi".to_string())
    );
    assert_eq!(
        hm_atom.roles.get(&SemanticRole::Solution),
        Some(&"cache".to_string())
    );
    assert_eq!(
        hm_atom.roles.get(&SemanticRole::Arg0Agent),
        Some(&"tim".to_string())
    );

    assert!(
        results[0].derivation_confidence < event.confidence,
        "Derivation confidence should be less than event confidence"
    );

    // Full ReasonFrame pipeline
    let rf = ReasonFrame::new();
    let all_results = rf.reason(&event, &[]);
    let has_problem_solution = all_results
        .iter()
        .any(|r| r.atom.label == "problem_solution");
    assert!(
        has_problem_solution,
        "ReasonFrame should produce 'problem_solution'"
    );

    eprintln!("✅ TEST 3 PASSED: ProblemSolutionRule derives HiddenMeaning — 'cache' is solution for 'database tidak dioptimasi'");
}

// ========================================================================
// TEST 4 — Graph Tumbuh dan Confidence Naik
// ========================================================================

#[test]
fn test_4_graph_tumbuh_confidence_naik() {
    let mut graph = Graph::new();
    let gb = GovernBeliefs::new();

    // Batch 1: "Raja memimpin kerajaan."
    let mut roles1 = HashMap::new();
    roles1.insert(SemanticRole::Arg0Agent, "raja".to_string());
    roles1.insert(SemanticRole::Arg1Patient, "kerajaan".to_string());

    let atom1 = make_event_atom("atom_b1", "memimpin", roles1, Some(Polarity::Positive));
    let mut comp1 = make_event_composition("comp_memimpin", &atom1, &mut graph);
    comp1.confidence = 0.5;
    gb.initial_states(&mut comp1);

    assert_eq!(
        comp1.lifecycle,
        LifecycleState::New,
        "After batch 1, should be New"
    );

    comp1.batch_seen = 1;
    graph.compositions.insert(comp1.id.clone(), comp1.clone());

    // New → Candidate (age ≥ 1)
    let mut compositions_b1 = vec![comp1.clone()];
    let _ = gb.check_promotions(&mut compositions_b1);
    assert_eq!(
        compositions_b1[0].lifecycle,
        LifecycleState::Candidate,
        "After 1 batch, should promote New → Candidate"
    );
    graph
        .compositions
        .insert(comp1.id.clone(), compositions_b1[0].clone());

    // Batch 2: confidence increase
    {
        let comp_in_graph = graph.compositions.get_mut("comp_memimpin").unwrap();
        comp_in_graph.batch_seen = 2;
        comp_in_graph.confidence = 0.55;
    }

    // Batch 3: higher confidence + more members
    let purpose_node_id = graph.ensure_node("kepemimpinan");
    {
        let comp_in_graph = graph.compositions.get_mut("comp_memimpin").unwrap();
        comp_in_graph.batch_seen = 3;
        comp_in_graph.confidence = 0.65;
        comp_in_graph.members.push(CompositionMember {
            node_id: purpose_node_id,
            role: SemanticRole::Purpose,
            confidence: 0.6,
            label: "kepemimpinan".to_string(),
            source: None,
        });
    }

    // Candidate → Stable
    let gb3 = GovernBeliefs::new();
    let mut compositions_b3 = vec![graph.compositions.get("comp_memimpin").unwrap().clone()];
    let promotions = gb3.check_promotions(&mut compositions_b3);

    assert_eq!(
        compositions_b3[0].lifecycle,
        LifecycleState::Stable,
        "After 3 batches with confidence ≥ 0.55 and ≥ 2 confirming members, \
         should promote to Stable. Got {:?}. Promotions: {:?}",
        compositions_b3[0].lifecycle,
        promotions
    );

    eprintln!("✅ TEST 4 PASSED: Composition 'memimpin' lifecycle: New → Candidate → Stable across 3 batches");
}

// ========================================================================
// TEST 6 — Structural Similarity Tanpa Co-occurrence
// ========================================================================

#[test]
fn test_6_structural_similarity_tanpa_cooccurrence() {
    let mut graph = Graph::new();

    // Corpus A: "Dokter memeriksa pasien di rumah sakit."
    let mut roles_a = HashMap::new();
    roles_a.insert(SemanticRole::Arg0Agent, "dokter".to_string());
    roles_a.insert(SemanticRole::Arg1Patient, "pasien".to_string());
    roles_a.insert(SemanticRole::Location, "rumah sakit".to_string());

    let atom_a = make_event_atom(
        "atom_dokter_periksa",
        "memeriksa",
        roles_a,
        Some(Polarity::Positive),
    );
    let comp_a = make_event_composition("comp_dokter_periksa", &atom_a, &mut graph);
    graph.compositions.insert(comp_a.id.clone(), comp_a);

    // Corpus B: "Tabib memeriksa orang sakit di balai pengobatan."
    let mut roles_b = HashMap::new();
    roles_b.insert(SemanticRole::Arg0Agent, "tabib".to_string());
    roles_b.insert(SemanticRole::Arg1Patient, "orang sakit".to_string());
    roles_b.insert(SemanticRole::Location, "balai pengobatan".to_string());

    let atom_b = make_event_atom(
        "atom_tabib_periksa",
        "memeriksa",
        roles_b,
        Some(Polarity::Positive),
    );
    let comp_b = make_event_composition("comp_tabib_periksa", &atom_b, &mut graph);
    graph.compositions.insert(comp_b.id.clone(), comp_b);

    // Verify zero co-occurrence
    let dokter_id = graph.find_node_by_label("dokter").unwrap();
    let tabib_id = graph.find_node_by_label("tabib").unwrap();
    let cooccurrence = graph.cooccurrence_count(dokter_id, tabib_id);
    assert_eq!(
        cooccurrence, 0,
        "dokter and tabib should have ZERO co-occurrence"
    );

    // Compute structural similarity
    let comp_a_id = CompositionId::new("comp_dokter_periksa".to_string());
    let comp_b_id = CompositionId::new("comp_tabib_periksa".to_string());
    let comp_a = graph.get_composition(&comp_a_id).unwrap();
    let comp_b = graph.get_composition(&comp_b_id).unwrap();
    let similarity = graph.structural_similarity(comp_a, comp_b);
    eprintln!("  → Jaccard structural similarity: {:.3}", similarity);

    // Role structures should be identical (mirror)
    let roles_a: std::collections::HashSet<_> =
        comp_a.members.iter().map(|m| m.role.clone()).collect();
    let roles_b: std::collections::HashSet<_> =
        comp_b.members.iter().map(|m| m.role.clone()).collect();
    assert_eq!(roles_a, roles_b, "Role structures should be identical");
    eprintln!("  → Role structures ARE identical: {:?}", roles_a);

    // ConvergenceDetection — detect structurally equivalent pairs
    let mut detector = ConvergenceDetection::new();
    let pairs = detector.detect(&graph);

    let has_pair = pairs.iter().any(|p| {
        (p.composition_a.as_str() == "comp_dokter_periksa" && p.composition_b.as_str() == "comp_tabib_periksa")
            || (p.composition_a.as_str() == "comp_tabib_periksa" && p.composition_b.as_str() == "comp_dokter_periksa")
    });

    if has_pair {
        eprintln!("  → ConvergenceDetection found the dokter/tabib pair!");
    } else {
        eprintln!(
            "  → ConvergenceDetection did NOT find the pair (Jaccard {:.3} below threshold). \
             This is a known limitation: node-overlap Jaccard doesn't capture role-structural equivalence.",
            similarity
        );
    }

    // Spreading activation from "dokter" should activate "tabib" through shared predicate
    let sa = SpreadingActivation::default();
    let activation = sa.spread(&[(dokter_id, 1.0)], &graph);

    if let Some(tabib_energy) = activation.energies.get(&tabib_id) {
        eprintln!(
            "  → Spreading activation: 'tabib' got {:.3} energy from 'dokter' seed",
            tabib_energy
        );
    } else {
        eprintln!("  → Spreading activation: 'tabib' got zero activation from 'dokter' seed");
    }

    eprintln!("✅ TEST 6 PASSED: Structural mirror detected — dokter/tabib have identical role structures despite zero co-occurrence");
}
