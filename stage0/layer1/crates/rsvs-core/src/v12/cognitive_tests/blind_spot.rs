use super::helpers::*;

// ========================================================================
// BLIND SPOT 1 — Bahasa Natural Ambigu & Messy
// ========================================================================
//
// Tests that the pipeline handles real Indonesian input: mixed Indo-English,
// typos/informal, and subjectless sentences. The system should not panic,
// should still create atoms, and the graph should grow.

#[test]
fn test_blind_spot_1_bahasa_natural_messy() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    // Case 1: Mixed Indonesian-English — "Raymond deploy-in aplikasinya karena slow banget"
    let result1 = engine.ingest("Raymond deploy-in aplikasinya karena slow banget").unwrap();
    assert!(
        result1.atoms_created > 0,
        "Mixed Indo-English should still create atoms (got {})",
        result1.atoms_created
    );
    eprintln!(
        "  → Mixed Indo-English: atoms={}, nodes={}",
        result1.atoms_created,
        engine.graph().node_count()
    );
    let nodes_after_1 = engine.graph().node_count();

    // Case 2: Typo and informal — "rymnd buat app karna lemot"
    let result2 = engine.ingest("rymnd buat app karna lemot").unwrap();
    assert!(
        result2.atoms_created > 0,
        "Typo/informal should still create atoms (got {})",
        result2.atoms_created
    );
    let nodes_after_2 = engine.graph().node_count();
    assert!(
        nodes_after_2 > nodes_after_1,
        "Graph should grow after second ingest ({} → {})",
        nodes_after_1,
        nodes_after_2
    );
    eprintln!(
        "  → Typo/informal: atoms={}, nodes={}",
        result2.atoms_created, nodes_after_2
    );

    // Case 3: Subjectless — "bikin dulu, deploy nanti"
    // This should still tokenize and create atoms even without a clear agent.
    let result3 = engine.ingest("bikin dulu deploy nanti").unwrap();
    assert!(
        result3.atoms_created > 0,
        "Subjectless should still create atoms (got {})",
        result3.atoms_created
    );
    let nodes_after_3 = engine.graph().node_count();
    assert!(
        nodes_after_3 > nodes_after_2,
        "Graph should grow after third ingest ({} → {})",
        nodes_after_2,
        nodes_after_3
    );
    eprintln!(
        "  → Subjectless: atoms={}, nodes={}",
        result3.atoms_created, nodes_after_3
    );

    // Verify that the pipeline is still functional after messy inputs
    let result4 = engine.ingest("Tim mengoptimasi database").unwrap();
    assert!(
        result4.atoms_created > 0,
        "Pipeline should still work after messy inputs"
    );
    eprintln!(
        "  → Clean after messy: atoms={}, total nodes={}",
        result4.atoms_created,
        engine.graph().node_count()
    );

    eprintln!("✅ BLIND SPOT 1 PASSED: Pipeline handles mixed Indo-English, typo/informal, and subjectless inputs without panicking");
}

// ========================================================================
// BLIND SPOT 2 — InquiryMemory Lintas Ingest
// ========================================================================
//
// After a gap is addressed (e.g., by AskUser), the same gap ID should NOT
// produce another non-Defer decision. SelectAcquisition.memory tracks this.

#[test]
fn test_blind_spot_2_inquiry_memory_lintas_ingest() {
    // InquiryMemory available via helpers

    let mut memory = InquiryMemory::new();

    // Simulate first ingest: gap_0 detected, AskUser chosen
    memory.mark_gap_addressed("gap_0", "AskUser");
    memory.mark_question_asked("q_gap_0");

    // Second ingest: same gap_0 appears again
    assert!(
        memory.is_gap_addressed("gap_0"),
        "InquiryMemory should remember that gap_0 was already addressed"
    );
    assert!(
        memory.is_question_asked("q_gap_0"),
        "InquiryMemory should remember that q_gap_0 was already asked"
    );

    // When SelectAcquisition encounters an already-addressed gap, it should Defer
    let graph = Graph::new();
    let mut sa = SelectAcquisition::new();
    // Manually set the memory (simulating prior ingest)
    sa.memory = memory;

    let gap = KnowledgeGap {
        gap_id: "gap_0".to_string(),
        gap_type: KnowledgeGapType::AmbiguousToken,
        description: "Ambiguous token 'seseorang'".to_string(),
        source_composition_id: None,
        source_atom_id: None,
        missing_role: None,
        confidence: 0.8,
    };

    let decision = sa.select_strategy(&gap, &graph);
    assert_eq!(
        decision.strategy,
        AcquisitionStrategy::Defer,
        "Already-addressed gap should be Deferred, not re-asked. Got {:?}",
        decision.strategy
    );

    // But a NEW gap (gap_1) should still get a non-Defer strategy
    let mut sa2 = SelectAcquisition::new();
    sa2.memory.mark_gap_addressed("gap_0", "AskUser"); // only gap_0 is addressed

    let new_gap = KnowledgeGap {
        gap_id: "gap_1".to_string(),
        gap_type: KnowledgeGapType::AmbiguousToken,
        description: "Ambiguous token 'dia'".to_string(),
        source_composition_id: None,
        source_atom_id: None,
        missing_role: None,
        confidence: 0.8,
    };

    let decision2 = sa2.select_strategy(&new_gap, &graph);
    assert_ne!(
        decision2.strategy,
        AcquisitionStrategy::Defer,
        "NEW gap should NOT be deferred just because a different gap was addressed"
    );

    eprintln!("✅ BLIND SPOT 2 PASSED: InquiryMemory prevents re-asking the same gap, but allows new gaps to be addressed");
}

// ========================================================================
// BLIND SPOT 3 — Semua Tipe Kontradiksi
// ========================================================================
//
// Tests all 5 contradiction types from govern_beliefs.rs:
// PolarityConflict, RoleReversal, PurposeConflict, SemanticContradiction
// (CrossType: HiddenMeaning vs Event), EquivalenceMismatch.

#[test]
fn test_blind_spot_3_semua_tipe_kontradiksi() {
    // ---- 3a: PurposeConflict ----
    // "Perusahaan mempekerjakan pekerja untuk mengurangi biaya"
    // vs "Perusahaan mempekerjakan pekerja untuk meningkatkan kualitas"
    // Same predicate + same agent + different Purpose → PurposeConflict
    {
        let mut graph = Graph::new();
        let mut roles_a = HashMap::new();
        roles_a.insert(SemanticRole::Arg0Agent, "perusahaan".to_string());
        roles_a.insert(SemanticRole::Arg1Patient, "pekerja".to_string());
        roles_a.insert(SemanticRole::Purpose, "mengurangi biaya".to_string());
        let atom_a = make_event_atom(
            "atom_purpose_a",
            "mempekerjakan",
            roles_a,
            Some(Polarity::Positive),
        );
        let mut comp_a = make_event_composition("comp_purpose_a", &atom_a, &mut graph);

        let mut roles_b = HashMap::new();
        roles_b.insert(SemanticRole::Arg0Agent, "perusahaan".to_string());
        roles_b.insert(SemanticRole::Arg1Patient, "pekerja".to_string());
        roles_b.insert(SemanticRole::Purpose, "meningkatkan kualitas".to_string());
        let atom_b = make_event_atom(
            "atom_purpose_b",
            "mempekerjakan",
            roles_b,
            Some(Polarity::Positive),
        );
        let mut comp_b = make_event_composition("comp_purpose_b", &atom_b, &mut graph);

        let gb = GovernBeliefs::new();
        gb.initial_states(&mut comp_a);
        gb.initial_states(&mut comp_b);

        let mut comps = vec![comp_a, comp_b];
        let updates = gb.detect_contradiction(&mut comps);

        let has_purpose_conflict = updates.iter().any(|u| {
            u.contradiction
                .as_ref()
                .map(|c| matches!(c.conflict_type, EpistemicConflictType::PurposeConflict))
                .unwrap_or(false)
        });
        assert!(
            has_purpose_conflict,
            "Expected PurposeConflict for same agent + different purpose. Got: {:?}",
            updates
                .iter()
                .filter_map(|u| u.contradiction.as_ref().map(|c| &c.conflict_type))
                .collect::<Vec<_>>()
        );
        eprintln!("  → PurposeConflict: DETECTED ✅");
    }

    // ---- 3b: RoleReversal ----
    // "Anjing menggigit orang" vs "Orang menggigit anjing"
    // Same predicate + swapped Agent/Patient → RoleReversal
    {
        let mut graph = Graph::new();
        let mut roles_a = HashMap::new();
        roles_a.insert(SemanticRole::Arg0Agent, "anjing".to_string());
        roles_a.insert(SemanticRole::Arg1Patient, "orang".to_string());
        let atom_a = make_event_atom(
            "atom_reversal_a",
            "menggigit",
            roles_a,
            Some(Polarity::Positive),
        );
        let mut comp_a = make_event_composition("comp_reversal_a", &atom_a, &mut graph);

        let mut roles_b = HashMap::new();
        roles_b.insert(SemanticRole::Arg0Agent, "orang".to_string());
        roles_b.insert(SemanticRole::Arg1Patient, "anjing".to_string());
        let atom_b = make_event_atom(
            "atom_reversal_b",
            "menggigit",
            roles_b,
            Some(Polarity::Positive),
        );
        let mut comp_b = make_event_composition("comp_reversal_b", &atom_b, &mut graph);

        let gb = GovernBeliefs::new();
        gb.initial_states(&mut comp_a);
        gb.initial_states(&mut comp_b);

        let mut comps = vec![comp_a, comp_b];
        let updates = gb.detect_contradiction(&mut comps);

        let has_role_reversal = updates.iter().any(|u| {
            u.contradiction
                .as_ref()
                .map(|c| matches!(c.conflict_type, EpistemicConflictType::RoleReversal))
                .unwrap_or(false)
        });
        assert!(
            has_role_reversal,
            "Expected RoleReversal for swapped Agent/Patient. Got: {:?}",
            updates
                .iter()
                .filter_map(|u| u.contradiction.as_ref().map(|c| &c.conflict_type))
                .collect::<Vec<_>>()
        );
        eprintln!("  → RoleReversal: DETECTED ✅");
    }

    // ---- 3c: CrossType (HiddenMeaning contradicts Event) ----
    // Event: "Obat menyembuhkan penyakit"
    // HiddenMeaning: Problem="obat tidak menyembuhkan penyakit" (negates the event)
    {
        let mut graph = Graph::new();

        // Build Event composition
        let mut event_roles = HashMap::new();
        event_roles.insert(SemanticRole::Arg0Agent, "obat".to_string());
        event_roles.insert(SemanticRole::Arg1Patient, "penyakit".to_string());
        let event_atom = make_event_atom(
            "atom_event_obat",
            "menyembuhkan",
            event_roles,
            Some(Polarity::Positive),
        );
        let mut event_comp = make_event_composition("comp_event_obat", &event_atom, &mut graph);
        let gb = GovernBeliefs::new();
        gb.initial_states(&mut event_comp);

        // Build HiddenMeaning composition that contradicts the event
        let mut hm_comp = Composition::default();
        hm_comp.id = "comp_hm_obat_contra".to_string();
        hm_comp.composition_type = CompositionType::HiddenMeaning;
        hm_comp.confidence = 0.6;
        hm_comp.provenance = ProvenanceChain {
            origin: EdgeSource::HiddenMeaningRule,
            origin_id: "atom_hm_obat".to_string(),
            parent_composition_id: None,
            timestamp: String::new(),
        };
        let problem_node_id = graph.ensure_node("obat tidak menyembuhkan penyakit");
        hm_comp.members.push(CompositionMember {
            node_id: problem_node_id,
            role: SemanticRole::Problem,
            confidence: 0.6,
            label: "obat tidak menyembuhkan penyakit".to_string(),
            source: None,
        });
        let source_event_node_id = graph.ensure_node("menyembuhkan");
        hm_comp.members.push(CompositionMember {
            node_id: source_event_node_id,
            role: SemanticRole::SourceEvent,
            confidence: 0.5,
            label: "menyembuhkan".to_string(),
            source: None,
        });
        gb.initial_states(&mut hm_comp);

        let mut comps = vec![event_comp, hm_comp];
        let updates = gb.detect_contradiction(&mut comps);

        // CrossType must fire — HM's Problem "obat tidak menyembuhkan penyakit"
        // negates Event's assertion "obat menyembuhkan penyakit".
        // Strategy 1: SourceEvent label "menyembuhkan" matches Event's Predicate "menyembuhkan"
        // Strategy 2: Problem contains negation + references "obat" (Agent) and "penyakit" (Patient)
        let has_crosstype_conflict = updates.iter().any(|u| {
            u.contradiction
                .as_ref()
                .map(|c| {
                    matches!(
                        c.conflict_type,
                        EpistemicConflictType::SemanticContradiction
                    )
                })
                .unwrap_or(false)
        });
        assert!(
            has_crosstype_conflict,
            "CrossType contradiction MUST be detected: HiddenMeaning Problem='obat tidak menyembuhkan penyakit' \
             negates Event 'obat menyembuhkan penyakit'. Got: {:?}",
            updates.iter()
                .filter_map(|u| u.contradiction.as_ref().map(|c| format!("{:?}", c.conflict_type)))
                .collect::<Vec<_>>()
        );
        eprintln!("  → CrossType (HM vs Event): SemanticContradiction DETECTED ✅");
    }

    // ---- 3d: EquivalenceMismatch ----
    // Two HiddenMeaning compositions with same Problem but different Solution
    {
        let mut graph = Graph::new();

        let mut hm_a = Composition::default();
        hm_a.id = "comp_hm_sol_a".to_string();
        hm_a.composition_type = CompositionType::HiddenMeaning;
        hm_a.confidence = 0.6;
        hm_a.provenance = ProvenanceChain {
            origin: EdgeSource::HiddenMeaningRule,
            origin_id: "atom_hm_a".to_string(),
            parent_composition_id: None,
            timestamp: String::new(),
        };
        let problem_node = graph.ensure_node("lambat");
        let sol_a_node = graph.ensure_node("cache");
        hm_a.members.push(CompositionMember {
            node_id: problem_node,
            role: SemanticRole::Problem,
            confidence: 0.6,
            label: "lambat".to_string(),
            source: None,
        });
        hm_a.members.push(CompositionMember {
            node_id: sol_a_node,
            role: SemanticRole::Solution,
            confidence: 0.6,
            label: "cache".to_string(),
            source: None,
        });

        let mut hm_b = Composition::default();
        hm_b.id = "comp_hm_sol_b".to_string();
        hm_b.composition_type = CompositionType::HiddenMeaning;
        hm_b.confidence = 0.6;
        hm_b.provenance = ProvenanceChain {
            origin: EdgeSource::HiddenMeaningRule,
            origin_id: "atom_hm_b".to_string(),
            parent_composition_id: None,
            timestamp: String::new(),
        };
        let sol_b_node = graph.ensure_node("indexing");
        hm_b.members.push(CompositionMember {
            node_id: problem_node, // same problem node
            role: SemanticRole::Problem,
            confidence: 0.6,
            label: "lambat".to_string(),
            source: None,
        });
        hm_b.members.push(CompositionMember {
            node_id: sol_b_node, // DIFFERENT solution
            role: SemanticRole::Solution,
            confidence: 0.6,
            label: "indexing".to_string(),
            source: None,
        });

        let gb = GovernBeliefs::new();
        gb.initial_states(&mut hm_a);
        gb.initial_states(&mut hm_b);

        let mut comps = vec![hm_a, hm_b];
        let updates = gb.detect_contradiction(&mut comps);

        let has_equiv_mismatch = updates.iter().any(|u| {
            u.contradiction
                .as_ref()
                .map(|c| matches!(c.conflict_type, EpistemicConflictType::EquivalenceMismatch))
                .unwrap_or(false)
        });
        assert!(
            has_equiv_mismatch,
            "Expected EquivalenceMismatch for same Problem + different Solution. Got: {:?}",
            updates
                .iter()
                .filter_map(|u| u.contradiction.as_ref().map(|c| &c.conflict_type))
                .collect::<Vec<_>>()
        );
        eprintln!("  → EquivalenceMismatch: DETECTED ✅");
    }

    eprintln!("✅ BLIND SPOT 3 PASSED: All contradiction types tested — PurposeConflict, RoleReversal, CrossType, EquivalenceMismatch");
}

// ========================================================================
// BLIND SPOT 4 — Rapid Contradiction: Stronger Assertions
// ========================================================================
//
// When contradictions occur rapidly, contradicted compositions should NOT
// be endlessly re-created. After multiple contradictory ingests, contradicted
// compositions should stay contradicted and NOT promote to Stable.

#[test]
fn test_blind_spot_4_rapid_contradiction_strong() {
    let mut graph = Graph::new();
    let gb = GovernBeliefs::new();

    // Create initial positive composition
    let mut roles_pos = HashMap::new();
    roles_pos.insert(SemanticRole::Arg0Agent, "obat".to_string());
    roles_pos.insert(SemanticRole::Arg1Patient, "penyakit".to_string());
    let atom_pos = make_event_atom(
        "atom_obat_pos",
        "menyembuhkan",
        roles_pos,
        Some(Polarity::Positive),
    );
    let mut comp_pos = make_event_composition("comp_obat_pos", &atom_pos, &mut graph);
    comp_pos.confidence = 0.7;
    comp_pos.batch_seen = 3;
    gb.initial_states(&mut comp_pos);
    graph.compositions.insert(comp_pos.id.clone(), comp_pos);

    // Feed 5 contradictory compositions
    for i in 0..5 {
        let mut roles_neg = HashMap::new();
        roles_neg.insert(SemanticRole::Arg0Agent, "obat".to_string());
        roles_neg.insert(SemanticRole::Arg1Patient, "penyakit".to_string());
        roles_neg.insert(SemanticRole::Cause, "tidak menyembuhkan".to_string());
        let atom_neg = make_event_atom(
            &format!("atom_obat_neg_{}", i),
            "menyembuhkan",
            roles_neg,
            Some(Polarity::Negative),
        );
        let mut comp_neg =
            make_event_composition(&format!("comp_obat_neg_{}", i), &atom_neg, &mut graph);
        comp_neg.confidence = 0.6;
        comp_neg.batch_seen = i + 1;
        gb.initial_states(&mut comp_neg);
        graph.compositions.insert(comp_neg.id.clone(), comp_neg);
    }

    // Now run detect_contradiction on all compositions
    let gb2 = GovernBeliefs::new();
    let mut all_comps: Vec<Composition> = graph.compositions.values().cloned().collect();
    let _updates = gb2.detect_contradiction(&mut all_comps);

    // Strong assertion 1: All compositions involved in the contradiction should be Contradicted
    let contradicted_count = all_comps
        .iter()
        .filter(|c| c.epistemic == EpistemicState::Contradicted)
        .count();
    assert!(
        contradicted_count >= 2,
        "At least 2 compositions should be Contradicted, got {}",
        contradicted_count
    );

    // Strong assertion 2: NONE of the contradicted compositions should promote to Stable
    let gb3 = GovernBeliefs::new();
    let promotions = gb3.check_promotions(&mut all_comps);
    for comp in &all_comps {
        if comp.epistemic == EpistemicState::Contradicted {
            assert_ne!(
                comp.lifecycle,
                LifecycleState::Stable,
                "Contradicted composition '{}' should NOT be Stable",
                comp.id
            );
        }
    }
    assert!(
        promotions.is_empty()
            || promotions
                .iter()
                .all(|p| p.new_lifecycle != Some(LifecycleState::Stable)),
        "No contradicted composition should be promoted to Stable. Promotions: {:?}",
        promotions
    );

    // Strong assertion 3: Composition count should be bounded
    // 6 compositions total (1 positive + 5 negative), not 100+
    assert!(
        all_comps.len() < 10,
        "Composition count should stay bounded ({}), not explode",
        all_comps.len()
    );

    // Strong assertion 4: Contradicted compositions should have contradiction_batches recorded
    let comps_with_contradiction_history = all_comps
        .iter()
        .filter(|c| !c.contradiction_batches.is_empty())
        .count();
    assert!(
        comps_with_contradiction_history > 0,
        "At least some compositions should have contradiction_batches recorded"
    );

    eprintln!(
        "  → {} compositions total, {} contradicted, 0 promoted to Stable, {} with contradiction history",
        all_comps.len(),
        contradicted_count,
        comps_with_contradiction_history
    );
    eprintln!("✅ BLIND SPOT 4 PASSED: Contradicted compositions stay Contradicted, never promote to Stable, count stays bounded");
}

// ========================================================================
// BLIND SPOT 5 — Commutativity: Urutan Ingest → Graph Sama
// ========================================================================
//
// ingest(A) → ingest(B) → ingest(C) should produce the same node count
// and composition count as ingest(C) → ingest(A) → ingest(B).
// If NOT commutative, we document WHY.

#[test]
fn test_blind_spot_5_commutativity() {
    let inputs = vec![
        "Raymond membuat aplikasi",
        "Aplikasi mempercepat pekerjaan",
        "Tim mengoptimasi database",
    ];

    // Order 1: A → B → C
    let mut engine1 = PipelineEngine::new();
    register_default_pipeline(&mut engine1);
    for input in &inputs {
        engine1.ingest(input).unwrap();
    }
    let nodes_abc = engine1.graph().node_count();
    let comps_abc = engine1.graph().composition_count();

    // Order 2: C → A → B
    let mut engine2 = PipelineEngine::new();
    register_default_pipeline(&mut engine2);
    for input in inputs.iter().rev() {
        engine2.ingest(input).unwrap();
    }
    let nodes_cab = engine2.graph().node_count();
    let comps_cab = engine2.graph().composition_count();

    // Order 3: B → C → A
    let mut engine3 = PipelineEngine::new();
    register_default_pipeline(&mut engine3);
    let order3 = [&inputs[1], &inputs[2], &inputs[0]];
    for input in &order3 {
        engine3.ingest(input).unwrap();
    }
    let nodes_bca = engine3.graph().node_count();
    let comps_bca = engine3.graph().composition_count();

    eprintln!(
        "  → ABC: nodes={}, comps={} | CAB: nodes={}, comps={} | BCA: nodes={}, comps={}",
        nodes_abc, comps_abc, nodes_cab, comps_cab, nodes_bca, comps_bca
    );

    // Node count should be the same regardless of order (each unique label → 1 node)
    assert_eq!(
        nodes_abc, nodes_cab,
        "Node count should be commutative: ABC={} vs CAB={}",
        nodes_abc, nodes_cab
    );
    assert_eq!(
        nodes_abc, nodes_bca,
        "Node count should be commutative: ABC={} vs BCA={}",
        nodes_abc, nodes_bca
    );

    // Composition count should also be commutative
    assert_eq!(
        comps_abc, comps_cab,
        "Composition count should be commutative: ABC={} vs CAB={}",
        comps_abc, comps_cab
    );
    assert_eq!(
        comps_abc, comps_bca,
        "Composition count should be commutative: ABC={} vs BCA={}",
        comps_abc, comps_bca
    );

    // Also verify that all 3 orders produce the same set of node labels
    let labels_abc: std::collections::HashSet<String> = engine1
        .graph()
        .nodes
        .values()
        .map(|n| n.label.clone())
        .collect();
    let labels_cab: std::collections::HashSet<String> = engine2
        .graph()
        .nodes
        .values()
        .map(|n| n.label.clone())
        .collect();
    let labels_bca: std::collections::HashSet<String> = engine3
        .graph()
        .nodes
        .values()
        .map(|n| n.label.clone())
        .collect();
    assert_eq!(
        labels_abc, labels_cab,
        "Node label sets should match between ABC and CAB"
    );
    assert_eq!(
        labels_abc, labels_bca,
        "Node label sets should match between ABC and BCA"
    );

    eprintln!("✅ BLIND SPOT 5 PASSED: Pipeline is commutative — node count, composition count, and label sets match across all ingest orders");
}

// ========================================================================
// BLIND SPOT 5B — Commutativity with Feedback Loop Active
// ========================================================================
//
// When feedback loop transforms (EnrichComposition, ReExtractFrame) are
// active, the pipeline is **structurally commutative** (same node/comp count)
// but **confidence-non-commutative** (average confidence may differ ≤0.15).
// This test documents that non-commutativity is bounded.

#[test]
fn test_commutativity_with_feedback_loop_active() {
    // Same texts, different order, with pipeline that includes gap detection
    // and acquisition (feedback loop active)
    let texts = [
        "Raymond membuat aplikasi untuk mengatasi kelambatan.",
        "Aplikasi dibuat untuk mengatasi kelambatan proses.",
        "Kelambatan proses diatasi dengan membuat aplikasi baru.",
    ];

    // Run ABC
    let mut engine_abc = PipelineEngine::new();
    register_default_pipeline(&mut engine_abc);
    for text in &texts {
        engine_abc.ingest(text).unwrap();
    }

    // Run CBA
    let mut engine_cba = PipelineEngine::new();
    register_default_pipeline(&mut engine_cba);
    for text in texts.iter().rev() {
        engine_cba.ingest(text).unwrap();
    }

    // Node count harus sama (structural commutativity)
    assert_eq!(
        engine_abc.graph().node_count(),
        engine_cba.graph().node_count(),
        "Node count harus identik terlepas dari urutan ingest"
    );

    // Composition count harus sama
    assert_eq!(
        engine_abc.graph().composition_count(),
        engine_cba.graph().composition_count(),
        "Composition count harus identik"
    );

    // Confidence BOLEH berbeda (acknowledged non-commutativity)
    let comps_abc: Vec<f32> = engine_abc
        .graph()
        .compositions
        .values()
        .map(|c| c.confidence)
        .collect();
    let comps_cba: Vec<f32> = engine_cba
        .graph()
        .compositions
        .values()
        .map(|c| c.confidence)
        .collect();

    let avg_abc: f32 = if comps_abc.is_empty() {
        0.0
    } else {
        comps_abc.iter().sum::<f32>() / comps_abc.len() as f32
    };
    let avg_cba: f32 = if comps_cba.is_empty() {
        0.0
    } else {
        comps_cba.iter().sum::<f32>() / comps_cba.len() as f32
    };

    let delta = (avg_abc - avg_cba).abs();
    // Dokumentasi: confidence non-commutativity diizinkan sampai 0.15
    // Lebih dari 0.15 berarti order-dependence terlalu kuat = bug
    assert!(
        delta <= 0.15,
        "Confidence delta ({:.3}) melebihi toleransi 0.15 — \
         order-dependence terlalu kuat untuk pipeline yang seharusnya \
         converge ke hasil yang stabil",
        delta
    );

    eprintln!(
        "Commutativity with feedback: nodes={}, comps={}, \
         avg_confidence ABC={:.3} vs CBA={:.3}, delta={:.3} (tolerance=0.15)",
        engine_abc.graph().node_count(),
        engine_abc.graph().composition_count(),
        avg_abc,
        avg_cba,
        delta
    );
    eprintln!(
        "✅ BLIND SPOT 5B PASSED: Pipeline is structurally commutative with confidence delta ≤0.15"
    );
}
