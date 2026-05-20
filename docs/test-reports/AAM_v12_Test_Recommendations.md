# AAM v12.0 — Rekomendasi Test Terbaik

> Dokumen ini berisi rekomendasi test untuk memvalidasi bahwa updates v12.0 bekerja sesuai spesifikasi design docs (MD-1 s/d MD-6). Test dikelompokkan berdasarkan prioritas dan jenisnya.

---

## Daftar Isi

1. [Ringkasan Prioritas](#1-ringkasan-prioritas)
2. [Pilihan A: Smoke Test Cepat (30 menit)](#2-pilihan-a-smoke-test-cepat-30-menit)
3. [Pilihan B: Focused Integration Test (2-3 jam)](#3-pilihan-b-focused-integration-test-2-3-jam)
4. [Pilihan C: Full E2E Validation (1 hari)](#4-pilihan-c-full-e2e-validation-1-hari)
5. [Pilihan D: Stress & Edge Case Test (2 hari)](#5-pilihan-d-stress--edge-case-test-2-hari)
6. [Test Mandiri per MD](#6-test-mandiri-per-md)
7. [Test Cross-Cutting (Antar MD)](#7-test-cross-cutting-antar-md)
8. [Test untuk Mismatch yang Ditemukan Audit](#8-test-untuk-mismatch-yang-ditemukan-audit)
9. [Rekomendasi Prioritas Eksekusi](#9-rekomendasi-prioritas-eksekusi)

---

## 1. Ringkasan Prioritas

| Tier | Kategori | Jumlah Test | Tujuan |
|------|----------|-------------|--------|
| 🔴 P0 | Compile & Type System | 5 | Pastikan kode kompilasi dan type system v12 konsisten |
| 🟠 P1 | Unit Test per Transform | 20 | Validasi setiap Transform bekerja sesuai MD spec |
| 🟡 P2 | Integration Test (Feedback Loop) | 10 | Validasi closed feedback loop: gap → enrich → re-govern |
| 🟢 P3 | E2E Pipeline Test | 8 | Validasi full pipeline dari text input sampai graph state |
| 🔵 P4 | Edge Case & Regression | 12 | Validasi boundary conditions dan backward compatibility |

**Total: ~55 test scenario** — bisa dipilih subset berdasarkan waktu yang tersedia.

---

## 2. Pilihan A: Smoke Test Cepat (30 menit)

> **Untuk**: Cepat cek apakah v12.0 basic infrastructure bekerja. Cocok untuk CI gate.

### A.1 — Cargo Check + Cargo Test (Existing)

```bash
cd layer1
cargo check --features v12
cargo test --features v12
```

**Apa yang dicek**:
- Semua tipe v12 (`SemanticAtom`, `Composition`, `LifecycleState`, `EpistemicState`, dll.) terdefinisi
- Trait `Transform` bisa di-implement
- Pipeline engine bisa di-instantiate
- 48 existing `#[test]` di v12 modules masih pass

**Pass criteria**: `cargo check` 0 errors, `cargo test` 0 failures.

### A.2 — SemanticAtom Construction Test

```rust
#[test]
fn semantic_atom_all_types_constructible() {
    // Token (sparse)
    let token = SemanticAtom {
        id: "tok_1".into(), label: "raja".into(),
        atom_type: AtomType::Token, roles: HashMap::new(),
        polarity: None, voice: None, variant: None,
        confidence: 0.5, source: EdgeSource::Bootstrap,
    };
    assert_eq!(token.atom_type, AtomType::Token);
    assert!(token.roles.is_empty());

    // Event (rich)
    let mut roles = HashMap::new();
    roles.insert(SemanticRole::Arg0Agent, "Raymond".into());
    roles.insert(SemanticRole::Arg1Patient, "aplikasi".into());
    let event = SemanticAtom {
        id: "evt_1".into(), label: "membuat".into(),
        atom_type: AtomType::Event, roles,
        polarity: Some(Polarity::Positive), voice: Some(Voice::Active),
        variant: Some(AtomVariant::FrameVariant(FrameSource::RuleBased)),
        confidence: 0.6, source: EdgeSource::FrameCompiler,
    };
    assert_eq!(event.roles.len(), 2);
    assert_eq!(event.polarity, Some(Polarity::Positive));

    // HiddenMeaning
    let mut hm_roles = HashMap::new();
    hm_roles.insert(SemanticRole::Problem, "lambat".into());
    hm_roles.insert(SemanticRole::Solution, "aplikasi".into());
    let hm = SemanticAtom {
        id: "hm_1".into(), label: "problem_solution".into(),
        atom_type: AtomType::HiddenMeaning, roles: hm_roles,
        polarity: None, voice: None,
        variant: Some(AtomVariant::MeaningVariant(HiddenMeaningType::ProblemSolutionPattern)),
        confidence: 0.6, source: EdgeSource::HiddenMeaningRule,
    };
    assert_eq!(hm.atom_type, AtomType::HiddenMeaning);

    // Acquisition
    let acq = SemanticAtom {
        id: "acq_1".into(), label: "Raymond".into(),
        atom_type: AtomType::Acquisition,
        roles: HashMap::new(), polarity: None, voice: None,
        variant: Some(AtomVariant::AcquisitionVariant(AcquisitionSource::UserAnswer)),
        confidence: 0.85, source: EdgeSource::AcquisitionUserAnswer,
    };
    assert_eq!(acq.source, EdgeSource::AcquisitionUserAnswer);
}
```

**Apa yang dicek**: Semua `AtomType` variant bisa dikonstruksi, termasuk `Acquisition` dan `AmbiguousToken` yang baru di v12.

### A.3 — Dual-Axis Status Test

```rust
#[test]
fn lifecycle_and_epistemic_are_independent() {
    // (Quarantine, Inferred) — hidden meaning default
    let comp = Composition {
        lifecycle: LifecycleState::Quarantine,
        epistemic: EpistemicState::Inferred,
        ..default_composition()
    };
    assert_ne!(comp.lifecycle, LifecycleState::Stable);
    assert_ne!(comp.epistemic, EpistemicState::Observed);

    // Transition lifecycle tanpa affect epistemic
    let mut comp2 = comp.clone();
    comp2.lifecycle = LifecycleState::Candidate;
    assert_eq!(comp2.epistemic, EpistemicState::Inferred); // unchanged

    // Transition epistemic tanpa affect lifecycle
    let mut comp3 = comp.clone();
    comp3.epistemic = EpistemicState::Grounded;
    assert_eq!(comp3.lifecycle, LifecycleState::Quarantine); // unchanged
}
```

**Apa yang dicek**: LifecycleState dan EpistemicState benar-benar orthogonally independent — core principle MD-3.

### A.4 — Pipeline Engine Registration Test

```rust
#[test]
fn default_pipeline_registers_all_transforms() {
    let engine = register_default_pipeline(PipelineContext::new());
    // Should have: Tokenize, ExtractFrame, ReasonFrame, IngestAtoms,
    //              GovernBeliefs, SeedAnchor, DetectGaps, SelectAcquisition,
    //              EnrichComposition, ReExtractFrame = 10 transforms
    assert!(engine.transform_count() >= 10);
}
```

**Apa yang dicek**: DAG pipeline engine bisa register semua 10 core transforms tanpa panic.

### A.5 — Roundtrip Serialization Test

```rust
#[test]
fn v12_types_serialize_deserialize() {
    let atom = test_event_atom();
    let json = serde_json::to_string(&atom).unwrap();
    let deserialized: SemanticAtom = serde_json::from_str(&json).unwrap();
    assert_eq!(atom.id, deserialized.id);
    assert_eq!(atom.atom_type, deserialized.atom_type);
    assert_eq!(atom.roles, deserialized.roles);
}
```

**Apa yang dicek**: Semua v12 types bisa serialize/deserialize via Serde — penting untuk persistence.

---

## 3. Pilihan B: Focused Integration Test (2-3 jam)

> **Untuk**: Validasi bahwa setiap MD Transform bekerja secara mandiri DAN terintegrasi dengan transforms lainnya. Fokus pada **feedback loop**.

### B.1 — MD-1: ExtractFrame Transform Test

```rust
#[test]
fn extract_frame_active_indonesian_sentence() {
    let ef = ExtractFrame::new(FrameCompilerConfig::default());
    let mut ctx = PipelineContext::new();
    ctx.set_raw_text("Raymond membuat aplikasi karena lambat");

    let result = ef.transform(&"Raymond membuat aplikasi karena lambat", &mut ctx);
    assert!(result.is_some());
    let atom = result.unwrap();
    assert_eq!(atom.atom_type, AtomType::Event);
    assert_eq!(atom.label, "membuat");
    assert_eq!(atom.roles.get(&SemanticRole::Arg0Agent), Some(&"Raymond".into()));
    assert_eq!(atom.roles.get(&SemanticRole::Arg1Patient), Some(&"aplikasi".into()));
    assert_eq!(atom.roles.get(&SemanticRole::Cause), Some(&"lambat".into()));
    assert_eq!(atom.polarity, Some(Polarity::Positive));
    assert_eq!(atom.voice, Some(Voice::Active));
}
```

```rust
#[test]
fn extract_frame_passive_indonesian_sentence() {
    let ef = ExtractFrame::new(FrameCompilerConfig::default());
    let mut ctx = PipelineContext::new();
    let result = ef.transform(&"Aplikasi dibuat oleh Raymond", &mut ctx);
    assert!(result.is_some());
    let atom = result.unwrap();
    assert_eq!(atom.voice, Some(Voice::Passive));
    assert_eq!(atom.roles.get(&SemanticRole::Arg1Patient), Some(&"Aplikasi".into()));
    assert_eq!(atom.roles.get(&SemanticRole::Arg0Agent), Some(&"Raymond".into()));
}
```

```rust
#[test]
fn extract_frame_negated_sentence() {
    let ef = ExtractFrame::new(FrameCompilerConfig::default());
    let result = ef.transform(&"Raymond tidak membuat aplikasi", &mut PipelineContext::new());
    assert!(result.is_some());
    assert_eq!(result.unwrap().polarity, Some(Polarity::Negative));
}
```

```rust
#[test]
fn extract_frame_token_input_returns_none() {
    let ef = ExtractFrame::new(FrameCompilerConfig::default());
    let result = ef.transform(&"raja", &mut PipelineContext::new());
    assert!(result.is_none()); // single token, not sentence-like
}
```

**Apa yang dicek**: Rule-based extraction MD-1 bekerja untuk active, passive, negated, dan token input.

### B.2 — MD-2: ReasonFrame Transform Test

```rust
#[test]
fn reason_frame_problem_solution_pattern() {
    let rf = ReasonFrame::new(vec![
        Box::new(ProblemSolutionRule),
        Box::new(GoalInferenceRule),
        Box::new(PolarityConflictRule),
    ]);
    let mut roles = HashMap::new();
    roles.insert(SemanticRole::Arg0Agent, "Raymond".into());
    roles.insert(SemanticRole::Arg1Patient, "aplikasi".into());
    roles.insert(SemanticRole::Cause, "lambat".into());
    let event = SemanticAtom {
        id: "evt_1".into(), label: "membuat".into(),
        atom_type: AtomType::Event, roles,
        polarity: Some(Polarity::Positive), voice: Some(Voice::Active),
        variant: Some(AtomVariant::FrameVariant(FrameSource::RuleBased)),
        confidence: 0.6, source: EdgeSource::FrameCompiler,
    };
    let mut ctx = PipelineContext::new();
    let results = rf.transform(&event, &mut ctx);
    assert!(!results.is_empty());
    let hm = &results[0];
    assert_eq!(hm.atom_type, AtomType::HiddenMeaning);
    assert_eq!(hm.label, "problem_solution");
    assert_eq!(hm.roles.get(&SemanticRole::Problem), Some(&"lambat".into()));
    assert_eq!(hm.roles.get(&SemanticRole::Solution), Some(&"aplikasi".into()));
}
```

```rust
#[test]
fn reason_frame_polarity_conflict() {
    // Feed positive event first, then negative with same predicate
    let rf = ReasonFrame::new(vec![Box::new(PolarityConflictRule)]);
    let mut ctx = PipelineContext::new();

    // First: positive event
    let pos = test_event_atom_with_polarity(Polarity::Positive);
    ctx.record_event(pos.clone());

    // Second: negative event (same predicate/agent/patient)
    let neg = test_event_atom_with_polarity(Polarity::Negative);
    let results = rf.transform(&neg, &mut ctx);
    assert!(results.iter().any(|r| r.label == "polarity_conflict"));
}
```

**Apa yang dicek**: 3 core reasoning rules MD-2 (ProblemSolution, GoalInference, PolarityConflict) bekerja.

### B.3 — MD-4: GovernBeliefs Transform Test

```rust
#[test]
fn govern_beliefs_initial_state_assignment() {
    let gb = GovernBeliefs::new(GovernanceConfig::default());

    // Event from FrameCompiler → (New, Observed)
    let event_comp = test_composition(CompositionType::Event, EdgeSource::FrameCompiler);
    let (lc, ep) = gb.initial_states(&event_comp);
    assert_eq!(lc, LifecycleState::New);
    assert_eq!(ep, EpistemicState::Observed);

    // HiddenMeaning → (Quarantine, Inferred)
    let hm_comp = test_composition(CompositionType::HiddenMeaning, EdgeSource::HiddenMeaningRule);
    let (lc, ep) = gb.initial_states(&hm_comp);
    assert_eq!(lc, LifecycleState::Quarantine);
    assert_eq!(ep, EpistemicState::Inferred);

    // Acquisition from UserAnswer → (Candidate, Observed)
    let acq_comp = test_composition(CompositionType::Acquisition, EdgeSource::AcquisitionUserAnswer);
    let (lc, ep) = gb.initial_states(&acq_comp);
    assert_eq!(lc, LifecycleState::Candidate);
    assert_eq!(ep, EpistemicState::Observed);
}
```

```rust
#[test]
fn govern_beliefs_contradiction_detection() {
    let gb = GovernBeliefs::new(GovernanceConfig::default());
    let mut graph = Graph::new();

    // Add two compositions with same predicate but conflicting polarity
    let comp_a = test_event_composition("membuat", "Raymond", "aplikasi", Polarity::Positive);
    let comp_b = test_event_composition("membuat", "Raymond", "tidak aplikasi", Polarity::Negative);
    graph.add_composition(comp_a);
    graph.add_composition(comp_b);

    let contradiction = gb.detect_contradiction(&comp_b, &graph);
    assert!(contradiction.is_some());
    assert_eq!(contradiction.unwrap().conflict_type, EpistemicConflictType::PolarityConflict);
}
```

```rust
#[test]
fn govern_beliefs_promotion_criteria() {
    let gb = GovernBeliefs::new(GovernanceConfig::default());
    let graph = Graph::new();

    // Too young — should be denied
    let young = test_composition_with_age(CompositionType::Event, 1);
    let verdict = gb.can_promote_to_stable(&young, &graph);
    assert!(matches!(verdict, PromotionVerdict::Denied(_)));

    // Old enough + high confidence + no contradiction
    let mature = test_composition_with_age_and_confidence(CompositionType::Event, 5, 0.7);
    let verdict = gb.can_promote_to_stable(&mature, &graph);
    assert!(matches!(verdict, PromotionVerdict::Approved));
}
```

**Apa yang dicek**: Initial state assignment per composition type, contradiction detection, dan promotion criteria sesuai MD-4 spec.

### B.4 — MD-6: DetectGaps + SelectAcquisition Test

```rust
#[test]
fn detect_gaps_event_missing_agent() {
    let dg = DetectGaps::new(GapDetectionConfig::default());

    // Event tanpa agent
    let mut roles = HashMap::new();
    roles.insert(SemanticRole::Arg1Patient, "aplikasi".into());
    let event = SemanticAtom {
        id: "evt_1".into(), label: "membuat".into(),
        atom_type: AtomType::Event, roles,
        polarity: Some(Polarity::Positive), voice: Some(Voice::Active),
        variant: Some(AtomVariant::FrameVariant(FrameSource::RuleBased)),
        confidence: 0.4, source: EdgeSource::FrameCompiler,
    };

    let snapshot = GraphSnapshot::from_atoms(vec![event]);
    let gaps = dg.transform(&snapshot, &mut PipelineContext::new());
    assert!(gaps.iter().any(|g| g.gap_type == KnowledgeGapType::MissingFieldGap
        && g.missing_role == Some(SemanticRole::Arg0Agent)));
}
```

```rust
#[test]
fn select_acquisition_prefers_passive_recall() {
    let sa = SelectAcquisition::new(InquiryMemory::new());
    let mut graph = Graph::new();
    // Add composition with known role filler for "membuat" → Agent
    graph.add_composition(test_event_with_agent("membuat", "Raymond"));

    let gap = KnowledgeGap {
        gap_type: KnowledgeGapType::MissingFieldGap,
        missing_role: Some(SemanticRole::Arg0Agent),
        source_composition_id: Some("comp_2".into()),
        ..test_gap()
    };

    let decision = sa.select_strategy(&gap, &graph);
    assert_eq!(decision.mode, AcquisitionMode::PassiveRecall);
    assert!(decision.action.is_some()); // Should have EnrichComposition action
}
```

```rust
#[test]
fn select_acquisition_asks_user_when_no_graph_context() {
    let sa = SelectAcquisition::new(InquiryMemory::new());
    let empty_graph = Graph::new();

    let gap = KnowledgeGap {
        gap_type: KnowledgeGapType::MissingFieldGap,
        missing_role: Some(SemanticRole::Arg0Agent),
        source_composition_id: None,
        ..test_gap()
    };

    let decision = sa.select_strategy(&gap, &empty_graph);
    assert_eq!(decision.mode, AcquisitionMode::AskUser);
}
```

**Apa yang dicek**: Gap detection menemukan missing roles, dan acquisition hierarchy mengikuti doctrine "Remember first, Study second, Ask last".

### B.5 — MD-5: Executive Mode Selection Test

```rust
#[test]
fn executive_selects_reactive_for_token() {
    let mode = select_cognitive_mode("raja", &Graph::new());
    assert_eq!(mode, CognitiveMode::Reactive);
}

#[test]
fn executive_selects_analytical_for_sentence() {
    let mode = select_cognitive_mode("Raymond membuat aplikasi karena lambat", &Graph::new());
    assert_eq!(mode, CognitiveMode::Analytical);
}

#[test]
fn executive_selects_reflective_for_contradictions() {
    let mut graph = Graph::new();
    // Add contradicted compositions in neighborhood
    graph.add_contradicted_composition("membuat");
    let mode = select_cognitive_mode("membuat aplikasi", &graph);
    assert_eq!(mode, CognitiveMode::Reflective);
}
```

```rust
#[test]
fn compute_budget_per_mode() {
    let reactive = ComputeBudget::for_mode(&CognitiveMode::Reactive);
    assert_eq!(reactive.max_enrichment_rounds, 0);
    assert_eq!(reactive.max_reflection_loops, 0);

    let analytical = ComputeBudget::for_mode(&CognitiveMode::Analytical);
    assert_eq!(analytical.max_enrichment_rounds, 1);
    assert_eq!(analytical.max_reflection_loops, 1);

    let reflective = ComputeBudget::for_mode(&CognitiveMode::Reflective);
    assert_eq!(reflective.max_enrichment_rounds, 2);
    assert_eq!(reflective.max_reflection_loops, 2);
}
```

**Apa yang dicek**: Mode selection deterministik dan compute budget sesuai MD-5 spec (0/1/2 enrichment rounds).

### B.6 — Closed Feedback Loop Test (KRITIS)

> **Ini test terpenting** — memvalidasi bahwa seluruh siklus gap → enrich → re-govern bekerja.

```rust
#[test]
fn feedback_loop_gap_to_enrich_to_re_govern() {
    let mut engine = register_default_pipeline(PipelineContext::new());

    // 1. Ingest sentence yang menghasilkan Event tanpa Agent
    let result = engine.ingest("membuat aplikasi karena lambat");
    assert!(result.is_ok());

    // 2. Cek bahwa gap terdeteksi
    let snapshot = engine.snapshot();
    let gaps = engine.run::<DetectGaps>(&snapshot);
    assert!(!gaps.is_empty());
    let agent_gap = gaps.iter()
        .find(|g| g.missing_role == Some(SemanticRole::Arg0Agent));
    assert!(agent_gap.is_some());

    // 3. Ingest lagi dengan full sentence → graph sekarang punya "Raymond" sbg Agent
    engine.ingest("Raymond membuat aplikasi karena lambat");

    // 4. SelectAcquisition seharusnya menemukan "Raymond" sbg candidate via PassiveRecall
    let decisions = engine.run::<SelectAcquisition>(&gaps);
    let recall_decision = decisions.iter()
        .find(|d| d.mode == AcquisitionMode::PassiveRecall);
    assert!(recall_decision.is_some());

    // 5. EnrichComposition seharusnya bisa fill missing Agent
    if let Some(decision) = recall_decision {
        if let Some(RecallAction::EnrichComposition { target_composition_id, role_to_fill, candidate_node_id }) = &decision.action {
            let request = EnrichmentRequest {
                target_composition_id: target_composition_id.clone(),
                role_to_fill: role_to_fill.clone(),
                candidate_node_id: *candidate_node_id,
                candidate_label: "Raymond".into(),
                source: EnrichmentSource::PassiveRecall,
                confidence: 0.7,
            };
            let delta = engine.run::<EnrichComposition>(&request);
            let governed = engine.run::<GovernBeliefs>(&delta);
            let anchored = engine.run::<SeedAnchor>(&governed);
            engine.apply(anchored);

            // 6. Composition sekarang punya Agent
            let comp = engine.graph().get_composition(target_composition_id);
            assert!(comp.is_some());
            let comp = comp.unwrap();
            assert!(comp.member_with_role(&SemanticRole::Arg0Agent).is_some());

            // 7. Re-governance: lifecycle should have advanced
            // (New, Observed) → (Candidate, Observed) after enrichment
            assert!(comp.lifecycle == LifecycleState::Candidate
                || comp.lifecycle == LifecycleState::Stable);
        }
    }
}
```

**Apa yang dicek**: Seluruh feedback loop MD-3: ExtractFrame → Ingest → DetectGaps → SelectAcquisition → EnrichComposition → GovernBeliefs → SeedAnchor → apply. Ini adalah **test integrasi terpenting** karena memvalidasi 5 dari 6 MD bekerja bersama.

---

## 4. Pilihan C: Full E2E Validation (1 hari)

> **Untuk**: Validasi lengkap bahwa v12.0 pipeline bekerja end-to-end dari text input sampai graph state akhir, termasuk executive mode selection dan enrichment loop.

### C.1 — Full Analytical Mode Pipeline

```rust
#[test]
fn analytical_mode_full_pipeline_indonesian() {
    let mut rsvs = create_v12_rsvs();
    let result = rsvs.ingest_text("Raymond membuat aplikasi karena proses manual terlalu lambat");

    // Verify:
    // 1. Token atoms created for each token
    // 2. Event atom created with correct roles
    // 3. HiddenMeaning atom created (problem_solution)
    // 4. Composition created in graph with correct lifecycle
    // 5. Gap detection ran (if any gaps)
    // 6. Enrichment loop ran (max 1 round for Analytical)

    let graph = rsvs.v12_graph();
    let event_comps: Vec<_> = graph.compositions()
        .filter(|c| c.composition_type == CompositionType::Event)
        .collect();
    assert!(!event_comps.is_empty());

    let hm_comps: Vec<_> = graph.compositions()
        .filter(|c| c.composition_type == CompositionType::HiddenMeaning)
        .collect();
    assert!(!hm_comps.is_empty());

    // HiddenMeaning harus (Quarantine, Inferred)
    let hm = &hm_comps[0];
    assert_eq!(hm.lifecycle, LifecycleState::Quarantine);
    assert_eq!(hm.epistemic, EpistemicState::Inferred);
}
```

### C.2 — Full Reflective Mode Pipeline

```rust
#[test]
fn reflective_mode_with_contradiction_resolution() {
    let mut rsvs = create_v12_rsvs();

    // Feed contradictory sentences
    rsvs.ingest_text("Raymond membuat aplikasi");
    rsvs.ingest_text("Raymond tidak membuat aplikasi");

    // Reflective mode should be triggered by local contradictions
    let graph = rsvs.v12_graph();
    let contradicted: Vec<_> = graph.compositions()
        .filter(|c| c.epistemic == EpistemicState::Contradicted)
        .collect();
    assert!(!contradicted.is_empty());

    // Check that contradiction has opposing_composition_id
    for comp in &contradicted {
        if let Some(opp_id) = comp.contradiction_opposing_id() {
            let opp = graph.get_composition(&opp_id);
            assert!(opp.is_some());
        }
    }
}
```

### C.3 — User Answer Processing & Merge

```rust
#[test]
fn user_answer_merges_into_original_composition() {
    let mut rsvs = create_v12_rsvs();

    // Ingest incomplete sentence → produces gap
    rsvs.ingest_text("membuat aplikasi");
    let gaps = rsvs.v12_detect_gaps();
    let agent_gap = gaps.iter()
        .find(|g| g.missing_role == Some(SemanticRole::Arg0Agent));
    assert!(agent_gap.is_some());

    // Generate question
    let question = rsvs.v12_generate_question(agent_gap.unwrap());
    assert!(question.is_some());
    assert_eq!(question.unwrap().question_type, InquiryQuestionType::MissingFieldClarification);

    // User answers
    let answer = "Raymond";
    let merge_result = process_user_answer_merge(
        answer,
        &question.unwrap(),
        agent_gap.unwrap(),
    );
    assert!(merge_result.is_ok());
    let request = merge_result.unwrap();
    assert_eq!(request.role_to_fill, SemanticRole::Arg0Agent);
    assert_eq!(request.candidate_label, "Raymond");
    assert_eq!(request.source, EnrichmentSource::UserAnswerMerge);

    // Apply enrichment
    rsvs.v12_apply_enrichment(request);

    // Verify composition now has Agent
    let graph = rsvs.v12_graph();
    let comp = graph.get_composition(&agent_gap.unwrap().source_composition_id.unwrap());
    assert!(comp.unwrap().member_with_role(&SemanticRole::Arg0Agent).is_some());
}
```

### C.4 — ReExtraction with Graph Context

```rust
#[test]
fn re_extract_frame_with_graph_context() {
    let mut rsvs = create_v12_rsvs();

    // First: full sentence → graph knows "Raymond" is Agent for "membuat"
    rsvs.ingest_text("Raymond membuat aplikasi karena lambat");

    // Second: incomplete sentence → missing Agent
    rsvs.ingest_text("membuat aplikasi karena cepat");

    // Find weak frames (low confidence, missing Agent)
    let weak_frames = rsvs.v12_find_weak_frames();
    assert!(!weak_frames.is_empty());

    // Re-extract with graph context
    for weak_frame in &weak_frames {
        let context = rsvs.v12_context_for(weak_frame);
        let request = ReExtractionRequest {
            original_text: weak_frame.source_text().unwrap_or_default().to_string(),
            original_atom_id: weak_frame.atom_id().to_string(),
            target_composition_id: weak_frame.composition_id().clone(),
            graph_context: context,
        };

        if let Some(improved) = rsvs.v12_re_extract_frame(&request) {
            assert!(improved.variant == Some(AtomVariant::FrameVariant(FrameSource::GraphAssisted))
                || improved.roles.contains_key(&SemanticRole::Arg0Agent));
        }
    }
}
```

### C.5 — Enrichment Loop Bounded by max_enrichment_rounds

```rust
#[test]
fn enrichment_loop_stops_at_budget_limit() {
    let mut rsvs = create_v12_rsvs_with_mode(CognitiveMode::Reflective);
    // Reflective: max_enrichment_rounds = 2

    // Feed text that produces many gaps
    rsvs.ingest_text("membuat karena untuk"); // extremely sparse

    // Check that enrichment ran at most 2 rounds
    let stats = rsvs.v12_last_ingest_stats();
    assert!(stats.enrichment_rounds <= 2);
}
```

### C.6 — SeedAnchor Confidence Adjustment

```rust
#[test]
fn seed_anchor_adjusts_confidence_only_with_alignment_data() {
    let sa = SeedAnchor::default();

    // No alignment data → no adjustment (weight = 0.0)
    let default_scores = HashMap::new(); // all defaults = 0.5
    let adjustment = sa.seed_anchored_confidence(&default_scores);
    assert_eq!(adjustment.weight, 0.0); // CRITICAL: no adjustment without real data

    // With real alignment data → adjustment applied
    let mut real_scores = HashMap::new();
    real_scores.insert(SeedPrimitive::Trust, 0.9);
    real_scores.insert(SeedPrimitive::Risk, 0.2);
    let adjustment = sa.seed_anchored_confidence(&real_scores);
    assert!(adjustment.weight > 0.0);
    assert!(adjustment.seed_confidence > 0.5);
}
```

### C.7 — ExtractionQualityTracker

```rust
#[test]
fn extraction_quality_tracks_gap_and_repair_rates() {
    let mut tracker = ExtractionQualityTracker::new();

    // Record 10 extractions with rule "CAUSE_ACTION_PATIENT"
    for _ in 0..10 {
        tracker.record_extraction("CAUSE_ACTION_PATIENT", 0.6);
    }
    // 3 gaps detected
    for _ in 0..3 {
        tracker.record_gap("CAUSE_ACTION_PATIENT", "MissingFieldGap");
    }
    // 1 gap repaired
    tracker.record_repair("CAUSE_ACTION_PATIENT");

    let quality = tracker.get("CAUSE_ACTION_PATIENT").unwrap();
    assert!((quality.gap_rate() - 0.30).abs() < 0.01);
    assert!((quality.repair_rate() - 1.0/3.0).abs() < 0.01);
    assert!(!quality.is_weak()); // gap_rate=0.30, repair_rate=0.33 → NOT weak (need repair_rate<0.50)
}
```

### C.8 — AmbiguousToken Detection and Resolution

```rust
#[test]
fn ambiguous_token_produces_gap() {
    let dg = DetectGaps::new(GapDetectionConfig::default());

    let ambiguous = SemanticAtom {
        id: "tok_5".into(), label: "dia".into(),
        atom_type: AtomType::AmbiguousToken, roles: HashMap::new(),
        polarity: None, voice: None, variant: None,
        confidence: 0.3, source: EdgeSource::Bootstrap,
    };

    let snapshot = GraphSnapshot::from_atoms(vec![ambiguous]);
    let gaps = dg.transform(&snapshot, &mut PipelineContext::new());
    assert!(gaps.iter().any(|g| g.gap_type == KnowledgeGapType::AmbiguousReferenceGap));
}
```

---

## 5. Pilihan D: Stress & Edge Case Test (2 hari)

> **Untuk**: Validasi bahwa v12.0 menangani edge cases, input aneh, dan stress conditions tanpa crash atau data corruption.

### D.1 — Empty Input

```rust
#[test]
fn empty_input_does_not_crash() {
    let mut rsvs = create_v12_rsvs();
    let result = rsvs.ingest_text("");
    assert!(result.is_ok());
}
```

### D.2 — Very Long Sentence

```rust
#[test]
fn very_long_sentence_does_not_overflow() {
    let mut rsvs = create_v12_rsvs();
    let long = format!("Raymond membuat {}", "aplikasi ".repeat(1000));
    let result = rsvs.ingest_text(&long);
    assert!(result.is_ok());
}
```

### D.3 — Rapid Contradiction Cycle

```rust
#[test]
fn rapid_contradiction_cycle_stabilizes() {
    let mut rsvs = create_v12_rsvs();
    // Feed alternating contradictory sentences 50 times
    for i in 0..50 {
        if i % 2 == 0 {
            rsvs.ingest_text("Raymond membuat aplikasi");
        } else {
            rsvs.ingest_text("Raymond tidak membuat aplikasi");
        }
    }
    // Graph should not have infinite compositions
    let graph = rsvs.v12_graph();
    let count = graph.composition_count();
    assert!(count < 100, "Should converge, not explode");
}
```

### D.4 — Unicode and Non-Latin Input

```rust
#[test]
fn indonesian_unicode_handled() {
    let mut rsvs = create_v12_rsvs();
    let result = rsvs.ingest_text("Raymond membuat aplikasi karena proses manual terlalu lambat");
    assert!(result.is_ok());
}
```

### D.5 — Enrichment Loop Convergence

```rust
#[test]
fn enrichment_loop_converges_even_with_recursive_gaps() {
    // Scenario: enrichment creates new gaps, which trigger more enrichment
    // But should still converge (bounded by max_enrichment_rounds)
    let mut rsvs = create_v12_rsvs_with_mode(CognitiveMode::Reflective);
    rsvs.ingest_text("membuat karena untuk dengan"); // multiple missing roles

    let stats = rsvs.v12_last_ingest_stats();
    assert!(stats.enrichment_rounds <= 2); // reflective max
    assert!(stats.total_time_ms < 10000); // within time budget
}
```

### D.6 — GraphNeighborhood for Empty Graph

```rust
#[test]
fn graph_neighborhood_empty_graph_is_safe() {
    let graph = Graph::new();
    let keywords = extract_keywords("Raymond membuat aplikasi");
    let neighborhood = graph.neighborhood_for(&keywords);
    assert!(!neighborhood.has_contradictions());
    assert_eq!(neighborhood.average_confidence(), 0.0);
}
```

### D.7 — Provenance Source Count

```rust
#[test]
fn provenance_source_count_counts_independent_sources() {
    let mut graph = Graph::new();
    let comp = test_composition_with_two_sources();
    let count = comp.provenance_source_count(&graph);
    assert!(count >= 2); // FrameCompiler + EnrichmentFeedback
}
```

### D.8 — Backward Compatibility (v11.0)

```rust
#[test]
fn v11_pipeline_unaffected_when_v12_disabled() {
    let mut rsvs = Rsvs::new(PipelineConfig {
        executive_enabled: false,
        ..PipelineConfig::default()
    }).unwrap();
    let stats = rsvs.ingest_text("Stone is hard and rough. Stone is solid and heavy.");
    assert!(stats.sentences_processed > 0);
    // All existing v11 tests should still pass
}
```

---

## 6. Test Mandiri per MD

### MD-1 (ExtractFrame) — 9 Test

| # | Test | Status di Codebase | Priority |
|---|------|--------------------|----------|
| 1 | Active Indonesian sentence extraction | ✅ Ada (`active_indonesian_sentence`) | P1 |
| 2 | Passive Indonesian sentence | ✅ Ada (`passive_indonesian_sentence`) | P1 |
| 3 | Negated sentence (tidak/bukan) | ✅ Ada (`negated_indonesian_sentence`) | P1 |
| 4 | Token input → None | ✅ Ada (`token_returns_none`) | P1 |
| 5 | Sentence with Cause (karena) | ✅ Ada | P1 |
| 6 | Sentence with Purpose (untuk) | ✅ Ada | P2 |
| 7 | ExtractionQualityTracker | ✅ Ada | P2 |
| 8 | Re-extraction with graph context | ⚠️ Partial | P1 |
| 9 | Graph-assisted variant marking | ❌ Missing | P1 |

### MD-2 (ReasonFrame) — 5 Test

| # | Test | Status | Priority |
|---|------|--------|----------|
| 1 | ProblemSolution rule | ✅ Ada | P1 |
| 2 | GoalInference rule | ✅ Ada | P1 |
| 3 | PolarityConflict rule | ✅ Ada | P1 |
| 4 | No output from incomplete event | ✅ Ada | P2 |
| 5 | HiddenMeaning → (Quarantine, Inferred) on ingest | ⚠️ Partial | P1 |

### MD-4 (GovernBeliefs) — 9 Test

| # | Test | Status | Priority |
|---|------|--------|----------|
| 1 | Initial state: Event → (New, Observed) | ✅ Ada | P1 |
| 2 | Initial state: HiddenMeaning → (Quarantine, Inferred) | ✅ Ada | P1 |
| 3 | Initial state: Acquisition/UserAnswer → (Candidate, Observed) | ✅ Ada | P1 |
| 4 | Contradiction detection: polarity conflict | ✅ Ada | P1 |
| 5 | Contradiction detection: role reversal | ⚠️ Partial | P1 |
| 6 | Promotion: Candidate → Stable | ✅ Ada | P1 |
| 7 | Promotion: Inferred → Grounded (needs 2 independent sources) | ⚠️ Simplified | P1 |
| 8 | Re-governance after enrichment | ⚠️ Partial | P1 |
| 9 | Contradiction resolution: voice confusion | ❌ Missing | P2 |

### MD-5 (Executive) — 8 Test

| # | Test | Status | Priority |
|---|------|--------|----------|
| 1 | Reactive mode for token | ✅ Ada | P1 |
| 2 | Analytical mode for sentence | ✅ Ada | P1 |
| 3 | Reflective mode for contradictions | ✅ Ada | P1 |
| 4 | ComputeBudget per mode | ✅ Ada | P1 |
| 5 | max_enrichment_rounds per mode | ⚠️ May be missing | P1 |
| 6 | StopCondition: confidence sufficient | ✅ Ada | P2 |
| 7 | StopCondition: budget exhausted | ✅ Ada | P2 |
| 8 | Reflect detects stagnant inferred | ⚠️ Partial | P2 |

### MD-6 (Acquisition) — 7 Test

| # | Test | Status | Priority |
|---|------|--------|----------|
| 1 | Gap detection: Event missing Agent | ✅ Ada | P1 |
| 2 | Gap detection: Event missing Patient | ✅ Ada | P1 |
| 3 | Gap detection: AmbiguousToken | ⚠️ Partial | P1 |
| 4 | Acquisition: PassiveRecall preferred | ✅ Ada | P1 |
| 5 | Acquisition: AskUser when no graph context | ✅ Ada | P1 |
| 6 | process_user_answer() | ❌ Missing (CRITICAL) | P0 |
| 7 | process_user_answer_merge() | ❌ Missing (CRITICAL) | P0 |

---

## 7. Test Cross-Cutting (Antar MD)

Test ini memvalidasi bahwa beberapa MD bekerja bersama secara benar.

### X.1 — MD-1 → MD-2 → MD-4 Pipeline

```
Input text → ExtractFrame (MD-1) → ReasonFrame (MD-2) → IngestAtoms → GovernBeliefs (MD-4)
```

**Validasi**:
- Event atom dari MD-1 punya `source: EdgeSource::FrameCompiler`
- HiddenMeaning atom dari MD-2 punya `source: EdgeSource::HiddenMeaningRule`
- GovernBeliefs assigns `(New, Observed)` untuk Event dan `(Quarantine, Inferred)` untuk HiddenMeaning

### X.2 — MD-6 → MD-3 (Feedback Loop) → MD-4

```
DetectGaps (MD-6) → SelectAcquisition (MD-6) → EnrichComposition (MD-3) → re_govern (MD-4)
```

**Validasi**:
- Gap yang terdeteksi punya `source_composition_id` → bisa trace ke composition
- PassiveRecall menghasilkan `RecallAction::EnrichComposition` dengan `target_composition_id`
- Setelah enrichment, composition lifecycle bisa naik (Quarantine → Candidate atau New → Candidate)
- Re-governance mengevaluasi `can_promote_to_grounded()` dengan full criteria

### X.3 — MD-5 (Executive) → MD-1 + MD-2 + MD-4 + MD-6

```
ExecutiveOrchestrator → select mode → run appropriate transforms → enrichment loop
```

**Validasi**:
- Reactive: hanya Tokenize + Ingest + Govern + Seed (no ExtractFrame, no gap detection)
- Analytical: full chain + 1 enrichment round
- Reflective: full chain + 2 enrichment rounds + reflection

### X.4 — Full Cycle: User Question → Answer → Merge

```
Gap → InquiryQuestion → User answers → process_user_answer_merge() → EnrichmentRequest → EnrichComposition → re_govern
```

**Validasi**:
- `process_user_answer_merge()` menghasilkan `EnrichmentRequest` (bukan separate SemanticAtom)
- `EnrichmentRequest.source == EnrichmentSource::UserAnswerMerge`
- Composition setelah merge punya Agent/Patient yang dijawab user
- InquiryMemory mencatat bahwa gap sudah resolved

---

## 8. Test untuk Mismatch yang Ditemukan Audit

Test berikut secara spesifik menargetkan **mismatch** yang ditemukan saat design-vs-implementation comparison audit:

### M.1 — `process_user_answer()` dan `process_user_answer_merge()` (KRITIS - MD-6)

```rust
#[test]
fn process_user_answer_creates_acquisition_atom() {
    let question = InquiryQuestion {
        question_id: "q_gap_1".into(),
        gap_id: "gap_1".into(),
        question_type: InquiryQuestionType::MissingFieldClarification,
        question_text: "Who performed this action?".into(),
        expected_answer_shape: ExpectedAnswerType::Entity,
    };

    let atom = process_user_answer("Raymond", &question);
    assert_eq!(atom.atom_type, AtomType::Acquisition);
    assert_eq!(atom.source, EdgeSource::AcquisitionUserAnswer);
    assert_eq!(atom.confidence, 0.85);
    assert_eq!(atom.variant, Some(AtomVariant::AcquisitionVariant(AcquisitionSource::UserAnswer)));
}

#[test]
fn process_user_answer_merge_creates_enrichment_request() {
    let question = InquiryQuestion {
        question_id: "q_gap_1".into(),
        gap_id: "gap_1".into(),
        question_type: InquiryQuestionType::MissingFieldClarification,
        question_text: "Who performed this action?".into(),
        expected_answer_shape: ExpectedAnswerType::Entity,
    };
    let gap = KnowledgeGap {
        gap_id: "gap_1".into(),
        gap_type: KnowledgeGapType::MissingFieldGap,
        missing_role: Some(SemanticRole::Arg0Agent),
        source_composition_id: Some("comp_1".into()),
        ..test_gap()
    };

    let result = process_user_answer_merge("Raymond", &question, &gap);
    assert!(result.is_ok());
    let request = result.unwrap();
    assert_eq!(request.target_composition_id, "comp_1");
    assert_eq!(request.role_to_fill, SemanticRole::Arg0Agent);
    assert_eq!(request.candidate_label, "Raymond");
    assert_eq!(request.source, EnrichmentSource::UserAnswerMerge);
}
```

### M.2 — `Graph::neighborhood_for()` (KRITIS - MD-3)

```rust
#[test]
fn graph_neighborhood_for_returns_local_compositions() {
    let mut graph = Graph::new();
    // Add compositions with keyword "membuat"
    graph.add_composition(test_event_composition("membuat", "Raymond", "aplikasi", Polarity::Positive));
    graph.add_composition(test_event_composition("membuat", "Andi", "website", Polarity::Positive));
    // Add unrelated composition
    graph.add_composition(test_event_composition("menggambar", "Budi", "lukisan", Polarity::Positive));

    let keywords = extract_keywords("membuat aplikasi");
    let neighborhood = graph.neighborhood_for(&keywords);

    // Should include "membuat" compositions but not "menggambar"
    assert!(neighborhood.compositions.len() >= 2);
    assert!(neighborhood.compositions.iter().all(|c|
        c.member_with_role(&SemanticRole::Predicate)
            .map(|m| m.label() == "membuat")
            .unwrap_or(false)
    ));
}
```

### M.3 — `extract_keywords()` (MD-3)

```rust
#[test]
fn extract_keywords_returns_content_words() {
    let keywords = extract_keywords("Raymond membuat aplikasi karena lambat");
    assert!(keywords.contains(&"Raymond".to_string())
         || keywords.contains(&"membuat".to_string())
         || keywords.contains(&"aplikasi".to_string()));
    // Should NOT contain stopwords
    assert!(!keywords.contains(&"karena".to_string()));
}
```

### M.4 — `provenance_source_count()` (MD-3)

```rust
#[test]
fn provenance_source_count_counts_unique_edge_sources() {
    let mut graph = Graph::new();
    let comp_id = CompositionId::new();

    // Composition from FrameCompiler
    let mut comp = test_composition_with_id(&comp_id, CompositionType::Event);
    comp.provenance.origin = EdgeSource::FrameCompiler;

    // Add edge from EnrichmentFeedback
    graph.add_edge(comp_id, node_id_1, SemanticEdge {
        relation: RelationType::Categorical,
        role: Some(SemanticRole::Arg0Agent),
        source: EdgeSource::EnrichmentFeedback,
    });

    graph.add_composition(comp);

    let comp = graph.get_composition(&comp_id).unwrap();
    let count = comp.provenance_source_count(&graph);
    assert!(count >= 2); // FrameCompiler + EnrichmentFeedback
}
```

### M.5 — Promotion Criteria: Full Spec (MD-4)

```rust
#[test]
fn can_promote_to_grounded_requires_independent_sources() {
    let gb = GovernBeliefs::new(GovernanceConfig::default());
    let mut graph = Graph::new();

    // Composition with only 1 source → cannot promote to Grounded
    let comp_single = test_composition_with_sources(1);
    let verdict = gb.can_promote_to_grounded(&comp_single, &graph);
    assert!(matches!(verdict, PromotionVerdict::Denied(reason) if reason.contains("source")));

    // Composition with 2 independent sources + high confidence + no contradictions
    let comp_multi = test_composition_with_sources(2);
    let verdict = gb.can_promote_to_grounded(&comp_multi, &graph);
    assert!(matches!(verdict, PromotionVerdict::Approved));
}

#[test]
fn can_promote_to_stable_requires_seed_alignment() {
    let gb = GovernBeliefs::new(GovernanceConfig::default());
    let graph = Graph::new();

    // Composition with negative seed alignment → denied
    let mut comp = test_mature_composition();
    comp.seed_scores.insert(SeedPrimitive::Trust, 0.1);
    comp.seed_scores.insert(SeedPrimitive::Risk, 0.9);
    let verdict = gb.can_promote_to_stable(&comp, &graph);
    assert!(matches!(verdict, PromotionVerdict::Denied(reason) if reason.contains("seed")));
}
```

### M.6 — `max_enrichment_rounds` in ComputeBudget (MD-5)

```rust
#[test]
fn compute_budget_includes_max_enrichment_rounds() {
    let budget = ComputeBudget::for_mode(&CognitiveMode::Reflective);
    assert_eq!(budget.max_enrichment_rounds, 2);

    let budget = ComputeBudget::for_mode(&CognitiveMode::Analytical);
    assert_eq!(budget.max_enrichment_rounds, 1);

    let budget = ComputeBudget::for_mode(&CognitiveMode::Reactive);
    assert_eq!(budget.max_enrichment_rounds, 0);
}
```

### M.7 — `CompositionMember::label()` (MD-3)

```rust
#[test]
fn composition_member_label_returns_node_label() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("Raymond");
    let member = CompositionMember {
        node_id,
        role: SemanticRole::Arg0Agent,
        confidence: 0.8,
    };
    // label() should return "Raymond", not empty string
    assert_eq!(member.label(), "Raymond");
}
```

---

## 9. Rekomendasi Prioritas Eksekusi

### Jika waktu sangat terbatas (1 jam):

```
1. cargo check --features v12          (2 menit)
2. cargo test --features v12           (5 menit)
3. Test B.6: Feedback Loop Test        (15 menit)
4. Test M.1: process_user_answer*      (10 menit)
5. Test B.3: GovernBeliefs promotion   (10 menit)
```

### Jika waktu cukup (setengah hari):

```
1. Semua Pilihan A (Smoke Test)        (30 menit)
2. Semua Pilihan B (Focused Int.)      (2-3 jam)
3. Test M.1-M.7 (Audit Mismatches)     (1 jam)
```

### Jika ingin validasi lengkap (1-2 hari):

```
1. Semua Pilihan A + B + C             (1 hari)
2. Semua Test M.1-M.7                  (2 jam)
3. Pilihan D (Stress & Edge Cases)     (4 jam)
4. Cross-Cutting Tests X.1-X.4         (2 jam)
```

---

## Quick Reference: Command Test

Untuk menjalankan test yang sudah ada:

```bash
# Semua v12 tests
cd layer1 && cargo test --features v12

# Hanya test di modul tertentu
cargo test --features v12 extract_frame
cargo test --features v12 reason_frame
cargo test --features v12 govern_beliefs
cargo test --features v12 acquisition
cargo test --features v12 executive
cargo test --features v12 pipeline

# Dengan output detail
cargo test --features v12 -- --nocapture

# Integration tests
cd layer1/tests && cargo test --features v12
```

---

> **Catatan Penting**: Test di atas ditulis sebagai **scenario dan pseudocode** — bukan copy-paste ready. Setiap test perlu disesuaikan dengan API actual yang ada di implementation. Prioritaskan test yang menargetkan **mismatch dari audit** (M.1-M.7) karena itu adalah gap paling kritis antara design dan implementation.
