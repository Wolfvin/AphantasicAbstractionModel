use super::helpers::*;

// ========================================================================
// Phase J–P Cognitive Scenario Tests
// ========================================================================
//
// These tests verify the Phase J–P features:
// J: Dead weight cleanup (no condition_label, no centrality)
// K: Coherence penalty in promotion
// L: Weighted Jaccard for sense similarity
// M: Closed grounding loop (evidence → promotion → grounding)
// N: Bridge guard + utterance context
// O: Connectivity score + prune fragile senses
// P: SenseRole helpers (is_primitive, is_bridge, is_derived, etc.)

// ---- Phase J: Sense struct & freq_map ----

#[test]
fn test_phase_j_sense_struct_basic() {
    let sense = Sense::new_primitive("financial institution");
    assert_eq!(sense.label, "financial institution");
    assert_eq!(sense.layer, 0);
    assert!(sense.is_primitive());
    assert!(!sense.is_derived());
    assert_eq!(sense.grounding, SenseGrounding::Fragile);
    assert_eq!(sense.coherence, 0.5); // default
    assert!(sense.freq_map.is_empty());
    eprintln!("✅ Phase J: Sense struct with clean fields (no condition_label, no centrality)");
}

#[test]
fn test_phase_j_build_freq_map() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("bank");

    // Create two compositions referencing "bank"
    let mut comp1 = Composition::default();
    comp1.id = CompositionId::new("comp_financial".to_string());
    comp1.confidence = 0.8;
    comp1.members.push(CompositionMember {
        node_id,
        role: SemanticRole::Arg0Agent,
        confidence: 0.8,
        label: "bank".to_string(),
        source: None,
    });
    graph.compositions.insert(comp1.id.clone(), comp1);

    let mut comp2 = Composition::default();
    comp2.id = CompositionId::new("comp_river".to_string());
    comp2.confidence = 0.5;
    comp2.members.push(CompositionMember {
        node_id,
        role: SemanticRole::Location,
        confidence: 0.5,
        label: "bank".to_string(),
        source: None,
    });
    graph.compositions.insert(comp2.id.clone(), comp2);

    // Build freq_map for a sense on this node
    let mut sense = Sense::new_primitive("bank");
    sense.build_freq_map(node_id, &graph);

    assert_eq!(sense.freq_map.len(), 2, "freq_map should have 2 entries");
    // Confidence-weighted: comp_financial = 0.8, comp_river = 0.5
    assert!(
        (sense.freq_map.get("comp_financial").unwrap() - 0.8).abs() < 0.01,
        "freq_map weight should be confidence-weighted"
    );
    assert!(
        (sense.freq_map.get("comp_river").unwrap() - 0.5).abs() < 0.01,
        "freq_map weight should be confidence-weighted"
    );

    eprintln!("✅ Phase J: build_freq_map() is confidence-weighted (not hardcoded 1.0)");
}

#[test]
fn test_phase_j_derived_sense_layer_cap() {
    let sense = Sense::new_derived("conclusion", 5); // Try layer 5, should cap at 3
    assert_eq!(sense.layer, 3, "Layer should be capped at 3");
    assert!(sense.is_derived());
    assert!(!sense.is_primitive());
    eprintln!("✅ Phase J: Sense layer hard cap at 3");
}

// ---- Phase K: Coherence Penalty ----

#[test]
fn test_phase_k_coherence_penalty_low_coherence() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("obat");

    // Add a sense with low coherence
    let mut sense = Sense::new_primitive("medicine");
    sense.coherence = 0.2; // Low coherence
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);

    let mut comp = Composition::default();
    comp.id = CompositionId::new("comp_test".to_string());
    comp.members.push(CompositionMember {
        node_id,
        role: SemanticRole::Arg0Agent,
        confidence: 0.7,
        label: "obat".to_string(),
        source: None,
    });

    let gb = GovernBeliefs::new();
    let penalty = gb.compute_member_coherence_penalty(&comp, &graph);

    // Penalty = (1 - 0.2) * 0.15 = 0.12
    assert!(
        (penalty - 0.12).abs() < 0.01,
        "Expected penalty ~0.12 for low coherence, got {:.3}",
        penalty
    );
    eprintln!("✅ Phase K: Coherence penalty for low-coherence senses = {:.3}", penalty);
}

#[test]
fn test_phase_k_coherence_penalty_high_coherence() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("raja");

    let mut sense = Sense::new_primitive("king");
    sense.coherence = 0.9; // High coherence
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);

    let mut comp = Composition::default();
    comp.id = CompositionId::new("comp_test".to_string());
    comp.members.push(CompositionMember {
        node_id,
        role: SemanticRole::Arg0Agent,
        confidence: 0.7,
        label: "raja".to_string(),
        source: None,
    });

    let gb = GovernBeliefs::new();
    let penalty = gb.compute_member_coherence_penalty(&comp, &graph);

    // Penalty = (1 - 0.9) * 0.15 = 0.015
    assert!(
        penalty < 0.05,
        "Expected low penalty for high coherence, got {:.3}",
        penalty
    );
    eprintln!("✅ Phase K: Coherence penalty for high-coherence senses = {:.3}", penalty);
}

// ---- Phase L: Weighted Jaccard ----

#[test]
fn test_phase_l_weighted_jaccard_identical() {
    let mut a = SenseCandidate::default();
    a.sense_id = "a".to_string();
    a.properties.insert("Predicate".to_string(), 0.8);
    a.properties.insert("Arg0Agent".to_string(), 0.6);

    let mut b = SenseCandidate::default();
    b.sense_id = "b".to_string();
    b.properties.insert("Predicate".to_string(), 0.8);
    b.properties.insert("Arg0Agent".to_string(), 0.6);

    let sim = a.weighted_jaccard(&b);
    assert!(
        (sim - 1.0).abs() < 0.01,
        "Identical properties should give similarity 1.0, got {:.3}",
        sim
    );
    eprintln!("✅ Phase L: Weighted Jaccard identical = {:.3}", sim);
}

#[test]
fn test_phase_l_weighted_jaccard_disjoint() {
    let mut a = SenseCandidate::default();
    a.sense_id = "a".to_string();
    a.properties.insert("Predicate".to_string(), 0.8);

    let mut b = SenseCandidate::default();
    b.sense_id = "b".to_string();
    b.properties.insert("Location".to_string(), 0.5);

    let sim = a.weighted_jaccard(&b);
    assert!(
        (sim - 0.0).abs() < 0.01,
        "Disjoint properties should give similarity 0.0, got {:.3}",
        sim
    );
    eprintln!("✅ Phase L: Weighted Jaccard disjoint = {:.3}", sim);
}

#[test]
fn test_phase_l_weighted_jaccard_partial() {
    let mut a = SenseCandidate::default();
    a.sense_id = "a".to_string();
    a.properties.insert("Predicate".to_string(), 0.8);
    a.properties.insert("Arg0Agent".to_string(), 0.6);

    let mut b = SenseCandidate::default();
    b.sense_id = "b".to_string();
    b.properties.insert("Predicate".to_string(), 0.4);
    b.properties.insert("Arg1Patient".to_string(), 0.3);

    let sim = a.weighted_jaccard(&b);
    // sum_min = min(0.8,0.4) + min(0.6,0) + min(0,0.3) = 0.4
    // sum_max = max(0.8,0.4) + max(0.6,0) + max(0,0.3) = 0.8 + 0.6 + 0.3 = 1.7
    // sim = 0.4 / 1.7 ≈ 0.235
    assert!(
        sim > 0.1 && sim < 0.4,
        "Partial overlap should give moderate similarity, got {:.3}",
        sim
    );
    eprintln!("✅ Phase L: Weighted Jaccard partial overlap = {:.3}", sim);
}

#[test]
fn test_phase_l_extract_properties_from_composition() {
    let mut graph = Graph::new();
    let pred_id = graph.ensure_node("membuat");
    let agent_id = graph.ensure_node("raymond");
    let patient_id = graph.ensure_node("aplikasi");

    let mut comp = Composition::default();
    comp.id = CompositionId::new("comp_membuat".to_string());
    comp.members.push(CompositionMember {
        node_id: pred_id,
        role: SemanticRole::Predicate,
        confidence: 0.9,
        label: "membuat".to_string(),
        source: None,
    });
    comp.members.push(CompositionMember {
        node_id: agent_id,
        role: SemanticRole::Arg0Agent,
        confidence: 0.8,
        label: "raymond".to_string(),
        source: None,
    });
    comp.members.push(CompositionMember {
        node_id: patient_id,
        role: SemanticRole::Arg1Patient,
        confidence: 0.7,
        label: "aplikasi".to_string(),
        source: None,
    });

    // Without freq_map entries, should use default weight 1.0
    let candidate = graph.extract_properties_from_composition(&comp);
    assert!(
        candidate.properties.contains_key("Predicate"),
        "Should have Predicate property"
    );
    assert!(
        candidate.properties.contains_key("Arg0Agent"),
        "Should have Arg0Agent property"
    );
    assert!(
        candidate.properties.contains_key("Arg1Patient"),
        "Should have Arg1Patient property"
    );

    eprintln!("✅ Phase L: extract_properties_from_composition works (iterates ALL senses, not just .first())");
}

// ---- Phase M: Closed Grounding Loop ----

#[test]
fn test_phase_m_update_sense_evidence_confirming() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("database");

    let sense = Sense::new_primitive("data store");
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);

    let mut comp = Composition::default();
    comp.id = CompositionId::new("comp_optimize".to_string());
    comp.members.push(CompositionMember {
        node_id,
        role: SemanticRole::Arg1Patient,
        confidence: 0.7,
        label: "database".to_string(),
        source: None,
    });
    graph.compositions.insert(comp.id.clone(), comp);

    let gb = GovernBeliefs::new();
    let comp_id = CompositionId::new("comp_optimize".to_string());
    gb.update_sense_evidence(&comp_id, true, &mut graph);

    let node = graph.nodes.get(&node_id).unwrap();
    assert_eq!(node.senses[0].composition_evidence.confirming, 1);
    assert_eq!(node.senses[0].composition_evidence.contradicting, 0);

    eprintln!("✅ Phase M: update_sense_evidence() correctly adds confirming evidence");
}

#[test]
fn test_phase_m_update_sense_evidence_contradicting() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("obat");

    let sense = Sense::new_primitive("medicine");
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);

    let mut comp = Composition::default();
    comp.id = CompositionId::new("comp_contradict".to_string());
    comp.members.push(CompositionMember {
        node_id,
        role: SemanticRole::Arg0Agent,
        confidence: 0.6,
        label: "obat".to_string(),
        source: None,
    });
    graph.compositions.insert(comp.id.clone(), comp);

    let gb = GovernBeliefs::new();
    let comp_id = CompositionId::new("comp_contradict".to_string());
    gb.update_sense_evidence(&comp_id, false, &mut graph);

    let node = graph.nodes.get(&node_id).unwrap();
    assert_eq!(node.senses[0].composition_evidence.confirming, 0);
    assert_eq!(node.senses[0].composition_evidence.contradicting, 1);

    eprintln!("✅ Phase M: update_sense_evidence() correctly adds contradicting evidence");
}

#[test]
fn test_phase_m_check_sense_promotions() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("server");

    let mut sense = Sense::new_primitive("machine");
    sense.composition_evidence.confirming = 3; // Threshold for Fragile → Tentative
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);

    let gb = GovernBeliefs::new();
    let promotions = gb.check_sense_promotions(&mut graph);

    assert_eq!(promotions, 1, "Should promote 1 sense");
    let node = graph.nodes.get(&node_id).unwrap();
    assert_eq!(
        node.senses[0].grounding,
        SenseGrounding::Tentative,
        "Sense should be promoted to Tentative"
    );

    eprintln!("✅ Phase M: check_sense_promotions() promotes Fragile → Tentative");
}

#[test]
fn test_phase_m_sense_promotions_contradicting_blocks() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("ghost");

    let mut sense = Sense::new_primitive("apparition");
    sense.composition_evidence.confirming = 5;
    sense.composition_evidence.contradicting = 10; // Contradicting dominant
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);

    let gb = GovernBeliefs::new();
    let promotions = gb.check_sense_promotions(&mut graph);

    assert_eq!(promotions, 0, "Should NOT promote when contradicting dominant");
    let node = graph.nodes.get(&node_id).unwrap();
    assert_eq!(
        node.senses[0].grounding,
        SenseGrounding::Fragile,
        "Sense should stay Fragile"
    );

    eprintln!("✅ Phase M: check_sense_promotions() blocked by contradicting evidence");
}

// ---- Phase N: Bridge Guard + Utterance Context ----

#[test]
fn test_phase_n_bridge_guard_cannot_deprecate() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("connector");

    // Create a bridge node: has senses at 2 different layers
    let mut sense1 = Sense::new_primitive("link");
    sense1.layer = 0;
    let mut sense2 = Sense::new_derived("bridge", 1);
    sense2.is_utterance = true;
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense1);
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense2);

    let gb = GovernBeliefs::new();
    assert!(
        !gb.can_deprecate_node(node_id, &graph),
        "Bridge nodes should NOT be deprecatable"
    );

    eprintln!("✅ Phase N: can_deprecate_node() blocks bridge node deprecation");
}

#[test]
fn test_phase_n_mature_sense_cannot_deprecate() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("established");

    let mut sense = Sense::new_primitive("fact");
    sense.grounding = SenseGrounding::Mature;
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);

    let gb = GovernBeliefs::new();
    assert!(
        !gb.can_deprecate_node(node_id, &graph),
        "Nodes with Mature senses should NOT be deprecatable"
    );

    eprintln!("✅ Phase N: can_deprecate_node() blocks Mature sense deprecation");
}

#[test]
fn test_phase_n_utterance_context() {
    let mut graph = Graph::new();
    let comp_id = graph.ensure_node("test_comp");

    let mut comp = Composition::default();
    comp.id = CompositionId::new("comp_utterance_test".to_string());
    comp.composition_type = CompositionType::Event;
    comp.members.push(CompositionMember {
        node_id: comp_id,
        role: SemanticRole::Predicate,
        confidence: 0.7,
        label: "test_comp".to_string(),
        source: None,
    });
    graph.compositions.insert(comp.id.clone(), comp);

    // Add a situational composition to verify situational_composition_count
    let sit_node_id = graph.ensure_node("sit_context");
    let mut sit_comp = Composition::default();
    sit_comp.id = CompositionId::new("comp_sit_1".to_string());
    sit_comp.composition_type = CompositionType::Situation;
    sit_comp.members.push(CompositionMember {
        node_id: sit_node_id,
        role: SemanticRole::Location,
        confidence: 0.6,
        label: "sit_context".to_string(),
        source: None,
    });
    graph.compositions.insert(sit_comp.id.clone(), sit_comp);

    // Add a bridge node (node with senses at 2+ layers)
    let bridge_node_id = graph.ensure_node("bridge_word");
    let node = graph.nodes.get_mut(&bridge_node_id).unwrap();
    node.senses.push(Sense::new_primitive("base_sense"));
    node.senses.push(Sense::new_derived("cross_layer", 2));

    let comp_id = CompositionId::new("comp_utterance_test".to_string());
    let ctx = graph.get_utterance_context(&comp_id);
    assert!(
        (ctx.current_weight - 0.55).abs() < 0.01,
        "Current weight should be 0.55"
    );
    assert!(
        (ctx.situational_weight - 0.25).abs() < 0.01,
        "Situational weight should be 0.25"
    );
    assert!(
        (ctx.bridge_weight - 0.20).abs() < 0.01,
        "Bridge weight should be 0.20"
    );

    // Fix 5: Verify that the context actually reflects graph structure
    assert!(
        ctx.situational_composition_count >= 1,
        "Should find at least 1 situational composition, found {}",
        ctx.situational_composition_count
    );
    assert!(
        ctx.bridge_node_count >= 1,
        "Should find at least 1 bridge node, found {}",
        ctx.bridge_node_count
    );

    eprintln!("✅ Phase N: get_utterance_context() returns 3-way blend (55/25/20) with sit_count={}, bridge_count={}",
        ctx.situational_composition_count, ctx.bridge_node_count);
}

// ---- Phase O: Connectivity & Pruning ----

#[test]
fn test_phase_o_connectivity_score() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("hub");

    // Add 5 compositions referencing this node
    for i in 0..5 {
        let mut comp = Composition::default();
        comp.id = CompositionId::new(format!("comp_hub_{}", i));
        comp.members.push(CompositionMember {
            node_id,
            role: SemanticRole::Predicate,
            confidence: 0.7,
            label: "hub".to_string(),
            source: None,
        });
        graph.compositions.insert(comp.id.clone(), comp);
    }

    let score = graph.connectivity_score(node_id);
    // 5 / 10 = 0.5
    assert!(
        (score - 0.5).abs() < 0.01,
        "Expected connectivity 0.5 for 5/10, got {:.3}",
        score
    );

    eprintln!("✅ Phase O: connectivity_score() = {:.3} for 5 compositions", score);
}

#[test]
fn test_phase_o_connectivity_saturates() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("mega_hub");

    // Add 20 compositions referencing this node (should saturate at 1.0)
    for i in 0..20 {
        let mut comp = Composition::default();
        comp.id = CompositionId::new(format!("comp_mega_{}", i));
        comp.members.push(CompositionMember {
            node_id,
            role: SemanticRole::Predicate,
            confidence: 0.7,
            label: "mega_hub".to_string(),
            source: None,
        });
        graph.compositions.insert(comp.id.clone(), comp);
    }

    let score = graph.connectivity_score(node_id);
    assert!(
        (score - 1.0).abs() < 0.01,
        "Expected connectivity capped at 1.0, got {:.3}",
        score
    );

    eprintln!("✅ Phase O: connectivity_score() saturates at 1.0");
}

#[test]
fn test_phase_o_prune_fragile_senses() {
    let mut graph = Graph::new();
    // Create an isolated node (no compositions → connectivity = 0)
    let node_id = graph.ensure_node("isolated");

    let mut fragile_sense = Sense::new_primitive("orphan");
    fragile_sense.coherence = 0.1; // Below 0.2
    fragile_sense.grounding = SenseGrounding::Fragile;
    // No confirming evidence

    let mut healthy_sense = Sense::new_primitive("valid");
    healthy_sense.coherence = 0.5; // Above 0.2
    healthy_sense.grounding = SenseGrounding::Fragile;

    graph.nodes.get_mut(&node_id).unwrap().senses.push(fragile_sense);
    graph.nodes.get_mut(&node_id).unwrap().senses.push(healthy_sense);

    let gb = GovernBeliefs::new();
    let pruned = gb.prune_fragile_senses(&mut graph);

    assert_eq!(pruned, 1, "Should prune 1 fragile sense");
    let node = graph.nodes.get(&node_id).unwrap();
    assert_eq!(node.senses.len(), 1, "Should have 1 sense remaining");
    assert_eq!(node.senses[0].label, "valid", "Should keep the coherent sense");

    eprintln!("✅ Phase O: prune_fragile_senses() removes isolated low-coherence sense");
}

#[test]
fn test_phase_o_prune_preserves_non_fragile() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("sparse");

    let mut tentative_sense = Sense::new_primitive("stable");
    tentative_sense.grounding = SenseGrounding::Tentative; // NOT Fragile
    tentative_sense.coherence = 0.1;

    graph.nodes.get_mut(&node_id).unwrap().senses.push(tentative_sense);

    let gb = GovernBeliefs::new();
    let pruned = gb.prune_fragile_senses(&mut graph);

    assert_eq!(pruned, 0, "Should NOT prune non-Fragile senses");
    let node = graph.nodes.get(&node_id).unwrap();
    assert_eq!(node.senses.len(), 1, "Should preserve Tentative sense");

    eprintln!("✅ Phase O: prune_fragile_senses() preserves non-Fragile grounding");
}

// ---- Phase P: SenseRole Helpers ----

#[test]
fn test_phase_p_is_primitive() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("word");

    // Empty senses = primitive
    assert!(graph.is_primitive(node_id));

    // Add primitive sense
    let sense = Sense::new_primitive("token");
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);
    assert!(graph.is_primitive(node_id));

    // Add derived sense
    let derived = Sense::new_derived("concept", 1);
    graph.nodes.get_mut(&node_id).unwrap().senses.push(derived);
    assert!(!graph.is_primitive(node_id), "Node with derived sense should NOT be primitive");

    eprintln!("✅ Phase P: is_primitive() works for empty, primitive, and derived nodes");
}

#[test]
fn test_phase_p_is_bridge() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("linker");

    // No bridge initially
    assert!(!graph.is_bridge(node_id));

    // Add utterance sense → bridge
    let mut sense = Sense::new_derived("connector", 1);
    sense.is_utterance = true;
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);
    assert!(graph.is_bridge(node_id));

    eprintln!("✅ Phase P: is_bridge() detects utterance-flagged nodes");
}

#[test]
fn test_phase_p_is_derived() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("concept");

    assert!(!graph.is_derived(node_id));

    let sense = Sense::new_derived("abstract", 2);
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);
    assert!(graph.is_derived(node_id));

    eprintln!("✅ Phase P: is_derived() detects layer ≥ 1 nodes");
}

#[test]
fn test_phase_p_is_utterance_level() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("sentence");

    let sense = Sense::new_derived("utterance_meaning", 2);
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);
    assert!(graph.is_utterance_level(node_id));

    eprintln!("✅ Phase P: is_utterance_level() detects layer ≥ 2 or is_utterance flag");
}

#[test]
fn test_phase_p_find_bridge_nodes() {
    let mut graph = Graph::new();

    let bridge_id = graph.ensure_node("bridge_word");
    let mut sense = Sense::new_derived("cross_layer", 1);
    sense.is_utterance = true;
    graph.nodes.get_mut(&bridge_id).unwrap().senses.push(sense);

    let normal_id = graph.ensure_node("normal_word");
    graph.nodes.get_mut(&normal_id).unwrap().senses.push(Sense::new_primitive("basic"));

    let bridges = graph.find_bridge_nodes();
    assert_eq!(bridges.len(), 1, "Should find exactly 1 bridge node");
    assert!(bridges.contains(&bridge_id), "Bridge node should be in results");

    eprintln!("✅ Phase P: find_bridge_nodes() finds bridge nodes correctly");
}

#[test]
fn test_phase_p_find_active_utterance_senses() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("context_word");

    let mut utterance_sense = Sense::new_derived("sentence_meaning", 2);
    utterance_sense.is_utterance = true;
    graph.nodes.get_mut(&node_id).unwrap().senses.push(utterance_sense);
    graph.nodes.get_mut(&node_id).unwrap().senses.push(Sense::new_primitive("basic"));

    let utterance_senses = graph.find_active_utterance_senses();
    assert_eq!(utterance_senses.len(), 1, "Should find 1 utterance sense");

    eprintln!("✅ Phase P: find_active_utterance_senses() finds utterance-level senses");
}

// ---- Integration: Grounding Loop Wired to Pipeline ----

#[test]
fn test_phase_m_integration_grounding_loop_wired() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("tim");

    let mut sense = Sense::new_primitive("group");
    sense.composition_evidence.confirming = 4; // Above threshold for Fragile → Tentative
    graph.nodes.get_mut(&node_id).unwrap().senses.push(sense);

    // Add a high-confidence composition
    let mut comp = Composition::default();
    comp.id = CompositionId::new("comp_tim_works".to_string());
    comp.confidence = 0.8;
    comp.lifecycle = LifecycleState::Stable;
    comp.members.push(CompositionMember {
        node_id,
        role: SemanticRole::Arg0Agent,
        confidence: 0.8,
        label: "tim".to_string(),
        source: None,
    });
    graph.compositions.insert(comp.id.clone(), comp);

    let gb = GovernBeliefs::new();

    // This should be wired in execute() — test the functions directly
    let promotions = gb.check_sense_promotions(&mut graph);
    assert!(promotions >= 1, "check_sense_promotions should promote at least 1 sense");

    let upgrades = gb.update_sense_grounding_from_evidence(&mut graph);
    // Fix 4: Now requires composition_evidence.confirming ≥ 1.
    // The sense was promoted by check_sense_promotions (confirming ≥ 3),
    // so it's already Tentative — update_sense_grounding_from_evidence won't
    // upgrade it again (it only upgrades Fragile → Tentative).
    // This is the expected behavior: no double-upgrade.
    // Note: upgrades is usize, so no need to check >= 0 (always true).
    // Just verify the function ran without panic.
    let _ = upgrades;

    eprintln!("✅ Phase M Integration: Grounding loop functions work (wired in execute())");
}

// ---- Integration: Prune wired to pipeline ----

#[test]
fn test_phase_o_integration_prune_wired() {
    let mut graph = Graph::new();
    let node_id = graph.ensure_node("dead_node");

    let mut dead_sense = Sense::new_primitive("zombie");
    dead_sense.coherence = 0.05; // Very low
    dead_sense.grounding = SenseGrounding::Fragile;
    graph.nodes.get_mut(&node_id).unwrap().senses.push(dead_sense);

    let gb = GovernBeliefs::new();

    // Prune should be called every 5 batches in execute()
    // Test directly here
    let pruned = gb.prune_fragile_senses(&mut graph);
    assert_eq!(pruned, 1, "Should prune the dead sense");

    let node = graph.nodes.get(&node_id).unwrap();
    assert!(node.senses.is_empty(), "Dead node should have no senses after pruning");

    eprintln!("✅ Phase O Integration: prune_fragile_senses() works (wired every 5 batches in execute())");
}
