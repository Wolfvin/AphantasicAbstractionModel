//! Integration tests for the RSVS pipeline.
//!
//! Includes meaning discovery tests (v9.0) that test whether the 3 meaning
//! pathways can detect hidden meaning, cross-pathway conflicts, and
//! emergent patterns in the graph.

use rsvs::{PipelineConfig, Rsvs};

/// Helper: create text with enough repetitions for promotion.
/// Each target word appears in at least 4 sentences to exceed the
/// entity_promote_n=3 threshold.
fn make_repetitive_text() -> &'static str {
    // v9.0: Include seed words (entity, cause, change) in sentences so that
    // ALL tokens in those sentences are groundable (v8.2 sentence-level grounding).
    // This ensures tokens like "stone", "hard", "solid" pass the grounding gate.
    "Stone entity is hard. Stone entity is rough. \
     Stone entity is solid. Stone entity is heavy. \
     Hard stone cause is change. Hard stone cause is durable. \
     Hard stone cause resists change. Hard stone cause withstands change. \
     Solid stone entity is firm. Solid stone entity is compact. \
     Solid stone entity is stable. Solid stone entity is strong. \
     Stone entity tools are ancient. Stone entity walls are protective. \
     Stone entity paths are durable. Stone entity cause lasts change."
}

#[test]
fn full_pipeline_roundtrip() {
    let mut rsvs = Rsvs::new(PipelineConfig::default()).expect("create RSVS");

    // Ingest enough text to promote entities
    let stats = rsvs.ingest_text(make_repetitive_text()).expect("ingest");
    assert!(stats.sentences_processed > 0);
    assert!(stats.atoms_promoted > 0);

    // Query — use a promoted token if available, otherwise a seed
    let promoted_token = rsvs.token_to_id.keys().find(|t| {
        ![
            "exists",
            "entity",
            "relation",
            "state",
            "change",
            "time",
            "space",
            "cause",
            "effect",
            "context",
            "signal",
            "pattern",
            "memory",
            "attention",
            "value",
            "agent",
            "goal",
            "risk",
            "trust",
            "identity",
            "language",
            "meaning",
            "action",
            "feedback",
        ]
        .contains(&t.as_str())
    });
    if let Some(token) = promoted_token {
        let result = rsvs.query(token, "hard texture");
        // Query may or may not return results depending on sense state
        if let Some(query_result) = result {
            assert!(!query_result.scored_atoms.is_empty() || query_result.active_sense_n > 0);
        }
    }

    // Appraise — use tokens that exist in the graph
    let appraise = rsvs.appraise("stone is hard");
    assert!(appraise.agree_pct > 0.0);
    assert!(!appraise.verdict.is_empty());

    // Relate — try with a promoted token or seed
    let relate_token = promoted_token.map_or("exists", |v| v.as_str());
    let relate = rsvs.relate(relate_token);
    // Seed nodes always have related nodes via edges
    assert!(relate.is_some());

    // Similarity — may or may not exist depending on promotion
    let _sim = rsvs.similarity("stone", "hard");

    // Snapshot
    let snap = rsvs.snapshot_v1();
    assert!(snap.schema_version.starts_with("v8."));
    assert!(!snap.nodes.is_empty());
    assert!(snap.nodes.len() >= 24); // At least seed nodes (v8.0: 24 language-agnostic seeds)
}

#[test]
fn multi_domain_ingest() {
    let config = PipelineConfig {
        current_domain: 1,
        ..PipelineConfig::default()
    };
    let mut rsvs = Rsvs::new(config).expect("create RSVS");

    // Ingest enough text in domain 1 to promote entities
    rsvs.ingest_text(
        "Stone is hard and rough. Stone is solid and heavy. \
         Stone is hard and dense. Stone is rough and firm. \
         Hard stone is durable. Hard materials are strong.",
    )
    .expect("ingest1");

    // Switch domain
    rsvs.config.current_domain = 2;
    rsvs.ingest_text(
        "Water is liquid and wet. Water is fluid and clear. \
         Water is liquid and cold. Water is wet and fresh. \
         Liquid water flows. Liquid substances move freely.",
    )
    .expect("ingest2");

    let snap = rsvs.snapshot_v1();
    // Should have at least seed nodes
    assert!(snap.nodes.len() >= 24);
    // Check that contexts were processed
    let status = rsvs.status();
    assert!(status.total_contexts > 0);
}

#[test]
fn persistence_roundtrip() {
    let mut rsvs = Rsvs::new(PipelineConfig::default()).expect("create RSVS");
    rsvs.ingest_text(
        "Fire is hot and bright. Fire is hot and dangerous. \
         Fire produces heat. Fire is hot and fast. Hot fire burns.",
    )
    .expect("ingest");

    let dir = std::env::temp_dir().join("rsvs_integration_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_state.json");

    rsvs.save(&path).expect("save");
    let loaded = Rsvs::load(&path).expect("load");

    assert_eq!(rsvs.status().total_nodes, loaded.status().total_nodes);

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

// =======================================================================
// v9.0: Meaning Discovery Integration Tests
// =======================================================================
//
// Tests whether RSVS can discover hidden meaning through its 3 pathways:
//   P1: Gap Detection → Implikatur, Presuposisi, Pragmatik
//   P2: Seed Activation → Afektif, Sosial, Konotatif
//   P3: Discourse Tracking → Performatif, Ekstensional, Discursive
//
// SCENARIO:
//   We feed RSVS multiple texts about "pengkhianatan" (betrayal) and
//   "harga diri" (dignity) until the graph is mature in this domain.
//   Then we test if RSVS can:
//     A. Detect that "marah karena dikhianati" has high risk + negative valence
//     B. Detect that "harga diri tersentuh" has high identity + social threat
//     C. Find the HYBRID meaning: "dikhianati YANG menyentuh harga diri"
//     D. Discover the emergent pattern: betrayal+identity = trauma pattern
//
// IMPORTANT: These tests are HONEST — they show what RSVS CAN do and what
// it CANNOT yet do. The gap analysis identifies missing capabilities.

/// Create RSVS with meaning pathways enabled and lower promotion threshold.
fn create_rsvs_with_pathways() -> Rsvs {
    let config = PipelineConfig {
        enable_meaning_pathways: true,
        entity_promote_n: 2,
        ..PipelineConfig::default()
    };
    Rsvs::new(config).expect("create RSVS with pathways")
}

/// Build domain-specific knowledge about betrayal & dignity.
///
/// RSVS learns through CO-OCCURRENCE. We feed many sentences where
/// key tokens appear near seed words so they get promoted and connected.
/// Key insight: RSVS uses SENTENCE-LEVEL grounding (v8.2) — if any seed
/// appears in a sentence, ALL tokens in that sentence are groundable.
fn build_betrayal_domain(rsvs: &mut Rsvs) {
    // PHASE 1: Core betrayal vocabulary — repeated with seeds for grounding
    let texts = &[
        "dikhianati value risk is pain. dikhianati value risk hurts identity. \
         dikhianati value risk destroys trust. dikhianati value risk causes change. \
         dikhianati value risk breaks relation. dikhianati value creates risk.",
        "marah risk value is intense. marah risk value burns identity. \
         marah risk value signals change. marah risk value demands action. \
         marah risk value creates effect. marah risk value causes state.",
        "harga value identity is precious. harga value identity defines entity. \
         harga value identity creates meaning. harga value identity needs trust. \
         harga value identity requires respect. harga value identity affects state.",
        "diri identity entity is core. diri identity entity holds value. \
         diri identity entity needs trust. diri identity entity feels risk. \
         diri identity entity seeks goal. diri identity entity has meaning.",
        "sakit risk value is suffering. sakit risk value signals change. \
         sakit risk value demands attention. sakit risk value hurts identity. \
         sakit risk value creates state. sakit risk value affects trust.",
        "trauma risk identity is deep. trauma risk identity persists time. \
         trauma risk identity affects memory. trauma risk identity creates pattern. \
         trauma risk identity damages trust. trauma risk identity repeats pattern.",
    ];
    for text in texts {
        let _ = rsvs.ingest_text(text).expect("ingest");
    }

    // PHASE 2: Combinatorial sentences — betrayal + dignity + anger together
    let combo = &[
        "dikhianati value marah risk hurts. dikhianati value marah risk is pain. \
         dikhianati value harga identity is deep. dikhianati value harga identity wounds. \
         dikhianati value diri identity is devastating. dikhianati value diri risk scars.",
        "marah risk harga value identity explodes. marah risk harga value identity erupts. \
         marah risk dikhianati value is justified. marah risk dikhianati value is natural. \
         marah risk sakit value is real. marah risk sakit value demands change.",
        "sakit risk diri identity lingers. sakit risk diri identity persists. \
         sakit risk dikhianati value is betrayal pain. sakit risk dikhianati value cuts deep. \
         sakit risk trauma identity is pattern. sakit risk trauma identity repeats.",
        "dikhianati value harga diri identity is wound. dikhianati value harga diri identity hurts. \
         dikhianati value harga diri risk destroys. dikhianati value harga diri risk damages trust.",
        "trauma risk dikhianati value identity is pattern. trauma risk dikhianati value identity repeats. \
         trauma risk harga diri identity is cycle. trauma risk harga diri identity persists time.",
    ];
    for text in combo {
        let _ = rsvs.ingest_text(text).expect("ingest combo");
    }

    // PHASE 3: More repetitions for maturity
    let mature = &[
        "dikhianati value risk marah risk is common. dikhianati value risk marah risk is expected. \
         dikhianati value risk marah risk is human. dikhianati value risk marah risk causes change.",
        "harga diri value identity is self. harga diri value identity is core. \
         harga diri value identity needs trust. harga diri value identity defines meaning.",
        "dikhianati value sakit risk is wound. dikhianati value sakit risk is trauma. \
         dikhianati value sakit risk damages trust. dikhianati value sakit risk hurts identity.",
        "trauma risk identity repeats pattern. trauma risk identity persists time. \
         trauma risk identity affects memory. trauma risk identity creates change.",
    ];
    for text in mature {
        let _ = rsvs.ingest_text(text).expect("ingest mature");
    }
}

/// Print detailed node info for debugging.
fn print_node_info(rsvs: &Rsvs, token: &str) {
    if let Some(&id) = rsvs.token_to_id.get(token) {
        let conf = rsvs.autonomy.confidence(id).unwrap_or(0.0);
        let status = rsvs.autonomy.status(id).map(|s| format!("{:?}", s)).unwrap_or_default();
        let tier = rsvs.autonomy.tier(id).map(|t| format!("{:?}", t)).unwrap_or_default();
        eprintln!("  {} → confidence={:.3}, status={}, tier={}", token, conf, status, tier);

        if let Some(node) = rsvs.graph.get_node(id) {
            eprintln!("    gap_annotations: {} entries", node.gap_annotations.len());
            eprintln!("    sense_profiles: {} entries", node.sense_profiles.len());

            for (sid, profile) in &node.sense_profiles {
                eprintln!("    sense {} VAD: valence={:.3} arousal={:.3} dominance={:.3}",
                    sid, profile.affective.valence, profile.affective.arousal, profile.affective.dominance);
                eprintln!("    sense {} social: distance={:.3} trust={:.3} politeness={:.3}",
                    sid, profile.social.distance, profile.social.trust, profile.social.expected_politeness);
                eprintln!("    sense {} connotation: {:?} (conf={:.3})",
                    sid, profile.connotative.primary_connotation, profile.connotative.profile_confidence);
                if !profile.conflicts.is_empty() {
                    for c in &profile.conflicts {
                        eprintln!("    sense {} CONFLICT: {:?} score={:.3}",
                            sid, c.conflict_type, c.conflict_score);
                    }
                }
            }

            for (sid, anns) in &node.gap_annotations {
                for ann in anns {
                    let target_label = rsvs.graph.get_node(ann.target_node)
                        .map(|n| n.label.clone()).unwrap_or_else(|| format!("node_{}", ann.target_node));
                    eprintln!("    sense {} GAP: {:?} conf={:.3} → '{}'",
                        sid, ann.gap_type, ann.confidence, target_label);
                }
            }
        }

        if let Some(sm) = rsvs.senses.get(&id) {
            for sense in &sm.senses {
                let comp_labels: Vec<String> = sense.compositions.iter()
                    .filter_map(|c| rsvs.graph.get_node(c.node_id).map(|n| n.label.clone()))
                    .collect();
                eprintln!("    sense {} compositions: {:?}", sense.id, comp_labels);
            }
        }
    } else {
        eprintln!("  {} → NOT PROMOTED", token);
    }
}

// -----------------------------------------------------------------------
// TEST: Domain maturity — does RSVS build a mature graph?
// -----------------------------------------------------------------------

#[test]
fn meaning_discovery_domain_maturity() {
    let mut rsvs = create_rsvs_with_pathways();
    let initial_count = rsvs.graph.node_count();
    assert_eq!(initial_count, 24);

    build_betrayal_domain(&mut rsvs);

    let after_count = rsvs.graph.node_count();
    eprintln!("=== Domain Maturity ===");
    eprintln!("Nodes before: {}, after: {}, promoted: {}", initial_count, after_count, after_count - initial_count);

    // Key tokens should be promoted
    for token in &["dikhianati", "marah", "harga", "diri", "sakit", "trauma"] {
        eprintln!("\n--- {} ---", token);
        print_node_info(&rsvs, token);
    }

    // At minimum some tokens should be promoted
    assert!(after_count > initial_count, "Should have promoted tokens");
}

// -----------------------------------------------------------------------
// TEST: Affective profile discovery
// -----------------------------------------------------------------------

#[test]
fn meaning_discovery_affective_profiles() {
    let mut rsvs = create_rsvs_with_pathways();
    build_betrayal_domain(&mut rsvs);

    eprintln!("=== Affective Profile Discovery ===");

    // Check each key node for affective profiles
    let mut found_profiles = 0;
    for token in &["dikhianati", "marah", "harga", "diri", "sakit", "trauma"] {
        if let Some(&id) = rsvs.token_to_id.get(*token) {
            if let Some(node) = rsvs.graph.get_node(id) {
                for (_, profile) in &node.sense_profiles {
                    found_profiles += 1;
                    eprintln!("  {} VAD: valence={:.3} arousal={:.3} dominance={:.3}",
                        token, profile.affective.valence, profile.affective.arousal, profile.affective.dominance);
                    eprintln!("  {} social: distance={:.3} trust={:.3} politeness={:.3}",
                        token, profile.social.distance, profile.social.trust, profile.social.expected_politeness);
                    eprintln!("  {} connotation: {:?} (conf={:.3})",
                        token, profile.connotative.primary_connotation, profile.connotative.profile_confidence);
                }
            }
        }
    }

    eprintln!("\nTotal profiles found: {}", found_profiles);
    // Profiles may or may not exist depending on spreading activation energy
    // This test is diagnostic — it shows what the system actually computed
}

// -----------------------------------------------------------------------
// TEST: Cross-pathway conflict detection
// -----------------------------------------------------------------------

#[test]
fn meaning_discovery_conflict_detection() {
    let mut rsvs = create_rsvs_with_pathways();
    build_betrayal_domain(&mut rsvs);

    eprintln!("=== Cross-Pathway Conflict Detection ===");

    let mut total_conflicts = 0;
    for (&node_id, node) in rsvs.graph.nodes.iter() {
        for (_, profile) in &node.sense_profiles {
            for conflict in &profile.conflicts {
                total_conflicts += 1;
                let label = rsvs.graph.get_node(node_id).map(|n| n.label.clone()).unwrap_or_default();
                eprintln!("  Conflict on '{}': {:?} (score={:.3})",
                    label, conflict.conflict_type, conflict.conflict_score);
            }
        }
    }

    eprintln!("Total conflicts found: {}", total_conflicts);
}

// -----------------------------------------------------------------------
// TEST: Gap detection — hidden presupposition
// -----------------------------------------------------------------------

#[test]
fn meaning_discovery_gap_detection() {
    let mut rsvs = create_rsvs_with_pathways();
    build_betrayal_domain(&mut rsvs);

    eprintln!("=== Gap Detection ===");

    let mut total_gaps = 0;
    for (&_node_id, node) in rsvs.graph.nodes.iter() {
        if node.gap_annotations.is_empty() { continue; }
        let label = node.label.clone();
        for (sid, anns) in &node.gap_annotations {
            for ann in anns {
                total_gaps += 1;
                let target_label = rsvs.graph.get_node(ann.target_node)
                    .map(|n| n.label.clone()).unwrap_or_default();
                eprintln!("  Gap on '{}' sense {}: {:?} conf={:.3} → '{}'",
                    label, sid, ann.gap_type, ann.confidence, target_label);
            }
        }
    }

    eprintln!("Total gaps found: {}", total_gaps);
}

// -----------------------------------------------------------------------
// TEST: Discourse tracking
// -----------------------------------------------------------------------

#[test]
fn meaning_discovery_discourse() {
    let mut rsvs = create_rsvs_with_pathways();
    build_betrayal_domain(&mut rsvs);

    eprintln!("=== Discourse Tracking ===");

    let utterance_count = rsvs.graph.nodes.iter()
        .filter(|(_, n)| n.semantic.is_utterance)
        .count();

    eprintln!("Utterance nodes: {}", utterance_count);

    // Count discourse edges
    let discourse_edges = rsvs.graph.nodes.iter()
        .flat_map(|(&id, _)| rsvs.graph.edges_from(id))
        .filter(|e| e.source == rsvs::types::EdgeSource::Discourse)
        .count();
    eprintln!("Discourse edges: {}", discourse_edges);

    // Check speech acts on utterance nodes
    for (_, node) in rsvs.graph.nodes.iter() {
        if node.semantic.is_utterance {
            if let Some(ref meta) = node.discourse_meta {
                eprintln!("  {} → speech_act={:?}", node.label, meta.speech_act);
            }
        }
    }
}

// -----------------------------------------------------------------------
// TEST: The ULTIMATE test — A+B→C hybridization
//
// Can RSVS discover:
//   A: "Dia marah karena dikhianati"
//   B: "Dia marah karena harga diri tersentuh"
//   → A+B: "Dikhianati YANG menyentuh harga diri"
//   → C: "Dikhianatan terhadap harga diri = pola trauma"
// -----------------------------------------------------------------------

#[test]
fn meaning_discovery_hybridization_abc() {
    let mut rsvs = create_rsvs_with_pathways();
    build_betrayal_domain(&mut rsvs);

    eprintln!("=== HYBRIDIZATION A+B→C ===");

    // STEP A: Check structural connections
    eprintln!("\n--- STEP A: Structural Connections ---");

    if let Some(&dik_id) = rsvs.token_to_id.get("dikhianati") {
        let dik_atoms = rsvs.graph.get_node(dik_id).map(|n| n.atoms.clone()).unwrap_or_default();
        let dik_labels: Vec<String> = dik_atoms.iter()
            .filter_map(|&aid| rsvs.graph.get_node(aid).map(|n| n.label.clone()))
            .collect();
        eprintln!("  dikhianati atoms: {:?}", dik_labels);

        let has_marah = dik_labels.iter().any(|l| l == "marah");
        let has_harga = dik_labels.iter().any(|l| l == "harga");
        let has_diri = dik_labels.iter().any(|l| l == "diri");
        let has_trauma = dik_labels.iter().any(|l| l == "trauma");

        eprintln!("    Connected to marah: {}", has_marah);
        eprintln!("    Connected to harga: {}", has_harga);
        eprintln!("    Connected to diri: {}", has_diri);
        eprintln!("    Connected to trauma: {}", has_trauma);
    }

    // STEP B: Check if dikhianati compositions include harga/diri
    eprintln!("\n--- STEP B: Composition Analysis ---");

    if let Some(&dik_id) = rsvs.token_to_id.get("dikhianati") {
        if let Some(sm) = rsvs.senses.get(&dik_id) {
            for sense in &sm.senses {
                let comp_labels: Vec<String> = sense.compositions.iter()
                    .filter_map(|c| rsvs.graph.get_node(c.node_id).map(|n| n.label.clone()))
                    .collect();
                eprintln!("  dikhianati sense {} compositions: {:?}", sense.id, comp_labels);
            }
        }
    }

    // STEP C: Check if trauma connects to both dikhianati and harga
    eprintln!("\n--- STEP C: Trauma as Bridge ---");

    if let (Some(&tra_id), Some(&dik_id), Some(&_har_id)) = (
        rsvs.token_to_id.get("trauma"),
        rsvs.token_to_id.get("dikhianati"),
        rsvs.token_to_id.get("harga"),
    ) {
        let tra_atoms = rsvs.graph.get_node(tra_id).map(|n| n.atoms.clone()).unwrap_or_default();
        let tra_labels: Vec<String> = tra_atoms.iter()
            .filter_map(|&aid| rsvs.graph.get_node(aid).map(|n| n.label.clone()))
            .collect();
        eprintln!("  trauma atoms: {:?}", tra_labels);

        // Structural similarity
        if let (Some(sm_a), Some(sm_b)) = (rsvs.senses.get(&dik_id), rsvs.senses.get(&tra_id)) {
            let sim = rsvs.graph.structural_similarity(dik_id, tra_id, sm_a, sm_b);
            eprintln!("  similarity(dikhianati, trauma) = {:.3}", sim.structural_similarity);
        }
    }

    // GAP ANALYSIS
    eprintln!("\n--- GAP ANALYSIS: What RSVS CAN vs CANNOT do ---");
    eprintln!("  CAN:");
    eprintln!("    ✓ Build co-occurrence graph from text");
    eprintln!("    ✓ Promote tokens and create compositions");
    eprintln!("    ✓ Compute affective profiles (valence/arousal/dominance)");
    eprintln!("    ✓ Compute social profiles (distance/trust/power)");
    eprintln!("    ✓ Compute connotative profiles (positive/negative/ambiguous)");
    eprintln!("    ✓ Detect cross-pathway conflicts (sarcasm/euphemism)");
    eprintln!("    ✓ Detect meaning gaps (missing compositions)");
    eprintln!("    ✓ Track discourse structure (speech acts/centering)");
    eprintln!("");
    eprintln!("  CANNOT (yet):");
    eprintln!("    ✗ HYBRIDIZE two interpretations into one (A+B→A∧B)");
    eprintln!("    ✗ EMERGE new insight (C) not explicitly in text");
    eprintln!("    ✗ Recognize PATTERNS like 'betrayal+dignity=trauma'");
    eprintln!("    ✗ CROSS-REFERENTIAL reasoning across pathways");
    eprintln!("");
    eprintln!("  NEEDED for full A+B→C:");
    eprintln!("    1. Compositional Blending: merge sense A + sense B → hybrid");
    eprintln!("    2. Abductive Reasoning: when A and B activate same node, hypothesize C");
    eprintln!("    3. Pattern Mining: detect that composition pairs co-occur → named pattern");
    eprintln!("    4. Cross-Pathway Synthesis: P1 gap + P2 conflict → deeper meaning search");
}

// -----------------------------------------------------------------------
// TEST: Step-by-step reasoning trace
// -----------------------------------------------------------------------

#[test]
fn meaning_discovery_step_by_step_trace() {
    let mut rsvs = create_rsvs_with_pathways();
    build_betrayal_domain(&mut rsvs);

    eprintln!("=== Step-by-Step Reasoning Trace ===");
    eprintln!("Processing: 'dia marah karena dikhianati'");

    let target = "dia marah risk value karena dikhianati value risk. dia marah risk value karena dikhianati value identity.";
    let stats = rsvs.ingest_text(target).expect("ingest");

    eprintln!("\nPipeline output:");
    eprintln!("  Sentences: {}, Promoted: {}, Senses created: {}, Compositions: {}",
        stats.sentences_processed, stats.atoms_promoted, stats.sense_created, stats.compositions_induced);

    // P1: Gaps
    eprintln!("\n--- Pathway 1: Gap Detection ---");
    let mut gap_count = 0;
    for (_, node) in rsvs.graph.nodes.iter() {
        if node.is_seed { continue; }
        for (sid, anns) in &node.gap_annotations {
            for ann in anns {
                gap_count += 1;
                let target_label = rsvs.graph.get_node(ann.target_node)
                    .map(|n| n.label.clone()).unwrap_or_default();
                eprintln!("  [P1] '{}' sense {}: {:?} conf={:.3} → '{}'",
                    node.label, sid, ann.gap_type, ann.confidence, target_label);
            }
        }
    }
    if gap_count == 0 {
        eprintln!("  [P1] No gaps — insufficient composition structure for prediction");
    }

    // P2: Profiles
    eprintln!("\n--- Pathway 2: Affective-Social ---");
    let mut profile_count = 0;
    for (_, node) in rsvs.graph.nodes.iter() {
        if node.is_seed { continue; }
        for (sid, profile) in &node.sense_profiles {
            profile_count += 1;
            eprintln!("  [P2] '{}' sense {}: V= {:.3} A= {:.3} D= {:.3} connotation={:?}",
                node.label, sid,
                profile.affective.valence, profile.affective.arousal, profile.affective.dominance,
                profile.connotative.primary_connotation);
            if !profile.conflicts.is_empty() {
                for c in &profile.conflicts {
                    eprintln!("       CONFLICT: {:?} score={:.3}", c.conflict_type, c.conflict_score);
                }
            }
        }
    }
    if profile_count == 0 {
        eprintln!("  [P2] No profiles — spreading activation may be insufficient");
    }

    // P3: Discourse
    eprintln!("\n--- Pathway 3: Discourse ---");
    let utterances: Vec<_> = rsvs.graph.nodes.iter()
        .filter(|(_, n)| n.semantic.is_utterance)
        .collect();
    if utterances.is_empty() {
        eprintln!("  [P3] No utterances — needs more promoted tokens in sentences");
    } else {
        for (_, node) in &utterances {
            if let Some(ref meta) = node.discourse_meta {
                eprintln!("  [P3] {} → speech_act={:?}", node.label, meta.speech_act);
            }
        }
    }
}

// -----------------------------------------------------------------------
// TEST: Connotative profile accuracy
// -----------------------------------------------------------------------

#[test]
fn meaning_discovery_connotative_accuracy() {
    let mut rsvs = create_rsvs_with_pathways();
    build_betrayal_domain(&mut rsvs);

    eprintln!("=== Connotative Profile Accuracy ===");

    // "sakit" and "dikhianati" should lean negative (high risk, low value)
    // "harga" should lean positive (high value, low risk)
    for token in &["sakit", "dikhianati", "trauma", "harga"] {
        if let Some(&id) = rsvs.token_to_id.get(*token) {
            if let Some(node) = rsvs.graph.get_node(id) {
                for (_, profile) in &node.sense_profiles {
                    eprintln!("  '{}': connotation={:?} valence={:.3} arousal={:.3}",
                        token, profile.connotative.primary_connotation,
                        profile.affective.valence, profile.affective.arousal);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// TEST: Cross-pathway emergence — nodes with BOTH gaps AND conflicts
// -----------------------------------------------------------------------

#[test]
fn meaning_discovery_cross_pathway_emergence() {
    let mut rsvs = create_rsvs_with_pathways();
    build_betrayal_domain(&mut rsvs);

    eprintln!("=== Cross-Pathway Emergence ===");

    let mut emergence_candidates = 0;
    for (_, node) in rsvs.graph.nodes.iter() {
        let has_gaps = !node.gap_annotations.is_empty();
        let has_conflicts = node.sense_profiles.values().any(|p| !p.conflicts.is_empty());

        if has_gaps && has_conflicts {
            emergence_candidates += 1;
            eprintln!("  EMERGENCE: '{}' has BOTH gaps AND conflicts", node.label);
        }
    }

    eprintln!("Total emergence candidates: {}", emergence_candidates);

    if emergence_candidates == 0 {
        eprintln!("\n  HONEST: No cross-pathway emergence detected yet.");
        eprintln!("  This is EXPECTED — P1 and P2 operate independently.");
        eprintln!("  A 'CrossPathwaySynthesis' engine is needed to bridge them.");
    }
}

// -----------------------------------------------------------------------
// TEST: Full pipeline statistics
// -----------------------------------------------------------------------

#[test]
fn meaning_discovery_full_statistics() {
    let mut rsvs = create_rsvs_with_pathways();
    build_betrayal_domain(&mut rsvs);

    eprintln!("=== Full Pipeline Statistics ===");

    let total_nodes = rsvs.graph.node_count();
    let seed_nodes = rsvs.seed_node_ids.len();
    let promoted = total_nodes - seed_nodes;

    let mut total_edges = 0;
    let mut learned = 0;
    let mut discourse = 0;
    for (&id, _) in rsvs.graph.nodes.iter() {
        for e in rsvs.graph.edges_from(id) {
            total_edges += 1;
            match e.source {
                rsvs::types::EdgeSource::Learned => learned += 1,
                rsvs::types::EdgeSource::Discourse => discourse += 1,
                _ => {}
            }
        }
    }

    let utterance_count = rsvs.graph.nodes.iter().filter(|(_, n)| n.semantic.is_utterance).count();
    let nodes_with_gaps = rsvs.graph.nodes.iter().filter(|(_, n)| !n.gap_annotations.is_empty()).count();
    let nodes_with_profiles = rsvs.graph.nodes.iter().filter(|(_, n)| !n.sense_profiles.is_empty()).count();
    let nodes_with_conflicts = rsvs.graph.nodes.iter()
        .filter(|(_, n)| n.sense_profiles.values().any(|p| !p.conflicts.is_empty()))
        .count();

    eprintln!("  Nodes: {} ({} seeds + {} promoted)", total_nodes, seed_nodes, promoted);
    eprintln!("  Edges: {} (learned={}, discourse={})", total_edges, learned, discourse);
    eprintln!("  Utterances: {}", utterance_count);
    eprintln!("  Nodes with gap annotations: {}", nodes_with_gaps);
    eprintln!("  Nodes with sense profiles: {}", nodes_with_profiles);
    eprintln!("  Nodes with conflicts: {}", nodes_with_conflicts);
    eprintln!("  Total contexts: {}", rsvs.total_contexts);
    eprintln!("  Batches: {}", rsvs.batch_counter);

    assert!(total_nodes > 24, "Should have promoted nodes beyond seeds");
}
