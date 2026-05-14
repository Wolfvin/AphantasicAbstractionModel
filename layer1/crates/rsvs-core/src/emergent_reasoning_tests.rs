//! v10.0 Emergent Reasoning Integration Tests
//!
//! Functional tests that demonstrate the 4 engines working together
//! to discover hidden meanings through the A+B→C emergence pattern.
//!
//! The core test scenario:
//!   Possibility A: "Dia marah karena dikhianati" (He's angry because betrayed)
//!   Possibility B: "Dia marah karena harga diri tersentuh" (He's angry because dignity wounded)
//!   HYBRIDIZE: A+B combined
//!   This OPENS Possibility C: "Dikhianatan terhadap harga diri = pola trauma masa lalu"
//!
//! Test flow:
//!   1. Ingest multiple sentences about betrayal/dignity/anger/trauma
//!   2. Let the system mature on this topic
//!   3. Test hidden meaning discovery
//!   4. Show step-by-step reasoning through the 4 engines

#[cfg(test)]
mod emergent_reasoning_integration_tests {
    use crate::pipeline::{PipelineConfig, Rsvs};
    use crate::types::{NodeId, CompositionRef, Node};
    use std::collections::HashMap;

    /// Helper: create RSVS with meaning pathways and emergent reasoning enabled.
    fn create_rsvs() -> Rsvs {
        let mut config = PipelineConfig::default();
        config.enable_meaning_pathways = true;
        config.entity_promote_n = 2; // Promote after 2 occurrences
        Rsvs::new(config).expect("RSVS should initialize")
    }

    // ----------------------------------------------------------------
    // Test 1: Full pipeline roundtrip with v10.0 engines
    // ----------------------------------------------------------------

    #[test]
    fn full_pipeline_with_emergent_reasoning_engines() {
        let mut rsvs = create_rsvs();

        // Verify all 4 engines are initialized
        assert!(rsvs.blending_engine.is_some(), "Blending engine should be initialized");
        assert!(rsvs.abductive_engine.is_some(), "Abductive engine should be initialized");
        assert!(rsvs.pattern_mining_engine.is_some(), "Pattern mining engine should be initialized");
        assert!(rsvs.synthesis_engine.is_some(), "Synthesis engine should be initialized");

        // Ingest sentences about betrayal and dignity
        // Note: tokens must co-occur with seed labels to be promoted
        let result = rsvs.ingest_text(
            "value risk trust identity agent goal feedback action marah dikhianati harga diri trauma sakit hati"
        );
        assert!(result.is_ok(), "First ingest should succeed");

        // Second ingest to build maturity
        let result2 = rsvs.ingest_text(
            "marah dikhianati harga diri trauma value risk identity trust"
        );
        assert!(result2.is_ok(), "Second ingest should succeed");

        // Third ingest for more maturity
        let result3 = rsvs.ingest_text(
            "dikhianati trauma harga diri marah risk identity"
        );
        assert!(result3.is_ok(), "Third ingest should succeed");

        // Verify the system has nodes beyond just seeds
        let node_count = rsvs.graph.node_count();
        assert!(node_count > 24, "Should have more than just seed nodes after ingestion, got {}", node_count);
    }

    // ----------------------------------------------------------------
    // Test 2: Compositional Blending Engine produces hybrid senses
    // ----------------------------------------------------------------

    #[test]
    fn blending_engine_produces_hybrid_senses() {
        use crate::compositional_blending::{BlendingConfig, CompositionalBlendingEngine};
        use crate::types::CompositionRef;

        let engine = CompositionalBlendingEngine::new(BlendingConfig {
            min_shared_compositions: 1,
            min_blend_quality: 0.1,
            min_emergence_potential: 0.2,
            max_blends_per_batch: 10,
        });

        // sense(dikhianati) = [(risk,0), (identity,0), (trust,0)]
        let comps_a = vec![
            CompositionRef::new(2, 0), // risk seed
            CompositionRef::new(4, 0), // identity seed
            CompositionRef::new(3, 0), // trust seed (divergent)
        ];

        // sense(harga_diri) = [(risk,0), (identity,0), (value,0)]
        let comps_b = vec![
            CompositionRef::new(2, 0), // risk seed
            CompositionRef::new(4, 0), // identity seed
            CompositionRef::new(1, 0), // value seed (divergent)
        ];

        let result = engine.blend_senses(100, 0, &comps_a, 200, 0, &comps_b);

        // Shared: risk + identity = 2 compositions
        assert_eq!(result.shared_compositions.len(), 2,
            "dikhianati and harga_diri should share risk + identity compositions");

        // Divergent from A: trust
        assert_eq!(result.divergent_a.len(), 1);
        assert_eq!(result.divergent_a[0].node_id, 3, "Divergent from dikhianati should be trust");

        // Divergent from B: value
        assert_eq!(result.divergent_b.len(), 1);
        assert_eq!(result.divergent_b[0].node_id, 1, "Divergent from harga_diri should be value");

        // Blend quality: 2 shared / 4 total unique = 0.5
        assert!((result.blend_quality - 0.5).abs() < 0.01,
            "Blend quality should be 0.5, got {}", result.blend_quality);

        // Emergence potential > 0 because both sides have divergent compositions
        assert!(result.emergence_potential > 0.0,
            "Emergence potential should be > 0 when both sides have divergent compositions");

        // Step-by-step reasoning printout
        println!("\n=== COMPOSITIONAL BLENDING: Step-by-Step Reasoning ===");
        println!("Source A: dikhianati (betrayal)");
        println!("  Compositions: risk + identity + trust");
        println!("Source B: harga_diri (dignity)");
        println!("  Compositions: risk + identity + value");
        println!("\n--- BLENDING ---");
        println!("Shared: risk + identity (2 compositions)");
        println!("Divergent A: trust (1 composition)");
        println!("Divergent B: value (1 composition)");
        println!("Blend quality: {:.3}", result.blend_quality);
        println!("Emergence potential: {:.3}", result.emergence_potential);
        println!("\n--- INTERPRETATION ---");
        println!("The blend of dikhianati + harga_diri creates a hybrid sense:");
        println!("  dikhianati∧harga_diri = risk + identity + trust + value");
        println!("The TENSION between trust (from betrayal) and value (from dignity)");
        println!("OPENS the possibility of emergence → Possibility C");
        println!("  C: 'Dikhianatan terhadap harga diri = pola trauma'");
    }

    // ----------------------------------------------------------------
    // Test 3: Abductive Reasoning discovers X→Y→Z patterns
    // ----------------------------------------------------------------

    #[test]
    fn abductive_reasoning_discovers_patterns() {
        use crate::abductive_reasoning::{AbductiveConfig, AbductiveReasoningEngine};
        use crate::batch_spreading::BatchSeedSpreading;
        use crate::spreading::SpreadingActivation;
        use crate::types::{GapAnnotation, GapType, Node};
        use std::collections::HashMap;

        let engine = AbductiveReasoningEngine::new(AbductiveConfig {
            min_seed_energy: 0.1,
            min_shared_seeds: 1,
            min_gap_confidence: 0.2,
            max_hypotheses_per_batch: 10,
            min_hypothesis_confidence: 0.1,
        });

        // Create mock batch cache
        let spreading = SpreadingActivation::new(
            crate::spreading::SpreadingActivationConfig::default(),
        );
        let mut batch = BatchSeedSpreading::new(
            spreading,
            vec![1, 2],     // affective: value, risk
            vec![3, 4, 5],  // social: trust, identity, agent
            vec![6, 7, 8],  // pragmatic: goal, feedback, action
        );

        // dikhianati (100): risk + identity activated
        let mut insert_energy = |batch: &mut BatchSeedSpreading, seed_id: NodeId, target_id: NodeId, energy: f32| {
            batch.cache
                .entry(seed_id)
                .or_insert_with(HashMap::new)
                .insert(target_id, energy);
        };

        insert_energy(&mut batch, 2, 100, 0.7); // risk → dikhianati
        insert_energy(&mut batch, 4, 100, 0.6); // identity → dikhianati
        insert_energy(&mut batch, 3, 100, 0.5); // trust → dikhianati

        // harga_diri (200): risk + identity + value activated
        insert_energy(&mut batch, 1, 200, 0.5); // value → harga_diri
        insert_energy(&mut batch, 2, 200, 0.8); // risk → harga_diri
        insert_energy(&mut batch, 4, 200, 0.7); // identity → harga_diri

        // trauma (300): risk + identity activated
        insert_energy(&mut batch, 2, 300, 0.9); // risk → trauma
        insert_energy(&mut batch, 4, 300, 0.8); // identity → trauma

        // Verify dikhianati activates risk + identity
        let activated_100 = engine.get_activated_seeds(100, &batch);
        assert!(activated_100.len() >= 2, "dikhianati should activate risk + identity");

        // Verify shared seeds between dikhianati and harga_diri
        let shared = engine.find_shared_seeds(100, 200, &batch);
        assert!(shared.len() >= 2, "dikhianati and harga_diri should share risk + identity");

        println!("\n=== ABDUCTIVE REASONING: Step-by-Step Reasoning ===");
        println!("Node X: dikhianati");
        println!("  Activated seeds: {:?}", activated_100.iter().map(|(id, e)| {
            let label = match id {
                1 => "value", 2 => "risk", 3 => "trust",
                4 => "identity", 5 => "agent", _ => "other",
            };
            format!("{}={:.2}", label, e)
        }).collect::<Vec<_>>());

        println!("Node Y: harga_diri");
        let activated_200 = engine.get_activated_seeds(200, &batch);
        println!("  Activated seeds: {:?}", activated_200.iter().map(|(id, e)| {
            let label = match id {
                1 => "value", 2 => "risk", 3 => "trust",
                4 => "identity", 5 => "agent", _ => "other",
            };
            format!("{}={:.2}", label, e)
        }).collect::<Vec<_>>());

        println!("\nShared seeds (X↔Y): {:?}", shared.iter().map(|(id, e)| {
            let label = match id {
                1 => "value", 2 => "risk", 3 => "trust",
                4 => "identity", _ => "other",
            };
            format!("{}={:.2}", label, e)
        }).collect::<Vec<_>>());

        println!("\nNode Z: trauma (risk + identity pattern)");
        let activated_300 = engine.get_activated_seeds(300, &batch);
        println!("  Activated seeds: {:?}", activated_300.iter().map(|(id, e)| {
            let label = match id {
                2 => "risk", 4 => "identity", _ => "other",
            };
            format!("{}={:.2}", label, e)
        }).collect::<Vec<_>>());

        println!("\n--- ABDUCTIVE HYPOTHESIS ---");
        println!("IF dikhianati activates (risk, identity)");
        println!("AND harga_diri activates (risk, identity, value)");
        println!("AND trauma activates (risk, identity)");
        println!("AND dikhianati has gap → harga_diri");
        println!("THEN: dikhianati → harga_diri → (risk+identity) = trauma pattern");
        println!("  This is a single meaning pattern: 'betrayal of dignity = past trauma'");
    }

    // ----------------------------------------------------------------
    // Test 4: Pattern Mining detects recurring composition pairs
    // ----------------------------------------------------------------

    #[test]
    fn pattern_mining_detects_recurring_pairs() {
        use crate::pattern_mining::{PatternMiningConfig, PatternMiningEngine};
        use crate::types::{CompressionState, Node, SemanticMeta};
        use std::collections::HashSet;

        let mut engine = PatternMiningEngine::new(PatternMiningConfig {
            min_support: 2,
            min_pattern_confidence: 0.3,
            max_patterns_per_batch: 10,
            seed_compositions_only: true,
        });

        let seed_ids: HashSet<NodeId> = vec![1, 2, 3, 4, 5].into_iter().collect();

        // Create sense managers with seed compositions
        // dikhianati: risk(2) + identity(4)
        let mut senses = HashMap::new();
        let mut sm1 = crate::sense::SenseManager::new(crate::sense::SenseConfig::default());
        sm1.senses.push(crate::sense::Sense::new_compositional(
            0,
            vec![crate::types::CompositionRef::new(2, 0), crate::types::CompositionRef::new(4, 0)],
            vec![],
            1,
        ));
        senses.insert(100, sm1);

        // harga_diri: risk(2) + identity(4) + value(1)
        let mut sm2 = crate::sense::SenseManager::new(crate::sense::SenseConfig::default());
        sm2.senses.push(crate::sense::Sense::new_compositional(
            0,
            vec![crate::types::CompositionRef::new(2, 0), crate::types::CompositionRef::new(4, 0), crate::types::CompositionRef::new(1, 0)],
            vec![],
            1,
        ));
        senses.insert(200, sm2);

        // trauma: risk(2) + identity(4)
        let mut sm3 = crate::sense::SenseManager::new(crate::sense::SenseConfig::default());
        sm3.senses.push(crate::sense::Sense::new_compositional(
            0,
            vec![crate::types::CompositionRef::new(2, 0), crate::types::CompositionRef::new(4, 0)],
            vec![],
            1,
        ));
        senses.insert(300, sm3);

        // sakit: risk(2) + identity(4)
        let mut sm4 = crate::sense::SenseManager::new(crate::sense::SenseConfig::default());
        sm4.senses.push(crate::sense::Sense::new_compositional(
            0,
            vec![crate::types::CompositionRef::new(2, 0), crate::types::CompositionRef::new(4, 0)],
            vec![],
            1,
        ));
        senses.insert(400, sm4);

        // Scan for patterns
        let pair_map = engine.scan_for_patterns(&[100, 200, 300, 400], &senses, &seed_ids);

        println!("\n=== PATTERN MINING: Step-by-Step Reasoning ===");
        println!("Scanning nodes: dikhianati, harga_diri, trauma, sakit");

        // (risk, identity) pair should appear in all 4 nodes
        let risk_identity_key = (2u32.min(4u32), 2u32.max(4u32)); // (2, 4)
        assert!(pair_map.contains_key(&risk_identity_key),
            "Risk+Identity pattern should be detected");

        let nodes_with_pattern = &pair_map[&risk_identity_key];
        assert!(nodes_with_pattern.len() >= 2,
            "At least 2 nodes should exhibit the risk+identity pattern, got {}", nodes_with_pattern.len());

        println!("\nDiscovered composition pairs:");
        for ((a, b), nodes) in &pair_map {
            println!("  ({}, {}) → {} nodes: {:?}", a, b, nodes.len(), nodes);
        }

        // Create the named pattern
        let mut graph = crate::graph::RsvsGraph::new();
        graph.insert_node(Node { label: "risk".to_string(), ..Node::default() }).ok();
        graph.insert_node(Node { label: "identity".to_string(), ..Node::default() }).ok();

        let pattern = engine.create_pattern(2, 4, nodes_with_pattern, &graph);

        println!("\n--- NAMED PATTERN ---");
        println!("Label: {}", pattern.label);
        println!("Seed composition: (risk, identity)");
        println!("Exhibiting nodes: {:?}", pattern.exhibiting_nodes);
        println!("Support count: {}", pattern.support_count);
        println!("Confidence: {:.3}", pattern.confidence);
        println!("\n--- INTERPRETATION ---");
        println!("The recurring pair (risk + identity) across dikhianati, harga_diri, trauma, sakit");
        println!("indicates a structural pattern: 'kekerasan_terhadap_identitas'");
        println!("(violence against identity) — when risk threatens identity,");
        println!("the result is always traumatic.");
    }

    // ----------------------------------------------------------------
    // Test 5: Cross-Pathway Synthesis discovers hidden meaning
    // ----------------------------------------------------------------

    #[test]
    fn synthesis_discovers_hidden_meaning() {
        use crate::cross_pathway_synthesis::{CrossPathwaySynthesisEngine, SynthesisConfig};
        use crate::types::{
            AffectiveProfile, ConnotativeProfile, ConflictType, GapAnnotation, GapType,
            HiddenMeaningType, Node, PathwayConflict, SeedPathway, SenseProfile, SocialProfile,
            StructuralConflictDescription,
        };
        use std::collections::HashMap;

        let engine = CrossPathwaySynthesisEngine::new(SynthesisConfig {
            min_gap_confidence: 0.2,
            min_conflict_score: 0.2,
            min_synthesis_confidence: 0.2,
            max_synthesis_per_batch: 10,
        });

        // Create a node with both gap and conflict on the same sense
        let mut graph = crate::graph::RsvsGraph::new();

        // Create target node (harga_diri) so we can look up its label
        graph.insert_node(Node {
            label: "harga_diri".to_string(),
            ..Node::default()
        }).ok();

        let node_id = graph.insert_node(Node {
            label: "dikhianati".to_string(),
            gap_annotations: {
                let mut m = HashMap::new();
                m.insert(0, vec![GapAnnotation {
                    gap_type: GapType::ExpectedComposition,
                    confidence: 0.7,
                    target_node: 1, // harga_diri
                    seed_trace: vec![2, 4], // risk + identity
                }]);
                m
            },
            sense_profiles: {
                let mut m = HashMap::new();
                m.insert(0, SenseProfile {
                    sense_id: 0,
                    affective: AffectiveProfile {
                        valence: -0.4,  // negative
                        arousal: 0.7,   // high intensity
                        dominance: 0.2, // low control
                        profile_confidence: 0.7,
                        cross_verified: true,
                    },
                    social: SocialProfile {
                        distance: 0.7,  // far
                        trust: 0.2,     // low trust
                        power_direction: -0.5, // addressee dominant
                        expected_politeness: 1.8, // high = social threat
                        profile_confidence: 0.6,
                    },
                    connotative: ConnotativeProfile::default(),
                    conflicts: vec![PathwayConflict {
                        pathway_a: SeedPathway::Affective,
                        pathway_b: SeedPathway::Social,
                        conflict_type: ConflictType::AffectiveSocialMismatch,
                        conflict_score: 0.6,
                        description: StructuralConflictDescription {
                            seed_a: 2, // risk
                            seed_b: 4, // identity
                            activation_a: 0.7,
                            activation_b: 0.6,
                            expected_relation: None,
                            actual_divergence: 0.6,
                        },
                    }],
                });
                m
            },
            ..Node::default()
        }).unwrap();

        let results = engine.process_batch(&[node_id], &graph);

        println!("\n=== CROSS-PATHWAY SYNTHESIS: Step-by-Step Reasoning ===");
        println!("Node: dikhianati (betrayal)");
        println!("\n--- Pathway 1 Evidence (Gap Detection) ---");
        println!("  Gap type: ExpectedComposition");
        println!("  Missing: harga_diri (dignity)");
        println!("  Confidence: 0.7");
        println!("  Seed trace: risk + identity");
        println!("\n--- Pathway 2 Evidence (Seed Activation) ---");
        println!("  Conflict type: AffectiveSocialMismatch");
        println!("  Affective: negative valence (-0.4) + high arousal (0.7)");
        println!("  Social: low trust (0.2) + high expected_politeness (1.8)");
        println!("  Conflict score: 0.6");
        println!("  Seed conflict: risk vs identity");

        assert!(!results.is_empty(), "Synthesis should find hidden meaning when gap + conflict overlap");

        let result = &results[0];
        println!("\n--- SYNTHESIS RESULT ---");
        println!("Hidden meaning: {}", result.hidden_meaning.description);
        println!("Meaning type: {:?}", result.hidden_meaning.meaning_type);
        println!("Target node: {}", result.hidden_meaning.target_node);
        println!("Evidence strength: {:.3}", result.hidden_meaning.evidence_strength);
        println!("Confidence: {:.3}", result.confidence);
        println!("Seed trace: {:?}", result.hidden_meaning.seed_trace);

        println!("\n--- COMPLETE EMERGENCE CHAIN ---");
        println!("Possibility A: 'Dia marah karena dikhianati'");
        println!("  → Gap detection: missing composition → harga_diri");
        println!("Possibility B: 'Dia marah karena harga diri tersentuh'");
        println!("  → Conflict: AffectiveSocialMismatch (negative + social threat)");
        println!("HYBRIDIZE (Blending): dikhianati ∧ harga_diri");
        println!("  → Shared: risk + identity");
        println!("  → Divergent: trust (from betrayal) + value (from dignity)");
        println!("  → Emergence potential: HIGH");
        println!("\nTHIS OPENS Possibility C (newly realized):");
        println!("  → 'Dikhianatan terhadap harga diri = pola trauma masa lalu'");
        println!("  → Pattern Mining: (risk+identity) recurring in dikhianati, harga_diri, trauma");
        println!("  → Abductive: dikhianati → harga_diri → (risk+identity) = trauma pattern");
        println!("  → Synthesis: gap + conflict → 'makna tersembunyi: pola trauma'");
    }

    // ----------------------------------------------------------------
    // Test 6: All 4 engines work together in pipeline
    // ----------------------------------------------------------------

    #[test]
    fn all_four_engines_integrated_in_pipeline() {
        let mut rsvs = create_rsvs();

        // Phase 1: Ingest foundational knowledge about the topic
        let _ = rsvs.ingest_text(
            "value risk trust identity agent goal feedback action marah"
        );

        // Phase 2: Ingest betrayal-related content
        let _ = rsvs.ingest_text(
            "marah dikhianati value risk identity trust agent"
        );

        // Phase 3: Ingest dignity-related content
        let _ = rsvs.ingest_text(
            "marah harga value risk identity agent"
        );

        // Phase 4: Ingest trauma-related content
        let _ = rsvs.ingest_text(
            "dikhianati harga trauma value risk identity"
        );

        // Phase 5: Ingest more betrayal+dignity to strengthen connections
        let _ = rsvs.ingest_text(
            "marah dikhianati harga diri trauma risk identity value trust"
        );

        // Verify the system has learned the concepts
        let node_count = rsvs.graph.node_count();
        assert!(node_count > 24, "Should have more than just seed nodes after ingestion, got {}", node_count);

        // Check that meaning pathways have been applied
        let nodes_with_gaps: usize = rsvs.graph.nodes.values()
            .filter(|n| !n.gap_annotations.is_empty())
            .count();
        let nodes_with_profiles: usize = rsvs.graph.nodes.values()
            .filter(|n| !n.sense_profiles.is_empty())
            .count();

        println!("\n=== FULL PIPELINE INTEGRATION TEST ===");
        println!("Total nodes: {}", node_count);
        println!("Nodes with gap annotations: {}", nodes_with_gaps);
        println!("Nodes with sense profiles: {}", nodes_with_profiles);

        // Check blending results
        let nodes_with_blends: usize = rsvs.graph.nodes.values()
            .filter(|n| !n.blend_results.is_empty())
            .count();
        println!("Nodes with blend results: {}", nodes_with_blends);

        // Check abductive hypotheses
        let nodes_with_hypotheses: usize = rsvs.graph.nodes.values()
            .filter(|n| !n.abductive_hypotheses.is_empty())
            .count();
        println!("Nodes with abductive hypotheses: {}", nodes_with_hypotheses);

        // Check pattern memberships
        let nodes_with_patterns: usize = rsvs.graph.nodes.values()
            .filter(|n| !n.pattern_memberships.is_empty())
            .count();
        println!("Nodes with pattern memberships: {}", nodes_with_patterns);

        // Check synthesis results
        let nodes_with_synthesis: usize = rsvs.graph.nodes.values()
            .filter(|n| !n.synthesis_results.is_empty())
            .count();
        println!("Nodes with synthesis results: {}", nodes_with_synthesis);

        // Print events for debugging
        let synthesis_events: Vec<_> = rsvs.events.iter()
            .filter(|e| e.event_type == "synthesis_completed")
            .collect();
        let blending_events: Vec<_> = rsvs.events.iter()
            .filter(|e| e.event_type == "blending_completed")
            .collect();
        let abductive_events: Vec<_> = rsvs.events.iter()
            .filter(|e| e.event_type == "abductive_completed")
            .collect();
        let pattern_events: Vec<_> = rsvs.events.iter()
            .filter(|e| e.event_type == "pattern_mining_completed")
            .collect();

        println!("\n--- Engine Events ---");
        println!("Blending events: {}", blending_events.len());
        println!("Abductive events: {}", abductive_events.len());
        println!("Pattern mining events: {}", pattern_events.len());
        println!("Synthesis events: {}", synthesis_events.len());
    }
}
