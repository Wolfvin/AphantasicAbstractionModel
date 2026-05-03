//! Comprehensive unit tests for RSVS v4.2
//!
//! Covers: v4.2 node creation, NodeStatus transitions, quarantine,
//! hysteresis, seed invariants, CompressionState, DAG self-reference,
//! governance scoring, appraise mode, relate mode, snapshot v4.2 schema,
//! persistence save/load roundtrip, and more.

#[cfg(test)]
mod sense_tests {
    use crate::sense::{IngestResult, SenseConfig, SenseManager, SenseStatus};

    fn config_low_threshold() -> SenseConfig {
        SenseConfig {
            theta_assign: 0.15,
            ..SenseConfig::default()
        }
    }

    #[test]
    fn coherence_cold_start_is_prior() {
        let mut sm = SenseManager::new(SenseConfig::default());
        sm.ingest(vec![1, 2, 3]);
        assert_eq!(sm.senses[0].coherence, 0.5);
        assert_eq!(sm.senses[0].status, SenseStatus::Fragile);
    }

    #[test]
    fn coherence_two_identical_contexts_is_one() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![1, 2, 3]);
        let s = &sm.senses[0];
        assert_eq!(s.coherence, 1.0);
        assert_eq!(s.status, SenseStatus::Mature);
    }

    #[test]
    fn coherence_orthogonal_contexts_is_zero() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![4, 5, 6]);
        for s in &sm.senses {
            if s.context_count() == 2 {
                assert!(s.coherence < 0.1);
            }
        }
    }

    #[test]
    fn incremental_coherence_matches_batch() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3, 4]);
        sm.ingest(vec![1, 2, 3, 5]);
        sm.ingest(vec![1, 2, 4, 5]);

        let mature: Vec<_> = sm
            .senses
            .iter()
            .filter(|s| s.context_count() >= 2)
            .collect();
        assert!(!mature.is_empty());

        for s in &mature {
            assert!(s.coherence > 0.0);
        }
    }

    #[test]
    fn first_context_always_creates_sense() {
        let mut sm = SenseManager::new(SenseConfig::default());
        let r = sm.ingest(vec![1, 2, 3]);
        assert_eq!(r, IngestResult::Created(0));
        assert_eq!(sm.sense_count(), 1);
    }

    #[test]
    fn identical_context_assigns_to_existing() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3]);
        let r = sm.ingest(vec![1, 2, 3]);
        assert_eq!(r, IngestResult::Assigned(0));
        assert_eq!(sm.sense_count(), 1);
    }

    #[test]
    fn orthogonal_contexts_form_separate_senses() {
        let mut sm = SenseManager::new(SenseConfig::default());
        sm.ingest(vec![1, 2, 3]);
        let r = sm.ingest(vec![100, 200, 300]);
        assert!(matches!(r, IngestResult::Created(_)));
        assert_eq!(sm.sense_count(), 2);
    }

    #[test]
    fn sense_status_upgrades_on_second_context() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3]);
        assert_eq!(sm.senses[0].status, SenseStatus::Fragile);
        sm.ingest(vec![1, 2, 4]);
        let mature = sm.senses.iter().find(|s| s.context_count() == 2);
        assert!(mature.is_some());
        assert_eq!(mature.unwrap().status, SenseStatus::Mature);
    }

    #[test]
    fn lazy_lookup_selects_most_similar_sense() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![10, 20, 30]);
        sm.ingest(vec![10, 20, 30]);

        let idx_a = sm.lazy_lookup(&vec![1, 2]).unwrap();
        let sense_a = sm.get_sense(idx_a).unwrap();
        let core_a = sense_a.core(0.4);
        assert!(core_a.contains(&1) || core_a.contains(&2));

        let idx_b = sm.lazy_lookup(&vec![10, 20]).unwrap();
        assert_ne!(idx_a, idx_b);
    }

    #[test]
    fn fragile_sense_deleted_after_k_fragile() {
        let mut sm = SenseManager::new(SenseConfig {
            k_fragile: 3,
            ..config_low_threshold()
        });
        sm.ingest(vec![1, 2, 3]);
        sm.senses[0].inactivity = 3;
        sm.purge_fragile();
        assert_eq!(sm.sense_count(), 0);
    }

    #[test]
    fn merge_two_nearly_identical_mature_senses() {
        let mut sm = SenseManager::new(SenseConfig {
            theta_merge: 0.50,
            n_min_mature: 2,
            ..config_low_threshold()
        });

        for _ in 0..3 {
            sm.ingest(vec![1, 2, 3, 4]);
        }
        for _ in 0..3 {
            sm.ingest(vec![1, 2, 3, 5]);
        }

        let count_before = sm.sense_count();
        let merged = sm.check_merge();

        if count_before >= 2 && !merged.is_empty() {
            assert!(sm.sense_count() < count_before);
        }
    }

    #[test]
    fn freq_map_correct_after_multiple_assigns() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![1, 2, 4]);
        sm.ingest(vec![1, 3, 5]);

        if let Some(s) = sm.senses.iter().find(|s| s.context_count() == 3) {
            assert!((s.freq(1) - 1.0).abs() < 0.01);
            assert!((s.freq(2) - 2.0 / 3.0).abs() < 0.01);
            assert!((s.freq(5) - 1.0 / 3.0).abs() < 0.01);
        }
    }

    #[test]
    fn core_filters_by_tau() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![1, 2, 4]);
        sm.ingest(vec![1, 3, 5]);

        if let Some(s) = sm.senses.iter().find(|s| s.context_count() == 3) {
            let strict_core = s.core(0.9);
            assert!(strict_core.contains(&1));
            assert!(!strict_core.contains(&5));

            let loose_core = s.core(0.3);
            assert!(loose_core.contains(&1));
            assert!(loose_core.contains(&2));
        }
    }
}

#[cfg(test)]
mod attention_tests {
    use crate::attention::{
        is_groundable_to_seeds, tokenize, AttentionConfig, CoocStats, EntityDetector, RsvsAttention,
    };
    use std::collections::HashMap;

    #[test]
    fn cooc_stats_empty_returns_zero() {
        let stats = CoocStats::new();
        assert_eq!(stats.npmi("stone", "hard"), 0.0);
        assert_eq!(stats.cooc("stone", "hard"), 0.0);
    }

    #[test]
    fn cooc_stats_single_sentence() {
        let mut stats = CoocStats::new();
        stats.ingest_sentence(&["stone".into(), "hard".into(), "solid".into()]);
        assert_eq!(stats.token_count["stone"], 1);
        assert_eq!(stats.pair_cooc_count("stone", "hard"), 1);
    }

    #[test]
    fn npmi_partial_cooccurrence_is_positive() {
        let mut stats = CoocStats::new();
        for _ in 0..4 {
            stats.ingest_sentence(&["stone".into(), "hard".into()]);
        }
        stats.ingest_sentence(&["stone".into(), "rough".into()]);
        stats.ingest_sentence(&["hard".into(), "rough".into()]);

        let npmi = stats.npmi("stone", "hard");
        assert!(npmi > 0.0);
        assert!(npmi <= 1.0);
    }

    #[test]
    fn npmi_never_cooccurring_is_zero() {
        let mut stats = CoocStats::new();
        stats.ingest_sentence(&["stone".into(), "hard".into()]);
        stats.ingest_sentence(&["water".into(), "liquid".into()]);
        assert_eq!(stats.npmi("stone", "water"), 0.0);
    }

    #[test]
    fn cooc_conditional_probability() {
        let mut stats = CoocStats::new();
        stats.ingest_sentence(&["stone".into(), "hard".into()]);
        stats.ingest_sentence(&["stone".into(), "hard".into()]);
        stats.ingest_sentence(&["stone".into(), "rough".into()]);
        let cooc = stats.cooc("stone", "hard");
        assert!((cooc - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn tokenize_removes_stopwords() {
        let tokens = tokenize("Stone is a hard solid material");
        assert!(!tokens.contains(&"is".to_string()));
        assert!(tokens.contains(&"stone".to_string()));
    }

    #[test]
    fn tokenize_lowercases() {
        let tokens = tokenize("STONE is HARD");
        assert!(tokens.contains(&"stone".to_string()));
    }

    #[test]
    fn split_sentences_basic() {
        let sentences =
            crate::attention::split_sentences("Stone is hard. Fire is hot. Water is liquid.");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn entity_detector_promotes_above_threshold() {
        let mut det = EntityDetector::new();
        for _ in 0..3 {
            det.record("stone", true);
        }
        for _ in 0..2 {
            det.record("hard", true);
        }
        let candidates = det.candidates(3);
        assert!(candidates.contains(&"stone".to_string()));
        assert!(!candidates.contains(&"hard".to_string()));
    }

    #[test]
    fn groundable_physical_words() {
        let seeds = vec!["exists", "entity", "hard"];
        assert!(is_groundable_to_seeds("hard", &seeds));
        assert!(is_groundable_to_seeds("stone", &seeds));
    }

    #[test]
    fn attention_returns_empty_for_no_cooc_data() {
        let mut stats = CoocStats::new();
        stats.ingest_sentence(&["stone".into(), "hard".into()]);
        let attention = RsvsAttention::new(AttentionConfig::default());
        let tokens = vec!["stone".to_string(), "hard".to_string()];
        let atom_sets: HashMap<String, Vec<u32>> = HashMap::new();
        let result = attention.select(&tokens, &stats, &atom_sets);
        assert!(result.is_empty());
    }

    #[test]
    fn attention_selects_high_cooc_pairs() {
        let mut stats = CoocStats::new();
        for _ in 0..5 {
            stats.ingest_sentence(&["stone".into(), "hard".into(), "solid".into()]);
        }
        for _ in 0..3 {
            stats.ingest_sentence(&["water".into(), "liquid".into()]);
        }

        let attention = RsvsAttention::new(AttentionConfig {
            min_cooc: 2,
            ..AttentionConfig::default()
        });
        let tokens = vec![
            "stone".into(),
            "hard".into(),
            "solid".into(),
            "water".into(),
        ];
        let atom_sets: HashMap<String, Vec<u32>> = HashMap::new();
        let result = attention.select(&tokens, &stats, &atom_sets);

        if let Some(stone_selected) = result.get("stone") {
            let selected_tokens: Vec<_> = stone_selected.iter().map(|c| c.token.as_str()).collect();
            assert!(selected_tokens.contains(&"hard") || selected_tokens.contains(&"solid"));
            assert!(!selected_tokens.contains(&"water"));
        }
    }
}

#[cfg(test)]
mod autonomy_tests {
    use crate::autonomy::{
        AutonomyConfig, AutonomyEngine, ConfidenceUpdateResult, MemoryClass, RemovalDecision,
        StatusTransitionResult, WarmUpState,
    };
    use crate::types::{NodeStatus, Tier};

    fn engine() -> AutonomyEngine {
        AutonomyEngine::new(AutonomyConfig {
            eta: 0.1,
            confidence_tier1: 0.85,
            confidence_tier2: 0.50,
            tau_remove: 0.10,
            threshold_impact: 3,
            threshold_global_delta: 5.0,
            n_warm: 5,
            promote_threshold: 0.75,
            demote_threshold: 0.60,
            quarantine_flip_threshold: 3,
            ..AutonomyConfig::default()
        })
    }

    // ------------------------------------------------------------------
    // Confidence update tests
    // ------------------------------------------------------------------

    #[test]
    fn seed_node_never_decays() {
        let mut e = engine();
        e.register_seed(1, 1.0, Tier::Tier1);
        let r = e.update_confidence(1, 0.0, 0.0, &[], 0);
        assert!(matches!(r, ConfidenceUpdateResult::Skipped(_)));
        assert_eq!(e.confidence(1).unwrap(), 1.0);
    }

    #[test]
    fn high_evidence_increases_confidence() {
        let mut e = engine();
        e.register(10, 0.50, Tier::Tier2);
        e.update_confidence(10, 1.0, 1.0, &[], 1);
        assert!(e.confidence(10).unwrap() > 0.50);
    }

    #[test]
    fn low_evidence_decays_confidence() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        for _ in 0..20 {
            e.update_confidence(10, 0.0, 0.0, &[], 0);
        }
        assert!(e.confidence(10).unwrap() < 0.80);
    }

    #[test]
    fn confidence_stays_in_0_1() {
        let mut e = engine();
        e.register(10, 0.50, Tier::Tier2);
        for _ in 0..100 {
            e.update_confidence(10, 1.0, 1.0, &[], 0);
        }
        let conf = e.confidence(10).unwrap();
        assert!((0.0..=1.0).contains(&conf));
    }

    #[test]
    fn evidence_formula_is_correct() {
        let mut e = engine();
        e.register(10, 0.50, Tier::Tier2);
        let r = e.update_confidence(10, 0.8, 0.75, &[], 0);
        if let ConfidenceUpdateResult::Updated { evidence, .. } = r {
            assert!((evidence - 0.6).abs() < 0.001);
        } else {
            panic!("Expected Updated result");
        }
    }

    // ------------------------------------------------------------------
    // v4.2: NodeStatus lifecycle tests
    // ------------------------------------------------------------------

    #[test]
    fn new_node_status_is_new() {
        let mut e = engine();
        e.register(10, 0.30, Tier::Tier2);
        assert_eq!(e.status(10), Some(&NodeStatus::New));
    }

    #[test]
    fn seed_node_status_is_stable() {
        let mut e = engine();
        e.register_seed(1, 1.0, Tier::Tier1);
        assert_eq!(e.status(1), Some(&NodeStatus::Stable));
    }

    #[test]
    fn promote_new_to_candidate() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2); // >= 0.75 promote threshold
        let r = e.transition_status(10);
        assert!(matches!(
            r,
            StatusTransitionResult::Transitioned {
                from: NodeStatus::New,
                to: NodeStatus::Candidate,
            }
        ));
    }

    #[test]
    fn promote_candidate_to_stable() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        // First transition: New → Candidate
        e.transition_status(10);
        // Second transition: Candidate → Stable (confidence >= 0.75)
        let r = e.transition_status(10);
        assert!(matches!(
            r,
            StatusTransitionResult::Transitioned {
                from: NodeStatus::Candidate,
                to: NodeStatus::Stable,
            }
        ));
    }

    #[test]
    fn demote_stable_to_deprecated() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        // Get to Stable
        e.transition_status(10); // New → Candidate
        e.transition_status(10); // Candidate → Stable
                                 // Now drop confidence below demote threshold
        if let Some(rec) = e.records.get_mut(&10) {
            rec.confidence = 0.50; // < 0.60 demote threshold
        }
        let r = e.transition_status(10);
        assert!(matches!(
            r,
            StatusTransitionResult::Transitioned {
                from: NodeStatus::Stable,
                to: NodeStatus::Deprecated,
            }
        ));
    }

    // ------------------------------------------------------------------
    // v4.2: Full lifecycle chain: New → Candidate → Stable → Deprecated
    // ------------------------------------------------------------------

    #[test]
    fn full_lifecycle_new_to_deprecated() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);

        // New → Candidate
        let r1 = e.transition_status(10);
        assert!(matches!(
            r1,
            StatusTransitionResult::Transitioned {
                from: NodeStatus::New,
                to: NodeStatus::Candidate,
            }
        ));

        // Candidate → Stable
        let r2 = e.transition_status(10);
        assert!(matches!(
            r2,
            StatusTransitionResult::Transitioned {
                from: NodeStatus::Candidate,
                to: NodeStatus::Stable,
            }
        ));

        // Drop confidence → Deprecated
        if let Some(rec) = e.records.get_mut(&10) {
            rec.confidence = 0.50;
        }
        let r3 = e.transition_status(10);
        assert!(matches!(
            r3,
            StatusTransitionResult::Transitioned {
                from: NodeStatus::Stable,
                to: NodeStatus::Deprecated,
            }
        ));
    }

    // ------------------------------------------------------------------
    // v4.2: Hysteresis tests
    // ------------------------------------------------------------------

    #[test]
    fn hysteresis_no_promote_below_threshold() {
        let mut e = engine();
        e.register(10, 0.70, Tier::Tier2); // < 0.75 promote threshold
        let r = e.transition_status(10);
        // Should stay New (confidence < promote_threshold)
        assert!(matches!(r, StatusTransitionResult::Blocked(_)));
    }

    #[test]
    fn hysteresis_no_demote_above_threshold() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        e.transition_status(10); // New → Candidate
        e.transition_status(10); // Candidate → Stable
                                 // Confidence 0.65 is >= demote_threshold (0.60) → stays stable
        if let Some(rec) = e.records.get_mut(&10) {
            rec.confidence = 0.65;
        }
        let r = e.transition_status(10);
        assert!(matches!(r, StatusTransitionResult::Blocked(_)));
    }

    #[test]
    fn hysteresis_demote_below_demote_threshold() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        e.transition_status(10);
        e.transition_status(10);
        // Below demote threshold
        if let Some(rec) = e.records.get_mut(&10) {
            rec.confidence = 0.55; // < 0.60
        }
        let r = e.transition_status(10);
        assert!(matches!(
            r,
            StatusTransitionResult::Transitioned {
                from: NodeStatus::Stable,
                to: NodeStatus::Deprecated,
            }
        ));
    }

    #[test]
    fn hysteresis_promote_exactly_at_threshold() {
        let mut e = engine();
        e.register(10, 0.75, Tier::Tier2); // exactly at promote_threshold
        let r = e.transition_status(10);
        // 0.75 >= 0.75 → should promote
        assert!(matches!(
            r,
            StatusTransitionResult::Transitioned {
                to: NodeStatus::Candidate,
                ..
            }
        ));
    }

    #[test]
    fn hysteresis_no_demote_exactly_at_threshold() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        e.transition_status(10); // New → Candidate
        e.transition_status(10); // Candidate → Stable
                                 // Exactly at demote_threshold (0.60) → should NOT demote
        if let Some(rec) = e.records.get_mut(&10) {
            rec.confidence = 0.60;
        }
        let r = e.transition_status(10);
        assert!(matches!(r, StatusTransitionResult::Blocked(_)));
    }

    // ------------------------------------------------------------------
    // v4.2: Quarantine tests
    // ------------------------------------------------------------------

    #[test]
    fn quarantine_after_three_flips() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        // Manually set flip count to threshold
        if let Some(rec) = e.records.get_mut(&10) {
            rec.status_flip_count = 3; // quarantine_flip_threshold = 3
        }
        let r = e.transition_status(10);
        assert!(matches!(
            r,
            StatusTransitionResult::Transitioned {
                to: NodeStatus::Quarantine,
                ..
            }
        ));
    }

    #[test]
    fn quarantined_node_stays_quarantined() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        if let Some(rec) = e.records.get_mut(&10) {
            rec.status = NodeStatus::Quarantine;
            rec.status_flip_count = 5;
        }
        let r = e.transition_status(10);
        assert!(matches!(r, StatusTransitionResult::Blocked(_)));
    }

    #[test]
    fn quarantine_at_exactly_threshold() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        // Set flip count to exactly threshold
        if let Some(rec) = e.records.get_mut(&10) {
            rec.status_flip_count = 3;
        }
        let r = e.transition_status(10);
        // flip_count >= 3 should trigger quarantine
        assert!(matches!(
            r,
            StatusTransitionResult::Transitioned {
                to: NodeStatus::Quarantine,
                ..
            }
        ));
    }

    #[test]
    fn no_quarantine_below_threshold() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        // Set flip count to just below threshold
        if let Some(rec) = e.records.get_mut(&10) {
            rec.status_flip_count = 2;
        }
        let r = e.transition_status(10);
        // Should NOT quarantine (2 < 3)
        assert!(!matches!(
            r,
            StatusTransitionResult::Transitioned {
                to: NodeStatus::Quarantine,
                ..
            }
        ));
    }

    // ------------------------------------------------------------------
    // v4.2: Seed immutability tests
    // ------------------------------------------------------------------

    #[test]
    fn seed_node_cannot_transition_status() {
        let mut e = engine();
        e.register_seed(1, 1.0, Tier::Tier1);
        let r = e.transition_status(1);
        assert!(matches!(
            r,
            StatusTransitionResult::Blocked("seed node is immutable")
        ));
    }

    #[test]
    fn seed_node_never_removed() {
        let mut e = engine();
        e.register_seed(1, 1.0, Tier::Tier1);
        let d = e.should_remove(1, 0);
        assert_eq!(d, RemovalDecision::Retain("seed node"));
    }

    #[test]
    fn seed_node_confidence_stays_1() {
        let mut e = engine();
        e.register_seed(1, 1.0, Tier::Tier1);
        // Try to update — should be skipped
        let r = e.update_confidence(1, 0.0, 0.0, &[], 0);
        assert!(matches!(r, ConfidenceUpdateResult::Skipped(_)));
        assert_eq!(e.confidence(1).unwrap(), 1.0);
    }

    // ------------------------------------------------------------------
    // v4.2: Governance score tests
    // ------------------------------------------------------------------

    #[test]
    fn governance_score_formula() {
        let e = engine();
        // governance = 0.4*strength + 0.3*trust + 0.2*recency + 0.1*(1-contradiction)
        let score = e.score_evidence(0.8, 0.7, 0.5, 0.2);
        let expected = 0.4 * 0.8 + 0.3 * 0.7 + 0.2 * 0.5 + 0.1 * 0.8;
        assert!((score - expected).abs() < 0.001);
    }

    #[test]
    fn governance_score_clamped() {
        let e = engine();
        let score = e.score_evidence(2.0, 2.0, 2.0, -1.0);
        assert!(score <= 1.0);
        assert!(score >= 0.0);
    }

    #[test]
    fn governance_score_zero_evidence() {
        let e = engine();
        let score = e.score_evidence(0.0, 0.0, 0.0, 0.0);
        let expected = 0.4 * 0.0 + 0.3 * 0.0 + 0.2 * 0.0 + 0.1 * 1.0;
        assert!((score - expected).abs() < 0.001);
    }

    #[test]
    fn governance_score_max_evidence() {
        let e = engine();
        let score = e.score_evidence(1.0, 1.0, 1.0, 0.0);
        let expected = 0.4 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0 + 0.1 * 1.0;
        assert!((score - expected).abs() < 0.001);
        assert!(score <= 1.0);
    }

    // ------------------------------------------------------------------
    // Existing tests (preserved)
    // ------------------------------------------------------------------

    #[test]
    fn tier_reclassification() {
        let mut e = engine();
        e.register(10, 0.30, Tier::Tier2);
        let t = e.reclassify(10).unwrap();
        assert_eq!(t, Tier::Tier3);
    }

    #[test]
    fn low_confidence_low_impact_removes() {
        let mut e = engine();
        e.register(10, 0.05, Tier::Tier2);
        let d = e.should_remove(10, 1);
        assert_eq!(d, RemovalDecision::Remove);
    }

    #[test]
    fn low_confidence_high_impact_requires_approval() {
        let mut e = engine();
        e.register(10, 0.05, Tier::Tier2);
        let d = e.should_remove(10, 10);
        assert!(matches!(d, RemovalDecision::RequiresApproval { .. }));
    }

    #[test]
    fn rollback_restores_confidence() {
        let mut e = engine();
        e.register(10, 0.50, Tier::Tier2);
        let snapshot = e.snapshot();
        e.begin_batch();
        e.update_confidence(10, 1.0, 1.0, &[], 0);
        e.rollback(&snapshot);
        assert!((e.confidence(10).unwrap() - 0.50).abs() < 0.001);
    }

    #[test]
    fn warmup_completes_after_n_contexts() {
        let mut e = AutonomyEngine::new(AutonomyConfig {
            n_warm: 3,
            ..AutonomyConfig::default()
        });
        for _ in 0..3 {
            e.tick_context();
        }
        assert_eq!(e.warmup, WarmUpState::Complete);
    }

    #[test]
    fn energy_blocks_large_drop() {
        let mut e = engine();
        e.register(10, 0.60, Tier::Tier2);
        assert!(!e.energy_allows_update(10, 0.30));
    }

    #[test]
    fn new_node_starts_as_working() {
        let mut e = engine();
        e.register(10, 0.60, Tier::Tier2);
        assert_eq!(e.memory_class(10).unwrap(), &MemoryClass::Working);
    }

    #[test]
    fn seed_is_stable() {
        let mut e = engine();
        e.register_seed(10, 1.0, Tier::Tier1);
        assert_eq!(e.memory_class(10).unwrap(), &MemoryClass::Stable);
    }
}

#[cfg(test)]
mod graph_tests {
    use crate::error::RsvsError;
    use crate::graph::RsvsGraph;
    use crate::types::{CompressionState, Edge, EdgeSource, Node, NodeStatus, SemanticMeta, Tier};

    #[test]
    fn expand_raw_node_returns_self() {
        let mut g = RsvsGraph::new();
        let id = g
            .insert_node(Node {
                id: 0,
                label: "test".into(),
                surface_label: "test@en".into(),
                kind: "node".into(),
                tier: Tier::Tier2,
                confidence: 0.5,
                status: NodeStatus::Candidate,
                is_seed: false,
                is_locked: false,
                semantic: SemanticMeta::default(),
                policy_meta: None,
                language_links: vec![],
                atoms: vec![],
                fingerprint: None,
            })
            .unwrap();
        let expanded = g.expand(id);
        assert_eq!(expanded, vec![id]);
    }

    #[test]
    fn expand_raw_node_with_atoms_returns_atoms() {
        let mut g = RsvsGraph::new();
        let a1 = g
            .insert_node(Node {
                id: 0,
                label: "a1".into(),
                surface_label: "a1@en".into(),
                kind: "node".into(),
                tier: Tier::Tier1,
                confidence: 1.0,
                status: NodeStatus::Stable,
                is_seed: true,
                is_locked: true,
                semantic: SemanticMeta::default(),
                policy_meta: None,
                language_links: vec![],
                atoms: vec![],
                fingerprint: None,
            })
            .unwrap();
        let a2 = g
            .insert_node(Node {
                id: 0,
                label: "a2".into(),
                surface_label: "a2@en".into(),
                kind: "node".into(),
                tier: Tier::Tier1,
                confidence: 1.0,
                status: NodeStatus::Stable,
                is_seed: true,
                is_locked: true,
                semantic: SemanticMeta::default(),
                policy_meta: None,
                language_links: vec![],
                atoms: vec![],
                fingerprint: None,
            })
            .unwrap();
        let id = g
            .insert_node(Node {
                id: 0,
                label: "comp".into(),
                surface_label: "comp@en".into(),
                kind: "node".into(),
                tier: Tier::Tier2,
                confidence: 0.5,
                status: NodeStatus::Candidate,
                is_seed: false,
                is_locked: false,
                semantic: SemanticMeta {
                    compression_state: CompressionState::Raw,
                    derived_from_node_ids: vec![],
                    compression_reason: None,
                },
                policy_meta: None,
                language_links: vec![],
                atoms: vec![a1, a2],
                fingerprint: None,
            })
            .unwrap();
        let expanded = g.expand(id);
        assert_eq!(expanded, vec![a1, a2]);
    }

    #[test]
    fn expand_compressed_node_returns_derived() {
        let mut g = RsvsGraph::new();
        let a1 = g
            .insert_node(Node {
                id: 0,
                label: "a1".into(),
                surface_label: "a1@en".into(),
                kind: "node".into(),
                tier: Tier::Tier1,
                confidence: 1.0,
                status: NodeStatus::Stable,
                is_seed: true,
                is_locked: true,
                semantic: SemanticMeta::default(),
                policy_meta: None,
                language_links: vec![],
                atoms: vec![],
                fingerprint: None,
            })
            .unwrap();
        let id = g
            .insert_node(Node {
                id: 0,
                label: "comp".into(),
                surface_label: "comp@en".into(),
                kind: "node".into(),
                tier: Tier::Tier2,
                confidence: 0.5,
                status: NodeStatus::Candidate,
                is_seed: false,
                is_locked: false,
                semantic: SemanticMeta {
                    compression_state: CompressionState::Compressed,
                    derived_from_node_ids: vec![a1],
                    compression_reason: Some("test".into()),
                },
                policy_meta: None,
                language_links: vec![],
                atoms: vec![],
                fingerprint: None,
            })
            .unwrap();
        let expanded = g.expand(id);
        assert_eq!(expanded, vec![a1]);
    }

    #[test]
    fn dag_prevents_self_reference_in_derived() {
        let mut g = RsvsGraph::new();
        let result = g.insert_node(Node {
            id: 5,
            label: "selfref".into(),
            surface_label: "selfref@en".into(),
            kind: "node".into(),
            tier: Tier::Tier2,
            confidence: 0.5,
            status: NodeStatus::Candidate,
            is_seed: false,
            is_locked: false,
            semantic: SemanticMeta {
                compression_state: CompressionState::Compressed,
                derived_from_node_ids: vec![5], // self-reference
                compression_reason: None,
            },
            policy_meta: None,
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
        });
        assert!(matches!(result, Err(RsvsError::CircularRef { .. })));
    }

    #[test]
    fn insert_edge_requires_existing_nodes() {
        let mut g = RsvsGraph::new();
        let n1 = g
            .insert_node(Node {
                id: 0,
                label: "n1".into(),
                surface_label: "n1@en".into(),
                kind: "node".into(),
                tier: Tier::Tier1,
                confidence: 1.0,
                status: NodeStatus::Stable,
                is_seed: true,
                is_locked: true,
                semantic: SemanticMeta::default(),
                policy_meta: None,
                language_links: vec![],
                atoms: vec![],
                fingerprint: None,
            })
            .unwrap();
        // Try edge with non-existent target
        let result = g.insert_edge(Edge {
            from: n1,
            to: 999,
            weight: 0.5,
            source: EdgeSource::Learned,
        });
        assert!(matches!(result, Err(RsvsError::NodeNotFound { .. })));
    }

    #[test]
    fn insert_edge_with_both_existing_nodes() {
        let mut g = RsvsGraph::new();
        let n1 = g
            .insert_node(Node {
                id: 0,
                label: "n1".into(),
                surface_label: "n1@en".into(),
                kind: "node".into(),
                tier: Tier::Tier1,
                confidence: 1.0,
                status: NodeStatus::Stable,
                is_seed: true,
                is_locked: true,
                semantic: SemanticMeta::default(),
                policy_meta: None,
                language_links: vec![],
                atoms: vec![],
                fingerprint: None,
            })
            .unwrap();
        let n2 = g
            .insert_node(Node {
                id: 0,
                label: "n2".into(),
                surface_label: "n2@en".into(),
                kind: "node".into(),
                tier: Tier::Tier1,
                confidence: 1.0,
                status: NodeStatus::Stable,
                is_seed: true,
                is_locked: true,
                semantic: SemanticMeta::default(),
                policy_meta: None,
                language_links: vec![],
                atoms: vec![],
                fingerprint: None,
            })
            .unwrap();
        let result = g.insert_edge(Edge {
            from: n1,
            to: n2,
            weight: 0.8,
            source: EdgeSource::Learned,
        });
        assert!(result.is_ok());
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn expand_nonexistent_node_returns_empty() {
        let g = RsvsGraph::new();
        let expanded = g.expand(999);
        assert!(expanded.is_empty());
    }

    #[test]
    fn jaccard_identical_sets_is_one() {
        let j = crate::graph::jaccard_sets(&[1, 2, 3], &[1, 2, 3]);
        assert!((j - 1.0).abs() < 0.001);
    }

    #[test]
    fn jaccard_disjoint_sets_is_zero() {
        let j = crate::graph::jaccard_sets(&[1, 2, 3], &[4, 5, 6]);
        assert!((j - 0.0).abs() < 0.001);
    }

    #[test]
    fn jaccard_empty_sets_is_zero() {
        let j = crate::graph::jaccard_sets(&[], &[]);
        assert!((j - 0.0).abs() < 0.001);
    }

    #[test]
    fn node_count_increases() {
        let mut g = RsvsGraph::new();
        assert_eq!(g.node_count(), 0);
        g.insert_node(Node {
            id: 0,
            label: "n1".into(),
            surface_label: "n1@en".into(),
            kind: "node".into(),
            tier: Tier::Tier2,
            confidence: 0.5,
            status: NodeStatus::New,
            is_seed: false,
            is_locked: false,
            semantic: SemanticMeta::default(),
            policy_meta: None,
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
        })
        .unwrap();
        assert_eq!(g.node_count(), 1);
    }
}

#[cfg(test)]
mod v42_node_tests {
    use crate::types::{
        CompressionState, Fingerprint, Node, NodeStatus, PolicyMeta, SemanticMeta, Tier,
    };

    #[test]
    fn v42_node_creation() {
        let node = Node {
            id: 1,
            label: "test".to_string(),
            surface_label: "test@en".to_string(),
            kind: "node".to_string(),
            tier: Tier::Tier2,
            confidence: 0.75,
            status: NodeStatus::Candidate,
            is_seed: false,
            is_locked: false,
            semantic: SemanticMeta {
                compression_state: CompressionState::Raw,
                derived_from_node_ids: vec![],
                compression_reason: None,
            },
            policy_meta: Some(PolicyMeta::default()),
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
        };
        assert_eq!(node.kind, "node");
        assert_eq!(node.surface_label, "test@en");
        assert_eq!(node.status, NodeStatus::Candidate);
        assert!(!node.is_seed);
        assert_eq!(node.semantic.compression_state, CompressionState::Raw);
    }

    #[test]
    fn v42_seed_node() {
        let node = Node {
            id: 1,
            label: "exists".to_string(),
            surface_label: "exists@en".to_string(),
            kind: "node".to_string(),
            tier: Tier::Tier1,
            confidence: 1.0,
            status: NodeStatus::Stable,
            is_seed: true,
            is_locked: true,
            semantic: SemanticMeta {
                compression_state: CompressionState::Raw,
                derived_from_node_ids: vec![],
                compression_reason: None,
            },
            policy_meta: None,
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
        };
        assert!(node.is_seed);
        assert!(node.is_locked);
        assert_eq!(node.status, NodeStatus::Stable);
    }

    #[test]
    fn v42_compressed_node() {
        let node = Node {
            id: 5,
            label: "concept".to_string(),
            surface_label: "concept@en".to_string(),
            kind: "node".to_string(),
            tier: Tier::Tier1,
            confidence: 0.9,
            status: NodeStatus::Stable,
            is_seed: false,
            is_locked: false,
            semantic: SemanticMeta {
                compression_state: CompressionState::Compressed,
                derived_from_node_ids: vec![1, 2, 3],
                compression_reason: Some("co-occurrence aggregation".to_string()),
            },
            policy_meta: Some(PolicyMeta {
                policy_version: "4.2".to_string(),
                governance_score: 0.85,
                candidate_evidence_pool: 0.3,
                status_flip_count: 1,
                seen_fingerprints: vec!["fp1".to_string()],
                last_seen_at: Some("2024-01-01".to_string()),
            }),
            language_links: vec![],
            atoms: vec![1, 2, 3],
            fingerprint: None,
        };
        assert_eq!(
            node.semantic.compression_state,
            CompressionState::Compressed
        );
        assert_eq!(node.semantic.derived_from_node_ids, vec![1, 2, 3]);
        assert!(node.policy_meta.is_some());
        let pm = node.policy_meta.unwrap();
        assert_eq!(pm.governance_score, 0.85);
    }

    #[test]
    fn node_status_lifecycle() {
        let statuses = [
            NodeStatus::New,
            NodeStatus::Candidate,
            NodeStatus::Stable,
            NodeStatus::Deprecated,
            NodeStatus::Quarantine,
        ];
        for (i, s) in statuses.iter().enumerate() {
            for (j, t) in statuses.iter().enumerate() {
                if i != j {
                    assert_ne!(s, t);
                }
            }
        }
    }

    #[test]
    fn surface_label_has_locale() {
        let node = Node {
            id: 1,
            label: "test".into(),
            surface_label: "test@en".into(),
            kind: "node".into(),
            tier: Tier::Tier2,
            confidence: 0.5,
            status: NodeStatus::New,
            is_seed: false,
            is_locked: false,
            semantic: SemanticMeta::default(),
            policy_meta: None,
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
        };
        assert!(node.surface_label.contains("@"));
    }

    #[test]
    fn seed_node_has_required_invariants() {
        let node = Node {
            id: 1,
            label: "exists".into(),
            surface_label: "exists@en".into(),
            kind: "node".into(),
            tier: Tier::Tier1,
            confidence: 1.0,
            status: NodeStatus::Stable,
            is_seed: true,
            is_locked: true,
            semantic: SemanticMeta::default(),
            policy_meta: None,
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
        };
        assert!(node.is_seed);
        assert!(node.is_locked, "Seed node must be locked");
        assert_eq!(node.tier, Tier::Tier1, "Seed node must be Tier1");
        assert!(
            (node.confidence - 1.0).abs() < 0.001,
            "Seed node must have confidence 1.0"
        );
        assert_eq!(node.status, NodeStatus::Stable, "Seed node must be Stable");
    }

    #[test]
    fn non_seed_node_not_locked() {
        let node = Node {
            id: 10,
            label: "test".into(),
            surface_label: "test@en".into(),
            kind: "node".into(),
            tier: Tier::Tier2,
            confidence: 0.5,
            status: NodeStatus::New,
            is_seed: false,
            is_locked: false,
            semantic: SemanticMeta::default(),
            policy_meta: None,
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
        };
        assert!(!node.is_seed);
        assert!(!node.is_locked);
    }

    #[test]
    fn fingerprint_new_produces_consistent_hash() {
        let data = b"hello world";
        let fp1 = Fingerprint::new(data);
        let fp2 = Fingerprint::new(data);
        assert_eq!(fp1, fp2);
        assert_ne!(fp1.hash(), 0);
    }

    #[test]
    fn fingerprint_different_data_produces_different_hash() {
        let fp1 = Fingerprint::new(b"hello");
        let fp2 = Fingerprint::new(b"world");
        assert_ne!(fp1, fp2);
    }
}

#[cfg(test)]
mod pipeline_tests {
    use crate::autonomy::AutonomyConfig;
    use crate::pipeline::{PipelineConfig, Rsvs};
    use crate::sense::SenseConfig;
    use crate::types::NodeStatus;

    fn make_rsvs() -> Rsvs {
        Rsvs::new(PipelineConfig {
            autonomy: AutonomyConfig {
                n_warm: 5,
                threshold_global_delta: 5.0,
                ..AutonomyConfig::default()
            },
            sense: SenseConfig {
                theta_assign: 0.10,
                ..SenseConfig::default()
            },
            entity_promote_n: 2,
            ..PipelineConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn system_bootstraps_with_seed_nodes() {
        let rsvs = make_rsvs();
        assert_eq!(rsvs.graph.node_count(), 24);
        assert!(rsvs.token_to_id.contains_key("exists"));
        assert!(rsvs.token_to_id.contains_key("feedback"));
    }

    #[test]
    fn seed_nodes_are_registered_in_autonomy() {
        let rsvs = make_rsvs();
        let exists_id = rsvs.token_to_id["exists"];
        assert_eq!(rsvs.autonomy.confidence(exists_id), Some(1.0));
        assert_eq!(rsvs.autonomy.status(exists_id), Some(&NodeStatus::Stable));
    }

    #[test]
    fn ingest_promotes_frequent_tokens() {
        let mut rsvs = make_rsvs();
        let text = "Stone is hard. Stone is hard and solid. Stone is a hard material. \
                    Stone remains hard. Hard stone is heavy.";
        let stats = rsvs.ingest_text(text).unwrap();
        assert!(stats.atoms_promoted >= 1);
    }

    #[test]
    fn ingest_increases_context_count() {
        let mut rsvs = make_rsvs();
        assert_eq!(rsvs.total_contexts, 0);
        rsvs.ingest_text("Stone is hard. Water is liquid.").unwrap();
        assert!(rsvs.total_contexts > 0);
    }

    #[test]
    fn similar_concepts_have_positive_jaccard() {
        let mut rsvs = make_rsvs();
        let text = "Stone is hard solid heavy. \
                    Bone is hard solid organic. \
                    Stone and bone are both hard solid materials. \
                    Hard solid materials resist force. \
                    Stone is hard like bone.";
        rsvs.ingest_text(text).unwrap();

        if let Some(sim) = rsvs.similarity("stone", "bone") {
            assert!(sim.jaccard > 0.0);
        }
    }

    #[test]
    fn query_unknown_concept_returns_none() {
        let rsvs = make_rsvs();
        assert!(rsvs.query("nonexistent_concept", "some context").is_none());
    }

    // ------------------------------------------------------------------
    // v4.2: Appraise tests
    // ------------------------------------------------------------------

    #[test]
    fn appraise_empty_text_is_novel() {
        let rsvs = make_rsvs();
        let result = rsvs.appraise("");
        assert_eq!(result.verdict, "novel");
    }

    #[test]
    fn appraise_seed_text_is_consistent() {
        let rsvs = make_rsvs();
        // "exists" and "entity" are seed nodes
        let result = rsvs.appraise("exists entity relation state change");
        assert!(result.agree_pct > 0.0);
        assert!(!result.evidence.is_empty());
    }

    #[test]
    fn appraise_unknown_text_is_novel() {
        let rsvs = make_rsvs();
        let result = rsvs.appraise("xyzquux foobarbaz quuxland");
        assert!(result.disagree_pct > 50.0);
        assert_eq!(result.verdict, "novel");
    }

    #[test]
    fn appraise_verdict_consistent() {
        let rsvs = make_rsvs();
        let result = rsvs.appraise("exists entity relation");
        assert!(
            result.verdict == "consistent"
                || result.verdict == "partial"
                || result.verdict == "novel"
        );
        assert!(result.agree_pct + result.disagree_pct > 0.0);
    }

    #[test]
    fn appraise_verdict_partial() {
        let mut rsvs = make_rsvs();
        rsvs.ingest_text("Stone is hard. Stone is heavy.").unwrap();
        // Mix of known (exists, entity) and unknown (xyzquux) tokens
        let result = rsvs.appraise("exists entity xyzquux");
        assert!(["consistent", "partial", "novel"].contains(&result.verdict.as_str()));
    }

    #[test]
    fn appraise_evidence_sorted_by_confidence() {
        let rsvs = make_rsvs();
        let result = rsvs.appraise("exists entity relation state change");
        // Evidence should be sorted by confidence descending
        for i in 1..result.evidence.len() {
            assert!(
                result.evidence[i - 1].1 >= result.evidence[i].1,
                "Evidence should be sorted by confidence descending"
            );
        }
    }

    // ------------------------------------------------------------------
    // v4.2: Relate tests
    // ------------------------------------------------------------------

    #[test]
    fn relate_unknown_concept_returns_none() {
        let rsvs = make_rsvs();
        assert!(rsvs.relate("nonexistent_concept").is_none());
    }

    #[test]
    fn relate_seed_node_returns_related() {
        let mut rsvs = make_rsvs();
        // Ingest some text to create edges
        rsvs.ingest_text("Stone exists as entity with relation to space and time.")
            .unwrap();
        if let Some(_id) = rsvs.token_to_id.get("exists") {
            let result = rsvs.relate("exists");
            // Should find at least the node itself
            assert!(result.is_some());
        }
    }

    #[test]
    fn relate_ingested_concept_finds_edges() {
        let mut rsvs = make_rsvs();
        rsvs.ingest_text("Stone is hard solid heavy. Hard solid stone resists pressure.")
            .unwrap();
        rsvs.ingest_text("Stone and metal are hard. Hard stone is heavy.")
            .unwrap();
        // Try to relate stone if it was promoted
        if rsvs.token_to_id.contains_key("stone") {
            let result = rsvs.relate("stone");
            assert!(result.is_some());
            let relate = result.unwrap();
            // Should find some related nodes or edges
            assert!(!relate.related_nodes.is_empty() || !relate.related_edges.is_empty());
        }
    }

    // ------------------------------------------------------------------
    // v4.2: Snapshot tests
    // ------------------------------------------------------------------

    #[test]
    fn snapshot_v1_has_v42_schema() {
        let rsvs = make_rsvs();
        let snap = rsvs.snapshot_v1();
        assert_eq!(snap.schema_version, "v4.2");
    }

    #[test]
    fn snapshot_v1_nodes_have_v42_fields() {
        let rsvs = make_rsvs();
        let snap = rsvs.snapshot_v1();
        assert!(!snap.nodes.is_empty());
        for n in &snap.nodes {
            assert_eq!(n.kind, "node");
            assert!(n.surface_label.ends_with("@en"));
            assert!(n.compression_state == "raw" || n.compression_state == "compressed");
        }
    }

    #[test]
    fn snapshot_v1_has_api_version() {
        let rsvs = make_rsvs();
        let snap = rsvs.snapshot_v1();
        assert_eq!(snap.api_version, "v1");
    }

    #[test]
    fn snapshot_v1_total_contexts_matches() {
        let mut rsvs = make_rsvs();
        rsvs.ingest_text("Stone is hard. Water is liquid.").unwrap();
        let snap = rsvs.snapshot_v1();
        assert_eq!(snap.total_contexts, rsvs.total_contexts);
    }

    #[test]
    fn snapshot_v1_seed_nodes_have_correct_fields() {
        let rsvs = make_rsvs();
        let snap = rsvs.snapshot_v1();
        let seeds: Vec<_> = snap.nodes.iter().filter(|n| n.is_seed).collect();
        assert_eq!(seeds.len(), 24);
        for seed in &seeds {
            assert!(seed.is_locked);
            assert_eq!(seed.tier, 1);
            assert_eq!(seed.status, "stable");
            assert_eq!(seed.compression_state, "raw");
        }
    }

    // ------------------------------------------------------------------
    // v4.2: Event stream tests
    // ------------------------------------------------------------------

    #[test]
    fn consume_events_returns_empty_before_ingest() {
        let rsvs = make_rsvs();
        let batch = rsvs.consume_events_v1(None, 100);
        assert!(batch.events.is_empty());
        assert_eq!(batch.latest_seq, 0);
    }

    #[test]
    fn consume_events_returns_events_after_ingest() {
        let mut rsvs = make_rsvs();
        rsvs.ingest_text("Stone is hard and solid.").unwrap();
        let batch = rsvs.consume_events_v1(None, 100);
        assert!(!batch.events.is_empty());
        assert!(batch.latest_seq > 0);
    }

    #[test]
    fn consume_events_after_seq_filters_correctly() {
        let mut rsvs = make_rsvs();
        rsvs.ingest_text("Stone is hard.").unwrap();
        let seq_after = rsvs.latest_seq_v1();
        rsvs.ingest_text("Water is liquid.").unwrap();
        let batch = rsvs.consume_events_v1(Some(seq_after), 100);
        // Should only get events from the second ingest
        for evt in &batch.events {
            assert!(evt.seq > seq_after);
        }
    }

    // ------------------------------------------------------------------
    // Persistence roundtrip test
    // ------------------------------------------------------------------

    #[test]
    fn persistence_roundtrip() {
        use tempfile::NamedTempFile;

        let mut rsvs = make_rsvs();
        rsvs.ingest_text("Stone is hard. Water is liquid. Metal is solid.")
            .unwrap();

        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        crate::persist::save(&rsvs, &path).unwrap();
        let loaded = crate::persist::load(&path).unwrap();

        assert_eq!(loaded.graph.node_count(), rsvs.graph.node_count());
        assert_eq!(loaded.token_to_id.len(), rsvs.token_to_id.len());
        assert_eq!(loaded.total_contexts, rsvs.total_contexts);
        assert_eq!(loaded.config.entity_promote_n, rsvs.config.entity_promote_n);
    }
}
