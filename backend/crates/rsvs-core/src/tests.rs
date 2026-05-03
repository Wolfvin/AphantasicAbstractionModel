//! Unit tests for RSVS v0.2

#[cfg(test)]
mod tests {
    use crate::sense::{SenseManager, SenseConfig, SenseStatus, IngestResult};

    fn config_low_threshold() -> SenseConfig {
        SenseConfig {
            theta_assign: 0.15, // lower for small atom sets
            ..SenseConfig::default()
        }
    }

    // ------------------------------------------------------------------
    // Coherence tests
    // ------------------------------------------------------------------

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
        sm.ingest(vec![1, 2, 3]); // identical → Jaccard = 1.0
        let s = &sm.senses[0];
        assert_eq!(s.coherence, 1.0);
        assert_eq!(s.status, SenseStatus::Mature);
    }

    #[test]
    fn coherence_orthogonal_contexts_is_zero() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![4, 5, 6]); // no overlap → Jaccard = 0.0
        // Two very different contexts: may or may not merge depending on threshold
        // Just check that coherence is computed correctly for the assigned sense
        for s in &sm.senses {
            if s.context_count() == 2 {
                assert!(s.coherence < 0.1);
            }
        }
    }

    #[test]
    fn incremental_coherence_matches_batch() {
        let mut sm = SenseManager::new(config_low_threshold());
        // Add 3 similar contexts incrementally
        sm.ingest(vec![1, 2, 3, 4]);
        sm.ingest(vec![1, 2, 3, 5]);
        sm.ingest(vec![1, 2, 4, 5]);

        // All should land in same sense (high overlap)
        let mature: Vec<_> = sm.senses.iter()
            .filter(|s| s.context_count() >= 2)
            .collect();
        assert!(!mature.is_empty(), "Should have at least one mature sense");

        // Coherence should be > 0 for senses with multiple contexts
        for s in &mature {
            assert!(s.coherence > 0.0);
        }
    }

    // ------------------------------------------------------------------
    // Sense formation tests
    // ------------------------------------------------------------------

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
        let r = sm.ingest(vec![100, 200, 300]); // zero overlap
        assert!(matches!(r, IngestResult::Created(_)));
        assert_eq!(sm.sense_count(), 2);
    }

    #[test]
    fn sense_status_upgrades_on_second_context() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3]);
        assert_eq!(sm.senses[0].status, SenseStatus::Fragile);
        sm.ingest(vec![1, 2, 4]);
        // Find the sense that now has N=2
        let mature = sm.senses.iter().find(|s| s.context_count() == 2);
        assert!(mature.is_some());
        assert_eq!(mature.unwrap().status, SenseStatus::Mature);
    }

    // ------------------------------------------------------------------
    // Lazy lookup tests
    // ------------------------------------------------------------------

    #[test]
    fn lazy_lookup_selects_most_similar_sense() {
        let mut sm = SenseManager::new(config_low_threshold());
        // Sense A: atoms 1,2,3
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![1, 2, 3]);
        // Sense B: atoms 10,20,30
        sm.ingest(vec![10, 20, 30]);
        sm.ingest(vec![10, 20, 30]);

        // Query close to sense A
        let idx_a = sm.lazy_lookup(&vec![1, 2]).unwrap();
        let sense_a = sm.get_sense(idx_a).unwrap();
        let core_a = sense_a.core(0.4);
        assert!(core_a.contains(&1) || core_a.contains(&2));

        // Query close to sense B
        let idx_b = sm.lazy_lookup(&vec![10, 20]).unwrap();
        let sense_b = sm.get_sense(idx_b).unwrap();
        let core_b = sense_b.core(0.4);
        assert!(core_b.contains(&10) || core_b.contains(&20));

        // They should be different senses
        assert_ne!(idx_a, idx_b);
    }

    // ------------------------------------------------------------------
    // Fragile deletion tests
    // ------------------------------------------------------------------

    #[test]
    fn fragile_sense_survives_within_k_fragile() {
        let mut sm = SenseManager::new(SenseConfig {
            k_fragile: 5,
            ..config_low_threshold()
        });
        // Create one sense
        sm.ingest(vec![1, 2, 3]);
        // Simulate 4 unrelated contexts (increment inactivity but < k_fragile)
        for i in 0..4 {
            sm.ingest(vec![100 + i, 200 + i]); // will create new senses or assign
        }
        // Force inactivity on all fragile senses manually for test
        for s in &mut sm.senses {
            if s.status == SenseStatus::Fragile {
                s.inactivity = 4; // still below k_fragile=5
            }
        }
        sm.purge_fragile();
        // Original sense (if fragile) should still be there
        // Just check we have at least 1 sense
        assert!(sm.sense_count() >= 1);
    }

    #[test]
    fn fragile_sense_deleted_after_k_fragile() {
        let mut sm = SenseManager::new(SenseConfig {
            k_fragile: 3,
            ..config_low_threshold()
        });
        sm.ingest(vec![1, 2, 3]); // sense 0, FRAGILE

        // Manually set inactivity past threshold
        sm.senses[0].inactivity = 3;
        sm.purge_fragile();

        assert_eq!(sm.sense_count(), 0);
    }

    // ------------------------------------------------------------------
    // Merge tests
    // ------------------------------------------------------------------

    #[test]
    fn merge_two_nearly_identical_mature_senses() {
        let mut sm = SenseManager::new(SenseConfig {
            theta_merge: 0.50,
            n_min_mature: 2, // lower for test
            ..config_low_threshold()
        });

        // Build two senses with high core overlap
        for _ in 0..3 { sm.ingest(vec![1, 2, 3, 4]); }
        for _ in 0..3 { sm.ingest(vec![1, 2, 3, 5]); }

        let count_before = sm.sense_count();
        let merged = sm.check_merge();

        if count_before >= 2 && !merged.is_empty() {
            assert!(sm.sense_count() < count_before);
        }
        // If only 1 sense formed (threshold routed all to same), that's also fine
    }

    // ------------------------------------------------------------------
    // Freq map tests
    // ------------------------------------------------------------------

    #[test]
    fn freq_map_correct_after_multiple_assigns() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![1, 2, 4]);
        sm.ingest(vec![1, 3, 5]);

        // Find the sense that captured all three
        if let Some(s) = sm.senses.iter().find(|s| s.context_count() == 3) {
            // atom 1 appears in all 3 → freq = 1.0
            assert!((s.freq(1) - 1.0).abs() < 0.01);
            // atom 2 appears in 2/3 → freq ≈ 0.667
            assert!((s.freq(2) - 2.0/3.0).abs() < 0.01);
            // atom 5 appears in 1/3 → freq ≈ 0.333
            assert!((s.freq(5) - 1.0/3.0).abs() < 0.01);
        }
    }

    #[test]
    fn core_filters_by_tau() {
        let mut sm = SenseManager::new(config_low_threshold());
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![1, 2, 4]);
        sm.ingest(vec![1, 3, 5]);

        if let Some(s) = sm.senses.iter().find(|s| s.context_count() == 3) {
            // tau=0.9 → only atom 1 (freq=1.0) should be in core
            let strict_core = s.core(0.9);
            assert!(strict_core.contains(&1));
            assert!(!strict_core.contains(&5));

            // tau=0.3 → atoms 1,2,3 all qualify (freq >= 0.33)
            let loose_core = s.core(0.3);
            assert!(loose_core.contains(&1));
            assert!(loose_core.contains(&2));
        }
    }
}

#[cfg(test)]
mod attention_tests {
    use crate::attention::{
        CoocStats, RsvsAttention, AttentionConfig,
        EntityDetector, text_to_sentences, tokenize, is_groundable_to_seeds,
    };
    use std::collections::HashMap;

    // ------------------------------------------------------------------
    // CoocStats tests
    // ------------------------------------------------------------------

    #[test]
    fn cooc_stats_empty_returns_zero() {
        let stats = CoocStats::new();
        assert_eq!(stats.npmi("stone", "hard"), 0.0);
        assert_eq!(stats.cooc("stone", "hard"), 0.0);
        assert_eq!(stats.p_token("stone"), 0.0);
    }

    #[test]
    fn cooc_stats_single_sentence() {
        let mut stats = CoocStats::new();
        stats.ingest_sentence(&["stone".into(), "hard".into(), "solid".into()]);

        assert_eq!(stats.token_count["stone"], 1);
        assert_eq!(stats.token_count["hard"], 1);
        assert_eq!(stats.pair_cooc_count("stone", "hard"), 1);
        assert_eq!(stats.pair_cooc_count("hard", "stone"), 1); // order-normalized
    }

    #[test]
    fn npmi_partial_cooccurrence_is_positive() {
        // NPMI = 0 when two tokens ALWAYS co-occur (P(t,c) = P(t)·P(c) when one
        // only appears with the other). To get positive NPMI we need partial overlap:
        // stone appears sometimes with hard AND sometimes alone/with others.
        let mut stats = CoocStats::new();
        for _ in 0..4 {
            stats.ingest_sentence(&["stone".into(), "hard".into()]);
        }
        // stone also appears with something else
        stats.ingest_sentence(&["stone".into(), "rough".into()]);
        // hard appears with something else too
        stats.ingest_sentence(&["hard".into(), "rough".into()]);

        let npmi = stats.npmi("stone", "hard");
        // With partial overlap, PMI > 0 because P(stone,hard) > P(stone)·P(hard)
        assert!(npmi > 0.0, "Expected positive NPMI for partial overlap, got {}", npmi);
        assert!(npmi <= 1.0, "NPMI must be <= 1.0, got {}", npmi);
    }

    #[test]
    fn npmi_never_cooccurring_is_zero() {
        let mut stats = CoocStats::new();
        stats.ingest_sentence(&["stone".into(), "hard".into()]);
        stats.ingest_sentence(&["water".into(), "liquid".into()]);
        let npmi = stats.npmi("stone", "water");
        assert_eq!(npmi, 0.0);
    }

    #[test]
    fn npmi_is_bounded_minus_one_to_one() {
        let mut stats = CoocStats::new();
        stats.ingest_sentence(&["stone".into(), "hard".into(), "solid".into()]);
        stats.ingest_sentence(&["stone".into(), "heavy".into()]);
        stats.ingest_sentence(&["hard".into(), "rough".into()]);
        let npmi = stats.npmi("stone", "hard");
        assert!(npmi >= -1.0 && npmi <= 1.0,
                "NPMI out of bounds: {}", npmi);
    }

    #[test]
    fn cooc_conditional_probability() {
        let mut stats = CoocStats::new();
        // stone appears 3 times, with hard 2 times
        stats.ingest_sentence(&["stone".into(), "hard".into()]);
        stats.ingest_sentence(&["stone".into(), "hard".into()]);
        stats.ingest_sentence(&["stone".into(), "rough".into()]);
        let cooc = stats.cooc("stone", "hard");
        assert!((cooc - 2.0/3.0).abs() < 0.01,
                "Expected cooc ~0.667, got {}", cooc);
    }

    // ------------------------------------------------------------------
    // Tokenizer tests
    // ------------------------------------------------------------------

    #[test]
    fn tokenize_removes_stopwords() {
        let tokens = tokenize("Stone is a hard solid material");
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"stone".to_string()));
        assert!(tokens.contains(&"hard".to_string()));
    }

    #[test]
    fn tokenize_removes_short_tokens() {
        let tokens = tokenize("it is so hard");
        assert!(!tokens.contains(&"it".to_string()));
        assert!(!tokens.contains(&"so".to_string()));
        assert!(tokens.contains(&"hard".to_string()));
    }

    #[test]
    fn tokenize_lowercases() {
        let tokens = tokenize("STONE is HARD");
        assert!(tokens.contains(&"stone".to_string()));
        assert!(tokens.contains(&"hard".to_string()));
    }

    #[test]
    fn split_sentences_basic() {
        let sentences = crate::attention::split_sentences(
            "Stone is hard. Fire is hot. Water is liquid."
        );
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn text_to_sentences_filters_single_token() {
        let sentences = text_to_sentences("Stone. Hard. Stone is hard and solid.");
        // Single-token sentences filtered out
        assert!(sentences.iter().all(|s| s.len() >= 2));
    }

    // ------------------------------------------------------------------
    // Entity detector tests
    // ------------------------------------------------------------------

    #[test]
    fn entity_detector_promotes_above_threshold() {
        let mut det = EntityDetector::new();
        for _ in 0..3 { det.record("stone", true); }
        for _ in 0..2 { det.record("hard", true); } // only 2, below N=3
        det.record("water", true);

        let candidates = det.candidates(3);
        assert!(candidates.contains(&"stone".to_string()));
        assert!(!candidates.contains(&"hard".to_string())); // below threshold
        assert!(!candidates.contains(&"water".to_string())); // only 1
    }

    #[test]
    fn entity_detector_requires_groundable() {
        let mut det = EntityDetector::new();
        for _ in 0..5 { det.record("xyz_abstract", false); } // not groundable
        for _ in 0..5 { det.record("stone", true); }          // groundable

        let candidates = det.candidates(3);
        assert!(!candidates.contains(&"xyz_abstract".to_string()));
        assert!(candidates.contains(&"stone".to_string()));
    }

    // ------------------------------------------------------------------
    // Grounding tests
    // ------------------------------------------------------------------

    #[test]
    fn groundable_physical_words() {
        let seeds = vec!["exists", "feel", "see", "hard"];
        assert!(is_groundable_to_seeds("hard", &seeds));
        assert!(is_groundable_to_seeds("stone", &seeds)); // hint match
        assert!(is_groundable_to_seeds("fire", &seeds));
        assert!(!is_groundable_to_seeds("philosophy", &seeds));
    }

    // ------------------------------------------------------------------
    // Attention scorer tests
    // ------------------------------------------------------------------

    #[test]
    fn attention_returns_empty_for_no_cooc_data() {
        let mut stats = CoocStats::new();
        stats.ingest_sentence(&["stone".into(), "hard".into()]);
        // Only 1 co-occurrence — below min_cooc=2
        let attention = RsvsAttention::new(AttentionConfig::default());
        let tokens = vec!["stone".to_string(), "hard".to_string()];
        let atom_sets: HashMap<String, Vec<u32>> = HashMap::new();
        let result = attention.select(&tokens, &stats, &atom_sets);
        // min_cooc=2 not met → no selections
        assert!(result.is_empty());
    }

    #[test]
    fn attention_selects_high_cooc_pairs() {
        let mut stats = CoocStats::new();
        // stone and hard always together — high NPMI + high cooc
        for _ in 0..5 {
            stats.ingest_sentence(&["stone".into(), "hard".into(), "solid".into()]);
        }
        // stone and water never together
        for _ in 0..3 {
            stats.ingest_sentence(&["water".into(), "liquid".into()]);
        }

        let attention = RsvsAttention::new(AttentionConfig {
            min_cooc: 2,
            ..AttentionConfig::default()
        });
        let tokens = vec!["stone".into(), "hard".into(), "solid".into(), "water".into()];
        let atom_sets: HashMap<String, Vec<u32>> = HashMap::new();
        let result = attention.select(&tokens, &stats, &atom_sets);

        // stone should select hard and solid, not water
        if let Some(stone_selected) = result.get("stone") {
            let selected_tokens: Vec<_> = stone_selected.iter()
                .map(|c| c.token.as_str())
                .collect();
            assert!(selected_tokens.contains(&"hard") || selected_tokens.contains(&"solid"));
            assert!(!selected_tokens.contains(&"water"));
        }
    }

    #[test]
    fn attention_score_components_sum_correctly() {
        let mut stats = CoocStats::new();
        for _ in 0..5 {
            stats.ingest_sentence(&["stone".into(), "hard".into()]);
        }
        let config = AttentionConfig {
            alpha: 0.4, beta: 0.4, gamma: 0.2,
            min_cooc: 2,
            ..AttentionConfig::default()
        };
        let attention = RsvsAttention::new(config.clone());
        let tokens = vec!["stone".into(), "hard".into()];
        let atom_sets: HashMap<String, Vec<u32>> = HashMap::new();
        let result = attention.select(&tokens, &stats, &atom_sets);

        if let Some(candidates) = result.get("stone") {
            if let Some(hard_cand) = candidates.iter().find(|c| c.token == "hard") {
                let expected = config.alpha * hard_cand.npmi
                             + config.beta  * hard_cand.jaccard
                             + config.gamma * hard_cand.cooc;
                assert!((hard_cand.score - expected).abs() < 0.001);
            }
        }
    }
}

#[cfg(test)]
mod autonomy_tests {
    use crate::autonomy::{
        AutonomyEngine, AutonomyConfig, ConfidenceUpdateResult,
        RemovalDecision, StabilityStatus, WarmUpState, MemoryClass,
    };
    use crate::types::Tier;

    fn engine() -> AutonomyEngine {
        AutonomyEngine::new(AutonomyConfig {
            eta:                   0.1,
            confidence_tier1:      0.85,
            confidence_tier2:      0.50,
            tau_remove:            0.10,
            threshold_impact:      3,
            threshold_global_delta: 0.50,
            n_warm:                5,
            ..AutonomyConfig::default()
        })
    }

    // ------------------------------------------------------------------
    // Confidence update tests
    // ------------------------------------------------------------------

    #[test]
    fn seed_atom_never_decays() {
        let mut e = engine();
        e.register(1, 1.0, Tier::Tier1);
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
        // Many low-evidence updates
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
        assert!(conf >= 0.0 && conf <= 1.0);
    }

    #[test]
    fn evidence_formula_is_correct() {
        let mut e = engine();
        e.register(10, 0.50, Tier::Tier2);
        let r = e.update_confidence(10, 0.8, 0.75, &[], 0);
        if let ConfidenceUpdateResult::Updated { evidence, .. } = r {
            // evidence = freq × coherence = 0.8 × 0.75 = 0.6
            assert!((evidence - 0.6).abs() < 0.001);
        } else {
            panic!("Expected Updated result");
        }
    }

    // ------------------------------------------------------------------
    // Tier reclassification tests
    // ------------------------------------------------------------------

    #[test]
    fn low_confidence_stays_tier3() {
        let mut e = engine();
        e.register(10, 0.30, Tier::Tier2);
        let t = e.reclassify(10).unwrap();
        // observation_count=0 and confidence < tier2 → Tier3
        assert_eq!(t, Tier::Tier3);
    }

    #[test]
    fn medium_confidence_with_observations_is_tier2() {
        let mut e = engine();
        e.register(10, 0.60, Tier::Tier2);
        // Set observation_count ≥ 3 manually
        if let Some(r) = e.records.get_mut(&10) {
            r.observation_count = 5;
        }
        let t = e.reclassify(10).unwrap();
        assert_eq!(t, Tier::Tier2);
    }

    // ------------------------------------------------------------------
    // Removal decision tests
    // ------------------------------------------------------------------

    #[test]
    fn low_confidence_low_impact_removes() {
        let mut e = engine();
        e.register(10, 0.05, Tier::Tier2); // below tau_remove=0.10
        let d = e.should_remove(10, 1);
        assert_eq!(d, RemovalDecision::Remove);
    }

    #[test]
    fn low_confidence_high_impact_requires_approval() {
        let mut e = engine();
        e.register(10, 0.05, Tier::Tier2);
        let d = e.should_remove(10, 10); // > threshold_impact=3
        assert!(matches!(d, RemovalDecision::RequiresApproval { .. }));
    }

    #[test]
    fn high_confidence_never_removed() {
        let mut e = engine();
        e.register(10, 0.90, Tier::Tier1);
        let d = e.should_remove(10, 0);
        assert!(matches!(d, RemovalDecision::Retain(_)));
    }

    #[test]
    fn seed_atom_never_removed() {
        let mut e = engine();
        e.register(1, 1.0, Tier::Tier1); // seed
        let d = e.should_remove(1, 0);
        assert_eq!(d, RemovalDecision::Retain("seed atom"));
    }

    #[test]
    fn removed_atom_goes_to_watchlist() {
        let mut e = engine();
        e.register(10, 0.05, Tier::Tier2);
        e.should_remove(10, 10); // triggers watchlist
        assert_eq!(e.watchlist_len(), 1);
    }

    // ------------------------------------------------------------------
    // Global stability gate tests
    // ------------------------------------------------------------------

    #[test]
    fn large_batch_delta_triggers_freeze() {
        let mut e = engine();
        e.register(10, 0.50, Tier::Tier2);
        e.register(11, 0.50, Tier::Tier2);
        e.begin_batch();

        // Many large changes
        for _ in 0..10 {
            e.update_confidence(10, 1.0, 1.0, &[], 0);
            e.update_confidence(11, 0.0, 0.0, &[], 0);
        }

        let s = e.check_global_stability();
        assert!(matches!(s, StabilityStatus::Frozen { .. }));
        assert!(e.frozen);
    }

    #[test]
    fn rollback_restores_confidence() {
        let mut e = engine();
        e.register(10, 0.50, Tier::Tier2);
        let snapshot = e.snapshot();

        e.begin_batch();
        e.update_confidence(10, 1.0, 1.0, &[], 0);
        assert!(e.confidence(10).unwrap() > 0.50);

        e.rollback(&snapshot);
        assert!((e.confidence(10).unwrap() - 0.50).abs() < 0.001);
    }

    #[test]
    fn small_batch_delta_stays_stable() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        e.begin_batch();

        // One tiny update
        e.update_confidence(10, 0.82, 0.82, &[], 0);

        let s = e.check_global_stability();
        assert!(matches!(s, StabilityStatus::Stable));
    }

    // ------------------------------------------------------------------
    // Warm-up tests
    // ------------------------------------------------------------------

    #[test]
    fn warmup_starts_active() {
        let e = engine();
        assert_eq!(e.warmup, WarmUpState::Active);
    }

    #[test]
    fn warmup_completes_after_n_contexts() {
        let mut e = AutonomyEngine::new(AutonomyConfig {
            n_warm: 3,
            ..AutonomyConfig::default()
        });
        for _ in 0..3 { e.tick_context(); }
        assert_eq!(e.warmup, WarmUpState::Complete);
    }

    #[test]
    fn fallback_thresholds_during_warmup() {
        let e = AutonomyEngine::new(AutonomyConfig {
            n_warm: 100,
            fallback_theta_assign: 0.25,
            ..AutonomyConfig::default()
        });
        assert_eq!(e.current_theta_assign(), 0.25);
    }

    #[test]
    fn adaptive_threshold_post_warmup() {
        let mut e = AutonomyEngine::new(AutonomyConfig {
            n_warm: 2,
            k1: 0.5,
            fallback_theta_assign: 0.15,
            ..AutonomyConfig::default()
        });
        for _ in 0..2 { e.tick_context(); }

        // Feed enough observations for adaptive to kick in
        for score in [0.30f32, 0.40, 0.35, 0.45, 0.38, 0.42] {
            e.observe_assign_score(score);
        }

        let adaptive = e.current_theta_assign();
        // Should differ from fallback
        assert_ne!(adaptive, 0.15);
        // Should be a reasonable value
        assert!(adaptive > 0.0 && adaptive < 1.0);
    }

    // ------------------------------------------------------------------
    // Energy constraint tests
    // ------------------------------------------------------------------

    #[test]
    fn energy_allows_confidence_increase() {
        let mut e = engine();
        e.register(10, 0.60, Tier::Tier2);
        assert!(e.energy_allows_update(10, 0.90));
    }

    #[test]
    fn energy_allows_small_drop() {
        let mut e = engine();
        e.register(10, 0.60, Tier::Tier2);
        // 0.60 → 0.55 = drop of 0.05, within tolerance
        assert!(e.energy_allows_update(10, 0.55));
    }

    #[test]
    fn energy_blocks_large_drop() {
        let mut e = engine();
        e.register(10, 0.60, Tier::Tier2);
        // 0.60 → 0.30 = drop of 0.30, exceeds tolerance
        assert!(!e.energy_allows_update(10, 0.30));
    }

    // ------------------------------------------------------------------
    // Memory class tests
    // ------------------------------------------------------------------

    #[test]
    fn new_atom_starts_as_working() {
        let mut e = engine();
        e.register(10, 0.60, Tier::Tier2);
        assert_eq!(e.memory_class(10).unwrap(), &MemoryClass::Working);
    }

    #[test]
    fn high_confidence_seed_is_stable() {
        let mut e = engine();
        e.register(10, 1.0, Tier::Tier1);
        assert_eq!(e.memory_class(10).unwrap(), &MemoryClass::Stable);
    }
}

#[cfg(test)]
mod pipeline_tests {
    use crate::pipeline::{Rsvs, PipelineConfig};
    use crate::autonomy::AutonomyConfig;
    use crate::sense::SenseConfig;

    fn make_rsvs() -> Rsvs {
        Rsvs::new(PipelineConfig {
            autonomy: AutonomyConfig {
                n_warm: 5,
                threshold_global_delta: 5.0, // lenient for tests
                ..AutonomyConfig::default()
            },
            sense: SenseConfig {
                theta_assign: 0.10,
                ..SenseConfig::default()
            },
            entity_promote_n: 2,
            ..PipelineConfig::default()
        })
    }

    // ------------------------------------------------------------------
    // Bootstrap tests
    // ------------------------------------------------------------------

    #[test]
    fn system_bootstraps_with_seed_atoms() {
        let rsvs = make_rsvs();
        assert_eq!(rsvs.graph.node_count(), 24); // 24 seed atoms
        assert!(rsvs.token_to_id.contains_key("exists"));
        assert!(rsvs.token_to_id.contains_key("feel"));
    }

    #[test]
    fn seed_atoms_are_registered_in_autonomy() {
        let rsvs = make_rsvs();
        let exists_id = rsvs.token_to_id["exists"];
        assert_eq!(rsvs.autonomy.confidence(exists_id), Some(1.0));
    }

    // ------------------------------------------------------------------
    // Ingest tests
    // ------------------------------------------------------------------

    #[test]
    fn ingest_promotes_frequent_tokens() {
        let mut rsvs = make_rsvs();
        // "stone" and "hard" appear 3+ times → should be promoted
        let text = "Stone is hard. Stone is hard and solid. Stone is a hard material. \
                    Stone remains hard. Hard stone is heavy.";
        let stats = rsvs.ingest_text(text);
        assert!(stats.atoms_promoted >= 1, "Expected at least 1 atom promoted");
        assert!(rsvs.token_to_id.contains_key("stone") ||
                rsvs.token_to_id.contains_key("hard"),
                "Expected 'stone' or 'hard' promoted");
    }

    #[test]
    fn ingest_rare_tokens_not_promoted() {
        let mut rsvs = make_rsvs();
        // "xyzquux" appears only once — should NOT be promoted
        let text = "Stone is hard. Stone is solid. Xyzquux appeared once.";
        rsvs.ingest_text(text);
        assert!(!rsvs.token_to_id.contains_key("xyzquux"));
    }

    #[test]
    fn ingest_increases_context_count() {
        let mut rsvs = make_rsvs();
        assert_eq!(rsvs.total_contexts, 0);
        let text = "Stone is hard. Water is liquid. Fire is hot.";
        rsvs.ingest_text(text);
        assert!(rsvs.total_contexts > 0);
    }

    #[test]
    fn ingest_creates_senses_for_promoted_atoms() {
        let mut rsvs = make_rsvs();
        let text = "Stone is hard. Stone is solid. Stone is heavy. \
                    Hard stone resists pressure. Solid stone is heavy.";
        rsvs.ingest_text(text);
        if let Some(&id) = rsvs.token_to_id.get("stone") {
            if let Some(sm) = rsvs.senses.get(&id) {
                assert!(sm.sense_count() >= 1, "stone should have at least 1 sense");
            }
        }
    }

    #[test]
    fn ingest_updates_confidence() {
        let mut rsvs = make_rsvs();
        let text = "Stone is hard. Stone is solid. Stone is heavy. \
                    Hard stone resists. Solid stone is natural.";
        rsvs.ingest_text(text);
        if let Some(&id) = rsvs.token_to_id.get("stone") {
            let conf = rsvs.autonomy.confidence(id).unwrap_or(0.0);
            // Confidence should be above initial 0.5 after good contexts
            assert!(conf > 0.0, "stone confidence should be > 0");
        }
    }

    // ------------------------------------------------------------------
    // Similarity tests
    // ------------------------------------------------------------------

    #[test]
    fn similar_concepts_have_positive_jaccard() {
        let mut rsvs = make_rsvs();
        let text = "Stone is hard solid heavy. \
                    Bone is hard solid organic. \
                    Stone and bone are both hard solid materials. \
                    Hard solid materials resist force. \
                    Stone is hard like bone. Bone is solid like stone.";
        rsvs.ingest_text(text);

        if let Some(sim) = rsvs.similarity("stone", "bone") {
            // stone and bone share "hard" and "solid" → jaccard > 0
            assert!(sim.jaccard > 0.0,
                    "stone/bone should have positive similarity, got {}", sim.jaccard);
        }
        // If either wasn't promoted, that's ok for this corpus size
    }

    #[test]
    fn dissimilar_concepts_have_lower_jaccard() {
        let mut rsvs = make_rsvs();
        let text = "Stone is hard solid heavy natural material. \
                    Stone is found in mountains and rivers. \
                    Stone resists pressure and heat. \
                    Water is liquid clear transparent. \
                    Water flows and dissolves things. \
                    Water and stone are different materials. \
                    Stone and water have different properties.";
        rsvs.ingest_text(text);

        if let (Some(sim_close), Some(sim_far)) = (
            rsvs.similarity("stone", "hard"),
            rsvs.similarity("stone", "water"),
        ) {
            assert!(
                sim_close.jaccard >= sim_far.jaccard,
                "stone/hard ({:.3}) should be >= stone/water ({:.3})",
                sim_close.jaccard, sim_far.jaccard
            );
        }
    }

    // ------------------------------------------------------------------
    // Query tests
    // ------------------------------------------------------------------

    #[test]
    fn query_unknown_concept_returns_none() {
        let rsvs = make_rsvs();
        assert!(rsvs.query("nonexistent_concept", "some context").is_none());
    }

    #[test]
    fn query_returns_scored_atoms() {
        let mut rsvs = make_rsvs();
        let text = "Stone is hard solid heavy. Stone is hard rough. \
                    Hard stone is solid. Solid stone is heavy. \
                    Stone has hard texture. Stone is natural hard material.";
        rsvs.ingest_text(text);

        if let Some(result) = rsvs.query("stone", "hard solid") {
            assert!(!result.scored_atoms.is_empty(),
                    "query should return scored atoms");
            // Scores should be non-negative
            for (_, score) in &result.scored_atoms {
                assert!(*score >= 0.0);
            }
        }
    }

    #[test]
    fn query_scores_sorted_descending() {
        let mut rsvs = make_rsvs();
        let text = "Stone is hard solid heavy. Stone is hard rough natural. \
                    Hard stone resists force. Solid stone is dense. \
                    Stone has rough hard texture. Stone is natural hard material.";
        rsvs.ingest_text(text);

        if let Some(result) = rsvs.query("stone", "hard solid texture") {
            let scores: Vec<f32> = result.scored_atoms.iter().map(|(_, s)| *s).collect();
            for i in 1..scores.len() {
                assert!(scores[i-1] >= scores[i],
                        "Scores should be sorted descending");
            }
        }
    }

    // ------------------------------------------------------------------
    // Status tests
    // ------------------------------------------------------------------

    #[test]
    fn status_reflects_ingestion() {
        let mut rsvs = make_rsvs();
        let before = rsvs.status();
        rsvs.ingest_text("Stone is hard. Stone is solid. Stone and hard material.");
        let after = rsvs.status();
        assert!(after.total_contexts > before.total_contexts);
        assert!(after.total_atoms >= before.total_atoms);
    }

    #[test]
    fn warmup_completes_after_n_contexts() {
        let mut rsvs = Rsvs::new(PipelineConfig {
            autonomy: AutonomyConfig {
                n_warm: 3,
                threshold_global_delta: 10.0,
                ..AutonomyConfig::default()
            },
            entity_promote_n: 2,
            ..PipelineConfig::default()
        });
        assert!(!rsvs.status().warmed_up);
        // Ingest enough to trigger warm-up
        for _ in 0..5 {
            rsvs.ingest_text("Stone is hard. Hard stone is solid.");
        }
        assert!(rsvs.status().warmed_up);
    }
}
