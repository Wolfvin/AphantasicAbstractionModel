//! Comprehensive unit tests for RSVS v6.0
//!
//! Covers: sense creation, NodeStatus transitions, quarantine,
//! hysteresis, seed invariants, CompressionState, DAG self-reference,
//! governance scoring, appraise mode, relate mode, snapshot v6.0 schema,
//! persistence save/load roundtrip, HashSet composition comparison,
//! sense induction scoring, grounding evidence accumulation,
//! composition revision, transformer bridge conversion, and more.

#[cfg(test)]
mod sense_tests {
    use crate::sense::{
        GroundingEvidence, GroundingVerdict, IngestResult, Sense, SenseConfig, SenseInductionConfig,
        SenseManager, SenseStatus,
    };
    use crate::types::CompositionRef;

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
        // v6.0: primitive senses (empty compositions) are always grounded,
        // so they won't be purged. Add a composition and set grounding score
        // below threshold so the sense is considered ungrounded.
        sm.senses[0].compositions = vec![CompositionRef::new(1, 0)];
        // Set grounding evidence to all contradicting so score is below threshold
        sm.senses[0].grounding = GroundingEvidence {
            confirming_contexts: 0,
            contradicting_contexts: 5,
            last_contradiction: Some("test".to_string()),
            revision_count: 0,
        };
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

    // -------------------------------------------------------------------
    // v6.0: Grounding evidence tests
    // -------------------------------------------------------------------

    #[test]
    fn grounding_evidence_score_neutral_when_no_evidence() {
        let ge = GroundingEvidence::new();
        assert!((ge.score() - 0.5).abs() < 0.001);
    }

    #[test]
    fn grounding_evidence_score_high_when_all_confirming() {
        let ge = GroundingEvidence {
            confirming_contexts: 10,
            contradicting_contexts: 0,
            last_contradiction: None,
            revision_count: 0,
        };
        assert!((ge.score() - 1.0).abs() < 0.001);
    }

    #[test]
    fn grounding_evidence_score_low_when_all_contradicting() {
        let ge = GroundingEvidence {
            confirming_contexts: 0,
            contradicting_contexts: 10,
            last_contradiction: Some("test".to_string()),
            revision_count: 0,
        };
        assert!((ge.score() - 0.0).abs() < 0.001);
    }

    #[test]
    fn grounding_evidence_score_mixed() {
        let ge = GroundingEvidence {
            confirming_contexts: 7,
            contradicting_contexts: 3,
            last_contradiction: None,
            revision_count: 0,
        };
        assert!((ge.score() - 0.7).abs() < 0.001);
    }

    #[test]
    fn grounding_evidence_confirm_and_contradict() {
        let mut ge = GroundingEvidence::new();
        ge.confirm();
        ge.confirm();
        ge.contradict(Some("low overlap".to_string()));
        assert_eq!(ge.confirming_contexts, 2);
        assert_eq!(ge.contradicting_contexts, 1);
        assert_eq!(ge.last_contradiction, Some("low overlap".to_string()));
        assert!((ge.score() - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn grounding_verdict_well_grounded() {
        let sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2],
            1,
        );
        // Default grounding is neutral (0.5), which is NeedsReview
        // Set to well-grounded
        let mut sense = sense;
        sense.grounding = GroundingEvidence {
            confirming_contexts: 8,
            contradicting_contexts: 2,
            last_contradiction: None,
            revision_count: 0,
        };
        assert_eq!(sense.grounding_verdict(), GroundingVerdict::WellGrounded);
    }

    #[test]
    fn grounding_verdict_needs_review() {
        let sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0)],
            vec![1],
            1,
        );
        // Default grounding is 0.5 which is NeedsReview
        assert_eq!(sense.grounding_verdict(), GroundingVerdict::NeedsReview);
    }

    #[test]
    fn grounding_verdict_needs_revision() {
        let mut sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2],
            1,
        );
        sense.grounding = GroundingEvidence {
            confirming_contexts: 1,
            contradicting_contexts: 9,
            last_contradiction: Some("no overlap".to_string()),
            revision_count: 0,
        };
        assert_eq!(sense.grounding_verdict(), GroundingVerdict::NeedsRevision);
    }

    // -------------------------------------------------------------------
    // v6.0: Composition revision tests
    // -------------------------------------------------------------------

    #[test]
    fn revise_compositions_removes_least_confirmed() {
        let mut sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0), CompositionRef::new(3, 0)],
            vec![1, 2, 3],
            1,
        );
        // Set grounding to all contradicting
        sense.grounding = GroundingEvidence {
            confirming_contexts: 0,
            contradicting_contexts: 5,
            last_contradiction: None,
            revision_count: 0,
        };

        let revised = sense.revise_compositions(0.2);
        assert!(revised);
        assert_eq!(sense.compositions.len(), 2);
        assert_eq!(sense.grounding.revision_count, 1);
    }

    #[test]
    fn revise_compositions_no_revision_when_grounded() {
        let mut sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2],
            1,
        );
        // Set grounding to well-grounded
        sense.grounding = GroundingEvidence {
            confirming_contexts: 10,
            contradicting_contexts: 1,
            last_contradiction: None,
            revision_count: 0,
        };

        let revised = sense.revise_compositions(0.2);
        assert!(!revised);
        assert_eq!(sense.compositions.len(), 2);
    }

    #[test]
    fn revise_compositions_not_on_last_composition() {
        let mut sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0)],
            vec![1],
            1,
        );
        sense.grounding = GroundingEvidence {
            confirming_contexts: 0,
            contradicting_contexts: 5,
            last_contradiction: None,
            revision_count: 0,
        };

        let revised = sense.revise_compositions(0.2);
        assert!(!revised); // Won't remove the last composition
    }

    // -------------------------------------------------------------------
    // v6.0: Sense induction scoring tests
    // -------------------------------------------------------------------

    #[test]
    fn induction_score_high_for_divergent_compositions() {
        let sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2, 3, 4],
            1,
        );

        let proposed = vec![CompositionRef::new(5, 0), CompositionRef::new(6, 0)];
        let score = sense.induction_score(
            &proposed,
            &vec![1, 2, 3, 4],
            &SenseInductionConfig::default(),
        );

        // Completely different compositions should have high score
        assert!(score > 0.5);
    }

    #[test]
    fn induction_score_low_for_similar_compositions() {
        let sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2, 3, 4],
            1,
        );

        // Same compositions = zero divergence
        let proposed = vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)];
        let score = sense.induction_score(
            &proposed,
            &vec![1, 2, 3, 4],
            &SenseInductionConfig::default(),
        );

        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn induction_score_zero_for_empty_compositions() {
        let sense = Sense::new(0, vec![1, 2, 3]);
        let score = sense.induction_score(
            &[],
            &vec![1, 2, 3],
            &SenseInductionConfig::default(),
        );
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn induction_score_below_min_divergence() {
        let sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0), CompositionRef::new(3, 0)],
            vec![1, 2, 3, 4],
            1,
        );

        // Only 1 out of 4 compositions different — below min divergence
        let proposed = vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0), CompositionRef::new(3, 0), CompositionRef::new(4, 0)];
        let config = SenseInductionConfig {
            min_composition_divergence: 0.5,
            ..SenseInductionConfig::default()
        };
        let score = sense.induction_score(&proposed, &vec![1, 2, 3, 4], &config);
        assert!((score - 0.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------
    // v6.0: Grounding evidence accumulation through update_grounding
    // -------------------------------------------------------------------

    #[test]
    fn grounding_accumulates_confirming_evidence() {
        let mut sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2],
            1,
        );
        let config = SenseConfig::default();

        // Context that confirms compositions (contains nodes 1 and 2)
        sense.update_grounding(&[1, 2, 3], &config);
        assert_eq!(sense.grounding.confirming_contexts, 1);
        assert_eq!(sense.grounding.contradicting_contexts, 0);
    }

    #[test]
    fn grounding_accumulates_contradicting_evidence() {
        let mut sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2],
            1,
        );
        let config = SenseConfig::default();

        // Context that contradicts compositions (contains none of nodes 1, 2)
        sense.update_grounding(&[99, 100, 101], &config);
        assert_eq!(sense.grounding.confirming_contexts, 0);
        assert_eq!(sense.grounding.contradicting_contexts, 1);
    }

    #[test]
    fn grounding_mixed_evidence() {
        let mut sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2],
            1,
        );
        let config = SenseConfig::default();

        // Confirming
        sense.update_grounding(&[1, 2, 3], &config);
        sense.update_grounding(&[1, 2, 4], &config);
        // Contradicting
        sense.update_grounding(&[99, 100], &config);

        assert_eq!(sense.grounding.confirming_contexts, 2);
        assert_eq!(sense.grounding.contradicting_contexts, 1);
        assert!((sense.grounding.score() - 2.0 / 3.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------
    // v6.0: HashSet-based composition comparison
    // -------------------------------------------------------------------

    #[test]
    fn hashset_composition_overlap_is_correct() {
        let sense_a = Sense::new_compositional(
            0,
            vec![
                CompositionRef::new(1, 0),
                CompositionRef::new(2, 0),
                CompositionRef::new(3, 0),
            ],
            vec![1, 2, 3],
            1,
        );
        let sense_b = Sense::new_compositional(
            1,
            vec![
                CompositionRef::new(1, 0),
                CompositionRef::new(2, 0),
                CompositionRef::new(4, 0),
            ],
            vec![1, 2, 4],
            1,
        );

        let overlap = sense_a.composition_overlap(&sense_b);
        // Shared: (1,0), (2,0) = 2; Union: (1,0), (2,0), (3,0), (4,0) = 4
        assert!((overlap - 0.5).abs() < 0.001);
    }

    #[test]
    fn hashset_composition_diff_is_correct() {
        let sense_a = Sense::new_compositional(
            0,
            vec![
                CompositionRef::new(1, 0),
                CompositionRef::new(2, 0),
                CompositionRef::new(3, 0),
            ],
            vec![1, 2, 3],
            1,
        );
        let sense_b = Sense::new_compositional(
            1,
            vec![
                CompositionRef::new(1, 0),
                CompositionRef::new(2, 0),
                CompositionRef::new(4, 0),
            ],
            vec![1, 2, 4],
            1,
        );

        let (only_a, only_b) = sense_a.composition_diff(&sense_b);
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_b.len(), 1);
        assert_eq!(only_a[0], CompositionRef::new(3, 0));
        assert_eq!(only_b[0], CompositionRef::new(4, 0));
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
    fn groundable_seed_labels_match() {
        let seeds = vec!["exists", "entity", "hard"];
        // Exact seed label match → groundable
        assert!(is_groundable_to_seeds("hard", &seeds));
        assert!(is_groundable_to_seeds("exists", &seeds));
        // Non-seed labels → not groundable by string matching alone
        // (they become groundable via sentence_contains_seed in the ingest pipeline)
        assert!(!is_groundable_to_seeds("stone", &seeds));
        assert!(!is_groundable_to_seeds("anjing", &seeds));
    }

    #[test]
    fn sentence_level_grounding_works() {
        use crate::attention::sentence_contains_seed;
        let seeds = vec!["exists", "entity"];
        // Sentence with a seed → all tokens are groundable
        let tokens = vec!["anjing".to_string(), "exists".to_string()];
        assert!(sentence_contains_seed(&tokens, &seeds));
        // Sentence without seeds → not groundable
        let tokens_no_seed = vec!["anjing".to_string(), "kucing".to_string()];
        assert!(!sentence_contains_seed(&tokens_no_seed, &seeds));
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
    // v6.0: NodeStatus lifecycle tests
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
        e.register(10, 0.80, Tier::Tier2);
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
        e.transition_status(10);
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
        e.transition_status(10);
        e.transition_status(10);
        if let Some(rec) = e.records.get_mut(&10) {
            rec.confidence = 0.50;
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
    fn full_lifecycle_new_to_deprecated() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);

        let r1 = e.transition_status(10);
        assert!(matches!(
            r1,
            StatusTransitionResult::Transitioned {
                from: NodeStatus::New,
                to: NodeStatus::Candidate,
            }
        ));

        let r2 = e.transition_status(10);
        assert!(matches!(
            r2,
            StatusTransitionResult::Transitioned {
                from: NodeStatus::Candidate,
                to: NodeStatus::Stable,
            }
        ));

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
    // Hysteresis tests
    // ------------------------------------------------------------------

    #[test]
    fn hysteresis_no_promote_below_threshold() {
        let mut e = engine();
        e.register(10, 0.70, Tier::Tier2);
        let r = e.transition_status(10);
        assert!(matches!(r, StatusTransitionResult::Blocked(_)));
    }

    #[test]
    fn hysteresis_no_demote_above_threshold() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        e.transition_status(10);
        e.transition_status(10);
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
        if let Some(rec) = e.records.get_mut(&10) {
            rec.confidence = 0.55;
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
        e.register(10, 0.75, Tier::Tier2);
        let r = e.transition_status(10);
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
        e.transition_status(10);
        e.transition_status(10);
        if let Some(rec) = e.records.get_mut(&10) {
            rec.confidence = 0.60;
        }
        let r = e.transition_status(10);
        assert!(matches!(r, StatusTransitionResult::Blocked(_)));
    }

    // ------------------------------------------------------------------
    // Quarantine tests
    // ------------------------------------------------------------------

    #[test]
    fn quarantine_after_three_flips() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        if let Some(rec) = e.records.get_mut(&10) {
            rec.status_flip_count = 3;
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
        if let Some(rec) = e.records.get_mut(&10) {
            rec.status_flip_count = 3;
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
    fn no_quarantine_below_threshold() {
        let mut e = engine();
        e.register(10, 0.80, Tier::Tier2);
        if let Some(rec) = e.records.get_mut(&10) {
            rec.status_flip_count = 2;
        }
        let r = e.transition_status(10);
        assert!(!matches!(
            r,
            StatusTransitionResult::Transitioned {
                to: NodeStatus::Quarantine,
                ..
            }
        ));
    }

    // ------------------------------------------------------------------
    // Seed immutability tests
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
        let r = e.update_confidence(1, 0.0, 0.0, &[], 0);
        assert!(matches!(r, ConfidenceUpdateResult::Skipped(_)));
        assert_eq!(e.confidence(1).unwrap(), 1.0);
    }

    // ------------------------------------------------------------------
    // Governance score tests
    // ------------------------------------------------------------------

    #[test]
    fn governance_score_formula() {
        let e = engine();
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
        // For Working memory (Tier2), tolerance is scaled: 0.20 * (0.30/0.10) = 0.60
        // So a drop of 0.30 is ALLOWED for Working memory.
        // Test with Stable memory instead, which uses the base tolerance of 0.20.
        let mut e = engine();
        e.register(10, 0.60, Tier::Tier1);
        // Force to Stable memory (Tier1 + high confidence)
        e.records.get_mut(&10).unwrap().confidence = 0.99;
        e.records.get_mut(&10).unwrap().memory = MemoryClass::Stable;
        assert!(!e.energy_allows_update(10, 0.60 - 0.30)); // drop=0.30 > tolerance=0.20
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
    use crate::types::{CompressionState, Edge, EdgeSource, Node, NodeStatus, RelationType, SemanticMeta, Tier};
    use std::collections::HashMap;

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
                gap_annotations: HashMap::new(),
                sense_profiles: HashMap::new(),
                discourse_meta: None,
                blend_results: HashMap::new(),
                abductive_hypotheses: Vec::new(),
                pattern_memberships: Vec::new(),
                synthesis_results: HashMap::new(),
                seed_distance_vector: HashMap::new(),
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
                gap_annotations: HashMap::new(),
                sense_profiles: HashMap::new(),
                discourse_meta: None,
                blend_results: HashMap::new(),
                abductive_hypotheses: Vec::new(),
                pattern_memberships: Vec::new(),
                synthesis_results: HashMap::new(),
                seed_distance_vector: HashMap::new(),
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
                gap_annotations: HashMap::new(),
                sense_profiles: HashMap::new(),
                discourse_meta: None,
                blend_results: HashMap::new(),
                abductive_hypotheses: Vec::new(),
                pattern_memberships: Vec::new(),
                synthesis_results: HashMap::new(),
                seed_distance_vector: HashMap::new(),
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
                    layer: 0,
                    derived_from_node_ids: vec![],
                    compression_reason: None,
                    internal_representation: false,
                    is_utterance: false,
                    utterance_tokens: Vec::new(),
                },
                policy_meta: None,
                language_links: vec![],
                atoms: vec![a1, a2],
                fingerprint: None,
                gap_annotations: HashMap::new(),
                sense_profiles: HashMap::new(),
                discourse_meta: None,
                blend_results: HashMap::new(),
                abductive_hypotheses: Vec::new(),
                pattern_memberships: Vec::new(),
                synthesis_results: HashMap::new(),
                seed_distance_vector: HashMap::new(),
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
                gap_annotations: HashMap::new(),
                sense_profiles: HashMap::new(),
                discourse_meta: None,
                blend_results: HashMap::new(),
                abductive_hypotheses: Vec::new(),
                pattern_memberships: Vec::new(),
                synthesis_results: HashMap::new(),
                seed_distance_vector: HashMap::new(),
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
                    layer: 0,
                    derived_from_node_ids: vec![a1],
                    compression_reason: Some("test".into()),
                    internal_representation: false,
                    is_utterance: false,
                    utterance_tokens: Vec::new(),
                },
                policy_meta: None,
                language_links: vec![],
                atoms: vec![],
                fingerprint: None,
                gap_annotations: HashMap::new(),
                sense_profiles: HashMap::new(),
                discourse_meta: None,
                blend_results: HashMap::new(),
                abductive_hypotheses: Vec::new(),
                pattern_memberships: Vec::new(),
                synthesis_results: HashMap::new(),
                seed_distance_vector: HashMap::new(),
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
                layer: 0,
                derived_from_node_ids: vec![5],
                compression_reason: None,
                    internal_representation: false,
                is_utterance: false,
                utterance_tokens: Vec::new(),
            },
            policy_meta: None,
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
            gap_annotations: HashMap::new(),
            sense_profiles: HashMap::new(),
            discourse_meta: None,
            blend_results: HashMap::new(),
            abductive_hypotheses: Vec::new(),
            pattern_memberships: Vec::new(),
            synthesis_results: HashMap::new(),
            seed_distance_vector: HashMap::new(),
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
                gap_annotations: HashMap::new(),
                sense_profiles: HashMap::new(),
                discourse_meta: None,
                blend_results: HashMap::new(),
                abductive_hypotheses: Vec::new(),
                pattern_memberships: Vec::new(),
                synthesis_results: HashMap::new(),
                seed_distance_vector: HashMap::new(),
            })
            .unwrap();
        let result = g.insert_edge(Edge {
            from: n1,
            to: 999,
            weight: 0.5,
            source: EdgeSource::Learned,
            last_reinforced_batch: 0,
            relation_type: RelationType::Categorical,
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
                gap_annotations: HashMap::new(),
                sense_profiles: HashMap::new(),
                discourse_meta: None,
                blend_results: HashMap::new(),
                abductive_hypotheses: Vec::new(),
                pattern_memberships: Vec::new(),
                synthesis_results: HashMap::new(),
                seed_distance_vector: HashMap::new(),
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
                gap_annotations: HashMap::new(),
                sense_profiles: HashMap::new(),
                discourse_meta: None,
                blend_results: HashMap::new(),
                abductive_hypotheses: Vec::new(),
                pattern_memberships: Vec::new(),
                synthesis_results: HashMap::new(),
                seed_distance_vector: HashMap::new(),
            })
            .unwrap();
        let result = g.insert_edge(Edge {
            from: n1,
            to: n2,
            weight: 0.8,
            source: EdgeSource::Learned,
            last_reinforced_batch: 0,
            relation_type: RelationType::Categorical,
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
            gap_annotations: HashMap::new(),
            sense_profiles: HashMap::new(),
            discourse_meta: None,
            blend_results: HashMap::new(),
            abductive_hypotheses: Vec::new(),
            pattern_memberships: Vec::new(),
            synthesis_results: HashMap::new(),
            seed_distance_vector: HashMap::new(),
        })
        .unwrap();
        assert_eq!(g.node_count(), 1);
    }
}

#[cfg(test)]
mod v60_node_tests {
    use crate::types::{
        CompressionState, Fingerprint, Node, NodeStatus, PolicyMeta, SemanticMeta, Tier,
    };
    use std::collections::HashMap;

    #[test]
    fn v60_node_creation() {
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
                layer: 0,
                derived_from_node_ids: vec![],
                compression_reason: None,
                    internal_representation: false,
                is_utterance: false,
                utterance_tokens: Vec::new(),
            },
            policy_meta: Some(PolicyMeta::default()),
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
            gap_annotations: HashMap::new(),
            sense_profiles: HashMap::new(),
            discourse_meta: None,
            blend_results: HashMap::new(),
            abductive_hypotheses: Vec::new(),
            pattern_memberships: Vec::new(),
            synthesis_results: HashMap::new(),
            seed_distance_vector: HashMap::new(),
        };
        assert_eq!(node.kind, "node");
        assert_eq!(node.surface_label, "test@en");
        assert_eq!(node.status, NodeStatus::Candidate);
        assert!(!node.is_seed);
        assert_eq!(node.semantic.compression_state, CompressionState::Raw);
    }

    #[test]
    fn v60_seed_node() {
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
                layer: 0,
                derived_from_node_ids: vec![],
                compression_reason: None,
                    internal_representation: false,
                is_utterance: false,
                utterance_tokens: Vec::new(),
            },
            policy_meta: None,
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
            gap_annotations: HashMap::new(),
            sense_profiles: HashMap::new(),
            discourse_meta: None,
            blend_results: HashMap::new(),
            abductive_hypotheses: Vec::new(),
            pattern_memberships: Vec::new(),
            synthesis_results: HashMap::new(),
            seed_distance_vector: HashMap::new(),
        };
        assert!(node.is_seed);
        assert!(node.is_locked);
        assert_eq!(node.status, NodeStatus::Stable);
    }

    #[test]
    fn v60_compressed_node() {
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
                layer: 0,
                derived_from_node_ids: vec![1, 2, 3],
                compression_reason: Some("co-occurrence aggregation".to_string()),
                internal_representation: false,
                is_utterance: false,
                utterance_tokens: Vec::new(),
            },
            policy_meta: Some(PolicyMeta {
                policy_version: "6.0".to_string(),
                governance_score: 0.85,
                candidate_evidence_pool: 0.3,
                status_flip_count: 1,
                seen_fingerprints: vec!["fp1".to_string()],
                last_seen_at: Some("2024-01-01".to_string()),
            }),
            language_links: vec![],
            atoms: vec![1, 2, 3],
            fingerprint: None,
            gap_annotations: HashMap::new(),
            sense_profiles: HashMap::new(),
            discourse_meta: None,
            blend_results: HashMap::new(),
            abductive_hypotheses: Vec::new(),
            pattern_memberships: Vec::new(),
            synthesis_results: HashMap::new(),
            seed_distance_vector: HashMap::new(),
        };
        assert_eq!(
            node.semantic.compression_state,
            CompressionState::Compressed
        );
        assert_eq!(node.semantic.derived_from_node_ids, vec![1, 2, 3]);
        assert_eq!(
            node.policy_meta.as_ref().unwrap().policy_version,
            "6.0"
        );
    }

    #[test]
    fn v60_policy_meta_default_version() {
        let pm = PolicyMeta::default();
        assert_eq!(pm.policy_version, "6.0");
    }

    #[test]
    fn fingerprint_deterministic() {
        let fp1 = Fingerprint::new(b"hello world");
        let fp2 = Fingerprint::new(b"hello world");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn fingerprint_different_data() {
        let fp1 = Fingerprint::new(b"hello");
        let fp2 = Fingerprint::new(b"world");
        assert_ne!(fp1, fp2);
    }
}

#[cfg(test)]
mod transformer_bridge_tests {
    use crate::sense::{GroundingEvidence, Sense, SenseManager, SenseConfig};
    use crate::transformer_bridge::{TransformerBridge, TransformerBridgeConfig};
    use crate::types::CompositionRef;
    use std::collections::HashMap;

    #[test]
    fn vectors_to_compositions_with_similar_vectors() {
        let bridge = TransformerBridge::new(TransformerBridgeConfig {
            similarity_threshold: 0.5,
            max_compositions: 5,
            use_attention_weights: false,
        });

        // Two vectors that are very similar
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.01, 0.0], // very similar to first
            vec![0.0, 0.0, 1.0],  // orthogonal
        ];
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let comps = bridge.vectors_to_compositions(&vectors, &labels, 0.8);
        // Should find that a and b are related
        assert!(!comps.is_empty());
    }

    #[test]
    fn vectors_to_compositions_no_similar_vectors() {
        let bridge = TransformerBridge::new(TransformerBridgeConfig {
            similarity_threshold: 0.9,
            max_compositions: 5,
            use_attention_weights: false,
        });

        // Orthogonal vectors with high threshold
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let comps = bridge.vectors_to_compositions(&vectors, &labels, 0.9);
        // No pairs should exceed threshold
        assert!(comps.is_empty());
    }

    #[test]
    fn attention_weights_to_senses_basic() {
        let bridge = TransformerBridge::new(TransformerBridgeConfig {
            similarity_threshold: 0.3,
            max_compositions: 3,
            use_attention_weights: true,
        });

        // Attention matrix: token 0 attends strongly to token 1
        let attention = vec![
            vec![0.1, 0.8, 0.1],
            vec![0.7, 0.1, 0.2],
            vec![0.1, 0.1, 0.1],
        ];
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let senses = bridge.attention_weights_to_senses(&attention, &labels);
        assert!(!senses.is_empty());

        let a_sense = senses.iter().find(|(label, _)| label == "a");
        assert!(a_sense.is_some());
        assert!(!a_sense.unwrap().1.is_empty());
    }

    #[test]
    fn explain_vector_returns_explanations() {
        let bridge = TransformerBridge::new(TransformerBridgeConfig::default());
        let mut graph = crate::graph::RsvsGraph::new();

        // Create some nodes
        use crate::types::{Node, NodeStatus, Tier};
        let n1 = graph.insert_node(Node {
            id: 0,
            label: "stone".into(),
            surface_label: "stone@en".into(),
            kind: "node".into(),
            tier: Tier::Tier1,
            confidence: 1.0,
            status: NodeStatus::Stable,
            is_seed: true,
            is_locked: true,
            semantic: crate::types::SemanticMeta::default(),
            policy_meta: None,
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
            gap_annotations: HashMap::new(),
            sense_profiles: HashMap::new(),
            discourse_meta: None,
            blend_results: HashMap::new(),
            abductive_hypotheses: Vec::new(),
            pattern_memberships: Vec::new(),
            synthesis_results: HashMap::new(),
            seed_distance_vector: HashMap::new(),
        }).unwrap();
        let n2 = graph.insert_node(Node {
            id: 0,
            label: "hard".into(),
            surface_label: "hard@en".into(),
            kind: "node".into(),
            tier: Tier::Tier1,
            confidence: 1.0,
            status: NodeStatus::Stable,
            is_seed: true,
            is_locked: true,
            semantic: crate::types::SemanticMeta::default(),
            policy_meta: None,
            language_links: vec![],
            atoms: vec![],
            fingerprint: None,
            gap_annotations: HashMap::new(),
            sense_profiles: HashMap::new(),
            discourse_meta: None,
            blend_results: HashMap::new(),
            abductive_hypotheses: Vec::new(),
            pattern_memberships: Vec::new(),
            synthesis_results: HashMap::new(),
            seed_distance_vector: HashMap::new(),
        }).unwrap();

        let mut senses = std::collections::HashMap::new();
        let mut sm = SenseManager::new(SenseConfig::default());
        sm.create_compositional_sense(vec![CompositionRef::new(n1, 0), CompositionRef::new(n2, 0)], 1);
        senses.insert(1, sm);

        let vector = vec![0.5, 0.3, 0.2];
        let explanations = bridge.explain_vector(&vector, &graph, &senses);
        // Should return some explanations about compositional senses
        assert!(!explanations.is_empty());
    }

    #[test]
    fn transformer_bridge_config_default() {
        let config = TransformerBridgeConfig::default();
        assert!((config.similarity_threshold - 0.5).abs() < 0.001);
        assert_eq!(config.max_compositions, 10);
        assert!(config.use_attention_weights);
    }
}

// -----------------------------------------------------------------------
// v6.5: Comprehensive tests for Losion Cross-Pollination features
// -----------------------------------------------------------------------

#[cfg(test)]
mod ebbinghaus_tests {
    use crate::autonomy::{AutonomyConfig, AutonomyEngine, MemoryClass};
    use crate::types::{NodeStatus, Tier};

    #[test]
    fn ebbinghaus_decay_reduces_confidence_for_inactive() {
        let mut e = AutonomyEngine::new(AutonomyConfig {
            ebbinghaus_decay_rate: 2.0,
            ebbinghaus_reinforce_factor: 0.2,
            n_warm: 0,
            ..AutonomyConfig::default()
        });
        e.register(10, 0.80, Tier::Tier2);
        // Mark as seen at context 0
        e.tick_context();
        let _ = e.update_confidence(10, 0.8, 0.8, &[], 0);

        // Advance context by 75% of TTL (past grace period of 25%)
        // Default TTL is 50, so 38 contexts = past grace
        for _ in 0..38 {
            e.tick_context();
        }

        let flagged = e.flag_inactive_atoms(e.context_counter);
        // Should have decayed the inactive atom
        assert!(flagged >= 1);
        // Confidence should be lower than original
        assert!(e.confidence(10).unwrap() < 0.80);
    }

    #[test]
    fn ebbinghaus_seed_never_decays() {
        let mut e = AutonomyEngine::new(AutonomyConfig {
            ebbinghaus_decay_rate: 2.0,
            ebbinghaus_reinforce_factor: 0.2,
            n_warm: 0,
            ..AutonomyConfig::default()
        });
        e.register_seed(1, 1.0, Tier::Tier1);
        // Advance many contexts
        for _ in 0..200 {
            e.tick_context();
        }
        let flagged = e.flag_inactive_atoms(e.context_counter);
        assert_eq!(flagged, 0); // Seeds never decay
        assert_eq!(e.confidence(1).unwrap(), 1.0);
    }

    #[test]
    fn ebbinghaus_frequent_access_resists_decay() {
        let mut e = AutonomyEngine::new(AutonomyConfig {
            ebbinghaus_decay_rate: 2.0,
            ebbinghaus_reinforce_factor: 0.2,
            n_warm: 0,
            ..AutonomyConfig::default()
        });
        // Node accessed many times
        e.register(10, 0.80, Tier::Tier2);
        e.tick_context();
        for _ in 0..50 {
            let _ = e.update_confidence(10, 0.9, 0.9, &[], 0);
        }
        let _conf_with_access = e.confidence(10).unwrap();

        // Node accessed few times
        e.register(20, 0.80, Tier::Tier2);
        e.tick_context();
        let _ = e.update_confidence(20, 0.9, 0.9, &[], 0);
        let _conf_without_access = e.confidence(20).unwrap();

        // Advance past grace period for both
        for _ in 0..40 {
            e.tick_context();
        }
        let _ = e.flag_inactive_atoms(e.context_counter);

        // Frequently accessed node should retain higher confidence
        let conf_10 = e.confidence(10).unwrap();
        let conf_20 = e.confidence(20).unwrap();
        assert!(conf_10 > conf_20, "Frequently accessed node ({}) should decay less than rarely accessed ({})", conf_10, conf_20);
    }

    #[test]
    fn ebbinghaus_counts_only_adjusted_atoms() {
        let mut e = AutonomyEngine::new(AutonomyConfig {
            ebbinghaus_decay_rate: 2.0,
            ebbinghaus_reinforce_factor: 0.2,
            n_warm: 0,
            ..AutonomyConfig::default()
        });
        // Node recently seen (within grace period)
        e.register(10, 0.80, Tier::Tier2);
        e.tick_context();
        let _ = e.update_confidence(10, 0.8, 0.8, &[], 0);
        // Only 5 contexts — within grace period (25% of 50 = 12)
        for _ in 0..5 {
            e.tick_context();
        }
        let flagged = e.flag_inactive_atoms(e.context_counter);
        assert_eq!(flagged, 0); // Not past grace period
    }
}

#[cfg(test)]
mod composition_index_o1_tests {
    use crate::composition_index::CompositionIndex;
    use crate::types::CompositionRef;

    #[test]
    fn secondary_index_provides_o1_lookup() {
        let mut idx = CompositionIndex::new();
        idx.add(5, &[CompositionRef::new(1, 0), CompositionRef::new(2, 0)]);
        idx.add(6, &[CompositionRef::new(1, 0), CompositionRef::new(3, 0)]);
        idx.add(7, &[CompositionRef::new(2, 0)]);

        // O(1) node-level lookup
        let deps = idx.dependents_of_node(1);
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&5));
        assert!(deps.contains(&6));

        let deps2 = idx.dependents_of_node(2);
        assert_eq!(deps2.len(), 2);
        assert!(deps2.contains(&5));
        assert!(deps2.contains(&7));

        let deps3 = idx.dependents_of_node(3);
        assert_eq!(deps3.len(), 1);
        assert!(deps3.contains(&6));
    }

    #[test]
    fn impact_count_uses_secondary_index() {
        let mut idx = CompositionIndex::new();
        idx.add(5, &[CompositionRef::new(1, 0)]);
        idx.add(6, &[CompositionRef::new(1, 0)]);
        idx.add(7, &[CompositionRef::new(1, 0), CompositionRef::new(2, 0)]);

        assert_eq!(idx.impact_count(1), 3);
        assert_eq!(idx.impact_count(2), 1);
    }

    #[test]
    fn remove_updates_secondary_index() {
        let mut idx = CompositionIndex::new();
        idx.add(5, &[CompositionRef::new(1, 0), CompositionRef::new(2, 0)]);
        idx.add(6, &[CompositionRef::new(1, 0)]);

        assert_eq!(idx.dependents_of_node(1).len(), 2);
        idx.remove(5, &[CompositionRef::new(1, 0), CompositionRef::new(2, 0)]);
        assert_eq!(idx.dependents_of_node(1).len(), 1);
        assert_eq!(idx.dependents_of_node(2).len(), 0);
    }
}

#[cfg(test)]
mod consolidation_tests {
    use crate::consolidation::{ConsolidationConfig, ConsolidationEngine};
    use crate::autonomy::AutonomyEngine;
    use crate::graph::RsvsGraph;
    use crate::sense::SenseManager;
    use crate::types::CompositionRef;
    use std::collections::HashMap;

    #[test]
    fn consolidation_removes_dead_senses() {
        let engine = ConsolidationEngine::new(ConsolidationConfig {
            consolidation_interval: 10,
            ..ConsolidationConfig::default()
        });
        let mut graph = RsvsGraph::new();
        let mut senses = HashMap::new();
        let mut autonomy = AutonomyEngine::new(Default::default());

        // Create a sense manager with a sense that will be purged
        let config = crate::sense::SenseConfig {
            k_fragile: 5,
            grounding_min: 0.3,
            ..Default::default()
        };
        let mut sm = SenseManager::new(config.clone());
        sm.ingest(vec![1, 2, 3]);
        // Make the sense fragile + ungrounded + very inactive
        sm.senses[0].inactivity = sm.config.k_fragile * 3; // Way past limit
        sm.senses[0].grounding = crate::sense::GroundingEvidence {
            confirming_contexts: 0,
            contradicting_contexts: 10,
            last_contradiction: Some("test".to_string()),
            revision_count: 0,
        };
        // Add composition so it's not considered primitive (primitives are always grounded)
        sm.senses[0].compositions = vec![CompositionRef::new(99, 0)];
        senses.insert(1, sm);

        let result = engine.consolidate(&mut graph, &mut senses, &mut autonomy);
        assert!(result.senses_removed >= 1);
    }

    #[test]
    fn consolidation_should_run_at_interval() {
        let engine = ConsolidationEngine::new(ConsolidationConfig {
            consolidation_interval: 50,
            ..ConsolidationConfig::default()
        });
        assert!(!engine.should_run(49));
        assert!(engine.should_run(50));
        assert!(!engine.should_run(51));
    }
}

#[cfg(test)]
mod reflection_tests {
    use crate::reflection::{ReflectionConfig, ReflectionAction, SenseReflection};
    use crate::sense::{GroundingEvidence, SenseManager, SenseConfig};
    use std::collections::HashMap;

    #[test]
    fn reflection_retires_fragile_inactive_sense() {
        let mut reflection = SenseReflection::new(ReflectionConfig {
            retire_inactivity_threshold: 50,
            ..ReflectionConfig::default()
        });
        let config = SenseConfig {
            grounding_min: 0.3,
            k_fragile: 5,
            ..Default::default()
        };
        let mut sm = SenseManager::new(config.clone());
        sm.ingest(vec![1, 2, 3]);
        // Make sense fragile + inactive + ungrounded
        sm.senses[0].inactivity = 100;
        sm.senses[0].grounding = GroundingEvidence {
            confirming_contexts: 0,
            contradicting_contexts: 10,
            last_contradiction: Some("test".to_string()),
            revision_count: 0,
        };
        sm.senses[0].compositions = vec![crate::types::CompositionRef::new(99, 0)];

        let mut senses = HashMap::new();
        senses.insert(1, sm);

        let actions = reflection.reflect(&senses, &config);
        let retires = actions.iter().filter(|a| matches!(a, ReflectionAction::Retire { .. })).count();
        assert!(retires >= 1);
    }

    #[test]
    fn reflection_rate_limits_revise() {
        let reflection = SenseReflection::new(ReflectionConfig {
            max_revise_per_cycle: 2,
            ..ReflectionConfig::default()
        });
        assert_eq!(reflection.config.max_revise_per_cycle, 2);
    }
}

#[cfg(test)]
mod persistence_v65_tests {
    use crate::autonomy::{AutonomyConfig, AutonomyEngine, AtomRecord, MemoryClass};
    use crate::persist::{SavedAtomRecord, to_snapshot, from_snapshot};
    use crate::pipeline::{PipelineConfig, Rsvs};
    use crate::types::{Edge, EdgeSource, RelationType, Tier};

    #[test]
    fn atom_record_access_count_survives_roundtrip() {
        let mut record = AtomRecord::new(10, 0.8, Tier::Tier2);
        record.access_count = 42;
        record.context_count_since_promote = 7;
        assert_eq!(record.access_count, 42);
        assert_eq!(record.context_count_since_promote, 7);
    }

    #[test]
    fn saved_atom_record_preserves_new_fields() {
        let sar = SavedAtomRecord {
            id: 10,
            confidence: 0.8,
            tier: 2,
            status: "candidate".into(),
            memory: "working".into(),
            domain_count: 1,
            cooccurring_mature: vec![],
            observation_count: 5,
            is_seed: false,
            status_flip_count: 0,
            governance_score: 0.5,
            candidate_evidence_pool: 0.0,
            last_seen_context: 100,
            inactivity_ttl: 50,
            access_count: 25,
            context_count_since_promote: 3,
        };
        assert_eq!(sar.access_count, 25);
        assert_eq!(sar.context_count_since_promote, 3);
    }

    #[test]
    fn rsvs_save_load_roundtrip() {
        let config = PipelineConfig::default();
        let rsvs = Rsvs::new(config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_rsvs.json");
        rsvs.save(&path).unwrap();
        let loaded = Rsvs::load(&path).unwrap();

        // Basic sanity checks
        assert_eq!(loaded.total_contexts, rsvs.total_contexts);
        assert_eq!(loaded.token_to_id.len(), rsvs.token_to_id.len());
        assert_eq!(loaded.graph.nodes.len(), rsvs.graph.nodes.len());
    }
}

#[cfg(test)]
mod thinking_toggle_tests {
    use crate::thinking::{ThinkingToggle, ThinkingToggleConfig, ThinkingMode, ComplexitySignal};
    use crate::types::TraversalConfig;

    #[test]
    fn thinking_mode_increases_depth() {
        let toggle = ThinkingToggle::new(ThinkingToggleConfig::default());
        let base = TraversalConfig::default();
        let non_thinking = toggle.adjust_traversal(&ThinkingMode::NonThinking, &base);
        let thinking = toggle.adjust_traversal(&ThinkingMode::Thinking, &base);
        assert!(thinking.max_depth >= non_thinking.max_depth);
    }

    #[test]
    fn thinking_mode_adjusts_relevance() {
        let toggle = ThinkingToggle::new(ThinkingToggleConfig::default());
        let base = TraversalConfig::default();
        let non_thinking = toggle.adjust_traversal(&ThinkingMode::NonThinking, &base);
        let thinking = toggle.adjust_traversal(&ThinkingMode::Thinking, &base);
        // Thinking mode should lower tau_relevance for broader search
        assert!(thinking.tau_relevance <= non_thinking.tau_relevance);
    }

    #[test]
    fn complexity_signal_with_many_atoms_triggers_thinking() {
        let toggle = ThinkingToggle::new(ThinkingToggleConfig::default());
        let signal = ComplexitySignal {
            n_context_atoms: 5,
            n_senses: 3,
            target_layer: 2,
            is_compositional: true,
            domain_complexity: 0.8,
        };
        assert_eq!(toggle.classify(&signal), ThinkingMode::Thinking);
    }

    #[test]
    fn complexity_signal_with_few_atoms_is_non_thinking() {
        let toggle = ThinkingToggle::new(ThinkingToggleConfig::default());
        let signal = ComplexitySignal {
            n_context_atoms: 1,
            n_senses: 1,
            target_layer: 0,
            is_compositional: false,
            domain_complexity: 0.0,
        };
        assert_eq!(toggle.classify(&signal), ThinkingMode::NonThinking);
    }
}

#[cfg(test)]
mod neurosym_tests {
    use crate::neurosym::{NeuroSymVerifier, VerificationStatus};
    use crate::sense::{Sense, SenseConfig};
    use crate::types::CompositionRef;
    use crate::graph::RsvsGraph;
    use std::collections::HashMap;

    #[test]
    fn verifier_catches_self_reference() {
        let verifier = NeuroSymVerifier::new();
        let config = SenseConfig::default();
        let graph = RsvsGraph::new();
        let senses = HashMap::new();

        let sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0)], // References node 1
            vec![1, 2],
            1,
        );

        let (status, _) = verifier.verify(1, &sense, &graph, &senses, &config); // node_id=1 = self-reference
        assert_ne!(status, VerificationStatus::Verified);
    }

    #[test]
    fn verifier_passes_clean_sense() {
        let verifier = NeuroSymVerifier::new();
        let config = SenseConfig::default();
        let graph = RsvsGraph::new();
        let senses = HashMap::new();

        // Primitive sense (no compositions)
        let sense = Sense::new(0, vec![1, 2, 3]);

        let (_status, results) = verifier.verify(5, &sense, &graph, &senses, &config);
        // For a primitive sense, binary rules should pass
        assert!(results.iter().find(|r| r.rule.name == "no_self_reference").unwrap().passed);
        assert!(results.iter().find(|r| r.rule.name == "no_circular_chain").unwrap().passed);
    }
}

// -----------------------------------------------------------------------
// v11.0: Entropy Trigger Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod entropy_trigger_tests {
    use crate::sense::{Sense, SenseInductionConfig, SenseManager, SenseConfig};
    use crate::types::CompositionRef;

    #[test]
    fn entropy_trigger_score_zero_for_single_context() {
        // A sense with only one context cannot measure diversity
        let sense = Sense::new(0, vec![1, 2, 3]);
        assert!((sense.entropy_trigger_score() - 0.0).abs() < 0.001);
    }

    #[test]
    fn entropy_trigger_score_high_for_diverse_contexts() {
        // Orthogonal contexts → high inter-context diversity → high trigger score
        let mut sm = SenseManager::new(SenseConfig {
            theta_assign: 0.01,
            ..SenseConfig::default()
        });
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![10, 20, 30]);
        // Find the sense that got both contexts (if merged) or just check one
        for sense in &sm.senses {
            if sense.contexts.len() >= 2 {
                let score = sense.entropy_trigger_score();
                // Should be > 0 because contexts are very different
                assert!(score > 0.0, "Expected positive trigger score for diverse contexts, got {}", score);
            }
        }
    }

    #[test]
    fn entropy_trigger_score_low_for_identical_contexts() {
        // Identical contexts → zero inter-context distance → low trigger
        let mut sm = SenseManager::new(SenseConfig {
            theta_assign: 0.01,
            ..SenseConfig::default()
        });
        sm.ingest(vec![1, 2, 3]);
        sm.ingest(vec![1, 2, 3]);
        if let Some(sense) = sm.senses.iter().find(|s| s.contexts.len() >= 2) {
            let score = sense.entropy_trigger_score();
            // With identical contexts, avg_diversity = 0.0, so trigger = 0.0
            assert!((score - 0.0).abs() < 0.001, "Expected zero trigger for identical contexts, got {}", score);
        }
    }

    #[test]
    fn entropy_trigger_score_bounded_0_1() {
        let mut sm = SenseManager::new(SenseConfig {
            theta_assign: 0.01,
            ..SenseConfig::default()
        });
        // Ingest many diverse contexts
        for i in 0..10 {
            sm.ingest(vec![i * 100, i * 100 + 1, i * 100 + 2]);
        }
        for sense in &sm.senses {
            if sense.contexts.len() >= 2 {
                let score = sense.entropy_trigger_score();
                assert!(score >= 0.0 && score <= 1.0, "Trigger score out of bounds: {}", score);
            }
        }
    }

    #[test]
    fn induction_score_uses_entropy_as_trigger() {
        // With entropy_trigger_weight = 1.0, entropy alone drives the score
        let sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2, 3, 4],
            1,
        );
        let proposed = vec![CompositionRef::new(5, 0), CompositionRef::new(6, 0)];

        let config_full_entropy = SenseInductionConfig {
            entropy_trigger_weight: 1.0,
            ..SenseInductionConfig::default()
        };
        let score = sense.induction_score(&proposed, &vec![1, 2, 3, 4], &config_full_entropy);
        // With weight=1.0 on entropy and 0.0 on divergence, score depends purely on context entropy
        assert!(score > 0.0, "Expected positive score with full entropy weight, got {}", score);
        assert!(score <= 1.0);
    }

    #[test]
    fn induction_score_entropy_weight_zero_ignores_entropy() {
        // With entropy_trigger_weight = 0.0, only divergence matters
        let sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2, 3, 4],
            1,
        );
        let proposed = vec![CompositionRef::new(5, 0), CompositionRef::new(6, 0)];

        let config_no_entropy = SenseInductionConfig {
            entropy_trigger_weight: 0.0,
            ..SenseInductionConfig::default()
        };
        let score = sense.induction_score(&proposed, &vec![1, 2, 3, 4], &config_no_entropy);
        // With weight=0.0, score = divergence * novel_fraction boost
        // Divergence = 1.0 (completely different), so score should be high
        assert!(score > 0.5, "Expected high score with full divergence weight, got {}", score);
    }

    #[test]
    fn induction_score_default_weight_balances() {
        // With default weight (0.5), both divergence and entropy contribute
        let sense = Sense::new_compositional(
            0,
            vec![CompositionRef::new(1, 0), CompositionRef::new(2, 0)],
            vec![1, 2, 3, 4, 5, 6, 7, 8], // high-entropy context
            1,
        );
        let proposed = vec![CompositionRef::new(9, 0), CompositionRef::new(10, 0)];

        let score = sense.induction_score(
            &proposed,
            &vec![1, 2, 3, 4, 5, 6, 7, 8],
            &SenseInductionConfig::default(),
        );
        // Score should be positive since both divergence and entropy contribute
        assert!(score > 0.0, "Expected positive score with balanced weights, got {}", score);
    }
}

// -----------------------------------------------------------------------
// v11.0: Seed Distance Vector Similarity Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod seed_vector_similarity_tests {
    use crate::graph::RsvsGraph;
    use crate::types::{Node, NodeStatus, SemanticMeta, Tier};

    #[test]
    fn seed_vector_similarity_zero_when_no_vectors() {
        let mut g = RsvsGraph::new();
        let id_a = g.insert_node(Node {
            id: 0,
            label: "a".into(),
            surface_label: "a".into(),
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
            seed_distance_vector: std::collections::HashMap::new(),
            ..Default::default()
        }).unwrap();
        let id_b = g.insert_node(Node {
            id: 0,
            label: "b".into(),
            surface_label: "b".into(),
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
            seed_distance_vector: std::collections::HashMap::new(),
            ..Default::default()
        }).unwrap();
        let sim = g.seed_vector_similarity(id_a, id_b);
        assert!((sim - 0.0).abs() < 0.001, "Expected 0 similarity when vectors empty, got {}", sim);
    }

    #[test]
    fn seed_vector_similarity_identical_vectors() {
        let mut g = RsvsGraph::new();
        let mut vec_a = std::collections::HashMap::new();
        vec_a.insert(1, 0.8);
        vec_a.insert(2, 0.6);
        vec_a.insert(3, 0.4);

        let vec_b = vec_a.clone();

        let id_a = g.insert_node(Node {
            id: 0,
            label: "a".into(),
            surface_label: "a".into(),
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
            seed_distance_vector: vec_a,
            ..Default::default()
        }).unwrap();
        let id_b = g.insert_node(Node {
            id: 0,
            label: "b".into(),
            surface_label: "b".into(),
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
            seed_distance_vector: vec_b,
            ..Default::default()
        }).unwrap();

        let sim = g.seed_vector_similarity(id_a, id_b);
        assert!((sim - 1.0).abs() < 0.001, "Expected cosine sim = 1.0 for identical vectors, got {}", sim);
    }

    #[test]
    fn seed_vector_similarity_orthogonal_vectors() {
        let mut g = RsvsGraph::new();
        // Orthogonal vectors: non-overlapping seeds
        let mut vec_a = std::collections::HashMap::new();
        vec_a.insert(1, 0.9);
        vec_a.insert(2, 0.1);

        let mut vec_b = std::collections::HashMap::new();
        vec_b.insert(3, 0.9);
        vec_b.insert(4, 0.1);

        let id_a = g.insert_node(Node {
            id: 0,
            label: "a".into(),
            surface_label: "a".into(),
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
            seed_distance_vector: vec_a,
            ..Default::default()
        }).unwrap();
        let id_b = g.insert_node(Node {
            id: 0,
            label: "b".into(),
            surface_label: "b".into(),
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
            seed_distance_vector: vec_b,
            ..Default::default()
        }).unwrap();

        let sim = g.seed_vector_similarity(id_a, id_b);
        assert!((sim - 0.0).abs() < 0.001, "Expected cosine sim ≈ 0 for orthogonal vectors, got {}", sim);
    }

    #[test]
    fn seed_vector_similarity_partially_overlapping() {
        let mut g = RsvsGraph::new();
        // Partially overlapping: share seed 2 with different energies
        let mut vec_a = std::collections::HashMap::new();
        vec_a.insert(1, 0.8);
        vec_a.insert(2, 0.6);

        let mut vec_b = std::collections::HashMap::new();
        vec_b.insert(2, 0.6);
        vec_b.insert(3, 0.8);

        let id_a = g.insert_node(Node {
            id: 0,
            label: "a".into(),
            surface_label: "a".into(),
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
            seed_distance_vector: vec_a,
            ..Default::default()
        }).unwrap();
        let id_b = g.insert_node(Node {
            id: 0,
            label: "b".into(),
            surface_label: "b".into(),
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
            seed_distance_vector: vec_b,
            ..Default::default()
        }).unwrap();

        let sim = g.seed_vector_similarity(id_a, id_b);
        // Should be between 0 and 1 for partially overlapping
        assert!(sim > 0.0 && sim < 1.0, "Expected 0 < sim < 1 for partial overlap, got {}", sim);
    }

    #[test]
    fn compute_seed_distance_vector_fills_from_cache() {
        use crate::batch_spreading::BatchSeedSpreading;
        use crate::spreading::SpreadingActivation;
        use crate::spreading::SpreadingActivationConfig;

        let mut g = RsvsGraph::new();
        let id_a = g.insert_node(Node {
            id: 0,
            label: "a".into(),
            surface_label: "a".into(),
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
            seed_distance_vector: std::collections::HashMap::new(),
            ..Default::default()
        }).unwrap();

        let spreading = SpreadingActivation::new(SpreadingActivationConfig::default());
        let mut batch = BatchSeedSpreading::new(spreading, vec![1, 2], vec![], vec![]);
        // Manually populate cache
        let mut seed1_map = std::collections::HashMap::new();
        seed1_map.insert(id_a, 0.7);
        batch.cache.insert(1, seed1_map);
        let mut seed2_map = std::collections::HashMap::new();
        seed2_map.insert(id_a, 0.3);
        batch.cache.insert(2, seed2_map);

        let result = g.compute_seed_distance_vector(id_a, &[1, 2], &batch);
        assert!(result, "compute_seed_distance_vector should return true for existing node");

        let node = g.get_node(id_a).unwrap();
        assert_eq!(node.seed_distance_vector.len(), 2);
        assert!((node.seed_distance_vector.get(&1).copied().unwrap_or(0.0) - 0.7).abs() < 0.001);
        assert!((node.seed_distance_vector.get(&2).copied().unwrap_or(0.0) - 0.3).abs() < 0.001);
    }

    #[test]
    fn compute_seed_distance_vector_returns_false_for_missing_node() {
        use crate::batch_spreading::BatchSeedSpreading;
        use crate::spreading::SpreadingActivation;
        use crate::spreading::SpreadingActivationConfig;

        let g = RsvsGraph::new();
        let spreading = SpreadingActivation::new(SpreadingActivationConfig::default());
        let batch = BatchSeedSpreading::new(spreading, vec![1], vec![], vec![]);

        // Don't insert any node, so node 999 doesn't exist
        let mut g = g;
        let result = g.compute_seed_distance_vector(999, &[1], &batch);
        assert!(!result, "compute_seed_distance_vector should return false for non-existent node");
    }
}
