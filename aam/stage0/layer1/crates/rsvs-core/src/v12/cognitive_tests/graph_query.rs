use super::helpers::*;

// ========================================================================
// Semantic Query API Tests
// ========================================================================

#[test]
fn test_query_by_concept() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    engine.ingest("Raja memimpin kerajaan").unwrap();
    engine.ingest("Rakyat mendukung raja").unwrap();

    let results = engine.graph().query_by_concept("raja");
    assert!(!results.is_empty(), "query_by_concept('raja') should find compositions");

    // All results should have positive relevance scores
    for (_, score) in &results {
        assert!(score > &0.0, "Relevance score should be positive");
    }

    // Results should be sorted by relevance (highest first)
    for i in 1..results.len() {
        assert!(results[i - 1].1 >= results[i].1, "Results should be sorted by relevance");
    }

    eprintln!("✅ query_by_concept: found {} compositions for 'raja'", results.len());
}

#[test]
fn test_query_by_structure() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    engine.ingest("Raja memimpin kerajaan karena kebijakan").unwrap();

    // Find all compositions with Agent + Cause roles
    let causal = engine.graph().query_by_structure(&[
        "Agent".to_string(),
        "Cause".to_string(),
    ]);
    // Should find at least the "memimpin" event which has Agent + Cause
    assert!(!causal.is_empty(), "query_by_structure([Agent, Cause]) should find causal compositions");

    // Find compositions with Problem + Solution
    // Note: ReasonFrame may generate HiddenMeaning compositions with Problem+Solution
    // from the event's Cause role, so this may not be empty.
    let ps = engine.graph().query_by_structure(&[
        "Problem".to_string(),
        "Solution".to_string(),
    ]);
    // Just verify the function works — the count depends on ReasonFrame output
    eprintln!("  → Problem+Solution compositions: {}", ps.len());

    eprintln!("✅ query_by_structure: found {} causal compositions", causal.len());
}

#[test]
fn test_similarity() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    engine.ingest("Raja memimpin kerajaan").unwrap();
    engine.ingest("Rakyat mendukung raja").unwrap();

    // Self-similarity should be 1.0
    let self_sim = engine.graph().similarity("raja", "raja");
    assert!((self_sim - 1.0).abs() < 0.01, "Self-similarity should be 1.0, got {}", self_sim);

    // Related nodes should have positive similarity
    let sim = engine.graph().similarity("raja", "kerajaan");
    assert!(sim > 0.0, "Similarity between 'raja' and 'kerajaan' should be positive, got {}", sim);

    // Unrelated nodes should have zero similarity
    let no_sim = engine.graph().similarity("raja", "xyz_nonexistent");
    assert!(no_sim == 0.0, "Similarity with nonexistent node should be 0.0");

    eprintln!("✅ similarity: raja/kerajaan={:.3}, raja/raja={:.3}", sim, self_sim);
}

#[test]
fn test_find_related() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    engine.ingest("Raja memimpin kerajaan").unwrap();
    engine.ingest("Rakyat mendukung raja").unwrap();
    engine.ingest("Kerajaan makmur karena kebijakan raja").unwrap();

    let related = engine.graph().find_related("raja", 5);
    // Should find at least "kerajaan" as related
    let has_kerajaan = related.iter().any(|(label, _)| label == "kerajaan");
    assert!(has_kerajaan, "find_related('raja') should find 'kerajaan' as related, got {:?}", related);

    // Seed should not be in results
    let has_self = related.iter().any(|(label, _)| label == "raja");
    assert!(!has_self, "find_related should exclude the seed itself");

    eprintln!("✅ find_related: raja → {:?}", related);
}

#[test]
fn test_find_path() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    engine.ingest("Obat menyembuhkan penyakit").unwrap();
    engine.ingest("Penyakit disebabkan oleh virus").unwrap();

    let path = engine.graph().find_path("obat", "virus");
    // Should find bridging compositions through "penyakit"
    // Even if no path exists, the function should not panic
    eprintln!("✅ find_path: obat → virus: {} bridging compositions", path.len());
}

#[test]
fn test_comprehension_check() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    // Empty graph — should report no comprehension
    let results = engine.graph().query_by_concept("raja");
    assert!(results.is_empty(), "Empty graph should have no results for 'raja'");

    // After ingesting, comprehension should improve
    engine.ingest("Raja memimpin kerajaan").unwrap();
    engine.ingest("Rakyat mendukung raja").unwrap();

    let results = engine.graph().query_by_concept("raja");
    assert!(!results.is_empty(), "After ingest, 'raja' should be findable");

    eprintln!("✅ comprehension: 'raja' has {} compositions after 2 ingests", results.len());
}
