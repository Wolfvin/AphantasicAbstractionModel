//! Integration tests for the RSVS pipeline.

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
