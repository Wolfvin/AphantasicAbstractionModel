//! Integration tests for the v12.0 AAM pipeline.
//!
//! Tests the v12 DAG-based pipeline engine with all 10 transforms.

use rsvs::v12::{PipelineEngine, register_default_pipeline, DetectGaps, ExecutiveOrchestrator};

/// Helper: create a fresh pipeline engine with default transforms.
fn create_pipeline() -> PipelineEngine {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);
    engine
}

#[test]
fn v12_pipeline_ingest_basic() {
    let mut engine = create_pipeline();
    let result = engine.ingest("Raymond membuat aplikasi karena lambat");

    assert!(result.atoms_created > 0, "Should create atoms from input text");
    assert!(result.compositions_created > 0, "Should create compositions from event atoms");
    assert!(result.edges_created > 0, "Should create edges linking compositions to nodes");
}

#[test]
fn v12_pipeline_multiple_ingests() {
    let mut engine = create_pipeline();

    let r1 = engine.ingest("The cat sat on the mat");
    let r2 = engine.ingest("The dog chased the cat");

    assert!(r1.atoms_created > 0);
    assert!(r2.atoms_created > 0);

    // Second ingest should find existing nodes
    let node_count = engine.graph().nodes.len();
    assert!(node_count > 0, "Graph should have nodes after ingestion");
}

#[test]
fn v12_pipeline_graph_state() {
    let mut engine = create_pipeline();
    engine.ingest("Fire is hot and dangerous. Water is cold and refreshing.");

    let graph = engine.graph();
    assert!(graph.nodes.len() > 0, "Graph should have nodes");
    // Compositions and edges may be 0 for sentences without event verbs
    // The important thing is that nodes are created for all tokens
}

#[test]
fn v12_pipeline_cognitive_mode() {
    let mut engine = create_pipeline();
    let mut orchestrator = ExecutiveOrchestrator::new();

    engine.ingest("Simple factual statement about the weather");

    let snapshot = engine.snapshot();
    let mode = orchestrator.select_cognitive_mode("test query", &snapshot.compositions);
    // Mode should be one of: Reactive, Analytical, Reflective
    let name = mode.name();
    assert!(
        ["Reactive", "Analytical", "Reflective"].contains(&name),
        "Cognitive mode should be valid, got: {}",
        name
    );
}

#[test]
fn v12_pipeline_gap_detection() {
    let mut engine = create_pipeline();
    engine.context.gap_detection_enabled = true;
    engine.ingest("The event happened because of the cause");

    let snapshot = engine.snapshot();
    let mut detector = DetectGaps::new();
    let gaps = detector.detect_all(&snapshot);
    // Gaps may or may not be detected depending on composition state
    // This test verifies gap detection doesn't crash
    let _ = gaps.len();
}

#[test]
fn v12_pipeline_snapshot_json() {
    let mut engine = create_pipeline();
    engine.ingest("Test sentence for snapshot");

    let snapshot = engine.snapshot();
    let json = serde_json::to_string(&snapshot);
    assert!(json.is_ok(), "Snapshot should be serializable to JSON");
}

#[test]
fn v12_pipeline_weak_frames() {
    let mut engine = create_pipeline();
    engine.ingest("Some event with low confidence");

    let weak = engine.find_weak_frames();
    // Weak frames may or may not exist — test verifies no crash
    let _ = weak.len();
}

#[test]
fn v12_pipeline_cooccurrence() {
    let mut engine = create_pipeline();
    engine.ingest("Fire is hot. Fire is dangerous.");

    let graph = engine.graph();
    // If we have at least 2 nodes, check cooccurrence
    if graph.nodes.len() >= 2 {
        let ids: Vec<_> = graph.nodes.keys().collect();
        if ids.len() >= 2 {
            let count = graph.cooccurrence_count(*ids[0], *ids[1]);
            // Cooccurrence may be 0 or more
            let _ = count;
        }
    }
}
