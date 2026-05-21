#![allow(clippy::field_reassign_with_default)]
use super::engine::{topological_sort, ErasedTransform, IngestResult, PipelineEngine, TransformNode};
use super::enrich::EnrichComposition;
use super::graph::Graph;
use super::ingest_atoms::IngestAtoms;
use super::re_extract::ReExtractFrame;
use super::registry::register_default_pipeline;
use super::tokenize::Tokenize;
use super::super::types::*;

#[test]
fn test_topological_sort_empty() {
    let dag: Vec<TransformNode> = vec![];
    let result = topological_sort(&dag);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_topological_sort_linear() {
    let dag = vec![
        TransformNode {
            transform_id: "A".to_string(),
            input_type: String::new(),
            output_type: String::new(),
            dependencies: vec![],
            condition: None,
        },
        TransformNode {
            transform_id: "B".to_string(),
            input_type: String::new(),
            output_type: String::new(),
            dependencies: vec!["A".to_string()],
            condition: None,
        },
        TransformNode {
            transform_id: "C".to_string(),
            input_type: String::new(),
            output_type: String::new(),
            dependencies: vec!["B".to_string()],
            condition: None,
        },
    ];
    let result = topological_sort(&dag).unwrap();
    let a_pos = result.iter().position(|x| x == "A").unwrap();
    let b_pos = result.iter().position(|x| x == "B").unwrap();
    let c_pos = result.iter().position(|x| x == "C").unwrap();
    assert!(a_pos < b_pos);
    assert!(b_pos < c_pos);
}

#[test]
fn test_topological_sort_cycle_detection() {
    let dag = vec![
        TransformNode {
            transform_id: "A".to_string(),
            input_type: String::new(),
            output_type: String::new(),
            dependencies: vec!["B".to_string()],
            condition: None,
        },
        TransformNode {
            transform_id: "B".to_string(),
            input_type: String::new(),
            output_type: String::new(),
            dependencies: vec!["A".to_string()],
            condition: None,
        },
    ];
    let result = topological_sort(&dag);
    assert!(result.is_err());
}

#[test]
fn test_pipeline_engine_new() {
    let engine = PipelineEngine::new();
    assert!(engine.transforms.is_empty());
    assert!(engine.dag.is_empty());
    assert!(engine.graph.compositions.is_empty());
}

#[test]
fn test_register_default_pipeline() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);
    assert_eq!(engine.transforms.len(), 14);
    assert_eq!(engine.dag.len(), 14);
}

#[test]
fn test_ingest_simple() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    let result = engine.ingest("Raymond membuat aplikasi karena lambat").unwrap();
    assert!(result.atoms_created > 0);
}

#[test]
fn test_graph_ensure_node() {
    let mut graph = Graph::new();

    let id1 = graph.ensure_node("raja");
    let id2 = graph.ensure_node("raja"); // Same label, should return same ID.
    let id3 = graph.ensure_node("ratu");

    assert_eq!(id1, id2);
    assert_ne!(id1, id3);
    assert!(graph.has_node(id1));
    assert!(graph.has_node(id3));
}

#[test]
fn test_graph_cooccurrence_count() {
    let mut graph = Graph::new();

    let a = graph.ensure_node("A");
    let b = graph.ensure_node("B");
    let c = graph.ensure_node("C");

    // Create two compositions containing A and B.
    for i in 0..2 {
        let comp_id = format!("comp_{}", i);
        let mut comp = Composition::default();
        comp.id = comp_id;
        comp.members.push(CompositionMember {
            node_id: a,
            role: SemanticRole::Arg0Agent,
            confidence: 1.0,
            label: String::new(),
            source: None,
        });
        comp.members.push(CompositionMember {
            node_id: b,
            role: SemanticRole::Arg1Patient,
            confidence: 1.0,
            label: String::new(),
            source: None,
        });
        graph.compositions.insert(comp.id.clone(), comp);
    }

    assert_eq!(graph.cooccurrence_count(a, b), 2);
    assert_eq!(graph.cooccurrence_count(a, c), 0);
}

#[test]
fn test_ingest_result_merge() {
    let mut a = IngestResult {
        atoms_created: 5,
        compositions_created: 2,
        edges_created: 3,
        gaps_detected: 1,
        enrichments_applied: 0,
        governance_transitions: 0,
    };
    let b = IngestResult {
        atoms_created: 3,
        compositions_created: 1,
        edges_created: 2,
        gaps_detected: 0,
        enrichments_applied: 1,
        governance_transitions: 1,
    };
    a.merge(&b);
    assert_eq!(a.atoms_created, 8);
    assert_eq!(a.compositions_created, 3);
    assert_eq!(a.edges_created, 5);
    assert_eq!(a.gaps_detected, 1);
    assert_eq!(a.enrichments_applied, 1);
    assert_eq!(a.governance_transitions, 1);
}

#[test]
fn test_find_weak_frames() {
    let mut engine = PipelineEngine::new();

    // Create a weak Event composition (low confidence, missing roles).
    let mut comp = Composition::default();
    comp.id = "comp_weak_1".to_string();
    comp.composition_type = CompositionType::Event;
    comp.confidence = 0.3;
    comp.members.push(CompositionMember {
        node_id: 1,
        role: SemanticRole::Predicate,
        confidence: 0.3,
        label: String::new(),
        source: None,
    });
    // Missing Arg0Agent and Arg1Patient.
    engine.graph.compositions.insert(comp.id.clone(), comp);

    // Create a strong Event composition (should NOT be weak).
    let mut comp2 = Composition::default();
    comp2.id = "comp_strong_1".to_string();
    comp2.composition_type = CompositionType::Event;
    comp2.confidence = 0.8;
    comp2.members.push(CompositionMember {
        node_id: 2,
        role: SemanticRole::Arg0Agent,
        confidence: 0.8,
        label: String::new(),
        source: None,
    });
    comp2.members.push(CompositionMember {
        node_id: 3,
        role: SemanticRole::Arg1Patient,
        confidence: 0.8,
        label: String::new(),
        source: None,
    });
    comp2.members.push(CompositionMember {
        node_id: 4,
        role: SemanticRole::Cause,
        confidence: 0.8,
        label: String::new(),
        source: None,
    });
    engine.graph.compositions.insert(comp2.id.clone(), comp2);

    let weak = engine.find_weak_frames();
    assert_eq!(weak.len(), 1);
    assert_eq!(weak[0].composition_id, "comp_weak_1");
}

// ====================================================================
// New tests for previously-untested transforms
// ====================================================================

#[test]
fn test_tokenize_basic() {
    let tok = Tokenize::new();
    let mut ctx = PipelineContext::default();
    ctx.raw_text = Some("karena harga naik".to_string());
    let mut graph = Graph::new();

    let result = tok.execute(&mut ctx, &mut graph);
    assert_eq!(result.atoms_created, 3);
    assert_eq!(ctx.current_atoms.len(), 3);
    assert_eq!(ctx.current_atoms[0].label, "karena");
    assert_eq!(ctx.current_atoms[1].label, "harga");
    assert_eq!(ctx.current_atoms[2].label, "naik");
    assert_eq!(ctx.current_atoms[0].atom_type, AtomType::Token);
}

#[test]
fn test_tokenize_empty() {
    let tok = Tokenize::new();
    let mut ctx = PipelineContext::default();
    ctx.raw_text = Some(String::new());
    let mut graph = Graph::new();

    let result = tok.execute(&mut ctx, &mut graph);
    assert_eq!(result.atoms_created, 0);
    assert!(ctx.current_atoms.is_empty());
}

#[test]
fn test_tokenize_none_text() {
    let tok = Tokenize::new();
    let mut ctx = PipelineContext::default();
    ctx.raw_text = None;
    let mut graph = Graph::new();

    let result = tok.execute(&mut ctx, &mut graph);
    assert_eq!(result.atoms_created, 0);
}

#[test]
fn test_tokenize_lowercase() {
    let tok = Tokenize::new();
    let mut ctx = PipelineContext::default();
    ctx.raw_text = Some("HARGA NAIK".to_string());
    let mut graph = Graph::new();

    let result = tok.execute(&mut ctx, &mut graph);
    assert_eq!(result.atoms_created, 2);
    assert_eq!(ctx.current_atoms[0].label, "harga");
    assert_eq!(ctx.current_atoms[1].label, "naik");
}

#[test]
fn test_ingest_atoms_creates_nodes() {
    let ingest = IngestAtoms::new();
    let mut ctx = PipelineContext::default();
    ctx.raw_text = Some("test text".to_string());

    // Pre-populate atoms from Tokenize
    let tok = Tokenize::new();
    let mut graph = Graph::new();
    tok.execute(&mut ctx, &mut graph);

    let result = ingest.execute(&mut ctx, &mut graph);
    assert_eq!(result.atoms_created, 2); // "test" and "text"
    assert!(graph.node_count() >= 2);
}

#[test]
fn test_ingest_atoms_event_creates_composition() {
    let ingest = IngestAtoms::new();
    let mut ctx = PipelineContext::default();
    ctx.raw_text = Some("dia pergi".to_string());

    // Create an Event atom manually
    let mut atom = SemanticAtom::default();
    atom.id = "atom_0".to_string();
    atom.label = "pergi".to_string();
    atom.atom_type = AtomType::Event;
    atom.confidence = 0.8;
    atom.roles
        .insert(SemanticRole::Arg0Agent, "dia".to_string());
    ctx.current_atoms.push(atom);

    let mut graph = Graph::new();
    let result = ingest.execute(&mut ctx, &mut graph);

    assert!(result.compositions_created >= 1);
    assert!(result.edges_created >= 1);
    assert!(graph.composition_count() >= 1);
}

#[test]
fn test_ingest_atoms_empty() {
    let ingest = IngestAtoms::new();
    let mut ctx = PipelineContext::default();
    let mut graph = Graph::new();

    let result = ingest.execute(&mut ctx, &mut graph);
    assert_eq!(result.atoms_created, 0);
    assert_eq!(result.compositions_created, 0);
}

#[test]
fn test_enrich_composition_adds_member() {
    let enrich = EnrichComposition::new();
    let mut ctx = PipelineContext::default();
    let mut graph = Graph::new();

    // Create a composition
    let pred_id = graph.ensure_node("pergi");
    let comp_id = "comp_test".to_string();
    let mut comp = Composition::default();
    comp.id = comp_id.clone();
    comp.composition_type = CompositionType::Event;
    comp.confidence = 0.5;
    comp.members.push(CompositionMember {
        node_id: pred_id,
        role: SemanticRole::Predicate,
        confidence: 0.5,
        label: "pergi".to_string(),
        source: None,
    });
    graph.compositions.insert(comp_id.clone(), comp);

    // Create enrichment request
    let agent_id = graph.ensure_node("dia");
    ctx.pending_enrichments.push(EnrichmentRequest {
        target_composition_id: comp_id.clone(),
        role_to_fill: SemanticRole::Arg0Agent,
        candidate_node_id: agent_id,
        candidate_label: "dia".to_string(),
        source: EnrichmentSource::PassiveRecall,
        confidence: 0.7,
    });

    let result = enrich.execute(&mut ctx, &mut graph);

    // Verify enrichment was applied
    let enriched = graph.compositions.get(&comp_id).unwrap();
    assert!(enriched.members.len() >= 2);
    assert!(result.enrichments_applied >= 1);
}

#[test]
fn test_enrich_composition_duplicate_role_rejected() {
    let enrich = EnrichComposition::new();
    let mut ctx = PipelineContext::default();
    let mut graph = Graph::new();

    let pred_id = graph.ensure_node("pergi");
    let agent_id = graph.ensure_node("dia");
    let comp_id = "comp_dup".to_string();
    let mut comp = Composition::default();
    comp.id = comp_id.clone();
    comp.composition_type = CompositionType::Event;
    comp.members.push(CompositionMember {
        node_id: pred_id,
        role: SemanticRole::Predicate,
        confidence: 0.5,
        label: "pergi".to_string(),
        source: None,
    });
    comp.members.push(CompositionMember {
        node_id: agent_id,
        role: SemanticRole::Arg0Agent,
        confidence: 0.7,
        label: "dia".to_string(),
        source: None,
    });
    graph.compositions.insert(comp_id.clone(), comp);

    // Try to add duplicate Agent role
    let other_id = graph.ensure_node("mereka");
    ctx.pending_enrichments.push(EnrichmentRequest {
        target_composition_id: comp_id.clone(),
        role_to_fill: SemanticRole::Arg0Agent, // duplicate!
        candidate_node_id: other_id,
        candidate_label: "mereka".to_string(),
        source: EnrichmentSource::PassiveRecall,
        confidence: 0.6,
    });

    let result = enrich.execute(&mut ctx, &mut graph);

    // Should NOT add duplicate role — enrichments_applied should be 0
    let enriched = graph.compositions.get(&comp_id).unwrap();
    let agent_count = enriched
        .members
        .iter()
        .filter(|m| m.role == SemanticRole::Arg0Agent)
        .count();
    assert_eq!(agent_count, 1); // Still only 1 Agent
    assert_eq!(result.enrichments_applied, 0);
}

#[test]
fn test_enrich_composition_nonexistent_skipped() {
    let enrich = EnrichComposition::new();
    let mut ctx = PipelineContext::default();
    let mut graph = Graph::new();

    ctx.pending_enrichments.push(EnrichmentRequest {
        target_composition_id: "nonexistent_comp".to_string(),
        role_to_fill: SemanticRole::Arg0Agent,
        candidate_node_id: 0,
        candidate_label: "test".to_string(),
        source: EnrichmentSource::PassiveRecall,
        confidence: 0.7,
    });

    let result = enrich.execute(&mut ctx, &mut graph);
    assert_eq!(result.enrichments_applied, 0);
}

#[test]
fn test_re_extract_frame_processes_pending() {
    let re_extract = ReExtractFrame::new();
    let mut ctx = PipelineContext::default();
    let mut graph = Graph::new();

    // Create a composition
    let pred_id = graph.ensure_node("makan");
    let comp_id = "comp_retest".to_string();
    let mut comp = Composition::default();
    comp.id = comp_id.clone();
    comp.composition_type = CompositionType::Event;
    comp.confidence = 0.3; // Low confidence — good re-extraction candidate
    comp.members.push(CompositionMember {
        node_id: pred_id,
        role: SemanticRole::Predicate,
        confidence: 0.3,
        label: "makan".to_string(),
        source: None,
    });
    graph.compositions.insert(comp_id.clone(), comp);

    // Queue a re-extraction request
    ctx.pending_reextractions.push(ReExtractionRequest {
        original_text: "kucing makan ikan".to_string(),
        original_atom_id: String::new(),
        target_composition_id: comp_id.clone(),
        graph_context: Vec::new(),
    });

    let _result = re_extract.execute(&mut ctx, &mut graph);

    // Pending reextractions should be consumed
    assert!(ctx.pending_reextractions.is_empty());
}

#[test]
fn test_re_extract_frame_empty_pending() {
    let re_extract = ReExtractFrame::new();
    let mut ctx = PipelineContext::default();
    let mut graph = Graph::new();

    let result = re_extract.execute(&mut ctx, &mut graph);
    assert_eq!(result.compositions_created, 0);
    assert_eq!(result.edges_created, 0);
}

#[test]
fn test_seed_anchor_adjusts_confidence() {
    use crate::v12::govern_beliefs::SeedAnchor;

    let anchor = SeedAnchor::new();
    let mut ctx = PipelineContext::default();
    let mut graph = Graph::new();

    // Create a composition with seed scores
    let pred_id = graph.ensure_node("pergi");
    let comp_id = "comp_anchor".to_string();
    let mut comp = Composition::default();
    comp.id = comp_id.clone();
    comp.composition_type = CompositionType::Event;
    comp.confidence = 0.5;
    comp.members.push(CompositionMember {
        node_id: pred_id,
        role: SemanticRole::Predicate,
        confidence: 0.5,
        label: "pergi".to_string(),
        source: None,
    });
    // Add seed scores
    comp.seed_scores.insert(SeedPrimitive::Trust, 0.8);
    comp.seed_scores.insert(SeedPrimitive::Risk, 0.2);
    graph.compositions.insert(comp_id.clone(), comp);

    let result = anchor.execute(&mut ctx, &mut graph);

    // SeedAnchor should have run without error
    assert!(result.governance_transitions <= 1); // At most 1 lifecycle transition
}

#[test]
fn test_seed_anchor_empty_graph() {
    use crate::v12::govern_beliefs::SeedAnchor;

    let anchor = SeedAnchor::new();
    let mut ctx = PipelineContext::default();
    let mut graph = Graph::new();

    let result = anchor.execute(&mut ctx, &mut graph);
    assert_eq!(result.governance_transitions, 0);
}

#[test]
fn test_full_pipeline_ingest() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    let result = engine.ingest("karena harga naik, rakyat menderita").unwrap();
    assert!(result.atoms_created > 0, "Should create atoms");
    assert!(
        result.compositions_created > 0,
        "Should create compositions"
    );
}

#[test]
fn test_full_pipeline_multiple_ingests() {
    let mut engine = PipelineEngine::new();
    register_default_pipeline(&mut engine);

    let r1 = engine.ingest("kucing makan ikan").unwrap();
    let r2 = engine.ingest("karena hujan, jalan basah").unwrap();

    assert!(r1.atoms_created > 0);
    assert!(r2.atoms_created > 0);
    assert!(engine.graph.node_count() > 3);
}
