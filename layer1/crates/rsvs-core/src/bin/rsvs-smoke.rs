//! RSVS CLI — smoke test v6.0 (end-to-end pipeline)

use rsvs::{AutonomyConfig, NodeStatus, PipelineConfig, Rsvs, SenseConfig, Tier};

fn main() {
    println!("=== RSVS Core v6.0 — End-to-End Pipeline ===\n");

    // ---------------------------------------------------------------
    // Init system
    // ---------------------------------------------------------------
    let config = PipelineConfig {
        autonomy: AutonomyConfig {
            n_warm: 10,
            threshold_global_delta: 2.0,
            ..AutonomyConfig::default()
        },
        sense: SenseConfig {
            theta_assign: 0.12,
            ..SenseConfig::default()
        },
        entity_promote_n: 3,
        ..PipelineConfig::default()
    };

    let mut rsvs = Rsvs::new(config).expect("Failed to initialize RSVS");
    println!(
        "System initialized. Seed nodes: {}",
        rsvs.graph.node_count()
    );

    // Verify v6.0 seed nodes
    if let Some(&id) = rsvs.token_to_id.get("exists") {
        let node = rsvs.graph.get_node(id).unwrap();
        println!(
            "  Seed 'exists': surface_label={}, is_seed={}, status={:?}",
            node.surface_label,
            node.is_seed,
            rsvs.autonomy.status(id).unwrap()
        );
    }

    // ---------------------------------------------------------------
    // Ingest corpus — domain 1: geology
    // ---------------------------------------------------------------
    let corpus_geo = vec![
        "Stone is a hard solid mineral material.",
        "Rock is a hard heavy solid natural substance.",
        "Stone is formed by heat and pressure over time.",
        "Granite is a hard rough stone found in mountains.",
        "Stone has a rough hard texture on its surface.",
        "Metal is a hard solid material that conducts heat.",
        "Stone and metal are both hard solid materials.",
        "Hard solid materials resist pressure and force.",
        "Stone is heavy and hard like metal or bone.",
        "Rough hard surfaces like stone resist erosion.",
        "Stone is formed deep underground under great pressure.",
        "Heat and pressure transform soft rock into hard stone.",
        "Hard materials like stone and metal are solid.",
        "Stone sinks in water because it is heavy and solid.",
        "The hard rough surface of stone feels cold.",
    ];

    println!(
        "\n--- Ingesting geology corpus ({} sentences) ---",
        corpus_geo.len()
    );
    let stats1 = rsvs
        .ingest_text(&corpus_geo.join(" "))
        .expect("Ingest failed");
    println!(
        "  sentences: {}  nodes promoted: {}  senses created: {}  updated: {}",
        stats1.sentences_processed,
        stats1.atoms_promoted,
        stats1.sense_created,
        stats1.confidence_updated
    );

    // ---------------------------------------------------------------
    // Ingest corpus — domain 2: water
    // ---------------------------------------------------------------
    rsvs.config.current_domain = 2;

    let corpus_water = vec![
        "Water is a clear transparent liquid substance.",
        "Water flows downhill because it is liquid.",
        "Rain is water falling from clouds in the sky.",
        "Ice is frozen solid water formed by cold temperature.",
        "Water dissolves many solid materials over time.",
        "Liquid water becomes solid ice when temperature drops.",
        "Ocean water is salty liquid found in large bodies.",
        "Water is essential for all living organisms to survive.",
        "Clear liquid water reflects light and has no color.",
        "Water pressure increases with depth in the ocean.",
    ];

    println!(
        "\n--- Ingesting water corpus ({} sentences) ---",
        corpus_water.len()
    );
    let stats2 = rsvs
        .ingest_text(&corpus_water.join(" "))
        .expect("Ingest failed");
    println!(
        "  sentences: {}  nodes promoted: {}  senses created: {}  updated: {}",
        stats2.sentences_processed,
        stats2.atoms_promoted,
        stats2.sense_created,
        stats2.confidence_updated
    );

    // ---------------------------------------------------------------
    // System status
    // ---------------------------------------------------------------
    let status = rsvs.status();
    println!("\n--- System Status ---");
    println!("  total nodes:    {}", status.total_nodes);
    println!("  total atoms:    {}", status.total_atoms);
    println!("  total contexts: {}", status.total_contexts);
    println!("  warmed up:      {}", status.warmed_up);

    // ---------------------------------------------------------------
    // v6.0 Node info
    // ---------------------------------------------------------------
    println!("\n--- Node Info (v6.0) ---");
    for token in &[
        "stone", "hard", "solid", "water", "liquid", "heat", "pressure",
    ] {
        if let Some(&id) = rsvs.token_to_id.get(*token) {
            let conf = rsvs.autonomy.confidence(id).unwrap_or(0.0);
            let tier = rsvs.autonomy.tier(id).cloned().unwrap_or(Tier::Tier3);
            let node_status = rsvs.autonomy.status(id).cloned().unwrap_or(NodeStatus::New);
            let node = rsvs.graph.get_node(id);
            let surface = node.map(|n| n.surface_label.clone()).unwrap_or_default();
            let is_seed = node.map(|n| n.is_seed).unwrap_or(false);
            let compression = node
                .map(|n| format!("{:?}", n.semantic.compression_state))
                .unwrap_or_default();
            println!(
                "  {:<10}: conf={:.3} tier={:?} status={:?} surface={} seed={} compression={}",
                token, conf, tier, node_status, surface, is_seed, compression
            );
        }
    }

    // ---------------------------------------------------------------
    // v6.0: Appraise
    // ---------------------------------------------------------------
    println!("\n--- Appraise (v6.0) ---");
    let appraise_result = rsvs.appraise("Stone is hard and solid like metal");
    println!(
        "  agree: {:.1}%  conflict: {:.1}%  neutral: {:.1}%  verdict: {}",
        appraise_result.agree_pct, appraise_result.disagree_pct, appraise_result.neutral_pct, appraise_result.verdict
    );
    println!(
        "  evidence: {:?}",
        appraise_result.evidence.iter().take(5).collect::<Vec<_>>()
    );

    // ---------------------------------------------------------------
    // v6.0: Relate
    // ---------------------------------------------------------------
    println!("\n--- Relate (v6.0) ---");
    if let Some(relate_result) = rsvs.relate("stone") {
        let node_labels: Vec<String> = relate_result
            .related_nodes
            .iter()
            .take(5)
            .filter_map(|(id, score)| {
                let label = rsvs.graph.get_node(*id)?.label.clone();
                Some(format!("{}({:.3})", label, score))
            })
            .collect();
        println!("  related nodes: {:?}", node_labels);
        println!(
            "  related edges: {} edges found",
            relate_result.related_edges.len()
        );
    }

    // ---------------------------------------------------------------
    // Snapshot v6.0
    // ---------------------------------------------------------------
    println!("\n--- Snapshot v6.0 ---");
    let snap = rsvs.snapshot_v1();
    println!("  schema_version: {}", snap.schema_version);
    println!("  nodes: {}  edges: {}", snap.nodes.len(), snap.edges.len());

    // ---------------------------------------------------------------
    // Similarity queries
    // ---------------------------------------------------------------
    println!("\n--- Similarity ---");
    for (a, b) in [("stone", "hard"), ("stone", "water"), ("hard", "solid")] {
        if let Some(sim) = rsvs.similarity(a, b) {
            let shared: Vec<_> = sim
                .shared
                .iter()
                .filter_map(|&id| Some(rsvs.graph.get_node(id)?.label.clone()))
                .collect();
            println!(
                "  sim({}, {}): jaccard={:.3} shared={:?}",
                a, b, sim.jaccard, shared
            );
        }
    }

    println!("\n=== v6.0 smoke test passed ===");
}
