//! RSVS CLI — smoke test v0.5 (end-to-end pipeline)

use rsvs::{Rsvs, PipelineConfig, AutonomyConfig, SenseConfig, Tier};

fn main() {
    println!("=== RSVS Core v0.5 — End-to-End Pipeline ===\n");

    // ---------------------------------------------------------------
    // Init system
    // ---------------------------------------------------------------
    let config = PipelineConfig {
        autonomy: AutonomyConfig {
            n_warm:                10,
            threshold_global_delta: 2.0, // lenient for small corpus
            ..AutonomyConfig::default()
        },
        sense: SenseConfig {
            theta_assign: 0.12,
            ..SenseConfig::default()
        },
        entity_promote_n: 3,
        ..PipelineConfig::default()
    };

    let mut rsvs = Rsvs::new(config);
    println!("System initialized. Seed atoms: {}", rsvs.graph.node_count());

    // ---------------------------------------------------------------
    // Ingest corpus — domain 1: geology / physical properties
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

    println!("\n--- Ingesting geology corpus ({} sentences) ---", corpus_geo.len());
    let stats1 = rsvs.ingest_text(&corpus_geo.join(" "));
    println!("  sentences: {}  atoms promoted: {}  senses created: {}  updated: {}",
             stats1.sentences_processed, stats1.atoms_promoted,
             stats1.sense_created, stats1.confidence_updated);

    // ---------------------------------------------------------------
    // Ingest corpus — domain 2: water / liquid
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

    println!("\n--- Ingesting water corpus ({} sentences) ---", corpus_water.len());
    let stats2 = rsvs.ingest_text(&corpus_water.join(" "));
    println!("  sentences: {}  atoms promoted: {}  senses created: {}  updated: {}",
             stats2.sentences_processed, stats2.atoms_promoted,
             stats2.sense_created, stats2.confidence_updated);

    // ---------------------------------------------------------------
    // System status
    // ---------------------------------------------------------------
    let status = rsvs.status();
    println!("\n--- System Status ---");
    println!("  total nodes:    {}", status.total_nodes);
    println!("  total atoms:    {}", status.total_atoms);
    println!("  total contexts: {}", status.total_contexts);
    println!("  warmed up:      {}", status.warmed_up);
    println!("  θ_assign:       {:.3}", status.theta_assign);
    println!("  θ_merge:        {:.3}", status.theta_merge);
    println!("  watchlist:      {}", status.watchlist_count);

    // ---------------------------------------------------------------
    // Confidence snapshot for key atoms
    // ---------------------------------------------------------------
    println!("\n--- Atom Confidence ---");
    for token in &["stone", "hard", "solid", "water", "liquid", "heat", "pressure"] {
        if let Some(&id) = rsvs.token_to_id.get(*token) {
            let conf = rsvs.autonomy.confidence(id).unwrap_or(0.0);
            let tier = rsvs.autonomy.tier(id).cloned().unwrap_or(Tier::Tier3);
            let n_senses = rsvs.senses.get(&id).map(|s| s.sense_count()).unwrap_or(0);
            println!("  {:<10}: conf={:.3} tier={:?} senses={}",
                     token, conf, tier, n_senses);
        }
    }

    // ---------------------------------------------------------------
    // Similarity queries
    // ---------------------------------------------------------------
    println!("\n--- Similarity ---");
    for (a, b) in [("stone", "hard"), ("stone", "water"), ("hard", "solid")] {
        if let Some(sim) = rsvs.similarity(a, b) {
            let shared: Vec<_> = sim.shared.iter()
                .filter_map(|&id| rsvs.graph.get_node(id)?.label.clone())
                .collect();
            println!("  sim({}, {}): jaccard={:.3} shared={:?}",
                     a, b, sim.jaccard, shared);
        }
    }

    // ---------------------------------------------------------------
    // Context-aware queries
    // ---------------------------------------------------------------
    println!("\n--- Context Queries ---");

    let queries = vec![
        ("stone", "hard rough surface texture"),
        ("stone", "heat pressure formation geology"),
        ("water", "liquid clear flow river"),
        ("hard",  "solid stone metal material"),
    ];

    for (concept, context) in &queries {
        if let Some(result) = rsvs.query(concept, context) {
            let top: Vec<_> = result.scored_atoms.iter()
                .take(4)
                .map(|(l, s)| format!("{}({:.2})", l, s))
                .collect();
            println!("  query({:?}, {:?})", concept, context);
            println!("    sense {} (N={}) → {:?}", result.active_sense_idx,
                     result.active_sense_n, top);
        } else {
            println!("  query({:?}) → not found", concept);
        }
    }

    println!("\n=== v0.5 smoke test passed ===");
}
