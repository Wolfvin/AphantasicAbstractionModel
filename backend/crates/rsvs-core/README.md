# rsvs-core

**Relational Symbolic Vocabulary System — Rust Core**

A cognitive symbolic engine implementing hard-attention knowledge graph construction, multi-sense disambiguation, and autonomous node lifecycle management.

## Features

- **Hard Attention Mechanism** — NPMI + Jaccard co-occurrence scoring with configurable thresholds
- **24 Seed Atoms** — Immutable bootstrap nodes (exists, entity, relation, …) that anchor all emergent knowledge
- **Multi-Sense Disambiguation** — Incremental coherence tracking, fragile-sense pruning, and mature-sense merging
- **Autonomy Engine** — EMA confidence updates, tiered classification (Tier1/2/3), hysteresis-based status lifecycle (New → Candidate → Stable → Deprecated → Quarantine)
- **Unified Node Model (v4.2)** — No more Atom/Composite split; compression is metadata
- **Policy Engine** — Governance scoring, dedup gates, status-flip budgets
- **Persistence** — Full JSON save/load with v4.2 snapshot format
- **Python Bindings** — Optional PyO3 bindings behind the `python` feature flag
- **Parallel Processing** — Rayon-based parallelism for entity detection and relate queries

## Quick Start

```rust
use rsvs::{Rsvs, PipelineConfig, AutonomyConfig, SenseConfig};

fn main() -> Result<(), rsvs::RsvsError> {
    let config = PipelineConfig {
        autonomy: AutonomyConfig { n_warm: 10, ..Default::default() },
        sense: SenseConfig { theta_assign: 0.12, ..Default::default() },
        entity_promote_n: 3,
        ..Default::default()
    };

    let mut rsvs = Rsvs::new(config)?;

    // Ingest text — nodes emerge from data
    let stats = rsvs.ingest_text("Stone is hard. Stone is solid. Metal is hard and solid.")?;
    println!("Promoted {} new nodes", stats.atoms_promoted);

    // Query a concept with context
    if let Some(result) = rsvs.query("stone", "hard solid material") {
        println!("Active sense: {} (N={})", result.active_sense_idx, result.active_sense_n);
    }

    // Appraise text against the graph
    let appraise = rsvs.appraise("Stone is hard and solid");
    println!("Verdict: {} ({:.1}% agree)", appraise.verdict, appraise.agree_pct);

    Ok(())
}
```

## API Surface

| Type | Description |
|------|-------------|
| `Rsvs` | Main system struct — holds graph, senses, autonomy engine, stats |
| `RsvsError` | Typed error enum (Graph, NodeNotFound, CircularRef, SeedInvariant, Persistence, Validation, Pipeline) |
| `PipelineConfig` | All tunable knobs (attention, sense, autonomy, entity threshold) |
| `IngestStats` | Return type from `ingest_text()` |
| `AppraiseResult` | Agree/disagree %, verdict, evidence from `appraise()` |
| `RelateResult` | Related nodes + edges from `relate()` |
| `Node` | v4.2 unified node with semantic metadata, policy meta, compression state |
| `Fingerprint` | Content-addressable hash for perceptual grounding |

### Key Methods on `Rsvs`

| Method | Returns | Description |
|--------|---------|-------------|
| `new(config)` | `Result<Self, RsvsError>` | Bootstrap with 24 seed atoms |
| `ingest_text(text)` | `Result<IngestStats, RsvsError>` | Process text, promote nodes, update confidence |
| `query(concept, context)` | `Option<QueryResult>` | Context-aware concept lookup |
| `similarity(a, b)` | `Option<SimilarityResult>` | Jaccard similarity between concepts |
| `appraise(text)` | `AppraiseResult` | Evaluate text against graph |
| `relate(concept)` | `Option<RelateResult>` | Find related nodes/edges |
| `snapshot_v1()` | `RuntimeSnapshot` | v4.2 format snapshot for UI |
| `status()` | `PipelineStatus` | System metrics |

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `python` | off | Enable PyO3 bindings for Python interop |

## Link

For the full project overview, architecture diagrams, and design decisions, see the [main project README](../../../README.md).
