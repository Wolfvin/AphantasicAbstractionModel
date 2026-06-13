# API Reference

Complete Python API reference for RSVS. All methods are available on the `Rsvs` class, imported as:

```python
from rsvs import Rsvs
```

---

## Constructor

```python
r = Rsvs(
    entity_promote_n=3,   # Minimum co-occurrence contexts for atom promotion
    theta_assign=0.12,    # Jaccard threshold for sense assignment
    n_warm=20,            # Warm-up observations before adaptive thresholds
    eta=0.1,              # EMA smoothing factor for confidence
)
```

---

## Method Categories

| Category | Methods | Description |
|----------|---------|-------------|
| [Core](core.md) | `ingest`, `query`, `context_query`, `compose` | Build and query the knowledge graph |
| [Analysis](analysis.md) | `similarity`, `structural_similarity`, `substitution_analysis`, `context_similarity`, `appraise`, `relate` | Compare and analyze concepts |
| [Reasoning](reasoning.md) | `mcts_query`, `set_thinking_mode`, `consolidate`, `run_reflection`, `verify` | Advanced reasoning and maintenance |
| [Inspection](inspection.md) | `node_info`, `senses`, `nodes`, `atoms`, `confidence_map`, `entity_candidates`, `status` | Inspect graph internals |
| [Persistence](persistence.md) | `save`, `load`, `snapshot_v1`, `consume_events_v1`, `latest_seq_v1` | Save/load and event streaming |

---

## Type Reference

All return types are PyO3-native classes compiled from Rust. They are typed (PEP 561) with `.pyi` stubs for IDE support.

| Type | Source | Description |
|------|--------|-------------|
| `IngestStats` | `rsvs._rsvs` | Statistics from an ingest operation |
| `QueryResult` | `rsvs._rsvs` | Context-aware query result with active sense |
| `SimResult` | `rsvs._rsvs` | Jaccard similarity result with shared/unique atoms |
| `StructuralSimResult` | `rsvs._rsvs` | Structural similarity with shared/differing compositions |
| `SubstitutionResult` | `rsvs._rsvs` | Substitution pairs transforming concept A into B |
| `NodeInfo` | `rsvs._rsvs` | Detailed node metadata |
| `SenseInfo` | `rsvs._rsvs` | Sense details with grounding evidence |
| `AppraiseResult` | `rsvs._rsvs` | Text plausibility verdict |
| `RelateResult` | `rsvs._rsvs` | Related concepts via spreading activation |
| `MCTSResult` | `rsvs._rsvs` | MCTS reasoning result |
| `ContextQueryResult` | `rsvs._rsvs` | Depth-controlled traversal result |
| `ConsolidationResult` | `rsvs._rsvs` | Graph cleanup statistics |
| `ReflectionResult` | `rsvs._rsvs` | Self-correction action results |
| `GroundingEvidence` | `rsvs._rsvs` | Confirming/contradicting context counts |
