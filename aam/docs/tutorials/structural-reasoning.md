# Structural Reasoning

RSVS provides sophisticated reasoning capabilities beyond simple lookup. This tutorial covers Monte Carlo Tree Search (MCTS), context-aware queries with depth-controlled traversal, and graph maintenance through consolidation and reflection.

---

## Build a Rich Knowledge Base

```python
from rsvs import Rsvs

r = Rsvs(entity_promote_n=2, theta_assign=0.10, n_warm=15, eta=0.1)

corpus = [
    # Geology
    "Batu adalah material padat dari mineral.",
    "Granit adalah batu beku yang terbentuk dari magma.",
    "Marmer terbentuk dari batu gamping yang mengalami metamorfisme.",
    "Pasir adalah butiran kecil dari batu yang tererosi.",
    # Biology
    "Tulang adalah jaringan keras yang menyusun rangka manusia.",
    "Kalsium adalah mineral utama dalam tulang.",
    "Kolagen adalah protein yang memberi fleksibilitas pada tulang.",
    "Otak adalah organ pusat sistem saraf manusia.",
    # Physics
    "Kekerasan adalah ketahanan material terhadap deformasi.",
    "Mohs scale mengukur kekerasan mineral dari 1 sampai 10.",
    "Intan memiliki kekerasan 10 pada skala Mohs.",
    "Elastisitas adalah kemampuan material kembali ke bentuk semula.",
    # Materials
    "Besi adalah logam yang keras dan magnetis.",
    "Baja adalah paduan besi dengan karbon.",
    "Kaca adalah material transparan dari silika.",
    "Plastik adalah polimer sintetis yang ringan.",
]

for sentence in corpus:
    r.ingest(sentence)
```

This multi-domain corpus creates a rich knowledge base with cross-domain connections — "batu" and "tulang" are both hard materials, "besi" and "baja" are related metals, and so on. These cross-domain connections enable the most interesting reasoning patterns.

---

## Context-Aware Queries

### Standard Query

```python
q_simple = r.query("batu", "material")
if q_simple:
    print(f"Sense: {q_simple.sense_idx}/{q_simple.sense_n}")
    print(f"Top atoms: {q_simple.top_atoms(5)}")
```

A standard query performs simple context disambiguation — it finds the best-matching sense of "batu" given the context "material". This is fast but shallow.

### Depth-Controlled Context Query

```python
cqr = r.context_query(
    "batu",
    context_atoms=["kekerasan", "mineral", "material"],
    max_depth=5,
    gamma=0.85,
    halt_confidence=0.7,
)
if cqr:
    print(f"Sense: {cqr.active_sense_idx}/{cqr.total_senses}")
    print(f"Depth reached: {cqr.depth_reached}")
    print(f"Halt reason: {cqr.halt_reason}")
    print(f"Top atoms: {cqr.scored_atoms[:5]}")
```

The context query performs depth-controlled lazy traversal starting from the provided context atoms. It expands the frontier one hop at a time, scoring each encountered node with P(a|S,q) — the probability that atom a is relevant given the seed set S and query context q. The traversal stops when one of three conditions is met:

| Halting Condition | Parameter | Description |
|-------------------|-----------|-------------|
| Score stability | — | Top-K ranked atoms haven't changed in the last two expansions |
| Confidence threshold | `halt_confidence` | Cumulative score mass exceeds the threshold |
| Information gain | `gamma` | Marginal information from the last expansion falls below gamma |

The `max_depth` parameter is a safety net — it caps the maximum number of hops regardless of other halting conditions. Setting it to 5 means the traversal will explore at most 5 hops from the seed atoms.

---

## MCTS Reasoning

Monte Carlo Tree Search treats the knowledge graph as a game tree where each "move" is traversing an edge and the "reward" is the relevance score of the destination node:

```python
mcts = r.mcts_query("batu", simulations=50, exploration=1.414)
if mcts:
    print(f"Active sense: {mcts.active_sense_idx}/{mcts.total_senses}")
    print(f"Simulations run: {mcts.simulations_run}")
    print(f"Depth reached: {mcts.depth_reached}")
    print(f"Halt reason: {mcts.halt_reason}")
    print(f"Best path: {mcts.best_path}")
    print(f"Top atoms: {mcts.scored_atoms[:5]}")
```

### How MCTS Works

RSVS's MCTS uses AlphaZero-style tree search with three key differences from traditional game-tree MCTS:

1. **UCB1 Selection**: `UCB1(child) = Q(child) + c_puct * sqrt(N(parent)) / N(child)` — balances exploitation (following high-value paths) with exploration (trying less-visited paths)
2. **Structural Value Function**: `value = grounding * coherence` — no neural networks, deterministic structural quality. This makes MCTS results reproducible and interpretable
3. **Backtracking**: Paths with value < 0.5 are abandoned with a penalty, preventing the search from wasting budget on low-quality branches

The `exploration` parameter (default: 1.414) controls the exploration-exploitation tradeoff. Higher values encourage more exploration of less-visited paths, which is useful for discovering unexpected connections. Lower values favor exploiting known high-value paths, which is better for targeted queries.

### When to Use MCTS

Use MCTS when:

- The concept has multiple senses and simple context disambiguation is insufficient
- You need to explore reasoning paths beyond the reach of greedy best-first search
- The query involves complex cross-domain reasoning

Use standard queries when:

- The concept has one or two well-defined senses
- A simple context string is sufficient for disambiguation
- Latency is critical (MCTS takes longer than direct lookup)

---

## Thinking Mode

RSVS supports a "thinking mode" toggle that controls traversal depth:

```python
# Fast mode — direct lookup or shallow traversal
r.set_thinking_mode("NON_THINKING")  # or "0"

# Deep mode — full MCTS with simulations
r.set_thinking_mode("THINKING")  # or "1"

# Auto mode — the system decides based on query complexity
r.set_thinking_mode("AUTO")  # or "-1" (default)
```

The AUTO mode uses five complexity signals to decide:

| Signal | Threshold | Meaning |
|--------|-----------|---------|
| `n_context_atoms` | > 5 | Many context atoms suggest complex query |
| `n_senses` | > 3 | Multiple senses require deeper disambiguation |
| `target_layer` | > 1 | Higher-layer concepts are structurally richer |
| `is_compositional` | true | Compositional concepts need structural analysis |
| `domain_complexity` | > 0.5 | Domain-specific vocabulary requires deeper search |

When 2 or more signals exceed their thresholds, AUTO switches to THINKING mode. This mirrors the System 1 / System 2 distinction from cognitive science — fast mode for real-time responses, deep mode for careful reasoning.

---

## Graph Maintenance

### Consolidation

```python
consolidation = r.consolidate()
print(f"Senses merged: {consolidation.senses_merged}")
print(f"Senses removed: {consolidation.senses_removed}")
print(f"Edges pruned: {consolidation.edges_pruned}")
print(f"Atoms compacted: {consolidation.atoms_compacted}")
```

Consolidation runs a four-phase cleanup:

1. **Remove dead senses**: Fragile + ungrounded + very inactive
2. **Merge similar senses**: Composition overlap >= 0.8 within the same node
3. **Prune weak edges**: Learned edges with weight < 0.02 (bootstrap and composition edges are preserved)
4. **Compact atom records**: Purge autonomy records for low-confidence nodes

Consolidation runs automatically every 50 ingest batches, but you can also trigger it manually. It is the graph equivalent of garbage collection — run it periodically to keep the graph lean and correct.

### Reflection

```python
reflection = r.run_reflection()
print(f"Actions total: {reflection.actions_total}")
print(f"Actions applied: {reflection.actions_applied}")
```

The reflection engine evaluates all senses and proposes corrective actions:

| Action | Trigger | Effect |
|--------|---------|--------|
| CONFIRM | Grounding >= 0.60 | No action — sense is well-grounded |
| REVIEW | Grounding 0.20–0.59 | Monitor — track consecutive reviews |
| REVISE | Grounding < 0.20 or >= 3 consecutive REVIEWs | Prune least-grounded composition |
| RETIRE | Fragile + ungrounded + inactivity >= 100 | Mark for deletion |

REVISE actions are rate-limited (max 3 per cycle) to prevent catastrophic pruning. Three consecutive REVIEWs automatically escalate to REVISE, ensuring that consistently marginal senses get corrected rather than accumulating indefinitely.

### Verification

```python
verify = r.verify()
print(f"Verification: {verify}")
```

Verification checks five structural invariants across the entire graph. It is useful for CI pipelines, debugging, and periodic health checks. If any violations are found, the DEPS planner can generate recovery plans.

---

## Entity Candidates and Pending Removals

```python
# Find tokens that should be promoted to nodes
candidates = r.entity_candidates(top_k=5)
for label, score in candidates[:5]:
    print(f"'{label}': score={score:.3f}")

# Check if any nodes are pending removal (need approval)
pending = r.pending_removals()
if pending:
    print(f"Pending removals: {pending}")
```

`entity_candidates()` returns tokens that are close to promotion threshold but haven't been promoted yet. These are useful for monitoring graph growth and identifying concepts that need more evidence. `pending_removals()` shows nodes that have been marked for deletion but have high impact (many dependents) and require explicit approval before removal.

---

## Next Steps

- [API Reference: Reasoning Operations](../api/reasoning.md) — Full MCTS and thinking mode documentation
- [API Reference: Inspection Operations](../api/inspection.md) — Entity candidates, confidence maps, and status
- [Architecture](../architecture.md) — How the Rust core implements these algorithms
