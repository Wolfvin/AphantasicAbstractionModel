# Reasoning Operations

Advanced reasoning and graph maintenance operations.

---

## `mcts_query()`

Monte Carlo Tree Search for complex disambiguation. Uses UCB1 selection and structural value functions.

```python
result = r.mcts_query(
    label: str,
    simulations: int = 10,
    exploration: float = 1.414,
) -> MCTSResult | None
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label` | `str` | — | Concept label to reason about |
| `simulations` | `int` | 10 | Number of MCTS simulations to run |
| `exploration` | `float` | 1.414 | UCB1 exploration constant (c_puct) |

### Returns

`MCTSResult | None` with:

| Field | Type | Description |
|-------|------|-------------|
| `active_sense_idx` | `int` | Sense that MCTS settled on |
| `total_senses` | `int` | Total senses for the concept |
| `scored_atoms` | `list[tuple[str, float]]` | Atoms ranked by MCTS visit count |
| `depth_reached` | `int` | Maximum depth explored |
| `halt_reason` | `str` | Why MCTS stopped |
| `simulations_run` | `int` | Actual simulations executed |
| `best_path` | `list[tuple[str, int]]` | Highest-reward path as (label, sense_id) |
| `layer` | `int` | Layer of the active sense |
| `grounding_score` | `float` | Grounding score of the active sense |

### Halt Reasons

| Reason | Meaning |
|--------|---------|
| `"budget_exhausted"` | All simulations completed |
| `"stability_reached"` | Score converged |
| `"confidence_threshold"` | High-confidence result found |

---

## `set_thinking_mode()`

Control traversal depth for subsequent queries.

```python
r.set_thinking_mode(mode: str) -> None
```

### Modes

| Mode | Value | Behavior |
|------|-------|----------|
| `"AUTO"` | `"-1"` | Router decides based on complexity signals (default) |
| `"NON_THINKING"` | `"0"` | Shallow, fast — direct lookup or shallow traversal |
| `"THINKING"` | `"1"` | Deep, thorough — full MCTS with simulations |

---

## `consolidate()`

Periodic graph cleanup: merge similar senses, remove dead senses, prune weak edges, compact records.

```python
result = r.consolidate() -> ConsolidationResult
```

### Returns

`ConsolidationResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `senses_merged` | `int` | Senses merged due to high composition overlap |
| `senses_removed` | `int` | Dead senses removed |
| `edges_pruned` | `int` | Low-weight edges removed |
| `atoms_compacted` | `int` | Atom records compacted |

### Four-Phase Process

1. **Remove dead senses**: Fragile + ungrounded + very inactive
2. **Merge similar senses**: Composition overlap >= 0.8 within the same node (max 5 per cycle)
3. **Prune weak edges**: Learned edges with weight < 0.02 (bootstrap/composition edges preserved)
4. **Compact atom records**: Purge autonomy records for low-confidence nodes

---

## `run_reflection()`

Self-evaluate all senses and propose corrective actions.

```python
result = r.run_reflection() -> ReflectionResult
```

### Returns

`ReflectionResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `actions_total` | `int` | Total proposed actions |
| `actions_applied` | `int` | Actions actually applied (may be less due to rate limiting) |

### Action Escalation

| Action | Trigger | Effect |
|--------|---------|--------|
| **CONFIRM** | Grounding >= 0.60 | No action |
| **REVIEW** | Grounding 0.20–0.59 | Monitor |
| **REVISE** | Grounding < 0.20 or >= 3 consecutive REVIEWs | Prune least-grounded composition |
| **RETIRE** | Fragile + ungrounded + inactivity >= 100 | Mark for deletion |

---

## `verify()`

Neuro-symbolic composition verification. Checks five structural rules.

```python
result = r.verify() -> dict[str, int]
```

### Verification Rules

| Rule | Weight | Type | Description |
|------|--------|------|-------------|
| `no_self_reference` | 1.0 | Binary | Compositions must not reference the same node |
| `layer_consistency` | 0.8 | Soft | Compositions should reference equal or lower layers |
| `grounding_threshold` | 0.7 | Soft | Composition targets should be grounded |
| `frequency_threshold` | 0.5 | Soft | Composition targets should have sufficient frequency |
| `no_circular_chain` | 1.0 | Binary | Transitive closure must not loop back |
