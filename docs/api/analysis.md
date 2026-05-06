# Analysis Operations

Methods for comparing concepts and analyzing text against the knowledge graph.

---

## `similarity()`

Flat Jaccard similarity based on shared atoms.

```python
result = r.similarity(a: str, b: str) -> SimResult | None
```

### Returns

`SimResult | None` — `None` if either concept is not in the graph.

| Field | Type | Description |
|-------|------|-------------|
| `jaccard` | `float` | Jaccard similarity score (0.0–1.0) |
| `shared` | `list[str]` | Labels of atoms shared between both concepts |
| `only_a` | `list[str]` | Labels of atoms unique to concept A |
| `only_b` | `list[str]` | Labels of atoms unique to concept B |

---

## `structural_similarity()`

Sense-level composition comparison. Compares the structural shape of meaning, not just atom overlap.

```python
result = r.structural_similarity(a: str, b: str) -> StructuralSimResult | None
```

### Returns

`StructuralSimResult | None` with:

| Field | Type | Description |
|-------|------|-------------|
| `sense_idx_a` | `int` | Best-matching sense index for concept A |
| `sense_idx_b` | `int` | Best-matching sense index for concept B |
| `structural_similarity` | `float` | Structural similarity score (0.0–1.0) |
| `shared_compositions` | `list[tuple[int, int]]` | Shared (node_id, sense_id) pairs |
| `only_a_compositions` | `list[tuple[int, int]]` | Compositions unique to A's best sense |
| `only_b_compositions` | `list[tuple[int, int]]` | Compositions unique to B's best sense |
| `layer_a` | `int` | Layer of A's matched sense |
| `layer_b` | `int` | Layer of B's matched sense |

### Label Resolution

```python
# Get human-readable labels instead of integer IDs
labels = result.shared_labels(r)   # list[tuple[str, int]]
```

---

## `substitution_analysis()`

Find the precise swaps that transform concept A into concept B.

```python
result = r.substitution_analysis(a: str, b: str) -> SubstitutionResult | None
```

### Returns

`SubstitutionResult | None` with:

| Field | Type | Description |
|-------|------|-------------|
| `sense_idx_a` | `int` | Sense index for A in the best-matching pair |
| `sense_idx_b` | `int` | Sense index for B in the best-matching pair |
| `structural_similarity` | `float` | Structural similarity of the matched senses |
| `substitutions` | `list[tuple[int, int, int, int]]` | (from_node, from_sense, to_node, to_sense) pairs |
| `unpaired_only_a` | `list[tuple[int, int]]` | A compositions with no match in B |
| `unpaired_only_b` | `list[tuple[int, int]]` | B compositions with no match in A |

### Label Resolution

```python
# Human-readable substitution labels
labels = result.substitution_labels(r)
# e.g., [("laki_laki", 0, "perempuan", 0)]
```

---

## `context_similarity()`

Context-weighted similarity. Weights shared atoms by their relevance to the provided context.

```python
score = r.context_similarity(a: str, b: str, context: list[str]) -> float | None
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `str` | First concept label |
| `b` | `str` | Second concept label |
| `context` | `list[str]` | Context atom labels for weighting |

### Returns

`float | None` — Context-weighted similarity score, or `None` if either concept is not found.

---

## `appraise()`

Evaluate text plausibility against the knowledge graph.

```python
result = r.appraise(text: str) -> AppraiseResult
```

### Returns

`AppraiseResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `agree_pct` | `float` | Percentage of supporting evidence |
| `disagree_pct` | `float` | Percentage of contradicting evidence |
| `verdict` | `str` | "agree" or "disagree" |
| `evidence` | `list[tuple[str, float]]` | Supporting/contradicting nodes with scores |
| `convergence_info` | `list[tuple[str, float]]` | Convergent evidence paths |

---

## `relate()`

Find related nodes and edges via spreading activation along composition edges.

```python
result = r.relate(concept: str) -> RelateResult | None
```

### Returns

`RelateResult | None` with:

| Field | Type | Description |
|-------|------|-------------|
| `related_nodes` | `list[tuple[int, float]]` | (node_id, score) pairs |
| `related_edges` | `list[tuple[int, int, float]]` | (from_id, to_id, weight) triples |
| `structural_relations` | `list[tuple[int, float]]` | Structurally related nodes with scores |

### Label Resolution

```python
labels = result.node_labels(r)         # list[tuple[str, float]]
structural = result.structural_labels(r) # list[tuple[str, float]]
```
