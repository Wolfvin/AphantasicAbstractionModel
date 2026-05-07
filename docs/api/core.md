# Core Operations

Primary operations for building and querying a knowledge graph.

---

## `ingest()`

Ingest text, update co-occurrence statistics, promote atoms, and induce senses.

```python
stats = r.ingest(text: str) -> IngestStats
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Input text to process. Multiple sentences are handled automatically. |

### Returns

`IngestStats` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `sentences_processed` | `int` | Number of sentences extracted from the input |
| `atoms_promoted` | `int` | New tokens promoted to atom status |
| `sense_assigned` | `int` | Contexts assigned to existing senses |
| `sense_created` | `int` | New senses created |
| `confidence_updated` | `int` | Nodes whose confidence was updated |
| `frozen_batches` | `int` | Batches that triggered consolidation |
| `compositions_induced` | `int` | Compositional references created |
| `atoms_flagged_inactive` | `int` | Atoms flagged for inactivity |

### Processing Pipeline

```
Tokenize → Co-occurrence → Entity Detection → Node Promotion → Sense Induction → Confidence Update
```

1. **Tokenize**: Split text into sentences and tokens
2. **Co-occurrence**: Update bigram and unigram counts, compute NPMI
3. **Entity Detection**: Check if tokens meet the promotion threshold
4. **Node Promotion**: Promote eligible tokens to atoms with initial confidence
5. **Sense Induction**: Assign contexts to existing senses or create new ones
6. **Confidence Update**: Apply EMA updates to all affected nodes

---

## `query()`

Context-aware concept query. Returns the active sense given a context string.

```python
result = r.query(concept: str, context: str) -> QueryResult | None
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `concept` | `str` | Label of the concept to query |
| `context` | `str` | Context string for disambiguation |

### Returns

`QueryResult | None` — `None` if the concept is not in the graph.

| Field | Type | Description |
|-------|------|-------------|
| `sense_idx` | `int` | Index of the active sense |
| `sense_n` | `int` | Total number of senses for this concept |
| `atoms` | `list[tuple[str, float]]` | Scored atoms (label, score) |
| `layer` | `int` | Compositional depth of the active sense |
| `grounding_score` | `float` | Grounding evidence score (0.0–1.0) |
| `compositions` | `list[tuple[str, int]]` | (label, sense_id) pairs defining this sense |
| `convergence_contributors` | `list[tuple[str, float]]` | Nodes contributing to convergent meaning paths |

### Example

```python
result = r.query("batu", "material keras")
if result:
    print(f"Active sense: {result.sense_idx}/{result.sense_n}")
    print(f"Grounding: {result.grounding_score:.3f}")
    print(f"Top atoms: {result.top_atoms(5)}")
```

---

## `context_query()`

Depth-controlled lazy traversal with context atoms. More powerful than `query()` for complex disambiguation.

```python
result = r.context_query(
    concept: str,
    context_atoms: list[str],
    max_depth: int | None = None,
    gamma: float | None = None,
    halt_confidence: float | None = None,
    tau_relevance: float | None = None,
) -> ContextQueryResult | None
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `concept` | `str` | — | Label of the concept to query |
| `context_atoms` | `list[str]` | — | Seed atoms for traversal |
| `max_depth` | `int` | 3 | Maximum traversal depth (safety net) |
| `gamma` | `float` | 0.01 | Stability threshold for halting |
| `halt_confidence` | `float` | 0.90 | Early halt when max score >= this |
| `tau_relevance` | `float` | 0.10 | Minimum relevance score for inclusion |

### Returns

`ContextQueryResult | None` with scored atoms, depth reached, and halt reason.

---

## `compose()`

Create a new compositional node from label/sense references.

```python
node_id = r.compose(
    label: str,
    compositions: list[tuple[str, int]],
    lang: str | None = None,
) -> int
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `label` | `str` | Label for the new node |
| `compositions` | `list[tuple[str, int]]` | List of (target_label, sense_index) pairs |
| `lang` | `str | None` | ISO 639-1 language code (e.g., "id", "en") |

### Returns

`int` — The NodeId of the newly created node.

### Example

```python
# "raja" = tahta_tertinggi + laki_laki + kerajaan
raja_id = r.compose("raja", [
    ("tahta_tertinggi", 0),
    ("laki_laki", 0),
    ("kerajaan", 0),
], lang="id")
```

### Validation

The compose operation verifies:

- No self-reference (compositions must not reference the same node they define)
- Layer consistency (compositions should reference equal or lower layers)
- No circular chains (transitive closure must not loop back)
- Grounding (composition targets should be grounded)

If validation fails, the DEPS planner provides structured recovery suggestions.
