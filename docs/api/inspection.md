# Inspection Operations

Methods for inspecting the internal state of the knowledge graph.

---

## `node_info()`

Detailed information about a specific node.

```python
info = r.node_info(label: str) -> NodeInfo
```

### Returns

`NodeInfo` with:

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | Canonical label |
| `surface_label` | `str` | Display form |
| `id` | `int` | Unique NodeId |
| `confidence` | `float` | Confidence score (0.0–1.0) |
| `tier` | `int` | Tier level (1, 2, or 3) |
| `status` | `str` | Node status: "New", "Candidate", "Stable", "Deprecated", "Quarantine" |
| `is_seed` | `bool` | Whether this is an immutable seed atom |
| `is_locked` | `bool` | Whether the node is protected from deletion |
| `is_stable` | `bool` | Whether the node has reached Stable status |
| `compression_state` | `str` | "raw", "compressed", or "composed" |
| `layer` | `int` | Compositional depth |
| `atoms` | `list[int]` | Atom set as NodeId list |
| `derived_from_node_ids` | `list[int]` | Source nodes for compressed/derived nodes |
| `compression_reason` | `str | None` | Reason for compression |

---

## `senses()`

All senses of a concept with grounding evidence and composition details.

```python
senses = r.senses(concept: str) -> list[SenseInfo]
```

### Returns

`list[SenseInfo]` — each entry has:

| Field | Type | Description |
|-------|------|-------------|
| `sense_idx` | `int` | Sense index within the node |
| `n_contexts` | `int` | Number of observational contexts |
| `coherence` | `float` | Internal consistency score |
| `status` | `str` | "Fragile" or "Mature" |
| `core_atoms` | `list[str]` | Atom labels for this sense |
| `layer` | `int` | Compositional depth |
| `grounding_score` | `float` | Net grounding score |
| `grounding_evidence` | `GroundingEvidence` | Full evidence trail |
| `compositions` | `list[tuple[str, int]]` | (label, sense_id) composition pairs |
| `condition_label` | `str | None` | Optional annotation |

---

## `nodes()` / `atoms()`

List all known node or atom labels.

```python
all_nodes = r.nodes(include_seeds: bool = False) -> list[str]
all_atoms = r.atoms(include_seeds: bool = False) -> list[str]
```

---

## `confidence_map()`

Confidence scores for all nodes.

```python
conf_map = r.confidence_map() -> dict[str, float]
```

---

## `entity_candidates()`

Unpromoted tokens with highest centrality scores.

```python
candidates = r.entity_candidates(top_k: int = 10) -> list[tuple[str, float]]
```

---

## `status()`

System status including total nodes, atoms, contexts, and configuration parameters.

```python
status = r.status() -> dict[str, float]
```

---

## `pending_removals()`

Nodes marked for deletion that require explicit approval (high-impact nodes).

```python
pending = r.pending_removals() -> list[int]
```
