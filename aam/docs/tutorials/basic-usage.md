# Basic Usage

This tutorial walks through the core RSVS workflow: ingest text, query concepts, compare similarity, and inspect the knowledge graph. By the end, you will understand how RSVS builds structured meaning from raw text.

---

## Create an Instance

```python
from rsvs import Rsvs

r = Rsvs(entity_promote_n=3, theta_assign=0.12, n_warm=20, eta=0.1)
```

The four hyperparameters control the fundamental behavior of the system:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `entity_promote_n` | 3 | Minimum co-occurrence contexts before a token becomes an atom |
| `theta_assign` | 0.12 | Jaccard overlap threshold for assigning a context to an existing sense |
| `n_warm` | 20 | Number of observations before the system leaves warm-up phase |
| `eta` | 0.1 | EMA smoothing factor for confidence updates (lower = more stable) |

For most use cases, the defaults work well. You can tune `entity_promote_n` down for small corpora (more promotions) or up for large corpora (fewer, more reliable promotions).

---

## Ingest Text

```python
sentences = [
    "Batu adalah material keras yang ditemukan di alam.",
    "Granit adalah jenis batu beku yang sangat keras.",
    "Marmer adalah batu metamorf yang digunakan untuk patung.",
    "Kayu adalah material organik yang berasal dari pohon.",
    "Besi adalah logam yang keras dan kuat.",
    "Air adalah zat cair yang esensial untuk kehidupan.",
    "Sungai mengalir dari gunung ke laut membawa batu dan pasir.",
    "Tulang manusia tersusun dari kalsium dan kolagen.",
]

for sentence in sentences:
    stats = r.ingest(sentence)
```

Each `ingest()` call returns an `IngestStats` object:

```python
print(f"Atoms promoted: {stats.atoms_promoted}")
print(f"Senses created: {stats.sense_created}")
print(f"Senses assigned: {stats.sense_assigned}")
print(f"Confidence updates: {stats.confidence_updated}")
```

During ingestion, RSVS tokenizes the input, tracks co-occurrence statistics, promotes tokens to atom status when they cross the `entity_promote_n` threshold, and induces new senses or assigns contexts to existing senses based on the `theta_assign` overlap threshold. The system bootstraps from 24 seed atoms (basic semantic primitives like "exists", "entity", "relation", etc.) that provide the initial vocabulary for compositional definitions.

---

## Query a Concept

```python
result = r.query("batu", "material keras")
if result:
    print(f"Sense: {result.sense_idx}/{result.sense_n}")
    print(f"Layer: {result.layer}")
    print(f"Grounding: {result.grounding_score:.3f}")
    print(f"Top atoms: {result.top_atoms(5)}")
    print(f"Compositions: {result.compositions}")
```

The `query()` method performs context-aware disambiguation. Given a concept label and a context string, it finds the best-matching sense of that concept. The result includes the active sense index, total number of senses for that concept, the compositional layer, grounding score, top-scored atoms, and the compositions that define the matched sense.

If the concept has multiple senses (polysemy), the context string determines which sense is active. For example, "batu" might have one sense for geological stone and another for gemstone — the context "material keras" would select the geological sense.

---

## Compare Similarity

RSVS provides two similarity methods:

### Jaccard Similarity (Flat)

```python
sim = r.similarity("batu", "kayu")
if sim:
    print(f"Jaccard: {sim.jaccard:.3f}")
    print(f"Shared: {sim.shared}")
    print(f"Only 'batu': {sim.only_a}")
    print(f"Only 'kayu': {sim.only_b}")
```

Jaccard similarity compares the atom sets of two nodes — the fraction of shared atoms over the union. This is a fast, interpretable baseline that works well for coarse-grained comparison.

### Structural Similarity (Compositional)

```python
ssim = r.structural_similarity("raja", "ratu")
if ssim:
    print(f"Score: {ssim.structural_similarity:.3f}")
    print(f"Shared: {ssim.shared_labels(r)}")
    print(f"Only 'raja': {ssim.only_a_compositions}")
    print(f"Only 'ratu': {ssim.only_b_compositions}")
```

Structural similarity compares composition structures at the sense level. Two nodes with the same Jaccard score can have very different structural similarity if their compositions are organized differently. This is RSVS's key advantage over flat similarity metrics — it captures the *shape* of meaning, not just the *bag* of atoms.

---

## Appraise Text

```python
verdict = r.appraise("Batu adalah material yang sangat keras dan kuat")
print(f"Verdict: {verdict.verdict}")           # "agree" or "disagree"
print(f"Agree: {verdict.agree_pct:.1f}%")
print(f"Disagree: {verdict.disagree_pct:.1f}%")
print(f"Evidence: {verdict.evidence[:3]}")
```

The `appraise()` method evaluates how well a piece of text aligns with the current knowledge graph. It returns a verdict (agree/disagree), the percentage of supporting and contradicting evidence, and a list of specific evidence nodes with their scores. This is useful for fact-checking, plausibility assessment, and detecting contradictions in text.

---

## Inspect the Graph

```python
# Detailed node information
info = r.node_info("batu")
print(f"ID: {info.id}, Confidence: {info.confidence:.3f}")
print(f"Tier: {info.tier}, Status: {info.status}, Layer: {info.layer}")

# All senses of a concept
senses = r.senses("batu")
for s in senses:
    print(f"Sense {s.sense_idx}: coherence={s.coherence:.3f}, "
          f"grounding={s.grounding_score:.3f}")

# List all nodes and atoms
all_nodes = r.nodes(include_seeds=False)
all_atoms = r.atoms(include_seeds=False)

# Confidence distribution
conf_map = r.confidence_map()
```

The inspection methods give you full visibility into the graph's internal state. `node_info()` returns detailed metadata including tier, status, layer, and compression state. `senses()` shows all senses of a concept with their grounding evidence and composition details. `nodes()` and `atoms()` enumerate the graph's contents, and `confidence_map()` gives you a complete confidence distribution for monitoring and debugging.

---

## Persist and Reload

```python
# Save to disk
r.save("my_graph.json")

# Load in a different session
r2 = Rsvs.load("my_graph.json")
print(r2.status())
```

The entire knowledge graph — nodes, edges, senses, autonomy records, co-occurrence statistics, and configuration — serializes to a single JSON file. This makes it easy to checkpoint, share, and version your graphs. The `load()` class method reconstructs the full graph state from the JSON file, including all Rust-side data structures.

---

## Next Steps

- [Indonesian NLP](indonesian-nlp.md) — Build a richer graph with domain-specific Indonesian corpora
- [Composition Demo](composition-demo.md) — Create explicit compositions and analyze structural similarity
- [Structural Reasoning](structural-reasoning.md) — MCTS reasoning and context-aware queries
