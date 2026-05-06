# Composition Demo

RSVS's core innovation is compositional sense definitions: a concept is defined by its components (atoms or other composed concepts), not just by co-occurrence statistics. This tutorial demonstrates creating explicit compositions, analyzing structural similarity, and running substitution analysis.

---

## Build a Base Vocabulary

```python
from rsvs import Rsvs

r = Rsvs(entity_promote_n=2, theta_assign=0.10)

base_sentences = [
    "Laki-laki adalah manusia berjenis kelamin pria.",
    "Perempuan adalah manusia berjenis kelamin wanita.",
    "Kekuasaan adalah kemampuan untuk mempengaruhi orang lain.",
    "Negara adalah wilayah dengan pemerintahan tersendiri.",
    "Keraton adalah istana tempat raja tinggal.",
    "Hukum adalah aturan yang mengatur kehidupan bermasyarakat.",
    "Keluarga adalah unit terkecil dalam masyarakat.",
    "Rakyat adalah orang-orang yang tinggal di suatu negara.",
    "Perang adalah konflik bersenjata antar kelompok.",
    "Perdamaian adalah keadaan tanpa konflik.",
]

for s in base_sentences:
    r.ingest(s)
```

This builds a foundation of base concepts. Each sentence introduces concepts that will become ingredients in our compositions. The key insight is that RSVS does not need you to define what "laki-laki" or "kekuasaan" means — it learns their properties from co-occurrence patterns during ingestion.

---

## Create Explicit Compositions

The `compose()` method creates a new node whose meaning is explicitly defined as a composition of other senses:

```python
# "raja" = laki-laki + kekuasaan
raja_id = r.compose("raja", [("laki-laki", 0), ("kekuasaan", 0)], lang="id")
print(f"Composed 'raja' (id={raja_id})")

# "ratu" = perempuan + kekuasaan
ratu_id = r.compose("ratu", [("perempuan", 0), ("kekuasaan", 0)], lang="id")
print(f"Composed 'ratu' (id={ratu_id})")

# "negarawan" = kekuasaan + negara
negarawan_id = r.compose("negarawan", [("kekuasaan", 0), ("negara", 0)], lang="id")
print(f"Composed 'negarawan' (id={negarawan_id})")
```

Each composition is a list of `(label, sense_index)` tuples. The `sense_index` specifies which sense of the target concept to use — this is critical for disambiguation. If "bank" has sense 0 (financial) and sense 1 (river), you can compose from the specific sense you intend.

The `lang` parameter records the language context, enabling cross-language composition and convergence detection.

---

## Structural Similarity

Structural similarity compares the composition structures of two concepts at the sense level:

```python
ssim = r.structural_similarity("raja", "ratu")
if ssim:
    print(f"Score: {ssim.structural_similarity:.3f}")
    print(f"Shared compositions: {len(ssim.shared_compositions)}")
    print(f"Only 'raja': {len(ssim.only_a_compositions)}")
    print(f"Only 'ratu': {len(ssim.only_b_compositions)}")

    # Get human-readable labels for shared compositions
    if ssim.shared_labels(r):
        print(f"Shared labels: {ssim.shared_labels(r)}")
```

For "raja" vs "ratu", the result is:

- **Structural similarity**: 0.500 (1 shared out of 2 total unique compositions)
- **Shared**: `kekuasaan` (sense 0)
- **Only 'raja'**: `laki-laki` (sense 0)
- **Only 'ratu'**: `perempuan` (sense 0)

This is fundamentally different from cosine similarity. Instead of a single opaque number, you get a decomposition of *what* is shared and *what* differs.

---

## Substitution Analysis

Substitution analysis finds the precise swaps that transform concept A into concept B:

```python
sub = r.substitution_analysis("raja", "ratu")
if sub:
    print(f"Structural similarity: {sub.structural_similarity:.3f}")
    print(f"Substitutions needed: {len(sub.substitutions)}")
    if sub.substitution_labels(r):
        labels = sub.substitution_labels(r)
        print(f"Substitutions: {labels}")
```

For "raja" → "ratu", the result is a single substitution: `laki_laki` (sense 0) → `perempuan` (sense 0). One swap is the entire semantic difference between king and queen — expressed as a precise structural transformation rather than a fuzzy vector distance.

### How It Works

Substitution analysis performs minimum-cost bipartite matching between the differing compositions of two concepts:

1. Find the best-matching sense pair (highest structural similarity)
2. Compute the composition diff: shared, only-A, only-B
3. Match only-A compositions to only-B compositions by similarity
4. Paired matches become substitutions; unpaired compositions remain as remainders

This algorithm produces human-readable results that can be inspected, verified, and composed into higher-order reasoning chains.

---

## Context-Weighted Similarity

Context similarity weights the comparison by relevance to a specific context:

```python
# Compare "raja" and "negarawan" in the context of "kekuasaan"
csim = r.context_similarity("raja", "negarawan", ["kekuasaan"])
if csim is not None:
    print(f"In context ['kekuasaan']: {csim:.3f}")

# Same concepts, different context
csim2 = r.context_similarity("raja", "negarawan", ["negara", "hukum"])
if csim2 is not None:
    print(f"In context ['negara', 'hukum']: {csim2:.3f}")
```

The same pair of concepts can have different similarity scores depending on the context. "Raja" and "negarawan" are more similar when thinking about power (kekuasaan) than when thinking about governance (negara, hukum), because their shared composition (kekuasaan) is more relevant to the first context.

---

## Verify the Graph

After creating compositions, verify structural invariants:

```python
verify_result = r.verify()
print(f"Verification: {verify_result}")
```

The verification engine checks five structural rules:

| Rule | Type | What it checks |
|------|------|---------------|
| `no_self_reference` | Binary | Compositions must not reference the same node they define |
| `layer_consistency` | Soft | Compositions should reference equal or lower layers |
| `grounding_threshold` | Soft | Composition targets should be grounded |
| `frequency_threshold` | Soft | Composition targets should have sufficient frequency |
| `no_circular_chain` | Binary | Transitive closure must not loop back to the node |

If verification fails, the DEPS (Dependency-aware Entity Promotion System) planner provides structured recovery suggestions.

---

## Next Steps

- [Structural Reasoning](structural-reasoning.md) — MCTS reasoning and depth-controlled queries
- [API Reference: Core Operations](../api/core.md) — Full method documentation
- [API Reference: Analysis Operations](../api/analysis.md) — Similarity and substitution details
