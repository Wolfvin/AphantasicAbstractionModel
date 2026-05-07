# Indonesian NLP

RSVS was designed with Bahasa Indonesia as its primary development and testing language. This tutorial shows how to build a rich Indonesian knowledge graph using RSVS's embedded corpora and domain-specific processing.

---

## Why Bahasa Indonesia?

Most NLP tools and libraries are built for English first. RSVS takes a different approach: it prioritizes Bahasa Indonesia because the language presents unique challenges that expose weaknesses in English-centric NLP systems. Indonesian has rich morphology (prefixes, suffixes, circumfixes), widespread code-mixing with regional languages (Javanese, Sundanese), and a relatively smaller digital corpus compared to English. Building for Indonesian first ensures the architecture is robust for any language, not just high-resource ones.

RSVS handles these challenges through its language-agnostic architecture. The system does not need linguistic metadata to understand that "anjing" is Indonesian and "dog" is English — it observes that their sense compositions are structurally similar and they never co-occur in the same text, then automatically creates a convergence link. This means the system works out of the box for any language without requiring language-specific configuration.

---

## Using the Embedded Corpus

RSVS ships with an embedded Indonesian corpus covering multiple domains. You can access it through the Python API:

```python
from rsvs import Rsvs, DOMAINS, get_domain_text, get_all_text, domain_names

# List available domains
print(domain_names())
# ['geology', 'water', 'biology', 'physics', 'materials', 'kerajaan', 'konsep']

# Get text from a specific domain
geology_text = get_domain_text("geology")
print(f"Geology corpus: {len(geology_text)} characters")

# Get all domain text
all_text = get_all_text()
```

The seven embedded domains cover:

| Domain | Content | Focus |
|--------|---------|-------|
| `geology` | Batu, mineral, formasi geologis | Geological materials and processes |
| `water` | Air, sungai, laut, hidrologi | Water systems and hydrology |
| `biology` | Organisme, sel, DNA, ekosistem | Biological systems and taxonomy |
| `physics` | Gaya, energi, gelombang, partikel | Physical laws and phenomena |
| `materials` | Logam, polimer, keramik, komposit | Engineering materials and properties |
| `kerajaan` | Raja, ratu, kerajaan, kekuasaan | Royal/political vocabulary |
| `konsep` | Abstrak, relasi, kausalitas | Abstract conceptual vocabulary |

---

## Building a Domain-Specific Graph

```python
from rsvs import Rsvs, get_domain_text

r = Rsvs(entity_promote_n=2, theta_assign=0.10)

# Ingest specific domains
for domain in ["geology", "materials"]:
    text = get_domain_text(domain)
    # The corpus text contains multiple sentences
    for sentence in text.split("."):
        sentence = sentence.strip()
        if sentence:
            stats = r.ingest(sentence + ".")

print(f"Graph status: {r.status()}")
```

Setting `entity_promote_n=2` lowers the promotion threshold, which is useful for smaller corpora where you want more concepts to be promoted to atom status. For large corpora (thousands of sentences), the default of 3 works better to filter out noise.

---

## Ingesting from the CLI

The `rsvs` command-line tool provides a convenient way to ingest corpus data:

```bash
# Ingest specific domains
rsvs ingest-corpus --domains geology materials --db my_graph.json

# Ingest all domains
rsvs ingest-corpus --all --db my_graph.json

# Then query the graph
rsvs query batu "material keras" --db my_graph.json
rsvs similarity batu kayu --db my_graph.json
```

The CLI persists the graph to a JSON file (default: `./rsvs.json`), so you can build up the graph incrementally across multiple sessions.

---

## Cross-Domain Analysis

One of RSVS's strengths is comparing concepts across domains. After ingesting multiple domains, you can discover structural similarities that cut across domain boundaries:

```python
from rsvs import Rsvs, get_domain_text

r = Rsvs(entity_promote_n=2, theta_assign=0.10)

# Build a multi-domain graph
for domain in ["geology", "biology", "materials"]:
    text = get_domain_text(domain)
    for sentence in text.split("."):
        sentence = sentence.strip()
        if sentence:
            r.ingest(sentence + ".")

# Compare across domains: geological stone vs biological bone
ssim = r.structural_similarity("batu", "tulang")
if ssim:
    print(f"Structural similarity 'batu' vs 'tulang': {ssim.structural_similarity:.3f}")
    print(f"Shared compositions: {len(ssim.shared_compositions)}")

# What transforms stone into bone?
sub = r.substitution_analysis("batu", "tulang")
if sub:
    print(f"Substitutions: {sub.substitution_labels(r)}")
```

Even though "batu" (stone) and "tulang" (bone) come from different domains, RSVS can identify their shared structural properties (both are hard materials) and the precise substitutions that distinguish them (mineral composition vs biological tissue).

---

## Convergence Detection

When RSVS encounters words from different languages that have structurally equivalent sense compositions and never co-occur in the same text, it automatically creates convergence links:

```python
# After ingesting both Indonesian and English text
r.ingest("Anjing adalah hewan peliharaan yang setia.")
r.ingest("Dogs are loyal pets that humans keep.")

# The convergence engine will detect that "anjing" and "dog"
# have structurally equivalent compositions
r.consolidate()  # triggers convergence detection
```

The convergence engine is part of the consolidation cycle. It evaluates pairs of non-seed, high-confidence nodes and creates bidirectional `LanguageLink` records when structural equivalence is detected. These links enable cross-language querying: asking about "dog" can return information from the "anjing" node and vice versa.

---

## Multi-Language Composition

RSVS supports composing concepts from ingredients in different languages:

```python
# Compose "demokrasi" from Indonesian and Greek-derived components
r.compose("demokrasi", [("rakyat", 0), ("kekuasaan", 0)], lang="id")

# The same concept in English
r.compose("democracy", [("people", 0), ("power", 0)], lang="en")

# Structural similarity will reveal their equivalence
sim = r.structural_similarity("demokrasi", "democracy")
```

The `lang` parameter records the language context of each composition. This metadata enables downstream analysis of cross-lingual composition patterns and supports the convergence detection engine in identifying equivalent concepts across languages.

---

## Next Steps

- [Composition Demo](composition-demo.md) — Create explicit compositions and analyze structural similarity
- [Structural Reasoning](structural-reasoning.md) — MCTS reasoning and depth-controlled queries
- [API Reference](../api/index.md) — Complete method documentation
