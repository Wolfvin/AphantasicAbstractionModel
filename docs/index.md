---
title: RSVS — Recursive Symbolic Vocabulary System
description: Compositional symbolic meaning, not embeddings. Traceable sense definitions with structural similarity.
---

# RSVS

**Compositional symbolic meaning, not embeddings.**

RSVS builds structured knowledge graphs from raw text where every concept is a composition of other concepts — and every composition is traceable. Unlike opaque vector embeddings, RSVS lets you follow the chain from any concept down to its constituent parts, explain precisely why two concepts are related, and identify the exact substitution that transforms one into another.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Quick Start__

    ---

    Install the library and build your first knowledge graph in under 2 minutes.

    ```bash
    pip install rsvs
    ```

    [:octicons-arrow-right-24: Getting started](tutorials/basic-usage.md)

-   :material-graph:{ .lg .middle } __Compositional Senses__

    ---

    Define concepts as compositions of other senses. "King" = "highest throne" + "male" + "kingdom" — precise, inspectable, not a statistical artifact.

    [:octicons-arrow-right-24: Composition demo](tutorials/composition-demo.md)

-   :material-swap-horizontal:{ .lg .middle } __Structural Similarity__

    ---

    Go beyond cosine similarity. RSVS tells you that "raja" and "ratu" share 2 compositions and differ in exactly 1 — with a name for the swap.

    [:octicons-arrow-right-24: Substitution analysis](tutorials/composition-demo.md#substitution-analysis)

-   :material-brain:{ .lg .middle } __MCTS Reasoning__

    ---

    Monte Carlo Tree Search for complex disambiguation paths. Structured value functions, no neural networks.

    [:octicons-arrow-right-24: Structural reasoning](tutorials/structural-reasoning.md)

</div>

---

## Why RSVS?

If you have used word embeddings or sentence transformers, you are familiar with the pattern: "raja and ratu have cosine similarity 0.87." But what does that number mean? Which aspects of meaning make them similar? What would you need to change to transform one into the other?

Embeddings cannot answer these questions because they compress meaning into a single opaque vector. Knowledge graphs and ontologies offer more structure, but they require manual schema design and struggle with ambiguity, multiple senses, and the fluid nature of natural language meaning.

RSVS occupies a different position:

| | Embeddings | Ontologies | **RSVS** |
|---|---|---|---|
| Traceability | None | Manual | Automatic |
| Ambiguity handling | None (one vector) | Manual multi-sense | Native multi-sense |
| Similarity explanation | Cosine number | Shared ancestors | Shared + differing compositions |
| Schema required | No | Yes | No (bootstraps from seeds) |
| Substitution analysis | Impossible | Manual | Automatic |

When RSVS tells you that "raja" and "ratu" share two compositions (`tahta_tertinggi` and `kerajaan`) and differ in exactly one (`laki_laki` versus `perempuan`), you have an answer you can inspect, debug, and reason about.

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                  Frontend (Optional / Demo)                  │
│           Next.js 16 · React Three Fiber · shadcn/ui        │
├─────────────────────────────────────────────────────────────┤
│              Python Bridge (HTTP + Validation)               │
│            FastAPI · PyO3 FFI · CLI · Artifacts              │
├─────────────────────────────────────────────────────────────┤
│                Rust Core (All Computation)                   │
│      Graph · Attention · Autonomy · Sense · MCTS · Persist  │
└─────────────────────────────────────────────────────────────┘
```

- **Rust Core**: All computational logic — zero I/O, zero HTTP, zero Python dependencies
- **Python Bridge**: HTTP layer, validation, artifact persistence — no computation
- **Frontend**: Interactive 3D graph visualization — demo/optional

[:octicons-arrow-right-24: Full architecture reference](architecture.md)

---

## Install

```bash
# Core library (zero Python dependencies)
pip install rsvs

# With optional FastAPI server
pip install rsvs[server]

# With development tools
pip install rsvs[dev]

# Everything
pip install rsvs[all]
```

The Rust engine is compiled into the wheel via PyO3 — no separate Rust toolchain needed at install time.

---

## 30-Second Example

```python
from rsvs import Rsvs

r = Rsvs()
r.ingest("Raja adalah pemimpin kerajaan laki-laki. "
         "Ratu adalah pemimpin kerajaan perempuan. "
         "Tahta tertinggi ada di kerajaan.")

r.compose("raja", [("tahta_tertinggi", 0), ("laki_laki", 0), ("kerajaan", 0)], lang="id")
r.compose("ratu", [("tahta_tertinggi", 0), ("perempuan", 0), ("kerajaan", 0)], lang="id")

sim = r.structural_similarity("raja", "ratu")
# structural_similarity = 0.667 (2 shared, 1 differing)

sub = r.substitution_analysis("raja", "ratu")
# Substitution: laki_laki → perempuan
```

One swap — `laki_laki` becomes `perempuan` — is the entire semantic difference between king and queen, expressed as a precise structural transformation rather than a fuzzy vector distance.

---

## Features

- **Compositional sense definitions** — every concept is a composition of other concepts, down to 24 epistemological seed atoms
- **Structural similarity** — compare sense-level composition structures, not just atom sets
- **Substitution analysis** — find the precise swaps that transform concept A into concept B
- **Multi-sense disambiguation** — native support for polysemy with automatic sense induction
- **Autonomous tiered memory** — nodes progress through New → Candidate → Stable → Deprecated lifecycle
- **MCTS reasoning** — Monte Carlo Tree Search for complex disambiguation with UCB1 selection
- **Convergence detection** — automatically detect when nodes from different languages represent the same concept
- **Bahasa Indonesia first** — primary development and testing language, with full language-agnostic architecture
- **Python-first API** — `from rsvs import Rsvs`, typed with `.pyi` stubs
- **Rust core** — all computation in Rust, compiled to Python via PyO3/maturin
- **Zero dependencies** — `pip install rsvs` gives you the engine with no extra packages
- **Optional FastAPI server** — production-ready HTTP API with auth, rate limiting, CORS

---

## Citation

If you use RSVS in your research, please cite it:

```bibtex
@software{rsvs2026,
  title = {RSVS: Recursive Symbolic Vocabulary System},
  author = {Wolfvin},
  year = {2026},
  version = {8.3.0},
  url = {https://github.com/Wolfvin/SymbolicPuzzle3D}
}
```

---

## License

Dual-licensed under [MIT](https://github.com/Wolfvin/SymbolicPuzzle3D/blob/main/LICENSE) OR [Apache-2.0](https://github.com/Wolfvin/SymbolicPuzzle3D/blob/main/LICENSE). You may choose either license at your option.
