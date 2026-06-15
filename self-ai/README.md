# AGNN — Aphantic Graph Neural Network

> A lightweight GNN for composable knowledge memory in small language models. (In development)

## Status

**In Development** — The project has pivoted from Self-AI v1 (hook injection + bge-m3 retrieval) to AGNN, a Graph Neural Network that learns from graph structure to compose knowledge rather than just retrieve it.

## Architecture

AGNN is built around five core modules:

- **graph.py** — Typed knowledge graph with subject → predicate → object edges
- **traversal.py** — Multi-hop graph traversal for reasoning chain generation
- **message_passing.py** — GNN neighborhood aggregation across the knowledge graph
- **embeddings.py** — Model-native embedding extraction (no external embedding model)
- **adapter.py** — Plug into any HuggingFace transformer; auto-detect `hidden_size` and `num_layers`

## Foundation (from Self-AI v1)

The following modules from Self-AI v1 are preserved as foundation code:

- `src/core/self.py` — The main SELF entity (teach, learn_from_failure)
- `src/derivation/` — Derivation engine, understanding builder, text comprehension, answer handlers
- `src/composition/` — Composition layer for reasoning and articulation

## Archive

Self-AI v1 code (hook injection, bge-m3 retrieval, projection trainer, FastAPI server) has been moved to `archive/self-ai-v1/`.

## Project Structure

```
self-ai/
├── src/
│   ├── core/           # Core SELF entity
│   ├── derivation/     # Derivation engine + understanding graph
│   ├── composition/    # Composition layer (reasoning + voice)
│   ├── agnn/           # AGNN modules (new)
│   │   ├── graph.py
│   │   ├── traversal.py
│   │   ├── message_passing.py
│   │   ├── embeddings.py
│   │   └── adapter.py
│   ├── governance/     # Lifecycle + epistemic state governance
│   ├── grammar/        # Grammar parsing + relation discovery
│   ├── introspection/  # Introspection engine
│   ├── axiom/          # Axiom store
│   ├── calibration/    # Platt scaling
│   └── training/       # Training agent + sessions
├── tests/
│   └── agnn/           # AGNN tests (new)
├── config/             # Thresholds + adaptive parameters
├── data/               # Runtime state + patterns
└── docs/               # Plans + audit records
```
