# Tutorials

Learn RSVS step by step. Each tutorial is a self-contained, runnable example.

---

## Getting Started

| Tutorial | Description | Time |
|----------|-------------|------|
| [Basic Usage](basic-usage.md) | Ingest text, query concepts, compare similarity, inspect the graph | 5 min |
| [Indonesian NLP](indonesian-nlp.md) | Build a Bahasa Indonesia knowledge graph with domain-specific corpora | 10 min |
| [Composition Demo](composition-demo.md) | Create explicit compositions, analyze structural similarity and substitutions | 8 min |
| [Structural Reasoning](structural-reasoning.md) | MCTS reasoning, context-aware queries, consolidation and reflection | 12 min |

---

## Prerequisites

All tutorials assume you have RSVS installed:

```bash
pip install rsvs
```

No other dependencies are required. The Rust engine is compiled into the wheel.

---

## What You Will Learn

By the end of these tutorials, you will understand how to:

1. **Ingest** raw text and build a knowledge graph
2. **Query** concepts with context-aware disambiguation
3. **Compose** explicit sense definitions from other senses
4. **Compare** concepts with structural similarity and substitution analysis
5. **Reason** with MCTS and depth-controlled traversal
6. **Maintain** the graph with consolidation, reflection, and verification
7. **Persist** the graph to disk and reload it later
