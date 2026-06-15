# AGNN — Aphantic Graph Neural Network

Status: In Development

## Vision

AGNN defines a typed knowledge graph where every piece of knowledge is stored as a structured triple (subject, predicate, object) with semantic type tags. Unlike the previous bge-m3 embedding approach where knowledge was stored as dense vectors and retrieved by cosine similarity, AGNN's graph stores knowledge as structured, interpretable triples. This enables compositional reasoning via multi-hop traversal (not just similarity), explainable reasoning chains (follow the edges), efficient message passing over the graph topology, and type-aware aggregation where different predicates drive different updates.

## Architecture

- **graph.py**: Typed knowledge graph (subject → predicate → object)
- **traversal.py**: Multi-hop reasoning chain generation
- **message_passing.py**: GNN neighborhood aggregation
- **embeddings.py**: Model-native embeddings (no external embedding model)
- **adapter.py**: Plug into any HuggingFace transformer

## Archive

Self-AI v1 code (hook injection, bge-m3 retrieval) ada di `archive/self-ai-v1/`.
