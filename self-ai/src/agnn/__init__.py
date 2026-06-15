"""AGNN — Aphantic Graph Neural Network.

A lightweight GNN for composable knowledge memory in small language models.

AGNN replaces the previous bge-m3 + hook injection approach with a graph
neural network that learns directly from the structure of a typed knowledge
graph. Instead of retrieving embeddings and injecting them into a language
model's hidden states, AGNN composes knowledge by performing message passing
over typed edges (subject → predicate → object), enabling multi-hop reasoning
chains that emerge from the graph topology itself.

Modules:
    graph           — Typed knowledge graph (subject, predicate, object)
    traversal       — Multi-hop graph traversal → reasoning chain
    message_passing — GNN neighborhood aggregation
    embeddings      — Model-native embedding extraction (no bge-m3)
    adapter         — Portability: auto-detect hidden_size, num_layers from model.config
"""
