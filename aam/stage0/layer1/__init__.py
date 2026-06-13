"""
AAM Layer 1 — Abstraction Engine

The RSVS (Recursive Symbolic Vocabulary System) Rust core.
Converts PerceptualObservation tuples into graph delta operations.

This layer is the heart of AAM:
- Atom promotion: new concepts enter the graph
- Sense composition: meaning is built from relations
- Spreading activation: recall via structural connections
- Confidence decay: unused knowledge weakens over time

Do not modify the Rust core or PyO3 bindings directly.
Use layer2/bridge.py as the adapter interface.
"""
