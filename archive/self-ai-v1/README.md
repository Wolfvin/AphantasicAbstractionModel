# Self-AI v1 Archive

This directory contains archived code from Self-AI v1 — the original architecture based on:
- **bge-m3** for semantic embedding and retrieval
- **Hook injection** (UnconsciousInjector) for activation steering
- **Projection matrix** (ProjectionTrainer) for directional alignment
- **FastAPI server** for HTTP API

These components have been superseded by AGNN (Aphantic Graph Neural Network),
which replaces the embedding+injection approach with graph-structural reasoning.

## Archived Components

| Component | Original Location | Description |
|-----------|-------------------|-------------|
| `unconscious/` | `src/unconscious/` | Hook injection, projection trainer, training pairs |
| `api/` | `src/api/` | FastAPI server (answer, learn, reinforce, introspect) |
| `test_projection_training.py` | `tests/` | Projection trainer tests |
| `test_multi_injection.py` | `tests/` | Multi-node injection tests |
| `benchmark_steering_quality.py` | `tests/` | Activation steering benchmark |
| `benchmark_l2_multi_injection.py` | `tests/` | L2 multi-injection benchmark |
| `test_unconscious_injection.py` | `tests/` | End-to-end injection test |
| `benchmark_contextual_injection.py` | `tests/` | Contextual injection benchmark |
| `poc_v1.1.py` | Root | TrainingAgent v1.1 proof of concept |
| `requirements-api.txt` | Root | FastAPI + uvicorn dependencies |
| `agent-ctx/` | `agent-ctx/` | Phase 1-3 implementation context |
| `docs-plans/` | `docs/plans/` | Training agent v1 plan |
| `benchmark/` | `benchmark/` | Empirical benchmarks + results |

This archive is kept for reference. Do not import from it in new code.
