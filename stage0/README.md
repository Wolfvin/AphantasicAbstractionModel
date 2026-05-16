# stage0 — AAM Rule-Based System (v1.0.0)

This folder contains the **complete rule-based architecture** of the Aphantasic Abstraction Model (AAM), covering the full pipeline from ingest through output and reasoning.

## Architecture

```
stage0/
├── layer0/              Perceptual Front-End (ingest)
│   ├── base.py          PerceptualTuple, ModalityType, base classes
│   ├── text.py          TextAbstractor (LLM-driven tuple extraction)
│   ├── image.py         ImageAbstractor (vision bridge)
│   ├── video.py         VideoAbstractor (frame sampling + temporal)
│   ├── audio.py         AudioAbstractor (Whisper STT pipeline)
│   └── adapter.py       L0→L1 bridge (PerceptualObservation → RSVS ingest)
│
├── layer1/              Rust Core + PyO3 Bridge
│   ├── crates/rsvs-core/  ALL Rust code (v12 DAG pipeline engine)
│   │   └── src/v12/       13-transform pipeline with 6 unified abstractions
│   ├── Cargo.toml        Workspace config
│   ├── pyproject.toml    Maturin build config
│   └── Makefile          Build, test, lint targets
│
├── layer2/              Cognitive Runtime (reasoning)
│   ├── bridge.py        V12PipelineBridge (PyO3 ↔ Python adapter)
│   ├── pipeline.py      GeniusPipeline (wire layers)
│   ├── pattern.py       Pattern completion + narrative
│   ├── predictive.py    Predictive coding engine
│   ├── temporal.py      Temporal tracking
│   ├── context.py       Context layer (internet search + scope)
│   ├── situation.py     Situation layer (chat as semantic memory)
│   ├── scope_control.py Hierarchical scope management
│   ├── policy_engine.py Rule-based compliance checking
│   ├── policy_rule_compiler.py  Compile graph patterns → PolicyRules
│   ├── coder_layer.py   Code understanding as structured knowledge
│   ├── embedding.py     Pluggable vector embeddings
│   ├── llm.py           LLM bridge (narrative FROM graph)
│   └── ...              (persistence, prediction_loop, hypothesis_combinator, etc.)
│
├── layer3/              Deductive Reasoning & Output
│   ├── reasoning.py     Deductive chain builder (5-step)
│   ├── lattice.py       Possibility lattice (dynamic hypothesis space)
│   ├── hypothesis.py    Hypothesis-driven active reasoning
│   ├── policy.py        DeductivePolicyEngine (RSVS-enhanced compliance)
│   └── coder.py         DeductiveCoderLayer (RSVS-enhanced code analysis)
│
├── pipeline.py          AamPipeline — wires ALL layers into one system
│
├── validation_gates/    5-Pillar Validation Gates
│   ├── signal_extraction.py      Gate 1: L0/L1 noise filter
│   ├── regime_detection.py       Gate 2: L2 cognitive regime
│   ├── uncertainty_calibration.py Gate 3: L3 confidence calibration
│   ├── statistical_edge.py       Gate 4: L4 positive EV check
│   └── execution_discipline.py   Gate 5: L5 bounded execution
│
├── python/              Python rsvs package (API, CLI, server)
│   └── rsvs/            Installable package (pip install rsvs)
│
└── __init__.py          Package init (auto-adds stage0/ to sys.path)
```

## Data Model

AAM uses **v12 as the sole data model**. The v12 architecture replaces all prior models (v8.x, v11.x) with 6 unified abstractions:

1. **SemanticAtom** — Universal ingest primitive
2. **Composition** — Universal structured grouping
3. **LifecycleState** + **EpistemicState** — Two orthogonal status axes
4. **SemanticEdge** — Single typed triple
5. **Transform** — Declarative transform graph
6. **SeedPrimitive** + seed scores — Seed-driven epistemic confidence

### v8.3 Backward Compatibility

v8.3 snapshots can be migrated to v12 using `Persistence::migrate_v83()`:
- v8.3 nodes → v12 SemanticAtoms with appropriate lifecycle states
- v8.3 senses → v12 Compositions (type: Hypothesis)
- v8.3 edges → v12 SemanticEdges (with synthetic compositions for node-to-node links)
- Seed atoms get `LifecycleState::Stable` automatically

## How Imports Work

All internal imports (`from layer2.bridge import ...`) work because `stage0/__init__.py` automatically adds the `stage0/` directory to `sys.path`. This means:

```python
# From OUTSIDE stage0/ (repo root, examples, etc.):
import stage0                     # Adds stage0/ to sys.path
from layer2.bridge import get_bridge
from pipeline import AamPipeline

# Or use the convenience re-export:
from stage0 import AamPipeline

# From INSIDE stage0/ (any subdirectory):
# Each file has a sys.path setup that walks up to find stage0/
from layer2.bridge import get_bridge  # Works automatically
```

## Version

**1.0.0** — First stable rule-based release. Self-learning (ReflexiveLayer) is planned for a future stage.

## Test

```bash
# Rust core tests (282 tests: 132 unit + 8 integration + 142 validation)
cd stage0/layer1 && cargo test --features v12

# Lint (0 warnings expected)
cd stage0/layer1 && cargo clippy --features v12

# Doc generation (0 warnings expected)
cd stage0/layer1 && cargo doc --features v12 --no-deps

# Release build
cd stage0/layer1 && cargo build --release --features v12

# Python end-to-end tests
cd stage0 && python test_e2e_mind_only.py
```
