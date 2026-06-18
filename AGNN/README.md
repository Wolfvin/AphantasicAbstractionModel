# AGNN - Aphantic Graph Neural Network

> **Rebuilding human brain with code.** Every folder, class, and function maps to a real neuroanatomical structure.

This package is the **AGNN skeleton** - a neuroanatomically-named graph neural network foundation
that will eventually replace the existing AGNNCore logic. See `ARCHITECTURE.md` for the full
blueprint (hippocampus, neocortex, limbic system, spiking dynamics, BA 44 deductive reasoning).

## Status

- **Phase:** Skeleton (NotImplementedError stubs)
- **Goal:** Provide directory structure + class/method signatures that downstream PRs will implement
- **Constraint:** Does NOT modify `self-ai/`. `EngramComplex` wraps (not replaces) `AGNNGraph`
  from `self-ai/src/agnn/graph.py`.

## Structure

```
AGNN/
+- core.py                       # AGNNCore - 8 public methods
+- hippocampus/                  # Fast encoding, episodic memory
+- neocortex/                    # Slow learning, semantic memory
+- limbic_system/                # Emotional modulation + confidence
+- diencephalon/                 # Relay + plasticity optimization
+- basal_ganglia/                # Action selection + reinforcement
+- cerebellum/                   # Predictive timing + spike dynamics (LIF)
+- brainstem/                    # Global activation (serotonin, dopamine)
+- commissures/                  # Bidirectional pathways
+- engrams/                      # Memory representations (Episome, Semesome, EngramComplex)
+- plasticity/                   # Learning mechanisms (neural replay, consolidation)
+- circuits/                     # Functional circuits (trisynaptic, Papez, mesolimbic)
+- tests/                        # test_deductive_reasoning.py (4 structural tests)
```

See `ARCHITECTURE.md` for the full neuroanatomical mapping.

## Install

```bash
pip install -r AGNN/requirements.txt
```

## Test

```bash
python -m pytest AGNN/tests/ -v
```

## License

MIT
