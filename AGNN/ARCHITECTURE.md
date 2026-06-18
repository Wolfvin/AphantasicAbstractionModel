# ARCHITECTURE - AGNN (Aphantic Graph Neural Network)

> Blueprint neuroanatomi lengkap. Setiap section menjelaskan region otak yang direplika,
> struktur Python-nya, formula matematis, dan mapping biology <-> code.

---

## 1. Pilar Arsitektur

AGNN disusun dari **tiga pilar neurosains** yang digabung menjadi satu graph neural network:

| Pilar | Asal Biologis | Analog AI |
|---|---|---|
| **Aphantasic representation** | Aphantasia - *conceptual knowledge without visual imagery* | Text + typed edges + embeddings (human-readable) |
| **Spiking message passing** | Sharp-wave ripple di hippocampus; Purkinje cell LIF | LIF neuron per node, sparse spike propagation |
| **Deductive reasoning** | Left Inferior Frontal Gyrus (BA 44 / Broca's area) | Rule-based transitivity, causal chaining, inversion |

Tiga pilar ini disatukan oleh **hippocampus-neocortex loop** untuk memisahkan *fast encoding*
(labile, hippocampal) dari *slow consolidation* (stable, neocortical), dengan **open world
assumption** sehingga graph bisa *infinite expand* tanpa retraining.

---

## 2. Definisi Aphantasia yang Relevan

| Aspek | Aphantasia (Manusia) | AGNN (AI) |
|---|---|---|
| **Definisi** | Absence of voluntary visual mental imagery | Representasi pengetahuan tidak eksplisit visual |
| **Yang muncul** | Pengetahuan konseptual/verbal tentang benda, tanpa gambar | Text + typed edges + embeddings - human-readable |
| **Cognitive route** | "Different neural route to same functional outcomes" | Graph topology yang reason, model kecil yang articulate |
| **Voluntary vs involuntary** | Voluntary imagery absent, involuntary preserved | Voluntary via `traverse()`, unconscious via message passing |

> *"When a person with aphantasia is asked to picture a red apple, nothing visual arrives;
> what arrives instead is conceptual or verbal knowledge about the thing, but no picture."*
> - [aphant.org](https://aphant.org/paper)

---

## 3. Deductive Reasoning di Otak - BA 44

### Neural Dissociation: Deduction vs Induction

| Reasoning Type | Brain Region | Activation Pattern |
|---|---|---|
| **Deduction** | Left inferior frontal gyrus (BA 44) | Rule-based inference: "All A are B, all B are C => all A are C" |
| **Induction** | Left dorsolateral prefrontal (BA 8/9) | Pattern generalization dari episodic memories |
| **Both** | Left lateral PFC + bilateral dorsal frontal + parietal + occipital | Shared working memory + attention |

> *"Greater involvement of left inferior frontal gyrus (BA 44) in deduction than induction"*
> - [PubMed 15178381](https://pubmed.ncbi.nlm.nih.gov/15178381/)

**BA 44 = Broca's area** - region untuk rule application + syntactic processing.

Pembagian labor di AGNN:
- `neocortex/inferior_frontal_gyrus.py` (BA 44) -> deduction (rule application)
- `neocortex/dorsolateral_pfc.py` (BA 8/9) -> induction (pattern generalization)

---

## 4. Blueprint AGNN v3

```python
class AGNNCoreV3:
    """
    Aphantic Graph Neural Network dengan:
    1. Aphantasic representation: conceptual without visual imagery
    2. Spiking message passing: neural replay untuk consolidate
    3. Deductive reasoning engine: BA 44-inspired rule application
    """

    def __init__(self, model_path):
        self.graph = AGNNGraph()                                   # aphantasic graph
        self.spiking_engine = SpikingMessagePass(tau=0.5, timesteps=10, threshold=1.0)
        self.deductive_engine = DeductiveReasoner(rules=[
            CategoricalTransitivity(),   # A->B (CAT), B->C (CAT) => A->C
            CausalChain(),                # A->B (CAUSAL), B->C (CAUSAL) => A->C (0.7*0.7=0.49)
            DifferentialInversion(),      # A->B (DIFF=-0.8) => B->A (-0.8)
            CausalDifferentialConflict()  # CAUSAL(0.7) + DIFFERENTIAL(-0.8) => conflict
        ])

    def learn(self, question, wrong, correction): ...
    def consolidate(self): ...
    def process(self, question): ...
    def introspect(self): ...
```

---

## 5. Deductive Reasoning Engine - Rule Types

| Rule | Pattern | Inference Weight | Contoh |
|---|---|---|---|
| `CATEGORICAL_TRANSITIVITY` | `A->B (CAT)`, `B->C (CAT)` | `1.0 * 1.0 = 1.0` | "Manusia adalah mamalia, mamalia adalah hewan" |
| `CAUSAL_CHAIN` | `A->B (CAUSAL)`, `B->C (CAUSAL)` | `0.7 * 0.7 = 0.49` | "Smoking -> lung damage -> cancer" |
| `DIFFERENTIAL_INVERSION` | `A->B (DIFF = -0.8)` | `-0.8` (simetris) | "More exercise -> less fat (-0.8)" |
| `CAUSAL_DIFFERENTIAL_CONFLICT` | `A->B (CAUSAL)`, `A->B (DIFF)` | `(0.7 + -0.8) / 2 = -0.05` | Conflict resolution |
| `FUNCTIONAL_COMPOSITION` | `A->B (FUNC)`, `B->C (FUNC)` | `0.6 * 0.6 = 0.36` | Functional chain |

---

## 6. Neuroanatomical Directory Structure

```
AGNN/
+- core.py                       # AGNNCore (main API)
+- hippocampus/                  # Fast encoding, episodic memory
|   +- entorhinal_cortex.py      # Primary input gateway
|   +- dentate_gyrus.py          # Pattern separation (new node creation)
|   +- ca3.py                    # Rapid autoassociation
|   +- ca1.py                    # Context integration
|   +- subiculum.py              # Primary output pathway
+- neocortex/                    # Slow learning, semantic memory
|   +- prefrontal_cortex.py      # Executive functions
|   +- inferior_frontal_gyrus.py # BA 44 (deductive reasoning)
|   +- dorsolateral_pfc.py       # BA 8/9 (inductive generalization)
|   +- association_cortex.py     # Generative network
+- limbic_system/                # Emotional modulation + confidence
|   +- amygdala.py               # Basolateral nucleus (confidence modulation)
|   +- cingulate_gyrus.py        # Conflict detection
|   +- parahippocampal_gyrus.py  # Scene/object recognition
+- diencephalon/                 # Relay + plasticity optimization
|   +- thalamus.py               # Anterior nuclei (consolidation relay)
|   +- mamillary_body.py         # Temporal memory signaling
+- basal_ganglia/                # Action selection + reinforcement
|   +- striatum.py               # Value-based action selection
|   +- globus_pallidus.py        # Output gating
+- cerebellum/                   # Predictive timing + spike dynamics
|   +- purkinje_cell.py          # LIF neuron (spike integration)
|   +- molecular_layer.py        # Spike propagation
+- brainstem/                    # Global activation
|   +- raphe_nucleus.py          # Serotonin (confidence decay)
|   +- tegmentum.py              # Dopamine (reinforcement signal)
+- commissures/                  # Bidirectional pathways
|   +- fornix.py                 # Hippocampo-neocortical pathway
|   +- corpus_callosum.py        # Inter-hemispheric integration
+- engrams/                      # Memory representations
|   +- episodic_engram.py        # Episome (node - specific fact)
|   +- semantic_engram.py        # Semesome (edge - abstract relation)
|   +- engram_complex.py         # Graph (wraps AGNNGraph dari self-ai/src/agnn/)
+- plasticity/                   # Learning mechanisms
|   +- synaptic_plasticity.py    # Confidence -> edge weight
|   +- neural_replay.py          # Spiking message passing
|   +- systems_consolidation.py  # Hippocampus -> neocortex transfer
+- circuits/                     # Functional circuits
|   +- trisynaptic_circuit.py    # EC -> DG -> CA3 -> CA1 -> Sub
|   +- papez_circuit.py          # Episodic memory loop
|   +- mesolimbic_circuit.py     # Confidence-reward loop
+- tests/
    +- test_deductive_reasoning.py
```

---

## 7. Class Inventory - 100% Biologis

| Level | Nama Biologis | Analog |
|---|---|---|
| **Folder** | `hippocampus/`, `neocortex/`, `limbic_system/`, `diencephalon/` | Brain regions |
| **Class** | `DentateGyrus`, `CA3`, `InferiorFrontalGyrus`, `PurkinjeCell` | Anatomical structures |
| **Function** | `encode_episome()`, `initiate_neural_replay()`, `execute_deductive_inference()` | Neurobiological processes |
| **Data Unit** | `Episome` (episodic), `Semesome` (semantic), `EngramComplex` (graph) | Memory units |
| **Circuit** | `TrisynapticCircuit`, `PapezCircuit`, `MesolimbicCircuit` | Functional pathways |

---

## 8. AGNNCore API - 8 Public Methods

```python
class AGNNCore:
    def __init__(self, model_path: str): ...
    def learn(self, question, wrong, correction) -> dict: ...
    def process(self, question) -> dict: ...
    def introspect(self) -> dict: ...
    def traverse(self, question, max_hops=2) -> str: ...
    def consolidate(self) -> dict: ...
    def reinforce(self, episome_id) -> None: ...
    def penalize(self, episome_id) -> None: ...
```

### Method Spec

| Method | Brain Region Analog | Description |
|---|---|---|
| `learn(question, wrong, correction)` | Hippocampus (Trisynaptic Circuit) | Fast encoding of episodic memory |
| `process(question)` | Neocortex + BA 44 | Retrieve -> Deduce -> Articulate |
| `introspect()` | Aphantasic introspect | Conceptual graph summary (no heatmap) |
| `traverse(question, max_hops=2)` | Fornix bidirectional beam search | Walk the graph along typed edges |
| `consolidate()` | HPC -> NC transfer | Spiking replay + embedding refinement |
| `reinforce(episome_id)` | Mesolimbic dopamine | `+0.1` confidence |
| `penalize(episome_id)` | Raphe serotonin | `-0.1` confidence |

---

## 9. Spiking Dynamics - LIF Formula

Setiap node di graph adalah **Leaky Integrate-and-Fire (LIF) neuron**.

```
Membrane potential:  tau * dU/dt = -(U - U_reset) + I_input
Spike threshold:     S = Theta(U - U_th)
Reset after fire:    U = U_reset  if S = 1
```

| Parameter | Symbol | Default | Meaning |
|---|---|---|---|
| Membrane decay | `tau` | `0.5` | How fast potential leaks |
| Timesteps | `T` | `10` | Neural replay duration |
| Fire threshold | `U_th` | `1.0` | Potential to trigger spike |
| Input current | `I_input` | derived from topology | Spike from neighbor activation |

---

## 10. Memory Units - Episome, Semesome, EngramComplex

| Unit | Brain Analog | Locus | Phase |
|---|---|---|---|
| `Episome` | Episodic memory | Hippocampus | Labile (fast encoding) |
| `Semesome` | Semantic memory | Neocortex | Stable (after consolidation) |
| `EngramComplex` | Memory network | HPC + NC | Both - wraps `AGNNGraph` |

### EngramComplex - WRAP, not replace

`EngramComplex` di `engrams/engram_complex.py` **membungkus** (bukan mengganti) `AGNNGraph`
dari `self-ai/src/agnn/graph.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'self-ai', 'src'))
from agnn.graph import AGNNGraph  # reuse yang sudah ada

class EngramComplex:
    """Connected memory network - wraps AGNNGraph"""
    def __init__(self):
        self._graph = AGNNGraph()  # delegate ke existing implementation
```

---

## 11. Functional Circuits

| Circuit | Flow | Used by |
|---|---|---|
| `TrisynapticCircuit` | EC -> DG -> CA3 -> CA1 -> Sub | `encode_episome()` |
| `PapezCircuit` | HC -> Mamillary -> Anterior Thalamus -> Cingulate -> PHG -> HC | `retrieve_semesome()` |
| `MesolimbicCircuit` | VTA -> Striatum -> PFC -> HC | `reinforce_episome()`, `penalize_episome()` |

---

## 12. Neuroanatomical Mapping Summary

| AGNN Component | Brain Region | Function |
|---|---|---|
| `learn()` | Hippocampus (DG->CA3->CA1) | Fast episodic encoding |
| `consolidate()` | HPC -> NC transfer | Semantic transformation |
| `process()` | Neocortex + BA 44 | Retrieve + deduce + articulate |
| `traverse()` | Fornix | Bidirectional beam search |
| `introspect()` | Aphantasic cognition | Conceptual introspect (no imagery) |
| `reinforce()` | VTA / Tegmentum (Dopamine) | Positive reward |
| `penalize()` | Raphe Nucleus (Serotonin) | Negative penalty |

---

## 13. Referensi Ilmiah

- Aphantasia - [Frontiers in Psychology, 2025](https://www.frontiersin.org/articles/10.3389/fpsyg.2025.1615860)
  - [aphant.org paper](https://aphant.org/paper)
  - [Nature Scientific Reports, 2025](https://www.nature.com/articles/s41598-025-27735-x)
- Deductive reasoning & BA 44 - [PubMed 15178381](https://pubmed.ncbi.nlm.nih.gov/15178381/)
- Hippocampus-Neocortex consolidation - [PubMed 27394150](https://pubmed.ncbi.nlm.nih.gov/27394150/)
- Spiking GNN - [arXiv 2509.21342](https://arxiv.org/html/2509.21342)
- Open world assumption - [patrickm.de](https://patrickm.de/knowledge-graph-open-or-closed-world/)
