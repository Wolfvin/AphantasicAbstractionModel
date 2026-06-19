# Research: Structured Prediction & Graph-Based Sequence Modeling — Validation and Borrow Opportunities for AGNN

> **Worker 1** of a 4-worker research batch.
> **Focus:** Structured prediction (CRFs, GNNs for NLP, message-passing
> networks) as alternatives to transformer next-token prediction, evaluated
> against AGNN's existing typed-edge + NeuralReplay message-passing design.
> **Status:** Research note (no code changes).
> **Author date:** 2026-06-20.

---

## 0. Scope and method

AGNN's core philosophy is that the **graph reasons** (typed RelationType
edges, BA44 deductive rules, LIF spiking via PurkinjeCell, NeuralReplay
message passing), while a **small Qwen3-0.6B model only articulates**
the answer. This deliberately rejects the transformer next-token
paradigm where the model itself does both reasoning and generation.

This document surveys the literature on **structured prediction** and
**graph-based sequence modeling** to answer two questions:

1. Does the literature validate AGNN's graph-reasoning-instead-of-attention
   approach, or is it idiosyncratic?
2. What specific techniques could AGNN borrow to strengthen its
   existing typed-edge + NeuralReplay design?

Method: web search across 13 queries (~90 hits) covering CRFs, GNNs for
NLP, MPNNs, R-GCNs, spiking GNNs, and non-autoregressive generation.
Page-reader was unreliable for full PDFs, so the analysis leans on
abstracts + snippets plus a full read of AGNN's `plasticity/neural_replay.py`.

---

## 1. Key techniques found

### 1.1 Conditional Random Fields (CRFs) for structured prediction

**Principle.** A CRF (Lafferty, McCallum, Pereira 2001) models
`P(y|x)` *globally* over the entire label sequence as a single
normalized distribution, with the partition function `Z(x)` computed
over **all** candidate labelings via forward-backward / Viterbi. CRFs
were invented to fix the **label bias problem** of MEMMs, which
normalize *locally* at each step. A linear-chain CRF uses pairwise
potentials between adjacent labels; general / skip-chain CRFs use
arbitrary graphical structure.

**Key sources:**
- Lafferty, McCallum, Pereira (2001), "Conditional Random Fields:
  Probabilistic Models for Segmenting and Labeling Sequence Data"
  — https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf
- Label-bias explainer (Awni Hannun) — https://awni.github.io/label-bias
- aman.ai primer — https://aman.ai/primers/ai/conditional-random-fields

**AGNN alignment — strong, partial.** The single most important idea
AGNN shares with CRFs is the rejection of **local, greedy per-step
decoding** in favor of **globally-coherent structure**. AGNN's
`NeuralReplay.pass_messages()` runs LIF dynamics over the *entire*
graph for 10 timesteps before any node embedding is read out — there
is no left-to-right "next token" sweep; the whole sub-graph settles
together. That is the CRF intuition translated to the graph domain.

However, AGNN currently lacks the formal CRF ingredient that makes
global coherence rigorous: an explicit partition function / energy
normalization that makes the joint distribution sum to 1. AGNN's
`_RELATION_AGGREGATION_WEIGHTS` (CATEGORICAL=1.0, CAUSAL=0.7,
DIFFERENTIAL=-0.8) play the *role* of CRF transition potentials (they
encode per-relation "compatibility" between connected nodes), but they
are hand-set scalars, not learned, globally-normalized potentials.

### 1.2 Graph Neural Networks for NLP (Marcheggiani-Titov edge-gated GCNs)

**Principle.** Represent a sentence as a graph (dependency tree,
semantic-role graph, AMR graph) and run message-passing so each
token's representation is updated by aggregating features from its
graph neighbors — not from a flat positional context window. The
canonical NLP instantiation is Marcheggiani & Titov (EMNLP 2017),
which generalizes GCNs to **directed, labeled graphs** by (a) using
separate parameters for incoming vs. outgoing edges (direction), and
(b) **edge-wise gating** — a learned scalar gate per edge that lets
the model up-weight informative syntactic edges and down-weight noisy
ones.

**Key sources:**
- Marcheggiani & Titov (2017), "Encoding Sentences with Graph
  Convolutional Networks for Semantic Role Labeling"
  — https://aclanthology.org/D17-1159 ; arXiv https://arxiv.org/abs/1703.04826
- Bastings et al. (2017), "Graph Convolutional Encoders for
  Syntax-aware NMT" — https://aclanthology.org/D17-1209.pdf
- Zhang et al. (2018), "Graph Convolution over Pruned Dependency Trees
  Improves Relation Extraction" — https://nlp.stanford.edu/pubs/zhang2018graph.pdf

**AGNN alignment — very strong; this is essentially AGNN's design
philosophy, validated.** AGNN's typed-edge graph
(CATEGORICAL/CAUSAL/DIFFERENTIAL/FUNCTIONAL/TEMPORAL/SPATIAL/DISCURSIVE)
is a *richer-typed* version of the directed, labeled graphs
Marcheggiani-Titov use. Two specific design choices AGNN has already
made map directly onto the literature:

1. **Typed edges** ↔ Marcheggiani-Titov's labeled edges (one param set
   per edge label). AGNN's `TypedEdge` carrying `confidence + role +
   context` is a superset of that schema.
2. **Per-type propagation weights** (`_SPREAD_DECAY`,
   `_RELATION_AGGREGATION_WEIGHTS`) ↔ R-GCN's per-relation weight
   matrices (§1.4) and edge-wise gating.

What AGNN does that the NLP-GNN literature mostly does *not*: it
replaces the smooth GCN aggregation with **discrete LIF spiking**
(`PurkinjeCell`), so the message-passing is *binary-spike* rather
than *continuous-embedding*. This is closer to **Spiking GNNs** (§1.5)
than to vanilla GCN-NLP. The downside: AGNN's spike aggregation in
`aggregate_spikes()` uses a fixed sinusoidal-carrier heuristic, which
is idiosyncratic — the literature uses learned transformations.

### 1.3 Message Passing Neural Networks (MPNNs) — Gilmer et al. 2017

**Principle.** Gilmer et al. (ICML 2017) unified a dozen GNN variants
(GGNN, NeuralFP, etc.) into one framework with four phases: (1)
**Message** — each edge computes a message from source + target + edge
features; (2) **Aggregate** — each node sums/pools messages from its
neighbors; (3) **Update** — node state `h_v` is updated from its old
state + aggregated message; (4) **Readout** — a global pooling produces
the graph-level output. Iterating message passing `T` times lets
information propagate `T` hops, so the readout is a function of the
graph's **global topology**, not just local neighborhoods — enabling
globally-coherent predictions without a left-to-right autoregressive
sweep.

**Key sources:**
- Gilmer et al. (2017), "Neural Message Passing for Quantum Chemistry"
  — https://arxiv.org/abs/1704.01212 ;
  PDF https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf
- NeurIPS 2020, "Building powerful and equivariant GNNs with any
  scalar" — https://proceedings.neurips.cc/paper/2020/file/a32d7eeaae19821fd9ce317f3ce952a7-Paper.pdf

**AGNN alignment — strong, with a structural mismatch.** AGNN's
`NeuralReplay` is, in MPNN terms, a spiking variant of the
message/aggregate/update loop:

- `_topology_currents()` aggregates edge confidences (the "message"),
- `PurkinjeCell` integrates-and-fires (a biophysical "update"),
- `pass_messages()` writes the new embeddings (the "readout" applied
  per-node).

Three structural notes:

- **AGNN's "message" is input-current only, computed once and held
  constant** for 10 LIF steps. Standard MPNNs recompute the message
  at every step from the *current* neighbor states — AGNN's spike
  train therefore encodes topology + initial current, not dynamic
  neighbor evolution. This is a real deviation: AGNN's message
  passing is **single-shot, fixed-input**, more like a fixed-point
  iteration of a static input than an iteratively-refined message
  exchange.
- **AGNN uses binary spikes, not continuous messages.** The MPNN
  framework is agnostic to this (it allows any message function),
  so it's legitimate, but it trades representational density for
  biological plausibility.
- **AGNN's `timesteps=10`** is the MPNN "T" (propagation depth) —
  exactly the right conceptual framing.

### 1.4 Relational Graph Convolutional Networks (R-GCNs)

**Principle.** Schlichtkrull et al. (2018) extends GCNs to
**multi-relational** graphs (knowledge graphs with dozens-to-hundreds
of edge types) by giving each relation type `r` its own weight matrix
`W_r` in the aggregation. Because `W_r` per relation blows up
parameters, they introduce two regularizations: (1) **basis
decomposition** — `W_r = Σ_b a_{rb} B_b`, sharing basis matrices
across all relations with only scalar coefficients per relation; (2)
**block-diagonal decomposition** — `W_r` is a stack of small
block-diagonal matrices. R-GCNs achieved SOTA on link prediction and
entity classification on FB15k / WN18.

**Key sources:**
- Schlichtkrull et al. (2018), "Modeling Relational Data with Graph
  Convolutional Networks" — https://arxiv.org/abs/1703.06103
- PMC "R-GCNs: a closer look" — https://pmc.ncbi.nlm.nih.gov/articles/PMC9680895
- DGL R-GCN tutorial — https://www.dgl.ai/dgl_docs/tutorials/models/1_gnn/4_rgcn.html

**AGNN alignment — extremely strong; AGNN is, structurally, a spiking
R-GCN variant.** AGNN's 7 RelationTypes are a (small, hand-curated)
relation set in exactly the R-GCN sense, and
`_RELATION_AGGREGATION_WEIGHTS` per type are a degenerate
(single-scalar, not matrix) version of R-GCN's per-relation `W_r`.
The DIFFERENTIAL=-0.8 negative weight is precisely the kind of
sign-carrying relational message that R-GCN's learned `W_r` would
discover automatically. **Honest gap:** AGNN's per-type "weight" is
one scalar, whereas R-GCN learns a full matrix per relation (then
compresses it). AGNN therefore cannot represent *direction-dependent*
transformations within a relation type — all CATEGORICAL edges mix
identically regardless of which two nodes they connect.

### 1.5 Spiking Graph Neural Networks (SGNNs)

**Principle.** A nascent but real literature embeds Leaky
Integrate-and-Fire (LIF) neurons into GNN message passing: each node
is a spiking neuron, spikes propagate along edges, and the spike
train encodes the graph's topology. The key inductive bias is that
**information lives in spike *timing/frequency*, not amplitude** — so
two graphs with identical topology but different edge confidences
produce distinguishable spike trains.

**Key sources:**
- SGNNBench (2025) — https://arxiv.org/html/2509.21342v1
- SpikeNet / HASNN (LIF for dynamic graphs) —
  https://scdm-shu.github.io/papers/2026-KBS-HASNN.pdf
- NeurIPS 2024 Spiking GNN on Riemannian Manifolds —
  https://neurips.cc/virtual/2024/poster/94910
- snnTorch LIF tutorial (formulation AGNN matches) —
  https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html

**AGNN alignment — validates AGNN's most idiosyncratic-seeming choice.**
AGNN's `PurkinjeCell`-driven `NeuralReplay` is, to a first
approximation, a hand-built SGNN. The LIF equation in `neural_replay.py`
matches the textbook formulation in snnTorch and the SpikeNet-style
papers. So while "spiking GNN" sounds exotic relative to transformers,
it is a *recognized research direction*, not an invention of the AGNN
authors. The genuine novelty AGNN adds on top is (a) the
**biological-naming layer** (PurkinjeCell, sharp-wave-ripple,
hippocampal replay) and (b) the **typed-edge + confidence weighting**
of the input current `_topology_currents()`. The honest concern: SGNNs
are still mostly benchmarked on node/graph classification (citation
networks, QM9), **not** on language generation or reasoning chains, so
AGNN is applying a validated *mechanism* in a much harder *domain* than
where it has been shown to work.

### 1.6 Energy-Based Models / Structured Prediction Energy Networks (SPENs)

**Principle.** LeCun's EBM tutorial (2006) and Belanger & McCallum
(ICML 2016, SPENs) define structured prediction as energy
minimization: learn an energy `E(x,y)` and predict
`ŷ = argmin_y E(x,y)`. The energy is a deep network over structured
`y`, and inference is gradient descent to the lowest-energy
configuration. This subsumes CRFs (a CRF is an EBM with
`E = -log·unnormalized-score`) and directly produces
**globally-coherent** outputs because the whole `y` is optimized
jointly rather than decoded token-by-token.

**Key sources:**
- LeCun, "A Tutorial on Energy-Based Learning" (2006) —
  http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf
- Belanger, Yang, McCallum, "Structured Prediction Energy Networks"
  (ICML 2016) — https://proceedings.mlr.press/v48/belanger16.html ;
  arXiv https://arxiv.org/abs/1511.06350

**AGNN alignment — conceptually validating, formally absent.** AGNN's
whole pitch — "the graph does the reasoning, the small LM just
articulates" — is an EBM story in disguise: the graph + LIF dynamics
define an implicit energy landscape, and `pass_messages()` is one
(gradient-free, biophysical) step of "settling" toward a low-energy
configuration before readout. But AGNN has **no explicit energy
function** and no `argmin` procedure; the spike-train→embedding map
is a fixed aggregation, not a learned minimization. This is the
deepest theoretical bridge available, and also the largest formal gap.

---

## 2. Specific actionable borrow opportunities for AGNN

Listed roughly in order of "effort to implement" × "expected payoff."

### B1. Adopt CRF-style global normalization over candidate reasonings

AGNN currently has no notion of a partition function — `pass_messages()`
produces embeddings deterministically. Borrowing CRF global
normalization would mean: given a query, enumerate the top-k candidate
reasoning chains (typed-edge paths through the EngramComplex), score
each with `exp(Σ edge-weight × edge-confidence)`, normalize by
`Z(query)=Σ over all candidates`, and read out the argmax. This gives
a principled answer to "which chain did the graph actually endorse?"
instead of implicitly trusting whatever embeddings settled.

- **Effort:** low (operates over existing `_RELATION_AGGREGATION_WEIGHTS`
  + `TypedEdge.confidence`).
- **Payoff:** high — converts AGNN from a deterministic embedding
  refiner into a globally-normalized structured predictor (the formal
  property that distinguishes CRFs from MEMMs).

### B2. Replace AGNN's per-type scalar weights with R-GCN per-relation weight matrices (with basis decomposition)

Today `_RELATION_AGGREGATION_WEIGHTS[CATEGORICAL]=1.0` is one float.
R-GCN's basis decomposition `W_r = Σ_b a_{rb} B_b` lets AGNN keep the
7-type structure but learn a small number (e.g. 4) of shared `B_b`
basis matrices + 7×4 scalar coefficients, instead of either (a) one
scalar per type or (b) a full matrix per type. This is the *exact*
literature-validated mechanism for the typed-edge design AGNN already
has, and it directly upgrades `DIFFERENTIAL=-0.8` (a hand-tuned
sign-and-magnitude) into a *learned* directional mixing.

- **Effort:** medium (needs torch; `neural_replay.py` is currently
  pure numpy by constraint).
- **Payoff:** high — closes the single biggest structural gap vs. R-GCN.

### B3. Add edge-wise gating (Marcheggiani-Titov) to `_topology_currents()`

Currently `_topology_currents()` sums `edge.confidence` for every
incident edge, treating all edges of a given type identically.
Marcheggiani-Titov's edge-wise gate `g_e = σ(w·[h_u, h_v, e_role])`
per edge would let AGNN **down-weight noisy/low-relevance edges** at
runtime rather than relying solely on the static `confidence` field.

- **Effort:** low.
- **Payoff:** medium — the literature shows 0.3-0.6 F1 gains from
  gating (note Zhang et al. 2018 found gating *hurt* on pruned trees,
  so A/B test).

### B4. Make message passing iterative (MPNN-style) rather than single-shot

Today `replay()` computes `input_currents` once and drives the LIF
cells with that *fixed* input for 10 steps. Standard MPNNs recompute
the message each step from the *current* neighbor states. AGNN could,
at each LIF timestep `t`, recompute each node's input current as a
function of which neighbors spiked at `t-1` — turning the replay into
true iterative message exchange rather than fixed-input integration.
This is the difference between "each node integrates its static
topology" and "the graph collectively computes."

- **Effort:** medium-high (restructures the inner loop).
- **Payoff:** high — this is the move that makes AGNN a *bona fide
  MPNN* rather than a per-node LIF simulator.

### B5. Add an explicit energy / readout head (EBM framing)

Define `E(query, answer) = -<aggregate_spikes(readout_node),
answer_embedding>` and select the answer that minimizes energy over
the LM's top-k candidates. This converts the Qwen3-0.6B "articulator"
from a next-token generator into an energy-minimizing decoder, which
is exactly the EBM/SPEN story and the cleanest theoretical statement
of "the graph does the reasoning."

- **Effort:** medium.
- **Payoff:** high theoretical payoff.

### B6. Position AGNN in the "Continuous Generation" taxonomy of non-autoregressive methods

The 2025 survey (arXiv:2509.24435) categorizes non-autoregressive
methods into 5 families, one of which — *Continuous Generation*
(diffusion / flow-matching / EBM over the whole sequence) — is
conceptually what AGNN's settle-then-articulate loop is doing.
Explicitly positioning AGNN in this taxonomy (rather than as
"anti-transformer") would (a) give it a defensible academic home and
(b) surface concrete techniques (iterative refinement schedules,
parallel decoding) AGNN could import.

- **Effort:** zero code; framing only.
- **Payoff:** high framing payoff.

### B7. Validate the LIF-on-graph mechanism against the SGNN literature

AGNN's `PurkinjeCell`/`NeuralReplay` is functionally a SpikeNet-style
SGNN. Cite that lineage explicitly (SGNNBench, SpikeNet, NeurIPS-2024
manifold spiking GNN) and, ideally, run AGNN's `neural_replay.py` on
a standard SGNN benchmark (Cora, Citeseer) to confirm the mechanism
is competitive *as a GNN* before claiming it as a reasoning engine.

- **Effort:** low for the citation; medium for the benchmark.
- **Payoff:** high credibility payoff.

---

## 3. Theoretical validation — does the literature support AGNN's approach?

**Honest assessment: yes for the *mechanisms*, unproven for the
*ambition*.**

### Where the literature squarely validates AGNN

- **Typed-edge message passing as a reasoning substrate** is exactly
  R-GCN (Schlichtkrull 2018) + Marcheggiani-Titov (2017). AGNN's 7
  RelationTypes are a legitimate, small relation vocabulary in the
  R-GCN sense, and per-type propagation weights are a (simplified)
  form of R-GCN's per-relation matrices. *This is not idiosyncratic;
  it is a well-established GNN design.*
- **Spiking-on-graph (LIF neurons doing the message passing)** is a
  real, if young, research line (SpikeNet, SGNNBench, NeurIPS-2024).
  AGNN's `PurkinjeCell` matches the textbook LIF equation. *Not
  idiosyncratic, but the literature has not validated SGNNs on
  language reasoning — only on node/graph classification.*
- **Globally-coherent structured prediction instead of local
  autoregressive decoding** is the founding motivation of CRFs
  (Lafferty 2001) and EBMs (LeCun 2006, Belanger 2016). AGNN's "the
  graph settles before readout" is the *spirit* of CRF/EBM global
  normalization, even though AGNN lacks the formal partition
  function / argmin.
- **"Attention/local-softmax is not the only way to produce text"**
  is itself now a surveyed position: the 2025 "Alternatives to Next
  Token Prediction" survey (arXiv:2509.24435) formally categorizes
  5 families of non-NTP methods, and AGNN's settle-then-articulate
  fits the "Continuous Generation" family.

### Where the literature does *not* (yet) support AGNN

- **No published result shows a spiking, typed-edge GNN beating or
  even matching a transformer on language reasoning/generation.** The
  GNN-for-NLP literature (Marcheggiani-Titov, Bastings, Zhang) uses
  GNNs as *encoders that augment* a transformer/BiLSTM, not as
  replacements for it. AGNN's stronger claim — that the *graph
  replaces attention as the reasoning engine* — is ahead of the
  evidence. The closest precedent (R-GCN for multi-hop QA,
  arXiv:2210.06418) still embeds the GNN inside a larger neural stack.
- **AGNN's spike-aggregation heuristic** (`tile + fixed sinusoidal
  carrier + L2-norm` in `aggregate_spikes()`) is *not* found in the
  literature surveyed. Standard SGNNs use learned readouts. This is a
  genuine idiosyncrasy that should either be justified empirically
  or replaced with a learned readout.
- **AGNN's per-type single-scalar weights**
  (`_RELATION_AGGREGATION_WEIGHTS`) are weaker than every comparable
  literature mechanism (R-GCN matrices, edge-wise gates, learned
  potentials). The literature would predict this limits
  expressiveness — though the hand-set signs (DIFFERENTIAL=-0.8) show
  the authors have the right intuition.
- **The biological framing** (hippocampus, Purkinje cells,
  sharp-wave ripples) is *not* load-bearing for the ML validity —
  it's motivational. The SGNN literature achieves the same
  LIF-on-graph mechanism without the neuroscience vocabulary. This
  isn't a weakness, but it shouldn't be cited as theoretical support.

### Bottom line

AGNN is **not idiosyncratic in its mechanisms** — every core component
(typed edges → R-GCN; LIF message passing → SGNNs; global-settling →
CRF/EBM; message-passing-as-reasoning → MPNN) has a literature
pedigree. AGNN **is idiosyncratic, and ahead of the evidence, in its
ambition**: combining all of these into a *transformer-replacement*
for language reasoning, with a fixed (non-learned) spike readout, has
no direct published precedent.

The honest research-program statement is:

> *"AGNN assembles five literature-validated mechanisms (R-GCN typed
> edges, MPNN message passing, SGNN LIF dynamics, CRF/EBM global
> coherence, non-autoregressive generation) into a single
> architecture, and hypothesizes that their composition is sufficient
> for graph-reasoning to replace attention. The components are
> validated; the composition and the replacement claim are not yet."*

The §2 borrow opportunities (especially B1 global normalization, B2
learned R-GCN matrices, B4 iterative messages) are exactly the moves
that would close the gap between AGNN's current state and the
strongest version the literature points toward.

---

## 4. Sources

**Conditional Random Fields / structured prediction**
1. https://en.wikipedia.org/wiki/Conditional_random_field
2. https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf — Lafferty, McCallum, Pereira (2001)
3. https://awni.github.io/label-bias — Label bias explainer
4. https://aman.ai/primers/ai/conditional-random-fields
5. https://arxiv.org/pdf/1910.11555 — Fast Structured Decoding for Sequence Models
6. http://proceedings.mlr.press/v9/do10a/do10a.pdf — Neural CRFs

**GNNs for NLP / message passing**
7. https://aclanthology.org/D17-1159 — Marcheggiani & Titov (2017) ⭐
8. https://arxiv.org/abs/1703.04826 — Marcheggiani-Titov arXiv
9. https://aclanthology.org/D17-1209.pdf — Bastings et al. (2017)
10. https://nlp.stanford.edu/pubs/zhang2018graph.pdf — Zhang et al. (2018)
11. https://zhijing-jin.com/files/papers/GNN4NLP_Survey_2020.pdf — GNN for NLP survey
12. https://github.com/thunlp/gnnpapers — Must-read GNN papers

**MPNN framework**
13. https://arxiv.org/abs/1704.01212 — Gilmer et al. (2017) ⭐
14. https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf — ICML PDF
15. https://proceedings.neurips.cc/paper/2020/file/a32d7eeaae19821fd9ce317f3ce952a7-Paper.pdf — NeurIPS 2020

**Relational GCN**
16. https://arxiv.org/abs/1703.06103 — Schlichtkrull et al. (2018) ⭐
17. https://pmc.ncbi.nlm.nih.gov/articles/PMC9680895 — R-GCNs: a closer look
18. https://www.dgl.ai/dgl_docs/tutorials/models/1_gnn/4_rgcn.html — DGL R-GCN tutorial
19. https://arxiv.org/abs/2210.06418 — R-GCN for Multihop QA

**Energy-based structured prediction**
20. http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf — LeCun, Energy-Based Learning
21. https://proceedings.mlr.press/v48/belanger16.html — SPENs (ICML 2016)
22. https://arxiv.org/abs/1511.06350 — SPENs arXiv

**Spiking GNNs / LIF on graphs**
23. https://arxiv.org/html/2509.21342v1 — SGNNBench
24. https://scdm-shu.github.io/papers/2026-KBS-HASNN.pdf — SpikeNet / HASNN
25. https://neurips.cc/virtual/2024/poster/94910 — NeurIPS 2024 Spiking GNN
26. https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html — snnTorch LIF tutorial

**Next-token-prediction alternatives / non-autoregressive generation**
27. https://arxiv.org/html/2509.24435v1 — "Alternatives To Next Token Prediction In Text Generation — A Survey" (2025) ⭐
28. https://proceedings.neurips.cc/paper_files/paper/2024/file/2aee1c4159e48407d68fe16ae8e6e49e-Paper-Conference.pdf — Diffusion Forcing (NeurIPS 2024)
29. https://github.com/JaydenTeoh/beyond-next-token-prediction — Curated list

⭐ = highest-priority papers for AGNN to cite as primary validation.
