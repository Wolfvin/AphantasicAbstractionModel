# Research: Spiking Neural Networks for Reasoning — Validation and Borrow Opportunities for AGNN's No-Backprop Stance

> **Worker 4** of a 4-worker research batch.
> **Focus:** Spiking neural networks (LIF neurons) used for
> reasoning/memory tasks, as alternatives to gradient-descent-trained
> networks. Evaluated against AGNN's `cerebellum/purkinje_cell.py`
> (PurkinjeCell LIF neuron, pure numpy) and `plasticity/neural_replay.py`
> (NeuralReplay message passing, no backprop) plus the
> `reinforce`/`penalize`/`consolidate` local-learning rules.
> **Status:** Research note (no code changes).
> **Author date:** 2026-06-20.

---

## 0. Scope and method

AGNN's spiking layer comprises:

- `cerebellum/purkinje_cell.py` — `PurkinjeCell` class implements LIF
  (Leaky Integrate-and-Fire) neuron in **PURE NUMPY, no torch**.
  Formula: `tau * dU/dt = -(U - U_reset) + I_input`;
  `S = Theta(U - U_th)`; `U = U_reset if S=1`. Default tau=0.5,
  threshold=1.0, u_reset=0.0, timesteps=10. Uses closed-form
  exponential update (more stable than forward-Euler).
- `plasticity/neural_replay.py` — `NeuralReplay` drives a population
  of PurkinjeCells (one per graph node) for `timesteps` steps. Methods:
  `replay()` returns spike matrix (nodes × timesteps),
  `aggregate_spikes()` converts spikes to fresh embeddings (rate-code
  via spike counts), `pass_messages()` updates node embeddings. Graph
  topology provides input currents via `_topology_currents()` (sums
  edge confidences per node, weighted by
  `_RELATION_AGGREGATION_WEIGHTS` per edge type).
- **NO backprop/gradient descent in the spiking layer.** Learning
  happens via:
  1. `reinforce(episome_id)` → +0.1 confidence (mesolimbic dopamine
     analog)
  2. `penalize(episome_id)` → −0.1 confidence (raphe serotonin analog)
  3. `consolidate()` → systems consolidation (hippocampus → neocortex
     transfer) via spiking replay

This is a deliberate alternative to gradient-based training:
biologically-inspired local learning rules + structural plasticity
(graph topology changes) instead of weight updates via backprop.

This document surveys the SNN literature to answer two questions:

1. Is AGNN's no-backprop stance defensible from the literature, or
   naive?
2. What specific techniques could AGNN borrow to strengthen its
   PurkinjeCell + NeuralReplay + reinforce/penalize design?

Method: 9 targeted web searches covering STDP, three-factor learning,
R-STDP, eligibility traces, SNNs for memory/reasoning, neuromorphic
computing, LIF vs ReLU theoretical power, and training-without-backprop.

---

## 1. Key techniques found

### 1.1 Three-Factor Learning Rules (Neuromodulated STDP)

**Principle.** Any synapse update can be written
`dw/dt = F(M, pre, post)`, where `pre`/`post` are *local* factors
(Hebbian/STDP coincidence) and `M` is a *global* modulatory third
factor (dopamine, serotonin, reward prediction error) shared across
many synapses. The local factor decides *which* synapses are eligible
to change; the global factor decides *whether and how much* they
actually change. This separates "candidate" from "commit" and is the
canonical biologically-plausible alternative to backprop credit
assignment.

**Key sources:**
- Frémaux, Sprekeler & Gerstner, "Neuromodulated
  Spike-Timing-Dependent Plasticity, and Theory of Three-Factor
  Learning Rules," Front. Neural Circuits 9:85 (2016) —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313
- Mazurek et al., "Three-Factor Learning in Spiking Neural Networks,"
  arXiv:2504.05341 (2025) — https://arxiv.org/abs/2504.05341

**AGNN alignment — direct, near-exact.** AGNN's
`reinforce(episome_id)` (+0.1, "mesolimbic dopamine analog") and
`penalize(episome_id)` (−0.1, "raphe serotonin analog") *are* a third
factor `M`. The edge confidences are the weights `w`. **The missing
piece is that AGNN has no explicit local factor** — `M` is applied
uniformly rather than gated by pre/post coincidence. This is the
single most important gap.

### 1.2 Reward-Modulated STDP (R-STDP) & the Distal Reward Problem

**Principle.** Izhikevich's "distal reward problem": reward arrives
*seconds after* the spikes that caused the rewarded behavior, so
direct Hebbian coincidence cannot assign credit. Solution: each synapse
maintains a silent **eligibility trace** `e_ij` (a decaying tag of
recent pre→post coincidence). When dopamine later arrives, it
multiplies the trace: `Δw_ij = M(t) · e_ij(t)`. Traces that don't
match the rewarded behavior decay to zero and cause no change. R-STDP
provably performs **stochastic policy-gradient reinforcement learning**
under standard assumptions (Frémaux/Sprekeler/Gerstner 2010;
Legenstein et al. 2008).

**Key sources:**
- Izhikevich, "Solving the distal reward problem through linkage of
  STDP and dopamine signaling," Cereb. Cortex 17:2443 (2007) —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6634722
- Legenstein, Pecevski & Maass, "A Learning Theory for
  Reward-Modulated STDP…," PLoS Comp Biol 4:e1000180 (2008) —
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000180

**AGNN alignment — high.** AGNN's `reinforce`/`penalize` is exactly
the delayed-reward scenario. Without eligibility traces, AGNN cannot
tell *which* edges within an episome produced the outcome — it credits
them all equally. R-STDP is the principled fix (see borrow list).

### 1.3 Eligibility Traces on Behavioral Timescales (NeoHebbian Plasticity)

**Principle.** Experimental review showing synapses hold a silent
eligibility "flag" `e_ij` for seconds, sensitive to pre-spike +
postsynaptic state, that dopamine later converts into actual LTP/LTD.
This is the biological substrate of three-factor rules
("neoHebbian"). Striatal experiments (Shindou et al. 2018) confirm a
*silent* eligibility trace enabling dopamine-dependent plasticity for
RL.

**Key source:**
- Gerstner et al., "Eligibility Traces and Plasticity on Behavioral
  Time Scales: Experimental Support of NeoHebbian Three-Factor
  Learning Rules," Front. Neural Circuits 12:53 (2018) —
  https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00053/full

**AGNN alignment — high.** AGNN's `NeuralReplay.replay()` already
computes the spike matrix (nodes × timesteps) needed to derive `e_ij`.
The infrastructure exists; the trace bookkeeping does not.

### 1.4 Hebbian/STDP + Structural Plasticity for Associative Memory (Attractor Networks)

**Principle.** Memories are stored as attractors: assemblies of
strongly-connected neurons formed by Hebbian learning; partial cues
trigger *pattern completion*. Critically, recent work argues the
classic Hopfield-style fixed-after-learning attractor is biologically
wrong — real memory is *dynamic*, involving **formation,
reinforcement, and forgetting** via ongoing plasticity. Feedforward
projections learn hidden representations; recurrent projections form
associative memory.

**Key sources:**
- Pang et al., "Spiking representation learning for associative
  memories," Front. Neurosci. 18:1439414 (2024) —
  https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1439414/full
- Klinshov et al., "A dynamic attractor network model of memory
  formation, reinforcement and forgetting," PMC10766193 (2023) —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10766193
- Gerstner, *Neuronal Dynamics*, Ch. 17.3 —
  https://neuronaldynamics.epfl.ch/online/Ch17.S3.html

**AGNN alignment — very high; this is the closest analog to AGNN's
whole philosophy.** AGNN already does structural plasticity (graph
topology changes), reinforcement (`reinforce`), and consolidation
replay. The attractor-memory literature validates the *shape* of
AGNN's approach and names the missing capability: **pattern
completion** at recall and **forgetting** (AGNN currently only
reinforces; there is no homeostatic decay).

### 1.5 SNNs for Causal/Graph Reasoning (not message-passing)

**Principle.** SNNs can perform reasoning by encoding propositions as
neuron populations and using STDP + (population / synaptic-delay)
coding to evaluate causal/relational structure — *without* gradient
training. The reasoning emerges from spiking dynamics + local
plasticity, not learned weights.

**Key sources:**
- Zhou et al., "A Brain-Inspired Causal Reasoning Model Based on
  Spiking Neural Networks (CRSNN)," IJCNN 2021 —
  https://ieeexplore.ieee.org/abstract/document/9534102 — *"Causal
  Reasoning SNN … with STDP learning rule and population coding
  mechanism … first time SNN is used to complete causal reasoning
  tasks."*
- BrainCog lecture: Symbolic Representation and Reasoning SNNs over
  knowledge graphs —
  https://www.brain-cog.network/docs/tutorial/13_krr.html
- "Temporal Spiking Neural Networks with Synaptic Delay for Graph
  Reasoning" (uses synaptic *delay* to encode relational structure —
  timing code, not rate).

**AGNN alignment — high.** CRSNN is essentially AGNN's thesis
(reasoning via spiking + STDP, no backprop), validating the approach
exists in the literature. The synaptic-delay line is a warning:
**timing codes** matter for graph reasoning, which AGNN's rate-style
`aggregate_spikes()` (spike counting) ignores.

### 1.6 Theoretical Computing Power: LIF/Spiking vs ReLU/Sigmoid, Rate vs Timing

**Principle.**

(a) Maass 1997: spiking networks are *"the third generation"* and,
measured by neurons needed, are **computationally more powerful** than
sigmoidal (1st/2nd gen) networks; noisy spiking neurons can simulate
arbitrary Boolean circuits and finite automata in real time (Maass &
Orponen, NIPS).

(b) But vanilla LIF has a known insufficiency: it cannot *jointly*
encode spatial intensity (rate) and temporal dynamics (timing) —
recent work generalizes LIF and ReLU as special cases of a richer
Multi-Synaptic-Firing (MSF) neuron.

(c) Rate vs timing: timing of first spikes carries information not
reducible to rate; they are complementary, not either/or.

**Key sources:**
- Maass, "Networks of Spiking Neurons: The Third Generation of Neural
  Network Models," Neural Networks 10(9):1659 (1997) —
  https://www.sciencedirect.com/science/article/pii/S0893608097000117
- Maass & Orponen, "On the Computational Power of Noisy Spiking
  Neurons," NIPS —
  https://proceedings.neurips.cc/paper/1158-on-the-computational-power-of-noisy-spiking-neurons.pdf
- "A multisynaptic spiking neuron…MSF…generalize LIF and ReLU as
  special cases…vanilla LIF struggle to jointly encode
  spatiotemporal dynamics," Nature Comms (2025) —
  https://www.nature.com/articles/s41467-025-62251-6
- Brette / Gerstner Ch. 7.6 on rate-vs-timing codes —
  https://neuronaldynamics.epfl.ch/online/Ch7.S6.html

**AGNN alignment — mixed / cautionary.** Maass validates
SNNs-as-compute-model broadly. But the MSF result is a direct critique
of AGNN's pure-LIF choice: **vanilla LIF (what `PurkinjeCell`
implements) is provably the weakest case**, unable to jointly carry
spatial + temporal codes. AGNN's `aggregate_spikes()` reduces spikes
to counts (rate code), throwing away the timing dimension that
graph-reasoning SNN work (synaptic-delay) relies on. This is the main
theoretical risk to AGNN's reasoning claims.

---

## 2. Specific actionable borrow opportunities for AGNN

Ranked by impact / implementation cost. All preserve the no-backprop
stance.

### B1. Add eligibility traces to `consolidate()`/`reinforce`/`penalize` — highest priority

During `NeuralReplay.replay()`, compute a decaying tag `e_ij` per edge
= STDP-style function of pre-node and post-node spike timing within
the timesteps window (e.g.,
`e_ij = Σ_t exp(-(t_post−t_pre)/τ)·pre(t)·post(t)`). Then change
`reinforce`/`penalize` from a **uniform** ±0.1 over all episome edges
to `Δconfidence_ij = M · e_ij` (clipped to ±0.1). This is a textbook
three-factor rule `Δw = M·e`, grounds the design formally, and
*solves credit assignment locally* using only the spike matrix AGNN
already computes.

- **Cost:** ~20 lines in numpy.
- **Borrowed from:** Frémaux & Gerstner 2016; Izhikevich 2007;
  Legenstein et al. 2008; Gerstner et al. 2018.
- **Effort:** low.
- **Payoff:** highest — turns AGNN from "biological-flavored
  one-factor bump" into "textbook three-factor rule with credit
  assignment."

### B2. Make `reinforce`/`penalize` a true gated rule, not a bump

Currently reward is applied at call-time to whatever edges exist.
With traces (item B1), reward should multiply the *lingering*
eligibility from the most recent replay, even if `reinforce` is called
later — this is precisely Izhikevich's distal-reward solution and
matches AGNN's episodic usage pattern.

- **Cost:** minimal once B1 is in place.
- **Effort:** low.
- **Payoff:** high (solves the distal-reward / delayed-feedback
  problem).

### B3. Add a forgetting / homeostatic term

The dynamic-attractor literature models *reinforcement AND
forgetting*; AGNN only reinforces. Add slow decay (or
`penalize`-driven weakening of *non-replayed* episomes) in
`consolidate()` to prevent runaway confidence inflation and implement
forgetting. This is homeostatic plasticity — fully biologically
grounded, no backprop.

- **Borrowed from:** Klinshov et al. 2023 (dynamic attractor networks
  with formation/reinforcement/forgetting).
- **Effort:** low.
- **Payoff:** medium-high (prevents confidence inflation, adds the
  missing "forget" half of the memory lifecycle).

### B4. Add STDP-style local plasticity inside `consolidate()`

During replay, edges whose pre-node fires *before* post-node within
the window get potentiated; anti-causal pairs get depressed. Purely
local, uses only the existing spike matrix. Turns the hand-tuned ±0.1
into a principled, data-driven local rule and directly mirrors CRSNN
(causal reasoning via STDP).

- **Borrowed from:** Zhou et al. 2021 (CRSNN); Caporale & Dan 2008
  (STDP as Hebbian rule).
- **Effort:** medium.
- **Payoff:** high (replaces ad-hoc confidence delta with
  data-driven local plasticity).

### B5. Add recurrent projection within an episome for pattern completion

`pass_messages()` currently does feedforward aggregation. Add
recurrent self-excitation within an episome's subgraph so partial cue
inputs complete to full episome activation — the hallmark of attractor
memory and AGNN's "recall." The fnins-2024 decomposition (feedforward
for representations, recurrent for associative memory) is a ready
blueprint.

- **Borrowed from:** Pang et al. 2024 (spiking representation
  learning for associative memories).
- **Effort:** medium.
- **Payoff:** high (enables pattern completion, the canonical
  attractor-memory operation).

### B6. Reconsider LIF insufficiency; at minimum acknowledge it

`PurkinjeCell` is vanilla LIF and `aggregate_spikes()` is rate coding.
Two options:

- *(a) cheap* — add a timing-code readout from the spike matrix
  (first-spike latency, inter-spike interval) alongside the count,
  and feed both into `aggregate_spikes()`.
- *(b) structural* — consider the MSF multi-threshold generalization
  if reasoning quality plateaus.

Either keeps numpy-only and backprop-free. Without this, AGNN's
reasoning claims rest on the weakest neuron model in the SNN family.

- **Borrowed from:** MSF (Nature Comms 2025); Brette rate-vs-timing
  analyses.
- **Effort:** low (a) / high (b).
- **Payoff:** high theoretical validity (closes the LIF-insufficiency
  gap).

### B7. (Lower priority / aspirational) Energy story

Neuromorphic hardware yields 10-1000× energy wins for SNNs (Nature
Comms 2024 2D-TFET platform; PNAS 2025; arXiv:2602.02439 edge-AI
framework). AGNN's pure-numpy-on-CPU implementation captures **none**
of this benefit today (dense numpy is less efficient than torch
matmul). The SNN choice is only an energy win *if* ported to
neuromorphic silicon; flag this honestly rather than claiming an
energy advantage prematurely.

- **Borrowed from:** neuromorphic hardware literature (2024-2025).
- **Effort:** N/A (documentation only).
- **Payoff:** honest framing — don't oversell.

---

## 3. Theoretical validation — is AGNN's "spiking + no-backprop + reinforce/penalize" stance defensible?

**Verdict: Defensible in principle, currently naive in implementation
— but cheaply fixable.**

### What the literature supports

- **The paradigm exists and is respected.** Three-factor / R-STDP
  learning is a first-class alternative to backprop for SNNs, with
  strong biological grounding and formal theory. Frémaux/Sprekeler/
  Gerstner (2010, PMC6634722) derived the *conditions under which
  R-STDP succeeds*, and Legenstein et al. (PLoS CB 2008) gave an
  analytic learning theory showing R-STDP **approximates
  policy-gradient RL**. So "no backprop + neuromodulated local
  plasticity" is not fringe — it is the canonical biologically
  plausible learning framework, and AGNN's dopamine/serotonin framing
  is exactly the intended analogy.
- **The reasoning-without-backprop angle is published.** CRSNN
  (IJCNN 2021) does causal reasoning via STDP + population coding
  with no gradient training — a direct precedent for AGNN's thesis.
- **The memory angle is strongly supported.** Dynamic attractor
  networks with Hebbian + structural plasticity implement
  formation/reinforcement/forgetting — the exact operations AGNN
  exposes. The fnins-2024 model shows feedforward+recurrent SNNs
  yield pattern completion and prototype extraction with no backprop.
- **SNNs are theoretically at least as powerful** as sigmoidal nets
  (Maass 1997), so AGNN is not sacrificing raw compute power by
  spiking.

### Where AGNN is currently naive / vulnerable

- **Credit assignment is unaddressed.** Uniform ±0.1 over all episome
  edges ignores *which* edges caused the outcome — the very
  distal-reward problem R-STDP was invented to solve. This is the
  most serious gap and the easiest to fix (borrow B1).
- **No local factor at all.** A three-factor rule needs the local
  (pre/post) factor; AGNN currently has only the global factor.
  Without it, "three-factor" is really "one-factor."
- **Pure LIF + rate coding is the weak case.** MSF (Nature Comms
  2025) shows vanilla LIF cannot jointly encode spatial+temporal
  dynamics; graph-reasoning SNN work uses timing/delay codes. AGNN's
  reasoning claims would be stronger with a timing readout (borrow
  B6).
- **Scale caveat (honest).** Surveys (Mazurek 2025;
  arXiv:2605.15058 local-learning survey) consistently flag
  **scalability** as the main open problem for backprop-free SNN
  training — pure STDP/Hebbian methods lag backprop on deep/bench-
  scale tasks. AGNN sidesteps this somewhat because it is *not*
  trying to learn deep feature hierarchies via weights; it maintains
  a graph memory via structural plasticity + confidence. That is a
  defensible niche (memory/reasoning, not ImageNet), but AGNN should
  not over-claim parity with gradient-trained reasoning systems on
  benchmark accuracy.
- **No forgetting / no homeostasis.** The literature treats forgetting
  as essential; AGNN only reinforces, risking confidence inflation.

### Bottom line

AGNN's stance is defensible *and* aligns with a respected research
line, **provided** it adds:

1. eligibility traces + gated `reinforce`/`penalize` (becoming a
   genuine three-factor rule),
2. a forgetting term,
3. ideally a timing-code readout.

Without those, it is a biologically-flavored but
credit-assignment-blind system. With them, it becomes a legitimate,
citable instance of neoHebbian three-factor learning applied to graph
memory/reasoning — a genuinely novel combination worth writing up.

---

## 4. The key recognition: AGNN's `reinforce`/`penalize` is a third factor in search of a local factor

This deserves to be called out separately because it's the single
highest-leverage insight from the survey.

AGNN's learning rule today:

```python
# core.py
def reinforce(self, episome_id):
    epi = self._find_episome(episome_id)
    epi.confidence = min(1.0, epi.confidence + 0.1)  # uniform +0.1
    self._mirror_confidence_to_graph(epi)

def penalize(self, episome_id):
    epi = self._find_episome(episome_id)
    epi.confidence = max(0.0, epi.confidence - 0.1)  # uniform -0.1
    self._mirror_confidence_to_graph(epi)
```

In three-factor learning terms:

- `M = ±0.1` (the global modulatory factor — dopamine/serotonin
  analog) ✓ present
- `e_ij` (the local eligibility trace — which specific edges are
  candidates for update) ✗ **absent**
- The update is `Δw = M` (one-factor), not `Δw = M · e_ij`
  (three-factor)

**The fix is ~20 lines of numpy:**

```python
# pseudocode for the eligibility trace, computed during NeuralReplay.replay()
def compute_eligibility(spike_matrix, tau_elig=2.0):
    """e_ij = sum over t of exp(-(t_post - t_pre)/tau) * pre(t) * post(t)"""
    # spike_matrix shape: (n_nodes, timesteps)
    # for each pair (i, j) where i is presynaptic, j is postsynaptic:
    # find spike times of i and j, compute STDP-style coincidence
    ...
    return eligibility_matrix  # (n_nodes, n_nodes)

# then reinforce/penalize becomes:
def reinforce(self, episome_id):
    epi = self._find_episome(episome_id)
    eligible_edges = self._eligibility_for_episome(episome_id)  # from last replay
    for edge in eligible_edges:
        delta = 0.1 * edge.eligibility  # three-factor: M * e
        edge.confidence = min(1.0, edge.confidence + delta)
```

This converts AGNN from "biological-flavored confidence bump" into
"textbook neoHebbian three-factor rule with credit assignment" —
the move the literature most directly endorses.

---

## 5. Sources

**Three-factor / neuromodulated STDP**
1. https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313 — Frémaux & Gerstner 2016 ⭐
2. https://arxiv.org/abs/2504.05341 — Mazurek et al. 2025
3. https://toyoizumilab.riken.jp/taro/papers/kusmierz17conb.pdf — Learning with three factors
4. https://neuronaldynamics.epfl.ch/online/Ch19.S5.html — Neuronal Dynamics, three-factor summary

**Reward-modulated STDP / distal reward**
5. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000180 — Legenstein/Pecevski/Maass 2008 ⭐
6. https://pmc.ncbi.nlm.nih.gov/articles/PMC6634722 — Functional Requirements for R-STDP (Frémaux/Sprekeler/Gerstner; cites Izhikevich 2007) ⭐
7. https://neuro.bstu.by/ai/Turkey-collabolation/06_modulated_STDP.pdf — Izhikevich, RL through modulation of STDP
8. https://mediatum.ub.tum.de/doc/1662599/ — r-STDP fine-tuning deep RL policies
9. https://brian2.readthedocs.io/en/stable/examples/frompapers.Izhikevich_2007.html — Izhikevich 2007 reproduction

**Eligibility traces / neoHebbian**
10. https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00053/full — Gerstner et al. 2018 ⭐
11. https://pmc.ncbi.nlm.nih.gov/articles/PMC6079224 — same, PMC
12. https://www.sciencedirect.com/science/article/abs/pii/S0959438825000091 — Eligibility traces as synaptic substrate
13. https://www.nature.com/articles/s41467-026-69898-9 — synaptic transistors, in-situ spiking RL

**Memory / hippocampus / attractors**
14. https://pmc.ncbi.nlm.nih.gov/articles/PMC10766193 — Dynamic attractor network: formation, reinforcement, forgetting ⭐
15. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1439414/full — Pang et al. 2024 ⭐
16. https://pmc.ncbi.nlm.nih.gov/articles/PMC8423376 — SNNs and Hippocampal Function
17. https://neuronaldynamics.epfl.ch/online/Ch17.S3.html — Memory networks with spiking neurons

**Reasoning / graph SNNs (reasoning angle)**
18. https://ieeexplore.ieee.org/abstract/document/9534102 — CRSNN (IJCNN 2021) ⭐
19. https://www.brain-cog.network/docs/tutorial/13_krr.html — BrainCog: Symbolic Representation & Reasoning SNNs over KGs
20. https://liner.com/review/temporal-spiking-neural-networks-with-synaptic-delay-for-graph-reasoning — Temporal SNN with synaptic delay

**STDP / local learning fundamentals**
21. https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity
22. http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity
23. https://www.nature.com/articles/s41380-023-02027-w — STDP overview
24. https://web.math.princeton.edu/~sswang/caporale_dan08_annu_rev_neurosci.pdf — Caporale & Dan, STDP as Hebbian rule
25. https://arxiv.org/html/2605.15058v1 — Surveying Local Learning Rules for SNNs

**Theoretical computing power / LIF vs ReLU / rate vs timing**
26. https://www.sciencedirect.com/science/article/pii/S0893608097000117 — Maass 1997, Third Generation ⭐
27. https://igi-web.tugraz.at/PDF/85a.pdf — Maass 1997 PDF
28. https://proceedings.neurips.cc/paper/1158-on-the-computational-power-of-noisy-spiking-neurons.pdf — Maass & Orponen
29. https://www.nature.com/articles/s41467-025-62251-6 — MSF neuron: LIF insufficient for joint spatiotemporal encoding ⭐
30. https://neuronaldynamics.epfl.ch/online/Ch7.S6.html — Neural coding: rate vs timing (Gerstner)
31. https://romainbrette.fr/rate-vs-timing-i-a-category-error — Brette, rate vs timing

**Neuromorphic / energy-efficient SNN (2024-2025)**
32. https://www.nature.com/articles/s41467-024-46397-3 — Ultra energy-efficient neuromorphic hardware platform
33. https://www.pnas.org/doi/10.1073/pnas.2528654122 — Can neuromorphic computing reduce AI's energy cost?
34. https://arxiv.org/html/2602.02439v1 — Energy-Efficient Neuromorphic Computing for Edge AI
35. https://cacm.acm.org/research/achieving-green-ai-with-energy-efficient-deep-learning-using-neuromorphic-computing — CACM, Green AI via neuromorphic

⭐ = highest-priority papers for AGNN to cite as primary validation.
