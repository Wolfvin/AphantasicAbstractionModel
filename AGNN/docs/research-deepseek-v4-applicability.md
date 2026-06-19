# Research: DeepSeek V4 Technical Report — Applicability to AGNN

> **Status:** Research note (no code changes — this is a synthesis of
> public technical material about DeepSeek V4 against AGNN's design).
> **Author date:** 2026-06-20.
> **Audience:** AGNN maintainers deciding whether to adopt any
> DeepSeek V4 architectural technique into the AGNN graph-reasoning
> stack.

---

## 0. Scope and method

This document answers one question: **are any of the specific
architectural techniques published in the DeepSeek V4 technical
report genuinely applicable to AGNN**, given AGNN's core philosophy
(small Qwen3-0.6B model + graph-based reasoning, NOT end-to-end
neural reasoning)?

The constraint is explicit: **the Qwen3-0.6B backbone is not up for
replacement**. The DeepSeek V4 techniques are evaluated ONLY as
algorithmic/architectural patterns that could be adapted to AGNN's
graph layer (AGNNGraph, TrisynapticCircuit, PapezCircuit,
PurkinjeCell, NeuralReplay, BA44 deductive engine,
PositionalClusterLearner).

**Method:**

1. Read AGNN/ARCHITECTURE.md to map every AGNN component to its
   functional role.
2. Survey the DeepSeek V4 technical report + 4 credible secondary
   analyses (listed in §6) to extract every named architectural
   technique.
3. For each technique: state the principle, evaluate whether the
   principle has a non-trivial analog in AGNN's graph paradigm, and
   rate the scope-of-change if adopted (kecil/sedang/besar).
4. Conclude with an explicit recommendation, including the honest
   case where no technique is genuinely worth adopting.

---

## 1. DeepSeek V4 — Key techniques from the technical report

DeepSeek V4 shipped 2026-04-24 as two open-weight MoE checkpoints
(V4-Pro 1.6T/49B active, V4-Flash 284B/13B active) with a 1M-token
context window. The technical report (`DeepSeek_V4.pdf` on
Hugging Face) and the official release notes
(`api-docs.deepseek.com/news/news260424`) call out the following
named techniques. Each is summarised here at the principle level
(implementation details and benchmark numbers are in the sources
linked in §6).

### 1.1 Auxiliary-Loss-Free MoE Load Balancing (carried over from V3)

Each expert has a per-expert **bias term** added to its routing
score; the bias is updated by a simple heuristic outside of
backpropagation. This balances expert utilization without an
auxiliary loss term that would interfere with the main training
objective.

- **Principle:** Decouple load-balancing signals from the gradient
  path of the main task. Bias nudges the Top-K router toward
  under-utilized experts without polluting the loss landscape.
- **Source:** [arXiv 2408.15664](https://arxiv.org/abs/2408.15664)
  (introduced for V3, retained in V4).

### 1.2 Compressed Sparse Attention (CSA)

Compresses KV entries **by 4× along the sequence dimension** using
softmax-gated pooling with a learned positional bias. A "lightning
indexer" (FP4-quantized, ReLU-scored multi-head dot product) picks
the top-k compressed blocks per query. Inherits the sparse-selection
idea from DeepSeek Sparse Attention (DSA) in V3.2 but runs it over
blocks that are already 4× shorter than the original sequence.

- **Principle:** Mild sequence-dimension compression + sparse
  top-k selection over the compressed stream → KV cache shrinks
  linearly with both the compression ratio and the top-k cap.
- **Source:** HF blog post on V4; Sebastian Raschka's analysis.

### 1.3 Heavily Compressed Attention (HCA)

Compresses KV entries by 128× and **drops the sparse selection** —
every query attends densely to every compressed block. The
compressed sequence is short enough that dense attention is cheap.

- **Principle:** Aggressive compression + dense attention on the
  short compressed stream. Trades token-level fidelity for global
  coverage at low cost.
- **Source:** Same as §1.2.

### 1.4 Hybrid CSA/HCA layer interleaving

In V4-Pro's 61-layer stack, layers 0–1 are HCA, layers 2–60
**alternate** CSA and HCA, and the MTP block at the end runs
sliding-window only. Different layers carry different attention
patterns; forcing one mechanism across all of them wastes capacity.

- **Principle:** Layer-wise attention pattern diversity. Cheap
  long-range coverage (HCA) and more-detailed sparse retrieval
  (CSA) cover different sub-tasks of attention; interleaving lets
  each layer specialize.
- **Source:** Same as §1.2; also Raschka §5.2.

### 1.5 Manifold-Constrained Hyper-Connections (mHC)

Replaces the single residual stream inside the transformer block
with **several parallel residual streams** + a learned Res Mapping
that mixes them across layers. mHC constrains the Res Mapping onto
the manifold of **doubly stochastic matrices** (all entries
non-negative, each row and column sums to 1) for numerical
stability at depth. Pre/Post mappings around each layer are
constrained to be non-negative and bounded.

- **Principle:** Widen the residual pathway (richer gradient and
  information flow) without widening the expensive Attention/FFN
  layers; constrain the mixing to a stable manifold so depth
  doesn't blow up the signal.
- **Source:** [arXiv 2512.24880](https://arxiv.org/abs/2512.24880);
  Raschka §5.1.
- **Cost:** +6.7% training-time overhead at n=4 parallel streams
  (per the mHC paper, 27B experimental model).

### 1.6 Engram — Conditional Memory via Scalable Lookup

**This is the most directly relevant technique for AGNN.** The
"Engram" module is a separate conditional-memory mechanism that
DeepSeek co-developed with Peking University
([arXiv 2601.07372](https://arxiv.org/pdf/2601.07372),
[github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)).
It is integrated into V4 as a complement to MoE.

The Engram module:

- Splits language modeling into two sub-task types:
  - **Combination & reasoning** — requires dynamic computation
    (long-range dependencies, chained reasoning, novel
    combinations).
  - **Pattern retrieval** — can be solved with O(1) lookup (entity
    names, fixed collocations, common phrases, grammar fragments,
    idioms).
- Offloads the second type to a deterministic **hashed N-gram
  lookup table** that retrieves pre-stored embeddings, fused into
  the hidden state via a context-aware gating mechanism (current
  hidden state = Query; retrieved memory = Key/Value; RMSNorm +
  short depth-causal conv expand the receptive field).
- Discovers a **Sparsity Allocation Law**: under a fixed parameter
  budget, the optimal split is **20–25% memory (Engram) and 75–80%
  computation (MoE)**. Below 20%, the model wastes compute
  "rediscovering" patterns; above 25%, reasoning capacity starves.
- Reports a **U-shaped scaling law** between MoE and Engram —
  adding more memory isn't monotonically better.
- Surprising finding: **offloading pattern work to memory lookup
  frees early Transformer layers to behave like deeper layers**,
  improving reasoning/math/coding benchmarks even though no new
  "facts" were added. The phrase from the paper: *"Engram doesn't
  make models smarter by adding facts — it makes them smarter by
  freeing compute."*
- **Hardware angle:** Engram indices are deterministic (depend
  only on input tokens, not activations), so memory retrieval can
  be prefetched asynchronously and 100B-parameter memory tables
  can be offloaded to CPU/SSD with <3% inference overhead.

**Why this matters for AGNN:** AGNN's `engrams/` directory already
contains `Episome`, `Semesome`, and `EngramComplex` — explicitly
named after the same neurology concept ("memory trace") that
DeepSeek's paper cites. AGNN's architecture *already implements*
the same memory/compute split, just at a different scale and with
different mechanics:

| | DeepSeek Engram | AGNN |
|---|---|---|
| **Static-pattern retrieval** | Hashed N-gram lookup table | PapezCircuit keyword-overlap scan over `EngramComplex` nodes |
| **Dynamic computation** | MoE + attention | Qwen3-0.6B articulation step |
| **Gating mechanism** | Attention-style gating on retrieved memory | BA44 deductive rules fire only when premise patterns match |
| **Offload target** | CPU/SSD | Already pure-Python graph; no GPU pressure |
| **Allocation tunable** | 20–25% memory / 75–80% compute (empirical law) | Currently ad-hoc — `top_k=3` in PapezCircuit, `_CHAIN_MAX_CHARS=800` chain truncation, no explicit memory/compute budget |

- **Source:** [arXiv 2601.07372](https://arxiv.org/pdf/2601.07372);
  [github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram);
  [deepseek.ai/blog/deepseek-engram-v4-architecture](https://deepseek.ai/blog/deepseek-engram-v4-architecture)
  (independent summary, DeepSeek-adjacent);
  [introl.com/blog/deepseek-engram-conditional-memory-architecture-january-2026](https://introl.com/blog/deepseek-engram-conditional-memory-architecture-january-2026).

### 1.7 Interleaved thinking across tool calls

V3.2 discarded reasoning traces whenever a new user message arrived.
V4 **preserves reasoning content across user message boundaries**
when the conversation contains tool calls, allowing a coherent
cumulative chain of thought over long-horizon agent tasks. For
conversational use without tools, the old behavior is preserved.

- **Principle:** For multi-turn agent workflows, the model's
  internal reasoning trace is itself part of the working state and
  should persist across external events (tool returns, user
  follow-ups) — not be flushed at every boundary.
- **Source:** HF blog post, section "Interleaved thinking across
  tool calls".

### 1.8 Tool-call schema with dedicated tokens (|DSML|)

V4 introduces a `|DSML|` special token and an XML-based tool-call
format. The schema separates string parameters (passed as-is with
`string="true"`) from structured parameters (passed as JSON with
`string="false"`). Reduces parsing failures common with
JSON-in-string tool calls.

- **Principle:** Dedicated tokens + structured schema remove a
  class of formatting/escaping errors at the protocol boundary
  between the model and external tools.
- **Source:** HF blog post, section "Tool-call schema with
  dedicated tokens".

### 1.9 Multi-Token Prediction (MTP)

Trains the model to predict 2+ tokens ahead; the MTP head can be
used at inference time for speculative decoding. V3 introduced
MTP-1 (predict one extra token); V4 retains the same pattern.

- **Principle:** Auxiliary training objective that improves
  representation quality and can speed up inference via
  speculative decoding when the MTP head is kept.
- **Source:** [DeepSeek V3 report](https://arxiv.org/abs/2412.19437);
  retained in V4 per the digitalapplied summary.

### 1.10 FP4/FP8 quantization

Instruct models use **FP4 for MoE expert weights** and FP8 for
everything else; base models are FP8 throughout. KV storage uses
FP8 for most entries and BF16 only for the RoPE dimensions; the
CSA lightning indexer runs in FP4. Compounds with the compression
ratios in §1.2/1.3 to produce the headline "10% KV cache vs V3.2"
figure.

- **Principle:** Aggressive mixed-precision storage at every
  sub-component where accuracy permits.
- **Source:** HF blog post; digitalapplied summary.

### 1.11 DSec — sandbox for RL rollouts

Rust platform exposing function calls, containers, microVMs
(Firecracker), and full VMs (QEMU) behind one Python SDK. Features:
fast image loading via layered 3FS storage, preemption-safe
trajectory replay, uniform API across substrates. Built for agent
RL training.

- **Principle:** RL training of agent behavior requires cheap,
  reproducible sandbox execution; uniform API across isolation
  tiers lets training harnesses scale without rewrites.
- **Source:** HF blog post, section "DSec".

---

## 2. Applicability assessment — technique by technique

Each subsection below takes one DeepSeek V4 technique, states its
principle, asks whether AGNN has a non-trivial analog, and rates
the scope-of-change if adopted.

### 2.1 Auxiliary-Loss-Free MoE Load Balancing → **Not applicable**

AGNN has no MoE. There is no router, no experts, no load to
balance. The closest AGNN analog (PapezCircuit's `top_k` retrieval)
is a deterministic keyword-overlap scan, not a learned router, so
"balancing load across retrieved nodes" doesn't have a training
signal to nudge.

The principle ("decouple load balancing from the main gradient
path") *could* be stretched to argue for an "exploration bonus"
that boosts nodes that are rarely retrieved by PapezCircuit — but
that is a graph-search heuristic, not a load-balancing analog, and
the AGNN open-world assumption already keeps every node reachable.

**Verdict:** No non-trivial analog. Skip.

### 2.2 CSA (Compressed Sparse Attention) → **Not applicable**

CSA compresses the **KV cache along the sequence dimension** and
selects top-k compressed blocks per query. AGNN's reasoning engine
doesn't use attention at all — PapezCircuit retrieves nodes by
keyword-set overlap (Jaccard-like), and BA44 fires deductive rules
on the resulting Semesome chain. There is no KV cache.

The *only* place attention enters AGNN is inside Qwen3-0.6B during
the articulate step, and the constraint forbids swapping that
backbone. Modifying Qwen3-0.6B's attention to use CSA would be a
model-retraining effort that violates the "small-model + graph
reasoning" vision.

A weak analog could be drawn to PapezCircuit's `top_k=3` cap: CSA's
top-k compressed blocks per query is conceptually similar to "pick
the top-k matching nodes per query." But PapezCircuit already does
this directly (no compression step needed because nodes are already
discrete). Adopting the CSA compression idea would mean
"compressing nodes into super-nodes" — which is what
`consolidate()` already does at the semantic level via
SystemsConsolidation. So the analog is already implemented.

**Verdict:** No non-trivial new technique. Skip.

### 2.3 HCA (Heavily Compressed Attention) → **Not applicable**

Same reasoning as §2.2. HCA is a KV-cache compression technique;
AGNN has no KV cache. The "dense attention over a heavily
compressed stream" idea has no analog in graph retrieval.

**Verdict:** No analog. Skip.

### 2.4 Hybrid CSA/HCA layer interleaving → **Not applicable**

This is a layer-wise attention-pattern diversity technique. AGNN's
"layers" are neuroanatomical components (EC → DG → CA3 → CA1 →
Sub), each with a distinct functional role already. The
TrisynapticCircuit already alternates functional roles across
stages. Forcing an "alternate two attention mechanisms" pattern
onto AGNN's pipeline would be a category error — there's no
attention to alternate.

**Verdict:** No analog. Skip.

### 2.5 mHC (Manifold-Constrained Hyper-Connections) → **Weak analog, not worth adopting**

mHC replaces a single residual stream with N parallel residual
streams + a doubly-stochastic mixing matrix. The benefit is richer
gradient and information flow at depth without widening the
expensive layers.

AGNN's NeuralReplay (`plasticity/neural_replay.py`) drives a
population of PurkinjeCells (one per graph node) for `timesteps`
steps and aggregates spike trains into fresh embeddings. This is
already a multi-stream computation (one stream per node), with
message passing between nodes via typed edges. The "parallel
residual streams" idea is structurally present.

The mHC doubly-stochastic constraint is the novel piece — it
stabilizes signal mixing across streams at depth. AGNN's LIF
neurons have their own stability mechanism (the closed-form
exponential decay in `PurkinjeCell.integrate_and_fire`), and the
`_SPREAD_DECAY` table per RelationType in `agnn/graph.py` already
caps how much signal propagates per edge type. Adding a
doubly-stochastic mixing matrix on top would be a non-trivial
rewrite of NeuralReplay's spike aggregation step, and the benefit
is unclear because AGNN's "depth" is 10 timesteps (per
`ARCHITECTURE.md` §9), not 60+ transformer layers. Stability
issues that mHC solves at depth-60 don't manifest at depth-10.

**Verdict:** Real but weak analog. The benefit doesn't justify the
implementation cost at AGNN's scale. Skip.

### 2.6 Engram (Conditional Memory) → **Genuinely applicable** ⭐

This is the one technique with a strong, non-trivial analog.

**Why it's applicable:** AGNN and DeepSeek Engram independently
arrived at the same architectural insight: separate **O(1) memory
retrieval** for static patterns from **dynamic computation** for
reasoning. AGNN's PapezCircuit + Qwen3-0.6B split is structurally
isomorphic to Engram's lookup-table + MoE split (see the table in
§1.6).

**What AGNN can learn from Engram:**

1. **The 20–25% Sparsity Allocation Law.** DeepSeek's empirical
   finding: under a fixed budget, ~20-25% should go to memory and
   ~75-80% to computation. Too little memory wastes compute
   "rediscovering" patterns; too much memory starves reasoning.
   AGNN currently has no explicit memory/compute budget. The
   closest tunables are:
   - `PapezCircuit.retrieve(top_k=3)` (memory side: how many
     nodes to retrieve)
   - `_CHAIN_MAX_CHARS = 800` (compute side: how much chain
     context to feed Qwen3-0.6B for articulation)

   These tunables are currently picked ad-hoc. The Engram finding
   suggests an **explicit allocation principle**: measure how much
   "knowledge" the graph holds (memory budget) vs how much
   Qwen3-0.6B has to compute from scratch (compute budget), and
   tune `top_k` × chain length so the graph carries roughly 20-25%
   of the "work" by retrieval, leaving 75-80% for the model to
   reason over. This is a **research hypothesis** for AGNN, not a
   proven result — but it gives a principled target where
   currently there is none.

2. **Context-aware gating on retrieved memory.** Engram's
   retrieved embeddings are *not* injected directly — they pass
   through an attention-style gate (current hidden state = Query,
   retrieved memory = Key/Value). This prevents noisy retrievals
   (hash collisions, polysemy) from polluting the hidden state.
   AGNN's analog: PapezCircuit retrieves nodes by keyword overlap,
   but the retrieval is currently a hard top-k cut — there's no
   "the retrieved node is only relevant if the current query
   matches its context" gating. The BA44 deductive engine provides
   *some* gating (rules fire only when premise patterns match),
   but that's post-retrieval. An **Engram-style pre-articulation
   gating step** — where each retrieved episome is re-scored by
   attention against the query before being added to the chain —
   could be a meaningful quality improvement.

3. **"Free compute" by offloading pattern work.** DeepSeek's
   counterintuitive finding: offloading pattern recognition to
   memory lookup *improves reasoning benchmarks* because early
   layers stop wasting cycles on pattern matching. AGNN already
   gets this benefit for free — its "early layers" are
   PapezCircuit (cheap keyword scan), and Qwen3-0.6B is reserved
   for articulation only. But the finding suggests a direction:
   **are there currently-Qwen3-handled steps that could be
   offloaded to the graph?** Candidate: the
   `PositionalClusterLearner.spo()` method currently delegates to
   Qwen3 only when graph clusters are unlabeled. If more of the
   SPO extraction could be moved to graph lookup, Qwen3 has more
   budget for articulation. (This is the Engram principle applied
   to AGNN's own design — and it's already partially implemented.)

4. **Memory-budget tradeoff curve (U-shaped scaling law).**
   DeepSeek reports a U-shaped curve: too little or too much
   memory both hurt. AGNN could run a parameter sweep on
   `top_k ∈ {1, 2, 3, 5, 8}` × `_CHAIN_MAX_CHARS ∈ {400, 800,
   1600, 3200}` and measure `chain_confidence` and answer quality
   on a fixed evaluation set. If the curve is U-shaped, that's
   empirical confirmation that the Engram law applies to AGNN's
   scale and paradigm; if it's monotonic, AGNN's regime is
   different and the law doesn't transfer. **This is a concrete
   experiment worth running.**

**Scope of change if adopted:**

| Sub-technique | Scope | Concrete change |
|---|---|---|
| 20-25% allocation principle as a tuning target | **Kecil** | Documentation + a measurement script that computes `memory_budget / (memory_budget + compute_budget)` for current settings. No code change to the runtime. |
| Engram-style pre-articulation gating | **Sedang** | Add a `_gate_retrieved_episomes(question, episomes)` step in `process()` between PapezCircuit retrieval and `_build_semesomes_from_graph`. Each retrieved episome gets an attention-style score against the question; only episomes scoring above a threshold proceed to the chain. New method on `AGNNCore`, no public API change. |
| U-shaped scaling sweep experiment | **Sedang** | New test file `tests/test_memory_compute_allocation.py` that runs the sweep and asserts the curve shape. Doesn't change production code. |
| "Move more steps from Qwen3 to graph" audit | **Sedang–Besar** | Audit every place `core.py` calls into Qwen3 (just `_articulate` currently) and check whether the input could be pre-resolved by graph lookup. This is open-ended; could spawn multiple PRs. |

**Verdict:** Genuinely applicable. Worth adopting in pieces — the
20-25% principle as a documentation target first (smallest change),
then the pre-articulation gating as a follow-up PR if the principle
holds up under measurement.

### 2.7 Interleaved thinking across tool calls → **Weak analog, defer**

AGNN doesn't currently have a multi-turn agent loop. `process()`
is single-turn: retrieve → deduce → articulate. There's no
"reasoning trace to preserve across tool calls" because there are
no tool calls.

If AGNN later adds an agent loop (e.g., `traverse()` returns a
chain, the caller asks a follow-up that should re-use the chain),
the V4 pattern of "preserve reasoning across user-message
boundaries when tools are in play" is exactly the right design.
But that's a future feature, not a current gap.

**Verdict:** Note for future agent-loop PR. Don't adopt now.

### 2.8 Tool-call schema with dedicated tokens → **Not applicable**

AGNN doesn't emit tool calls. The Qwen3 chat-template wrap in
`_generate()` is for free-form articulation, not structured tool
invocation. Adopting `|DSML|`-style tokens would require a
tool-use protocol that AGNN doesn't have.

**Verdict:** Not applicable. Skip.

### 2.9 Multi-Token Prediction → **Not applicable**

MTP requires training the model. AGNN's Qwen3-0.6B is a fixed
pretrained checkpoint; the constraint forbids retraining it.
Speculative decoding at inference time is a serving-layer feature
that lives in the inference engine, not the application stack, and
is orthogonal to AGNN's graph-reasoning design.

**Verdict:** Not applicable. Skip.

### 2.10 FP4/FP8 quantization → **Not applicable (operational, not architectural)**

AGNN could quantize its Qwen3-0.6B weights via bitsandbytes or
similar at load time. That's an operational/CLI flag, not an
architectural change to AGNN's graph reasoning. The DeepSeek V4
technique is about *where in the architecture* to use which
precision (FP4 for experts, FP8 for KV, BF16 for RoPE) — that
level of precision-routing only makes sense inside a model the
team trains, which AGNN doesn't.

**Verdict:** Not applicable to AGNN's architecture. (Operational
note: AGNN *could* quantize Qwen3-0.6B to reduce its CPU/RAM
footprint, but that's a `transformers`-load flag, not a DeepSeek
V4 technique.)

### 2.11 DSec sandbox → **Not applicable**

AGNN doesn't train agents via RL. There's no rollout sandbox to
build. If AGNN later adds an RL-trained traversal policy (e.g.,
learning when to call `traverse()` vs `process()`), the DSec
pattern of "uniform API across isolation tiers" would be relevant.
But that's a major scope expansion that the current architecture
doesn't envision.

**Verdict:** Not applicable. Skip.

---

## 3. Summary table

| § | DeepSeek V4 technique | Applicable to AGNN? | Scope if adopted |
|---|---|---|---|
| 1.1 | Aux-loss-free MoE balancing | No | — |
| 1.2 | CSA (sparse compressed attention) | No | — |
| 1.3 | HCA (heavily compressed attention) | No | — |
| 1.4 | Hybrid CSA/HCA interleaving | No | — |
| 1.5 | mHC (parallel residual streams) | Weak analog, not worth it | (skip) |
| **1.6** | **Engram (conditional memory)** | **Yes** ⭐ | **Kecil → Sedang** |
| 1.7 | Interleaved thinking across tool calls | Defer (future agent loop) | — |
| 1.8 | Tool-call schema (|DSML|) | No | — |
| 1.9 | Multi-Token Prediction | No (would require retraining) | — |
| 1.10 | FP4/FP8 quantization | No (operational, not architectural) | — |
| 1.11 | DSec RL sandbox | No | — |

---

## 4. Conclusion — explicit recommendation

**Of the 11 techniques surveyed, exactly 1 is genuinely worth
adopting: Engram (§1.6 / §2.6).** Two more have weak analogs
(mHC §1.5, interleaved thinking §1.7) but don't justify adoption
at AGNN's current scale and scope.

### Recommended priority order for Engram-inspired work

1. **Document the 20-25% Sparsity Allocation Law as a tuning
   target for AGNN.** (Scope: **kecil**) Add a section to
   `ARCHITECTURE.md` or a new `docs/allocation-target.md` that
   explains the Engram analogy, the current AGNN tunables
   (`top_k`, `_CHAIN_MAX_CHARS`), and what the principle suggests
   for future tuning. No production code change.

2. **Run a memory-vs-compute allocation sweep.** (Scope:
   **sedang**) New test file that sweeps `top_k × _CHAIN_MAX_CHARS`
   and reports `chain_confidence` + answer quality on a fixed
   evaluation corpus. Look for a U-shaped curve. If found, that
   confirms the Engram law applies to AGNN's scale. If not, the
   principle doesn't transfer and the experiment closes the
   question.

3. **Add Engram-style pre-articulation gating.** (Scope:
   **sedang**) Only if step 2 shows the principle holds. Add a
   `_gate_retrieved_episomes(question, episomes)` method on
   `AGNNCore` that re-scores each retrieved episome by an
   attention-style match against the question before building the
   Semesome chain. Public API unchanged; the gating is internal
   to `process()`.

4. **Audit Qwen3-bound work for graph-offload opportunities.**
   (Scope: **sedang-besar**, open-ended) Per the Engram finding
   that "freeing compute improves reasoning," check whether any
   pre-articulation step currently delegated to Qwen3 could be
   moved to the graph. Candidates: SPO extraction when clusters
   are unlabeled, chain ordering, summary generation. This is a
   multi-PR thread, not a single change.

### What NOT to do

- **Do not** swap Qwen3-0.6B for a DeepSeek checkpoint or any
  other backbone. That violates the core AGNN vision and is
  explicitly out of scope per the task brief.
- **Do not** port CSA, HCA, mHC, or any other attention/residual
  technique into AGNN. AGNN's graph layer doesn't use attention,
  and the benefits at depth-60 don't transfer to depth-10 LIF
  simulations.
- **Do not** adopt the |DSML| tool-call schema. AGNN has no
  tool-call protocol.
- **Do not** add FP4/FP8 quantization routing. That's a
  serving-layer concern and AGNN doesn't train the model.

### Honest caveat

The single applicable technique (Engram) is **not a copy-paste
port** — it's a **principle analog**. DeepSeek's implementation
(hashed N-gram lookup tables, attention-style gating on retrieved
embeddings, MoE+Engram parameter split) is end-to-end-neural and
doesn't translate to AGNN's discrete graph. What translates is the
**architectural insight**: separate O(1) memory retrieval from
dynamic computation, and the empirical finding that there's a
principled allocation target between the two. AGNN already
implements the separation; the contribution of this research is
the **principled target** (20-25%) and the **gating mechanism**
idea, both of which are testable improvements on AGNN's current
ad-hoc tunables.

---

## 5. Glossary

- **AGNN** — Aphantic Graph Neural Network. The repo this document
  lives in.
- **AGNNGraph** — The underlying typed-graph data structure
  (defined in `self-ai/src/agnn/graph.py`, wrapped by
  `engrams/engram_complex.py`).
- **BA44** — Left inferior frontal gyrus (Broca's area). In AGNN,
  the deductive reasoning engine (`neocortex/inferior_frontal_gyrus.py`).
- **CSA / HCA** — Compressed Sparse Attention / Heavily Compressed
  Attention. DeepSeek V4's two attention mechanisms (§1.2, §1.3).
- **Engram** — Neurology term for "memory trace". Used by both
  DeepSeek (as the name of their conditional-memory module) and
  AGNN (as the name of the `engrams/` package + `EngramComplex`
  class).
- **Episome / Semesome** — AGNN's memory units: episodic (node,
  labile) and semantic (edge, stable). Roughly correspond to
  DeepSeek Engram's "static-pattern retrieval" target.
- **mHC** — Manifold-Constrained Hyper-Connections. DeepSeek V4's
  residual-stream design (§1.5).
- **MoE** — Mixture of Experts.
- **MTP** — Multi-Token Prediction.
- **PapezCircuit** — AGNN's retrieval loop (`circuits/papez_circuit.py`).
- **PurkinjeCell** — AGNN's LIF neuron implementation
  (`cerebellum/purkinje_cell.py`), pure numpy.
- **Qwen3-0.6B** — AGNN's articulation backbone. Fixed pretrained
  model; not retrained by AGNN.

---

## 6. Sources

Primary (DeepSeek-authored):

- **DeepSeek V4 technical report (PDF):**
  https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
- **DeepSeek V4 Preview Release notes:**
  https://api-docs.deepseek.com/news/news260424
- **Engram paper (DeepSeek × Peking University):**
  https://arxiv.org/pdf/2601.07372
- **Engram GitHub:**
  https://github.com/deepseek-ai/Engram
- **mHC paper (DeepSeek, Dec 2025):**
  https://arxiv.org/abs/2512.24880
- **Aux-Loss-Free Load Balancing paper (DeepSeek, Aug 2024):**
  https://arxiv.org/abs/2408.15664
- **DeepSeek V3 technical report (V3 baseline, MTP introduced
  here):** https://arxiv.org/abs/2412.19437

Secondary (independent technical analyses):

- **Hugging Face blog — "DeepSeek-V4: a million-token context that
  agents can actually use"** (2026-04-24, by Ben Burtenshaw):
  https://huggingface.co/blog/deepseekv4
- **Sebastian Raschka — "Recent Developments in LLM Architectures:
  KV Sharing, mHC, and Compressed Attention"** (2026-05-16):
  https://magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures
- **Digital Applied — "DeepSeek V4 Launches: 1.6T MoE, 1M Context,
  10% KV"** (2026-04-24):
  https://www.digitalapplied.com/blog/deepseek-v4-preview-launch-1m-context-efficiency
- **Kili Technology — "Data Story: A Deep Dive into DeepSeek V4"**
  (updated May 2026):
  https://kili-technology.com/blog/data-story-deepseek-v4

Note on source credibility: the HuggingFace blog, Raschka's
analysis, and the arXiv papers are the most authoritative. The
`deepseek.ai/blog/deepseek-engram-v4-architecture` URL is an
independent fan-site summary (not affiliated with DeepSeek per
its own disclaimer) — used here only for the Engram explanation,
which is consistent with the arXiv paper. The official DeepSeek
sources (api-docs + huggingface model card + arXiv papers) are
the canonical references.
