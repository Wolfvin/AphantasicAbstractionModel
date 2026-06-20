# Research: Neuro-Symbolic Reasoning — Paths to Evolving BA44's Hardcoded Rules into Learned Ones

> **Worker 3** of a 4-worker research batch.
> **Focus:** Neuro-symbolic AI — systems that combine symbolic
> reasoning (rule-based, like AGNN's BA44 deductive rules) with
> learned representations. Evaluated against AGNN's
> `neocortex/inferior_frontal_gyrus.py` (BA44 engine with 5 hardcoded
> rules and fixed weights).
> **Status:** Research note (no code changes).
> **Author date:** 2026-06-20.

---

## 0. Scope and method

AGNN's `neocortex/inferior_frontal_gyrus.py` implements BA44 (Broca's
area) as a deductive reasoning engine. It has **5 hardcoded
human-authored rules**:

1. `CATEGORICAL_TRANSITIVITY`: A→B (CAT), B→C (CAT) ⇒ A→C (weight
   `1.0 × 1.0 = 1.0`)
2. `CAUSAL_CHAIN`: A→B (CAUSAL), B→C (CAUSAL) ⇒ A→C (weight
   `0.7 × 0.7 = 0.49`)
3. `DIFFERENTIAL_INVERSION`: A→B (DIFF=−0.8) ⇒ B→A (DIFF=−0.8)
   [symmetric]
4. `CAUSAL_DIFFERENTIAL_CONFLICT`: A→B (CAUSAL) + A→B (DIFF) ⇒
   conflict, weight `(0.7 + −0.8)/2 = −0.05`
5. `FUNCTIONAL_COMPOSITION`: A→B (FUNC), B→C (FUNC) ⇒ A→C (weight
   `0.6 × 0.6 = 0.36`)

The weights (0.7, 0.8, 0.6, etc.) are **hardcoded constants**, not
learned. The rule *structures* themselves are also fixed at design
time. The engine operates over `Semesome` edges (each has type,
weight, source, target) and returns a `Deduction` with inferences,
inferred_edges, rule_count, applied_rules, confidence, context.

AGNN's design philosophy: *model kecil TIDAK perlu reason sendiri —
graph yang reason (typed edges, BA44 deductive rules), model hanya
articulate hasil reasoning ke bahasa natural.*

This document surveys the neuro-symbolic AI literature to answer two
questions:

1. Is there a literature path for making BA44's deductive rules
   *emergent* (learned) rather than hand-authored?
2. What specific techniques could AGNN borrow to evolve the 5
   hardcoded rules without breaking the small-model philosophy?

Method: 9 targeted web searches covering neural theorem provers,
differentiable logic, Markov Logic Networks, ProbLog, rule mining,
neural program induction, neuro-symbolic surveys, and LLM+symbolic
solvers.

---

## 1. Key techniques found

### 1.1 Differentiable Logic / Logic Tensor Networks (LTN) + t-norm fuzzy logic

**Principle.** LTN defines "Real Logic," a first-order language whose
formulas take continuous truth values in [0,1] and whose
constants/predicates are grounded onto real-valued tensors (feature
vectors / neural nets). Logical connectives are implemented as
**differentiable t-norms**: product (a·b), Łukasiewicz
(max(0,a+b−1)), Gödel (min(a,b)). This makes a whole knowledge base
differentiable, so rule *parameters* and even grounding networks can
be trained end-to-end by gradient descent on a satisfaction loss. A
key follow-up (van Krieken et al., "Analyzing Differentiable Fuzzy
Logic Operators," AIJ 2022) shows that the *choice* of t-norm
materially affects gradient behaviour and rule learning quality.

**Key sources:**
- Badreddine, d'Avila Garcez, Serafini, "Logic Tensor Networks,"
  arXiv:1606.04422 (2016); AIJ 2022 version at
  https://dl.acm.org/doi/10.1016/j.artint.2021.103649
- Operator analysis:
  https://www.sciencedirect.com/science/article/pii/S0004370221001533

**AGNN-BA44 alignment — HIGH.** This is the most directly relevant
technique. AGNN's current weight arithmetic (`0.7×0.7=0.49` for causal
chains, `0.6×0.6=0.36` for functional composition) is *exactly the
product t-norm* — **AGNN is already an implicit (and unconscious)
product-t-norm fuzzy logic**. Recognising this reframes BA44's
`weight * weight` as a principled fuzzy conjunction rather than an
ad-hoc constant, and immediately suggests (a) making the per-type
weights learnable parameters, and (b) offering a configurable t-norm
(product vs. Łukasiewicz vs. Gödel) whose choice can itself be tuned.

### 1.2 Neural Theorem Provers (NTP) and descendants

**Principle.** NTP (Rocktäschel & Riedel 2017) makes Prolog-style
**backward chaining differentiable** by replacing hard symbolic
unification with a **radial basis function (RBF) kernel** over dense
vector representations of symbols: unification score = similarity in
embedding space, bounded in [0,1]. Proof success becomes a
recursively-computed differentiable score; gradients flow into both
rule weights and symbol embeddings. NTP can *induce* function-free
first-order logic rules from an incomplete KB and outperformed
ComplEx on 3/4 benchmarks.

**Key sources:**
- Rocktäschel & Riedel, "End-to-End Differentiable Proving,"
  arXiv:1705.11040, NIPS 2017 — https://arxiv.org/abs/1705.11040
- Scale-up: Minervini & Bosnjak,
  https://www.semanticscholar.org/paper/Towards-Neural-Theorem-Proving-at-Scale-Minervini-Bosnjak/47ebd48e05efb305785b1c4f6a91bac1e891feed
- pLogicNet (NeurIPS 2019):
  https://proceedings.neurips.cc/paper/2019/hash/13e5ebb0fa112fe1b31a1067962d74a7-Abstract.html

**AGNN-BA44 alignment — MEDIUM.** Conceptually BA44's rule application
*is* a form of (forward) chaining, so the differentiable-proof idea
maps naturally. However NTP's value proposition is learning **dense
symbol embeddings** — which conflicts with AGNN's "graph already
represents, model is small" stance. NTP is most useful as
*inspiration* for a differentiable unification score (e.g., when two
edges "match" a rule template, score the match by a similarity rather
than a hard boolean), not as a runtime engine.

### 1.3 Neural Program Induction / Differentiable interpreters (NTM, DNC) — cautionary tale

**Principle.** Neural Turing Machines (Graves 2014) and
Differentiable Neural Computers (Graves et al. 2016, Nature) couple a
recurrent neural controller to an **external addressable memory** with
differentiable read/write heads, enabling the network to *learn
algorithms* (sort, copy, associative recall) by gradient descent.
Programs are implicit in the controller's learned read/write policy.

**Key sources:**
- DeepMind DNC blog + code:
  https://deepmind.google/blog/differentiable-neural-computers ;
  https://github.com/google-deepmind/dnc
- Community post-mortem on training instability:
  https://www.reddit.com/r/MachineLearning/comments/qwwf82

**AGNN-BA44 alignment — LOW.** This is the *cautionary tale* of the
list. The community consensus is that NTMs/DNCs are "still recurrent
networks" — inefficient and unstable to train, and they were largely
eclipsed by attention/transformers. For AGNN this is a **negative
result to heed**: trying to make BA44's rule *application* an emergent
differentiable interpreter inside the small model is likely to be
brittle. The lesson is to keep the symbolic interpreter (BA44)
hard-structured and let learning happen *around* it, not *inside* it.

### 1.4 Probabilistic Logic: Markov Logic Networks, ProbLog, DeepProbLog

**Principle.** Three complementary flavours of "logic + probability."

- **MLNs** (Richardson & Domingos 2006) attach real-valued *weights*
  to first-order formulas; the weighted formula set defines a Markov
  network over possible worlds, so weighted rules become a joint
  probability distribution — weights are learned by pseudo-likelihood
  / contrastive divergence.
- **ProbLog** gives logic programs possible-world semantics where
  facts have probabilities; inference = weighted model counting.
- **DeepProbLog** (Manhaeve et al. 2018) extends ProbLog with
  **neural predicates**: a neural net's output distribution becomes
  the probabilistic choices feeding a logic program, trained
  end-to-end.
- **pLogicNet** (NeurIPS 2019) fuses MLN rule reasoning with KG
  embeddings.

**Key sources:**
- MLN: https://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf
- DeepProbLog: https://arxiv.org/abs/1805.10872
- pLogicNet: https://proceedings.neurips.cc/paper/2019/hash/13e5ebb0fa112fe1b31a1067962d74a7-Abstract.html
- Neural Markov Logic Networks:
  https://proceedings.mlr.press/v161/marra21a/marra21a.pdf

**AGNN-BA44 alignment — HIGH (especially DeepProbLog).** The
**neural-predicate** pattern is an almost exact match for AGNN's
philosophy: the *small model* provides noisy/graded truth values on
atomic facts (edge existence, edge weight), and the *logic layer*
(BA44) does deterministic structured reasoning over them. MLN's
*weighted-rule* formulation is the natural language for BA44's fixed
0.7/0.8/0.6 weights — those should be MLN-style rule weights learned
against a Semesome coherence objective. DeepProbLog demonstrates
concretely that you can train a *small* neural net whose only job is
to feed graded facts into a fixed logic program — i.e., exactly
"model articulate, graph reason."

### 1.5 LLM + symbolic solvers; chain-of-thought as implicit theorem proving

**Principle.** A recent (2023-2026) line treats the LLM as an
*articulator/translator*, not a reasoner. **Proof of Thought**
(Ghosh et al. 2024) has the LLM draft its reasoning as a structured
JSON DSL, deterministically compiles that to first-order logic, and
runs **Z3 (SMT)** to verify — the LLM never "does" the math; the
solver does. "Symbolic Chain-of-Thought," Lean+LLM for math (Turing),
and tool-augmented logical reasoning (ALTA 2024) follow the same
split: **neural articulation + symbolic verification**. This is
structurally identical to AGNN's stated philosophy.

**Key sources:**
- Proof of Thought: https://arxiv.org/html/2409.17270v1 ; code
  https://github.com/DebarghaG/proofofthought
- Symbolic-Aided CoT: https://arxiv.org/html/2508.12425v1
- Tool-based logical reasoning: https://aclanthology.org/2024.alta-1.4.pdf
- Lean+LLMs:
  https://www.turing.com/resources/lean-and-symbolic-reasoning-in-llms-for-math-problem-solving

**AGNN-BA44 alignment — HIGH (philosophical) / MEDIUM (technical).**
This validates AGNN's *design stance* — the literature is converging
on exactly "small/neural model articulates, symbolic layer reasons."
The borrow is conceptual but powerful: AGNN can position BA44 as the
**deterministic verifier/solver behind an articulating model**, and
the "CoT as implicit theorem proving" framing gives a principled
account of *why* typed-edge deduction is the right substrate.
Technically, the Proof-of-Thought pattern (structured DSL → solver)
suggests AGNN could expose BA44 as a callable "solver" that a larger
host LLM routes to, rather than embedding reasoning in the model
weights.

### 1.6 Rule mining / rule learning from knowledge graphs (NeuralLP, DRUM, RNNLogic, AnyBURL)

**Principle.** These methods **discover Horn-clause rules from a KG**
rather than hand-authoring them.

- **NeuralLP** (Yang et al. 2017) compiles inference into
  differentiable TensorLog operations and trains a neural controller
  to *compose* them, jointly learning rule *structure* (discrete) and
  *parameters* (continuous) end-to-end.
- **DRUM** (Sadeghian et al. 2019) uses **bidirectional RNNs** to
  share information across relations when mining closed connected
  Horn clauses; it connects rule-confidence learning to low-rank
  tensor approximation and supports *inductive* (unseen-entity) link
  prediction.
- **RNNLogic** (Qu et al. 2020) treats rules as **latent variables**
  and trains a rule generator + reasoning predictor jointly via **EM**
  (E-step: posterior-select high-quality rules; M-step: update
  generator), avoiding both huge search spaces and sparse RL rewards.
- **AnyBURL** (Meilicke et al. 2019) is an *anytime bottom-up* rule
  learner — fast, lightweight, interpretable, designed for KG
  completion at scale.

**Key sources:**
- NeuralLP: https://arxiv.org/abs/1702.08367 (code
  https://github.com/fanyangxyz/Neural-LP)
- DRUM: https://arxiv.org/abs/1911.00055 (project
  https://dsr.cise.ufl.edu/drum-kg/index.html)
- RNNLogic: https://arxiv.org/abs/2010.04029
- AnyBURL: https://www.ijcai.org/proceedings/2019/0435.pdf (site
  https://web.informatik.uni-mannheim.de/AnyBURL)
- Review of the area: https://d-nb.info/1367508878/34

**AGNN-BA44 alignment — HIGHEST (this is the direct answer to "evolve
hardcoded rules into learned ones").** The Semesome *is* a typed KG.
BA44's 5 rules are exactly the kind of closed connected Horn clauses
these systems mine. AnyBURL/DRUM can be run **offline** on an
accumulated Semesome corpus to *discover* that
"CAUSAL→CAUSAL⇒CAUSAL" is a high-support pattern, with a learned
confidence — turning the hardcoded 0.7 into a mined statistic.
RNNLogic's latent-variable EM is the most principled "rules emerge"
formulation but is heavier. AnyBURL is the pragmatic choice for AGNN's
scale.

---

## 2. Specific actionable borrow opportunities for AGNN's BA44 engine

Ordered roughly by effort (low → high) and risk (low → high).

### B1. Recognise BA44's weight arithmetic as a t-norm and make it explicit + configurable

AGNN's `0.7 * 0.7` and `0.6 * 0.6` are already the **product t-norm**
for conjunction. Replace the hard-coded multiplication with a named,
swappable t-norm (`product` default, `lukasiewicz`, `godel`). This is
a pure refactor that (a) makes the design self-documenting, (b) lets
you A/B test t-norms per edge-type, and (c) grounds future gradient
learning in a principled semantics.

- **Borrowed from:** LTN / van Krieken operator analysis.
- **Effort:** low.
- **Risk:** low.

### B2. Promote the 5 magic constants (1.0, 0.7, 0.8, 0.6, conflict-blend 0.5) to learnable parameters

Treat them as MLN-style rule weights. Train against a Semesome
*coherence / link-prediction* objective: for edges that BA44 *infers*
(via a rule), reward; for edges present in the Semesome but
contradicted by an inferred edge (e.g., a CAUSAL vs DIFF conflict),
penalise. This is a handful of scalar parameters — perfectly
consistent with "small model."

- **Borrowed from:** MLN weight learning, pLogicNet.
- **Effort:** medium.
- **Risk:** low.

### B3. Add an offline "sleep" rule-mining pass over the Semesome (AnyBURL or DRUM)

Periodically run a rule miner on the accumulated Semesome KG. Compare
mined rules against BA44's 5 hand-authored ones: (a) confirm the 5
are high-support (validates the design), (b) surface *new* rule
templates (e.g., a recurring `AUX→VERB` pattern) as candidates to
promote into BA44. This is the cleanest path to **structural
emergence** without touching runtime or the small model. The model
never sees the miner.

- **Borrowed from:** AnyBURL, DRUM, NeuralLP.
- **Effort:** medium (integrate AnyBURL as an offline tool).
- **Risk:** low (offline; no runtime impact).

### B4. Adopt the DeepProbLog "neural predicate" pattern for edge weighting

Let a small neural head (already part of AGNN) output a graded truth
value / weight for candidate Semesome edges; BA44 then reasons over
those graded edges with t-norm composition. This makes the *edge
weights BA44 consumes* learned, even if BA44's rule *templates* stay
fixed — matching "model articulates, graph reasons" precisely.

- **Borrowed from:** DeepProbLog, NTP's graded unification.
- **Effort:** medium.
- **Risk:** medium.

### B5. Make rule *application* differentiable via an NTP-style soft unification score

When a rule template fires, instead of a hard boolean "does edge A→B
match the rule's body slot?", compute a soft match score
(cosine/RBF over a lightweight embedding of edge types). The inferred
edge weight becomes `t_norm(soft_match, body_weight)`. This lets BA44
*generalise* rules to near-miss edges and gives gradients into type
embeddings.

- **Borrowed from:** NTP RBF unification.
- **Effort:** medium (needs an embedding per edge-type — small).
- **Risk:** medium.

### B6. Reframe BA44 as a "solver" behind an articulating model (Proof-of-Thought pattern)

Expose BA44's `Deduction` as a structured artefact (it already returns
`inferences`, `applied_rules`, `confidence`, `context`) that a host
model *cites* rather than re-derives. This is essentially already
AGNN's design — make it explicit and defensible by citing the
Proof-of-Thought / symbolic-CoT literature.

- **Borrowed from:** Proof-of-Thought, symbolic-CoT.
- **Effort:** low (mostly documentation + API clarity).
- **Risk:** low.
- **Payoff:** high conceptual framing.

### B7. (Longer-term, higher-risk) RNNLogic-style latent-variable rule induction

Treat BA44's rule set as a latent variable; train a rule generator +
a reasoning predictor with EM. This is the only technique that offers
genuine *online* rule-structure emergence. Honest take: probably
overkill and too compute-heavy for AGNN's stated small-model scale —
keep it as a research direction, not a near-term borrow.

- **Borrowed from:** RNNLogic.
- **Effort:** high.
- **Risk:** high.

---

## 3. Theoretical validation — can BA44 rules become *emergent* at AGNN's scale?

**Honest, layered answer.**

### 1. Emergent rule WEIGHTS — yes, trivially

Replacing 0.7/0.8/0.6 with learned scalars is standard MLN/pLogicNet
territory. A few parameters trained on Semesome coherence. No conflict
with the small-model philosophy. **Feasible now.**

### 2. Emergent rule STRUCTURE — yes, but offline, not runtime

AnyBURL/DRUM/NeuralLP all *discover* Horn-clause structures from KGs.
None require a large neural model; AnyBURL is explicitly
fast/lightweight/anytime. The clean architectural move is a periodic
**offline "sleep"/consolidation phase** that mines the Semesome and
promotes validated rules into BA44's kernel. The small runtime model
still only articulates. This is the *strongest* literature-supported
path and is **feasible at AGNN's scale**.

### 3. Emergent rule TYPES (genuinely novel reasoning patterns not in the 5-rule kernel, appearing during inference) — partially, with caveats

True online emergence is supported in the literature only by (a) large
pretrained differentiable provers (NTP family) or (b) expensive EM/RL
search (RNNLogic, NeuralLP controllers). Both require compute and
scale that AGNN's philosophy rejects. The honest conclusion:
**runtime emergence of new rule types is not well-supported at
small-model scale.** The pragmatic compromise is to *expand* the rule
kernel periodically (point 2) so that "new" rule types are promoted
into the fixed runtime set rather than invented live.

### 4. What breaks the small-model philosophy, and what doesn't

- *Breaks it:* putting a differentiable theorem prover or a learned
  interpreter *inside* the runtime model (NTP-as-engine, DNC-style
  controller). The community's experience with NTMs/DNCs (Section 1.3)
  is a direct warning.
- *Doesn't break it:* (i) learned scalar weights, (ii) offline rule
  mining that the model never loads, (iii) a small neural head that
  only grades edge truth (DeepProbLog neural predicate), (iv) keeping
  BA44 as the deterministic solver an articulating model routes to
  (Proof-of-Thought).

### 5. Bottom line

There *is* a credible literature path from "5 hand-authored rules with
magic constants" to "learned-weight rules with a periodically-mined,
growing rule kernel," and it preserves AGNN's "small model
articulates, graph reasons" stance. Full *autonomous* emergence of
deductive rules at runtime is **not** supported at AGNN's target scale
by the current state of the art — and the literature's failed attempts
(DNC) suggest it would be unwise to pursue there.

The recommended posture is **hybrid and staged**: keep the symbolic
kernel (interpretability + speed), learn its weights, mine its
structure offline, and let the small model remain a pure articulator.

---

## 4. The key recognition: AGNN is *already* an (unconscious) product-t-norm fuzzy logic

This deserves to be called out separately because it's the single
highest-leverage insight from the survey.

AGNN's BA44 weight arithmetic:

```
CATEGORICAL_TRANSITIVITY: 1.0 * 1.0 = 1.0
CAUSAL_CHAIN:             0.7 * 0.7 = 0.49
FUNCTIONAL_COMPOSITION:   0.6 * 0.6 = 0.36
CONFLICT_BLEND:           (0.7 + -0.8) / 2 = -0.05  [average, not t-norm]
DIFFERENTIAL_INVERSION:   -0.8 (symmetric, no composition)
```

The first three rules are **literally the product t-norm**:
`T(a, b) = a·b`. This is one of the three standard fuzzy-logic
conjunctions (alongside Łukasiewicz `max(0, a+b-1)` and Gödel
`min(a,b)`). The conflict-blend rule is an arithmetic mean, which is
*not* a t-norm but is a recognized fuzzy aggregation operator.

**Implications:**

1. AGNN's design is not arbitrary — it's an instance of a well-studied
   framework, even if the authors didn't cite it.
2. The choice of product t-norm over Łukasiewicz or Gödel is
   consequential: van Krieken et al. (AIJ 2022) showed product t-norms
   have steeper gradients than Łukasiewicz for low-truth inputs, which
   affects rule learning quality. AGNN should make this choice
   *deliberately* and *configurable*, not implicit.
3. Once recognized as fuzzy logic, BA44's weights naturally become
   candidates for MLN-style learning (B2 above), and the rule
   structures become candidates for AnyBURL-style mining (B3).

This single recognition is the foundation for all the borrow
opportunities in §2. Documenting it explicitly in `inferior_frontal_gyrus.py`'s
docstring (citing LTN) would be the cheapest, highest-leverage change
AGNN can make to BA44.

---

## 5. Sources

**Neural Theorem Provers**
1. https://arxiv.org/abs/1705.11040 — Rocktäschel & Riedel, NTP ⭐
2. https://papers.nips.cc/paper/6969-end-to-end-differentiable-proving
3. https://www.semanticscholar.org/paper/Towards-Neural-Theorem-Proving-at-Scale-Minervini-Bosnjak/47ebd48e05efb305785b1c4f6a91bac1e891feed
4. https://arxiv.org/pdf/1705.11040

**Logic Tensor Networks / differentiable fuzzy logic**
5. https://arxiv.org/abs/1606.04422 — LTN ⭐
6. https://dl.acm.org/doi/10.1016/j.artint.2021.103649 — LTN AIJ 2022
7. https://www.sciencedirect.com/science/article/pii/S0004370221001533 — Analyzing Differentiable Fuzzy Logic Operators ⭐
8. https://en.wikipedia.org/wiki/T-norm_fuzzy_logics

**Markov Logic / ProbLog / DeepProbLog**
9. https://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf — Richardson & Domingos, MLN ⭐
10. https://proceedings.mlr.press/v161/marra21a/marra21a.pdf — Neural Markov Logic Networks
11. https://proceedings.neurips.cc/paper/2019/hash/13e5ebb0fa112fe1b31a1067962d74a7-Abstract.html — pLogicNet
12. https://arxiv.org/abs/1805.10872 — DeepProbLog ⭐
13. https://github.com/ML-KULeuven/deepproblog

**Rule mining (NeuralLP / DRUM / RNNLogic / AnyBURL)**
14. https://arxiv.org/abs/1702.08367 — NeuralLP ⭐
15. https://github.com/fanyangxyz/Neural-LP
16. https://arxiv.org/abs/1911.00055 — DRUM ⭐
17. https://github.com/alisadeghian/DRUM
18. https://arxiv.org/abs/2010.04029 — RNNLogic
19. https://github.com/DeepGraphLearning/RNNLogic
20. https://www.ijcai.org/proceedings/2019/0435.pdf — AnyBURL ⭐
21. https://web.informatik.uni-mannheim.de/AnyBURL
22. https://d-nb.info/1367508878/34 — Rule Learning over KGs: A Review

**Neural program induction / DNC (cautionary)**
23. https://github.com/google-deepmind/dnc
24. https://deepmind.google/blog/differentiable-neural-computers
25. https://www.reddit.com/r/MachineLearning/comments/qwwf82 — NTM/DNC post-mortem

**LLM + symbolic solvers / CoT-as-proving**
26. https://arxiv.org/html/2409.17270v1 — Proof of Thought ⭐
27. https://github.com/DebarghaG/proofofthought
28. https://arxiv.org/html/2508.12425v1 — Symbolic-Aided CoT
29. https://aclanthology.org/2024.alta-1.4.pdf — Tool-based logical reasoning with LLMs
30. https://www.turing.com/resources/lean-and-symbolic-reasoning-in-llms-for-math-problem-solving

**Neuro-symbolic surveys (2024-2025)**
31. https://arxiv.org/pdf/2501.05435 — Neuro-Symbolic AI in 2024: A Systematic Review ⭐
32. https://www.ijcai.org/proceedings/2025/1157.pdf — NeSy task-directed survey, IJCAI 2025
33. https://www.sciencedirect.com/science/article/pii/S2667305325000675 — Review of NeSy reasoning+learning

⭐ = highest-priority papers for AGNN-BA44 to cite as primary validation.
