# Research: Unsupervised Grammar Induction — Validation and Borrow Opportunities for PositionalClusterLearner

> **Worker 2** of a 4-worker research batch.
> **Focus:** Unsupervised / self-supervised grammar induction — how
> systems discover syntactic structure (subject/predicate/object,
> dependency parsing) from raw text WITHOUT human labels. Evaluated
> against AGNN's `PositionalClusterLearner` (PCL), which discovers
> action clusters from positional co-occurrence.
> **Status:** Research note (no code changes).
> **Author date:** 2026-06-20.

---

## 0. Scope and method

AGNN's `neocortex/positional_cluster_learner.py` (PCL v2, ~1500 lines)
is a **zero-bias emergent structure discovery** module. It learns
action clusters from positional co-occurrence in raw text:

- **Position buckets:** 0=agent, 1=action, 2=object (3-token) or
  -1=object (>3-token). Same token can appear in multiple buckets
  (soft counts) — this is the polysemy fix vs. PR #69's hard
  assignment.
- **Clustering:** weighted Jaccard similarity of object distributions,
  greedy agglomerative merge with threshold 0.13.
- **Recent (PR #74):** connector-signal detection — separates
  structurally different actions ("adalah" direct-object vs "berbeda"
  connector+object) by detecting corpus-wide connector tokens purely
  from positional/frequency statistics, no hardcoded word lists.
- **Hand-authored pieces (intentionally narrow):** `_ACTION_STOPLIST`
  (function words like "itu", "sangat", "tampak"), `_COPULAS`
  ("adalah", "merupakan", "ialah"), `_VERB_PREFIXES` (me-, ber-, di-,
  ter-).
- **Pure Python + numpy.** No torch.

This document surveys the literature on **unsupervised grammar
induction** to answer two questions:

1. Is PCL's positional-co-occurrence approach theoretically grounded,
   or naive?
2. What specific techniques could PCL borrow to improve clustering
   accuracy while staying pure-Python + numpy?

Method: 11 targeted web searches covering DMV, Brown/Clark POS
induction, ON-LSTM, PRPN, neural grammar induction, constituency
parsing, distributional structure, word-class induction, HMM POS
induction, Harris 1954, and CCM. Plus a full read of PCL's actual
code (~700 lines).

---

## 1. Key techniques found

### 1.1 Dependency Model with Valence (DMV) — Klein & Manning 2004

**Principle.** A generative probabilistic model over dependency
trees. The root POS is sampled first, then for each head the model
recurses head-outward, deciding for the left side and right side
*whether to stop generating dependents* (the "valence") and, if not,
which POS to attach. Parameters (attachment + stopping probabilities)
are estimated with EM. The breakthrough was beating the trivial
right-branching baseline — the first unsupervised dependency parser
to do so.

**Key sources:**
- Klein & Manning, "Corpus-Based Induction of Syntactic Structure:
  Models of Dependency and Constituency," ACL 2004 —
  https://sites.socsci.uci.edu/~lpearl/courses/readings/KleinManning2004_CorpusBasedInductionStructure.pdf
- Stanford project page — https://nlp.stanford.edu/projects/up-gi.shtml
- Spitkovsky "Baby Steps" NAACL 2010 —
  https://web.stanford.edu/~jurafsky/babysteps.pdf
- Headden et al. 2009 (richer contexts + smoothing → SOTA) —
  https://aclanthology.org/N09-1012.pdf

**PCL alignment — strong.** PCL's position buckets
(agent/action/object) are a *degenerate* form of DMV's head-outward
dependency structure: it implicitly fixes the root as the action slot,
allows exactly one left dependent (agent) and one right dependent
(object), and assumes valence = 1 on both sides. DMV generalizes this
to arbitrary arity. PCL's `has_connector` signal is a hand-coded
approximation of DMV's right-side valence (detecting that the head
expects a function word before its argument, i.e.
subcategorization). The DMV literature is essentially the formal
version of what PCL is doing informally.

### 1.2 Constituent-Context Model (CCM) — Klein & Manning 2002

**Principle.** Explicitly model constituent *yields* (the spanned
word sequence) and constituent *contexts* (the words immediately
outside the span). EM infers which spans are constituents by
maximizing likelihood. Spans whose internal word-pair statistics
differ from their external contexts get promoted to constituents. The
factored DMV+CCM combination was the strongest unsupervised parser of
its era.

**Key sources:**
- Klein & Manning, "A Generative Constituent-Context Model for
  Improved Grammar Induction," ACL 2002 —
  https://sites.socsci.uci.edu/~lpearl/courses/readings/KleinManning2002_GrammarInduction.pdf
- "Distributional Phrase Structure Induction," CoNLL 2001 —
  https://aclanthology.org/W01-0714.pdf

**PCL alignment — moderate-to-strong.** PCL effectively treats
positions 0 and 2/-1 as "contexts" of position 1, but it never models
the *span* structure (where does the object NP begin and end?). PCL's
`between-first` slot is a tiny glimpse of context modelling — it is
exactly the "what sits immediately to the right of the head before
the next constituent" question. CCM formalizes this: every candidate
bracketing gets a context-likelihood score.

### 1.3 Brown Clustering / Context-Distribution Clustering (POS induction)

**Principle.** Brown et al. 1992 cluster words by maximizing the
mutual information of class-based bigrams `log P(class_i |
class_{i-1})` — a hierarchical agglomerative procedure that yields a
binary tree of word classes. Clark 2000/2003 extends this with
"Context Distribution Clustering" (CDC): each word is represented as
the distribution of contexts (left/right neighbours) it occurs in,
and words are clustered by distributional similarity (KL-divergence /
symmetric). Schütze 1995 uses 4m-dimensional context vectors (m
positions × 4 features) reduced by SVD. The literature repeatedly
shows that *context distribution alone* recovers most POS categories
with ~60-80% accuracy on many-to-one metrics.

**Key sources:**
- Brown clustering overview — https://en.wikipedia.org/wiki/Brown_clustering
- Reference impl — https://github.com/percyliang/brown-cluster
- Clark 2003, "Combining Distributional and Morphological Information
  for POS Induction" — https://aclanthology.org/C08-1042.pdf
- Clark 2000 CDC — https://aclanthology.org/W00-0717.pdf
- Survey — https://arxiv.org/pdf/1801.03564

**PCL alignment — very strong; this is the closest family to PCL.**
PCL is a coarse-grained context-distribution clusterer restricted to
3-4 positional buckets. Brown/CDC generalize the same intuition to
arbitrary n-gram contexts and provide a principled objective (bigram
MI). PCL's `action_object_freq` is exactly the kind of co-occurrence
statistic Brown clustering would maximize over.

### 1.4 ON-LSTM and PRPN — neural unsupervised parsing

**Principle.** PRPN (Shen et al. 2018) introduces a differentiable
"syntactic distance" between adjacent tokens inside a language model;
thresholding the distance yields a constituency tree, trained
end-to-end on next-word prediction. ON-LSTM (Shen et al. 2019)
replaces the LSTM forget/update gates with a cumulative-max ("cumax")
scheme that forces neurons to fire in an ordered hierarchy: the
distance between the forget cumax and the update cumax at each
timestep is a soft constituency boundary. Both extract trees *for
free* from an LM objective, with no treebank.

**Key sources:**
- ON-LSTM: https://arxiv.org/abs/1810.09536 (ICLR 2019 oral) + code
  https://github.com/yikangshen/Ordered-Neurons
- PRPN: https://arxiv.org/abs/1711.02013 + code
  https://github.com/yikangshen/PRPN
- Critique: https://ui.adsabs.harvard.edu/abs/arXiv:2010.04926

**PCL alignment — weak at the *implementation* level (PCL is
pure-Python/numpy; ON-LSTM/PRPN require torch + GPU).** But the
*concept* of a continuous syntactic-distance / boundary-strength
signal between adjacent tokens is directly portable: PCL already has
the raw material (positional co-occurrence statistics) to compute a
numpy boundary-strength score without a neural net (see borrow list,
item B8).

### 1.5 Anchor HMMs — spectral, EM-free POS induction

**Principle.** Bai et al. (TACL 2016) learn an HMM for POS tagging
without EM and without local optima, using the "anchor word"
assumption: for every hidden state (POS) there exists at least one
word that appears *only* with that state. Anchor words are found via
a separability / non-negativity test on the word-context
co-occurrence matrix; transition + emission probabilities are then
recovered by a method-of-moments / spectral decomposition. No
iterative training, deterministic, no initialization games.

**Key source:**
- "Unsupervised Part-Of-Speech Tagging with Anchor Hidden Markov
  Models" — https://aclanthology.org/Q16-1018

**PCL alignment — strong and underused.** PCL already has anchor
tokens implicitly — `_COPULAS` ("adalah", "merupakan", "ialah") and
`_VERB_PREFIXES` are exactly the kind of high-purity positional
anchors Anchor HMM formalizes. The anchor framework would let PCL
*discover* these anchors from the positional distribution (tokens
whose positional_freq is concentrated in one bucket with near-1
probability) rather than hardcode them, and then propagate cluster
labels from anchors to non-anchors via co-occurrence — a principled
replacement for the greedy agglomerative merge.

---

## 2. Specific actionable borrow opportunities for PositionalClusterLearner

Concrete, ordered roughly by effort/impact. Each ties to a specific
place in the current code.

### B1. Replace greedy agglomerative merge with a few EM / Viterbi rounds (DMV-style)

Currently `_cluster_action_group` does order-dependent greedy merges
until a fixpoint. Greedy agglomerative is known to be order-fragile.
Add a post-pass: treat each cluster as a multinomial
`P(object | cluster)` with a Dirichlet prior, and re-assign each
action to the cluster maximizing `P(action's object multiset |
cluster)` for K iterations (soft or hard Viterbi). This is
implementable in pure Python + numpy, escapes local optima from the
greedy pass, and is exactly the DMV-style refinement the field
considers mandatory.

- **Effort:** low (no new deps).
- **Payoff:** high (escapes order-dependent local optima).

### B2. Brown-cluster the object vocabulary before clustering actions

The code's own comment (lines 282-305) admits the central failure:
'adalah' and 'merupakan' both take *class-noun* objects (mamalia,
reptil, ikan…) but rarely the *same* object, so even weighted Jaccard
struggles. Solution: pre-cluster the *object* vocabulary via Brown
clustering (maximize bigram MI of object-class sequences), producing
~20-50 object classes. Then represent each action as a distribution
over object-CLASSES, not raw objects. This single change should let
synonyms merge even when their literal objects never overlap. Brown
clustering is ~100 lines of Python (Percy Liang's reference C++ is
short; the algorithm is greedy agglomerative over
`P(class_i | class_{i-1})` MI). This directly generalizes the existing
`_weighted_jaccard` into a generative likelihood comparison.

- **Effort:** medium (new helper module).
- **Payoff:** high (fixes the documented central failure).

### B3. Replace `_ACTION_STOPLIST` / `_COPULAS` with anchor-word discovery

The hardcoded `_ACTION_STOPLIST` and `_COPULAS` are inconsistent with
the module's stated "zero-bias" contract. Anchor-HMM gives a
principled, data-driven replacement: a token is an "anchor" for
bucket B if `positional_freq[token][B] / sum(positional_freq[token])
≥ τ` (e.g. τ=0.9) AND its total count is above a floor. Copulas and
intensifiers will emerge as action-bucket anchors automatically; nouns
as object-bucket anchors; etc. This eliminates the most obvious
human-authored bias in the module while preserving — and usually
improving — accuracy. The connector detector at
`_compute_connector_signature` already does a similar corpus-wide
discovery for connectors; the same pattern can be applied to stoplist
words and copulas.

- **Effort:** medium.
- **Payoff:** high (closes the "zero-bias" claim that the stoplist
  currently breaks).

### B4. Add explicit valence / subcategorization frames (DMV valence + Schütze 1995)

PCL currently records only `(action, object)` and the `between_first`
token. Generalize to a *subcategorization frame signature*: for each
action, store the distribution of sentence-shape patterns it occurs
in, e.g. `{SVO_3, SVO_N, SV_connector_O, SV_only, SVOO}`. Use frame
distribution as a *second* clustering feature alongside object
distribution — combine via weighted sum in the similarity score. This
is what DMV's left/right valence captures and what Schütze's 4m-dim
context vectors encode. The `has_connector` boolean is a 1-bit version
of this; generalizing it to a frame histogram is a natural, low-effort
extension of the existing `_extract_between_token` machinery.

- **Effort:** medium.
- **Payoff:** high (adds structural dimension beyond object
  distributions).

### B5. Use context-distribution similarity (KL-divergence) instead of weighted Jaccard (Clark 2000 CDC)

Weighted Jaccard compares two count maps as `Σmin / Σmax`. Context
Distribution Clustering compares them as probability distributions via
symmetric KL-divergence `½(KL(P‖Q)+KL(Q‖P))`. KL is the
information-theoretically principled distance between distributions;
weighted Jaccard is a heuristic that over-weights high-count tokens.
Implementation: normalize each action's object-count map to a
distribution with add-λ smoothing, then compute symmetric KL. ~10
lines, no new deps. This is the single highest-leverage change to the
similarity function.

- **Effort:** low.
- **Payoff:** high (principled similarity replaces heuristic).

### B6. Baby-steps curriculum (Spitkovsky 2010)

Train `train()` in two passes: first only on sentences with exactly
3 tokens (where bucket assignment is unambiguous), then on the full
corpus. The 3-token pass produces a stable seed clustering; the
full-corpus pass refines it. Spitkovsky showed this dramatically
reduces local optima for DMV. Trivial to add: split `corpus_lines` by
token length and call the existing logic twice.

- **Effort:** low.
- **Payoff:** medium (better initialization escapes local optima).

### B7. Encode morphological features explicitly in similarity (Clark 2003)

`_VERB_PREFIXES` (me-, ber-, di-, ter-) is already in the module.
Currently it's used to *identify* actions but not in the *similarity*
computation. Add a morphology feature: two actions sharing the same
prefix family get a similarity boost. Clark 2003 showed this is a
major accuracy lever for agglutinative languages — exactly Bahasa
Indonesia's profile.

- **Effort:** low.
- **Payoff:** medium (especially for Indonesian).

### B8. Boundary-strength scores instead of hard 3-bucket assignment (PRPN/ON-LSTM concept, numpy implementation)

Instead of forcing each token into bucket 0/1/2/-1, compute a
continuous *boundary strength* `b_i` between every adjacent pair
`(tokens[i], tokens[i+1])` from co-occurrence statistics:
`b_i = -log P(tokens[i+1] | tokens[i])` normalized. High boundary
strength = constituent break; the action slot is the token with the
lowest cumulative incoming boundary strength. This is the PRPN
"syntactic distance" idea implemented with nothing but n-gram
log-probabilities. It removes the brittleness of fixed
position-1-as-action for >3-token sentences (which the code already
has special-case handling for at `spo()`).

- **Effort:** medium.
- **Payoff:** high (removes bucket-assignment brittleness).

### B9. Posterior smoothing of positional_freq with corpus-wide unigram (Headden 2009)

`positional_freq[token]` for rare tokens is noisy. Smooth each bucket
count with a corpus-wide prior:
`P_smooth(token, bucket) = (count + α · P_corpus(token)) / (total + α)`.
Headden et al. showed smoothing is the single biggest accuracy gain
for DMV. ~5 lines.

- **Effort:** low.
- **Payoff:** medium (better statistics for rare tokens).

---

## 3. Theoretical validation — is positional co-occurrence a sound signal?

**Verdict: yes, it is theoretically grounded — this is the oldest and
best-validated single cue in the literature. But PCL's current
implementation is a coarse / naive instance of an otherwise sound
principle, and the literature points to specific ways naive
positional-only approaches underperform.**

### Evidence supporting PCL's core idea

- **Harris 1954, "Distributional Structure"**
  (https://www.its.caltech.edu/~matilde/ZelligHarrisDistributionalStructure1954.pdf)
  is the foundational citation. Harris defines an element's
  *environment* as "an existing array of its co-occurrents, each in a
  particular position, with which A occurs to yield an utterance."
  This is *literally* PCL's `positional_freq` — soft counts per
  position bucket. The distributional hypothesis is the theoretical
  bedrock PCL stands on, and it is 70 years old and still standing.
- **Redington, Chater & Finch 1998** ("Distributional information: a
  powerful cue for acquiring syntactic categories,"
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3621024) empirically
  demonstrated that *positional co-occurrence alone*, with no
  semantics, recovers syntactic categories from child-directed speech.
  The Reeder/Newport/Aslin follow-ups confirm positional cues are
  sufficient.
- The entire POS-induction literature (Brown 1992, Clark 2000/2003,
  Schütze 1995, Anchor HMM 2016) is a 30-year confirmation that
  distributional/positional signals induce syntactic categories to
  ~60-80% many-to-one accuracy *without any other cue*.
- Klein & Manning 2004 and Spitkovsky's body of work show
  positional/distributional structure is enough to induce *full
  dependency trees*, not just categories.

### Where PCL is naive relative to the literature (honest assessment)

1. **Coarse positional buckets.** PCL uses 3-4 buckets
   (agent/action/object/last). Brown/CDC use n-gram contexts of
   arbitrary width; DMV uses full dependency arcs. PCL cannot
   represent internal NP structure, adjuncts, ditransitives, or
   coordination. This caps its expressive ceiling — it can cluster
   actions by *broad* argument type but cannot distinguish
   "X memakan Y dengan cepat" (manner adjunct) from "X memakan Y di
   rumah" (locative adjunct) the way a richer model could.
2. **Weighted Jaccard is a weak similarity.** The
   information-theoretically correct measure is KL-divergence or a
   generative likelihood; Jaccard (even weighted) is a heuristic that
   does not correspond to any probability model. The literature
   universally uses probabilistic similarities for this reason.
3. **Greedy agglomerative merge is order-dependent and suboptimal.**
   Every serious grammar-induction system uses EM (Klein & Manning),
   Viterbi EM (Spitkovsky), or spectral methods (Anchor HMM) — all of
   which *iterate* to escape bad initial configurations. PCL's single
   greedy pass has no such guarantee; the 0.13 threshold is a tuned
   patch over this weakness.
4. **No notion of valence / subcategorization arity.** DMV's central
   innovation — modelling *how many* dependents a head expects — is
   absent. PCL's `has_connector` is a 1-bit proxy. An action that
   takes 0 objects (intransitive) and one that takes 1 (transitive)
   are not distinguished structurally.
5. **Hardcoded `_ACTION_STOPLIST`, `_COPULAS`, `_VERB_PREFIXES`**
   contradict the "zero-bias" framing. The literature's answer is to
   *discover* these from the corpus (Anchor HMM, CDC). The connector
   detector already does this correctly for connectors — the same
   pattern should replace the stoplists.
6. **No iterative refinement or curriculum.** Spitkovsky's "Baby
   Steps" and Viterbi-EM results show that *how* you train matters as
   much as the model class. PCL trains in a single pass with no
   curriculum.
7. **Post-hoc human labelling** (`label_clusters`) is fine and
   explicitly stated as a design choice, but it means PCL is
   *semi-supervised at the cluster-naming step*, not fully
   unsupervised. This is honest in the docstring but should not be
   conflated with true unsupervised grammar induction.

### Bottom line

PCL is *theoretically aligned* with the most validated cue in
unsupervised grammar induction (Harris 1954 → Redington 1998 → Klein
& Manning 2004). It is *empirically naive* in implementation: it uses
a coarse-grained, heuristic-similarity, single-pass,
partially-handcoded version of ideas the literature has spent 30 years
optimizing.

The good news — and the actionable upshot — is that *every* weakness
above has a well-documented fix that fits PCL's pure-Python + numpy
constraint (see borrow list). None of the recommended improvements
require torch or a treebank. PCL's `has_connector` detector is
actually a small, principled innovation (corpus-wide discovery of
function-word slots from positional evidence) that the mainstream
literature would recognize as a valid contribution — it resembles DMV
valence detection done distributionally rather than parametrically.

---

## 4. Sources

**Primary papers / canonical sources**
1. https://sites.socsci.uci.edu/~lpearl/courses/readings/KleinManning2004_CorpusBasedInductionStructure.pdf — Klein & Manning 2004 (DMV + CCM, factored) ⭐
2. https://sites.socsci.uci.edu/~lpearl/courses/readings/KleinManning2002_GrammarInduction.pdf — Klein & Manning 2002 (CCM)
3. https://aclanthology.org/W01-0714.pdf — Klein & Manning, CoNLL 2001
4. https://nlp.stanford.edu/pubs/klein2004induction.pdf — Klein & Manning overview
5. https://web.stanford.edu/~jurafsky/babysteps.pdf — Spitkovsky, "Baby Steps," NAACL 2010
6. https://arxiv.org/abs/1810.09536 — Shen et al., ON-LSTM (ICLR 2019)
7. https://github.com/yikangshen/Ordered-Neurons — ON-LSTM code
8. https://arxiv.org/abs/1711.02013 — Shen et al., PRPN (2018)
9. https://aclanthology.org/W00-0717.pdf — Clark, CDC, CoNLL 2000
10. https://aclanthology.org/C08-1042.pdf — Clark 2003
11. https://arxiv.org/pdf/1801.03564 — POS induction survey
12. https://aclanthology.org/Q16-1018 — Bai et al., Anchor HMM (TACL 2016) ⭐
13. https://en.wikipedia.org/wiki/Brown_clustering — Brown clustering overview
14. https://github.com/percyliang/brown-cluster — Brown clustering reference impl
15. https://www.its.caltech.edu/~matilde/ZelligHarrisDistributionalStructure1954.pdf — Harris 1954 ⭐
16. https://pmc.ncbi.nlm.nih.gov/articles/PMC3621024 — Redington/Chater/Finch 1998

**Overviews / surveys**
17. https://nlp.stanford.edu/projects/up-gi.shtml — Stanford NLP unsupervised parsing overview
18. https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/coling20survey.pdf — Survey of unsupervised dependency parsing
19. https://aclanthology.org/N09-1012.pdf — Headden et al. 2009 (richer contexts + smoothing)
20. https://homes.cs.washington.edu/~nasmith/papers/gimpel+smith.naacl12b.pdf — Gimpel & Smith, concavity & initialization for DMV
21. https://aclanthology.org/D16-1073.pdf — Unsupervised Neural Dependency Parsing
22. https://aclanthology.org/2020.aacl-main.43 — Heads-up! Unsupervised Constituency Parsing via Self-Attention
23. https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1106/handouts/GrammarInduction2010-1up.pdf — Manning, CS224N grammar-induction lecture

⭐ = highest-priority papers for PCL to cite as primary validation.
