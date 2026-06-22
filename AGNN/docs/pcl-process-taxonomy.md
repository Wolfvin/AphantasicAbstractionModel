# PCL Process Taxonomy — Formal Names for Every Mechanism

**Date:** 2026-06-22
**Mode:** Reference document only — no code changes. Gives every distinct
algorithmic process in `PositionalClusterLearner` a formal name, grounding
it in established literature where one genuinely applies, and coining a
precise name where the mechanism is original to this project.

---

## Overarching paradigm

### **Zero-Bias Two-Stage Discovery** (cluster-first, label-second)

The governing principle behind every mechanism below. Corresponds to the
established ML paradigm of **unsupervised clustering followed by post-hoc
cluster labelling** (cf. semi-supervised learning's "pretrain unsupervised,
then attach labels"), but with NO gradient step anywhere — labelling is a
human (or scripted) review step over already-formed clusters, never a
training signal. Implemented as the structural contract every `label_*`/
`mark_*` method follows: `label_clusters()`, `label_particle_clusters()`,
`mark_clause_coordinator_clusters()`.

---

## Discovery-phase mechanisms (Stage 1 — unnamed cluster formation)

### 1. **Positional Distributional Tagging**
Recording each token's bucket position (0/1/2/-1, the coarse 4-slot
scheme) across every training sentence. Root data structure:
`positional_freq`. Directly descends from Harris's **distributional
hypothesis** (1954) — "words that occur in similar positional contexts
share grammatical properties" — applied at the level of a fixed
sentence-position slot rather than a sliding window.

### 2. **Distributional Positional Anchoring (DPA)**
The discovery test underlying `_compute_anchor_words`: a token qualifies
as an "anchor" for a bucket when it shows **low Shannon entropy** over its
positional distribution (concentrated, not spread out) AND clears a
frequency floor. This is a direct, mechanical application of **Shannon
entropy** (1948) as a concentration/skew measure — the same mathematical
tool used in information theory for measuring distributional surprise,
repurposed here as a *structural regularity detector*. Generalised in
round 9 from "only the verb-slot bucket" to **all buckets** —
`bucket_anchors: Dict[int, Set[str]]`.

### 3. **Entropy-Based Function Word Discovery (EFWD)**
The inverse application of (2): tokens with **HIGH** positional entropy
(spread across many slots, not concentrated) plus a frequency floor are
flagged as `function_word_candidates`. In corpus linguistics this
corresponds to identifying the **closed grammatical class** (determiners,
prepositions, conjunctions) via distributional regularity rather than a
hand-built stop-word list — the zero-bias replacement for what older NLP
pipelines did with a fixed stoplist.

### 4. **Brown Clustering** (distributional class induction)
`_cluster_object_vocabulary` / `object_supercluster_id`. This one keeps
its real, established name — it IS the classic **Brown clustering**
algorithm (Brown, Della Pietra, Della Pietra & Mercer, 1992): hierarchical
merging of word classes by co-occurrence similarity, used here to project
the sparse literal-object vocabulary into denser "super-cluster" ids
before the Q/K/V step.

### 5. **Sequential Q/K/V Soft Clustering** (a.k.a. **Attention-Analogous
Incremental Clustering**)
`_cluster_action_group_qkv`. Mathematically: **sequential/online
leader-clustering** (Hartigan's "Leader algorithm," 1975) — process items
in descending-frequency order, compare each to existing cluster centroids
via **cosine similarity**, assign to the best match above a threshold or
seed a new singleton cluster. The "Query/Key/Value" naming borrows
*structure* from transformer self-attention (query vs. key vs. value
roles, softmax-normalised scores) without any of its machinery — no
learned projection matrices, no backpropagation. More precisely: a
**zero-parameter, cosine-scored greedy online clustering algorithm**.

### 6. **Stratified Pre-Clustering Partitioning**
`action_connector_signature`: split the clusterable population into two
disjoint groups by a binary structural feature BEFORE clustering, so the
two groups can never merge regardless of similarity score. This is
**stratified sampling/partitioning** (a standard statistics term) applied
as a clustering precondition rather than a sampling step.

### 7. **Structural Context Stratification (SCS)**
Round 19's generalisation of (6) down to the level of individual
*observations* rather than whole tokens: `action_object_context_freq`
tags each `(action, object)` co-occurrence by a purely structural test
(is there a non-particle token between them?) before the signatures are
aggregated. Names a genuinely original mechanism — the closest analogue
in the literature is **context-conditioned co-occurrence counting**, used
in some word-sense-induction work, but applied here to *syntactic*
structure rather than semantic sense.

### 8. **Seed-Anchored Euclidean Particle Clustering**
`_cluster_particles`. Particles cluster by **negative Euclidean distance**
over a 4-dimensional structural feature vector (pre-object rate,
between-first rate, fine-position entropy, bucket entropy), seeded
sequentially the same way as (5) but with a *different* distance metric —
chosen after an empirically documented failure of cosine similarity on
this specific low-dimensional feature space (cosine produced false-
positive merges; Euclidean did not).

---

## Structural-pattern mechanisms (positional facts used as detectors)

### 9. **Anchor-Based Lazy Clause Segmentation (ALCS)**
`_detect_clause_anchors` / the lazy anchor-split mechanism (rounds 1, 4,
5). A simplified, zero-training analogue of **clause boundary
chunking**/shallow parsing — instead of a trained sequence tagger, a
particle (or, after round 5, a marked coordinator) is treated as a
boundary when it sits between two recognised predicates, or at the
sentence edge with a predicate on one side.

### 10. **Structural Operator Cluster Tagging**
`mark_clause_coordinator_clusters` (round 5). Post-hoc flagging of an
ACTION-bucket cluster as a *syntactic operator* (joins two clauses) rather
than a *semantic predicate* — distinguishes "this cluster behaves like a
predicate positionally but functions like a conjunction structurally."

### 11. **Null-Argument Signature Convergence**
Round 9's trick for predicate-final adjectives: record the (action,
object) pair with an **empty-string object sentinel** (`""`) when no real
object exists, so every intransitive/no-object predicate gets an
*identical* signature and therefore converges to the same Q/K/V cluster
purely by similarity — no adjective word list, no POS category named in
advance.

### 12. **Generalised Positional Predicate Discovery (GPPD)**
The round-9 architectural shift this enabled: extending the predicate-
extraction fallback to check `bucket_anchors[-1]` (sentence-final
position) when no mid-clause action is found, instead of assuming every
predicate sits in the historically-privileged ACTION bucket.

---

## Output/inference-phase mechanisms (Stage 2 — consuming labelled clusters)

### 13. **Multi-Clause Semantic Role Extraction**
`spo_all()` / `_parse_all_clauses` (round 6). A zero-training analogue of
**Semantic Role Labelling (SRL)** restricted to AGENT/ACTION/OBJECT roles,
extended to return one role-assignment per *independent clause* rather
than collapsing a multi-clause sentence into one tuple.

### 14. **Polysemy-Preserving Role Assignment**
The standing design rule (present since before round 1, re-confirmed
throughout): a token's AGENT-vs-OBJECT role is computed from its position
**in the current sentence**, never from a global per-token lookup — so
the same noun can be AGENT in one sentence and OBJECT in another. This is
the project's version of **context-dependent role disambiguation**,
achieved by construction (positional, not by a disambiguation model).

---

## Audit/verification mechanisms (round 19's contribution)

### 15. **Context-Split Cosine Diagnostic**
`inspect_context_split()`. Not a clustering step — a **post-hoc
similarity diagnostic**: cosine similarity between two
already-stratified sub-signatures (see #7), used to verify whether a
hypothesised structural distinction (e.g. embedded-predicate fragment vs.
ordinary object) is actually visible in the data BEFORE any code commits
to acting on it.

---

## Summary table

| # | Formal name | Code location | Established analogue |
|---|---|---|---|
| 1 | Positional Distributional Tagging | `positional_freq` | Harris's distributional hypothesis |
| 2 | Distributional Positional Anchoring | `_compute_anchor_words`, `bucket_anchors` | Shannon entropy as concentration measure |
| 3 | Entropy-Based Function Word Discovery | `function_word_candidates` | Closed-class detection via distributional regularity |
| 4 | Brown Clustering | `_cluster_object_vocabulary` | Brown et al. 1992 (verbatim) |
| 5 | Sequential Q/K/V Soft Clustering | `_cluster_action_group_qkv` | Leader algorithm (Hartigan) + cosine scoring |
| 6 | Stratified Pre-Clustering Partitioning | `action_connector_signature` | Stratified partitioning |
| 7 | Structural Context Stratification | `action_object_context_freq` | Context-conditioned co-occurrence (novel application) |
| 8 | Seed-Anchored Euclidean Particle Clustering | `_cluster_particles` | Leader algorithm, Euclidean variant |
| 9 | Anchor-Based Lazy Clause Segmentation | `_detect_clause_anchors` | Clause boundary chunking (zero-training analogue) |
| 10 | Structural Operator Cluster Tagging | `mark_clause_coordinator_clusters` | Original |
| 11 | Null-Argument Signature Convergence | round-9 empty-object sentinel | Original |
| 12 | Generalised Positional Predicate Discovery | `bucket_anchors[-1]` fallback | Original |
| 13 | Multi-Clause Semantic Role Extraction | `spo_all` | Semantic Role Labelling (zero-training analogue) |
| 14 | Polysemy-Preserving Role Assignment | per-sentence role computation | Context-dependent role disambiguation (by construction) |
| 15 | Context-Split Cosine Diagnostic | `inspect_context_split` | Post-hoc similarity diagnostic |

---

## What is deliberately NOT named here

Hyperparameters/thresholds (`_ACTION_ANCHOR_MIN_FREQ`,
`qkv_action_similarity_threshold`, etc.) are calibration constants, not
processes — naming them would imply a theoretical status they don't have.
They are documented inline at their definition sites instead.
