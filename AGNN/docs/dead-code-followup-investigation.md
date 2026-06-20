# Dead-Code Audit — Follow-up Investigation

**Date:** 2026-06-20
**Scope:** Targeted investigation of three "perlu verifikasi manual"
items flagged in `AGNN/docs/dead-code-audit.md` (§4.1, §4.2, §5.2).
**Mode:** **Investigation only — no production code modified.** The
single test file added under `AGNN/tests/` is an *investigation
artifact* that encodes the current (buggy) behaviour; it is NOT a
regression guard against future fixes (see §1.4 below).
**Sibling worklogs:** `/home/z/my-project/worklog.md` (Task IDs
`audit-followup-1`).

---

## 0. TL;DR

| Section | Audit claim | Investigation finding | Verdict |
|---|---|---|---|
| **§4.1** | CA1 fallback override firing condition is stale post-PCL migration; risk of misfire on correct CATEGORICAL sentences. | **Misfire is REAL and reproducible** on the current committed cluster state. PCL correctly returns CATEGORICAL for Indonesian `adalah`/`merupakan`/`termasuk` predicates, but CA1's English-only cue table overrides the classification whenever the *stimulus* (user's question) contains an English cue-word like `causes`, `requires`, `affects`. 3 distinct misfire cases proven by `test_audit_4_1_ca1_fallback_misfire.py`. **Scope on the committed corpus is narrow (2/3290 = 0.06% of corpus lines), but the failure mode is open-ended on real user input.** | Misfire confirmed. No fix applied. |
| **§4.2** | `self-ai/test_training_agent.py` imports `benchmark/` and `benchmark_empiris` which do not exist in this repo; was it deliberately removed or is it a sibling repo? | **Deliberately archived.** `self-ai/benchmark/` (9 files, 2,367 LOC) was moved to `archive/self-ai-v1/benchmark/` in commit `56aaad7` (2026-06-15, "Pivot to AGNN: archive v1, remove dead code"). The commit message explicitly lists `benchmark/` under "Archive v1 — moved to archive/self-ai-v1/". `test_training_agent.py` was left behind in `self-ai/` but never re-targeted at the new path. | Dangling reference left by the AGNN pivot. |
| **§5.2** | PCL coverage gap: only 6.9% of corpus verbs hit the PCL primary path. How much additional corpus is needed to reach 30% / 50% coverage? | **Coverage uplift depends on the strategy.** Three strategies modelled: (A) label more existing PCL clusters — 0 corpus lines, ~17 manual labels for 30% / ~31 for 50%. (B) expand corpus only — ~10,900 more lines for 30% / ~20,500 for 50%, with low expected uplift per line because labelled clusters are gated by `EXPECTED_VERB_GROUPS` (a fixed canonical set). (C) combined (widen `EXPECTED_VERB_GROUPS` + generate matching corpus) — ~3,400–4,100 new lines for 30% / ~5,300–6,200 for 50%. | Strategy A is the cheapest by 4 orders of magnitude. |

---

## 1. §4.1 — CA1 fallback override misfire

### 1.1 What the audit said

> **File:** `AGNN/circuits/trisynaptic_circuit.py:255-263` (~9 LOC)
> **Code:** when the classifier returns `RelationType.CATEGORICAL` and
> the correction is non-empty, the circuit calls
> `self.ca1.integrate_context(stimulus, correction)` and overrides the
> edge_type if CA1 disagrees.
> **History:** originally designed for the SemanticRoleClassifier-as-primary
> era, when SRC returning CATEGORICAL often meant *"I gave up, here's the
> default."* With PositionalClusterLearner as the primary, this branch now
> fires for two distinct cases: (1) PCL's labelled cluster 60 returned
> CATEGORICAL (a *correct* classification of "X adalah Y" — CA1 may then
> MIS-override it based on its own cue-word scan of the stimulus); (2) PCL
> fell back to SRC which returned CATEGORICAL as its default.
> **Confidence:** perlu verifikasi manual — not dead, but the firing
> condition is stale after the PCL migration.

### 1.2 The code under investigation

`AGNN/circuits/trisynaptic_circuit.py:253-263`:

```python
relation = self.role_classifier.classify(correction or stimulus)
edge_type = relation.name
if (relation == RelationType.CATEGORICAL
        and correction.strip()):
    # Classifier fell back to default. Give CA1 one more shot
    # at the combined stimulus + correction text - its cue
    # list includes tokens ("causes", "requires") that may
    # appear in the stimulus half of the input.
    ca1_type = self.ca1.integrate_context(stimulus, correction)
    if ca1_type != "CATEGORICAL":
        edge_type = ca1_type
```

`AGNN/hippocampus/ca1.py:31-53` — CA1's cue table is **English-only**:

```python
_CUES: Dict[str, frozenset] = {
    "CATEGORICAL": frozenset({"is", "are", "was", "were", "member", ...}),
    "CAUSAL": frozenset({"causes", "caused", "cause", "leads", ...}),
    "DIFFERENTIAL": frozenset({"not", "unlike", "contrasts", ...}),
    "FUNCTIONAL": frozenset({"requires", "enables", "uses", ...}),
}
```

### 1.3 PCL state on the committed cluster file

`AGNN/data/cluster_learner_state.json` loads a `PositionalClusterLearner`
with `is_trained=True`, `is_labelled=True`, 305 distinct action tokens,
5 labelled clusters:

| Cluster ID | Label | Tokens |
|---|---|---|
| 42 | CAUSAL | `berakibat`, `membuat`, `memicu`, `mengakibatkan`, `menghasilkan`, `menyebabkan` |
| 57 | FUNCTIONAL | `membutuhkan`, `memerlukan`, `perlu`, `tergantung` |
| 60 | CATEGORICAL | `adalah`, `merupakan`, `termasuk` |
| 98 | TEMPORAL | `kemudian`, `ketika`, `lalu`, `saat`, `sebelum`, `setelah` |
| 124 | DIFFERENTIAL | `berbeda`, `berlawanan` |

The audit's "6.9%" coverage figure (=21/305) is reproducible from this
file. (Note: there is a JSON type mismatch in the state file —
`cluster_labels` keys are strings while `cluster_id_of` values are ints.
The PCL's `load()` method normalizes this internally so the runtime is
unaffected, but raw JSON inspection requires `int(k) for k in labels`.)

### 1.4 Misfire proof — `AGNN/tests/test_audit_4_1_ca1_fallback_misfire.py`

The new test file (added in this PR) is **NOT a regression guard** —
it is an *investigation artifact* that encodes the current misfire so
the failure mode is documented in executable form. The assertions
deliberately check `episome.edge_type == "CAUSAL"` (the wrong value)
for cases where the correct answer is `"CATEGORICAL"`. If/when the
audit's suggested fix is applied, those assertions will fail and the
test must be flipped to `== "CATEGORICAL"`.

**Test results (all 7 tests pass on the committed cluster state):**

```
AGNN/tests/test_audit_4_1_ca1_fallback_misfire.py::
  test_audit_4_1_pcl_correctly_classifies_pure_indonesian_categorical   PASSED
  test_audit_4_1_misfire_on_stimulus_with_english_causal_cue            PASSED
  test_audit_4_1_misfire_on_stimulus_with_english_functional_cue        PASSED
  test_audit_4_1_misfire_on_stimulus_with_english_affects_cue           PASSED
  test_audit_4_1_no_misfire_when_stimulus_has_no_english_cue            PASSED
  test_audit_4_1_no_misfire_when_cue_ties_with_categorical              PASSED
  test_audit_4_1_corpus_scope_estimate                                   PASSED

7 passed in 0.61s
```

The three misfire tests construct the exact scenario the audit
described:

| Test | Stimulus (user question) | Correction (taught fact) | PCL alone | CA1 alone | `episome.edge_type` (final) | Misfire? |
|---|---|---|---|---|---|---|
| `english_causal_cue` | `What causes Hamilton to be a city?` | `hamilton merupakan kota di selandia baru` | `CATEGORICAL` (correct) | `CAUSAL` (English cue `causes`) | **`CAUSAL`** (wrong) | **YES** |
| `english_functional_cue` | `What requires Hamilton to be a city?` | `hamilton merupakan kota di selandia baru` | `CATEGORICAL` (correct) | `FUNCTIONAL` (English cue `requires`) | **`FUNCTIONAL`** (wrong) | **YES** |
| `english_affects_cue` | `What affects Hamilton's status?` | `hamilton merupakan kota di selandia baru` | `CATEGORICAL` (correct) | `CAUSAL` (English cue `affects`) | **`CAUSAL`** (wrong) | **YES** |

### 1.5 Negative controls (proving the misfire scope is bounded)

Two negative-control tests prove the misfire does NOT fire when CA1
agrees with PCL:

| Test | Stimulus | Correction | CA1 returns | Final `edge_type` | Why no misfire |
|---|---|---|---|---|---|
| `no_misfire_when_stimulus_has_no_english_cue` | `Apakah Hamilton sebuah kota?` | `hamilton merupakan kota di selandia baru` | `CATEGORICAL` (no English cues) | `CATEGORICAL` | CA1 agrees with PCL |
| `no_misfire_when_cue_ties_with_categorical` | `What is not Hamilton?` | `hamilton merupakan kota di selandia baru` | `CATEGORICAL` (tie `is`=1 vs `not`=1, dict-order tiebreak) | `CATEGORICAL` | CA1 ties at CATEGORICAL due to dict ordering |

The second negative control is a *coincidence* — Python's `max`
returns the first key with the highest score, and `CATEGORICAL`
happens to be first in the `_CUES` dict. A future re-ordering of
`CA1._CUES` would silently turn this into a misfire. The test
documents this dependency so any such re-ordering surfaces as a test
failure.

### 1.6 Misfire scope on the committed corpus

The `test_audit_4_1_corpus_scope_estimate` test scans
`AGNN/data/pretrain_corpus.txt` + `pretrain_corpus_depth.txt` (3,290
lines total) and counts how many lines contain at least one of CA1's
non-CATEGORICAL English cue-words.

| Metric | Count | % |
|---|---|---|
| Total corpus lines (excl. comments) | 3,290 | — |
| Lines containing any CA1 cue-word | 2 | 0.06% |
| Lines containing a non-CATEGORICAL CA1 cue-word (misfire-eligible if used as stimulus) | 2 | 0.06% |

The 2 misfire-eligible lines are:

```
lumut masuk golongan tumbuhan non-pembuluh
lumut tergolong ke dalam tumbuhan non-pembuluh
```

Both contain the substring `non-`, which CA1's `DIFFERENTIAL` cue
table matches as the cue `non` (via the regex `\bnon\b` — actually,
since the line contains `non-pembuluh`, the regex match would succeed
only if the hyphen is treated as a word boundary, which Python's
`\b` does). In practice these are Indonesian CATEGORICAL statements
("lumut tergolong tumbuhan non-pembuluh" = "mosses are classified as
non-vascular plants"), so if they were used as a stimulus alongside
a separate CATEGORICAL correction, CA1 would override the
classification to `DIFFERENTIAL`.

**Conclusion on corpus scope:** the misfire cannot be reproduced by
running `encode()` on the committed corpus lines themselves (the
corpus is overwhelmingly Indonesian and CA1's cues are
English-only). The misfire is realistic on **user input** —
specifically bilingual scenarios where the user asks a question in
English ("What causes X?") and the correction is taught in Indonesian
("X merupakan Y"). This is exactly the scenario the audit's
hypothetical risk described.

### 1.7 Why this matters

The misfire changes the `edge_type` stored on the Episome and in the
AGNNGraph `TypedEdge`. Downstream, `InferiorFrontalGyrus` (BA 44)
selects which deductive rule to fire based on `edge_type`:

- `CATEGORICAL_TRANSITIVITY` fires only on chains of `CATEGORICAL`
  edges (test: `test_categorical_chain_fires_categorical_transitivity`).
- `CAUSAL_CHAIN` fires only on chains of `CAUSAL` edges.
- Mixed chains fire neither (negative control:
  `test_mixed_chain_does_not_fire_either_transitivity_rule`).

A CATEGORICAL fact that was mis-encoded as CAUSAL will:
1. Fail to fire `CATEGORICAL_TRANSITIVITY` in a categorical chain
   (breaking positive deduction).
2. Spuriously enable `CAUSAL_CHAIN` if it happens to chain with
   another CAUSAL-misencoded fact (false positive deduction).
3. Trigger `CingulateGyrus` conflict detection when retrieved
   alongside correctly-encoded CATEGORICAL facts (false conflict
   alarm).

### 1.8 Verdict on §4.1

The audit's claim is **confirmed by executable evidence** — not just
theoretically. The misfire fires on the current committed cluster
state for any bilingual (English-stimulus + Indonesian-correction)
input where the stimulus contains a CA1 cue-word that maps to a
non-CATEGORICAL relation type.

Per the task constraint ("JANGAN fix kalau memang ditemukan masalah,
cukup laporkan"), no production code was modified. The audit's
suggested fix (gate the override on a PCL-side
`_last_classification_was_fallback` flag) is one valid remediation;
an alternative would be to restrict the override to fire only when
`role_classifier` is exactly a `SemanticRoleClassifier` (not a PCL),
preserving the original "SRC fell back to default" design intent
without requiring PCL to expose new state.

---

## 2. §4.2 — `self-ai/test_training_agent.py` broken imports

### 2.1 What the audit said

> **File:** `self-ai/test_training_agent.py:20, 216`
> **Issue:** `sys.path.insert(0, .../benchmark)` references a directory
> that does not exist in this repo; line 216
> `from benchmark_empiris import TEST_SOAL` references a module that
> does not exist anywhere in the repo.
> **Confidence:** perlu verifikasi manual — was `benchmark/` deliberately
> removed (in which case PoC 3 should be removed or rewritten), or is it
> a sibling repo not visible in this sparse checkout (in which case the
> script is fine and the import works at runtime)?
> **Recommendation:** confirm intent before any change.

### 2.2 Git history investigation

`self-ai/benchmark/` **was** in the repo at one point. Full history
of any path matching `*benchmark*`:

```
8d5ee25 2026-06-13 refactor: reorganize repo — move existing content to aam/, add self-ai/
         ADD: self-ai/benchmark/  (9 files, 2,367 LOC)
              ├── BENCHMARK_V38_REPORT.md
              ├── KELAS4_BENCHMARK.md
              ├── adversarial_results.json
              ├── benchmark_data_bottleneck.py
              ├── benchmark_empiris.py        ← contains TEST_SOAL at line 45
              ├── data_bottleneck_results.json
              ├── empirical_v38_results.json
              ├── run_empiris.py
              └── training_agent_test_results.json

8eeaf56 (later) feat(benchmark): v2 honest metrics — keyword_hit_rate + answer_alignment
         ADD: self-ai/benchmark/l2_results.json
         (and contextual_results.json later)

a3ba13f (later) fix(benchmark): set HF_HUB_OFFLINE + fix embedding_model_loaded flag
         MOD: self-ai/benchmark/contextual_results.json

56aaad7 2026-06-15 Pivot to AGNN: archive v1, remove dead code, create AGNN foundation
         DELETE: self-ai/benchmark/  (11 files, 2,415 LOC removed)
              (all 9 original + 2 added later = 11 files)
```

### 2.3 Was the deletion intentional?

**Yes — explicitly documented in the commit message.** Commit
`56aaad7` ("Pivot to AGNN: archive v1, remove dead code, create AGNN
foundation") body says, under "FASE 2 — CLEANUP + FONDASI BARU":

> **A. Archive v1 — moved to `archive/self-ai-v1/`:**
> - src/unconscious/ (injector, projection_trainer, training_pairs_dataset)
> - src/api/ (FastAPI server v1)
> - 6 v1-specific test files
> - poc_v1.1.py, requirements-api.txt
> - agent-ctx/, docs/plans/, **`benchmark/`**

So `benchmark/` was **archived** (not deleted). The full content was
moved to `archive/self-ai-v1/benchmark/`, which still exists in HEAD:

```
$ git ls-tree -r HEAD --name-only | grep 'archive/self-ai-v1/benchmark'
archive/self-ai-v1/benchmark/BENCHMARK_V38_REPORT.md
archive/self-ai-v1/benchmark/KELAS4_BENCHMARK.md
archive/self-ai-v1/benchmark/adversarial_results.json
archive/self-ai-v1/benchmark/benchmark_data_bottleneck.py
archive/self-ai-v1/benchmark/benchmark_empiris.py          ← contains TEST_SOAL
archive/self-ai-v1/benchmark/contextual_results.json
archive/self-ai-v1/benchmark/data_bottleneck_results.json
archive/self-ai-v1/benchmark/empirical_v38_results.json
archive/self-ai-v1/benchmark/l2_results.json
archive/self-ai-v1/benchmark/run_empiris.py
archive/self-ai-v1/benchmark/training_agent_test_results.json
```

### 2.4 Why does `test_training_agent.py` still import the old path?

`self-ai/test_training_agent.py` was added in the *same commit* as
`self-ai/benchmark/` (commit `8d5ee25`, 2026-06-13). When commit
`56aaad7` archived the benchmark directory, the test file was **left
in place at `self-ai/test_training_agent.py`** (NOT moved to
`archive/self-ai-v1/`) but its imports were never updated to point
at the new archive location.

Two specific dangling references in `self-ai/test_training_agent.py`:

| Line | Code | Issue |
|---|---|---|
| 20 | `sys.path.insert(0, os.path.join(PROJECT_ROOT, 'benchmark'))` | `PROJECT_ROOT = os.path.dirname(__file__)` = `self-ai/`, so this inserts `self-ai/benchmark/` which does not exist. |
| 216 | `from benchmark_empiris import TEST_SOAL` | `benchmark_empiris` module is not on `sys.path` (the `sys.path.insert` on line 20 is a no-op because the directory doesn't exist). Import will raise `ModuleNotFoundError` at runtime. |

### 2.5 Production impact

`self-ai/test_training_agent.py` is a **standalone script**, not
imported by any production code or by any pytest test under
`AGNN/tests/` or `self-ai/tests/`. Confirmed by grep across the
sparse checkout:

- No `import test_training_agent` anywhere.
- No `from test_training_agent` anywhere.
- The file is invoked manually (`python self-ai/test_training_agent.py`)
  per its docstring ("This script runs a real session programmatically
  (not interactive CLI).").

So the broken import does not break any test or production path.
Running the script directly will:

- PoC 1 and PoC 2 run fine (they don't reach line 216).
- PoC 3 (accuracy improvement) crashes at line 216 with
  `ModuleNotFoundError: No module named 'benchmark_empiris'`.

### 2.6 Verdict on §4.2

The audit's two hypotheses resolve as follows:

| Hypothesis | Verdict |
|---|---|
| (a) `benchmark/` was deliberately removed | **Correct** — explicitly archived in commit `56aaad7` to `archive/self-ai-v1/benchmark/`. |
| (b) `benchmark/` is a sibling repo not visible in this sparse checkout | **Wrong** — `archive/self-ai-v1/benchmark/` IS visible in this sparse checkout (the `archive/` directory is at repo root, alongside `AGNN/` and `self-ai/`). |

The script is therefore **fixable in two ways** (per the task
constraint, no fix is applied here — only the diagnosis is
reported):

1. **Update the import path** to point at `archive/self-ai-v1/benchmark/`:
   change line 20 to `sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', 'archive', 'self-ai-v1', 'benchmark'))`.
   PoC 3 would then run against the archived `TEST_SOAL` dataset.

2. **Move `test_training_agent.py` itself to `archive/self-ai-v1/`**,
   alongside the benchmark it depends on. This is more consistent
   with the AGNN pivot's "archive v1" intent — the test is a v1-era
   artifact that should not be in the active `self-ai/` tree.

Per the audit's stated PoC3 impact ("PoC 3 would crash at line
216"), the diagnosis is **confirmed**: PoC 3 indeed crashes, and the
crash is reproducible at runtime. The script's first two PoCs run
fine.

---

## 3. §5.2 — PCL coverage gap (corpus growth estimation)

### 3.1 What the audit said

> Not dead code, but a structural finding worth flagging: 93% of the
> pretrain-corpus verbs fall back to `SemanticRoleClassifier`. Either (a)
> re-run `python -m neocortex.bootstrap_classifier` with a wider corpus
> and broader labelling rules to lift PCL coverage, or (b) accept the
> current two-tier design where PCL is a fast path for the 7% most-common
> verbs and SRC handles the long tail. Both are defensible; the current
> state leans on (b) by default.

The investigation task: estimate how much additional corpus would be
needed to lift coverage to 30% / 50%. This is **an estimation, not a
decision to widen the corpus**.

### 3.2 Current baseline (recomputed from
`AGNN/data/cluster_learner_state.json`)

| Bucket | Tokens | % |
|---|---|---|
| Total distinct action tokens tracked by PCL | 305 | 100.00% |
| In labelled clusters (PCL primary path — "coverage") | 21 | 6.89% |
| In unlabelled clusters (PCL clusters exist, but `cluster_labels` doesn't map them → fallback to SRC) | 137 | 44.92% |
| Unclustered (`cluster_id == -1`, below `min_action_observations=2` → fallback to SRC) | 147 | 48.20% |

Labelled cluster contents (5 clusters, 21 tokens total):

| Cluster ID | Label | Tokens | Size |
|---|---|---|---|
| 42 | CAUSAL | `berakibat`, `membuat`, `memicu`, `mengakibatkan`, `menghasilkan`, `menyebabkan` | 6 |
| 57 | FUNCTIONAL | `membutuhkan`, `memerlukan`, `perlu`, `tergantung` | 4 |
| 60 | CATEGORICAL | `adalah`, `merupakan`, `termasuk` | 3 |
| 98 | TEMPORAL | `kemudian`, `ketika`, `lalu`, `saat`, `sebelum`, `setelah` | 6 |
| 124 | DIFFERENTIAL | `berbeda`, `berlawanan` | 2 |
| | | **Average labelled cluster size** | **4.2** |

PCL cluster size distribution (305 tokens across 126 clusters):

| Cluster size | # of clusters | # labelled |
|---|---|---|
| 1 (singleton) | 110 | 0 |
| 2 | 8 | 1 (cluster 124 = DIFFERENTIAL) |
| 3 | 2 | 1 (cluster 60 = CATEGORICAL) |
| 4 | 2 | 1 (cluster 57 = FUNCTIONAL) |
| 6 | 3 | 2 (clusters 42 + 98 = CAUSAL + TEMPORAL) |
| 147 (the unclustered bucket, `cluster_id == -1`) | 1 | 0 |

**Key observation:** 110 of 126 clusters are singletons. The PCL
clustering algorithm (greedy agglomerative merge with
`similarity_threshold=0.13`) is conservative — most distinct action
tokens do not find a morphological/positional partner above
threshold and end up either as a singleton cluster or unclustered.

### 3.3 Why corpus growth alone does NOT increase coverage

`AGNN/neocortex/bootstrap_classifier.py` defines
`EXPECTED_VERB_GROUPS` — a *fixed* canonical mapping of 5
RelationTypes to specific verb sets (lines 93-110):

```python
EXPECTED_VERB_GROUPS: Dict[RelationType, Set[str]] = {
    RelationType.CAUSAL: {"berakibat", "membuat", "memicu",
                          "mengakibatkan", "menghasilkan", "menyebabkan"},
    RelationType.FUNCTIONAL: {"membutuhkan", "memerlukan", "perlu", "tergantung"},
    RelationType.CATEGORICAL: {"adalah", "merupakan", "termasuk"},
    RelationType.TEMPORAL: {"kemudian", "ketika", "lalu", "saat",
                            "sebelum", "setelah"},
    RelationType.DIFFERENTIAL: {"berbeda", "berlawanan"},
}
```

`build_labelled_cluster_learner()` finds the cluster whose action set
is a *superset* of each expected group, and labels only those 5
clusters. **Adding corpus lines does NOT change `EXPECTED_VERB_GROUPS`**
— it only changes which action tokens PCL tracks and how they cluster.

There are three distinct mechanisms by which coverage can grow:

1. **Token merge** — a new corpus verb that is morphologically similar
   to an existing labelled verb (e.g. `menyebabkan` → `sebabkan`) may
   cross the `similarity_threshold=0.13` and merge into the labelled
   cluster 42 (CAUSAL), becoming a labelled token "for free".
2. **Cluster labelling** — a human operator (or an expanded
   `EXPECTED_VERB_GROUPS`) labels an existing unlabelled cluster,
   turning its tokens from "in unlabelled cluster" to "labelled"
   without any corpus change.
3. **`min_action_observations` lowering** — currently 2; lowering to
   1 would promote unclustered tokens (those with only 1 corpus
   occurrence) into singleton clusters. They would still be
   unlabelled, so coverage does not change unless paired with
   mechanism 2.

### 3.4 Coverage uplift estimation — three strategies

Target coverage:

- 30% = 91 labelled tokens (= `0.30 × 305`; gap from current 21 = 70)
- 50% = 152 labelled tokens (= `0.50 × 305`; gap from current 21 = 131)

#### Strategy A — Label more existing PCL clusters (no corpus change)

| Action | Cost | Reach 30% | Reach 50% |
|---|---|---|---|
| Human inspects `learner.inspect_cluster_details()` and manually labels N existing unlabelled clusters | 0 corpus lines; ~1-2 hours of human time per cluster (inspect + label + verify) | Label `(91-21) / 4.2 ≈ 17` more clusters | Label `(152-21) / 4.2 ≈ 31` more clusters |

**Feasibility:** High. 120 unlabelled clusters exist (8 of size ≥ 2 + 110 singletons + the unclustered bucket). Labelling 17-31 of the size-≥-2 clusters is mechanical.

**Caveat:** Requires extending `bootstrap_classifier.EXPECTED_VERB_GROUPS` with the new verb sets, or writing a separate "manual labelling" entrypoint that bypasses the superset-match contract.

#### Strategy B — Expand corpus only (no `EXPECTED_VERB_GROUPS` change)

Empirical baseline: 3,290 corpus lines → 21 labelled tokens (hit rate
= 0.64% of lines, or 1 labelled token per ~157 corpus lines). This
includes the depth corpus which was *designed* to maximise the 5
canonical verb sets' cluster cohesion — arbitrary corpus expansion
will have a *lower* hit rate.

Optimistic assumption: hit rate stays at 0.64% (linear growth).

| Target | Tokens to add | Corpus lines needed (linear extrapolation) |
|---|---|---|
| 30% | +70 labelled tokens | ~10,900 new corpus lines |
| 50% | +131 labelled tokens | ~20,500 new corpus lines |

Realistic assumption: hit rate drops to ~0.3% (the depth corpus was
already targeted at the canonical verbs; generic corpus expansion
will mostly add new singleton clusters, not merge into labelled
clusters). Lines needed: ~23,000 for 30%, ~43,000 for 50%.

**Feasibility:** Low. The hit rate is gated by
`similarity_threshold=0.13` and the morphological co-occurrence
patterns of Indonesian verbs; corpus expansion yields diminishing
returns unless paired with strategy A or C.

#### Strategy C — Combined: widen `EXPECTED_VERB_GROUPS` + generate matching corpus

Each new verb group added to `EXPECTED_VERB_GROUPS` requires:
- ~6 new verbs selected to form a cohesive cluster (matches the
  canonical groups' size of 2-6 verbs).
- ~240 corpus sentences using those verbs (matches the depth corpus
  format: 6 predicates × 40 sentences = 240 lines per pattern).
- ~4-6 labelled tokens added to coverage (1 new cluster × avg
  labelled cluster size 4.2).

| Target | New groups needed | New corpus lines needed |
|---|---|---|
| 30% | (91-21) / 4.2 ≈ 14-17 new groups | 14-17 × 240 ≈ **3,400-4,100 new corpus lines** |
| 50% | (152-21) / 4.2 ≈ 22-26 new groups | 22-26 × 240 ≈ **5,300-6,200 new corpus lines** |

**Feasibility:** Medium. Requires identifying 14-26 new verb groups
that (a) exist in the corpus or can be added, (b) form cohesive
positional clusters above `similarity_threshold=0.13`, and (c) have
a clear RelationType label. Candidate groups not currently covered:

- SPATIAL verbs (`di`, `pada`, `ke`, `dari`) — currently 0 labelled.
- DISCURSIVE verbs (`menurut`, `berdasarkan`, `according to`) —
  currently 0 labelled.
- Negation patterns (`bukan`, `tidak`) — currently 0 labelled (the
  audit notes these are intentionally left to SRC because
  `bukan`/`tidak` are not single-verb predicates).
- Morphological variants of existing labelled verbs (`sebabkan`,
  `akibatkan` passive forms; `butuhkan`, `perlukan` transitive forms).

### 3.5 Verdict on §5.2

| Strategy | Corpus cost | Time cost | Recommendation |
|---|---|---|---|
| A: label existing clusters | **0 lines** | ~17-31 human-hours | **Cheapest by 4 orders of magnitude.** If coverage uplift is the goal, this is the obvious first move. |
| B: expand corpus only | ~10,900-43,000 lines | Days of corpus generation | **Not recommended.** Yield is gated by `EXPECTED_VERB_GROUPS` being fixed. |
| C: combined widen + generate | ~3,400-6,200 lines | 1-2 days corpus gen + ~14-26 verb-group definitions | **Reasonable if coverage uplift is paired with expanding the RelationType taxonomy** (e.g. adding SPATIAL / DISCURSIVE as new RelationTypes). |

This is an **estimation only**. Per the task constraint, no decision
is made here about widening the corpus. The current state's "lean on
(b) by default" stance (per the audit's wording) remains the
production choice.

---

## 4. Constraints honoured

- **No production code modified.** The only Python file added is
  `AGNN/tests/test_audit_4_1_ca1_fallback_misfire.py` — a test
  artifact under `AGNN/tests/`, not under `AGNN/`'s production tree.
  All assertions in that file encode the *current buggy behaviour*
  (the misfire), not a contract for future behaviour; the assertions
  will need to be flipped if/when the misfire is fixed.
- **No fix applied.** The audit's suggested fix for §4.1 (gate the
  override on a PCL-side `_last_classification_was_fallback` flag)
  is documented in §1.8 as one valid remediation, but not
  implemented.
- **No restoration applied.** §4.2's `self-ai/test_training_agent.py`
  broken imports are documented but not fixed; the two remediation
  options (re-point import path vs. archive the test file alongside
  its dependency) are listed but not executed.
- **No corpus widening.** §5.2's coverage uplift is modelled as an
  estimation; no corpus files were modified.

---

## 5. Reproducibility

All findings in this document can be reproduced by running:

```bash
cd <repo-root>

# §4.1 misfire proof (7 tests, ~1s)
python -m pytest AGNN/tests/test_audit_4_1_ca1_fallback_misfire.py -v

# §4.2 git history
git log --all --oneline -- 'self-ai/benchmark'
git show --stat --diff-filter=D --format="%h %s" 56aaad7 -- 'self-ai/benchmark/*'
git ls-tree -r HEAD --name-only | grep 'archive/self-ai-v1/benchmark'

# §5.2 coverage computation
python3 -c "
import json
with open('AGNN/data/cluster_learner_state.json') as f:
    state = json.load(f)
labels = {int(k): v for k, v in state['cluster_labels'].items()}
cid_of = state['cluster_id_of']
total = len(cid_of)
labelled = sum(1 for c in cid_of.values() if c in labels)
print(f'Coverage: {labelled}/{total} = {labelled/total*100:.2f}%')
"

# Regression sanity (existing tests, no production code touched)
python -m pytest AGNN/tests/test_e2e_logical_validity.py \
                 AGNN/tests/test_deductive_reasoning.py \
                 AGNN/tests/test_bootstrap_classifier.py -v
# Expected: 51 passed
```

The investigation test file
(`AGNN/tests/test_audit_4_1_ca1_fallback_misfire.py`) is added to the
repo in this PR. The investigation document you are reading
(`AGNN/docs/dead-code-followup-investigation.md`) is also added in
this PR. No other files are touched.
