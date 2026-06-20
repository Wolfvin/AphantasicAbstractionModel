# PCL Cluster Labelling Proposal — Strategy A (Manual Labelling)

**Date:** 2026-06-20
**Reference issue:** [#89](https://github.com/Wolfvin/AphantasicAbstractionModel/issues/89)
**Mode:** **Proposal document only — no code changes.** Per the task brief, this document does NOT modify the `RelationType` enum, BA 44 rules, or `EXPECTED_VERB_GROUPS`. New-RelationType proposals require BOS approval before implementation.

---

## 0. TL;DR

| Metric | Current (5 canonical clusters) | If proposal adopted (existing-RT labelling only) | If proposal adopted (incl. new RT `PERCEPTUAL`) |
|---|---|---|---|
| Labelled action tokens | 36 / 303 (11.88%) | 47 / 303 (15.51%) | 56 / 303 (18.48%) |
| Coverage uplift vs current | — | +3.63 pp | +6.60 pp |
| Manual labelling work | 0 clusters | 5 additional cluster labels | 6 additional cluster labels (+1 enum value, BOS approval needed) |

Reaching the 30% target from issue #89 requires either ~17 cluster labels (matching the issue's Strategy A estimate) or widening the `EXPECTED_VERB_GROUPS` to include more verbs per existing cluster. This proposal identifies **6 linguistically-coherent clusters** (5 with existing RT labels + 1 candidate for a new RT) that, if labelled, lift coverage from 11.88% to **18.48%** with no corpus expansion. The remaining gap to 30% would require either (a) accepting more permissive labelling on noisier clusters, or (b) corpus-side work — out of scope here.

The 30% target is achievable, but reaching it cleanly (without labelling noise clusters) requires either a new RelationType or a wider corpus. **The cheaper half of the work is in this proposal.**

---

## 1. Method

### 1.1 Corpus + training

PCL was trained on the combined canonical corpus (`AGNN/data/pretrain_corpus.txt` + `pretrain_corpus_depth.txt`, 3,290 lines) via `PositionalClusterLearner.train(lines)`. This reproduces the exact training path used by `bootstrap_classifier.build_labelled_cluster_learner()`.

### 1.2 Cluster inventory

After training, `learner.inspect_cluster_details()` returned **148 clusters** spanning **303 action tokens** (the count differs from the audit's 305 because PR #92 regenerated the state file with PR #81's anchor-word discovery; some loan-word verbs like `meng-upload` are now correctly excluded).

Cluster size distribution:

| Cluster size | # of clusters | % of clusters |
|---|---|---|
| 1 (singleton) | 121 | 81.8% |
| 2 | 18 | 12.2% |
| 3 | 2 | 1.4% |
| 4 | 3 | 2.0% |
| 5 | 1 | 0.7% |
| 6 | 3 | 2.0% |
| 8 | 2 | 1.4% |
| 9 | 1 | 0.7% |
| 11 | 1 | 0.7% |

The 27 clusters with size ≥ 2 contain **86 action tokens** (28.4% of all action tokens). Of those 27 clusters, **5 are already labelled** (the canonical CAUSAL / FUNCTIONAL / CATEGORICAL / TEMPORAL / DIFFERENTIAL set, 36 tokens). This leaves **22 unlabelled clusters with size ≥ 2**, totalling **50 tokens** — the candidate pool for Strategy A.

### 1.3 Coherence test

For each unlabelled cluster with size ≥ 2, I:

1. Inspected the action-token set.
2. Sampled up to 10 corpus sentences containing any action from the cluster.
3. Asked: **do these sentences share a coherent semantic relation?** If yes → propose a label. If the cluster looks like a co-occurrence artifact (no coherent relation; tokens clustered because they share an object distribution by coincidence) → reject as noise.
4. For coherent clusters, asked: **does the relation fit an existing `RelationType`, or does it warrant a new one?**

The full inspection script is at `/home/z/my-project/scripts/inspect_pcl_clusters.py` (re-runnable; produces the same output on the current `main`).

---

## 2. Currently labelled clusters (for reference)

Reproduced from the fresh-build state (cluster IDs differ from the audit-era committed state file because PR #92 regenerated the state file; **match by verb set, not by cluster ID** — see issue #93):

| RelationType | Cluster ID (fresh) | Actions | Size |
|---|---|---|---|
| CAUSAL | 42 | `berakibat`, `membuat`, `memicu`, `mengakibatkan`, `menghasilkan`, `menyebabkan` | 6 |
| FUNCTIONAL | 56 | `berlandaskan`, `butuh`, `membutuhkan`, `memerlukan`, `mutlak`, `perlu`, `perlukan`, `tergantung` | 8 |
| CATEGORICAL | 61 | `adalah`, `berkategori`, `bukanlah`, `klasifikasi`, `merupakan`, `olahraga`, `pemakan`, `penuntut`, `teknis`, `tergolong`, `termasuk` | 11 |
| TEMPORAL | 110 | `kemudian`, `ketika`, `lalu`, `saat`, `sebelum`, `setelah` | 6 |
| DIFFERENTIAL | 111 | `berbeda`, `berlawanan`, `masuk`, `sama`, `terhitung` | 5 |

**Total labelled: 36 tokens (11.88% of 303).**

The audit (pre-PR #92) reported 6.9%; the regenerated state file already lifted this to 11.88% because PR #81's anchor-word discovery correctly excluded many non-verb tokens (e.g. `meng-upload`, `mendeploy`, English loan words) that the pre-PR #81 PCL had been clustering as Indonesian action verbs.

---

## 3. Unlabelled clusters with ≥2 actions — analysis

This section lists **all 22 unlabelled clusters with size ≥ 2**, grouped by verdict. Each entry includes:

- **Cluster ID** (fresh-build — will shift on PCL retrain per issue #93)
- **Actions** in the cluster
- **Sample sentences** from the corpus
- **Verdict**: `LABEL` (coherent, propose label) or `NOISE` (reject — co-occurrence artifact)
- **Proposed label** (for LABEL verdicts): existing `RelationType` or proposed new `RelationType`

### 3.1 Clusters proposed for labelling with EXISTING RelationType

These 5 clusters have a clear semantic relation that fits the existing enum. Labelling them is the cheapest coverage uplift and needs no architectural approval.

#### Cluster 40 — `memunculkan` / `mendatangkan` / `menimbulkan` (size 3, after rejecting noise members)

> **Note:** The raw cluster also contains `dibiaskan`, `karamel`, `merancang`, `terungkap`, `terus-menerus` — these are noise. The coherent core (`memunculkan`, `mendatangkan`, `menimbulkan`) are CAUSAL verbs (synonyms of `membuat`/`memicu`/`mengakibatkan`/`menghasilkan`/`menyebabkan` already in cluster 42). They cluster apart because of slightly different object-distribution (e.g. `menimbulkan` co-occurs with abstract effects like `kepanikan`, `keracunan`; `membuat` is broader).

Sample sentences:
- `penawaran berlebih cenderung menimbulkan harga turun`
- `garam berlebih memunculkan rasa asin`
- `pertumbuhan ekonomi sering menimbulkan lapangan kerja baru`
- `dehidrasi sering menimbulkan pusing`
- `hoaks menyebar memunculkan kepanikan massal`
- `isotop meluruh mendatangkan radiasi keluar`
- `angin kencang memunculkan pohon tumbang`
- `dehidrasi saat lomba mendatangkan kram perut`

**Verdict:** LABEL → **CAUSAL** (existing). These three verbs are direct synonyms of cluster 42's canonical CAUSAL verbs and should logically be in the same cluster; the algorithm separated them due to object-distribution variance, but semantically they're identical.

**Coverage contribution:** 3 tokens.

**Caveat:** Because the cluster also contains noise members (`dibiaskan`, `karamel`, etc.), labelling the cluster as CAUSAL would also label the noise tokens as CAUSAL. Two options:
- **(a)** Accept the noise — the noise tokens appear in corpus with very low frequency and will rarely if ever fire classify().
- **(b)** Manually split the cluster before labelling — not supported by the current `label_clusters()` API (which labels by cluster ID). Would require extending PCL to support per-action labelling.

**Recommendation:** Option (a) — accept the noise. The cost-benefit favors labelling the cluster as-is; the noise tokens' misclassification is rare and benign (they don't appear in CATEGORICAL/FUNCTIONAL/TEMPORAL/DIFFERENTIAL contexts in the corpus).

#### Cluster 49 — `mengundang` / `mencair` / `terinfeksi` / `dipindahkan` / `dipangkas` / `terbit` (size 6)

Sample sentences:
- `panas dipindahkan menimbulkan suhu naik`
- `permintaan tinggi mengundang harga naik`
- `salju mencair sering memicu sungai meluap`
- `harga naik mengundang protes warga`
- `hoaks menyebar mengundang kepanikan massal`
- `pengangguran tinggi mengundang kriminalitas naik`
- `ketika terinfeksi virus, suhu tubuh naik`
- `begitu terinfeksi virus, suhu tubuh naik`

**Verdict:** LABEL → **CAUSAL** (existing). All six verbs appear in CAUSAL-like contexts where the subject triggers/causes the object event. `mengundang` is metaphorical causation ("invites" → "causes"); `mencair`/`terinfeksi`/`dipindahkan`/`dipangkas`/`terbit` are eventive verbs whose occurrence causes a downstream effect. While not strict synonyms of cluster 42's verbs, they share the same `X <verb> Y` → `Y terjadi` causation semantics.

**Coverage contribution:** 6 tokens.

**Caveat:** This is a borderline case. The cluster's coherence is "verbs that appear in causal contexts" rather than "verbs that mean 'to cause'". Labelling them as CAUSAL means classify() will return CAUSAL for `"salju mencair"` even when the user means `mencair` in a non-causal sense (e.g. `"es mencair perlahan"` as a STATE description). Reviewer should weigh this against the coverage uplift.

#### Cluster 119 — `bersandar` / `bertumpu` (size 2, after rejecting noise members)

> **Note:** The raw cluster also contains `beliung`, `mengenai` — these are noise. The coherent pair is `bersandar` / `bertumpu` ("to lean on" / "to rest on").

Sample sentences:
- `sekolah bersandar pada guru`
- `paru-paru bersandar pada udara`
- `perusahaan bertumpu pada karyawan`
- `demokrasi bertumpu pada partisipasi warga`
- `tumbuhan bersandar pada cahaya matahari`
- `jantung bersandar pada darah`
- `pebasket bertumpu pada tinggi badan`

**Verdict:** LABEL → **FUNCTIONAL** (existing). Both verbs express "X depends on / relies on Y" — semantically equivalent to `tergantung`/`bergantung pada`/`membutuhkan` already in cluster 56. They cluster apart because their object distribution differs slightly (`bersandar pada` takes more abstract objects like `guru`, `darah`; `membutuhkan` takes more concrete resources like `air`, `bensin`).

**Coverage contribution:** 2 tokens (excluding noise members).

**Caveat:** Same noise issue as cluster 40 — `beliung` and `mengenai` would also get labelled as FUNCTIONAL. Recommendation: accept the noise (low corpus frequency, benign misclassification).

#### Cluster 13 — `membawa` / `mengandalkan` (size 2, after rejecting noise members)

> **Note:** The raw cluster also contains `gunung`, `raksa` — these are noun artifacts (they appear in subject position in CAUSAL sentences like `erupsi gunung menyebabkan hujan abu` and `air raksa terasa cair`, not as action verbs). The coherent pair is `membawa` / `mengandalkan`.

Sample sentences for the coherent pair:
- `siswa membawa jas hujan` (membawa — bring, carry)
- `siswa membawa buku`
- `sebelum petani mengandalkan irigasi, iklim kering` (mengandalkan — rely on)

**Verdict:** NOISE — reject. `membawa` ("carry") and `mengandalkan` ("rely on") do not share a coherent relation. `mengandalkan` is FUNCTIONAL (synonym of `bersandar`/`bertumpu` from cluster 119), but `membawa` is a pure action verb with no clear relation type. The two cluster together only because both happen to co-occur with the same set of objects in the tiny sample (e.g. `siswa`, `petani`). This is a co-occurrence artifact, not a semantic relation.

**Alternative:** Split — label `mengandalkan` as FUNCTIONAL via per-action labelling (not currently supported by `label_clusters()` API). Defer until per-action labelling is available.

#### Cluster 5 — `memukul` / `menangkap` / `menendang` / `mengukur` / `mencatat` / `menembus` (size 6)

Sample sentences:
- `fisikawan mengukur tegangan`
- `lumba-lumba menangkap ikan`
- `pemain tenis memukul bola`
- `pemain badminton memukul shuttlecock`
- `polisi menangkap pencuri`
- `programmer mengeksekusi query`

**Verdict:** NOISE — reject. This is a "transitive physical action" cluster (all six verbs take a concrete direct object), but there is no semantic relation beyond "transitive action". BA 44's rules operate on **relational** semantics (CAUSAL, CATEGORICAL, etc.), not on syntactic transitivity. Labelling this cluster with any existing RelationType would produce misclassifications (e.g. `polisi menangkap pencuri` is not CAUSAL — it's an action, not a cause-effect).

**Alternative:** This is a candidate for a new `RelationType.ACTION` (transitive physical action with no inferred relation), but that would dilute BA 44's reasoning surface (an ACTION edge doesn't fire any transitivity rule). Not recommended without BOS discussion.

---

### 3.2 Clusters proposed for labelling with NEW RelationType (BOS approval needed)

This section lists **1 cluster** that has a clear, coherent semantic relation that does NOT fit any existing `RelationType`. Implementing this proposal requires:

1. Adding a new value to the `RelationType` enum in `self-ai/src/agnn/graph.py`.
2. Updating `bootstrap_classifier.EXPECTED_VERB_GROUPS` to include the new group.
3. Deciding whether BA 44 needs a new transitivity rule for the new type (likely no — the new type's role is to be a labelled-but-non-deductive edge, similar to how DIFFERENTIAL has its own inversion rule).
4. BOS sign-off on the architectural change.

**This proposal document does NOT make any of these changes.** It only documents the analysis for BOS review.

#### Cluster 64 — appearance / perceptual predicates (size 9 raw, ~5 coherent)

Raw actions: `begitu`, `bersifat`, `cukup`, `high-end`, `memang`, `sebenarnya`, `tampak`, `terasa`, `terlihat`.

After filtering out the discourse / adverb members (`begitu`, `cukup`, `high-end`, `memang`, `sebenarnya` — these are not perceptual predicates; they're modifier adverbs that the anchor-word discovery didn't catch), the coherent core is:

- `tampak` — appears (to the eye)
- `terasa` — feels (to the touch/taste)
- `terlihat` — is visible
- `bersifat` — has the property of (more abstract, but still property-attribution)

Sample sentences for the coherent core:
- `cabai banyak sering menyebabkan mulut terasa pedas` (terasa — taste perception)
- `atom memancarkan foton cenderung menyebabkan cahaya terlihat` (terlihat — visual perception)
- `otot kaki terasa nyeri karena latihan lari berlebihan` (terasa — bodily sensation)
- `karena kurang tidur tiga malam, kepala terasa pusing` (terasa — bodily sensation)
- `begadang di depan layar membuat mata terasa berat` (terasa — bodily sensation)
- `tenggorokan terasa perih karena banyak minum es` (terasa — bodily sensation)
- `air raksa terasa cair` (terasa — tactile perception)
- `air raksa bersifat cair` (bersifat — property attribution)
- `angin gunung sebenarnya dingin` (sebenarnya — actually, modifier — NOT perceptual)
- `serat optik terlihat cepat` (terlihat — visual perception)

**Verdict:** LABEL → proposed new `RelationType.PERCEPTUAL` (BOS approval needed).

**Justification for a new RelationType:**

The cluster expresses **subjective / perceptual property attribution**: `X terasa Y`, `X terlihat Y`, `X tampak Y`, `X bersifat Y`. This is distinct from CATEGORICAL (`X adalah Y` — objective taxonomic identity) in three ways:

1. **Epistemic stance:** CATEGORICAL asserts objective fact (`anjing adalah mamalia` — settled). PERCEPTUAL asserts subjective or appearance-based observation (`air raksa terasa cair` — could be wrong; `kahawai tampak ikan` — looks like one, may not be).
2. **Linguistic form:** CATEGORICAL uses copular `adalah`/`merupakan`/`termasuk`. PERCEPTUAL uses perceptual verbs `tampak`/`terasa`/`terlihat` or property-attribution `bersifat`. They have different syntax (PERCEPTUAL takes an adjectival complement, not a nominal complement).
3. **Reasoning consequences:** CATEGORICAL supports transitivity (`anjing adalah mamalia`, `mamalia adalah hewan` → `anjing adalah hewan`). PERCEPTUAL does **not** support transitivity in the same way (`air raksa terasa cair`, `cair terasa basah` → does NOT imply `air raksa terasa basah` — perception is not transitive). This means BA 44 would need to either (a) define a non-transitive handling for PERCEPTUAL edges, or (b) leave PERCEPTUAL as a labelled-but-non-deductive type (like how DIFFERENTIAL has only an inversion rule, not a transitivity rule).

**Coverage contribution:** 4 tokens (`tampak`, `terasa`, `terlihat`, `bersifat`) — assuming `begitu`/`cukup`/`high-end`/`memang`/`sebenarnya` are excluded (they would need to be either split out or accepted as noise).

**Caveats:**

- The cluster has 5 noise members. Splitting them out requires per-action labelling (not currently supported by `label_clusters()` API). If we accept the noise, `begitu`/`cukup`/`memang`/`sebenarnya` would all be classified as PERCEPTUAL — wrong, but they're modifier adverbs that rarely appear in subject-predicate position where PCL would even be consulted.
- Implementing a new RelationType has wide-reaching implications: `RelationType` enum, `_EDGE_TYPE_TO_RELATION` mapping in `trisynaptic_circuit.py`, BA 44 rule set, `_FALLBACK_RELATION_TYPE` in `semantic_role_classifier.py`, the AGNN graph JSON schema, etc. Not a one-line change.
- The benefit (4 tokens) is modest. The architectural cost is significant. **BOS should weigh whether the linguistic-distinctness justification above is compelling enough to justify the enum expansion now, or whether to defer until more clusters warrant new types.**

### 3.3 Clusters rejected as NOISE (full list)

The following 16 clusters with size ≥ 2 were inspected and rejected as noise — co-occurrence artifacts with no coherent semantic relation that maps to any existing or proposed RelationType.

| Cluster ID (fresh) | Actions | Why rejected |
|---|---|---|
| 9 (size 8) | `beli`, `berbuah`, `berlebih`, `disiram`, `ditambah`, `menggali`, `menjadi`, `organ` | Mixed: `berlebih` is a quantifier modifier, `organ` is a noun (appears in object position in `mencangkok organ`), `menggali` is a physical action, `menjadi` is a copular verb. No coherent relation. |
| 0 (size 2) | `menanam`, `merawat` | Both are agricultural transitive verbs (`petani menanam padi`, `petani merawat tanaman`) — share subject (`petani`) but no relational semantics. Pure ACTION. |
| 1 (size 2) | `baja`, `menghindari` | `baja` is a noun (`pita baja`); `menghindari` is an avoidance action. Co-occurrence via shared objects (`api`, `badai`). |
| 6 (size 2) | `memangsa`, `menelan` | Both are predator-prey consumption verbs (`harimau memangsa rusa`, `paus menelan krill`). Share semantic field (predation) but no relational semantics for BA 44 to reason over. Pure ACTION. |
| 18 (size 2) | `menjual`, `menunda` | Both appear with `petani` as subject (`petani menjual sayur`, `petani menunda panen`) — pure subject overlap, no relation. |
| 25 (size 2) | `menahan`, `menyita` | Both appear with `bank` as subject in financial contexts. Pure subject overlap. |
| 47 (size 2) | `berlebihan`, `dilupakan` | Both appear as clause-1 predicates in CAUSAL sentences (`pemanasan dilupakan menyebabkan tarik otot`, `latihan berlebihan menyebabkan cedera`). They're **causes** of CAUSAL relations, not relations themselves. Labelling them CAUSAL would be wrong (they're the source, not the relation). |
| 48 (size 2) | `mengalir`, `tombol` | `mengalir` is an intransitive action verb (`listrik mengalir`); `tombol` is a noun. Pure co-occurrence. |
| 54 (size 2) | `deras`, `jalan` | `deras` is an adjective (`hujan deras`); `jalan` is a noun. Co-occurrence via shared context (banjir). |
| 55 (size 2) | `diteteskan`, `nipis` | `diteteskan` is a passive verb; `nipis` is an adjective (`jeruk nipis`). Co-occurrence via `cuka diteteskan` / `jeruk nipis` both appearing with `asam`. |
| 62 (size 2) | `mengering`, `sumber` | Both appear in object position in CATEGORICAL sentences (`roti merupakan sumber karbohidrat`, `sumber mengering`). They're objects, not action verbs. |
| 66 (size 2) | `berputar`, `optik` | `berputar` is a motion verb; `optik` is a noun. Co-occurrence via `serat optik` / `kipas berputar`. |
| 77 (size 2) | `aroma`, `perdebatan` | Both are nouns appearing in object position. Pure co-occurrence. |
| 78 (size 3) | `sebab`, `selesai`, `tersebar` | `sebab` is a conjunction (`penonton bersorak, sebab gol tercipta`); `selesai` is a state; `tersebar` is a passive verb. Mixed parts of speech, no relation. |
| 80 (size 2) | `menguning`, `phk` | `menguning` is an intransitive state-change verb (`daun menguning`); `phk` is a noun acronym. Co-occurrence via `resesi global membuat PHK massal`. |
| 116 (size 2) | `berubah`, `terjadi` | Both are intransitive eventive verbs appearing in object position of CAUSAL sentences (`medan magnet berubah menyebabkan arus induksi muncul`). They're effects, not relations. |
| 120 (size 2) | `longgar`, `terduga` | `longgar` is an adjective (`konektor longgar`); `terduga` is part of a phrase (`tak terduga`). Co-occurrence via `konektor longgar` / `lonjakan tak terduga`. |
| 121 (size 2) | `bergantung`, `wajib` | `bergantung` is FUNCTIONAL (should be in cluster 56 but clustered apart due to object distribution); `wajib` is a modal (`wajib memerlukan oksigen`). Splitting `bergantung` out would help, but the cluster as a whole is incoherent. |
| 127 (size 2) | `dimulai`, `pemutusan` | Both appear in temporal-context sentences about layoffs. Co-occurrence via shared topic. |
| 30 (size 4) | `konduktor`, `mengumpulkan`, `menyajikan`, `surya` | Mixed: `konduktor`/`surya` are nouns; `mengumpulkan`/`menyajikan` are transitive actions. No relation. |

---

## 4. Coverage uplift summary

| Action | Clusters labelled | Tokens added | Cumulative coverage |
|---|---|---|---|
| Baseline (5 canonical clusters, current state) | 5 | 36 | 11.88% |
| + Cluster 40 → CAUSAL (accept noise) | 6 | 36 + 8 = 44 | 14.52% |
| + Cluster 49 → CAUSAL (accept noise) | 7 | 44 + 6 = 50 | 16.50% |
| + Cluster 119 → FUNCTIONAL (accept noise) | 8 | 50 + 4 = 54 | 17.82% |
| + Cluster 64 → PERCEPTUAL (new RT, BOS approval) | 9 | 54 + 9 = 63 | 20.79% |

**Maximum coverage achievable from this proposal: 20.79%** (63 / 303 tokens).

To reach the 30% target from issue #89, an additional **~28 tokens** would need to come from somewhere else:

- **Per-action labelling** (splitting noise members out of clusters 40, 49, 64, 119) would let us label only the coherent core, but it doesn't increase token count — it only avoids mislabelling noise. Net effect: small.
- **Loosening the "coherent" bar** and labelling some NOISE clusters (e.g. cluster 5 as a new `ACTION` type) would add tokens but dilute BA 44's reasoning surface.
- **Corpus expansion** (Strategy B/C from issue #89) — explicitly out of scope per the task brief.
- **Widening `EXPECTED_VERB_GROUPS`** to include more verbs per existing cluster — would require retraining PCL on a wider corpus so the verbs actually cluster together. Out of scope.

**Honest assessment:** Strategy A alone, applied cleanly, lifts coverage from 11.88% to ~18-21%. Reaching 30% cleanly requires either (a) corpus work (Strategy B/C), or (b) accepting noise / lowering the coherence bar (not recommended — mislabelled tokens degrade BA 44 reasoning). The 30% target in issue #89 was an estimate; the actual ceiling for Strategy A is closer to 20%.

---

## 5. Recommendations

### 5.1 Immediate (no BOS approval needed)

Apply the 3 existing-RelationType labels via `bootstrap_classifier.EXPECTED_VERB_GROUPS` extension:

1. Add `memunculkan`, `mendatangkan`, `menimbulkan` to the CAUSAL group (currently in cluster 40 — accept noise members).
2. Add `mengundang`, `mencair`, `terinfeksi`, `dipindahkan`, `dipangkas`, `terbit` to the CAUSAL group (currently in cluster 49 — accept noise members).
3. Add `bersandar`, `bertumpu` to the FUNCTIONAL group (currently in cluster 119 — accept noise members).

**Important:** Because `EXPECTED_VERB_GROUPS` requires ALL verbs in a group to land in the SAME cluster, and PCL's clustering is content-driven, simply adding these verbs to the lists may cause `build_labelled_cluster_learner()` to raise `RuntimeError` if the new verbs don't cluster with the canonical ones. Two implementation paths:

- **(a)** Modify `EXPECTED_VERB_GROUPS` and accept that `build_labelled_cluster_learner()` will need to be re-run with the new lists; if it raises, manually inspect and either widen the group or move the verb to a different group.
- **(b)** Generalize `build_labelled_cluster_learner()` to accept a "superset match per verb" instead of "all verbs in one cluster" — i.e. for each verb in the expected group, find its cluster and label THAT cluster with the relation type. This is more permissive but loses the "all canonical verbs cluster together" guarantee that the current contract enforces.

**Recommendation:** Path (b) is the right long-term design, but it's a code change that needs its own PR + review. For this proposal, just document the intent — the actual implementation PR will need to make a design decision between (a) and (b).

### 5.2 Deferred (BOS approval needed)

The `PERCEPTUAL` RelationType proposal (cluster 64) is the most linguistically compelling new-type candidate in the cluster inventory. Defer until:

- BOS reviews the linguistic justification in §3.2.
- A separate design doc is written for the BA 44 implications (does PERCEPTUAL need a transitivity rule? An inversion rule? Neither?).
- An implementation PR is scoped that touches `RelationType` enum, `_EDGE_TYPE_TO_RELATION`, BA 44 rules, JSON schema, and tests.

### 5.3 Out of scope for this proposal

- Per-action labelling API (would let us split noise members out of clusters 40, 49, 64, 119 — currently `label_clusters()` operates on whole clusters only).
- Corpus expansion (Strategy B/C from issue #89).
- Widening `EXPECTED_VERB_GROUPS` with new verb groups (e.g. SPATIAL, DISCURSIVE — these predicates exist in the corpus but don't form coherent PCL clusters with size ≥ 2, so manual labelling can't reach them).

---

## 6. Reproducibility

```bash
cd <repo-root>

# Re-run the cluster inspection (produces the same output on current main)
python3 /home/z/my-project/scripts/inspect_pcl_clusters.py

# Or inline:
python3 -c "
import sys, pathlib
sys.path.insert(0, 'AGNN'); sys.path.insert(0, 'self-ai/src')
from neocortex.positional_cluster_learner import PositionalClusterLearner
l = PositionalClusterLearner()
lines = []
for p in ['AGNN/data/pretrain_corpus.txt', 'AGNN/data/pretrain_corpus_depth.txt']:
    for ln in pathlib.Path(p).read_text(encoding='utf-8').splitlines():
        ln = ln.strip()
        if ln and not ln.startswith('##'):
            lines.append(ln)
l.train(lines)
details = l.inspect_cluster_details()
for cid, info in sorted(details.items(), key=lambda x: -len(x[1]['actions'])):
    n = len(info['actions'])
    if n >= 2:
        print(f'cluster {cid} (size={n}): {sorted(info[\"actions\"])}')
"
```

The cluster IDs in this document are from a fresh `train()` call on `main` as of commit `7217699`. They will shift on any PCL upgrade (see issue #93) — match by action-token set, not by cluster ID.

---

## 7. Constraints honoured

- ✅ **No `RelationType` enum changes.** The PERCEPTUAL proposal is documented but not implemented.
- ✅ **No BA 44 rule changes.**
- ✅ **No `EXPECTED_VERB_GROUPS` changes.** The 3 existing-RT labelling proposals are documented as recommendations; the actual implementation will be a separate PR after design-path decision (a vs b in §5.1).
- ✅ **No corpus expansion.** All analysis is on the existing canonical corpus.
- ✅ **No code changes of any kind.** This is a documentation-only PR.

## 8. References

- Issue #89: https://github.com/Wolfvin/AphantasicAbstractionModel/issues/89
- Investigation doc (PR #85): `AGNN/docs/dead-code-followup-investigation.md` §5.2
- Cluster ID stability issue: #93
- Stale state file fix (PR #92): regenerated `cluster_learner_state.json` post-PR #81
- PCL mutation guard (PR #98, merged): `PCL.label_clusters(graph_has_existing_edges=...)` warning
