# Riset: Applicability Intuitive AI Academy Curriculum ke AGNN

> **Worker task**: Pelajari kurikulum di https://www.intuitiveai.academy/ dan identifikasi
> konsep/teknik yang punya prinsip serupa dengan masalah yang AGNN selesaikan.
> Laporkan secara jujur — kalau tidak ada yang genuinely applicable, jangan dipaksakan.

---

## 0. Metadata Riset

| Item | Nilai |
|---|---|
| Tanggal riset | 2026-06-20 |
| Sumber | https://www.intuitiveai.academy/ |
| Akses | Publik (homepage + syllabus outline); login wall untuk chapter content |
| Metode | Fetch homepage + syllabus via `z-ai page_reader`, eksplorasi via `agent-browser`, web search untuk validasi |
| Constraint | Tidak boleh rekomendasi ganti backbone Qwen3-0.6B |

### Catatan akses

Situs Intuitive AI Academy adalah kursus berbayar ($12/bulan atau $120/tahun). Setiap URL
chapter (`/en/llm-fundamentals/tokenization`, `/learn/...`, `/chapter/...`, dll.) me-redirect
ke `/sign-in?redirect_url=...`. Web search engine juga tidak meng-index isi chapter (hanya
homepage, sign-in, dan sign-up yang ter-index).

**Yang berhasil diakses**: outline syllabus publik dari homepage (24+ chapter titles across
5 sections). Outline ini cukup untuk evaluasi prinsip — chapter titles adalah konsep LLM
standar (Tokenization, Attention, RLHF, MoE, Distillation, dll.) yang prinsipnya well-known,
jadi evaluasi bisa dilakukan dari judul + prinsip yang dijelaskan di tiap chapter.

**Yang TIDAK berhasil diakses**: penjelasan detail, diagram, visualisasi di tiap chapter.
Konsekuensinya: riset ini menilai prinsip konsep (bukan detail implementasi kurikulum).
Kalau ada teknik spesifik di dalam chapter yang tidak terlihat dari judul, riset ini tidak
menangkapnya.

---

## 1. Ringkasan Konsep Kunci dari Kurikulum yang Relevan

Kurikulum Intuitive AI Academy dibagi 5 section dengan total 24+ chapter:

### LLM Fundamentals / Architecture (9 topics)
- Introduction
- Tokenization
- The Embedding Layer
- Positional Encoding
- Attention
- Layers of Understanding
- Learning to Predict
- Instruction Tuning and RLHF
- GPT-2 from Scratch

### Pre-Training (8 topics)
- Overview
- Training Objectives and Architectural Details
- Scaling Laws and Optimization
- Training Data Engineering
- Training Infrastructure and Systems
- Advanced Pretraining Objectives
- Evaluation During Pretraining
- Case Study - LLaMA 3

### Post-Training (5 topics)
- Overview
- Supervised Fine-Tuning
- Preference Optimization
- Tools and Safety Tuning
- Case Study on Tulu 3

### LLM Advanced
- Distillation
- LoRA
- Mixture of Experts (MoE)
- Optimizers

### Reinforcement Learning
- RL Fundamentals
- RLHF

Dari 24+ chapter tersebut, **4 konsep punya prinsip yang cukup relevan** untuk dievaluasi
terhadap arsitektur AGNN. Sisanya tidak applicable (lihat Section 4 untuk detailnya).

Konsep yang relevan:

1. **Positional Encoding** — posisi token adalah sinyal struktural, bukan hanya identitas token
2. **Learning to Predict (self-supervised next-token objective)** — belajar struktur dari raw data tanpa label
3. **Advanced Pretraining Objectives** (secara spesifik: contrastive learning & masked prediction) — sinyal training yang lebih kaya dari plain co-occurrence counting
4. **Layers of Understanding** — representasi bertingkat: deeper layer = more abstract meaning

---

## 2. Evaluasi Applicability per Konsep

Untuk tiap konsep yang dinilai relevan: prinsipnya, kenapa applicable ke AGNN, dan scope
perubahan kalau diadopsi.

### 2.1 Positional Encoding → AGNN PositionalClusterLearner

**Prinsip kurikulum**: Posisi token dalam sequence adalah sinyal struktural. Transformer
menambahkan positional encoding (sinusoidal, learned, RoPE, ALiBi) ke embedding karena
self-attention sendiri permutation-invariant — tanpa informasi posisi, model tidak bisa
membedakan "ayam makan pakan" dari "pakan makan ayam". Posisi bisa *absolut* (token ke-N)
atau *relatif* (jarak antara token A dan token B).

**Prinsip AGNN saat ini**: `PositionalClusterLearner` (lihat `AGNN/neocortex/positional_cluster_learner.py`)
sudah memakai posisi sebagai sinyal struktural utama — ini adalah core design choice. Position
buckets: `0` = agent slot, `1` = action slot, `2` = object slot (3-token), `-1` = object slot
(>3-token). Tapi posisi yang dipakai adalah **absolut** (indeks 0/1/2/-1), bukan relatif.

**Kenapa applicable**: Konsep "relative position" dari kurikulum cocok dengan gap yang
AGNN punya saat ini. Ada beberapa case di AGNN di mana posisi relatif lebih informatif
dari posisi absolut:

- **Connector detection**: AGNN punya `action_connector_signature` (boolean: apakah action
  ini punya connector seperti "dari", "dengan", "sebagai"). Saat ini deteksi connector
  dilakukan via token matching (apakah token setelah action ada di `connector_tokens` list).
  Positional info relatif ("connector muncul tepat 1 token setelah action") lebih robust
  dari absolut ("connector ada di indeks 2") karena indeks bisa shift kalau subject
  multi-word.

- **Negation scope**: AGNN cek negation via `_has_negation_before(subject)` — apakah
  token terakhir subject adalah negation token. Ini sudah relatif ("negation tepat sebelum
  predicate"). Tapi untuk negation yang tidak adjacent (misal "tidak pernah menyebabkan"
  — ada "pernah" di antara "tidak" dan "menyebabkan"), positional info relatif bisa
  catch case yang sekarang miss.

- **Polysemy resolution**: Saat ini `positional_freq[token]` track *bucket* (0/1/2/-1).
  Relative position (seberapa jauh token dari action) bisa jadi sinyal tambahan untuk
  bedakan "ayam" sebagai subject (relatif: -2 dari action) vs "ayam" sebagai object
  (relatif: +1 dari action).

**Scope perubahan**: **KECIL**

- Tambah method `_relative_position(tokens, anchor_idx, target_idx) -> int` di
  `PositionalClusterLearner` — helper pure function.
- Tambah field `relative_position_freq` di state (mirip `positional_freq` tapi key-nya
  relative offset dari action, bukan absolut bucket).
- Update `_extract_action_object` untuk juga bump `relative_position_freq` untuk token
  connector dan negation (sekarang hanya `positional_freq` yang di-bump).
- Tidak mengubah API publik (`train()`, `classify()`, `spo()` signature tetap sama).
- Cluster algorithm tidak berubah — relative position jadi sinyal tambahan untuk
  `action_connector_signature`, bukan untuk cluster similarity.

**Risiko**: Bisa over-engineering. Saat ini connector detection via token matching sudah
cukup untuk corpus AGNN. Relative position baru worth-it kalau corpus bertambah kompleks
(connector multi-token, negation non-adjacent).

**Verdict**: **Genuinely worth considering** — prinsip "relative position" dari kurikulum
mengidentifikasi gap real di AGNN, dan scope perubahan kecil. Tapi tidak urgent; baru
worth-it kalau corpus bertambah kompleks.

---

### 2.2 Learning to Predict (self-supervised next-token objective) → AGNN PositionalClusterLearner

**Prinsip kurikulum**: Model belajar struktur bahasa dengan memprediksi token berikutnya
dari konteks sebelumnya. Tidak ada label manusia — sinyal training adalah kemampuan model
untuk memprediksi dengan benar. Ini adalah self-supervised learning: data sendiri (sequence
token) yang jadi ground truth.

**Prinsip AGNN saat ini**: `PositionalClusterLearner` juga self-supervised — tidak ada label
manusia, hanya positional co-occurrence counts. Tapi sinyal trainingnya berbeda: AGNN
menghitung **frekuensi co-occurrence** (action "menyebabkan" muncul dengan object "panas"
sebanyak 5 kali), bukan **predictive error**. Clustering berdasarkan similarity distribusi
object, bukan berdasarkan kemampuan memprediksi object dari action.

**Kenapa applicable**: Prinsip "predictive signal" dari kurikulum menyoroti perbedaan
fundamental antara approach AGNN (counting) vs approach transformer (prediction). Keduanya
self-supervised, tapi sinyalnya berbeda:

- **Counting** (AGNN saat ini): "berapa kali action A muncul dengan object B" → cluster
  action yang punya distribusi object mirip.
- **Prediction** (transformer): "seberapa baik action A memprediksi object B" → cluster
  action yang punya *predictive pattern* mirip.

Predictive signal lebih kaya dari counting karena dia implicitly menangkap **distribusi
probabilitas**, bukan hanya **frekuensi absolut**. Dua action yang sama-sama muncul 5 kali
dengan object "panas" punya count yang sama, tapi salah satunya mungkin lebih predictive
(munculnya action itu menjamin object "panas") vs yang lain (object "panas" hanya satu
dari banyak possibility).

**Scope perubahan**: **SEDANG**

- Tambah training mode baru di `PositionalClusterLearner`: `train_predictive(corpus_lines)`.
  Mirip `train()` tapi setelah building `action_object_freq`, hitung conditional probability
  `P(object | action)` untuk tiap (action, object) pair.
- Tambah metric similarity baru: `predictive_jaccard` — weighted Jaccard di mana weight-nya
  bukan count mentah, tapi `P(object | action)` (conditional probability).
- Cluster ulang pakai metric baru ini, bandingkan dengan cluster existing (lihat apakah
  cluster boundaries shift).
- Tidak menggantikan counting — keduanya bisa coexist. Counting untuk `action_object_freq`
  (raw stat), predictive probability untuk cluster similarity.

**Risiko**: Complexity bertambah. Untuk corpus AGNN yang sekarang (3290 kalimat, 5 cluster
clean), counting + weighted Jaccard sudah cukup. Predictive signal baru worth-it kalau
corpus grow sampai puluhan ribu kalimat dan cluster count bertambah sehingga boundary
decision jadi lebih ambiguous.

**Verdict**: **Genuinely worth considering untuk future scaling** — prinsip predictive
signal dari kurikulum menambah opsi metric yang lebih kaya dari counting. Tapi untuk skala
corpus sekarang, tidak memberikan peningkatan praktis. Note sebagai future work kalau
corpus AGNN grow significantly.

---

### 2.3 Advanced Pretraining Objectives (Contrastive Learning & Masked Prediction) → AGNN clustering

**Prinsip kurikulum**: Beyond next-token prediction, ada objective training lain yang
memberi sinyal berbeda:

- **Masked prediction (BERT-style)**: Mask sebagian token, prediksi dari konteks sekitarnya.
  Sinyalnya: token yang bisa diprediksi dari konteks → token tersebut terikat konteks
  (positional/structural role).
- **Contrastive learning**: Bangun pasangan positif (mirip) dan negatif (tidak mirip),
  belajar representation yang meminimalkan jarak positif dan memaksimalkan jarak negatif.
  Sinyalnya: similarity adalah property yang dipelajari dari data, bukan di-hardcode
  (cosine, Jaccard, dll).

**Prinsip AGNN saat ini**: `PositionalClusterLearner` pakai **weighted Jaccard** pada
object count maps sebagai similarity metric. Threshold 0.13. Metric ini di-hardcode
(dipilih manual setelah tuning). Tidak ada mekanisme untuk belajar metric dari data.

**Kenapa applicable**: Konsep "learned similarity metric" dari contrastive learning
menjawab limitasi mendasar dari weighted Jaccard. Problem saat ini: weighted Jaccard
harus dituning manual (0.13 dipilih karena merge adalah+merupakan; kalau corpus berubah,
threshold mungkin perlu dituning ulang). Contrastive learning bisa otomatis belajar
threshold + metric shape dari data.

Implementasi konkret yang masuk akal untuk AGNN:

- **Contrastive pre-training phase**: Setelah `train()` existing, jalankan contrastive
  phase untuk belajar similarity metric. Positive pairs: action yang sama-sama sering
  muncul dengan object yang sama (sudah ada di `action_object_freq`). Negative pairs:
  action yang object set-nya disjoint.
- **Learned similarity function**: Hasil contrastive phase adalah similarity function
  `sim(action_a, action_b)` yang bisa dipakai sebagai pengganti weighted Jaccard.
- **Masked prediction variant**: Mask action token di sebagian kalimat corpus, prediksi
  action dari (subject, object) context. Action yang punya distribusi prediksi mirip
  → cluster sama. Ini lebih kaya dari co-occurrence counting karena menangkap
  *functional similarity* (bisa saling menggantikan dalam konteks yang sama).

**Scope perubahan**: **BESAR**

- Tambah module baru `AGNN/neocortex/contrastive_metric_learner.py` — belajar similarity
  function dari `action_object_freq` existing.
- Update `PositionalClusterLearner._cluster_actions` untuk support pluggable similarity
  function (saat ini hardcoded ke `_weighted_jaccard`).
- Tambah training mode baru: `train_with_contrastive(corpus_lines)` yang hybrid —
  counting phase + contrastive phase.
- Update `bootstrap_classifier.py` untuk support both modes.
- State file format berubah (tambah `learned_metric_weights` field).
- Test suite baru untuk contrastive phase.

**Risiko**:
- Complexity besar. AGNN sekarang adalah pure Python + numpy, no torch. Contrastive
  learning biasanya butuh gradient descent →要么 add torch dependency (broke design
  constraint), atau implementasi sederhana dengan numpy (mungkin terlalu lambat).
- Benefit marginal untuk corpus skala sekarang. 5 cluster clean sudah didapat dari
  weighted Jaccard. Contrastive learning baru worth-it kalau AGNN mau scale ke puluhan
  RelationType dan puluhan ribu kalimat corpus.
- Trade-off: zero-bias principle AGNN bisa ter-owner kalau contrastive learning
  mulai inject inductive bias dari architecture-nya.

**Verdict**: **Genuinely worth considering untuk long-term evolution** — prinsip learned
similarity dari contrastive learning adalah generalization yang lebih principled dari
hardcoded weighted Jaccard. Tapi scope besar dan benefit marginal untuk skala sekarang.
Note sebagai long-term research direction, bukan immediate adoption.

---

### 2.4 Layers of Understanding → AGNN Aphantasic 3-Layer Representation

**Prinsip kurikulum**: Di transformer, layer yang berbeda menangkap level abstraksi yang
berbeda. Layer awal menangkap surface feature (token, posisi). Layer tengah menangkap
syntactic pattern (subject-verb-object). Layer akhir menangkap semantic meaning dan
task-specific reasoning. Ini emergent property — tidak di-design explicit, muncul dari
training.

**Prinsip AGNN saat ini**: AGNN punya aphantasic node representation 3 layer yang
**explicitly designed** (bukan emergent):

- **Layer 1: Surface text** — episome.text (raw correction string)
- **Layer 2: Amodal definition** — di-generate lazy oleh `DefinitionExtractor` (Qwen3-0.6B
  articulate definisi amodal dari node)
- **Layer 3: Causal anchors** — tuple `(relation_type, target)` yang di-extract oleh
  `CausalAnchorBuilder` dari SPO parse correction

Lihat `AGNN/neocortex/aphantasic_chain_formatter.py` dan `AGNN/neocortex/definition_extractor.py`
untuk implementasinya.

**Kenapa applicable**: Konsep "layers of abstraction" dari kurikulum **memvalidasi** design
3-layer AGNN. Prinsipnya sama: representasi pengetahuan yang lebih kaya dari sekadar
surface form. Tapi AGNN lebih structured (3 layer explicit) vs transformer (emergent
layer behavior).

Yang menarik: kurikulum menyebut "deeper layers capture more abstract meaning". Di AGNN,
Layer 3 (causal anchors) sudah lebih abstract dari Layer 1 (surface text). Tapi apakah
AGNN butuh Layer 4 yang lebih abstract lagi?

Kandidat Layer 4 yang mungkin:
- **Causal chain summary** — ringkasan multi-hop causal chain yang melibatkan node ini
  (misal: "smoking → lung damage → cancer" untuk node "smoking").
- **Semantic role abstraction** — kategori RelationType yang dimiliki node (misal:
  node "api" punya CAUSAL ke "panas", "kebakaran", "kerusakan" → "api adalah agent
  of destruction").
- **Cross-node pattern** — pola yang muncul lintas node (misal: "X menyebabkan Y, Y
  menyebabkan Z" → chain pattern).

**Scope perubahan**: **KECIL untuk validasi; SEDANG kalau add Layer 4**

- Validasi (kecil): dokumentasikan di `ARCHITECTURE.md` bahwa 3-layer representation
  AGNN align dengan prinsip "layers of understanding" dari transformer. Ini hanya
  dokumentasi, tidak ada code change.
- Layer 4 (sedang): kalau memang diperlukan, tambah field di `Episome` (misal
  `causal_chain_summary: str`), lazy-generate di `_articulate` mirip `amodal_definition`.
  Butuh test baru + format state migration.

**Risiko**: Untuk Layer 4 — risk of over-abstraction. AGNN's value proposition adalah
aphantasic representation yang human-readable (text + typed edges). Kalau Layer 4 jadi
terlalu abstract (misal embedding numerical), kehilangan readability. Layer 4 harus
tetap text-based.

**Verdict**: **Validates existing design; Layer 4 is speculative** — prinsip kurikulum
memvalidasi 3-layer representation AGNN. Layer 4 (causal chain summary) mungkin worth-it
kalau AGNN mulai handle multi-hop reasoning yang complex (sekarang BA44 rules sudah
handle chain via CategoricalTransitivity/CausalChain, jadi Layer 4 mungkin redundant).
Note sebagai possible future work, tapi tidak urgent.

---

## 3. Konsep yang TIDAK Applicable

Untuk kelengkapan, berikut konsep kurikulum yang dievaluasi tapi tidak applicable ke AGNN.

| Konsep Kurikulum | Kenapa Tidak Applicable |
|---|---|
| Tokenization | AGNN pakai whitespace + punctuation strip (lihat `_tokenize` di `positional_cluster_learner.py`). BPE/SentencePiece tidak relevan karena graph nodes are surface forms, bukan sub-word units. |
| The Embedding Layer | AGNN pakai embedding sebagai sinyal sekunder (CA3 autoassociation). Prinsip "dense vs one-hot" sudah well-known; kurikulum tidak menambah teknik baru untuk AGNN. |
| Attention | AGNN pakai graph traversal (typed edges) bukan attention. QKV self-attention adalah mekanisme internal transformer; AGNN's design philosophy adalah graph yang reason, bukan model yang attend. Adopsi attention akan menambah complexity tanpa clear benefit. |
| Instruction Tuning and RLHF | Model Qwen3-0.6B sudah di-instruction-tune oleh manufacturer. AGNN tidak post-train model. Prinsip "alignment via feedback" sudah diimplementasi simple version via `reinforce()`/`penalize()` (mesolimbic circuit). |
| GPT-2 from Scratch | Bertentangan dengan visi core AGNN (small model + graph reasoning, bukan implement transformer dari nol). |
| Scaling Laws and Optimization | Bertentangan dengan thesis AGNN: small model + graph escape scaling laws. Kurikulum adalah filosofi opposite. |
| Training Data Engineering | Corpus AGNN (3290 kalimat) hand-curated. Teknik dedup/quality filtering kurikulum bisa improve corpus quality tapi tidak ubah arsitektur. |
| Training Infrastructure and Systems | AGNN tidak train model; tidak relevan. |
| Evaluation During Pretraining | AGNN tidak pretrain model. Evaluasi AGNN via test suite (331 tests) sudah cover ini. |
| Case Study - LLaMA 3 | Model spesifik, tidak relevan. |
| Supervised Fine-Tuning | AGNN tidak SFT model. |
| Preference Optimization (DPO, KTO) | AGNN's reinforce/penalize lebih sederhana; DPO bisa inspirasi pairwise confidence update tapi stretch. |
| Tools and Safety Tuning | Prinsip "tools augment small models" memvalidasi thesis AGNN (graph adalah "tool" yang dipakai model), tapi tidak menambah teknik. |
| Case Study on Tulu 3 | Model spesifik, tidak relevan. |
| Distillation | Filosofi opposite: AGNN = small model + external reasoning; distillation = compress large model into small. Tidak applicable. |
| LoRA | AGNN tidak fine-tune model. |
| Mixture of Experts (MoE) | AGNN's typed edges + BA44 rules sudah implementasi prinsip "route to specialist" (tiap RelationType punya rule specialist di `inferior_frontal_gyrus.py`). Tidak menambah teknik baru. |
| Optimizers | AGNN tidak gradient descent. |
| RL Fundamentals | `reinforce()`/`penalize()` AGNN adalah +/- delta sederhana, bukan RL. Prinsip "reward signal shapes policy" bisa inspirasi smarter confidence update tapi stretch. |
| RLHF | Sama seperti Preference Optimization. |

---

## 4. Kesimpulan Eksplisit

Setelah evaluasi menyeluruh terhadap 24+ chapter kurikulum Intuitive AI Academy:

**3 konsep genuinely worth dipertimbangkan** (ada gap real di AGNN yang bisa di-address):

1. **Relative Position (dari Positional Encoding chapter)** — Scope: KECIL. Address gap
   di connector detection & negation scope. Tidak urgent; baru worth-it kalau corpus
   bertambah kompleks.

2. **Predictive Signal (dari Learning to Predict chapter)** — Scope: SEDANG. Sinyal
   training yang lebih kaya dari counting. Tidak urgent; baru worth-it kalau corpus
   grow ke puluhan ribu kalimat.

3. **Contrastive Learning (dari Advanced Pretraining Objectives chapter)** — Scope: BESAR.
   Learned similarity metric sebagai generalisasi dari hardcoded weighted Jaccard.
   Long-term research direction; tidak untuk immediate adoption.

**1 konsep yang memvalidasi design existing** (tidak ada perubahan actionable):

4. **Layers of Understanding ↔ AGNN 3-Layer Aphantasic Representation** — Prinsip kurikulum
   memvalidasi design AGNN. Layer 4 (causal chain summary) mungkin worth-it kalau AGNN
   mulai handle multi-hop reasoning yang complex, tapi sekarang BA44 rules sudah handle
   chain via CategoricalTransitivity/CausalChain.

**Sisanya (20+ konsep)**: tidak applicable karena salah satu dari:
- Bertentangan dengan visi core AGNN (Scaling Laws, GPT-2 from Scratch, Distillation)
- Tidak relevan dengan design AGNN (LoRA, Optimizers, Training Infrastructure, SFT, DPO)
- Sudah diimplementasi di AGNN dengan cara berbeda (MoE ↔ typed edges + BA44 rules,
  Attention ↔ graph traversal, RL ↔ reinforce/penalize)
- Model-specific case study (LLaMA 3, Tulu 3)

### Catatan jujur tentang keterbatasan riset

Riset ini **tidak bisa mengakses isi chapter** (login wall). Penilaian dilakukan berdasarkan:
- Outline syllabus publik (24+ chapter titles)
- Prinsip well-known dari tiap konsep (positional encoding, attention, contrastive learning,
  dll. adalah konsep textbook)
- Cross-reference dengan arsitektur AGNN (baca `AGNN/ARCHITECTURE.md` + source code
  `AGNN/neocortex/positional_cluster_learner.py`, `AGNN/neocortex/bootstrap_classifier.py`,
  `AGNN/core.py`)

Konsekuensinya: riset ini menilai **prinsip konsep**, bukan **detail implementasi kurikulum**.
Kalau ada teknik spesifik di dalam chapter (visualisasi, analogi, demo) yang tidak terlihat
dari judul dan bisa memberi insight baru untuk AGNN, riset ini tidak menangkapnya.

### Rekomendasi tindak lanjut

1. **Immediate**: Tidak ada perubahan kode yang perlu diadopsi sekarang. 3 konsep yang
   genuinely applicable (relative position, predictive signal, contrastive learning) adalah
   future work untuk skala yang lebih besar.

2. **Short-term (kalau corpus bertambah kompleks)**: Pertimbangkan adopt relative position
   untuk connector detection (scope KECIL, benefit immediate).

3. **Long-term (kalau corpus grow ke 10K+ kalimat dan cluster count bertambah)**:
   Pertimbangkan predictive signal dan contrastive learning. Keduanya generalisasi yang
   lebih principled dari approach counting + weighted Jaccard saat ini.

4. **Tidak direkomendasikan**: Ganti backbone Qwen3-0.6B ke model lain, atau adopsi
   approach yang bertentangan dengan visi AGNN (scaling laws, distillation, full
   transformer reasoning).
