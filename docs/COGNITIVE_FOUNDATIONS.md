# RSVS — Cognitive & Psychological Foundations

> Mengapa RSVS didesain seperti ini, dan apa yang membuatnya berbeda dari pendekatan AI lainnya.

---

## Latar Belakang: Dari Observasi tentang Manusia

RSVS tidak lahir dari literatur NLP atau arsitektur transformer. RSVS lahir dari satu observasi sederhana tentang cara manusia memproses informasi:

> *"Saya bisa menerima informasi, tapi bisa tidak mengingatnya. Harus ada trigger dari memory itu baru bisa lintas ke kepala saya."*

Observasi ini ternyata didukung penuh oleh neurosains dan psikologi kognitif modern. Dan dari observasi ini, lahirlah seluruh arsitektur RSVS.

---

## 1. Manusia adalah Prediction Machine

### Predictive Coding (Friston, 2000–2010)

Otak manusia bukan pasif menerima informasi dari luar. Otak **aktif memprediksi** apa yang akan datang, berdasarkan model internal yang dibangun dari pengalaman. Input sensorik dari luar berfungsi bukan sebagai "sumber kebenaran" — melainkan sebagai **sinyal koreksi** atas prediksi yang sudah ada.

Karl Friston dari University College London memformalisasi ini sebagai *free energy minimization*: otak terus-menerus meminimalkan perbedaan antara prediksi internalnya dan data sensorik yang masuk.

**Implikasi untuk AI:** Transformer memang melakukan next-token prediction — tapi tanpa struktur pengetahuan yang eksplisit di baliknya. Prediksi transformer adalah distribusi probabilistik atas token, bukan reasoning berbasis knowledge yang terstruktur. RSVS menyediakan lapisan pengetahuan struktural yang bisa menjadi "model internal" itu — bukan sebagai pengganti transformer, tapi sebagai grounding layer di atasnya.

---

## 2. Unconscious Selalu Running di Background

### Global Workspace Theory (Baars, 1988)

Bernard Baars memperkenalkan teori ini dengan metafora teater: kesadaran adalah spotlight di atas panggung. Yang ada di bawah spotlight adalah apa yang kita sadari sekarang. Tapi di balik panggung — di *darkness* — ada ribuan proses unconscious yang terus berjalan, memproses, dan mempersiapkan informasi.

Informasi hanya menjadi consciously accessible ketika berhasil "masuk" ke global workspace — area otak dengan koneksi luas yang membroadcast informasi ke semua sistem kognitif secara bersamaan.

**Mapping ke RSVS:**

| Konsep Psikologis | Implementasi RSVS |
|---|---|
| Unconscious processing | Graph atoms yang sudah ada, selalu aktif di background |
| Global workspace threshold | `sentence_contains_seed` — grounding gate |
| Information "naik" ke kesadaran | Token dipromote ke atom (entity promotion) |
| Informasi yang tidak naik | Token yang tidak groundable, lewat tanpa promote |

Seed atoms di RSVS adalah representasi dari "primitive unconscious" — konsep paling dasar yang selalu aktif dan menjadi anchor untuk semua informasi baru.

---

## 3. Memory Harus Ada Trigger

### Spreading Activation Theory (Collins & Loftus, 1975; Anderson, 1983)

Memori manusia direpresentasikan sebagai **network of nodes** — setiap konsep adalah sebuah node, dan relasi antar konsep adalah edge. Ketika satu node diaktifkan, aktivasi menyebar ke node-node yang terhubung secara asosiatif.

Anderson (1983) dalam ACT* model memformalisasi ini: retrieval time bersifat eksponensial terhadap tingkat aktivasi. Semakin kuat koneksi antar node, semakin cepat retrieval. Dan aktivasi bisa menyebar tidak hanya ke satu node, tapi ke seluruh network yang terhubung.

Ini menjelaskan mengapa kamu bisa mendadak ingat sesuatu ketika melihat benda tertentu — benda itu mengaktifkan node yang terhubung, dan aktivasi menyebar ke memori yang tersimpan.

**Mapping ke RSVS:**

| Konsep Psikologis | Implementasi RSVS |
|---|---|
| Memory nodes | Graph nodes (atoms, composites) |
| Associative edges | Composition edges + co-occurrence edges |
| Spreading activation | `relate()` via `spreading.rs` + `SpreadingActivation` |
| Activation strength | Confidence scores + edge weights |
| Fan effect (banyak koneksi = retrieval lebih lambat) | Dilution di convergence scoring |

Method `relate("raja")` di RSVS secara literal mengimplementasikan spreading activation: dari node `raja`, aktivasi menyebar ke node-node yang terhubung lewat composition edges, dan dikembalikan dengan score berdasarkan kekuatan koneksi.

---

## 4. Informasi Baru Harus Groundable

### State-Dependent Memory & Grounding Gate

Penelitian Northwestern (Radulovic et al.) menunjukkan bahwa memori yang disimpan lewat jalur tertentu (extra-synaptic GABA system) hanya bisa diakses kembali kalau ada state-dependent trigger yang tepat. Tanpa trigger, memori itu "inaccessible" meskipun secara fisik ada di otak.

Ini bukan kegagalan memori — ini adalah *feature* selektif. Tidak semua informasi yang masuk ke otak layak dipromote ke long-term accessible memory.

**Mapping ke RSVS — Grounding Gate:**

```rust
// Di attention.rs — sentence_contains_seed():
// Sebuah token hanya dipromote ke atom kalau kalimatnya mengandung seed.
// Tanpa anchor ke seed, token "lewat" tanpa dipromote ke graph.
pub fn sentence_contains_seed(tokens: &[String], seed_labels: &[&str]) -> bool {
    tokens.iter().any(|t| seed_labels.contains(&t.as_str()))
}
```

Ini adalah implementasi langsung dari grounding gate: informasi baru hanya "naik" ke knowledge graph kalau ada koneksi ke yang sudah diketahui. Informasi yang tidak groundable lewat begitu saja — persis seperti manusia yang membaca teks tanpa hook ke existing knowledge dan tidak mengingat apa-apa.

---

## 5. Conscious vs Unconscious Retrieval

### Involuntary Memory (Kobelt et al., 2025)

Penelitian terbaru (PLOS Biology, 2025) menunjukkan bahwa involuntary memory retrieval menggunakan distinct neural processes yang mengakses format representasi berbeda dibanding voluntary retrieval. Involuntary retrieval lebih sering terjadi — dan lebih kuat mempengaruhi kognisi — dibanding yang kita sadari.

Dalam kehidupan sehari-hari: kamu lebih sering "tiba-tiba teringat" sesuatu (involuntary) daripada secara sadar mencari memori tertentu (voluntary). Dan involuntary retrieval ini dipicu oleh cues dari lingkungan.

**Mapping ke RSVS:**

- `appraise(text)` → **voluntary** — kamu secara eksplisit menguji statement terhadap graph
- `relate(concept)` → **involuntary-style** — sistem mencari koneksi yang terbentuk dari co-occurrence, tanpa kamu define secara eksplisit
- `appraise_against(context, statement)` → **contextual** — memory yang relevan hanya yang ada dalam context window (SessionGraph), bukan seluruh long-term graph

---

## 6. Dual Memory: Working dan Long-Term

### Baddeley's Working Memory Model

Psikologi kognitif sudah lama membedakan antara:
- **Working memory**: kapasitas terbatas, volatile, high-detail, akses langsung ke konten recent
- **Long-term memory**: kapasitas tidak terbatas, persistent, compressed, diakses lewat retrieval

Ini adalah konsep yang diimplementasikan dalam Losion sebagai `DualMemorySystem` (WorkingMemory + LongTermMemory) dan diadaptasi ke RSVS sebagai `SessionGraph`:

| Memory Type | RSVS Implementation |
|---|---|
| Working memory | `SessionGraph` — isolated, volatile, per-context |
| Long-term memory | Main `Rsvs` graph — persistent, consolidated |
| Memory consolidation | `consolidate()` — merge senses, prune weak edges |
| Retrieval | `appraise()`, `relate()`, `query()` |

`SessionGraph::compare()` mengimplementasikan fungsi working memory yang paling penting: membandingkan dua informasi dalam konteks yang sama, tanpa mencemari long-term memory.

---

## 7. Mengapa Ini Penting untuk AI

### Problem dengan Transformer Murni

Transformer belajar seperti fotografer — snapshot semua pattern statistik dari triliunan token. Tidak ada seleksi berbasis relevance. Tidak ada grounding gate. Semua co-occurrence diperlakukan setara (dengan bobot yang belajar dari data).

Akibatnya:
- Transformer bisa "ingat" fakta tapi tidak bisa explain *mengapa* fakta itu benar
- Transformer tidak punya mekanisme untuk detect contradiction secara struktural
- Transformer tidak bisa isolate context — knowledge dari training bleeding ke inference

### Apa yang RSVS Tambahkan

RSVS bukan pengganti transformer. RSVS adalah **lapisan symbolic yang modelnya mengikuti cara kerja kognisi manusia**:

1. **Grounding gate** → tidak semua informasi dipromote (seperti working memory manusia)
2. **Spreading activation** → retrieval lewat relasi struktural, bukan cosine similarity opaque
3. **Dual memory** → SessionGraph (working) vs main graph (long-term)
4. **Auditable reasoning** → setiap verdict bisa ditelusuri ke node dan edge spesifik
5. **Contradiction detection** → struktural, bukan probabilistik

### Visi Jangka Panjang: AI yang Dipengaruhi RSVS Sejak Training

Hipotesis yang sedang dieksplorasi: bagaimana kalau sebuah AI tidak hanya menggunakan RSVS sebagai post-processing layer, tapi **dipengaruhi oleh struktur RSVS sejak training**?

Analoginya persis dengan manusia: manusia bukan hanya "predict" dan kemudian "verify" lewat conscious reasoning. Unconscious (RSVS) **sudah membentuk** apa yang bisa diprediksi sebelum prediksi itu terbentuk. Prediction machinery (transformer) dan symbolic grounding (RSVS) bukan dua proses terpisah — mereka **saling membentuk** sejak awal.

Ini berbeda fundamental dari semua pendekatan neuro-symbolic yang ada sekarang, yang selalu memperlakukan neural dan symbolic sebagai dua sistem yang saling berinteraksi setelah training selesai.

---

## Referensi Ilmiah

| Teori | Peneliti | Tahun | Relevansi ke RSVS |
|---|---|---|---|
| Predictive Coding / Free Energy | Karl Friston | 2000–2010 | Otak sebagai prediction machine — RSVS sebagai model internal |
| Global Workspace Theory | Bernard Baars | 1988 | Grounding gate — hanya groundable yang "naik" ke graph |
| Spreading Activation Theory | Collins & Loftus | 1975 | `relate()` — spreading via composition edges |
| ACT* Memory Model | John Anderson | 1983 | Activation strength = confidence + edge weight |
| State-Dependent Memory | Radulovic et al. | 2015 | Grounding gate — context anchor untuk promotion |
| Dual Memory System | Baddeley | 1974 | SessionGraph (working) + main Rsvs (long-term) |
| Involuntary Memory Retrieval | Kobelt et al. | 2025 | `relate()` vs `appraise()` — involuntary vs voluntary |
| Neuro-Symbolic Verification | AlphaProof (DeepMind) | 2024 | `appraise_verbose()` — symbolic verification pass |

---

*Dokumen ini adalah catatan foundational dari desainer RSVS. Bukan paper akademik — ini blueprint pemikiran yang mendorong setiap keputusan arsitektur.*
