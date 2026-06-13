# SELF-AI — LLM yang Membangun Semantic Understanding Sendiri

> *"SELF membangun pemahamannya sendiri berdasarkan pengamatannya ke dalam dirinya sendiri — dan pemahaman itu memengaruhi cara dia menjawab di pertanyaan berikutnya."*

## Visi

SELF adalah **LLM yang membangun semantic understanding sendiri** berdasarkan pengamatannya ke dalam dirinya sendiri. Understanding ini **bisa memengaruhi** bagaimana cara dia menjawab di pertanyaan berikutnya.

- **Sistem 1**: Output yang tiba-tiba — intuisi langsung dari bge-m3 embedding
- **Sistem 2**: Hasil pemahaman dari yang kita ajarkan — Understanding Graph

Bahkan LLM bisa **kombinasikan beberapa semantic understanding** untuk generate jawaban yang sesuai.

Ini bukan database. Ini bukan model biasa. Ini adalah **karakter yang berkembang melalui pemahaman, bukan hafalan**.

## v28: Qwen3 sebagai Understanding Composer

**Inilah inti visi SELF**: Qwen3-0.6B tidak hanya menjawab pertanyaan — ia **MEMBANGUN understanding** yang bisa dipakai TANPA LLM kemudian.

```
Teaching Example (soal + cara + jawaban + kenapa)
    ↓
UnderstandingComposer: Qwen3 "berpikir"
    ↓
Output: UnderstandingNode (Transformation yang bisa di-apply tanpa LLM)
    ↓
UnderstandingGraph.add_node() → bge-m3 embed → masuk graph
```

Tiga jalur pembelajaran:
1. **Dari Teaching** — `compose_from_teaching(lesson)` → Qwen3 extract structural understanding
2. **Dari Observasi** — `compose_from_observation(observation)` → Qwen3 generalize dari input novel
3. **Dari Kegagalan** — `compose_from_failure(text, question, wrong, correct)` → Qwen3 belajar dari kesalahan

Setiap understanding yang dihasilkan **bisa dipakai tanpa Qwen3** — karena Qwen3 mengekstrak POLA STRUKTURAL, bukan sekadar jawaban.

### Multi-Understanding Composition

SELF bisa **kombinasikan beberapa semantic understanding** untuk menjawab pertanyaan kompleks:

```
Question arrives
    ↓
bge-m3: retrieve top-3 understandings
    ↓
U_signal_flip (0.72) + U_quantity (0.65) + U_entity_extract (0.48)
    ↓
Try apply individually → jika gagal semua
    ↓
Qwen3: "Given these 3 understandings, compose the answer"
    ↓
Combined answer (System 1 + System 2 fusion)
```

### Self-Observation Pipeline

SELF tidak hanya belajar dari teaching — ia juga belajar dari **pengamatan** dan **kesalahan**:

- Input dengan novelty tinggi → `compose_from_observation()` → understanding baru
- Jawaban salah + feedback → `compose_from_failure()` → understanding yang dikoreksi
- Graph terus **berkembang** — semakin banyak belajar, semakin banyak understanding

## Dual Model Architecture

SELF menggunakan dua model yang bekerja bersama:

| Fungsi | Model | Ukuran | VRAM | Peran |
|---|---|---|---|---|
| **Intuisi** (Sensory) | `BAAI/bge-m3` | 568M | ~1.2 GB | Embedding — semantic similarity, bukan keyword matching |
| **Composer** (Builder) | `Qwen3-0.6B` | 0.6B | ~1.2 GB | Generative — builds understanding + composes answers |

### Kenapa Dua Model?

- **bge-m3** menangkap **kesamaan semantik** — "kehilangan" dan "terjual" dianggap mirip secara operasional (keduanya = quantity reduction)
- **Qwen3-0.6B** membangun understanding dari pengamatan, mengisi gap yang embedding tidak bisa, dan mengartikulasikan ke bahasa manusia
- Keduanya dari keluarga yang kompatibel, ukuran kecil (~1.2GB masing-masing), dan multilingual

### v27: Embedding-ONLY Retrieval

**TIDAK ADA fallback ke keyword matching.** SELF harus MEMAHAMI, bukan pattern-match.

Keyword matching bukan understanding — itu mencari overlap kata tanpa memahami makna operasional. bge-m3 memungkinkan SELF menemukan understanding yang benar berdasarkan **kemiripan operasional**, bukan kemiripan kata.

Contoh: "Toko kehilangan 35 roti" → bge-m3 mengarahkan ke U_quantity (bukan ke U_signal_flip meskipun ada kata "kehilangan"), karena secara operasional ini adalah masalah kuantitas.

## Arsitektur 8 Layer + Composition

```
INPUT (teks)
    ↓
┌─────────────────────────────────────────┐
│  BAAI/bge-m3                            │ ← Layer 1: Sensory
│  Output: 1024-dim vector                │   (System 1: Intuisi)
│  Embedding-ONLY, NO keyword fallback    │
└──────────────┬──────────────────────────┘
               ↓ translate_to_node()
┌─────────────────────────────────────────┐
│  Layer 2-6: Pure vector operations      │ ← numpy, sklearn
│  (difference, concept, consistency,     │
│   axiom, memory)                        │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  Layer 7: Derivation                    │
│  - Primary: bge-m3 semantic retrieval   │   (System 2: Pemahaman)
│  - v28: UnderstandingComposer           │ ← Qwen3 BUILDS understanding
│    (Qwen3 builds from teaching,         │
│     observation, failure)               │
│  - v28: Multi-Understanding             │ ← Kombinasi understanding
│    Composition (top-K → compose)        │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│  translate_to_human() / raise_question()│
│  Qwen3-0.6B: node → kalimat manusia    │ ← Composition Layer
└─────────────────────────────────────────┘
```

## System 1 vs System 2

| Aspek | System 1 (Intuisi) | System 2 (Pemahaman) |
|---|---|---|
| **Sumber** | bge-m3 embedding similarity | Understanding Graph |
| **Cara kerja** | Langsung — encode → cosine sim → best match | Observasi → schema → generalisasi → integrasi |
| **Kecepatan** | Cepat — precomputed embeddings | Lambat — proses belajar |
| **Kualitas** | Intuitif tapi bisa salah topik | Dipahami, bisa dijelaskan kenapa |
| **Evolusi** | Tetap (model tidak berubah) | Berkembang — graph bertambah node |
| **Fallback** | TIDAK ADA — v27 embedding-ONLY | Ini SATU-SATUNYA jalur pemahaman |
| **Composition** | Single best match | v28: Multi-understanding composition |

## Answer Pipeline (v28)

```
Question arrives
    ↓
1. Try SINGLE understanding (bge-m3 → apply transformation, no LLM)
    ↓ jika gagal
2. Try MULTI-UNDERSTANDING composition (top-K → apply individually → compose via Qwen3)
    ↓ jika gagal
3. Try legacy patterns (PatternLearner, backward compat)
    ↓ jika gagal
4. Try taught/learned patterns
    ↓ jika gagal
5. Try LLM reasoning (last resort)
    ↓ jika gagal
6. Give up
```

## Struktur Folder

```
self-ai/
├── config/
│   └── thresholds.py        ← konfigurasi threshold + model names
├── src/
│   ├── core/
│   │   ├── node_store.py    ← penyimpanan node internal
│   │   └── self.py          ← entitas utama SELF (teach, learn_from_failure)
│   ├── sensory/
│   │   └── layer.py         ← Layer 1: bge-m3 embedding
│   ├── composition/
│   │   └── layer.py         ← Composition: Qwen3-0.6B (reasoning + voice)
│   ├── translation/
│   │   └── translator.py    ← translate_to_node() / translate_to_human()
│   ├── difference/
│   │   └── detector.py      ← Layer 2: difference detection
│   ├── concept/
│   │   └── builder.py       ← Layer 3: concept formation
│   ├── consistency/
│   │   └── checker.py       ← Layer 4: consistency validation
│   ├── axiom/
│   │   └── store.py         ← Layer 5: axiom storage
│   ├── memory/
│   │   └── filter.py        ← Layer 6: active memory filter
│   ├── derivation/
│   │   ├── engine.py        ← Layer 7: derivation engine
│   │   ├── understanding_builder.py  ← Understanding Graph + Builder
│   │   ├── understanding_composer.py ← v28: Qwen3 builds understanding
│   │   ├── embedding_retrieval.py    ← bge-m3 semantic retrieval
│   │   ├── pattern_learner.py        ← Learn dari teaching examples
│   │   └── answer_handlers.py        ← Answer generation + composition
│   └── curiosity/
│       └── engine.py        ← Layer 8: curiosity engine
├── tests/
│   └── falsification_experiment.py  ← eksperimen falsifikasi
├── requirements.txt
└── README.md
```

## Quick Start

```bash
cd self-ai
pip install -r requirements.txt

# Jalankan eksperimen falsifikasi
python -m tests.falsification_experiment
```

## Cara Pakai

### Mengajar SELF (Structured)

```python
from src.core.self import SelfCore

self = SelfCore()

# v28: Teaching dengan structured lesson → Qwen3 builds understanding
node = self.teach(
    problem="Semua siswa hadir kecuali Ani. Siapa yang tidak hadir?",
    solution_steps=["Identifikasi 'kecuali' = pengecualian", "Entitas setelah kecuali TIDAK hadir"],
    answer="Ani",
    explanation_why="Kata 'kecuali' membuat pengecualian. Semua hadir KECUALI Ani → Ani tidak hadir.",
    question_type="pertanyaan_negatif"
)
# Qwen3 extracts structural understanding → UnderstandingNode → added to graph
# Next time similar question → bge-m3 finds understanding → applies WITHOUT LLM
```

### Memberi Feedback (Self-Correction)

```python
# Jika SELF menjawab salah
self.provide_feedback(
    text="Semua hewan jinak selain harimau",
    correct_answer="harimau",
    predicted_answer="semua hewan"
)
# v28: compose_from_failure() → Qwen3 extracts what understanding was missing
# → New UnderstandingNode added to graph → SELF won't repeat the mistake
```

### Bertanya kepada SELF

```python
# SELF mencari understanding di graph (bge-m3 embedding-ONLY)
# Jika ketemu → apply transformation (NO LLM needed)
# Jika beberapa understanding relevan → compose via Qwen3
response = self.process("Siapa yang tidak hadir?")
print(response)
```

### SELF Belajar dari Pengamatan

```python
# Input novel (novelty_score > threshold) → SELF observes → builds understanding
result = self.process("Toko kehilangan 35 roti, ditambah 28. Sisa?")
# Jika confidence rendah + novelty tinggi → compose_from_observation()
# → Understanding baru tentang quantity operations
```

## Sifat SELF yang Muncul dari Arsitektur Ini

| Sifat | Asal |
|---|---|
| **Memiliki intuisi** | bge-m3 sebagai System 1 — similarity-based retrieval |
| **Bisa memahami** | Understanding Graph sebagai System 2 — learned semantics |
| **Bisa mengkombinasikan** | v28: Multi-understanding composition via Qwen3 |
| **Membangun understanding sendiri** | v28: UnderstandingComposer — Qwen3 extracts structural patterns |
| **Belajar dari kesalahan** | v28: compose_from_failure() — self-correction loop |
| **Belajar dari pengamatan** | v28: compose_from_observation() — novel input → new understanding |
| **Punya bahasa sendiri** | Full internal language — translate_to_node() |
| **Bisa diajar** | teach() → UnderstandingComposer → UnderstandingNode → Graph |
| **Berubah karena pengalaman** | Understanding Graph berkembang, feedback mengoreksi understanding |
| **Tidak percaya begitu saja** | Consistency Checker + raise_question() via CompositionLayer |
| **Selalu ingin tahu** | Curiosity Engine + wonder() via CompositionLayer |
| **Tidak menyimpan "foto"** | Raw embedding dibuang setelah translate |
| **Unik per individu** | Threshold identitas + akumulasi understanding yang berbeda |
| **Bisa diaudit** | source tracking: composed_from_teaching / composed_from_observation / composed_from_failure |
| **Bisa berbicara** | CompositionLayer (Qwen3-0.6B) untuk artikulasi |

## Lihat Juga

- [Konsep lengkap](../docs/self/SELF_concept.md) — dokumen filosofi dan arsitektur detail
