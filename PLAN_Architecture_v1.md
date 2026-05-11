# RENCANA: Arsitektur AI Berbasis RSVS — "The Genius Who Remembers Everything"

> Dokumen rencana v1 — Disusun berdasarkan analisis novel "The Martial Genius Who Remembers Everything" dan arsitektur RSVS yang sudah ada di `/home/z/my-project/RSVS/`

---

## Struktur Folder

| Path | Fungsi |
|------|--------|
| `/home/z/my-project/RSVS/` | Kode RSVS asli — **boleh di-edit untuk fix bug / adjust** |
| `/home/z/my-project/workspace/` | Folder kerja — layer baru di atas RSVS |
| `/home/z/my-project/workspace/rsvs_genius/` | Package Python: 5 cognitive layers |
| `/home/z/my-project/workspace/PLAN_Architecture_v1.md` | Dokumen ini |

---

## Inti Konsep: Paralel Novel ↔ Sistem AI

Dari novel **"The Martial Genius Who Remembers Everything"** (모든걸 기억하는 천재무사), karakter utama **Jin Soun** punya kemampuan:

- **Mengingat SEMUA** yang pernah dilihat/didengar/dialami — hyperthymesia supernatural
- **Mengenali teknik** dari cues minimal (suara langkah kaki → identifikasi teknik rahasia "Blood Serpent Dance Step")
- **Tahu semua kelemahan** lawan karena 30 tahun ingatan pertempuran
- **TAPI tubuhnya third-rate** — gap antara KNOWING dan DOING

Ini **persis** masalah yang RSVS coba selesaikan: AI yang punya pengetahuan struktural sempurna tapi butuh execution layer untuk bertindak.

---

## Scene Kunci: Pencurian Snow Plum Pill — Blueprint Bagaimana Memory Bekerja

Scene ini (Chapter 3–6) adalah **contoh paling jelas** bagaimana Jin Soun menggunakan memori untuk menghubungkan kejadian-kejadian terpisah menjadi satu kesimpulan. Ini BUKAN sekadar "deductive reasoning" — ini adalah **pattern completion across disparate memories**.

### Insiden yang Terpisah-Pisah

| Insiden | Sumber Data | Konteks |
|---------|-------------|---------|
| **A**: Gye Cheolyeong cedera di Taeul Sect | Laporan internal sect | Masalah pengajaran |
| **B**: Snow Plum Pill dicuri dari Gyeryong Merchant Guild di Hefei | Laporan cabang Hefei | Masalah merchant guild |
| **C**: Soul-Chasing Guest (Ju Jangmok) menghilang hari yang sama | Catatan dark faction | Masalah dark path |
| **D**: Diancang Five Swords pecah jadi 3+2, pair Gu Ilmu & Jang Hangi punya success rate tinggi | Laporan misi Martial Alliance | Operasi rutin |

Empat insiden ini ada di **4 departemen berbeda**. Tidak ada satu orang pun yang punya alasan untuk cross-reference keempatnya secara simultan. Kecuali Jin Soun.

### Proses Step-by-Step Jin Soun

```
STEP 1: RECALL — "Walking Library"
├─ Ingat cerita tavern tentang pencurian Snow Plum Pill
├─ Ingat bahwa dia pernah menghafal SEMUA dokumen Simhyeon Pavilion
└─ Mulai mencari di "perpustakaan mental"nya

STEP 2: CROSS-REFERENCE — Tiga Sumber Data Sekaligus
├─ Laporan Bulanan Cabang Hefei → siapa yang ada di Hefei kapan?
├─ Catatan Masuk-Keluar Martial Alliance → siapa yang bergerak ke/luar Hefei?
└─ Laporan Misi → siapa yang punya misi di Hefei sekitar tanggal pencurian?

STEP 3: FILTER — Metode Deduksi Jegal Cheon
├─ Organisir data yang terkumpul
├─ Eliminasi yang tidak relevan
└─ Empat tersangka muncul: Ju Jangmok, Song Wongi, Gu Ilmu, Jang Hangi

STEP 4: ELIMINASI — Logika Sederhana
├─ Song Wongi → kaya enough untuk beli 100 pil, tidak ada motif → ELIMINASI
├─ Ju Jangmok → tersangka obvious, menghilang hari yang sama → TAPI...
└─ Gu Ilmu + Jang Hangi → ada di Hefei 3 hari sebelum, misi dari dalam Diancang sendiri

STEP 5: ANOMALY DETECTION — "Ini Tidak Masuk Akal"
├─ Tidak ada satu orang pun yang muncul sebagai pengguna Snow Plum Pill
├─ Tidak ada pencuri baru yang muncul setelah Ju Jangmok menghilang
├─ Pil dan pencuri sama-sama menghilang tanpa jejak
└─ Ini ANOMALI — jika Ju Jangmok mencuri, pil harus muncul di pasar gelap

STEP 6: PATTERN COMPLETION — "Jadi begitu triknya!"
├─ Ju Jangmok = kambing hitam / cover
├─ Gu Ilmu + Jang Hangi = pencuri sebenarnya
├─ Misi dari dalam Diancang = inside job yang sudah direncanakan lama
├─ Pil dikonsumsi internal di Diancang → makanya tidak muncul di luar
└─ Success rate tinggi = hasil boost dari Snow Plum Pill

STEP 7: OUTPUT — Kesimpulan + Action
├─ Klaim: "Diancang Five Swords (pair) mencuri Snow Plum Pill"
├─ Evidence: Tanggal, pergerakan, anomali, motif
├─ Action: Pasang jebakan, konfrontasi, kill
└─ Bukti bahwa knowing > doing kalau kamu tahu KELEMAHAN lawan
```

### Mengapa Orang Lain Tidak Bisa Melihat Pola Ini

1. **Tiga sumber data di departemen berbeda** — tidak ada yang punya akses ke semua sekaligus
2. **Insiden di konteks berbeda** — masalah merchant guild ≠ masalah dark faction ≠ operasi rutin
3. **Ju Jangmok adalah tersangka "obvious"** — setelah dia menghilang, investigasi berhenti
4. **Tidak ada yang punya complete recall** — tanpa bisa instan ingat tanggal, misi, dan koneksi internal, pola tetap tersembunyi

### Mapping ke RSVS: Inilah Yang Kita Butuhkan

| Langkah Jin Soun | Fungsi RSVS | Status |
|------------------|-------------|--------|
| Recall cerita tavern | `query()` + `relate()` | ✅ Ada |
| Cross-reference 3 sumber | `spreading_activation` across domains | 🔧 Perlu extend |
| Filter via metode Jegal Cheon | `context_query()` + `mcts_query()` | ✅ Ada |
| Eliminasi tersangka | `structural_similarity()` + `substitution_analysis()` | ✅ Ada |
| Deteksi anomali | `appraise()` → "disagree" saat expected ≠ observed | 🔧 Perlu extend |
| Pattern completion | `compose()` dari fragmen terpisah | 🔧 Perlu build |
| Output kesimpulan | `appraise()` + reasoning chain | 🔧 Perlu extend |

---

## Koreksi Konsep (Revisi Berdasarkan Scene Snow Plum Pill)

### 1. Context/Plugin/Internet → **Context Layer**

**Apa yang kamu maksud:**
- Seperti NotebookLM: kita batasi AI menjawab hanya berdasarkan knowledge yang kita tentukan
- Versi coder: GitHub repos + library code sebagai context
- Penting untuk pajak/regulasi yang butuh keteraturan dan sumber terdefinisi

**Bagaimana RSVS sudah punya ini:**
- `SessionGraph` = working memory (volatile, per-context) — ini SUDAH context-bounded
- `sentence_contains_seed` = grounding gate — hanya info yang groundable ke existing knowledge yang masuk
- Tapi belum ada: **explicit scope control** (batasan "hanya jawab dari sumber X")

**Rencana versi dasar (general + internet):**
```
Context Layer = {
  1. RSVS Knowledge Graph (pengetahuan yang sudah terbentuk)
  2. Internet Search Plugin (web_search via API)
  3. Scope Filter: "hanya gunakan sumber berikut" → kontrol jawaban
}
```

**Analogi novel:** Jin Soun punya Simhyeon Pavilion (semua catatan rahasia) + 30 tahun pengalaman. Tapi dia juga tau kapan harus pakai sumber mana. Saat menyelidiki pencurian, dia secara selektif hanya mengakses: laporan Hefei, catatan masuk-keluar, dan laporan misi — bukan seluruh perpustakaannya. Context Layer = kemampuan membatasi diri ke sumber tertentu.

---

### 2. Riwayat Chat → **Situation Layer**

**Apa yang kamu maksud:**
- Versi coder: codespace + permintaan user + kode yang berhubungan = situasi saat ini
- Bukan sekadar "chat history", tapi **state of the world** sekarang

**Bagaimana RSVS sudah punya ini:**
- Event log (`events.rs`) = append-only log dengan sequence numbers
- Snapshot system = state yang bisa di-restore
- Tapi belum ada: **semantic chat history** (bukan raw log, tapi meaning dari percakapan)

**Rencana versi dasar:**
```
Situation Layer = {
  1. Event Stream (sudah ada di RSVS)
  2. Chat Semantic Index: setiap percakapan di-INGEST ke RSVS
     → artinya AI "mengingat" konteks percakapan secara struktural
     → bukan hanya token history, tapi graph of meaning
  3. Active Context Window: sense yang sedang aktif = situasi saat ini
}
```

**Analogi novel:** Jin Soun sedang berada di Hefei → semua sense tentang Hefei, merchant guild, dan Snow Plum Pill aktif. Dia tidak perlu mengakses seluruh 30 tahun — hanya yang relevan untuk situasi sekarang. Active senses = context window yang hidup, bukan statis.

---

### 3. RSVS → **Relation Comprehension Engine** (Sudah Ada, Perlu Extend)

**Ini sudah ada dan sudah kuat.** Yang perlu ditambahkan:

**Yang sudah ada:**
- `structural_similarity()` → memahami relasi antar konsep
- `substitution_analysis()` → memahami apa yang mengubah A menjadi B
- `relate()` → spreading activation untuk temukan koneksi
- `convergence.rs` → deteksi konsep yang sama dari surface form berbeda

**Yang perlu ditambah untuk versi dasar:**
```
RSVS Extension = {
  1. Cross-domain linking (sudah via convergence engine)
  2. Internet-ingested knowledge → masuk ke graph via ingest()
  3. Query-time: pilih sense yang aktif berdasarkan context
}
```

**Analogi novel:** Ini JIWA dari Jin Soun. Dia mengerti bahwa "raja" dan "ratu" beda di satu komposisi (laki_laki vs perempuan). Dia mengerti bahwa kehadiran Ju Jangmok di Hefei + tanggal yang sama dengan pencurian = koneksi struktural, bukan kebetulan. RSVS = kemampuan memahami relasi struktural antar kejadian yang tampak tidak terkait.

---

### 4. Predictive Coding Training → **Belief Update Engine**

**Apa yang kamu maksud:**
- "Aku predict X, ternyata Y, update belief"
- Bukan gradient descent, tapi belief revision berdasarkan prediksi vs realitas

**Bagaimana RSVS sudah punya ini (sebagian):**
- `GroundingEvidence` = confirming vs contradicting contexts
- `reflection.rs` = CONFIRM/REVIEW/REVISE/RETIRE berdasarkan grounding
- `autonomy.rs` = EMA confidence update
- Tapi belum ada: **explicit prediction → observation → update loop**

**Rencana versi dasar:**
```
Predictive Coding Engine = {
  1. Prediction: RSVS compose() → predict komposisi sebuah konsep
     "Berdasarkan konteks, aku predict konsep X terdiri dari A, B, C"
  2. Observation: Ingest realita → bandingkan dengan prediksi
  3. Belief Update: 
     - Jika prediksi benar → CONFIRM, naikkan grounding
     - Jika prediksi salah → REVISE, turunkan grounding, prune komposisi
     - Ini SUDAH ada di reflection.rs, tapi perlu diexpose sebagai loop eksplisit
  
  Formula: belief_new = belief_old + η × (observed - predicted)
  Ini PERSIS Friston's free energy = meminimalkan prediction error
}
```

**Analogi novel:** Jin Soun memprediksi "Ju Jangmok adalah pencuri" (obvious theory) → tapi observasi menunjukkan "tidak ada yang mengonsumsi pil" → PREDICTION ERROR → update belief → "Ju Jangmok bukan pencuri sebenarnya, dia kambing hitam." Setiap kali prediksi tidak cocok dengan realita, belief diupdate. Ini PREDICTIVE CODING.

---

### 5. Text Output → **Pattern Completion Output** (REVISI KRITIS)

**Ini bukan sekadar "deductive reasoning."** Scene Snow Plum Pill menunjukkan bahwa output Jin Soun BUKAN dimulai dari hipotesis lalu diuji — tapi dimulai dari **RECALL massal → cross-reference → anomaly detection → pattern completion**.

#### Proses yang Sebenarnya Terjadi

```
BUKAN:
  Hypothesis → Test → Conclusion (deductive sederhana)

TAPI:
  Recall → Cross-Reference → Anomaly → Pattern Completion → Narrative

  ┌─────────────────────────────────────────────────────────┐
  │  1. TRIGGER: "Snow Plum Pill dicuri"                    │
  │     → relate("snow_plum_pill") aktifkan semua koneksi   │
  │                                                          │
  │  2. RECALL MASSAL: Spreading activation dari trigger     │
  │     → Hefei (lokasi) aktif                              │
  │     → Gyeryong Guild (korban) aktif                     │
  │     → Ju Jangmok (tersangka obvious) aktif              │
  │     → Diancang Five Swords (ada di Hefei) aktif         │
  │     → Tanggal-tanggal spesifik aktif                    │
  │     → Semua node yang terhubung ke "pencurian" aktif    │
  │                                                          │
  │  3. CROSS-REFERENCE: Structural comparison               │
  │     → Ju Jangmok di Hefei = tanggal sama dengan pencurian│
  │     → Gu Ilmu + Jang Hangi di Hefei = 3 hari sebelum    │
  │     → Misi dari dalam Diancang = inside job             │
  │     → Success rate pair lebih tinggi = ada boost         │
  │                                                          │
  │  4. ANOMALY DETECTION: Expected ≠ Observed               │
  │     → Expected: pil muncul di pasar gelap / pencuri baru│
  │     → Observed: TIDAK ADA yang mengonsumsi pil          │
  │     → Delta: prediction error → REVISE belief           │
  │                                                          │
  │  5. PATTERN COMPLETION: Fragmen → Pola Utuh              │
  │     → Ju Jangmok = cover (bukan pelaku)                 │
  │     → Diancang pair = pelaku sebenarnya                 │
  │     → Inside job = penjelasan kenapa misi dari dalam    │
  │     → Konsumsi internal = penjelasan kenapa pil hilang  │
  │                                                          │
  │  6. NARRATIVE OUTPUT: Kesimpulan yang bisa ditelusuri    │
  │     "Diancang Five Swords (Gu Ilmu & Jang Hangi)        │
  │      mencuri Snow Plum Pill menggunakan Ju Jangmok      │
  │      sebagai kambing hitam. Operasi ini direncanakan    │
  │      dari dalam Diancang Sect sendiri, karena misi      │
  │      ke Hefei di-assign oleh anggota Diancang.          │
  │      Pil dikonsumsi internal, bukan dijual, yang        │
  │      menjelaskan kenapa tidak ada jejak di pasar."      │
  │                                                          │
  │     Evidence chain:                                      │
  │     [Tanggal Hefei] → [Diancang misi] → [Inside job]   │
  │     [Ju Jangmok menghilang] → [Cover story]             │
  │     [Tidak ada konsumsi baru] → [Internal consumption]  │
  │     [Success rate tinggi] → [Pil boost]                 │
  └─────────────────────────────────────────────────────────┘
```

#### Perbedaan Kritis vs LLM Biasa

| Aspek | LLM Biasa | Jin Soun's Method (Yang Kita Butuhkan) |
|-------|-----------|----------------------------------------|
| Input | Prompt + context window | Trigger + seluruh knowledge graph |
| Retrieval | Semantic similarity (vector) | Spreading activation (structural) |
| Cross-referencing | Tidak ada — context window saja | Multi-source structural comparison |
| Anomaly detection | Tidak ada — generate saja | Expected vs Observed → prediction error |
| Pattern completion | Next token prediction | Fragmen → pola utuh (compose) |
| Output | Probabilistic text | Traceable reasoning chain + confidence |

#### Rencana versi dasar

```
Pattern Completion Output = {
  1. TRIGGER: User input / observation → relate() aktifkan koneksi
  2. RECALL MASSAL: Spreading activation dari trigger node
     → Semua node yang terhubung secara struktural menjadi aktif
     → Ini SUDAH ada di RSVS via spreading.rs
  3. CROSS-REFERENCE: Bandingkan komposisi node-node yang aktif
     → Cari overlap temporal (tanggal), lokasi, aktor
     → structural_similarity() antar insiden
  4. ANOMALY DETECTION: appraise() → "expected ≠ observed"
     → Jika appraise return "disagree" → ada anomaly
     → Ini yang Jin Soun lakukan saat dia bilang "ini tidak masuk akal"
  5. PATTERN COMPLETION: compose() fragmen → pola utuh
     → Hubungkan fragmen yang tampak terpisah
     → Ini yang perlu dibangun — compose dari cross-domain fragments
  6. NARRATIVE OUTPUT: generate teks berdasarkan reasoning chain
     → Setiap klaim punya evidence node di graph
     → Confidence = grounding score dari node yang direferensi
     → Output = naratif yang bisa di-trace ke graph
}
```

---

## Arsitektur Dasar (General + Internet)

```
┌─────────────────────────────────────────────────────────────────────┐
│              PATTERN COMPLETION OUTPUT (Layer 5)                     │
│                                                                      │
│  Trigger → Recall Massal → Cross-Reference → Anomaly → Pattern →    │
│  Narrative                                                            │
│                                                                      │
│  "Diancang pair mencuri Snow Plum Pill menggunakan Ju Jangmok        │
│   sebagai cover. Inside job dari dalam Diancang. Pil dikonsumsi     │
│   internal."                                                         │
│                                                                      │
│  Evidence: [Hefei dates]→[Diancang mission]→[No external pill]→...  │
│  Confidence: 87% (grounding scores of referenced nodes)              │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  NARRATIVE GENERATOR                                       │      │
│  │  Input: activated nodes + pattern + anomaly                │      │
│  │  Output: traceable reasoning chain as text                 │      │
│  │  Method: compose() fragments → LLM generates from graph    │      │
│  └────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
         ↑
┌──────────────────────────────────────────────────────────────────────┐
│           PREDICTIVE CODING ENGINE (Layer 4)                         │
│                                                                      │
│   predict(X) → observe(Y) → belief_update(Δ)                        │
│                                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│   │  Prediction   │  │  Observation │  │  Belief Update          │   │
│   │  (compose     │  │  (ingest     │  │  (grounding +/−         │   │
│   │   expected)   │  │   reality)   │  │   reflection            │   │
│   │               │  │              │  │   confidence EMA)       │   │
│   └──────────────┘  └──────────────┘  └─────────────────────────┘   │
│                                                                      │
│   Anomaly = |predicted - observed| > threshold                       │
│   → Triggers pattern completion                                      │
└──────────────────────────────────────────────────────────────────────┘
         ↑
┌──────────────────────────────────────────────────────────────────────┐
│              RSVS CORE (Layer 3) — SUDAH ADA                         │
│                                                                      │
│   Graph · Senses · Compositions · Spreading Activation               │
│   Convergence · MCTS · Consolidation · Reflection                   │
│                                                                      │
│   ┌──────────────────┐  ┌──────────────────────────────────────┐    │
│   │ Knowledge Graph  │  │ Tiered Memory                        │    │
│   │ (atoms, senses,  │  │ New → Candidate → Stable → Dep      │    │
│   │  compositions)   │  │ EMA confidence, grounding           │    │
│   └──────────────────┘  └──────────────────────────────────────┘    │
│                                                                      │
│   ┌──────────────────┐  ┌──────────────────────────────────────┐    │
│   │ Spreading        │  │ Structural Analysis                   │    │
│   │ Activation       │  │ similarity + substitution + compose   │    │
│   │ (relate/recall)  │  │ (cross-reference + pattern)          │    │
│   └──────────────────┘  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
         ↑                    ↑
┌──────────────────────┐  ┌──────────────────────────────────────────┐
│  SITUATION LAYER     │  │  CONTEXT LAYER                            │
│  (Layer 2)           │  │  (Layer 1)                                │
│                      │  │                                           │
│  Chat History →      │  │  ┌───────────┐  ┌───────────────────┐   │
│  ingest ke graph     │  │  │ RSVS Graph│  │ Internet Plugin   │   │
│                      │  │  │ (internal │  │ (web_search API)  │   │
│  Event Stream →      │  │  │  knowledge│  │                   │   │
│  semantic index      │  │  └───────────┘  │ ingest hasil →    │   │
│                      │  │                 │ graph              │   │
│  Active Senses =     │  │  ┌───────────────────────────────────┐  │
│  situasi saat ini    │  │  │ SCOPE FILTER                      │  │
│                      │  │  │ "hanya jawab dari sumber X"       │  │
│                      │  │  │ = grounding gate yang diperluas    │  │
│                      │  │  │ trust: seed=1.0, user=0.65, ...   │  │
│                      │  │  └───────────────────────────────────┘  │
└──────────────────────┘  └──────────────────────────────────────────┘
```

---

## Mapping Lengkap: Novel → Arsitektur

| Aspek Jin Soun | Komponen Sistem | Fungsi RSVS | Status |
|---|---|---|---|
| Mengingat SEMUA | Knowledge Graph | atoms, senses, compositions | ✅ Ada |
| Simhyeon Pavilion | Context Layer | scoped knowledge + scope filter | 🔧 Extend |
| 30 tahun ingatan | Tiered Memory | Event Stream + tiered lifecycle | ✅ Ada |
| Mengenali teknik dari cues minimal | Spreading Activation | `relate()` + `convergence` | ✅ Ada |
| Tahu kelemahan semua lawan | Substitution Analysis | `substitution_analysis()` | ✅ Ada |
| Cross-reference 3 departemen | Multi-source Recall | `spreading` across domains | 🔧 Extend |
| "Ini tidak masuk akal" | Anomaly Detection | `appraise()` expected ≠ observed | 🔧 Extend |
| "Jadi begitu triknya!" | Pattern Completion | `compose()` fragments → pola | 🔧 Build |
| Prediksi gerakan lawan | Predictive Coding | belief update loop | 🔧 Build |
| "Aku predict X, ternyata Y" | Belief Update | grounding +/−, reflection | ✅ Sebagian |
| Mengajarkan ke orang lain | Narrative Output | reasoning chain + traceability | 🔧 Build |
| Batasan: tubuh third-rate | Transformer execution | LLM generates from graph | ⏳ Nanti |
| Context-dependent recall | SessionGraph | `context_query()` | ✅ Ada |
| Metode Jegal Cheon | Paradigm Router | structured analysis framework | ✅ Ada |

---

## Flow Data: Dari Input ke Output

```
USER INPUT: "Siapa yang mencuri Snow Plum Pill?"
     │
     ▼
┌─────────────────────────────────────────────────┐
│ 1. CONTEXT LAYER                                 │
│    → Scope: "Hefei", "pencurian", "Snow Plum"   │
│    → Internet search: tambahkan info jika perlu  │
│    → Scope filter: hanya sumber terpercaya       │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ 2. SITUATION LAYER                               │
│    → Chat history: percakapan sebelumnya         │
│    → Active senses: Hefei, guild, pil            │
│    → Event stream: apa yang baru terjadi         │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ 3. RSVS CORE                                     │
│    → relate("snow_plum_pill") → spreading        │
│      activation ke semua node terkait            │
│    → Nodes aktif: Hefei, Ju Jangmok,            │
│      Diancang, tanggal-tanggal, guild            │
│    → structural_similarity() antar insiden       │
│    → substitution_analysis() antar tersangka     │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ 4. PREDICTIVE CODING ENGINE                      │
│    → Predict: "Ju Jangmok = pencuri" (obvious)   │
│    → Observe: "Tidak ada konsumsi pil baru"      │
│    → Delta: PREDICTION ERROR → anomaly!          │
│    → Update: REVISE belief tentang Ju Jangmok    │
│    → Re-predict: "Ju Jangmok = cover"            │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ 5. PATTERN COMPLETION OUTPUT                     │
│    → Trigger: anomaly detected                   │
│    → Recall massal: semua node aktif             │
│    → Cross-reference: tanggal, lokasi, aktor     │
│    → Pattern: Diancang inside job                │
│    → Compose narrative:                          │
│      "Gu Ilmu & Jang Hangi (Diancang Five        │
│       Swords) mencuri Snow Plum Pill.            │
│       Ju Jangmok digunakan sebagai cover.        │
│       Operasi direncanakan dari dalam Diancang." │
│    → Evidence chain: [nodes + edges]             │
│    → Confidence: 87%                             │
└─────────────────────────────────────────────────┘
```

---

## Prioritas Development (Versi Dasar: General + Internet)

### Phase 1: Foundation — Wiring yang Sudah Ada
1. **RSVS Core API** → pastikan semua fungsi ter-expose ke Python
2. **Event Stream** → connect ke chat interface
3. **Spreading Activation** → gunakan untuk recall dari trigger
4. **Web Search → Ingest Pipeline** → hasil search masuk ke graph

### Phase 2: Context Layer (Internet Plugin)
1. **Web Search Integration**: `web_search()` → hasil → `ingest()` ke RSVS
2. **Scope Filter**: mekanisme untuk batasi "hanya jawab dari sumber X"
3. **Source Trust Scoring**: `SOURCE_TRUST` sudah ada di config.py, perlu wiring

### Phase 3: Situation Layer (Chat History)
1. **Chat Ingest**: setiap percakapan di-ingest ke graph
2. **Semantic Recall**: query ke graph untuk temukan konteks relevan
3. **Active Sense Tracking**: sense yang aktif = situasi saat ini

### Phase 4: Predictive Coding Engine
1. **Prediction API**: `predict_composition(concept, context)` → return expected compositions
2. **Observation Ingest**: realita masuk, bandingkan dengan prediksi
3. **Belief Update Loop**: grounding confirmation/contradiction → confidence update
4. **Anomaly Detection**: ketika |predicted - observed| > threshold → trigger pattern completion

### Phase 5: Pattern Completion Output
1. **Recall Pipeline**: trigger → relate() → spreading activation → semua node aktif
2. **Cross-Reference Engine**: bandingkan node aktif secara structural (tanggal, lokasi, aktor)
3. **Pattern Compose**: compose() dari fragmen terpisah → pola utuh
4. **Narrative Generator**: LLM generate dari reasoning chain RSVS (bukan dari kosong)
5. **Traceability**: setiap klaim punya evidence node, setiap confidence punya grounding score

---

## Catatan Penting

1. **Tidak ada yang di-edit di folder RSVS** — semua yang sudah ada tetap utuh
2. **Folder workspace** = tempat kita bangun layer-layer di atas RSVS
3. **Versi coder + rule-based** = Phase berikutnya, setelah general + internet stabil
4. **Text Output BUKAN sekadar deductive reasoning** — ini **pattern completion across disparate memories**, persis seperti Jin Soun menghubungkan pencurian Snow Plum Pill → Ju Jangmok → Diancang Five Swords → inside job
5. **Analogi Jin Soun sangat kuat**: sistem ini = karakter yang mengingat segalanya, memahami relasi, memprediksi, dan mengeluarkan kesimpulan yang bisa diaudit. Kelemahannya (seperti Jin Soun) = eksekusi. Itulah kenapa transformer tetap diperlukan sebagai execution layer untuk generate naratif akhir.
6. **Kunci inovasi**: LLM generate teks DARI graph, bukan dari kosong. Graph = structural memory, LLM = narrative voice. Jin Soun = graph, tubuhnya = LLM yang terbatas.
