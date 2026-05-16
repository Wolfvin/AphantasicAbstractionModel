# AAM — Aphantasic Abstraction Model

> "Bukan AI yang menyimpan foto. AI yang memahami relasi."

## Filosofi

AAM terinspirasi dari **Aphantasia** — kondisi kognitif di mana seseorang tidak bisa
membentuk gambaran visual di pikiran, tapi tetap bisa berpikir, bernalar, dan memahami
dunia secara penuh. Mereka tidak menyimpan "foto" dari pengalaman — mereka menyimpan
*relasi*, *kategori*, dan *struktur makna*.

Cara kerja AAM:
- Input masuk → langsung diabstraksi ke relasi → raw data tidak tersimpan
- Yang tersimpan: relasi antar konsep, confidence, konteks
- Semakin sering konsep dilihat → graph makin kuat → storage tidak membesar

## 6 Unified Abstractions (v12.0)

v12.0 mengganti patchwork tipe yang overlapping dengan **6 abstraksi inti**:

| # | Abstraksi | Mengganti | Tujuan |
|---|-----------|-----------|--------|
| 1 | **SemanticAtom** | Token, EventFrame, HiddenMeaningCandidate | Primitive ingest universal |
| 2 | **Composition** | EventFrame, HiddenMeaningCandidate, Pattern, Hypothesis | Pengelompokan terstruktur universal |
| 3 | **LifecycleState + EpistemicState** | NodeStatus, CandidateStatus, BeliefState, GroundingVerdict | Dua sumbu status orthogonal |
| 4 | **SemanticEdge** | RelationType, EdgeSource, SemanticRole, ProvenanceSource | Triple bertipe tunggal |
| 5 | **Transform (DAG)** | Hardcoded pipeline stages | Graph transform deklaratif |
| 6 | **Seed Anchoring** | Source trust weight system | Confidence epistemik berbasis seed |

**SemanticAtom** menyatukan semua jalur ingest menjadi satu: token (sparse atom),
event frame (rich atom), hidden meaning (derived atom) — semua masuk melalui satu tipe.

**Composition** menyatukan semua pengelompokan terstruktur: Event, HiddenMeaning, Pattern,
Hypothesis, Situation — satu mekanisme pengelompokan dengan role-based members.

**LifecycleState + EpistemicState** mengganti 4 enum yang overlapping dengan 2 sumbu
independent: maturity struktural (New → Candidate → Stable → Deprecated) dan
confidence epistemik (Observed → Inferred → Grounded / Contradicted).

**SemanticEdge** menyatukan 4 sistem edge/classification menjadi satu triple:
(relation, role?, source) — WHAT kind, OPTIONAL role, WHERE from.

**Transform (DAG)** mengganti pipeline hardcoded dengan engine DAG deklaratif:
setiap transform mendeklarasikan input/output, engine merutekan otomatis.

**Seed Anchoring** mengganti source trust weight system dengan seed-driven confidence:
setiap Composition membawa `seed_scores` relatif terhadap 5 primitif epistemik
(Trust, Risk, Value, Goal, Identity).

## Closed Feedback Loop

v12.0 menutup loop deteksi-perbaikan yang sebelumnya terputus:

```
DetectGaps → SelectAcquisition → EnrichComposition / ReExtractFrame → GovernBeliefs re-evaluation
```

Ketika `DetectGaps` menemukan kelemahan (missing roles, low confidence, ambiguous tokens),
`SelectAcquisition` memilih strategi akuisisi (Remember first, Study second, Ask last).
Hasilnya bisa berupa `EnrichComposition` (menambah member ke Composition yang sudah ada)
atau `ReExtractFrame` (re-ekstraksi frame dengan konteks graph).
Setelah itu, `GovernBeliefs` melakukan re-evaluasi — composition bisa dipromosikan
setelah mendapat evidence baru.

Fitur kunci:
- `source_composition_id` di KnowledgeGap untuk traceability
- `RecallAction` menghasilkan aksi konkret (bukan hanya mode)
- `process_user_answer_merge()` menggabungkan jawaban user ke Composition yang sudah ada
- `ExtractionQualityTracker` mengidentifikasi aturan ekstraksi yang lemah
- `max_enrichment_rounds` membatasi loop per cognitive mode (0/1/2)

## 3 Cognitive Modes

v12.0 mendukung 3 mode kognitif yang dikendalikan oleh Executive Cognition Layer:

| Mode | Behavior | Enrichment Rounds |
|------|----------|-------------------|
| **Reactive** | Quick response, no gap detection | 0 |
| **Analytical** | Gap detection + enrichment loop | 1 |
| **Reflective** | Full loop: gaps + enrichment + weak frame re-extraction | 2 |

Mode dipilih secara otomatis berdasarkan `GraphNeighborhood` — kontradiksi local
memicu Reflective, query sederhana cukup Reactive.

## Layer Architecture

```
AphantasicAbstractionModel/
│
├── layer0/   Perceptual Front-End
│             Raw input → PerceptualTuple[]
│             text · image · video · audio
│
├── layer1/   Abstraction Engine (RSVS Core — Rust)
│   ├── v12/          Unified Abstraction Types & Pipeline Engine
│   │   ├── types.rs     SemanticAtom, Composition, LifecycleState,
│   │   │                 EpistemicState, SemanticEdge, SeedPrimitive,
│   │   │                 PipelineContext, Transform trait
│   │   ├── pipeline.rs   PipelineEngine (DAG-based), register_default_pipeline(),
│   │   │                 Graph, Tokenize, IngestAtoms, EnrichComposition,
│   │   │                 ReExtractFrame, topological_sort
│   │   └── mod.rs        Re-exports + documentation
│   │
│   └── (v8.3 types)  Node, Edge, CompositionRef — additive, not replaced
│
│   v12.0 Transforms (registered in DAG order):
│     Tokenize → ExtractFrame → ReasonFrame → IngestAtoms →
│     GovernBeliefs → SeedAnchor → DetectGaps → SelectAcquisition →
│     EnrichComposition / ReExtractFrame
│
├── layer2/   Cognitive Runtime
│             context        : scoped knowledge + internet
│             scope_control  : hierarchical scope management (domain/subdomain/topic)
│             chat_index     : semantic chat index — conversations as graph of meaning
│             situation      : chat history as semantic memory
│             predictive     : belief update + anomaly detection
│             prediction_loop: explicit predict/observe/update lifecycle (Friston)
│             pattern        : pattern completion + narrative
│
├── layer3/   Deductive Reasoning & Output
│             policy    : rule-based compliance (tax, regulation)
│             coder     : code as structured knowledge
│             reasoning : traceable deductive chain (stub)
│
└── pipeline.py  AamPipeline — wires all layers
```

## Analogi: Jin Soun

Dari novel *"The Martial Genius Who Remembers Everything"* — Jin Soun mengingat
segalanya bukan sebagai replay sensorik, tapi sebagai relasi struktural yang
bisa di-cross-reference instan dari 4 departemen berbeda.

**AAM = Jin Soun's memory system. LLM = Jin Soun's tubuh (execution layer).**

## Status

| Komponen | Status |
|----------|--------|
| layer0/text | TextAbstractor — functional (stub without LLM) |
| layer0/image | Stub — planned (CLIP/LLaVA) |
| layer0/video | Stub — planned |
| layer0/audio | Stub — planned (Whisper) |
| layer1 RSVS Core (v8.3) | Stable |
| layer1 v12.0 Types (SemanticAtom, Composition, etc.) | Implemented |
| layer1 v12.0 PipelineEngine (DAG) | Implemented (stubs for ExtractFrame, ReasonFrame, GovernBeliefs, SeedAnchor, DetectGaps, SelectAcquisition) |
| layer1 v12.0 Feedback Loop (EnrichComposition, ReExtractFrame) | Implemented |
| layer2 Cognitive Runtime | Stable v0.6 |
| layer3/policy | Stable |
| layer3/coder | Stable |
| layer3/reasoning | Stub — in progress |
| pipeline.py AamPipeline | Stable |

## Origin

Previously: AAM v12.0 — rule-based cognitive architecture
Renamed in v1.0.0-alpha to reflect the true cognitive model.
v12.0 architecture refactor: 6 Unified Abstractions + Closed Feedback Loop.
