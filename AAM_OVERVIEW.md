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

## Layer Architecture

```
AphantasicAbstractionModel/
│
├── layer0/   Perceptual Front-End
│             Raw input → PerceptualTuple[]
│             text · image · video · audio
│
├── layer1/   Abstraction Engine (RSVS Core — Rust)
│             PerceptualTuple[] → graph delta
│             atoms · senses · compositions · spreading activation
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
| layer1 RSVS Core | Stable v8.3 |
| layer2 Cognitive Runtime | Stable v0.6 |
| layer3/policy | Stable |
| layer3/coder | Stable |
| layer3/reasoning | Stub — in progress |
| pipeline.py AamPipeline | Stable |

## Origin

Previously: **SymbolicPuzzle3D** / **RSVS Genius**  
Renamed in v1.0.0-alpha to reflect the true cognitive model.
