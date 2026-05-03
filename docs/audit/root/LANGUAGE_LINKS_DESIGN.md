# RSVS Language Links Design (Reconstructed Final)

## Context
Model aktif dikembalikan ke format lama:
- `surface_label` menjadi identity surface-locale: `<surface>@<lang>`
- relasi lintas bahasa utama memakai `language_links.same_as`
- `sense_state` opsional/sekunder, bukan owner utama konteks bahasa

Contoh target:

```json
"surface_label": "air@id",
"language_links": [
  { "type": "same_as", "target_id": 2001 }
]
```

## Core Contract
### Input
- `raw_text`
- `candidate_surface`
- `language_code` (ISO-like, contoh: `id`, `en`)
- `candidate_target_id` (opsional, untuk same_as)
- `equivalence_score` (0.0-1.0)

### Output
- `surface_label`: `<normalized_surface>@<language_code>`
- `language_links[]`: relasi lintas bahasa
  - `type`: `same_as`
  - `target_id`: int

### Fail-fast Errors
- `invalid_language_code`
- `surface_label_collision`
- `target_not_found`
- `self_link_forbidden`
- `equivalence_threshold_not_met`
- `schema_model_mismatch_sense_centric`

## Building Rules
### 1) Build `surface_label`
1. normalize surface (lowercase, trim, collapse spaces)
2. validate `language_code`
3. build: `surface_label = "<surface>@<lang>"`

Examples:
- `Air` + `id` => `air@id`
- `Air` + `en` => `air@en`

### 2) Build `language_links.same_as`
`same_as` hanya dibuat jika:
- `equivalence_score >= 0.90`
- `target_id` ada
- `target_id != current_id`
- target bukan model sense-centric

Jika score 0.75-0.89:
- simpan sebagai candidate relation (belum `same_as` final)

## Decision Table
| Condition | Action |
|---|---|
| score >= 0.90 + target valid + not self | create `same_as` |
| score 0.75-0.89 | keep candidate relation |
| score < 0.75 | reject relation |
| collision `surface@lang` di node/sense sama | reject |
| payload/artifact sense-centric terdeteksi | hard reject |

## Hard Break Policy
- Reader/writer hanya menerima model `surface@lang + same_as`.
- Model sense-centric (`sense_state.semantic_index`, `senses[].layer_1/layer_2`) ditolak eksplisit.
- Tidak ada dual-read fallback.

## Risks and Mitigation
1. False equivalence merge
- Mitigasi: threshold tinggi (0.90) + candidate stage.

2. Collision label
- Mitigasi: unique check pada `surface@lang`.

3. Graph drift relasi bahasa
- Mitigasi: canonical writer + dedup relation guard.

4. Regression dari model sense-layer lama
- Mitigasi: hard reject dengan error eksplisit.

## Verification Matrix
### policy
- `surface_label` selalu terbentuk `<surface>@<lang>`.
- `same_as` gagal jika di bawah threshold.

### loop
- ingest bilingual menghasilkan node locale terpisah + relation lintas bahasa.

### integrity
- tidak ada self-link, duplicate same_as, dangling target.
- model sense-centric ditolak deterministik.
