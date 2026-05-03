# RSVS v4.2 Latest Confirmation (Single Node + Semantic Compression)

Date: 2026-04-22
Source checked:
- `backend/docs/skills/rsvs-v4.skill`
- extracted refs: `context.md`, `notation.md`, `schema.md`, `SKILL.md`

## Current Implementation Status (Latest)

Status: `implemented` and `verified` (bridge runtime lane).

Implemented:
1. Runtime hard-break schema
- `schema_version=v4.2` enforced.
- Legacy payload (`kind=composite` / old schema) rejected with `schema_version_mismatch`.

2. Single node model in bridge
- `kind=node` enforced.
- Semantic compression contract validated (`compression_state`, `derived_from_node_ids`, `compression_reason`).

3. Policy engine governance in ingest path
- Seed bootstrap rule active (24 seed nodes):
  - `is_seed=true`
  - `is_locked=true`
  - `tier=1`
  - `confidence=1.0`
  - `status=stable`
- Non-seed lifecycle active with hysteresis:
  - promote threshold `>= 0.75`
  - demote threshold `< 0.60`
- Dedup replay guard active (fingerprint-based) to prevent score inflation.
- Quarantine guard active when status flips exceed budget.

4. Mode output contract alignment
- `relate/appraise` output now carries governance fields (`status`, `confidence`, `is_seed`, `is_locked`) through projection.
- `view=compact|detail` preserved.

Verification evidence:
1. Targeted V4.2 test suite:
- `python/tests/test_bridge_v42.py` => `6 passed in 0.14s`

2. Manual smoke run:
- ingest -> ingest ulang -> relate(detail) path passed.
- Seed invariants observed as expected.

Notes:
- Verification above is for Python bridge runtime scope.
- Full backend matrix (all python/rust test suites) remains a separate run.

## Confirmed Core Rules (Latest)

1. Integer-first representation
- ID is integer (`int32`) as primary identity.
- String/label mapping is output/helper layer only.

2. Language handling
- Language identity in active model uses `surface_label=<surface>@<lang>`.
- Multilingual semantics remains pointer-based (`batu@id : same_as stone@en`) with strict equivalence gating.
- `@lang` suffix is part of primary surface-locale identity.
- Language labels are AI-assigned from data, not hardcoded in definition.

3. Single node semantics
- Semua entitas semantik menggunakan model tunggal `node`.
- Tidak ada tipe ontologis terpisah `composite`.
- Kompresi dinyatakan sebagai metadata semantik (`compression_state` + provenance), bukan jenis node.
- Negation vs missing tetap dipisahkan tegas.

4. Seed policy
- 24 seed atoms (Layer 0 + Layer 1) are mandatory at startup.
- Seed atoms have confidence 1.0, Tier 1, and cannot be removed.
- Layer 2+ atoms emerge from data.

5. Compression + sense policy
- `compression_state`:
  - `raw`: node dasar hasil sinyal ingest.
  - `compressed`: node ringkas yang merefer ke node asal.
- `derived_from_node_ids` wajib valid untuk `compressed`.
- `sense_state` tetap data-formed dan tidak hardcoded.

6. Attention + scoring
- RSVS Attention uses hard selection (sparse) with:
  - `score = α*NPMI + β*Jaccard + γ*cooc`
- NPMI (bounded) is required, not PMI.

7. Tiered autonomy
- Tier 1: autonomous
- Tier 2: flagged/candidate (revocable)
- Tier 3: blocked (human decision)
- Confidence update formula is explicitly defined in latest context.

## Corrections Locked (Post Senior Review)

1. Language wire format is no longer open
- Final direction: use `surface@lang` + pointer relation (`same_as`).
- Not using "reserved atom pattern" for language semantics.

2. Negation vs missing must both be explicit
- Keep both semantics:
  - explicit negation: `weight: 0.0` + `negated: true`
  - missing/unclassified: outside `atom_refs` (or dedicated missing field)
- Do not allow `weight: 0.0` without `negated: true`.

3. Seed lock enforcement is both runtime and persisted
- Runtime rule + persisted fields are both required.
- Keep `is_seed`, `is_locked`, and non-revocable autonomy signals.

4. Sense-aware weighting is required
- Add sense-scoped weighting or sense-level atom weights.
- `sense_state` must be populated (not always empty).

5. Language framing correction
- Language context is encoded in `surface_label` suffix (`@lang`).
- Pointer mapping (`same_as`) remains for cross-language equivalence.
- `sense_state` is optional/secondary in this reconstructed model.

6. Hard break schema policy
- Payload lama (`schema_version` tidak sesuai atau `kind=composite`) ditolak.
- Writer hanya menulis schema `v4.2`.
- Reader mengembalikan error `schema_version_mismatch` untuk artefak lama.

## JSON/Schema Implications (Must Follow)

1. Use pointer-based multilingual links
- Example relation: `same_as`.
- Collision example via `surface_label`: `air@id`, `air@en`.

2. Distinguish explicit negation vs unknown
- Explicit negation in `atom_refs` with paired fields.
- Missing/unclassified represented separately from positive refs.

3. Single node + compression invariants
- `kind` harus `node` (jika field dipertahankan).
- `semantic.compression_state` enum: `raw | compressed`.
- Jika `compressed`, wajib:
  - `semantic.derived_from_node_ids` non-empty.
  - `semantic.compression_reason` non-empty.
- `derived_from_node_ids` tidak boleh self-id, duplikat, atau referensi node yang tidak ada.

4. Persist and enforce seed lock
- Persisted fields + runtime checks must both exist.

5. Keep mode outputs ID/edge-grounded
- Appraise/Relate evidence should refer to node IDs/relations, not summary text only.

## Updated JSON Example (v4.2-aligned)

```json
{
  "schema_version": "v4.2",
  "id": 1001,
  "kind": "node",
  "tier": 1,
  "confidence": 1.0,
  "status": "stable",
  "is_seed": true,
  "is_locked": true,

  "surface_label": "air@id",
  "language_links": [
    { "type": "same_as", "target_id": 2001 }
  ],

  "semantic": {
    "compression_state": "compressed",
    "derived_from_node_ids": [1002, 1003],
    "compression_reason": "merged recurring wetness-flow pattern",
    "atom_refs": [
      { "atom_id": 11, "weight": 1.0 },
      { "atom_id": 24, "weight": 0.8, "sense_scoped": true },
      { "atom_id": 31, "weight": 0.0, "negated": true }
    ],
    "missing_atom_ids": [45, 62]
  },

  "sense_state": {
    "active_sense_id": 1,
    "senses": [
      {
        "sense_id": 1,
        "core_atom_ids": [11, 24],
        "coherence": 0.92,
        "n_contexts": 8,
        "status": "mature"
      },
      {
        "sense_id": 2,
        "core_atom_ids": [31],
        "coherence": 0.81,
        "n_contexts": 5,
        "status": "mature"
      }
    ]
  },

  "metrics": {
    "freq": 18,
    "coherence": 0.81,
    "domain_diversity": 3,
    "last_updated_at": "2026-04-22T09:12:00Z"
  },

  "autonomy": {
    "tier": 1,
    "reason": "seed atom",
    "revocable": false
  },

  "render": {
    "position": { "x": 1.2, "y": -0.7, "z": 3.4 },
    "size": 1.1,
    "color": "#69F0AE",
    "glow": 0.72
  },

  "provenance": {
    "source_batch_id": "seed_bootstrap",
    "source_domain": "core_seed",
    "source_type": "bootstrap"
  }
}
```
