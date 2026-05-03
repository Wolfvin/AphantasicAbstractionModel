# RSVS Gap Audit V1

Date: 2026-04-22

## Missing / Open Gaps

1. No dedicated tests for new mode API (`POST /run`)
- Belum ada test otomatis untuk kontrak mode v1:
  - valid `ingest|appraise|relate`
  - `invalid_mode`
  - `text_required`
  - `GET /latest?mode=...`
  - artifact creation per mode

2. No tests for CLI mode commands
- `cli/rsvs_agent_cli.py` belum punya test harness untuk:
  - `run --mode ingest|appraise|relate`
  - `latest --mode ...`
  - `atom-show appraise|relate`
  - alias kompatibilitas `ingest`

3. Relate mode can return empty result too easily
- Pada smoke check, `relate` sempat menghasilkan:
  - `nodes=0`, `edges=0`
- Tidak broken, tapi UX bisa terasa “noisy/flat”.
- Kandidat perbaikan: fallback retrieval (mis. top nodes by confidence) saat overlap token = 0.

4. Frontend appraise/relate output still summary-only
- UI sudah mode-aware dan slash-aware.
- Tetapi render hasil mode baru masih ringkas (status line), belum structured panel:
  - Appraise: support/conflict evidence list
  - Relate: ranked nodes/edges list

5. Legacy compatibility endpoints still present (intentional)
- Endpoint lama masih ada untuk compatibility:
  - `POST /ingest`
  - `GET /latest` default shape legacy
- Ini technical debt yang bisa dihapus setelah transisi frontend benar-benar stabil.

6. Frontend and bridge docs belum expose `view=compact|detail` end-to-end
- Runtime bridge sudah support `view`.
- UI adapter dan docs mode contract belum sepenuhnya sinkron untuk konsumsi detail view.

## What Is Already Solid

1. Unified mode API works
- `POST /run` aktif untuk 3 mode (`ingest`, `appraise`, `relate`).

2. CLI works with mode flow
- `run --mode ...`, `latest --mode ...`, `atom-show appraise|relate` sudah jalan.

3. Frontend mode workflow works
- Input composer sudah support mode selector + slash command.
- Frontend ingest path sudah pindah ke `/run` mode `ingest`.

4. Docs baseline exists
- Mode contract + run guide tersedia.

5. V4.2 hard-break + single node core sudah aktif di bridge runtime
- `schema_version=v4.2` ditulis pada artifact mode.
- `kind=composite` dan payload tanpa schema sesuai ditolak (`schema_version_mismatch`).
- `semantic.compression_state` + provenance diwajibkan pada node contract.

## Hardening Backlog (Recommended Next)

1. Backend contract tests for `/run`
- Tambah test mode-level contract + error codes + artifact checks.

2. CLI smoke tests
- Tambah script/test untuk command utama mode v1.

3. Rich frontend rendering for mode results
- Appraise evidence cards + Relate result list.

4. Controlled legacy cleanup phase
- Setelah frontend stabil, deprecate endpoint lama (`/ingest`, legacy `/latest`).

## JSON Schema Hardening Gaps (V4.1)

1. `missing_atom_ids` belum final
- Definisi belum dikunci (node-local unknown vs globally-expected-missing).

2. Belum ada `schema_version` per payload
- Migrasi lintas versi berisiko tanpa version tag eksplisit.

3. `language_links` contract belum ketat
- Enum relasi + validasi arah/target belum dikunci.

4. Constraint `atom_refs` belum lengkap
- Perlu guard duplikat `atom_id`, range `weight`, dan rule `weight=0.0 => negated=true`.

5. Invariant `sense_state` belum dipaksa
- `active_sense_id` harus valid dan metrik sense harus tervalidasi.

6. Redundansi tier
- `tier` di root dan `autonomy.tier` berpotensi double source-of-truth.

7. `metrics` belum punya definisi window
- `freq`/`domain_diversity` belum jelas basis waktunya.

8. `provenance` belum cukup untuk replay audit
- Perlu `ingest_run_id` dan `generator_version`.

9. `render` masih bercampur domain
- Field UI sebaiknya optional/terpisah dari core semantic node.

10. Belum ada policy migrasi schema
- Belum ada dokumen transform/backward-compat antar versi schema.

## JSON Schema Hardening Status (V4.2 Update)

Done:
1. `schema_version` per payload sudah diterapkan di bridge artifacts.
2. Hard break legacy schema diaktifkan (`schema_version_mismatch`).
3. Model ontologi runtime diganti ke single node (`kind=node`) + semantic compression invariants.
4. Runtime validator sudah menolak:
- self-reference pada `derived_from_node_ids`
- duplikat `derived_from_node_ids`
- referensi id turunan yang tidak ada
- compressed node tanpa provenance wajib

Remaining:
1. Definisi final `missing_atom_ids` masih perlu dikunci lintas dokumen.
2. Kontrak `language_links` masih perlu enum + validasi arah/target formal.
3. Rule `sense_state` lintas seluruh producer belum tervalidasi penuh.
4. Kontrak output frontend untuk `detail_view` belum sepenuhnya diadopsi UI.
