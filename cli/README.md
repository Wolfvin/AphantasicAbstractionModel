# RSVS CLI for Agents

Location: `/home/raymond/workspace/projets/skills_and_mcp/RSVS/cli`

CLI ini dibuat untuk agent/operator agar bisa pakai RSVS tanpa buka UI.

## Quick Start

```bash
cd /home/raymond/workspace/projets/skills_and_mcp/RSVS/backend
make bridge-run
```

Di terminal lain:

```bash
cd /home/raymond/workspace/projets/skills_and_mcp/RSVS/cli
python3 rsvs_agent_cli.py health
python3 rsvs_agent_cli.py run --mode ingest --text "water flows through porous stone" --summary
python3 rsvs_agent_cli.py run --mode appraise --text "water is solid" --summary
python3 rsvs_agent_cli.py run --mode relate --text "water stone" --top-k 10 --view detail --summary
python3 rsvs_agent_cli.py latest --mode relate --view detail --summary
python3 rsvs_agent_cli.py atom-ls
python3 rsvs_agent_cli.py atom-show appraise
```

## Commands

- `health`
  - Cek koneksi ke backend bridge (`/health`).

- `run`
  - Jalankan mode utama ke endpoint `/run`.
  - Opsi:
    - `--mode ingest|appraise|relate`
    - `--text "..."`
    - `--file /path/input.txt`
    - `--correlation-id custom_id`
    - `--top-k 10` (untuk mode `relate`)
    - `--view compact|detail` (default `compact`)
    - `--summary`

- `ingest`
  - Alias kompatibilitas untuk `run --mode ingest`.

- `latest`
  - Ambil latest output.
  - Opsi:
    - `--mode ingest|appraise|relate`
    - `--view compact|detail` (default `compact`)
    - `--summary`

- `atom-ls`
  - List artefak terbaru di folder `atom/`.
  - Opsi:
    - `--limit 5`

- `atom-show`
  - Tampilkan konten artefak terbaru.
  - `atom-show snapshot`
  - `atom-show events --tail 50`
  - `atom-show report`
  - `atom-show appraise`
  - `atom-show relate`

## Expected Artifact Files
- `snapshot-*.json`
- `events-*.jsonl`
- `report-*.json`
- `appraise-*.json`
- `relate-*.json`

## Environment
- `RSVS_BRIDGE_URL` (default `http://127.0.0.1:8787`)
- `RSVS_ATOM_DIR` (default `/home/raymond/workspace/projets/skills_and_mcp/RSVS/atom`)

## Additional Spec
See:
`/home/raymond/workspace/projets/skills_and_mcp/RSVS/docs/audit/root/MODES_SPEC_V1.md`
