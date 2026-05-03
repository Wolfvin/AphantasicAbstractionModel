# RSVS CLI for Agents

CLI tool for agents/operators to use RSVS without the UI.

## Quick Start

```bash
cd backend
python -m rsvs.fastapi_server
```

In another terminal:

```bash
cd cli
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
  - Check connection to backend bridge (`/health`).

- `run`
  - Execute a mode via the `/run` endpoint.
  - Options:
    - `--mode ingest|appraise|relate`
    - `--text "..."`
    - `--file /path/input.txt`
    - `--correlation-id custom_id`
    - `--top-k 10` (for `relate` mode)
    - `--view compact|detail` (default `compact`)
    - `--summary`

- `ingest`
  - Compatibility alias for `run --mode ingest`.

- `latest`
  - Retrieve latest output.
  - Options:
    - `--mode ingest|appraise|relate`
    - `--view compact|detail` (default `compact`)
    - `--summary`

- `atom-ls`
  - List recent artifacts in the `atom/` directory.
  - Options:
    - `--limit 5`

- `atom-show`
  - Display artifact content.
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
- `RSVS_BRIDGE_URL` (default `http://127.0.0.1:8000`)
- `RSVS_ATOM_DIR` (default `./atom`)
