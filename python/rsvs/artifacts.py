"""RSVS Bridge artifact I/O — file persistence for frontend/agents."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CONFIG, iso_now, make_id
from .validation import _validate_snapshot_contract


__all__ = [
    "_latest_file",
    "_write_json",
    "_write_events_jsonl",
    "_write_ingest_artifacts",
    "_read_latest_ingest_bundle",
]


# ---------------------------------------------------------------------------
# Low-level file helpers
# ---------------------------------------------------------------------------


def _latest_file(pattern: str) -> Path | None:
    """Return the most recently modified file matching *pattern* in the atom dir."""
    if not CONFIG.atom_dir.exists():
        return None
    files = sorted(
        CONFIG.atom_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_events_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    """Write a list of event dicts as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for evt in events:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# High-level artifact I/O
# ---------------------------------------------------------------------------


def _write_ingest_artifacts(
    snapshot: dict[str, Any],
    events: list[dict[str, Any]],
    stats: dict[str, Any],
) -> dict[str, str]:
    """Write snapshot, events, and report artifacts and return their paths."""
    CONFIG.atom_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    snapshot_path = CONFIG.atom_dir / f"snapshot-{stamp}.json"
    events_path = CONFIG.atom_dir / f"events-{stamp}.jsonl"
    report_path = CONFIG.atom_dir / f"report-{stamp}.json"

    _write_json(snapshot_path, snapshot)
    _write_events_jsonl(events_path, events)
    _write_json(report_path, {**stats, "created_at": iso_now(), "mode": "ingest"})

    return {
        "snapshot": str(snapshot_path),
        "events": str(events_path),
        "report": str(report_path),
    }


def _read_latest_ingest_bundle() -> dict[str, Any] | None:
    """Read the latest ingest artifact bundle (snapshot + events + report)."""
    if not CONFIG.atom_dir.exists():
        return None

    snapshot_path = _latest_file("snapshot-*.json")
    if snapshot_path is None:
        return None

    events_path = _latest_file("events-*.jsonl")
    report_path = _latest_file("report-*.json")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    # Validate only if schema_version present; old snapshots may lack it
    if snapshot.get("schema_version") is not None:
        _validate_snapshot_contract(snapshot)

    events: list[dict[str, Any]] = []
    if events_path and events_path.exists():
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

    report = None
    if report_path and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    messages: list[dict[str, Any]] = []
    if report:
        messages.append(
            {
                "id": make_id("msg"),
                "type": "system_ingest_status",
                "content": (
                    f"Restored latest batch {report.get('batch_id', '-')}: "
                    f"{report.get('node_count', 0)} nodes, "
                    f"{report.get('edge_count', 0)} edges."
                ),
                "timestamp": iso_now(),
            }
        )

    return {
        "snapshot": snapshot,
        "events": events,
        "messages": messages,
        "files": {
            "snapshot": str(snapshot_path),
            "events": str(events_path) if events_path else None,
            "report": str(report_path) if report_path else None,
        },
    }
