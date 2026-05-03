"""RSVS Bridge configuration and shared constants."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

API_VERSION = "v1"
SCHEMA_VERSION = "v4.2"

# ---------------------------------------------------------------------------
# Mode / domain constants
# ---------------------------------------------------------------------------

VALID_MODES = {"ingest", "appraise", "relate"}

SOURCE_TRUST = {
    "trusted_seed": 1.0,
    "governance_manual": 0.95,
    "verified_runtime": 0.8,
    "user_input": 0.65,
    "unknown_external": 0.4,
}

SEED_LABELS = (
    "exists", "entity", "relation", "state", "change", "time", "space",
    "cause", "effect", "context", "signal", "pattern", "memory",
    "attention", "value", "agent", "goal", "risk", "trust", "identity",
    "language", "meaning", "action", "feedback",
)

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

PROMOTION_THRESHOLD = 0.75
DEMOTION_THRESHOLD = 0.60
QUARANTINE_FLIP_BUDGET = 3
MAX_CONFIDENCE_DELTA = 0.12
SHORT_WINDOW_BATCH = 3
LONG_WINDOW_BATCH = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def iso_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def make_id(prefix: str) -> str:
    """Generate a unique identifier with the given prefix."""
    return f"{prefix}_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{random.randint(1000, 9999)}"


# ---------------------------------------------------------------------------
# Bridge configuration
# ---------------------------------------------------------------------------


@dataclass
class BridgeConfig:
    host: str
    port: int
    atom_dir: Path


CONFIG = BridgeConfig(
    host=os.environ.get("RSVS_BRIDGE_HOST", "127.0.0.1"),
    port=int(os.environ.get("RSVS_BRIDGE_PORT", "8000")),
    atom_dir=Path(
        os.environ.get(
            "RSVS_ATOM_OUTPUT_DIR",
            "./atom",
        )
    ),
)

# Path for the Rust core's full-state persistence file
RSVS_STATE_PATH = CONFIG.atom_dir / "rsvs-state.json"
