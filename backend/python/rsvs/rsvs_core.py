"""RSVS Rust core singleton management."""

from __future__ import annotations

from typing import Any

from .config import CONFIG, RSVS_STATE_PATH
from .exceptions import RustCoreUnavailableError


# ---------------------------------------------------------------------------
# Rust core import & availability flag
# ---------------------------------------------------------------------------

_RSVS_AVAILABLE = False
_Rsvs = None  # type: ignore[assignment]

try:
    from rsvs import Rsvs as _RsvsClass  # type: ignore[import]
    _Rsvs = _RsvsClass
    _RSVS_AVAILABLE = True
except Exception:
    _RSVS_AVAILABLE = False


# Module-level singleton state
_rsvs_instance: Any | None = None
_last_ingest_seq: int = 0  # track seq for event consumption


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_rust_core_available() -> bool:
    """Return True if the Rust core native module is importable."""
    return _RSVS_AVAILABLE


def _get_rsvs() -> Any:
    """Return the singleton Rsvs instance, creating or loading as needed.

    Returns None if the Rust core is not available.
    """
    global _rsvs_instance, _last_ingest_seq

    if _rsvs_instance is not None:
        return _rsvs_instance

    if not _RSVS_AVAILABLE:
        return None

    # Try to load from saved state
    if RSVS_STATE_PATH.exists():
        try:
            _rsvs_instance = _Rsvs.load(str(RSVS_STATE_PATH))
            _last_ingest_seq = _rsvs_instance.latest_seq_v1()
            print(f"[bridge] Loaded Rsvs state from {RSVS_STATE_PATH} "
                  f"(seq={_last_ingest_seq})")
            return _rsvs_instance
        except Exception as exc:
            print(f"[bridge] WARNING: Failed to load Rsvs state: {exc}")
            print("[bridge] Creating fresh Rsvs instance instead.")

    # Fresh instance
    _rsvs_instance = _Rsvs()
    _last_ingest_seq = 0
    print("[bridge] Created fresh Rsvs instance")
    return _rsvs_instance


def _save_rsvs() -> None:
    """Persist the Rsvs instance to disk."""
    r = _get_rsvs()
    if r is None:
        return
    try:
        CONFIG.atom_dir.mkdir(parents=True, exist_ok=True)
        r.save(str(RSVS_STATE_PATH))
    except Exception as exc:
        print(f"[bridge] WARNING: Failed to save Rsvs state: {exc}")


def _get_last_ingest_seq() -> int:
    """Return the last ingest sequence number."""
    return _last_ingest_seq


def _set_last_ingest_seq(seq: int) -> None:
    """Set the last ingest sequence number."""
    global _last_ingest_seq
    _last_ingest_seq = seq


def require_rust_core() -> Any:
    """Return the Rsvs instance or raise RustCoreUnavailableError."""
    r = _get_rsvs()
    if r is None:
        raise RustCoreUnavailableError(
            "Rust core is not available. Build with `maturin develop`."
        )
    return r
