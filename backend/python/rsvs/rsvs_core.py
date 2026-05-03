"""RSVS Rust core singleton management — thread-safe."""

from __future__ import annotations

import threading
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


# ---------------------------------------------------------------------------
# Thread-safe singleton state
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_instance: Any | None = None
_last_ingest_seq: int = 0  # track seq for event consumption


def _create_instance() -> Any:
    """Create or load an Rsvs instance. Must be called with _lock held."""
    if not _RSVS_AVAILABLE:
        return None

    # Try to load from saved state
    if RSVS_STATE_PATH.exists():
        try:
            inst = _Rsvs.load(str(RSVS_STATE_PATH))  # type: ignore[union-attr]
            global _last_ingest_seq
            _last_ingest_seq = inst.latest_seq_v1()
            print(f"[bridge] Loaded Rsvs state from {RSVS_STATE_PATH} "
                  f"(seq={_last_ingest_seq})")
            return inst
        except Exception as exc:
            print(f"[bridge] WARNING: Failed to load Rsvs state: {exc}")
            print("[bridge] Creating fresh Rsvs instance instead.")

    # Fresh instance
    _last_ingest_seq = 0
    print("[bridge] Created fresh Rsvs instance")
    return _Rsvs()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "get_rsvs_instance",
    "is_rust_core_available",
    "require_rust_core",
    "_get_rsvs",
    "_save_rsvs",
    "_get_last_ingest_seq",
    "_set_last_ingest_seq",
]


def is_rust_core_available() -> bool:
    """Return True if the Rust core native module is importable."""
    return _RSVS_AVAILABLE


def get_rsvs_instance() -> Any:
    """Return the singleton Rsvs instance, creating or loading as needed.

    Thread-safe double-checked locking pattern.
    Raises RustCoreUnavailableError if the Rust core is not available.
    """
    global _instance

    if _instance is not None:
        return _instance

    with _lock:
        if _instance is None:
            _instance = _create_instance()
        if _instance is None:
            raise RustCoreUnavailableError(
                "Rust core is not available. Build with `maturin develop`."
            )
        return _instance


def _get_rsvs() -> Any:
    """Return the singleton Rsvs instance, or None if unavailable.

    Backward-compatible accessor used by bridge_server.py.
    """
    global _instance

    if _instance is not None:
        return _instance

    if not _RSVS_AVAILABLE:
        return None

    with _lock:
        if _instance is None:
            _instance = _create_instance()
        return _instance


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
    return get_rsvs_instance()
