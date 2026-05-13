"""
Persistence Utilities — Atomic save/load for the full pipeline state (P2-8)

Provides `save_pipeline_state()` and `load_pipeline_state()` that
serialize the cognitive state of all pipeline layers (SituationLayer,
PredictiveEngine, and optionally other layers) into a single JSON file
using atomic writes (write to temp file, then rename).

Schema versioning is included for forward compatibility — older saved
states can be loaded by newer code with best-effort restoration.

Analogi: Jin Soun menyimpan seluruh arsip Simhyeon Pavilion ke
gudang penyimpanan — jika paviliun terbakar, dia bisa memulihkan
semua catatan dari cadangan. Penyimpanan dilakukan secara atomik:
tulis dulu ke file sementara, lalu ganti file utama, sehingga
tidak ada data yang hilang jika terjadi kegagalan di tengah jalan.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version — bump this when the persistence format changes in a
# way that is NOT backward-compatible.  The loader will log a warning
# but still attempt a best-effort restore.
# ---------------------------------------------------------------------------

_PIPELINE_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Pipeline-level save / load
# ---------------------------------------------------------------------------

def save_pipeline_state(pipeline: Any, path: str) -> dict:
    """Save all layer states to a single JSON file using atomic writes.

    Collects serializable state from each pipeline layer that supports
    `save_to_dict()`.  Writes to a temporary file first, then renames
    to the target path — this ensures the target file is never in a
    partially-written state.

    The RSVS bridge / Rust core graph is NOT serialized here; only the
    Python-side cognitive state is persisted.

    Args:
        pipeline: An AamPipeline instance (or any object with `.situation`
            and `.predictive` attributes that implement `save_to_dict()`).
        path: Filesystem path for the output JSON file.

    Returns:
        A summary dict with stats about what was saved, including
        per-layer stats and overall success/failure.
    """
    overall_start = time.time()
    summary: dict = {
        "path": path,
        "schema_version": _PIPELINE_SCHEMA_VERSION,
        "layers": {},
        "success": False,
    }

    # Collect state from each layer
    state: dict = {
        "schema_version": _PIPELINE_SCHEMA_VERSION,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "layers": {},
    }

    # --- SituationLayer ---
    situation = getattr(pipeline, "situation", None)
    if situation is not None and hasattr(situation, "save_to_dict"):
        try:
            situation_state = situation.save_to_dict()
            state["layers"]["situation"] = situation_state
            summary["layers"]["situation"] = {
                "messages": len(situation_state.get("messages", [])),
                "active_senses": len(situation_state.get("active_senses", [])),
                "success": True,
            }
        except Exception as exc:
            summary["layers"]["situation"] = {"success": False, "error": str(exc)}
            logger.error("Failed to save SituationLayer state: %s", exc)
    else:
        summary["layers"]["situation"] = {"success": False, "error": "no save_to_dict"}

    # --- PredictiveEngine ---
    predictive = getattr(pipeline, "predictive", None)
    if predictive is not None and hasattr(predictive, "save_to_dict"):
        try:
            predictive_state = predictive.save_to_dict()
            state["layers"]["predictive"] = predictive_state
            summary["layers"]["predictive"] = {
                "predictions": len(predictive_state.get("predictions", [])),
                "belief_updates": len(predictive_state.get("belief_updates", [])),
                "anomalies": len(predictive_state.get("anomalies", [])),
                "success": True,
            }
        except Exception as exc:
            summary["layers"]["predictive"] = {"success": False, "error": str(exc)}
            logger.error("Failed to save PredictiveEngine state: %s", exc)
    else:
        summary["layers"]["predictive"] = {"success": False, "error": "no save_to_dict"}

    # --- ContextLayer (if it supports persistence) ---
    context = getattr(pipeline, "context", None)
    if context is not None and hasattr(context, "save_to_dict"):
        try:
            context_state = context.save_to_dict()
            state["layers"]["context"] = context_state
            summary["layers"]["context"] = {"success": True}
        except Exception as exc:
            summary["layers"]["context"] = {"success": False, "error": str(exc)}
            logger.debug("ContextLayer save_to_dict failed: %s", exc)

    # --- PatternOutput (if it supports persistence) ---
    pattern = getattr(pipeline, "pattern", None)
    if pattern is not None and hasattr(pattern, "save_to_dict"):
        try:
            pattern_state = pattern.save_to_dict()
            state["layers"]["pattern"] = pattern_state
            summary["layers"]["pattern"] = {"success": True}
        except Exception as exc:
            summary["layers"]["pattern"] = {"success": False, "error": str(exc)}
            logger.debug("PatternOutput save_to_dict failed: %s", exc)

    # --- Pipeline metadata ---
    # Capture pipeline-level metadata (not the full pipeline state)
    if hasattr(pipeline, "get_status"):
        try:
            state["pipeline_status"] = pipeline.get_status()
        except Exception:
            pass

    # --- Atomic write: temp file → rename ---
    try:
        # Ensure parent directory exists
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        # Write to a temp file in the same directory (same filesystem)
        dir_name = parent if parent else "."
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=".aam_persist_",
            dir=dir_name,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            # Atomic rename (POSIX) or best-effort on Windows
            os.replace(tmp_path, path)
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        summary["success"] = True
        summary["duration_s"] = round(time.time() - overall_start, 3)
        logger.info(
            "Pipeline state saved to %s (%.3fs)",
            path, time.time() - overall_start,
        )
    except (OSError, TypeError) as exc:
        summary["error"] = str(exc)
        summary["duration_s"] = round(time.time() - overall_start, 3)
        logger.error("Pipeline state save failed: %s", exc)

    return summary


def load_pipeline_state(pipeline: Any, path: str) -> dict:
    """Load all layer states from a JSON file produced by `save_pipeline_state`.

    Restores serializable state to each pipeline layer that supports
    `load_from_dict()`.  If the saved schema version differs from the
    current version, a warning is logged but restoration proceeds with
    best-effort semantics.

    The RSVS bridge / Rust core graph is NOT restored here; only the
    Python-side cognitive state is deserialized.

    Args:
        pipeline: An AamPipeline instance (or any object with `.situation`
            and `.predictive` attributes that implement `load_from_dict()`).
        path: Filesystem path to read the JSON file from.

    Returns:
        A summary dict with stats about what was loaded, including
        per-layer stats and overall success/failure.
    """
    overall_start = time.time()
    summary: dict = {
        "path": path,
        "layers": {},
        "success": False,
    }

    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        summary["error"] = str(exc)
        logger.error("Pipeline state load failed: %s", exc)
        return summary

    # Schema compatibility check
    saved_version = state.get("schema_version", "0.0")
    if saved_version != _PIPELINE_SCHEMA_VERSION:
        logger.warning(
            "Pipeline state schema version mismatch: saved=%s, current=%s. "
            "Proceeding with best-effort restore.",
            saved_version, _PIPELINE_SCHEMA_VERSION,
        )
    summary["schema_version"] = saved_version
    summary["saved_at"] = state.get("saved_at", "unknown")

    layers_data = state.get("layers", {})

    # --- SituationLayer ---
    situation = getattr(pipeline, "situation", None)
    if situation is not None and hasattr(situation, "load_from_dict"):
        situation_data = layers_data.get("situation")
        if situation_data is not None:
            try:
                situation.load_from_dict(situation_data)
                summary["layers"]["situation"] = {
                    "messages": len(situation._messages),
                    "active_senses": len(situation._active_senses),
                    "success": True,
                }
            except Exception as exc:
                summary["layers"]["situation"] = {"success": False, "error": str(exc)}
                logger.error("Failed to restore SituationLayer state: %s", exc)
        else:
            summary["layers"]["situation"] = {"success": False, "error": "no data in file"}
    else:
        summary["layers"]["situation"] = {"success": False, "error": "no load_from_dict"}

    # --- PredictiveEngine ---
    predictive = getattr(pipeline, "predictive", None)
    if predictive is not None and hasattr(predictive, "load_from_dict"):
        predictive_data = layers_data.get("predictive")
        if predictive_data is not None:
            try:
                predictive.load_from_dict(predictive_data)
                summary["layers"]["predictive"] = {
                    "predictions": len(predictive._predictions),
                    "belief_updates": len(predictive._belief_updates),
                    "anomalies": len(predictive._anomalies),
                    "success": True,
                }
            except Exception as exc:
                summary["layers"]["predictive"] = {"success": False, "error": str(exc)}
                logger.error("Failed to restore PredictiveEngine state: %s", exc)
        else:
            summary["layers"]["predictive"] = {"success": False, "error": "no data in file"}
    else:
        summary["layers"]["predictive"] = {"success": False, "error": "no load_from_dict"}

    # --- ContextLayer (if it supports persistence) ---
    context = getattr(pipeline, "context", None)
    if context is not None and hasattr(context, "load_from_dict"):
        context_data = layers_data.get("context")
        if context_data is not None:
            try:
                context.load_from_dict(context_data)
                summary["layers"]["context"] = {"success": True}
            except Exception as exc:
                summary["layers"]["context"] = {"success": False, "error": str(exc)}
                logger.debug("ContextLayer load_from_dict failed: %s", exc)

    # --- PatternOutput (if it supports persistence) ---
    pattern = getattr(pipeline, "pattern", None)
    if pattern is not None and hasattr(pattern, "load_from_dict"):
        pattern_data = layers_data.get("pattern")
        if pattern_data is not None:
            try:
                pattern.load_from_dict(pattern_data)
                summary["layers"]["pattern"] = {"success": True}
            except Exception as exc:
                summary["layers"]["pattern"] = {"success": False, "error": str(exc)}
                logger.debug("PatternOutput load_from_dict failed: %s", exc)

    summary["success"] = True
    summary["duration_s"] = round(time.time() - overall_start, 3)
    logger.info(
        "Pipeline state loaded from %s (%.3fs)",
        path, time.time() - overall_start,
    )

    return summary
