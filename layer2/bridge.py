"""
RSVS Bridge — v12.0 DAG Pipeline adapter for PyO3 Rust core.

This module provides the SINGLE point of contact with the RSVS v12 core.
All layer2 modules should use this bridge instead of directly importing
from `rsvs` — this ensures consistent error handling, proper API adaptation,
and graceful fallback when the Rust core isn't built.

Architecture (v12.0):
    layer2 modules → V12PipelineBridge → PyV12Pipeline (PyO3) or fallback

The old v8.3 AbstractionBridge / RsvsBridge / _FallbackGraph have been
removed. The v12 DAG-based pipeline is now the ONLY architecture.

Key design decisions:
1. The bridge wraps PyO3 objects in plain Python dicts/lists so
   downstream code never needs to handle PyO3 objects directly.
2. All methods return Optional[T] — None means "not available" or
   "concept not found", never raises.
3. Fallback mode provides a lightweight in-memory graph for testing
   and development without needing to build the Rust core.
4. Gap detection, cognitive modes, and composition inspection are
   the primary API surface.

Analogi: Ini adalah "penerjemah" antara bahasa Rust (RSVS v12 core)
dan bahasa Python (layer2 modules). Bridge ini menerjemahkan API Rust
ke Python dengan cara yang konsisten.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Epistemological seed labels — must match Rust core's SEED_LABEL_LIST
# ---------------------------------------------------------------------------

# Must match Rust SeedPrimitive enum variants exactly.
# Rust: pub enum SeedPrimitive { Trust, Risk, Value, Goal, Identity }
SEED_LABELS = ["Trust", "Risk", "Value", "Goal", "Identity"]

# ---------------------------------------------------------------------------
# Try to import the Rust core v12 pipeline
# ---------------------------------------------------------------------------

_v12_available = False

try:
    from rsvs import PyV12Pipeline as _PyV12Pipeline  # type: ignore[import]
    _v12_available = True
except Exception:
    pass


def is_rust_core_available() -> bool:
    """Check if the RSVS v12 Rust core is importable."""
    return _v12_available


# ---------------------------------------------------------------------------
# Fallback graph — lightweight in-memory knowledge store for v12
# ---------------------------------------------------------------------------

@dataclass
class _FallbackComposition:
    """A composition in the fallback graph."""
    comp_id: str
    composition_type: str = "Event"
    confidence: float = 0.5
    members: list[dict] = field(default_factory=list)
    source_text: Optional[str] = None
    lifecycle: str = "New"
    epistemic: str = "Observed"


class _FallbackGraph:
    """Lightweight fallback knowledge graph for when Rust core is unavailable.

    This is a simple in-memory store that mimics the v12 API surface
    (compositions, nodes, gap detection) so that layer2 modules can
    function during development and testing without the Rust core.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._compositions: dict[str, _FallbackComposition] = {}
        self._edges: dict[str, list[str]] = {}
        self._gaps: list[dict] = []
        self._comp_seed_scores: dict[str, dict] = {}
        self._next_comp_id: int = 0

    def ingest(self, text: str) -> dict:
        """Ingest text by extracting keywords as nodes and creating a v12 composition."""
        words = self._extract_keywords(text)
        atoms_promoted = 0
        edges_created = 0

        for word in words:
            if word not in self._nodes:
                self._nodes[word] = {
                    "label": word,
                    "confidence": 0.5,
                    "observation_count": 1,
                }
                atoms_promoted += 1
            else:
                self._nodes[word]["observation_count"] += 1
                self._nodes[word]["confidence"] = min(
                    1.0, self._nodes[word]["confidence"] + 0.05
                )

            for other in words:
                if other != word:
                    if word not in self._edges:
                        self._edges[word] = []
                    if other not in self._edges[word]:
                        self._edges[word].append(other)
                        edges_created += 1

        # Create a v12-compatible Composition only when we have keywords
        compositions_created = 0
        gaps_detected = 0

        if words:
            comp_id = f"fallback-comp-{self._next_comp_id}"
            self._next_comp_id += 1

            # Confidence heuristic: fewer keywords → lower confidence
            comp_confidence = min(0.6, 0.1 + len(words) * 0.1)

            members = [
                {
                    "node_id": word,
                    "role": "keyword",
                    "label": word,
                    "confidence": self._nodes[word]["confidence"],
                }
                for word in words
            ]

            composition = _FallbackComposition(
                comp_id=comp_id,
                composition_type="Event",
                confidence=comp_confidence,
                members=members,
                source_text=text,
                lifecycle="New",
                epistemic="Observed",
            )
            self._compositions[comp_id] = composition
            self._comp_seed_scores[comp_id] = {}  # placeholder
            compositions_created = 1

            # Count how many gaps this composition introduces
            gaps_detected = 1 if comp_confidence < 0.3 else 0

        return {
            "atoms_created": atoms_promoted,
            "compositions_created": compositions_created,
            "gaps_detected": gaps_detected,
            "edges_created": edges_created,
            "enrichments_applied": 0,
            "governance_transitions": 0,
            "cognitive_mode": "Reactive",
            "fallback": True,
        }

    def compositions(self) -> list[_FallbackComposition]:
        """Return all compositions."""
        return list(self._compositions.values())

    def detect_gaps(self) -> list[dict]:
        """Return detected gaps for low-confidence compositions."""
        gaps: list[dict] = []
        for comp in self._compositions.values():
            if comp.confidence < 0.3:
                gaps.append({
                    "gap_id": f"gap-{comp.comp_id}",
                    "gap_type": "LowConfidence",
                    "description": (
                        f"Composition '{comp.comp_id}' has low confidence "
                        f"({comp.confidence:.2f})"
                    ),
                    "confidence": comp.confidence,
                    "severity": "low",
                    "missing_role": "unknown",
                    "source_composition_id": comp.comp_id,
                })
        return gaps

    def comp_seed_scores(self, comp_id: str) -> dict:
        """Return seed_scores dict for a composition (empty dict placeholder)."""
        return self._comp_seed_scores.get(comp_id, {})

    def composition_count(self) -> int:
        return len(self._compositions)

    def node_count(self) -> int:
        return len(self._nodes)

    def snapshot_json(self) -> str:
        return json.dumps({
            "nodes": list(self._nodes.keys()),
            "compositions": len(self._compositions),
        })

    def find_weak_frames(self) -> list[str]:
        return []

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        stop_words = {
            "that", "this", "with", "from", "have", "been", "they",
            "their", "which", "would", "there", "could", "about",
            "other", "into", "more", "than", "then", "some", "very",
            "also", "just", "like", "only", "over", "such", "after",
            "yang", "dan", "dari", "untuk", "dengan", "adalah", "itu",
            "ini", "ke", "di", "pada", "tidak", "akan", "telah", "oleh",
            "the", "and", "but", "for", "not", "you", "all", "can",
        }
        words = text.lower().replace(",", " ").replace(".", " ").split()
        return [w for w in words if len(w) > 2 and w not in stop_words][:30]


# ---------------------------------------------------------------------------
# V12PipelineBridge — the ONLY bridge
# ---------------------------------------------------------------------------

class V12PipelineBridge:
    """Unified adapter for the v12.0 DAG pipeline (PyV12Pipeline or fallback).

    This is the SINGLE point of contact for all layer2 modules.
    It provides a consistent Python API regardless of whether the
    Rust v12 core is available or not.

    Usage:
        bridge = V12PipelineBridge()
        if bridge.available:
            result = bridge.ingest("some text")
            print(f"Mode: {result['cognitive_mode']}")
            for comp in bridge.compositions():
                print(f"  {comp.id}: {comp.composition_type}")

    Attributes:
        available: Whether a working v12 pipeline is connected.
    """

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._fallback: Optional[_FallbackGraph] = None

        if _v12_available:
            try:
                self._pipeline = _PyV12Pipeline()
                logger.info("V12PipelineBridge initialized with Rust v12 core")
            except Exception as exc:
                logger.warning("Failed to initialize PyV12Pipeline: %s", exc)
                self._fallback = _FallbackGraph()
                logger.info("V12PipelineBridge initialized in FALLBACK mode")
        else:
            self._fallback = _FallbackGraph()
            logger.info("V12PipelineBridge initialized in FALLBACK mode (no Rust core)")

    @property
    def available(self) -> bool:
        """Whether the v12 pipeline (Rust or fallback) is available."""
        return self._pipeline is not None or self._fallback is not None

    @property
    def is_rust_core(self) -> bool:
        """Whether the Rust v12 core is being used."""
        return self._pipeline is not None

    # ------------------------------------------------------------------
    # Core: ingest
    # ------------------------------------------------------------------

    def ingest(self, text: str) -> dict:
        """Ingest text through the v12 DAG pipeline.

        Returns a dict with summary statistics including:
        - atoms_created: Number of new atoms created
        - compositions_created: Number of new compositions
        - gaps_detected: Number of knowledge gaps found
        - cognitive_mode: The cognitive mode selected (Reactive/Analytical/Reflective)
        """
        if self._pipeline is not None:
            try:
                result = self._pipeline.v12_ingest(text)
                return {
                    "atoms_created": result.atoms_created,
                    "compositions_created": result.compositions_created,
                    "gaps_detected": result.gaps_detected,
                    "edges_created": result.edges_created,
                    "enrichments_applied": result.enrichments_applied,
                    "governance_transitions": result.governance_transitions,
                    "cognitive_mode": result.cognitive_mode,
                }
            except Exception as exc:
                logger.warning("v12_ingest failed, using fallback: %s", exc)

        if self._fallback is not None:
            return self._fallback.ingest(text)

        return {
            "atoms_created": 0,
            "compositions_created": 0,
            "gaps_detected": 0,
            "edges_created": 0,
            "enrichments_applied": 0,
            "governance_transitions": 0,
            "cognitive_mode": "Reactive",
        }

    # ------------------------------------------------------------------
    # Cognitive mode
    # ------------------------------------------------------------------

    def cognitive_mode(self, text: str) -> str:
        """Select cognitive mode for the given text.

        Returns one of: "Reactive", "Analytical", "Reflective".
        """
        if self._pipeline is not None:
            try:
                return self._pipeline.select_cognitive_mode(text)
            except Exception:
                pass
        return "Reactive"

    # ------------------------------------------------------------------
    # Composition inspection
    # ------------------------------------------------------------------

    def compositions(self) -> list[dict]:
        """Get all compositions in the v12 graph.

        Returns a list of dicts, each with: id, composition_type,
        lifecycle, epistemic, confidence, members, seed_scores, etc.
        """
        if self._pipeline is not None:
            try:
                result = []
                for comp in self._pipeline.compositions():
                    result.append({
                        "id": comp.id,
                        "composition_type": comp.composition_type,
                        "lifecycle": comp.lifecycle,
                        "epistemic": comp.epistemic,
                        "confidence": comp.confidence,
                        "provenance": comp.provenance,
                        "members": [
                            {
                                "node_id": m.node_id,
                                "role": m.role,
                                "label": m.label,
                                "confidence": m.confidence,
                            }
                            for m in comp.members
                        ],
                        "seed_scores": comp.seed_scores,
                        "source_text": comp.source_text,
                        "batch_seen": comp.batch_seen,
                        "contradiction": comp.contradiction,
                    })
                return result
            except Exception as exc:
                logger.warning("compositions() failed: %s", exc)

        if self._fallback is not None:
            return [
                {
                    "id": c.comp_id,
                    "composition_type": c.composition_type,
                    "lifecycle": c.lifecycle,
                    "epistemic": c.epistemic,
                    "confidence": c.confidence,
                    "members": c.members,
                    "seed_scores": self._fallback.comp_seed_scores(c.comp_id),
                    "source_text": c.source_text,
                }
                for c in self._fallback.compositions()
            ]

        return []

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def detect_gaps(self) -> list[dict]:
        """Detect knowledge gaps in the current graph state.

        Returns a list of dicts with: gap_id, gap_type, description,
        confidence, severity, missing_role, source_composition_id.
        """
        if self._pipeline is not None:
            try:
                result = []
                for gap in self._pipeline.detect_gaps():
                    result.append({
                        "gap_id": gap.gap_id,
                        "gap_type": gap.gap_type,
                        "description": gap.description,
                        "confidence": gap.confidence,
                        "severity": gap.severity,
                        "missing_role": gap.missing_role,
                        "source_composition_id": gap.source_composition_id,
                    })
                return result
            except Exception as exc:
                logger.warning("detect_gaps() failed: %s", exc)

        if self._fallback is not None:
            return self._fallback.detect_gaps()

        return []

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def composition_count(self) -> int:
        """Number of compositions in the graph."""
        if self._pipeline is not None:
            try:
                return self._pipeline.composition_count()
            except Exception:
                pass
        if self._fallback is not None:
            return self._fallback.composition_count()
        return 0

    def node_count(self) -> int:
        """Number of nodes in the graph."""
        if self._pipeline is not None:
            try:
                return self._pipeline.node_count()
            except Exception:
                pass
        if self._fallback is not None:
            return self._fallback.node_count()
        return 0

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot_json(self) -> str:
        """Get a JSON snapshot of the current graph state."""
        if self._pipeline is not None:
            try:
                return self._pipeline.snapshot_json()
            except Exception:
                pass
        if self._fallback is not None:
            return self._fallback.snapshot_json()
        return "{}"

    # ------------------------------------------------------------------
    # Gap detection toggle
    # ------------------------------------------------------------------

    def set_gap_detection(self, enabled: bool) -> None:
        """Enable or disable gap detection for subsequent ingest calls."""
        if self._pipeline is not None:
            try:
                self._pipeline.set_gap_detection(enabled)
            except Exception:
                pass

    def gap_detection_enabled(self) -> bool:
        """Check whether gap detection is currently enabled."""
        if self._pipeline is not None:
            try:
                return self._pipeline.gap_detection_enabled()
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------
    # Weak frames
    # ------------------------------------------------------------------

    def find_weak_frames(self) -> list[str]:
        """Find low-confidence Event compositions missing expected roles."""
        if self._pipeline is not None:
            try:
                return self._pipeline.find_weak_frames()
            except Exception:
                pass
        if self._fallback is not None:
            return self._fallback.find_weak_frames()
        return []

    # ------------------------------------------------------------------
    # Get specific composition
    # ------------------------------------------------------------------

    def get_composition(self, comp_id: str) -> Optional[dict]:
        """Get a specific composition by its ID.

        Returns None if no composition with the given ID exists.
        """
        if self._pipeline is not None:
            try:
                comp = self._pipeline.get_composition(comp_id)
                if comp is not None:
                    return {
                        "id": comp.id,
                        "composition_type": comp.composition_type,
                        "lifecycle": comp.lifecycle,
                        "epistemic": comp.epistemic,
                        "confidence": comp.confidence,
                        "provenance": comp.provenance,
                        "members": [
                            {
                                "node_id": m.node_id,
                                "role": m.role,
                                "label": m.label,
                                "confidence": m.confidence,
                            }
                            for m in comp.members
                        ],
                        "seed_scores": comp.seed_scores,
                        "source_text": comp.source_text,
                    }
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Backward-compatible methods (for layer3/reasoning.py compatibility)
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Backward-compatible alias for `available`."""
        return self.available

    def senses(self, concept: str) -> Optional[list[dict]]:
        """Get sense-like info for a concept.

        In v12, "senses" are replaced by compositions that reference
        the concept node. This method returns composition members as
        sense-like dicts for backward compatibility.
        """
        comps = self.compositions()
        related = []
        for comp in comps:
            for m in comp.get("members", []):
                if m.get("label", "").lower() == concept.lower():
                    related.append({
                        "sense_idx": 0,
                        "n_contexts": 1,
                        "coherence": comp.get("confidence", 0.5),
                        "status": comp.get("lifecycle", "New").lower(),
                        "core_atoms": [m2["label"] for m2 in comp.get("members", []) if m2.get("label")],
                        "layer": 0,
                        "grounding_score": comp.get("confidence", 0.5),
                    })
                    break  # One "sense" per composition
        return related if related else None

    def compose(self, label: str, compositions: list[tuple[str, str]], lang: Optional[str] = None) -> Optional[int]:
        """Compose — in v12, this is handled by the DAG pipeline.

        This is a no-op that returns a placeholder ID for backward
        compatibility. Actual composition happens during ingest.
        """
        # In v12, compositions are created automatically by the pipeline
        # during ingest. This method exists for backward compatibility.
        logger.debug("compose() called in v12 mode — compositions are pipeline-driven")
        return hash(label) % (2**31)

    def mcts_query(self, node_label: str, max_depth: int = 3, simulations: int = 50) -> Optional[dict]:
        """MCTS query — in v12, replaced by cognitive mode exploration.

        Returns a dict compatible with the old MCTSResult format,
        using composition-based expansion instead of MCTS.
        """
        comps = self.compositions()
        scored_atoms: list[tuple[str, float]] = []
        for comp in comps:
            conf = comp.get("confidence", 0.5)
            comp_type = comp.get("composition_type", "Unknown")
            scored_atoms.append((f"{comp_type}:{comp['id'][:20]}", conf))

        return {
            "active_sense_idx": 0,
            "total_senses": 1,
            "scored_atoms": scored_atoms[:20],
            "depth_reached": 1,
            "halt_reason": "v12_cognitive_mode",
            "simulations_run": 0,
            "best_path": [(node_label, 0)] + [(s, 0) for s, _ in scored_atoms[:5]],
            "layer": 0,
            "grounding_score": 0.5,
        }

    def query(self, concept: str, context: str = "") -> Optional[dict]:
        """Query a concept — in v12, use compositions() instead.

        Returns a v12-compatible query result for backward compatibility.
        """
        comps = self.compositions()
        atoms = []
        for comp in comps:
            for m in comp.get("members", []):
                if m.get("label", "").lower() == concept.lower():
                    atoms.append((m.get("label", ""), m.get("confidence", 0.5)))

        if not atoms:
            return None

        return {
            "sense_idx": 0,
            "sense_n": 1,
            "atoms": atoms[:10],
            "layer": 0,
            "grounding_score": 0.5,
            "compositions": [(c["id"], 0) for c in comps[:10]],
        }

    def nodes(self, include_seeds: bool = False) -> list[str]:
        """List all node labels in the graph."""
        comps = self.compositions()
        labels = set()
        for comp in comps:
            for m in comp.get("members", []):
                if m.get("label"):
                    labels.add(m["label"])
        return list(labels)

    def confidence_map(self) -> dict[str, float]:
        """Return confidence scores for all compositions."""
        comps = self.compositions()
        return {c["id"]: c.get("confidence", 0.5) for c in comps}


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

# Backward-compatible aliases.
# AbstractionBridge was the v8.3 bridge class — now unified into V12PipelineBridge.
# RsvsBridge was the v8.3–v11 name — now also unified.
# Both aliases exist so that existing import statements don't break,
# but new code should use V12PipelineBridge directly.
AbstractionBridge = V12PipelineBridge
RsvsBridge = V12PipelineBridge


def get_bridge() -> V12PipelineBridge:
    """Get or create the global V12PipelineBridge instance.

    Returns a singleton bridge. All layer2 modules should use this
    instead of creating their own bridge instances.
    """
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = V12PipelineBridge()
    return _global_bridge


_global_bridge: Optional[V12PipelineBridge] = None
