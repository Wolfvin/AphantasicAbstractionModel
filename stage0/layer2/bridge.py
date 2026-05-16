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
from collections import deque
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
        """Ingest text by extracting keywords as nodes and creating a v12 composition.

        STUB:IMPROVED — detects basic Subject-Verb-Object patterns,
        assigns typed roles, and computes seed scores.
        """
        import re as _re

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

        # Detect basic Subject-Verb-Object event patterns
        tokens = _re.sub(r"[^\w\s'-]", " ", text).split()
        role_map = self._detect_svo_roles(tokens, words)

        # Determine composition type and lifecycle/epistemic from text
        composition_type = self._infer_composition_type(text)
        lifecycle = self._infer_lifecycle(text)
        epistemic = self._infer_epistemic(text)

        # Create a v12-compatible Composition only when we have keywords
        compositions_created = 0
        gaps_detected = 0

        if words:
            comp_id = f"fallback-comp-{self._next_comp_id}"
            self._next_comp_id += 1

            # Confidence heuristic: more structured → higher confidence
            has_roles = any(r != "keyword" for r in role_map.values())
            comp_confidence = min(0.8, 0.2 + len(words) * 0.08
                                  + (0.15 if has_roles else 0.0))

            members = [
                {
                    "node_id": word,
                    "role": role_map.get(word, "keyword"),
                    "label": word,
                    "confidence": self._nodes[word]["confidence"],
                }
                for word in words
            ]

            composition = _FallbackComposition(
                comp_id=comp_id,
                composition_type=composition_type,
                confidence=comp_confidence,
                members=members,
                source_text=text,
                lifecycle=lifecycle,
                epistemic=epistemic,
            )
            self._compositions[comp_id] = composition

            # Compute seed scores based on composition structure
            self._comp_seed_scores[comp_id] = self._compute_seed_scores(
                composition
            )
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

    # -- Ingest helpers --

    @staticmethod
    def _detect_svo_roles(tokens: list[str], keywords: list[str]) -> dict[str, str]:
        """Detect Subject-Verb-Object roles for Indonesian/English patterns.

        STUB:IMPROVED — basic SVO pattern detection.
        Returns a mapping of keyword → role string.
        """
        role_map: dict[str, str] = {}
        keyword_set = set(keywords)

        # Common verb patterns (Indonesian + English)
        verb_markers = {
            # Indonesian
            "membuat", "mengambil", "memberi", "pergi", "datang", "melihat",
            "mendengar", "menulis", "membaca", "berbicara", "mengerjakan",
            "menyelesaikan", "mencari", "menemukan", "menggunakan", "mengirim",
            "menerima", "menjual", "membeli", "memasak", "berlari",
            "bermain", "belajar", "mengajar", "bekerja", "tinggal",
            # English
            "make", "take", "give", "go", "come", "see", "hear",
            "write", "read", "speak", "work", "find", "use", "send",
            "receive", "sell", "buy", "cook", "run", "play", "study",
            "teach", "live", "build", "destroy", "create", "destroy",
            "eat", "drink", "sleep", "walk", "drive", "fly",
        }

        # Simple scan: look for keyword tokens and assign roles
        prev_role = None
        for tok in tokens:
            lower = tok.lower()
            if lower not in keyword_set:
                continue
            if lower in verb_markers:
                role_map[lower] = "Action"
                prev_role = "Action"
            elif prev_role is None or prev_role == "Action":
                # First keyword or after a verb → Subject or Object
                if prev_role is None:
                    role_map[lower] = "Agent"
                    prev_role = "Agent"
                else:
                    role_map[lower] = "Patient"
                    prev_role = "Patient"
            else:
                role_map[lower] = "Context"
                prev_role = "Context"

        return role_map

    @staticmethod
    def _infer_composition_type(text: str) -> str:
        """Infer composition type from text content."""
        lower = text.lower()
        # Hypothesis markers
        if any(m in lower for m in ["mungkin", "barangkali", "perhaps", "maybe",
                                     "possibly", "probably", "kemungkinan"]):
            return "Hypothesis"
        # Question markers
        if any(m in lower for m in ["?", "apakah", "bagaimana", "why", "how",
                                     "what", "who", "where", "when"]):
            return "Question"
        # Rule / generalization markers
        if any(m in lower for m in ["selalu", "tidak pernah", "always", "never",
                                     "setiap", "every", "all", "semua"]):
            return "Rule"
        return "Event"

    @staticmethod
    def _infer_lifecycle(text: str) -> str:
        """Infer lifecycle state from text."""
        lower = text.lower()
        if any(m in lower for m in ["mungkin", "perhaps", "maybe", "possibly"]):
            return "Candidate"
        if any(m in lower for m in ["seharusnya", "should", "must", "harus"]):
            return "Proposed"
        return "New"

    @staticmethod
    def _infer_epistemic(text: str) -> str:
        """Infer epistemic state from text."""
        lower = text.lower()
        if any(m in lower for m in ["katanya", "dikatakan", "rumor", "reportedly",
                                     "allegedly", "konon"]):
            return "Hearsay"
        if any(m in lower for m in ["saya lihat", "saya dengar", "i saw",
                                     "i heard", "terlihat", "visible"]):
            return "Observed"
        if any(m in lower for m in ["seharusnya", "should be", "supposed to"]):
            return "Inferred"
        return "Observed"

    @staticmethod
    def _compute_seed_scores(composition: _FallbackComposition) -> dict:
        """Compute epistemological seed scores based on composition structure.

        STUB:IMPROVED — derives seed scores from member roles and confidence.
        """
        scores: dict[str, float] = {}
        member_roles = [m.get("role", "keyword") for m in composition.members]
        has_agent = "Agent" in member_roles
        has_action = "Action" in member_roles
        has_patient = "Patient" in member_roles

        # Trust: higher when we have Agent+Action (someone did something)
        scores["Trust"] = min(1.0, 0.3 + (0.3 if has_agent else 0.0)
                             + (0.2 if has_action else 0.0)
                             + (0.1 if composition.epistemic == "Observed" else 0.0))

        # Risk: higher for hypotheses and hearsay
        scores["Risk"] = min(1.0, 0.2 + (0.3 if composition.composition_type == "Hypothesis" else 0.0)
                            + (0.3 if composition.epistemic == "Hearsay" else 0.0)
                            + (0.2 if composition.confidence < 0.4 else 0.0))

        # Value: higher for well-structured events
        scores["Value"] = min(1.0, 0.2 + (0.2 if has_agent else 0.0)
                             + (0.2 if has_action else 0.0)
                             + (0.2 if has_patient else 0.0)
                             + (0.2 if composition.confidence > 0.6 else 0.0))

        # Goal: higher for questions and proposed compositions
        scores["Goal"] = min(1.0, 0.2 + (0.4 if composition.composition_type == "Question" else 0.0)
                            + (0.2 if composition.lifecycle == "Proposed" else 0.0)
                            + (0.2 if composition.composition_type == "Hypothesis" else 0.0))

        # Identity: higher for named entities and repeated observations
        obs_count = sum(1 for m in composition.members if m.get("role") in ("Agent", "Patient"))
        scores["Identity"] = min(1.0, 0.2 + 0.2 * obs_count)

        return scores

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
        """Return seed_scores dict for a composition."""
        # STUB:IMPROVED — now returns real seed scores computed during ingest
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
        # STUB:MINIMAL — placeholder, no weak frame detection in fallback
        return []

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract keywords with improved stop-word handling and noun-phrase detection."""
        # STUB:IMPROVED — real keyword extraction with noun phrases
        stop_words = {
            # English
            "that", "this", "with", "from", "have", "been", "they",
            "their", "which", "would", "there", "could", "about",
            "other", "into", "more", "than", "then", "some", "very",
            "also", "just", "like", "only", "over", "such", "after",
            "before", "because", "between", "through", "during", "without",
            "these", "those", "each", "where", "when", "what", "how",
            "was", "were", "been", "being", "had", "has", "did", "does",
            "shall", "should", "may", "might", "must", "need",
            # Indonesian
            "yang", "dan", "dari", "untuk", "dengan", "adalah", "itu",
            "ini", "ke", "di", "pada", "tidak", "akan", "telah", "oleh",
            "sebuah", "seorang", "secara", "karena", "jika", "atau",
            "tetapi", "namun", "sementara", "sedangkan", "melalui",
            "lagi", "sudah", "belum", "masih", "hanya", "bahwa",
            "dia", "mereka", "kita", "kami", "saya", "anda",
            # Common articles / short words
            "the", "and", "but", "for", "not", "you", "all", "can",
            "her", "him", "his", "our", "its", "she",
        }

        # Strip punctuation properly (preserve internal hyphens/apostrophes)
        import re as _re
        cleaned = _re.sub(r"[^\w\s'-]", " ", text)
        tokens = cleaned.split()

        # Detect multi-word noun phrases: capitalized words not at start of sentence
        # A word at position 0 after punctuation or start is likely sentence-initial
        phrases: list[str] = []
        current_phrase: list[str] = []

        for i, tok in enumerate(tokens):
            lower = tok.lower()
            # Check if capitalized and not at sentence start
            is_capitalized = tok[0].isupper() if tok else False
            prev_is_boundary = (i == 0
                                or tokens[i - 1].endswith((".", "!", "?"))
                                or tokens[i - 1].lower() in stop_words)

            if is_capitalized and not prev_is_boundary and len(lower) > 2 and lower not in stop_words:
                current_phrase.append(lower)
            else:
                if len(current_phrase) >= 2:
                    phrases.append(" ".join(current_phrase))
                current_phrase = []

        if len(current_phrase) >= 2:
            phrases.append(" ".join(current_phrase))

        # Single-word keywords
        single_words = [w.lower() for w in tokens if len(w) > 2 and w.lower() not in stop_words]

        # Combine: phrases first, then single words (dedup)
        seen = set()
        result: list[str] = []
        for p in phrases:
            if p not in seen:
                seen.add(p)
                result.append(p)
        for w in single_words:
            if w not in seen:
                seen.add(w)
                result.append(w)

        return result[:30]


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

    def ingest(self, text: str, source_provenance: Optional[str] = None) -> dict:
        """Ingest text through the v12 DAG pipeline.

        Args:
            text: The text to ingest.
            source_provenance: Optional provenance tag (e.g. "user", "recall").
                In v12, this is logged but does not change pipeline behavior.

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
        # STUB:MINIMAL — fallback always returns Reactive; Rust core selects properly
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
        # STUB:PLACEHOLDER — fallback always returns False
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
        # STUB:MINIMAL — only works with Rust core; fallback has no lookup by ID
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

        STUB:MINIMAL — works but simplified (one sense per composition)
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
        """Compose — create a real composition from member tuples.

        STUB:IMPROVED — creates a _FallbackComposition in fallback mode
        with proper members, typed roles, and composition_type inference.
        In Rust-core mode, delegates to the pipeline.

        Args:
            label: Composition label (prefix like deduction_ → Hypothesis).
            compositions: List of (node_id, sense_id) tuples for members.
            lang: Optional language hint.

        Returns:
            The composition ID as an int (hash of comp_id), or None.
        """
        if self._pipeline is not None:
            try:
                # Delegate to Rust core if available
                result = self._pipeline.compose(label, compositions, lang)
                if result is not None:
                    return result
            except Exception as exc:
                logger.debug("Rust compose() failed, using fallback: %s", exc)

        # Fallback: create a real _FallbackComposition
        if self._fallback is not None:
            fg = self._fallback
            comp_id = f"compose-{fg._next_comp_id}"
            fg._next_comp_id += 1

            # Infer composition_type from label prefix
            composition_type = "Event"
            label_lower = label.lower()
            if label_lower.startswith("deduction_"):
                composition_type = "Hypothesis"
            elif label_lower.startswith("induction_"):
                composition_type = "Rule"
            elif label_lower.startswith("abduction_"):
                composition_type = "Hypothesis"
            elif label_lower.startswith("analogy_"):
                composition_type = "Hypothesis"
            elif label_lower.startswith("composition_"):
                composition_type = "Event"
            elif label_lower.startswith("question_"):
                composition_type = "Question"

            # Build members from tuples
            members = []
            for node_id, sense_id in compositions:
                # Ensure the node exists in the graph
                if node_id not in fg._nodes:
                    fg._nodes[node_id] = {
                        "label": node_id,
                        "confidence": 0.5,
                        "observation_count": 1,
                    }
                members.append({
                    "node_id": node_id,
                    "role": "evidence",
                    "label": node_id,
                    "confidence": fg._nodes[node_id]["confidence"],
                })

            # Confidence based on member count and type
            conf = min(0.8, 0.3 + len(members) * 0.1)
            if composition_type == "Hypothesis":
                conf = min(0.7, conf)  # Hypotheses are less confident

            comp = _FallbackComposition(
                comp_id=comp_id,
                composition_type=composition_type,
                confidence=conf,
                members=members,
                source_text=label,
                lifecycle="Candidate" if composition_type == "Hypothesis" else "New",
                epistemic="Inferred",
            )
            fg._compositions[comp_id] = comp
            fg._comp_seed_scores[comp_id] = fg._compute_seed_scores(comp)

            # Create edges between all member nodes
            for i, m1 in enumerate(members):
                for j, m2 in enumerate(members):
                    if i != j:
                        n1, n2 = m1["node_id"], m2["node_id"]
                        if n1 not in fg._edges:
                            fg._edges[n1] = []
                        if n2 not in fg._edges[n1]:
                            fg._edges[n1].append(n2)

            logger.debug(
                "compose() created fallback composition '%s' type=%s with %d members",
                comp_id, composition_type, len(members)
            )
            # Return a stable int ID derived from the comp_id
            return hash(comp_id) % (2**31)

        return None

    def mcts_query(self, node_label: str, max_depth: int = 3, simulations: int = 50) -> Optional[dict]:
        """MCTS query — BFS graph traversal from the given node_label.

        STUB:IMPROVED — replaced fake MCTS with real BFS traversal using
        _FallbackGraph._edges. Builds scored paths by following edges
        up to max_depth.

        Args:
            node_label: Starting node for traversal.
            max_depth: Maximum BFS depth.
            simulations: Ignored in fallback mode (kept for API compat).

        Returns:
            Dict compatible with the old MCTSResult format.
        """
        if self._pipeline is not None:
            try:
                result = self._pipeline.mcts_query(node_label, max_depth, simulations)
                if result is not None:
                    return result
            except Exception as exc:
                logger.debug("Rust mcts_query() failed, using fallback: %s", exc)

        # BFS traversal over the fallback graph
        scored_atoms: list[tuple[str, float]] = []
        visited: set[str] = set()
        best_path: list[tuple[str, int]] = [(node_label, 0)]
        depth_reached = 0

        if self._fallback is not None:
            edges = self._fallback._edges
            nodes = self._fallback._nodes

            # BFS queue: (node_label, depth)
            queue: deque[tuple[str, int]] = deque()

            # Start from the exact label or try lowercase match
            start = node_label
            if start not in edges and start.lower() in {k.lower(): k for k in edges}:
                start = {k.lower(): k for k in edges}[start.lower()]

            queue.append((start, 0))
            visited.add(start.lower())

            while queue:
                current, depth = queue.popleft()
                if depth >= max_depth:
                    continue

                # Score current node by its confidence in the graph
                conf = nodes.get(current, {}).get("confidence", 0.3)
                scored_atoms.append((current, conf))

                # Explore neighbors
                for neighbor in edges.get(current, []):
                    if neighbor.lower() not in visited:
                        visited.add(neighbor.lower())
                        queue.append((neighbor, depth + 1))
                        if depth + 1 > depth_reached:
                            depth_reached = depth + 1

            # Build best_path from highest-scored atoms
            sorted_atoms = sorted(scored_atoms, key=lambda x: -x[1])
            best_path = [(node_label, 0)] + [(s, d) for s, d in
                       [(a[0], 0) for a in sorted_atoms[:max_depth * 3]]]

        # If fallback not available or no results, use composition-based fallback
        if not scored_atoms:
            comps = self.compositions()
            for comp in comps:
                conf = comp.get("confidence", 0.5)
                comp_type = comp.get("composition_type", "Unknown")
                scored_atoms.append((f"{comp_type}:{comp['id'][:20]}", conf))

        # Compute grounding score from scored atoms
        grounding_score = 0.5
        if scored_atoms:
            grounding_score = sum(s for _, s in scored_atoms) / len(scored_atoms)

        return {
            "active_sense_idx": 0,
            "total_senses": max(1, len(scored_atoms)),
            "scored_atoms": scored_atoms[:20],
            "depth_reached": depth_reached,
            "halt_reason": "bfs_complete" if depth_reached > 0 else "no_edges",
            "simulations_run": len(visited),
            "best_path": best_path[:10],
            "layer": 0,
            "grounding_score": grounding_score,
        }

    def query(self, concept: str, context: str = "") -> Optional[dict]:
        """Query a concept — in v12, use compositions() instead.

        Returns a v12-compatible query result for backward compatibility.

        STUB:MINIMAL — works but simplified
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
        # STUB:MINIMAL — ignores include_seeds
        comps = self.compositions()
        labels = set()
        for comp in comps:
            for m in comp.get("members", []):
                if m.get("label"):
                    labels.add(m["label"])
        return list(labels)

    def confidence_map(self) -> dict[str, float]:
        """Return confidence scores for all compositions."""
        # STUB:MINIMAL — works but simplified
        comps = self.compositions()
        return {c["id"]: c.get("confidence", 0.5) for c in comps}

    # ------------------------------------------------------------------
    # Layer2 compatibility methods (for context, situation, predictive, pattern)
    # ------------------------------------------------------------------

    def appraise(self, text: str) -> dict:
        """Appraise text — evaluate confidence and quality.

        In v12, this delegates to ingest() and returns a quality
        assessment of the result. Used by context.py and predictive.py.

        STUB:MINIMAL — works but simplified
        """
        result = self.ingest(text)
        comps_count = self.composition_count()
        avg_conf = 0.0
        if comps_count > 0:
            cmap = self.confidence_map()
            avg_conf = sum(cmap.values()) / len(cmap) if cmap else 0.0

        return {
            "confidence": avg_conf,
            "quality": "high" if avg_conf > 0.7 else "moderate" if avg_conf > 0.4 else "low",
            "n_compositions": comps_count,
            "n_gaps": len(self.detect_gaps()),
            "positive": avg_conf >= 0.4,
        }

    def relate(self, concept: str) -> list[str]:
        """Find concepts related to the given concept.

        In v12, this uses composition membership and edge traversal
        to find related nodes. Returns a list of related node labels.

        STUB:MINIMAL — works but simplified
        """
        related = set()
        comps = self.compositions()
        concept_lower = concept.lower()

        for comp in comps:
            members = comp.get("members", [])
            labels_in_comp = [m.get("label", "").lower() for m in members]
            if concept_lower in labels_in_comp:
                for m in members:
                    label = m.get("label", "")
                    if label.lower() != concept_lower and label:
                        related.add(label)

        # Also check fallback edges if available
        if self._fallback is not None:
            edges = self._fallback._edges.get(concept.lower(), [])
            related.update(e for e in edges if e.lower() != concept_lower)

        return list(related)

    def structural_similarity(self, a: str, b: str) -> float:
        """Compute structural similarity between two concepts.

        In v12, this uses Jaccard similarity on the neighborhoods
        of nodes a and b. Returns a float in [0, 1].

        STUB:MINIMAL — works but simplified
        """
        neighbors_a = set(self.relate(a))
        neighbors_b = set(self.relate(b))
        neighbors_a.add(a.lower())
        neighbors_b.add(b.lower())

        if not neighbors_a and not neighbors_b:
            return 0.0
        intersection = neighbors_a & neighbors_b
        union = neighbors_a | neighbors_b
        return len(intersection) / len(union) if union else 0.0

    def context_query(self, concept: str, context: str = "") -> Optional[dict]:
        """Query a concept with optional context.

        Combines relate() and query() for a richer result.
        Used by predictive.py.

        STUB:MINIMAL — works but simplified
        """
        base = self.query(concept, context)
        if base is None:
            return None
        base["related"] = self.relate(concept)
        base["similarity_to_context"] = (
            self.structural_similarity(concept, context) if context else 0.0
        )
        return base

    def substitution_analysis(self, a: str, b: str) -> dict:
        """Analyze whether concept a can substitute for concept b.

        In v12, this uses structural similarity and shared composition
        membership to determine substitutability.

        STUB:MINIMAL — works but simplified
        """
        sim = self.structural_similarity(a, b)
        related_a = set(self.relate(a))
        related_b = set(self.relate(b))
        shared = related_a & related_b

        return {
            "similarity": sim,
            "shared_contexts": len(shared),
            "a_unique_contexts": len(related_a - related_b),
            "b_unique_contexts": len(related_b - related_a),
            "substitutable": sim > 0.5,
        }

    def node_info(self, label: str) -> Optional[dict]:
        """Get information about a specific node.

        Returns a dict with node details, or None if not found.

        STUB:MINIMAL — works but simplified
        """
        comps = self.compositions()
        appearances = []
        for comp in comps:
            for m in comp.get("members", []):
                if m.get("label", "").lower() == label.lower():
                    appearances.append({
                        "composition_id": comp["id"],
                        "composition_type": comp.get("composition_type", "Unknown"),
                        "role": m.get("role", "Unknown"),
                        "confidence": m.get("confidence", 0.5),
                    })

        if not appearances:
            return None

        avg_conf = sum(a["confidence"] for a in appearances) / len(appearances)
        return {
            "label": label,
            "n_appearances": len(appearances),
            "avg_confidence": avg_conf,
            "appearances": appearances,
            "related": self.relate(label),
        }

    def latest_seq_v1(self) -> int:
        """Get the latest event sequence number.

        In v12, this returns the composition count as a proxy
        for sequence tracking. Used by situation.py.

        STUB:PLACEHOLDER — uses composition_count as proxy
        """
        return self.composition_count()

    def consume_events_v1(self, after_seq: int = 0) -> list[dict]:
        """Consume events since the given sequence number.

        In v12, this returns compositions created after the given
        sequence. Used by situation.py and predictive.py.

        STUB:MINIMAL — works but simplified (index-based slicing)
        """
        all_comps = self.compositions()
        # In fallback mode, return all compositions after the given index
        if after_seq < len(all_comps):
            return all_comps[after_seq:]
        return []

    def status(self) -> dict:
        """Get the current status of the bridge and pipeline.

        Returns a dict with operational status information.
        Used by situation.py.
        """
        return {
            "available": self.available,
            "is_rust_core": self.is_rust_core,
            "n_nodes": self.node_count(),
            "n_compositions": self.composition_count(),
            "n_gaps": len(self.detect_gaps()),
            "gap_detection_enabled": self.gap_detection_enabled(),
            "mode": "rust_v12" if self.is_rust_core else "fallback",
        }

    # ------------------------------------------------------------------
    # Enrichment loop
    # ------------------------------------------------------------------

    def run_enrichment_loop(self) -> dict:
        """Run the active enrichment loop: DetectGaps → SelectAcquisition → EnrichComposition.

        Only available when the Rust core is active. In fallback mode,
        this is a no-op that returns empty stats.

        STUB:PLACEHOLDER — returns empty stats in fallback mode

        Returns a dict with enrichment statistics.
        """
        if self._pipeline is not None:
            try:
                result = self._pipeline.run_enrichment_loop()
                return {
                    "enrichments_applied": result.enrichments_applied,
                    "governance_transitions": result.governance_transitions,
                    "cognitive_mode": result.cognitive_mode,
                }
            except Exception as exc:
                logger.warning("run_enrichment_loop() failed: %s", exc)

        return {
            "enrichments_applied": 0,
            "governance_transitions": 0,
            "cognitive_mode": "fallback",
        }

    # ------------------------------------------------------------------
    # Pending gaps (structured)
    # ------------------------------------------------------------------

    def pending_gaps(self) -> list[dict]:
        """Get all pending knowledge gaps with full structure.

        Returns a list of dicts with: gap_id, gap_type, description,
        confidence, severity, missing_role, source_composition_id.
        """
        if self._pipeline is not None:
            try:
                result = []
                for gap in self._pipeline.pending_gaps():
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
                logger.warning("pending_gaps() failed: %s", exc)

        # Fallback to detect_gaps in fallback mode
        return self.detect_gaps()

    # ------------------------------------------------------------------
    # Submit user answer
    # ------------------------------------------------------------------

    def submit_answer(self, gap_id: str, answer: str) -> bool:
        """Submit a user answer to fill a knowledge gap.

        Args:
            gap_id: The gap ID from a pending_gaps() call.
            answer: The user's answer text.

        Returns True if the answer was applied, False if gap not found
        or Rust core is unavailable.

        STUB:PLACEHOLDER — always returns False in fallback mode
        """
        if self._pipeline is not None:
            try:
                return self._pipeline.submit_answer(gap_id, answer)
            except Exception as exc:
                logger.warning("submit_answer() failed: %s", exc)

        return False

    # ------------------------------------------------------------------
    # Graph summary
    # ------------------------------------------------------------------

    def graph_summary(self) -> str:
        """Get a human-readable summary of the current graph state.

        Returns a string like: "Graph: 6 nodes, 2 compositions (1 stable, 0 candidate, 0 contradicted)"
        """
        if self._pipeline is not None:
            try:
                return self._pipeline.graph_summary()
            except Exception as exc:
                logger.warning("graph_summary() failed: %s", exc)

        # Fallback summary
        return (
            f"Graph: {self.node_count()} nodes, {self.composition_count()} compositions (fallback)"
        )

    # ------------------------------------------------------------------
    # Persistence (save/load)
    # ------------------------------------------------------------------

    def save(self, path: str) -> bool:
        """Save the current graph state to a JSON file.

        Args:
            path: File path to save to.

        Returns True on success, False on failure or fallback mode.

        STUB:PLACEHOLDER — not implemented in fallback mode
        """
        if self._pipeline is not None:
            try:
                self._pipeline.save(path)
                return True
            except Exception as exc:
                logger.warning("save() failed: %s", exc)

        return False

    def load(self, path: str) -> bool:
        """Load graph state from a JSON file, replacing current graph.

        Args:
            path: File path to load from.

        Returns True on success, False on failure or fallback mode.

        STUB:PLACEHOLDER — not implemented in fallback mode
        """
        if self._pipeline is not None:
            try:
                self._pipeline.load(path)
                return True
            except Exception as exc:
                logger.warning("load() failed: %s", exc)

        return False


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
