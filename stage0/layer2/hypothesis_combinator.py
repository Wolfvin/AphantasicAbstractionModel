"""
AAM Layer 2 — Hypothesis Combinator

Combines complementary hypotheses into hybrids that may reveal
entirely new possibilities neither parent could reach alone.

Key insight: Not A ∧ B (conjunction), but A × B (composition).
A hybrid combines insights from both parents and may have
emergent properties that neither parent has alone.

Example:
    A: "Dia marah karena dikhianati"
    B: "Dia marah karena harga diri tersentuh"
    A × B: "Dia marah karena dikhianati YANG menyentuh harga dirinya"
    Emergent: "Dikhianatan terhadap harga diri = pola trauma masa lalu"

The emergent possibility was NOT in the original space — it only
became visible after A and B were hybridized.

Composition modes:
    - CONCATENATIVE: Simple "A ∘ B" string composition (fast, no external call)
    - LLM_DRIVEN: Semantic composition via LLM callback (true A × B)
      The LLM synthesizes a new statement that neither parent could
      produce alone — this is the TRUE multiplicative composition.
"""

from __future__ import annotations

import sys as _stage0_sys
from pathlib import Path as _stage0_Path
_stage0_dir = str(_stage0_Path(__file__).resolve().parent)
while _stage0_dir and not _stage0_Path(_stage0_dir, "layer0").is_dir() and _stage0_Path(_stage0_dir).parent != _stage0_dir:
    _stage0_dir = str(_stage0_Path(_stage0_dir).parent)
if _stage0_dir not in _stage0_sys.path:
    _stage0_sys.path.insert(0, _stage0_dir)

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from layer2.bridge import RsvsBridge, get_bridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

_DEFAULT_COMPLEMENTARITY_THRESHOLD = 0.4
_DEFAULT_MAX_HYBRIDS = 30
_DEFAULT_IMPLICATION_DEPTH = 2

# Maximum scored_atoms to explore in implication tracing
_DEFAULT_MAX_IMPLICATION_ATOMS = 8

# Minimum path length for an implication to be considered novel
_DEFAULT_MIN_IMPLICATION_PATH_LEN = 1


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CompositionMode(Enum):
    """How hybrid statements are composed from parents.

    CONCATENATIVE: Fast "A ∘ B" string composition. No external calls.
        Good for testing and when no LLM is available.

    LLM_DRIVEN: True semantic composition via an LLM callback.
        The callback receives both parent statements and evidence
        context, and returns a NEW statement that synthesizes both.
        This is the TRUE A × B multiplicative composition.
    """
    CONCATENATIVE = "concatenative"
    LLM_DRIVEN = "llm_driven"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """Result of hybridizing two possibilities.

    Attributes:
        hybrid_id: Unique identifier for the hybrid.
        statement: The hybrid statement (A ∘ B composition).
        parent_a_id: ID of first parent.
        parent_b_id: ID of second parent.
        confidence: Combined confidence (with boost for complementarity).
        complementarity: How complementary the parents were.
        explained_evidence: Union of both parents' explained evidence.
        implications: Emergent possibilities traced from this hybrid.
    """

    hybrid_id: str
    statement: str
    parent_a_id: str
    parent_b_id: str
    confidence: float = 0.5
    complementarity: float = 0.0
    explained_evidence: set[str] = field(default_factory=set)
    implications: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hybrid_id": self.hybrid_id,
            "statement": self.statement,
            "parent_a_id": self.parent_a_id,
            "parent_b_id": self.parent_b_id,
            "confidence": round(self.confidence, 4),
            "complementarity": round(self.complementarity, 4),
            "explained_evidence_count": len(self.explained_evidence),
            "implications": list(self.implications),
        }


@dataclass
class ComplementarityScore:
    """Score of how complementary two possibilities are.

    Attributes:
        overlap_ratio: How much evidence they share (0.0–1.0, lower = more complementary).
        joint_coverage: How much evidence they cover together (0.0–1.0, higher = better).
        score: Combined complementarity score (0.0–1.0).
    """

    overlap_ratio: float = 0.0
    joint_coverage: float = 0.0
    score: float = 0.0


# ---------------------------------------------------------------------------
# HypothesisCombinator
# ---------------------------------------------------------------------------

class HypothesisCombinator:
    """Combines complementary possibilities into hybrid possibilities.

    The combinator finds pairs of possibilities that explain DIFFERENT
    parts of the evidence (low overlap, high joint coverage) and
    composes them into hybrids that are more complete than either parent.

    Key principle: A × B, not A ∧ B.
    The hybrid doesn't just say "both A and B are true" — it creates
    a NEW interpretation that synthesizes the insights of both parents.

    Additionally traces implications of each hybrid via RSVS MCTS
    to find emergent possibilities that neither parent could see alone.

    Composition modes:
        - CONCATENATIVE: Fast "A ∘ B" string composition.
        - LLM_DRIVEN: Semantic composition via callback — TRUE A × B.
          The callback synthesizes a new statement that captures what
          NEITHER parent alone could express.

    Attributes:
        rsvs_available: Whether a working RSVS instance is connected.
        is_rust_core: Whether the Rust core is being used.
    """

    def __init__(
        self,
        bridge: Optional[RsvsBridge] = None,
        complementarity_threshold: float = _DEFAULT_COMPLEMENTARITY_THRESHOLD,
        max_hybrids: int = _DEFAULT_MAX_HYBRIDS,
        implication_depth: int = _DEFAULT_IMPLICATION_DEPTH,
        composition_mode: CompositionMode = CompositionMode.CONCATENATIVE,
        compose_callback: Optional[Callable[[str, str, str], str]] = None,
    ) -> None:
        """Initialize the Hypothesis Combinator.

        Args:
            bridge: Optional pre-built RsvsBridge.
            complementarity_threshold: Minimum complementarity for hybridization.
            max_hybrids: Maximum hybrids per call to hybridize().
            implication_depth: Depth for MCTS implication traversal.
            composition_mode: How to compose hybrid statements.
                CONCATENATIVE: Fast "A ∘ B" (default).
                LLM_DRIVEN: Semantic composition via compose_callback.
            compose_callback: Required when composition_mode=LLM_DRIVEN.
                Signature: (parent_a_statement, parent_b_statement, evidence_context) -> composed_statement
                The callback should return a NEW statement that synthesizes
                both parents — not just concatenate them.
                Example: compose_callback("betrayed", "pride wounded", "emotional context")
                → "betrayal that wounded their pride — a pattern of trust violation"
        """
        if bridge is not None:
            self._bridge = bridge
        else:
            self._bridge = get_bridge()

        self.rsvs_available = self._bridge.is_available
        self.is_rust_core = self._bridge.is_rust_core
        self._complementarity_threshold = complementarity_threshold
        self._max_hybrids = max_hybrids
        self._implication_depth = implication_depth
        self._composition_mode = composition_mode
        self._compose_callback = compose_callback

        if composition_mode == CompositionMode.LLM_DRIVEN and compose_callback is None:
            logger.warning(
                "CompositionMode.LLM_DRIVEN requires a compose_callback. "
                "Falling back to CONCATENATIVE mode."
            )
            self._composition_mode = CompositionMode.CONCATENATIVE

    def hybridize(
        self,
        possibilities: list[dict],
        all_evidence: set[str] | None = None,
    ) -> list[HybridResult]:
        """Find complementary pairs and create hybrids.

        Args:
            possibilities: List of possibility dicts (must have 'id', 'statement',
                'confidence', 'explained_evidence' keys).
            all_evidence: All evidence in the current context.

        Returns:
            List of HybridResult objects.
        """
        all_evidence = all_evidence or set()
        hybrids: list[HybridResult] = []
        hybrid_count = 0

        for i, p_a in enumerate(possibilities):
            if hybrid_count >= self._max_hybrids:
                break
            for p_b in possibilities[i + 1:]:
                if hybrid_count >= self._max_hybrids:
                    break

                # Measure complementarity
                comp_score = self.measure_complementarity(p_a, p_b, all_evidence)

                if comp_score.score >= self._complementarity_threshold:
                    hybrid = self.compose(p_a, p_b, comp_score, all_evidence)
                    hybrids.append(hybrid)
                    hybrid_count += 1

        logger.debug(
            "HypothesisCombinator: %d possibilities → %d hybrids",
            len(possibilities), len(hybrids),
        )

        return hybrids

    def measure_complementarity(
        self,
        a: dict,
        b: dict,
        all_evidence: set[str],
    ) -> ComplementarityScore:
        """Measure how complementary two possibilities are.

        Two possibilities are complementary if:
        - They explain DIFFERENT parts of the evidence (low overlap)
        - Together they cover MORE evidence (high joint coverage)
        - Neither subsumes the other

        Returns:
            A ComplementarityScore with detailed metrics.
        """
        a_evidence = set(a.get("explained_evidence", []))
        b_evidence = set(b.get("explained_evidence", []))

        joint = a_evidence | b_evidence

        if not joint and not all_evidence:
            # No evidence — use structural similarity as proxy
            overlap_ratio = 0.5  # Unknown
            joint_coverage = 0.0

            if self.rsvs_available:
                try:
                    sim = self._bridge.structural_similarity(
                        str(a.get("statement", ""))[:50],
                        str(b.get("statement", ""))[:50],
                    )
                    if sim and isinstance(sim, dict):
                        sim_val = sim.get("structural_similarity", 0.5)
                        if isinstance(sim_val, (int, float)):
                            overlap_ratio = float(sim_val)
                except Exception:
                    pass
        else:
            overlap = a_evidence & b_evidence
            overlap_ratio = len(overlap) / max(len(joint), 1)
            joint_coverage = len(joint) / max(len(all_evidence), 1)

        # High complementarity = low overlap + high joint coverage
        score = joint_coverage * (1.0 - overlap_ratio)

        return ComplementarityScore(
            overlap_ratio=overlap_ratio,
            joint_coverage=joint_coverage,
            score=score,
        )

    def compose(
        self,
        a: dict,
        b: dict,
        comp_score: ComplementarityScore,
        all_evidence: set[str],
    ) -> HybridResult:
        """Compose two possibilities into a hybrid (A × B).

        The hybrid is NOT A ∧ B (conjunction), but A × B (composition):
        a new possibility that combines insights from both parents
        and may have emergent properties.

        Composition modes:
            CONCATENATIVE: "A ∘ B" — fast but shallow.
            LLM_DRIVEN: Semantic composition via callback — TRUE A × B.
                The LLM synthesizes a NEW statement that captures what
                NEITHER parent alone could express. For example:
                A="betrayed" + B="pride wounded"
                → "betrayal that wounded their pride"

        Confidence boost: since the hybrid covers more evidence
        than either parent alone, it gets a confidence boost
        proportional to the additional coverage AND complementarity.
        """
        a_id = str(a.get("id", a.get("possibility_id", uuid.uuid4().hex[:8])))
        b_id = str(b.get("id", b.get("possibility_id", uuid.uuid4().hex[:8])))
        a_stmt = str(a.get("statement", ""))
        b_stmt = str(b.get("statement", ""))
        a_conf = float(a.get("confidence", 0.5))
        b_conf = float(b.get("confidence", 0.5))

        a_evidence = set(a.get("explained_evidence", []))
        b_evidence = set(b.get("explained_evidence", []))
        combined_evidence = a_evidence | b_evidence

        # Confidence: minimum of parents * coverage boost * complementarity bonus
        # This is truly multiplicative (A × B), not additive
        base_conf = min(a_conf, b_conf)
        coverage_boost = 1.0 + (comp_score.joint_coverage * 0.15)
        complementarity_bonus = 1.0 + (comp_score.score * 0.1)
        hybrid_conf = min(0.95, base_conf * coverage_boost * complementarity_bonus)

        # Compose statement based on mode
        hybrid_stmt = self._compose_statement(
            a_stmt, b_stmt, combined_evidence,
        )

        # Trace implications — the CRUCIAL step
        implications = self._trace_implications(hybrid_stmt)

        return HybridResult(
            hybrid_id=uuid.uuid4().hex[:8],
            statement=hybrid_stmt,
            parent_a_id=a_id,
            parent_b_id=b_id,
            confidence=hybrid_conf,
            complementarity=comp_score.score,
            explained_evidence=combined_evidence,
            implications=implications,
        )

    def _compose_statement(
        self,
        a_stmt: str,
        b_stmt: str,
        combined_evidence: set[str],
    ) -> str:
        """Compose a hybrid statement from two parent statements.

        In CONCATENATIVE mode, this produces "A ∘ B".
        In LLM_DRIVEN mode, this calls the LLM callback to produce
        a NEW statement that truly synthesizes both parents' meaning.

        Args:
            a_stmt: First parent's statement.
            b_stmt: Second parent's statement.
            combined_evidence: Union of both parents' explained evidence.

        Returns:
            The composed hybrid statement.
        """
        if self._composition_mode == CompositionMode.LLM_DRIVEN and self._compose_callback is not None:
            try:
                evidence_context = ", ".join(sorted(combined_evidence)[:5]) if combined_evidence else ""
                composed = self._compose_callback(a_stmt, b_stmt, evidence_context)
                if composed and isinstance(composed, str) and len(composed.strip()) > 0:
                    return composed.strip()
                # If callback returns empty, fall through to concatenative
                logger.debug("compose_callback returned empty, falling back to concatenative")
            except Exception as exc:
                logger.debug("compose_callback failed: %s, falling back to concatenative", exc)

        # Fallback: concatenative composition
        return f"{a_stmt} ∘ {b_stmt}"

    def _trace_implications(self, hybrid_statement: str) -> list[str]:
        """Trace implications of a hybrid via RSVS MCTS.

        The CRUCIAL step: a hybrid may open doors that neither parent
        could see alone. Each implication is a NEW possibility.

        Enhanced tracing strategy:
        1. Traverse best_path from MCTS (direct implications)
        2. Explore scored_atoms for high-value but non-obvious connections
        3. Filter out nodes already in the hybrid statement (avoid trivial)
        4. Deduplicate by label
        """
        implications: list[str] = []
        seen_labels: set[str] = set()

        if not self.rsvs_available:
            return implications

        # Normalize hybrid keywords to avoid trivial self-references
        hybrid_words = set(hybrid_statement.lower().split())

        try:
            mcts_result = self._bridge.mcts_query(
                node_label=hybrid_statement[:100],
                max_depth=self._implication_depth,
                simulations=20,
            )
            if mcts_result and isinstance(mcts_result, dict):
                # Direct implications from best_path
                best_path = mcts_result.get("best_path", [])
                for node_label in best_path[:5]:
                    if isinstance(node_label, str):
                        norm = node_label.lower().strip()
                        if norm not in seen_labels and norm not in hybrid_words:
                            seen_labels.add(norm)
                            implications.append(node_label)

                # Deeper implications from scored_atoms
                scored_atoms = mcts_result.get("scored_atoms", [])
                for atom_entry in scored_atoms[:_DEFAULT_MAX_IMPLICATION_ATOMS]:
                    label = self._extract_label(atom_entry)
                    if label:
                        norm = label.lower().strip()
                        if norm not in seen_labels and norm not in hybrid_words:
                            seen_labels.add(norm)
                            implications.append(label)

        except Exception as exc:
            logger.debug("Implication tracing failed: %s", exc)

        return implications

    @staticmethod
    def _extract_label(entry: Any) -> str | None:
        """Extract a label from various RSVS result formats."""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            return str(entry[0])
        if isinstance(entry, dict):
            return str(entry.get("label", entry.get("node", "")))
        return None
