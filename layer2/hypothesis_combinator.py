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
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from layer2.bridge import RsvsBridge, get_bridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

_DEFAULT_COMPLEMENTARITY_THRESHOLD = 0.4
_DEFAULT_MAX_HYBRIDS = 30
_DEFAULT_IMPLICATION_DEPTH = 2


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
    ) -> None:
        if bridge is not None:
            self._bridge = bridge
        else:
            self._bridge = get_bridge()

        self.rsvs_available = self._bridge.is_available
        self.is_rust_core = self._bridge.is_rust_core
        self._complementarity_threshold = complementarity_threshold
        self._max_hybrids = max_hybrids
        self._implication_depth = implication_depth

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

        Confidence boost: since the hybrid covers more evidence
        than either parent alone, it gets a confidence boost
        proportional to the additional coverage.
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

        # Confidence: minimum of parents * coverage boost
        base_conf = min(a_conf, b_conf)
        coverage_boost = 1.0 + (comp_score.joint_coverage * 0.1)
        hybrid_conf = min(0.95, base_conf * coverage_boost)

        # Compose statement: A ∘ B
        hybrid_stmt = f"{a_stmt} ∘ {b_stmt}"

        # Trace implications
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

    def _trace_implications(self, hybrid_statement: str) -> list[str]:
        """Trace implications of a hybrid via RSVS MCTS.

        The CRUCIAL step: a hybrid may open doors that neither parent
        could see alone. Each implication is a NEW possibility.
        """
        implications: list[str] = []

        if not self.rsvs_available:
            return implications

        try:
            mcts_result = self._bridge.mcts_query(
                node_label=hybrid_statement[:100],
                max_depth=self._implication_depth,
                simulations=15,
            )
            if mcts_result and isinstance(mcts_result, dict):
                best_path = mcts_result.get("best_path", [])
                for node_label in best_path[:3]:
                    if isinstance(node_label, str):
                        implications.append(node_label)
        except Exception as exc:
            logger.debug("Implication tracing failed: %s", exc)

        return implications
