"""
AAM Layer 2 — Possibility Generator

Enumerates all possible interpretations from the RSVS graph
for use by the Possibility Lattice (Layer 3).

Generation angles:
1. Context-based: What does the graph say given current context?
2. Input-specific: What does the query itself suggest?
3. Cross-referential: What emerges from combining context + input?
4. Structural: What alternative structures exist in the graph?

For a mature AAM, this typically produces 50-150 possibilities
per query, because meaning is bounded and well-structured.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from layer2.bridge import RsvsBridge, get_bridge

logger = logging.getLogger(__name__)


@dataclass
class GeneratedPossibility:
    """A single generated possibility before entering the lattice.

    Attributes:
        id: Unique identifier.
        statement: What this possibility claims.
        source: How it was generated ("context", "input", "cross_ref", "structural").
        confidence: Initial confidence estimate.
        evidence_ids: Evidence items this possibility is based on.
        parent_query: The query that triggered this generation.
    """

    id: str
    statement: str
    source: str = "context"
    confidence: float = 0.4
    evidence_ids: set[str] = field(default_factory=set)
    parent_query: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "source": self.source,
            "confidence": round(self.confidence, 4),
            "evidence_ids": sorted(self.evidence_ids),
            "parent_query": self.parent_query[:100],
        }


class PossibilityGenerator:
    """Enumerate all possible interpretations from the RSVS graph.

    This is the GENERATE phase of the Possibility Lattice.
    It systematically enumerates possibilities from multiple angles,
    ensuring comprehensive coverage of the possibility space.

    For a mature AAM with well-structured RSVS graph, this produces
    ~50-150 possibilities per query. The number is bounded because:
    - Meaning is compositional → bounded combinatorics
    - Redundant interpretations merge → automatic dedup
    - Graph structure constrains the space → no wild guesses

    Attributes:
        rsvs_available: Whether a working RSVS instance is connected.
        is_rust_core: Whether the Rust core is being used.
    """

    def __init__(
        self,
        bridge: Optional[RsvsBridge] = None,
        max_possibilities: int = 150,
    ) -> None:
        if bridge is not None:
            self._bridge = bridge
        else:
            self._bridge = get_bridge()

        self.rsvs_available = self._bridge.is_available
        self.is_rust_core = self._bridge.is_rust_core
        self._max_possibilities = max_possibilities

    def generate(
        self,
        query: str,
        context: list[str] | None = None,
        evidence: list[str] | None = None,
    ) -> list[GeneratedPossibility]:
        """Generate all possibilities from the RSVS graph.

        Args:
            query: The query to generate possibilities for.
            context: Optional context atoms.
            evidence: Optional evidence to incorporate.

        Returns:
            A list of GeneratedPossibility objects (deduplicated).
        """
        context = context or []
        evidence = evidence or []
        all_possibilities: list[GeneratedPossibility] = []

        # Angle 1: Context-based
        if context:
            ctx_poss = self._from_context(query, context)
            all_possibilities.extend(ctx_poss)

        # Angle 2: Input-specific
        input_poss = self._from_input(query)
        all_possibilities.extend(input_poss)

        # Angle 3: Cross-referential
        if context and self.rsvs_available:
            cross_poss = self._from_cross_reference(query, context)
            all_possibilities.extend(cross_poss)

        # Angle 4: Structural alternatives
        if self.rsvs_available:
            struct_poss = self._from_structure(query)
            all_possibilities.extend(struct_poss)

        # Deduplicate
        all_possibilities = self._deduplicate(all_possibilities)

        # Cap
        all_possibilities = all_possibilities[:self._max_possibilities]

        logger.info(
            "PossibilityGenerator: %d possibilities for '%s'",
            len(all_possibilities), query[:50],
        )

        return all_possibilities

    def _from_context(
        self,
        query: str,
        context: list[str],
    ) -> list[GeneratedPossibility]:
        """Generate from context atoms via RSVS."""
        possibilities: list[GeneratedPossibility] = []

        for ctx_atom in context[:10]:
            if not self.rsvs_available:
                # Fallback: simple keyword association
                poss_id = uuid.uuid4().hex[:8]
                possibilities.append(GeneratedPossibility(
                    id=poss_id,
                    statement=f"'{ctx_atom}' relates to '{query}'",
                    source="context",
                    confidence=0.3,
                    evidence_ids={ctx_atom},
                    parent_query=query,
                ))
                continue

            # Use relate()
            try:
                result = self._bridge.relate(ctx_atom)
                if result and isinstance(result, dict):
                    related = result.get("related_nodes", [])
                    for entry in related[:5]:
                        label = self._extract_label(entry)
                        if label:
                            poss_id = uuid.uuid4().hex[:8]
                            possibilities.append(GeneratedPossibility(
                                id=poss_id,
                                statement=f"'{label}' is connected to '{query}' via '{ctx_atom}'",
                                source="context",
                                confidence=0.4,
                                evidence_ids={ctx_atom},
                                parent_query=query,
                            ))
            except Exception as exc:
                logger.debug("relate() generation failed: %s", exc)

            # Use senses()
            try:
                senses = self._bridge.senses(ctx_atom)
                if senses and isinstance(senses, list):
                    for sense in senses[:2]:
                        if isinstance(sense, dict):
                            gs = sense.get("grounding_score", 0.5)
                            poss_id = uuid.uuid4().hex[:8]
                            possibilities.append(GeneratedPossibility(
                                id=poss_id,
                                statement=f"Sense interpretation of '{ctx_atom}' for '{query}'",
                                source="context",
                                confidence=min(0.5, gs),
                                evidence_ids={ctx_atom},
                                parent_query=query,
                            ))
            except Exception as exc:
                logger.debug("senses() generation failed: %s", exc)

        return possibilities

    def _from_input(self, query: str) -> list[GeneratedPossibility]:
        """Generate from the query input itself."""
        possibilities: list[GeneratedPossibility] = []
        concepts = self._extract_concepts(query)

        for concept in concepts[:8]:
            poss_id = uuid.uuid4().hex[:8]
            possibilities.append(GeneratedPossibility(
                id=poss_id,
                statement=f"'{concept}' is central to '{query}'",
                source="input",
                confidence=0.5,
                evidence_ids={concept},
                parent_query=query,
            ))

            if self.rsvs_available:
                try:
                    senses = self._bridge.senses(concept)
                    if senses and isinstance(senses, list):
                        for i, sense in enumerate(senses[:2]):
                            if isinstance(sense, dict):
                                gs = sense.get("grounding_score", 0.5)
                                poss_id = uuid.uuid4().hex[:8]
                                possibilities.append(GeneratedPossibility(
                                    id=poss_id,
                                    statement=f"Sense {i} of '{concept}' interprets '{query}'",
                                    source="input",
                                    confidence=min(0.55, gs * 1.1),
                                    evidence_ids={concept},
                                    parent_query=query,
                                ))
                except Exception as exc:
                    logger.debug("senses() input generation failed: %s", exc)

        return possibilities

    def _from_cross_reference(
        self,
        query: str,
        context: list[str],
    ) -> list[GeneratedPossibility]:
        """Generate from cross-referencing context and input via MCTS."""
        possibilities: list[GeneratedPossibility] = []

        try:
            mcts_result = self._bridge.mcts_query(
                node_label=query[:100],
                max_depth=3,
                simulations=30,
            )
            if mcts_result and isinstance(mcts_result, dict):
                best_path = mcts_result.get("best_path", [])
                for node_label in best_path[:5]:
                    if isinstance(node_label, str):
                        poss_id = uuid.uuid4().hex[:8]
                        possibilities.append(GeneratedPossibility(
                            id=poss_id,
                            statement=f"MCTS path: '{node_label}' connects to '{query[:50]}'",
                            source="cross_ref",
                            confidence=0.45,
                            evidence_ids=set(context[:3]),
                            parent_query=query,
                        ))

                scored = mcts_result.get("scored_atoms", [])
                for entry in scored[:5]:
                    label = self._extract_label(entry)
                    if label:
                        poss_id = uuid.uuid4().hex[:8]
                        possibilities.append(GeneratedPossibility(
                            id=poss_id,
                            statement=f"MCTS scored: '{label}' is relevant to '{query[:50]}'",
                            source="cross_ref",
                            confidence=0.4,
                            evidence_ids=set(),
                            parent_query=query,
                        ))
        except Exception as exc:
            logger.debug("mcts_query() cross-ref generation failed: %s", exc)

        return possibilities

    def _from_structure(self, query: str) -> list[GeneratedPossibility]:
        """Generate from structural alternatives in the RSVS graph."""
        possibilities: list[GeneratedPossibility] = []

        # Use appraise() to find alternative viewpoints
        try:
            appraise_result = self._bridge.appraise(query)
            if appraise_result and isinstance(appraise_result, dict):
                agree_pct = appraise_result.get("agree_pct", 0)
                disagree_pct = appraise_result.get("disagree_pct", 0)
                if isinstance(disagree_pct, (int, float)) and float(disagree_pct) > 0.3:
                    poss_id = uuid.uuid4().hex[:8]
                    possibilities.append(GeneratedPossibility(
                        id=poss_id,
                        statement=f"Alternative view: '{query}' has significant disagreement",
                        source="structural",
                        confidence=0.35,
                        evidence_ids=set(),
                        parent_query=query,
                    ))
        except Exception as exc:
            logger.debug("appraise() structural generation failed: %s", exc)

        return possibilities

    # ==================================================================
    # Utility methods
    # ==================================================================

    @staticmethod
    def _extract_concepts(text: str) -> list[str]:
        """Extract key concepts from text."""
        stop = {
            "that", "this", "with", "from", "have", "been", "they",
            "their", "which", "would", "there", "could", "about",
            "yang", "dan", "dari", "untuk", "dengan", "adalah", "itu",
            "ini", "ke", "di", "pada", "tidak", "akan", "telah", "oleh",
        }
        words = [w.strip() for w in text.split() if len(w.strip()) > 2]
        return [w for w in words if w.lower() not in stop][:15]

    @staticmethod
    def _extract_label(entry: Any) -> str | None:
        """Extract a label from RSVS result entry."""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            return str(entry[0])
        if isinstance(entry, dict):
            return str(entry.get("label", entry.get("node", "")))
        return None

    def _deduplicate(
        self,
        possibilities: list[GeneratedPossibility],
    ) -> list[GeneratedPossibility]:
        """Deduplicate by statement similarity."""
        unique: list[GeneratedPossibility] = []
        seen: set[str] = set()

        for poss in possibilities:
            sig = poss.statement[:60].lower().strip()
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(poss)

        return unique
