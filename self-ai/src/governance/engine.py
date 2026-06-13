# @WHO:   self-ai/src/governance/engine.py
# @WHAT:  GovernanceEngine — manages lifecycle/epistemic state transitions for UnderstandingNodes
# @PART:  self-ai/governance
# @ENTRY: GovernanceEngine

"""GovernanceEngine — the brain behind SELF's knowledge quality control.

Purpose:
    UnderstandingNodes in SELF's graph need governance — they shouldn't
    all be treated equally. Some are fresh and unverified, some are
    battle-tested and reliable, some are outdated and should be silenced.

    GovernanceEngine manages this lifecycle:

    1. PROMOTION: NEW → CANDIDATE → STABLE as understanding proves useful
    2. DEPRECATION: STABLE → DEPRECATED when understanding is no longer relevant
    3. REACTIVATION: DEPRECATED → CANDIDATE if new evidence emerges
    4. EPISTEMIC UPGRADE: OBSERVED → GROUNDED when independently verified
    5. CONTRADICTION: * → CONTRADICTED when evidence conflicts

    The KEY principle: DEACTIVATE, DON'T DELETE.
    Deprecated nodes remain in the graph for:
      - Introspection ("why did you used to think X?")
      - Reactivation ("actually, X is still relevant")
      - Traceability (every answer can be traced to its sources)

Usage:
    engine = GovernanceEngine(graph)
    engine.promote(node_id)          # NEW → CANDIDATE → STABLE
    engine.deactivate(node_id, reason="user said irrelevant")
    engine.contradict(node_id, evidence="new data conflicts")
    engine.reactivate(node_id, reason="new evidence supports")

    # Get only active nodes for injection
    active = engine.get_active_nodes()
"""

import time
import logging
from typing import Optional, List, Tuple, Dict

from governance.states import (
    LifecycleState,
    EpistemicState,
    SeedScores,
    UnderstandingMember,
    can_transition_lifecycle,
    can_transition_epistemic,
)

logger = logging.getLogger(__name__)


class GovernanceEngine:
    """Manages lifecycle and epistemic state transitions for UnderstandingNodes.

    This engine is the quality control system for SELF's knowledge graph.
    It ensures that:
      - New knowledge is verified before being fully trusted
      - Outdated knowledge is silenced but preserved
      - Contradictions are detected and flagged
      - Re-activation is possible when evidence changes

    Attributes:
        graph: The UnderstandingGraph to govern (set lazily)
        deprecation_threshold: Number of consecutive non-uses before auto-deprecation
        grounding_threshold: Number of independent confirmations needed for GROUNDED
    """

    def __init__(self, graph=None,
                 deprecation_threshold: int = 50,
                 grounding_threshold: int = 2):
        self._graph = graph
        self.deprecation_threshold = deprecation_threshold
        self.grounding_threshold = grounding_threshold

    @property
    def graph(self):
        """Lazy-access the understanding graph."""
        if self._graph is None:
            try:
                from derivation.understanding_builder import get_shared_graph
                self._graph = get_shared_graph()
            except Exception as e:
                logger.warning("GovernanceEngine cannot access graph: %s", e)
        return self._graph

    @graph.setter
    def graph(self, value):
        self._graph = value

    # ═══════════════ PROMOTION ═══════════════

    def promote(self, node_id: str) -> Optional[LifecycleState]:
        """Promote a node to the next lifecycle stage.

        Rules:
            NEW → CANDIDATE: Node has been applied at least once
            CANDIDATE → STABLE: Node has been correct ≥3 times

        Returns:
            New LifecycleState if transition happened, None if not.
        """
        node = self._get_node(node_id)
        if node is None:
            return None

        current = node.lifecycle

        if current == LifecycleState.NEW:
            # NEW → CANDIDATE if applied at least once
            if node.times_applied >= 1:
                return self._transition_lifecycle(node, LifecycleState.CANDIDATE)

        elif current == LifecycleState.CANDIDATE:
            # CANDIDATE → STABLE if correct ≥3 times
            if node.times_correct >= 3:
                return self._transition_lifecycle(node, LifecycleState.STABLE)

        elif current == LifecycleState.DEPRECATED:
            # Cannot promote from DEPRECATED — use reactivate()
            logger.debug("Cannot promote DEPRECATED node %s — use reactivate()", node_id)
            return None

        return None  # No transition

    def reactivate(self, node_id: str, reason: str = '') -> Optional[LifecycleState]:
        """Reactivate a DEPRECATED node — it becomes CANDIDATE again.

        This is the "actually, that stove IS still hot" moment.
        The experience was silenced, but new evidence brings it back.

        Args:
            node_id: The node to reactivate
            reason: Why it's being reactivated (for traceability)

        Returns:
            New LifecycleState if transition happened, None if not.
        """
        node = self._get_node(node_id)
        if node is None:
            return None

        if node.lifecycle != LifecycleState.DEPRECATED:
            logger.debug("Node %s is not DEPRECATED — cannot reactivate", node_id)
            return None

        # DEPRECATED → CANDIDATE
        new_state = self._transition_lifecycle(node, LifecycleState.CANDIDATE)
        if new_state is not None:
            node.deprecated_reason = None
            node.deprecated_at = None
            logger.info("REACTIVATED node %s: %s", node_id, reason or "new evidence")
        return new_state

    # ═══════════════ DEACTIVATION ═══════════════

    def deactivate(self, node_id: str, reason: str = '') -> Optional[LifecycleState]:
        """Deactivate a node — set it to DEPRECATED without deleting it.

        This is the core of the "deactivate, don't delete" principle.
        The node still exists in the graph, but:
          - It is NOT retrieved for unconscious injection
          - It is NOT used in reasoning
          - It CAN be introspected (if asked "why did you used to think X?")
          - It CAN be reactivated

        Args:
            node_id: The node to deactivate
            reason: Why it's being deactivated (for traceability)

        Returns:
            New LifecycleState if transition happened, None if not.
        """
        node = self._get_node(node_id)
        if node is None:
            return None

        if node.lifecycle == LifecycleState.DEPRECATED:
            logger.debug("Node %s is already DEPRECATED", node_id)
            return None

        new_state = self._transition_lifecycle(node, LifecycleState.DEPRECATED)
        if new_state is not None:
            node.deprecated_reason = reason or "deactivated by governance"
            node.deprecated_at = time.time()
            logger.info("DEACTIVATED node %s: %s", node_id, reason or "governance decision")
        return new_state

    # ═══════════════ EPISTEMIC TRANSITIONS ═══════════════

    def ground(self, node_id: str) -> Optional[EpistemicState]:
        """Upgrade epistemic state to GROUNDED — independently verified.

        Rules:
            OBSERVED → GROUNDED if confirmed by ≥2 independent sources
            INFERRED → GROUNDED if independently verified
        """
        node = self._get_node(node_id)
        if node is None:
            return None

        current = node.epistemic

        if current == EpistemicState.OBSERVED:
            # Check if we have enough independent confirmations
            # Simple heuristic: if times_correct >= grounding_threshold,
            # we consider it grounded
            if node.times_correct >= self.grounding_threshold:
                return self._transition_epistemic(node, EpistemicState.GROUNDED)

        elif current == EpistemicState.INFERRED:
            if node.times_correct >= self.grounding_threshold:
                return self._transition_epistemic(node, EpistemicState.GROUNDED)

        elif current == EpistemicState.CONTRADICTED:
            # Resolution — can go to GROUNDED if contradiction is resolved
            # This is rare and requires explicit confirmation
            return None  # Require explicit resolve_contradiction()

        return None

    def contradict(self, node_id: str, evidence: str = '') -> Optional[EpistemicState]:
        """Flag a node as CONTRADICTED — evidence conflicts with this understanding.

        A contradicted understanding:
          - Still exists in the graph
          - Is NOT injected unconsciously
          - Is flagged for re-evaluation
          - Can be introspected ("I used to think X, but new evidence contradicts it")

        Args:
            node_id: The node to flag
            evidence: What evidence contradicts it (for traceability)
        """
        node = self._get_node(node_id)
        if node is None:
            return None

        new_state = self._transition_epistemic(node, EpistemicState.CONTRADICTED)
        if new_state is not None:
            logger.info("CONTRADICTED node %s: %s", node_id, evidence or "evidence conflict")
            # Also lower confidence
            node.confidence = max(0.1, node.confidence * 0.5)
        return new_state

    def resolve_contradiction(self, node_id: str,
                               resolution: str = '') -> Optional[EpistemicState]:
        """Resolve a contradiction — move from CONTRADICTED to GROUNDED.

        This is rare and requires explicit action. The contradiction
        must be resolved (e.g., the conflicting evidence was wrong,
        or the understanding was refined).

        Args:
            node_id: The node to resolve
            resolution: How the contradiction was resolved
        """
        node = self._get_node(node_id)
        if node is None:
            return None

        if node.epistemic != EpistemicState.CONTRADICTED:
            return None

        new_state = self._transition_epistemic(node, EpistemicState.GROUNDED)
        if new_state is not None:
            logger.info("RESOLVED contradiction for node %s: %s",
                       node_id, resolution or "contradiction resolved")
        return new_state

    # ═══════════════ QUERY METHODS ═══════════════

    def get_active_nodes(self) -> List[Tuple]:
        """Get all nodes eligible for unconscious injection.

        Only nodes that are:
          - Lifecycle: CANDIDATE or STABLE (not NEW, not DEPRECATED)
          - Epistemic: not CONTRADICTED

        are considered active enough to influence behavior unconsciously.

        Returns:
            List of (UnderstandingNode, relevance_score) tuples.
            relevance_score is based on seed_scores.overall() * confidence.
        """
        if self.graph is None:
            return []

        active = []
        for node_id, node in self.graph._nodes.items():
            if not self.is_injectable(node):
                continue
            # Score: combination of seed scores and confidence
            score = node.seed_scores.overall() * node.confidence
            active.append((node, score))

        # Sort by score descending
        active.sort(key=lambda x: x[1], reverse=True)
        return active

    def is_injectable(self, node) -> bool:
        """Check if a node is eligible for unconscious injection.

        Rules:
            - Lifecycle must be CANDIDATE or STABLE
            - Epistemic must NOT be CONTRADICTED
            - Confidence must be above minimum threshold (0.2)

        NEW nodes are not injectable because they haven't been verified.
        DEPRECATED nodes are not injectable because they've been silenced.
        CONTRADICTED nodes are not injectable because they're flagged as wrong.
        """
        if node.lifecycle not in (LifecycleState.CANDIDATE, LifecycleState.STABLE):
            return False
        if node.epistemic == EpistemicState.CONTRADICTED:
            return False
        if node.confidence < 0.2:
            return False
        return True

    def detect_contradictions(self) -> List[Dict]:
        """Detect potential contradictions between active understandings.

        This is a simplified version of AAM's GovernBeliefs contradiction
        detection. We look for:
          - High-similarity nodes with different transformation kinds
          - Nodes with opposing member roles (e.g., same trigger, opposite result)

        Returns:
            List of dicts with keys: 'node_a', 'node_b', 'type', 'description'
        """
        if self.graph is None:
            return []

        contradictions = []
        active_nodes = [
            (nid, n) for nid, n in self.graph._nodes.items()
            if self.is_injectable(n)
        ]

        # Check pairwise for structural contradictions
        for i in range(len(active_nodes)):
            for j in range(i + 1, len(active_nodes)):
                nid_a, node_a = active_nodes[i]
                nid_b, node_b = active_nodes[j]

                # Check for trigger-result conflict
                # Same trigger but opposite result
                trigger_a = self._get_member_by_role(node_a, 'trigger')
                trigger_b = self._get_member_by_role(node_b, 'trigger')
                result_a = self._get_member_by_role(node_a, 'result')
                result_b = self._get_member_by_role(node_b, 'result')

                if (trigger_a and trigger_b and result_a and result_b and
                    trigger_a.description == trigger_b.description and
                    result_a.description != result_b.description):
                    contradictions.append({
                        'node_a': nid_a,
                        'node_b': nid_b,
                        'type': 'trigger_result_conflict',
                        'description': (
                            f"Same trigger '{trigger_a.description}' but "
                            f"different results: '{result_a.description}' vs '{result_b.description}'"
                        ),
                    })

        return contradictions

    def check_auto_deprecation(self) -> int:
        """Check for nodes that should be auto-deprecated.

        Auto-deprecation happens when a node hasn't been used in a long time.
        This is NOT deletion — the node is marked DEPRECATED.

        Currently a no-op placeholder — requires interaction count tracking
        that isn't yet implemented. Will be connected to the curiosity/usage
        tracking system in a future phase.

        Returns:
            Number of nodes auto-deprecated.
        """
        # Future: check usage count and deprecate stale nodes
        return 0

    # ═══════════════ INTERNAL ═══════════════

    def _get_node(self, node_id: str):
        """Get a node from the graph."""
        if self.graph is None:
            return None
        return self.graph._nodes.get(node_id)

    def _transition_lifecycle(self, node, target: LifecycleState) -> Optional[LifecycleState]:
        """Attempt a lifecycle transition on a node."""
        current = node.lifecycle
        if not can_transition_lifecycle(current, target):
            logger.debug(
                "Invalid lifecycle transition: %s → %s for node %s",
                current.value, target.value, node.id
            )
            return None

        node.lifecycle = target
        self._save_graph()
        return target

    def _transition_epistemic(self, node, target: EpistemicState) -> Optional[EpistemicState]:
        """Attempt an epistemic transition on a node."""
        current = node.epistemic
        if not can_transition_epistemic(current, target):
            logger.debug(
                "Invalid epistemic transition: %s → %s for node %s",
                current.value, target.value, node.id
            )
            return None

        node.epistemic = target
        self._save_graph()
        return target

    def _save_graph(self):
        """Persist the graph after a state change."""
        if self.graph is not None:
            try:
                self.graph._save()
            except Exception as e:
                logger.warning("Failed to save graph after governance transition: %s", e)

    @staticmethod
    def _get_member_by_role(node, role: str) -> Optional[UnderstandingMember]:
        """Get the first member with a given role from a node."""
        if not hasattr(node, 'members') or not node.members:
            return None
        for member in node.members:
            if member.role == role:
                return member
        return None
