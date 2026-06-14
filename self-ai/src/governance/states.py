# @WHO:   self-ai/src/governance/states.py
# @WHAT:  Dual-axis governance states + compositional member types for UnderstandingNode
# @PART:  self-ai/governance
# @ENTRY: LifecycleState, EpistemicState, SeedPrimitive, UnderstandingMember, SeedScores

"""Governance States — the structural backbone for SELF's memory governance.

Origin:
    These types are inspired by AAM (AphantasicAbstractionModel)'s dual-axis
    governance system, adapted for SELF-AI's UnderstandingNode architecture.

    AAM separates two independent concerns about knowledge:
      - LIFECYCLE: How mature is this knowledge structurally?
      - EPISTEMIC: How credible is this knowledge epistemically?

    This separation enables states like "structurally complete but contradicted"
    which a single confidence score cannot express.

Vision:
    When SELF learns something new:
      - It starts as NEW (just created, not yet verified)
      - After one successful use → CANDIDATE (promising, needs confirmation)
      - After repeated verification → STABLE (reliable knowledge)
      - If user says "this is no longer relevant" → DEPRECATED (still exists,
        but no longer influences new answers — deactivate, don't delete)

    When SELF verifies its knowledge:
      - OBSERVED: Seen directly (from teaching/observation)
      - INFERRED: Deduced from other understandings
      - GROUNDED: Confirmed by independent evidence
      - CONTRADICTED: Conflicts with other evidence

    This enables the "burned by stove" vision:
      - The experience of being burned is STABLE+GROUNDED (verified, reliable)
      - It influences behavior UNCONSCIOUSLY (via UnconsciousInjector)
      - If asked "why are you careful?" → introspection surfaces the experience
      - If told "that stove is safe now" → DEPRECATED (still remembered, but
        no longer influences — the hands relax)
      - The experience is NEVER deleted — it can be reactivated if needed
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ═══════════════ LIFECYCLE STATE ═══════════════

class LifecycleState(str, Enum):
    """How mature is this understanding structurally?

    Transition rules:
        NEW → CANDIDATE:  After the understanding is used once in reasoning
        CANDIDATE → STABLE: After ≥3 successful applications
        STABLE → DEPRECATED: When user says irrelevant, or contradicted, or unused
        NEW → DEPRECATED: v36 — When temporal decay drops confidence below threshold
        CANDIDATE → DEPRECATED: v36 — When temporal decay drops confidence below threshold
        DEPRECATED → CANDIDATE: Reactivation — new evidence supports it

    A DEPRECATED understanding:
        - Still EXISTS in the graph (never deleted)
        - Is NOT retrieved for injection or reasoning
        - CAN be introspected (user asks "why did you used to think X?")
        - CAN be reactivated if new evidence emerges
    """
    NEW = "new"
    CANDIDATE = "candidate"
    STABLE = "stable"
    DEPRECATED = "deprecated"


# ═══════════════ EPISTEMIC STATE ═══════════════

class EpistemicState(str, Enum):
    """How credible is this understanding epistemically?

    Transition rules:
        OBSERVED → INFERRED: If composed from other understandings (not direct input)
        OBSERVED → GROUNDED: If confirmed by ≥2 independent sources
        INFERRED → GROUNDED: If independently verified
        * → CONTRADICTED: If conflicts with new evidence
        CONTRADICTED → GROUNDED: If contradiction is resolved (rare but possible)

    CONTRADICTED understandings:
        - Are flagged for re-evaluation
        - Are NOT injected unconsciously
        - Still exist in the graph for traceability
        - Their contradiction can be explained via introspection
    """
    OBSERVED = "observed"
    INFERRED = "inferred"
    GROUNDED = "grounded"
    CONTRADICTED = "contradicted"


# ═══════════════ SEED PRIMITIVES ═══════════════

class SeedPrimitive(str, Enum):
    """Epistemic confidence primitives — what dimensions matter for this understanding?

    Inspired by AAM's SeedPrimitive system. Each understanding has scores
    across 5 epistemic dimensions:

    - TRUST: How reliable is the source of this understanding?
    - RISK: How bad would it be if this understanding is wrong?
    - VALUE: How useful is this understanding for SELF's goals?
    - GOAL: How relevant is this to SELF's current active goals?
    - IDENTITY: How central is this to SELF's core knowledge?

    These are NOT just a single confidence number — they capture the
    multi-dimensional nature of epistemic judgment.
    """
    TRUST = "trust"
    RISK = "risk"
    VALUE = "value"
    GOAL = "goal"
    IDENTITY = "identity"


# ═══════════════ SEED SCORES ═══════════════

@dataclass
class SeedScores:
    """Multi-dimensional epistemic confidence scores.

    Each dimension ranges from 0.0 to 1.0:
        - 0.0 = minimum confidence in this dimension
        - 0.5 = neutral
        - 1.0 = maximum confidence in this dimension

    Example:
        An understanding about "signal_flip" (kata pengecualian membalik jawaban):
        - trust: 0.9 (learned from multiple confirmed examples)
        - risk: 0.1 (if wrong, the answer is just flipped — not catastrophic)
        - value: 0.85 (very useful for Indonesian language questions)
        - goal: 0.7 (relevant to SELF's goal of accurate question answering)
        - identity: 0.6 (part of SELF's core reasoning toolkit)
    """
    trust: float = 0.5
    risk: float = 0.3
    value: float = 0.5
    goal: float = 0.5
    identity: float = 0.3

    def to_dict(self) -> Dict[str, float]:
        return {
            'trust': self.trust,
            'risk': self.risk,
            'value': self.value,
            'goal': self.goal,
            'identity': self.identity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'SeedScores':
        if d is None:
            return cls()
        return cls(
            trust=d.get('trust', 0.5),
            risk=d.get('risk', 0.3),
            value=d.get('value', 0.5),
            goal=d.get('goal', 0.5),
            identity=d.get('identity', 0.3),
        )

    def overall(self) -> float:
        """Weighted average of all dimensions.

        Weights reflect the relative importance of each dimension
        for determining whether an understanding should influence behavior:
        - Trust is most important (if you can't trust it, don't use it)
        - Risk is inverted (high risk = should be more cautious)
        - Value, Goal, Identity are equally weighted
        """
        return (
            self.trust * 0.30 +
            (1.0 - self.risk) * 0.20 +
            self.value * 0.20 +
            self.goal * 0.15 +
            self.identity * 0.15
        )


# ═══════════════ UNDERSTANDING MEMBER ═══════════════

@dataclass
class UnderstandingMember:
    """A role-bearing member within an UnderstandingNode.

    Inspired by AAM's CompositionMember — each understanding can have
    structured roles that capture HOW its parts relate to each other.

    Without members, an understanding is a flat record:
        concept: "Kata pengecualian membalik jawaban"
        transformation: "IF 'kecuali' → jawaban = OPPOSITE"

    With members, it becomes a structured experience:
        Trigger:  "kata pengecualian (kecuali, selain, tidak)"
        Default:  "pernyataan utama sebelum pengecualian"
        Exception: "entitas setelah kata pengecualian"
        Result:   "jawaban = Exception (bukan Default)"

    This structure enables:
        - Per-role injection (inject only the Trigger when needed)
        - Gap detection ("this understanding is missing the Result role")
        - Composability (two understandings can link via matching roles)
        - Detailed introspection ("I was influenced by the Trigger role
          of the signal_flip understanding")

    Attributes:
        role: The semantic role this member plays (e.g., "trigger", "default",
              "exception", "result", "cause", "agent", "patient", "context")
        description: What fills this role in this specific understanding
        confidence: How confident SELF is that this role assignment is correct
        embedding: Optional per-member embedding for granular injection.
                   If None, the parent node's condition_embedding is used.
    """
    role: str
    description: str
    confidence: float = 0.8
    embedding: Optional[List[float]] = None

    def to_dict(self) -> dict:
        d = {
            'role': self.role,
            'description': self.description,
            'confidence': self.confidence,
        }
        if self.embedding is not None:
            d['embedding'] = self.embedding
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'UnderstandingMember':
        return cls(
            role=d.get('role', ''),
            description=d.get('description', ''),
            confidence=d.get('confidence', 0.8),
            embedding=d.get('embedding'),
        )


# ═══════════════ SEMANTIC ROLES ═══════════════

class SemanticRole(str, Enum):
    """Standardized role names for UnderstandingMembers.

    These are inspired by AAM's SemanticRole system but adapted for
    SELF-AI's transformation-oriented understanding model.

    Each role describes a PART that an understanding can have:
        - TRIGGER: What signal/condition activates this understanding
        - DEFAULT: What the normal/default answer would be
        - EXCEPTION: What overrides the default
        - RESULT: What the transformed answer is
        - CAUSE: Why this transformation applies
        - AGENT: Who/what performs the action
        - PATIENT: Who/what is affected
        - CONTEXT: The situational context
        - EVIDENCE: Supporting evidence for this understanding
        - COUNTEREXAMPLE: Cases where this understanding does NOT apply
    """
    TRIGGER = "trigger"
    DEFAULT = "default"
    EXCEPTION = "exception"
    RESULT = "result"
    CAUSE = "cause"
    AGENT = "agent"
    PATIENT = "patient"
    CONTEXT = "context"
    EVIDENCE = "evidence"
    COUNTEREXAMPLE = "counterexample"


# ═══════════════ LIFECYCLE TRANSITIONS ═══════════════

# Valid transitions: (from_state, to_state) pairs
LIFECYCLE_TRANSITIONS = {
    (LifecycleState.NEW, LifecycleState.CANDIDATE),
    (LifecycleState.CANDIDATE, LifecycleState.STABLE),
    (LifecycleState.STABLE, LifecycleState.DEPRECATED),
    (LifecycleState.DEPRECATED, LifecycleState.CANDIDATE),  # Reactivation
    # v36: Temporal decay — any non-deprecated state can transition to DEPRECATED
    # when confidence drops below threshold
    (LifecycleState.NEW, LifecycleState.DEPRECATED),
    (LifecycleState.CANDIDATE, LifecycleState.DEPRECATED),
    # Allow same-state (no-op)
    (LifecycleState.NEW, LifecycleState.NEW),
    (LifecycleState.CANDIDATE, LifecycleState.CANDIDATE),
    (LifecycleState.STABLE, LifecycleState.STABLE),
    (LifecycleState.DEPRECATED, LifecycleState.DEPRECATED),
}

# Valid epistemic transitions: (from_state, to_state) pairs
EPISTEMIC_TRANSITIONS = {
    (EpistemicState.OBSERVED, EpistemicState.INFERRED),
    (EpistemicState.OBSERVED, EpistemicState.GROUNDED),
    (EpistemicState.INFERRED, EpistemicState.GROUNDED),
    (EpistemicState.OBSERVED, EpistemicState.CONTRADICTED),
    (EpistemicState.INFERRED, EpistemicState.CONTRADICTED),
    (EpistemicState.GROUNDED, EpistemicState.CONTRADICTED),
    (EpistemicState.CONTRADICTED, EpistemicState.GROUNDED),  # Resolution
    # Allow same-state (no-op)
    (EpistemicState.OBSERVED, EpistemicState.OBSERVED),
    (EpistemicState.INFERRED, EpistemicState.INFERRED),
    (EpistemicState.GROUNDED, EpistemicState.GROUNDED),
    (EpistemicState.CONTRADICTED, EpistemicState.CONTRADICTED),
}


def can_transition_lifecycle(current: LifecycleState, target: LifecycleState) -> bool:
    """Check if a lifecycle transition is valid."""
    return (current, target) in LIFECYCLE_TRANSITIONS


def can_transition_epistemic(current: EpistemicState, target: EpistemicState) -> bool:
    """Check if an epistemic transition is valid."""
    return (current, target) in EPISTEMIC_TRANSITIONS
