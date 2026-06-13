# @WHO:   self-ai/src/governance/__init__.py
# @WHAT:  Governance module — dual-axis lifecycle + epistemic governance for understanding
# @PART:  self-ai/governance

"""Governance module — SELF's knowledge lifecycle and quality control.

This module implements the dual-axis governance system inspired by AAM:
  - LifecycleState: How mature is this knowledge? (NEW → CANDIDATE → STABLE → DEPRECATED)
  - EpistemicState: How credible is this knowledge? (OBSERVED → INFERRED → GROUNDED → CONTRADICTED)

Key principle: DEACTIVATE, DON'T DELETE.
When an understanding is no longer relevant, it becomes DEPRECATED.
It still exists in the graph for introspection and potential reactivation,
but it no longer influences new answers unconsciously.

This is the "burned by stove" vision:
  - The experience is stored structurally (UnderstandingMember roles)
  - It influences behavior unconsciously (UnconsciousInjector)
  - If asked why, it can be explained (Introspector)
  - If told it's no longer relevant, it's DEPRECATED, not deleted
"""

from governance.states import (
    LifecycleState,
    EpistemicState,
    SeedPrimitive,
    SeedScores,
    UnderstandingMember,
    SemanticRole,
    can_transition_lifecycle,
    can_transition_epistemic,
)
from governance.engine import GovernanceEngine

__all__ = [
    'LifecycleState',
    'EpistemicState',
    'SeedPrimitive',
    'SeedScores',
    'UnderstandingMember',
    'SemanticRole',
    'GovernanceEngine',
    'can_transition_lifecycle',
    'can_transition_epistemic',
]
