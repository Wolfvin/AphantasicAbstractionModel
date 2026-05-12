"""
AAM Layer 3 — Deductive Reasoning & Output

Traceable, auditable conclusions from Layer 2 graph state.

Modules:
  coder    : Code understanding as structured knowledge graph
  policy   : Rule-based compliance checking (tax, regulation)
  reasoning: Deductive chain builder + structured output
"""

__version__ = "1.2.0"

from .coder import CoderLayer, CodeElement, CodeAnalysisResult, DeductiveCoderLayer
from .policy import PolicyEngine, PolicyRule, PolicyViolation, DeductivePolicyEngine
from .reasoning import ReasoningEngine, DeductiveChain, DeductiveStep
