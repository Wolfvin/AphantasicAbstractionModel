"""
AAM Layer 3 — Deductive Reasoning & Output

Traceable, auditable conclusions from Layer 2 graph state.

Modules:
  coder    : Code understanding as structured knowledge graph
  policy   : Rule-based compliance checking (tax, regulation)
  reasoning: Deductive chain builder + structured output (stub)
"""

__version__ = "1.0.0-alpha"

from .coder import CoderLayer, CodeElement, CodeAnalysisResult
from .policy import PolicyEngine, PolicyRule, PolicyViolation
