"""Neocortex package - slow learning, semantic memory.

Exports the four neocortical substructures for reasoning orchestration.
"""

from .prefrontal_cortex import PrefrontalCortex
from .inferior_frontal_gyrus import InferiorFrontalGyrus
from .dorsolateral_pfc import DorsolateralPFC
from .association_cortex import AssociationCortex

__all__ = [
    "PrefrontalCortex",
    "InferiorFrontalGyrus",
    "DorsolateralPFC",
    "AssociationCortex",
]
