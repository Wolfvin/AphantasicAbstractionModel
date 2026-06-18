"""Engrams package - memory representations.

Exports Episome (node), Semesome (edge), EngramComplex (graph wrapper).
"""

from .episodic_engram import Episome
from .semantic_engram import Semesome
from .engram_complex import EngramComplex

__all__ = ["Episome", "Semesome", "EngramComplex"]
