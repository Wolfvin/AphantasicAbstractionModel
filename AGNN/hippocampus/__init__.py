"""Hippocampus package - fast encoding, episodic memory.

Exports the five hippocampal substructures used by the trisynaptic circuit.
"""

from .entorhinal_cortex import EntorhinalCortex
from .dentate_gyrus import DentateGyrus
from .ca3 import CA3
from .ca1 import CA1
from .subiculum import Subiculum

__all__ = ["EntorhinalCortex", "DentateGyrus", "CA3", "CA1", "Subiculum"]
