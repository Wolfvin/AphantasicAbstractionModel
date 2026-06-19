"""Neocortex package - slow learning, semantic memory.

Exports the four neocortical substructures for reasoning orchestration.
"""

import os as _os
import sys as _sys

# Sibling AGNN subpackages (engrams, hippocampus, ...) import each
# other as top-level modules (e.g. ``from engrams.episodic_engram
# import Episome``), not as ``AGNN.engrams...``. For those imports to
# resolve, the AGNN/ directory must be on sys.path. The existing test
# suite handles this by inserting _AGNP_ROOT before importing; when a
# caller imports us via ``from AGNN.neocortex...`` (the standalone
# use case in the Phase 3 brief), we do the same insertion here at
# package load time. This mirrors the bootstrap already present in
# AGNN/core.py and is idempotent.
_AGNP_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _AGNP_ROOT not in _sys.path:
    _sys.path.insert(0, _AGNP_ROOT)

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
