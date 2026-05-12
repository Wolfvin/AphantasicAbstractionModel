"""
AAM Layer 0 — Perceptual Front-End

Raw world input → PerceptualObservation (structured relational tuples).
No raw data stored. Only relations enter the graph.

Modalities:
  text  : TextAbstractor  — functional (stub without LLM)
  image : ImageAbstractor — stub (planned: CLIP/LLaVA)
  video : VideoAbstractor — stub (planned: frame sampling)
  audio : AudioAbstractor — stub (planned: Whisper)
"""

from .base import (
    BasePerceptualAbstractor,
    PerceptualObservation,
    PerceptualTuple,
    ModalityType,
    RelationType,
)
from .text  import TextAbstractor
from .image import ImageAbstractor
from .video import VideoAbstractor
from .audio import AudioAbstractor
