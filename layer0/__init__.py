"""
AAM Layer 0 — Perceptual Front-End

Raw world input → PerceptualObservation (structured relational tuples).
No raw data stored. Only relations enter the graph.

Modalities:
  text  : TextAbstractor  — LLM-driven with retry + fallback
  image : ImageAbstractor — LLM/vision bridge + fallback
  video : VideoAbstractor — frame sampling + temporal linking
  audio : AudioAbstractor — Whisper STT → TextAbstractor pipeline

Adapters:
  adapter : Layer 0 → Layer 1 bridge (PerceptualObservation → RSVS ingest)
"""

from .base import (
    BasePerceptualAbstractor,
    PerceptualObservation,
    PerceptualTuple,
    PerceptualTupleMeta,
    ModalityType,
    RelationType,
)
from .text  import TextAbstractor
from .image import ImageAbstractor
from .video import VideoAbstractor
from .audio import AudioAbstractor
from .adapter import (
    observation_to_ingest_data,
    observation_to_ingest_dicts,
    ingest_observation,
    ingest_observations,
    RsvsIngestProtocol,
    perceptual_tuple_to_v12_input,
    V12Adapter,
)
