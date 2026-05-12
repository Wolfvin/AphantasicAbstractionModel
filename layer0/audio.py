"""
AAM Layer 0 — Audio Perceptual Abstractor

Status: STUB — belum diimplementasi.

Rencana implementasi:
  - Speech: Whisper STT → TextAbstractor
  - Non-speech: audio feature extraction → temporal + causal tuples
  - Tidak menyimpan audio — hanya PerceptualTuple hasil abstraksi
"""

from .base import BasePerceptualAbstractor, PerceptualObservation, ModalityType


class AudioAbstractor(BasePerceptualAbstractor):
    modality = ModalityType.AUDIO

    def abstract(self, raw_input, context={}):
        raise NotImplementedError(
            "AudioAbstractor belum diimplementasi. "
            "Rencana: Whisper STT → TextAbstractor, atau audio features → temporal tuples."
        )
