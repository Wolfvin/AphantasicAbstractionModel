"""
AAM Layer 0 — Video Perceptual Abstractor

Status: STUB — belum diimplementasi.

Rencana implementasi:
  - Frame sampling (misal: 1 frame/detik)
  - Per-frame: ImageAbstractor → spatial + categorical tuples
  - Antar-frame: temporal tuples ("objek X bergerak ke Y")
  - Audio track: AudioAbstractor jika ada narasi/dialog
  - Tidak menyimpan frame — hanya PerceptualTuple per sampled frame
"""

from .base import BasePerceptualAbstractor, PerceptualObservation, ModalityType


class VideoAbstractor(BasePerceptualAbstractor):
    modality = ModalityType.VIDEO

    def abstract(self, raw_input, context={}):
        raise NotImplementedError(
            "VideoAbstractor belum diimplementasi. "
            "Rencana: frame sampling → ImageAbstractor + temporal tuples."
        )
