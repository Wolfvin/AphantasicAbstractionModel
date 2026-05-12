"""
AAM Layer 0 — Image Perceptual Abstractor

Status: STUB — belum diimplementasi.

Rencana implementasi:
  Model : CLIP atau LLaVA (vision-language model)
  Output: categorical + differential + spatial PerceptualTuple per objek
  Note  : Tidak menyimpan pixel — hanya PerceptualTuple hasil abstraksi.
          Gambar yang sama dilihat berkali-kali hanya update edge weight.
"""

from .base import BasePerceptualAbstractor, PerceptualObservation, ModalityType


class ImageAbstractor(BasePerceptualAbstractor):
    modality = ModalityType.IMAGE

    def abstract(self, raw_input, context={}):
        raise NotImplementedError(
            "ImageAbstractor belum diimplementasi. "
            "Rencana: CLIP/LLaVA → categorical + spatial PerceptualTuple."
        )
