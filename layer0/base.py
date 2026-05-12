"""
AAM Layer 0 — Perceptual Front-End Base

Filosofi Aphantasic: Input masuk → langsung diabstraksi ke structured tuples.
Tidak ada "foto" yang disimpan. Hanya relasi dan properti yang masuk ke graph.

Otak aphantasic tidak menyimpan gambar mental. Saat melihat apel, yang
tersimpan bukan pixel — tapi: "ini buah", "lebih bulat dari pir", "bisa dimakan".
Layer 0 meniru proses ini untuk setiap modality input.
"""

from dataclasses import dataclass, field
from typing import Any, Union
from enum import Enum


class ModalityType(Enum):
    TEXT  = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class RelationType(Enum):
    CATEGORICAL  = "categorical"   # "ini adalah buah"
    DIFFERENTIAL = "differential"  # "lebih bulat dari pir"
    FUNCTIONAL   = "functional"    # "bisa dimakan", "tumbuh di pohon"
    SPATIAL      = "spatial"       # "di atas meja", "di kiri pintu"
    TEMPORAL     = "temporal"      # "terjadi sebelum X", "berlangsung 5 detik"
    CAUSAL       = "causal"        # "menyebabkan Y", "akibat dari Z"


# ---------------------------------------------------------------------------
# L0-06: Structured metadata for PerceptualTuple
# ---------------------------------------------------------------------------

@dataclass
class PerceptualTupleMeta:
    """
    Structured metadata contract for PerceptualTuple.

    Provides a typed schema instead of an opaque dict, ensuring every tuple
    carries provenance information. Backward-compatible: accepts plain dict
    input via from_dict().
    """
    source_url: str = ""
    extraction_model: str = ""
    extraction_timestamp: float = 0.0

    # Allow additional fields beyond the required ones
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for Rust interop / JSON)."""
        result = {
            "source_url": self.source_url,
            "extraction_model": self.extraction_model,
            "extraction_timestamp": self.extraction_timestamp,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "PerceptualTupleMeta":
        """Create from a plain dict — backward compatible with old metadata."""
        if isinstance(d, PerceptualTupleMeta):
            return d
        return cls(
            source_url=d.get("source_url", ""),
            extraction_model=d.get("extraction_model", ""),
            extraction_timestamp=d.get("extraction_timestamp", 0.0),
            extra={k: v for k, v in d.items()
                   if k not in ("source_url", "extraction_model", "extraction_timestamp")},
        )


@dataclass
class PerceptualTuple:
    """
    Unit terkecil dari abstraksi perceptual.
    Bukan piksel, bukan token — satu relasi yang bisa masuk ke graph.

    Contoh:
        subject="apel", relation_type=DIFFERENTIAL, predicate="pir",
        dimension="bentuk", direction="lebih_bulat", confidence=0.85
    """
    subject:         str
    relation_type:   RelationType
    predicate:       str
    dimension:       str | None = None
    direction:       str | None = None
    confidence:      float = 1.0
    source_modality: ModalityType = ModalityType.TEXT
    metadata:        Union[PerceptualTupleMeta, dict] = field(default_factory=PerceptualTupleMeta)

    def __post_init__(self):
        """Ensure metadata is always PerceptualTupleMeta (backward compat)."""
        if isinstance(self.metadata, dict):
            self.metadata = PerceptualTupleMeta.from_dict(self.metadata)

    def get_metadata_dict(self) -> dict:
        """Get metadata as a plain dict regardless of internal type."""
        if isinstance(self.metadata, PerceptualTupleMeta):
            return self.metadata.to_dict()
        return self.metadata


@dataclass
class PerceptualObservation:
    """
    Hasil abstraksi dari satu input.
    Kumpulan PerceptualTuple siap di-ingest ke Layer 1 (RSVS graph).

    raw_input_ref: HANYA referensi (hash/path/url) — bukan konten aslinya.
    Enforces the aphantasic principle: kita tidak menyimpan "foto".
    """
    modality:      ModalityType
    raw_input_ref: str
    tuples:        list[PerceptualTuple]
    context:       dict = field(default_factory=dict)
    timestamp:     str = ""


class BasePerceptualAbstractor:
    """
    Base class untuk semua Layer 0 abstractors.
    Subclass harus implement method abstract().
    """
    modality: ModalityType = NotImplemented

    def abstract(self, raw_input: Any, context: Optional[dict] = None) -> PerceptualObservation:
        """
        Terima raw input, kembalikan PerceptualObservation.
        TIDAK menyimpan raw input — hanya structured tuples.
        """
        raise NotImplementedError

    def _make_categorical(self, subject: str, category: str, confidence: float = 1.0) -> PerceptualTuple:
        return PerceptualTuple(subject=subject, relation_type=RelationType.CATEGORICAL,
                               predicate=category, confidence=confidence,
                               source_modality=self.modality)

    def _make_differential(self, subject: str, compared_to: str, dimension: str,
                           direction: str, confidence: float = 1.0) -> PerceptualTuple:
        return PerceptualTuple(subject=subject, relation_type=RelationType.DIFFERENTIAL,
                               predicate=compared_to, dimension=dimension, direction=direction,
                               confidence=confidence, source_modality=self.modality)

    def _make_functional(self, subject: str, function: str, confidence: float = 1.0) -> PerceptualTuple:
        return PerceptualTuple(subject=subject, relation_type=RelationType.FUNCTIONAL,
                               predicate=function, confidence=confidence,
                               source_modality=self.modality)

    def _make_spatial(self, subject: str, location: str, confidence: float = 1.0) -> PerceptualTuple:
        return PerceptualTuple(subject=subject, relation_type=RelationType.SPATIAL,
                               predicate=location, confidence=confidence,
                               source_modality=self.modality)

    def _make_temporal(self, subject: str, relation: str, confidence: float = 1.0) -> PerceptualTuple:
        return PerceptualTuple(subject=subject, relation_type=RelationType.TEMPORAL,
                               predicate=relation, confidence=confidence,
                               source_modality=self.modality)

    def _make_causal(self, subject: str, effect: str, confidence: float = 1.0) -> PerceptualTuple:
        return PerceptualTuple(subject=subject, relation_type=RelationType.CAUSAL,
                               predicate=effect, confidence=confidence,
                               source_modality=self.modality)
