"""
AAM Layer 0 — Perceptual Front-End Base

Filosofi Aphantasic: Input masuk → langsung diabstraksi ke structured tuples.
Tidak ada "foto" yang disimpan. Hanya relasi dan properti yang masuk ke graph.

Otak aphantasic tidak menyimpan gambar mental. Saat melihat apel, yang
tersimpan bukan pixel — tapi: "ini buah", "lebih bulat dari pir", "bisa dimakan".
Layer 0 meniru proses ini untuk setiap modality input.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union
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

    5-Pillar Enrichment (Gate 1: Signal Extraction):
        predictive_value: Estimated predictive value of this signal.
            Causal signal (0.8) > Temporal (0.7) > Comparative (0.6) > Factual (0.3).
            Used by SignalExtractionGate to filter noise from signal.
        signal_type: Classification of the signal type extracted from this tuple.
            Maps directly to the relation type for automated classification.
    """
    subject:         str
    relation_type:   RelationType
    predicate:       str
    dimension:       str | None = None
    direction:       str | None = None
    confidence:      float = 1.0
    source_modality: ModalityType = ModalityType.TEXT
    metadata:        Union[PerceptualTupleMeta, dict] = field(default_factory=PerceptualTupleMeta)

    # 5-Pillar: Gate 1 — Signal Extraction enrichment
    predictive_value: float = 0.5
    """Estimated predictive value: how much this signal helps predict future outcomes.
    Causal=0.8, Temporal=0.7, Comparative=0.6, Relational=0.5, Categorical=0.4, Factual=0.3.
    Set automatically by adapter.py when SignalExtractionGate is used."""

    signal_type: str = ""
    """Signal type classification from Gate 1.
    Populated by SignalExtractionGate: causal, temporal, comparative, relational, categorical, factual."""

    def __post_init__(self):
        """Ensure metadata is always PerceptualTupleMeta (backward compat).

        Also auto-populate signal_type and predictive_value from relation_type
        if they weren't explicitly set (5-Pillar Gate 1 enrichment).
        """
        if isinstance(self.metadata, dict):
            self.metadata = PerceptualTupleMeta.from_dict(self.metadata)

        # Auto-populate signal metadata from relation type if not explicitly set
        if not self.signal_type:
            self.signal_type = self._infer_signal_type()
        if self.predictive_value == 0.5 and self.signal_type:
            self.predictive_value = self._infer_predictive_value()

    def _infer_signal_type(self) -> str:
        """Infer signal type from relation type for 5-Pillar Gate 1."""
        mapping = {
            RelationType.CAUSAL: "causal",
            RelationType.TEMPORAL: "temporal",
            RelationType.DIFFERENTIAL: "comparative",
            RelationType.CATEGORICAL: "categorical",
            RelationType.FUNCTIONAL: "factual",
            RelationType.SPATIAL: "factual",
        }
        return mapping.get(self.relation_type, "factual")

    def _infer_predictive_value(self) -> float:
        """Infer predictive value from signal type for 5-Pillar Gate 1."""
        pv_map = {
            "causal": 0.8,
            "temporal": 0.7,
            "comparative": 0.6,
            "relational": 0.5,
            "categorical": 0.4,
            "factual": 0.3,
        }
        return pv_map.get(self.signal_type, 0.3) * self.confidence

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
