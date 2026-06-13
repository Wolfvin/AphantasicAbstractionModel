# @WHO:   self-ai/src/concept/builder.py
# @WHAT:  Layer 3 — Concept Builder: menciptakan node internal dari pola berulang
# @PART:  concept
# @ENTRY: ConceptBuilder.process_difference(), ConceptBuilder.get_concepts()

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass

from src.translation.translator import NodeID
from src.difference.detector import DifferenceResult, DifferenceDetector
from config.thresholds import MIN_PATTERN_REPEAT, CONCEPT_CONFIDENCE_INIT


@dataclass
class Concept:
    """Konsep internal SELF — bukan kata manusia, bukan label luar."""
    id: int
    pattern_dims: List[int]           # dimensi yang mendefinisikan konsep ini
    pattern_vector: np.ndarray        # vektor rata-rata pola
    source_node_pairs: List[tuple]    # pasangan node yang membentuk pola ini
    occurrence_count: int = 0
    confidence: float = CONCEPT_CONFIDENCE_INIT


class ConceptBuilder:
    """
    Layer 3: Concept Builder — "Saya beri nama pada ini"

    Ketika pola perbedaan muncul berulang dan konsisten:
    SELF menciptakan node internal baru.

    Bukan kata manusia. Bukan label luar.
    Node #001, #002, #003 — bahasa yang dia ciptakan sendiri.

    Ini adalah "huruf" dari bahasa internalnya.
    """

    def __init__(self, diff_detector: DifferenceDetector):
        # @FLOW:     CONCEPT_INIT
        # @CALLS:    DifferenceDetector reference
        # @MUTATES:  none
        self.diff_detector = diff_detector
        self._concepts: Dict[int, Concept] = {}
        self._dim_pattern_tracker: Dict[tuple, int] = {}  # frozen dims → count
        self._concept_counter = 0
        self._pending_patterns: Dict[tuple, List[DifferenceResult]] = {}

    def process_difference(self, diff: DifferenceResult) -> Optional[Concept]:
        """
        @FLOW:     CONCEPT_PROCESS
        @CALLS:    DifferenceDetector.analyze()
        @MUTATES:  self._concepts, self._pending_patterns
        @BEHAVIOR: Memproses perbedaan baru untuk mendeteksi pola berulang.
                   Jika pola muncul >= MIN_PATTERN_REPEAT kali, buat konsep baru.
                   Mengembalikan Concept jika baru terbentuk, None jika belum cukup bukti.
        """
        # Buat key dari dimensi dominan (frozenset untuk order-independent)
        dim_key = tuple(sorted(diff.dominant_dims[:5]))  # top 5 dims sebagai signature

        if dim_key not in self._pending_patterns:
            self._pending_patterns[dim_key] = []

        self._pending_patterns[dim_key].append(diff)

        pattern_count = len(self._pending_patterns[dim_key])

        if pattern_count >= MIN_PATTERN_REPEAT:
            # Pola konsisten — buat konsep baru
            return self._form_concept(dim_key, self._pending_patterns[dim_key])
        else:
            return None

    def _form_concept(self, dim_key: tuple, diffs: List[DifferenceResult]) -> Concept:
        """
        @FLOW:     CONCEPT_FORM
        @CALLS:    none
        @MUTATES:  self._concepts, self._concept_counter
        @BEHAVIOR: Membentuk konsep baru dari kumpulan perbedaan yang konsisten.
                   Vektor konsep = rata-rata dari semua diff_vector yang membentuk pola.
        """
        self._concept_counter += 1

        # Rata-rata vektor perbedaan sebagai representasi konsep
        diff_vectors = [d.diff_vector for d in diffs]
        avg_vector = np.mean(diff_vectors, axis=0).astype(np.float32)

        # Normalisasi
        norm = np.linalg.norm(avg_vector)
        if norm > 0:
            avg_vector = avg_vector / norm

        concept = Concept(
            id=self._concept_counter,
            pattern_dims=list(dim_key),
            pattern_vector=avg_vector,
            occurrence_count=len(diffs),
            confidence=CONCEPT_CONFIDENCE_INIT + min(0.3, len(diffs) * 0.05),
            source_node_pairs=[(d.node_a.id, d.node_b.id) for d in diffs],
        )

        self._concepts[concept.id] = concept

        # Cleanup pending — sudah jadi konsep
        del self._pending_patterns[dim_key]

        return concept

    def get_concepts(self) -> List[Concept]:
        """Mengembalikan semua konsep yang sudah terbentuk."""
        return list(self._concepts.values())

    def get_concept(self, concept_id: int) -> Optional[Concept]:
        """Mengembalikan konsep berdasarkan ID."""
        return self._concepts.get(concept_id)
