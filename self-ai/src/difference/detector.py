# @WHO:   self-ai/src/difference/detector.py
# @WHAT:  Layer 2 — Difference Detector: mendeteksi perbedaan antar node internal
# @PART:  difference
# @ENTRY: DifferenceDetector.compute_diff(), DifferenceDetector.analyze()

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.translation.translator import NodeID
from src.core.node_store import NodeStore


@dataclass
class DifferenceResult:
    """Hasil analisis perbedaan antar dua node."""
    node_a: NodeID
    node_b: NodeID
    diff_vector: np.ndarray       # vektor perbedaan (A - B)
    magnitude: float              # besarnya perbedaan
    dominant_dims: List[int]      # dimensi dengan perbedaan terbesar
    dominant_magnitudes: List[float]  # magnitudo per dimensi dominan


class DifferenceDetector:
    """
    Layer 2: Difference Detector — "Ini terasa berbeda"

    Bukan cosine similarity biasa.
    SELF belajar:
    — "perbedaan di dimensi ini tentang SESUATU"
    — "perbedaan di dimensi itu tentang HAL LAIN"

    Dia belum punya nama untuk itu.
    Tapi dia tahu polanya konsisten.
    """

    def __init__(self, node_store: NodeStore, top_k_dims: int = 10):
        # @FLOW:     DIFFERENCE_INIT
        # @CALLS:    NodeStore reference
        # @MUTATES:  none
        self.node_store = node_store
        self.top_k_dims = top_k_dims
        self._diff_history: List[DifferenceResult] = []

    def compute_diff(self, node_a: NodeID, node_b: NodeID) -> Optional[DifferenceResult]:
        """
        @FLOW:     DIFFERENCE_COMPUTE
        @CALLS:    NodeStore.get_vector()
        @MUTATES:  self._diff_history
        @BEHAVIOR: Menghitung perbedaan antara dua node internal.
                   Bukan hanya similarity — tapi dimensi SPAK mana yang membedakan.
                   Mengembalikan None jika salah satu node tidak ditemukan.
        """
        vec_a = self.node_store.get_vector(node_a)
        vec_b = self.node_store.get_vector(node_b)

        if vec_a is None or vec_b is None:
            return None

        diff_vector = vec_a - vec_b
        magnitude = float(np.linalg.norm(diff_vector))

        # Temukan dimensi dominan — dimensi mana yang paling membedakan
        abs_diff = np.abs(diff_vector)
        top_indices = np.argsort(abs_diff)[-self.top_k_dims:][::-1]
        dominant_dims = [int(i) for i in top_indices]
        dominant_magnitudes = [float(abs_diff[i]) for i in top_indices]

        result = DifferenceResult(
            node_a=node_a,
            node_b=node_b,
            diff_vector=diff_vector,
            magnitude=magnitude,
            dominant_dims=dominant_dims,
            dominant_magnitudes=dominant_magnitudes,
        )

        self._diff_history.append(result)
        return result

    def analyze(self, diff: DifferenceResult) -> dict:
        """
        @FLOW:     DIFFERENCE_ANALYZE
        @CALLS:    none
        @MUTATES:  none
        @BEHAVIOR: Menganalisis pola perbedaan dan mengembalikan insight.
                   Mengidentifikasi apakah pola ini konsisten dengan
                   perbedaan yang pernah ditemukan sebelumnya.
        """
        # Cek apakah dimensi dominan ini muncul berulang di history
        dim_frequency = {}
        for past_diff in self._diff_history:
            for dim in past_diff.dominant_dims:
                dim_frequency[dim] = dim_frequency.get(dim, 0) + 1

        recurring_dims = [
            dim for dim in diff.dominant_dims
            if dim_frequency.get(dim, 0) >= 2
        ]

        return {
            "magnitude": diff.magnitude,
            "dominant_dims": diff.dominant_dims,
            "recurring_dims": recurring_dims,
            "is_novel": len(recurring_dims) < len(diff.dominant_dims) // 2,
            "dim_frequency": dim_frequency,
        }
