# @WHO:   self-ai/src/memory/filter.py
# @WHAT:  Layer 6 — Active Memory Filter: memory sebagai lensa yang membentuk interpretasi
# @PART:  memory
# @ENTRY: ActiveMemoryFilter.influence(), ActiveMemoryFilter.update_lens()

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from src.axiom.store import AxiomStore, Axiom
from src.core.node_store import NodeStore
from config.thresholds import GLOBAL_LENS_WEIGHT, MEMORY_TOP_K


class ActiveMemoryFilter:
    """
    Layer 6: Active Memory Filter — "Memory mempengaruhi"

    Memory bukan storage pasif.
    Memory adalah LENSA yang aktif membentuk interpretasi.

    Mekanisme: Cross-Attention
    keys   = relevant memory vectors
    values = memory vectors × lens_weight
    scores = new_input · keys.T → softmax → weighted sum

    output = new_input + (memory_influence × global_lens_weight)

    Input yang SAMA diinterpretasi BERBEDA
    tergantung memory yang sudah terbentuk.

    Dua SELF yang diajar berbeda = dua entitas yang berbeda.
    """

    def __init__(self, node_store: NodeStore, axiom_store: AxiomStore,
                 global_lens_weight: float = GLOBAL_LENS_WEIGHT):
        # @FLOW:     MEMORY_INIT
        # @CALLS:    NodeStore, AxiomStore references
        # @MUTATES:  none
        self.node_store = node_store
        self.axiom_store = axiom_store
        self.global_lens_weight = global_lens_weight

    def influence(self, new_input: np.ndarray) -> np.ndarray:
        """
        @FLOW:     MEMORY_INFLUENCE
        @CALLS:    AxiomStore.retrieve_relevant(), NodeStore.get_vector()
        @MUTATES:  none (pure transformation)
        @BEHAVIOR: Menggabungkan input baru dengan pengaruh memory melalui
                   cross-attention mechanism. Input yang SAMA bisa menghasilkan
                   output BERBEDA tergantung memory yang sudah terbentuk.
                   Jika tidak ada memory relevan, return input apa adanya.
        """
        # Ambil memory relevan
        relevant_axioms = self.axiom_store.retrieve_relevant(new_input, top_k=MEMORY_TOP_K)

        if not relevant_axioms:
            return new_input

        # Bangun keys dan values dari memory
        keys = []
        values = []
        for axiom in relevant_axioms:
            vec = self.node_store.get_vector(axiom.node_a)
            if vec is not None:
                keys.append(vec)
                values.append(vec * axiom.lens_weight)

        if not keys:
            return new_input

        keys_matrix = np.stack(keys)       # (k, d)
        values_matrix = np.stack(values)   # (k, d)

        # Cross-attention: softmax(new_input · keys.T)
        scores = new_input @ keys_matrix.T  # (k,)
        scores = self._softmax(scores)      # (k,)

        # Weighted sum: scores @ values
        memory_influence = scores @ values_matrix  # (d,)

        # Output = input + (memory_influence × global_lens_weight)
        output = new_input + (memory_influence * self.global_lens_weight)

        # Normalisasi output agar tetap dalam unit sphere
        norm = np.linalg.norm(output)
        if norm > 0:
            output = output / norm

        return output.astype(np.float32)

    def update_lens(self, axiom_id: int, new_lens_weight: float):
        """
        @FLOW:     MEMORY_UPDATE_LENS
        @CALLS:    AxiomStore
        @MUTATES:  Axiom lens_weight
        @BEHAVIOR: Mengupdate lens weight axiom tertentu.
                   Axiom yang sering diakses dan terbukti benar
                   mendapat lens_weight lebih tinggi — pengaruhnya lebih kuat.
        """
        axiom = self.axiom_store._axioms.get(axiom_id)
        if axiom is not None:
            axiom.lens_weight = new_lens_weight

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Stable softmax implementation."""
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-10)
