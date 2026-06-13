# @WHO:   self-ai/src/translation/translator.py
# @WHAT:  Mekanisme translate_to_node() dan translate_to_human() — bahasa internal SELF
# @PART:  translation
# @ENTRY: InternalTranslator.translate_to_node(), InternalTranslator.translate_to_human()

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field

from config.thresholds import IDENTITY_THRESHOLD, MINIMUM_NODES


@dataclass
class NodeID:
    """Identifier unik untuk node internal SELF."""
    id: int
    created_at: int = 0  # timestamp sederhana


class InternalTranslator:
    """
    Bahasa internal SELF.

    Semua yang masuk ke dalam SELF diterjemahkan dulu ke node internal.
    Tidak ada pengecualian.

    Manusia → [translate_to_node()] → Node Internal → semua layer
    Manusia ← [translate_to_human()] ←─────────────── output
    """

    def __init__(self, node_store: 'NodeStore'):
        # @FLOW:     TRANSLATION_INIT
        # @CALLS:    NodeStore reference
        # @MUTATES:  none
        self.node_store = node_store
        self._human_cache = {}  # NodeID → deskripsi manusia (hanya untuk output)

    def translate_to_node(self, embedding: np.ndarray) -> NodeID:
        """
        @FLOW:     TRANSLATE_TO_NODE
        @CALLS:    NodeStore.find_nearest(), NodeStore.create()
        @MUTATES:  NodeStore (bisa menambah node baru)
        @BEHAVIOR: Cold start: selalu buat node baru sampai MINIMUM_NODES tercapai.
                   Jika embedding mirip node yang ada (>= IDENTITY_THRESHOLD), merge.
                   Jika berbeda, buat node baru. Threshold identitas menentukan
                   seberapa "ketat" SELF menganggap dua hal sama — ini membuat
                   setiap SELF unik berdasarkan pengalamannya.
        """
        if len(self.node_store) < MINIMUM_NODES:
            return self.node_store.create(embedding)

        nearest_node, similarity = self.node_store.find_nearest(embedding)

        if similarity >= IDENTITY_THRESHOLD:
            # Merge — konsep ini sudah ada, cukup referensikan
            return nearest_node
        else:
            # Konsep baru — SELF belajar sesuatu yang belum dia kenal
            return self.node_store.create(embedding)

    def translate_to_human(self, node_id: NodeID, description: Optional[str] = None) -> str:
        """
        @FLOW:     TRANSLATE_TO_HUMAN
        @CALLS:    NodeStore.get()
        @MUTATES:  self._human_cache (cache deskripsi untuk output)
        @BEHAVIOR: Mengembalikan deskripsi manusia untuk node internal.
                   Jika tidak ada deskripsi, mengembalikan representasi generik.
                   Deskripsi hanya disimpan untuk keperluan output — tidak
                   pernah digunakan di internal processing.
        """
        if description is not None:
            self._human_cache[node_id.id] = description

        if node_id.id in self._human_cache:
            return self._human_cache[node_id.id]

        return f"Node#{node_id.id:04d}"
