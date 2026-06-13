# @WHO:   self-ai/src/core/node_store.py
# @WHAT:  Penyimpanan node internal SELF — basis dari semua layer
# @PART:  core
# @ENTRY: NodeStore.create(), NodeStore.find_nearest(), NodeStore.get()

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import sqlite3
import json
import os

from src.translation.translator import NodeID


@dataclass
class InternalNode:
    """Node internal SELF — bahasa yang SELF ciptakan sendiri."""
    id: int
    vector: np.ndarray          # representasi vektor (bukan raw embedding — sudah tertranslate)
    confidence: float = 0.5
    hit_count: int = 0          # berapa kali node ini diakses/referensikan
    created_at: int = 0


class NodeStore:
    """
    Penyimpanan semua node internal SELF.

    Bukan database kata manusia. Bukan label luar.
    Node #001, #002, #003 — bahasa yang SELF ciptakan sendiri.
    Ini adalah "huruf" dari bahasa internalnya.
    """

    def __init__(self, db_path: str = ":memory:"):
        # @FLOW:     NODE_STORE_INIT
        # @CALLS:    sqlite3.connect()
        # @MUTATES:  DB file (create tables jika belum ada)
        self._nodes: dict = {}  # id → InternalNode
        self._vectors = None    # numpy matrix untuk fast nearest-neighbor
        self._id_counter = 0
        self._dirty = True      # flag: perlu rebuild _vectors matrix

        # SQLite persistence (opsional)
        self._db_path = db_path
        if db_path != ":memory:":
            self._init_db(db_path)

    def _init_db(self, db_path: str):
        """Inisialisasi SQLite untuk persistent storage."""
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                vector BLOB NOT NULL,
                confidence REAL DEFAULT 0.5,
                hit_count INTEGER DEFAULT 0,
                created_at INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def create(self, embedding: np.ndarray) -> NodeID:
        """
        @FLOW:     NODE_CREATE
        @CALLS:    none (internal state mutation)
        @MUTATES:  self._nodes, self._vectors matrix (lazy rebuild)
        @BEHAVIOR: Membuat node internal baru dari embedding.
                   Embedding disimpan sebagai representasi vektor node.
                   ID bersifat auto-increment.
        """
        self._id_counter += 1
        node = InternalNode(
            id=self._id_counter,
            vector=embedding.astype(np.float32),
            confidence=0.5,
            hit_count=1,
        )
        self._nodes[node.id] = node
        self._dirty = True
        return NodeID(id=node.id, created_at=0)

    def find_nearest(self, embedding: np.ndarray) -> Tuple[NodeID, float]:
        """
        @FLOW:     NODE_FIND_NEAREST
        @CALLS:    numpy cosine similarity computation
        # @MUTATES:  none (read-only)
        @BEHAVIOR: Mencari node terdekat berdasarkan cosine similarity.
                   Mengembalikan (NodeID, similarity_score).
                   Jika store kosong, raise ValueError.
        """
        if not self._nodes:
            raise ValueError("NodeStore kosong — tidak ada node untuk dicari")

        self._rebuild_matrix_if_needed()

        # Cosine similarity: dot product (sudah normalized)
        query = embedding.astype(np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        mat_norm = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10)

        similarities = mat_norm @ query_norm
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        # Map matrix index ke node id
        node_ids = sorted(self._nodes.keys())
        node_id = node_ids[best_idx]

        # Update hit count
        self._nodes[node_id].hit_count += 1

        return NodeID(id=node_id), best_sim

    def get(self, node_id: NodeID) -> Optional[InternalNode]:
        """
        @FLOW:     NODE_GET
        @CALLS:    none
        @MUTATES:  none
        @BEHAVIOR: Mengembalikan InternalNode berdasarkan NodeID, atau None jika tidak ada.
        """
        return self._nodes.get(node_id.id)

    def get_vector(self, node_id: NodeID) -> Optional[np.ndarray]:
        """Mengembalikan vektor node berdasarkan NodeID."""
        node = self._nodes.get(node_id.id)
        return node.vector if node else None

    def update_confidence(self, node_id: NodeID, confidence: float):
        """Update confidence score node."""
        if node_id.id in self._nodes:
            self._nodes[node_id.id].confidence = confidence

    def _rebuild_matrix_if_needed(self):
        """Rebuild numpy matrix dari dict jika ada perubahan."""
        if not self._dirty and self._vectors is not None:
            return
        if not self._nodes:
            return
        node_ids = sorted(self._nodes.keys())
        self._vectors = np.stack([self._nodes[nid].vector for nid in node_ids])
        self._dirty = False

    def __len__(self) -> int:
        return len(self._nodes)

    def all_node_ids(self) -> List[NodeID]:
        """Mengembalikan semua NodeID yang ada di store."""
        return [NodeID(id=nid) for nid in sorted(self._nodes.keys())]

    def save_to_db(self):
        """Simpan semua node ke SQLite."""
        if self._db_path == ":memory:":
            return
        conn = sqlite3.connect(self._db_path)
        for node in self._nodes.values():
            vec_blob = node.vector.tobytes()
            conn.execute(
                "INSERT OR REPLACE INTO nodes (id, vector, confidence, hit_count, created_at) VALUES (?, ?, ?, ?, ?)",
                (node.id, vec_blob, node.confidence, node.hit_count, node.created_at)
            )
        conn.commit()
        conn.close()

    def load_from_db(self):
        """Load semua node dari SQLite."""
        if self._db_path == ":memory:":
            return
        if not os.path.exists(self._db_path):
            return
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT id, vector, confidence, hit_count, created_at FROM nodes ORDER BY id")
        for row in cursor:
            nid, vec_blob, confidence, hit_count, created_at = row
            vector = np.frombuffer(vec_blob, dtype=np.float32)
            node = InternalNode(
                id=nid,
                vector=vector,
                confidence=confidence,
                hit_count=hit_count,
                created_at=created_at,
            )
            self._nodes[nid] = node
            if nid > self._id_counter:
                self._id_counter = nid
        conn.close()
        self._dirty = True
