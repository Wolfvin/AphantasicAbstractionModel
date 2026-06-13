# @WHO:   self-ai/src/axiom/store.py
# @WHAT:  Layer 5 — Axiom Store: menyimpan kebenaran yang SELF yakini + quantities
# @PART:  axiom
# @ENTRY: AxiomStore.store(), AxiomStore.retrieve_relevant(), AxiomStore.get_all()
#
# v7: QUANTITY-AWARE AXIOMS
#
# Axiom sekarang bisa menyimpan metadata termasuk kuantitas.
# Ini memungkinkan SELF tidak hanya menyimpan relasi simbolik
# tapi juga operational semantics — "berapa banyak", "dari mana",
# "tersisa berapa".
#
# Dari metadata kuantitas inilah SELF bisa DISCOVER bahwa
# "makan" itu SUBTRACT, "ditambah" itu ADD, dll.
# Semua EMERGENT — tidak ada hardcoded operational rules.

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import sqlite3
import os
import json

from src.translation.translator import NodeID
from src.core.node_store import NodeStore
from config.thresholds import AXIOM_MIN_CONFIDENCE, LENS_WEIGHT_DEFAULT


@dataclass
class Axiom:
    """
    Axiom — kebenaran yang SELF yakini.

    (Node#A, Node#rel, Node#B, confidence, source)

    source = "autonomous" | "teaching" | "derived"
    → audit trail: dari mana SELF belajar ini?

    v7: metadata field menyimpan kuantitas dan info lain.
    metadata["quantities"] = [
        {"value": 4, "unit": "apel", "role": "consumed"},
        {"value": 10, "unit": "apel", "role": "total"},
    ]

    RAW EMBEDDING DIBUANG SETELAH TRANSLATE.
    Yang disimpan hanya struktur — bukan foto.
    """
    id: int
    node_a: NodeID
    relation: NodeID
    node_b: NodeID
    confidence: float
    source: str                # "autonomous" | "teaching" | "derived"
    lens_weight: float = LENS_WEIGHT_DEFAULT
    flag: Optional[str] = None  # None | "uncertain"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata keys:
    #   "quantities": List[Dict] — [{"value": 4.0, "unit": "apel", "role": "consumed"}]
    #   "operational_schema": str — "SUBTRACT" jika axiom ini mengkonfirmasi schema
    #   Any other key-value pairs


class AxiomStore:
    """
    Layer 5: Axiom Store — "Kebenaran yang saya yakini"

    Semua axiom disimpan sebagai NODE INTERNAL.
    Tidak ada label manusia di dalam.

    Inilah memory aphantastic:
    Tidak menyimpan gambar. Menyimpan makna.

    v7: Axiom bisa menyimpan kuantitas — memungkinkan
    SELF belajar operational semantics dari pengamatan.
    """

    def __init__(self, node_store: NodeStore, db_path: str = ":memory:"):
        # @FLOW:     AXIOM_STORE_INIT
        # @CALLS:    NodeStore reference, sqlite3
        # @MUTATES:  DB file
        self.node_store = node_store
        self._axioms: dict = {}  # id → Axiom
        self._id_counter = 0
        self._db_path = db_path

        if db_path != ":memory:":
            self._init_db(db_path)

    def _init_db(self, db_path: str):
        """Inisialisasi SQLite untuk persistent axiom storage."""
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS axioms (
                id INTEGER PRIMARY KEY,
                node_a_id INTEGER NOT NULL,
                relation_id INTEGER NOT NULL,
                node_b_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                lens_weight REAL DEFAULT 0.3,
                flag TEXT,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

    def store(self, node_a: NodeID, relation: NodeID, node_b: NodeID,
              confidence: float, source: str, flag: Optional[str] = None,
              metadata: Optional[Dict[str, Any]] = None) -> Axiom:
        """
        @FLOW:     AXIOM_STORE
        @CALLS:    NodeStore reference check
        @MUTATES:  self._axioms, self._id_counter
        @BEHAVIOR: Menyimpan axiom baru jika confidence >= AXIOM_MIN_CONFIDENCE.
                   Jika confidence terlalu rendah, axiom tidak disimpan (dibuang).
                   Source wajib: "autonomous", "teaching", atau "derived".
                   Flag "uncertain" ditambahkan jika ada konflik parsial.
                   v7: metadata menyimpan kuantitas dan info operational.
        """
        if confidence < AXIOM_MIN_CONFIDENCE:
            # Confidence terlalu rendah — buang
            return None

        self._id_counter += 1

        axiom = Axiom(
            id=self._id_counter,
            node_a=node_a,
            relation=relation,
            node_b=node_b,
            confidence=confidence,
            source=source,
            lens_weight=LENS_WEIGHT_DEFAULT,
            flag=flag,
            metadata=metadata or {},
        )

        self._axioms[axiom.id] = axiom
        return axiom

    def update_metadata(self, axiom_id: int, key: str, value: Any):
        """Update metadata field pada axiom tertentu."""
        if axiom_id in self._axioms:
            self._axioms[axiom_id].metadata[key] = value

    def get_quantitative_axioms(self) -> List[Axiom]:
        """Mengembalikan axiom yang punya kuantitas metadata."""
        return [a for a in self._axioms.values() if a.metadata.get("quantities")]

    def retrieve_relevant(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Axiom]:
        """
        @FLOW:     AXIOM_RETRIEVE
        @CALLS:    NodeStore.get_vector(), cosine similarity
        @MUTATES:  none
        @BEHAVIOR: Mengambil axiom yang relevan dengan query berdasarkan
                   cosine similarity node_a atau node_b terhadap query.
                   Mengembalikan top_k axiom paling relevan.
        """
        if not self._axioms:
            return []

        scored_axioms = []
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        for axiom in self._axioms.values():
            # Hitung similarity dengan node_a dan node_b
            vec_a = self.node_store.get_vector(axiom.node_a)
            vec_b = self.node_store.get_vector(axiom.node_b)

            max_sim = 0.0
            for vec in [vec_a, vec_b]:
                if vec is not None:
                    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
                    sim = float(np.dot(query_norm, vec_norm))
                    max_sim = max(max_sim, sim)

            # Weighted by confidence
            scored_axioms.append((max_sim * axiom.confidence, axiom))

        # Sort by score descending
        scored_axioms.sort(key=lambda x: x[0], reverse=True)
        return [axiom for _, axiom in scored_axioms[:top_k]]

    def get_all(self) -> List[Axiom]:
        """Mengembalikan semua axiom di store."""
        return list(self._axioms.values())

    def get_by_source(self, source: str) -> List[Axiom]:
        """Mengembalikan axiom berdasarkan source."""
        return [a for a in self._axioms.values() if a.source == source]

    def get_uncertain(self) -> List[Axiom]:
        """Mengembalikan axiom yang berflag uncertain."""
        return [a for a in self._axioms.values() if a.flag == "uncertain"]

    def save_to_db(self):
        """Simpan semua axiom ke SQLite."""
        if self._db_path == ":memory:":
            return
        conn = sqlite3.connect(self._db_path)
        for axiom in self._axioms.values():
            metadata_json = json.dumps(axiom.metadata) if axiom.metadata else None
            conn.execute(
                """INSERT OR REPLACE INTO axioms
                   (id, node_a_id, relation_id, node_b_id, confidence, source, lens_weight, flag, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (axiom.id, axiom.node_a.id, axiom.relation.id, axiom.node_b.id,
                 axiom.confidence, axiom.source, axiom.lens_weight, axiom.flag,
                 metadata_json)
            )
        conn.commit()
        conn.close()

    def load_from_db(self):
        """Load semua axiom dari SQLite."""
        if self._db_path == ":memory:":
            return
        if not os.path.exists(self._db_path):
            return
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT id, node_a_id, relation_id, node_b_id, confidence, source, lens_weight, flag, metadata FROM axioms ORDER BY id")
        for row in cursor:
            aid, na, rel, nb, conf, src, lw, flag, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            axiom = Axiom(
                id=aid,
                node_a=NodeID(id=na),
                relation=NodeID(id=rel),
                node_b=NodeID(id=nb),
                confidence=conf,
                source=src,
                lens_weight=lw,
                flag=flag,
                metadata=metadata,
            )
            self._axioms[aid] = axiom
            if aid > self._id_counter:
                self._id_counter = aid
        conn.close()
