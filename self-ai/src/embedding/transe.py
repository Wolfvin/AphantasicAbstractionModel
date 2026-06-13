# @WHO:   self-ai/src/embedding/transe.py
# @WHAT:  TransE Knowledge Graph Embedding — generalisasi geometris
# @PART:  embedding
# @ENTRY: TransEModel.train(), TransEModel.predict_tail(), TransEModel.predict_relation()
#
# TransE: vec(h) + vec(r) ≈ vec(t)
#
# Model belajar bahwa "kucing + IS_A ≈ mamalia" secara geometris.
# Ini memungkinkan SELF melakukan:
# 1. Link prediction: "paus BREATHES_WITH ?" → paru-paru (generalisasi!)
# 2. Noise detection: "kucing LIVES_IN luar angkasa" → score rendah → suspicious
# 3. Confidence reconciliation: gabungkan rule-based + embedding score
#
# Training: margin ranking loss + negative sampling, CPU-friendly

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import random

from src.translation.translator import NodeID
from src.axiom.store import AxiomStore, Axiom
from src.core.node_store import NodeStore


@dataclass
class Prediction:
    """Hasil prediksi TransE — candidate tail/relation."""
    node_id: int
    description: str
    score: float          # reconstruction score (higher = more likely)
    distance: float       # ||h + r - t|| (lower = more likely)


@dataclass
class TrainingReport:
    """Laporan training TransE."""
    epochs: int
    final_loss: float
    n_triplets: int
    n_entities: int
    n_relations: int
    duration_seconds: float


class TransEModel:
    """
    TransE Knowledge Graph Embedding.

    Prinsip: vec(head) + vec(relation) ≈ vec(tail)

    Ini adalah "generalisasi geometris" — bukan rule, tapi spatial proximity
    di vector space. Kalau embedding-nya bagus:

        vec(paus) + vec(IS_A) ≈ vec(mamalia)

    Padahal "paus IS_A mamalia" mungkin tidak pernah diajarkan eksplisit.
    SELF tahu karena paus dekat secara geometris dengan entitas mamalia lain.

    Training: margin ranking loss dengan negative sampling.
    - Positive: (h, r, t) yang valid dari AxiomStore
    - Negative: (h', r, t) atau (h, r, t') yang dikorupsi
    - Loss: max(0, margin + pos_distance - neg_distance)

    Ringan: d=64, bisa jalan di CPU dalam hitungan detik.
    """

    def __init__(self, dim: int = 64, margin: float = 1.0, learning_rate: float = 0.01,
                 seed: int = 42):
        self.dim = dim
        self.margin = margin
        self.lr = learning_rate
        self.seed = seed

        # Embeddings: node_id → vector, relation_name → vector
        self._entity_embeddings: Dict[int, np.ndarray] = {}
        self._relation_embeddings: Dict[str, np.ndarray] = {}

        # Trained flag
        self._trained = False
        self._training_report: Optional[TrainingReport] = None

        # Reverse lookup: node_id → description (for predictions)
        self._node_descriptions: Dict[int, str] = {}

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def training_report(self) -> Optional[TrainingReport]:
        return self._training_report

    def train(self, axiom_store: AxiomStore, node_store: NodeStore,
              relation_lookup: Dict[int, str],
              translator=None,
              epochs: int = 100, neg_samples: int = 5) -> TrainingReport:
        """
        Train TransE dari axiom yang ada di AxiomStore.

        @FLOW:     TRANSE_TRAIN
        @CALLS:    _init_embeddings(), _generate_negative_sample()
        @MUTATES:  self._entity_embeddings, self._relation_embeddings
        @BEHAVIOR: Training loop: margin ranking loss + negative sampling.
                   Mengambil semua axiom dari store, konversi ke triplet,
                   train dengan SGD. Setelah selesai, model bisa predict.
        """
        import time
        start = time.time()

        # 1. Kumpulkan triplet dari axiom store
        triplets = self._collect_triplets(axiom_store, relation_lookup)
        if len(triplets) < 3:
            return TrainingReport(0, 0.0, 0, 0, 0, 0.0)

        # 2. Init embeddings
        all_entity_ids = set()
        all_relation_names = set()
        for h, r_name, t in triplets:
            all_entity_ids.add(h)
            all_entity_ids.add(t)
            all_relation_names.add(r_name)

        rng = np.random.RandomState(self.seed)
        bound = 6.0 / self.dim  # Xavier-like init

        for eid in all_entity_ids:
            self._entity_embeddings[eid] = rng.uniform(-bound, bound, self.dim).astype(np.float32)
            # Cache description
            if translator is not None:
                desc = translator.translate_to_human(NodeID(id=eid))
                self._node_descriptions[eid] = desc
            else:
                self._node_descriptions[eid] = f"Node#{eid:04d}"

        for rname in all_relation_names:
            self._relation_embeddings[rname] = rng.uniform(-bound, bound, self.dim).astype(np.float32)

        # 3. Training loop
        entity_list = list(all_entity_ids)
        best_loss = float('inf')

        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(triplets)

            for h, r_name, t in triplets:
                # Positive sample
                h_vec = self._entity_embeddings[h]
                r_vec = self._relation_embeddings[r_name]
                t_vec = self._entity_embeddings[t]

                pos_dist = np.linalg.norm(h_vec + r_vec - t_vec)

                # Negative samples
                corrupt_tail = random.random() < 0.7
                for _ in range(neg_samples):
                    neg_t_vec = None
                    neg_h_vec = None

                    if corrupt_tail:
                        # Korupsi tail
                        neg_t = random.choice(entity_list)
                        if neg_t == t:
                            continue
                        neg_t_vec = self._entity_embeddings[neg_t]
                        neg_dist = np.linalg.norm(h_vec + r_vec - neg_t_vec)
                    else:
                        # Korupsi head
                        neg_h = random.choice(entity_list)
                        if neg_h == h:
                            continue
                        neg_h_vec = self._entity_embeddings[neg_h]
                        neg_dist = np.linalg.norm(neg_h_vec + r_vec - t_vec)

                    # Margin ranking loss
                    loss = max(0.0, self.margin + pos_dist - neg_dist)
                    total_loss += loss

                    if loss > 0:
                        # Gradient update: minimize pos_dist, maximize neg_dist
                        pos_diff = h_vec + r_vec - t_vec
                        pos_norm = np.linalg.norm(pos_diff) + 1e-10
                        grad_pos = pos_diff / pos_norm

                        if corrupt_tail:
                            # Korupsi tail case
                            neg_diff = h_vec + r_vec - neg_t_vec
                            neg_norm = np.linalg.norm(neg_diff) + 1e-10
                            grad_neg = neg_diff / neg_norm

                            h_vec = h_vec - self.lr * (grad_pos - grad_neg)
                            r_vec = r_vec - self.lr * (grad_pos - grad_neg)
                            t_vec = t_vec + self.lr * grad_pos
                            neg_t_vec = neg_t_vec - self.lr * grad_neg

                            # Normalize
                            self._entity_embeddings[h] = self._normalize(h_vec)
                            self._entity_embeddings[neg_t] = self._normalize(neg_t_vec)
                        else:
                            # Korupsi head case
                            neg_diff = neg_h_vec + r_vec - t_vec
                            neg_norm = np.linalg.norm(neg_diff) + 1e-10
                            grad_neg = neg_diff / neg_norm

                            h_vec = h_vec - self.lr * grad_pos
                            neg_h_vec = neg_h_vec + self.lr * grad_neg
                            r_vec = r_vec - self.lr * (grad_pos - grad_neg)
                            t_vec = t_vec + self.lr * grad_pos

                            # Normalize
                            self._entity_embeddings[h] = self._normalize(h_vec)
                            self._entity_embeddings[neg_h] = self._normalize(neg_h_vec)

                        self._relation_embeddings[r_name] = self._normalize(r_vec)
                        self._entity_embeddings[t] = self._normalize(t_vec)

            avg_loss = total_loss / max(len(triplets), 1)
            best_loss = min(best_loss, avg_loss)

        duration = time.time() - start
        self._trained = True

        self._training_report = TrainingReport(
            epochs=epochs,
            final_loss=best_loss,
            n_triplets=len(triplets),
            n_entities=len(all_entity_ids),
            n_relations=len(all_relation_names),
            duration_seconds=duration,
        )

        return self._training_report

    def predict_tail(self, head_id: int, relation_name: str, top_k: int = 5) -> List[Prediction]:
        """
        Prediksi tail node yang paling cocok untuk (head, relation, ?).

        @FLOW:     TRANSE_PREDICT_TAIL
        @CALLS:    cosine similarity search
        @MUTATES:  none
        @BEHAVIOR: Mengembalikan candidate tail nodes berdasarkan
                   ||vec(head) + vec(relation) - vec(tail)|| yang terkecil.
                   Ini adalah link prediction — generalisasi geometris.
        """
        if not self._trained:
            return []

        h_vec = self._entity_embeddings.get(head_id)
        r_vec = self._relation_embeddings.get(relation_name)
        if h_vec is None or r_vec is None:
            return []

        target = h_vec + r_vec  # Ideal: target ≈ t_vec

        candidates = []
        for eid, e_vec in self._entity_embeddings.items():
            dist = np.linalg.norm(target - e_vec)
            score = 1.0 / (1.0 + dist)  # Convert distance to score (0-1)
            desc = self._node_descriptions.get(eid, f"Node#{eid:04d}")
            candidates.append(Prediction(
                node_id=eid,
                description=desc,
                score=score,
                distance=dist,
            ))

        # Sort by distance (ascending) — nearest first
        candidates.sort(key=lambda p: p.distance)
        return candidates[:top_k]

    def predict_relation(self, head_id: int, tail_id: int, top_k: int = 5) -> List[Prediction]:
        """
        Prediksi relasi yang paling cocok untuk (head, ?, tail).

        @FLOW:     TRANSE_PREDICT_RELATION
        @CALLS:    distance computation per relation
        @MUTATES:  none
        @BEHAVIOR: Mengembalikan candidate relations berdasarkan
                   ||vec(head) + vec(relation) - vec(tail)|| yang terkecil.
        """
        if not self._trained:
            return []

        h_vec = self._entity_embeddings.get(head_id)
        t_vec = self._entity_embeddings.get(tail_id)
        if h_vec is None or t_vec is None:
            return []

        candidates = []
        for rname, r_vec in self._relation_embeddings.items():
            dist = np.linalg.norm(h_vec + r_vec - t_vec)
            score = 1.0 / (1.0 + dist)
            candidates.append(Prediction(
                node_id=0,
                description=rname,
                score=score,
                distance=dist,
            ))

        candidates.sort(key=lambda p: p.distance)
        return candidates[:top_k]

    def score_triplet(self, head_id: int, relation_name: str, tail_id: int) -> float:
        """
        Hitung reconstruction score untuk triplet (h, r, t).

        @BEHAVIOR: Score tinggi = triplet konsisten dengan pola global.
                   Score rendah = triplet mungkin noise/outlier.

                   score = -||vec(h) + vec(r) - vec(t)||
                   (negatif karena distance kecil = score tinggi)
        """
        if not self._trained:
            return 0.0

        h_vec = self._entity_embeddings.get(head_id)
        r_vec = self._relation_embeddings.get(relation_name)
        t_vec = self._entity_embeddings.get(tail_id)

        if h_vec is None or r_vec is None or t_vec is None:
            return 0.0

        dist = np.linalg.norm(h_vec + r_vec - t_vec)
        return -dist  # Negative distance = score

    def score_all_axioms(self, axiom_store: AxiomStore,
                         relation_lookup: Dict[int, str]) -> Dict[int, float]:
        """
        Hitung reconstruction score untuk semua axiom.

        @BEHAVIOR: Mengembalikan mapping axiom_id → score.
                   Axiom dengan score rendah = kandidat noise.
        """
        if not self._trained:
            return {}

        scores = {}
        for axiom in axiom_store.get_all():
            r_name = relation_lookup.get(axiom.relation.id)
            if r_name is None:
                continue
            score = self.score_triplet(axiom.node_a.id, r_name, axiom.node_b.id)
            scores[axiom.id] = score

        return scores

    def _collect_triplets(self, axiom_store: AxiomStore,
                          relation_lookup: Dict[int, str]) -> List[Tuple[int, str, int]]:
        """Kumpulkan semua triplet (h, r_name, t) dari axiom store."""
        triplets = []
        for axiom in axiom_store.get_all():
            r_name = relation_lookup.get(axiom.relation.id)
            if r_name is None:
                continue
            triplets.append((axiom.node_a.id, r_name, axiom.node_b.id))
        return triplets

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec
