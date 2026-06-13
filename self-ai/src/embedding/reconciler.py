# @WHO:   self-ai/src/embedding/reconciler.py
# @WHAT:  Confidence Reconciler — hybrid confidence dengan learnable alpha
# @PART:  embedding
# @ENTRY: ConfidenceReconciler.reconcile(), ConfidenceReconciler.get_flag()
#
# v6: Learnable alpha — SELF belajar rasio optimal antara
# rule-based dan embedding confidence dari feedback.
#
# Hybrid confidence: final_conf = α × rule_conf + (1-α) × embedding_score
#
# Alpha belajar dari:
# - Ketika derived axiom dikonfirmasi oleh teaching → positive feedback
# - Ketika derived axiom bertentangan dengan teaching → negative feedback
# - Jika rule-based lebih akurat → alpha naik
# - Jika embedding lebih akurat → alpha turun

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.axiom.store import AxiomStore, Axiom
from src.translation.translator import NodeID


class ConfidenceFlag(str, Enum):
    """Flag confidence — hasil reconciler."""
    CONFIRMED = "confirmed"
    SPECULATIVE = "speculative"
    SUSPICIOUS = "suspicious"
    CONFLICT = "conflict"
    UNTRAINED = "untrained"


@dataclass
class ReconciledAxiom:
    """Axiom yang sudah di-reconcile — punya hybrid confidence + flag."""
    axiom_id: int
    rule_confidence: float
    embedding_score: float
    hybrid_confidence: float
    flag: ConfidenceFlag
    rule_source: str


class ConfidenceReconciler:
    """
    Confidence Reconciler — "Hakim Kedua" SELF.

    v6: Alpha sekarang LEARNABLE.
    - _alpha_history: track feedback untuk learning
    - learn_from_feedback(): adjust alpha berdasarkan hasil
    - Jika rule-based lebih akurat → alpha naik
    - Jika embedding lebih akurat → alpha turun
    """

    def __init__(self, alpha: float = 0.6, embedding_threshold: float = -1.5,
                 rule_threshold: float = 0.5, conflict_gap: float = 0.4):
        self.alpha = alpha
        self.embedding_threshold = embedding_threshold
        self.rule_threshold = rule_threshold
        self.conflict_gap = conflict_gap

        # Results cache
        self._reconciled: Dict[int, ReconciledAxiom] = {}

        # v6: Alpha learning state
        self._alpha_history: List[Dict] = []  # feedback records
        self._alpha_lr = 0.05  # learning rate untuk alpha adjustment
        self._alpha_min = 0.2  # minimum alpha (embedding max 80%)
        self._alpha_max = 0.9  # maximum alpha (rule max 90%)

        # Adaptive thresholds — belajar dari data
        self._rule_threshold_adaptive = rule_threshold
        self._emb_threshold_adaptive = 0.5  # normalized threshold

    def reconcile(self, axiom_store: AxiomStore,
                  embedding_scores: Dict[int, float],
                  trained: bool = True) -> Dict[int, ReconciledAxiom]:
        """
        Menggabungkan rule-based confidence dengan embedding score.
        """
        self._reconciled = {}

        # Collect all confidence values for adaptive thresholding
        rule_confs = []
        emb_norms = []

        # First pass: compute all normalized values
        temp_results = {}
        for axiom in axiom_store.get_all():
            rule_conf = axiom.confidence
            emb_score = embedding_scores.get(axiom.id, 0.0)

            if not trained:
                flag = ConfidenceFlag.UNTRAINED
                hybrid_conf = rule_conf
            else:
                emb_normalized = self._normalize_embedding_score(emb_score)
                hybrid_conf = self.alpha * rule_conf + (1 - self.alpha) * emb_normalized
                flag = self._classify_flag(rule_conf, emb_normalized, axiom.source)
                rule_confs.append(rule_conf)
                emb_norms.append(emb_normalized)

            temp_results[axiom.id] = (rule_conf, emb_score, hybrid_conf, flag)

        # v6: Adaptive thresholds — update berdasarkan distribusi data
        if rule_confs and emb_norms:
            self._update_adaptive_thresholds(rule_confs, emb_norms)

        # Second pass: apply adaptive thresholds
        for axiom in axiom_store.get_all():
            if axiom.id in temp_results:
                rule_conf, emb_score, hybrid_conf, flag = temp_results[axiom.id]

                self._reconciled[axiom.id] = ReconciledAxiom(
                    axiom_id=axiom.id,
                    rule_confidence=rule_conf,
                    embedding_score=emb_score,
                    hybrid_confidence=hybrid_conf,
                    flag=flag,
                    rule_source=axiom.source,
                )

        return self._reconciled

    def learn_from_feedback(self, axiom_id: int, confirmed: bool):
        """
        v6: Belajar dari feedback — adjust alpha.

        Dipanggil ketika:
        - Derived axiom dikonfirmasi oleh teaching (confirmed=True)
        - Derived axiom bertentangan dengan teaching (confirmed=False)

        Logic:
        - Jika rule confidence benar (tinggi + confirmed, rendah + !confirmed) → alpha naik
        - Jika embedding score benar → alpha turun
        """
        r = self._reconciled.get(axiom_id)
        if r is None:
            return

        # Determine which source was "right"
        rule_correct = (r.rule_confidence >= self._rule_threshold_adaptive) == confirmed
        emb_correct = (r.embedding_score >= 0.0) == confirmed  # emb_score is -distance, positive = good

        # Adjust alpha
        if rule_correct and not emb_correct:
            # Rule was right, embedding was wrong → increase alpha
            self.alpha = min(self._alpha_max, self.alpha + self._alpha_lr)
        elif emb_correct and not rule_correct:
            # Embedding was right, rule was wrong → decrease alpha
            self.alpha = max(self._alpha_min, self.alpha - self._alpha_lr)

        # Record feedback
        self._alpha_history.append({
            "axiom_id": axiom_id,
            "confirmed": confirmed,
            "rule_correct": rule_correct,
            "emb_correct": emb_correct,
            "alpha_after": self.alpha,
        })

        # Keep last 100 feedback records
        if len(self._alpha_history) > 100:
            self._alpha_history = self._alpha_history[-100:]

    def _update_adaptive_thresholds(self, rule_confs: List[float],
                                     emb_norms: List[float]):
        """
        v6: Update thresholds berdasarkan distribusi data.

        Menggunakan median sebagai threshold:
        - Jika median rule_confidence tinggi → threshold bisa lebih ketat
        - Jika median emb_normalized rendah → threshold lebih longgar
        """
        if not rule_confs or not emb_norms:
            return

        rule_median = np.median(rule_confs)
        emb_median = np.median(emb_norms)

        # Adaptive rule threshold: 70% of median, min 0.3
        self._rule_threshold_adaptive = max(0.3, rule_median * 0.7)

        # Adaptive embedding threshold: 50% of median, min 0.2
        self._emb_threshold_adaptive = max(0.2, emb_median * 0.5)

    def get_flag(self, axiom_id: int) -> ConfidenceFlag:
        """Mendapatkan flag untuk axiom tertentu."""
        if axiom_id in self._reconciled:
            return self._reconciled[axiom_id].flag
        return ConfidenceFlag.UNTRAINED

    def get_reconciled(self, axiom_id: int) -> Optional[ReconciledAxiom]:
        """Mendapatkan ReconciledAxiom untuk axiom tertentu."""
        return self._reconciled.get(axiom_id)

    def get_by_flag(self, flag: ConfidenceFlag) -> List[ReconciledAxiom]:
        """Mendapatkan semua axiom dengan flag tertentu."""
        return [r for r in self._reconciled.values() if r.flag == flag]

    def get_stats(self) -> Dict[str, int]:
        """Statistik reconciler — jumlah per flag."""
        stats = {f.value: 0 for f in ConfidenceFlag}
        for r in self._reconciled.values():
            stats[r.flag.value] += 1
        return stats

    def get_alpha_status(self) -> Dict:
        """v6: Status alpha learning."""
        return {
            "current_alpha": round(self.alpha, 3),
            "alpha_min": self._alpha_min,
            "alpha_max": self._alpha_max,
            "feedback_count": len(self._alpha_history),
            "rule_threshold": round(self._rule_threshold_adaptive, 3),
            "emb_threshold": round(self._emb_threshold_adaptive, 3),
        }

    def _classify_flag(self, rule_conf: float, emb_normalized: float,
                       source: str) -> ConfidenceFlag:
        """Klasifikasi flag — menggunakan adaptive thresholds."""
        # v6: Gunakan adaptive thresholds
        rule_high = rule_conf >= self._rule_threshold_adaptive
        emb_high = emb_normalized >= self._emb_threshold_adaptive

        if rule_high and emb_high:
            return ConfidenceFlag.CONFIRMED
        elif not rule_high and emb_high:
            return ConfidenceFlag.SPECULATIVE
        elif rule_high and not emb_high:
            return ConfidenceFlag.SUSPICIOUS
        else:
            return ConfidenceFlag.CONFLICT

    @staticmethod
    def _normalize_embedding_score(score: float) -> float:
        """Normalisasi embedding score ke range [0, 1]."""
        distance = -score
        return 1.0 / (1.0 + max(distance, 0.0))
