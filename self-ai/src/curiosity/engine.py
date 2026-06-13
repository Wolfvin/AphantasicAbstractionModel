# @WHO:   self-ai/src/curiosity/engine.py
# @WHAT:  Layer 8 — Curiosity Engine: SELF SELALU bisa bertanya
# @PART:  curiosity
# @ENTRY: CuriosityEngine.compute_score(), CuriosityEngine.find_curiosity_target()
#
# v7: ADAPTIVE CURIOSITY WEIGHTS
#
# Curiosity weights tidak lagi hardcoded!
# SELF belajar dari pengalaman weight mana yang paling efektif:
# - Jika pertanyaan tentang inconsistency menghasilkan pengetahuan baru → weight naik
# - Jika pertanyaan tentang gaps tidak produktif → weight turun
#
# Curiosity bukan reward dari luar.
# Curiosity adalah ketidaknyamanan internal — rasa "belum paham".
# SELF SELALU punya sesuatu yang belum dipahami.

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from src.axiom.store import AxiomStore
from src.derivation.engine import DerivationEngine
from src.difference.detector import DifferenceDetector
from src.core.node_store import NodeStore
from src.translation.translator import NodeID
from config.thresholds import (
    CURIOSITY_WEIGHT_INCONSISTENCY,
    CURIOSITY_WEIGHT_UNEXPLAINED,
    CURIOSITY_WEIGHT_FAILED_DERIVATION,
    adaptive,
)


@dataclass
class CuriosityScore:
    """Skor curiosity untuk area pengetahuan tertentu."""
    node_id: int
    score: float
    inconsistency_contribution: float
    gap_contribution: float
    derivation_gap_contribution: float
    reason: str
    suggested_question_topic: str = ""  # Topik spesifik untuk pertanyaan


class CuriosityEngine:
    """
    Layer 8: Curiosity Engine — "Apa yang belum saya mengerti?"

    SELF SELALU punya curiosity.
    Karena SELF SELALU ada yang belum dipahami.

    v7: Curiosity weights sekarang ADAPTIVE.
    SELF belajar dari pengalaman weight mana yang paling efektif:
    - Jika pertanyaan tentang inconsistency menghasilkan pengetahuan baru
      → weight naik
    - Jika pertanyaan tentang gaps tidak produktif → weight turun

    Curiosity score = (
        n_inconsistencies     × w_inconsistency +
        knowledge_gaps        × w_unexplained +
        derivation_gaps       × w_derivation
    )

    Weights di-adjust berdasarkan feedback — tidak hardcoded!
    """

    def __init__(self, node_store: NodeStore, axiom_store: AxiomStore,
                 diff_detector: DifferenceDetector, derivation_engine: DerivationEngine):
        self.node_store = node_store
        self.axiom_store = axiom_store
        self.diff_detector = diff_detector
        self.derivation_engine = derivation_engine

    def _get_weights(self) -> Dict[str, float]:
        """v7: Mendapatkan adaptive curiosity weights."""
        return adaptive.get_curiosity_weights()

    def compute_score(self, node_id: int) -> CuriosityScore:
        """
        @FLOW:     CURIOSITY_COMPUTE
        @CALLS:    AxiomStore, DerivationEngine
        @MUTATES:  none
        @BEHAVIOR: Menghitung curiosity score untuk node tertentu.
                   Score tinggi = area yang SELF belum pahami.
        """
        # v7: Get adaptive weights
        weights = self._get_weights()
        w_incon = weights["inconsistency"]
        w_unexp = weights["unexplained"]
        w_deriv = weights["derivation"]

        # 1. Inconsistency: uncertain axioms yang melibatkan node ini
        n_inconsistencies = 0
        for axiom in self.axiom_store.get_uncertain():
            if axiom.node_a.id == node_id or axiom.node_b.id == node_id:
                n_inconsistencies += 1

        # 2. Knowledge gaps: berapa banyak relasi yang sudah diketahui vs yang bisa diketahui?
        known_relations = set()
        total_axioms_involving = 0
        for axiom in self.axiom_store.get_all():
            if axiom.node_a.id == node_id or axiom.node_b.id == node_id:
                total_axioms_involving += 1
                rel_name = self.derivation_engine._get_relation_name(axiom.relation)
                if rel_name:
                    known_relations.add(rel_name)

        # Semakin sedikit relasi yang diketahui → semakin besar gap
        # Node yang cuma muncul di 1-2 axiom = gap besar
        gap_score = 0.0
        suggested_topic = ""

        if total_axioms_involving <= 2:
            gap_score = 1.0  # Node hampir tidak punya relasi → besar gap
            suggested_topic = "hubungan dan properti"
        elif total_axioms_involving <= 4:
            gap_score = 0.5
            suggested_topic = "relasi tambahan"
        else:
            gap_score = 0.1  # Node cukup dipahami

        # 3. Derivation gaps: apakah node ini punya IS_A chain tapi belum punya property inheritance?
        derivation_gap = 0.0
        # Cek apakah node ini adalah subjek dari IS_A axiom
        is_a_targets = set()
        for axiom in self.axiom_store.get_all():
            rel_name = self.derivation_engine._get_relation_name(axiom.relation)
            if rel_name == "IS_A" and axiom.node_a.id == node_id:
                is_a_targets.add(axiom.node_b.id)

        # Jika SELF tahu X IS_A Y, tapi belum tahu apa yang Y HAS/BREATHES_WITH/dll
        # → derivation gap
        for target_id in is_a_targets:
            target_has_properties = False
            for axiom in self.axiom_store.get_all():
                if axiom.node_a.id == target_id:
                    rel_name = self.derivation_engine._get_relation_name(axiom.relation)
                    if rel_name and rel_name not in ("IS_A", "INSTANCE_OF"):
                        target_has_properties = True
                        break
            if not target_has_properties:
                derivation_gap += 0.5  # Tahu IS_A tapi tidak tahu properties parent

        # v7: Weighted score menggunakan adaptive weights
        inconsistency_c = n_inconsistencies * w_incon
        gap_c = gap_score * w_unexp
        derivation_c = derivation_gap * w_deriv

        total_score = inconsistency_c + gap_c + derivation_c

        # Tentukan alasan utama
        max_contrib = max(inconsistency_c, gap_c, derivation_c)
        if max_contrib == inconsistency_c and inconsistency_c > 0:
            reason = "inkonsistensi dalam memory"
        elif max_contrib == derivation_c and derivation_c > 0:
            reason = "hubungan yang belum bisa disimpulkan"
        elif gap_c > 0:
            reason = "pengetahuan yang belum lengkap"
        else:
            reason = "ingin memahami lebih dalam"

        return CuriosityScore(
            node_id=node_id,
            score=total_score,
            inconsistency_contribution=inconsistency_c,
            gap_contribution=gap_c,
            derivation_gap_contribution=derivation_c,
            reason=reason,
            suggested_question_topic=suggested_topic,
        )

    def find_curiosity_target(self) -> Optional[CuriosityScore]:
        """
        @FLOW:     CURIOSITY_FIND_TARGET
        @CALLS:    compute_score() untuk setiap node
        @MUTATES:  none
        @BEHAVIOR: Menemukan area dengan curiosity score tertinggi.
                   SELALU mengembalikan sesuatu — SELF selalu penasaran.
        """
        all_nodes = self.node_store.all_node_ids()
        if not all_nodes:
            return None

        scores = []
        for node_id in all_nodes:
            score = self.compute_score(node_id.id)
            if score.score > 0:
                scores.append(score)

        if not scores:
            # Tidak ada curiosity > 0 → berarti SELF merasa sudah paham semuanya
            # Tapi itu MUSTAHIL — selalu ada yang belum dipahami
            # Buat curiosity dari node yang paling sedikit dijelaskan
            return self._find_least_understood_node()

        # Return area dengan score tertinggi
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores[0]

    def _find_least_understood_node(self) -> Optional[CuriosityScore]:
        """
        Fallback: menemukan node yang paling sedikit dipahami.
        Node yang paling sedikit muncul di axioms = paling tidak dipahami.
        """
        axiom_count_per_node: Dict[int, int] = {}
        for axiom in self.axiom_store.get_all():
            axiom_count_per_node[axiom.node_a.id] = axiom_count_per_node.get(axiom.node_a.id, 0) + 1
            axiom_count_per_node[axiom.node_b.id] = axiom_count_per_node.get(axiom.node_b.id, 0) + 1

        if not axiom_count_per_node:
            # Ambil node pertama saja
            all_nodes = self.node_store.all_node_ids()
            if all_nodes:
                return CuriosityScore(
                    node_id=all_nodes[0].id,
                    score=0.1,
                    inconsistency_contribution=0,
                    gap_contribution=0.1,
                    derivation_gap_contribution=0,
                    reason="ingin memahami lebih dalam",
                    suggested_question_topic="apa itu",
                )
            return None

        # Node dengan axiom count paling rendah
        min_count = min(axiom_count_per_node.values())
        least_understood = [nid for nid, cnt in axiom_count_per_node.items() if cnt == min_count]

        if least_understood:
            return CuriosityScore(
                node_id=least_understood[0],
                score=0.1,
                inconsistency_contribution=0,
                gap_contribution=0.1,
                derivation_gap_contribution=0,
                reason="pengetahuan yang belum lengkap",
                suggested_question_topic="hubungan dan properti",
            )

        return None

    def find_derivation_curiosity(self) -> List[Dict]:
        """
        Menemukan area dimana SELF bisa belajar satu fakta
        dan mendapatkan banyak turunan dari deductive reasoning.

        Returns list of dicts:
        {
            "bridge_desc": str,
            "known_from": [...],
            "missing_rels": [...],
            "potential_derivations": int,  # berapa banyak yang bisa di-derive
        }
        """
        gaps = self.derivation_engine.find_derivable_gaps()
        curiosity_gaps = []

        for gap in gaps:
            # Hitung berapa banyak turunan yang bisa di-derive
            # = jumlah node yang IS_A ke bridge ini
            potential = len(gap["known_from"]) * len(gap["missing_rels"])

            curiosity_gaps.append({
                "bridge_desc": gap["bridge_desc"],
                "known_from": gap["known_from"],
                "missing_rels": gap["missing_rels"],
                "existing_rels": gap["existing_rels"],
                "potential_derivations": potential,
            })

        # Sort by potential derivations (most impactful first)
        curiosity_gaps.sort(key=lambda g: g["potential_derivations"], reverse=True)
        return curiosity_gaps

    def get_all_scores(self) -> List[CuriosityScore]:
        """Mengembalikan curiosity score untuk semua node."""
        all_nodes = self.node_store.all_node_ids()
        return [self.compute_score(n.id) for n in all_nodes]
