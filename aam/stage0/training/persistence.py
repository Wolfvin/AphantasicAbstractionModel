"""
Training Persistence — RSVS-Rich Persistent Storage for AAM Training.

This is NOT just in-memory. The full training state is persisted to disk:
1. Knowledge graph (nodes, compositions, edges)
2. Inquiry memory (prevents re-asking questions)
3. Question-answer history
4. Pattern mining results
5. Training records/audit trail
6. Metadata (version, timestamps, statistics)

The RSVS-rich format stores everything as structured JSON, making it:
- Human-readable (you can inspect what the system learned)
- Machine-processable (can be loaded by any tool)
- Versioned (schema_version tracks format changes)
- Auditable (every composition has provenance and timestamps)
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .types import Composition, CompositionMember, KnowledgeGap, TrainingRecord, PatternObservation

# Current schema version
SCHEMA_VERSION = "v12-training-1.0"


class TrainingPersistence:
    """
    Persistent storage for the AAM training system.

    Saves the full training state to RSVS-rich format:
    - graph.json: The knowledge graph (nodes, compositions, edges)
    - inquiry_memory.json: Gap tracking and question history
    - patterns.json: Discovered patterns
    - records.json: Training audit trail
    - metadata.json: Version, timestamps, statistics
    """

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

    def save(self, trainer) -> None:
        """Save the full training state to disk."""
        # 1. Save knowledge graph
        self._save_graph(trainer)

        # 2. Save inquiry memory
        self._save_inquiry_memory(trainer)

        # 3. Save patterns
        self._save_patterns(trainer)

        # 4. Save training records
        self._save_records(trainer)

        # 5. Save metadata
        self._save_metadata(trainer)

        # 6. Save a human-readable summary
        self._save_summary(trainer)

    def load(self, trainer) -> bool:
        """Load existing training state from disk. Returns True if state was loaded."""
        graph_path = os.path.join(self.persist_dir, "graph.json")
        if not os.path.exists(graph_path):
            return False

        try:
            self._load_graph(trainer)
            self._load_inquiry_memory(trainer)
            self._load_patterns(trainer)
            self._load_records(trainer)
            return True
        except Exception as e:
            print(f"[TrainingPersistence] Warning: could not load state: {e}")
            return False

    # ────────────────────────────────────────────────────────────────
    # Graph Persistence
    # ────────────────────────────────────────────────────────────────

    def _save_graph(self, trainer) -> None:
        """Save the knowledge graph in RSVS-rich format."""
        graph_data = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "batch_number": trainer._batch_number,
            "next_node_id": trainer._next_node_id,
            "next_comp_id": trainer._next_comp_id,
            "next_gap_id": trainer._next_gap_id,
            "nodes": {
                label: node_id
                for label, node_id in trainer.nodes.items()
            },
            "compositions": {
                comp_id: comp.to_dict()
                for comp_id, comp in trainer.compositions.items()
            },
            "edges": [
                {"comp_id": comp_id, "target_node_id": target, "role": role}
                for comp_id, target, role in trainer.edges
            ],
            "statistics": {
                "total_nodes": len(trainer.nodes),
                "total_compositions": len(trainer.compositions),
                "total_edges": len(trainer.edges),
                "by_lifecycle": {
                    state: len([c for c in trainer.compositions.values() if c.lifecycle == state])
                    for state in ["New", "Candidate", "Stable", "Deprecated", "Quarantine"]
                },
                "by_epistemic": {
                    state: len([c for c in trainer.compositions.values() if c.epistemic == state])
                    for state in ["Observed", "Inferred", "Hypothesis", "Grounded", "Contradicted"]
                },
                "average_confidence": (
                    sum(c.confidence for c in trainer.compositions.values()) / len(trainer.compositions)
                    if trainer.compositions else 0.0
                ),
            },
        }

        path = os.path.join(self.persist_dir, "graph.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False, default=str)

    def _load_graph(self, trainer) -> None:
        """Load the knowledge graph from disk."""
        path = os.path.join(self.persist_dir, "graph.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trainer._batch_number = data.get("batch_number", 0)
        trainer._next_node_id = data.get("next_node_id", 1)
        trainer._next_comp_id = data.get("next_comp_id", 1)
        trainer._next_gap_id = data.get("next_gap_id", 1)

        # Reconstruct nodes
        trainer.nodes = {}
        trainer.node_labels = {}
        for label, node_id in data.get("nodes", {}).items():
            node_id = int(node_id)
            trainer.nodes[label] = node_id
            trainer.node_labels[node_id] = label

        # Reconstruct compositions
        trainer.compositions = {}
        for comp_id, comp_data in data.get("compositions", {}).items():
            trainer.compositions[comp_id] = Composition.from_dict(comp_data)

        # Reconstruct edges
        trainer.edges = []
        for edge_data in data.get("edges", []):
            trainer.edges.append((
                edge_data["comp_id"],
                int(edge_data["target_node_id"]),
                edge_data["role"],
            ))

    # ────────────────────────────────────────────────────────────────
    # Inquiry Memory Persistence
    # ────────────────────────────────────────────────────────────────

    def _save_inquiry_memory(self, trainer) -> None:
        """Save inquiry memory (gap tracking + question history)."""
        data = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "inquiry_memory": trainer.inquiry_memory,
            "question_history": trainer.question_history,
            "gaps": {
                gap_id: gap.to_dict()
                for gap_id, gap in trainer.gaps.items()
            },
        }

        path = os.path.join(self.persist_dir, "inquiry_memory.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _load_inquiry_memory(self, trainer) -> None:
        """Load inquiry memory from disk."""
        path = os.path.join(self.persist_dir, "inquiry_memory.json")
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trainer.inquiry_memory = data.get("inquiry_memory", {})
        trainer.question_history = data.get("question_history", {})

        # Reconstruct gaps
        trainer.gaps = {}
        for gap_id, gap_data in data.get("gaps", {}).items():
            trainer.gaps[gap_id] = KnowledgeGap.from_dict(gap_data)

    # ────────────────────────────────────────────────────────────────
    # Pattern Persistence
    # ────────────────────────────────────────────────────────────────

    def _save_patterns(self, trainer) -> None:
        """Save discovered patterns."""
        data = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "patterns": {
                key: pattern.to_dict()
                for key, pattern in trainer.patterns.items()
            },
        }

        path = os.path.join(self.persist_dir, "patterns.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _load_patterns(self, trainer) -> None:
        """Load patterns from disk."""
        path = os.path.join(self.persist_dir, "patterns.json")
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trainer.patterns = {}
        for key, pattern_data in data.get("patterns", {}).items():
            trainer.patterns[key] = PatternObservation(**pattern_data)

    # ────────────────────────────────────────────────────────────────
    # Training Records Persistence
    # ────────────────────────────────────────────────────────────────

    def _save_records(self, trainer) -> None:
        """Save training audit trail."""
        data = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "records": [r.to_dict() for r in trainer.records],
        }

        path = os.path.join(self.persist_dir, "records.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _load_records(self, trainer) -> None:
        """Load training records from disk."""
        path = os.path.join(self.persist_dir, "records.json")
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trainer.records = [TrainingRecord(**r) for r in data.get("records", [])]

    # ────────────────────────────────────────────────────────────────
    # Metadata
    # ────────────────────────────────────────────────────────────────

    def _save_metadata(self, trainer) -> None:
        """Save version and metadata."""
        data = {
            "schema_version": SCHEMA_VERSION,
            "aam_training_version": "1.0.0",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "total_batches": trainer._batch_number,
            "total_compositions": len(trainer.compositions),
            "total_nodes": len(trainer.nodes),
            "total_patterns": len(trainer.patterns),
            "total_gaps_addressed": len([g for g in trainer.gaps.values() if g.addressed]),
            "total_corrections": len([h for h in trainer.question_history.values() if h is not None]),
        }

        path = os.path.join(self.persist_dir, "metadata.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    # ────────────────────────────────────────────────────────────────
    # Human-Readable Summary
    # ────────────────────────────────────────────────────────────────

    def _save_summary(self, trainer) -> None:
        """Save a human-readable text summary of the training state."""
        lines = []
        lines.append("=" * 60)
        lines.append("AAM Training Summary")
        lines.append("=" * 60)
        lines.append(f"Saved: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"Batches: {trainer._batch_number}")
        lines.append(f"Nodes: {len(trainer.nodes)}")
        lines.append(f"Compositions: {len(trainer.compositions)}")
        lines.append(f"Edges: {len(trainer.edges)}")
        lines.append(f"Patterns: {len(trainer.patterns)}")
        lines.append(f"Gaps addressed: {len([g for g in trainer.gaps.values() if g.addressed])} / {len(trainer.gaps)}")
        lines.append("")

        # Stable/Grounded compositions
        stable = [c for c in trainer.compositions.values() if c.lifecycle == "Stable"]
        lines.append(f"--- Stable Compositions ({len(stable)}) ---")
        for comp in stable[:20]:
            members_str = ", ".join(f"{m.role}={m.label}" for m in comp.members)
            lines.append(f"  [{comp.epistemic}] {comp.id}: {members_str}")
        if len(stable) > 20:
            lines.append(f"  ... and {len(stable) - 20} more")
        lines.append("")

        # Grounded patterns
        grounded = [p for p in trainer.patterns.values() if p.epistemic == "Grounded"]
        lines.append(f"--- Grounded Patterns ({len(grounded)}) ---")
        for p in grounded[:20]:
            lines.append(f"  {p.predicate} + {p.role} = {p.filler} (×{p.observation_count})")
        lines.append("")

        # Unresolved gaps
        unresolved = [g for g in trainer.gaps.values() if not g.addressed]
        lines.append(f"--- Unresolved Gaps ({len(unresolved)}) ---")
        for gap in unresolved[:10]:
            lines.append(f"  [{gap.gap_type}] {gap.description}")
        lines.append("=" * 60)

        path = os.path.join(self.persist_dir, "summary.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
